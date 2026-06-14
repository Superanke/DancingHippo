# keyframe_gen.py -- Keyframe image generation + AI evaluation via Z-Image Turbo

from __future__ import annotations

import copy
import json
import logging
import os
import random
import shutil
import base64
from io import BytesIO

import ollama_client

from config import (
    COMFYUI_OUTPUT_DIR, OLLAMA_MODEL_FAST,
    KF_PROMPT_NODE_ID, KF_SEED_NODE_ID, KF_LATENT_NODE_ID,
    KF_CANDIDATES, KF_WIDTH, KF_HEIGHT, SKIP_KF_EVAL,
    OLLAMA_VISION_MAX_EDGE,
)

log = logging.getLogger(__name__)


def _crop_to_video_ar(image_path: str, target_w: int = 1024, target_h: int = 432):
    """Crop a square image to the video's widescreen aspect ratio (center crop)."""
    from PIL import Image as PILImage
    img = PILImage.open(image_path)
    w, h = img.size

    # Calculate crop box for target aspect ratio
    target_ar = target_w / target_h  # e.g. 2.37:1
    current_ar = w / h

    if current_ar < target_ar:
        # Image is too tall, crop top and bottom
        new_h = int(w / target_ar)
        top = (h - new_h) // 2
        box = (0, top, w, top + new_h)
    else:
        # Image is too wide, crop sides
        new_w = int(h * target_ar)
        left = (w - new_w) // 2
        box = (left, 0, left + new_w, h)

    cropped = img.crop(box)
    cropped.save(image_path)


def load_keyframe_template(path: str = None) -> dict:
    if path is None:
        path = os.path.join(os.path.dirname(__file__), "keyframe_template.json")
    with open(path) as f:
        return json.load(f)


def _detect_keyframe_nodes(wf: dict) -> dict:
    """Auto-detect keyframe workflow node IDs from the template."""
    detected = {
        "prompt": KF_PROMPT_NODE_ID,
        "seed": KF_SEED_NODE_ID,
        "latent": KF_LATENT_NODE_ID,
        "sampler": None,
        "vae": None,
    }

    all_present = all(k in wf for k in [KF_PROMPT_NODE_ID, KF_SEED_NODE_ID, KF_LATENT_NODE_ID])
    if not all_present:
        log.info("Default keyframe node IDs not found — auto-detecting...")

        # Find prompt node (CLIPTextEncode)
        clip_nodes = [nid for nid, n in wf.items()
                      if isinstance(n, dict) and n.get("class_type") == "CLIPTextEncode"]
        if clip_nodes:
            detected["prompt"] = clip_nodes[0]

        # Find seed node (KSampler or KSamplerAdvanced)
        for cls in ["KSampler", "KSamplerAdvanced", "RandomNoise"]:
            seed_nodes = [nid for nid, n in wf.items()
                          if isinstance(n, dict) and n.get("class_type") == cls]
            if seed_nodes:
                detected["seed"] = seed_nodes[0]
                break

        # Find latent node (EmptySD3LatentImage, EmptyLatentImage, etc.)
        for cls in ["EmptySD3LatentImage", "EmptyLatentImage", "EmptyImage"]:
            latent_nodes = [nid for nid, n in wf.items()
                            if isinstance(n, dict) and n.get("class_type") == cls]
            if latent_nodes:
                detected["latent"] = latent_nodes[0]
                break

    # Find sampler node used for reference-image conditioning.
    for cls in ["KSampler", "KSamplerAdvanced", "RandomNoise"]:
        sampler_nodes = [nid for nid, n in wf.items()
                         if isinstance(n, dict) and n.get("class_type") == cls]
        if sampler_nodes:
            detected["sampler"] = sampler_nodes[0]
            break

    vae_nodes = [nid for nid, n in wf.items()
                 if isinstance(n, dict) and n.get("class_type") == "VAELoader"]
    if vae_nodes:
        detected["vae"] = vae_nodes[0]

    log.info("Auto-detected keyframe node IDs: %s", detected)
    return detected


def _next_numeric_node_id(wf: dict) -> str:
    """Return a ComfyUI node id that is not already present."""
    numeric_ids = []
    for node_id in wf:
        try:
            numeric_ids.append(int(str(node_id).split(":")[-1]))
        except ValueError:
            continue
    next_id = (max(numeric_ids) + 1) if numeric_ids else 1
    while str(next_id) in wf:
        next_id += 1
    return str(next_id)


def _existing_load_image_nodes(wf: dict) -> list[str]:
    """Return LoadImage nodes already present in a workflow."""
    return [
        nid for nid, node in wf.items()
        if isinstance(node, dict) and node.get("class_type") == "LoadImage"
    ]


def _inject_reference_image_latent(wf: dict, nodes: dict, image_filename: str) -> None:
    """Inject a reference image as the actual starting latent for keyframe generation."""
    load_nodes = _existing_load_image_nodes(wf)
    for node_id in load_nodes:
        wf[node_id]["inputs"]["image"] = image_filename

    sampler_id = nodes.get("sampler")
    vae_id = nodes.get("vae")
    if not sampler_id or sampler_id not in wf:
        raise ValueError("Keyframe workflow cannot use reference images: no KSampler node was found.")
    if not vae_id or vae_id not in wf:
        raise ValueError("Keyframe workflow cannot use reference images: no VAELoader node was found.")

    if load_nodes:
        load_id = load_nodes[0]
    else:
        load_id = _next_numeric_node_id(wf)
        wf[load_id] = {
            "inputs": {"image": image_filename},
            "class_type": "LoadImage",
            "_meta": {"title": "DancingHippo Reference Image"},
        }

    vae_encode_id = _next_numeric_node_id(wf)
    wf[vae_encode_id] = {
        "inputs": {
            "pixels": [load_id, 0],
            "vae": [vae_id, 0],
        },
        "class_type": "VAEEncode",
        "_meta": {"title": "DancingHippo Reference VAE Encode"},
    }
    wf[sampler_id]["inputs"]["latent_image"] = [vae_encode_id, 0]
    if "denoise" in wf[sampler_id].get("inputs", {}):
        wf[sampler_id]["inputs"]["denoise"] = 0.72


def build_keyframe_workflow(template: dict, prompt_text: str, seed: int,
                            width: int, height: int,
                            reference_image_filename: str | None) -> dict:
    """Build a keyframe image workflow with the given prompt and seed.
    Auto-detects node IDs if the configured defaults don't match the template."""
    wf = copy.deepcopy(template)
    nodes = _detect_keyframe_nodes(wf)
    wf[nodes["prompt"]]["inputs"]["text"] = prompt_text
    wf[nodes["seed"]]["inputs"]["seed"] = seed
    wf[nodes["latent"]]["inputs"]["width"] = width
    wf[nodes["latent"]]["inputs"]["height"] = height
    if reference_image_filename:
        _inject_reference_image_latent(wf, nodes, reference_image_filename)
    return wf


def get_image_output_path(history: dict, output_dir: str = COMFYUI_OUTPUT_DIR) -> str:
    """Extract image path from ComfyUI history."""
    outputs = history.get("outputs", {})
    for node_id, node_output in outputs.items():
        if "images" in node_output:
            for item in node_output["images"]:
                if not isinstance(item, dict):
                    continue
                filename = item.get("filename", "")
                if filename.endswith((".png", ".jpg", ".jpeg", ".webp")):
                    subfolder = item.get("subfolder", "")
                    return os.path.join(output_dir, subfolder, filename)
    raise ValueError("No image output found in history")


def image_to_base64(image_path: str) -> str:
    """Read an image file and return PNG base64 for Ollama vision."""
    from PIL import Image as PILImage

    try:
        with PILImage.open(image_path) as image:
            if image.mode not in {"RGB", "RGBA"}:
                image = image.convert("RGB")
            max_edge = max(image.size)
            if max_edge > OLLAMA_VISION_MAX_EDGE:
                scale = OLLAMA_VISION_MAX_EDGE / max_edge
                resized_size = (
                    max(1, int(image.width * scale)),
                    max(1, int(image.height * scale)),
                )
                image = image.resize(resized_size, PILImage.Resampling.LANCZOS)
            buffer = BytesIO()
            image.save(buffer, format="PNG")
            return base64.b64encode(buffer.getvalue()).decode("utf-8")
    except Exception as exc:
        raise ValueError(f"Could not prepare image for Ollama vision: {image_path}") from exc


def _reference_images_for_scene(scene: dict, character_references: dict) -> list[tuple[str, str]]:
    """Return existing character reference image paths for the scene."""
    references = []
    for char_id in scene.get("characters_in_scene", []):
        reference = character_references.get(char_id)
        if not isinstance(reference, dict):
            continue
        path = reference.get("path")
        if isinstance(path, str) and os.path.exists(path):
            references.append((char_id, path))
    return references


def _project_reference_images(project_references: list[dict]) -> list[tuple[str, str, str, str]]:
    """Return existing project reference image paths."""
    references = []
    for reference in project_references:
        if not isinstance(reference, dict):
            continue
        path = reference.get("path")
        if not isinstance(path, str) or not os.path.exists(path):
            continue
        reference_id = str(reference.get("id") or "project_reference")
        kind = str(reference.get("kind") or "style")
        label = str(reference.get("label") or kind.title())
        references.append((reference_id, kind, label, path))
    return references


def _direct_reference_images_for_scene(
    scene: dict,
    character_references: dict,
    project_references: list[dict],
) -> list[tuple[str, str]]:
    """Return reference image labels and paths that should condition keyframe generation."""
    scene_references = [
        (f"Character {char_id}", path)
        for char_id, path in _reference_images_for_scene(scene, character_references)
    ]
    project_reference_items = [
        (f"{kind.title()} {label}", path)
        for _, kind, label, path in _project_reference_images(project_references)
    ]
    return scene_references + project_reference_items


def _build_reference_board(
    references: list[tuple[str, str]],
    output_path: str,
    width: int,
    height: int,
) -> str:
    """Create one scene reference image for ComfyUI from all active references."""
    from PIL import Image as PILImage

    if not references:
        raise ValueError("Cannot build a reference board without reference images.")

    board = PILImage.new("RGB", (width, height), (18, 18, 18))
    count = len(references)
    columns = 1 if count == 1 else 2
    rows = (count + columns - 1) // columns
    cell_width = width // columns
    cell_height = height // rows

    for index, (_, path) in enumerate(references):
        column = index % columns
        row = index // columns
        x0 = column * cell_width
        y0 = row * cell_height
        with PILImage.open(path) as image:
            rgb_image = image.convert("RGB")
            scale = min(cell_width / rgb_image.width, cell_height / rgb_image.height)
            resized_size = (
                max(1, int(rgb_image.width * scale)),
                max(1, int(rgb_image.height * scale)),
            )
            resized = rgb_image.resize(resized_size, PILImage.Resampling.LANCZOS)
            paste_x = x0 + (cell_width - resized.width) // 2
            paste_y = y0 + (cell_height - resized.height) // 2
            board.paste(resized, (paste_x, paste_y))

    board.save(output_path)
    return output_path


def _project_reference_prompt_guidance(project_references: list[dict]) -> str:
    """Build text guidance from project references for text-only keyframe workflows."""
    reference_images = _project_reference_images(project_references)
    if not reference_images:
        return ""
    lines = []
    for index, (_, kind, label, _) in enumerate(reference_images, start=1):
        if kind == "character":
            instruction = (
                "Preserve the face, body shape, age, hair/fur, clothing language, "
                "and silhouette implied by this character reference."
            )
        elif kind == "style":
            instruction = (
                "Match the color palette, lighting, texture, lens language, art direction, "
                "and overall visual finish implied by this style reference."
            )
        elif kind == "location":
            instruction = (
                "Keep the architecture, spatial layout, materials, atmosphere, and location "
                "logic consistent with this place reference."
            )
        else:
            instruction = (
                "Keep the shape, material, scale, and design language of this prop consistent "
                "with the reference."
            )
        lines.append(f"{index}. {kind.upper()} reference '{label}': {instruction}")
    return "\n\nPROJECT VISUAL REFERENCES:\n" + "\n".join(lines)


def _reference_prompt_block(
    reference_images: list[tuple[str, str]],
    project_reference_images: list[tuple[str, str, str, str]],
) -> str:
    """Build the reference image instructions for visual evaluation."""
    if not reference_images and not project_reference_images:
        return ""
    character_lines = "\n".join(
        f"- Reference image {index + 2}: {char_id}"
        for index, (char_id, _) in enumerate(reference_images)
    )
    project_start_index = len(reference_images) + 2
    project_lines = "\n".join(
        f"- Reference image {project_start_index + index}: {kind} / {label}"
        for index, (_, kind, label, _) in enumerate(project_reference_images)
    )
    character_block = ""
    if character_lines:
        character_block = f"""

LOCKED CHARACTER REFERENCES:
{character_lines}

REFERENCE_MATCH: Compare each visible character against their locked reference image.
   - Face, fur/skin markings, body shape, age, clothing, and silhouette must stay consistent
   - FAIL if the character looks like a different individual from the reference
   - FAIL if facial structure is warped, melted, or visibly inconsistent with the reference
"""
    project_block = ""
    if project_lines:
        project_block = f"""

PROJECT REFERENCE IMAGES:
{project_lines}

PROJECT_REFERENCE_MATCH: Compare the candidate against project references.
   - Character references guide identity, face/body proportions, clothing language, and silhouette
   - Style references guide palette, lighting, lens language, texture, mood, and finish
   - Location references guide architecture, layout, materials, and atmosphere
   - Prop references guide shape, scale, materials, and design language
   - Mark poor only when the candidate clearly ignores an important project reference
"""
    return f"""

REFERENCE IMAGES:
Image 1 is the candidate keyframe.
{character_block}
{project_block}
"""


def evaluate_keyframe(
    image_path: str,
    scene: dict,
    characters: dict,
    character_references: dict,
    project_references: list[dict],
) -> dict:
    """Thoroughly evaluate a keyframe image against the scene + character descriptions.

    Returns dict with scores and verdict.
    """
    img_b64 = image_to_base64(image_path)
    reference_images = _reference_images_for_scene(scene, character_references)
    project_reference_images = _project_reference_images(project_references)
    reference_prompt = _reference_prompt_block(reference_images, project_reference_images)
    reference_schema = (
        '"reference_match": "good|fair|poor",\n'
        if reference_images else ""
    )
    project_reference_schema = (
        '"project_reference_match": "good|fair|poor",\n'
        if project_reference_images else ""
    )

    # Build character checklist
    chars_in_scene = scene.get("characters_in_scene", [])
    char_checklist = ""
    if chars_in_scene and characters:
        char_checklist = "\n\nCHARACTER CHECKLIST -- verify EACH of these against the image:\n"
        for char_id in chars_in_scene:
            desc = characters.get(char_id, "")
            if desc:
                char_checklist += f"\n{char_id}:\n{desc}\n"
                char_checklist += "Check: face shape? skin tone? hair color/style? clothing match? age range? build?\n"

    eval_prompt = f"""You are a STRICT storyboard quality control reviewer. Your job is to REJECT images that don't match the character descriptions or scene requirements. Do NOT approve mediocre images.

SCENE REQUIREMENTS:
- Description: {scene['description']}
- Shot type: {scene.get('shot_type', 'not specified')}
- Setting: {scene.get('setting_description', 'not specified')}
- Lighting: {scene.get('lighting_description', 'not specified')}
- Mood: {scene.get('mood', 'not specified')}
{char_checklist}

EVALUATE EACH CATEGORY -- score as "good", "fair", or "poor":

1. CHARACTER_ACCURACY: Does each character match their description?
   - Compare face, skin tone, hair, age, build against the description POINT BY POINT
   - Is the clothing exactly right? (color, style, accessories)
   - FAIL if: wrong face features, wrong hair color, wrong clothing, wrong age range
   - FAIL if: the person is clearly not who they're supposed to be

2. SETTING_ACCURACY: Does the environment match?
   - Background, room type, props, furniture
   - FAIL if: completely wrong location or setting

3. COMPOSITION: Is the shot type correct?
   - Camera angle, framing, subject placement
   - FAIL if: requested close-up but got wide shot, etc.

4. LIGHTING_MOOD: Does the lighting/atmosphere match?
   - Color temperature, shadow direction, mood
   - Minor variations are OK

5. IMAGE_QUALITY: Technical quality
   - No distorted faces, extra limbs, text artifacts, blurriness
   - FAIL if: deformed features, extra fingers, melted faces
{reference_prompt}

VERDICT RULES:
- FAIL only for major blockers that would ruin production:
  - the main character is clearly the wrong person, wrong gender, wrong species, or wrong age group
  - severe face/body deformation, extra limbs, unreadable image, or major technical artifact
  - locked reference images are provided and the character clearly does not match the reference
- PASS WITH WARNINGS for usable storyboard frames with imperfect props, imperfect lighting,
  approximate shot type, missing minor texture details, or partially matching setting.
- Do NOT fail only because a prop material is approximate, the exact camera transition cannot be shown
  in one still frame, the wall/floor texture is not exact, or the mood color is only approximate.

Think carefully about each point before scoring.

Respond with valid JSON:
{{"character_accuracy": "good|fair|poor", "setting_accuracy": "good|fair|poor",
"composition": "good|fair|poor", "lighting_mood": "good|fair|poor",
"image_quality": "good|fair|poor", {reference_schema}{project_reference_schema}"verdict": "PASS|FAIL",
"fail_reason": "specific description of what's wrong, or null",
"character_notes": "what specifically matches or doesn't match the character description",
"reference_notes": "what matches or doesn't match the locked reference images, or null"}}"""

    images = [img_b64]
    images.extend(image_to_base64(path) for _, path in reference_images)
    images.extend(image_to_base64(path) for _, _, _, path in project_reference_images)
    response = ollama_client.chat(
        model=OLLAMA_MODEL_FAST,
        messages=[{"role": "user", "content": eval_prompt, "images": images}],
        options={
            "num_predict": 1024,
            "temperature": 0.2,  # Very analytical
        },
    )

    raw = response["message"]["content"].strip()
    try:
        # Find JSON in response
        start = raw.find("{")
        end = raw.rfind("}") + 1
        if start >= 0 and end > start:
            result = json.loads(raw[start:end])
        else:
            raise ValueError("No JSON found")

        result["verdict"] = result.get("verdict", "FAIL").upper()
        if result["verdict"] not in ("PASS", "FAIL"):
            result["verdict"] = "FAIL"

        # Enforce rules programmatically
        result = _enforce_keyframe_rules(result)
        return result

    except (json.JSONDecodeError, ValueError):
        log.warning("Keyframe eval parse failed: %s", raw[:200])
        return {
            "character_accuracy": "unknown", "setting_accuracy": "unknown",
            "composition": "unknown", "lighting_mood": "unknown",
            "image_quality": "unknown", "verdict": "FAIL",
            "reference_match": "unknown" if reference_images else None,
            "project_reference_match": "unknown" if project_reference_images else None,
            "fail_reason": "Evaluation failed to parse",
            "character_notes": None,
            "reference_notes": None,
        }


def _manual_keyframe_eval(reason: str) -> dict:
    """Return a non-blocking manual-review result when AI evaluation is unavailable."""
    return {
        "character_accuracy": "unknown",
        "setting_accuracy": "unknown",
        "composition": "unknown",
        "lighting_mood": "unknown",
        "image_quality": "unknown",
        "verdict": "PASS",
        "quality_gate": "manual_review",
        "notes": "Manual review required",
        "warning_reason": reason,
        "fail_reason": None,
        "character_notes": reason,
        "reference_notes": None,
    }


def _enforce_keyframe_rules(result: dict) -> dict:
    """Programmatically enforce FAIL conditions."""
    hard_fail_categories = ["character_accuracy", "image_quality"]
    if result.get("reference_match") is not None:
        hard_fail_categories.append("reference_match")

    hard_fail_reasons = []
    if result.get("character_accuracy") == "poor":
        hard_fail_reasons.append("character_accuracy")
    if result.get("image_quality") == "poor":
        hard_fail_reasons.append("image_quality")
    if result.get("reference_match") == "poor":
        hard_fail_reasons.append("reference_match")

    if hard_fail_reasons:
        result["verdict"] = "FAIL"
        if not result.get("fail_reason"):
            result["fail_reason"] = f"Hard fail: {', '.join(hard_fail_reasons)}"
        result.pop("quality_gate", None)
        return result

    soft_categories = [
        "setting_accuracy", "composition", "lighting_mood", "project_reference_match",
        *[category for category in hard_fail_categories if result.get(category) == "fair"],
    ]
    warnings = [
        category
        for category in soft_categories
        if result.get(category) in {"fair", "poor"}
    ]
    if result.get("verdict") == "FAIL" or warnings:
        result["verdict"] = "PASS"
        result["quality_gate"] = "warning"
        if result.get("fail_reason"):
            result["warning_reason"] = result["fail_reason"]
            result["fail_reason"] = None
        elif warnings:
            result["warning_reason"] = f"Soft warning: {', '.join(warnings)}"
    else:
        result.pop("quality_gate", None)

    return result


def _rewrite_keyframe_prompt(scene: dict, fail_reasons: list[str], original_prompt: str) -> str:
    """Ask the LLM to rewrite the image prompt based on what kept failing."""
    reasons_text = "\n".join(f"  - Attempt {i+1}: {r}" for i, r in enumerate(fail_reasons))

    rewrite_request = f"""The following image prompt was used to generate keyframe images for a scene, but it FAILED evaluation {len(fail_reasons)} times in a row.

ORIGINAL PROMPT:
{original_prompt}

FAILURE REASONS:
{reasons_text}

SCENE DESCRIPTION: {scene['description']}

Rewrite the prompt to fix these issues. Focus on:
- If characters didn't match: emphasize their physical description more prominently
- If setting was wrong: be more explicit about the environment
- If composition was bad: specify the camera angle and framing more clearly
- Simplify overly complex descriptions that the image model can't handle
- Keep it to one clear moment — don't describe sequential actions

Respond with ONLY the rewritten prompt text, nothing else."""

    response = ollama_client.chat(
        model=OLLAMA_MODEL_FAST,
        messages=[{"role": "user", "content": rewrite_request}],
        options={"num_predict": 2048, "temperature": 0.5},
    )
    return response["message"]["content"].strip()


def _run_keyframe_round(client, template: dict, scene: dict, characters: dict,
                        keyframe_dir: str, prompt: str, candidates: list,
                        character_references: dict, project_references: list[dict],
                        max_attempts: int, scene_num: int) -> bool:
    """Run a round of keyframe generation attempts. Returns True if one passed."""
    seed = random.randint(0, 2**32 - 1)
    start_idx = len(candidates)

    for i in range(max_attempts):
        candidate_num = start_idx + i + 1
        log.info("  Keyframe %d for scene %d (seed %d)...",
                 candidate_num, scene_num, seed)

        try:
            direct_references = _direct_reference_images_for_scene(
                scene,
                character_references,
                project_references,
            )
            reference_image_filename = None
            if direct_references:
                board_path = os.path.join(
                    keyframe_dir,
                    f"scene_{scene_num:03d}_reference_board.png",
                )
                _build_reference_board(direct_references, board_path, KF_WIDTH, KF_HEIGHT)
                reference_image_filename = client.upload_image(board_path)
                log.info(
                    "  Direct reference image conditioning enabled with %d reference(s)",
                    len(direct_references),
                )

            workflow = build_keyframe_workflow(
                template,
                prompt,
                seed,
                width=KF_WIDTH,
                height=KF_HEIGHT,
                reference_image_filename=reference_image_filename,
            )
            prompt_id = client.queue_prompt(workflow)
            history = client.wait_for_completion(prompt_id, timeout=120)
            raw_path = client.get_output_file(
                history, keyframe_dir, (".png", ".jpg", ".jpeg", ".webp")
            )
        except Exception as e:
            log.error("  Keyframe gen failed: %s", e)
            candidates.append({"candidate": candidate_num, "status": "failed", "error": str(e)})
            seed = random.randint(0, 2**32 - 1)
            continue

        kf_filename = f"scene_{scene_num:03d}_kf_{candidate_num}.png"
        kf_path = os.path.join(keyframe_dir, kf_filename)
        shutil.copy2(raw_path, kf_path)

        if SKIP_KF_EVAL:
            log.info("  Keyframe %d generated (evaluation skipped)", candidate_num)
            eval_result = _manual_keyframe_eval("AI keyframe evaluation was skipped.")
        else:
            log.info("  Evaluating keyframe %d...", candidate_num)
            try:
                eval_result = evaluate_keyframe(
                    kf_path,
                    scene,
                    characters,
                    character_references,
                    project_references,
                )
            except Exception as exc:
                eval_result = _manual_keyframe_eval(
                    f"AI keyframe evaluation timed out or failed: {exc}"
                )
                log.warning("  Keyframe %d evaluation unavailable: %s", candidate_num, exc)
            log.info("  Keyframe %d: %s -- %s",
                     candidate_num, eval_result["verdict"],
                     eval_result.get("fail_reason")
                     or eval_result.get("warning_reason")
                     or eval_result.get("reference_notes")
                     or eval_result.get("character_notes", "OK"))

        candidates.append({
            "candidate": candidate_num,
            "status": "generated",
            "path": kf_path,
            "seed": seed,
            "eval": eval_result,
        })

        if eval_result["verdict"] == "PASS":
            log.info("  Keyframe PASSED -- moving on from scene %d", scene_num)
            return True

        seed = random.randint(0, 2**32 - 1)

    return False


def generate_keyframes(client, scene: dict, characters: dict,
                       project_dir: str, character_references: dict,
                       project_references: list[dict], brief: str = "") -> list[dict]:
    """Generate keyframe candidates for a scene with prompt rewriting on failure.

    Round 1: Try KF_CANDIDATES attempts with the original prompt.
    If all fail, rewrite the prompt based on failure reasons and try again.
    """
    from director import write_prompt

    template = load_keyframe_template()
    scene_num = scene["scene_number"]

    # Write an image-specific prompt (reusing the rich prompt writer)
    if not scene.get("keyframe_prompt"):
        scene["keyframe_prompt"] = write_prompt(scene, brief=brief)

    prompt = scene["keyframe_prompt"] + _project_reference_prompt_guidance(project_references)
    log.info("  Image prompt (%d words):", len(prompt.split()))
    for line in prompt.split("\n"):
        log.info("    | %s", line)

    keyframe_dir = os.path.join(project_dir, "keyframes")
    os.makedirs(keyframe_dir, exist_ok=True)

    candidates = []

    # Round 1: original prompt
    log.info("  Round 1: trying %d candidates with original prompt", KF_CANDIDATES)
    if _run_keyframe_round(client, template, scene, characters,
                           keyframe_dir, prompt, candidates,
                           character_references, project_references,
                           KF_CANDIDATES, scene_num):
        return candidates

    # All failed — collect failure reasons and rewrite the prompt
    fail_reasons = []
    for c in candidates:
        reason = c.get("eval", {}).get("fail_reason") or c.get("error", "unknown")
        if reason:
            fail_reasons.append(reason)

    log.info("  All %d candidates failed. Rewriting prompt based on failures...", KF_CANDIDATES)
    new_prompt = _rewrite_keyframe_prompt(scene, fail_reasons, prompt)
    scene["keyframe_prompt"] = new_prompt
    new_prompt_with_references = new_prompt + _project_reference_prompt_guidance(project_references)

    log.info("  Rewritten prompt (%d words):", len(new_prompt_with_references.split()))
    for line in new_prompt_with_references.split("\n"):
        log.info("    | %s", line)

    # Round 2: rewritten prompt
    log.info("  Round 2: trying %d candidates with rewritten prompt", KF_CANDIDATES)
    _run_keyframe_round(client, template, scene, characters,
                        keyframe_dir, new_prompt_with_references, candidates,
                        character_references, project_references,
                        KF_CANDIDATES, scene_num)

    return candidates
