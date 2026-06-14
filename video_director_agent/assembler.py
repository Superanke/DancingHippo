# assembler.py — FFmpeg concat of approved clips + final continuity eval

import os
import subprocess
import tempfile
import logging
import ollama_client

from config import OLLAMA_MODEL_FAST, EVAL_TOKEN_BUDGET, FFMPEG_PATH, FFPROBE_PATH
from evaluator import extract_frames

log = logging.getLogger(__name__)


def concat_scenes(scene_paths: list[str], output_path: str):
    """Concatenate scene clips using FFmpeg concat demuxer (no re-encode)."""
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    # Write concat list to temp file
    list_file = os.path.join(tempfile.gettempdir(), "concat_list.txt")
    with open(list_file, "w") as f:
        for path in scene_paths:
            abs_path = os.path.abspath(path).replace("\\", "/")
            f.write(f"file '{abs_path}'\n")

    log.info("Concatenating %d clips into %s", len(scene_paths), output_path)

    # First try lossless copy
    result = subprocess.run(
        [
            FFMPEG_PATH, "-y",
            "-f", "concat",
            "-safe", "0",
            "-i", list_file,
            "-c", "copy",
            output_path,
        ],
        capture_output=True, text=True,
    )

    if result.returncode != 0:
        log.warning("Lossless concat failed, re-encoding: %s", result.stderr[:200])
        # Fallback: re-encode to ensure compatibility
        subprocess.run(
            [
                FFMPEG_PATH, "-y",
                "-f", "concat",
                "-safe", "0",
                "-i", list_file,
                "-c:v", "libx264", "-crf", "18", "-preset", "fast",
                "-c:a", "aac", "-b:a", "192k",
                "-movflags", "+faststart",
                output_path,
            ],
            capture_output=True, text=True, check=True,
        )

    log.info("Final film assembled: %s", output_path)


def probe_clip(path: str) -> dict:
    """Get clip metadata via ffprobe."""
    result = subprocess.run(
        [
            FFPROBE_PATH, "-v", "quiet",
            "-print_format", "json",
            "-show_streams",
            path,
        ],
        capture_output=True, text=True,
    )
    if result.returncode != 0:
        return {}
    import json
    return json.loads(result.stdout)


def final_continuity_eval(final_path: str, scenes: list[dict]) -> dict:
    """Sample frames across the full assembled video and check pacing/continuity."""
    frames = extract_frames(final_path, fps_sample=0.5)  # 1 frame every 2 seconds

    # Subsample to ~30 frames for the full film eval
    if len(frames) > 30:
        step = len(frames) // 30
        frames = frames[::step][:30]

    scene_summary = "\n".join(
        f"  Scene {s['scene_number']}: {s['description']} ({s['duration_seconds']}s)"
        for s in scenes
    )

    eval_prompt = f"""You are reviewing a completed AI-generated film assembled from {len(scenes)} scenes.

Scene plan:
{scene_summary}

These frames are sampled across the full video. Evaluate:
1. PACING: Does the film flow well between scenes?
2. CONTINUITY: Are there jarring visual jumps between scenes?
3. OVERALL_QUALITY: General impression.
4. SUGGESTIONS: Any scenes that should be regenerated?

Respond ONLY with valid JSON:
{{"pacing": "good|fair|poor", "continuity": "good|fair|poor",
"overall_quality": "good|fair|poor",
"notes": "brief overall impression",
"problem_scenes": [list of scene numbers that need work, or empty list]}}"""

    log.info("Running final continuity eval on assembled film (%d frames)...", len(frames))
    response = ollama_client.chat(
        model=OLLAMA_MODEL_FAST,
        messages=[{"role": "user", "content": eval_prompt, "images": frames}],
        options={"num_predict": EVAL_TOKEN_BUDGET},
    )

    raw = response["message"]["content"].strip()
    try:
        start = raw.find("{")
        end = raw.rfind("}") + 1
        if start >= 0 and end > start:
            raw = raw[start:end]
        import json
        return json.loads(raw)
    except Exception:
        log.warning("Final eval parse failed: %s", raw[:200])
        return {"pacing": "unknown", "continuity": "unknown", "overall_quality": "unknown",
                "notes": raw[:200], "problem_scenes": []}
