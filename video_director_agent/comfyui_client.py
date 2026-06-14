# comfyui_client.py — ComfyUI API: queue, watch, retrieve

import uuid
import json
import copy
import os
import time
import logging
import urllib.parse
import urllib.error
import urllib.request
import websocket

from config import (
    COMFYUI_HOST, COMFYUI_OUTPUT_DIR,
    PROMPT_NODE_ID, NEG_PROMPT_NODE_ID, FRAMES_NODE_ID,
    SEED_NODE_ID_PASS1, SEED_NODE_ID_PASS2,
    VIDEO_RES_NODE_ID, VIDEO_WIDTH, VIDEO_HEIGHT,
    LTX_FPS, NEGATIVE_PROMPT,
)

log = logging.getLogger(__name__)


class ComfyUIClient:
    def __init__(self, host: str = COMFYUI_HOST):
        self.host = host
        self.client_id = str(uuid.uuid4())
        self.ws = None

    # ── Connection ─────────────────────────────────────────────────────────

    def connect(self):
        """Open WebSocket to ComfyUI for execution tracking."""
        ws_url = f"ws://{self.host}/ws?clientId={self.client_id}"
        log.info("Connecting to ComfyUI at %s", ws_url)
        self.ws = websocket.WebSocket()
        self.ws.connect(ws_url)
        log.info("Connected.")

    def disconnect(self):
        if self.ws:
            self.ws.close()
            self.ws = None

    def check_alive(self) -> bool:
        """Verify ComfyUI is reachable."""
        try:
            url = f"http://{self.host}/system_stats"
            with urllib.request.urlopen(url, timeout=5) as r:
                return r.status == 200
        except Exception:
            return False

    def upload_image(self, image_path: str, subfolder: str = "", overwrite: bool = True) -> str:
        """Upload an image to ComfyUI's input directory. Returns the filename."""
        import mimetypes
        filename = os.path.basename(image_path)
        content_type = mimetypes.guess_type(image_path)[0] or "image/png"

        boundary = f"----WebKitFormBoundary{uuid.uuid4().hex[:16]}"
        body = b""

        # image file part
        body += f"--{boundary}\r\n".encode()
        body += f'Content-Disposition: form-data; name="image"; filename="{filename}"\r\n'.encode()
        body += f"Content-Type: {content_type}\r\n\r\n".encode()
        with open(image_path, "rb") as f:
            body += f.read()
        body += b"\r\n"

        # subfolder part
        if subfolder:
            body += f"--{boundary}\r\n".encode()
            body += b'Content-Disposition: form-data; name="subfolder"\r\n\r\n'
            body += subfolder.encode() + b"\r\n"

        # overwrite part
        body += f"--{boundary}\r\n".encode()
        body += b'Content-Disposition: form-data; name="overwrite"\r\n\r\n'
        body += (b"true" if overwrite else b"false") + b"\r\n"

        body += f"--{boundary}--\r\n".encode()

        req = urllib.request.Request(
            f"http://{self.host}/upload/image",
            data=body,
            headers={"Content-Type": f"multipart/form-data; boundary={boundary}"},
            method="POST",
        )
        try:
            with urllib.request.urlopen(req) as resp:
                result = json.loads(resp.read())
        except urllib.error.HTTPError as e:
            body = e.read().decode("utf-8", errors="replace")
            raise RuntimeError(
                f"ComfyUI /upload/image failed: status={e.code}, body={body}"
            ) from e
        uploaded_name = result.get("name", filename)
        log.info("Uploaded image to ComfyUI: %s", uploaded_name)
        return uploaded_name

    # ── Queue & Wait ───────────────────────────────────────────────────────

    def queue_prompt(self, workflow: dict) -> str:
        """Submit a workflow to ComfyUI. Returns the prompt_id."""
        payload = json.dumps({
            "prompt": workflow,
            "client_id": self.client_id,
        }).encode("utf-8")
        req = urllib.request.Request(
            f"http://{self.host}/prompt",
            data=payload,
            headers={"Content-Type": "application/json"},
        )
        try:
            with urllib.request.urlopen(req) as resp:
                result = json.loads(resp.read())
        except urllib.error.HTTPError as e:
            body = e.read().decode("utf-8", errors="replace")
            raise RuntimeError(
                f"ComfyUI /prompt failed: status={e.code}, body={body}"
            ) from e
        prompt_id = result["prompt_id"]
        log.info("Queued prompt %s", prompt_id)
        return prompt_id

    def wait_for_completion(self, prompt_id: str, timeout: int = 900) -> dict:
        """Block on WebSocket until execution completes or errors.

        Falls back to polling /history if the socket drops.
        Returns the history dict for this prompt_id.
        """
        deadline = time.time() + timeout
        while time.time() < deadline:
            try:
                self.ws.settimeout(30)
                raw = self.ws.recv()

                # ComfyUI sends binary frames for latent preview images — skip them
                if not isinstance(raw, str):
                    continue

                msg = json.loads(raw)
                msg_type = msg.get("type")
                data = msg.get("data", {})

                if msg_type == "executing":
                    if data.get("node") is None and data.get("prompt_id") == prompt_id:
                        time.sleep(0.5)  # Let outputs persist to history
                        return self.get_history(prompt_id)

                elif msg_type == "execution_success":
                    if data.get("prompt_id") == prompt_id:
                        time.sleep(0.5)
                        return self.get_history(prompt_id)

                elif msg_type in ("execution_error", "execution_interrupted"):
                    if data.get("prompt_id") == prompt_id:
                        err = data.get("exception_message", "Unknown error")
                        raise RuntimeError(f"ComfyUI execution error: {err}")

            except json.JSONDecodeError:
                log.debug("Non-JSON text message received, skipping")
                continue
            except websocket.WebSocketTimeoutException:
                # Check history as fallback
                hist = self._poll_history(prompt_id)
                if hist is not None:
                    return hist
            except (websocket.WebSocketConnectionClosedException, ConnectionError):
                log.warning("WebSocket dropped, reconnecting...")
                time.sleep(2)
                self.connect()

        raise TimeoutError(f"Prompt {prompt_id} did not complete within {timeout}s")

    def _poll_history(self, prompt_id: str):
        """Check if prompt already finished (fallback when WS is unreliable)."""
        try:
            hist = self.get_history(prompt_id)
            if hist and hist.get("outputs"):
                return hist
        except Exception:
            pass
        return None

    # ── History & Output ───────────────────────────────────────────────────

    def get_history(self, prompt_id: str) -> dict:
        url = f"http://{self.host}/history/{prompt_id}"
        with urllib.request.urlopen(url) as r:
            data = json.loads(r.read())
        return data.get(prompt_id, {})

    @staticmethod
    def _find_output_item(history: dict, suffixes: tuple[str, ...]) -> dict:
        """Extract the first matching ComfyUI output item from history."""
        outputs = history.get("outputs", {})
        for node_id, node_output in outputs.items():
            for key in ("images", "videos", "gifs"):
                if key not in node_output:
                    continue
                items = node_output[key]
                for item in items:
                    if not isinstance(item, dict):
                        continue
                    filename = item.get("filename", "")
                    if filename.endswith(suffixes):
                        return item
        raise ValueError(f"No output found for suffixes: {suffixes}")

    @staticmethod
    def get_output_path(history: dict, output_dir: str = COMFYUI_OUTPUT_DIR) -> str:
        """Extract the video file path from a completed history dict."""
        item = ComfyUIClient._find_output_item(history, (".mp4", ".webm", ".avi", ".mov"))
        subfolder = item.get("subfolder", "")
        return os.path.join(output_dir, subfolder, item["filename"])

    def download_output_item(self, item: dict, local_dir: str) -> str:
        """Download an output file from ComfyUI's /view API into local_dir."""
        os.makedirs(local_dir, exist_ok=True)
        filename = item.get("filename", "")
        if not filename:
            raise ValueError(f"ComfyUI output item has no filename: {item}")

        params = {
            "filename": filename,
            "subfolder": item.get("subfolder", ""),
            "type": item.get("type", "output"),
        }
        url = f"http://{self.host}/view?{urllib.parse.urlencode(params)}"
        local_path = os.path.join(local_dir, os.path.basename(filename))
        log.info("Downloading ComfyUI output: %s", url)
        with urllib.request.urlopen(url, timeout=120) as resp:
            data = resp.read()
        with open(local_path, "wb") as f:
            f.write(data)
        return local_path

    def get_output_file(self, history: dict, local_dir: str,
                        suffixes: tuple[str, ...]) -> str:
        """Return a local file for a ComfyUI output, downloading it if needed."""
        item = self._find_output_item(history, suffixes)
        subfolder = item.get("subfolder", "")
        local_comfy_path = os.path.join(COMFYUI_OUTPUT_DIR, subfolder, item["filename"])
        if os.path.exists(local_comfy_path):
            return local_comfy_path
        return self.download_output_item(item, local_dir)


# ── Workflow helpers ───────────────────────────────────────────────────────

def _find_nodes_by_class(wf: dict, class_type: str) -> list[str]:
    """Find all node IDs in a workflow matching a given class_type."""
    return [nid for nid, node in wf.items()
            if isinstance(node, dict) and node.get("class_type") == class_type]


def _detect_video_nodes(wf: dict) -> dict:
    """Auto-detect video workflow node IDs from the template.

    Returns a dict with keys: prompt, neg_prompt, frames, seed1, seed2, resolution.
    Falls back to config defaults for any nodes it can't find.
    """
    detected = {
        "prompt": PROMPT_NODE_ID,
        "neg_prompt": NEG_PROMPT_NODE_ID,
        "frames": FRAMES_NODE_ID,
        "seed1": SEED_NODE_ID_PASS1,
        "seed2": SEED_NODE_ID_PASS2,
        "resolution": VIDEO_RES_NODE_ID,
    }

    # Check if defaults already exist in the workflow
    all_present = all(k in wf for k in [
        PROMPT_NODE_ID, NEG_PROMPT_NODE_ID, FRAMES_NODE_ID,
        SEED_NODE_ID_PASS1, SEED_NODE_ID_PASS2, VIDEO_RES_NODE_ID
    ])
    if all_present:
        return detected

    log.info("Default node IDs not found in workflow — auto-detecting...")

    # Find CLIPTextEncode nodes (positive + negative prompt)
    clip_nodes = _find_nodes_by_class(wf, "CLIPTextEncode")
    if len(clip_nodes) >= 2:
        detected["prompt"] = clip_nodes[0]
        detected["neg_prompt"] = clip_nodes[1]
    elif len(clip_nodes) == 1:
        detected["prompt"] = clip_nodes[0]

    # Find RandomNoise nodes (seeds for pass 1 and pass 2)
    noise_nodes = _find_nodes_by_class(wf, "RandomNoise")
    if len(noise_nodes) >= 2:
        detected["seed1"] = noise_nodes[0]
        detected["seed2"] = noise_nodes[1]
    elif len(noise_nodes) == 1:
        detected["seed1"] = noise_nodes[0]
        detected["seed2"] = noise_nodes[0]

    # Find frame count node (PrimitiveInt or similar)
    for class_name in ["PrimitiveInt", "Primitive int", "INTConstant"]:
        frame_nodes = _find_nodes_by_class(wf, class_name)
        if frame_nodes:
            detected["frames"] = frame_nodes[0]
            break

    # Find resolution node (EmptyImage)
    res_nodes = _find_nodes_by_class(wf, "EmptyImage")
    if res_nodes:
        detected["resolution"] = res_nodes[0]

    log.info("Auto-detected node IDs: %s", detected)
    return detected


def load_workflow_template(path: str = None) -> dict:
    """Load the API-format workflow template JSON."""
    if path is None:
        path = os.path.join(os.path.dirname(__file__), "workflow_template.json")
    with open(path) as f:
        return json.load(f)


def build_workflow(template: dict, prompt_text: str, frames: int, seed: int) -> dict:
    """Clone the template and inject per-scene values.

    Only touches prompt text, frame count, and seeds.
    All other settings (sampler, sigmas, CFG, models, LoRA, etc.)
    stay exactly as tuned in the template.
    Auto-detects node IDs if the configured defaults don't match the template.
    """
    wf = copy.deepcopy(template)
    nodes = _detect_video_nodes(wf)
    wf[nodes["prompt"]]["inputs"]["text"] = prompt_text
    wf[nodes["neg_prompt"]]["inputs"]["text"] = NEGATIVE_PROMPT
    wf[nodes["frames"]]["inputs"]["value"] = frames
    wf[nodes["seed1"]]["inputs"]["noise_seed"] = seed
    wf[nodes["seed2"]]["inputs"]["noise_seed"] = seed + 1
    wf[nodes["resolution"]]["inputs"]["width"] = VIDEO_WIDTH
    wf[nodes["resolution"]]["inputs"]["height"] = VIDEO_HEIGHT
    return wf


def calc_frames(duration_sec: int, fps: int = LTX_FPS) -> int:
    """Calculate frame count obeying LTX rule: frames = (8n + 1)."""
    raw = duration_sec * fps
    n = round((raw - 1) / 8)
    return (8 * n) + 1


# ── I2V (Image-to-Video) Workflow ─────────────────────────────────────────

def load_i2v_template(path: str = None) -> dict:
    """Load the i2v API-format workflow template JSON."""
    if path is None:
        path = os.path.join(os.path.dirname(__file__), "i2v_template.json")
    if not os.path.exists(path):
        return None
    with open(path) as f:
        return json.load(f)


def _find_node_by_title(wf: dict, title: str) -> str | None:
    """Find a node ID by its _meta title."""
    for nid, node in wf.items():
        if isinstance(node, dict):
            meta_title = node.get("_meta", {}).get("title", "")
            if meta_title.lower() == title.lower():
                return nid
    return None


def _detect_i2v_nodes(wf: dict) -> dict:
    """Auto-detect i2v workflow node IDs from the template."""
    detected = {}

    # Find prompt node — PrimitiveStringMultiline (titled "Prompt") or CLIPTextEncode
    prompt_node = _find_node_by_title(wf, "Prompt")
    if prompt_node:
        detected["prompt"] = prompt_node
    else:
        # Fall back: find PrimitiveStringMultiline, then CLIPTextEncode
        for cls in ["PrimitiveStringMultiline", "CLIPTextEncode"]:
            nodes = _find_nodes_by_class(wf, cls)
            if nodes:
                detected["prompt"] = nodes[0]
                break

    # Find negative prompt — second CLIPTextEncode
    clip_nodes = _find_nodes_by_class(wf, "CLIPTextEncode")
    if len(clip_nodes) >= 2:
        detected["neg_prompt"] = clip_nodes[1]
    elif len(clip_nodes) == 1:
        detected["neg_prompt"] = clip_nodes[0]

    # Find seed nodes (RandomNoise)
    noise_nodes = _find_nodes_by_class(wf, "RandomNoise")
    if len(noise_nodes) >= 2:
        detected["seed1"] = noise_nodes[0]
        detected["seed2"] = noise_nodes[1]
    elif len(noise_nodes) == 1:
        detected["seed1"] = noise_nodes[0]
        detected["seed2"] = noise_nodes[0]

    # Find frame count node (titled "Length" or generic PrimitiveInt)
    length_node = _find_node_by_title(wf, "Length")
    if length_node:
        detected["frames"] = length_node
    else:
        int_nodes = _find_nodes_by_class(wf, "PrimitiveInt")
        if int_nodes:
            detected["frames"] = int_nodes[0]

    # Find width/height/fps nodes by title
    width_node = _find_node_by_title(wf, "Width")
    height_node = _find_node_by_title(wf, "Height")
    fps_node = _find_node_by_title(wf, "FPS")
    if width_node:
        detected["width"] = width_node
    if height_node:
        detected["height"] = height_node
    if fps_node:
        detected["fps"] = fps_node

    # Find LoadImage node (for the keyframe input)
    load_nodes = _find_nodes_by_class(wf, "LoadImage")
    if load_nodes:
        detected["load_image"] = load_nodes[0]

    # Find T2V switch (PrimitiveBoolean)
    switch_node = _find_node_by_title(wf, "Switch to Text to Video?")
    if switch_node:
        detected["t2v_switch"] = switch_node

    log.info("Auto-detected i2v node IDs: %s", detected)
    return detected


def build_i2v_workflow(template: dict, prompt_text: str, frames: int, seed: int,
                       image_filename: str) -> dict:
    """Clone the i2v template and inject per-scene values + keyframe image.

    image_filename should be the name returned by upload_image() (already in ComfyUI's input dir).
    """
    wf = copy.deepcopy(template)
    nodes = _detect_i2v_nodes(wf)

    # Set prompt text
    if "prompt" in nodes:
        node = wf[nodes["prompt"]]
        if "text" in node.get("inputs", {}):
            node["inputs"]["text"] = prompt_text
        elif "value" in node.get("inputs", {}):
            node["inputs"]["value"] = prompt_text

    # Set negative prompt
    if "neg_prompt" in nodes:
        wf[nodes["neg_prompt"]]["inputs"]["text"] = NEGATIVE_PROMPT

    # Set frame count
    if "frames" in nodes:
        wf[nodes["frames"]]["inputs"]["value"] = frames

    # Set seeds
    if "seed1" in nodes:
        wf[nodes["seed1"]]["inputs"]["noise_seed"] = seed
    if "seed2" in nodes:
        wf[nodes["seed2"]]["inputs"]["noise_seed"] = seed + 1

    # Set resolution
    if "width" in nodes:
        wf[nodes["width"]]["inputs"]["value"] = VIDEO_WIDTH
    if "height" in nodes:
        wf[nodes["height"]]["inputs"]["value"] = VIDEO_HEIGHT
    if "fps" in nodes:
        wf[nodes["fps"]]["inputs"]["value"] = LTX_FPS

    # Set the keyframe image
    if "load_image" in nodes:
        wf[nodes["load_image"]]["inputs"]["image"] = image_filename

    # Make sure T2V switch is off (we want i2v mode)
    if "t2v_switch" in nodes:
        wf[nodes["t2v_switch"]]["inputs"]["value"] = False

    return wf
