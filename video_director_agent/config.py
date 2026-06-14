# config.py — All settings in one place.
# User-specific paths are loaded from user_settings.json (auto-created on first run).

import json
import os

_SETTINGS_PATH = os.path.join(os.path.dirname(__file__), "user_settings.json")

# --- Defaults (overridden by user_settings.json) ---
_DEFAULTS = {
    "comfyui_root": r"C:\ComfyUI\ComfyUI_windows_portable",
    "comfyui_host": "127.0.0.1:8188",
    "comfyui_launcher": "run_nvidia_gpu.bat",
    "ollama_host": "http://localhost:11434",
    "ollama_model_creative": "gemma4:26b",
    "ollama_model_fast": "gemma4:e4b",
    "ollama_request_timeout_sec": 90,
    "ollama_vision_max_edge": 768,
    "kf_width": 2048,
    "kf_height": 1024,
    "video_width": 1024,
    "video_height": 432,
    "output_language": "english",
}


def _load_user_settings() -> dict:
    if os.path.exists(_SETTINGS_PATH):
        with open(_SETTINGS_PATH) as f:
            return json.load(f)
    return {}


def save_user_settings(settings: dict):
    merged = _load_user_settings()
    merged.update(settings)
    with open(_SETTINGS_PATH, "w") as f:
        json.dump(merged, f, indent=2)


_user = _load_user_settings()


def _get(key: str) -> str:
    return _user.get(key, _DEFAULTS[key])


# --- Derived paths from ComfyUI root ---
_comfyui_root = _get("comfyui_root")

COMFYUI_HOST = _get("comfyui_host")
COMFYUI_OUTPUT_DIR = os.path.join(_comfyui_root, "ComfyUI", "output")
COMFYUI_LAUNCHER = os.path.join(_comfyui_root, _get("comfyui_launcher"))
COMFYUI_STARTUP_TIMEOUT = 120

# --- FFmpeg (search in ComfyUI root, then system PATH) ---
_ffmpeg_candidates = [
    os.path.join(_comfyui_root, "ffmpeg.exe"),
    os.path.join(_comfyui_root, "ffmpeg", "ffmpeg.exe"),
]
FFMPEG_PATH = "ffmpeg"  # fallback to PATH
for _f in _ffmpeg_candidates:
    if os.path.exists(_f):
        FFMPEG_PATH = _f
        break

_ffprobe_candidates = [
    os.path.join(_comfyui_root, "ffprobe.exe"),
    os.path.join(_comfyui_root, "ffmpeg", "ffprobe.exe"),
]
# Search deeper for ffprobe
for _root, _dirs, _files in os.walk(_comfyui_root):
    for _fname in _files:
        if _fname == "ffprobe.exe":
            _ffprobe_candidates.append(os.path.join(_root, _fname))
    if len(_ffprobe_candidates) > 5:
        break
FFPROBE_PATH = "ffprobe"
for _f in _ffprobe_candidates:
    if os.path.exists(_f):
        FFPROBE_PATH = _f
        break

# --- Ollama / LLM ---
OLLAMA_HOST = _get("ollama_host")
OLLAMA_MODEL_CREATIVE = _get("ollama_model_fast")   # Using fast model for everything — 26B has JSON issues
OLLAMA_MODEL_FAST = _get("ollama_model_fast")
OLLAMA_MODEL = OLLAMA_MODEL_FAST
OLLAMA_REQUEST_TIMEOUT_SEC = int(_get("ollama_request_timeout_sec"))
OLLAMA_VISION_MAX_EDGE = int(_get("ollama_vision_max_edge"))
OUTPUT_LANGUAGE = _get("output_language")

# --- LTX-AV (video generation) ---
LTX_FPS = 24

# --- Scene duration limits (seconds) ---
SCENE_MIN_SEC = 2
SCENE_MAX_SEC = 30
SCENE_SWEET_SPOT_SEC = 15

# --- Video workflow node IDs (t2v transformer pipeline) ---
PROMPT_NODE_ID = "153:132"
NEG_PROMPT_NODE_ID = "153:123"
FRAMES_NODE_ID = "153:125"
SEED_NODE_ID_PASS1 = "153:151"
SEED_NODE_ID_PASS2 = "153:127"

VIDEO_RES_NODE_ID = "153:124"   # EmptyImage node that sets video resolution

# --- Keyframe image workflow node IDs (Z-Image Turbo) ---
KF_PROMPT_NODE_ID = "57:27"
KF_SEED_NODE_ID = "57:3"
KF_LATENT_NODE_ID = "57:13"
KF_WIDTH = int(_get("kf_width"))
KF_HEIGHT = int(_get("kf_height"))

# --- Video resolution ---
VIDEO_WIDTH = int(_get("video_width"))
VIDEO_HEIGHT = int(_get("video_height"))

# --- Agent behavior ---
TAKES_PER_SCENE = 3
USE_KEYFRAMES = True
KF_CANDIDATES = 4
LAZY_MODE = False                # AI auto-selects best keyframes and takes (no manual review)
SKIP_KF_EVAL = True              # Skip AI evaluation of keyframes (experimental)
EVAL_FRAME_SAMPLE_RATE = 1
EVAL_MAX_FRAMES = 20
EVAL_TOKEN_BUDGET = 1024
OUTPUT_DIR = "./output"

# --- Negative prompt ---
NEGATIVE_PROMPT = (
    "background music, soundtrack, musical score, "
    "blurry, low quality, still frame, frozen motion, watermark, overlay, "
    "titles, subtitles, text on screen, deformed hands, extra fingers"
)
