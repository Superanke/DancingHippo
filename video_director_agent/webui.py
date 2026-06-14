#!/usr/bin/env python3
"""Browser-based production UI for DancingHippo."""

from __future__ import annotations

import argparse
import base64
import binascii
import json
import logging
import mimetypes
import os
import platform
import re
import subprocess
import sys
import threading
import time
import traceback
import urllib.parse
import uuid
from dataclasses import dataclass, field
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import agent as agent_mod
import assembler
import comfyui_client
import config
import director
import keyframe_gen
import ollama_client
from agent import load_state, preflight, run
from comfyui_client import ComfyUIClient


APP_DIR = Path(__file__).resolve().parent
ROOT_DIR = APP_DIR.parent
STATIC_DIR = APP_DIR / "web_static"
OUTPUT_DIR = APP_DIR / "output"
PROJECT_REFERENCE_KINDS = {"character", "style", "location", "prop"}
REFERENCE_MIME_EXTENSIONS = {
    "image/png": ".png",
    "image/jpeg": ".jpg",
    "image/webp": ".webp",
}
MAX_REFERENCE_BYTES = 20 * 1024 * 1024


@dataclass
class WebJob:
    """In-memory status for the currently running production job."""

    project_name: str
    started_at: float
    status: str
    logs: list[str] = field(default_factory=list)
    action: dict[str, Any] | None = None
    error: str | None = None
    finished_at: float | None = None


@dataclass
class WebAppState:
    """Shared state for the threaded HTTP server."""

    active_job: WebJob | None = None
    lock: threading.Lock = field(default_factory=threading.Lock)


class JobLogHandler(logging.Handler):
    """Send backend log records to the browser status panel."""

    def __init__(self, state: WebAppState) -> None:
        super().__init__()
        self.state = state

    def emit(self, record: logging.LogRecord) -> None:
        if record.name.startswith(("webui.http", "httpx")):
            return
        message = self.format(record)
        with self.state.lock:
            if self.state.active_job is not None:
                self.state.active_job.logs.append(message)
                self.state.active_job.logs = self.state.active_job.logs[-300:]


def _json_load(path: Path) -> dict[str, Any]:
    with path.open(encoding="utf-8") as f:
        return json.load(f)


def _json_save(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)


def _project_state_path(project_name: str) -> Path:
    return OUTPUT_DIR / project_name / "state.json"


def _safe_project_name(raw_name: str) -> str:
    cleaned = raw_name.strip()
    if not cleaned:
        raise ValueError("Project name is required.")
    if "/" in cleaned or "\\" in cleaned or cleaned in (".", ".."):
        raise ValueError(f"Invalid project name: {cleaned}")
    return cleaned


def _media_url(path_value: str) -> str:
    return "/media?path=" + urllib.parse.quote(path_value)


def _file_payload(path_value: str | None) -> dict[str, Any] | None:
    if not path_value:
        return None
    path = Path(path_value)
    if not path.exists():
        return {"path": path_value, "exists": False, "url": None, "size": 0}
    return {
        "path": path_value,
        "exists": True,
        "url": _media_url(path_value),
        "size": path.stat().st_size,
    }


def _character_reference_payloads(state: dict[str, Any]) -> dict[str, Any]:
    references = {}
    for character_id, reference in state.get("character_references", {}).items():
        if not isinstance(reference, dict):
            continue
        item = dict(reference)
        item["file"] = _file_payload(reference.get("path"))
        references[character_id] = item
    return references


def _project_reference_payloads(state: dict[str, Any]) -> list[dict[str, Any]]:
    references = []
    for reference in state.get("project_references", []):
        if not isinstance(reference, dict):
            continue
        item = dict(reference)
        item["file"] = _file_payload(reference.get("path"))
        references.append(item)
    return references


def _scene_payload(scene: dict[str, Any]) -> dict[str, Any]:
    display_scene = {
        key: value
        for key, value in scene.items()
        if key not in {"keyframe_prompt", "ltx_prompt", "negative_prompt"}
    }
    candidates = []
    for candidate in scene.get("keyframe_candidates", []):
        item = dict(candidate)
        item["file"] = _file_payload(candidate.get("path"))
        candidates.append(item)
    takes = []
    for take in scene.get("takes", []):
        item = dict(take)
        item["file"] = _file_payload(take.get("path"))
        takes.append(item)
    selected_keyframe = _file_payload(scene.get("selected_keyframe"))
    selected_take = _file_payload(scene.get("selected_take"))
    return {
        **display_scene,
        "keyframe_candidates": candidates,
        "takes": takes,
        "selected_keyframe_file": selected_keyframe,
        "selected_take_file": selected_take,
    }


def _project_phase(state: dict[str, Any]) -> str:
    if state.get("completed_at"):
        return "complete"
    if state.get("last_error"):
        return "error"
    scenes = state.get("scenes", [])
    if not scenes:
        return "planning"
    if not state.get("storyboard_approved"):
        has_all_keyframes = all(scene.get("keyframe_candidates") for scene in scenes)
        return "review_storyboard" if has_all_keyframes else "generating_storyboard"
    if not state.get("generation_completed_at"):
        return "ready_for_video"
    has_all_takes = all(scene.get("takes_done") for scene in scenes)
    return "review_takes" if has_all_takes else "generating_video"


def _project_payload(project_name: str) -> dict[str, Any]:
    state_path = _project_state_path(project_name)
    if not state_path.exists():
        return {"name": project_name, "exists": False, "state": None}
    state = _json_load(state_path)
    scenes = [_scene_payload(scene) for scene in state.get("scenes", [])]
    final_file = _file_payload(state.get("final_path"))
    return {
        "name": project_name,
        "exists": True,
        "phase": _project_phase(state),
        "state": {
            **state,
            "scenes": scenes,
            "final_file": final_file,
            "character_references": _character_reference_payloads(state),
            "project_references": _project_reference_payloads(state),
        },
        "updated_at": state_path.stat().st_mtime,
    }


def _project_summary(project_name: str) -> dict[str, Any]:
    state_path = _project_state_path(project_name)
    if not state_path.exists():
        return {"name": project_name, "exists": False}
    state = _json_load(state_path)
    scenes = state.get("scenes", [])
    keyframe_count = sum(1 for scene in scenes if scene.get("keyframe_candidates"))
    take_count = sum(1 for scene in scenes if scene.get("takes"))
    final_path = state.get("final_path")
    return {
        "name": project_name,
        "exists": True,
        "phase": _project_phase(state),
        "scene_count": len(scenes),
        "keyframe_count": keyframe_count,
        "take_count": take_count,
        "reference_count": len(state.get("project_references", [])),
        "final_exists": bool(final_path and Path(final_path).exists()),
        "updated_at": state_path.stat().st_mtime,
    }


def _list_projects() -> list[dict[str, Any]]:
    if not OUTPUT_DIR.exists():
        return []
    projects = []
    for state_path in OUTPUT_DIR.glob("*/state.json"):
        projects.append(_project_summary(state_path.parent.name))
    return sorted(projects, key=lambda item: item.get("updated_at", 0), reverse=True)


def _health_payload() -> dict[str, Any]:
    comfy_client = ComfyUIClient()
    comfy_ok = comfy_client.check_alive()
    try:
        model_names = [model.model for model in ollama_client.list_models().models]
        ollama_ok = config.OLLAMA_MODEL in model_names
    except Exception:
        model_names = []
        ollama_ok = False
    return {
        "comfyui": {
            "ok": comfy_ok,
            "host": config.COMFYUI_HOST,
        },
        "ollama": {
            "ok": ollama_ok,
            "host": config.OLLAMA_HOST,
            "model": config.OLLAMA_MODEL,
            "models": model_names,
        },
        "output_language": config.OUTPUT_LANGUAGE,
    }


def _apply_runtime_settings(settings: dict[str, Any]) -> None:
    takes = int(settings["takes_per_scene"])
    scene_min = int(settings["scene_min"])
    scene_max = int(settings["scene_max"])
    use_keyframes = bool(settings["use_keyframes"])
    skip_keyframe_eval = bool(settings["skip_keyframe_eval"])
    keyframe_width = int(settings["keyframe_width"])
    keyframe_height = int(settings["keyframe_height"])
    video_width = int(settings["video_width"])
    video_height = int(settings["video_height"])
    output_language = str(settings["output_language"])
    if output_language not in {"english", "simplified_chinese"}:
        raise ValueError(f"Unsupported final language: {output_language}")

    config.TAKES_PER_SCENE = takes
    agent_mod.TAKES_PER_SCENE = takes
    config.SCENE_MIN_SEC = scene_min
    config.SCENE_MAX_SEC = scene_max
    director.SCENE_MIN_SEC = scene_min
    director.SCENE_MAX_SEC = scene_max
    config.USE_KEYFRAMES = use_keyframes
    agent_mod.USE_KEYFRAMES = use_keyframes
    config.SKIP_KF_EVAL = skip_keyframe_eval
    keyframe_gen.SKIP_KF_EVAL = skip_keyframe_eval
    config.KF_WIDTH = keyframe_width
    config.KF_HEIGHT = keyframe_height
    keyframe_gen.KF_WIDTH = keyframe_width
    keyframe_gen.KF_HEIGHT = keyframe_height
    config.VIDEO_WIDTH = video_width
    config.VIDEO_HEIGHT = video_height
    comfyui_client.VIDEO_WIDTH = video_width
    comfyui_client.VIDEO_HEIGHT = video_height
    config.OUTPUT_LANGUAGE = output_language
    director.OUTPUT_LANGUAGE = output_language


def _job_settings(payload: dict[str, Any]) -> dict[str, Any]:
    return {
        "takes_per_scene": int(payload["takes_per_scene"]),
        "scene_min": int(payload["scene_min"]),
        "scene_max": int(payload["scene_max"]),
        "use_keyframes": bool(payload["use_keyframes"]),
        "skip_keyframe_eval": bool(payload["skip_keyframe_eval"]),
        "keyframe_width": int(payload["keyframe_width"]),
        "keyframe_height": int(payload["keyframe_height"]),
        "video_width": int(payload["video_width"]),
        "video_height": int(payload["video_height"]),
        "output_language": str(payload["output_language"]),
    }


def _save_project_output_language(project_name: str, output_language: str) -> None:
    state_path = _project_state_path(project_name)
    if not state_path.exists():
        return
    state = _json_load(state_path)
    previous_language = state.get("output_language")
    if previous_language and previous_language != output_language:
        for scene in state.get("scenes", []):
            scene.pop("ltx_prompt", None)
    state["output_language"] = output_language
    _json_save(state_path, state)


def _ensure_project_shell(
    project_name: str,
    brief: str,
    is_script: bool,
    output_language: str,
) -> None:
    state_path = _project_state_path(project_name)
    if state_path.exists():
        return
    state = {
        "project_name": project_name,
        "brief": brief,
        "created_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "total_scenes": 0,
        "scenes": [],
        "is_script": is_script,
        "output_language": output_language,
        "planning_started_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
    }
    _json_save(state_path, state)


def _clear_project_error(project_name: str) -> None:
    state_path = _project_state_path(project_name)
    if not state_path.exists():
        return
    state = _json_load(state_path)
    if state.pop("last_error", None) is not None:
        _json_save(state_path, state)


def _record_project_error(project_name: str, exc: Exception, traceback_text: str) -> None:
    state_path = _project_state_path(project_name)
    state = _json_load(state_path) if state_path.exists() else {
        "project_name": project_name,
        "brief": "",
        "created_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "total_scenes": 0,
        "scenes": [],
    }
    state["last_error"] = {
        "message": str(exc),
        "traceback": traceback_text,
        "created_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
    }
    _json_save(state_path, state)


def _run_production_job(
    state: WebAppState,
    project_name: str,
    brief: str,
    is_script: bool,
    lazy: bool,
    settings: dict[str, Any],
) -> None:
    logger = logging.getLogger()
    handler = JobLogHandler(state)
    handler.setFormatter(logging.Formatter("%(asctime)s [%(levelname)s] %(name)s: %(message)s"))
    logger.addHandler(handler)
    logger.setLevel(logging.INFO)
    try:
        with state.lock:
            if state.active_job is not None:
                state.active_job.status = "running"
        _apply_runtime_settings(settings)
        _save_project_output_language(project_name, str(settings["output_language"]))
        _clear_project_error(project_name)
        preflight(ComfyUIClient(), logging.getLogger("webui"))
        run(brief, project_name, logging.getLogger("webui"), is_script=is_script, lazy=lazy)
        _save_project_output_language(project_name, str(settings["output_language"]))
        with state.lock:
            if state.active_job is not None:
                state.active_job.status = "finished"
                state.active_job.finished_at = time.time()
    except Exception as exc:
        traceback_text = traceback.format_exc()
        _record_project_error(project_name, exc, traceback_text)
        with state.lock:
            if state.active_job is not None:
                state.active_job.status = "error"
                state.active_job.error = f"{exc}\n{traceback_text}"
                state.active_job.finished_at = time.time()
    finally:
        logger.removeHandler(handler)


def _start_background_job(
    state: WebAppState,
    project_name: str,
    brief: str,
    is_script: bool,
    lazy: bool,
    settings: dict[str, Any],
    action: dict[str, Any],
) -> None:
    with state.lock:
        if state.active_job is not None and state.active_job.status == "running":
            raise RuntimeError(f"Project '{state.active_job.project_name}' is already running.")
        state.active_job = WebJob(
            project_name=project_name,
            started_at=time.time(),
            status="queued",
            action=action,
        )
    thread = threading.Thread(
        target=_run_production_job,
        args=(state, project_name, brief, is_script, lazy, settings),
        daemon=True,
    )
    thread.start()


def _approve_storyboard(project_name: str) -> dict[str, Any]:
    state_path = _project_state_path(project_name)
    state = _json_load(state_path)
    for scene in state.get("scenes", []):
        candidates = [item for item in scene.get("keyframe_candidates", []) if item.get("status") == "generated"]
        if not candidates:
            raise ValueError(f"Scene {scene.get('scene_number')} has no generated keyframe.")
        passed = [item for item in candidates if item.get("eval", {}).get("verdict") == "PASS"]
        manual_review = [
            item for item in candidates
            if item.get("eval", {}).get("notes") == "Evaluation skipped" or not item.get("eval")
        ]
        if passed:
            selected = passed[0]
        elif manual_review and not any(item.get("eval", {}).get("verdict") == "FAIL" for item in candidates):
            selected = manual_review[0]
            scene["quality_gate"] = "manual_review"
        else:
            raise ValueError(
                f"Scene {scene.get('scene_number')} has no AI-passed keyframe. "
                "Regenerate the storyboard or manually review the failed candidates."
            )
        scene["selected_keyframe"] = selected["path"]
        scene["keyframe_approved"] = True
    state["storyboard_approved"] = True
    _json_save(state_path, state)
    return _project_payload(project_name)


def _mark_scene_for_keyframe_regeneration(project_name: str, scene_number: int, notes: str) -> dict[str, Any]:
    state_path = _project_state_path(project_name)
    state = _json_load(state_path)
    for scene in state.get("scenes", []):
        if int(scene.get("scene_number", 0)) != scene_number:
            continue
        scene["rejection_notes"] = notes.strip() or "Regenerate this storyboard keyframe."
        scene["keyframe_regeneration_requested_at"] = time.strftime("%Y-%m-%dT%H:%M:%S")
        scene.pop("selected_keyframe", None)
        scene.pop("keyframe_approved", None)
        state.pop("storyboard_approved", None)
        _json_save(state_path, state)
        return state
    raise ValueError(f"Scene {scene_number} was not found.")


def _assemble_first_takes(project_name: str) -> dict[str, Any]:
    state_path = _project_state_path(project_name)
    state = _json_load(state_path)
    selected_paths = []
    for scene in state.get("scenes", []):
        takes = [item for item in scene.get("takes", []) if item.get("status") == "generated"]
        if not takes:
            raise ValueError(f"Scene {scene.get('scene_number')} has no generated take.")
        selected = takes[0]
        scene["selected_take"] = selected["path"]
        scene["status"] = "approved"
        selected_paths.append(selected["path"])
    final_path = str(OUTPUT_DIR / project_name / "final.mp4")
    assembler.concat_scenes(selected_paths, final_path)
    state["final_path"] = final_path
    state["completed_at"] = time.strftime("%Y-%m-%dT%H:%M:%S")
    _json_save(state_path, state)
    return _project_payload(project_name)


def _select_take(project_name: str, scene_number: int, take_number: int) -> dict[str, Any]:
    state_path = _project_state_path(project_name)
    state = _json_load(state_path)
    for scene in state.get("scenes", []):
        if int(scene.get("scene_number", 0)) != scene_number:
            continue
        takes = [item for item in scene.get("takes", []) if item.get("status") == "generated"]
        for take in takes:
            if int(take.get("take", 0)) == take_number:
                scene["selected_take"] = take["path"]
                scene["status"] = "approved"
                _json_save(state_path, state)
                return _project_payload(project_name)
        raise ValueError(f"Scene {scene_number} does not have generated take {take_number}.")
    raise ValueError(f"Scene {scene_number} was not found.")


def _require_project_media_path(project_name: str, path_value: str) -> Path:
    path = Path(path_value).resolve()
    project_root = (OUTPUT_DIR / project_name).resolve()
    if project_root not in path.parents and path != project_root:
        raise ValueError("Reference image must be inside this project's output folder.")
    if not path.exists() or not path.is_file():
        raise FileNotFoundError(f"Reference image does not exist: {path}")
    if path.suffix.lower() not in {".png", ".jpg", ".jpeg", ".webp"}:
        raise ValueError("Reference image must be a PNG, JPG, JPEG, or WEBP file.")
    return path


def _set_character_reference(
    project_name: str,
    character_id: str,
    path_value: str,
    source_scene_number: int,
    source_candidate: int,
) -> dict[str, Any]:
    state_path = _project_state_path(project_name)
    state = _json_load(state_path)
    characters = state.get("characters", {})
    if character_id not in characters:
        raise ValueError(f"Unknown character: {character_id}")
    reference_path = _require_project_media_path(project_name, path_value)
    references = state.setdefault("character_references", {})
    references[character_id] = {
        "path": str(reference_path),
        "source": "scene_keyframe",
        "source_scene_number": source_scene_number,
        "source_candidate": source_candidate,
        "created_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
    }
    _json_save(state_path, state)
    return _project_payload(project_name)


def _clear_character_reference(project_name: str, character_id: str) -> dict[str, Any]:
    state_path = _project_state_path(project_name)
    state = _json_load(state_path)
    references = state.setdefault("character_references", {})
    if character_id in references:
        references.pop(character_id)
        _json_save(state_path, state)
    return _project_payload(project_name)


def _safe_reference_label(raw_label: str, fallback: str) -> str:
    label = raw_label.strip()
    if not label:
        return fallback
    return label[:80]


def _safe_reference_filename(raw_filename: str, extension: str) -> str:
    filename = Path(raw_filename).name.strip()
    stem = Path(filename).stem if filename else "reference"
    clean_stem = re.sub(r"[^A-Za-z0-9_.-]+", "_", stem).strip("._")
    if not clean_stem:
        clean_stem = "reference"
    return f"{clean_stem[:60]}{extension}"


def _decode_reference_image(data_url: str) -> tuple[str, bytes]:
    match = re.fullmatch(r"data:(image/(?:png|jpeg|webp));base64,(.+)", data_url, re.DOTALL)
    if not match:
        raise ValueError("Reference upload must be a PNG, JPG, or WEBP data URL.")
    mime_type = match.group(1)
    try:
        image_bytes = base64.b64decode(match.group(2), validate=True)
    except binascii.Error as exc:
        raise ValueError("Reference image data is not valid base64.") from exc
    if len(image_bytes) > MAX_REFERENCE_BYTES:
        raise ValueError("Reference image is larger than 20 MB.")
    if not image_bytes:
        raise ValueError("Reference image is empty.")
    try:
        from PIL import Image as PILImage
        from io import BytesIO

        with PILImage.open(BytesIO(image_bytes)) as image:
            image.verify()
    except Exception as exc:
        raise ValueError("Reference image could not be opened as a valid image.") from exc
    return mime_type, image_bytes


def _load_or_create_project_state(project_name: str, brief: str, is_script: bool) -> dict[str, Any]:
    state_path = _project_state_path(project_name)
    if state_path.exists():
        return _json_load(state_path)
    clean_brief = brief.strip()
    if not clean_brief:
        raise ValueError("Brief is required before adding project references.")
    state = agent_mod.create_state(project_name, clean_brief)
    state["is_script"] = is_script
    _json_save(state_path, state)
    return state


def _add_project_reference(payload: dict[str, Any]) -> dict[str, Any]:
    project_name = _safe_project_name(str(payload["project_name"]))
    reference_kind = str(payload["kind"]).strip().lower()
    if reference_kind not in PROJECT_REFERENCE_KINDS:
        raise ValueError(f"Unsupported reference type: {reference_kind}")
    mime_type, image_bytes = _decode_reference_image(str(payload["data_url"]))
    extension = REFERENCE_MIME_EXTENSIONS[mime_type]
    state = _load_or_create_project_state(
        project_name,
        str(payload.get("brief", "")),
        bool(payload.get("is_script", False)),
    )
    output_language = str(payload.get("output_language", "")).strip()
    if output_language:
        if output_language not in {"english", "simplified_chinese"}:
            raise ValueError(f"Unsupported final language: {output_language}")
        previous_language = state.get("output_language")
        if previous_language and previous_language != output_language:
            for scene in state.get("scenes", []):
                scene.pop("ltx_prompt", None)
        state["output_language"] = output_language
    reference_id = f"ref_{uuid.uuid4().hex[:12]}"
    label = _safe_reference_label(str(payload.get("label", "")), reference_kind.title())
    original_name = Path(str(payload.get("filename", ""))).name
    safe_filename = _safe_reference_filename(original_name, extension)
    stored_filename = f"{reference_id}_{safe_filename}"
    reference_dir = OUTPUT_DIR / project_name / "references"
    reference_dir.mkdir(parents=True, exist_ok=True)
    reference_path = reference_dir / stored_filename
    reference_path.write_bytes(image_bytes)
    references = state.setdefault("project_references", [])
    references.append({
        "id": reference_id,
        "kind": reference_kind,
        "label": label,
        "path": str(reference_path),
        "original_name": original_name,
        "created_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
    })
    _json_save(_project_state_path(project_name), state)
    return _project_payload(project_name)


def _delete_project_reference(project_name: str, reference_id: str) -> dict[str, Any]:
    state_path = _project_state_path(project_name)
    state = _json_load(state_path)
    references = state.setdefault("project_references", [])
    kept_references = []
    deleted_path = None
    for reference in references:
        if not isinstance(reference, dict) or reference.get("id") != reference_id:
            kept_references.append(reference)
            continue
        deleted_path = reference.get("path")
    if len(kept_references) == len(references):
        raise ValueError(f"Project reference not found: {reference_id}")
    if isinstance(deleted_path, str):
        path = Path(deleted_path).resolve()
        project_root = (OUTPUT_DIR / project_name).resolve()
        if (project_root in path.parents or path == project_root) and path.exists():
            path.unlink()
    state["project_references"] = kept_references
    _json_save(state_path, state)
    return _project_payload(project_name)


def _assemble_selected_takes(project_name: str) -> dict[str, Any]:
    state_path = _project_state_path(project_name)
    state = _json_load(state_path)
    selected_paths = []
    for scene in state.get("scenes", []):
        selected_take = scene.get("selected_take")
        if selected_take and Path(selected_take).exists():
            selected_paths.append(selected_take)
            scene["status"] = "approved"
            continue
        takes = [item for item in scene.get("takes", []) if item.get("status") == "generated"]
        if not takes:
            raise ValueError(f"Scene {scene.get('scene_number')} has no generated take.")
        selected = takes[0]
        scene["selected_take"] = selected["path"]
        scene["status"] = "approved"
        selected_paths.append(selected["path"])
    final_path = str(OUTPUT_DIR / project_name / "final.mp4")
    assembler.concat_scenes(selected_paths, final_path)
    state["final_path"] = final_path
    state["completed_at"] = time.strftime("%Y-%m-%dT%H:%M:%S")
    _json_save(state_path, state)
    return _project_payload(project_name)


def _open_path(path: Path) -> None:
    if not path.exists():
        raise FileNotFoundError(f"Path does not exist: {path}")
    system = platform.system()
    if system == "Darwin":
        subprocess.Popen(["open", str(path)])
        return
    if system == "Windows":
        os.startfile(str(path))  # type: ignore[attr-defined]
        return
    subprocess.Popen(["xdg-open", str(path)])


def _open_output_folder(project_name: str) -> dict[str, Any]:
    output_path = OUTPUT_DIR / project_name
    _open_path(output_path)
    return {"ok": True, "path": str(output_path)}


def _open_final_video(project_name: str) -> dict[str, Any]:
    state_path = _project_state_path(project_name)
    state = _json_load(state_path)
    final_path = state.get("final_path")
    if not final_path:
        raise ValueError(f"Project '{project_name}' does not have a final film yet.")
    _open_path(Path(final_path))
    return {"ok": True, "path": final_path}


class WebUIHandler(BaseHTTPRequestHandler):
    """HTTP handler for the local browser UI."""

    app_state: WebAppState

    def do_HEAD(self) -> None:
        parsed = urllib.parse.urlparse(self.path)
        if parsed.path == "/":
            self._send_file_headers(STATIC_DIR / "index.html")
            return
        if parsed.path == "/favicon.ico":
            self._send_no_content()
            return
        if parsed.path.startswith("/static/"):
            name = parsed.path.removeprefix("/static/")
            self._send_file_headers(STATIC_DIR / name)
            return
        self._send_json(404, {"error": "Not found"})

    def do_GET(self) -> None:
        parsed = urllib.parse.urlparse(self.path)
        try:
            if parsed.path == "/":
                self._send_file(STATIC_DIR / "index.html")
            elif parsed.path == "/favicon.ico":
                self._send_no_content()
            elif parsed.path.startswith("/static/"):
                name = parsed.path.removeprefix("/static/")
                self._send_file(STATIC_DIR / name)
            elif parsed.path == "/media":
                self._send_media(parsed.query)
            elif parsed.path == "/api/status":
                self._send_json(200, self._status_payload())
            elif parsed.path == "/api/projects":
                self._send_json(200, {"projects": _list_projects()})
            elif parsed.path.startswith("/api/projects/"):
                project_name = urllib.parse.unquote(parsed.path.removeprefix("/api/projects/"))
                self._send_json(200, _project_payload(project_name))
            else:
                self._send_json(404, {"error": "Not found"})
        except Exception as exc:
            self._send_json(500, {"error": str(exc)})

    def do_POST(self) -> None:
        parsed = urllib.parse.urlparse(self.path)
        try:
            payload = self._read_json()
            if parsed.path == "/api/start":
                project_name = _safe_project_name(str(payload["project_name"]))
                existing = load_state(project_name)
                settings = _job_settings(payload)
                if existing is not None:
                    brief = existing["brief"]
                    is_script = bool(existing.get("is_script", False))
                else:
                    brief = str(payload["brief"]).strip()
                    if not brief:
                        raise ValueError("Brief is required for a new project.")
                    is_script = bool(payload["is_script"])
                    _ensure_project_shell(project_name, brief, is_script, str(settings["output_language"]))
                _start_background_job(
                    self.app_state,
                    project_name,
                    brief,
                    is_script,
                    bool(payload["lazy"]),
                    settings,
                    {"type": "production"},
                )
                self._send_json(200, {"ok": True, "project": project_name})
            elif parsed.path == "/api/project-references/upload":
                self._send_json(200, _add_project_reference(payload))
            elif parsed.path.startswith("/api/projects/") and parsed.path.endswith("/regenerate-keyframe"):
                project_name = urllib.parse.unquote(parsed.path.split("/")[3])
                state = _mark_scene_for_keyframe_regeneration(
                    project_name,
                    int(payload["scene_number"]),
                    str(payload.get("notes", "")),
                )
                settings = _job_settings(payload)
                _start_background_job(
                    self.app_state,
                    project_name,
                    state["brief"],
                    bool(state.get("is_script", False)),
                    bool(payload["lazy"]),
                    settings,
                    {
                        "type": "regenerate_keyframe",
                        "scene_number": int(payload["scene_number"]),
                    },
                )
                self._send_json(200, {"ok": True, "project": project_name})
            elif parsed.path.startswith("/api/projects/") and parsed.path.endswith("/approve-storyboard"):
                project_name = urllib.parse.unquote(parsed.path.split("/")[3])
                self._send_json(200, _approve_storyboard(project_name))
            elif parsed.path.startswith("/api/projects/") and parsed.path.endswith("/generate-videos"):
                project_name = urllib.parse.unquote(parsed.path.split("/")[3])
                state = load_state(project_name)
                if state is None:
                    raise ValueError(f"Project not found: {project_name}")
                settings = _job_settings(payload)
                _start_background_job(
                    self.app_state,
                    project_name,
                    state["brief"],
                    bool(state.get("is_script", False)),
                    bool(payload["lazy"]),
                    settings,
                    {"type": "video_generation"},
                )
                self._send_json(200, {"ok": True, "project": project_name})
            elif parsed.path.startswith("/api/projects/") and parsed.path.endswith("/assemble-first-takes"):
                project_name = urllib.parse.unquote(parsed.path.split("/")[3])
                self._send_json(200, _assemble_first_takes(project_name))
            elif parsed.path.startswith("/api/projects/") and parsed.path.endswith("/select-take"):
                project_name = urllib.parse.unquote(parsed.path.split("/")[3])
                self._send_json(
                    200,
                    _select_take(
                        project_name,
                        int(payload["scene_number"]),
                        int(payload["take_number"]),
                    ),
                )
            elif parsed.path.startswith("/api/projects/") and parsed.path.endswith("/set-character-reference"):
                project_name = urllib.parse.unquote(parsed.path.split("/")[3])
                self._send_json(
                    200,
                    _set_character_reference(
                        project_name,
                        str(payload["character_id"]),
                        str(payload["path"]),
                        int(payload["source_scene_number"]),
                        int(payload["source_candidate"]),
                    ),
                )
            elif parsed.path.startswith("/api/projects/") and parsed.path.endswith("/clear-character-reference"):
                project_name = urllib.parse.unquote(parsed.path.split("/")[3])
                self._send_json(
                    200,
                    _clear_character_reference(project_name, str(payload["character_id"])),
                )
            elif parsed.path.startswith("/api/projects/") and parsed.path.endswith("/delete-project-reference"):
                project_name = urllib.parse.unquote(parsed.path.split("/")[3])
                self._send_json(
                    200,
                    _delete_project_reference(project_name, str(payload["reference_id"])),
                )
            elif parsed.path.startswith("/api/projects/") and parsed.path.endswith("/assemble-selected-takes"):
                project_name = urllib.parse.unquote(parsed.path.split("/")[3])
                self._send_json(200, _assemble_selected_takes(project_name))
            elif parsed.path.startswith("/api/projects/") and parsed.path.endswith("/open-output-folder"):
                project_name = urllib.parse.unquote(parsed.path.split("/")[3])
                self._send_json(200, _open_output_folder(project_name))
            elif parsed.path.startswith("/api/projects/") and parsed.path.endswith("/open-final-video"):
                project_name = urllib.parse.unquote(parsed.path.split("/")[3])
                self._send_json(200, _open_final_video(project_name))
            else:
                self._send_json(404, {"error": "Not found"})
        except Exception as exc:
            self._send_json(500, {"error": str(exc)})

    def log_message(self, format_value: str, *args: Any) -> None:
        logging.getLogger("webui.http").info(format_value, *args)

    def _status_payload(self) -> dict[str, Any]:
        with self.app_state.lock:
            job = self.app_state.active_job
            job_payload = None
            if job is not None:
                job_payload = {
                    "project_name": job.project_name,
                    "started_at": job.started_at,
                    "finished_at": job.finished_at,
                    "status": job.status,
                    "action": job.action,
                    "logs": job.logs[-120:],
                    "error": job.error,
                }
        return {
            "health": _health_payload(),
            "job": job_payload,
            "projects": _list_projects(),
        }

    def _read_json(self) -> dict[str, Any]:
        content_length = int(self.headers.get("Content-Length", "0"))
        raw = self.rfile.read(content_length).decode("utf-8")
        if not raw:
            return {}
        return json.loads(raw)

    def _send_json(self, status: int, payload: dict[str, Any]) -> None:
        body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Cache-Control", "no-store")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def _send_no_content(self) -> None:
        self.send_response(204)
        self.send_header("Cache-Control", "no-store")
        self.send_header("Content-Length", "0")
        self.end_headers()

    def _send_file(self, path: Path) -> None:
        if not path.exists() or not path.is_file():
            self._send_json(404, {"error": "File not found"})
            return
        content_type = mimetypes.guess_type(str(path))[0] or "application/octet-stream"
        file_size = path.stat().st_size
        range_header = self.headers.get("Range")
        if range_header:
            byte_range = _parse_range_header(range_header, file_size)
            if byte_range is None:
                self.send_response(416)
                self.send_header("Content-Range", f"bytes */{file_size}")
                self.end_headers()
                return
            start, end = byte_range
            self.send_response(206)
            self.send_header("Content-Type", content_type)
            self.send_header("Cache-Control", "no-store")
            self.send_header("Accept-Ranges", "bytes")
            self.send_header("Content-Range", f"bytes {start}-{end}/{file_size}")
            self.send_header("Content-Length", str(end - start + 1))
            self.end_headers()
            self._write_file_range(path, start, end)
            return

        self.send_response(200)
        self.send_header("Content-Type", content_type)
        self.send_header("Cache-Control", "no-store")
        self.send_header("Accept-Ranges", "bytes")
        self.send_header("Content-Length", str(file_size))
        self.end_headers()
        self._write_file_range(path, 0, file_size - 1)

    def _send_file_headers(self, path: Path) -> None:
        if not path.exists() or not path.is_file():
            self.send_response(404)
            self.send_header("Content-Type", "application/json; charset=utf-8")
            self.end_headers()
            return
        content_type = mimetypes.guess_type(str(path))[0] or "application/octet-stream"
        self.send_response(200)
        self.send_header("Content-Type", content_type)
        self.send_header("Cache-Control", "no-store")
        self.send_header("Accept-Ranges", "bytes")
        self.send_header("Content-Length", str(path.stat().st_size))
        self.end_headers()

    def _write_file_range(self, path: Path, start: int, end: int) -> None:
        remaining = end - start + 1
        with path.open("rb") as file:
            file.seek(start)
            while remaining > 0:
                chunk = file.read(min(1024 * 1024, remaining))
                if not chunk:
                    return
                try:
                    self.wfile.write(chunk)
                except (BrokenPipeError, ConnectionResetError):
                    return
                remaining -= len(chunk)

    def _send_media(self, query: str) -> None:
        params = urllib.parse.parse_qs(query)
        raw_path = params.get("path", [""])[0]
        path = Path(raw_path).resolve()
        output_root = OUTPUT_DIR.resolve()
        if output_root not in path.parents and path != output_root:
            self._send_json(403, {"error": "Media path is outside output directory."})
            return
        self._send_file(path)


def _parse_range_header(range_header: str, file_size: int) -> tuple[int, int] | None:
    if not range_header.startswith("bytes="):
        return None
    value = range_header.removeprefix("bytes=").split(",", 1)[0].strip()
    if "-" not in value:
        return None
    start_text, end_text = value.split("-", 1)
    try:
        if start_text == "":
            suffix_size = int(end_text)
            if suffix_size <= 0:
                return None
            return max(file_size - suffix_size, 0), file_size - 1
        start = int(start_text)
        end = int(end_text) if end_text else file_size - 1
    except ValueError:
        return None
    if start < 0 or end < start or start >= file_size:
        return None
    return start, min(end, file_size - 1)


def run_server(host: str, port: int) -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
    state = WebAppState()

    class BoundHandler(WebUIHandler):
        app_state = state

    server = ThreadingHTTPServer((host, port), BoundHandler)
    logging.getLogger("webui").info("DancingHippo Web UI running at http://%s:%d", host, port)
    server.serve_forever()


def main() -> None:
    parser = argparse.ArgumentParser(description="DancingHippo Web UI")
    parser.add_argument("--host", required=False, default="127.0.0.1")
    parser.add_argument("--port", required=False, type=int, default=8788)
    args = parser.parse_args()
    run_server(str(args.host), int(args.port))


if __name__ == "__main__":
    main()
