from __future__ import annotations

import json
import mimetypes
import re
import shutil
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional
from uuid import uuid4

from fastapi import File, Form, UploadFile
from fastapi.responses import FileResponse, JSONResponse
from pydantic import BaseModel

from api_server import CHUNK_SIZE_BYTES, ONLINE_SESSIONS_DIR, PROJECT_ROOT, app
from online_pipeline.stream_timeline import append_timeline_event
from online_preprocess.io_utils import read_json, utc_now_iso, write_json_atomic, write_status
from online_preprocess.task_queue import enqueue_preprocess_task

_SESSION_RE = re.compile(r"^[A-Za-z0-9_.-]{1,96}$")
_VIDEO_SUFFIXES = {".mp4", ".mov", ".m4v", ".webm", ".mkv", ".avi"}


class DemoPlaybackRequest(BaseModel):
    current_time: float = 0.0
    paused: bool = False
    playback_speed: float = 1.0


def _safe_session_id(session_id: str) -> bool:
    return bool(_SESSION_RE.match(session_id or "")) and "/" not in session_id and ".." not in session_id


def _session_dir(session_id: str) -> Path:
    return ONLINE_SESSIONS_DIR / session_id


def _demo_dir(session_dir: Path) -> Path:
    path = session_dir / "demo"
    path.mkdir(parents=True, exist_ok=True)
    return path


def _metadata_path(session_dir: Path) -> Path:
    return _demo_dir(session_dir) / "manifest.json"


def _playback_path(session_dir: Path) -> Path:
    return _demo_dir(session_dir) / "playback.json"


def _video_path(session_dir: Path) -> Path:
    metadata = read_json(_metadata_path(session_dir), default={})
    if isinstance(metadata, dict) and metadata.get("video_path"):
        candidate = Path(str(metadata["video_path"]))
        if candidate.exists():
            return candidate
    return session_dir / "input.mp4"


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _parse_metadata(raw: str | None) -> dict[str, Any]:
    if not raw:
        return {}
    try:
        parsed = json.loads(raw)
    except Exception:
        return {"raw_metadata": raw}
    return parsed if isinstance(parsed, dict) else {"metadata": parsed}


def _probe_video(path: Path) -> dict[str, Any]:
    info: dict[str, Any] = {"duration": None, "fps": None, "frame_count": None, "width": None, "height": None}
    try:
        import cv2  # type: ignore
    except Exception as exc:
        info["probe_error"] = f"opencv unavailable: {exc}"
        return info

    capture = cv2.VideoCapture(str(path))
    try:
        if not capture.isOpened():
            info["probe_error"] = "opencv could not open uploaded video"
            return info
        fps = float(capture.get(cv2.CAP_PROP_FPS) or 0.0)
        frame_count = int(capture.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
        width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
        height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)
        duration = frame_count / fps if fps > 0 and frame_count > 0 else None
        info.update({
            "duration": round(duration, 3) if duration is not None else None,
            "fps": round(fps, 3) if fps > 0 else None,
            "frame_count": frame_count if frame_count > 0 else None,
            "width": width if width > 0 else None,
            "height": height if height > 0 else None,
        })
        return info
    finally:
        capture.release()


def _write_stream_state(session_dir: Path, *, status: str, current_time: float, playback_speed: float) -> dict[str, Any]:
    stream_dir = session_dir / "stream"
    stream_dir.mkdir(parents=True, exist_ok=True)
    state_path = stream_dir / "stream_state.json"
    state = read_json(state_path, default={})
    if not isinstance(state, dict):
        state = {}
    state.update({
        "session_id": session_dir.name,
        "stream_id": state.get("stream_id") or f"demo_{session_dir.name}",
        "input_mode": "demo_video",
        "status": status,
        "demo": True,
        "current_time": round(max(0.0, float(current_time)), 3),
        "playback_speed": float(playback_speed or 1.0),
        "updated_at": _now(),
    })
    state.setdefault("started_at", _now())
    write_json_atomic(state_path, state)
    return state


def _write_playback(session_dir: Path, *, status: str, current_time: float, paused: bool, playback_speed: float, extra: dict[str, Any] | None = None) -> dict[str, Any]:
    payload = {
        "status": status,
        "session_id": session_dir.name,
        "current_time": round(max(0.0, float(current_time)), 3),
        "paused": bool(paused),
        "playback_speed": float(playback_speed or 1.0),
        "updated_at": _now(),
    }
    if extra:
        payload.update(extra)
    write_json_atomic(_playback_path(session_dir), payload)
    _write_stream_state(session_dir, status=status, current_time=current_time, playback_speed=playback_speed)
    return payload


def _extract_current_frame(session_dir: Path, current_time: float) -> dict[str, Any]:
    video_path = _video_path(session_dir)
    if not video_path.exists():
        raise FileNotFoundError(f"demo video not found: {video_path}")

    try:
        import cv2  # type: ignore
    except Exception as exc:
        raise RuntimeError(f"opencv-python is required for demo tick frame extraction: {exc}") from exc

    current_time = max(0.0, float(current_time or 0.0))
    capture = cv2.VideoCapture(str(video_path))
    try:
        if not capture.isOpened():
            raise RuntimeError(f"opencv could not open demo video: {video_path}")
        capture.set(cv2.CAP_PROP_POS_MSEC, current_time * 1000.0)
        ok, frame = capture.read()
        if not ok or frame is None:
            fps = float(capture.get(cv2.CAP_PROP_FPS) or 0.0)
            if fps > 0:
                capture.set(cv2.CAP_PROP_POS_FRAMES, max(0, int(current_time * fps)))
                ok, frame = capture.read()
        if not ok or frame is None:
            raise RuntimeError(f"could not extract frame at {current_time:.3f}s")
    finally:
        capture.release()

    frames_dir = session_dir / "stream" / "frames"
    frames_dir.mkdir(parents=True, exist_ok=True)
    frame_index = int(round(current_time * 1000.0))
    frame_path = frames_dir / f"demo_frame_{frame_index:09d}.jpg"
    if not cv2.imwrite(str(frame_path), frame):
        raise RuntimeError(f"failed to write extracted frame: {frame_path}")

    rel_path = frame_path.relative_to(session_dir).as_posix()
    from online_current.mcur_store import MCurStore

    state = MCurStore(session_dir).update_from_frame_stream(
        frame_index=frame_index,
        frame_path=rel_path,
        relative_ts_ms=frame_index,
        client_ts_ms=int(datetime.now(timezone.utc).timestamp() * 1000),
        source="demo_video",
    )
    return {
        "frame_index": frame_index,
        "frame_path": rel_path,
        "current_frame_path": state.get("current_frame_path"),
        "mcur_ready": bool(state.get("mcur_ready")),
        "mcur_version": state.get("mcur_version"),
        "frame_count": state.get("frame_count"),
    }


def _demo_response(session_id: str, *, status: str = "uploaded", extra: dict[str, Any] | None = None) -> dict[str, Any]:
    session_dir = _session_dir(session_id)
    metadata = read_json(_metadata_path(session_dir), default={})
    playback = read_json(_playback_path(session_dir), default={})
    if not isinstance(metadata, dict):
        metadata = {}
    if not isinstance(playback, dict):
        playback = {}
    payload = {
        "status": status,
        "session_id": session_id,
        "video_url": f"/demo/{session_id}/video",
        "start_url": f"/demo/{session_id}/start",
        "tick_url": f"/demo/{session_id}/tick",
        "ask_stream_url": f"/ask/{session_id}/stream",
        "ask_url": f"/ask/{session_id}",
        "manifest_url": f"/demo/{session_id}/manifest",
        "prepared": bool(metadata.get("prepared", True)),
        "duration": metadata.get("duration"),
        "frame_count": metadata.get("demo_frame_count") or metadata.get("frame_count"),
        "width": metadata.get("width"),
        "height": metadata.get("height"),
        "playback": playback,
        "preprocess_queued": bool(metadata.get("preprocess_queued", False)),
    }
    if extra:
        payload.update(extra)
    return payload


@app.post("/demo/upload")
async def demo_upload(
    video: Optional[UploadFile] = File(default=None),
    sample_fps: float = Form(default=1.0),
    auto_prepare: bool = Form(default=True),
    enqueue_preprocess: bool = Form(default=False),
    owner_id: Optional[str] = Form(default=None),
    device_id: Optional[str] = Form(default=None),
    device_type: Optional[str] = Form(default=None),
    metadata: Optional[str] = Form(default=None),
    force_preprocess: bool = Form(default=False),
) -> JSONResponse:
    if video is None:
        return JSONResponse(status_code=400, content={"status": "error", "message": "No video uploaded. Expected form field video."})

    session_id = f"demo_{uuid4().hex[:12]}"
    session_dir = _session_dir(session_id)
    video_path = session_dir / "input.mp4"
    size_bytes = 0
    try:
        session_dir.mkdir(parents=True, exist_ok=False)
        with video_path.open("wb") as output_file:
            while True:
                chunk = await video.read(CHUNK_SIZE_BYTES)
                if not chunk:
                    break
                size_bytes += len(chunk)
                output_file.write(chunk)
        if size_bytes <= 0:
            shutil.rmtree(session_dir, ignore_errors=True)
            return JSONResponse(status_code=400, content={"status": "error", "message": "Uploaded video is empty."})

        suffix = Path(video.filename or "").suffix.lower()
        if suffix and suffix not in _VIDEO_SUFFIXES:
            suffix = ""
        content_type = video.content_type or mimetypes.guess_type(video.filename or "")[0] or "video/mp4"
        probe = _probe_video(video_path) if auto_prepare else {}
        duration = probe.get("duration")
        estimated_demo_frames = None
        if duration is not None:
            estimated_demo_frames = int(max(1.0, float(duration) * max(0.1, float(sample_fps or 1.0))))
        manifest = {
            "session_id": session_id,
            "status": "uploaded",
            "prepared": True,
            "video_path": str(video_path),
            "original_filename": video.filename,
            "content_type": content_type,
            "suffix": suffix or ".mp4",
            "size_bytes": size_bytes,
            "sample_fps": float(sample_fps or 1.0),
            "demo_frame_count": estimated_demo_frames,
            "upload_time": _now(),
            "owner_id": owner_id,
            "device_id": device_id,
            "device_type": device_type,
            "client_metadata": _parse_metadata(metadata),
            **probe,
        }
        preprocess_task = None
        if enqueue_preprocess:
            preprocess_task = enqueue_preprocess_task(PROJECT_ROOT, session_id, force=force_preprocess)
            manifest["preprocess_queued"] = True
            manifest["preprocess_task_path"] = str(preprocess_task)
        else:
            manifest["preprocess_queued"] = False
        write_json_atomic(_metadata_path(session_dir), manifest)
        _write_playback(session_dir, status="uploaded", current_time=0.0, paused=True, playback_speed=1.0)
        write_status(
            session_dir=session_dir,
            session_id=session_id,
            status="uploaded",
            stage="demo_uploaded",
            progress=100,
            error=None,
        )
        append_timeline_event(session_dir, "demo_uploaded", metadata={"size_bytes": size_bytes, "duration": duration})
        return JSONResponse(status_code=200, content=_demo_response(session_id, status="uploaded"))
    except Exception as exc:
        shutil.rmtree(session_dir, ignore_errors=True)
        return JSONResponse(status_code=500, content={"status": "error", "message": str(exc)})
    finally:
        await video.close()


@app.get("/demo/{session_id}/video")
async def demo_video(session_id: str):
    if not _safe_session_id(session_id):
        return JSONResponse(status_code=400, content={"status": "error", "message": "invalid session_id"})
    session_dir = _session_dir(session_id)
    video_path = _video_path(session_dir)
    if not video_path.exists():
        return JSONResponse(status_code=404, content={"status": "error", "message": "demo video not found"})
    metadata = read_json(_metadata_path(session_dir), default={})
    media_type = "video/mp4"
    if isinstance(metadata, dict):
        media_type = str(metadata.get("content_type") or media_type)
    # Keep the response inline so <video> can stream it instead of downloading it.
    return FileResponse(video_path, media_type=media_type)


@app.get("/demo/{session_id}/manifest")
async def demo_manifest(session_id: str) -> JSONResponse:
    if not _safe_session_id(session_id):
        return JSONResponse(status_code=400, content={"status": "error", "message": "invalid session_id"})
    session_dir = _session_dir(session_id)
    metadata = read_json(_metadata_path(session_dir), default={})
    if not session_dir.exists() or not isinstance(metadata, dict):
        return JSONResponse(status_code=404, content={"status": "error", "message": "demo session not found"})
    return JSONResponse(status_code=200, content=_demo_response(session_id, status=str(metadata.get("status") or "uploaded"), extra={"manifest": metadata}))


@app.get("/demo/{session_id}/status")
async def demo_status(session_id: str) -> JSONResponse:
    if not _safe_session_id(session_id):
        return JSONResponse(status_code=400, content={"status": "error", "message": "invalid session_id"})
    session_dir = _session_dir(session_id)
    if not session_dir.exists():
        return JSONResponse(status_code=404, content={"status": "error", "message": "demo session not found"})
    playback = read_json(_playback_path(session_dir), default={})
    current = read_json(session_dir / "current" / "current_state.json", default={})
    return JSONResponse(status_code=200, content=_demo_response(
        session_id,
        status=str((playback or {}).get("status") or "uploaded"),
        extra={"current": current if isinstance(current, dict) else {}},
    ))


@app.post("/demo/{session_id}/start")
async def demo_start(session_id: str, request: DemoPlaybackRequest) -> JSONResponse:
    if not _safe_session_id(session_id):
        return JSONResponse(status_code=400, content={"status": "error", "message": "invalid session_id"})
    session_dir = _session_dir(session_id)
    if not session_dir.exists():
        return JSONResponse(status_code=404, content={"status": "error", "message": "demo session not found"})
    try:
        frame = _extract_current_frame(session_dir, request.current_time)
        playback = _write_playback(session_dir, status="running", current_time=request.current_time, paused=False, playback_speed=request.playback_speed, extra=frame)
        write_status(session_dir=session_dir, session_id=session_id, status="streaming", stage="demo_running", progress=100, error=None)
        append_timeline_event(session_dir, "demo_started", metadata={"current_time": request.current_time})
        return JSONResponse(status_code=200, content=_demo_response(session_id, status="running", extra={**playback, **frame, "can_ask": True}))
    except Exception as exc:
        return JSONResponse(status_code=500, content={"status": "error", "message": str(exc)})


@app.post("/demo/{session_id}/tick")
async def demo_tick(session_id: str, request: DemoPlaybackRequest) -> JSONResponse:
    if not _safe_session_id(session_id):
        return JSONResponse(status_code=400, content={"status": "error", "message": "invalid session_id"})
    session_dir = _session_dir(session_id)
    if not session_dir.exists():
        return JSONResponse(status_code=404, content={"status": "error", "message": "demo session not found"})
    try:
        frame = _extract_current_frame(session_dir, request.current_time)
        status = "paused" if request.paused else "running"
        playback = _write_playback(session_dir, status=status, current_time=request.current_time, paused=request.paused, playback_speed=request.playback_speed, extra=frame)
        return JSONResponse(status_code=200, content=_demo_response(session_id, status=status, extra={**playback, **frame, "can_ask": True}))
    except Exception as exc:
        return JSONResponse(status_code=500, content={"status": "error", "message": str(exc)})


@app.post("/demo/{session_id}/pause")
async def demo_pause(session_id: str, request: DemoPlaybackRequest) -> JSONResponse:
    if not _safe_session_id(session_id):
        return JSONResponse(status_code=400, content={"status": "error", "message": "invalid session_id"})
    session_dir = _session_dir(session_id)
    if not session_dir.exists():
        return JSONResponse(status_code=404, content={"status": "error", "message": "demo session not found"})
    playback = _write_playback(session_dir, status="paused", current_time=request.current_time, paused=True, playback_speed=request.playback_speed)
    return JSONResponse(status_code=200, content=_demo_response(session_id, status="paused", extra=playback))


@app.post("/demo/{session_id}/stop")
async def demo_stop(session_id: str, request: DemoPlaybackRequest) -> JSONResponse:
    if not _safe_session_id(session_id):
        return JSONResponse(status_code=400, content={"status": "error", "message": "invalid session_id"})
    session_dir = _session_dir(session_id)
    if not session_dir.exists():
        return JSONResponse(status_code=404, content={"status": "error", "message": "demo session not found"})
    playback = _write_playback(session_dir, status="stopped", current_time=request.current_time, paused=True, playback_speed=request.playback_speed)
    write_status(session_dir=session_dir, session_id=session_id, status="stopped", stage="demo_stopped", progress=100, error=None)
    append_timeline_event(session_dir, "demo_stopped", metadata={"current_time": request.current_time})
    return JSONResponse(status_code=200, content=_demo_response(session_id, status="stopped", extra=playback))
