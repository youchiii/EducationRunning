from __future__ import annotations

import json
import os
from datetime import datetime
from pathlib import Path
from threading import Lock
from typing import Any, Dict
from uuid import uuid4

from fastapi import APIRouter, Depends, File, HTTPException, UploadFile, status
from fastapi.concurrency import run_in_threadpool

from ..motion_analysis.skeleton_estimation import (
    SkeletonEstimator,
    encode_video,
    render_sequence_frames,
)

from ..config import get_settings
from ..security import get_current_student_user

router = APIRouter(prefix="/pose", tags=["pose"], dependencies=[Depends(get_current_student_user)])

settings = get_settings()
TMP_DIR = settings.pose_tmp_dir
TMP_DIR.mkdir(parents=True, exist_ok=True)

UPLOAD_REGISTRY: Dict[str, Dict[str, Any]] = {}
RESULTS_DB: Dict[str, Dict[str, Any]] = {}
REGISTRY_LOCK = Lock()
RESULTS_LOCK = Lock()


async def _save_upload_file(upload: UploadFile, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    with destination.open("wb") as buffer:
        while True:
            chunk = await upload.read(1024 * 1024)
            if not chunk:
                break
            buffer.write(chunk)
    await upload.close()


def _pose_sequence_to_dict(sequence) -> Dict[str, Any]:
    return {
        "sequence_id": sequence.sequence_id,
        "width": sequence.width,
        "height": sequence.height,
        "frame_count": sequence.frame_count,
        "source_path": os.path.basename(sequence.source_path),
        "frames": [
            {"frame_id": frame.frame_id, "landmarks": frame.landmarks}
            for frame in sequence.frames
        ],
    }


def _write_json(path: Path, payload: Dict[str, Any]) -> None:
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def _analyze_session(request: Dict[str, Any], paths: Dict[str, Any]) -> Dict[str, Any]:
    session_dir: Path = paths["session_dir"]
    estimator = SkeletonEstimator(
        model_complexity=request.get("model_complexity", 1),
        min_detection_confidence=request.get("min_detection_confidence", 0.5),
        min_tracking_confidence=request.get("min_tracking_confidence", 0.5),
    )

    reference_sequence = estimator.extract_landmarks(str(paths["reference"]))
    comparison_sequence = estimator.extract_landmarks(str(paths["comparison"]))
    metrics = estimator.compare_sequences(reference_sequence, comparison_sequence)

    canvas_width = max(reference_sequence.width, comparison_sequence.width)
    canvas_height = max(reference_sequence.height, comparison_sequence.height)

    reference_frames, anchor = render_sequence_frames(
        reference_sequence,
        canvas_size=(canvas_width, canvas_height),
    )
    comparison_frames, _ = render_sequence_frames(
        comparison_sequence,
        align_to=anchor,
        canvas_size=(canvas_width, canvas_height),
    )

    reference_video_bytes = encode_video(reference_frames, width=canvas_width, height=canvas_height)
    comparison_video_bytes = encode_video(comparison_frames, width=canvas_width, height=canvas_height)

    preview_reference_path = session_dir / "preview_reference.mp4"
    preview_comparison_path = session_dir / "preview_comparison.mp4"
    preview_reference_path.write_bytes(reference_video_bytes)
    preview_comparison_path.write_bytes(comparison_video_bytes)

    reference_landmarks_path = session_dir / "reference_landmarks.json"
    comparison_landmarks_path = session_dir / "comparison_landmarks.json"
    results_json_path = session_dir / "results.json"

    _write_json(reference_landmarks_path, _pose_sequence_to_dict(reference_sequence))
    _write_json(comparison_landmarks_path, _pose_sequence_to_dict(comparison_sequence))
    _write_json(
        results_json_path,
        {
            "session_id": request["session_id"],
            "metrics": metrics,
            "generated_at": datetime.utcnow().isoformat() + "Z",
            "settings": {
                "model_complexity": request.get("model_complexity", 1),
                "min_detection_confidence": request.get("min_detection_confidence", 0.5),
                "min_tracking_confidence": request.get("min_tracking_confidence", 0.5),
            },
        },
    )

    session_fragment = f"/tmp/{request['session_id']}"
    payload = {
        "session_id": request["session_id"],
        "metrics": metrics,
        "analysis_settings": request,
        "preview_videos": {
            "reference": f"{session_fragment}/{preview_reference_path.name}",
            "comparison": f"{session_fragment}/{preview_comparison_path.name}",
        },
        "downloads": {
            "metrics": f"{session_fragment}/{results_json_path.name}",
            "reference_landmarks": f"{session_fragment}/{reference_landmarks_path.name}",
            "comparison_landmarks": f"{session_fragment}/{comparison_landmarks_path.name}",
        },
        "source_videos": {
            "reference": f"{session_fragment}/{Path(paths['reference']).name}",
            "comparison": f"{session_fragment}/{Path(paths['comparison']).name}",
        },
        "original_filenames": {
            "reference": paths.get("reference_original_name"),
            "comparison": paths.get("comparison_original_name"),
        },
        "uploaded_at": paths.get("uploaded_at"),
        "updated_at": datetime.utcnow().isoformat() + "Z",
    }

    with RESULTS_LOCK:
        RESULTS_DB[request["session_id"]] = payload
    return payload


@router.post("/upload")
async def upload_videos(
    reference_video: UploadFile = File(...),
    comparison_video: UploadFile = File(...),
) -> Dict[str, Any]:
    session_id = uuid4().hex
    session_dir = TMP_DIR / session_id
    session_dir.mkdir(parents=True, exist_ok=True)

    timestamp = datetime.utcnow().isoformat() + "Z"
    reference_suffix = Path(reference_video.filename or "").suffix or ".mp4"
    comparison_suffix = Path(comparison_video.filename or "").suffix or ".mp4"

    reference_path = session_dir / f"reference{reference_suffix}"
    comparison_path = session_dir / f"comparison{comparison_suffix}"

    await _save_upload_file(reference_video, reference_path)
    await _save_upload_file(comparison_video, comparison_path)

    with REGISTRY_LOCK:
        UPLOAD_REGISTRY[session_id] = {
            "reference": reference_path,
            "comparison": comparison_path,
            "session_dir": session_dir,
            "reference_original_name": reference_video.filename,
            "comparison_original_name": comparison_video.filename,
            "uploaded_at": timestamp,
        }

    with RESULTS_LOCK:
        RESULTS_DB.pop(session_id, None)

    return {
        "session_id": session_id,
        "reference_video": reference_video.filename,
        "comparison_video": comparison_video.filename,
        "uploaded_at": timestamp,
    }


@router.post("/analyze")
async def analyze(request: Dict[str, Any]) -> Dict[str, Any]:
    session_id = request.get("session_id")
    if not session_id:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="session_id is required")

    with REGISTRY_LOCK:
        session_paths = UPLOAD_REGISTRY.get(session_id)

    if not session_paths:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Session not found. Upload videos first.")

    request_payload = {
        "session_id": session_id,
        "model_complexity": request.get("model_complexity", 1),
        "min_detection_confidence": request.get("min_detection_confidence", 0.5),
        "min_tracking_confidence": request.get("min_tracking_confidence", 0.5),
    }

    try:
        result = await run_in_threadpool(_analyze_session, request_payload, session_paths.copy())
    except FileNotFoundError as exc:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=str(exc)) from exc
    except ValueError as exc:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(exc)) from exc
    except Exception as exc:  # pragma: no cover - defensive catch
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail="Analysis failed") from exc

    return result


@router.get("/results")
async def get_results(session_id: str) -> Dict[str, Any]:
    with RESULTS_LOCK:
        result = RESULTS_DB.get(session_id)

    if not result:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Results not found for session")

    return result
