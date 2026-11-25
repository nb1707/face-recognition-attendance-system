"""Validated data models for recognition sessions and controlled trials."""

from __future__ import annotations

import math
import sys
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Any

CAMERA_TEST_SESSION = "camera_test"
REAL_CLASS_SESSION = "real_class"
UNKNOWN_EXPECTED_USER_ID = "__unknown__"
SUPPORTED_SESSION_TYPES = {CAMERA_TEST_SESSION, REAL_CLASS_SESSION}


def _required_text(value: str, label: str) -> str:
    normalized = value.strip()
    if not normalized:
        raise ValueError(f"{label} is required.")
    return normalized


def _optional_float(
    value: str,
    label: str,
    *,
    minimum: float | None = None,
) -> float | None:
    normalized = value.strip()
    if not normalized:
        return None
    try:
        number = float(normalized)
    except ValueError as error:
        raise ValueError(f"{label} must be a number.") from error
    if not math.isfinite(number):
        raise ValueError(f"{label} must be a finite number.")
    if minimum is not None and number < minimum:
        raise ValueError(f"{label} must be at least {minimum:g}.")
    return number


def _required_nonnegative_float(value: str, label: str) -> float:
    number = _optional_float(value, label, minimum=0.0)
    if number is None:
        raise ValueError(f"{label} is required.")
    return number


@dataclass(frozen=True)
class RecognitionConfiguration:
    model_name: str
    model_threshold: float
    roi_x1: float
    roi_y1: float
    roi_x2: float
    roi_y2: float
    tile_count: int
    tile_overlap_ratio: float
    nms_iou_threshold: float
    recognition_settings: dict | None = None

    def to_snapshot(
        self,
        *,
        camera_resolution: str | None = None,
        camera_fps: float | None = None,
    ) -> dict[str, Any]:
        return {
            "camera_resolution": camera_resolution,
            "camera_fps": camera_fps,
            "roi": {
                "x1": self.roi_x1,
                "y1": self.roi_y1,
                "x2": self.roi_x2,
                "y2": self.roi_y2,
            },
            "tile_count": self.tile_count,
            "tile_overlap_ratio": self.tile_overlap_ratio,
            "nms_iou_threshold": self.nms_iou_threshold,
            "recognition_settings": self.recognition_settings,
        }


@dataclass(frozen=True)
class RecognitionSessionInput:
    class_code: str
    subject_code: str
    session_type: str
    notes: str

    @classmethod
    def from_form(cls, values: dict[str, str]) -> RecognitionSessionInput:
        class_code = _required_text(values.get("class_code", ""), "Class Code")
        subject_code = _required_text(
            values.get("subject_code", ""), "Subject Code"
        )
        session_type = values.get("session_type", "").strip().lower()
        if session_type not in SUPPORTED_SESSION_TYPES:
            raise ValueError("Session Type must be Camera Test or Real Class.")
        return cls(
            class_code=class_code,
            subject_code=subject_code,
            session_type=session_type,
            notes=values.get("notes", ""),
        )

    def to_insert_payload(
        self, configuration: RecognitionConfiguration
    ) -> dict[str, Any]:
        return {
            "class_code": self.class_code,
            "subject_code": self.subject_code,
            "session_type": self.session_type,
            "model_name": configuration.model_name,
            "model_threshold": configuration.model_threshold,
            "config_snapshot": configuration.to_snapshot(),
            "notes": self.notes if self.notes.strip() else None,
        }


@dataclass(frozen=True)
class RecognitionTrialInput:
    expected_user_id: str
    distance_m: float
    notes: str | None

    @classmethod
    def from_form(cls, values: dict[str, str]) -> RecognitionTrialInput:
        return cls(
            expected_user_id=_required_text(
                values.get("expected_user_id", ""), "Expected User"
            ),
            distance_m=_required_nonnegative_float(
                values.get("distance_m", ""), "Distance"
            ),
            notes=(
                values.get("trial_notes", "")
                if values.get("trial_notes", "").strip()
                else None
            ),
        )


@dataclass(frozen=True)
class RecognitionTrialSnapshot:
    detected: bool
    detector_confidence: float | None
    predicted_user_id: str | None
    predicted_name: str | None
    similarity: float | None
    threshold: float
    recognized: bool
    detection_time_ms: float
    recognition_time_ms: float
    diagnostics: dict | None = None


@dataclass(frozen=True)
class RecognitionSession:
    session_id: str
    context: RecognitionSessionInput
    configuration: RecognitionConfiguration
    started_at: str | None


def load_recognition_configuration_snapshot() -> RecognitionConfiguration:
    """Read the actual active backend settings without importing main.py."""
    backend_directory = Path(__file__).resolve().parents[2] / "backend"
    backend_path = str(backend_directory)
    if backend_path not in sys.path:
        sys.path.insert(0, backend_path)
    from recognition.detection_config import (
        CLASSROOM_ROI,
        TILED_DETECTION_CONFIG,
    )
    from recognition.offline_config import load_offline_config
    offline = load_offline_config()

    model_name = "adaface"
    return RecognitionConfiguration(
        model_name=model_name,
        model_threshold=offline.operating_point.similarity_threshold,
        recognition_settings=asdict(offline),
        roi_x1=float(CLASSROOM_ROI["x1"]),
        roi_y1=float(CLASSROOM_ROI["y1"]),
        roi_x2=float(CLASSROOM_ROI["x2"]),
        roi_y2=float(CLASSROOM_ROI["y2"]),
        tile_count=int(TILED_DETECTION_CONFIG["tile_count"]),
        tile_overlap_ratio=float(TILED_DETECTION_CONFIG["overlap_ratio"]),
        nms_iou_threshold=float(
            TILED_DETECTION_CONFIG["nms_iou_threshold"]
        ),
    )
