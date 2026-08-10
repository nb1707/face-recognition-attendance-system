"""Realtime face-detection support for the desktop application."""

from .renderer import draw_face_detections
from .worker import (
    DetectedFace,
    DetectionResult,
    DetectionStatus,
    DetectionWorker,
)

__all__ = [
    "DetectedFace",
    "DetectionResult",
    "DetectionStatus",
    "DetectionWorker",
    "draw_face_detections",
]
