"""Realtime AdaFace recognition support for the desktop application."""

from .renderer import draw_recognition_results
from .worker import (
    RecognizedFace,
    RecognitionResult,
    RecognitionStatus,
    RecognitionWorker,
)

__all__ = [
    "RecognizedFace",
    "RecognitionResult",
    "RecognitionStatus",
    "RecognitionWorker",
    "draw_recognition_results",
]
