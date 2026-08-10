"""Draw detection-only overlays without changing camera-owned frames."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

from .worker import DetectedFace

try:
    import cv2
except ImportError:
    cv2 = None


def draw_face_detections(
    frame: Any,
    faces: Sequence[DetectedFace],
) -> Any:
    """Return a frame copy annotated with full-frame detection coordinates."""
    if cv2 is None:
        raise RuntimeError("OpenCV is unavailable for detection drawing.")

    if frame is None or getattr(frame, "size", 0) == 0:
        raise ValueError("Cannot annotate an empty camera frame.")

    output = frame.copy()
    frame_height, frame_width = output.shape[:2]

    for face in faces:
        x1, y1, x2, y2 = face.bbox
        draw_x1 = max(0, min(int(round(x1)), frame_width - 1))
        draw_y1 = max(0, min(int(round(y1)), frame_height - 1))
        draw_x2 = max(0, min(int(round(x2)), frame_width - 1))
        draw_y2 = max(0, min(int(round(y2)), frame_height - 1))

        cv2.rectangle(
            output,
            (draw_x1, draw_y1),
            (draw_x2, draw_y2),
            (0, 255, 0),
            2,
        )

        label = f"Face {face.detector_confidence:.2f}"
        cv2.putText(
            output,
            label,
            (draw_x1, max(20, draw_y1 - 8)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (0, 255, 0),
            2,
            cv2.LINE_AA,
        )

    return output
