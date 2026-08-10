"""OpenCV rendering for AdaFace recognition results."""

from __future__ import annotations

from typing import Any, Iterable

import cv2


def draw_recognition_results(
    frame: Any,
    faces: Iterable[Any],
) -> Any:
    """Draw multi-face recognition labels on a copy of a BGR frame."""
    if frame is None or getattr(frame, "size", 0) == 0:
        raise ValueError("Recognition display frame is invalid.")

    annotated = frame.copy()

    for face in faces:
        x1, y1, x2, y2 = (
            int(round(value))
            for value in face.bbox
        )
        recognized = bool(face.recognized)
        color = (0, 200, 0) if recognized else (0, 0, 255)
        identity = face.full_name if recognized else "Unknown"
        similarity = float(face.similarity)
        label = (
            f"{identity} {similarity:.2f}"
            if similarity >= -0.999
            else identity
        )

        cv2.rectangle(
            annotated,
            (x1, y1),
            (x2, y2),
            color,
            2,
        )
        cv2.putText(
            annotated,
            label,
            (x1, max(y1 - 9, 22)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.58,
            color,
            2,
            cv2.LINE_AA,
        )

    return annotated


__all__ = ["draw_recognition_results"]
