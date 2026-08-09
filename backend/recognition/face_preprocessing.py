import cv2
import numpy as np
from insightface.utils import face_align


DEFAULT_ALIGNED_SIZE = 112


def align_detected_face(
    image: np.ndarray,
    detected_face,
    output_size: int = DEFAULT_ALIGNED_SIZE
) -> np.ndarray:
    if image is None:
        raise ValueError("Input image is missing.")

    landmarks = getattr(
        detected_face,
        "kps",
        None
    )

    if landmarks is None:
        raise ValueError(
            "Detected face does not contain 5-point landmarks."
        )

    aligned_face = face_align.norm_crop(
        image,
        landmark=landmarks,
        image_size=output_size
    )

    if aligned_face is None or aligned_face.size == 0:
        raise ValueError(
            "Face alignment failed."
        )

    return aligned_face


def resize_aligned_face(
    aligned_face: np.ndarray,
    output_size: int
) -> np.ndarray:
    if aligned_face is None or aligned_face.size == 0:
        raise ValueError(
            "Aligned face image is empty."
        )

    return cv2.resize(
        aligned_face,
        (output_size, output_size),
        interpolation=cv2.INTER_LINEAR
    )