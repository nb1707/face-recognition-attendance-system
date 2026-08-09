from typing import Optional

import numpy as np


def resolve_roi(
    image: np.ndarray,
    roi_config: Optional[dict]
) -> tuple[np.ndarray, int, int]:
    """
    Return:
        cropped ROI
        original x offset
        original y offset
    """
    if image is None:
        raise ValueError("Input image is missing.")

    height, width = image.shape[:2]

    if roi_config is None:
        return image, 0, 0

    x1 = int(width * roi_config.get("x1", 0.0))
    y1 = int(height * roi_config.get("y1", 0.0))
    x2 = int(width * roi_config.get("x2", 1.0))
    y2 = int(height * roi_config.get("y2", 1.0))

    x1 = max(0, min(x1, width - 1))
    y1 = max(0, min(y1, height - 1))
    x2 = max(x1 + 1, min(x2, width))
    y2 = max(y1 + 1, min(y2, height))

    roi_image = image[y1:y2, x1:x2]

    if roi_image.size == 0:
        raise ValueError("Resolved ROI is empty.")

    return roi_image, x1, y1