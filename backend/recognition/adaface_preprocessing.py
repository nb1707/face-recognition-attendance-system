"""Application contract: uint8 OpenCV BGR -> RGB NCHW float32 in [-1, 1].

The local IR101 YAML declares RGB and its forward does not flip channels.
Version differs from the historical BGR adapter: old galleries must regenerate.
"""
import cv2
import numpy as np

PREPROCESSING_VERSION = "adaface-rgb112-v1"
EMBEDDING_DIMENSION = 512
ADAFACE_GALLERY_REGEN_REQUIRED = True


def preprocess_bgr(image):
    if not isinstance(image, np.ndarray) or image.dtype != np.uint8:
        raise ValueError("Expected uint8 OpenCV BGR image")
    if image.ndim != 3 or image.shape[2] != 3 or not image.size:
        raise ValueError("Expected nonempty HWC BGR image with three channels")
    resized = cv2.resize(image, (112, 112), interpolation=cv2.INTER_LINEAR)
    rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB).astype(np.float32)
    return np.ascontiguousarray(((rgb / 255.0 - 0.5) / 0.5).transpose(2, 0, 1)[None])
