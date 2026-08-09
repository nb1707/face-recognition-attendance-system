import os
import sys
import time

import cv2


CURRENT_DIR = os.path.dirname(
    os.path.abspath(__file__)
)

BACKEND_DIR = os.path.dirname(
    CURRENT_DIR
)

if BACKEND_DIR not in sys.path:
    sys.path.insert(
        0,
        BACKEND_DIR
    )


from recognition.face_detector import (
    get_shared_detector
)

from recognition.detection_config import (
    CLASSROOM_ROI
)


IMAGE_PATH = os.path.join(
    BACKEND_DIR,
    "test_classroom.png"
)


def measure(
    function,
    *args,
    **kwargs
):
    started_at = time.perf_counter()

    faces = function(
        *args,
        **kwargs
    )

    elapsed_ms = (
        time.perf_counter()
        - started_at
    ) * 1000

    return faces, elapsed_ms


def main():
    image = cv2.imread(
        IMAGE_PATH
    )

    if image is None:
        raise RuntimeError(
            f"Cannot read image: {IMAGE_PATH}"
        )

    detector = get_shared_detector()

    full_faces, full_time = measure(
        detector.detect_faces,
        image
    )

    roi_faces, roi_time = measure(
        detector.detect_faces_in_roi,
        image,
        CLASSROOM_ROI
    )

    tiled_faces, tiled_time = measure(
        detector.detect_faces_tiled,
        image=image,
        roi_config=CLASSROOM_ROI,
        tile_count=3,
        overlap_ratio=0.20,
        nms_iou_threshold=0.40
    )

    print()
    print(
        f"{'Mode':<20}"
        f"{'Faces':<10}"
        f"{'Time (ms)':<12}"
    )

    print("-" * 42)

    print(
        f"{'Full frame':<20}"
        f"{len(full_faces):<10}"
        f"{full_time:<12.2f}"
    )

    print(
        f"{'ROI':<20}"
        f"{len(roi_faces):<10}"
        f"{roi_time:<12.2f}"
    )

    print(
        f"{'ROI + 3 tiles':<20}"
        f"{len(tiled_faces):<10}"
        f"{tiled_time:<12.2f}"
    )


if __name__ == "__main__":
    main()