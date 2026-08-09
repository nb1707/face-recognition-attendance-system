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


OUTPUT_PATH = os.path.join(
    BACKEND_DIR,
    "storage",
    "tiled_detection_test.jpg"
)


def draw_faces(
    image,
    faces,
    box_color
):
    output = image.copy()

    for index, face in enumerate(
        faces,
        start=1
    ):
        x1, y1, x2, y2 = (
            face.bbox.astype(int)
        )

        cv2.rectangle(
            output,
            (x1, y1),
            (x2, y2),
            box_color,
            2
        )

        detection_score = float(
            getattr(
                face,
                "det_score",
                0.0
            )
        )

        label = (
            f"{index}: "
            f"{detection_score:.2f}"
        )

        cv2.putText(
            output,
            label,
            (
                x1,
                max(20, y1 - 6)
            ),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            box_color,
            2,
            cv2.LINE_AA
        )

    return output


def main():
    image = cv2.imread(
        IMAGE_PATH
    )

    if image is None:
        raise RuntimeError(
            f"Cannot read image: {IMAGE_PATH}"
        )

    detector = get_shared_detector()

    start_time = time.perf_counter()

    faces = detector.detect_faces_tiled(
        image=image,
        roi_config=CLASSROOM_ROI,
        tile_count=3,
        overlap_ratio=0.20,
        nms_iou_threshold=0.40
    )

    elapsed_ms = (
        time.perf_counter()
        - start_time
    ) * 1000

    output = draw_faces(
        image=image,
        faces=faces,
        box_color=(0, 255, 0)
    )

    height, width = image.shape[:2]

    roi_x1 = int(
        width * CLASSROOM_ROI["x1"]
    )

    roi_y1 = int(
        height * CLASSROOM_ROI["y1"]
    )

    roi_x2 = int(
        width * CLASSROOM_ROI["x2"]
    )

    roi_y2 = int(
        height * CLASSROOM_ROI["y2"]
    )

    cv2.rectangle(
        output,
        (roi_x1, roi_y1),
        (roi_x2, roi_y2),
        (255, 0, 0),
        3
    )

    saved = cv2.imwrite(
        OUTPUT_PATH,
        output
    )

    if not saved:
        raise RuntimeError(
            f"Could not save output: {OUTPUT_PATH}"
        )

    print(
        "Image shape:",
        image.shape
    )

    print(
        "Detected faces after NMS:",
        len(faces)
    )

    print(
        "Detection time:",
        round(elapsed_ms, 2),
        "ms"
    )

    print(
        "Output:",
        OUTPUT_PATH
    )

    for index, face in enumerate(
        faces,
        start=1
    ):
        bbox = face.bbox.astype(int)

        face_width = (
            bbox[2] - bbox[0]
        )

        face_height = (
            bbox[3] - bbox[1]
        )

        print(
            f"Face {index}: "
            f"bbox={bbox.tolist()}, "
            f"size={face_width}x{face_height}, "
            f"score={float(face.det_score):.3f}"
        )


if __name__ == "__main__":
    main()