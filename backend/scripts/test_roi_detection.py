import os
import sys
import time

import cv2


CURRENT_DIR = os.path.dirname(
    os.path.abspath(__file__)
)

BACKEND_DIR = os.path.dirname(CURRENT_DIR)

if BACKEND_DIR not in sys.path:
    sys.path.insert(0, BACKEND_DIR)


from recognition.face_detector import get_shared_detector
from recognition.detection_config import CLASSROOM_ROI


IMAGE_PATH = os.path.join(
    BACKEND_DIR,
    "test_classroom.png"
)


def main():
    image = cv2.imread(IMAGE_PATH)

    if image is None:
        raise RuntimeError(
            f"Cannot read image: {IMAGE_PATH}"
        )

    detector = get_shared_detector()

    start_time = time.perf_counter()

    faces = detector.detect_faces_in_roi(
        image=image,
        roi_config=CLASSROOM_ROI
    )

    elapsed_ms = (
        time.perf_counter() - start_time
    ) * 1000

    output = image.copy()

    height, width = image.shape[:2]

    roi_x1 = int(width * CLASSROOM_ROI["x1"])
    roi_y1 = int(height * CLASSROOM_ROI["y1"])
    roi_x2 = int(width * CLASSROOM_ROI["x2"])
    roi_y2 = int(height * CLASSROOM_ROI["y2"])

    cv2.rectangle(
        output,
        (roi_x1, roi_y1),
        (roi_x2, roi_y2),
        (255, 0, 0),
        3
    )

    for face in faces:
        x1, y1, x2, y2 = face.bbox.astype(int)

        cv2.rectangle(
            output,
            (x1, y1),
            (x2, y2),
            (0, 255, 0),
            2
        )

    output_path = os.path.join(
        BACKEND_DIR,
        "storage",
        "roi_detection_test.jpg"
    )

    cv2.imwrite(output_path, output)

    print("Detected faces:", len(faces))
    print("Detection time:", round(elapsed_ms, 2), "ms")
    print("Output:", output_path)


if __name__ == "__main__":
    main()