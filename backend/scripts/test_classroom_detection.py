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
    sys.path.insert(0, BACKEND_DIR)


from recognition.face_detector import get_shared_detector


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

    faces = detector.detect_faces(image)

    elapsed_ms = (
        time.perf_counter() - start_time
    ) * 1000

    print("Image shape:", image.shape)
    print("Detected faces:", len(faces))
    print("Detection time:", round(elapsed_ms, 2), "ms")

    for index, face in enumerate(faces, start=1):
        bbox = face.bbox.astype(int)

        width = bbox[2] - bbox[0]
        height = bbox[3] - bbox[1]

        print(
            f"Face {index}: "
            f"bbox={bbox.tolist()}, "
            f"size={width}x{height}, "
            f"score={float(face.det_score):.3f}"
        )


if __name__ == "__main__":
    main()