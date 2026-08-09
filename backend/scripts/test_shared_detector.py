import os
import sys

import cv2


CURRENT_DIR = os.path.dirname(
    os.path.abspath(__file__)
)

BACKEND_DIR = os.path.dirname(
    CURRENT_DIR
)

if BACKEND_DIR not in sys.path:
    sys.path.insert(0, BACKEND_DIR)


from recognition.face_detector import (
    get_shared_detector
)


IMAGE_PATH = os.path.join(
    BACKEND_DIR,
    "test_face.jpg"
)


def main():
    image = cv2.imread(
        IMAGE_PATH
    )

    if image is None:
        raise RuntimeError(
            f"Cannot read image: {IMAGE_PATH}"
        )

    detector = get_shared_detector()

    faces = detector.detect_faces(
        image
    )

    print(
        "Detected faces:",
        len(faces)
    )

    for index, face in enumerate(
        faces,
        start=1
    ):
        print(
            f"\nFace {index}"
        )

        print(
            "bbox:",
            getattr(
                face,
                "bbox",
                None
            )
        )

        print(
            "kps:",
            getattr(
                face,
                "kps",
                None
            )
        )


if __name__ == "__main__":
    main()