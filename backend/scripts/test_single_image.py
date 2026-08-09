import os
import sys

import cv2
import numpy as np


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


from recognition.model_factory import get_model
from recognition.face_detector import get_shared_detector
from recognition.face_preprocessing import align_detected_face


TEST_IMAGE = os.path.join(
    BACKEND_DIR,
    "evaluation_dataset",
    "student_001",
    "img14.png"
)


MODELS = [
    "buffalo_l",
    "facenet",
    "adaface",
    "arcface_r100"
]


def main():
    image = cv2.imread(
        TEST_IMAGE
    )

    if image is None:
        print(
            f"Cannot read image: {TEST_IMAGE}"
        )
        return

    detector = get_shared_detector()

    faces = detector.detect_faces(
        image
    )

    print(
        f"Testing image: {TEST_IMAGE}"
    )

    print(
        "Detected faces:",
        len(faces)
    )

    if len(faces) == 0:
        print("No face detected.")
        return

    if len(faces) > 1:
        print(
            "Multiple faces detected. "
            "Using the largest face."
        )

        face = max(
            faces,
            key=lambda detected_face: (
                (
                    detected_face.bbox[2]
                    - detected_face.bbox[0]
                )
                *
                (
                    detected_face.bbox[3]
                    - detected_face.bbox[1]
                )
            )
        )
    else:
        face = faces[0]

    try:
        aligned_face = align_detected_face(
            image=image,
            detected_face=face,
            output_size=112
        )
    except Exception as error:
        print(
            "Face alignment failed:",
            error
        )
        return

    print(
        "Aligned face shape:",
        aligned_face.shape
    )

    for model_name in MODELS:
        print("\n====================")
        print("MODEL:", model_name)

        try:
            model = get_model(
                model_name
            )

            embedding = model.extract_embedding(
                aligned_face
            )

            embedding = np.asarray(
                embedding,
                dtype=np.float32
            ).reshape(-1)

            print(
                "Embedding shape:",
                embedding.shape
            )

            print(
                "Embedding norm:",
                float(
                    np.linalg.norm(
                        embedding
                    )
                )
            )

            print(
                "First 5 values:",
                embedding[:5]
            )

        except Exception as error:
            print(
                "FAILED:",
                error
            )


if __name__ == "__main__":
    main()