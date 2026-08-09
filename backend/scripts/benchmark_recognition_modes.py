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
    sys.path.insert(
        0,
        BACKEND_DIR
    )


from main import (
    recognize_faces_with_model,
    load_users,
    load_cloud_mean_embeddings
)


IMAGE_PATH = os.path.join(
    BACKEND_DIR,
    "test_classroom.png"
)

MODEL_NAME = "adaface"

REPEAT_COUNT = 5


def main():
    image = cv2.imread(
        IMAGE_PATH
    )

    if image is None:
        raise RuntimeError(
            f"Cannot read image: "
            f"{IMAGE_PATH}"
        )

    users = load_users()

    embeddings = (
        load_cloud_mean_embeddings(
            MODEL_NAME
        )
    )

    modes = [
        "full",
        "roi",
        "tiled"
    ]

    for mode in modes:
        detection_times = []
        recognition_times = []
        total_times = []
        face_counts = []

        print()
        print(
            f"Benchmark mode: {mode}"
        )

        recognize_faces_with_model(
            img=image,
            model_name=MODEL_NAME,
            cached_users=users,
            cached_embeddings=embeddings,
            detection_mode=mode
        )

        for run_index in range(
            1,
            REPEAT_COUNT + 1
        ):
            result = (
                recognize_faces_with_model(
                    img=image,
                    model_name=MODEL_NAME,
                    cached_users=users,
                    cached_embeddings=embeddings,
                    detection_mode=mode
                )
            )

            detection_times.append(
                result.get(
                    "detection_time_ms",
                    0.0
                )
            )

            recognition_times.append(
                result.get(
                    "recognition_time_ms",
                    0.0
                )
            )

            total_times.append(
                result.get(
                    "processing_time_ms",
                    0.0
                )
            )

            face_counts.append(
                result.get(
                    "total_faces",
                    0
                )
            )

            print(
                f"run={run_index} "
                f"faces={face_counts[-1]} "
                f"detection="
                f"{detection_times[-1]:.2f} ms "
                f"recognition="
                f"{recognition_times[-1]:.2f} ms "
                f"total="
                f"{total_times[-1]:.2f} ms"
            )

        print(
            f"Average faces: "
            f"{sum(face_counts) / len(face_counts):.1f}"
        )

        print(
            f"Average detection: "
            f"{sum(detection_times) / len(detection_times):.2f} ms"
        )

        print(
            f"Average recognition: "
            f"{sum(recognition_times) / len(recognition_times):.2f} ms"
        )

        print(
            f"Average total: "
            f"{sum(total_times) / len(total_times):.2f} ms"
        )


if __name__ == "__main__":
    main()