from typing import Any, Callable

import cv2

from evaluation.dataset_loader import (
    load_evaluation_dataset,
    UNKNOWN_LABEL
)
from evaluation.metrics import (
    calculate_metrics,
    NO_FACE_LABEL,
    MULTIPLE_FACES_LABEL
)


RecognitionFunction = Callable[..., dict[str, Any]]


def evaluate_single_image(
    sample: dict[str, Any],
    model_name: str,
    recognition_function: RecognitionFunction,
    cached_users: dict,
    cached_embeddings: dict
) -> dict[str, Any]:
    image_path = sample["image_path"]
    expected_user_id = sample["expected_user_id"]

    image = cv2.imread(image_path)

    if image is None:
        return {
            "image_name": sample["image_name"],
            "image_path": image_path,
            "expected_user_id": expected_user_id,
            "predicted_user_id": NO_FACE_LABEL,
            "confidence": -1.0,
            "is_correct": False,
            "processing_time_ms": 0.0,
            "error_message": "Could not read image"
        }

    recognition_result = recognition_function(
        img=image,
        model_name=model_name,
        cached_users=cached_users,
        cached_embeddings=cached_embeddings
    )

    face_results = recognition_result.get(
        "results",
        []
    )

    processing_time_ms = recognition_result.get(
        "processing_time_ms",
        0.0
    )

    if len(face_results) == 0:
        predicted_user_id = NO_FACE_LABEL
        confidence = -1.0
        error_message = "No face detected"

    elif len(face_results) > 1:
        predicted_user_id = MULTIPLE_FACES_LABEL
        confidence = max(
            float(face.get("confidence", -1.0))
            for face in face_results
        )
        error_message = "Multiple faces detected"

    else:
        face = face_results[0]
        confidence = float(
            face.get("confidence", -1.0)
        )

        if face.get("recognized"):
            predicted_user_id = face.get(
                "user_id",
                UNKNOWN_LABEL
            )
        else:
            predicted_user_id = UNKNOWN_LABEL

        error_message = None

    return {
        "image_name": sample["image_name"],
        "image_path": image_path,
        "expected_user_id": expected_user_id,
        "predicted_user_id": predicted_user_id,
        "confidence": confidence,
        "is_correct": (
            expected_user_id == predicted_user_id
        ),
        "processing_time_ms": (
            float(processing_time_ms)
        ),
        "error_message": error_message
    }


def run_model_benchmark(
    model_name: str,
    dataset_root: str,
    recognition_function: RecognitionFunction,
    cached_users: dict,
    cached_embeddings: dict
) -> dict[str, Any]:
    samples = load_evaluation_dataset(
        dataset_root
    )

    records: list[dict[str, Any]] = []

    for index, sample in enumerate(
        samples,
        start=1
    ):
        print(
            f"[{model_name}] "
            f"{index}/{len(samples)} "
            f"{sample['image_path']}"
        )

        record = evaluate_single_image(
            sample=sample,
            model_name=model_name,
            recognition_function=recognition_function,
            cached_users=cached_users,
            cached_embeddings=cached_embeddings
        )

        records.append(record)

    metrics = calculate_metrics(records)

    return {
        "model_name": model_name,
        "dataset_root": dataset_root,
        "records": records,
        "metrics": metrics
    }