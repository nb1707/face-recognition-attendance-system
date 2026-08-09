from typing import Any

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    precision_recall_fscore_support,
    classification_report
)


UNKNOWN_LABEL = "__unknown__"
NO_FACE_LABEL = "__no_face__"
MULTIPLE_FACES_LABEL = "__multiple_faces__"


# def calculate_open_set_counts(
#     expected_labels: list[str],
#     predicted_labels: list[str]
# ) -> dict[str, int]:
#     true_known = 0
#     false_accept = 0
#     false_reject = 0
#     true_unknown = 0

#     for expected, predicted in zip(
#         expected_labels,
#         predicted_labels
#     ):
#         expected_is_known = expected != UNKNOWN_LABEL
#         predicted_is_known = predicted not in {
#             UNKNOWN_LABEL,
#             NO_FACE_LABEL,
#             MULTIPLE_FACES_LABEL
#         }

#         if expected_is_known and predicted_is_known:
#             true_known += 1

#         elif not expected_is_known and predicted_is_known:
#             false_accept += 1

#         elif expected_is_known and not predicted_is_known:
#             false_reject += 1

#         else:
#             true_unknown += 1

#     return {
#         "true_known": true_known,
#         "false_accept": false_accept,
#         "false_reject": false_reject,
#         "true_unknown": true_unknown
#     }


def safe_divide(
    numerator: float,
    denominator: float
) -> float:
    if denominator == 0:
        return 0.0

    return float(numerator / denominator)

def calculate_open_set_counts(
    expected_labels: list[str],
    predicted_labels: list[str]
) -> dict[str, int]:
    correct_known_identity = 0
    wrong_known_identity = 0
    false_accept = 0
    false_reject = 0
    true_unknown = 0

    rejected_labels = {
        UNKNOWN_LABEL,
        NO_FACE_LABEL,
        MULTIPLE_FACES_LABEL
    }

    for expected, predicted in zip(
        expected_labels,
        predicted_labels
    ):
        expected_is_known = expected != UNKNOWN_LABEL
        predicted_is_known = predicted not in rejected_labels

       # registered user and predicted as registered user
        if expected_is_known and predicted_is_known:
            if expected == predicted:
                correct_known_identity += 1
            else:
                wrong_known_identity += 1

        # unknown user but predicted as registered user
        elif not expected_is_known and predicted_is_known:
            false_accept += 1

        # registered user but rejected or detection failed
        elif expected_is_known and not predicted_is_known:
            false_reject += 1

        # unknown user and system also rejects correctly
        else:
            true_unknown += 1

    return {
        "correct_known_identity": correct_known_identity,
        "wrong_known_identity": wrong_known_identity,
        "false_accept": false_accept,
        "false_reject": false_reject,
        "true_unknown": true_unknown
    }

def calculate_metrics(
    records: list[dict[str, Any]]
) -> dict[str, Any]:
    if not records:
        return {
            "total_images": 0,
            "accuracy": 0.0,
            "precision": 0.0,
            "recall": 0.0,
            "f1_score": 0.0
        }

    expected_labels = [
        record["expected_user_id"]
        for record in records
    ]

    predicted_labels = [
        record["predicted_user_id"]
        for record in records
    ]

    labels = sorted(
        set(expected_labels) |
        set(predicted_labels)
    )

    accuracy = accuracy_score(
        expected_labels,
        predicted_labels
    )

    precision, recall, f1, _ = (
        precision_recall_fscore_support(
            expected_labels,
            predicted_labels,
            average="macro",
            zero_division=0
        )
    )

    matrix = confusion_matrix(
        expected_labels,
        predicted_labels,
        labels=labels
    )

    report = classification_report(
        expected_labels,
        predicted_labels,
        labels=labels,
        output_dict=True,
        zero_division=0
    )

    open_set_counts = calculate_open_set_counts(
        expected_labels,
        predicted_labels
    )

    false_accept_rate = safe_divide(
        open_set_counts["false_accept"],
        (
            open_set_counts["false_accept"] +
            open_set_counts["true_unknown"]
        )
    )

    total_known_samples = (
        open_set_counts["correct_known_identity"]
        + open_set_counts["wrong_known_identity"]
        + open_set_counts["false_reject"]
    )

    false_reject_rate = safe_divide(
        open_set_counts["false_reject"],
        total_known_samples
    )

    wrong_identity_rate = safe_divide(
        open_set_counts["wrong_known_identity"],
        total_known_samples
    )

    # known_records = [
    #     record
    #     for record in records
    #     if record["expected_user_id"] != UNKNOWN_LABEL
    # ]

    # known_correct = sum(
    #     1
    #     for record in known_records
    #     if record["expected_user_id"]
    #     == record["predicted_user_id"]
    # )

    known_identity_accuracy = safe_divide(
        open_set_counts["correct_known_identity"],
        total_known_samples
    )

    processing_times = [
        float(record["processing_time_ms"])
        for record in records
        if record.get("processing_time_ms") is not None
    ]

    average_processing_time_ms = (
        float(np.mean(processing_times))
        if processing_times
        else 0.0
    )

    return {
        "total_images": len(records),
        "correct_predictions": sum(
            1 for record in records
            if record["is_correct"]
        ),
        "accuracy": float(accuracy),
        "precision": float(precision),
        "recall": float(recall),
        "f1_score": float(f1),
        "false_accept_rate": false_accept_rate,
        "false_reject_rate": false_reject_rate,
        "wrong_identity_rate": wrong_identity_rate,
        "known_identity_accuracy": (
            known_identity_accuracy
        ),
        "average_processing_time_ms": (
            average_processing_time_ms
        ),
        "labels": labels,
        "confusion_matrix": matrix.tolist(),
        "classification_report": report,
        "open_set_counts": open_set_counts
    }
#Use macro precision, recall, and F1-score to evaluate the model's performance across all classes equally, regardless of class imbalance. 
#This ensures that the evaluation metrics reflect the model's ability to correctly identify both known and unknown identities in the dataset.
#Scikit-learn directly provides metric classifications and confusion matrix