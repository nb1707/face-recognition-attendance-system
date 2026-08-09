from pathlib import Path
from typing import Any


SUPPORTED_EXTENSIONS = {
    ".jpg",
    ".jpeg",
    ".png",
    ".webp"
}

UNKNOWN_FOLDER_NAME = "unknown"
UNKNOWN_LABEL = "__unknown__"


def load_evaluation_dataset(
    dataset_root: str
) -> list[dict[str, Any]]:
    root = Path(dataset_root)

    if not root.exists():
        raise FileNotFoundError(
            f"Evaluation dataset does not exist: {root}"
        )

    if not root.is_dir():
        raise NotADirectoryError(
            f"Evaluation dataset path is not a folder: {root}"
        )

    samples: list[dict[str, Any]] = []

    for identity_folder in sorted(root.iterdir()):
        if not identity_folder.is_dir():
            continue

        folder_name = identity_folder.name

        expected_user_id = (
            UNKNOWN_LABEL
            if folder_name.lower() == UNKNOWN_FOLDER_NAME
            else folder_name
        )

        for image_path in sorted(identity_folder.rglob("*")):
            if not image_path.is_file():
                continue

            if image_path.suffix.lower() not in SUPPORTED_EXTENSIONS:
                continue

            samples.append({
                "image_path": str(image_path),
                "image_name": image_path.name,
                "expected_user_id": expected_user_id,
                "source_folder": folder_name
            })

    return samples

#Allow to approve regression each file in the evaluation dataset, to avoid false positives and false negatives, and to improve the accuracy of the model.