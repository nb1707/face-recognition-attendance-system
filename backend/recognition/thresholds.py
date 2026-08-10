"""Recognition thresholds shared by web and desktop entry points."""


MODEL_THRESHOLDS = {
    "buffalo_l": 0.45,
    "facenet": 0.65,
    "adaface": 0.40,
    "arcface_r100": 0.45,
}


def get_model_threshold(model_name: str) -> float:
    """Return the configured threshold without changing model behavior."""
    normalized_name = model_name.strip().lower()

    if normalized_name not in MODEL_THRESHOLDS:
        raise ValueError(
            "Recognition threshold is not configured for model: "
            f"{model_name}"
        )

    return MODEL_THRESHOLDS[normalized_name]


__all__ = ["MODEL_THRESHOLDS", "get_model_threshold"]
