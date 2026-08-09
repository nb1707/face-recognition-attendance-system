from recognition.buffalo_l_model import BuffaloLModel
from recognition.facenet_model import FaceNetModel

from recognition.adaface_model import AdaFaceModel
from recognition.arcface_r100_model import ArcFaceR100Model
_model_cache = {}


def get_model(model_name: str):
    normalized_name = (model_name.strip().lower())

    if normalized_name in _model_cache:
        return _model_cache[normalized_name]


    if normalized_name == "buffalo_l":
        model = BuffaloLModel()

    elif normalized_name == "facenet":
        model = FaceNetModel()

    elif normalized_name == "adaface":
        model = AdaFaceModel()

    elif normalized_name == "arcface_r100":
        model = ArcFaceR100Model()
    else:
        raise ValueError(
            "Unsupported recognition model: "f"{model_name}"
        )

    _model_cache[
        normalized_name
    ] = model

    return model