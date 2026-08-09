from pathlib import Path

import numpy as np
from insightface.model_zoo import model_zoo

from recognition.base_model import (
    FaceRecognitionModel
)
from recognition.face_preprocessing import (
    resize_aligned_face
)


class BuffaloLModel(FaceRecognitionModel):

    def __init__(self):
        self.recognition_model = (
            self._load_recognition_model()
        )

    @property
    def model_name(self) -> str:
        return "buffalo_l"

    @staticmethod
    def _model_path() -> Path:
        return (
            Path.home()
            / ".insightface"
            / "models"
            / "buffalo_l"
            / "w600k_r50.onnx"
        )

    def _load_recognition_model(self):
        model_path = self._model_path()

        if not model_path.exists():
            raise FileNotFoundError(
                "Buffalo_L recognition model not found: "
                f"{model_path}"
            )

        model = model_zoo.get_model(
            str(model_path),
            providers=[
                "CPUExecutionProvider"
            ]
        )

        model.prepare(
            ctx_id=-1
        )

        return model

    def extract_embedding(
        self,
        aligned_face: np.ndarray
    ) -> np.ndarray:
        face_112 = resize_aligned_face(
            aligned_face,
            output_size=112
        )

        embedding = (
            self.recognition_model
            .get_feat(face_112)
        )

        embedding = np.asarray(
            embedding,
            dtype=np.float32
        )

        if embedding.ndim == 2:
            embedding = embedding[0]

        return self.normalize_embedding(
            embedding
        )