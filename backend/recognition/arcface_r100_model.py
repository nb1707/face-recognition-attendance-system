from pathlib import Path

import numpy as np
from insightface.model_zoo import model_zoo

from recognition.base_model import FaceRecognitionModel
from recognition.face_preprocessing import resize_aligned_face


class ArcFaceR100Model(FaceRecognitionModel):

    def __init__(self):

        self.recognition_model = (
            self._load_model()
        )


    @property
    def model_name(self):
        return "arcface_r100"


    def _load_model(self):

        model_path = (
            Path(__file__)
            .resolve()
            .parent.parent
            /
            "model_weights"
            /
            "arcface"
            /
            "arcface_r100.onnx"
        )


        if not model_path.exists():
            raise FileNotFoundError(
                f"ArcFace model missing: {model_path}"
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
    ):

        face_112 = resize_aligned_face(
            aligned_face,
            112
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