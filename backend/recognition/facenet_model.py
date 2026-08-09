import cv2
import numpy as np
import torch
from facenet_pytorch import InceptionResnetV1

from recognition.base_model import (
    FaceRecognitionModel
)
from recognition.face_preprocessing import (
    resize_aligned_face
)


class FaceNetModel(FaceRecognitionModel):

    def __init__(self):
        self.device = torch.device(
            "cuda"
            if torch.cuda.is_available()
            else "cpu"
        )

        self.model = (
            InceptionResnetV1(
                pretrained="vggface2"
            )
            .eval()
            .to(self.device)
        )

    @property
    def model_name(self) -> str:
        return "facenet"

    def extract_embedding(
        self,
        aligned_face: np.ndarray
    ) -> np.ndarray:
        face_160 = resize_aligned_face(
            aligned_face,
            output_size=160
        )

        rgb_face = cv2.cvtColor(
            face_160,
            cv2.COLOR_BGR2RGB
        )

        tensor = torch.from_numpy(
            rgb_face
        ).float()

        tensor = tensor.permute(
            2,
            0,
            1
        )

        tensor = (
            tensor - 127.5
        ) / 128.0

        tensor = tensor.unsqueeze(0).to(
            self.device
        )

        with torch.no_grad():
            embedding = self.model(tensor)

        embedding = (
            embedding
            .squeeze(0)
            .cpu()
            .numpy()
            .astype(np.float32)
        )

        return self.normalize_embedding(
            embedding
        )