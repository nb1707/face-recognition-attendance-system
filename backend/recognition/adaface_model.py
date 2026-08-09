import cv2
import numpy as np
import torch

from recognition.base_model import (
    FaceRecognitionModel
)
from recognition.adaface_loader import (
    load_adaface_from_local
)
from recognition.face_preprocessing import (
    resize_aligned_face
)


class AdaFaceModel(FaceRecognitionModel):

    def __init__(self):
        self.device = torch.device(
            "cuda"
            if torch.cuda.is_available()
            else "cpu"
        )

        self.model = (
            load_adaface_from_local()
            .eval()
            .to(self.device)
        )

    @property
    def model_name(self) -> str:
        return "adaface"

    def extract_embedding(
        self,
        aligned_face: np.ndarray
    ) -> np.ndarray:
        face_112 = resize_aligned_face(
            aligned_face,
            output_size=112
        )

        # AdaFace model expects BGR input.
        tensor = torch.from_numpy(
            face_112
        ).float()

        tensor = tensor.permute(
            2,
            0,
            1
        )

        tensor = tensor / 255.0
        tensor = (tensor - 0.5) / 0.5

        tensor = tensor.unsqueeze(0).to(
            self.device
        )

        with torch.no_grad():
            output = self.model(tensor)

        embedding = self._extract_feature_tensor(
            output
        )

        embedding = (
            embedding
            .squeeze(0)
            .detach()
            .cpu()
            .numpy()
            .astype(np.float32)
        )

        return self.normalize_embedding(
            embedding
        )

    @staticmethod
    def _extract_feature_tensor(output):
        if isinstance(output, torch.Tensor):
            return output

        if isinstance(output, (tuple, list)):
            for item in output:
                if (
                    isinstance(item, torch.Tensor)
                    and item.ndim >= 2
                ):
                    return item

        if isinstance(output, dict):
            for key in [
                "embedding",
                "embeddings",
                "feature",
                "features"
            ]:
                value = output.get(key)

                if isinstance(value, torch.Tensor):
                    return value

        for attribute_name in [
            "embedding",
            "embeddings",
            "feature",
            "features"
        ]:
            value = getattr(
                output,
                attribute_name,
                None
            )

            if isinstance(value, torch.Tensor):
                return value

        raise TypeError(
            "Unsupported AdaFace output type: "
            f"{type(output)}"
        )