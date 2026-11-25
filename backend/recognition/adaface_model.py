import cv2
import numpy as np
import torch

from recognition.base_model import (
    FaceRecognitionModel
)
from recognition.adaface_loader import (
    load_adaface_from_local, local_model_provenance
)
from recognition.adaface_preprocessing import preprocess_bgr, EMBEDDING_DIMENSION
from recognition.embeddings import normalized_embedding


class AdaFaceModel(FaceRecognitionModel):

    def __init__(self):
        self.provenance = local_model_provenance()
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
        if local_model_provenance() != self.provenance:
            raise RuntimeError("AdaFace checkpoint changed during loading; restart with a stable artifact")

    @property
    def model_name(self) -> str:
        return "adaface"

    def extract_embedding(
        self,
        aligned_face: np.ndarray
    ) -> np.ndarray:
        # All callers supply OpenCV BGR. RGB conversion belongs here only.
        tensor = torch.from_numpy(preprocess_bgr(aligned_face)).to(self.device)

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

        return normalized_embedding(embedding.reshape(-1), EMBEDDING_DIMENSION)

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
