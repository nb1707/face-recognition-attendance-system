from abc import ABC, abstractmethod

import numpy as np


class FaceRecognitionModel(ABC):
    """
    Base interface for embedding models.

    Face detection and alignment are handled outside this class.
    Every model receives the same aligned face image.
    """

    @property
    @abstractmethod
    def model_name(self) -> str:
        pass

    @abstractmethod
    def extract_embedding(
        self,
        aligned_face: np.ndarray
    ) -> np.ndarray:
        """
        Input:
            aligned_face: BGR image, normally 112×112.

        Output:
            L2-normalized one-dimensional embedding.
        """
        pass

    def normalize_embedding(
        self,
        embedding: np.ndarray
    ) -> np.ndarray:
        embedding = np.asarray(
            embedding,
            dtype=np.float32
        ).reshape(-1)

        norm = np.linalg.norm(embedding)

        if norm == 0:
            raise ValueError(
                "Cannot normalize a zero embedding."
            )

        return embedding / norm

    def compare(
        self,
        embedding_a: np.ndarray,
        embedding_b: np.ndarray
    ) -> float:
        normalized_a = self.normalize_embedding(
            embedding_a
        )

        normalized_b = self.normalize_embedding(
            embedding_b
        )

        return float(
            np.dot(
                normalized_a,
                normalized_b
            )
        )