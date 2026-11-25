"""Bounded Top-K selected observations and per-observation embedding cache."""
from dataclasses import dataclass
import math
import numpy as np
from recognition.embeddings import normalized_embedding


@dataclass(frozen=True)
class EvidenceConfig:
    top_k: int = 3
    max_age_seconds: float = 5.0
    aggregate_epsilon: float = 1e-6
    recent_refresh_seconds: float = .25

    def __post_init__(self):
        if self.top_k < 1 or not math.isfinite(self.max_age_seconds) or self.max_age_seconds <= 0:
            raise ValueError("Invalid evidence bounds")
        if not math.isfinite(self.aggregate_epsilon) or self.aggregate_epsilon <= 0:
            raise ValueError("Invalid aggregate epsilon")
        if not math.isfinite(self.recent_refresh_seconds) or not 0 < self.recent_refresh_seconds <= self.max_age_seconds:
            raise ValueError("Invalid recent representative refresh interval")


@dataclass
class Observation:
    track_id: int
    frame_sequence: int
    acquisition_time: float
    quality_score: float
    aligned_crop: np.ndarray
    bbox: tuple = ()
    landmarks: tuple = ()
    quality_measurements: object = None
    embedding: np.ndarray | None = None


def aggregate_embeddings(embeddings, weights, dimension, epsilon=1e-6):
    if not embeddings or len(embeddings) != len(weights):
        raise ValueError("empty_or_mismatched_aggregate")
    weights = np.asarray(weights, dtype=np.float64)
    if not np.isfinite(weights).all() or np.any(weights <= 0):
        raise ValueError("invalid_aggregate_weight")
    values = np.stack([normalized_embedding(value, dimension) for value in embeddings])
    # Dividing weights by their sum avoids scale-related near-zero decisions.
    return normalized_embedding(np.sum(values * (weights/weights.sum())[:, None], axis=0), dimension, epsilon)


class BestFrameBuffer:
    """Quality-ranked K with a periodically refreshed recent representative.

    At refresh, keep the best K-1 others plus the fresh observation. Between
    refreshes selection is quality-ranked; ties prefer newer frame sequences.

    Retained crops must be 112x112 BGR. Eviction drops both crop and embedding.
    High-water sequence rejects duplicates even after an observation was evicted.
    """
    def __init__(self, track_id, generation, dimension, config=None):
        self.track_id, self.generation, self.dimension = track_id, generation, dimension
        self.config = config or EvidenceConfig()
        self.observations = []
        self.last_sequence = -1
        self.inference_count = 0

    def expire(self, now):
        self.observations = [o for o in self.observations if 0 <= now-o.acquisition_time <= self.config.max_age_seconds]

    def offer(self, observation, quality_passed):
        o = observation
        if o.track_id != self.track_id:
            raise ValueError("observation_from_different_track")
        if o.frame_sequence <= self.last_sequence:
            return False
        self.last_sequence = o.frame_sequence
        self.expire(o.acquisition_time)
        if not quality_passed or not math.isfinite(o.quality_score) or not 0 < o.quality_score <= 1:
            return False
        if o.aligned_crop.shape != (112, 112, 3) or o.aligned_crop.dtype != np.uint8:
            raise ValueError("invalid_aligned_crop")
        refresh = bool(self.observations) and o.acquisition_time - max(
            item.acquisition_time for item in self.observations) >= self.config.recent_refresh_seconds - 1e-9
        self.observations.append(o)
        self.observations.sort(key=lambda item: (-item.quality_score, -item.frame_sequence))
        ranked = self.observations[:self.config.top_k]
        if refresh and not any(item is o for item in ranked):
            ranked = self.observations[:self.config.top_k-1] + [o]
        self.observations = ranked
        return any(item is o for item in self.observations)

    def aggregate(self, extract_embedding):
        for observation in self.observations:
            if observation.embedding is None:
                self.inference_count += 1
                try:
                    observation.embedding = normalized_embedding(extract_embedding(observation.aligned_crop), self.dimension)
                except (ValueError, TypeError, RuntimeError):
                    # A failed selected observation must not be inferred again
                    # when a later frame asks to aggregate retained evidence.
                    self.observations = [o for o in self.observations if o is not observation]
                    raise
        return aggregate_embeddings([o.embedding for o in self.observations],
                                    [o.quality_score for o in self.observations],
                                    self.dimension, self.config.aggregate_epsilon)

    @property
    def fingerprint(self):
        return tuple(sorted(o.frame_sequence for o in self.observations))

    @property
    def memory_bytes(self):
        return sum(o.aligned_crop.nbytes + (o.embedding.nbytes if o.embedding is not None else 0)
                   for o in self.observations)
