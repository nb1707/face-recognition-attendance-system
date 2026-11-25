from dataclasses import dataclass
import math
import numpy as np

from enrollment.config import REPRESENTATIVE_TARGETS
from recognition.embeddings import normalized_embedding


@dataclass(frozen=True)
class Candidate:
    sample_id: str
    pose: str
    quality_score: float
    embedding: np.ndarray


def select_representatives(candidates, dimension=512, targets=REPRESENTATIVE_TARGETS):
    selected = []
    for pose, target in targets.items():
        pool = []
        for row in candidates:
            if row.pose != pose:
                continue
            quality = float(row.quality_score)
            if not math.isfinite(quality) or not 0 <= quality <= 1:
                raise ValueError(f"invalid quality score for {row.sample_id}")
            pool.append(Candidate(row.sample_id, row.pose, quality,
                                  normalized_embedding(row.embedding, dimension)))
        pool.sort(key=lambda row: (-row.quality_score, row.sample_id))
        if len(pool) < target:
            raise ValueError(f"insufficient candidates for {pose}")
        pose_selected = [pool.pop(0)]
        while len(pose_selected) < target:
            def rank(row):
                max_similarity = max(float(row.embedding @ chosen.embedding) for chosen in pose_selected)
                diversity = 1.0 - max_similarity
                return (-(0.70 * row.quality_score + 0.30 * diversity), row.sample_id)
            choice_index = min(range(len(pool)), key=lambda index: rank(pool[index]))
            choice = pool.pop(choice_index)
            pose_selected.append(choice)
        selected.extend(pose_selected)
    return tuple(selected)
