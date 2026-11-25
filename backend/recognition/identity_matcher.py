"""Stateless cosine matcher. Pose scores aggregate by MAX per distinct user."""
from dataclasses import dataclass
import numpy as np
from recognition.embeddings import normalized_embedding
from recognition.local_gallery import POSE_ORDER


@dataclass(frozen=True)
class IdentityScore:
    user_id: str
    score: float
    winning_pose: str
    winning_prototype_id: str | None = None


@dataclass(frozen=True)
class MatchResult:
    top1_user_id: str | None = None
    top1_score: float | None = None
    top1_winning_pose: str | None = None
    top2_user_id: str | None = None
    top2_score: float | None = None
    top2_winning_pose: str | None = None
    margin: float | None = None
    candidate_count: int = 0
    comparisons: int = 0
    diagnostics: tuple[str, ...] = ()
    top1_winning_prototype_id: str | None = None
    top2_winning_prototype_id: str | None = None


def match_identity(query, identities, dimension):
    scores, diagnostics, comparisons = score_identities(query, identities, dimension)
    first = scores[0] if scores else None
    second = scores[1] if len(scores) > 1 else None
    return MatchResult(
        top1_user_id=first.user_id if first else None,
        top1_score=first.score if first else None,
        top1_winning_pose=first.winning_pose if first else None,
        top2_user_id=second.user_id if second else None,
        top2_score=second.score if second else None,
        top2_winning_pose=second.winning_pose if second else None,
        margin=first.score - second.score if second else None,
        candidate_count=len(scores),
        comparisons=comparisons,
        diagnostics=diagnostics,
        top1_winning_prototype_id=(
            first.winning_prototype_id if first else None
        ),
        top2_winning_prototype_id=(
            second.winning_prototype_id if second else None
        ),
    )


def score_identities(query, identities, dimension):
    """Validate prototypes, prefer poses over mean, and return deterministic ranks.

    Duplicate identity containers are merged by user ID before the pose policy.
    Ties: lexical user_id, then configured pose order. Scores are not probabilities.
    Invalid prototypes are skipped with diagnostics; invalid query raises.
    """
    query = normalized_embedding(query, dimension)
    grouped, diagnostics = {}, []
    for identity in identities:
        for prototype in identity.prototypes:
            try:
                if prototype.pose not in POSE_ORDER:
                    raise ValueError("invalid_pose")
                vector = normalized_embedding(prototype.embedding, dimension)
                grouped.setdefault(identity.user_id, []).append((
                    prototype.pose,
                    vector,
                    prototype.prototype_id,
                ))
            except (ValueError, TypeError) as error:
                diagnostics.append(f"{identity.user_id}/{prototype.pose}: {error}")
    scores, comparisons = [], 0
    for user_id in sorted(grouped):
        prototypes = grouped[user_id]
        poses = [p for p in prototypes if p[0] != "mean"]
        selected = sorted(poses or prototypes, key=lambda p: POSE_ORDER[p[0]])
        values = [
            (
                float(np.clip(query @ vector, -1, 1)),
                pose,
                prototype_id,
            )
            for pose, vector, prototype_id in selected
        ]
        comparisons += len(values)
        score, pose, prototype_id = min(
            values,
            key=lambda p: (-p[0], POSE_ORDER[p[1]]),
        )
        scores.append(IdentityScore(user_id, score, pose, prototype_id))
    return tuple(sorted(scores, key=lambda s: (-s.score, s.user_id))), tuple(diagnostics), comparisons
