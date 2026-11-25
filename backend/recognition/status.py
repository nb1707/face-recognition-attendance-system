"""Shared diagnostic vocabulary; UI text may simplify, records must not."""
from enum import StrEnum


class RecognitionReason(StrEnum):
    NO_FACE = "no_face"
    INVALID = "invalid_observation"
    LOW_QUALITY = "low_quality"
    BELOW_THRESHOLD = "below_similarity_threshold"
    AMBIGUOUS = "ambiguous_identity"
    TEMPORAL_PENDING = "temporal_pending"
    CONFIRMED = "confirmed_identity"
    NO_IDENTITIES = "no_valid_identities"
    MARGIN_UNAVAILABLE = "margin_unavailable"
