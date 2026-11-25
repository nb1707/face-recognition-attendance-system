"""Gate A is stateless; Gate B consumes distinct new evidence per track."""
from dataclasses import dataclass, field
import math
from recognition.status import RecognitionReason as Reason


@dataclass(frozen=True)
class OperatingPoint:
    similarity_threshold: float
    margin_threshold: float
    fail_closed_when_margin_unavailable: bool = True

    def __post_init__(self):
        if not -1 <= self.similarity_threshold <= 1 or not 0 <= self.margin_threshold <= 2:
            raise ValueError("Invalid similarity/margin operating point")


def gate_a(match, point):
    if match.top1_user_id is None or match.top1_score is None:
        return False, Reason.NO_IDENTITIES.value
    if not math.isfinite(match.top1_score) or match.top1_score < point.similarity_threshold:
        return False, Reason.BELOW_THRESHOLD.value
    if match.top2_user_id is None:
        return ((False, Reason.MARGIN_UNAVAILABLE.value) if point.fail_closed_when_margin_unavailable
                else (True, Reason.TEMPORAL_PENDING.value))
    if match.top2_user_id == match.top1_user_id or match.margin is None or not math.isfinite(match.margin) or match.margin < point.margin_threshold:
        return False, Reason.AMBIGUOUS.value
    return True, Reason.TEMPORAL_PENDING.value


@dataclass(frozen=True)
class TemporalConfig:
    min_observations: int = 3
    min_attempts: int = 3
    min_duration_seconds: float = .5
    evidence_window_seconds: float = 5
    max_votes: int = 256

    def __post_init__(self):
        if min(self.min_observations, self.min_attempts) < 2:
            raise ValueError("One frame/attempt must never confirm")
        if self.max_votes < max(self.min_observations, self.min_attempts):
            raise ValueError("Vote bound must support the configured confirmation count")
        if not 0 < self.min_duration_seconds <= self.evidence_window_seconds or not math.isfinite(self.evidence_window_seconds):
            raise ValueError("Invalid confirmation duration/window")


@dataclass
class TemporalConfirmation:
    config: TemporalConfig = field(default_factory=TemporalConfig)
    candidate: str | None = None
    votes: list = field(default_factory=list)
    last_sequence: int = -1
    last_fingerprint: tuple = ()
    confirmed: bool = False
    confirmed_since: float | None = None

    def reset(self):
        self.candidate, self.votes, self.last_fingerprint, self.confirmed = None, [], (), False
        self.confirmed_since = None
        # Keep the high-water sequence: resetting a streak does not make an old
        # frame independent evidence again. New stream uses a new object.

    def observe(self, user_id, passed, sequence, timestamp, fingerprint):
        if sequence <= self.last_sequence:
            return self.confirmed
        self.last_sequence = sequence
        if not passed:
            if not self.confirmed:
                self.reset()
            return self.confirmed
        if user_id != self.candidate:
            self.reset()
            self.candidate = user_id
        self.votes = [(seq, t) for seq, t in self.votes if 0 <= timestamp-t <= self.config.evidence_window_seconds]
        # Merely aging/reevaluating the same aggregate cannot create a vote.
        if fingerprint == self.last_fingerprint or sequence not in fingerprint:
            return self.confirmed
        self.last_fingerprint = fingerprint
        self.votes.append((sequence, timestamp))
        self.votes = self.votes[-self.config.max_votes:]
        elapsed = timestamp-self.votes[0][1]
        self.confirmed = self.confirmed or (len(self.votes) >= max(self.config.min_observations, self.config.min_attempts)
                          and elapsed >= self.config.min_duration_seconds - 1e-9)
        if self.confirmed and self.confirmed_since is None:
            self.confirmed_since = timestamp
        return self.confirmed

    @property
    def elapsed(self):
        return self.votes[-1][1]-self.votes[0][1] if self.votes else 0
