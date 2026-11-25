"""Confirmed-only local attendance boundary, separate from temporary track IDs."""
from dataclasses import asdict, dataclass
from pathlib import Path
import json


@dataclass(frozen=True)
class ConfirmedIdentityEvent:
    session_id: str
    gallery_generation: str
    track_id: int
    user_id: str
    top1_score: float
    top2_score: float | None
    margin: float | None
    valid_observations: int
    confirmation_attempts: int
    elapsed_confirmation_time: float
    selected_frame_count: int
    acquisition_time: float
    frame_sequence: int
    status: str = "confirmed_identity"


class LocalAttendanceSink:
    """Dedup exactly once per (logical session, user), not per track.

    Optional JSONL persists dedup across restarts. One coordinator owns the sink;
    this is not a multi-process database replacement.
    """
    def __init__(self, path=None, session_id=None):
        self.path = Path(path) if path else None
        self.session_id = session_id
        self.events = []
        self.seen = set()
        if self.path and self.path.exists():
            with self.path.open(encoding="utf-8") as source:
                for line in source:
                    row = json.loads(line)
                    if self.session_id is None or row['session_id'] == self.session_id:
                        self.seen.add((row['session_id'], row['user_id']))

    def accept(self, event):
        if not isinstance(event, ConfirmedIdentityEvent) or event.status != "confirmed_identity" or event.valid_observations < 2:
            raise ValueError("Only confirmed identity events may reach attendance")
        if self.session_id is not None and event.session_id != self.session_id:
            raise ValueError("Confirmed event belongs to another session")
        key = (event.session_id, event.user_id)
        if key in self.seen:
            return False
        if self.path:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            with self.path.open("a", encoding="utf-8") as target:
                target.write(json.dumps(asdict(event), allow_nan=False) + "\n")
        self.seen.add(key)
        self.events.append(event)
        return True
