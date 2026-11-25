"""Local session/trial journal implementing the existing repository boundary."""
from datetime import datetime, timezone
import json
from pathlib import Path
import uuid
from .repository import SessionPersistenceError


class LocalRecognitionSessionRepository:
    def __init__(self, root=None, load_cache=None):
        self.root = Path(root) if root else Path(__file__).resolve().parents[2] / "backend/storage/offline_sessions"
        self._load_cache = load_cache

    def _append(self, kind, payload):
        row = dict(payload, journal_type=kind)
        try:
            self.root.mkdir(parents=True, exist_ok=True)
            with (self.root / "sessions_trials.jsonl").open("a", encoding="utf-8") as target:
                target.write(json.dumps(row, allow_nan=False) + "\n")
        except (OSError, ValueError) as error:
            raise SessionPersistenceError("Local session journal could not be written") from error
        return row

    def create_session(self, payload):
        return self._append("session_started", dict(payload, id=uuid.uuid4().hex,
                            started_at=datetime.now(timezone.utc).isoformat()))

    def update_config_snapshot(self, session_id, *, config_snapshot):
        self._append("configuration", dict(session_id=session_id, config_snapshot=config_snapshot))

    def load_users(self):
        from desktop.recognition.cache import load_adaface_recognition_cache
        cache = (self._load_cache or load_adaface_recognition_cache)()
        return tuple((identity.user_id, identity.full_name) for identity in cache.identities)

    def insert_trial(self, payload):
        return self._append("trial", dict(payload, id=uuid.uuid4().hex))

    def end_session(self, session_id):
        ended_at = datetime.now(timezone.utc).isoformat()
        self._append("session_ended", dict(session_id=session_id, ended_at=ended_at))
        return ended_at
