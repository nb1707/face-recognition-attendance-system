from copy import deepcopy
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import re
import shutil
import uuid

from enrollment.config import POSE_TARGETS, VALID_ROLES
from enrollment.errors import EnrollmentError, enrollment_error


SAFE_USER_ID = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.-]{0,63}$")


def utc_now():
    return datetime.now(timezone.utc).isoformat()


def validate_identity(user_id, full_name, role):
    user_id = str(user_id or "").strip()
    full_name = str(full_name or "").strip()
    role = str(role or "").strip().lower()
    if not SAFE_USER_ID.fullmatch(user_id) or user_id in {".", ".."}:
        raise EnrollmentError("INVALID_USER_ID",
            "User ID must use 1-64 letters, numbers, dots, underscores, or hyphens.", 422)
    if not full_name or len(full_name) > 160:
        raise EnrollmentError("INVALID_FULL_NAME", "Full name is required (maximum 160 characters).", 422)
    if role not in VALID_ROLES:
        raise EnrollmentError("INVALID_ROLE", "Role must be student, teacher, or staff.", 422)
    return user_id, full_name, role


class EnrollmentRepository:
    def __init__(self, root):
        self.root = Path(root).resolve()
        self.root.mkdir(parents=True, exist_ok=True)
        self.users_path = self.root / "users.json"

    def create_session(self, user_id, full_name, role):
        session_id = uuid.uuid4().hex
        now = utc_now()
        session = {
            "schema_version": 1,
            "session_id": session_id,
            "user_id": user_id,
            "full_name": full_name,
            "role": role,
            "state": "CAPTURING",
            "created_at": now,
            "updated_at": now,
            "targets": dict(POSE_TARGETS),
            "samples": [],
            "rejected": {"total": 0, "by_code": {}, "recent": []},
            "finalization": {"status": "not_started"},
        }
        directory = self.session_dir(session_id)
        directory.mkdir(parents=False, exist_ok=False)
        (directory / "samples").mkdir()
        self.save_session(session)
        return session

    def session_dir(self, session_id):
        if not re.fullmatch(r"[0-9a-f]{32}", str(session_id or "")):
            raise enrollment_error("INVALID_SESSION", status_code=404)
        return self._contained(self.root, str(session_id))

    def sample_dir(self, session_id, sample_id):
        if not re.fullmatch(r"[0-9a-f]{32}", str(sample_id or "")):
            raise EnrollmentError("SAMPLE_NOT_FOUND", "Enrollment sample was not found.", 404)
        return self._contained(self.session_dir(session_id) / "samples", str(sample_id))

    def load_session(self, session_id):
        path = self.session_dir(session_id) / "session.json"
        try:
            session = json.loads(path.read_text(encoding="utf-8"))
        except (FileNotFoundError, json.JSONDecodeError, OSError):
            raise enrollment_error("INVALID_SESSION", status_code=404)
        if session.get("session_id") != session_id:
            raise enrollment_error("INVALID_SESSION", status_code=404)
        return session

    def save_session(self, session):
        session = deepcopy(session)
        session["updated_at"] = utc_now()
        path = self.session_dir(session["session_id"]) / "session.json"
        self._atomic_json(path, session)
        return session

    def store_sample(self, session, crop_bytes, aligned_bytes, metadata):
        sample_id = metadata["sample_id"]
        directory = self.sample_dir(session["session_id"], sample_id)
        try:
            directory.mkdir(parents=False, exist_ok=False)
            self._atomic_bytes(directory / "crop.jpg", crop_bytes)
            self._atomic_bytes(directory / "aligned.jpg", aligned_bytes)
            self._atomic_json(directory / "metadata.json", metadata)
        except (OSError, ValueError) as error:
            if directory.exists():
                shutil.rmtree(directory, ignore_errors=True)
            raise enrollment_error("STORAGE_ERROR", status_code=500, details={"reason": str(error)})
        session["samples"].append(metadata)
        return self.save_session(session)

    def update_sample_metadata(self, session_id, metadata):
        directory = self.sample_dir(session_id, metadata["sample_id"])
        self._atomic_json(directory / "metadata.json", metadata)

    def delete_sample(self, session, sample_id):
        matches = [row for row in session["samples"] if row["sample_id"] == sample_id]
        if not matches:
            raise EnrollmentError("SAMPLE_NOT_FOUND", "Enrollment sample was not found.", 404)
        directory = self.sample_dir(session["session_id"], sample_id)
        if directory.exists():
            shutil.rmtree(directory)
        session["samples"] = [row for row in session["samples"] if row["sample_id"] != sample_id]
        return self.save_session(session)

    def record_rejection(self, session, code, details=None):
        rejected = session.setdefault("rejected", {"total": 0, "by_code": {}, "recent": []})
        rejected["total"] += 1
        rejected["by_code"][code] = rejected["by_code"].get(code, 0) + 1
        rejected["recent"].append({"code": code, "at": utc_now(), "details": details or {}})
        rejected["recent"] = rejected["recent"][-20:]
        return self.save_session(session)

    def preview_path(self, session_id, sample_id):
        session = self.load_session(session_id)
        if not any(row["sample_id"] == sample_id for row in session["samples"]):
            raise EnrollmentError("SAMPLE_NOT_FOUND", "Enrollment sample was not found.", 404)
        path = self.sample_dir(session_id, sample_id) / "crop.jpg"
        if not path.is_file():
            raise EnrollmentError("SAMPLE_NOT_FOUND", "Enrollment preview was not found.", 404)
        return path

    def cancel(self, session_id):
        session = self.load_session(session_id)
        if session["state"] == "COMPLETE":
            raise enrollment_error("SESSION_FINALIZED", status_code=409)
        directory = self.session_dir(session_id)
        shutil.rmtree(directory)

    def load_users(self):
        try:
            data = json.loads(self.users_path.read_text(encoding="utf-8"))
        except FileNotFoundError:
            return {"schema_version": 1, "users": {}}
        except (json.JSONDecodeError, OSError) as error:
            raise enrollment_error("STORAGE_ERROR", status_code=500, details={"reason": str(error)})
        return data

    def user_exists(self, user_id):
        return user_id in self.load_users().get("users", {})

    def complete_user(self, session, generation):
        data = self.load_users()
        now = utc_now()
        previous = data.setdefault("users", {}).get(session["user_id"], {})
        data["users"][session["user_id"]] = {
            "user_id": session["user_id"],
            "full_name": session["full_name"],
            "role": session["role"],
            "latest_completed_enrollment_session": session["session_id"],
            "gallery_generation": generation,
            "created_at": previous.get("created_at", now),
            "updated_at": now,
        }
        self._atomic_json(self.users_path, data)

    def remove_users(self, user_ids):
        data = self.load_users()
        users = data.setdefault("users", {})
        removed = tuple(user_id for user_id in user_ids if user_id in users)
        if removed:
            for user_id in removed:
                users.pop(user_id)
            self._atomic_json(self.users_path, data)
        return removed

    def snapshot_users(self):
        try:
            return self.users_path.read_bytes()
        except FileNotFoundError:
            return None

    def restore_users(self, snapshot):
        if snapshot is None:
            self.users_path.unlink(missing_ok=True)
        else:
            self._atomic_bytes(self.users_path, snapshot)

    @staticmethod
    def progress(session):
        values = {pose: 0 for pose in POSE_TARGETS}
        for sample in session.get("samples", []):
            if sample.get("pose") in values:
                values[sample["pose"]] += 1
        return values

    @staticmethod
    def _contained(root, relative):
        root = Path(root).resolve()
        result = (root / relative).resolve()
        if not result.is_relative_to(root):
            raise enrollment_error("STORAGE_ERROR", status_code=500, details={"reason": "path_escape"})
        return result

    @staticmethod
    def _atomic_bytes(path, data):
        temporary = path.with_name(f".{path.name}.{uuid.uuid4().hex}.tmp")
        try:
            temporary.write_bytes(data)
            os.replace(temporary, path)
        finally:
            if temporary.exists():
                temporary.unlink()

    @classmethod
    def _atomic_json(cls, path, data):
        payload = json.dumps(data, indent=2, ensure_ascii=False).encode("utf-8")
        cls._atomic_bytes(path, payload)
