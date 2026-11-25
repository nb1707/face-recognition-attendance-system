"""Read-only import of enrollment identity metadata into attendance students."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from .repository import AttendanceRepository


DEFAULT_USERS_PATH = (
    Path(__file__).resolve().parents[1] / "storage" / "enrollments" / "users.json"
)


def upsert_enrollment_student(
    repository: AttendanceRepository,
    *,
    student_id: str,
    full_name: str,
    role: str,
) -> None:
    """Synchronize one finalized enrollment without changing memberships.

    An existing student's administrative active state is intentionally
    preserved.  Biometric data remains in the enrollment/gallery stores.
    """
    existing = repository.get_student(student_id)
    repository.upsert_student(
        student_id=student_id,
        full_name=full_name,
        role=role,
        active=existing.active if existing is not None else True,
    )


def import_enrollment_users(
    repository: AttendanceRepository,
    users_path: str | Path = DEFAULT_USERS_PATH,
) -> int:
    """Upsert only identities explicitly present in ``users.json``.

    The source file and all enrollment/gallery artifacts are read-only.  The
    returned count is the number of source identities processed.
    """
    path = Path(users_path)
    data: Any = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise ValueError("Enrollment users.json root must be an object")
    users = data.get("users", {})
    if not isinstance(users, dict):
        raise ValueError("Enrollment users.json must contain an object named 'users'")
    for key, metadata in users.items():
        if not isinstance(metadata, dict):
            raise ValueError(f"Invalid enrollment user metadata for {key}")
        student_id = str(metadata.get("user_id") or key).strip()
        if student_id != str(key):
            raise ValueError(f"Enrollment user key and user_id differ for {key}")
        upsert_enrollment_student(
            repository,
            student_id=student_id,
            full_name=str(metadata.get("full_name") or student_id),
            role=str(metadata.get("role") or "student"),
        )
    return len(users)
