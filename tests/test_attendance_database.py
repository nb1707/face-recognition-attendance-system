import hashlib
import json
from pathlib import Path
import sqlite3
import sys

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "backend"))

from attendance.database import AttendanceDatabase
from attendance.import_enrollment_users import import_enrollment_users
from attendance.models import FinalAttendanceStatus, SessionStatus, SuggestionStatus
from attendance.repository import (
    AttendanceRepository,
    AttendanceRepositoryError,
    EmptyRosterError,
    InvalidSessionTransitionError,
    RosterViolationError,
)


@pytest.fixture
def repository(tmp_path):
    return AttendanceRepository(tmp_path / "attendance.db")


def add_student_and_class(repository, student_ids=("stu_001", "stu_002")):
    repository.create_class("class_1", "COMP1682", "Final Project")
    for student_id in student_ids:
        repository.upsert_student(student_id, f"Name {student_id}")
        repository.add_class_member("class_1", student_id)


def create_session(repository, session_id="session_1"):
    return repository.create_attendance_session(
        session_id=session_id,
        class_id="class_1",
        gallery_generation="gallery-generation-1",
        recognition_config={"threshold": 0.4, "nested": {"margin": 0.05}},
        camera_source="camera-0",
        operator_name="Teacher",
    )


def insert_event(repository, event_id="event_1", student_id="stu_001"):
    return repository.insert_recognition_event(
        event_id=event_id,
        session_id="session_1",
        student_id=student_id,
        gallery_generation="gallery-generation-1",
        status="confirmed_identity",
        track_id=7,
        top1_score=0.81,
        top2_score=0.32,
        margin=0.49,
        valid_observations=3,
        confirmation_attempts=3,
        confirmation_elapsed_seconds=0.6,
        selected_frame_count=3,
        frame_sequence=42,
        acquisition_time=12.5,
    )


def test_schema_initialization_is_idempotent_and_foreign_keys_are_enabled(tmp_path):
    database = AttendanceDatabase(tmp_path / "nested" / "attendance.db")
    database.initialize_schema()
    database.initialize_schema()

    with database.connection() as connection:
        tables = {
            row["name"]
            for row in connection.execute(
                "SELECT name FROM sqlite_master WHERE type = ?", ("table",)
            )
        }
        assert {
            "students",
            "classes",
            "class_members",
            "attendance_sessions",
            "session_roster",
            "recognition_events",
            "attendance_records",
            "attendance_audit",
        } <= tables
        assert connection.execute("PRAGMA foreign_keys").fetchone()[0] == 1
        with pytest.raises(sqlite3.IntegrityError):
            connection.execute(
                """
                INSERT INTO class_members (class_id, student_id, joined_at_utc, active)
                VALUES (?, ?, ?, ?)
                """,
                ("missing", "missing", "2026-01-01T00:00:00+00:00", 1),
            )


def test_student_upsert_list_and_active_flag(repository):
    created = repository.upsert_student("stu_001", "First Name")
    assert created.full_name == "First Name"
    assert created.active is True
    created_at = created.created_at_utc

    updated = repository.upsert_student("stu_001", "Updated Name", role="staff")
    assert updated.full_name == "Updated Name"
    assert updated.role == "staff"
    assert updated.created_at_utc == created_at
    repository.set_student_active("stu_001", False)
    assert repository.get_student("stu_001").active is False
    assert repository.list_students(active_only=True) == []
    assert len(repository.list_students()) == 1


def test_classes_membership_uniqueness_and_inactive_roster(repository):
    first = repository.create_class("class_1", "CODE-1", "Class One")
    assert repository.get_class("class_1") == first
    assert repository.get_class_by_code("CODE-1") == first
    updated = repository.update_class("class_1", class_name="Renamed")
    assert updated.class_name == "Renamed"
    with pytest.raises(sqlite3.IntegrityError):
        repository.create_class("class_2", "CODE-1", "Duplicate Code")

    repository.upsert_student("stu_001", "Student")
    repository.add_class_member("class_1", "stu_001")
    repository.add_class_member("class_1", "stu_001")
    assert repository.is_class_member("class_1", "stu_001")
    assert [row.student_id for row in repository.list_class_roster("class_1")] == ["stu_001"]
    assert repository.remove_class_member("class_1", "stu_001") is True
    assert repository.list_class_roster("class_1") == []
    assert repository.is_class_member("class_1", "stu_001", active_only=False)

    repository.set_class_active("class_1", False)
    assert repository.list_classes() == []
    assert len(repository.list_classes(active_only=False)) == 1


def test_membership_foreign_keys_reject_unknown_student_and_class(repository):
    repository.create_class("class_1", "CODE-1", "Class One")
    repository.upsert_student("stu_001", "Student")
    with pytest.raises(sqlite3.IntegrityError):
        repository.add_class_member("class_1", "unknown")
    with pytest.raises(sqlite3.IntegrityError):
        repository.add_class_member("unknown", "stu_001")


def test_session_snapshots_active_roster_and_remains_stable(repository):
    add_student_and_class(repository)
    repository.remove_class_member("class_1", "stu_002")
    session = create_session(repository)

    assert session.status is SessionStatus.ACTIVE
    assert session.gallery_generation == "gallery-generation-1"
    assert json.loads(session.recognition_config_json) == {
        "nested": {"margin": 0.05},
        "threshold": 0.4,
    }
    assert session.recognition_config_json == (
        '{"nested":{"margin":0.05},"threshold":0.4}'
    )
    roster = repository.list_session_roster("session_1")
    assert [(row.student_id, row.snapshot_name) for row in roster] == [
        ("stu_001", "Name stu_001")
    ]
    assert repository.get_session_allowed_student_ids("session_1") == ("stu_001",)
    assert repository.session_allows_student("session_1", "stu_001")
    assert not repository.session_allows_student("session_1", "stu_002")

    repository.remove_class_member("class_1", "stu_001")
    repository.add_class_member("class_1", "stu_002")
    repository.upsert_student("stu_001", "Later Name")
    assert [(row.student_id, row.snapshot_name) for row in repository.list_session_roster("session_1")] == [
        ("stu_001", "Name stu_001")
    ]


def test_session_creation_rejects_empty_active_roster(repository):
    repository.create_class("class_1", "CODE-1", "Empty")
    with pytest.raises(EmptyRosterError):
        create_session(repository)
    assert repository.get_attendance_session("session_1") is None


def test_session_state_transitions_are_strict(repository):
    add_student_and_class(repository, ("stu_001",))
    create_session(repository)
    with pytest.raises(InvalidSessionTransitionError):
        repository.finalize_attendance_session("session_1")
    ended = repository.end_attendance_session("session_1")
    assert ended.status is SessionStatus.ENDED
    assert ended.ended_at_utc is not None
    finalized = repository.finalize_attendance_session("session_1")
    assert finalized.status is SessionStatus.FINALIZED
    with pytest.raises(InvalidSessionTransitionError):
        repository.end_attendance_session("session_1")


def test_recognition_event_is_roster_guarded_unique_and_active_only(repository):
    add_student_and_class(repository)
    repository.remove_class_member("class_1", "stu_002")
    create_session(repository)
    event = insert_event(repository)
    assert event.student_id == "stu_001"
    assert event.track_id == "7"
    assert repository.get_recognition_event("event_1") == event
    assert repository.get_recognition_event_for_student("session_1", "stu_001") == event
    assert repository.list_recognition_events("session_1") == [event]

    with pytest.raises(RosterViolationError):
        insert_event(repository, event_id="event_outside", student_id="stu_002")
    with pytest.raises(sqlite3.IntegrityError):
        insert_event(repository, event_id="event_duplicate", student_id="stu_001")

    repository.end_attendance_session("session_1")
    with pytest.raises(AttendanceRepositoryError):
        repository.insert_recognition_event(
            "event_after_end",
            "session_1",
            "stu_001",
            "gallery-generation-1",
            "confirmed_identity",
        )


def test_event_rejects_missing_session(repository):
    repository.upsert_student("stu_001", "Student")
    with pytest.raises(AttendanceRepositoryError):
        repository.insert_recognition_event(
            "event_1", "missing", "stu_001", "generation", "confirmed_identity"
        )


def test_recognition_event_table_enforces_foreign_keys(repository):
    with repository.database.connection() as connection:
        with pytest.raises(sqlite3.IntegrityError):
            connection.execute(
                """
                INSERT INTO recognition_events (
                    event_id, session_id, student_id, occurred_at_utc,
                    gallery_generation, status, created_at_utc
                ) VALUES (?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    "event_bad_fk",
                    "missing_session",
                    "missing_student",
                    "2026-01-01T00:00:00+00:00",
                    "generation",
                    "confirmed_identity",
                    "2026-01-01T00:00:00+00:00",
                ),
            )


def test_attendance_initialization_is_idempotent_and_never_marks_absent(repository):
    add_student_and_class(repository)
    create_session(repository)
    first = repository.create_or_initialize_attendance_records_for_session("session_1")
    second = repository.create_or_initialize_attendance_records_for_session("session_1")
    assert first == second
    assert len(first) == 2
    for record in first:
        assert record.suggestion_status is SuggestionStatus.NOT_OBSERVED
        assert record.final_status is None
        assert record.final_status is not FinalAttendanceStatus.ABSENT
        assert repository.get_attendance_record("session_1", record.student_id) == record


def test_audit_append_and_foreign_keys(repository):
    add_student_and_class(repository, ("stu_001",))
    create_session(repository)
    record = repository.create_or_initialize_attendance_records_for_session("session_1")[0]
    audit = repository.append_attendance_audit(
        "audit_1",
        record.attendance_id,
        "session_1",
        "stu_001",
        "Teacher",
        "reviewed",
        previous_status="not_observed",
        new_status="present",
        reason="Manual review",
    )
    assert audit.new_status == "present"
    with pytest.raises(sqlite3.IntegrityError):
        repository.append_attendance_audit(
            "audit_bad", "missing", "session_1", "stu_001", "Teacher", "reviewed"
        )


def test_enrollment_import_is_idempotent_updates_metadata_and_touches_no_artifacts(
    repository, tmp_path
):
    enrollment = tmp_path / "enrollments"
    gallery = tmp_path / "local_gallery"
    enrollment.mkdir()
    gallery.mkdir()
    users_path = enrollment / "users.json"
    current_path = gallery / "current.json"
    prototype_path = gallery / "prototype.npy"
    users = {
        "schema_version": 1,
        "users": {
            "stu_001": {
                "user_id": "stu_001",
                "full_name": "Original Name",
                "role": "student",
            }
        },
    }
    users_path.write_text(json.dumps(users, indent=2), encoding="utf-8")
    current_path.write_bytes(b'{"generation":"frozen"}\n')
    prototype_path.write_bytes(b"frozen prototype bytes")

    def digest(path):
        return hashlib.sha256(path.read_bytes()).hexdigest()

    original_users = users_path.read_bytes()
    gallery_digests = (digest(current_path), digest(prototype_path))
    assert import_enrollment_users(repository, users_path) == 1
    assert import_enrollment_users(repository, users_path) == 1
    assert users_path.read_bytes() == original_users
    assert (digest(current_path), digest(prototype_path)) == gallery_digests
    assert repository.get_student("stu_001").full_name == "Original Name"

    users["users"]["stu_001"]["full_name"] = "Updated Name"
    users["users"]["stu_001"]["role"] = "staff"
    users_path.write_text(json.dumps(users, indent=2), encoding="utf-8")
    source_after_update = users_path.read_bytes()
    import_enrollment_users(repository, users_path)
    imported = repository.get_student("stu_001")
    assert (imported.full_name, imported.role) == ("Updated Name", "staff")
    assert users_path.read_bytes() == source_after_update
    assert (digest(current_path), digest(prototype_path)) == gallery_digests
