from pathlib import Path
import sys

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "backend"))

from attendance.models import FinalAttendanceStatus, SuggestionStatus
from attendance.repository import AttendanceRepository, RecordNotFoundError
from attendance.service import (
    FinalizedSessionRecommendationError,
    build_or_refresh_attendance_suggestions,
)


@pytest.fixture
def repository(tmp_path):
    repository = AttendanceRepository(tmp_path / "attendance.db")
    repository.create_class("class_1", "COMP1682", "Final Project")
    for student_id in ("stu_001", "stu_002", "stu_003"):
        repository.upsert_student(student_id, f"Name {student_id}")
        repository.add_class_member("class_1", student_id)
    repository.create_attendance_session(
        "session_1",
        "class_1",
        "gallery-1",
        {"threshold": 0.4, "margin": 0.05},
    )
    return repository


def insert_event(
    repository,
    event_id,
    student_id,
    *,
    status="confirmed_identity",
    generation="gallery-1",
):
    return repository.insert_recognition_event(
        event_id,
        "session_1",
        student_id,
        generation,
        status,
        valid_observations=3,
        confirmation_attempts=3,
    )


def by_student(records):
    return {record.student_id: record for record in records}


def test_no_events_initializes_roster_as_not_observed(repository):
    records = build_or_refresh_attendance_suggestions(repository, "session_1")
    assert len(records) == 3
    assert {record.suggestion_status for record in records} == {
        SuggestionStatus.NOT_OBSERVED
    }
    assert all(record.evidence_event_id is None for record in records)
    assert all(record.final_status is None for record in records)
    assert all(record.final_status is not FinalAttendanceStatus.ABSENT for record in records)


def test_confirmed_event_suggests_present_and_assigns_evidence(repository):
    event = insert_event(repository, "event_1", "stu_002")
    records = by_student(build_or_refresh_attendance_suggestions(repository, "session_1"))
    assert records["stu_002"].suggestion_status is SuggestionStatus.SUGGESTED_PRESENT
    assert records["stu_002"].evidence_event_id == event.event_id
    assert records["stu_001"].suggestion_status is SuggestionStatus.NOT_OBSERVED
    assert records["stu_003"].suggestion_status is SuggestionStatus.NOT_OBSERVED


def test_refresh_is_idempotent_and_supports_live_updates(repository):
    initial = build_or_refresh_attendance_suggestions(repository, "session_1")
    repeated = build_or_refresh_attendance_suggestions(repository, "session_1")
    assert repeated == initial

    insert_event(repository, "event_1", "stu_002")
    first_update = by_student(build_or_refresh_attendance_suggestions(repository, "session_1"))
    assert first_update["stu_002"].suggestion_status is SuggestionStatus.SUGGESTED_PRESENT
    assert first_update["stu_001"].suggestion_status is SuggestionStatus.NOT_OBSERVED

    insert_event(repository, "event_2", "stu_001")
    second_update = by_student(build_or_refresh_attendance_suggestions(repository, "session_1"))
    assert second_update["stu_001"].suggestion_status is SuggestionStatus.SUGGESTED_PRESENT
    assert second_update["stu_002"].suggestion_status is SuggestionStatus.SUGGESTED_PRESENT
    assert len(repository.list_attendance_records("session_1")) == 3


@pytest.mark.parametrize("final_status", list(FinalAttendanceStatus))
def test_refresh_preserves_teacher_decision_and_review_fields(repository, final_status):
    build_or_refresh_attendance_suggestions(repository, "session_1")
    with repository.database.connection() as connection, connection:
        connection.execute(
            """
            UPDATE attendance_records
            SET final_status = ?, reviewed_by = ?, reviewed_at_utc = ?, review_note = ?
            WHERE session_id = ? AND student_id = ?
            """,
            (
                final_status.value,
                "Teacher",
                "2026-01-01T00:00:00+00:00",
                "manual decision",
                "session_1",
                "stu_001",
            ),
        )
    insert_event(repository, "event_1", "stu_001")
    record = by_student(build_or_refresh_attendance_suggestions(repository, "session_1"))[
        "stu_001"
    ]
    assert record.final_status is final_status
    assert record.reviewed_by == "Teacher"
    assert record.reviewed_at_utc == "2026-01-01T00:00:00+00:00"
    assert record.review_note == "manual decision"


def test_missing_gallery_member_remains_not_observed(repository):
    # Gallery membership is deliberately not an input to this service.
    record = by_student(build_or_refresh_attendance_suggestions(repository, "session_1"))[
        "stu_003"
    ]
    assert record.suggestion_status is SuggestionStatus.NOT_OBSERVED
    assert record.final_status is None


def test_session_roster_snapshot_is_authoritative(repository):
    repository.remove_class_member("class_1", "stu_003")
    repository.upsert_student("stu_004", "Later Student")
    repository.add_class_member("class_1", "stu_004")
    records = build_or_refresh_attendance_suggestions(repository, "session_1")
    assert {record.student_id for record in records} == {"stu_001", "stu_002", "stu_003"}


def test_active_and_ended_refresh_allowed_but_finalized_rejected(repository):
    assert len(build_or_refresh_attendance_suggestions(repository, "session_1")) == 3
    repository.end_attendance_session("session_1")
    assert len(build_or_refresh_attendance_suggestions(repository, "session_1")) == 3
    repository.finalize_attendance_session("session_1")
    with pytest.raises(FinalizedSessionRecommendationError):
        build_or_refresh_attendance_suggestions(repository, "session_1")


@pytest.mark.parametrize(
    ("status", "generation"),
    (("unconfirmed", "gallery-1"), ("confirmed_identity", "wrong-gallery")),
)
def test_invalid_or_conflicting_persisted_evidence_needs_review(
    repository, status, generation
):
    event = insert_event(
        repository,
        "event_1",
        "stu_001",
        status=status,
        generation=generation,
    )
    record = by_student(build_or_refresh_attendance_suggestions(repository, "session_1"))[
        "stu_001"
    ]
    assert record.suggestion_status is SuggestionStatus.NEEDS_REVIEW
    assert record.evidence_event_id == event.event_id


def test_suggestion_update_changes_only_recommendation_fields(repository):
    before = build_or_refresh_attendance_suggestions(repository, "session_1")[0]
    insert_event(repository, "event_1", "stu_001")
    after = by_student(build_or_refresh_attendance_suggestions(repository, "session_1"))[
        "stu_001"
    ]
    assert after.suggestion_status is SuggestionStatus.SUGGESTED_PRESENT
    assert after.evidence_event_id == "event_1"
    assert after.attendance_id == before.attendance_id
    assert after.created_at_utc == before.created_at_utc
    assert (after.final_status, after.reviewed_by, after.reviewed_at_utc, after.review_note) == (
        before.final_status,
        before.reviewed_by,
        before.reviewed_at_utc,
        before.review_note,
    )


def test_missing_session_is_rejected(repository):
    with pytest.raises(RecordNotFoundError):
        build_or_refresh_attendance_suggestions(repository, "missing")
