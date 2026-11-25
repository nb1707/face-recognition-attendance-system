from datetime import datetime, timezone
from pathlib import Path
import sqlite3
import sys

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "backend"))

from attendance.models import FinalAttendanceStatus, SessionStatus, SuggestionStatus
from attendance.repository import AttendanceRepository, RecordNotFoundError, RosterViolationError
from attendance.service import (
    AttendanceReviewError,
    IncompleteAttendanceReviewError,
    build_or_refresh_attendance_suggestions,
    finalize_reviewed_attendance_session,
    review_attendance_record,
)


@pytest.fixture
def repository(tmp_path):
    repository = AttendanceRepository(tmp_path / "attendance.db")
    repository.create_class("class_1", "COMP1682", "Final Project")
    for student_id in ("stu_001", "stu_002"):
        repository.upsert_student(student_id, f"Name {student_id}")
        repository.add_class_member("class_1", student_id)
    repository.create_attendance_session(
        "session_1", "class_1", "gallery-1", {"threshold": 0.4, "margin": 0.05}
    )
    build_or_refresh_attendance_suggestions(repository, "session_1")
    return repository


def end_session(repository):
    repository.end_attendance_session("session_1")


def review_all(repository):
    for student_id in ("stu_001", "stu_002"):
        review_attendance_record(
            repository,
            "session_1",
            student_id,
            FinalAttendanceStatus.PRESENT,
            "Teacher",
        )


@pytest.mark.parametrize("final_status", list(FinalAttendanceStatus))
def test_ended_session_accepts_each_final_status(repository, final_status):
    end_session(repository)
    record = review_attendance_record(
        repository,
        "session_1",
        "stu_001",
        final_status,
        "Teacher",
        "manual decision",
    )
    assert record.final_status is final_status
    assert record.reviewed_by == "Teacher"
    assert record.review_note == "manual decision"
    reviewed_at = datetime.fromisoformat(record.reviewed_at_utc)
    assert reviewed_at.utcoffset() == timezone.utc.utcoffset(reviewed_at)


def test_review_preserves_recommendation_evidence_and_recognition_event(repository):
    event = repository.insert_recognition_event(
        "event_1",
        "session_1",
        "stu_001",
        "gallery-1",
        "confirmed_identity",
        valid_observations=3,
        confirmation_attempts=3,
    )
    suggested = build_or_refresh_attendance_suggestions(repository, "session_1")[0]
    end_session(repository)
    reviewed = review_attendance_record(
        repository,
        "session_1",
        "stu_001",
        FinalAttendanceStatus.ABSENT,
        "Teacher",
        "identity disputed",
    )
    assert reviewed.suggestion_status is suggested.suggestion_status
    assert reviewed.suggestion_status is SuggestionStatus.SUGGESTED_PRESENT
    assert reviewed.evidence_event_id == event.event_id
    assert repository.get_recognition_event(event.event_id) == event


def test_first_review_and_status_change_append_correct_audits(repository):
    end_session(repository)
    review_attendance_record(
        repository,
        "session_1",
        "stu_001",
        FinalAttendanceStatus.PRESENT,
        "Teacher A",
        "first reason",
    )
    review_attendance_record(
        repository,
        "session_1",
        "stu_001",
        FinalAttendanceStatus.EXCUSED,
        "Teacher B",
        "medical note",
    )
    audits = repository.list_attendance_audit("session_1")
    first = next(audit for audit in audits if audit.action == "initial_review")
    changed = next(audit for audit in audits if audit.action == "status_changed")
    assert (first.previous_status, first.new_status) == (None, "present")
    assert (first.actor, first.reason) == ("Teacher A", "first reason")
    assert (changed.previous_status, changed.new_status) == ("present", "excused")
    assert (changed.actor, changed.reason) == ("Teacher B", "medical note")
    for audit in audits:
        changed_at = datetime.fromisoformat(audit.changed_at_utc)
        assert changed_at.utcoffset() == timezone.utc.utcoffset(changed_at)


def test_identical_review_is_a_true_no_op(repository):
    end_session(repository)
    first = review_attendance_record(
        repository,
        "session_1",
        "stu_001",
        FinalAttendanceStatus.PRESENT,
        "Teacher",
        "seen in class",
    )
    audit_count = len(repository.list_attendance_audit("session_1"))
    repeated = review_attendance_record(
        repository,
        "session_1",
        "stu_001",
        FinalAttendanceStatus.PRESENT,
        "Teacher",
        "seen in class",
    )
    assert repeated == first
    assert repeated.updated_at_utc == first.updated_at_utc
    assert len(repository.list_attendance_audit("session_1")) == audit_count


def test_note_change_updates_note_and_appends_note_audit(repository):
    end_session(repository)
    review_attendance_record(
        repository,
        "session_1",
        "stu_001",
        FinalAttendanceStatus.PRESENT,
        "Teacher",
        "first note",
    )
    updated = review_attendance_record(
        repository,
        "session_1",
        "stu_001",
        FinalAttendanceStatus.PRESENT,
        "Teacher",
        "corrected note",
    )
    assert updated.review_note == "corrected note"
    audit = repository.list_attendance_audit("session_1")[-1]
    assert audit.action == "note_updated"
    assert (audit.previous_status, audit.new_status) == ("present", "present")
    assert audit.reason == "corrected note"


def test_active_and_finalized_sessions_reject_review(repository):
    with pytest.raises(AttendanceReviewError):
        review_attendance_record(
            repository,
            "session_1",
            "stu_001",
            FinalAttendanceStatus.PRESENT,
            "Teacher",
        )
    end_session(repository)
    review_all(repository)
    finalize_reviewed_attendance_session(repository, "session_1", "Teacher")
    with pytest.raises(AttendanceReviewError):
        review_attendance_record(
            repository,
            "session_1",
            "stu_001",
            FinalAttendanceStatus.ABSENT,
            "Teacher",
        )


@pytest.mark.parametrize(
    "invalid_status",
    (None, "present", "suggested_present", SuggestionStatus.NEEDS_REVIEW),
)
def test_invalid_final_status_is_rejected(repository, invalid_status):
    end_session(repository)
    with pytest.raises(ValueError):
        review_attendance_record(
            repository, "session_1", "stu_001", invalid_status, "Teacher"
        )


@pytest.mark.parametrize("actor", (None, "", "   "))
def test_blank_actor_is_rejected(repository, actor):
    end_session(repository)
    with pytest.raises(ValueError):
        review_attendance_record(
            repository,
            "session_1",
            "stu_001",
            FinalAttendanceStatus.PRESENT,
            actor,
        )


def test_unknown_and_non_roster_students_are_rejected(repository):
    repository.upsert_student("stu_outside", "Outside Student")
    end_session(repository)
    for student_id in ("unknown", "stu_outside"):
        with pytest.raises(RosterViolationError):
            review_attendance_record(
                repository,
                "session_1",
                student_id,
                FinalAttendanceStatus.PRESENT,
                "Teacher",
            )


def test_missing_session_is_rejected(repository):
    with pytest.raises(RecordNotFoundError):
        review_attendance_record(
            repository,
            "missing",
            "stu_001",
            FinalAttendanceStatus.PRESENT,
            "Teacher",
        )
    with pytest.raises(RecordNotFoundError):
        finalize_reviewed_attendance_session(repository, "missing", "Teacher")


def test_finalization_requires_actor(repository):
    end_session(repository)
    with pytest.raises(ValueError):
        finalize_reviewed_attendance_session(repository, "session_1", "   ")


def test_missing_roster_attendance_record_is_rejected(repository):
    with repository.database.connection() as connection, connection:
        connection.execute(
            "DELETE FROM attendance_records WHERE session_id = ? AND student_id = ?",
            ("session_1", "stu_001"),
        )
    end_session(repository)
    with pytest.raises(RecordNotFoundError):
        review_attendance_record(
            repository,
            "session_1",
            "stu_001",
            FinalAttendanceStatus.PRESENT,
            "Teacher",
        )


def test_finalization_requires_ended_session_and_all_records_reviewed(repository):
    with pytest.raises(AttendanceReviewError):
        finalize_reviewed_attendance_session(repository, "session_1", "Teacher")
    end_session(repository)
    with pytest.raises(IncompleteAttendanceReviewError):
        finalize_reviewed_attendance_session(repository, "session_1", "Teacher")
    review_all(repository)
    finalized = finalize_reviewed_attendance_session(repository, "session_1", "Teacher")
    assert finalized.status is SessionStatus.FINALIZED
    with pytest.raises(AttendanceReviewError):
        finalize_reviewed_attendance_session(repository, "session_1", "Teacher")


def test_not_observed_remains_without_final_status_until_teacher_choice(repository):
    before = repository.get_attendance_record("session_1", "stu_001")
    assert before.suggestion_status is SuggestionStatus.NOT_OBSERVED
    assert before.final_status is None
    end_session(repository)
    with pytest.raises(IncompleteAttendanceReviewError):
        finalize_reviewed_attendance_session(repository, "session_1", "Teacher")
    assert repository.get_attendance_record("session_1", "stu_001").final_status is None


def test_audit_insert_failure_rolls_back_record_update(repository):
    end_session(repository)
    first, audit = repository.update_attendance_review(
        "session_1",
        "stu_001",
        FinalAttendanceStatus.PRESENT,
        "Teacher",
        audit_id="fixed-audit-id",
    )
    assert audit is not None
    with pytest.raises(sqlite3.IntegrityError):
        repository.update_attendance_review(
            "session_1",
            "stu_001",
            FinalAttendanceStatus.ABSENT,
            "Teacher",
            audit_id="fixed-audit-id",
        )
    after = repository.get_attendance_record("session_1", "stu_001")
    assert after == first
    assert after.final_status is FinalAttendanceStatus.PRESENT
    assert len(repository.list_attendance_audit("session_1")) == 1
