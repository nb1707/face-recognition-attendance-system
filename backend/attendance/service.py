"""Deterministic attendance recommendations from already-persisted evidence.

This service never runs recognition, interprets recognition scores, or writes a
final attendance decision. The immutable session roster is its population.
"""

from __future__ import annotations

from .models import (
    AttendanceRecord,
    AttendanceSession,
    FinalAttendanceStatus,
    RecognitionEvent,
    SessionStatus,
    SuggestionStatus,
)
from .repository import (
    AttendanceRepository,
    AttendanceRepositoryError,
    InactiveClassError,
    InvalidSessionTransitionError,
    RecordNotFoundError,
    RosterViolationError,
)


class AttendanceAdministrationError(AttendanceRepositoryError):
    """Raised when a class administration command violates domain state."""


def create_attendance_class(
    repository: AttendanceRepository,
    class_id: str,
    class_code: str,
    class_name: str,
):
    """Create a class, accepting an exact repeated request as idempotent."""
    existing = repository.get_class(class_id)
    if existing is not None:
        if existing.class_code == class_code and existing.class_name == class_name:
            return existing
        raise AttendanceAdministrationError(
            f"Class ID already exists with different details: {class_id}"
        )
    code_match = repository.get_class_by_code(class_code)
    if code_match is not None:
        raise AttendanceAdministrationError(
            f"Class code already exists: {class_code}"
        )
    return repository.create_class(class_id, class_code, class_name)


def add_student_to_class(
    repository: AttendanceRepository, class_id: str, student_id: str
) -> None:
    """Add or reactivate one future-roster membership."""
    attendance_class = repository.get_class(class_id)
    if attendance_class is None:
        raise RecordNotFoundError(f"Class does not exist: {class_id}")
    if not attendance_class.active:
        raise InactiveClassError(f"Class is inactive: {class_id}")
    student = repository.get_student(student_id)
    if student is None:
        raise RecordNotFoundError(f"Student does not exist: {student_id}")
    if not student.active:
        raise AttendanceAdministrationError(f"Student is inactive: {student_id}")
    repository.add_class_member(class_id, student_id)


def remove_student_from_class(
    repository: AttendanceRepository, class_id: str, student_id: str
) -> None:
    """Soft-remove a membership without touching student or session snapshots."""
    attendance_class = repository.get_class(class_id)
    if attendance_class is None:
        raise RecordNotFoundError(f"Class does not exist: {class_id}")
    if not attendance_class.active:
        raise InactiveClassError(f"Class is inactive: {class_id}")
    if repository.get_student(student_id) is None:
        raise RecordNotFoundError(f"Student does not exist: {student_id}")
    repository.remove_class_member(class_id, student_id)


CONFIRMED_EVENT_STATUS = "confirmed_identity"


class FinalizedSessionRecommendationError(AttendanceRepositoryError):
    """Raised when recommendation mutation is requested for a finalized session."""


class AttendanceReviewError(AttendanceRepositoryError):
    """Raised when a human attendance decision fails review policy."""


class IncompleteAttendanceReviewError(AttendanceReviewError):
    """Raised when session finalization finds missing or unreviewed records."""


def build_or_refresh_attendance_suggestions(
    repository: AttendanceRepository,
    session_id: str,
) -> list[AttendanceRecord]:
    """Create or refresh one recommendation for each session-roster member.

    Active and ended sessions may be refreshed. Finalized sessions are
    immutable. A valid confirmed event suggests present; no event means not
    observed. An event needs review only when its status is not
    ``confirmed_identity`` or its gallery generation contradicts the session.
    Teacher-owned final/review fields are never changed.
    """
    session = repository.get_attendance_session(session_id)
    if session is None:
        raise RecordNotFoundError(f"Attendance session does not exist: {session_id}")
    if session.status is SessionStatus.FINALIZED:
        raise FinalizedSessionRecommendationError(
            f"Attendance suggestions are immutable for finalized session: {session_id}"
        )

    roster = repository.list_session_roster(session_id)
    repository.create_or_initialize_attendance_records_for_session(session_id)
    events_by_student = {
        event.student_id: event for event in repository.list_recognition_events(session_id)
    }

    refreshed: list[AttendanceRecord] = []
    for member in roster:
        event = events_by_student.get(member.student_id)
        suggestion = _recommendation_for_event(event, session.gallery_generation)
        refreshed.append(
            repository.update_attendance_suggestion(
                session_id=session_id,
                student_id=member.student_id,
                suggestion_status=suggestion,
                evidence_event_id=event.event_id if event is not None else None,
            )
        )
    return refreshed


def _recommendation_for_event(
    event: RecognitionEvent | None,
    session_gallery_generation: str,
) -> SuggestionStatus:
    if event is None:
        return SuggestionStatus.NOT_OBSERVED
    if (
        event.status == CONFIRMED_EVENT_STATUS
        and event.gallery_generation == session_gallery_generation
    ):
        return SuggestionStatus.SUGGESTED_PRESENT
    return SuggestionStatus.NEEDS_REVIEW


def review_attendance_record(
    repository: AttendanceRepository,
    session_id: str,
    student_id: str,
    final_status: FinalAttendanceStatus,
    actor: str,
    reason: str | None = None,
) -> AttendanceRecord:
    """Apply one explicit human decision to an ended attendance session.

    Active sessions remain recognition-only and finalized sessions are
    immutable. The recommendation and recognition evidence are preserved.
    """
    if not isinstance(final_status, FinalAttendanceStatus):
        raise ValueError("final_status must be present, absent, or excused")
    if not isinstance(actor, str) or not actor.strip():
        raise ValueError("actor is required")

    session = repository.get_attendance_session(session_id)
    if session is None:
        raise RecordNotFoundError(f"Attendance session does not exist: {session_id}")
    if session.status is not SessionStatus.ENDED:
        raise AttendanceReviewError(
            f"Attendance review requires an ended session: {session_id} "
            f"({session.status.value})"
        )
    if not repository.session_allows_student(session_id, student_id):
        raise RosterViolationError(
            f"Student {student_id} is not in session roster {session_id}"
        )
    if repository.get_attendance_record(session_id, student_id) is None:
        raise RecordNotFoundError(
            f"Attendance record does not exist: {session_id}/{student_id}"
        )

    try:
        record, _audit = repository.update_attendance_review(
            session_id=session_id,
            student_id=student_id,
            final_status=final_status,
            actor=actor.strip(),
            reason=reason,
        )
    except InvalidSessionTransitionError as error:
        # Preserve the service-level policy error if session state changed
        # between validation and the transactional repository update.
        raise AttendanceReviewError(str(error)) from error
    return record


def finalize_reviewed_attendance_session(
    repository: AttendanceRepository,
    session_id: str,
    actor: str,
) -> AttendanceSession:
    """Finalize an ended session only after every roster member is reviewed."""
    if not isinstance(actor, str) or not actor.strip():
        raise ValueError("actor is required")
    session = repository.get_attendance_session(session_id)
    if session is None:
        raise RecordNotFoundError(f"Attendance session does not exist: {session_id}")
    if session.status is not SessionStatus.ENDED:
        raise AttendanceReviewError(
            f"Finalization requires an ended session: {session_id} "
            f"({session.status.value})"
        )

    roster_ids = {member.student_id for member in repository.list_session_roster(session_id)}
    records = repository.list_attendance_records(session_id)
    records_by_student = {record.student_id: record for record in records}
    missing = sorted(roster_ids - records_by_student.keys())
    unreviewed = sorted(
        student_id
        for student_id in roster_ids
        if student_id in records_by_student
        and records_by_student[student_id].final_status is None
    )
    unexpected = sorted(records_by_student.keys() - roster_ids)
    if missing or unreviewed or unexpected:
        details = []
        if missing:
            details.append(f"missing records: {', '.join(missing)}")
        if unreviewed:
            details.append(f"unreviewed: {', '.join(unreviewed)}")
        if unexpected:
            details.append(f"outside roster: {', '.join(unexpected)}")
        raise IncompleteAttendanceReviewError(
            "Cannot finalize attendance session; " + "; ".join(details)
        )

    try:
        return repository.finalize_attendance_session(session_id)
    except InvalidSessionTransitionError as error:
        raise AttendanceReviewError(str(error)) from error
