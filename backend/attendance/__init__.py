"""Authoritative local academic, session, and attendance persistence."""

from .database import AttendanceDatabase, DEFAULT_DATABASE_PATH
from .models import FinalAttendanceStatus, SessionStatus, SuggestionStatus
from .repository import (
    AttendanceRepository,
    AttendanceRepositoryError,
    EmptyRosterError,
    InactiveClassError,
    InvalidSessionTransitionError,
    RecordNotFoundError,
    RosterViolationError,
)
from .import_enrollment_users import import_enrollment_users
from .service import (
    AttendanceReviewError,
    FinalizedSessionRecommendationError,
    IncompleteAttendanceReviewError,
    build_or_refresh_attendance_suggestions,
    finalize_reviewed_attendance_session,
    review_attendance_record,
)

__all__ = [
    "AttendanceDatabase",
    "AttendanceRepository",
    "AttendanceRepositoryError",
    "AttendanceReviewError",
    "DEFAULT_DATABASE_PATH",
    "EmptyRosterError",
    "FinalAttendanceStatus",
    "FinalizedSessionRecommendationError",
    "IncompleteAttendanceReviewError",
    "InactiveClassError",
    "InvalidSessionTransitionError",
    "import_enrollment_users",
    "RecordNotFoundError",
    "RosterViolationError",
    "SessionStatus",
    "SuggestionStatus",
    "build_or_refresh_attendance_suggestions",
    "finalize_reviewed_attendance_session",
    "review_attendance_record",
]
