"""Desktop transport and presentation helpers for local attendance."""

from .client import AttendanceApiClient, AttendanceClientError
from .state import (
    attendance_counts,
    can_finalize_records,
    decision_label,
    suggestion_label,
)

__all__ = [
    "AttendanceApiClient",
    "AttendanceClientError",
    "attendance_counts",
    "can_finalize_records",
    "decision_label",
    "suggestion_label",
]
