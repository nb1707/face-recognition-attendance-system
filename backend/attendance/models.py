"""Lightweight domain types for the attendance repository."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum


class SessionStatus(str, Enum):
    ACTIVE = "active"
    ENDED = "ended"
    FINALIZED = "finalized"


class SuggestionStatus(str, Enum):
    SUGGESTED_PRESENT = "suggested_present"
    NEEDS_REVIEW = "needs_review"
    NOT_OBSERVED = "not_observed"


class FinalAttendanceStatus(str, Enum):
    PRESENT = "present"
    ABSENT = "absent"
    EXCUSED = "excused"


@dataclass(frozen=True)
class Student:
    student_id: str
    full_name: str
    role: str
    active: bool
    created_at_utc: str
    updated_at_utc: str


@dataclass(frozen=True)
class Class:
    class_id: str
    class_code: str
    class_name: str
    active: bool
    created_at_utc: str
    updated_at_utc: str


@dataclass(frozen=True)
class RosterMember:
    student_id: str
    full_name: str
    role: str
    active: bool
    joined_at_utc: str


@dataclass(frozen=True)
class SessionRosterMember:
    session_id: str
    student_id: str
    snapshot_name: str
    snapshot_role: str


@dataclass(frozen=True)
class AttendanceSession:
    session_id: str
    class_id: str
    started_at_utc: str
    ended_at_utc: str | None
    status: SessionStatus
    camera_source: str | None
    gallery_generation: str
    recognition_config_json: str
    operator_name: str | None
    created_at_utc: str
    updated_at_utc: str


@dataclass(frozen=True)
class RecognitionEvent:
    event_id: str
    session_id: str
    student_id: str
    track_id: str | None
    top1_score: float | None
    top2_score: float | None
    margin: float | None
    valid_observations: int | None
    confirmation_attempts: int | None
    confirmation_elapsed_seconds: float | None
    selected_frame_count: int | None
    frame_sequence: int | None
    acquisition_time: float | None
    occurred_at_utc: str
    gallery_generation: str
    status: str
    created_at_utc: str


@dataclass(frozen=True)
class AttendanceRecord:
    attendance_id: str
    session_id: str
    student_id: str
    suggestion_status: SuggestionStatus
    final_status: FinalAttendanceStatus | None
    evidence_event_id: str | None
    reviewed_by: str | None
    reviewed_at_utc: str | None
    review_note: str | None
    created_at_utc: str
    updated_at_utc: str


@dataclass(frozen=True)
class AttendanceAudit:
    audit_id: str
    attendance_id: str
    session_id: str
    student_id: str
    actor: str
    action: str
    previous_status: str | None
    new_status: str | None
    reason: str | None
    changed_at_utc: str
