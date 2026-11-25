"""Thin FastAPI transport adapter for the local attendance domain."""

from __future__ import annotations

import sqlite3
from typing import Any, Callable, TypeVar
import uuid

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

from .models import AttendanceRecord, AttendanceSession, FinalAttendanceStatus
from .repository import (
    AttendanceRepository,
    AttendanceRepositoryError,
    EmptyRosterError,
    InactiveClassError,
    InvalidSessionTransitionError,
    RecordNotFoundError,
    RosterViolationError,
)
from .service import (
    AttendanceAdministrationError,
    AttendanceReviewError,
    FinalizedSessionRecommendationError,
    IncompleteAttendanceReviewError,
    add_student_to_class,
    build_or_refresh_attendance_suggestions,
    create_attendance_class,
    finalize_reviewed_attendance_session,
    remove_student_from_class,
    review_attendance_record,
)


class CreateAttendanceSessionRequest(BaseModel):
    class_id: str
    gallery_generation: str
    recognition_config: dict[str, Any]
    camera_source: str | None = None
    operator_name: str | None = None


class ReviewAttendanceRequest(BaseModel):
    final_status: str
    actor: str
    reason: str | None = None


class FinalizeAttendanceSessionRequest(BaseModel):
    actor: str


class ClassResponse(BaseModel):
    class_id: str
    class_code: str
    class_name: str
    active: bool


class StudentResponse(BaseModel):
    student_id: str
    full_name: str
    role: str
    active: bool


class CreateClassRequest(BaseModel):
    class_id: str
    class_code: str
    class_name: str


class RosterMemberResponse(BaseModel):
    student_id: str
    full_name: str
    role: str


class SessionSummaryResponse(BaseModel):
    session_id: str
    class_id: str
    status: str
    started_at_utc: str
    roster_count: int


class SessionDetailResponse(SessionSummaryResponse):
    ended_at_utc: str | None
    camera_source: str | None
    gallery_generation: str
    operator_name: str | None
    recognition_event_count: int
    attendance_record_count: int


class AttendanceRecordResponse(BaseModel):
    student_id: str
    snapshot_name: str
    suggestion_status: str
    final_status: str | None
    evidence_event_id: str | None
    reviewed_by: str | None
    reviewed_at_utc: str | None
    review_note: str | None
    event_exists: bool
    top1_score: float | None
    margin: float | None
    occurred_at_utc: str | None


class EndAttendanceSessionResponse(BaseModel):
    session: SessionDetailResponse
    records: list[AttendanceRecordResponse]


class AttendanceAuditResponse(BaseModel):
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


T = TypeVar("T")


def _domain_call(function: Callable[..., T], *args: Any, **kwargs: Any) -> T:
    """Translate domain failures without leaking SQLite tracebacks."""
    try:
        return function(*args, **kwargs)
    except RecordNotFoundError as error:
        raise HTTPException(status_code=404, detail=str(error)) from error
    except (ValueError, TypeError) as error:
        raise HTTPException(status_code=400, detail=str(error)) from error
    except (
        EmptyRosterError,
        InactiveClassError,
        InvalidSessionTransitionError,
        RosterViolationError,
        AttendanceAdministrationError,
        AttendanceReviewError,
        IncompleteAttendanceReviewError,
        FinalizedSessionRecommendationError,
    ) as error:
        raise HTTPException(status_code=409, detail=str(error)) from error
    except sqlite3.IntegrityError as error:
        raise HTTPException(status_code=409, detail="Attendance data conflict") from error
    except AttendanceRepositoryError as error:
        raise HTTPException(status_code=409, detail=str(error)) from error


def _required_text(value: str, name: str) -> str:
    cleaned = value.strip() if isinstance(value, str) else ""
    if not cleaned:
        raise HTTPException(status_code=400, detail=f"{name} is required")
    return cleaned


def _session_summary(
    repository: AttendanceRepository, session: AttendanceSession
) -> SessionSummaryResponse:
    return SessionSummaryResponse(
        session_id=session.session_id,
        class_id=session.class_id,
        status=session.status.value,
        started_at_utc=session.started_at_utc,
        roster_count=len(repository.list_session_roster(session.session_id)),
    )


def _session_detail(
    repository: AttendanceRepository, session: AttendanceSession
) -> SessionDetailResponse:
    summary = _session_summary(repository, session)
    return SessionDetailResponse(
        **summary.model_dump(),
        ended_at_utc=session.ended_at_utc,
        camera_source=session.camera_source,
        gallery_generation=session.gallery_generation,
        operator_name=session.operator_name,
        recognition_event_count=len(repository.list_recognition_events(session.session_id)),
        attendance_record_count=len(repository.list_attendance_records(session.session_id)),
    )


def _record_responses(
    repository: AttendanceRepository,
    session_id: str,
    records: list[AttendanceRecord] | None = None,
) -> list[AttendanceRecordResponse]:
    roster = {member.student_id: member for member in repository.list_session_roster(session_id)}
    events = {
        event.student_id: event for event in repository.list_recognition_events(session_id)
    }
    source_records = records if records is not None else repository.list_attendance_records(session_id)
    responses = []
    for record in source_records:
        member = roster.get(record.student_id)
        if member is None:
            continue
        event = events.get(record.student_id)
        responses.append(
            AttendanceRecordResponse(
                student_id=record.student_id,
                snapshot_name=member.snapshot_name,
                suggestion_status=record.suggestion_status.value,
                final_status=record.final_status.value if record.final_status else None,
                evidence_event_id=record.evidence_event_id,
                reviewed_by=record.reviewed_by,
                reviewed_at_utc=record.reviewed_at_utc,
                review_note=record.review_note,
                event_exists=event is not None,
                top1_score=event.top1_score if event else None,
                margin=event.margin if event else None,
                occurred_at_utc=event.occurred_at_utc if event else None,
            )
        )
    return responses


def create_attendance_router(repository: AttendanceRepository) -> APIRouter:
    """Create an attendance router bound to one injectable repository instance."""
    router = APIRouter(prefix="/attendance", tags=["local-attendance"])

    def require_session(session_id: str) -> AttendanceSession:
        session = repository.get_attendance_session(session_id)
        if session is None:
            raise HTTPException(status_code=404, detail=f"Session does not exist: {session_id}")
        return session

    @router.get("/classes", response_model=list[ClassResponse])
    def list_classes() -> list[ClassResponse]:
        return [
            ClassResponse(
                class_id=item.class_id,
                class_code=item.class_code,
                class_name=item.class_name,
                active=item.active,
            )
            for item in repository.list_classes(active_only=True)
        ]

    @router.get("/students", response_model=list[StudentResponse])
    def list_students() -> list[StudentResponse]:
        return [
            StudentResponse(
                student_id=item.student_id,
                full_name=item.full_name,
                role=item.role,
                active=item.active,
            )
            for item in repository.list_students(active_only=True)
        ]

    @router.post("/classes", response_model=ClassResponse, status_code=201)
    def create_class(request: CreateClassRequest) -> ClassResponse:
        item = _domain_call(
            create_attendance_class,
            repository,
            _required_text(request.class_id, "class_id"),
            _required_text(request.class_code, "class_code"),
            _required_text(request.class_name, "class_name"),
        )
        return ClassResponse(
            class_id=item.class_id,
            class_code=item.class_code,
            class_name=item.class_name,
            active=item.active,
        )

    @router.get("/classes/{class_id}/roster", response_model=list[RosterMemberResponse])
    def get_class_roster(class_id: str) -> list[RosterMemberResponse]:
        if repository.get_class(class_id) is None:
            raise HTTPException(status_code=404, detail=f"Class does not exist: {class_id}")
        return [
            RosterMemberResponse(
                student_id=member.student_id,
                full_name=member.full_name,
                role=member.role,
            )
            for member in repository.list_class_roster(class_id, active_only=True)
        ]

    @router.post(
        "/classes/{class_id}/students/{student_id}",
        response_model=RosterMemberResponse,
    )
    def add_class_student(class_id: str, student_id: str) -> RosterMemberResponse:
        _domain_call(add_student_to_class, repository, class_id, student_id)
        student = repository.get_student(student_id)
        if student is None:  # Guard for type narrowing; service validated it.
            raise HTTPException(status_code=404, detail=f"Student does not exist: {student_id}")
        return RosterMemberResponse(
            student_id=student.student_id,
            full_name=student.full_name,
            role=student.role,
        )

    @router.delete(
        "/classes/{class_id}/students/{student_id}",
        status_code=204,
    )
    def remove_class_student(class_id: str, student_id: str) -> None:
        _domain_call(remove_student_from_class, repository, class_id, student_id)

    @router.post("/sessions", response_model=SessionSummaryResponse, status_code=201)
    def create_session(request: CreateAttendanceSessionRequest) -> SessionSummaryResponse:
        session = _domain_call(
            repository.create_attendance_session,
            session_id=uuid.uuid4().hex,
            class_id=_required_text(request.class_id, "class_id"),
            gallery_generation=_required_text(
                request.gallery_generation, "gallery_generation"
            ),
            recognition_config=request.recognition_config,
            camera_source=request.camera_source,
            operator_name=request.operator_name,
        )
        _domain_call(
            repository.create_or_initialize_attendance_records_for_session,
            session.session_id,
        )
        return _session_summary(repository, session)

    @router.get("/sessions/{session_id}", response_model=SessionDetailResponse)
    def get_session(session_id: str) -> SessionDetailResponse:
        return _session_detail(repository, require_session(session_id))

    @router.post(
        "/sessions/{session_id}/refresh-suggestions",
        response_model=list[AttendanceRecordResponse],
    )
    def refresh_suggestions(session_id: str) -> list[AttendanceRecordResponse]:
        records = _domain_call(
            build_or_refresh_attendance_suggestions, repository, session_id
        )
        return _record_responses(repository, session_id, records)

    @router.get(
        "/sessions/{session_id}/records",
        response_model=list[AttendanceRecordResponse],
    )
    def list_records(session_id: str) -> list[AttendanceRecordResponse]:
        require_session(session_id)
        return _record_responses(repository, session_id)

    @router.post(
        "/sessions/{session_id}/end", response_model=EndAttendanceSessionResponse
    )
    def end_session(session_id: str) -> EndAttendanceSessionResponse:
        # End first to freeze event ingestion. Recommendation refresh explicitly
        # supports ended sessions and then consumes the final persisted evidence.
        session = _domain_call(repository.end_attendance_session, session_id)
        records = _domain_call(
            build_or_refresh_attendance_suggestions, repository, session_id
        )
        return EndAttendanceSessionResponse(
            session=_session_detail(repository, session),
            records=_record_responses(repository, session_id, records),
        )

    @router.post(
        "/sessions/{session_id}/students/{student_id}/review",
        response_model=AttendanceRecordResponse,
    )
    def review_student(
        session_id: str, student_id: str, request: ReviewAttendanceRequest
    ) -> AttendanceRecordResponse:
        try:
            final_status = FinalAttendanceStatus(request.final_status)
        except ValueError as error:
            raise HTTPException(
                status_code=400, detail="final_status must be present, absent, or excused"
            ) from error
        record = _domain_call(
            review_attendance_record,
            repository,
            session_id,
            student_id,
            final_status,
            request.actor,
            request.reason,
        )
        return _record_responses(repository, session_id, [record])[0]

    @router.post("/sessions/{session_id}/finalize", response_model=SessionDetailResponse)
    def finalize_session(
        session_id: str, request: FinalizeAttendanceSessionRequest
    ) -> SessionDetailResponse:
        session = _domain_call(
            finalize_reviewed_attendance_session,
            repository,
            session_id,
            request.actor,
        )
        return _session_detail(repository, session)

    @router.get(
        "/sessions/{session_id}/audit", response_model=list[AttendanceAuditResponse]
    )
    def list_audit(session_id: str) -> list[AttendanceAuditResponse]:
        require_session(session_id)
        return [
            AttendanceAuditResponse(**audit.__dict__)
            for audit in repository.list_attendance_audit(session_id)
        ]

    return router
