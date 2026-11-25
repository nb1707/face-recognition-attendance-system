"""Attendance-session adapters around the frozen recognition boundary.

The global :class:`recognition.local_gallery.Gallery` remains immutable.  A
session view reuses its immutable ``Identity``/``Prototype`` objects and changes
only the identities tuple supplied to ``RecognitionCoordinator``.  No gallery
manifest, embedding, or ``current.json`` file is written.
"""

from __future__ import annotations

from dataclasses import dataclass
import sqlite3
from typing import Callable, Iterable
import uuid

from recognition.events import ConfirmedIdentityEvent
from recognition.local_gallery import Gallery

from .models import RecognitionEvent, SessionStatus
from .repository import (
    AttendanceRepository,
    AttendanceRepositoryError,
    RecordNotFoundError,
    RosterViolationError,
)


class SessionRecognitionError(AttendanceRepositoryError):
    """Base error for a fail-closed live attendance-recognition context."""


class EmptySessionRosterError(SessionRecognitionError):
    pass


class NoRosterIdentitiesInGalleryError(SessionRecognitionError):
    pass


class GalleryGenerationMismatchError(SessionRecognitionError):
    pass


class InactiveAttendanceSessionError(SessionRecognitionError):
    pass


@dataclass(frozen=True)
class RosterGalleryFilterResult:
    gallery: Gallery
    allowed_student_ids: tuple[str, ...]
    matched_student_ids: tuple[str, ...]
    missing_gallery_student_ids: tuple[str, ...]


@dataclass(frozen=True)
class SessionRecognitionContext:
    session_id: str
    class_id: str
    allowed_student_ids: tuple[str, ...]
    gallery_generation: str
    filtered_gallery: Gallery
    missing_gallery_student_ids: tuple[str, ...]

    @property
    def gallery(self) -> Gallery:
        """Alias used by callers that do not need to distinguish the view."""
        return self.filtered_gallery


def filter_gallery_for_roster(
    global_gallery: Gallery,
    allowed_student_ids: Iterable[str],
) -> RosterGalleryFilterResult:
    """Return an immutable, gallery-compatible roster view.

    Missing roster members are reported but do not prevent live recognition if
    at least one roster identity has usable gallery prototypes.  Such members
    receive no synthetic identity and can later remain Not Observed/Needs Review.
    """
    normalized = set()
    for value in allowed_student_ids:
        if value is None:
            continue
        student_id = str(value).strip()
        if student_id:
            normalized.add(student_id)
    allowed = tuple(sorted(normalized))
    if not allowed:
        raise EmptySessionRosterError("Attendance session roster is empty")

    allowed_set = set(allowed)
    filtered_identities = tuple(
        identity for identity in global_gallery.identities if identity.user_id in allowed_set
    )
    matched = tuple(identity.user_id for identity in filtered_identities)
    matched_set = set(matched)
    missing = tuple(student_id for student_id in allowed if student_id not in matched_set)
    if not filtered_identities:
        raise NoRosterIdentitiesInGalleryError(
            "No attendance-session roster identities exist in the loaded gallery"
        )

    filtered_gallery = Gallery(
        generation=global_gallery.generation,
        provenance=global_gallery.provenance,
        identities=filtered_identities,
        diagnostics=global_gallery.diagnostics,
    )
    return RosterGalleryFilterResult(
        gallery=filtered_gallery,
        allowed_student_ids=allowed,
        matched_student_ids=matched,
        missing_gallery_student_ids=missing,
    )


def create_session_recognition_context(
    repository: AttendanceRepository,
    session_id: str,
    global_gallery: Gallery,
) -> SessionRecognitionContext:
    """Load an active session and build its roster-filtered global-gallery view."""
    session = repository.get_attendance_session(session_id)
    if session is None:
        raise RecordNotFoundError(f"Attendance session does not exist: {session_id}")
    if session.status is not SessionStatus.ACTIVE:
        raise InactiveAttendanceSessionError(
            f"Attendance session is not active: {session_id} ({session.status.value})"
        )
    if session.gallery_generation != global_gallery.generation:
        raise GalleryGenerationMismatchError(
            "Attendance session gallery generation does not match the loaded gallery"
        )

    allowed = repository.get_session_allowed_student_ids(session_id)
    filtered = filter_gallery_for_roster(global_gallery, allowed)
    return SessionRecognitionContext(
        session_id=session.session_id,
        class_id=session.class_id,
        allowed_student_ids=filtered.allowed_student_ids,
        gallery_generation=session.gallery_generation,
        filtered_gallery=filtered.gallery,
        missing_gallery_student_ids=filtered.missing_gallery_student_ids,
    )


class SQLiteAttendanceEventSink:
    """Persist confirmed evidence to SQLite; an existing row wins on duplicates."""

    def __init__(
        self,
        repository: AttendanceRepository,
        context: SessionRecognitionContext,
        *,
        event_id_factory: Callable[[], str] | None = None,
    ):
        self.repository = repository
        self.context = context
        self._event_id_factory = event_id_factory or (lambda: uuid.uuid4().hex)
        self.last_persisted_event: RecognitionEvent | None = None

    def accept(self, event: ConfirmedIdentityEvent) -> bool:
        if not isinstance(event, ConfirmedIdentityEvent):
            raise TypeError("SQLite attendance sink accepts ConfirmedIdentityEvent only")
        if event.status != "confirmed_identity" or event.valid_observations < 2:
            raise ValueError("Only confirmed identity evidence may be persisted")
        if event.session_id != self.context.session_id:
            raise ValueError("Confirmed event belongs to another attendance session")
        if event.gallery_generation != self.context.gallery_generation:
            raise GalleryGenerationMismatchError(
                "Confirmed event gallery generation does not match its attendance session"
            )
        session = self.repository.get_attendance_session(self.context.session_id)
        if session is None:
            raise RecordNotFoundError(
                f"Attendance session does not exist: {self.context.session_id}"
            )
        if session.status is not SessionStatus.ACTIVE:
            raise InactiveAttendanceSessionError(
                f"Attendance session is not active: {self.context.session_id}"
            )
        if event.user_id not in self.context.allowed_student_ids:
            raise RosterViolationError(
                "Confirmed identity is outside the attendance session roster"
            )

        existing = self.repository.get_recognition_event_for_student(
            event.session_id, event.user_id
        )
        if existing is not None:
            self.last_persisted_event = existing
            return False

        try:
            persisted = self.repository.insert_recognition_event(
                event_id=self._event_id_factory(),
                session_id=event.session_id,
                student_id=event.user_id,
                gallery_generation=event.gallery_generation,
                status=event.status,
                track_id=event.track_id,
                top1_score=event.top1_score,
                top2_score=event.top2_score,
                margin=event.margin,
                valid_observations=event.valid_observations,
                confirmation_attempts=event.confirmation_attempts,
                confirmation_elapsed_seconds=event.elapsed_confirmation_time,
                selected_frame_count=event.selected_frame_count,
                frame_sequence=event.frame_sequence,
                acquisition_time=event.acquisition_time,
            )
        except sqlite3.IntegrityError:
            # Another writer may have won the UNIQUE(session_id, student_id)
            # race. Only that specific, now-existing evidence row is a duplicate.
            existing = self.repository.get_recognition_event_for_student(
                event.session_id, event.user_id
            )
            if existing is None:
                raise
            self.last_persisted_event = existing
            return False
        self.last_persisted_event = persisted
        return True
