"""Repository primitives for the authoritative local attendance database.

Recognition integration is deliberately absent.  The future integration path is
``session_id -> get_session_allowed_student_ids() -> in-memory global-gallery
filter``.  A confirmed identity must be in the immutable session roster; event
insertion enforces that invariant again at the persistence boundary.
"""

from __future__ import annotations

from datetime import datetime, timezone
import json
from pathlib import Path
import sqlite3
from typing import Any, Mapping
import uuid

from .database import AttendanceDatabase, DEFAULT_DATABASE_PATH
from .models import (
    AttendanceAudit,
    AttendanceRecord,
    AttendanceSession,
    Class,
    FinalAttendanceStatus,
    RecognitionEvent,
    RosterMember,
    SessionRosterMember,
    SessionStatus,
    Student,
    SuggestionStatus,
)


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def normalize_utc_timestamp(value: str | None = None) -> str:
    if value is None:
        return utc_now()
    try:
        parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    except (AttributeError, ValueError) as error:
        raise ValueError("Timestamp must be valid ISO-8601") from error
    if parsed.tzinfo is None:
        raise ValueError("Timestamp must include a UTC offset")
    return parsed.astimezone(timezone.utc).isoformat()


class AttendanceRepositoryError(RuntimeError):
    pass


class RecordNotFoundError(AttendanceRepositoryError):
    pass


class InactiveClassError(AttendanceRepositoryError):
    pass


class EmptyRosterError(AttendanceRepositoryError):
    pass


class InvalidSessionTransitionError(AttendanceRepositoryError):
    pass


class RosterViolationError(AttendanceRepositoryError):
    pass


class AttendanceRepository:
    def __init__(
        self,
        database: AttendanceDatabase | str | Path = DEFAULT_DATABASE_PATH,
        *,
        initialize: bool = True,
    ):
        self.database = (
            database if isinstance(database, AttendanceDatabase) else AttendanceDatabase(database)
        )
        if initialize:
            self.database.initialize_schema()

    # Students
    def upsert_student(
        self, student_id: str, full_name: str, role: str = "student", active: bool = True
    ) -> Student:
        student_id = self._required(student_id, "student_id")
        full_name = self._required(full_name, "full_name")
        role = self._required(role, "role")
        now = utc_now()
        with self.database.connection() as connection, connection:
            connection.execute(
                """
                INSERT INTO students (
                    student_id, full_name, role, active, created_at_utc, updated_at_utc
                ) VALUES (?, ?, ?, ?, ?, ?)
                ON CONFLICT(student_id) DO UPDATE SET
                    full_name = excluded.full_name,
                    role = excluded.role,
                    active = excluded.active,
                    updated_at_utc = excluded.updated_at_utc
                """,
                (student_id, full_name, role, int(active), now, now),
            )
        return self.get_student(student_id)

    def get_student(self, student_id: str) -> Student | None:
        row = self._fetch_one("SELECT * FROM students WHERE student_id = ?", (student_id,))
        return self._student(row) if row else None

    def list_students(self, active_only: bool = False) -> list[Student]:
        sql = "SELECT * FROM students"
        parameters: tuple[Any, ...] = ()
        if active_only:
            sql += " WHERE active = ?"
            parameters = (1,)
        sql += " ORDER BY student_id"
        return [self._student(row) for row in self._fetch_all(sql, parameters)]

    def set_student_active(self, student_id: str, active: bool) -> Student:
        with self.database.connection() as connection, connection:
            cursor = connection.execute(
                """
                UPDATE students SET active = ?, updated_at_utc = ? WHERE student_id = ?
                """,
                (int(active), utc_now(), student_id),
            )
            if cursor.rowcount != 1:
                raise RecordNotFoundError(f"Student does not exist: {student_id}")
        return self._require_student(student_id)

    # Classes
    def create_class(self, class_id: str, class_code: str, class_name: str) -> Class:
        now = utc_now()
        with self.database.connection() as connection, connection:
            connection.execute(
                """
                INSERT INTO classes (
                    class_id, class_code, class_name, active, created_at_utc, updated_at_utc
                ) VALUES (?, ?, ?, 1, ?, ?)
                """,
                (
                    self._required(class_id, "class_id"),
                    self._required(class_code, "class_code"),
                    self._required(class_name, "class_name"),
                    now,
                    now,
                ),
            )
        return self.get_class(class_id)  # type: ignore[return-value]

    def update_class(
        self,
        class_id: str,
        *,
        class_code: str | None = None,
        class_name: str | None = None,
    ) -> Class:
        if class_code is None and class_name is None:
            raise ValueError("At least one class field must be supplied")
        current = self.get_class(class_id)
        if current is None:
            raise RecordNotFoundError(f"Class does not exist: {class_id}")
        now = utc_now()
        with self.database.connection() as connection, connection:
            connection.execute(
                """
                UPDATE classes SET class_code = ?, class_name = ?, updated_at_utc = ?
                WHERE class_id = ?
                """,
                (
                    self._required(class_code, "class_code") if class_code is not None else current.class_code,
                    self._required(class_name, "class_name") if class_name is not None else current.class_name,
                    now,
                    class_id,
                ),
            )
        return self.get_class(class_id)  # type: ignore[return-value]

    def get_class(self, class_id: str) -> Class | None:
        row = self._fetch_one("SELECT * FROM classes WHERE class_id = ?", (class_id,))
        return self._class(row) if row else None

    def get_class_by_code(self, class_code: str) -> Class | None:
        row = self._fetch_one("SELECT * FROM classes WHERE class_code = ?", (class_code,))
        return self._class(row) if row else None

    def list_classes(self, active_only: bool = True) -> list[Class]:
        sql = "SELECT * FROM classes"
        parameters: tuple[Any, ...] = ()
        if active_only:
            sql += " WHERE active = ?"
            parameters = (1,)
        sql += " ORDER BY class_code"
        return [self._class(row) for row in self._fetch_all(sql, parameters)]

    def set_class_active(self, class_id: str, active: bool) -> Class:
        with self.database.connection() as connection, connection:
            cursor = connection.execute(
                """
                UPDATE classes SET active = ?, updated_at_utc = ? WHERE class_id = ?
                """,
                (int(active), utc_now(), class_id),
            )
            if cursor.rowcount != 1:
                raise RecordNotFoundError(f"Class does not exist: {class_id}")
        result = self.get_class(class_id)
        if result is None:
            raise RecordNotFoundError(f"Class does not exist: {class_id}")
        return result

    # Class membership
    def add_class_member(self, class_id: str, student_id: str) -> None:
        now = utc_now()
        with self.database.connection() as connection, connection:
            connection.execute(
                """
                INSERT INTO class_members (class_id, student_id, joined_at_utc, active)
                VALUES (?, ?, ?, 1)
                ON CONFLICT(class_id, student_id) DO UPDATE SET active = 1
                """,
                (class_id, student_id, now),
            )

    def remove_class_member(self, class_id: str, student_id: str) -> bool:
        with self.database.connection() as connection, connection:
            cursor = connection.execute(
                "UPDATE class_members SET active = 0 WHERE class_id = ? AND student_id = ?",
                (class_id, student_id),
            )
            return cursor.rowcount == 1

    def list_class_roster(self, class_id: str, active_only: bool = True) -> list[RosterMember]:
        sql = """
            SELECT s.student_id, s.full_name, s.role, cm.active, cm.joined_at_utc
            FROM class_members AS cm
            JOIN students AS s ON s.student_id = cm.student_id
            WHERE cm.class_id = ?
        """
        parameters: list[Any] = [class_id]
        if active_only:
            sql += " AND cm.active = ? AND s.active = ?"
            parameters.extend((1, 1))
        sql += " ORDER BY s.student_id"
        return [
            RosterMember(
                student_id=row["student_id"],
                full_name=row["full_name"],
                role=row["role"],
                active=bool(row["active"]),
                joined_at_utc=row["joined_at_utc"],
            )
            for row in self._fetch_all(sql, tuple(parameters))
        ]

    def is_class_member(
        self, class_id: str, student_id: str, active_only: bool = True
    ) -> bool:
        sql = "SELECT 1 FROM class_members WHERE class_id = ? AND student_id = ?"
        parameters: list[Any] = [class_id, student_id]
        if active_only:
            sql += " AND active = ?"
            parameters.append(1)
        return self._fetch_one(sql, tuple(parameters)) is not None

    # Sessions and immutable roster snapshots
    def create_attendance_session(
        self,
        session_id: str,
        class_id: str,
        gallery_generation: str,
        recognition_config: Mapping[str, Any],
        camera_source: str | None = None,
        operator_name: str | None = None,
    ) -> AttendanceSession:
        config_json = json.dumps(
            recognition_config,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
            allow_nan=False,
        )
        now = utc_now()
        with self.database.connection() as connection, connection:
            class_row = connection.execute(
                "SELECT active FROM classes WHERE class_id = ?", (class_id,)
            ).fetchone()
            if class_row is None:
                raise RecordNotFoundError(f"Class does not exist: {class_id}")
            if not class_row["active"]:
                raise InactiveClassError(f"Class is inactive: {class_id}")
            roster = connection.execute(
                """
                SELECT s.student_id, s.full_name, s.role
                FROM class_members AS cm
                JOIN students AS s ON s.student_id = cm.student_id
                WHERE cm.class_id = ? AND cm.active = 1 AND s.active = 1
                ORDER BY s.student_id
                """,
                (class_id,),
            ).fetchall()
            if not roster:
                raise EmptyRosterError(f"Class has no active roster members: {class_id}")
            connection.execute(
                """
                INSERT INTO attendance_sessions (
                    session_id, class_id, started_at_utc, ended_at_utc, status,
                    camera_source, gallery_generation, recognition_config_json,
                    operator_name, created_at_utc, updated_at_utc
                ) VALUES (?, ?, ?, NULL, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    self._required(session_id, "session_id"),
                    class_id,
                    now,
                    SessionStatus.ACTIVE.value,
                    camera_source,
                    self._required(gallery_generation, "gallery_generation"),
                    config_json,
                    operator_name,
                    now,
                    now,
                ),
            )
            connection.executemany(
                """
                INSERT INTO session_roster (
                    session_id, student_id, snapshot_name, snapshot_role
                ) VALUES (?, ?, ?, ?)
                """,
                [
                    (session_id, row["student_id"], row["full_name"], row["role"])
                    for row in roster
                ],
            )
        return self.get_attendance_session(session_id)  # type: ignore[return-value]

    def get_attendance_session(self, session_id: str) -> AttendanceSession | None:
        row = self._fetch_one(
            "SELECT * FROM attendance_sessions WHERE session_id = ?", (session_id,)
        )
        return self._session(row) if row else None

    def list_session_roster(self, session_id: str) -> list[SessionRosterMember]:
        return [
            SessionRosterMember(**dict(row))
            for row in self._fetch_all(
                "SELECT * FROM session_roster WHERE session_id = ? ORDER BY student_id",
                (session_id,),
            )
        ]

    def session_allows_student(self, session_id: str, student_id: str) -> bool:
        return self._fetch_one(
            "SELECT 1 FROM session_roster WHERE session_id = ? AND student_id = ?",
            (session_id, student_id),
        ) is not None

    def get_session_allowed_student_ids(self, session_id: str) -> tuple[str, ...]:
        return tuple(member.student_id for member in self.list_session_roster(session_id))

    def end_attendance_session(self, session_id: str) -> AttendanceSession:
        return self._transition_session(
            session_id, SessionStatus.ACTIVE, SessionStatus.ENDED, set_ended_at=True
        )

    def finalize_attendance_session(self, session_id: str) -> AttendanceSession:
        return self._transition_session(
            session_id, SessionStatus.ENDED, SessionStatus.FINALIZED, set_ended_at=False
        )

    # Confirmed recognition evidence (not final attendance)
    def insert_recognition_event(
        self,
        event_id: str,
        session_id: str,
        student_id: str,
        gallery_generation: str,
        status: str,
        *,
        track_id: str | int | None = None,
        top1_score: float | None = None,
        top2_score: float | None = None,
        margin: float | None = None,
        valid_observations: int | None = None,
        confirmation_attempts: int | None = None,
        confirmation_elapsed_seconds: float | None = None,
        selected_frame_count: int | None = None,
        frame_sequence: int | None = None,
        acquisition_time: float | None = None,
        occurred_at_utc: str | None = None,
    ) -> RecognitionEvent:
        now = utc_now()
        occurred = normalize_utc_timestamp(occurred_at_utc)
        with self.database.connection() as connection, connection:
            session = connection.execute(
                "SELECT status FROM attendance_sessions WHERE session_id = ?", (session_id,)
            ).fetchone()
            if session is None:
                raise RecordNotFoundError(f"Session does not exist: {session_id}")
            if session["status"] != SessionStatus.ACTIVE.value:
                raise AttendanceRepositoryError(
                    f"Recognition events require an active session: {session_id}"
                )
            allowed = connection.execute(
                "SELECT 1 FROM session_roster WHERE session_id = ? AND student_id = ?",
                (session_id, student_id),
            ).fetchone()
            if allowed is None:
                raise RosterViolationError(
                    f"Student {student_id} is not in session roster {session_id}"
                )
            connection.execute(
                """
                INSERT INTO recognition_events (
                    event_id, session_id, student_id, track_id, top1_score, top2_score,
                    margin, valid_observations, confirmation_attempts,
                    confirmation_elapsed_seconds, selected_frame_count, frame_sequence,
                    acquisition_time, occurred_at_utc, gallery_generation, status,
                    created_at_utc
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    self._required(event_id, "event_id"),
                    session_id,
                    student_id,
                    None if track_id is None else str(track_id),
                    top1_score,
                    top2_score,
                    margin,
                    valid_observations,
                    confirmation_attempts,
                    confirmation_elapsed_seconds,
                    selected_frame_count,
                    frame_sequence,
                    acquisition_time,
                    occurred,
                    self._required(gallery_generation, "gallery_generation"),
                    self._required(status, "status"),
                    now,
                ),
            )
        return self.get_recognition_event(event_id)  # type: ignore[return-value]

    def get_recognition_event(self, event_id: str) -> RecognitionEvent | None:
        row = self._fetch_one(
            "SELECT * FROM recognition_events WHERE event_id = ?", (event_id,)
        )
        return RecognitionEvent(**dict(row)) if row else None

    def list_recognition_events(self, session_id: str) -> list[RecognitionEvent]:
        return [
            RecognitionEvent(**dict(row))
            for row in self._fetch_all(
                """
                SELECT * FROM recognition_events WHERE session_id = ?
                ORDER BY occurred_at_utc, event_id
                """,
                (session_id,),
            )
        ]

    def get_recognition_event_for_student(
        self, session_id: str, student_id: str
    ) -> RecognitionEvent | None:
        row = self._fetch_one(
            """
            SELECT * FROM recognition_events WHERE session_id = ? AND student_id = ?
            """,
            (session_id, student_id),
        )
        return RecognitionEvent(**dict(row)) if row else None

    # Attendance records and append-only audit
    def create_or_initialize_attendance_records_for_session(
        self, session_id: str
    ) -> list[AttendanceRecord]:
        now = utc_now()
        with self.database.connection() as connection, connection:
            if connection.execute(
                "SELECT 1 FROM attendance_sessions WHERE session_id = ?", (session_id,)
            ).fetchone() is None:
                raise RecordNotFoundError(f"Session does not exist: {session_id}")
            students = connection.execute(
                "SELECT student_id FROM session_roster WHERE session_id = ? ORDER BY student_id",
                (session_id,),
            ).fetchall()
            connection.executemany(
                """
                INSERT INTO attendance_records (
                    attendance_id, session_id, student_id, suggestion_status,
                    final_status, evidence_event_id, reviewed_by, reviewed_at_utc,
                    review_note, created_at_utc, updated_at_utc
                ) VALUES (?, ?, ?, ?, NULL, NULL, NULL, NULL, NULL, ?, ?)
                ON CONFLICT(session_id, student_id) DO NOTHING
                """,
                [
                    (
                        uuid.uuid4().hex,
                        session_id,
                        row["student_id"],
                        SuggestionStatus.NOT_OBSERVED.value,
                        now,
                        now,
                    )
                    for row in students
                ],
            )
        return self.list_attendance_records(session_id)

    def get_attendance_record(
        self, session_id: str, student_id: str
    ) -> AttendanceRecord | None:
        row = self._fetch_one(
            "SELECT * FROM attendance_records WHERE session_id = ? AND student_id = ?",
            (session_id, student_id),
        )
        return self._attendance_record(row) if row else None

    def list_attendance_records(self, session_id: str) -> list[AttendanceRecord]:
        return [
            self._attendance_record(row)
            for row in self._fetch_all(
                "SELECT * FROM attendance_records WHERE session_id = ? ORDER BY student_id",
                (session_id,),
            )
        ]

    def update_attendance_suggestion(
        self,
        session_id: str,
        student_id: str,
        suggestion_status: SuggestionStatus,
        evidence_event_id: str | None,
    ) -> AttendanceRecord:
        """Update system-owned recommendation fields without touching teacher review."""
        if not isinstance(suggestion_status, SuggestionStatus):
            raise TypeError("suggestion_status must be a SuggestionStatus")

        now = utc_now()
        with self.database.connection() as connection, connection:
            current = connection.execute(
                """
                SELECT suggestion_status, evidence_event_id
                FROM attendance_records
                WHERE session_id = ? AND student_id = ?
                """,
                (session_id, student_id),
            ).fetchone()
            if current is None:
                raise RecordNotFoundError(
                    f"Attendance record does not exist: {session_id}/{student_id}"
                )
            if evidence_event_id is not None:
                evidence = connection.execute(
                    """
                    SELECT 1 FROM recognition_events
                    WHERE event_id = ? AND session_id = ? AND student_id = ?
                    """,
                    (evidence_event_id, session_id, student_id),
                ).fetchone()
                if evidence is None:
                    raise AttendanceRepositoryError(
                        "Recommendation evidence must belong to the attendance record"
                    )

            # Avoid changing updated_at_utc when a refresh produces identical state.
            if (
                current["suggestion_status"] != suggestion_status.value
                or current["evidence_event_id"] != evidence_event_id
            ):
                connection.execute(
                    """
                    UPDATE attendance_records
                    SET suggestion_status = ?, evidence_event_id = ?, updated_at_utc = ?
                    WHERE session_id = ? AND student_id = ?
                    """,
                    (
                        suggestion_status.value,
                        evidence_event_id,
                        now,
                        session_id,
                        student_id,
                    ),
                )

        result = self.get_attendance_record(session_id, student_id)
        if result is None:  # Defensive against an unexpected concurrent delete.
            raise RecordNotFoundError(
                f"Attendance record does not exist: {session_id}/{student_id}"
            )
        return result

    def update_attendance_review(
        self,
        session_id: str,
        student_id: str,
        final_status: FinalAttendanceStatus,
        actor: str,
        reason: str | None = None,
        *,
        audit_id: str | None = None,
        reviewed_at_utc: str | None = None,
    ) -> tuple[AttendanceRecord, AttendanceAudit | None]:
        """Atomically apply a human decision and append its immutable audit row."""
        if not isinstance(final_status, FinalAttendanceStatus):
            raise TypeError("final_status must be a FinalAttendanceStatus")
        actor = self._required(actor, "actor")
        reviewed_at = normalize_utc_timestamp(reviewed_at_utc)
        audit_identifier = audit_id or uuid.uuid4().hex

        with self.database.connection() as connection, connection:
            session = connection.execute(
                "SELECT status FROM attendance_sessions WHERE session_id = ?",
                (session_id,),
            ).fetchone()
            if session is None:
                raise RecordNotFoundError(f"Session does not exist: {session_id}")
            if session["status"] != SessionStatus.ENDED.value:
                raise InvalidSessionTransitionError(
                    "Attendance review requires an ended session: "
                    f"{session_id} ({session['status']})"
                )
            if connection.execute(
                "SELECT 1 FROM session_roster WHERE session_id = ? AND student_id = ?",
                (session_id, student_id),
            ).fetchone() is None:
                raise RosterViolationError(
                    f"Student {student_id} is not in session roster {session_id}"
                )

            current = connection.execute(
                """
                SELECT * FROM attendance_records
                WHERE session_id = ? AND student_id = ?
                """,
                (session_id, student_id),
            ).fetchone()
            if current is None:
                raise RecordNotFoundError(
                    f"Attendance record does not exist: {session_id}/{student_id}"
                )

            previous_status = current["final_status"]
            if (
                previous_status == final_status.value
                and current["reviewed_by"] == actor
                and current["review_note"] == reason
            ):
                return self._attendance_record(current), None

            if previous_status is None:
                action = "initial_review"
            elif previous_status != final_status.value:
                action = "status_changed"
            elif current["review_note"] != reason:
                action = "note_updated"
            else:
                action = "reviewer_changed"

            connection.execute(
                """
                UPDATE attendance_records
                SET final_status = ?, reviewed_by = ?, reviewed_at_utc = ?,
                    review_note = ?, updated_at_utc = ?
                WHERE attendance_id = ?
                """,
                (
                    final_status.value,
                    actor,
                    reviewed_at,
                    reason,
                    reviewed_at,
                    current["attendance_id"],
                ),
            )
            connection.execute(
                """
                INSERT INTO attendance_audit (
                    audit_id, attendance_id, session_id, student_id, actor, action,
                    previous_status, new_status, reason, changed_at_utc
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    audit_identifier,
                    current["attendance_id"],
                    session_id,
                    student_id,
                    actor,
                    action,
                    previous_status,
                    final_status.value,
                    reason,
                    reviewed_at,
                ),
            )

        record = self.get_attendance_record(session_id, student_id)
        audit = self._fetch_one(
            "SELECT * FROM attendance_audit WHERE audit_id = ?", (audit_identifier,)
        )
        if record is None or audit is None:
            raise AttendanceRepositoryError("Attendance review transaction did not persist")
        return record, AttendanceAudit(**dict(audit))

    def list_attendance_audit(self, session_id: str) -> list[AttendanceAudit]:
        return [
            AttendanceAudit(**dict(row))
            for row in self._fetch_all(
                """
                SELECT * FROM attendance_audit
                WHERE session_id = ?
                ORDER BY changed_at_utc, audit_id
                """,
                (session_id,),
            )
        ]

    def append_attendance_audit(
        self,
        audit_id: str,
        attendance_id: str,
        session_id: str,
        student_id: str,
        actor: str,
        action: str,
        previous_status: str | None = None,
        new_status: str | None = None,
        reason: str | None = None,
        changed_at_utc: str | None = None,
    ) -> AttendanceAudit:
        changed = normalize_utc_timestamp(changed_at_utc)
        with self.database.connection() as connection, connection:
            connection.execute(
                """
                INSERT INTO attendance_audit (
                    audit_id, attendance_id, session_id, student_id, actor, action,
                    previous_status, new_status, reason, changed_at_utc
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    self._required(audit_id, "audit_id"),
                    attendance_id,
                    session_id,
                    student_id,
                    self._required(actor, "actor"),
                    self._required(action, "action"),
                    previous_status,
                    new_status,
                    reason,
                    changed,
                ),
            )
        row = self._fetch_one("SELECT * FROM attendance_audit WHERE audit_id = ?", (audit_id,))
        return AttendanceAudit(**dict(row))  # type: ignore[arg-type]

    # Internal helpers
    def _transition_session(
        self,
        session_id: str,
        expected: SessionStatus,
        target: SessionStatus,
        *,
        set_ended_at: bool,
    ) -> AttendanceSession:
        now = utc_now()
        with self.database.connection() as connection, connection:
            current = connection.execute(
                "SELECT status FROM attendance_sessions WHERE session_id = ?", (session_id,)
            ).fetchone()
            if current is None:
                raise RecordNotFoundError(f"Session does not exist: {session_id}")
            if current["status"] != expected.value:
                raise InvalidSessionTransitionError(
                    f"Cannot transition session from {current['status']} to {target.value}"
                )
            if set_ended_at:
                connection.execute(
                    """
                    UPDATE attendance_sessions
                    SET status = ?, ended_at_utc = ?, updated_at_utc = ?
                    WHERE session_id = ?
                    """,
                    (target.value, now, now, session_id),
                )
            else:
                connection.execute(
                    """
                    UPDATE attendance_sessions SET status = ?, updated_at_utc = ?
                    WHERE session_id = ?
                    """,
                    (target.value, now, session_id),
                )
        return self.get_attendance_session(session_id)  # type: ignore[return-value]

    def _require_student(self, student_id: str) -> Student:
        result = self.get_student(student_id)
        if result is None:
            raise RecordNotFoundError(f"Student does not exist: {student_id}")
        return result

    def _fetch_one(self, sql: str, parameters: tuple[Any, ...]) -> sqlite3.Row | None:
        with self.database.connection() as connection:
            return connection.execute(sql, parameters).fetchone()

    def _fetch_all(self, sql: str, parameters: tuple[Any, ...]) -> list[sqlite3.Row]:
        with self.database.connection() as connection:
            return connection.execute(sql, parameters).fetchall()

    @staticmethod
    def _required(value: str | None, name: str) -> str:
        cleaned = str(value or "").strip()
        if not cleaned:
            raise ValueError(f"{name} is required")
        return cleaned

    @staticmethod
    def _student(row: sqlite3.Row) -> Student:
        values = dict(row)
        values["active"] = bool(values["active"])
        return Student(**values)

    @staticmethod
    def _class(row: sqlite3.Row) -> Class:
        values = dict(row)
        values["active"] = bool(values["active"])
        return Class(**values)

    @staticmethod
    def _session(row: sqlite3.Row) -> AttendanceSession:
        values = dict(row)
        values["status"] = SessionStatus(values["status"])
        return AttendanceSession(**values)

    @staticmethod
    def _attendance_record(row: sqlite3.Row) -> AttendanceRecord:
        values = dict(row)
        values["suggestion_status"] = SuggestionStatus(values["suggestion_status"])
        if values["final_status"] is not None:
            values["final_status"] = FinalAttendanceStatus(values["final_status"])
        return AttendanceRecord(**values)
