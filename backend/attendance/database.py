"""SQLite connection and schema boundary for local attendance data.

The global recognition gallery remains immutable and may contain identities that
are not in a class.  A future recognition integration must obtain allowed IDs
from a session roster and filter the already-loaded gallery in memory.  This
module never creates a per-class gallery or changes gallery artifacts.
"""

from __future__ import annotations

from contextlib import contextmanager
from pathlib import Path
import sqlite3
from typing import Iterator


DEFAULT_DATABASE_PATH = (
    Path(__file__).resolve().parents[1] / "storage" / "attendance" / "attendance.db"
)


SCHEMA_SQL = """
CREATE TABLE IF NOT EXISTS students (
    student_id TEXT PRIMARY KEY,
    full_name TEXT NOT NULL,
    role TEXT NOT NULL DEFAULT 'student',
    active INTEGER NOT NULL DEFAULT 1 CHECK (active IN (0, 1)),
    created_at_utc TEXT NOT NULL,
    updated_at_utc TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS classes (
    class_id TEXT PRIMARY KEY,
    class_code TEXT NOT NULL UNIQUE,
    class_name TEXT NOT NULL,
    active INTEGER NOT NULL DEFAULT 1 CHECK (active IN (0, 1)),
    created_at_utc TEXT NOT NULL,
    updated_at_utc TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS class_members (
    class_id TEXT NOT NULL,
    student_id TEXT NOT NULL,
    joined_at_utc TEXT NOT NULL,
    active INTEGER NOT NULL DEFAULT 1 CHECK (active IN (0, 1)),
    PRIMARY KEY (class_id, student_id),
    FOREIGN KEY (class_id) REFERENCES classes(class_id),
    FOREIGN KEY (student_id) REFERENCES students(student_id)
);

CREATE TABLE IF NOT EXISTS attendance_sessions (
    session_id TEXT PRIMARY KEY,
    class_id TEXT NOT NULL,
    started_at_utc TEXT NOT NULL,
    ended_at_utc TEXT NULL,
    status TEXT NOT NULL CHECK (status IN ('active', 'ended', 'finalized')),
    camera_source TEXT NULL,
    gallery_generation TEXT NOT NULL,
    recognition_config_json TEXT NOT NULL,
    operator_name TEXT NULL,
    created_at_utc TEXT NOT NULL,
    updated_at_utc TEXT NOT NULL,
    FOREIGN KEY (class_id) REFERENCES classes(class_id)
);

CREATE TABLE IF NOT EXISTS session_roster (
    session_id TEXT NOT NULL,
    student_id TEXT NOT NULL,
    snapshot_name TEXT NOT NULL,
    snapshot_role TEXT NOT NULL DEFAULT 'student',
    PRIMARY KEY (session_id, student_id),
    FOREIGN KEY (session_id) REFERENCES attendance_sessions(session_id),
    FOREIGN KEY (student_id) REFERENCES students(student_id)
);

CREATE TABLE IF NOT EXISTS recognition_events (
    event_id TEXT PRIMARY KEY,
    session_id TEXT NOT NULL,
    student_id TEXT NOT NULL,
    track_id TEXT,
    top1_score REAL,
    top2_score REAL,
    margin REAL,
    valid_observations INTEGER,
    confirmation_attempts INTEGER,
    confirmation_elapsed_seconds REAL,
    selected_frame_count INTEGER,
    frame_sequence INTEGER,
    acquisition_time REAL,
    occurred_at_utc TEXT NOT NULL,
    gallery_generation TEXT NOT NULL,
    status TEXT NOT NULL,
    created_at_utc TEXT NOT NULL,
    UNIQUE (session_id, student_id),
    FOREIGN KEY (session_id) REFERENCES attendance_sessions(session_id),
    FOREIGN KEY (student_id) REFERENCES students(student_id)
);

CREATE TABLE IF NOT EXISTS attendance_records (
    attendance_id TEXT PRIMARY KEY,
    session_id TEXT NOT NULL,
    student_id TEXT NOT NULL,
    suggestion_status TEXT NOT NULL CHECK (
        suggestion_status IN ('suggested_present', 'needs_review', 'not_observed')
    ),
    final_status TEXT NULL CHECK (
        final_status IS NULL OR final_status IN ('present', 'absent', 'excused')
    ),
    evidence_event_id TEXT NULL,
    reviewed_by TEXT NULL,
    reviewed_at_utc TEXT NULL,
    review_note TEXT NULL,
    created_at_utc TEXT NOT NULL,
    updated_at_utc TEXT NOT NULL,
    UNIQUE (session_id, student_id),
    FOREIGN KEY (session_id) REFERENCES attendance_sessions(session_id),
    FOREIGN KEY (student_id) REFERENCES students(student_id),
    FOREIGN KEY (evidence_event_id) REFERENCES recognition_events(event_id)
);

CREATE TABLE IF NOT EXISTS attendance_audit (
    audit_id TEXT PRIMARY KEY,
    attendance_id TEXT NOT NULL,
    session_id TEXT NOT NULL,
    student_id TEXT NOT NULL,
    actor TEXT NOT NULL,
    action TEXT NOT NULL,
    previous_status TEXT NULL,
    new_status TEXT NULL,
    reason TEXT NULL,
    changed_at_utc TEXT NOT NULL,
    FOREIGN KEY (attendance_id) REFERENCES attendance_records(attendance_id),
    FOREIGN KEY (session_id) REFERENCES attendance_sessions(session_id),
    FOREIGN KEY (student_id) REFERENCES students(student_id)
);
"""


class AttendanceDatabase:
    """Creates configured SQLite connections without retaining mutable globals."""

    def __init__(self, path: str | Path = DEFAULT_DATABASE_PATH):
        self.path = Path(path)

    def connect(self) -> sqlite3.Connection:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        connection = sqlite3.connect(self.path)
        connection.row_factory = sqlite3.Row
        connection.execute("PRAGMA foreign_keys = ON")
        return connection

    @contextmanager
    def connection(self) -> Iterator[sqlite3.Connection]:
        connection = self.connect()
        try:
            yield connection
        finally:
            connection.close()

    def initialize_schema(self) -> None:
        with self.connection() as connection:
            connection.executescript(SCHEMA_SQL)
