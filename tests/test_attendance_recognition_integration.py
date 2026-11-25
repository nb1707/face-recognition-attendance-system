from datetime import datetime, timezone
from pathlib import Path
import sys

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "backend"))

from attendance.models import SessionStatus
from attendance.recognition_integration import (
    EmptySessionRosterError,
    GalleryGenerationMismatchError,
    InactiveAttendanceSessionError,
    NoRosterIdentitiesInGalleryError,
    SQLiteAttendanceEventSink,
    create_session_recognition_context,
    filter_gallery_for_roster,
)
from attendance.repository import AttendanceRepository, RosterViolationError
from recognition.events import ConfirmedIdentityEvent
from recognition.identity_matcher import match_identity
from recognition.local_gallery import Gallery, Identity, Prototype


DIMENSION = 20
GENERATION = "gallery-generation-1"
PROVENANCE = {"embedding_dimension": DIMENSION, "model": "fake-adaface"}


def fake_gallery(count=20, generation=GENERATION):
    identities = []
    for index in range(count):
        vector = np.zeros(DIMENSION, dtype=np.float32)
        vector[index] = 1.0
        student_id = f"stu_{index:03d}"
        identities.append(
            Identity(
                user_id=student_id,
                full_name=f"Student {index}",
                role="student",
                prototypes=(Prototype("front", vector, "front_01"),),
            )
        )
    return Gallery(
        generation=generation,
        provenance=PROVENANCE,
        identities=tuple(identities),
        diagnostics=("global diagnostic",),
    )


def repository_with_session(tmp_path, roster=("stu_000", "stu_001")):
    repository = AttendanceRepository(tmp_path / "attendance.db")
    repository.create_class("class_1", "COMP1682", "Final Project")
    for student_id in roster:
        repository.upsert_student(student_id, f"Name {student_id}")
        repository.add_class_member("class_1", student_id)
    repository.create_attendance_session(
        "session_1",
        "class_1",
        GENERATION,
        {"operating_point": {"similarity_threshold": 0.4}},
    )
    return repository


def confirmed_event(**overrides):
    values = {
        "session_id": "session_1",
        "gallery_generation": GENERATION,
        "track_id": 17,
        "user_id": "stu_000",
        "top1_score": 0.82,
        "top2_score": 0.31,
        "margin": 0.51,
        "valid_observations": 3,
        "confirmation_attempts": 3,
        "elapsed_confirmation_time": 0.75,
        "selected_frame_count": 3,
        "acquisition_time": 123.5,
        "frame_sequence": 91,
        "status": "confirmed_identity",
    }
    values.update(overrides)
    return ConfirmedIdentityEvent(**values)


def test_global_twenty_roster_seven_filters_exactly_without_mutation():
    gallery = fake_gallery()
    original_identities = gallery.identities
    allowed = ("stu_000", "stu_002", "stu_004", "stu_006", "stu_010", "stu_012", "stu_018")

    result = filter_gallery_for_roster(gallery, allowed)

    assert tuple(identity.user_id for identity in result.gallery.identities) == allowed
    assert result.matched_student_ids == allowed
    assert result.missing_gallery_student_ids == ()
    assert result.gallery.generation == gallery.generation
    assert result.gallery.provenance is gallery.provenance
    assert result.gallery.diagnostics == gallery.diagnostics
    assert gallery.identities is original_identities
    assert len(gallery.identities) == 20
    assert all(
        filtered is original_identities[int(filtered.user_id.split("_")[1])]
        for filtered in result.gallery.identities
    )


def test_filter_reports_missing_members_and_never_falls_back_to_global():
    gallery = fake_gallery()
    result = filter_gallery_for_roster(
        gallery, ("stu_001", "stu_missing", "stu_005")
    )
    assert tuple(identity.user_id for identity in result.gallery.identities) == (
        "stu_001",
        "stu_005",
    )
    assert result.missing_gallery_student_ids == ("stu_missing",)
    assert len(result.gallery.identities) == 2
    assert len(gallery.identities) == 20


def test_empty_and_zero_match_rosters_fail_closed():
    gallery = fake_gallery()
    with pytest.raises(EmptySessionRosterError):
        filter_gallery_for_roster(gallery, ())
    with pytest.raises(NoRosterIdentitiesInGalleryError):
        filter_gallery_for_roster(gallery, ("missing_a", "missing_b"))


def test_context_uses_session_snapshot_not_current_class_members(tmp_path):
    repository = repository_with_session(tmp_path, ("stu_000",))
    repository.upsert_student("stu_001", "Later Student")
    repository.remove_class_member("class_1", "stu_000")
    repository.add_class_member("class_1", "stu_001")

    context = create_session_recognition_context(
        repository, "session_1", fake_gallery()
    )
    assert context.session_id == "session_1"
    assert context.class_id == "class_1"
    assert context.allowed_student_ids == ("stu_000",)
    assert tuple(identity.user_id for identity in context.gallery.identities) == (
        "stu_000",
    )
    assert context.gallery_generation == GENERATION


def test_context_rejects_ended_finalized_and_generation_mismatch(tmp_path):
    repository = repository_with_session(tmp_path)
    with pytest.raises(GalleryGenerationMismatchError):
        create_session_recognition_context(
            repository,
            "session_1",
            fake_gallery(generation="different-generation"),
        )

    repository.end_attendance_session("session_1")
    with pytest.raises(InactiveAttendanceSessionError):
        create_session_recognition_context(repository, "session_1", fake_gallery())
    repository.finalize_attendance_session("session_1")
    assert repository.get_attendance_session("session_1").status is SessionStatus.FINALIZED
    with pytest.raises(InactiveAttendanceSessionError):
        create_session_recognition_context(repository, "session_1", fake_gallery())


def test_matcher_boundary_can_only_return_filtered_roster_identity():
    gallery = fake_gallery()
    filtered = filter_gallery_for_roster(gallery, ("stu_001", "stu_004"))
    outsider_query = gallery.identities[19].prototypes[0].embedding

    unrestricted = match_identity(outsider_query, gallery.identities, DIMENSION)
    roster_only = match_identity(
        outsider_query, filtered.gallery.identities, DIMENSION
    )

    assert unrestricted.top1_user_id == "stu_019"
    assert roster_only.top1_user_id in {"stu_001", "stu_004"}
    assert roster_only.top1_user_id != "stu_019"
    assert roster_only.candidate_count == 2


def test_sqlite_sink_persists_all_evidence_fields_and_utc_time(tmp_path):
    repository = repository_with_session(tmp_path)
    context = create_session_recognition_context(
        repository, "session_1", fake_gallery()
    )
    sink = SQLiteAttendanceEventSink(
        repository, context, event_id_factory=lambda: "event_1"
    )

    assert sink.accept(confirmed_event()) is True
    event = repository.get_recognition_event("event_1")
    assert event.session_id == "session_1"
    assert event.student_id == "stu_000"
    assert event.track_id == "17"
    assert event.top1_score == 0.82
    assert event.top2_score == 0.31
    assert event.margin == 0.51
    assert event.valid_observations == 3
    assert event.confirmation_attempts == 3
    assert event.confirmation_elapsed_seconds == 0.75
    assert event.selected_frame_count == 3
    assert event.frame_sequence == 91
    assert event.acquisition_time == 123.5
    assert event.gallery_generation == GENERATION
    assert event.status == "confirmed_identity"
    occurred = datetime.fromisoformat(event.occurred_at_utc)
    assert occurred.tzinfo is not None
    assert occurred.utcoffset() == timezone.utc.utcoffset(occurred)


def test_sqlite_sink_rejects_outsider_and_session_mismatch(tmp_path):
    repository = repository_with_session(tmp_path)
    context = create_session_recognition_context(
        repository, "session_1", fake_gallery()
    )
    sink = SQLiteAttendanceEventSink(repository, context)

    with pytest.raises(RosterViolationError):
        sink.accept(confirmed_event(user_id="stu_019"))
    with pytest.raises(ValueError, match="another attendance session"):
        sink.accept(confirmed_event(session_id="other_session"))
    with pytest.raises(GalleryGenerationMismatchError):
        sink.accept(confirmed_event(gallery_generation="other-generation"))
    assert repository.list_recognition_events("session_1") == []


def test_sqlite_sink_duplicate_is_existing_row_wins_without_crash(tmp_path):
    repository = repository_with_session(tmp_path)
    context = create_session_recognition_context(
        repository, "session_1", fake_gallery()
    )
    generated_ids = iter(("event_1", "event_2"))
    sink = SQLiteAttendanceEventSink(
        repository, context, event_id_factory=lambda: next(generated_ids)
    )
    event = confirmed_event()

    assert sink.accept(event) is True
    assert sink.accept(event) is False
    rows = repository.list_recognition_events("session_1")
    assert len(rows) == 1
    assert rows[0].event_id == "event_1"
    assert sink.last_persisted_event == rows[0]


def test_sqlite_sink_rechecks_active_session_at_persistence_time(tmp_path):
    repository = repository_with_session(tmp_path)
    context = create_session_recognition_context(
        repository, "session_1", fake_gallery()
    )
    sink = SQLiteAttendanceEventSink(repository, context)
    repository.end_attendance_session("session_1")

    with pytest.raises(InactiveAttendanceSessionError):
        sink.accept(confirmed_event())
    assert repository.list_recognition_events("session_1") == []


def test_worker_keeps_explicit_separate_attendance_mode():
    from desktop.recognition.worker import RecognitionWorker

    worker = RecognitionWorker()
    assert worker.attendance_context is None
    assert callable(worker.start)
    assert callable(worker.start_attendance_session)
