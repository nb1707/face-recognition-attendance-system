from pathlib import Path
import sys

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "backend"))

from attendance.api import create_attendance_router
from attendance.repository import AttendanceRepository


@pytest.fixture
def api(tmp_path):
    database_path = tmp_path / "attendance-api.db"
    repository = AttendanceRepository(database_path)
    repository.create_class("class_1", "COMP1682", "Final Project")
    for student_id in ("stu_001", "stu_002"):
        repository.upsert_student(student_id, f"Name {student_id}")
        repository.add_class_member("class_1", student_id)
    app = FastAPI()
    app.include_router(create_attendance_router(repository))
    with TestClient(app) as client:
        yield client, repository, database_path


def create_session(client, class_id="class_1"):
    return client.post(
        "/attendance/sessions",
        json={
            "class_id": class_id,
            "gallery_generation": "gallery-1",
            "recognition_config": {"threshold": 0.4, "margin": 0.05},
            "camera_source": "camera-0",
            "operator_name": "Teacher",
        },
    )


def end_session(client, session_id):
    return client.post(f"/attendance/sessions/{session_id}/end")


def review(client, session_id, student_id, status="present", reason=None):
    return client.post(
        f"/attendance/sessions/{session_id}/students/{student_id}/review",
        json={"final_status": status, "actor": "Teacher", "reason": reason},
    )


def test_list_classes_and_class_roster(api):
    client, repository, _ = api
    repository.create_class("inactive", "OLD", "Inactive Class")
    repository.set_class_active("inactive", False)
    classes = client.get("/attendance/classes")
    assert classes.status_code == 200
    assert classes.json() == [
        {
            "class_id": "class_1",
            "class_code": "COMP1682",
            "class_name": "Final Project",
            "active": True,
        }
    ]
    roster = client.get("/attendance/classes/class_1/roster")
    assert roster.status_code == 200
    assert [member["student_id"] for member in roster.json()] == ["stu_001", "stu_002"]
    assert client.get("/attendance/classes/missing/roster").status_code == 404


def test_class_administration_lifecycle_is_idempotent_and_isolated(api, tmp_path):
    client, repository, _ = api
    repository.upsert_student("available", "Available Student", "student")
    repository.upsert_student("inactive", "Inactive Student", "student", active=False)
    repository.create_class("class_2", "COMP2000", "Other Class")

    students = client.get("/attendance/students")
    assert students.status_code == 200
    assert [item["student_id"] for item in students.json()] == [
        "available", "stu_001", "stu_002"
    ]
    assert all(item["active"] for item in students.json())

    create_payload = {
        "class_id": "class_3",
        "class_code": "COMP3000",
        "class_name": "Created Class",
    }
    created = client.post("/attendance/classes", json=create_payload)
    assert created.status_code == 201
    assert client.post("/attendance/classes", json=create_payload).status_code == 201
    conflicting = dict(create_payload, class_name="Different")
    assert client.post("/attendance/classes", json=conflicting).status_code == 409

    users_file = tmp_path / "users.json"
    gallery_file = tmp_path / "current.json"
    users_file.write_text('{"unchanged": true}', encoding="utf-8")
    gallery_file.write_text('{"manifest": "same"}', encoding="utf-8")

    assert client.post("/attendance/classes/class_1/students/available").status_code == 200
    assert client.post("/attendance/classes/class_1/students/available").status_code == 200
    assert repository.is_class_member("class_1", "available")
    assert repository.list_class_roster("class_2") == []
    with repository.database.connection() as connection:
        assert connection.execute("SELECT COUNT(*) FROM attendance_sessions").fetchone()[0] == 0

    removed = client.delete("/attendance/classes/class_1/students/available")
    assert removed.status_code == 204
    assert not repository.is_class_member("class_1", "available")
    assert repository.get_student("available") is not None
    assert client.post("/attendance/classes/class_1/students/available").status_code == 200
    assert repository.is_class_member("class_1", "available")
    assert users_file.read_text(encoding="utf-8") == '{"unchanged": true}'
    assert gallery_file.read_text(encoding="utf-8") == '{"manifest": "same"}'
    with repository.database.connection() as connection:
        assert connection.execute("SELECT COUNT(*) FROM attendance_sessions").fetchone()[0] == 0


def test_class_administration_rejects_unknown_and_inactive_entities(api):
    client, repository, _ = api
    assert client.post("/attendance/classes/missing/students/stu_001").status_code == 404
    assert client.post("/attendance/classes/class_1/students/missing").status_code == 404
    assert client.delete("/attendance/classes/class_1/students/missing").status_code == 404

    repository.create_class("inactive_class", "INACTIVE", "Inactive")
    repository.set_class_active("inactive_class", False)
    assert client.post(
        "/attendance/classes/inactive_class/students/stu_001"
    ).status_code == 409
    assert client.delete(
        "/attendance/classes/inactive_class/students/stu_001"
    ).status_code == 409


def test_roster_edits_only_affect_future_session_snapshots(api):
    client, repository, _ = api
    first = create_session(client).json()["session_id"]
    original = repository.get_session_allowed_student_ids(first)

    repository.upsert_student("later", "Later Student")
    assert client.post("/attendance/classes/class_1/students/later").status_code == 200
    assert client.delete("/attendance/classes/class_1/students/stu_001").status_code == 204

    assert repository.get_session_allowed_student_ids(first) == original
    second = create_session(client).json()["session_id"]
    assert repository.get_session_allowed_student_ids(second) == ("later", "stu_002")


def test_create_session_snapshots_roster_and_gets_counts(api):
    client, repository, database_path = api
    response = create_session(client)
    assert response.status_code == 201
    created = response.json()
    assert created["status"] == "active"
    assert created["roster_count"] == 2
    session_id = created["session_id"]
    assert repository.get_session_allowed_student_ids(session_id) == ("stu_001", "stu_002")
    assert repository.database.path == database_path

    detail = client.get(f"/attendance/sessions/{session_id}")
    assert detail.status_code == 200
    assert detail.json()["recognition_event_count"] == 0
    assert detail.json()["attendance_record_count"] == 2
    assert client.get("/attendance/sessions/missing").status_code == 404


def test_create_session_rejects_empty_roster(api):
    client, repository, _ = api
    repository.create_class("empty", "EMPTY", "Empty Class")
    response = create_session(client, "empty")
    assert response.status_code == 409
    assert repository.list_attendance_records("empty") == []


def test_refresh_suggestions_and_evidence_summary(api):
    client, repository, _ = api
    session_id = create_session(client).json()["session_id"]
    initial = client.post(f"/attendance/sessions/{session_id}/refresh-suggestions")
    assert initial.status_code == 200
    assert {record["suggestion_status"] for record in initial.json()} == {"not_observed"}
    assert all(not record["event_exists"] for record in initial.json())

    event = repository.insert_recognition_event(
        "event_1",
        session_id,
        "stu_001",
        "gallery-1",
        "confirmed_identity",
        top1_score=0.82,
        margin=0.31,
        valid_observations=3,
        confirmation_attempts=3,
    )
    refreshed = client.post(f"/attendance/sessions/{session_id}/refresh-suggestions")
    records = {record["student_id"]: record for record in refreshed.json()}
    assert records["stu_001"]["suggestion_status"] == "suggested_present"
    assert records["stu_001"]["evidence_event_id"] == event.event_id
    assert records["stu_001"]["event_exists"] is True
    assert records["stu_001"]["top1_score"] == pytest.approx(0.82)
    assert records["stu_001"]["margin"] == pytest.approx(0.31)
    assert records["stu_002"]["suggestion_status"] == "not_observed"


def test_end_session_refreshes_and_rejects_second_end(api):
    client, repository, _ = api
    session_id = create_session(client).json()["session_id"]
    repository.insert_recognition_event(
        "event_1", session_id, "stu_002", "gallery-1", "confirmed_identity"
    )
    ended = end_session(client, session_id)
    assert ended.status_code == 200
    assert ended.json()["session"]["status"] == "ended"
    records = {item["student_id"]: item for item in ended.json()["records"]}
    assert records["stu_002"]["suggestion_status"] == "suggested_present"
    assert end_session(client, session_id).status_code == 409


def test_review_active_rejected_and_ended_final_statuses_allowed(api):
    client, _, _ = api
    session_id = create_session(client).json()["session_id"]
    assert review(client, session_id, "stu_001").status_code == 409
    assert end_session(client, session_id).status_code == 200
    for student_id, status in (("stu_001", "absent"), ("stu_002", "excused")):
        response = review(client, session_id, student_id, status, "teacher decision")
        assert response.status_code == 200
        assert response.json()["final_status"] == status
        assert response.json()["reviewed_by"] == "Teacher"
        assert response.json()["suggestion_status"] == "not_observed"


def test_review_request_validation(api):
    client, _, _ = api
    session_id = create_session(client).json()["session_id"]
    end_session(client, session_id)
    invalid = review(client, session_id, "stu_001", "suggested_present")
    assert invalid.status_code == 400
    blank_actor = client.post(
        f"/attendance/sessions/{session_id}/students/stu_001/review",
        json={"final_status": "present", "actor": "   "},
    )
    assert blank_actor.status_code == 400
    assert review(client, session_id, "outside", "present").status_code == 409


def test_finalization_requires_all_reviewed_and_blocks_later_review(api):
    client, _, _ = api
    session_id = create_session(client).json()["session_id"]
    end_session(client, session_id)
    assert review(client, session_id, "stu_001", "present").status_code == 200
    incomplete = client.post(
        f"/attendance/sessions/{session_id}/finalize", json={"actor": "Teacher"}
    )
    assert incomplete.status_code == 409
    assert review(client, session_id, "stu_002", "absent").status_code == 200
    finalized = client.post(
        f"/attendance/sessions/{session_id}/finalize", json={"actor": "Teacher"}
    )
    assert finalized.status_code == 200
    assert finalized.json()["status"] == "finalized"
    assert review(client, session_id, "stu_001", "excused").status_code == 409


def test_audit_endpoint_returns_review_history_in_order(api):
    client, _, _ = api
    session_id = create_session(client).json()["session_id"]
    end_session(client, session_id)
    review(client, session_id, "stu_001", "present", "first note")
    review(client, session_id, "stu_001", "excused", "corrected")
    response = client.get(f"/attendance/sessions/{session_id}/audit")
    assert response.status_code == 200
    audits = response.json()
    assert [audit["action"] for audit in audits] == ["initial_review", "status_changed"]
    assert audits[0]["previous_status"] is None
    assert audits[0]["new_status"] == "present"
    assert audits[1]["previous_status"] == "present"
    assert audits[1]["new_status"] == "excused"
    assert client.get("/attendance/sessions/missing/audit").status_code == 404
