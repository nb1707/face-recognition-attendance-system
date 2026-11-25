from io import BytesIO
import json
from pathlib import Path
import socket
import sys
import tkinter as tk
import unittest
from urllib.error import HTTPError, URLError

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from desktop.attendance.client import AttendanceApiClient, AttendanceClientError
from desktop.attendance.state import (
    attendance_counts,
    can_finalize_records,
    decision_label,
    suggestion_label,
)
from desktop.views.recognition_view import RecognitionView


class FakeResponse:
    def __init__(self, payload):
        self.body = json.dumps(payload).encode("utf-8")

    def __enter__(self):
        return self

    def __exit__(self, *_args):
        return False

    def read(self):
        return self.body


class AttendanceApiClientTests(unittest.TestCase):
    def test_successful_json_request_is_centralized(self):
        calls = []

        def opener(request, timeout):
            calls.append((request, timeout))
            return FakeResponse([{"class_id": "class_1"}])

        client = AttendanceApiClient(
            "http://localhost:9000", timeout_seconds=2.5, opener=opener
        )
        self.assertEqual(client.list_classes(), [{"class_id": "class_1"}])
        request, timeout = calls[0]
        self.assertEqual(request.full_url, "http://localhost:9000/attendance/classes")
        self.assertEqual(request.method, "GET")
        self.assertEqual(timeout, 2.5)

    def test_review_request_serializes_expected_payload(self):
        captured = {}

        def opener(request, timeout):
            captured.update(json.loads(request.data.decode("utf-8")))
            return FakeResponse({"student_id": "stu_001", "final_status": "present"})

        client = AttendanceApiClient(opener=opener)
        response = client.review_student(
            "session_1", "stu_001", "present", "Teacher", "verified"
        )
        self.assertEqual(response["final_status"], "present")
        self.assertEqual(
            captured,
            {
                "final_status": "present",
                "actor": "Teacher",
                "reason": "verified",
            },
        )

    def test_http_error_maps_backend_detail(self):
        def opener(_request, timeout):
            raise HTTPError(
                "http://localhost/attendance",
                409,
                "Conflict",
                {},
                BytesIO(b'{"detail":"Session is finalized"}'),
            )

        with self.assertRaises(AttendanceClientError) as caught:
            AttendanceApiClient(opener=opener).list_classes()
        self.assertEqual(caught.exception.status_code, 409)
        self.assertEqual(str(caught.exception), "Session is finalized")

    def test_timeout_and_connection_errors_are_sanitized(self):
        failures = (
            lambda _request, timeout: (_ for _ in ()).throw(socket.timeout()),
            lambda _request, timeout: (_ for _ in ()).throw(URLError("offline")),
        )
        for opener in failures:
            with self.subTest(opener=opener):
                with self.assertRaisesRegex(
                    AttendanceClientError, "unavailable or timed out"
                ):
                    AttendanceApiClient(opener=opener).list_classes()


class AttendancePresentationStateTests(unittest.TestCase):
    def test_not_observed_is_never_presented_as_absent(self):
        self.assertEqual(suggestion_label("not_observed"), "Not Observed")
        self.assertEqual(decision_label(None), "Not Reviewed")
        self.assertNotEqual(suggestion_label("not_observed"), decision_label("absent"))

    def test_finalize_requires_explicit_decision_for_every_record(self):
        records = [
            {"student_id": "a", "final_status": "present"},
            {"student_id": "b", "final_status": None},
        ]
        self.assertFalse(can_finalize_records(records))
        records[1]["final_status"] = "excused"
        self.assertTrue(can_finalize_records(records))
        self.assertFalse(can_finalize_records([]))

    def test_counts_keep_evidence_and_final_decisions_separate(self):
        counts = attendance_counts(
            [
                {"event_exists": True, "final_status": "absent"},
                {"event_exists": False, "final_status": "present"},
                {"event_exists": False, "final_status": "excused"},
            ]
        )
        self.assertEqual(counts["evidence"], 1)
        self.assertEqual(counts["not_observed"], 2)
        self.assertEqual(counts["present"], 1)
        self.assertEqual(counts["absent"], 1)
        self.assertEqual(counts["excused"], 1)


class AttendanceViewStateTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        try:
            cls.root = tk.Tk()
            cls.root.withdraw()
        except tk.TclError as error:
            raise unittest.SkipTest(f"Tk is unavailable: {error}") from error

    @classmethod
    def tearDownClass(cls):
        cls.root.destroy()

    def setUp(self):
        self.view = RecognitionView(self.root)
        self.view.pack(fill="both", expand=True)
        self.root.update_idletasks()

    def tearDown(self):
        self.view.destroy()
        self.root.update_idletasks()

    def test_start_disabled_until_class_and_roster_are_loaded(self):
        self.assertEqual(str(self.view.start_button.cget("state")), "disabled")
        self.view.set_classes(
            [{"class_id": "class_1", "class_code": "C1", "class_name": "Class"}]
        )
        self.view.selected_class.set("C1 — Class")
        self.view.set_roster([{"student_id": "stu_001", "full_name": "Student"}])
        self.view._refresh_action_states()
        self.assertEqual(str(self.view.start_button.cget("state")), "normal")

    def test_finalized_state_disables_review_controls(self):
        self.view.set_attendance_records(
            [
                {
                    "student_id": "stu_001",
                    "snapshot_name": "Student",
                    "suggestion_status": "not_observed",
                    "final_status": "present",
                    "event_exists": False,
                    "review_note": None,
                }
            ],
            phase="ended",
        )
        self.view.attendance_table.selection_set("stu_001")
        self.view._on_review_row_selected()
        self.view.set_attendance_session("finalized", {"session_id": "session_1"})
        self.assertEqual(str(self.view.save_review_button.cget("state")), "disabled")
        self.assertEqual(str(self.view.finalize_button.cget("state")), "disabled")


if __name__ == "__main__":
    unittest.main()
