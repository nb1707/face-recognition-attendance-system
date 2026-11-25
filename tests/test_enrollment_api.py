from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest

import cv2
from fastapi import FastAPI
import numpy as np
from fastapi.testclient import TestClient

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "backend"))

from enrollment.api import create_enrollment_router
from enrollment.service import EnrollmentService


class Detector:
    def detect_faces(self, image):
        return [SimpleNamespace(bbox=np.array((40, 40, 160, 160), np.float32), det_score=.9,
            kps=np.array(((30, 30), (70, 30), (50, 50), (36, 75), (64, 75)), np.float32))]


def camera_jpeg():
    pattern = ((np.indices((200, 200)).sum(axis=0) % 2) * 80 + 80).astype(np.uint8)
    image = np.repeat(pattern[:, :, None], 3, axis=2)
    return cv2.imencode(".jpg", image)[1].tobytes()


class EnrollmentApiTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        root = Path(self.temp.name)
        self.service = EnrollmentService(Detector(), root / "enrollments", root / "gallery",
            aligner=lambda image, detected_face, output_size: cv2.resize(image, (112, 112)))
        app = FastAPI()
        app.include_router(create_enrollment_router(self.service))
        self.client = TestClient(app)

    def tearDown(self): self.temp.cleanup()

    def start(self):
        return self.client.post("/enrollments", json={
            "user_id": "api-user", "full_name": "API User", "role": "student"})

    def test_start_inspect_and_cancel_status_codes(self):
        started = self.start()
        self.assertEqual(started.status_code, 201)
        session_id = started.json()["session_id"]
        self.assertEqual(self.client.get(f"/enrollments/{session_id}").status_code, 200)
        self.assertEqual(self.client.delete(f"/enrollments/{session_id}").status_code, 200)
        self.assertEqual(self.client.get(f"/enrollments/{session_id}").status_code, 404)

    def test_invalid_user_id_returns_422_stable_code(self):
        response = self.client.post("/enrollments", json={
            "user_id": "../escape", "full_name": "Bad", "role": "student"})
        self.assertEqual(response.status_code, 422)
        self.assertEqual(response.json()["error_code"], "INVALID_USER_ID")

    def test_sample_acceptance_preview_and_delete(self):
        session_id = self.start().json()["session_id"]
        response = self.client.post(f"/enrollments/{session_id}/samples",
            data={"pose": "front"}, files={"image": ("front.jpg", camera_jpeg(), "image/jpeg")})
        self.assertEqual(response.status_code, 201)
        sample_id = response.json()["sample_id"]
        self.assertEqual(self.client.get(
            f"/enrollments/{session_id}/samples/{sample_id}/preview").status_code, 200)
        deleted = self.client.delete(f"/enrollments/{session_id}/samples/{sample_id}")
        self.assertEqual(deleted.json()["progress"]["front"], 0)

    def test_incomplete_finalize_is_409(self):
        session_id = self.start().json()["session_id"]
        response = self.client.post(f"/enrollments/{session_id}/finalize")
        self.assertEqual(response.status_code, 409)
        self.assertEqual(response.json()["error_code"], "SESSION_NOT_READY")

    def test_completed_session_cannot_be_cancelled(self):
        session_id = self.start().json()["session_id"]
        session = self.service.repository.load_session(session_id)
        session["state"] = "COMPLETE"
        self.service.repository.save_session(session)

        response = self.client.delete(f"/enrollments/{session_id}")

        self.assertEqual(response.status_code, 409)
        self.assertEqual(response.json()["error_code"], "SESSION_FINALIZED")
        self.assertEqual(
            self.client.get(f"/enrollments/{session_id}").json()["state"],
            "COMPLETE",
        )


if __name__ == "__main__":
    unittest.main()
