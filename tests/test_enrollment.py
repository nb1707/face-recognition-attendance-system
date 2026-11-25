import json
import hashlib
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "backend"))

from enrollment.config import DEFAULT_POLICY, POSE_TARGETS, REPRESENTATIVE_TARGETS
from enrollment.errors import EnrollmentError
from enrollment.pose import evaluate_pose, measure_landmarks
from enrollment.quality import evaluate_quality, measure_quality
from enrollment.repository import EnrollmentRepository, validate_identity
from enrollment.representatives import Candidate, select_representatives
from enrollment.service import EnrollmentService
from recognition.local_gallery import LocalGalleryProvider
from recognition.identity_matcher import match_identity


def landmarks(pose):
    nose = {"front": (50, 50), "left": (60, 50), "right": (40, 50),
            "up": (50, 45), "down": (50, 62)}[pose]
    return np.array([(30, 30), (70, 30), nose, (36, 75), (64, 75)], np.float32)


def accepted_poses(yaw_ratio, vertical_ratio):
    measurements = {
        "yaw_ratio": yaw_ratio,
        "vertical_ratio": vertical_ratio,
        "eye_distance": 40.0,
        "eye_mouth_distance": 45.0,
        "mouth_width": 28.0,
    }
    accepted = []
    for pose in POSE_TARGETS:
        try:
            evaluate_pose(pose, measurements, DEFAULT_POLICY)
            accepted.append(pose)
        except EnrollmentError as error:
            if error.code != "WRONG_POSE":
                raise
    return accepted


def sharp_image(delta=0):
    pattern = ((np.indices((200, 200)).sum(axis=0) % 2) * 80 + 80 + delta).astype(np.uint8)
    return np.repeat(pattern[:, :, None], 3, axis=2)


def jpeg(image):
    ok, value = cv2.imencode(".jpg", image)
    assert ok
    return value.tobytes()


def face(pose="front", confidence=.9, bbox=(40, 40, 160, 160)):
    return SimpleNamespace(bbox=np.array(bbox, np.float32), kps=landmarks(pose), det_score=confidence)


class Detector:
    def __init__(self, faces): self.faces = faces
    def detect_faces(self, image): return self.faces


class FakeAdaFace:
    provenance = {"model_name": "adaface", "model_revision": "test", "checkpoint_sha256": "abc",
                  "embedding_dimension": 512, "preprocessing_version": "test-v1"}
    def extract_embedding(self, image):
        vector = np.zeros(512, np.float32)
        index = int(np.mean(image)) % 500
        vector[index] = 1
        vector[(index + 7) % 512] = .2
        return vector


class IdentityValidationTests(unittest.TestCase):
    def test_valid_identity(self):
        self.assertEqual(validate_identity("student_01", "Student One", "STUDENT"),
                         ("student_01", "Student One", "student"))

    def test_path_traversal_and_separators_rejected(self):
        for value in ("../user", "a/b", "a\\b", ".", "..", "C:evil", "/absolute"):
            with self.subTest(value=value), self.assertRaises(EnrollmentError):
                validate_identity(value, "Student", "student")

    def test_invalid_name_and_role(self):
        for name, role in (("", "student"), ("Name", "admin")):
            with self.assertRaises(EnrollmentError): validate_identity("valid", name, role)


class PoseTests(unittest.TestCase):
    def test_all_five_pose_measurements_and_acceptance(self):
        for pose in POSE_TARGETS:
            with self.subTest(pose=pose):
                measured = measure_landmarks(landmarks(pose))
                self.assertTrue(evaluate_pose(pose, measured, DEFAULT_POLICY)["accepted"])

    def test_wrong_pose_rejected_for_each_pose(self):
        wrong = {"front": "left", "left": "front", "right": "front", "up": "front", "down": "front"}
        for requested, actual in wrong.items():
            with self.subTest(requested=requested), self.assertRaises(EnrollmentError) as caught:
                evaluate_pose(requested, measure_landmarks(landmarks(actual)), DEFAULT_POLICY)
            self.assertEqual(caught.exception.code, "WRONG_POSE")

    def test_horizontal_pose_boundaries_and_dead_zones(self):
        policy = DEFAULT_POLICY
        neutral = (policy.front_min_vertical_ratio + policy.front_max_vertical_ratio) / 2
        left_dead_zone = (policy.front_max_yaw + policy.side_min_yaw) / 2
        right_dead_zone = -left_dead_zone

        cases = (
            (-policy.side_min_yaw, ["right"]),
            (right_dead_zone, []),
            (-policy.front_max_yaw, ["front"]),
            (0.0, ["front"]),
            (policy.front_max_yaw, ["front"]),
            (left_dead_zone, []),
            (policy.side_min_yaw, ["left"]),
        )
        for yaw, expected in cases:
            with self.subTest(yaw=yaw):
                self.assertEqual(accepted_poses(yaw, neutral), expected)

    def test_vertical_pose_bands_are_exclusive_from_front_and_sides(self):
        policy = DEFAULT_POLICY
        up = (policy.up_min_vertical_ratio + policy.up_max_vertical_ratio) / 2
        up_dead_zone = (policy.up_max_vertical_ratio + policy.front_min_vertical_ratio) / 2
        neutral = (policy.front_min_vertical_ratio + policy.front_max_vertical_ratio) / 2
        down_dead_zone = (policy.front_max_vertical_ratio + policy.down_min_vertical_ratio) / 2
        down = (policy.down_min_vertical_ratio + policy.down_max_vertical_ratio) / 2

        self.assertEqual(accepted_poses(0.0, up), ["up"])
        self.assertEqual(accepted_poses(0.0, up_dead_zone), [])
        self.assertEqual(accepted_poses(0.0, neutral), ["front"])
        self.assertEqual(accepted_poses(0.0, down_dead_zone), [])
        self.assertEqual(accepted_poses(0.0, down), ["down"])

        # Mixed side-yaw and vertical-pose geometry is outside every valid band.
        self.assertEqual(accepted_poses(policy.side_min_yaw, up), [])
        self.assertEqual(accepted_poses(-policy.side_min_yaw, down), [])

    def test_policy_rejects_overlapping_or_misordered_bands(self):
        from enrollment.config import EnrollmentPolicy
        with self.assertRaises(ValueError):
            EnrollmentPolicy(side_min_yaw=DEFAULT_POLICY.front_max_yaw)
        with self.assertRaises(ValueError):
            EnrollmentPolicy(vertical_pose_max_yaw=DEFAULT_POLICY.side_min_yaw)

    def test_invalid_landmark_shape_nonfinite_and_degenerate(self):
        bad = [np.zeros((4, 2)), np.full((5, 2), np.nan), np.array([(1, 1)] * 5)]
        for value in bad:
            with self.assertRaises(EnrollmentError) as caught: measure_landmarks(value)
            self.assertEqual(caught.exception.code, "INVALID_LANDMARKS")


class QualityTests(unittest.TestCase):
    def test_good_close_up_quality(self):
        measured = measure_quality(sharp_image(), face())
        result = evaluate_quality(measured, DEFAULT_POLICY)
        self.assertTrue(result["accepted"])
        self.assertEqual(measured["detector_confidence"], .9)

    def test_blur_lighting_confidence_size_and_center(self):
        cases = [
            (np.full((200, 200, 3), 120, np.uint8), face(), "BLURRED_FACE"),
            (np.repeat((((np.indices((200, 200)).sum(axis=0) % 2) * 20 + 5).astype(np.uint8))[:, :, None], 3, axis=2), face(), "POOR_LIGHTING"),
            (sharp_image(), face(confidence=.2), "LOW_DETECTOR_CONFIDENCE"),
            (sharp_image(), face(bbox=(90, 90, 105, 105)), "FACE_TOO_SMALL"),
            (sharp_image(), face(bbox=(0, 40, 80, 160)), "FACE_NOT_CENTERED"),
        ]
        for image, detected, code in cases:
            with self.subTest(code=code), self.assertRaises(EnrollmentError) as caught:
                evaluate_quality(measure_quality(image, detected), DEFAULT_POLICY)
            self.assertEqual(caught.exception.code, code)


class RepositoryTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.repo = EnrollmentRepository(Path(self.temp.name) / "enrollments")
        self.session = self.repo.create_session("student", "Student", "student")

    def tearDown(self): self.temp.cleanup()

    def test_start_resume_and_cancel(self):
        loaded = self.repo.load_session(self.session["session_id"])
        self.assertEqual(loaded["targets"], dict(POSE_TARGETS))
        self.repo.cancel(self.session["session_id"])
        with self.assertRaises(EnrollmentError): self.repo.load_session(self.session["session_id"])

    def test_unique_sample_storage_progress_delete(self):
        for index in range(2):
            sample_id = f"{index + 1:032x}"
            metadata = {"sample_id": sample_id, "pose": "front", "quality": {"score": .8, "measurements": {}},
                        "pose_decision": {}, "aligned_sha256": str(index), "files": {}}
            self.session = self.repo.store_sample(self.session, b"crop", b"aligned", metadata)
        self.assertEqual(self.repo.progress(self.session)["front"], 2)
        self.assertNotEqual(self.session["samples"][0]["sample_id"], self.session["samples"][1]["sample_id"])
        self.session = self.repo.delete_sample(self.session, self.session["samples"][0]["sample_id"])
        self.assertEqual(self.repo.progress(self.session)["front"], 1)

    def test_local_user_metadata_is_atomic_and_versioned(self):
        self.repo.complete_user(self.session, "generation-one")
        user = self.repo.load_users()["users"]["student"]
        self.assertEqual(user["gallery_generation"], "generation-one")
        self.assertEqual(user["latest_completed_enrollment_session"], self.session["session_id"])


class ServiceCaptureTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        root = Path(self.temp.name)
        self.service = EnrollmentService(Detector([face()]), root / "enrollments", root / "gallery",
            aligner=lambda image, detected_face, output_size: cv2.resize(image, (112, 112)), model_factory=FakeAdaFace)
        self.session = self.service.start_session("student", "Student", "student")

    def tearDown(self): self.temp.cleanup()

    def test_sample_acceptance_response_and_resume(self):
        result = self.service.submit_sample(self.session["session_id"], "front", jpeg(sharp_image()))
        self.assertTrue(result["accepted"])
        self.assertEqual(result["progress"]["front"], 1)
        self.assertIn("/preview", result["preview_url"])
        self.assertEqual(self.service.get_session(self.session["session_id"])["progress"]["front"], 1)

    def test_no_face_and_multiple_faces(self):
        for faces, code in (([], "NO_FACE"), ([face(), face()], "MULTIPLE_FACES")):
            self.service.detector = Detector(faces)
            with self.assertRaises(EnrollmentError) as caught:
                self.service.submit_sample(self.session["session_id"], "front", jpeg(sharp_image()))
            self.assertEqual(caught.exception.code, code)

    def test_exact_and_near_duplicates(self):
        self.service.submit_sample(self.session["session_id"], "front", jpeg(sharp_image()))
        with self.assertRaises(EnrollmentError) as exact:
            self.service.submit_sample(self.session["session_id"], "front", jpeg(sharp_image()))
        self.assertEqual(exact.exception.code, "EXACT_DUPLICATE")
        with self.assertRaises(EnrollmentError) as near:
            self.service.submit_sample(self.session["session_id"], "front", jpeg(sharp_image(1)))
        self.assertEqual(near.exception.code, "NEAR_DUPLICATE")

    def test_wrong_pose_and_invalid_image(self):
        with self.assertRaises(EnrollmentError) as wrong:
            self.service.submit_sample(self.session["session_id"], "left", jpeg(sharp_image()))
        self.assertEqual(wrong.exception.code, "WRONG_POSE")
        with self.assertRaises(EnrollmentError) as invalid:
            self.service.submit_sample(self.session["session_id"], "front", b"bad")
        self.assertEqual(invalid.exception.code, "INVALID_IMAGE")

    def test_pose_quota_reached(self):
        session = self.service.repository.load_session(self.session["session_id"])
        for index in range(5):
            session["samples"].append({"sample_id": f"{index + 1:032x}", "pose": "front"})
        self.service.repository.save_session(session)
        with self.assertRaises(EnrollmentError) as caught:
            self.service.submit_sample(self.session["session_id"], "front", jpeg(sharp_image()))
        self.assertEqual(caught.exception.code, "POSE_QUOTA_REACHED")

    def test_delete_updates_server_progress(self):
        result = self.service.submit_sample(self.session["session_id"], "front", jpeg(sharp_image()))
        session = self.service.delete_sample(self.session["session_id"], result["sample_id"])
        self.assertEqual(session["progress"]["front"], 0)

    def test_duplicate_completed_user_requires_explicit_version(self):
        raw = self.service.repository.load_session(self.session["session_id"])
        self.service.repository.complete_user(raw, "old")
        with self.assertRaises(EnrollmentError) as caught:
            self.service.start_session("student", "Student", "student")
        self.assertEqual(caught.exception.code, "USER_ALREADY_ENROLLED")
        replacement = self.service.start_session("student", "Student", "student", True)
        self.assertNotEqual(replacement["session_id"], self.session["session_id"])


class RepresentativeTests(unittest.TestCase):
    def test_counts_determinism_quality_and_diversity(self):
        rows = []
        for pose, count in POSE_TARGETS.items():
            for index in range(count):
                vector = np.zeros(512); vector[index + list(POSE_TARGETS).index(pose) * 10] = 1
                rows.append(Candidate(f"{pose}-{index}", pose, .9 - index * .01, vector))
        first = select_representatives(rows)
        second = select_representatives(list(reversed(rows)))
        self.assertEqual([x.sample_id for x in first], [x.sample_id for x in second])
        self.assertEqual(len(first), sum(REPRESENTATIVE_TARGETS.values()))
        self.assertEqual(sum(row.pose == "front" for row in first), 2)
        self.assertNotEqual(first[0].sample_id, first[1].sample_id)


class FinalizationTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        root = Path(self.temp.name)
        self.service = EnrollmentService(Detector([face()]), root / "enrollments", root / "gallery",
            aligner=lambda image, detected_face, output_size: cv2.resize(image, (112, 112)), model_factory=FakeAdaFace)
        presented = self.service.start_session("new-user", "New User", "student")
        self.session_id = presented["session_id"]
        session = self.service.repository.load_session(self.session_id)
        number = 0
        for pose, count in POSE_TARGETS.items():
            for _ in range(count):
                number += 1
                image = np.full((112, 112, 3), 50 + number * 5, np.uint8)
                aligned_bytes = jpeg(image)
                metadata = {"sample_id": f"{number:032x}", "pose": pose,
                    "quality": {"score": .70 + number / 100, "measurements": {}},
                    "pose_decision": {"accepted": True},
                    "aligned_sha256": hashlib.sha256(aligned_bytes).hexdigest(), "files": {}}
                session = self.service.repository.store_sample(
                    session, jpeg(cv2.resize(image, (224, 224))), aligned_bytes, metadata)

    def tearDown(self): self.temp.cleanup()

    def test_incomplete_finalize_rejected(self):
        session = self.service.repository.load_session(self.session_id)
        session = self.service.repository.delete_sample(session, session["samples"][0]["sample_id"])
        with self.assertRaises(EnrollmentError) as caught: self.service.finalize(self.session_id)
        self.assertEqual(caught.exception.code, "SESSION_NOT_READY")

    def test_success_generates_15_embeddings_six_prototypes_and_mean(self):
        result = self.service.finalize(self.session_id)
        self.assertEqual(result["prototype_count"], 6)
        finalization = result["session"]["finalization"]
        self.assertEqual(finalization["generated_embeddings"], 15)
        manifest = Path(result["manifest"])
        data = json.loads(manifest.read_text())
        user_rows = [row for row in data["embeddings"] if row["user_id"] == "new-user"]
        self.assertEqual(len(user_rows), 7)
        self.assertEqual(sum(row["mean_fallback"] for row in user_rows), 1)
        for row in user_rows:
            vector = np.load(manifest.parent / row["embedding_file"], allow_pickle=False)
            self.assertEqual(vector.shape, (512,))
            self.assertTrue(np.isfinite(vector).all())
            self.assertAlmostEqual(float(np.linalg.norm(vector)), 1, places=5)
        gallery = LocalGalleryProvider(manifest).load(FakeAdaFace.provenance)
        self.assertEqual(len(gallery.identities), 1)
        with self.assertRaises(EnrollmentError) as caught: self.service.finalize(self.session_id)
        self.assertEqual(caught.exception.code, "SESSION_FINALIZED")

    def test_success_syncs_student_without_membership_or_session(self):
        from attendance.import_enrollment_users import upsert_enrollment_student
        from attendance.repository import AttendanceRepository

        attendance = AttendanceRepository(Path(self.temp.name) / "attendance.db")
        attendance.create_class("class_1", "COMP1682-A", "Demo Class")
        self.service.student_sync = lambda **identity: upsert_enrollment_student(
            attendance, **identity
        )

        result = self.service.finalize(self.session_id)

        users = self.service.repository.load_users()["users"]
        self.assertIn("new-user", users)
        self.assertTrue(Path(result["manifest"]).is_file())
        student = attendance.get_student("new-user")
        self.assertIsNotNone(student)
        self.assertEqual(student.full_name, "New User")
        self.assertEqual(student.role, "student")
        self.assertEqual(attendance.list_class_roster("class_1"), [])
        with attendance.database.connection() as connection:
            session_count = connection.execute(
                "SELECT COUNT(*) FROM attendance_sessions"
            ).fetchone()[0]
        self.assertEqual(session_count, 0)

    def test_sync_preserves_existing_inactive_student_state(self):
        from attendance.import_enrollment_users import upsert_enrollment_student
        from attendance.repository import AttendanceRepository

        attendance = AttendanceRepository(Path(self.temp.name) / "attendance.db")
        attendance.upsert_student("new-user", "Old Name", active=False)
        self.service.student_sync = lambda **identity: upsert_enrollment_student(
            attendance, **identity
        )

        self.service.finalize(self.session_id)

        student = attendance.get_student("new-user")
        self.assertEqual(student.full_name, "New User")
        self.assertFalse(student.active)

    def test_copy_forward_preserves_old_identity(self):
        root = self.service.publisher.output_root
        old = root / "old"; (old / "embeddings").mkdir(parents=True)
        np.save(old / "embeddings/old.npy", np.eye(1, 512, dtype=np.float32)[0])
        manifest = {"schema_version": 1, "generation": "old", "provenance": FakeAdaFace.provenance,
                    "embeddings": [{"user_id": "old-user", "pose": "front", "embedding_file": "embeddings/old.npy", "mean_fallback": False}]}
        (old / "manifest.json").write_text(json.dumps(manifest))
        (root / "current.json").write_text(json.dumps({"manifest": "old/manifest.json"}))
        result = self.service.finalize(self.session_id)
        gallery = LocalGalleryProvider(result["manifest"]).load(FakeAdaFace.provenance)
        self.assertEqual({row.user_id for row in gallery.identities}, {"old-user", "new-user"})

    def test_publish_failure_preserves_current_pointer_and_review_session(self):
        root = self.service.publisher.output_root
        old = root / "old"; old.mkdir(parents=True)
        incompatible = dict(FakeAdaFace.provenance, preprocessing_version="other")
        (old / "manifest.json").write_text(json.dumps({"schema_version": 1, "generation": "old",
            "provenance": incompatible, "embeddings": []}))
        pointer = json.dumps({"manifest": "old/manifest.json"})
        (root / "current.json").write_text(pointer)
        with self.assertRaises(EnrollmentError) as caught: self.service.finalize(self.session_id)
        self.assertEqual(caught.exception.code, "GALLERY_PUBLISH_ERROR")
        self.assertEqual((root / "current.json").read_text(), pointer)
        self.assertEqual(self.service.get_session(self.session_id)["state"], "REVIEW")

    def test_cancel_partial_reenrollment_preserves_active_gallery(self):
        root = self.service.publisher.output_root
        old = root / "old"
        (old / "embeddings").mkdir(parents=True)
        np.save(
            old / "embeddings/old.npy",
            np.eye(1, 512, dtype=np.float32)[0],
        )
        manifest = {
            "schema_version": 1,
            "generation": "old",
            "provenance": FakeAdaFace.provenance,
            "embeddings": [{
                "user_id": "old-user",
                "full_name": "Old User",
                "role": "student",
                "pose": "front",
                "embedding_file": "embeddings/old.npy",
                "mean_fallback": False,
            }],
        }
        (old / "manifest.json").write_text(json.dumps(manifest))
        pointer = json.dumps({"manifest": "old/manifest.json"})
        (root / "current.json").write_text(pointer)
        replacement = self.service.start_session(
            "old-user",
            "Old User",
            "student",
            True,
        )
        replacement_dir = self.service.repository.session_dir(
            replacement["session_id"]
        )

        self.service.cancel_session(replacement["session_id"])

        self.assertFalse(replacement_dir.exists())
        self.assertEqual((root / "current.json").read_text(), pointer)
        gallery = LocalGalleryProvider(root / "current.json").load(
            FakeAdaFace.provenance
        )
        self.assertEqual(
            {identity.user_id for identity in gallery.identities},
            {"old-user"},
        )

    def test_completed_enrollment_cannot_be_cancelled(self):
        self.service.repository.save_session(
            dict(
                self.service.repository.load_session(self.session_id),
                state="REVIEW",
            )
        )
        result = self.service.finalize(self.session_id)
        pointer = self.service.publisher.pointer.read_text()

        with self.assertRaises(EnrollmentError) as caught:
            self.service.cancel_session(self.session_id)

        self.assertEqual(caught.exception.code, "SESSION_FINALIZED")
        self.assertEqual(
            self.service.get_session(self.session_id)["state"],
            "COMPLETE",
        )
        self.assertEqual(self.service.publisher.pointer.read_text(), pointer)


class GalleryCompatibilityTests(unittest.TestCase):
    def test_old_manifest_and_multiple_new_prototypes_load(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp); provenance = {"embedding_dimension": 2}
            np.save(root / "a.npy", np.array([1., 0.])); np.save(root / "b.npy", np.array([0., 1.]))
            for schema, rows in ((1, [{"user_id": "u", "pose": "front", "embedding_file": "a.npy", "mean_fallback": False}]),
                (2, [{"user_id": "u", "pose": "front", "prototype_id": "front_01", "embedding_file": "a.npy", "mean_fallback": False},
                     {"user_id": "u", "pose": "front", "prototype_id": "front_02", "embedding_file": "b.npy", "mean_fallback": False}])):
                path = root / f"m{schema}.json"; path.write_text(json.dumps({"schema_version": schema, "generation": str(schema), "provenance": provenance, "embeddings": rows}))
                gallery = LocalGalleryProvider(path).load(provenance)
                self.assertEqual(len(gallery.identities[0].prototypes), len(rows))
                if schema == 1:
                    self.assertEqual(
                        gallery.identities[0].prototypes[0].prototype_id,
                        "front_01",
                    )

    def test_matcher_consumes_multiple_and_vertical_pose_prototypes(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp); provenance = {"embedding_dimension": 2}
            rows = []
            for prototype_id, pose, vector in (("front_01", "front", [0., 1.]),
                    ("front_02", "front", [1., 0.]), ("up_01", "up", [.8, .2]),
                    ("down_01", "down", [.2, .8])):
                filename = f"{prototype_id}.npy"; np.save(root / filename, np.array(vector))
                rows.append({"user_id": "u", "pose": pose, "prototype_id": prototype_id,
                             "embedding_file": filename, "mean_fallback": False})
            path = root / "manifest.json"
            path.write_text(json.dumps({"schema_version": 2, "generation": "new",
                "provenance": provenance, "embeddings": rows}))
            identity = LocalGalleryProvider(path).load(provenance).identities[0]
            result = match_identity([1, 0], [identity], 2)
            self.assertEqual((result.top1_user_id, result.top1_winning_pose, result.comparisons),
                             ("u", "front", 4))
            self.assertEqual(result.top1_winning_prototype_id, "front_02")


if __name__ == "__main__":
    unittest.main()
