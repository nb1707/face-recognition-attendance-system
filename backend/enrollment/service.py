import hashlib
from pathlib import Path
import threading
import uuid

import cv2
import numpy as np

from enrollment.config import DEFAULT_POLICY, POSE_TARGETS, REPRESENTATIVE_TARGETS
from enrollment.errors import EnrollmentError, enrollment_error
from enrollment.gallery import LocalGalleryPublisher
from enrollment.pose import evaluate_pose, measure_landmarks
from enrollment.quality import evaluate_quality, measure_quality
from enrollment.repository import EnrollmentRepository, utc_now, validate_identity
from enrollment.representatives import Candidate, select_representatives
from recognition.embeddings import normalized_embedding
from recognition.face_preprocessing import align_detected_face


BACKEND = Path(__file__).resolve().parents[1]


class EnrollmentService:
    def __init__(self, detector, root=None, gallery_root=None, policy=DEFAULT_POLICY,
                 aligner=align_detected_face, model_factory=None, student_sync=None):
        self.repository = EnrollmentRepository(root or BACKEND / "storage" / "enrollments")
        self.publisher = LocalGalleryPublisher(gallery_root or BACKEND / "storage" / "local_gallery")
        self.detector = detector
        self.policy = policy
        self.aligner = aligner
        self.model_factory = model_factory or self._default_model_factory
        self.student_sync = student_sync
        self._locks_guard = threading.Lock()
        self._locks = {}

    def start_session(self, user_id, full_name, role, allow_reenroll=False):
        user_id, full_name, role = validate_identity(user_id, full_name, role)
        already_enrolled = self.repository.user_exists(user_id) or self.publisher.identity_exists(user_id)
        if already_enrolled and not allow_reenroll:
            raise enrollment_error("USER_ALREADY_ENROLLED", status_code=409,
                                   details={"user_id": user_id, "can_create_version": True})
        session = self.repository.create_session(user_id, full_name, role)
        return self.present(session)

    def get_session(self, session_id):
        return self.present(self.repository.load_session(session_id))

    def submit_sample(self, session_id, pose, image_bytes):
        with self._lock(session_id):
            session = self.repository.load_session(session_id)
            try:
                return self._submit_sample(session, str(pose or "").strip().lower(), image_bytes)
            except EnrollmentError as error:
                if session.get("state") not in {"COMPLETE", "FINALIZING"}:
                    session = self.repository.record_rejection(session, error.code, error.details)
                    progress = self.repository.progress(session)
                    error.details.setdefault("pose", str(pose or "").strip().lower())
                    error.details.setdefault("progress", progress)
                    error.details.setdefault("next_recommended_pose", next(
                        (key for key, target in POSE_TARGETS.items() if progress[key] < target), None))
                    error.details.setdefault("remaining_count", max(
                        0, POSE_TARGETS.get(error.details["pose"], 0) - progress.get(error.details["pose"], 0)))
                raise

    def _submit_sample(self, session, pose, image_bytes):
        if session["state"] == "COMPLETE":
            raise enrollment_error("SESSION_FINALIZED", status_code=409)
        if session["state"] == "FINALIZING":
            raise EnrollmentError("SESSION_FINALIZING", "Enrollment is being finalized.", 409)
        if pose not in POSE_TARGETS:
            raise EnrollmentError("INVALID_POSE", "Pose must be front, left, right, up, or down.", 422)
        progress = self.repository.progress(session)
        if progress[pose] >= POSE_TARGETS[pose]:
            raise enrollment_error("POSE_QUOTA_REACHED", status_code=409, details={"pose": pose})
        if not image_bytes:
            raise enrollment_error("INVALID_IMAGE")
        encoded = np.frombuffer(image_bytes, dtype=np.uint8)
        image = cv2.imdecode(encoded, cv2.IMREAD_COLOR)
        if image is None or image.size == 0:
            raise enrollment_error("INVALID_IMAGE")
        faces = self.detector.detect_faces(image)
        if len(faces) == 0:
            raise enrollment_error("NO_FACE")
        if len(faces) > 1:
            raise enrollment_error("MULTIPLE_FACES")
        face = faces[0]
        quality_measurements = measure_quality(image, face)
        quality = evaluate_quality(quality_measurements, self.policy)
        landmarks = measure_landmarks(getattr(face, "kps", None), self.policy.min_eye_distance_pixels)
        pose_decision = evaluate_pose(pose, landmarks, self.policy)
        try:
            aligned = self.aligner(image=image, detected_face=face, output_size=112)
        except EnrollmentError:
            raise
        except Exception as error:
            raise enrollment_error("INVALID_LANDMARKS", details={"reason": str(error)})
        if aligned is None or aligned.shape != (112, 112, 3) or aligned.dtype != np.uint8:
            raise enrollment_error("INVALID_LANDMARKS", details={"reason": "invalid aligned image"})
        crop = self._crop(image, face)
        fingerprint = hashlib.sha256(aligned.tobytes()).hexdigest()
        duplicate = self._duplicate_status(session, aligned, fingerprint)
        if duplicate:
            code, details = duplicate
            raise enrollment_error(code, details=details)
        crop_bytes = self._jpeg(crop)
        aligned_bytes = self._jpeg(aligned)
        sample_id = uuid.uuid4().hex
        metadata = {
            "sample_id": sample_id, "pose": pose, "accepted_at": utc_now(),
            "quality": quality, "pose_decision": pose_decision,
            "duplicate_fingerprint": fingerprint,
            "aligned_sha256": hashlib.sha256(aligned_bytes).hexdigest(),
            "files": {"crop": "crop.jpg", "aligned": "aligned.jpg", "metadata": "metadata.json"},
        }
        session = self.repository.store_sample(session, crop_bytes, aligned_bytes, metadata)
        progress = self.repository.progress(session)
        if self._eligible(progress):
            session["state"] = "REVIEW"
            session = self.repository.save_session(session)
        presented = self.present(session)
        accepted = next(row for row in presented["accepted_samples"] if row["sample_id"] == sample_id)
        return {
            "success": True, "accepted": True, "sample_id": sample_id, "pose": pose,
            "quality": quality, "quality_decision": "ACCEPTED",
            "pose_measurements": pose_decision, "pose_accepted": True,
            "duplicate_status": "UNIQUE", "preview_url": accepted["preview_url"],
            "progress": presented["progress"], "remaining_count": POSE_TARGETS[pose] - progress[pose],
            "next_recommended_pose": presented["next_recommended_pose"],
            "error_code": None, "message": "Sample accepted.", "session": presented,
        }

    def delete_sample(self, session_id, sample_id):
        with self._lock(session_id):
            session = self.repository.load_session(session_id)
            if session["state"] == "COMPLETE":
                raise enrollment_error("SESSION_FINALIZED", status_code=409)
            if session["state"] == "FINALIZING":
                raise EnrollmentError("SESSION_FINALIZING", "Enrollment is being finalized.", 409)
            session = self.repository.delete_sample(session, sample_id)
            session["state"] = "REVIEW" if self._eligible(self.repository.progress(session)) else "CAPTURING"
            return self.present(self.repository.save_session(session))

    def cancel_session(self, session_id):
        with self._lock(session_id):
            self.repository.cancel(session_id)

    def preview_path(self, session_id, sample_id):
        return self.repository.preview_path(session_id, sample_id)

    def finalize(self, session_id):
        with self._lock(session_id):
            session = self.repository.load_session(session_id)
            if session["state"] == "COMPLETE":
                raise enrollment_error("SESSION_FINALIZED", status_code=409)
            progress = self.repository.progress(session)
            if not self._eligible(progress):
                raise enrollment_error("SESSION_NOT_READY", status_code=409,
                                       details={"progress": progress, "targets": dict(POSE_TARGETS)})
            session["state"] = "FINALIZING"
            session["finalization"] = {"status": "running", "started_at": utc_now(),
                                       "step": "GENERATING_EMBEDDINGS"}
            session = self.repository.save_session(session)
            try:
                model = self.model_factory()
                provenance = dict(model.provenance)
                dimension = int(provenance.get("embedding_dimension", 512))
                if dimension != 512:
                    raise ValueError("AdaFace embedding dimension must be 512")
                candidates = []
                for sample in session["samples"]:
                    embedding = self._embedding_for_sample(session, sample, model, provenance, dimension)
                    candidates.append(Candidate(sample["sample_id"], sample["pose"],
                                                sample["quality"]["score"], embedding))
                representatives = select_representatives(candidates, dimension, REPRESENTATIVE_TARGETS)
                mean = normalized_embedding(np.mean([row.embedding for row in representatives], axis=0), dimension)
                session["finalization"]["step"] = "PUBLISHING_GALLERY"
                session = self.repository.save_session(session)
                previous_users = self.repository.snapshot_users()
                published = self.publisher.publish(session, representatives, mean, provenance)
                previous_pointer = published.pop("_previous_pointer")
                try:
                    self.repository.complete_user(session, published["generation"])
                    if self.student_sync is not None:
                        self.student_sync(
                            student_id=session["user_id"],
                            full_name=session["full_name"],
                            role=session["role"],
                        )
                    session["state"] = "COMPLETE"
                    session["finalization"] = {
                        "status": "complete", "completed_at": utc_now(), "step": "COMPLETE",
                        "gallery_generation": published["generation"],
                        "generated_embeddings": len(candidates),
                        "representative_count": len(representatives),
                        "representatives": [{"sample_id": row.sample_id, "pose": row.pose}
                                                for row in representatives],
                        "mean_fallback": True,
                    }
                    session = self.repository.save_session(session)
                except Exception as error:
                    self.publisher.rollback_pointer(published["generation"], previous_pointer)
                    self.repository.restore_users(previous_users)
                    raise enrollment_error("GALLERY_PUBLISH_ERROR", status_code=500,
                                           details={"reason": f"post-publish local commit failed: {error}"})
                return {"success": True, "message": "Enrollment finalized and local gallery published.",
                        "session": self.present(session), **published}
            except EnrollmentError as error:
                self._mark_finalize_error(session, error.code, error.message)
                raise
            except Exception as error:
                self._mark_finalize_error(session, "EMBEDDING_ERROR", str(error))
                raise enrollment_error("EMBEDDING_ERROR", status_code=500,
                                       details={"reason": f"{type(error).__name__}: {error}"})

    def present(self, session):
        progress = self.repository.progress(session)
        samples = []
        per_pose_index = {pose: 0 for pose in POSE_TARGETS}
        for row in session.get("samples", []):
            per_pose_index[row["pose"]] += 1
            samples.append({
                "sample_id": row["sample_id"], "pose": row["pose"],
                "sample_number": per_pose_index[row["pose"]],
                "quality_score": row["quality"]["score"],
                "quality": row["quality"]["measurements"],
                "pose_measurements": row["pose_decision"],
                "preview_url": f"/enrollments/{session['session_id']}/samples/{row['sample_id']}/preview",
            })
        next_pose = next((pose for pose, target in POSE_TARGETS.items() if progress[pose] < target), None)
        return {
            "success": True, "session_id": session["session_id"], "user_id": session["user_id"],
            "full_name": session["full_name"], "role": session["role"], "state": session["state"],
            "created_at": session["created_at"], "updated_at": session["updated_at"],
            "targets": dict(POSE_TARGETS), "progress": progress,
            "accepted_samples": samples, "rejected_summary": session.get("rejected", {}),
            "overall": {"accepted": sum(progress.values()), "target": sum(POSE_TARGETS.values())},
            "next_recommended_pose": next_pose, "finalize_eligible": self._eligible(progress),
            "finalization": session.get("finalization", {}),
        }

    def _embedding_for_sample(self, session, sample, model, provenance, dimension):
        directory = self.repository.sample_dir(session["session_id"], sample["sample_id"])
        aligned_path = directory / "aligned.jpg"
        if not (directory / "crop.jpg").is_file() or not (directory / "metadata.json").is_file():
            raise ValueError(f"incomplete accepted sample {sample['sample_id']}")
        aligned_file_hash = hashlib.sha256(aligned_path.read_bytes()).hexdigest()
        if aligned_file_hash != sample["aligned_sha256"]:
            raise ValueError(f"aligned sample hash mismatch {sample['sample_id']}")
        image = cv2.imread(str(aligned_path))
        if image is None or image.shape != (112, 112, 3):
            raise ValueError(f"invalid aligned sample {sample['sample_id']}")
        cache_path = directory / "embedding.npy"
        cache = sample.get("embedding_cache", {})
        if cache_path.is_file() and cache.get("provenance") == provenance and cache.get("aligned_sha256") == aligned_file_hash:
            return normalized_embedding(np.load(cache_path, allow_pickle=False), dimension)
        embedding = normalized_embedding(model.extract_embedding(image), dimension)
        temporary = directory / f".embedding-{uuid.uuid4().hex}.tmp.npy"
        np.save(temporary, embedding, allow_pickle=False)
        temporary.replace(cache_path)
        sample["embedding_cache"] = {"provenance": provenance, "aligned_sha256": aligned_file_hash}
        self.repository.update_sample_metadata(session["session_id"], sample)
        self.repository.save_session(session)
        return embedding

    def _duplicate_status(self, session, aligned, fingerprint):
        for sample in session.get("samples", []):
            if sample.get("duplicate_fingerprint") == fingerprint:
                return "EXACT_DUPLICATE", {"matching_sample_id": sample["sample_id"], "difference": 0.0}
            path = self.repository.sample_dir(session["session_id"], sample["sample_id"]) / "aligned.jpg"
            previous = cv2.imread(str(path))
            if previous is None or previous.shape != aligned.shape:
                continue
            difference = float(np.mean(cv2.absdiff(previous, aligned)) / 255.0)
            if difference <= self.policy.near_duplicate_max_difference:
                return "NEAR_DUPLICATE", {"matching_sample_id": sample["sample_id"],
                                          "difference": difference,
                                          "threshold": self.policy.near_duplicate_max_difference}
        return None

    def _mark_finalize_error(self, session, code, message):
        session["state"] = "REVIEW"
        session["finalization"] = {"status": "error", "failed_at": utc_now(),
                                   "step": "ERROR", "error_code": code, "message": message}
        self.repository.save_session(session)

    @staticmethod
    def _crop(image, face, margin_ratio=0.25):
        x1, y1, x2, y2 = np.asarray(face.bbox, dtype=int)
        height, width = image.shape[:2]
        margin_x, margin_y = int((x2 - x1) * margin_ratio), int((y2 - y1) * margin_ratio)
        crop = image[max(0, y1-margin_y):min(height, y2+margin_y),
                     max(0, x1-margin_x):min(width, x2+margin_x)]
        if crop.size == 0:
            raise enrollment_error("INVALID_IMAGE")
        return cv2.resize(crop, (224, 224), interpolation=cv2.INTER_LINEAR)

    @staticmethod
    def _jpeg(image):
        ok, encoded = cv2.imencode(".jpg", image, [cv2.IMWRITE_JPEG_QUALITY, 92])
        if not ok:
            raise enrollment_error("STORAGE_ERROR")
        return encoded.tobytes()

    @staticmethod
    def _eligible(progress):
        return all(progress[pose] >= target for pose, target in POSE_TARGETS.items())

    def _lock(self, session_id):
        with self._locks_guard:
            return self._locks.setdefault(str(session_id), threading.RLock())

    @staticmethod
    def _default_model_factory():
        from recognition.adaface_model import AdaFaceModel
        return AdaFaceModel()
