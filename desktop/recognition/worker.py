"""Latest-result worker for offline AdaFace track confirmation."""

from __future__ import annotations

import sys
import threading
import time
import traceback
import logging
import math
import uuid
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Any, Literal

from desktop.detection import DetectionResult

from .cache import (
    ADAFACE_MODEL_NAME,
    RecognitionCache,
    RecognitionCacheError,
    load_adaface_recognition_cache,
)


LOGGER = logging.getLogger(__name__)


RecognitionState = Literal[
    "idle",
    "loading_model",
    "loading_cache",
    "ready",
    "recognizing",
    "stopping",
    "stopped",
    "error",
]


@dataclass(frozen=True)
class RecognitionStatus:
    """Immutable worker state consumed by the Tkinter event loop."""

    state: RecognitionState
    message: str
    revision: int


@dataclass(frozen=True)
class RecognizedFace:
    """One recognition decision in original full-frame coordinates."""

    bbox: tuple[float, float, float, float]
    landmarks: tuple[tuple[float, float], ...] | None
    detector_confidence: float
    face_width: float
    face_height: float
    recognized: bool
    user_id: str | None
    full_name: str | None
    role: str | None
    best_user_id: str | None
    best_full_name: str | None
    best_role: str | None
    similarity: float
    threshold: float
    processing_time_ms: float
    failure_reason: str | None
    aligned_shape: tuple[int, ...] | None
    embedding_dimension: int | None
    embedding_norm: float | None
    track_id: int | None = None
    top2_user_id: str | None = None
    top2_score: float | None = None
    margin: float | None = None
    quality_passed: bool = False
    selected_frames: int = 0
    valid_observations: int = 0
    diagnostics: dict | None = None


@dataclass(frozen=True)
class RecognitionResult:
    """Latest completed recognition result for a detected full frame."""

    revision: int
    detection_revision: int
    frame_sequence: int
    faces: tuple[RecognizedFace, ...]
    processing_time_ms: float
    threshold: float
    cached_identity_count: int
    detection_processing_time_ms: float = 0.0
    acquisition_time: float = 0.0
    stream_id: str = ""
    source_frame: Any = None


class RecognitionWorker:
    """Run AdaFace and cached comparisons outside Tkinter's main thread.

    Exactly one pending detection result is retained. A newer result replaces
    any older unprocessed job while AdaFace is busy.
    """

    def __init__(self) -> None:
        self._state_lock = threading.Lock()
        self._job_lock = threading.Lock()
        self._result_lock = threading.Lock()

        self._thread: threading.Thread | None = None
        self._stop_event: threading.Event | None = None
        self._job_ready_event: threading.Event | None = None

        self._pending_detection: DetectionResult | None = None
        self._latest_result: RecognitionResult | None = None
        self._result_revision = 0

        self._coordinator = None
        self._session_id = uuid.uuid4().hex
        self._attendance_repository = None
        self._attendance_context = None
        self._model: Any | None = None
        self._align_detected_face: Any | None = None
        self._threshold: float | None = None
        self._cache: RecognitionCache | None = None
        self._model_class_name: str | None = None

        self._status = RecognitionStatus(
            state="idle",
            message="AdaFace recognition is idle.",
            revision=0,
        )

    @property
    def status(self) -> RecognitionStatus:
        with self._state_lock:
            return self._status

    @property
    def is_active(self) -> bool:
        with self._state_lock:
            return self._thread is not None and self._thread.is_alive()

    @property
    def threshold(self) -> float | None:
        with self._state_lock:
            return self._threshold

    @property
    def model_class_name(self) -> str | None:
        with self._state_lock:
            return self._model_class_name

    @property
    def attendance_context(self):
        """Active roster-filtered context, or ``None`` in development mode."""
        return self._attendance_context

    def latest_result(
        self,
        after_revision: int = -1,
    ) -> RecognitionResult | None:
        with self._result_lock:
            if (
                self._latest_result is None
                or self._latest_result.revision <= after_revision
            ):
                return None
            return self._latest_result

    def start(self, session_id: str | None = None) -> bool:
        """Start the existing unrestricted development/test recognition path."""
        return self._start(session_id or uuid.uuid4().hex, attendance_repository=None)

    def start_attendance_session(self, session_id: str, repository=None) -> bool:
        """Start fail-closed recognition for one authoritative attendance session."""
        session_id = str(session_id or "").strip()
        if not session_id:
            raise ValueError("An attendance session ID is required")
        if repository is None:
            from attendance.repository import AttendanceRepository
            repository = AttendanceRepository()
        return self._start(session_id, attendance_repository=repository)

    def _start(self, session_id: str, *, attendance_repository) -> bool:
        """Load AdaFace/cache and start recognition in one worker thread."""
        with self._state_lock:
            if self._thread is not None and self._thread.is_alive():
                return False

            self._session_id = session_id
            self._attendance_repository = attendance_repository
            self._attendance_context = None
            self._coordinator = None
            stop_event = threading.Event()
            job_ready_event = threading.Event()
            worker_thread = threading.Thread(
                target=self._run,
                args=(stop_event, job_ready_event),
                name="desktop-adaface-recognition",
                daemon=False,
            )

            self._stop_event = stop_event
            self._job_ready_event = job_ready_event
            self._thread = worker_thread
            self._cache = None
            self._set_status_locked(
                state="loading_model",
                message="Loading AdaFace...",
            )
            self._clear_pending_job()
            self._clear_latest_result()
            worker_thread.start()

        return True

    def submit_detection(self, detection: DetectionResult) -> bool:
        """Replace the pending recognition job with the latest detection."""
        if not self._valid_detection(detection):
            self._set_status(
                state="error",
                message="Recognition error: invalid detection frame.",
            )
            return False

        with self._state_lock:
            worker_active = self._thread is not None and self._thread.is_alive()
            stop_requested = (
                self._stop_event is None
                or self._stop_event.is_set()
            )
            job_ready_event = self._job_ready_event

        if not worker_active or stop_requested or job_ready_event is None:
            return False

        with self._job_lock:
            self._pending_detection = detection
            job_ready_event.set()

        return True

    def stop(
        self,
        *,
        wait: bool = True,
        join_timeout: float = 5.0,
    ) -> bool:
        """Stop accepting recognition jobs and optionally join the worker."""
        if join_timeout < 0:
            raise ValueError("join_timeout cannot be negative.")

        with self._state_lock:
            worker_thread = self._thread
            stop_event = self._stop_event
            job_ready_event = self._job_ready_event

            if worker_thread is None:
                self._set_status_locked(
                    state="stopped",
                    message="Recognition stopped.",
                )
                self._cache = None
                self._clear_pending_job()
                self._clear_latest_result()
                return True

            if stop_event is not None:
                stop_event.set()
            if job_ready_event is not None:
                job_ready_event.set()

            if worker_thread.is_alive():
                self._set_status_locked(
                    state="stopping",
                    message="Stopping recognition...",
                )

        self._clear_pending_job()
        self._clear_latest_result()

        if (
            wait
            and worker_thread is not threading.current_thread()
            and worker_thread.ident is not None
        ):
            worker_thread.join(timeout=join_timeout)

        worker_alive = worker_thread.is_alive()

        if not worker_alive:
            with self._state_lock:
                if self._thread is worker_thread:
                    self._thread = None
                    self._stop_event = None
                    self._job_ready_event = None
                self._cache = None
                self._set_status_locked(
                    state="stopped",
                    message="Recognition stopped.",
                )

        return not worker_alive

    def _run(
        self,
        stop_event: threading.Event,
        job_ready_event: threading.Event,
    ) -> None:
        startup_stage = "recognition worker startup"

        try:
            startup_stage = "backend recognition imports"
            self._ensure_backend_components()

            if stop_event.is_set():
                return

            if self._model is None:
                startup_stage = "AdaFace model loading"
                self._set_status_while_running(
                    stop_event,
                    "loading_model",
                    "Loading AdaFace...",
                )
                self._model = self._get_model(ADAFACE_MODEL_NAME)

                if (
                    self._model is None
                    or not callable(getattr(self._model, "extract_embedding", None))
                    or not callable(getattr(self._model, "compare", None))
                ):
                    raise TypeError("AdaFace model interface is invalid.")

                model_class_name = (
                    f"{type(self._model).__module__}."
                    f"{type(self._model).__qualname__}"
                )
                with self._state_lock:
                    self._model_class_name = model_class_name

                print(
                    "[Recognition startup] "
                    f"AdaFace loaded: class={model_class_name}",
                    flush=True,
                )

            startup_stage = "AdaFace threshold loading"
            from recognition.offline_config import load_offline_config
            self._offline_config = load_offline_config()
            self._threshold = self._offline_config.operating_point.similarity_threshold

            if stop_event.is_set():
                return

            startup_stage = "local AdaFace cache loading"
            self._set_status_while_running(
                stop_event,
                "loading_cache",
                "Loading registered AdaFace embeddings...",
            )
            self._cache = load_adaface_recognition_cache(
                expected_provenance=self._model.provenance,
                status_callback=lambda message: (
                    self._set_status_while_running(
                        stop_event,
                        "loading_cache",
                        message,
                    )
                )
            )

            if self._attendance_repository is not None:
                startup_stage = "attendance session roster filtering"
                from attendance.recognition_integration import (
                    create_session_recognition_context,
                )

                self._attendance_context = create_session_recognition_context(
                    self._attendance_repository,
                    self._session_id,
                    self._cache.gallery,
                )
                self._cache = RecognitionCache(
                    self._attendance_context.filtered_gallery
                )
                if self._attendance_context.missing_gallery_student_ids:
                    LOGGER.warning(
                        "Attendance roster members missing from gallery; "
                        "session_id=%s student_ids=%s",
                        self._session_id,
                        ",".join(
                            self._attendance_context.missing_gallery_student_ids
                        ),
                    )

            if stop_event.is_set():
                return

            identity_count = len(self._cache.identities)
            skipped_count = self._cache.skipped_user_count
            print(
                "[Recognition cache] "
                f"registered_users={self._cache.registered_user_count} "
                "adaface_local_records="
                f"{self._cache.prototype_record_count} "
                f"cached_identities={identity_count} "
                f"skipped_users={skipped_count} "
                "skipped_embeddings="
                f"{self._cache.skipped_embedding_count}",
                flush=True,
            )
            self._set_status_while_running(
                stop_event,
                "ready",
                (
                    f"AdaFace ready. {identity_count} identities cached."
                    + (
                        f" {skipped_count} users skipped without a usable "
                        "AdaFace prototype."
                        if skipped_count
                        else ""
                    )
                ),
            )

            while not stop_event.is_set():
                job_ready_event.wait(timeout=0.1)

                if stop_event.is_set():
                    break

                detection = self._take_pending_job(job_ready_event)
                if detection is None:
                    continue

                if not self._set_status_while_running(
                    stop_event,
                    "recognizing",
                    "Recognizing...",
                ):
                    break

                try:
                    result = self._recognize_detection(
                        detection=detection,
                        stop_event=stop_event,
                    )
                except Exception as error:
                    if not stop_event.is_set():
                        self._log_safe_failure(
                            "recognition result processing",
                            error,
                        )
                        self._set_status_while_running(
                            stop_event,
                            "error",
                            "Recognition error: Face processing failed.",
                        )
                    continue

                if result is None:
                    break

                if not result.faces:
                    message = "No faces detected."
                else:
                    recognized_faces = [
                        face for face in result.faces if face.recognized
                    ]
                    if recognized_faces:
                        best_face = max(
                            recognized_faces,
                            key=lambda face: face.similarity,
                        )
                        message = f"Recognized: {best_face.full_name}"
                    else:
                        best_face = max(
                            result.faces,
                            key=lambda face: face.similarity,
                        )
                        message = {
                            "alignment_failure": (
                                "Recognition error: Face alignment failed."
                            ),
                            "model_inference_error": (
                                "Recognition error: Embedding extraction failed."
                            ),
                            "no_compatible_embeddings": (
                                "Recognition error: No compatible AdaFace "
                                "embeddings."
                            ),
                            "low_similarity": "Unknown.",
                        }.get(best_face.failure_reason, "Unknown.")

                self._set_status_while_running(
                    stop_event,
                    "ready",
                    message,
                )

        except RecognitionCacheError as error:
            if not stop_event.is_set():
                self._log_safe_failure(startup_stage, error)
                self._set_status_while_running(
                    stop_event,
                    "error",
                    f"Recognition error: {error}",
                )
        except ModuleNotFoundError as error:
            if not stop_event.is_set():
                self._log_safe_failure(startup_stage, error)
                self._set_status_while_running(
                    stop_event,
                    "error",
                    "Recognition error: AdaFace dependencies are unavailable.",
                )
        except Exception as error:
            if not stop_event.is_set():
                self._log_safe_failure(startup_stage, error)
                safe_message = {
                    "backend recognition imports": (
                        "Recognition backend could not be loaded."
                    ),
                    "AdaFace model loading": "AdaFace model failed to load.",
                    "AdaFace threshold loading": (
                        "AdaFace threshold configuration could not be loaded."
                    ),
                    "local AdaFace cache loading": (
                        "Registered AdaFace embeddings could not be loaded."
                    ),
                    "attendance session roster filtering": (
                        "Attendance session or roster gallery is unavailable."
                    ),
                }.get(startup_stage, "Recognition startup failed.")
                self._set_status_while_running(
                    stop_event,
                    "error",
                    f"Recognition error: {safe_message}",
                )
        finally:
            self._clear_pending_job()
            self._cache = None

            if stop_event.is_set():
                self._set_status(
                    state="stopped",
                    message="Recognition stopped.",
                )

    def _ensure_backend_components(self) -> None:
        if self._align_detected_face is not None:
            return

        backend_directory = Path(__file__).resolve().parents[2] / "backend"
        backend_path = str(backend_directory)

        if backend_path not in sys.path:
            sys.path.insert(0, backend_path)

        from recognition.face_preprocessing import align_detected_face
        from recognition.model_factory import get_model

        self._align_detected_face = align_detected_face
        self._get_model = get_model

    def _recognize_detection(
        self,
        *,
        detection: DetectionResult,
        stop_event: threading.Event,
    ) -> RecognitionResult | None:
        if self._model is None or self._cache is None or self._threshold is None:
            raise RuntimeError("Recognition worker is not initialized.")

        started_at = time.perf_counter()
        recognized_faces: list[RecognizedFace] = []

        from recognition.coordinator import RecognitionCoordinator
        from recognition.offline_config import load_offline_config
        if self._coordinator is None:
            config = getattr(self, "_offline_config", None) or load_offline_config()
            if self._attendance_repository is not None:
                if self._attendance_context is None:
                    raise RuntimeError(
                        "Attendance recognition context was not initialized"
                    )
                from attendance.recognition_integration import (
                    SQLiteAttendanceEventSink,
                )
                sink = SQLiteAttendanceEventSink(
                    self._attendance_repository,
                    self._attendance_context,
                )
            else:
                from recognition.events import LocalAttendanceSink
                event_path = Path(__file__).resolve().parents[2] / "backend/storage/offline_sessions/attendance.jsonl"
                sink = LocalAttendanceSink(
                    event_path,
                    session_id=self._session_id,
                )
            self._coordinator = RecognitionCoordinator(self._model, self._align_detected_face,
                self._cache.gallery, config, self._session_id, sink)
        if stop_event.is_set():
            return None
        decisions = self._coordinator.process(detection.frame, detection.faces, detection.frame_sequence,
                                             detection.acquisition_time, detection.stream_id)
        operating_point = getattr(
            getattr(self._coordinator, "config", None),
            "operating_point",
            None,
        )
        if decisions and operating_point is None:
            config = (
                getattr(self, "_offline_config", None)
                or load_offline_config()
            )
            operating_point = config.operating_point
        users = {identity.user_id: identity for identity in self._cache.identities}
        for detected, decision in zip(detection.faces, decisions):
            candidate = users.get(decision.match.top1_user_id)
            second_candidate = users.get(decision.match.top2_user_id)
            match = decision.match
            similarity_gate_passed = bool(
                match.top1_user_id is not None
                and match.top1_score is not None
                and math.isfinite(match.top1_score)
                and match.top1_score >= operating_point.similarity_threshold
            )
            margin_gate_passed = bool(
                match.top2_user_id is not None
                and match.top2_user_id != match.top1_user_id
                and match.margin is not None
                and math.isfinite(match.margin)
                and match.margin >= operating_point.margin_threshold
            )
            matcher_diagnostics = {
                "top1": {
                    "user_id": match.top1_user_id,
                    "full_name": candidate.full_name if candidate else None,
                    "score": match.top1_score,
                    "winning_pose": match.top1_winning_pose,
                    "winning_prototype_id": (
                        match.top1_winning_prototype_id
                    ),
                },
                "top2": {
                    "user_id": match.top2_user_id,
                    "full_name": (
                        second_candidate.full_name
                        if second_candidate else None
                    ),
                    "score": match.top2_score,
                    "winning_pose": match.top2_winning_pose,
                    "winning_prototype_id": (
                        match.top2_winning_prototype_id
                    ),
                },
                "margin": match.margin,
                "similarity_threshold": (
                    operating_point.similarity_threshold
                ),
                "margin_threshold": operating_point.margin_threshold,
                "similarity_gate_passed": similarity_gate_passed,
                "margin_gate_passed": margin_gate_passed,
                "gate_a_passed": decision.gate_a_passed,
                "gate_a_reason": (
                    decision.observation_reason or decision.reason
                ),
            }
            serialized_diagnostics = asdict(decision)
            serialized_diagnostics["matcher_diagnostics"] = (
                matcher_diagnostics
            )
            recognized_faces.append(RecognizedFace(
                bbox=detected.bbox, landmarks=detected.landmarks,
                detector_confidence=detected.detector_confidence,
                face_width=detected.face_width, face_height=detected.face_height,
                recognized=decision.confirmed,
                user_id=candidate.user_id if candidate and decision.confirmed else None,
                full_name=candidate.full_name if candidate and decision.confirmed else None,
                role=candidate.role if candidate and decision.confirmed else None,
                best_user_id=candidate.user_id if candidate else None,
                best_full_name=candidate.full_name if candidate else None,
                best_role=candidate.role if candidate else None,
                similarity=decision.match.top1_score if decision.match.top1_score is not None else -1.,
                threshold=self._threshold, processing_time_ms=0.,
                failure_reason=None if decision.confirmed else decision.reason,
                aligned_shape=None, embedding_dimension=None, embedding_norm=None,
                track_id=decision.track_id, top2_user_id=decision.match.top2_user_id,
                top2_score=decision.match.top2_score, margin=decision.match.margin,
                quality_passed=bool(decision.quality_decision and decision.quality_decision.passed),
                selected_frames=decision.selected_frames, valid_observations=decision.valid_observations,
                diagnostics=serialized_diagnostics))

        processing_time_ms = (time.perf_counter() - started_at) * 1000.0

        with self._result_lock:
            if stop_event.is_set():
                return None

            self._result_revision += 1
            result = RecognitionResult(
                revision=self._result_revision,
                detection_revision=detection.revision,
                frame_sequence=detection.frame_sequence,
                faces=tuple(recognized_faces),
                processing_time_ms=processing_time_ms,
                threshold=self._threshold,
                cached_identity_count=len(self._cache.identities),
                detection_processing_time_ms=detection.processing_time_ms,
                acquisition_time=detection.acquisition_time, stream_id=detection.stream_id,
                source_frame=detection.frame,
            )
            self._latest_result = result
            return result

    def _take_pending_job(
        self,
        job_ready_event: threading.Event,
    ) -> DetectionResult | None:
        with self._job_lock:
            detection = self._pending_detection
            self._pending_detection = None
            job_ready_event.clear()
            return detection

    @staticmethod
    def _valid_detection(detection: DetectionResult) -> bool:
        frame = getattr(detection, "frame", None)
        return (
            isinstance(detection, DetectionResult)
            and frame is not None
            and getattr(frame, "size", 0) > 0
        )

    def _clear_pending_job(self) -> None:
        with self._job_lock:
            self._pending_detection = None

    def _clear_latest_result(self) -> None:
        with self._result_lock:
            self._latest_result = None

    @staticmethod
    def _log_safe_failure(stage: str, error: Exception) -> None:
        """Log stack locations and exception types without sensitive values."""
        exception_types = [type(error).__name__]
        stack_parts = [traceback.format_tb(error.__traceback__)]
        cause = error.__cause__

        if cause is not None:
            exception_types.append(type(cause).__name__)
            stack_parts.append(traceback.format_tb(cause.__traceback__))

        safe_stack = "".join(
            line
            for stack in stack_parts
            for line in stack
        )
        LOGGER.error(
            "Recognition failure at %s; exception_types=%s\n%s",
            stage,
            ",".join(exception_types),
            safe_stack,
        )

    def _set_status(
        self,
        *,
        state: RecognitionState,
        message: str,
    ) -> None:
        with self._state_lock:
            self._set_status_locked(state=state, message=message)

    def _set_status_while_running(
        self,
        stop_event: threading.Event,
        state: RecognitionState,
        message: str,
    ) -> bool:
        with self._state_lock:
            if stop_event.is_set():
                return False
            self._set_status_locked(state=state, message=message)
            return True

    def _set_status_locked(
        self,
        *,
        state: RecognitionState,
        message: str,
    ) -> None:
        self._status = RecognitionStatus(
            state=state,
            message=message,
            revision=self._status.revision + 1,
        )


__all__ = [
    "RecognizedFace",
    "RecognitionResult",
    "RecognitionStatus",
    "RecognitionWorker",
]
