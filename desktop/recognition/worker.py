"""Latest-result worker for baseline single-frame AdaFace recognition."""

from __future__ import annotations

import sys
import threading
import time
import traceback
import logging
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Literal

import numpy as np

from desktop.detection import DetectedFace, DetectionResult

from .cache import (
    ADAFACE_MODEL_NAME,
    CachedIdentity,
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

        self._model: Any | None = None
        self._align_detected_face: Any | None = None
        self._threshold: float | None = None
        self._cache: RecognitionCache | None = None
        self._model_class_name: str | None = None
        self._first_face_diagnostics_reported = False

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

    def start(self) -> bool:
        """Load AdaFace/cache and start recognition in one worker thread."""
        with self._state_lock:
            if self._thread is not None and self._thread.is_alive():
                return False

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
            self._first_face_diagnostics_reported = False
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
            self._threshold = float(
                self._get_model_threshold(ADAFACE_MODEL_NAME)
            )

            if stop_event.is_set():
                return

            startup_stage = "Supabase AdaFace cache loading"
            self._set_status_while_running(
                stop_event,
                "loading_cache",
                "Loading registered AdaFace embeddings...",
            )
            self._cache = load_adaface_recognition_cache(
                status_callback=lambda message: (
                    self._set_status_while_running(
                        stop_event,
                        "loading_cache",
                        message,
                    )
                )
            )

            if stop_event.is_set():
                return

            identity_count = len(self._cache.identities)
            skipped_count = self._cache.skipped_user_count
            print(
                "[Recognition cache] "
                f"registered_users={self._cache.registered_user_count} "
                "adaface_mean_records="
                f"{self._cache.mean_embedding_record_count} "
                "downloaded="
                f"{self._cache.downloaded_embedding_count} "
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
                        "AdaFace mean embedding."
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
                    "Supabase AdaFace cache loading": (
                        "Registered AdaFace embeddings could not be loaded."
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
        from recognition.thresholds import get_model_threshold

        self._align_detected_face = align_detected_face
        self._get_model = get_model
        self._get_model_threshold = get_model_threshold

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

        for detected_face in detection.faces:
            if stop_event.is_set():
                return None

            recognized_faces.append(
                self._recognize_face(
                    frame=detection.frame,
                    detected_face=detected_face,
                    identities=self._cache.identities,
                    threshold=self._threshold,
                )
            )

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
            )
            self._latest_result = result
            return result

    def _recognize_face(
        self,
        *,
        frame: Any,
        detected_face: DetectedFace,
        identities: tuple[CachedIdentity, ...],
        threshold: float,
    ) -> RecognizedFace:
        face_started_at = time.perf_counter()
        base_data = {
            "bbox": detected_face.bbox,
            "landmarks": detected_face.landmarks,
            "detector_confidence": detected_face.detector_confidence,
            "face_width": detected_face.face_width,
            "face_height": detected_face.face_height,
            "threshold": threshold,
        }

        if detected_face.landmarks is None or len(detected_face.landmarks) != 5:
            failed_face = self._failed_face(
                base_data,
                face_started_at,
                "alignment_failure",
            )
            self._report_first_face_diagnostics(failed_face)
            return failed_face

        face_adapter = SimpleNamespace(
            bbox=np.asarray(detected_face.bbox, dtype=np.float32),
            kps=np.asarray(detected_face.landmarks, dtype=np.float32),
        )

        try:
            aligned_face = self._align_detected_face(
                image=frame,
                detected_face=face_adapter,
                output_size=112,
            )
            aligned_shape = tuple(int(value) for value in aligned_face.shape)
        except Exception as error:
            self._log_safe_failure("face alignment", error)
            failed_face = self._failed_face(
                base_data,
                face_started_at,
                "alignment_failure",
            )
            self._report_first_face_diagnostics(failed_face)
            return failed_face

        try:
            query_embedding = self._model.extract_embedding(aligned_face)
            query_embedding = np.asarray(
                query_embedding,
                dtype=np.float32,
            ).reshape(-1)
            embedding_norm = float(np.linalg.norm(query_embedding))

            if (
                query_embedding.size == 0
                or not np.all(np.isfinite(query_embedding))
                or not np.isfinite(embedding_norm)
                or embedding_norm == 0.0
            ):
                raise ValueError("AdaFace returned an invalid embedding.")
        except Exception as error:
            self._log_safe_failure("AdaFace embedding extraction", error)
            failed_face = self._failed_face(
                base_data,
                face_started_at,
                "model_inference_error",
            )
            self._report_first_face_diagnostics(failed_face)
            return failed_face

        best_identity: CachedIdentity | None = None
        best_similarity = -1.0
        first_comparison_error: Exception | None = None

        for identity in identities:
            try:
                similarity = float(
                    self._model.compare(
                        query_embedding,
                        identity.embedding,
                    )
                )
            except Exception as error:
                if first_comparison_error is None:
                    first_comparison_error = error
                continue

            if similarity > best_similarity:
                best_similarity = similarity
                best_identity = identity

        if best_identity is None:
            if first_comparison_error is not None:
                self._log_safe_failure(
                    "AdaFace cosine comparison",
                    first_comparison_error,
                )
            failed_face = self._failed_face(
                base_data,
                face_started_at,
                "no_compatible_embeddings",
            )
            self._report_first_face_diagnostics(failed_face)
            return failed_face

        recognized = best_similarity >= threshold
        elapsed_ms = (time.perf_counter() - face_started_at) * 1000.0

        recognized_face = RecognizedFace(
            **base_data,
            recognized=recognized,
            user_id=best_identity.user_id if recognized else None,
            full_name=best_identity.full_name if recognized else None,
            role=best_identity.role if recognized else None,
            best_user_id=best_identity.user_id,
            best_full_name=best_identity.full_name,
            best_role=best_identity.role,
            similarity=best_similarity,
            processing_time_ms=elapsed_ms,
            failure_reason=None if recognized else "low_similarity",
            aligned_shape=aligned_shape,
            embedding_dimension=int(query_embedding.size),
            embedding_norm=embedding_norm,
        )
        self._report_first_face_diagnostics(recognized_face)
        return recognized_face

    def _failed_face(
        self,
        base_data: dict[str, Any],
        started_at: float,
        reason: str,
    ) -> RecognizedFace:
        return RecognizedFace(
            **base_data,
            recognized=False,
            user_id=None,
            full_name=None,
            role=None,
            best_user_id=None,
            best_full_name=None,
            best_role=None,
            similarity=-1.0,
            processing_time_ms=(time.perf_counter() - started_at) * 1000.0,
            failure_reason=reason,
            aligned_shape=None,
            embedding_dimension=None,
            embedding_norm=None,
        )

    def _report_first_face_diagnostics(self, face: RecognizedFace) -> None:
        if self._first_face_diagnostics_reported:
            return

        self._first_face_diagnostics_reported = True
        decision = "Recognized" if face.recognized else "Unknown"
        print(
            "[Recognition face] "
            f"bbox={face.bbox} "
            f"landmarks={len(face.landmarks) if face.landmarks else 0} "
            f"detector_confidence={face.detector_confidence:.4f} "
            f"aligned_shape={face.aligned_shape} "
            f"embedding_dimension={face.embedding_dimension} "
            f"embedding_norm={face.embedding_norm} "
            f"best_user_id={face.best_user_id or '--'} "
            f"best_name={face.best_full_name or '--'} "
            f"best_similarity={face.similarity:.4f} "
            f"threshold={face.threshold:.2f} "
            f"decision={decision} "
            f"failure_reason={face.failure_reason or '--'}",
            flush=True,
        )

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
