"""Latest-frame worker for the existing tiled InsightFace detector."""

from __future__ import annotations

import sys
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

DetectionState = Literal[
    "idle",
    "loading",
    "ready",
    "detecting",
    "stopping",
    "stopped",
    "error",
]


@dataclass(frozen=True)
class DetectionStatus:
    """Immutable worker state consumed by the Tkinter event loop."""

    state: DetectionState
    message: str
    revision: int


@dataclass(frozen=True)
class DetectedFace:
    """Detection metadata expressed in original full-frame coordinates."""

    bbox: tuple[float, float, float, float]
    landmarks: tuple[tuple[float, float], ...] | None
    detector_confidence: float
    face_width: float
    face_height: float


@dataclass(frozen=True)
class DetectionResult:
    """The latest completed tiled-detection result."""

    revision: int
    frame_sequence: int
    frame: Any
    faces: tuple[DetectedFace, ...]
    processing_time_ms: float
    detection_fps: float
    acquisition_time: float = 0.0
    stream_id: str = ""


class DetectionWorker:
    """Run the existing ROI+tiled detector outside Tkinter's main thread.

    Only one pending camera frame is retained. Submitting a newer frame replaces
    an older unprocessed frame, preventing inference latency from accumulating.
    """

    def __init__(self) -> None:
        self._state_lock = threading.Lock()
        self._frame_lock = threading.Lock()
        self._result_lock = threading.Lock()

        self._thread: threading.Thread | None = None
        self._stop_event: threading.Event | None = None
        self._frame_ready_event: threading.Event | None = None

        self._pending_frame: Any | None = None
        self._pending_frame_sequence = -1
        self._pending_acquisition_time = 0.0
        self._pending_stream_id = ""
        self._latest_result: DetectionResult | None = None
        self._result_revision = 0

        self._detector: Any | None = None
        self._classroom_roi: dict[str, float] | None = None
        self._tiled_config: dict[str, Any] | None = None

        self._status = DetectionStatus(
            state="idle",
            message="Face detector is idle.",
            revision=0,
        )

    @property
    def status(self) -> DetectionStatus:
        with self._state_lock:
            return self._status

    @property
    def is_active(self) -> bool:
        with self._state_lock:
            return self._thread is not None and self._thread.is_alive()

    def latest_result(
        self,
        after_revision: int = -1,
    ) -> DetectionResult | None:
        """Return the newest result only when it is new to the caller."""
        with self._result_lock:
            if (
                self._latest_result is None
                or self._latest_result.revision <= after_revision
            ):
                return None
            return self._latest_result

    def start(self) -> bool:
        """Start detector initialization and processing in one worker thread."""
        with self._state_lock:
            if self._thread is not None and (
                self._thread.is_alive()
                or self._status.state in {
                    "loading",
                    "ready",
                    "detecting",
                    "stopping",
                }
            ):
                return False

            stop_event = threading.Event()
            frame_ready_event = threading.Event()
            worker_thread = threading.Thread(
                target=self._run,
                args=(stop_event, frame_ready_event),
                name="desktop-face-detection",
                daemon=False,
            )

            self._stop_event = stop_event
            self._frame_ready_event = frame_ready_event
            self._thread = worker_thread
            self._set_status_locked(
                state="loading",
                message="Loading face detector...",
            )
            self._clear_pending_frame()
            self._clear_latest_result()
            worker_thread.start()

        return True

    def submit_frame(
        self,
        frame: Any,
        frame_sequence: int,
        acquisition_time: float,
        stream_id: str = "",
    ) -> bool:
        """Replace the pending detection input with the newest camera frame."""
        if not self._valid_frame(frame):
            self._set_status(
                state="error",
                message="Detection error: invalid camera frame.",
            )
            return False

        with self._state_lock:
            worker_active = self._thread is not None and self._thread.is_alive()
            stop_requested = (
                self._stop_event is None
                or self._stop_event.is_set()
            )
            frame_ready_event = self._frame_ready_event

        if not worker_active or stop_requested or frame_ready_event is None:
            return False

        with self._frame_lock:
            self._pending_frame = frame
            self._pending_frame_sequence = frame_sequence
            self._pending_acquisition_time = acquisition_time
            self._pending_stream_id = stream_id
            frame_ready_event.set()

        return True

    def stop(
        self,
        *,
        wait: bool = True,
        join_timeout: float = 5.0,
    ) -> bool:
        """Stop accepting frames and optionally join the detection thread."""
        if join_timeout < 0:
            raise ValueError("join_timeout cannot be negative.")

        with self._state_lock:
            worker_thread = self._thread
            stop_event = self._stop_event
            frame_ready_event = self._frame_ready_event

            if worker_thread is None:
                self._set_status_locked(
                    state="stopped",
                    message="Face detector stopped.",
                )
                self._clear_pending_frame()
                self._clear_latest_result()
                return True

            if stop_event is not None:
                stop_event.set()
            if frame_ready_event is not None:
                frame_ready_event.set()

            if worker_thread.is_alive():
                self._set_status_locked(
                    state="stopping",
                    message="Stopping face detector...",
                )

        self._clear_pending_frame()
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
                    self._frame_ready_event = None
                self._set_status_locked(
                    state="stopped",
                    message="Face detector stopped.",
                )

        return not worker_alive

    def _run(
        self,
        stop_event: threading.Event,
        frame_ready_event: threading.Event,
    ) -> None:
        try:
            self._ensure_detector_loaded()

            if stop_event.is_set():
                return

            if not self._set_status_while_running(
                stop_event=stop_event,
                state="ready",
                message="Face detector ready.",
            ):
                return

            while not stop_event.is_set():
                frame_ready_event.wait(timeout=0.1)

                if stop_event.is_set():
                    break

                frame, frame_sequence, acquisition_time, stream_id = self._take_pending_frame(
                    frame_ready_event
                )
                if frame is None:
                    continue

                try:
                    detection_frame = frame.copy()
                except Exception:
                    self._set_status_while_running(
                        stop_event=stop_event,
                        state="error",
                        message="Detection error: invalid camera frame.",
                    )
                    continue

                if not self._set_status_while_running(
                    stop_event=stop_event,
                    state="detecting",
                    message="Detecting...",
                ):
                    break

                detection_started_at = time.perf_counter()

                try:
                    faces = self._detect_tiled(detection_frame)
                    processing_time_ms = (
                        time.perf_counter() - detection_started_at
                    ) * 1000.0
                    serialized_faces = self._serialize_faces(faces)
                except Exception:
                    if not stop_event.is_set():
                        self._set_status_while_running(
                            stop_event=stop_event,
                            state="error",
                            message="Detection error.",
                        )
                    continue

                if stop_event.is_set():
                    break

                detection_fps = (
                    1000.0 / processing_time_ms
                    if processing_time_ms > 0.0
                    else 0.0
                )
                result = self._publish_result(
                    stop_event=stop_event,
                    frame_sequence=frame_sequence,
                    frame=detection_frame,
                    faces=serialized_faces,
                    processing_time_ms=processing_time_ms,
                    detection_fps=detection_fps,
                    acquisition_time=acquisition_time, stream_id=stream_id,
                )

                if result is None:
                    break

                face_count = len(result.faces)
                self._set_status_while_running(
                    stop_event=stop_event,
                    state="ready",
                    message=(
                        f"Detected {face_count} faces."
                        if face_count != 1
                        else "Detected 1 face."
                    ) if face_count else "No faces detected.",
                )

        except ModuleNotFoundError:
            if not stop_event.is_set():
                self._set_status_while_running(
                    stop_event=stop_event,
                    state="error",
                    message="Face detector dependencies are unavailable.",
                )
        except Exception:
            if not stop_event.is_set():
                self._set_status_while_running(
                    stop_event=stop_event,
                    state="error",
                    message="Face detector initialization failed.",
                )
        finally:
            self._clear_pending_frame()

            if stop_event.is_set():
                self._set_status(
                    state="stopped",
                    message="Face detector stopped.",
                )

    def _ensure_detector_loaded(self) -> None:
        if self._detector is not None:
            return

        backend_directory = (
            Path(__file__).resolve().parents[2]
            / "backend"
        )
        backend_path = str(backend_directory)

        if backend_path not in sys.path:
            sys.path.insert(0, backend_path)

        from recognition.detection_config import (
            CLASSROOM_ROI,
            TILED_DETECTION_CONFIG,
        )
        from recognition.face_detector import get_shared_detector
        from recognition.offline_runtime import require_local_detector

        if not TILED_DETECTION_CONFIG.get("enabled", False):
            raise RuntimeError("Tiled face detection is disabled.")

        require_local_detector()
        self._detector = get_shared_detector()
        self._classroom_roi = CLASSROOM_ROI
        self._tiled_config = TILED_DETECTION_CONFIG

    def _detect_tiled(self, frame: Any) -> list[Any]:
        if self._detector is None:
            raise RuntimeError("Face detector is not loaded.")
        if self._classroom_roi is None or self._tiled_config is None:
            raise RuntimeError("Tiled detection configuration is unavailable.")

        from recognition.detection_modes import detect_for_mode, DESKTOP_EQUIVALENT_ROI_TILED
        return detect_for_mode(self._detector, frame, DESKTOP_EQUIVALENT_ROI_TILED,
                               self._classroom_roi, self._tiled_config)

    def _take_pending_frame(
        self,
        frame_ready_event: threading.Event,
    ) -> tuple[Any | None, int, float, str]:
        with self._frame_lock:
            frame = self._pending_frame
            frame_sequence = self._pending_frame_sequence
            acquisition_time = self._pending_acquisition_time
            stream_id = self._pending_stream_id
            self._pending_frame = None
            self._pending_frame_sequence = -1
            frame_ready_event.clear()
            return frame, frame_sequence, acquisition_time, stream_id

    def _publish_result(
        self,
        *,
        stop_event: threading.Event,
        frame_sequence: int,
        frame: Any,
        faces: tuple[DetectedFace, ...],
        processing_time_ms: float,
        detection_fps: float,
        acquisition_time: float,
        stream_id: str,
    ) -> DetectionResult | None:
        with self._result_lock:
            if stop_event.is_set():
                return None

            self._result_revision += 1
            result = DetectionResult(
                revision=self._result_revision,
                frame_sequence=frame_sequence,
                frame=frame,
                faces=faces,
                processing_time_ms=processing_time_ms,
                detection_fps=detection_fps,
                acquisition_time=acquisition_time,
                stream_id=stream_id,
            )
            self._latest_result = result
            return result

    @staticmethod
    def _serialize_faces(faces: list[Any]) -> tuple[DetectedFace, ...]:
        serialized = []

        for face in faces:
            bbox_values = tuple(
                float(value)
                for value in face.bbox.reshape(-1).tolist()
            )
            if len(bbox_values) != 4:
                raise ValueError("Detected face bbox must contain four values.")

            x1, y1, x2, y2 = bbox_values
            landmark_values = getattr(face, "kps", None)
            landmarks = None

            if landmark_values is not None:
                landmarks = tuple(
                    (float(point[0]), float(point[1]))
                    for point in landmark_values
                )

            serialized.append(
                DetectedFace(
                    bbox=(x1, y1, x2, y2),
                    landmarks=landmarks,
                    detector_confidence=float(
                        getattr(face, "det_score", 0.0)
                    ),
                    face_width=max(0.0, x2 - x1),
                    face_height=max(0.0, y2 - y1),
                )
            )

        return tuple(serialized)

    @staticmethod
    def _valid_frame(frame: Any) -> bool:
        if frame is None or getattr(frame, "size", 0) == 0:
            return False
        shape = getattr(frame, "shape", ())
        return len(shape) == 3 and shape[2] >= 3

    def _clear_pending_frame(self) -> None:
        with self._frame_lock:
            self._pending_frame = None
            self._pending_frame_sequence = -1

    def _clear_latest_result(self) -> None:
        with self._result_lock:
            self._latest_result = None

    def _set_status(
        self,
        *,
        state: DetectionState,
        message: str,
    ) -> None:
        with self._state_lock:
            self._set_status_locked(state=state, message=message)

    def _set_status_while_running(
        self,
        *,
        stop_event: threading.Event,
        state: DetectionState,
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
        state: DetectionState,
        message: str,
    ) -> None:
        self._status = DetectionStatus(
            state=state,
            message=message,
            revision=self._status.revision + 1,
        )
