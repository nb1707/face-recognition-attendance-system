"""Single-worker OpenCV camera capture for Tkinter previews."""

from __future__ import annotations

import threading
from dataclasses import dataclass
from typing import Any, Literal

try:
    import cv2
except ImportError:
    cv2 = None


CameraState = Literal[
    "idle",
    "connecting",
    "connected",
    "stopping",
    "stopped",
    "error",
]


@dataclass(frozen=True)
class CameraStatus:
    """Thread-safe status snapshot exposed to the Tkinter event loop."""

    state: CameraState
    message: str
    revision: int


@dataclass(frozen=True)
class CameraFrame:
    """The newest available BGR frame and its sequence number."""

    sequence: int
    image: Any


class CameraWorker:
    """Own one camera capture and retain only its latest frame.

    The worker has no Tkinter dependency. The UI thread polls ``latest_frame``
    and ``status``; the capture thread never calls a widget or ``after``.
    """

    MAX_CONSECUTIVE_READ_FAILURES = 30
    READ_RETRY_SECONDS = 0.1
    RTSP_OPEN_TIMEOUT_MS = 5000
    RTSP_READ_TIMEOUT_MS = 2000

    def __init__(self) -> None:
        self._state_lock = threading.Lock()
        self._frame_lock = threading.Lock()

        self._thread: threading.Thread | None = None
        self._stop_event: threading.Event | None = None
        self._capture: Any | None = None
        self._source_kind: Literal["webcam", "rtsp"] | None = None

        self._status = CameraStatus(
            state="idle",
            message="Camera is idle.",
            revision=0,
        )
        self._latest_frame: Any | None = None
        self._frame_sequence = 0

    @property
    def status(self) -> CameraStatus:
        """Return the most recent immutable status snapshot."""
        with self._state_lock:
            return self._status

    @property
    def is_active(self) -> bool:
        """Return whether a camera thread is still alive."""
        with self._state_lock:
            return self._thread is not None and self._thread.is_alive()

    def latest_frame(
        self,
        after_sequence: int = -1,
    ) -> CameraFrame | None:
        """Return the newest frame only when it is newer than the caller's.

        Exactly one frame is retained. New camera frames overwrite stale ones,
        so a slow display cannot create an unbounded backlog.
        """
        with self._frame_lock:
            if (
                self._latest_frame is None
                or self._frame_sequence <= after_sequence
            ):
                return None

            return CameraFrame(
                sequence=self._frame_sequence,
                image=self._latest_frame,
            )

    def start(self, source: int | str) -> bool:
        """Start one background capture for a webcam index or RTSP source."""
        source_kind = self._validate_source(source)

        with self._state_lock:
            if self._thread is not None and (
                self._thread.is_alive()
                or self._status.state in {
                    "connecting",
                    "connected",
                    "stopping",
                }
            ):
                return False

            if cv2 is None:
                self._set_status_locked(
                    state="error",
                    message=(
                        "OpenCV is unavailable. Camera preview cannot start."
                    ),
                )
                return False

            stop_event = threading.Event()
            worker_thread = threading.Thread(
                target=self._capture_loop,
                args=(source, source_kind, stop_event),
                name="desktop-camera-preview",
                daemon=False,
            )

            self._stop_event = stop_event
            self._thread = worker_thread
            self._capture = None
            self._source_kind = source_kind
            self._set_status_locked(
                state="connecting",
                message="Camera connecting...",
            )
            self._clear_latest_frame()
            worker_thread.start()

        return True

    def stop(
        self,
        *,
        wait: bool = True,
        join_timeout: float = 5.0,
    ) -> bool:
        """Request shutdown and optionally join the capture worker.

        ``wait=False`` is used for interactive UI actions so a slow driver
        cannot freeze Tkinter. The worker owns and releases ``VideoCapture`` in
        its ``finally`` block after a bounded open/read returns.
        """
        if join_timeout < 0:
            raise ValueError("join_timeout cannot be negative.")

        with self._state_lock:
            worker_thread = self._thread
            stop_event = self._stop_event

            if worker_thread is None:
                self._thread = None
                self._stop_event = None
                self._capture = None
                self._source_kind = None
                self._set_status_locked(
                    state="stopped",
                    message="Camera stopped.",
                )
                self._clear_latest_frame()
                return True

            if stop_event is not None:
                stop_event.set()

            if worker_thread.is_alive():
                self._set_status_locked(
                    state="stopping",
                    message="Camera stopping...",
                )

        self._clear_latest_frame()

        if (
            wait
            and worker_thread is not threading.current_thread()
            and worker_thread.ident is not None
        ):
            worker_thread.join(timeout=join_timeout)

        worker_alive = worker_thread.is_alive()

        if wait and worker_alive:
            self._set_status(
                state="error",
                message="Camera worker did not stop cleanly.",
            )
        elif not worker_alive:
            with self._state_lock:
                if self._thread is worker_thread:
                    self._thread = None
                    self._stop_event = None
                    self._capture = None
                    self._source_kind = None
                self._set_status_locked(
                    state="stopped",
                    message="Camera stopped.",
                )

        return not worker_alive

    @staticmethod
    def _validate_source(
        source: int | str,
    ) -> Literal["webcam", "rtsp"]:
        if isinstance(source, bool):
            raise TypeError("Camera source must be a webcam index or RTSP URL.")

        if isinstance(source, int):
            if source < 0:
                raise ValueError("Webcam index cannot be negative.")
            return "webcam"

        if isinstance(source, str):
            if not source.lower().startswith("rtsp://"):
                raise ValueError("Network camera source must be an RTSP URL.")
            return "rtsp"

        raise TypeError("Camera source must be a webcam index or RTSP URL.")

    def _capture_loop(
        self,
        source: int | str,
        source_kind: Literal["webcam", "rtsp"],
        stop_event: threading.Event,
    ) -> None:
        capture = None

        try:
            capture = self._open_capture(
                source=source,
                source_kind=source_kind,
                stop_event=stop_event,
            )

            if capture is None or stop_event.is_set():
                if not stop_event.is_set():
                    self._set_status_while_running(
                        stop_event=stop_event,
                        state="error",
                        message=self._open_failure_message(source_kind),
                    )
                return

            if not self._set_status_while_running(
                stop_event=stop_event,
                state="connected",
                message="Camera connected.",
            ):
                return

            consecutive_failures = 0

            while not stop_event.is_set():
                try:
                    frame_ok, frame = capture.read()
                except Exception:
                    frame_ok, frame = False, None

                if stop_event.is_set():
                    break

                if not frame_ok or frame is None:
                    consecutive_failures += 1

                    if (
                        consecutive_failures
                        >= self.MAX_CONSECUTIVE_READ_FAILURES
                    ):
                        self._set_status_while_running(
                            stop_event=stop_event,
                            state="error",
                            message=self._stream_lost_message(source_kind),
                        )
                        break

                    stop_event.wait(self.READ_RETRY_SECONDS)
                    continue

                if consecutive_failures:
                    self._set_status_while_running(
                        stop_event=stop_event,
                        state="connected",
                        message="Camera connected.",
                    )
                    consecutive_failures = 0

                with self._frame_lock:
                    if not stop_event.is_set():
                        self._latest_frame = frame
                        self._frame_sequence += 1

        except Exception:
            if not stop_event.is_set():
                self._set_status_while_running(
                    stop_event=stop_event,
                    state="error",
                    message=self._open_failure_message(source_kind),
                )
        finally:
            self._release_capture(capture)
            self._clear_latest_frame()

            with self._state_lock:
                if self._capture is capture:
                    self._capture = None

                if stop_event.is_set():
                    self._set_status_locked(
                        state="stopped",
                        message="Camera stopped.",
                    )

    def _open_capture(
        self,
        *,
        source: int | str,
        source_kind: Literal["webcam", "rtsp"],
        stop_event: threading.Event,
    ) -> Any | None:
        if source_kind == "rtsp":
            capture = self._create_capture(
                source=source,
                source_kind=source_kind,
                backend=cv2.CAP_FFMPEG,
            )

            if not self._capture_is_open(capture):
                self._release_capture(capture)
                self._clear_capture_reference(capture)

                if stop_event.is_set():
                    return None

                capture = self._create_capture(
                    source=source,
                    source_kind=source_kind,
                )
        else:
            capture = self._create_capture(
                source=source,
                source_kind=source_kind,
            )

        if not self._capture_is_open(capture):
            self._release_capture(capture)
            self._clear_capture_reference(capture)
            return None

        if stop_event.is_set():
            self._release_capture(capture)
            self._clear_capture_reference(capture)
            return None

        try:
            capture.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        except Exception:
            pass

        return capture

    def _create_capture(
        self,
        *,
        source: int | str,
        source_kind: Literal["webcam", "rtsp"],
        backend: int | None = None,
    ) -> Any:
        # Create the handle before opening it so the worker remains its sole
        # owner for open/read/release. RTSP opens use OpenCV's backend timeout
        # parameters to keep stop and application shutdown bounded.
        capture = cv2.VideoCapture()

        with self._state_lock:
            self._capture = capture

        try:
            if source_kind == "rtsp":
                open_parameters = self._rtsp_open_parameters()
                api_preference = (
                    cv2.CAP_ANY
                    if backend is None
                    else backend
                )
                capture.open(
                    source,
                    api_preference,
                    open_parameters,
                )
            elif backend is None:
                capture.open(source)
            else:
                capture.open(source, backend)
        except Exception:
            pass

        return capture

    def _rtsp_open_parameters(self) -> list[int]:
        open_timeout_property = getattr(
            cv2,
            "CAP_PROP_OPEN_TIMEOUT_MSEC",
            None,
        )
        read_timeout_property = getattr(
            cv2,
            "CAP_PROP_READ_TIMEOUT_MSEC",
            None,
        )

        if open_timeout_property is None or read_timeout_property is None:
            raise RuntimeError(
                "This OpenCV build does not support bounded RTSP timeouts."
            )

        return [
            int(open_timeout_property),
            self.RTSP_OPEN_TIMEOUT_MS,
            int(read_timeout_property),
            self.RTSP_READ_TIMEOUT_MS,
        ]

    @staticmethod
    def _capture_is_open(capture: Any | None) -> bool:
        if capture is None:
            return False

        try:
            return bool(capture.isOpened())
        except Exception:
            return False

    @staticmethod
    def _release_capture(capture: Any | None) -> None:
        if capture is None:
            return

        try:
            capture.release()
        except Exception:
            pass

    def _clear_capture_reference(self, capture: Any) -> None:
        with self._state_lock:
            if self._capture is capture:
                self._capture = None

    def _clear_latest_frame(self) -> None:
        with self._frame_lock:
            self._latest_frame = None
            self._frame_sequence = 0

    def _set_status(
        self,
        *,
        state: CameraState,
        message: str,
    ) -> None:
        with self._state_lock:
            self._set_status_locked(
                state=state,
                message=message,
            )

    def _set_status_while_running(
        self,
        *,
        stop_event: threading.Event,
        state: CameraState,
        message: str,
    ) -> bool:
        """Publish status only if this capture has not been asked to stop."""
        with self._state_lock:
            if stop_event.is_set():
                return False

            self._set_status_locked(
                state=state,
                message=message,
            )
            return True

    def _set_status_locked(
        self,
        *,
        state: CameraState,
        message: str,
    ) -> None:
        self._status = CameraStatus(
            state=state,
            message=message,
            revision=self._status.revision + 1,
        )

    @staticmethod
    def _open_failure_message(
        source_kind: Literal["webcam", "rtsp"],
    ) -> str:
        return (
            "RTSP connection failed."
            if source_kind == "rtsp"
            else "Camera unavailable."
        )

    @staticmethod
    def _stream_lost_message(
        source_kind: Literal["webcam", "rtsp"],
    ) -> str:
        return (
            "RTSP stream connection was lost."
            if source_kind == "rtsp"
            else "Camera connection was lost."
        )
