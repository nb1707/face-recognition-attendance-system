"""Tkinter application shell.

This module contains navigation, main-thread callback dispatch, and the single
application shutdown path. It does not import the FastAPI app or initialize AI
models, cameras, or cloud clients on the Tkinter main thread.
"""

from __future__ import annotations

import logging
import queue
import tkinter as tk
from collections.abc import Callable
from typing import Any
from tkinter import ttk

from .camera import (
    CameraConfigurationError,
    CameraWorker,
    build_hikvision_rtsp_source,
)
from .detection import (
    DetectionResult,
    DetectionWorker,
    draw_face_detections,
)
from .recognition import (
    RecognitionResult,
    RecognitionWorker,
    draw_recognition_results,
)
from .views import DetectionView, HomeView, RecognitionView, RegisterView


LOGGER = logging.getLogger(__name__)

UiCallback = tuple[
    Callable[..., Any],
    tuple[Any, ...],
    dict[str, Any],
]


class FaceAttendanceDesktopApp(tk.Tk):
    """Single-root Tkinter shell for all desktop views."""

    APP_TITLE = "AI Face Recognition Attendance System"
    UI_QUEUE_POLL_MS = 50
    MAX_UI_CALLBACKS_PER_POLL = 50
    CAMERA_FRAME_POLL_MS = 30

    def __init__(self) -> None:
        super().__init__()

        self._is_shutting_down = False
        self._shutdown_handlers: list[Callable[[], None]] = []
        self._ui_callbacks: queue.Queue[UiCallback] = queue.Queue()
        self._ui_poll_after_id: str | None = None

        self._camera_worker = CameraWorker()
        self._detection_worker = DetectionWorker()
        self._recognition_worker = RecognitionWorker()
        self._camera_poll_after_id: str | None = None
        self._shutdown_after_id: str | None = None
        self._camera_owner_view: str | None = None
        self._camera_last_frame_sequence = -1
        self._camera_last_status_revision = -1
        self._detection_last_status_revision = -1
        self._detection_last_result_revision = -1
        self._latest_detection_result: DetectionResult | None = None
        self._recognition_last_status_revision = -1
        self._recognition_last_result_revision = -1
        self._latest_recognition_result: RecognitionResult | None = None
        self._recognition_start_pending = False

        self._views: dict[str, ttk.Frame] = {}
        self._navigation_buttons: dict[str, ttk.Button] = {}
        self._current_view_name = tk.StringVar(
            master=self,
            value="Home",
        )

        self._configure_window()
        self._build_layout()
        self._bind_camera_commands()
        self.show_view("home")
        self._schedule_ui_queue_poll()
        self._schedule_camera_poll()

    def _configure_window(self) -> None:
        self.title(self.APP_TITLE)
        self.geometry("1280x760")
        self.minsize(1024, 640)
        self.protocol("WM_DELETE_WINDOW", self.shutdown)

        self.columnconfigure(1, weight=1)
        self.rowconfigure(0, weight=1)

        style = ttk.Style(self)
        style.configure(
            "AppTitle.TLabel",
            font=("Segoe UI", 18, "bold"),
        )
        style.configure(
            "PageTitle.TLabel",
            font=("Segoe UI", 16, "bold"),
        )
        style.configure(
            "SectionTitle.TLabel",
            font=("Segoe UI", 11, "bold"),
        )
        style.configure(
            "Navigation.TButton",
            anchor="w",
            padding=(14, 10),
        )
        style.configure(
            "Selected.Navigation.TButton",
            anchor="w",
            padding=(14, 10),
            relief="sunken",
        )

    def _build_layout(self) -> None:
        navigation = ttk.Frame(self, padding=(18, 20))
        navigation.grid(row=0, column=0, sticky="nsew")
        navigation.columnconfigure(0, weight=1)
        navigation.rowconfigure(7, weight=1)

        ttk.Label(
            navigation,
            text="AI Attendance",
            style="AppTitle.TLabel",
        ).grid(row=0, column=0, sticky="w", pady=(0, 6))

        ttk.Label(
            navigation,
            text="Desktop Application",
        ).grid(row=1, column=0, sticky="w", pady=(0, 22))

        navigation_items = (
            ("home", "Home"),
            ("register", "Register Face"),
            ("detection", "Detect Face"),
            ("recognition", "Recognize Face"),
        )

        for row_index, (view_name, label) in enumerate(
            navigation_items,
            start=2,
        ):
            button = ttk.Button(
                navigation,
                text=label,
                style="Navigation.TButton",
                command=lambda name=view_name: self.show_view(name),
            )
            button.grid(
                row=row_index,
                column=0,
                sticky="ew",
                pady=3,
            )
            self._navigation_buttons[view_name] = button

        ttk.Separator(
            navigation,
            orient="horizontal",
        ).grid(row=6, column=0, sticky="ew", pady=(18, 12))

        ttk.Button(
            navigation,
            text="Exit",
            style="Navigation.TButton",
            command=self.shutdown,
        ).grid(row=8, column=0, sticky="ew", pady=(12, 0))

        main_panel = ttk.Frame(self, padding=(24, 18))
        main_panel.grid(row=0, column=1, sticky="nsew")
        main_panel.columnconfigure(0, weight=1)
        main_panel.rowconfigure(1, weight=1)

        header = ttk.Frame(main_panel)
        header.grid(row=0, column=0, sticky="ew", pady=(0, 16))
        header.columnconfigure(0, weight=1)

        ttk.Label(
            header,
            text=self.APP_TITLE,
            style="AppTitle.TLabel",
        ).grid(row=0, column=0, sticky="w")

        ttk.Label(
            header,
            textvariable=self._current_view_name,
        ).grid(row=0, column=1, sticky="e")

        content = ttk.Frame(main_panel)
        content.grid(row=1, column=0, sticky="nsew")
        content.columnconfigure(0, weight=1)
        content.rowconfigure(0, weight=1)

        view_types: tuple[tuple[str, str, type[ttk.Frame]], ...] = (
            ("home", "Home", HomeView),
            ("register", "Register Face", RegisterView),
            ("detection", "Detect Face", DetectionView),
            ("recognition", "Recognize Face", RecognitionView),
        )

        self._view_titles: dict[str, str] = {}

        for view_name, title, view_type in view_types:
            view = view_type(content)
            view.grid(row=0, column=0, sticky="nsew")
            self._views[view_name] = view
            self._view_titles[view_name] = title

    def show_view(self, view_name: str) -> None:
        """Raise an existing view inside the shared content panel."""
        if view_name not in self._views:
            raise ValueError(f"Unknown desktop view: {view_name}")

        if (
            self._camera_owner_view is not None
            and self._camera_owner_view != view_name
        ):
            self._stop_camera_preview()

        self._views[view_name].tkraise()
        self._current_view_name.set(self._view_titles[view_name])

        for name, button in self._navigation_buttons.items():
            button.configure(
                style=(
                    "Selected.Navigation.TButton"
                    if name == view_name
                    else "Navigation.TButton"
                )
            )

        self._views[view_name].event_generate("<<ViewShown>>")

    def _bind_camera_commands(self) -> None:
        register_view = self._views["register"]
        detection_view = self._views["detection"]
        recognition_view = self._views["recognition"]

        if not isinstance(register_view, RegisterView):
            raise TypeError("Register view is not configured correctly.")
        if not isinstance(detection_view, DetectionView):
            raise TypeError("Detection view is not configured correctly.")
        if not isinstance(recognition_view, RecognitionView):
            raise TypeError("Recognition view is not configured correctly.")

        register_view.set_camera_commands(
            start_command=self._start_registration_camera,
            stop_command=self._stop_camera_preview,
        )
        detection_view.set_camera_commands(
            start_command=self._start_detection_camera,
            stop_command=self._stop_camera_preview,
        )
        recognition_view.set_recognition_commands(
            start_command=self._start_recognition,
            stop_command=self._stop_camera_preview,
        )

    def _start_registration_camera(self) -> None:
        self._start_camera_preview(
            owner_view="register",
            source=0,
        )

    def _start_detection_camera(self) -> None:
        detection_view = self._views["detection"]
        if not isinstance(detection_view, DetectionView):
            return

        if self._detection_worker.is_active:
            detection_view.set_camera_state(
                "Face detector is already running or still stopping.",
                active=False,
            )
            return

        detection_view.set_camera_state(
            "Camera connecting...",
            active=True,
        )

        try:
            source = build_hikvision_rtsp_source()
        except CameraConfigurationError as error:
            detection_view.set_camera_state(
                str(error),
                active=False,
            )
            return
        except Exception:
            detection_view.set_camera_state(
                "Hikvision camera configuration could not be loaded.",
                active=False,
            )
            return

        camera_started = self._start_camera_preview(
            owner_view="detection",
            source=source,
        )

        if not camera_started:
            return

        self._reset_detection_state(detection_view)
        detection_view.status.set("Loading face detector...")

        if not self._detection_worker.start():
            detection_view.set_camera_state(
                self._detection_worker.status.message,
                active=False,
            )
            self._camera_worker.stop(wait=False)

    def _start_recognition(self) -> None:
        recognition_view = self._views["recognition"]
        if not isinstance(recognition_view, RecognitionView):
            return

        if self._recognition_worker.is_active:
            recognition_view.set_camera_state(
                "Recognition is already running or still stopping.",
                active=False,
            )
            return

        recognition_view.active_model.set("Active Model: Loading AdaFace...")
        recognition_view.recognition_status.set(
            "Recognition Status: Loading"
        )
        recognition_view.set_camera_state(
            "Camera connecting...",
            active=True,
        )

        try:
            source = build_hikvision_rtsp_source()
        except CameraConfigurationError as error:
            recognition_view.set_camera_state(
                str(error),
                active=False,
            )
            recognition_view.recognition_status.set(
                "Recognition Status: Error"
            )
            return
        except Exception:
            recognition_view.set_camera_state(
                "Hikvision camera configuration could not be loaded.",
                active=False,
            )
            recognition_view.recognition_status.set(
                "Recognition Status: Error"
            )
            return

        camera_started = self._start_camera_preview(
            owner_view="recognition",
            source=source,
        )

        if not camera_started:
            recognition_view.recognition_status.set(
                "Recognition Status: Error"
            )
            return

        self._reset_recognition_state(recognition_view)
        recognition_view.active_model.set("Active Model: Loading AdaFace...")
        recognition_view.recognition_status.set(
            "Recognition Status: Loading"
        )
        recognition_view.status.set("Loading face detector...")
        self._recognition_start_pending = True

        if not self._detection_worker.start():
            self._recognition_start_pending = False
            recognition_view.set_camera_state(
                self._detection_worker.status.message,
                active=False,
            )
            recognition_view.recognition_status.set(
                "Recognition Status: Error"
            )
            self._camera_worker.stop(wait=False)

    def _start_camera_preview(
        self,
        *,
        owner_view: str,
        source: int | str,
    ) -> bool:
        if self._is_shutting_down:
            return False

        target_view = self._views[owner_view]

        if self._camera_worker.is_active:
            self._set_view_camera_state(
                target_view,
                "Another camera is active or still stopping.",
                active=False,
            )
            return False

        if self._detection_worker.is_active:
            self._set_view_camera_state(
                target_view,
                "Face detector is still stopping.",
                active=False,
            )
            return False

        if self._recognition_worker.is_active:
            self._set_view_camera_state(
                target_view,
                "Recognition is still stopping.",
                active=False,
            )
            return False

        self._recognition_worker.stop(
            wait=True,
            join_timeout=0.0,
        )

        self._detection_worker.stop(
            wait=True,
            join_timeout=0.0,
        )

        previous_status = self._camera_worker.status
        self._camera_worker.stop(
            wait=True,
            join_timeout=0.0,
        )
        self._finalize_camera_owner(
            final_message=(
                previous_status.message
                if previous_status.state == "error"
                else None
            )
        )

        if not target_view.video_panel.pillow_available:
            self._set_view_camera_state(
                target_view,
                "Pillow is unavailable. Camera frames cannot be displayed.",
                active=False,
            )
            return False

        self._camera_owner_view = owner_view
        self._camera_last_frame_sequence = -1
        self._camera_last_status_revision = -1

        target_view.video_panel.clear("Camera connecting...")
        self._set_view_camera_state(
            target_view,
            "Camera connecting...",
            active=True,
        )

        try:
            started = self._camera_worker.start(source)
        except (TypeError, ValueError):
            started = False

        if not started:
            status = self._camera_worker.status
            self._set_view_camera_state(
                target_view,
                status.message,
                active=False,
            )
            self._camera_owner_view = None

        return started

    def _stop_camera_preview(self) -> None:
        owner_view_name = self._camera_owner_view

        if owner_view_name is None:
            return

        owner_view = self._views[owner_view_name]
        detection_owner = isinstance(owner_view, DetectionView)
        recognition_owner = isinstance(owner_view, RecognitionView)

        if recognition_owner:
            self._recognition_start_pending = False
            self._recognition_worker.stop(wait=False)
            self._detection_worker.stop(wait=False)
            self._reset_recognition_state(owner_view)
            owner_view.recognition_status.set("Recognition Status: Stopping")
        elif detection_owner:
            self._detection_worker.stop(wait=False)
            self._reset_detection_state(owner_view)

        self._set_view_camera_state(
            owner_view,
            (
                "Stopping recognition, face detector, and camera..."
                if recognition_owner
                else "Stopping face detector and camera..."
                if detection_owner
                else "Camera stopping..."
            ),
            active=False,
            stopping=True,
        )
        owner_view.video_panel.clear("Camera stopped")

        stopped = self._camera_worker.stop(wait=False)

        if (
            stopped
            and not self._detection_worker.is_active
            and not self._recognition_worker.is_active
        ):
            self._set_view_camera_state(
                owner_view,
                "Camera stopped.",
                active=False,
            )
            self._camera_owner_view = None

    def _finalize_camera_owner(
        self,
        final_message: str | None = None,
    ) -> None:
        """Restore the previous view after its camera worker has stopped."""
        owner_view_name = self._camera_owner_view

        if owner_view_name is None:
            return

        owner_view = self._views[owner_view_name]

        if isinstance(owner_view, RecognitionView):
            self._recognition_start_pending = False
            self._recognition_worker.stop(
                wait=True,
                join_timeout=0.0,
            )
            self._detection_worker.stop(
                wait=True,
                join_timeout=0.0,
            )
            self._reset_recognition_state(owner_view)
            owner_view.recognition_status.set("Recognition Status: Stopped")
        elif isinstance(owner_view, DetectionView):
            self._detection_worker.stop(
                wait=True,
                join_timeout=0.0,
            )
            self._reset_detection_state(owner_view)

        worker_status = self._camera_worker.status
        resolved_message = (
            final_message
            or (
                worker_status.message
                if worker_status.state == "error"
                else "Camera stopped."
            )
        )
        self._set_view_camera_state(
            owner_view,
            resolved_message,
            active=False,
        )
        owner_view.video_panel.clear(resolved_message)

        self._camera_owner_view = None

    @staticmethod
    def _set_view_camera_state(
        view: ttk.Frame,
        message: str,
        *,
        active: bool,
        stopping: bool = False,
    ) -> None:
        if isinstance(view, (RegisterView, DetectionView, RecognitionView)):
            view.set_camera_state(
                message,
                active=active,
                stopping=stopping,
            )

    def _schedule_camera_poll(self) -> None:
        if self._is_shutting_down:
            return

        self._camera_poll_after_id = self.after(
            self.CAMERA_FRAME_POLL_MS,
            self._poll_camera_preview,
        )

    def _poll_camera_preview(self) -> None:
        self._camera_poll_after_id = None

        if self._is_shutting_down:
            return

        owner_view_name = self._camera_owner_view

        if owner_view_name is not None:
            owner_view = self._views[owner_view_name]
            status = self._camera_worker.status

            if status.revision != self._camera_last_status_revision:
                self._camera_last_status_revision = status.revision
                active = status.state in {"connecting", "connected"}
                stopping = status.state == "stopping"
                self._set_view_camera_state(
                    owner_view,
                    status.message,
                    active=active,
                    stopping=stopping,
                )

                if status.state == "error":
                    owner_view.video_panel.clear(status.message)
                    if isinstance(owner_view, RecognitionView):
                        self._recognition_start_pending = False
                        self._recognition_worker.stop(wait=False)
                        self._detection_worker.stop(wait=False)
                        self._latest_recognition_result = None
                        self._latest_detection_result = None
                        owner_view.recognition_status.set(
                            "Recognition Status: Error"
                        )
                    elif isinstance(owner_view, DetectionView):
                        self._detection_worker.stop(wait=False)
                        self._latest_detection_result = None
                        self._reset_detection_metrics(owner_view)

            if isinstance(owner_view, DetectionView):
                self._poll_detection_updates(
                    owner_view,
                    camera_state=status.state,
                )
            elif isinstance(owner_view, RecognitionView):
                self._poll_recognition_updates(
                    owner_view,
                    camera_state=status.state,
                )

            frame = self._camera_worker.latest_frame(
                after_sequence=self._camera_last_frame_sequence
            )

            if frame is not None:
                self._camera_last_frame_sequence = frame.sequence
                display_frame = frame.image

                if isinstance(owner_view, DetectionView):
                    self._detection_worker.submit_frame(
                        frame=frame.image,
                        frame_sequence=frame.sequence,
                    )

                    if self._latest_detection_result is not None:
                        try:
                            display_frame = draw_face_detections(
                                frame.image,
                                self._latest_detection_result.faces,
                            )
                        except Exception:
                            owner_view.status.set("Detection drawing error.")
                            self._latest_detection_result = None
                            self._reset_detection_metrics(owner_view)
                elif isinstance(owner_view, RecognitionView):
                    self._detection_worker.submit_frame(
                        frame=frame.image,
                        frame_sequence=frame.sequence,
                    )

                    if self._latest_recognition_result is not None:
                        try:
                            display_frame = draw_recognition_results(
                                frame.image,
                                self._latest_recognition_result.faces,
                            )
                        except Exception:
                            owner_view.status.set(
                                "Recognition drawing error."
                            )
                            self._latest_recognition_result = None
                    elif self._latest_detection_result is not None:
                        try:
                            display_frame = draw_face_detections(
                                frame.image,
                                self._latest_detection_result.faces,
                            )
                        except Exception:
                            owner_view.status.set("Detection drawing error.")
                            self._latest_detection_result = None

                try:
                    owner_view.video_panel.show_bgr_frame(display_frame)
                except Exception:
                    self._set_view_camera_state(
                        owner_view,
                        "Camera frame could not be displayed.",
                        active=False,
                    )
                    if isinstance(owner_view, RecognitionView):
                        self._recognition_worker.stop(wait=False)
                        self._detection_worker.stop(wait=False)
                    elif isinstance(owner_view, DetectionView):
                        self._detection_worker.stop(wait=False)
                    self._camera_worker.stop(wait=False)

            if (
                status.state in {"stopped", "error"}
                and not self._camera_worker.is_active
                and (
                    not isinstance(owner_view, (DetectionView, RecognitionView))
                    or not self._detection_worker.is_active
                )
                and (
                    not isinstance(owner_view, RecognitionView)
                    or not self._recognition_worker.is_active
                )
            ):
                self._camera_worker.stop(
                    wait=True,
                    join_timeout=0.0,
                )
                if status.state == "stopped":
                    self._finalize_camera_owner()
                else:
                    self._camera_owner_view = None

        self._schedule_camera_poll()

    def _poll_detection_updates(
        self,
        detection_view: DetectionView,
        *,
        camera_state: str,
    ) -> None:
        detection_status = self._detection_worker.status

        if (
            camera_state not in {"error", "stopping", "stopped"}
            and detection_status.state not in {"idle", "stopped"}
        ):
            detection_view.status.set(detection_status.message)

        if (
            detection_status.revision
            != self._detection_last_status_revision
        ):
            self._detection_last_status_revision = detection_status.revision

            if detection_status.state == "error":
                self._latest_detection_result = None
                self._reset_detection_metrics(detection_view)

        result = self._detection_worker.latest_result(
            after_revision=self._detection_last_result_revision
        )

        if result is None:
            return

        self._detection_last_result_revision = result.revision
        self._latest_detection_result = result
        detection_view.detected_faces.set(
            f"Detected Faces: {len(result.faces)}"
        )
        detection_view.fps.set(
            f"FPS: {result.detection_fps:.2f}"
        )
        detection_view.processing_time.set(
            f"Processing Time: {result.processing_time_ms:.2f} ms"
        )

    def _poll_recognition_updates(
        self,
        recognition_view: RecognitionView,
        *,
        camera_state: str,
    ) -> None:
        detection_status = self._detection_worker.status

        if detection_status.state == "error":
            self._recognition_start_pending = False
            self._recognition_worker.stop(wait=False)
            recognition_view.status.set(detection_status.message)
            recognition_view.recognition_status.set(
                "Recognition Status: Error"
            )
        elif (
            self._recognition_start_pending
            and detection_status.state in {"ready", "detecting"}
        ):
            self._recognition_start_pending = False
            if self._recognition_worker.start():
                recognition_view.status.set("Loading AdaFace...")
                recognition_view.recognition_status.set(
                    "Recognition Status: Loading"
                )
            else:
                recognition_view.status.set(
                    "AdaFace recognition could not be started."
                )
                recognition_view.recognition_status.set(
                    "Recognition Status: Error"
                )
        elif (
            self._recognition_start_pending
            and camera_state not in {"error", "stopping", "stopped"}
        ):
            recognition_view.status.set(detection_status.message)

        detection_result = self._detection_worker.latest_result(
            after_revision=self._detection_last_result_revision
        )

        if detection_result is not None:
            self._detection_last_result_revision = detection_result.revision
            self._latest_detection_result = detection_result
            self._recognition_worker.submit_detection(detection_result)

        recognition_status = self._recognition_worker.status

        if (
            recognition_status.revision
            != self._recognition_last_status_revision
        ):
            self._recognition_last_status_revision = recognition_status.revision

            if recognition_status.state not in {"idle", "stopped"}:
                recognition_view.status.set(recognition_status.message)

            if recognition_status.state in {
                "loading_model",
                "loading_cache",
            }:
                recognition_view.active_model.set(
                    "Active Model: AdaFace"
                    if self._recognition_worker.model_class_name
                    else "Active Model: Loading AdaFace..."
                )
                recognition_view.recognition_status.set(
                    "Recognition Status: Loading"
                )
            elif recognition_status.state == "recognizing":
                recognition_view.active_model.set("Active Model: AdaFace")
                recognition_view.recognition_status.set(
                    "Recognition Status: Recognizing"
                )
            elif recognition_status.state == "ready":
                recognition_view.active_model.set("Active Model: AdaFace")
            elif recognition_status.state == "error":
                recognition_view.active_model.set(
                    "Active Model: AdaFace"
                    if self._recognition_worker.model_class_name
                    else "Active Model: Not Loaded"
                )
                recognition_view.recognition_status.set(
                    "Recognition Status: Error"
                )
                self._latest_recognition_result = None

        if recognition_status.state == "error":
            # Keep the recognition failure visible even if a later camera
            # status revision reports that the RTSP stream is connected.
            recognition_view.status.set(recognition_status.message)

        result = self._recognition_worker.latest_result(
            after_revision=self._recognition_last_result_revision
        )

        if result is None:
            return

        self._recognition_last_result_revision = result.revision
        self._latest_recognition_result = result
        recognition_view.processing_time.set(
            f"Processing Time: {result.processing_time_ms:.2f} ms"
        )
        self._update_recognition_details(recognition_view, result)

    @staticmethod
    def _update_recognition_details(
        recognition_view: RecognitionView,
        result: RecognitionResult,
    ) -> None:
        if not result.faces:
            recognition_view.identity.set("Identity: --")
            recognition_view.user_id.set("User ID: --")
            recognition_view.role.set("Role: --")
            recognition_view.similarity.set("Similarity / Confidence: --")
            recognition_view.recognition_status.set(
                "Recognition Status: No faces"
            )
            return

        best_face = max(
            result.faces,
            key=lambda face: face.similarity,
        )
        recognized_count = sum(
            1 for face in result.faces if face.recognized
        )

        if best_face.recognized:
            recognition_view.identity.set(
                f"Identity: {best_face.full_name}"
            )
            recognition_view.user_id.set(
                f"User ID: {best_face.user_id}"
            )
            recognition_view.role.set(f"Role: {best_face.role or '--'}")
            recognition_view.recognition_status.set(
                "Recognition Status: "
                f"Recognized ({recognized_count}/{len(result.faces)})"
            )
        else:
            recognition_view.identity.set("Identity: Unknown")
            recognition_view.user_id.set("User ID: --")
            recognition_view.role.set("Role: --")
            failure_labels = {
                "alignment_failure": "Alignment failed",
                "model_inference_error": "Model/inference failed",
                "no_compatible_embeddings": "No compatible embeddings",
                "low_similarity": "Below threshold",
            }
            reason = failure_labels.get(
                best_face.failure_reason,
                "Unknown",
            )
            recognition_view.recognition_status.set(
                f"Recognition Status: Unknown ({reason})"
            )

        if best_face.similarity > -1.0:
            recognition_view.similarity.set(
                "Similarity / Confidence: "
                f"{best_face.similarity:.4f} "
                f"(threshold {result.threshold:.2f})"
            )
        else:
            recognition_view.similarity.set(
                "Similarity / Confidence: --"
            )

    def _reset_detection_state(
        self,
        detection_view: DetectionView,
    ) -> None:
        self._detection_last_status_revision = -1
        self._detection_last_result_revision = -1
        self._latest_detection_result = None
        self._reset_detection_metrics(detection_view)

    def _reset_recognition_state(
        self,
        recognition_view: RecognitionView,
    ) -> None:
        self._detection_last_status_revision = -1
        self._detection_last_result_revision = -1
        self._recognition_last_status_revision = -1
        self._recognition_last_result_revision = -1
        self._latest_detection_result = None
        self._latest_recognition_result = None
        recognition_view.active_model.set("Active Model: Not Loaded")
        recognition_view.reset_recognition_details()

    @staticmethod
    def _reset_detection_metrics(
        detection_view: DetectionView,
    ) -> None:
        detection_view.detected_faces.set("Detected Faces: 0")
        detection_view.fps.set("FPS: 0.0")
        detection_view.processing_time.set("Processing Time: -- ms")

    def post_to_ui(
        self,
        callback: Callable[..., Any],
        *args: Any,
        **kwargs: Any,
    ) -> None:
        """Queue a callback for execution by the Tkinter event loop.

        Future worker threads should use this method instead of calling Tkinter
        widgets or ``after`` directly. High-rate camera frames should eventually
        use a separate bounded latest-frame buffer rather than this general queue.
        """
        if self._is_shutting_down:
            return

        self._ui_callbacks.put((callback, args, kwargs))

    def add_shutdown_handler(
        self,
        handler: Callable[[], None],
    ) -> None:
        """Register cleanup to run before the root window is destroyed."""
        if handler not in self._shutdown_handlers:
            self._shutdown_handlers.append(handler)

    def _schedule_ui_queue_poll(self) -> None:
        if self._is_shutting_down:
            return

        self._ui_poll_after_id = self.after(
            self.UI_QUEUE_POLL_MS,
            self._drain_ui_queue,
        )

    def _drain_ui_queue(self) -> None:
        self._ui_poll_after_id = None

        if self._is_shutting_down:
            return

        for _ in range(self.MAX_UI_CALLBACKS_PER_POLL):
            try:
                callback, args, kwargs = self._ui_callbacks.get_nowait()
            except queue.Empty:
                break

            try:
                callback(*args, **kwargs)
            except Exception:
                LOGGER.exception("Desktop UI callback failed")

        self._schedule_ui_queue_poll()

    def shutdown(self) -> None:
        """Request worker cleanup before destroying the one Tk root."""
        if self._is_shutting_down:
            return

        self._is_shutting_down = True

        # Hide the window immediately while the event loop remains alive long
        # enough for the non-daemon camera worker to release and exit.
        self.withdraw()

        if self._ui_poll_after_id is not None:
            try:
                self.after_cancel(self._ui_poll_after_id)
            except tk.TclError:
                pass
            self._ui_poll_after_id = None

        if self._camera_poll_after_id is not None:
            try:
                self.after_cancel(self._camera_poll_after_id)
            except tk.TclError:
                pass
            self._camera_poll_after_id = None

        self._recognition_start_pending = False
        self._recognition_worker.stop(wait=False)
        self._detection_worker.stop(wait=False)
        self._camera_worker.stop(wait=False)
        self._finish_shutdown_when_workers_stop()

    def _finish_shutdown_when_workers_stop(self) -> None:
        """Keep Tk alive until detection and capture workers have exited."""
        self._shutdown_after_id = None

        if (
            self._recognition_worker.is_active
            or self._detection_worker.is_active
            or self._camera_worker.is_active
        ):
            self._shutdown_after_id = self.after(
                50,
                self._finish_shutdown_when_workers_stop,
            )
            return

        # Reap completed threads and clear worker-owned state.
        self._recognition_worker.stop(
            wait=True,
            join_timeout=0.5,
        )
        self._detection_worker.stop(
            wait=True,
            join_timeout=0.5,
        )
        self._camera_worker.stop(
            wait=True,
            join_timeout=0.5,
        )

        for handler in reversed(self._shutdown_handlers):
            try:
                handler()
            except Exception:
                LOGGER.exception("Desktop shutdown handler failed")

        self.destroy()


def main() -> int:
    """Launch the desktop application."""
    application = FaceAttendanceDesktopApp()
    application.mainloop()
    return 0


__all__ = ["FaceAttendanceDesktopApp", "main"]
