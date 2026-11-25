"""Tkinter application shell.

This module contains navigation, main-thread callback dispatch, and the single
application shutdown path. It does not import the FastAPI app or initialize AI
models, cameras, or cloud clients on the Tkinter main thread.
"""

from __future__ import annotations

import logging
import queue
import time
import tkinter as tk
from collections.abc import Callable
from concurrent.futures import Future, ThreadPoolExecutor
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
from .logging_config import configure_desktop_logging
from .recognition import (
    RecognitionResult,
    RecognitionWorker,
    draw_recognition_results,
)
from .session import (
    CAMERA_TEST_SESSION,
    RecognitionSession,
    RecognitionSessionInput,
    RecognitionSessionService,
    RecognitionTrialInput,
    RecognitionTrialSnapshot,
    load_recognition_configuration_snapshot,
)
from .views import DetectionView, HomeView, RecognitionView, RegisterView
from .session.local_repository import LocalRecognitionSessionRepository
from .camera.recognition_source import build_recognition_camera_source
from .attendance import AttendanceApiClient, AttendanceClientError
from .recognition.cache import load_adaface_recognition_cache


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
        self._session_service = RecognitionSessionService(repository=LocalRecognitionSessionRepository())
        self._attendance_client = AttendanceApiClient()
        self._attendance_executor = ThreadPoolExecutor(
            max_workers=1, thread_name_prefix="desktop-attendance-api"
        )
        self._attendance_session_id: str | None = None
        self._attendance_session: dict[str, Any] | None = None
        self._attendance_api_pending = False
        self._attendance_poll_pending = False
        self._attendance_next_poll_at = 0.0
        self._attendance_missing_gallery_reported = False
        self._attendance_end_requested = False
        self._attendance_executor_closed = False
        self._shutdown_attendance_close_attempted = False
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
        self._recognition_session_create_pending = False
        self._recognition_session_cancel_requested = False
        self._session_camera_metadata_saved = False
        self._session_close_requested = False
        self._shutdown_session_close_attempts = 0
        self._expected_users_loaded = False
        self._expected_users_load_pending = False
        self._visible_view_name = "home"

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
        LOGGER.info("Application started")

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

        recognition_view = self._views.get("recognition")
        if (
            self._visible_view_name == "recognition"
            and view_name != "recognition"
            and isinstance(recognition_view, RecognitionView)
            and recognition_view.attendance_phase == "active"
        ):
            self._end_attendance_session()

        if (
            self._camera_owner_view is not None
            and self._camera_owner_view != view_name
        ):
            self._stop_camera_preview()
        elif (
            self._visible_view_name == "recognition"
            and view_name != "recognition"
            and self._recognition_session_create_pending
        ):
            self._cancel_pending_recognition_start()

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
        self._visible_view_name = view_name
        if view_name == "recognition":
            self._load_attendance_classes()

    def _load_expected_users(self) -> None:
        if (
            self._is_shutting_down
            or self._expected_users_loaded
            or self._expected_users_load_pending
        ):
            return

        recognition_view = self._views.get("recognition")
        if not isinstance(recognition_view, RecognitionView):
            return

        self._expected_users_load_pending = True
        if self._session_service.active_session is None:
            recognition_view.status.set("Loading registered users...")
        started = self._session_service.load_expected_users_async(
            on_success=lambda users: self.post_to_ui(
                self._on_expected_users_loaded, users
            ),
            on_error=lambda message: self.post_to_ui(
                self._on_expected_users_load_failed, message
            ),
        )
        if not started:
            self._expected_users_load_pending = False

    def _on_expected_users_loaded(
        self,
        users: tuple[tuple[str, str], ...],
    ) -> None:
        self._expected_users_load_pending = False
        self._expected_users_loaded = True
        recognition_view = self._views.get("recognition")
        if isinstance(recognition_view, RecognitionView):
            recognition_view.set_expected_users(users)
            if (
                self._session_service.active_session is None
                and not self._recognition_session_create_pending
            ):
                recognition_view.status.set(
                    f"Loaded {len(users)} registered users."
                )

    def _on_expected_users_load_failed(self, message: str) -> None:
        self._expected_users_load_pending = False
        recognition_view = self._views.get("recognition")
        if isinstance(recognition_view, RecognitionView):
            recognition_view.set_expected_users(())
            if self._session_service.active_session is None:
                recognition_view.status.set(message)

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
            stop_command=self._end_attendance_session,
            record_trial_command=self._record_recognition_trial,
        )
        recognition_view.set_attendance_commands(
            class_selected_command=self._load_selected_class_roster,
            review_command=self._review_selected_student,
            finalize_command=self._finalize_attendance_session,
        )

    def _start_registration_camera(self) -> None:
        LOGGER.info("Local registration camera start requested")
        self._start_camera_preview(
            owner_view="register",
            source=0,
        )

    def _start_detection_camera(self) -> None:
        LOGGER.info("Hikvision detection camera start requested")
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
        else:
            LOGGER.info("Detection worker started")

    def _start_recognition(self) -> None:
        recognition_view = self._views["recognition"]
        if not isinstance(recognition_view, RecognitionView):
            return

        values = recognition_view.attendance_start_values()
        if not values["class_id"] or not values["operator_name"]:
            recognition_view.status.set("Select a class and enter an operator name.")
            return
        if self._attendance_session_id is not None or self._attendance_api_pending:
            recognition_view.status.set("An attendance operation is already active.")
            return

        camera_key = values["camera_source"].strip().lower()
        recognition_view.set_attendance_session("starting")

        def operation() -> dict[str, Any]:
            configuration = load_recognition_configuration_snapshot()
            cache = load_adaface_recognition_cache()
            session = self._attendance_client.create_session(
                class_id=values["class_id"],
                gallery_generation=cache.gallery.generation,
                recognition_config=(
                    configuration.recognition_settings
                    or configuration.to_snapshot()
                ),
                camera_source=camera_key,
                operator_name=values["operator_name"],
            )
            if self._is_shutting_down:
                self._attendance_client.end_session(str(session["session_id"]))
            return session

        self._run_attendance_operation(
            operation,
            self._on_attendance_session_created,
            busy_message="Creating attendance session...",
            on_error=lambda message: recognition_view.set_attendance_session("idle"),
        )

    def _load_attendance_classes(self) -> None:
        recognition_view = self._views.get("recognition")
        if not isinstance(recognition_view, RecognitionView):
            return
        if recognition_view.backend_available or self._attendance_api_pending:
            return
        self._run_attendance_operation(
            self._attendance_client.list_classes,
            self._on_attendance_classes_loaded,
            busy_message="Loading attendance classes...",
            on_error=recognition_view.set_backend_unavailable,
        )

    def _on_attendance_classes_loaded(self, classes: list[dict[str, Any]]) -> None:
        recognition_view = self._views.get("recognition")
        if not isinstance(recognition_view, RecognitionView):
            return
        recognition_view.set_classes(classes)
        recognition_view.status.set(
            "Select a class to view its roster."
            if classes
            else "No active attendance classes are available."
        )

    def _load_selected_class_roster(self) -> None:
        recognition_view = self._views.get("recognition")
        if not isinstance(recognition_view, RecognitionView):
            return
        class_id = recognition_view.selected_class_id()
        if not class_id:
            recognition_view.set_roster([])
            return
        self._run_attendance_operation(
            lambda: self._attendance_client.get_class_roster(class_id),
            self._on_attendance_roster_loaded,
            busy_message="Loading class roster...",
        )

    def _on_attendance_roster_loaded(self, roster: list[dict[str, Any]]) -> None:
        recognition_view = self._views.get("recognition")
        if not isinstance(recognition_view, RecognitionView):
            return
        recognition_view.set_roster(roster)
        recognition_view.status.set(
            f"Roster loaded: {len(roster)} students."
            if roster
            else "The selected class has no active roster members."
        )

    def _on_attendance_session_created(self, session: dict[str, Any]) -> None:
        recognition_view = self._views.get("recognition")
        if not isinstance(recognition_view, RecognitionView):
            return
        self._attendance_session_id = str(session["session_id"])
        self._attendance_session = dict(session)
        self._attendance_next_poll_at = 0.0
        self._attendance_missing_gallery_reported = False
        self._attendance_end_requested = False
        self._session_camera_metadata_saved = False
        recognition_view.set_attendance_session("active", session)
        if self._is_shutting_down or self._visible_view_name != "recognition":
            self._end_attendance_session()
            return
        recognition_view.status.set("Session created. Camera connecting...")
        self._start_recognition_pipeline(recognition_view)

    def _run_attendance_operation(
        self,
        operation: Callable[[], Any],
        on_success: Callable[[Any], None],
        *,
        busy_message: str,
        on_error: Callable[[str], None] | None = None,
    ) -> None:
        recognition_view = self._views.get("recognition")
        if self._attendance_api_pending:
            if isinstance(recognition_view, RecognitionView):
                recognition_view.status.set("Another attendance request is in progress.")
            return
        self._attendance_api_pending = True
        if isinstance(recognition_view, RecognitionView):
            recognition_view.set_attendance_busy(True, busy_message)
        future = self._attendance_executor.submit(operation)

        def completed(result_future: Future) -> None:
            try:
                result = result_future.result()
            except AttendanceClientError as error:
                message = str(error)
                self.post_to_ui(self._complete_attendance_failure, message, on_error)
            except Exception:
                LOGGER.exception("Unexpected desktop attendance operation failure")
                self.post_to_ui(
                    self._complete_attendance_failure,
                    "Attendance operation failed.",
                    on_error,
                )
            else:
                self.post_to_ui(self._complete_attendance_success, on_success, result)

        future.add_done_callback(completed)

    def _complete_attendance_success(
        self, callback: Callable[[Any], None], result: Any
    ) -> None:
        self._attendance_api_pending = False
        recognition_view = self._views.get("recognition")
        if isinstance(recognition_view, RecognitionView):
            recognition_view.set_attendance_busy(False)
        callback(result)

    def _complete_attendance_failure(
        self,
        message: str,
        callback: Callable[[str], None] | None,
    ) -> None:
        self._attendance_api_pending = False
        recognition_view = self._views.get("recognition")
        if isinstance(recognition_view, RecognitionView):
            recognition_view.set_attendance_busy(False)
            recognition_view.status.set(message)
        if callback is not None:
            callback(message)

    def _on_recognition_session_created(
        self,
        session: RecognitionSession,
    ) -> None:
        self._recognition_session_create_pending = False
        recognition_view = self._views["recognition"]
        if not isinstance(recognition_view, RecognitionView):
            return

        if (
            self._is_shutting_down
            or self._recognition_session_cancel_requested
            or self._visible_view_name != "recognition"
        ):
            self._request_recognition_session_end(recognition_view)
            return

        recognition_view.set_session_active(
            active=True,
            session_type=session.context.session_type,
            trial_count=0,
        )
        recognition_view.status.set("Session created. Camera connecting...")
        self._start_recognition_pipeline(recognition_view)

    def _on_recognition_session_create_failed(self, message: str) -> None:
        self._recognition_session_create_pending = False
        self._recognition_session_cancel_requested = False
        recognition_view = self._views["recognition"]
        if isinstance(recognition_view, RecognitionView):
            recognition_view.set_session_pending(False)
            recognition_view.set_camera_state(message, active=False)
            recognition_view.recognition_status.set(
                "Recognition Status: Error"
            )

    def _start_recognition_pipeline(
        self,
        recognition_view: RecognitionView,
    ) -> None:
        try:
            source = build_recognition_camera_source(
                recognition_view.camera_source.get()
            )
        except CameraConfigurationError as error:
            recognition_view.set_camera_state(str(error), active=False)
            recognition_view.recognition_status.set(
                "Recognition Status: Error"
            )
            self._end_attendance_session_after_failure(str(error))
            return
        except Exception:
            recognition_view.set_camera_state(
                "Recognition camera configuration could not be loaded.",
                active=False,
            )
            recognition_view.recognition_status.set(
                "Recognition Status: Error"
            )
            self._end_attendance_session_after_failure(
                "Recognition camera configuration could not be loaded."
            )
            return

        camera_started = self._start_camera_preview(
            owner_view="recognition", source=source
        )

        if not camera_started:
            recognition_view.recognition_status.set(
                "Recognition Status: Error"
            )
            self._end_attendance_session_after_failure(
                "Recognition camera could not be started."
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
            self._end_attendance_session_after_failure(
                self._detection_worker.status.message
            )
        else:
            LOGGER.info("Detection worker started for recognition")

    def _cancel_pending_recognition_start(self) -> None:
        self._recognition_session_cancel_requested = True
        recognition_view = self._views.get("recognition")
        if isinstance(recognition_view, RecognitionView):
            recognition_view.status.set("Cancelling recognition session start...")

    def _end_attendance_session_after_failure(self, message: str) -> None:
        recognition_view = self._views.get("recognition")
        if isinstance(recognition_view, RecognitionView):
            recognition_view.status.set(message)
        self._end_attendance_session()

    def _end_attendance_session(self) -> None:
        recognition_view = self._views.get("recognition")
        session_id = self._attendance_session_id
        if not isinstance(recognition_view, RecognitionView) or session_id is None:
            return
        if self._attendance_api_pending:
            recognition_view.status.set("Another attendance request is in progress.")
            return
        recognition_view.set_attendance_session("ending", self._attendance_session)
        self._stop_camera_preview()
        if not (
            self._camera_worker.is_active
            or self._detection_worker.is_active
            or self._recognition_worker.is_active
        ):
            self._submit_attendance_end()

    def _submit_attendance_end(self) -> None:
        recognition_view = self._views.get("recognition")
        session_id = self._attendance_session_id
        if (
            not isinstance(recognition_view, RecognitionView)
            or session_id is None
            or self._attendance_end_requested
        ):
            return
        self._attendance_end_requested = True
        self._run_attendance_operation(
            lambda: self._attendance_client.end_session(session_id),
            self._on_attendance_session_ended,
            busy_message="Ending session and refreshing suggestions...",
            on_error=self._on_attendance_end_failed,
        )

    def _on_attendance_end_failed(self, _message: str) -> None:
        self._attendance_end_requested = False
        recognition_view = self._views.get("recognition")
        if isinstance(recognition_view, RecognitionView):
            recognition_view.set_attendance_session("active", self._attendance_session)

    def _on_attendance_session_ended(self, payload: dict[str, Any]) -> None:
        recognition_view = self._views.get("recognition")
        if not isinstance(recognition_view, RecognitionView):
            return
        session = dict(payload.get("session") or {})
        records = list(payload.get("records") or [])
        self._attendance_session = session
        self._attendance_end_requested = True
        recognition_view.set_attendance_session("ended", session)
        recognition_view.set_attendance_records(records, phase="ended")
        recognition_view.status.set(
            "Session ended. Review every student before finalizing."
        )

    def _review_selected_student(self) -> None:
        recognition_view = self._views.get("recognition")
        session_id = self._attendance_session_id
        if not isinstance(recognition_view, RecognitionView) or session_id is None:
            return
        values = recognition_view.selected_review_values()
        if values is None or not values["final_status"]:
            recognition_view.status.set("Select a student and final decision.")
            return
        if not values["actor"]:
            recognition_view.status.set("Operator name is required.")
            return

        def operation() -> list[dict[str, Any]]:
            self._attendance_client.review_student(
                session_id,
                values["student_id"],
                values["final_status"],
                values["actor"],
                values["reason"] or None,
            )
            return self._attendance_client.list_records(session_id)

        self._run_attendance_operation(
            operation,
            lambda records: self._on_attendance_review_saved(
                values["student_id"], records
            ),
            busy_message=f"Saving review for {values['student_id']}...",
        )

    def _on_attendance_review_saved(
        self, student_id: str, records: list[dict[str, Any]]
    ) -> None:
        recognition_view = self._views.get("recognition")
        if isinstance(recognition_view, RecognitionView):
            recognition_view.set_attendance_records(records, phase="ended")
            recognition_view.status.set(f"Review saved for {student_id}.")

    def _finalize_attendance_session(self) -> None:
        recognition_view = self._views.get("recognition")
        session_id = self._attendance_session_id
        if not isinstance(recognition_view, RecognitionView) or session_id is None:
            return
        actor = recognition_view.operator_name.get().strip()
        if not actor:
            recognition_view.status.set("Operator name is required.")
            return
        self._run_attendance_operation(
            lambda: self._attendance_client.finalize_session(session_id, actor),
            self._on_attendance_session_finalized,
            busy_message="Finalizing attendance session...",
        )

    def _on_attendance_session_finalized(self, session: dict[str, Any]) -> None:
        recognition_view = self._views.get("recognition")
        if not isinstance(recognition_view, RecognitionView):
            return
        self._attendance_session = dict(session)
        recognition_view.set_attendance_session("finalized", session)
        recognition_view.status.set("Attendance session finalized. Records are read-only.")

    def _request_recognition_session_end(
        self,
        recognition_view: RecognitionView,
    ) -> None:
        if self._session_close_requested:
            return
        if self._session_service.active_session is None:
            recognition_view.set_session_active(active=False)
            recognition_view.set_session_pending(False)
            return

        self._session_close_requested = True
        recognition_view.status.set("Ending recognition session...")
        recognition_view.set_trial_available(False)
        started = self._session_service.end_active_session_async(
            on_success=lambda _ended_at: self.post_to_ui(
                self._on_recognition_session_ended
            ),
            on_error=lambda message: self.post_to_ui(
                self._on_recognition_session_end_failed, message
            ),
        )
        if not started:
            self._session_close_requested = False

    def _on_recognition_session_ended(self) -> None:
        self._session_close_requested = False
        self._recognition_session_cancel_requested = False
        recognition_view = self._views["recognition"]
        if isinstance(recognition_view, RecognitionView):
            recognition_view.set_session_active(active=False)
            recognition_view.status.set("Recognition session ended.")

    def _on_recognition_session_end_failed(self, message: str) -> None:
        self._session_close_requested = False
        recognition_view = self._views["recognition"]
        if isinstance(recognition_view, RecognitionView):
            recognition_view.status.set(message)

    def _record_recognition_trial(self) -> None:
        recognition_view = self._views["recognition"]
        if not isinstance(recognition_view, RecognitionView):
            return
        session = self._session_service.active_session
        if session is None or session.context.session_type != CAMERA_TEST_SESSION:
            recognition_view.status.set(
                "Record Trial is available only during an active Camera Test."
            )
            return
        result = self._latest_recognition_result
        if result is None:
            recognition_view.status.set(
                "No completed recognition result is available to record yet."
            )
            return

        if not 0 <= time.monotonic() - result.acquisition_time <= 1.0:
            recognition_view.status.set("Recognition result is stale; wait for a fresh observation.")
            return
        if len(result.faces) > 1:
            recognition_view.status.set("Record Trial requires a single observed face.")
            return

        try:
            trial = RecognitionTrialInput.from_form(
                recognition_view.trial_form_values()
            )
        except ValueError as error:
            recognition_view.status.set(str(error))
            return

        best_face = (
            max(result.faces, key=lambda face: face.similarity)
            if result.faces
            else None
        )
        snapshot = RecognitionTrialSnapshot(
            detected=best_face is not None,
            detector_confidence=(
                best_face.detector_confidence if best_face is not None else None
            ),
            predicted_user_id=(
                best_face.user_id
                if best_face is not None and best_face.recognized
                else None
            ),
            predicted_name=(
                best_face.full_name
                if best_face is not None and best_face.recognized
                else None
            ),
            similarity=(
                best_face.similarity
                if best_face is not None and best_face.similarity > -1.0
                else None
            ),
            threshold=result.threshold,
            recognized=bool(best_face and best_face.recognized),
            detection_time_ms=result.detection_processing_time_ms,
            recognition_time_ms=result.processing_time_ms,
            diagnostics=dict(frame_sequence=result.frame_sequence, acquisition_time=result.acquisition_time,
                             stream_id=result.stream_id, face=best_face.diagnostics if best_face else None),
        )
        recognition_view.set_trial_available(False)
        recognition_view.status.set("Saving recognition test trial...")
        started = self._session_service.record_trial_async(
            trial,
            snapshot,
            on_success=lambda trial_number: self.post_to_ui(
                self._on_recognition_trial_saved, trial_number
            ),
            on_error=lambda message: self.post_to_ui(
                self._on_recognition_trial_save_failed, message
            ),
        )
        if not started:
            recognition_view.set_trial_available(True)
            recognition_view.status.set(
                "Another trial or session operation is still in progress."
            )

    def _on_recognition_trial_saved(self, trial_number: int) -> None:
        recognition_view = self._views["recognition"]
        if isinstance(recognition_view, RecognitionView):
            recognition_view.update_trial_count(trial_number)
            recognition_view.set_trial_available(
                self._latest_recognition_result is not None
            )
            recognition_view.status.set(
                f"Trial {trial_number} recorded successfully."
            )

    def _on_recognition_trial_save_failed(self, message: str) -> None:
        recognition_view = self._views["recognition"]
        if isinstance(recognition_view, RecognitionView):
            recognition_view.set_trial_available(
                self._latest_recognition_result is not None
            )
            recognition_view.status.set(message)

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
            recognition_view = self._views.get("recognition")
            if self._recognition_session_create_pending:
                self._cancel_pending_recognition_start()
                if isinstance(recognition_view, RecognitionView):
                    recognition_view.set_session_pending(True)
                return
            if (
                self._session_service.active_session is not None
                and isinstance(recognition_view, RecognitionView)
            ):
                self._request_recognition_session_end(recognition_view)
            return

        owner_view = self._views[owner_view_name]
        detection_owner = isinstance(owner_view, DetectionView)
        recognition_owner = isinstance(owner_view, RecognitionView)

        if recognition_owner:
            LOGGER.info("Recognition stop requested")
            self._recognition_start_pending = False
            self._recognition_worker.stop(wait=False)
            self._detection_worker.stop(wait=False)
            self._reset_recognition_state(owner_view)
            owner_view.recognition_status.set("Recognition Status: Stopping")
        elif detection_owner:
            LOGGER.info("Detection stop requested")
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
            LOGGER.info("Recognition workers stopped")
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
            self._request_recognition_session_end(owner_view)
            if owner_view.attendance_phase == "ending":
                self._submit_attendance_end()
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
                LOGGER.info("Camera state changed; state=%s", status.state)
                active = status.state in {"connecting", "connected"}
                stopping = status.state == "stopping"
                self._set_view_camera_state(
                    owner_view,
                    status.message,
                    active=active,
                    stopping=stopping,
                )

                if status.state == "error":
                    LOGGER.error("Camera error reported")
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

                if (
                    status.state == "connected"
                    and isinstance(owner_view, RecognitionView)
                    and not self._session_camera_metadata_saved
                ):
                    metadata = self._camera_worker.metadata
                    if metadata is not None:
                        resolution = (
                            f"{metadata.width}x{metadata.height}"
                            if metadata.width is not None
                            and metadata.height is not None
                            else None
                        )
                        self._session_camera_metadata_saved = (
                            self._session_service.update_camera_metadata_async(
                                camera_resolution=resolution,
                                camera_fps=metadata.fps,
                            )
                        )

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
                        acquisition_time=frame.acquisition_time, stream_id=frame.stream_id,
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
                        acquisition_time=frame.acquisition_time, stream_id=frame.stream_id,
                    )

                    if (self._latest_recognition_result is not None
                        and self._latest_recognition_result.stream_id == frame.stream_id
                        and 0 <= time.monotonic() - self._latest_recognition_result.acquisition_time <= 1.0):
                        try:
                            display_frame = draw_recognition_results(
                                self._latest_recognition_result.source_frame,
                                self._latest_recognition_result.faces,
                            )
                        except Exception:
                            owner_view.status.set(
                                "Recognition drawing error."
                            )
                            self._latest_recognition_result = None
                    elif (self._latest_detection_result is not None
                          and self._latest_detection_result.stream_id == frame.stream_id
                          and self._latest_detection_result.frame_sequence == frame.sequence):
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
                    self._finalize_camera_owner(final_message=status.message)

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
            if self._attendance_session_id is not None:
                started = self._recognition_worker.start_attendance_session(
                    self._attendance_session_id
                )
            else:
                session = self._session_service.active_session
                started = bool(
                    session is not None
                    and self._recognition_worker.start(session_id=session.session_id)
                )
            if started:
                LOGGER.info("AdaFace recognition worker started")
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
                self._end_attendance_session_after_failure(
                    "Attendance recognition could not be started."
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
                recognition_view.set_quality_diagnostics(None)
                recognition_view.set_matcher_diagnostics(None)
                self._latest_recognition_result = None

        if recognition_status.state == "error":
            # Keep the recognition failure visible even if a later camera
            # status revision reports that the RTSP stream is connected.
            recognition_view.status.set(recognition_status.message)

        context = self._recognition_worker.attendance_context
        if context is not None and not self._attendance_missing_gallery_reported:
            self._attendance_missing_gallery_reported = True
            recognition_view.set_missing_gallery_students(
                context.missing_gallery_student_ids
            )

        self._poll_live_attendance_records(recognition_view)

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
        recognition_view.threshold.set(f"Threshold: {result.threshold:.2f}")
        active_session = self._session_service.active_session
        recognition_view.set_trial_available(
            active_session is not None
            and active_session.context.session_type == CAMERA_TEST_SESSION
        )
        self._update_recognition_details(recognition_view, result)

    def _poll_live_attendance_records(
        self, recognition_view: RecognitionView
    ) -> None:
        session_id = self._attendance_session_id
        if (
            session_id is None
            or recognition_view.attendance_phase != "active"
            or self._attendance_poll_pending
            or self._attendance_api_pending
            or time.monotonic() < self._attendance_next_poll_at
        ):
            return
        self._attendance_poll_pending = True
        self._attendance_next_poll_at = time.monotonic() + 1.0
        future = self._attendance_executor.submit(
            self._attendance_client.list_records, session_id
        )

        def completed(result_future: Future) -> None:
            try:
                records = result_future.result()
            except AttendanceClientError as error:
                self.post_to_ui(self._on_attendance_poll_failed, str(error))
            except Exception:
                LOGGER.exception("Live attendance record polling failed")
                self.post_to_ui(
                    self._on_attendance_poll_failed,
                    "Live attendance evidence could not be refreshed.",
                )
            else:
                self.post_to_ui(
                    self._on_attendance_poll_succeeded, session_id, records
                )

        future.add_done_callback(completed)

    def _on_attendance_poll_succeeded(
        self, session_id: str, records: list[dict[str, Any]]
    ) -> None:
        self._attendance_poll_pending = False
        recognition_view = self._views.get("recognition")
        if (
            isinstance(recognition_view, RecognitionView)
            and session_id == self._attendance_session_id
            and recognition_view.attendance_phase == "active"
        ):
            recognition_view.set_attendance_records(records, phase="active")

    def _on_attendance_poll_failed(self, message: str) -> None:
        self._attendance_poll_pending = False
        recognition_view = self._views.get("recognition")
        if isinstance(recognition_view, RecognitionView):
            recognition_view.status.set(message)

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
            recognition_view.set_quality_diagnostics(None)
            recognition_view.set_matcher_diagnostics(None)
            return

        best_face = max(
            result.faces,
            key=lambda face: face.similarity,
        )
        recognized_count = sum(
            1 for face in result.faces if face.recognized
        )
        recognition_view.set_quality_diagnostics(best_face.diagnostics)
        recognition_view.set_matcher_diagnostics(best_face.diagnostics)

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
                best_face.failure_reason or "Unknown",
            )
            recognition_view.recognition_status.set(
                f"Recognition Status: Unknown ({reason})"
            )

        if best_face.similarity > -1.0:
            recognition_view.similarity.set(
                "Similarity / Confidence: "
                f"{best_face.similarity:.4f} "
                f"(T {result.threshold:.2f}, margin {best_face.margin}, "
                f"track {best_face.track_id}, evidence {best_face.valid_observations})"
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
        self._recognition_session_cancel_requested = True
        LOGGER.info("Application shutdown requested")

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

        # Let any in-flight session insert/trial/metadata operation finish,
        # then close the active Supabase session before destroying Tk. The
        # event loop remains alive while these network operations run.
        if self._session_service.is_busy:
            self._shutdown_after_id = self.after(
                50,
                self._finish_shutdown_when_workers_stop,
            )
            return

        if self._session_service.active_session is not None:
            if self._shutdown_session_close_attempts < 3:
                started = self._session_service.end_active_session_async()
                if started:
                    self._shutdown_session_close_attempts += 1
                self._shutdown_after_id = self.after(
                    50,
                    self._finish_shutdown_when_workers_stop,
                )
                return
            LOGGER.error(
                "Recognition session could not be closed during shutdown "
                "after %d attempts",
                self._shutdown_session_close_attempts,
            )

        self._session_service.shutdown()

        if not self._attendance_executor_closed:
            self._attendance_executor.shutdown(wait=True, cancel_futures=True)
            self._attendance_executor_closed = True
        recognition_view = self._views.get("recognition")
        if (
            self._attendance_session_id is not None
            and isinstance(recognition_view, RecognitionView)
            and recognition_view.attendance_phase in {"active", "starting", "ending"}
            and not self._shutdown_attendance_close_attempted
        ):
            self._shutdown_attendance_close_attempted = True
            try:
                self._attendance_client.end_session(self._attendance_session_id)
            except AttendanceClientError as error:
                LOGGER.error("Attendance session could not be ended on shutdown: %s", error)

        for handler in reversed(self._shutdown_handlers):
            try:
                handler()
            except Exception:
                LOGGER.exception("Desktop shutdown handler failed")

        LOGGER.info("Application shutdown complete")
        self.destroy()


def main() -> int:
    """Launch the desktop application."""
    configure_desktop_logging()
    application = FaceAttendanceDesktopApp()
    application.mainloop()
    return 0


__all__ = ["FaceAttendanceDesktopApp", "main"]
