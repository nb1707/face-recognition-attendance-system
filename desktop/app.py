"""Tkinter application shell.

This module contains navigation, main-thread callback dispatch, and the single
application shutdown path. It deliberately does not import the FastAPI app,
camera code, AI models, or cloud integrations.
"""

from __future__ import annotations

import logging
import queue
import tkinter as tk
from collections.abc import Callable
from typing import Any
from tkinter import ttk

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

    def __init__(self) -> None:
        super().__init__()

        self._is_shutting_down = False
        self._shutdown_handlers: list[Callable[[], None]] = []
        self._ui_callbacks: queue.Queue[UiCallback] = queue.Queue()
        self._ui_poll_after_id: str | None = None

        self._views: dict[str, ttk.Frame] = {}
        self._navigation_buttons: dict[str, ttk.Button] = {}
        self._current_view_name = tk.StringVar(
            master=self,
            value="Home",
        )

        self._configure_window()
        self._build_layout()
        self.show_view("home")
        self._schedule_ui_queue_poll()

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
        """Run all cleanup hooks and destroy the one Tk root exactly once."""
        if self._is_shutting_down:
            return

        self._is_shutting_down = True

        if self._ui_poll_after_id is not None:
            try:
                self.after_cancel(self._ui_poll_after_id)
            except tk.TclError:
                pass
            self._ui_poll_after_id = None

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

