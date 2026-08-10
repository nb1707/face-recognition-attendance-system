"""Hikvision realtime face-detection view."""

from __future__ import annotations

import tkinter as tk
from collections.abc import Callable
from tkinter import ttk

from ..widgets import VideoPanel


class DetectionView(ttk.Frame):
    """Hikvision tiled-detection preview and metrics."""

    def __init__(self, parent: tk.Misc) -> None:
        super().__init__(parent, padding=20)
        self.columnconfigure(0, weight=1)
        self.rowconfigure(1, weight=1)

        self.detected_faces = tk.StringVar(
            master=self,
            value="Detected Faces: 0",
        )
        self.fps = tk.StringVar(master=self, value="FPS: 0.0")
        self.processing_time = tk.StringVar(
            master=self,
            value="Processing Time: -- ms",
        )
        self.status = tk.StringVar(
            master=self,
            value="Camera stopped.",
        )
        self._start_camera_command: Callable[[], None] | None = None
        self._stop_camera_command: Callable[[], None] | None = None

        ttk.Label(
            self,
            text="Detect Face",
            style="PageTitle.TLabel",
        ).grid(row=0, column=0, sticky="w", pady=(0, 14))

        body = ttk.Frame(self)
        body.grid(row=1, column=0, sticky="nsew")
        body.columnconfigure(0, weight=3)
        body.columnconfigure(1, weight=1, minsize=240)
        body.rowconfigure(0, weight=1)

        preview = ttk.LabelFrame(
            body,
            text="Camera Preview",
            padding=12,
        )
        preview.grid(row=0, column=0, sticky="nsew", padx=(0, 14))
        preview.columnconfigure(0, weight=1)
        preview.rowconfigure(0, weight=1)

        self.video_panel = VideoPanel(
            preview,
            width=720,
            height=440,
            placeholder="Detection camera preview",
        )
        self.video_panel.grid(row=0, column=0, sticky="nsew")

        controls = ttk.LabelFrame(
            body,
            text="Detection Controls",
            padding=16,
        )
        controls.grid(row=0, column=1, sticky="nsew")
        controls.columnconfigure(0, weight=1)

        self.start_button = ttk.Button(
            controls,
            text="Start Camera",
            command=self._start_camera,
        )
        self.start_button.grid(row=0, column=0, sticky="ew", pady=(0, 8))

        self.stop_button = ttk.Button(
            controls,
            text="Stop Camera",
            command=self._stop_camera,
            state="disabled",
        )
        self.stop_button.grid(row=1, column=0, sticky="ew", pady=(0, 22))

        ttk.Separator(
            controls,
            orient="horizontal",
        ).grid(row=2, column=0, sticky="ew", pady=(0, 16))

        for row_index, variable in enumerate(
            (
                self.detected_faces,
                self.fps,
                self.processing_time,
            ),
            start=3,
        ):
            ttk.Label(
                controls,
                textvariable=variable,
            ).grid(row=row_index, column=0, sticky="w", pady=6)

        status_area = ttk.LabelFrame(
            self,
            text="Status",
            padding=(12, 8),
        )
        status_area.grid(row=2, column=0, sticky="ew", pady=(14, 0))
        status_area.columnconfigure(0, weight=1)

        ttk.Label(
            status_area,
            textvariable=self.status,
            wraplength=820,
            justify="left",
        ).grid(row=0, column=0, sticky="w")

    def set_camera_commands(
        self,
        *,
        start_command: Callable[[], None],
        stop_command: Callable[[], None],
    ) -> None:
        """Bind application-owned camera lifecycle commands."""
        self._start_camera_command = start_command
        self._stop_camera_command = stop_command

    def set_camera_state(
        self,
        message: str,
        *,
        active: bool,
        stopping: bool = False,
    ) -> None:
        """Update camera controls from the Tkinter main thread."""
        self.status.set(message)
        self.start_button.configure(
            state="disabled" if active or stopping else "normal"
        )
        self.stop_button.configure(
            state="normal" if active and not stopping else "disabled"
        )

    def _start_camera(self) -> None:
        if self._start_camera_command is not None:
            self._start_camera_command()

    def _stop_camera(self) -> None:
        if self._stop_camera_command is not None:
            self._stop_camera_command()
