"""Realtime AdaFace recognition view."""

import tkinter as tk
from collections.abc import Callable
from tkinter import ttk

from ..widgets import VideoPanel


class RecognitionView(ttk.Frame):
    """Hikvision preview and baseline AdaFace recognition details."""

    def __init__(self, parent: tk.Misc) -> None:
        super().__init__(parent, padding=20)
        self.columnconfigure(0, weight=1)
        self.rowconfigure(1, weight=1)

        self.active_model = tk.StringVar(
            master=self,
            value="Active Model: Not Loaded",
        )
        self.identity = tk.StringVar(master=self, value="Identity: --")
        self.user_id = tk.StringVar(master=self, value="User ID: --")
        self.role = tk.StringVar(master=self, value="Role: --")
        self.similarity = tk.StringVar(
            master=self,
            value="Similarity / Confidence: --",
        )
        self.recognition_status = tk.StringVar(
            master=self,
            value="Recognition Status: Idle",
        )
        self.processing_time = tk.StringVar(
            master=self,
            value="Processing Time: -- ms",
        )
        self.status = tk.StringVar(
            master=self,
            value="Idle. No recognition model has been loaded.",
        )

        ttk.Label(
            self,
            text="Recognize Face",
            style="PageTitle.TLabel",
        ).grid(row=0, column=0, sticky="w", pady=(0, 14))

        body = ttk.Frame(self)
        body.grid(row=1, column=0, sticky="nsew")
        body.columnconfigure(0, weight=3)
        body.columnconfigure(1, weight=1, minsize=270)
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
            width=700,
            height=440,
            placeholder="Recognition camera preview",
        )
        self.video_panel.grid(row=0, column=0, sticky="nsew")

        details = ttk.LabelFrame(
            body,
            text="Recognition Details",
            padding=16,
        )
        details.grid(row=0, column=1, sticky="nsew")
        details.columnconfigure(0, weight=1)

        self.start_button = ttk.Button(
            details,
            text="Start Recognition",
        )
        self.start_button.grid(row=0, column=0, sticky="ew", pady=(0, 8))

        self.stop_button = ttk.Button(
            details,
            text="Stop Recognition",
            state="disabled",
        )
        self.stop_button.grid(row=1, column=0, sticky="ew", pady=(0, 20))

        ttk.Separator(
            details,
            orient="horizontal",
        ).grid(row=2, column=0, sticky="ew", pady=(0, 12))

        detail_variables = (
            self.active_model,
            self.identity,
            self.user_id,
            self.role,
            self.similarity,
            self.recognition_status,
            self.processing_time,
        )

        for row_index, variable in enumerate(detail_variables, start=3):
            ttk.Label(
                details,
                textvariable=variable,
                wraplength=240,
                justify="left",
            ).grid(row=row_index, column=0, sticky="w", pady=5)

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

    def set_recognition_commands(
        self,
        *,
        start_command: Callable[[], None],
        stop_command: Callable[[], None],
    ) -> None:
        """Bind application-owned lifecycle commands to the view."""
        self.start_button.configure(command=start_command)
        self.stop_button.configure(command=stop_command)

    def set_camera_state(
        self,
        message: str,
        *,
        active: bool,
        stopping: bool = False,
    ) -> None:
        """Update controls from Tkinter's event-loop thread."""
        self.status.set(message)

        if stopping:
            self.start_button.configure(state="disabled")
            self.stop_button.configure(state="disabled")
        elif active:
            self.start_button.configure(state="disabled")
            self.stop_button.configure(state="normal")
        else:
            self.start_button.configure(state="normal")
            self.stop_button.configure(state="disabled")

    def reset_recognition_details(self) -> None:
        self.identity.set("Identity: --")
        self.user_id.set("User ID: --")
        self.role.set("Role: --")
        self.similarity.set("Similarity / Confidence: --")
        self.recognition_status.set("Recognition Status: Idle")
        self.processing_time.set("Processing Time: -- ms")
