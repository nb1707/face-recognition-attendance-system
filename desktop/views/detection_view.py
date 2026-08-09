"""Face-detection view skeleton."""

import tkinter as tk
from tkinter import ttk

from ..widgets import VideoPanel


class DetectionView(ttk.Frame):
    """Camera and metric placeholders for future face detection."""

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
            value="Idle. Face detection is not loaded.",
        )

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
            command=lambda: self.status.set(
                "Camera startup will be implemented in the next phase."
            ),
        )
        self.start_button.grid(row=0, column=0, sticky="ew", pady=(0, 8))

        self.stop_button = ttk.Button(
            controls,
            text="Stop Camera",
            command=lambda: self.status.set("Camera is not running."),
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

