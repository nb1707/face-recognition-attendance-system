"""Face-registration view with a local-camera preview."""

from __future__ import annotations

import tkinter as tk
from collections.abc import Callable
from tkinter import ttk

from ..widgets import VideoPanel


class RegisterView(ttk.Frame):
    """Input placeholders plus a local-webcam preview."""

    def __init__(self, parent: tk.Misc) -> None:
        super().__init__(parent, padding=20)
        self.columnconfigure(0, weight=1)
        self.rowconfigure(1, weight=1)

        self.user_id = tk.StringVar(master=self)
        self.full_name = tk.StringVar(master=self)
        self.role = tk.StringVar(master=self, value="Student")
        self.status = tk.StringVar(
            master=self,
            value="Camera stopped. Registration is not connected yet.",
        )
        self._start_camera_command: Callable[[], None] | None = None
        self._stop_camera_command: Callable[[], None] | None = None

        ttk.Label(
            self,
            text="Register Face",
            style="PageTitle.TLabel",
        ).grid(row=0, column=0, sticky="w", pady=(0, 14))

        body = ttk.Frame(self)
        body.grid(row=1, column=0, sticky="nsew")
        body.columnconfigure(0, weight=1, minsize=300)
        body.columnconfigure(1, weight=2)
        body.rowconfigure(0, weight=1)

        form = ttk.LabelFrame(body, text="User Details", padding=18)
        form.grid(row=0, column=0, sticky="nsew", padx=(0, 14))
        form.columnconfigure(0, weight=1)

        ttk.Label(form, text="User ID").grid(
            row=0,
            column=0,
            sticky="w",
        )
        self.user_id_entry = ttk.Entry(
            form,
            textvariable=self.user_id,
        )
        self.user_id_entry.grid(
            row=1,
            column=0,
            sticky="ew",
            pady=(4, 14),
        )

        ttk.Label(form, text="Name").grid(
            row=2,
            column=0,
            sticky="w",
        )
        ttk.Entry(
            form,
            textvariable=self.full_name,
        ).grid(row=3, column=0, sticky="ew", pady=(4, 14))

        ttk.Label(form, text="Role").grid(
            row=4,
            column=0,
            sticky="w",
        )
        ttk.Combobox(
            form,
            textvariable=self.role,
            values=("Student", "Teacher", "Staff"),
            state="readonly",
        ).grid(row=5, column=0, sticky="ew", pady=(4, 22))

        pose_group = ttk.LabelFrame(
            form,
            text="Pose Progress",
            padding=12,
        )
        pose_group.grid(row=6, column=0, sticky="ew")
        pose_group.columnconfigure(0, weight=1)

        self.pose_labels: dict[str, ttk.Label] = {}
        for row_index, pose_name in enumerate(("Front", "Left", "Right")):
            pose_label = ttk.Label(
                pose_group,
                text=f"{pose_name} pose: Not captured",
            )
            pose_label.grid(
                row=row_index,
                column=0,
                sticky="w",
                pady=3,
            )
            self.pose_labels[pose_name.lower()] = pose_label

        preview = ttk.LabelFrame(
            body,
            text="Camera Preview",
            padding=12,
        )
        preview.grid(row=0, column=1, sticky="nsew")
        preview.columnconfigure(0, weight=1)
        preview.rowconfigure(0, weight=1)

        self.video_panel = VideoPanel(
            preview,
            width=640,
            height=400,
            placeholder="Registration camera preview",
        )
        self.video_panel.grid(row=0, column=0, sticky="nsew")

        actions = ttk.Frame(self)
        actions.grid(row=2, column=0, sticky="ew", pady=(14, 0))
        actions.columnconfigure(4, weight=1)

        self.start_camera_button = ttk.Button(
            actions,
            text="Start Camera",
            command=self._start_camera,
        )
        self.start_camera_button.grid(row=0, column=0, padx=(0, 8))

        self.stop_camera_button = ttk.Button(
            actions,
            text="Stop Camera",
            command=self._stop_camera,
            state="disabled",
        )
        self.stop_camera_button.grid(row=0, column=1, padx=(0, 8))

        self.capture_button = ttk.Button(
            actions,
            text="Capture",
            command=lambda: self._set_placeholder_status(
                "Capture is not implemented in this preview-only phase."
            ),
        )
        self.capture_button.grid(row=0, column=2, padx=(0, 8))

        self.register_button = ttk.Button(
            actions,
            text="Register",
            command=lambda: self._set_placeholder_status(
                "Registration will be connected in a later phase."
            ),
        )
        self.register_button.grid(row=0, column=3, padx=(0, 16))

        status_area = ttk.LabelFrame(
            actions,
            text="Status / Message",
            padding=(12, 8),
        )
        status_area.grid(row=0, column=4, sticky="ew")
        status_area.columnconfigure(0, weight=1)

        ttk.Label(
            status_area,
            textvariable=self.status,
            wraplength=560,
            justify="left",
        ).grid(row=0, column=0, sticky="w")

    def _set_placeholder_status(self, message: str) -> None:
        self.status.set(message)

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
        """Update controls from the Tkinter main thread."""
        self.status.set(message)
        self.start_camera_button.configure(
            state="disabled" if active or stopping else "normal"
        )
        self.stop_camera_button.configure(
            state="normal" if active and not stopping else "disabled"
        )

    def _start_camera(self) -> None:
        if self._start_camera_command is not None:
            self._start_camera_command()

    def _stop_camera(self) -> None:
        if self._stop_camera_command is not None:
            self._stop_camera_command()
