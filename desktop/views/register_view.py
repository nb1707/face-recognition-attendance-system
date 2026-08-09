"""Face-registration view skeleton."""

import tkinter as tk
from tkinter import ttk

from ..widgets import VideoPanel


class RegisterView(ttk.Frame):
    """Input and capture placeholders for future face registration."""

    def __init__(self, parent: tk.Misc) -> None:
        super().__init__(parent, padding=20)
        self.columnconfigure(0, weight=1)
        self.rowconfigure(1, weight=1)

        self.user_id = tk.StringVar(master=self)
        self.full_name = tk.StringVar(master=self)
        self.role = tk.StringVar(master=self, value="Student")
        self.status = tk.StringVar(
            master=self,
            value="Ready. Registration logic is not connected in this phase.",
        )

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
        actions.columnconfigure(2, weight=1)

        self.capture_button = ttk.Button(
            actions,
            text="Capture",
            command=lambda: self._set_placeholder_status(
                "Capture will be connected in the camera phase."
            ),
        )
        self.capture_button.grid(row=0, column=0, padx=(0, 8))

        self.register_button = ttk.Button(
            actions,
            text="Register",
            command=lambda: self._set_placeholder_status(
                "Registration will be connected in a later phase."
            ),
        )
        self.register_button.grid(row=0, column=1, padx=(0, 16))

        status_area = ttk.LabelFrame(
            actions,
            text="Status / Message",
            padding=(12, 8),
        )
        status_area.grid(row=0, column=2, sticky="ew")
        status_area.columnconfigure(0, weight=1)

        ttk.Label(
            status_area,
            textvariable=self.status,
            wraplength=560,
            justify="left",
        ).grid(row=0, column=0, sticky="w")

    def _set_placeholder_status(self, message: str) -> None:
        self.status.set(message)

