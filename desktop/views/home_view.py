"""Home view for the desktop shell."""

import tkinter as tk
from tkinter import ttk


class HomeView(ttk.Frame):
    """Overview of the desktop modules available in future phases."""

    def __init__(self, parent: tk.Misc) -> None:
        super().__init__(parent, padding=28)
        self.columnconfigure(0, weight=1)

        ttk.Label(
            self,
            text="AI Face Recognition Attendance System",
            style="PageTitle.TLabel",
        ).grid(row=0, column=0, sticky="w", pady=(0, 10))

        ttk.Label(
            self,
            text=(
                "Desktop foundation for the project's face registration, "
                "detection, and recognition workflows."
            ),
            wraplength=760,
            justify="left",
        ).grid(row=1, column=0, sticky="w", pady=(0, 26))

        modules = ttk.LabelFrame(
            self,
            text="Available Modules",
            padding=20,
        )
        modules.grid(row=2, column=0, sticky="ew")
        modules.columnconfigure(0, weight=1)

        module_descriptions = (
            (
                "Face Registration",
                "Capture front, left, and right face poses for a user.",
            ),
            (
                "Face Detection",
                "Preview camera frames and inspect face-detection metrics.",
            ),
            (
                "Face Recognition",
                "Display identity and similarity results from a selected model.",
            ),
        )

        for row_index, (name, description) in enumerate(module_descriptions):
            module = ttk.Frame(modules, padding=(8, 10))
            module.grid(row=row_index, column=0, sticky="ew")
            module.columnconfigure(1, weight=1)

            ttk.Label(
                module,
                text=f"{row_index + 1}.",
                style="SectionTitle.TLabel",
            ).grid(row=0, column=0, sticky="nw", padx=(0, 12))

            ttk.Label(
                module,
                text=name,
                style="SectionTitle.TLabel",
            ).grid(row=0, column=1, sticky="w")

            ttk.Label(
                module,
                text=description,
                wraplength=680,
                justify="left",
            ).grid(row=1, column=1, sticky="w", pady=(3, 0))

        ttk.Label(
            self,
            text=(
                "This phase provides interface placeholders only. No camera, "
                "AI model, or cloud connection is active."
            ),
            wraplength=760,
            justify="left",
        ).grid(row=3, column=0, sticky="w", pady=(24, 0))

