"""Reusable image/video display panel for Tkinter views."""

from __future__ import annotations

import tkinter as tk
from typing import Any
from tkinter import ttk

try:
    from PIL import Image, ImageTk
except ImportError:
    Image = None
    ImageTk = None


class VideoPanel(ttk.Frame):
    """Fixed-size preview area with optional Pillow image conversion.

    Image display methods must be called on the Tkinter main thread. Future
    OpenCV frames can be passed to :meth:`show_bgr_frame`; this widget converts
    BGR to RGB without importing OpenCV itself.
    """

    def __init__(
        self,
        parent: tk.Misc,
        *,
        width: int = 640,
        height: int = 480,
        placeholder: str = "No video source connected",
    ) -> None:
        super().__init__(parent)

        if width <= 0 or height <= 0:
            raise ValueError("VideoPanel dimensions must be positive.")

        self.display_width = width
        self.display_height = height
        self.default_placeholder = placeholder
        self._photo_image: Any | None = None

        self.columnconfigure(0, weight=1)
        self.rowconfigure(0, weight=1)

        self.canvas = tk.Canvas(
            self,
            width=width,
            height=height,
            background="#111827",
            highlightthickness=0,
        )
        self.canvas.grid(row=0, column=0, sticky="nsew")

        self._image_item = self.canvas.create_image(
            width // 2,
            height // 2,
            anchor=tk.CENTER,
        )
        self._placeholder_item = self.canvas.create_text(
            width // 2,
            height // 2,
            text=placeholder,
            fill="#d1d5db",
            font=("Segoe UI", 12),
            justify=tk.CENTER,
            width=max(100, width - 40),
        )

        if not self.pillow_available:
            ttk.Label(
                self,
                text=(
                    "Pillow is unavailable. The shell can run, but image "
                    "display is disabled."
                ),
                wraplength=width,
                justify="left",
            ).grid(row=1, column=0, sticky="w", pady=(6, 0))

    @property
    def pillow_available(self) -> bool:
        """Return whether PIL.Image and ImageTk are importable."""
        return Image is not None and ImageTk is not None

    def show_bgr_frame(self, frame: Any) -> None:
        """Display an OpenCV-style BGR array using Pillow.

        The conversion pipeline is BGR array -> RGB array -> PIL Image ->
        ImageTk.PhotoImage -> Tk canvas.
        """
        if not self.pillow_available:
            raise RuntimeError(
                "Pillow is required to display OpenCV frames in Tkinter."
            )

        if frame is None:
            raise ValueError("Cannot display a missing frame.")

        try:
            rgb_frame = frame[:, :, :3][:, :, ::-1].copy()
        except (IndexError, TypeError, AttributeError) as error:
            raise ValueError(
                "Expected a NumPy-like BGR image with three channels."
            ) from error

        self.show_pil_image(Image.fromarray(rgb_frame))

    def show_pil_image(self, image: Any) -> None:
        """Resize and display a Pillow image while retaining its Tk reference."""
        if not self.pillow_available:
            raise RuntimeError(
                "Pillow is required to display images in Tkinter."
            )

        if image is None:
            raise ValueError("Cannot display a missing image.")

        display_image = image.copy()
        resampling = getattr(Image, "Resampling", Image)
        display_image.thumbnail(
            (self.display_width, self.display_height),
            resampling.LANCZOS,
        )

        photo_image = ImageTk.PhotoImage(
            image=display_image,
            master=self,
        )

        self.canvas.itemconfigure(
            self._image_item,
            image=photo_image,
        )
        self.canvas.itemconfigure(
            self._placeholder_item,
            state="hidden",
        )

        # Tkinter does not retain this object on behalf of the canvas.
        self._photo_image = photo_image

    def clear(self, message: str | None = None) -> None:
        """Remove the displayed image and restore placeholder text."""
        self.canvas.itemconfigure(self._image_item, image="")
        self.canvas.itemconfigure(
            self._placeholder_item,
            text=message or self.default_placeholder,
            state="normal",
        )
        self._photo_image = None

