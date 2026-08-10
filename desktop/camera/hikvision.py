"""Resolve the existing Hikvision environment configuration safely."""

from __future__ import annotations

import os
from pathlib import Path
from urllib.parse import quote


class CameraConfigurationError(RuntimeError):
    """A safe, user-facing camera configuration error."""


_BACKEND_ENV_FILE = (
    Path(__file__).resolve().parents[2]
    / "backend"
    / ".env"
)


def _load_existing_environment() -> None:
    """Load the project's existing environment file without overriding values."""
    try:
        from dotenv import load_dotenv
    except ImportError as error:
        raise CameraConfigurationError(
            "Camera configuration support is unavailable."
        ) from error

    load_dotenv(
        dotenv_path=_BACKEND_ENV_FILE,
        override=False,
    )


def _read_integer_setting(
    name: str,
    default: str,
) -> int:
    try:
        return int(os.getenv(name, default))
    except (TypeError, ValueError) as error:
        raise CameraConfigurationError(
            "Hikvision camera configuration is invalid."
        ) from error


def build_hikvision_rtsp_source() -> str:
    """Build the configured RTSP source without logging credentials.

    This intentionally mirrors only the URL construction currently used by the
    web application. It does not initialize the FastAPI application or any
    backend, cloud, or recognition resources.
    """
    _load_existing_environment()

    host = os.getenv("HIKVISION_HOST", "").strip()
    username = os.getenv("HIKVISION_USERNAME", "").strip()
    password = os.getenv("HIKVISION_PASSWORD", "")
    port = _read_integer_setting("HIKVISION_RTSP_PORT", "554")
    channel = _read_integer_setting("HIKVISION_CHANNEL", "1")
    stream_type = _read_integer_setting("HIKVISION_STREAM_TYPE", "1")

    if not host or not username or not password:
        raise CameraConfigurationError(
            "Hikvision camera configuration is incomplete."
        )

    if not 1 <= port <= 65535 or channel < 1 or stream_type not in {1, 2, 3}:
        raise CameraConfigurationError(
            "Hikvision camera configuration is invalid."
        )

    encoded_username = quote(username, safe="")
    encoded_password = quote(password, safe="")
    channel_id = channel * 100 + stream_type

    return (
        f"rtsp://{encoded_username}:{encoded_password}@"
        f"{host}:{port}/Streaming/Channels/{channel_id}"
    )
