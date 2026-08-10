"""Framework-neutral camera preview support for the desktop application."""

from .hikvision import CameraConfigurationError, build_hikvision_rtsp_source
from .worker import CameraFrame, CameraStatus, CameraWorker

__all__ = [
    "CameraConfigurationError",
    "CameraFrame",
    "CameraStatus",
    "CameraWorker",
    "build_hikvision_rtsp_source",
]
