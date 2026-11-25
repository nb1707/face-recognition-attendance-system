"""Select a Recognition camera source; constructing a source never opens it."""
import os
from .hikvision import CameraConfigurationError, build_hikvision_rtsp_source


def build_recognition_camera_source(selection: str | None = None) -> int | str:
    selection = (selection or os.environ.get(
        'RECOGNITION_CAMERA_SOURCE', 'webcam'
    )).strip().lower()
    if selection == 'hikvision':
        return build_hikvision_rtsp_source()
    if selection != 'webcam':
        raise CameraConfigurationError('Recognition camera source must be webcam or hikvision.')
    try:
        index = int(os.environ.get('RECOGNITION_WEBCAM_INDEX', '0'))
    except ValueError:
        raise CameraConfigurationError('Recognition webcam index must be a nonnegative integer.') from None
    if index < 0:
        raise CameraConfigurationError('Recognition webcam index must be a nonnegative integer.')
    return index
