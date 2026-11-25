"""Recognition session persistence for the desktop application."""

from .models import (
    CAMERA_TEST_SESSION,
    REAL_CLASS_SESSION,
    UNKNOWN_EXPECTED_USER_ID,
    RecognitionConfiguration,
    RecognitionSession,
    RecognitionSessionInput,
    RecognitionTrialInput,
    RecognitionTrialSnapshot,
    load_recognition_configuration_snapshot,
)
from .repository import RecognitionSessionRepository, SessionPersistenceError
from .service import RecognitionSessionService

__all__ = [
    "CAMERA_TEST_SESSION",
    "REAL_CLASS_SESSION",
    "UNKNOWN_EXPECTED_USER_ID",
    "RecognitionConfiguration",
    "RecognitionSession",
    "RecognitionSessionInput",
    "RecognitionSessionRepository",
    "RecognitionSessionService",
    "RecognitionTrialInput",
    "RecognitionTrialSnapshot",
    "SessionPersistenceError",
    "load_recognition_configuration_snapshot",
]
