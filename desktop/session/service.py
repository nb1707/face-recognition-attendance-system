"""Asynchronous session/trial lifecycle used by the Tkinter coordinator."""

from __future__ import annotations

import logging
import threading
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor
from typing import Any

from .models import (
    CAMERA_TEST_SESSION,
    RecognitionConfiguration,
    RecognitionSession,
    RecognitionSessionInput,
    RecognitionTrialInput,
    RecognitionTrialSnapshot,
)
from .repository import RecognitionSessionRepository, SessionPersistenceError

LOGGER = logging.getLogger(__name__)
SuccessCallback = Callable[[Any], None]
ErrorCallback = Callable[[str], None]


class RecognitionSessionService:
    """Serialize repository operations on one non-UI background thread."""

    def __init__(
        self,
        repository: RecognitionSessionRepository | None = None,
    ) -> None:
        self._repository = repository or RecognitionSessionRepository()
        self._executor = ThreadPoolExecutor(
            max_workers=1,
            thread_name_prefix="desktop-recognition-session",
        )
        self._lock = threading.Lock()
        self._active_session: RecognitionSession | None = None
        self._trial_count = 0
        self._operation_count = 0
        self._creating = False
        self._ending = False
        self._trial_pending = False
        self._users_loading = False
        self._closed = False

    @property
    def active_session(self) -> RecognitionSession | None:
        with self._lock:
            return self._active_session

    @property
    def trial_count(self) -> int:
        with self._lock:
            return self._trial_count

    @property
    def is_busy(self) -> bool:
        with self._lock:
            return self._operation_count > 0

    @property
    def is_creating(self) -> bool:
        with self._lock:
            return self._creating

    def create_session_async(
        self,
        context: RecognitionSessionInput,
        configuration: RecognitionConfiguration,
        *,
        on_success: SuccessCallback,
        on_error: ErrorCallback,
    ) -> bool:
        with self._lock:
            if (
                self._closed
                or self._creating
                or self._active_session is not None
                or self._ending
            ):
                return False
            self._creating = True
            self._operation_count += 1

        def operation() -> None:
            try:
                row = self._repository.create_session(
                    context.to_insert_payload(configuration)
                )
                session_id = str(row.get("id") or "").strip()
                if not session_id:
                    raise SessionPersistenceError(
                        "Recognition session did not return an ID."
                    )
                session = RecognitionSession(
                    session_id=session_id,
                    context=context,
                    configuration=configuration,
                    started_at=(
                        str(row.get("started_at"))
                        if row.get("started_at") is not None
                        else None
                    ),
                )
                with self._lock:
                    self._active_session = session
                    self._trial_count = 0
                LOGGER.info(
                    "Recognition session created; session_id=%s type=%s",
                    session_id,
                    context.session_type,
                )
                on_success(session)
            except SessionPersistenceError as error:
                on_error(str(error))
            except Exception as error:
                LOGGER.error(
                    "Unexpected session creation failure; exception_type=%s",
                    type(error).__name__,
                )
                on_error("Recognition session could not be created.")
            finally:
                with self._lock:
                    self._creating = False
                    self._operation_count -= 1

        self._executor.submit(operation)
        return True

    def record_trial_async(
        self,
        trial: RecognitionTrialInput,
        snapshot: RecognitionTrialSnapshot,
        *,
        on_success: SuccessCallback,
        on_error: ErrorCallback,
    ) -> bool:
        with self._lock:
            session = self._active_session
            if (
                self._closed
                or session is None
                or session.context.session_type != CAMERA_TEST_SESSION
                or self._trial_pending
                or self._ending
            ):
                return False
            trial_number = self._trial_count + 1
            self._trial_pending = True
            self._operation_count += 1

        payload = {
            "session_id": session.session_id,
            "trial_number": trial_number,
            "expected_user_id": trial.expected_user_id,
            "distance_m": trial.distance_m,
            "detected": snapshot.detected,
            "detector_confidence": snapshot.detector_confidence,
            "predicted_user_id": snapshot.predicted_user_id,
            "predicted_name": snapshot.predicted_name,
            "similarity": snapshot.similarity,
            "threshold": snapshot.threshold,
            "recognized": snapshot.recognized,
            "detection_time_ms": snapshot.detection_time_ms,
            "recognition_time_ms": snapshot.recognition_time_ms,
            "notes": trial.notes,
        }
        if snapshot.diagnostics is not None:
            payload["diagnostics"] = snapshot.diagnostics

        def operation() -> None:
            try:
                self._repository.insert_trial(payload)
                with self._lock:
                    self._trial_count = trial_number
                LOGGER.info(
                    "Recognition test trial recorded; session_id=%s trial=%d",
                    session.session_id,
                    trial_number,
                )
                on_success(trial_number)
            except SessionPersistenceError as error:
                on_error(str(error))
            except Exception as error:
                LOGGER.error(
                    "Unexpected trial persistence failure; exception_type=%s",
                    type(error).__name__,
                )
                on_error(
                    "Trial recognition succeeded but log could not be saved."
                )
            finally:
                with self._lock:
                    self._trial_pending = False
                    self._operation_count -= 1

        self._executor.submit(operation)
        return True

    def load_expected_users_async(
        self,
        *,
        on_success: SuccessCallback,
        on_error: ErrorCallback,
    ) -> bool:
        with self._lock:
            if self._closed or self._users_loading:
                return False
            self._users_loading = True
            self._operation_count += 1

        def operation() -> None:
            try:
                users = self._repository.load_users()
                LOGGER.info(
                    "Expected-user options loaded; count=%d",
                    len(users),
                )
                on_success(users)
            except SessionPersistenceError as error:
                on_error(str(error))
            finally:
                with self._lock:
                    self._users_loading = False
                    self._operation_count -= 1

        self._executor.submit(operation)
        return True

    def update_camera_metadata_async(
        self,
        *,
        camera_resolution: str | None,
        camera_fps: float | None,
    ) -> bool:
        with self._lock:
            session = self._active_session
            if self._closed or session is None or self._ending:
                return False
            self._operation_count += 1

        def operation() -> None:
            try:
                self._repository.update_config_snapshot(
                    session.session_id,
                    config_snapshot=session.configuration.to_snapshot(
                        camera_resolution=camera_resolution,
                        camera_fps=camera_fps,
                    ),
                )
                LOGGER.info(
                    "Recognition runtime configuration saved; session_id=%s",
                    session.session_id,
                )
            except SessionPersistenceError as error:
                LOGGER.warning("%s", error)
            finally:
                with self._lock:
                    self._operation_count -= 1

        self._executor.submit(operation)
        return True

    def end_active_session_async(
        self,
        *,
        on_success: SuccessCallback | None = None,
        on_error: ErrorCallback | None = None,
    ) -> bool:
        with self._lock:
            session = self._active_session
            if self._closed or session is None or self._ending:
                return False
            self._ending = True
            self._operation_count += 1

        def operation() -> None:
            try:
                ended_at = self._repository.end_session(session.session_id)
                with self._lock:
                    if self._active_session == session:
                        self._active_session = None
                        self._trial_count = 0
                LOGGER.info(
                    "Recognition session ended; session_id=%s",
                    session.session_id,
                )
                if on_success is not None:
                    on_success(ended_at)
            except SessionPersistenceError as error:
                if on_error is not None:
                    on_error(str(error))
            finally:
                with self._lock:
                    self._ending = False
                    self._operation_count -= 1

        self._executor.submit(operation)
        return True

    def shutdown(self) -> None:
        with self._lock:
            self._closed = True
        self._executor.shutdown(wait=True, cancel_futures=False)
