"""Small read/write adapter for recognition-session Supabase tables."""

from __future__ import annotations

import logging
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

LOGGER = logging.getLogger(__name__)
_BACKEND_ENV_FILE = Path(__file__).resolve().parents[2] / "backend" / ".env"


class SessionPersistenceError(RuntimeError):
    """Sanitized persistence failure suitable for the desktop status area."""


class RecognitionSessionRepository:
    """Persist session metadata without importing the FastAPI application."""

    def __init__(self) -> None:
        self._client: Any | None = None

    def create_session(self, payload: dict[str, Any]) -> dict[str, Any]:
        try:
            response = (
                self._get_client()
                .table("recognition_sessions")
                .insert(payload)
                .execute()
            )
            if not response.data:
                raise RuntimeError("Session insert returned no row.")
            return dict(response.data[0])
        except SessionPersistenceError:
            raise
        except Exception as error:
            self._log_failure("create recognition session", error)
            raise SessionPersistenceError(
                "Recognition session could not be created. Apply the Phase "
                "4.5 SQL migration and verify Supabase access."
            ) from error

    def update_config_snapshot(
        self,
        session_id: str,
        *,
        config_snapshot: dict[str, Any],
    ) -> None:
        try:
            response = (
                self._get_client()
                .table("recognition_sessions")
                .update({"config_snapshot": config_snapshot})
                .eq("id", session_id)
                .execute()
            )
            if response.data is not None and len(response.data) == 0:
                raise RuntimeError("Session metadata update matched no row.")
        except Exception as error:
            self._log_failure("update recognition configuration snapshot", error)
            raise SessionPersistenceError(
                "Runtime recognition configuration could not be saved."
            ) from error

    def load_users(self) -> tuple[tuple[str, str], ...]:
        try:
            response = (
                self._get_client()
                .table("users")
                .select("user_id,full_name")
                .execute()
            )
            users = []
            for row in response.data or []:
                user_id = str(row.get("user_id") or "").strip()
                if not user_id:
                    continue
                full_name = str(row.get("full_name") or user_id).strip()
                users.append((user_id, full_name))
            return tuple(sorted(users, key=lambda user: user[0].lower()))
        except Exception as error:
            self._log_failure("load expected-user options", error)
            raise SessionPersistenceError(
                "Registered users could not be loaded for Expected User."
            ) from error

    def insert_trial(self, payload: dict[str, Any]) -> dict[str, Any]:
        try:
            response = (
                self._get_client()
                .table("recognition_test_trials")
                .insert(payload)
                .execute()
            )
            if not response.data:
                raise RuntimeError("Trial insert returned no row.")
            return dict(response.data[0])
        except Exception as error:
            self._log_failure("create recognition test trial", error)
            raise SessionPersistenceError(
                "Trial recognition succeeded but log could not be saved."
            ) from error

    def end_session(self, session_id: str) -> str:
        ended_at = datetime.now(timezone.utc).isoformat()
        try:
            self._get_client().table("recognition_sessions").update(
                {"ended_at": ended_at}
            ).eq("id", session_id).is_("ended_at", "null").execute()
            return ended_at
        except Exception as error:
            self._log_failure("end recognition session", error)
            raise SessionPersistenceError(
                "Recognition session end time could not be saved."
            ) from error

    def _get_client(self):
        if self._client is not None:
            return self._client
        try:
            from dotenv import load_dotenv
            from supabase import create_client
        except ImportError as error:
            raise SessionPersistenceError(
                "Supabase session logging support is unavailable."
            ) from error

        load_dotenv(dotenv_path=_BACKEND_ENV_FILE, override=False)
        supabase_url = os.getenv("SUPABASE_URL", "").strip()
        supabase_key = os.getenv("SUPABASE_KEY", "").strip()
        if not supabase_url or not supabase_key:
            raise SessionPersistenceError(
                "Supabase session logging configuration is incomplete."
            )
        try:
            self._client = create_client(supabase_url, supabase_key)
        except Exception as error:
            self._log_failure("initialize Supabase session client", error)
            raise SessionPersistenceError(
                "Supabase session logging connection could not be created."
            ) from error
        return self._client

    @staticmethod
    def _log_failure(operation: str, error: Exception) -> None:
        LOGGER.error(
            "Could not %s; exception_type=%s",
            operation,
            type(error).__name__,
        )
