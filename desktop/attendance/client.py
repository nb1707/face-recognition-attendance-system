"""Small JSON/HTTP boundary for the local FastAPI attendance routes."""

from __future__ import annotations

import json
import os
import socket
from typing import Any, Callable
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen


class AttendanceClientError(RuntimeError):
    """Sanitized API failure suitable for display in the desktop UI."""

    def __init__(self, message: str, *, status_code: int | None = None):
        super().__init__(message)
        self.status_code = status_code


class AttendanceApiClient:
    def __init__(
        self,
        base_url: str | None = None,
        *,
        timeout_seconds: float = 5.0,
        opener: Callable[..., Any] = urlopen,
    ) -> None:
        configured = base_url or os.getenv(
            "ATTENDANCE_API_BASE_URL", "http://127.0.0.1:8000"
        )
        self.base_url = configured.rstrip("/")
        self.timeout_seconds = float(timeout_seconds)
        if self.timeout_seconds <= 0:
            raise ValueError("Attendance API timeout must be positive")
        self._opener = opener

    def list_classes(self) -> list[dict[str, Any]]:
        return self._request("GET", "/attendance/classes")

    def get_class_roster(self, class_id: str) -> list[dict[str, Any]]:
        return self._request("GET", f"/attendance/classes/{class_id}/roster")

    def create_session(
        self,
        *,
        class_id: str,
        gallery_generation: str,
        recognition_config: dict[str, Any],
        camera_source: str | None = None,
        operator_name: str | None = None,
    ) -> dict[str, Any]:
        return self._request(
            "POST",
            "/attendance/sessions",
            {
                "class_id": class_id,
                "gallery_generation": gallery_generation,
                "recognition_config": recognition_config,
                "camera_source": camera_source,
                "operator_name": operator_name,
            },
        )

    def get_session(self, session_id: str) -> dict[str, Any]:
        return self._request("GET", f"/attendance/sessions/{session_id}")

    def refresh_suggestions(self, session_id: str) -> list[dict[str, Any]]:
        return self._request(
            "POST", f"/attendance/sessions/{session_id}/refresh-suggestions"
        )

    def list_records(self, session_id: str) -> list[dict[str, Any]]:
        return self._request("GET", f"/attendance/sessions/{session_id}/records")

    def end_session(self, session_id: str) -> dict[str, Any]:
        return self._request("POST", f"/attendance/sessions/{session_id}/end")

    def review_student(
        self,
        session_id: str,
        student_id: str,
        final_status: str,
        actor: str,
        reason: str | None = None,
    ) -> dict[str, Any]:
        return self._request(
            "POST",
            f"/attendance/sessions/{session_id}/students/{student_id}/review",
            {"final_status": final_status, "actor": actor, "reason": reason},
        )

    def finalize_session(self, session_id: str, actor: str) -> dict[str, Any]:
        return self._request(
            "POST",
            f"/attendance/sessions/{session_id}/finalize",
            {"actor": actor},
        )

    def get_audit(self, session_id: str) -> list[dict[str, Any]]:
        return self._request("GET", f"/attendance/sessions/{session_id}/audit")

    def _request(
        self,
        method: str,
        path: str,
        payload: dict[str, Any] | None = None,
    ) -> Any:
        body = None
        headers = {"Accept": "application/json"}
        if payload is not None:
            try:
                body = json.dumps(payload, allow_nan=False).encode("utf-8")
            except (TypeError, ValueError) as error:
                raise AttendanceClientError("Attendance request data is invalid.") from error
            headers["Content-Type"] = "application/json"
        request = Request(
            f"{self.base_url}{path}", data=body, headers=headers, method=method
        )
        try:
            with self._opener(request, timeout=self.timeout_seconds) as response:
                raw = response.read()
        except HTTPError as error:
            detail = self._error_detail(error)
            raise AttendanceClientError(detail, status_code=error.code) from error
        except (URLError, TimeoutError, socket.timeout) as error:
            raise AttendanceClientError(
                "Attendance backend is unavailable or timed out."
            ) from error
        except OSError as error:
            raise AttendanceClientError("Attendance backend request failed.") from error
        try:
            return json.loads(raw.decode("utf-8")) if raw else None
        except (UnicodeDecodeError, json.JSONDecodeError) as error:
            raise AttendanceClientError(
                "Attendance backend returned an invalid response."
            ) from error

    @staticmethod
    def _error_detail(error: HTTPError) -> str:
        try:
            payload = json.loads(error.read().decode("utf-8"))
            detail = payload.get("detail") if isinstance(payload, dict) else None
            if isinstance(detail, str) and detail.strip():
                return detail.strip()
        except (OSError, UnicodeDecodeError, json.JSONDecodeError):
            pass
        return f"Attendance request failed with HTTP {error.code}."
