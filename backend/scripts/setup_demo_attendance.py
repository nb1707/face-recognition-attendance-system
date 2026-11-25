"""Prepare and validate local attendance demo data without running recognition."""

from __future__ import annotations

import argparse
from dataclasses import dataclass
import json
from pathlib import Path
import sqlite3
import sys
from typing import Any, Callable, Sequence
from urllib.error import HTTPError, URLError
from urllib.parse import quote
from urllib.request import urlopen


BACKEND_DIR = Path(__file__).resolve().parents[1]
if str(BACKEND_DIR) not in sys.path:
    sys.path.insert(0, str(BACKEND_DIR))

from attendance.database import DEFAULT_DATABASE_PATH  # noqa: E402
from attendance.import_enrollment_users import (  # noqa: E402
    DEFAULT_USERS_PATH,
    import_enrollment_users,
)
from attendance.repository import AttendanceRepository  # noqa: E402
from recognition.adaface_loader import local_model_provenance  # noqa: E402
from recognition.local_gallery import (  # noqa: E402
    DEFAULT_GALLERY,
    Gallery,
    LocalGalleryProvider,
)


DEFAULT_CONFIG_PATH = BACKEND_DIR / "config" / "demo_classes.json"


class DemoSetupError(RuntimeError):
    """A clear, expected setup or readiness failure."""


@dataclass(frozen=True)
class DemoClassConfig:
    class_id: str
    class_code: str
    class_name: str
    student_ids: tuple[str, ...]
    duplicate_student_ids: tuple[str, ...] = ()


@dataclass(frozen=True)
class SetupResult:
    output: str
    ready: bool
    warning_count: int
    error_count: int
    api_ready: bool | None


def _required_text(value: Any, field: str) -> str:
    cleaned = value.strip() if isinstance(value, str) else ""
    if not cleaned:
        raise DemoSetupError(f"{field} must be a non-empty string")
    return cleaned


def load_enrollment_users(path: str | Path) -> dict[str, dict[str, str]]:
    source = Path(path)
    try:
        data = json.loads(source.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise DemoSetupError(f"Cannot load enrollment users from {source}: {error}") from error
    users = data.get("users") if isinstance(data, dict) else None
    if not isinstance(users, dict):
        raise DemoSetupError("Enrollment users.json must contain an object named 'users'")
    result: dict[str, dict[str, str]] = {}
    for key, raw in users.items():
        if not isinstance(raw, dict):
            raise DemoSetupError(f"Invalid enrollment user metadata for {key}")
        student_id = str(raw.get("user_id") or key).strip()
        if student_id != str(key):
            raise DemoSetupError(f"Enrollment user key and user_id differ for {key}")
        result[student_id] = {
            "full_name": str(raw.get("full_name") or student_id),
            "role": str(raw.get("role") or "student"),
        }
    return result


def format_available_users(users: dict[str, dict[str, str]]) -> str:
    rows = [(student_id, item["full_name"], item["role"]) for student_id, item in sorted(users.items())]
    id_width = max([len("student_id"), *(len(row[0]) for row in rows)])
    name_width = max([len("full_name"), *(len(row[1]) for row in rows)])
    lines = ["AVAILABLE ENROLLED USERS", "=" * 50]
    lines.append(f"{'student_id':<{id_width}}  {'full_name':<{name_width}}  role")
    lines.extend(f"{student_id:<{id_width}}  {name:<{name_width}}  {role}" for student_id, name, role in rows)
    lines.append(f"TOTAL_USERS = {len(rows)}")
    return "\n".join(lines)


def load_demo_config(path: str | Path) -> tuple[DemoClassConfig, ...]:
    source = Path(path)
    try:
        data = json.loads(source.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise DemoSetupError(f"Cannot load demo config from {source}: {error}") from error
    raw_classes = data.get("classes") if isinstance(data, dict) else None
    if not isinstance(raw_classes, list) or not raw_classes:
        raise DemoSetupError("Demo config must contain a non-empty 'classes' array")
    classes: list[DemoClassConfig] = []
    seen_ids: set[str] = set()
    seen_codes: set[str] = set()
    for index, raw in enumerate(raw_classes):
        if not isinstance(raw, dict):
            raise DemoSetupError(f"classes[{index}] must be an object")
        prefix = f"classes[{index}]"
        class_id = _required_text(raw.get("class_id"), f"{prefix}.class_id")
        class_code = _required_text(raw.get("class_code"), f"{prefix}.class_code")
        class_name = _required_text(raw.get("class_name"), f"{prefix}.class_name")
        if class_id in seen_ids:
            raise DemoSetupError(f"Duplicate class_id in config: {class_id}")
        if class_code in seen_codes:
            raise DemoSetupError(f"Duplicate class_code in config: {class_code}")
        raw_students = raw.get("student_ids")
        if not isinstance(raw_students, list) or not raw_students:
            raise DemoSetupError(f"Class {class_id} has an empty roster")
        students: list[str] = []
        duplicates: list[str] = []
        seen_students: set[str] = set()
        for student_index, value in enumerate(raw_students):
            student_id = _required_text(value, f"{prefix}.student_ids[{student_index}]")
            if student_id in seen_students:
                if student_id not in duplicates:
                    duplicates.append(student_id)
                continue
            students.append(student_id)
            seen_students.add(student_id)
        classes.append(DemoClassConfig(class_id, class_code, class_name, tuple(students), tuple(duplicates)))
        seen_ids.add(class_id)
        seen_codes.add(class_code)
    return tuple(classes)


def load_gallery_without_inference(path: str | Path) -> Gallery:
    gallery_path = Path(path).expanduser().resolve()
    try:
        # This hashes the installed checkpoint and reads preprocessing constants;
        # it does not instantiate the model or perform AdaFace inference.
        expected_provenance = local_model_provenance()
        return LocalGalleryProvider(gallery_path).load(expected_provenance)
    except (OSError, ValueError, KeyError, TypeError, ImportError) as error:
        raise DemoSetupError(f"Gallery validation failed: {error}") from error


def _check_api(base_url: str, classes: Sequence[DemoClassConfig]) -> tuple[bool, list[str]]:
    base = base_url.rstrip("/")

    def get_json(path: str) -> Any:
        with urlopen(base + path, timeout=3.0) as response:  # noqa: S310 - user-selected local API
            if response.status != 200:
                raise DemoSetupError(f"HTTP {response.status} for {path}")
            return json.loads(response.read().decode("utf-8"))

    messages: list[str] = []
    try:
        available = get_json("/attendance/classes")
        available_ids = {item.get("class_id") for item in available if isinstance(item, dict)}
        for configured in classes:
            if configured.class_id not in available_ids:
                messages.append(f"API missing class {configured.class_id}")
                continue
            roster = get_json(f"/attendance/classes/{quote(configured.class_id, safe='')}/roster")
            roster_ids = {item.get("student_id") for item in roster if isinstance(item, dict)}
            missing = sorted(set(configured.student_ids) - roster_ids)
            if missing:
                messages.append(f"API roster {configured.class_id} missing: {', '.join(missing)}")
    except (HTTPError, URLError, TimeoutError, OSError, ValueError, json.JSONDecodeError, DemoSetupError) as error:
        messages.append(f"API check failed: {error}")
    return not messages, messages


def _class_label(index: int) -> str:
    return chr(ord("A") + index) if index < 26 else str(index + 1)


def setup_demo(
    *,
    config_path: str | Path = DEFAULT_CONFIG_PATH,
    database_path: str | Path = DEFAULT_DATABASE_PATH,
    users_path: str | Path = DEFAULT_USERS_PATH,
    gallery_path: str | Path = DEFAULT_GALLERY,
    expected_gallery_generation: str | None = None,
    replace_rosters: bool = False,
    check_api: bool = False,
    api_base_url: str = "http://127.0.0.1:8000",
    gallery_loader: Callable[[str | Path], Gallery] = load_gallery_without_inference,
) -> SetupResult:
    """Validate inputs, mutate only attendance setup tables, and report readiness."""
    classes = load_demo_config(config_path)
    users = load_enrollment_users(users_path)
    unknown = sorted({student_id for item in classes for student_id in item.student_ids} - users.keys())
    if unknown:
        raise DemoSetupError(f"Unknown configured student ID(s): {', '.join(unknown)}")
    gallery = gallery_loader(gallery_path)
    if expected_gallery_generation and gallery.generation != expected_gallery_generation:
        raise DemoSetupError(
            "Gallery generation mismatch: "
            f"expected {expected_gallery_generation}, found {gallery.generation}"
        )

    repository = AttendanceRepository(database_path)
    imported = import_enrollment_users(repository, users_path)
    for configured in classes:
        current = repository.get_class(configured.class_id)
        if current is None:
            repository.create_class(configured.class_id, configured.class_code, configured.class_name)
        else:
            repository.update_class(
                configured.class_id,
                class_code=configured.class_code,
                class_name=configured.class_name,
            )
            if not current.active:
                repository.set_class_active(configured.class_id, True)
        configured_ids = set(configured.student_ids)
        if replace_rosters:
            for member in repository.list_class_roster(configured.class_id, active_only=False):
                if member.active and member.student_id not in configured_ids:
                    repository.remove_class_member(configured.class_id, member.student_id)
        for student_id in configured.student_ids:
            repository.add_class_member(configured.class_id, student_id)

    gallery_ids = {identity.user_id for identity in gallery.identities}
    warning_count = sum(len(item.duplicate_student_ids) for item in classes)
    error_count = 0
    class_sections: list[str] = []
    all_classes_ready = True
    for index, configured in enumerate(classes):
        class_row = repository.get_class(configured.class_id)
        roster = repository.list_class_roster(configured.class_id)
        roster_by_id = {member.student_id: member for member in roster}
        missing_gallery = [student_id for student_id in configured.student_ids if student_id not in gallery_ids]
        inactive = [student_id for student_id in configured.student_ids if student_id not in roster_by_id]
        warning_count += len(missing_gallery)
        class_errors: list[str] = []
        if class_row is None or not class_row.active:
            class_errors.append("class does not exist or is inactive")
        if not roster:
            class_errors.append("active roster is empty")
        if not any(student_id in gallery_ids for student_id in roster_by_id):
            class_errors.append("no active roster student has a gallery identity")
        if inactive:
            class_errors.append(f"configured student(s) are administratively inactive: {', '.join(inactive)}")
        error_count += len(class_errors)
        all_classes_ready = all_classes_ready and not class_errors
        lines = [
            f"CLASS {_class_label(index)}",
            f"class_id: {configured.class_id}",
            f"Class: {configured.class_code}",
            f"class_name: {configured.class_name}",
            f"Roster count: {len(roster)}",
        ]
        for student_id in configured.student_ids:
            metadata = users[student_id]
            found = "YES" if student_id in gallery_ids else "NO"
            lines.append(f"{student_id}    {metadata['full_name']}    {metadata['role']}    gallery={found}")
        for duplicate in configured.duplicate_student_ids:
            lines.append(f"WARNING: duplicate student_id normalized: {duplicate}")
        for message in class_errors:
            lines.append(f"ERROR: {message}")
        lines.append(f"READY = {'YES' if not class_errors else 'NO'}")
        lines.append(f"WARNINGS = {len(missing_gallery) + len(configured.duplicate_student_ids)}")
        class_sections.append("\n".join(lines))

    api_ready: bool | None = None
    api_messages: list[str] = []
    if check_api:
        api_ready, api_messages = _check_api(api_base_url, classes)
        if not api_ready:
            error_count += len(api_messages)

    ready = all_classes_ready and (api_ready is not False)
    desktop_ready = ready
    lines = [
        "DEMO ATTENDANCE SETUP",
        "=" * 50,
        f"Database: {Path(database_path).resolve()}",
        f"Enrollment source users: {len(users)}",
        f"Imported/upserted students: {imported}",
        f"Replace rosters: {'YES' if replace_rosters else 'NO'}",
        f"Gallery generation: {gallery.generation}",
        f"Gallery identities: {len(gallery.identities)}",
        "Gallery provenance: LOADED",
        "",
        "\n\n".join(class_sections),
    ]
    if check_api:
        lines.extend(["", f"API_READY = {'YES' if api_ready else 'NO'}"])
        lines.extend(f"API_ERROR: {message}" for message in api_messages)
    lines.extend(
        [
            "",
            "=" * 50,
            f"DESKTOP_DATA_READY = {'YES' if desktop_ready else 'NO'}",
            f"READY_FOR_DEMO = {'YES' if ready else 'NO'}",
            f"TOTAL_CLASSES = {len(classes)}",
            f"TOTAL_WARNINGS = {warning_count}",
            f"TOTAL_ERRORS = {error_count}",
        ]
    )
    return SetupResult("\n".join(lines), ready, warning_count, error_count, api_ready)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--list-users", action="store_true", help="list enrolled users and exit")
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG_PATH)
    parser.add_argument("--database", type=Path, default=DEFAULT_DATABASE_PATH)
    parser.add_argument("--users", type=Path, default=DEFAULT_USERS_PATH)
    parser.add_argument("--gallery", type=Path, default=DEFAULT_GALLERY)
    parser.add_argument("--expected-gallery-generation")
    parser.add_argument("--replace-rosters", action="store_true")
    parser.add_argument("--check-api", action="store_true")
    parser.add_argument("--api-base-url", default="http://127.0.0.1:8000")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        if args.list_users:
            print(format_available_users(load_enrollment_users(args.users)))
            return 0
        result = setup_demo(
            config_path=args.config,
            database_path=args.database,
            users_path=args.users,
            gallery_path=args.gallery,
            expected_gallery_generation=args.expected_gallery_generation,
            replace_rosters=args.replace_rosters,
            check_api=args.check_api,
            api_base_url=args.api_base_url,
        )
        print(result.output)
        return 0 if result.ready else 1
    except (DemoSetupError, sqlite3.Error, ValueError) as error:
        print(f"DEMO SETUP FAILED: {error}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
