import hashlib
import json
from pathlib import Path
import sqlite3
import sys
from types import SimpleNamespace

import pytest


sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "backend"))

from attendance.repository import AttendanceRepository
from scripts.setup_demo_attendance import (
    DemoSetupError,
    load_demo_config,
    main,
    setup_demo,
)


def write_users(path, student_ids=("stu_001", "stu_002", "stu_003")):
    path.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "users": {
                    student_id: {
                        "user_id": student_id,
                        "full_name": f"Name {student_id}",
                        "role": "student",
                    }
                    for student_id in student_ids
                },
            },
            indent=2,
        ),
        encoding="utf-8",
    )


def write_config(path, class_a=("stu_001", "stu_002"), class_b=("stu_002",)):
    path.write_text(
        json.dumps(
            {
                "classes": [
                    {
                        "class_id": "demo-class-a",
                        "class_code": "COMP1682-A",
                        "class_name": "Demo A",
                        "student_ids": list(class_a),
                    },
                    {
                        "class_id": "demo-class-b",
                        "class_code": "COMP1682-B",
                        "class_name": "Demo B",
                        "student_ids": list(class_b),
                    },
                ]
            },
            indent=2,
        ),
        encoding="utf-8",
    )


def fake_gallery(*student_ids, generation="gallery-1"):
    return SimpleNamespace(
        generation=generation,
        provenance={"embedding_dimension": 2},
        identities=tuple(SimpleNamespace(user_id=value) for value in student_ids),
    )


@pytest.fixture
def demo_files(tmp_path):
    users = tmp_path / "users.json"
    config = tmp_path / "demo_classes.json"
    gallery = tmp_path / "current.json"
    database = tmp_path / "attendance.db"
    write_users(users)
    write_config(config)
    gallery.write_text('{"sentinel": true}\n', encoding="utf-8")
    return SimpleNamespace(users=users, config=config, gallery=gallery, database=database)


def run_setup(files, gallery=None, **kwargs):
    selected_gallery = gallery or fake_gallery("stu_001", "stu_002", "stu_003")
    return setup_demo(
        config_path=files.config,
        database_path=files.database,
        users_path=files.users,
        gallery_path=files.gallery,
        gallery_loader=lambda _: selected_gallery,
        **kwargs,
    )


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_list_users_is_read_only_and_sorted(demo_files, capsys):
    before = digest(demo_files.users)
    assert main(["--list-users", "--users", str(demo_files.users)]) == 0
    output = capsys.readouterr().out
    assert "AVAILABLE ENROLLED USERS" in output
    assert "student_id" in output and "full_name" in output and "role" in output
    assert output.index("stu_001") < output.index("stu_002") < output.index("stu_003")
    assert digest(demo_files.users) == before
    assert not demo_files.database.exists()


def test_two_classes_same_student_idempotent_and_no_sessions(demo_files):
    users_before = digest(demo_files.users)
    gallery_before = digest(demo_files.gallery)
    first = run_setup(demo_files)
    second = run_setup(demo_files)
    repository = AttendanceRepository(demo_files.database)

    assert first.ready and second.ready
    assert [row.student_id for row in repository.list_class_roster("demo-class-a")] == [
        "stu_001",
        "stu_002",
    ]
    assert [row.student_id for row in repository.list_class_roster("demo-class-b")] == [
        "stu_002"
    ]
    assert first.output == second.output
    with repository.database.connection() as connection:
        assert connection.execute("SELECT COUNT(*) FROM attendance_sessions").fetchone()[0] == 0
        assert connection.execute("SELECT COUNT(*) FROM session_roster").fetchone()[0] == 0
    assert digest(demo_files.users) == users_before
    assert digest(demo_files.gallery) == gallery_before


@pytest.mark.parametrize(
    ("class_a", "message"),
    [
        ((), "empty roster"),
        (("unknown",), "Unknown configured student ID"),
    ],
)
def test_invalid_rosters_fail_before_database_mutation(demo_files, class_a, message):
    write_config(demo_files.config, class_a=class_a)
    with pytest.raises(DemoSetupError, match=message):
        run_setup(demo_files)
    assert not demo_files.database.exists()


def test_duplicate_student_ids_are_normalized_deterministically(demo_files):
    write_config(demo_files.config, class_a=("stu_001", "stu_001", "stu_002", "stu_001"))
    parsed = load_demo_config(demo_files.config)
    assert parsed[0].student_ids == ("stu_001", "stu_002")
    assert parsed[0].duplicate_student_ids == ("stu_001",)
    result = run_setup(demo_files)
    assert "WARNING: duplicate student_id normalized: stu_001" in result.output
    assert result.warning_count == 1


def test_replace_rosters_soft_removes_unspecified_member(demo_files):
    run_setup(demo_files)
    write_config(demo_files.config, class_a=("stu_001",))
    run_setup(demo_files, replace_rosters=True)
    repository = AttendanceRepository(demo_files.database)
    assert [row.student_id for row in repository.list_class_roster("demo-class-a")] == ["stu_001"]
    assert repository.get_student("stu_002") is not None
    assert repository.is_class_member("demo-class-a", "stu_002", active_only=False)
    assert not repository.is_class_member("demo-class-a", "stu_002")


def test_non_replace_mode_preserves_extra_members(demo_files):
    run_setup(demo_files)
    write_config(demo_files.config, class_a=("stu_001",))
    run_setup(demo_files, replace_rosters=False)
    repository = AttendanceRepository(demo_files.database)
    assert [row.student_id for row in repository.list_class_roster("demo-class-a")] == [
        "stu_001",
        "stu_002",
    ]


def test_expected_gallery_mismatch_fails_closed(demo_files):
    with pytest.raises(DemoSetupError, match="Gallery generation mismatch"):
        run_setup(demo_files, expected_gallery_generation="other-generation")
    assert not demo_files.database.exists()


def test_missing_gallery_student_warns_but_class_with_a_match_is_ready(demo_files):
    write_config(demo_files.config, class_a=("stu_001", "stu_002"), class_b=("stu_001",))
    result = run_setup(demo_files, gallery=fake_gallery("stu_001"))
    assert result.ready
    assert "stu_001    Name stu_001    student    gallery=YES" in result.output
    assert "stu_002    Name stu_002    student    gallery=NO" in result.output
    assert result.warning_count == 1


def test_missing_gallery_student_is_reported_as_warning(demo_files):
    result = run_setup(demo_files, gallery=fake_gallery("stu_001"))
    assert not result.ready  # class B contains only stu_002, so it also has zero matches.
    assert result.warning_count == 2  # stu_002 is reported once in each class.
    assert result.output.count("stu_002    Name stu_002    student    gallery=NO") == 2


def test_zero_gallery_matches_for_class_fails_and_summary_is_deterministic(demo_files):
    gallery = fake_gallery("stu_001")
    first = run_setup(demo_files, gallery=gallery)
    second = run_setup(demo_files, gallery=gallery)
    assert not first.ready
    assert "ERROR: no active roster student has a gallery identity" in first.output
    assert "READY_FOR_DEMO = NO" in first.output
    assert "TOTAL_CLASSES = 2" in first.output
    assert first.output == second.output


def test_inactive_student_administrative_state_is_preserved(demo_files):
    run_setup(demo_files)
    repository = AttendanceRepository(demo_files.database)
    repository.set_student_active("stu_001", False)
    result = run_setup(demo_files)
    assert repository.get_student("stu_001").active is False
    assert not result.ready
    assert "administratively inactive: stu_001" in result.output
