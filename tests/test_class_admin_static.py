from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def test_class_admin_page_has_required_local_admin_surfaces():
    html = (ROOT / "backend" / "static" / "class_admin.html").read_text(
        encoding="utf-8"
    )
    script = (ROOT / "backend" / "static" / "class_admin.js").read_text(
        encoding="utf-8"
    )

    for label in (
        "Class Administration",
        "Current Roster",
        "Available Students",
        "Create Class",
    ):
        assert label in html
    for endpoint in (
        '"/attendance/classes"',
        '"/attendance/students"',
        "/attendance/classes/${encodeURIComponent(item.class_id)}/roster",
    ):
        assert endpoint in script
    assert "fetch(" in script
    assert "supabase" not in script.lower()
