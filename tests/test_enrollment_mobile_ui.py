from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
STATIC = ROOT / "backend" / "static"


def test_mobile_pose_overlay_and_responsive_camera_rules_exist():
    html = (STATIC / "index.html").read_text(encoding="utf-8")
    css = (STATIC / "style.css").read_text(encoding="utf-8")
    script = (STATIC / "app.js").read_text(encoding="utf-8")

    for element_id in (
        "poseOverlay",
        "previewPoseLabel",
        "previewPoseProgress",
        "previewCaptureStatus",
        "previewInstruction",
        "previewGuidanceDetail",
    ):
        assert f'id="{element_id}"' in html
        assert f'"{element_id}"' in script

    assert "@media (max-width: 620px)" in css
    assert ".camera-stage .face-guide" in css
    assert "aspect-ratio: 3 / 4" in css
    assert ".preview-pose-overlay" in css
    assert "min-height: 48px" in css
    assert ".progress-panel { order: -1; }" in css


def test_mobile_polish_does_not_change_enrollment_contract_or_targets():
    script = (STATIC / "app.js").read_text(encoding="utf-8")

    assert 'const DEFAULT_ENROLLMENT_TARGET = 15;' in script
    assert 'api("/enrollments"' in script
    assert 'api(`/enrollments/${state.sessionId}/samples`' in script
    assert 'form.append("pose", pose)' in script
    assert 'state.session.next_recommended_pose' in script
