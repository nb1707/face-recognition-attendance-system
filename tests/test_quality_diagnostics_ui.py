from pathlib import Path
import sys
import unittest
from types import SimpleNamespace

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from desktop.views.recognition_view import (
    format_matcher_diagnostics,
    format_quality_diagnostics,
)


def diagnostics(*, valid=True, reason=None, passed=True, reasons=()):
    return {
        "quality_measurements": {
            "valid": valid,
            "reason": reason,
            "width": 156,
            "height": 184,
            "area_ratio": .1875,
            "detector_confidence": .91,
            "sharpness": 14.8,
            "brightness": 103.2,
            "nose_offset": .08,
            "roll_degrees": 3.4,
        },
        "quality_decision": {
            "passed": passed,
            "quality_score": .58,
            "reasons": reasons,
            "components": {},
        },
    }


class QualityDiagnosticsFormatterTests(unittest.TestCase):
    def test_displays_multiple_rejection_reasons(self):
        text = format_quality_diagnostics(diagnostics(
            passed=False,
            reasons=("blurred_face", "poor_lighting"),
        ))

        self.assertIn(
            "Rejection reasons: blurred_face, poor_lighting",
            text,
        )
        self.assertIn("Quality passed: No", text)

    def test_displays_valid_measurements(self):
        text = format_quality_diagnostics(diagnostics())

        expected_lines = (
            "Face size: 156 × 184 px",
            "Area ratio: 0.1875",
            "Detector confidence: 0.91",
            "Sharpness: 14.8",
            "Brightness: 103.2",
            "Nose offset: 0.08",
            "Roll: 3.4 degrees",
            "Quality score: 0.58",
            "Quality passed: Yes",
            "Rejection reasons: None",
        )
        for line in expected_lines:
            self.assertIn(line, text)

    def test_invalid_measurement_displays_exact_reason(self):
        text = format_quality_diagnostics(diagnostics(
            valid=False,
            reason="invalid_landmarks",
            passed=False,
            reasons=("invalid_landmarks",),
        ))

        self.assertEqual(
            text,
            "Quality Diagnostics\n"
            "Measurement status: invalid\n"
            "Reason: invalid_landmarks",
        )


class MatcherDiagnosticsFormatterTests(unittest.TestCase):
    def test_displays_candidates_thresholds_and_gate_failure(self):
        value = {
            "matcher_diagnostics": {
                "top1": {
                    "user_id": "enroll_test_001",
                    "full_name": "Bo",
                    "score": .8017,
                    "winning_pose": "front",
                    "winning_prototype_id": "front_02",
                },
                "top2": {
                    "user_id": "enroll_test_002",
                    "full_name": "Bo",
                    "score": .7848,
                    "winning_pose": "front",
                    "winning_prototype_id": "front_01",
                },
                "margin": .0169,
                "similarity_threshold": .40,
                "margin_threshold": .05,
                "similarity_gate_passed": True,
                "margin_gate_passed": False,
                "gate_a_passed": False,
                "gate_a_reason": "ambiguous_identity",
            },
        }

        text = format_matcher_diagnostics(value)

        expected_lines = (
            "Top1: enroll_test_001 — Bo — 0.8017 — front_02",
            "Top1 pose: front",
            "Top2: enroll_test_002 — Bo — 0.7848 — front_01",
            "Top2 pose: front",
            "Margin: 0.0169",
            "Similarity threshold T: 0.40",
            "Margin threshold M: 0.05",
            "Similarity Gate: PASS",
            "Margin Gate: FAIL",
            "Gate A: FAIL — ambiguous_identity",
        )
        for line in expected_lines:
            self.assertIn(line, text)

    def test_unaccepted_candidate_remains_unknown_in_recognition_view(self):
        from desktop.app import FaceAttendanceDesktopApp

        class Variable:
            def __init__(self):
                self.value = None

            def set(self, value):
                self.value = value

        class View:
            def __init__(self):
                self.identity = Variable()
                self.user_id = Variable()
                self.role = Variable()
                self.similarity = Variable()
                self.recognition_status = Variable()
                self.quality = None
                self.matcher = None

            def set_quality_diagnostics(self, value):
                self.quality = value

            def set_matcher_diagnostics(self, value):
                self.matcher = format_matcher_diagnostics(value)

        diagnostics = {
            "matcher_diagnostics": {
                "top1": {
                    "user_id": "candidate",
                    "full_name": "Candidate Name",
                    "score": .80,
                    "winning_pose": "front",
                    "winning_prototype_id": "front_02",
                },
                "top2": {},
                "similarity_threshold": .40,
                "margin_threshold": .05,
                "similarity_gate_passed": True,
                "margin_gate_passed": False,
                "gate_a_passed": False,
                "gate_a_reason": "ambiguous_identity",
            },
        }
        face = SimpleNamespace(
            recognized=False,
            similarity=.80,
            diagnostics=diagnostics,
            full_name=None,
            user_id=None,
            role=None,
            failure_reason="ambiguous_identity",
            margin=.01,
            track_id=1,
            valid_observations=0,
        )
        view = View()

        FaceAttendanceDesktopApp._update_recognition_details(
            view,
            SimpleNamespace(faces=(face,), threshold=.40),
        )

        self.assertEqual(view.identity.value, "Identity: Unknown")
        self.assertEqual(view.user_id.value, "User ID: --")
        self.assertIn("Top1: candidate", view.matcher)
        self.assertIn("Gate A: FAIL — ambiguous_identity", view.matcher)


if __name__ == "__main__":
    unittest.main()
