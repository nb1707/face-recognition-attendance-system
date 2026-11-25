"""Pure Phase 8B.1 persisted-diagnostic analysis tests."""
import ast
import csv
import json
from pathlib import Path
import sys
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'backend'))

from analysis.quality_sweep import (analyze, available_matcher,
    candidate_quality_passes, gate_a_would_pass, normalize_record, write_outputs)


QUALITY = dict(crop_size=112, min_size=32, min_area_ratio=0, min_detector=.5,
    min_sharpness=20, min_brightness=35, max_brightness=225,
    max_nose_offset=.65, max_roll_degrees=40, sharpness_reference=200,
    size_reference=112, brightness_target=128, component_weights=[1, 1, 1, 1, 1])


def record(size=32, sharpness=20, reasons=(), **changes):
    value = dict(source_video='fixture.mp4', bbox=[0, 0, size, size],
        face_width=float(size), face_height=float(size), area_ratio=.01,
        detector_confidence=.9, sharpness=float(sharpness), brightness=100.,
        nose_offset=.1, roll_degrees=2., valid_observation=True,
        alignment_succeeded=True, quality_passed=not reasons,
        quality_reasons=list(reasons), top1_user_id='production-user',
        top1_score=.6, top2_user_id='other', top2_score=.3, margin=.3,
        shadow_match_evaluated=True, shadow_top1_user_id='shadow-user',
        shadow_top1_score=.8, shadow_top2_user_id='other-shadow',
        shadow_top2_score=.2, shadow_margin=.6)
    value.update(changes)
    return value


def summary(persisted_quality_pass=1):
    return dict(gallery_generation='fixture-gallery', configuration=dict(
        quality=QUALITY, operating_point=dict(similarity_threshold=.4,
        margin_threshold=.05)), videos=[dict(video='fixture.mp4', sha256='abc',
        funnel=dict(quality_passed=persisted_quality_pass))])


class QualityPolicyTests(unittest.TestCase):
    def test_a_baseline_32_20_reproduces_fixture_admission(self):
        self.assertTrue(candidate_quality_passes(record(), 32, 20, QUALITY))

    def test_b_lower_size_admits_face_too_small(self):
        row = record(24, 25, ('face_too_small',))
        self.assertFalse(candidate_quality_passes(row, 32, 20, QUALITY))
        self.assertTrue(candidate_quality_passes(row, 24, 20, QUALITY))

    def test_c_lower_sharpness_admits_blurred_face(self):
        row = record(40, 12, ('blurred_face',))
        self.assertFalse(candidate_quality_passes(row, 32, 20, QUALITY))
        self.assertTrue(candidate_quality_passes(row, 32, 12, QUALITY))

    def test_d_lower_size_alone_does_not_bypass_sharpness(self):
        self.assertFalse(candidate_quality_passes(
            record(24, 8, ('face_too_small', 'blurred_face')), 24, 20, QUALITY))

    def test_e_lower_sharpness_alone_does_not_bypass_size(self):
        self.assertFalse(candidate_quality_passes(
            record(20, 20, ('face_too_small',)), 32, 8, QUALITY))

    def test_f_non_variable_failure_remains_rejected(self):
        row = record(40, 30, ('poor_lighting',), brightness=20.)
        self.assertFalse(candidate_quality_passes(row, 20, 5, QUALITY))


class MatcherTests(unittest.TestCase):
    def test_g_production_matcher_has_precedence(self):
        match = available_matcher(record())
        self.assertEqual((match['source'], match['top1_user_id']),
                         ('production', 'production-user'))

    def test_h_shadow_used_only_when_production_unavailable(self):
        match = available_matcher(record(top1_score=None, top2_score=None, margin=None))
        self.assertEqual((match['source'], match['top1_user_id']), ('shadow', 'shadow-user'))

    def test_i_matcher_unavailable_cannot_pass_gate_a(self):
        match = available_matcher(record(top1_score=None, shadow_match_evaluated=False,
            shadow_top1_score=None, shadow_margin=None))
        self.assertFalse(gate_a_would_pass(match, .4, .05))

    def test_j_gate_a_requires_similarity_and_margin(self):
        base = dict(source='shadow', top1_user_id='a', top2_user_id='b', top2_score=.2)
        self.assertFalse(gate_a_would_pass(dict(base, top1_score=.39, margin=.2), .4, .05))
        self.assertFalse(gate_a_would_pass(dict(base, top1_score=.6, margin=.04), .4, .05))
        self.assertTrue(gate_a_would_pass(dict(base, top1_score=.6, margin=.2), .4, .05))


class AnalysisContractTests(unittest.TestCase):
    def make_input(self, directory):
        path = Path(directory) / 'observations.jsonl'
        rows = [record(), record(24, 12, ('face_too_small', 'blurred_face'),
                top1_score=None, top2_score=None, margin=None)]
        path.write_text('\n'.join(json.dumps(row) for row in rows), encoding='utf-8')
        return path

    def test_k_no_gate_b_confirmation_is_calculated(self):
        with tempfile.TemporaryDirectory() as directory:
            path = self.make_input(directory)
            result = analyze(path, summary(), [20, 24, 28, 32], [5, 8, 12, 16, 20])
        self.assertFalse(result['metadata']['gate_b_simulated'])
        for row in result['candidates']:
            self.assertFalse(any('gate_b' in key or 'confirm' in key for key in row))

    def test_l_analysis_source_imports_no_inference_or_video_modules(self):
        paths = [ROOT/'backend/analysis/quality_sweep.py',
                 ROOT/'backend/scripts/analyze_phase8a_quality_sweep.py']
        imported = set()
        for path in paths:
            tree = ast.parse(path.read_text(encoding='utf-8'))
            for node in ast.walk(tree):
                if isinstance(node, ast.Import):
                    imported.update(alias.name.split('.')[0] for alias in node.names)
                elif isinstance(node, ast.ImportFrom) and node.module:
                    imported.add(node.module.split('.')[0])
        self.assertTrue({'cv2', 'torch', 'recognition', 'insightface'} .isdisjoint(imported))

    def test_m_csv_has_twenty_pairs_and_one_baseline(self):
        with tempfile.TemporaryDirectory() as directory:
            path = self.make_input(directory)
            result = analyze(path, summary(), [20, 24, 28, 32], [5, 8, 12, 16, 20])
            output = Path(directory)/'output'
            write_outputs(result, output)
            with (output/'quality_sweep.csv').open(encoding='utf-8') as source:
                rows = list(csv.DictReader(source))
        self.assertEqual(len(rows), 20)
        self.assertEqual(sum(row['baseline'] == 'True' for row in rows), 1)
        self.assertEqual(result['metadata']['offline_baseline_quality_pass'], 1)
        self.assertTrue(result['metadata']['baseline_reproduction_exact'])

    def test_shortlist_removes_more_permissive_identical_outcomes(self):
        with tempfile.TemporaryDirectory() as directory:
            path = self.make_input(directory)
            result = analyze(path, summary(), [20, 24, 28, 32], [5, 8, 12, 16, 20])
        signatures = [(row['candidate_quality_pass'], row['gate_a_would_pass'])
                      for row in result['development_shortlist']]
        self.assertEqual(len(signatures), len(set(signatures)))


if __name__ == '__main__':
    unittest.main()
