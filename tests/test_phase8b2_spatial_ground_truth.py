"""Phase 8B.2 manual spatial ground-truth tooling tests; no inference."""
import ast
import json
from pathlib import Path
import sys
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/'backend'))

from analysis.spatial_ground_truth import (MetricAccumulator, aggregate_labelled,
    assign_observation, bbox_center_normalized, candidate_metrics, discover_replays,
    validate_annotations)


def region(region_id, user, distance, x1, y1, x2, y2):
    return dict(region_id=region_id, expected_user_id=user, distance_group=distance,
                x1=x1, y1=y1, x2=x2, y2=y2)


def clip_annotation():
    return dict(frame_width=100, frame_height=100,
        fixed_regions=[region('seat-a', 'a', 'near', 0, 0, .5, .5),
                       region('seat-b', 'b', 'far', .5, .5, 1, 1)],
        moving_segments=[], conditions=[])


def observation(x=10, y=10, size=10, user='a', score=.7, margin=.2,
                quality=True, sharpness=30, frame=10):
    return dict(frame_sequence=frame, bbox=[x, y, x+size, y+size],
        valid_observation=True, alignment_succeeded=True, quality_passed=quality,
        quality_reasons=[] if quality else ['blurred_face'], area_ratio=.01,
        detector_confidence=.9, sharpness=sharpness, brightness=100,
        nose_offset=.1, roll_degrees=2, top1_user_id=user, top1_score=score,
        top2_user_id='other', top2_score=.2, margin=margin,
        shadow_match_evaluated=False, shadow_top1_score=None, shadow_margin=None)


def item(record, expected='a', distance='near', clip='clip_01', lighting='normal'):
    return dict(clip_id=clip, lighting=lighting, record=record,
                assignment=dict(status='assigned', expected_user_id=expected,
                                distance_group=distance, condition=None))


QUALITY = dict(min_size=32, min_sharpness=20, min_area_ratio=0,
    min_detector=.5, min_brightness=35, max_brightness=225,
    max_nose_offset=.65, max_roll_degrees=40)


class AssignmentTests(unittest.TestCase):
    def test_fixed_region_assignment(self):
        result = assign_observation(observation(), clip_annotation())
        self.assertEqual((result['status'], result['expected_user_id'], result['distance_group']),
                         ('assigned', 'a', 'near'))

    def test_moving_region_takes_precedence(self):
        annotation = clip_annotation()
        annotation['moving_segments'] = [dict(segment_id='move-c', user_id='c',
            distance_group='moving', frame_start=1, frame_end=20,
            region=dict(x1=0, y1=0, x2=.5, y2=.5))]
        result = assign_observation(observation(frame=10), annotation)
        self.assertEqual((result['expected_user_id'], result['annotation_source']),
                         ('c', 'moving_segment'))

    def test_no_region_is_unassigned(self):
        annotation = clip_annotation()
        annotation['fixed_regions'] = []
        self.assertEqual(assign_observation(observation(), annotation)['status'], 'unassigned')

    def test_overlapping_fixed_regions_are_ambiguous(self):
        annotation = clip_annotation()
        annotation['fixed_regions'].append(region('seat-c', 'c', 'medium', 0, 0, .4, .4))
        result = assign_observation(observation(), annotation)
        self.assertEqual(result['status'], 'ambiguous')
        self.assertCountEqual(result['annotation_ids'], ['seat-a', 'seat-c'])

    def test_normalized_bbox_center(self):
        self.assertEqual(bbox_center_normalized(
            dict(bbox=[10, 20, 30, 40]), 100, 200), (.2, .15))


class MatcherMetricTests(unittest.TestCase):
    def metric(self, record, expected='a'):
        metric = MetricAccumulator()
        metric.add(item(record, expected))
        return metric.summary()

    def test_correct_top1(self):
        result = self.metric(observation(user='a'))
        self.assertEqual((result['predicted_top1_correct'], result.get('predicted_top1_wrong', 0)), (1, 0))

    def test_wrong_top1(self):
        result = self.metric(observation(user='b'))
        self.assertEqual(result['predicted_top1_wrong'], 1)

    def test_correct_gate_a(self):
        result = self.metric(observation(user='a', score=.6, margin=.1))
        self.assertEqual((result['observation_level_gate_a_pass'], result['correct_gate_a']), (1, 1))

    def test_wrong_gate_a(self):
        result = self.metric(observation(user='b', score=.6, margin=.1))
        self.assertEqual((result['observation_level_gate_a_pass'], result['wrong_gate_a']), (1, 1))

    def test_candidate_quality_policy(self):
        row = observation(size=30, sharpness=15, quality=False)
        results = candidate_metrics([item(row)], QUALITY, policies=((32, 20), (28, 12)))
        self.assertEqual(results[0]['breakdown']['overall']['all'].get('candidate_quality_pass', 0), 0)
        self.assertEqual(results[1]['breakdown']['overall']['all']['candidate_quality_pass'], 1)


class AggregationTests(unittest.TestCase):
    def test_distance_aggregation(self):
        result = aggregate_labelled([item(observation(), distance='near'),
                                     item(observation(), distance='far')])
        self.assertEqual(result['distance']['near']['assigned_observations'], 1)
        self.assertEqual(result['distance']['far']['assigned_observations'], 1)

    def test_clip_aggregation(self):
        result = aggregate_labelled([item(observation(), clip='clip-a'),
                                     item(observation(), clip='clip-b')])
        self.assertEqual(set(result['clip']), {'clip-a', 'clip-b'})

    def test_unassigned_is_excluded_from_accuracy(self):
        assignment = assign_observation(observation(), dict(frame_width=100, frame_height=100,
            fixed_regions=[], moving_segments=[], conditions=[]))
        labelled = [] if assignment['status'] != 'assigned' else [item(observation())]
        self.assertEqual(aggregate_labelled(labelled)['identity'], {})

    def test_ambiguous_is_excluded_from_accuracy(self):
        annotation = clip_annotation()
        annotation['fixed_regions'].append(region('overlap', 'c', 'medium', 0, 0, .5, .5))
        assignment = assign_observation(observation(), annotation)
        labelled = [] if assignment['status'] != 'assigned' else [item(observation())]
        self.assertEqual(assignment['status'], 'ambiguous')
        self.assertEqual(aggregate_labelled(labelled)['identity'], {})


class SafetyAndValidationTests(unittest.TestCase):
    def imported_modules(self):
        paths = [ROOT/'backend/analysis/spatial_ground_truth.py',
                 ROOT/'backend/scripts/analyze_phase8b2_identity_distance.py']
        imported = set()
        for path in paths:
            tree = ast.parse(path.read_text(encoding='utf-8'))
            for node in ast.walk(tree):
                if isinstance(node, ast.Import):
                    imported.update(alias.name.split('.')[0] for alias in node.names)
                elif isinstance(node, ast.ImportFrom) and node.module:
                    imported.add(node.module.split('.')[0])
        return imported

    def test_no_adaface_or_model_loading(self):
        self.assertTrue({'torch', 'adaface_model', 'recognition', 'insightface'}.isdisjoint(
            self.imported_modules()))

    def test_no_detector_or_video_loading(self):
        self.assertTrue({'cv2', 'face_detector', 'imageio', 'av'}.isdisjoint(
            self.imported_modules()))

    def test_no_gate_b_simulation(self):
        result = candidate_metrics([item(observation())], QUALITY, policies=((32, 20),))
        serialized = json.dumps(result).lower()
        self.assertNotIn('gate_b', serialized)
        self.assertNotIn('confirm', serialized)

    def test_incomplete_template_fails_annotation_validation(self):
        metadata = {'clip': {'lighting': 'normal', 'near': ['a']}}
        annotations = {'schema_version': 1, 'complete': False, 'clips': {'clip': {
            'frame_width': None, 'frame_height': None,
            'fixed_regions': [region('a', 'a', 'near', None, None, None, None)],
            'moving_segments': [], 'conditions': []}}}
        result = validate_annotations(metadata, annotations)
        self.assertFalse(result['complete'])
        self.assertTrue(result['errors'])

    def test_gallery_mismatch_fails_closed(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)/'run'
            root.mkdir()
            (root/'replay_summary.json').write_text(json.dumps(dict(
                gallery_generation='wrong', shadow_matcher=True,
                videos=[dict(video='clip.mp4')])), encoding='utf-8')
            with self.assertRaises(ValueError):
                discover_replays(directory, {'clip': {}}, expected_gallery='expected')


if __name__ == '__main__':
    unittest.main()
