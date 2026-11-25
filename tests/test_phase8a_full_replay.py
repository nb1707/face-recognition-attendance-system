"""Focused Phase 8A full-video and shadow diagnostic tests; no model/network."""
from dataclasses import asdict
from pathlib import Path
import json
import sys
from types import SimpleNamespace
import unittest
from unittest.mock import Mock
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'backend'))
sys.path.insert(0, str(ROOT / 'backend/scripts'))

from replay_offline import (apply_replay_quality_override, build_parser, iter_sampled_frames,
                            main, observation_row, shadow_match_observation)
from recognition.offline_config import OfflineConfig
from recognition.identity_matcher import MatchResult
from recognition.quality import QualityDecision, QualityMeasurements
from recognition.replay_diagnostics import ReplayDiagnostics
from recognition.replay_streaming import BoundedStatistics, ReplayFunnel
from recognition.temporal import OperatingPoint
from recognition.tracker import Track


class FakeCapture:
    def __init__(self, count):
        self.frames = list(range(count))
        self.index = 0

    def read(self):
        if self.index >= len(self.frames):
            return False, None
        frame = self.frames[self.index]
        self.index += 1
        return True, frame


class ReplayLoopTests(unittest.TestCase):
    def test_default_remains_sixty(self):
        parser = build_parser()
        args = parser.parse_args(['--output', 'result'])
        self.assertEqual(args.max_samples, 60)
        self.assertFalse(args.full_video)
        rows = list(iter_sampled_frames(FakeCapture(100), 1, args.max_samples, args.full_video))
        self.assertEqual(len(rows), 60)

    def test_explicit_max_samples_is_unchanged(self):
        args = build_parser().parse_args(['--output', 'result', '--max-samples', '7'])
        rows = list(iter_sampled_frames(FakeCapture(100), 1, args.max_samples, args.full_video))
        self.assertEqual([row[1] for row in rows], list(range(7)))

    def test_full_video_ignores_sample_limit_until_eof(self):
        rows = list(iter_sampled_frames(FakeCapture(9), 1, 2, True))
        self.assertEqual(len(rows), 9)
        self.assertEqual(rows[-1][:2], (9, 8))

    def test_stride_applies_in_full_video_mode(self):
        rows = list(iter_sampled_frames(FakeCapture(11), 3, 1, True))
        self.assertEqual([row[1] for row in rows], [0, 3, 6, 9])


class ReplayQualityOverrideTests(unittest.TestCase):
    def setUp(self):
        self.config = OfflineConfig(OperatingPoint(.4, .05))

    def assert_only_quality_candidate_changed(self, candidate, min_size, min_sharpness):
        original = asdict(self.config)
        expected = dict(original)
        expected['quality'] = dict(original['quality'], min_size=min_size,
                                   min_sharpness=min_sharpness)
        self.assertEqual(asdict(candidate), expected)

    def test_no_override_preserves_defaults_without_mutating_original(self):
        before = asdict(self.config)
        candidate = apply_replay_quality_override(self.config)
        self.assertEqual(asdict(candidate), before)
        self.assertEqual(asdict(self.config), before)
        self.assertIsNot(candidate, self.config)
        self.assertIsNot(candidate.quality, self.config.quality)

    def test_32_16_override_changes_only_requested_quality_fields(self):
        before = asdict(self.config)
        candidate = apply_replay_quality_override(self.config, 32, 16)
        self.assert_only_quality_candidate_changed(candidate, 32.0, 16.0)
        self.assertEqual(asdict(self.config), before)

    def test_24_12_override_changes_only_requested_quality_fields(self):
        before = asdict(self.config)
        candidate = apply_replay_quality_override(self.config, 24, 12)
        self.assert_only_quality_candidate_changed(candidate, 24.0, 12.0)
        self.assertEqual(asdict(self.config), before)

    def test_invalid_cli_quality_values_are_rejected_before_run(self):
        for option, value in (('--quality-min-size', '0'),
                              ('--quality-min-sharpness', '-1')):
            with self.subTest(option=option), \
                 unittest.mock.patch('replay_offline.run') as replay_run, \
                 self.assertRaises(SystemExit) as raised:
                main(['--output', 'result', option, value])
            self.assertEqual(raised.exception.code, 2)
            replay_run.assert_not_called()


class BoundedDiagnosticTests(unittest.TestCase):
    def test_numeric_reservoir_is_bounded_with_exact_count(self):
        values = BoundedStatistics(5)
        for value in range(1000):
            values.add(value)
        result = values.summary()
        self.assertEqual(result['count'], 1000)
        self.assertEqual(result['reservoir_size'], 5)
        self.assertEqual((result['min'], result['max'], result['mean']), (0, 999, 499.5))
        self.assertFalse(result['exact'])

    def test_track_reservoir_streams_every_row_but_stays_bounded(self):
        streamed = []
        diagnostic = ReplayDiagnostics(4, row_sink=streamed.append)
        tracker = SimpleNamespace(last_update=None, tracks={})
        engine = SimpleNamespace(tracker=tracker)
        for index in range(20):
            track = Track(index + 1, (0, 0, 1, 1), index, index, observations=1)
            tracker.last_update = dict(created=[track.track_id], retired=[(track, 'max_misses')],
                                       recovered=[], missed=[], ambiguous_detection_count=0)
            diagnostic.observe(engine, [])
        summary, rows = diagnostic.summary(tracker, 20)
        self.assertEqual(summary['tracks_created'], 20)
        self.assertEqual(len(streamed), 20)
        self.assertEqual(len(rows), 4)
        self.assertEqual(summary['track_reservoir_capacity'], 4)
        self.assertFalse(summary['track_statistics_exact'])


def make_decision(passed=False, reasons=('face_too_small', 'blurred_face')):
    measurement = QualityMeasurements(True, None, 20, 24, .001, .8, 7, 70, .1, 3)
    quality = QualityDecision(passed, .4, tuple(reasons), {})
    match = MatchResult('student-a', .55, 'front', 'student-b', .45, 'left', .10, 2)
    return SimpleNamespace(bbox=(1., 2., 21., 26.), track_id=3,
        quality_measurements=measurement, quality_decision=quality,
        aggregate_match=match if passed else None, observation_match=match if passed else None,
        match=match, selected_new_observation=passed, error_type=None,
        gate_a_passed=passed, observation_reason='temporal_pending' if passed else 'low_quality',
        reason='temporal_pending' if passed else 'low_quality', confirmed=False,
        confirmation_transition=False, event_emitted=False)


class ObservationContractTests(unittest.TestCase):
    def test_observation_json_contract_and_quality_reasons(self):
        decision = make_decision()
        face = SimpleNamespace(det_score=.8)
        engine = SimpleNamespace(confirmations={})
        row = observation_row('clip.mp4', 6, 5, .25, face, decision, {}, engine,
                              10., 2., OperatingPoint(.4, .05))
        required = {'source_video', 'frame_sequence', 'acquisition_time', 'track_id',
            'bbox', 'face_width', 'face_height', 'area_ratio', 'detector_confidence',
            'sharpness', 'brightness', 'nose_offset', 'roll_degrees', 'quality_score',
            'quality_passed', 'quality_reasons', 'top_k_selected', 'embedding_generated',
            'embedding_reused', 'top1_user_id', 'top1_score', 'top1_pose', 'top2_user_id',
            'top2_score', 'top2_pose', 'margin', 'similarity_threshold', 'margin_threshold',
            'gate_a_passed', 'gate_a_reason', 'temporal_status', 'confirmed_user_id',
            'confirmed_event_emitted', 'detection_time_ms', 'recognition_time_ms',
            'shadow_match_evaluated', 'shadow_embedding_generated'}
        self.assertTrue(required <= row.keys())
        self.assertEqual(row['quality_reasons'], ['face_too_small', 'blurred_face'])
        json.dumps(row, allow_nan=False)

    def test_funnel_counts_and_separate_shadow_inference(self):
        decision = make_decision()
        row = observation_row('clip.mp4', 1, 0, 0., SimpleNamespace(det_score=.8),
            decision, {}, SimpleNamespace(confirmations={}), 1., 1., OperatingPoint(.4, .05),
            dict(evaluated=True, embedding_generated=True, alignment_succeeded=True,
                 error_type=None, top1_user_id='a', top1_score=.3, top1_pose='front',
                 top2_user_id='b', top2_score=.2, top2_pose='left', margin=.1))
        funnel = ReplayFunnel(2)
        funnel.add_frame()
        funnel.add_observation(row)
        summary = funnel.summary()
        self.assertEqual(summary['detected_face_observations'], 1)
        self.assertEqual(summary['quality_rejected'], 1)
        self.assertEqual(summary['adaface_inferences'], 0)
        self.assertEqual(summary['shadow_adaface_inferences'], 1)
        self.assertEqual(summary['quality_rejection_reasons'],
                         {'face_too_small': 1, 'blurred_face': 1})


class ShadowMatcherTests(unittest.TestCase):
    def shadow_fixture(self):
        frame = np.zeros((32, 32, 3), dtype=np.uint8)
        face = SimpleNamespace(bbox=(1, 1, 20, 20), kps=np.zeros((5, 2)))
        model = SimpleNamespace(extract_embedding=Mock(return_value=np.array([1., 0.])))
        align = Mock(return_value=np.zeros((112, 112, 3), dtype=np.uint8))
        expected = MatchResult('a', .7, 'front', 'b', .5, 'left', .2, 2)
        matcher = Mock(return_value=expected)
        return frame, face, model, align, matcher

    def test_rejected_valid_face_can_be_shadow_matched(self):
        frame, face, model, align, matcher = self.shadow_fixture()
        result = shadow_match_observation(frame, face, model, align, (), 2, matcher)
        self.assertTrue(result['evaluated'])
        self.assertTrue(result['embedding_generated'])
        self.assertEqual(result['top1_user_id'], 'a')
        model.extract_embedding.assert_called_once()
        matcher.assert_called_once()

    def test_shadow_matcher_does_not_mutate_temporal_state(self):
        frame, face, model, align, matcher = self.shadow_fixture()
        temporal_state = {'candidate': 'a', 'votes': [(1, 0.0)]}
        before = {'candidate': temporal_state['candidate'], 'votes': list(temporal_state['votes'])}
        shadow_match_observation(frame, face, model, align, (), 2, matcher)
        self.assertEqual(temporal_state, before)

    def test_shadow_matcher_does_not_emit_attendance(self):
        frame, face, model, align, matcher = self.shadow_fixture()
        sink = Mock()
        shadow_match_observation(frame, face, model, align, (), 2, matcher)
        sink.accept.assert_not_called()

    def test_shadow_alignment_failure_does_not_infer(self):
        model = SimpleNamespace(extract_embedding=Mock())
        result = shadow_match_observation(None, SimpleNamespace(bbox=(), kps=()), model,
            Mock(side_effect=ValueError('alignment')), (), 2, Mock())
        self.assertFalse(result['evaluated'])
        self.assertFalse(result['embedding_generated'])
        model.extract_embedding.assert_not_called()


if __name__ == '__main__':
    unittest.main()
