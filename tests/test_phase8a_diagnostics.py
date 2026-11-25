"""Diagnostic math, output bounds and behavioral transparency; no model/network."""
from dataclasses import asdict, replace
import json
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]/'backend'))
from recognition.phase8a_diagnostics import stats, buckets, summarize
from recognition.phase8a_probe import ReplayProbe
from scripts.phase8a_replay import JsonlOutputs, local_video


def frame(**changes):
    value = dict.fromkeys(('detections_raw', 'detections_after_nms', 'quality_pass_count',
        'quality_reject_count', 'topk_selected_count', 'adaface_inference_count', 'gate_a_pass_count',
        'gate_b_confirm_count', 'confirmed_event_count', 'active_track_count', 'ambiguous_associations',
        'short_miss_state_preserved', 'detection_ms', 'tracking_ms', 'quality_ms', 'adaface_ms',
        'matching_ms', 'temporal_ms', 'recognition_coordinator_ms', 'total_recognition_pipeline_ms'), 0)
    return dict(value, **changes)


def observation(passed=True, **changes):
    measurement = dict(valid=True, width=40, height=50, area_ratio=.01, sharpness=30,
                       brightness=100, detector_confidence=.9, nose_offset=.1, roll_degrees=2)
    value = dict(video_id='a', track_id=1, face_width=40, face_height=50, face_area_ratio=.01,
        detector_confidence=.9, quality=dict(passed=passed, reasons=[] if passed else ['face_too_small', 'blurred_face']),
        quality_measurements=measurement, adaface_calls=[], matcher_attempts=[], gate_a_passed=False,
        temporal=dict(identity_switch_reset=False, duplicate_observation_ignored=False))
    return dict(value, **changes)


class SummaryTests(unittest.TestCase):
    def test_percentiles_and_nonfinite(self):
        value = stats([0, 10, 20, 30, None, float('nan'), float('inf')])
        self.assertEqual(value['count'], 4)
        self.assertEqual(value['median'], 15)
        self.assertEqual(value['p25'], 7.5)
        self.assertAlmostEqual(value['p95'], 28.5)

    def test_empty_schema_and_zero_denominators(self):
        value = summarize([], [], [], 0)
        self.assertEqual(set(value), {'schema_version', 'labels_available', 'detection', 'quality',
                                    'tracking', 'recognition', 'performance', 'funnel'})
        self.assertIsNone(value['funnel']['conditional_rates']['gate_b_per_candidate_track']['percent'])
        self.assertEqual(stats([])['count'], 0)
        json.dumps(value, allow_nan=False)

    def test_bucket_boundaries(self):
        value = buckets([9.99, 10, 19.99, 20, 39.99, 40, 59.99, 60, 79.99, 80])
        self.assertEqual(list(value.values()), [1, 2, 1, 1, 2, 2, 1])

    def test_funnel_denominators(self):
        value = summarize([frame(detections_raw=12, detections_after_nms=10, quality_pass_count=2,
            quality_reject_count=8, topk_selected_count=1)], [observation(), observation()]+[observation(False)]*8, [], 1)
        rates = value['funnel']['conditional_rates']
        self.assertEqual(rates['quality_pass_per_detection']['percent'], 20)
        self.assertEqual(rates['selected_per_quality_pass']['percent'], 50)
        self.assertEqual(value['detection']['duplicate_detections_removed'], 2)

    def test_overlapping_quality_reasons(self):
        value = summarize([frame(detections_after_nms=1)], [observation(False)], [], 1)['quality']
        self.assertEqual(value['quality_reject'], 1)
        self.assertEqual(sum(value['overlapping_reason_counts'].values()), 2)

    def test_invalid_quality_defaults_excluded(self):
        row = observation(False)
        row['quality_measurements']['valid'] = False
        value = summarize([frame(detections_after_nms=1)], [row], [], 1)['quality']
        self.assertEqual(value['quality_total'], 1)
        self.assertEqual(value['distributions']['all']['sharpness']['count'], 0)

    def test_gate_a_calls_are_not_observation_count(self):
        match = dict(top1_score=.6, top2_score=.2, margin=.4, similarity_pass=True,
                     margin_pass=True, gate_a_pass=True, gate_a_reason='temporal_pending')
        row = observation(matcher_attempts=[match, match], gate_a_passed=True)
        value = summarize([frame(detections_after_nms=1, gate_a_pass_count=1)], [row], [], 1)['recognition']['gate_a']
        self.assertEqual(value['matcher_calls'], 2)
        self.assertEqual(value['production_observation_attempts'], 1)
        self.assertEqual(value['production_pass_count'], 1)

    def test_gate_b_denominator_unique_video_tracks(self):
        rows = [observation(gate_a_passed=True), observation(gate_a_passed=True),
                observation(gate_a_passed=True, video_id='b')]
        value = summarize([frame(gate_b_confirm_count=1)], rows, [], 1)['recognition']['gate_b']
        self.assertEqual(value['candidate_tracks'], 2)
        self.assertEqual(value['confirmation_rate']['percent'], 50)

    def test_track_lifetimes_and_retirements(self):
        rows = [dict(first_seen=0, last_seen=2, observations=3, short_miss_recoveries=1,
                     retirement_reason='max_misses', censored=False),
                dict(first_seen=1, last_seen=1, observations=1, short_miss_recoveries=0,
                     retirement_reason=None, censored=True)]
        value = summarize([], [], rows, 10)['tracking']
        self.assertEqual(value['lifetime_seconds']['mean'], 1)
        self.assertEqual(value['tracks_created_per_source_minute'], 12)
        self.assertEqual(value['tracks_expired'], 1)
        self.assertEqual(value['single_observation_track_rate']['percent'], 50)

    def test_no_unlabeled_accuracy_metric_keys(self):
        forbidden = {'accuracy', 'FAR', 'FRR', 'precision', 'recall', 'F1', 'wrong_identity_rate'}
        def check(value):
            if isinstance(value, dict):
                self.assertFalse(forbidden & value.keys())
                for child in value.values():
                    check(child)
        check(summarize([], [], [], 0))

    def test_output_schema_and_byte_bound(self):
        with tempfile.TemporaryDirectory() as directory:
            with JsonlOutputs(Path(directory), 60) as output:
                output.write('confirmed_events', dict(video_id='a'))
                with self.assertRaises(RuntimeError):
                    output.write('confirmed_events', dict(video_id='a'*100))
            row = json.loads((Path(directory)/'confirmed_events.jsonl').read_text())
            self.assertEqual(row, dict(schema_version=1, video_id='a'))

    def test_reject_remote_video_input(self):
        import argparse
        for value in ('rtsp://localhost/stream', 'https://localhost/test.mp4', '\\\\host\\test.mp4'):
            with self.assertRaises(argparse.ArgumentTypeError):
                local_video(value)


class ProbeTests(unittest.TestCase):
    def fixture(self):
        from test_confirmation import CoordinatorTests
        fixture = CoordinatorTests()
        fixture.setUp()
        return fixture

    def test_instrumentation_preserves_decisions_and_counts(self):
        plain, measured = self.fixture(), self.fixture()
        detector = SimpleNamespace(face_app=SimpleNamespace(get=lambda image: []))
        with ReplayProbe(measured.engine, detector) as probe:
            for sequence, timestamp in ((1, 0), (2, .3), (3, .6)):
                expected = plain.engine.process(plain.frame, [plain.face], sequence, timestamp, 'test')[0]
                probe.begin_frame()
                actual = measured.engine.process(measured.frame, [measured.face], sequence, timestamp, 'test')[0]
                self.assertEqual(asdict(expected), asdict(actual))
                detail = probe.details[(actual.track_id, sequence)]
                self.assertEqual(len(detail['adaface_calls']), 1)
                self.assertEqual(len(detail['matcher_attempts']), 2)
                self.assertTrue(all(m['gate_a_pass'] for m in detail['matcher_attempts']))
                self.assertNotIn('embedding', detail['adaface_calls'][0])
            probe.begin_frame()
            measured.engine.process(measured.frame, [], 4, .7, 'test')
            self.assertEqual(probe.preserved_misses(), [actual.track_id])
        self.assertEqual(measured.engine.inference_count, plain.engine.inference_count)

    def test_probe_restores_functions_on_exception(self):
        from recognition import coordinator
        original = coordinator.match_identity
        fixture = self.fixture()
        detector = SimpleNamespace(face_app=SimpleNamespace(get=lambda image: []))
        with self.assertRaises(RuntimeError):
            with ReplayProbe(fixture.engine, detector):
                raise RuntimeError('test cleanup')
        self.assertIs(coordinator.match_identity, original)

    def test_temporal_refresh_metadata(self):
        from recognition.quality import decide_quality
        fixture = self.fixture()
        detector = SimpleNamespace(face_app=SimpleNamespace(get=lambda image: []))
        with ReplayProbe(fixture.engine, detector) as probe:
            for seq, timestamp, score in ((1, 0., 1.), (2, .1, .99), (3, .2, .98), (4, .5, .7)):
                probe.begin_frame()
                with patch('recognition.coordinator.decide_quality', side_effect=lambda m, c: replace(decide_quality(m, c), quality_score=score)):
                    result = fixture.engine.process(fixture.frame, [fixture.face], seq, timestamp, 'test')[0]
            selection = probe.details[(result.track_id, 4)]['selection']
            self.assertTrue(selection['forced_temporal_refresh'])
            self.assertEqual(selection['evicted_observation_ids'], [3])


if __name__ == '__main__':
    unittest.main()
