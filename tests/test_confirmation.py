from pathlib import Path
import sys
from dataclasses import replace
from types import SimpleNamespace
import unittest
import numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "backend"))
from recognition.temporal import gate_a, OperatingPoint, TemporalConfirmation, TemporalConfig
from recognition.identity_matcher import MatchResult
from recognition.events import LocalAttendanceSink, ConfirmedIdentityEvent
from recognition.local_gallery import Gallery, Identity, Prototype
from recognition.offline_config import OfflineConfig
from recognition.coordinator import RecognitionCoordinator


class ConfidenceGateTests(unittest.TestCase):
    def setUp(self):
        self.point = OperatingPoint(.5, .1)
        self.match = MatchResult('a', .8, 'front', 'b', .4, 'left', .4, 2)

    def test_similarity_and_margin(self):
        self.assertTrue(gate_a(self.match, self.point)[0])
        self.assertEqual(gate_a(replace(self.match, top1_score=.3), self.point)[1], 'below_similarity_threshold')
        self.assertEqual(gate_a(replace(self.match, margin=.01), self.point)[1], 'ambiguous_identity')
        self.assertFalse(gate_a(replace(self.match, top2_user_id='a'), self.point)[0])

    def test_margin_unavailable_policy(self):
        m = replace(self.match, top2_user_id=None, top2_score=None, margin=None)
        self.assertFalse(gate_a(m, self.point)[0])
        self.assertTrue(gate_a(m, replace(self.point, fail_closed_when_margin_unavailable=False))[0])

    def test_prototype_metadata_does_not_change_gate_a(self):
        enriched = replace(
            self.match,
            top1_winning_prototype_id="front_02",
            top2_winning_prototype_id="left_01",
        )

        self.assertEqual(
            gate_a(enriched, self.point),
            gate_a(self.match, self.point),
        )


class TemporalTests(unittest.TestCase):
    def test_distinct_frames_duration_and_duplicate(self):
        c = TemporalConfirmation()
        self.assertFalse(c.observe('a', True, 1, 0, (1,)))
        self.assertFalse(c.observe('a', True, 1, .3, (1,)))
        self.assertEqual(len(c.votes), 1)
        self.assertFalse(c.observe('a', True, 2, .3, (1, 2)))
        self.assertTrue(c.observe('a', True, 3, .6, (1, 2, 3)))

    def test_same_aggregate_not_new_evidence(self):
        c = TemporalConfirmation()
        c.observe('a', True, 1, 0, (1,))
        self.assertFalse(c.observe('a', True, 2, 1, (1,)))
        self.assertEqual(len(c.votes), 1)

    def test_switch_and_expiry(self):
        c = TemporalConfirmation()
        c.observe('a', True, 1, 0, (1,))
        c.observe('a', True, 2, .3, (1, 2))
        self.assertFalse(c.observe('b', True, 3, .6, (3,)))
        self.assertEqual(len(c.votes), 1)
        self.assertFalse(c.observe('b', True, 4, 7, (4,)))
        self.assertEqual(len(c.votes), 1)


class CoordinatorTests(unittest.TestCase):
    def setUp(self):
        pattern = ((np.indices((112, 112)).sum(axis=0) % 2)*100 + 70).astype(np.uint8)
        self.frame = np.repeat(pattern[:, :, None], 3, axis=2)
        self.face = SimpleNamespace(bbox=(0, 0, 112, 112), landmarks=((32, 35), (80, 35), (56, 58), (38, 82), (74, 82)), detector_confidence=.9)
        self.vector = np.array([1., 0.])
        gallery = Gallery('g', {'embedding_dimension': 2}, (
            Identity('a', 'A', 'student', (Prototype('front', np.array([1., 0.])),)),
            Identity('b', 'B', 'student', (Prototype('front', np.array([0., 1.])),))))
        self.sink = LocalAttendanceSink()
        self.engine = RecognitionCoordinator(SimpleNamespace(extract_embedding=lambda crop:self.vector),
            lambda **kwargs:self.frame.copy(), gallery, OfflineConfig(OperatingPoint(.5, .1)), 'session', self.sink)

    def step(self, seq, t=None, stream='camera'):
        return self.engine.process(self.frame, [self.face], seq, seq*.3 if t is None else t, stream)[0]

    def test_only_confirmed_reaches_sink(self):
        self.assertFalse(self.step(1).confirmed)
        self.assertFalse(self.step(2).confirmed)
        self.assertEqual(len(self.sink.events), 0)
        self.assertTrue(self.step(3).confirmed)
        self.assertEqual(len(self.sink.events), 1)
        self.assertEqual(self.engine.inference_count, 3)
        self.assertTrue(self.step(3).confirmed)  # Duplicate preserves state, never adds evidence.
        self.assertEqual(self.engine.inference_count, 3)

    def test_switch_clears_old_embeddings(self):
        self.step(1); self.step(2)
        self.vector = np.array([0., 1.])
        result = self.step(3)
        self.assertFalse(result.confirmed)
        self.assertEqual(result.selected_frames, 1)
        self.assertEqual(result.match.top1_user_id, 'b')
        self.step(4)
        self.assertTrue(self.step(5).confirmed)
        self.assertEqual(self.sink.events[0].user_id, 'b')

    def test_expiry_reconnect_generation_and_attendance_dedup(self):
        for seq in range(1, 4): self.step(seq)
        first_track = self.sink.events[0].track_id
        result = self.step(4, 5)
        self.assertFalse(result.confirmed)
        self.assertNotEqual(result.track_id, first_track)
        self.step(5, 5.3); self.step(6, 5.6)
        self.assertEqual(len(self.sink.events), 1)
        self.assertFalse(self.step(1, 0, stream='reconnected').confirmed)
        self.engine.gallery = replace(self.engine.gallery, generation='g2')
        self.assertFalse(self.step(2, .3, stream='reconnected').confirmed)

    def test_quality_failure_and_short_miss_preserve_confirmed_state(self):
        self.step(1); self.step(2)
        self.engine.process(self.frame, [], 3, .9, 'camera')
        self.assertTrue(self.step(4).confirmed)
        self.frame[:] = 0
        result = self.step(5)
        self.assertTrue(result.confirmed)
        self.assertEqual(result.observation_reason, 'low_quality')
        self.assertEqual(len(self.sink.events), 1)

    def test_sink_rejects_unconfirmed_objects(self):
        with self.assertRaises(ValueError): self.sink.accept({'user_id':'a'})
