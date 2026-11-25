"""Regression scenarios from the Phase 7.1 review; no models or cameras opened."""
from dataclasses import replace
import unittest
from unittest.mock import patch, Mock
import numpy as np


class StateRegressionTests(unittest.TestCase):
    def setUp(self):
        from test_confirmation import CoordinatorTests
        fixture = CoordinatorTests()
        fixture.setUp()
        self.fixture, self.engine = fixture, fixture.engine
        self.sink = Mock()
        self.sink.accept.return_value = True
        self.engine.sink = self.sink

    def step(self, seq, timestamp, score=.9):
        from recognition.quality import decide_quality
        with patch('recognition.coordinator.decide_quality', side_effect=lambda m, c: replace(decide_quality(m, c), quality_score=score)):
            return self.engine.process(self.fixture.frame, [self.fixture.face], seq, timestamp, 'local')[0]

    def test_short_miss_preserves_pending_evidence(self):
        first = self.step(1, 0)
        self.engine.process(self.fixture.frame, [], 2, .1, 'local')
        self.assertIn(first.track_id, self.engine.buffers)
        self.assertEqual(len(self.engine.confirmations[first.track_id].votes), 1)
        returned = self.step(3, .3)
        self.assertEqual(first.track_id, returned.track_id)
        self.assertEqual(returned.valid_observations, 2)
        self.assertTrue(self.step(4, .6).confirmed)

    def test_early_top_k_does_not_stall_duration(self):
        for seq, timestamp, score in [(1, 0., 1.), (2, .1, .99), (3, .2, .98)]:
            self.assertFalse(self.step(seq, timestamp, score).confirmed)
        results = [self.step(seq, timestamp, .7) for seq, timestamp in [(4, .3), (5, .4), (6, .5), (7, .6)]]
        self.assertTrue(any(result.confirmed for result in results))
        self.assertLessEqual(len(next(iter(self.engine.buffers.values())).observations), 3)
        self.assertGreater(self.engine.inference_count, 3)

    def test_confirmed_survives_unselected_frame_and_no_event_repeat(self):
        for seq, t in [(1, 0), (2, .3), (3, .6)]:
            result = self.step(seq, t)
        self.assertTrue(result.confirmed)
        calls = self.engine.inference_count
        self.assertTrue(self.step(4, .61, .1).confirmed)
        self.assertEqual(self.engine.inference_count, calls)
        self.assertEqual(self.sink.accept.call_count, 1)

    def test_confirmed_duplicate_does_not_vote_or_emit(self):
        for seq, t in [(1, 0), (2, .3), (3, .6)]:
            result = self.step(seq, t)
        votes = result.valid_observations
        again = self.step(3, .7)  # A duplicate sequence cannot acquire fresh time evidence.
        self.assertTrue(again.confirmed)
        self.assertEqual(again.valid_observations, votes)
        self.assertEqual(self.engine.tracker.last_time, .6)
        self.assertEqual(self.engine.inference_count, 3)
        self.assertEqual(self.sink.accept.call_count, 1)

    def test_expiry_clears_evidence_and_confirmed_state(self):
        for seq, t in [(1, 0), (2, .3), (3, .6)]:
            old = self.step(seq, t)
        for seq in (4, 5, 6):
            self.engine.process(self.fixture.frame, [], seq, seq*.3, 'local')
        self.assertNotIn(old.track_id, self.engine.buffers)
        new = self.step(7, 2.1)
        self.assertNotEqual(new.track_id, old.track_id)
        self.assertFalse(new.confirmed)

    def test_strong_contradiction_resets_confirmed_identity(self):
        for seq, t in [(1, 0), (2, .3), (3, .6)]:
            self.step(seq, t)
        self.fixture.vector = np.array([0., 1.])
        result = self.step(4, .9)
        self.assertFalse(result.confirmed)
        self.assertEqual(result.match.top1_user_id, 'b')
        self.assertEqual(result.valid_observations, 1)
        self.assertEqual(result.selected_frames, 1)

    def test_quality_failure_does_not_clear_confirmed_identity(self):
        for seq, t in [(1, 0), (2, .3), (3, .6)]:
            self.step(seq, t)
        self.fixture.frame[:] = 0
        result = self.step(4, .7)
        self.assertTrue(result.confirmed)
        self.assertFalse(result.quality_decision.passed)
        self.assertEqual(self.sink.accept.call_count, 1)

    def test_fresh_same_identity_has_one_transition_event(self):
        for seq in range(1, 8):
            result = self.step(seq, seq*.3)
        self.assertTrue(result.confirmed)
        self.assertEqual(self.sink.accept.call_count, 1)

    def test_weak_contradiction_is_not_identity_switch(self):
        for seq, t in [(1, 0), (2, .3), (3, .6)]:
            self.step(seq, t)
        self.fixture.vector = np.array([.7, .71])  # Ambiguous margin; no strong B evidence.
        result = self.step(4, .9)
        self.assertTrue(result.confirmed)
        self.assertEqual(result.match.top1_user_id, 'a')
        self.assertFalse(result.gate_a_passed)
        self.assertEqual(self.sink.accept.call_count, 1)

    def test_ttl_removes_confirmed_state(self):
        for seq, t in [(1, 0), (2, .3), (3, .6)]:
            result = self.step(seq, t)
        self.engine.process(self.fixture.frame, [], 4, 3., 'local')
        self.assertNotIn(result.track_id, self.engine.confirmed_states)
        self.assertNotIn(result.track_id, self.engine.confirmations)

    def test_ambiguity_invalidates_confirmed_state(self):
        for seq, t in [(1, 0), (2, .3), (3, .6)]:
            result = self.step(seq, t)
        decisions = self.engine.process(self.fixture.frame, [self.fixture.face, self.fixture.face], 4, .9, 'local')
        self.assertNotIn(result.track_id, self.engine.confirmed_states)
        self.assertTrue(all(not d.confirmed for d in decisions))

    def test_replay_lifecycle_diagnostics_are_accounted(self):
        from recognition.replay_diagnostics import ReplayDiagnostics
        diagnostics = ReplayDiagnostics(20)
        diagnostics.observe(self.engine, [self.step(1, 0)])
        decisions = self.engine.process(self.fixture.frame, [], 2, .1, 'local')
        diagnostics.observe(self.engine, decisions)
        diagnostics.observe(self.engine, [self.step(3, .3)])
        decisions = self.engine.process(self.fixture.frame, [], 4, 3, 'local')
        diagnostics.observe(self.engine, decisions)
        report, rows = diagnostics.summary(self.engine.tracker, 3)
        self.assertEqual(report['tracks_created'], 1)
        self.assertEqual(report['short_misses_recovered'], 1)
        self.assertEqual(report['track_expirations'], 1)
        self.assertEqual(report['median_observations_per_track'], 2)
        self.assertAlmostEqual(report['median_observed_track_lifetime_seconds'], .3)
        self.assertEqual(report['percent_tracks_surviving_short_miss'], 100.)

    def test_invalid_embedding_dropped_without_repeat_inference(self):
        from recognition.evidence import BestFrameBuffer, Observation
        buffer = BestFrameBuffer(1, 'g', 2)
        bad = Mock(return_value=np.array([float('nan'), 0.]))
        buffer.offer(Observation(1, 1, 0, .9, self.fixture.frame), True)
        with self.assertRaises(ValueError):
            buffer.aggregate(bad)
        self.assertEqual(buffer.inference_count, 1)
        self.assertEqual(buffer.observations, [])
        self.assertFalse(buffer.offer(Observation(1, 1, 0, .9, self.fixture.frame), True))
        self.assertEqual(bad.call_count, 1)

    def test_session_scoped_sink_reload_dedup_is_bounded(self):
        import json
        import tempfile
        from pathlib import Path
        from recognition.events import LocalAttendanceSink, ConfirmedIdentityEvent
        with tempfile.TemporaryDirectory() as root:
            path = Path(root) / 'events.jsonl'
            path.write_text('\n'.join(json.dumps(dict(session_id=session, user_id='a'))
                            for session in ('previous', 'current')), encoding='utf-8')
            sink = LocalAttendanceSink(path, session_id='current')
            self.assertEqual(sink.seen, {('current', 'a')})
            event = ConfirmedIdentityEvent('current', 'g', 9, 'a', .9, .1, .8, 3, 3, .6, 3, .6, 3)
            self.assertFalse(sink.accept(event))
            with self.assertRaises(ValueError):
                sink.accept(replace(event, session_id='different'))
