import json
from pathlib import Path
import sys
import tempfile
import unittest
from types import SimpleNamespace
from unittest.mock import patch
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'backend'))


class OfflineIntegrationTests(unittest.TestCase):
    def test_worker_emits_only_temporally_confirmed_results(self):
        from test_confirmation import CoordinatorTests
        from desktop.recognition.worker import RecognitionWorker
        from desktop.recognition.cache import RecognitionCache
        from desktop.detection.worker import DetectionResult, DetectedFace
        import threading
        fixture = CoordinatorTests()
        fixture.setUp()
        worker = RecognitionWorker()
        worker._model, worker._coordinator = fixture.engine.model, fixture.engine
        worker._cache = RecognitionCache(fixture.engine.gallery)
        worker._threshold = .5
        face = DetectedFace(fixture.face.bbox, fixture.face.landmarks, .9, 112, 112)
        with patch('socket.socket.connect', side_effect=AssertionError('Network forbidden')):
            for seq in range(1, 4):
                detection = DetectionResult(seq, seq, fixture.frame, (face,), 12.5, 30., seq*.3, 'local')
                result = worker._recognize_detection(detection=detection, stop_event=threading.Event())
                self.assertEqual(result.faces[0].recognized, seq == 3)
                self.assertIs(result.source_frame, fixture.frame)
                self.assertEqual(result.detection_processing_time_ms, 12.5)
                json.dumps(result.faces[0].diagnostics, allow_nan=False)
                matcher = result.faces[0].diagnostics['matcher_diagnostics']
                self.assertEqual(matcher['top1']['user_id'], 'a')
                self.assertEqual(matcher['top1']['full_name'], 'A')
                self.assertEqual(matcher['top2']['user_id'], 'b')
                self.assertTrue(matcher['similarity_gate_passed'])
                self.assertTrue(matcher['margin_gate_passed'])
                self.assertTrue(matcher['gate_a_passed'])
        self.assertEqual(len(fixture.sink.events), 1)

    def test_worker_preserves_low_quality_measurements_and_reasons(self):
        from test_confirmation import CoordinatorTests
        from desktop.recognition.worker import RecognitionWorker
        from desktop.recognition.cache import RecognitionCache
        from desktop.detection.worker import DetectionResult, DetectedFace
        import threading

        fixture = CoordinatorTests()
        fixture.setUp()
        worker = RecognitionWorker()
        worker._model, worker._coordinator = fixture.engine.model, fixture.engine
        worker._cache = RecognitionCache(fixture.engine.gallery)
        worker._threshold = .5
        blurred_dark_frame = np.zeros_like(fixture.frame)
        face = DetectedFace(
            fixture.face.bbox,
            fixture.face.landmarks,
            .9,
            112,
            112,
        )
        detection = DetectionResult(
            1,
            1,
            blurred_dark_frame,
            (face,),
            12.5,
            30.,
            .3,
            'local',
        )

        result = worker._recognize_detection(
            detection=detection,
            stop_event=threading.Event(),
        )
        recognized_face = result.faces[0]
        measurements = recognized_face.diagnostics['quality_measurements']
        decision = recognized_face.diagnostics['quality_decision']

        self.assertEqual(recognized_face.failure_reason, 'low_quality')
        self.assertTrue(measurements['valid'])
        self.assertEqual(measurements['brightness'], 0)
        self.assertEqual(measurements['sharpness'], 0)
        self.assertFalse(decision['passed'])
        self.assertEqual(
            set(decision['reasons']),
            {'blurred_face', 'poor_lighting'},
        )

    def test_session_repository_without_network(self):
        from desktop.session.local_repository import LocalRecognitionSessionRepository
        with tempfile.TemporaryDirectory() as root, patch('socket.socket.connect', side_effect=AssertionError('Network forbidden')):
            repository = LocalRecognitionSessionRepository(root, lambda: SimpleNamespace(identities=(SimpleNamespace(user_id='a', full_name='A'),)))
            self.assertEqual(repository.load_users(), (('a', 'A'),))
            session = repository.create_session({'class_code': 'offline'})
            repository.update_config_snapshot(session['id'], config_snapshot={'local': True})
            quality_diagnostics = {
                'quality_measurements': {
                    'valid': True,
                    'sharpness': 0,
                    'brightness': 0,
                },
                'quality_decision': {
                    'passed': False,
                    'reasons': ['blurred_face', 'poor_lighting'],
                },
                'matcher_diagnostics': {
                    'top1': {
                        'user_id': 'a',
                        'winning_prototype_id': 'front_02',
                    },
                    'top2': {
                        'user_id': 'b',
                        'winning_prototype_id': 'front_01',
                    },
                    'margin': .0169,
                    'gate_a_passed': False,
                    'gate_a_reason': 'ambiguous_identity',
                },
            }
            repository.insert_trial({
                'session_id': session['id'],
                'recognized': False,
                'diagnostics': {'face': quality_diagnostics},
            })
            self.assertTrue(repository.end_session(session['id']))
            rows = [json.loads(line) for line in (Path(root) / 'sessions_trials.jsonl').read_text().splitlines()]
            self.assertEqual([r['journal_type'] for r in rows], ['session_started', 'configuration', 'trial', 'session_ended'])
            self.assertEqual(
                rows[2]['diagnostics']['face'],
                quality_diagnostics,
            )

    def test_config_snapshot_uses_offline_point(self):
        from desktop.session.models import load_recognition_configuration_snapshot
        from recognition.offline_config import load_offline_config
        snapshot = load_recognition_configuration_snapshot()
        point = load_offline_config().operating_point
        self.assertEqual(snapshot.model_threshold, point.similarity_threshold)
        self.assertEqual(point.similarity_threshold, .40)
        self.assertEqual(point.margin_threshold, .05)
        self.assertIn('temporal', snapshot.to_snapshot()['recognition_settings'])

    def test_temporal_memory_bound(self):
        from recognition.temporal import TemporalConfirmation, TemporalConfig
        state = TemporalConfirmation(TemporalConfig(max_votes=8))
        for seq in range(100):
            state.observe('a', True, seq, seq / 100, (seq,))
        self.assertEqual(len(state.votes), 8)

    def test_missing_detector_never_downloads(self):
        from recognition.offline_runtime import require_local_detector
        with tempfile.TemporaryDirectory() as root, patch('pathlib.Path.home', return_value=Path(root)):
            with self.assertRaises(FileNotFoundError):
                require_local_detector()
