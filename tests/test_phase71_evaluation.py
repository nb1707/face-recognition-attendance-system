from dataclasses import asdict
from pathlib import Path
import sys
import unittest
from unittest.mock import Mock, patch
from types import SimpleNamespace
import numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'backend'))
from recognition.offline_evaluation import (UNKNOWN, metrics, enrollment_category, ENROLLED_KNOWN,
    UNENROLLED_IMPOSTOR, EXPLICIT_UNKNOWN, MATCHER_DIAGNOSTIC, PRODUCTION_SIMULATION)
from recognition.identity_matcher import MatchResult
from recognition.temporal import OperatingPoint


def row(expected, prediction='a', enrolled=False, score=.8):
    return dict(expected_user_id=expected, gallery_identity_available=enrolled, quality_passed=True,
        status='evaluable', match=asdict(MatchResult(prediction, score, 'front', 'other', .1, 'left', score-.1, 2)))


class EnrollmentMetricsTests(unittest.TestCase):
    point = OperatingPoint(.4, .05)

    def test_categories_from_actual_gallery(self):
        self.assertEqual(enrollment_category('a', {'a'}), ENROLLED_KNOWN)
        self.assertEqual(enrollment_category('visitor', {'a'}), UNENROLLED_IMPOSTOR)
        self.assertEqual(enrollment_category(UNKNOWN, {'a'}), EXPLICIT_UNKNOWN)

    def test_unenrolled_acceptance_is_false_accept_not_wrong_known(self):
        result = metrics([row('visitor')], self.point)
        self.assertEqual(result['false_accept'], 1)
        self.assertEqual(result['wrong_known'], 0)
        self.assertEqual(result['unenrolled_impostor_FAR'], 1.)

    def test_unenrolled_not_in_frr_denominator(self):
        result = metrics([row('visitor', score=.1), row('a', enrolled=True, score=.1)], self.point)
        self.assertEqual(result['enrolled_known_count'], 1)
        self.assertEqual(result['false_reject'], 1)
        self.assertEqual(result['true_reject'], 1)

    def test_wrong_enrolled_identity_is_wrong_known(self):
        result = metrics([row('b', enrolled=True)], self.point)
        self.assertEqual(result['wrong_known'], 1)
        self.assertEqual(result['false_accept'], 0)

    def test_explicit_unknown_acceptance_is_false_accept(self):
        result = metrics([row(UNKNOWN)], self.point)
        self.assertEqual(result['false_accept'], 1)
        self.assertEqual(result['explicit_unknown_FAR'], 1.)

    def test_combined_impostor_denominator(self):
        result = metrics([row(UNKNOWN), row('v1'), row('v2', score=.1), row('a', enrolled=True)], self.point)
        self.assertEqual(result['impostor_count'], 3)
        self.assertEqual(result['false_accept'], 2)
        self.assertAlmostEqual(result['FAR'], 2/3)
        self.assertEqual(result['impostors_reaching_matcher'], 3)

    def test_conditional_denominator_matches_selected_mode(self):
        sample = dict(row('visitor'), quality_passed=False)
        diagnostic = metrics([sample], self.point, conditional=True, mode=MATCHER_DIAGNOSTIC)
        production = metrics([sample], self.point, conditional=True, mode=PRODUCTION_SIMULATION)
        self.assertEqual(diagnostic['impostor_count'], 1)
        self.assertEqual(production['impostor_count'], 0)
        self.assertIsNone(production['FAR'])


class BenchmarkModesTests(unittest.TestCase):
    def setUp(self):
        from test_confirmation import CoordinatorTests
        fixture = CoordinatorTests()
        fixture.setUp()
        self.gallery, self.config = fixture.engine.gallery, fixture.engine.config
        self.frame = np.full((112, 112, 3), 128, np.uint8)  # Valid geometry, deliberately blurred.
        self.face = SimpleNamespace(bbox=fixture.face.bbox, kps=fixture.face.landmarks, det_score=.9)
        self.detector = Mock()
        self.detector.detect_faces.return_value = [self.face]
        self.detector.detect_faces_tiled.return_value = [self.face]
        self.model = Mock()
        self.model.extract_embedding.return_value = np.array([1., 0.])
        self.align = Mock(return_value=self.frame)

    def evaluate(self, mode, detector_mode='full_frame'):
        from recognition.benchmark_observation import evaluate_observation
        return evaluate_observation(self.frame, self.detector, self.model, self.align,
                                    self.gallery, self.config, detector_mode, mode)

    def test_diagnostic_scores_quality_rejected_valid_face(self):
        result = self.evaluate(MATCHER_DIAGNOSTIC)
        self.assertFalse(result['quality_passed'])
        self.assertIn('blurred_face', result['quality_reasons'])
        self.assertTrue(result['aligned'])
        self.assertEqual(result['status'], 'evaluable')
        self.assertIsNotNone(result['top1_score'])
        self.model.extract_embedding.assert_called_once()

    def test_production_enforces_quality_and_skips_embedding(self):
        result = self.evaluate(PRODUCTION_SIMULATION)
        self.assertEqual(result['status'], 'low_quality')
        self.model.extract_embedding.assert_not_called()

    def test_full_frame_and_desktop_share_existing_paths_and_config(self):
        from recognition.detection_config import CLASSROOM_ROI, TILED_DETECTION_CONFIG
        self.evaluate(MATCHER_DIAGNOSTIC, 'full_frame')
        self.detector.detect_faces.assert_called_once()
        self.evaluate(MATCHER_DIAGNOSTIC, 'desktop_tiled')
        arguments = self.detector.detect_faces_tiled.call_args.kwargs
        self.assertIs(arguments['roi_config'], CLASSROOM_ROI)
        for key in ('tile_count', 'overlap_ratio', 'nms_iou_threshold'):
            self.assertEqual(arguments[key], TILED_DETECTION_CONFIG[key])

    def test_alignment_failure_never_reaches_matcher(self):
        self.align.side_effect = ValueError('bad alignment')
        result = self.evaluate(MATCHER_DIAGNOSTIC)
        self.assertEqual(result['error_reason'], 'alignment_failure')
        self.model.extract_embedding.assert_not_called()

    def test_rejection_combinations_and_metric_ranges(self):
        from recognition.benchmark_observation import quality_rejection_report
        result = self.evaluate(MATCHER_DIAGNOSTIC)
        report = quality_rejection_report([result])
        self.assertEqual(report['reasons']['blurred_face'], 1)
        self.assertEqual(report['rejected_valid_measurement_ranges']['brightness']['median'], 128.)


class CameraSourceTests(unittest.TestCase):
    def test_webcam_integer_without_hikvision_builder(self):
        from desktop.camera.recognition_source import build_recognition_camera_source
        with patch.dict('os.environ', {'RECOGNITION_CAMERA_SOURCE':'webcam', 'RECOGNITION_WEBCAM_INDEX':'2'}), \
             patch('desktop.camera.recognition_source.build_hikvision_rtsp_source') as builder:
            self.assertEqual(build_recognition_camera_source(), 2)
            builder.assert_not_called()

    def test_hikvision_delegates_without_camera_open(self):
        from desktop.camera.recognition_source import build_recognition_camera_source
        with patch.dict('os.environ', {'RECOGNITION_CAMERA_SOURCE':'hikvision'}), \
             patch('desktop.camera.recognition_source.build_hikvision_rtsp_source', return_value='rtsp://test.invalid/channel') as builder, \
             patch('socket.socket.connect', side_effect=AssertionError('Network forbidden')), \
             patch('cv2.VideoCapture', side_effect=AssertionError('Capture forbidden')):
            self.assertEqual(build_recognition_camera_source(), 'rtsp://test.invalid/channel')
            builder.assert_called_once_with()

    def test_existing_builder_constructs_fake_rtsp_without_dotenv_or_network(self):
        from desktop.camera.recognition_source import build_recognition_camera_source
        settings = {'RECOGNITION_CAMERA_SOURCE':'hikvision', 'HIKVISION_HOST':'test.invalid',
            'HIKVISION_USERNAME':'fake', 'HIKVISION_PASSWORD':'fake', 'HIKVISION_RTSP_PORT':'554',
            'HIKVISION_CHANNEL':'1', 'HIKVISION_STREAM_TYPE':'1'}
        with patch.dict('os.environ', settings, clear=True), patch('desktop.camera.hikvision._load_existing_environment'), \
             patch('socket.socket.connect', side_effect=AssertionError('Network forbidden')), \
             patch('cv2.VideoCapture', side_effect=AssertionError('Capture forbidden')):
            self.assertTrue(build_recognition_camera_source().endswith('/Streaming/Channels/101'))

    def test_invalid_source_is_sanitized(self):
        from desktop.camera.recognition_source import build_recognition_camera_source
        from desktop.camera.hikvision import CameraConfigurationError
        with patch.dict('os.environ', {'RECOGNITION_CAMERA_SOURCE':'invalid'}):
            with self.assertRaises(CameraConfigurationError):
                build_recognition_camera_source()
