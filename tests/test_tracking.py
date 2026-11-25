from pathlib import Path
import sys
import unittest
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "backend"))
from recognition.tracker import FaceTracker, TrackerConfig


class TrackerTests(unittest.TestCase):
    def test_stationary_and_moving(self):
        t = FaceTracker()
        self.assertEqual(t.update([(0, 0, 20, 20)], 1, 0), [1])
        self.assertEqual(t.update([(1, 0, 21, 20)], 2, .2), [1])
        self.assertEqual(t.update([(1, 0, 21, 20)], 3, .4), [1])

    def test_two_faces_unique(self):
        t = FaceTracker()
        a, b = (0, 0, 20, 20), (50, 0, 70, 20)
        self.assertEqual(t.update([a, b], 1, 0), [1, 2])
        self.assertEqual(t.update([b, a], 2, .2), [2, 1])

    def test_short_miss_and_expiration(self):
        t = FaceTracker()
        box = (0, 0, 20, 20)
        t.update([box], 1, 0)
        t.update([], 2, .1)
        self.assertEqual(t.update([box], 3, .2), [1])
        self.assertEqual(t.update([box], 4, 3), [2])

    def test_ambiguous_crossing_retires_evidence_ids(self):
        t = FaceTracker()
        t.update([(0, 0, 20, 20), (25, 0, 45, 20)], 1, 0)
        ids = t.update([(10, 0, 30, 20), (9, 0, 29, 20)], 2, .1)
        self.assertEqual(len(set(ids)), 2)
        self.assertNotIn(1, ids)

    def test_duplicate_rejected_without_state_change(self):
        t = FaceTracker()
        t.update([(0, 0, 20, 20)], 1, 0)
        with self.assertRaises(ValueError):
            t.update([], 1, .1)
        self.assertEqual(len(t.tracks), 1)

    def test_bounds_and_misses(self):
        t = FaceTracker(TrackerConfig(max_tracks=1, max_misses=0))
        self.assertEqual(t.update([(0, 0, 10, 10), (30, 0, 40, 10)], 1, 0), [1, None])
        t.update([], 2, .1)
        self.assertEqual(t.tracks, {})

    def test_reset_does_not_reuse_track_id(self):
        t = FaceTracker()
        box = (0, 0, 20, 20)
        t.update([box], 1, 0)
        t.reset()
        self.assertEqual(t.update([box], 1, 0), [2])

    def test_timestamp_contract_through_workers(self):
        import threading
        import numpy as np
        from desktop.camera.worker import CameraWorker
        from desktop.detection.worker import DetectionWorker
        camera = CameraWorker()
        camera._latest_frame = np.zeros((8, 8, 3), dtype=np.uint8)
        camera._frame_sequence = 7
        camera._acquisition_time = 123.45
        camera._stream_id = "session-camera"
        frame = camera.latest_frame()
        detector = DetectionWorker()
        detector._thread = threading.current_thread()
        detector._stop_event = threading.Event()
        detector._frame_ready_event = threading.Event()
        self.assertTrue(detector.submit_frame(frame.image, frame.sequence, frame.acquisition_time, frame.stream_id))
        image, sequence, timestamp, stream_id = detector._take_pending_frame(detector._frame_ready_event)
        result = detector._publish_result(stop_event=detector._stop_event, frame_sequence=sequence,
            frame=image, faces=(), processing_time_ms=5, detection_fps=200,
            acquisition_time=timestamp, stream_id=stream_id)
        self.assertEqual((result.frame_sequence, result.acquisition_time, result.stream_id), (7, 123.45, "session-camera"))
