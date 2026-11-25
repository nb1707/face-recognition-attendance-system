from pathlib import Path
import sys
import unittest
import numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "backend"))
from recognition.quality import QualityConfig, measure_quality, decide_quality


class QualityTests(unittest.TestCase):
    def setUp(self):
        pattern = ((np.indices((112, 112)).sum(axis=0) % 2)*100 + 70).astype(np.uint8)
        self.frame = np.repeat(pattern[:, :, None], 3, axis=2)
        self.box = (0, 0, 112, 112)
        self.points = [(32, 35), (80, 35), (56, 58), (38, 82), (74, 82)]

    def check(self, frame=None, box=None, points=None, confidence=.9):
        m = measure_quality(self.frame if frame is None else frame, self.box if box is None else box,
                            self.points if points is None else points, confidence)
        return m, decide_quality(m)

    def test_valid_and_bounded_visible_score(self):
        m, d = self.check()
        self.assertTrue(d.passed)
        self.assertEqual(len(d.components), 5)
        self.assertTrue(0 < d.quality_score <= 1)
        self.assertEqual(m.area_ratio, 1)

    def test_small_face(self):
        _, d = self.check(box=(0, 0, 28, 28), points=np.array(self.points)/4)
        self.assertIn("face_too_small", d.reasons)

    def test_blur(self):
        _, d = self.check(frame=np.full_like(self.frame, 128))
        self.assertIn("blurred_face", d.reasons)

    def test_lighting(self):
        for value in [0, 255]:
            _, d = self.check(frame=np.full_like(self.frame, value))
            self.assertIn("poor_lighting", d.reasons)

    def test_detector_confidence(self):
        _, d = self.check(confidence=.1)
        self.assertIn("low_detector_confidence", d.reasons)

    def test_invalid_landmarks(self):
        for points in [[], [(float('nan'), 0)]*5, [(0, 0)]*5]:
            m, d = self.check(points=points)
            self.assertFalse(m.valid)
            self.assertFalse(d.passed)

    def test_background_does_not_change_crop_measurement(self):
        padded = np.zeros((224, 224, 3), np.uint8)
        padded[:112, :112] = self.frame
        a, _ = self.check()
        b, _ = self.check(frame=padded)
        self.assertEqual((a.sharpness, a.brightness), (b.sharpness, b.brightness))

    def test_provisional_quality_thresholds_are_unchanged(self):
        config = QualityConfig()
        self.assertEqual(config.min_size, 32)
        self.assertEqual(config.min_detector, .5)
        self.assertEqual(config.min_sharpness, 20)
        self.assertEqual((config.min_brightness, config.max_brightness), (35, 225))
        self.assertEqual(config.max_nose_offset, .65)
        self.assertEqual(config.max_roll_degrees, 40)
