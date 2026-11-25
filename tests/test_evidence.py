from pathlib import Path
import sys
import unittest
import numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "backend"))
from recognition.evidence import BestFrameBuffer, EvidenceConfig, Observation, aggregate_embeddings


def observation(sequence, score=.8, timestamp=None, track_id=1):
    return Observation(track_id, sequence, float(sequence if timestamp is None else timestamp), score,
                       np.zeros((112, 112, 3), np.uint8))


class EvidenceTests(unittest.TestCase):
    def test_quality_only_and_top_k_ties(self):
        b = BestFrameBuffer(1, "g", 2, EvidenceConfig(top_k=2))
        self.assertFalse(b.offer(observation(1), False))
        for i in range(2, 6):
            b.offer(observation(i), True)
        self.assertEqual(b.fingerprint, (4, 5))
        self.assertLessEqual(b.memory_bytes, 2*112*112*3)

    def test_lower_quality_not_selected(self):
        b = BestFrameBuffer(1, "g", 2, EvidenceConfig(top_k=1))
        b.offer(observation(1), True)
        self.assertFalse(b.offer(observation(2, .1, timestamp=1.1), True))
        self.assertEqual(b.fingerprint, (1,))

    def test_cache_and_eviction(self):
        b = BestFrameBuffer(1, "g", 2, EvidenceConfig(top_k=1))
        calls = []
        def embed(crop):
            calls.append(1)
            return np.array([1., 0.])
        b.offer(observation(1), True)
        b.aggregate(embed)
        b.aggregate(embed)
        self.assertFalse(b.offer(observation(1), True))
        self.assertEqual(len(calls), 1)
        b.offer(observation(2), True)
        b.aggregate(embed)
        self.assertEqual(len(calls), 2)
        self.assertEqual(len(b.observations), 1)
        b.expire(9)
        self.assertEqual(b.memory_bytes, 0)
        self.assertFalse(b.offer(observation(1), True))

    def test_weighted_aggregation(self):
        np.testing.assert_allclose(aggregate_embeddings([[1, 0]], [1], 2), [1, 0])
        expected = np.array([1., 2.])/np.sqrt(5)
        np.testing.assert_allclose(aggregate_embeddings([[1, 0], [0, 1]], [1, 2], 2), expected, atol=1e-6)

    def test_invalid_weights_and_cancellation(self):
        for weights in [[0], [-1], [float('nan')], [float('inf')]]:
            with self.assertRaises(ValueError):
                aggregate_embeddings([[1, 0]], weights, 2)
        with self.assertRaises(ValueError):
            aggregate_embeddings([[1, 0], [-1, 0]], [1, 1], 2)

    def test_cross_track_rejected(self):
        with self.assertRaises(ValueError):
            BestFrameBuffer(1, "g", 2).offer(observation(1, track_id=2), True)
