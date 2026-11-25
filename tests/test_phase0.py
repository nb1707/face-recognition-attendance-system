import json
from pathlib import Path
import sys
import tempfile
import threading
import unittest
from unittest.mock import patch

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "backend"))
from recognition.adaface_preprocessing import preprocess_bgr
from recognition.embeddings import normalized_embedding
from recognition.local_gallery import LocalGalleryProvider, configured_gallery_provider


class PhaseZeroTests(unittest.TestCase):
    def test_rgb_channel_contract(self):
        image = np.zeros((112, 112, 3), np.uint8)
        image[:] = [0, 127, 255]
        actual = preprocess_bgr(image)
        self.assertEqual(actual.shape, (1, 3, 112, 112))
        np.testing.assert_allclose(actual[0, :, 0, 0], [1, 127/127.5-1, -1], atol=1e-6)

    def test_actual_adapter_model_input_and_output(self):
        import torch
        from recognition.adaface_model import AdaFaceModel
        model = AdaFaceModel.__new__(AdaFaceModel)
        model.device = torch.device("cpu")
        captured = []
        def inference(tensor):
            captured.append(tensor.numpy())
            return torch.ones((1, 512))
        model.model = inference
        image = np.full((112, 112, 3), [0, 127, 255], dtype=np.uint8)
        embedding = model.extract_embedding(image)
        self.assertAlmostEqual(float(np.linalg.norm(embedding)), 1, places=6)
        self.assertEqual(captured[0][0, 0, 0, 0], 1)
        self.assertEqual(captured[0][0, 2, 0, 0], -1)

    def test_embedding_validation(self):
        for bad in ([], [0, 0], [np.nan, 1], [np.inf, 1], [1, 2, 3], [[1, 2]]):
            with self.subTest(bad=bad), self.assertRaises(ValueError):
                normalized_embedding(bad, 2)
        np.testing.assert_allclose(normalized_embedding([3, 4], 2), [.6, .8])

    def test_manifest_rejects_incompatible_and_skips_corrupt(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            np.save(root / "good.npy", np.array([3., 4.]))
            (root / "bad.npy").write_bytes(b"corrupt")
            provenance = dict(embedding_dimension=2, preprocessing_version="test-v1")
            rows = [dict(user_id="a", pose=pose, embedding_file=file, mean_fallback=False)
                    for pose, file in [("left", "bad.npy"), ("front", "good.npy")]]
            path = root / "manifest.json"
            path.write_text(json.dumps(dict(schema_version=1, generation="test", provenance=provenance, embeddings=rows)))
            gallery = LocalGalleryProvider(path).load(provenance)
            self.assertEqual(len(gallery.identities), 1)
            self.assertEqual(len(gallery.diagnostics), 1)
            with self.assertRaises(ValueError):
                LocalGalleryProvider(path).load(dict(provenance, preprocessing_version="old"))

    def test_no_implicit_cloud_fallback(self):
        with patch.dict("os.environ", {"RECOGNITION_GALLERY_PROVIDER": "supabase"}):
            with self.assertRaises(ValueError):
                configured_gallery_provider()

    def test_detection_timing_propagates(self):
        from desktop.detection.worker import DetectionResult
        from desktop.recognition.worker import RecognitionWorker
        from types import SimpleNamespace
        worker = RecognitionWorker()
        worker._model = object()
        worker._cache = SimpleNamespace(identities=())
        worker._coordinator = SimpleNamespace(process=lambda *args: [])
        worker._threshold = .4
        detection = DetectionResult(1, 7, np.zeros((8, 8, 3), np.uint8), (), 12.5, 80.)
        result = worker._recognize_detection(detection=detection, stop_event=threading.Event())
        self.assertEqual(result.detection_processing_time_ms, 12.5)


if __name__ == "__main__":
    unittest.main()
