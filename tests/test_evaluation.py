from dataclasses import asdict
from pathlib import Path
import sys
import unittest
import tempfile
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'backend'))
from recognition.identity_matcher import MatchResult
from recognition.offline_evaluation import UNKNOWN, metrics, sweep, pareto_candidates
from recognition.temporal import OperatingPoint


def row(expected, predicted, score=.8, margin=.3, status='evaluable'):
    return dict(expected_user_id=expected, gallery_identity_available=True, status=status,
                match=asdict(MatchResult(predicted, score, 'front', 'other', score-margin, 'left', margin, 2, 6)))


class EvaluationTests(unittest.TestCase):
    def test_discovery_does_not_invent_labels_for_root_files(self):
        from scripts.offline_benchmark import discover
        with tempfile.TemporaryDirectory() as root:
            root = Path(root)
            (root / 'screenshot.png').touch()
            (root / 'unknown').mkdir()
            (root / 'unknown/x.png').touch()
            (root / 'student').mkdir()
            (root / 'student/y.jpg').touch()
            rows = {path.name: label for path, label, split in discover(root)}
            self.assertIsNone(rows['screenshot.png'])
            self.assertEqual(rows['x.png'], UNKNOWN)
            self.assertEqual(rows['y.jpg'], 'student')

    def test_error_modes_and_conditional_denominators(self):
        rows = [row('a', 'a'), row('b', 'a'), row('a', 'a', status='no_face'),
                row(UNKNOWN, 'a'), row(UNKNOWN, 'a', score=.1)]
        result = metrics(rows, OperatingPoint(.4, .05))
        self.assertEqual(result['FAR'], .5)
        self.assertAlmostEqual(result['FRR'], 1/3)
        self.assertAlmostEqual(result['wrong_identity_rate'], 1/3)
        self.assertAlmostEqual(result['known_identity_accuracy'], 1/3)
        self.assertEqual(result['accuracy'], .4)
        self.assertEqual(metrics(rows, OperatingPoint(.4, .05), conditional=True)['samples'], 4)

    def test_undefined_and_exclusions(self):
        self.assertIsNone(metrics([], OperatingPoint(.4, .05))['FAR'])
        duplicate = dict(row('a', 'a'), excluded_reason='duplicate_image')
        self.assertEqual(metrics([duplicate], OperatingPoint(.4, .05))['samples'], 0)

    def test_sweep_pareto_tradeoff(self):
        rows = [row('a', 'a', score=.6), row(UNKNOWN, 'a', score=.7)]
        candidates = pareto_candidates(sweep(rows, [.5, .65, .8], [.0]))
        self.assertEqual([r['threshold'] for r in candidates], [.5, .8])

    def test_missing_gallery_is_not_relabelled_unknown(self):
        rows = [dict(row('missing', 'a', score=.1), gallery_identity_available=False)]
        result = metrics(rows, OperatingPoint(.4, .05))
        self.assertIsNone(result['FRR'])
        self.assertEqual(result['true_reject'], 1)
        self.assertEqual(metrics(rows, OperatingPoint(.4, .05), enrolled_only=True)['samples'], 0)
