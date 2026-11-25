from pathlib import Path
import sys
import unittest
import numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "backend"))
from recognition.local_gallery import Identity, Prototype
from recognition.identity_matcher import score_identities, match_identity


def identity(user, **poses):
    return Identity(user, user, "student", tuple(Prototype(k, np.array(v)) for k, v in poses.items()))


class PrototypeTests(unittest.TestCase):
    def test_max_pose_not_average(self):
        scores, _, comparisons = score_identities([1, 0], [identity("a", front=[0, 1], left=[1, 0], right=[-1, 0])], 2)
        self.assertEqual((scores[0].score, scores[0].winning_pose, comparisons), (1, "left", 3))

    def test_mean_ignored_with_poses(self):
        scores, _, count = score_identities([1, 0], [identity("a", front=[0, 1], mean=[1, 0])], 2)
        self.assertEqual((scores[0].score, scores[0].winning_pose, count), (0, "front", 1))

    def test_valid_mean_fallback_after_invalid_pose(self):
        scores, diagnostics, _ = score_identities([1, 0], [identity("a", front=[0, 0], mean=[1, 0])], 2)
        self.assertEqual(scores[0].winning_pose, "mean")
        self.assertEqual(len(diagnostics), 1)

    def test_pose_order_deterministic(self):
        for poses in [dict(right=[1, 0], front=[1, 0]), dict(front=[1, 0], right=[1, 0])]:
            scores, _, _ = score_identities([1, 0], [identity("a", **poses)], 2)
            self.assertEqual(scores[0].winning_pose, "front")

    def test_corrupt_nonfinite_and_wrong_dimensions(self):
        values = [identity("a", front=[1, 2, 3]), identity("b", front=[float('nan'), 1]),
                  identity("c", front=[float('inf'), 1]), identity("d", front=[0, 0])]
        scores, diagnostics, _ = score_identities([1, 0], values, 2)
        self.assertEqual(scores, ())
        self.assertEqual(len(diagnostics), 4)

    def test_actual_max_scoring_prototype_id_is_preserved(self):
        value = Identity(
            "a",
            "A",
            "student",
            (
                Prototype("front", np.array([0., 1.]), "front_01"),
                Prototype("front", np.array([1., 0.]), "front_02"),
            ),
        )

        scores, _, _ = score_identities([1, 0], [value], 2)

        self.assertEqual(scores[0].score, 1)
        self.assertEqual(scores[0].winning_pose, "front")
        self.assertEqual(scores[0].winning_prototype_id, "front_02")


class RankingTests(unittest.TestCase):
    def test_distinct_top_two_and_duplicate_container(self):
        result = match_identity([1, 0], [identity("a", front=[1, 0]),
            identity("a", left=[1, .1]), identity("b", front=[0, 1])], 2)
        self.assertEqual((result.top1_user_id, result.top2_user_id, result.candidate_count), ("a", "b", 2))
        self.assertEqual(result.margin, 1)

    def test_empty_gallery(self):
        result = match_identity([1, 0], [], 2)
        self.assertIsNone(result.top1_user_id)
        self.assertIsNone(result.top2_score)
        self.assertIsNone(result.margin)

    def test_single_identity_no_synthetic_margin(self):
        result = match_identity([1, 0], [identity("a", front=[1, 0])], 2)
        self.assertEqual(result.top1_user_id, "a")
        self.assertIsNone(result.top2_score)
        self.assertIsNone(result.margin)

    def test_user_tie_lexical(self):
        result = match_identity([1, 0], [identity("b", front=[1, 0]), identity("a", left=[1, 0])], 2)
        self.assertEqual((result.top1_user_id, result.top2_user_id, result.margin), ("a", "b", 0))

    def test_metadata_does_not_change_scores_or_ranking(self):
        vectors = (
            ("front", np.array([0., 1.])),
            ("front", np.array([1., 0.])),
        )
        legacy = (
            Identity("a", "A", "student", tuple(
                Prototype(pose, vector) for pose, vector in vectors
            )),
            Identity("b", "B", "student", (
                Prototype("left", np.array([.8, .6])),
            )),
        )
        enriched = (
            Identity("a", "A", "student", tuple(
                Prototype(pose, vector, f"front_0{index}")
                for index, (pose, vector) in enumerate(vectors, start=1)
            )),
            Identity("b", "B", "student", (
                Prototype("left", np.array([.8, .6]), "left_01"),
            )),
        )

        before = match_identity([1, 0], legacy, 2)
        after = match_identity([1, 0], enriched, 2)

        ranking_fields = (
            "top1_user_id",
            "top1_score",
            "top1_winning_pose",
            "top2_user_id",
            "top2_score",
            "top2_winning_pose",
            "margin",
            "candidate_count",
            "comparisons",
        )
        self.assertEqual(
            tuple(getattr(before, field) for field in ranking_fields),
            tuple(getattr(after, field) for field in ranking_fields),
        )
        self.assertEqual(after.top1_winning_prototype_id, "front_02")
        self.assertEqual(after.top2_user_id, "b")
        self.assertEqual(after.top2_winning_prototype_id, "left_01")
