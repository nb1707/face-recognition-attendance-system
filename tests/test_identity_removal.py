from contextlib import redirect_stdout
import io
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "backend"))

from enrollment.identity_removal import (
    LocalIdentityRemoval,
    LocalIdentityRemovalError,
)
from enrollment.repository import EnrollmentRepository
from recognition.local_gallery import LocalGalleryProvider
from scripts.remove_local_identities import main as remove_main


class LocalIdentityRemovalTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        root = Path(self.temp.name)
        self.gallery_root = root / "gallery"
        self.enrollments_root = root / "enrollments"
        self.old_generation = self.gallery_root / "old-generation"
        embeddings = self.old_generation / "embeddings"
        embeddings.mkdir(parents=True)
        self.enrollments_root.mkdir()
        self.provenance = {
            "model_name": "test",
            "embedding_dimension": 2,
            "preprocessing_version": "test-v1",
        }
        rows = []
        for index, user_id in enumerate(("alpha", "bravo", "charlie")):
            vector = np.array((1., index + 1.), dtype=np.float32)
            vector /= np.linalg.norm(vector)
            filename = f"{user_id}.npy"
            np.save(embeddings / filename, vector, allow_pickle=False)
            rows.append({
                "user_id": user_id,
                "full_name": user_id.title(),
                "role": "student",
                "pose": "front",
                "prototype_id": "front_01",
                "prototype_index": 1,
                "embedding_file": f"embeddings/{filename}",
                "mean_fallback": False,
            })
        manifest = {
            "schema_version": 2,
            "generation": "old-generation",
            "provenance": self.provenance,
            "embeddings": rows,
        }
        (self.old_generation / "manifest.json").write_text(
            json.dumps(manifest, indent=2),
            encoding="utf-8",
        )
        self.pointer = self.gallery_root / "current.json"
        self.pointer.write_text(
            json.dumps({"manifest": "old-generation/manifest.json"}),
            encoding="utf-8",
        )
        users = {
            "schema_version": 1,
            "users": {
                user_id: {
                    "user_id": user_id,
                    "gallery_generation": "old-generation",
                }
                for user_id in ("alpha", "bravo", "charlie")
            },
        }
        (self.enrollments_root / "users.json").write_text(
            json.dumps(users, indent=2),
            encoding="utf-8",
        )
        self.historical_session = self.enrollments_root / ("a" * 32)
        self.historical_session.mkdir()
        (self.historical_session / "session.json").write_text("historical")
        self.manager = LocalIdentityRemoval(
            self.gallery_root,
            self.enrollments_root,
        )
        self.old_pointer = self.pointer.read_bytes()
        self.old_users = (self.enrollments_root / "users.json").read_bytes()

    def tearDown(self):
        self.temp.cleanup()

    def active_gallery(self):
        return LocalGalleryProvider(self.pointer).load(self.provenance)

    def filesystem_snapshot(self):
        root = Path(self.temp.name)
        return {
            str(path.relative_to(root)): path.read_bytes()
            for path in root.rglob("*")
            if path.is_file()
        }, tuple(sorted(
            str(path.relative_to(root))
            for path in root.rglob("*")
            if path.is_dir()
        ))

    def test_remove_one_publishes_new_generation_and_updates_users(self):
        result = self.manager.remove(["bravo"])

        self.assertEqual(result.previous_generation, "old-generation")
        self.assertNotEqual(result.generation, "old-generation")
        self.assertEqual(result.removed_user_ids, ("bravo",))
        self.assertEqual(result.identity_count_after, 2)
        self.assertTrue(self.old_generation.is_dir())
        self.assertTrue(self.historical_session.is_dir())
        self.assertEqual(
            {identity.user_id for identity in self.active_gallery().identities},
            {"alpha", "charlie"},
        )
        manifest = json.loads(Path(result.manifest).read_text(encoding="utf-8"))
        self.assertEqual(manifest["provenance"], self.provenance)
        self.assertNotIn(
            "bravo",
            {row["user_id"] for row in manifest["embeddings"]},
        )
        users = json.loads(
            (self.enrollments_root / "users.json").read_text(encoding="utf-8")
        )["users"]
        self.assertEqual(set(users), {"alpha", "charlie"})

    def test_remove_multiple_preserves_all_remaining_identities(self):
        result = self.manager.remove(["alpha", "charlie"])

        self.assertEqual(result.removed_user_ids, ("alpha", "charlie"))
        self.assertEqual(result.identity_count_before, 3)
        self.assertEqual(result.identity_count_after, 1)
        gallery = LocalGalleryProvider(result.manifest).load(self.provenance)
        self.assertEqual(
            {identity.user_id for identity in gallery.identities},
            {"bravo"},
        )

    def test_unknown_identity_fails_without_partial_removal(self):
        before = self.filesystem_snapshot()

        with self.assertRaises(LocalIdentityRemovalError) as caught:
            self.manager.remove(["alpha", "missing"])

        self.assertIn("missing", str(caught.exception))
        self.assertEqual(self.filesystem_snapshot(), before)
        self.assertEqual(self.pointer.read_bytes(), self.old_pointer)

    def test_dry_run_cli_prints_plan_and_changes_nothing(self):
        before = self.filesystem_snapshot()
        output = io.StringIO()

        with redirect_stdout(output):
            exit_code = remove_main([
                "alpha",
                "missing",
                "--dry-run",
                "--gallery-root",
                str(self.gallery_root),
                "--enrollments-root",
                str(self.enrollments_root),
            ])

        text = output.getvalue()
        self.assertEqual(exit_code, 0)
        self.assertIn("Current generation: old-generation", text)
        self.assertIn("Identity count before: 3", text)
        self.assertIn("IDs requested for deletion:", text)
        self.assertIn("IDs found:\n- alpha", text)
        self.assertIn("IDs not found:\n- missing", text)
        self.assertIn("Identity count after: 2", text)
        self.assertIn("DRY RUN ONLY — no changes made.", text)
        self.assertEqual(self.filesystem_snapshot(), before)

    def test_validation_failure_never_changes_current_pointer(self):
        original_validate = self.manager._validate
        observed_pointer = []

        def fail_new_generation(manifest_path, provenance):
            if Path(manifest_path).parent.name.startswith("."):
                observed_pointer.append(self.pointer.read_bytes())
                raise ValueError("injected validation failure")
            return original_validate(manifest_path, provenance)

        with patch.object(
            self.manager,
            "_validate",
            side_effect=fail_new_generation,
        ), self.assertRaises(LocalIdentityRemovalError):
            self.manager.remove(["alpha"])

        self.assertEqual(observed_pointer, [self.old_pointer])
        self.assertEqual(self.pointer.read_bytes(), self.old_pointer)
        self.assertEqual(
            {path.name for path in self.gallery_root.iterdir() if path.is_dir()},
            {"old-generation"},
        )

    def test_metadata_failure_rolls_back_pointer_and_users(self):
        with patch.object(
            EnrollmentRepository,
            "remove_users",
            side_effect=OSError("injected users failure"),
        ), self.assertRaises(LocalIdentityRemovalError):
            self.manager.remove(["alpha"])

        self.assertEqual(self.pointer.read_bytes(), self.old_pointer)
        self.assertEqual(
            (self.enrollments_root / "users.json").read_bytes(),
            self.old_users,
        )
        self.assertEqual(
            {path.name for path in self.gallery_root.iterdir() if path.is_dir()},
            {"old-generation"},
        )

    def test_cannot_remove_every_identity(self):
        before = self.filesystem_snapshot()

        with self.assertRaises(LocalIdentityRemovalError) as caught:
            self.manager.remove(["alpha", "bravo", "charlie"])

        self.assertIn("every active identity", str(caught.exception))
        self.assertEqual(self.filesystem_snapshot(), before)


if __name__ == "__main__":
    unittest.main()
