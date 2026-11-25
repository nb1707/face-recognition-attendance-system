from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import shutil
import uuid

import numpy as np

from enrollment.errors import enrollment_error
from recognition.embeddings import normalized_embedding
from recognition.local_gallery import LocalGalleryProvider


class LocalGalleryPublisher:
    def __init__(self, output_root):
        self.output_root = Path(output_root).resolve()
        self.output_root.mkdir(parents=True, exist_ok=True)
        self.pointer = self.output_root / "current.json"

    def identity_exists(self, user_id):
        try:
            _, data = self._current_manifest()
        except FileNotFoundError:
            return False
        except (ValueError, KeyError, json.JSONDecodeError, OSError) as error:
            raise enrollment_error("STORAGE_ERROR", status_code=500,
                                   details={"reason": f"invalid current gallery: {error}"})
        return any(str(row.get("user_id")) == user_id for row in data.get("embeddings", []))

    def publish(self, session, representatives, mean_embedding, provenance):
        dimension = int(provenance["embedding_dimension"])
        generation = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ") + "-" + uuid.uuid4().hex[:8]
        temporary_generation = self.output_root / f".{generation}.tmp"
        final_generation = self.output_root / generation
        try:
            temporary_generation.mkdir(parents=False, exist_ok=False)
            rows = self._copy_forward(temporary_generation, session["user_id"], provenance)
            user_key = hashlib.sha256(session["user_id"].encode("utf-8")).hexdigest()[:24]
            pose_indices = {}
            for candidate in representatives:
                pose_indices[candidate.pose] = pose_indices.get(candidate.pose, 0) + 1
                prototype_id = f"{candidate.pose}_{pose_indices[candidate.pose]:02d}"
                relative = f"embeddings/{user_key}/{prototype_id}.npy"
                self._save_embedding(temporary_generation, relative, candidate.embedding, dimension)
                rows.append({
                    "user_id": session["user_id"], "full_name": session["full_name"],
                    "role": session["role"], "prototype_id": prototype_id,
                    "prototype_index": pose_indices[candidate.pose], "pose": candidate.pose,
                    "embedding_file": relative, "source_sample_id": candidate.sample_id,
                    "quality_score": float(candidate.quality_score), "mean_fallback": False,
                })
            mean_relative = f"embeddings/{user_key}/mean.npy"
            self._save_embedding(temporary_generation, mean_relative, mean_embedding, dimension)
            rows.append({
                "user_id": session["user_id"], "full_name": session["full_name"],
                "role": session["role"], "prototype_id": "mean", "prototype_index": 1,
                "pose": "mean", "embedding_file": mean_relative,
                "source_sample_id": None, "quality_score": None, "mean_fallback": True,
            })
            manifest = {
                "schema_version": 2, "generation": generation, "provenance": dict(provenance),
                "embeddings": rows, "published_from_session": session["session_id"],
                "published_at": datetime.now(timezone.utc).isoformat(),
            }
            manifest_path = temporary_generation / "manifest.json"
            manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
            LocalGalleryProvider(manifest_path).load(provenance)
            os.replace(temporary_generation, final_generation)
            LocalGalleryProvider(final_generation / "manifest.json").load(provenance)
            previous_pointer = self.pointer.read_bytes() if self.pointer.is_file() else None
            pointer_temp = self.output_root / f".current-{uuid.uuid4().hex}.tmp"
            try:
                pointer_temp.write_text(json.dumps({"manifest": f"{generation}/manifest.json"}), encoding="utf-8")
                os.replace(pointer_temp, self.pointer)
            finally:
                if pointer_temp.exists():
                    pointer_temp.unlink()
            return {"generation": generation, "manifest": str(final_generation / "manifest.json"),
                    "prototype_count": len(representatives), "identity_count": len({r["user_id"] for r in rows}),
                    "_previous_pointer": previous_pointer}
        except Exception as error:
            if temporary_generation.exists():
                shutil.rmtree(temporary_generation, ignore_errors=True)
            raise enrollment_error("GALLERY_PUBLISH_ERROR", status_code=500,
                                   details={"reason": f"{type(error).__name__}: {error}"})

    def rollback_pointer(self, generation, previous_pointer):
        try:
            _, current = self._current_manifest()
            if current.get("generation") != generation:
                return
            if previous_pointer is None:
                self.pointer.unlink(missing_ok=True)
                return
            temporary = self.output_root / f".rollback-{uuid.uuid4().hex}.tmp"
            try:
                temporary.write_bytes(previous_pointer)
                os.replace(temporary, self.pointer)
            finally:
                if temporary.exists():
                    temporary.unlink()
        except FileNotFoundError:
            return

    def _copy_forward(self, target, replaced_user_id, provenance):
        try:
            manifest_path, data = self._current_manifest()
        except FileNotFoundError:
            return []
        if data.get("provenance") != provenance:
            raise ValueError("current gallery provenance is incompatible")
        dimension = int(provenance["embedding_dimension"])
        rows = []
        for index, old in enumerate(data.get("embeddings", [])):
            if str(old.get("user_id")) == replaced_user_id:
                continue
            source = self._contained(manifest_path.parent, old["embedding_file"])
            vector = normalized_embedding(np.load(source, allow_pickle=False), dimension)
            pose = old["pose"]
            prototype_id = old.get("prototype_id") or ("mean" if pose == "mean" else f"{pose}_01")
            user_key = hashlib.sha256(str(old["user_id"]).encode("utf-8")).hexdigest()[:24]
            relative = f"embeddings/{user_key}/{prototype_id}.npy"
            self._save_embedding(target, relative, vector, dimension)
            row = dict(old)
            row.update({"prototype_id": prototype_id, "embedding_file": relative,
                        "prototype_index": int(old.get("prototype_index", 1))})
            rows.append(row)
        return rows

    def _current_manifest(self):
        if not self.pointer.is_file():
            raise FileNotFoundError(self.pointer)
        pointer = json.loads(self.pointer.read_text(encoding="utf-8"))
        manifest = self._contained(self.output_root, pointer["manifest"])
        return manifest, json.loads(manifest.read_text(encoding="utf-8"))

    @staticmethod
    def _save_embedding(root, relative, vector, dimension):
        path = root / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        np.save(path, normalized_embedding(vector, dimension), allow_pickle=False)

    @staticmethod
    def _contained(root, relative):
        root = Path(root).resolve()
        result = (root / relative).resolve()
        if not result.is_relative_to(root):
            raise ValueError("gallery path escape")
        return result
