"""Explicit local gallery IO and provenance boundary. Never imports Supabase."""
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol
import json
import math
import os

import numpy as np
from recognition.embeddings import normalized_embedding

POSE_ORDER = {"front": 0, "left": 1, "right": 2, "up": 3, "down": 4, "mean": 5}
DEFAULT_GALLERY = Path(__file__).resolve().parents[1] / "storage/local_gallery/current.json"


@dataclass(frozen=True)
class Prototype:
    pose: str
    embedding: np.ndarray
    prototype_id: str | None = None
    source_sample_id: str | None = None
    quality_score: float | None = None


@dataclass(frozen=True)
class Identity:
    user_id: str
    full_name: str
    role: str
    prototypes: tuple[Prototype, ...]


@dataclass(frozen=True)
class Gallery:
    generation: str
    provenance: dict
    identities: tuple[Identity, ...]
    diagnostics: tuple[str, ...] = ()


class GalleryProvider(Protocol):
    def load(self, expected_provenance: dict) -> Gallery: ...


class LocalGalleryProvider:
    def __init__(self, path):
        self.path = Path(path).expanduser().resolve()

    def load(self, expected_provenance):
        manifest_path = self.path
        data = json.loads(manifest_path.read_text(encoding="utf-8"))
        if "manifest" in data:
            manifest_path = self._contained(manifest_path.parent, data["manifest"])
            data = json.loads(manifest_path.read_text(encoding="utf-8"))
        schema_version = data.get("schema_version")
        if schema_version not in {1, 2} or not data.get("generation"):
            raise ValueError("Invalid gallery manifest version/generation")
        if data.get("provenance") != expected_provenance:
            raise ValueError("Incompatible gallery provenance; regenerate with this local model/preprocessing")
        grouped, metadata, diagnostics, seen = {}, {}, [], set()
        dimension = expected_provenance["embedding_dimension"]
        for index, row in enumerate(data.get("embeddings", [])):
            try:
                user_id, pose = str(row["user_id"]).strip(), row["pose"]
                if not user_id or pose not in POSE_ORDER:
                    raise ValueError("invalid user_id/pose")
                if bool(row.get("mean_fallback")) != (pose == "mean"):
                    raise ValueError("invalid fallback flag")
                prototype_id = row.get("prototype_id") or ("mean" if pose == "mean" else f"{pose}_01")
                if not isinstance(prototype_id, str) or not prototype_id:
                    raise ValueError("invalid prototype_id")
                if (user_id, prototype_id) in seen:
                    raise ValueError("duplicate user/prototype")
                path = self._contained(manifest_path.parent, row["embedding_file"])
                vector = normalized_embedding(np.load(path, allow_pickle=False), dimension)
                quality = row.get("quality_score")
                if quality is not None:
                    quality = float(quality)
                    if not math.isfinite(quality) or not 0 <= quality <= 1:
                        raise ValueError("invalid quality_score")
                grouped.setdefault(user_id, []).append(Prototype(
                    pose, vector, prototype_id, row.get("source_sample_id"), quality))
                metadata[user_id] = (str(row.get("full_name") or user_id), str(row.get("role") or "student"))
                seen.add((user_id, prototype_id))
            except (ValueError, KeyError, TypeError, OSError, EOFError) as error:
                diagnostics.append(f"record {index}: {type(error).__name__}: {error}")
        identities = tuple(Identity(key, *metadata[key], tuple(sorted(
            grouped[key], key=lambda p: (POSE_ORDER[p.pose], p.prototype_id or ""))))
                           for key in sorted(grouped))
        if not identities:
            raise ValueError("No valid local identities; " + "; ".join(diagnostics))
        return Gallery(data["generation"], dict(data["provenance"]), identities, tuple(diagnostics))

    @staticmethod
    def _contained(root, relative):
        root = root.resolve()
        result = (root / relative).resolve()
        if not result.is_relative_to(root):
            raise ValueError("Gallery path escapes manifest directory")
        return result


def configured_gallery_provider():
    name = os.getenv("RECOGNITION_GALLERY_PROVIDER", "local").strip().lower()
    if name != "local":
        raise ValueError("This offline runtime supports provider=local only; cloud integration is deferred")
    return LocalGalleryProvider(os.getenv("RECOGNITION_LOCAL_GALLERY_PATH", str(DEFAULT_GALLERY)))
