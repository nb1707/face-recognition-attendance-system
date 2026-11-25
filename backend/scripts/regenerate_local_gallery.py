"""Generate a new local AdaFace gallery from existing *_aligned.jpg registrations.

Never reads legacy .npy files or contacts Supabase. Output lives in ignored storage.
Only aligned poses are eligible; ordinary padded crops are not silently substituted.
"""
import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import sys
import uuid

import cv2
import numpy as np

BACKEND = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(BACKEND))
from recognition.adaface_model import AdaFaceModel
from recognition.embeddings import normalized_embedding


def regenerate(images, output_root):
    images, output_root = Path(images).resolve(), Path(output_root).resolve()
    generation = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ") + "-" + uuid.uuid4().hex[:8]
    output = output_root / generation
    output.mkdir(parents=True, exist_ok=False)
    model = AdaFaceModel()
    provenance = model.provenance
    rows, skipped, hashes = [], [], []
    inference_count = 0
    for user in sorted(images.iterdir()):
        if not user.is_dir():
            continue
        vectors = []
        user_key = hashlib.sha256(user.name.encode()).hexdigest()[:24]
        for pose in ("front", "left", "right"):
            source = user / f"{pose}_aligned.jpg"
            if not source.is_file():
                skipped.append(dict(user_id=user.name, pose=pose, reason="missing_aligned_image"))
                continue
            try:
                image = cv2.imread(str(source))
                if image is None or image.shape[:2] != (112, 112):
                    raise ValueError("Invalid aligned registration image; expected 112x112")
                embedding = normalized_embedding(model.extract_embedding(image), 512)
                inference_count += 1
                relative = f"embeddings/{user_key}/{pose}.npy"
                (output / relative).parent.mkdir(parents=True, exist_ok=True)
                np.save(output / relative, embedding, allow_pickle=False)
                rows.append(dict(user_id=user.name, pose=pose, embedding_file=relative, mean_fallback=False))
                vectors.append(embedding)
                hashes.append(hashlib.sha256(source.read_bytes()).hexdigest())
            except (ValueError, OSError) as error:
                skipped.append(dict(user_id=user.name, pose=pose, reason=str(error)))
        if len(vectors) == 3:
            relative = f"embeddings/{user_key}/mean.npy"
            np.save(output / relative, normalized_embedding(np.mean(vectors, axis=0), 512), allow_pickle=False)
            rows.append(dict(user_id=user.name, pose="mean", embedding_file=relative, mean_fallback=True))
    if not rows:
        raise ValueError("No valid local aligned images; no current-gallery pointer was written")
    manifest = dict(schema_version=1, generation=generation, provenance=provenance,
                    embeddings=rows, registration_sha256=hashes, skipped=skipped,
                    inference_count=inference_count)
    (output / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    temporary = output_root / f"current-{uuid.uuid4().hex}.tmp"
    temporary.write_text(json.dumps({"manifest": f"{generation}/manifest.json"}), encoding="utf-8")
    temporary.replace(output_root / "current.json")
    return dict(generation=generation, identities=len({r['user_id'] for r in rows}),
                prototypes=sum(not r['mean_fallback'] for r in rows),
                means=sum(r['mean_fallback'] for r in rows), inference_count=inference_count,
                skipped_poses=len(skipped), manifest=str(output / "manifest.json"))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--images", type=Path, default=BACKEND / "storage/images")
    parser.add_argument("--output-root", type=Path, default=BACKEND / "storage/local_gallery")
    args = parser.parse_args()
    print(json.dumps(regenerate(args.images, args.output_root), indent=2))
