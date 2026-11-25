"""Explicit local AdaFace gallery adapter. Cloud providers are deferred."""
from dataclasses import dataclass
from pathlib import Path
import sys

BACKEND = Path(__file__).resolve().parents[2] / "backend"
if str(BACKEND) not in sys.path:
    sys.path.insert(0, str(BACKEND))
from recognition.local_gallery import Gallery, Identity as CachedIdentity, configured_gallery_provider
from recognition.adaface_loader import local_model_provenance

ADAFACE_MODEL_NAME = "adaface"


class RecognitionCacheError(RuntimeError):
    pass


@dataclass(frozen=True)
class RecognitionCache:
    gallery: Gallery

    @property
    def identities(self):
        return self.gallery.identities

    @property
    def registered_user_count(self):
        return len(self.identities)

    @property
    def prototype_record_count(self):
        return sum(len(identity.prototypes) for identity in self.identities)

    @property
    def skipped_user_count(self):
        return 0

    @property
    def skipped_embedding_count(self):
        return len(self.gallery.diagnostics)


def load_adaface_recognition_cache(status_callback=None, expected_provenance=None):
    if status_callback:
        status_callback("Loading local AdaFace gallery...")
    try:
        gallery = configured_gallery_provider().load(expected_provenance or local_model_provenance())
    except (OSError, ValueError, KeyError) as error:
        raise RecognitionCacheError(f"Local gallery unavailable: {error}") from error
    return RecognitionCache(gallery)
