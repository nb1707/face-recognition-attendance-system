"""Read-only Supabase cache for registered AdaFace mean embeddings."""

from __future__ import annotations

import os
import logging
from dataclasses import dataclass
from io import BytesIO
from pathlib import Path
from collections.abc import Callable
from typing import Any

import numpy as np


LOGGER = logging.getLogger(__name__)

ADAFACE_MODEL_NAME = "adaface"
FACE_EMBEDDING_BUCKET = "face-embeddings"
_BACKEND_ENV_FILE = Path(__file__).resolve().parents[2] / "backend" / ".env"


class RecognitionCacheError(RuntimeError):
    """A sanitized, user-facing recognition-cache error."""


@dataclass(frozen=True)
class CachedIdentity:
    """One registered user paired with an AdaFace mean embedding."""

    user_id: str
    full_name: str
    role: str
    embedding: np.ndarray


@dataclass(frozen=True)
class RecognitionCache:
    """Immutable snapshot loaded once for a recognition worker run."""

    identities: tuple[CachedIdentity, ...]
    registered_user_count: int
    mean_embedding_record_count: int
    downloaded_embedding_count: int
    skipped_user_count: int
    skipped_embedding_count: int


def load_adaface_recognition_cache(
    status_callback: Callable[[str], None] | None = None,
) -> RecognitionCache:
    """Load users and AdaFace mean embeddings once from Supabase.

    The function intentionally performs no writes and never returns credentials.
    It is called only from the recognition worker thread.
    """
    _notify(status_callback, "Initializing Supabase client...")
    client = _create_supabase_client()

    _notify(status_callback, "Loading registered users...")

    try:
        users_response = (
            client.table("users")
            .select("user_id,full_name,role")
            .execute()
        )
    except Exception as error:
        raise RecognitionCacheError(
            "Registered users could not be loaded."
        ) from error

    users: dict[str, dict[str, Any]] = {}

    for user in users_response.data or []:
        user_id = str(user.get("user_id") or "").strip()
        if user_id:
            users[user_id] = user

    if not users:
        _print_cache_counts(
            registered_users=0,
            mean_records=0,
            downloaded=0,
            cached_identities=0,
            skipped_users=0,
            skipped_embeddings=0,
        )
        raise RecognitionCacheError("No registered users found.")

    _notify(
        status_callback,
        f"Loaded {len(users)} registered users. Loading AdaFace metadata...",
    )

    try:
        embeddings_response = (
            client.table("face_embeddings")
            .select("user_id,pose,model_name,embedding_path")
            .eq("pose", "mean")
            .eq("model_name", ADAFACE_MODEL_NAME)
            .execute()
        )
    except Exception as error:
        raise RecognitionCacheError(
            "AdaFace embedding metadata could not be loaded."
        ) from error

    embedding_records = embeddings_response.data or []
    mean_embedding_record_count = len(embedding_records)

    if mean_embedding_record_count == 0:
        _print_cache_counts(
            registered_users=len(users),
            mean_records=0,
            downloaded=0,
            cached_identities=0,
            skipped_users=len(users),
            skipped_embeddings=0,
        )
        raise RecognitionCacheError(
            "No registered AdaFace mean embeddings found. "
            "Run the existing embedding-generation script."
        )

    _notify(
        status_callback,
        f"Found {mean_embedding_record_count} AdaFace mean embeddings. "
        "Downloading registered embeddings...",
    )

    embeddings_by_user: dict[str, np.ndarray] = {}
    skipped_embedding_count = 0
    downloaded_embedding_count = 0

    for record in embedding_records:
        user_id = str(record.get("user_id") or "").strip()
        embedding_path = str(record.get("embedding_path") or "").strip()

        if not user_id or user_id not in users or not embedding_path:
            skipped_embedding_count += 1
            LOGGER.warning(
                "Skipping invalid AdaFace mean-embedding metadata record."
            )
            continue

        try:
            file_bytes = (
                client.storage
                .from_(FACE_EMBEDDING_BUCKET)
                .download(embedding_path)
            )
            embedding = np.load(
                BytesIO(file_bytes),
                allow_pickle=False,
            )
            embedding = np.asarray(
                embedding,
                dtype=np.float32,
            ).reshape(-1)

            if (
                embedding.size == 0
                or not np.all(np.isfinite(embedding))
                or float(np.linalg.norm(embedding)) == 0.0
            ):
                raise ValueError("Invalid embedding data.")

            embeddings_by_user[user_id] = embedding
            downloaded_embedding_count += 1
        except Exception as error:
            skipped_embedding_count += 1
            LOGGER.warning(
                "Skipping AdaFace mean embedding for user_id=%s; "
                "exception_type=%s",
                user_id or "--",
                type(error).__name__,
            )

    identities = tuple(
        CachedIdentity(
            user_id=user_id,
            full_name=str(user.get("full_name") or user_id),
            role=str(user.get("role") or ""),
            embedding=embeddings_by_user[user_id],
        )
        for user_id, user in users.items()
        if user_id in embeddings_by_user
    )

    if not identities:
        _print_cache_counts(
            registered_users=len(users),
            mean_records=mean_embedding_record_count,
            downloaded=downloaded_embedding_count,
            cached_identities=0,
            skipped_users=len(users),
            skipped_embeddings=skipped_embedding_count,
        )
        raise RecognitionCacheError(
            "AdaFace mean embeddings were found, but none could be "
            "downloaded or deserialized."
        )

    return RecognitionCache(
        identities=identities,
        registered_user_count=len(users),
        mean_embedding_record_count=mean_embedding_record_count,
        downloaded_embedding_count=downloaded_embedding_count,
        skipped_user_count=len(users) - len(identities),
        skipped_embedding_count=skipped_embedding_count,
    )


def _notify(
    callback: Callable[[str], None] | None,
    message: str,
) -> None:
    if callback is not None:
        callback(message)


def _print_cache_counts(
    *,
    registered_users: int,
    mean_records: int,
    downloaded: int,
    cached_identities: int,
    skipped_users: int,
    skipped_embeddings: int,
) -> None:
    print(
        "[Recognition cache] "
        f"registered_users={registered_users} "
        f"adaface_mean_records={mean_records} "
        f"downloaded={downloaded} "
        f"cached_identities={cached_identities} "
        f"skipped_users={skipped_users} "
        f"skipped_embeddings={skipped_embeddings}",
        flush=True,
    )


def _create_supabase_client():
    try:
        from dotenv import load_dotenv
        from supabase import create_client
    except ImportError as error:
        raise RecognitionCacheError(
            "Supabase recognition support is unavailable."
        ) from error

    load_dotenv(
        dotenv_path=_BACKEND_ENV_FILE,
        override=False,
    )

    supabase_url = os.getenv("SUPABASE_URL", "").strip()
    supabase_key = os.getenv("SUPABASE_KEY", "").strip()

    if not supabase_url or not supabase_key:
        raise RecognitionCacheError(
            "Supabase recognition configuration is incomplete."
        )

    try:
        return create_client(supabase_url, supabase_key)
    except Exception as error:
        raise RecognitionCacheError(
            "Supabase recognition connection could not be created."
        ) from error


__all__ = [
    "ADAFACE_MODEL_NAME",
    "CachedIdentity",
    "RecognitionCache",
    "RecognitionCacheError",
    "load_adaface_recognition_cache",
]
