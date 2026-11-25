"""Transactional local-gallery identity removal without historical deletion."""

from dataclasses import dataclass
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import shutil
import uuid

from enrollment.repository import EnrollmentRepository
from recognition.local_gallery import LocalGalleryProvider


class LocalIdentityRemovalError(RuntimeError):
    """A fail-closed local identity removal error."""


@dataclass(frozen=True)
class IdentityRemovalPlan:
    current_generation: str
    identity_count_before: int
    requested_user_ids: tuple[str, ...]
    found_user_ids: tuple[str, ...]
    not_found_user_ids: tuple[str, ...]
    identity_count_after: int


@dataclass(frozen=True)
class IdentityRemovalResult:
    previous_generation: str
    generation: str
    removed_user_ids: tuple[str, ...]
    metadata_records_removed: tuple[str, ...]
    identity_count_before: int
    identity_count_after: int
    manifest: str


class LocalIdentityRemoval:
    """Publish a new active gallery excluding selected identities."""

    def __init__(self, gallery_root, enrollments_root):
        self.gallery_root = Path(gallery_root).resolve()
        self.enrollments_root = Path(enrollments_root).resolve()
        self.pointer = self.gallery_root / "current.json"

    def plan(self, user_ids):
        requested = self._normalize_user_ids(user_ids)
        _pointer_bytes, manifest_path, manifest, gallery = self._load_active()
        active_ids = {identity.user_id for identity in gallery.identities}
        found = tuple(user_id for user_id in requested if user_id in active_ids)
        not_found = tuple(user_id for user_id in requested if user_id not in active_ids)
        return IdentityRemovalPlan(
            current_generation=str(manifest["generation"]),
            identity_count_before=len(active_ids),
            requested_user_ids=requested,
            found_user_ids=found,
            not_found_user_ids=not_found,
            identity_count_after=len(active_ids) - len(found),
        )

    def remove(self, user_ids, *, expected_generation=None):
        requested = self._normalize_user_ids(user_ids)
        pointer_snapshot, source_manifest_path, source_manifest, gallery = (
            self._load_active()
        )
        active_ids = {identity.user_id for identity in gallery.identities}
        found = tuple(user_id for user_id in requested if user_id in active_ids)
        not_found = tuple(user_id for user_id in requested if user_id not in active_ids)
        plan = IdentityRemovalPlan(
            current_generation=str(source_manifest["generation"]),
            identity_count_before=len(active_ids),
            requested_user_ids=requested,
            found_user_ids=found,
            not_found_user_ids=not_found,
            identity_count_after=len(active_ids) - len(found),
        )
        if expected_generation is not None and plan.current_generation != expected_generation:
            raise LocalIdentityRemovalError(
                "Active gallery changed after confirmation; no identities were removed."
            )
        if not_found:
            raise LocalIdentityRemovalError(
                "Unknown active-gallery user ID(s): " + ", ".join(not_found)
            )
        if not found:
            raise LocalIdentityRemovalError(
                "None of the requested user IDs exist in the active gallery."
            )
        if plan.identity_count_after == 0:
            raise LocalIdentityRemovalError(
                "Refusing to remove every active identity; no force-empty mode is available."
            )

        generation = (
            datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
            + "-"
            + uuid.uuid4().hex[:8]
        )
        temporary_generation = self.gallery_root / f".{generation}.tmp"
        final_generation = self.gallery_root / generation
        pointer_temp = self.gallery_root / f".current-{uuid.uuid4().hex}.tmp"
        repository = None
        users_snapshot = None
        users_snapshot_captured = False
        pointer_switched = False
        final_created = False
        pointer_restored = True
        try:
            temporary_generation.mkdir(parents=False, exist_ok=False)
            rows = self._copy_remaining_rows(
                source_manifest_path,
                source_manifest,
                temporary_generation,
                set(found),
            )
            manifest = {
                "schema_version": 2,
                "generation": generation,
                "provenance": source_manifest["provenance"],
                "embeddings": rows,
                "published_at": datetime.now(timezone.utc).isoformat(),
                "source_generation": plan.current_generation,
                "operation": {
                    "type": "identity_removal",
                    "removed_user_ids": list(found),
                },
            }
            temporary_manifest = temporary_generation / "manifest.json"
            temporary_manifest.write_text(
                json.dumps(manifest, indent=2, ensure_ascii=False),
                encoding="utf-8",
            )
            temporary_gallery = self._validate(
                temporary_manifest,
                source_manifest["provenance"],
            )
            if len(temporary_gallery.identities) != plan.identity_count_after:
                raise ValueError("validated identity count does not match removal plan")

            os.replace(temporary_generation, final_generation)
            final_created = True
            final_manifest = final_generation / "manifest.json"
            final_gallery = self._validate(
                final_manifest,
                source_manifest["provenance"],
            )
            if len(final_gallery.identities) != plan.identity_count_after:
                raise ValueError("final identity count does not match removal plan")

            repository = EnrollmentRepository(self.enrollments_root)
            users_snapshot = repository.snapshot_users()
            users_snapshot_captured = True
            if self.pointer.read_bytes() != pointer_snapshot:
                raise LocalIdentityRemovalError(
                    "Active gallery changed during removal; no pointer was updated."
                )
            pointer_temp.write_text(
                json.dumps({"manifest": f"{generation}/manifest.json"}),
                encoding="utf-8",
            )
            os.replace(pointer_temp, self.pointer)
            pointer_switched = True
            pointer_restored = False
            metadata_removed = repository.remove_users(found)
            return IdentityRemovalResult(
                previous_generation=plan.current_generation,
                generation=generation,
                removed_user_ids=found,
                metadata_records_removed=metadata_removed,
                identity_count_before=plan.identity_count_before,
                identity_count_after=plan.identity_count_after,
                manifest=str(final_manifest),
            )
        except Exception as error:
            rollback_errors = []
            if pointer_switched:
                try:
                    self._atomic_bytes(self.pointer, pointer_snapshot)
                    pointer_restored = True
                except Exception as rollback_error:
                    rollback_errors.append(f"pointer rollback failed: {rollback_error}")
            if repository is not None and users_snapshot_captured:
                try:
                    repository.restore_users(users_snapshot)
                except Exception as rollback_error:
                    rollback_errors.append(f"users metadata rollback failed: {rollback_error}")
            if final_created and pointer_restored and final_generation.exists():
                shutil.rmtree(final_generation, ignore_errors=True)
            detail = f"{type(error).__name__}: {error}"
            if rollback_errors:
                detail += "; " + "; ".join(rollback_errors)
            raise LocalIdentityRemovalError(detail) from error
        finally:
            if temporary_generation.exists():
                shutil.rmtree(temporary_generation, ignore_errors=True)
            pointer_temp.unlink(missing_ok=True)

    def _load_active(self):
        try:
            pointer_bytes = self.pointer.read_bytes()
            pointer = json.loads(pointer_bytes.decode("utf-8"))
            manifest_path = self._contained(
                self.gallery_root,
                pointer["manifest"],
            )
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            provenance = manifest["provenance"]
            gallery = self._validate(manifest_path, provenance)
            if gallery.diagnostics:
                raise ValueError(
                    "active gallery has invalid records: "
                    + "; ".join(gallery.diagnostics)
                )
            return pointer_bytes, manifest_path, manifest, gallery
        except LocalIdentityRemovalError:
            raise
        except Exception as error:
            raise LocalIdentityRemovalError(
                f"Active local gallery is unavailable or invalid: {error}"
            ) from error

    def _copy_remaining_rows(
        self,
        source_manifest_path,
        source_manifest,
        target,
        removed_user_ids,
    ):
        rows = []
        for index, old in enumerate(source_manifest.get("embeddings", [])):
            user_id = str(old["user_id"])
            if user_id in removed_user_ids:
                continue
            source = self._contained(
                source_manifest_path.parent,
                old["embedding_file"],
            )
            pose = old["pose"]
            prototype_id = old.get("prototype_id") or (
                "mean" if pose == "mean" else f"{pose}_01"
            )
            user_key = hashlib.sha256(user_id.encode("utf-8")).hexdigest()[:24]
            file_key = hashlib.sha256(
                f"{user_id}\0{prototype_id}\0{index}".encode("utf-8")
            ).hexdigest()[:24]
            relative = f"embeddings/{user_key}/{file_key}.npy"
            destination = self._contained(target, relative)
            destination.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(source, destination)
            row = dict(old)
            row.update({
                "prototype_id": prototype_id,
                "prototype_index": int(old.get("prototype_index", 1)),
                "embedding_file": relative,
            })
            rows.append(row)
        return rows

    @staticmethod
    def _normalize_user_ids(user_ids):
        if isinstance(user_ids, str):
            user_ids = (user_ids,)
        result = []
        for value in user_ids:
            user_id = str(value or "").strip()
            if not user_id:
                raise LocalIdentityRemovalError("User IDs must not be empty.")
            if user_id not in result:
                result.append(user_id)
        if not result:
            raise LocalIdentityRemovalError("At least one user ID is required.")
        return tuple(result)

    @staticmethod
    def _validate(manifest_path, provenance):
        return LocalGalleryProvider(manifest_path).load(provenance)

    @staticmethod
    def _contained(root, relative):
        root = Path(root).resolve()
        result = (root / relative).resolve()
        if not result.is_relative_to(root):
            raise ValueError("path escapes local gallery root")
        return result

    @staticmethod
    def _atomic_bytes(path, payload):
        temporary = path.with_name(f".{path.name}.{uuid.uuid4().hex}.tmp")
        try:
            temporary.write_bytes(payload)
            os.replace(temporary, path)
        finally:
            temporary.unlink(missing_ok=True)
