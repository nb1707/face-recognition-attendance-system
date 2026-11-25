# Local guided face enrollment

## Purpose and status

The web enrollment at `/app` is local-first. It does not query or mutate Supabase and does not use Hikvision. The server owns session progress, accepted samples, finalization state, and local gallery publication.

## UI workflow

1. Enter a safe user ID, full name, and role, then create a session.
2. Start any browser-supported camera. Camera resolution is not prescribed; acceptance depends on measured face quality.
3. Follow the server-recommended poses: front 5, left 3, right 3, up 2, and down 2.
4. Capture manually or use bounded Guided Scan. Guided Scan retries rejected samples, advances only when the server quota is met, and pauses after repeated rejection.
5. Review grouped thumbnails. Delete a sample to reopen that pose for retake.
6. Explicitly finalize. Completion is never triggered merely by reaching 15 samples.

The browser stores only the current `session_id` in local storage. Reload reconstructs the UI through `GET /enrollments/{session_id}`. The server remains authoritative.

## API

- `POST /enrollments` starts a session. JSON fields are `user_id`, `full_name`, `role`, and optional `allow_reenroll`.
- `GET /enrollments/{session_id}` returns metadata, targets, progress, samples, rejection summary, and finalize eligibility.
- `POST /enrollments/{session_id}/samples` accepts multipart `pose` and `image`.
- `GET /enrollments/{session_id}/samples/{sample_id}/preview` safely serves that session-owned crop.
- `DELETE /enrollments/{session_id}/samples/{sample_id}` deletes one accepted sample and updates progress.
- `POST /enrollments/{session_id}/finalize` creates embeddings and publishes a local gallery generation.
- `DELETE /enrollments/{session_id}` cancels only an unfinished session.

Validation rejections use HTTP 400, unknown resources use 404, state conflicts use 409, schema/identity validation uses 422, and unexpected generation/storage failures use 500. Responses include stable `error_code` values; the UI does not infer behavior from message text.

## Session and storage

Sessions use server-generated UUIDs and live under:

```text
backend/storage/enrollments/
  users.json
  <session_id>/
    session.json
    samples/
      <sample_id>/
        crop.jpg
        aligned.jpg
        metadata.json
        embedding.npy       # created/cached during finalize
```

User IDs are validated and never used as session directory names. Repository reads are containment-checked and JSON/binary updates use temporary files plus atomic replacement. Rejected images are not stored; only a bounded rejection summary is kept.

Local `users.json` records user metadata, the latest completed session, and active gallery generation. A completed identity causes `USER_ALREADY_ENROLLED`; the UI can explicitly request a new version. The old identity remains active until the replacement finalizes and publishes successfully.

## Quality and pose policy

Enrollment measurement and policy are separate modules. The baseline preserves the old web checks:

- sharpness (whole-frame Laplacian variance) at least 50;
- brightness in `[40, 220]`;
- face area ratio at least 0.05;
- horizontal center offset at most 0.18 frame widths;
- vertical center offset at most 0.20 frame heights.

An explicit detector-confidence minimum of 0.50 is **PROVISIONAL**. It is a conservative development value, not a production-calibrated threshold. The API returns detector confidence and all quality measurements.

Landmarks must be exactly five finite 2D points, with non-degenerate eye, eye-to-mouth, and mouth geometry. Front/left/right retain the horizontal nose-to-eye ratio. Up/down use the vertical nose position divided by the eye-to-mouth vertical distance. This is a deterministic landmark-geometry heuristic, not pitch degrees. Thresholds are centralized in `enrollment/config.py`; up/down ranges are provisional pending enrollment-set calibration.

## Duplicate rejection

The aligned 112×112 pixel array is SHA-256 fingerprinted before JPEG storage. An equal fingerprint is `EXACT_DUPLICATE`. Otherwise the server compares the new aligned pixels to decoded accepted aligned samples. Normalized mean absolute pixel difference at or below `0.012` is `NEAR_DUPLICATE`. The threshold is intentionally narrow so natural movement is accepted while re-encoded nearly identical frames are rejected.

## Finalization and embeddings

Finalize verifies all quotas and accepted files, locks capture mutation, then lazily loads the existing AdaFace adapter. Every accepted aligned sample receives one validated 512-dimensional, finite, non-zero, L2-normalized embedding. A cached embedding is reused only when model provenance and aligned-file SHA-256 match.

Representative counts are centralized development policy: front 2, left 1, right 1, up 1, down 1. The highest-quality sample is selected first per pose. Additional representatives maximize a deterministic score containing 70% quality and 30% embedding diversity. Published rows retain `source_sample_id`, `quality_score`, explicit `pose`, `prototype_id`, and `prototype_index`.

The mean fallback is:

```text
normalize(average(selected representative embeddings))
```

It is marked `mean_fallback: true`. The recognition matcher continues to compare pose prototypes first and uses mean only if an identity has no valid pose prototype.

## Gallery schema and atomic publication

New manifests use schema version 2 and permit multiple explicit prototypes per pose. The loader remains compatible with schema version 1 rows (`front`, `left`, `right`, `mean`). It also accepts `up` and `down` without changing recognition thresholds or the max-cosine-per-identity rule.

Finalization reads the current manifest, validates provenance, copies every identity except the user being re-enrolled into a new immutable generation, then adds the replacement identity. It validates the temporary and final manifests through `LocalGalleryProvider`. Only afterward does it atomically replace `current.json`. Any earlier failure leaves the previous pointer intact and accepted session samples available for retry.

## Removing identities from the active local gallery

Use the local-only removal command from the repository root. Always inspect a dry run first:

```powershell
python -m backend.scripts.remove_local_identities demo_student_001 demo_student_002 --dry-run
python -m backend.scripts.remove_local_identities demo_student_001 demo_student_002
```

The live command prints the same plan and requires interactive confirmation. `--yes` is available for an already-reviewed non-interactive invocation. Unknown IDs make a live multi-user removal fail as a whole. Removing every active identity is not supported; there is deliberately no force-empty mode.

Removal never edits an active manifest or embedding file in place. It copies all remaining identities into a new generation, preserves the active provenance object, validates both the temporary and final manifest through `LocalGalleryProvider`, and atomically replaces `current.json`. The previous generation remains intact for rollback.

The corresponding records are removed from `backend/storage/enrollments/users.json` after the new pointer is installed. If that metadata update fails, the pointer is atomically rolled back and the failed new generation is removed. Historical enrollment session directories, old gallery generations, trial logs, and diagnostic logs are retained.
