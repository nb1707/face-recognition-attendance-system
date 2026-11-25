# Offline recognition: Phase 0–7

Historical Phase 0–7 report. See [Phase 7.1 corrections](phase71_stabilization.md)
for current lifecycle, Top-K refresh, enrollment-aware metrics and evaluation CLI.
The historical taxonomy/results below are retained as the original experiment,
not the corrected deployment-oriented metrics.

This implements the desktop AdaFace path and local evaluation tools. The legacy
FastAPI/cloud recognition path has not been migrated. Do not compare its old
AdaFace galleries with the corrected RGB query adapter. Phase 8 remains deferred.

## Run locally

Use the existing project virtual environment; no packages were installed for this
work. Run commands from the repository root:

```powershell
$env:RECOGNITION_GALLERY_PROVIDER = "local"
$env:RECOGNITION_LOCAL_GALLERY_PATH = "backend/storage/local_gallery/current.json"
$env:RECOGNITION_CONFIG_PATH = "backend/recognition/offline_development.json"
$env:RECOGNITION_WEBCAM_INDEX = "0"
.venv/Scripts/python.exe -B -m desktop
```

The Recognition view uses the configured local webcam. Its session, trial and
confirmed attendance journals are local JSONL under
`backend/storage/offline_sessions`. The existing Detection view's camera setup
was not redesigned. This work did not open any live camera or RTSP connection.

The detector uses the InsightFace `buffalo_l` pack and requires
`~/.insightface/models/buffalo_l/det_10g.onnx`. The installed InsightFace
implementation downloads the named pack when the entire `buffalo_l` directory
is absent. If that directory already exists but is incomplete or lacks
`det_10g.onnx`, the availability check does not repair it and detector
initialisation fails. Provision and verify the complete pack before an offline
demonstration; see `docs/model_setup.md`. AdaFace loads local IR101 source and
weights only. Unsupported gallery providers fail explicitly.

```powershell
# Only when creating a new gallery generation; never uploads or overwrites old generations.
.venv/Scripts/python.exe -B backend/scripts/regenerate_local_gallery.py

# Output directories must be new, protecting previous experiment results.
.venv/Scripts/python.exe -B backend/scripts/offline_benchmark.py --output backend/storage/offline_evaluation/my-run
.venv/Scripts/python.exe -B backend/scripts/calibrate_offline.py --records backend/storage/offline_evaluation/my-run/raw_scores.jsonl --output backend/storage/offline_evaluation/my-run/calibration.json
.venv/Scripts/python.exe -B backend/scripts/replay_offline.py --output backend/storage/offline_evaluation/my-replay --max-videos 2 --max-samples 60 --stride 5
.venv/Scripts/python.exe -B -m unittest discover -s tests -v
```

Replay accepts `--video <local-file>` to choose a particular video. It processes
bounded samples with the existing tiled detector and shared coordinator. Static
image benchmarking uses the existing detector in full-frame mode, rejects
multiple-face images as ambiguous, and measures Gate A only. It cannot calibrate
temporal confirmation or establish classroom performance. Benchmark/replay
entrypoints block Python socket connections during execution.

## Modules and contracts

| Responsibility | Module |
|---|---|
| BGR input, RGB tensor conversion | `backend/recognition/adaface_preprocessing.py` |
| Finite, dimension-correct normalized embeddings | `embeddings.py` |
| Explicit manifest/provider boundary | `local_gallery.py` |
| Stateless max-prototype identity ranking | `identity_matcher.py` |
| Crop measurements and separate quality policy | `quality.py` |
| Temporary deterministic IoU tracks | `tracker.py` |
| Bounded Top-K observations and embedding cache | `evidence.py` |
| Confidence and temporal gates | `temporal.py` |
| Confirmed-only event and local dedup sink | `events.py` |
| Thread-owned orchestration | `coordinator.py` |
| Configurable provisional operating values | `offline_config.py`, `offline_development.json` |
| Saved-score metrics, sweeps and Pareto candidates | `offline_evaluation.py` |

### AdaFace and gallery provenance

The actual local `pretrained_model/model.yaml` specifies RGB. The local model
forward does not flip the incoming channels. The previous application adapter
passed BGR unchanged. **ADAFACE_GALLERY_REGEN_REQUIRED = YES.**

`AdaFaceModel.extract_embedding` receives uint8 OpenCV BGR, resizes to 112×112,
converts internally to RGB, maps channels to float32 [-1,1], and returns a finite
L2-normalized 512-vector. Callers must not manually flip channels.

Generation `20260925T091732Z-e3df6674` contains 11 identities, 33 front/left/right
prototypes and 11 optional normalized mean fallbacks. Regeneration used 33 local
aligned registration images and made 33 AdaFace inferences; 60 missing pose
files were reported. Old `.npy` files were not reused or overwritten.

Manifest provenance records model name, pinned artifact revision
`f2b38d9e24bfe301490d8dd081d8924b102333dd`, checkpoint SHA256,
dimension, preprocessing version `adaface-rgb112-v1`, generation, pose,
identity, relative embedding path, fallback flag, and registration image hashes.
The checkpoint digest is
`7a3341c3afc507fd6f50345638d2f3ef2f0e931d5b4f5aba60e15709853fcf5e`.
In-memory models retain their loading provenance; worker galleries must match
that provenance. Restart the application after replacing model artifacts.

The loader rejects incompatible manifests and path traversal, skips corrupt
prototypes with diagnostics, and fails if no valid identities remain. Missing
display names use local user IDs; no names or metadata are fetched from cloud.

### Identity ranking

For each distinct user, use valid pose prototypes if any exist. Use a valid mean
only when no valid pose survives. Score is the maximum prototype cosine, never
the average of pose similarities. Retain the winning pose. Ties use lexical
user ID, then front/left/right/mean order.

Top1 and Top2 are distinct identities; margin is Top1 score minus Top2 score.
Zero identities returns no candidates; one identity returns no Top2 and no
margin. The matcher does not fabricate competitors or decide acceptance.
Cosine scores and the quality score are not probabilities.

### Quality, tracking and evidence

Measurements use the clipped face bbox, resized to 112×112 grayscale for
Laplacian variance and brightness. Record face dimensions/area ratio, detector
confidence, nose offset projected along the eye line, and absolute eye-line
roll. Validate five finite landmarks and basic geometry. This is a geometric
heuristic, not calibrated yaw or a learned quality estimator.

Provisional defaults: minimum side 32 px, confidence 0.5, Laplacian variance 20,
brightness 35–225, nose offset ≤0.65, roll ≤40°. The ranking score is the weighted
mean of five clipped [0,1] components:

1. detector confidence;
2. sharpness / 200;
3. minimum face side / 112;
4. `1 − max(nose_offset / 0.65, roll / 40)`;
5. triangular lighting fitness centered at 128 and zero at brightness bounds.

All constants and component weights are configurable. Defaults are development
values, not optimized classroom thresholds.

Tracking consumes existing boxes only. It retains a track only for an
unambiguous mutual one-to-one IoU association (default IoU ≥0.3). Multiple
eligible associations retire affected tracks and their evidence. Defaults are
TTL 2 seconds, at most 2 misses, at most 64 tracks. This favors fragmentation
over evidence contamination and does not establish robust occlusion handling.

Capture stamps frames with monotonic acquisition time; detection and recognition
propagate it unchanged. Replay uses frame index/source FPS. Processing completion
time never substitutes for acquisition time. Duplicate/out-of-order inputs
cannot vote. Reconnect, session restart, gallery generation/provenance changes,
expiry and ambiguous association clear state. A missing or low-quality face
also breaks its confirmation evidence. Track IDs are not student IDs.

Each track retains at most K=3 quality-passed aligned crops and cached embeddings,
with maximum age 5 seconds. Highest quality wins; ties favor newer sequences.
Each newly selected observation runs AdaFace once. Rejected/unselected or
duplicate observations do not run inference. Eviction releases crop and vector.

`E_track = normalize(sum(quality_score_i * normalized_embedding_i))`.
Weights must be positive and finite; dimensions, vectors and near-zero sums are
validated. Default retained crop/vector payload is at most
`64 * 3 * (112*112*3 + 512*4) = 7,618,560 bytes`, plus bounded Python metadata and
at most 256 temporal votes per track. Worker queues retain only latest/pending
frames, not an unlimited full-frame history.

### Confirmation and attendance

Gate A requires Top1 ≥T and margin ≥M. If Top2 is unavailable, the default is
fail closed; the policy is explicit and configurable. Checked-in development
values remain T=0.4 and M=0.05. Calibration never rewrites runtime configuration.

Gate B requires at least three distinct selected, quality-passed observations,
three fresh candidate evaluations agreeing on identity, and at least 0.5 seconds
of acquisition time within a 5-second window. Re-evaluating an unchanged
aggregate cannot add a vote. A new observation whose Top1 differs from the
current candidate clears previous candidate embeddings and votes before the
new identity can build a streak. Confidence failure resets the streak.

Only `ConfirmedIdentityEvent` reaches the attendance sink. It includes logical
session, gallery generation, temporary track, user, Top1/Top2 scores, margin,
valid observation/attempt counts, elapsed confirmation time, selected count,
frame sequence and acquisition time. Deduplication uses `(session_id, user_id)`
across track changes and JSONL reloads. The sink is single-owner local storage,
not a concurrent database. Sessions/trials remain diagnostic journals; an
unconfirmed trial is not attendance.

Recognition results expose source frame/time/stream, raw quality/ranking/gate
diagnostics and detection processing time. Desktop overlays use the matching
source frame and expire after one second; trial capture rejects stale and
multiple-face results. Detailed statuses include no face, invalid observation,
low quality, below threshold, ambiguous identity, missing margin, temporal
pending and confirmed identity.

## Evaluation interpretation

Saved JSONL contains expected label, source hash/split, gallery availability,
detection result, quality measurements/decision, raw Top1/Top2/winning poses,
margin, timings and rejection/error categories. Unlabeled root screenshots,
exact registration overlap and byte-identical duplicates are excluded explicitly.
Known identities absent from the local gallery remain known labels; they are
never silently reclassified as unknown.

The sweep has 325 operating points: T=0.200…0.800 in 0.025 steps and M=0…0.300
in 0.025 steps. Report end-to-end, recognition-conditional and enrolled-only
conditional metrics. FAR = accepted unknown / unknown; FRR = rejected known /
known; wrong identity rate = known accepted as another user / known; known
identity accuracy = correctly accepted known / known. Accuracy includes correct
unknown rejection. Macro F1 averages the union of true and predicted labels,
including unknown. Undefined denominators are null.

Pareto candidates minimize false accepts, wrong-known acceptances and known
rejections jointly; equal empirical count triples retain the least restrictive
lexicographic (T,M). This is an exploratory tradeoff table, not a claim that the
lowest threshold is optimal. No target risk policy was supplied. A detector or
quality gate rejecting every unknown does not validate the matcher’s FAR.

Exact hash checks cannot detect re-encoded/cropped registration leakage or
adjacent frames. Dataset capture/session provenance is unavailable, so directory
names alone do not establish an independent calibration/test split. Static
images do not calibrate Gate B; unlabeled replay does not establish accuracy.

## Measured results, 2026-09-25

The final artifacts are `backend/storage/offline_evaluation/run-20260925-final/`
(`raw_scores.jsonl`, `benchmark_summary.json`, `calibration.json`) and
`backend/storage/offline_evaluation/replay-20260925/` (per-frame JSONL and
`replay_summary.json`). The earlier `run-20260925` is superseded: it exposed
unlabeled root screenshots being incorrectly treated as labels; discovery was
fixed, covered by a regression test, and the benchmark rerun.

Dataset: `backend/evaluation_dataset`. Discovered 343 images: 281 known images
across 15 populated labels, 36 unknown images and 26 unlabeled root screenshots.
Excluded the 26 screenshots and 13 byte-identical unknown duplicates. No exact
registration-image hash overlap was found. Empty calibration/test directories
do not provide an independent split. Final denominators: 304 samples, comprising
281 known and 23 unique unknown. Eighty known samples refer to identities absent from the
11-identity local gallery. They are retained in end-to-end metrics.

Of 304 included samples: 166 no-face, 114 quality-rejected, 16 invalid observations,
8 recognition-evaluable (all known, seven enrolled). No unknown sample reaches
the matcher, so recognition-conditional FAR is undefined.

At the unchanged development point T=0.4, M=0.05:

| Metric | End-to-end, n=304 | Recognition-conditional, n=8 |
|---|---:|---:|
| FAR | 0/23 = 0%* | undefined (0 unknown) |
| FRR | 274/281 = 97.509% | 1/8 = 12.5% |
| Wrong Identity Rate | 0% | 0% |
| Known Identity Accuracy | 7/281 = 2.491% | 7/8 = 87.5% |
| Accuracy | 30/304 = 9.868% | 87.5% |
| Macro F1 | 0.046295 | 0.600000 |
| Accepted / rejected | 7 / 297 | 7 / 1 |

*The end-to-end FAR of zero is caused by upstream rejection of all unknown
samples. It is not validation of recognition FAR. The enrolled-only conditional
subset is 7/7 correct, but far too small and insufficiently independent to
support a generalization claim.

The 325-point sweep yields two representative Pareto count triples:

| T | M | False accepts | Wrong-known accepts | Known rejections | Correct known |
|---:|---:|---:|---:|---:|---:|
| 0.200 | 0.000 | 0 | 1 | 273 | 7 |
| 0.200 | 0.050 | 0 | 0 | 274 | 7 |

The second trades one wrong-known acceptance for one rejection. T=0.4/M=0.05
has the same observed counts as the second representative. **No production
operating point is recommended from this dataset.** Runtime settings were not
changed by calibration; labeled, enrolled, independent known/unknown observations
that actually pass detection/quality are needed before selecting a risk policy.

Replay used two distinct video hashes, each 60 samples at stride 5, covering
approximately the first 14.73 source seconds. These are stored recordings, not
live Hikvision validation. No labels were available.

| Video identifier | Tracks created / maximum active | Quality / invalid / below T | AdaFace calls | Events | Mean / p95 processing |
|---|---:|---:|---:|---:|---:|
| Video 1 | 83 / 32 | 1167 / 252 / 3 | 3 | 0 | 2858 / 5737 ms |
| Video 2 | 106 / 26 | 935 / 74 / 0 | 0 | 0 | 2000 / 3396 ms |

Counts are per detected face observation, not per video frame. No confirmed
identity was observed; confirmation latency is unavailable. Zero measured
candidate changes is not evidence of stable identity tracking: almost every
observation was rejected upstream, and no ground-truth track labels exist.
Track creation indicates substantial fragmentation. The processing timings are
bounded CPU replay measurements, affected by system load; they do not establish
realtime throughput. After each decision retained evidence was zero because
quality/confidence failed; this is not a measurement of transient peak allocation.

Gallery matching uses three active prototypes per user, 33 cosine comparisons
per matcher query. The coordinator additionally scores a new observation for
identity-switch protection, hence up to 66 comparisons per selected update.
The final static benchmark made eight AdaFace calls in 149.0 seconds overall;
mean measured AdaFace extraction time was 334.7 ms on CPU. Regeneration made 33
calls, the superseded exploratory benchmark 11, and video replay three. Selected
frame cache behavior and confirmation success are separately verified by
synthetic integration tests; they were not demonstrated as successful attendance
on these unlabeled videos.

## Repository safety and deferred work

Validation: 56 tests passed, zero failed, zero skipped. AST syntax checks passed
for all 77 Python files under recognition, scripts, desktop and tests. Test files:
`test_phase0.py` (6), `test_matching.py` (9), `test_quality.py` (7),
`test_tracking.py` (8), `test_evidence.py` (6), `test_confirmation.py` (10),
`test_offline_integration.py` (5), `test_evaluation.py` (5).
Checkpoint tests were run while progressing through phases, followed by full
regression. No live webcam/Tkinter interaction or Phase 8 camera/cloud checks
were run; offline worker integration and bounded video replay were exercised.

Files created by this task:

- `backend/recognition/`: `adaface_preprocessing.py`, `embeddings.py`,
  `local_gallery.py`, `identity_matcher.py`, `status.py`, `quality.py`,
  `tracker.py`, `evidence.py`, `temporal.py`, `events.py`, `coordinator.py`,
  `offline_config.py`, `offline_development.json`, `offline_runtime.py`,
  `offline_evaluation.py`.
- `backend/scripts/`: `regenerate_local_gallery.py`, `offline_benchmark.py`,
  `calibrate_offline.py`, `replay_offline.py`.
- `desktop/session/local_repository.py`, the eight test files above, and this doc.

Existing files edited by this task:

- `backend/recognition/adaface_loader.py`, `adaface_model.py`.
- `desktop/app.py`, `desktop/camera/worker.py`, `desktop/detection/worker.py`,
  `desktop/recognition/cache.py`, `desktop/recognition/worker.py`.
- Pre-existing untracked `desktop/session/models.py`, `desktop/session/service.py`.

Final `git diff --check` passed. The tracked diff against HEAD reports 12 files,
999 insertions and 699 deletions; it includes pre-existing work and does not
include untracked additions. No unexpected non-ignored files over 1 MiB were
found. Full final status/stat are saved separately with the generated evaluation
artifacts. Nothing was staged, committed or pushed.

Initial branch: `tkinter-gui`. The tree already contained eight tracked modified
files and untracked SQL/session/logging/audit work. Before editing, the tracked
binary diff and status were saved outside the repository at
`<temporary-baseline-directory>`.
Current working-tree contents were used as the base. No reset, checkout, clean,
merge, commit or push was performed. `.env`, legacy cloud thresholds and
unrelated pre-existing UI/SQL work were not changed by this task.

SUPABASE ACCESSED = NO<br>
SUPABASE MUTATED = NO<br>
CLOUD EMBEDDINGS UPDATED = NO<br>
CLOUD ATTENDANCE WRITTEN = NO

Generated gallery/results stay in ignored `backend/storage/`; datasets, images,
videos and weights were not copied into source or staged. No packages were
installed, no model fine-tuning, second detector, multi-camera fusion, cross-camera
ReID or deep face-quality model was introduced.

Phase 8 still needs live classroom face-size distributions, compression/motion
blur evaluation, occlusion/track-switch labels, realtime throughput, real unknown
FAR, confirmation latency, restored cloud attendance, gallery synchronization,
and cloud session/trial logging. It has not started.
