# Phase 7.1 stabilization and evaluation correction

Phase 8 live-camera validation has not started. This document supersedes the
Phase 0–7 report's short-miss, Top-K, confirmed-state and enrollment metric policies.
The AdaFace BGR→RGB contract, model revision, preprocessing version, 512 dimensions,
gallery generation, max-pose scoring, distinct Top1/Top2, alignment and detector
configuration are unchanged. No gallery regeneration or cloud operation occurred.

## Runtime policy

The regression suite was added before fixing the reviewed lifecycle issues:
eight initial scenarios produced six failures and two passes. Failures reproduced
short-miss deletion, the early Top-K duration stall, confirmed-state flicker on
non-selected/duplicate/poor frames, and repeated calls to the event sink.

Recognition state now follows tracker retirement, not visibility. A short miss
retains crops/vectors and pending votes while the track remains alive. Observation
age (5 seconds) and vote window (5 seconds) still expire naturally. TTL (2 seconds),
miss count (>2), ambiguity retirement, stream/reconnect/session reset and gallery
or model provenance change clear state. Mismatched model/gallery provenance fails
closed. Missing or poor observations add no identity votes.

Top-K keeps quality ranking and refreshes a recent representative slot when the
newest retained observation is at least `recent_refresh_seconds` old (default
0.25 seconds). On refresh it keeps the best K−1 other observations plus the new
quality-passed observation. Between refreshes, lower-quality frames can still be
unselected. Equal scores prefer newer sequences. K remains 3; count/duration
requirements remain three observations and 0.5 seconds. Configuration overrides
can set `evidence.recent_refresh_seconds`; it must be positive and no greater than
the evidence age. This is deterministic and provisional, not video-calibrated.

The exact 1.00/0.99/0.98 quality sequence at 0/0.1/0.2 seconds now admits a fresh
lower-quality observation at 0.5 seconds. Confirmation advances only after that
new observation is embedded and recognized. Reusing the same aggregate or a
duplicate frame does not vote. Embeddings are cached; failed embeddings are
removed so later aggregates cannot repeatedly infer the same failed observation.

Confirmed identity snapshots are separate from event emission. Non-selected,
duplicate, poor-quality or weak/ambiguous observations preserve the existing
confirmed identity while the tracker remains alive. A new **quality-passed,
embedded observation passing the unchanged Gate A for a different identity**
is a strong contradiction: previous vectors, votes and confirmed identity reset,
and the new identity starts with only its own current evidence. Weak contradictory
scores do not switch identity or add a vote. Each fresh vote requires both the
new observation and aggregate to pass Gate A and agree on identity.

`FaceDecision` distinguishes `confirmed`, `confirmation_transition`,
`event_emitted`, `state_from_history`, `confirmed_since`, `observation_reason`,
`observation_match` and `aggregate_match`. `match` on a sticky confirmed result
describes the confirmed identity snapshot, not a new inference. Only a transition
calls the confirmed-event sink; sink deduplication still uses (session, user).
Reconfirmation after a reset may create a transition, but cannot duplicate that
user's attendance in the same logical session.

State remains bounded: 64 tracks, K=3 crop/vector pairs per track, 256 temporal
votes per track, one confirmed snapshot per live track, and one latest-result
snapshot. Crop/vector payload bound is 7,618,560 bytes plus bounded metadata.
Tracker lifecycle diagnostics contain only the latest update, not a lifetime
history. Offline replay history is separately capped by max_samples × max_tracks.
Desktop/replay JSONL sinks stream the existing journal and retain dedup keys only
for their current logical session; old sessions no longer accumulate in runtime
dedup memory. Within a session, unique emitted identities are bounded by the
loaded gallery. Persistent journal size on disk is separate from track memory.

## Evaluation contracts

Enrollment is derived from the actual loaded local gallery on each run:

- `enrolled_known`: person label exists in gallery.
- `unenrolled_impostor`: labeled person absent from gallery.
- `explicit_unknown`: explicit unknown folder.

The original person label is retained. Only the open-set truth for the latter two
groups becomes rejection. An unenrolled person accepted as any registered user
is a false accept, never a wrong-known identity. FRR and Wrong Identity Rate use
only enrolled-known samples. Reports include both impostor subgroups and combined
FAR, with total population, population reaching the matcher, and accepted counts.
Zero denominators are null. Accuracy/Macro-F1 are secondary diagnostics.

Two independent evaluation dimensions are exposed:

| Dimension | Modes |
|---|---|
| Detection | `full_frame` diagnostic, `desktop_tiled` deployment-equivalent |
| Evaluation | `matcher_diagnostic`, `production_simulation`, or `both` |

Both detection modes use the existing shared detector. Desktop, replay and the
tiled benchmark call `recognition.detection_modes.detect_for_mode`, which forwards
the existing CLASSROOM_ROI and tile/overlap/NMS config without changing values.
No second detector, coordinate restoration or NMS implementation was added.

Matcher diagnostic collects raw scores for a single face with valid landmarks,
successful alignment and embedding, even when provisional quality policy rejects
it. It never produces attendance or represents production acceptance. Production
simulation requires quality pass before its matcher/Gate A decision. In `both`
collection mode, one diagnostic inference supplies saved raw scores; the
production view masks quality-rejected observations. A production-only run skips
those inferences, and its scores cannot reconstruct matcher diagnostics.

Multiple-face static images are rejected because no target face annotation is
available. Invalid landmarks/alignment are not bypassed by diagnostic mode.
Full-frame versus tiled comparisons must therefore not infer better identity
performance from more detections alone. Static simulation does not calibrate
tracking or Gate B.

Quality reports count no-face, invalid landmarks, alignment failure, small face,
blur, lighting, pose, detector confidence and other failures; overlapping reasons
and combinations are reported separately. Ranges contain numerical measurements
only, not images. Thresholds remain unchanged and provisional.

## Offline commands

Run from the repository root using the existing environment. No packages were
installed. Generated outputs stay under ignored storage; use a fresh output
directory for each benchmark/replay.

```powershell
.venv/Scripts/python.exe -B -m unittest discover -s tests -v
.venv/Scripts/python.exe -B backend/scripts/offline_benchmark.py --output backend/storage/offline_evaluation/phase71-20261009 --detection-mode both --evaluation-mode both
.venv/Scripts/python.exe -B backend/scripts/calibrate_offline.py --records backend/storage/offline_evaluation/phase71-20261009/full_frame/raw_scores.jsonl --output backend/storage/offline_evaluation/phase71-20261009/full_frame/calibration.json --mode both
.venv/Scripts/python.exe -B backend/scripts/calibrate_offline.py --records backend/storage/offline_evaluation/phase71-20261009/desktop_tiled/raw_scores.jsonl --output backend/storage/offline_evaluation/phase71-20261009/desktop_tiled/calibration.json --mode both
.venv/Scripts/python.exe -B backend/scripts/replay_offline.py --output backend/storage/offline_evaluation/phase71-replay-20261009 --max-videos 2 --max-samples 60 --stride 5
```

Calibration evaluates 325 unchanged grid points per evaluation view: T=0.2…0.8
and M=0…0.3, step 0.025. Raw scores are independent of the sweep. Nothing writes
a new runtime T/M. Exact-hash duplicate/registration checks do not establish
independent capture sessions or exclude re-encoded/adjacent-frame leakage.

Replay chooses the same first two distinct local video hashes as the earlier
run. Track lifetimes mean last observed time minus first observed time; tracks
still active at replay end are right-censored. Recovery percentages report both
all-track and missed-track denominators. These are descriptive measurements,
not true track-switch accuracy on unlabeled recordings.

## Camera-source readiness

`desktop.camera.recognition_source.build_recognition_camera_source()` returns an
integer webcam index or delegates to the existing Hikvision RTSP builder. The
Recognition view passes that result to the unchanged CameraWorker contract.
Default `RECOGNITION_CAMERA_SOURCE=webcam`; index remains
`RECOGNITION_WEBCAM_INDEX` (default 0). Explicit `hikvision` selection constructs
the existing configured source. No credentials or full URL are logged, no .env
is edited, and tests use mocked/fake configuration without opening a capture or
network connection. No live-camera command was executed in this task.

PHASE_8_CODE_PATH_READY = YES

Live authorization, camera/network availability, real classroom labels, latency,
occlusion/blur/face-size evidence and restored cloud persistence remain future
validation requirements. Selecting a source is not Phase 8 validation.

## Measured image evaluation and calibration

Artifacts: `backend/storage/offline_evaluation/phase71-20261009/comparison.json`,
with `raw_scores.jsonl`, `benchmark_summary.json` and `calibration.json` under
each of `full_frame/` and `desktop_tiled/`. No old experimental output was replaced.

The unchanged local gallery generation is `20260925T091732Z-e3df6674`.
Rediscovery found 343 images: 201 enrolled-known, 80 unenrolled labeled people,
36 explicit-unknown and 26 unlabeled screenshots. Excluding 26 unlabeled images
and 13 duplicate unknown images leaves **304 samples = 201 enrolled + 103
impostors (80 unenrolled + 23 explicit unknown)**. All 11 gallery identities have
evaluation samples: eleven enrolled identities (identifiers redacted). Four
additional labelled identities were absent from the gallery; those labels are
redacted and no spelling aliases or enrollment were invented. No exact registration hash overlap was
found. Capture/session provenance and independent holdout remain unavailable.

At unchanged T=0.4/M=0.05, end-to-end ratios (both modes include all 304 samples):

| Detector / view | FAR | FRR | Wrong Identity Rate | Known Identity Accuracy | Impostors reaching matcher |
|---|---:|---:|---:|---:|---:|
| Full-frame diagnostic | 3/103 | 151/201 | 0/201 | 50/201 | 39/103 |
| Full-frame production | 0/103 | 194/201 | 0/201 | 7/201 | 1/103 |
| ROI+tiled diagnostic | 0/103 | 199/201 | 0/201 | 2/201 | 6/103 |
| ROI+tiled production | 0/103 | 200/201 | 0/201 | 1/201 | 0/103 |

Recognition-conditional ratios (only samples reaching that view's matcher):

| Detector / view | Samples: enrolled + impostors | FAR | FRR | Wrong Identity Rate | Known Identity Accuracy |
|---|---:|---:|---:|---:|---:|
| Full-frame diagnostic | 83 + 39 | 3/39 | 33/83 | 0/83 | 50/83 |
| Full-frame production | 7 + 1 | 0/1 | 0/7 | 0/7 | 7/7 |
| ROI+tiled diagnostic | 18 + 6 | 0/6 | 16/18 | 0/18 | 2/18 |
| ROI+tiled production | 1 + 0 | undefined | 0/1 | 0/1 | 1/1 |

Full-frame diagnostic FAR subgroups: unenrolled 2/80 end-to-end and 2/29
conditional; explicit unknown 1/23 end-to-end and 1/10 conditional. In full-frame
production, subgroup end-to-end FAR is 0/80 and 0/23, with only one unenrolled
and zero explicit unknown reaching matcher (conditional 0/1 and undefined).
Tiled diagnostic subgroup end-to-end FAR is 0/80 and 0/23, with three samples of
each subgroup reaching matcher (conditional 0/3 and 0/3). Tiled production has
zero samples of either impostor subgroup at the matcher: conditional FAR is
undefined, despite end-to-end 0/103. These denominators do not validate safety.

| Detection coverage / quality | Full-frame | ROI+tiled |
|---|---:|---:|
| Included samples | 304 | 304 |
| No face | 166 | 278 |
| Multiple faces | 0 | 0 |
| Valid alignment | 122 | 24 |
| Quality passed | 8 | 1 |
| Quality rejected after valid measurements | 114 | 23 |
| Invalid landmarks | 16 | 2 |
| Matcher diagnostic eligible | 122 | 24 |
| Production matcher eligible | 8 | 1 |

Quality reasons overlap: full-frame small=84, blur=113, lighting=2, pose=11,
invalid-landmarks=16, alignment-failure=0, low-detector-confidence=0, other=0;
tiled small=23, blur=23, invalid-landmarks=2, all other listed quality reasons=0.
No-face counts are listed separately above. Common full-frame combination is
small+blur (73); tiled has small+blur for all 23 otherwise-valid rejected faces.
For those rejected faces, width min/median/max is 8/24/48 px full-frame versus
5/10/28 px tiled; sharpness min/median/max is 1.73/8.34/22.60 versus
1.49/2.32/8.57. Full numerical ranges and combinations are in JSON. No threshold
was lowered to improve coverage. The static dataset and classroom ROI have a
domain mismatch; this is not a comparative live-camera accuracy experiment.

The sweep contains 325 T/M pairs per view × four detector/view combinations =
1,300 result rows, each with counts and end-to-end/conditional metrics. It found
10 Pareto representatives for full-frame diagnostic, and one each for the other
three combinations. Full-frame diagnostic representatives (denominators remain
201 enrolled and 103 impostors):

| T | M | False accepts | False rejects | Wrong known | Correct known |
|---:|---:|---:|---:|---:|---:|
| .200 | .000 | 32 | 123 | 2 | 76 |
| .200 | .025 | 20 | 125 | 1 | 75 |
| .200 | .050 | 13 | 129 | 1 | 71 |
| .200 | .075 | 9 | 132 | 0 | 69 |
| .200 | .100 | 5 | 135 | 0 | 66 |
| .200 | .125 | 3 | 139 | 0 | 62 |
| .200 | .150 | 1 | 146 | 0 | 55 |
| .200 | .175 | 0 | 154 | 0 | 47 |
| .225 | .025 | 17 | 127 | 1 | 73 |
| .225 | .050 | 12 | 131 | 1 | 69 |

Other representative points: full-frame production .2/.05 gives FA=0/103,
FR=194/201, WI=0/201, correct=7/201; tiled diagnostic .2/.05 gives FA=0/103,
FR=195/201, WI=0/201, correct=6/201; tiled production .2/0 gives FA=0/103,
FR=200/201, WI=0/201, correct=1/201. Equal outcome triples retain the least
restrictive lexicographic T/M representative, not a scientifically optimal value.
No point was applied to runtime. Raw arrays permit other risk policies later.

The earlier report's 80 absent-gallery samples were incorrectly part of known
FRR/Wrong Identity denominators. Those historical results are superseded here.
The corrected diagnostic view now reveals false accepts hidden by quality
rejections, but independent capture/video provenance, larger production-eligible
impostor populations and labeled deployment evidence are still missing.

## Measured replay, tests and change inventory

Replay outputs: `backend/storage/offline_evaluation/phase71-replay-20261009/`.
Video 1 and Video 2 are private local recordings. Their private filenames and
content hashes are redacted; internal hashes matched the prior run.
Each replay samples 60 frames at stride 5, covering 14.73 source seconds, with
no labels. These are local files, not RTSP connections.

| Diagnostic | Video 1 | Video 2 |
|---|---:|---:|
| Detected face observations | 1422 | 1009 |
| Tracks created | 83 | 106 |
| Tracks expired (TTL/miss policy) | 54 | 80 |
| Short-miss recoveries | 18 | 19 |
| Ambiguous detection associations | 1 | 2 |
| Tracks retired by ambiguity | 2 | 1 |
| Tracks created per sampled source minute | 338.00 | 431.72 |
| Median observed lifetime, seconds | 1.498 | 0.000 |
| Median observations per track | 7 | 1 |
| One-observation tracks | 26/83 (31.33%) | 69/106 (65.09%) |
| Tracks recovering a short miss, all-track denominator | 11/83 (13.25%) | 13/106 (12.26%) |
| Tracks recovering, missed-track denominator | 11/59 (18.64%) | 13/92 (14.13%) |
| Active tracks censored at replay end | 27 | 25 |
| Maximum simultaneously active tracks | 32 | 26 |
| Quality-passed / selected observations | 3 / 3 | 0 / 0 |
| AdaFace calls | 3 | 0 |
| Gate A passes / Gate B transitions / events | 0 / 0 / 0 | 0 / 0 / 0 |
| Peak retained crop/vector payload observed after embedding, bytes | 39,680 | 0 |

The configured per-track bounds remain three observations/embeddings and a
five-second evidence window; track TTL remains two seconds. Runtime dictionaries
are pruned when tracks retire, and per-frame lifecycle lists are replaced on
each update. Median lifetime zero means at least half of tracks were observed
only once, not that they consumed zero processing time. Short-miss counts can
exceed recovered-track counts because a track can recover repeatedly.

The videos demonstrate descriptive recovery/fragmentation and bounded execution,
but not successful identity confirmation: no observation passed Gate A. The
sticky-state and Top-K duration fixes are demonstrated by deterministic regression
tests, not a claim of labeled-video accuracy. No tracker threshold was changed.
Static evaluation made 146 AdaFace calls (122 full-frame + 24 tiled); production
views reuse their eligible subset's scores without additional inference.

Validation: **86 passed, 0 failed, 0 skipped** (56 prior tests + 30 new tests).
Four prior tests were updated because their assertions encoded the explicitly
superseded contracts: miss/poor-frame clearing, duplicate confirmed-state loss,
quality-only selection without refresh, and absent-gallery FRR accounting.
They remain in the suite; no test was deleted to hide a failure. The initial red
regression run had six expected failures, all fixed before final evaluation.
AST syntax checks passed for 83 Python files. No live camera/GUI/cloud test was run.

New files (7):

- `backend/recognition/benchmark_observation.py`
- `backend/recognition/detection_modes.py`
- `backend/recognition/replay_diagnostics.py`
- `desktop/camera/recognition_source.py`
- `tests/test_phase71_state.py`
- `tests/test_phase71_evaluation.py`
- `docs/phase71_stabilization.md`

Existing working-tree files modified (16):

- `backend/recognition/`: `coordinator.py`, `evidence.py`, `temporal.py`,
  `tracker.py`, `events.py`, `offline_evaluation.py`.
- `backend/scripts/`: `offline_benchmark.py`, `calibrate_offline.py`, `replay_offline.py`.
- `desktop/app.py`, `desktop/detection/worker.py`, `desktop/recognition/worker.py`.
- `tests/test_confirmation.py`, `tests/test_evidence.py`, `tests/test_evaluation.py`.
- `docs/offline_recognition.md` (historical report marked superseded).

Final tracked diff against HEAD: 12 files, +1000/−706, including pre-existing
changes and excluding untracked source. `git diff --check` passes. No staging,
commit, push, unexpected new large binaries or dataset/model duplication.
Final status/stat and comparison to the starting source snapshot are saved under
the Phase 7.1 evaluation output directory. Protected model/preprocessing/matcher,
alignment, detector/ROI/config, T/M and gallery-pointer hashes remain unchanged.

## Repository safety record

Branch `tkinter-gui`; starting tracked diff: 12 modified files, +999/−699,
including prior work. Starting status also included Phase 0–7 source/tests/docs,
session/SQL/logging/audit files, and `phase0_7_review/` plus its zip. They were not
reset, staged or removed. Snapshot outside the repo:
`<temporary-baseline-directory>`.
It contains binary tracked diff, status, small source snapshots (including
untracked code being edited), and protected-file hashes. No dataset/model copy.

SUPABASE ACCESSED = NO

SUPABASE MUTATED = NO

CLOUD EMBEDDINGS UPDATED = NO

CLOUD ATTENDANCE WRITTEN = NO

PRODUCTION_OPERATING_POINT_SELECTED = NO

INSUFFICIENT FOR PRODUCTION CALIBRATION
