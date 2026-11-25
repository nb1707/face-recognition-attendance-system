# Phase 8A: old-video diagnostic replay

## Full-video extension

`replay_offline.py` preserves its original default of 60 sampled frames and now
has an explicit EOF mode:

```powershell
.venv/Scripts/python.exe -B backend/scripts/replay_offline.py --video <local-clip.mp4> --full-video --stride 5 --output backend/storage/phase8a/<new-run-id>
```

Add `--shadow-matcher` only for offline diagnostic matching of valid faces that
production rejected at Quality Gate. Shadow embeddings and matches are counted
separately. They cannot update the coordinator, Gate B, or the local event sink.

Full-video replay decodes one frame at a time and streams `frames-0.jsonl`,
`observations-0.jsonl`, and `tracks-0.jsonl`. Exact counters, min/max/means remain
streaming. Median and percentile calculation uses a deterministic reservoir of
4,096 values per metric by default; change the fixed cap with
`--diagnostic-sample-cap`. Track rows use the same bounded reservoir for medians
while every finalized row is streamed to JSONL. The tracker remains capped by its
existing configuration and each track retains only the existing Top-K evidence.
No frames, crops, or embedding vectors are serialized.

The output directory also contains `events-0.jsonl` (possibly empty) and
`replay_summary.json`. Numbered filenames repeat for additional discovered videos.
`--stride` applies in both limited and full-video modes. Without `--full-video`,
`--max-samples` retains its prior semantics and defaults to 60.

This phase measures the current pipeline without changing detection, quality,
tracking, embedding, matching or temporal policy. It does not calibrate an
operating point or validate unlabeled identities.

```powershell
.venv/Scripts/python.exe -B backend/scripts/phase8a_replay.py --output backend/storage/phase8a/<new-run-id> --max-videos 2 --max-samples 60 --stride 5
```

Repeat `--video <local-file>` for explicit videos. Discovery otherwise uses
`backend/storage/videos`, sorted and deduplicated by streaming SHA-256. Increase
`--max-videos` when specifying more than two. Existing output directories are
rejected. No source videos are copied. Network URLs and UNC inputs are rejected;
Python socket connections are blocked and the detector must already exist locally.

## Shared execution path

The harness calls the desktop's `detect_for_mode(..., desktop_tiled)` and
`RecognitionCoordinator`. The detector restores ROI/tile coordinates and performs
cross-tile NMS. The coordinator tracks before quality measurement, aligns only
quality-passed detections, offers them to the existing Top-K buffer, extracts
uncached selected embeddings, and computes the weighted aggregate. It matches
both the individual fresh embedding and aggregate against distinct identities.
Both Gate-A results must pass and agree to add temporal evidence. Only a Gate-B
confirmation transition can emit a local event; the existing sink deduplicates
per session/user. Track ID is not student ID.

Replay intentionally uses source frame index / FPS, never completion time, as
the acquisition-equivalent timestamp. Sequence is source index + 1. Desktop's
asynchronous latest-result scheduling is not simulated: this is serial, fixed
stride replay through the same core algorithms.

## Instrumentation and interpretation

`phase8a_probe.py` wraps existing calls within a single-thread context and restores
them on exit. It does not replace their outputs or policies. Per-tile detector
results provide raw counts and surviving tile IDs without another detector pass.
"Raw" means results before **cross-tile** NMS; internal detector proposals are not
exposed. The Top-K explanation compares scalar pre/post metadata; it does not
select frames. Actual model calls include duration, dimension, norm, finiteness
and failure type, never embedding values.

Each observation nests its quality measurements/decision, selection metadata,
actual model calls, matcher attempts, Gate-A results and temporal state. Every
matcher attempt names `observation` or `aggregate`. Matcher-level Gate-A counts
include both calls; production counts represent agreeing observation/aggregate
pairs. Gate-B confirmation rate uses unique `(video_id, track_id)` with at least
one production Gate-A pass, and is null if there are none. Events are
pipeline-confirmed identities, not verified identities.

Quality score ranks observations; it is not a probability. Similarity is cosine
similarity. Detection width/height use floating restored boxes; quality sizes use
clipped integer crops. Invalid quality measurement objects retain their raw
defaults in logs but are excluded from numerical quality distributions. They
remain in reject totals. Reason counts overlap and can exceed reject counts.

Track lifetime is last observed minus first observed source time; active tracks
are right-censored. Expirations exclude ambiguity retirements, which are reported
separately. These are fragmentation diagnostics, not tracking accuracy or switch
rate. Aggregate tracks/minute uses the sum of sampled source spans. Frame timing
distributions include zeros for stages not invoked; per-call AdaFace timing does
not. Total pipeline timing includes detector and coordinator, excluding decode
and JSON writes. Coordinator timing includes alignment and aggregation overhead.

`short_miss_state_preserved` counts missed track-frame states with retained
evidence or confirmation. It is distinct from geometric short-miss recoveries.
An empty evidence buffer surviving a miss is not counted as preserved evidence.

## Outputs and bounds

The run manifest records video metadata, hashes, model/gallery provenance,
unchanged configuration, source hashes and the comparison with local Phase 7.1
artifacts when present. Frame, observation, track and confirmed-event JSONL files
use schema version 1. Six summary JSON files provide aggregate and per-video
statistics. Empty event output is valid. Undefined rates/statistics are JSON null.

Defaults cap each video at 60 samples, discovery at two videos, total observation
records at 200,000 and total JSONL storage at 128 MiB. Exceeding a bound fails
explicitly; it never silently drops records. Summaries retain scalar records
within those bounds. Only the current decoded frame and runtime's bounded crops/
embeddings are held as image data. No frame dumps, full embeddings or copied
videos are produced. Generated storage stays ignored by Git.

Run tests with bytecode disabled:

```powershell
.venv/Scripts/python.exe -B -m unittest discover -s tests -q
```

The 2026-10-26 baseline artifacts live in
`backend/storage/phase8a/20261026-baseline/`. See its `report.md` for measured
findings. Phase 8B and live Hikvision work are separate and have not been started.
