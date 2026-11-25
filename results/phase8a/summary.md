# Phase 8A diagnostic replay aggregate

Source: existing `backend/storage/phase8a/20261026-baseline/report.md`.

The unlabelled diagnostic replay sampled 120 frames from two private videos. No private filenames or frames are reproduced.

| Funnel stage | Count |
|---|---:|
| Processed frames | 120 |
| Post-NMS detections | 2,431 |
| Quality pass | 3 |
| Top-K selected | 3 |
| AdaFace calls | 3 |
| Gate A pass | 0 |
| Gate B confirmation | 0 |
| Confirmed events | 0 |

Quality admission was 3/2,431 (0.123406%). Detection consumed 97.16% of measured pipeline time, with mean detection time 1,166.287 ms and mean total sampled-frame pipeline time 1,200.431 ms. These timings include cold-start effects and do not establish live FPS.

The run was unlabelled and cannot provide identity accuracy, tracking accuracy, detector recall or a threshold recommendation. No tuning or production configuration change was made.
