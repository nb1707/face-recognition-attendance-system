# Phase 8B.1 quality sweep aggregate

Source: existing Phase 8B.1 quality-sweep summaries under ignored `backend/storage/phase8b/`.

Seven private recorded sequences contributed 95,344 persisted observations. The analysis reused persisted metadata and matcher scores: it did not rerun detection, AdaFace inference or Gate B.

Gate A remained fixed at similarity 0.40 and margin 0.05. All quality conditions other than minimum face size and minimum sharpness remained fixed.

| Candidate minimum size/sharpness | Quality admitted | Increase over 32/20 |
|---|---:|---:|
| 32/20 baseline | 1,331 | 0 |
| 32/16 | 1,764 | 433 |
| 24/12 | 2,677 | 1,346 |

The sweep measured admission and what-if Gate A outcomes only. It did not establish event-level identity correctness or select a production operating point. Phase 8B.2 and 8B.3 supplied additional labelled and temporal evidence.
