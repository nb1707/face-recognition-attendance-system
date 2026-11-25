# Phase 8B.3 controlled replay

Source: existing `backend/storage/phase8b3/analysis/20261110T100546Z/phase8b3_summary.md`.

Seven private sequences were replayed for each candidate through the actual coordinator, including temporal Gate B. Raw videos, paths, observations and identity assignments are excluded.

| Candidate | Wrong assigned events | Correct identity coverage | Gate B confirmations | Events | Median first confirmation (s) | Quality passed | Admission increase vs 32/20 |
|---|---:|---:|---:|---:|---:|---:|---:|
| 32/20 baseline | 0 | 1 | 12 | 5 | 16.481 | 1,331 | 0 |
| 32/16 conservative | 0 | 1 | 12 | 5 | 16.481 | 1,764 | 433 |
| 24/12 balanced | 0 | 3 | 23 | 11 | 71.422 | 2,677 | 1,346 |

| Candidate | Correct assigned | Wrong assigned | Unassigned | Conflict | Ambiguous | Assigned-event precision |
|---|---:|---:|---:|---:|---:|---:|
| 32/20 baseline | 4 | 0 | 1 | 0 | 0 | 1.000 |
| 32/16 conservative | 4 | 0 | 1 | 0 | 0 | 1.000 |
| 24/12 balanced | 8 | 0 | 3 | 0 | 0 | 1.000 |

The precision denominator excludes unassigned events. Zero wrong assigned events across seven sequences is not a general precision guarantee. Some manually verified regions had no detector observations, and the evaluator measures only existing replay outputs. No production configuration was automatically selected.
