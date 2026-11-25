# Phase 8B.2 labelled spatial analysis

Source: existing final labelled analysis under `backend/storage/phase8b2/20261107-202208-final-labelled-analysis/`.

Manual spatial annotations supplied ground truth; predicted identity was never used as ground truth. The analysis contained 31,924 labelled observations. No Gate B was simulated and no production operating point was selected.

| Candidate size/sharpness | Quality pass | Matcher available | Correct top-1 | Wrong top-1 | Gate A pass | Correct Gate A | Wrong Gate A | Newly admitted correct Gate A | Newly admitted wrong Gate A |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 32/20 baseline | 1,093 | 944 | 941 | 3 | 914 | 914 | 0 | 0 | 0 |
| 32/16 | 1,374 | 1,225 | 1,220 | 5 | 1,123 | 1,123 | 0 | 209 | 0 |
| 32/12 | 1,622 | 1,473 | 1,462 | 11 | 1,278 | 1,277 | 1 | 363 | 1 |
| 28/12 | 1,687 | 1,538 | 1,517 | 21 | 1,318 | 1,317 | 1 | 403 | 1 |
| 24/12 | 1,757 | 1,608 | 1,578 | 30 | 1,362 | 1,361 | 1 | 447 | 1 |
| 24/8 | 3,412 | 3,263 | 3,119 | 144 | 2,534 | 2,522 | 12 | 1,608 | 12 |

At the 32/20 baseline, near observations had 1,082 quality passes out of 2,010 assigned observations, while far observations had zero quality passes out of 10,893. This is development evidence about the recorded conditions, not a universal distance-performance claim.

Limitations: the result is observation-level, derived from a limited labelled development set, and cannot replace event-level Gate B evaluation.
