# Phase 7.1 aggregate summary

Source: existing `backend/storage/offline_evaluation/phase71-20261009/comparison.json` and `docs/phase71_stabilization.md`.

The static evaluation included 304 privacy-sensitive source images. Only aggregate counts are reproduced here.

| Detector path | Included | No face | Valid alignment | Quality pass | Quality rejected | Correct known, production simulation | Wrong known | End-to-end FAR | End-to-end FRR |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Full frame | 304 | 166 | 122 | 8 | 130 | 7 | 0 | 0.0000 | 0.9652 |
| Desktop tiled | 304 | 278 | 24 | 1 | 25 | 1 | 0 | 0.0000 | 0.9950 |

Configuration: similarity 0.40, margin 0.05, minimum size 32, minimum sharpness 20, top-K 3, five-second evidence window. This evaluation applied static Gate A only; it did not simulate attendance or temporal Gate B.

Limitations: the set was development data, detection yield was low, identities and source files are withheld, and the result was explicitly insufficient for production calibration.
