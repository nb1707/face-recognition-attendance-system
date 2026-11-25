# Model setup

Model weights are local dependencies and are not included in the repository. Do not commit them or configure Git LFS unless redistribution rights are separately established.

## AdaFace IR-101 WebFace4M

| Property | Audited value |
|---|---|
| Repository identifier | `minchul/cvlface_adaface_ir101_webface4m` |
| Pinned revision | `f2b38d9e24bfe301490d8dd081d8924b102333dd` |
| Embedding dimension | 512 |
| Runtime checkpoint | `backend/model_weights/adaface/ir101_webface4m/pretrained_model/model.pt` |
| Audited size | 261,111,273 bytes |
| Audited SHA-256 | `7a3341c3afc507fd6f50345638d2f3ef2f0e931d5b4f5aba60e15709853fcf5e` |
| Source | Hugging Face repository above; its project page links to `https://github.com/mk-minchul/CVLface` |
| Licence/redistribution | **NOT VERIFIED.** The downloaded README says to follow the training-dataset licence but does not establish a redistribution right for this submission. |

The local directory also contained `model.safetensors` during the audit:

| Property | Audited value |
|---|---|
| Path | `backend/model_weights/adaface/ir101_webface4m/model.safetensors` |
| Size | 260,980,552 bytes |
| SHA-256 | `21adb6220e8799a0e658f16946df9649c7269f432fe9810a7b9c4ad1241080a8` |

The final loader opens `pretrained_model/model.pt`; it does not open the top-level safetensors file.

After reviewing upstream terms, download the pinned repository from the repository root:

```powershell
Set-Location backend
python -m scripts.download_adaface
```

The download is hundreds of megabytes. Confirm the runtime checkpoint hash afterward; upstream packaging could differ from the audited artifact.

## InsightFace detector

| Property | Audited value |
|---|---|
| Pack name used by final detector | `buffalo_l` |
| Allowed module | `detection` |
| Detection input size | 960 × 960 |
| Runtime | CPU (`ctx_id=-1`) |
| Default Windows cache | `%USERPROFILE%\.insightface\models\buffalo_l` |
| Required detection file observed locally | `det_10g.onnx` |
| Audited size | 16,923,827 bytes |
| Audited SHA-256 | `5838f7fe053675b1c7a08b633df49e7af5495cee0493c7dcf6697200b85b5b91` |
| Download URL used by installed InsightFace 1.0.1 | `https://github.com/deepinsight/insightface/releases/download/v0.7/buffalo_l.zip` |
| Pack revision/checksum published upstream | **NOT VERIFIED** |
| Licence/redistribution | **NOT VERIFIED** for the pack and training data |

The installed InsightFace 1.0.1 source calls `ensure_available("models",
"buffalo_l")` from `FaceAnalysis`. Its behavior is conditional:

- If `%USERPROFILE%\.insightface\models\buffalo_l` does not exist, it attempts
  to download and extract the pack from the URL above.
- If the directory already exists, it is accepted without checking that
  `det_10g.onnx` is present. An incomplete directory therefore does not trigger
  repair; `FaceAnalysis` subsequently fails because no detection model loaded.

For a connected first run, remove or repair an incomplete cache only after
backing it up and obtaining author approval, then allow InsightFace to provision
the absent directory. For an offline examination, obtain the complete upstream
pack separately under its applicable terms, place it at the cache path, and
verify that `det_10g.onnx` matches the audited hash before starting FastAPI. Do
not copy the cache into Git.

## Historical/comparison ArcFace asset

`backend/model_weights/arcface/arcface_r100.onnx` is a historical/comparison asset, not a requirement for the approved AdaFace startup path.

| Property | Audited value |
|---|---|
| Size | 260,665,334 bytes |
| SHA-256 | `4ab1d6435d639628a6f3e5008dd4f929edf4c4124b1a7169e1048f9fef534cdf` |
| Original source, revision and licence | **NOT VERIFIED** |

## Verification boundaries

- Hashes describe files present on the audited development machine on 25 November 2026.
- They do not prove redistribution permission.
- The files exceed GitHub's normal per-file limit and remain ignored.
- A clean model download and clean-environment application startup were not performed during preparation.
