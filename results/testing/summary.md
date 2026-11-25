# Final automated test evidence

Reviewed: 25 November 2026 (Asia/Saigon).

Both executions were performed by the author in the existing Python 3.12.4 project environment. Codex reviewed the durable artifacts but did not independently execute either run because its restricted process context could not access the author's interpreter.

## Execution comparison

| Result | Original run | Repository-local temporary-directory rerun |
|---|---:|---:|
| Started | 15:17:43 | 15:36:36 |
| Finished | 15:18:12 | 15:37:00 |
| Test scope | `tests/` | `tests/` |
| Collected pytest items | 318 | 318 |
| Passed | 238 | 318 |
| Failed assertions | 0 | 0 |
| Setup errors | 80 | 0 |
| Skipped | 0 | 0 |
| Warnings | 2 | 2 |
| Pytest duration | 27.09 s | 22.38 s |
| Metadata wall duration | 28.4651760 s | 23.5901087 s |
| Exit code | 1 | 0 |

Both sessions record win32, Python 3.12.4, pytest 9.2.0, pluggy 1.7.0, and anyio 4.14.2.

## Original execution

The original run's exact invocation was not recorded. All 80 errors occurred during fixture setup, before the affected assertions, because pytest received `PermissionError: [WinError 5] Access is denied` for its Windows user temporary root. No application assertion failure was recorded.

Its JUnit suite records 357 counted results: 277 without failure/error, 80 errors, 0 failures, and 0 skipped, with a suite time of 25.160 seconds. The difference between 357 JUnit results and 318 collected items is 39 additional subtest results.

## Successful final automated regression rerun

The rerun metadata records this command:

```text
python -m pytest tests/ --basetemp=<isolated-temp> --junitxml=results/testing/pytest-final-rerun.xml
```

The placeholder is part of the privacy-safe metadata; the exact repository-local directory name and the precise TEMP/TMP assignment commands are not recorded in the supplied artifacts. The author reports process-scoped TEMP and TMP overrides and an isolated repository-local pytest base temporary directory.

The rerun completed successfully with exit code 0: all 318 collected items passed, with no failed assertions, setup errors, or skips. Its JUnit suite reports 357 counted results, 0 failures, 0 errors, and 0 skipped, with a suite time of 22.376 seconds. It contains 318 testcase elements; the suite total therefore includes 39 additional subtest results, all successful. This is the successful final automated regression run.

The disappearance of all 80 setup errors under the temporary-directory override, without changes to tests or application code for the rerun, establishes that those errors were environmental rather than application assertion failures.

Both runs recorded the same two warnings:

1. AnyIO's pytest plugin uses a `CallSpec2` alias scheduled for removal in pytest 10.
2. FastAPI/Starlette `TestClient` uses a deprecated httpx integration path.

## Evidence integrity and privacy

SHA-256 hashes:

| Artifact | SHA-256 |
|---|---|
| `pytest-final.txt` | `799860b1e54344010c8ca30084ecb23e839e1da9499aafacd4d33f1014d5c103` |
| `pytest-final.xml` | `fbcf04281cb68157905175223bfce76fe52bdd8f70155927521318b72697c5e7` |
| `execution-metadata.txt` | `c582c57ed87fce35315f7e154116ffe2600c7247252193fdaae2efcab7393805` |
| `pytest-final-rerun.txt` | `edb6b05d0107cddb0b00defb2be392401bdb97f9f7e732ebc90febfb659a5770` |
| `pytest-final-rerun.xml` | `b90bb5bbf7b168c33e598bc7eba92103c367da07d5c7ddb1b3fcb208c1295c48` |
| `execution-metadata-rerun.txt` | `a0064960af528b3e20d60e63f436ab0e0d4f718c9e6fe4e1567f0927090c06b7` |

The four raw text/XML reports contain personal absolute paths, and the rerun XML also contains a machine hostname. They remain local and are excluded from the public release allowlist. Both privacy-safe metadata files and this sanitized summary are eligible for release. Historical evidence was preserved and not overwritten.

## Reproducibility boundary

- Existing Python environment: author verified.
- Automated regression suite: successful author-executed rerun supported by durable evidence.
- Clean dependency installation: **not verified**.
- Clean model download: **not verified**.
- Full fresh-machine deployment: **not verified**.

The successful existing-environment rerun does not establish clean-machine reproducibility.
