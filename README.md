# AdaFace-Based Face Recognition for Student Attendance

This repository contains the final research prototype for the University of Greenwich undergraduate project **Investigation and Optimization of an AdaFace-Based Face Recognition Pipeline for Student Attendance in Challenging Classroom Environments**.

The project investigates guided facial enrollment and roster-constrained recognition under classroom conditions. Recognition is evidence for a human-reviewed attendance workflow; it does not automatically decide that a student is absent.

## Final product scope

- Guided browser enrollment at `/app`.
- Browser class and roster administration at `/class-admin`.
- Tkinter desktop attendance application.
- Local FastAPI services and SQLite attendance repository.
- InsightFace detection and AdaFace embeddings.
- Immutable, versioned local biometric gallery.
- Recognition restricted to the selected session roster.
- Similarity/margin Gate A and temporal Gate B confirmation.
- Attendance recommendations followed by operator review and finalisation.
- Optional temporary Cloudflare Quick Tunnel for a supervised demonstration.

This is a research prototype, not a production-ready biometric platform. The checked-in recognition thresholds are provisional development settings rather than a production calibration.

## Architecture and data authority

The browser UI and desktop client communicate with the local FastAPI application. Enrollment metadata, samples and immutable gallery generations are stored under `backend/storage/` on the demonstration machine. Classes, roster snapshots, recognition events, recommendations, final decisions and audit history are stored in local SQLite.

SQLite is the authoritative final attendance database. Earlier Supabase-backed routes remain as legacy/research components, but Supabase is not the final attendance authority and is not required for the approved local workflow.

Private storage, databases, face images, embeddings, evaluation datasets, recordings and model weights are deliberately excluded from Git.

## Repository map

| Path | Purpose |
|---|---|
| `backend/main.py` | FastAPI application and route integration |
| `backend/enrollment/` | Guided enrollment and immutable gallery publication |
| `backend/attendance/` | SQLite schema, repository, APIs, recommendations and review |
| `backend/recognition/` | Detection, quality, matching, tracking and Gate A/B pipeline |
| `backend/static/` | Browser enrollment and class-administration interfaces |
| `desktop/` | Tkinter camera, recognition and attendance application |
| `tests/` | Automated tests using synthetic and isolated fixtures |
| `docs/` | Architecture, model, experiment and demonstration documentation |
| `results/` | Curated privacy-safe aggregate evidence and test reports |

## Supported environment

Development was recorded on Windows with Python 3.12.4. `requirements-freeze.txt` records the original environment. A clean installation has not yet been independently verified.

From the repository root in PowerShell:

```powershell
py -3.12 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r backend\requirements.txt
```

For automated tests and research analysis tools:

```powershell
python -m pip install -r requirements-dev.txt
```

Do not use `requirements-freeze.txt` as the primary installation interface unless reproducing the exact recorded environment.

## Model provisioning

Model files are not distributed in this repository. The final path requires InsightFace `buffalo_l`, with its detection model in the InsightFace cache, and AdaFace IR-101 WebFace4M in the path expected by `backend/recognition/adaface_loader.py`.

See [docs/model_setup.md](docs/model_setup.md) for verified identifiers, expected paths, hashes of audited local files, download instructions and licensing limitations. Model downloads require network access and substantial disk space.

## Safe environment configuration

Copy the template without replacing an existing local file:

```powershell
Copy-Item backend\.env.example backend\.env
```

The approved local browser workflow does not require Supabase credentials. Webcam use requires no camera credentials. Hikvision/RTSP and legacy cloud values are optional and should be supplied only on a trusted local machine. Never commit `backend/.env`.

## Initialize an empty SQLite database

The attendance repository creates its schema when it first opens the configured database. From the repository root:

```powershell
python -c "from backend.attendance.repository import AttendanceRepository; AttendanceRepository()"
```

This creates `backend/storage/attendance/attendance.db`. It is private runtime data and is ignored by Git.

## Start FastAPI

Current static and storage paths are backend-relative:

```powershell
Set-Location backend
python -m uvicorn main:app --host 127.0.0.1 --port 8000
```

Open:

- `http://127.0.0.1:8000/app`
- `http://127.0.0.1:8000/class-admin`
- `http://127.0.0.1:8000/docs` for developer API inspection

## Start the Tkinter application

In a second terminal, from the repository root:

```powershell
python -m desktop
```

The desktop client uses `http://127.0.0.1:8000` unless `ATTENDANCE_API_BASE_URL` overrides it.

## Demonstration workflow

### Guided enrollment

1. Open `/app` and provide a consented demonstration identity.
2. Allow browser camera access.
3. Follow the five guided poses and retry rejected samples where requested.
4. Finalise enrollment. The service publishes a new immutable gallery generation and synchronizes the student record to SQLite.

### Class and roster administration

1. Open `/class-admin`.
2. Create a class or select a synthetic demonstration class.
3. Add enrolled participants to its roster.
4. Verify the roster before starting attendance.

Do not place real student details in files intended for Git. Both
`backend/config/demo_classes.json` and
`backend/config/demo_classes.example.json` contain author-confirmed synthetic
configuration. The setup script uses `demo_classes.json` by default, and every
configured identity must already exist in the local enrollment registry and
current gallery:

```powershell
python -m backend.scripts.setup_demo_attendance
```

An examiner may instead create a fresh class through `/class-admin` and enroll
only consented demonstration participants. A new machine does not require a
specific historical gallery generation: the first completed enrollment creates
and publishes a new immutable generation and updates the current pointer.

### Webcam attendance

1. Start FastAPI and the Tkinter application.
2. Select the webcam source and class.
3. Start the session, which snapshots the class roster and gallery generation.
4. Present consented participants to the camera.
5. End the session.
6. Review recommendations, edit final statuses where required, and finalise.

Recognition events are evidence. A person who is not observed is not automatically declared absent; the operator retains final authority.

## Automated testing

The intended command is:

```powershell
python -m pytest
```

Tests use temporary directories, synthetic arrays/data and mocked camera/network boundaries. Consult `results/testing/summary.md` for the latest actual execution status. A clean environment or test suite must not be described as successful unless the recorded run completed successfully.

## Known limitations

- Recognition quality falls for small, blurred, poorly lit, distant or moving faces.
- Thresholds are provisional and were evaluated on limited development evidence.
- The prototype has no liveness or spoof-resistance mechanism.
- It is not a claim of legal, security or biometric-governance compliance.
- Final local interfaces do not provide production authentication, authorization, encryption-at-rest or a formal retention system.
- Model redistribution rights are not established by this repository.
- Hardware, driver and package compatibility outside the documented Windows/Python environment is not verified.
- Legacy cloud/video-analysis routes remain in the monolithic backend and are outside final attendance authority.

## Privacy and security

Use only informed, consented demonstration participants. Do not commit or upload `.env` files, databases, enrollment samples, faces, embeddings, galleries, classroom recordings, raw experimental observations or model weights. Curated `results/` files contain aggregate evidence only.

## Optional temporary remote access

For a supervised demonstration, with `cloudflared` installed:

```powershell
cloudflared tunnel --url http://127.0.0.1:8000
```

Use only the temporary URL printed for that session, share it only with the intended examiner, and stop the process immediately afterward. The tunnel is optional and does not move attendance authority away from local SQLite.
