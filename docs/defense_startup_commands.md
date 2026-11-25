# Defense Startup Commands

Start in the repository root with the verified Python environment activated.
Run the configuration block below in **each terminal** used for the backend or
desktop; process environment variables do not propagate to a separately opened
terminal.
The following variables make the intended local development configuration
explicit and avoid relying on shell working-directory defaults:

```powershell
$repo = (Get-Location).Path
$env:RECOGNITION_GALLERY_PROVIDER = "local"
$env:RECOGNITION_LOCAL_GALLERY_PATH = Join-Path $repo "backend\storage\local_gallery\current.json"
$env:RECOGNITION_CONFIG_PATH = Join-Path $repo "backend\recognition\offline_development.json"
$env:RECOGNITION_CAMERA_SOURCE = "webcam"
$env:RECOGNITION_WEBCAM_INDEX = "0"
$env:ATTENDANCE_API_BASE_URL = "http://127.0.0.1:8000"
```

`RECOGNITION_GALLERY_PROVIDER` supports `local` only in the final offline
runtime. `RECOGNITION_LOCAL_GALLERY_PATH` points to the current-gallery pointer,
not a historical generation. On a new machine the pointer is established after
the first successful guided enrollment publishes a gallery generation. Change
the camera source to `hikvision` only when the optional Hikvision variables in
`backend/.env.example` have been configured locally.

## 1. Backend

Working directory: `backend` from the repository root.

Command:

python -m uvicorn main:app --host 127.0.0.1 --port 8000

Local URLs:

http://127.0.0.1:8000/app
http://127.0.0.1:8000/class-admin

Developer only:

http://127.0.0.1:8000/docs


## 2. Desktop

Working directory: the repository root.

Command:

python -m desktop


## 3. Cloudflare Quick Tunnel

Command:

cloudflared tunnel --url http://127.0.0.1:8000

After startup, use the temporary hostname printed by the process. Do not record
the live hostname in repository files. The placeholder form is:

https://<random>.trycloudflare.com

Use:

https://<random>.trycloudflare.com/app
https://<random>.trycloudflare.com/class-admin


## 4. Final Demo Order

1. Start backend
2. Start Cloudflare Tunnel
3. Verify /app on phone
4. Verify /class-admin
5. Start desktop
6. Examiner enrolls
7. Finalize enrollment
8. Add examiner to Class A
9. Refresh/select Class A in desktop
10. Start Attendance Session
11. Run classroom recognition
12. End Session
13. Review Suggested Present / Not Observed
14. Teacher final review
15. Finalize
16. Show audit/API if needed

For a preconfigured synthetic demonstration roster, run from the repository
root after the configured identities have been enrolled:

```powershell
python -m backend.scripts.setup_demo_attendance
```

The script uses `backend/config/demo_classes.json` by default. It does not
create biometric identities; every configured synthetic student must already
exist in the local enrollment registry and current gallery. Alternatively,
create classes and rosters through `/class-admin`.


## 5. Fallbacks

Tunnel failure:
Use local or LAN URL.

Security limitation:
Quick Tunnel creates a temporary public route without providing the prototype
with production authentication or authorization. Use it only under supervision,
share the generated hostname only with the examiner, and stop it immediately
after the demonstration.

Internet failure:
Use http://127.0.0.1:8000/app on laptop.

Hikvision failure:
Use Webcam.

Live enrollment difficulty:
Use a pre-enrolled demo identity.

Recognition limitation:
Explain Not Observed safety and hardware/camera limitations.


## 6. Frozen Runtime

These values are provisional research/development settings, not a production
calibration.

Active gallery:
Resolved dynamically through `backend/storage/local_gallery/current.json`.
On a fresh installation, the first successful guided enrollment publishes a
new gallery generation and establishes this pointer.

Gate A roster requirement:
The current fail-closed margin policy requires at least two eligible enrolled
candidate identities in the active session roster. With fewer than two, the
second-candidate margin is unavailable and Gate A does not confirm a match.

Recognition config:
min_size=24
min_sharpness=12
similarity=0.40
margin=0.05
top_k=3
evidence_window=5.0
min_observations=3
min_attempts=3
min_duration=0.5
