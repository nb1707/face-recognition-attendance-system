# Final Report Evidence Audit

**Audit basis:** current working tree, inspected 21 November 2026<br>
**Purpose:** primary technical evidence map for the University of Greenwich final project report<br>
**Permitted write:** this document only<br>
**Audit mode:** read-only inspection; no application execution, recognition, enrollment, camera access, database mutation, cloud access, dependency installation, or secret inspection

## Evidence labels and interpretation rules

| Label | Meaning in this audit |
|---|---|
| **IMPLEMENTED** | Current code/configuration implements the capability. This does not itself prove successful execution. |
| **TESTED** | A relevant automated test exists and a test outcome is available. Where the outcome is not stored in the repository, it is explicitly labelled human-reported. |
| **LIVE-VALIDATED** | The capability was exercised in a manual live rehearsal. These claims are human-reported unless a durable artefact is named. |
| **RESEARCH RESULT** | A stored experiment or evaluation artefact reports the result. It is not automatically a final-product or production result. |
| **LEGACY** | Retained earlier or research functionality outside the final product boundary. |
| **LIMITATION** | A demonstrated or clearly evidenced constraint. |
| **NOT VERIFIED** | Evidence is incomplete, inaccessible, or not stored in the repository. |
| **NOT IMPLEMENTED** | The current working tree contains no supporting implementation. |

Code existence is never treated as proof of testing. Development thresholds are never described as production-calibrated. Supabase functionality is explicitly separated from the final local attendance architecture. The older `docs/final_current_architecture_audit.md` remains useful supporting evidence, but its former class-administration and enrollment-to-SQLite gaps are superseded by the current code.

---

# Executive project snapshot

## Current state

The project is a local-first classroom face-recognition attendance prototype. It enrolls a student through a browser-guided five-pose process, publishes an immutable local biometric gallery generation, synchronizes student identity metadata into local SQLite, lets an operator create classes and edit rosters, and runs classroom recognition from a desktop application. Recognition generates evidence; it does not directly decide final attendance. A teacher reviews recommendations and remains the authority who finalizes the attendance record.

| Boundary | Current source-of-truth snapshot |
|---|---|
| Project purpose | Explore and implement face-recognition evidence for classroom attendance while preserving teacher control over final attendance. |
| Final product | `/app` enrollment, `/class-admin` administration, desktop camera/recognition/attendance workflow, local gallery, and SQLite attendance. |
| Research boundary | Model adapters and benchmark routes; offline replay; Phase 7.1, 8A, 8B.1, 8B.2, and 8B.3 artefacts; quality/distance/configuration analysis. |
| Legacy boundary | Older Supabase registration, recognition, attendance, benchmark, and video-analysis/session paths. These are not final attendance authority. |
| Primary technologies | Python 3.12.4, FastAPI, Uvicorn, HTML/CSS/JavaScript, Tkinter, OpenCV, InsightFace, ONNX Runtime, PyTorch, AdaFace, NumPy, SQLite, pytest, Cloudflare Tunnel, Hikvision RTSP. |
| Runtime interfaces | Browser `/app`; browser `/class-admin`; desktop Tkinter application; local FastAPI attendance/enrollment APIs. |
| Persistence | `enrollments/users.json`, enrollment sample/session artefacts, immutable `local_gallery` generations plus `current.json`, and `attendance.db`. Research artefacts are stored separately. |
| Recognition architecture | ROI/tiled detection, cross-tile NMS, IoU tracking, quality gate, alignment, Top-K evidence, AdaFace embeddings, roster-filtered multi-prototype cosine matching, Gate A, Gate B, confirmed identity events. |
| Enrollment architecture | Five guided poses, sample quality/pose validation, alignment, AdaFace embeddings, representative prototypes plus mean fallback, immutable gallery publication, student synchronization to SQLite. |
| Attendance architecture | Class roster, immutable session roster snapshot, recognition-event persistence, recommendations, teacher review, audit, and finalization. |
| Demo/deployment | FastAPI and desktop run on the demonstration laptop. A Cloudflare Quick Tunnel can expose `/app` and `/class-admin` over HTTPS while all authoritative data remains local. |

## Final product interfaces

- **WEB — IMPLEMENTED:** `/app` five-pose enrollment and `/class-admin` students/classes/roster administration.
- **DESKTOP — IMPLEMENTED:** webcam/Hikvision input, tiled detection, tracking, quality gating, AdaFace recognition, roster-filtered gallery, Gate A, Gate B, `RecognitionEvent` persistence, recommendations, teacher review, finalization, and audit.
- **PERSISTENCE — IMPLEMENTED:** local enrollment metadata/samples, immutable local gallery generations and pointer, local SQLite attendance database.
- **REMOTE DEFENCE ACCESS — IMPLEMENTED runbook; LIVE-VALIDATED (human-reported):** local FastAPI exposed through a temporary HTTPS Quick Tunnel.
- **LEGACY/RESEARCH:** Supabase-backed earlier flows, benchmark routes, and video analysis/session routes remain in the monolithic backend but are outside the final attendance path.

---

# 1. Introduction

## What the chapter should explain

The chapter should introduce the classroom attendance problem, explain why face recognition is investigated, distinguish machine recognition evidence from the institutionally meaningful final attendance decision, define the product/research boundaries, state a defensible aim and measurable objectives, and summarize the iterative engineering and evaluation methodology.

## 1.1 Background

The evidence supports framing the problem as attendance capture under real classroom conditions: multiple faces, distance, motion, camera field of view, low face resolution, blur/sharpness variation, unknown people, and the need to relate recognition only to the selected class roster. The design deliberately avoids equating a match with attendance. A confirmed recognition event is evidence used by a recommendation service; a teacher reviews and finalizes the record.

Strongest sources are `backend/recognition/`, `backend/attendance/recognition_integration.py`, `backend/attendance/recommendation.py`, `backend/attendance/review.py`, `backend/attendance/database.py`, the Phase 8 artefacts under `backend/storage/`, and `docs/phase71_stabilization.md`.

Safe claims:

- The final prototype implements a local classroom attendance workflow assisted by face-recognition evidence.
- Recognition evidence and final attendance are different persisted concepts.
- Classroom distance, face size, sharpness, motion, and camera placement materially affect the evaluated pipeline.

Claims not to make:

- Face recognition eliminates the need for teacher judgement.
- All students can be recognized across a classroom.
- The system is production-ready or institutionally approved.

## 1.2 Aim and objectives

### Derived aim

> To investigate, implement, and evaluate a local-first face-recognition prototype that supports classroom attendance through roster-constrained recognition evidence, while leaving review and final attendance decisions under teacher control.

This aim is derived from the current implementation and technical documentation. No approved proposal document or authoritative original aim was located in the accessible repository; `PROPOSAL_TECHNICAL_AUDIT.md`, if used, is an audit rather than the approved proposal itself. The final report should reconcile this derived aim with the submitted proposal before publication.

### Objective evidence matrix

| ID | Defensible objective | Implementation evidence | Test/evaluation evidence | Status |
|---|---|---|---|---|
| O1 | Investigate face detection and embedding approaches suitable for the project. | Model adapters/factory for Buffalo_L/InsightFace, FaceNet, AdaFace, and ArcFace R100; common recognition pipeline. | Offline evaluation infrastructure and model-specific thresholds exist, but no complete final common-protocol comparative result artefact was located. | **PARTIALLY ACHIEVED** |
| O2 | Implement guided student face enrollment and a versioned local recognition gallery. | `backend/enrollment/service.py`, enrollment config/gallery modules, `backend/static/app.js`, `backend/main.py`. | Enrollment/API/mobile/static tests; manual enrollment-to-gallery rehearsal reported. | **ACHIEVED** |
| O3 | Implement classroom face detection, tracking, quality control, and recognition. | `backend/recognition/`, `desktop/recognition/worker.py`. | Quality, tracking, matching, confirmation, offline integration, replay, Phase 7.1/8 tests and artefacts. | **ACHIEVED** for prototype scope |
| O4 | Integrate webcam and Hikvision/IP-camera sources into the desktop workflow. | Desktop camera/source selection and RTSP-related implementation; `desktop/app.py`. | Desktop tests and manual classroom rehearsal reported; broad hardware coverage not demonstrated. | **ACHIEVED** for tested devices |
| O5 | Implement class, roster, session, recommendation, review, finalization, and audit workflows. | `backend/attendance/`, `backend/static/class_admin.*`, desktop attendance UI. | Attendance database/API/recommendation/review/desktop/integration tests; live rehearsal reported. | **ACHIEVED** |
| O6 | Restrict recognition to the selected class and preserve a session roster snapshot. | `backend/attendance/recognition_integration.py`, repository/service, `desktop/recognition/worker.py`. | Roster-recognition integration and attendance API/database tests. | **ACHIEVED** |
| O7 | Evaluate recognition under representative classroom quality, distance, motion, and temporal conditions. | Replay/evaluation scripts and stored Phase 7.1/8 artefacts. | Phase 8B.1 quality sweep, 8B.2 labelled spatial analysis, and 8B.3 seven-clip controlled replay. | **PARTIALLY ACHIEVED**: useful development evidence, limited scale and no production calibration |
| O8 | Demonstrate end-to-end remote enrollment and local classroom attendance at defence time. | Defence runbook and Quick Tunnel deployment design. | **MANUAL LIVE REHEARSAL** reported for public access, remote enrollment, SQLite sync, roster-to-desktop, recognition, review, and finalization; no durable execution log. | **ACHIEVED** as a demonstration, not production deployment |

## 1.3 Methodologies used

The repository supports an iterative/prototyping methodology rather than a single linear build. Evidence includes phased offline development, regression-driven fixes, replay diagnostics, labelled spatial analysis, controlled configuration comparisons, a recognition-core freeze, and subsequent attendance integration. The engineering method combines:

1. modular implementation and unit/integration tests;
2. offline replay to make camera observations repeatable;
3. labelled diagnostic analysis for quality, distance, and wrong-match behaviour;
4. controlled comparison of candidate quality configurations through the actual temporal coordinator;
5. manual end-to-end rehearsal for interfaces and device integration;
6. local-first risk reduction and a temporary remote-access layer for demonstration.

Major phases are detailed in Chapter 6.

## 1.4 Overview of report

The final prose should signpost: literature and related products; requirements; architecture and database/pipeline design; iterative implementation; legal/social/ethical/professional issues; evaluation; and conclusions/future work.

### Chapter 1 evidence summary

| Item | Evidence/figure/table | Safe claim | Gap |
|---|---|---|---|
| Problem and boundary | Current code, architecture audit, Phase 8 artefacts; overall architecture figure | A teacher-controlled local prototype was built for challenging classroom conditions. | Formal stakeholder/problem evidence and original proposal need locating. |
| Aim/objectives | Derived matrix above | Objectives can be measured against implementation and evaluation. | Confirm exact approved proposal wording. |
| Methodology | Phase documents, tests, scripts, artefacts | Development was iterative, test-supported, and replay/evaluation driven. | Original project plan and revised schedule not consolidated. |

---

# 2. Literature review

## What the chapter should explain

This chapter should establish the theory behind the implemented choices and the metrics used to judge them. It must use independently verified academic/authoritative references. No complete repository bibliography supporting the final literature review was located, so citations must be researched rather than invented.

## Topic map

| Topic | Why relevant | Implementation location | Design decision supported | Refer back from |
|---|---|---|---|---|
| Biometric face recognition | Defines identification, templates, and error risks. | Enrollment and recognition packages. | Embedding-based identity evidence. | Design, ethics, evaluation |
| Face detection | Recognition depends on locating usable faces. | `backend/recognition/face_detector.py`, tiled detector. | InsightFace detector; ROI/tiled mode. | Implementation, evaluation |
| Facial landmarks/alignment | Normalizes pose/geometry before embeddings. | Alignment in recognition/enrollment flow. | Five-point alignment and fixed model input. | Design, implementation |
| Face embeddings | Explains vector representation and comparison. | AdaFace/model adapters, enrollment gallery. | Store prototypes rather than raw classifier classes. | Design, implementation |
| L2 normalization/cosine similarity | Supports comparable similarity scoring. | Embedding/matcher modules. | Cosine scoring on normalized vectors. | Design, algorithms |
| FaceNet | Research baseline/model alternative. | FaceNet adapter/model factory. | Comparative model capability. | Literature, model comparison |
| InsightFace/ArcFace/Buffalo_L | Detection and recognition baseline context. | Detector and adapters. | Buffalo_L detection-only in final runtime. | Literature, implementation |
| AdaFace | Final embedding model and quality-aware research context. | AdaFace adapter/checkpoint path. | Final runtime recognizer selection. | Literature, design |
| Image quality in recognition | Small/blurred faces dominate classroom failure. | `quality.py`, Phase 8B.1/8B.2. | Minimum size/sharpness gate. | Evaluation, limitations |
| Multi-prototype identities | Pose diversity is not always represented by one mean. | Enrollment representative selection and identity matcher. | Per-identity prototypes plus mean fallback. | Design, implementation |
| Object/face tracking | Links frame observations and bounds repeated work. | Tracker module. | Conservative IoU tracker with miss/TTL policy. | Design, algorithms |
| Temporal evidence | Reduces one-frame decisions and explains Gate B. | Evidence and temporal/coordinator modules. | Top-K evidence and minimum observations/time. | Design, evaluation |
| Open-set recognition | Unknown people require rejection, not forced classification. | Top-1/top-2, similarity and margin gates. | Fail-closed Gate A. | Requirements, evaluation |
| Classroom attendance systems | Provides domain and workflow context. | Attendance service/review/audit. | Recognition evidence separated from attendance authority. | Requirements, ethics |
| FAR/FRR/precision/recall/F1 | Defines reported evaluation measures and denominators. | Evaluation scripts/artefacts. | Metric selection and threshold analysis. | Evaluation |
| Privacy and biometric data | Images and embeddings are sensitive. | Local persistence and legacy/cloud boundary. | Local-first processing and minimized final authority. | Ethics, design |
| IP camera/RTSP | Explains latency, resolution, field of view, and reliability. | Desktop camera integration. | Hikvision/webcam sources. | Design, limitations |
| Edge/local-first vs cloud | Frames architecture, privacy, availability, and deployment trade-offs. | Local gallery/SQLite; Quick Tunnel; legacy Supabase. | Local authority with temporary web exposure. | Design, ethics |

### Existing references

No sufficiently complete, verified reference list was found in the inspected repository evidence. Names of models/frameworks in code and documentation are not substitutes for citations. The final author should source the original FaceNet, ArcFace/InsightFace, AdaFace, detection/tracking/quality, biometric-evaluation, privacy, and classroom-attendance literature and verify every bibliographic field.

### Safe and unsafe claims

- **Safe:** these topics directly explain implemented choices.
- **With limitation:** AdaFace was chosen operationally; the repository does not contain a complete final artefact proving it outperformed every alternative under one fair protocol.
- **Do not claim:** literature consensus, state-of-the-art performance, or superiority without sourced evidence.

### Evidence gap

Chapter 2 is currently **WEAK**: it has a strong implementation-led topic plan but lacks a verified scholarly bibliography and synthesized critical review.

### Chapter 2 evidence capsule

| Required audit item | Current answer |
|---|---|
| Project-specific subjects | Detection, alignment, embeddings, model families, similarity/open-set rejection, quality, tracking, temporal evidence, classroom attendance, metrics, privacy, cameras, and local/cloud design. |
| Strongest repository evidence | Recognition/enrollment modules, model adapters, configuration, attendance workflow, and the implementation-linked topic map above. |
| Strongest experimental evidence | Phase 7.1 and Phase 8 show why quality, open-set gates, temporal evidence, and camera constraints require literature support. |
| Possible figures/tables | Literature concept map; table linking theory → design decision → implementation → evaluation. |
| Safe claim | These bodies of literature are directly relevant to choices present in the prototype. |
| Must not claim | That the repository already contains a complete critical review, or that any model is state of the art/superior without verified sources and comparative data. |
| Remaining gap | Find, read, critically compare, and correctly reference authoritative sources. |

---

# 3. Review of other products

## What the chapter should explain

The chapter should compare the project with existing attendance products or systems and use that comparison to justify requirements. No reliable repository artefact documenting named competitor products and their verified capabilities was located. External product research is still required.

## Comparison criteria to research

| Criterion | Why it matters to this project |
|---|---|
| Enrollment workflow | Number/guidance of captures, validation, accessibility, and recovery. |
| Real-time recognition | Number of people, latency, and behaviour under distance/motion. |
| Roster integration | Whether recognition is constrained to the actual class. |
| Teacher approval | Whether the human remains final attendance authority. |
| Audit trail | Whether recommendations and changes are explainable. |
| Unknown/not-observed handling | Whether lack of evidence is incorrectly treated as absence. |
| Cloud dependency | Availability, privacy, cost, and offline operation. |
| Camera requirements | Resolution, placement, device compatibility, and scale. |
| Privacy/security | Authentication, retention, encryption, consent, and access control. |
| Evaluation transparency | Published datasets, metrics, thresholds, and limitations. |

Do not state any competitor capability until verified from an authoritative product or research source. Suggested output is a criteria-based comparison table followed by a requirements trace. Chapter 3 is currently **WEAK**.

### Chapter 3 evidence capsule

| Required audit item | Current answer |
|---|---|
| Project-specific subjects | Enrollment, recognition, roster integration, teacher control, audit, unknown/not-observed handling, camera requirements, cloud dependence, privacy, and evaluation transparency. |
| Strongest repository evidence | Current product capabilities provide the comparison columns; no reliable named-competitor research artefact was located. |
| Strongest experimental evidence | Phase 8 and live-rehearsal boundaries define meaningful criteria, but do not evaluate other products. |
| Possible figures/tables | Criteria-based product matrix; workflow comparison; requirement consequences table. |
| Safe claim | External product research remains necessary and should use the criteria above. |
| Must not claim | Any named product capability, price, accuracy, compliance, or superiority until independently verified and cited. |
| Remaining gap | Select relevant products/research systems, use authoritative current sources, and record access dates where appropriate. |

---

# 4. Requirements analysis

## What the chapter should explain

Requirements should connect the classroom problem and human roles to testable system behaviour. Priorities below reflect the current final product; they should be reconciled with the approved proposal if its original MoSCoW set differs.

## Functional requirements

| ID | Requirement | Priority | Status | Implementation evidence | Automated evidence | Live evidence |
|---|---|---|---|---|---|---|
| FR01 | Enroll a student face identity. | Must | **IMPLEMENTED** | Enrollment service/API and `/app`. | Enrollment and API tests; latest run human-reported. | Manual live rehearsal |
| FR02 | Guide five poses: front, left, right, up, down. | Must | **IMPLEMENTED** | Enrollment config and web UI. | Enrollment/mobile/static tests. | Mobile guided-pose rehearsal |
| FR03 | Validate samples for pose, quality, and duplication. | Must | **IMPLEMENTED** | Enrollment validation/service. | Enrollment tests. | Manual enrollment rehearsal |
| FR04 | Publish an immutable gallery generation and update the active pointer safely. | Must | **IMPLEMENTED** | Enrollment gallery/service and `current.json` publication. | Enrollment/identity-removal tests. | Enrollment-to-gallery rehearsal |
| FR05 | Synchronize finalized enrollment identity metadata into SQLite. | Must | **IMPLEMENTED** | `EnrollmentService.student_sync`; `upsert_enrollment_student`; wiring in `main.py`. | Enrollment SQLite-sync test. | Remote/local sync rehearsal |
| FR06 | Create a class and list students/classes. | Must | **IMPLEMENTED** | Attendance API and `/class-admin`. | Attendance API and class-admin static tests. | Class-admin rehearsal |
| FR07 | Add/remove students from a class roster. | Must | **IMPLEMENTED** | Attendance API/repository and class-admin JS. | Roster idempotence/error/future-snapshot tests. | Add-to-Class-A rehearsal |
| FR08 | Create/end/finalize an attendance session. | Must | **IMPLEMENTED** | Attendance service/API/database. | Database/API/review tests. | Multiple sessions reported |
| FR09 | Snapshot the class roster when a session begins. | Must | **IMPLEMENTED** | Database/repository/service. | Database/API and integration tests. | Session snapshot rehearsal |
| FR10 | Select webcam or supported Hikvision/IP camera. | Must | **IMPLEMENTED** | Desktop app/camera modules. | Desktop tests. | Classroom camera rehearsal |
| FR11 | Detect multiple faces using classroom ROI/tiled detection. | Must | **IMPLEMENTED** | Detector/tiled pipeline and worker. | Offline/quality/replay tests. | Classroom rehearsal |
| FR12 | Track face observations across frames. | Must | **IMPLEMENTED** | Tracker/coordinator. | Tracking/state tests. | Indirect live rehearsal |
| FR13 | Measure and gate face quality. | Must | **IMPLEMENTED** | `quality.py`, config. | Quality/diagnostics/Phase 8 tests. | Indirect live rehearsal |
| FR14 | Extract AdaFace embeddings and match identities. | Must | **IMPLEMENTED** | AdaFace adapter and matcher. | Matching/offline integration tests. | New-identity recognition rehearsal |
| FR15 | Filter the gallery to the session roster. | Must | **IMPLEMENTED** | Recognition integration and desktop worker. | Roster-filtered integration tests. | Roster refresh/rehearsal |
| FR16 | Apply similarity and top-1/top-2 margin Gate A. | Must | **IMPLEMENTED** | Matcher/coordinator/config. | Matching/confirmation tests. | Indirect live rehearsal |
| FR17 | Apply temporal Gate B before confirming identity. | Must | **IMPLEMENTED** | Evidence/temporal/coordinator. | Confirmation/Phase 7.1 tests; Phase 8B.3. | Indirect live rehearsal |
| FR18 | Persist a recognition event as evidence. | Must | **IMPLEMENTED** | SQLite event sink/repository/schema. | Integration/database tests. | Recognition rehearsal |
| FR19 | Generate attendance recommendations without overwriting teacher authority. | Must | **IMPLEMENTED** | Recommendation service. | Recommendation tests. | Review rehearsal |
| FR20 | Let a teacher review/edit/finalize attendance. | Must | **IMPLEMENTED** | Review/service/API/desktop. | Review/finalization tests. | Manual review/finalization rehearsal |
| FR21 | Preserve attendance audit history. | Must | **IMPLEMENTED** | `attendance_audit` schema/repository. | Database/review tests. | Manual audit rehearsal |

## Non-functional requirements

| ID | Requirement | Priority | Status and evidence | Gap/limitation |
|---|---|---|---|---|
| NFR01 | Operate with final identity/attendance authority local to the demo laptop. | Must | **IMPLEMENTED**: local gallery and SQLite. | Deployment synchronization is not implemented. |
| NFR02 | Fail closed when margin/evidence is unavailable. | Must | **IMPLEMENTED** in config/matcher; tested by matching/confirmation tests. | Thresholds are provisional development values. |
| NFR03 | Preserve reproducible runtime dependencies/configuration. | Should | Root `environment-freeze.txt`, `requirements-freeze.txt`, recognition JSON, and defence runbook exist. | `backend/requirements.txt` is empty; model checkpoint acquisition/integrity needs clearer documentation. |
| NFR04 | Provide understandable guided enrollment and administration interfaces. | Should | Web UI and static tests; mobile improvements live-rehearsed. | No formal usability study or accessibility audit. |
| NFR05 | Minimize biometric/cloud exposure. | Must | Final processing/storage are local; temporary tunnel only exposes web endpoints. | No authentication, encryption policy, or retention policy. |
| NFR06 | Provide usable processing responsiveness. | Should | Latest-frame dropping/worker design and replay timings exist. | Final smoke mean processing about 1.22 s per sampled frame and p95 about 1.57 s; no defensible general real-time guarantee. |
| NFR07 | Preserve reliable state through short misses and poor frames. | Should | Phase 7.1 policies and regression tests. | Long occlusion/camera failure remains operationally limiting. |
| NFR08 | Handle camera/network errors gracefully. | Should | Error-handling paths and defence fallbacks are documented. | Broad fault-injection/live reliability evidence is limited. |

## Research/evaluation requirements

| ID | Requirement | Status |
|---|---|---|
| RR01 | Evaluate the offline recognition pipeline with explicit denominators and FAR/FRR-type measures. | **RESEARCH RESULT**, Phase 7.1/offline evaluation. |
| RR02 | Measure quality-threshold effects. | **RESEARCH RESULT**, Phase 8B.1/8B.2. |
| RR03 | Evaluate near, medium, far, and moving observations. | **RESEARCH RESULT**, Phase 8B.2. |
| RR04 | Compare candidate configurations through actual Gate B/event generation. | **RESEARCH RESULT**, Phase 8B.3. |
| RR05 | Compare recognition models fairly under a common protocol. | **PARTIALLY ACHIEVED / NOT VERIFIED**: adapters and infrastructure exist; final comparable result artefact is incomplete. |
| RR06 | Validate the full classroom workflow. | **LIVE-VALIDATED (human-reported)**; no durable execution report. |

## Out of scope/not implemented

- **NOT IMPLEMENTED:** liveness/anti-spoofing.
- **NOT IMPLEMENTED:** production authentication/authorization and institutional role management.
- **NOT IMPLEMENTED:** formal encryption and biometric-retention policy enforcement.
- **NOT IMPLEMENTED:** production cloud/edge synchronization or persistent public deployment.
- **NOT IMPLEMENTED:** broad multi-camera fusion.
- **NOT VERIFIED:** production-scale classroom accuracy, availability, security compliance, or accessibility compliance.

## Actors and diagram contents

| Diagram | Recommended contents |
|---|---|
| Use-case diagram | Student: enroll. Administrator/teacher: create class, manage roster. Teacher/operator: select camera, start/end session, review recommendations, edit and finalize. Examiner/demo participant: access remote enrollment/admin. Camera/device: supply frames. System: publish gallery, filter roster, recognize, record evidence, recommend, audit. |
| Enrollment activity | Open `/app` → enter identity → permission/camera → five guided poses → per-sample validation/retry → finalize → create embeddings/prototypes → publish gallery → synchronize SQLite student → completion/error rollback. |
| Attendance activity | Select class/camera → start session and snapshot roster → build roster-filtered gallery → detect/track/quality/recognize → Gate A/B → persist evidence → end session → recommend → teacher review/edit → finalize/audit. |
| Recognition sequence | Desktop → camera frame → detector/tile/NMS → tracker → quality/alignment/model → matcher → Gate A → evidence/temporal Gate B → event sink → SQLite → attendance UI. |
| Remote-demo sequence | Examiner browser → Quick Tunnel → local FastAPI → enrollment/class-admin services → local gallery/SQLite; desktop independently consumes the same local authority. |

Recommended figures are the use-case, enrollment activity, attendance activity, and recognition sequence diagrams. The strongest gap is formal stakeholder validation of the requirement set.

---

# 5. Design

## What the chapter should explain

The design chapter should show boundaries and data authority before implementation detail. It should make the separation between enrollment, recognition evidence, and final attendance unmistakable.

## A. Final architecture

```text
Browser                         Desktop application
  |                                  |
  +-- /app enrollment                +-- Webcam / Hikvision RTSP
  +-- /class-admin                   +-- ROI + tiled detection + NMS
  |                                  +-- tracking + quality + alignment
  v                                  +-- AdaFace + roster matcher + Gates A/B
FastAPI ------------------------------+
  | enrollment / class / attendance APIs
  +-- users.json + enrollment sessions
  +-- immutable local gallery + current.json
  +-- attendance.db (SQLite)
```

The final runtime is local-first. Temporary HTTPS access reaches the same laptop and does not move final attendance authority to the cloud.

## B. Persistence architecture

| Store | Purpose | Authority/classification |
|---|---|---|
| `enrollments/users.json` | Enrollment identity metadata. | Final local enrollment metadata |
| Enrollment session/sample directories | Source samples, progress, and derived caches used during enrollment. | Local biometric artefacts; retention requires governance |
| `local_gallery/<generation>/...` | Immutable manifests and embedding arrays for one gallery generation. | Final local biometric gallery |
| `local_gallery/current.json` | Pointer to the active immutable generation. | Final local activation pointer |
| `attendance.db` | Students, classes, roster, sessions, events, records, and audit. | Final attendance authority |
| Offline JSON/CSV/Markdown artefacts | Replay/evaluation inputs and results. | Research evidence, not live authority |
| Supabase tables/storage | Earlier registration/recognition/attendance and research paths. | **LEGACY/RESEARCH**, not final authority |

## C. Database ERD evidence

| Entity | Purpose | Key/important fields | Relationships and constraints |
|---|---|---|---|
| `students` | Local student directory synchronized from enrollment. | PK `student_id`; name, role, active state. | Active-state check; referenced by membership, roster, events, records, audit. |
| `classes` | Class definitions. | PK class ID; unique code; display fields. | One-to-many membership/sessions. |
| `class_members` | Current editable roster. | Composite PK `(class_id, student_id)`. | FKs to class/student; changes affect future sessions. |
| `attendance_sessions` | A class meeting and lifecycle. | PK session ID; class, start/end, status. | Status constrained to active/ended/finalized; owns snapshot/events/records. |
| `session_roster` | Immutable roster snapshot for a session. | Composite PK `(session_id, student_id)`. | FKs to session/student; separates historical session membership from later class edits. |
| `recognition_events` | Confirmed machine evidence. | PK event ID; session/student, confidence/evidence fields. | Unique session/student event constraint; referenced by attendance records. |
| `attendance_records` | Suggested and final attendance states. | PK record ID; session/student; suggestion/final fields; evidence FK. | Unique session/student; state checks; teacher decision remains distinct from event. |
| `attendance_audit` | History of attendance actions/changes. | PK audit ID; actor/action/timestamp/details. | FKs to session/record as applicable. |

Suggested ERD: `classes` → `class_members` ← `students`; `classes` → `attendance_sessions` → `session_roster` ← `students`; sessions/students → `recognition_events` and `attendance_records`; records/sessions → `attendance_audit`.

## D. Recognition pipeline

```text
Camera → ROI → three overlapping tiles → per-tile detection
       → cross-tile NMS → IoU tracker → quality measurement/gate
       → five-point alignment → Top-K recent evidence → AdaFace embedding
       → roster-filtered multi-prototype cosine scores → Top1/Top2
       → Gate A (similarity + margin) → Gate B (temporal confirmation)
       → ConfirmedIdentityEvent → SQLite recognition event
```

The configured thresholds are explicitly provisional development values, not production-approved.

## E. Enrollment design

```text
Browser camera → guided front/left/right/up/down samples
→ pose, quality and duplicate validation → alignment → AdaFace embeddings
→ representative prototypes (front and other poses) + mean fallback
→ immutable gallery generation → atomic active pointer update
→ users.json completion → SQLite student upsert
```

Current defaults collect 15 accepted pose samples (front 5, left 3, right 3, up 2, down 2) and select six representative prototypes (two front, one from each other pose), with a mean representation available. Finalization contains rollback behaviour if downstream publication/synchronization fails.

## F. Attendance design

```text
Editable class roster → start session → roster snapshot
→ roster-filtered gallery → recognition events
→ recommendation → teacher review/change → audit → finalization
```

Key semantic choices are: current roster and historical snapshot are different; a person with no recognition evidence is *Not Observed*, not automatically *Absent*; recognition evidence may support a recommendation but cannot silently finalize attendance.

## G. Deployment design

```text
Examiner browser → temporary Cloudflare Quick Tunnel → FastAPI on demo laptop
                                                     ↘ local gallery / SQLite
Desktop application on same laptop ------------------↗
```

The repository supports the local-first choice through local biometric/attendance stores, no cloud fallback in the final gallery path, and a defence runbook that treats the tunnel as an access layer. This reduces migration risk and keeps one local authority for the demonstration. It does not establish production security, uptime, synchronization, or scalability.

## Design decision register

| Decision | Alternatives evidenced | Chosen solution | Reason/evidence | Trade-off |
|---|---|---|---|---|
| Final embedding model | Buffalo_L recognizer, FaceNet, AdaFace, ArcFace R100 adapters | AdaFace | Final config/worker and pinned adapter/checkpoint. | Final cross-model superiority is not proven by a complete common-protocol artefact. |
| Detector role | Model packs/adapters | Buffalo_L/InsightFace detection-only in final runtime | `FaceAnalysis(... allowed_modules=["detection"])`. | Large detector dependency; SCRFD graph identity is indirect, not separately verified. |
| Classroom detection | Full-frame, ROI, tiled modes | ROI plus three overlapping tiles | Small/distant faces motivated tiled evaluation. | More inference cost and duplicate detections requiring NMS. |
| Duplicate suppression | Retain all tile detections | Cross-tile NMS | Prevent duplicate faces from overlap. | Pairwise comparison cost and IoU sensitivity. |
| Identity representation | Mean-only or single template | Multiple representative prototypes plus mean fallback | Preserve pose diversity from guided enrollment. | More storage and comparisons per identity. |
| Similarity | Model-specific classifier or vector distance | L2-normalized cosine similarity | Shared matcher across embedding adapters. | Threshold remains model/domain dependent. |
| Immediate acceptance | Top-1 only | Gate A similarity plus top-1/top-2 margin | Reject low-confidence and ambiguous matches. | Increases false rejection, especially with weak imagery. |
| Single-frame decision | Immediate match event | Gate B temporal confirmation | Requires repeated evidence and duration before event. | Delays confirmation and may miss brief appearances. |
| Evidence history | Latest frame only | Top-K recent usable observations | Stabilizes evidence and prevents one poor frame dominating. | Requires bounded state/provenance handling. |
| Tracking | Complex re-identification | Conservative IoU tracking | Simple, local, testable lifecycle. | Fragile under fast motion/occlusion/crossing. |
| Quality handling | Match every detection | Minimum size/sharpness gate | Avoid unreliable embeddings from unusable crops. | Far students are often excluded. |
| Gallery persistence | Mutable files or cloud-only store | Immutable generation + active pointer | Atomic activation, provenance, rollback. | Requires generation cleanup/retention governance. |
| Attendance persistence | Supabase/cloud or scattered files | SQLite | One local transactional authority for final workflow. | Single-machine deployment and no production sync. |
| Candidate scope | Full gallery | Session-roster filtered gallery | Prevent out-of-class identities competing. | Requires accurate roster/session setup. |
| Recognition semantics | Match equals attendance | Event is evidence; recommendation and final record are separate | Schema/services/tests preserve teacher authority. | Additional workflow/UI steps. |
| Missing evidence semantics | Automatically absent | Not Observed remains distinct | Avoid unfair negative inference from camera failure. | Teacher must resolve incomplete evidence. |
| Remote demonstration | Broad cloud migration | Local-first plus temporary tunnel | Reuses the same local authority and limits late architectural change. | No production availability/security guarantee. |
| Historical cloud code | Remove or make authoritative | Retain as legacy/research only | Preserves earlier work while isolating final path. | Monolithic backend remains harder to explain/maintain. |

### Chapter 5 evidence summary

Evidence is **STRONG** for architecture, pipeline, schema, and local authority. Recommended figures: overall architecture, ERD, enrollment pipeline, recognition pipeline, attendance lifecycle, and demo deployment. Remaining work is to render clean diagrams and document exact checkpoint provisioning without exposing private/runtime data.

---

# 6. Development / implementation

## What the chapter should explain

This chapter should tell an iterative engineering story: what failed, why the design changed, how tests/replay supported decisions, and which parts form the current final product.

## Chronological implementation map

| Phase | Purpose/problem | Main evidence | Solution and outcome | Later change/limitation |
|---|---|---|---|---|
| Phases 0–7 | Establish local gallery, matching, quality, tracking, temporal confirmation, and offline replay. | `backend/recognition/`, `tests/test_phase0.py`, matching/tracking/confirmation/offline tests, `docs/offline_recognition.md`. | Modular offline-capable recognition core and fail-closed matching. | Early evaluation showed very high upstream rejection and domain mismatch. |
| Phase 7.1 stabilization | Fix temporal state destroyed by brief misses/poor frames and evidence-duration stalls. | `docs/phase71_stabilization.md`, `tests/test_phase71_state.py`, Phase 7.1 evaluation. | Preserve appropriate pending evidence, refresh Top-K, maintain confirmed state through poor frames. | Still dependent on tracker continuity and usable crops. |
| Phase 8A diagnostics | Improve replay observability and full-pipeline diagnostics. | Phase 8A tests/scripts and stored diagnostics. | Better counters, rejection reasons, and repeatable replay. | Diagnostic results are not live production outcomes. |
| Phase 8B.1 quality sweep | Test whether size/sharpness gates block useful observations. | `backend/storage/phase8b/*quality-sweep/`. | What-if sweeps show admission gains and risks as thresholds relax. | Unlabelled sweeps cannot establish correctness or Gate B outcome. |
| Phase 8B.2 labelled spatial analysis | Measure correctness by quality setting and distance/motion. | `backend/storage/phase8b2/.../phase8b2_candidate_comparison.csv`, per-distance CSV, results JSON. | 24/12 identified as a balanced **development** setting; 24/8 admitted substantially more wrong matches. | Observation-level diagnostic; no Gate B/production selection. |
| Phase 8B.3 controlled replay | Compare candidates through actual coordinator/Gate B events. | `backend/storage/phase8b3/analysis/.../phase8b3_summary.md`. | Seven clips compared 32/20, 32/16, 24/12; 24/12 increased identity coverage/events with zero assigned wrong events in this small set. | Limited clips, unassigned events, long confirmation timing; no production calibration. |
| Recognition core freeze | Fix a defensible demo configuration and runbook. | `offline_development.json`, defence runbook, freeze files. | AdaFace/tiled/24/12 with explicit provisional warning and frozen demo instructions. | Dirty working tree/no formal release tag; configuration is development-only. |
| SQLite attendance foundation | Replace fragmented/final cloud authority with local relational workflow. | `backend/attendance/database.py`, repository/service, database tests. | Students, classes, memberships, sessions, snapshots, events, records, audit. | Local single-machine authority only. |
| Roster recognition and event integration | Prevent recognition against irrelevant identities and persist evidence. | `recognition_integration.py`, worker, integration tests. | Session context builds roster-filtered gallery and SQLite event sink. | Depends on roster accuracy and gallery membership. |
| Recommendation/review/finalization | Preserve teacher authority. | Recommendation/review/service/API and tests. | Suggested state separated from final state; edits audited; finalization controlled. | No authentication/role enforcement. |
| Desktop attendance UI | Integrate camera, recognition, session, review, and finalization. | `desktop/app.py`, worker, recognition view, desktop tests. | One operator workflow over local API/recognizer. | Detailed UI presentation may prioritize one selected/latest face although core processes multiple. |
| Class administration and enrollment sync | Remove manual data gap between enrollment and attendance. | `class_admin.*`, attendance API, `student_sync` wiring, new tests. | Finalized enrollment upserts student; web UI manages classes/rosters. | Authentication and bulk institutional import remain absent. |
| Demo/mobile/tunnel readiness | Make examiner enrollment/admin practical without cloud migration. | Static UI, mobile tests, defence runbook. | Responsive guidance and temporary public HTTPS routing to local FastAPI. | Manual usability evidence only; public exposure is demo-only. |
| Final dependency/config freeze | Record the runnable environment. | `environment-freeze.txt`, `requirements-freeze.txt`, runbook. | Python/package versions and commands captured. | Empty `backend/requirements.txt` and external model files weaken reproducibility. |

## Strongest implementation examples to explain

1. **Tiled detection and cross-tile NMS:** a concrete response to small classroom faces and overlapping tiles.
2. **Tracker lifecycle and temporal-state preservation:** demonstrates debugging of short misses and poor frames.
3. **Quality gate plus Phase 8 evidence:** connects code parameters directly to measured trade-offs.
4. **Top-K evidence and Gate A/Gate B:** shows open-set ambiguity rejection followed by temporal confirmation.
5. **Multi-prototype matcher:** uses pose-diverse enrollment rather than a single identity vector.
6. **Immutable gallery publication:** demonstrates atomic data/version management and rollback.
7. **Roster-filtered recognition:** connects domain requirements to candidate-space reduction.
8. **SQLite session roster snapshot:** preserves historical truth when the live class roster changes.
9. **Recognition event → recommendation → teacher decision:** strongest human-in-the-loop contribution.
10. **Audit/finalization constraints:** shows transactional integrity and accountability.

## Algorithm inventory

| Component | Input → output | Current parameters | Complexity/performance | File/function evidence |
|---|---|---|---|---|
| InsightFace detection | Image/tile → boxes, landmarks, scores | Buffalo_L detection module; CPU context `-1`; detector input 960×960 | Neural inference dominates per tile. | `backend/recognition/face_detector.py` |
| ROI/tiled detection | Frame → combined detections | ROI normalized `(0.08,0.27)–(0.91,0.84)`; 3 tiles; overlap 0.20 | About three detector calls per processed frame. | `detection_config.py`, tiled detector |
| NMS | Candidate boxes → de-duplicated boxes | IoU threshold 0.40 | Typical implementation O(n²) in candidate count. | detection module/config |
| IoU tracking | Detections + tracks → track assignments | min IoU 0.30; TTL 2.0 s; max misses 2; max tracks 64 | Approx. O(tracks × detections) per update. | tracker/config |
| Quality measurement | Face crop → size/sharpness/acceptance | Final development min size 24 px; min sharpness 12 | Pixel/crop processing; cheap relative to inference. | `quality.py`, `offline_development.json` |
| Five-point alignment | Image + landmarks → normalized face | Model input 112×112 for final AdaFace | Fixed-size geometric transform. | alignment/model pipeline |
| AdaFace | Aligned RGB face → 512-D embedding | Pinned IR-101/WebFace4M checkpoint; normalized model input | PyTorch inference dominates after detection. | AdaFace adapter |
| L2 normalization | Embedding → unit vector | epsilon/guard in implementation | O(d), d=512 final model. | embedding/matcher modules |
| Cosine similarity | Two unit vectors → score | Dot product after normalization | O(d). | identity matcher |
| Multi-prototype max | Query + identity prototypes → identity score | 6 representative pose prototypes plus mean fallback at enrollment defaults | O(Pd) per identity; more robust but more work than mean-only. | enrollment config/gallery, matcher |
| Top-1/top-2 margin | Identity scores → best, runner-up, margin | Similarity 0.40; margin 0.05 | Ranking/sort cost depends on implementation/identity count. | matcher/config |
| Top-K evidence | Track observations → bounded best/recent evidence | K=3; max age 5.0 s | Bounded memory/work per track. | evidence/config |
| Gate A | Top scores → accepted/rejected observation | similarity ≥0.40 and margin ≥0.05; fail closed if margin unavailable | O(1) after scores. | matcher/coordinator/config |
| Gate B | Accepted temporal observations → confirmed event | min observations 3; min attempts 3; min duration 0.5 s; window 5.0 s | Bounded per-track temporal state. | temporal/coordinator/config |
| Roster filtering | Full gallery + snapshot IDs → session gallery | Current session roster | O(identities + roster), then reduces downstream match candidates. | attendance recognition integration |

All numeric recognition thresholds above are **provisional development values only**, as stated in the configuration.

## Technology stack

| Technology | Purpose/location | Why appropriate | Classification/evidence |
|---|---|---|---|
| Python 3.12.4 | Backend, recognition, desktop, scripts/tests | Shared language across ML, API, UI, and evaluation. | Current; `environment-freeze.txt` |
| FastAPI 0.141.1 | Enrollment/attendance/static routes | Typed local HTTP interface and static serving. | Current; freeze and `main.py` |
| Uvicorn 0.52.3 | ASGI server | Simple local/demo hosting. | Current |
| HTML/CSS/JavaScript | Enrollment/class-admin browser UIs | Device-accessible camera/admin interface. | Current |
| Tkinter | Desktop operator UI | Bundled Python desktop interface. | Current |
| OpenCV 4.10.0.84 | Camera frames, crops, transforms, image metrics | Mature vision/video primitives. | Current |
| InsightFace 1.0.1 | Buffalo_L detector and model ecosystem | Provides landmarked face detections. | Current detection; some recognizer code research/legacy |
| ONNX Runtime 1.29.0 | ONNX model execution | CPU inference for supported adapters. | Current/research |
| PyTorch 2.2.2 / torchvision 0.17.2 | AdaFace and FaceNet execution | Required by final AdaFace checkpoint and research adapter. | Current/research |
| AdaFace | Final 512-D face embedding model | Operational final recognizer. | Current; superiority not fully verified |
| NumPy 1.26.4 | Embeddings, gallery arrays, scoring | Efficient numerical storage/computation. | Current |
| SQLite | Attendance relational persistence | Transactional, local, portable single-machine authority. | Current final authority |
| pytest 9.2.0 | Automated unit/integration/regression tests | Broad modular verification. | Current |
| Cloudflare Tunnel | Temporary HTTPS access to local FastAPI | Avoids broad cloud migration for defence. | Demo only; live rehearsal human-reported |
| Hikvision RTSP | Classroom camera source | Supports the available classroom/device setup. | Current device integration |
| Supabase 2.31.0 | Older cloud registration/recognition/attendance/research paths | Historical platform integration. | **LEGACY/RESEARCH** |

The root freeze records the environment and installed packages. `backend/requirements.txt` is empty, so it should not be presented as the reproducibility source. Final runtime also depends on local detector/model packs and a pinned AdaFace checkpoint; the report should document provenance and setup without embedding secrets or private model/data paths.

### Chapter 6 evidence summary

Evidence is **STRONG** for current implementation and iterative fixes. Recommended figures/tables: phase timeline, pipeline diagrams, selected code listings, and failure/iteration table. The main gap is a clean version/release marker and consolidated changelog rather than a dirty working tree snapshot.

---

# 7. Legal, social, ethical and professional issues

## What the chapter should explain

This chapter should identify risks, affected people, mitigations, and unresolved responsibilities. It must not make legal-compliance conclusions. Applicable institutional policy and jurisdiction-specific law require authoritative research and supervisor/institution review.

| Issue | Why relevant | Current mitigation/evidence | Current limitation | Potential future control |
|---|---|---|---|---|
| Biometric privacy | Face images/embeddings can identify people. | Final processing/gallery are local; cloud is not final authority. | Local storage is still sensitive; access policy not implemented. | Access control, encryption, DPIA/privacy review, least privilege. |
| Consent | Enrollment and classroom capture affect students. | Guided enrollment is an explicit interaction. | No institutional consent/withdrawal workflow. | Recorded informed consent, alternative attendance route, withdrawal/deletion process. |
| Image retention | Enrollment samples may persist. | Immutable generations aid provenance. | No formal retention/deletion schedule. | Defined retention periods, automatic deletion, documented exceptions. |
| Embedding retention | Embeddings remain biometric templates. | Local versioned gallery and identity-removal tests. | No formal retention/encryption policy. | Encrypted storage, key management, deletion verification. |
| Attendance decisions | False results can affect students. | Event ≠ final attendance; teacher review/finalization. | Teacher may over-trust recommendations; no authenticated actor. | Explainable evidence, training, appeals/correction workflow, role controls. |
| Human oversight | Automation should not silently decide attendance. | Recommendation/final distinction and audit. | No formal governance/SOP. | Written teacher procedures and accountability. |
| Bias/fairness | Performance may vary by identity and conditions. | Labelled identity/distance diagnostics expose variation. | Dataset is small/private and not a demographic fairness study. | Representative consented dataset and subgroup evaluation. |
| False acceptance | Wrong identity can create misleading evidence. | Similarity+margin Gate A, temporal Gate B, roster filter, teacher review. | Development thresholds; limited test scale. | Calibrated thresholds, monitoring, liveness, stronger cameras. |
| False rejection | Students may be missed, especially far/blurred. | Not Observed is not automatically Absent. | Far/moving/low-sharpness limitations remain. | Camera redesign, multi-view capture, alternative check-in. |
| Security | Public/local endpoints and biometric data need protection. | Final authority remains local; temporary tunnel limits migration. | **NOT IMPLEMENTED:** authentication/authorization; no security audit. | TLS plus authenticated roles, network controls, threat model, penetration testing. |
| Liveness/spoofing | A photo/screen might be presented. | No evidenced mitigation. | **NOT IMPLEMENTED.** | Liveness/anti-spoofing and supervised enrollment. |
| Encryption | Protects data at rest/in transit. | Tunnel supplies HTTPS for the remote demo path. | No documented at-rest encryption/key policy; local traffic/security not comprehensively assessed. | Encrypt stores/backups, managed keys, TLS policy. |
| Public tunnel exposure | Temporary URL exposes app routes. | Intended only for supervised defence; local data authority retained. | No production auth/rate limiting/uptime evidence. | Authenticated reverse proxy, firewall, persistent managed deployment. |
| Camera surveillance | Continuous classroom capture can affect autonomy. | Session/operator-controlled workflow. | No signage/notice/camera-governance artefact. | Notice, purpose limitation, camera activation policy, restricted viewing. |
| Data minimization | Only necessary data should be retained. | Roster filtering and local processing reduce scope. | Samples/research artefacts may accumulate. | Inventory, minimization review, deletion schedule. |
| Auditability | Decisions must be reviewable. | Events, records, teacher changes, and audit table are distinct. | Audit actor identity lacks production authentication. | Signed actor identities, tamper controls, export/reporting. |
| Accessibility/usability | Camera/pose workflow may exclude some users. | Guided poses, responsive/mobile UI, retry feedback. | No formal accessibility evaluation or alternative workflow in UI. | WCAG review, assisted/alternative enrollment, usability study. |
| Professional responsibility | Claims and deployment must match evidence. | Config labels thresholds provisional; this audit separates evidence types. | Legacy code and demo access can be misunderstood. | Clear documentation, code separation, model/data cards, change control. |
| BCS/professional considerations | Public interest, competence, privacy, and truthful claims are relevant. | Human oversight and evidence-qualified reporting align with these concerns. | A formal mapping to the current BCS code has not been completed. | Cite and map authoritative current professional guidance. |

Explicit absences: production authentication, liveness/anti-spoofing, an encryption policy, a retention policy, and an institutional consent workflow are **NOT IMPLEMENTED**. Chapter 7 evidence is **MEDIUM**: system mitigations are clear, but governance/legal research and formal policies remain.

---

# 8. Evaluation

## What the chapter should explain

Evaluation must separate component tests, offline research, controlled replay, and manual live validation. Every rate needs its denominator and every result needs its configuration and limitations.

## Evidence inventory

| Area | Strongest source | Sample/configuration | Main usable result | Interpretation | Limitation |
|---|---|---|---|---|---|
| A. Model comparison | Model adapters/factory, thresholds, benchmark infrastructure | Buffalo_L/ArcFace, FaceNet, AdaFace paths | Comparison capability and model-specific preprocessing/thresholds exist. | The system investigated multiple embedding approaches. | No complete accessible final fair comparison artefact; do not claim a winner from accuracy. |
| B. Offline recognition | `docs/offline_recognition.md`; Phase 7.1 comparison JSON/docs | 304 included samples in Phase 7.1; 201 known, 103 unknown | At T=.40/M=.05 full-frame diagnostic: FAR 3/103, FRR 151/201, correct 50/201, wrong 0/201; production policy: FAR 0/103, FRR 194/201, correct 7/201. | Upstream/domain mismatch caused severe rejection; policy reduced false acceptance at large recall cost. | Historical gallery/domain, not final runtime validation. |
| C. Quality sweep | `backend/storage/phase8b/*quality-sweep/quality_sweep_summary.md` | Candidate size/sharpness pairs | On one clip quality passes rose from 616 at 32/20 to 872 at 24/12 and 1747 at 24/8. | Relaxing gates admits more evidence. | Unlabelled what-if; no accuracy or Gate B conclusion. |
| D. Spatial/distance | Phase 8B.2 candidate/per-distance CSV and JSON | 31,924 labelled observations | 24/12: 1,757 quality pass, 1,578 correct top-1, 30 wrong top-1; Gate A 1,361 correct/1 wrong. | 24/12 balanced admission and diagnostic error better than aggressive 24/8. | Observation-level diagnostic; no Gate B/production selection. |
| E. Controlled replay | Phase 8B.3 summary | 7 clips; 32/20, 32/16, 24/12 | 24/12 produced coverage 3, 23 confirmations, 11 events, 8 assigned-correct/0 assigned-wrong and 3 unassigned events. | Actual coordinator showed added coverage in this small controlled set. | Small set, unassigned events, median first confirmation 71.422 s; no general precision claim. |
| F. Automated tests | 30 current test modules; user-supplied latest run | Focused 130 +31 subtests; full 318 +39 subtests | Broad unit/integration/regression coverage; 2 dependency deprecation warnings in each reported run. | Strong functional regression evidence. | **HUMAN-REPORTED VERIFIED RUN**; no durable current test log located. |
| G. End-to-end classroom | Defence runbook plus reported rehearsals | New enrollment through recognition/finalization; multiple sessions | Full local and remote-demo flows reportedly completed. | Practical integration is demonstrated. | **MANUAL LIVE REHEARSAL** only; no machine-readable log/screenshots supplied. |
| H. Usability/demo | Web/static tests and reported mobile rehearsal | Guided mobile enrollment and class admin | Pose guidance improved/retested; remote pages usable in rehearsal. | Supports demo readiness. | No formal participants, task measures, questionnaire, or accessibility study. |

## Offline recognition detail

Phase 7.1 discovered 343 samples and included 304: 201 enrolled-known and 103 impostor/unknown. At the provisional 0.40 similarity and 0.05 margin operating point:

| Mode/policy | FAR | FRR | Wrong identity | Correct known |
|---|---:|---:|---:|---:|
| Full-frame diagnostic | 3/103 | 151/201 | 0/201 | 50/201 |
| Full-frame production-style policy | 0/103 | 194/201 | 0/201 | 7/201 |
| Tiled diagnostic | 0/103 | 199/201 | 0/201 | 2/201 |
| Tiled production-style policy | 0/103 | 200/201 | 0/201 | 1/201 |

The same artefact reports full-frame `no_face=166`, `quality_pass=8`, elapsed 143.66 s; tiled `no_face=278`, `quality_pass=1`, elapsed 352.72 s. This is strong evidence of an unsuitable historical sample/gallery domain and severe upstream rejection, not evidence that the final tiled design is generally worse. An older offline summary reports only 8 matched samples from 304 and therefore has too small a conditional subset for broad accuracy claims.

## Phase 8B quality configuration results

### Phase 8B.2 labelled candidate comparison

Source: `backend/storage/phase8b2/20261107-202208-final-labelled-analysis/phase8b2_candidate_comparison.csv` and associated JSON/summary. Dataset: 31,924 labelled observations. Gate A figures are diagnostic observation-level results; Gate B was not run.

| Min size/sharpness | Quality passes | Correct top-1 | Wrong top-1 | Correct Gate A | Wrong Gate A | Newly admitted correct Gate A | Newly admitted wrong Gate A |
|---|---:|---:|---:|---:|---:|---:|---:|
| 32/20 | 1,093 | 941 | 3 | 914 | 0 | 0 | 0 |
| 32/16 | 1,374 | 1,220 | 5 | 1,123 | 0 | 209 | 0 |
| 32/12 | 1,622 | 1,462 | 11 | 1,277 | 1 | 363 | 1 |
| 28/12 | 1,687 | 1,517 | 21 | 1,317 | 1 | 403 | 1 |
| 24/12 | 1,757 | 1,578 | 30 | 1,361 | 1 | 447 | 1 |
| 24/8 | 3,412 | 3,119 | 144 | 2,522 | 12 | 1,608 | 12 |

Why 24/12 was selected as a **development configuration**: relative to 32/20 it admitted 447 additional correct Gate A observations with one additional wrong Gate A observation in this labelled set, while the more aggressive 24/8 setting increased wrong Gate A outcomes to 12. This is a development trade-off, not production calibration.

### Distance and motion

Source: `phase8b2_per_distance.csv`. The rows mix structural/quality admission and diagnostic matching evidence; they do not represent final attendance events.

| Group | Assigned observations | Structurally valid | Baseline quality passes | Diagnostic Gate A correct/wrong | Interpretation |
|---|---:|---:|---:|---:|---|
| Near | 2,010 | 1,899 | 1,082 | 1,285 / 0 | Strongest usable imagery and diagnostic evidence. |
| Medium | 15,354 | 13,188 | 2 | 2,989 / 9 | Baseline quality almost entirely rejects this group; shadow candidate analysis shows possible evidence with some errors. |
| Far | 10,893 | 10,835 | 0 | 20 / 1 | Runtime-quality admission fails; faces were often around the roughly 20-pixel scale in classroom evidence. |
| Moving | 3,667 | 3,075 | 9 | 58 / 54 | Motion is especially ambiguous and demonstrates why quality/temporal gates are necessary. |

Identity-level analysis shows substantial variation and at least one anonymized identity contributing many diagnostic wrong Gate A outcomes. Private names/IDs are intentionally omitted. The report may discuss identity-specific weakness only in anonymized aggregate form.

## Phase 8B.3 actual temporal/event comparison

Source: `backend/storage/phase8b3/analysis/20261110T100546Z/phase8b3_summary.md`.

| Config | Quality passed | Correct identity coverage | Gate B confirmations | Events | Assigned correct/wrong | Unassigned | Median first confirmation |
|---|---:|---:|---:|---:|---:|---:|---:|
| 32/20 | 1,331 | 1 | 12 | 5 | 4 / 0 | 1 | 16.481 s |
| 32/16 | 1,764 | 1 | 12 | 5 | 4 / 0 | 1 | 16.481 s |
| 24/12 | 2,677 | 3 | 23 | 11 | 8 / 0 | 3 | 71.422 s |

It is safe to say that 24/12 increased coverage and events in these seven clips without an assigned wrong event. It is not safe to report universal precision of 1.0, because three events were unassigned, the dataset is small, and missing identities/late confirmations remain.

## Final smoke evidence

A stored final-smoke replay used one unlabelled poor/low-light clip: 60 sampled frames, 558 detections, 4 quality passes, 2 Gate A observations, 0 Gate B confirmations, and 0 events. Mean processing was approximately 1,220 ms per sampled frame and p95 approximately 1,570 ms. This is useful failure/throughput evidence, not an accuracy result and not support for a general real-time claim.

## Model comparison evidence

| Item | Safely reportable evidence | Missing/unsafe conclusion |
|---|---|---|
| Common pipeline | Model factory/adapters feed the surrounding detection/alignment/matching/evaluation structure. | A perfectly controlled final experiment is not established. |
| Buffalo_L | Final runtime uses its InsightFace detection component; a recognition adapter/baseline path also exists. | Do not imply the final runtime uses Buffalo_L embeddings. |
| FaceNet | InceptionResnetV1/VGGFace2 adapter with 160×160 preprocessing exists. | No final stored accuracy/FAR/FRR/time table found. |
| AdaFace | Pinned IR-101/WebFace4M adapter, 112×112 input, 512-D output; selected final runtime. | Selection is operational, not proof of universal superiority. |
| ArcFace R100 | Adapter/threshold path exists. | Final comparative result incomplete. |
| Thresholds | Legacy/model defaults include Buffalo 0.45, FaceNet 0.65, AdaFace 0.40, ArcFace R100 0.45. | These are not directly comparable calibrated operating points. |
| Metrics | Infrastructure/docs reference accuracy, precision, recall, F1, FAR, FRR, wrong identity, and processing time. | Do not populate absent model-by-model numbers. |

## Automated test evidence

No durable current console/test report was located. Therefore the latest supplied results are recorded exactly as **HUMAN-REPORTED VERIFIED RUN**:

- Focused final-product suite: **130 passed, 31 subtests passed, 2 dependency deprecation warnings**.
- Full suite: **318 passed, 39 subtests passed, 2 dependency deprecation warnings**.

Historical file-backed results include 56 passes in `docs/offline_recognition.md` and 86 passes in `docs/phase71_stabilization.md`; these are not the current full suite.

Current static inventory contains 306 `test_...` functions across 30 modules. The discrepancy between test-function count and pytest pass/subtest totals is plausible because parametrization/subtests can expand cases, but the current run remains human-reported without a stored log.

| Test area | Main modules/capability |
|---|---|
| Attendance database/API | Schema constraints, CRUD, lifecycle, errors, class/roster endpoints, snapshots. |
| Recognition integration | Roster-filtered gallery, session context, SQLite event sink. |
| Recommendation/review | Suggestions, Not Observed semantics, edits, audit, finalization. |
| Desktop/demo | Attendance UI/controller, setup/demo data, recognition layout. |
| Enrollment | Service/API/cancel/mobile/static UI, immutable publication, SQLite synchronization. |
| Recognition core | Quality, tracker, matcher, evidence, temporal confirmation, offline integration. |
| Evaluation | Phase 7.1, 8A diagnostics/replay, 8B sweeps/spatial evaluator. |

## Live end-to-end validation

No machine-readable rehearsal log was found. The following are **LIVE-VALIDATED — MANUAL LIVE REHEARSAL (human-reported)**:

| Flow | Reported result | Repository support | Evidence limitation |
|---|---|---|---|
| New enrollment → gallery | Completed | Enrollment service/gallery/tests | No live log/screenshots |
| New enrollment → SQLite student | Completed | Current sync hook/wiring/test | No live log |
| `/class-admin` available student | Completed | UI/API/tests | No live log |
| Add student to Class A | Completed | Roster endpoints/UI/tests | Private identifiers omitted |
| Desktop roster refresh | Completed | API client/worker/integration | No live log |
| Session roster snapshot | Completed | Schema/service/tests | No live log |
| Recognition of newly enrolled identity | Completed | Full pipeline implementation | No labelled machine artefact from rehearsal |
| Multiple classroom sessions | Completed | Session workflow | Sample size/details not durably recorded |
| Teacher review/finalization/audit | Completed | Services/schema/tests | No live log |
| Public HTTPS and remote `/app`/`/class-admin` | Completed | Defence Quick Tunnel runbook | Temporary demo exposure only |
| Remote enrollment/SQLite sync/roster→desktop | Completed | Shared local architecture | No live trace |
| Mobile pose guidance | Improved and retested | Static/mobile tests | No formal usability dataset |

## Recognition limitation table

| Category | Limitation | Evidence/impact |
|---|---|---|
| Algorithm | Thresholds are provisional development settings. | Explicit warning in `offline_development.json`; no production calibration. |
| Algorithm | Conservative IoU tracking can lose continuity under motion/occlusion. | Moving group ambiguity and tracker design. |
| Algorithm | Latest-frame dropping favours throughput over processing every frame. | Worker architecture; brief evidence may be skipped. |
| Hardware | Far faces are often around the ~20 px scale and fail quality admission. | Phase 7.1/8B.2 size evidence. |
| Hardware | Sharpness frequently falls below the gate under classroom optics/motion. | Quality sweeps and spatial analysis. |
| Hardware | Field of view/resolution limits distant recognition. | Far group has zero baseline quality passes. |
| Evaluation | Labelled and controlled video sets are limited and private. | 31,924 observations but only seven controlled clips and limited identities. |
| Evaluation | Model comparison artefacts are incomplete. | Adapters exist; final common-protocol result table absent. |
| Evaluation | Live multi-person rehearsal lacks durable metrics. | Human-reported only. |
| Performance | Stored smoke replay does not support a general real-time claim. | ~1.22 s mean processed sample, ~1.57 s p95. |
| UX | Core processes multiple faces, while detailed desktop presentation may focus on a selected/latest result. | Multi-face decisions in worker versus focused result panel/layout. |
| Security | No production authentication/authorization. | No supporting implementation found. |
| Security | No liveness/anti-spoofing. | No supporting implementation found. |
| Deployment | Quick Tunnel is a temporary demo access layer; repository contains no production SLA/uptime evidence. | Defence runbook/local design. |
| Deployment | No cloud/edge synchronization architecture. | Single local gallery and SQLite authority. |
| Deployment | Legacy/cloud paths remain in a monolithic backend. | Current routes/modules include legacy/research features. |
| Scope | Broad multi-camera support/fusion is absent. | Webcam/Hikvision source support is not multi-camera fusion. |

## Failure and iteration history

| Problem | Symptom/investigation | Root cause | Design response/result | Remaining limitation |
|---|---|---|---|---|
| Short detector miss erased evidence | Gate B failed after brief no-face frames; Phase 7.1 regression tests. | State reset policy was too aggressive. | Preserve pending evidence across bounded misses. | Long misses/track loss still reset continuity. |
| Top-K/Gate B stall | Early evidence did not progress duration/attempt state. | Evidence refresh/temporal accounting interaction. | Refresh recent Top-K and test early-evidence progression. | Confirmation still needs sustained usable views. |
| Poor frame cleared confirmed identity | UI/state became unstable. | Quality failure was treated as identity loss. | Preserve confirmed state through poor frames. | Identity eventually expires by lifecycle policy. |
| Gallery/model provenance | Risk of mismatched embeddings/configuration. | Mutable or unclear model/gallery lineage. | Immutable generations, manifest/pointer, provenance checks. | Formal model/data-card and cleanup policy absent. |
| Old Supabase dependency | Local recognition could depend on cloud state. | Earlier architecture used cloud gallery/services. | Local gallery with no cloud fallback in final path. | Legacy routes remain and must be clearly bounded. |
| Fragmented attendance persistence | No single local transactional authority. | Multiple files/cloud-era paths. | SQLite schema/repository/service. | No production sync/high availability. |
| Not Observed treated as Absent risk | Lack of camera evidence could become a negative decision. | Semantic conflation. | Separate recommendation/final state; Not Observed preserved. | Teachers need guidance for unresolved cases. |
| Recognition included non-class identities | Irrelevant candidates could compete. | Full-gallery matching lacked session scope. | Snapshot roster and build roster-filtered gallery. | Incorrect roster still causes exclusion/inclusion errors. |
| Small/blurred classroom faces | Most distant faces failed quality or matching. | Camera resolution/FOV, motion, distance. | Tiling and measured 24/12 development gate. | Hardware remains the dominant far-distance constraint. |
| Mobile pose guidance usability | Pose capture/retry was difficult on phone. | Guidance/layout did not sufficiently fit mobile interaction. | Responsive/guided UI improvements and static tests; manual retest reported. | No formal usability/accessibility study. |
| Class administration gap | Newly enrolled students/classes required awkward setup. | No final local admin UI and sync bridge. | `/class-admin`, local APIs, automatic enrollment-to-SQLite upsert. | No auth/bulk institutional integration. |
| Ephemeral cloud deployment risk | A cloud migration could split authority and add late failure modes. | Final state was local and not synchronized. | Keep local authority; expose pages through temporary tunnel. | Demo access is not production deployment. |

### Chapter 8 evidence summary

Evidence is **STRONG** for a BSc prototype evaluation because it contains stored quantitative experiments, broad tests, and a reported live rehearsal. Its boundaries must remain explicit: incomplete final model comparison, development thresholds, limited clips/identities, no production calibration, and no durable current test/live logs.

Recommended figures: quality-setting bar/line chart, distance admission chart, Phase 8B.3 event/coverage chart, evaluation flow diagram, and anonymized classroom-camera setup. Recommended tables are the results above.

---

# 9. Conclusion evidence map

Do not copy this section as final conclusion prose. Use it to draft a balanced answer to the aim.

| Conclusion question | Evidence-backed answer |
|---|---|
| What was successfully achieved? | Five-pose local enrollment, immutable gallery, automatic SQLite student sync, class/roster administration, camera-driven recognition, roster filtering, recognition-event persistence, recommendation, teacher review, audit, finalization, and a temporary remote-demo path. |
| What was partially achieved? | Classroom recognition and evaluation: useful near/medium evidence and controlled replay exist, but distance/motion, scale, and calibration remain limited. Model comparison infrastructure exists without a complete final fair results artefact. |
| What was not achieved? | Production readiness, authentication/roles, liveness, formal privacy/retention/encryption controls, broad multi-camera fusion, persistent cloud deployment/sync, and production-calibrated accuracy. |
| Strongest technical contribution | The end-to-end evidence chain from pose-diverse immutable enrollment through roster-constrained temporal recognition to teacher-controlled audited attendance. |
| Strongest experimental finding | Quality relaxation has a measurable trade-off: 24/12 admitted substantially more correct evidence than 32/20 in labelled analysis, while 24/8 sharply increased wrong matches. |
| Strongest practical finding | Recognition evidence can reach a usable attendance workflow without making the machine the final attendance authority; live rehearsal reportedly completed this chain. |
| Most important limitation | Classroom optics—especially tiny, blurred, moving, distant faces—constrain recognition before matching, and the evaluation scale is insufficient for production claims. |
| Logical future work | Better/higher-resolution or optical-zoom cameras; optimized placement; larger consented classroom dataset; classroom-specific calibration; liveness; authenticated roles; encryption/retention/consent controls; persistent deployment and secure synchronization; richer audit/report UI; multi-camera fusion only after single-camera evidence is mature. |

Chapter 9 evidence is **STRONG** for a balanced prototype conclusion, provided limitations are not softened.

---

# References plan

The final reference list must be generated from sources actually read and cited. Required categories include original papers/documentation for FaceNet, ArcFace/InsightFace, AdaFace, detector/alignment methods, tracking/temporal evidence, biometric evaluation metrics, face-image quality, classroom attendance systems, privacy/biometric governance, IP-camera/edge processing, SQLite/FastAPI where technically relevant, and the current BCS professional code. Repository file paths belong in evidence notes/appendices, not as substitutes for academic references.

---

# Figure and screenshot inventory

| Figure | Chapter | What it should show | Existing source | New capture needed? |
|---|---:|---|---|---|
| Overall final architecture | 1/5 | Browser, FastAPI, desktop, gallery, SQLite, legacy boundary. | Code/audit | Yes: draw clean diagram |
| Enrollment pipeline | 5 | Five poses through immutable publication and SQLite sync. | Enrollment code/config | Yes: draw |
| Recognition pipeline | 5 | ROI/tile/NMS/tracker/quality/AdaFace/matcher/Gates/events. | Recognition code/config | Yes: draw |
| Attendance lifecycle | 5 | Roster snapshot → event → recommendation → review → audit/final. | Attendance services/schema | Yes: draw |
| Database ERD | 5 | Eight current entities and cardinalities. | `database.py` | Yes: generate diagram |
| Five-pose enrollment UI | 6 | Desktop/mobile layout, pose guidance and progress. | `/app` static files | Yes: privacy-safe screenshot |
| Mobile pose guidance | 6/8 | Responsive view and overlay/retry feedback. | Static UI/tests | Yes |
| Class administration | 6 | Students/classes/current roster/add/remove. | `class_admin.*` | Yes, anonymized |
| Desktop RecognitionView | 6 | Camera, detected faces, session controls/results. | Desktop UI/layout tests | Yes, anonymized |
| Roster-filtered recognition | 5/6 | Full gallery vs snapshot candidate set. | Integration code/tests | Draw rather than private screenshot |
| Teacher review | 6/7 | Recommendations and editable final decision. | Desktop/API/review | Yes, synthetic data |
| Finalized attendance/audit | 6/7 | Final state and change trail. | Schema/UI | Yes, synthetic data |
| Quality comparison | 8 | Candidate quality passes and wrong Gate A trade-off. | Phase 8B.2 CSV | Yes: chart from aggregate data |
| Distance/motion comparison | 8 | Admission/correct-wrong evidence by group. | Phase 8B.2 per-distance CSV | Yes: chart |
| Controlled replay comparison | 8 | Coverage, confirmations, events for three configs. | Phase 8B.3 summary | Yes: chart |
| Hikvision classroom setup | 1/8 | Camera placement/FOV and distance constraints. | Not safely available in audited text | Yes, consented/anonymized |
| Cloudflare demo architecture | 5 | Temporary tunnel to local FastAPI/local authority. | Defence runbook | Yes: draw |

No screenshot should expose faces, student IDs, attendance records, secrets, local private gallery IDs, or tunnel URLs.

# Recommended report tables

| Table | Chapter | Source/evidence | Data ready? |
|---|---:|---|---|
| Objectives and status | 1/9 | This audit + proposal reconciliation | Mostly; proposal confirmation needed |
| Product comparison | 3 | External authoritative research | No |
| Functional/non-functional requirements | 4 | Current code/tests and this audit | Yes |
| Technology stack | 6 | Freeze files/code | Yes |
| Design decision register | 5 | Code/docs/evaluation | Yes |
| Recognition configuration | 5/6 | `offline_development.json`, detection/enrollment config | Yes; label development-only |
| Model comparison | 8 | Adapters/benchmark results | Partial; numeric final comparison missing |
| Quality candidate comparison | 8 | Phase 8B.2 | Yes |
| Distance/motion evaluation | 8 | Phase 8B.2 | Yes |
| Controlled replay events | 8 | Phase 8B.3 | Yes |
| Automated test coverage/results | 8 | Test tree + current stored run log | Coverage yes; current outcome only human-reported |
| End-to-end validation | 8 | Rehearsal record | Human-reported; durable evidence needed |
| Ethical risks/mitigations | 7 | Current controls/gaps + authoritative sources | Technical side ready; policy research needed |
| Limitations/future work | 8/9 | Evaluation and implementation | Yes |

# Appendix plan

1. Current automated-test module map and a future durable full-run summary.
2. Detailed Phase 7.1 and Phase 8 aggregate result tables, excluding private identity data.
3. Full architecture, ERD, sequence, and activity diagrams.
4. API route summary for final enrollment/class/attendance routes, with legacy routes visibly separated.
5. Short selected listings for immutable publication, roster snapshot/filter, Gate A/B, event sink, and finalization/audit—not the entire codebase.
6. SQLite schema and important constraints.
7. Exact provisional recognition/enrollment configuration with a development-only warning.
8. Defence/demo runbook with machine-specific paths, private generation IDs, and transient URLs removed.
9. Privacy/ethical risk assessment, consent/retention proposal, and data inventory.
10. Original versus revised project schedule and phase timeline, once recovered.

---

# Claim safety table

| Category | Claim | Required wording/evidence boundary |
|---|---|---|
| **SAFE TO CLAIM** | The system implements guided five-pose enrollment. | Supported by config/UI/service/tests. |
| **SAFE TO CLAIM** | Final attendance authority is local SQLite. | Supported by current desktop/API/repository/schema; Supabase is legacy/research. |
| **SAFE TO CLAIM** | Each session snapshots its roster. | Supported by schema/service/tests. |
| **SAFE TO CLAIM** | Recognition events are evidence, not final attendance. | Supported by separate tables/services and teacher review. |
| **SAFE TO CLAIM** | A teacher reviews and finalizes attendance, with audit history. | Supported by services/schema/tests. |
| **SAFE TO CLAIM** | The active local gallery is a pointer to an immutable generation. | Supported by enrollment/gallery implementation/tests. |
| **SAFE TO CLAIM** | Recognition is restricted to the selected session roster. | Supported by integration/worker/tests. |
| **SAFE TO CLAIM** | The latest full suite was reported as 318 passes plus 39 subtests and two dependency warnings. | Say **human-reported verified run; no stored current log located**. |
| **SAFE TO CLAIM** | Live rehearsal demonstrated newly enrolled identities reaching the classroom workflow. | Say **manual live rehearsal (human-reported)**. |
| **CLAIM WITH LIMITATION** | The prototype supports classroom recognition. | Scope to tested local hardware/scenarios; far/moving limitations. |
| **CLAIM WITH LIMITATION** | It processes multiple faces. | Core supports multiple detections/tracks; UI detail and live multi-person evidence are limited. |
| **CLAIM WITH LIMITATION** | 24/12 improved useful admission. | Development result on the labelled Phase 8B evidence, not production calibration. |
| **CLAIM WITH LIMITATION** | No wrong assigned events occurred at 24/12 in controlled replay. | Only seven clips; three unassigned events; do not generalize precision. |
| **CLAIM WITH LIMITATION** | The system provides public web access. | Temporary supervised Quick Tunnel demo, not production hosting. |
| **CLAIM WITH LIMITATION** | The system is real-time. | Prefer “camera-driven/online pipeline”; stored timing does not support a general real-time guarantee. |
| **CLAIM WITH LIMITATION** | AdaFace was the best model. | Replace with “AdaFace was selected for the final runtime”; final fair comparative proof is incomplete. |
| **DO NOT CLAIM** | Production-ready or production-calibrated. | Explicitly contradicted by config/evidence scope. |
| **DO NOT CLAIM** | All students are recognized at classroom distance. | Far baseline quality admission was zero in Phase 8B.2. |
| **DO NOT CLAIM** | The system is accurate at a universal percentage. | Results depend on dataset, policy, distance, thresholds, and denominators. |
| **DO NOT CLAIM** | Biometric/legal/security compliance. | Policies, auth, liveness, and formal assessment are absent. |
| **DO NOT CLAIM** | Liveness or spoof resistance. | Not implemented. |
| **DO NOT CLAIM** | Production cloud scale, uptime, or synchronization. | Not implemented/verified. |
| **DO NOT CLAIM** | Supabase is the final attendance database. | Final authority is SQLite. |

---

# Report-writing source index

| Topic | Best source files/directories |
|---|---|
| Final architecture | `docs/final_current_architecture_audit.md` (supporting, with superseded gaps); `backend/main.py`; this audit |
| Enrollment | `backend/enrollment/service.py`; enrollment config/gallery modules; `backend/static/app.js`; enrollment tests |
| Enrollment-to-SQLite sync | `backend/main.py`; `backend/enrollment/service.py`; `backend/attendance/import_enrollment_users.py`; `tests/test_enrollment.py` |
| Class admin | `backend/static/class_admin.html`; `.js`; `.css`; `backend/attendance/api.py`; class-admin/API tests |
| Detection | `backend/recognition/face_detector.py`; detection config/tiled detector; replay diagnostics |
| Recognition | coordinator, tracker, `quality.py`, evidence, identity matcher, temporal, model adapters, `offline_development.json` |
| Attendance schema | `backend/attendance/database.py` |
| Attendance workflow | attendance repository, service, API, recognition integration, recommendation, review modules |
| Desktop | `desktop/app.py`; `desktop/recognition/worker.py`; recognition view and desktop tests |
| Phase 7.1 | `docs/phase71_stabilization.md`; `backend/storage/offline_evaluation/phase71-20261009/comparison.json` |
| Phase 8B.1 | `backend/storage/phase8b/*quality-sweep/` |
| Phase 8B.2 | `backend/storage/phase8b2/20261107-202208-final-labelled-analysis/` |
| Phase 8B.3 | `backend/storage/phase8b3/analysis/20261110T100546Z/phase8b3_summary.md` |
| Automated tests | `tests/` (30 modules/306 static test functions); current run result is human-reported |
| Environment | `environment-freeze.txt`; `requirements-freeze.txt`; recognition/enrollment config |
| Deployment/demo | `docs/defense_startup_commands.md`; `backend/main.py`; local static routes |
| Legacy/research cloud | Supabase clients/routes, benchmark/video-analysis modules; keep visibly outside final architecture |

---

# Final master matrix

| Report section | Required content | Current evidence | Strength | Recommended figures | Recommended tables | Remaining work |
|---|---|---|---|---|---|---|
| 1 Introduction | Problem, aim, objectives, method, scope | Architecture/code, phases, evaluation, derived objectives | **MEDIUM** | Overall architecture, phase timeline | Objectives/status | Recover approved proposal and original schedule; validate aim wording. |
| 2 Literature review | Critical theory and research basis | Strong topic map tied to implementation; no verified bibliography | **WEAK** | Conceptual embedding/open-set diagram if useful | Literature synthesis matrix | Conduct and cite academic/authoritative research. |
| 3 Other products | Verified competitor/system comparison | Criteria only; no reliable product evidence | **WEAK** | Optional workflow comparison | Product comparison | Research named products and verify capabilities. |
| 4 Requirements | Actors, FR/NFR/research/out-of-scope, traceability | Current services, APIs, UI, tests, rehearsal | **STRONG** | Use case/activity/sequence | Requirements traceability | Reconcile priorities with proposal/stakeholders. |
| 5 Design | Architecture, persistence, ERD, pipelines, decisions, deployment | Detailed current code/schema/config and runbook | **STRONG** | Architecture, ERD, three pipelines, deployment | Decision register/config | Render diagrams; document model provisioning. |
| 6 Implementation | Iterative phases and significant code | Phase docs, code, tests, artefacts | **STRONG** | Timeline, selected screenshots | Phase map, algorithm/stack | Curate concise code extracts and change narrative. |
| 7 Legal/social/ethical/professional | Risks, mitigations, governance | Human oversight/local design/audit plus explicit gaps | **MEDIUM** | Data-flow/privacy boundary | Issue/evidence matrix | Authoritative legal/BCS research; consent, retention, security plans. |
| 8 Evaluation | Models, offline, quality, distance, replay, tests, live/usability | Rich Phase 7.1/8 artefacts; tests; human-reported live run | **STRONG** | Three result charts, setup | Result/test/live/limitation tables | Store test log and live evidence; complete model comparison; add usability method. |
| 9 Conclusion | Objective outcomes, contributions, limits, future work | Clear trace from objectives and evaluation | **STRONG** | Optional contribution summary | Achievement/future work | Draft only after proposal and literature reconciliation. |
| References | Complete verified citations | No complete verified list | **MISSING** | None | Reference tracker | Build bibliography from sources actually read. |
| Appendices | Supporting tests/results/schema/routes/runbook | Most technical material exists | **STRONG** | Full diagrams | Detailed aggregate tables | Sanitize private data and capture durable logs. |

---

# Final status

```text
FINAL_PRODUCT_ARCHITECTURE_DOCUMENTED = YES
REPORT_CHAPTER_1_EVIDENCE = MEDIUM
REPORT_CHAPTER_2_EVIDENCE = WEAK
REPORT_CHAPTER_3_EVIDENCE = WEAK
REPORT_CHAPTER_4_EVIDENCE = STRONG
REPORT_CHAPTER_5_EVIDENCE = STRONG
REPORT_CHAPTER_6_EVIDENCE = STRONG
REPORT_CHAPTER_7_EVIDENCE = MEDIUM
REPORT_CHAPTER_8_EVIDENCE = STRONG
REPORT_CHAPTER_9_EVIDENCE = STRONG

RECOGNITION_CORE_FROZEN = YES
FINAL_ATTENDANCE_LOCAL = YES
SUPABASE_FINAL_AUTHORITY = NO
LIVE_END_TO_END_EVIDENCE_EXISTS = YES
FULL_TEST_SUITE_EVIDENCE_EXISTS = YES
FINAL_REPORT_CAN_NOW_BE_DRAFTED = YES
```

Qualifications: the recognition core is operationally frozen for the documented demonstration, but the working tree is not evidenced as a tagged release and its thresholds remain provisional. Live end-to-end and current full-suite evidence exists as **human-reported verification**, not as durable repository logs.

## Top 10 remaining report-writing tasks

1. Recover the approved proposal and GDM template, then reconcile the exact aim, objectives, scope, and original schedule with Chapter 1.
2. Build and verify the scholarly literature review, starting with the topic map in Chapter 2; maintain a citation tracker and do not cite unread sources.
3. Research and source a factual criteria-based review of relevant attendance products/systems for Chapter 3.
4. Save a privacy-safe, durable full pytest run report that records environment, command, counts, subtests, warnings, and result.
5. Produce a signed/date-free or appropriately dated, privacy-safe end-to-end rehearsal record with steps, expected/actual outcomes, screenshots, and anonymized data.
6. Render the overall architecture, ERD, enrollment, recognition, attendance, deployment, activity, and sequence diagrams from the verified design in this audit.
7. Generate publication-quality charts from Phase 8B.2/8B.3 aggregate artefacts and verify every denominator/label against source files.
8. Decide how to present the incomplete model comparison: either run a separately authorized, controlled experiment later or explicitly report only the adapters/method and evidence gap.
9. Research authoritative legal, privacy, accessibility, and current BCS guidance; draft consent, retention, authentication, encryption, and governance recommendations without claiming compliance.
10. Draft chapters in evidence order (4–6, 8, 7, 1, 2–3, 9), then perform a final claim-safety, reference, figure/table numbering, and privacy review.
