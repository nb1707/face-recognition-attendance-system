const POSE_UI = {
    front: { label: "Front", instruction: "Look straight at the camera", detail: "Keep your face centered and hold still.", arrow: "●" },
    left: { label: "Left", instruction: "Turn your face gently to the left", detail: "Use a mild angle; keep both eyes visible.", arrow: "←" },
    right: { label: "Right", instruction: "Turn your face gently to the right", detail: "Use a mild angle; keep both eyes visible.", arrow: "→" },
    up: { label: "Up", instruction: "Raise your chin slightly", detail: "Keep looking toward the camera; do not tilt too far.", arrow: "↑" },
    down: { label: "Down", instruction: "Lower your chin slightly", detail: "Keep looking toward the camera; do not tilt too far.", arrow: "↓" }
};

const ERROR_MESSAGES = {
    NO_FACE: "No face detected. Move closer and face the camera.", MULTIPLE_FACES: "Only one person should be visible.",
    FACE_TOO_SMALL: "Move closer to the camera.", BLURRED_FACE: "Hold still and try again.",
    POOR_LIGHTING: "Improve the lighting and try again.", LOW_DETECTOR_CONFIDENCE: "Make sure your face is clear and fully visible.",
    INVALID_LANDMARKS: "Face landmarks were unclear. Face the camera and try again.",
    WRONG_POSE: "Adjust your face to match the requested pose.", FACE_NOT_CENTERED: "Center your face inside the oval.",
    EXACT_DUPLICATE: "That is the same frame as an accepted sample. Move naturally before retrying.",
    NEAR_DUPLICATE: "Sample too similar to a previous sample. Vary your expression or position slightly.",
    POSE_QUOTA_REACHED: "That pose already has enough samples.", SESSION_NOT_READY: "Capture all required samples before finalizing.",
    STORAGE_ERROR: "Local storage failed. Your accepted session data has been retained where possible.",
    EMBEDDING_ERROR: "Embedding generation failed. Review samples and retry finalization.",
    GALLERY_PUBLISH_ERROR: "Gallery publication failed. Samples are retained and finalization can be retried."
};

const DEFAULT_ENROLLMENT_TARGET = 15;
const CANCELLABLE_SESSION_STATES = new Set(["CAPTURING", "REVIEW", "ERROR"]);
const CANCEL_ENROLLMENT_CONFIRMATION = "Cancel this unfinished enrollment session?\nAccepted samples from this session will be deleted.\nThe currently active gallery will not be changed.";

const ui = Object.fromEntries(["video", "canvas", "cameraPlaceholder", "startCameraBtn", "stopCameraBtn", "captureBtn",
    "autoBtn", "stopAutoBtn", "currentPoseLabel", "poseArrow", "instruction", "guidanceDetail", "userId", "fullName",
    "role", "startSessionBtn", "sessionBadge", "progressList", "overallProgress", "overallBar", "statusBox", "reviewBtn",
    "finalizeBtn", "cancelSessionBtn", "newUserBtn", "sampleGroups", "reviewSection", "finalizeSteps", "poseOverlay",
    "previewPoseLabel", "previewPoseProgress", "previewCaptureStatus", "previewInstruction", "previewGuidanceDetail"].map(id => [id, document.getElementById(id)]));

const state = { sessionId: null, session: null, stream: null, requestInFlight: false, autoRunning: false, autoAttempts: 0, autoRejects: 0 };

function setStatus(message, kind = "info") { ui.statusBox.className = `status-box ${kind}`; ui.statusBox.textContent = message; }

async function api(url, options = {}) {
    const response = await fetch(url, options);
    let payload;
    try { payload = await response.json(); }
    catch { payload = { success: false, error_code: "NETWORK_ERROR", message: `Server returned ${response.status}.` }; }
    if (!response.ok || payload.success === false) {
        const error = new Error(payload.message || "Request failed."); error.payload = payload; error.status = response.status; throw error;
    }
    return payload;
}

async function startCamera() {
    if (state.stream) stopCamera();
    try {
        state.stream = await navigator.mediaDevices.getUserMedia({ video: { width: { ideal: 1280 }, height: { ideal: 720 }, facingMode: "user" }, audio: false });
        ui.video.srcObject = state.stream; await ui.video.play(); ui.cameraPlaceholder.hidden = true;
        ui.startCameraBtn.disabled = true; ui.stopCameraBtn.disabled = false;
        setStatus("Camera ready. Follow the requested pose guidance.", "success"); renderControls();
    } catch (error) {
        state.stream = null; setStatus(`Camera could not start: ${error.message || "permission or device error"}`, "error"); renderControls();
    }
}

function stopCamera() {
    stopAutoCapture(); if (state.stream) state.stream.getTracks().forEach(track => track.stop());
    state.stream = null; ui.video.srcObject = null; ui.cameraPlaceholder.hidden = false;
    ui.startCameraBtn.disabled = false; ui.stopCameraBtn.disabled = true; renderControls();
}

async function startSession(allowReenroll = false) {
    const request = { user_id: ui.userId.value.trim(), full_name: ui.fullName.value.trim(), role: ui.role.value, allow_reenroll: allowReenroll };
    lockRequest(true);
    try {
        const session = await api("/enrollments", { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify(request) });
        state.sessionId = session.session_id; localStorage.setItem("faceEnrollmentSessionId", state.sessionId); applySession(session);
        setStatus("Enrollment session created. Start the camera and capture the requested pose.", "success");
    } catch (error) {
        if (error.payload?.error_code === "USER_ALREADY_ENROLLED" && !allowReenroll) {
            const confirmed = confirm("This user already has a completed local enrollment. Create a new enrollment version? The active gallery will not change until finalization succeeds.");
            if (confirmed) { lockRequest(false); await startSession(true); return; }
        }
        showRequestError(error);
    } finally { lockRequest(false); }
}

async function resumeSession() {
    const sessionId = localStorage.getItem("faceEnrollmentSessionId"); if (!sessionId) return;
    try {
        const session = await api(`/enrollments/${sessionId}`); state.sessionId = sessionId; applySession(session);
        setStatus(session.state === "COMPLETE" ? "Completed enrollment restored." : "Partial enrollment restored from local storage.", "success");
    } catch { localStorage.removeItem("faceEnrollmentSessionId"); }
}

function frameBlob() {
    if (!state.stream || !ui.video.videoWidth) throw new Error("Camera is not ready.");
    ui.canvas.width = ui.video.videoWidth; ui.canvas.height = ui.video.videoHeight;
    ui.canvas.getContext("2d").drawImage(ui.video, 0, 0);
    return new Promise((resolve, reject) => ui.canvas.toBlob(blob => blob ? resolve(blob) : reject(new Error("Could not capture camera frame.")), "image/jpeg", 0.92));
}

async function captureSample(fromAuto = false) {
    if (state.requestInFlight || !state.session || !state.stream) return false;
    const pose = state.session.next_recommended_pose; if (!pose) return false;
    lockRequest(true); setStatus(`Validating ${POSE_UI[pose].label.toLowerCase()} sample…`, "working");
    try {
        const blob = await frameBlob(); const form = new FormData(); form.append("pose", pose); form.append("image", blob, `${pose}.jpg`);
        const result = await api(`/enrollments/${state.sessionId}/samples`, { method: "POST", body: form });
        applySession(result.session); setStatus(`${POSE_UI[pose].label} sample accepted. ${result.remaining_count} remaining for this pose.`, "success"); return true;
    } catch (error) { showRequestError(error, "warning"); return false; }
    finally { lockRequest(false); if (!fromAuto) renderControls(); }
}

async function startAutoCapture() {
    if (state.autoRunning || !state.session || !state.stream) return;
    state.autoRunning = true; state.autoAttempts = 0; state.autoRejects = 0; renderControls();
    while (state.autoRunning && state.session?.next_recommended_pose && state.autoAttempts < 60 && state.autoRejects < 10) {
        await delay(1200); if (!state.autoRunning) break; state.autoAttempts++;
        const accepted = await captureSample(true); state.autoRejects = accepted ? 0 : state.autoRejects + 1; renderControls();
    }
    if (state.autoRunning && state.autoRejects >= 10) setStatus("Guided scan paused after repeated rejections. Adjust using the latest feedback and resume when ready.", "warning");
    if (state.autoRunning && state.session?.state === "REVIEW")
        setStatus(`All ${state.session.overall.target} samples are accepted. Review them before finalizing.`, "success");
    stopAutoCapture();
}

function stopAutoCapture() { state.autoRunning = false; renderControls(); }

async function deleteSample(sampleId) {
    if (state.requestInFlight || !confirm("Delete this accepted sample and retake its pose?")) return;
    lockRequest(true);
    try { const session = await api(`/enrollments/${state.sessionId}/samples/${sampleId}`, { method: "DELETE" }); applySession(session); setStatus("Sample deleted. Capture a replacement for the requested pose.", "warning"); }
    catch (error) { showRequestError(error); } finally { lockRequest(false); }
}

async function finalizeEnrollment() {
    if (!state.session?.finalize_eligible || state.requestInFlight) return;
    stopAutoCapture(); lockRequest(true); ui.finalizeSteps.hidden = false;
    setStatus("Finalizing locally: validating samples and generating AdaFace embeddings…", "working");
    try {
        const result = await api(`/enrollments/${state.sessionId}/finalize`, { method: "POST" }); applySession(result.session);
        setStatus(`Enrollment complete. Local gallery generation ${result.generation} is active.`, "success");
    } catch (error) {
        showRequestError(error); try { applySession(await api(`/enrollments/${state.sessionId}`)); } catch { /* retain review */ }
    } finally { ui.finalizeSteps.hidden = true; lockRequest(false); }
}

function canCancelEnrollmentSession() {
    return Boolean(state.sessionId && state.session && CANCELLABLE_SESSION_STATES.has(state.session.state));
}

function resetEnrollmentUi() {
    stopAutoCapture();
    state.sessionId = null; state.session = null; state.autoAttempts = 0; state.autoRejects = 0;
    localStorage.removeItem("faceEnrollmentSessionId");
    ui.userId.value = ""; ui.fullName.value = ""; ui.role.value = "student";
    stopCamera();
    render();
}

async function cancelEnrollmentSession() {
    if (state.requestInFlight || !canCancelEnrollmentSession()) return false;
    if (!confirm(CANCEL_ENROLLMENT_CONFIRMATION)) return false;

    const sessionId = state.sessionId;
    stopAutoCapture(); lockRequest(true);
    setStatus("Cancelling unfinished enrollment sessionâ€¦", "working");
    try {
        await api(`/enrollments/${sessionId}`, { method: "DELETE" });
    } catch (error) {
        showRequestError(error); lockRequest(false); return false;
    }

    resetEnrollmentUi(); lockRequest(false);
    setStatus("Unfinished enrollment session cancelled. The active gallery was not changed.", "success");
    return true;
}

async function registerNewUser() {
    if (canCancelEnrollmentSession()) {
        await cancelEnrollmentSession();
        return;
    }
    if (state.session?.state === "FINALIZING") {
        setStatus("Enrollment is being finalized and cannot be cancelled.", "warning");
        return;
    }
    resetEnrollmentUi();
    setStatus("Ready to create a new local enrollment session.", "info"); render();
}

function applySession(session) {
    state.session = session; state.sessionId = session.session_id; ui.userId.value = session.user_id;
    ui.fullName.value = session.full_name; ui.role.value = session.role; render();
}

function render() {
    renderProgress(); renderSamples(); renderGuidance(); renderControls();
    const active = Boolean(state.session); [ui.userId, ui.fullName, ui.role, ui.startSessionBtn].forEach(element => element.disabled = active || state.requestInFlight);
    ui.sessionBadge.textContent = active ? state.session.state : "No session"; ui.sessionBadge.dataset.state = active ? state.session.state : "";
}

function renderProgress() {
    if (!state.session) { ui.progressList.innerHTML = '<p class="empty-copy">Targets load from the server when a session starts.</p>'; ui.overallProgress.textContent = `0 / ${DEFAULT_ENROLLMENT_TARGET}`; ui.overallBar.style.width = "0%"; return; }
    ui.progressList.innerHTML = Object.entries(state.session.targets).map(([pose, target]) => {
        const value = state.session.progress[pose]; return `<div class="progress-row"><span>${POSE_UI[pose].label}</span><div class="mini-track"><i style="width:${value/target*100}%"></i></div><strong>${value}/${target}</strong></div>`;
    }).join("");
    const { accepted, target } = state.session.overall; ui.overallProgress.textContent = `${accepted} / ${target}`; ui.overallBar.style.width = `${accepted / target * 100}%`;
}

function renderGuidance() {
    const pose = state.session?.next_recommended_pose || "front"; const guide = POSE_UI[pose]; const reviewing = state.session?.state === "REVIEW";
    ui.currentPoseLabel.textContent = reviewing ? "REVIEW" : guide.label.toUpperCase(); ui.poseArrow.textContent = reviewing ? "✓" : guide.arrow;
    ui.instruction.textContent = reviewing ? "Review all accepted samples" : guide.instruction;
    ui.guidanceDetail.textContent = reviewing ? "Delete and retake any sample, or finalize when satisfied." : guide.detail;
    ui.poseOverlay.hidden = !state.session;
    ui.previewPoseLabel.textContent = reviewing ? "REVIEW" : guide.label.toUpperCase();
    ui.previewInstruction.textContent = reviewing ? "Review all accepted samples" : guide.instruction;
    ui.previewGuidanceDetail.textContent = reviewing ? "Delete and retake any sample, or finalize when satisfied." : guide.detail;
    if (state.session) {
        const progress = reviewing
            ? state.session.overall
            : { accepted: state.session.progress[pose], target: state.session.targets[pose] };
        ui.previewPoseProgress.textContent = `${reviewing ? "REVIEW" : guide.label.toUpperCase()} · ${progress.accepted} / ${progress.target}`;
    }
    renderPreviewCaptureStatus();
}

function renderPreviewCaptureStatus() {
    if (!state.session) ui.previewCaptureStatus.textContent = "Create a session to begin";
    else if (state.session.state === "REVIEW") ui.previewCaptureStatus.textContent = "Ready to review";
    else if (state.requestInFlight) ui.previewCaptureStatus.textContent = "Checking sample…";
    else if (state.autoRunning) ui.previewCaptureStatus.textContent = "Guided scan active";
    else if (state.stream) ui.previewCaptureStatus.textContent = "Camera ready";
    else ui.previewCaptureStatus.textContent = "Start camera to capture";
}

function renderSamples() {
    if (!state.session?.accepted_samples.length) { ui.sampleGroups.innerHTML = '<p class="empty-copy">Accepted samples will appear here.</p>'; return; }
    ui.sampleGroups.innerHTML = Object.keys(state.session.targets).map(pose => {
        const samples = state.session.accepted_samples.filter(sample => sample.pose === pose);
        return `<section class="sample-group"><h3>${POSE_UI[pose].label} <span>${samples.length}/${state.session.targets[pose]}</span></h3><div class="thumbnail-row">${samples.map(sample => `
            <article class="sample-card"><img src="${sample.preview_url}?v=${encodeURIComponent(state.session.updated_at)}" alt="${pose} sample ${sample.sample_number}">
            <div><strong>${POSE_UI[pose].label} ${sample.sample_number}</strong><small>Quality ${(sample.quality_score*100).toFixed(0)}%</small></div>
            <button type="button" class="delete-sample" data-sample-id="${sample.sample_id}" ${state.session.state === "COMPLETE" ? "disabled" : ""}>Delete</button></article>`).join("") || '<p class="empty-copy">No accepted samples yet.</p>'}</div></section>`;
    }).join("");
    document.querySelectorAll(".delete-sample").forEach(button => button.addEventListener("click", () => deleteSample(button.dataset.sampleId)));
}

function renderControls() {
    const capturing = state.session && ["CAPTURING", "REVIEW", "ERROR"].includes(state.session.state);
    const cancellable = canCancelEnrollmentSession();
    const canCapture = capturing && state.stream && state.session.next_recommended_pose && !state.requestInFlight && !state.autoRunning;
    ui.captureBtn.disabled = !canCapture; ui.autoBtn.disabled = !canCapture; ui.autoBtn.hidden = state.autoRunning; ui.stopAutoBtn.hidden = !state.autoRunning;
    ui.reviewBtn.disabled = !state.session?.accepted_samples.length;
    ui.finalizeBtn.disabled = !state.session?.finalize_eligible || state.requestInFlight || state.session?.state === "COMPLETE";
    ui.cancelSessionBtn.hidden = !cancellable;
    ui.cancelSessionBtn.disabled = !cancellable || state.requestInFlight;
    renderPreviewCaptureStatus();
}

function lockRequest(locked) {
    state.requestInFlight = locked;
    ui.startSessionBtn.disabled = locked || Boolean(state.session);
    ui.userId.disabled = locked || Boolean(state.session);
    ui.fullName.disabled = locked || Boolean(state.session);
    ui.role.disabled = locked || Boolean(state.session);
    renderControls();
}
function showRequestError(error, kind = "error") { const payload = error.payload || {}; setStatus(payload.guidance || ERROR_MESSAGES[payload.error_code] || payload.message || error.message || "Request failed.", kind); }
const delay = ms => new Promise(resolve => setTimeout(resolve, ms));

ui.startCameraBtn.addEventListener("click", startCamera); ui.stopCameraBtn.addEventListener("click", stopCamera);
ui.startSessionBtn.addEventListener("click", () => startSession(false)); ui.captureBtn.addEventListener("click", () => captureSample(false));
ui.autoBtn.addEventListener("click", startAutoCapture); ui.stopAutoBtn.addEventListener("click", stopAutoCapture);
ui.reviewBtn.addEventListener("click", () => ui.reviewSection.scrollIntoView({ behavior: "smooth" }));
ui.finalizeBtn.addEventListener("click", finalizeEnrollment); ui.cancelSessionBtn.addEventListener("click", cancelEnrollmentSession);
ui.newUserBtn.addEventListener("click", registerNewUser);
window.addEventListener("beforeunload", () => { if (state.stream) state.stream.getTracks().forEach(track => track.stop()); });

render(); resumeSession();
