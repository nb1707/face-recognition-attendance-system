import json
from pathlib import Path
import shutil
import subprocess
import unittest


APP_JS = Path(__file__).resolve().parents[1] / "backend" / "static" / "app.js"
NODE = shutil.which("node")


HARNESS = r"""
const fs = require("fs");
const vm = require("vm");
const assert = require("assert");

const elements = new Map();
function makeElement(id) {
    return {
        id,
        disabled: false,
        hidden: false,
        value: id === "role" ? "student" : "",
        textContent: "",
        className: "",
        dataset: {},
        style: { width: "" },
        innerHTML: "",
        srcObject: null,
        videoWidth: 640,
        videoHeight: 480,
        listeners: {},
        addEventListener(type, callback) { this.listeners[type] = callback; },
        scrollIntoView() {},
        getContext() { return { drawImage() {} }; },
        toBlob(callback) { callback({}); },
        async play() {},
    };
}

const storage = new Map();
const context = {
    console,
    require,
    assert,
    process,
    setTimeout,
    clearTimeout,
    Promise,
    FormData: class { append() {} },
    document: {
        getElementById(id) {
            if (!elements.has(id)) elements.set(id, makeElement(id));
            return elements.get(id);
        },
        querySelectorAll() { return []; },
    },
    localStorage: {
        getItem(key) { return storage.has(key) ? storage.get(key) : null; },
        setItem(key, value) { storage.set(key, String(value)); },
        removeItem(key) { storage.delete(key); },
    },
    navigator: {
        mediaDevices: {
            async getUserMedia() { throw new Error("camera not expected"); },
        },
    },
    window: { addEventListener() {} },
    confirm() { return false; },
    async fetch() { throw new Error("unexpected fetch"); },
};
context.globalThis = context;
vm.createContext(context);
vm.runInContext(fs.readFileSync(APP_JS_PATH, "utf8"), context);

vm.runInContext(`
function testSession(sessionState = "CAPTURING") {
    return {
        session_id: "session-123",
        user_id: "student-1",
        full_name: "Student One",
        role: "student",
        state: sessionState,
        targets: { front: 5, left: 3, right: 3, up: 2, down: 2 },
        progress: { front: 1, left: 1, right: 1, up: 0, down: 0 },
        accepted_samples: [{
            sample_id: "sample-1",
            pose: "front",
            sample_number: 1,
            quality_score: .8,
            preview_url: "/preview",
        }],
        overall: { accepted: 3, target: 15 },
        next_recommended_pose: "front",
        finalize_eligible: sessionState === "REVIEW",
        updated_at: "now",
    };
}
function activateTestSession(sessionState = "CAPTURING") {
    const session = testSession(sessionState);
    localStorage.setItem("faceEnrollmentSessionId", session.session_id);
    applySession(session);
    return session;
}
`, context);
"""


@unittest.skipUnless(NODE, "Node.js is required for browser-script tests")
class EnrollmentCancelUiTests(unittest.TestCase):
    def run_javascript(self, body):
        source = (
            f"const APP_JS_PATH = {json.dumps(str(APP_JS))};\n"
            + HARNESS
            + "\n"
            + body
        )
        result = subprocess.run(
            [NODE, "-e", source],
            capture_output=True,
            text=True,
            timeout=10,
            check=False,
        )
        self.assertEqual(result.returncode, 0, result.stderr or result.stdout)

    def test_button_visibility_tracks_only_unfinished_cancellable_states(self):
        self.run_javascript(r"""
vm.runInContext(`
for (const sessionState of ["CAPTURING", "REVIEW", "ERROR"]) {
    activateTestSession(sessionState);
    assert.strictEqual(ui.cancelSessionBtn.hidden, false, sessionState);
    assert.strictEqual(ui.cancelSessionBtn.disabled, false, sessionState);
}
for (const sessionState of ["COMPLETE", "FINALIZING"]) {
    activateTestSession(sessionState);
    assert.strictEqual(ui.cancelSessionBtn.hidden, true, sessionState);
    assert.strictEqual(ui.cancelSessionBtn.disabled, true, sessionState);
}
state.session = null;
state.sessionId = null;
render();
assert.strictEqual(ui.cancelSessionBtn.hidden, true);
assert.strictEqual(ui.cancelSessionBtn.disabled, true);
assert.strictEqual(typeof ui.cancelSessionBtn.listeners.click, "function");
`, context);
""")

    def test_preview_pose_overlay_tracks_existing_session_pose_and_progress(self):
        self.run_javascript(r"""
vm.runInContext(`
const session = activateTestSession("CAPTURING");
session.next_recommended_pose = "left";
session.progress.left = 2;
applySession(session);

assert.strictEqual(ui.poseOverlay.hidden, false);
assert.strictEqual(ui.previewPoseLabel.textContent, "LEFT");
assert.strictEqual(ui.previewPoseProgress.textContent, "LEFT · 2 / 3");
assert.strictEqual(ui.previewInstruction.textContent, POSE_UI.left.instruction);
assert.strictEqual(ui.previewGuidanceDetail.textContent, POSE_UI.left.detail);
assert.strictEqual(ui.previewCaptureStatus.textContent, "Start camera to capture");

state.stream = { getTracks: () => [] };
renderControls();
assert.strictEqual(ui.previewCaptureStatus.textContent, "Camera ready");
state.autoRunning = true;
renderControls();
assert.strictEqual(ui.previewCaptureStatus.textContent, "Guided scan active");
`, context);
""")

    def test_cancel_confirmation_decline_preserves_session(self):
        self.run_javascript(r"""
vm.runInContext(`
(async () => {
    const session = activateTestSession("CAPTURING");
    let confirmation = null;
    let fetchCalls = 0;
    confirm = message => { confirmation = message; return false; };
    fetch = async () => { fetchCalls += 1; throw new Error("not expected"); };

    const cancelled = await cancelEnrollmentSession();

    assert.strictEqual(cancelled, false);
    assert.strictEqual(confirmation, CANCEL_ENROLLMENT_CONFIRMATION);
    assert.strictEqual(confirmation, "Cancel this unfinished enrollment session?\\nAccepted samples from this session will be deleted.\\nThe currently active gallery will not be changed.");
    assert.strictEqual(fetchCalls, 0);
    assert.strictEqual(state.session, session);
    assert.strictEqual(state.sessionId, "session-123");
})().catch(error => { console.error(error); process.exitCode = 1; });
`, context);
""")

    def test_success_calls_delete_and_resets_ui(self):
        self.run_javascript(r"""
vm.runInContext(`
(async () => {
    activateTestSession("REVIEW");
    let stoppedTracks = 0;
    state.stream = { getTracks: () => [{ stop() { stoppedTracks += 1; } }] };
    state.autoRunning = true;
    confirm = () => true;
    const calls = [];
    fetch = async (url, options) => {
        calls.push({ url, options });
        return { ok: true, status: 200, json: async () => ({ success: true }) };
    };

    const cancelled = await cancelEnrollmentSession();

    assert.strictEqual(cancelled, true);
    assert.strictEqual(calls.length, 1);
    assert.strictEqual(calls[0].url, "/enrollments/session-123");
    assert.strictEqual(calls[0].options.method, "DELETE");
    assert.strictEqual(state.session, null);
    assert.strictEqual(state.sessionId, null);
    assert.strictEqual(state.autoRunning, false);
    assert.strictEqual(state.stream, null);
    assert.strictEqual(stoppedTracks, 1);
    assert.strictEqual(localStorage.getItem("faceEnrollmentSessionId"), null);
    assert.strictEqual(ui.overallProgress.textContent, "0 / 15");
    assert.strictEqual(ui.currentPoseLabel.textContent, "FRONT");
    assert.ok(ui.sampleGroups.innerHTML.includes("Accepted samples will appear here"));
    assert.strictEqual(ui.userId.disabled, false);
    assert.strictEqual(ui.fullName.disabled, false);
    assert.strictEqual(ui.role.disabled, false);
    assert.strictEqual(ui.cancelSessionBtn.hidden, true);
})().catch(error => { console.error(error); process.exitCode = 1; });
`, context);
""")

    def test_failed_cancel_preserves_session_and_allows_retry(self):
        self.run_javascript(r"""
vm.runInContext(`
(async () => {
    const session = activateTestSession("CAPTURING");
    confirm = () => true;
    fetch = async () => ({
        ok: false,
        status: 500,
        json: async () => ({ success: false, message: "Cancellation failed." }),
    });

    const cancelled = await cancelEnrollmentSession();

    assert.strictEqual(cancelled, false);
    assert.strictEqual(state.session, session);
    assert.strictEqual(state.sessionId, "session-123");
    assert.strictEqual(localStorage.getItem("faceEnrollmentSessionId"), "session-123");
    assert.ok(ui.sampleGroups.innerHTML.includes("sample-card"));
    assert.strictEqual(ui.statusBox.textContent, "Cancellation failed.");
    assert.strictEqual(ui.cancelSessionBtn.hidden, false);
    assert.strictEqual(ui.cancelSessionBtn.disabled, false);
    assert.strictEqual(ui.userId.disabled, true);
})().catch(error => { console.error(error); process.exitCode = 1; });
`, context);
""")


if __name__ == "__main__":
    unittest.main()
