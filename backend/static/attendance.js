let recognitionInterval = null;

async function startAttendanceCamera() {
    const video = document.getElementById("attendanceVideo");

    const stream = await navigator.mediaDevices.getUserMedia({
        video: true,
        audio: false
    });

    video.srcObject = stream;
    document.getElementById("attendanceStatus").innerText = "Camera started";
}

async function recognizeCurrentFrame() {
    const video = document.getElementById("attendanceVideo");
    const canvas = document.getElementById("attendanceCanvas");
    const context = canvas.getContext("2d");

    //Camera ID
    // const cameraId = document.getElementById("cameraId").value;

    // let cameraLocation = "Unknown";
    // let cameraType = "unknown";

    // if (cameraId === "cam_gate_01") {
    //     cameraLocation = "School Entrance";
    //     cameraType = "gate";
    // }

    // if (cameraId === "cam_hallway_a_02") {
    //     cameraLocation = "Hallway A Floor 2";
    //     cameraType = "hallway";
    // }

    // get camera metadata based on selected camera ID
    const camera = getCameraMetadata();

    const cameraId = camera.cameraId;
    const cameraLocation = camera.location;
    const cameraType = camera.type;

    context.drawImage(video, 0, 0, canvas.width, canvas.height);

    const blob = await new Promise((resolve) => {
        canvas.toBlob(resolve, "image/jpeg");
    });

    const formData = new FormData();
    formData.append("image", blob, "attendance.jpg");

    formData.append("camera_id", cameraId);
    formData.append("camera_location", cameraLocation);
    formData.append("camera_type", cameraType);

    const response = await fetch("/recognize-face", {
        method: "POST",
        body: formData
    });

    const result = await response.json();

    if (result.success && result.results) {
        drawRecognitionOverlay(result.results);
    } else {
        drawRecognitionOverlay([]);
    }

    if (!result.success) {
        document.getElementById("attendanceStatus").innerText =
            result.message;
        return;
    }

    if (!result.results || result.results.length === 0) {
        document.getElementById("attendanceStatus").innerText =
            "No face detected";
        return;
    }

    const recognizedFaces = result.results.filter(face => face.recognized);
    const unknownFaces = result.results.filter(face => !face.recognized);

    document.getElementById("attendanceStatus").innerText =
        "Detected faces: " +
        result.total_faces +
        " | Recognized: " +
        recognizedFaces.length +
        " | Unknown: " +
        unknownFaces.length;

    if (recognizedFaces.some(face => face.logged)) {
        loadAttendanceLogs();
        loadDashboardSummary();
    }

    if (unknownFaces.length > 0) {
        loadUnknownFaces();
        loadDashboardSummary();
    }
}

function startRecognitionLoop() {
    if (recognitionInterval !== null) {
        return;
    }

    recognitionInterval = setInterval(() => {
        recognizeCurrentFrame();
    }, 3000);

    document.getElementById("attendanceStatus").innerText =
        "Recognition started";
}

function stopRecognitionLoop() {
    if (recognitionInterval !== null) {
        clearInterval(recognitionInterval);
        recognitionInterval = null;
    }

    document.getElementById("attendanceStatus").innerText =
        "Recognition stopped";
}

async function loadAttendanceLogs() {
    const response = await fetch("/attendance-logs");
    const result = await response.json();

    const logsList = document.getElementById("attendanceLogs");
    logsList.innerHTML = "";

    if (result.total === 0) {
        logsList.innerHTML = "<li>No attendance logs</li>";
        return;
    }

    result.logs.slice().reverse().forEach(log => {
        const item = document.createElement("li");
        item.innerText =
            log.timestamp.replace("T", " ").split(".")[0] +
            " | " +
            log.camera_location +
            " | " +
            log.user_id +
            " - " +
            log.full_name +
            " | " +
            log.role +
            " | confidence: " +
            Number(log.confidence).toFixed(3);

        logsList.appendChild(item);
    });
}

// function to clear attendance logs
async function clearAttendanceCandidates() {
    const confirmed = confirm("Are you sure you want to clear all attendance candidates?");

    if (!confirmed) {
        return;
    }

    const response = await fetch("/attendance-candidates", {
        method: "DELETE"
    });

    const result = await response.json();
    alert(result.message);

    loadAttendanceCandidates();
    loadDashboardSummary();
}

// Load unknown faces captured during recognition
async function loadUnknownFaces() {
    const response = await fetch("/unknown-faces");
    const result = await response.json();

    const list = document.getElementById("unknownFacesList");
    list.innerHTML = "";

    if (result.total === 0) {
        list.innerHTML = "<li>No unknown faces</li>";
        return;
    }

    result.files.slice().reverse().forEach(file => {
        const item = document.createElement("li");
        item.innerText = file.filename;
        list.appendChild(item);
    });
}

//draw bounding box and label on canvas
function drawRecognitionOverlay(results) {
    const canvas = document.getElementById("overlayCanvas");
    const ctx = canvas.getContext("2d");

    ctx.clearRect(0, 0, canvas.width, canvas.height);

    if (!results || results.length === 0) {
        return;
    }

    results.forEach(face => {
        if (!face.bbox) {
            return;
        }

        const bbox = face.bbox;

        const x = bbox.x1;
        const y = bbox.y1;
        const w = bbox.x2 - bbox.x1;
        const h = bbox.y2 - bbox.y1;

        ctx.lineWidth = 3;

        if (face.recognized) {
            ctx.strokeStyle = "lime";
            ctx.fillStyle = "lime";
        } else {
            ctx.strokeStyle = "red";
            ctx.fillStyle = "red";
        }

        ctx.strokeRect(x, y, w, h);

        const label = face.recognized
            ? `${face.full_name} ${face.user_id} ${Number(face.confidence).toFixed(2)}`
            : `Unknown ${Number(face.confidence).toFixed(2)}`;

        ctx.font = "16px Arial";
        ctx.fillText(label, x, y > 20 ? y - 8 : y + h + 20);
    });
}

//Load attendance logs and unknown faces on page load
async function loadDashboardSummary() {
    const response = await fetch("/dashboard-summary");
    const result = await response.json();

    if (!result.success) {
        return;
    }

    document.getElementById("registeredUsersCount").innerText =
        result.registered_users_count;

    document.getElementById("todayPresentCount").innerText =
        result.today_total_present;

    document.getElementById("unknownDetectionsCount").innerText =
        result.unknown_detections;
}

// function to clear attendance logs
async function clearAttendanceLogs() {
    const confirmed = confirm("Are you sure you want to clear all attendance logs?");

    if (!confirmed) {
        return;
    }

    const response = await fetch("/attendance-logs", {
        method: "DELETE"
    });

    const result = await response.json();
    alert(result.message);

    loadAttendanceLogs();
    loadDashboardSummary();
}

// function to clear unknown faces
async function clearUnknownFaces() {
    const confirmed = confirm("Are you sure you want to clear all unknown faces?");

    if (!confirmed) {
        return;
    }

    const response = await fetch("/unknown-faces", {
        method: "DELETE"
    });

    const result = await response.json();
    alert(result.message);

    loadUnknownFaces();
    loadDashboardSummary();
}

async function loadAttendanceCandidates() {
    const response = await fetch("/attendance-candidates");
    const result = await response.json();

    const list = document.getElementById("attendanceCandidatesList");
    list.innerHTML = "";

    if (result.total === 0) {
        list.innerHTML = "<li>No attendance candidates</li>";
        return;
    }

    result.candidates.slice().reverse().forEach(candidate => {
        const item = document.createElement("li");

        const time = candidate.timestamp.replace("T", " ").split(".")[0];

        item.innerText =
            time +
            " | " +
            candidate.camera_id +
            " | " +
            candidate.camera_location +
            " | " +
            candidate.user_id +
            " - " +
            candidate.full_name +
            " | " +
            candidate.status +
            " | confidence: " +
            Number(candidate.confidence).toFixed(3);

        list.appendChild(item);
    });
}

function getCameraMetadata() {
    const cameraId = document.getElementById("cameraId").value;

    if (cameraId === "cam_gate_01") {
        return {
            cameraId: "cam_gate_01",
            location: "School Entrance",
            type: "gate",
            purpose: "Detect arrivals at the school entrance"
        };
    }

    if (cameraId === "cam_hallway_a_02") {
        return {
            cameraId: "cam_hallway_a_02",
            location: "Hallway A Floor 2",
            type: "hallway",
            purpose: "Monitor student and staff movement in hallway"
        };
    }

    return {
        cameraId: "unknown",
        location: "Unknown",
        type: "unknown",
        purpose: "Unknown camera"
    };
}

function updateCameraInfo() {
    const camera = getCameraMetadata();

    document.getElementById("cameraInfo").innerHTML =
        "Camera ID: " + camera.cameraId + "<br>" +
        "Location: " + camera.location + "<br>" +
        "Type: " + camera.type + "<br>" +
        "Purpose: " + camera.purpose;
}

async function checkCloudHealth() {
    const response = await fetch("/cloud-health");
    const result = await response.json();

    document.getElementById("cloudHealthResult").innerText =
        JSON.stringify(result, null, 2);
}

// ==================================================
// HIKVISION RTSP CAMERA
// ==================================================

let hikvisionStatusInterval = null;
let hikvisionDashboardInterval = null;


/**
 * Get Hikvision UI elements from attendance.html.
 */
function getHikvisionElements() {
    return {
        feed: document.getElementById(
            "hikvisionFeed"
        ),

        status: document.getElementById(
            "cameraStatus"
        ),

        message: document.getElementById(
            "cameraMessage"
        ),

        startButton: document.getElementById(
            "startCameraButton"
        ),

        stopButton: document.getElementById(
            "stopCameraButton"
        ),

        reloadButton: document.getElementById(
            "reloadCameraDataButton"
        )
    };
}


/**
 * Start the RTSP camera in FastAPI.
 *
 * Backend flow:
 * POST /camera/start
 * GET /camera/feed
 */
async function startHikvisionCamera() {
    const elements = getHikvisionElements();

    if (
        !elements.feed ||
        !elements.status ||
        !elements.message
    ) {
        console.error(
            "Hikvision camera HTML elements are missing."
        );

        return;
    }

    elements.status.textContent = "Starting...";
    elements.message.textContent =
        "Connecting to Hikvision RTSP stream...";

    if (elements.startButton) {
        elements.startButton.disabled = true;
    }

    try {
        const response = await fetch(
            "/camera/start",
            {
                method: "POST"
            }
        );

        const result = await response.json();

        if (
            !response.ok ||
            !result.success
        ) {
            throw new Error(
                result.message ||
                "Could not start Hikvision camera."
            );
        }

        /*
         * The <img> element displays the MJPEG stream
         * generated by FastAPI.
         *
         * Date.now() avoids browser caching.
         */
        elements.feed.src =
            "/camera/feed?t=" + Date.now();

        elements.status.textContent = "Live";

        elements.message.textContent =
            result.message ||
            "Hikvision camera started successfully.";

        if (elements.stopButton) {
            elements.stopButton.disabled = false;
        }

        // Update dashboard after camera starts.
        await loadDashboardSummary();
        await loadAttendanceLogs();

    } catch (error) {
        console.error(
            "Start Hikvision camera failed:",
            error
        );

        elements.status.textContent = "Error";

        elements.message.textContent =
            error.message;

        elements.feed.removeAttribute("src");

    } finally {
        if (elements.startButton) {
            elements.startButton.disabled = false;
        }
    }
}


/**
 * Stop backend RTSP processing.
 */
async function stopHikvisionCamera() {
    const elements = getHikvisionElements();

    if (
        !elements.feed ||
        !elements.status ||
        !elements.message
    ) {
        return;
    }

    elements.status.textContent = "Stopping...";
    elements.message.textContent =
        "Stopping Hikvision camera...";

    if (elements.stopButton) {
        elements.stopButton.disabled = true;
    }

    try {
        const response = await fetch(
            "/camera/stop",
            {
                method: "POST"
            }
        );

        const result = await response.json();

        if (
            !response.ok ||
            !result.success
        ) {
            throw new Error(
                result.message ||
                "Could not stop Hikvision camera."
            );
        }

        /*
         * Remove the MJPEG URL so the browser
         * closes its HTTP stream connection.
         */
        elements.feed.removeAttribute("src");

        elements.status.textContent = "Stopped";

        elements.message.textContent =
            result.message ||
            "Hikvision camera stopped.";

    } catch (error) {
        console.error(
            "Stop Hikvision camera failed:",
            error
        );

        elements.status.textContent = "Error";

        elements.message.textContent =
            error.message;

    } finally {
        if (elements.stopButton) {
            elements.stopButton.disabled = false;
        }
    }
}


/**
 * Reload users and mean embeddings from Supabase.
 *
 * Call this after:
 * - registering a new user;
 * - generating new embeddings;
 * - updating existing user embeddings.
 */
async function reloadHikvisionRecognitionData() {
    const elements = getHikvisionElements();

    if (!elements.message) {
        return;
    }

    elements.message.textContent =
        "Reloading users and embeddings...";

    if (elements.reloadButton) {
        elements.reloadButton.disabled = true;
    }

    try {
        const response = await fetch(
            "/camera/reload-recognition-data",
            {
                method: "POST"
            }
        );

        const result = await response.json();

        if (
            !response.ok ||
            !result.success
        ) {
            throw new Error(
                result.message ||
                "Could not reload recognition data."
            );
        }

        const usersCount =
            result.cached_users ??
            result.registered_users ??
            0;

        const embeddingsCount =
            result.cached_embeddings ??
            result.mean_embeddings ??
            0;

        elements.message.textContent =
            "Recognition data reloaded. " +
            "Users: " +
            usersCount +
            ", mean embeddings: " +
            embeddingsCount +
            ".";

    } catch (error) {
        console.error(
            "Reload recognition data failed:",
            error
        );

        elements.message.textContent =
            error.message;

    } finally {
        if (elements.reloadButton) {
            elements.reloadButton.disabled = false;
        }
    }
}


/**
 * Check whether FastAPI camera service is running.
 */
async function checkHikvisionCameraStatus() {
    const elements = getHikvisionElements();

    if (
        !elements.feed ||
        !elements.status ||
        !elements.message
    ) {
        return;
    }

    try {
        const response = await fetch(
            "/camera/status",
            {
                cache: "no-store"
            }
        );

        const result = await response.json();

        if (!response.ok) {
            throw new Error(
                result.message ||
                "Could not read camera status."
            );
        }

        if (result.running) {
            elements.status.textContent = "Live";

            /*
             * Reconnect the browser to MJPEG if
             * backend is running but img has no URL.
             */
            if (
                !elements.feed.getAttribute("src")
            ) {
                elements.feed.src =
                    "/camera/feed?t=" +
                    Date.now();
            }

            if (!result.last_error) {
                elements.message.textContent =
                    "Camera is running. Model: " +
                    (result.model_name || "unknown") +
                    ". Cached users: " +
                    Number(
                        result.cached_users || 0
                    ) +
                    ". Cached embeddings: " +
                    Number(
                        result.cached_embeddings || 0
                    ) +
                    ".";
            }

        } else {
            elements.status.textContent = "Stopped";

            elements.feed.removeAttribute("src");

            if (!result.last_error) {
                elements.message.textContent =
                    "Camera is currently stopped.";
            }
        }

        if (result.last_error) {
            elements.status.textContent = "Error";

            elements.message.textContent =
                result.last_error;
        }

    } catch (error) {
        console.error(
            "Camera status check failed:",
            error
        );

        elements.status.textContent =
            "Disconnected";

        elements.message.textContent =
            "Cannot connect to the camera API: " +
            error.message;
    }
}

/*Connect HTML buttons with Hikvision functions.*/
function initializeHikvisionCameraControls() {
    const elements = getHikvisionElements();

    if (
        !elements.startButton ||
        !elements.stopButton ||
        !elements.reloadButton
    ) {
        console.warn(
            "Hikvision buttons were not found."
        );
        return;
    }

    elements.startButton.addEventListener(
        "click",
        startHikvisionCamera
    );

    elements.stopButton.addEventListener(
        "click",
        stopHikvisionCamera
    );

    elements.reloadButton.addEventListener(
        "click",
        reloadHikvisionRecognitionData
    );

    /*Check backend camera status every 5 seconds.*/
    if (hikvisionStatusInterval === null) {
        hikvisionStatusInterval = setInterval(
            checkHikvisionCameraStatus,
            5000
        );
    }

    /*
     * RTSP recognition happens in the backend.
     * Frontend only reloads logs and dashboard.
     */
    if (hikvisionDashboardInterval === null) {
        hikvisionDashboardInterval = setInterval(
            async () => {
                await loadAttendanceLogs();
                await loadDashboardSummary();
            },
            5000
        );
    }

    checkHikvisionCameraStatus();
}

//auto load dashboard ưhen page is loaded
window.addEventListener("load", () => {
    loadDashboardSummary();
    loadAttendanceLogs();
    loadUnknownFaces();
    loadAttendanceCandidates();
    updateCameraInfo();
    initializeHikvisionCameraControls();
});