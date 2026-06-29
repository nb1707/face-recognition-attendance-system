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

//auto load dashboard ưhen page is loaded
window.addEventListener("load", () => {
    loadDashboardSummary();
    loadAttendanceLogs();
    loadUnknownFaces();
    loadAttendanceCandidates();
    updateCameraInfo();
});