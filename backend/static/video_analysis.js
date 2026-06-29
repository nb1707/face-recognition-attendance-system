function getVideoCameraMetadata() {
    const cameraId = document.getElementById("cameraId").value;

    if (cameraId === "front_cam_01") {
        return {
            cameraId: "front_cam_01",
            location: "front cam",
            purpose: "Capture full classroom view"
        };
    }

    if (cameraId === "side_cam_02") {
        return {
            cameraId: "side_cam_02",
            location: "side cam",
            purpose: "Capture side angle view"
        };
    }

    return {
        cameraId: "unknown",
        location: "Unknown",
        purpose: "Unknown camera"
    };
}

function updateVideoCameraInfo() {
    const camera = getVideoCameraMetadata();

    document.getElementById("videoCameraInfo").innerHTML =
        "Camera ID: " + camera.cameraId + "<br>" +
        "Location: " + camera.location + "<br>" +
        "Purpose: " + camera.purpose;
}

async function processVideo() {
    const sessionId = document.getElementById("sessionId").value.trim();
    const className = document.getElementById("className").value.trim();
    const sampleInterval = document.getElementById("sampleInterval").value;
    const videoFile = document.getElementById("videoFile").files[0];

    const camera = getVideoCameraMetadata();

    if (!sessionId || !className) {
        alert("Please enter Session ID and Class Name.");
        return;
    }

    if (!videoFile) {
        alert("Please select a video file.");
        return;
    }

    const formData = new FormData();
    formData.append("session_id", sessionId);
    formData.append("class_name", className);
    formData.append("camera_id", camera.cameraId);
    formData.append("camera_location", camera.location);
    formData.append("sample_interval_seconds", sampleInterval);
    formData.append("video", videoFile);

    document.getElementById("videoStatus").innerText =
        "Processing video... Please wait.";

    document.getElementById("videoResult").innerText = "";

    try {
        const response = await fetch("/process-video", {
            method: "POST",
            body: formData
        });

        const result = await response.json();

        if (result.success) {
            document.getElementById("videoStatus").innerText =
                "Video processed successfully.";
        } else {
            document.getElementById("videoStatus").innerText =
                "Video processing failed.";
        }

        document.getElementById("videoResult").innerText =
            JSON.stringify(result, null, 2);

    } catch (error) {
        document.getElementById("videoStatus").innerText =
            "Error while processing video.";

        document.getElementById("videoResult").innerText = error;
    }
}

function goToSummary() {
    const sessionId = document.getElementById("sessionId").value.trim();

    if (!sessionId) {
        alert("Please enter Session ID.");
        return;
    }

    window.location.href = "/video-summary-page?session_id=" + sessionId;
}

window.addEventListener("load", () => {
    updateVideoCameraInfo();
});