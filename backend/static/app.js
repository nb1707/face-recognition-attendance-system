
const steps = [
    { pose: "front", text: "Please look straight at the camera" },
    { pose: "left", text: "Please slowly turn your face to the left" },
    { pose: "right", text: "Please slowly turn your face to the right" }
];
let currentStep = 0;
let overwriteConfirmedUserId = null;

async function startCamera() {
    const video = document.getElementById("video");

    const stream = await navigator.mediaDevices.getUserMedia({
        video: true,
        audio: false
    });

    video.srcObject = stream;
    document.getElementById("status").innerText = "Camera started";
}

//Auto scan function
async function startAutoScan() {
    const valid = await validateUserBeforeScan();

    if (!valid) {
        return;
    }

    setScanningState(true);

    try {
        document.getElementById("status").innerText =
            "Auto scan started. Please follow the instructions.";

        while (currentStep < steps.length) {
            document.getElementById("instruction").innerText =
                "Step " + (currentStep + 1) + ": " + steps[currentStep].text;

            document.getElementById("guideMessage").innerText = steps[currentStep].text;

            await showCountdown(3);

            const success = await captureAndUpload(true);

            if (!success) {
                document.getElementById("status").innerText =
                    "Capture failed. Please adjust your face and try again.";
                break;
            }

            await sleep(1000);
        }
    } finally {
        if (currentStep < steps.length) {
            setScanningState(false);
        }
    }
}

//countdown function
async function showCountdown(seconds) {
    for (let i = seconds; i > 0; i--) {
        document.getElementById("countdown").innerText = i;
        await sleep(1000);
    }

    document.getElementById("countdown").innerText = "Capturing...";
    await sleep(500);
    document.getElementById("countdown").innerText = "";
}

async function captureAndUpload(skipUserCheck = false) {
    const userId = document.getElementById("userId").value.trim();
    const fullName = document.getElementById("fullName").value.trim();
    const role = document.getElementById("role").value.trim();

    if (!userId || !fullName || !role) {
        document.getElementById("status").innerText =
            "Please enter User ID, Full Name, and Role before capturing.";
        return false;
    }

    if (!skipUserCheck) {
        const valid = await validateUserBeforeScan();

        if (!valid) {
            return false;
        }
    }

    if (currentStep >= steps.length) {
        return false;
    }

    const stepName = steps[currentStep].pose;

    const video = document.getElementById("video");
    const canvas = document.getElementById("canvas");
    const context = canvas.getContext("2d");

    context.drawImage(video, 0, 0, canvas.width, canvas.height);

    const blob = await new Promise((resolve) => {
        canvas.toBlob(resolve, "image/jpeg");
    });


    //send to backend with form data
    const formData = new FormData();
    formData.append("user_id", userId);
    formData.append("pose", stepName);
    formData.append("full_name", fullName);
    formData.append("role", role);
    formData.append("image", blob, stepName + ".jpg");

    const response = await fetch("/register-face", {
        method: "POST",
        body: formData
    });

    const result = await response.json();

    if (result.success) {
        document.getElementById("status").innerText =
            "Saved " + stepName + " successfully";

        updateProgress(stepName);
        currentStep++;

        if (currentStep < steps.length) {
            document.getElementById("instruction").innerText =
                "Step " + (currentStep + 1) + ": " + steps[currentStep].text;
        } else {
            document.getElementById("instruction").innerText =
                "Registration completed";
            document.getElementById("status").innerText =
                "All face scanning steps completed";

            document.getElementById("captureBtn").disabled = true;

            loadUsers();
        }

        return true;
    }

    document.getElementById("status").innerText =
        "Error: " + result.message;

    return false;
}
function updateProgress(stepName) {
    const item = document.getElementById(stepName + "Status");
    item.innerText = stepName.charAt(0).toUpperCase() + stepName.slice(1) + ": ✅ Captured";
}

function resetRegistration() {
    document.getElementById("autoScanBtn").disabled = false;

    currentStep = 0;

    document.getElementById("instruction").innerText =
        "Step 1: Please look straight at the camera";

    document.getElementById("status").innerText = "";

    document.getElementById("captureBtn").disabled = false;

    document.getElementById("frontStatus").innerText =
        "Front: ⬜ Not captured";

    document.getElementById("leftStatus").innerText =
        "Left: ⬜ Not captured";

    document.getElementById("rightStatus").innerText =
        "Right: ⬜ Not captured";

    document.getElementById("userId").value = "";
    document.getElementById("fullName").value = "";
    document.getElementById("role").value = "student";
}

function sleep(ms) {
    return new Promise(resolve => setTimeout(resolve, ms));
}

function setScanningState(isScanning) {
    document.getElementById("autoScanBtn").disabled = isScanning;
    document.getElementById("captureBtn").disabled = isScanning;
}

// Load registered users
async function loadUsers() {
    const response = await fetch("/users");
    const result = await response.json();

    const usersList = document.getElementById("usersList");
    usersList.innerHTML = "";

    if (result.total === 0) {
        usersList.innerHTML = "<li>No registered users</li>";
        return;
    }

    result.users.forEach(user => {
        const item = document.createElement("li");
        item.innerHTML = `
        ${user.user_id} - ${user.full_name} (${user.role})
        <button onclick="deleteUser('${user.user_id}')">Delete</button>
    `;
    usersList.appendChild(item);
    });
}

// Delete user function
async function deleteUser(userId) {
    const confirmed = confirm("Delete user " + userId + "?");

    if (!confirmed) {
        return;
    }

    const response = await fetch("/users/" + userId, {
        method: "DELETE"
    });

    const result = await response.json();
    alert(result.message);

    loadUsers();
}

//function check user id exists
async function checkUserExists(userId) {
    const response = await fetch("/users/" + userId + "/exists");
    const result = await response.json();
    return result.exists;
}

// async function validateUserBeforeScan() {
//     const userId = document.getElementById("userId").value.trim();

//     if (!userId) {
//         document.getElementById("status").innerText =
//             "Please enter User ID before scanning.";
//         return false;
//     }

//     if (overwriteConfirmedUserId === userId) {
//         return true;
//     }

//     const exists = await checkUserExists(userId);

//     if (exists) {
//         const confirmed = confirm(
//             "User ID already exists. Do you want to overwrite this user's data?"
//         );

//         if (!confirmed) {
//             return false;
//         }

//         overwriteConfirmedUserId = userId;
//     }

//     return true;
// }
async function validateUserBeforeScan() {
    const userId = document.getElementById("userId").value.trim();

    if (!userId) {
        document.getElementById("status").innerText =
            "Please enter User ID before scanning.";
        return false;
    }

    if (overwriteConfirmedUserId === userId) {
        return true;
    }

    const statusResponse = await fetch("/users/" + userId + "/embedding-status");
    const statusResult = await statusResponse.json();

    const isFullyRegistered =
        statusResult.main_embedding_exists &&
        statusResult.pose_embeddings.front &&
        statusResult.pose_embeddings.left &&
        statusResult.pose_embeddings.right;

    if (isFullyRegistered) {
        const confirmed = confirm(
            "User ID already exists and has completed face registration. Do you want to overwrite this user's data?"
        );

        if (!confirmed) {
            return false;
        }

        overwriteConfirmedUserId = userId;
    }

    return true;
}