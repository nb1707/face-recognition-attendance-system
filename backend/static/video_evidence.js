function getSessionId() {
    const params = new URLSearchParams(window.location.search);
    return params.get("session_id");
}

async function loadEvidenceFrames() {

    const sessionId = getSessionId();

    document.getElementById("sessionTitle").innerText =
        "Session: " + sessionId;

    const response =
        await fetch("/video-evidence/" + sessionId);

    const result =
        await response.json();

    const gallery =
        document.getElementById("evidenceGallery");

    gallery.innerHTML = "";

    result.frames.forEach(frame => {

        const card =
            document.createElement("div");

        card.className = "evidence-card";

        card.innerHTML = `
            <h4>
                ${frame.frame_time_seconds.toFixed(1)} sec
            </h4>

            <p>
                Detected Faces:
                ${frame.detected_count}
            </p>

            <img
                src="${frame.public_url}"
                width="500"
            >
        `;

        gallery.appendChild(card);
    });
}

window.onload = loadEvidenceFrames;