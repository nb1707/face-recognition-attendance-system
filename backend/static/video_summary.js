function getSessionIdFromUrl() {
    const params = new URLSearchParams(window.location.search);
    return params.get("session_id");
}

async function loadSummary(sessionId) {

    // const response =
    //     await fetch("/video-summary/" + sessionId);
    // const sessionId = getSessionIdFromUrl();

    if (!sessionId) {
        alert("Session ID not found.");
        return;
    }

    const response =
        await fetch("/video-summary/" + sessionId);

    const result =
        await response.json();

    document.getElementById("sessionTitle").innerText =
        "Session: " + sessionId;

    const tbody =
        document.getElementById("summaryTableBody");

    tbody.innerHTML = "";

    let presentCount = 0;
    let absentCount = 0;

    result.summaries.forEach(item => {

        if (item.suggested_status === "present") {
            presentCount++;
        } else {
            absentCount++;
        }

        const ratio =
            (item.presence_ratio * 100).toFixed(1);

        const row =
            document.createElement("tr");

        row.innerHTML = `
            <td>${item.user_id}</td>
            <td>${item.full_name}</td>
            <td>${item.detected_windows}</td>
            <td>${item.total_windows}</td>
            <td>${ratio}%</td>
            <td>${item.suggested_status}</td>
        `;

        tbody.appendChild(row);
    });

    document.getElementById("totalStudents").innerText =
        result.total;

    document.getElementById("presentStudents").innerText =
        presentCount;

    document.getElementById("absentStudents").innerText =
        absentCount;
}

function viewEvidence() {

    // const sessionId =
    //     getSessionIdFromUrl();

    const sessionId =
        document.getElementById("sessionSelect").value;
    window.location.href =
        "/video-evidence-page?session_id=" +
        sessionId;
}

async function loadSessions(){

    const response =
        await fetch("/video-sessions");

    const result =
        await response.json();

    const select =
        document.getElementById("sessionSelect");

    select.innerHTML = "";

    result.sessions.forEach(session=>{

        const option =
            document.createElement("option");

        option.value = session.session_id;

        option.text =
            `${session.session_id} | ${session.class_name} | ${session.camera_id}`;

        select.appendChild(option);

    });

}
async function loadSelectedSession(){

    const sessionId =
        document.getElementById("sessionSelect").value;

    loadSummary(sessionId);

}
window.onload = async ()=>{

    await loadSessions();

    const first =
        document.getElementById("sessionSelect").value;

    loadSummary(first);

}