const elements = {
  refresh: document.querySelector("#refreshButton"),
  status: document.querySelector("#status"),
  classSelect: document.querySelector("#classSelect"),
  classCode: document.querySelector("#selectedClassCode"),
  className: document.querySelector("#selectedClassName"),
  rosterBody: document.querySelector("#rosterBody"),
  availableBody: document.querySelector("#availableBody"),
  createForm: document.querySelector("#createClassForm"),
  newClassId: document.querySelector("#newClassId"),
  newClassCode: document.querySelector("#newClassCode"),
  newClassName: document.querySelector("#newClassName"),
};

const state = { classes: [], students: [], roster: [] };

async function request(url, options = {}) {
  const response = await fetch(url, options);
  if (!response.ok) {
    let message = `Request failed (${response.status})`;
    try {
      const body = await response.json();
      message = body.detail || message;
    } catch (_) { /* Keep the HTTP fallback message. */ }
    throw new Error(message);
  }
  return response.status === 204 ? null : response.json();
}

function setStatus(message = "", kind = "") {
  elements.status.textContent = message;
  elements.status.className = `status ${kind}`.trim();
}

function selectedClass() {
  return state.classes.find((item) => item.class_id === elements.classSelect.value);
}

function renderClassSelection(preferredId) {
  const selectedId = state.classes.some((item) => item.class_id === preferredId)
    ? preferredId
    : (state.classes[0]?.class_id || "");
  elements.classSelect.replaceChildren(...state.classes.map((item) => {
    const option = document.createElement("option");
    option.value = item.class_id;
    option.textContent = `${item.class_code} — ${item.class_name}`;
    option.selected = item.class_id === selectedId;
    return option;
  }));
  elements.classSelect.disabled = state.classes.length === 0;
  renderSelectedClass();
}

function renderSelectedClass() {
  const item = selectedClass();
  elements.classCode.textContent = item?.class_code || "—";
  elements.className.textContent = item?.class_name || "—";
}

function actionButton(label, className, handler) {
  const button = document.createElement("button");
  button.type = "button";
  button.textContent = label;
  button.className = className;
  button.addEventListener("click", async () => {
    button.disabled = true;
    try { await handler(); } finally { button.disabled = false; }
  });
  return button;
}

function renderTable(body, rows, actionLabel, actionClass, handler) {
  body.replaceChildren();
  if (!rows.length) {
    const row = body.insertRow();
    const cell = row.insertCell();
    cell.colSpan = 4;
    cell.className = "empty";
    cell.textContent = selectedClass() ? "No students to show." : "Select or create a class.";
    return;
  }
  rows.forEach((student) => {
    const row = body.insertRow();
    [student.student_id, student.full_name, student.role].forEach((value) => {
      row.insertCell().textContent = value;
    });
    row.insertCell().append(actionButton(
      actionLabel,
      actionClass,
      () => handler(student.student_id),
    ));
  });
}

function renderStudents() {
  const rosterIds = new Set(state.roster.map((item) => item.student_id));
  const available = state.students.filter((item) => !rosterIds.has(item.student_id));
  renderTable(elements.rosterBody, state.roster, "Remove", "danger", removeStudent);
  renderTable(elements.availableBody, available, "Add", "compact", addStudent);
}

async function loadRoster() {
  const item = selectedClass();
  state.roster = item
    ? await request(`/attendance/classes/${encodeURIComponent(item.class_id)}/roster`)
    : [];
  renderSelectedClass();
  renderStudents();
}

async function refreshAll({ preserveSelection = true, successMessage = "" } = {}) {
  const previousId = preserveSelection ? elements.classSelect.value : "";
  setStatus("Refreshing…");
  elements.refresh.disabled = true;
  try {
    [state.classes, state.students] = await Promise.all([
      request("/attendance/classes"),
      request("/attendance/students"),
    ]);
    renderClassSelection(previousId);
    await loadRoster();
    setStatus(successMessage, successMessage ? "success" : "");
  } catch (error) {
    setStatus(error.message, "error");
  } finally {
    elements.refresh.disabled = false;
  }
}

async function addStudent(studentId) {
  const item = selectedClass();
  if (!item) return;
  try {
    await request(
      `/attendance/classes/${encodeURIComponent(item.class_id)}/students/${encodeURIComponent(studentId)}`,
      { method: "POST" },
    );
    state.students = await request("/attendance/students");
    await loadRoster();
    setStatus(`Added ${studentId} to ${item.class_code}.`, "success");
  } catch (error) {
    setStatus(error.message, "error");
  }
}

async function removeStudent(studentId) {
  const item = selectedClass();
  if (!item) return;
  try {
    await request(
      `/attendance/classes/${encodeURIComponent(item.class_id)}/students/${encodeURIComponent(studentId)}`,
      { method: "DELETE" },
    );
    state.students = await request("/attendance/students");
    await loadRoster();
    setStatus(`Removed ${studentId} from ${item.class_code}.`, "success");
  } catch (error) {
    setStatus(error.message, "error");
  }
}

elements.refresh.addEventListener("click", () => refreshAll());
elements.classSelect.addEventListener("change", async () => {
  setStatus("");
  try { await loadRoster(); } catch (error) { setStatus(error.message, "error"); }
});
elements.createForm.addEventListener("submit", async (event) => {
  event.preventDefault();
  const payload = {
    class_id: elements.newClassId.value.trim(),
    class_code: elements.newClassCode.value.trim(),
    class_name: elements.newClassName.value.trim(),
  };
  try {
    const created = await request("/attendance/classes", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(payload),
    });
    elements.createForm.reset();
    await refreshAll({ preserveSelection: false });
    elements.classSelect.value = created.class_id;
    await loadRoster();
    setStatus(`Created class ${created.class_code}.`, "success");
  } catch (error) {
    setStatus(error.message, "error");
  }
});

refreshAll();
