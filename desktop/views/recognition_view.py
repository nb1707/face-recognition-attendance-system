"""Realtime AdaFace recognition and session/test controls."""

from __future__ import annotations

import tkinter as tk
from collections.abc import Callable, Mapping
from tkinter import ttk

from ..attendance.state import (
    attendance_counts,
    can_finalize_records,
    decision_label,
    suggestion_label,
)
from ..widgets import VideoPanel


class VerticalScrolledFrame(ttk.Frame):
    """A vertically scrollable ttk content frame with local wheel handling."""

    def __init__(self, parent: tk.Misc) -> None:
        super().__init__(parent)
        self.columnconfigure(0, weight=1)
        self.rowconfigure(0, weight=1)

        background = ttk.Style(self).lookup("TFrame", "background")
        self.canvas = tk.Canvas(
            self,
            borderwidth=0,
            highlightthickness=0,
            background=background or None,
        )
        self.scrollbar = ttk.Scrollbar(
            self,
            orient="vertical",
            command=self.canvas.yview,
        )
        self.canvas.configure(yscrollcommand=self.scrollbar.set)
        self.canvas.grid(row=0, column=0, sticky="nsew")
        self.scrollbar.grid(row=0, column=1, sticky="ns")

        self.content = ttk.Frame(self.canvas)
        self._content_window = self.canvas.create_window(
            (0, 0),
            window=self.content,
            anchor="nw",
        )
        self._wheel_bind_id: str | None = None

        self.content.bind(
            "<Configure>",
            self._on_content_configure,
            add="+",
        )
        self.canvas.bind(
            "<Configure>",
            self._on_canvas_configure,
            add="+",
        )
        for widget in (self, self.canvas, self.scrollbar, self.content):
            widget.bind(
                "<Enter>",
                self._schedule_wheel_activation,
                add="+",
            )
            widget.bind(
                "<Leave>",
                self._schedule_wheel_deactivation,
                add="+",
            )

    def _on_content_configure(self, _event: tk.Event | None = None) -> None:
        self.canvas.configure(scrollregion=self.canvas.bbox("all"))

    def _on_canvas_configure(self, event: tk.Event) -> None:
        self.canvas.itemconfigure(self._content_window, width=event.width)

    def _schedule_wheel_activation(self, _event: tk.Event) -> None:
        self.after_idle(self._update_wheel_binding)

    def _schedule_wheel_deactivation(self, _event: tk.Event) -> None:
        self.after_idle(self._update_wheel_binding)

    def _update_wheel_binding(self) -> None:
        pointer_x, pointer_y = self.winfo_pointerxy()
        pointer_widget = self.winfo_containing(pointer_x, pointer_y)
        pointer_inside = False
        while pointer_widget is not None:
            if pointer_widget is self:
                pointer_inside = True
                break
            pointer_widget = pointer_widget.master

        top_level = self.winfo_toplevel()
        if pointer_inside and self._wheel_bind_id is None:
            self._wheel_bind_id = top_level.bind(
                "<MouseWheel>",
                self._on_mousewheel,
                add="+",
            )
        elif not pointer_inside and self._wheel_bind_id is not None:
            top_level.unbind("<MouseWheel>", self._wheel_bind_id)
            self._wheel_bind_id = None

    def _on_mousewheel(self, event: tk.Event) -> str:
        delta = int(getattr(event, "delta", 0))
        if delta:
            units = -int(delta / 120)
            if units == 0:
                units = -1 if delta > 0 else 1
            self.canvas.yview_scroll(units, "units")
        return "break"

    def destroy(self) -> None:
        if self._wheel_bind_id is not None:
            self.winfo_toplevel().unbind("<MouseWheel>", self._wheel_bind_id)
            self._wheel_bind_id = None
        super().destroy()


def _display_number(value: object, precision: int) -> str:
    try:
        return f"{float(value):.{precision}f}"
    except (TypeError, ValueError):
        return "--"


def _display_pixels(value: object) -> str:
    try:
        return f"{float(value):g}"
    except (TypeError, ValueError):
        return "--"


def format_quality_diagnostics(diagnostics: Mapping | None) -> str:
    """Format quality data already produced by the recognition coordinator."""
    if not isinstance(diagnostics, Mapping):
        return "Quality Diagnostics\nMeasurements unavailable."

    measurements = diagnostics.get("quality_measurements")
    decision = diagnostics.get("quality_decision")
    if not isinstance(measurements, Mapping):
        return "Quality Diagnostics\nMeasurements unavailable."

    if not measurements.get("valid", False):
        reason = measurements.get("reason") or "invalid_observation"
        return (
            "Quality Diagnostics\n"
            "Measurement status: invalid\n"
            f"Reason: {reason}"
        )

    decision = decision if isinstance(decision, Mapping) else {}
    reasons = decision.get("reasons") or ()
    if isinstance(reasons, str):
        reasons = (reasons,)
    rejection_text = ", ".join(str(reason) for reason in reasons) or "None"
    passed = "Yes" if decision.get("passed", False) else "No"

    return "\n".join((
        "Quality Diagnostics",
        "Face size: "
        f"{_display_pixels(measurements.get('width'))} × "
        f"{_display_pixels(measurements.get('height'))} px",
        f"Area ratio: {_display_number(measurements.get('area_ratio'), 4)}",
        "Detector confidence: "
        f"{_display_number(measurements.get('detector_confidence'), 2)}",
        f"Sharpness: {_display_number(measurements.get('sharpness'), 1)}",
        f"Brightness: {_display_number(measurements.get('brightness'), 1)}",
        f"Nose offset: {_display_number(measurements.get('nose_offset'), 2)}",
        f"Roll: {_display_number(measurements.get('roll_degrees'), 1)} degrees",
        f"Quality score: {_display_number(decision.get('quality_score'), 2)}",
        f"Quality passed: {passed}",
        f"Rejection reasons: {rejection_text}",
    ))


def format_matcher_diagnostics(diagnostics: Mapping | None) -> str:
    """Format matcher metadata already preserved by the recognition worker."""
    if not isinstance(diagnostics, Mapping):
        return "Matcher Diagnostics\nCandidates unavailable."

    matcher = diagnostics.get("matcher_diagnostics")
    if not isinstance(matcher, Mapping):
        return "Matcher Diagnostics\nCandidates unavailable."

    def candidate_lines(label: str, candidate: object) -> tuple[str, str]:
        candidate = candidate if isinstance(candidate, Mapping) else {}
        user_id = candidate.get("user_id") or "--"
        full_name = candidate.get("full_name") or "--"
        prototype_id = candidate.get("winning_prototype_id") or "--"
        pose = candidate.get("winning_pose") or "--"
        return (
            f"{label}: {user_id} — {full_name} — "
            f"{_display_number(candidate.get('score'), 4)} — {prototype_id}",
            f"{label} pose: {pose}",
        )

    top1_lines = candidate_lines("Top1", matcher.get("top1"))
    top2_lines = candidate_lines("Top2", matcher.get("top2"))
    similarity_gate = (
        "PASS" if matcher.get("similarity_gate_passed", False) else "FAIL"
    )
    margin_gate = (
        "PASS" if matcher.get("margin_gate_passed", False) else "FAIL"
    )
    gate_a = "PASS" if matcher.get("gate_a_passed", False) else "FAIL"
    gate_a_reason = matcher.get("gate_a_reason") or "--"

    return "\n".join((
        "Matcher Diagnostics",
        *top1_lines,
        *top2_lines,
        f"Margin: {_display_number(matcher.get('margin'), 4)}",
        "Similarity threshold T: "
        f"{_display_number(matcher.get('similarity_threshold'), 2)}",
        "Margin threshold M: "
        f"{_display_number(matcher.get('margin_threshold'), 2)}",
        f"Similarity Gate: {similarity_gate}",
        f"Margin Gate: {margin_gate}",
        f"Gate A: {gate_a} — {gate_a_reason}",
    ))


class RecognitionView(ttk.Frame):
    """Hikvision recognition view with explicit session context."""

    CAMERA_TEST_LABEL = "Camera Test"
    REAL_CLASS_LABEL = "Real Class"
    UNKNOWN_EXPECTED_LABEL = "Unknown / Unregistered Person"

    def __init__(self, parent: tk.Misc) -> None:
        super().__init__(parent, padding=16)
        self.columnconfigure(0, weight=1)
        self.rowconfigure(2, weight=1)

        self.class_code = tk.StringVar(master=self)
        self.subject_code = tk.StringVar(master=self)
        self.session_type = tk.StringVar(
            master=self, value=self.CAMERA_TEST_LABEL
        )
        self.notes = tk.StringVar(master=self)
        self.operator_name = tk.StringVar(master=self, value="Teacher")
        self.selected_class = tk.StringVar(master=self)
        self.camera_source = tk.StringVar(master=self, value="Webcam")
        self.attendance_state = tk.StringVar(master=self, value="Session: IDLE")
        self.attendance_counts = tk.StringVar(
            master=self, value="Roster: 0 | Evidence: 0 | Not yet observed: 0"
        )
        self.review_decision = tk.StringVar(master=self)
        self.review_note = tk.StringVar(master=self)
        self._class_ids_by_label: dict[str, str] = {}
        self._attendance_records: dict[str, dict] = {}
        self._attendance_phase = "idle"
        self._attendance_busy = False
        self._backend_available = False
        self._roster_count = 0
        self._missing_gallery_student_ids: set[str] = set()

        self.distance_m = tk.StringVar(master=self)
        self.expected_user_display = tk.StringVar(master=self)
        self.trial_notes = tk.StringVar(master=self)
        self._expected_user_ids_by_label: dict[str, str] = {}

        self.active_model = tk.StringVar(
            master=self, value="Active Model: Not Loaded"
        )
        self.identity = tk.StringVar(master=self, value="Identity: --")
        self.user_id = tk.StringVar(master=self, value="User ID: --")
        self.role = tk.StringVar(master=self, value="Role: --")
        self.similarity = tk.StringVar(
            master=self, value="Similarity / Confidence: --"
        )
        self.threshold = tk.StringVar(master=self, value="Threshold: 0.40")
        self.recognition_status = tk.StringVar(
            master=self, value="Recognition Status: Idle"
        )
        self.quality_diagnostics = tk.StringVar(
            master=self,
            value=format_quality_diagnostics(None),
        )
        self.matcher_diagnostics = tk.StringVar(
            master=self,
            value=format_matcher_diagnostics(None),
        )
        self.processing_time = tk.StringVar(
            master=self, value="Processing Time: -- ms"
        )
        self.current_session = tk.StringVar(
            master=self, value="Session: Not started"
        )
        self.status = tk.StringVar(
            master=self,
            value="Enter the required session information before recognition.",
        )

        self._session_widgets: list[tuple[tk.Widget, str]] = []
        self._session_active = False
        self._session_pending = False
        self._camera_active = False
        self._camera_stopping = False
        self._trial_available = False
        self._trial_count = 0
        self._active_session_type: str | None = None

        ttk.Label(
            self, text="Recognize Face", style="PageTitle.TLabel"
        ).grid(row=0, column=0, sticky="w", pady=(0, 8))

        self._build_session_form()
        self._build_recognition_body()
        self._build_status_area()
        self.set_expected_users(())

        self.session_type.trace_add(
            "write", lambda *_args: self._update_test_visibility()
        )
        self.distance_m.trace_add(
            "write", lambda *_args: self._refresh_session_summary()
        )
        self.review_decision.trace_add(
            "write", lambda *_args: self._refresh_action_states()
        )
        self._update_test_visibility()

    def _build_session_form(self) -> None:
        session = ttk.LabelFrame(
            self, text="Attendance Session", padding=(12, 8)
        )
        session.grid(row=1, column=0, sticky="ew", pady=(0, 8))
        for column in (1, 3, 5):
            session.columnconfigure(column, weight=1)

        ttk.Label(session, text="Class *").grid(row=0, column=0, sticky="w")
        self.class_combo = ttk.Combobox(
            session,
            textvariable=self.selected_class,
            state="disabled",
            width=32,
        )
        self.class_combo.grid(row=0, column=1, sticky="ew", padx=(4, 10))
        self._session_widgets.append((self.class_combo, "readonly"))

        ttk.Label(session, text="Operator *").grid(row=0, column=2, sticky="w")
        self.operator_entry = ttk.Entry(session, textvariable=self.operator_name)
        self.operator_entry.grid(row=0, column=3, sticky="ew", padx=(4, 10))
        self._session_widgets.append((self.operator_entry, "normal"))

        ttk.Label(session, text="Camera *").grid(row=0, column=4, sticky="w")
        self.camera_combo = ttk.Combobox(
            session,
            textvariable=self.camera_source,
            values=("Webcam", "Hikvision"),
            state="readonly",
            width=14,
        )
        self.camera_combo.grid(row=0, column=5, sticky="ew", padx=(4, 0))
        self._session_widgets.append((self.camera_combo, "readonly"))

        ttk.Label(
            session, textvariable=self.attendance_state, justify="left"
        ).grid(row=1, column=0, columnspan=3, sticky="w", pady=(6, 0))
        ttk.Label(
            session, textvariable=self.attendance_counts, justify="left"
        ).grid(row=1, column=3, columnspan=3, sticky="e", pady=(6, 0))

        # Retained as hidden compatibility fields for the existing diagnostic
        # helpers; the final attendance workflow does not use legacy sessions.
        self.test_fields = ttk.Frame(session)
        self.expected_user_combo = ttk.Combobox(self.test_fields)
        self.operator_name.trace_add(
            "write", lambda *_args: self._refresh_action_states()
        )

    def _add_session_field(
        self,
        parent: ttk.Frame,
        label: str,
        variable: tk.StringVar,
        row: int,
        column: int,
    ) -> None:
        ttk.Label(parent, text=label).grid(
            row=row, column=column, sticky="w", padx=(8 if column else 0, 4)
        )
        entry = ttk.Entry(parent, textvariable=variable, width=16)
        entry.grid(row=row, column=column + 1, sticky="ew")
        self._session_widgets.append((entry, "normal"))

    def _build_recognition_body(self) -> None:
        body = ttk.Frame(self)
        body.grid(row=2, column=0, sticky="nsew")
        body.columnconfigure(0, weight=3)
        body.columnconfigure(1, weight=1, minsize=300)
        body.rowconfigure(0, weight=1)
        body.rowconfigure(1, weight=1)

        preview = ttk.LabelFrame(body, text="Camera Preview", padding=10)
        preview.grid(row=0, column=0, sticky="nsew", padx=(0, 10))
        preview.columnconfigure(0, weight=1)
        preview.rowconfigure(0, weight=1)
        self.video_panel = VideoPanel(
            preview,
            width=640,
            height=310,
            placeholder="Recognition camera preview",
        )
        self.video_panel.grid(row=0, column=0, sticky="nsew")

        details = ttk.LabelFrame(
            body, text="Recognition Details", padding=12
        )
        details.grid(row=0, column=1, sticky="nsew")
        details.columnconfigure(0, weight=1)
        details.rowconfigure(0, weight=1)

        self.details_scroller = VerticalScrolledFrame(details)
        self.details_scroller.grid(row=0, column=0, sticky="nsew")
        detail_content = self.details_scroller.content
        detail_content.columnconfigure(0, weight=1)

        self.start_button = ttk.Button(
            detail_content, text="Start Attendance Session"
        )
        self.start_button.grid(row=0, column=0, sticky="ew", pady=(0, 5))
        self.stop_button = ttk.Button(
            detail_content, text="End Attendance Session", state="disabled"
        )
        self.stop_button.grid(row=1, column=0, sticky="ew", pady=(0, 5))
        self.record_trial_button = ttk.Button(
            detail_content, text="Record Trial", state="disabled"
        )
        self.record_trial_button.grid(
            row=2, column=0, sticky="ew", pady=(0, 8)
        )
        self.record_trial_button.grid_remove()

        ttk.Label(
            detail_content,
            textvariable=self.current_session,
            wraplength=280,
            justify="left",
        ).grid(row=3, column=0, sticky="w", pady=(0, 6))

        detail_variables = (
            self.active_model,
            self.identity,
            self.user_id,
            self.role,
            self.similarity,
            self.threshold,
            self.recognition_status,
            self.quality_diagnostics,
            self.matcher_diagnostics,
            self.processing_time,
        )
        for row_index, variable in enumerate(detail_variables, start=4):
            ttk.Label(
                detail_content,
                textvariable=variable,
                wraplength=280,
                justify="left",
            ).grid(row=row_index, column=0, sticky="w", pady=2)

        attendance = ttk.LabelFrame(
            body, text="Roster, Evidence, and Teacher Review", padding=8
        )
        attendance.grid(
            row=1, column=0, columnspan=2, sticky="nsew", pady=(8, 0)
        )
        attendance.columnconfigure(0, weight=3)
        attendance.columnconfigure(1, weight=1)
        attendance.rowconfigure(0, weight=1)

        columns = ("student", "name", "suggestion", "evidence", "decision", "note")
        self.attendance_table = ttk.Treeview(
            attendance, columns=columns, show="headings", height=7, selectmode="browse"
        )
        headings = {
            "student": "Student ID",
            "name": "Name",
            "suggestion": "Suggestion",
            "evidence": "Evidence",
            "decision": "Final Decision",
            "note": "Note",
        }
        widths = {"student": 105, "name": 155, "suggestion": 120, "evidence": 100,
                  "decision": 105, "note": 180}
        for column in columns:
            self.attendance_table.heading(column, text=headings[column])
            self.attendance_table.column(column, width=widths[column], anchor="w")
        self.attendance_table.grid(row=0, column=0, sticky="nsew", padx=(0, 8))
        self.attendance_table.bind("<<TreeviewSelect>>", self._on_review_row_selected)

        self.review_scroller = VerticalScrolledFrame(attendance)
        self.review_scroller.grid(row=0, column=1, sticky="nsew")
        # Keep this panel compact on laptop screens; its contents scroll
        # instead of increasing the requested height of the attendance row.
        self.review_scroller.canvas.configure(width=240, height=160)
        controls = self.review_scroller.content
        controls.configure(padding=(0, 0, 4, 8))
        controls.columnconfigure(0, weight=1)
        ttk.Label(controls, text="Selected student decision").grid(
            row=0, column=0, sticky="w"
        )
        self.review_decision_combo = ttk.Combobox(
            controls,
            textvariable=self.review_decision,
            values=("Present", "Absent", "Excused"),
            state="disabled",
        )
        self.review_decision_combo.grid(row=1, column=0, sticky="ew", pady=(2, 5))
        ttk.Label(controls, text="Optional note").grid(row=2, column=0, sticky="w")
        self.review_note_entry = ttk.Entry(controls, textvariable=self.review_note)
        self.review_note_entry.grid(row=3, column=0, sticky="ew", pady=(2, 5))
        self.save_review_button = ttk.Button(
            controls, text="Save Review", state="disabled"
        )
        self.save_review_button.grid(row=4, column=0, sticky="ew", pady=(0, 5))
        self.finalize_button = ttk.Button(
            controls, text="Finalize Session", state="disabled"
        )
        self.finalize_button.grid(row=5, column=0, sticky="ew", pady=(0, 8))
        ttk.Label(controls, text="Recent Recognition Events").grid(
            row=6, column=0, sticky="w"
        )
        self.recent_events = tk.Listbox(controls, height=4)
        self.recent_events.grid(row=7, column=0, sticky="nsew", pady=(2, 8))

    def _build_status_area(self) -> None:
        status_area = ttk.LabelFrame(
            self, text="Status", padding=(12, 7)
        )
        status_area.grid(row=3, column=0, sticky="ew", pady=(8, 0))
        status_area.columnconfigure(0, weight=1)
        ttk.Label(
            status_area,
            textvariable=self.status,
            wraplength=900,
            justify="left",
        ).grid(row=0, column=0, sticky="w")

    def set_recognition_commands(
        self,
        *,
        start_command: Callable[[], None],
        stop_command: Callable[[], None],
        record_trial_command: Callable[[], None],
    ) -> None:
        self.start_button.configure(command=start_command)
        self.stop_button.configure(command=stop_command)
        self.record_trial_button.configure(command=record_trial_command)

    def set_attendance_commands(
        self,
        *,
        class_selected_command: Callable[[], None],
        review_command: Callable[[], None],
        finalize_command: Callable[[], None],
    ) -> None:
        self.class_combo.bind(
            "<<ComboboxSelected>>", lambda _event: class_selected_command()
        )
        self.save_review_button.configure(command=review_command)
        self.finalize_button.configure(command=finalize_command)

    def set_classes(self, classes: list[dict]) -> None:
        previous_id = self.selected_class_id()
        options = []
        for item in classes:
            class_id = str(item.get("class_id") or "").strip()
            if not class_id:
                continue
            label = f"{item.get('class_code') or class_id} — {item.get('class_name') or class_id}"
            options.append((label, class_id))
        self._class_ids_by_label = dict(options)
        self.class_combo.configure(values=tuple(label for label, _ in options))
        selected = next(
            (label for label, class_id in options if class_id == previous_id), ""
        )
        self.selected_class.set(selected)
        self._backend_available = True
        self._roster_count = 0
        self._attendance_records.clear()
        self._clear_attendance_table()
        self._refresh_action_states()

    def set_backend_unavailable(self, message: str) -> None:
        self._backend_available = False
        self._roster_count = 0
        self.status.set(message)
        self._refresh_action_states()

    def selected_class_id(self) -> str:
        return self._class_ids_by_label.get(self.selected_class.get(), "")

    @property
    def backend_available(self) -> bool:
        return self._backend_available

    @property
    def attendance_phase(self) -> str:
        return self._attendance_phase

    def attendance_start_values(self) -> dict[str, str]:
        return {
            "class_id": self.selected_class_id(),
            "operator_name": self.operator_name.get().strip(),
            "camera_source": self.camera_source.get().strip(),
        }

    def set_roster(self, roster: list[dict]) -> None:
        self._roster_count = len(roster)
        self._attendance_records = {
            str(item.get("student_id")): {
                "student_id": str(item.get("student_id")),
                "snapshot_name": str(item.get("full_name") or ""),
                "suggestion_status": None,
                "final_status": None,
                "event_exists": False,
                "review_note": None,
            }
            for item in roster
            if item.get("student_id")
        }
        self._render_attendance_records()
        self._refresh_action_states()

    def set_attendance_session(self, phase: str, session: Mapping | None = None) -> None:
        self._attendance_phase = phase
        session = session if isinstance(session, Mapping) else {}
        session_id = str(session.get("session_id") or "")
        class_id = str(session.get("class_id") or self.selected_class_id())
        started = str(session.get("started_at_utc") or "")
        lines = [f"Session: {phase.upper()}"]
        if class_id:
            lines.append(f"Class: {class_id}")
        if session_id:
            lines.append(f"ID: {session_id}")
        if started:
            lines.append(f"Started: {started}")
        lines.append(f"Camera: {self.camera_source.get()}")
        self.attendance_state.set(" | ".join(lines))
        self._refresh_action_states()

    def set_attendance_busy(self, busy: bool, message: str | None = None) -> None:
        self._attendance_busy = busy
        if message:
            self.status.set(message)
        self._refresh_action_states()

    def set_attendance_records(self, records: list[dict], *, phase: str | None = None) -> None:
        self._attendance_records = {
            str(record.get("student_id")): dict(record)
            for record in records
            if record.get("student_id")
        }
        self._roster_count = len(self._attendance_records)
        if phase is not None:
            self._attendance_phase = phase
        self._render_attendance_records()
        self._refresh_action_states()

    def set_missing_gallery_students(self, student_ids: tuple[str, ...]) -> None:
        self._missing_gallery_student_ids = set(student_ids)
        self._render_attendance_records()
        if student_ids:
            self.status.set(
                "No enrolled face/gallery entry: " + ", ".join(student_ids)
            )

    def selected_review_values(self) -> dict[str, str] | None:
        selection = self.attendance_table.selection()
        if not selection:
            return None
        labels = {"Present": "present", "Absent": "absent", "Excused": "excused"}
        return {
            "student_id": selection[0],
            "final_status": labels.get(self.review_decision.get(), ""),
            "reason": self.review_note.get(),
            "actor": self.operator_name.get().strip(),
        }

    def _on_review_row_selected(self, _event: tk.Event | None = None) -> None:
        selection = self.attendance_table.selection()
        record = self._attendance_records.get(selection[0]) if selection else None
        reverse = {"present": "Present", "absent": "Absent", "excused": "Excused"}
        self.review_decision.set(reverse.get((record or {}).get("final_status"), ""))
        self.review_note.set(str((record or {}).get("review_note") or ""))
        self._refresh_action_states()

    def _clear_attendance_table(self) -> None:
        for item in self.attendance_table.get_children():
            self.attendance_table.delete(item)
        self.recent_events.delete(0, tk.END)
        self.attendance_counts.set("Roster: 0 | Evidence: 0 | Not yet observed: 0")

    def _render_attendance_records(self) -> None:
        selected = self.attendance_table.selection()
        selected_id = selected[0] if selected else None
        self._clear_attendance_table()
        ordered = sorted(
            self._attendance_records.values(), key=lambda row: str(row.get("student_id"))
        )
        for record in ordered:
            student_id = str(record.get("student_id"))
            if record.get("event_exists"):
                evidence = "Evidence received"
            elif student_id in self._missing_gallery_student_ids:
                evidence = "No gallery enrollment"
            else:
                evidence = "Not observed"
            self.attendance_table.insert(
                "",
                "end",
                iid=student_id,
                values=(
                    student_id,
                    record.get("snapshot_name") or record.get("full_name") or "",
                    suggestion_label(record.get("suggestion_status")),
                    evidence,
                    decision_label(record.get("final_status")),
                    record.get("review_note") or "",
                ),
            )
        if selected_id and self.attendance_table.exists(selected_id):
            self.attendance_table.selection_set(selected_id)

        counts = attendance_counts(ordered)
        summary = (
            f"Roster: {counts['roster']} | Evidence: {counts['evidence']} | "
            f"Not yet observed: {counts['not_observed']}"
        )
        if self._attendance_phase in {"ended", "finalized"}:
            summary += (
                f" | Present: {counts['present']} | Absent: {counts['absent']} | "
                f"Excused: {counts['excused']}"
            )
        self.attendance_counts.set(summary)

        events = sorted(
            (row for row in ordered if row.get("event_exists")),
            key=lambda row: str(row.get("occurred_at_utc") or ""),
            reverse=True,
        )[:5]
        for row in events:
            occurred = str(row.get("occurred_at_utc") or "")
            clock = occurred[11:19] if len(occurred) >= 19 else "--:--:--"
            self.recent_events.insert(
                tk.END, f"{clock}  {row.get('student_id')}  Confirmed evidence"
            )

    def session_form_values(self) -> dict[str, str]:
        session_type = (
            "camera_test"
            if self.session_type.get() == self.CAMERA_TEST_LABEL
            else "real_class"
        )
        return {
            "class_code": self.class_code.get(),
            "subject_code": self.subject_code.get(),
            "session_type": session_type,
            "notes": self.notes.get(),
        }

    def trial_form_values(self) -> dict[str, str]:
        return {
            "distance_m": self.distance_m.get(),
            "expected_user_id": self._expected_user_ids_by_label.get(
                self.expected_user_display.get(), ""
            ),
            "trial_notes": self.trial_notes.get(),
        }

    def set_expected_users(
        self,
        users: tuple[tuple[str, str], ...],
    ) -> None:
        """Populate user labels while retaining IDs as stored values."""
        previous_id = self._expected_user_ids_by_label.get(
            self.expected_user_display.get()
        )
        options = [(self.UNKNOWN_EXPECTED_LABEL, "__unknown__")]
        options.extend(
            (f"{user_id} — {full_name}", user_id)
            for user_id, full_name in users
        )
        self._expected_user_ids_by_label = dict(options)
        labels = tuple(label for label, _user_id in options)
        self.expected_user_combo.configure(values=labels, state="readonly")

        matching_label = next(
            (
                label
                for label, user_id in options
                if previous_id is not None and user_id == previous_id
            ),
            "",
        )
        self.expected_user_display.set(matching_label)

    def set_session_active(
        self,
        *,
        active: bool,
        session_type: str | None = None,
        trial_count: int = 0,
    ) -> None:
        self._session_active = active
        self._session_pending = False
        self._active_session_type = session_type if active else None
        self._trial_count = trial_count if active else 0
        for widget, idle_state in self._session_widgets:
            widget.configure(state="disabled" if active else idle_state)
        if not active:
            self.current_session.set("Session: Not started")
        else:
            self._refresh_session_summary(trial_count)
        self._refresh_action_states()

    def set_session_pending(self, pending: bool) -> None:
        self._session_pending = pending
        self._refresh_action_states()

    def set_trial_available(self, available: bool) -> None:
        self._trial_available = available
        self._refresh_action_states()

    def update_trial_count(self, trial_count: int) -> None:
        self._trial_count = trial_count
        self._refresh_session_summary()

    def set_camera_state(
        self,
        message: str,
        *,
        active: bool,
        stopping: bool = False,
    ) -> None:
        self.status.set(message)
        self._camera_active = active
        self._camera_stopping = stopping
        self._refresh_action_states()

    def reset_recognition_details(self) -> None:
        self.identity.set("Identity: --")
        self.user_id.set("User ID: --")
        self.role.set("Role: --")
        self.similarity.set("Similarity / Confidence: --")
        self.recognition_status.set("Recognition Status: Idle")
        self.set_quality_diagnostics(None)
        self.set_matcher_diagnostics(None)
        self.processing_time.set("Processing Time: -- ms")
        self._trial_available = False
        self._refresh_action_states()

    def set_quality_diagnostics(self, diagnostics: Mapping | None) -> None:
        self.quality_diagnostics.set(format_quality_diagnostics(diagnostics))

    def set_matcher_diagnostics(self, diagnostics: Mapping | None) -> None:
        self.matcher_diagnostics.set(format_matcher_diagnostics(diagnostics))

    def _refresh_action_states(self) -> None:
        start_disabled = not (
            self._backend_available
            and self.selected_class_id()
            and self._roster_count > 0
            and self.operator_name.get().strip()
            and self._attendance_phase == "idle"
            and not self._attendance_busy
            and not self._camera_active
            and not self._camera_stopping
        )
        self.start_button.configure(
            state="disabled" if start_disabled else "normal"
        )
        stop_enabled = self._attendance_phase == "active" and not self._attendance_busy
        self.stop_button.configure(
            state="normal"
            if stop_enabled and not self._camera_stopping
            else "disabled"
        )
        self.record_trial_button.configure(state="disabled")

        class_state = (
            "readonly"
            if self._backend_available
            and self._attendance_phase == "idle"
            and not self._attendance_busy
            else "disabled"
        )
        self.class_combo.configure(state=class_state)
        session_locked = self._attendance_phase in {"active", "ended", "finalized"}
        self.camera_combo.configure(
            state="disabled" if session_locked or self._attendance_busy else "readonly"
        )
        self.operator_entry.configure(
            state="disabled" if self._attendance_phase == "finalized" else "normal"
        )
        selection = self.attendance_table.selection()
        review_enabled = (
            self._attendance_phase == "ended"
            and bool(selection)
            and bool(self.operator_name.get().strip())
            and not self._attendance_busy
        )
        self.review_decision_combo.configure(
            state="readonly" if review_enabled else "disabled"
        )
        self.review_note_entry.configure(
            state="normal" if review_enabled else "disabled"
        )
        self.save_review_button.configure(
            state="normal"
            if review_enabled and bool(self.review_decision.get())
            else "disabled"
        )
        self.finalize_button.configure(
            state="normal"
            if self._attendance_phase == "ended"
            and can_finalize_records(self._attendance_records.values())
            and bool(self.operator_name.get().strip())
            and not self._attendance_busy
            else "disabled"
        )

    def _update_test_visibility(self) -> None:
        self.test_fields.grid_remove()
        self._refresh_action_states()

    def _refresh_session_summary(self, trial_count: int | None = None) -> None:
        if not self._session_active:
            return
        if trial_count is not None:
            self._trial_count = trial_count
        trial_text = str(self._trial_count)
        type_label = (
            "Camera Test"
            if self._active_session_type == "camera_test"
            else "Real Class"
        )
        summary = (
            f"Session\nClass: {self.class_code.get().strip()}\n"
            f"Subject: {self.subject_code.get().strip()}\nType: {type_label}"
        )
        if self._active_session_type == "camera_test":
            distance = self.distance_m.get().strip() or "--"
            summary += f"\nDistance: {distance} m\nTrial: {trial_text}"
        self.current_session.set(summary)
