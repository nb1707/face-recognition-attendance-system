"""Pure presentation helpers; attendance decisions remain backend-owned."""

from __future__ import annotations

from collections.abc import Iterable, Mapping


SUGGESTION_LABELS = {
    "suggested_present": "Suggested Present",
    "needs_review": "Needs Review",
    "not_observed": "Not Observed",
}

DECISION_LABELS = {
    "present": "Present",
    "absent": "Absent",
    "excused": "Excused",
}


def suggestion_label(value: object) -> str:
    return SUGGESTION_LABELS.get(str(value or ""), "--")


def decision_label(value: object) -> str:
    return DECISION_LABELS.get(str(value or ""), "Not Reviewed")


def can_finalize_records(records: Iterable[Mapping]) -> bool:
    rows = tuple(records)
    return bool(rows) and all(
        str(row.get("final_status") or "") in DECISION_LABELS for row in rows
    )


def attendance_counts(records: Iterable[Mapping]) -> dict[str, int]:
    rows = tuple(records)
    return {
        "roster": len(rows),
        "evidence": sum(bool(row.get("event_exists")) for row in rows),
        "not_observed": sum(not bool(row.get("event_exists")) for row in rows),
        "present": sum(row.get("final_status") == "present" for row in rows),
        "absent": sum(row.get("final_status") == "absent" for row in rows),
        "excused": sum(row.get("final_status") == "excused" for row in rows),
    }
