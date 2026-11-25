"""Enrollment-aware open-set metrics from saved scores, without model/cloud IO."""
from collections import Counter
from recognition.identity_matcher import MatchResult
from recognition.temporal import OperatingPoint, gate_a

UNKNOWN = '__unknown__'
ENROLLED_KNOWN = 'enrolled_known'
UNENROLLED_IMPOSTOR = 'unenrolled_impostor'
EXPLICIT_UNKNOWN = 'explicit_unknown'
MATCHER_DIAGNOSTIC = 'matcher_diagnostic'
PRODUCTION_SIMULATION = 'production_simulation'
EVALUATION_MODES = (MATCHER_DIAGNOSTIC, PRODUCTION_SIMULATION)


def enrollment_category(expected_user_id, gallery_user_ids):
    if expected_user_id is None:
        return None
    if expected_user_id == UNKNOWN:
        return EXPLICIT_UNKNOWN
    return ENROLLED_KNOWN if expected_user_id in gallery_user_ids else UNENROLLED_IMPOSTOR


def category(record):
    # Legacy records carry gallery availability. Preserve original person label,
    # but derive open-set truth from enrollment rather than folder naming.
    return enrollment_category(record['expected_user_id'],
        {record['expected_user_id']} if record.get('gallery_identity_available') else set())


def view_status(record, mode=PRODUCTION_SIMULATION):
    if mode not in EVALUATION_MODES:
        raise ValueError('Unsupported evaluation mode')
    if record.get('status') != 'evaluable':
        return record.get('status', 'invalid_observation')
    if mode == PRODUCTION_SIMULATION and not record.get('quality_passed', True):
        return 'low_quality'
    return 'evaluable'


def predict(record, point, mode=PRODUCTION_SIMULATION):
    if view_status(record, mode) != 'evaluable':
        return UNKNOWN
    match = MatchResult(**record['match'])
    return match.top1_user_id if gate_a(match, point)[0] else UNKNOWN


def metrics(records, point, conditional=False, enrolled_only=False, mode=PRODUCTION_SIMULATION):
    rows = [r for r in records if not r.get('excluded_reason') and r['expected_user_id'] is not None
            and (not conditional or view_status(r, mode) == 'evaluable')
            and (not enrolled_only or category(r) != UNENROLLED_IMPOSTOR)]
    categories = [category(r) for r in rows]
    truth = [r['expected_user_id'] if c == ENROLLED_KNOWN else UNKNOWN for r, c in zip(rows, categories)]
    predictions = [predict(r, point, mode) for r in rows]
    known = categories.count(ENROLLED_KNOWN)
    impostors = len(rows)-known
    false_accept = sum(t == UNKNOWN and p != UNKNOWN for t, p in zip(truth, predictions))
    false_reject = sum(t != UNKNOWN and p == UNKNOWN for t, p in zip(truth, predictions))
    wrong_known = sum(t != UNKNOWN and p != UNKNOWN and t != p for t, p in zip(truth, predictions))
    correct_known = sum(t != UNKNOWN and t == p for t, p in zip(truth, predictions))
    true_reject = impostors-false_accept
    ratio = lambda n, d: n/d if d else None
    subsets = {}
    for c in (UNENROLLED_IMPOSTOR, EXPLICIT_UNKNOWN):
        count = categories.count(c)
        accepted = sum(label == c and p != UNKNOWN for label, p in zip(categories, predictions))
        reached = sum(label == c and view_status(r, mode) == 'evaluable' for label, r in zip(categories, rows))
        subsets[c] = dict(count=count, reaching_matcher=reached, accepted=accepted, FAR=ratio(accepted, count))
    classes = sorted(set(truth) | set(predictions))
    f1 = []
    for label in classes:
        tp = sum(t == label and p == label for t, p in zip(truth, predictions))
        fp = sum(t != label and p == label for t, p in zip(truth, predictions))
        fn = sum(t == label and p != label for t, p in zip(truth, predictions))
        f1.append(2*tp/(2*tp+fp+fn) if 2*tp+fp+fn else 0.)
    return dict(mode=mode, conditional=conditional, samples=len(rows),
        enrolled_known_count=known, impostor_count=impostors, known=known, unknown=impostors,
        correct_known=correct_known, wrong_known=wrong_known, false_reject=false_reject,
        false_accept=false_accept, true_reject=true_reject,
        false_accepts=false_accept, false_rejects=false_reject,
        impostors_reaching_matcher=sum(s['reaching_matcher'] for s in subsets.values()),
        FAR=ratio(false_accept, impostors), combined_impostor_FAR=ratio(false_accept, impostors),
        explicit_unknown_FAR=subsets[EXPLICIT_UNKNOWN]['FAR'],
        unenrolled_impostor_FAR=subsets[UNENROLLED_IMPOSTOR]['FAR'], impostor_subsets=subsets,
        FRR=ratio(false_reject, known), wrong_identity_rate=ratio(wrong_known, known),
        known_identity_accuracy=ratio(correct_known, known), accuracy=ratio(correct_known+true_reject, len(rows)),
        macro_f1=sum(f1)/len(f1) if f1 else None,
        accepted=sum(p != UNKNOWN for p in predictions), rejected=sum(p == UNKNOWN for p in predictions),
        status_counts=dict(Counter(view_status(r, mode) for r in rows)))


def sweep(records, thresholds, margins, mode=PRODUCTION_SIMULATION):
    return [dict(threshold=t, margin=m, mode=mode,
                 end_to_end=metrics(records, OperatingPoint(t, m), mode=mode),
                 recognition_conditional=metrics(records, OperatingPoint(t, m), conditional=True, mode=mode))
            for t in thresholds for m in margins]


def pareto_candidates(rows):
    representatives = {}
    for row in sorted(rows, key=lambda r: (r['threshold'], r['margin'])):
        m = row['end_to_end']
        key = (m['false_accepts'], m['wrong_known'], m['false_rejects'])
        representatives.setdefault(key, row)
    return [row for key, row in representatives.items()
            if not any(other != key and all(a <= b for a, b in zip(other, key)) for other in representatives)]
