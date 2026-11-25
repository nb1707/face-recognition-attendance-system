"""Pure persisted-diagnostic Quality Gate what-if analysis.

This module intentionally imports only the Python standard library. It never
loads video, OpenCV, a detector, AdaFace, a gallery, or coordinator state.
"""
from collections import Counter
import csv
import json
import math
from pathlib import Path
import statistics


SIZE_REASONS = {'face_too_small'}
SHARPNESS_REASONS = {'blurred_face'}
VARIABLE_REASONS = SIZE_REASONS | SHARPNESS_REASONS


def _finite(value):
    return value is not None and math.isfinite(float(value))


def percentile(sorted_values, percent):
    if not sorted_values:
        return None
    index = (len(sorted_values) - 1) * percent / 100
    lower, upper = math.floor(index), math.ceil(index)
    if lower == upper:
        return float(sorted_values[lower])
    fraction = index - lower
    return float(sorted_values[lower] * (1 - fraction) + sorted_values[upper] * fraction)


def distribution(values):
    values = sorted(float(value) for value in values if _finite(value))
    if not values:
        return dict(count=0, min=None, mean=None, median=None, p25=None,
                    p75=None, p90=None, p95=None, max=None)
    return dict(count=len(values), min=values[0], mean=statistics.fmean(values),
                median=float(statistics.median(values)), p25=percentile(values, 25),
                p75=percentile(values, 75), p90=percentile(values, 90),
                p95=percentile(values, 95), max=values[-1])


def quality_sides(record):
    """Reproduce measure_quality integer crop size semantics from persisted bbox.

    Classroom detections are inside the frame. Lower coordinates are clipped at
    zero exactly like measure_quality; upper coordinates use their persisted
    in-frame integer values. The analysis validates positive finite sides.
    """
    box = record.get('bbox')
    if not isinstance(box, list) or len(box) != 4 or not all(_finite(v) for v in box):
        return None, None
    x1, y1 = max(0, int(box[0])), max(0, int(box[1]))
    x2, y2 = int(box[2]), int(box[3])
    width, height = x2 - x1, y2 - y1
    return (float(width), float(height)) if width > 0 and height > 0 else (None, None)


def available_matcher(record):
    """Normalize one persisted result, preferring production over shadow."""
    if _finite(record.get('top1_score')):
        return dict(source='production', top1_user_id=record.get('top1_user_id'),
            top1_score=float(record['top1_score']), top2_user_id=record.get('top2_user_id'),
            top2_score=float(record['top2_score']) if _finite(record.get('top2_score')) else None,
            margin=float(record['margin']) if _finite(record.get('margin')) else None)
    if record.get('shadow_match_evaluated') and _finite(record.get('shadow_top1_score')):
        return dict(source='shadow', top1_user_id=record.get('shadow_top1_user_id'),
            top1_score=float(record['shadow_top1_score']),
            top2_user_id=record.get('shadow_top2_user_id'),
            top2_score=float(record['shadow_top2_score']) if _finite(record.get('shadow_top2_score')) else None,
            margin=float(record['shadow_margin']) if _finite(record.get('shadow_margin')) else None)
    return dict(source='unavailable', top1_user_id=None, top1_score=None,
                top2_user_id=None, top2_score=None, margin=None)


def non_variable_quality_passes(record, config):
    if not record.get('valid_observation') or not record.get('alignment_succeeded'):
        return False
    required = ('detector_confidence', 'brightness', 'nose_offset', 'roll_degrees', 'area_ratio')
    if not all(_finite(record.get(name)) for name in required):
        return False
    return (float(record['detector_confidence']) >= config['min_detector'] and
            float(record['area_ratio']) >= config['min_area_ratio'] and
            config['min_brightness'] <= float(record['brightness']) <= config['max_brightness'] and
            float(record['nose_offset']) <= config['max_nose_offset'] and
            float(record['roll_degrees']) <= config['max_roll_degrees'])


def candidate_quality_passes(record, min_size, min_sharpness, config):
    if not non_variable_quality_passes(record, config) or not _finite(record.get('sharpness')):
        return False
    width, height = quality_sides(record)
    return (width is not None and min(width, height) >= min_size and
            float(record['sharpness']) >= min_sharpness)


def gate_a_would_pass(match, similarity_threshold, margin_threshold):
    return (match['source'] != 'unavailable' and _finite(match['top1_score']) and
            _finite(match['margin']) and match['top1_score'] >= similarity_threshold and
            match['margin'] >= margin_threshold)


def normalize_record(record, baseline_size, baseline_sharpness, quality_config):
    width, height = quality_sides(record)
    match = available_matcher(record)
    return dict(raw=record, quality_width=width, quality_height=height,
        min_side=min(width, height) if width is not None else None,
        structurally_valid=bool(record.get('valid_observation')),
        eligible=non_variable_quality_passes(record, quality_config),
        baseline_pass=candidate_quality_passes(record, baseline_size, baseline_sharpness, quality_config),
        match=match)


def _risk(matches, similarity_threshold, margin_threshold):
    return dict(
        low_margin_count=sum(_finite(m['margin']) and m['margin'] < margin_threshold for m in matches),
        borderline_similarity_count=sum(.35 <= m['top1_score'] < similarity_threshold for m in matches),
        high_similarity_low_margin_count=sum(m['top1_score'] >= similarity_threshold and
            _finite(m['margin']) and m['margin'] < margin_threshold for m in matches))


def analyze_candidate(records, min_size, min_sharpness, baseline_size,
                      baseline_sharpness, similarity_threshold, margin_threshold,
                      quality_config):
    passed = [r for r in records if candidate_quality_passes(
        r['raw'], min_size, min_sharpness, quality_config)]
    newly = [r for r in passed if not r['baseline_pass']]
    matches = [r['match'] for r in passed if r['match']['source'] != 'unavailable']
    new_matches = [r['match'] for r in newly if r['match']['source'] != 'unavailable']
    gate_matches = [m for m in matches if gate_a_would_pass(m, similarity_threshold, margin_threshold)]
    new_gate_matches = [m for m in new_matches if gate_a_would_pass(m, similarity_threshold, margin_threshold)]
    reason_counts = Counter(reason for r in newly for reason in r['raw'].get('quality_reasons', []))
    reason_counts['small_and_blurred'] = sum(
        {'face_too_small', 'blurred_face'} <= set(r['raw'].get('quality_reasons', [])) for r in newly)
    for reason in ('face_too_small', 'blurred_face', 'extreme_pose', 'poor_lighting',
                   'low_detector_confidence', 'invalid_landmarks', 'alignment_failure', 'other'):
        reason_counts.setdefault(reason, 0)
    risk = _risk(matches, similarity_threshold, margin_threshold)
    new_risk = _risk(new_matches, similarity_threshold, margin_threshold)
    total = len(records)
    identity_all = Counter(m['top1_user_id'] or '__missing_user_id__' for m in matches)
    identity_gate = Counter(m['top1_user_id'] or '__missing_user_id__' for m in gate_matches)
    row = dict(min_size=min_size, min_sharpness=min_sharpness,
        baseline=min_size == baseline_size and min_sharpness == baseline_sharpness,
        total_detected=total,
        structurally_valid=sum(r['structurally_valid'] for r in records),
        eligible_for_what_if=sum(r['eligible'] for r in records),
        candidate_quality_pass=len(passed),
        candidate_quality_pass_rate=len(passed) / total if total else None,
        newly_admitted=len(newly), still_rejected=total-len(passed),
        matcher_available=len(matches), matcher_unavailable=len(passed)-len(matches),
        production_matcher_count=sum(m['source'] == 'production' for m in matches),
        shadow_matcher_count=sum(m['source'] == 'shadow' for m in matches),
        similarity_pass=sum(m['top1_score'] >= similarity_threshold for m in matches),
        margin_pass=sum(_finite(m['margin']) and m['margin'] >= margin_threshold for m in matches),
        gate_a_would_pass=len(gate_matches),
        gate_a_would_pass_rate=len(gate_matches)/len(matches) if matches else None,
        newly_admitted_matcher_available=len(new_matches),
        newly_admitted_matcher_unavailable=len(newly)-len(new_matches),
        newly_admitted_gate_a_would_pass=len(new_gate_matches),
        newly_admitted_gate_a_rate=len(new_gate_matches)/len(new_matches) if new_matches else None,
        original_reason_counts=dict(sorted(reason_counts.items())),
        predicted_identity_distribution_all=dict(sorted(identity_all.items())),
        predicted_identity_distribution_gate_a=dict(sorted(identity_gate.items())),
        top1_score=distribution(m['top1_score'] for m in matches),
        margin=distribution(m['margin'] for m in matches),
        newly_admitted_top1=distribution(m['top1_score'] for m in new_matches),
        newly_admitted_margin=distribution(m['margin'] for m in new_matches),
        newly_admitted_min_side=distribution(r['min_side'] for r in newly),
        newly_admitted_face_width=distribution(r['quality_width'] for r in newly),
        newly_admitted_face_height=distribution(r['quality_height'] for r in newly),
        newly_admitted_sharpness=distribution(r['raw'].get('sharpness') for r in newly),
        **risk,
        newly_admitted_low_margin_count=new_risk['low_margin_count'],
        newly_admitted_borderline_similarity_count=new_risk['borderline_similarity_count'],
        newly_admitted_high_similarity_low_margin_count=new_risk['high_similarity_low_margin_count'])
    return row


def shortlist(rows, baseline, similarity_threshold):
    candidates = [row for row in rows if not row['baseline'] and
                  row['newly_admitted_matcher_available'] and
                  row['newly_admitted_gate_a_would_pass'] and
                  row['newly_admitted_gate_a_rate'] >= .5 and
                  row['newly_admitted_top1']['median'] >= similarity_threshold]
    # If two policies produce exactly the same measured outcome, retain the
    # least-relaxed one. A more permissive duplicate adds no measured benefit.
    size_span = max(1, baseline['min_size'] - min(r['min_size'] for r in rows))
    sharp_span = max(1, baseline['min_sharpness'] - min(r['min_sharpness'] for r in rows))
    def relaxation(row):
        return ((baseline['min_size']-row['min_size'])/size_span +
                (baseline['min_sharpness']-row['min_sharpness'])/sharp_span)
    unique = {}
    for row in sorted(candidates, key=lambda item: (relaxation(item), -item['min_size'],
                                                     -item['min_sharpness'])):
        signature = (row['candidate_quality_pass'], row['matcher_available'],
            row['gate_a_would_pass'], row['newly_admitted_gate_a_would_pass'],
            row['newly_admitted_low_margin_count'],
            row['newly_admitted_borderline_similarity_count'],
            row['newly_admitted_top1']['median'], row['newly_admitted_margin']['median'])
        unique.setdefault(signature, row)
    candidates = list(unique.values())
    frontier = []
    for row in candidates:
        row_risk = row['newly_admitted_low_margin_count'] / row['newly_admitted_matcher_available']
        dominated = False
        for other in candidates:
            if other is row:
                continue
            other_risk = other['newly_admitted_low_margin_count'] / other['newly_admitted_matcher_available']
            at_least = (other['gate_a_would_pass'] >= row['gate_a_would_pass'] and
                        other['newly_admitted_gate_a_rate'] >= row['newly_admitted_gate_a_rate'] and
                        other_risk <= row_risk and
                        other['newly_admitted'] <= row['newly_admitted'] and
                        other['newly_admitted_low_margin_count'] <=
                            row['newly_admitted_low_margin_count'])
            strict = (other['gate_a_would_pass'] > row['gate_a_would_pass'] or
                      other['newly_admitted_gate_a_rate'] > row['newly_admitted_gate_a_rate'] or
                      other_risk < row_risk or
                      other['newly_admitted'] < row['newly_admitted'] or
                      other['newly_admitted_low_margin_count'] <
                          row['newly_admitted_low_margin_count'])
            if at_least and strict:
                dominated = True
                break
        if not dominated:
            frontier.append(row)
    frontier.sort(key=lambda row: (relaxation(row), -row['gate_a_would_pass']))
    return frontier[:4]


def analyze(observation_path, summary, sizes, sharpnesses):
    config = summary['configuration']['quality']
    point = summary['configuration']['operating_point']
    baseline_size, baseline_sharpness = config['min_size'], config['min_sharpness']
    records = []
    with Path(observation_path).open(encoding='utf-8') as source:
        for line_number, line in enumerate(source, 1):
            if not line.strip():
                continue
            try:
                raw = json.loads(line)
            except json.JSONDecodeError as error:
                raise ValueError(f'Invalid observation JSON at line {line_number}') from error
            records.append(normalize_record(raw, baseline_size, baseline_sharpness, config))
    pairs = [(float(size), float(sharp)) for size in sizes for sharp in sharpnesses]
    if (float(baseline_size), float(baseline_sharpness)) not in pairs:
        pairs.append((float(baseline_size), float(baseline_sharpness)))
    rows = [analyze_candidate(records, size, sharp, baseline_size, baseline_sharpness,
            point['similarity_threshold'], point['margin_threshold'], config)
            for size, sharp in pairs]
    baseline = next(row for row in rows if row['baseline'])
    persisted = summary['videos'][0]['funnel']
    metadata = dict(observations_path=str(Path(observation_path).resolve()),
        source_video=summary['videos'][0]['video'], video_sha256=summary['videos'][0]['sha256'],
        gallery_generation=summary.get('gallery_generation'), total_observations=len(records),
        baseline_min_size=baseline_size, baseline_min_sharpness=baseline_sharpness,
        similarity_threshold=point['similarity_threshold'], margin_threshold=point['margin_threshold'],
        persisted_quality_pass=persisted['quality_passed'],
        offline_baseline_quality_pass=baseline['candidate_quality_pass'],
        baseline_reproduction_exact=baseline['candidate_quality_pass'] == persisted['quality_passed'],
        predicted_identity_semantics='Predicted identity distribution only; NOT ground-truth accuracy.',
        gate_b_simulated=False, adaface_inference_performed=False, detector_run=False,
        runtime_configuration_modified=False)
    return dict(metadata=metadata, candidates=rows,
                development_shortlist=shortlist(
                    rows, baseline, point['similarity_threshold']))


CSV_COLUMNS = ('min_size', 'min_sharpness', 'baseline', 'total_detected',
    'structurally_valid', 'eligible_for_what_if', 'candidate_quality_pass',
    'candidate_quality_pass_rate', 'newly_admitted', 'still_rejected',
    'matcher_available', 'matcher_unavailable', 'production_matcher_count',
    'shadow_matcher_count', 'similarity_pass', 'margin_pass', 'gate_a_would_pass',
    'gate_a_would_pass_rate', 'low_margin_count', 'borderline_similarity_count',
    'high_similarity_low_margin_count', 'newly_admitted_matcher_available',
    'newly_admitted_matcher_unavailable', 'newly_admitted_gate_a_would_pass',
    'newly_admitted_gate_a_rate', 'newly_admitted_low_margin_count',
    'newly_admitted_borderline_similarity_count',
    'newly_admitted_high_similarity_low_margin_count',
    'newly_admitted_top1_median', 'newly_admitted_margin_median',
    'newly_admitted_min_side_median', 'newly_admitted_sharpness_median',
    'originally_face_too_small', 'originally_blurred_face',
    'originally_small_and_blurred', 'originally_extreme_pose',
    'originally_poor_lighting', 'originally_low_detector_confidence',
    'originally_invalid_landmarks', 'originally_alignment_failure', 'originally_other')


def csv_row(row):
    result = {key: row.get(key) for key in CSV_COLUMNS}
    result.update(newly_admitted_top1_median=row['newly_admitted_top1']['median'],
        newly_admitted_margin_median=row['newly_admitted_margin']['median'],
        newly_admitted_min_side_median=row['newly_admitted_min_side']['median'],
        newly_admitted_sharpness_median=row['newly_admitted_sharpness']['median'])
    reasons = row['original_reason_counts']
    for reason in ('face_too_small', 'blurred_face', 'extreme_pose', 'poor_lighting',
                   'low_detector_confidence', 'invalid_landmarks', 'alignment_failure', 'other'):
        result['originally_'+reason] = reasons.get(reason, 0)
    result['originally_small_and_blurred'] = reasons.get('small_and_blurred', 0)
    return result


def write_outputs(result, output):
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    with (output/'quality_sweep.csv').open('x', newline='', encoding='utf-8') as target:
        writer = csv.DictWriter(target, fieldnames=CSV_COLUMNS)
        writer.writeheader()
        writer.writerows(csv_row(row) for row in result['candidates'])
    (output/'quality_sweep.json').write_text(
        json.dumps(result, indent=2, allow_nan=False), encoding='utf-8')
    identities = {f"size={row['min_size']:g},sharpness={row['min_sharpness']:g}": dict(
        all_matcher_available=row['predicted_identity_distribution_all'],
        gate_a_would_pass=row['predicted_identity_distribution_gate_a'])
        for row in result['candidates']}
    (output/'candidate_identity_distributions.json').write_text(
        json.dumps(dict(semantics='Predicted identity distribution only; NOT ground-truth accuracy.',
                        candidates=identities), indent=2), encoding='utf-8')
    (output/'quality_sweep_summary.md').write_text(render_markdown(result), encoding='utf-8')


def _number(value, digits=4):
    return 'N/A' if value is None else f'{value:.{digits}f}'


def render_markdown(result):
    meta, rows = result['metadata'], result['candidates']
    baseline = next(row for row in rows if row['baseline'])
    table = ['| Size | Sharpness | Baseline | Quality pass | Newly admitted | Matcher available | Gate A what-if | New Top1 median | New margin median |',
             '|---:|---:|:---:|---:|---:|---:|---:|---:|---:|']
    for row in rows:
        table.append(f"| {row['min_size']:g} | {row['min_sharpness']:g} | "
            f"{'yes' if row['baseline'] else 'no'} | {row['candidate_quality_pass']} | "
            f"{row['newly_admitted']} | {row['matcher_available']} | {row['gate_a_would_pass']} | "
            f"{_number(row['newly_admitted_top1']['median'])} | {_number(row['newly_admitted_margin']['median'])} |")
    shortlist_lines = []
    for row in result['development_shortlist']:
        shortlist_lines.append(
            f"- `{row['min_size']:g}/{row['min_sharpness']:g}`: +{row['newly_admitted']} observations, "
            f"+{row['newly_admitted_gate_a_would_pass']} observation-level Gate-A passes; "
            f"new Top1 median {_number(row['newly_admitted_top1']['median'])}, "
            f"new margin median {_number(row['newly_admitted_margin']['median'])}, "
            f"new low-margin {row['newly_admitted_low_margin_count']}. Development candidate only.")
    permissive = next((row for row in rows if row['min_size'] == min(r['min_size'] for r in rows)
                       and row['min_sharpness'] == min(r['min_sharpness'] for r in rows)), None)
    discrepancy = ('exactly matches' if meta['baseline_reproduction_exact'] else 'DOES NOT exactly match')
    return '\n'.join([
        '# Phase 8B.1 offline Quality Gate sweep', '', '## A. Input',
        f"- Video: `{meta['source_video']}`", f"- SHA-256: `{meta['video_sha256']}`",
        f"- Observations: {meta['total_observations']}",
        f"- Baseline: size {meta['baseline_min_size']:g}, sharpness {meta['baseline_min_sharpness']:g}",
        f"- Fixed Gate A: T={meta['similarity_threshold']:g}, M={meta['margin_threshold']:g}", '',
        '## B. Method',
        'Persisted observation metadata and persisted production/shadow matcher scores only.',
        'NO AdaFace inference was performed. NO detector was run. NO runtime configuration was modified.',
        'Size follows the runtime integer crop rule: `min(width, height) >= min_size`. All detector, area, lighting, pose, validity, and alignment conditions stay fixed.',
        'Production matcher results take precedence; shadow results are used only when production results are unavailable.', '',
        '## C. Baseline',
        f"Offline baseline {discrepancy} persisted quality admission: {baseline['candidate_quality_pass']} vs {meta['persisted_quality_pass']}.", '',
        '## D. Sweep table', *table, '',
        '## E. Pareto-like development candidates',
        'Screen: newly admitted Top1 median must meet fixed T and at least half of newly admitted matcher-available observations must pass observation-level Gate A. Dominated and outcome-duplicate policies are removed.',
        *(shortlist_lines or ['- No defensible candidate emerged from the measured frontier.']), '',
        '## F. Why not lower everything',
        (f"The most permissive `{permissive['min_size']:g}/{permissive['min_sharpness']:g}` pair admits "
         f"{permissive['newly_admitted']} new observations. Among {permissive['newly_admitted_matcher_available']} "
         f"with matcher evidence, {permissive['newly_admitted_gate_a_would_pass']} pass observation-level Gate A, "
         f"{permissive['newly_admitted_low_margin_count']} have margin below M, and "
         f"{permissive['newly_admitted_borderline_similarity_count']} are borderline similarity. This is descriptive evidence, not acceptance safety."), '',
        '## G. Limitations',
        '- No per-observation ground truth; predicted identities are not accuracy.',
        '- No Accuracy, FAR, FRR, Wrong Identity Rate, precision, recall, or F1.',
        '- Observation-level Gate-A what-if only; no Gate-B simulation.',
        '- One normal-light classroom clip only.',
        '- New thresholds require other-lighting analysis and real coordinator replay before production use.',
        '- Existing shadow embeddings cannot reconstruct changed Top-K, aggregation, or temporal sequences.', '',
        '## H. Conclusion',
        'QUALITY_SWEEP_COMPLETED = YES',
        'PRODUCTION_QUALITY_THRESHOLDS_CHANGED = NO',
        'PRODUCTION_OPERATING_POINT_SELECTED = NO', ''])
