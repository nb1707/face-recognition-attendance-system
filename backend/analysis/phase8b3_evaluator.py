"""Evaluate persisted Phase 8B.3 controlled replays without running inference."""

from __future__ import annotations

from collections import Counter, defaultdict
import csv
from datetime import datetime, timezone
import json
import math
from pathlib import Path
import re
import statistics

from analysis.spatial_ground_truth import assign_observation


EXPECTED_GALLERY = '20261103T035251Z-11abb72e'
CANDIDATES = {
    '32_20': {'min_size': 32.0, 'min_sharpness': 20.0, 'label': '32/20 baseline'},
    '32_16': {'min_size': 32.0, 'min_sharpness': 16.0, 'label': '32/16 conservative'},
    '24_12': {'min_size': 24.0, 'min_sharpness': 12.0, 'label': '24/12 balanced'},
}
CANDIDATE_ORDER = tuple(CANDIDATES)
REQUIRED_FILES = (
    'frames-0.jsonl', 'observations-0.jsonl', 'tracks-0.jsonl',
    'events-0.jsonl', 'replay_summary.json',
)
RUN_TIMESTAMP = re.compile(r'-(\d{8})-(\d{6})$')


def load_json(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def read_jsonl(path):
    records = []
    with Path(path).open(encoding='utf-8') as source:
        for line_number, line in enumerate(source, 1):
            if not line.strip():
                continue
            try:
                value = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f'{path}:{line_number}: invalid JSON: {exc}') from exc
            if not isinstance(value, dict):
                raise ValueError(f'{path}:{line_number}: expected a JSON object')
            records.append(value)
    return records


def _run_rank(path):
    """Prefer the timestamp encoded by replay_offline in the directory name."""
    match = RUN_TIMESTAMP.search(path.name)
    if match:
        return (1, match.group(1) + match.group(2), path.name)
    summary = path / 'replay_summary.json'
    return (0, f'{summary.stat().st_mtime_ns:020d}', path.name)


def _summary_clip_id(summary):
    videos = summary.get('videos')
    if not isinstance(videos, list) or len(videos) != 1:
        return None
    video = videos[0].get('video')
    return Path(video).stem if isinstance(video, str) and video else None


def select_runs(replay_root, metadata, candidates=CANDIDATE_ORDER):
    """Select the latest complete, valid-summary run for every candidate/clip."""
    replay_root = Path(replay_root)
    selected, rejected = {}, []
    clip_ids = set(metadata)
    for candidate in candidates:
        candidate_root = replay_root / candidate
        eligible = defaultdict(list)
        if not candidate_root.is_dir():
            raise ValueError(f'{candidate}: candidate directory is missing')
        for run_dir in sorted((p for p in candidate_root.iterdir() if p.is_dir()),
                              key=lambda p: p.name):
            missing = [name for name in REQUIRED_FILES if not (run_dir / name).is_file()]
            if missing:
                rejected.append({'candidate': candidate, 'directory': str(run_dir.resolve()),
                                 'reason': 'incomplete', 'missing_files': missing})
                continue
            try:
                summary = load_json(run_dir / 'replay_summary.json')
            except (OSError, json.JSONDecodeError) as exc:
                rejected.append({'candidate': candidate, 'directory': str(run_dir.resolve()),
                                 'reason': 'invalid_replay_summary', 'error': str(exc)})
                continue
            if not isinstance(summary, dict):
                rejected.append({'candidate': candidate, 'directory': str(run_dir.resolve()),
                                 'reason': 'invalid_replay_summary_type'})
                continue
            clip_id = _summary_clip_id(summary)
            if clip_id not in clip_ids:
                rejected.append({'candidate': candidate, 'directory': str(run_dir.resolve()),
                                 'reason': 'unknown_or_multi_video_clip', 'clip_id': clip_id})
                continue
            eligible[clip_id].append({'candidate': candidate, 'clip_id': clip_id,
                                      'directory': run_dir, 'summary': summary})
        for clip_id in metadata:
            choices = eligible.get(clip_id, [])
            if not choices:
                raise ValueError(f'{candidate}/{clip_id}: no complete run with a valid summary')
            choices.sort(key=lambda row: _run_rank(row['directory']))
            chosen = choices[-1]
            selected[(candidate, clip_id)] = chosen
            for older in choices[:-1]:
                rejected.append({'candidate': candidate,
                                 'directory': str(older['directory'].resolve()),
                                 'reason': 'older_complete_duplicate', 'clip_id': clip_id,
                                 'selected_directory': str(chosen['directory'].resolve())})
    return selected, rejected


def _same_number(actual, expected):
    return (isinstance(actual, (int, float)) and not isinstance(actual, bool)
            and math.isclose(float(actual), float(expected), rel_tol=0, abs_tol=1e-12))


def invariant_errors(run):
    candidate, summary = run['candidate'], run['summary']
    expected = CANDIDATES[candidate]
    configuration = summary.get('configuration', {})
    operating = configuration.get('operating_point', {})
    quality = configuration.get('quality', {})
    videos = summary.get('videos', [])
    video = videos[0] if len(videos) == 1 else {}
    checks = (
        ('gallery_generation', summary.get('gallery_generation') == EXPECTED_GALLERY,
         summary.get('gallery_generation'), EXPECTED_GALLERY),
        ('similarity_threshold', _same_number(operating.get('similarity_threshold'), .40),
         operating.get('similarity_threshold'), .40),
        ('margin_threshold', _same_number(operating.get('margin_threshold'), .05),
         operating.get('margin_threshold'), .05),
        ('full_video', summary.get('full_video') is True and video.get('full_video') is True,
         {'summary': summary.get('full_video'), 'video': video.get('full_video')}, True),
        ('stride', video.get('stride') == 5, video.get('stride'), 5),
        ('detector_mode', summary.get('detector_mode') == 'existing tiled detector'
         and configuration.get('detection_mode') == 'tiled',
         {'summary': summary.get('detector_mode'),
          'configuration': configuration.get('detection_mode')},
         {'summary': 'existing tiled detector', 'configuration': 'tiled'}),
        ('quality_candidate.min_size',
         _same_number(summary.get('quality_candidate', {}).get('min_size'), expected['min_size'])
         and _same_number(quality.get('min_size'), expected['min_size']),
         {'candidate': summary.get('quality_candidate', {}).get('min_size'),
          'configuration': quality.get('min_size')}, expected['min_size']),
        ('quality_candidate.min_sharpness',
         _same_number(summary.get('quality_candidate', {}).get('min_sharpness'),
                      expected['min_sharpness'])
         and _same_number(quality.get('min_sharpness'), expected['min_sharpness']),
         {'candidate': summary.get('quality_candidate', {}).get('min_sharpness'),
          'configuration': quality.get('min_sharpness')}, expected['min_sharpness']),
    )
    return [{'candidate': candidate, 'clip_id': run['clip_id'],
             'directory': str(run['directory'].resolve()), 'invariant': name,
             'actual': actual, 'expected': wanted}
            for name, passed, actual, wanted in checks if not passed]


def validate_invariants(selected):
    errors = [error for run in selected.values() for error in invariant_errors(run)]
    if errors:
        details = '; '.join(f"{e['candidate']}/{e['clip_id']} {e['invariant']}="
                            f"{e['actual']!r} (expected {e['expected']!r})" for e in errors)
        raise ValueError('Phase 8B.3 invariant failure: ' + details)
    return []


def _at_or_before(record, event):
    comparisons = []
    for key in ('frame_sequence', 'acquisition_time'):
        if record.get(key) is not None and event.get(key) is not None:
            comparisons.append(record[key] <= event[key])
    return all(comparisons) if comparisons else False


def classify_event(event, observations, clip_annotation):
    """Resolve event GT from same-track observations available at event time."""
    relevant = [row for row in observations
                if row.get('track_id') == event.get('track_id') and _at_or_before(row, event)]
    assignments = [assign_observation(row, clip_annotation) for row in relevant]
    identities = sorted({row['expected_user_id'] for row in assignments
                         if row['status'] == 'assigned'})
    ambiguous = [row for row in assignments if row['status'] == 'ambiguous']
    unassigned_count = sum(row['status'] == 'unassigned' for row in assignments)
    if len(identities) > 1:
        gt_status, gt_identity, classification = 'conflict', None, 'conflict'
    elif len(identities) == 1:
        gt_status, gt_identity = 'unique', identities[0]
        classification = 'correct' if event.get('user_id') == gt_identity else 'wrong'
    elif ambiguous:
        gt_status, gt_identity, classification = 'ambiguous', None, 'ambiguous'
    else:
        gt_status, gt_identity, classification = 'unassigned', None, 'unassigned'
    return {
        'event_gt_status': gt_status,
        'event_gt_identity': gt_identity,
        'classification': classification,
        'relevant_observations': len(relevant),
        'assigned_observations': sum(row['status'] == 'assigned' for row in assignments),
        'unassigned_observations': unassigned_count,
        'ambiguous_observations': len(ambiguous),
        'assigned_identities': identities,
        'spatial_ambiguity_present': bool(ambiguous),
        'ambiguous_annotation_ids': sorted({annotation_id for row in ambiguous
                                            for annotation_id in row['annotation_ids']}),
    }


def _expected_identities(clip_meta):
    return sorted({user for group in ('near', 'medium', 'far', 'moving')
                   for user in clip_meta.get(group, [])})


def _summary_counts(run):
    video = run['summary']['videos'][0]
    funnel = video.get('funnel', {})
    return {
        'detected_observations': funnel.get('detected_face_observations', 0),
        'quality_passed': funnel.get('quality_passed', 0),
        'quality_rejected': funnel.get('quality_rejected', 0),
        'adaface_inferences': funnel.get('adaface_inferences', video.get('adaface_inferences', 0)),
        'gate_a_passes': funnel.get('gate_a_passes', 0),
        'gate_b_confirmations': funnel.get('gate_b_confirmations', 0),
        'first_confirm_source_time': video.get('first_confirm_source_time'),
    }


def _event_csv_row(candidate, clip_id, run, event, resolution, event_index):
    return {
        'candidate': candidate, 'clip_id': clip_id, 'selected_run_directory': str(run.resolve()),
        'event_index': event_index, 'track_id': event.get('track_id'),
        'event_user_id': event.get('user_id'), 'frame_sequence': event.get('frame_sequence'),
        'acquisition_time': event.get('acquisition_time'),
        'event_gt_status': resolution['event_gt_status'],
        'event_gt_identity': resolution['event_gt_identity'],
        'classification': resolution['classification'],
        'relevant_observations': resolution['relevant_observations'],
        'assigned_observations': resolution['assigned_observations'],
        'unassigned_observations': resolution['unassigned_observations'],
        'ambiguous_observations': resolution['ambiguous_observations'],
        'spatial_ambiguity_present': resolution['spatial_ambiguity_present'],
        'assigned_identities': '|'.join(resolution['assigned_identities']),
        'ambiguous_annotation_ids': '|'.join(resolution['ambiguous_annotation_ids']),
        'top1_score': event.get('top1_score'), 'margin': event.get('margin'),
    }


def evaluate(replay_root, metadata_path, annotations_path):
    metadata, annotations = load_json(metadata_path), load_json(annotations_path)
    if not isinstance(metadata, dict) or not isinstance(annotations.get('clips'), dict):
        raise ValueError('Metadata and spatial ground truth must contain clip mappings')
    selected, rejected = select_runs(replay_root, metadata)
    validate_invariants(selected)

    per_clip, event_rows = [], []
    identity_counters = defaultdict(Counter)
    for candidate in CANDIDATE_ORDER:
        for clip_id in metadata:
            run = selected[(candidate, clip_id)]
            run_dir = run['directory']
            observations = read_jsonl(run_dir / 'observations-0.jsonl')
            events = read_jsonl(run_dir / 'events-0.jsonl')
            annotation = annotations['clips'].get(clip_id)
            if not isinstance(annotation, dict):
                raise ValueError(f'{clip_id}: spatial ground truth is missing')

            for observation in observations:
                assignment = assign_observation(observation, annotation)
                if assignment['status'] == 'assigned':
                    identity_counters[(candidate, assignment['expected_user_id'])]['labelled_observations'] += 1

            classifications = Counter()
            correct_ids, wrong_ids = [], []
            for index, event in enumerate(events, 1):
                resolution = classify_event(event, observations, annotation)
                classifications[resolution['classification']] += 1
                event_rows.append(_event_csv_row(candidate, clip_id, run_dir, event,
                                                 resolution, index))
                gt_identity = resolution['event_gt_identity']
                if resolution['classification'] == 'correct':
                    correct_ids.append(event.get('user_id'))
                    identity_counters[(candidate, gt_identity)]['correct_confirmation_event_count'] += 1
                elif resolution['classification'] == 'wrong':
                    wrong_ids.append(event.get('user_id'))
                    identity_counters[(candidate, gt_identity)]['wrong_confirmation_against_gt_identity'] += 1

            counts = _summary_counts(run)
            per_clip.append({
                'candidate': candidate, 'clip_id': clip_id,
                'selected_run_directory': str(run_dir.resolve()),
                **counts, 'attendance_events': len(events),
                'correct_events': classifications['correct'],
                'wrong_events': classifications['wrong'],
                'unassigned_events': classifications['unassigned'],
                'conflict_events': classifications['conflict'],
                'ambiguous_events': classifications['ambiguous'],
                'event_precision': _precision(classifications),
                'unique_correctly_confirmed_identities': len(set(correct_ids)),
                'correct_event_identities': '|'.join(correct_ids),
                'wrong_event_identities': '|'.join(wrong_ids),
            })

            for identity in _expected_identities(metadata[clip_id]):
                identity_counters[(candidate, identity)]['expected_in_clips'] += 1

    per_identity = []
    for (candidate, identity), counts in sorted(identity_counters.items()):
        correct = counts['correct_confirmation_event_count']
        per_identity.append({
            'candidate': candidate, 'expected_gt_identity': identity,
            'expected_in_clips': counts['expected_in_clips'],
            'labelled_observations': counts['labelled_observations'],
            'correct_confirmation_event_count': correct,
            'wrong_confirmation_against_gt_identity':
                counts['wrong_confirmation_against_gt_identity'],
            'ever_correctly_confirmed': correct > 0,
        })

    candidate_summary = _candidate_summary(per_clip, per_identity)
    return {
        'schema_version': 1,
        'generated_at_utc': datetime.now(timezone.utc).isoformat(),
        'statements': {'GATE_B_SIMULATED': True, 'PRODUCTION_CONFIG_CHANGED': False,
                       'PRODUCTION_OPERATING_POINT_SELECTED': False},
        'limitations': [
            'This analysis measures persisted detections and confirmations; it does not measure detector recall.',
            'stu_008 has manually verified regions with zero detector observations in several clips.',
        ],
        'invariant_failures': [],
        'selected_runs': [{'candidate': candidate, 'clip_id': clip_id,
                           'directory': str(run['directory'].resolve())}
                          for (candidate, clip_id), run in selected.items()],
        'rejected_runs': rejected,
        'candidate_summary': candidate_summary,
        'per_clip': per_clip,
        'per_identity': per_identity,
        'events': event_rows,
    }


def _precision(counts):
    denominator = counts['correct'] + counts['wrong']
    return counts['correct'] / denominator if denominator else None


def _candidate_summary(per_clip, per_identity):
    rows = []
    for candidate in CANDIDATE_ORDER:
        clips = [row for row in per_clip if row['candidate'] == candidate]
        totals = Counter()
        for row in clips:
            for field in ('detected_observations', 'quality_passed', 'quality_rejected',
                          'adaface_inferences', 'gate_a_passes', 'gate_b_confirmations',
                          'attendance_events', 'correct_events', 'wrong_events',
                          'unassigned_events', 'conflict_events', 'ambiguous_events'):
                totals[field] += row[field]
        correct_identities = {row['expected_gt_identity'] for row in per_identity
                              if row['candidate'] == candidate and row['ever_correctly_confirmed']}
        times = [row['first_confirm_source_time'] for row in clips
                 if row['first_confirm_source_time'] is not None]
        denominator = totals['correct_events'] + totals['wrong_events']
        rows.append({'candidate': candidate, 'candidate_label': CANDIDATES[candidate]['label'],
                     'clips': len(clips), **dict(totals),
                     'event_precision': totals['correct_events'] / denominator if denominator else None,
                     'unique_correctly_confirmed_identities': len(correct_identities),
                     'median_first_confirm_source_time': statistics.median(times) if times else None})
    return rows


def _write_csv(path, rows):
    if not rows:
        raise ValueError(f'Cannot write empty CSV: {path.name}')
    with path.open('w', encoding='utf-8', newline='') as target:
        writer = csv.DictWriter(target, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def _fmt(value):
    if value is None:
        return 'N/A'
    if isinstance(value, float):
        return f'{value:.3f}'
    return str(value)


def render_summary(result):
    rows = result['candidate_summary']
    baseline_passed = next(row['quality_passed'] for row in rows if row['candidate'] == '32_20')
    lines = [
        '# Phase 8B.3 controlled-replay evaluation', '',
        'GATE_B_SIMULATED = TRUE (because this is actual coordinator replay)',
        'PRODUCTION_CONFIG_CHANGED = NO',
        'PRODUCTION_OPERATING_POINT_SELECTED = NO', '',
        'No production operating point is automatically selected. Compare wrong confirmed identities '
        'first, correct identity coverage second, then Gate B/event count, confirmation time, and '
        'quality/admission increase.', '',
        '| Candidate | Wrong events | Correct identity coverage | Gate B confirmations | Events | '
        'Median first confirm (s) | Quality passed | Admission increase vs 32/20 |',
        '|---|---:|---:|---:|---:|---:|---:|---:|',
    ]
    for row in rows:
        increase = row['quality_passed'] - baseline_passed
        lines.append(f"| {row['candidate_label']} | {row['wrong_events']} | "
                     f"{row['unique_correctly_confirmed_identities']} | "
                     f"{row['gate_b_confirmations']} | {row['attendance_events']} | "
                     f"{_fmt(row['median_first_confirm_source_time'])} | "
                     f"{row['quality_passed']} | {increase:+d} |")
    lines.extend(['', '## Event diagnostics', '',
                  '| Candidate | Correct | Wrong | Unassigned | Conflict | Ambiguous | Precision |',
                  '|---|---:|---:|---:|---:|---:|---:|'])
    for row in rows:
        lines.append(f"| {row['candidate_label']} | {row['correct_events']} | "
                     f"{row['wrong_events']} | {row['unassigned_events']} | "
                     f"{row['conflict_events']} | {row['ambiguous_events']} | "
                     f"{_fmt(row['event_precision'])} |")
    lines.extend(['', '## Selected runs', ''])
    for row in result['selected_runs']:
        lines.append(f"- {row['candidate']} / {row['clip_id']}: `{row['directory']}`")
    lines.extend(['', '## Limitations', '',
                  '- Do not interpret these results as detector recall. The evaluator only analyzes '
                  'existing replay outputs.',
                  '- `stu_008` has manually verified regions with zero detector observations in '
                  'several clips; those missed faces cannot become labelled observations or events.',
                  '- Predicted identities are never used as ground truth. Event GT comes only from '
                  'manual spatial assignment of same-track observations through event time.', ''])
    return '\n'.join(lines)


def write_outputs(result, output_dir):
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=False)
    _write_csv(output_dir / 'phase8b3_candidate_summary.csv', result['candidate_summary'])
    _write_csv(output_dir / 'phase8b3_per_clip.csv', result['per_clip'])
    _write_csv(output_dir / 'phase8b3_per_identity.csv', result['per_identity'])
    _write_csv(output_dir / 'phase8b3_events.csv', result['events'])
    (output_dir / 'phase8b3_results.json').write_text(
        json.dumps(result, indent=2, ensure_ascii=False) + '\n', encoding='utf-8')
    (output_dir / 'phase8b3_summary.md').write_text(render_summary(result), encoding='utf-8')
    return output_dir
