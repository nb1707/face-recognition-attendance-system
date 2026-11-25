"""Pure Phase 8B.2 spatial annotation and labelled diagnostic analysis.

Only persisted JSON/JSONL is consumed. This module imports no video, detector,
model, embedding, coordinator, attendance, cloud, or network code.
"""
from collections import Counter, defaultdict
import csv
import json
import math
from pathlib import Path

from analysis.quality_sweep import (available_matcher, candidate_quality_passes,
                                    distribution, gate_a_would_pass, quality_sides)


EXPECTED_GALLERY = '20261103T035251Z-11abb72e'
CANDIDATE_POLICIES = ((32, 20), (32, 16), (32, 12),
                      (28, 12), (24, 12), (24, 8))
DISTANCE_GROUPS = ('near', 'medium', 'far', 'moving')
LIGHTING_GROUPS = ('normal', 'low', 'poor_low')


def _finite(value):
    return value is not None and math.isfinite(float(value))


def load_json(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def bbox_center_normalized(record, frame_width, frame_height):
    box = record.get('bbox')
    if (not isinstance(box, list) or len(box) != 4 or
            not all(_finite(value) for value in box)):
        raise ValueError('Observation bbox must contain four finite coordinates')
    if not _finite(frame_width) or not _finite(frame_height) or frame_width <= 0 or frame_height <= 0:
        raise ValueError('Positive manually verified frame dimensions are required')
    x = (float(box[0]) + float(box[2])) / (2 * float(frame_width))
    y = (float(box[1]) + float(box[3])) / (2 * float(frame_height))
    return x, y


def _valid_box(region):
    values = [region.get(key) for key in ('x1', 'y1', 'x2', 'y2')]
    return (all(_finite(value) for value in values) and
            0 <= values[0] < values[2] <= 1 and 0 <= values[1] < values[3] <= 1)


def _contains(region, x, y):
    return region['x1'] <= x <= region['x2'] and region['y1'] <= y <= region['y2']


def _active(item, sequence):
    """
    Return whether an annotation item is active at this frame_sequence.

    Moving segments and conditions require explicit frame_start/frame_end.

    Fixed regions may omit frame_start/frame_end, in which case they are
    active for the entire replay.
    """

    frame_start = item.get('frame_start')
    frame_end = item.get('frame_end')

    if frame_start is None and frame_end is None:
        return True

    if not isinstance(frame_start, int):
        return False

    if not isinstance(frame_end, int):
        return False

    return frame_start <= sequence <= frame_end


def assign_observation(record, clip_annotation):
    """Moving segments take precedence; overlaps never force an identity."""
    x, y = bbox_center_normalized(record, clip_annotation['frame_width'],
                                  clip_annotation['frame_height'])
    sequence = record.get('frame_sequence')
    if not isinstance(sequence, int):
        raise ValueError('Observation frame_sequence must be an integer')
    moving = [segment for segment in clip_annotation.get('moving_segments', [])
              if _active(segment, sequence) and _contains(segment['region'], x, y)]
    if len(moving) > 1:
        return dict(status='ambiguous', expected_user_id=None, distance_group=None,
                    annotation_source='moving_segment', annotation_ids=[s['segment_id'] for s in moving],
                    center_x=x, center_y=y, condition=None)
    if len(moving) == 1:
        selected = moving[0]
        result = dict(status='assigned', expected_user_id=selected['user_id'],
            distance_group=selected.get('distance_group', 'moving'),
            annotation_source='moving_segment', annotation_ids=[selected['segment_id']],
            center_x=x, center_y=y, condition=None)
    else:
        fixed = [
            region
            for region in clip_annotation.get('fixed_regions', [])
            if _active(region, sequence)
            and _contains(region, x, y)
        ]
        if len(fixed) > 1:
            return dict(status='ambiguous', expected_user_id=None, distance_group=None,
                        annotation_source='fixed_region', annotation_ids=[r['region_id'] for r in fixed],
                        center_x=x, center_y=y, condition=None)
        if not fixed:
            return dict(status='unassigned', expected_user_id=None, distance_group=None,
                        annotation_source=None, annotation_ids=[], center_x=x, center_y=y,
                        condition=None)
        selected = fixed[0]
        result = dict(status='assigned', expected_user_id=selected['expected_user_id'],
            distance_group=selected['distance_group'], annotation_source='fixed_region',
            annotation_ids=[selected['region_id']], center_x=x, center_y=y, condition=None)
    conditions = [condition['condition'] for condition in clip_annotation.get('conditions', [])
                  if condition['user_id'] == result['expected_user_id'] and
                  _active(condition, sequence)]
    if len(conditions) > 1:
        result['condition'] = '+'.join(sorted(set(conditions)))
    elif conditions:
        result['condition'] = conditions[0]
    return result


def validate_annotations(metadata, annotations):
    errors, warnings = [], []
    if annotations.get('schema_version') != 1:
        errors.append('schema_version must be 1')
    clips = annotations.get('clips')
    if not isinstance(clips, dict):
        return dict(complete=False, errors=['clips must be an object'], warnings=[])
    missing = sorted(set(metadata) - set(clips))
    extra = sorted(set(clips) - set(metadata))
    if missing:
        errors.append('missing clip annotations: '+', '.join(missing))
    if extra:
        errors.append('unknown clip annotations: '+', '.join(extra))
    for clip_id, clip_meta in metadata.items():
        clip = clips.get(clip_id)
        if not isinstance(clip, dict):
            continue
        if not _finite(clip.get('frame_width')) or not _finite(clip.get('frame_height')):
            errors.append(f'{clip_id}: frame_width/frame_height require manually verified values')
        elif clip['frame_width'] <= 0 or clip['frame_height'] <= 0:
            errors.append(f'{clip_id}: frame dimensions must be positive')
        expected_by_distance = {distance: set(clip_meta.get(distance, []))
                                for distance in DISTANCE_GROUPS}
        seen_ids = set()
        for region in clip.get('fixed_regions', []):
            region_id = region.get('region_id')
            if not region_id or region_id in seen_ids:
                errors.append(f'{clip_id}: fixed region_id is missing or duplicated')
            seen_ids.add(region_id)
            distance, user = region.get('distance_group'), region.get('expected_user_id')
            if distance not in DISTANCE_GROUPS:
                errors.append(f'{clip_id}/{region_id}: invalid distance_group')
            elif user not in expected_by_distance.get(distance, set()):
                errors.append(f'{clip_id}/{region_id}: identity is inconsistent with clip metadata')
            if not _valid_box(region):
                errors.append(f'{clip_id}/{region_id}: normalized coordinates are incomplete or invalid')
            frame_start = region.get('frame_start')
            frame_end = region.get('frame_end')

            if frame_start is not None or frame_end is not None:

                if not isinstance(frame_start, int) or not isinstance(frame_end, int):
                    errors.append(
                        f'{clip_id}/{region_id}: '
                        'fixed region frame_start/frame_end must both be integers'
                    )

                elif frame_start > frame_end:
                    errors.append(
                        f'{clip_id}/{region_id}: '
                        'fixed region frame range is reversed'
                    )
        for segment in clip.get('moving_segments', []):
            segment_id = segment.get('segment_id')
            if not segment_id or segment_id in seen_ids:
                errors.append(f'{clip_id}: moving segment_id is missing or duplicated')
            seen_ids.add(segment_id)
            if segment.get('user_id') not in expected_by_distance.get('moving', set()):
                errors.append(f'{clip_id}/{segment_id}: moving identity is inconsistent with metadata')
            if not isinstance(segment.get('frame_start'), int) or not isinstance(segment.get('frame_end'), int):
                errors.append(f'{clip_id}/{segment_id}: integer frame range is required')
            elif segment['frame_start'] > segment['frame_end']:
                errors.append(f'{clip_id}/{segment_id}: frame range is reversed')
            if not isinstance(segment.get('region'), dict) or not _valid_box(segment['region']):
                errors.append(f'{clip_id}/{segment_id}: normalized moving region is incomplete or invalid')
        for condition in clip.get('conditions', []):
            if not condition.get('user_id') or not condition.get('condition'):
                errors.append(f'{clip_id}: condition user_id/condition is required')
            if not isinstance(condition.get('frame_start'), int) or not isinstance(condition.get('frame_end'), int):
                errors.append(f'{clip_id}: condition requires integer frame range')
            elif condition['frame_start'] > condition['frame_end']:
                errors.append(f'{clip_id}: condition frame range is reversed')
        annotated = Counter(region.get('distance_group') for region in clip.get('fixed_regions', []))
        annotated['moving'] += len(clip.get('moving_segments', []))
        annotated_users = defaultdict(set)
        for region in clip.get('fixed_regions', []):
            annotated_users[region.get('distance_group')].add(region.get('expected_user_id'))
        for segment in clip.get('moving_segments', []):
            annotated_users['moving'].add(segment.get('user_id'))
        for distance, users in expected_by_distance.items():
            if users and annotated[distance] == 0:
                warnings.append(f'{clip_id}: no completed {distance} annotation exists')
            missing_users = sorted(users - annotated_users[distance])
            if missing_users:
                errors.append(f"{clip_id}: missing {distance} identities: {', '.join(missing_users)}")
    complete_flag = annotations.get('complete') is True
    if not complete_flag:
        errors.append('annotation complete flag is false; manual review is required')
    return dict(complete=complete_flag and not errors, errors=errors, warnings=warnings)


def discover_replays(replay_root, metadata, expected_gallery=EXPECTED_GALLERY):
    """Select exactly one shadow replay per clip and fail closed on its gallery."""
    records = []
    for summary_path in Path(replay_root).rglob('replay_summary.json'):
        summary = load_json(summary_path)
        videos = summary.get('videos', [])
        if len(videos) != 1:
            continue
        clip_id = Path(videos[0].get('video', '')).stem
        if clip_id not in metadata or summary.get('shadow_matcher') is not True:
            continue
        records.append(dict(clip_id=clip_id, summary_path=summary_path,
            observations_path=summary_path.parent/'observations-0.jsonl',
            gallery_generation=summary.get('gallery_generation'), summary=summary))
    selected, rejected = {}, []
    for clip_id in metadata:
        matches = [row for row in records if row['clip_id'] == clip_id]
        valid = [row for row in matches if row['gallery_generation'] == expected_gallery]
        rejected.extend(dict(clip_id=clip_id, summary_path=str(row['summary_path'].resolve()),
                             gallery_generation=row['gallery_generation'], reason='gallery_mismatch')
                        for row in matches if row['gallery_generation'] != expected_gallery)
        if len(valid) != 1:
            raise ValueError(f'{clip_id}: expected exactly one shadow replay with gallery {expected_gallery}; found {len(valid)}')
        row = valid[0]
        if not row['observations_path'].is_file():
            raise ValueError(f'{clip_id}: observations-0.jsonl is missing')
        selected[clip_id] = row
    return selected, rejected


def validate_assignment_coverage(selected, metadata, annotations):
    clips, total = {}, Counter()
    for clip_id, replay in selected.items():
        clip_annotation = annotations['clips'][clip_id]
        counts, regions, identities = Counter(), Counter(), Counter()
        with replay['observations_path'].open(encoding='utf-8') as source:
            for line in source:
                if not line.strip():
                    continue
                assignment = assign_observation(json.loads(line), clip_annotation)
                counts[assignment['status']] += 1
                total[assignment['status']] += 1
                if assignment['status'] == 'assigned':
                    identities[assignment['expected_user_id']] += 1
                    regions[assignment['annotation_ids'][0]] += 1
        clips[clip_id] = dict(video=replay['summary']['videos'][0]['video'],
            expected_identities=sorted({user for group in DISTANCE_GROUPS
                                        for user in metadata[clip_id].get(group, [])}),
            counts=dict(counts), observations_per_region=dict(sorted(regions.items())),
            observations_per_identity=dict(sorted(identities.items())))
    return dict(total=dict(total), clips=clips)


class MetricAccumulator:
    def __init__(self):
        self.counts = Counter()
        self.reasons = Counter()
        self.values = defaultdict(list)

    def add(self, labelled):
        record, expected = labelled['record'], labelled['assignment']['expected_user_id']
        self.counts['assigned_observations'] += 1
        valid = bool(record.get('valid_observation'))
        quality = bool(record.get('quality_passed'))
        self.counts['structurally_valid'] += valid
        self.counts['quality_pass' if quality else 'quality_reject'] += 1
        if not quality:
            for reason in record.get('quality_reasons', []):
                mapped = {'face_too_small': 'small', 'blurred_face': 'blur',
                          'poor_lighting': 'lighting', 'extreme_pose': 'pose',
                          'low_detector_confidence': 'detector'}.get(reason, 'other')
                self.reasons[mapped] += 1
        width, height = quality_sides(record)
        for name, value in (('face_width', width), ('face_height', height),
            ('minimum_face_side', min(width, height) if width is not None else None),
            ('sharpness', record.get('sharpness')),
            ('detector_confidence', record.get('detector_confidence'))):
            if _finite(value):
                self.values[name].append(float(value))
        match = available_matcher(record)
        if match['source'] == 'unavailable':
            return
        self.counts['matcher_available'] += 1
        correct = match['top1_user_id'] == expected
        self.counts['predicted_top1_correct' if correct else 'predicted_top1_wrong'] += 1
        if _finite(match['top1_score']):
            self.values['top1'].append(match['top1_score'])
            self.values['correct_top1_score' if correct else 'wrong_top1_score'].append(match['top1_score'])
        if _finite(match['margin']):
            self.values['margin'].append(match['margin'])
            self.values['correct_margin' if correct else 'wrong_margin'].append(match['margin'])
        gate = gate_a_would_pass(match, .4, .05)
        self.counts['observation_level_gate_a_pass'] += gate
        if gate:
            self.counts['correct_gate_a' if correct else 'wrong_gate_a'] += 1

    def summary(self):
        assigned, matcher = self.counts['assigned_observations'], self.counts['matcher_available']
        result = dict(self.counts)
        result.update(quality_pass_rate=self.counts['quality_pass']/assigned if assigned else None,
            correct_top1_rate=self.counts['predicted_top1_correct']/matcher if matcher else None,
            wrong_top1_rate=self.counts['predicted_top1_wrong']/matcher if matcher else None,
            correct_gate_a_rate=self.counts['correct_gate_a']/matcher if matcher else None,
            wrong_gate_a_rate=self.counts['wrong_gate_a']/matcher if matcher else None,
            rejection_reasons={name: self.reasons[name] for name in
                ('small', 'blur', 'lighting', 'pose', 'detector', 'other')},
            distributions={name: distribution(values) for name, values in self.values.items()})
        return result


def labelled_records(selected, annotations):
    """Stream only unambiguous assigned observations into labelled analysis."""
    for clip_id, replay in selected.items():
        clip_annotation = annotations['clips'][clip_id]
        with replay['observations_path'].open(encoding='utf-8') as source:
            for line in source:
                if not line.strip():
                    continue
                record = json.loads(line)
                assignment = assign_observation(record, clip_annotation)
                if assignment['status'] == 'assigned':
                    yield dict(clip_id=clip_id, record=record, assignment=assignment)


def aggregate_labelled(records):
    groups = dict(identity=defaultdict(MetricAccumulator),
                  distance=defaultdict(MetricAccumulator),
                  clip=defaultdict(MetricAccumulator),
                  lighting=defaultdict(MetricAccumulator),
                  distance_lighting=defaultdict(MetricAccumulator),
                  condition=defaultdict(MetricAccumulator))
    for item in records:
        assignment = item['assignment']
        keys = dict(identity=assignment['expected_user_id'], distance=assignment['distance_group'],
                    clip=item['clip_id'], lighting=item['lighting'],
                    distance_lighting=f"{assignment['distance_group']}|{item['lighting']}",
                    condition=assignment.get('condition'))
        for dimension, key in keys.items():
            if key:
                groups[dimension][key].add(item)
    return {dimension: {key: value.summary() for key, value in sorted(items.items())}
            for dimension, items in groups.items()}


def candidate_metrics(records, quality_config, policies=CANDIDATE_POLICIES):
    dimensions = ('overall', 'identity', 'distance', 'clip', 'lighting', 'distance_lighting')
    output = []
    for size, sharpness in policies:
        accumulators = {dimension: defaultdict(Counter) for dimension in dimensions}
        for item in records:
            record, assignment = item['record'], item['assignment']
            match = available_matcher(record)
            passed = candidate_quality_passes(record, size, sharpness, quality_config)
            baseline = candidate_quality_passes(record, 32, 20, quality_config)
            keys = dict(overall='all', identity=assignment['expected_user_id'],
                        distance=assignment['distance_group'], clip=item['clip_id'],
                        lighting=item['lighting'],
                        distance_lighting=f"{assignment['distance_group']}|{item['lighting']}")
            for dimension, key in keys.items():
                counter = accumulators[dimension][key]
                counter['labelled_observations'] += 1
                if not passed:
                    continue
                counter['candidate_quality_pass'] += 1
                if match['source'] == 'unavailable':
                    continue
                counter['matcher_available'] += 1
                correct = match['top1_user_id'] == assignment['expected_user_id']
                counter['correct_top1' if correct else 'wrong_top1'] += 1
                gate = gate_a_would_pass(match, .4, .05)
                counter['gate_a_pass'] += gate
                if gate:
                    counter['correct_gate_a' if correct else 'wrong_gate_a'] += 1
                    if not baseline:
                        counter['newly_admitted_correct_gate_a' if correct else
                                'newly_admitted_wrong_gate_a'] += 1
        output.append(dict(min_size=size, min_sharpness=sharpness,
            baseline=size == 32 and sharpness == 20,
            breakdown={dimension: {key: dict(value) for key, value in items.items()}
                       for dimension, items in accumulators.items()}))
    return output


def run_full_analysis(selected, metadata, annotations):
    items = []
    for item in labelled_records(selected, annotations):
        item['lighting'] = metadata[item['clip_id']]['lighting']
        items.append(item)
    summary = next(iter(selected.values()))['summary']
    quality_config = summary['configuration']['quality']
    return dict(gallery_generation=summary['gallery_generation'], gate_b_simulated=False,
                metrics=aggregate_labelled(items),
                candidate_comparison=candidate_metrics(items, quality_config),
                limitations=['manually defined spatial ground truth',
                    'moving subject requires manual segments',
                    'unassigned/ambiguous observations excluded',
                    'observation-level Gate A only', 'no Gate B',
                    'no production threshold selected', 'development dataset used for tuning'])


def write_validation(output, validation, discovery, coverage=None):
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    value = dict(annotation_validation=validation, input_discovery=discovery,
                 assignment_coverage=coverage,
                 ready_for_labelled_metrics=bool(validation['complete'] and coverage is not None))
    (output/'phase8b2_annotation_validation.json').write_text(
        json.dumps(value, indent=2, allow_nan=False), encoding='utf-8')
    lines = ['# Phase 8B.2 annotation validation', '',
        f"READY_FOR_LABELLED_METRICS = {'YES' if value['ready_for_labelled_metrics'] else 'NO'}", '',
        '## Errors', *(f'- {item}' for item in validation['errors']), '',
        '## Warnings', *(f'- {item}' for item in validation['warnings']), '',
        'No identity accuracy metrics were produced by validation mode.']
    (output/'phase8b2_annotation_validation.md').write_text('\n'.join(lines)+'\n', encoding='utf-8')


def write_full_outputs(output, result, coverage):
    """Write labelled outputs only after complete annotation validation."""
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    (output/'phase8b2_results.json').write_text(json.dumps(
        dict(result=result, assignment_coverage=coverage), indent=2, allow_nan=False), encoding='utf-8')
    metric_columns = ('group', 'assigned_observations', 'structurally_valid', 'quality_pass',
        'quality_reject', 'quality_pass_rate', 'matcher_available', 'predicted_top1_correct',
        'predicted_top1_wrong', 'correct_top1_rate', 'wrong_top1_rate',
        'observation_level_gate_a_pass', 'correct_gate_a', 'wrong_gate_a',
        'correct_gate_a_rate', 'wrong_gate_a_rate')
    for dimension, filename in (('identity', 'phase8b2_per_identity.csv'),
                                ('distance', 'phase8b2_per_distance.csv'),
                                ('clip', 'phase8b2_per_clip.csv')):
        with (output/filename).open('x', newline='', encoding='utf-8') as target:
            writer = csv.DictWriter(target, fieldnames=metric_columns)
            writer.writeheader()
            for group, row in result['metrics'][dimension].items():
                writer.writerow({key: group if key == 'group' else row.get(key) for key in metric_columns})
    candidate_columns = ('min_size', 'min_sharpness', 'baseline', 'dimension', 'group',
        'labelled_observations', 'candidate_quality_pass', 'matcher_available', 'correct_top1',
        'wrong_top1', 'gate_a_pass', 'correct_gate_a', 'wrong_gate_a',
        'newly_admitted_correct_gate_a', 'newly_admitted_wrong_gate_a')
    with (output/'phase8b2_candidate_comparison.csv').open('x', newline='', encoding='utf-8') as target:
        writer = csv.DictWriter(target, fieldnames=candidate_columns)
        writer.writeheader()
        for policy in result['candidate_comparison']:
            for dimension, groups in policy['breakdown'].items():
                for group, row in groups.items():
                    writer.writerow({key: (policy.get(key) if key in policy else
                        dimension if key == 'dimension' else group if key == 'group' else row.get(key, 0))
                        for key in candidate_columns})
    (output/'phase8b2_assignment_diagnostics.csv').write_text(
        'See phase8b2_results.json assignment_coverage for complete region counts.\n', encoding='utf-8')
    (output/'phase8b2_summary.md').write_text('\n'.join([
        '# Phase 8B.2 labelled spatial analysis', '',
        'This report uses manually supplied spatial annotations. Predicted identity is never used as ground truth.', '',
        'No Gate B was simulated. No production threshold was selected.', '',
        'PRODUCTION_QUALITY_THRESHOLDS_CHANGED = NO',
        'PRODUCTION_OPERATING_POINT_SELECTED = NO', '']), encoding='utf-8')
