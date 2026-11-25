"""Descriptive, bounded offline replay summaries. No identity ground truth."""
from collections import Counter
import math
import numpy as np

SCHEMA_VERSION = 1
REASONS = ('face_too_small', 'blurred_face', 'poor_lighting', 'extreme_pose',
           'low_detector_confidence', 'invalid_landmarks', 'other')


def stats(values):
    values = [float(v) for v in values if v is not None and math.isfinite(v)]
    keys = ('min', 'median', 'mean', 'p25', 'p75', 'p90', 'p95', 'max')
    if not values:
        return dict(count=0, **dict.fromkeys(keys))
    return dict(count=len(values), min=min(values), median=float(np.median(values)),
                mean=float(np.mean(values)), max=max(values),
                **{f'p{p}': float(np.percentile(values, p)) for p in (25, 75, 90, 95)})


def rate(numerator, denominator):
    return dict(numerator=numerator, denominator=denominator,
                percent=100*numerator/denominator if denominator else None)


def buckets(values):
    edges = (0, 10, 20, 30, 40, 60, 80, float('inf'))
    labels = ('<10', '10-19', '20-29', '30-39', '40-59', '60-79', '>=80')
    return {label: sum(low <= v < high for v in values)
            for label, low, high in zip(labels, edges, edges[1:])}


def summarize(frames, observations, tracks, source_seconds):
    """Input is capped by harness sample/record limits; every rate names its base."""
    measured = [o for o in observations if o['quality'] is not None]
    passed = [o for o in measured if o['quality']['passed']]
    rejected = [o for o in measured if not o['quality']['passed']]
    calls = [c for o in observations for c in o['adaface_calls']]
    matches = [m for o in observations for m in o['matcher_attempts']]
    # One production observation attempt may invoke the matcher twice.
    attempts = [o for o in observations if o['matcher_attempts']]
    candidate_tracks = {(o['video_id'], o['track_id']) for o in observations if o['gate_a_passed']}
    totals = {key: sum(f[key] for f in frames) for key in (
        'detections_raw', 'detections_after_nms', 'quality_pass_count', 'quality_reject_count',
        'topk_selected_count', 'adaface_inference_count', 'gate_a_pass_count',
        'gate_b_confirm_count', 'confirmed_event_count')}
    n = totals['detections_after_nms']
    detection = dict(processed_frames=len(frames),
        frames_with_at_least_one_detection=sum(f['detections_after_nms'] > 0 for f in frames),
        raw_detections=totals['detections_raw'], post_nms_detections=n,
        duplicate_detections_removed=totals['detections_raw']-n,
        detections_per_processed_frame=n/len(frames) if frames else None,
        distributions={key: stats(o[key] for o in observations) for key in
                       ('face_width', 'face_height', 'face_area_ratio', 'detector_confidence')},
        face_width_buckets=buckets([o['face_width'] for o in observations]),
        face_height_buckets=buckets([o['face_height'] for o in observations]))
    reason_counts = Counter(reason if reason in REASONS else 'other'
                            for o in rejected for reason in o['quality']['reasons'])
    quality = dict(quality_total=len(measured), quality_pass=len(passed), quality_reject=len(rejected),
        unmeasured_detections=n-len(measured), quality_pass_rate=rate(len(passed), len(measured)),
        overlapping_reason_counts={r: reason_counts[r] for r in REASONS},
        reason_note='Reasons overlap; their sum can exceed rejected observations. Invalid measurements retain raw defaults but are excluded from numerical distributions.',
        distributions={})
    for label, rows in (('all', measured), ('pass', passed), ('reject', rejected)):
        valid = [o['quality_measurements'] for o in rows if o['quality_measurements']['valid']]
        quality['distributions'][label] = {key: stats(m[key] for m in valid) for key in
            ('width', 'height', 'area_ratio', 'sharpness', 'brightness', 'detector_confidence', 'nose_offset', 'roll_degrees')}
    lifetimes = [t['last_seen']-t['first_seen'] for t in tracks]
    tracking = dict(tracks_created=len(tracks),
        tracks_expired=sum(t['retirement_reason'] in ('ttl', 'max_misses') for t in tracks),
        tracks_retired=sum(not t['censored'] for t in tracks),
        maximum_active_tracks=max((f['active_track_count'] for f in frames), default=0),
        tracks_created_per_source_minute=60*len(tracks)/source_seconds if source_seconds else None,
        lifetime_seconds=stats(lifetimes), observations_per_track=stats(t['observations'] for t in tracks),
        single_observation_tracks=sum(t['observations'] == 1 for t in tracks),
        single_observation_track_rate=rate(sum(t['observations'] == 1 for t in tracks), len(tracks)),
        short_miss_recoveries=sum(t['short_miss_recoveries'] for t in tracks),
        ambiguous_associations=sum(f['ambiguous_associations'] for f in frames),
        ambiguity_resets=sum(t['retirement_reason'] == 'ambiguity' for t in tracks),
        active_tracks_censored=sum(t['censored'] for t in tracks),
        definition='Fragmentation diagnostics, not tracking accuracy. Lifetime is last observed minus first observed source time; active tracks right-censored.')
    gate_a = dict(matcher_calls=len(matches), production_observation_attempts=len(attempts),
        similarity_pass_count=sum(m['similarity_pass'] for m in matches),
        margin_pass_count=sum(m['margin_pass'] for m in matches),
        gate_a_pass_count=sum(m['gate_a_pass'] for m in matches),
        gate_a_fail_count=sum(not m['gate_a_pass'] for m in matches),
        pass_rate=rate(sum(m['gate_a_pass'] for m in matches), len(matches)),
        rejection_reasons=dict(Counter(m['gate_a_reason'] for m in matches if not m['gate_a_pass'])),
        production_pass_count=totals['gate_a_pass_count'],
        production_pass_rate=rate(totals['gate_a_pass_count'], len(attempts)))
    recognition = dict(
        top_k=dict(quality_passed_observations=len(passed), selected_observations=totals['topk_selected_count'],
            selection_rate=rate(totals['topk_selected_count'], len(passed)),
            temporal_refresh_count=sum(bool(o.get('selection', {}).get('forced_temporal_refresh')) for o in observations),
            replacements=sum(len(o.get('selection', {}).get('evicted_observation_ids', [])) for o in observations)),
        adaface=dict(calls=len(calls), inference_ms=stats(c['inference_time_ms'] for c in calls),
            dimensions=dict(Counter(str(c['embedding_dimension']) for c in calls)),
            embedding_norm=stats(c['embedding_norm'] for c in calls),
            finite_outputs=sum(c['embedding_finite'] is True for c in calls),
            calls_per_processed_frame=len(calls)/len(frames) if frames else None,
            calls_per_quality_pass=len(calls)/len(passed) if passed else None),
        matcher=dict(attempts=len(matches), distributions={key: stats(m[key] for m in matches)
                     for key in ('top1_score', 'top2_score', 'margin')},
                     score_semantics='Cosine similarity, not probability. Both observation and aggregate calls included.'),
        gate_a=gate_a,
        gate_b=dict(gate_a_passes_entering_temporal_state=totals['gate_a_pass_count'],
            candidate_tracks=len(candidate_tracks), confirmations=totals['gate_b_confirm_count'],
            confirmation_rate=rate(totals['gate_b_confirm_count'], len(candidate_tracks)),
            identity_switch_resets=sum(o['temporal']['identity_switch_reset'] for o in observations),
            duplicate_observations_ignored=sum(o['temporal']['duplicate_observation_ignored'] for o in observations),
            short_miss_state_preserved=sum(f['short_miss_state_preserved'] for f in frames)),
        confirmed_events=totals['confirmed_event_count'])
    performance = {key: stats(f[key] for f in frames) for key in (
        'detection_ms', 'tracking_ms', 'quality_ms', 'adaface_ms', 'matching_ms',
        'temporal_ms', 'recognition_coordinator_ms', 'total_recognition_pipeline_ms')}
    total_ms = sum(f['total_recognition_pipeline_ms'] for f in frames)
    performance['processed_samples_per_pipeline_second'] = 1000*len(frames)/total_ms if total_ms else None
    performance['timing_note'] = 'Per-frame stage totals include zero for unexecuted stages. Coordinator includes alignment, aggregation and instrumentation. Total includes detection + coordinator; excludes decode/JSON writing. Not live FPS.'
    funnel = dict(processed_frames=len(frames), post_nms_detections=n,
        quality_pass=totals['quality_pass_count'], topk_selected=totals['topk_selected_count'],
        adaface_calls=totals['adaface_inference_count'], gate_a_pass=totals['gate_a_pass_count'],
        gate_b_confirm=totals['gate_b_confirm_count'], confirmed_events=totals['confirmed_event_count'])
    funnel['observation_rates_of_detections'] = {key: rate(funnel[key], n) for key in
        ('quality_pass', 'topk_selected', 'adaface_calls', 'gate_a_pass')}
    funnel['conditional_rates'] = dict(quality_pass_per_detection=rate(len(passed), n),
        selected_per_quality_pass=rate(funnel['topk_selected'], len(passed)),
        adaface_per_selected=rate(len(calls), funnel['topk_selected']),
        gate_a_per_matcher_call=gate_a['pass_rate'],
        production_gate_a_per_observation_attempt=gate_a['production_pass_rate'],
        gate_b_per_candidate_track=recognition['gate_b']['confirmation_rate'])
    return dict(schema_version=SCHEMA_VERSION, labels_available=False, detection=detection,
                quality=quality, tracking=tracking, recognition=recognition, performance=performance, funnel=funnel)
