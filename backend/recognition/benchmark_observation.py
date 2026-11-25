"""Injectable static-observation evaluation, independent of model loading/IO."""
from collections import Counter
from dataclasses import asdict
import time
import cv2
import numpy as np
from recognition.detection_modes import detect_for_mode
from recognition.identity_matcher import match_identity
from recognition.quality import measure_quality, decide_quality
from recognition.offline_evaluation import MATCHER_DIAGNOSTIC, PRODUCTION_SIMULATION, predict, UNKNOWN


def evaluate_observation(frame, detector, model, align, gallery, config, detection_mode, evaluation_mode='both'):
    if evaluation_mode not in ('both', MATCHER_DIAGNOSTIC, PRODUCTION_SIMULATION):
        raise ValueError('Unsupported evaluation mode')
    row = dict(status='invalid_observation', detected=False, aligned=False, quality_passed=False,
        quality_reasons=[], match=None, quality=None, measurements=None, detection_count=0,
        inference_count=0, error_reason=None)
    stage = 'detection_failure'
    started = time.perf_counter()
    try:
        if frame is None:
            raise ValueError('Unreadable image')
        faces = detect_for_mode(detector, frame, detection_mode)
        row['detection_ms'] = (time.perf_counter()-started)*1000
        row['detection_count'], row['detected'] = len(faces), bool(faces)
        if len(faces) != 1:
            row['status'] = 'no_face' if not faces else 'multiple_faces'
            row['error_reason'] = row['status']
            return row
        face = faces[0]
        m = measure_quality(frame, face.bbox, face.kps, float(face.det_score), config.quality.crop_size)
        q = decide_quality(m, config.quality)
        row.update(measurements=asdict(m), quality=asdict(q), quality_passed=q.passed, quality_reasons=list(q.reasons))
        if not m.valid:
            row['error_reason'] = m.reason
            return row
        stage = 'alignment_failure'
        crop = align(image=frame, detected_face=face, output_size=112)
        if crop is None or crop.shape != (112, 112, 3):
            raise ValueError('Invalid aligned crop')
        row['aligned'] = True
        if evaluation_mode == PRODUCTION_SIMULATION and not q.passed:
            row['status'] = 'low_quality'
            return row
        stage = 'embedding_failure'
        begin = time.perf_counter()
        row['inference_count'] = 1
        embedding = model.extract_embedding(crop)
        row['embedding_ms'] = (time.perf_counter()-begin)*1000
        stage = 'matcher_failure'
        match = match_identity(embedding, gallery.identities, gallery.provenance['embedding_dimension'])
        row.update(status='evaluable', match=asdict(match),
            top1_user_id=match.top1_user_id, top1_score=match.top1_score, top1_pose=match.top1_winning_pose,
            top2_user_id=match.top2_user_id, top2_score=match.top2_score, top2_pose=match.top2_winning_pose,
            margin=match.margin, similarity_threshold=config.operating_point.similarity_threshold,
            margin_threshold=config.operating_point.margin_threshold)
        row['outcomes'] = {mode: ('accepted' if predict(row, config.operating_point, mode) != UNKNOWN else 'rejected')
                           for mode in (MATCHER_DIAGNOSTIC, PRODUCTION_SIMULATION)}
        return row
    except (ValueError, TypeError, AttributeError, RuntimeError, cv2.error) as error:
        row.update(error_reason=stage, error_type=type(error).__name__)
        return row


def quality_rejection_report(records):
    names = ('no_face', 'invalid_landmarks', 'alignment_failure', 'face_too_small',
             'blurred_face', 'poor_lighting', 'extreme_pose', 'low_detector_confidence', 'other')
    counts, combinations = Counter({name: 0 for name in names}), Counter()
    rows = [r for r in records if not r.get('excluded_reason')]
    for row in rows:
        reasons = set(row.get('quality_reasons') or [])
        if row.get('error_reason'):
            reasons.add(row['error_reason'])
        mapped = {reason if reason in counts else 'other' for reason in reasons}
        counts.update(mapped)
        if mapped:
            combinations[' + '.join(sorted(mapped))] += 1
    ranges = {}
    for name in ('width', 'height', 'area_ratio', 'detector_confidence', 'sharpness', 'brightness', 'nose_offset', 'roll_degrees'):
        values = [(r.get('measurements') or {}).get(name) for r in rows
                  if (r.get('measurements') or {}).get('valid') and not r.get('quality_passed')]
        values = [v for v in values if v is not None and np.isfinite(v)]
        ranges[name] = dict(n=len(values), min=float(min(values)), median=float(np.median(values)), max=float(max(values))) if values else dict(n=0)
    return dict(samples=len(rows), reasons=dict(counts), combinations=dict(combinations),
                rejected_valid_measurement_ranges=ranges,
                note='Reason counts overlap; other includes multi-face, unreadable and inference errors. No private images exported.')
