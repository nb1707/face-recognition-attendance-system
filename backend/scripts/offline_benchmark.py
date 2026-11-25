"""Enrollment-aware static evaluation; two detector paths and two score views."""
import argparse
from collections import Counter
from dataclasses import asdict
import hashlib
import json
from pathlib import Path
import socket
import sys
import time
from unittest.mock import patch

BACKEND = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(BACKEND))
from recognition.offline_evaluation import (UNKNOWN, enrollment_category, metrics, view_status,
    EVALUATION_MODES, ENROLLED_KNOWN, UNENROLLED_IMPOSTOR)
from recognition.detection_modes import DETECTION_MODES


def discover(dataset):
    extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.webp'}
    for path in sorted(dataset.rglob('*')):
        if path.suffix.lower() not in extensions or not path.is_file():
            continue
        parts = path.relative_to(dataset).parts
        split = parts[0] if parts[0] in {'test', 'calibration'} else 'unspecified'
        label_index = 1 if split != 'unspecified' else 0
        if len(parts) <= label_index+1:
            yield path, None, 'unlabeled'
            continue
        label = parts[label_index]
        yield path, UNKNOWN if label.lower() == 'unknown' else label, split


def run(dataset, output, detection_mode='both', evaluation_mode='both'):
    import cv2
    from recognition.adaface_loader import local_model_provenance
    from recognition.adaface_model import AdaFaceModel
    from recognition.local_gallery import configured_gallery_provider
    from recognition.offline_runtime import require_local_detector
    from recognition.face_detector import get_shared_detector
    from recognition.face_preprocessing import align_detected_face
    from recognition.offline_config import load_offline_config
    from recognition.benchmark_observation import evaluate_observation, quality_rejection_report
    from recognition.detection_config import CLASSROOM_ROI, TILED_DETECTION_CONFIG
    require_local_detector()
    config = load_offline_config()
    provider = configured_gallery_provider()
    gallery = provider.load(local_model_provenance())
    manifest_path = provider.path
    manifest = json.loads(manifest_path.read_text(encoding='utf-8'))
    if 'manifest' in manifest:
        manifest = json.loads((manifest_path.parent / manifest['manifest']).read_text(encoding='utf-8'))
    registration_hashes = set(manifest.get('registration_sha256', []))
    model, detector = AdaFaceModel(), get_shared_detector()
    if model.provenance != gallery.provenance:
        raise ValueError('Checkpoint changed between gallery and model loading')
    available = {identity.user_id for identity in gallery.identities}
    output.mkdir(parents=True, exist_ok=False)
    samples, seen = [], {}
    for path, expected, split in discover(dataset):
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
        row = dict(dataset_path=str(path.resolve()), sample=str(path.relative_to(dataset)), expected_user_id=expected,
            declared_split=split, sha256=digest, gallery_identity_available=expected in available,
            enrollment_category=enrollment_category(expected, available), excluded_reason=None,
            gallery_generation=gallery.generation)
        if expected is None:
            row['excluded_reason'] = 'unlabeled_image'
        elif digest in registration_hashes:
            row['excluded_reason'] = 'exact_registration_overlap'
        elif digest in seen:
            row['excluded_reason'], row['duplicate_of'] = 'duplicate_image', seen[digest]
        if expected is not None:
            seen.setdefault(digest, row['sample'])
        samples.append((path, row))
    summaries = {}
    modes = DETECTION_MODES if detection_mode == 'both' else (detection_mode,)
    for mode in modes:
        target_dir = output / mode
        target_dir.mkdir()
        records, started = [], time.perf_counter()
        with (target_dir / 'raw_scores.jsonl').open('w', encoding='utf-8') as target:
            for index, (path, base) in enumerate(samples, 1):
                row = dict(base, detection_mode=mode, collection_mode=evaluation_mode, status='excluded')
                begin = time.perf_counter()
                if not row['excluded_reason']:
                    row.update(evaluate_observation(cv2.imread(str(path)), detector, model, align_detected_face,
                                                    gallery, config, mode, evaluation_mode))
                row['processing_ms'] = (time.perf_counter()-begin)*1000
                records.append(row)
                target.write(json.dumps(row, allow_nan=False)+'\n')
                target.flush()
                if index % 40 == 0:
                    print(f'{mode}: {index}/{len(samples)} images', flush=True)
        eligible = [r for r in records if not r['excluded_reason']]
        selected_views = EVALUATION_MODES if evaluation_mode == 'both' else (evaluation_mode,)
        views = {view: dict(end_to_end=metrics(records, config.operating_point, mode=view),
            recognition_conditional=metrics(records, config.operating_point, conditional=True, mode=view),
            matcher_evaluable=sum(view_status(r, view) == 'evaluable' for r in eligible)) for view in selected_views}
        summary = dict(dataset=str(dataset.resolve()), generation=gallery.generation, provenance=gallery.provenance,
            configuration=asdict(config), detector_mode=mode,
            detection_configuration=dict(roi=CLASSROOM_ROI, tiled=TILED_DETECTION_CONFIG),
            discovered=len(records), included=len(eligible),
            enrollment_counts=dict(Counter(r['enrollment_category'] for r in eligible)),
            discovered_enrollment_counts=dict(Counter(r['enrollment_category'] or 'unlabeled' for r in records)),
            enrolled_identities=sorted({r['expected_user_id'] for r in eligible if r['enrollment_category'] == ENROLLED_KNOWN}),
            unenrolled_labels=sorted({r['expected_user_id'] for r in eligible if r['enrollment_category'] == UNENROLLED_IMPOSTOR}),
            excluded=dict(Counter(r['excluded_reason'] for r in records if r['excluded_reason'])),
            no_face=sum(r['status'] == 'no_face' for r in eligible),
            multi_face=sum(r['status'] == 'multiple_faces' for r in eligible),
            valid_alignment=sum(bool(r.get('aligned')) for r in eligible),
            quality_pass=sum(bool(r.get('quality_passed')) for r in eligible),
            quality_rejected=sum(bool(r.get('quality_reasons')) for r in eligible),
            quality_report=quality_rejection_report(records), views=views,
            inference_count=sum(r.get('inference_count', 0) for r in records), elapsed_seconds=time.perf_counter()-started,
            PRODUCTION_OPERATING_POINT_SELECTED='NO', calibration_status='INSUFFICIENT FOR PRODUCTION CALIBRATION',
            limitations='Static Gate A only; no attendance. Matcher diagnostic bypasses quality decisions, not invalid landmarks/alignment. Multiple faces rejected without inventing a target. No independent capture/session provenance; exact hashes cannot detect cropped/re-encoded/adjacent-frame leakage.')
        (target_dir / 'benchmark_summary.json').write_text(json.dumps(summary, indent=2), encoding='utf-8')
        summaries[mode] = summary
        print(json.dumps(dict(mode=mode, counts=summary['enrollment_counts'], views=views)), flush=True)
    (output / 'comparison.json').write_text(json.dumps(summaries, indent=2), encoding='utf-8')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--dataset', type=Path, default=BACKEND / 'evaluation_dataset')
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--detection-mode', choices=(*DETECTION_MODES, 'both'), default='both')
    parser.add_argument('--evaluation-mode', choices=(*EVALUATION_MODES, 'both'), default='both')
    args = parser.parse_args()
    with patch.object(socket.socket, 'connect', side_effect=RuntimeError('Network disabled for offline benchmark')):
        run(args.dataset, args.output, args.detection_mode, args.evaluation_mode)
