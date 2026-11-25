"""Phase 8A: bounded local video diagnostics using unchanged desktop algorithms."""
import argparse
from contextlib import ExitStack
from dataclasses import asdict
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import socket
import sys
import time
from types import SimpleNamespace
from unittest.mock import patch

BACKEND = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(BACKEND))


def sha256(path):
    with path.open('rb') as source:
        return hashlib.file_digest(source, 'sha256').hexdigest()


def local_video(value):
    path = Path(value)
    # Never allow an RTSP/HTTP source or network share into VideoCapture.
    if '://' in value or value.startswith(('\\\\', '//')) or not path.is_file():
        raise argparse.ArgumentTypeError('An existing local video file is required')
    return path


class JsonlOutputs:
    def __init__(self, directory, max_bytes):
        self.directory, self.max_bytes, self.bytes_written = directory, max_bytes, 0
        self.stack, self.files = ExitStack(), {}

    def __enter__(self):
        for name in ('frame_diagnostics', 'observation_diagnostics', 'track_diagnostics', 'confirmed_events'):
            self.files[name] = self.stack.enter_context((self.directory / (name+'.jsonl')).open('x', encoding='utf-8'))
        return self

    def write(self, name, row):
        line = json.dumps(dict(schema_version=1, **row), allow_nan=False)+'\n'
        size = len(line.encode('utf-8'))
        if self.bytes_written+size > self.max_bytes:
            raise RuntimeError('Diagnostic byte bound exceeded; reduce sample limit')
        self.files[name].write(line)
        self.bytes_written += size

    def __exit__(self, *args):
        return self.stack.__exit__(*args)


def write_json(path, value):
    path.write_text(json.dumps(value, indent=2, allow_nan=False), encoding='utf-8')


def run(args):
    import cv2
    import numpy as np
    from recognition.adaface_model import AdaFaceModel
    from recognition.adaface_loader import local_model_provenance
    from recognition.local_gallery import configured_gallery_provider
    from recognition.offline_runtime import require_local_detector
    from recognition.face_detector import get_shared_detector
    from recognition.face_preprocessing import align_detected_face
    from recognition.coordinator import RecognitionCoordinator
    from recognition.events import LocalAttendanceSink
    from recognition.offline_config import load_offline_config
    from recognition.detection_config import CLASSROOM_ROI, TILED_DETECTION_CONFIG
    from recognition.detection_modes import detect_for_mode, DESKTOP_EQUIVALENT_ROI_TILED
    from recognition.replay_diagnostics import ReplayDiagnostics
    from recognition.phase8a_probe import ReplayProbe
    from recognition.phase8a_diagnostics import summarize

    files = args.video or sorted((BACKEND / 'storage/videos').glob('*'))
    selected, seen = [], set()
    for path in files:
        if not path.is_file() or path.suffix.lower() not in {'.mp4', '.avi', '.mov', '.mkv'}:
            continue
        digest = sha256(path)
        if digest not in seen:
            seen.add(digest)
            selected.append((path, digest))
        if len(selected) >= args.max_videos:
            break
    if not selected:
        raise ValueError('No local videos found')
    args.output.mkdir(parents=True, exist_ok=False)
    protected_paths = sorted((BACKEND/'recognition').glob('*.py')) + [BACKEND/'recognition/offline_development.json', BACKEND/'storage/local_gallery/current.json']
    protected_hashes = {p.relative_to(BACKEND).as_posix(): sha256(p) for p in protected_paths}
    require_local_detector()
    gallery = configured_gallery_provider().load(local_model_provenance())
    model, detector, config = AdaFaceModel(), get_shared_detector(), load_offline_config()
    if model.provenance != gallery.provenance:
        raise ValueError('Model/gallery provenance mismatch')
    manifest = dict(schema_version=1, phase='8A', status='running', started_at_utc=datetime.now(timezone.utc).isoformat(),
        labels_available=False, gallery_generation=gallery.generation, model_provenance=model.provenance,
        configuration=asdict(config), classroom_roi=CLASSROOM_ROI, tiled_config=TILED_DETECTION_CONFIG,
        detection_mode=DESKTOP_EQUIVALENT_ROI_TILED, protected_sha256=protected_hashes,
        max_samples_per_video=args.max_samples, stride=args.stride, max_observation_records=args.max_records,
        max_jsonl_bytes=args.max_log_mb*1024*1024, videos=[],
        source_time_definition='source_frame_index / FPS; frame_sequence = source_frame_index + 1',
        pipeline='shared ROI+tiled detector/restoration/NMS -> tracker -> quality -> alignment -> Top-K -> cached AdaFace -> weighted aggregate -> observation and aggregate multi-prototype matching/Gate A -> agreeing fresh vote -> Gate B -> local event',
        limitations=['Unlabeled, sampled old video; no identity or tracking accuracy metrics.',
            'Desktop uses asynchronous latest-result scheduling; replay uses fixed source stride serially.',
            'Raw detections mean per-tile FaceAnalysis results before cross-tile NMS, not neural pre-NMS proposals.',
            'Quality sizes use clipped integer crops; detection sizes use restored floating bboxes.',
            'Observed timing includes small diagnostic overhead; no live throughput claim.'])
    write_json(args.output/'run_manifest.json', manifest)
    all_frames, all_observations, all_tracks, per_video = [], [], [], {}
    source_seconds = 0
    with JsonlOutputs(args.output, args.max_log_mb*1024*1024) as output:
        for index, (path, digest) in enumerate(selected):
            video_id = digest[:16]
            capture = cv2.VideoCapture(str(path.resolve()))
            if not capture.isOpened():
                raise ValueError('Cannot open local video')
            fps = capture.get(cv2.CAP_PROP_FPS)
            if not np.isfinite(fps) or fps <= 0:
                capture.release()
                raise ValueError('FPS unavailable; cannot invent source timeline')
            count = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
            info = dict(video_id=video_id, sha256=digest, filename=path.name,
                width=int(capture.get(cv2.CAP_PROP_FRAME_WIDTH)), height=int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT)),
                fps=fps, frame_count=count, duration_seconds=count/fps if count > 0 else None,
                stride=args.stride, sample_interval_seconds=args.stride/fps)
            sink = LocalAttendanceSink(session_id='phase8a-'+digest)
            engine = RecognitionCoordinator(model, align_detected_face, gallery, config, 'phase8a-'+digest, sink)
            track_diagnostics = ReplayDiagnostics(args.max_records)
            frames, observations, track_frames = [], [], {}
            frame_index, sampled_time = -1, 0
            started = time.perf_counter()
            try:
                with ReplayProbe(engine, detector) as probe:
                    while len(frames) < args.max_samples:
                        ok, frame = capture.read()
                        if not ok:
                            break
                        frame_index += 1
                        if frame_index % args.stride:
                            continue
                        sequence, sampled_time = frame_index+1, frame_index/fps
                        context = dict(video_id=video_id, frame_sequence=sequence, source_frame_index=frame_index,
                                       source_time_seconds=sampled_time, acquisition_time=sampled_time)
                        probe.begin_frame()
                        begin = time.perf_counter()
                        faces = detect_for_mode(detector, frame, DESKTOP_EQUIVALENT_ROI_TILED)
                        detection_ms = (time.perf_counter()-begin)*1000
                        if len(all_observations)+len(observations)+len(faces) > args.max_records:
                            raise RuntimeError('Observation bound exceeded; reduce samples')
                        inputs = [SimpleNamespace(bbox=f.bbox, landmarks=f.kps, detector_confidence=float(f.det_score)) for f in faces]
                        recognition_start = time.perf_counter()
                        decisions = engine.process(frame, inputs, sequence, sampled_time, digest)
                        recognition_ms = (time.perf_counter()-recognition_start)*1000
                        total_ms = (time.perf_counter()-begin)*1000
                        track_diagnostics.observe(engine, decisions)
                        preserved = probe.preserved_misses()
                        frame_row = dict(context, detections_raw=probe.raw_count, detections_after_nms=len(faces),
                            active_track_count=len(engine.tracker.tracks),
                            quality_pass_count=sum(bool(d.quality_decision and d.quality_decision.passed) for d in decisions),
                            quality_reject_count=sum(bool(d.quality_decision and not d.quality_decision.passed) for d in decisions),
                            topk_selected_count=sum(d.selected_new_observation for d in decisions),
                            adaface_inference_count=sum(len(d['adaface_calls']) for d in probe.details.values()),
                            gate_a_pass_count=sum(d.gate_a_passed for d in decisions),
                            gate_b_confirm_count=sum(d.confirmation_transition for d in decisions),
                            confirmed_event_count=sum(d.event_emitted for d in decisions),
                            ambiguous_associations=engine.tracker.last_update.get('ambiguous_detection_count', 0),
                            short_miss_state_preserved=len(preserved), short_miss_preserved_track_ids=preserved,
                            raw_tile_detection_counts=probe.tile_counts, detection_ms=detection_ms,
                            recognition_coordinator_ms=recognition_ms, total_recognition_pipeline_ms=total_ms,
                            processing_time_ms=total_ms,
                            **{key: probe.timings[key] for key in ('tracking_ms', 'quality_ms', 'adaface_ms', 'matching_ms', 'temporal_ms')})
                        frames.append(frame_row)
                        output.write('frame_diagnostics', frame_row)
                        for face, d in zip(faces, decisions):
                            detail = probe.details.get((d.track_id, sequence), {})
                            width, height = float(d.bbox[2]-d.bbox[0]), float(d.bbox[3]-d.bbox[1])
                            tile_id, tile_count = probe.face_tiles.get(id(face), (None, None))
                            confirmation = engine.confirmations.get(d.track_id)
                            row = dict(context, bbox=d.bbox, face_width=width, face_height=height,
                                face_area=width*height, face_area_ratio=width*height/(frame.shape[0]*frame.shape[1]),
                                detector_confidence=float(face.det_score), track_id=d.track_id,
                                tile_id=tile_id, raw_tile_detection_count=tile_count,
                                quality_measurements=asdict(d.quality_measurements) if d.quality_measurements else None,
                                quality=asdict(d.quality_decision) if d.quality_decision else None,
                                topk_selected=d.selected_new_observation, selection=detail.get('selection', {}),
                                adaface_calls=detail.get('adaface_calls', []), matcher_attempts=detail.get('matcher_attempts', []),
                                gate_a_passed=d.gate_a_passed, observation_reason=d.observation_reason or d.reason,
                                temporal=dict(candidate_user_id=confirmation.candidate if confirmation else None,
                                    distinct_observation_count=len(confirmation.votes) if confirmation else 0,
                                    confirmation_elapsed_seconds=confirmation.elapsed if confirmation else 0,
                                    confirmation_state='confirmed' if d.confirmed else ('pending' if confirmation and confirmation.candidate else 'no_candidate'),
                                    duplicate_observation_ignored=detail.get('duplicate_observation_ignored', False),
                                    short_miss_state_preserved=d.track_id in preserved,
                                    identity_switch_reset=detail.get('identity_switch_reset', False),
                                    gate_b_confirmed=d.confirmation_transition), error_type=d.error_type)
                            observations.append(row)
                            output.write('observation_diagnostics', row)
                            if d.track_id is not None:
                                track_frames.setdefault(d.track_id, [frame_index, frame_index])[1] = frame_index
                            if d.event_emitted:
                                output.write('confirmed_events', dict(context, track_id=d.track_id, user_id=d.match.top1_user_id,
                                    top1_score=d.match.top1_score, top2_score=d.match.top2_score, margin=d.match.margin,
                                    observation_count=d.valid_observations, confirmation_elapsed_seconds=d.elapsed,
                                    semantics='pipeline-confirmed identity; no ground truth'))
                        if len(frames) % 10 == 0:
                            print(f'Video {index+1}: {len(frames)} samples, {engine.inference_count} AdaFace calls', flush=True)
            finally:
                capture.release()
            _, tracks = track_diagnostics.summary(engine.tracker, sampled_time)
            for row in tracks:
                row.update(video_id=video_id, created_time=row['first_seen'], last_seen_time=row['last_seen'],
                    first_source_frame=track_frames[row['track_id']][0], last_source_frame=track_frames[row['track_id']][1],
                    retired=not row['censored'], ambiguity_events=int(row['retirement_reason'] == 'ambiguity'))
                output.write('track_diagnostics', row)
            summary = summarize(frames, observations, tracks, sampled_time)
            if summary['recognition']['adaface']['calls'] != engine.inference_count:
                raise AssertionError('Inference instrumentation mismatch')
            info.update(processed_samples=len(frames), sampled_source_span_seconds=sampled_time,
                elapsed_seconds=time.perf_counter()-started, maximum_evidence_bytes=engine.maximum_buffer_bytes,
                maximum_active_tracks=engine.maximum_tracks)
            manifest['videos'].append(info)
            per_video[video_id] = summary
            all_frames.extend(frames)
            all_observations.extend(observations)
            all_tracks.extend(tracks)
            source_seconds += sampled_time
        manifest['jsonl_bytes'] = output.bytes_written
    aggregate = summarize(all_frames, all_observations, all_tracks, source_seconds)
    for section in ('detection', 'quality', 'tracking', 'recognition', 'performance', 'funnel'):
        write_json(args.output/(section+'_summary.json'), dict(schema_version=1,
            aggregate=aggregate[section], videos={key: value[section] for key, value in per_video.items()}))
    previous_path = BACKEND/'storage/offline_evaluation/phase71-replay-20261009/replay_summary.json'
    comparison = []
    if previous_path.is_file():
        previous = json.loads(previous_path.read_text(encoding='utf-8'))
        for info in manifest['videos']:
            old = next((v for v in previous['videos'] if v['sha256'] == info['sha256']), None)
            if old:
                current = per_video[info['video_id']]
                comparison.append(dict(video_id=info['video_id'], same_stride=old['stride'] == args.stride,
                    same_sample_count=old['samples'] == info['processed_samples'],
                    old_tracks=old['tracking_diagnostics']['tracks_created'], current_tracks=current['tracking']['tracks_created'],
                    old_adaface_calls=old['adaface_inferences'], current_adaface_calls=current['recognition']['adaface']['calls'],
                    old_detections=old['tracking_diagnostics']['detections'], current_detections=current['detection']['post_nms_detections']))
    manifest.update(status='complete', previous_replay_comparison=comparison,
        protected_files_unchanged=all(sha256(BACKEND/key) == value for key, value in protected_hashes.items()))
    if not manifest['protected_files_unchanged']:
        raise AssertionError('Protected file changed during replay')
    write_json(args.output/'run_manifest.json', manifest)
    # Full machine-readable summaries accompany a compact human entry point.
    report = ['# Phase 8A diagnostic replay', '',
        'Unlabeled local replay. No tuning, calibration, cloud access or live-camera validation.', '',
        '## Actual pipeline', manifest['pipeline'], '', '## Aggregate funnel',
        '```json', json.dumps(aggregate['funnel'], indent=2), '```', '',
        '## Previous replay comparison', '```json', json.dumps(comparison, indent=2), '```', '',
        '## Interpretation limits', *['- '+item for item in manifest['limitations']], '',
        'All six *_summary.json files contain aggregate and per-video distributions; raw logs contain scalar metadata only.']
    (args.output/'report.md').write_text('\n'.join(report)+'\n', encoding='utf-8')
    print(json.dumps(dict(output=str(args.output), funnel=aggregate['funnel'], comparison=comparison), indent=2))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--video', type=local_video, action='append', help='Repeat for multiple local videos')
    parser.add_argument('--max-videos', type=int, default=2)
    parser.add_argument('--max-samples', type=int, default=60)
    parser.add_argument('--stride', type=int, default=5)
    parser.add_argument('--max-records', type=int, default=200000)
    parser.add_argument('--max-log-mb', type=int, default=128)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if min(args.max_videos, args.max_samples, args.stride, args.max_records, args.max_log_mb) < 1:
        parser.error('All bounds must be positive')
    with patch.object(socket.socket, 'connect', side_effect=RuntimeError('Network disabled for Phase 8A')), \
         patch.object(socket.socket, 'connect_ex', side_effect=RuntimeError('Network disabled for Phase 8A')):
        run(args)


if __name__ == '__main__':
    main()
