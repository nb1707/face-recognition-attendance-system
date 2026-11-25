"""Local video replay through the same coordinator as desktop. No network."""
import argparse
from collections import Counter
from dataclasses import asdict, replace
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


def apply_replay_quality_override(config, min_size=None, min_sharpness=None):
    """Return replay-only quality settings; production/development JSON is not modified."""
    quality = replace(
        config.quality,
        min_size=(config.quality.min_size if min_size is None else float(min_size)),
        min_sharpness=(config.quality.min_sharpness
                       if min_sharpness is None else float(min_sharpness)),
    )
    return replace(config, quality=quality)


def file_sha256(path, chunk_size=1024 * 1024):
    digest = hashlib.sha256()
    with path.open('rb') as source:
        while chunk := source.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


def iter_sampled_frames(capture, stride, max_samples, full_video=False):
    """Read one frame at a time; in full mode EOF is the only sample limit."""
    source_index = -1
    samples = 0
    while True:
        if not full_video and samples >= max_samples:
            break
        ok, frame = capture.read()
        if not ok:
            break
        source_index += 1
        if source_index % stride:
            continue
        samples += 1
        yield source_index + 1, source_index, frame


def shadow_match_observation(frame, face, model, align, identities, dimension, matcher):
    """Stateless offline-only evaluation; has no coordinator/sink argument."""
    result = dict(evaluated=False, embedding_generated=False, alignment_succeeded=False,
                  error_type=None, top1_user_id=None, top1_score=None, top1_pose=None,
                  top2_user_id=None, top2_score=None, top2_pose=None, margin=None)
    try:
        crop = align(image=frame, detected_face=SimpleNamespace(bbox=face.bbox, kps=face.kps),
                     output_size=112)
        result['alignment_succeeded'] = True
        embedding = model.extract_embedding(crop)
        result['embedding_generated'] = True
        match = matcher(embedding, identities, dimension)
        result.update(evaluated=True, top1_user_id=match.top1_user_id,
            top1_score=match.top1_score, top1_pose=match.top1_winning_pose,
            top2_user_id=match.top2_user_id, top2_score=match.top2_score,
            top2_pose=match.top2_winning_pose, margin=match.margin)
    except (ValueError, TypeError, RuntimeError) as error:
        result['error_type'] = type(error).__name__
    return result


def _quality_reasons(decision, production_alignment_succeeded):
    if decision.quality_decision is None:
        return ['other']
    reasons = list(decision.quality_decision.reasons)
    measurement = decision.quality_measurements
    if measurement is not None and not measurement.valid:
        reason = measurement.reason or 'other'
        reasons = ['invalid_landmarks' if 'landmark' in reason else 'other']
    elif decision.quality_decision.passed and not production_alignment_succeeded:
        reasons = ['alignment_failure']
    return reasons


def observation_row(video_name, sequence, source_index, timestamp, face, decision,
                    detail, engine, detection_ms, recognition_ms, point, shadow=None):
    measurement = decision.quality_measurements
    quality = decision.quality_decision
    production_calls = detail.get('adaface_calls', [])
    matches = detail.get('matcher_attempts', [])
    match = next((row for row in reversed(matches) if row.get('role') == 'aggregate'), None)
    if match is None and decision.aggregate_match is not None:
        match = asdict(decision.aggregate_match)
    if match is None and decision.observation_match is not None:
        match = asdict(decision.observation_match)
    match = match or {}
    production_alignment_succeeded = bool(detail.get('selection'))
    shadow = shadow or dict(evaluated=False, embedding_generated=False,
                            alignment_succeeded=False, error_type=None)
    confirmation = engine.confirmations.get(decision.track_id)
    width = float(decision.bbox[2] - decision.bbox[0])
    height = float(decision.bbox[3] - decision.bbox[1])
    return dict(source_video=video_name, frame_sequence=sequence,
        source_frame_index=source_index, acquisition_time=timestamp,
        track_id=decision.track_id, bbox=list(decision.bbox), face_width=width,
        face_height=height,
        area_ratio=(measurement.area_ratio if measurement and measurement.valid else None),
        detector_confidence=float(face.det_score),
        sharpness=measurement.sharpness if measurement and measurement.valid else None,
        brightness=measurement.brightness if measurement and measurement.valid else None,
        nose_offset=measurement.nose_offset if measurement and measurement.valid else None,
        roll_degrees=measurement.roll_degrees if measurement and measurement.valid else None,
        valid_observation=bool(measurement and measurement.valid),
        production_alignment_succeeded=production_alignment_succeeded,
        shadow_alignment_succeeded=bool(shadow.get('alignment_succeeded')),
        alignment_succeeded=production_alignment_succeeded or bool(shadow.get('alignment_succeeded')),
        quality_score=quality.quality_score if quality else None,
        quality_passed=bool(quality and quality.passed),
        quality_reasons=_quality_reasons(decision, production_alignment_succeeded),
        top_k_selected=bool(decision.selected_new_observation),
        embedding_generated=bool(production_calls),
        embedding_reused=bool(decision.selected_new_observation and not production_calls and not decision.error_type),
        production_adaface_call_count=len(production_calls),
        top1_user_id=match.get('top1_user_id'), top1_score=match.get('top1_score'),
        top1_pose=match.get('top1_winning_pose'), top2_user_id=match.get('top2_user_id'),
        top2_score=match.get('top2_score'), top2_pose=match.get('top2_winning_pose'),
        margin=match.get('margin'), similarity_threshold=point.similarity_threshold,
        margin_threshold=point.margin_threshold, gate_a_passed=bool(decision.gate_a_passed),
        gate_a_reason=decision.observation_reason or decision.reason,
        temporal_status=('confirmed' if decision.confirmed else
                         ('pending' if confirmation and confirmation.candidate else 'no_candidate')),
        temporal_candidate_user_id=confirmation.candidate if confirmation else None,
        temporal_observation_count=len(confirmation.votes) if confirmation else 0,
        gate_b_confirmed=bool(decision.confirmation_transition),
        confirmed_user_id=decision.match.top1_user_id if decision.confirmed else None,
        confirmed_event_emitted=bool(decision.event_emitted),
        production_quality_passed=bool(quality and quality.passed),
        shadow_match_evaluated=bool(shadow.get('evaluated')),
        shadow_embedding_generated=bool(shadow.get('embedding_generated')),
        shadow_error_type=shadow.get('error_type'),
        shadow_top1_user_id=shadow.get('top1_user_id'),
        shadow_top1_score=shadow.get('top1_score'), shadow_top1_pose=shadow.get('top1_pose'),
        shadow_top2_user_id=shadow.get('top2_user_id'),
        shadow_top2_score=shadow.get('top2_score'), shadow_top2_pose=shadow.get('top2_pose'),
        shadow_margin=shadow.get('margin'), detection_time_ms=detection_ms,
        recognition_time_ms=recognition_ms)


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
    from recognition.detection_modes import detect_for_mode, DESKTOP_EQUIVALENT_ROI_TILED
    from recognition.identity_matcher import match_identity
    from recognition.replay_diagnostics import ReplayDiagnostics
    from recognition.replay_streaming import BoundedStatistics, ReplayFunnel
    from recognition.phase8a_probe import ReplayProbe

    require_local_detector()
    gallery = configured_gallery_provider().load(local_model_provenance())
    if (args.expected_gallery_generation is not None and
            gallery.generation != args.expected_gallery_generation):
        raise RuntimeError(
            f'Gallery generation mismatch: expected {args.expected_gallery_generation!r}, '
            f'actual {gallery.generation!r}')
    model, detector = AdaFaceModel(), get_shared_detector()
    config = load_offline_config()
    config = apply_replay_quality_override(
        config, args.quality_min_size, args.quality_min_sharpness)
    if model.provenance != gallery.provenance:
        raise ValueError('Checkpoint changed between gallery and model loading')
    files = [args.video] if args.video else sorted((BACKEND / 'storage/videos').glob('*'))
    selected, hashes = [], set()
    for path in files:
        if not path.is_file() or path.suffix.lower() not in {'.mp4', '.avi', '.mov', '.mkv'}:
            continue
        digest = file_sha256(path)
        if digest not in hashes:
            selected.append((path, digest))
            hashes.add(digest)
        if len(selected) >= args.max_videos:
            break
    if not selected:
        raise ValueError('No local videos found')
    args.output.mkdir(parents=True, exist_ok=False)
    summaries = []
    for video_index, (path, digest) in enumerate(selected):
        capture = cv2.VideoCapture(str(path.resolve()))
        if not capture.isOpened():
            raise ValueError('Local video could not be opened')
        fps = capture.get(cv2.CAP_PROP_FPS)
        if not np.isfinite(fps) or fps <= 0:
            capture.release()
            raise ValueError('Video FPS unavailable; cannot invent acquisition timestamps')
        events_path = args.output / f'events-{video_index}.jsonl'
        events_path.touch(exist_ok=False)
        sink = LocalAttendanceSink(events_path, session_id=f'offline-replay-{digest}')
        engine = RecognitionCoordinator(model, align_detected_face, gallery, config,
                                        f'offline-replay-{digest}', sink)
        reasons = Counter()
        funnel = ReplayFunnel(args.diagnostic_sample_cap)
        timings = BoundedStatistics(args.diagnostic_sample_cap)
        first_confirm_time = None
        last_timestamp = 0.0
        started = time.perf_counter()
        frames_path = args.output / f'frames-{video_index}.jsonl'
        observations_path = args.output / f'observations-{video_index}.jsonl'
        tracks_path = args.output / f'tracks-{video_index}.jsonl'
        try:
            with frames_path.open('w', encoding='utf-8') as frame_target, \
                 observations_path.open('w', encoding='utf-8') as observation_target, \
                 tracks_path.open('w', encoding='utf-8') as track_target:
                def write_track(row):
                    track_target.write(json.dumps(row, allow_nan=False) + '\n')

                diagnostics = ReplayDiagnostics(args.diagnostic_sample_cap, row_sink=write_track)
                with ReplayProbe(engine, detector) as probe:
                    for sequence, source_index, frame in iter_sampled_frames(
                            capture, args.stride, args.max_samples, args.full_video):
                        timestamp = source_index / fps
                        last_timestamp = timestamp
                        probe.begin_frame()
                        detection_start = time.perf_counter()
                        faces = detect_for_mode(detector, frame, DESKTOP_EQUIVALENT_ROI_TILED)
                        detection_ms = (time.perf_counter() - detection_start) * 1000
                        inputs = [SimpleNamespace(bbox=f.bbox, landmarks=f.kps,
                                  detector_confidence=float(f.det_score)) for f in faces]
                        recognition_start = time.perf_counter()
                        decisions = engine.process(frame, inputs, sequence, timestamp, digest)
                        recognition_ms = (time.perf_counter() - recognition_start) * 1000
                        processing_ms = detection_ms + recognition_ms
                        timings.add(processing_ms)
                        funnel.add_frame()
                        diagnostics.observe(engine, decisions)
                        if not decisions:
                            reasons['no_face'] += 1
                        for face, decision in zip(faces, decisions):
                            detail = probe.details.get((decision.track_id, sequence), {})
                            shadow = None
                            quality = decision.quality_decision
                            measurement = decision.quality_measurements
                            if (args.shadow_matcher and quality and not quality.passed and
                                    measurement and measurement.valid):
                                shadow = shadow_match_observation(frame, face, model,
                                    align_detected_face, gallery.identities,
                                    gallery.provenance['embedding_dimension'], match_identity)
                            row = observation_row(path.name, sequence, source_index, timestamp,
                                face, decision, detail, engine, detection_ms, recognition_ms,
                                config.operating_point, shadow)
                            observation_target.write(json.dumps(row, allow_nan=False) + '\n')
                            funnel.add_observation(row)
                            reasons[decision.observation_reason or decision.reason] += 1
                            if decision.confirmed and first_confirm_time is None:
                                first_confirm_time = timestamp
                        frame_target.write(json.dumps(dict(sequence=sequence,
                            source_frame_index=source_index, acquisition_time=timestamp,
                            detection_ms=detection_ms, recognition_ms=recognition_ms,
                            processing_ms=processing_ms, decisions=[asdict(d) for d in decisions]),
                            allow_nan=False) + '\n')
                        samples = funnel.counts['sampled_frames']
                        if samples % 10 == 0:
                            print(f'Video {video_index+1}: {samples} samples; '
                                  f'{engine.inference_count} production inferences; '
                                  f'{funnel.counts["shadow_adaface_inferences"]} shadow inferences; '
                                  f'{len(sink.events)} attendance events', flush=True)
                tracking_summary, _ = diagnostics.summary(engine.tracker, last_timestamp)
        finally:
            capture.release()
        funnel_summary = funnel.summary()
        timing_summary = timings.summary()
        if funnel_summary['adaface_inferences'] != engine.inference_count:
            raise AssertionError('Production AdaFace inference diagnostics diverged from coordinator count')
        summaries.append(dict(video=path.name, sha256=digest, labels_available=False,
            source_fps=fps, samples=funnel_summary['sampled_frames'], stride=args.stride,
            full_video=bool(args.full_video), stopped_at='eof' if args.full_video else 'max_samples_or_eof',
            source_duration_sampled=last_timestamp, status_counts=dict(reasons),
            unique_tracks=tracking_summary['tracks_created'],
            tracking_diagnostics=tracking_summary, funnel=funnel_summary,
            maximum_observations_per_track=config.evidence.top_k,
            maximum_embeddings_per_track=config.evidence.top_k,
            buffer_window_seconds=config.evidence.max_age_seconds,
            track_ttl_seconds=config.tracker.ttl_seconds,
            maximum_active_tracks=engine.maximum_tracks,
            maximum_evidence_bytes=engine.maximum_buffer_bytes,
            adaface_inferences=engine.inference_count,
            shadow_adaface_inferences=funnel_summary['shadow_adaface_inferences'],
            attendance_events=len(sink.events), first_confirm_source_time=first_confirm_time,
            processing_ms=timing_summary,
            processing_ms_mean=timing_summary['mean'], processing_ms_p95=timing_summary['p95'],
            elapsed_seconds=time.perf_counter()-started,
            output_files=dict(frames=frames_path.name, observations=observations_path.name,
                              tracks=tracks_path.name, events=f'events-{video_index}.jsonl')))
    result = dict(phase='8B.3', gallery_generation=gallery.generation,
        quality_candidate=dict(min_size=config.quality.min_size,
                               min_sharpness=config.quality.min_sharpness),
        configuration=asdict(config),
        detector_mode='existing tiled detector', full_video=bool(args.full_video),
        shadow_matcher=bool(args.shadow_matcher), diagnostic_sample_cap=args.diagnostic_sample_cap,
        memory_bounds=dict(decoded_frames_in_memory=1,
            tracker_active_tracks=config.tracker.max_tracks,
            evidence_observations_per_track=config.evidence.top_k,
            track_statistics_reservoir=args.diagnostic_sample_cap,
            numeric_statistics_reservoir_per_metric=args.diagnostic_sample_cap,
            raw_metadata='streamed incrementally to JSONL; no frames, crops, or embeddings serialized'),
        videos=summaries,
        limitations=('Unlabeled replay. No identity accuracy, FAR/FRR, true track-switch rate, '
                     'realtime throughput or live-camera validation claim. Timestamps use source '
                     'frame index/FPS. Shadow matcher is diagnostic only and cannot vote or emit events.'))
    (args.output / 'replay_summary.json').write_text(json.dumps(result, indent=2, allow_nan=False), encoding='utf-8')
    print(json.dumps(result, indent=2, allow_nan=False))


def build_parser():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--video', type=Path)
    parser.add_argument('--max-videos', type=int, default=2)
    parser.add_argument('--max-samples', type=int, default=60,
                        help='sample limit per video unless --full-video is used (default: 60)')
    parser.add_argument('--full-video', action='store_true',
                        help='process sampled frames until EOF; --stride still applies')
    parser.add_argument('--shadow-matcher', action='store_true',
                        help='offline-only matching of valid quality-rejected faces; never votes or emits events')
    parser.add_argument('--stride', type=int, default=5)
    parser.add_argument('--diagnostic-sample-cap', type=int, default=4096,
                        help='fixed in-memory reservoir capacity for descriptive statistics')
    parser.add_argument('--quality-min-size', type=float, default=None,
                        help='offline replay-only override for QualityConfig.min_size')
    parser.add_argument('--quality-min-sharpness', type=float, default=None,
                        help='offline replay-only override for QualityConfig.min_sharpness')
    parser.add_argument('--expected-gallery-generation', type=str, default=None,
                        help='fail closed if the loaded gallery generation differs')
    parser.add_argument('--output', type=Path, required=True)
    return parser


def main(argv=None):
    parser = build_parser()
    args = parser.parse_args(argv)
    if min(args.max_videos, args.max_samples, args.stride, args.diagnostic_sample_cap) < 1:
        parser.error('Video/sample/stride/diagnostic bounds must be positive')
    if args.quality_min_size is not None and not args.quality_min_size > 0:
        parser.error('--quality-min-size must be > 0')
    if args.quality_min_sharpness is not None and not args.quality_min_sharpness >= 0:
        parser.error('--quality-min-sharpness must be >= 0')
    with patch.object(socket.socket, 'connect', side_effect=RuntimeError('Network disabled for offline replay')), \
         patch.object(socket.socket, 'connect_ex', side_effect=RuntimeError('Network disabled for offline replay')):
        run(args)


if __name__ == '__main__':
    main()
