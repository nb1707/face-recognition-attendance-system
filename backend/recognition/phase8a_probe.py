"""Single-thread offline instrumentation. Delegates every algorithm unchanged.

Patches are scoped to a replay context, never installed in the desktop runtime.
Only scalar metadata is retained; no frames, crops or embeddings are logged.
"""
from collections import defaultdict
from contextlib import ExitStack
from dataclasses import asdict
import time
from unittest.mock import patch
import numpy as np


class ReplayProbe:
    def __init__(self, engine, detector):
        self.engine, self.detector = engine, detector
        self.stack = ExitStack()
        self.begin_frame()

    def begin_frame(self):
        self.timings = defaultdict(float)
        self.raw_count, self.tile_counts, self.face_tiles = 0, [], {}
        self.details, self.current_key = {}, None
        self.before_state = {k: (id(b), b.fingerprint, tuple(self.engine.confirmations[k].votes))
                             for k, b in self.engine.buffers.items()}

    def detail(self, key):
        return self.details.setdefault(key, dict(adaface_calls=[], matcher_attempts=[],
            selection={}, identity_switch_reset=False, duplicate_observation_ignored=False))

    def timed(self, name, function):
        def invoke(*args, **kwargs):
            start = time.perf_counter()
            try:
                return function(*args, **kwargs)
            finally:
                self.timings[name] += (time.perf_counter()-start)*1000
        return invoke

    def __enter__(self):
        from recognition import coordinator
        from recognition.evidence import BestFrameBuffer
        from recognition.temporal import TemporalConfirmation
        original_get = self.detector.face_app.get

        def detect(*args, **kwargs):
            faces = original_get(*args, **kwargs)
            tile_id = len(self.tile_counts)
            self.tile_counts.append(len(faces))
            self.raw_count += len(faces)
            for face in faces:
                self.face_tiles[id(face)] = (tile_id, len(faces))
            return faces

        self.stack.enter_context(patch.object(self.detector.face_app, 'get', detect))
        self.stack.enter_context(patch.object(self.engine.tracker, 'update',
            self.timed('tracking_ms', self.engine.tracker.update)))
        for name in ('measure_quality', 'decide_quality'):
            self.stack.enter_context(patch.object(coordinator, name,
                self.timed('quality_ms', getattr(coordinator, name))))
        original_offer = BestFrameBuffer.offer

        def offer(buffer, observation, quality_passed):
            self.current_key = (observation.track_id, observation.frame_sequence)
            detail = self.detail(self.current_key)
            before = list(buffer.observations)
            selected = original_offer(buffer, observation, quality_passed)
            retained = {o.frame_sequence for o in buffer.observations}
            # Explain the existing selection by comparing pre/post metadata only.
            eligible = [o for o in before if 0 <= observation.acquisition_time-o.acquisition_time <= buffer.config.max_age_seconds]
            ranking = sorted(eligible+[observation], key=lambda o: (-o.quality_score, -o.frame_sequence))[:buffer.config.top_k]
            forced = selected and all(o is not observation for o in ranking)
            detail['selection'] = dict(topk_selected=selected, forced_temporal_refresh=forced,
                reason='temporal_refresh' if forced else ('quality_rank' if selected else 'not_selected'),
                evicted_observation_ids=[o.frame_sequence for o in eligible if o.frame_sequence not in retained],
                expired_observation_ids=[o.frame_sequence for o in before if not any(o is item for item in eligible)])
            return selected

        self.stack.enter_context(patch.object(BestFrameBuffer, 'offer', offer))
        original_aggregate = BestFrameBuffer.aggregate

        def aggregate(buffer, extract):
            def measured_extract(crop):
                observation = next(o for o in buffer.observations if o.aligned_crop is crop and o.embedding is None)
                row = dict(track_id=observation.track_id, frame_sequence=observation.frame_sequence,
                    source_time_seconds=observation.acquisition_time, embedding_dimension=None,
                    embedding_norm=None, embedding_finite=None, error_type=None)
                start = time.perf_counter()
                try:
                    embedding = extract(crop)
                except Exception as error:
                    row['error_type'] = type(error).__name__
                    raise
                else:
                    elapsed = (time.perf_counter()-start)*1000
                    value = np.asarray(embedding)
                    row.update(embedding_dimension=int(value.size), embedding_finite=bool(np.isfinite(value).all()),
                        embedding_norm=float(np.linalg.norm(value)) if np.isfinite(value).all() else None)
                    return embedding
                finally:
                    row['inference_time_ms'] = elapsed if 'elapsed' in locals() else (time.perf_counter()-start)*1000
                    self.timings['adaface_ms'] += row['inference_time_ms']
                    self.detail(self.current_key)['adaface_calls'].append(row)
            return original_aggregate(buffer, measured_extract)

        self.stack.enter_context(patch.object(BestFrameBuffer, 'aggregate', aggregate))
        original_match, original_gate = coordinator.match_identity, coordinator.gate_a

        def match(*args, **kwargs):
            result = self.timed('matching_ms', original_match)(*args, **kwargs)
            rows = self.detail(self.current_key)['matcher_attempts']
            rows.append(dict(asdict(result), role='observation' if not rows else 'aggregate'))
            return result

        def gate(match, point):
            passed, reason = original_gate(match, point)
            self.detail(self.current_key)['matcher_attempts'][-1].update(
                similarity_threshold=point.similarity_threshold, margin_threshold=point.margin_threshold,
                similarity_pass=bool(match.top1_score is not None and np.isfinite(match.top1_score)
                                     and match.top1_score >= point.similarity_threshold),
                margin_pass=bool(match.top2_user_id is not None and match.top2_user_id != match.top1_user_id
                    and match.margin is not None and np.isfinite(match.margin) and match.margin >= point.margin_threshold),
                gate_a_pass=passed, gate_a_reason=reason)
            return passed, reason

        self.stack.enter_context(patch.object(coordinator, 'match_identity', match))
        self.stack.enter_context(patch.object(coordinator, 'gate_a', gate))
        original_reset, original_observe = TemporalConfirmation.reset, TemporalConfirmation.observe

        def reset(confirmation):
            if confirmation.candidate is not None and self.current_key is not None:
                self.detail(self.current_key)['identity_switch_reset'] = True
            return original_reset(confirmation)

        def observe(confirmation, user_id, passed, sequence, timestamp, fingerprint):
            duplicate = sequence <= confirmation.last_sequence or (
                user_id == confirmation.candidate and
                (fingerprint == confirmation.last_fingerprint or sequence not in fingerprint))
            self.detail(self.current_key)['duplicate_observation_ignored'] = duplicate
            return self.timed('temporal_ms', original_observe)(confirmation, user_id, passed, sequence, timestamp, fingerprint)

        self.stack.enter_context(patch.object(TemporalConfirmation, 'reset', reset))
        self.stack.enter_context(patch.object(TemporalConfirmation, 'observe', observe))
        return self

    def __exit__(self, *args):
        return self.stack.__exit__(*args)

    def preserved_misses(self):
        keys = self.engine.tracker.last_update['missed']
        preserved = []
        for key in keys:
            before = self.before_state.get(key)
            buffer = self.engine.buffers.get(key)
            if before and buffer is not None and before[0] == id(buffer):
                confirmation = self.engine.confirmations[key]
                if set(before[1]) & set(buffer.fingerprint) or set(before[2]) & set(confirmation.votes) or confirmation.confirmed:
                    preserved.append(key)
        return preserved
