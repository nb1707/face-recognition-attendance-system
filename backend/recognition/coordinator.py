"""Thread-owned recognition state; tracking lifetime is independent of voting."""
from dataclasses import dataclass, field, replace
from types import SimpleNamespace
import numpy as np
from recognition.evidence import BestFrameBuffer, Observation
from recognition.events import ConfirmedIdentityEvent, LocalAttendanceSink
from recognition.identity_matcher import MatchResult, match_identity
from recognition.quality import measure_quality, decide_quality
from recognition.temporal import TemporalConfirmation, gate_a
from recognition.tracker import FaceTracker


@dataclass
class FaceDecision:
    bbox: tuple
    track_id: int | None
    reason: str
    match: MatchResult = field(default_factory=MatchResult)
    quality_measurements: object = None
    quality_decision: object = None
    gate_a_passed: bool = False
    confirmed: bool = False
    valid_observations: int = 0
    selected_frames: int = 0
    elapsed: float = 0
    error_type: str | None = None
    observation_reason: str | None = None
    observation_match: MatchResult | None = None
    aggregate_match: MatchResult | None = None
    aggregate_match: MatchResult | None = None
    selected_new_observation: bool = False
    confirmation_transition: bool = False
    event_emitted: bool = False
    state_from_history: bool = False
    confirmed_since: float | None = None


class RecognitionCoordinator:
    """One session/stream. Confirmed snapshots survive weak/missing evidence.

    Only a fresh observation passing Gate A for a DIFFERENT identity contradicts
    the current candidate. It clears old vectors/votes/state before rebuilding.
    Missing/poor/weak observations never vote and do not contradict identity.
    Track retirement, stream/session/model/gallery reset remove all track state.
    """
    def __init__(self, model, align, gallery, config, session_id, sink=None):
        self.model, self.align, self.gallery, self.config = model, align, gallery, config
        self.session_id = session_id
        self.sink = sink if sink is not None else LocalAttendanceSink()
        self.tracker = FaceTracker(config.tracker)
        self.buffers, self.confirmations, self.confirmed_states = {}, {}, {}
        self.stream_id, self.gallery_token = None, None
        self.inference_count = self.maximum_tracks = self.maximum_buffer_bytes = 0
        self.last_reason = 'no_face'
        self.last_frame_key, self.last_results = None, ()

    def reset(self):
        self.tracker.reset()
        self.buffers.clear()
        self.confirmations.clear()
        self.confirmed_states.clear()
        self.last_frame_key, self.last_results = None, ()

    def _restore_confirmed(self, result):
        state = self.confirmed_states.get(result.track_id)
        if state is not None:
            result.match = state.match
            result.confirmed, result.reason = True, 'confirmed_identity'
            result.valid_observations = state.valid_observations
            result.selected_frames, result.elapsed = state.selected_frames, state.elapsed
            result.confirmed_since = state.confirmed_since
            result.state_from_history = not result.confirmation_transition

    def process(self, frame, faces, sequence, acquisition_time, stream_id):
        provenance = getattr(self.model, 'provenance', self.gallery.provenance)
        if provenance != self.gallery.provenance:
            self.reset()
            raise ValueError('Model/gallery provenance mismatch')
        token = (self.gallery.generation, tuple(sorted(provenance.items())), id(self.model), self.session_id)
        if stream_id != self.stream_id or token != self.gallery_token:
            self.reset()
            self.stream_id, self.gallery_token = stream_id, token
        boxes = [tuple(float(v) for v in face.bbox) for face in faces]
        frame_key = (sequence, tuple(boxes))  # Replayed frame IDs never gain a new acquisition time/vote.
        if frame_key == self.last_frame_key:
            return [replace(r, confirmation_transition=False, event_emitted=False,
                            selected_new_observation=False, gate_a_passed=False,
                            state_from_history=r.confirmed, observation_reason='duplicate_frame') for r in self.last_results]
        try:
            ids = self.tracker.update(boxes, sequence, acquisition_time)
        except ValueError:
            return [FaceDecision(box, None, 'invalid_observation') for box in boxes]
        active = set(self.tracker.tracks)
        for key in list(self.buffers):
            if key not in active:
                self.buffers.pop(key, None)
                self.confirmations.pop(key, None)
                self.confirmed_states.pop(key, None)
            else:
                self.buffers[key].expire(acquisition_time)
                confirmation = self.confirmations[key]
                confirmation.votes = [(seq, t) for seq, t in confirmation.votes
                    if 0 <= acquisition_time-t <= self.config.temporal.evidence_window_seconds]
        dimension = self.gallery.provenance['embedding_dimension']
        results = []
        for face, box, key in zip(faces, boxes, ids):
            result = FaceDecision(box, key, 'invalid_observation')
            results.append(result)
            if key is None:
                continue
            measurements = measure_quality(frame, box, face.landmarks, face.detector_confidence, self.config.quality.crop_size)
            quality = decide_quality(measurements, self.config.quality)
            result.quality_measurements, result.quality_decision = measurements, quality
            if not quality.passed:
                result.observation_reason = result.reason = 'low_quality' if measurements.valid else 'invalid_observation'
                self._restore_confirmed(result)
                continue
            if key not in self.buffers:
                self.buffers[key] = BestFrameBuffer(key, self.gallery.generation, dimension, self.config.evidence)
                self.confirmations[key] = TemporalConfirmation(self.config.temporal)
            buffer, confirmation = self.buffers[key], self.confirmations[key]
            try:
                crop = self.align(image=frame, detected_face=SimpleNamespace(bbox=np.array(box), kps=np.array(face.landmarks)), output_size=112)
                observation = Observation(key, sequence, acquisition_time, quality.quality_score, crop,
                                          box, tuple(face.landmarks), measurements)
                result.selected_new_observation = buffer.offer(observation, True)
                result.selected_frames = len(buffer.observations)
                if not result.selected_new_observation:
                    result.reason = result.observation_reason = 'temporal_pending'
                    result.valid_observations, result.elapsed = len(confirmation.votes), confirmation.elapsed
                    self._restore_confirmed(result)
                    continue
                before = buffer.inference_count
                try:
                    aggregate = buffer.aggregate(self.model.extract_embedding)
                finally:
                    self.inference_count += buffer.inference_count-before
                self.maximum_buffer_bytes = max(self.maximum_buffer_bytes, sum(b.memory_bytes for b in self.buffers.values()))
                current = match_identity(observation.embedding, self.gallery.identities, dimension)
                result.observation_match = current
                current_passed, current_reason = gate_a(current, self.config.operating_point)
                if current_passed and confirmation.candidate is not None and current.top1_user_id != confirmation.candidate:
                    buffer.observations = [observation]
                    confirmation.reset()
                    self.confirmed_states.pop(key, None)
                    aggregate = observation.embedding
                result.match = match_identity(aggregate, self.gallery.identities, dimension)
                result.aggregate_match = result.match
                result.aggregate_match = result.match
                passed, reason = gate_a(result.match, self.config.operating_point)
                # Every vote needs fresh recognized evidence of this identity.
                passed = passed and current_passed and current.top1_user_id == result.match.top1_user_id
                if not current_passed:
                    reason = current_reason
                    buffer.observations = [o for o in buffer.observations if o is not observation]
                result.gate_a_passed = passed
                was_confirmed = key in self.confirmed_states
                if passed:
                    result.confirmed = confirmation.observe(result.match.top1_user_id, True, sequence,
                                                            acquisition_time, buffer.fingerprint)
                result.reason = result.observation_reason = reason
                result.valid_observations = len(confirmation.votes)
                result.selected_frames, result.elapsed = len(buffer.observations), confirmation.elapsed
                if result.confirmed and not was_confirmed:
                    result.confirmation_transition = True
                    result.confirmed_since = confirmation.confirmed_since
                    result.reason = 'confirmed_identity'
                    self.confirmed_states[key] = replace(result)
                    m = result.match
                    result.event_emitted = bool(self.sink.accept(ConfirmedIdentityEvent(self.session_id,
                        self.gallery.generation, key, m.top1_user_id, m.top1_score, m.top2_score, m.margin,
                        len(confirmation.votes), len(confirmation.votes), confirmation.elapsed,
                        len(buffer.observations), acquisition_time, sequence)))
                self._restore_confirmed(result)
            except (ValueError, TypeError, RuntimeError) as error:
                result.reason = result.observation_reason = 'invalid_observation'
                result.error_type = type(error).__name__
                result.gate_a_passed = False
                self._restore_confirmed(result)
        self.maximum_tracks = max(self.maximum_tracks, len(self.tracker.tracks))
        self.last_reason = 'no_face' if not results else results[0].reason
        self.last_frame_key, self.last_results = frame_key, tuple(results)
        return results
