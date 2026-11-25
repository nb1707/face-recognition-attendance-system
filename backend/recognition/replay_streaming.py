"""Fixed-memory helpers for long offline replays."""
from collections import Counter
import math
import random
import numpy as np


class BoundedStatistics:
    """Exact count/min/max/mean and bounded-reservoir percentiles."""
    def __init__(self, capacity=4096, seed=0):
        if capacity < 1:
            raise ValueError('capacity must be positive')
        self.capacity = capacity
        self.count = 0
        self.minimum = self.maximum = None
        self.total = 0.0
        self.sample = []
        self._random = random.Random(seed)

    def add(self, value):
        if value is None or not math.isfinite(float(value)):
            return
        value = float(value)
        self.count += 1
        self.total += value
        self.minimum = value if self.minimum is None else min(self.minimum, value)
        self.maximum = value if self.maximum is None else max(self.maximum, value)
        if len(self.sample) < self.capacity:
            self.sample.append(value)
        else:
            index = self._random.randrange(self.count)
            if index < self.capacity:
                self.sample[index] = value

    def summary(self):
        if not self.count:
            return dict(count=0, min=None, median=None, mean=None, p25=None,
                        p75=None, p90=None, p95=None, max=None,
                        reservoir_size=0, reservoir_capacity=self.capacity, exact=True)
        return dict(count=self.count, min=self.minimum,
            median=float(np.median(self.sample)), mean=self.total/self.count,
            p25=float(np.percentile(self.sample, 25)), p75=float(np.percentile(self.sample, 75)),
            p90=float(np.percentile(self.sample, 90)), p95=float(np.percentile(self.sample, 95)),
            max=self.maximum, reservoir_size=len(self.sample), reservoir_capacity=self.capacity,
            exact=len(self.sample) == self.count)


class ReplayFunnel:
    """Streaming Phase 8A counters and bounded descriptive distributions."""
    DISTRIBUTIONS = ('face_width', 'face_height', 'sharpness', 'detector_confidence',
                     'top1_score', 'top2_score', 'margin', 'shadow_top1_score',
                     'shadow_top2_score', 'shadow_margin')

    def __init__(self, statistics_capacity=4096):
        self.counts = Counter()
        self.quality_reasons = Counter()
        self.distributions = {name: BoundedStatistics(statistics_capacity, index)
                              for index, name in enumerate(self.DISTRIBUTIONS)}

    def add_frame(self):
        self.counts['sampled_frames'] += 1

    def add_observation(self, row):
        self.counts['detected_face_observations'] += 1
        if row['valid_observation']:
            self.counts['valid_observations'] += 1
        if row['alignment_succeeded']:
            self.counts['aligned_observations'] += 1
        quality_passed = bool(row['quality_passed'])
        self.counts['quality_passed' if quality_passed else 'quality_rejected'] += 1
        self.counts['top_k_selected'] += bool(row['top_k_selected'])
        self.counts['adaface_inferences'] += row['production_adaface_call_count']
        self.counts['shadow_adaface_inferences'] += bool(row['shadow_embedding_generated'])
        self.counts['gate_a_passes'] += bool(row['gate_a_passed'])
        self.counts['gate_b_confirmations'] += bool(row['gate_b_confirmed'])
        self.counts['confirmed_events'] += bool(row['confirmed_event_emitted'])
        for reason in row['quality_reasons']:
            self.quality_reasons[reason] += 1
        for name in self.DISTRIBUTIONS:
            self.distributions[name].add(row.get(name))

    def summary(self):
        keys = ('sampled_frames', 'detected_face_observations', 'valid_observations',
                'aligned_observations', 'quality_passed', 'quality_rejected',
                'top_k_selected', 'adaface_inferences', 'shadow_adaface_inferences',
                'gate_a_passes', 'gate_b_confirmations', 'confirmed_events')
        result = {key: self.counts[key] for key in keys}
        result['quality_rejection_reasons'] = dict(self.quality_reasons)
        result['distributions'] = {key: value.summary() for key, value in self.distributions.items()}
        result['statistics_note'] = ('Counts, min, max and means are exact. Percentiles/medians are exact until the '
                                     'fixed reservoir fills, then descriptive reservoir estimates.')
        return result
