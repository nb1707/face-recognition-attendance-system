"""Bounded descriptive tracking diagnostics; never claims labeled accuracy."""
from dataclasses import asdict
from collections import Counter
import random
import numpy as np


class ReplayDiagnostics:
    """Exact counters plus a fixed-size deterministic reservoir of track rows.

    ``row_sink`` may stream every retired/final active row to JSONL.  The in-RAM
    reservoir is used only for descriptive medians and remains bounded even when
    replay continues to EOF.
    """
    def __init__(self, max_track_records, row_sink=None):
        if max_track_records < 1:
            raise ValueError('max_track_records must be positive')
        self.max_track_records = max_track_records
        self.retired = []
        self.counts = Counter()
        self.row_sink = row_sink
        self.rows_seen = 0
        self._random = random.Random(0)
        self._finished = False

    def _record(self, row):
        if self.row_sink is not None:
            self.row_sink(row)
        self.rows_seen += 1
        self.counts['one_observation_tracks'] += row['observations'] == 1
        self.counts['tracks_with_misses'] += bool(row['had_miss'])
        self.counts['tracks_recovered_after_miss'] += row['short_miss_recoveries'] > 0
        self.counts['active_tracks_censored'] += bool(row['censored'])
        if len(self.retired) < self.max_track_records:
            self.retired.append(row)
            return
        index = self._random.randrange(self.rows_seen)
        if index < self.max_track_records:
            self.retired[index] = row

    def observe(self, engine, decisions):
        update = engine.tracker.last_update
        self.counts['tracks_created'] += len(update['created'])
        self.counts['short_misses_recovered'] += len(update['recovered'])
        self.counts['ambiguous_associations'] += update.get('ambiguous_detection_count', 0)
        for track, reason in update['retired']:
            self._record(dict(asdict(track), retirement_reason=reason, censored=False))
            self.counts['ambiguity_resets' if reason == 'ambiguity' else 'track_expirations'] += 1
        self.counts['detections'] += len(decisions)
        self.counts['quality_passed_observations'] += sum(bool(d.quality_decision and d.quality_decision.passed) for d in decisions)
        self.counts['selected_top_k_observations'] += sum(d.selected_new_observation for d in decisions)
        self.counts['gate_a_passes'] += sum(d.gate_a_passed for d in decisions)
        self.counts['gate_b_confirmations'] += sum(d.confirmation_transition for d in decisions)
        self.counts['confirmed_events'] += sum(d.event_emitted for d in decisions)

    def summary(self, tracker, duration):
        if not self._finished:
            for track in tracker.tracks.values():
                self._record(dict(asdict(track), retirement_reason=None, censored=True))
            self._finished = True
        rows = list(self.retired)
        sample_n = len(rows)
        n = self.counts['tracks_created']
        missed = self.counts['tracks_with_misses']
        recovered = self.counts['tracks_recovered_after_miss']
        result = {key: self.counts[key] for key in ('tracks_created', 'short_misses_recovered', 'track_expirations',
            'ambiguity_resets', 'ambiguous_associations', 'detections', 'quality_passed_observations', 'selected_top_k_observations',
            'gate_a_passes', 'gate_b_confirmations', 'confirmed_events')}
        result.update(tracks_created_per_sampled_minute=n*60/duration if duration > 0 else None,
            median_observed_track_lifetime_seconds=float(np.median([r['last_seen']-r['first_seen'] for r in rows])) if sample_n else None,
            median_observations_per_track=float(np.median([r['observations'] for r in rows])) if sample_n else None,
            one_observation_tracks=self.counts['one_observation_tracks'], tracks_total=n,
            sampled_track_rows=sample_n, track_statistics_exact=sample_n == n,
            track_reservoir_capacity=self.max_track_records,
            percent_one_observation_tracks=100*self.counts['one_observation_tracks']/n if n else None,
            tracks_with_misses=missed, tracks_recovered_after_miss=recovered,
            percent_tracks_surviving_short_miss=100*recovered/n if n else None,
            percent_missed_tracks_recovered=100*recovered/missed if missed else None,
            active_tracks_censored=self.counts['active_tracks_censored'],
            lifetime_definition='last observed acquisition time minus first; active tracks right-censored at replay end',
            statistics_note=('Exact because every track fit the reservoir.' if sample_n == n else
                             'Medians use a deterministic bounded reservoir; lifecycle counters and rates are exact.'))
        return result, rows
