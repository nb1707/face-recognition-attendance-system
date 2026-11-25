"""Focused Phase 8B.3 evaluator tests; no video or inference."""

import json
from pathlib import Path
import sys

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'backend'))

from analysis.phase8b3_evaluator import REQUIRED_FILES, classify_event, select_runs


def annotation(overlap=False):
    regions = [dict(region_id='a', expected_user_id='alice', distance_group='near',
                    x1=0, y1=0, x2=.5, y2=1)]
    if overlap:
        regions.append(dict(region_id='overlap', expected_user_id='bob', distance_group='far',
                            x1=.25, y1=0, x2=.75, y2=1))
    return dict(frame_width=100, frame_height=100, fixed_regions=regions,
                moving_segments=[], conditions=[])


def observation(track, frame, x):
    return {'track_id': track, 'frame_sequence': frame, 'acquisition_time': frame / 10,
            'bbox': [x, 10, x + 10, 20]}


def event(user='alice', track=1, frame=3):
    return {'user_id': user, 'track_id': track, 'frame_sequence': frame,
            'acquisition_time': frame / 10}


@pytest.mark.parametrize(('user', 'expected'), [('alice', 'correct'), ('mallory', 'wrong')])
def test_event_unique_gt_classification(user, expected):
    result = classify_event(event(user), [observation(1, 1, 10)], annotation())
    assert result['classification'] == expected
    assert result['event_gt_identity'] == 'alice'


def test_event_conflict_never_forces_majority():
    rows = [observation(1, 1, 10), observation(1, 2, 10), observation(1, 3, 70)]
    clip = annotation()
    clip['fixed_regions'].append(dict(region_id='b', expected_user_id='bob',
                                      distance_group='far', x1=.5, y1=0, x2=1, y2=1))
    result = classify_event(event(), rows, clip)
    assert result['classification'] == 'conflict'
    assert result['assigned_identities'] == ['alice', 'bob']


def test_event_ambiguity_is_diagnostic_and_not_identity_truth():
    result = classify_event(event(), [observation(1, 1, 30)], annotation(overlap=True))
    assert result['classification'] == 'ambiguous'
    assert result['spatial_ambiguity_present'] is True
    assert result['event_gt_identity'] is None


def test_event_ignores_later_and_other_track_observations():
    rows = [observation(1, 4, 70), observation(2, 1, 10)]
    result = classify_event(event(), rows, annotation())
    assert result['classification'] == 'unassigned'
    assert result['relevant_observations'] == 0


def _complete_run(path, clip):
    path.mkdir(parents=True)
    for name in REQUIRED_FILES:
        (path / name).write_text('' if name != 'replay_summary.json' else
                                 json.dumps({'videos': [{'video': clip + '.mp4'}]}),
                                 encoding='utf-8')


def test_run_selection_uses_latest_complete_and_reports_incomplete(tmp_path):
    root = tmp_path / 'replays'
    for candidate in ('32_20', '32_16', '24_12'):
        _complete_run(root / candidate / 'clip-20261108-010000', 'clip')
        _complete_run(root / candidate / 'clip-20261108-020000', 'clip')
        incomplete = root / candidate / 'clip-20261108-030000'
        incomplete.mkdir(parents=True)
        (incomplete / 'replay_summary.json').write_text(
            json.dumps({'videos': [{'video': 'clip.mp4'}]}), encoding='utf-8')
    selected, rejected = select_runs(root, {'clip': {}})
    assert all(row['directory'].name.endswith('020000') for row in selected.values())
    assert sum(row['reason'] == 'incomplete' for row in rejected) == 3
    assert sum(row['reason'] == 'older_complete_duplicate' for row in rejected) == 3


def test_run_selection_never_uses_invalid_summary(tmp_path):
    root = tmp_path / 'replays'
    for candidate in ('32_20', '32_16', '24_12'):
        run = root / candidate / 'clip-20261108-010000'
        _complete_run(run, 'clip')
        (run / 'replay_summary.json').write_text('{broken', encoding='utf-8')
    with pytest.raises(ValueError, match='no complete run with a valid summary'):
        select_runs(root, {'clip': {}})
