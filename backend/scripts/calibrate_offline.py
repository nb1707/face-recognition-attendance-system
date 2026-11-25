"""Sweep saved raw scores only; does not load models or edit runtime configuration."""
import argparse
import json
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from recognition.offline_evaluation import sweep, pareto_candidates, EVALUATION_MODES


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--records', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--mode', choices=(*EVALUATION_MODES, 'both'), default='both')
    args = parser.parse_args()
    records = [json.loads(line) for line in args.records.read_text(encoding='utf-8').splitlines()]
    thresholds = [round(.2+i*.025, 3) for i in range(25)]
    margins = [round(i*.025, 3) for i in range(13)]
    modes = EVALUATION_MODES if args.mode == 'both' else (args.mode,)
    if any(r.get('collection_mode') not in ('both', 'matcher_diagnostic') for r in records
           if not r.get('excluded_reason')) and 'matcher_diagnostic' in modes:
        raise ValueError('Production-only scores cannot reconstruct matcher diagnostics')
    views = {}
    for mode in modes:
        rows = sweep(records, thresholds, margins, mode=mode)
        views[mode] = dict(operating_points=rows, pareto_candidates=pareto_candidates(rows))
    result = dict(threshold_grid=thresholds, margin_grid=margins, views=views,
                  PRODUCTION_OPERATING_POINT_SELECTED='NO', calibration_status='INSUFFICIENT FOR PRODUCTION CALIBRATION',
                  metric_definitions=dict(FAR='accepted (unenrolled + explicit unknown) / all impostors',
                      FRR='rejected enrolled-known / enrolled-known',
                      wrong_identity_rate='enrolled-known accepted as another enrolled user / enrolled-known',
                      known_identity_accuracy='correctly accepted enrolled-known / enrolled-known',
                      macro_f1='union of true/predicted labels; both impostor categories share rejection truth'),
                  limitations='Exploratory, not held-out validation. Static Gate A only. Undefined denominators are null. Original person labels preserved; enrollment determines metric population. No threshold auto-application.')
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2), encoding='utf-8')
    print(json.dumps(dict(points_per_mode=len(rows), modes=list(views),
                         pareto_counts={mode:len(data['pareto_candidates']) for mode, data in views.items()})))


if __name__ == '__main__':
    main()
