"""Analyze existing Phase 8B.3 controlled-replay outputs; performs no inference."""

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import sys

BACKEND = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(BACKEND))

from analysis.phase8b3_evaluator import evaluate, write_outputs


def build_parser():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--replay-root', type=Path,
                        default=BACKEND / 'storage' / 'phase8b3')
    parser.add_argument('--metadata', type=Path,
                        default=BACKEND / 'storage' / 'phase8b2' / 'clip_metadata.json')
    parser.add_argument('--ground-truth', type=Path,
                        default=BACKEND / 'storage' / 'phase8b2' / 'spatial_ground_truth.json')
    parser.add_argument('--output', type=Path, help='Must not already exist. Defaults to analysis/<UTC timestamp>.')
    return parser


def main(argv=None):
    args = build_parser().parse_args(argv)
    output = args.output or (args.replay_root / 'analysis' /
                             datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ'))
    result = evaluate(args.replay_root, args.metadata, args.ground_truth)
    write_outputs(result, output)
    print(json.dumps({'output': str(output.resolve()),
                      'selected_runs': result['selected_runs'],
                      'invariant_failures': result['invariant_failures']}, indent=2))


if __name__ == '__main__':
    main()
