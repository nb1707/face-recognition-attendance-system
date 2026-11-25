"""Analyze persisted Phase 8A observation diagnostics; performs no inference."""
import argparse
import json
from pathlib import Path
import sys

BACKEND = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(BACKEND))

from analysis.quality_sweep import analyze, write_outputs


def build_parser():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--observations', type=Path, required=True)
    parser.add_argument('--summary', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--sizes', type=float, nargs='+', default=(20, 24, 28, 32))
    parser.add_argument('--sharpnesses', type=float, nargs='+', default=(5, 8, 12, 16, 20))
    return parser


def main(argv=None):
    args = build_parser().parse_args(argv)
    if not args.observations.is_file() or not args.summary.is_file():
        raise ValueError('Existing observations JSONL and replay summary are required')
    if any(value <= 0 for value in args.sizes) or any(value < 0 for value in args.sharpnesses):
        raise ValueError('Quality sweep thresholds must be nonnegative and size must be positive')
    summary = json.loads(args.summary.read_text(encoding='utf-8'))
    result = analyze(args.observations, summary, args.sizes, args.sharpnesses)
    result['metadata']['summary_path'] = str(args.summary.resolve())
    if len(result['candidates']) != len(set((r['min_size'], r['min_sharpness'])
                                            for r in result['candidates'])):
        raise ValueError('Sweep grid contains duplicate candidate pairs')
    write_outputs(result, args.output)
    print(json.dumps(dict(output=str(args.output), metadata=result['metadata'],
                          development_shortlist=[dict(min_size=r['min_size'],
                            min_sharpness=r['min_sharpness'])
                            for r in result['development_shortlist']]), indent=2))


if __name__ == '__main__':
    main()
