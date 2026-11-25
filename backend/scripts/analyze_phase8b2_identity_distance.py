"""Validate or analyze manual Phase 8B.2 spatial ground truth; no inference."""
import argparse
import json
from pathlib import Path
import sys

BACKEND = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(BACKEND))

from analysis.spatial_ground_truth import (EXPECTED_GALLERY, discover_replays,
    load_json, run_full_analysis, validate_annotations, validate_assignment_coverage,
    write_full_outputs, write_validation)


def build_parser():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--metadata', type=Path,
        default=BACKEND/'storage/phase8b2/clip_metadata.json')
    parser.add_argument('--annotations', type=Path,
        default=BACKEND/'storage/phase8b2/spatial_ground_truth.json')
    parser.add_argument('--replay-root', type=Path,
        default=BACKEND/'storage/phase8a')
    parser.add_argument('--expected-gallery', default=EXPECTED_GALLERY)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--validate-annotations', action='store_true')
    return parser


def main(argv=None):
    args = build_parser().parse_args(argv)
    metadata, annotations = load_json(args.metadata), load_json(args.annotations)
    validation = validate_annotations(metadata, annotations)
    selected, rejected = discover_replays(args.replay_root, metadata, args.expected_gallery)
    discovery = dict(expected_gallery=args.expected_gallery,
        selected={clip_id: dict(summary_path=str(row['summary_path'].resolve()),
            observations_path=str(row['observations_path'].resolve()),
            gallery_generation=row['gallery_generation'],
            observations=row['summary']['videos'][0]['funnel']['detected_face_observations'])
            for clip_id, row in selected.items()}, rejected=rejected)
    coverage = validate_assignment_coverage(selected, metadata, annotations) if validation['complete'] else None
    if args.validate_annotations:
        write_validation(args.output, validation, discovery, coverage)
        print(json.dumps(dict(validation=validation, discovery=discovery,
                              coverage=coverage), indent=2))
        return 0 if validation['complete'] else 2
    if not validation['complete']:
        raise ValueError('Manual spatial annotations are incomplete; run --validate-annotations and review the template')
    result = run_full_analysis(selected, metadata, annotations)
    write_full_outputs(args.output, result, coverage)
    print(json.dumps(dict(output=str(args.output), gallery_generation=result['gallery_generation'],
                          gate_b_simulated=False), indent=2))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
