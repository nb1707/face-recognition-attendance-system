"""Safely remove identities from the active local gallery via a new generation."""

import argparse
from pathlib import Path
import sys


BACKEND = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(BACKEND))

from enrollment.identity_removal import (  # noqa: E402
    LocalIdentityRemoval,
    LocalIdentityRemovalError,
)


def build_parser():
    parser = argparse.ArgumentParser(
        description=(
            "Publish a new local gallery generation without selected user IDs."
        )
    )
    parser.add_argument("user_ids", nargs="+", help="Active user ID(s) to remove")
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Show the removal plan without changing any file",
    )
    parser.add_argument(
        "--yes",
        action="store_true",
        help="Skip the interactive live-removal confirmation",
    )
    parser.add_argument(
        "--gallery-root",
        type=Path,
        default=BACKEND / "storage" / "local_gallery",
        help=argparse.SUPPRESS,
    )
    parser.add_argument(
        "--enrollments-root",
        type=Path,
        default=BACKEND / "storage" / "enrollments",
        help=argparse.SUPPRESS,
    )
    return parser


def print_plan(plan):
    def print_ids(label, values):
        print(f"{label}:")
        if values:
            for user_id in values:
                print(f"- {user_id}")
        else:
            print("- (none)")

    print(f"Current generation: {plan.current_generation}")
    print(f"Identity count before: {plan.identity_count_before}")
    print_ids("IDs requested for deletion", plan.requested_user_ids)
    print_ids("IDs found", plan.found_user_ids)
    print_ids("IDs not found", plan.not_found_user_ids)
    print(f"Identity count after: {plan.identity_count_after}")


def main(argv=None):
    args = build_parser().parse_args(argv)
    manager = LocalIdentityRemoval(args.gallery_root, args.enrollments_root)
    try:
        plan = manager.plan(args.user_ids)
        print_plan(plan)
        if args.dry_run:
            print("DRY RUN ONLY — no changes made.")
            return 0
        if plan.not_found_user_ids:
            raise LocalIdentityRemovalError(
                "Unknown active-gallery user ID(s): "
                + ", ".join(plan.not_found_user_ids)
            )
        if plan.identity_count_after == 0:
            raise LocalIdentityRemovalError(
                "Refusing to remove every active identity; no force-empty mode is available."
            )
        if not args.yes:
            answer = input(
                "Publish a new active generation with these identities removed? [y/N] "
            )
            if answer.strip().lower() not in {"y", "yes"}:
                print("Cancelled — no changes made.")
                return 1
        result = manager.remove(
            args.user_ids,
            expected_generation=plan.current_generation,
        )
        print(f"New generation: {result.generation}")
        print(f"Removed identities: {', '.join(result.removed_user_ids)}")
        print(f"Remaining identities: {result.identity_count_after}")
        print("Old generation preserved for rollback.")
        return 0
    except LocalIdentityRemovalError as error:
        print(f"ERROR: {error}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
