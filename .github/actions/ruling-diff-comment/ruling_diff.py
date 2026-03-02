from __future__ import annotations

import argparse
import logging
import os
import sys

from ruling_diff_core import build_rule_diffs, format_comment
from ruling_diff_io import (
    GitHubActionIO,
    get_changed_ruling_files,
    post_or_update_comment,
)


def configure_logging() -> None:
    level = logging.DEBUG if os.environ.get("RUNNER_DEBUG") else logging.INFO
    logging.basicConfig(level=level, format="%(asctime)s %(levelname)s %(message)s")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate and post ruling diff comment"
    )
    parser.add_argument("--pr-number", required=True)
    parser.add_argument("--repository", required=True)
    parser.add_argument("--base-sha", required=True)
    parser.add_argument("--head-sha", required=True)
    args = parser.parse_args()
    if "/" not in args.repository:
        raise ValueError("--repository must be in owner/repo format")
    return args


def has_required_context(args: argparse.Namespace) -> bool:
    return bool(
        args.pr_number.strip() and args.base_sha.strip() and args.head_sha.strip()
    )


def main() -> None:
    configure_logging()
    args = parse_args()
    if not has_required_context(args):
        logging.info("Missing pr/base/head arguments. Skipping ruling diff comment.")
        return

    changed_files = get_changed_ruling_files(args.base_sha, args.head_sha)
    if not changed_files:
        logging.info("No changed ruling json files found. Nothing to do.")
        return

    logging.info("Found %d changed ruling json files", len(changed_files))
    io = GitHubActionIO()
    rule_diffs = build_rule_diffs(
        changed_files,
        args.base_sha,
        args.head_sha,
        io,
    )
    if not rule_diffs:
        logging.info("Changed files have no issue deltas. No comment will be posted.")
        return

    comment = format_comment(rule_diffs)
    post_or_update_comment(args.pr_number, args.repository, comment)


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        logging.error("Failed to generate ruling diff comment: %s", exc)
        sys.exit(1)
