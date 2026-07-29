#!/usr/bin/env python3
"""CLI entry point: converts one raw Genie Studio recording to a LeRobot v2.1 dataset.

Usage:
    python convert.py --input <raw_recording_dir> --output <lerobot_dataset_dir> [--repo-id <id>]
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

from episode_builder import build_episode

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

# Default landing spot for converted datasets, next to this script — this is what the
# vla-app docker-compose.yml mounts as its episodes directory.
DEFAULT_OUTPUT_ROOT = Path(__file__).resolve().parent / "lerobot_format"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True, type=Path, help="Raw Genie Studio recording directory")
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help=f"Output LeRobot dataset directory (default: {DEFAULT_OUTPUT_ROOT}/<recording-name>)",
    )
    parser.add_argument(
        "--repo-id", default="local/agibot_g2", help="Local repo id passed to LeRobotDataset.create()"
    )
    args = parser.parse_args()

    if not args.input.is_dir():
        parser.error(f"--input {args.input} is not a directory")

    output = args.output if args.output is not None else DEFAULT_OUTPUT_ROOT / args.input.name

    try:
        build_episode(args.input, output, args.repo_id)
    except ValueError as exc:
        logger.error("Skipped %s: %s", args.input, exc)
        return 1

    return 0


if __name__ == "__main__":
    sys.exit(main())
