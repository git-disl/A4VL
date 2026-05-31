#!/usr/bin/env python3
"""Unified launcher for A4VL benchmark entry scripts."""

from __future__ import annotations

import argparse
import sys

from utils.runner import available_targets, launch_target


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run one A4VL benchmark script (or all of them) from a single entry point."
    )
    parser.add_argument(
        "target",
        nargs="?",
        help="Benchmark target: nextqa | ego | mlvu | all",
    )
    parser.add_argument(
        "--list",
        action="store_true",
        help="List available targets and exit.",
    )
    parser.add_argument(
        "script_args",
        nargs=argparse.REMAINDER,
        help="Arguments forwarded to the underlying script. Use '--' before forwarded args.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()

    if args.list:
        for name in available_targets():
            print(name)
        return 0

    if not args.target:
        print("Missing target. Use --list to see choices.", file=sys.stderr)
        return 2

    script_args = args.script_args
    if script_args and script_args[0] == "--":
        script_args = script_args[1:]

    return launch_target(args.target, script_args)


if __name__ == "__main__":
    raise SystemExit(main())
