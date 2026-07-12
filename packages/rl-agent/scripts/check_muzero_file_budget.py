#!/usr/bin/env python3
"""Check MuZero Python files against the 2,000-line refactor budget.

Default behavior:
  * scans muzero/, sts2_env/, scripts/, tests/, and legacy/ when present
  * fails for every file above the line budget
  * has no repository legacy exceptions
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

PACKAGE_ROOT = Path(__file__).resolve().parents[1]
if str(PACKAGE_ROOT) not in sys.path:
    sys.path.insert(0, str(PACKAGE_ROOT))

from muzero.training.file_budget import (  # noqa: E402
    DEFAULT_MAX_LINES,
    DEFAULT_SCAN_ROOTS,
    collect_file_budget_records,
)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--package-root", type=Path, default=PACKAGE_ROOT)
    parser.add_argument("--max-lines", type=int, default=DEFAULT_MAX_LINES)
    parser.add_argument("--root", action="append", dest="roots", default=None)
    parser.add_argument(
        "--quiet-ok",
        action="store_true",
        help="Suppress the success line when no blocking violations are found.",
    )
    return parser.parse_args()


def main() -> int:
    args = _parse_args()
    roots = tuple(args.roots) if args.roots else DEFAULT_SCAN_ROOTS
    records = collect_file_budget_records(
        args.package_root,
        roots=roots,
        max_lines=args.max_lines,
    )
    over_budget = [record for record in records if record.over_budget]
    blocking = list(over_budget)

    if over_budget:
        print(f"Python files over {args.max_lines} lines:")
        for record in over_budget:
            print(f"  [BLOCK] {record.line_count:6d} {record.relative_path.as_posix()}")

    if blocking:
        print(
            f"\nERROR: {len(blocking)} non-compliant file(s) exceed the line budget. "
            "Split the file instead of adding it to the allowlist.",
            file=sys.stderr,
        )
        return 1

    if not args.quiet_ok:
        print(f"OK: no Python file exceeds {args.max_lines} lines.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
