#!/usr/bin/env python3
"""One-shot Act1 mainline monitor.

This wrapper intentionally keeps the mainline check boring and repeatable:

1. show the active MuZero training process, if any;
2. run the Act1 gate report;
3. run the death-deck post-mortem report.

The separate reports remain the source of truth.  This file exists so the
"look at logs" workflow does not forget the user-required death deck/card
composition check while we iterate toward stable Act1 clears.
"""

from __future__ import annotations

import argparse
import shutil
import subprocess
import sys
from pathlib import Path


def _script_dir() -> Path:
    return Path(__file__).resolve().parent


def _run(cmd: list[str], *, check: bool = False) -> int:
    print(f"\n$ {' '.join(cmd)}", flush=True)
    try:
        completed = subprocess.run(cmd, check=check)
    except FileNotFoundError as exc:
        print(f"[monitor_act1_mainline] command not found: {exc}", file=sys.stderr)
        return 127
    return int(completed.returncode)


def _print_train_processes() -> int:
    if not shutil.which("ps"):
        print("Train processes: ps not available on this platform")
        return 0
    print("Train processes:")
    cmd = [
        "bash",
        "-lc",
        "ps -eo pid,etime,pcpu,pmem,args | "
        "grep -E 'muzero\\.train|train.py' | grep -v grep || true",
    ]
    if not shutil.which("bash"):
        # Windows-native fallback.  Most training runs are launched under WSL,
        # so this is best-effort only.
        cmd = [
            "powershell",
            "-NoProfile",
            "-Command",
            "Get-Process | Where-Object { $_.ProcessName -match 'python' } | "
            "Select-Object Id,CPU,PM,ProcessName | Format-Table -AutoSize",
        ]
        if not shutil.which("powershell"):
            print("  no bash/powershell process lister available")
            return 0
    return _run(cmd)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tail", type=int, default=50, help="Act1 gate scalar tail window.")
    parser.add_argument("--death-tail", type=int, default=10, help="Number of death decks to show.")
    parser.add_argument(
        "--show-cards",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Show full card lists in the death-deck report.",
    )
    parser.add_argument(
        "--processes",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Print active train processes before the reports.",
    )
    parser.add_argument(
        "--strict",
        action="store_true",
        help="Return non-zero if any child report exits non-zero.",
    )
    args = parser.parse_args(argv)

    scripts = _script_dir()
    python = sys.executable
    exit_codes: list[int] = []

    if args.processes:
        exit_codes.append(_print_train_processes())

    gate_cmd = [
        python,
        str(scripts / "monitor_fullrun_act1_gate.py"),
        "--tail",
        str(max(int(args.tail), 1)),
    ]
    exit_codes.append(_run(gate_cmd))

    death_cmd = [
        python,
        str(scripts / "report_fullrun_death_decks.py"),
        "--tail",
        str(max(int(args.death_tail), 1)),
    ]
    if bool(args.show_cards):
        death_cmd.append("--show-cards")
    exit_codes.append(_run(death_cmd))

    worst = max(exit_codes or [0])
    if args.strict and worst != 0:
        return worst
    return 0


if __name__ == "__main__":  # pragma: no cover - CLI entrypoint
    raise SystemExit(main())
