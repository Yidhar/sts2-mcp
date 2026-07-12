#!/usr/bin/env python3
"""RC-1 resume dashboard: quantify buffer-drop restarts and win-rate resets.

Read-only verification tool for the RC-1 fix (replay buffer + optimizer should
survive restarts and guard/head edits). Scans the ``[resume] ...`` and
``RecentTail`` lines that ``muzero.train`` writes to each run's ``train.out`` and
reports, across the run lineage:

  * resumes that started with an EMPTY / dropped replay buffer (the failure RC-1
    fixes -- before the fix, every guard edit forced ``--resume-without-buffer``);
  * resumes where the optimizer was reset to fresh moments;
  * RecentTail win-rate that RESET to 0 immediately after a resume (the
    "transient 3% normal-win wiped on the next restart" pattern).

Usage:
    python scripts/rc1_resume_dashboard.py                      # scan runs/*/
    python scripts/rc1_resume_dashboard.py --logs-dir runs
    python scripts/rc1_resume_dashboard.py --run-dir runs/<run>

Exit code is 1 if any buffer-drop resume is found, so it can gate CI / a launch
script ("never start cold unless explicitly intended").
"""

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

RL_AGENT_ROOT = Path(__file__).resolve().parents[1]
if str(RL_AGENT_ROOT) not in sys.path:
    sys.path.insert(0, str(RL_AGENT_ROOT))

from sts2_rl.artifacts import resolve_external_input_path  # noqa: E402

_RE_LOADED = re.compile(r"\[resume\] Loaded total_steps=(\d+).*?buffer=(\d+)")
_RE_BUFFER_SKIP = re.compile(r"\[resume\] Skipping replay buffer warm-start")
_RE_BUFFER_EMPTY_WARN = re.compile(r"\[resume\]\[WARN\] Buffer load was requested but the buffer is EMPTY")
_RE_OPT_FRESH = re.compile(r"fresh optimizer|could not be matched to any current parameter")
_RE_OPT_PARTIAL = re.compile(r"Optimizer momentum partially restored: loaded=(\d+) reinitialized=(\d+)")
_RE_TAIL_WIN = re.compile(r"RecentTail.*?256\[win=([0-9.]+)")


def _scan_run(train_out: Path) -> dict:
    text = train_out.read_text(encoding="utf-8", errors="replace")
    loaded = _RE_LOADED.search(text)
    resumed_step = int(loaded.group(1)) if loaded else None
    loaded_buffer = int(loaded.group(2)) if loaded else None
    buffer_dropped = bool(
        _RE_BUFFER_SKIP.search(text)
        or _RE_BUFFER_EMPTY_WARN.search(text)
        or (loaded_buffer == 0 and resumed_step is not None)
    )
    opt_fresh = bool(_RE_OPT_FRESH.search(text)) and not _RE_OPT_PARTIAL.search(text)
    wins = [float(m) for m in _RE_TAIL_WIN.findall(text)]
    return {
        "run": train_out.parent.name,
        "resumed_step": resumed_step,
        "loaded_buffer": loaded_buffer,
        "buffer_dropped": buffer_dropped,
        "optimizer_fresh": opt_fresh,
        "win_first": wins[0] if wins else None,
        "win_max": max(wins) if wins else None,
        "win_last": wins[-1] if wins else None,
        "win_samples": len(wins),
    }


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument(
        "--logs-dir",
        default=None,
        help="Directory containing per-run subdirs (default: <STS2_ARTIFACT_ROOT>/runs).",
    )
    ap.add_argument("--run-dir", default=None, help="Analyze a single run dir instead of the whole logs dir.")
    args = ap.parse_args()

    base = resolve_external_input_path(args.logs_dir, default="runs")
    if args.run_dir:
        run_dirs = [resolve_external_input_path(args.run_dir, root=base)]
    else:
        run_dirs = sorted(p for p in base.iterdir() if p.is_dir()) if base.exists() else []

    rows = []
    for rd in run_dirs:
        train_out = rd / "train.out"
        if train_out.exists() and train_out.stat().st_size > 0:
            rows.append(_scan_run(train_out))

    if not rows:
        print(f"No runs with a non-empty train.out under {args.run_dir or args.logs_dir}.")
        return 0

    rows.sort(key=lambda r: (r["resumed_step"] is None, r["resumed_step"] or 0))
    dropped = [r for r in rows if r["buffer_dropped"]]
    win_resets = [r for r in rows if (r["win_max"] or 0) > 0 and (r["win_first"] == 0.0)]

    print(f"{'run':<70} {'resume@':>9} {'buf':>7} {'drop':>5} {'optFresh':>9} {'win f/max/last':>18}")
    print("-" * 124)
    for r in rows:
        winstr = "/".join(
            "-" if r[k] is None else f"{r[k]:.3f}" for k in ("win_first", "win_max", "win_last")
        )
        print(
            f"{r['run'][:70]:<70} {str(r['resumed_step']):>9} {str(r['loaded_buffer']):>7} "
            f"{'YES' if r['buffer_dropped'] else '-':>5} {'YES' if r['optimizer_fresh'] else '-':>9} {winstr:>18}"
        )

    print("-" * 124)
    print(f"runs scanned: {len(rows)}")
    print(f"buffer-drop resumes (RC-1 target): {len(dropped)}")
    print(f"optimizer-reset resumes: {sum(1 for r in rows if r['optimizer_fresh'])}")
    print(f"win-rate reset-to-0 after resume (had win>0, started at 0): {len(win_resets)}")
    if dropped:
        print("\nBuffer-drop runs (these lost their accumulated replay experience):")
        for r in dropped[:20]:
            print(f"  - {r['run']}  (resume@{r['resumed_step']}, loaded_buffer={r['loaded_buffer']})")
    return 1 if dropped else 0


if __name__ == "__main__":
    raise SystemExit(main())
