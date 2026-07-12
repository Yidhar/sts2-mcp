#!/usr/bin/env python3
"""Guards-OFF vs guards-FULL A/B comparison (Step-0 verify tool for RC-4).

The RC-4 finding is that ~25 hard guards overwrite the executed action AND rewrite
the stored policy target, so the trained policy != the executed policy. Before
changing that, we need to MEASURE whether the guards actually help: this tool diffs
the win/reward/override metrics of two matched runs -- one trained with
``--combat-hard-guard-policy full`` and one with ``... off`` -- so "guards fire a lot
but win-rate is no better" becomes visible instead of assumed.

Two modes:
  * diff (default): compare two existing TensorBoard run dirs.
        python scripts/ab_guards_off_vs_full.py --full-run-dir runs/<full> \
                                                 --off-run-dir  runs/<off>
  * --print-launch: emit the two matched launch commands (the ONLY CLI delta is the
    guard-policy token). Intentionally does not execute training itself.

Reuses summarize_scalar from monitor_fullrun_act1_gate so tail statistics match the
existing gate. Tags absent from a run print as "-".
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

_SCRIPTS_DIR = Path(__file__).resolve().parent
_RL_AGENT = _SCRIPTS_DIR.parent
for _p in (str(_SCRIPTS_DIR), str(_RL_AGENT)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

# (tag, "higher is better" | "lower is better" | "context")
_DEFAULT_TAGS: tuple[tuple[str, str], ...] = (
    ("recent_tail/256/win_rate", "higher"),
    ("recent_tail/256/reward_mean", "higher"),
    ("recent_tail/256/act1_boss_seen_rate", "higher"),
    ("recent_tail/64/win_rate", "higher"),
    ("combat_guards/hard_guard_override_any_rate", "context"),
    ("search/combat/post_search_hard_guard_policy_retargeted_rate", "context"),
    ("combat/intent_quality_selected_end_turn_rate", "lower"),
    ("combat/intent_quality_missed_lethal_rate", "lower"),
    ("combat/intent_quality_no_block_under_pressure_rate", "lower"),
)


def _summarize(run_dir: Path, tags: tuple[tuple[str, str], ...], tail: int) -> dict[str, dict]:
    # Lazy import so --print-launch / --help work without TensorBoard installed.
    from monitor_fullrun_act1_gate import EventAccumulator, summarize_scalar

    ea = EventAccumulator(str(run_dir), size_guidance={"scalars": 0})
    ea.Reload()
    present = set(ea.Tags().get("scalars", []))
    out: dict[str, dict] = {}
    for tag, _ in tags:
        out[tag] = summarize_scalar(ea, tag, tail) if tag in present else {"last": None, "avg_tail": None, "n": 0}
    return out


def _fmt(v) -> str:
    return "-" if v is None else f"{v:.4f}"


def _print_launch(base_run: str, total_timesteps: int) -> None:
    print("# Run these two matched commands; the ONLY difference is the guard-policy token.")
    print("# (fill in your real --resume-from / model / session flags from the active launch.sh)")
    for policy in ("full", "off"):
        print(
            f"\npython -m muzero.train --combat-hard-guard-policy {policy} "
            f"--total-timesteps {total_timesteps} "
            f"--log-dir runs/{base_run}_guards_{policy} "
            f"--checkpoint-dir checkpoints/{base_run}_guards_{policy} "
            "# + identical model/env/resume flags"
        )


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--full-run-dir", type=Path, help="Run dir trained with --combat-hard-guard-policy full")
    ap.add_argument("--off-run-dir", type=Path, help="Run dir trained with --combat-hard-guard-policy off")
    ap.add_argument("--tail", type=int, default=256, help="Tail window for avg statistics (default 256).")
    ap.add_argument("--field", choices=("last", "avg_tail"), default="avg_tail", help="Which statistic to compare.")
    ap.add_argument("--print-launch", metavar="BASE_RUN", help="Print the two matched launch commands and exit.")
    ap.add_argument("--launch-timesteps", type=int, default=20000)
    args = ap.parse_args()

    if args.print_launch:
        _print_launch(args.print_launch, args.launch_timesteps)
        return 0

    if not args.full_run_dir or not args.off_run_dir:
        ap.error("diff mode needs both --full-run-dir and --off-run-dir (or use --print-launch)")

    full = _summarize(args.full_run_dir, _DEFAULT_TAGS, args.tail)
    off = _summarize(args.off_run_dir, _DEFAULT_TAGS, args.tail)

    print(f"A/B guards comparison (field={args.field}, tail={args.tail})")
    print(f"  FULL = {args.full_run_dir.name}")
    print(f"  OFF  = {args.off_run_dir.name}\n")
    print(f"{'tag':<58} {'FULL':>10} {'OFF':>10} {'Δ(FULL-OFF)':>12}  note")
    print("-" * 104)
    for tag, direction in _DEFAULT_TAGS:
        a = full[tag].get(args.field)
        b = off[tag].get(args.field)
        delta = (a - b) if (a is not None and b is not None) else None
        print(f"{tag:<58} {_fmt(a):>10} {_fmt(b):>10} {_fmt(delta):>12}  ({direction})")

    print("-" * 104)
    fw = full["recent_tail/256/win_rate"].get(args.field)
    ow = off["recent_tail/256/win_rate"].get(args.field)
    ov = full["combat_guards/hard_guard_override_any_rate"].get(args.field)
    if fw is not None and ow is not None:
        if ov is not None and ov > 0.05 and fw <= ow + 1e-4:
            print(
                f"VERDICT: guards fire often (override_rate={ov:.3f}) but FULL win_rate ({fw:.4f}) is NOT above "
                f"OFF ({ow:.4f}) -> guards are masking, not teaching. Strong case for RC-4 (stop target-rewrite)."
            )
        elif fw > ow + 1e-4:
            print(f"VERDICT: FULL win_rate ({fw:.4f}) > OFF ({ow:.4f}) -> guards currently help; keep RC-4 conservative.")
        else:
            print(f"VERDICT: FULL ({fw:.4f}) ~= OFF ({ow:.4f}); inconclusive on win_rate -- look at the other rows.")
    else:
        print("VERDICT: win_rate missing in one run (full-run mode does not log per-room win; use a sandbox A/B).")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
