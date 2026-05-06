#!/usr/bin/env python3
"""Monitor MuZero boss-recovery runs.

Reads TensorBoard scalar events from the latest logs_muzero run (or a supplied
run directory) and prints the exact gate metrics required by
``docs/muzero-boss-winrate-50-execution-plan-20260503.md``.

This script intentionally treats missing exact tags as a signal: it prints fuzzy
matches, but does not silently substitute them, because guessed metric names have
previously hidden observability bugs.
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from statistics import mean
from typing import Iterable

try:
    from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
except Exception as exc:  # pragma: no cover - operator-facing failure
    raise SystemExit(
        "TensorBoard EventAccumulator import failed. Run from packages/rl-agent venv. "
        f"Original error: {exc}"
    )

DEFAULT_ROOT = Path(__file__).resolve().parents[1] / "logs_muzero"

EXACT_TAGS: list[str] = [
    "recent_tail/64/boss_win_rate",
    "recent_tail/256/boss_win_rate",
    "boss/win",
    "boss/loss",
    "search/root_bias_nonzero_rate",
    "search/root_bias_changed_top1_rate",
    "search/root_bias_abs_mean",
    "search/root_bias_scale_effective_mean",
    "boss_combat/root_bias_nonzero_rate",
    "boss_combat/root_bias_changed_top1_rate",
    "boss_combat/true_wasteful_end_turn_selected_rate",
    "boss_combat/bad_end_turn_selected_rate",
    "boss_combat/forced_end_turn_selected_rate",
    "boss_combat/strategic_defer_end_turn_selected_rate",
    "boss_combat/kaiser_crab_boss/kaiser_facing_change_candidate_count_mean",
    "boss_combat/kaiser_crab_boss/kaiser_facing_change_selected_rate",
    "boss_combat/kaiser_crab_boss/kaiser_risky_end_turn_selected_rate",
    "boss_combat/kaiser_crab_boss/root_bias_changed_top1_rate",
    "boss_combat/ceremonial_beast_boss/ceremonial_high_impact_selected_rate",
    "boss_combat/ceremonial_beast_boss/ceremonial_low_impact_selected_rate",
    "boss_combat/ceremonial_beast_boss/ceremonial_missed_stun_window_rate",
    "boss_combat/ceremonial_beast_boss/root_bias_changed_top1_rate",
    "boss_combat/potion_low_urgency_selected_rate",
    "boss_combat/potion_high_urgency_selected_rate",
    "boss_combat/potion_lethal_selected_rate",
    "boss_combat/potion_prevent_lethal_selected_rate",
    "boss_combat/x_cost_zero_bad_selected_rate_p0",
    "boss_combat/hp_cost_self_lethal_selected_rate",
    "boss_combat/hp_cost_low_margin_selected_rate",
    "loss/total",
    "loss/future_world_aux",
    "loss/future_bank_state",
    "loss/future_bank_delta",
]

FUZZY_NEEDLES: list[str] = [
    "boss_win_rate",
    "root_bias",
    "kaiser",
    "ceremonial",
    "true_wasteful",
    "bad_end_turn",
    "forced_end_turn",
    "strategic_defer",
    "potion_high",
    "potion_low",
    "potion_lethal",
    "x_cost_zero",
    "hp_cost",
    "future_world_aux",
    "future_bank_state",
    "future_bank_delta",
]


def latest_run(root: Path) -> Path:
    runs = [p for p in root.iterdir() if p.is_dir()] if root.exists() else []
    if not runs:
        raise SystemExit(f"No run directories found under {root}")
    return max(runs, key=lambda p: p.stat().st_mtime)


def finite(values: Iterable[float]) -> list[float]:
    return [float(v) for v in values if isinstance(v, (int, float)) and math.isfinite(float(v))]


def scalar_summary(ea: EventAccumulator, tag: str, tail: int = 20) -> dict[str, float | int | None]:
    events = ea.Scalars(tag)
    vals = finite(ev.value for ev in events)
    last_ev = events[-1] if events else None
    tail_vals = vals[-tail:]
    return {
        "n": len(events),
        "step": int(last_ev.step) if last_ev else None,
        "last": float(last_ev.value) if last_ev else None,
        "mean_tail": float(mean(tail_vals)) if tail_vals else None,
        "min_tail": float(min(tail_vals)) if tail_vals else None,
        "max_tail": float(max(tail_vals)) if tail_vals else None,
    }


def fuzzy_matches(tags: set[str], missing_tag: str, max_items: int = 8) -> list[str]:
    lowered = missing_tag.lower()
    parts = [p for p in lowered.replace("/", "_").split("_") if len(p) >= 4]
    candidates: list[tuple[int, str]] = []
    for tag in tags:
        lt = tag.lower()
        score = 0
        if lowered in lt or lt.endswith(lowered):
            score += 10
        for part in parts:
            if part in lt:
                score += 1
        if score > 0:
            candidates.append((score, tag))
    candidates.sort(key=lambda item: (-item[0], item[1]))
    return [tag for _, tag in candidates[:max_items]]


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=DEFAULT_ROOT, help="logs_muzero root")
    parser.add_argument("--run-dir", type=Path, default=None, help="specific run directory")
    parser.add_argument("--tail", type=int, default=20, help="tail window for means")
    parser.add_argument("--json", action="store_true", help="emit JSON only")
    parser.add_argument("--list-fuzzy", action="store_true", help="list all tags matching fuzzy needles")
    args = parser.parse_args()

    run_dir = args.run_dir or latest_run(args.root)
    if not run_dir.exists():
        raise SystemExit(f"Run directory does not exist: {run_dir}")

    ea = EventAccumulator(str(run_dir), size_guidance={"scalars": 0})
    ea.Reload()
    tags = set(ea.Tags().get("scalars", []))

    rows: dict[str, dict[str, object]] = {}
    missing: dict[str, list[str]] = {}
    for tag in EXACT_TAGS:
        if tag in tags:
            rows[tag] = scalar_summary(ea, tag, tail=args.tail)
        else:
            missing[tag] = fuzzy_matches(tags, tag)

    fuzzy: dict[str, list[str]] = {}
    if args.list_fuzzy:
        for needle in FUZZY_NEEDLES:
            matches = sorted(t for t in tags if needle.lower() in t.lower())
            fuzzy[needle] = matches

    payload = {
        "run_dir": str(run_dir),
        "scalar_tag_count": len(tags),
        "metrics": rows,
        "missing_exact_tags": missing,
        "fuzzy_matches": fuzzy,
    }

    if args.json:
        print(json.dumps(payload, indent=2, ensure_ascii=False))
        return 0

    print(f"run_dir: {run_dir}")
    print(f"scalar_tag_count: {len(tags)}")
    print()
    print("EXACT METRICS")
    print("-------------")
    width = max(len(tag) for tag in EXACT_TAGS)
    for tag in EXACT_TAGS:
        if tag not in rows:
            print(f"{tag:<{width}}  MISSING")
            continue
        item = rows[tag]
        print(
            f"{tag:<{width}}  "
            f"last={item['last']} mean{args.tail}={item['mean_tail']} "
            f"min{args.tail}={item['min_tail']} max{args.tail}={item['max_tail']} "
            f"step={item['step']} n={item['n']}"
        )

    if missing:
        print()
        print("MISSING EXACT TAGS WITH FUZZY SUGGESTIONS")
        print("-----------------------------------------")
        for tag, matches in missing.items():
            if matches:
                print(f"{tag}: {matches}")
            else:
                print(f"{tag}: []")

    if args.list_fuzzy:
        print()
        print("FUZZY TAG GROUPS")
        print("----------------")
        for needle, matches in fuzzy.items():
            print(f"{needle}: {len(matches)}")
            for match in matches[:40]:
                print(f"  {match}")
            if len(matches) > 40:
                print("  ...")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
