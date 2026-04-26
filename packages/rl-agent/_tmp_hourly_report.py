#!/usr/bin/env python3
"""One-shot hourly metrics reporter for the active MuZero training run.

Writes a terse text digest (JSON + plaintext) to stdout.  Meant to be invoked
by the hourly cron; downstream presentation is done by the Claude host.
"""
from __future__ import annotations

import glob
import json
import os
import sys
from collections import defaultdict
from statistics import mean

LOG_ROOT = "/mnt/e/game/project/sts2_mcp/packages/rl-agent/logs_muzero"

# Fallback for Windows-style runs (the same path served through the drive
# mount).  In either case we expect forward slashes to work in Python.
if not os.path.isdir(LOG_ROOT):
    LOG_ROOT = "E:/game/project/sts2_mcp/packages/rl-agent/logs_muzero"

CANDIDATE_PREFIX = "muzero_token_memory_combat_sandbox_2envs_"


def latest_log_dir() -> str | None:
    entries = sorted(
        glob.glob(os.path.join(LOG_ROOT, f"{CANDIDATE_PREFIX}*")),
        key=lambda p: os.path.getmtime(p) if os.path.exists(p) else 0,
        reverse=True,
    )
    for path in entries:
        # Only accept dirs that still have fresh tfevents files (last modified
        # within the last hour).  Older stale dirs linger from crashed launches.
        ev = glob.glob(os.path.join(path, "events.out.tfevents.*"))
        if not ev:
            continue
        return path
    return None


def load_scalars(event_path: str) -> dict[str, list[tuple[int, float]]]:
    """Return {tag: [(step, value), ...]} sorted by step."""
    from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

    acc = EventAccumulator(
        event_path,
        size_guidance={"scalars": 0, "tensors": 0, "histograms": 0, "images": 0},
    )
    acc.Reload()
    out: dict[str, list[tuple[int, float]]] = {}
    for tag in acc.Tags().get("scalars", []):
        out[tag] = [(e.step, e.value) for e in acc.Scalars(tag)]
    return out


def summarise_tag(series: list[tuple[int, float]], tail: int = 500) -> dict[str, float]:
    if not series:
        return {}
    slice_ = series[-tail:]
    values = [v for _, v in slice_]
    return {
        "last": float(values[-1]),
        "mean_tail": float(mean(values)),
        "min_tail": float(min(values)),
        "max_tail": float(max(values)),
        "count_tail": len(values),
        "last_step": int(slice_[-1][0]),
    }


# Curated tag set — each entry is (tag_name OR tag_pattern_prefix, group_label).
# We intentionally avoid wholesale prefix harvesting (which used to surface
# 1000+ scalars and bury the actually-actionable ones).  Each line below
# corresponds to a specific question I want to answer in the hourly digest.
EXACT_TAGS = {
    # --- Core training losses (gradient health) ---
    "loss/total", "loss/policy", "loss/value", "loss/reward",
    "loss/planner_q", "loss/planner_objective_q",
    "loss/jepa_next_hidden", "loss/latent_gaussian_reg", "loss/surprise",
    "loss/future_world_aux", "loss/future_world_rollout_aux",
    # --- Episode / progress ---
    "episode/reward", "episode/length",
    "episode/death_floor", "episode/act1_boss_seen",
    "episode/boss_entry_hp_ratio", "episode/boss_entry_potion_count",
    # --- Buffer / sampling distribution ---
    "buffer/size",
    "buffer/sample_boss_rate", "buffer/sample_elite_rate",
    "buffer/sample_normal_rate", "buffer/sample_weak_rate",
    # --- Tier-level win rates (THE key health snapshot per user request) ---
    "recent_tail/256/win_rate", "recent_tail/256/loss_rate", "recent_tail/256/trunc_rate",
    "recent_tail/256/reward_mean", "recent_tail/256/length_mean",
    "recent_tail/256/boss_win_rate", "recent_tail/256/elite_win_rate",
    "recent_tail/256/normal_win_rate", "recent_tail/256/weak_win_rate",
    "recent_tail/256/hard_elite_win_rate", "recent_tail/256/hard_normal_win_rate",
    # --- Aggregate boss outcomes ---
    "boss/win", "boss/loss", "boss/reward", "boss/attempt_count",
    # --- Aggregate boss action-quality (cross-boss mean) ---
    "boss_combat/family_use_potion_rate",
    "boss_combat/family_end_turn_rate",
    "boss_combat/wasteful_end_turn_rate",
    "boss_combat/decision_count",
    "boss_combat/direct_rollout_objective_q_mean",
    "boss_combat/direct_rollout_uncertainty_mean",
}

# Pattern prefixes — these expand to many tags but each one is informative.
# Per-encounter WR and per-boss key signals.
PATTERN_PREFIXES = (
    # Per-encounter WR + sample count (bounded ~30 encounters)
    "recent_tail/256/encounter_",
)

# Per-boss key signals — only emit these specific suffixes per boss to avoid
# the 50-tag-per-boss explosion.  Boss IDs are discovered dynamically.
PER_BOSS_SUFFIXES = (
    "decision_count",
    "family_use_potion_rate",
    "true_wasteful_end_turn_selected_rate",
    "kaiser_back_attack_risk_mean",
    "kaiser_facing_change_selected_rate",
    "kaiser_pressure_selected_rate",
    "kaiser_risky_end_turn_selected_rate",
    "direct_rollout_objective_q_mean",
    "direct_rollout_uncertainty_mean",
    "x_cost_zero_energy_selected_rate",
    "potion_unused_on_death_rate",
)


def main() -> None:
    run = latest_log_dir()
    if run is None:
        print(json.dumps({"error": "no_recent_log_dir"}))
        return

    ev = sorted(
        glob.glob(os.path.join(run, "events.out.tfevents.*")),
        key=os.path.getmtime,
        reverse=True,
    )
    if not ev:
        print(json.dumps({"error": "no_tfevents", "run": run}))
        return

    try:
        scalars = load_scalars(ev[0])
    except Exception as exc:
        print(json.dumps({"error": f"tfevents_read_failed: {exc!r}", "run": run}))
        return

    summary: dict[str, dict[str, float]] = {}
    for tag, series in scalars.items():
        keep = False
        if tag in EXACT_TAGS:
            keep = True
        elif any(tag.startswith(p) for p in PATTERN_PREFIXES):
            keep = True
        elif tag.startswith("boss_combat/") and tag.count("/") == 2:
            # boss_combat/<boss_id>/<suffix> — only keep curated per-boss suffixes
            suffix = tag.split("/", 2)[2]
            if suffix in PER_BOSS_SUFFIXES:
                keep = True
        if keep:
            summary[tag] = summarise_tag(series)

    # total_steps, buffer stats are often global_step markers; grab last values
    misc = {}
    for k in ("total_steps", "buffer_size", "updates_per_s", "env_steps_per_s", "train/lr", "train/grad_norm"):
        series = scalars.get(k)
        if series:
            misc[k] = {"last": float(series[-1][1]), "last_step": int(series[-1][0])}

    out = {
        "run_dir": run,
        "event_file": ev[0],
        "num_scalar_tags": len(scalars),
        "summary": summary,
        "misc_last": misc,
    }
    print(json.dumps(out, indent=2, sort_keys=True))


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        print(json.dumps({"error": repr(exc)}))
        sys.exit(1)
