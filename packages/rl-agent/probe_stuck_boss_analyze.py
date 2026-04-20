"""Analyze probe_stuck_boss_obs.py output. Surfaces structural diffs between
stuck-boss trajectories and control (winnable) trajectories: power coverage,
intent patterns, death turn, action diversity.
"""
from __future__ import annotations

import argparse
import collections
import json
import statistics
from pathlib import Path


def load_traces(directory: Path) -> dict[str, list[dict]]:
    by_enc: dict[str, list[dict]] = collections.defaultdict(list)
    for jsonl in sorted(directory.glob("*.jsonl")):
        with jsonl.open(encoding="utf-8") as fh:
            for line in fh:
                row = json.loads(line)
                by_enc[row["encounter_id"]].append(row)
    return by_enc


def collect_power_titles(steps: list[dict], side: str) -> set[str]:
    titles = set()
    for s in steps:
        if side == "player":
            for p in (s.get("player") or {}).get("powers") or []:
                t = (p.get("title") or "").strip()
                if t:
                    titles.add(t)
        else:
            for e in s.get("enemies") or []:
                for p in e.get("powers") or []:
                    t = (p.get("title") or "").strip()
                    if t:
                        titles.add(t)
    return titles


def collect_intent_types(steps: list[dict]) -> collections.Counter:
    cnt: collections.Counter = collections.Counter()
    for s in steps:
        for e in s.get("enemies") or []:
            it = (e.get("intent") or {}).get("type")
            if it:
                cnt[str(it)] += 1
    return cnt


def collect_action_kinds(steps: list[dict]) -> collections.Counter:
    cnt: collections.Counter = collections.Counter()
    for s in steps:
        ch = s.get("chosen") or {}
        k = ch.get("kind") or "?"
        cnt[str(k)] += 1
    return cnt


def max_damage_landed(steps: list[dict]) -> tuple[float, float]:
    """Max cumulative damage dealt to any single enemy across the episode,
    as fraction of that enemy's max_hp; plus absolute max."""
    if not steps:
        return 0.0, 0.0
    first = steps[0].get("enemies") or []
    initial = {e.get("id"): float(e.get("max_hp") or 0) for e in first if isinstance(e, dict)}
    best_frac = 0.0
    best_abs = 0.0
    for s in steps[1:]:
        for e in s.get("enemies") or []:
            eid = e.get("id")
            max_hp = initial.get(eid) or float(e.get("max_hp") or 0)
            hp = float(e.get("hp") or 0)
            if max_hp > 0:
                damage = max(0.0, max_hp - hp)
                frac = damage / max_hp
                best_frac = max(best_frac, frac)
                best_abs = max(best_abs, damage)
    return best_frac, best_abs


def analyze(by_enc: dict[str, list[dict]]) -> None:
    # Group by kind
    stuck_eps = []
    control_eps = []
    for enc_id, eps in by_enc.items():
        for ep in eps:
            kind = ep.get("kind")
            if kind == "stuck":
                stuck_eps.append(ep)
            elif kind == "control":
                control_eps.append(ep)

    def rollup(label: str, eps: list[dict]) -> None:
        print(f"\n=== {label} ({len(eps)} episodes) ===")
        if not eps:
            return
        outcomes = collections.Counter(ep.get("outcome") for ep in eps)
        print(f"outcomes: {dict(outcomes)}")
        steps_counts = [ep.get("steps_count", 0) for ep in eps]
        rewards = [ep.get("reward_sum", 0.0) for ep in eps]
        final_hps = [ep.get("final_hp") for ep in eps if ep.get("final_hp") is not None]
        print(f"mean steps: {statistics.mean(steps_counts):.1f}")
        print(f"mean reward: {statistics.mean(rewards):+.2f}")
        if final_hps:
            print(f"mean final_hp: {statistics.mean(final_hps):.1f}")

        # Aggregate powers and intents
        all_enemy_powers: collections.Counter = collections.Counter()
        all_player_powers: collections.Counter = collections.Counter()
        all_intents: collections.Counter = collections.Counter()
        all_actions: collections.Counter = collections.Counter()
        max_damage_fracs = []
        for ep in eps:
            steps = ep.get("steps") or []
            for t in collect_power_titles(steps, "enemy"):
                all_enemy_powers[t] += 1
            for t in collect_power_titles(steps, "player"):
                all_player_powers[t] += 1
            all_intents.update(collect_intent_types(steps))
            all_actions.update(collect_action_kinds(steps))
            frac, _ = max_damage_landed(steps)
            max_damage_fracs.append(frac)

        print(f"best-single-enemy damage frac (mean across eps): {statistics.mean(max_damage_fracs):.1%}"
              f" (max: {max(max_damage_fracs):.1%})")
        print(f"enemy power titles seen (count = eps featuring it):")
        for title, n in all_enemy_powers.most_common(20):
            print(f"  {n:>3}  {title}")
        print(f"player power titles seen:")
        for title, n in all_player_powers.most_common(15):
            print(f"  {n:>3}  {title}")
        print(f"top intent types: {dict(all_intents.most_common(8))}")
        print(f"top action kinds: {dict(all_actions.most_common(10))}")

    rollup("STUCK", stuck_eps)
    rollup("CONTROL", control_eps)

    # Diff: powers appearing in stuck but NOT control, and vice versa
    stuck_enemy_powers = set()
    control_enemy_powers = set()
    stuck_player_powers = set()
    control_player_powers = set()
    for ep in stuck_eps:
        stuck_enemy_powers.update(collect_power_titles(ep.get("steps") or [], "enemy"))
        stuck_player_powers.update(collect_power_titles(ep.get("steps") or [], "player"))
    for ep in control_eps:
        control_enemy_powers.update(collect_power_titles(ep.get("steps") or [], "enemy"))
        control_player_powers.update(collect_power_titles(ep.get("steps") or [], "player"))

    only_stuck_enemy = sorted(stuck_enemy_powers - control_enemy_powers)
    only_control_enemy = sorted(control_enemy_powers - stuck_enemy_powers)
    only_stuck_player = sorted(stuck_player_powers - control_player_powers)
    print("\n=== STRUCTURAL DIFF ===")
    print(f"enemy powers ONLY in stuck (count {len(only_stuck_enemy)}):")
    for t in only_stuck_enemy:
        print(f"  - {t}")
    print(f"enemy powers ONLY in control (count {len(only_control_enemy)}):")
    for t in only_control_enemy:
        print(f"  - {t}")
    print(f"player powers ONLY in stuck (count {len(only_stuck_player)}):")
    for t in only_stuck_player:
        print(f"  - {t}")

    # Per-encounter detail
    print("\n=== PER-ENCOUNTER DETAIL ===")
    for enc_id in sorted(by_enc):
        eps = by_enc[enc_id]
        if not eps:
            continue
        outcomes = collections.Counter(ep.get("outcome") for ep in eps)
        steps = [ep.get("steps_count", 0) for ep in eps]
        rewards = [ep.get("reward_sum", 0.0) for ep in eps]
        best_fracs = []
        for ep in eps:
            frac, _ = max_damage_landed(ep.get("steps") or [])
            best_fracs.append(frac)
        print(
            f"  {enc_id:<42}  n={len(eps)}  outcomes={dict(outcomes)}  "
            f"steps μ={statistics.mean(steps):.0f}  "
            f"reward μ={statistics.mean(rewards):+.2f}  "
            f"best-dmg-frac μ={statistics.mean(best_fracs):.0%}"
        )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("directory", help="Directory produced by probe_stuck_boss_obs.py")
    args = parser.parse_args()
    by_enc = load_traces(Path(args.directory))
    print(f"loaded traces: {len(by_enc)} encounters, "
          f"{sum(len(v) for v in by_enc.values())} episodes")
    analyze(by_enc)


if __name__ == "__main__":
    main()
