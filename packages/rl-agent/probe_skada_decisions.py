"""Probe SP cleaned runs for BC-label extractability.

Answers 3 go/no-go questions for Phase 7:
  1. Map BC:   Can we walk visited_coords as (coord_from, options, chosen_child)
               and have chosen_child always appear in options's children?
  2. Rest BC:  For room_type="R" floors, is hp_delta a clean rest-vs-forge label?
               (Probed at scale in clean_skada_runs.py; this verifies per-run.)
  3. Relic BC: Do ancient/elite/boss relic_choices each have exactly one
               was_picked=true entry?

Also produces a SAMPLE DECISION FILE showing 3 example map decisions, 3 rest
decisions, and 3 relic decisions in the format a future BC loader would emit —
lets us eyeball whether the reconstructed state has enough context for the
policy to predict the action.

Usage:
  python probe_skada_decisions.py [--limit N] [--dump-samples PATH]
"""
from __future__ import annotations

import argparse
import collections
import json
from pathlib import Path

from sts2_rl.artifacts import resolve_artifact_path, resolve_external_input_path

SP_DIR = resolve_external_input_path(None, default="datasets/skada_clean/sp")


def load_runs(split_dir: Path, limit: int):
    """Yield (run_dict, outcome) from JSONL shards under sp/victory + sp/failure."""
    count = 0
    for outcome in ("victory", "failure"):
        for shard in sorted((split_dir / outcome).glob("*.jsonl")):
            with open(shard, "r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        yield json.loads(line), outcome
                    except Exception:
                        continue
                    count += 1
                    if limit and count >= limit:
                        return


def probe_map_decisions(obj) -> list[dict]:
    """Reconstruct (coord_from → chosen_child) pairs.

    Returns list of {act, from, options_count, chosen, chosen_in_options}.
    If chosen_in_options is False for any transition, map walk is broken.
    """
    out = []
    for map_act in obj.get("map_acts") or []:
        act = map_act.get("act")
        nodes = map_act.get("nodes") or []
        visited = map_act.get("visited_coords") or []
        if len(visited) < 2:
            continue
        # Index nodes by coord so we can look up children fast.
        by_coord = {tuple(n["coord"]): n for n in nodes if isinstance(n, dict)}
        for i in range(len(visited) - 1):
            a = tuple(visited[i])
            b = tuple(visited[i + 1])
            node = by_coord.get(a)
            if node is None:
                out.append({
                    "act": act, "from": list(a), "chosen": list(b),
                    "options_count": 0, "chosen_in_options": False,
                    "options_unresolved": True,
                })
                continue
            children = [tuple(c) for c in (node.get("children") or [])]
            chosen_in = b in children
            out.append({
                "act": act, "from": list(a),
                "chosen": list(b),
                "options": [list(c) for c in children],
                "options_count": len(children),
                "chosen_in_options": chosen_in,
                "options_unresolved": False,
                "room_type_from": node.get("type"),
            })
    return out


def probe_rest_decisions(obj) -> list[dict]:
    """For each room_type=R floor, emit rest-vs-forge label from HP delta."""
    out = []
    for ft in obj.get("floor_timeline") or []:
        if ft.get("room_type") != "R":
            continue
        hp_b = ft.get("hp_before")
        hp_a = ft.get("hp_after")
        if not isinstance(hp_b, int) or not isinstance(hp_a, int):
            continue
        delta = hp_a - hp_b
        if delta > 0:
            label = "rest"
        elif delta == 0:
            label = "forge"  # assumed — deck presumably changed
        else:
            label = "unknown_hp_loss"
        out.append({
            "floor": ft.get("floor"),
            "hp_before": hp_b,
            "hp_after": hp_a,
            "hp_delta": delta,
            "label": label,
        })
    return out


def probe_relic_decisions(obj) -> list[dict]:
    """Extract relic-pick decisions at each floor.

    ancient_choices: ancient event relic picks (one is chosen)
    relic_choices:   combat-reward or shop relic picks (from room or shop)
    """
    out = []
    for ft in obj.get("floor_timeline") or []:
        floor = ft.get("floor")
        room = ft.get("room_type")
        # Ancient choice (present at room_type=A typically)
        ac = ft.get("ancient_choices") or []
        if isinstance(ac, list) and ac:
            picks = [c for c in ac if isinstance(c, dict) and c.get("was_picked")]
            out.append({
                "floor": floor, "room": room, "kind": "ancient",
                "options_count": len(ac),
                "picked": [c["relic_id"] for c in picks if "relic_id" in c],
                "skipped": [c["relic_id"] for c in ac if not c.get("was_picked") and "relic_id" in c],
            })
        rc = ft.get("relic_choices") or []
        if isinstance(rc, list) and rc:
            # Relic choices from elite/boss/shop rooms. was_picked can be False for all (skipped).
            picks = [c for c in rc if isinstance(c, dict) and c.get("was_picked")]
            out.append({
                "floor": floor, "room": room, "kind": "relic",
                "options_count": len(rc),
                "picked": [c["relic_id"] for c in picks if "relic_id" in c],
                "skipped": [c["relic_id"] for c in rc if not c.get("was_picked") and "relic_id" in c],
            })
    return out


def reconstruct_context(obj, at_floor: int) -> dict:
    """Best-effort state snapshot at the *start* of a given floor.

    Returns whatever we can recover: floor, hp_ratio (from timeline), gold,
    relic_ids_so_far, act_index. Used to eyeball whether a BC policy could
    condition on enough context.
    """
    run = obj.get("run") or {}
    ctx = {
        "floor": at_floor,
        "character": run.get("character"),
        "ascension": run.get("ascension"),
        "seed": run.get("seed"),
        "act": None,          # derived from map_acts
        "hp_before": None,
        "gold_before": None,
        "relics_so_far": [],  # derived from floor_timeline[0..at_floor-1]
        "deck_size_unknown": True,  # flag: skada doesn't ship per-floor deck
    }
    for ft in obj.get("floor_timeline") or []:
        fl = ft.get("floor")
        if fl is None:
            continue
        if fl == at_floor:
            ctx["hp_before"] = ft.get("hp_before")
            ctx["gold_before"] = ft.get("gold_before")
            ctx["room_type"] = ft.get("room_type")
        if fl < at_floor:
            for ac in (ft.get("ancient_choices") or []):
                if isinstance(ac, dict) and ac.get("was_picked"):
                    ctx["relics_so_far"].append(ac.get("relic_id"))
            for rc in (ft.get("relic_choices") or []):
                if isinstance(rc, dict) and rc.get("was_picked"):
                    ctx["relics_so_far"].append(rc.get("relic_id"))
    # Derive act from which map_acts contains the floor. Act boundaries are
    # approximate in STS2: act1 ≈ floors 1-15, act2 ≈ 16-30, act3 ≈ 31-45.
    ctx["act"] = (at_floor - 1) // 15 if at_floor else None
    return ctx


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--limit", type=int, default=5000,
                        help="number of SP runs to probe (0 = all)")
    parser.add_argument("--dump-samples", type=str, default="data/skada_clean/phase7_samples.json",
                        help="write a handful of reconstructed decisions for eyeball review")
    args = parser.parse_args()

    if not SP_DIR.exists():
        raise SystemExit(f"SP dir missing: {SP_DIR} — run clean_skada_runs.py first")

    # Aggregate stats
    map_stats = collections.Counter()
    rest_stats = collections.Counter()
    relic_stats = collections.Counter()
    chars = collections.Counter()
    acts_seen = collections.Counter()
    options_hist_map = collections.Counter()
    picked_rate_ancient = collections.Counter()  # picks count per offered

    # Samples to dump (3 per kind, picked early for diversity)
    samples = {"map": [], "rest": [], "relic": [], "context": []}
    SAMPLE_CAP = 3

    runs_seen = 0
    for obj, outcome in load_runs(SP_DIR, args.limit):
        runs_seen += 1
        run = obj.get("run") or {}
        chars[run.get("character") or "?"] += 1

        map_tuples = probe_map_decisions(obj)
        for t in map_tuples:
            map_stats["transitions"] += 1
            acts_seen[t.get("act")] += 1
            if t.get("options_unresolved"):
                map_stats["from_not_in_nodes"] += 1
                continue
            options_hist_map[t["options_count"]] += 1
            if t["chosen_in_options"]:
                map_stats["chosen_in_options"] += 1
            else:
                map_stats["chosen_NOT_in_options"] += 1
        if map_tuples and len(samples["map"]) < SAMPLE_CAP:
            samples["map"].append({"run_id": run.get("run_id"), "transitions": map_tuples[:5]})

        rest_tuples = probe_rest_decisions(obj)
        for t in rest_tuples:
            rest_stats[t["label"]] += 1
        if rest_tuples and len(samples["rest"]) < SAMPLE_CAP:
            samples["rest"].append({"run_id": run.get("run_id"), "decisions": rest_tuples})

        relic_tuples = probe_relic_decisions(obj)
        for t in relic_tuples:
            relic_stats[f"{t['kind']}_offered"] += t["options_count"]
            relic_stats[f"{t['kind']}_picked"] += len(t["picked"])
            relic_stats[f"{t['kind']}_skipped"] += len(t["skipped"])
            if t["kind"] == "ancient":
                picked_rate_ancient[(t["options_count"], len(t["picked"]))] += 1
        if relic_tuples and len(samples["relic"]) < SAMPLE_CAP:
            samples["relic"].append({"run_id": run.get("run_id"), "decisions": relic_tuples[:6]})

        # Context-reconstruction sample at a random mid-run floor
        if len(samples["context"]) < SAMPLE_CAP and run.get("floor_reached", 0) > 8:
            mid_floor = run["floor_reached"] // 2
            samples["context"].append({
                "run_id": run.get("run_id"),
                "probe_floor": mid_floor,
                "context": reconstruct_context(obj, mid_floor),
            })

    # ---- Report ----
    print(f"probed {runs_seen:,} SP runs\n")
    print("Character distribution:")
    for c, n in chars.most_common(10):
        print(f"  {c}: {n:,}")
    print()

    print("== Map decisions ==")
    tot = map_stats["transitions"]
    print(f"  transitions:              {tot:,}")
    print(f"  from_not_in_nodes (bad):  {map_stats['from_not_in_nodes']:,}")
    in_opt = map_stats["chosen_in_options"]
    out_opt = map_stats["chosen_NOT_in_options"]
    if tot:
        print(f"  chosen_in_options:        {in_opt:,} ({in_opt/tot*100:.2f}%)")
        print(f"  chosen_NOT_in_options:    {out_opt:,} ({out_opt/tot*100:.2f}%)")
    print(f"  act distribution: {dict(sorted(acts_seen.items()))}")
    print(f"  options_count histogram: {dict(sorted(options_hist_map.items()))}")
    print()

    print("== Rest decisions ==")
    tot_rest = sum(rest_stats.values())
    print(f"  total rest-floor entries: {tot_rest:,}")
    for label, n in rest_stats.most_common():
        pct = n / tot_rest * 100 if tot_rest else 0
        print(f"  {label:18s} {n:>7,} ({pct:5.1f}%)")
    print()

    print("== Relic decisions ==")
    for k, v in relic_stats.most_common():
        print(f"  {k:22s} {v:>7,}")
    if picked_rate_ancient:
        print(f"  ancient (n_options, n_picked) histogram: {dict(picked_rate_ancient)}")
    print()

    # Dump samples
    if args.dump_samples:
        dump_path = resolve_artifact_path(args.dump_samples)
        dump_path.parent.mkdir(parents=True, exist_ok=True)
        with open(dump_path, "w", encoding="utf-8") as f:
            json.dump(samples, f, ensure_ascii=False, indent=2)
        print(f"sample decisions dumped to {dump_path}")


if __name__ == "__main__":
    main()
