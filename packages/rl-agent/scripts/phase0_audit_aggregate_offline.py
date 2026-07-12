"""Offline aggregator for Phase 0 audit.jsonl files.

When the live audit script gets killed by an external timeout (e.g.,
``timeout 1500`` in shell) before its finally block can write
``summary.json``, this helper rebuilds the summary from the audit.jsonl
records on disk. Loses unique-card metrics (those required the live
in-memory tracker) but recovers all per-step / per-episode rates.

Usage:
    python scripts/phase0_audit_aggregate_offline.py logs_audit/<dir>
"""
from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

# Reuse the same thresholds + key list as the live audit so output matches.
from scripts.phase0_audit import (  # type: ignore
    MIN_MAP_ACTIONS_FOR_PASS,
    MIN_MAP_STEPS_FOR_PASS,
    MIN_UNIQUE_CARDS_FOR_PASS,
    SOFT_THRESHOLD,
    THRESHOLD,
)
from sts2_rl.artifacts import (
    resolve_artifact_path,
    resolve_external_input_path,
    validate_artifact_component,
)


def aggregate_offline(audit_paths: list[Path]) -> dict:
    """Aggregate one or more audit.jsonl files. When multiple are given,
    episode indices are renumbered globally so floor-monotonicity etc.
    are computed per-source-episode.
    """
    records = []
    ep_offset = 0
    for audit_path in audit_paths:
        ep_max = -1
        with audit_path.open("r", encoding="utf-8") as fh:
            for line in fh:
                line = line.strip()
                if not line:
                    continue
                try:
                    rec = json.loads(line)
                except Exception:
                    continue
                rec["_episode_index"] = int(rec.get("_episode_index", 0)) + ep_offset
                ep_max = max(ep_max, rec["_episode_index"])
                records.append(rec)
        if ep_max >= 0:
            ep_offset = ep_max + 1
    n = len(records)
    if n == 0:
        return {"records": 0, "note": "empty audit.jsonl"}

    deck_present = sum(r["deck"]["deck_cards_present"] for r in records)
    deck_with_cards = [r["deck"] for r in records if r["deck"]["deck_size"] > 0]
    total_cards = sum(d["deck_size"] for d in deck_with_cards)
    metadata_hits = sum(d["metadata_hit_count"] for d in deck_with_cards)
    costs = sum(d["cost_present_count"] for d in deck_with_cards)
    types = sum(d["type_present_count"] for d in deck_with_cards)
    upgrades = sum(d["upgrade_present_count"] for d in deck_with_cards)

    map_step_records = [r for r in records if r["route"]["map_action_count"] > 0]
    total_map_actions = sum(r["route"]["map_action_count"] for r in map_step_records)
    summary_full_keys = sum(r["route"]["summary_full_keys_count"] for r in map_step_records)
    summary_present = sum(r["route"]["summary_present_count"] for r in map_step_records)
    nodes_present = sum(r["route"]["nodes_present_count"] for r in map_step_records)
    map_steps = len(map_step_records)
    aggregate_missing_keys: Counter = Counter()
    for r in map_step_records:
        for k, v in r["route"].get("missing_key_counter", {}).items():
            aggregate_missing_keys[k] += v

    floor_present = sum(r["floor"]["current_floor_present"] for r in records)
    act_present = sum(r["floor"]["act_id_present"] for r in records)
    room_present = sum(r["floor"]["room_type_present"] for r in records)
    active_present = sum(r["floor"]["active_present"] for r in records)

    schema_match = sum(r["encoded"].get("schema_match", 0) for r in records)
    action_mask_valid = sum(r["encoded"].get("action_mask_valid", 0) for r in records)
    no_nan = sum(r["encoded"].get("no_nan", 1) for r in records)
    no_inf = sum(r["encoded"].get("no_inf", 1) for r in records)
    encoded_present = sum(r["encoded"].get("encoded_present", 0) for r in records)

    # Per-episode reconstruction.
    ep_floor_traces: dict[int, list[int]] = {}
    for r in records:
        ep = r["_episode_index"]
        ep_floor_traces.setdefault(ep, []).append(r["floor"]["current_floor_value"])
    ep_indices = sorted(ep_floor_traces.keys())
    n_eps = len(ep_indices)
    monotonic_ok = 0
    reconstructable = 0
    max_floor_overall = 0
    for ep, flrs in ep_floor_traces.items():
        if all(flrs[i] <= flrs[i + 1] for i in range(len(flrs) - 1)):
            monotonic_ok += 1
        max_f = max(flrs) if flrs else 0
        if max_f > 0:
            reconstructable += 1
        max_floor_overall = max(max_floor_overall, max_f)

    # Episode-end signal: an episode is "ended cleanly" iff its final-step
    # record carries terminated=True OR truncated=True. Offline we don't
    # store those flags per-record; we infer from "episode_ended_rate" by
    # checking whether the LAST record of each episode was followed by a
    # different episode index (i.e., the run continued past it). For now
    # we treat reconstructable + non-empty episode trace as ended.
    ep_ended = reconstructable  # weak proxy; live audit has the real flag

    rates = {
        "deck/present_rate": deck_present / n if n else 0.0,
        "deck/metadata_hit_rate": (metadata_hits / total_cards) if total_cards else 0.0,
        "deck/cost_present_rate": (costs / total_cards) if total_cards else 0.0,
        "deck/type_present_rate": (types / total_cards) if total_cards else 0.0,
        "deck/upgrade_present_rate": (upgrades / total_cards) if total_cards else 0.0,
        "route/summary_present_rate": (summary_present / total_map_actions) if total_map_actions else 0.0,
        "route/full_20_key_present_rate": (summary_full_keys / total_map_actions) if total_map_actions else 0.0,
        "route/nodes_present_rate": (nodes_present / total_map_actions) if total_map_actions else 0.0,
        "route/map_step_share": map_steps / n if n else 0.0,
        "route/total_map_actions_observed": float(total_map_actions),
        "route/total_map_steps_observed": float(map_steps),
        "floor/current_floor_present_rate": floor_present / n if n else 0.0,
        "floor/act_id_present_rate": act_present / n if n else 0.0,
        "floor/room_type_present_rate": room_present / n if n else 0.0,
        "floor/run_active_present_rate": active_present / n if n else 0.0,
        "floor/episode_max_floor_reconstructable_rate": (reconstructable / n_eps) if n_eps else 0.0,
        "floor/current_floor_monotonic_or_valid_rate": (monotonic_ok / n_eps) if n_eps else 0.0,
        "floor/max_floor_reached_overall": float(max_floor_overall),
        "encoded/present_rate": encoded_present / n if n else 0.0,
        "encoded/schema_match_rate": schema_match / n if n else 0.0,
        "encoded/action_mask_valid_rate": action_mask_valid / n if n else 0.0,
        "encoded/no_nan_rate": no_nan / n if n else 0.0,
        "encoded/no_inf_rate": no_inf / n if n else 0.0,
    }

    sample_sufficient = {
        "route": (
            total_map_actions >= MIN_MAP_ACTIONS_FOR_PASS
            and map_steps >= MIN_MAP_STEPS_FOR_PASS
        ),
        # Unique-card check unavailable offline (raw card_ids not stored).
        "deck_unique": False,
    }

    threshold_map = {
        "deck/present_rate": THRESHOLD,
        "deck/metadata_hit_rate": SOFT_THRESHOLD,
        "deck/cost_present_rate": THRESHOLD,
        "deck/type_present_rate": THRESHOLD,
        "deck/upgrade_present_rate": SOFT_THRESHOLD,
        "route/summary_present_rate": THRESHOLD,
        "route/full_20_key_present_rate": THRESHOLD,
        "route/nodes_present_rate": THRESHOLD,
        "floor/current_floor_present_rate": THRESHOLD,
        "floor/act_id_present_rate": THRESHOLD,
        "floor/episode_max_floor_reconstructable_rate": THRESHOLD,
        "floor/current_floor_monotonic_or_valid_rate": THRESHOLD,
        "encoded/schema_match_rate": THRESHOLD,
        "encoded/action_mask_valid_rate": THRESHOLD,
        "encoded/no_nan_rate": THRESHOLD,
        "encoded/no_inf_rate": THRESHOLD,
    }

    passes: dict[str, dict] = {}
    for key, threshold in threshold_map.items():
        rate_value = rates.get(key, 0.0)
        bucket = "route" if key.startswith("route/") else None
        threshold_met = bool(rate_value >= threshold)
        if bucket == "route" and not sample_sufficient["route"]:
            passes[key] = {
                "rate": rate_value,
                "threshold": threshold,
                "threshold_met": threshold_met,
                "sample_sufficient": False,
                "pass": False,
                "status": "FIELD_PASS_SAMPLE_INSUFFICIENT" if threshold_met else "FAIL",
            }
        else:
            passes[key] = {
                "rate": rate_value,
                "threshold": threshold,
                "threshold_met": threshold_met,
                "sample_sufficient": True,
                "pass": threshold_met,
                "status": "PASS" if threshold_met else "FAIL",
            }

    return {
        "source": "offline_aggregator",
        "audit_path": str(audit_path),
        "records": n,
        "episodes": n_eps,
        "rates": rates,
        "thresholds": threshold_map,
        "passes": passes,
        "all_pass": all(info["pass"] for info in passes.values()),
        "sample_sufficient": sample_sufficient,
        "min_map_actions_for_pass": MIN_MAP_ACTIONS_FOR_PASS,
        "min_map_steps_for_pass": MIN_MAP_STEPS_FOR_PASS,
        "min_unique_cards_for_pass": MIN_UNIQUE_CARDS_FOR_PASS,
        "total_cards_observed_per_step": total_cards,
        "total_map_actions_observed": int(total_map_actions),
        "total_map_steps_observed": int(map_steps),
        "max_floor_reached_overall": int(max_floor_overall),
        "route_missing_key_counts": dict(aggregate_missing_keys),
        "note": (
            "Offline-rebuilt from audit.jsonl. Unique-card coverage "
            "(deck/unique_*) cannot be reconstructed without the live "
            "in-memory tracker; rerun phase0_audit.py end-to-end if "
            "needed."
        ),
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("audit_dirs", nargs="+", help="One or more audit dirs (each with audit.jsonl).")
    parser.add_argument("--out", default=None, help="Output summary path (default: first dir / summary_offline.json).")
    args = parser.parse_args()

    audit_paths: list[Path] = []
    for d in args.audit_dirs:
        p = resolve_external_input_path(d) / "audit.jsonl"
        if not p.exists():
            print(f"missing {p}", file=sys.stderr)
            return 2
        audit_paths.append(p)
    default_name = validate_artifact_component(
        f"{audit_paths[0].parent.name}-summary-offline.json",
        label="phase-0 aggregate report name",
    )
    out_path = resolve_artifact_path(
        args.out,
        default=f"reports/phase0/{default_name}",
    )
    out_path.parent.mkdir(parents=True, exist_ok=True)

    summary = aggregate_offline(audit_paths)
    summary["audit_paths"] = [str(p) for p in audit_paths]
    out_path.write_text(
        json.dumps(summary, ensure_ascii=False, indent=2, default=str),
        encoding="utf-8",
    )
    print(f"wrote {out_path}")
    print()
    rates = summary.get("rates", {}) or {}
    passes = summary.get("passes", {}) or {}
    for key in sorted(passes.keys()):
        info = passes[key]
        print(f"  {key:60s} {rates.get(key, 0):6.4f}  {info['status']}")
    print()
    print(f"all_pass={summary.get('all_pass', False)}")
    print(f"sample_sufficient={summary.get('sample_sufficient', {})}")
    print(f"map_actions={summary.get('total_map_actions_observed', 0)} (need ≥{MIN_MAP_ACTIONS_FOR_PASS})")
    print(f"map_steps={summary.get('total_map_steps_observed', 0)} (need ≥{MIN_MAP_STEPS_FOR_PASS})")
    return 0 if summary.get("all_pass", False) else 1


if __name__ == "__main__":
    raise SystemExit(main())
