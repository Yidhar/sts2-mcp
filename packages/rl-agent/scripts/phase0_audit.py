"""Phase 0 schema audit (recovery 2026-05-08).

Verifies that the fields required by the upcoming Deck-quality v2,
Route-heuristic, and Long-horizon training phases are actually present
in live bridge observations at >= 98% coverage. Per the review doc
(``docs/muzero-route-deck-long-horizon-review-20260508.md`` §5), this
audit MUST pass before any of the later phases ship.

Usage:
    python scripts/phase0_audit.py \
        --episodes 5 \
        --max-steps-per-episode 200 \
        --seed K8R3LFN7ZQ \
        --output-dir logs_audit/phase0_20260508

Per-step audit checks (raw bridge obs):
* deck_cards present + per-card id/cost/type/upgrade fields
* every map legal action carries a route_summary with ALL 20 documented keys
* every map legal action carries route_nodes
* run.floor / run.act_id / run.room_type / run.active

Per-step audit checks (encoded obs):
* run_memory / route_summary / route_nodes / objective_context / action_mask shapes
* no NaN / Inf in any encoded tensor

Episode-end checks:
* terminated / truncated emitted (long-horizon target masking)
* episode max_floor reconstructable from per-step current_floor
* current_floor monotonic-non-decreasing within an episode (modulo reset)

Unique-card audit:
* track set of distinct card_ids and verify metadata coverage on the
  *unique* set (not per-step duplicates)

Sample-size sanity:
* map_action_count and map_step_count must clear minimum thresholds
  before route/* PASS flags are reported as true.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from content_registry import get_card_metadata
from sts2_env.env_v2 import SlayTheSpire2EnvV2
from sts2_env.observation_v3 import WorldTokenObservationEncoder
from sts2_rl.artifacts import resolve_artifact_path, resolve_external_input_path

THRESHOLD = 0.98
SOFT_THRESHOLD = 0.95

# Minimum sample sizes before route/* rates can be reported PASS.
MIN_MAP_ACTIONS_FOR_PASS = 150
MIN_MAP_STEPS_FOR_PASS = 50
MIN_UNIQUE_CARDS_FOR_PASS = 12  # starter deck has ~6 unique; want ≥ starter+small reward picks

# All 20 documented route_summary keys (observation_common ROUTE_SUMMARY_DIM=20).
ROUTE_SUMMARY_FULL_KEYS: tuple[str, ...] = (
    "reachable_node_count",
    "max_depth",
    "direct_child_count",
    "forced_path_steps_before_branch",
    "count_monster",
    "count_elite",
    "count_boss",
    "count_event",
    "count_question_mark",
    "count_rest_site",
    "count_shop",
    "count_treasure",
    "next_elite_steps",
    "next_rest_steps",
    "next_shop_steps",
    "next_event_steps",
    "next_question_mark_steps",
    "next_treasure_steps",
    "can_reach_rest_site_before_elite",
    "can_reach_elite_then_rest_site",
)


# ---------------------------------------------------------------------------
# Per-step audit primitives
# ---------------------------------------------------------------------------

def audit_deck(obs: dict[str, Any], unique_cards: dict[str, dict[str, Any]]) -> dict[str, Any]:
    """Per-step deck audit + accumulate unique_cards across audit run.

    ``unique_cards`` maps card_id -> dict with metadata_hit, has_cost,
    has_type, has_upgrade flags. Mutated in place.
    """
    player = obs.get("player") if isinstance(obs.get("player"), dict) else {}
    deck_cards = player.get("deck_cards") if isinstance(player, dict) else None
    has_deck_cards = isinstance(deck_cards, list)
    deck_size = len(deck_cards) if has_deck_cards else 0

    card_id_present = 0
    metadata_hit = 0
    upgrade_present = 0
    cost_present = 0
    type_present = 0
    if has_deck_cards:
        for card in deck_cards:
            if not isinstance(card, dict):
                continue
            card_id = str(card.get("id") or "").strip()
            if card_id:
                card_id_present += 1
                metadata = get_card_metadata(card_id)
                meta_hit = bool(metadata)
                if meta_hit:
                    metadata_hit += 1
                # Track unique card coverage (Hole C).
                bucket = unique_cards.get(card_id)
                if bucket is None:
                    bucket = {
                        "metadata_hit": meta_hit,
                        "has_cost": card.get("cost") is not None,
                        "has_type": bool(card.get("type")),
                        "has_upgrade": (
                            card.get("upgrade_level") is not None
                            or card.get("upgraded") is not None
                        ),
                        "first_seen_step_index": None,  # filled by caller
                    }
                    unique_cards[card_id] = bucket
                else:
                    if meta_hit:
                        bucket["metadata_hit"] = True
                    if card.get("cost") is not None:
                        bucket["has_cost"] = True
                    if card.get("type"):
                        bucket["has_type"] = True
                    if card.get("upgrade_level") is not None or card.get("upgraded") is not None:
                        bucket["has_upgrade"] = True
            if card.get("upgrade_level") is not None or card.get("upgraded") is not None:
                upgrade_present += 1
            if card.get("cost") is not None:
                cost_present += 1
            if card.get("type"):
                type_present += 1

    return {
        "deck_cards_present": 1 if has_deck_cards else 0,
        "deck_size": deck_size,
        "card_id_present_count": card_id_present,
        "metadata_hit_count": metadata_hit,
        "upgrade_present_count": upgrade_present,
        "cost_present_count": cost_present,
        "type_present_count": type_present,
    }


def audit_route(legal_actions: list[Any]) -> dict[str, Any]:
    """Per-step route audit: full 20-key check + nodes presence.

    Hole A: must check all 20 documented keys, not just the 6 we happen
    to know first. Hole E: nodes_present must be aggregated and reported.
    """
    map_action_count = 0
    summary_present = 0
    summary_full_keys = 0
    nodes_present = 0
    elite_count_total = 0
    rest_count_total = 0
    missing_key_counter: Counter[str] = Counter()

    for action in legal_actions or []:
        if not isinstance(action, dict):
            continue
        if str(action.get("kind") or "").strip().lower() != "map":
            continue
        map_action_count += 1
        summary = action.get("route_summary")
        if isinstance(summary, dict):
            summary_present += 1
            keys = set(summary.keys())
            missing = [k for k in ROUTE_SUMMARY_FULL_KEYS if k not in keys]
            if not missing:
                summary_full_keys += 1
            else:
                for k in missing:
                    missing_key_counter[k] += 1
            elite_count_total += int(summary.get("count_elite") or 0)
            rest_count_total += int(summary.get("count_rest_site") or 0)
        nodes = action.get("route_nodes")
        if isinstance(nodes, list):
            nodes_present += 1

    return {
        "map_action_count": map_action_count,
        "summary_present_count": summary_present,
        "summary_full_keys_count": summary_full_keys,
        "nodes_present_count": nodes_present,
        "elite_count_total": elite_count_total,
        "rest_count_total": rest_count_total,
        "missing_key_counter": dict(missing_key_counter),
    }


def audit_floor(obs: dict[str, Any]) -> dict[str, Any]:
    run = obs.get("run") if isinstance(obs.get("run"), dict) else {}
    return {
        "current_floor_present": 1 if run and run.get("floor") is not None else 0,
        "current_floor_value": int(run.get("floor") or 0) if run else 0,
        "act_id_present": 1 if run and run.get("act_id") else 0,
        "act_id_value": str(run.get("act_id") or "") if run else "",
        "room_type_present": 1 if run and run.get("room_type") else 0,
        "active_present": 1 if run and run.get("active") is not None else 0,
        "run_active_value": bool(run.get("active")) if run else False,
    }


def audit_encoded(
    obs_encoded: dict[str, Any] | None,
    encoded_template: dict[str, tuple] | None,
) -> dict[str, Any]:
    """Hole G: validate the encoder-output schema is **stable across steps**.

    Encoder-agnostic: instead of hard-coding dense_v2 shapes (which
    differ from token_v3 entirely), we capture the schema from the
    FIRST step (shape + dtype.kind per key) and verify every subsequent
    step matches. Catches:

    * encoder dropping a key mid-episode,
    * encoder emitting a different shape (e.g., dynamic candidate count),
    * encoder leaking NaN / Inf in any float tensor,
    * encoder failing to emit ``action_mask`` with at least one valid bit.
    """
    out = {
        "encoded_present": 0,
        "schema_match": 0,
        "action_mask_valid": 0,
        "no_nan": 1,
        "no_inf": 1,
        "missing_keys": [],
        "shape_mismatches": [],
    }
    if not isinstance(obs_encoded, dict):
        return out
    out["encoded_present"] = 1

    # Schema check.
    if encoded_template:
        missing = []
        mismatches = []
        for key, (expected_shape, expected_kind) in encoded_template.items():
            value = obs_encoded.get(key)
            if not isinstance(value, np.ndarray):
                missing.append(key)
                continue
            if tuple(value.shape) != expected_shape:
                mismatches.append({"key": key, "got": list(value.shape), "want": list(expected_shape)})
                continue
            if value.dtype.kind != expected_kind:
                mismatches.append({"key": key, "got": value.dtype.kind, "want": expected_kind})
                continue
        if not missing and not mismatches:
            out["schema_match"] = 1
        out["missing_keys"] = missing
        out["shape_mismatches"] = mismatches

    # action_mask sanity.
    action_mask = obs_encoded.get("action_mask")
    if isinstance(action_mask, np.ndarray) and action_mask.size > 0 and float(action_mask.sum()) > 0:
        out["action_mask_valid"] = 1

    # NaN / Inf scan over numeric arrays.
    for value in obs_encoded.values():
        if not isinstance(value, np.ndarray):
            continue
        if value.dtype.kind in ("f", "c"):
            if not np.all(np.isfinite(value)):
                if np.any(np.isnan(value)):
                    out["no_nan"] = 0
                if np.any(np.isinf(value)):
                    out["no_inf"] = 0
    return out


def build_encoded_template(obs_encoded: dict[str, Any] | None) -> dict[str, tuple]:
    """Snapshot (shape, dtype.kind) per ndarray key from the first obs."""
    if not isinstance(obs_encoded, dict):
        return {}
    template: dict[str, tuple] = {}
    for key, value in obs_encoded.items():
        if isinstance(value, np.ndarray):
            template[key] = (tuple(value.shape), value.dtype.kind)
    return template


def audit_step(
    obs_raw: dict[str, Any],
    legal_actions: list[Any],
    obs_encoded: dict[str, Any] | None,
    unique_cards: dict[str, dict[str, Any]],
    encoded_template: dict[str, tuple] | None,
) -> dict[str, Any]:
    return {
        "deck": audit_deck(obs_raw, unique_cards),
        "route": audit_route(legal_actions),
        "floor": audit_floor(obs_raw),
        "encoded": audit_encoded(obs_encoded, encoded_template),
        "phase": str(obs_raw.get("phase") or ""),
        "decision_domain": str(obs_raw.get("decision_domain") or ""),
        "legal_action_count": len(legal_actions or []),
    }


# ---------------------------------------------------------------------------
# Aggregation
# ---------------------------------------------------------------------------

def aggregate(
    records: list[dict[str, Any]],
    episode_summaries: list[dict[str, Any]],
    unique_cards: dict[str, dict[str, Any]],
) -> dict[str, Any]:
    n = len(records)
    if n == 0:
        return {"episodes": 0, "steps": 0}

    deck_present = sum(r["deck"]["deck_cards_present"] for r in records)
    deck_with_cards = [r["deck"] for r in records if r["deck"]["deck_size"] > 0]
    total_cards = sum(d["deck_size"] for d in deck_with_cards)
    card_id_present = sum(d["card_id_present_count"] for d in deck_with_cards)
    metadata_hits = sum(d["metadata_hit_count"] for d in deck_with_cards)
    upgrades = sum(d["upgrade_present_count"] for d in deck_with_cards)
    costs = sum(d["cost_present_count"] for d in deck_with_cards)
    types = sum(d["type_present_count"] for d in deck_with_cards)

    # Hole C: unique-card coverage.
    unique_n = len(unique_cards)
    unique_meta_hit = sum(1 for v in unique_cards.values() if v["metadata_hit"])
    unique_has_cost = sum(1 for v in unique_cards.values() if v["has_cost"])
    unique_has_type = sum(1 for v in unique_cards.values() if v["has_type"])
    unique_has_upgrade = sum(1 for v in unique_cards.values() if v["has_upgrade"])
    missing_meta_ids = sorted(
        cid for cid, info in unique_cards.items() if not info["metadata_hit"]
    )

    map_step_records = [r for r in records if r["route"]["map_action_count"] > 0]
    map_steps = len(map_step_records)
    total_map_actions = sum(r["route"]["map_action_count"] for r in map_step_records)
    summary_presents = sum(r["route"]["summary_present_count"] for r in map_step_records)
    summary_full_keys = sum(r["route"]["summary_full_keys_count"] for r in map_step_records)
    nodes_presents = sum(r["route"]["nodes_present_count"] for r in map_step_records)
    aggregate_missing_keys: Counter[str] = Counter()
    for r in map_step_records:
        for k, v in r["route"]["missing_key_counter"].items():
            aggregate_missing_keys[k] += v

    floor_present = sum(r["floor"]["current_floor_present"] for r in records)
    act_present = sum(r["floor"]["act_id_present"] for r in records)
    room_present = sum(r["floor"]["room_type_present"] for r in records)
    active_present = sum(r["floor"]["active_present"] for r in records)

    # Hole F: long-horizon prerequisites.
    n_eps = len(episode_summaries)
    full_run_eps = sum(1 for e in episode_summaries if e["was_full_run"])
    terminated_present = sum(1 for e in episode_summaries if e["terminated_present"])
    truncated_present = sum(1 for e in episode_summaries if e["truncated_present"])
    # Combined episode-ended signal — either terminated or truncated must
    # be true for the episode to have produced a usable long-horizon
    # target. A "stuck" episode (neither flag fires) is the actual failure
    # mode this gate watches for.
    episode_ended = sum(
        1 for e in episode_summaries
        if e["terminated_present"] or e["truncated_present"]
    )
    max_floor_reconstructable = sum(1 for e in episode_summaries if e["max_floor_reconstructable"])
    monotonic_floor = sum(1 for e in episode_summaries if e["floor_monotonic_or_valid"])
    max_floor_reached_overall = max((e["max_floor"] for e in episode_summaries), default=0)

    # Hole G: encoded shape rates (encoder-agnostic).
    encoded_present = sum(r["encoded"]["encoded_present"] for r in records)
    schema_match = sum(r["encoded"]["schema_match"] for r in records)
    action_mask_valid = sum(r["encoded"]["action_mask_valid"] for r in records)
    no_nan = sum(r["encoded"]["no_nan"] for r in records)
    no_inf = sum(r["encoded"]["no_inf"] for r in records)
    schema_drift_steps = [
        r["_step_index"]
        for r in records
        if r["encoded"]["encoded_present"] and not r["encoded"]["schema_match"]
    ]

    phase_counter = Counter(r["phase"] for r in records)
    domain_counter = Counter(r["decision_domain"] for r in records)

    rates: dict[str, float] = {
        # Deck (per-step counted)
        "deck/present_rate": deck_present / n if n else 0.0,
        "deck/card_id_present_rate": (card_id_present / total_cards) if total_cards else 0.0,
        "deck/metadata_hit_rate": (metadata_hits / total_cards) if total_cards else 0.0,
        "deck/upgrade_present_rate": (upgrades / total_cards) if total_cards else 0.0,
        "deck/cost_present_rate": (costs / total_cards) if total_cards else 0.0,
        "deck/type_present_rate": (types / total_cards) if total_cards else 0.0,
        "deck/avg_size_in_present_steps": (
            sum(d["deck_size"] for d in deck_with_cards) / len(deck_with_cards)
            if deck_with_cards
            else 0.0
        ),
        # Deck (unique-card coverage — Hole C)
        "deck/unique_card_id_count": float(unique_n),
        "deck/unique_metadata_hit_rate": (unique_meta_hit / unique_n) if unique_n else 0.0,
        "deck/unique_cost_present_rate": (unique_has_cost / unique_n) if unique_n else 0.0,
        "deck/unique_type_present_rate": (unique_has_type / unique_n) if unique_n else 0.0,
        "deck/unique_upgrade_present_rate": (unique_has_upgrade / unique_n) if unique_n else 0.0,
        # Route (per-action / per-step)
        "route/summary_present_rate": (summary_presents / total_map_actions) if total_map_actions else 0.0,
        "route/full_20_key_present_rate": (summary_full_keys / total_map_actions) if total_map_actions else 0.0,
        "route/nodes_present_rate": (nodes_presents / total_map_actions) if total_map_actions else 0.0,
        "route/map_step_share": map_steps / n if n else 0.0,
        "route/avg_map_actions_per_map_step": (
            total_map_actions / map_steps if map_steps else 0.0
        ),
        "route/total_map_actions_observed": float(total_map_actions),
        "route/total_map_steps_observed": float(map_steps),
        # Floor (per-step / per-episode)
        "floor/current_floor_present_rate": floor_present / n if n else 0.0,
        "floor/act_id_present_rate": act_present / n if n else 0.0,
        "floor/room_type_present_rate": room_present / n if n else 0.0,
        "floor/run_active_present_rate": active_present / n if n else 0.0,
        "floor/full_run_episode_rate": (full_run_eps / n_eps) if n_eps else 0.0,
        "floor/terminated_present_rate": (terminated_present / n_eps) if n_eps else 0.0,
        "floor/truncated_present_rate": (truncated_present / n_eps) if n_eps else 0.0,
        "floor/episode_ended_rate": (episode_ended / n_eps) if n_eps else 0.0,
        "floor/episode_max_floor_reconstructable_rate": (
            max_floor_reconstructable / n_eps if n_eps else 0.0
        ),
        "floor/current_floor_monotonic_or_valid_rate": (monotonic_floor / n_eps) if n_eps else 0.0,
        "floor/max_floor_reached_overall": float(max_floor_reached_overall),
        # Encoded obs (Hole G — encoder-agnostic, schema captured from
        # first step then verified for stability across all subsequent steps).
        "encoded/present_rate": encoded_present / n if n else 0.0,
        "encoded/schema_match_rate": schema_match / n if n else 0.0,
        "encoded/action_mask_valid_rate": action_mask_valid / n if n else 0.0,
        "encoded/no_nan_rate": no_nan / n if n else 0.0,
        "encoded/no_inf_rate": no_inf / n if n else 0.0,
    }

    # Sample-size flags (Hole B).
    sample_sufficient = {
        "route": (
            total_map_actions >= MIN_MAP_ACTIONS_FOR_PASS
            and map_steps >= MIN_MAP_STEPS_FOR_PASS
        ),
        "deck_unique": unique_n >= MIN_UNIQUE_CARDS_FOR_PASS,
    }

    threshold_map = {
        # Deck
        "deck/present_rate": THRESHOLD,
        "deck/card_id_present_rate": THRESHOLD,
        "deck/metadata_hit_rate": SOFT_THRESHOLD,
        "deck/upgrade_present_rate": SOFT_THRESHOLD,
        "deck/cost_present_rate": THRESHOLD,
        "deck/type_present_rate": THRESHOLD,
        "deck/unique_metadata_hit_rate": SOFT_THRESHOLD,
        "deck/unique_cost_present_rate": THRESHOLD,
        "deck/unique_type_present_rate": THRESHOLD,
        # Route
        "route/summary_present_rate": THRESHOLD,
        "route/full_20_key_present_rate": THRESHOLD,
        "route/nodes_present_rate": THRESHOLD,
        # Floor / long-horizon. Note: terminated_present_rate and
        # truncated_present_rate are reported but NOT gated — they are
        # mutually-exclusive episode-end modes (an episode either
        # naturally ends or hits the step cap), so neither should
        # individually be ≥0.98. The actually-load-bearing check is
        # ``episode_ended_rate`` (terminated OR truncated) which must be
        # 1.0 for long-horizon target masking to work.
        "floor/current_floor_present_rate": THRESHOLD,
        "floor/act_id_present_rate": THRESHOLD,
        "floor/episode_ended_rate": THRESHOLD,
        "floor/episode_max_floor_reconstructable_rate": THRESHOLD,
        "floor/current_floor_monotonic_or_valid_rate": THRESHOLD,
        # Encoded
        "encoded/schema_match_rate": THRESHOLD,
        "encoded/action_mask_valid_rate": THRESHOLD,
        "encoded/no_nan_rate": THRESHOLD,
        "encoded/no_inf_rate": THRESHOLD,
    }

    passes: dict[str, dict[str, Any]] = {}
    for key, threshold in threshold_map.items():
        rate_value = rates.get(key, 0.0)
        # Hole B: enforce sample-size guard for route/* and unique deck rates.
        bucket = "route" if key.startswith("route/") else (
            "deck_unique" if key.startswith("deck/unique_") else None
        )
        threshold_met = bool(rate_value >= threshold)
        if bucket is not None and not sample_sufficient[bucket]:
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
        "episodes": n_eps,
        "steps": n,
        "rates": rates,
        "thresholds": threshold_map,
        "passes": passes,
        "all_pass": all(info["pass"] for info in passes.values()),
        "sample_sufficient": sample_sufficient,
        "phase_distribution": dict(phase_counter.most_common()),
        "decision_domain_distribution": dict(domain_counter.most_common()),
        "total_cards_observed": int(total_cards),
        "total_map_actions_observed": int(total_map_actions),
        "total_map_steps_observed": int(map_steps),
        "unique_card_id_count": int(unique_n),
        "missing_metadata_unique_ids": missing_meta_ids,
        "route_missing_key_counts": dict(aggregate_missing_keys),
        "min_map_actions_for_pass": MIN_MAP_ACTIONS_FOR_PASS,
        "min_map_steps_for_pass": MIN_MAP_STEPS_FOR_PASS,
        "min_unique_cards_for_pass": MIN_UNIQUE_CARDS_FOR_PASS,
    }


# ---------------------------------------------------------------------------
# Driver
# ---------------------------------------------------------------------------

def random_valid_action(action_mask: np.ndarray, rng: np.random.Generator) -> int:
    valid = np.flatnonzero(np.asarray(action_mask) > 0)
    if valid.size == 0:
        return 0
    return int(rng.choice(valid))


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--session-file", default=None)
    parser.add_argument("--character", default="ironclad")
    parser.add_argument("--seed", default=None)
    parser.add_argument("--episodes", type=int, default=5)
    parser.add_argument("--max-steps-per-episode", type=int, default=300)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--rng-seed", type=int, default=42)
    parser.add_argument("--reset-timeout-ms", type=int, default=60000)
    parser.add_argument("--step-timeout-ms", type=int, default=20000)
    args = parser.parse_args()

    output_dir = resolve_artifact_path(args.output_dir)
    args.session_file = (
        str(resolve_external_input_path(args.session_file))
        if args.session_file
        else None
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    audit_path = output_dir / "audit.jsonl"
    summary_path = output_dir / "summary.json"

    rng = np.random.default_rng(args.rng_seed)
    encoder = WorldTokenObservationEncoder(use_text=False)
    seed_pool = [args.seed] if args.seed else []
    env = SlayTheSpire2EnvV2(
        session_file=args.session_file,
        character=args.character,
        defensive_buffs=False,
        reset_timeout_ms=args.reset_timeout_ms,
        step_timeout_ms=args.step_timeout_ms,
        obs_encoder=encoder,
        seed_pool=seed_pool,
        seed_strategy="round_robin",
    )

    records: list[dict[str, Any]] = []
    episode_summaries: list[dict[str, Any]] = []
    unique_cards: dict[str, dict[str, Any]] = {}
    encoded_template: dict[str, tuple] | None = None
    started_wall = time.time()
    fatal_exception: str | None = None

    try:
        with audit_path.open("w", encoding="utf-8") as fh:
            for ep_idx in range(args.episodes):
                ep_start = time.time()
                try:
                    obs_encoded, info = env.reset()
                except Exception as exc:
                    fatal_exception = (
                        f"env.reset() failed on episode {ep_idx}: {type(exc).__name__}: {exc}"
                    )
                    print(f"[audit] {fatal_exception}; aborting further episodes", flush=True)
                    break
                episode_steps = 0
                terminated = False
                truncated = False
                terminated_seen = False
                truncated_seen = False
                floor_trace: list[int] = []
                episode_max_floor = 0
                episode_action_ids: list[int] = []
                was_full_run = True  # phase3 launch is always full_run

                while episode_steps < args.max_steps_per_episode:
                    raw_obs = env._last_obs_raw or {}
                    legal = env._legal_actions or []
                    if encoded_template is None and isinstance(obs_encoded, dict):
                        encoded_template = build_encoded_template(obs_encoded)
                    record = audit_step(raw_obs, legal, obs_encoded, unique_cards, encoded_template)
                    record["_episode_index"] = ep_idx
                    record["_step_index"] = episode_steps
                    records.append(record)
                    fh.write(json.dumps(record, ensure_ascii=False, default=str) + "\n")
                    fh.flush()

                    cur_floor = record["floor"]["current_floor_value"]
                    floor_trace.append(cur_floor)
                    episode_max_floor = max(episode_max_floor, cur_floor)

                    if terminated or truncated:
                        break

                    action_mask = obs_encoded.get("action_mask")
                    if action_mask is None:
                        break
                    action = random_valid_action(action_mask, rng)
                    episode_action_ids.append(action)
                    try:
                        obs_encoded, _reward, terminated, truncated, info = env.step(action)
                    except Exception as exc:
                        print(f"[audit] env.step raised: {exc}; ending episode {ep_idx}", flush=True)
                        break
                    if terminated is True:
                        terminated_seen = True
                    if truncated is True:
                        truncated_seen = True
                    episode_steps += 1

                # Hole F: monotonicity check — current_floor should never strictly
                # decrease within an episode. Allow noise == equal floors.
                monotonic_or_valid = all(
                    floor_trace[i] <= floor_trace[i + 1]
                    for i in range(len(floor_trace) - 1)
                ) if floor_trace else True

                # max_floor_reconstructable: at least one step had current_floor>=1.
                max_floor_reconstructable = episode_max_floor > 0

                ep_elapsed = time.time() - ep_start
                ep_summary = {
                    "episode_index": ep_idx,
                    "steps": episode_steps,
                    "terminated": bool(terminated),
                    "truncated": bool(truncated),
                    "terminated_present": bool(terminated_seen or terminated),
                    "truncated_present": bool(truncated_seen or truncated),
                    "max_floor": int(episode_max_floor),
                    "max_floor_reconstructable": bool(max_floor_reconstructable),
                    "floor_monotonic_or_valid": bool(monotonic_or_valid),
                    "was_full_run": was_full_run,
                    "wall_seconds": float(ep_elapsed),
                }
                episode_summaries.append(ep_summary)
                print(
                    f"[audit] episode {ep_idx}: steps={episode_steps} max_floor={episode_max_floor} "
                    f"term={terminated} trunc={truncated} monotonic={monotonic_or_valid} "
                    f"wall={ep_elapsed:.1f}s",
                    flush=True,
                )
    except Exception as exc:
        fatal_exception = f"audit loop failed: {type(exc).__name__}: {exc}"
        print(f"[audit] {fatal_exception}", flush=True)
    finally:
        try:
            env.close()
        except Exception:
            pass

    # Always write the summary, even if the audit died mid-loop. Partial
    # data is still useful for diagnosing bridge stability + schema drift.
    try:
        summary = aggregate(records, episode_summaries, unique_cards)
    except Exception as exc:
        summary = {
            "aggregate_failed": True,
            "aggregate_error": f"{type(exc).__name__}: {exc}",
        }
    summary["episodes_run"] = len(episode_summaries)
    summary["episode_summaries"] = episode_summaries
    summary["wall_seconds_total"] = float(time.time() - started_wall)
    summary["encoded_schema_template"] = (
        {k: {"shape": list(s), "kind": kind} for k, (s, kind) in (encoded_template or {}).items()}
    )
    if fatal_exception:
        summary["fatal_exception"] = fatal_exception
        summary["partial"] = True

    summary_path.write_text(
        json.dumps(summary, ensure_ascii=False, indent=2, default=str),
        encoding="utf-8",
    )

    print()
    print(f"[audit] Wrote {audit_path}")
    print(f"[audit] Wrote {summary_path}")
    print()
    print("=== Summary ===")
    print(
        f"episodes={summary.get('episodes_run', 0)}  steps={summary.get('steps', 0)}  "
        f"unique_cards={summary.get('unique_card_id_count', 0)}  "
        f"map_actions={summary.get('total_map_actions_observed', 0)}  "
        f"map_steps={summary.get('total_map_steps_observed', 0)}"
    )
    sample_suff = summary.get("sample_sufficient", {})
    print(f"sample_sufficient: route={sample_suff.get('route')} deck_unique={sample_suff.get('deck_unique')}")
    if summary.get("missing_metadata_unique_ids"):
        print(f"missing_metadata_ids: {summary['missing_metadata_unique_ids'][:10]}{'...' if len(summary['missing_metadata_unique_ids'])>10 else ''}")
    if summary.get("route_missing_key_counts"):
        print(f"route_missing_keys: {dict(list(summary['route_missing_key_counts'].items())[:5])}")
    print()
    rates = summary.get("rates", {})
    passes = summary.get("passes", {})
    for key in sorted(passes.keys()):
        info = passes[key]
        rate = rates.get(key, 0.0)
        print(f"  {key:60s} {rate:6.4f}  {info['status']}")
    print()
    print(f"all_pass={summary.get('all_pass', False)}")

    return 0 if summary.get("all_pass", False) else 1


if __name__ == "__main__":
    raise SystemExit(main())
