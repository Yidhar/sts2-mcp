#!/usr/bin/env python3
"""Audit snapshot-level human/local samples for deck-quality survival signals.

This is intentionally *not* a behaviour-cloning trainer.  The bootstrap
``human_zip`` rows are combat/start snapshots with deck/relic/HP/outcome
metadata; most of them do not contain ordered legal actions or selected
action ids.  They are therefore suitable for deck-quality calibration,
survival/remaining-floor targets, curriculum design, and death-deck reports,
but not for root-policy CE.

Example:

    python scripts/audit_human_zip_deck_survival.py \
      --input analysis/bootstrap_human_plus_local_act1clear.no_combat_reset_failed_rows.jsonl \
      --input analysis/bootstrap_human_plus_local_act1clear.combat_reset_failed_rows.jsonl \
      --out tmp/human_zip_deck_survival_audit.json
"""
from __future__ import annotations

import argparse
import json
import math
import sys
from collections import Counter, defaultdict
from pathlib import Path
from statistics import mean
from typing import Any, Iterable

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from sts2_env.deck_quality import deck_quality_v2
from sts2_rl.artifacts import resolve_artifact_path, resolve_external_input_path

DEFAULT_INPUTS = (
    "analysis/bootstrap_human_plus_local_act1clear.no_combat_reset_failed_rows.jsonl",
    "analysis/bootstrap_human_plus_local_act1clear.combat_reset_failed_rows.jsonl",
)

FEATURES = (
    "deck_size_raw",
    "metadata_hit_rate",
    "raw_avg_damage_per_energy",
    "raw_avg_block_per_energy",
    "raw_expected_cards_seen_per_turn",
    "raw_expected_playable_cards_per_turn",
    "raw_expected_playable_attack_damage_per_turn",
    "raw_expected_playable_block_per_turn",
    "expected_energy_utilization_score",
    "expected_playable_frontload_score",
    "expected_playable_block_score",
    "expected_hand_useful_quality_score",
    "frontload_score",
    "block_score",
    "scaling_score",
    "draw_engine_score",
    "energy_engine_score",
    "pollution_score",
    "consistency_score",
    "elite_readiness_score",
    "boss_readiness_score",
    "combo_option_value_score",
    "combo_unmet_dependency_score",
    "delayed_payoff_option_value_score",
    "delayed_payoff_unrealized_risk_score",
)


def _safe_float(value: Any, default: float = 0.0) -> float:
    try:
        out = float(value)
    except (TypeError, ValueError):
        return default
    return out if math.isfinite(out) else default


def _bucket_floor(floor: Any) -> str:
    value = int(_safe_float(floor, 0.0))
    if value <= 0:
        return "unknown"
    if value <= 3:
        return "01-03"
    if value <= 7:
        return "04-07"
    if value <= 11:
        return "08-11"
    if value <= 15:
        return "12-15"
    return "16+"


def _as_bool(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        return bool(value)
    if isinstance(value, str):
        return value.strip().lower() in {"1", "true", "yes", "y"}
    return False


def _expand_deck_before(deck_before: Any) -> list[dict[str, Any]]:
    """Expand aggregated ``deck_before.cards`` into deck_quality_v2 cards."""
    if not isinstance(deck_before, dict):
        return []
    cards = deck_before.get("cards")
    if not isinstance(cards, list):
        return []
    expanded: list[dict[str, Any]] = []
    for entry in cards:
        if not isinstance(entry, dict):
            continue
        card_id = str(entry.get("id") or "").strip()
        if not card_id:
            continue
        count = max(int(_safe_float(entry.get("count"), 0.0)), 0)
        upgraded_count = max(int(_safe_float(entry.get("upgraded_count"), 0.0)), 0)
        max_upgrade_level = max(int(_safe_float(entry.get("max_upgrade_level"), 1.0)), 1)
        for idx in range(count):
            upgrade_level = max_upgrade_level if idx < upgraded_count else 0
            expanded.append(
                {
                    "id": card_id,
                    "upgrade_level": upgrade_level,
                    "upgraded": upgrade_level > 0,
                }
            )
    return expanded


class _Group:
    def __init__(self) -> None:
        self.rows = 0
        self.act1_clear = 0
        self.run_win = 0
        self.hp_ratio_sum = 0.0
        self.max_act_sum = 0.0
        self.feature_sums: dict[str, float] = defaultdict(float)
        self.encounters: Counter[str] = Counter()
        self.killed_by: Counter[str] = Counter()

    def add(self, row: dict[str, Any], features: dict[str, float]) -> None:
        self.rows += 1
        self.act1_clear += int(_as_bool(row.get("cleared_act1")))
        self.run_win += int(_as_bool(row.get("source_run_win")))
        self.hp_ratio_sum += _safe_float(row.get("snapshot_hp_ratio"), 0.0)
        self.max_act_sum += _safe_float(row.get("max_act_index"), 0.0)
        encounter = str(row.get("encounter_id") or row.get("room_model_id") or "unknown")
        killed_by = str(row.get("source_killed_by_encounter") or "none")
        self.encounters[encounter] += 1
        self.killed_by[killed_by] += 1
        for key in FEATURES:
            self.feature_sums[key] += _safe_float(features.get(key), 0.0)

    def to_json(self) -> dict[str, Any]:
        denom = max(self.rows, 1)
        return {
            "rows": self.rows,
            "act1_clear_rate": self.act1_clear / denom,
            "source_run_win_rate": self.run_win / denom,
            "snapshot_hp_ratio_mean": self.hp_ratio_sum / denom,
            "max_act_index_mean": self.max_act_sum / denom,
            "feature_means": {key: self.feature_sums[key] / denom for key in FEATURES},
            "top_encounters": self.encounters.most_common(12),
            "top_killed_by": self.killed_by.most_common(12),
        }


def _load_rows(paths: Iterable[Path]) -> Iterable[tuple[Path, dict[str, Any]]]:
    for path in paths:
        with path.open("r", encoding="utf-8") as handle:
            for line_no, line in enumerate(handle, 1):
                line = line.strip()
                if not line:
                    continue
                try:
                    row = json.loads(line)
                except json.JSONDecodeError as exc:
                    raise SystemExit(f"{path}:{line_no}: invalid json: {exc}") from exc
                if isinstance(row, dict):
                    yield path, row


def _delta(a: dict[str, float], b: dict[str, float]) -> dict[str, float]:
    return {key: _safe_float(a.get(key)) - _safe_float(b.get(key)) for key in FEATURES}


def build_report(paths: list[Path], *, min_group_rows: int = 10) -> dict[str, Any]:
    groups: dict[str, _Group] = defaultdict(_Group)
    by_clear_features: dict[bool, dict[str, list[float]]] = {
        True: defaultdict(list),
        False: defaultdict(list),
    }
    rows = 0
    source_counts: Counter[str] = Counter()
    quality_counts: Counter[str] = Counter()

    for path, row in _load_rows(paths):
        rows += 1
        cards = _expand_deck_before(row.get("deck_before"))
        features = deck_quality_v2(cards)
        clear = _as_bool(row.get("cleared_act1"))
        source = str(row.get("provenance_origin") or "unknown")
        quality = str(row.get("quality_coarse") or "unknown")
        floor_bucket = _bucket_floor(row.get("floor_number"))
        source_counts[source] += 1
        quality_counts[quality] += 1

        group_keys = (
            "all",
            f"source:{source}",
            f"quality:{quality}",
            f"act1_clear:{int(clear)}",
            f"floor_bucket:{floor_bucket}",
            f"source:{source}|floor_bucket:{floor_bucket}",
            f"quality:{quality}|floor_bucket:{floor_bucket}",
        )
        for key in group_keys:
            groups[key].add(row, features)
        for key in FEATURES:
            by_clear_features[clear][key].append(_safe_float(features.get(key), 0.0))

    serialised_groups = {
        key: group.to_json()
        for key, group in sorted(groups.items())
        if group.rows >= min_group_rows or key in {"all", "act1_clear:0", "act1_clear:1"}
    }
    clear_means = {
        str(clear): {
            key: (mean(values) if values else 0.0)
            for key, values in feature_lists.items()
        }
        for clear, feature_lists in by_clear_features.items()
    }
    return {
        "inputs": [str(path) for path in paths],
        "rows": rows,
        "source_counts": source_counts.most_common(),
        "quality_counts": quality_counts.most_common(),
        "groups": serialised_groups,
        "act1_clear_feature_delta": _delta(clear_means.get("True", {}), clear_means.get("False", {})),
        "act1_clear_feature_means": clear_means,
        "notes": [
            "Rows are snapshot-level; do not use this report as action-policy CE labels.",
            "Positive act1_clear_feature_delta means the feature is higher in Act1-clear snapshots.",
            "Use this for deck-quality calibration, survival targets, curriculum buckets, and death-deck diagnostics.",
        ],
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input",
        action="append",
        default=None,
        help="Input jsonl path. Can be repeated. Defaults to bootstrap human/local Act1 files.",
    )
    parser.add_argument("--out", default=None, help="Optional JSON output path.")
    parser.add_argument("--min-group-rows", type=int, default=10)
    args = parser.parse_args()

    paths = [resolve_external_input_path(p) for p in (args.input or DEFAULT_INPUTS)]
    missing = [str(path) for path in paths if not path.exists()]
    if missing:
        raise SystemExit(f"missing input(s): {missing}")

    report = build_report(paths, min_group_rows=max(int(args.min_group_rows), 1))
    if args.out:
        out = resolve_artifact_path(args.out)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")

    all_group = report["groups"].get("all", {})
    clear_group = report["groups"].get("act1_clear:1", {})
    fail_group = report["groups"].get("act1_clear:0", {})
    print(json.dumps(
        {
            "rows": report["rows"],
            "source_counts": report["source_counts"],
            "quality_counts": report["quality_counts"],
            "all": {
                "act1_clear_rate": all_group.get("act1_clear_rate"),
                "hp_ratio_mean": all_group.get("snapshot_hp_ratio_mean"),
                "max_act_index_mean": all_group.get("max_act_index_mean"),
            },
            "act1_clear_rows": clear_group.get("rows", 0),
            "act1_fail_rows": fail_group.get("rows", 0),
            "top_positive_clear_deltas": sorted(
                report["act1_clear_feature_delta"].items(),
                key=lambda item: item[1],
                reverse=True,
            )[:8],
            "top_negative_clear_deltas": sorted(
                report["act1_clear_feature_delta"].items(),
                key=lambda item: item[1],
            )[:8],
            "out": args.out,
        },
        ensure_ascii=False,
        indent=2,
    ))


if __name__ == "__main__":
    main()
