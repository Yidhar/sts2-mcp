from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path
from typing import Any

from content_registry import humanize_game_id
from offline_dataset_loader import load_dataset
from sts2_rl.artifacts import resolve_external_input_path
from sts2_rl.game_data import repository_root, resolve_generated_game_data_output

_CURSE_CARD_TOKENS = {
    "ASCENDERS_BANE",
    "CLUMSY",
    "CURSE_OF_THE_BELL",
    "DECAY",
    "DEBT",
    "DOUBT",
    "INJURY",
    "NECRONOMICURSE",
    "NORMALITY",
    "PAIN",
    "PARASITE",
    "REGRET",
    "SHAME",
    "WRITHE",
}
_STATUS_CARD_TOKENS = {
    "BURN",
    "DAZED",
    "SLIMED",
    "VOID",
    "WOUND",
}
_TASK_SPECS: tuple[tuple[str, str, str], ...] = (
    ("regular_card_reward", "regular_card_reward_samples", "reward"),
    ("event_card_bundle", "event_card_bundle_samples", "reward"),
    ("smith_target", "smith_target_samples", "smith"),
    ("remove_card_step", "remove_card_step_samples", "remove"),
    ("shop_remove_target_step", "shop_remove_target_step_samples", "shop_remove"),
    ("transform_card_step", "transform_card_step_samples", "transform"),
)
_PRIOR_KEYS = ("reward", "smith", "remove", "shop_remove", "remove_any", "transform")
_MIN_SUPPORT_FOR_BUCKET = {
    "reward": 5,
    "smith": 5,
    "remove": 5,
    "shop_remove": 4,
    "remove_any": 5,
    "transform": 3,
    "keep": 5,
}
_DEFAULT_OUTPUT = repository_root() / "game-data" / "generated" / "cards.generated.json"


def _card_tail(card_id: str | None) -> str:
    return str(card_id or "").strip().split(".", 1)[-1]


def _is_starter_strike(card_id: str | None) -> bool:
    return _card_tail(card_id).startswith("STRIKE_")


def _is_starter_defend(card_id: str | None) -> bool:
    return _card_tail(card_id).startswith("DEFEND_")


def _is_ascenders_bane(card_id: str | None) -> bool:
    return _card_tail(card_id) == "ASCENDERS_BANE"


def _is_curse_card(card_id: str | None) -> bool:
    return _card_tail(card_id) in _CURSE_CARD_TOKENS


def _is_status_card(card_id: str | None) -> bool:
    tail = _card_tail(card_id)
    return tail in _STATUS_CARD_TOKENS or tail.startswith("SLIME")


def _role_tag(card_id: str) -> str:
    if _is_starter_strike(card_id):
        return "starter_attack"
    if _is_starter_defend(card_id):
        return "starter_defend"
    if _is_ascenders_bane(card_id):
        return "bane"
    if _is_curse_card(card_id):
        return "curse"
    if _is_status_card(card_id):
        return "status"
    return "nonstarter"


def _bucket_from_percentile(percentile: float | None) -> str:
    if percentile is None:
        return "unknown"
    if percentile >= 0.90:
        return "very_high"
    if percentile >= 0.70:
        return "high"
    if percentile <= 0.10:
        return "very_low"
    if percentile <= 0.30:
        return "low"
    return "medium"


def _round(value: float | None, digits: int = 4) -> float | None:
    if value is None:
        return None
    return round(float(value), digits)


def _smoothed_rate(*, picked: int, shown: int, prior_mean: float, prior_strength: float = 8.0) -> float:
    if shown <= 0:
        return float(prior_mean)
    return float((picked + prior_strength * prior_mean) / (shown + prior_strength))


def _percentiles(values: dict[str, float]) -> dict[str, float]:
    if not values:
        return {}
    ordered = sorted(values.items(), key=lambda item: (item[1], item[0]))
    if len(ordered) == 1:
        return {ordered[0][0]: 0.5}
    max_rank = float(len(ordered) - 1)
    return {
        card_id: rank / max_rank
        for rank, (card_id, _value) in enumerate(ordered)
    }


def _summary_parts(*parts: str) -> str:
    return ". ".join(part for part in parts if part)


def _prior_part(label: str, bucket: str) -> str:
    if not bucket or bucket == "unknown":
        return ""
    return f"{label}={bucket}"


def _choose_bucket(card: dict[str, Any], key: str) -> str:
    priors = card.get("priors") or {}
    info = priors.get(key) or {}
    return str(info.get("bucket") or "unknown")


def _build_generic_summary(card_id: str, metadata: dict[str, Any]) -> str:
    keep_bucket = str((metadata.get("keep") or {}).get("bucket") or "unknown")
    return _summary_parts(
        f"role={_role_tag(card_id)}",
        _prior_part("keep", keep_bucket),
        _prior_part("reward", _choose_bucket(metadata, "reward")),
        _prior_part("smith", _choose_bucket(metadata, "smith")),
        _prior_part("remove", _choose_bucket(metadata, "remove_any")),
        _prior_part("transform", _choose_bucket(metadata, "transform")),
    )


def _build_task_summaries(card_id: str, metadata: dict[str, Any]) -> dict[str, str]:
    role = f"role={_role_tag(card_id)}"
    keep_part = _prior_part("keep", str((metadata.get("keep") or {}).get("bucket") or "unknown"))
    remove_part = _prior_part("remove", _choose_bucket(metadata, "remove_any"))
    shop_remove_bucket = _choose_bucket(metadata, "shop_remove")
    if shop_remove_bucket == "unknown":
        shop_remove_bucket = _choose_bucket(metadata, "remove_any")
    return {
        "smith_target": _summary_parts(
            role,
            keep_part,
            _prior_part("smith", _choose_bucket(metadata, "smith")),
            remove_part,
            _prior_part("reward", _choose_bucket(metadata, "reward")),
        ),
        "remove_card_step": _summary_parts(
            role,
            keep_part,
            remove_part,
            _prior_part("smith", _choose_bucket(metadata, "smith")),
            _prior_part("reward", _choose_bucket(metadata, "reward")),
        ),
        "shop_remove_target_step": _summary_parts(
            role,
            keep_part,
            _prior_part("shop_remove", shop_remove_bucket),
            _prior_part("smith", _choose_bucket(metadata, "smith")),
            _prior_part("reward", _choose_bucket(metadata, "reward")),
        ),
        "transform_card_step": _summary_parts(
            role,
            keep_part,
            _prior_part("transform", _choose_bucket(metadata, "transform")),
            remove_part,
            _prior_part("reward", _choose_bucket(metadata, "reward")),
        ),
    }


def build_card_registry(
    dataset_root: str | Path,
    *,
    dataset_format: str = "parquet",
    partition_kind: str | None = None,
    partition_value: str | None = None,
) -> dict[str, dict[str, Any]]:
    stats: dict[str, dict[str, int]] = defaultdict(lambda: defaultdict(int))
    seen_deck_groups: set[tuple[str, str]] = set()

    for task_name, dataset_name, prior_key in _TASK_SPECS:
        dataset = load_dataset(
            dataset_root,
            dataset_name,
            fmt=dataset_format,
            partition_kind=partition_kind,
            partition_value=partition_value,
        )
        for index in range(len(dataset)):
            row = dataset[index]
            option_ids = [str(value) for value in row.get("option_ids") or [] if value]
            label_id = str(row.get("label_id") or "")
            for card_id in option_ids:
                bucket = stats[card_id]
                bucket[f"{prior_key}_shown"] += 1
                if prior_key in {"remove", "shop_remove"}:
                    bucket["remove_any_shown"] += 1

            if label_id and label_id != "<skip>":
                bucket = stats[label_id]
                bucket[f"{prior_key}_picked"] += 1
                if prior_key in {"remove", "shop_remove"}:
                    bucket["remove_any_picked"] += 1

            if task_name == "transform_card_step":
                for item in row.get("transforms") or []:
                    if not isinstance(item, dict):
                        continue
                    original_card = (item.get("original_card") or {}).get("id")
                    final_card = (item.get("final_card") or {}).get("id")
                    if original_card:
                        stats[str(original_card)]["transform_original_count"] += 1
                    if final_card:
                        stats[str(final_card)]["transform_into_count"] += 1

            choice_group_id = str(row.get("choice_group_id") or row.get("sample_id") or f"{dataset_name}:{index}")
            deck_group_key = (dataset_name, choice_group_id)
            if deck_group_key in seen_deck_groups:
                continue
            seen_deck_groups.add(deck_group_key)
            deck_before = row.get("deck_before") or {}
            for card in deck_before.get("cards") or []:
                if not isinstance(card, dict):
                    continue
                card_id = str(card.get("id") or "")
                if not card_id:
                    continue
                bucket = stats[card_id]
                bucket["deck_group_count"] += 1
                bucket["copies_total"] += max(int(card.get("count") or 0), 0)
                bucket["upgraded_copies_total"] += max(int(card.get("upgraded_count") or 0), 0)
                bucket["max_upgrade_level_seen"] = max(
                    int(bucket.get("max_upgrade_level_seen", 0)),
                    int(card.get("max_upgrade_level") or 0),
                )

    prior_means: dict[str, float] = {}
    for key in _PRIOR_KEYS:
        shown_total = sum(card.get(f"{key}_shown", 0) for card in stats.values())
        picked_total = sum(card.get(f"{key}_picked", 0) for card in stats.values())
        prior_means[key] = (picked_total / shown_total) if shown_total > 0 else 0.0

    prior_percentiles: dict[str, dict[str, float]] = {}
    for key in _PRIOR_KEYS:
        supported_values = {
            card_id: _smoothed_rate(
                picked=card.get(f"{key}_picked", 0),
                shown=card.get(f"{key}_shown", 0),
                prior_mean=prior_means[key],
            )
            for card_id, card in stats.items()
            if card.get(f"{key}_shown", 0) >= _MIN_SUPPORT_FOR_BUCKET.get(key, 5)
        }
        prior_percentiles[key] = _percentiles(supported_values)

    keep_scores: dict[str, float] = {}
    for card_id, card in stats.items():
        support_total = sum(card.get(f"{key}_shown", 0) for key in ("reward", "smith", "remove_any", "transform"))
        if support_total < _MIN_SUPPORT_FOR_BUCKET["keep"]:
            continue
        reward_rate = _smoothed_rate(
            picked=card.get("reward_picked", 0),
            shown=card.get("reward_shown", 0),
            prior_mean=prior_means["reward"],
        )
        smith_rate = _smoothed_rate(
            picked=card.get("smith_picked", 0),
            shown=card.get("smith_shown", 0),
            prior_mean=prior_means["smith"],
        )
        remove_rate = _smoothed_rate(
            picked=card.get("remove_any_picked", 0),
            shown=card.get("remove_any_shown", 0),
            prior_mean=prior_means["remove_any"],
        )
        transform_rate = _smoothed_rate(
            picked=card.get("transform_picked", 0),
            shown=card.get("transform_shown", 0),
            prior_mean=prior_means["transform"],
        )
        keep_scores[card_id] = 1.2 * reward_rate + 1.0 * smith_rate - 1.3 * remove_rate - 0.8 * transform_rate

    keep_percentiles = _percentiles(keep_scores)

    output: dict[str, dict[str, Any]] = {}
    for card_id in sorted(stats):
        card = stats[card_id]
        priors: dict[str, Any] = {}
        for key in _PRIOR_KEYS:
            shown = int(card.get(f"{key}_shown", 0))
            picked = int(card.get(f"{key}_picked", 0))
            rate = _smoothed_rate(picked=picked, shown=shown, prior_mean=prior_means[key])
            percentile = prior_percentiles[key].get(card_id)
            priors[key] = {
                "shown": shown,
                "picked": picked,
                "rate": _round(rate),
                "bucket": _bucket_from_percentile(percentile) if shown >= _MIN_SUPPORT_FOR_BUCKET.get(key, 5) else "unknown",
            }

        deck_group_count = int(card.get("deck_group_count", 0))
        copies_total = int(card.get("copies_total", 0))
        upgraded_copies_total = int(card.get("upgraded_copies_total", 0))
        avg_copies = (copies_total / deck_group_count) if deck_group_count > 0 else None
        upgraded_share = (upgraded_copies_total / copies_total) if copies_total > 0 else None
        keep_score = keep_scores.get(card_id)
        keep_bucket = _bucket_from_percentile(keep_percentiles.get(card_id)) if card_id in keep_scores else "unknown"

        metadata = {
            "title": humanize_game_id(card_id),
            "tags": [
                _role_tag(card_id),
                *(["duplicate_friendly"] if avg_copies is not None and avg_copies >= 1.5 else []),
                *([f"keep_{keep_bucket}"] if keep_bucket != "unknown" else []),
            ],
            "priors": priors,
            "keep": {
                "score": _round(keep_score),
                "bucket": keep_bucket,
                "support": int(sum(card.get(f"{key}_shown", 0) for key in ("reward", "smith", "remove_any", "transform"))),
            },
            "observations": {
                "deck_group_count": deck_group_count,
                "avg_copies": _round(avg_copies),
                "upgraded_share": _round(upgraded_share),
                "max_upgrade_level_seen": int(card.get("max_upgrade_level_seen", 0)),
                "transform_into_count": int(card.get("transform_into_count", 0)),
            },
        }
        metadata["summary"] = _build_generic_summary(card_id, metadata)
        metadata["task_summaries"] = _build_task_summaries(card_id, metadata)
        output[card_id] = metadata
    return output


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate card semantic priors for offline STS2 build training.")
    parser.add_argument("--dataset-root", required=True, type=str)
    parser.add_argument("--dataset-format", default="parquet", choices=["parquet", "jsonl"])
    parser.add_argument("--partition-kind", default=None, choices=[None, "build_id", "build_family"])
    parser.add_argument("--partition-value", default=None, type=str)
    parser.add_argument("--output", default=str(_DEFAULT_OUTPUT), type=str)
    args = parser.parse_args()

    args.dataset_root = str(resolve_external_input_path(args.dataset_root))

    output = build_card_registry(
        args.dataset_root,
        dataset_format=args.dataset_format,
        partition_kind=args.partition_kind,
        partition_value=args.partition_value,
    )
    output_path = resolve_generated_game_data_output(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_bytes(
        (json.dumps(output, ensure_ascii=False, indent=2, sort_keys=True) + "\n").encode("utf-8")
    )
    print(f"wrote {len(output)} card semantic entries -> {output_path}")


if __name__ == "__main__":
    main()
