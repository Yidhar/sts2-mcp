"""Offline multi-task pretraining directly on the live online policy.

This stage reuses the same policy/checkpoint lineage as `train_v2.py` so
combat sandbox PPO -> offline build/route pretrain -> full-run PPO can share
one weight file.
"""

from __future__ import annotations

import argparse
import json
import random
import time
from collections import Counter
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F
from gymnasium import Env, spaces
from sb3_contrib import MaskablePPO
from stable_baselines3.common.vec_env import DummyVecEnv
from torch.utils.data import DataLoader

from offline_training_data import (
    ACTION_ONLY_CLASS_TASKS,
    ACTION_ONLY_CARD_TASKS,
    AUXILIARY_TASKS,
    BUILD_V2_CANDIDATE_TASKS,
    BUILD_V2_CLASS_TASKS,
    ROUTE_TASKS,
    SUPERVISED_TASKS,
    OfflineRowsDataset,
    load_task_rows,
)
from content_registry import (
    build_candidate_semantic_text,
    build_card_label,
    build_card_semantic_text,
    build_entity_text,
    get_card_metadata,
    humanize_game_id,
    summarize_entity_ids,
)
from sts2_env.checkpoint import (
    load_online_checkpoint_metadata,
    load_online_policy_state_dict,
    save_online_checkpoint,
)
from sts2_env.model import STS2CandidateScoringPolicy
from sts2_env.observation_v2 import DictObservationEncoder, MAX_ACTIONS


DEFAULT_TASKS = [
    "regular_card_reward",
    "event_card_bundle",
    "ancient_choice",
    "relic_choice_step",
    "potion_choice_step",
    "rest_action",
    "smith_target",
    "remove_card_step",
    "transform_card_step",
    "shop_relic_pick_step",
    "shop_potion_pick_step",
    "shop_remove_binary",
    "shop_remove_target_step",
    "shop_bundle_aux",
]

ROUTE_SUPERVISION_ERROR = (
    "Offline route pretraining now requires full candidate route supervision "
    "(route_candidates + label_index). Native .run history exports only chosen_path_only "
    "route samples, which teach room-type priors instead of real map topology. "
    "Remove route_* tasks from --tasks, or regenerate the dataset with per-candidate map actions."
)
REST_SITE_FALLBACK_ACTIONS = ["rest", "smith"]
SHOP_REMOVE_BINARY_CANDIDATES = ["remove_card", "skip_remove"]
SHOP_BUNDLE_AUX_ACTIONS = (
    ("did_buy_any_card", "shop_bundle_aux:buy_any_card", "shop"),
    ("did_buy_any_relic", "shop_bundle_aux:buy_any_relic", "shop"),
    ("did_buy_any_potion", "shop_bundle_aux:buy_any_potion", "shop"),
    ("did_remove_card", "shop_bundle_aux:remove_card", "shop"),
    ("leave_only", "shop_bundle_aux:leave_only", "proceed"),
)

SCALAR_FLOOR = 0
SCALAR_ACT = 1
SCALAR_PATH = 2
SCALAR_ASCENSION = 3
SCALAR_HP_BEFORE = 4
SCALAR_CURRENT_HP = 5
SCALAR_MAX_HP = 6
SCALAR_GOLD_BEFORE = 9
SCALAR_CURRENT_GOLD = 10
SCALAR_TURNS = 11


@dataclass
class OfflineOnlineConfig:
    tasks: list[str]
    dataset_root: str
    dataset_format: str = "parquet"
    partition_kind: str | None = None
    partition_value: str | None = None
    out_dir: str = "offline_runs"
    run_name: str | None = None
    init_checkpoint: str | None = None
    batch_size: int = 64
    epochs: int = 20
    learning_rate: float = 1e-4
    weight_decay: float = 1e-4
    device: str = "cpu"
    seed: int = 7
    num_workers: int = 0
    eval_split: str = "val"
    max_train_batches: int | None = None
    max_eval_batches: int | None = None
    task_repeat_factors: dict[str, int] | None = None
    no_text: bool = False
    text_model: str = "BAAI/bge-small-zh-v1.5"
    combat_embed_dim: int = 64
    build_embed_dim: int = 64
    route_embed_dim: int = 48
    n_heads: int = 2
    text_proj_dim: int = 32
    context_text_dim: int = 48
    shared_hidden_dim: int = 96
    shared_output_dim: int = 128
    combat_scorer_hidden: int = 128
    build_scorer_hidden: int = 96
    route_scorer_hidden: int = 64
    critic_domain_dim: int = 64


class OfflinePolicyEnv(Env):
    def __init__(self, obs_space: spaces.Dict):
        super().__init__()
        self.observation_space = obs_space
        self.action_space = spaces.Discrete(MAX_ACTIONS)

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        return _zero_obs_from_space(self.observation_space), {}

    def step(self, action):
        return _zero_obs_from_space(self.observation_space), 0.0, True, False, {}


def _zero_obs_from_space(obs_space: spaces.Dict) -> dict[str, np.ndarray]:
    return {
        key: np.zeros(space.shape, dtype=np.float32)
        for key, space in obs_space.spaces.items()
    }


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def parse_tasks(raw: str | None) -> list[str]:
    if not raw:
        return list(DEFAULT_TASKS)
    tasks = [task.strip() for task in raw.split(",") if task.strip()]
    invalid = [task for task in tasks if task not in SUPERVISED_TASKS]
    if invalid:
        raise SystemExit(f"Unsupported tasks: {invalid}")
    return tasks


def parse_task_repeat_factors(raw: str | None, tasks: list[str]) -> dict[str, int]:
    repeats = {task: 1 for task in tasks}
    if not raw:
        return repeats

    for part in raw.split(","):
        part = part.strip()
        if not part:
            continue
        if ":" not in part:
            raise SystemExit(
                f"Invalid --task-repeat-factors entry '{part}'. Expected task:factor."
            )
        task, factor_raw = [item.strip() for item in part.split(":", 1)]
        if task not in tasks:
            raise SystemExit(f"Unknown task in --task-repeat-factors: {task}")
        try:
            factor = int(factor_raw)
        except ValueError as exc:
            raise SystemExit(f"Invalid repeat factor for task={task}: {factor_raw}") from exc
        if factor <= 0:
            raise SystemExit(f"Repeat factor must be >= 1 for task={task}, got {factor}")
        repeats[task] = factor
    return repeats


def build_weighted_task_order(tasks: list[str], repeat_factors: dict[str, int] | None) -> list[str]:
    repeats = repeat_factors or {}
    weighted: list[str] = []
    for task in tasks:
        factor = max(int(repeats.get(task, 1)), 1)
        weighted.extend([task] * factor)
    return weighted


def build_policy_kwargs(config: OfflineOnlineConfig) -> dict[str, Any]:
    return {
        "combat_embed_dim": config.combat_embed_dim,
        "build_embed_dim": config.build_embed_dim,
        "route_embed_dim": config.route_embed_dim,
        "n_heads": config.n_heads,
        "text_proj_dim": config.text_proj_dim,
        "context_text_dim": config.context_text_dim,
        "shared_hidden_dim": config.shared_hidden_dim,
        "shared_output_dim": config.shared_output_dim,
        "combat_scorer_hidden": config.combat_scorer_hidden,
        "build_scorer_hidden": config.build_scorer_hidden,
        "route_scorer_hidden": config.route_scorer_hidden,
        "critic_domain_dim": config.critic_domain_dim,
    }


def make_dummy_env(obs_space: spaces.Dict):
    return DummyVecEnv([lambda: OfflinePolicyEnv(obs_space)])


def move_batch_to_device(batch: dict[str, Any], device: torch.device) -> dict[str, Any]:
    moved: dict[str, Any] = {}
    for key, value in batch.items():
        if isinstance(value, torch.Tensor):
            moved[key] = value.to(device)
        elif isinstance(value, dict):
            moved[key] = move_batch_to_device(value, device)
        else:
            moved[key] = value
    return moved


def iter_round_robin(loaders: dict[str, DataLoader], task_order: list[str]):
    iterators = {task: iter(loader) for task, loader in loaders.items()}
    active = set(task_order)
    while active:
        progressed = False
        for task in task_order:
            if task not in active:
                continue
            try:
                batch = next(iterators[task])
                progressed = True
                yield task, batch
            except StopIteration:
                active.remove(task)
        if not progressed:
            break


def normalize_room_type(value: str | None) -> str:
    lookup = {
        "monster": "Monster",
        "elite": "Elite",
        "boss": "Boss",
        "event": "Event",
        "rest": "Rest",
        "rest_site": "Rest",
        "merchant": "Merchant",
        "shop": "Merchant",
        "treasure": "Treasure",
    }
    return lookup.get((value or "").strip().lower(), "Event")


def normalize_route_point_type(value: str | None) -> str:
    return DictObservationEncoder._normalize_route_point_type(value)


def route_row_has_candidate_supervision(row: dict[str, Any]) -> bool:
    route_candidates = row.get("route_candidates")
    label_index = row.get("label_index")
    return (
        isinstance(route_candidates, list)
        and len(route_candidates) > 0
        and isinstance(label_index, int)
        and 0 <= label_index < len(route_candidates)
    )


def filter_route_rows(rows: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], dict[str, int]]:
    accepted: list[dict[str, Any]] = []
    diagnostics = {
        "accepted": 0,
        "chosen_path_only": 0,
        "missing_candidates": 0,
    }
    for row in rows:
        if route_row_has_candidate_supervision(row):
            accepted.append(row)
            diagnostics["accepted"] += 1
            continue
        if str(row.get("supervision_type") or "") == "chosen_path_only":
            diagnostics["chosen_path_only"] += 1
        else:
            diagnostics["missing_candidates"] += 1
    return accepted, diagnostics


def action_card_label(card_id: str, upgrade_level: float = 0.0) -> str:
    return build_card_label(card_id, upgrade_level)


REMOVE_TEXT_TASKS = {"remove_card_step", "shop_remove_target_step"}
SEMANTIC_CARD_TASKS = REMOVE_TEXT_TASKS | {"smith_target", "transform_card_step"}
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


def _infer_card_type(card_id: str | None) -> str | None:
    if _is_starter_strike(card_id):
        return "Attack"
    if _is_starter_defend(card_id):
        return "Skill"
    if _is_ascenders_bane(card_id) or _is_curse_card(card_id):
        return "Curse"
    if _is_status_card(card_id):
        return "Status"
    return None


def _deck_card_rows(row: dict[str, Any]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    selected_prefix_counts = Counter(
        str(value)
        for value in (row.get("selected_prefix_ids") or [])
        if value
    )
    for card_id, count, upgraded_count, max_upgrade_level in zip(
        row.get("deck_ids") or [],
        row.get("deck_counts") or [],
        row.get("deck_upgraded_counts") or [],
        row.get("deck_max_upgrade_levels") or [],
    ):
        if not card_id:
            continue
        card_key = str(card_id)
        remaining_count = max(int(count) - int(selected_prefix_counts.get(card_key, 0)), 0)
        if remaining_count <= 0:
            continue
        rows.append(
            {
                "id": card_key,
                "count": remaining_count,
                "upgraded_count": min(max(int(upgraded_count), 0), remaining_count),
                "max_upgrade_level": max(float(max_upgrade_level), 0.0),
            }
        )
    return rows


def _summarize_counted_cards(entries: list[dict[str, Any]], *, limit: int = 3) -> str:
    if not entries:
        return "none"
    ordered = sorted(
        [entry for entry in entries if int(entry.get("count", 0)) > 0],
        key=lambda item: (-int(item.get("count", 0)), build_card_label(str(item.get("id") or ""))),
    )
    if not ordered:
        return "none"
    parts = [
        f"{build_card_label(str(entry.get('id') or ''))} x{int(entry.get('count', 0))}"
        for entry in ordered[:limit]
    ]
    if len(ordered) > limit:
        parts.append(f"+{len(ordered) - limit} more")
    return ", ".join(parts)


def _build_remove_deck_profile(row: dict[str, Any]) -> dict[str, Any]:
    cache = row.get("_semantic_cache")
    if not isinstance(cache, dict):
        cache = {}
        row["_semantic_cache"] = cache
    cached = cache.get("remove_deck_profile")
    if isinstance(cached, dict):
        return cached

    cards = _deck_card_rows(row)
    by_id = {str(entry["id"]): entry for entry in cards}
    size = sum(int(entry["count"]) for entry in cards)
    starter_attacks = [entry for entry in cards if _is_starter_strike(entry["id"])]
    starter_defends = [entry for entry in cards if _is_starter_defend(entry["id"])]
    banes = [entry for entry in cards if _is_ascenders_bane(entry["id"])]
    curses = [
        entry
        for entry in cards
        if _is_curse_card(entry["id"]) and not _is_ascenders_bane(entry["id"])
    ]
    statuses = [entry for entry in cards if _is_status_card(entry["id"])]
    junk_ids = {
        str(entry["id"])
        for entry in starter_attacks + starter_defends + banes + curses + statuses
    }
    non_junk_entries = [entry for entry in cards if str(entry["id"]) not in junk_ids]
    duplicate_entries = [entry for entry in cards if int(entry["count"]) > 1]
    max_count = max((int(entry["count"]) for entry in cards), default=1)

    ordered_core_ids = [
        str(entry["id"])
        for entry in sorted(
            non_junk_entries,
            key=lambda item: (-int(item["count"]), build_card_label(str(item["id"]))),
        )
    ]
    smithable_total = sum(max(int(entry["count"]) - int(entry["upgraded_count"]), 0) for entry in cards)
    smithable_nonjunk = sum(
        max(int(entry["count"]) - int(entry["upgraded_count"]), 0)
        for entry in non_junk_entries
    )
    profile = {
        "size": size,
        "distinct": len(cards),
        "upgraded_total": sum(int(entry["upgraded_count"]) for entry in cards),
        "smithable_total": smithable_total,
        "smithable_nonjunk": smithable_nonjunk,
        "unupgraded_ratio": (smithable_total / max(size, 1)),
        "starter_attack_count": sum(int(entry["count"]) for entry in starter_attacks),
        "starter_defend_count": sum(int(entry["count"]) for entry in starter_defends),
        "bane_count": sum(int(entry["count"]) for entry in banes),
        "curse_count": sum(int(entry["count"]) for entry in curses),
        "status_count": sum(int(entry["count"]) for entry in statuses),
        "junk_count": sum(
            int(entry["count"])
            for entry in starter_attacks + starter_defends + banes + curses + statuses
        ),
        "duplicate_count": sum(int(entry["count"]) - 1 for entry in duplicate_entries),
        "max_count": max_count,
        "top_duplicates_text": _summarize_counted_cards(duplicate_entries),
        "core_text": summarize_entity_ids("card", ordered_core_ids, limit=5),
        "by_id": by_id,
    }
    cache["remove_deck_profile"] = profile
    return profile


def _build_semantic_tag_profile(row: dict[str, Any]) -> dict[str, Any]:
    cache = row.get("_semantic_cache")
    if not isinstance(cache, dict):
        cache = {}
        row["_semantic_cache"] = cache
    cached = cache.get("semantic_tag_profile")
    if isinstance(cached, dict):
        return cached

    counts = {key: 0 for key in _SEMANTIC_TAG_GROUPS}
    by_group_ids: dict[str, list[str]] = {key: [] for key in _SEMANTIC_TAG_GROUPS}
    for entry in _deck_card_rows(row):
        card_id = str(entry.get("id") or "")
        if not card_id:
            continue
        count = max(int(entry.get("count", 0)), 0)
        tags = _card_semantic_tags(card_id)
        if count <= 0 or not tags:
            continue
        for group_name, group_tags in _SEMANTIC_TAG_GROUPS.items():
            if not tags.intersection(group_tags):
                continue
            counts[group_name] += count
            by_group_ids[group_name].extend([card_id] * count)

    profile = {
        "counts": counts,
        "texts": {
            group_name: summarize_entity_ids("card", values, limit=3)
            for group_name, values in by_group_ids.items()
        },
    }
    cache["semantic_tag_profile"] = profile
    return profile


def _remove_role_tags(
    card_id: str,
    *,
    count: int,
    upgraded_count: int,
    max_count: int,
) -> list[str]:
    tags: list[str] = []
    if _is_starter_strike(card_id):
        tags.append("starter_attack")
    elif _is_starter_defend(card_id):
        tags.append("starter_defend")
    elif _is_ascenders_bane(card_id):
        tags.append("bane")
    elif _is_curse_card(card_id):
        tags.append("curse")
    elif _is_status_card(card_id):
        tags.append("status")
    else:
        tags.append("nonstarter")

    if count >= 2:
        tags.append("duplicate")
    if count == 1:
        tags.append("single_copy")
    if count >= 3:
        tags.append("thick_stack")
    if max_count > 1 and count == max_count:
        tags.append("largest_stack")
    if upgraded_count > 0:
        tags.append("has_upgrade")
    if upgraded_count >= count and count > 0:
        tags.append("all_upgraded")
    return tags


_BUCKET_SCORES = {
    "very_low": 0.0,
    "low": 0.25,
    "medium": 0.5,
    "high": 0.75,
    "very_high": 1.0,
    "unknown": 0.5,
}

_SEMANTIC_TAG_GROUPS: dict[str, set[str]] = {
    "exhaust_enablers": {"exhaust_enabler"},
    "exhaust_payoffs": {"exhaust_payoff"},
    "star_sources": {"star_source"},
    "peck_engines": {"peck_engine", "transform_engine"},
    "peck_tokens": {"peck_token"},
    "upgrade_enablers": {"upgrade_enabler", "mass_upgrade"},
}


def _bucket_score(value: str | None, default: float = 0.5) -> float:
    return float(_BUCKET_SCORES.get(str(value or "unknown").strip().lower(), default))


def _card_semantic_tags(card_id: str | None) -> set[str]:
    metadata = get_card_metadata(card_id) or {}
    values = metadata.get("semantic_tags")
    if not isinstance(values, list):
        return set()
    return {str(value).strip() for value in values if str(value).strip()}


def _card_prior_entry(card_id: str, key: str) -> dict[str, Any]:
    metadata = get_card_metadata(card_id) or {}
    entry = metadata.get("priors") or {}
    value = entry.get(key)
    return value if isinstance(value, dict) else {}


def _card_prior_rate(card_id: str, key: str) -> float:
    entry = _card_prior_entry(card_id, key)
    if entry.get("rate") is not None:
        return float(entry.get("rate") or 0.0)
    return _bucket_score(entry.get("bucket"))


def _card_keep_score(card_id: str) -> float:
    metadata = get_card_metadata(card_id) or {}
    keep = metadata.get("keep")
    if isinstance(keep, dict):
        bucket = keep.get("bucket")
        if bucket:
            return _bucket_score(str(bucket))
        score = keep.get("score")
        if score is not None:
            raw = float(score)
            return max(min((raw + 1.0) / 2.0, 1.0), 0.0)
    return 0.5


def _rank_positions(score_by_id: dict[str, float]) -> dict[str, int]:
    ordered = sorted(score_by_id.items(), key=lambda item: (-float(item[1]), item[0]))
    return {card_id: index + 1 for index, (card_id, _score) in enumerate(ordered)}


def _rank_band(rank: int, total: int) -> str:
    if rank <= 0 or total <= 0:
        return "unknown"
    if rank == 1:
        return "best"
    if rank <= max(2, total // 4):
        return "high"
    if rank >= total:
        return "worst"
    if rank > total - max(2, total // 4):
        return "low"
    return "mid"


def _starter_balance_tag(starter_attack_after: int, starter_defend_after: int) -> str:
    diff = int(starter_attack_after) - int(starter_defend_after)
    if diff >= 2:
        return "attack_heavy"
    if diff == 1:
        return "attack_bias"
    if diff == 0:
        return "balanced"
    if diff == -1:
        return "defend_bias"
    return "defend_heavy"


def _build_remove_choice_context(row: dict[str, Any], task: str) -> dict[str, Any]:
    cache = row.get("_semantic_cache")
    if not isinstance(cache, dict):
        cache = {}
        row["_semantic_cache"] = cache
    cache_key = f"remove_choice_context:{task}"
    cached = cache.get(cache_key)
    if isinstance(cached, dict):
        return cached

    profile = _build_remove_deck_profile(row)
    option_ids = [
        str(value)
        for value in (row.get("candidate_ids") or [])
        if value and str(value) != "<skip>"
    ]
    option_order: list[str] = []
    seen: set[str] = set()
    for candidate_id in option_ids:
        if candidate_id in seen:
            continue
        seen.add(candidate_id)
        option_order.append(candidate_id)

    if not option_order:
        context = {
            "option_order": [],
            "option_total": 0,
            "by_id": {},
            "primary_frontier_text": "none",
            "rate_frontier_text": "none",
            "remove_frontier_text": "none",
            "keep_frontier_text": "none",
        }
        cache[cache_key] = context
        return context

    by_id: dict[str, dict[str, Any]] = {}
    remove_scores: dict[str, float] = {}
    rate_scores: dict[str, float] = {}
    keep_scores: dict[str, float] = {}
    deck_size = max(int(profile.get("size", 0)), 1)
    max_count = max(int(profile.get("max_count", 1)), 1)
    starter_attack_total = int(profile.get("starter_attack_count", 0))
    starter_defend_total = int(profile.get("starter_defend_count", 0))
    junk_total = int(profile.get("junk_count", 0))
    current_gap = abs(starter_attack_total - starter_defend_total)

    for candidate_id in option_order:
        deck_entry = profile["by_id"].get(candidate_id, {})
        count = max(int(deck_entry.get("count", 0)), 1)
        upgraded_count = max(int(deck_entry.get("upgraded_count", 0)), 0)

        keep_score = _card_keep_score(candidate_id)
        reward_rate = _card_prior_rate(candidate_id, "reward")
        smith_rate = _card_prior_rate(candidate_id, "smith")
        remove_any_rate = _card_prior_rate(candidate_id, "remove_any")
        shop_remove_rate = _card_prior_rate(candidate_id, "shop_remove")
        transform_rate = _card_prior_rate(candidate_id, "transform")

        if task == "shop_remove_target_step":
            primary_rate = shop_remove_rate if int(_card_prior_entry(candidate_id, "shop_remove").get("shown") or 0) > 0 else remove_any_rate
        elif task == "remove_card_step":
            primary_rate = remove_any_rate
        elif task == "smith_target":
            primary_rate = smith_rate
        elif task == "transform_card_step":
            primary_rate = transform_rate
        else:
            primary_rate = remove_any_rate

        if task in REMOVE_TEXT_TASKS:
            primary_score = primary_rate + (1.0 - keep_score)
        elif task == "smith_target":
            primary_score = smith_rate + (0.5 * keep_score) - (0.25 * remove_any_rate)
        elif task == "transform_card_step":
            primary_score = transform_rate + (0.5 * remove_any_rate) - (0.5 * keep_score)
        else:
            primary_score = primary_rate

        starter_attack_after = max(starter_attack_total - (1 if _is_starter_strike(candidate_id) else 0), 0)
        starter_defend_after = max(starter_defend_total - (1 if _is_starter_defend(candidate_id) else 0), 0)
        junk_after = max(
            junk_total
            - (1 if (
                _is_starter_strike(candidate_id)
                or _is_starter_defend(candidate_id)
                or _is_ascenders_bane(candidate_id)
                or _is_curse_card(candidate_id)
                or _is_status_card(candidate_id)
            ) else 0),
            0,
        )

        by_id[candidate_id] = {
            "count": count,
            "upgraded_count": upgraded_count,
            "keep_score": keep_score,
            "reward_rate": reward_rate,
            "smith_rate": smith_rate,
            "remove_any_rate": remove_any_rate,
            "shop_remove_rate": shop_remove_rate,
            "transform_rate": transform_rate,
            "primary_rate": primary_rate,
            "primary_score": primary_score,
            "starter_attack_after": starter_attack_after,
            "starter_defend_after": starter_defend_after,
            "junk_after": junk_after,
            "deck_after_size": max(deck_size - 1, 0),
            "count_share": count / max(deck_size, 1),
            "is_largest_stack": 1.0 if count >= max_count else 0.0,
        }
        remove_scores[candidate_id] = primary_score
        rate_scores[candidate_id] = primary_rate
        keep_scores[candidate_id] = keep_score

    remove_ranks = _rank_positions(remove_scores)
    rate_ranks = _rank_positions(rate_scores)
    keep_ranks = _rank_positions(keep_scores)
    for candidate_id, entry in by_id.items():
        entry["remove_rank"] = int(remove_ranks.get(candidate_id, 0))
        entry["rate_rank"] = int(rate_ranks.get(candidate_id, 0))
        entry["keep_rank"] = int(keep_ranks.get(candidate_id, 0))
        entry["remove_band"] = _rank_band(entry["remove_rank"], len(option_order))
        entry["rate_band"] = _rank_band(entry["rate_rank"], len(option_order))
        entry["keep_band"] = _rank_band(entry["keep_rank"], len(option_order))
        entry["starter_gap_after"] = abs(
            int(entry.get("starter_attack_after", 0)) - int(entry.get("starter_defend_after", 0))
        )
        entry["starter_gap_delta"] = int(entry["starter_gap_after"]) - int(current_gap)
        entry["gap_improves"] = 1.0 if int(entry["starter_gap_after"]) < int(current_gap) else 0.0
        entry["balanced_after"] = 1.0 if int(entry["starter_gap_after"]) == 0 else 0.0
        entry["starter_balance_tag"] = _starter_balance_tag(
            int(entry.get("starter_attack_after", 0)),
            int(entry.get("starter_defend_after", 0)),
        )

    remove_frontier = sorted(option_order, key=lambda candidate_id: (int(remove_ranks[candidate_id]), candidate_id))[:4]
    rate_frontier = sorted(option_order, key=lambda candidate_id: (int(rate_ranks[candidate_id]), candidate_id))[:4]
    keep_frontier = sorted(option_order, key=lambda candidate_id: (int(keep_ranks[candidate_id]), candidate_id))[:4]
    context = {
        "option_order": option_order,
        "option_total": len(option_order),
        "by_id": by_id,
        "primary_frontier_text": summarize_entity_ids("card", remove_frontier, limit=4),
        "rate_frontier_text": summarize_entity_ids("card", rate_frontier, limit=4),
        "remove_frontier_text": summarize_entity_ids("card", remove_frontier, limit=4),
        "keep_frontier_text": summarize_entity_ids("card", keep_frontier, limit=4),
    }
    cache[cache_key] = context
    return context


def _build_remove_aux_payload(
    candidate_id: str,
    *,
    task: str,
    choice_context: dict[str, Any] | None,
) -> dict[str, Any] | None:
    if not isinstance(choice_context, dict):
        return None
    option_entry = (choice_context.get("by_id") or {}).get(str(candidate_id))
    option_total = int(choice_context.get("option_total", 0))
    if not isinstance(option_entry, dict) or option_total <= 0:
        return None
    count_before = max(int(option_entry.get("count", 0)), 0)
    upgraded_count = max(int(option_entry.get("upgraded_count", 0)), 0)
    starter_attack_after = max(int(option_entry.get("starter_attack_after", 0)), 0)
    starter_defend_after = max(int(option_entry.get("starter_defend_after", 0)), 0)
    junk_after = max(int(option_entry.get("junk_after", 0)), 0)
    deck_after_size = max(int(option_entry.get("deck_after_size", 0)), 0)
    return {
        "semantic_task": task,
        "option_total": option_total,
        "remove_rank": int(option_entry.get("remove_rank", 0)),
        "rate_rank": int(option_entry.get("rate_rank", 0)),
        "keep_rank": int(option_entry.get("keep_rank", 0)),
        "reward_rate": float(option_entry.get("reward_rate", 0.0)),
        "smith_rate": float(option_entry.get("smith_rate", 0.0)),
        "remove_any_rate": float(option_entry.get("remove_any_rate", 0.0)),
        "shop_remove_rate": float(option_entry.get("shop_remove_rate", 0.0)),
        "transform_rate": float(option_entry.get("transform_rate", 0.0)),
        "primary_rate": float(option_entry.get("primary_rate", 0.0)),
        "primary_score": float(option_entry.get("primary_score", 0.0)),
        "keep_score": float(option_entry.get("keep_score", 0.0)),
        "count_before": count_before,
        "count_after": max(count_before - 1, 0),
        "upgraded_count": upgraded_count,
        "unupgraded_count": max(count_before - upgraded_count, 0),
        "upgrade_share": (upgraded_count / count_before) if count_before > 0 else 0.0,
        "all_upgraded": 1.0 if count_before > 0 and upgraded_count >= count_before else 0.0,
        "junk_after": junk_after,
        "deck_after_size": deck_after_size,
        "starter_attack_after": starter_attack_after,
        "starter_defend_after": starter_defend_after,
        "starter_gap_after": abs(starter_attack_after - starter_defend_after),
        "starter_gap_delta": int(option_entry.get("starter_gap_delta", 0)),
        "gap_improves": float(option_entry.get("gap_improves", 0.0)),
        "balanced_after": float(option_entry.get("balanced_after", 0.0)),
        "count_share": float(option_entry.get("count_share", 0.0)),
        "is_largest_stack": float(option_entry.get("is_largest_stack", 0.0)),
        "is_starter_attack": 1.0 if _is_starter_strike(candidate_id) else 0.0,
        "is_starter_defend": 1.0 if _is_starter_defend(candidate_id) else 0.0,
        "is_curse_or_status": 1.0 if (_is_curse_card(candidate_id) or _is_status_card(candidate_id) or _is_ascenders_bane(candidate_id)) else 0.0,
    }


def _build_remove_deck_card_text(
    card_id: str,
    *,
    task: str,
    copy_index: int,
    count: int,
    upgraded_count: int,
    upgrade_level: float,
    profile: dict[str, Any],
    choice_context: dict[str, Any] | None = None,
) -> str:
    label = build_card_semantic_text(card_id, upgrade_level=upgrade_level, task=task)
    tags = _remove_role_tags(
        card_id,
        count=count,
        upgraded_count=upgraded_count,
        max_count=int(profile.get("max_count", 1)),
    )
    option_entry = ((choice_context or {}).get("by_id") or {}).get(card_id, {})
    option_total = int((choice_context or {}).get("option_total", 0))
    parts = [
        f"copy {copy_index + 1}/{max(count, 1)}",
    ]
    if option_entry and option_total > 0:
        parts.extend(
            [
                f"rem_rank {int(option_entry.get('remove_rank', 0))}/{option_total}",
                f"rem_band {str(option_entry.get('remove_band') or 'unknown')}",
                f"rate_rank {int(option_entry.get('rate_rank', 0))}/{option_total}",
                f"keep_rank {int(option_entry.get('keep_rank', 0))}/{option_total}",
                f"keep_band {str(option_entry.get('keep_band') or 'unknown')}",
                f"after {max(int(option_entry.get('count', count)) - 1, 0)}",
                (
                    f"junk_after {int(option_entry.get('junk_after', 0))}/"
                    f"{max(int(option_entry.get('deck_after_size', 0)), 1)}"
                ),
                f"atk_after {int(option_entry.get('starter_attack_after', 0))}",
                f"def_after {int(option_entry.get('starter_defend_after', 0))}",
                f"starter_gap {int(option_entry.get('starter_gap_after', 0))}",
                f"gap_delta {int(option_entry.get('starter_gap_delta', 0)):+d}",
                f"balanced_after {int(option_entry.get('balanced_after', 0))}",
                f"starter_mix {str(option_entry.get('starter_balance_tag') or 'unknown')}",
            ]
        )
    parts.extend(
        [
            label,
            f"role {' '.join(tags)}",
            f"stack {count}",
        ]
    )
    if upgraded_count > 0:
        parts.append(f"upgraded_copies {upgraded_count}")
    return ". ".join(parts)




def _build_remove_candidate_text(
    row: dict[str, Any],
    *,
    task: str,
    candidate_id: str,
    count: float,
    upgrade_level: float,
) -> str:
    profile = _build_remove_deck_profile(row)
    choice_context = _build_remove_choice_context(row, task)
    option_entry = choice_context.get("by_id", {}).get(str(candidate_id), {})
    option_total = int(choice_context.get("option_total", 0))
    deck_entry = profile["by_id"].get(str(candidate_id), {})
    count_int = max(int(count), int(deck_entry.get("count", 0)), 1)
    upgraded_count = max(int(deck_entry.get("upgraded_count", 0)), 0)
    tags = _remove_role_tags(
        str(candidate_id),
        count=count_int,
        upgraded_count=upgraded_count,
        max_count=int(profile.get("max_count", 1)),
    )
    prefix = "shop remove" if task == "shop_remove_target_step" else "remove"
    parts = [
        f"{prefix} choice",
    ]
    if option_entry and option_total > 0:
        parts.extend(
            [
                f"rem_rank {int(option_entry.get('remove_rank', 0))}/{option_total}",
                f"rem_band {str(option_entry.get('remove_band') or 'unknown')}",
                f"rate_rank {int(option_entry.get('rate_rank', 0))}/{option_total}",
                f"keep_rank {int(option_entry.get('keep_rank', 0))}/{option_total}",
                f"keep_band {str(option_entry.get('keep_band') or 'unknown')}",
                (
                    f"junk_after {int(option_entry.get('junk_after', 0))}/"
                    f"{max(int(option_entry.get('deck_after_size', 0)), 1)}"
                ),
                f"atk_after {int(option_entry.get('starter_attack_after', 0))}",
                f"def_after {int(option_entry.get('starter_defend_after', 0))}",
                f"starter_gap {int(option_entry.get('starter_gap_after', 0))}",
                f"gap_delta {int(option_entry.get('starter_gap_delta', 0)):+d}",
                f"balanced_after {int(option_entry.get('balanced_after', 0))}",
                f"starter_mix {str(option_entry.get('starter_balance_tag') or 'unknown')}",
            ]
        )
    parts.extend(
        [
            f"copies {count_int}->{max(count_int - 1, 0)}",
            f"junk_now {int(profile.get('junk_count', 0))}/{max(int(profile.get('size', 0)), 1)}",
            build_card_semantic_text(str(candidate_id), upgrade_level=upgrade_level, task=task),
            f"tags {' '.join(tags)}",
        ]
    )
    if upgraded_count > 0:
        parts.append(f"upgraded_copies {upgraded_count}")
    return ". ".join(parts)




def _build_semantic_candidate_text(
    row: dict[str, Any],
    *,
    task: str,
    candidate_id: str,
    count: float,
    upgrade_level: float,
) -> str:
    if task in REMOVE_TEXT_TASKS:
        return _build_remove_candidate_text(
            row,
            task=task,
            candidate_id=candidate_id,
            count=count,
            upgrade_level=upgrade_level,
        )

    profile = _build_remove_deck_profile(row)
    choice_context = _build_remove_choice_context(row, task)
    tag_profile = _build_semantic_tag_profile(row)
    tag_counts = tag_profile.get("counts", {})
    option_entry = choice_context.get("by_id", {}).get(str(candidate_id), {})
    option_total = int(choice_context.get("option_total", 0))
    deck_entry = profile["by_id"].get(str(candidate_id), {})
    count_int = max(int(count), int(deck_entry.get("count", 0)), 1)
    upgraded_count = max(int(deck_entry.get("upgraded_count", 0)), 0)
    candidate_tags = _card_semantic_tags(str(candidate_id))
    prefix = {
        "smith_target": "smith",
        "transform_card_step": "transform",
    }.get(task, task)
    if option_entry and option_total > 0:
        rank_parts = [
            f"primary_rank {int(option_entry.get('remove_rank', 0))}/{option_total}",
            f"rate_rank {int(option_entry.get('rate_rank', 0))}/{option_total}",
            f"keep_rank {int(option_entry.get('keep_rank', 0))}/{option_total}",
        ]
    else:
        rank_parts = []
    parts = [
        f"{prefix} choice",
        *rank_parts,
        build_card_semantic_text(str(candidate_id), upgrade_level=upgrade_level, task=task),
        f"copies {count_int}",
    ]
    if task == "smith_target":
        parts.append(f"smithable {max(count_int - upgraded_count, 0)}")
        parts.append(
            f"deck_smithable {int(profile.get('smithable_total', 0))}/{max(int(profile.get('size', 0)), 1)}"
        )
        parts.append(f"core_smithable {int(profile.get('smithable_nonjunk', 0))}")
    elif option_entry and option_total > 0:
        parts.extend(
            [
                f"primary_band {str(option_entry.get('remove_band') or 'unknown')}",
                f"rate_band {str(option_entry.get('rate_band') or 'unknown')}",
                f"keep_band {str(option_entry.get('keep_band') or 'unknown')}",
            ]
        )
    if count_int >= 2:
        parts.append("duplicate")
    if upgraded_count > 0:
        parts.append(f"upgraded_copies {upgraded_count}")
        if upgraded_count >= count_int:
            parts.append("all_upgraded")
    if task == "smith_target":
        parts.append(f"deck_upgraded {int(profile.get('upgraded_total', 0))}")
    if task == "transform_card_step":
        parts.append(
            f"junk {int(profile.get('junk_count', 0))}/{max(int(profile.get('size', 0)), 1)}"
        )
    if "mass_upgrade" in candidate_tags or "upgrade_enabler" in candidate_tags:
        parts.append("mass_upgrade")
        if int(profile.get("smithable_nonjunk", 0)) >= 6:
            parts.append("upgrade_all_live")
            parts.append("beats_single_engine_upgrade")
    if "exhaust_payoff" in candidate_tags:
        parts.append(f"exhaust_enablers {int(tag_counts.get('exhaust_enablers', 0))}")
        if int(tag_counts.get("exhaust_enablers", 0)) >= 2:
            parts.append("exhaust_engine_live")
    if "star_source" in candidate_tags:
        parts.append(f"star_sources {int(tag_counts.get('star_sources', 0))}")
        if int(tag_counts.get("star_sources", 0)) <= 1:
            parts.append("rare_star_source")
    if "peck_engine" in candidate_tags or "transform_engine" in candidate_tags:
        parts.append(f"peck_tokens {int(tag_counts.get('peck_tokens', 0))}")
        parts.append("charge_combo_live" if int(tag_counts.get("peck_tokens", 0)) >= 1 else "needs_peck_setup")
        if int(tag_counts.get("peck_tokens", 0)) >= 1:
            parts.append("prefer_engine_upgrade")
            parts.append("beats_generic_draw_upgrade")
    if "peck_token" in candidate_tags:
        parts.append(f"peck_engines {int(tag_counts.get('peck_engines', 0))}")
        if int(tag_counts.get("peck_engines", 0)) >= 1:
            parts.append("charge_support_live")
            parts.append("single_piece_not_engine")
    return ". ".join(parts)


def _build_semantic_deck_card_text(
    task: str,
    card_id: str,
    *,
    copy_index: int,
    count: int,
    upgraded_count: int,
    upgrade_level: float,
    profile: dict[str, Any],
    choice_context: dict[str, Any] | None = None,
) -> str:
    if task in REMOVE_TEXT_TASKS:
        return _build_remove_deck_card_text(
            card_id,
            task=task,
            copy_index=copy_index,
            count=count,
            upgraded_count=upgraded_count,
            upgrade_level=upgrade_level,
            profile=profile,
            choice_context=choice_context,
        )

    option_entry = ((choice_context or {}).get("by_id") or {}).get(card_id, {})
    option_total = int((choice_context or {}).get("option_total", 0))
    if option_entry and option_total > 0:
        rank_parts = [
            f"primary_rank {int(option_entry.get('remove_rank', 0))}/{option_total}",
            f"rate_rank {int(option_entry.get('rate_rank', 0))}/{option_total}",
            f"keep_rank {int(option_entry.get('keep_rank', 0))}/{option_total}",
        ]
    else:
        rank_parts = []
    parts = [
        f"copy {copy_index + 1}/{max(count, 1)}",
        *rank_parts,
        build_card_semantic_text(card_id, upgrade_level=upgrade_level, task=task),
    ]
    if task == "smith_target":
        parts.append(f"smithable {max(count - upgraded_count, 0)}")
    elif option_entry and option_total > 0:
        parts.extend(
            [
                f"primary_band {str(option_entry.get('remove_band') or 'unknown')}",
                f"rate_band {str(option_entry.get('rate_band') or 'unknown')}",
                f"keep_band {str(option_entry.get('keep_band') or 'unknown')}",
            ]
        )
    if count >= 2:
        parts.append(f"stack {count}")
    if upgraded_count > 0:
        parts.append(f"upgraded_copies {upgraded_count}")
    if task == "smith_target" and copy_index < upgraded_count:
        parts.append("already_upgraded")
    if task == "transform_card_step":
        parts.append(
            f"junk {int(profile.get('junk_count', 0))}/{max(int(profile.get('size', 0)), 1)}"
        )
    return ". ".join(parts)


def _build_semantic_decision_text(
    row: dict[str, Any],
    *,
    task: str,
    character: str,
    build_id: str,
    floor: int,
    room_type: str,
    hp: float,
    max_hp: float,
    gold: float,
    step_index: int,
    steps_total: int,
    prefix_text: str,
) -> str:
    profile = _build_remove_deck_profile(row)
    tag_profile = _build_semantic_tag_profile(row)
    tag_counts = tag_profile.get("counts", {})
    tag_texts = tag_profile.get("texts", {})
    floor_stage = "early" if floor <= 10 else ("mid" if floor <= 30 else "late")
    base = (
        f"task={task} character={character} build={build_id} "
        f"floor={floor} stage={floor_stage} room={room_type} "
        f"hp={int(hp)}/{int(max_hp)} gold={int(gold)} "
        f"step={step_index + 1}/{max(steps_total, 1)} selected={prefix_text} "
        f"deck={int(profile['size'])} distinct={int(profile['distinct'])} upgraded={int(profile['upgraded_total'])} "
        f"smithable={int(profile.get('smithable_total', 0))} core_smithable={int(profile.get('smithable_nonjunk', 0))} "
        f"junk={int(profile['junk_count'])} strikes={int(profile['starter_attack_count'])} "
        f"defends={int(profile['starter_defend_count'])} bane={int(profile['bane_count'])} "
        f"curses={int(profile['curse_count'])} statuses={int(profile['status_count'])} "
        f"duplicates={profile['top_duplicates_text']} core={profile['core_text']} "
        f"sem_exhaust={int(tag_counts.get('exhaust_enablers', 0))}/{int(tag_counts.get('exhaust_payoffs', 0))} "
        f"sem_star={int(tag_counts.get('star_sources', 0))} "
        f"sem_peck={int(tag_counts.get('peck_engines', 0))}/{int(tag_counts.get('peck_tokens', 0))}"
    )
    if int(tag_counts.get("exhaust_enablers", 0)) > 0 or int(tag_counts.get("exhaust_payoffs", 0)) > 0:
        base += f" exhaust_core={tag_texts.get('exhaust_enablers', 'none')}"
    if int(tag_counts.get("star_sources", 0)) > 0:
        base += f" star_core={tag_texts.get('star_sources', 'none')}"
    if int(tag_counts.get("peck_engines", 0)) > 0 or int(tag_counts.get("peck_tokens", 0)) > 0:
        base += f" peck_core={tag_texts.get('peck_engines', 'none')} / {tag_texts.get('peck_tokens', 'none')}"
    if task in {"smith_target", "transform_card_step"}:
        choice_context = _build_remove_choice_context(row, task)
        frontier_label = "smith_frontier" if task == "smith_target" else "transform_frontier"
        return (
            f"{base} "
            f"{frontier_label}={choice_context['primary_frontier_text']} "
            f"rate_frontier={choice_context['rate_frontier_text']} "
            f"keep_frontier={choice_context['keep_frontier_text']}"
        )
    return base


def _build_remove_decision_text(
    row: dict[str, Any],
    *,
    task: str,
    character: str,
    build_id: str,
    floor: int,
    room_type: str,
    hp: float,
    max_hp: float,
    gold: float,
    step_index: int,
    steps_total: int,
    prefix_text: str,
) -> str:
    profile = _build_remove_deck_profile(row)
    choice_context = _build_remove_choice_context(row, task)
    floor_stage = "early" if floor <= 10 else ("mid" if floor <= 30 else "late")
    gold_key = "shop_gold" if task == "shop_remove_target_step" else "gold"
    return (
        f"task={task} character={character} build={build_id} "
        f"floor={floor} stage={floor_stage} room={room_type} "
        f"hp={int(hp)}/{int(max_hp)} {gold_key}={int(gold)} "
        f"step={step_index + 1}/{max(steps_total, 1)} selected={prefix_text} "
        f"deck={int(profile['size'])} distinct={int(profile['distinct'])} upgraded={int(profile['upgraded_total'])} "
        f"junk={int(profile['junk_count'])} strikes={int(profile['starter_attack_count'])} "
        f"defends={int(profile['starter_defend_count'])} bane={int(profile['bane_count'])} "
        f"curses={int(profile['curse_count'])} statuses={int(profile['status_count'])} "
        f"duplicates={profile['top_duplicates_text']} core={profile['core_text']} "
        f"remove_frontier={choice_context['remove_frontier_text']} "
        f"keep_frontier={choice_context['keep_frontier_text']}"
    )


def expand_deck_cards(row: dict[str, Any], task: str | None = None) -> list[dict[str, Any]]:
    cards: list[dict[str, Any]] = []
    semantic_profile = _build_remove_deck_profile(row) if task in SEMANTIC_CARD_TASKS else None
    semantic_choice_context = (
        _build_remove_choice_context(row, str(task))
        if task in SEMANTIC_CARD_TASKS
        else None
    )
    if semantic_profile is not None:
        source_rows = _deck_card_rows(row)
    else:
        source_rows = [
            {
                "id": card_id,
                "count": max(int(count), 0),
                "upgraded_count": max(int(upgraded_count), 0),
                "max_upgrade_level": max(float(max_upgrade_level), 0.0),
            }
            for card_id, count, upgraded_count, max_upgrade_level in zip(
                row.get("deck_ids") or [],
                row.get("deck_counts") or [],
                row.get("deck_upgraded_counts") or [],
                row.get("deck_max_upgrade_levels") or [],
            )
            if card_id
        ]
    for source in source_rows:
        card_id = str(source.get("id") or "")
        if not card_id:
            continue
        count = max(int(source.get("count", 0)), 0)
        upgraded_count = max(int(source.get("upgraded_count", 0)), 0)
        max_upgrade_level = max(float(source.get("max_upgrade_level", 0.0)), 0.0)
        copies = max(int(count), 0)
        upgraded_copies = max(int(upgraded_count), 0)
        for copy_index in range(copies):
            upgrade_level = max_upgrade_level if copy_index < upgraded_copies else 0.0
            entry: dict[str, Any] = {
                "id": card_id,
                "canonical_text": (
                    _build_semantic_deck_card_text(
                        task,
                        str(card_id),
                        copy_index=copy_index,
                        count=copies,
                        upgraded_count=upgraded_copies,
                        upgrade_level=upgrade_level,
                        profile=semantic_profile or {},
                        choice_context=semantic_choice_context,
                    )
                    if semantic_profile is not None
                    else action_card_label(card_id, upgrade_level)
                ),
            }
            if semantic_choice_context is not None:
                build_aux = _build_remove_aux_payload(
                    card_id,
                    task=str(task),
                    choice_context=semantic_choice_context,
                )
                if build_aux is not None:
                    entry["build_aux"] = build_aux
            inferred_type = _infer_card_type(card_id)
            if inferred_type:
                entry["type"] = inferred_type
            cards.append(entry)
    return cards[:40]


def build_relic_entries(row: dict[str, Any]) -> list[dict[str, Any]]:
    return [
        {
            "title": relic_id,
            "canonical_text": build_entity_text("relic", relic_id),
        }
        for relic_id in (row.get("relic_ids") or [])
    ][:20]


def build_decision_text(row: dict[str, Any], task: str) -> str:
    scalars = row.get("scalars") or []
    floor = int(_scalar_value(scalars, SCALAR_FLOOR))
    hp = _scalar_value(scalars, SCALAR_CURRENT_HP)
    max_hp = _scalar_value(scalars, SCALAR_MAX_HP)
    gold = _scalar_value(scalars, SCALAR_CURRENT_GOLD)
    room_type = row.get("room_type") or "unknown"
    character = row.get("character") or "unknown"
    build_id = row.get("build_id") or "unknown"
    step_index = int(row.get("selection_step_index") or 0)
    steps_total = int(row.get("selection_steps_total") or 0)
    prefix = row.get("selected_prefix_ids") or []
    prefix_text = (
        ", ".join(build_card_label(str(value)) for value in prefix[:6])
        if prefix
        else "-"
    )
    if task in REMOVE_TEXT_TASKS:
        return _build_remove_decision_text(
            row,
            task=task,
            character=str(character),
            build_id=str(build_id),
            floor=floor,
            room_type=str(room_type),
            hp=hp,
            max_hp=max_hp,
            gold=_scalar_value(scalars, SCALAR_GOLD_BEFORE) or gold,
            step_index=step_index,
            steps_total=steps_total,
            prefix_text=prefix_text,
        )
    if task in SEMANTIC_CARD_TASKS:
        return _build_semantic_decision_text(
            row,
            task=task,
            character=str(character),
            build_id=str(build_id),
            floor=floor,
            room_type=str(room_type),
            hp=hp,
            max_hp=max_hp,
            gold=_scalar_value(scalars, SCALAR_GOLD_BEFORE) or gold,
            step_index=step_index,
            steps_total=steps_total,
            prefix_text=prefix_text,
        )
    return (
        f"task={task} character={character} build={build_id} "
        f"floor={floor} room={room_type} hp={int(hp)}/{int(max_hp)} gold={int(gold)} "
        f"step={step_index + 1}/{max(steps_total, 1)} selected={prefix_text}"
    )


def build_base_obs(
    row: dict[str, Any],
    *,
    task: str,
    phase: str,
    option_count: int,
    can_skip: bool,
    selected_count: int,
) -> dict[str, Any]:
    scalars = row.get("scalars") or []
    floor = int(_scalar_value(scalars, SCALAR_FLOOR))
    act_index = int(_scalar_value(scalars, SCALAR_ACT))
    turns = int(_scalar_value(scalars, SCALAR_TURNS))
    room_type = normalize_room_type(row.get("room_type"))
    domain = "route" if task in ROUTE_TASKS else "build"

    return {
        "phase": phase,
        "decision_domain": domain,
        "run": {
            "active": True,
            "game_over": False,
            "act_id": f"ACT.{act_index}",
            "act_floor": floor,
            "floor": floor,
            "room_type": room_type,
        },
        "player": {
            "hp": _scalar_value(scalars, SCALAR_CURRENT_HP),
            "max_hp": _scalar_value(scalars, SCALAR_MAX_HP),
            "gold": _scalar_value(scalars, SCALAR_CURRENT_GOLD),
            "block": 0.0,
            "relics": build_relic_entries(row),
            "potions": [],
            "deck_cards": expand_deck_cards(row, task),
        },
        "decision": {
            "decision_text": build_decision_text(row, task),
            "option_count": option_count,
            "can_skip": can_skip,
            "selected_count": selected_count,
            "min_select": 0 if selected_count <= 1 else selected_count,
            "max_select": max(selected_count, 1),
            "is_open": True,
            "travelable_count": option_count if task in ROUTE_TASKS else 0,
            "can_proceed": False,
            "proceed_only": False,
            "reward_count": option_count if task in {
                "card_choice",
                "relic_choice",
                "potion_choice",
                "regular_card_reward",
                "relic_choice_step",
                "potion_choice_step",
                "shop_relic_pick_step",
                "shop_potion_pick_step",
            } else 0,
            "item_count": option_count if task not in ROUTE_TASKS else 0,
        },
        "combat": None,
        "context": {
            "turns": turns,
            "path_index": _scalar_value(scalars, SCALAR_PATH),
            "ascension": _scalar_value(scalars, SCALAR_ASCENSION),
        },
    }


def _format_v2_candidate_text(
    *,
    task: str,
    option_kind: str,
    candidate_id: str,
    count: float = 1.0,
    upgrade_level: float = 0.0,
) -> str:
    return build_candidate_semantic_text(
        task=task,
        option_kind=option_kind,
        candidate_id=candidate_id,
        count=count,
        upgrade_level=upgrade_level,
    )


def build_v2_candidate_actions(row: dict[str, Any], task: str) -> tuple[list[dict[str, Any]], int]:
    actions: list[dict[str, Any]] = []
    label_index = int(row["label_index"])
    option_kind = str(row.get("option_kind") or "unknown")
    semantic_profile = _build_remove_deck_profile(row) if task in SEMANTIC_CARD_TASKS else None
    semantic_choice_context = _build_remove_choice_context(row, task) if task in SEMANTIC_CARD_TASKS else None

    for index, candidate_id in enumerate(row.get("candidate_ids") or []):
        count = float((row.get("candidate_counts") or [1.0] * len(row.get("candidate_ids") or []))[index])
        upgrade_level = float((row.get("candidate_upgrade_levels") or [0.0] * len(row.get("candidate_ids") or []))[index])
        if candidate_id == "<skip>":
            actions.append(
                {
                    "kind": "proceed",
                    "action_id": f"{task}:skip",
                    "index": index,
                    "skip": True,
                    "canonical_text": f"{task} skip",
                }
            )
            continue

        canonical = (
            _build_semantic_candidate_text(
                row,
                task=task,
                candidate_id=str(candidate_id),
                count=count,
                upgrade_level=upgrade_level,
            )
            if task in SEMANTIC_CARD_TASKS and option_kind == "card"
            else _format_v2_candidate_text(
                task=task,
                option_kind=option_kind,
                candidate_id=str(candidate_id),
                count=count,
                upgrade_level=upgrade_level,
            )
        )

        if option_kind == "card":
            kind = "card_reward" if task == "regular_card_reward" else ("deck_upgrade" if task == "smith_target" else "card_selection")
            deck_entry = semantic_profile["by_id"].get(str(candidate_id), {}) if semantic_profile is not None else {}
            action: dict[str, Any] = {
                "kind": kind,
                "action_id": f"{task}:{index}",
                "index": index,
                "card": {
                    "id": str(candidate_id),
                    "canonical_text": action_card_label(str(candidate_id), upgrade_level),
                },
                "canonical_text": canonical,
            }
            inferred_type = _infer_card_type(str(candidate_id))
            if inferred_type:
                action["card"]["type"] = inferred_type
            if deck_entry:
                action["card"]["count"] = int(deck_entry.get("count", 0))
                action["card"]["upgraded_count"] = int(deck_entry.get("upgraded_count", 0))
            if task == "smith_target":
                action["upgrade_preview"] = {"available": True}
            if semantic_choice_context is not None:
                build_aux = _build_remove_aux_payload(
                    str(candidate_id),
                    task=task,
                    choice_context=semantic_choice_context,
                )
                if build_aux is not None:
                    action["build_aux"] = build_aux
            actions.append(action)
            continue

        if option_kind == "relic":
            kind = "shop" if task == "shop_relic_pick_step" else "treasure_relic"
            actions.append(
                {
                    "kind": kind,
                    "action_id": f"{task}:{index}",
                    "index": index,
                    "canonical_text": canonical,
                }
            )
            continue

        if option_kind == "potion":
            kind = "shop" if task == "shop_potion_pick_step" else "reward"
            actions.append(
                {
                    "kind": kind,
                    "action_id": f"{task}:{index}",
                    "index": index,
                    "canonical_text": canonical,
                }
            )
            continue

        actions.append(
            {
                "kind": "event_option",
                "action_id": f"{task}:{index}",
                "index": index,
                "canonical_text": canonical,
            }
        )

    return actions, label_index


def build_card_reward_actions(row: dict[str, Any]) -> tuple[list[dict[str, Any]], int]:
    actions: list[dict[str, Any]] = []
    label_index = int(row["label_index"])
    for index, (card_id, upgrade_level) in enumerate(
        zip(row.get("candidate_ids") or [], row.get("candidate_upgrade_levels") or [])
    ):
        if card_id == "<skip>":
            actions.append(
                {
                    "kind": "card_reward",
                    "action_id": "card_reward:skip",
                    "index": index,
                    "skip": True,
                    "canonical_text": "skip card reward",
                }
            )
            continue
        actions.append(
            {
                "kind": "card_reward",
                "action_id": f"card_reward:{index}",
                "index": index,
                "card": {
                    "id": card_id,
                    "canonical_text": action_card_label(card_id, upgrade_level),
                },
                "canonical_text": f"card reward {action_card_label(card_id, upgrade_level)}",
            }
        )
    return actions, label_index


def build_single_choice_actions(row: dict[str, Any], *, kind: str, prefix: str) -> tuple[list[dict[str, Any]], int]:
    actions: list[dict[str, Any]] = []
    label_index = int(row["label_index"])
    for index, option_id in enumerate(row.get("candidate_ids") or []):
        option_kind = {
            "relic_choice": "relic",
            "potion_choice": "potion",
        }.get(prefix, "choice")
        actions.append(
            {
                "kind": kind,
                "action_id": f"{prefix}:{index}",
                "index": index,
                "canonical_text": build_candidate_semantic_text(
                    task=prefix,
                    option_kind=option_kind,
                    candidate_id=str(option_id),
                ) if option_kind != "choice" else f"{prefix} {humanize_game_id(str(option_id))}",
            }
        )
    return actions, label_index


def build_auxiliary_actions(row: dict[str, Any], task: str) -> tuple[list[dict[str, Any]], np.ndarray]:
    if task != "shop_bundle_aux":
        raise ValueError(f"Unsupported auxiliary task: {task}")

    actions: list[dict[str, Any]] = []
    labels = np.zeros(MAX_ACTIONS, dtype=np.float32)

    card_summary = summarize_entity_ids("card", row.get("available_card_ids") or [])
    relic_summary = summarize_entity_ids("relic", row.get("available_relic_ids") or [])
    potion_summary = summarize_entity_ids("potion", row.get("available_potion_ids") or [])

    canonical_by_field = {
        "did_buy_any_card": f"shop buy any card from {card_summary}",
        "did_buy_any_relic": f"shop buy any relic from {relic_summary}",
        "did_buy_any_potion": f"shop buy any potion from {potion_summary}",
        "did_remove_card": "shop remove a card from deck",
        "leave_only": "leave shop without buying or removing",
    }

    for index, (field_name, action_id, kind) in enumerate(SHOP_BUNDLE_AUX_ACTIONS):
        action: dict[str, Any] = {
            "kind": kind,
            "action_id": action_id,
            "index": index,
            "canonical_text": canonical_by_field[field_name],
        }
        if field_name == "leave_only":
            action["skip"] = True
        actions.append(action)
        labels[index] = 1.0 if row.get(field_name) else 0.0

    return actions, labels


def build_route_actions(row: dict[str, Any], *, task: str) -> tuple[list[dict[str, Any]], int]:
    route_candidates = row.get("route_candidates") or []
    label_index = row.get("label_index")
    if not route_row_has_candidate_supervision(row):
        raise ValueError(
            f"Route sample {row.get('sample_id')} lacks full candidate supervision; "
            f"expected route_candidates + valid label_index."
        )

    actions: list[dict[str, Any]] = []
    for index, candidate in enumerate(route_candidates):
        point_type = normalize_route_point_type(candidate.get("point_type_norm") or candidate.get("point_type"))
        coord = candidate.get("coord") if isinstance(candidate.get("coord"), dict) else None
        route_summary = candidate.get("route_summary") if isinstance(candidate.get("route_summary"), dict) else None
        route_nodes = candidate.get("route_nodes")
        if not isinstance(route_nodes, list):
            route_nodes = []

        action: dict[str, Any] = {
            "kind": "map",
            "action_id": str(candidate.get("action_id") or f"map:{index}"),
            "index": index,
            "point_type": point_type,
            "point_type_norm": point_type,
            "canonical_text": _build_route_candidate_text(point_type, coord, route_summary, candidate),
        }
        if coord is not None:
            action["coord"] = coord
        if route_summary is not None:
            action["route_summary"] = route_summary
        if route_nodes:
            action["route_nodes"] = route_nodes
        actions.append(action)

    return actions, label_index


def _build_route_candidate_text(
    point_type: str,
    coord: dict[str, Any] | None,
    route_summary: dict[str, Any] | None,
    candidate: dict[str, Any],
) -> str:
    canonical = str(candidate.get("canonical_text") or "").strip()
    if canonical:
        return canonical

    parts = [f"route {point_type}"]
    if isinstance(coord, dict):
        row = coord.get("row")
        col = coord.get("col")
        if row is not None and col is not None:
            parts.append(f"coord({col},{row})")
    if isinstance(route_summary, dict):
        if route_summary.get("forced_path_steps_before_branch") is not None:
            parts.append(f"forced={route_summary.get('forced_path_steps_before_branch')}")
        if route_summary.get("next_elite_steps") is not None:
            parts.append(f"elite_in={route_summary.get('next_elite_steps')}")
        if route_summary.get("next_rest_steps") is not None:
            parts.append(f"rest_in={route_summary.get('next_rest_steps')}")
        if route_summary.get("next_shop_steps") is not None:
            parts.append(f"shop_in={route_summary.get('next_shop_steps')}")
    return " ".join(parts)


def build_classification_actions(row: dict[str, Any], task: str, task_context: dict[str, Any]) -> tuple[list[dict[str, Any]], int]:
    if task in {"rest_site", "rest_action"}:
        candidates = task_context["rest_site_candidates"]
        label = str(row["label"])
        actions = [
            {
                "kind": "rest_site",
                "action_id": f"{task}:{candidate}",
                "index": index,
                "canonical_text": f"campfire {candidate}",
            }
            for index, candidate in enumerate(candidates)
        ]
    elif task == "shop_remove_binary":
        candidates = SHOP_REMOVE_BINARY_CANDIDATES
        label = str(row["label"])
        actions = [
            {
                "kind": "shop",
                "action_id": f"shop_remove_binary:{candidate}",
                "index": index,
                "canonical_text": f"shop {candidate}",
            }
            for index, candidate in enumerate(candidates)
        ]
    else:
        raise ValueError(f"Unsupported classification task: {task}")

    try:
        label_index = candidates.index(label)
    except ValueError as exc:
        raise ValueError(f"Unknown label '{label}' for task={task} sample={row.get('sample_id')}") from exc
    return actions, label_index


def build_cardset_actions(row: dict[str, Any], *, task: str) -> tuple[list[dict[str, Any]], np.ndarray, int]:
    actions: list[dict[str, Any]] = []
    positive = np.zeros(MAX_ACTIONS, dtype=np.float32)
    selected_ids = set(row.get("selected_slot_ids") or [])
    selected_count = max(len(selected_ids), 1)

    for index, (card_id, upgraded_count, max_upgrade_level) in enumerate(
        zip(
            row.get("deck_ids") or [],
            row.get("deck_upgraded_counts") or [],
            row.get("deck_max_upgrade_levels") or [],
        )
    ):
        if index >= MAX_ACTIONS:
            break
        canonical = action_card_label(card_id, max_upgrade_level if upgraded_count > 0 else 0.0)
        action: dict[str, Any] = {
            "index": index,
            "card": {
                "id": card_id,
                "canonical_text": canonical,
            },
            "canonical_text": canonical,
        }
        if task == "upgrade":
            action["kind"] = "deck_upgrade"
            action["action_id"] = f"deck_upgrade:{index}"
            action["upgrade_preview"] = {"available": True}
        else:
            action["kind"] = "card_selection"
            action["action_id"] = f"card_selection:{index}"
        actions.append(action)
        if card_id in selected_ids:
            positive[index] = 1.0

    return actions, positive, selected_count


def build_encoded_sample(row: dict[str, Any], task: str, obs_encoder: DictObservationEncoder, task_context: dict[str, Any]) -> dict[str, Any]:
    if task in BUILD_V2_CANDIDATE_TASKS:
        actions, label_index = build_v2_candidate_actions(row, task)
        phase = {
            "regular_card_reward": "card_reward",
            "event_card_bundle": "card_selection",
            "ancient_choice": "event",
            "relic_choice_step": "treasure",
            "potion_choice_step": "reward",
            "smith_target": "deck_upgrade",
            "remove_card_step": "card_selection",
            "transform_card_step": "card_selection",
            "shop_relic_pick_step": "shop",
            "shop_potion_pick_step": "shop",
            "shop_remove_target_step": "card_selection",
        }.get(task, "reward")
        selected_count = max(int(row.get("selection_steps_total") or 1), 1)
        obs = build_base_obs(
            row,
            task=task,
            phase=phase,
            option_count=len(actions),
            can_skip=any(action.get("skip") for action in actions),
            selected_count=selected_count,
        )
        return {"obs": obs_encoder.encode(obs, actions), "sample_id": row["sample_id"], "label_index": label_index}

    if task in AUXILIARY_TASKS:
        actions, label_multi_hot = build_auxiliary_actions(row, task)
        obs = build_base_obs(
            row,
            task=task,
            phase="shop",
            option_count=len(actions),
            can_skip=False,
            selected_count=0,
        )
        return {
            "obs": obs_encoder.encode(obs, actions),
            "sample_id": row["sample_id"],
            "label_multi_hot": label_multi_hot,
        }

    if task == "card_choice":
        actions, label_index = build_card_reward_actions(row)
        obs = build_base_obs(row, task=task, phase="card_reward", option_count=len(actions), can_skip=any(action.get("skip") for action in actions), selected_count=0)
        return {"obs": obs_encoder.encode(obs, actions), "sample_id": row["sample_id"], "label_index": label_index}

    if task == "ancient_choice":
        actions, label_index = build_single_choice_actions(row, kind="event_option", prefix="ancient_choice")
        obs = build_base_obs(row, task=task, phase="event", option_count=len(actions), can_skip=False, selected_count=0)
        return {"obs": obs_encoder.encode(obs, actions), "sample_id": row["sample_id"], "label_index": label_index}

    if task == "relic_choice":
        actions, label_index = build_single_choice_actions(row, kind="treasure_relic", prefix="relic_choice")
        obs = build_base_obs(row, task=task, phase="treasure", option_count=len(actions), can_skip=False, selected_count=0)
        return {"obs": obs_encoder.encode(obs, actions), "sample_id": row["sample_id"], "label_index": label_index}

    if task == "potion_choice":
        actions, label_index = build_single_choice_actions(row, kind="reward", prefix="potion_choice")
        obs = build_base_obs(row, task=task, phase="reward", option_count=len(actions), can_skip=False, selected_count=0)
        return {"obs": obs_encoder.encode(obs, actions), "sample_id": row["sample_id"], "label_index": label_index}

    if task in ROUTE_TASKS:
        actions, label_index = build_route_actions(row, task=task)
        obs = build_base_obs(row, task=task, phase="map", option_count=len(actions), can_skip=False, selected_count=0)
        return {"obs": obs_encoder.encode(obs, actions), "sample_id": row["sample_id"], "label_index": label_index}

    if task in ACTION_ONLY_CLASS_TASKS | BUILD_V2_CLASS_TASKS:
        actions, label_index = build_classification_actions(row, task, task_context)
        phase = "shop" if task == "shop_remove_binary" else "rest_site"
        obs = build_base_obs(row, task=task, phase=phase, option_count=len(actions), can_skip=False, selected_count=0)
        return {"obs": obs_encoder.encode(obs, actions), "sample_id": row["sample_id"], "label_index": label_index}

    if task in ACTION_ONLY_CARD_TASKS:
        phase = "deck_upgrade" if task == "upgrade" else "card_selection"
        actions, label_multi_hot, selected_count = build_cardset_actions(row, task=task)
        obs = build_base_obs(row, task=task, phase=phase, option_count=len(actions), can_skip=False, selected_count=selected_count)
        return {
            "obs": obs_encoder.encode(obs, actions),
            "sample_id": row["sample_id"],
            "label_multi_hot": label_multi_hot,
            "selected_count": selected_count,
        }

    raise ValueError(f"Unsupported task: {task}")


def make_online_collate_fn(task: str, obs_encoder: DictObservationEncoder, task_context: dict[str, Any]):
    def collate(batch: list[dict[str, Any]]) -> dict[str, Any]:
        encoded_samples = [build_encoded_sample(row, task, obs_encoder, task_context) for row in batch]
        obs_keys = encoded_samples[0]["obs"].keys()
        obs = {
            key: torch.tensor(np.stack([sample["obs"][key] for sample in encoded_samples]), dtype=torch.float32)
            for key in obs_keys
        }
        payload: dict[str, Any] = {
            "task": task,
            "obs": obs,
            "sample_ids": [sample["sample_id"] for sample in encoded_samples],
        }
        if task in ACTION_ONLY_CARD_TASKS:
            payload["label_multi_hot"] = torch.tensor(
                np.stack([sample["label_multi_hot"] for sample in encoded_samples]),
                dtype=torch.float32,
            )
            payload["selected_count"] = torch.tensor(
                [int(sample["selected_count"]) for sample in encoded_samples],
                dtype=torch.long,
            )
        elif task in AUXILIARY_TASKS:
            payload["label_multi_hot"] = torch.tensor(
                np.stack([sample["label_multi_hot"] for sample in encoded_samples]),
                dtype=torch.float32,
            )
        else:
            payload["label_index"] = torch.tensor(
                [int(sample["label_index"]) for sample in encoded_samples],
                dtype=torch.long,
            )
        return payload

    return collate


def _scalar_value(values: list[float], index: int) -> float:
    if index >= len(values):
        return 0.0
    try:
        return float(values[index])
    except (TypeError, ValueError):
        return 0.0


def compute_loss_and_accuracy(task: str, logits: torch.Tensor, batch: dict[str, Any]) -> tuple[torch.Tensor, float]:
    if task in ACTION_ONLY_CARD_TASKS:
        targets = batch["label_multi_hot"]
        valid = batch["obs"]["action_mask"] > 0.5
        valid_logits = logits[valid]
        valid_targets = targets[valid]
        positives = float(valid_targets.sum().item())
        negatives = float(valid_targets.numel() - positives)
        pos_weight = torch.tensor(
            max(negatives / max(positives, 1.0), 1.0),
            device=logits.device,
            dtype=logits.dtype,
        )
        loss = F.binary_cross_entropy_with_logits(valid_logits, valid_targets, pos_weight=pos_weight)
        accuracy = cardset_exact_match_accuracy(logits, targets, valid, batch["selected_count"])
        return loss, accuracy

    if task in AUXILIARY_TASKS:
        targets = batch["label_multi_hot"]
        valid = batch["obs"]["action_mask"] > 0.5
        valid_logits = logits[valid]
        valid_targets = targets[valid]
        positives = float(valid_targets.sum().item())
        negatives = float(valid_targets.numel() - positives)
        pos_weight = torch.tensor(
            max(negatives / max(positives, 1.0), 1.0),
            device=logits.device,
            dtype=logits.dtype,
        )
        loss = F.binary_cross_entropy_with_logits(valid_logits, valid_targets, pos_weight=pos_weight)
        accuracy = auxiliary_binary_accuracy(logits, targets, valid)
        return loss, accuracy

    labels = batch["label_index"]
    loss = F.cross_entropy(logits, labels)
    accuracy = float((logits.argmax(dim=1) == labels).float().mean().item())
    return loss, accuracy


def cardset_exact_match_accuracy(
    logits: torch.Tensor,
    targets: torch.Tensor,
    valid_mask: torch.Tensor,
    selected_count: torch.Tensor,
) -> float:
    matches = []
    for row_index in range(logits.shape[0]):
        valid_indices = valid_mask[row_index].nonzero(as_tuple=False).reshape(-1)
        target_indices = (targets[row_index] > 0.5).nonzero(as_tuple=False).reshape(-1)
        k = min(int(selected_count[row_index].item()), int(valid_indices.numel()))
        predicted_mask = torch.zeros_like(valid_mask[row_index], dtype=torch.bool)
        if k > 0 and valid_indices.numel() > 0:
            topk = logits[row_index, valid_indices].topk(k=k).indices
            predicted_mask[valid_indices[topk]] = True
        target_mask = torch.zeros_like(valid_mask[row_index], dtype=torch.bool)
        target_mask[target_indices] = True
        matches.append(bool(torch.equal(predicted_mask[valid_indices], target_mask[valid_indices])))
    return float(np.mean(matches)) if matches else 0.0


def auxiliary_binary_accuracy(
    logits: torch.Tensor,
    targets: torch.Tensor,
    valid_mask: torch.Tensor,
) -> float:
    predicted = (logits > 0).float()
    correct = (predicted == targets).float()
    masked = correct * valid_mask.float()
    total = float(valid_mask.sum().item())
    if total <= 0:
        return 0.0
    return float(masked.sum().item() / total)


def evaluate_multitask(
    model: MaskablePPO,
    eval_loaders: dict[str, DataLoader],
    *,
    device: torch.device,
    max_eval_batches: int | None,
) -> dict[str, dict[str, float]]:
    model.policy.eval()
    metrics: dict[str, dict[str, float]] = {}
    with torch.no_grad():
        for task, loader in eval_loaders.items():
            total_loss = 0.0
            total_correct = 0.0
            total_examples = 0
            for batch_index, batch in enumerate(loader):
                if max_eval_batches is not None and batch_index >= max_eval_batches:
                    break
                batch = move_batch_to_device(batch, device)
                logits = model.policy.score_action_logits(batch["obs"], action_masks=batch["obs"]["action_mask"])
                loss, accuracy = compute_loss_and_accuracy(task, logits, batch)
                batch_size = batch["obs"]["scalars"].shape[0]
                total_loss += float(loss.item()) * batch_size
                total_correct += accuracy * batch_size
                total_examples += batch_size
            metrics[task] = {
                "loss": total_loss / max(total_examples, 1),
                "accuracy": total_correct / max(total_examples, 1),
                "examples": total_examples,
            }
    return metrics


def build_metadata(
    *,
    config: OfflineOnlineConfig,
    policy_kwargs: dict[str, Any],
    use_text: bool,
    text_model_name: str | None,
    source_timesteps: int,
    epoch: int,
    best_macro_accuracy: float,
    train_rows_by_task: dict[str, list[dict[str, Any]]],
    eval_rows_by_task: dict[str, list[dict[str, Any]]],
) -> dict[str, Any]:
    return {
        "format": "sts2-online-policy-v1",
        "policy_class": "sts2_env.model.STS2CandidateScoringPolicy",
        "policy_kwargs": policy_kwargs,
        "use_text": use_text,
        "text_model": text_model_name if use_text else None,
        "timesteps": int(source_timesteps),
        "mode": "offline_pretrain",
        "character": None,
        "offline_tasks": list(config.tasks),
        "offline_epoch": int(epoch),
        "offline_best_macro_accuracy": float(best_macro_accuracy),
        "offline_config": asdict(config),
        "task_repeat_factors": config.task_repeat_factors or {task: 1 for task in config.tasks},
        "source_checkpoint": config.init_checkpoint,
        "train_rows": {task: len(rows) for task, rows in train_rows_by_task.items()},
        "eval_rows": {task: len(rows) for task, rows in eval_rows_by_task.items()},
    }


def build_task_context(
    train_rows_by_task: dict[str, list[dict[str, Any]]],
    eval_rows_by_task: dict[str, list[dict[str, Any]]],
) -> dict[str, Any]:
    rest_site_candidates = set(REST_SITE_FALLBACK_ACTIONS)
    for rows in (
        train_rows_by_task.get("rest_site", []),
        eval_rows_by_task.get("rest_site", []),
        train_rows_by_task.get("rest_action", []),
        eval_rows_by_task.get("rest_action", []),
    ):
        for row in rows:
            label = row.get("label")
            if label:
                rest_site_candidates.add(str(label))
    return {
        "rest_site_candidates": sorted(rest_site_candidates),
        "shop_remove_binary_candidates": list(SHOP_REMOVE_BINARY_CANDIDATES),
    }


def train(config: OfflineOnlineConfig) -> dict[str, Any]:
    set_seed(config.seed)
    device = torch.device(config.device)

    init_metadata = load_online_checkpoint_metadata(config.init_checkpoint) if config.init_checkpoint else None
    use_text = False if config.no_text else bool((init_metadata or {}).get("use_text", True))
    text_model_name = (init_metadata or {}).get("text_model") or config.text_model
    policy_kwargs = (init_metadata or {}).get("policy_kwargs") or build_policy_kwargs(config)

    if use_text:
        from sts2_env.text_encoder import get_text_encoder

        get_text_encoder(model_name=text_model_name).ensure_ready()

    obs_encoder = DictObservationEncoder(use_text=use_text)

    train_rows_by_task: dict[str, list[dict[str, Any]]] = {}
    eval_rows_by_task: dict[str, list[dict[str, Any]]] = {}
    for task in config.tasks:
        train_rows = load_task_rows(
            config.dataset_root,
            task,
            fmt=config.dataset_format,
            partition_kind=config.partition_kind,
            partition_value=config.partition_value,
            split="train",
        )
        eval_rows = load_task_rows(
            config.dataset_root,
            task,
            fmt=config.dataset_format,
            partition_kind=config.partition_kind,
            partition_value=config.partition_value,
            split=config.eval_split,
        )
        if not eval_rows:
            eval_rows = load_task_rows(
                config.dataset_root,
                task,
                fmt=config.dataset_format,
                partition_kind=config.partition_kind,
                partition_value=config.partition_value,
                split="test",
            )
        if not eval_rows:
            eval_rows = list(train_rows)
        if task in ROUTE_TASKS:
            train_rows, train_route_diag = filter_route_rows(train_rows)
            eval_rows, eval_route_diag = filter_route_rows(eval_rows)
            if not train_rows or not eval_rows:
                raise SystemExit(
                    f"{ROUTE_SUPERVISION_ERROR} task={task} "
                    f"(train accepted={train_route_diag['accepted']}, "
                    f"chosen_path_only={train_route_diag['chosen_path_only']}, "
                    f"missing_candidates={train_route_diag['missing_candidates']}; "
                    f"eval accepted={eval_route_diag['accepted']}, "
                    f"chosen_path_only={eval_route_diag['chosen_path_only']}, "
                    f"missing_candidates={eval_route_diag['missing_candidates']})"
                )
        if not train_rows:
            raise SystemExit(f"No training rows found for task={task}")
        if not eval_rows:
            raise SystemExit(f"No evaluation rows found for task={task}")
        train_rows_by_task[task] = train_rows
        eval_rows_by_task[task] = eval_rows

    task_context = build_task_context(train_rows_by_task, eval_rows_by_task)
    weighted_task_order = build_weighted_task_order(config.tasks, config.task_repeat_factors)

    train_loaders: dict[str, DataLoader] = {}
    eval_loaders: dict[str, DataLoader] = {}
    for task in config.tasks:
        collate_fn = make_online_collate_fn(task, obs_encoder, task_context)
        train_loaders[task] = DataLoader(
            OfflineRowsDataset(train_rows_by_task[task]),
            batch_size=config.batch_size,
            shuffle=True,
            num_workers=config.num_workers,
            collate_fn=collate_fn,
        )
        eval_loaders[task] = DataLoader(
            OfflineRowsDataset(eval_rows_by_task[task]),
            batch_size=config.batch_size,
            shuffle=False,
            num_workers=config.num_workers,
            collate_fn=collate_fn,
        )

    env = make_dummy_env(obs_encoder.obs_space)
    model = MaskablePPO(
        STS2CandidateScoringPolicy,
        env,
        learning_rate=config.learning_rate,
        n_steps=64,
        batch_size=min(config.batch_size, 64),
        n_epochs=1,
        gamma=0.99,
        gae_lambda=0.95,
        clip_range=0.2,
        ent_coef=0.0,
        vf_coef=0.5,
        max_grad_norm=0.5,
        policy_kwargs=policy_kwargs,
        verbose=0,
        device=config.device,
    )
    if config.init_checkpoint:
        loaded_metadata = load_online_policy_state_dict(model, config.init_checkpoint, device=config.device, strict=True)
        source_timesteps = int(loaded_metadata.get("timesteps", 0))
    else:
        source_timesteps = 0

    optimizer = torch.optim.AdamW(
        model.policy.parameters(),
        lr=config.learning_rate,
        weight_decay=config.weight_decay,
    )

    run_name = config.run_name or f"online-pretrain-{config.partition_kind or 'root'}-{config.partition_value or 'all'}-{time.strftime('%Y%m%d-%H%M%S')}"
    run_dir = Path(config.out_dir) / run_name
    run_dir.mkdir(parents=True, exist_ok=True)
    metrics_path = run_dir / "metrics.jsonl"

    best_macro_accuracy = -1.0
    best_dir = run_dir / "best"

    for epoch in range(1, config.epochs + 1):
        model.policy.train()
        train_stats = {task: {"loss": 0.0, "correct": 0.0, "examples": 0} for task in config.tasks}

        for batch_index, (task, batch) in enumerate(iter_round_robin(train_loaders, weighted_task_order)):
            if config.max_train_batches is not None and batch_index >= config.max_train_batches:
                break
            batch = move_batch_to_device(batch, device)
            optimizer.zero_grad(set_to_none=True)
            logits = model.policy.score_action_logits(batch["obs"], action_masks=batch["obs"]["action_mask"])
            loss, accuracy = compute_loss_and_accuracy(task, logits, batch)
            loss.backward()
            optimizer.step()

            batch_size = batch["obs"]["scalars"].shape[0]
            train_stats[task]["loss"] += float(loss.item()) * batch_size
            train_stats[task]["correct"] += accuracy * batch_size
            train_stats[task]["examples"] += batch_size

        eval_metrics = evaluate_multitask(
            model,
            eval_loaders,
            device=device,
            max_eval_batches=config.max_eval_batches,
        )
        train_metrics = {
            task: {
                "loss": stats["loss"] / max(stats["examples"], 1),
                "accuracy": stats["correct"] / max(stats["examples"], 1),
                "examples": stats["examples"],
            }
            for task, stats in train_stats.items()
        }
        macro_eval_accuracy = float(np.mean([metric["accuracy"] for metric in eval_metrics.values()])) if eval_metrics else 0.0

        epoch_metrics: dict[str, Any] = {"epoch": epoch, "eval_macro_accuracy": macro_eval_accuracy}
        for task in config.tasks:
            epoch_metrics[f"{task}_train_loss"] = train_metrics[task]["loss"]
            epoch_metrics[f"{task}_train_accuracy"] = train_metrics[task]["accuracy"]
            epoch_metrics[f"{task}_eval_loss"] = eval_metrics[task]["loss"]
            epoch_metrics[f"{task}_eval_accuracy"] = eval_metrics[task]["accuracy"]
        with metrics_path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(epoch_metrics, ensure_ascii=False))
            handle.write("\n")

        summary_line = " ".join(
            f"{task}:train={train_metrics[task]['accuracy']:.3f}/eval={eval_metrics[task]['accuracy']:.3f}"
            for task in config.tasks
        )
        print(f"[epoch {epoch:02d}] macro_eval={macro_eval_accuracy:.4f} {summary_line}")

        if macro_eval_accuracy > best_macro_accuracy:
            best_macro_accuracy = macro_eval_accuracy
            save_online_checkpoint(
                model,
                best_dir,
                metadata=build_metadata(
                    config=config,
                    policy_kwargs=policy_kwargs,
                    use_text=use_text,
                    text_model_name=text_model_name,
                    source_timesteps=source_timesteps,
                    epoch=epoch,
                    best_macro_accuracy=best_macro_accuracy,
                    train_rows_by_task=train_rows_by_task,
                    eval_rows_by_task=eval_rows_by_task,
                ),
            )

    final_dir = run_dir / "final"
    save_online_checkpoint(
        model,
        final_dir,
        metadata=build_metadata(
            config=config,
            policy_kwargs=policy_kwargs,
            use_text=use_text,
            text_model_name=text_model_name,
            source_timesteps=source_timesteps,
            epoch=config.epochs,
            best_macro_accuracy=best_macro_accuracy,
            train_rows_by_task=train_rows_by_task,
            eval_rows_by_task=eval_rows_by_task,
        ),
    )

    if use_text:
        try:
            from sts2_env.text_encoder import get_text_encoder

            get_text_encoder().save_cache()
        except Exception:
            pass

    summary = {
        "run_dir": str(run_dir),
        "best_dir": str(best_dir),
        "final_dir": str(final_dir),
        "tasks": config.tasks,
        "best_macro_accuracy": best_macro_accuracy,
        "train_rows": {task: len(rows) for task, rows in train_rows_by_task.items()},
        "eval_rows": {task: len(rows) for task, rows in eval_rows_by_task.items()},
        "source_checkpoint": config.init_checkpoint,
        "source_timesteps": source_timesteps,
    }
    (run_dir / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    env.close()
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description="Offline multi-task pretraining on the online STS2 policy.")
    parser.add_argument("--tasks", default=",".join(DEFAULT_TASKS), type=str)
    parser.add_argument("--dataset-root", required=True, type=str)
    parser.add_argument("--dataset-format", default="parquet", choices=["parquet", "jsonl"])
    parser.add_argument("--partition-kind", default=None, choices=["build_id", "build_family"])
    parser.add_argument("--partition-value", default=None, type=str)
    parser.add_argument("--out-dir", default="offline_runs", type=str)
    parser.add_argument("--run-name", default=None, type=str)
    parser.add_argument("--init-checkpoint", default=None, type=str)
    parser.add_argument("--batch-size", default=64, type=int)
    parser.add_argument("--epochs", default=20, type=int)
    parser.add_argument("--learning-rate", default=1e-4, type=float)
    parser.add_argument("--weight-decay", default=1e-4, type=float)
    parser.add_argument("--device", default="cpu", type=str)
    parser.add_argument("--seed", default=7, type=int)
    parser.add_argument("--num-workers", default=0, type=int)
    parser.add_argument("--eval-split", default="val", choices=["val", "test"])
    parser.add_argument("--max-train-batches", default=None, type=int)
    parser.add_argument("--max-eval-batches", default=None, type=int)
    parser.add_argument(
        "--task-repeat-factors",
        default=None,
        type=str,
        help="Comma-separated task:factor sampling multipliers, e.g. smith_target:4,transform_card_step:5.",
    )
    parser.add_argument("--no-text", action="store_true", default=False)
    parser.add_argument("--text-model", default="BAAI/bge-small-zh-v1.5", type=str)
    parser.add_argument("--combat-embed-dim", type=int, default=64)
    parser.add_argument("--build-embed-dim", type=int, default=64)
    parser.add_argument("--route-embed-dim", type=int, default=48)
    parser.add_argument("--n-heads", type=int, default=2)
    parser.add_argument("--text-proj-dim", type=int, default=32)
    parser.add_argument("--context-text-dim", type=int, default=48)
    parser.add_argument("--shared-hidden-dim", type=int, default=96)
    parser.add_argument("--shared-output-dim", type=int, default=128)
    parser.add_argument("--combat-scorer-hidden", type=int, default=128)
    parser.add_argument("--build-scorer-hidden", type=int, default=96)
    parser.add_argument("--route-scorer-hidden", type=int, default=64)
    parser.add_argument("--critic-domain-dim", type=int, default=64)
    args = parser.parse_args()

    tasks = parse_tasks(args.tasks)
    config = OfflineOnlineConfig(
        tasks=tasks,
        dataset_root=args.dataset_root,
        dataset_format=args.dataset_format,
        partition_kind=args.partition_kind,
        partition_value=args.partition_value,
        out_dir=args.out_dir,
        run_name=args.run_name,
        init_checkpoint=args.init_checkpoint,
        batch_size=args.batch_size,
        epochs=args.epochs,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        device=args.device,
        seed=args.seed,
        num_workers=args.num_workers,
        eval_split=args.eval_split,
        max_train_batches=args.max_train_batches,
        max_eval_batches=args.max_eval_batches,
        task_repeat_factors=parse_task_repeat_factors(args.task_repeat_factors, tasks),
        no_text=args.no_text,
        text_model=args.text_model,
        combat_embed_dim=args.combat_embed_dim,
        build_embed_dim=args.build_embed_dim,
        route_embed_dim=args.route_embed_dim,
        n_heads=args.n_heads,
        text_proj_dim=args.text_proj_dim,
        context_text_dim=args.context_text_dim,
        shared_hidden_dim=args.shared_hidden_dim,
        shared_output_dim=args.shared_output_dim,
        combat_scorer_hidden=args.combat_scorer_hidden,
        build_scorer_hidden=args.build_scorer_hidden,
        route_scorer_hidden=args.route_scorer_hidden,
        critic_domain_dim=args.critic_domain_dim,
    )
    summary = train(config)
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
