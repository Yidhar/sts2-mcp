"""Episode-level shop telemetry helpers.

The STS2 bridge exposes card removal as a shop purchase:

``shop_action="buy"`` + ``item.item_kind="card_removal"``

Older Python-side telemetry only looked for ``"remove"`` in ``shop_action``.
That made remove actions nearly invisible in replay diagnostics and TensorBoard.
This module keeps the shop bookkeeping out of ``self_play.py`` while producing
actionable counters for open/buy/remove/leave decisions.
"""

from __future__ import annotations

from dataclasses import dataclass
import json
import math
import time
from typing import Any

from muzero.diagnostics.deck_build_metrics import (
    compute_deck_quality_summary,
    extract_deck_cards_from_obs_like,
)


SHOP_REMOVE_KINDS = {"card_removal", "remove", "removal", "purge"}


SHOP_TB_KEYS: tuple[tuple[str, str], ...] = (
    ("seen", "shop_seen_count"),
    ("open_rate", "shop_open_rate"),
    ("buy_any_rate", "shop_buy_any_rate"),
    ("buy_card_rate", "shop_buy_card_rate"),
    ("buy_relic_rate", "shop_buy_relic_rate"),
    ("buy_potion_rate", "shop_buy_potion_rate"),
    ("remove_rate", "shop_remove_rate"),
    ("leave_rate", "shop_leave_rate"),
    ("back_rate", "shop_back_rate"),
    ("remove_available_rate", "shop_remove_available_rate"),
    ("remove_affordable_rate", "shop_remove_affordable_rate"),
    ("leave_with_gold_ge_100_rate", "shop_leave_with_gold_ge_100_rate"),
    ("leave_with_remove_affordable_rate", "shop_leave_with_remove_affordable_rate"),
    ("deck_context_present_rate", "shop_deck_context_present_rate"),
    ("deck_size_mean", "shop_deck_size_mean"),
    ("starter_count_mean", "shop_starter_count_mean"),
    ("starter_ratio_mean", "shop_starter_ratio_mean"),
    ("nonstarter_count_mean", "shop_nonstarter_count_mean"),
    ("starter_heavy_rate", "shop_starter_heavy_rate"),
    ("buy_with_affordable_remove_rate", "shop_buy_with_affordable_remove_rate"),
    ("buy_blocks_affordable_remove_rate", "shop_buy_blocks_affordable_remove_rate"),
    (
        "buy_blocks_affordable_remove_starter_heavy_rate",
        "shop_buy_blocks_affordable_remove_starter_heavy_rate",
    ),
    ("gold_before_mean", "shop_gold_before_mean"),
    ("selected_cost_mean", "shop_selected_cost_mean"),
    ("affordable_item_count_mean", "shop_affordable_item_count_mean"),
)


def _safe_float(value: Any, default: float = 0.0) -> float:
    try:
        out = float(value)
    except (TypeError, ValueError):
        return float(default)
    return out if math.isfinite(out) else float(default)


def _safe_bool(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        return bool(value)
    if isinstance(value, str):
        return value.strip().lower() in {"1", "true", "yes", "y", "on"}
    return bool(value)


def _lower(value: Any) -> str:
    return str(value or "").strip().lower()


def _shop_item(action: dict[str, Any] | None) -> dict[str, Any]:
    if not isinstance(action, dict):
        return {}
    item = action.get("item")
    if isinstance(item, dict):
        return item
    # Compact signatures flatten the same fields.
    compact_kind = action.get("shop_item_kind")
    if compact_kind is not None:
        return {
            "item_kind": compact_kind,
            "title": action.get("shop_item_title"),
            "cost": action.get("shop_item_cost"),
            "is_affordable": action.get("shop_item_affordable"),
            "used": action.get("shop_item_used"),
        }
    return {}


def shop_item_kind(action: dict[str, Any] | None) -> str:
    item = _shop_item(action)
    semantic = action.get("semantic") if isinstance(action, dict) and isinstance(action.get("semantic"), dict) else {}
    return _lower(item.get("item_kind") or item.get("kind") or semantic.get("shop_item_kind"))


def _shop_action(action: dict[str, Any] | None) -> str:
    if not isinstance(action, dict):
        return ""
    explicit = _lower(action.get("shop_action"))
    if explicit:
        return explicit
    action_id = _lower(action.get("action_id"))
    if action_id.startswith("shop:"):
        parts = action_id.split(":")
        if len(parts) >= 2:
            return parts[1]
    return ""


def is_shop_action(action: Any) -> bool:
    if not isinstance(action, dict):
        return False
    kind = _lower(action.get("kind"))
    action_id = _lower(action.get("action_id"))
    # Choosing a map node whose room is a future shop is a route decision, not
    # an in-shop merchant decision.  Keep those out of build/shop_* telemetry
    # even if a bridge/semantic payload labels the destination room as "shop".
    if kind in {"map", "route"} or action_id.startswith(("map:", "route:")):
        return False
    if kind == "shop":
        return True
    if _shop_action(action):
        return True
    if action_id.startswith("shop:"):
        return True
    semantic = action.get("semantic")
    return isinstance(semantic, dict) and _lower(semantic.get("family")) == "shop"


def _item_affordable(action: dict[str, Any] | None) -> bool:
    item = _shop_item(action)
    value = item.get("is_affordable")
    if value is None:
        value = item.get("affordable")
    if value is None:
        value = item.get("enough_gold")
    if value is None and isinstance(action, dict):
        value = action.get("shop_item_affordable")
    return _safe_bool(value)


def _item_cost(action: dict[str, Any] | None) -> float:
    item = _shop_item(action)
    if item.get("cost") is not None:
        return _safe_float(item.get("cost"), 0.0)
    if isinstance(action, dict):
        return _safe_float(action.get("shop_item_cost"), 0.0)
    return 0.0


def _item_title(action: dict[str, Any] | None) -> str:
    item = _shop_item(action)
    if item.get("title") is not None:
        return str(item.get("title") or "")
    if isinstance(action, dict):
        for key in ("shop_item_title", "title", "label", "name"):
            if action.get(key):
                return str(action.get(key))
        for nested_key in ("card", "relic", "potion"):
            payload = action.get(nested_key)
            if isinstance(payload, dict) and (payload.get("title") or payload.get("name")):
                return str(payload.get("title") or payload.get("name"))
    return ""


def _shop_is_remove(action: dict[str, Any] | None) -> bool:
    item = _shop_item(action)
    semantic = action.get("semantic") if isinstance(action, dict) and isinstance(action.get("semantic"), dict) else {}
    return bool(
        "remove" in _shop_action(action)
        or shop_item_kind(action) in SHOP_REMOVE_KINDS
        or "remove" in _lower(item.get("title") or _item_title(action))
        or "purge" in _lower(item.get("title") or _item_title(action))
        or _safe_bool(semantic.get("shop_is_remove"))
    )


def shop_action_kind(action: dict[str, Any] | None) -> str:
    """Classify a raw or compact shop action into stable telemetry buckets."""

    if not is_shop_action(action):
        return ""
    shop_action = _shop_action(action)
    item_kind = shop_item_kind(action)
    if _shop_is_remove(action):
        return "remove"
    if shop_action == "open" or "open" in shop_action:
        return "open"
    if shop_action == "leave" or "leave" in shop_action:
        return "leave"
    if shop_action == "back" or "back" in shop_action:
        return "back"
    if shop_action == "buy" or "buy" in shop_action or "purchase" in shop_action:
        if item_kind in {"card", "relic", "potion"}:
            return f"buy_{item_kind}"
        return "buy_other"
    return "other"


def _compact_shop_action(action: dict[str, Any] | None, *, index: int | None = None, prob: float | None = None) -> dict[str, Any]:
    if not isinstance(action, dict):
        return {}
    out: dict[str, Any] = {
        "index": int(index) if index is not None else None,
        "prob": float(prob) if prob is not None else None,
        "kind": action.get("kind"),
        "action_id": action.get("action_id"),
        "shop_action": _shop_action(action),
        "action_kind": shop_action_kind(action),
        "item_kind": shop_item_kind(action),
        "title": _item_title(action),
        "cost": _item_cost(action),
        "affordable": _item_affordable(action),
        "used": _safe_bool(_shop_item(action).get("used")),
    }
    return {key: value for key, value in out.items() if value not in (None, "")}


def shop_available_summary(legal_actions: list[Any]) -> dict[str, Any]:
    shop_actions = [action for action in legal_actions if is_shop_action(action)]
    kinds = [shop_action_kind(action) for action in shop_actions]
    remove_actions = [action for action in shop_actions if shop_action_kind(action) == "remove"]
    remove_costs = [_item_cost(action) for action in remove_actions if _item_cost(action) > 0.0]
    affordable_remove_costs = [
        _item_cost(action)
        for action in remove_actions
        if _item_affordable(action) and _item_cost(action) > 0.0
    ]
    affordable_actions = [
        action
        for action in shop_actions
        if shop_action_kind(action) in {"buy_card", "buy_relic", "buy_potion", "buy_other", "remove"}
        and _item_affordable(action)
    ]
    remove_affordable = any(_item_affordable(action) for action in remove_actions)
    return {
        "shop_action_count": float(len(shop_actions)),
        "open_available": any(kind == "open" for kind in kinds),
        "buy_available": any(kind.startswith("buy_") for kind in kinds),
        "remove_available": bool(remove_actions),
        "remove_affordable": bool(remove_affordable),
        "leave_available": any(kind == "leave" for kind in kinds),
        "back_available": any(kind == "back" for kind in kinds),
        "affordable_item_count": float(len(affordable_actions)),
        "available_card_count": float(sum(1 for action in shop_actions if shop_item_kind(action) == "card")),
        "available_relic_count": float(sum(1 for action in shop_actions if shop_item_kind(action) == "relic")),
        "available_potion_count": float(sum(1 for action in shop_actions if shop_item_kind(action) == "potion")),
        "available_remove_count": float(len(remove_actions)),
        "remove_cost_min": float(min(remove_costs)) if remove_costs else 0.0,
        "remove_affordable_cost_min": float(min(affordable_remove_costs)) if affordable_remove_costs else 0.0,
    }


def _top_policy_actions(legal_actions: list[Any], search_policy: Any, *, max_topk: int) -> list[dict[str, Any]]:
    if max_topk <= 0 or not isinstance(legal_actions, list):
        return []
    probs: list[float] = []
    try:
        flat = list(search_policy)
        probs = [_safe_float(value, 0.0) for value in flat[: len(legal_actions)]]
    except Exception:
        probs = []
    if len(probs) < len(legal_actions):
        probs.extend([0.0] * (len(legal_actions) - len(probs)))
    indexed = [
        (idx, probs[idx], action)
        for idx, action in enumerate(legal_actions)
        if is_shop_action(action)
    ]
    indexed.sort(key=lambda item: item[1], reverse=True)
    return [
        _compact_shop_action(action, index=idx, prob=prob)
        for idx, prob, action in indexed[: max(1, int(max_topk))]
    ]


def _deck_context(raw_obs: Any) -> dict[str, Any]:
    """Return compact deck/shop context for diagnosing missed Act1 removal.

    This is diagnostic-only: it deliberately does not affect replay schemas or
    model inputs.  The fields answer the operator question "did we buy before
    removing while still starter-heavy?" without requiring a death slice.
    """

    cards = extract_deck_cards_from_obs_like(raw_obs)
    if not cards:
        return {
            "deck_context_present": False,
            "deck_size_before": 0.0,
            "starter_count_before": 0.0,
            "starter_ratio_before": 0.0,
            "nonstarter_count_before": 0.0,
            "starter_heavy_before": False,
            "damage_per_energy_before": 0.0,
            "block_per_energy_before": 0.0,
            "draw_engine_score_before": 0.0,
            "boss_readiness_before": 0.0,
            "elite_readiness_before": 0.0,
        }
    quality = compute_deck_quality_summary(cards)
    deck_size = float(len(cards))
    starter_count = _safe_float(quality.get("starter_count"), 0.0)
    starter_ratio = _safe_float(quality.get("starter_ratio"), 0.0)
    # Act1 shops with 7+ starter cards or >45% starters are the exact failure
    # mode observed in current runs: buy a mediocre card first, lose remove,
    # then die with Strike/Defend still dominating the deck.
    starter_heavy = starter_count >= 7.0 or (deck_size >= 12.0 and starter_ratio >= 0.45)
    return {
        "deck_context_present": True,
        "deck_size_before": deck_size,
        "starter_count_before": starter_count,
        "starter_ratio_before": starter_ratio,
        "nonstarter_count_before": _safe_float(quality.get("nonstarter_count"), max(0.0, deck_size - starter_count)),
        "starter_heavy_before": bool(starter_heavy),
        "damage_per_energy_before": _safe_float(quality.get("raw_avg_damage_per_energy"), 0.0),
        "block_per_energy_before": _safe_float(quality.get("raw_avg_block_per_energy"), 0.0),
        "draw_engine_score_before": _safe_float(quality.get("draw_engine_score"), 0.0),
        "boss_readiness_before": _safe_float(quality.get("boss_readiness_score"), 0.0),
        "elite_readiness_before": _safe_float(quality.get("elite_readiness_score"), 0.0),
    }


def build_shop_choice_payload(
    *,
    decision_domain: str,
    phase: str,
    legal_actions: list[Any],
    chosen_action: dict[str, Any] | None,
    chosen_signature: dict[str, Any] | None,
    selected_index: int,
    progress: dict[str, Any] | None = None,
    gold: Any = None,
    raw_obs: Any = None,
    search_policy: Any = None,
    max_topk: int = 8,
) -> dict[str, Any] | None:
    """Build a compact JSONL/TB payload for one shop decision, if present."""

    legal_actions = legal_actions if isinstance(legal_actions, list) else []
    shop_context = (
        "shop" in _lower(phase)
        or "merchant" in _lower(phase)
        or is_shop_action(chosen_action)
        or is_shop_action(chosen_signature)
        or any(is_shop_action(action) for action in legal_actions)
    )
    if not shop_context:
        return None

    selected = chosen_action if isinstance(chosen_action, dict) else chosen_signature
    selected_kind = shop_action_kind(selected if isinstance(selected, dict) else None)
    summary = shop_available_summary(legal_actions)
    gold_value = _safe_float(gold, 0.0)
    leave_with_gold_ge_100 = selected_kind in {"leave", "back"} and gold_value >= 100.0
    leave_with_remove_affordable = selected_kind in {"leave", "back"} and bool(summary["remove_affordable"])
    deck = _deck_context(raw_obs)
    selected_cost = _item_cost(selected if isinstance(selected, dict) else None)
    selected_is_buy = selected_kind.startswith("buy_")
    selected_buy_with_affordable_remove = bool(selected_is_buy and summary["remove_affordable"])
    remove_affordable_cost = _safe_float(summary.get("remove_affordable_cost_min"), 0.0)
    selected_buy_blocks_affordable_remove = bool(
        selected_buy_with_affordable_remove
        and selected_cost > 0.0
        and remove_affordable_cost > 0.0
        and (gold_value - selected_cost) < remove_affordable_cost
    )
    selected_buy_blocks_affordable_remove_starter_heavy = bool(
        selected_buy_blocks_affordable_remove and _safe_bool(deck.get("starter_heavy_before"))
    )

    payload: dict[str, Any] = {
        "decision_domain": str(decision_domain or ""),
        "phase": str(phase or ""),
        "selected_index": int(selected_index),
        "selected_action_kind": selected_kind or "non_shop",
        "selected_shop_action": _shop_action(selected if isinstance(selected, dict) else None),
        "selected_item_kind": shop_item_kind(selected if isinstance(selected, dict) else None),
        "selected_title": _item_title(selected if isinstance(selected, dict) else None),
        "selected_cost": selected_cost,
        "selected_affordable": _item_affordable(selected if isinstance(selected, dict) else None),
        "gold_before": gold_value,
        "leave_with_gold_ge_100": bool(leave_with_gold_ge_100),
        "leave_with_remove_affordable": bool(leave_with_remove_affordable),
        "selected_buy_with_affordable_remove": bool(selected_buy_with_affordable_remove),
        "selected_buy_blocks_affordable_remove": bool(selected_buy_blocks_affordable_remove),
        "selected_buy_blocks_affordable_remove_starter_heavy": bool(
            selected_buy_blocks_affordable_remove_starter_heavy
        ),
        "selected_action": _compact_shop_action(selected if isinstance(selected, dict) else None, index=selected_index),
        "top_policy_shop_actions": _top_policy_actions(legal_actions, search_policy, max_topk=max_topk),
        **summary,
        **deck,
    }
    if isinstance(progress, dict):
        for key in ("floor", "act_id", "room_type", "room_type_lower", "encounter_id"):
            if key in progress:
                payload[key] = progress.get(key)
    return payload


@dataclass
class ShopEpisodeTracker:
    """Track shop decisions across one episode."""

    seen: int = 0
    open: int = 0
    buy_any: int = 0
    buy_card: int = 0
    buy_relic: int = 0
    buy_potion: int = 0
    remove: int = 0
    leave: int = 0
    back: int = 0
    remove_available: int = 0
    remove_affordable: int = 0
    leave_with_gold_ge_100: int = 0
    leave_with_remove_affordable: int = 0
    deck_context_present: int = 0
    deck_size_sum: float = 0.0
    deck_size_count: int = 0
    starter_count_sum: float = 0.0
    starter_ratio_sum: float = 0.0
    nonstarter_count_sum: float = 0.0
    starter_heavy: int = 0
    buy_with_affordable_remove: int = 0
    buy_blocks_affordable_remove: int = 0
    buy_blocks_affordable_remove_starter_heavy: int = 0
    gold_before_sum: float = 0.0
    gold_before_count: int = 0
    selected_cost_sum: float = 0.0
    selected_cost_count: int = 0
    affordable_item_count_sum: float = 0.0
    affordable_item_count_count: int = 0

    def update(self, payload: dict[str, Any] | None) -> None:
        if not isinstance(payload, dict):
            return
        self.seen += 1
        kind = _lower(payload.get("selected_action_kind"))
        if kind == "open":
            self.open += 1
        elif kind == "remove":
            self.remove += 1
            self.buy_any += 1
        elif kind == "buy_card":
            self.buy_card += 1
            self.buy_any += 1
        elif kind == "buy_relic":
            self.buy_relic += 1
            self.buy_any += 1
        elif kind == "buy_potion":
            self.buy_potion += 1
            self.buy_any += 1
        elif kind == "buy_other":
            self.buy_any += 1
        elif kind == "leave":
            self.leave += 1
        elif kind == "back":
            self.back += 1

        if _safe_bool(payload.get("remove_available")):
            self.remove_available += 1
        if _safe_bool(payload.get("remove_affordable")):
            self.remove_affordable += 1
        if _safe_bool(payload.get("leave_with_gold_ge_100")):
            self.leave_with_gold_ge_100 += 1
        if _safe_bool(payload.get("leave_with_remove_affordable")):
            self.leave_with_remove_affordable += 1
        if _safe_bool(payload.get("deck_context_present")):
            self.deck_context_present += 1
            self.deck_size_sum += _safe_float(payload.get("deck_size_before"), 0.0)
            self.starter_count_sum += _safe_float(payload.get("starter_count_before"), 0.0)
            self.starter_ratio_sum += _safe_float(payload.get("starter_ratio_before"), 0.0)
            self.nonstarter_count_sum += _safe_float(payload.get("nonstarter_count_before"), 0.0)
            self.deck_size_count += 1
        if _safe_bool(payload.get("starter_heavy_before")):
            self.starter_heavy += 1
        if _safe_bool(payload.get("selected_buy_with_affordable_remove")):
            self.buy_with_affordable_remove += 1
        if _safe_bool(payload.get("selected_buy_blocks_affordable_remove")):
            self.buy_blocks_affordable_remove += 1
        if _safe_bool(payload.get("selected_buy_blocks_affordable_remove_starter_heavy")):
            self.buy_blocks_affordable_remove_starter_heavy += 1

        self.gold_before_sum += _safe_float(payload.get("gold_before"), 0.0)
        self.gold_before_count += 1
        selected_cost = _safe_float(payload.get("selected_cost"), 0.0)
        if selected_cost > 0.0:
            self.selected_cost_sum += selected_cost
            self.selected_cost_count += 1
        self.affordable_item_count_sum += _safe_float(payload.get("affordable_item_count"), 0.0)
        self.affordable_item_count_count += 1

    def as_metadata(self) -> dict[str, float]:
        seen_safe = max(int(self.seen), 1)
        return {
            "shop_seen_count": float(self.seen),
            "shop_open_count": float(self.open),
            "shop_buy_any_count": float(self.buy_any),
            "shop_buy_card_count": float(self.buy_card),
            "shop_buy_relic_count": float(self.buy_relic),
            "shop_buy_potion_count": float(self.buy_potion),
            "shop_remove_count": float(self.remove),
            "shop_leave_count": float(self.leave),
            "shop_back_count": float(self.back),
            "shop_open_rate": float(self.open) / float(seen_safe),
            "shop_buy_any_rate": float(self.buy_any) / float(seen_safe),
            "shop_buy_card_rate": float(self.buy_card) / float(seen_safe),
            "shop_buy_relic_rate": float(self.buy_relic) / float(seen_safe),
            "shop_buy_potion_rate": float(self.buy_potion) / float(seen_safe),
            "shop_remove_rate": float(self.remove) / float(seen_safe),
            "shop_leave_rate": float(self.leave) / float(seen_safe),
            "shop_back_rate": float(self.back) / float(seen_safe),
            "shop_remove_available_rate": float(self.remove_available) / float(seen_safe),
            "shop_remove_affordable_rate": float(self.remove_affordable) / float(seen_safe),
            "shop_leave_with_gold_ge_100_rate": float(self.leave_with_gold_ge_100) / float(seen_safe),
            "shop_leave_with_remove_affordable_rate": float(self.leave_with_remove_affordable) / float(seen_safe),
            "shop_deck_context_present_rate": float(self.deck_context_present) / float(seen_safe),
            "shop_deck_size_mean": self.deck_size_sum / float(max(self.deck_size_count, 1)),
            "shop_starter_count_mean": self.starter_count_sum / float(max(self.deck_size_count, 1)),
            "shop_starter_ratio_mean": self.starter_ratio_sum / float(max(self.deck_size_count, 1)),
            "shop_nonstarter_count_mean": self.nonstarter_count_sum / float(max(self.deck_size_count, 1)),
            "shop_starter_heavy_rate": float(self.starter_heavy) / float(seen_safe),
            "shop_buy_with_affordable_remove_rate": float(self.buy_with_affordable_remove) / float(seen_safe),
            "shop_buy_blocks_affordable_remove_rate": float(self.buy_blocks_affordable_remove) / float(seen_safe),
            "shop_buy_blocks_affordable_remove_starter_heavy_rate": (
                float(self.buy_blocks_affordable_remove_starter_heavy) / float(seen_safe)
            ),
            "shop_gold_before_mean": self.gold_before_sum / float(max(self.gold_before_count, 1)),
            "shop_selected_cost_mean": self.selected_cost_sum / float(max(self.selected_cost_count, 1)),
            "shop_affordable_item_count_mean": self.affordable_item_count_sum / float(max(self.affordable_item_count_count, 1)),
        }


def dump_shop_choice_diagnostic(trainer: Any, payload: dict[str, Any] | None) -> None:
    """Append one bounded ``shop_choices.jsonl`` record using trainer paths."""

    if not isinstance(payload, dict):
        return
    if getattr(trainer, "_shop_choice_dump_disabled", False):
        return
    cap = int(getattr(trainer, "_shop_choice_dump_max", 100000))
    count = int(getattr(trainer, "_shop_choice_dump_count", 0) or 0)
    if cap > 0 and count >= cap:
        return
    path_getter = getattr(trainer, "_diagnostic_jsonl_path", None)
    if not callable(path_getter):
        return
    try:
        record = {
            "time": time.time(),
            "global_step": int(getattr(trainer, "total_steps", 0)),
            "episode_id": int(getattr(trainer, "episode_count", 0)),
            **payload,
        }
        path = path_getter("shop_choices.jsonl")
        if path is None:
            return
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("a", encoding="utf-8") as fh:
            fh.write(json.dumps(record, ensure_ascii=False, default=str) + "\n")
        trainer._shop_choice_dump_count = count + 1
    except Exception:
        return
