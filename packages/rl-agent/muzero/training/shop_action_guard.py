"""Narrow shop hard guard for build-domain self-play decisions.

The bridge represents card removal as a shop purchase:

``shop_action="buy"`` + ``item.item_kind="card_removal"``.

This module does **not** try to decide which card/relic/potion is optimal.
That is a long-horizon deck-building problem and should be learned from
offline/build replay.  The guard only blocks two high-confidence failure modes
seen in full-run training:

* leaving a shop without opening the merchant inventory while the player has
  enough gold to plausibly use the shop;
* leaving/closing an opened shop while affordable card removal is legal and
  the deck still contains many starter/junk cards;
* buying an ordinary card that consumes the last card-removal gold while the
  deck is still starter/junk heavy.

It is intentionally isolated from ``self_play.py``/``train.py`` so future shop
policy work can grow here without making the trainer monolithic again.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from muzero.diagnostics.deck_build_metrics import extract_deck_cards_from_obs_like
from muzero.diagnostics.shop_metrics import is_shop_action, shop_action_kind
from sts2_env.observation_v2 import MAX_ACTIONS
from sts2_env.selection_typed import selection_view


SHOP_OPEN_GOLD_THRESHOLD = 50.0
SHOP_REMOVE_STARTER_THRESHOLD = 6
SHOP_REMOVE_STARTER_RATIO_THRESHOLD = 0.50


SHOP_REMOVE_PREMIUM_CARD_ALLOWLIST = {
    # High-confidence Act1 stabilizers / engines: do not let the removal guard
    # veto these, because they can be worth more than thinning one starter.
    "shrug it off",
    "shrug_it_off",
    "耸肩无视",
    "聳肩無視",
    "flame barrier",
    "flame_barrier",
    "火焰屏障",
    "火焰屏障",
    "battle trance",
    "battle_trance",
    "战斗专注",
    "戰鬥專注",
    "offering",
    "献祭",
    "獻祭",
    "impervious",
    "巍不动",
    "岿然不动",
    "岿然不动",
    "armaments",
    "武装",
    "武裝",
    "feel no pain",
    "feel_no_pain",
    "无惧疼痛",
    "無懼疼痛",
    "dark embrace",
    "dark_embrace",
    "黑暗之拥",
    "黑暗之擁",
    "corruption",
    "腐化",
}


SHOP_ACTION_GUARD_SEARCH_SUFFIXES: dict[str, str] = {
    "shop_action_guard_enabled": "shop_action_guard_enabled",
    "shop_action_guard_shop_surface": "shop_action_guard_shop_surface_rate",
    "shop_action_guard_selected_open": "shop_action_guard_selected_open_rate",
    "shop_action_guard_selected_buy": "shop_action_guard_selected_buy_rate",
    "shop_action_guard_selected_remove": "shop_action_guard_selected_remove_rate",
    "shop_action_guard_selected_leave": "shop_action_guard_selected_leave_rate",
    "shop_action_guard_selected_back": "shop_action_guard_selected_back_rate",
    "shop_action_guard_open_available": "shop_action_guard_open_available_rate",
    "shop_action_guard_remove_affordable_available": "shop_action_guard_remove_affordable_available_rate",
    "shop_action_guard_gold": "shop_action_guard_gold_mean",
    "shop_action_guard_starter_count": "shop_action_guard_starter_count_mean",
    "shop_action_guard_junk_count": "shop_action_guard_junk_count_mean",
    "shop_action_guard_open_applicable": "shop_action_guard_open_applicable_rate",
    "shop_action_guard_open_applied": "shop_action_guard_open_applied_rate",
    "shop_action_guard_remove_applicable": "shop_action_guard_remove_applicable_rate",
    "shop_action_guard_remove_applied": "shop_action_guard_remove_applied_rate",
    "shop_action_guard_buy_blocks_remove_applicable": "shop_action_guard_buy_blocks_remove_applicable_rate",
    "shop_action_guard_buy_blocks_remove_applied": "shop_action_guard_buy_blocks_remove_applied_rate",
    "shop_action_guard_buy_blocks_remove_premium_allow": "shop_action_guard_buy_blocks_remove_premium_allow_rate",
    "shop_action_guard_selected_buy_cost": "shop_action_guard_selected_buy_cost_mean",
    "shop_action_guard_remove_cost_min": "shop_action_guard_remove_cost_min_mean",
    "shop_action_guard_gold_after_selected_buy": "shop_action_guard_gold_after_selected_buy_mean",
    "shop_action_guard_remove_selection_surface": "shop_action_guard_remove_selection_surface_rate",
    "shop_action_guard_remove_selection_selected_score": "shop_action_guard_remove_selection_selected_score_mean",
    "shop_action_guard_remove_selection_best_score": "shop_action_guard_remove_selection_best_score_mean",
    "shop_action_guard_remove_selection_applied": "shop_action_guard_remove_selection_applied_rate",
    "shop_action_guard_alignment_error": "shop_action_guard_alignment_error_rate",
    "shop_action_guard_invalid_obs": "shop_action_guard_invalid_obs_rate",
}


def shop_action_guard_metric_keys() -> tuple[str, ...]:
    return tuple(SHOP_ACTION_GUARD_SEARCH_SUFFIXES.keys())


def _lower(value: Any) -> str:
    return str(value or "").strip().lower()


def _safe_float(value: Any, default: float = 0.0) -> float:
    try:
        out = float(value)
    except (TypeError, ValueError):
        return float(default)
    if not np.isfinite(out):
        return float(default)
    return out


def _safe_bool(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        return bool(value)
    if isinstance(value, str):
        return value.strip().lower() in {"1", "true", "yes", "y", "on"}
    return bool(value)


def _shop_item(action: Any) -> dict[str, Any]:
    if not isinstance(action, dict):
        return {}
    item = action.get("item")
    if isinstance(item, dict):
        return item
    if action.get("shop_item_kind") is not None:
        return {
            "item_kind": action.get("shop_item_kind"),
            "cost": action.get("shop_item_cost"),
            "is_affordable": action.get("shop_item_affordable"),
            "used": action.get("shop_item_used"),
            "title": action.get("shop_item_title"),
        }
    return {}


def _item_affordable(action: Any) -> bool:
    item = _shop_item(action)
    value = item.get("is_affordable")
    if value is None:
        value = item.get("affordable")
    if value is None:
        value = item.get("enough_gold")
    if value is None and isinstance(action, dict):
        value = action.get("shop_item_affordable")
    return _safe_bool(value)


def _item_cost(action: Any) -> float:
    item = _shop_item(action)
    for value in (item.get("cost"), item.get("price")):
        if value is not None:
            return _safe_float(value, 0.0)
    if isinstance(action, dict):
        for key in ("shop_item_cost", "cost", "price"):
            if action.get(key) is not None:
                return _safe_float(action.get(key), 0.0)
    return 0.0


def _item_title(action: Any) -> str:
    item = _shop_item(action)
    for value in (item.get("title"), item.get("name"), item.get("id"), item.get("card_id")):
        if value:
            return str(value)
    if isinstance(action, dict):
        for key in ("shop_item_title", "title", "label", "name", "id", "card_id"):
            if action.get(key):
                return str(action.get(key))
        for nested_key in ("card", "relic", "potion", "source"):
            payload = action.get(nested_key)
            if isinstance(payload, dict):
                for key in ("title", "name", "id", "card_id"):
                    if payload.get(key):
                        return str(payload.get(key))
    return ""


def _card_identity(card: dict[str, Any]) -> str:
    return _lower(
        card.get("id")
        or card.get("card_id")
        or card.get("model_id")
        or card.get("modelId")
        or card.get("title")
        or card.get("name")
        or card.get("localized_title")
    )


def _card_type(card: dict[str, Any]) -> str:
    return _lower(card.get("type") or card.get("card_type"))


def _is_starter_card(card: dict[str, Any]) -> bool:
    ident = _card_identity(card)
    if not ident:
        return False
    # Accept both bridge ids (CARD.STRIKE_IRONCLAD) and localized starter
    # titles from diagnostics/death dumps.
    return bool(
        "strike_ironclad" in ident
        or "defend_ironclad" in ident
        or ident in {"strike", "defend", "打击", "防御", "打擊", "防禦"}
    )


def _is_upgraded(card: dict[str, Any]) -> bool:
    if _safe_bool(card.get("upgraded")):
        return True
    try:
        return int(card.get("upgrade_level") or 0) > 0
    except (TypeError, ValueError):
        return False


def _is_junk_card(card: dict[str, Any]) -> bool:
    ctype = _card_type(card)
    if ctype in {"curse", "status"}:
        return True
    ident = _card_identity(card)
    return bool("curse" in ident or "status" in ident or "wound" in ident or "受伤" in ident)


def _selection_card(action: Any) -> dict[str, Any]:
    if not isinstance(action, dict):
        return {}
    card = action.get("card")
    if isinstance(card, dict):
        return card
    source = action.get("source")
    if isinstance(source, dict):
        card = source.get("card")
        if isinstance(card, dict):
            return card
    return {}


def _is_shop_or_deck_remove_selection(action: Any) -> bool:
    if not isinstance(action, dict):
        return False
    view = selection_view(action)
    if view.get("operation_type") != "remove":
        return False
    source = _lower(view.get("source"))
    source_zone = _lower(view.get("source_zone"))
    # Shop removal is the main target.  Some bridge builds only tag the source
    # zone as deck, so accept deck-sourced remove selections as well.  Do not
    # touch hand/combat purge selections.
    return source == "shop" or source_zone == "deck"


def _remove_target_score(card: dict[str, Any]) -> float:
    if not isinstance(card, dict) or not card:
        return 0.0
    if _is_junk_card(card):
        return 100.0
    ident = _card_identity(card)
    upgraded = _is_upgraded(card)
    if "strike_ironclad" in ident or ident in {"strike", "打击", "打擊"}:
        return 65.0 if upgraded else 85.0
    if "defend_ironclad" in ident or ident in {"defend", "防御", "防禦"}:
        return 55.0 if upgraded else 75.0
    if _is_starter_card(card):
        return 50.0 if upgraded else 70.0
    return 0.0


def _deck_counts(raw_obs: Any) -> tuple[int, int, int]:
    cards = extract_deck_cards_from_obs_like(raw_obs) if isinstance(raw_obs, dict) else []
    starter_count = 0
    junk_count = 0
    for card in cards:
        if not isinstance(card, dict):
            continue
        if _is_starter_card(card):
            starter_count += 1
        if _is_junk_card(card):
            junk_count += 1
    return starter_count, junk_count, len(cards)


def _player_gold(raw_obs: Any) -> tuple[float, bool]:
    if not isinstance(raw_obs, dict):
        return 0.0, False
    candidates: list[Any] = [raw_obs.get("gold")]
    player = raw_obs.get("player") if isinstance(raw_obs.get("player"), dict) else {}
    run = raw_obs.get("run") if isinstance(raw_obs.get("run"), dict) else {}
    candidates.extend([player.get("gold"), run.get("gold")])
    for value in candidates:
        if value is None:
            continue
        return _safe_float(value, 0.0), True
    return 0.0, False


def _valid_legal_count(legal_actions: list[Any], action_mask: Any) -> int:
    try:
        mask_np = np.asarray(action_mask, dtype=np.float32).reshape(-1)
    except Exception:
        return 0
    if mask_np.size <= 0:
        return 0
    return min(len(legal_actions), int(mask_np.shape[0]), MAX_ACTIONS)


def _mask_array(action_mask: Any) -> np.ndarray | None:
    try:
        mask_np = np.asarray(action_mask, dtype=np.float32).reshape(-1)
    except Exception:
        return None
    return mask_np if mask_np.size > 0 else None


def _action_source_for_guard(
    *,
    legal_actions: list[Any],
    full_legal_actions: list[Any] | None,
    legal_count: int,
) -> list[Any] | None:
    if isinstance(full_legal_actions, list) and len(full_legal_actions) >= legal_count:
        return full_legal_actions
    if len(legal_actions) >= legal_count:
        return legal_actions
    return None


def _legal_indices_by_kind(
    *,
    action_source: list[Any],
    mask_np: np.ndarray,
    legal_count: int,
    wanted: str,
) -> list[int]:
    out: list[int] = []
    for idx in range(legal_count):
        if mask_np[idx] <= 0:
            continue
        action = action_source[idx]
        if not is_shop_action(action):
            continue
        if shop_action_kind(action) == wanted:
            out.append(int(idx))
    return out


def _affordable_remove_indices(
    *,
    action_source: list[Any],
    mask_np: np.ndarray,
    legal_count: int,
) -> list[int]:
    out: list[int] = []
    for idx in range(legal_count):
        if mask_np[idx] <= 0:
            continue
        action = action_source[idx]
        if is_shop_action(action) and shop_action_kind(action) == "remove" and _item_affordable(action):
            out.append(int(idx))
    return out


def _cheapest_remove_cost(
    *,
    action_source: list[Any],
    remove_indices: list[int],
) -> float:
    costs = [_item_cost(action_source[idx]) for idx in remove_indices if _item_cost(action_source[idx]) > 0.0]
    return float(min(costs)) if costs else 0.0


def _is_premium_card_purchase(action: Any) -> bool:
    """Return True for cards that should not be vetoed by remove-priority.

    This is intentionally a small high-confidence allowlist.  Unknown shop
    cards remain learnable; the hard guard only intervenes when buying them
    would spend the last card-removal gold in a starter-heavy Act1 deck.
    """

    title = _lower(_item_title(action)).replace("-", "_")
    if not title:
        return False
    normalized = title.replace(" ", "_")
    allow = {entry.replace(" ", "_") for entry in SHOP_REMOVE_PREMIUM_CARD_ALLOWLIST}
    return title in SHOP_REMOVE_PREMIUM_CARD_ALLOWLIST or normalized in allow


def _apply_remove_selection_guard(
    *,
    selected_idx: int,
    action_source: list[Any],
    mask_np: np.ndarray,
    legal_count: int,
    search_stats: dict[str, Any],
) -> int | None:
    """Force the actual deck-removal target away from cancel/valuable cards.

    After buying shop card removal the game opens a card-selection surface.  The
    previous guard can only buy the removal service; this second stage makes
    sure the selected card is an obvious junk/starter when such a target is
    legal.  It returns ``None`` when the current surface is not a deck-removal
    selection.
    """

    remove_indices: list[tuple[int, float]] = []
    for idx in range(legal_count):
        if mask_np[idx] <= 0:
            continue
        action = action_source[idx]
        if not _is_shop_or_deck_remove_selection(action):
            continue
        score = _remove_target_score(_selection_card(action))
        if score > 0.0:
            remove_indices.append((int(idx), float(score)))

    if not remove_indices:
        return None

    search_stats["shop_action_guard_remove_selection_surface"] = 1.0
    selected_score = 0.0
    if 0 <= selected_idx < legal_count and _is_shop_or_deck_remove_selection(action_source[selected_idx]):
        selected_score = _remove_target_score(_selection_card(action_source[selected_idx]))
    best_idx, best_score = max(remove_indices, key=lambda pair: (pair[1], -pair[0]))
    search_stats["shop_action_guard_remove_selection_selected_score"] = float(selected_score)
    search_stats["shop_action_guard_remove_selection_best_score"] = float(best_score)
    if best_score > max(0.0, selected_score):
        search_stats["shop_action_guard_remove_selection_applied"] = 1.0
        return int(best_idx)
    return int(selected_idx)


def apply_shop_action_guard(
    *,
    action_idx: int,
    legal_actions: list[Any] | None,
    full_legal_actions: list[Any] | None,
    action_mask: Any,
    raw_obs: dict[str, Any] | None,
    search_stats: dict[str, Any],
) -> int:
    """Return a safer shop action index when the selected one is a clear miss.

    The guard is fail-open: any missing payload, unknown gold, or index mismatch
    returns ``action_idx`` unchanged.  It only overrides high-confidence shop
    mistakes: skip/open misses, leaving despite affordable removal, and
    ordinary card buys that spend the last removal gold in starter-heavy decks.
    """

    for key in shop_action_guard_metric_keys():
        search_stats.setdefault(key, 0.0)
    search_stats["shop_action_guard_enabled"] = 1.0

    if not isinstance(legal_actions, list) or not legal_actions:
        return action_idx
    mask_np = _mask_array(action_mask)
    if mask_np is None:
        return action_idx
    legal_count = _valid_legal_count(legal_actions, mask_np)
    if legal_count <= 0:
        return action_idx
    try:
        selected_idx = int(action_idx)
    except Exception:
        return action_idx
    if not (0 <= selected_idx < legal_count) or mask_np[selected_idx] <= 0:
        return action_idx

    action_source = _action_source_for_guard(
        legal_actions=legal_actions,
        full_legal_actions=full_legal_actions,
        legal_count=legal_count,
    )
    if action_source is None:
        search_stats["shop_action_guard_alignment_error"] = 1.0
        return action_idx

    selection_override = _apply_remove_selection_guard(
        selected_idx=selected_idx,
        action_source=action_source,
        mask_np=mask_np,
        legal_count=legal_count,
        search_stats=search_stats,
    )
    if selection_override is not None:
        return int(selection_override)

    if not any(is_shop_action(action_source[idx]) for idx in range(legal_count) if mask_np[idx] > 0):
        return action_idx
    search_stats["shop_action_guard_shop_surface"] = 1.0

    selected = action_source[selected_idx]
    selected_kind = shop_action_kind(selected)
    search_stats["shop_action_guard_selected_open"] = 1.0 if selected_kind == "open" else 0.0
    search_stats["shop_action_guard_selected_buy"] = 1.0 if selected_kind.startswith("buy_") else 0.0
    search_stats["shop_action_guard_selected_remove"] = 1.0 if selected_kind == "remove" else 0.0
    search_stats["shop_action_guard_selected_leave"] = 1.0 if selected_kind == "leave" else 0.0
    search_stats["shop_action_guard_selected_back"] = 1.0 if selected_kind == "back" else 0.0

    open_indices = _legal_indices_by_kind(
        action_source=action_source,
        mask_np=mask_np,
        legal_count=legal_count,
        wanted="open",
    )
    remove_indices = _affordable_remove_indices(
        action_source=action_source,
        mask_np=mask_np,
        legal_count=legal_count,
    )
    search_stats["shop_action_guard_open_available"] = 1.0 if open_indices else 0.0
    search_stats["shop_action_guard_remove_affordable_available"] = 1.0 if remove_indices else 0.0

    gold, gold_valid = _player_gold(raw_obs)
    if not gold_valid:
        search_stats["shop_action_guard_invalid_obs"] = 1.0
        return selected_idx
    search_stats["shop_action_guard_gold"] = float(gold)

    # Failure mode 1: the closed-shop surface offers "open inventory" and
    # "leave/back"; leaving with usable gold means the model never even
    # inspected the purchasable items.  Some bridge builds label the closed-shop
    # exit as ``back`` rather than ``leave``; treat both as a reversible open
    # miss when an explicit open action is legal.
    if selected_kind in {"leave", "back"} and open_indices and gold >= SHOP_OPEN_GOLD_THRESHOLD:
        search_stats["shop_action_guard_open_applicable"] = 1.0
        override_idx = int(open_indices[0])
        if 0 <= override_idx < legal_count and mask_np[override_idx] > 0:
            search_stats["shop_action_guard_open_applied"] = 1.0
            return override_idx
        search_stats["shop_action_guard_alignment_error"] = 1.0
        return selected_idx

    starter_count, junk_count, deck_size = _deck_counts(raw_obs)
    search_stats["shop_action_guard_starter_count"] = float(starter_count)
    search_stats["shop_action_guard_junk_count"] = float(junk_count)
    starter_ratio = float(starter_count) / float(max(deck_size, 1))
    remove_worth_forcing = (
        starter_count >= SHOP_REMOVE_STARTER_THRESHOLD
        or (starter_count >= 4 and deck_size >= 10 and starter_ratio >= SHOP_REMOVE_STARTER_RATIO_THRESHOLD)
        or junk_count > 0
    )
    remove_cost_min = _cheapest_remove_cost(action_source=action_source, remove_indices=remove_indices)
    if remove_cost_min > 0.0:
        search_stats["shop_action_guard_remove_cost_min"] = float(remove_cost_min)

    # Failure mode 2: the opened-shop surface offers affordable removal, but
    # the model chooses to buy an ordinary card that spends the last removal
    # gold while the deck is still starter/junk heavy.  This is deliberately
    # narrower than "always prefer remove": it does not touch relics/potions,
    # does not touch high-confidence premium cards, and does not fire when the
    # selected purchase still leaves enough gold to remove afterwards.
    if selected_kind == "buy_card" and remove_indices and remove_worth_forcing:
        selected_cost = _item_cost(selected)
        gold_after_selected_buy = float(gold) - float(selected_cost)
        search_stats["shop_action_guard_selected_buy_cost"] = float(selected_cost)
        search_stats["shop_action_guard_gold_after_selected_buy"] = float(gold_after_selected_buy)
        if (
            _item_affordable(selected)
            and selected_cost > 0.0
            and remove_cost_min > 0.0
            and gold_after_selected_buy + 1e-6 < remove_cost_min
        ):
            search_stats["shop_action_guard_buy_blocks_remove_applicable"] = 1.0
            if _is_premium_card_purchase(selected):
                search_stats["shop_action_guard_buy_blocks_remove_premium_allow"] = 1.0
                return selected_idx
            override_idx = int(remove_indices[0])
            if 0 <= override_idx < legal_count and mask_np[override_idx] > 0:
                search_stats["shop_action_guard_buy_blocks_remove_applied"] = 1.0
                search_stats["shop_action_guard_remove_applied"] = 1.0
                return override_idx
            search_stats["shop_action_guard_alignment_error"] = 1.0
            return selected_idx

    # Failure mode 3: the opened-shop surface offers affordable removal, but
    # the model chooses to leave/back while the deck is still starter/junk
    # heavy.
    if selected_kind not in {"leave", "back"} or not remove_indices:
        return selected_idx

    if not remove_worth_forcing:
        return selected_idx

    search_stats["shop_action_guard_remove_applicable"] = 1.0
    override_idx = int(remove_indices[0])
    if 0 <= override_idx < legal_count and mask_np[override_idx] > 0:
        search_stats["shop_action_guard_remove_applied"] = 1.0
        return override_idx
    search_stats["shop_action_guard_alignment_error"] = 1.0
    return selected_idx


__all__ = [
    "SHOP_ACTION_GUARD_SEARCH_SUFFIXES",
    "SHOP_OPEN_GOLD_THRESHOLD",
    "SHOP_REMOVE_STARTER_THRESHOLD",
    "apply_shop_action_guard",
    "shop_action_guard_metric_keys",
]
