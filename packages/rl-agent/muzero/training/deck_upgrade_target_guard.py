"""Conservative campfire deck-upgrade target guard.

Full-run monitoring showed the important chain now works:

    high HP campfire -> SMITH -> deck_upgrade surface -> upgrade applied

but one observed Act1 death upgraded ``打击`` while much better starter/core
targets were available.  This module keeps the fix deliberately narrow: only
override obviously bad starter-target upgrades, primarily Strike, when the same
upgrade surface contains a clearly better target such as Bash/痛击 or a
high-priority non-starter card.  It does not attempt to solve all deck-building
credit assignment; it only prevents wasting scarce Act1 smiths on low-value
starter upgrades.
"""

from __future__ import annotations

import math
import re
from typing import Any

import numpy as np

from muzero.diagnostics.deck_upgrade_metrics import (
    is_deck_upgrade_action,
    is_deck_upgrade_terminal_action,
)
from sts2_env.observation_v2 import MAX_ACTIONS


DECK_UPGRADE_TARGET_GUARD_SEARCH_SUFFIXES: dict[str, str] = {
    "deck_upgrade_target_guard_enabled": "deck_upgrade_target_guard_enabled",
    "deck_upgrade_target_guard_context": "deck_upgrade_target_guard_context_rate",
    "deck_upgrade_target_guard_selected_starter": "deck_upgrade_target_guard_selected_starter_rate",
    "deck_upgrade_target_guard_selected_strike": "deck_upgrade_target_guard_selected_strike_rate",
    "deck_upgrade_target_guard_selected_defend": "deck_upgrade_target_guard_selected_defend_rate",
    "deck_upgrade_target_guard_selected_close": "deck_upgrade_target_guard_selected_close_rate",
    "deck_upgrade_target_guard_close_with_upgrade_available": "deck_upgrade_target_guard_close_with_upgrade_available_rate",
    "deck_upgrade_target_guard_close_override": "deck_upgrade_target_guard_close_override_rate",
    "deck_upgrade_target_guard_better_available": "deck_upgrade_target_guard_better_available_rate",
    "deck_upgrade_target_guard_bash_available": "deck_upgrade_target_guard_bash_available_rate",
    "deck_upgrade_target_guard_applied": "deck_upgrade_target_guard_applied_rate",
    "deck_upgrade_target_guard_override": "deck_upgrade_target_guard_override_rate",
    "deck_upgrade_target_guard_alignment_error": "deck_upgrade_target_guard_alignment_error_rate",
    "deck_upgrade_target_guard_selected_score": "deck_upgrade_target_guard_selected_score_mean",
    "deck_upgrade_target_guard_best_score": "deck_upgrade_target_guard_best_score_mean",
    "deck_upgrade_target_guard_best_minus_selected": "deck_upgrade_target_guard_best_minus_selected_mean",
}


def deck_upgrade_target_guard_metric_keys() -> tuple[str, ...]:
    return tuple(DECK_UPGRADE_TARGET_GUARD_SEARCH_SUFFIXES)


def _set_default_metrics(search_stats: dict[str, Any]) -> None:
    for key in deck_upgrade_target_guard_metric_keys():
        search_stats.setdefault(key, 0.0)
    search_stats["deck_upgrade_target_guard_enabled"] = 1.0


def _safe_float(value: Any, default: float = 0.0) -> float:
    try:
        out = float(value)
    except (TypeError, ValueError):
        return float(default)
    return out if math.isfinite(out) else float(default)


def _action_containers(action: Any) -> list[dict[str, Any]]:
    if not isinstance(action, dict):
        return []
    containers = [action]
    payload = action.get("payload")
    if isinstance(payload, dict):
        containers.append(payload)
    return containers


def _card_payload(action: Any) -> dict[str, Any]:
    for container in _action_containers(action):
        card = container.get("card")
        if isinstance(card, dict):
            return card
        item = container.get("item")
        if isinstance(item, dict):
            card = item.get("card")
            if isinstance(card, dict):
                return card
    return {}


def _norm(value: Any) -> str:
    return str(value or "").strip().lower()


def _compact_title(value: Any) -> str:
    text = str(value or "").strip().lower()
    # Normalize common upgrade marks without stripping meaningful internal
    # characters from non-starter Chinese names such as "双重打击".
    text = re.sub(r"[\s\+\＋]+$", "", text)
    text = re.sub(r"\s*\+?\s*upgraded\s*$", "", text)
    return text.strip()


def card_id(action: Any) -> str:
    card = _card_payload(action)
    for container in _action_containers(action):
        value = (
            card.get("id")
            or container.get("card_id")
            or container.get("card_uuid")
            or container.get("id")
        )
        if value:
            return str(value)
    return str(card.get("id") or "")


def card_title(action: Any) -> str:
    card = _card_payload(action)
    for container in _action_containers(action):
        value = (
            card.get("title")
            or card.get("name")
            or container.get("card_title")
            or container.get("title")
            or container.get("label")
            or container.get("name")
        )
        if value:
            return str(value)
    return str(card.get("title") or card.get("name") or "")


def _action_text(action: Any) -> str:
    parts: list[str] = [card_id(action), card_title(action)]
    card = _card_payload(action)
    for key in ("type", "rarity", "description", "text", "keywords"):
        value = card.get(key)
        if value is not None:
            parts.append(str(value))
    for container in _action_containers(action):
        for key in (
            "action_id",
            "kind",
            "action_type",
            "card_id",
            "card_title",
            "title",
            "label",
            "name",
            "description",
            "canonical_text",
        ):
            value = container.get(key)
            if value is not None:
                parts.append(str(value))
    return " ".join(parts).lower()


def is_starter_strike(action: Any) -> bool:
    cid = _norm(card_id(action))
    title = _compact_title(card_title(action))
    if cid.endswith("strike_ironclad") or cid in {"card.strike_ironclad", "strike_ironclad"}:
        return True
    return title in {"strike", "打击"}


def is_starter_defend(action: Any) -> bool:
    cid = _norm(card_id(action))
    title = _compact_title(card_title(action))
    if cid.endswith("defend_ironclad") or cid in {"card.defend_ironclad", "defend_ironclad"}:
        return True
    return title in {"defend", "防御"}


def is_starter_low_value(action: Any) -> bool:
    return is_starter_strike(action) or is_starter_defend(action)


def is_deck_upgrade_close_action(action: Any) -> bool:
    """Return true for terminal close/done/cancel on the upgrade-target surface."""

    return is_deck_upgrade_terminal_action(action)


def is_bash(action: Any) -> bool:
    cid = _norm(card_id(action))
    title = _compact_title(card_title(action))
    if cid.endswith("bash_ironclad") or cid in {"card.bash_ironclad", "bash_ironclad"}:
        return True
    return title in {"bash", "痛击"}


KNOWN_PRIORITY_BY_TITLE: dict[str, float] = {
    # Core starter / strong Act1 upgrades.
    "bash": 120.0,
    "痛击": 120.0,
    # Strong powers / engines seen in the current data.
    "demon form": 90.0,
    "恶魔形态": 90.0,
    "shrug it off": 84.0,
    "耸肩无视": 84.0,
    "colossus": 80.0,
    "巨像": 80.0,
    # Good but less universally critical Act1 upgrades.
    "headbutt": 72.0,
    "头槌": 72.0,
    "double strike": 68.0,
    "双重打击": 68.0,
    "body slam": 66.0,
    "全身撞击": 66.0,
    "ember": 62.0,
    "余烬": 62.0,
}


def upgrade_target_score(action: Any) -> float:
    """Small, schema-tolerant priority score for upgrade-target ordering.

    The absolute value is only used by the conservative guard below.  Known
    cards get stable priorities; unknown non-starters are valued above Strike
    but usually below the override threshold, so the guard will not blindly
    rewrite all starter upgrades to unknown cards.
    """

    if not is_deck_upgrade_action(action):
        return -1.0e9
    title = _compact_title(card_title(action))
    if title in KNOWN_PRIORITY_BY_TITLE:
        return float(KNOWN_PRIORITY_BY_TITLE[title])
    if is_starter_strike(action):
        return 5.0
    if is_starter_defend(action):
        return 18.0

    text = _action_text(action)
    score = 50.0
    if any(token in text for token in ("power", "能力", "strength", "力量", "dex", "敏捷")):
        score += 18.0
    if any(token in text for token in ("draw", "抽", "card", "牌")):
        score += 12.0
    if any(token in text for token in ("block", "格挡", "防御")):
        score += 10.0
    if any(token in text for token in ("damage", "伤害", "attack", "攻击")):
        score += 8.0

    card = _card_payload(action)
    rarity = _norm(card.get("rarity"))
    if rarity in {"rare", "稀有"}:
        score += 6.0
    elif rarity in {"uncommon", "罕见"}:
        score += 3.0
    cost = card.get("cost")
    if cost is None:
        for container in _action_containers(action):
            cost = container.get("card_cost")
            if cost is not None:
                break
    if _safe_float(cost, -1.0) >= 2.0:
        score += 3.0
    return float(score)


def _legal_count(legal_actions: list[Any], action_mask: Any) -> tuple[int, np.ndarray] | None:
    try:
        mask_np = np.asarray(action_mask, dtype=np.float32).reshape(-1)
    except Exception:
        return None
    legal_count = min(len(legal_actions), int(mask_np.shape[0]) if mask_np.size else 0, MAX_ACTIONS)
    return legal_count, mask_np


def _alignment_error(
    compact_actions: list[Any] | None,
    full_actions: list[Any] | None,
    *,
    max_index: int,
) -> bool:
    if not isinstance(compact_actions, list) or not isinstance(full_actions, list):
        return False
    if max_index < 0:
        return False
    if len(compact_actions) <= max_index or len(full_actions) <= max_index:
        return True
    for idx in range(max_index + 1):
        compact = compact_actions[idx]
        full = full_actions[idx]
        if not isinstance(compact, dict) or not isinstance(full, dict):
            continue
        compared = False
        for key in ("action_id", "kind", "target_index", "choice_index", "index"):
            cv = compact.get(key)
            fv = full.get(key)
            if cv is None or fv is None:
                continue
            compared = True
            if str(cv) != str(fv):
                return True
        compact_cid = card_id(compact)
        full_cid = card_id(full)
        if compact_cid and full_cid:
            compared = True
            if _norm(compact_cid) != _norm(full_cid):
                return True
        compact_title = _compact_title(card_title(compact))
        full_title = _compact_title(card_title(full))
        if compact_title and full_title:
            compared = True
            if compact_title != full_title:
                return True
        _ = compared
    return False


def apply_deck_upgrade_target_guard(
    *,
    action_idx: int,
    legal_actions: list[Any] | None,
    full_legal_actions: list[Any] | None,
    action_mask: Any,
    raw_obs: dict[str, Any] | None,
    search_stats: dict[str, Any],
    min_score_gap: float = 35.0,
) -> int:
    """Override obvious low-value starter upgrade targets.

    ``raw_obs`` is accepted for API symmetry with other build guards; this
    conservative first pass does not need to inspect the deck because it only
    blocks clear Strike/Defend mistakes on an already-visible deck-upgrade
    surface.
    """

    _ = raw_obs
    _set_default_metrics(search_stats)
    if not isinstance(legal_actions, list) or len(legal_actions) == 0:
        return int(action_idx)
    count_and_mask = _legal_count(legal_actions, action_mask)
    if count_and_mask is None:
        return int(action_idx)
    legal_count, mask_np = count_and_mask
    action_idx = int(action_idx)
    if not (0 <= action_idx < legal_count) or mask_np[action_idx] <= 0:
        return action_idx

    action_source = full_legal_actions if isinstance(full_legal_actions, list) else legal_actions
    if not isinstance(action_source, list) or len(action_source) < legal_count:
        search_stats["deck_upgrade_target_guard_alignment_error"] = 1.0
        return action_idx
    if isinstance(full_legal_actions, list) and _alignment_error(legal_actions, full_legal_actions, max_index=legal_count - 1):
        search_stats["deck_upgrade_target_guard_alignment_error"] = 1.0
        return action_idx

    upgrade_indices: list[int] = [
        int(idx)
        for idx in range(legal_count)
        if mask_np[idx] > 0 and is_deck_upgrade_action(action_source[idx])
    ]
    if not upgrade_indices:
        return action_idx
    search_stats["deck_upgrade_target_guard_context"] = 1.0

    selected_action = action_source[action_idx]
    selected_is_close = is_deck_upgrade_close_action(selected_action)
    if selected_is_close:
        search_stats["deck_upgrade_target_guard_selected_close"] = 1.0
        search_stats["deck_upgrade_target_guard_close_with_upgrade_available"] = 1.0
    elif not is_deck_upgrade_action(selected_action):
        return action_idx

    selected_score = 0.0 if selected_is_close else upgrade_target_score(selected_action)
    scored = [(idx, upgrade_target_score(action_source[idx])) for idx in upgrade_indices]
    best_idx, best_score = max(scored, key=lambda item: item[1])
    search_stats["deck_upgrade_target_guard_selected_score"] = float(selected_score)
    search_stats["deck_upgrade_target_guard_best_score"] = float(best_score)
    search_stats["deck_upgrade_target_guard_best_minus_selected"] = float(best_score - selected_score)
    if any(is_bash(action_source[idx]) for idx in upgrade_indices):
        search_stats["deck_upgrade_target_guard_bash_available"] = 1.0
    if is_starter_low_value(selected_action):
        search_stats["deck_upgrade_target_guard_selected_starter"] = 1.0
    if is_starter_strike(selected_action):
        search_stats["deck_upgrade_target_guard_selected_strike"] = 1.0
    if is_starter_defend(selected_action):
        search_stats["deck_upgrade_target_guard_selected_defend"] = 1.0

    # Close/done on a deck-upgrade target surface is never useful when concrete
    # upgrade targets are legal.  This exact failure mode left Act1 death decks
    # with upgraded_count=0 despite the policy choosing SMITH at campfires.
    if selected_is_close:
        if not (0 <= int(best_idx) < legal_count) or mask_np[int(best_idx)] <= 0:
            search_stats["deck_upgrade_target_guard_alignment_error"] = 1.0
            return action_idx
        search_stats["deck_upgrade_target_guard_better_available"] = 1.0
        search_stats["deck_upgrade_target_guard_applied"] = 1.0
        search_stats["deck_upgrade_target_guard_override"] = 1.0
        search_stats["deck_upgrade_target_guard_close_override"] = 1.0
        return int(best_idx)

    if best_idx == action_idx:
        return action_idx

    # Conservative override rule:
    # * Strike can be replaced by Bash or a clearly premium non-starter target.
    # * Defend is only replaced by Bash.  Defend upgrades can be defensible in
    #   block-starved decks, so do not broadly rewrite them.
    selected_is_strike = is_starter_strike(selected_action)
    selected_is_defend = is_starter_defend(selected_action)
    best_action = action_source[best_idx]
    best_is_bash = is_bash(best_action)
    if not selected_is_strike and not (selected_is_defend and best_is_bash):
        return action_idx
    if float(best_score - selected_score) < float(min_score_gap):
        return action_idx
    if not best_is_bash and float(best_score) < 75.0:
        return action_idx
    if not (0 <= int(best_idx) < legal_count) or mask_np[int(best_idx)] <= 0:
        search_stats["deck_upgrade_target_guard_alignment_error"] = 1.0
        return action_idx

    search_stats["deck_upgrade_target_guard_better_available"] = 1.0
    search_stats["deck_upgrade_target_guard_applied"] = 1.0
    if int(best_idx) != action_idx:
        search_stats["deck_upgrade_target_guard_override"] = 1.0
    return int(best_idx)
