"""Compact action signatures for RL info payloads and replay traces."""

from __future__ import annotations

from typing import Any

from .semantic_action import compact_semantic_signature, semantic_action_signature


def _coerce_float(value: Any) -> float | int | None:
    if value is None:
        return None
    try:
        numeric = float(value)
    except (TypeError, ValueError):
        return None
    if numeric.is_integer():
        return int(numeric)
    return numeric


def _event_effect_deltas(action: dict[str, Any]) -> dict[str, Any]:
    """Return event option effect_deltas from either live or nested payload."""
    containers = [action]
    payload = action.get("payload")
    if isinstance(payload, dict):
        containers.append(payload)
    for container in containers:
        deltas = container.get("effect_deltas")
        if isinstance(deltas, dict):
            return deltas
        option = container.get("option") if isinstance(container.get("option"), dict) else {}
        deltas = option.get("effect_deltas")
        if isinstance(deltas, dict):
            return deltas
    return {}


def compact_action_signature(action: Any) -> dict[str, Any]:
    """Return a compact, replay-friendly signature for one legal action."""

    if not isinstance(action, dict):
        return {}

    reward = action.get("reward") if isinstance(action.get("reward"), dict) else {}
    reward_card = reward.get("card") if isinstance(reward.get("card"), dict) else {}
    card = action.get("card") if isinstance(action.get("card"), dict) else {}
    potion = action.get("potion") if isinstance(action.get("potion"), dict) else {}
    relic = action.get("relic") if isinstance(action.get("relic"), dict) else {}
    option = action.get("option") if isinstance(action.get("option"), dict) else {}
    item = action.get("item") if isinstance(action.get("item"), dict) else {}
    item_card = item.get("card") if isinstance(item.get("card"), dict) else {}
    item_relic = item.get("relic") if isinstance(item.get("relic"), dict) else {}
    item_potion = item.get("potion") if isinstance(item.get("potion"), dict) else {}
    event_deltas = _event_effect_deltas(action)

    title = (
        card.get("title")
        or reward_card.get("title")
        or potion.get("title")
        or relic.get("title")
        or option.get("title")
        or item_card.get("title")
        or item_relic.get("title")
        or item_potion.get("title")
        or item.get("title")
        or action.get("title")
        or action.get("label")
        or action.get("name")
    )

    compact = {
        "action_id": action.get("action_id"),
        "kind": action.get("kind"),
        "action_type": action.get("action_type"),
        "surface": action.get("surface"),
        "reward_type": action.get("reward_type") or reward.get("type"),
        "target_index": action.get("target_index"),
        "title": title,
        # Keep the concrete action target (creature dict) separate from card/potion
        # target_type (SingleEnemy/Self/AllEnemies). Mixing the two under "target"
        # caused targeted-action diagnostics to lose combat_id when a compact card
        # also carried target metadata.
        "target": action.get("target") or card.get("target") or potion.get("target"),
        "target_type": card.get("target_type") or potion.get("target_type") or card.get("target") or potion.get("target"),
        "card_id": card.get("id") or reward_card.get("id") or action.get("card_id"),
        "card_title": card.get("title") or reward_card.get("title") or action.get("card_title"),
        "card_cost": _coerce_float(
            card.get("cost")
            if card
            else reward_card.get("cost")
            if reward_card
            else action.get("card_cost")
        ),
        # card.type is Attack / Skill / Power / Status / Curse — keep it so the
        # Python positive-action judgement can distinguish playable output cards
        # (Attack/Skill/Power) from unplayable forced draws (Status/Curse) even
        # when the bridge-provided semantic.roles list is empty.
        "card_type": card.get("type") or reward_card.get("type") or action.get("card_type"),
        "potion_id": potion.get("id") or action.get("potion_id"),
        "relic_id": relic.get("id") or action.get("relic_id"),
        "choice_index": action.get("index") if action.get("index") is not None else action.get("choice_index"),
        # Campfire/rest-site diagnostics need the bridge option identity even
        # when the full raw action is not carried through info/replay.  Live
        # RestSite actions are often generic ``rest_site:0`` / ``rest_site:1``
        # ids whose HEAL/SMITH meaning only appears in the nested option.
        "option_id": option.get("option_id") or option.get("id") or action.get("option_id"),
        "option_type": option.get("option_type") or option.get("type") or action.get("option_type"),
        "option_description": option.get("description") or action.get("description"),
        "price": _coerce_float(action.get("price")),
        # Shop actions are encoded by the bridge as:
        #   shop_action=open/back/leave
        #   shop_action=buy + item.item_kind=card/relic/potion/card_removal
        # Keep the nested item identity in replay traces so card removal is not
        # collapsed into a generic "buy" action.
        "shop_action": action.get("shop_action"),
        "shop_item_kind": item.get("item_kind") or item.get("kind"),
        "shop_item_title": item.get("title"),
        "shop_item_cost": _coerce_float(item.get("cost")),
        "shop_item_affordable": (
            item.get("is_affordable")
            if item.get("is_affordable") is not None
            else item.get("affordable")
            if item.get("affordable") is not None
            else item.get("enough_gold")
        ),
        "shop_item_used": item.get("used"),
        "shop_item_card_id": item_card.get("id"),
        "shop_item_relic_id": item_relic.get("id"),
        "shop_item_potion_id": item_potion.get("id"),
        # Combat card-selection surfaces (Purity/净化, retain/discard/transform
        # choices, etc.) need the toggle state on the action token itself.  Without
        # these fields replay diagnostics and observation encoders cannot
        # distinguish "select a fresh card" from "click an already-selected card
        # and deselect it", which creates pick/deselect loops.
        "selection": action.get("selection") or action.get("selection_action"),
        "selection_action": action.get("selection_action") or action.get("selection"),
        "selection_semantics": action.get("selection_semantics"),
        "selection_prompt": action.get("selection_prompt"),
        "is_selected": action.get("is_selected"),
        "selected_count": _coerce_float(action.get("selected_count")),
        "min_select": _coerce_float(action.get("min_select")),
        "max_select": _coerce_float(action.get("max_select")),
        "remaining_select": _coerce_float(action.get("remaining_select")),
        "confirm_ready": action.get("confirm_ready"),
        "can_skip": action.get("can_skip"),
        "requires_manual_confirmation": action.get("requires_manual_confirmation"),
        "cancelable": action.get("cancelable"),
        "selection_ready": action.get("selection_ready"),
        "opened_age_ms": _coerce_float(action.get("opened_age_ms")),
        # Event-option diagnostics.  These are compact, structured versions of
        # effect_deltas so episode tails can explain dangerous event choices
        # such as optional combat ("我能打两个") without dumping the full action.
        "event_enter_combat": bool(event_deltas.get("enter_combat")) if event_deltas else None,
        "event_hp_delta": _coerce_float(event_deltas.get("hp_delta")) if event_deltas else None,
        "event_max_hp_delta": _coerce_float(event_deltas.get("max_hp_delta")) if event_deltas else None,
        "event_gold_delta": _coerce_float(event_deltas.get("gold_delta")) if event_deltas else None,
        "event_card_add_count": _coerce_float(event_deltas.get("card_add_count")) if event_deltas else None,
        "event_card_remove_count": _coerce_float(event_deltas.get("card_remove_count")) if event_deltas else None,
        "event_card_transform_count": _coerce_float(event_deltas.get("card_transform_count")) if event_deltas else None,
        "event_card_upgrade_count": _coerce_float(event_deltas.get("card_upgrade_count")) if event_deltas else None,
        "event_relic_gain": bool(event_deltas.get("relic_gain")) if event_deltas else None,
        "event_potion_gain": bool(event_deltas.get("potion_gain")) if event_deltas else None,
    }
    result = {key: value for key, value in compact.items() if value is not None}
    semantic = compact_semantic_signature(semantic_action_signature(action))
    if semantic:
        result["semantic"] = semantic
    return result


def compact_legal_actions(actions: list[Any] | None) -> list[dict[str, Any]]:
    """Compact an action list while preserving positional alignment."""

    if not isinstance(actions, list):
        return []
    return [compact_action_signature(action) for action in actions]
