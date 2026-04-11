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


def compact_action_signature(action: Any) -> dict[str, Any]:
    """Return a compact, replay-friendly signature for one legal action."""

    if not isinstance(action, dict):
        return {}

    card = action.get("card") if isinstance(action.get("card"), dict) else {}
    potion = action.get("potion") if isinstance(action.get("potion"), dict) else {}
    relic = action.get("relic") if isinstance(action.get("relic"), dict) else {}

    title = (
        card.get("title")
        or potion.get("title")
        or relic.get("title")
        or action.get("title")
        or action.get("label")
        or action.get("name")
    )

    compact = {
        "action_id": action.get("action_id"),
        "kind": action.get("kind"),
        "action_type": action.get("action_type"),
        "surface": action.get("surface"),
        "target_index": action.get("target_index"),
        "title": title,
        "target": card.get("target") or potion.get("target") or action.get("target"),
        "card_id": card.get("id") or action.get("card_id"),
        "card_cost": _coerce_float(card.get("cost") if card else action.get("card_cost")),
        "potion_id": potion.get("id") or action.get("potion_id"),
        "relic_id": relic.get("id") or action.get("relic_id"),
        "choice_index": action.get("index") if action.get("index") is not None else action.get("choice_index"),
        "price": _coerce_float(action.get("price")),
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
