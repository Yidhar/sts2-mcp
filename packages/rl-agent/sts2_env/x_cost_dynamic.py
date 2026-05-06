"""X-cost / Star-X dynamic preview helper (P0-4).

The bridge will eventually attach a unified ``action["x_cost"]`` block::

    {
        "x_cost": {
            "has_x_cost": true,
            "resource": "energy | stars | other",
            "current_value": 0,
            "is_zero": true,
            "effect_scaled": true,
            "preview_scale_source": "energy_x | star_x | none",
            "semantics": "repeat | damage | block | unknown"
        }
    }

This module is the canonical Python-side reader.  Until the bridge ships
the typed block, we fall back to:

* ``card.cost == "X"`` / ``card.x_cost`` / ``card.costs_x``
* ``card.has_star_cost_x`` / ``card.star_cost`` / ``card.preview_star_cost``
* ``card.canonical_energy_cost == "X"``
* ``card.card_effect_profile.derived_view.cost.is_x_cost`` (TASK-D1)

All callers MUST distinguish three cases:

1. **Not X-cost** — ignore.
2. **X-cost with `current_value == 0` AND no non-X effect** — this is the
   classic "wasted 0-energy X" play; trainer should treat as deferable /
   negative.
3. **X-cost with `current_value == 0` AND non-X effect present** — the
   card still does something useful (hand mutation, set_replay, modifier);
   trainer MUST NOT treat as bad.
"""

from __future__ import annotations

from typing import Any

from .card_effect_profile import (
    aggregate_card_effect_profile_semantics,
    card_cost_view,
)


def _safe_int(value: Any, default: int = 0) -> int:
    if value is None:
        return default
    try:
        return int(value)
    except (TypeError, ValueError):
        try:
            return int(float(value))
        except (TypeError, ValueError):
            return default


def _safe_text(value: Any) -> str:
    return str(value or "").strip().lower()


def _action_card(action: Any) -> dict[str, Any]:
    if not isinstance(action, dict):
        return {}
    card = action.get("card")
    return card if isinstance(card, dict) else {}


def _typed_block(action: Any) -> dict[str, Any]:
    if not isinstance(action, dict):
        return {}
    block = action.get("x_cost")
    return block if isinstance(block, dict) else {}


def _player_energy(raw_obs: Any) -> int:
    if not isinstance(raw_obs, dict):
        return 0
    combat = raw_obs.get("combat")
    if isinstance(combat, dict) and combat.get("energy") is not None:
        return _safe_int(combat.get("energy"))
    player = raw_obs.get("player")
    if isinstance(player, dict) and player.get("energy") is not None:
        return _safe_int(player.get("energy"))
    return 0


def _player_stars(raw_obs: Any) -> int:
    if not isinstance(raw_obs, dict):
        return 0
    combat = raw_obs.get("combat")
    if isinstance(combat, dict):
        for key in ("stars", "star_count", "current_stars"):
            if combat.get(key) is not None:
                return _safe_int(combat.get(key))
    player = raw_obs.get("player")
    if isinstance(player, dict):
        for key in ("stars", "star_count", "current_stars"):
            if player.get(key) is not None:
                return _safe_int(player.get(key))
    return 0


def _has_x_cost(card: dict[str, Any]) -> bool:
    if not isinstance(card, dict):
        return False
    if bool(card.get("x_cost") or card.get("costs_x") or card.get("is_x_cost")):
        return True
    if _safe_text(card.get("cost")) == "x":
        return True
    if _safe_text(card.get("canonical_energy_cost")) == "x":
        return True
    derived = card_cost_view(card)
    if derived and bool(derived.get("is_x_cost")):
        return True
    return False


def _has_star_x(card: dict[str, Any]) -> bool:
    if not isinstance(card, dict):
        return False
    if card.get("has_star_cost_x") or card.get("star_cost_x"):
        return True
    if _safe_text(card.get("star_cost")) == "x":
        return True
    return False


def _has_non_x_effect(card: dict[str, Any]) -> bool:
    """True when the card has a non-X structural effect that fires
    regardless of the spent X — hand mutation, replay, modifier, etc.
    """
    if not isinstance(card, dict):
        return False
    sem = aggregate_card_effect_profile_semantics(card)
    if not sem:
        return False
    return any(
        sem.get(key, 0.0) > 0.0
        for key in (
            "typed_modifies_hand",
            "typed_upgrade_hand",
            "typed_exhaust_cards",
            "typed_discard_cards",
            "typed_transform_cards",
            "typed_copy_cards",
            "typed_add_modifier",
            "typed_add_keyword",
            "typed_set_replay",
            "typed_retain_cards",
            "typed_card_state_mutation",
        )
    )


def x_cost_view(action: Any, raw_obs: Any | None = None) -> dict[str, Any]:
    """Return a typed X-cost / Star-X dict for ``action``.

    Stable shape:

        {
            "has_x_cost": bool,
            "resource": "energy" | "stars" | "other" | "none",
            "current_value": int,
            "is_zero": bool,
            "effect_scaled": bool,
            "preview_scale_source": "energy_x" | "star_x" | "none",
            "semantics": "repeat" | "damage" | "block" | "unknown",
            "non_x_effect_present": bool,
            "zero_x_bad": bool,         # True iff is_zero AND no non-X effect
            "source_confidence": "runtime_internal" | "fallback" | "none",
        }

    ``zero_x_bad`` is the gate the trainer should use to decide whether a
    0-X play deserves a negative classification.  When ``non_x_effect_present``
    is True the card still does something (e.g. add modifier, exhaust hand
    cards, set replay), so a 0-X play is NOT automatically bad.
    """
    out: dict[str, Any] = {
        "has_x_cost": False,
        "resource": "none",
        "current_value": 0,
        "is_zero": False,
        "effect_scaled": False,
        "preview_scale_source": "none",
        "semantics": "unknown",
        "non_x_effect_present": False,
        "zero_x_bad": False,
        "source_confidence": "none",
    }

    typed = _typed_block(action)
    if typed:
        out["has_x_cost"] = bool(typed.get("has_x_cost"))
        resource = _safe_text(typed.get("resource")) or "none"
        out["resource"] = resource if resource in {"energy", "stars", "other", "none"} else "other"
        if typed.get("current_value") is not None:
            out["current_value"] = _safe_int(typed.get("current_value"))
        if typed.get("is_zero") is not None:
            out["is_zero"] = bool(typed.get("is_zero"))
        else:
            out["is_zero"] = out["current_value"] == 0
        out["effect_scaled"] = bool(typed.get("effect_scaled"))
        scale_source = _safe_text(typed.get("preview_scale_source"))
        if scale_source in {"energy_x", "star_x", "none"}:
            out["preview_scale_source"] = scale_source
        semantics = _safe_text(typed.get("semantics"))
        if semantics in {"repeat", "damage", "block", "unknown"}:
            out["semantics"] = semantics
        out["source_confidence"] = "runtime_internal"

    card = _action_card(action)
    if not out["has_x_cost"]:
        if _has_star_x(card):
            out["has_x_cost"] = True
            out["resource"] = "stars"
            out["preview_scale_source"] = "star_x"
            out["current_value"] = _player_stars(raw_obs)
            out["source_confidence"] = "fallback"
        elif _has_x_cost(card):
            out["has_x_cost"] = True
            out["resource"] = "energy"
            out["preview_scale_source"] = "energy_x"
            out["current_value"] = _player_energy(raw_obs)
            out["source_confidence"] = "fallback"

    if out["has_x_cost"]:
        out["is_zero"] = out["current_value"] == 0
        out["non_x_effect_present"] = _has_non_x_effect(card)
        out["zero_x_bad"] = bool(out["is_zero"] and not out["non_x_effect_present"])
    return out


def is_zero_x_bad(action: Any, raw_obs: Any | None = None) -> bool:
    """True iff the action is an X-cost play with zero current X-resource
    AND no non-X effect.  Trainer guard for the wasted 0-X classification."""
    return bool(x_cost_view(action, raw_obs).get("zero_x_bad"))


def is_x_cost_action(action: Any, raw_obs: Any | None = None) -> bool:
    return bool(x_cost_view(action, raw_obs).get("has_x_cost"))


def x_cost_resource(action: Any, raw_obs: Any | None = None) -> str:
    return x_cost_view(action, raw_obs).get("resource", "none") or "none"
