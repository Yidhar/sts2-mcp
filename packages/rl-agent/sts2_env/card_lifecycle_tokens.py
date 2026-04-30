"""Lifecycle-aware token feature builders (TASK-D2).

The world/action token encoders consume two helpers from this module:

* :func:`build_action_lifecycle_features` — given an action dict (and optional
  live state context such as remaining energy), returns the structured
  "what happens after I play this" feature dict the action token surfaces.
* :func:`build_pile_summary_features` — given the current draw/discard/exhaust
  pile contents, returns aggregate counts (totals + cycle-density bands).

Both helpers read the structured ``derived_view`` block from the card's effect
profile (TASK-D1), so they NEVER walk localized card text.  Observation/action
encoders that consume these helpers can stay localization-free.
"""

from __future__ import annotations

from typing import Any, Iterable

from .card_effect_profile import (
    aggregate_card_effect_profile_semantics,
    card_combat_effect_view,
    card_cost_view,
    card_derived_view,
    card_hand_mutation_view,
    card_lifecycle_view,
    card_mechanism_effect_view,
    card_pile_mutation_view,
)


# ---------------------------------------------------------------------------
# Action lifecycle features
# ---------------------------------------------------------------------------


def _float(value: Any, default: float = 0.0) -> float:
    if value is None:
        return default
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _bool(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    if value is None:
        return False
    if isinstance(value, (int, float)):
        return float(value) != 0.0
    return bool(value)


def _resolve_card_payload(action: dict[str, Any] | None) -> dict[str, Any]:
    if not isinstance(action, dict):
        return {}
    card = action.get("card")
    if isinstance(card, dict):
        return card
    return {}


def _resolve_effective_cost(card: dict[str, Any], cost_view: dict[str, Any], energy: float | None) -> float:
    """Resolve the energy that would be spent if this card is played now."""
    if cost_view.get("is_x_cost"):
        if energy is None:
            return 0.0
        return max(0.0, float(energy))
    raw = cost_view.get("base")
    if isinstance(raw, (int, float)):
        return float(raw)
    fallback = card.get("current_cost") if isinstance(card.get("current_cost"), (int, float)) else card.get("cost")
    if isinstance(fallback, (int, float)):
        return float(fallback)
    return 0.0


def _expected_pile_destination(
    card: dict[str, Any], lifecycle: dict[str, Any], pile: dict[str, Any]
) -> str:
    if lifecycle.get("ethereal") and lifecycle.get("exhausts_on_play"):
        return "exhaust_pile"
    if pile.get("moves_to_exhaust_self"):
        return "exhaust_pile"
    if lifecycle.get("returns_to_hand"):
        return "hand"
    if pile.get("moves_to_discard_self"):
        return "discard_pile"
    card_type = (card.get("type") or "").lower()
    if card_type == "power":
        return "removed"
    if card_type in {"status", "curse"}:
        return "exhaust_pile" if lifecycle.get("ethereal") else "discard_pile"
    return "discard_pile"


def _has_static_followup(legal_actions: Iterable[dict[str, Any]] | None, *, exclude_card_id: str | None) -> bool:
    if not legal_actions:
        return False
    for other in legal_actions:
        if not isinstance(other, dict):
            continue
        kind = other.get("kind")
        if kind in {"end_turn", "proceed"}:
            continue
        other_card = other.get("card") if isinstance(other.get("card"), dict) else None
        if other_card and exclude_card_id and other_card.get("id") == exclude_card_id:
            continue
        return True
    return False


def _expected_followup_count(
    card: dict[str, Any],
    lifecycle: dict[str, Any],
    cost_view: dict[str, Any],
    combat: dict[str, Any],
    *,
    energy: float | None,
    legal_actions: Iterable[dict[str, Any]] | None,
) -> int:
    """Cheap proxy for "how many useful actions remain after this play"."""
    if not legal_actions:
        return 0
    has_followup = _has_static_followup(legal_actions, exclude_card_id=card.get("id"))
    if not has_followup:
        return 0
    energy_gain = float(combat.get("energy_gain") or 0)
    spend = _resolve_effective_cost(card, cost_view, energy)
    if energy is None:
        budget = max(0.0, energy_gain - spend)
    else:
        budget = max(0.0, float(energy) + energy_gain - spend)
    if lifecycle.get("returns_to_hand"):
        return max(1, int(budget))
    return max(0, int(budget))


def build_action_lifecycle_features(
    action: dict[str, Any] | None,
    *,
    energy: float | None = None,
    legal_actions: Iterable[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    """Lifecycle / outcome features for a single action token.

    Returns a JSON-stable dict that can be fed directly into the action token
    encoder.  Callers SHOULD pass the live remaining ``energy`` so X-cost
    cards report the right ``effective_cost``.  Other callers (test fixtures
    or non-combat actions) can omit it.
    """
    card = _resolve_card_payload(action)
    derived = card_derived_view(card)
    cost = card_cost_view(card) if derived else {}
    lifecycle = card_lifecycle_view(card) if derived else {}
    hand_mut = card_hand_mutation_view(card) if derived else {}
    pile = card_pile_mutation_view(card) if derived else {}
    combat = card_combat_effect_view(card) if derived else {}
    mech = card_mechanism_effect_view(card) if derived else {}
    typed = aggregate_card_effect_profile_semantics(card) if card else {}
    select = hand_mut.get("select_cards") or {}
    requires_selection = bool(select.get("enabled"))

    expected_pile = _expected_pile_destination(card, lifecycle, pile) if card else "none"
    energy_gain = float(combat.get("energy_gain") or 0)
    base_cost = float(cost.get("base") or 0) if cost else _float(card.get("cost"))
    effective_cost = _resolve_effective_cost(card, cost, energy) if card else 0.0
    cost_after_modifiers = float(card.get("current_cost") if isinstance(card.get("current_cost"), (int, float)) else base_cost)

    hand_mutation_score = float(
        bool(hand_mut.get("upgrades_hand"))
        + bool(hand_mut.get("transforms_cards"))
        + bool(hand_mut.get("copies_cards"))
        + bool(hand_mut.get("creates_cards"))
        + bool(hand_mut.get("discard_hand"))
    )
    future_cycle_value = float(hand_mut.get("draw") or 0) + (1.0 if pile.get("shuffles_into_draw") else 0.0)
    if lifecycle.get("replay_or_duplicate"):
        future_cycle_value += 1.0
    if lifecycle.get("retain"):
        future_cycle_value += 0.5

    expected_followup_count = _expected_followup_count(
        card,
        lifecycle,
        cost,
        combat,
        energy=energy,
        legal_actions=legal_actions,
    )

    return {
        # ------- card identity and cost -------
        "is_x_cost": bool(cost.get("is_x_cost")),
        "base_cost": base_cost,
        "current_cost": cost_after_modifiers,
        "effective_cost": effective_cost,
        # ------- lifecycle -------
        "is_exhaust": bool(lifecycle.get("exhausts_on_play")),
        "is_ethereal": bool(lifecycle.get("ethereal")),
        "is_retain": bool(lifecycle.get("retain")),
        "is_void_or_status": (card.get("type") or "").lower() in {"status", "curse"},
        "replay_or_duplicate": bool(lifecycle.get("replay_or_duplicate")),
        # ------- hand / pile mutation -------
        "energy_gain": energy_gain,
        "draw_count": int(hand_mut.get("draw") or 0),
        "create_count": 1 if hand_mut.get("creates_cards") else 0,
        "upgrade_hand": bool(hand_mut.get("upgrades_hand")),
        "transform_hand": bool(hand_mut.get("transforms_cards")),
        "copy_card": bool(hand_mut.get("copies_cards")),
        "cost_reduction": bool(cost.get("can_change_cost")),
        "requires_card_selection": requires_selection,
        "selected_cards_min": int(select.get("min") or 0),
        "selected_cards_max": int(select.get("max") or 0),
        "expected_pile_destination": expected_pile,
        "hand_mutation_score": hand_mutation_score,
        "future_cycle_value": future_cycle_value,
        # ------- action-token "what happens after" outcome -------
        "action_exhausts_card": bool(lifecycle.get("exhausts_on_play")) or bool(typed.get("typed_exhaust_cards")),
        "action_changes_hand": bool(typed.get("typed_modifies_hand")),
        "action_draws_cards": int(hand_mut.get("draw") or 0) > 0,
        "action_creates_cards": bool(hand_mut.get("creates_cards")),
        "action_refunds_energy": energy_gain > 0,
        "action_reduces_cost": bool(cost.get("can_change_cost")),
        "action_requires_selection": requires_selection,
        "action_changes_facing": bool(mech.get("can_change_facing")),
        "action_has_boss_mechanism_value": bool(
            mech.get("can_strip_artifact") or mech.get("can_trigger_stun")
        ),
        "action_expected_followup_count": expected_followup_count,
    }


# ---------------------------------------------------------------------------
# Pile summary
# ---------------------------------------------------------------------------


def _important_predicate(card: dict[str, Any]) -> bool:
    """Heuristic — the card is "important" if it has a structured profile that
    surfaces a non-trivial mechanical signal (damage / block / draw / energy /
    hand mutation).
    """
    derived = card_derived_view(card)
    if not derived:
        return False
    combat = derived.get("combat_effect") or {}
    hand_mut = derived.get("hand_mutation") or {}
    if (combat.get("damage") or 0) >= 6:
        return True
    if (combat.get("block") or 0) >= 6:
        return True
    if (combat.get("energy_gain") or 0) >= 1:
        return True
    if (hand_mut.get("draw") or 0) >= 1:
        return True
    if hand_mut.get("upgrades_hand") or hand_mut.get("transforms_cards") or hand_mut.get("copies_cards"):
        return True
    return False


def _cycle_role(card: dict[str, Any]) -> set[str]:
    derived = card_derived_view(card)
    combat = (derived.get("combat_effect") or {}) if derived else {}
    hand_mut = (derived.get("hand_mutation") or {}) if derived else {}
    roles: set[str] = set()
    if (combat.get("damage") or 0) > 0:
        roles.add("attack")
    if (combat.get("block") or 0) > 0:
        roles.add("block")
    if (hand_mut.get("draw") or 0) > 0:
        roles.add("draw")
    if (combat.get("energy_gain") or 0) > 0:
        roles.add("energy")
    return roles


def _exhausted_key_card_count(exhaust_pile: list[dict[str, Any]]) -> int:
    return sum(1 for card in exhaust_pile if _important_predicate(card))


def build_pile_summary_features(piles: dict[str, Any] | None) -> dict[str, Any]:
    """Aggregate pile counts + cycle-density features from a piles dict.

    ``piles`` is expected to look like::

        {
            "draw_pile":     [card_dict, ...],
            "discard_pile":  [card_dict, ...],
            "exhaust_pile":  [card_dict, ...],
        }

    Missing keys default to empty lists.  The output is JSON-stable.
    """
    piles = piles or {}
    draw = list(piles.get("draw_pile") or [])
    discard = list(piles.get("discard_pile") or [])
    exhaust = list(piles.get("exhaust_pile") or [])

    important_draw = sum(1 for c in draw if _important_predicate(c))
    important_discard = sum(1 for c in discard if _important_predicate(c))
    important_exhaust = sum(1 for c in exhaust if _important_predicate(c))

    total_cycleable = max(1, len(draw) + len(discard))
    role_counters: dict[str, int] = {"attack": 0, "block": 0, "draw": 0, "energy": 0}
    for card in draw + discard:
        for role in _cycle_role(card):
            role_counters[role] = role_counters.get(role, 0) + 1

    return {
        "draw_pile_count": len(draw),
        "discard_pile_count": len(discard),
        "exhaust_pile_count": len(exhaust),
        "important_draw_pile_count": important_draw,
        "important_discard_pile_count": important_discard,
        "important_exhaust_pile_count": important_exhaust,
        "cycle_density_attack": role_counters["attack"] / total_cycleable,
        "cycle_density_block": role_counters["block"] / total_cycleable,
        "cycle_density_draw": role_counters["draw"] / total_cycleable,
        "cycle_density_energy": role_counters["energy"] / total_cycleable,
        "exhausted_key_card_count": _exhausted_key_card_count(exhaust),
    }
