"""No-pressure pure-block detection for combat action quality.

The bridge exposes legal card actions even when an enemy has no incoming damage
intent.  A legal Defend-like card is not automatically a good action: if the
card only creates block and there is no remaining damage pressure, spending
energy on it increases combat length and hp loss indirectly by reducing damage
or setup opportunities.

This module deliberately avoids text-regex card matching.  It consumes typed
semantic roles and effect metrics gathered by the trainer/bridge.  ``train.py``
should only adapt bridge payloads into this pure function.
"""

from __future__ import annotations

from typing import Any, Mapping


BLOCK_ROLES = frozenset(
    {
        "block",
        "defense",
        "defend",
        "guard",
        "shield",
        "barrier",
        "plated_armor",
    }
)

# Roles that mean the action advances the fight or preserves future options.
# These make a block card *not* "pure block".
PROGRESS_ROLES = frozenset(
    {
        "attack",
        "damage",
        "draw",
        "discard",
        "exhaust_other",
        "debuff",
        "weak",
        "vulnerable",
        "poison",
        "buff",
        "scaling",
        "power",
        "setup",
        "resource",
        "energy",
        "heal",
        "card_state",
        "hand_mutation",
        "card_rule_modifier",
        "cost_reduction",
        "copy",
        "transform",
        "upgrade",
        "enchant",
        "enchantment",
        "modifier",
        "keyword",
        "replay",
        "facing_change",
        "stun",
        "artifact_strip",
        "lock",
        "mechanism",
    }
)

# Numeric/boolean metrics that indicate non-block side effects.  The trainer
# passes values from semantic.card_effect_profile, compact semantic payloads,
# and source card fields.  Self-exhaust alone is intentionally not included:
# an exhaust-only block card under no pressure is still a bad spend.
SIDE_EFFECT_METRIC_KEYS = frozenset(
    {
        "draw",
        "cards_drawn",
        "energy",
        "energy_gain",
        "typed_gain_energy",
        "typed_gain_energy_amount",
        "heal",
        "weak",
        "vulnerable",
        "poison",
        "strength",
        "dexterity",
        "artifact",
        "stun",
        "facing_change",
        "typed_debuff",
        "typed_apply_debuff",
        "typed_apply_power",
        "typed_apply_buff",
        "typed_modifies_hand",
        "typed_modify_cost",
        "typed_upgrade_hand",
        "typed_upgrade_cards",
        "typed_discard_cards",
        "typed_transform_cards",
        "typed_copy_cards",
        "typed_add_modifier",
        "typed_add_keyword",
        "typed_set_replay",
        "typed_retain_cards",
        "typed_card_state_mutation",
        "typed_requires_followup",
        "typed_strategic_skip_if_no_followup",
        "typed_no_draw",
        "typed_future_penalty",
        "typed_consumes_future_resource",
    }
)


def _float(value: Any) -> float:
    try:
        return float(value or 0.0)
    except (TypeError, ValueError):
        return 0.0


def _truthy_metric(metrics: Mapping[str, Any], keys: frozenset[str]) -> bool:
    for key in keys:
        if _float(metrics.get(key)) > 0.0:
            return True
    return False


def card_block_waste_profile(
    action: Any,
    *,
    family: str,
    roles: set[str] | frozenset[str],
    card_type: str,
    incoming: float,
    current_block: float,
    damage: float,
    block: float,
    draw: float,
    energy_gain: float,
    heal: float,
    hp_loss: float,
    metrics: Mapping[str, Any] | None = None,
    mechanism_urgent: bool = False,
) -> dict[str, Any]:
    """Return no-pressure pure-block diagnostics for one legal action.

    ``block_waste`` is intentionally narrow:

    * only play-card actions can be flagged;
    * the card must produce block / have a block role;
    * no remaining damage pressure can exist after current block;
    * attacks, powers, debuffs, card-state mutations, draw, energy, healing,
      and boss-mechanism answers are excluded.

    This lets the trainer stop treating no-threat Defend as urgent while still
    pushing attacks/setup/draw when enemies are buffing or stunned.
    """

    metrics = metrics or {}
    family_l = str(family or "").strip().lower()
    card_type_l = str(card_type or "").strip().lower()
    roles_l = {str(role or "").strip().lower() for role in roles if str(role or "").strip()}

    incoming_v = max(_float(incoming), 0.0)
    current_block_v = max(_float(current_block), 0.0)
    threat_gap = max(0.0, incoming_v - current_block_v)

    damage_v = max(_float(damage), _float(metrics.get("damage")), _float(metrics.get("total_damage")))
    block_v = max(_float(block), _float(metrics.get("block")), _float(metrics.get("total_block")))
    draw_v = max(_float(draw), _float(metrics.get("draw")), _float(metrics.get("cards_drawn")))
    energy_gain_v = max(
        _float(energy_gain),
        _float(metrics.get("energy")),
        _float(metrics.get("energy_gain")),
        _float(metrics.get("typed_gain_energy_amount")),
        _float(metrics.get("typed_gain_energy")),
    )
    heal_v = max(_float(heal), _float(metrics.get("heal")))
    hp_loss_v = max(_float(hp_loss), _float(metrics.get("hp_loss")), _float(metrics.get("hp_cost")))

    is_play_card = family_l == "play_card"
    is_block_card = bool(block_v > 0.0 or roles_l.intersection(BLOCK_ROLES))
    no_damage_pressure = bool(threat_gap <= 0.05)

    role_progress = bool(roles_l.intersection(PROGRESS_ROLES))
    metric_progress = bool(
        damage_v > 0.0
        or draw_v > 0.0
        or energy_gain_v > 0.0
        or heal_v > 0.0
        or _truthy_metric(metrics, SIDE_EFFECT_METRIC_KEYS)
    )
    type_progress = card_type_l == "power"
    has_side_effect = bool(role_progress or metric_progress or type_progress or mechanism_urgent)

    pure_block = bool(is_play_card and is_block_card and not has_side_effect)
    block_waste = bool(pure_block and no_damage_pressure)

    return {
        "is_block_card": bool(is_block_card),
        "pure_block": bool(pure_block),
        "no_damage_pressure": bool(no_damage_pressure),
        "block_waste": bool(block_waste),
        "has_side_effect": bool(has_side_effect),
        "incoming": float(incoming_v),
        "current_block": float(current_block_v),
        "threat_gap": float(threat_gap),
        "damage": float(damage_v),
        "block": float(block_v),
        "draw": float(draw_v),
        "energy_gain": float(energy_gain_v),
        "heal": float(heal_v),
        "hp_loss": float(hp_loss_v),
    }
