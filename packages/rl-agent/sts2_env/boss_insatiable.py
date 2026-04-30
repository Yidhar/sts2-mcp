"""The Insatiable boss mechanics + offender taxonomy (TASK-E3).

Insatiable wins via growth/scaling: every turn the boss eats and gets bigger.
Underperformance against this boss usually splits into four offender shapes:

* **strategic_skip** — the model skipped a card with no real future-reason
  signal (mirrors TASK-B2 narrowing inside the boss namespace).
* **refund_no_followup** — refund-style cards played without a meaningful
  follow-up (mirrors TASK-B3).
* **bad_end_turn** — turn ended while incoming pressure exceeds standing
  block (TASK-B1's bad_end_turn shape, attributed inside this encounter).
* **missed_pressure_window** — a high-pressure / urgent window opened
  (e.g. boss is about to eat, scaling is about to spike, growth counter
  ticked to a critical threshold) but the player chose a low-impact action
  or ended the turn.

This module is a pure helper.  Detection runs on ``enemy.powers[]`` and the
action's structured derived view + an externally-provided diagnostic context
(the same ``action_diagnostics`` shape produced by Phase 1 — strategic_skip
and refund classifier).  No localized card text.
"""

from __future__ import annotations

from typing import Any


_INSATIABLE_TOKENS = {
    "the_insatiable",
    "theinsatiable",
    "the_insatiable_boss",
    "insatiable",
    "insatiableboss",
}
_GROWTH_TOKENS = {
    "insatiablegrowthpower",
    "insatiable_growth_power",
    "growthpower",
    "growth_power",
    "scalingpower",
}
_PRESSURE_TOKENS = {
    "insatiablepressurepower",
    "insatiable_pressure_power",
    "pressurepower",
    "urgent_window",
}
_DEVOUR_TOKENS = {
    "insatiabledevourpower",
    "devourpower",
    "devour",
    "feed_intent",
}


def _power_id(power: Any) -> str:
    if isinstance(power, dict):
        for key in ("id", "power_id", "name"):
            value = power.get(key)
            if value:
                return str(value).strip().lower().replace(" ", "")
    return str(power or "").strip().lower().replace(" ", "")


def _power_amount(power: Any) -> float:
    if isinstance(power, dict):
        for key in ("amount", "value", "stacks", "counter"):
            v = power.get(key)
            if isinstance(v, (int, float)):
                return float(v)
    return 0.0


def _enemy_id(enemy: Any) -> str:
    if isinstance(enemy, dict):
        for key in ("combat_id", "id", "model_id", "name"):
            value = enemy.get(key)
            if value:
                return str(value)
    return ""


def _is_insatiable(enemy: dict[str, Any]) -> bool:
    eid = _enemy_id(enemy).lower().replace(" ", "_")
    if any(token in eid for token in _INSATIABLE_TOKENS):
        return True
    powers = enemy.get("powers") if isinstance(enemy.get("powers"), list) else []
    for power in powers:
        pid = _power_id(power)
        if pid in _GROWTH_TOKENS or pid in _PRESSURE_TOKENS or pid in _DEVOUR_TOKENS:
            return True
    return False


def _enemy_intent_damage(enemy: Any) -> float:
    if not isinstance(enemy, dict):
        return 0.0
    intent = enemy.get("intent")
    if isinstance(intent, dict):
        for key in ("total_damage", "damage", "damage_per_hit"):
            v = intent.get(key)
            if isinstance(v, (int, float)) and v > 0:
                return float(v)
    return 0.0


def build_insatiable_state(combat_obs: dict[str, Any] | None) -> dict[str, Any]:
    """Return the ``boss_mechanics.insatiable`` block.

    Stable shape; ``active=False`` means no Insatiable enemy was detected.
    Pressure is derived as ``min(growth_or_scaling / 5 + special_counter / 5, 1)``
    so heads consume a [0, 1] signal regardless of the underlying power-stack
    semantics.
    """
    combat = combat_obs if isinstance(combat_obs, dict) else {}
    enemies = combat.get("enemies") if isinstance(combat.get("enemies"), list) else []
    active = False
    growth_or_scaling = 0.0
    special_counter = 0.0
    urgent_window = False
    captured_powers: list[str] = []
    incoming_damage = 0.0
    enemy_hp_ratio = 0.0
    for enemy in enemies:
        if not isinstance(enemy, dict):
            continue
        if not _is_insatiable(enemy):
            continue
        active = True
        powers = enemy.get("powers") if isinstance(enemy.get("powers"), list) else []
        for power in powers:
            pid = _power_id(power)
            captured_powers.append(pid)
            amount = _power_amount(power)
            if pid in _GROWTH_TOKENS:
                growth_or_scaling = max(growth_or_scaling, amount)
            elif pid in _PRESSURE_TOKENS:
                special_counter = max(special_counter, amount)
                if amount > 0:
                    urgent_window = True
            elif pid in _DEVOUR_TOKENS:
                special_counter = max(special_counter, amount)
                urgent_window = True
        incoming_damage = max(incoming_damage, _enemy_intent_damage(enemy))
        try:
            hp = float(enemy.get("hp") or 0)
            mhp = float(enemy.get("max_hp") or 0)
            if mhp > 0:
                enemy_hp_ratio = max(enemy_hp_ratio, hp / mhp)
        except (TypeError, ValueError):
            pass
    pressure = min(1.0, growth_or_scaling / 5.0 + special_counter / 5.0)
    return {
        "active": active,
        "pressure": float(pressure),
        "growth_or_scaling": float(growth_or_scaling),
        "special_counter": float(special_counter),
        "urgent_window": bool(urgent_window),
        "incoming_damage": float(incoming_damage),
        "enemy_hp_ratio": float(enemy_hp_ratio),
        "powers": list(captured_powers),
    }


def _action_kind(action: dict[str, Any] | None) -> str:
    if not isinstance(action, dict):
        return ""
    return str(action.get("kind") or action.get("family") or "").strip().lower()


def _action_card_combat(action: dict[str, Any] | None) -> dict[str, Any]:
    if not isinstance(action, dict):
        return {}
    card = action.get("card")
    profile = card.get("card_effect_profile") if isinstance(card, dict) and isinstance(card.get("card_effect_profile"), dict) else {}
    derived = profile.get("derived_view") if isinstance(profile, dict) else {}
    if not isinstance(derived, dict):
        return {}
    section = derived.get("combat_effect")
    return section if isinstance(section, dict) else {}


def _player_block(player_obs: dict[str, Any] | None) -> float:
    if not isinstance(player_obs, dict):
        return 0.0
    try:
        return float(player_obs.get("block") or 0)
    except (TypeError, ValueError):
        return 0.0


def classify_insatiable_action_offenders(
    combat_obs: dict[str, Any] | None,
    action: dict[str, Any] | None,
    *,
    player_obs: dict[str, Any] | None = None,
    action_diagnostics: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Per-action Insatiable offender flag dict.

    ``action_diagnostics`` mirrors the Phase 1 taxonomy context: it is the
    dict produced by the strategic-skip + refund classifiers attached to the
    selected action.  When omitted, only state-derivable offenders fire
    (`bad_end_turn`, `missed_pressure_window`).
    """
    state = build_insatiable_state(combat_obs)
    base = {
        "insatiable_strategic_skip": False,
        "insatiable_refund_no_followup": False,
        "insatiable_bad_end_turn": False,
        "insatiable_missed_pressure_window": False,
        "insatiable_pressure_action_selected": False,
    }
    if not state["active"]:
        return base

    diag = action_diagnostics if isinstance(action_diagnostics, dict) else {}
    kind = _action_kind(action)
    combat_eff = _action_card_combat(action)
    damage = float(combat_eff.get("damage") or 0)
    block = float(combat_eff.get("block") or 0)
    weak = float(combat_eff.get("weak") or 0)
    vulnerable = float(combat_eff.get("vulnerable") or 0)
    incoming = state["incoming_damage"]
    standing_block = _player_block(player_obs)
    incoming_after_block = max(0.0, incoming - standing_block - block)
    urgent = state["urgent_window"] or state["pressure"] >= 0.5
    high_impact = damage >= 8 or vulnerable > 0 or weak > 0 or block + standing_block >= max(incoming, 1.0)

    return {
        "insatiable_strategic_skip": bool(diag.get("strategic_skip_selected")),
        "insatiable_refund_no_followup": bool(
            diag.get("refund_followup_class") == "refund_no_followup_low_value"
        ),
        "insatiable_bad_end_turn": bool(
            kind == "end_turn" and incoming_after_block >= 12.0
        ),
        "insatiable_missed_pressure_window": bool(
            urgent and not high_impact and kind in {"end_turn", "play_card"}
        ),
        "insatiable_pressure_action_selected": bool(urgent and high_impact and kind == "play_card"),
    }
