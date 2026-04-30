"""Ceremonial Beast mechanics (TASK-E2).

The Ceremonial Beast can apply a one-card lock and a stun window — two
mechanics that completely change the value calculus of a turn:

* **One-card lock** — only one player action resolves before the lock kicks
  the turn back to the boss.  ``draw`` / ``create`` cards lose follow-up
  value because the next playable card cannot be reached.
* **Stun window** — a brief opportunity where a high-impact action stuns the
  boss for a turn.  Wasting the window with a low-impact card or
  ``end_turn`` is a classic offender.

This module exposes two pure helpers:

* :func:`build_ceremonial_state` — returns the ``boss_mechanics.ceremonial``
  block from the combat observation.
* :func:`classify_ceremonial_action_mechanism` — returns the per-action
  ``mechanism`` dict (impact score + waste / use / setup flags).

Both helpers stay localization-free: power detection runs on the
``enemy.powers[].id`` channel and the action's structured ``derived_view``
(TASK-D1) — never on localized card text.
"""

from __future__ import annotations

from typing import Any


_ONE_CARD_LOCK_TOKENS = {
    "ceremonialonecardlockpower",
    "ceremonial_one_card_lock_power",
    "onecardlockpower",
    "one_card_lock_power",
    "onecardlock",
}
_STUN_WINDOW_TOKENS = {
    "ceremonialstunwindowpower",
    "ceremonial_stun_window_power",
    "stunwindowpower",
    "stun_window_power",
    "stunwindow",
}
_STUN_VULNERABILITY_TOKENS = {
    "ceremonialstunvulnerablepower",
    "ceremonial_stun_vulnerable_power",
    "stunvulnerable",
}
_BEAST_TOKENS = {
    "ceremonial_beast",
    "ceremonialbeast",
    "ceremonial_beast_boss",
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


def _enemy_powers(enemy: Any) -> list[Any]:
    if not isinstance(enemy, dict):
        return []
    powers = enemy.get("powers")
    return powers if isinstance(powers, list) else []


def _enemy_intent_damage(enemy: Any) -> float:
    if not isinstance(enemy, dict):
        return 0.0
    intent = enemy.get("intent")
    if isinstance(intent, dict):
        for key in ("total_damage", "damage", "damage_per_hit"):
            value = intent.get(key)
            if isinstance(value, (int, float)) and value > 0:
                return float(value)
    return 0.0


def _enemy_id(enemy: Any) -> str:
    if isinstance(enemy, dict):
        for key in ("combat_id", "id", "model_id", "name"):
            value = enemy.get(key)
            if value:
                return str(value)
    return ""


def _is_ceremonial_enemy(enemy: dict[str, Any]) -> bool:
    eid = _enemy_id(enemy).lower().replace(" ", "_")
    if any(token in eid for token in _BEAST_TOKENS):
        return True
    for power in _enemy_powers(enemy):
        pid = _power_id(power)
        if pid in _ONE_CARD_LOCK_TOKENS or pid in _STUN_WINDOW_TOKENS:
            return True
    return False


def build_ceremonial_state(combat_obs: dict[str, Any] | None) -> dict[str, Any]:
    """Return the ``boss_mechanics.ceremonial`` block.

    The block always carries a stable shape; ``active=False`` means no
    Ceremonial enemy was detected and downstream code should ignore the
    other counters.
    """
    combat = combat_obs if isinstance(combat_obs, dict) else {}
    enemies = combat.get("enemies") if isinstance(combat.get("enemies"), list) else []
    active = False
    one_card_lock_active = False
    stun_window_active = False
    lock_counter = 0.0
    stun_counter = 0.0
    incoming_after_lock = 0.0
    for enemy in enemies:
        if not isinstance(enemy, dict):
            continue
        if not _is_ceremonial_enemy(enemy):
            continue
        active = True
        for power in _enemy_powers(enemy):
            pid = _power_id(power)
            if pid in _ONE_CARD_LOCK_TOKENS:
                one_card_lock_active = True
                lock_counter = max(lock_counter, _power_amount(power))
            elif pid in _STUN_WINDOW_TOKENS:
                stun_window_active = True
                stun_counter = max(stun_counter, _power_amount(power))
        intent = _enemy_intent_damage(enemy)
        incoming_after_lock = max(incoming_after_lock, intent)
    actions_remaining = 1 if one_card_lock_active else 0
    return {
        "active": active,
        "one_card_lock_active": bool(one_card_lock_active),
        "stun_window_active": bool(stun_window_active),
        "actions_remaining_this_turn": int(actions_remaining),
        "lock_counter": float(lock_counter),
        "stun_counter": float(stun_counter),
        "incoming_after_lock": float(incoming_after_lock),
    }


def _action_card(action: dict[str, Any] | None) -> dict[str, Any]:
    if not isinstance(action, dict):
        return {}
    card = action.get("card")
    return card if isinstance(card, dict) else {}


def _action_derived_combat(action: dict[str, Any] | None) -> dict[str, Any]:
    card = _action_card(action)
    profile = card.get("card_effect_profile") if isinstance(card.get("card_effect_profile"), dict) else {}
    derived = profile.get("derived_view") if isinstance(profile, dict) else {}
    if not isinstance(derived, dict):
        return {}
    section = derived.get("combat_effect")
    return section if isinstance(section, dict) else {}


def _action_derived_hand_mutation(action: dict[str, Any] | None) -> dict[str, Any]:
    card = _action_card(action)
    profile = card.get("card_effect_profile") if isinstance(card.get("card_effect_profile"), dict) else {}
    derived = profile.get("derived_view") if isinstance(profile, dict) else {}
    if not isinstance(derived, dict):
        return {}
    section = derived.get("hand_mutation")
    return section if isinstance(section, dict) else {}


def _action_kind(action: dict[str, Any] | None) -> str:
    if not isinstance(action, dict):
        return ""
    return str(action.get("kind") or action.get("family") or "").strip().lower()


def _player_hp(player_obs: dict[str, Any] | None) -> float:
    if not isinstance(player_obs, dict):
        return 0.0
    try:
        return float(player_obs.get("hp") or 0)
    except (TypeError, ValueError):
        return 0.0


def _player_block(player_obs: dict[str, Any] | None) -> float:
    if not isinstance(player_obs, dict):
        return 0.0
    try:
        return float(player_obs.get("block") or 0)
    except (TypeError, ValueError):
        return 0.0


def _ceremonial_target_hp(combat_obs: dict[str, Any] | None) -> float:
    enemies = (combat_obs or {}).get("enemies") if isinstance((combat_obs or {}).get("enemies"), list) else []
    for enemy in enemies:
        if isinstance(enemy, dict) and _is_ceremonial_enemy(enemy):
            try:
                return float(enemy.get("hp") or 0)
            except (TypeError, ValueError):
                return 0.0
    return 0.0


def classify_ceremonial_action_mechanism(
    combat_obs: dict[str, Any] | None,
    action: dict[str, Any] | None,
    *,
    player_obs: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Per-action Ceremonial mechanism dict.

    Returns the ``mechanism`` shape from the Phase 4 spec for any action.
    When Ceremonial is not active the fields collapse to neutral defaults.
    """
    state = build_ceremonial_state(combat_obs)
    base = {
        "ceremonial_single_action_impact_score": 0.0,
        "ceremonial_wastes_one_card_lock": False,
        "ceremonial_uses_stun_window": False,
        "ceremonial_sets_up_stun": False,
        "ceremonial_low_impact_under_lock": False,
        "ceremonial_missed_stun_window": False,
        "ceremonial_bad_end_turn_under_lock": False,
    }
    if not state["active"]:
        return base

    kind = _action_kind(action)
    combat_eff = _action_derived_combat(action)
    hand_mut = _action_derived_hand_mutation(action)
    damage = float(combat_eff.get("damage") or 0)
    block = float(combat_eff.get("block") or 0)
    weak = float(combat_eff.get("weak") or 0)
    vulnerable = float(combat_eff.get("vulnerable") or 0)
    draw = int(hand_mut.get("draw") or 0)
    creates = bool(hand_mut.get("creates_cards"))

    target_hp = _ceremonial_target_hp(combat_obs)
    incoming = state["incoming_after_lock"]
    player_hp = _player_hp(player_obs)
    player_block = _player_block(player_obs)
    incoming_after_block = max(0.0, incoming - player_block)

    # ---------------- Score model under one-card lock ----------------
    impact = 0.0
    if damage > 0 and target_hp > 0 and damage >= target_hp:
        impact = max(impact, 1.0)  # lethal
    if damage > 0:
        # damage component scales with the fraction of remaining boss HP it shaves.
        if target_hp > 0:
            impact = max(impact, min(damage / target_hp, 0.95))
        else:
            impact = max(impact, min(damage / 20.0, 0.6))
    if block > 0 and incoming > 0:
        if block + player_block >= incoming:
            impact = max(impact, 0.85)  # block-lethal coverage
        else:
            impact = max(impact, min(block / max(incoming, 1.0), 0.7))
    if weak > 0:
        impact = max(impact, 0.45)
    if vulnerable > 0:
        impact = max(impact, 0.55)

    # Draw-only under lock: low impact because the drawn card cannot be played
    # this turn under the one-card lock.
    if state["one_card_lock_active"] and draw > 0 and damage == 0 and block == 0:
        impact = min(impact, 0.15)

    # Stun window — using a high-impact action inside the window is the
    # textbook "use the window" play.  An impact >= 0.5 inside the window
    # is considered "uses the window".
    uses_stun_window = bool(state["stun_window_active"] and impact >= 0.5)
    sets_up_stun = bool(
        not state["stun_window_active"]
        and (weak > 0 or vulnerable > 0 or block + player_block >= incoming)
    )

    # Offenders --------------------------------------------------------------
    low_impact_under_lock = bool(state["one_card_lock_active"] and impact < 0.25 and kind == "play_card")
    bad_end_turn_under_lock = bool(state["one_card_lock_active"] and kind == "end_turn" and incoming_after_block >= max(1.0, 0.3 * max(player_hp, 1.0)))
    missed_stun_window = bool(state["stun_window_active"] and not uses_stun_window and kind in {"end_turn", "play_card"})
    wastes_one_card_lock = bool(state["one_card_lock_active"] and kind == "play_card" and impact < 0.15)

    return {
        "ceremonial_single_action_impact_score": float(impact),
        "ceremonial_wastes_one_card_lock": wastes_one_card_lock,
        "ceremonial_uses_stun_window": uses_stun_window,
        "ceremonial_sets_up_stun": sets_up_stun,
        "ceremonial_low_impact_under_lock": low_impact_under_lock,
        "ceremonial_missed_stun_window": missed_stun_window,
        "ceremonial_bad_end_turn_under_lock": bad_end_turn_under_lock,
    }
