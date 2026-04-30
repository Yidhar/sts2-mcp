"""Kaiser Crab back-attack semantics (TASK-E1).

The bridge surfaces Kaiser-specific mechanics through two power channels:

* ``BackAttackLeftPower`` / ``BackAttackRightPower`` on enemy parts: tells us
  which side the enemy lives on (NOT ``enemy.side`` — that is the camp
  ally/enemy axis, NOT left/right).
* ``SurroundedPower`` on the player creature: ``surrounded.facing`` resolves
  to ``"left"`` or ``"right"``.  When the player is facing ``X`` and an enemy
  with a ``BackAttackXPower`` strikes from ``X``'s OPPOSITE side, damage is
  multiplied (back-attack multiplier).

This module is a pure helper.  Observation/action encoders consume:

* :func:`build_kaiser_state` — returns the ``boss_mechanics.kaiser`` block.
* :func:`classify_kaiser_action_mechanism` — returns the per-action
  ``mechanism`` dict including facing-change estimates and risk delta.
"""

from __future__ import annotations

from typing import Any


_BACK_ATTACK_LEFT_TOKENS = {"backattackleftpower", "back_attack_left_power", "backattackleft"}
_BACK_ATTACK_RIGHT_TOKENS = {"backattackrightpower", "back_attack_right_power", "backattackright"}
_SURROUNDED_TOKENS = {"surroundedpower", "surrounded_power", "surrounded"}
_BACK_ATTACK_MULTIPLIER = 1.5
_FRONT_ATTACK_MULTIPLIER = 1.0


def _power_id(power: Any) -> str:
    if isinstance(power, dict):
        for key in ("id", "power_id", "name"):
            value = power.get(key)
            if value:
                return str(value).strip().lower()
    return str(power or "").strip().lower()


def _enemy_powers(enemy: Any) -> list[Any]:
    if not isinstance(enemy, dict):
        return []
    powers = enemy.get("powers")
    return powers if isinstance(powers, list) else []


def _enemy_back_attack_position(enemy: dict[str, Any]) -> str | None:
    for power in _enemy_powers(enemy):
        pid = _power_id(power)
        if pid in _BACK_ATTACK_LEFT_TOKENS:
            return "left"
        if pid in _BACK_ATTACK_RIGHT_TOKENS:
            return "right"
    return None


def _enemy_id(enemy: Any) -> str:
    if isinstance(enemy, dict):
        for key in ("combat_id", "id", "model_id", "name"):
            value = enemy.get(key)
            if value:
                return str(value)
    return ""


def _surrounded_facing(player_payload: dict[str, Any] | None) -> str | None:
    if not isinstance(player_payload, dict):
        return None
    powers = player_payload.get("powers")
    if isinstance(powers, list):
        for power in powers:
            pid = _power_id(power)
            if pid not in _SURROUNDED_TOKENS:
                continue
            if isinstance(power, dict):
                facing = power.get("facing") or power.get("Facing")
                if isinstance(facing, str):
                    text = facing.strip().lower()
                    if text in {"left", "right"}:
                        return text
                if isinstance(facing, dict):
                    direction = facing.get("direction") or facing.get("side")
                    if isinstance(direction, str):
                        text = direction.strip().lower()
                        if text in {"left", "right"}:
                            return text
    direct = player_payload.get("facing")
    if isinstance(direct, str):
        text = direct.strip().lower()
        if text in {"left", "right"}:
            return text
    return None


def _attack_intent_damage(enemy: dict[str, Any]) -> float:
    intent = enemy.get("intent") if isinstance(enemy, dict) else None
    if isinstance(intent, dict):
        for key in ("total_damage", "damage", "damage_per_hit"):
            value = intent.get(key)
            if isinstance(value, (int, float)) and value > 0:
                return float(value)
    return 0.0


def _enemy_position(enemy: dict[str, Any], left_ids: set[str], right_ids: set[str]) -> str | None:
    eid = _enemy_id(enemy)
    if eid and eid in left_ids:
        return "left"
    if eid and eid in right_ids:
        return "right"
    return _enemy_back_attack_position(enemy)


def build_kaiser_state(combat_obs: dict[str, Any] | None, player_obs: dict[str, Any] | None = None) -> dict[str, Any]:
    """Return the ``boss_mechanics.kaiser`` block.

    ``combat_obs`` is the combat sub-dict (the same dict bridged as
    ``observation["combat"]``).  ``player_obs`` is the player sub-dict; if
    omitted we fall back to ``combat_obs.get("player")``.
    """
    combat = combat_obs if isinstance(combat_obs, dict) else {}
    enemies = combat.get("enemies") if isinstance(combat.get("enemies"), list) else []
    player = player_obs if isinstance(player_obs, dict) else (combat.get("player") if isinstance(combat.get("player"), dict) else {})

    left_ids: list[str] = []
    right_ids: list[str] = []
    has_left = False
    has_right = False
    for enemy in enemies:
        if not isinstance(enemy, dict):
            continue
        position = _enemy_back_attack_position(enemy)
        if position == "left":
            has_left = True
            eid = _enemy_id(enemy)
            if eid:
                left_ids.append(eid)
        elif position == "right":
            has_right = True
            eid = _enemy_id(enemy)
            if eid:
                right_ids.append(eid)

    active = bool(has_left or has_right)
    facing = _surrounded_facing(player) if active else None

    incoming_multiplier = _FRONT_ATTACK_MULTIPLIER
    back_attack_risk = 0.0
    if active and facing:
        # Back attack fires when an enemy has the BackAttack power on the side
        # OPPOSITE to player's facing — i.e. enemy strikes from the player's
        # back.  Player facing "left" => back is "right" => danger if a right
        # enemy holds BackAttackRightPower.
        opposite_side = "right" if facing == "left" else "left"
        back_attack_present = (opposite_side == "left" and has_left) or (
            opposite_side == "right" and has_right
        )
        if back_attack_present:
            incoming_multiplier = _BACK_ATTACK_MULTIPLIER
            back_attack_risk = 1.0

    return {
        "active": active,
        "player_facing": facing,
        "back_attack_risk": float(back_attack_risk),
        "incoming_multiplier": float(incoming_multiplier),
        "left_enemy_ids": list(left_ids),
        "right_enemy_ids": list(right_ids),
        "has_left_back_attack_power": bool(has_left),
        "has_right_back_attack_power": bool(has_right),
    }


def _action_target_combat_id(action: dict[str, Any] | None) -> str:
    if not isinstance(action, dict):
        return ""
    target = action.get("target")
    if isinstance(target, dict):
        for key in ("combat_id", "id", "model_id"):
            value = target.get(key)
            if value:
                return str(value)
    for key in ("target_combat_id", "target_id"):
        value = action.get(key)
        if value:
            return str(value)
    return ""


def _action_targets_self(action: dict[str, Any] | None) -> bool:
    if not isinstance(action, dict):
        return False
    target = action.get("target") if isinstance(action.get("target"), dict) else {}
    scope = str(target.get("scope") or action.get("target_scope") or "").strip().lower()
    if scope in {"self", "player"}:
        return True
    card = action.get("card") if isinstance(action.get("card"), dict) else {}
    target_type = str(card.get("target_type") or card.get("target") or "").strip().lower()
    return target_type in {"self", "none"}


def _action_is_targeted(action: dict[str, Any] | None) -> bool:
    if not isinstance(action, dict):
        return False
    if _action_target_combat_id(action):
        return True
    target = action.get("target") if isinstance(action.get("target"), dict) else {}
    scope = str(target.get("scope") or action.get("target_scope") or "").strip().lower()
    return scope in {"single_enemy", "any_enemy", "single_opponent", "all_enemies"}


def classify_kaiser_action_mechanism(
    combat_obs: dict[str, Any] | None,
    action: dict[str, Any] | None,
    *,
    player_obs: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Per-action Kaiser mechanism dict.

    Returns the ``mechanism`` shape documented in the Phase 4 spec for any
    action.  When Kaiser is not active the fields collapse to neutral defaults
    (``can_change_facing=False``, ``risk_delta=0.0``).
    """
    state = build_kaiser_state(combat_obs, player_obs=player_obs)
    facing = state["player_facing"]
    family = str((action or {}).get("kind") or (action or {}).get("family") or "").lower()

    base_mechanism: dict[str, Any] = {
        "kaiser_can_change_facing": False,
        "kaiser_changes_facing": False,
        "kaiser_facing_before": facing,
        "kaiser_facing_after_if_action": facing,
        "kaiser_incoming_multiplier_before": state["incoming_multiplier"],
        "kaiser_incoming_multiplier_after_estimate": state["incoming_multiplier"],
        "kaiser_risk_delta": 0.0,
        "kaiser_defense_candidate": False,
        "kaiser_pressure_candidate": False,
    }

    if not state["active"]:
        return base_mechanism

    target_id = _action_target_combat_id(action)
    targets_self = _action_targets_self(action)
    is_targeted = _action_is_targeted(action) and not targets_self

    # Defense candidate: anything that adds block / weak / debuff / reduces
    # incoming damage AND does not change facing.  We rely on the action's
    # derived view (TASK-D1) when available; fall back to card semantic tags.
    defense_candidate = False
    pressure_candidate = False
    card = (action or {}).get("card") if isinstance((action or {}).get("card"), dict) else {}
    profile = card.get("card_effect_profile") if isinstance(card.get("card_effect_profile"), dict) else {}
    derived = profile.get("derived_view") if isinstance(profile, dict) else None
    if isinstance(derived, dict):
        combat_effect = derived.get("combat_effect") or {}
        if (combat_effect.get("block") or 0) > 0 or (combat_effect.get("weak") or 0) > 0 or (combat_effect.get("frail") or 0) > 0:
            defense_candidate = True
        damage = combat_effect.get("damage") or 0
        vulnerable = combat_effect.get("vulnerable") or 0
        if damage >= 6 or vulnerable >= 1:
            pressure_candidate = True

    # Self-target defense action: only contributes to the defense_candidate flag.
    if targets_self:
        return {
            **base_mechanism,
            "kaiser_defense_candidate": bool(defense_candidate),
            "kaiser_pressure_candidate": False,
        }

    if not is_targeted or not target_id:
        return {
            **base_mechanism,
            "kaiser_defense_candidate": bool(defense_candidate),
            "kaiser_pressure_candidate": bool(pressure_candidate),
        }

    enemies = (combat_obs or {}).get("enemies") if isinstance((combat_obs or {}).get("enemies"), list) else []
    target_position: str | None = None
    for enemy in enemies:
        if not isinstance(enemy, dict):
            continue
        if _enemy_id(enemy) == target_id:
            target_position = _enemy_back_attack_position(enemy)
            break

    if target_position is None or facing is None:
        return {
            **base_mechanism,
            "kaiser_defense_candidate": bool(defense_candidate),
            "kaiser_pressure_candidate": bool(pressure_candidate),
        }

    can_change_facing = bool(family in {"play_card", "use_potion"})
    changes_facing = can_change_facing and (target_position != facing)
    facing_after = target_position if changes_facing else facing
    multiplier_before = state["incoming_multiplier"]
    multiplier_after_estimate = multiplier_before
    if changes_facing:
        opposite = "right" if facing_after == "left" else "left"
        back_present = (opposite == "left" and state["has_left_back_attack_power"]) or (
            opposite == "right" and state["has_right_back_attack_power"]
        )
        multiplier_after_estimate = _BACK_ATTACK_MULTIPLIER if back_present else _FRONT_ATTACK_MULTIPLIER
    risk_delta = float(multiplier_after_estimate - multiplier_before)

    return {
        "kaiser_can_change_facing": bool(can_change_facing),
        "kaiser_changes_facing": bool(changes_facing),
        "kaiser_facing_before": facing,
        "kaiser_facing_after_if_action": facing_after,
        "kaiser_incoming_multiplier_before": float(multiplier_before),
        "kaiser_incoming_multiplier_after_estimate": float(multiplier_after_estimate),
        "kaiser_risk_delta": risk_delta,
        "kaiser_defense_candidate": bool(defense_candidate),
        "kaiser_pressure_candidate": bool(pressure_candidate),
    }
