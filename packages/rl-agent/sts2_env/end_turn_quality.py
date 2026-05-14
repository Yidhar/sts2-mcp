"""Strict End Turn quality helpers shared by combat/full-run envs.

The important distinction is that **leftover energy is not a mistake**.  End
Turn is only wasteful when a stable legal frontier contains a non-EndTurn
action that is both safe and urgent/mandatory.  This module intentionally keeps
the detector conservative so reward shaping does not punish legitimate cases
such as:

* all playable cards were already exhausted/played and only energy remains;
* only optional/non-urgent potion actions remain;
* only block cards remain while enemies are not attacking;
* only setup/self-damage/deferable actions remain.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from typing import Any


Action = Mapping[str, Any]
ScoreFn = Callable[[Action], float]
StrategicSkipFn = Callable[[Action, float, Sequence[Action]], bool]


def _float(value: Any, default: float = 0.0) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return float(default)


def _source_metric(source: Mapping[str, Any] | None, key: str) -> float:
    if not isinstance(source, Mapping):
        return 0.0
    effect_preview = source.get("effect_preview")
    if isinstance(effect_preview, Mapping) and effect_preview.get(key) is not None:
        return max(_float(effect_preview.get(key)), 0.0)
    return max(_float(source.get(key)), 0.0)


def _source_any_metric(source: Mapping[str, Any] | None, keys: Sequence[str]) -> float:
    return max((_source_metric(source, key) for key in keys), default=0.0)


def _card_cost(card: Mapping[str, Any] | None) -> float:
    if not isinstance(card, Mapping):
        return 0.0
    for key in ("cost", "resolved_energy_cost", "canonical_energy_cost"):
        value = card.get(key)
        if value is None:
            continue
        text = str(value).strip().upper()
        if text == "X":
            return 0.0
        parsed = _float(value, default=-1.0)
        if parsed >= 0.0:
            return parsed
    return 0.0


def _source_self_damage(source: Mapping[str, Any] | None) -> float:
    return _source_any_metric(
        source,
        (
            "self_damage",
            "hp_loss",
            "hp_cost",
            "lose_hp",
            "life_loss",
        ),
    )


def incoming_damage_pressure(obs: Mapping[str, Any] | None) -> tuple[float, float, float]:
    """Return ``(incoming_damage, current_block, current_hp)``.

    Damage is summed across living enemies because for EndTurn safety the
    player receives the combined attack pressure.
    """

    if not isinstance(obs, Mapping):
        return (0.0, 0.0, 0.0)
    player = obs.get("player") if isinstance(obs.get("player"), Mapping) else {}
    combat = obs.get("combat") if isinstance(obs.get("combat"), Mapping) else {}
    enemies = combat.get("enemies") if isinstance(combat, Mapping) else []
    incoming = 0.0
    if isinstance(enemies, Sequence) and not isinstance(enemies, (str, bytes)):
        for enemy in enemies:
            if not isinstance(enemy, Mapping):
                continue
            hp = _float(enemy.get("hp", enemy.get("current_hp")))
            if hp <= 0.0:
                continue
            intent = enemy.get("intent") if isinstance(enemy.get("intent"), Mapping) else {}
            total = _float(intent.get("total_damage"))
            if total <= 0.0:
                per_hit = _float(intent.get("damage_per_hit"))
                repeats = max(_float(intent.get("repeats"), 1.0), 1.0)
                total = per_hit * repeats
            incoming += max(total, 0.0)
    return (
        float(incoming),
        _float(player.get("block")),
        _float(player.get("hp", player.get("current_hp"))),
    )


def _enemy_identity_values(enemy: Mapping[str, Any]) -> set[str]:
    values: set[str] = set()
    for key in ("id", "combat_id", "uuid", "instance_uuid", "name"):
        value = enemy.get(key)
        if value is not None:
            values.add(str(value))
    return values


def _action_target_values(action: Action) -> set[str]:
    values: set[str] = set()
    target = action.get("target")
    if isinstance(target, Mapping):
        for key in ("id", "combat_id", "uuid", "instance_uuid", "name"):
            value = target.get(key)
            if value is not None:
                values.add(str(value))
    for key in ("target_id", "target_combat_id", "target_uuid", "target_name"):
        value = action.get(key)
        if value is not None:
            values.add(str(value))
    return values


def _target_enemy_hp(action: Action, obs: Mapping[str, Any] | None) -> float:
    if not isinstance(obs, Mapping):
        return 0.0
    combat = obs.get("combat") if isinstance(obs.get("combat"), Mapping) else {}
    enemies = combat.get("enemies") if isinstance(combat, Mapping) else []
    if not isinstance(enemies, Sequence) or isinstance(enemies, (str, bytes)):
        return 0.0

    target_values = _action_target_values(action)
    living_hp: list[float] = []
    for enemy in enemies:
        if not isinstance(enemy, Mapping):
            continue
        hp = _float(enemy.get("hp", enemy.get("current_hp")))
        if hp <= 0.0:
            continue
        living_hp.append(hp)
        if target_values and target_values.intersection(_enemy_identity_values(enemy)):
            return hp
    if len(living_hp) == 1:
        return living_hp[0]
    return min(living_hp) if living_hp else 0.0


def _total_enemy_hp(obs: Mapping[str, Any] | None) -> float:
    if not isinstance(obs, Mapping):
        return 0.0
    combat = obs.get("combat") if isinstance(obs.get("combat"), Mapping) else {}
    enemies = combat.get("enemies") if isinstance(combat, Mapping) else []
    total = 0.0
    if isinstance(enemies, Sequence) and not isinstance(enemies, (str, bytes)):
        for enemy in enemies:
            if isinstance(enemy, Mapping):
                total += max(_float(enemy.get("hp", enemy.get("current_hp"))), 0.0)
    return total


def _action_roles(action: Action) -> set[str]:
    roles: set[str] = set()
    raw = action.get("semantic_roles")
    if isinstance(raw, Sequence) and not isinstance(raw, (str, bytes)):
        roles.update(str(item).strip().lower() for item in raw if str(item).strip())
    semantic = action.get("semantic")
    if isinstance(semantic, Mapping):
        raw = semantic.get("roles")
        if isinstance(raw, Sequence) and not isinstance(raw, (str, bytes)):
            roles.update(str(item).strip().lower() for item in raw if str(item).strip())
    return roles


def _is_strategic_skip(
    action: Action,
    energy: float,
    legal_actions: Sequence[Action],
    strategic_skip_fn: StrategicSkipFn | None,
) -> bool:
    if strategic_skip_fn is None:
        return False
    try:
        return bool(strategic_skip_fn(action, energy, legal_actions))
    except Exception:
        return False


def _positive_score(action: Action, positive_score_fn: ScoreFn | None) -> float:
    if positive_score_fn is not None:
        try:
            return max(float(positive_score_fn(action)), 0.0)
        except Exception:
            pass
    kind = str(action.get("kind") or "").strip()
    source = action.get("card") if kind == "play_card" else action.get("potion")
    if not isinstance(source, Mapping):
        return 0.0
    return (
        _source_any_metric(source, ("damage", "total_damage", "preview_damage")) * 1.0
        + _source_metric(source, "block") * 0.8
        + _source_metric(source, "draw") * 3.0
        + _source_metric(source, "weak") * 4.0
        + _source_metric(source, "vulnerable") * 4.0
        + _source_metric(source, "heal") * 2.0
        + _source_metric(source, "strength") * 3.0
        + _source_metric(source, "dexterity") * 3.0
        + _source_metric(source, "summon") * 5.0
    )


def _is_urgent_progress_action(
    action: Action,
    *,
    obs: Mapping[str, Any] | None,
    energy: float,
    incoming: float,
    current_block: float,
    hp: float,
    legal_actions: Sequence[Action],
    strategic_skip_fn: StrategicSkipFn | None,
    positive_score_fn: ScoreFn | None,
) -> tuple[bool, str]:
    kind = str(action.get("kind") or "").strip()
    if kind not in ("play_card", "use_potion"):
        return (False, "non_combat_action")

    source = action.get("card") if kind == "play_card" else action.get("potion")
    if not isinstance(source, Mapping):
        return (False, "missing_source")

    if _positive_score(action, positive_score_fn) <= 0.0:
        return (False, "not_positive")
    if _is_strategic_skip(action, energy, legal_actions, strategic_skip_fn):
        return (False, "strategic_defer")

    self_damage = _source_self_damage(source)
    if hp > 0.0 and self_damage >= hp:
        return (False, "self_lethal")

    damage = _source_any_metric(source, ("damage", "total_damage", "preview_damage", "expected_damage"))
    target_hp = _target_enemy_hp(action, obs)
    total_hp = _total_enemy_hp(obs)
    if damage > 0.0:
        if target_hp > 0.0 and damage >= target_hp - 1e-6:
            return (True, "lethal")
        if total_hp > 0.0 and damage >= total_hp - 1e-6:
            return (True, "combat_lethal")

    block = _source_any_metric(source, ("block", "total_block", "preview_block"))
    heal = _source_metric(source, "heal")
    weak = _source_metric(source, "weak")
    vulnerable = _source_metric(source, "vulnerable")
    block_like = block + heal + weak * 3.0
    missing_block = max(incoming - current_block, 0.0)
    would_take_lethal = hp > 0.0 and missing_block >= hp
    if incoming > current_block + 0.5 and block_like > 0.0:
        if would_take_lethal:
            return (True, "prevent_lethal")
        if block > 0.0 and missing_block >= 4.0:
            return (True, "prevent_damage")
        if weak > 0.0 and missing_block >= 8.0:
            return (True, "mitigate_big_attack")

    roles = _action_roles(action)
    if roles.intersection({"mandatory", "mechanism", "boss_mechanism", "required"}):
        return (True, "mandatory_mechanism")

    # Zero-cost high-impact actions are mandatory enough that ending the turn is
    # almost never correct, while low-value zero-cost setup is left alone.
    if kind == "play_card" and _card_cost(source) <= 0.0:
        score = _positive_score(action, positive_score_fn)
        if score >= 8.0 and self_damage <= 0.0:
            return (True, "zero_cost_strong_progress")

    # Non-urgent damage/setup/potions are not strict EndTurn mistakes.  They may
    # still be learned by value/policy, but reward shaping must not punish every
    # leftover-energy defer.
    return (False, "not_urgent")


def strict_end_turn_waste_context(
    obs: Mapping[str, Any] | None,
    legal_actions: Sequence[Action] | None,
    chosen_action: Action | None,
    *,
    positive_score_fn: ScoreFn | None = None,
    strategic_skip_fn: StrategicSkipFn | None = None,
) -> dict[str, Any]:
    """Conservative EndTurn classifier for reward/diagnostic use."""

    actions = [action for action in (legal_actions or []) if isinstance(action, Mapping)]
    selected_is_end_turn = str((chosen_action or {}).get("action_id") or "") == "end_turn"
    incoming, current_block, hp = incoming_damage_pressure(obs)
    combat = obs.get("combat") if isinstance(obs, Mapping) and isinstance(obs.get("combat"), Mapping) else {}
    energy = _float(combat.get("energy"))

    non_end_turn_count = 0
    positive_count = 0
    urgent_indices: list[int] = []
    urgent_reasons: list[str] = []
    playable_card_count = 0
    for idx, action in enumerate(actions):
        if str(action.get("action_id") or "") == "end_turn":
            continue
        non_end_turn_count += 1
        if str(action.get("kind") or "").strip() == "play_card":
            playable_card_count += 1
        if _positive_score(action, positive_score_fn) > 0.0:
            positive_count += 1
        urgent, reason = _is_urgent_progress_action(
            action,
            obs=obs,
            energy=energy,
            incoming=incoming,
            current_block=current_block,
            hp=hp,
            legal_actions=actions,
            strategic_skip_fn=strategic_skip_fn,
            positive_score_fn=positive_score_fn,
        )
        if urgent:
            urgent_indices.append(int(idx))
            urgent_reasons.append(reason)

    wasteful_available = bool(urgent_indices)
    return {
        "energy": float(energy),
        "incoming_damage": float(incoming),
        "current_block": float(current_block),
        "current_hp": float(hp),
        "selected_is_end_turn": bool(selected_is_end_turn),
        "non_end_turn_action_count": int(non_end_turn_count),
        "playable_card_count": int(playable_card_count),
        "positive_action_count": int(positive_count),
        "urgent_positive_action_count": int(len(urgent_indices)),
        "urgent_positive_indices": urgent_indices,
        "urgent_positive_reasons": urgent_reasons,
        "wasteful_end_turn_available": bool(wasteful_available),
        "wasteful_end_turn_selected": bool(selected_is_end_turn and wasteful_available),
        "forced_end_turn": bool(selected_is_end_turn and non_end_turn_count <= 0),
        "benign_leftover_energy": bool(
            selected_is_end_turn
            and energy > 0.05
            and non_end_turn_count <= 0
            and not wasteful_available
        ),
    }
