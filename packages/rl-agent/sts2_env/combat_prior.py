"""Pragmatic combat prior biasing for fixed-template combat search."""

from __future__ import annotations

from typing import Any

import numpy as np

from .combat_fixed_action import (
    END_TURN_SLOT,
    MAX_ENEMIES,
    NUM_FIXED_COMBAT_ACTIONS,
    PLAY_ENEMY_BASE,
    PLAY_SELF_BASE,
    PLAY_SELF_COUNT,
    POTION_ENEMY_BASE,
    POTION_SELF_BASE,
    POTION_SELF_COUNT,
    FixedCombatActionBinding,
)
from .combat_tactical_local import CombatTacticalAnalysis, TacticalAction, analyze_local_combat_turn


def _float(value: Any, default: float = 0.0) -> float:
    if value is None:
        return default
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _preview_metric(source: dict[str, Any] | None, key: str) -> float:
    if not isinstance(source, dict):
        return 0.0
    effect_preview = source.get("effect_preview") if isinstance(source.get("effect_preview"), dict) else {}
    value = effect_preview.get(key, source.get(key))
    try:
        return float(value or 0.0)
    except (TypeError, ValueError):
        return 0.0


def _positive_progress_score(action: dict[str, Any]) -> float:
    kind = str(action.get("kind") or "").strip().lower()
    if kind not in {"play_card", "use_potion"}:
        return 0.0
    source = action.get("card") if kind == "play_card" else action.get("potion")
    if not isinstance(source, dict):
        return 0.0
    score = 0.0
    for key, weight in (
        ("damage", 1.0),
        ("block", 0.8),
        ("draw", 0.6),
        ("heal", 0.6),
        ("weak", 0.5),
        ("vulnerable", 0.5),
        ("strength", 0.4),
        ("dexterity", 0.4),
    ):
        score += _preview_metric(source, key) * weight
    if str(source.get("type") or "").strip().lower() == "power":
        score += 2.0
    if _float(source.get("cost"), 99.0) <= 0.0:
        score += 0.75
    return score


def _enemy_target_bonus(raw_obs: dict[str, Any], action: dict[str, Any]) -> float:
    combat = raw_obs.get("combat") if isinstance(raw_obs.get("combat"), dict) else {}
    enemies = combat.get("enemies") if isinstance(combat.get("enemies"), list) else []
    target = action.get("target") if isinstance(action.get("target"), dict) else {}
    target_index = target.get("target_index")
    if target_index is None:
        target_index = target.get("index", action.get("target_index"))
    try:
        target_index = int(target_index)
    except (TypeError, ValueError):
        return 0.0
    if not (0 <= target_index < len(enemies)):
        return 0.0
    enemy = enemies[target_index]
    if not isinstance(enemy, dict):
        return 0.0
    hp = _float(enemy.get("hp", enemy.get("current_hp")))
    total_damage = _preview_metric(action.get("card") if isinstance(action.get("card"), dict) else action.get("potion"), "damage")
    bonus = 0.0
    if hp > 0.0 and total_damage >= hp:
        bonus += 0.45
    intent = enemy.get("intent") if isinstance(enemy.get("intent"), dict) else {}
    incoming = _float(intent.get("total_damage"))
    bonus += min(incoming / 40.0, 0.2)
    return bonus


def _is_pure_defense_action(action: dict[str, Any]) -> bool:
    card = action.get("card")
    if not isinstance(card, dict):
        return False
    return (
        _preview_metric(card, "block") > 0.0
        and _preview_metric(card, "damage") <= 0.0
        and _preview_metric(card, "draw") <= 0.0
        and _preview_metric(card, "weak") <= 0.0
        and _preview_metric(card, "vulnerable") <= 0.0
        and _preview_metric(card, "heal") <= 0.0
        and _preview_metric(card, "strength") <= 0.0
        and _preview_metric(card, "dexterity") <= 0.0
    )


def _slot_matches_tactical(
    slot: int,
    tactical_action: TacticalAction,
    raw_obs: dict[str, Any],
) -> bool:
    if PLAY_SELF_BASE <= slot < PLAY_SELF_BASE + PLAY_SELF_COUNT:
        hand_index = slot - PLAY_SELF_BASE
        return hand_index == tactical_action.card_index and not tactical_action.target_entity_id

    if PLAY_ENEMY_BASE <= slot < POTION_SELF_BASE:
        offset = slot - PLAY_ENEMY_BASE
        hand_index = offset // MAX_ENEMIES
        enemy_index = offset % MAX_ENEMIES
        if hand_index != tactical_action.card_index:
            return False
        combat = raw_obs.get("combat") if isinstance(raw_obs.get("combat"), dict) else {}
        enemies = combat.get("enemies") if isinstance(combat.get("enemies"), list) else []
        if not (0 <= enemy_index < len(enemies)):
            return False
        enemy = enemies[enemy_index]
        if not isinstance(enemy, dict):
            return False
        entity_id = str(enemy.get("entity_id", enemy.get("id", enemy.get("name", ""))) or "")
        return entity_id == tactical_action.target_entity_id
    return False


def _end_turn_bias(raw_obs: dict[str, Any], legal_actions: list[dict[str, Any]]) -> float:
    combat = raw_obs.get("combat") if isinstance(raw_obs.get("combat"), dict) else {}
    energy = _float(combat.get("energy"))
    if energy <= 0.0:
        return 0.0
    positive_scores = [_positive_progress_score(action) for action in legal_actions]
    if not any(score > 0.5 for score in positive_scores):
        return 0.0
    zero_cost_exists = any(
        isinstance(action.get("card"), dict) and _float((action.get("card") or {}).get("cost"), 99.0) <= 0.0 and _positive_progress_score(action) > 0.5
        for action in legal_actions
    )
    bias = -(0.9 + (0.15 * min(energy, 3.0)))
    if zero_cost_exists:
        bias -= 0.25
    return bias


def build_combat_prior_bias(
    raw_obs: dict[str, Any] | None,
    legal_actions: list[dict[str, Any]] | None,
    binding: FixedCombatActionBinding,
    *,
    tactical_analysis: CombatTacticalAnalysis | None = None,
) -> np.ndarray:
    bias = np.zeros(NUM_FIXED_COMBAT_ACTIONS, dtype=np.float32)
    if not isinstance(raw_obs, dict) or not isinstance(legal_actions, list):
        return bias

    analysis = tactical_analysis or analyze_local_combat_turn(raw_obs)
    combat = raw_obs.get("combat") if isinstance(raw_obs.get("combat"), dict) else {}
    player = raw_obs.get("player") if isinstance(raw_obs.get("player"), dict) else {}
    current_block = _float(player.get("block"))
    incoming_damage = 0.0
    enemies = combat.get("enemies") if isinstance(combat.get("enemies"), list) else []
    for enemy in enemies:
        if not isinstance(enemy, dict):
            continue
        intent = enemy.get("intent") if isinstance(enemy.get("intent"), dict) else {}
        incoming_damage += _float(intent.get("total_damage"))

    for slot, legal_index in binding.slot_to_legal.items():
        action = legal_actions[legal_index]
        kind = str(action.get("kind") or "").strip().lower()
        if kind in {"play_card", "use_potion"}:
            local_bias = 0.0
            progress = _positive_progress_score(action)
            if progress > 0.0:
                local_bias += min(progress / 12.0, 0.35)
            if kind == "play_card":
                local_bias += _enemy_target_bonus(raw_obs, action)
                card = action.get("card") if isinstance(action.get("card"), dict) else {}
                if str(card.get("type") or "").strip().lower() == "power" and _float(combat.get("round")) <= 3.0:
                    local_bias += 0.15
                if _is_pure_defense_action(action):
                    if current_block >= max(0.0, incoming_damage - 0.5):
                        local_bias -= 0.20
                    elif incoming_damage > current_block:
                        local_bias += min((incoming_damage - current_block) / 25.0, 0.25)
            else:
                if incoming_damage > current_block:
                    local_bias += min((incoming_damage - current_block) / 30.0, 0.25)
            bias[slot] += float(local_bias)

    if binding.mask[END_TURN_SLOT]:
        bias[END_TURN_SLOT] += _end_turn_bias(raw_obs, legal_actions)

    preferred_actions = analysis.lethal_root_actions or analysis.best_root_actions
    if analysis.available and preferred_actions:
        boost = 1.75 if analysis.lethal_root_actions else 1.15
        for slot in binding.legal_slots():
            if any(_slot_matches_tactical(slot, tactical_action, raw_obs) for tactical_action in preferred_actions):
                bias[slot] += boost
            elif PLAY_ENEMY_BASE <= slot < POTION_SELF_BASE and analysis.min_required_block_after_best_kill <= current_block:
                bias[slot] -= 0.20
        if binding.mask[END_TURN_SLOT] and preferred_actions:
            bias[END_TURN_SLOT] -= 0.35

    return bias
