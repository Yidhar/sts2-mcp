"""Objective-head utilities for long-horizon, low-damage planning.

This module centralizes the planner-facing objective decomposition used by the
target-state refactor:

- survival
- hp_preservation
- build_progress
- resource_efficiency

The same utilities are reused by:
- MuZero multi-head value / reward heads
- objective-aware MCTS scalarization
- replay-buffer target construction from observed transitions
"""

from __future__ import annotations

from typing import Any

import numpy as np
import torch

from .run_memory import _build_profile, _clip01, _count_nonempty_potions, _float
from .semantic_action import semantic_action_signature

OBJECTIVE_HEAD_NAMES = (
    "survival",
    "hp_preservation",
    "build_progress",
    "resource_efficiency",
)
NUM_OBJECTIVE_HEADS = len(OBJECTIVE_HEAD_NAMES)

HEAD_SURVIVAL = 0
HEAD_HP_PRESERVATION = 1
HEAD_BUILD_PROGRESS = 2
HEAD_RESOURCE_EFFICIENCY = 3


def _as_np_context(objective_context: Any) -> np.ndarray:
    if objective_context is None:
        return np.zeros(0, dtype=np.float32)
    if isinstance(objective_context, np.ndarray):
        return objective_context.astype(np.float32, copy=False).reshape(-1)
    if isinstance(objective_context, torch.Tensor):
        return objective_context.detach().cpu().numpy().astype(np.float32, copy=False).reshape(-1)
    return np.asarray(objective_context, dtype=np.float32).reshape(-1)


def _heuristic_context_strength_np(ctx: np.ndarray) -> float:
    """How strongly to trust heuristic objective context in the current episode mode.

    Full-run episodes can lean on heuristic long-horizon priors much more heavily.
    Combat-sandbox episodes should treat them only as weak initialization hints.
    """
    if ctx.size <= 15:
        return 0.35
    long_horizon_mode = float(np.clip(ctx[15], 0.0, 1.0))
    return float(0.20 + 0.80 * long_horizon_mode)


def objective_weight_vector_np(objective_context: Any) -> np.ndarray:
    """Return normalized planner weights for the 4 objective heads."""
    ctx = _as_np_context(objective_context)
    uniform = np.full(NUM_OBJECTIVE_HEADS, 1.0 / NUM_OBJECTIVE_HEADS, dtype=np.float32)
    weights = np.ones(NUM_OBJECTIVE_HEADS, dtype=np.float32)
    if ctx.size >= NUM_OBJECTIVE_HEADS:
        weights[:] = np.clip(ctx[:NUM_OBJECTIVE_HEADS], 0.05, 4.0)
    if ctx.size > 4:
        weights[HEAD_SURVIVAL] += 0.35 * float(ctx[4])  # preserve_hp_bias
    if ctx.size > 14:
        weights[HEAD_HP_PRESERVATION] += 0.40 * float(ctx[14])  # zero_damage_desire
    if ctx.size > 7:
        weights[HEAD_BUILD_PROGRESS] += 0.25 * float(ctx[7])  # greed_upgrade_mode
    if ctx.size > 5:
        weights[HEAD_RESOURCE_EFFICIENCY] += 0.25 * float(ctx[5])  # save_potion_mode
    total = float(weights.sum())
    if total <= 1e-6:
        return uniform
    normalized = weights / total
    heuristic_strength = _heuristic_context_strength_np(ctx)
    return (1.0 - heuristic_strength) * uniform + heuristic_strength * normalized


def objective_weight_vector_torch(
    objective_context: torch.Tensor | None,
    *,
    device: torch.device | None = None,
    dtype: torch.dtype = torch.float32,
) -> torch.Tensor:
    """Torch counterpart of :func:`objective_weight_vector_np`."""
    if objective_context is None:
        target_device = device if device is not None else torch.device("cpu")
        return torch.full(
            (1, NUM_OBJECTIVE_HEADS),
            1.0 / NUM_OBJECTIVE_HEADS,
            device=target_device,
            dtype=dtype,
        )

    ctx = objective_context.to(device=device or objective_context.device, dtype=dtype)
    if ctx.dim() == 1:
        ctx = ctx.unsqueeze(0)

    batch = ctx.shape[0]
    uniform = torch.full((batch, NUM_OBJECTIVE_HEADS), 1.0 / NUM_OBJECTIVE_HEADS, device=ctx.device, dtype=dtype)
    weights = torch.ones(batch, NUM_OBJECTIVE_HEADS, device=ctx.device, dtype=dtype)
    if ctx.shape[1] >= NUM_OBJECTIVE_HEADS:
        weights = torch.clamp(ctx[:, :NUM_OBJECTIVE_HEADS], 0.05, 4.0)
    if ctx.shape[1] > 4:
        weights[:, HEAD_SURVIVAL] = weights[:, HEAD_SURVIVAL] + 0.35 * ctx[:, 4]
    if ctx.shape[1] > 14:
        weights[:, HEAD_HP_PRESERVATION] = weights[:, HEAD_HP_PRESERVATION] + 0.40 * ctx[:, 14]
    if ctx.shape[1] > 7:
        weights[:, HEAD_BUILD_PROGRESS] = weights[:, HEAD_BUILD_PROGRESS] + 0.25 * ctx[:, 7]
    if ctx.shape[1] > 5:
        weights[:, HEAD_RESOURCE_EFFICIENCY] = weights[:, HEAD_RESOURCE_EFFICIENCY] + 0.25 * ctx[:, 5]
    weights_sum = weights.sum(dim=-1, keepdim=True).clamp(min=1e-6)
    normalized = weights / weights_sum
    if ctx.shape[1] > 15:
        heuristic_strength = 0.20 + 0.80 * torch.clamp(ctx[:, 15], 0.0, 1.0)
    else:
        heuristic_strength = torch.full((batch,), 0.35, device=ctx.device, dtype=dtype)
    heuristic_strength = heuristic_strength.unsqueeze(-1)
    return (1.0 - heuristic_strength) * uniform + heuristic_strength * normalized


def scalarize_objective_components_np(components: Any, objective_context: Any) -> np.ndarray:
    """Scalarize objective-head values with planner weights."""
    values = np.asarray(components, dtype=np.float32)
    weights = objective_weight_vector_np(objective_context)
    return np.tensordot(values, weights, axes=([-1], [0]))


def scalarize_objective_components_torch(
    components: torch.Tensor,
    objective_context: torch.Tensor | None,
) -> torch.Tensor:
    """Torch scalarization helper for objective-head values."""
    weights = objective_weight_vector_torch(
        objective_context,
        device=components.device,
        dtype=components.dtype,
    )
    if components.dim() == 1:
        return (components.unsqueeze(0) * weights).sum(dim=-1)
    while weights.dim() < components.dim():
        weights = weights.unsqueeze(1)
    return (components * weights).sum(dim=-1)


def blended_objective_scalar_np(
    scalar_value: float,
    components: Any,
    objective_context: Any,
    *,
    component_blend: float = 0.70,
) -> float:
    """Blend legacy scalar value with objective-head scalarization."""
    component_blend = float(min(max(component_blend, 0.0), 1.0))
    objective_scalar = float(np.asarray(scalarize_objective_components_np(components, objective_context)).reshape(-1)[0])
    return float((1.0 - component_blend) * float(scalar_value) + component_blend * objective_scalar)


def _player_hp(obs: dict[str, Any] | None) -> float:
    player = (obs or {}).get("player") if isinstance(obs, dict) else {}
    return _float(player.get("hp")) if isinstance(player, dict) else 0.0


def _player_max_hp(obs: dict[str, Any] | None) -> float:
    player = (obs or {}).get("player") if isinstance(obs, dict) else {}
    return max(_float(player.get("max_hp"), 1.0), 1.0) if isinstance(player, dict) else 1.0


def _player_block(obs: dict[str, Any] | None) -> float:
    combat = (obs or {}).get("combat") if isinstance(obs, dict) else {}
    return _float(combat.get("block")) if isinstance(combat, dict) else 0.0


def _player_gold(obs: dict[str, Any] | None) -> float:
    player = (obs or {}).get("player") if isinstance(obs, dict) else {}
    return _float(player.get("gold")) if isinstance(player, dict) else 0.0


def _player_relic_count(obs: dict[str, Any] | None) -> int:
    player = (obs or {}).get("player") if isinstance(obs, dict) else {}
    relics = player.get("relics") if isinstance(player, dict) else None
    return len(relics) if isinstance(relics, list) else 0


def _combat_enemy_total_hp(obs: dict[str, Any] | None) -> float:
    combat = (obs or {}).get("combat") if isinstance(obs, dict) else {}
    enemies = combat.get("enemies") if isinstance(combat, dict) else None
    if not isinstance(enemies, list):
        return 0.0
    total = 0.0
    for enemy in enemies:
        if not isinstance(enemy, dict):
            continue
        total += _float(enemy.get("hp", enemy.get("current_hp")))
    return total


def _combat_total_intent_damage(obs: dict[str, Any] | None) -> float:
    combat = (obs or {}).get("combat") if isinstance(obs, dict) else {}
    enemies = combat.get("enemies") if isinstance(combat, dict) else None
    if not isinstance(enemies, list):
        return 0.0
    total = 0.0
    for enemy in enemies:
        if not isinstance(enemy, dict):
            continue
        intent = enemy.get("intent")
        if not isinstance(intent, dict):
            continue
        total_damage = _float(intent.get("total_damage"))
        if total_damage <= 0.0:
            damage_per_hit = _float(intent.get("damage_per_hit"))
            repeats = max(_float(intent.get("repeats"), 1.0), 1.0)
            total_damage = damage_per_hit * repeats
        total += max(total_damage, 0.0)
    return total


def _in_combat(obs: dict[str, Any] | None) -> bool:
    combat = (obs or {}).get("combat") if isinstance(obs, dict) else None
    return isinstance(combat, dict) and bool(combat)


def _floor(obs: dict[str, Any] | None) -> int:
    run = (obs or {}).get("run") if isinstance(obs, dict) else {}
    return int(_float(run.get("floor"))) if isinstance(run, dict) else 0


def _act(obs: dict[str, Any] | None) -> int:
    run = (obs or {}).get("run") if isinstance(obs, dict) else {}
    return int(_float(run.get("act_id"))) if isinstance(run, dict) else 0


def _build_quality_score(obs: dict[str, Any] | None) -> float:
    build = _build_profile(obs)
    deck_size = float(build.get("deck_size", 0.0) or 0.0)
    # Act1 recovery traces showed 16-18 card decks built mostly by repeatedly
    # taking mediocre attacks.  The profile above is density-based, so adding
    # one more attack can look neutral/positive even though it slows rotation.
    # Keep the penalty small and only start it after the starter+early-pick
    # range, so truly high-impact cards can still win via block/draw/scaling.
    deck_bloat = float(np.clip((deck_size - 12.0) / 18.0, 0.0, 1.0))
    score = (
        0.14 * build["frontload"]
        + 0.24 * build["block"]
        + 0.16 * build["draw"]
        + 0.20 * build["scaling"]
        + 0.08 * build["aoe"]
        + 0.05 * build["heal"]
        + 0.18 * build["consistency"]
        + 0.05 * build["x_cost_density"]
        - 0.12 * build["curse_density"]
        - 0.08 * build["high_cost_density"]
        - 0.10 * deck_bloat
    )
    return float(min(max(score, -1.0), 1.5))


def _resource_state_score(obs: dict[str, Any] | None) -> float:
    gold = _player_gold(obs)
    potion_count = _count_nonempty_potions(obs)
    relic_count = _player_relic_count(obs)
    score = (
        0.50 * min(np.log1p(max(gold, 0.0)) / np.log1p(500.0), 1.0)
        + 0.30 * min(potion_count / 5.0, 1.0)
        + 0.20 * min(relic_count / 18.0, 1.0)
    )
    return float(min(max(score, 0.0), 1.5))


def compute_transition_objective_rewards(
    prev_obs: dict[str, Any] | None,
    action: dict[str, Any] | None,
    next_obs: dict[str, Any] | None,
    *,
    prev_planner_context: dict[str, Any] | None = None,
    next_planner_context: dict[str, Any] | None = None,
    terminated: bool = False,
    truncated: bool = False,
) -> np.ndarray:
    """Decompose one observed transition into planner-aligned reward components."""
    prev_hp = _player_hp(prev_obs)
    next_hp = _player_hp(next_obs)
    max_hp = max(_player_max_hp(prev_obs), _player_max_hp(next_obs), 1.0)
    hp_loss = max(prev_hp - next_hp, 0.0)
    prev_block = _player_block(prev_obs)
    next_block = _player_block(next_obs)

    prev_enemy_hp = _combat_enemy_total_hp(prev_obs)
    next_enemy_hp = _combat_enemy_total_hp(next_obs)
    enemy_hp_delta = max(prev_enemy_hp - next_enemy_hp, 0.0)

    prev_intent = _combat_total_intent_damage(prev_obs)
    next_intent = _combat_total_intent_damage(next_obs)
    intent_reduction = prev_intent - next_intent

    prev_build_score = _build_quality_score(prev_obs)
    next_build_score = _build_quality_score(next_obs)
    build_delta = next_build_score - prev_build_score

    prev_resource_score = _resource_state_score(prev_obs)
    next_resource_score = _resource_state_score(next_obs)
    resource_delta = next_resource_score - prev_resource_score

    prev_floor = _floor(prev_obs)
    next_floor = _floor(next_obs)
    prev_act = _act(prev_obs)
    next_act = _act(next_obs)

    semantic = {}
    if isinstance(action, dict):
        if "family" in action and ("semantic_key" in action or "domain" in action):
            semantic = dict(action)
        else:
            semantic = semantic_action_signature(action)
    family = str(semantic.get("family") or "")

    next_objective = (
        next_planner_context.get("objective_context_vector")
        if isinstance(next_planner_context, dict)
        else None
    )
    next_weights = objective_weight_vector_np(next_objective)
    save_potion_mode = float(_as_np_context(next_objective)[5]) if _as_np_context(next_objective).size > 5 else 0.0
    force_rest_mode = float(_as_np_context(next_objective)[6]) if _as_np_context(next_objective).size > 6 else 0.0
    greed_upgrade_mode = float(_as_np_context(next_objective)[7]) if _as_np_context(next_objective).size > 7 else 0.0

    player_dead = next_hp <= 0.0 and (terminated or prev_hp > 0.0)
    cleared_combat = _in_combat(prev_obs) and not _in_combat(next_obs) and next_hp > 0.0
    room_progress = 0.05 * max(next_floor - prev_floor, 0) + 0.10 * max(next_act - prev_act, 0)

    survival = 0.01 + room_progress
    if cleared_combat:
        survival += 0.25
    if terminated and not truncated:
        survival += 0.20 if next_hp > 0.0 else -1.0
    if player_dead:
        survival = -1.0
    survival = float(np.clip(survival, -1.0, 1.0))

    hp_preservation = (
        -min((hp_loss / max_hp) * 4.0, 1.0)
        + float(np.clip(intent_reduction / 35.0, -0.40, 0.55))
        + float(np.clip((next_block - prev_block) / 30.0, -0.20, 0.25))
        + float(np.clip(enemy_hp_delta / 120.0, 0.0, 0.20))
    )
    if cleared_combat and hp_loss <= 0.0:
        hp_preservation += 0.20
    if family == "end_turn" and prev_intent > max(prev_block, 0.0):
        hp_preservation -= 0.10 * (0.5 + next_weights[HEAD_HP_PRESERVATION])
    hp_preservation = float(np.clip(hp_preservation, -1.0, 1.0))

    build_progress = float(np.clip(build_delta * 2.5, -1.0, 1.0))
    if family in {"smith", "deck_upgrade"}:
        build_progress += 0.12 + 0.10 * greed_upgrade_mode
    elif family in {"card_reward", "treasure_relic"}:
        build_progress += 0.10
    elif family == "shop" and build_delta > 0.0:
        build_progress += 0.08
    elif family == "rest":
        build_progress -= 0.05 * max(1.0 - force_rest_mode, 0.0)
    build_progress = float(np.clip(build_progress, -1.0, 1.0))

    resource_efficiency = float(np.clip(resource_delta * 2.0, -1.0, 1.0))
    if family == "use_potion":
        # Potion use is a cost in hallway fights, but in boss/lethal windows the
        # old blanket penalty taught the policy to die with potions.  Without
        # encounter metadata in this pure transition function, infer tactical
        # pressure from incoming intent, low HP, and large enemy HP pools.
        hp_ratio = next_hp / max_hp if max_hp > 0.0 else 0.0
        lethal_pressure = prev_intent > max(prev_block + prev_hp * 0.35, 0.0)
        boss_like_pressure = prev_enemy_hp >= 180.0 or next_enemy_hp >= 180.0
        if lethal_pressure or hp_ratio <= 0.35 or boss_like_pressure:
            resource_efficiency += 0.12 + 0.10 * float(lethal_pressure) + 0.08 * float(boss_like_pressure)
        else:
            resource_efficiency -= 0.10 + 0.20 * save_potion_mode
    elif family == "rest":
        resource_efficiency -= 0.04 * max(1.0 - force_rest_mode, 0.0)
    elif family == "shop" and build_progress > 0.0:
        resource_efficiency += 0.12 * build_progress
    elif family in {"reward", "treasure_relic"} and resource_delta >= 0.0:
        resource_efficiency += 0.08
    resource_efficiency = float(np.clip(resource_efficiency, -1.0, 1.0))

    components = np.asarray(
        [
            survival,
            hp_preservation,
            build_progress,
            resource_efficiency,
        ],
        dtype=np.float32,
    )

    # Keep reward components on a similar dynamic range as the scalar reward path.
    # Low-damage priority should remain the dominant objective under pressure.
    if player_dead:
        components[HEAD_HP_PRESERVATION] = min(components[HEAD_HP_PRESERVATION], -1.0)
    return components
