"""Replay buffer and trajectory storage for MuZero training.

This module implements:
- GameTrajectory: stores one episode with observation, actions, rewards, and targets
- MuZeroReplayBuffer: prioritized sampling buffer for training the MuZero network
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List

import numpy as np
import torch

from combat_snapshot_dataset import normalize_encounter_id
from sts2_env.objective_heads import (
    HEAD_BUILD_PROGRESS,
    HEAD_HP_PRESERVATION,
    HEAD_RESOURCE_EFFICIENCY,
    HEAD_SURVIVAL,
    NUM_OBJECTIVE_HEADS,
    scalarize_objective_components_np,
)
from sts2_env.observation_v2 import (
    ACTION_FEAT_DIM,
    CARD_FEAT_DIM,
    DECK_FEAT_DIM,
    ENEMY_FEAT_DIM,
    MAX_ACTIONS,
    MAX_DECK,
    MAX_ENEMIES,
    MAX_HAND,
    MAX_POTIONS,
    MAX_RELICS,
    MAX_ROUTE_NODES,
    NUM_DOMAINS,
    OBJECTIVE_DIM,
    POWER_DIM,
    RELIC_SIGNAL_DIM,
    ROUTE_NODE_FEAT_DIM,
    ROUTE_SUMMARY_DIM,
    RUN_MEMORY_DIM,
    SCALAR_DIM,
    SEM_ACTION_FEAT_DIM,
)
from sts2_env.observation_v3 import (
    ENTITY_HASH_BUCKETS,
    MAX_CANDIDATE_LOCAL_TOKENS,
    MAX_OWNER_ID,
    MAX_ORDER_ID,
    MAX_ROLE_ID,
    MAX_WORLD_TOKENS,
    MAX_ZONE_ID,
    NUM_TOKEN_TYPES,
    TOKEN_FEAT_DIM,
)
from muzero.sts2_env.semantic_rollout import SEMANTIC_ROLLOUT_SIZE, aggregate_concrete_policy_to_semantic
from sts2_env.text_encoder import TEXT_DIM


_PACKED_OBS_MARKER = "__packed_obs_v1__"
MUZERO_REPLAY_SCHEMA_VERSION = "muzero_replay_v2_token_v3_pass_large_caps"
MUZERO_OBS_SCHEMA_VERSION = "token_v3_attention_obs_v5_pass_large"
MUZERO_CHECKPOINT_COMPATIBILITY_VERSION = 2


def observation_shape_caps() -> Dict[str, int]:
    """Runtime observation caps that affect packed replay tensor shapes.

    These values are schema-defining: changing any of them makes old packed
    replay buffers unsafe to mix into a new run even when Python can technically
    pad/trim arrays.  Checkpoints persist this payload so resume can refuse
    replay warm-starts after shape-cap upgrades.
    """

    return {
        "max_hand": int(MAX_HAND),
        "max_deck": int(MAX_DECK),
        "max_enemies": int(MAX_ENEMIES),
        "max_relics": int(MAX_RELICS),
        "max_potions": int(MAX_POTIONS),
        "max_actions": int(MAX_ACTIONS),
        "max_route_nodes": int(MAX_ROUTE_NODES),
        "max_world_tokens": int(MAX_WORLD_TOKENS),
        "max_candidate_local_tokens": int(MAX_CANDIDATE_LOCAL_TOKENS),
    }


def replay_schema_metadata() -> Dict[str, Any]:
    """Replay/checkpoint schema metadata for compatibility guards."""

    return {
        "obs_schema_version": MUZERO_OBS_SCHEMA_VERSION,
        "replay_schema_version": MUZERO_REPLAY_SCHEMA_VERSION,
        "checkpoint_compatibility_version": int(MUZERO_CHECKPOINT_COMPATIBILITY_VERSION),
        "observation_shape_caps": observation_shape_caps(),
    }

_BUILD_ROUTE_DECISION_DOMAINS = frozenset({"build", "route"})
_BUILD_ROUTE_ACTION_FAMILIES = frozenset({
    "map",
    "reward",
    "card_reward",
    "shop",
    "rest",
    "smith",
    "deck_upgrade",
    "event_option",
    "treasure_relic",
})
_WASTEFUL_SKIP_SELECTIONS = frozenset({"skip", "cancel", "close"})
_VALID_ENCOUNTER_TIERS = frozenset({"weak", "normal", "elite", "boss"})

_DENSE_OBS_FIXED_SHAPES: dict[str, tuple[int, ...]] = {
    "scalars": (SCALAR_DIM,),
    "decision_domain": (NUM_DOMAINS,),
    "hand": (MAX_HAND, CARD_FEAT_DIM),
    "hand_text": (MAX_HAND, TEXT_DIM),
    "hand_mask": (MAX_HAND,),
    "deck": (MAX_DECK, DECK_FEAT_DIM),
    "deck_text": (MAX_DECK, TEXT_DIM),
    "deck_mask": (MAX_DECK,),
    "enemies": (MAX_ENEMIES, ENEMY_FEAT_DIM),
    "enemy_text": (MAX_ENEMIES, TEXT_DIM),
    "enemy_mask": (MAX_ENEMIES,),
    "player_powers": (POWER_DIM,),
    "relic_signals": (RELIC_SIGNAL_DIM,),
    "run_memory": (RUN_MEMORY_DIM,),
    "objective_context": (OBJECTIVE_DIM,),
    "relics": (MAX_RELICS, TEXT_DIM),
    "relic_mask": (MAX_RELICS,),
    "potions": (MAX_POTIONS, TEXT_DIM),
    "potion_mask": (MAX_POTIONS,),
    "context_text": (TEXT_DIM,),
    "actions": (MAX_ACTIONS, ACTION_FEAT_DIM),
    "action_text": (MAX_ACTIONS, TEXT_DIM),
    "semantic_actions": (MAX_ACTIONS, SEM_ACTION_FEAT_DIM),
    "semantic_action_text": (MAX_ACTIONS, TEXT_DIM),
    "route_summary": (MAX_ACTIONS, ROUTE_SUMMARY_DIM),
    "route_nodes": (MAX_ACTIONS, MAX_ROUTE_NODES, ROUTE_NODE_FEAT_DIM),
    "route_node_mask": (MAX_ACTIONS, MAX_ROUTE_NODES),
    "action_mask": (MAX_ACTIONS,),
}

_TOKEN_OBS_FIXED_SHAPES: dict[str, tuple[int, ...]] = {
    "world_tokens": (MAX_WORLD_TOKENS, TOKEN_FEAT_DIM),
    "world_token_mask": (MAX_WORLD_TOKENS,),
    "world_token_type_ids": (MAX_WORLD_TOKENS,),
    "world_token_role_ids": (MAX_WORLD_TOKENS,),
    "world_entity_owner_ids": (MAX_WORLD_TOKENS,),
    "world_token_entity_ids": (MAX_WORLD_TOKENS,),
    "world_token_zone_ids": (MAX_WORLD_TOKENS,),
    "world_token_order_ids": (MAX_WORLD_TOKENS,),
    "candidate_query_tokens": (MAX_ACTIONS, TOKEN_FEAT_DIM),
    "candidate_query_type_ids": (MAX_ACTIONS,),
    "candidate_query_role_ids": (MAX_ACTIONS,),
    "candidate_query_owner_ids": (MAX_ACTIONS,),
    "candidate_query_entity_ids": (MAX_ACTIONS,),
    "candidate_query_zone_ids": (MAX_ACTIONS,),
    "candidate_query_order_ids": (MAX_ACTIONS,),
    "candidate_query_target_owner_ids": (MAX_ACTIONS,),
    "candidate_query_target_entity_ids": (MAX_ACTIONS,),
    "candidate_local_tokens": (MAX_ACTIONS, MAX_CANDIDATE_LOCAL_TOKENS, TOKEN_FEAT_DIM),
    "candidate_local_masks": (MAX_ACTIONS, MAX_CANDIDATE_LOCAL_TOKENS),
    "candidate_local_type_ids": (MAX_ACTIONS, MAX_CANDIDATE_LOCAL_TOKENS),
    "candidate_local_role_ids": (MAX_ACTIONS, MAX_CANDIDATE_LOCAL_TOKENS),
    "candidate_local_owner_ids": (MAX_ACTIONS, MAX_CANDIDATE_LOCAL_TOKENS),
    "candidate_local_entity_ids": (MAX_ACTIONS, MAX_CANDIDATE_LOCAL_TOKENS),
    "candidate_local_zone_ids": (MAX_ACTIONS, MAX_CANDIDATE_LOCAL_TOKENS),
    "candidate_local_order_ids": (MAX_ACTIONS, MAX_CANDIDATE_LOCAL_TOKENS),
}

_OBS_FIXED_SHAPES: dict[str, tuple[int, ...]] = {
    **_DENSE_OBS_FIXED_SHAPES,
    **_TOKEN_OBS_FIXED_SHAPES,
}

_OBS_BINARY_KEYS = {
    "decision_domain",
    "hand_mask",
    "deck_mask",
    "enemy_mask",
    "relic_mask",
    "potion_mask",
    "route_node_mask",
    "action_mask",
    "world_token_mask",
    "candidate_local_masks",
}

_OBS_INTEGER_KEYS = {
    "world_token_type_ids",
    "world_token_role_ids",
    "world_entity_owner_ids",
    "world_token_entity_ids",
    "world_token_zone_ids",
    "world_token_order_ids",
    "candidate_query_type_ids",
    "candidate_query_role_ids",
    "candidate_query_owner_ids",
    "candidate_query_entity_ids",
    "candidate_query_zone_ids",
    "candidate_query_order_ids",
    "candidate_query_target_owner_ids",
    "candidate_query_target_entity_ids",
    "candidate_local_type_ids",
    "candidate_local_role_ids",
    "candidate_local_owner_ids",
    "candidate_local_entity_ids",
    "candidate_local_zone_ids",
    "candidate_local_order_ids",
}

_OBS_ROW_TRIM_MASKS = {
    "hand": "hand_mask",
    "hand_text": "hand_mask",
    "deck": "deck_mask",
    "deck_text": "deck_mask",
    "enemies": "enemy_mask",
    "enemy_text": "enemy_mask",
    "relics": "relic_mask",
    "potions": "potion_mask",
    "actions": "action_mask",
    "action_text": "action_mask",
    "semantic_actions": "action_mask",
    "semantic_action_text": "action_mask",
    "route_summary": "action_mask",
    "route_nodes": "action_mask",
    "route_node_mask": "action_mask",
    "world_tokens": "world_token_mask",
    "world_token_mask": "world_token_mask",
    "world_token_type_ids": "world_token_mask",
    "world_token_role_ids": "world_token_mask",
    "world_entity_owner_ids": "world_token_mask",
    "world_token_entity_ids": "world_token_mask",
    "world_token_zone_ids": "world_token_mask",
    "world_token_order_ids": "world_token_mask",
    "candidate_query_tokens": "action_mask",
    "candidate_query_type_ids": "action_mask",
    "candidate_query_role_ids": "action_mask",
    "candidate_query_owner_ids": "action_mask",
    "candidate_query_entity_ids": "action_mask",
    "candidate_query_zone_ids": "action_mask",
    "candidate_query_order_ids": "action_mask",
    "candidate_query_target_owner_ids": "action_mask",
    "candidate_query_target_entity_ids": "action_mask",
    "candidate_local_tokens": "action_mask",
    "candidate_local_masks": "action_mask",
    "candidate_local_type_ids": "action_mask",
    "candidate_local_role_ids": "action_mask",
    "candidate_local_owner_ids": "action_mask",
    "candidate_local_entity_ids": "action_mask",
    "candidate_local_zone_ids": "action_mask",
    "candidate_local_order_ids": "action_mask",
}


def _safe_text(value: Any) -> str:
    if value is None:
        return ""
    return str(value).strip()


def _safe_float_value(value: Any, default: float = 0.0) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return float(default)


def _step_action_info(step: Dict[str, Any]) -> Dict[str, Any]:
    info = step.get("action_info")
    return info if isinstance(info, dict) else {}


def _step_semantic_info(step: Dict[str, Any]) -> Dict[str, Any]:
    semantic = _step_action_info(step).get("semantic")
    return semantic if isinstance(semantic, dict) else {}


def _step_action_family(step: Dict[str, Any]) -> str:
    return (
        _safe_text(step.get("action_family"))
        or _safe_text(_step_semantic_info(step).get("family"))
    ).lower()


def _step_decision_domain(step: Dict[str, Any]) -> str:
    return (
        _safe_text(step.get("decision_domain"))
        or _safe_text(step.get("semantic_domain"))
        or _safe_text(_step_semantic_info(step).get("domain"))
    ).lower()


def _step_phase(step: Dict[str, Any]) -> str:
    return (
        _safe_text(step.get("phase"))
        or _safe_text(_step_action_info(step).get("phase"))
        or _safe_text(step.get("surface"))
        or _safe_text(_step_action_info(step).get("surface"))
    ).lower()


def _step_surface(step: Dict[str, Any]) -> str:
    return (
        _safe_text(step.get("surface"))
        or _safe_text(_step_action_info(step).get("surface"))
    ).lower()


def _is_boundary_replay_step(step: Dict[str, Any]) -> bool:
    family = _step_action_family(step)
    if family in _BUILD_ROUTE_ACTION_FAMILIES:
        return True
    if family in {"end_turn", "proceed"}:
        return False
    domain = _step_decision_domain(step)
    return domain in _BUILD_ROUTE_DECISION_DOMAINS


def _is_settlement_eligible_step(step: Dict[str, Any]) -> bool:
    return _step_action_family(step) in _BUILD_ROUTE_ACTION_FAMILIES


def _is_wasteful_proceed_step(step: Dict[str, Any]) -> bool:
    if bool(step.get("wasteful_proceed")):
        return True
    family = _step_action_family(step)
    phase = _step_phase(step)
    surface = _step_surface(step)
    selection = _safe_text(step.get("selection")).lower()
    if family == "proceed" and (phase in {"reward", "card_reward"} or surface in {"reward", "card_reward"}):
        return True
    return family in {"reward", "card_reward"} and selection in _WASTEFUL_SKIP_SELECTIONS


def _infer_encounter_tier_from_metadata(metadata: Dict[str, Any] | None) -> str:
    if not isinstance(metadata, dict):
        return ""
    tier = _safe_text(metadata.get("encounter_tier")).lower()
    if tier in _VALID_ENCOUNTER_TIERS:
        return tier

    encounter_id = _safe_text(metadata.get("encounter_id")).upper()
    if encounter_id.endswith("_WEAK"):
        return "weak"
    if encounter_id.endswith("_ELITE"):
        return "elite"
    if encounter_id.endswith("_BOSS"):
        return "boss"
    if encounter_id.endswith("_NORMAL") or encounter_id.endswith("_NORMAL_ALT"):
        return "normal"
    return ""


def _encounter_id_from_metadata(metadata: Dict[str, Any] | None) -> str:
    if not isinstance(metadata, dict):
        return ""
    return normalize_encounter_id(metadata.get("encounter_id"))


def _is_packed_observation(obs: Dict[str, Any] | None) -> bool:
    return isinstance(obs, dict) and bool(obs.get(_PACKED_OBS_MARKER))


def _count_active_rows(mask_value: Any, max_rows: int) -> int:
    if mask_value is None:
        return max_rows
    mask_arr = np.asarray(mask_value)
    if mask_arr.size == 0:
        return 0
    return max(0, min(int(np.count_nonzero(mask_arr > 0.5)), max_rows))


def _pack_observation(obs: Dict[str, Any]) -> Dict[str, Any]:
    packed: Dict[str, Any] = {_PACKED_OBS_MARKER: True}
    for key, shape in _OBS_FIXED_SHAPES.items():
        value = obs.get(key)
        if value is None:
            continue

        arr = np.asarray(value)
        if arr.shape != shape:
            try:
                arr = np.asarray(value, dtype=np.float32).reshape(shape)
            except Exception:
                arr = np.zeros(shape, dtype=np.float32)

        if key in _OBS_ROW_TRIM_MASKS:
            row_mask_key = _OBS_ROW_TRIM_MASKS[key]
            row_count = _count_active_rows(obs.get(row_mask_key), shape[0])
            arr = arr[:row_count]

        if key in _OBS_BINARY_KEYS:
            packed[key] = np.asarray(arr > 0.5, dtype=np.uint8)
        elif key in _OBS_INTEGER_KEYS:
            packed[key] = np.asarray(arr, dtype=np.uint16)
        else:
            packed[key] = np.asarray(arr, dtype=np.float16)

    return packed


def _batched_observations_to_numpy(obs_list: List[Dict[str, Any]]) -> Dict[str, np.ndarray]:
    if not obs_list:
        return {}

    batch_size = len(obs_list)
    batched = {
        key: np.zeros((batch_size, *shape), dtype=np.float32)
        for key, shape in _OBS_FIXED_SHAPES.items()
    }

    for batch_idx, obs in enumerate(obs_list):
        if _is_packed_observation(obs):
            for key, shape in _OBS_FIXED_SHAPES.items():
                stored = obs.get(key)
                if stored is None:
                    continue
                if key in _OBS_BINARY_KEYS:
                    data = np.asarray(stored, dtype=np.uint8)
                elif key in _OBS_INTEGER_KEYS:
                    data = np.asarray(stored, dtype=np.uint16)
                else:
                    data = np.asarray(stored, dtype=np.float32)

                if key in _OBS_ROW_TRIM_MASKS:
                    rows = min(int(data.shape[0]) if data.ndim >= 1 else 0, shape[0])
                    if rows > 0:
                        batched[key][batch_idx, :rows] = data[:rows].astype(np.float32, copy=False)
                else:
                    if data.shape != shape:
                        try:
                            data = data.reshape(shape)
                        except Exception:
                            continue
                    batched[key][batch_idx] = data.astype(np.float32, copy=False)
            continue

        for key, shape in _OBS_FIXED_SHAPES.items():
            value = obs.get(key) if isinstance(obs, dict) else None
            if value is None:
                continue
            data = np.asarray(value, dtype=np.float32)
            if data.shape != shape:
                try:
                    data = data.reshape(shape)
                except Exception:
                    continue
            batched[key][batch_idx] = data

    return batched


_ZERO_PACKED_OBSERVATION: Dict[str, Any] = {
    _PACKED_OBS_MARKER: True,
    **{
        key: (
            np.zeros(
                (0, *shape[1:]),
                dtype=np.uint8 if key in _OBS_BINARY_KEYS else (np.uint16 if key in _OBS_INTEGER_KEYS else np.float16),
            )
            if key in _OBS_ROW_TRIM_MASKS
            else np.zeros(
                shape,
                dtype=np.uint8 if key in _OBS_BINARY_KEYS else (np.uint16 if key in _OBS_INTEGER_KEYS else np.float16),
            )
        )
        for key, shape in _OBS_FIXED_SHAPES.items()
    },
}


@dataclass
class GameTrajectory:
    """Stores a single episode trajectory for MuZero training.

    Each step records the observation, action taken, reward received, action mask,
    search policy from MCTS, and root value estimate. Targets are computed on demand.
    """

    steps: List[Dict[str, Any]] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)
    """List of transition dicts with keys:
    - obs: dict observation
    - action: int action index [0, MAX_ACTIONS)
    - reward: float immediate reward
    - reward_components: np.ndarray [NUM_OBJECTIVE_HEADS]
    - action_mask: np.ndarray [MAX_ACTIONS] binary mask of legal actions
    - search_policy: np.ndarray [MAX_ACTIONS] action probs from MCTS
    - root_value: float value estimate at this step
    - root_value_components: np.ndarray [NUM_OBJECTIVE_HEADS]
    - semantic_action: int semantic rollout action index
    - semantic_policy: np.ndarray [SEMANTIC_ROLLOUT_SIZE] aggregated semantic policy
    - action_info: compact logical action signature for postmortem analysis
    - search_stats: compact MCTS summary for this decision
    - decision_diagnostics: compact pre-step legal-action / inventory facts for
      postmortem analysis.  This must stay small because trajectories can enter
      replay; large raw observations belong in JSONL diagnostics only.
    - room_type / encounter_id / encounter_tier / floor / act_id:
      decision-time progress metadata for tier-specific diagnostics
    """

    def __len__(self) -> int:
        """Return number of steps in trajectory."""
        return len(self.steps)

    def add_step(
        self,
        obs: Dict[str, Any],
        action: int,
        reward: float,
        reward_components: np.ndarray | list[float] | None,
        action_mask: np.ndarray,
        search_policy: np.ndarray,
        root_value: float,
        root_value_components: np.ndarray | list[float] | None = None,
        objective_context: np.ndarray | list[float] | None = None,
        semantic_action: int | None = None,
        semantic_policy: np.ndarray | list[float] | None = None,
        action_info: Dict[str, Any] | None = None,
        search_stats: Dict[str, Any] | None = None,
        decision_diagnostics: Dict[str, Any] | None = None,
        decision_domain: str | None = None,
        phase: str | None = None,
        action_family: str | None = None,
        semantic_domain: str | None = None,
        surface: str | None = None,
        selection: str | None = None,
        wasteful_end_turn: bool = False,
        wasteful_proceed: bool = False,
        room_type: str | None = None,
        encounter_id: str | None = None,
        encounter_tier: str | None = None,
        floor: float | int | str | None = None,
        act_id: float | int | str | None = None,
    ) -> None:
        """Add a step to the trajectory.

        Args:
            obs: Observation dict.
            action: Action index.
            reward: Immediate reward.
            reward_components: Objective-decomposed immediate reward.
            action_mask: [MAX_ACTIONS] binary mask.
            search_policy: [MAX_ACTIONS] MCTS policy.
            root_value: Value estimate from root of MCTS tree.
            root_value_components: Objective-decomposed root value estimate.
            objective_context: Planner objective-context vector at this decision.
            semantic_action: Chosen semantic rollout action index.
            semantic_policy: Aggregated semantic policy target.
            action_info: Compact action signature for debugging / replay analysis.
            search_stats: Compact search summary for this step.
            decision_diagnostics: Compact pre-step diagnostics for death slices.
        """
        reward_components_arr = np.zeros(NUM_OBJECTIVE_HEADS, dtype=np.float32)
        if reward_components is not None:
            reward_components_arr = np.asarray(reward_components, dtype=np.float32).reshape(-1)
            if reward_components_arr.shape[0] != NUM_OBJECTIVE_HEADS:
                reward_components_arr = np.resize(reward_components_arr, NUM_OBJECTIVE_HEADS).astype(np.float32, copy=False)

        root_value_components_arr = np.zeros(NUM_OBJECTIVE_HEADS, dtype=np.float32)
        if root_value_components is not None:
            root_value_components_arr = np.asarray(root_value_components, dtype=np.float32).reshape(-1)
            if root_value_components_arr.shape[0] != NUM_OBJECTIVE_HEADS:
                root_value_components_arr = np.resize(root_value_components_arr, NUM_OBJECTIVE_HEADS).astype(np.float32, copy=False)

        objective_context_arr = np.asarray(objective_context, dtype=np.float32).reshape(-1) if objective_context is not None else np.zeros(0, dtype=np.float32)
        semantic_policy_arr = np.zeros(SEMANTIC_ROLLOUT_SIZE, dtype=np.float32)
        if semantic_policy is not None:
            semantic_policy_arr = np.asarray(semantic_policy, dtype=np.float32).reshape(-1)
            if semantic_policy_arr.shape[0] != SEMANTIC_ROLLOUT_SIZE:
                semantic_policy_arr = np.resize(semantic_policy_arr, SEMANTIC_ROLLOUT_SIZE).astype(np.float32, copy=False)
        self.steps.append({
            "obs": _pack_observation(obs),
            "action": int(action),
            "reward": float(reward),
            "reward_components": reward_components_arr.astype(np.float32, copy=False),
            "action_mask": np.array(action_mask, dtype=np.float32),
            "search_policy": np.array(search_policy, dtype=np.float32),
            "root_value": float(root_value),
            "root_value_components": root_value_components_arr.astype(np.float32, copy=False),
            "objective_context": objective_context_arr.astype(np.float32, copy=False),
            "semantic_action": int(semantic_action or 0),
            "semantic_policy": semantic_policy_arr.astype(np.float32, copy=False),
            "action_info": dict(action_info or {}),
            "search_stats": dict(search_stats or {}),
            "decision_diagnostics": dict(decision_diagnostics or {}),
            "decision_domain": _safe_text(decision_domain).lower(),
            "phase": _safe_text(phase).lower(),
            "action_family": _safe_text(action_family).lower(),
            "semantic_domain": _safe_text(semantic_domain).lower(),
            "surface": _safe_text(surface).lower(),
            "selection": _safe_text(selection).lower(),
            "wasteful_end_turn": bool(wasteful_end_turn),
            "wasteful_proceed": bool(wasteful_proceed),
            "room_type": _safe_text(room_type).lower(),
            "encounter_id": _safe_text(encounter_id),
            "encounter_tier": _safe_text(encounter_tier).lower(),
            "floor": _safe_float_value(floor),
            "act_id": _safe_float_value(act_id),
            "settlement_bonus": 0.0,
        })

    def apply_light_episode_settlement(
        self,
        settlement_signal: float,
        *,
        settlement_weight: float = 0.12,
        decay: float = 0.99,
        max_steps: int = 12,
    ) -> Dict[str, float]:
        """Backfill a light episode outcome signal onto recent build / route steps only."""
        applied_steps = 0
        total_bonus = 0.0
        settlement_signal = float(np.clip(settlement_signal, -1.0, 1.0))
        settlement_weight = max(float(settlement_weight), 0.0)
        decay = float(np.clip(decay, 0.0, 1.0))
        max_steps = max(int(max_steps), 0)

        eligible_indices = [
            index for index, step in enumerate(self.steps)
            if _is_settlement_eligible_step(step)
        ]

        if settlement_signal == 0.0 or settlement_weight <= 0.0 or max_steps <= 0 or not eligible_indices:
            self.metadata.update({
                "settlement_signal": settlement_signal,
                "settlement_eligible_steps": int(len(eligible_indices)),
                "settlement_applied_steps": 0,
                "settlement_total_bonus": 0.0,
            })
            return {
                "signal": settlement_signal,
                "eligible_steps": float(len(eligible_indices)),
                "applied_steps": 0.0,
                "total_bonus": 0.0,
            }

        recent_indices = eligible_indices[-max_steps:]
        positive_mix = np.zeros(NUM_OBJECTIVE_HEADS, dtype=np.float32)
        positive_mix[HEAD_SURVIVAL] = 0.20
        positive_mix[HEAD_HP_PRESERVATION] = 0.05
        positive_mix[HEAD_BUILD_PROGRESS] = 0.60
        positive_mix[HEAD_RESOURCE_EFFICIENCY] = 0.15

        negative_mix = np.zeros(NUM_OBJECTIVE_HEADS, dtype=np.float32)
        negative_mix[HEAD_SURVIVAL] = 0.40
        negative_mix[HEAD_HP_PRESERVATION] = 0.10
        negative_mix[HEAD_BUILD_PROGRESS] = 0.35
        negative_mix[HEAD_RESOURCE_EFFICIENCY] = 0.15

        for reverse_offset, step_idx in enumerate(reversed(recent_indices)):
            step_bonus = settlement_signal * settlement_weight * (decay ** reverse_offset)
            if abs(step_bonus) <= 1e-8:
                continue
            step = self.steps[step_idx]
            component_mix = positive_mix if step_bonus >= 0.0 else negative_mix
            step["reward"] = float(step.get("reward", 0.0)) + float(step_bonus)
            reward_components = np.asarray(
                step.get("reward_components", np.zeros(NUM_OBJECTIVE_HEADS, dtype=np.float32)),
                dtype=np.float32,
            ).reshape(-1)
            if reward_components.shape[0] != NUM_OBJECTIVE_HEADS:
                reward_components = np.resize(reward_components, NUM_OBJECTIVE_HEADS).astype(np.float32, copy=False)
            reward_components = reward_components + component_mix * float(step_bonus)
            step["reward_components"] = reward_components.astype(np.float32, copy=False)
            step["settlement_bonus"] = float(step.get("settlement_bonus", 0.0)) + float(step_bonus)
            applied_steps += 1
            total_bonus += float(step_bonus)

        self.metadata.update({
            "settlement_signal": settlement_signal,
            "settlement_eligible_steps": int(len(eligible_indices)),
            "settlement_applied_steps": int(applied_steps),
            "settlement_total_bonus": float(total_bonus),
        })
        return {
            "signal": settlement_signal,
            "eligible_steps": float(len(eligible_indices)),
            "applied_steps": float(applied_steps),
            "total_bonus": float(total_bonus),
        }

    def compute_target_values(
        self,
        discount: float = 0.997,
        n_steps: int = 10,
    ) -> List[float]:
        """Compute n-step bootstrapped returns for each step.

        For step t, the n-step return is:
            G_t = sum_{k=0}^{n-1} gamma^k * r_{t+k} + gamma^n * V_{t+n}

        Where V_t is the root_value at step t. Past episode end, value is treated as 0.

        Args:
            discount: Discount factor gamma.
            n_steps: Number of steps to bootstrap.

        Returns:
            List of n-step target values, one per step.
        """
        targets = []
        ep_len = len(self.steps)

        for step_idx in range(ep_len):
            target = 0.0
            discounted = 1.0

            # Sum discounted rewards over n steps
            for k in range(n_steps):
                if step_idx + k < ep_len:
                    target += discounted * self.steps[step_idx + k]["reward"]
                    discounted *= discount
                else:
                    break

            # Bootstrap value at step_idx + n_steps
            if step_idx + n_steps < ep_len:
                bootstrap_value = self.steps[step_idx + n_steps]["root_value"]
                target += discounted * bootstrap_value

            targets.append(target)

        return targets

    def compute_target_value_components(
        self,
        discount: float = 0.997,
        n_steps: int = 10,
    ) -> List[np.ndarray]:
        """Compute n-step bootstrapped targets for each objective head."""
        targets: List[np.ndarray] = []
        ep_len = len(self.steps)

        for step_idx in range(ep_len):
            target = np.zeros(NUM_OBJECTIVE_HEADS, dtype=np.float32)
            discounted = 1.0

            for k in range(n_steps):
                if step_idx + k < ep_len:
                    target += discounted * np.asarray(
                        self.steps[step_idx + k].get("reward_components", np.zeros(NUM_OBJECTIVE_HEADS, dtype=np.float32)),
                        dtype=np.float32,
                    )
                    discounted *= discount
                else:
                    break

            if step_idx + n_steps < ep_len:
                bootstrap_value = np.asarray(
                    self.steps[step_idx + n_steps].get("root_value_components", np.zeros(NUM_OBJECTIVE_HEADS, dtype=np.float32)),
                    dtype=np.float32,
                )
                target += discounted * bootstrap_value

            targets.append(target.astype(np.float32, copy=False))

        return targets


class MuZeroReplayBuffer:
    """Prioritized replay buffer for MuZero training.

    Stores complete game trajectories and samples mini-batches for training.
    Prioritizes sampling based on absolute difference between predicted and
    bootstrapped values.
    """

    def __init__(
        self,
        capacity: int = 100_000,
        *,
        boundary_build_bonus: float = 1.0,
        boundary_route_bonus: float = 1.5,
        boundary_family_bonus: float = 0.75,
        trajectory_quality_bonus: float = 0.25,
        wasteful_end_turn_scale: float = 0.35,
        wasteful_proceed_scale: float = 0.55,
        encounter_tier_weights: Dict[str, float] | None = None,
        encounter_priority_weights: Dict[str, float] | None = None,
        tier_quota_config: Any = None,
    ):
        """Initialize replay buffer.

        Args:
            capacity: Maximum total transitions to store.
            tier_quota_config: Optional ``TierQuotaConfig`` (see
                ``muzero.replay_scheduler``) that activates the P0-2 hard
                batch-level quota path. When ``None``, sample_batch falls
                back to the legacy soft-weight path.
        """
        self.capacity = max(int(capacity), 1)
        self.trajectories: List[GameTrajectory] = []
        self.priorities: List[float] = []
        self._total_transitions = 0
        self.boundary_build_bonus = max(float(boundary_build_bonus), 0.0)
        self.boundary_route_bonus = max(float(boundary_route_bonus), 0.0)
        self.boundary_family_bonus = max(float(boundary_family_bonus), 0.0)
        self.trajectory_quality_bonus = max(float(trajectory_quality_bonus), 0.0)
        self.wasteful_end_turn_scale = float(np.clip(wasteful_end_turn_scale, 0.05, 1.0))
        self.wasteful_proceed_scale = float(np.clip(wasteful_proceed_scale, 0.05, 1.0))
        self.encounter_tier_weights = {
            str(tier).strip().lower(): max(float(weight), 0.0)
            for tier, weight in (encounter_tier_weights or {}).items()
            if str(tier).strip().lower() in _VALID_ENCOUNTER_TIERS and float(weight) >= 0.0
        }
        self.encounter_priority_weights = {
            normalize_encounter_id(encounter_id): max(float(weight), 0.0)
            for encounter_id, weight in (encounter_priority_weights or {}).items()
            if normalize_encounter_id(encounter_id) and float(weight) >= 0.0
        }
        # P0-2 (recovery 2026-05-06): hard per-batch tier quotas. When set,
        # ``sample_batch`` allocates fixed slot counts per tier and only
        # falls back to the global priority pool for "flex" slots that no
        # tier could fill.
        self.tier_quota_config = tier_quota_config

    def _trajectory_priority_multiplier(self, metadata: Dict[str, Any] | None) -> float:
        if not isinstance(metadata, dict) or self.trajectory_quality_bonus <= 0.0:
            return 1.0
        max_floor = max(float(metadata.get("max_floor", 0.0) or 0.0), 0.0)
        elite_rooms_seen = max(float(metadata.get("elite_rooms_seen", 0.0) or 0.0), 0.0)
        quality = 0.0
        quality += 0.30 * min(max_floor / 16.0, 1.0)
        quality += 0.15 * min(elite_rooms_seen / 2.0, 1.0)
        quality += 0.20 if bool(metadata.get("act1_boss_seen")) else 0.0
        quality += 0.35 if bool(metadata.get("act1_clear")) else 0.0
        return 1.0 + self.trajectory_quality_bonus * float(np.clip(quality, 0.0, 1.0))

    def _trajectory_sampling_info(self, metadata: Dict[str, Any] | None) -> Dict[str, Any]:
        encounter_id = _encounter_id_from_metadata(metadata)
        encounter_tier = _infer_encounter_tier_from_metadata(metadata)
        tier_weight = float(self.encounter_tier_weights.get(encounter_tier, 1.0))
        encounter_weight = float(self.encounter_priority_weights.get(encounter_id, 1.0))
        sampling_multiplier = max(tier_weight * encounter_weight, 1e-6)
        return {
            "encounter_id": encounter_id,
            "encounter_tier": encounter_tier,
            "tier_weight": tier_weight,
            "encounter_weight": encounter_weight,
            "sampling_multiplier": sampling_multiplier,
            "hard_encounter": encounter_weight > 1.0 + 1e-6,
            "hard_normal": encounter_weight > 1.0 + 1e-6 and encounter_tier == "normal",
            "hard_elite": encounter_weight > 1.0 + 1e-6 and encounter_tier == "elite",
        }

    def _position_weight(self, step: Dict[str, Any]) -> float:
        family = _step_action_family(step)
        domain = _step_decision_domain(step)
        weight = 1.0
        if domain == "build":
            weight += self.boundary_build_bonus
        elif domain == "route":
            weight += self.boundary_route_bonus
        if family in _BUILD_ROUTE_ACTION_FAMILIES:
            weight += self.boundary_family_bonus
        settlement_bonus = abs(float(step.get("settlement_bonus", 0.0) or 0.0))
        if settlement_bonus > 1e-6:
            weight += min(settlement_bonus * 8.0, 1.0)

        if bool(step.get("wasteful_end_turn")):
            return max(0.05, min(weight, 1.0) * self.wasteful_end_turn_scale)
        if _is_wasteful_proceed_step(step):
            return max(0.05, min(weight, 1.0) * self.wasteful_proceed_scale)
        return max(weight, 0.05)

    def _trajectory_position_weights(self, trajectory: GameTrajectory) -> np.ndarray:
        weights = np.asarray(
            [self._position_weight(step) for step in trajectory.steps],
            dtype=np.float32,
        )
        if weights.size == 0 or float(weights.sum()) <= 1e-9:
            return np.ones((max(len(trajectory), 1),), dtype=np.float32)

        sampling_info = self._trajectory_sampling_info(getattr(trajectory, "metadata", None))
        if sampling_info.get("encounter_tier") == "boss":
            # Bosses are not failing because whole trajectories are unseen; they are
            # failing because decisive turns (lethal windows, low-hp survival turns,
            # potion turns, final death/win turns) are diluted across long fights.
            # Keep trajectory-level boss oversampling, but also reshape *position*
            # sampling toward the states that teach the planner what changes the
            # outcome.
            n = int(weights.size)
            tail = min(max(4, n // 4), n)
            if tail > 0:
                ramp = np.linspace(1.35, 2.75, tail, dtype=np.float32)
                weights[-tail:] *= ramp
            for idx, step in enumerate(trajectory.steps):
                if not isinstance(step, dict):
                    continue
                family = _step_action_family(step)
                reward_abs = abs(float(step.get("reward", 0.0) or 0.0))
                if family in {"use_potion", "discard_potion"}:
                    weights[idx] *= 2.25
                if bool(step.get("wasteful_end_turn")):
                    weights[idx] *= 0.35
                if reward_abs >= 1.0:
                    weights[idx] *= 1.0 + min(reward_abs / 6.0, 2.0)
                search_stats = step.get("search_stats") if isinstance(step.get("search_stats"), dict) else {}
                try:
                    objective_q = float(search_stats.get("direct_rollout_objective_q_mean", 0.0) or 0.0)
                    risk_q = float(search_stats.get("direct_rollout_risk_q_mean", 0.0) or 0.0)
                    uncertainty = float(search_stats.get("direct_rollout_uncertainty_mean", 0.0) or 0.0)
                except (TypeError, ValueError):
                    objective_q, risk_q, uncertainty = 0.0, 0.0, 0.0
                if objective_q < -0.25 or risk_q < -0.25:
                    weights[idx] *= 1.35
                if uncertainty > 20.0:
                    weights[idx] *= 1.20
            weights = np.clip(weights, 0.05, 12.0).astype(np.float32, copy=False)

        if weights.size == 0 or float(weights.sum()) <= 1e-9:
            return np.ones((max(len(trajectory), 1),), dtype=np.float32)
        return weights

    def save_episode(
        self,
        trajectory: GameTrajectory,
        *,
        discount: float = 0.997,
        n_steps: int = 10,
    ) -> None:
        """Save a completed episode trajectory to the buffer.

        If buffer exceeds capacity (by number of transitions), remove oldest
        trajectories until within capacity.

        Args:
            trajectory: Completed GameTrajectory with all steps.
        """
        if len(trajectory) == 0:
            return

        scalar_targets = trajectory.compute_target_values(discount=discount, n_steps=n_steps)
        component_targets = trajectory.compute_target_value_components(discount=discount, n_steps=n_steps)

        # Compute priorities: use deviation between predicted root value(s) and bootstrapped target(s)
        ep_priority = 0.0
        for step_index, step in enumerate(trajectory.steps):
            scalar_target = scalar_targets[step_index] if step_index < len(scalar_targets) else 0.0
            component_target = component_targets[step_index] if step_index < len(component_targets) else np.zeros(NUM_OBJECTIVE_HEADS, dtype=np.float32)
            component_pred = np.asarray(step.get("root_value_components", np.zeros(NUM_OBJECTIVE_HEADS, dtype=np.float32)), dtype=np.float32)
            objective_context = step.get("objective_context")
            scalarized_component_target = float(scalarize_objective_components_np(component_target, objective_context))
            scalarized_component_pred = float(scalarize_objective_components_np(component_pred, objective_context))
            delta = abs(float(step["root_value"]) - float(scalar_target))
            delta += 0.5 * abs(scalarized_component_pred - scalarized_component_target)
            ep_priority += delta
        ep_priority /= max(len(trajectory), 1)
        ep_priority *= self._trajectory_priority_multiplier(getattr(trajectory, "metadata", None))
        ep_priority += 1e-6  # Small epsilon to avoid zero priority
        self.trajectories.append(trajectory)
        self.priorities.append(ep_priority)
        self._total_transitions += len(trajectory)

        # Evict oldest trajectories if over capacity
        while self._total_transitions > self.capacity and len(self.trajectories) > 0:
            removed_len = len(self.trajectories[0])
            self.trajectories.pop(0)
            self.priorities.pop(0)
            self._total_transitions -= removed_len

    @staticmethod
    def _zero_obs_like(obs: Dict[str, Any]) -> Dict[str, Any]:
        """Create a zero observation with the same schema/shapes as `obs`."""
        if _is_packed_observation(obs):
            return _ZERO_PACKED_OBSERVATION
        zero_obs: Dict[str, Any] = {}
        for key, value in obs.items():
            if isinstance(value, np.ndarray):
                zero_obs[key] = np.zeros_like(value)
            elif isinstance(value, torch.Tensor):
                zero_obs[key] = torch.zeros_like(value)
            else:
                zero_obs[key] = 0
        return zero_obs

    @staticmethod
    def batch_observations(obs_list: List[Dict[str, Any]]) -> Dict[str, np.ndarray]:
        """Materialize a batch of observations into dense numpy arrays."""
        return _batched_observations_to_numpy(obs_list)

    def sample_batch(
        self,
        batch_size: int,
        unroll_steps: int = 5,
        rng: np.random.Generator | None = None,
        discount: float = 0.997,
        n_step_return: int = 10,
    ) -> Dict[str, Any]:
        """Sample a mini-batch for training.

        Samples batch_size (trajectory, position) pairs using prioritized sampling.
        For each sample, unrolls K steps forward for computing unrolled losses.

        Returns tensors with batch dimension B = batch_size:
        - obs_batch: list of B observation dicts
        - action_batch: [B, K] int actions for unroll
        - reward_target: [B, K] float rewards
        - reward_component_target: [B, K, NUM_OBJECTIVE_HEADS] decomposed rewards
        - value_target: [B, K+1] float bootstrapped values
        - value_component_target: [B, K+1, NUM_OBJECTIVE_HEADS] bootstrapped decomposed values
        - policy_target: [B, K+1, 80] MCTS policy distributions
        - semantic_action_batch: [B, K] semantic rollout action ids
        - semantic_policy_target: [B, K+1, SEMANTIC_ROLLOUT_SIZE] aggregated semantic policy distributions
        - action_mask_batch: [B, K+1, 80] legal action masks

        Past episode end, all targets are zero-padded.

        Args:
            batch_size: Number of samples.
            unroll_steps: Number of steps to unroll.
            rng: Random generator for sampling.

        Returns:
            Dict with tensors for training.
        """
        if len(self.trajectories) == 0:
            raise RuntimeError("Cannot sample from empty replay buffer")

        if rng is None:
            rng = np.random.default_rng()

        # Prioritized sampling: sample trajectory, then random position
        priorities_array = np.asarray(self.priorities, dtype=np.float32)
        sampling_multipliers = np.asarray(
            [
                self._trajectory_sampling_info(getattr(trajectory, "metadata", None))["sampling_multiplier"]
                for trajectory in self.trajectories
            ],
            dtype=np.float32,
        )
        priorities_array = priorities_array * sampling_multipliers
        total_priority = float(priorities_array.sum())
        if not np.isfinite(total_priority) or total_priority <= 1e-9:
            priorities_array = np.full((len(self.trajectories),), 1.0 / float(len(self.trajectories)), dtype=np.float32)
        else:
            priorities_array /= total_priority

        batch_obs_sequence = []
        batch_actions = []
        batch_semantic_actions = []
        batch_rewards = []
        batch_reward_components = []
        batch_values = []
        batch_value_components = []
        batch_policies = []
        batch_semantic_policies = []
        batch_masks = []
        batch_position_weights = []
        batch_boundary_flags = []
        batch_build_route_flags = []
        batch_wasteful_flags = []
        batch_settlement_abs = []
        batch_sample_weak_flags = []
        batch_sample_normal_flags = []
        batch_sample_elite_flags = []
        batch_sample_boss_flags = []
        batch_sample_hard_encounter_flags = []
        batch_sample_hard_normal_flags = []
        batch_sample_hard_elite_flags = []
        batch_sample_tier_weight = []
        batch_sample_encounter_weight = []
        batch_sample_sampling_scale = []

        # P0-2: when a TierQuotaConfig is set, group trajectories by tier
        # once and decide a per-slot tier plan up front. The legacy soft
        # path (``slot_tier_plan is None``) is preserved for backward
        # compatibility and for any caller that has not yet opted in.
        slot_tier_plan: list[str] | None = None
        tier_indices: dict[str, list[int]] = {}
        tier_priority_arrays: dict[str, np.ndarray] = {}
        quota_info: dict[str, Any] | None = None
        if self.tier_quota_config is not None:
            from muzero.replay_scheduler import allocate_tier_quotas  # local import avoids cycle

            for idx, trajectory in enumerate(self.trajectories):
                info = self._trajectory_sampling_info(getattr(trajectory, "metadata", None))
                t = str(info.get("encounter_tier") or "normal").strip().lower()
                if t not in {"boss", "elite", "normal", "weak"}:
                    t = "normal"
                tier_indices.setdefault(t, []).append(idx)
            for t, idxs in tier_indices.items():
                pri_slice = np.asarray([priorities_array[i] for i in idxs], dtype=np.float64)
                pri_sum = float(pri_slice.sum())
                if not np.isfinite(pri_sum) or pri_sum <= 1e-9:
                    pri_slice = np.full((len(idxs),), 1.0 / max(1, len(idxs)), dtype=np.float64)
                else:
                    pri_slice = pri_slice / pri_sum
                tier_priority_arrays[t] = pri_slice
            pool_sizes = {t: len(tier_indices.get(t, [])) for t in ("boss", "elite", "normal", "weak")}
            allocation = allocate_tier_quotas(
                batch_size=batch_size,
                pool_sizes=pool_sizes,
                config=self.tier_quota_config,
            )
            slot_tier_plan = []
            for tier, count in allocation.quotas.items():
                slot_tier_plan.extend([tier] * int(count))
            # Stochastic order so position-related batch-level statistics
            # (e.g. boundary rate) do not become tier-correlated.
            rng.shuffle(slot_tier_plan)
            quota_info = {
                "quotas": dict(allocation.quotas),
                "target_counts": dict(allocation.target_counts),
                "boss_cap_hit": bool(allocation.boss_cap_hit),
                "normal_min_unfilled": bool(allocation.normal_min_unfilled),
                "elite_min_unfilled": bool(allocation.elite_min_unfilled),
                "fallback_count": int(allocation.quotas.get("flex", 0)),
                "pool_sizes": dict(pool_sizes),
            }

        for slot in range(batch_size):
            # Sample trajectory by priority, optionally restricted to a
            # tier pool when the P0-2 quota plan is active.
            if slot_tier_plan is not None:
                tier = slot_tier_plan[slot]
                if tier == "flex" or not tier_indices.get(tier):
                    traj_idx = int(rng.choice(len(self.trajectories), p=priorities_array))
                else:
                    local = int(rng.choice(len(tier_indices[tier]), p=tier_priority_arrays[tier]))
                    traj_idx = int(tier_indices[tier][local])
            else:
                traj_idx = int(rng.choice(len(self.trajectories), p=priorities_array))
            trajectory = self.trajectories[traj_idx]
            sampling_info = self._trajectory_sampling_info(getattr(trajectory, "metadata", None))
            scalar_targets = trajectory.compute_target_values(discount=discount, n_steps=n_step_return)
            component_targets = trajectory.compute_target_value_components(discount=discount, n_steps=n_step_return)

            # Sample position with boundary-weighted replay emphasis.
            position_weights = self._trajectory_position_weights(trajectory)
            position_probs = position_weights / max(float(position_weights.sum()), 1e-9)
            pos = int(rng.choice(len(trajectory), p=position_probs))
            sampled_step = trajectory.steps[pos]

            zero_obs = _ZERO_PACKED_OBSERVATION
            obs_sequence = []
            for k in range(unroll_steps + 1):
                step_idx = pos + k
                if step_idx < len(trajectory):
                    obs_sequence.append(trajectory.steps[step_idx]["obs"])
                else:
                    obs_sequence.append(zero_obs)
            batch_obs_sequence.append(obs_sequence)

            # Collect actions, rewards, values, policies for unroll
            actions = []
            semantic_actions = []
            rewards = []
            reward_components = []
            values = [scalar_targets[pos] if pos < len(scalar_targets) else trajectory.steps[pos]["root_value"]]
            value_components = [
                component_targets[pos] if pos < len(component_targets)
                else np.asarray(trajectory.steps[pos].get("root_value_components", np.zeros(NUM_OBJECTIVE_HEADS, dtype=np.float32)), dtype=np.float32)
            ]
            policies = [trajectory.steps[pos]["search_policy"]]
            semantic_policies = [
                np.asarray(
                    trajectory.steps[pos].get("semantic_policy", np.zeros(SEMANTIC_ROLLOUT_SIZE, dtype=np.float32)),
                    dtype=np.float32,
                )
            ]
            masks = [trajectory.steps[pos]["action_mask"]]

            for k in range(unroll_steps):
                step_idx = pos + k
                if step_idx < len(trajectory):
                    step = trajectory.steps[step_idx]
                    actions.append(step["action"])
                    semantic_actions.append(int(step.get("semantic_action", 0)))
                    rewards.append(step["reward"])
                    reward_components.append(
                        np.asarray(
                            step.get("reward_components", np.zeros(NUM_OBJECTIVE_HEADS, dtype=np.float32)),
                            dtype=np.float32,
                        )
                    )
                    if step_idx + 1 < len(trajectory):
                        next_target_idx = step_idx + 1
                        values.append(
                            scalar_targets[next_target_idx]
                            if next_target_idx < len(scalar_targets)
                            else trajectory.steps[step_idx + 1]["root_value"]
                        )
                        value_components.append(
                            component_targets[next_target_idx]
                            if next_target_idx < len(component_targets)
                            else np.asarray(
                                trajectory.steps[step_idx + 1].get(
                                    "root_value_components",
                                    np.zeros(NUM_OBJECTIVE_HEADS, dtype=np.float32),
                                ),
                                dtype=np.float32,
                            )
                        )
                        policies.append(trajectory.steps[step_idx + 1]["search_policy"])
                        semantic_policies.append(
                            np.asarray(
                                trajectory.steps[step_idx + 1].get(
                                    "semantic_policy",
                                    np.zeros(SEMANTIC_ROLLOUT_SIZE, dtype=np.float32),
                                ),
                                dtype=np.float32,
                            )
                        )
                        masks.append(trajectory.steps[step_idx + 1]["action_mask"])
                    else:
                        # Terminal state
                        values.append(0.0)
                        value_components.append(np.zeros(NUM_OBJECTIVE_HEADS, dtype=np.float32))
                        policies.append(np.zeros(len(step["search_policy"]), dtype=np.float32))
                        semantic_policies.append(np.zeros(SEMANTIC_ROLLOUT_SIZE, dtype=np.float32))
                        masks.append(np.zeros(len(step["action_mask"]), dtype=np.float32))
                else:
                    # Past episode end
                    actions.append(0)
                    semantic_actions.append(0)
                    rewards.append(0.0)
                    reward_components.append(np.zeros(NUM_OBJECTIVE_HEADS, dtype=np.float32))
                    values.append(0.0)
                    value_components.append(np.zeros(NUM_OBJECTIVE_HEADS, dtype=np.float32))
                    policies.append(np.zeros(len(trajectory.steps[-1]["search_policy"]), dtype=np.float32))
                    semantic_policies.append(np.zeros(SEMANTIC_ROLLOUT_SIZE, dtype=np.float32))
                    masks.append(np.zeros(len(trajectory.steps[-1]["action_mask"]), dtype=np.float32))

            batch_actions.append(actions)
            batch_semantic_actions.append(semantic_actions)
            batch_rewards.append(rewards)
            batch_reward_components.append(reward_components)
            batch_values.append(values)
            batch_value_components.append(value_components)
            batch_policies.append(policies)
            batch_semantic_policies.append(semantic_policies)
            batch_masks.append(masks)
            batch_position_weights.append(float(position_weights[pos]))
            batch_boundary_flags.append(1.0 if _is_boundary_replay_step(sampled_step) else 0.0)
            batch_build_route_flags.append(1.0 if _step_decision_domain(sampled_step) in _BUILD_ROUTE_DECISION_DOMAINS else 0.0)
            batch_wasteful_flags.append(
                1.0
                if bool(sampled_step.get("wasteful_end_turn")) or _is_wasteful_proceed_step(sampled_step)
                else 0.0
            )
            batch_settlement_abs.append(abs(float(sampled_step.get("settlement_bonus", 0.0) or 0.0)))
            batch_sample_weak_flags.append(1.0 if sampling_info["encounter_tier"] == "weak" else 0.0)
            batch_sample_normal_flags.append(1.0 if sampling_info["encounter_tier"] == "normal" else 0.0)
            batch_sample_elite_flags.append(1.0 if sampling_info["encounter_tier"] == "elite" else 0.0)
            batch_sample_boss_flags.append(1.0 if sampling_info["encounter_tier"] == "boss" else 0.0)
            batch_sample_hard_encounter_flags.append(1.0 if sampling_info["hard_encounter"] else 0.0)
            batch_sample_hard_normal_flags.append(1.0 if sampling_info["hard_normal"] else 0.0)
            batch_sample_hard_elite_flags.append(1.0 if sampling_info["hard_elite"] else 0.0)
            batch_sample_tier_weight.append(float(sampling_info["tier_weight"]))
            batch_sample_encounter_weight.append(float(sampling_info["encounter_weight"]))
            batch_sample_sampling_scale.append(float(sampling_info["sampling_multiplier"]))

        # Convert dense numeric targets through numpy first to avoid slow tensor construction
        action_batch = torch.from_numpy(np.asarray(batch_actions, dtype=np.int64))  # [B, K]
        semantic_action_batch = torch.from_numpy(np.asarray(batch_semantic_actions, dtype=np.int64))  # [B, K]
        reward_batch = torch.from_numpy(np.asarray(batch_rewards, dtype=np.float32))  # [B, K]
        reward_component_batch = torch.from_numpy(np.asarray(batch_reward_components, dtype=np.float32))  # [B, K, H]
        value_batch = torch.from_numpy(np.asarray(batch_values, dtype=np.float32))  # [B, K+1]
        value_component_batch = torch.from_numpy(np.asarray(batch_value_components, dtype=np.float32))  # [B, K+1, H]
        policy_batch = torch.from_numpy(np.asarray(batch_policies, dtype=np.float32))  # [B, K+1, 80]
        semantic_policy_batch = torch.from_numpy(np.asarray(batch_semantic_policies, dtype=np.float32))  # [B, K+1, S]
        mask_batch = torch.from_numpy(np.asarray(batch_masks, dtype=np.float32))  # [B, K+1, 80]
        position_weight_batch = torch.from_numpy(np.asarray(batch_position_weights, dtype=np.float32))
        boundary_flag_batch = torch.from_numpy(np.asarray(batch_boundary_flags, dtype=np.float32))
        build_route_flag_batch = torch.from_numpy(np.asarray(batch_build_route_flags, dtype=np.float32))
        wasteful_flag_batch = torch.from_numpy(np.asarray(batch_wasteful_flags, dtype=np.float32))
        settlement_abs_batch = torch.from_numpy(np.asarray(batch_settlement_abs, dtype=np.float32))
        sample_weak_flag_batch = torch.from_numpy(np.asarray(batch_sample_weak_flags, dtype=np.float32))
        sample_normal_flag_batch = torch.from_numpy(np.asarray(batch_sample_normal_flags, dtype=np.float32))
        sample_elite_flag_batch = torch.from_numpy(np.asarray(batch_sample_elite_flags, dtype=np.float32))
        sample_boss_flag_batch = torch.from_numpy(np.asarray(batch_sample_boss_flags, dtype=np.float32))
        sample_hard_encounter_flag_batch = torch.from_numpy(np.asarray(batch_sample_hard_encounter_flags, dtype=np.float32))
        sample_hard_normal_flag_batch = torch.from_numpy(np.asarray(batch_sample_hard_normal_flags, dtype=np.float32))
        sample_hard_elite_flag_batch = torch.from_numpy(np.asarray(batch_sample_hard_elite_flags, dtype=np.float32))
        sample_tier_weight_batch = torch.from_numpy(np.asarray(batch_sample_tier_weight, dtype=np.float32))
        sample_encounter_weight_batch = torch.from_numpy(np.asarray(batch_sample_encounter_weight, dtype=np.float32))
        sample_sampling_scale_batch = torch.from_numpy(np.asarray(batch_sample_sampling_scale, dtype=np.float32))

        return {
            "obs_sequence_batch": batch_obs_sequence,
            "action_batch": action_batch,
            "semantic_action_batch": semantic_action_batch,
            "reward_target": reward_batch,
            "reward_component_target": reward_component_batch,
            "value_target": value_batch,
            "value_component_target": value_component_batch,
            "policy_target": policy_batch,
            "semantic_policy_target": semantic_policy_batch,
            "action_mask_batch": mask_batch,
            "position_replay_weight": position_weight_batch,
            "sample_boundary_flag": boundary_flag_batch,
            "sample_build_route_flag": build_route_flag_batch,
            "sample_wasteful_flag": wasteful_flag_batch,
            "sample_settlement_abs": settlement_abs_batch,
            "sample_weak_flag": sample_weak_flag_batch,
            "sample_normal_flag": sample_normal_flag_batch,
            "sample_elite_flag": sample_elite_flag_batch,
            "sample_boss_flag": sample_boss_flag_batch,
            "sample_hard_encounter_flag": sample_hard_encounter_flag_batch,
            "sample_hard_normal_flag": sample_hard_normal_flag_batch,
            "sample_hard_elite_flag": sample_hard_elite_flag_batch,
            "sample_tier_weight": sample_tier_weight_batch,
            "sample_encounter_weight": sample_encounter_weight_batch,
            "sample_sampling_scale": sample_sampling_scale_batch,
            "tier_quota_info": quota_info,  # None when quota config is not set
        }

    def update_priorities(self, indices: List[int], new_priorities: List[float]) -> None:
        """Update priorities for sampled trajectories.

        Args:
            indices: Trajectory indices.
            new_priorities: New priority values.
        """
        for idx, priority in zip(indices, new_priorities):
            if 0 <= idx < len(self.priorities):
                self.priorities[idx] = max(float(priority), 1e-6)

    def state_dict(self) -> Dict[str, Any]:
        """Serialize replay buffer contents for checkpointing."""
        return {
            "schema_version": MUZERO_REPLAY_SCHEMA_VERSION,
            "checkpoint_compatibility_version": int(MUZERO_CHECKPOINT_COMPATIBILITY_VERSION),
            "observation_shape_caps": observation_shape_caps(),
            "capacity": int(self.capacity),
            "trajectories": self.trajectories,
            "priorities": list(self.priorities),
            "total_transitions": int(self._total_transitions),
            "boundary_build_bonus": float(self.boundary_build_bonus),
            "boundary_route_bonus": float(self.boundary_route_bonus),
            "boundary_family_bonus": float(self.boundary_family_bonus),
            "trajectory_quality_bonus": float(self.trajectory_quality_bonus),
            "wasteful_end_turn_scale": float(self.wasteful_end_turn_scale),
            "wasteful_proceed_scale": float(self.wasteful_proceed_scale),
            "encounter_tier_weights": dict(self.encounter_tier_weights),
            "encounter_priority_weights": dict(self.encounter_priority_weights),
        }

    def load_state_dict(self, state: Dict[str, Any]) -> None:
        """Restore replay buffer contents from a serialized state."""
        self.capacity = max(int(state.get("capacity", self.capacity)), 1)
        self.boundary_build_bonus = max(float(state.get("boundary_build_bonus", self.boundary_build_bonus)), 0.0)
        self.boundary_route_bonus = max(float(state.get("boundary_route_bonus", self.boundary_route_bonus)), 0.0)
        self.boundary_family_bonus = max(float(state.get("boundary_family_bonus", self.boundary_family_bonus)), 0.0)
        self.trajectory_quality_bonus = max(float(state.get("trajectory_quality_bonus", self.trajectory_quality_bonus)), 0.0)
        self.wasteful_end_turn_scale = float(np.clip(state.get("wasteful_end_turn_scale", self.wasteful_end_turn_scale), 0.05, 1.0))
        self.wasteful_proceed_scale = float(np.clip(state.get("wasteful_proceed_scale", self.wasteful_proceed_scale), 0.05, 1.0))
        self.encounter_tier_weights = {
            str(tier).strip().lower(): max(float(weight), 0.0)
            for tier, weight in (state.get("encounter_tier_weights") or self.encounter_tier_weights).items()
            if str(tier).strip().lower() in _VALID_ENCOUNTER_TIERS and float(weight) >= 0.0
        }
        self.encounter_priority_weights = {
            normalize_encounter_id(encounter_id): max(float(weight), 0.0)
            for encounter_id, weight in (state.get("encounter_priority_weights") or self.encounter_priority_weights).items()
            if normalize_encounter_id(encounter_id) and float(weight) >= 0.0
        }
        self.trajectories = list(state.get("trajectories") or [])
        self.priorities = [max(float(priority), 1e-6) for priority in (state.get("priorities") or [])]
        if self.priorities and len(self.priorities) != len(self.trajectories):
            raise ValueError("Replay buffer priority count does not match trajectory count.")
        if "total_transitions" in state:
            self._total_transitions = int(state["total_transitions"])
        else:
            self._total_transitions = sum(len(trajectory) for trajectory in self.trajectories)

        # Older checkpoints may predate episode metadata / per-step action traces.
        # Backfill them here so downstream analysis code can rely on the keys.
        for trajectory in self.trajectories:
            if not hasattr(trajectory, "metadata") or trajectory.metadata is None:
                trajectory.metadata = {}
            for step in getattr(trajectory, "steps", []):
                obs = step.get("obs")
                if isinstance(obs, dict) and not _is_packed_observation(obs):
                    step["obs"] = _pack_observation(obs)
                elif obs is None:
                    step["obs"] = _ZERO_PACKED_OBSERVATION
                if "reward_components" not in step or step["reward_components"] is None:
                    step["reward_components"] = np.zeros(NUM_OBJECTIVE_HEADS, dtype=np.float32)
                if "root_value_components" not in step or step["root_value_components"] is None:
                    step["root_value_components"] = np.zeros(NUM_OBJECTIVE_HEADS, dtype=np.float32)
                if "objective_context" not in step or step["objective_context"] is None:
                    step["objective_context"] = np.zeros(0, dtype=np.float32)
                if "semantic_action" not in step or step["semantic_action"] is None:
                    step["semantic_action"] = 0
                if "semantic_policy" not in step or step["semantic_policy"] is None:
                    step["semantic_policy"] = np.zeros(SEMANTIC_ROLLOUT_SIZE, dtype=np.float32)
                if "action_info" not in step or step["action_info"] is None:
                    step["action_info"] = {}
                if "search_stats" not in step or step["search_stats"] is None:
                    step["search_stats"] = {}
                if "decision_domain" not in step or step["decision_domain"] is None:
                    step["decision_domain"] = _step_decision_domain(step)
                if "phase" not in step or step["phase"] is None:
                    step["phase"] = _step_phase(step)
                if "action_family" not in step or step["action_family"] is None:
                    step["action_family"] = _step_action_family(step)
                if "semantic_domain" not in step or step["semantic_domain"] is None:
                    step["semantic_domain"] = _safe_text(_step_semantic_info(step).get("domain")).lower()
                if "surface" not in step or step["surface"] is None:
                    step["surface"] = _step_surface(step)
                if "selection" not in step or step["selection"] is None:
                    step["selection"] = _safe_text(_step_action_info(step).get("selection")).lower()
                if "wasteful_end_turn" not in step or step["wasteful_end_turn"] is None:
                    step["wasteful_end_turn"] = False
                if "wasteful_proceed" not in step or step["wasteful_proceed"] is None:
                    step["wasteful_proceed"] = False
                if "room_type" not in step or step["room_type"] is None:
                    step["room_type"] = ""
                else:
                    step["room_type"] = _safe_text(step.get("room_type")).lower()
                if "encounter_id" not in step or step["encounter_id"] is None:
                    step["encounter_id"] = ""
                else:
                    step["encounter_id"] = _safe_text(step.get("encounter_id"))
                if "encounter_tier" not in step or step["encounter_tier"] is None:
                    step["encounter_tier"] = ""
                else:
                    step["encounter_tier"] = _safe_text(step.get("encounter_tier")).lower()
                if "floor" not in step or step["floor"] is None:
                    step["floor"] = 0.0
                else:
                    step["floor"] = _safe_float_value(step.get("floor"))
                if "act_id" not in step or step["act_id"] is None:
                    step["act_id"] = 0.0
                else:
                    step["act_id"] = _safe_float_value(step.get("act_id"))
                if "settlement_bonus" not in step or step["settlement_bonus"] is None:
                    step["settlement_bonus"] = 0.0

    def __len__(self) -> int:
        """Return total number of transitions stored."""
        return self._total_transitions
