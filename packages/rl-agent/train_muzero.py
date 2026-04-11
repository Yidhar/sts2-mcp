"""MuZero training for Slay the Spire 2 combat scenarios.

Usage:
    python train_muzero.py --total-timesteps 100000 --combat-sandbox \\
        --encounter-pool "ENCOUNTER.SLIMES_WEAK,ENCOUNTER.SHRINKER_BEETLE_WEAK"

This implements self-play MCTS + replay buffer training for the MuZero network.
"""

from __future__ import annotations

import argparse
from collections import defaultdict
import json
import math
import pickle
import signal
import time
from pathlib import Path
from typing import Any, Callable

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import gymnasium as gym
from torch.utils.tensorboard import SummaryWriter

from combat_snapshot_dataset import (
    CombatSnapshotPool,
    DEFAULT_CURATED_COMBINED_SUBSET,
    VALID_ENCOUNTER_TIERS,
    VALID_CURATED_COMBINED_SUBSETS,
    infer_encounter_tier,
    load_combat_snapshot_rows,
)
from launcher import get_session_files as get_default_multi_session_files
from sts2_env.action_compact import compact_action_signature
from sts2_env.bridge_client import BridgeClient
from sts2_env.combat_env import CombatSandboxEnv
from sts2_env.env_v2 import SlayTheSpire2EnvV2
from sts2_env.mcts import MCTS
from sts2_env.muzero_buffer import GameTrajectory, MuZeroReplayBuffer
from sts2_env.muzero_model import MuZeroNetwork, scalar_to_support
from sts2_env.objective_heads import (
    NUM_OBJECTIVE_HEADS,
    compute_transition_objective_rewards,
)
from sts2_env.observation_v2 import DECISION_DOMAINS, DictObservationEncoder, MAX_ACTIONS, NUM_PHASES
from sts2_env.semantic_rollout import (
    SEMANTIC_ROLLOUT_SIZE,
    aggregate_concrete_policy_to_semantic,
    semantic_rollout_index,
)


DEFAULT_COMBAT_SANDBOX_TRAIN_POOL = ",".join([
    "ENCOUNTER.SLIMES_WEAK",
    "ENCOUNTER.SHRINKER_BEETLE_WEAK",
    "ENCOUNTER.FUZZY_WURM_CRAWLER_WEAK",
    "ENCOUNTER.NIBBITS_WEAK",
])


def mask_fn(env):
    """Extract action mask from environment."""
    return env.unwrapped.action_masks()


def parse_encounter_pool(raw: str | None) -> list[str]:
    """Parse comma-separated encounter pool."""
    if not raw:
        return []
    return [entry.strip() for entry in raw.split(",") if entry.strip()]


def parse_session_files(raw: str | None) -> list[str]:
    """Parse comma-separated session files."""
    if not raw:
        return []
    return [entry.strip() for entry in raw.split(",") if entry.strip()]


def parse_encounter_tiers(raw: str | None) -> list[str]:
    """Parse comma-separated encounter tiers."""
    if not raw:
        return []
    tiers = [entry.strip().lower() for entry in raw.split(",") if entry.strip()]
    invalid = sorted(set(tiers).difference(VALID_ENCOUNTER_TIERS))
    if invalid:
        raise ValueError(f"Unsupported --combat-encounter-tiers values: {invalid}")
    return tiers


def parse_tier_weights(raw: str | None) -> dict[str, float]:
    """Parse comma-separated tier weights like ``weak=0.6,normal=0.4``."""
    if not raw:
        return {}

    weights: dict[str, float] = {}
    for chunk in raw.split(","):
        entry = chunk.strip()
        if not entry:
            continue
        if "=" not in entry:
            raise ValueError(
                f"Invalid --combat-tier-weights entry {entry!r}; expected tier=weight pairs."
            )
        tier_raw, weight_raw = entry.split("=", 1)
        tier = tier_raw.strip().lower()
        if tier not in VALID_ENCOUNTER_TIERS:
            raise ValueError(f"Unsupported encounter tier in --combat-tier-weights: {tier!r}")
        try:
            weight = float(weight_raw)
        except ValueError as exc:
            raise ValueError(
                f"Invalid weight {weight_raw!r} for encounter tier {tier!r}."
            ) from exc
        if weight < 0.0:
            raise ValueError(f"Encounter tier weight must be >= 0 for {tier!r}, got {weight}.")
        weights[tier] = weight
    return weights


def parse_encounter_weights(raw: str | None) -> dict[str, float]:
    """Parse comma-separated encounter weights like ``ENCOUNTER.X=3.0``."""
    if not raw:
        return {}

    weights: dict[str, float] = {}
    for chunk in raw.split(","):
        entry = chunk.strip()
        if not entry:
            continue
        if "=" not in entry:
            raise ValueError(
                f"Invalid --combat-encounter-weights entry {entry!r}; expected encounter=weight pairs."
            )
        encounter_raw, weight_raw = entry.split("=", 1)
        encounter_id = encounter_raw.strip()
        if not encounter_id:
            raise ValueError("Encounter id in --combat-encounter-weights cannot be empty.")
        try:
            weight = float(weight_raw)
        except ValueError as exc:
            raise ValueError(
                f"Invalid weight {weight_raw!r} for encounter {encounter_id!r}."
            ) from exc
        if weight < 0.0:
            raise ValueError(
                f"Encounter weight must be >= 0 for {encounter_id!r}, got {weight}."
            )
        weights[encounter_id] = weight
    return weights


def _get_live_supported_encounter_ids(session_file: str | None = None) -> set[str]:
    client = BridgeClient(session_path=session_file)
    catalog = client.combat_catalog()
    return {
        str(entry.get("encounter_id"))
        for entry in (catalog.get("encounters") or [])
        if entry.get("encounter_id")
    }


def resolve_training_session_files(
    *,
    n_envs: int,
    session_file: str | None,
    session_files: list[str],
) -> list[str | None]:
    """Resolve session files for parallel environments."""
    if n_envs < 1:
        raise ValueError("--n-envs must be >= 1")

    if session_file and session_files:
        raise ValueError("Use either --session-file or --session-files, not both.")

    if session_files:
        if len(session_files) != n_envs:
            raise ValueError(
                f"--session-files count ({len(session_files)}) must match --n-envs ({n_envs})."
            )
        return session_files

    if n_envs == 1:
        return [session_file]

    return get_default_multi_session_files(n_envs)


def build_train_env(
    *,
    env_index: int,
    session_file: str | None,
    combat_sandbox: bool,
    character: str | None,
    defensive_buffs: bool,
    encounter_id: str | None,
    encounter_pool: list[str],
    snapshot_pool: CombatSnapshotPool | None,
    reset_timeout_ms: int,
    step_timeout_ms: int,
) -> gym.Env:
    """Build a single training environment."""
    obs_encoder = DictObservationEncoder(use_text=False)

    if combat_sandbox:
        env = CombatSandboxEnv(
            session_file=session_file,
            character=character,
            encounter_id=encounter_id,
            encounter_pool=encounter_pool,
            snapshot_pool=snapshot_pool,
            reset_timeout_ms=reset_timeout_ms,
            step_timeout_ms=step_timeout_ms,
            obs_encoder=obs_encoder,
        )
        return env

    env = SlayTheSpire2EnvV2(
        session_file=session_file,
        character=character,
        defensive_buffs=defensive_buffs,
        reset_timeout_ms=reset_timeout_ms,
        step_timeout_ms=step_timeout_ms,
        obs_encoder=obs_encoder,
    )
    return env


def dict_obs_to_torch(obs: dict[str, Any], device: str = "cpu") -> dict[str, torch.Tensor]:
    """Convert dict observation to torch tensors with batch dimension 1."""
    torch_obs = {}
    for key, value in obs.items():
        if isinstance(value, np.ndarray):
            tensor = torch.from_numpy(value).to(device)
        elif isinstance(value, torch.Tensor):
            tensor = value.to(device)
        else:
            tensor = torch.tensor(value, device=device)

        # Add batch dimension if needed
        if tensor.dim() == 0:
            tensor = tensor.unsqueeze(0)
        elif tensor.dim() > 0 and tensor.shape[0] != 1:
            tensor = tensor.unsqueeze(0)

        torch_obs[key] = tensor

    return torch_obs


def load_resume_checkpoint(
    resume_from: str | Path,
    *,
    network: MuZeroNetwork,
    optimizer: optim.Optimizer,
    buffer: MuZeroReplayBuffer,
    device: str,
    load_buffer: bool = True,
) -> dict[str, Any]:
    """Load network/optimizer/(optional) replay buffer from a MuZero checkpoint directory."""
    checkpoint_path = Path(resume_from)
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Resume checkpoint path not found: {checkpoint_path}")

    network_path = checkpoint_path / "network.pt"
    optimizer_path = checkpoint_path / "optimizer.pt"
    metadata_path = checkpoint_path / "metadata.json"
    replay_buffer_path = checkpoint_path / "replay_buffer.pkl"

    if not network_path.exists():
        raise FileNotFoundError(f"Missing network.pt in checkpoint: {checkpoint_path}")
    if not optimizer_path.exists():
        raise FileNotFoundError(f"Missing optimizer.pt in checkpoint: {checkpoint_path}")
    if not metadata_path.exists():
        raise FileNotFoundError(f"Missing metadata.json in checkpoint: {checkpoint_path}")

    network_state = torch.load(network_path, map_location=device)
    optimizer_state = torch.load(optimizer_path, map_location=device)

    current_state = network.state_dict()
    compatible_state = {}
    skipped_keys: list[str] = []
    for key, value in network_state.items():
        current_value = current_state.get(key)
        if current_value is not None and tuple(current_value.shape) == tuple(value.shape):
            compatible_state[key] = value
        else:
            skipped_keys.append(key)
    missing_keys = [key for key in current_state.keys() if key not in compatible_state]

    merged_state = dict(current_state)
    merged_state.update(compatible_state)
    network.load_state_dict(merged_state)

    if skipped_keys or missing_keys:
        print(
            "[resume] Partial network warm-start: "
            f"loaded={len(compatible_state)} skipped={len(skipped_keys)} missing={len(missing_keys)}"
        )
        if skipped_keys:
            print(f"[resume] Skipped checkpoint keys (first 12): {skipped_keys[:12]}")
        if missing_keys:
            print(f"[resume] Newly initialized model keys (first 12): {missing_keys[:12]}")

    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))

    allow_exact_resume = not (skipped_keys or missing_keys)

    if allow_exact_resume:
        try:
            optimizer.load_state_dict(optimizer_state)
        except Exception as exc:
            print(f"[resume] Optimizer state not loaded ({exc}); starting with fresh optimizer.")
    else:
        print(
            "[resume] Skipping optimizer warm-start because network state was only partially restored; "
            "starting with fresh optimizer."
        )

    allow_replay_buffer_load = allow_exact_resume and load_buffer
    if replay_buffer_path.exists() and allow_replay_buffer_load:
        with replay_buffer_path.open("rb") as handle:
            replay_buffer_state = pickle.load(handle)
        buffer.load_state_dict(replay_buffer_state)
    elif replay_buffer_path.exists() and not load_buffer:
        print("[resume] Skipping replay buffer warm-start by request; starting with empty buffer.")
    elif replay_buffer_path.exists():
        print(
            "[resume] Skipping replay buffer warm-start because observation/model schema changed; "
            "starting with empty buffer."
        )
    else:
        print(f"[resume] replay_buffer.pkl not found in {checkpoint_path}; resuming with empty buffer.")

    return metadata


class MuZeroTrainer:
    """Orchestrates MuZero self-play training."""

    def __init__(
        self,
        network: MuZeroNetwork,
        mcts: MCTS,
        buffer: MuZeroReplayBuffer,
        env: CombatSandboxEnv,
        optimizer: optim.Optimizer,
        device: str = "cpu",
        discount: float = 0.997,
        n_step_return: int = 10,
        max_grad_norm: float = 1.0,
        latent_policy_distill_weight: float = 0.25,
        latent_policy_target_weight: float = 0.5,
        objective_value_weight: float = 1.0,
        objective_reward_weight: float = 0.75,
        semantic_policy_weight: float = 1.0,
        semantic_value_weight: float = 1.0,
        semantic_reward_weight: float = 0.75,
        semantic_state_consistency_weight: float = 2.0,
        objective_diversity_weight: float = 0.05,
        semantic_policy_label_smoothing: float = 0.02,
        state_consistency_weight: float = 2.0,
        surface_mask_weight: float = 1.5,
        surface_count_weight: float = 0.25,
        surface_domain_weight: float = 0.15,
        surface_phase_weight: float = 0.1,
        log_dir: str = "runs",
        checkpoint_dir: str = "checkpoints",
    ):
        """Initialize trainer.

        Args:
            network: MuZeroNetwork instance.
            mcts: MCTS instance for action selection.
            buffer: Replay buffer.
            env: Training environment.
            optimizer: Adam optimizer for network.
            device: Device for training.
            discount: Discount factor.
            n_step_return: Steps for bootstrapped returns.
            max_grad_norm: Max gradient norm for clipping.
            log_dir: TensorBoard log directory.
            checkpoint_dir: Checkpoint save directory.
        """
        self.network = network
        self.mcts = mcts
        self.buffer = buffer
        self.env = env
        self.optimizer = optimizer
        self.device = device
        self.discount = discount
        self.n_step_return = n_step_return
        self.max_grad_norm = max_grad_norm
        self.latent_policy_distill_weight = latent_policy_distill_weight
        self.latent_policy_target_weight = latent_policy_target_weight
        self.objective_value_weight = max(float(objective_value_weight), 0.0)
        self.objective_reward_weight = max(float(objective_reward_weight), 0.0)
        self.semantic_policy_weight = max(float(semantic_policy_weight), 0.0)
        self.semantic_value_weight = max(float(semantic_value_weight), 0.0)
        self.semantic_reward_weight = max(float(semantic_reward_weight), 0.0)
        self.semantic_state_consistency_weight = max(float(semantic_state_consistency_weight), 0.0)
        self.objective_diversity_weight = max(float(objective_diversity_weight), 0.0)
        self.semantic_policy_label_smoothing = min(max(float(semantic_policy_label_smoothing), 0.0), 0.25)
        self.state_consistency_weight = state_consistency_weight
        self.surface_mask_weight = surface_mask_weight
        self.surface_count_weight = max(float(surface_count_weight), 0.0)
        self.surface_domain_weight = surface_domain_weight
        self.surface_phase_weight = surface_phase_weight

        Path(log_dir).mkdir(parents=True, exist_ok=True)
        Path(checkpoint_dir).mkdir(parents=True, exist_ok=True)
        self.log_dir = log_dir
        self.checkpoint_dir = checkpoint_dir

        self.writer = SummaryWriter(log_dir=log_dir)
        self.total_steps = 0
        self.episode_count = 0
        self.last_episode_metrics: dict[str, Any] = {}

    @staticmethod
    def _compact_search_stats(search_stats: dict[str, Any]) -> dict[str, Any]:
        if not search_stats:
            return {}
        compact: dict[str, Any] = {}
        for key in (
            "root_candidates",
            "mean_expanded_children",
            "mean_predicted_legal_count",
            "mean_surface_keep_count",
            "max_search_depth",
            "mean_leaf_depth",
            "depth_ge_2_rate",
            "depth_ge_3_rate",
            "end_turn_bias_applied",
            "objective_prior_applied",
            "objective_weight_survival",
            "objective_weight_hp",
            "objective_weight_build",
            "objective_weight_resource",
            "root_objective_value",
            "root_bias_scale",
            "semantic_switch_depth",
            "semantic_rollout_enabled",
            "semantic_expansion_rate",
            "semantic_switch_rate",
            "semantic_chain_steps_mean",
            "semantic_drill_rate",
        ):
            value = search_stats.get(key)
            if value is None:
                continue
            try:
                compact[key] = float(value)
            except (TypeError, ValueError):
                continue
        return compact

    @staticmethod
    def _decision_domain_name(
        obs: dict[str, Any] | None,
        fallback: str = "build",
    ) -> str:
        if not isinstance(obs, dict):
            return fallback
        raw = obs.get("decision_domain")
        if isinstance(raw, str) and raw in DECISION_DOMAINS:
            return raw
        if raw is None:
            return fallback
        raw_np = np.asarray(raw, dtype=np.float32).reshape(-1)
        if raw_np.size == len(DECISION_DOMAINS) and float(raw_np.sum()) > 0.0:
            index = int(np.argmax(raw_np))
            return DECISION_DOMAINS[index]
        return fallback

    def _obs_list_to_torch(self, obs_list: list[dict[str, Any]]) -> dict[str, torch.Tensor]:
        """Convert a list of dict observations into a batched torch dict."""
        obs_torch_batch: dict[str, torch.Tensor] = {}
        for key in obs_list[0].keys():
            values = [obs[key] for obs in obs_list]
            if isinstance(values[0], np.ndarray):
                obs_torch_batch[key] = torch.from_numpy(np.stack(values, axis=0)).to(self.device)
            else:
                obs_torch_batch[key] = torch.as_tensor(values, device=self.device)
        return obs_torch_batch

    def compute_temperature(self, step: int, total_steps: int) -> float:
        """Compute temperature schedule: linear decay from 1.0 to 0.1."""
        progress = min(step / max(total_steps, 1), 1.0)
        return 1.0 - progress * 0.9

    def self_play_episode(self, temperature: float = 1.0) -> tuple[float, int]:
        """Run one self-play episode using MCTS action selection.

        Args:
            temperature: Temperature for action selection.

        Returns:
            (episode_reward, episode_length)
        """
        obs, info = self.env.reset()
        initial_info = dict(info)
        trajectory = GameTrajectory()

        episode_reward = 0.0
        episode_length = 0
        search_root_candidates: list[float] = []
        search_expanded_children: list[float] = []
        search_predicted_legal: list[float] = []
        search_surface_keep: list[float] = []
        search_depths: list[float] = []
        search_depth_ge_2: list[float] = []
        search_depth_ge_3: list[float] = []
        search_semantic_expansion: list[float] = []
        search_semantic_switch: list[float] = []
        search_semantic_chain_steps: list[float] = []
        search_semantic_drill: list[float] = []
        search_root_bias_scale: list[float] = []
        search_end_turn_guard_applied: list[float] = []
        search_end_turn_guard_forced_alt: list[float] = []
        domain_search_stats: dict[str, dict[str, list[float]]] = {
            domain: defaultdict(list) for domain in DECISION_DOMAINS
        }
        decision_counts: dict[str, int] = {domain: 0 for domain in DECISION_DOMAINS}
        selected_family_counts: dict[str, dict[str, int]] = {
            domain: defaultdict(int) for domain in DECISION_DOMAINS
        }
        terminated = False
        truncated = False

        while True:
            prev_info = dict(info)
            action_mask = info.get("action_mask", np.ones(MAX_ACTIONS, dtype=bool))
            legal_actions = (
                info.get("legal_actions_compact")
                or info.get("legal_actions")
                or getattr(self.env.unwrapped, "get_compact_legal_actions", lambda: [])()
            )
            planner_context = prev_info.get("planner_context") if isinstance(prev_info.get("planner_context"), dict) else {}
            semantic_actions = planner_context.get("semantic_actions") if isinstance(planner_context.get("semantic_actions"), list) else []
            decision_domain = self._decision_domain_name(obs)
            if decision_domain not in decision_counts:
                decision_domain = "build"
            decision_counts[decision_domain] += 1

            # Run MCTS to get action and policy
            with torch.no_grad():
                self.mcts.set_training_step(self.total_steps)
                self.mcts.set_root_bias_enabled(True)
                self.mcts.set_semantic_rollout_enabled(self.mcts._semantic_rollout_enabled)
                action_idx, search_policy = self.mcts.run(
                    self.network,
                    obs,
                    action_mask,
                    temperature=temperature,
                )
            search_stats = getattr(self.mcts, "last_run_stats", {}) or {}
            if search_stats:
                search_root_candidates.append(float(search_stats.get("root_candidates", 0.0)))
                search_expanded_children.append(float(search_stats.get("mean_expanded_children", 0.0)))
                search_predicted_legal.append(float(search_stats.get("mean_predicted_legal_count", 0.0)))
                search_surface_keep.append(float(search_stats.get("mean_surface_keep_count", 0.0)))
                search_depths.append(float(search_stats.get("mean_leaf_depth", 0.0)))
                search_depth_ge_2.append(float(search_stats.get("depth_ge_2_rate", 0.0)))
                search_depth_ge_3.append(float(search_stats.get("depth_ge_3_rate", 0.0)))
                search_semantic_expansion.append(float(search_stats.get("semantic_expansion_rate", 0.0)))
                search_semantic_switch.append(float(search_stats.get("semantic_switch_rate", 0.0)))
                search_semantic_chain_steps.append(float(search_stats.get("semantic_chain_steps_mean", 0.0)))
                search_semantic_drill.append(float(search_stats.get("semantic_drill_rate", 0.0)))
                search_root_bias_scale.append(float(search_stats.get("root_bias_scale", 0.0)))
                search_end_turn_guard_applied.append(float(search_stats.get("end_turn_guard_applied", 0.0)))
                search_end_turn_guard_forced_alt.append(float(search_stats.get("end_turn_guard_forced_alternative", 0.0)))
                for metric_key, metric_value in search_stats.items():
                    try:
                        domain_search_stats[decision_domain][metric_key].append(float(metric_value))
                    except (TypeError, ValueError):
                        continue

            # Get initial inference for root value
            obs_batch = dict_obs_to_torch(obs, device=self.device)
            with torch.no_grad():
                initial = self.network.initial_inference(obs_batch)
                root_value = initial.value.squeeze(0).item()
                root_value_components = initial.value_components.squeeze(0).detach().cpu().numpy()

            chosen_action = legal_actions[action_idx] if action_idx < len(legal_actions) else None
            chosen_semantic = (
                semantic_actions[action_idx]
                if action_idx < len(semantic_actions) and isinstance(semantic_actions[action_idx], dict)
                else None
            )
            semantic_candidate_indices = [
                semantic_rollout_index(signature)
                for signature in semantic_actions[:MAX_ACTIONS]
            ]
            semantic_policy = aggregate_concrete_policy_to_semantic(
                search_policy,
                semantic_candidate_indices,
            )
            semantic_action = semantic_rollout_index(chosen_semantic or chosen_action)
            chosen_signature = compact_action_signature(chosen_action)
            if chosen_signature:
                chosen_signature = {
                    **chosen_signature,
                    "selected_index": int(action_idx),
                    "phase": info.get("phase"),
                    "legal_action_count": int(info.get("legal_action_count", 0) or 0),
                }
                semantic_compact = chosen_signature.get("semantic")
                if isinstance(semantic_compact, dict):
                    family = str(semantic_compact.get("family") or "").strip()
                    if family:
                        selected_family_counts[decision_domain][family] += 1

            # Store transition
            trajectory.add_step(
                obs=obs,
                action=action_idx,
                reward=0.0,  # Reward is assigned after step
                reward_components=np.zeros(NUM_OBJECTIVE_HEADS, dtype=np.float32),
                action_mask=action_mask,
                search_policy=search_policy,
                root_value=root_value,
                root_value_components=root_value_components,
                objective_context=obs.get("objective_context"),
                semantic_action=semantic_action,
                semantic_policy=semantic_policy,
                action_info=chosen_signature,
                search_stats=self._compact_search_stats(search_stats),
            )

            # Take action in environment
            obs, reward, terminated, truncated, info = self.env.step(int(action_idx))
            episode_reward += reward
            episode_length += 1

            reward_components = compute_transition_objective_rewards(
                prev_info.get("transition_state") if isinstance(prev_info.get("transition_state"), dict) else None,
                chosen_semantic or chosen_action,
                info.get("transition_state") if isinstance(info.get("transition_state"), dict) else None,
                prev_planner_context=planner_context,
                next_planner_context=info.get("planner_context") if isinstance(info.get("planner_context"), dict) else None,
                terminated=terminated,
                truncated=truncated,
            )

            # Update last step's reward
            trajectory.steps[-1]["reward"] = reward
            trajectory.steps[-1]["reward_components"] = reward_components.astype(np.float32, copy=False)

            if terminated or truncated:
                break

        final_info = dict(info)
        total_decisions = max(sum(decision_counts.values()), 1)
        trajectory.metadata = {
            "episode_mode": initial_info.get("episode_mode"),
            "encounter_id": initial_info.get("encounter_id"),
            "encounter_tier": infer_encounter_tier(initial_info.get("encounter_id")),
            "encounter_pool": list(initial_info.get("encounter_pool") or []),
            "snapshot_sample_id": initial_info.get("snapshot_sample_id"),
            "snapshot_run_id": initial_info.get("snapshot_run_id"),
            "snapshot_floor_number": initial_info.get("snapshot_floor_number"),
            "snapshot_build_id": initial_info.get("snapshot_build_id"),
            "temperature": float(temperature),
            "semantic_rollout_enabled": bool(self.mcts._semantic_rollout_enabled),
            "semantic_switch_depth": int(self.mcts.semantic_switch_depth),
            "episode_total_reward": float(episode_reward),
            "episode_length": int(episode_length),
            "terminated": bool(terminated),
            "truncated": bool(truncated),
            "initial_phase": initial_info.get("phase"),
            "final_phase": final_info.get("phase"),
            "negative_reward_episode": bool(episode_reward < 0.0),
            "decision_counts": {domain: int(count) for domain, count in decision_counts.items()},
        }

        self.buffer.save_episode(
            trajectory,
            discount=self.discount,
            n_steps=self.n_step_return,
        )
        self.episode_count += 1

        if search_root_candidates:
            self.writer.add_scalar(
                "search/max_depth_mean",
                float(np.mean([float(s.get("max_search_depth", 0.0)) for s in trajectory.steps if isinstance(s.get("search_stats"), dict)]))
                if trajectory.steps
                else 0.0,
                self.episode_count,
            )
            self.writer.add_scalar(
                "search/root_candidates_mean",
                float(np.mean(search_root_candidates)),
                self.episode_count,
            )
            self.writer.add_scalar(
                "search/expanded_children_mean",
                float(np.mean(search_expanded_children)),
                self.episode_count,
            )
            self.writer.add_scalar(
                "search/predicted_legal_mean",
                float(np.mean(search_predicted_legal)),
                self.episode_count,
            )
            self.writer.add_scalar(
                "search/surface_keep_mean",
                float(np.mean(search_surface_keep)) if search_surface_keep else 0.0,
                self.episode_count,
            )
            self.writer.add_scalar(
                "search/leaf_depth_mean",
                float(np.mean(search_depths)),
                self.episode_count,
            )
            self.writer.add_scalar(
                "search/depth_ge_2_rate",
                float(np.mean(search_depth_ge_2)) if search_depth_ge_2 else 0.0,
                self.episode_count,
            )
            self.writer.add_scalar(
                "search/depth_ge_3_rate",
                float(np.mean(search_depth_ge_3)) if search_depth_ge_3 else 0.0,
                self.episode_count,
            )
            self.writer.add_scalar(
                "search/semantic_expansion_rate",
                float(np.mean(search_semantic_expansion)) if search_semantic_expansion else 0.0,
                self.episode_count,
            )
            self.writer.add_scalar(
                "search/semantic_switch_rate",
                float(np.mean(search_semantic_switch)) if search_semantic_switch else 0.0,
                self.episode_count,
            )
            self.writer.add_scalar(
                "search/semantic_chain_steps_mean",
                float(np.mean(search_semantic_chain_steps)) if search_semantic_chain_steps else 0.0,
                self.episode_count,
            )
            self.writer.add_scalar(
                "search/semantic_drill_rate",
                float(np.mean(search_semantic_drill)) if search_semantic_drill else 0.0,
                self.episode_count,
            )
            self.writer.add_scalar(
                "search/root_bias_scale",
                float(np.mean(search_root_bias_scale)) if search_root_bias_scale else 0.0,
                self.episode_count,
            )
            self.writer.add_scalar(
                "search/end_turn_guard_applied",
                float(np.mean(search_end_turn_guard_applied)) if search_end_turn_guard_applied else 0.0,
                self.episode_count,
            )
            self.writer.add_scalar(
                "search/end_turn_guard_forced_alternative",
                float(np.mean(search_end_turn_guard_forced_alt)) if search_end_turn_guard_forced_alt else 0.0,
                self.episode_count,
            )
            objective_weight_keys = (
                ("search/objective_weight_survival", "objective_weight_survival"),
                ("search/objective_weight_hp", "objective_weight_hp"),
                ("search/objective_weight_build", "objective_weight_build"),
                ("search/objective_weight_resource", "objective_weight_resource"),
                ("search/root_objective_value", "root_objective_value"),
                ("search/end_turn_bias_applied", "end_turn_bias_applied"),
                ("search/objective_prior_applied", "objective_prior_applied"),
            )
            for writer_key, stat_key in objective_weight_keys:
                values = [
                    float(step.get("search_stats", {}).get(stat_key, 0.0))
                    for step in trajectory.steps
                    if isinstance(step.get("search_stats"), dict)
                ]
                if values:
                    self.writer.add_scalar(writer_key, float(np.mean(values)), self.episode_count)

        domain_search_means: dict[str, dict[str, float]] = {}
        domain_family_rates: dict[str, dict[str, float]] = {}

        self.writer.add_scalar("decision/total_count", float(total_decisions), self.episode_count)
        for domain in DECISION_DOMAINS:
            domain_count = decision_counts.get(domain, 0)
            self.writer.add_scalar(f"decision/{domain}_count", float(domain_count), self.episode_count)
            self.writer.add_scalar(f"decision/{domain}_share", float(domain_count) / float(total_decisions), self.episode_count)
            metric_lists = domain_search_stats.get(domain) or {}
            if not metric_lists:
                continue
            metric_name_map = {
                "root_candidates": "root_candidates_mean",
                "mean_expanded_children": "expanded_children_mean",
                "mean_predicted_legal_count": "predicted_legal_mean",
                "mean_surface_keep_count": "surface_keep_mean",
                "max_search_depth": "max_depth_mean",
                "mean_leaf_depth": "leaf_depth_mean",
                "depth_ge_2_rate": "depth_ge_2_rate",
                "depth_ge_3_rate": "depth_ge_3_rate",
                "semantic_switch_rate": "semantic_switch_rate",
                "semantic_expansion_rate": "semantic_expansion_rate",
                "semantic_chain_steps_mean": "semantic_chain_steps_mean",
                "semantic_drill_rate": "semantic_drill_rate",
                "root_bias_scale": "root_bias_scale",
                "objective_weight_survival": "objective_weight_survival",
                "objective_weight_hp": "objective_weight_hp",
                "objective_weight_build": "objective_weight_build",
                "objective_weight_resource": "objective_weight_resource",
                "root_objective_value": "root_objective_value",
                "end_turn_bias_applied": "end_turn_bias_applied",
                "end_turn_guard_applied": "end_turn_guard_applied",
                "end_turn_guard_forced_alternative": "end_turn_guard_forced_alternative",
                "objective_prior_applied": "objective_prior_applied",
            }
            for stat_key, writer_suffix in metric_name_map.items():
                values = metric_lists.get(stat_key)
                if values:
                    domain_search_means.setdefault(domain, {})[stat_key] = float(np.mean(values))
                    self.writer.add_scalar(
                        f"search/{domain}/{writer_suffix}",
                        float(np.mean(values)),
                        self.episode_count,
                    )

            family_counts = selected_family_counts.get(domain) or {}
            if family_counts and domain_count > 0:
                for family, count in sorted(family_counts.items()):
                    safe_family = family.replace("/", "_").replace(" ", "_")
                    domain_family_rates.setdefault(domain, {})[family] = float(count) / float(domain_count)
                    self.writer.add_scalar(
                        f"decision/{domain}/family_{safe_family}_rate",
                        float(count) / float(domain_count),
                        self.episode_count,
                    )

        self.last_episode_metrics = {
            "decision_counts": {domain: int(count) for domain, count in decision_counts.items()},
            "decision_total": int(total_decisions),
            "domain_search_means": domain_search_means,
            "domain_family_rates": domain_family_rates,
            "episode_reward": float(episode_reward),
            "episode_length": int(episode_length),
        }

        return episode_reward, episode_length

    def train_step(self, batch_size: int, unroll_steps: int = 5) -> dict[str, float]:
        """Perform one training step.

        Args:
            batch_size: Batch size for training.
            unroll_steps: Number of unroll steps.

        Returns:
            Dict with loss values.
        """
        if len(self.buffer) < batch_size:
            return {}

        batch = self.buffer.sample_batch(
            batch_size,
            unroll_steps=unroll_steps,
            discount=self.discount,
            n_step_return=self.n_step_return,
        )

        # Extract tensors
        obs_sequence_batch = batch["obs_sequence_batch"]
        action_batch = batch["action_batch"].to(self.device)  # [B, K]
        semantic_action_batch = batch["semantic_action_batch"].to(self.device)  # [B, K]
        reward_target = batch["reward_target"].to(self.device)  # [B, K]
        reward_component_target = batch["reward_component_target"].to(self.device)  # [B, K, H]
        value_target = batch["value_target"].to(self.device)  # [B, K+1]
        value_component_target = batch["value_component_target"].to(self.device)  # [B, K+1, H]
        policy_target = batch["policy_target"].to(self.device)  # [B, K+1, 80]
        semantic_policy_target = batch["semantic_policy_target"].to(self.device)  # [B, K+1, S]
        action_mask_batch = batch["action_mask_batch"].to(self.device)  # [B, K+1, 80]

        batch_size_actual = int(action_batch.shape[0])

        # Convert observation sequence to torch batches.
        # obs_sequence_batch[b][k] = observation at position pos+k (or zero obs past terminal)
        obs_torch_sequence = [
            self._obs_list_to_torch([sequence[k] for sequence in obs_sequence_batch])
            for k in range(unroll_steps + 1)
        ]

        # Initial inference
        initial = self.network.initial_inference(obs_torch_sequence[0])
        hidden_state = initial.hidden_state
        policy_logits = initial.policy_logits
        value_logits = initial.value_logits

        # Losses
        total_loss = 0.0
        policy_loss_sum = 0.0
        latent_policy_loss_sum = 0.0
        value_loss_sum = 0.0
        objective_value_loss_sum = 0.0
        reward_loss_sum = 0.0
        objective_reward_loss_sum = 0.0
        semantic_policy_loss_sum = 0.0
        semantic_value_loss_sum = 0.0
        semantic_reward_loss_sum = 0.0
        semantic_state_consistency_loss_sum = 0.0
        objective_diversity_loss_sum = 0.0
        latent_policy_distill_loss_sum = 0.0
        state_consistency_loss_sum = 0.0
        latent_policy_distill_kl_sum = 0.0
        latent_policy_distill_agreement_sum = 0.0
        teacher_entropy_sum = 0.0
        student_entropy_sum = 0.0
        state_consistency_cosine_sum = 0.0
        surface_mask_loss_sum = 0.0
        surface_count_loss_sum = 0.0
        surface_domain_loss_sum = 0.0
        surface_phase_loss_sum = 0.0
        surface_precision_sum = 0.0
        surface_recall_sum = 0.0
        surface_f1_sum = 0.0
        surface_count_mae_sum = 0.0
        surface_domain_acc_sum = 0.0
        surface_phase_acc_sum = 0.0
        semantic_training_active = (
            self.semantic_policy_weight > 0.0
            or self.semantic_value_weight > 0.0
            or self.semantic_reward_weight > 0.0
            or self.semantic_state_consistency_weight > 0.0
            or self.objective_diversity_weight > 0.0
        )

        # Initial step loss
        policy_loss = self._policy_loss(policy_logits, policy_target[:, 0], action_mask_batch[:, 0])
        latent_policy_loss = self._policy_loss(
            initial.latent_policy_logits,
            policy_target[:, 0],
            action_mask_batch[:, 0],
        )
        value_loss = self._value_loss(value_logits, value_target[:, 0])
        objective_value_loss = self._objective_value_loss(
            initial.value_component_logits,
            value_component_target[:, 0],
        )
        if semantic_training_active:
            semantic_root = self.network.semantic_prediction(
                self.network.project_to_semantic_latent(hidden_state),
                obs=obs_torch_sequence[0],
            )
            semantic_policy_loss = self._semantic_policy_loss(
                semantic_root.semantic_policy_logits,
                semantic_policy_target[:, 0],
            )
            semantic_value_loss = self._value_loss(
                semantic_root.value_logits,
                value_target[:, 0],
            )
            semantic_objective_value_loss = self._objective_value_loss(
                semantic_root.value_component_logits,
                value_component_target[:, 0],
            )
            objective_diversity_loss = self._objective_head_diversity_loss(
                initial.value_components,
                semantic_root.value_components,
            )
        else:
            semantic_policy_loss = value_logits.new_zeros(())
            semantic_value_loss = value_logits.new_zeros(())
            semantic_objective_value_loss = value_logits.new_zeros(())
            objective_diversity_loss = value_logits.new_zeros(())
        latent_policy_distill_loss = self._latent_policy_distill_loss(
            initial.latent_policy_logits,
            policy_logits,
            action_mask_batch[:, 0],
        )
        latent_policy_distill_metrics = self._latent_policy_distill_metrics(
            initial.latent_policy_logits,
            policy_logits,
            action_mask_batch[:, 0],
        )
        loss = (
            policy_loss
            + value_loss
            + self.objective_value_weight * objective_value_loss
            + self.semantic_policy_weight * semantic_policy_loss
            + self.semantic_value_weight * (semantic_value_loss + semantic_objective_value_loss)
            + self.objective_diversity_weight * objective_diversity_loss
            + self.latent_policy_target_weight * latent_policy_loss
            + self.latent_policy_distill_weight * latent_policy_distill_loss
        )
        total_loss += loss / (unroll_steps + 1)
        policy_loss_sum += policy_loss.item()
        latent_policy_loss_sum += latent_policy_loss.item()
        value_loss_sum += value_loss.item()
        objective_value_loss_sum += objective_value_loss.item()
        semantic_policy_loss_sum += semantic_policy_loss.item()
        semantic_value_loss_sum += (semantic_value_loss.item() + semantic_objective_value_loss.item())
        objective_diversity_loss_sum += objective_diversity_loss.item()
        latent_policy_distill_loss_sum += latent_policy_distill_loss.item()
        latent_policy_distill_kl_sum += latent_policy_distill_metrics["kl"]
        latent_policy_distill_agreement_sum += latent_policy_distill_metrics["top1_agreement"]
        teacher_entropy_sum += latent_policy_distill_metrics["teacher_entropy"]
        student_entropy_sum += latent_policy_distill_metrics["student_entropy"]

        # Unrolled steps
        for step_k in range(unroll_steps):
            current_obs_torch = obs_torch_sequence[step_k]
            next_obs_torch = obs_torch_sequence[step_k + 1]
            action_embeddings = self.network.encode_actions(current_obs_torch)  # [B, 80, 64]

            # Select action embeddings
            action_indices = action_batch[:, step_k]  # [B]
            action_emb = action_embeddings[torch.arange(batch_size_actual), action_indices]  # [B, 64]

            # Recurrent inference
            recurrent = self.network.recurrent_inference(hidden_state, action_emb, next_obs=next_obs_torch)
            semantic_recurrent = None
            if semantic_training_active:
                current_semantic_hidden = self.network.project_to_semantic_latent(hidden_state)
                semantic_recurrent = self.network.semantic_recurrent_inference(
                    current_semantic_hidden,
                    semantic_action_batch[:, step_k],
                    next_obs=next_obs_torch,
                )

            # Losses for this step
            policy_loss = self._policy_loss(
                recurrent.policy_logits,
                policy_target[:, step_k + 1],
                action_mask_batch[:, step_k + 1],
            )
            latent_policy_loss = self._policy_loss(
                recurrent.latent_policy_logits,
                policy_target[:, step_k + 1],
                action_mask_batch[:, step_k + 1],
            )
            value_loss = self._value_loss(recurrent.value_logits, value_target[:, step_k + 1])
            reward_loss = self._reward_loss(recurrent.reward_logits, reward_target[:, step_k])
            objective_value_loss = self._objective_value_loss(
                recurrent.value_component_logits,
                value_component_target[:, step_k + 1],
            )
            objective_reward_loss = self._objective_reward_loss(
                recurrent.reward_component_logits,
                reward_component_target[:, step_k],
            )
            if semantic_training_active and semantic_recurrent is not None:
                semantic_policy_loss = self._semantic_policy_loss(
                    semantic_recurrent.semantic_policy_logits,
                    semantic_policy_target[:, step_k + 1],
                )
                semantic_value_loss = self._value_loss(
                    semantic_recurrent.value_logits,
                    value_target[:, step_k + 1],
                )
                semantic_objective_value_loss = self._objective_value_loss(
                    semantic_recurrent.value_component_logits,
                    value_component_target[:, step_k + 1],
                )
                semantic_reward_loss = self._reward_loss(
                    semantic_recurrent.reward_logits,
                    reward_target[:, step_k],
                )
                semantic_objective_reward_loss = self._objective_reward_loss(
                    semantic_recurrent.reward_component_logits,
                    reward_component_target[:, step_k],
                )
            else:
                semantic_policy_loss = value_logits.new_zeros(())
                semantic_value_loss = value_logits.new_zeros(())
                semantic_objective_value_loss = value_logits.new_zeros(())
                semantic_reward_loss = value_logits.new_zeros(())
                semantic_objective_reward_loss = value_logits.new_zeros(())
            latent_policy_distill_loss = self._latent_policy_distill_loss(
                recurrent.latent_policy_logits,
                recurrent.policy_logits,
                action_mask_batch[:, step_k + 1],
            )
            with torch.no_grad():
                teacher_hidden = self.network.representation(next_obs_torch)
                teacher_projected = self.network.project_hidden_state(teacher_hidden)
                teacher_semantic = self.network.project_to_semantic_latent(teacher_hidden)
            student_projected = self.network.project_hidden_state(recurrent.next_hidden_state)
            state_consistency_loss, state_consistency_cosine = self._state_consistency_loss(
                student_projected,
                teacher_projected,
            )
            if semantic_training_active and semantic_recurrent is not None:
                semantic_state_consistency_loss, _ = self._state_consistency_loss(
                    semantic_recurrent.next_semantic_hidden_state,
                    teacher_semantic.detach(),
                )
                objective_diversity_loss = self._objective_head_diversity_loss(
                    recurrent.value_components,
                    semantic_recurrent.value_components,
                    recurrent.reward_components,
                    semantic_recurrent.reward_components,
                )
            else:
                semantic_state_consistency_loss = value_logits.new_zeros(())
                objective_diversity_loss = value_logits.new_zeros(())
            surface_mask_loss = self._surface_mask_loss(
                recurrent.next_action_mask_logits,
                action_mask_batch[:, step_k + 1],
            )
            surface_count_loss = self._surface_count_loss(
                recurrent.next_action_mask_logits,
                action_mask_batch[:, step_k + 1],
            )
            surface_domain_loss = self._surface_domain_loss(
                recurrent.next_decision_domain_logits,
                next_obs_torch["decision_domain"],
            )
            surface_phase_loss = self._surface_phase_loss(
                recurrent.next_phase_logits,
                next_obs_torch["scalars"],
            )
            latent_policy_distill_metrics = self._latent_policy_distill_metrics(
                recurrent.latent_policy_logits,
                recurrent.policy_logits,
                action_mask_batch[:, step_k + 1],
            )
            surface_metrics = self._surface_metrics(
                recurrent.next_action_mask_logits,
                action_mask_batch[:, step_k + 1],
                recurrent.next_decision_domain_logits,
                next_obs_torch["decision_domain"],
                recurrent.next_phase_logits,
                next_obs_torch["scalars"],
            )

            loss = (
                policy_loss
                + value_loss
                + reward_loss
                + self.objective_value_weight * objective_value_loss
                + self.objective_reward_weight * objective_reward_loss
                + self.semantic_policy_weight * semantic_policy_loss
                + self.semantic_value_weight * (semantic_value_loss + semantic_objective_value_loss)
                + self.semantic_reward_weight * (semantic_reward_loss + semantic_objective_reward_loss)
                + self.semantic_state_consistency_weight * semantic_state_consistency_loss
                + self.objective_diversity_weight * objective_diversity_loss
                + self.latent_policy_target_weight * latent_policy_loss
                + self.latent_policy_distill_weight * latent_policy_distill_loss
                + self.state_consistency_weight * state_consistency_loss
                + self.surface_mask_weight * surface_mask_loss
                + self.surface_count_weight * surface_count_loss
                + self.surface_domain_weight * surface_domain_loss
                + self.surface_phase_weight * surface_phase_loss
            ) / (unroll_steps + 1)
            total_loss += loss
            policy_loss_sum += policy_loss.item()
            latent_policy_loss_sum += latent_policy_loss.item()
            value_loss_sum += value_loss.item()
            reward_loss_sum += reward_loss.item()
            objective_value_loss_sum += objective_value_loss.item()
            objective_reward_loss_sum += objective_reward_loss.item()
            semantic_policy_loss_sum += semantic_policy_loss.item()
            semantic_value_loss_sum += (semantic_value_loss.item() + semantic_objective_value_loss.item())
            semantic_reward_loss_sum += (semantic_reward_loss.item() + semantic_objective_reward_loss.item())
            semantic_state_consistency_loss_sum += semantic_state_consistency_loss.item()
            objective_diversity_loss_sum += objective_diversity_loss.item()
            latent_policy_distill_loss_sum += latent_policy_distill_loss.item()
            state_consistency_loss_sum += state_consistency_loss.item()
            surface_mask_loss_sum += surface_mask_loss.item()
            surface_count_loss_sum += surface_count_loss.item()
            surface_domain_loss_sum += surface_domain_loss.item()
            surface_phase_loss_sum += surface_phase_loss.item()
            latent_policy_distill_kl_sum += latent_policy_distill_metrics["kl"]
            latent_policy_distill_agreement_sum += latent_policy_distill_metrics["top1_agreement"]
            teacher_entropy_sum += latent_policy_distill_metrics["teacher_entropy"]
            student_entropy_sum += latent_policy_distill_metrics["student_entropy"]
            state_consistency_cosine_sum += state_consistency_cosine
            surface_precision_sum += surface_metrics["legal_precision"]
            surface_recall_sum += surface_metrics["legal_recall"]
            surface_f1_sum += surface_metrics["legal_f1"]
            surface_count_mae_sum += surface_metrics["legal_count_mae"]
            surface_domain_acc_sum += surface_metrics["decision_domain_acc"]
            surface_phase_acc_sum += surface_metrics["phase_acc"]

            hidden_state = recurrent.next_hidden_state

        # Backward pass
        self.optimizer.zero_grad()
        total_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.network.parameters(), self.max_grad_norm)
        self.optimizer.step()

        policy_terms = float(unroll_steps + 1)
        reward_terms = float(max(unroll_steps, 1))
        surface_terms = float(max(unroll_steps, 1))
        return {
            "loss/total": total_loss.item(),
            "loss/policy": policy_loss_sum / policy_terms,
            "loss/semantic_policy": semantic_policy_loss_sum / policy_terms,
            "loss/latent_policy_target": latent_policy_loss_sum / policy_terms,
            "loss/value": value_loss_sum / policy_terms,
            "loss/objective_value": objective_value_loss_sum / policy_terms,
            "loss/semantic_value": semantic_value_loss_sum / policy_terms,
            "loss/reward": reward_loss_sum / reward_terms,
            "loss/objective_reward": objective_reward_loss_sum / reward_terms,
            "loss/semantic_reward": semantic_reward_loss_sum / reward_terms,
            "loss/semantic_state_consistency": semantic_state_consistency_loss_sum / surface_terms,
            "loss/objective_diversity": objective_diversity_loss_sum / policy_terms,
            "loss/latent_policy_distill": latent_policy_distill_loss_sum / policy_terms,
            "loss/state_consistency": state_consistency_loss_sum / surface_terms,
            "loss/surface_mask": surface_mask_loss_sum / surface_terms,
            "loss/surface_count": surface_count_loss_sum / surface_terms,
            "loss/surface_domain": surface_domain_loss_sum / surface_terms,
            "loss/surface_phase": surface_phase_loss_sum / surface_terms,
            "metric/latent_policy_distill_kl": latent_policy_distill_kl_sum / policy_terms,
            "metric/latent_policy_distill_top1_agreement": latent_policy_distill_agreement_sum / policy_terms,
            "metric/teacher_policy_entropy": teacher_entropy_sum / policy_terms,
            "metric/student_policy_entropy": student_entropy_sum / policy_terms,
            "metric/state_consistency_cosine": state_consistency_cosine_sum / surface_terms,
            "metric/predicted_legal_precision": surface_precision_sum / surface_terms,
            "metric/predicted_legal_recall": surface_recall_sum / surface_terms,
            "metric/predicted_legal_f1": surface_f1_sum / surface_terms,
            "metric/predicted_legal_count_mae": surface_count_mae_sum / surface_terms,
            "metric/decision_domain_acc": surface_domain_acc_sum / surface_terms,
            "metric/phase_acc": surface_phase_acc_sum / surface_terms,
        }

    def _policy_loss(
        self,
        logits: torch.Tensor,
        targets: torch.Tensor,
        action_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Cross-entropy loss for policy head."""
        # logits: [B, 80], targets: [B, 80] (soft targets from MCTS)
        if action_mask is not None:
            logits = logits.masked_fill(action_mask <= 0, -1e9)
        log_probs = torch.log_softmax(logits, dim=-1)
        loss = -(targets * log_probs).sum(dim=-1).mean()
        return loss

    def _semantic_policy_loss(
        self,
        logits: torch.Tensor,
        targets: torch.Tensor,
    ) -> torch.Tensor:
        smoothing = self.semantic_policy_label_smoothing
        if smoothing > 0.0:
            targets = (1.0 - smoothing) * targets + smoothing / float(SEMANTIC_ROLLOUT_SIZE)
        log_probs = torch.log_softmax(logits, dim=-1)
        return -(targets * log_probs).sum(dim=-1).mean()

    def _value_loss(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """Value loss using categorical support targets."""
        support_targets = scalar_to_support(
            targets.reshape(-1),
            support_size=self.network.support_size,
        ).to(logits.device)
        log_probs = torch.log_softmax(logits, dim=-1)
        loss = -(support_targets * log_probs).sum(dim=-1).mean()
        return loss

    def _reward_loss(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """Reward loss using categorical support targets."""
        support_targets = scalar_to_support(
            targets.reshape(-1),
            support_size=self.network.support_size,
        ).to(logits.device)
        log_probs = torch.log_softmax(logits, dim=-1)
        loss = -(support_targets * log_probs).sum(dim=-1).mean()
        return loss

    def _objective_value_loss(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """Multi-head value loss for survival / HP / build / resource heads."""
        batch, head_count, num_bins = logits.shape
        flat_logits = logits.reshape(batch * head_count, num_bins)
        flat_targets = targets.reshape(batch * head_count)
        support_targets = scalar_to_support(
            flat_targets,
            support_size=self.network.support_size,
        ).to(logits.device)
        log_probs = torch.log_softmax(flat_logits, dim=-1)
        loss = -(support_targets * log_probs).sum(dim=-1).mean()
        return loss

    def _objective_reward_loss(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """Multi-head reward loss for planner-aligned reward decomposition."""
        batch, head_count, num_bins = logits.shape
        flat_logits = logits.reshape(batch * head_count, num_bins)
        flat_targets = targets.reshape(batch * head_count)
        support_targets = scalar_to_support(
            flat_targets,
            support_size=self.network.support_size,
        ).to(logits.device)
        log_probs = torch.log_softmax(flat_logits, dim=-1)
        loss = -(support_targets * log_probs).sum(dim=-1).mean()
        return loss

    def _objective_head_diversity_loss(self, *component_tensors: torch.Tensor) -> torch.Tensor:
        valid = [tensor for tensor in component_tensors if isinstance(tensor, torch.Tensor) and tensor.numel() > 0]
        if not valid:
            return torch.zeros((), device=self.device)
        stacked = torch.cat([tensor.reshape(-1, NUM_OBJECTIVE_HEADS) for tensor in valid], dim=0)
        if stacked.shape[0] < 2:
            return stacked.new_zeros(())
        centered = stacked - stacked.mean(dim=0, keepdim=True)
        normalized = centered / centered.norm(dim=0, keepdim=True).clamp(min=1e-6)
        corr = normalized.transpose(0, 1) @ normalized
        corr = corr / max(normalized.shape[0], 1)
        off_diag = corr - torch.eye(NUM_OBJECTIVE_HEADS, device=corr.device, dtype=corr.dtype)
        return off_diag.pow(2).mean()

    def _latent_policy_distill_loss(
        self,
        student_logits: torch.Tensor,
        teacher_logits: torch.Tensor,
        action_mask: torch.Tensor,
    ) -> torch.Tensor:
        """Distill latent-policy logits toward observation-conditioned logits."""
        valid_rows = (action_mask > 0).any(dim=-1)
        if not valid_rows.any():
            return student_logits.new_zeros(())

        masked_student = student_logits[valid_rows].masked_fill(action_mask[valid_rows] <= 0, -1e9)
        masked_teacher = teacher_logits[valid_rows].masked_fill(action_mask[valid_rows] <= 0, -1e9)
        teacher_probs = torch.softmax(masked_teacher.detach(), dim=-1)
        student_log_probs = torch.log_softmax(masked_student, dim=-1)
        return -(teacher_probs * student_log_probs).sum(dim=-1).mean()

    def _latent_policy_distill_metrics(
        self,
        student_logits: torch.Tensor,
        teacher_logits: torch.Tensor,
        action_mask: torch.Tensor,
    ) -> dict[str, float]:
        """Return diagnostic agreement metrics between latent and conditioned policies."""
        valid_rows = (action_mask > 0).any(dim=-1)
        if not valid_rows.any():
            return {
                "kl": 0.0,
                "top1_agreement": 0.0,
                "teacher_entropy": 0.0,
                "student_entropy": 0.0,
            }

        masked_student = student_logits[valid_rows].masked_fill(action_mask[valid_rows] <= 0, -1e9)
        masked_teacher = teacher_logits[valid_rows].masked_fill(action_mask[valid_rows] <= 0, -1e9)

        teacher_probs = torch.softmax(masked_teacher.detach(), dim=-1)
        student_probs = torch.softmax(masked_student, dim=-1)
        student_log_probs = torch.log_softmax(masked_student, dim=-1)
        teacher_log_probs = torch.log_softmax(masked_teacher.detach(), dim=-1)

        kl = (teacher_probs * (teacher_log_probs - student_log_probs)).sum(dim=-1).mean().item()
        agreement = (teacher_probs.argmax(dim=-1) == student_probs.argmax(dim=-1)).float().mean().item()
        teacher_entropy = -(teacher_probs * teacher_log_probs).sum(dim=-1).mean().item()
        student_entropy = -(student_probs * student_log_probs).sum(dim=-1).mean().item()

        return {
            "kl": float(kl),
            "top1_agreement": float(agreement),
            "teacher_entropy": float(teacher_entropy),
            "student_entropy": float(student_entropy),
        }

    def _state_consistency_loss(
        self,
        student_projected: torch.Tensor,
        teacher_projected: torch.Tensor,
    ) -> tuple[torch.Tensor, float]:
        student = torch.nn.functional.normalize(student_projected, dim=-1)
        teacher = torch.nn.functional.normalize(teacher_projected.detach(), dim=-1)
        cosine = (student * teacher).sum(dim=-1)
        loss = (1.0 - cosine).mean()
        return loss, float(cosine.mean().item())

    def _surface_mask_loss(
        self,
        logits: torch.Tensor,
        target_mask: torch.Tensor,
    ) -> torch.Tensor:
        targets = target_mask.float()
        # Adaptive pos_weight: penalise false-positives proportionally to
        # the actual legal/illegal ratio in this batch.  Typical combat has
        # ~8 legal out of 80 slots, so neg/pos ≈ 9 → false positives get
        # 9× the gradient of false negatives.  Clamped to [1, 15].
        n_pos = targets.sum().clamp(min=1.0)
        n_neg = (targets.numel() - n_pos).clamp(min=1.0)
        pos_w = (n_neg / n_pos).clamp(1.0, 15.0)
        # Use per-element pos_weight via manual weighting:
        # BCE already gives per-element loss; we scale positive targets up.
        weight = torch.where(targets > 0.5, pos_w, torch.ones_like(targets))
        bce = torch.nn.functional.binary_cross_entropy_with_logits(
            logits, targets, reduction="none",
        )
        return (weight * bce).mean()

    def _surface_count_loss(
        self,
        logits: torch.Tensor,
        target_mask: torch.Tensor,
    ) -> torch.Tensor:
        predicted_count = torch.sigmoid(logits).sum(dim=-1)
        target_count = target_mask.float().sum(dim=-1)
        return torch.nn.functional.smooth_l1_loss(predicted_count, target_count)

    def _surface_domain_loss(
        self,
        logits: torch.Tensor,
        target_domain: torch.Tensor,
    ) -> torch.Tensor:
        valid_rows = target_domain.sum(dim=-1) > 0
        if not valid_rows.any():
            return logits.new_zeros(())
        targets = target_domain[valid_rows].argmax(dim=-1)
        return torch.nn.functional.cross_entropy(logits[valid_rows], targets)

    def _surface_phase_loss(
        self,
        logits: torch.Tensor,
        target_scalars: torch.Tensor,
    ) -> torch.Tensor:
        phase_targets = target_scalars[:, :NUM_PHASES]
        valid_rows = phase_targets.sum(dim=-1) > 0
        if not valid_rows.any():
            return logits.new_zeros(())
        targets = phase_targets[valid_rows].argmax(dim=-1)
        return torch.nn.functional.cross_entropy(logits[valid_rows], targets)

    def _surface_metrics(
        self,
        mask_logits: torch.Tensor,
        target_mask: torch.Tensor,
        domain_logits: torch.Tensor,
        target_domain: torch.Tensor,
        phase_logits: torch.Tensor,
        target_scalars: torch.Tensor,
    ) -> dict[str, float]:
        pred_mask = torch.sigmoid(mask_logits) >= 0.5
        pred_count = torch.sigmoid(mask_logits).sum(dim=-1)
        target_count = target_mask.sum(dim=-1)
        target_mask_bool = target_mask > 0.5
        tp = (pred_mask & target_mask_bool).sum().item()
        fp = (pred_mask & ~target_mask_bool).sum().item()
        fn = (~pred_mask & target_mask_bool).sum().item()

        precision = tp / max(tp + fp, 1.0)
        recall = tp / max(tp + fn, 1.0)
        f1 = 2.0 * precision * recall / max(precision + recall, 1e-9)

        domain_valid = target_domain.sum(dim=-1) > 0
        if domain_valid.any():
            domain_acc = (
                domain_logits[domain_valid].argmax(dim=-1)
                == target_domain[domain_valid].argmax(dim=-1)
            ).float().mean().item()
        else:
            domain_acc = 0.0

        phase_targets = target_scalars[:, :NUM_PHASES]
        phase_valid = phase_targets.sum(dim=-1) > 0
        if phase_valid.any():
            phase_acc = (
                phase_logits[phase_valid].argmax(dim=-1)
                == phase_targets[phase_valid].argmax(dim=-1)
            ).float().mean().item()
        else:
            phase_acc = 0.0

        return {
            "legal_precision": float(precision),
            "legal_recall": float(recall),
            "legal_f1": float(f1),
            "legal_count_mae": float((pred_count - target_count).abs().mean().item()),
            "decision_domain_acc": float(domain_acc),
            "phase_acc": float(phase_acc),
        }

    def save_checkpoint(self, tag: str = "") -> None:
        """Save training checkpoint."""
        checkpoint_dir = Path(self.checkpoint_dir)
        checkpoint_dir.mkdir(parents=True, exist_ok=True)

        if tag:
            name = f"muzero_{tag}"
        else:
            name = f"muzero_step_{self.total_steps:08d}"

        checkpoint_path = checkpoint_dir / name
        checkpoint_path.mkdir(parents=True, exist_ok=True)

        torch.save(self.network.state_dict(), checkpoint_path / "network.pt")
        torch.save(self.optimizer.state_dict(), checkpoint_path / "optimizer.pt")
        with (checkpoint_path / "replay_buffer.pkl").open("wb") as handle:
            pickle.dump(self.buffer.state_dict(), handle, protocol=pickle.HIGHEST_PROTOCOL)

        metadata = {
            "format": "muzero-v1",
            "total_steps": int(self.total_steps),
            "episode_count": int(self.episode_count),
            "buffer_size": len(self.buffer),
            "replay_buffer_saved": True,
            "latent_policy_distill_weight": float(self.latent_policy_distill_weight),
            "latent_policy_target_weight": float(self.latent_policy_target_weight),
            "objective_value_weight": float(self.objective_value_weight),
            "objective_reward_weight": float(self.objective_reward_weight),
            "semantic_policy_weight": float(self.semantic_policy_weight),
            "semantic_value_weight": float(self.semantic_value_weight),
            "semantic_reward_weight": float(self.semantic_reward_weight),
            "semantic_state_consistency_weight": float(self.semantic_state_consistency_weight),
            "objective_diversity_weight": float(self.objective_diversity_weight),
            "semantic_policy_label_smoothing": float(self.semantic_policy_label_smoothing),
            "semantic_training_enabled": bool(
                self.semantic_policy_weight > 0.0
                or self.semantic_value_weight > 0.0
                or self.semantic_reward_weight > 0.0
                or self.semantic_state_consistency_weight > 0.0
                or self.objective_diversity_weight > 0.0
            ),
            "state_consistency_weight": float(self.state_consistency_weight),
            "surface_mask_weight": float(self.surface_mask_weight),
            "surface_count_weight": float(self.surface_count_weight),
            "surface_domain_weight": float(self.surface_domain_weight),
            "surface_phase_weight": float(self.surface_phase_weight),
            "mcts": {
                "num_simulations": int(self.mcts.num_simulations),
                "max_sampled_actions": int(self.mcts.max_sampled_actions),
                "min_expanded_actions": int(self.mcts.min_expanded_actions),
                "root_max_sampled_actions": int(self.mcts.root_max_sampled_actions),
                "child_max_sampled_actions": int(self.mcts.child_max_sampled_actions),
                "root_min_expanded_actions": int(self.mcts.root_min_expanded_actions),
                "child_min_expanded_actions": int(self.mcts.child_min_expanded_actions),
                "prior_floor": float(self.mcts.prior_floor),
                "relative_prior_floor": float(self.mcts.relative_prior_floor),
                "surface_mask_blend": float(self.mcts.surface_mask_blend),
                "end_turn_prior_bias": float(self.mcts.end_turn_prior_bias),
                "objective_value_blend": float(self.mcts.objective_value_blend),
                "objective_prior_blend": float(self.mcts.objective_prior_blend),
                "semantic_switch_depth": int(self.mcts.semantic_switch_depth),
                "enable_semantic_rollout": bool(self.mcts._semantic_rollout_enabled),
                "semantic_rollout_chain_steps": int(self.mcts.semantic_rollout_chain_steps),
                "semantic_revisit_bonus": float(self.mcts.semantic_revisit_bonus),
                "root_bias_enabled": bool(self.mcts._root_bias_enabled),
                "root_bias_min_scale": float(self.mcts.root_bias_min_scale),
                "root_bias_decay_steps": int(self.mcts.root_bias_decay_steps),
            },
        }
        (checkpoint_path / "metadata.json").write_text(json.dumps(metadata, indent=2))
        print(f"[checkpoint] Saved to {checkpoint_path}")


def main():
    """Main training loop."""
    parser = argparse.ArgumentParser(description="MuZero Training for STS2")

    # Core training
    parser.add_argument("--total-timesteps", type=int, default=100_000)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--unroll-steps", type=int, default=5)
    parser.add_argument("--num-simulations", type=int, default=50)
    parser.add_argument("--max-sampled-actions", type=int, default=32)
    parser.add_argument("--min-expanded-actions", type=int, default=4)
    parser.add_argument("--root-max-sampled-actions", type=int, default=None)
    parser.add_argument("--child-max-sampled-actions", type=int, default=8)
    parser.add_argument("--root-min-expanded-actions", type=int, default=None)
    parser.add_argument("--child-min-expanded-actions", type=int, default=4)
    parser.add_argument("--prior-floor", type=float, default=0.0)
    parser.add_argument("--relative-prior-floor", type=float, default=0.0)
    parser.add_argument("--surface-mask-blend", type=float, default=0.15,
                        help="Blend predicted latent legal-mask into child expansion ranking.")
    parser.add_argument("--end-turn-prior-bias", type=float, default=-1.0,
                        help="Root-only logit bias for end_turn when energy remains and a playable card exists.")
    parser.add_argument("--objective-value-blend", type=float, default=0.70,
                        help="Blend planner-objective value heads into MCTS backup/search value.")
    parser.add_argument("--objective-prior-blend", type=float, default=0.30,
                        help="Scale objective-aware semantic prior bonuses at the MCTS root.")
    parser.add_argument("--semantic-switch-depth", type=int, default=2,
                        help="Depth after which MCTS switches from concrete rollout to semantic rollout.")
    parser.add_argument("--disable-semantic-rollout", action="store_true",
                        help="Ablation: keep MCTS in pure concrete rollout mode.")
    parser.add_argument("--semantic-rollout-chain-steps", type=int, default=1,
                        help="Extra semantic plies to auto-drill inside the same simulation after crossing the semantic switch.")
    parser.add_argument("--semantic-revisit-bonus", type=float, default=0.15,
                        help="Soft UCB bonus for semantic nodes so search revisits abstract branches instead of spending all budget breadth-first.")
    parser.add_argument("--disable-semantic-training", action="store_true",
                        help="Ablation: zero out semantic rollout losses during optimization.")
    parser.add_argument("--root-bias-min-scale", type=float, default=0.33,
                        help="Final scale of root prior bias after linear decay.")
    parser.add_argument("--root-bias-decay-steps", type=int, default=200000,
                        help="Training steps over which root prior bias linearly decays.")
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--buffer-size", type=int, default=100_000)
    parser.add_argument("--min-buffer-size", type=int, default=500)
    parser.add_argument("--train-every", type=int, default=50,
                        help="Train every N environment steps")
    parser.add_argument("--updates-per-train", type=int, default=3,
                        help="Gradient updates per training call")
    parser.add_argument("--checkpoint-freq", type=int, default=2048)
    parser.add_argument("--resume-from", type=str, default=None,
                        help="Resume MuZero training from a checkpoint directory")
    parser.add_argument("--resume-without-buffer", action="store_true", default=False,
                        help="Resume weights/optimizer from checkpoint but start with an empty replay buffer.")
    parser.add_argument("--latent-policy-distill-weight", type=float, default=0.25,
                        help="Distill latent search policy toward observation-conditioned policy.")
    parser.add_argument("--latent-policy-target-weight", type=float, default=0.5,
                        help="Direct supervised target loss on latent-policy logits.")
    parser.add_argument("--objective-value-weight", type=float, default=1.0,
                        help="Loss weight for multi-head objective value prediction.")
    parser.add_argument("--objective-reward-weight", type=float, default=0.75,
                        help="Loss weight for multi-head objective reward prediction.")
    parser.add_argument("--semantic-policy-weight", type=float, default=1.0,
                        help="Loss weight for semantic rollout policy supervision.")
    parser.add_argument("--semantic-value-weight", type=float, default=1.0,
                        help="Loss weight for semantic rollout value supervision.")
    parser.add_argument("--semantic-reward-weight", type=float, default=0.75,
                        help="Loss weight for semantic rollout reward supervision.")
    parser.add_argument("--semantic-state-consistency-weight", type=float, default=2.0,
                        help="Consistency weight between semantic rollout latent and projected teacher latent.")
    parser.add_argument("--objective-diversity-weight", type=float, default=0.05,
                        help="Regularize objective heads against collapse into the same signal.")
    parser.add_argument("--semantic-policy-label-smoothing", type=float, default=0.02,
                        help="Label smoothing applied to semantic rollout policy targets.")
    parser.add_argument("--state-consistency-weight", type=float, default=2.0,
                        help="Weight for hidden-state consistency between dynamics rollout and representation(next_obs).")
    parser.add_argument("--surface-mask-weight", type=float, default=1.5,
                        help="Weight for next-step legal-mask prediction loss.")
    parser.add_argument("--surface-count-weight", type=float, default=0.25,
                        help="Weight for predicted legal-count calibration loss.")
    parser.add_argument("--surface-domain-weight", type=float, default=0.15,
                        help="Weight for next-step decision-domain prediction loss.")
    parser.add_argument("--surface-phase-weight", type=float, default=0.1,
                        help="Weight for next-step phase prediction loss.")

    # Directories
    parser.add_argument("--log-dir", type=str, default="runs")
    parser.add_argument("--checkpoint-dir", type=str, default="checkpoints")

    # Environment setup
    parser.add_argument("--combat-sandbox", action="store_true", default=False)
    parser.add_argument("--encounter-pool", type=str, default=None)
    parser.add_argument("--combat-encounter-tiers", type=str, default=None,
                        help="Comma-separated tier filter for combat sandbox snapshots / encounters: weak,normal,elite,boss")
    parser.add_argument("--combat-snapshot-dataset", type=str, default=None)
    parser.add_argument(
        "--combat-curated-subset",
        type=str,
        default=DEFAULT_CURATED_COMBINED_SUBSET,
        choices=sorted(VALID_CURATED_COMBINED_SUBSETS),
        help=(
            "When --combat-snapshot-dataset points at a curated combat root/combined dir, "
            "resolve this subset. Default keeps all human rows plus only local runs that cleared Act 1."
        ),
    )
    parser.add_argument("--combat-snapshot-split", type=str, default="train")
    parser.add_argument(
        "--combat-snapshot-sample-mode",
        type=str,
        default="encounter_balanced",
        choices=["row_uniform", "encounter_balanced", "tier_weighted_encounter_balanced"],
        help="Sampling mode for combat snapshot curriculum.",
    )
    parser.add_argument(
        "--combat-tier-weights",
        type=str,
        default=None,
        help="Optional tier sampling weights, e.g. weak=0.6,normal=0.4",
    )
    parser.add_argument(
        "--combat-encounter-weights",
        type=str,
        default=None,
        help="Optional encounter weights, e.g. ENCOUNTER.BOWLBUGS_WEAK=3,ENCOUNTER.TUNNELER_WEAK=3",
    )
    parser.add_argument("--character", type=str, default=None,
                        help="Character to use for training")
    parser.add_argument("--defensive-buffs", action="store_true", default=False,
                        help="Apply defensive buffs on env/reset (mainly for debugging).")
    parser.add_argument("--combat-snapshot-character", type=str, default=None)
    parser.add_argument("--session-file", type=str, default=None)
    parser.add_argument("--session-files", type=str, default=None)
    parser.add_argument("--n-envs", type=int, default=1)
    parser.add_argument("--device", type=str, default="cpu")

    # Discount factors
    parser.add_argument("--discount", type=float, default=0.997)
    parser.add_argument("--n-step-return", type=int, default=10)

    args = parser.parse_args()

    # Setup directories
    Path(args.log_dir).mkdir(parents=True, exist_ok=True)
    Path(args.checkpoint_dir).mkdir(parents=True, exist_ok=True)

    # Resolve session files
    session_files = resolve_training_session_files(
        n_envs=args.n_envs,
        session_file=args.session_file,
        session_files=parse_session_files(args.session_files),
    )

    # For now, only support single environment
    if args.n_envs != 1:
        print("[warning] Only single environment (--n-envs 1) is currently fully supported for MuZero")

    session_file = session_files[0] if session_files else None
    supported_encounter_ids: set[str] | None = None

    encounter_tiers = parse_encounter_tiers(args.combat_encounter_tiers)
    tier_weights = parse_tier_weights(args.combat_tier_weights)
    encounter_weights = parse_encounter_weights(args.combat_encounter_weights)
    snapshot_pool = None
    encounter_pool: list[str] = []
    if args.combat_sandbox:
        if args.encounter_pool:
            encounter_pool = parse_encounter_pool(args.encounter_pool)
        else:
            encounter_pool = parse_encounter_pool(DEFAULT_COMBAT_SANDBOX_TRAIN_POOL)
        if encounter_tiers:
            encounter_pool = [
                encounter_id
                for encounter_id in encounter_pool
                if infer_encounter_tier(encounter_id) in encounter_tiers
            ]

    # Setup snapshot pool if provided
    if args.combat_sandbox and args.combat_snapshot_dataset:
        print(f"[setup] Loading combat snapshot dataset from {args.combat_snapshot_dataset}...")
        supported_encounter_ids = _get_live_supported_encounter_ids(session_file)
        print(f"[setup] Live combat catalog supports {len(supported_encounter_ids)} encounters")
        snapshot_pool = CombatSnapshotPool.from_path(
            args.combat_snapshot_dataset,
            curated_subset=args.combat_curated_subset,
            split=args.combat_snapshot_split or None,
            character=args.combat_snapshot_character,
            encounter_tiers=encounter_tiers or None,
            sample_mode=args.combat_snapshot_sample_mode,
            tier_weights=tier_weights or None,
            encounter_weights=encounter_weights or None,
            supported_encounter_ids=supported_encounter_ids,
        )
        print(f"[setup] Loaded {len(snapshot_pool)} snapshots")
        print(f"[setup] Snapshot summary: {snapshot_pool.summary()}")

    print(f"[setup] Creating training environment...")
    env = build_train_env(
        env_index=0,
        session_file=session_file,
        combat_sandbox=args.combat_sandbox,
        character=args.character,
        defensive_buffs=args.defensive_buffs,
        encounter_id=None,
        encounter_pool=encounter_pool,
        snapshot_pool=snapshot_pool,
        reset_timeout_ms=15000 if args.combat_sandbox else 60000,
        step_timeout_ms=20000,
    )

    # Create network, MCTS, buffer
    print(f"[setup] Creating MuZero network...")
    network = MuZeroNetwork()
    network.to(args.device)

    mcts = MCTS(
        num_simulations=args.num_simulations,
        discount=args.discount,
        max_sampled_actions=args.max_sampled_actions,
        min_expanded_actions=args.min_expanded_actions,
        root_max_sampled_actions=args.root_max_sampled_actions,
        child_max_sampled_actions=args.child_max_sampled_actions,
        root_min_expanded_actions=args.root_min_expanded_actions,
        child_min_expanded_actions=args.child_min_expanded_actions,
        prior_floor=args.prior_floor,
        relative_prior_floor=args.relative_prior_floor,
        surface_mask_blend=args.surface_mask_blend,
        end_turn_prior_bias=args.end_turn_prior_bias,
        objective_value_blend=args.objective_value_blend,
        objective_prior_blend=args.objective_prior_blend,
        semantic_switch_depth=args.semantic_switch_depth,
        enable_semantic_rollout=not args.disable_semantic_rollout,
        semantic_rollout_chain_steps=args.semantic_rollout_chain_steps,
        semantic_revisit_bonus=args.semantic_revisit_bonus,
        root_bias_min_scale=args.root_bias_min_scale,
        root_bias_decay_steps=args.root_bias_decay_steps,
    )

    buffer = MuZeroReplayBuffer(capacity=args.buffer_size)

    optimizer = optim.Adam(
        network.parameters(),
        lr=args.lr,
        weight_decay=args.weight_decay,
    )

    trainer = MuZeroTrainer(
        network=network,
        mcts=mcts,
        buffer=buffer,
        env=env,
        optimizer=optimizer,
        device=args.device,
        discount=args.discount,
        n_step_return=args.n_step_return,
        latent_policy_distill_weight=args.latent_policy_distill_weight,
        latent_policy_target_weight=args.latent_policy_target_weight,
        objective_value_weight=args.objective_value_weight,
        objective_reward_weight=args.objective_reward_weight,
        semantic_policy_weight=0.0 if args.disable_semantic_training else args.semantic_policy_weight,
        semantic_value_weight=0.0 if args.disable_semantic_training else args.semantic_value_weight,
        semantic_reward_weight=0.0 if args.disable_semantic_training else args.semantic_reward_weight,
        semantic_state_consistency_weight=0.0 if args.disable_semantic_training else args.semantic_state_consistency_weight,
        objective_diversity_weight=0.0 if args.disable_semantic_training else args.objective_diversity_weight,
        semantic_policy_label_smoothing=args.semantic_policy_label_smoothing,
        state_consistency_weight=args.state_consistency_weight,
        surface_mask_weight=args.surface_mask_weight,
        surface_count_weight=args.surface_count_weight,
        surface_domain_weight=args.surface_domain_weight,
        surface_phase_weight=args.surface_phase_weight,
        log_dir=args.log_dir,
        checkpoint_dir=args.checkpoint_dir,
    )

    if args.resume_from:
        print(f"[resume] Loading checkpoint from {args.resume_from}...")
        resume_metadata = load_resume_checkpoint(
            args.resume_from,
            network=network,
            optimizer=optimizer,
            buffer=buffer,
            device=args.device,
            load_buffer=not args.resume_without_buffer,
        )
        trainer.total_steps = int(resume_metadata.get("total_steps", 0))
        trainer.episode_count = int(resume_metadata.get("episode_count", 0))
        print(
            f"[resume] Loaded total_steps={trainer.total_steps} "
            f"episode_count={trainer.episode_count} buffer={len(buffer)}"
        )
        if args.total_timesteps <= trainer.total_steps:
            raise ValueError(
                f"--total-timesteps ({args.total_timesteps}) must be greater than resumed total_steps "
                f"({trainer.total_steps})."
            )

    mode_name = "combat_sandbox" if args.combat_sandbox else "full_run"
    semantic_training_enabled = not args.disable_semantic_training and (
        args.semantic_policy_weight > 0.0
        or args.semantic_value_weight > 0.0
        or args.semantic_reward_weight > 0.0
        or args.semantic_state_consistency_weight > 0.0
        or args.objective_diversity_weight > 0.0
    )
    root_bias_enabled = (args.objective_prior_blend > 0.0) or (args.end_turn_prior_bias != 0.0)
    print(f"[setup] Starting training with {args.total_timesteps} timesteps")
    print(f"[setup] Mode: {mode_name}")
    print(
        "[setup] Semantic rollout: "
        f"search={'on' if not args.disable_semantic_rollout else 'off'} "
        f"(switch_depth={args.semantic_switch_depth}, "
        f"chain_steps={args.semantic_rollout_chain_steps}, "
        f"revisit_bonus={args.semantic_revisit_bonus:.2f}) | "
        f"training={'on' if semantic_training_enabled else 'off'}"
    )
    print(
        "[setup] Root prior bias: "
        f"{'on' if root_bias_enabled else 'off'} "
        f"(objective_prior_blend={args.objective_prior_blend:.2f}, "
        f"end_turn_bias={args.end_turn_prior_bias:.2f}, "
        f"min_scale={args.root_bias_min_scale:.2f}, "
        f"decay_steps={args.root_bias_decay_steps})"
    )
    if encounter_tiers:
        print(f"[setup] Encounter tiers: {encounter_tiers}")
    if tier_weights:
        print(f"[setup] Encounter tier weights: {tier_weights}")
    if encounter_weights:
        print(f"[setup] Encounter weights: {encounter_weights}")
    if snapshot_pool is not None:
        print(f"[setup] Snapshot sample mode: {args.combat_snapshot_sample_mode}")
    if encounter_pool:
        print(f"[setup] Encounter pool: {encounter_pool}")

    # ``--train-every`` is documented as environment-step cadence, not episode cadence.
    # Use a step-based trigger so variable-length combats do not delay updates by dozens
    # of episodes and so resumed runs keep a stable optimizer rhythm.
    last_train_step = int(trainer.total_steps)

    # Training loop
    try:
        while trainer.total_steps < args.total_timesteps:
            # Self-play episode
            temperature = trainer.compute_temperature(trainer.total_steps, args.total_timesteps)
            ep_reward, ep_length = trainer.self_play_episode(temperature=temperature)

            trainer.total_steps += ep_length
            trainer.writer.add_scalar("episode/reward", ep_reward, trainer.episode_count)
            trainer.writer.add_scalar("episode/length", ep_length, trainer.episode_count)
            trainer.writer.add_scalar("schedule/temperature", temperature, trainer.total_steps)

            # Train on buffer
            train_updates_triggered = 0
            while len(buffer) >= args.min_buffer_size and (trainer.total_steps - last_train_step) >= args.train_every:
                for _ in range(args.updates_per_train):
                    losses = trainer.train_step(
                        batch_size=args.batch_size,
                        unroll_steps=args.unroll_steps,
                    )
                    for key, value in losses.items():
                        trainer.writer.add_scalar(key, value, trainer.total_steps)
                last_train_step += int(args.train_every)
                train_updates_triggered += int(args.updates_per_train)
            if train_updates_triggered > 0:
                print(
                    f"[train] Update | Episode {trainer.episode_count} | "
                    f"Steps {trainer.total_steps} | Buffer {len(buffer)} | "
                    f"OptimizerSteps {train_updates_triggered}"
                )
                trainer.writer.flush()

            # Logging
            if (trainer.episode_count % 10) == 0:
                print(
                    f"[train] Episode {trainer.episode_count} | "
                    f"Steps {trainer.total_steps}/{args.total_timesteps} | "
                    f"Reward {ep_reward:+.4f} | "
                    f"Length {ep_length} | "
                    f"Buffer {len(buffer)} | "
                    f"T {temperature:.3f}"
                )
                episode_metrics = trainer.last_episode_metrics or {}
                decision_counts = episode_metrics.get("decision_counts") if isinstance(episode_metrics, dict) else {}
                domain_search_means = episode_metrics.get("domain_search_means") if isinstance(episode_metrics, dict) else {}
                domain_family_rates = episode_metrics.get("domain_family_rates") if isinstance(episode_metrics, dict) else {}
                if isinstance(decision_counts, dict) and decision_counts:
                    print(
                        "[train] Decisions | "
                        + " ".join(
                            f"{domain}={int(decision_counts.get(domain, 0))}"
                            for domain in DECISION_DOMAINS
                        )
                    )
                if mode_name == "full_run" and isinstance(domain_search_means, dict):
                    planner_parts: list[str] = []
                    for domain in ("build", "route"):
                        metrics = domain_search_means.get(domain) if isinstance(domain_search_means.get(domain), dict) else None
                        if not metrics:
                            continue
                        segment = (
                            f"{domain}[root={float(metrics.get('root_candidates', 0.0)):.2f} "
                            f"expand={float(metrics.get('mean_expanded_children', 0.0)):.2f} "
                            f"leaf={float(metrics.get('mean_leaf_depth', 0.0)):.2f} "
                            f"max={float(metrics.get('max_search_depth', 0.0)):.2f}"
                        )
                        family_rates = domain_family_rates.get(domain) if isinstance(domain_family_rates, dict) else None
                        if isinstance(family_rates, dict) and family_rates:
                            top_family, top_rate = max(family_rates.items(), key=lambda item: item[1])
                            segment += f" top={top_family}:{float(top_rate):.2f}"
                        segment += "]"
                        planner_parts.append(segment)
                    if planner_parts:
                        print("[train] PlannerSearch | " + " | ".join(planner_parts))

            # Checkpointing
            if (trainer.total_steps % args.checkpoint_freq) < ep_length:
                trainer.save_checkpoint()

            trainer.writer.add_scalar("buffer/size", len(buffer), trainer.total_steps)
            if (trainer.episode_count % 5) == 0:
                trainer.writer.flush()

    except KeyboardInterrupt:
        print("[interrupt] Caught KeyboardInterrupt, saving emergency checkpoint...")
        trainer.save_checkpoint(tag="crash")
        raise

    print(f"[train] Finished training after {trainer.total_steps} steps")
    trainer.save_checkpoint(tag="final")
    trainer.writer.close()


if __name__ == "__main__":
    main()
