"""Legacy MuZero training path for Slay the Spire 2 combat scenarios.

Usage:
    python train_muzero.py --total-timesteps 100000 --combat-sandbox \\
        --encounter-pool "ENCOUNTER.SLIMES_WEAK,ENCOUNTER.SHRINKER_BEETLE_WEAK"

This implements self-play MCTS + replay buffer training for the MuZero network.
The mainline online acting stack now uses ``train_attention_policy.py``.
"""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict, deque
import json
import math
import os
import pickle
import queue
import re
import shutil
import signal
import threading
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
from sts2_env.bridge_client import BridgeClient, BridgeError
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
from sts2_env.path_utils import normalize_path_str, resolve_torch_device, running_in_wsl
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
DEFAULT_COMBAT_SNAPSHOT_SAMPLE_MODE = "tier_weighted_encounter_balanced"
DEFAULT_COMBAT_ENCOUNTER_TIER_WEIGHTS = "weak=0.70,normal=1.00,elite=1.45,boss=1.75"
DEFAULT_COMBAT_ENCOUNTER_PRIORITY_WEIGHTS = ",".join([
    "ENCOUNTER.CONSTRUCT_MENAGERIE_NORMAL=2.20",
    "ENCOUNTER.SLUMBERING_BEETLE_NORMAL=1.90",
    "ENCOUNTER.OVICOPTER_NORMAL=1.80",
    "ENCOUNTER.SLIMED_BERSERKER_NORMAL=1.60",
    "ENCOUNTER.THE_OBSCURA_NORMAL=1.60",
    "ENCOUNTER.SOUL_NEXUS_ELITE=2.20",
    "ENCOUNTER.KNIGHTS_ELITE=2.00",
    "ENCOUNTER.PHROG_PARASITE_ELITE=1.85",
    "ENCOUNTER.BYGONE_EFFIGY_ELITE=1.75",
    "ENCOUNTER.DECIMILLIPEDE_ELITE=1.70",
    "ENCOUNTER.CEREMONIAL_BEAST_BOSS=2.60",
    "ENCOUNTER.KAISER_CRAB_BOSS=2.40",
    "ENCOUNTER.THE_KIN_BOSS=2.20",
    "ENCOUNTER.THE_INSATIABLE_BOSS=2.00",
    "ENCOUNTER.KNOWLEDGE_DEMON_BOSS=1.70",
])
DEFAULT_REPLAY_ENCOUNTER_TIER_WEIGHTS = "weak=0.55,normal=1.15,elite=2.10,boss=2.40"
DEFAULT_REPLAY_ENCOUNTER_PRIORITY_WEIGHTS = ",".join([
    "ENCOUNTER.CONSTRUCT_MENAGERIE_NORMAL=2.60",
    "ENCOUNTER.SLUMBERING_BEETLE_NORMAL=2.20",
    "ENCOUNTER.OVICOPTER_NORMAL=2.00",
    "ENCOUNTER.SLIMED_BERSERKER_NORMAL=1.80",
    "ENCOUNTER.THE_OBSCURA_NORMAL=1.80",
    "ENCOUNTER.SOUL_NEXUS_ELITE=2.80",
    "ENCOUNTER.KNIGHTS_ELITE=2.40",
    "ENCOUNTER.PHROG_PARASITE_ELITE=2.20",
    "ENCOUNTER.BYGONE_EFFIGY_ELITE=2.00",
    "ENCOUNTER.DECIMILLIPEDE_ELITE=1.90",
    "ENCOUNTER.CEREMONIAL_BEAST_BOSS=4.00",
    "ENCOUNTER.KAISER_CRAB_BOSS=3.60",
    "ENCOUNTER.THE_KIN_BOSS=3.20",
    "ENCOUNTER.THE_INSATIABLE_BOSS=2.80",
    "ENCOUNTER.KNOWLEDGE_DEMON_BOSS=2.20",
])
DEFAULT_RECENT_TAIL_WINDOWS = (64, 256)
DEFAULT_RECENT_TAIL_TRACKED_ENCOUNTERS = ",".join([
    "ENCOUNTER.CONSTRUCT_MENAGERIE_NORMAL",
    "ENCOUNTER.SLUMBERING_BEETLE_NORMAL",
    "ENCOUNTER.OVICOPTER_NORMAL",
    "ENCOUNTER.SOUL_NEXUS_ELITE",
    "ENCOUNTER.KNIGHTS_ELITE",
    "ENCOUNTER.PHROG_PARASITE_ELITE",
    "ENCOUNTER.CEREMONIAL_BEAST_BOSS",
    "ENCOUNTER.KAISER_CRAB_BOSS",
    "ENCOUNTER.THE_KIN_BOSS",
    "ENCOUNTER.THE_INSATIABLE_BOSS",
    "ENCOUNTER.KNOWLEDGE_DEMON_BOSS",
])

BUILD_ROUTE_SETTLEMENT_FAMILIES = frozenset({
    "map",
    "reward",
    "card_reward",
    "shop",
    "rest",
    "smith",
    "deck_upgrade",
    "event_option",
})
WASTEFUL_REWARD_PHASES = frozenset({"reward", "card_reward"})
TRIVIAL_BUILD_FAST_PATH_REASONS = (
    "reward_gold",
    "reward_potion",
    "proceed_only",
    "startup_only",
)
TRIVIAL_BUILD_FAST_PATH_COMPLEX_FAMILIES = frozenset({
    "map",
    "card_reward",
    "shop",
    "rest",
    "smith",
    "deck_upgrade",
    "event_option",
    "treasure_relic",
})


class NullSummaryWriter:
    """No-op SummaryWriter used by async combat actors."""

    def add_scalar(self, *args, **kwargs) -> None:
        return None

    def flush(self) -> None:
        return None

    def close(self) -> None:
        return None


class RecentCombatMonitor:
    """Rolling recent-tail combat monitor for sandbox convergence checks."""

    def __init__(
        self,
        *,
        windows: tuple[int, ...] = DEFAULT_RECENT_TAIL_WINDOWS,
        tracked_encounters: list[str] | tuple[str, ...] | None = None,
        min_samples: int = 4,
    ) -> None:
        resolved_windows = sorted({max(int(window), 1) for window in windows if int(window) > 0})
        self.windows = tuple(resolved_windows or DEFAULT_RECENT_TAIL_WINDOWS)
        self.max_window = max(self.windows)
        self.min_samples = max(int(min_samples), 1)
        self.tracked_encounters = tuple(
            encounter_id
            for encounter_id in (tracked_encounters or [])
            if str(encounter_id).strip()
        )
        self.hard_normal_encounters = tuple(
            encounter_id for encounter_id in self.tracked_encounters if infer_encounter_tier(encounter_id) == "normal"
        )
        self.hard_elite_encounters = tuple(
            encounter_id for encounter_id in self.tracked_encounters if infer_encounter_tier(encounter_id) == "elite"
        )
        self._episodes: deque[dict[str, Any]] = deque(maxlen=self.max_window)

    @staticmethod
    def _classify_outcome(metadata: dict[str, Any]) -> str:
        if bool(metadata.get("truncated", False)):
            return "truncated"
        if float(metadata.get("death_floor", 0.0) or 0.0) > 0.0:
            return "loss"
        if bool(metadata.get("terminated", False)):
            return "win"
        return "other"

    @staticmethod
    def _safe_tag(value: str) -> str:
        sanitized = re.sub(r"[^0-9a-zA-Z]+", "_", str(value or "").strip().lower())
        return sanitized.strip("_") or "unknown"

    @staticmethod
    def _aggregate(entries: list[dict[str, Any]]) -> dict[str, Any]:
        total = len(entries)
        if total <= 0:
            return {
                "episodes": 0,
                "win_rate": 0.0,
                "loss_rate": 0.0,
                "trunc_rate": 0.0,
                "other_rate": 0.0,
                "reward_mean": 0.0,
                "reward_median": 0.0,
                "length_mean": 0.0,
            }

        outcome_counter = Counter(entry["outcome"] for entry in entries)
        rewards = np.asarray([float(entry.get("reward", 0.0) or 0.0) for entry in entries], dtype=np.float32)
        lengths = np.asarray([float(entry.get("length", 0.0) or 0.0) for entry in entries], dtype=np.float32)
        return {
            "episodes": total,
            "win_rate": float(outcome_counter.get("win", 0) / total),
            "loss_rate": float(outcome_counter.get("loss", 0) / total),
            "trunc_rate": float(outcome_counter.get("truncated", 0) / total),
            "other_rate": float(outcome_counter.get("other", 0) / total),
            "reward_mean": float(rewards.mean()) if rewards.size else 0.0,
            "reward_median": float(np.median(rewards)) if rewards.size else 0.0,
            "length_mean": float(lengths.mean()) if lengths.size else 0.0,
        }

    def record_episode(self, metadata: dict[str, Any] | None) -> dict[int, dict[str, Any]]:
        if not isinstance(metadata, dict):
            return {}
        encounter_id = str(metadata.get("encounter_id") or "").strip()
        encounter_tier = str(metadata.get("encounter_tier") or infer_encounter_tier(encounter_id)).strip().lower()
        entry = {
            "encounter_id": encounter_id,
            "encounter_tier": encounter_tier,
            "outcome": self._classify_outcome(metadata),
            "reward": float(
                metadata.get("episode_total_reward", metadata.get("episode_reward", 0.0)) or 0.0
            ),
            "length": int(metadata.get("episode_length", 0) or 0),
        }
        self._episodes.append(entry)
        return self.snapshot()

    def snapshot(self) -> dict[int, dict[str, Any]]:
        entries = list(self._episodes)
        results: dict[int, dict[str, Any]] = {}
        for window in self.windows:
            subset = entries[-window:]
            stats = self._aggregate(subset)
            tiers: dict[str, dict[str, Any]] = {}
            for tier in ("weak", "normal", "elite", "boss"):
                tier_entries = [entry for entry in subset if entry.get("encounter_tier") == tier]
                tier_stats = self._aggregate(tier_entries)
                tier_stats["sample_share"] = float(len(tier_entries) / len(subset)) if subset else 0.0
                tiers[tier] = tier_stats

            groups = {
                "hard_normal": self._aggregate(
                    [entry for entry in subset if entry.get("encounter_id") in self.hard_normal_encounters]
                ),
                "hard_elite": self._aggregate(
                    [entry for entry in subset if entry.get("encounter_id") in self.hard_elite_encounters]
                ),
            }

            tracked: dict[str, dict[str, Any]] = {}
            for encounter_id in self.tracked_encounters:
                encounter_entries = [entry for entry in subset if entry.get("encounter_id") == encounter_id]
                tracked[encounter_id] = self._aggregate(encounter_entries)

            stats["tiers"] = tiers
            stats["groups"] = groups
            stats["tracked_encounters"] = tracked
            results[int(window)] = stats
        return results

    def log_to_writer(self, writer: SummaryWriter | NullSummaryWriter, episode_step: int) -> dict[int, dict[str, Any]]:
        snapshot = self.snapshot()
        for window, stats in snapshot.items():
            prefix = f"recent_tail/{window}"
            writer.add_scalar(f"{prefix}/win_rate", float(stats.get("win_rate", 0.0)), episode_step)
            writer.add_scalar(f"{prefix}/loss_rate", float(stats.get("loss_rate", 0.0)), episode_step)
            writer.add_scalar(f"{prefix}/trunc_rate", float(stats.get("trunc_rate", 0.0)), episode_step)
            writer.add_scalar(f"{prefix}/reward_mean", float(stats.get("reward_mean", 0.0)), episode_step)
            writer.add_scalar(f"{prefix}/reward_median", float(stats.get("reward_median", 0.0)), episode_step)
            writer.add_scalar(f"{prefix}/length_mean", float(stats.get("length_mean", 0.0)), episode_step)

            tiers = stats.get("tiers") if isinstance(stats.get("tiers"), dict) else {}
            for tier, tier_stats in tiers.items():
                writer.add_scalar(
                    f"{prefix}/{tier}_win_rate",
                    float(tier_stats.get("win_rate", 0.0)),
                    episode_step,
                )
                writer.add_scalar(
                    f"{prefix}/{tier}_sample_share",
                    float(tier_stats.get("sample_share", 0.0)),
                    episode_step,
                )

            groups = stats.get("groups") if isinstance(stats.get("groups"), dict) else {}
            for group_name in ("hard_normal", "hard_elite"):
                group_stats = groups.get(group_name) if isinstance(groups.get(group_name), dict) else None
                if not group_stats:
                    continue
                writer.add_scalar(
                    f"{prefix}/{group_name}_win_rate",
                    float(group_stats.get("win_rate", 0.0)),
                    episode_step,
                )
                writer.add_scalar(
                    f"{prefix}/{group_name}_sample_count",
                    float(group_stats.get("episodes", 0.0)),
                    episode_step,
                )

            tracked = stats.get("tracked_encounters") if isinstance(stats.get("tracked_encounters"), dict) else {}
            for encounter_id, encounter_stats in tracked.items():
                if int(encounter_stats.get("episodes", 0) or 0) < self.min_samples:
                    continue
                safe_encounter = self._safe_tag(encounter_id.replace("ENCOUNTER.", ""))
                writer.add_scalar(
                    f"{prefix}/encounter_{safe_encounter}_win_rate",
                    float(encounter_stats.get("win_rate", 0.0)),
                    episode_step,
                )
        return snapshot


class EpisodeCaptureBuffer:
    """Lightweight buffer that only captures the most recent completed trajectory."""

    def __init__(self) -> None:
        self.latest_trajectory: GameTrajectory | None = None

    def save_episode(
        self,
        trajectory: GameTrajectory,
        *,
        discount: float = 0.997,
        n_steps: int = 10,
    ) -> None:
        self.latest_trajectory = trajectory

    def pop_latest(self) -> GameTrajectory | None:
        trajectory = self.latest_trajectory
        self.latest_trajectory = None
        return trajectory

    def __len__(self) -> int:
        return int(len(self.latest_trajectory) if self.latest_trajectory is not None else 0)


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
    return [
        normalize_path_str(entry.strip()) or entry.strip()
        for entry in raw.split(",")
        if entry.strip()
    ]


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


def parse_int_list(raw: str | None) -> tuple[int, ...]:
    """Parse comma-separated positive integers."""
    if not raw:
        return ()
    values: list[int] = []
    for chunk in raw.split(","):
        entry = chunk.strip()
        if not entry:
            continue
        try:
            value = int(entry)
        except ValueError as exc:
            raise ValueError(f"Invalid integer entry {entry!r}.") from exc
        if value <= 0:
            raise ValueError(f"Expected positive integer in list, got {value}.")
        values.append(value)
    return tuple(values)


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
    combat_sandbox_potions: bool,
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
            sandbox_supports_potions=combat_sandbox_potions,
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
        combat_num_simulations: int = 16,
        build_num_simulations: int = 12,
        route_num_simulations: int = 8,
        settlement_weight: float = 0.12,
        settlement_decay: float = 0.99,
        settlement_max_steps: int = 12,
        trivial_build_fast_path: bool = True,
        recent_tail_windows: tuple[int, ...] = DEFAULT_RECENT_TAIL_WINDOWS,
        recent_tail_tracked_encounters: list[str] | None = None,
        recent_tail_min_samples: int = 4,
        log_dir: str = "runs",
        checkpoint_dir: str = "checkpoints",
        checkpoint_keep_last: int = 3,
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
        self.settlement_weight = max(float(settlement_weight), 0.0)
        self.settlement_decay = float(np.clip(settlement_decay, 0.0, 1.0))
        self.settlement_max_steps = max(int(settlement_max_steps), 0)
        self.trivial_build_fast_path = bool(trivial_build_fast_path)
        self.recent_tail_windows = tuple(sorted({max(int(window), 1) for window in recent_tail_windows}))
        self.recent_tail_min_samples = max(int(recent_tail_min_samples), 1)
        self.recent_tail_tracked_encounters = [
            str(encounter_id).strip()
            for encounter_id in (recent_tail_tracked_encounters or [])
            if str(encounter_id).strip()
        ]
        self.domain_num_simulations = {
            "combat": max(int(combat_num_simulations), 1),
            "build": max(int(build_num_simulations), 1),
            "route": max(int(route_num_simulations), 1),
        }

        Path(log_dir).mkdir(parents=True, exist_ok=True)
        Path(checkpoint_dir).mkdir(parents=True, exist_ok=True)
        self.log_dir = log_dir
        self.checkpoint_dir = checkpoint_dir
        self.checkpoint_keep_last = max(int(checkpoint_keep_last), 0)

        self.writer = SummaryWriter(log_dir=log_dir)
        self.total_steps = 0
        self.episode_count = 0
        self.last_episode_metrics: dict[str, Any] = {}
        self.recent_combat_monitor = RecentCombatMonitor(
            windows=self.recent_tail_windows,
            tracked_encounters=self.recent_tail_tracked_encounters,
            min_samples=self.recent_tail_min_samples,
        )

    @staticmethod
    def _compact_search_stats(search_stats: dict[str, Any]) -> dict[str, Any]:
        if not search_stats:
            return {}
        compact: dict[str, Any] = {}
        for key in (
            "num_simulations",
            "root_candidates",
            "root_selectable_children_mean",
            "mean_expanded_children",
            "mean_predicted_legal_count",
            "mean_surface_keep_count",
            "max_search_depth",
            "mean_leaf_depth",
            "mean_concrete_leaf_depth",
            "depth_ge_2_rate",
            "depth_ge_3_rate",
            "root_top1_visit_share",
            "root_visit_entropy",
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
            "combat_grounded_root_enabled",
            "q_value_ucb_enabled",
        ):
            value = search_stats.get(key)
            if value is None:
                continue
            try:
                compact[key] = float(value)
            except (TypeError, ValueError):
                continue
        return compact

    def record_recent_combat_episode(self, metadata: dict[str, Any] | None) -> dict[int, dict[str, Any]]:
        snapshot = self.recent_combat_monitor.record_episode(metadata)
        if snapshot:
            self.recent_combat_monitor.log_to_writer(self.writer, self.episode_count)
        return snapshot

    def _num_simulations_for_domain(self, decision_domain: str) -> int:
        return int(self.domain_num_simulations.get(decision_domain, self.mcts.num_simulations))

    @staticmethod
    def _episode_progress_snapshot(info: dict[str, Any] | None) -> dict[str, Any]:
        transition_state = info.get("transition_state") if isinstance(info, dict) else None
        run = transition_state.get("run") if isinstance(transition_state, dict) else None
        player = transition_state.get("player") if isinstance(transition_state, dict) else None
        if not isinstance(run, dict):
            return {}
        room_type = str(run.get("room_type") or "")
        floor = float(run.get("floor") or 0.0)
        act_id = float(run.get("act_id") or 0.0)
        hp = float(player.get("hp") or 0.0) if isinstance(player, dict) else 0.0
        max_hp = float(player.get("max_hp") or 0.0) if isinstance(player, dict) else 0.0
        return {
            "floor": floor,
            "act_id": act_id,
            "room_type": room_type,
            "room_type_lower": room_type.lower(),
            "hp": hp,
            "max_hp": max_hp,
        }

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

    @staticmethod
    def _semantic_family_from_signature(signature: dict[str, Any] | None) -> str:
        semantic = signature.get("semantic") if isinstance(signature, dict) else None
        if isinstance(semantic, dict):
            return str(semantic.get("family") or "").strip().lower()
        return ""

    @staticmethod
    def _semantic_domain_from_signature(signature: dict[str, Any] | None) -> str:
        semantic = signature.get("semantic") if isinstance(signature, dict) else None
        if isinstance(semantic, dict):
            return str(semantic.get("domain") or "").strip().lower()
        return ""

    @staticmethod
    def _build_route_step_flag(
        decision_domain: str,
        action_family: str,
    ) -> bool:
        return action_family in BUILD_ROUTE_SETTLEMENT_FAMILIES or decision_domain in {"build", "route"}

    @staticmethod
    def _wasteful_proceed_flag(
        signature: dict[str, Any] | None,
        *,
        decision_domain: str,
        phase: str,
    ) -> bool:
        if not isinstance(signature, dict):
            return False
        family = MuZeroTrainer._semantic_family_from_signature(signature)
        surface = str(signature.get("surface") or "").strip().lower()
        selection = str(signature.get("selection") or "").strip().lower()
        phase = str(phase or "").strip().lower()
        rewardish = phase in WASTEFUL_REWARD_PHASES or surface in WASTEFUL_REWARD_PHASES
        if family == "proceed" and rewardish:
            return True
        if family in {"reward", "card_reward"} and selection == "skip":
            return True
        if family == "card_reward" and selection == "cancel":
            return True
        return family == "proceed" and decision_domain == "build" and rewardish

    @staticmethod
    def _episode_settlement_signal(
        *,
        max_floor: float,
        death_floor: float,
        elite_rooms_seen: int,
        act1_boss_seen: bool,
        act1_clear: bool,
        terminated: bool,
    ) -> float:
        floor_norm = float(np.clip(max_floor / 16.0, 0.0, 1.0))
        elite_norm = float(np.clip(float(elite_rooms_seen) / 2.0, 0.0, 1.0))
        signal = 0.45 * floor_norm
        signal += 0.10 * elite_norm
        signal += 0.20 if act1_boss_seen else 0.0
        signal += 0.35 if act1_clear else 0.0
        if terminated and not act1_clear:
            death_ref = death_floor if death_floor > 0.0 else max_floor
            early_death_penalty = float(np.clip((16.0 - death_ref) / 16.0, 0.0, 1.0))
            signal -= 0.40 * early_death_penalty
        return float(np.clip(signal, -1.0, 1.0))

    def _get_live_legal_actions(
        self,
        expected_count: int,
    ) -> list[dict[str, Any]]:
        env_unwrapped = getattr(self.env, "unwrapped", self.env)
        raw_actions = getattr(env_unwrapped, "_legal_actions", None)
        if not isinstance(raw_actions, list):
            return []
        if expected_count <= 0 or len(raw_actions) < expected_count:
            return []
        live_actions = raw_actions[:expected_count]
        if not all(isinstance(action, dict) for action in live_actions):
            return []
        return list(live_actions)

    def _empty_potion_slots(self) -> int:
        env_unwrapped = getattr(self.env, "unwrapped", self.env)
        raw_obs = getattr(env_unwrapped, "_last_obs_raw", None)
        player = raw_obs.get("player") if isinstance(raw_obs, dict) else None
        potions = player.get("potions") if isinstance(player, dict) else None
        if not isinstance(potions, list):
            return 0

        empty = 0
        for potion in potions:
            if isinstance(potion, dict):
                title = str(potion.get("title") or potion.get("id") or "").strip().lower()
            else:
                title = str(potion or "").strip().lower()
            if title in {"", "[empty]", "empty", "空", "空位"}:
                empty += 1
        return int(empty)

    @staticmethod
    def _reward_type_from_action(
        action: dict[str, Any] | None,
        signature: dict[str, Any] | None = None,
    ) -> str:
        if not isinstance(action, dict):
            action = {}
        if not isinstance(signature, dict):
            signature = {}

        action_id = str(
            signature.get("action_id")
            or action.get("action_id")
            or ""
        ).strip().lower()
        selection = str(
            action.get("selection")
            or signature.get("selection")
            or ""
        ).strip().lower()
        text_chunks = [
            str(action.get("title") or ""),
            str(action.get("label") or ""),
            str(signature.get("title") or ""),
            str(signature.get("action_id") or action.get("action_id") or ""),
        ]
        text = " ".join(chunk for chunk in text_chunks if chunk).strip().lower()

        is_skip_reward = (
            action_id.startswith("reward:skip_")
            or selection == "skip"
            or "skip potion" in text
            or "skip reward" in text
            or "跳过药水" in text
            or "跳过奖励" in text
        )
        if is_skip_reward:
            return ""

        reward_payload = action.get("reward") if isinstance(action.get("reward"), dict) else {}
        reward_type = str(
            reward_payload.get("type")
            or action.get("reward_type")
            or action.get("selection")
            or ""
        ).strip().lower()
        if reward_type in {"gold", "potion", "card", "relic"}:
            return reward_type

        semantic = signature.get("semantic") if isinstance(signature.get("semantic"), dict) else {}
        semantic_key = str(semantic.get("semantic_key") or "").strip().lower()
        for token, normalized in (
            ("|gold", "gold"),
            ("|potion", "potion"),
            ("|card", "card"),
            ("|relic", "relic"),
        ):
            if token in semantic_key:
                return normalized

        if not text:
            return ""
        if "gold" in text or "金币" in text:
            return "gold"
        if "potion" in text or "药水" in text:
            return "potion"
        if "card reward" in text or "card_reward" in text or "卡牌奖励" in text:
            return "card"
        if "relic" in text or "遗物" in text:
            return "relic"
        return ""

    def _trivial_build_fast_path_choice(
        self,
        *,
        decision_domain: str,
        action_mask: np.ndarray,
        legal_actions: list[dict[str, Any]],
    ) -> dict[str, Any] | None:
        if not self.trivial_build_fast_path or decision_domain != "build":
            return None

        compact_count = min(len(legal_actions), MAX_ACTIONS)
        if compact_count <= 0:
            return None

        mask = np.asarray(action_mask, dtype=np.float32).reshape(-1)
        if mask.size < compact_count:
            return None

        live_actions = self._get_live_legal_actions(compact_count)
        action_source: list[dict[str, Any]]
        if live_actions:
            action_source = live_actions
        else:
            filtered_compact = [
                action if isinstance(action, dict) else {}
                for action in legal_actions[:compact_count]
            ]
            action_source = filtered_compact

        enabled_indices = [idx for idx in range(compact_count) if float(mask[idx]) > 0.0]
        if not enabled_indices:
            return None

        gold_indices: list[int] = []
        potion_indices: list[int] = []
        proceed_indices: list[int] = []
        startup_indices: list[int] = []
        complex_family_present = False

        empty_potion_slots = self._empty_potion_slots()

        for idx in enabled_indices:
            source_action = action_source[idx]
            signature = (
                compact_action_signature(source_action)
                if live_actions
                else (source_action if isinstance(source_action, dict) else {})
            )
            family = self._semantic_family_from_signature(signature)

            if family in TRIVIAL_BUILD_FAST_PATH_COMPLEX_FAMILIES:
                complex_family_present = True
            elif family == "reward":
                reward_type = self._reward_type_from_action(source_action, signature)
                if reward_type == "gold":
                    gold_indices.append(idx)
                elif reward_type == "potion" and empty_potion_slots > 0:
                    potion_indices.append(idx)
            elif family == "proceed":
                proceed_indices.append(idx)
            elif family == "startup":
                startup_indices.append(idx)

        if gold_indices:
            return {"action_idx": int(gold_indices[0]), "reason": "reward_gold"}
        if potion_indices:
            return {"action_idx": int(potion_indices[0]), "reason": "reward_potion"}
        if complex_family_present:
            return None
        if len(enabled_indices) == 1:
            only_idx = int(enabled_indices[0])
            only_signature = (
                compact_action_signature(action_source[only_idx])
                if live_actions
                else action_source[only_idx]
            )
            only_family = self._semantic_family_from_signature(only_signature)
            if only_family == "proceed":
                return {"action_idx": only_idx, "reason": "proceed_only"}
            if only_family == "startup":
                return {"action_idx": only_idx, "reason": "startup_only"}
        return None

    def _combat_card_selection_fast_path_choice(
        self,
        *,
        decision_domain: str,
        action_mask: np.ndarray,
        legal_actions: list[dict[str, Any]],
        policy_logits: torch.Tensor,
    ) -> dict[str, Any] | None:
        if decision_domain != "combat":
            return None

        env_unwrapped = getattr(self.env, "unwrapped", self.env)
        raw_obs = getattr(env_unwrapped, "_last_obs_raw", None)
        if not isinstance(raw_obs, dict):
            return None

        card_selection = raw_obs.get("card_selection")
        if not isinstance(card_selection, dict):
            return None

        screen = str(raw_obs.get("screen") or "").strip().upper()
        if screen != "CARD_SELECTION" and not bool(card_selection.get("visible")):
            return None

        if bool(card_selection.get("requires_manual_confirmation")):
            return None
        if bool(card_selection.get("confirm_visible")):
            return None

        max_select_raw = card_selection.get("max_select")
        try:
            max_select = int(max_select_raw) if max_select_raw is not None else 1
        except (TypeError, ValueError):
            return None
        if max_select != 1:
            return None

        compact_count = min(len(legal_actions), MAX_ACTIONS)
        if compact_count <= 0:
            return None

        mask = np.asarray(action_mask, dtype=np.float32).reshape(-1)
        if mask.size < compact_count:
            return None

        live_actions = self._get_live_legal_actions(compact_count)
        action_source: list[dict[str, Any]]
        if live_actions:
            action_source = live_actions
        else:
            action_source = [
                action if isinstance(action, dict) else {}
                for action in legal_actions[:compact_count]
            ]

        enabled_indices = [idx for idx in range(compact_count) if float(mask[idx]) > 0.0]
        if not enabled_indices:
            return None

        selectable_indices: list[int] = []
        for idx in enabled_indices:
            action = action_source[idx]
            if not isinstance(action, dict):
                return None
            kind = str(action.get("kind") or "").strip().lower()
            action_id = str(action.get("action_id") or "").strip().lower()
            selection = str(
                action.get("selection_action")
                or action.get("selection")
                or ""
            ).strip().lower()
            if kind == "card_selection":
                selectable_indices.append(idx)
                continue
            if any(token in action_id for token in ("skip", "cancel", "close")) or selection in {"skip", "cancel", "close"}:
                selectable_indices.append(idx)
                continue
            return None

        if not selectable_indices:
            return None

        logits = (
            policy_logits.detach()
            .reshape(-1)
            .to(dtype=torch.float32, device="cpu")
            .numpy()
        )
        best_idx = max(
            selectable_indices,
            key=lambda idx: float(logits[idx]) if 0 <= idx < len(logits) else float("-inf"),
        )
        return {"action_idx": int(best_idx), "reason": "combat_card_selection"}

    def _obs_list_to_torch(self, obs_list: list[dict[str, Any]]) -> dict[str, torch.Tensor]:
        """Convert a list of dict observations into a batched torch dict."""
        obs_torch_batch: dict[str, torch.Tensor] = {}
        obs_numpy_batch = MuZeroReplayBuffer.batch_observations(obs_list)
        for key, value in obs_numpy_batch.items():
            obs_torch_batch[key] = torch.from_numpy(value).to(self.device)
        return obs_torch_batch

    def compute_temperature(self, step: int, total_steps: int) -> float:
        """Compute temperature schedule: linear decay from 1.0 to 0.1."""
        progress = min(step / max(total_steps, 1), 1.0)
        return 1.0 - progress * 0.9

    def self_play_episode_on_env(self, env: gym.Env, *, temperature: float = 1.0) -> tuple[float, int]:
        """Run one self-play episode on the provided environment.

        This is used by the combat-sandbox multi-env harness so the trainer can
        round-robin across multiple live bridge sessions without changing the
        full-run single-env path.
        """
        previous_env = self.env
        self.env = env
        try:
            return self.self_play_episode(temperature=temperature)
        finally:
            self.env = previous_env

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
        progress_snapshots: list[dict[str, Any]] = []
        seen_floors: set[int] = set()
        elite_floors: set[int] = set()
        boss_floors: set[int] = set()
        death_floor = 0.0
        search_root_candidates: list[float] = []
        search_root_selectable_children: list[float] = []
        search_expanded_children: list[float] = []
        search_predicted_legal: list[float] = []
        search_surface_keep: list[float] = []
        search_max_depths: list[float] = []
        search_depths: list[float] = []
        search_concrete_depths: list[float] = []
        search_depth_ge_2: list[float] = []
        search_depth_ge_3: list[float] = []
        search_semantic_expansion: list[float] = []
        search_semantic_switch: list[float] = []
        search_semantic_chain_steps: list[float] = []
        search_semantic_drill: list[float] = []
        search_root_top1_visit_share: list[float] = []
        search_root_visit_entropy: list[float] = []
        search_num_simulations: list[float] = []
        search_root_bias_scale: list[float] = []
        search_end_turn_guard_applied: list[float] = []
        search_end_turn_guard_forced_alt: list[float] = []
        domain_search_stats: dict[str, dict[str, list[float]]] = {
            domain: defaultdict(list) for domain in DECISION_DOMAINS
        }
        decision_counts: dict[str, int] = {domain: 0 for domain in DECISION_DOMAINS}
        fast_path_counts: dict[str, int] = {domain: 0 for domain in DECISION_DOMAINS}
        fast_path_reason_counts: dict[str, int] = defaultdict(int)
        selected_family_counts: dict[str, dict[str, int]] = {
            domain: defaultdict(int) for domain in DECISION_DOMAINS
        }
        terminated = False
        truncated = False
        initial_progress = self._episode_progress_snapshot(initial_info)
        if initial_progress:
            progress_snapshots.append(initial_progress)

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
            num_simulations = self._num_simulations_for_domain(decision_domain)
            obs_batch = dict_obs_to_torch(obs, device=self.device)
            with torch.no_grad():
                initial = self.network.initial_inference(obs_batch)
                root_value = initial.value.squeeze(0).item()
                root_value_components = initial.value_components.squeeze(0).detach().cpu().numpy()

            fast_path_choice = self._trivial_build_fast_path_choice(
                decision_domain=decision_domain,
                action_mask=np.asarray(action_mask, dtype=np.float32),
                legal_actions=legal_actions,
            )
            if fast_path_choice is None:
                fast_path_choice = self._combat_card_selection_fast_path_choice(
                    decision_domain=decision_domain,
                    action_mask=np.asarray(action_mask, dtype=np.float32),
                    legal_actions=legal_actions,
                    policy_logits=initial.policy_logits.squeeze(0),
                )
            fast_path_reason = ""
            if fast_path_choice is not None:
                action_idx = int(fast_path_choice["action_idx"])
                fast_path_reason = str(fast_path_choice.get("reason") or "").strip().lower()
                search_policy = np.zeros(MAX_ACTIONS, dtype=np.float32)
                if 0 <= action_idx < MAX_ACTIONS:
                    search_policy[action_idx] = 1.0
                fast_path_counts[decision_domain] = fast_path_counts.get(decision_domain, 0) + 1
                if fast_path_reason:
                    fast_path_reason_counts[fast_path_reason] += 1
                search_stats: dict[str, Any] = {}
            else:
                # Run MCTS to get action and policy
                with torch.no_grad():
                    self.mcts.set_training_step(self.total_steps)
                    self.mcts.set_root_bias_enabled(True)
                    self.mcts.set_semantic_rollout_enabled(self.mcts._semantic_rollout_enabled)
                    action_idx, search_policy = self.mcts.run(
                        self.network,
                        obs,
                        action_mask,
                        num_simulations=num_simulations,
                        temperature=temperature,
                        decision_domain=decision_domain,
                    )
                search_stats = getattr(self.mcts, "last_run_stats", {}) or {}
                if search_stats:
                    search_num_simulations.append(float(search_stats.get("num_simulations", num_simulations)))
                    search_root_candidates.append(float(search_stats.get("root_candidates", 0.0)))
                    search_root_selectable_children.append(float(search_stats.get("root_selectable_children_mean", 0.0)))
                    search_expanded_children.append(float(search_stats.get("mean_expanded_children", 0.0)))
                    search_predicted_legal.append(float(search_stats.get("mean_predicted_legal_count", 0.0)))
                    search_surface_keep.append(float(search_stats.get("mean_surface_keep_count", 0.0)))
                    search_max_depths.append(float(search_stats.get("max_search_depth", 0.0)))
                    search_depths.append(float(search_stats.get("mean_leaf_depth", 0.0)))
                    search_concrete_depths.append(float(search_stats.get("mean_concrete_leaf_depth", 0.0)))
                    search_depth_ge_2.append(float(search_stats.get("depth_ge_2_rate", 0.0)))
                    search_depth_ge_3.append(float(search_stats.get("depth_ge_3_rate", 0.0)))
                    search_semantic_expansion.append(float(search_stats.get("semantic_expansion_rate", 0.0)))
                    search_semantic_switch.append(float(search_stats.get("semantic_switch_rate", 0.0)))
                    search_semantic_chain_steps.append(float(search_stats.get("semantic_chain_steps_mean", 0.0)))
                    search_semantic_drill.append(float(search_stats.get("semantic_drill_rate", 0.0)))
                    search_root_top1_visit_share.append(float(search_stats.get("root_top1_visit_share", 0.0)))
                    search_root_visit_entropy.append(float(search_stats.get("root_visit_entropy", 0.0)))
                    search_root_bias_scale.append(float(search_stats.get("root_bias_scale", 0.0)))
                    search_end_turn_guard_applied.append(float(search_stats.get("end_turn_guard_applied", 0.0)))
                    search_end_turn_guard_forced_alt.append(float(search_stats.get("end_turn_guard_forced_alternative", 0.0)))
                    for metric_key, metric_value in search_stats.items():
                        try:
                            domain_search_stats[decision_domain][metric_key].append(float(metric_value))
                        except (TypeError, ValueError):
                            continue

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
            action_family = ""
            semantic_domain = ""
            phase = str(info.get("phase") or "").strip().lower()
            surface = ""
            selection = ""
            wasteful_end_turn = False
            wasteful_proceed = False
            if chosen_signature:
                chosen_signature = {
                    **chosen_signature,
                    "selected_index": int(action_idx),
                    "phase": info.get("phase"),
                    "legal_action_count": int(info.get("legal_action_count", 0) or 0),
                    "selection": chosen_action.get("selection") if isinstance(chosen_action, dict) else None,
                    "fast_path_applied": bool(fast_path_reason),
                    "fast_path_reason": fast_path_reason or None,
                    "reward_type": (
                        (chosen_action.get("reward") or {}).get("type")
                        if isinstance(chosen_action, dict) and isinstance(chosen_action.get("reward"), dict)
                        else None
                    ),
                    "shop_action": (
                        chosen_action.get("shop_action")
                        if isinstance(chosen_action, dict)
                        else None
                    ),
                }
                surface = str(chosen_signature.get("surface") or "").strip().lower()
                selection = str(chosen_signature.get("selection") or "").strip().lower()
                semantic_compact = chosen_signature.get("semantic")
                if isinstance(semantic_compact, dict):
                    family = str(semantic_compact.get("family") or "").strip().lower()
                    action_family = family
                    semantic_domain = str(semantic_compact.get("domain") or "").strip().lower()
                    if family:
                        selected_family_counts[decision_domain][family] += 1
            if action_family == "end_turn":
                try:
                    wasteful_end_turn = bool(
                        self.mcts._root_end_turn_context(
                            obs,
                            np.asarray(action_mask, dtype=np.float32).reshape(-1),
                        ).get("wasteful")
                    )
                except Exception:
                    wasteful_end_turn = False
            wasteful_proceed = self._wasteful_proceed_flag(
                chosen_signature,
                decision_domain=decision_domain,
                phase=phase,
            )

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
                decision_domain=decision_domain,
                phase=phase,
                action_family=action_family,
                semantic_domain=semantic_domain,
                surface=surface,
                selection=selection,
                wasteful_end_turn=wasteful_end_turn,
                wasteful_proceed=wasteful_proceed,
            )

            # Take action in environment
            obs, reward, terminated, truncated, info = self.env.step(int(action_idx))
            episode_reward += reward
            episode_length += 1
            progress_snapshot = self._episode_progress_snapshot(info)
            if progress_snapshot:
                progress_snapshots.append(progress_snapshot)
                floor_int = int(progress_snapshot["floor"])
                if floor_int > 0:
                    seen_floors.add(floor_int)
                room_type_lower = str(progress_snapshot.get("room_type_lower") or "")
                if "elite" in room_type_lower:
                    elite_floors.add(floor_int)
                if "boss" in room_type_lower:
                    boss_floors.add(floor_int)
                if progress_snapshot.get("hp", 0.0) <= 0.0:
                    death_floor = max(death_floor, float(progress_snapshot.get("floor", 0.0)))

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
        max_floor = max((float(snapshot.get("floor", 0.0)) for snapshot in progress_snapshots), default=0.0)
        max_act_id = max((float(snapshot.get("act_id", 0.0)) for snapshot in progress_snapshots), default=0.0)
        rooms_seen = len(seen_floors)
        act1_boss_seen = any(
            float(snapshot.get("act_id", 0.0)) <= 1.0 and "boss" in str(snapshot.get("room_type_lower") or "")
            for snapshot in progress_snapshots
        )
        act1_clear = max_act_id >= 2.0
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
            "max_floor": float(max_floor),
            "max_act_id": float(max_act_id),
            "rooms_seen": int(rooms_seen),
            "death_floor": float(death_floor),
            "elite_rooms_seen": int(len(elite_floors)),
            "boss_rooms_seen": int(len(boss_floors)),
            "act1_boss_seen": bool(act1_boss_seen),
            "act1_clear": bool(act1_clear),
            "decision_counts": {domain: int(count) for domain, count in decision_counts.items()},
        }

        settlement_signal = self._episode_settlement_signal(
            max_floor=max_floor,
            death_floor=death_floor,
            elite_rooms_seen=len(elite_floors),
            act1_boss_seen=act1_boss_seen,
            act1_clear=act1_clear,
            terminated=terminated,
        )
        settlement_stats = trajectory.apply_light_episode_settlement(
            settlement_signal,
            settlement_weight=self.settlement_weight,
            decay=self.settlement_decay,
            max_steps=self.settlement_max_steps,
        )
        augmented_episode_reward = float(episode_reward + settlement_stats.get("total_bonus", 0.0))
        trajectory.metadata.update({
            "settlement_weight": float(self.settlement_weight),
            "settlement_decay": float(self.settlement_decay),
            "settlement_max_steps": int(self.settlement_max_steps),
            "episode_augmented_total_reward": augmented_episode_reward,
        })

        self.buffer.save_episode(
            trajectory,
            discount=self.discount,
            n_steps=self.n_step_return,
        )
        self.episode_count += 1
        recent_tail_snapshot = self.record_recent_combat_episode(trajectory.metadata)

        if search_root_candidates:
            self.writer.add_scalar(
                "search/max_depth_mean",
                float(np.mean(search_max_depths)) if search_max_depths else 0.0,
                self.episode_count,
            )
            self.writer.add_scalar(
                "search/num_simulations",
                float(np.mean(search_num_simulations)) if search_num_simulations else 0.0,
                self.episode_count,
            )
            self.writer.add_scalar(
                "search/root_candidates_mean",
                float(np.mean(search_root_candidates)),
                self.episode_count,
            )
            self.writer.add_scalar(
                "search/root_selectable_children_mean",
                float(np.mean(search_root_selectable_children)) if search_root_selectable_children else 0.0,
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
                "search/concrete_leaf_depth_mean",
                float(np.mean(search_concrete_depths)) if search_concrete_depths else 0.0,
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
                "search/root_top1_visit_share",
                float(np.mean(search_root_top1_visit_share)) if search_root_top1_visit_share else 0.0,
                self.episode_count,
            )
            self.writer.add_scalar(
                "search/root_visit_entropy",
                float(np.mean(search_root_visit_entropy)) if search_root_visit_entropy else 0.0,
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
        total_fast_paths = int(sum(fast_path_counts.values()))

        self.writer.add_scalar("decision/total_count", float(total_decisions), self.episode_count)
        self.writer.add_scalar("decision/fast_path_count", float(total_fast_paths), self.episode_count)
        self.writer.add_scalar(
            "decision/fast_path_rate",
            float(total_fast_paths) / float(total_decisions),
            self.episode_count,
        )
        for domain in DECISION_DOMAINS:
            domain_count = decision_counts.get(domain, 0)
            self.writer.add_scalar(f"decision/{domain}_count", float(domain_count), self.episode_count)
            self.writer.add_scalar(f"decision/{domain}_share", float(domain_count) / float(total_decisions), self.episode_count)
            domain_fast_paths = fast_path_counts.get(domain, 0)
            if domain_count > 0:
                self.writer.add_scalar(
                    f"decision/{domain}/fast_path_rate",
                    float(domain_fast_paths) / float(domain_count),
                    self.episode_count,
                )
            metric_lists = domain_search_stats.get(domain) or {}
            if not metric_lists:
                continue
            metric_name_map = {
                "root_candidates": "root_candidates_mean",
                "root_selectable_children_mean": "root_selectable_children_mean",
                "num_simulations": "num_simulations",
                "mean_expanded_children": "expanded_children_mean",
                "mean_predicted_legal_count": "predicted_legal_mean",
                "mean_surface_keep_count": "surface_keep_mean",
                "max_search_depth": "max_depth_mean",
                "mean_leaf_depth": "leaf_depth_mean",
                "mean_concrete_leaf_depth": "concrete_leaf_depth_mean",
                "depth_ge_2_rate": "depth_ge_2_rate",
                "depth_ge_3_rate": "depth_ge_3_rate",
                "root_top1_visit_share": "root_top1_visit_share",
                "root_visit_entropy": "root_visit_entropy",
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
                "combat_grounded_root_enabled": "combat_grounded_root_enabled",
                "q_value_ucb_enabled": "q_value_ucb_enabled",
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
        if total_fast_paths > 0:
            for reason in TRIVIAL_BUILD_FAST_PATH_REASONS:
                safe_reason = reason.replace("/", "_").replace(" ", "_")
                count = int(fast_path_reason_counts.get(reason, 0))
                self.writer.add_scalar(
                    f"decision/fast_path/reason_{safe_reason}_rate",
                    float(count) / float(total_fast_paths),
                    self.episode_count,
                )

        self.writer.add_scalar("episode/max_floor", float(max_floor), self.episode_count)
        self.writer.add_scalar("episode/max_act_id", float(max_act_id), self.episode_count)
        self.writer.add_scalar("episode/rooms_seen", float(rooms_seen), self.episode_count)
        self.writer.add_scalar("episode/death_floor", float(death_floor), self.episode_count)
        self.writer.add_scalar("episode/elite_rooms_seen", float(len(elite_floors)), self.episode_count)
        self.writer.add_scalar("episode/act1_boss_seen", 1.0 if act1_boss_seen else 0.0, self.episode_count)
        self.writer.add_scalar("episode/act1_clear", 1.0 if act1_clear else 0.0, self.episode_count)
        self.writer.add_scalar("episode/settlement_signal", float(settlement_stats.get("signal", 0.0)), self.episode_count)
        self.writer.add_scalar("episode/settlement_applied_steps", float(settlement_stats.get("applied_steps", 0.0)), self.episode_count)
        self.writer.add_scalar("episode/settlement_total_bonus", float(settlement_stats.get("total_bonus", 0.0)), self.episode_count)
        self.writer.add_scalar("episode/reward_augmented", float(augmented_episode_reward), self.episode_count)

        self.last_episode_metrics = {
            "decision_counts": {domain: int(count) for domain, count in decision_counts.items()},
            "decision_total": int(total_decisions),
            "fast_path_counts": {domain: int(count) for domain, count in fast_path_counts.items()},
            "fast_path_total": int(total_fast_paths),
            "fast_path_reason_counts": {
                reason: int(fast_path_reason_counts.get(reason, 0))
                for reason in TRIVIAL_BUILD_FAST_PATH_REASONS
                if int(fast_path_reason_counts.get(reason, 0)) > 0
            },
            "domain_search_means": domain_search_means,
            "domain_family_rates": domain_family_rates,
            "episode_reward": float(episode_reward),
            "episode_length": int(episode_length),
            "max_floor": float(max_floor),
            "max_act_id": float(max_act_id),
            "rooms_seen": int(rooms_seen),
            "death_floor": float(death_floor),
            "elite_rooms_seen": int(len(elite_floors)),
            "act1_boss_seen": bool(act1_boss_seen),
            "act1_clear": bool(act1_clear),
            "settlement_signal": float(settlement_stats.get("signal", 0.0)),
            "settlement_applied_steps": int(settlement_stats.get("applied_steps", 0.0)),
            "settlement_total_bonus": float(settlement_stats.get("total_bonus", 0.0)),
            "episode_reward_augmented": float(augmented_episode_reward),
        }
        if recent_tail_snapshot:
            self.last_episode_metrics["recent_tail"] = recent_tail_snapshot

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
        position_replay_weight = batch["position_replay_weight"].to(self.device)
        sample_boundary_flag = batch["sample_boundary_flag"].to(self.device)
        sample_build_route_flag = batch["sample_build_route_flag"].to(self.device)
        sample_wasteful_flag = batch["sample_wasteful_flag"].to(self.device)
        sample_settlement_abs = batch["sample_settlement_abs"].to(self.device)
        sample_weak_flag = batch["sample_weak_flag"].to(self.device)
        sample_normal_flag = batch["sample_normal_flag"].to(self.device)
        sample_elite_flag = batch["sample_elite_flag"].to(self.device)
        sample_boss_flag = batch["sample_boss_flag"].to(self.device)
        sample_hard_encounter_flag = batch["sample_hard_encounter_flag"].to(self.device)
        sample_hard_normal_flag = batch["sample_hard_normal_flag"].to(self.device)
        sample_hard_elite_flag = batch["sample_hard_elite_flag"].to(self.device)
        sample_tier_weight = batch["sample_tier_weight"].to(self.device)
        sample_encounter_weight = batch["sample_encounter_weight"].to(self.device)
        sample_sampling_scale = batch["sample_sampling_scale"].to(self.device)

        batch_size_actual = int(action_batch.shape[0])

        # Materialize only the current/next observation batches on demand to keep
        # peak host memory lower during training.
        current_obs_torch = self._obs_list_to_torch(
            [sequence[0] for sequence in obs_sequence_batch]
        )

        # Initial inference
        initial = self.network.initial_inference(current_obs_torch)
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
        state_consistency_mse_sum = 0.0
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
                obs=current_obs_torch,
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
            next_obs_torch = self._obs_list_to_torch(
                [sequence[step_k + 1] for sequence in obs_sequence_batch]
            )
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
                teacher_semantic = self.network.project_to_semantic_latent(teacher_hidden)
            state_consistency_loss, state_consistency_cosine, state_consistency_mse = self._state_consistency_loss(
                recurrent.next_hidden_state,
                teacher_hidden,
            )
            if semantic_training_active and semantic_recurrent is not None:
                semantic_state_consistency_loss, _, _ = self._state_consistency_loss(
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
            state_consistency_mse_sum += state_consistency_mse
            surface_precision_sum += surface_metrics["legal_precision"]
            surface_recall_sum += surface_metrics["legal_recall"]
            surface_f1_sum += surface_metrics["legal_f1"]
            surface_count_mae_sum += surface_metrics["legal_count_mae"]
            surface_domain_acc_sum += surface_metrics["decision_domain_acc"]
            surface_phase_acc_sum += surface_metrics["phase_acc"]

            hidden_state = recurrent.next_hidden_state
            current_obs_torch = next_obs_torch

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
            "metric/state_consistency_mse": state_consistency_mse_sum / surface_terms,
            "metric/predicted_legal_precision": surface_precision_sum / surface_terms,
            "metric/predicted_legal_recall": surface_recall_sum / surface_terms,
            "metric/predicted_legal_f1": surface_f1_sum / surface_terms,
            "metric/predicted_legal_count_mae": surface_count_mae_sum / surface_terms,
            "metric/decision_domain_acc": surface_domain_acc_sum / surface_terms,
            "metric/phase_acc": surface_phase_acc_sum / surface_terms,
            "buffer/sample_position_weight_mean": float(position_replay_weight.mean().item()),
            "buffer/sample_boundary_rate": float(sample_boundary_flag.mean().item()),
            "buffer/sample_build_route_rate": float(sample_build_route_flag.mean().item()),
            "buffer/sample_wasteful_rate": float(sample_wasteful_flag.mean().item()),
            "buffer/sample_settlement_abs_mean": float(sample_settlement_abs.mean().item()),
            "buffer/sample_weak_rate": float(sample_weak_flag.mean().item()),
            "buffer/sample_normal_rate": float(sample_normal_flag.mean().item()),
            "buffer/sample_elite_rate": float(sample_elite_flag.mean().item()),
            "buffer/sample_boss_rate": float(sample_boss_flag.mean().item()),
            "buffer/sample_hard_encounter_rate": float(sample_hard_encounter_flag.mean().item()),
            "buffer/sample_hard_normal_rate": float(sample_hard_normal_flag.mean().item()),
            "buffer/sample_hard_elite_rate": float(sample_hard_elite_flag.mean().item()),
            "buffer/sample_tier_weight_mean": float(sample_tier_weight.mean().item()),
            "buffer/sample_encounter_weight_mean": float(sample_encounter_weight.mean().item()),
            "buffer/sample_sampling_scale_mean": float(sample_sampling_scale.mean().item()),
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
        student_state: torch.Tensor,
        teacher_state: torch.Tensor,
    ) -> tuple[torch.Tensor, float, float]:
        """
        Align dynamics(next_hidden_state) with representation(next_obs).

        We intentionally compare the raw hidden states here instead of passing
        both sides through the same learned projector. The shared-projector
        version can collapse to a near-constant embedding and drive the cosine
        metric to a misleading 1.0 with zero loss, which is exactly the failure
        mode we want to detect and prevent.
        """
        student = torch.nn.functional.normalize(student_state, dim=-1)
        teacher = torch.nn.functional.normalize(teacher_state.detach(), dim=-1)
        cosine = (student * teacher).sum(dim=-1)
        mse = torch.nn.functional.mse_loss(student, teacher, reduction="none").mean(dim=-1)
        loss = (1.0 - cosine).mean() + 0.25 * mse.mean()
        return loss, float(cosine.mean().item()), float(mse.mean().item())

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
            "env": {
                "mode": "combat_sandbox" if hasattr(self.env, "sandbox_supports_potions") else "full_run",
                "potion_mechanics_available": bool(getattr(self.env, "sandbox_supports_potions", True)),
            },
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
            "settlement_weight": float(self.settlement_weight),
            "settlement_decay": float(self.settlement_decay),
            "settlement_max_steps": int(self.settlement_max_steps),
            "trivial_build_fast_path": bool(self.trivial_build_fast_path),
            "domain_num_simulations": {
                domain: int(value) for domain, value in sorted(self.domain_num_simulations.items())
            },
            "checkpoint_keep_last": int(self.checkpoint_keep_last),
            "replay": {
                "boundary_build_bonus": float(self.buffer.boundary_build_bonus),
                "boundary_route_bonus": float(self.buffer.boundary_route_bonus),
                "boundary_family_bonus": float(self.buffer.boundary_family_bonus),
                "trajectory_quality_bonus": float(self.buffer.trajectory_quality_bonus),
                "wasteful_end_turn_scale": float(self.buffer.wasteful_end_turn_scale),
                "wasteful_proceed_scale": float(self.buffer.wasteful_proceed_scale),
                "encounter_tier_weights": dict(sorted(self.buffer.encounter_tier_weights.items())),
                "encounter_priority_weights": dict(sorted(self.buffer.encounter_priority_weights.items())),
            },
            "recent_tail": {
                "windows": [int(window) for window in self.recent_tail_windows],
                "tracked_encounters": list(self.recent_tail_tracked_encounters),
                "min_samples": int(self.recent_tail_min_samples),
            },
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
                "combat_search_mode": str(self.mcts.combat_search_mode),
                "combat_full_root_frontier": bool(self.mcts.combat_full_root_frontier),
                "root_progressive_widening_init": int(self.mcts.root_progressive_widening_init),
                "child_progressive_widening_init": int(self.mcts.child_progressive_widening_init),
                "root_progressive_widening_growth": float(self.mcts.root_progressive_widening_growth),
                "child_progressive_widening_growth": float(self.mcts.child_progressive_widening_growth),
                "root_bias_enabled": bool(self.mcts._root_bias_enabled),
                "root_bias_min_scale": float(self.mcts.root_bias_min_scale),
                "root_bias_decay_steps": int(self.mcts.root_bias_decay_steps),
            },
        }
        (checkpoint_path / "metadata.json").write_text(json.dumps(metadata, indent=2))
        print(f"[checkpoint] Saved to {checkpoint_path}")
        self._prune_old_step_checkpoints(checkpoint_dir)

    def _prune_old_step_checkpoints(self, checkpoint_dir: Path) -> None:
        keep_last = int(self.checkpoint_keep_last)
        if keep_last <= 0:
            return

        step_dirs = sorted(
            (
                path
                for path in checkpoint_dir.iterdir()
                if path.is_dir() and path.name.startswith("muzero_step_")
            ),
            key=lambda path: path.name,
        )
        if len(step_dirs) <= keep_last:
            return

        prune_targets = step_dirs[:-keep_last]
        freed_bytes = 0
        pruned_names: list[str] = []

        for target in prune_targets:
            try:
                freed_bytes += sum(
                    file_path.stat().st_size
                    for file_path in target.rglob("*")
                    if file_path.is_file()
                )
            except FileNotFoundError:
                continue

            shutil.rmtree(target, ignore_errors=False)
            pruned_names.append(target.name)

        if pruned_names:
            freed_gb = freed_bytes / (1024 ** 3)
            preview = ", ".join(pruned_names[:4])
            suffix = " ..." if len(pruned_names) > 4 else ""
            print(
                "[checkpoint] Pruned "
                f"{len(pruned_names)} old step checkpoint(s), freed ~{freed_gb:.2f} GB "
                f"(keep_last={keep_last}): {preview}{suffix}"
            )


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
    parser.add_argument("--combat-num-simulations", type=int, default=16,
                        help="Per-decision MCTS simulations for combat domain self-play.")
    parser.add_argument("--build-num-simulations", type=int, default=12,
                        help="Per-decision MCTS simulations for build domain self-play (rewards/shop/rest/events).")
    parser.add_argument("--route-num-simulations", type=int, default=8,
                        help="Per-decision MCTS simulations for route domain self-play.")
    parser.add_argument("--root-progressive-widening-init", type=int, default=2,
                        help="Initially selectable root children before visit-based widening grows the frontier.")
    parser.add_argument("--child-progressive-widening-init", type=int, default=1,
                        help="Initially selectable non-root children before visit-based widening grows the frontier.")
    parser.add_argument("--root-progressive-widening-growth", type=float, default=1.0,
                        help="sqrt(visit)-scaled widening growth coefficient for the root.")
    parser.add_argument("--child-progressive-widening-growth", type=float, default=1.0,
                        help="sqrt(visit)-scaled widening growth coefficient for non-root nodes.")
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
    parser.add_argument("--combat-search-mode", type=str, default="grounded_root",
                        choices=["full", "grounded_root"],
                        help="Combat-domain planner mode. grounded_root keeps combat search on root-level real actions and blocks deeper latent rollout.")
    parser.add_argument("--disable-combat-full-root-frontier", action="store_true",
                        help="When combat grounded_root mode is active, keep root progressive widening instead of exposing the whole root frontier.")
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
    parser.add_argument("--checkpoint-keep-last", type=int, default=3,
                        help="Auto-prune old muzero_step_* checkpoints in the current run directory after each save; 0 disables pruning.")
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
                        help="Weight for raw hidden-state consistency between dynamics rollout and representation(next_obs).")
    parser.add_argument("--surface-mask-weight", type=float, default=1.5,
                        help="Weight for next-step legal-mask prediction loss.")
    parser.add_argument("--surface-count-weight", type=float, default=0.25,
                        help="Weight for predicted legal-count calibration loss.")
    parser.add_argument("--surface-domain-weight", type=float, default=0.15,
                        help="Weight for next-step decision-domain prediction loss.")
    parser.add_argument("--surface-phase-weight", type=float, default=0.1,
                        help="Weight for next-step phase prediction loss.")
    parser.add_argument("--settlement-weight", type=float, default=0.12,
                        help="Light episode-settlement weight, only backfilled to recent build/route decisions.")
    parser.add_argument("--settlement-decay", type=float, default=0.99,
                        help="Decay factor for backward episode settlement over recent build/route steps.")
    parser.add_argument("--settlement-max-steps", type=int, default=12,
                        help="Maximum number of recent build/route steps that receive settlement credit.")
    parser.add_argument("--boundary-replay-build-bonus", type=float, default=1.0,
                        help="Extra replay sampling bonus for build-domain starting positions.")
    parser.add_argument("--boundary-replay-route-bonus", type=float, default=1.5,
                        help="Extra replay sampling bonus for route-domain starting positions.")
    parser.add_argument("--boundary-replay-family-bonus", type=float, default=0.75,
                        help="Extra replay bonus for map/reward/shop/rest/event-style boundary actions.")
    parser.add_argument("--boundary-replay-quality-bonus", type=float, default=0.25,
                        help="Episode-level replay priority bonus for deeper / stronger full-run trajectories.")
    parser.add_argument("--wasteful-end-turn-replay-scale", type=float, default=0.35,
                        help="Replay downweight for wasteful end_turn samples that ignore playable progress.")
    parser.add_argument("--wasteful-proceed-replay-scale", type=float, default=0.55,
                        help="Replay downweight for wasteful proceed/skip samples on reward-like surfaces.")
    parser.add_argument("--replay-encounter-tier-weights", type=str, default=DEFAULT_REPLAY_ENCOUNTER_TIER_WEIGHTS,
                        help="Trajectory-level replay oversampling weights by encounter tier, e.g. weak=0.8,normal=1.1,elite=1.8,boss=1.35")
    parser.add_argument("--replay-encounter-weights", type=str, default=DEFAULT_REPLAY_ENCOUNTER_PRIORITY_WEIGHTS,
                        help="Trajectory-level replay oversampling weights for hard encounters, e.g. ENCOUNTER.X=2.5")
    parser.add_argument("--recent-tail-windows", type=str, default="64,256",
                        help="Comma-separated rolling episode windows to monitor, e.g. 64,256")
    parser.add_argument("--recent-tail-tracked-encounters", type=str, default=DEFAULT_RECENT_TAIL_TRACKED_ENCOUNTERS,
                        help="Comma-separated encounter ids to expose in recent-tail monitoring.")
    parser.add_argument("--recent-tail-min-samples", type=int, default=4,
                        help="Minimum recent samples required before per-encounter recent-tail win-rate scalars are emitted.")
    parser.add_argument("--disable-trivial-build-fast-path", action="store_true",
                        help="Disable shortcutting obvious build-domain decisions like taking gold, safe potion claims, and proceed-only screens.")

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
            "resolve this subset. Default keeps all human rows plus only local runs that cleared Act 1, "
            "then drops explicit losing-room combat snapshots."
        ),
    )
    parser.add_argument("--combat-snapshot-split", type=str, default="train")
    parser.add_argument(
        "--combat-snapshot-sample-mode",
        type=str,
        default=DEFAULT_COMBAT_SNAPSHOT_SAMPLE_MODE,
        choices=["row_uniform", "encounter_balanced", "tier_weighted_encounter_balanced"],
        help="Sampling mode for combat snapshot curriculum.",
    )
    parser.add_argument(
        "--combat-tier-weights",
        type=str,
        default=DEFAULT_COMBAT_ENCOUNTER_TIER_WEIGHTS,
        help="Optional tier sampling weights, e.g. weak=0.6,normal=0.4",
    )
    parser.add_argument(
        "--combat-encounter-weights",
        type=str,
        default=DEFAULT_COMBAT_ENCOUNTER_PRIORITY_WEIGHTS,
        help="Optional encounter weights, e.g. ENCOUNTER.BOWLBUGS_WEAK=3,ENCOUNTER.TUNNELER_WEAK=3",
    )
    parser.add_argument("--character", type=str, default=None,
                        help="Character to use for training")
    parser.add_argument("--defensive-buffs", action="store_true", default=False,
                        help="Apply defensive buffs on env/reset (mainly for debugging).")
    parser.add_argument("--combat-snapshot-character", type=str, default=None)
    parser.add_argument(
        "--combat-sandbox-potions",
        action=argparse.BooleanOptionalAction,
        default=True,
        help=(
            "Enable potion state/use in combat sandbox resets when snapshot or override potion data is available. "
            "Use --no-combat-sandbox-potions for ablations."
        ),
    )
    parser.add_argument("--session-file", type=str, default=None)
    parser.add_argument("--session-files", type=str, default=None)
    parser.add_argument("--n-envs", type=int, default=1)
    parser.add_argument("--device", type=str, default="cpu")

    # Discount factors
    parser.add_argument("--discount", type=float, default=0.997)
    parser.add_argument("--n-step-return", type=int, default=10)

    args = parser.parse_args()
    args.device = resolve_torch_device(args.device)
    args.log_dir = normalize_path_str(args.log_dir) or args.log_dir
    args.checkpoint_dir = normalize_path_str(args.checkpoint_dir) or args.checkpoint_dir
    args.resume_from = normalize_path_str(args.resume_from)
    args.combat_snapshot_dataset = normalize_path_str(args.combat_snapshot_dataset)
    args.session_file = normalize_path_str(args.session_file)

    if args.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError(
            "Requested GPU device ('cuda' / ROCm) but torch.cuda.is_available() is False. "
            "Inside WSL, make sure ROCm is installed and the WSL relay base URL is configured."
        )

    # Setup directories
    Path(args.log_dir).mkdir(parents=True, exist_ok=True)
    Path(args.checkpoint_dir).mkdir(parents=True, exist_ok=True)

    if args.device == "cuda":
        device_name = torch.cuda.get_device_name(0)
        print(f"[setup] Using GPU device: {device_name}")
    else:
        print(f"[setup] Using device: {args.device}")

    if running_in_wsl():
        print(f"[setup] WSL detected: distro={os.environ.get('WSL_DISTRO_NAME', 'unknown')}")
        print(f"[setup] STS2_BRIDGE_SESSION_FILE={os.environ.get('STS2_BRIDGE_SESSION_FILE', '<default>')}")
        print(f"[setup] STS2_BRIDGE_BASE_URL={os.environ.get('STS2_BRIDGE_BASE_URL', '<session.json base_url>')}")

    # Resolve session files
    session_files = resolve_training_session_files(
        n_envs=args.n_envs,
        session_file=args.session_file,
        session_files=parse_session_files(args.session_files),
    )

    combat_multi_env = bool(args.combat_sandbox and args.n_envs > 1)
    if not args.combat_sandbox and args.n_envs != 1:
        print(
            "[warning] Only single environment (--n-envs 1) is supported for full-run MuZero; "
            "extra session files will be ignored outside --combat-sandbox."
        )

    train_session_files = session_files if args.combat_sandbox else [session_files[0] if session_files else None]
    session_file = train_session_files[0] if train_session_files else None
    supported_encounter_ids: set[str] | None = None

    encounter_tiers = parse_encounter_tiers(args.combat_encounter_tiers)
    tier_weights = parse_tier_weights(args.combat_tier_weights)
    encounter_weights = parse_encounter_weights(args.combat_encounter_weights)
    replay_tier_weights = parse_tier_weights(args.replay_encounter_tier_weights)
    replay_encounter_weights = parse_encounter_weights(args.replay_encounter_weights)
    recent_tail_windows = parse_int_list(args.recent_tail_windows) or DEFAULT_RECENT_TAIL_WINDOWS
    recent_tail_tracked_encounters = parse_encounter_pool(args.recent_tail_tracked_encounters)
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

    def build_env_for_index(env_index: int) -> gym.Env:
        return build_train_env(
            env_index=env_index,
            session_file=train_session_files[env_index],
            combat_sandbox=args.combat_sandbox,
            combat_sandbox_potions=args.combat_sandbox_potions,
            character=args.character,
            defensive_buffs=args.defensive_buffs,
            encounter_id=None,
            encounter_pool=encounter_pool,
            snapshot_pool=snapshot_pool,
            reset_timeout_ms=15000 if args.combat_sandbox else 60000,
            step_timeout_ms=20000,
        )

    def close_env_safely(train_env: gym.Env | None, env_index: int) -> None:
        if train_env is None:
            return
        try:
            train_env.close()
        except Exception as exc:
            print(f"[cleanup] Failed to close env[{env_index}]: {exc}")

    print(f"[setup] Creating training environment...")
    envs = [build_env_for_index(env_index) for env_index in range(len(train_session_files))]
    env = envs[0]

    # Create network, MCTS, buffer
    print(f"[setup] Creating MuZero network...")
    network = MuZeroNetwork()
    network.to(args.device)

    def build_mcts_instance() -> MCTS:
        return MCTS(
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
            root_progressive_widening_init=args.root_progressive_widening_init,
            child_progressive_widening_init=args.child_progressive_widening_init,
            root_progressive_widening_growth=args.root_progressive_widening_growth,
            child_progressive_widening_growth=args.child_progressive_widening_growth,
            root_bias_min_scale=args.root_bias_min_scale,
            root_bias_decay_steps=args.root_bias_decay_steps,
            combat_search_mode=args.combat_search_mode,
            combat_full_root_frontier=not args.disable_combat_full_root_frontier,
        )

    mcts = build_mcts_instance()

    buffer = MuZeroReplayBuffer(
        capacity=args.buffer_size,
        boundary_build_bonus=args.boundary_replay_build_bonus,
        boundary_route_bonus=args.boundary_replay_route_bonus,
        boundary_family_bonus=args.boundary_replay_family_bonus,
        trajectory_quality_bonus=args.boundary_replay_quality_bonus,
        wasteful_end_turn_scale=args.wasteful_end_turn_replay_scale,
        wasteful_proceed_scale=args.wasteful_proceed_replay_scale,
        encounter_tier_weights=replay_tier_weights,
        encounter_priority_weights=replay_encounter_weights,
    )

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
        combat_num_simulations=args.combat_num_simulations,
        build_num_simulations=args.build_num_simulations,
        route_num_simulations=args.route_num_simulations,
        settlement_weight=args.settlement_weight,
        settlement_decay=args.settlement_decay,
        settlement_max_steps=args.settlement_max_steps,
        trivial_build_fast_path=not args.disable_trivial_build_fast_path,
        recent_tail_windows=recent_tail_windows,
        recent_tail_tracked_encounters=recent_tail_tracked_encounters,
        recent_tail_min_samples=args.recent_tail_min_samples,
        log_dir=args.log_dir,
        checkpoint_dir=args.checkpoint_dir,
        checkpoint_keep_last=args.checkpoint_keep_last,
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
    async_combat_actor_learner = bool(args.combat_sandbox and len(envs) > 1)
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
        "[setup] Combat planner: "
        f"mode={args.combat_search_mode} "
        f"(full_root_frontier={'off' if args.disable_combat_full_root_frontier else 'on'})"
    )
    print(
        "[setup] Root prior bias: "
        f"{'on' if root_bias_enabled else 'off'} "
        f"(objective_prior_blend={args.objective_prior_blend:.2f}, "
        f"end_turn_bias={args.end_turn_prior_bias:.2f}, "
        f"min_scale={args.root_bias_min_scale:.2f}, "
        f"decay_steps={args.root_bias_decay_steps})"
    )
    print(
        "[setup] Search budgets: "
        f"combat={args.combat_num_simulations}, "
        f"build={args.build_num_simulations}, "
        f"route={args.route_num_simulations}, "
        f"fallback={args.num_simulations}"
    )
    print(
        "[setup] Replay shaping: "
        f"build_bonus={args.boundary_replay_build_bonus:.2f}, "
        f"route_bonus={args.boundary_replay_route_bonus:.2f}, "
        f"family_bonus={args.boundary_replay_family_bonus:.2f}, "
        f"quality_bonus={args.boundary_replay_quality_bonus:.2f}, "
        f"wasteful_end_turn_scale={args.wasteful_end_turn_replay_scale:.2f}, "
        f"wasteful_proceed_scale={args.wasteful_proceed_replay_scale:.2f}"
    )
    print(
        "[setup] Replay encounter oversampling: "
        f"tier_weights={dict(sorted(replay_tier_weights.items()))} | "
        f"hard_encounters={dict(list(sorted(replay_encounter_weights.items()))[:10])}"
    )
    print(
        "[setup] Recent-tail monitor: "
        f"windows={list(recent_tail_windows)} "
        f"tracked={recent_tail_tracked_encounters} "
        f"min_samples={args.recent_tail_min_samples}"
    )
    print(
        "[setup] Episode settlement: "
        f"weight={args.settlement_weight:.3f}, "
        f"decay={args.settlement_decay:.3f}, "
        f"max_steps={args.settlement_max_steps}"
    )
    print(
        "[setup] Build fast-path: "
        f"{'on' if not args.disable_trivial_build_fast_path else 'off'} "
        "(gold=yes, safe_potion=yes, proceed_only=yes)"
    )
    print(
        "[setup] Progressive widening: "
        f"root_init={args.root_progressive_widening_init}, "
        f"child_init={args.child_progressive_widening_init}, "
        f"root_growth={args.root_progressive_widening_growth:.2f}, "
        f"child_growth={args.child_progressive_widening_growth:.2f}"
    )
    print(
        "[setup] Checkpoint retention: "
        + (
            "disabled"
            if args.checkpoint_keep_last <= 0
            else f"keep last {args.checkpoint_keep_last} muzero_step_* checkpoint(s) per run"
        )
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
    if args.combat_sandbox:
        print(
            "[setup] Combat sandbox envs: "
            f"n_envs={len(envs)} "
            f"(multi_env={'on' if combat_multi_env else 'off'})"
        )
        print(
            "[setup] Combat sandbox potions: "
            f"{'on' if args.combat_sandbox_potions else 'off'}"
        )
        print(
            "[setup] Combat collector: "
            f"{'async_actor_learner' if async_combat_actor_learner else 'single_thread'}"
        )
        if len(train_session_files) > 1:
            print(f"[setup] Session files: {train_session_files}")

    # ``--train-every`` is documented as environment-step cadence, not episode cadence.
    # Use a step-based trigger so variable-length combats do not delay updates by dozens
    # of episodes and so resumed runs keep a stable optimizer rhythm.
    last_train_step = int(trainer.total_steps)
    actor_threads: list[threading.Thread] = []
    actor_stop_event = threading.Event()
    async_episode_queue: queue.Queue[dict[str, Any]] = queue.Queue(maxsize=max(len(envs) * 2, 8))
    actor_failure_counts: list[int] = [0 for _ in envs]
    actor_completed_episodes: list[int] = [0 for _ in envs]
    latest_weight_lock = threading.Lock()
    latest_weight_state: dict[str, Any] = {
        "version": 0,
        "total_steps": int(trainer.total_steps),
        "state_dict": {},
    }

    def snapshot_network_state() -> dict[str, torch.Tensor]:
        return {
            key: value.detach().cpu().clone()
            for key, value in network.state_dict().items()
        }

    def publish_latest_weights() -> None:
        with latest_weight_lock:
            latest_weight_state["version"] = int(latest_weight_state.get("version", 0)) + 1
            latest_weight_state["total_steps"] = int(trainer.total_steps)
            latest_weight_state["state_dict"] = snapshot_network_state()

    def emit_async_env_failure(
        *,
        actor_index: int,
        failure_count: int,
        cooldown_s: float,
        message: str,
    ) -> None:
        packet = {
            "kind": "actor_recover",
            "actor_index": int(actor_index),
            "failure_count": int(failure_count),
            "cooldown_s": float(cooldown_s),
            "message": str(message),
        }
        try:
            async_episode_queue.put(packet, timeout=0.1)
        except queue.Full:
            pass

    def actor_thread_main(actor_index: int) -> None:
        actor_network = MuZeroNetwork()
        actor_network.to(args.device)
        actor_optimizer = optim.SGD(actor_network.parameters(), lr=0.0)
        actor_buffer = EpisodeCaptureBuffer()
        actor_log_dir = str(Path(args.log_dir) / "async_actor_scratch" / f"actor_{actor_index}")
        actor_ckpt_dir = str(Path(args.checkpoint_dir) / "async_actor_scratch" / f"actor_{actor_index}")
        actor_trainer = MuZeroTrainer(
            network=actor_network,
            mcts=build_mcts_instance(),
            buffer=actor_buffer,  # type: ignore[arg-type]
            env=envs[actor_index],
            optimizer=actor_optimizer,
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
            combat_num_simulations=args.combat_num_simulations,
            build_num_simulations=args.build_num_simulations,
            route_num_simulations=args.route_num_simulations,
            settlement_weight=args.settlement_weight,
            settlement_decay=args.settlement_decay,
            settlement_max_steps=args.settlement_max_steps,
            trivial_build_fast_path=not args.disable_trivial_build_fast_path,
            recent_tail_windows=recent_tail_windows,
            recent_tail_tracked_encounters=recent_tail_tracked_encounters,
            recent_tail_min_samples=args.recent_tail_min_samples,
            log_dir=actor_log_dir,
            checkpoint_dir=actor_ckpt_dir,
            checkpoint_keep_last=0,
        )
        try:
            actor_trainer.writer.close()
        except Exception:
            pass
        actor_trainer.writer = NullSummaryWriter()

        loaded_weight_version = -1
        while not actor_stop_event.is_set():
            try:
                latest_total_steps = int(trainer.total_steps)
                state_snapshot: dict[str, torch.Tensor] | None = None
                with latest_weight_lock:
                    weight_version = int(latest_weight_state.get("version", 0))
                    latest_total_steps = int(latest_weight_state.get("total_steps", trainer.total_steps))
                    if weight_version != loaded_weight_version:
                        state_snapshot = {
                            key: value.clone()
                            for key, value in latest_weight_state.get("state_dict", {}).items()
                        }

                if state_snapshot is not None:
                    actor_network.load_state_dict(
                        {
                            key: value.to(args.device)
                            for key, value in state_snapshot.items()
                        }
                    )
                    loaded_weight_version = weight_version

                actor_trainer.total_steps = int(latest_total_steps)
                temperature = actor_trainer.compute_temperature(latest_total_steps, args.total_timesteps)
                ep_reward, ep_length = actor_trainer.self_play_episode(temperature=temperature)
                trajectory = actor_buffer.pop_latest()
                if trajectory is None:
                    raise RuntimeError(f"Async combat actor {actor_index} produced no trajectory")

                actor_failure_counts[actor_index] = 0
                actor_completed_episodes[actor_index] += 1
                packet = {
                    "kind": "episode",
                    "actor_index": int(actor_index),
                    "episode_reward": float(ep_reward),
                    "episode_length": int(ep_length),
                    "temperature": float(temperature),
                    "trajectory": trajectory,
                    "episode_metrics": dict(actor_trainer.last_episode_metrics or {}),
                    "weight_version": int(loaded_weight_version),
                    "actor_episode_count": int(actor_completed_episodes[actor_index]),
                }
                while not actor_stop_event.is_set():
                    try:
                        async_episode_queue.put(packet, timeout=0.5)
                        break
                    except queue.Full:
                        continue
            except BridgeError as exc:
                actor_failure_counts[actor_index] += 1
                failure_count = int(actor_failure_counts[actor_index])
                cooldown_s = min(60.0, 5.0 * float(failure_count))
                emit_async_env_failure(
                    actor_index=actor_index,
                    failure_count=failure_count,
                    cooldown_s=cooldown_s,
                    message=(
                        f"[warn] Async combat actor[{actor_index}] bridge failure: {exc} "
                        f"| cooldown={cooldown_s:.1f}s | failures={failure_count}"
                    ),
                )
                close_env_safely(envs[actor_index], actor_index)
                deadline = time.time() + cooldown_s
                while not actor_stop_event.is_set() and time.time() < deadline:
                    time.sleep(0.25)
                if actor_stop_event.is_set():
                    break
                try:
                    envs[actor_index] = build_env_for_index(actor_index)
                    actor_trainer.env = envs[actor_index]
                    emit_async_env_failure(
                        actor_index=actor_index,
                        failure_count=failure_count,
                        cooldown_s=0.0,
                        message=f"[recover] Rebuilt async combat env[{actor_index}] bound to {train_session_files[actor_index]}",
                    )
                except Exception as rebuild_exc:
                    emit_async_env_failure(
                        actor_index=actor_index,
                        failure_count=failure_count,
                        cooldown_s=cooldown_s,
                        message=f"[recover] Failed to rebuild async combat env[{actor_index}]: {rebuild_exc}",
                    )
            except Exception as exc:
                try:
                    async_episode_queue.put(
                        {
                            "kind": "actor_fatal",
                            "actor_index": int(actor_index),
                            "message": f"Async combat actor[{actor_index}] crashed: {type(exc).__name__}: {exc}",
                        },
                        timeout=0.1,
                    )
                except queue.Full:
                    pass
                actor_stop_event.set()
                break

    def log_async_episode_scalars(actor_index: int, episode_metrics: dict[str, Any]) -> None:
        metric_map = {
            "max_floor": "episode/max_floor",
            "max_act_id": "episode/max_act_id",
            "rooms_seen": "episode/rooms_seen",
            "death_floor": "episode/death_floor",
            "elite_rooms_seen": "episode/elite_rooms_seen",
            "settlement_signal": "episode/settlement_signal",
            "settlement_applied_steps": "episode/settlement_applied_steps",
            "settlement_total_bonus": "episode/settlement_total_bonus",
            "episode_reward_augmented": "episode/reward_augmented",
        }
        for metric_key, writer_key in metric_map.items():
            if metric_key in episode_metrics:
                trainer.writer.add_scalar(writer_key, float(episode_metrics.get(metric_key, 0.0) or 0.0), trainer.episode_count)
        for metric_key, writer_key in (
            ("act1_boss_seen", "episode/act1_boss_seen"),
            ("act1_clear", "episode/act1_clear"),
        ):
            if metric_key in episode_metrics:
                trainer.writer.add_scalar(writer_key, 1.0 if bool(episode_metrics.get(metric_key)) else 0.0, trainer.episode_count)
        trainer.writer.add_scalar(f"env/{actor_index}_episodes", float(actor_completed_episodes[actor_index]), trainer.total_steps)

    # Training loop
    try:
        if async_combat_actor_learner:
            publish_latest_weights()
            for actor_index in range(len(envs)):
                actor_thread = threading.Thread(
                    target=actor_thread_main,
                    args=(actor_index,),
                    name=f"combat-actor-{actor_index}",
                    daemon=True,
                )
                actor_thread.start()
                actor_threads.append(actor_thread)

        while trainer.total_steps < args.total_timesteps:
            active_env_index = 0
            temperature = trainer.compute_temperature(trainer.total_steps, args.total_timesteps)
            if async_combat_actor_learner:
                try:
                    packet = async_episode_queue.get(timeout=5.0)
                except queue.Empty:
                    print("[warn] Async combat collector idle for 5s; actors are still gathering trajectories")
                    continue

                packet_kind = str(packet.get("kind") or "episode")
                if packet_kind == "actor_recover":
                    active_env_index = int(packet.get("actor_index", 0) or 0)
                    print(str(packet.get("message") or ""))
                    trainer.writer.add_scalar("env/failure_total", float(sum(actor_failure_counts)), trainer.total_steps)
                    trainer.writer.add_scalar(
                        f"env/{active_env_index}_failure_count",
                        float(packet.get("failure_count", 0.0) or 0.0),
                        trainer.total_steps,
                    )
                    trainer.writer.add_scalar(
                        f"env/{active_env_index}_cooldown_seconds",
                        float(packet.get("cooldown_s", 0.0) or 0.0),
                        trainer.total_steps,
                    )
                    continue
                if packet_kind == "actor_fatal":
                    raise RuntimeError(str(packet.get("message") or "Async combat actor crashed"))

                active_env_index = int(packet.get("actor_index", 0) or 0)
                ep_reward = float(packet.get("episode_reward", 0.0) or 0.0)
                ep_length = int(packet.get("episode_length", 0) or 0)
                temperature = float(packet.get("temperature", temperature) or temperature)
                trajectory = packet.get("trajectory")
                if not isinstance(trajectory, GameTrajectory):
                    raise RuntimeError(f"Async combat actor[{active_env_index}] returned invalid trajectory payload")

                buffer.save_episode(
                    trajectory,
                    discount=trainer.discount,
                    n_steps=trainer.n_step_return,
                )
                trainer.total_steps += ep_length
                trainer.episode_count += 1
                trainer.last_episode_metrics = (
                    packet.get("episode_metrics")
                    if isinstance(packet.get("episode_metrics"), dict)
                    else {}
                )
                recent_tail_snapshot = trainer.record_recent_combat_episode(getattr(trajectory, "metadata", None))
                if recent_tail_snapshot:
                    trainer.last_episode_metrics["recent_tail"] = recent_tail_snapshot

                trainer.writer.add_scalar("episode/reward", ep_reward, trainer.episode_count)
                trainer.writer.add_scalar("episode/length", ep_length, trainer.episode_count)
                trainer.writer.add_scalar("schedule/temperature", temperature, trainer.total_steps)
                trainer.writer.add_scalar("schedule/active_env_index", float(active_env_index), trainer.total_steps)
                trainer.writer.add_scalar("schedule/weight_version", float(packet.get("weight_version", 0) or 0), trainer.total_steps)
                log_async_episode_scalars(active_env_index, trainer.last_episode_metrics or {})
            else:
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
                if async_combat_actor_learner:
                    publish_latest_weights()
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
                fast_path_counts = episode_metrics.get("fast_path_counts") if isinstance(episode_metrics, dict) else {}
                fast_path_total = int(episode_metrics.get("fast_path_total", 0)) if isinstance(episode_metrics, dict) else 0
                fast_path_reason_counts = episode_metrics.get("fast_path_reason_counts") if isinstance(episode_metrics, dict) else {}
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
                if fast_path_total > 0 and isinstance(fast_path_counts, dict):
                    reason_summary = ""
                    if isinstance(fast_path_reason_counts, dict) and fast_path_reason_counts:
                        ranked_reasons = sorted(
                            fast_path_reason_counts.items(),
                            key=lambda item: (-int(item[1]), str(item[0])),
                        )
                        reason_summary = " | reasons=" + ",".join(
                            f"{reason}:{int(count)}"
                            for reason, count in ranked_reasons[:3]
                        )
                    print(
                        "[train] FastPath | "
                        f"total={fast_path_total} "
                        + " ".join(
                            f"{domain}={int(fast_path_counts.get(domain, 0))}"
                            for domain in DECISION_DOMAINS
                        )
                        + reason_summary
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
                recent_tail = episode_metrics.get("recent_tail") if isinstance(episode_metrics, dict) else None
                if isinstance(recent_tail, dict) and recent_tail:
                    tail_parts: list[str] = []
                    for window in sorted(recent_tail.keys()):
                        window_stats = recent_tail.get(window) if isinstance(recent_tail.get(window), dict) else None
                        if not window_stats:
                            continue
                        groups = window_stats.get("groups") if isinstance(window_stats.get("groups"), dict) else {}
                        tiers = window_stats.get("tiers") if isinstance(window_stats.get("tiers"), dict) else {}
                        elite_stats = tiers.get("elite") if isinstance(tiers.get("elite"), dict) else {}
                        normal_stats = tiers.get("normal") if isinstance(tiers.get("normal"), dict) else {}
                        hard_normal_stats = groups.get("hard_normal") if isinstance(groups.get("hard_normal"), dict) else {}
                        hard_elite_stats = groups.get("hard_elite") if isinstance(groups.get("hard_elite"), dict) else {}
                        tail_parts.append(
                            f"{window}[win={float(window_stats.get('win_rate', 0.0)):.3f} "
                            f"reward={float(window_stats.get('reward_mean', 0.0)):+.3f} "
                            f"normal={float(normal_stats.get('win_rate', 0.0)):.3f} "
                            f"elite={float(elite_stats.get('win_rate', 0.0)):.3f} "
                            f"hardN={float(hard_normal_stats.get('win_rate', 0.0)):.3f} "
                            f"hardE={float(hard_elite_stats.get('win_rate', 0.0)):.3f}]"
                        )
                    if tail_parts:
                        print("[train] RecentTail | " + " | ".join(tail_parts))

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
    else:
        print(f"[train] Finished training after {trainer.total_steps} steps")
        trainer.save_checkpoint(tag="final")
    finally:
        actor_stop_event.set()
        for actor_thread in actor_threads:
            actor_thread.join(timeout=5.0)
        for env_index, train_env in enumerate(envs):
            close_env_safely(train_env, env_index)
        trainer.writer.close()


if __name__ == "__main__":
    main()
