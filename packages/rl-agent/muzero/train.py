"""MuZero / token-memory / search-free planner training for Slay the Spire 2.

Usage:
    python -m muzero.train --total-timesteps 100000 --combat-sandbox \\
        --encounter-pool "ENCOUNTER.SLIMES_WEAK,ENCOUNTER.SHRINKER_BEETLE_WEAK"

This module contains the MuZero replay-buffer training stack, including the
legacy MCTS path and the newer token-memory / JEPA-style / search-free combat
direct-policy path.  New automation should prefer the module entrypoint above;
``train_muzero.py`` remains as a compatibility wrapper.
"""

from __future__ import annotations

import argparse
import copy
from collections import Counter, defaultdict, deque
from contextlib import nullcontext
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
from sts2_env.boss_mechanics import build_boss_mechanics_context
from sts2_env.combat_env import CombatSandboxEnv
from sts2_env.env_v2 import SlayTheSpire2EnvV2
from muzero.sts2_env.mcts import MCTS
from muzero.sts2_env.muzero_buffer import GameTrajectory, MuZeroReplayBuffer
from muzero.sts2_env.muzero_model import (
    MuZeroNetwork,
    RecurrentMuZeroOutput,
    scalar_to_support,
    support_tensor_to_scalar,
    support_to_scalar,
)
from muzero.sts2_env.latent_regularizers import (
    latent_gaussian_regularizer,
    slot_latent_gaussian_regularizer,
)
from sts2_env.objective_heads import (
    NUM_OBJECTIVE_HEADS,
    compute_transition_objective_rewards,
)
from sts2_env.observation_v2 import DECISION_DOMAINS, DictObservationEncoder, MAX_ACTIONS, NUM_PHASES
from sts2_env.card_effect_profile import aggregate_card_effect_profile_semantics
from sts2_env.semantic_action import SEMANTIC_ACTION_FAMILIES, SEMANTIC_ROLE_NAMES
from sts2_env.observation_v3 import WorldTokenObservationEncoder
from sts2_env.potion_profiles import DEFAULT_EFFECT_PROFILE as _POTION_EFFECT_DEFAULT, get_potion_profile as _get_potion_profile
from sts2_env.path_utils import normalize_path_str, resolve_torch_device, running_in_wsl
from muzero.sts2_env.semantic_rollout import (
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
    "ENCOUNTER.CEREMONIAL_BEAST_BOSS=3.40",
    "ENCOUNTER.KAISER_CRAB_BOSS=3.20",
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
    "ENCOUNTER.CEREMONIAL_BEAST_BOSS=5.20",
    "ENCOUNTER.KAISER_CRAB_BOSS=4.80",
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
                writer.add_scalar(
                    f"{prefix}/{tier}_sample_count",
                    float(tier_stats.get("episodes", 0.0)),
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
                writer.add_scalar(
                    f"{prefix}/encounter_{safe_encounter}_sample_count",
                    float(encounter_stats.get("episodes", 0.0)),
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
    obs_mode: str,
) -> gym.Env:
    """Build a single training environment."""
    obs_mode = str(obs_mode or "dense_v2").strip().lower()
    if obs_mode == "token_v3":
        obs_encoder = WorldTokenObservationEncoder(use_text=False)
    else:
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
    load_optimizer: bool = True,
    token_target_encoder: nn.Module | None = None,
    amp_grad_scaler: torch.amp.GradScaler | None = None,
) -> dict[str, Any]:
    """Load network/optimizer/(optional) replay buffer from a MuZero checkpoint directory.

    Set ``load_optimizer=False`` to warm-start weights only and re-initialize
    the optimizer (Adam moments, etc.) — useful when the aux-target schema
    changed and the old momentum is steering the loss landscape away from
    the new objective.
    """
    checkpoint_path = Path(resume_from)
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Resume checkpoint path not found: {checkpoint_path}")

    network_path = checkpoint_path / "network.pt"
    token_target_encoder_path = checkpoint_path / "token_target_encoder.pt"
    optimizer_path = checkpoint_path / "optimizer.pt"
    amp_scaler_path = checkpoint_path / "amp_scaler.pt"
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

    if allow_exact_resume and load_optimizer:
        try:
            optimizer.load_state_dict(optimizer_state)
        except Exception as exc:
            print(f"[resume] Optimizer state not loaded ({exc}); starting with fresh optimizer.")
    elif allow_exact_resume and not load_optimizer:
        print(
            "[resume] Skipping optimizer warm-start by request; starting with fresh optimizer."
        )
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

    if token_target_encoder is not None:
        if token_target_encoder_path.exists():
            target_state = torch.load(token_target_encoder_path, map_location=device)
            current_target_state = token_target_encoder.state_dict()
            merged_target_state = dict(current_target_state)
            merged_target_state.update(
                {
                    key: value
                    for key, value in target_state.items()
                    if key in current_target_state and tuple(current_target_state[key].shape) == tuple(value.shape)
                }
            )
            token_target_encoder.load_state_dict(merged_target_state)
        else:
            token_target_encoder.load_state_dict(network.token_encoder.state_dict())
    if (
        allow_exact_resume
        and load_optimizer
        and amp_grad_scaler is not None
        and amp_scaler_path.exists()
        and bool(amp_grad_scaler.is_enabled())
    ):
        try:
            amp_grad_scaler.load_state_dict(torch.load(amp_scaler_path, map_location=device))
            print("[resume] AMP GradScaler state restored.")
        except Exception as exc:
            print(f"[resume] AMP GradScaler state not loaded ({exc}); starting with fresh scaler.")

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
        planner_q_loss_weight: float = 0.75,
        planner_objective_q_loss_weight: float = 0.75,
        objective_value_weight: float = 1.0,
        objective_reward_weight: float = 0.75,
        semantic_policy_weight: float = 1.0,
        semantic_value_weight: float = 1.0,
        semantic_reward_weight: float = 0.75,
        semantic_state_consistency_weight: float = 2.0,
        objective_diversity_weight: float = 0.05,
        semantic_policy_label_smoothing: float = 0.02,
        state_consistency_weight: float = 2.0,
        future_bank_state_weight: float = 1.0,
        future_bank_delta_weight: float = 0.5,
        future_bank_occupancy_weight: float = 0.25,
        future_bank_token_presence_weight: float = 0.25,
        future_bank_token_distribution_weight: float = 0.2,
        future_bank_token_slot_state_weight: float = 0.5,
        future_bank_token_slot_mask_weight: float = 0.15,
        future_bank_token_slot_type_weight: float = 0.2,
        future_bank_token_slot_zone_weight: float = 0.15,
        future_bank_token_slot_source_weight: float = 0.2,
        future_world_aux_weight: float = 1.0,
        future_world_rollout_weight: float = 0.35,
        future_world_rollout_steps: int = 2,
        future_world_rollout_decay: float = 0.7,
        token_teacher_ema_decay: float = 0.995,
        latent_gaussian_reg_weight: float = 0.005,
        latent_gaussian_reg_projections: int = 64,
        latent_gaussian_reg_slot_weight: float = 0.25,
        latent_gaussian_reg_dynamics_weight: float = 0.5,
        latent_gaussian_reg_cov_weight: float = 0.05,
        surprise_loss_weight: float = 0.25,
        surprise_hidden_scale: float = 100.0,
        surprise_surface_scale: float = 5.0,
        surprise_future_aux_scale: float = 1.0,
        surprise_target_cap: float = 50.0,
        surface_mask_weight: float = 1.5,
        surface_count_weight: float = 0.25,
        surface_domain_weight: float = 0.15,
        surface_phase_weight: float = 0.1,
        combat_direct_policy: bool = False,
        combat_rollout_q_blend: float = 0.75,
        combat_rollout_objective_q_blend: float = 0.5,
        combat_rollout_risk_blend: float = 0.35,
        combat_rollout_steps: int = 2,
        combat_rollout_beam_width: int = 2,
        combat_rollout_legal_logit_scale: float = 0.75,
        combat_rollout_uncertainty_blend: float = 0.35,
        combat_rollout_uncertainty_surprise_weight: float = 1.0,
        combat_rollout_uncertainty_surface_weight: float = 0.10,
        combat_rollout_uncertainty_latent_weight: float = 0.05,
        combat_rollout_uncertainty_disagreement_weight: float = 0.25,
        combat_rollout_continuation_uncertainty_penalty: float = 0.25,
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
        mixed_precision: str = "auto",
        amp_init_scale: float = 65536.0,
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
        self.planner_q_loss_weight = max(float(planner_q_loss_weight), 0.0)
        self.planner_objective_q_loss_weight = max(float(planner_objective_q_loss_weight), 0.0)
        self.objective_value_weight = max(float(objective_value_weight), 0.0)
        self.objective_reward_weight = max(float(objective_reward_weight), 0.0)
        self.semantic_policy_weight = max(float(semantic_policy_weight), 0.0)
        self.semantic_value_weight = max(float(semantic_value_weight), 0.0)
        self.semantic_reward_weight = max(float(semantic_reward_weight), 0.0)
        self.semantic_state_consistency_weight = max(float(semantic_state_consistency_weight), 0.0)
        self.objective_diversity_weight = max(float(objective_diversity_weight), 0.0)
        self.semantic_policy_label_smoothing = min(max(float(semantic_policy_label_smoothing), 0.0), 0.25)
        self.state_consistency_weight = state_consistency_weight
        self.future_bank_state_weight = max(float(future_bank_state_weight), 0.0)
        self.future_bank_delta_weight = max(float(future_bank_delta_weight), 0.0)
        self.future_bank_occupancy_weight = max(float(future_bank_occupancy_weight), 0.0)
        self.future_bank_token_presence_weight = max(float(future_bank_token_presence_weight), 0.0)
        self.future_bank_token_distribution_weight = max(float(future_bank_token_distribution_weight), 0.0)
        self.future_bank_token_slot_state_weight = max(float(future_bank_token_slot_state_weight), 0.0)
        self.future_bank_token_slot_mask_weight = max(float(future_bank_token_slot_mask_weight), 0.0)
        self.future_bank_token_slot_type_weight = max(float(future_bank_token_slot_type_weight), 0.0)
        self.future_bank_token_slot_zone_weight = max(float(future_bank_token_slot_zone_weight), 0.0)
        self.future_bank_token_slot_source_weight = max(float(future_bank_token_slot_source_weight), 0.0)
        self.future_world_aux_weight = max(float(future_world_aux_weight), 0.0)
        self.future_world_rollout_weight = max(float(future_world_rollout_weight), 0.0)
        self.future_world_rollout_steps = max(int(future_world_rollout_steps), 0)
        self.future_world_rollout_decay = float(np.clip(future_world_rollout_decay, 0.0, 1.0))
        self.token_teacher_ema_decay = float(np.clip(token_teacher_ema_decay, 0.0, 1.0))
        self.latent_gaussian_reg_weight = max(float(latent_gaussian_reg_weight), 0.0)
        self.latent_gaussian_reg_projections = max(int(latent_gaussian_reg_projections), 0)
        self.latent_gaussian_reg_slot_weight = max(float(latent_gaussian_reg_slot_weight), 0.0)
        self.latent_gaussian_reg_dynamics_weight = max(float(latent_gaussian_reg_dynamics_weight), 0.0)
        self.latent_gaussian_reg_cov_weight = max(float(latent_gaussian_reg_cov_weight), 0.0)
        self.surprise_loss_weight = max(float(surprise_loss_weight), 0.0)
        self.surprise_hidden_scale = max(float(surprise_hidden_scale), 1e-6)
        self.surprise_surface_scale = max(float(surprise_surface_scale), 0.0)
        self.surprise_future_aux_scale = max(float(surprise_future_aux_scale), 0.0)
        self.surprise_target_cap = max(float(surprise_target_cap), 0.0)
        self.surface_mask_weight = surface_mask_weight
        self.surface_count_weight = max(float(surface_count_weight), 0.0)
        self.surface_domain_weight = surface_domain_weight
        self.surface_phase_weight = surface_phase_weight
        self.combat_direct_policy = bool(combat_direct_policy)
        self.combat_rollout_q_blend = max(float(combat_rollout_q_blend), 0.0)
        self.combat_rollout_objective_q_blend = max(float(combat_rollout_objective_q_blend), 0.0)
        self.combat_rollout_risk_blend = max(float(combat_rollout_risk_blend), 0.0)
        self.combat_rollout_steps = max(int(combat_rollout_steps), 1)
        self.combat_rollout_beam_width = max(int(combat_rollout_beam_width), 1)
        self.combat_rollout_legal_logit_scale = float(combat_rollout_legal_logit_scale)
        self.combat_rollout_uncertainty_blend = max(float(combat_rollout_uncertainty_blend), 0.0)
        self.combat_rollout_uncertainty_surprise_weight = max(float(combat_rollout_uncertainty_surprise_weight), 0.0)
        self.combat_rollout_uncertainty_surface_weight = max(float(combat_rollout_uncertainty_surface_weight), 0.0)
        self.combat_rollout_uncertainty_latent_weight = max(float(combat_rollout_uncertainty_latent_weight), 0.0)
        self.combat_rollout_uncertainty_disagreement_weight = max(float(combat_rollout_uncertainty_disagreement_weight), 0.0)
        self.combat_rollout_continuation_uncertainty_penalty = max(
            float(combat_rollout_continuation_uncertainty_penalty),
            0.0,
        )
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
        (
            self.mixed_precision,
            self.amp_enabled,
            self.amp_device_type,
            self.amp_dtype,
            self.amp_scaler_enabled,
        ) = self._resolve_mixed_precision(mixed_precision)
        self.amp_grad_scaler = torch.amp.GradScaler(
            self.amp_device_type,
            init_scale=float(amp_init_scale),
            enabled=bool(self.amp_scaler_enabled),
        )

        self.writer = SummaryWriter(log_dir=log_dir)
        self.total_steps = 0
        self.episode_count = 0
        self.last_episode_metrics: dict[str, Any] = {}
        self._kaiser_facing_debug_dump_count = 0
        self._end_turn_debug_dump_count = 0
        self._end_turn_context_dump_count = 0
        try:
            self._end_turn_context_dump_max = int(os.environ.get("MUZERO_END_TURN_CONTEXT_DUMP_MAX", "50000"))
        except (TypeError, ValueError):
            self._end_turn_context_dump_max = 50000
        self._end_turn_context_dump_disabled = (
            os.environ.get("MUZERO_END_TURN_CONTEXT_DUMP", "1").strip() == "0"
        )
        self._action_offender_dump_count = 0
        try:
            self._action_offender_dump_max = int(os.environ.get("MUZERO_ACTION_OFFENDER_DUMP_MAX", "100000"))
        except (TypeError, ValueError):
            self._action_offender_dump_max = 100000
        self._action_offender_dump_disabled = (
            os.environ.get("MUZERO_ACTION_OFFENDER_DUMP", "1").strip() == "0"
        )
        self._potion_transition_dump_count = 0
        try:
            self._potion_transition_dump_max = int(os.environ.get("MUZERO_POTION_TRANSITION_DUMP_MAX", "200000"))
        except (TypeError, ValueError):
            self._potion_transition_dump_max = 200000
        self._potion_transition_dump_disabled = (
            os.environ.get("MUZERO_POTION_TRANSITION_DUMP", "1").strip() == "0"
        )
        self.recent_combat_monitor = RecentCombatMonitor(
            windows=self.recent_tail_windows,
            tracked_encounters=self.recent_tail_tracked_encounters,
            min_samples=self.recent_tail_min_samples,
        )
        self.token_target_encoder = self._build_token_target_encoder()
        self._sync_token_target_encoder(hard=True)

    def _build_token_target_encoder(self) -> nn.Module | None:
        if not getattr(self.network, "is_token_mode", False):
            return None
        target_encoder = copy.deepcopy(self.network.token_encoder)
        target_encoder.to(self.device)
        target_encoder.eval()
        for param in target_encoder.parameters():
            param.requires_grad_(False)
        return target_encoder

    def _sync_token_target_encoder(self, *, hard: bool = False) -> None:
        if self.token_target_encoder is None:
            return
        decay = 0.0 if hard else self.token_teacher_ema_decay
        if hard or decay <= 0.0:
            self.token_target_encoder.load_state_dict(self.network.token_encoder.state_dict())
            return
        with torch.no_grad():
            target_state = self.token_target_encoder.state_dict()
            online_state = self.network.token_encoder.state_dict()
            updated_state: dict[str, torch.Tensor] = {}
            for key, target_value in target_state.items():
                online_value = online_state[key]
                if not torch.is_floating_point(target_value):
                    updated_state[key] = online_value.detach().clone()
                else:
                    updated_state[key] = target_value.detach().mul(decay).add(online_value.detach(), alpha=1.0 - decay)
            self.token_target_encoder.load_state_dict(updated_state)

    def _encode_target_token_obs(self, obs: dict[str, torch.Tensor] | None) -> Any:
        if self.token_target_encoder is None or obs is None:
            return None
        with torch.no_grad():
            return self.token_target_encoder(obs)

    def load_target_encoder_checkpoint(self, checkpoint_path: str | Path) -> None:
        if self.token_target_encoder is None:
            return
        target_path = Path(checkpoint_path) / "token_target_encoder.pt"
        if not target_path.exists():
            self._sync_token_target_encoder(hard=True)
            return
        target_state = torch.load(target_path, map_location=self.device)
        current_state = self.token_target_encoder.state_dict()
        merged_state = dict(current_state)
        merged_state.update(
            {
                key: value
                for key, value in target_state.items()
                if key in current_state and tuple(current_state[key].shape) == tuple(value.shape)
            }
        )
        self.token_target_encoder.load_state_dict(merged_state)

    def _resolve_mixed_precision(
        self,
        mode: str,
    ) -> tuple[str, bool, str, torch.dtype | None, bool]:
        """Resolve user AMP mode into autocast/scaler settings.

        Target-state token-memory training is activation-heavy.  ``auto`` is
        intentionally GPU-first: CUDA uses bf16 when available (no scaler,
        wider dynamic range), otherwise fp16 with GradScaler.  CPU defaults to
        fp32/off because bf16 CPU autocast is usually slower for this workload
        and does not address the user's VRAM bottleneck.
        """

        raw_mode = str(mode or "auto").strip().lower()
        alias = {
            "amp": "auto",
            "mixed": "auto",
            "mixed_precision": "auto",
            "none": "off",
            "false": "off",
            "0": "off",
            "fp32": "off",
            "float32": "off",
            "bfloat16": "bf16",
            "float16": "fp16",
            "half": "fp16",
        }
        normalized = alias.get(raw_mode, raw_mode)
        if normalized not in {"auto", "off", "bf16", "fp16"}:
            raise ValueError(
                f"Unsupported --mixed-precision mode: {mode!r}. "
                "Expected one of: auto, off/fp32, bf16, fp16."
            )

        try:
            device_type = torch.device(str(self.device)).type
        except (TypeError, RuntimeError):
            device_type = "cuda" if str(self.device).startswith("cuda") else "cpu"

        if normalized == "off":
            return "off", False, device_type, None, False

        if normalized == "auto":
            if device_type == "cuda" and torch.cuda.is_available():
                bf16_supported = False
                try:
                    bf16_supported = bool(torch.cuda.is_bf16_supported())
                except Exception:
                    bf16_supported = False
                if bf16_supported:
                    return "bf16", True, device_type, torch.bfloat16, False
                return "fp16", True, device_type, torch.float16, True
            return "off", False, device_type, None, False

        if normalized == "bf16":
            if device_type not in {"cuda", "cpu"}:
                print(f"[amp] bf16 requested on {device_type}; disabling mixed precision.")
                return "off", False, device_type, None, False
            return "bf16", True, device_type, torch.bfloat16, False

        # fp16 is only a target implementation for CUDA.  CPU fp16 autocast is
        # both slow and numerically fragile for the support/log-softmax losses.
        if device_type != "cuda" or not torch.cuda.is_available():
            print("[amp] fp16 requested on non-CUDA device; disabling mixed precision.")
            return "off", False, device_type, None, False
        return "fp16", True, device_type, torch.float16, True

    def _amp_autocast(self):
        if not self.amp_enabled or self.amp_dtype is None:
            return nullcontext()
        return torch.amp.autocast(
            device_type=self.amp_device_type,
            dtype=self.amp_dtype,
            enabled=True,
        )

    def _device_memory_stats(self) -> dict[str, float]:
        """Return CUDA/HIP caching allocator stats in GiB.

        PyTorch exposes ROCm memory accounting through ``torch.cuda`` as well,
        so these metrics are valid for both NVIDIA CUDA and HIP builds.  They
        make allocator fragmentation visible: a large
        reserved_minus_allocated_gb gap means the caching allocator holds blocks
        that are not currently live tensors.
        """

        try:
            device = torch.device(str(self.device))
        except (TypeError, RuntimeError):
            return {}
        if device.type != "cuda" or not torch.cuda.is_available():
            return {}
        try:
            if device.index is None:
                device_index: int | torch.device = torch.cuda.current_device()
            else:
                device_index = device
            gib = float(1024 ** 3)
            allocated = float(torch.cuda.memory_allocated(device_index)) / gib
            reserved = float(torch.cuda.memory_reserved(device_index)) / gib
            max_allocated = float(torch.cuda.max_memory_allocated(device_index)) / gib
            max_reserved = float(torch.cuda.max_memory_reserved(device_index)) / gib
            return {
                "allocated_gb": allocated,
                "reserved_gb": reserved,
                "reserved_minus_allocated_gb": max(reserved - allocated, 0.0),
                "max_allocated_gb": max_allocated,
                "max_reserved_gb": max_reserved,
            }
        except Exception:
            return {}

    @staticmethod
    def _with_decision_domain(
        obs: dict[str, Any] | None,
        domain: str,
    ) -> dict[str, Any] | None:
        if not isinstance(obs, dict) or domain not in DECISION_DOMAINS:
            return obs
        patched = dict(obs)
        index = DECISION_DOMAINS.index(domain)
        existing = obs.get("decision_domain")
        if isinstance(existing, torch.Tensor):
            vector = torch.zeros_like(existing)
            if vector.ndim == 0:
                vector = torch.zeros((len(DECISION_DOMAINS),), dtype=torch.float32, device=existing.device)
            if vector.shape[-1] == len(DECISION_DOMAINS):
                vector[..., index] = 1.0
            else:
                vector = torch.zeros((len(DECISION_DOMAINS),), dtype=existing.dtype, device=existing.device)
                vector[index] = 1.0
            patched["decision_domain"] = vector
            return patched
        dtype = np.asarray(existing).dtype if existing is not None else np.float32
        if dtype == np.dtype("O"):
            dtype = np.float32
        vector_np = np.zeros((len(DECISION_DOMAINS),), dtype=dtype)
        vector_np[index] = 1.0
        patched["decision_domain"] = vector_np.astype(np.float32, copy=False)
        return patched

    @staticmethod
    def _looks_like_combat_decision(
        obs: dict[str, Any] | None,
        legal_actions: list[Any] | None,
        info: dict[str, Any] | None = None,
    ) -> bool:
        """Infer combat even when the encoded decision_domain vector is stale.

        The bridge often reports the actionable combat screen as phase
        ``actions``.  The old encoder only treated phase == ``combat`` as
        combat, so combat-sandbox decisions were stored as build.  This helper
        uses the collector-side info/action surface, not just the encoded
        vector, so direct combat policy cannot be silently bypassed.
        """

        info = info if isinstance(info, dict) else {}
        phase = str(info.get("phase") or "").strip().lower()
        if phase == "combat":
            return True
        if str(info.get("episode_mode") or "").strip().lower() == "combat_sandbox":
            return True
        transition_state = info.get("transition_state") if isinstance(info.get("transition_state"), dict) else {}
        transition_combat = transition_state.get("combat") if isinstance(transition_state, dict) else None
        if isinstance(transition_combat, dict) and transition_combat:
            if phase in {"actions", "combat", "card_selection", "settling", ""}:
                return True

        raw_combat = obs.get("combat") if isinstance(obs, dict) else None
        if isinstance(raw_combat, dict) and raw_combat:
            return True

        combat_kinds = {
            "play_card",
            "use_potion",
            "discard_potion",
            "combat",
            "combat_select",
            "combat_select_card",
        }
        combat_action_ids = {"end_turn"}
        for action in (legal_actions or [])[:MAX_ACTIONS]:
            if not isinstance(action, dict):
                continue
            kind = str(action.get("kind") or "").strip().lower()
            action_id = str(action.get("action_id") or "").strip().lower()
            surface = str(action.get("surface") or "").strip().lower()
            if kind in combat_kinds or action_id in combat_action_ids:
                return True
            if action_id.startswith(("play_card", "use_potion", "discard_potion", "combat_select")):
                return True
            if surface == "combat":
                return True
            semantic = action.get("semantic")
            if isinstance(semantic, dict):
                semantic_domain = str(semantic.get("domain") or "").strip().lower()
                semantic_family = str(semantic.get("family") or "").strip().lower()
                if semantic_domain == "combat" or semantic_family in {"end_turn", "play_card", "use_potion"}:
                    return True
        return False

    def _resolve_acting_decision_domain(
        self,
        obs: dict[str, Any] | None,
        legal_actions: list[Any] | None,
        info: dict[str, Any] | None,
    ) -> tuple[str, str, bool]:
        encoded_domain = self._decision_domain_name(obs)
        if self._looks_like_combat_decision(obs, legal_actions, info):
            return "combat", encoded_domain, encoded_domain != "combat"
        if encoded_domain not in DECISION_DOMAINS:
            return "build", encoded_domain, encoded_domain != "build"
        return encoded_domain, encoded_domain, False

    @staticmethod
    def _semantic_family(action: Any) -> str:
        if not isinstance(action, dict):
            return ""
        semantic = action.get("semantic") if isinstance(action.get("semantic"), dict) else {}
        family = str(semantic.get("family") or "").strip().lower()
        if family:
            return family
        kind = str(action.get("kind") or action.get("action_type") or "").strip().lower()
        action_id = str(action.get("action_id") or "").strip().lower()
        if action_id == "end_turn" or kind == "end_turn":
            return "end_turn"
        if kind in {"play_card", "use_potion", "discard_potion"}:
            return kind
        if action_id.startswith("play_card"):
            return "play_card"
        if action_id.startswith("use_potion"):
            return "use_potion"
        return kind

    @staticmethod
    def _action_roles(action: Any) -> set[str]:
        if not isinstance(action, dict):
            return set()
        semantic = action.get("semantic") if isinstance(action.get("semantic"), dict) else {}
        roles = semantic.get("roles")
        if not isinstance(roles, list):
            return set()
        return {str(role).strip().lower() for role in roles if str(role).strip()}

    @staticmethod
    def _action_metric(action: Any, key: str) -> float:
        if not isinstance(action, dict):
            return 0.0
        aliases = {
            # Keep amount fields before boolean typed flags.  Compact semantic
            # actions may expose ``typed_gain_energy`` as an amount, while the
            # full aggregate profile uses it as a flag plus
            # ``typed_gain_energy_amount``.
            "energy": ("energy", "energy_gain", "typed_gain_energy_amount", "typed_gain_energy"),
            "energy_gain": ("energy_gain", "energy", "typed_gain_energy_amount", "typed_gain_energy"),
            "draw": ("draw", "cards_drawn", "typed_draw_amount", "typed_draw_cards"),
            "hp_loss": ("hp_loss", "hp_cost", "typed_hp_loss"),
            "hp_cost": ("hp_cost", "hp_loss", "typed_hp_loss"),
            "damage": ("damage", "total_damage"),
            "block": ("block", "total_block"),
        }
        keys = aliases.get(key, (key,))
        semantic = action.get("semantic") if isinstance(action.get("semantic"), dict) else {}
        semantic_profile = semantic.get("card_effect_profile") if isinstance(semantic.get("card_effect_profile"), dict) else {}
        source: dict[str, Any] = {}
        for source_key in ("card", "potion", "relic", "reward"):
            candidate = action.get(source_key)
            if isinstance(candidate, dict):
                source = candidate
                break
        source_profile = source.get("card_effect_profile") if isinstance(source.get("card_effect_profile"), dict) else {}
        action_profile = action.get("card_effect_profile") if isinstance(action.get("card_effect_profile"), dict) else {}
        typed_effects: dict[str, float] = {}
        try:
            if isinstance(action.get("card"), dict):
                typed_effects = aggregate_card_effect_profile_semantics(action.get("card"))
            elif isinstance(source, dict) and ("card_effect_profile" in source or "operations" in source):
                typed_effects = aggregate_card_effect_profile_semantics(source)
            elif "card_effect_profile" in action or "operations" in action:
                typed_effects = aggregate_card_effect_profile_semantics(action)
        except Exception:
            typed_effects = {}
        containers = (
            typed_effects,
            semantic_profile,
            semantic,
            action_profile,
            action,
            source_profile,
            source,
        )
        best = 0.0
        for source_obj in containers:
            if not isinstance(source_obj, dict):
                continue
            for candidate_key in keys:
                if candidate_key not in source_obj:
                    continue
                value = source_obj.get(candidate_key)
                if isinstance(value, (list, dict)):
                    continue
                try:
                    best = max(best, float(value or 0.0))
                except (TypeError, ValueError):
                    pass
        return float(best)

    @classmethod
    def _action_immediate_impact(cls, action: Any) -> float:
        roles = cls._action_roles(action)
        damage = cls._action_metric(action, "damage")
        block = cls._action_metric(action, "block")
        hits = max(cls._action_metric(action, "hits"), 1.0 if damage > 0.0 else 0.0)
        draw = cls._action_metric(action, "draw")
        energy_gain = max(cls._action_metric(action, "energy"), cls._action_metric(action, "energy_gain"))
        hp_loss = max(cls._action_metric(action, "hp_loss"), cls._action_metric(action, "hp_cost"))
        return float(
            damage
            + 0.75 * block
            + 1.5 * max(hits - 1.0, 0.0)
            + 2.0 * min(max(draw, 0.0), 3.0)
            + 2.0 * min(max(energy_gain, 0.0), 3.0)
            - 0.5 * min(max(hp_loss, 0.0), 6.0)
            + (8.0 if roles.intersection({"debuff", "weak", "vulnerable", "poison", "exhaust", "discard"}) else 0.0)
            + (6.0 if roles.intersection({"scaling", "power", "draw", "energy", "resource", "retain"}) else 0.0)
        )

    @staticmethod
    def _boss_context_max(context: dict[str, Any], key: str) -> float:
        if not isinstance(context, dict):
            return 0.0
        vals: list[float] = []
        player_state = context.get("player_state") if isinstance(context.get("player_state"), dict) else {}
        if key in player_state:
            try:
                vals.append(float(player_state.get(key) or 0.0))
            except (TypeError, ValueError):
                pass
        enemy_states = context.get("enemy_states_by_index")
        if isinstance(enemy_states, list):
            for state in enemy_states:
                if isinstance(state, dict) and key in state:
                    try:
                        vals.append(float(state.get(key) or 0.0))
                    except (TypeError, ValueError):
                        pass
        return max(vals) if vals else 0.0

    @staticmethod
    def _obs_energy(obs: dict[str, Any] | None) -> float:
        if not isinstance(obs, dict):
            return 0.0
        combat = obs.get("combat")
        if isinstance(combat, dict):
            try:
                return max(float(combat.get("energy") or 0.0), 0.0)
            except (TypeError, ValueError):
                pass
        scalars = obs.get("scalars")
        try:
            scalars_np = scalars.detach().cpu().numpy() if isinstance(scalars, torch.Tensor) else np.asarray(scalars)
            if scalars_np.ndim >= 1 and scalars_np.shape[0] > 29:
                return max(float(scalars_np.reshape(-1)[29]) * 10.0, 0.0)
        except Exception:
            return 0.0
        return 0.0

    def _current_raw_combat_obs(self) -> dict[str, Any] | None:
        raw_obs = getattr(getattr(self.env, "unwrapped", self.env), "_last_obs_raw", None)
        return raw_obs if isinstance(raw_obs, dict) else None

    def _combat_energy(self, encoded_obs: dict[str, Any] | None = None, raw_obs: dict[str, Any] | None = None) -> float:
        # Prefer live/raw combat energy.  The encoded scalar slot has changed across
        # observation versions and previously made boss_combat/energy_mean stay at 0.
        energy = self._obs_energy(raw_obs)
        if energy > 1e-6:
            return energy
        return self._obs_energy(encoded_obs)

    @staticmethod
    def _action_source(action: Any) -> dict[str, Any]:
        if not isinstance(action, dict):
            return {}
        for key in ("card", "potion", "relic", "reward"):
            source = action.get(key)
            if isinstance(source, dict):
                return source
        return action


    def _is_zero_cost_action(self, action: Any) -> bool:
        if not isinstance(action, dict):
            return False
        source = self._action_source(action)
        cost = action.get("card_cost", source.get("cost"))
        if isinstance(cost, str) and cost.strip().upper() == "X":
            return False
        try:
            return float(cost) <= 0.0
        except (TypeError, ValueError):
            return False

    def _is_positive_combat_action(self, action: Any) -> bool:
        if not isinstance(action, dict):
            return False
        family = self._semantic_family(action)
        if family in {"end_turn", "discard_potion"}:
            return False
        # card.type decisive:
        #   Attack / Skill / Power — always positive progress when legal
        #     (bridge-provided roles/damage can be empty for e.g. Bleed+ or
        #     unnamed skills, which previously caused detector to miss real
        #     wasteful end_turn; see diagnostics/end_turn_selections.jsonl).
        #   Status / Curse — forced unplayable draws, never positive even
        #     if they slip through as legal.
        source = self._action_source(action)
        card_type = str(
            (action.get("card_type") if isinstance(action.get("card_type"), str) else None)
            or source.get("type")
            or ""
        ).strip().lower()
        if card_type in {"status", "curse"}:
            return False
        if card_type in {"attack", "skill", "power"}:
            return True
        roles = self._action_roles(action)
        if roles.intersection({"attack", "block", "draw", "debuff", "buff", "heal", "setup", "scaling", "resource"}):
            return True
        if self._is_facing_change_action(action):
            return True
        for key in ("damage", "total_damage", "block", "total_block", "draw", "weak", "vulnerable", "heal", "strength", "dexterity", "energy"):
            if self._action_metric(action, key) > 0.0:
                return True
            try:
                if float(source.get(key) or 0.0) > 0.0:
                    return True
            except (TypeError, ValueError):
                pass
        text = " ".join(
            str(x or "")
            for x in (
                action.get("action_id"), action.get("kind"), action.get("title"), action.get("label"),
                source.get("id"), source.get("name"), source.get("title"), source.get("type"),
            )
        ).lower()
        return any(token in text for token in ("attack", "strike", "defend", "block", "skill", "power"))

    def _action_text(self, action: Any) -> str:
        if not isinstance(action, dict):
            return ""
        source = self._action_source(action)
        parts: list[str] = []
        for container in (action, source):
            if isinstance(container, dict):
                for key in ("action_id", "kind", "title", "label", "id", "name", "description", "text", "canonical_text", "type"):
                    value = str(container.get(key) or "").strip()
                    if value:
                        parts.append(value)
                keywords = container.get("keywords")
                if isinstance(keywords, list):
                    parts.extend(str(x or "").strip() for x in keywords if str(x or "").strip())
        return " | ".join(parts).lower()

    def _is_exhausting_action(self, action: Any) -> bool:
        if not isinstance(action, dict):
            return False
        roles = self._action_roles(action)
        if "exhaust" in roles:
            return True
        if (
            self._action_metric(action, "typed_exhaust_cards") > 0.0
            or self._action_metric(action, "typed_once_or_exhaust_self") > 0.0
        ):
            return True
        source = self._action_source(action)
        for key in ("exhaust", "exhaust_self", "will_exhaust"):
            if bool(action.get(key) or source.get(key)):
                return True
        return "exhaust" in self._action_text(action)

    def _is_ethereal_action(self, action: Any) -> bool:
        if not isinstance(action, dict):
            return False
        roles = self._action_roles(action)
        if "ethereal" in roles:
            return True
        source = self._action_source(action)
        for key in ("ethereal", "is_ethereal"):
            if bool(action.get(key) or source.get(key)):
                return True
        return "ethereal" in self._action_text(action) or "??" in self._action_text(action)

    def _is_retain_action(self, action: Any) -> bool:
        if not isinstance(action, dict):
            return False
        roles = self._action_roles(action)
        if "retain" in roles:
            return True
        if self._action_metric(action, "typed_retain_cards") > 0.0:
            return True
        source = self._action_source(action)
        for key in ("retain", "self_retain", "is_retained"):
            if bool(action.get(key) or source.get(key)):
                return True
        text = self._action_text(action)
        return "retain" in text or "??" in text

    def _incoming_damage_pressure(self, raw_obs: dict[str, Any] | None) -> tuple[float, float, float]:
        if not isinstance(raw_obs, dict):
            return 0.0, 0.0, 0.0
        combat = raw_obs.get("combat") if isinstance(raw_obs.get("combat"), dict) else {}
        player = raw_obs.get("player") if isinstance(raw_obs.get("player"), dict) else {}
        if not player and isinstance(combat.get("player"), dict):
            player = combat.get("player")
        block = self._safe_float(player.get("block")) if isinstance(player, dict) else 0.0
        hp = self._safe_float(player.get("hp", player.get("current_hp"))) if isinstance(player, dict) else 0.0
        incoming = 0.0
        enemies = combat.get("enemies") if isinstance(combat, dict) else []
        if isinstance(enemies, list):
            for enemy in enemies:
                if not isinstance(enemy, dict):
                    continue
                intent = enemy.get("intent") if isinstance(enemy.get("intent"), dict) else {}
                for key in ("total_damage", "damage", "intent_damage", "attack_damage"):
                    incoming += max(self._safe_float(intent.get(key)), self._safe_float(enemy.get(key)))
                    if incoming > 0.0:
                        break
        return incoming, block, hp

    def _action_numeric_value(self, action: Any, keys: tuple[str, ...] | list[str] | set[str]) -> float:
        """Read an unnormalised numeric action/card/potion metric.

        Bridge payloads are not perfectly uniform: card previews may live under
        ``action.semantic``, direct action fields, or the nested card/potion
        source.  Potion timing must not depend on one bridge revision's field
        placement, so this helper scans every action-local container and returns
        the largest positive value.
        """
        if not isinstance(action, dict):
            return 0.0
        semantic = action.get("semantic") if isinstance(action.get("semantic"), dict) else {}
        source = self._action_source(action)
        containers: list[Any] = [semantic, action, source]
        for key in ("card", "potion", "item", "target"):
            value = action.get(key)
            if isinstance(value, dict):
                containers.append(value)
                nested_card = value.get("card")
                nested_potion = value.get("potion")
                if isinstance(nested_card, dict):
                    containers.append(nested_card)
                if isinstance(nested_potion, dict):
                    containers.append(nested_potion)

        best = 0.0
        for container in containers:
            if not isinstance(container, dict):
                continue
            for key in keys:
                value = container.get(key)
                if isinstance(value, (list, dict)):
                    continue
                try:
                    best = max(best, float(value or 0.0))
                except (TypeError, ValueError):
                    continue
        return float(best)

    def _action_cost_value(self, action: Any) -> float:
        if not isinstance(action, dict):
            return 0.0
        source = self._action_source(action)
        for container in (action, source):
            if not isinstance(container, dict):
                continue
            for key in ("card_cost", "cost", "energy_cost", "base_cost"):
                if key not in container:
                    continue
                value = container.get(key)
                if isinstance(value, str) and value.strip().upper() == "X":
                    return 0.0
                try:
                    return max(float(value or 0.0), 0.0)
                except (TypeError, ValueError):
                    continue
        return 0.0

    def _combat_enemies_from_raw(self, raw_obs: dict[str, Any] | None) -> list[dict[str, Any]]:
        combat = raw_obs.get("combat") if isinstance(raw_obs, dict) and isinstance(raw_obs.get("combat"), dict) else {}
        for key in ("enemies", "monsters", "creatures"):
            enemies = combat.get(key)
            if isinstance(enemies, list):
                return [enemy for enemy in enemies if isinstance(enemy, dict)]
        return []

    def _alive_enemy_hp_values(self, raw_obs: dict[str, Any] | None) -> list[float]:
        values: list[float] = []
        for enemy in self._combat_enemies_from_raw(raw_obs):
            if bool(enemy.get("is_dead") or enemy.get("dead")):
                continue
            if enemy.get("alive") is False:
                continue
            hp = max(
                self._safe_float(enemy.get("hp")),
                self._safe_float(enemy.get("current_hp")),
                self._safe_float(enemy.get("health")),
            )
            if hp > 0.0:
                values.append(float(hp))
        return values

    def _target_enemy_hp(self, action: Any, raw_obs: dict[str, Any] | None) -> float:
        """Best-effort target HP for lethal/overkill potion timing.

        Falls back to the lowest alive enemy HP because untargeted damage potions
        are often aimed by bridge defaults at a killable target.
        """
        if not isinstance(action, dict):
            vals = self._alive_enemy_hp_values(raw_obs)
            return min(vals) if vals else 0.0
        target = action.get("target") if isinstance(action.get("target"), dict) else {}
        if isinstance(target, dict):
            hp = max(
                self._safe_float(target.get("hp")),
                self._safe_float(target.get("current_hp")),
                self._safe_float(target.get("health")),
            )
            if hp > 0.0:
                return float(hp)

        target_id = self._action_target_combat_id(action)
        target_name = ""
        for container in (action, target):
            if isinstance(container, dict):
                target_name = str(container.get("target_name") or container.get("name") or target_name or "").strip().lower()
        for enemy in self._combat_enemies_from_raw(raw_obs):
            enemy_id = enemy.get("combat_id", enemy.get("id"))
            try:
                if target_id is not None and enemy_id is not None and int(enemy_id) == int(target_id):
                    return max(
                        self._safe_float(enemy.get("hp")),
                        self._safe_float(enemy.get("current_hp")),
                        self._safe_float(enemy.get("health")),
                    )
            except (TypeError, ValueError):
                pass
            enemy_name = str(enemy.get("name") or enemy.get("title") or "").strip().lower()
            if target_name and enemy_name and target_name == enemy_name:
                return max(
                    self._safe_float(enemy.get("hp")),
                    self._safe_float(enemy.get("current_hp")),
                    self._safe_float(enemy.get("health")),
                )

        vals = self._alive_enemy_hp_values(raw_obs)
        return min(vals) if vals else 0.0

    def _combat_encounter_tier_from_raw(self, raw_obs: dict[str, Any] | None) -> str:
        encounter_id = ""
        if isinstance(raw_obs, dict):
            encounter_id = str(raw_obs.get("encounter_id") or raw_obs.get("encounter") or "").strip()
            combat = raw_obs.get("combat") if isinstance(raw_obs.get("combat"), dict) else {}
            if not encounter_id and isinstance(combat, dict):
                encounter_id = str(combat.get("encounter_id") or combat.get("encounter") or "").strip()
            if not encounter_id:
                try:
                    boss_ctx = build_boss_mechanics_context(raw_obs)
                    encounter_id = str(boss_ctx.get("encounter_key") or "").strip()
                except Exception:
                    encounter_id = ""
        try:
            return str(infer_encounter_tier(encounter_id)).strip().lower() if encounter_id else "normal"
        except Exception:
            return "normal"

    def _has_energy_followup(self, action_index: int, legal_actions: list[Any] | None, energy_after: float, mask_np: np.ndarray) -> bool:
        legal_count = min(len(legal_actions or []), MAX_ACTIONS, mask_np.shape[0])
        for other_idx in range(legal_count):
            if other_idx == action_index or mask_np[other_idx] <= 0:
                continue
            other = (legal_actions or [])[other_idx]
            if self._semantic_family(other) != "play_card":
                continue
            if self._is_x_cost_action(None, other_idx, other):
                continue
            source = self._action_source(other)
            cost_raw = other.get("card_cost", source.get("cost")) if isinstance(other, dict) else 0
            try:
                cost = max(float(cost_raw or 0.0), 0.0)
            except (TypeError, ValueError):
                cost = 0.0
            if cost <= energy_after + 1e-6 and self._action_immediate_impact(other) >= 4.0:
                return True
        return False

    def _has_resource_followup(
        self,
        action_index: int,
        legal_actions: list[Any] | None,
        energy_after: float,
        mask_np: np.ndarray,
        *,
        allow_cost_reduction: bool = False,
    ) -> bool:
        """Whether a resource/draw/energy potion can be converted this turn.

        We intentionally look for follow-up *cards*, not another potion.  The
        problematic behaviour is spending an energy/draw potion when the hand has
        no meaningful card action left, which creates "I used a potion" reward but
        no combat tempo.
        """
        legal_count = min(len(legal_actions or []), MAX_ACTIONS, mask_np.shape[0])
        for other_idx in range(legal_count):
            if other_idx == action_index or mask_np[other_idx] <= 0:
                continue
            other = (legal_actions or [])[other_idx]
            if self._semantic_family(other) != "play_card":
                continue
            if self._is_x_cost_action(None, other_idx, other):
                if energy_after <= 0.05:
                    continue
                return True
            cost = self._action_cost_value(other)
            if cost > energy_after + 1e-6 and not allow_cost_reduction:
                continue
            roles = self._action_roles(other)
            if (
                self._action_immediate_impact(other) >= 3.0
                or roles.intersection({"attack", "block", "draw", "debuff", "weak", "vulnerable", "scaling", "power", "resource", "energy"})
            ):
                return True
        return False

    def _extract_potion_effect_profile(self, action: Any) -> dict[str, Any]:
        """Phase 4 of potion-timing-modeling-plan.md §8.

        Returns the merged {effect_profile, effect_family, timing_tags,
        target_scope, training_tags} resolved from the action's potion payload
        (bridge live values) with the Python registry as fallback.  Either side
        may be absent; the result always has every effect_profile slot.
        """
        if not isinstance(action, dict):
            return {
                "effect_profile": dict(_POTION_EFFECT_DEFAULT),
                "effect_family": [],
                "semantic_tags": [],
                "timing_tags": [],
                "training_tags": [],
                "target_scope": "",
            }
        potion = action.get("potion") if isinstance(action.get("potion"), dict) else None
        pid = ""
        if potion:
            pid = str(potion.get("id") or "").strip()
        registry = _get_potion_profile(pid) if pid else {}
        effect = dict(_POTION_EFFECT_DEFAULT)
        effect.update(registry.get("effect_profile") or {})
        if potion and isinstance(potion.get("effect_profile"), dict):
            effect.update({k: v for k, v in potion["effect_profile"].items() if v is not None})
        def _pick(field: str) -> Any:
            if potion is not None and potion.get(field):
                return potion.get(field)
            return registry.get(field) or []
        return {
            "effect_profile": effect,
            "effect_family": list(_pick("effect_family") or []),
            "semantic_tags": list(_pick("semantic_tags") or []),
            "timing_tags": list(_pick("timing_tags") or []),
            "training_tags": list(_pick("training_tags") or []),
            "target_scope": str((potion.get("target_scope") if potion else None) or registry.get("target_scope") or ""),
            "potion_id": pid,
            "rarity": str((potion.get("rarity") if potion else None) or registry.get("rarity") or ""),
        }

    def _potion_timing_profile(
        self,
        action: Any,
        index: int,
        encoded_obs: dict[str, Any] | None,
        raw_obs: dict[str, Any] | None,
        legal_actions: list[Any] | None,
        mask_np: np.ndarray,
        energy: float,
    ) -> dict[str, Any]:
        """Timing-aware potion affordance used by planner bias and aux metrics.

        Potion should not be a flat "positive" action.  It is urgent when it kills,
        prevents lethal/major damage, answers a fight mechanism (Kaiser facing /
        back attack), or converts immediately into a strong follow-up.  Otherwise
        it is usually deferable/savable so end_turn is not marked wasteful simply
        because a potion button is legal.
        """
        family = self._semantic_family(action)
        is_potion = family in {"use_potion", "potion"}
        default = {
            "is_potion": False,
            "available": False,
            "positive": False,
            "urgent": False,
            "deferable": False,
            "low_urgency": False,
            "save_recommended": False,
            "no_followup": False,
            "lethal": False,
            "prevent_lethal": False,
            "prevent_major_loss": False,
            "mechanism_answer": False,
            "facing_change": False,
            "overkill": False,
            "block_waste": False,
            "followup_available": False,
            "use_quality": 0.0,
            "waste_risk": 0.0,
            "damage": 0.0,
            "block": 0.0,
        }
        if not is_potion or not isinstance(action, dict):
            return default

        raw_obs = raw_obs if isinstance(raw_obs, dict) else self._current_raw_combat_obs()
        roles = self._action_roles(action)
        merged_potion = self._extract_potion_effect_profile(action)
        eff = merged_potion["effect_profile"]
        timing_tags = merged_potion["timing_tags"]
        effect_family = merged_potion["effect_family"]
        target_scope = merged_potion["target_scope"]
        training_tags = merged_potion["training_tags"]

        damage = max(
            float(eff.get("damage") or 0.0),
            self._action_numeric_value(action, ("damage", "total_damage", "attack_damage", "preview_damage")),
            self._action_metric(action, "damage"),
            self._action_metric(action, "total_damage"),
        )
        block = max(
            float(eff.get("block") or 0.0),
            self._action_numeric_value(action, ("block", "total_block", "preview_block")),
            self._action_metric(action, "block"),
            self._action_metric(action, "total_block"),
        )
        heal = max(
            float(eff.get("heal") or 0.0),
            self._action_numeric_value(action, ("heal", "healing", "hp_gain")),
            self._action_metric(action, "heal"),
        )
        draw = max(
            float(eff.get("draw") or 0.0),
            self._action_numeric_value(action, ("draw", "cards_drawn", "card_draw")),
            self._action_metric(action, "draw"),
        )
        energy_gain = max(
            float(eff.get("energy_gain") or 0.0),
            self._action_numeric_value(action, ("energy", "energy_gain", "gain_energy")),
            self._action_metric(action, "energy"),
            self._action_metric(action, "energy_gain"),
        )
        hits = max(self._action_numeric_value(action, ("hits", "times", "repeat")), self._action_metric(action, "hits"), 1.0 if damage > 0.0 else 0.0)
        weak_v = float(eff.get("weak") or 0.0)
        vuln_v = float(eff.get("vulnerable") or 0.0)
        poison_v = float(eff.get("poison") or 0.0)
        debuff = bool(
            roles.intersection({"debuff", "weak", "vulnerable", "poison"})
            or weak_v > 0.0 or vuln_v > 0.0 or poison_v > 0.0
            or self._action_numeric_value(action, ("weak", "vulnerable", "poison")) > 0.0
        )
        gen_card_v = float(eff.get("generate_card_count") or 0.0)
        discover_v = float(eff.get("discover_count") or 0.0)
        retrieve_v = float(eff.get("retrieve_from_discard") or 0.0)
        upgrade_v = float(eff.get("upgrade_hand") or 0.0)
        dup_v = float(eff.get("duplicate_next") or 0.0)
        replace_v = float(eff.get("replace_or_transform_hand") or 0.0)
        resource_like = bool(
            energy_gain > 0.0 or draw > 0.0
            or gen_card_v > 0.0 or discover_v > 0.0 or retrieve_v > 0.0
            or roles.intersection({"resource", "energy", "draw"})
        )
        hand_transform_like = bool(upgrade_v > 0.0 or dup_v > 0.0 or replace_v > 0.0)
        long_term_like = bool(eff.get("long_term_value") or "long_term_value" in timing_tags)
        passive_or_triggered = bool(eff.get("passive_or_triggered"))

        incoming, current_block, hp = self._incoming_damage_pressure(raw_obs)
        threat_gap = max(0.0, incoming - current_block)
        target_hp = self._target_enemy_hp(action, raw_obs)
        aoe = bool(
            roles.intersection({"aoe", "all_enemies"})
            or bool(eff.get("aoe"))
            or str(action.get("target_scope") or target_scope).lower() in {"all_enemies", "aoe", "allenemies", "allcreatures"}
        )
        lethal = bool(damage > 0.0 and target_hp > 0.0 and damage >= target_hp)
        overkill = bool(
            damage > 0.0
            and target_hp > 0.0
            and not aoe
            and damage > target_hp + max(6.0, 0.50 * target_hp)
        )

        defensive = bool(block > 0.0 or heal > 0.0 or debuff)
        prevent_lethal = bool(hp > 0.0 and threat_gap >= max(hp, 1.0) and defensive)
        prevent_major_loss = bool(threat_gap >= max(8.0, 0.25 * max(hp, 1.0)) and defensive)
        block_waste = bool(block > 0.0 and threat_gap <= 0.05)

        energy_after = max(0.0, float(energy) + energy_gain)
        followup_available = self._has_resource_followup(index, legal_actions, energy_after, mask_np) if resource_like else False
        no_followup = bool(resource_like and not followup_available)

        facing_change = False
        mechanism_answer = False
        kaiser_risk = 0.0
        try:
            facing_change = bool(self._is_kaiser_facing_change_action(action, raw_obs))
            boss_ctx = build_boss_mechanics_context(raw_obs) if isinstance(raw_obs, dict) else {}
            kaiser_risk = max(
                self._boss_context_max(boss_ctx, "back_attack_risk"),
                self._boss_context_max(boss_ctx, "back_attack_active"),
                self._boss_context_max(boss_ctx, "incoming_damage_multiplier_norm"),
            )
        except Exception:
            kaiser_risk = 0.0
        if facing_change:
            mechanism_answer = True
        elif kaiser_risk > 0.05:
            # Damage-only potion is only a mechanism answer if it kills or is a
            # real pressure action; arbitrary potion use must not satisfy Kaiser
            # defense metrics.
            mechanism_answer = bool(
                lethal
                or (damage >= 12.0 and target_hp <= 0.0)
                or (target_hp > 0.0 and damage >= min(target_hp, max(12.0, 0.35 * target_hp)))
                or block > 0.0
                or debuff
            )

        encounter_tier = self._combat_encounter_tier_from_raw(raw_obs)
        hp_ratio = (hp / max(self._safe_float((raw_obs or {}).get("player", {}).get("max_hp")) if isinstance((raw_obs or {}).get("player"), dict) else hp, 1.0)) if hp > 0.0 else 1.0
        high_damage = bool(damage >= 18.0 or (target_hp > 0.0 and damage >= max(12.0, 0.35 * target_hp)))

        use_quality = 0.08
        if lethal:
            use_quality += 0.85
        elif damage > 0.0:
            use_quality += min(0.34, damage / 55.0)
            if high_damage:
                use_quality += 0.16
        if prevent_lethal:
            use_quality += 0.95
        elif prevent_major_loss:
            use_quality += 0.52
        elif block > 0.0 and threat_gap > 0.0:
            use_quality += 0.35 * min(block / max(threat_gap, 1.0), 1.0)
        if heal > 0.0:
            use_quality += 0.25 if hp_ratio <= 0.55 else 0.10
        if debuff and incoming > 0.0:
            use_quality += 0.30
        if mechanism_answer:
            use_quality += 0.62
        if resource_like:
            use_quality += 0.36 if followup_available else -0.42
        if encounter_tier in {"elite", "boss"} and (lethal or prevent_major_loss or mechanism_answer or high_damage):
            use_quality += 0.12

        waste_risk = 0.0
        if no_followup:
            waste_risk += 0.45
        if block_waste:
            waste_risk += 0.35
        if overkill and not mechanism_answer:
            waste_risk += 0.25
        low_threat = threat_gap <= 2.0 and not prevent_major_loss and not prevent_lethal
        save_recommended = bool(
            low_threat
            and hp_ratio >= 0.55
            and not lethal
            and not mechanism_answer
            and not (encounter_tier in {"elite", "boss"} and high_damage)
        )
        if save_recommended:
            waste_risk += 0.42 if encounter_tier in {"weak", "normal"} else 0.22

        # Phase 4 of potion-timing-modeling-plan.md §8.6 / §8.7: hand-transform
        # potions are good iff hand has high-value targets and a usable
        # follow-up window; otherwise they should defer.
        hand_size = 0
        try:
            player = (raw_obs or {}).get("player") if isinstance(raw_obs, dict) else None
            hand_size = len(player.get("hand") or []) if isinstance(player, dict) else 0
        except Exception:
            hand_size = 0
        hand_context_good = bool(hand_transform_like and hand_size >= 3)
        hand_context_bad = bool(hand_transform_like and hand_size <= 1)
        if hand_transform_like and hand_context_good:
            use_quality += 0.30
        if hand_transform_like and hand_context_bad:
            waste_risk += 0.30
            save_recommended = True
        if long_term_like and not (urgent_threat := (prevent_lethal or prevent_major_loss)):
            # Long-term-only potion in combat sandbox: don't reward as immediate
            # combat positive; it is mostly a save-or-use-out-of-combat candidate.
            use_quality = max(0.0, use_quality - 0.20)
            save_recommended = True
        if passive_or_triggered:
            # Passive/triggered potions don't have an immediate "use now" payoff.
            use_quality = max(0.0, use_quality - 0.10)

        use_quality = float(np.clip(use_quality - waste_risk, 0.0, 1.0))
        waste_risk = float(np.clip(waste_risk, 0.0, 1.0))
        # save_value: how much value remains if the agent saves the potion for
        # later. Approximated as inverse-of-use-quality + bonus for defensive/
        # mechanism tools whose later utility is high.
        save_value = float(np.clip(
            (1.0 - use_quality) * 0.6
            + (0.25 if any(t in timing_tags for t in ("prevent_lethal_tool", "lethal_tool", "mechanism_answer_candidate")) else 0.0)
            + (0.20 if long_term_like else 0.0)
            - (0.30 if (lethal or prevent_lethal or mechanism_answer) else 0.0),
            0.0, 1.0,
        ))
        urgent = bool(
            lethal
            or prevent_lethal
            or mechanism_answer
            or (prevent_major_loss and use_quality >= 0.45)
            or use_quality >= 0.62
        )
        low_urgency = bool((use_quality < 0.35) or (waste_risk > use_quality and not urgent))
        positive = bool(urgent or use_quality >= 0.32)
        deferable = bool(not urgent and (low_urgency or save_recommended or no_followup or block_waste or overkill))
        requires_followup = bool(resource_like or hand_transform_like)

        return {
            "is_potion": True,
            "available": True,
            "potion_id": merged_potion.get("potion_id", ""),
            "rarity": merged_potion.get("rarity", ""),
            "effect_family": list(effect_family),
            "timing_tags": list(timing_tags),
            "training_tags": list(training_tags),
            "damage": float(damage),
            "block": float(block),
            "energy_gain": float(energy_gain),
            "draw": float(draw),
            "heal": float(heal),
            "weak": float(weak_v),
            "vulnerable": float(vuln_v),
            "poison": float(poison_v),
            "debuff": bool(debuff),
            "incoming": float(incoming),
            "current_block": float(current_block),
            "hp": float(hp),
            "threat_gap": float(threat_gap),
            "target_hp": float(target_hp),
            "aoe": bool(aoe),
            "lethal": bool(lethal),
            "prevent_lethal": bool(prevent_lethal),
            "prevent_major_loss": bool(prevent_major_loss),
            "mechanism_answer": bool(mechanism_answer),
            "facing_change": bool(facing_change),
            "followup_available": bool(followup_available),
            "overkill": bool(overkill),
            "block_waste": bool(block_waste),
            "no_followup": bool(no_followup),
            "save_recommended": bool(save_recommended),
            "low_urgency": bool(low_urgency),
            "urgency": float(use_quality),
            "use_quality": float(use_quality),
            "waste_risk": float(waste_risk),
            "save_value": float(save_value),
            "positive": bool(positive),
            "urgent": bool(urgent),
            "deferable": bool(deferable),
            "requires_followup": bool(requires_followup),
            "hand_context_good": bool(hand_context_good),
            "hand_context_bad": bool(hand_context_bad),
            "long_term_value": bool(long_term_like),
            "passive_or_triggered": bool(passive_or_triggered),
            "hand_transform": bool(hand_transform_like),
            "resource_like": bool(resource_like),
        }


    def _classify_positive_combat_action(
        self,
        action: Any,
        index: int,
        encoded_obs: dict[str, Any] | None,
        raw_obs: dict[str, Any] | None,
        legal_actions: list[Any] | None,
        mask_np: np.ndarray,
        energy: float,
    ) -> dict[str, Any]:
        family = self._semantic_family(action)
        if family in {"use_potion", "potion"}:
            profile = self._potion_timing_profile(
                action,
                index,
                encoded_obs,
                raw_obs,
                legal_actions,
                mask_np,
                energy,
            )
            return {
                "positive": bool(profile.get("positive", False)),
                "urgent": bool(profile.get("urgent", False)),
                "deferable": bool(profile.get("deferable", False)),
                "exhausting": False,
                "deferable_exhaust": False,
                "ethereal_urgent": False,
                "energy_without_followup": bool(profile.get("no_followup", False)),
                "x_cost_zero": False,
                "potion_available": bool(profile.get("available", False)),
                "potion_urgent": bool(profile.get("urgent", False)),
                "potion_low_urgency": bool(profile.get("low_urgency", False)),
                "potion_save_recommended": bool(profile.get("save_recommended", False)),
                "potion_no_followup": bool(profile.get("no_followup", False)),
                "potion_lethal": bool(profile.get("lethal", False)),
                "potion_prevent_lethal": bool(profile.get("prevent_lethal", False)),
                "potion_mechanism_answer": bool(profile.get("mechanism_answer", False)),
                "potion_overkill": bool(profile.get("overkill", False)),
                "potion_block_waste": bool(profile.get("block_waste", False)),
                "potion_use_quality": float(profile.get("use_quality", 0.0) or 0.0),
                "potion_waste_risk": float(profile.get("waste_risk", 0.0) or 0.0),
                "potion_save_value": float(profile.get("save_value", 0.0) or 0.0),
                "potion_hand_context_good": bool(profile.get("hand_context_good", False)),
                "potion_hand_context_bad": bool(profile.get("hand_context_bad", False)),
                "potion_long_term_value": bool(profile.get("long_term_value", False)),
                "potion_requires_followup": bool(profile.get("requires_followup", False)),
                "potion_effect_family": list(profile.get("effect_family", []) or []),
                "potion_id": str(profile.get("potion_id", "") or ""),
                "potion_facing_change": bool(profile.get("facing_change", False)),
            }
        positive = self._is_positive_combat_action(action)
        roles = self._action_roles(action)
        exhausting = family == "play_card" and self._is_exhausting_action(action)
        ethereal = family == "play_card" and self._is_ethereal_action(action)
        retain = family == "play_card" and self._is_retain_action(action)
        x_cost_zero = family == "play_card" and self._is_x_cost_action(encoded_obs, index, action) and energy <= 0.05
        damage = max(self._action_metric(action, "damage"), self._action_metric(action, "total_damage"))
        block = max(self._action_metric(action, "block"), self._action_metric(action, "total_block"))
        energy_gain = max(self._action_metric(action, "energy"), self._action_metric(action, "energy_gain"))
        hp_loss = max(self._action_metric(action, "hp_loss"), self._action_metric(action, "hp_cost"))
        typed_requires_followup = self._action_metric(action, "typed_requires_followup") > 0.0
        typed_strategic_skip_if_no_followup = self._action_metric(action, "typed_strategic_skip_if_no_followup") > 0.0
        typed_modify_cost = self._action_metric(action, "typed_modify_cost") > 0.0
        typed_no_draw = self._action_metric(action, "typed_no_draw") > 0.0
        typed_future_penalty = self._action_metric(action, "typed_future_penalty") > 0.0
        typed_consumes_future_resource = self._action_metric(action, "typed_consumes_future_resource") > 0.0
        typed_card_state_mutation = self._action_metric(action, "typed_card_state_mutation") > 0.0
        typed_modifies_hand = self._action_metric(action, "typed_modifies_hand") > 0.0
        setup_followup_dependent = bool(
            family == "play_card"
            and (
                typed_requires_followup
                or typed_strategic_skip_if_no_followup
                or typed_modify_cost
                or typed_no_draw
                or typed_future_penalty
                or typed_consumes_future_resource
            )
        )
        future_or_no_draw = bool(typed_no_draw or typed_future_penalty or typed_consumes_future_resource)
        card_state_setup = bool(typed_card_state_mutation or typed_modifies_hand)
        incoming, current_block, _hp = self._incoming_damage_pressure(raw_obs)
        threat_gap = max(0.0, incoming - current_block)
        impact = self._action_immediate_impact(action)
        energy_after = max(0.0, energy - self._action_cost_value(action) + energy_gain)
        energy_without_followup = bool(family == "play_card" and energy_gain > 0.0 and not self._has_energy_followup(index, legal_actions, energy_after, mask_np))
        setup_followup_available = bool(
            setup_followup_dependent
            and self._has_resource_followup(
                index,
                legal_actions,
                energy_after,
                mask_np,
                allow_cost_reduction=typed_modify_cost,
            )
        )
        followup_missing = bool(setup_followup_dependent and not setup_followup_available)

        mechanism_urgent = False
        try:
            mechanism_urgent = bool(
                self._is_kaiser_facing_change_action(action, raw_obs)
                or self._is_kaiser_risk_handling_action(action, raw_obs)
            )
        except Exception:
            mechanism_urgent = False

        urgent = bool(
            positive
            and not x_cost_zero
            and (
                mechanism_urgent
                or ethereal
                or (block > 0.0 and threat_gap > 0.0)
                or (damage >= 12.0)
                or (roles.intersection({"weak", "vulnerable", "debuff"}) and incoming > 0.0)
                or (
                    setup_followup_dependent
                    and setup_followup_available
                    and not energy_without_followup
                    and (energy_gain > 0.0 or typed_modify_cost or self._action_metric(action, "draw") > 0.0)
                )
                or (
                    not exhausting
                    and not energy_without_followup
                    and not followup_missing
                    and not retain
                    and not (future_or_no_draw and impact < 8.0)
                    and impact >= 3.0
                )
            )
        )
        deferable = bool(
            positive
            and not urgent
            and family == "play_card"
            and (
                exhausting
                or retain
                or x_cost_zero
                or energy_without_followup
                or followup_missing
                or (future_or_no_draw and not setup_followup_available)
                or (card_state_setup and not setup_followup_available and impact < 6.0)
                or (hp_loss > 0.0 and energy_without_followup)
            )
        )
        return {
            "positive": bool(positive),
            "urgent": bool(urgent),
            "deferable": bool(deferable),
            "exhausting": bool(exhausting),
            "deferable_exhaust": bool(deferable and exhausting),
            "ethereal_urgent": bool(urgent and ethereal),
            "energy_without_followup": bool(energy_without_followup),
            "x_cost_zero": bool(x_cost_zero),
            "typed_requires_followup": bool(typed_requires_followup),
            "typed_strategic_skip_if_no_followup": bool(typed_strategic_skip_if_no_followup),
            "typed_modify_cost": bool(typed_modify_cost),
            "typed_no_draw": bool(typed_no_draw),
            "typed_future_penalty": bool(typed_future_penalty),
            "typed_consumes_future_resource": bool(typed_consumes_future_resource),
            "typed_card_state_mutation": bool(typed_card_state_mutation),
            "typed_modifies_hand": bool(typed_modifies_hand),
            "setup_followup_dependent": bool(setup_followup_dependent),
            "setup_followup_available": bool(setup_followup_available),
            "followup_missing": bool(followup_missing),
        }

    def _is_facing_change_action(self, action: Any) -> bool:
        """Fallback for explicit facing actions.

        In live STS2 surrounded combat, facing usually changes implicitly by using
        a targeted card/potion on an enemy on that side.  Do not use this text
        heuristic as the primary detector; use
        _action_changes_facing_toward_target() when raw combat state is available.
        """
        if not isinstance(action, dict):
            return False
        family = self._semantic_family(action)
        if family == "end_turn":
            return False
        source = self._action_source(action)
        text = " ".join(
            str(x or "")
            for x in (
                action.get("action_id"), action.get("kind"), action.get("selection"), action.get("title"), action.get("label"),
                source.get("id"), source.get("name"), source.get("title"), source.get("description"),
            )
        ).lower()
        patterns = (
            "change_facing", "change-facing", "change facing", "set_facing", "set-facing", "set facing",
            "turn_around", "turn-around", "turn around", "turnaround", "rotate", "facing",
            "face_left", "face-right", "face right", "face left", "surrounded",
        )
        return any(pattern in text for pattern in patterns)

    @staticmethod
    def _normalize_side(value: Any) -> str:
        text = str(value or "").strip().lower()
        if not text:
            return ""
        if text in {"l", "left", "side.left", "creatureside.left"} or text.endswith(".left"):
            return "left"
        if text in {"r", "right", "side.right", "creatureside.right"} or text.endswith(".right"):
            return "right"
        if "left" in text:
            return "left"
        if "right" in text:
            return "right"
        return ""

    def _combat_player_facing(self, raw_obs: Any | None = None) -> str:
        raw_obs = raw_obs if isinstance(raw_obs, dict) else self._current_raw_combat_obs()
        combat = raw_obs.get("combat") if isinstance(raw_obs, dict) and isinstance(raw_obs.get("combat"), dict) else {}
        return self._normalize_side(combat.get("facing"))

    def _action_target_combat_id(self, action: Any) -> int | None:
        if not isinstance(action, dict):
            return None
        candidates: list[Any] = [
            action.get("target_combat_id"),
            action.get("target_id"),
            action.get("enemy_combat_id"),
            action.get("enemy_id"),
        ]
        target = action.get("target")
        if isinstance(target, dict):
            candidates.extend([target.get("combat_id"), target.get("id"), target.get("target_combat_id")])
        target_mapping = action.get("target_mapping")
        if isinstance(target_mapping, dict):
            candidates.extend([target_mapping.get("combat_id"), target_mapping.get("id")])
        for value in candidates:
            try:
                if value is not None and str(value).strip() != "":
                    return int(value)
            except (TypeError, ValueError):
                continue
        return None

    @staticmethod
    def _enemy_back_attack_position(enemy: Any) -> str:
        """Infer left/right Kaiser/Rocket/Crusher body position from BackAttack powers.

        Bridge Creature.Side is faction (Player/Enemy), not left/right.  For the
        surrounded back-attack family, the reliable left/right cue is the enemy
        power id: BACK_ATTACK_LEFT_POWER / BACK_ATTACK_RIGHT_POWER.
        """
        if not isinstance(enemy, dict):
            return ""
        powers = enemy.get("powers")
        if not isinstance(powers, list):
            return ""
        for power in powers:
            if not isinstance(power, dict):
                continue
            text = " ".join(str(power.get(k) or "") for k in ("id", "title", "name")).upper()
            if "BACK_ATTACK_LEFT" in text:
                return "left"
            if "BACK_ATTACK_RIGHT" in text:
                return "right"
        return ""

    def _action_target_back_attack_position(self, action: Any, raw_obs: Any | None = None) -> str:
        if not isinstance(action, dict):
            return ""
        target = action.get("target")
        pos = self._enemy_back_attack_position(target) if isinstance(target, dict) else ""
        if pos:
            return pos
        target_id = self._action_target_combat_id(action)
        if target_id is None:
            return ""
        raw_obs = raw_obs if isinstance(raw_obs, dict) else self._current_raw_combat_obs()
        combat = raw_obs.get("combat") if isinstance(raw_obs, dict) and isinstance(raw_obs.get("combat"), dict) else {}
        for key in ("enemies", "monsters", "creatures"):
            entries = combat.get(key)
            if not isinstance(entries, list):
                continue
            for enemy in entries:
                if not isinstance(enemy, dict):
                    continue
                try:
                    enemy_id = int(enemy.get("combat_id", enemy.get("id")))
                except (TypeError, ValueError):
                    continue
                if enemy_id == target_id:
                    return self._enemy_back_attack_position(enemy)
        return ""

    def _action_target_side(self, action: Any, raw_obs: Any | None = None) -> str:
        if not isinstance(action, dict):
            return ""
        back_attack_pos = self._action_target_back_attack_position(action, raw_obs)
        if back_attack_pos:
            return back_attack_pos
        candidates: list[Any] = [action.get("target_side"), action.get("side")]
        target = action.get("target")
        if isinstance(target, dict):
            candidates.extend([target.get("side"), target.get("target_side")])
        target_mapping = action.get("target_mapping")
        if isinstance(target_mapping, dict):
            candidates.extend([target_mapping.get("side"), target_mapping.get("target_side")])
        for value in candidates:
            side = self._normalize_side(value)
            if side:
                return side

        target_id = self._action_target_combat_id(action)
        if target_id is None:
            return ""
        raw_obs = raw_obs if isinstance(raw_obs, dict) else self._current_raw_combat_obs()
        combat = raw_obs.get("combat") if isinstance(raw_obs, dict) and isinstance(raw_obs.get("combat"), dict) else {}
        for key in ("enemies", "monsters", "creatures"):
            entries = combat.get(key)
            if not isinstance(entries, list):
                continue
            for enemy in entries:
                if not isinstance(enemy, dict):
                    continue
                try:
                    enemy_id = int(enemy.get("combat_id", enemy.get("id")))
                except (TypeError, ValueError):
                    continue
                if enemy_id == target_id:
                    return self._normalize_side(enemy.get("side") or enemy.get("target_side"))
        return ""

    def _is_targeted_enemy_action(self, action: Any) -> bool:
        if not isinstance(action, dict):
            return False
        if self._semantic_family(action) not in {"play_card", "use_potion", "potion"}:
            return False
        target = action.get("target")
        if isinstance(target, dict) and any(target.get(k) is not None for k in ("combat_id", "id", "name", "side")):
            return True
        return self._action_target_combat_id(action) is not None or bool(action.get("target_name") or action.get("target_side"))

    def _action_changes_facing_toward_target(self, action: Any, raw_obs: Any | None = None) -> bool:
        """True when a targeted card/potion points at an enemy on the other side.

        STS2 Surrounded/BackAttack facing is not normally represented as a
        separate legal action.  The game turns the player by resolving targeted
        operations toward a left/right enemy; therefore a Strike/Bash/potion aimed
        at the opposite side is the real facing-change action.  For Kaiser-like
        enemies, left/right is inferred from BACK_ATTACK_LEFT/RIGHT_POWER, not
        Creature.Side (that field is faction: Player/Enemy).
        """
        if not self._is_targeted_enemy_action(action):
            return False
        current = self._combat_player_facing(raw_obs)
        target_side = self._action_target_side(action, raw_obs)
        return bool(current and target_side and current != target_side)

    def _is_kaiser_facing_change_action(self, action: Any, raw_obs: Any | None = None) -> bool:
        return self._action_changes_facing_toward_target(action, raw_obs) or self._is_facing_change_action(action)

    def _dump_kaiser_facing_diagnostic(
        self,
        raw_obs: Any,
        legal_actions: list[Any] | None,
        action_mask: np.ndarray,
        encounter: str,
        risk: float,
    ) -> None:
        if self._kaiser_facing_debug_dump_count >= 24:
            return
        try:
            mask_np = np.asarray(action_mask, dtype=np.float32).reshape(-1)
            combat = raw_obs.get("combat") if isinstance(raw_obs, dict) and isinstance(raw_obs.get("combat"), dict) else {}
            rows = []
            for idx, action in enumerate(legal_actions or []):
                if idx >= MAX_ACTIONS or idx >= mask_np.shape[0] or mask_np[idx] <= 0:
                    continue
                if not isinstance(action, dict):
                    continue
                target = action.get("target") if isinstance(action.get("target"), dict) else {}
                rows.append({
                    "idx": idx,
                    "family": self._semantic_family(action),
                    "kind": action.get("kind"),
                    "action_id": action.get("action_id"),
                    "label": action.get("label"),
                    "target_combat_id": self._action_target_combat_id(action),
                    "target_name": action.get("target_name") or target.get("name"),
                    "target_side_or_back_attack_position": self._action_target_side(action, raw_obs),
                    "target_back_attack_position": self._action_target_back_attack_position(action, raw_obs),
                    "changes_facing": self._action_changes_facing_toward_target(action, raw_obs),
                    "explicit_facing": self._is_facing_change_action(action),
                    "roles": sorted(self._action_roles(action)),
                    "card": (action.get("card") if isinstance(action.get("card"), dict) else {}).get("title")
                        or (action.get("card") if isinstance(action.get("card"), dict) else {}).get("name"),
                    "potion": (action.get("potion") if isinstance(action.get("potion"), dict) else {}).get("title")
                        or (action.get("potion") if isinstance(action.get("potion"), dict) else {}).get("name"),
                    "keys": sorted(str(k) for k in action.keys()),
                })
            path = Path(self.log_dir) / "diagnostics" / "kaiser_facing_legal_actions.jsonl"
            path.parent.mkdir(parents=True, exist_ok=True)
            payload = {
                "time": time.time(),
                "episode_count": int(getattr(self, "episode_count", 0)),
                "total_steps": int(getattr(self, "total_steps", 0)),
                "encounter": encounter,
                "risk": float(risk),
                "player_facing": self._combat_player_facing(raw_obs),
                "combat_facing_raw": combat.get("facing"),
                "enemies": combat.get("enemies"),
                "actions": rows,
            }
            with path.open("a", encoding="utf-8") as fh:
                fh.write(json.dumps(payload, ensure_ascii=False, default=str) + "\n")
            self._kaiser_facing_debug_dump_count += 1
        except Exception:
            return

    def _dump_end_turn_diagnostic(
        self,
        raw_obs: Any,
        encoded_obs: dict[str, Any] | None,
        legal_actions: list[Any] | None,
        action_mask: np.ndarray,
        context: dict[str, Any],
        chosen_idx: int,
        encounter: str,
        wasteful_end_turn: bool,
    ) -> None:
        """One-shot per end_turn selection: dump full decision state for post-mortem.

        Written to <log_dir>/diagnostics/end_turn_selections.jsonl when the policy
        chooses end_turn.  Captures mask / playability / positive-action judgement
        per legal action so we can prove whether the policy was genuinely forced
        into end_turn (no positive action playable) or whether the detector mis-
        categorised a hand card and missed a real wasteful.
        """
        if self._end_turn_debug_dump_count >= 80:
            return
        try:
            mask_np = np.asarray(action_mask, dtype=np.float32).reshape(-1)
            raw_combat = raw_obs.get("combat") if isinstance(raw_obs, dict) and isinstance(raw_obs.get("combat"), dict) else {}
            raw_player = raw_obs.get("player") if isinstance(raw_obs, dict) and isinstance(raw_obs.get("player"), dict) else {}
            energy_raw = self._obs_energy(raw_obs)
            energy_encoded = self._obs_energy(encoded_obs)
            hand_cards = raw_combat.get("hand") if isinstance(raw_combat.get("hand"), list) else []

            rows: list[dict[str, Any]] = []
            for idx, action in enumerate(legal_actions or []):
                if idx >= MAX_ACTIONS:
                    break
                if not isinstance(action, dict):
                    continue
                mask_value = float(mask_np[idx]) if idx < mask_np.shape[0] else 0.0
                family = self._semantic_family(action)
                roles = sorted(self._action_roles(action))
                is_positive = self._is_positive_combat_action(action) if mask_value > 0 else False
                card = action.get("card") if isinstance(action.get("card"), dict) else {}
                rows.append({
                    "idx": idx,
                    "mask": mask_value,
                    "is_chosen": (idx == int(chosen_idx)),
                    "family": family,
                    "action_id": action.get("action_id"),
                    "title": action.get("title") or card.get("title"),
                    "is_positive": is_positive,
                    "roles": roles,
                    "damage": self._action_metric(action, "damage"),
                    "block": self._action_metric(action, "block"),
                    "cost": card.get("cost"),
                    "card_id": card.get("id"),
                })

            path = Path(self.log_dir) / "diagnostics" / "end_turn_selections.jsonl"
            path.parent.mkdir(parents=True, exist_ok=True)
            payload = {
                "time": time.time(),
                "episode_count": int(getattr(self, "episode_count", 0)),
                "total_steps": int(getattr(self, "total_steps", 0)),
                "encounter": encounter,
                "wasteful_end_turn": bool(wasteful_end_turn),
                "wasteful_context": {
                    "wasteful": bool(context.get("wasteful", False)),
                    "energy_from_context": float(context.get("energy", 0.0) or 0.0),
                    "end_turn_indices": list(context.get("end_turn_indices", [])),
                    "positive_indices": list(context.get("positive_indices", [])),
                    "positive_progress_count": int(context.get("positive_progress_count", 0) or 0),
                    "setup_scaling_indices": list(context.get("setup_scaling_indices", [])),
                    "zero_cost_positive": bool(context.get("zero_cost_positive", False)),
                    "severity": float(context.get("severity", 0.0) or 0.0),
                },
                "energy": {
                    "raw": energy_raw,
                    "encoded": energy_encoded,
                    "combat.energy": raw_combat.get("energy"),
                    "combat.max_energy": raw_combat.get("max_energy"),
                },
                "player_hp": raw_player.get("hp"),
                "combat_round": raw_combat.get("round"),
                "raw_hand_cards": [
                    {
                        "title": c.get("title") if isinstance(c, dict) else str(c),
                        "cost": c.get("cost") if isinstance(c, dict) else None,
                        "is_playable": c.get("is_playable") if isinstance(c, dict) else None,
                    }
                    for c in hand_cards[:10]
                ],
                "chosen_idx": int(chosen_idx),
                "legal_actions": rows,
            }
            with path.open("a", encoding="utf-8") as fh:
                fh.write(json.dumps(payload, ensure_ascii=False, default=str) + "\n")
            self._end_turn_debug_dump_count += 1
        except Exception:
            return

    @staticmethod
    def _classify_end_turn_action(
        context: dict[str, Any],
        action_diagnostics: dict[str, Any] | None = None,
        boss_signals: dict[str, Any] | None = None,
    ) -> tuple[str, dict[str, bool]]:
        """Single classifier for selected end_turn actions.

        Used by both the bias-side stats and the selected-action JSONL dumper so
        ``bad_end_turn``/``forced_end_turn``/``strategic_defer_end_turn`` rates
        cannot drift between detector and tracker.  Priority is strict:
        ``bad > transient(forced) > forced > strategic_defer > unknown``.

        ``boss_signals`` carries already-computed mechanism flags (Kaiser back
        attack + facing candidate, Ceremonial stun window, etc.) so end_turn
        chosen *under* those windows is upgraded to ``bad_end_turn`` even when
        the static positive/urgent counters say nothing was urgently playable.
        """

        end_turn_indices = list(context.get("end_turn_indices") or [])
        wasteful = bool(context.get("wasteful", False))
        strategic_defer_available = bool(context.get("strategic_defer_available", False))
        positive_count = int(context.get("positive_progress_count", 0) or 0)
        urgent_count = int(context.get("urgent_positive_count", 0) or 0)
        deferable_count = int(context.get("deferable_positive_count", 0) or 0)
        energy = float(context.get("energy", 0.0) or 0.0)
        diag = action_diagnostics if isinstance(action_diagnostics, dict) else {}
        transient = bool(diag.get("transient_only_end_turn", False))
        boss = boss_signals if isinstance(boss_signals, dict) else {}
        kaiser_risk = float(boss.get("kaiser_back_attack_risk", 0.0) or 0.0)
        kaiser_facing_cands = float(boss.get("kaiser_facing_change_candidate_count", 0.0) or 0.0)
        kaiser_defense_cands = float(boss.get("kaiser_defense_candidate_count", 0.0) or 0.0)
        ceremonial_stun = float(boss.get("ceremonial_stun_window", 0.0) or 0.0)
        ceremonial_high_impact = float(boss.get("ceremonial_high_impact_count", 0.0) or 0.0)
        kaiser_pressure_window = (
            kaiser_risk > 0.05 and (kaiser_facing_cands >= 1.0 or kaiser_defense_cands >= 1.0)
        )
        ceremonial_open_window = ceremonial_stun > 0.05 and ceremonial_high_impact >= 1.0
        boss_window_open = bool(kaiser_pressure_window or ceremonial_open_window)
        flags = {
            "no_legal_positive_action": positive_count == 0,
            "transient_only_end_turn": transient,
            "has_energy_and_positive_action": energy > 0.05 and positive_count > 0,
            "has_urgent_or_mandatory_action": urgent_count > 0,
            "has_strategic_defer_reason": strategic_defer_available,
            "has_deferable_action": deferable_count > 0,
            "kaiser_pressure_window_open": kaiser_pressure_window,
            "ceremonial_open_window": ceremonial_open_window,
            "boss_window_open": boss_window_open,
        }
        if not end_turn_indices:
            return "unknown", flags
        # Transient (bridge handed us only end_turn this frame) is treated as a
        # forced selection, not bad — boss-window override does not apply.
        if transient:
            return "forced_end_turn", flags
        if wasteful or boss_window_open:
            return "bad_end_turn", flags
        if positive_count == 0:
            return "forced_end_turn", flags
        if strategic_defer_available:
            return "strategic_defer_end_turn", flags
        return "unknown", flags

    def _dump_selected_end_turn_context(
        self,
        encoded_obs: dict[str, Any] | None,
        raw_obs: dict[str, Any] | None,
        action_mask: np.ndarray,
        legal_actions: list[Any] | None,
        chosen_idx: int,
        context: dict[str, Any],
        search_policy: np.ndarray | None,
        search_stats: dict[str, Any] | None,
        action_diagnostics: dict[str, Any] | None,
        encounter: str,
        tier: str,
    ) -> None:
        """Append one JSONL entry per selected end_turn action.

        Designed to stay cheap: reuses precomputed ``context`` indices and
        existing search probabilities; does not loop legality classifiers again.
        """

        if getattr(self, "_end_turn_context_dump_disabled", False):
            return
        cap = int(getattr(self, "_end_turn_context_dump_max", 50000))
        if cap > 0 and self._end_turn_context_dump_count >= cap:
            return
        try:
            mask_np = np.asarray(action_mask, dtype=np.float32).reshape(-1)
            class_name, reason_flags = self._classify_end_turn_action(context, action_diagnostics)

            raw_combat = raw_obs.get("combat") if isinstance(raw_obs, dict) and isinstance(raw_obs.get("combat"), dict) else {}
            raw_player = raw_obs.get("player") if isinstance(raw_obs, dict) and isinstance(raw_obs.get("player"), dict) else {}
            if not raw_player and isinstance(raw_combat.get("player"), dict):
                raw_player = raw_combat.get("player")
            incoming, block, hp = self._incoming_damage_pressure(raw_obs)
            energy_val = float(context.get("energy", 0.0) or 0.0)
            try:
                hand_count = len(raw_combat.get("hand") or []) if isinstance(raw_combat.get("hand"), list) else 0
                draw_count = len(raw_combat.get("draw_pile") or []) if isinstance(raw_combat.get("draw_pile"), list) else 0
                discard_count = len(raw_combat.get("discard_pile") or []) if isinstance(raw_combat.get("discard_pile"), list) else 0
                exhaust_count = len(raw_combat.get("exhaust_pile") or []) if isinstance(raw_combat.get("exhaust_pile"), list) else 0
            except Exception:
                hand_count = draw_count = discard_count = exhaust_count = 0

            x_cost_count = 0
            zero_x_count = 0
            try:
                legal_count = min(len(legal_actions or []), MAX_ACTIONS, mask_np.shape[0])
                for idx in range(legal_count):
                    if mask_np[idx] <= 0:
                        continue
                    action = (legal_actions or [])[idx]
                    if self._semantic_family(action) == "play_card" and self._is_x_cost_action(encoded_obs, idx, action):
                        x_cost_count += 1
                        if energy_val <= 0.05:
                            zero_x_count += 1
            except Exception:
                pass

            boss_context: dict[str, Any] = {}
            try:
                if isinstance(raw_obs, dict):
                    boss_ctx = build_boss_mechanics_context(raw_obs)
                    boss_context = {
                        "kaiser_back_attack_risk": float(self._boss_context_max(boss_ctx, "back_attack_risk")),
                        "kaiser_back_attack_active": float(self._boss_context_max(boss_ctx, "back_attack_active")),
                        "incoming_damage_multiplier_norm": float(self._boss_context_max(boss_ctx, "incoming_damage_multiplier_norm")),
                        "kaiser_facing_change_candidate_count": int(float(
                            (search_stats or {}).get("combat_quality_kaiser_facing_change_candidate_count", 0.0) or 0.0
                        )),
                        "ceremonial_one_card_lock": float(self._boss_context_max(boss_ctx, "one_card_lock")),
                        "ceremonial_stun_window": float(self._boss_context_max(boss_ctx, "stun_window")),
                    }
            except Exception:
                boss_context = {}

            top_actions: list[dict[str, Any]] = []
            try:
                policy_np = (
                    np.asarray(search_policy, dtype=np.float32).reshape(-1)
                    if search_policy is not None
                    else np.zeros(MAX_ACTIONS, dtype=np.float32)
                )
                # Rank legal actions by search_policy probability; fall back to mask order.
                legal_count = min(len(legal_actions or []), MAX_ACTIONS, mask_np.shape[0], policy_np.shape[0])
                ranked: list[tuple[int, float]] = []
                for idx in range(legal_count):
                    if mask_np[idx] <= 0:
                        continue
                    score = float(policy_np[idx]) if idx < policy_np.shape[0] else 0.0
                    ranked.append((idx, score))
                ranked.sort(key=lambda kv: kv[1], reverse=True)
                for rank, (idx, score) in enumerate(ranked[:6], start=1):
                    action = (legal_actions or [])[idx]
                    family = self._semantic_family(action) if isinstance(action, dict) else ""
                    card = action.get("card") if isinstance(action, dict) and isinstance(action.get("card"), dict) else {}
                    target = action.get("target") if isinstance(action, dict) and isinstance(action.get("target"), dict) else {}
                    tags: list[str] = []
                    if isinstance(action, dict):
                        if int(idx) in {int(i) for i in context.get("urgent_positive_indices") or []}:
                            tags.append("urgent_positive")
                        if int(idx) in {int(i) for i in context.get("deferable_positive_indices") or []}:
                            tags.append("deferable")
                        if int(idx) in {int(i) for i in context.get("setup_scaling_indices") or []}:
                            tags.append("setup_scaling")
                        if family == "end_turn":
                            tags.append("end_turn")
                        if family == "play_card" and self._is_x_cost_action(encoded_obs, idx, action):
                            tags.append("x_cost")
                            if energy_val <= 0.05:
                                tags.append("zero_energy_x_cost")
                        if self._is_kaiser_facing_change_action(action, raw_obs):
                            tags.append("kaiser_facing_change")
                    top_actions.append({
                        "rank": rank,
                        "action_idx": int(idx),
                        "is_chosen": int(idx) == int(chosen_idx),
                        "family": family,
                        "card_id": card.get("id") if isinstance(card, dict) else None,
                        "title": (action.get("title") if isinstance(action, dict) else None) or (card.get("title") if isinstance(card, dict) else None),
                        "target": target.get("name") if isinstance(target, dict) else None,
                        "score": float(score),
                        "tags": tags,
                    })
            except Exception:
                top_actions = []

            payload = {
                "time": time.time(),
                "global_step": int(getattr(self, "total_steps", 0)),
                "episode_id": int(getattr(self, "episode_count", 0)),
                "encounter_id": str(encounter or ""),
                "tier": str(tier or ""),
                "turn": raw_combat.get("round"),
                "selected_action_idx": int(chosen_idx),
                "selected_family": "end_turn",
                "end_turn_class": class_name,
                "reason_flags": reason_flags,
                "player": {
                    "hp": float(hp),
                    "max_hp": float(self._safe_float(raw_player.get("max_hp"))) if isinstance(raw_player, dict) else 0.0,
                    "block": float(block),
                    "energy": float(energy_val),
                },
                "combat": {
                    "incoming_damage": float(incoming),
                    "hand_count": int(hand_count),
                    "draw_count": int(draw_count),
                    "discard_count": int(discard_count),
                    "exhaust_count": int(exhaust_count),
                },
                "counts": {
                    "legal_action_count": int(np.sum(mask_np > 0)),
                    "playable_cards_left": int(len(context.get("positive_indices") or [])),
                    "positive_action_count": int(context.get("positive_progress_count", 0) or 0),
                    "urgent_action_count": int(context.get("urgent_positive_count", 0) or 0),
                    "deferable_action_count": int(context.get("deferable_positive_count", 0) or 0),
                    "deferable_exhaust_count": int(context.get("deferable_exhaust_count", 0) or 0),
                    "x_cost_candidate_count": int(x_cost_count),
                    "zero_energy_x_cost_candidate_count": int(zero_x_count),
                    "potion_available_count": int(context.get("potion_available_count", 0) or 0),
                    "potion_urgent_count": int(context.get("potion_urgent_count", 0) or 0),
                },
                "boss_context": boss_context,
                "top_legal_actions": top_actions,
            }
            path = Path(self.log_dir) / "diagnostics" / "end_turn_contexts.jsonl"
            path.parent.mkdir(parents=True, exist_ok=True)
            with path.open("a", encoding="utf-8") as fh:
                fh.write(json.dumps(payload, ensure_ascii=False, default=str) + "\n")
            self._end_turn_context_dump_count += 1
        except Exception:
            return

    def _dump_potion_transition(self, record: dict[str, Any] | None) -> None:
        """Append one JSONL line per use_potion transition emitted by combat_env."""

        if not isinstance(record, dict):
            return
        if getattr(self, "_potion_transition_dump_disabled", False):
            return
        cap = int(getattr(self, "_potion_transition_dump_max", 200000))
        if cap > 0 and getattr(self, "_potion_transition_dump_count", 0) >= cap:
            return
        try:
            payload = {
                "time": time.time(),
                "global_step": int(getattr(self, "total_steps", 0)),
                "episode_id": int(getattr(self, "episode_count", 0)),
                **record,
            }
            path = Path(self.log_dir) / "diagnostics" / "potion_transitions.jsonl"
            path.parent.mkdir(parents=True, exist_ok=True)
            with path.open("a", encoding="utf-8") as fh:
                fh.write(json.dumps(payload, ensure_ascii=False, default=str) + "\n")
            self._potion_transition_dump_count += 1
        except Exception:
            return

    def _dump_death_final_potions(
        self,
        *,
        loss: bool,
        final_potion_count: int,
        used_potion_count: int,
        adjusted_unused: int,
        encounter: str,
        tier: str,
        potion_dump: list[dict[str, Any]] | None,
    ) -> None:
        """Append a death-final potion accounting line to the same JSONL.

        Lets post-mortem confirm whether ``potion_unused_on_death`` flipped
        because of a missing transition record vs a real hoarded potion.
        """

        if getattr(self, "_potion_transition_dump_disabled", False):
            return
        cap = int(getattr(self, "_potion_transition_dump_max", 200000))
        if cap > 0 and getattr(self, "_potion_transition_dump_count", 0) >= cap:
            return
        try:
            payload = {
                "time": time.time(),
                "global_step": int(getattr(self, "total_steps", 0)),
                "episode_id": int(getattr(self, "episode_count", 0)),
                "event": "death_final_potions",
                "loss": bool(loss),
                "encounter_id": str(encounter or ""),
                "tier": str(tier or ""),
                "final_potion_count": int(final_potion_count),
                "used_potion_count": int(used_potion_count),
                "adjusted_unused": int(adjusted_unused),
                "potion_dump": list(potion_dump or []),
            }
            path = Path(self.log_dir) / "diagnostics" / "potion_transitions.jsonl"
            path.parent.mkdir(parents=True, exist_ok=True)
            with path.open("a", encoding="utf-8") as fh:
                fh.write(json.dumps(payload, ensure_ascii=False, default=str) + "\n")
            self._potion_transition_dump_count += 1
        except Exception:
            return

    @staticmethod
    def _classify_action_offenders(
        *,
        search_stats: dict[str, Any] | None,
        encounter: str,
        family: str,
    ) -> list[str]:
        """Return the offender types triggered by the selected action's stats.

        Pure function over stats already populated by ``_combat_action_quality_bias``
        and ``_selected_combat_quality_stats``.  Boss-specific offenders are gated
        by ``encounter`` so e.g. ``kaiser_*`` will never fire on a Construct fight.
        """

        s = search_stats if isinstance(search_stats, dict) else {}
        encounter_l = str(encounter or "").lower()
        family_l = str(family or "").lower()

        def gt(key: str, threshold: float = 0.5) -> bool:
            try:
                return float(s.get(key, 0.0) or 0.0) > threshold
            except (TypeError, ValueError):
                return False

        out: list[str] = []
        if gt("combat_quality_wasteful_end_turn_selected"):
            out.append("bad_end_turn")
        if gt("combat_quality_strategic_defer_end_turn_selected"):
            out.append("strategic_defer_end_turn")
        if gt("combat_quality_strategic_skip_selected"):
            out.append("strategic_skip_selected")
        if gt("combat_quality_refund_no_followup_selected"):
            out.append("refund_no_followup_selected")
        if gt("combat_quality_zero_energy_x_cost_selected"):
            out.append("zero_energy_x_cost_selected")
        if gt("combat_quality_x_cost_bad_selected"):
            out.append("x_cost_low_value_selected")
        if family_l in {"use_potion", "potion"}:
            if (
                gt("combat_quality_potion_low_urgency_selected")
                or gt("combat_quality_potion_save_recommended_selected")
                or gt("combat_quality_potion_no_followup_selected")
                or gt("combat_quality_potion_block_waste_selected")
                or gt("combat_quality_potion_overkill_selected")
            ):
                out.append("low_quality_potion_selected")
        # End-turn while save-value-recommended potions exist.
        if family_l == "end_turn":
            try:
                save_value = float(s.get("combat_quality_potion_save_value_mean", 0.0) or 0.0)
                save_recommended = float(s.get("combat_quality_potion_save_recommended_count", 0.0) or 0.0)
                if save_value >= 0.6 and save_recommended >= 1.0:
                    out.append("high_save_value_potion_unused")
            except (TypeError, ValueError):
                pass

        is_kaiser = "kaiser" in encounter_l
        if is_kaiser:
            risk = float(s.get("combat_quality_kaiser_back_attack_risk", 0.0) or 0.0)
            if risk > 0.05:
                if gt("combat_quality_kaiser_risky_end_turn_selected"):
                    out.append("kaiser_risky_end_turn")
                facing_cands = float(s.get("combat_quality_kaiser_facing_change_candidate_count", 0.0) or 0.0)
                facing_sel = float(s.get("combat_quality_kaiser_facing_change_selected", 0.0) or 0.0)
                if facing_cands >= 1.0 and facing_sel < 0.5 and family_l != "end_turn":
                    out.append("kaiser_facing_missed")

        if "ceremonial" in encounter_l:
            if gt("combat_quality_ceremonial_low_impact_selected"):
                out.append("ceremonial_low_impact_under_lock")
            stun_window = float(s.get("combat_quality_ceremonial_stun_window", 0.0) or 0.0)
            high_impact_sel = float(s.get("combat_quality_ceremonial_high_impact_selected", 0.0) or 0.0)
            if stun_window > 0.05 and high_impact_sel < 0.5:
                out.append("ceremonial_missed_stun_window")

        if "insatiable" in encounter_l:
            if gt("combat_quality_strategic_skip_selected"):
                out.append("insatiable_strategic_skip")
            if gt("combat_quality_refund_no_followup_selected"):
                out.append("insatiable_strategic_skip")
            if gt("combat_quality_wasteful_end_turn_selected"):
                out.append("insatiable_strategic_skip")

        # Preserve insertion order on dedup so primary offenders rank first.
        seen: set[str] = set()
        deduped: list[str] = []
        for item in out:
            if item in seen:
                continue
            seen.add(item)
            deduped.append(item)
        return deduped

    def _dump_action_offender(
        self,
        *,
        encoded_obs: dict[str, Any] | None,
        raw_obs: dict[str, Any] | None,
        action_mask: np.ndarray,
        legal_actions: list[Any] | None,
        chosen_idx: int,
        chosen_action: Any,
        offender_types: list[str],
        search_stats: dict[str, Any] | None,
        encounter: str,
        tier: str,
    ) -> None:
        """Append one JSONL entry per (step, offender_type) pair."""

        if not offender_types:
            return
        if getattr(self, "_action_offender_dump_disabled", False):
            return
        cap = int(getattr(self, "_action_offender_dump_max", 100000))
        if cap > 0 and getattr(self, "_action_offender_dump_count", 0) >= cap:
            return
        try:
            mask_np = np.asarray(action_mask, dtype=np.float32).reshape(-1)
            raw_combat = raw_obs.get("combat") if isinstance(raw_obs, dict) and isinstance(raw_obs.get("combat"), dict) else {}
            raw_player = raw_obs.get("player") if isinstance(raw_obs, dict) and isinstance(raw_obs.get("player"), dict) else {}
            if not raw_player and isinstance(raw_combat.get("player"), dict):
                raw_player = raw_combat.get("player")
            incoming, block, hp = self._incoming_damage_pressure(raw_obs)
            family = self._semantic_family(chosen_action) if isinstance(chosen_action, dict) else ""
            card = chosen_action.get("card") if isinstance(chosen_action, dict) and isinstance(chosen_action.get("card"), dict) else {}
            potion = chosen_action.get("potion") if isinstance(chosen_action, dict) and isinstance(chosen_action.get("potion"), dict) else {}
            target = chosen_action.get("target") if isinstance(chosen_action, dict) and isinstance(chosen_action.get("target"), dict) else {}
            stats = search_stats if isinstance(search_stats, dict) else {}
            state_summary = {
                "energy": float(stats.get("combat_quality_energy", 0.0) or 0.0),
                "incoming_damage": float(incoming),
                "block": float(block),
                "hp": float(hp),
                "max_hp": float(self._safe_float(raw_player.get("max_hp"))) if isinstance(raw_player, dict) else 0.0,
                "legal_action_count": int(np.sum(mask_np > 0)),
                "positive_action_count": float(stats.get("combat_quality_positive_action_count", 0.0) or 0.0),
                "urgent_positive_action_count": float(stats.get("combat_quality_urgent_positive_action_count", 0.0) or 0.0),
                "deferable_positive_action_count": float(stats.get("combat_quality_deferable_positive_action_count", 0.0) or 0.0),
                "x_cost_available_count": float(stats.get("combat_quality_x_cost_available_count", 0.0) or 0.0),
                "zero_energy_x_cost_count": float(stats.get("combat_quality_zero_energy_x_cost_count", 0.0) or 0.0),
                "kaiser_back_attack_risk": float(stats.get("combat_quality_kaiser_back_attack_risk", 0.0) or 0.0),
                "kaiser_facing_change_candidate_count": float(stats.get("combat_quality_kaiser_facing_change_candidate_count", 0.0) or 0.0),
                "ceremonial_one_card_lock": float(stats.get("combat_quality_ceremonial_one_card_lock", 0.0) or 0.0),
                "ceremonial_stun_window": float(stats.get("combat_quality_ceremonial_stun_window", 0.0) or 0.0),
            }
            reason_flags = {
                key: float(stats.get(f"combat_quality_{key}", 0.0) or 0.0) > 0.5
                for key in (
                    "wasteful_end_turn_selected",
                    "strategic_defer_end_turn_selected",
                    "strategic_skip_selected",
                    "refund_no_followup_selected",
                    "zero_energy_x_cost_selected",
                    "potion_low_urgency_selected",
                    "potion_save_recommended_selected",
                    "potion_no_followup_selected",
                    "potion_block_waste_selected",
                    "potion_overkill_selected",
                    "kaiser_facing_change_selected",
                    "kaiser_pressure_selected",
                    "kaiser_risky_end_turn_selected",
                    "ceremonial_low_impact_selected",
                    "ceremonial_high_impact_selected",
                )
            }
            alternatives: list[dict[str, Any]] = []
            try:
                count = min(len(legal_actions or []), MAX_ACTIONS, mask_np.shape[0])
                for idx in range(count):
                    if idx == int(chosen_idx) or mask_np[idx] <= 0:
                        continue
                    cand = (legal_actions or [])[idx]
                    if not isinstance(cand, dict):
                        continue
                    cand_card = cand.get("card") if isinstance(cand.get("card"), dict) else {}
                    alternatives.append({
                        "action_idx": int(idx),
                        "family": self._semantic_family(cand),
                        "card_id": cand_card.get("id") if isinstance(cand_card, dict) else None,
                        "title": cand.get("title") or (cand_card.get("title") if isinstance(cand_card, dict) else None),
                    })
                    if len(alternatives) >= 5:
                        break
            except Exception:
                alternatives = []
            path = Path(self.log_dir) / "diagnostics" / "action_offenders.jsonl"
            path.parent.mkdir(parents=True, exist_ok=True)
            for offender_type in offender_types:
                payload = {
                    "time": time.time(),
                    "global_step": int(getattr(self, "total_steps", 0)),
                    "episode_id": int(getattr(self, "episode_count", 0)),
                    "encounter_id": str(encounter or ""),
                    "tier": str(tier or ""),
                    "turn": raw_combat.get("round"),
                    "offender_type": offender_type,
                    "selected_action_idx": int(chosen_idx),
                    "selected_family": family,
                    "selected_card_id": card.get("id") if isinstance(card, dict) else None,
                    "selected_potion_id": potion.get("id") if isinstance(potion, dict) else None,
                    "selected_title": (chosen_action.get("title") if isinstance(chosen_action, dict) else None)
                        or (card.get("title") if isinstance(card, dict) else None)
                        or (potion.get("title") if isinstance(potion, dict) else None),
                    "selected_target": target.get("name") if isinstance(target, dict) else None,
                    "reason_flags": reason_flags,
                    "state_summary": state_summary,
                    "alternative_actions": alternatives,
                }
                with path.open("a", encoding="utf-8") as fh:
                    fh.write(json.dumps(payload, ensure_ascii=False, default=str) + "\n")
                self._action_offender_dump_count = int(getattr(self, "_action_offender_dump_count", 0)) + 1
                if cap > 0 and self._action_offender_dump_count >= cap:
                    return
        except Exception:
            return

    def _is_kaiser_risk_handling_action(self, action: Any, raw_obs: Any | None = None) -> bool:
        """Actions that can directly answer Kaiser/Rocket/Crusher back-attack risk.

        This intentionally includes explicit facing changes as *risk handling*, not only
        block/debuff cards.  The old boss_combat/kaiser_defense_candidate_count metric
        was interpreted as "defense / avoidance candidate"; if we keep facing separate
        only, a real turn-around action can exist while the legacy candidate metric stays
        at zero and misleads diagnosis.
        """
        if not isinstance(action, dict):
            return False
        family = self._semantic_family(action)
        if family == "end_turn":
            return False
        if self._is_kaiser_facing_change_action(action, raw_obs):
            return True
        roles = self._action_roles(action)
        if roles.intersection({"block", "debuff", "weak", "vulnerable"}):
            return True
        block = max(
            self._action_metric(action, "block"),
            self._action_metric(action, "total_block"),
            self._action_numeric_value(action, ("block", "total_block", "preview_block")),
        )
        if block > 0.0:
            return True
        damage = max(
            self._action_metric(action, "damage"),
            self._action_metric(action, "total_damage"),
            self._action_numeric_value(action, ("damage", "total_damage", "attack_damage", "preview_damage")),
        )
        if damage <= 0.0:
            return False
        target_hp = self._target_enemy_hp(action, raw_obs if isinstance(raw_obs, dict) else None)
        if target_hp > 0.0:
            # Damage only answers the back-attack mechanic if it kills the risky
            # side or represents a real phase/kill pressure, not because it is a
            # potion/card button that happens to be legal.
            return bool(damage >= target_hp or damage >= min(target_hp, max(12.0, 0.35 * target_hp)))
        return bool(damage >= 18.0)

    def _is_kaiser_pressure_action(self, action: Any) -> bool:
        """Damage-only actions that may race/kill but do not by themselves solve facing.

        Keeping this separate prevents a Strike-only hand from being mislabeled as having
        "defense" available, while still making snapshot composition visible in TB.
        """
        if not isinstance(action, dict):
            return False
        if self._semantic_family(action) != "play_card":
            return False
        roles = self._action_roles(action)
        return (
            bool(roles.intersection({"attack", "damage"}))
            or self._action_metric(action, "damage") > 0.0
            or self._action_metric(action, "total_damage") > 0.0
        )

    def _raw_end_turn_context(
        self,
        encoded_obs: dict[str, Any] | None,
        action_mask: np.ndarray,
        legal_actions: list[Any] | None,
    ) -> dict[str, Any]:
        mask_np = np.asarray(action_mask, dtype=np.float32).reshape(-1)
        raw_obs = self._current_raw_combat_obs()
        energy = self._combat_energy(encoded_obs, raw_obs)
        end_turn_indices: list[int] = []
        positive_indices: list[int] = []
        urgent_positive_indices: list[int] = []
        deferable_positive_indices: list[int] = []
        deferable_exhaust_indices: list[int] = []
        ethereal_urgent_indices: list[int] = []
        energy_gain_without_followup_indices: list[int] = []
        typed_followup_missing_indices: list[int] = []
        typed_future_penalty_indices: list[int] = []
        typed_no_draw_indices: list[int] = []
        typed_card_state_mutation_indices: list[int] = []
        setup_followup_dependent_indices: list[int] = []
        setup_followup_available_indices: list[int] = []
        potion_available_indices: list[int] = []
        potion_urgent_indices: list[int] = []
        potion_low_urgency_indices: list[int] = []
        potion_save_recommended_indices: list[int] = []
        potion_no_followup_indices: list[int] = []
        potion_lethal_indices: list[int] = []
        potion_prevent_lethal_indices: list[int] = []
        potion_mechanism_indices: list[int] = []
        potion_overkill_indices: list[int] = []
        potion_block_waste_indices: list[int] = []
        potion_use_quality_values: list[float] = []
        potion_waste_risk_values: list[float] = []
        potion_save_value_values: list[float] = []
        potion_hand_context_good_indices: list[int] = []
        potion_hand_context_bad_indices: list[int] = []
        potion_long_term_indices: list[int] = []
        potion_requires_followup_indices: list[int] = []
        potion_family_counts: dict[str, int] = {}
        potion_id_present: dict[str, int] = {}
        setup_scaling_indices: list[int] = []
        zero_cost_positive = False
        zero_cost_urgent = False
        legal_count = min(len(legal_actions or []), MAX_ACTIONS, mask_np.shape[0])
        for idx in range(legal_count):
            if mask_np[idx] <= 0:
                continue
            action = (legal_actions or [])[idx]
            family = self._semantic_family(action)
            if family == "end_turn":
                end_turn_indices.append(int(idx))
                continue
            classification = self._classify_positive_combat_action(
                action,
                idx,
                encoded_obs,
                raw_obs,
                legal_actions,
                mask_np,
                energy,
            )
            if bool(classification.get("potion_available", False)):
                potion_available_indices.append(int(idx))
                potion_use_quality_values.append(float(classification.get("potion_use_quality", 0.0) or 0.0))
                potion_waste_risk_values.append(float(classification.get("potion_waste_risk", 0.0) or 0.0))
                potion_save_value_values.append(float(classification.get("potion_save_value", 0.0) or 0.0))
                if bool(classification.get("potion_hand_context_good", False)):
                    potion_hand_context_good_indices.append(int(idx))
                if bool(classification.get("potion_hand_context_bad", False)):
                    potion_hand_context_bad_indices.append(int(idx))
                if bool(classification.get("potion_long_term_value", False)):
                    potion_long_term_indices.append(int(idx))
                if bool(classification.get("potion_requires_followup", False)):
                    potion_requires_followup_indices.append(int(idx))
                fam_list = classification.get("potion_effect_family") or []
                for fam in fam_list:
                    fam_key = str(fam).lower()
                    if fam_key:
                        potion_family_counts[fam_key] = potion_family_counts.get(fam_key, 0) + 1
                pid_str = str(classification.get("potion_id") or "")
                if pid_str:
                    potion_id_present[pid_str] = potion_id_present.get(pid_str, 0) + 1
            if bool(classification.get("potion_urgent", False)):
                potion_urgent_indices.append(int(idx))
            if bool(classification.get("potion_low_urgency", False)):
                potion_low_urgency_indices.append(int(idx))
            if bool(classification.get("potion_save_recommended", False)):
                potion_save_recommended_indices.append(int(idx))
            if bool(classification.get("potion_no_followup", False)):
                potion_no_followup_indices.append(int(idx))
            if bool(classification.get("potion_lethal", False)):
                potion_lethal_indices.append(int(idx))
            if bool(classification.get("potion_prevent_lethal", False)):
                potion_prevent_lethal_indices.append(int(idx))
            if bool(classification.get("potion_mechanism_answer", False)):
                potion_mechanism_indices.append(int(idx))
            if bool(classification.get("potion_overkill", False)):
                potion_overkill_indices.append(int(idx))
            if bool(classification.get("potion_block_waste", False)):
                potion_block_waste_indices.append(int(idx))
            if classification["positive"]:
                positive_indices.append(int(idx))
                roles = self._action_roles(action)
                if roles.intersection({"setup", "scaling", "power"}):
                    setup_scaling_indices.append(int(idx))
                if family == "play_card" and self._is_zero_cost_action(action):
                    zero_cost_positive = True
                if classification["urgent"]:
                    urgent_positive_indices.append(int(idx))
                    if family == "play_card" and self._is_zero_cost_action(action):
                        zero_cost_urgent = True
                if classification["deferable"]:
                    deferable_positive_indices.append(int(idx))
                if classification["deferable_exhaust"]:
                    deferable_exhaust_indices.append(int(idx))
                if classification["ethereal_urgent"]:
                    ethereal_urgent_indices.append(int(idx))
                if classification["energy_without_followup"]:
                    energy_gain_without_followup_indices.append(int(idx))
                if bool(classification.get("followup_missing", False)):
                    typed_followup_missing_indices.append(int(idx))
                if bool(classification.get("typed_future_penalty", False)):
                    typed_future_penalty_indices.append(int(idx))
                if bool(classification.get("typed_no_draw", False)):
                    typed_no_draw_indices.append(int(idx))
                if bool(classification.get("typed_card_state_mutation", False)):
                    typed_card_state_mutation_indices.append(int(idx))
                if bool(classification.get("setup_followup_dependent", False)):
                    setup_followup_dependent_indices.append(int(idx))
                if bool(classification.get("setup_followup_available", False)):
                    setup_followup_available_indices.append(int(idx))
        positive_progress_count = len(positive_indices)
        urgent_positive_count = len(urgent_positive_indices)
        deferable_positive_count = len(deferable_positive_indices)
        # True waste is now gated on urgent progress.  Exhaust/retain/HP-cost
        # resource cards can be legal and positive but strategically correct to
        # let flow to discard/retain instead of consuming the combat loop.
        wasteful = bool(end_turn_indices) and urgent_positive_count > 0 and (energy > 0.05 or zero_cost_urgent)
        strategic_defer_available = bool(end_turn_indices) and positive_progress_count > 0 and urgent_positive_count == 0 and deferable_positive_count > 0
        severity = 0.0
        if wasteful:
            severity = (
                1.0
                + 0.45 * min(max(float(energy), 0.0), 3.0)
                + 0.35 * min(float(urgent_positive_count), 4.0)
                + (0.75 if zero_cost_urgent else 0.0)
                + (0.35 if setup_scaling_indices else 0.0)
                + (0.25 if ethereal_urgent_indices else 0.0)
            )
        return {
            "wasteful": wasteful,
            "true_wasteful": wasteful,
            "strategic_defer_available": strategic_defer_available,
            "energy": float(energy),
            "end_turn_indices": end_turn_indices,
            "positive_indices": positive_indices,
            "urgent_positive_indices": urgent_positive_indices,
            "deferable_positive_indices": deferable_positive_indices,
            "deferable_exhaust_indices": deferable_exhaust_indices,
            "ethereal_urgent_indices": ethereal_urgent_indices,
            "energy_gain_without_followup_indices": energy_gain_without_followup_indices,
            "typed_followup_missing_indices": typed_followup_missing_indices,
            "typed_future_penalty_indices": typed_future_penalty_indices,
            "typed_no_draw_indices": typed_no_draw_indices,
            "typed_card_state_mutation_indices": typed_card_state_mutation_indices,
            "setup_followup_dependent_indices": setup_followup_dependent_indices,
            "setup_followup_available_indices": setup_followup_available_indices,
            "potion_available_indices": potion_available_indices,
            "potion_urgent_indices": potion_urgent_indices,
            "potion_low_urgency_indices": potion_low_urgency_indices,
            "potion_save_recommended_indices": potion_save_recommended_indices,
            "potion_no_followup_indices": potion_no_followup_indices,
            "potion_lethal_indices": potion_lethal_indices,
            "potion_prevent_lethal_indices": potion_prevent_lethal_indices,
            "potion_mechanism_indices": potion_mechanism_indices,
            "potion_overkill_indices": potion_overkill_indices,
            "potion_block_waste_indices": potion_block_waste_indices,
            "setup_scaling_indices": setup_scaling_indices,
            "positive_progress_count": positive_progress_count,
            "urgent_positive_count": urgent_positive_count,
            "deferable_positive_count": deferable_positive_count,
            "deferable_exhaust_count": len(deferable_exhaust_indices),
            "ethereal_urgent_count": len(ethereal_urgent_indices),
            "energy_gain_without_followup_count": len(energy_gain_without_followup_indices),
            "typed_followup_missing_count": len(typed_followup_missing_indices),
            "typed_future_penalty_count": len(typed_future_penalty_indices),
            "typed_no_draw_count": len(typed_no_draw_indices),
            "typed_card_state_mutation_count": len(typed_card_state_mutation_indices),
            "setup_followup_dependent_count": len(setup_followup_dependent_indices),
            "setup_followup_available_count": len(setup_followup_available_indices),
            "potion_available_count": len(potion_available_indices),
            "potion_urgent_count": len(potion_urgent_indices),
            "potion_low_urgency_count": len(potion_low_urgency_indices),
            "potion_save_recommended_count": len(potion_save_recommended_indices),
            "potion_no_followup_count": len(potion_no_followup_indices),
            "potion_lethal_count": len(potion_lethal_indices),
            "potion_prevent_lethal_count": len(potion_prevent_lethal_indices),
            "potion_mechanism_count": len(potion_mechanism_indices),
            "potion_overkill_count": len(potion_overkill_indices),
            "potion_block_waste_count": len(potion_block_waste_indices),
            "potion_use_quality_mean": float(np.mean(potion_use_quality_values)) if potion_use_quality_values else 0.0,
            "potion_waste_risk_mean": float(np.mean(potion_waste_risk_values)) if potion_waste_risk_values else 0.0,
            "potion_save_value_mean": float(np.mean(potion_save_value_values)) if potion_save_value_values else 0.0,
            "potion_hand_context_good_indices": potion_hand_context_good_indices,
            "potion_hand_context_bad_indices": potion_hand_context_bad_indices,
            "potion_long_term_indices": potion_long_term_indices,
            "potion_requires_followup_indices": potion_requires_followup_indices,
            "potion_hand_context_good_count": len(potion_hand_context_good_indices),
            "potion_hand_context_bad_count": len(potion_hand_context_bad_indices),
            "potion_long_term_count": len(potion_long_term_indices),
            "potion_requires_followup_count": len(potion_requires_followup_indices),
            "potion_family_counts": potion_family_counts,
            "potion_id_present": potion_id_present,
            "zero_cost_positive": zero_cost_positive,
            "zero_cost_urgent": zero_cost_urgent,
            "severity": float(severity),
        }

    def _obs_semantic_role_active(self, obs: dict[str, Any] | None, index: int, role: str) -> bool:
        if not isinstance(obs, dict) or index < 0:
            return False
        semantic_actions = obs.get("semantic_actions")
        if semantic_actions is None:
            return False
        try:
            semantic_np = semantic_actions.detach().cpu().numpy() if isinstance(semantic_actions, torch.Tensor) else np.asarray(semantic_actions)
            if semantic_np.ndim != 2 or index >= semantic_np.shape[0]:
                return False
            role_offset = len(SEMANTIC_ACTION_FAMILIES) + 0  # target scopes added below dynamically for import-stability
            # Avoid importing target-scope count into every caller; the vector layout is families + target_scopes + roles.
            from sts2_env.semantic_action import SEMANTIC_TARGET_SCOPES
            role_offset = len(SEMANTIC_ACTION_FAMILIES) + len(SEMANTIC_TARGET_SCOPES)
            role_index = SEMANTIC_ROLE_NAMES.index(role)
            vector_index = role_offset + role_index
            return bool(semantic_np[index].shape[0] > vector_index and semantic_np[index][vector_index] > 0.5)
        except Exception:
            return False

    def _is_x_cost_action(self, obs: dict[str, Any] | None, index: int, action: Any) -> bool:
        if self._obs_semantic_role_active(obs, index, "x_cost"):
            return True
        if not isinstance(action, dict):
            return False
        semantic = action.get("semantic") if isinstance(action.get("semantic"), dict) else {}
        if "x_cost" in self._action_roles(action):
            return True
        try:
            if float(semantic.get("x_cost_value") or 0.0) > 0.0:
                return True
        except (TypeError, ValueError):
            pass
        card = action.get("card") if isinstance(action.get("card"), dict) else {}
        if bool(card.get("x_cost") or card.get("costs_x") or action.get("x_cost") or action.get("costs_x")):
            return True
        cost = action.get("card_cost", card.get("cost"))
        return str(cost).strip().upper() == "X"

    @classmethod
    def _x_cost_has_non_energy_effect(cls, action: Any) -> bool:
        """Whether an X-cost play produces value independent of current energy.

        StS2 X-cost cards usually scale with the energy spent, so a 0-energy X
        play is dominated.  But X-cost pile-manipulation/exhaust/transform/
        retain/keyword cards still mutate state at 0 energy — those plays must
        not be flagged as ``zero_energy_x_cost_selected`` offenders.
        """

        if not isinstance(action, dict):
            return False
        semantic = action.get("semantic") if isinstance(action.get("semantic"), dict) else {}
        if bool(semantic.get("x_cost_has_non_energy_effect", False)):
            return True
        for key in (
            "typed_modifies_hand",
            "typed_upgrade_hand",
            "typed_exhaust_cards",
            "typed_discard_cards",
            "typed_transform_cards",
            "typed_copy_cards",
            "typed_add_modifier",
            "typed_add_keyword",
            "typed_set_replay",
            "typed_retain_cards",
            "typed_card_state_mutation",
        ):
            if bool(semantic.get(key, False)):
                return True
        if cls._action_roles(action).intersection(
            {"facing_change", "stun", "artifact_strip", "lock", "mechanism"}
        ):
            return True
        return False

    @classmethod
    def _x_cost_diagnostic(cls, action: Any, current_energy: float) -> dict[str, float]:
        """Static-state X-cost view for diagnostics: effective energy + non-energy effect."""

        is_x = False
        if isinstance(action, dict):
            semantic = action.get("semantic") if isinstance(action.get("semantic"), dict) else {}
            card = action.get("card") if isinstance(action.get("card"), dict) else {}
            try:
                if float(semantic.get("x_cost_value") or 0.0) > 0.0:
                    is_x = True
            except (TypeError, ValueError):
                pass
            if not is_x:
                if bool(semantic.get("is_x_cost") or card.get("x_cost") or card.get("costs_x")
                        or action.get("x_cost") or action.get("costs_x")):
                    is_x = True
            if not is_x:
                cost_text = str(action.get("card_cost") or card.get("cost") or "").strip().upper()
                if cost_text == "X":
                    is_x = True
        if not is_x:
            return {
                "is_x_cost": 0.0,
                "x_cost_effective_energy": 0.0,
                "x_cost_has_non_energy_effect": 0.0,
                "x_cost_bad": 0.0,
            }
        effective_energy = max(float(current_energy), 0.0)
        non_energy = cls._x_cost_has_non_energy_effect(action)
        bad = effective_energy <= 0.05 and not non_energy
        return {
            "is_x_cost": 1.0,
            "x_cost_effective_energy": effective_energy,
            "x_cost_has_non_energy_effect": 1.0 if non_energy else 0.0,
            "x_cost_bad": 1.0 if bad else 0.0,
        }

    def _combat_action_quality_bias(
        self,
        obs: dict[str, Any],
        action_mask: np.ndarray,
        legal_actions: list[Any],
    ) -> tuple[np.ndarray, dict[str, float], set[int]]:
        """Hard combat action-quality prior for search-free policy selection.

        This is not a learning target replacement.  It prevents the direct rollout
        planner from repeatedly sampling actions that are mechanically dominated:
        ending turn while playable progress remains, and spending an X-card at
        zero energy.  MCTS already had an end-turn guard; direct policy did not.
        """

        mask_np = np.asarray(action_mask, dtype=np.float32).reshape(-1)
        bias = np.zeros(MAX_ACTIONS, dtype=np.float32)
        raw_obs = self._current_raw_combat_obs()
        energy = self._combat_energy(obs, raw_obs)
        zero_energy_x_indices: set[int] = set()
        playable_indices: set[int] = set()
        end_turn_indices: set[int] = set()
        x_cost_indices: set[int] = set()
        x_cost_bad_indices: set[int] = set()
        x_cost_effective_energy_sum = 0.0

        legal_count = min(len(legal_actions or []), MAX_ACTIONS, mask_np.shape[0])
        for idx in range(legal_count):
            if mask_np[idx] <= 0:
                continue
            action = legal_actions[idx]
            family = self._semantic_family(action)
            if family == "end_turn":
                end_turn_indices.add(idx)
            elif family == "play_card":
                playable_indices.add(idx)
            if family == "play_card" and self._is_x_cost_action(obs, idx, action):
                x_cost_indices.add(idx)
                x_diag = self._x_cost_diagnostic(action, float(energy))
                x_cost_effective_energy_sum += float(x_diag.get("x_cost_effective_energy", 0.0))
                if energy <= 0.05:
                    zero_energy_x_indices.add(idx)
                if float(x_diag.get("x_cost_bad", 0.0)) > 0.5:
                    x_cost_bad_indices.add(idx)

        context: dict[str, Any]
        try:
            context = self._raw_end_turn_context(obs, mask_np, legal_actions)
        except Exception:
            context = {}
        wasteful = bool(context.get("wasteful", False))
        positive_indices = {int(i) for i in context.get("positive_indices", []) if 0 <= int(i) < MAX_ACTIONS}
        urgent_positive_indices = {int(i) for i in context.get("urgent_positive_indices", []) if 0 <= int(i) < MAX_ACTIONS}
        deferable_positive_indices = {int(i) for i in context.get("deferable_positive_indices", []) if 0 <= int(i) < MAX_ACTIONS}
        deferable_exhaust_indices = {int(i) for i in context.get("deferable_exhaust_indices", []) if 0 <= int(i) < MAX_ACTIONS}
        energy_gain_without_followup_indices = {int(i) for i in context.get("energy_gain_without_followup_indices", []) if 0 <= int(i) < MAX_ACTIONS}
        typed_followup_missing_indices = {int(i) for i in context.get("typed_followup_missing_indices", []) if 0 <= int(i) < MAX_ACTIONS}
        typed_future_penalty_indices = {int(i) for i in context.get("typed_future_penalty_indices", []) if 0 <= int(i) < MAX_ACTIONS}
        typed_no_draw_indices = {int(i) for i in context.get("typed_no_draw_indices", []) if 0 <= int(i) < MAX_ACTIONS}
        typed_card_state_mutation_indices = {int(i) for i in context.get("typed_card_state_mutation_indices", []) if 0 <= int(i) < MAX_ACTIONS}
        setup_followup_dependent_indices = {int(i) for i in context.get("setup_followup_dependent_indices", []) if 0 <= int(i) < MAX_ACTIONS}
        setup_followup_available_indices = {int(i) for i in context.get("setup_followup_available_indices", []) if 0 <= int(i) < MAX_ACTIONS}
        potion_available_indices = {int(i) for i in context.get("potion_available_indices", []) if 0 <= int(i) < MAX_ACTIONS}
        potion_urgent_indices = {int(i) for i in context.get("potion_urgent_indices", []) if 0 <= int(i) < MAX_ACTIONS}
        potion_low_urgency_indices = {int(i) for i in context.get("potion_low_urgency_indices", []) if 0 <= int(i) < MAX_ACTIONS}
        potion_save_recommended_indices = {int(i) for i in context.get("potion_save_recommended_indices", []) if 0 <= int(i) < MAX_ACTIONS}
        potion_no_followup_indices = {int(i) for i in context.get("potion_no_followup_indices", []) if 0 <= int(i) < MAX_ACTIONS}
        potion_lethal_indices = {int(i) for i in context.get("potion_lethal_indices", []) if 0 <= int(i) < MAX_ACTIONS}
        potion_prevent_lethal_indices = {int(i) for i in context.get("potion_prevent_lethal_indices", []) if 0 <= int(i) < MAX_ACTIONS}
        potion_mechanism_indices = {int(i) for i in context.get("potion_mechanism_indices", []) if 0 <= int(i) < MAX_ACTIONS}
        potion_overkill_indices = {int(i) for i in context.get("potion_overkill_indices", []) if 0 <= int(i) < MAX_ACTIONS}
        potion_block_waste_indices = {int(i) for i in context.get("potion_block_waste_indices", []) if 0 <= int(i) < MAX_ACTIONS}
        setup_indices = {int(i) for i in context.get("setup_scaling_indices", []) if 0 <= int(i) < MAX_ACTIONS}
        end_turn_indices.update(int(i) for i in context.get("end_turn_indices", []) if 0 <= int(i) < MAX_ACTIONS)
        if not positive_indices and not urgent_positive_indices:
            # Fallback is intentionally weaker than the old detector: do not
            # label every legal play_card as urgent, because exhaust/resource
            # cards may be strategically deferred into the discard loop.
            positive_indices = set(playable_indices)
            urgent_positive_indices = {idx for idx in playable_indices if not self._is_exhausting_action(legal_actions[idx])}
            wasteful = energy > 0.05 and bool(end_turn_indices) and bool(urgent_positive_indices)
        boost_indices = urgent_positive_indices if urgent_positive_indices else positive_indices
        severity = max(float(context.get("severity", 0.0) or 0.0), 1.0 if wasteful else 0.0)

        if wasteful and end_turn_indices:
            end_turn_penalty = min(5.5, 2.25 + 0.70 * severity + 0.25 * min(energy, 4.0))
            positive_bonus = min(1.00, 0.25 + 0.14 * severity)
            setup_bonus = 0.15 if setup_indices else 0.0
            for idx in end_turn_indices:
                if 0 <= idx < MAX_ACTIONS and idx < mask_np.shape[0] and mask_np[idx] > 0:
                    bias[idx] -= end_turn_penalty
            for idx in boost_indices:
                if 0 <= idx < MAX_ACTIONS and idx < mask_np.shape[0] and mask_np[idx] > 0:
                    bias[idx] += positive_bonus + (setup_bonus if idx in setup_indices else 0.0)
        else:
            end_turn_penalty = 0.0

        if zero_energy_x_indices:
            for idx in zero_energy_x_indices:
                if 0 <= idx < MAX_ACTIONS and idx < mask_np.shape[0] and mask_np[idx] > 0:
                    bias[idx] -= 4.5

        for idx in typed_followup_missing_indices:
            if 0 <= idx < MAX_ACTIONS and idx < mask_np.shape[0] and mask_np[idx] > 0:
                # Production/Bloodletting/Bullet-Time-like cards are legal but
                # dominated when the current hand cannot convert the generated
                # energy/cost rule/no-draw tradeoff this turn.  Keep this softer
                # than zero-energy X, because sometimes retaining/setting up a
                # card-state mutation is still a legitimate long-horizon choice.
                bias[idx] -= 0.65
        for idx in typed_future_penalty_indices | typed_no_draw_indices:
            if idx not in setup_followup_available_indices and 0 <= idx < MAX_ACTIONS and idx < mask_np.shape[0] and mask_np[idx] > 0:
                bias[idx] -= 0.25

        for idx in potion_urgent_indices:
            if 0 <= idx < MAX_ACTIONS and idx < mask_np.shape[0] and mask_np[idx] > 0:
                bias[idx] += 0.85
        for idx in potion_lethal_indices | potion_prevent_lethal_indices:
            if 0 <= idx < MAX_ACTIONS and idx < mask_np.shape[0] and mask_np[idx] > 0:
                bias[idx] += 1.15
        for idx in potion_mechanism_indices:
            if 0 <= idx < MAX_ACTIONS and idx < mask_np.shape[0] and mask_np[idx] > 0:
                bias[idx] += 0.90
        for idx in potion_low_urgency_indices:
            if 0 <= idx < MAX_ACTIONS and idx < mask_np.shape[0] and mask_np[idx] > 0:
                bias[idx] -= 0.55
        for idx in potion_save_recommended_indices:
            if 0 <= idx < MAX_ACTIONS and idx < mask_np.shape[0] and mask_np[idx] > 0:
                bias[idx] -= 0.65
        for idx in potion_no_followup_indices:
            if 0 <= idx < MAX_ACTIONS and idx < mask_np.shape[0] and mask_np[idx] > 0:
                bias[idx] -= 0.85
        for idx in potion_overkill_indices:
            if 0 <= idx < MAX_ACTIONS and idx < mask_np.shape[0] and mask_np[idx] > 0:
                bias[idx] -= 0.45
        for idx in potion_block_waste_indices:
            if 0 <= idx < MAX_ACTIONS and idx < mask_np.shape[0] and mask_np[idx] > 0:
                bias[idx] -= 0.45

        kaiser_back_attack_risk = 0.0
        ceremonial_one_card_lock = 0.0
        ceremonial_stun_window = 0.0
        ceremonial_low_impact_count = 0
        ceremonial_high_impact_count = 0
        kaiser_defense_candidate_count = 0
        kaiser_facing_change_candidate_count = 0
        kaiser_pressure_candidate_count = 0
        encounter = ""
        if isinstance(raw_obs, dict):
            try:
                boss_ctx = build_boss_mechanics_context(raw_obs)
                encounter = str(boss_ctx.get("encounter_key") or "").lower()
                kaiser_back_attack_risk = max(
                    self._boss_context_max(boss_ctx, "back_attack_risk"),
                    self._boss_context_max(boss_ctx, "back_attack_active"),
                    self._boss_context_max(boss_ctx, "incoming_damage_multiplier_norm"),
                )
                ceremonial_one_card_lock = self._boss_context_max(boss_ctx, "one_card_lock")
                ceremonial_stun_window = self._boss_context_max(boss_ctx, "stun_window")
            except Exception:
                encounter = ""

        if kaiser_back_attack_risk > 0.05 and ("kaiser" in encounter or not encounter):
            risk = min(1.0, float(kaiser_back_attack_risk))
            for idx in range(legal_count):
                if mask_np[idx] <= 0:
                    continue
                action = legal_actions[idx]
                family = self._semantic_family(action)
                roles = self._action_roles(action)
                block = self._action_metric(action, "block")
                if family == "end_turn":
                    bias[idx] -= 1.1 * risk
                elif self._is_kaiser_facing_change_action(action, raw_obs):
                    # Surrounded facing changes implicitly by resolving any targeted
                    # card/potion toward an enemy on the opposite side.  This is not a
                    # guessed "turn around" action-name field: it is derived from
                    # combat.facing + target_combat_id and the target enemy's
                    # BACK_ATTACK_LEFT/RIGHT_POWER marker.
                    bias[idx] += 1.10 * risk
                    kaiser_facing_change_candidate_count += 1
                    kaiser_defense_candidate_count += 1
                elif self._is_kaiser_risk_handling_action(action, raw_obs):
                    bias[idx] += 0.55 * risk
                    kaiser_defense_candidate_count += 1
                elif self._is_kaiser_pressure_action(action):
                    # Strike/Bash-only hands are not "defense" candidates, but they are
                    # useful to diagnose snapshot composition and can still be correct if
                    # they kill or push a phase before the back attack lands.
                    bias[idx] += 0.15 * risk
                    kaiser_pressure_candidate_count += 1
            if kaiser_facing_change_candidate_count == 0:
                self._dump_kaiser_facing_diagnostic(raw_obs, legal_actions, mask_np, encounter, risk)

        if ceremonial_one_card_lock > 0.05 and ("ceremonial" in encounter or not encounter):
            for idx in range(legal_count):
                if mask_np[idx] <= 0:
                    continue
                action = legal_actions[idx]
                family = self._semantic_family(action)
                impact = self._action_immediate_impact(action)
                roles = self._action_roles(action)
                if family == "end_turn":
                    bias[idx] -= 1.25
                elif family in {"use_potion", "potion"}:
                    low_impact, high_impact = self._ceremonial_action_timing_flags(
                        action,
                        idx,
                        obs,
                        raw_obs,
                        legal_actions,
                        mask_np,
                        energy,
                    )
                    if high_impact:
                        bias[idx] += 0.65 + (0.30 if ceremonial_stun_window > 0.05 else 0.0)
                        ceremonial_high_impact_count += 1
                    elif low_impact:
                        bias[idx] -= 0.85
                        ceremonial_low_impact_count += 1
                elif family == "play_card":
                    if impact >= 12.0:
                        bias[idx] += 0.75 + (0.35 if ceremonial_stun_window > 0.05 else 0.0)
                        ceremonial_high_impact_count += 1
                    elif impact <= 2.0 and not roles.intersection({"draw", "energy", "scaling", "power"}):
                        bias[idx] -= 0.95
                        ceremonial_low_impact_count += 1

        legal_mask_fixed = np.zeros(MAX_ACTIONS, dtype=bool)
        valid_len = min(mask_np.shape[0], MAX_ACTIONS)
        if valid_len > 0:
            legal_mask_fixed[:valid_len] = mask_np[:valid_len] > 0
        applied_mask = legal_mask_fixed & (np.abs(bias[:MAX_ACTIONS]) > 1e-6)
        stats = {
            "combat_quality_bias_applied": 1.0 if bool(applied_mask.any()) else 0.0,
            "combat_quality_bias_abs_mean": float(np.mean(np.abs(bias[:MAX_ACTIONS][legal_mask_fixed]))) if bool(legal_mask_fixed.any()) else 0.0,
            # State-level: an end_turn candidate exists while positive progress is still available.
            # This is an availability/bias-applied signal, not necessarily the selected action.
            "combat_quality_wasteful_end_turn_bias_applied": 1.0 if wasteful and bool(end_turn_indices) else 0.0,
            "combat_quality_wasteful_end_turn_available": 1.0 if wasteful and bool(end_turn_indices) else 0.0,
            "combat_quality_end_turn_penalty_max": float(end_turn_penalty),
            "combat_quality_energy": float(energy),
            "combat_quality_positive_action_count": float(len(positive_indices)),
            "combat_quality_urgent_positive_action_count": float(len(urgent_positive_indices)),
            "combat_quality_deferable_positive_action_count": float(len(deferable_positive_indices)),
            "combat_quality_deferable_exhaust_card_count": float(len(deferable_exhaust_indices)),
            "combat_quality_energy_gain_without_followup_count": float(len(energy_gain_without_followup_indices)),
            "combat_quality_typed_followup_missing_count": float(len(typed_followup_missing_indices)),
            "combat_quality_typed_future_penalty_count": float(len(typed_future_penalty_indices)),
            "combat_quality_typed_no_draw_count": float(len(typed_no_draw_indices)),
            "combat_quality_typed_card_state_mutation_count": float(len(typed_card_state_mutation_indices)),
            "combat_quality_setup_followup_dependent_count": float(len(setup_followup_dependent_indices)),
            "combat_quality_setup_followup_available_count": float(len(setup_followup_available_indices)),
            "combat_quality_potion_available_count": float(len(potion_available_indices)),
            "combat_quality_potion_urgent_count": float(len(potion_urgent_indices)),
            "combat_quality_potion_low_urgency_count": float(len(potion_low_urgency_indices)),
            "combat_quality_potion_save_recommended_count": float(len(potion_save_recommended_indices)),
            "combat_quality_potion_no_followup_count": float(len(potion_no_followup_indices)),
            "combat_quality_potion_lethal_count": float(len(potion_lethal_indices)),
            "combat_quality_potion_prevent_lethal_count": float(len(potion_prevent_lethal_indices)),
            "combat_quality_potion_mechanism_count": float(len(potion_mechanism_indices)),
            "combat_quality_potion_overkill_count": float(len(potion_overkill_indices)),
            "combat_quality_potion_block_waste_count": float(len(potion_block_waste_indices)),
            "combat_quality_potion_use_quality_mean": float(context.get("potion_use_quality_mean", 0.0) or 0.0),
            "combat_quality_potion_waste_risk_mean": float(context.get("potion_waste_risk_mean", 0.0) or 0.0),
            "combat_quality_potion_save_value_mean": float(context.get("potion_save_value_mean", 0.0) or 0.0),
            "combat_quality_potion_hand_context_good_count": float(context.get("potion_hand_context_good_count", 0) or 0),
            "combat_quality_potion_hand_context_bad_count": float(context.get("potion_hand_context_bad_count", 0) or 0),
            "combat_quality_potion_long_term_count": float(context.get("potion_long_term_count", 0) or 0),
            "combat_quality_potion_requires_followup_count": float(context.get("potion_requires_followup_count", 0) or 0),
            "combat_quality_strategic_defer_available": 1.0 if bool(context.get("strategic_defer_available", False)) and bool(end_turn_indices) else 0.0,
            "combat_quality_true_wasteful_end_turn_available": 1.0 if wasteful and bool(end_turn_indices) else 0.0,
            "combat_quality_bad_end_turn_available": 1.0 if (wasteful or (
                bool(end_turn_indices)
                and (kaiser_back_attack_risk > 0.05 and kaiser_facing_change_candidate_count >= 1)
            )) else 0.0,
            "combat_quality_forced_end_turn_available": 1.0 if (
                bool(end_turn_indices) and not wasteful and len(positive_indices) == 0
            ) else 0.0,
            "combat_quality_playable_action_count": float(len(playable_indices)),
            "combat_quality_end_turn_severity": float(severity),
            "combat_quality_x_cost_available_count": float(sum(
                1
                for _idx in range(legal_count)
                if mask_np[_idx] > 0 and self._semantic_family(legal_actions[_idx]) == "play_card" and self._is_x_cost_action(obs, _idx, legal_actions[_idx])
            )),
            "combat_quality_zero_energy_x_cost_count": float(len(zero_energy_x_indices)),
            "combat_quality_x_cost_bad_count": float(len(x_cost_bad_indices)),
            "combat_quality_x_cost_effective_energy_sum": float(x_cost_effective_energy_sum),
            "combat_quality_x_cost_effective_energy_mean": (
                float(x_cost_effective_energy_sum / max(len(x_cost_indices), 1))
                if x_cost_indices
                else 0.0
            ),
            "combat_quality_kaiser_back_attack_risk": float(kaiser_back_attack_risk),
            "combat_quality_kaiser_defense_candidate_count": float(kaiser_defense_candidate_count),
            "combat_quality_kaiser_facing_change_candidate_count": float(kaiser_facing_change_candidate_count),
            "combat_quality_kaiser_pressure_candidate_count": float(kaiser_pressure_candidate_count),
            "combat_quality_ceremonial_one_card_lock": float(ceremonial_one_card_lock),
            "combat_quality_ceremonial_stun_window": float(ceremonial_stun_window),
            "combat_quality_ceremonial_low_impact_count": float(ceremonial_low_impact_count),
            "combat_quality_ceremonial_high_impact_count": float(ceremonial_high_impact_count),
        }
        return bias, stats, zero_energy_x_indices

    def _ceremonial_action_timing_flags(
        self,
        action: Any,
        action_idx: int,
        obs: dict[str, Any] | None,
        raw_obs: dict[str, Any] | None,
        legal_actions: list[Any] | None,
        mask_np: np.ndarray,
        energy: float,
    ) -> tuple[bool, bool]:
        """Classify one-card-lock actions as low/high timing quality.

        Ceremonial Beast's one-card-lock punishes spending the single allowed
        card/action on a low-impact move.  Potion actions must use the same
        timing-aware profile as the direct planner; otherwise diagnostics would
        regress to the old "potion immediate impact" view and again label almost
        every potion as a reasonable one-card-lock spend.

        Returns ``(low_impact, high_impact)``.
        """

        family = self._semantic_family(action)
        impact = self._action_immediate_impact(action)
        if family in {"use_potion", "potion"}:
            profile = self._potion_timing_profile(action, action_idx, obs, raw_obs, legal_actions, mask_np, energy)
            use_quality = float(profile.get("use_quality", 0.0) or 0.0)
            high_impact = bool(profile.get("urgent", False) or use_quality >= 0.55)
            low_impact = bool(
                profile.get("low_urgency", False)
                or profile.get("save_recommended", False)
                or profile.get("no_followup", False)
                or profile.get("block_waste", False)
                or (use_quality < 0.25 and impact <= 2.0)
            )
            return low_impact, high_impact

        if family == "play_card":
            roles = self._action_roles(action)
            high_impact = bool(impact >= 12.0)
            low_impact = bool(impact <= 2.0 and not roles.intersection({"draw", "energy", "scaling", "power"}))
            return low_impact, high_impact

        return False, False

    def _selected_combat_quality_stats(
        self,
        obs: dict[str, Any] | None,
        action_idx: int,
        legal_actions: list[Any] | None,
        search_stats: dict[str, Any] | None,
    ) -> dict[str, float]:
        """Selection-side combat diagnostics that do not depend on planner mode.

        MCTS, direct-policy, and direct-rollout all eventually choose one legal action.
        These metrics classify that chosen action against the *current* raw combat energy
        and the already-computed root availability flags, so zero-energy X-cost and
        strategic-defer/true-wasteful rates have one consistent definition.
        """
        if not isinstance(legal_actions, list) or action_idx < 0 or action_idx >= len(legal_actions):
            return {}
        action = legal_actions[action_idx]
        stats = search_stats if isinstance(search_stats, dict) else {}
        raw_obs = self._current_raw_combat_obs()
        energy = self._combat_energy(obs, raw_obs)
        family = self._semantic_family(action)
        end_turn_selected = family == "end_turn"
        x_selected = bool(family == "play_card" and self._is_x_cost_action(obs, action_idx, action))
        zero_x_selected = bool(x_selected and energy <= 0.05)
        x_diag = self._x_cost_diagnostic(action, float(energy)) if x_selected else {}
        x_has_non_energy = float(x_diag.get("x_cost_has_non_energy_effect", 0.0) or 0.0) > 0.5
        x_bad_selected = bool(x_selected and float(x_diag.get("x_cost_bad", 0.0) or 0.0) > 0.5)
        true_waste_available = float(stats.get("combat_quality_true_wasteful_end_turn_available", stats.get("combat_quality_wasteful_end_turn_available", 0.0)) or 0.0) > 0.5
        strategic_defer_available = float(stats.get("combat_quality_strategic_defer_available", 0.0) or 0.0) > 0.5
        potion_selected = family in {"use_potion", "potion"}
        potion_available = float(stats.get("combat_quality_potion_available_count", 0.0) or 0.0) > 0.0
        potion_profile: dict[str, Any] = {}
        if potion_selected:
            mask_np = np.ones(MAX_ACTIONS, dtype=np.float32)
            potion_profile = self._potion_timing_profile(action, action_idx, obs, raw_obs, legal_actions, mask_np, energy)
        # TASK-B1: re-classify the selected end_turn through the central
        # taxonomy so the selected-side rate uses the same priority as the
        # bias-side context.  Reconstruct a minimal context dict from the
        # already-computed bias stats — avoids re-walking legal_actions and
        # keeps the taxonomy frame-consistent with the bias judgment.
        end_turn_class = "unknown"
        end_turn_class_flags: dict[str, bool] = {}
        if end_turn_selected:
            taxonomy_context = {
                "end_turn_indices": [int(action_idx)],
                "wasteful": bool(true_waste_available),
                "strategic_defer_available": bool(strategic_defer_available),
                "positive_progress_count": int(float(stats.get("combat_quality_positive_action_count", 0.0) or 0.0)),
                "urgent_positive_count": int(float(stats.get("combat_quality_urgent_positive_action_count", 0.0) or 0.0)),
                "deferable_positive_count": int(float(stats.get("combat_quality_deferable_positive_action_count", 0.0) or 0.0)),
                "energy": float(stats.get("combat_quality_energy", energy) or energy),
            }
            boss_signals = {
                "kaiser_back_attack_risk": float(stats.get("combat_quality_kaiser_back_attack_risk", 0.0) or 0.0),
                "kaiser_facing_change_candidate_count": float(stats.get("combat_quality_kaiser_facing_change_candidate_count", 0.0) or 0.0),
                "kaiser_defense_candidate_count": float(stats.get("combat_quality_kaiser_defense_candidate_count", 0.0) or 0.0),
                "ceremonial_stun_window": float(stats.get("combat_quality_ceremonial_stun_window", 0.0) or 0.0),
                "ceremonial_high_impact_count": float(stats.get("combat_quality_ceremonial_high_impact_count", 0.0) or 0.0),
            }
            end_turn_class, end_turn_class_flags = self._classify_end_turn_action(
                taxonomy_context,
                None,
                boss_signals,
            )
        bad_selected = end_turn_selected and end_turn_class == "bad_end_turn"
        forced_selected = end_turn_selected and end_turn_class == "forced_end_turn"
        defer_selected_taxonomy = end_turn_selected and end_turn_class == "strategic_defer_end_turn"
        unknown_selected = end_turn_selected and end_turn_class == "unknown"
        result = {
            "combat_quality_x_cost_selected": 1.0 if x_selected else 0.0,
            "combat_quality_x_cost_selected_energy": float(energy) if x_selected else 0.0,
            "combat_quality_x_cost_zero_energy_selected": 1.0 if zero_x_selected else 0.0,
            "combat_quality_zero_energy_x_cost_selected": 1.0 if zero_x_selected else 0.0,
            "combat_quality_x_cost_selected_effective_energy": float(x_diag.get("x_cost_effective_energy", 0.0) or 0.0) if x_selected else 0.0,
            "combat_quality_x_cost_has_non_energy_effect_selected": 1.0 if (x_selected and x_has_non_energy) else 0.0,
            "combat_quality_x_cost_bad_selected": 1.0 if x_bad_selected else 0.0,
            "combat_quality_end_turn_selected": 1.0 if end_turn_selected else 0.0,
            # Legacy alias kept so existing dashboards stay readable.  The
            # taxonomy view (bad/forced/strategic_defer) is the new source of
            # truth — bad_selected is wider than wasteful because it folds in
            # boss-pressure windows.
            "combat_quality_true_wasteful_end_turn_selected": 1.0 if (end_turn_selected and true_waste_available) else 0.0,
            "combat_quality_wasteful_end_turn_selected": 1.0 if bad_selected else (1.0 if (end_turn_selected and true_waste_available) else 0.0),
            "combat_quality_bad_end_turn_selected": 1.0 if bad_selected else 0.0,
            "combat_quality_forced_end_turn_selected": 1.0 if forced_selected else 0.0,
            "combat_quality_strategic_defer_end_turn_selected": 1.0 if (defer_selected_taxonomy or (end_turn_selected and strategic_defer_available and not bad_selected)) else 0.0,
            "combat_quality_end_turn_unknown_selected": 1.0 if unknown_selected else 0.0,
            "combat_quality_end_turn_class": end_turn_class,
        }
        result.update(
            {
                "combat_quality_potion_selected": 1.0 if potion_selected else 0.0,
                "combat_quality_potion_selected_when_available": 1.0 if (potion_selected and potion_available) else 0.0,
                "combat_quality_potion_high_urgency_selected": 1.0 if (potion_selected and bool(potion_profile.get("urgent", False))) else 0.0,
                "combat_quality_potion_low_urgency_selected": 1.0 if (potion_selected and bool(potion_profile.get("low_urgency", False))) else 0.0,
                "combat_quality_potion_save_recommended_selected": 1.0 if (potion_selected and bool(potion_profile.get("save_recommended", False))) else 0.0,
                "combat_quality_potion_no_followup_selected": 1.0 if (potion_selected and bool(potion_profile.get("no_followup", False))) else 0.0,
                "combat_quality_potion_lethal_selected": 1.0 if (potion_selected and bool(potion_profile.get("lethal", False))) else 0.0,
                "combat_quality_potion_prevent_lethal_selected": 1.0 if (potion_selected and bool(potion_profile.get("prevent_lethal", False))) else 0.0,
                "combat_quality_potion_mechanism_selected": 1.0 if (potion_selected and bool(potion_profile.get("mechanism_answer", False))) else 0.0,
                "combat_quality_potion_overkill_selected": 1.0 if (potion_selected and bool(potion_profile.get("overkill", False))) else 0.0,
                "combat_quality_potion_block_waste_selected": 1.0 if (potion_selected and bool(potion_profile.get("block_waste", False))) else 0.0,
                "combat_quality_potion_use_quality_selected": float(potion_profile.get("use_quality", 0.0) or 0.0) if potion_selected else 0.0,
                "combat_quality_potion_waste_risk_selected": float(potion_profile.get("waste_risk", 0.0) or 0.0) if potion_selected else 0.0,
            }
        )
        return result

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
            "end_turn_guard_applied",
            "end_turn_guard_forced_alternative",
            "zero_energy_x_cost_guard_applied",
            "zero_energy_x_cost_guard_forced_alternative",
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
            "search_mode_direct_policy",
            "search_mode_direct_rollout_planner",
            "direct_rollout_q_mean",
            "direct_rollout_objective_q_mean",
            "direct_rollout_risk_q_mean",
            "direct_rollout_uncertainty_mean",
            "direct_rollout_uncertainty_bias_abs_mean",
            "direct_rollout_surprise_mean",
            "direct_rollout_surface_entropy_mean",
            "direct_rollout_latent_drift_mean",
            "direct_rollout_branch_disagreement_mean",
            "direct_rollout_steps_used",
            "direct_rollout_branch_count_mean",
            "direct_rollout_root_valid_count",
            "direct_rollout_root_bucket_size",
            "direct_rollout_bucket_padding_ratio",
            "direct_rollout_max_branch_bucket_size",
            "direct_rollout_branch_padding_ratio",
            "combat_quality_bias_applied",
            "combat_quality_bias_abs_mean",
            "combat_quality_wasteful_end_turn_bias_applied",
            "combat_quality_wasteful_end_turn_available",
            "combat_quality_wasteful_end_turn_selected",
            "combat_quality_true_wasteful_end_turn_available",
            "combat_quality_true_wasteful_end_turn_selected",
            "combat_quality_strategic_defer_available",
            "combat_quality_strategic_defer_end_turn_selected",
            "combat_quality_end_turn_penalty_max",
            "combat_quality_energy",
            "combat_quality_positive_action_count",
            "combat_quality_mandatory_positive_action_count",
            "combat_quality_strategic_skip_candidate_count",
            "combat_quality_refund_no_followup_available",
            "combat_quality_refund_no_followup_selected",
            "combat_quality_strategic_skip_selected",
            "combat_quality_enchantment_seen",
            "combat_quality_affliction_seen",
            "combat_quality_urgent_positive_action_count",
            "combat_quality_deferable_positive_action_count",
            "combat_quality_deferable_exhaust_card_count",
            "combat_quality_energy_gain_without_followup_count",
            "combat_quality_typed_followup_missing_count",
            "combat_quality_typed_future_penalty_count",
            "combat_quality_typed_no_draw_count",
            "combat_quality_typed_card_state_mutation_count",
            "combat_quality_setup_followup_dependent_count",
            "combat_quality_setup_followup_available_count",
            "combat_quality_potion_available_count",
            "combat_quality_potion_urgent_count",
            "combat_quality_potion_low_urgency_count",
            "combat_quality_potion_save_recommended_count",
            "combat_quality_potion_no_followup_count",
            "combat_quality_potion_lethal_count",
            "combat_quality_potion_prevent_lethal_count",
            "combat_quality_potion_mechanism_count",
            "combat_quality_potion_overkill_count",
            "combat_quality_potion_block_waste_count",
            "combat_quality_potion_use_quality_mean",
            "combat_quality_potion_waste_risk_mean",
            "combat_quality_potion_save_value_mean",
            "combat_quality_potion_hand_context_good_count",
            "combat_quality_potion_hand_context_bad_count",
            "combat_quality_potion_long_term_count",
            "combat_quality_potion_requires_followup_count",
            "combat_quality_potion_selected",
            "combat_quality_potion_selected_when_available",
            "combat_quality_potion_high_urgency_selected",
            "combat_quality_potion_low_urgency_selected",
            "combat_quality_potion_save_recommended_selected",
            "combat_quality_potion_no_followup_selected",
            "combat_quality_potion_lethal_selected",
            "combat_quality_potion_prevent_lethal_selected",
            "combat_quality_potion_mechanism_selected",
            "combat_quality_potion_overkill_selected",
            "combat_quality_potion_block_waste_selected",
            "combat_quality_potion_use_quality_selected",
            "combat_quality_potion_waste_risk_selected",
            "combat_quality_kaiser_back_attack_risk",
            "combat_quality_kaiser_defense_candidate_count",
            "combat_quality_kaiser_facing_change_candidate_count",
            "combat_quality_kaiser_pressure_candidate_count",
            "combat_quality_kaiser_facing_change_selected",
            "combat_quality_kaiser_pressure_selected",
            "combat_quality_kaiser_risky_end_turn_selected",
            "combat_quality_ceremonial_one_card_lock",
            "combat_quality_ceremonial_stun_window",
            "combat_quality_ceremonial_low_impact_count",
            "combat_quality_ceremonial_high_impact_count",
            "combat_quality_kaiser_defense_selected",
            "combat_quality_ceremonial_low_impact_selected",
            "combat_quality_ceremonial_high_impact_selected",
            "combat_quality_playable_action_count",
            "combat_quality_end_turn_severity",
            "combat_quality_x_cost_available_count",
            "combat_quality_x_cost_selected",
            "combat_quality_x_cost_selected_energy",
            "combat_quality_x_cost_zero_energy_selected",
            "combat_quality_zero_energy_x_cost_count",
            "combat_quality_zero_energy_x_cost_selected",
            "combat_quality_x_cost_bad_count",
            "combat_quality_x_cost_effective_energy_sum",
            "combat_quality_x_cost_effective_energy_mean",
            "combat_quality_x_cost_selected_effective_energy",
            "combat_quality_x_cost_has_non_energy_effect_selected",
            "combat_quality_x_cost_bad_selected",
            "combat_quality_bad_end_turn_available",
            "combat_quality_bad_end_turn_selected",
            "combat_quality_forced_end_turn_available",
            "combat_quality_forced_end_turn_selected",
            "combat_quality_end_turn_unknown_selected",
            "combat_quality_end_turn_selected",
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
    def _normalize_action_values(values: torch.Tensor, action_mask: torch.Tensor) -> torch.Tensor:
        action_mask = action_mask.to(device=values.device, dtype=torch.bool)
        masked_values = torch.where(action_mask, values, torch.zeros_like(values))
        valid_count = action_mask.float().sum(dim=-1, keepdim=True).clamp(min=1.0)
        mean = masked_values.sum(dim=-1, keepdim=True) / valid_count
        centered = torch.where(action_mask, values - mean, torch.zeros_like(values))
        variance = centered.pow(2).sum(dim=-1, keepdim=True) / valid_count
        std = variance.clamp(min=1e-6).sqrt()
        normalized = centered / std
        return torch.where(action_mask, normalized, torch.zeros_like(normalized))

    def record_recent_combat_episode(self, metadata: dict[str, Any] | None) -> dict[int, dict[str, Any]]:
        snapshot = self.recent_combat_monitor.record_episode(metadata)
        if snapshot:
            self.recent_combat_monitor.log_to_writer(self.writer, self.episode_count)
        return snapshot

    def _num_simulations_for_domain(self, decision_domain: str) -> int:
        return int(self.domain_num_simulations.get(decision_domain, self.mcts.num_simulations))

    def _latent_regularization_loss(
        self,
        hidden_state: torch.Tensor,
        *,
        dynamics: bool = False,
    ) -> tuple[torch.Tensor, dict[str, float]]:
        if self.latent_gaussian_reg_weight <= 0.0:
            return hidden_state.new_zeros(()), {
                "mean_abs": 0.0,
                "var_mean": 0.0,
                "var_std": 0.0,
                "cov_offdiag": 0.0,
                "proj_var_mean": 0.0,
                "slot_var_mean": 0.0,
            }
        base_loss, base_metrics = latent_gaussian_regularizer(
            hidden_state,
            projection_count=self.latent_gaussian_reg_projections,
            cov_weight=self.latent_gaussian_reg_cov_weight,
        )
        total = base_loss
        slot_var_mean = 0.0
        if getattr(self.network, "is_token_mode", False) and self.latent_gaussian_reg_slot_weight > 0.0:
            prediction = getattr(self.network, "prediction", None)
            slot_count = int(getattr(prediction, "num_memory_slots", 0) or 0)
            d_model = int(getattr(prediction, "d_model", 0) or 0)
            if slot_count > 0 and d_model > 0 and hidden_state.shape[-1] == slot_count * d_model:
                slots = hidden_state.reshape(hidden_state.shape[0], slot_count, d_model)
                slot_loss, slot_metrics = slot_latent_gaussian_regularizer(
                    slots,
                    projection_count=self.latent_gaussian_reg_projections,
                    cov_weight=self.latent_gaussian_reg_cov_weight,
                )
                total = total + self.latent_gaussian_reg_slot_weight * slot_loss
                slot_var_mean = slot_metrics.var_mean
        if dynamics:
            total = self.latent_gaussian_reg_dynamics_weight * total
        return total, {
            "mean_abs": base_metrics.mean_abs,
            "var_mean": base_metrics.var_mean,
            "var_std": base_metrics.var_std,
            "cov_offdiag": base_metrics.cov_offdiag,
            "proj_var_mean": base_metrics.proj_var_mean,
            "slot_var_mean": slot_var_mean,
        }

    def _hidden_surprise_target(
        self,
        student_state: torch.Tensor,
        teacher_state: torch.Tensor,
    ) -> torch.Tensor:
        student = torch.nn.functional.normalize(student_state.detach(), dim=-1)
        teacher = torch.nn.functional.normalize(teacher_state.detach(), dim=-1)
        mse = torch.nn.functional.mse_loss(student, teacher, reduction="none").mean(dim=-1)
        cosine_gap = 1.0 - (student * teacher).sum(dim=-1)
        return (mse + cosine_gap.clamp(min=0.0)) * self.surprise_hidden_scale

    def _augment_surprise_target(
        self,
        surprise_target: torch.Tensor,
        *,
        future_world_aux_loss: torch.Tensor | None = None,
        surface_mask_loss: torch.Tensor | None = None,
        surface_count_loss: torch.Tensor | None = None,
        surface_domain_loss: torch.Tensor | None = None,
        surface_phase_loss: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, float]:
        """Fold world/surface prediction miss into the dynamics surprise target.

        The hidden JEPA error stays per-sample.  Future-bank and surface losses
        are batch-mean auxiliary errors, so they become a detached global offset
        on that step.  This keeps the surprise head aligned with the scalar
        uncertainty later consumed by the search-free latent rollout planner.
        """

        target = surprise_target.detach()
        offset = target.new_zeros(())
        if self.surprise_future_aux_scale > 0.0 and future_world_aux_loss is not None:
            offset = offset + self.surprise_future_aux_scale * future_world_aux_loss.detach().to(target.device).float()
        if self.surprise_surface_scale > 0.0:
            surface_offset = target.new_zeros(())
            for maybe_loss in (
                surface_mask_loss,
                surface_count_loss,
                surface_domain_loss,
                surface_phase_loss,
            ):
                if maybe_loss is not None:
                    surface_offset = surface_offset + maybe_loss.detach().to(target.device).float()
            offset = offset + self.surprise_surface_scale * surface_offset
        if offset.ndim > 0:
            offset = offset.mean()
        target = target + offset
        if self.surprise_target_cap > 0.0:
            target = target.clamp(max=self.surprise_target_cap)
        return target, float(offset.detach().item())

    def _surprise_loss(
        self,
        predicted_surprise: torch.Tensor,
        surprise_target: torch.Tensor,
    ) -> tuple[torch.Tensor, float, float, float]:
        pred = predicted_surprise.reshape(-1).float()
        target = surprise_target.detach().reshape(-1).float().to(pred.device)
        if pred.numel() == 0:
            return predicted_surprise.new_zeros(()), 0.0, 0.0, 0.0
        loss = torch.nn.functional.smooth_l1_loss(torch.log1p(pred.clamp(min=0.0)), torch.log1p(target.clamp(min=0.0)))
        mae = (pred.detach() - target).abs().mean().item()
        return loss, float(target.mean().item()), float(pred.detach().mean().item()), float(mae)

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
    def _safe_float(value: Any, default: float = 0.0) -> float:
        try:
            return float(value)
        except (TypeError, ValueError):
            return float(default)

    @staticmethod
    def _safe_rate(numerator: float, denominator: float) -> float:
        return float(numerator) / float(denominator) if float(denominator) > 0.0 else 0.0

    @staticmethod
    def _potion_count_from_info(info: dict[str, Any] | None) -> int:
        transition_state = info.get("transition_state") if isinstance(info, dict) else None
        player = transition_state.get("player") if isinstance(transition_state, dict) else None
        potions = player.get("potions") if isinstance(player, dict) else None
        if not isinstance(potions, list):
            return 0
        count = 0
        # IMPORTANT: STS2 bridge serializes empty potion slots as
        # {title:"[empty]", id:null, ...} — note the BRACKETS.  An earlier
        # exclusion set of {"empty","none","null"} did NOT match the bracket
        # form, so every empty slot was being counted as a usable potion.
        # That made `boss_combat/<boss>/potion_unused_on_death_rate` track
        # the boss loss rate (70-85%) instead of actual leftover potions, and
        # repeatedly misled the curriculum diagnostic.
        EMPTY_TITLES = {"empty", "[empty]", "none", "null", ""}
        for potion in potions:
            if isinstance(potion, dict):
                if bool(potion.get("empty")):
                    continue
                # A used potion can remain in PotionSlots briefly as queued or
                # removed-from-state.  Do not count those as "unused on death".
                if bool(potion.get("is_queued")) or bool(potion.get("has_been_removed_from_state")):
                    continue
                if potion.get("is_usable") is False:
                    continue
                title = str(potion.get("title") or potion.get("id") or "").strip().lower()
            else:
                title = str(potion or "").strip().lower()
            if title and title not in EMPTY_TITLES:
                count += 1
        return count

    @classmethod
    def _boss_entry_snapshot_from_episode(
        cls,
        initial_info: dict[str, Any] | None,
        progress_snapshots: list[dict[str, Any]],
        *,
        encounter_tier: str,
    ) -> dict[str, float]:
        candidates: list[dict[str, Any]] = []
        if str(encounter_tier or "").strip().lower() == "boss":
            first = cls._episode_progress_snapshot(initial_info)
            if first:
                candidates.append(first)
        candidates.extend(
            snapshot for snapshot in progress_snapshots
            if "boss" in str(snapshot.get("room_type_lower") or "")
        )
        if not candidates:
            return {}
        entry = candidates[0]
        hp = cls._safe_float(entry.get("hp"))
        max_hp = cls._safe_float(entry.get("max_hp"))
        return {
            "hp": hp,
            "max_hp": max_hp,
            "hp_ratio": hp / max_hp if max_hp > 0.0 else 0.0,
        }

    @staticmethod
    def _step_family(step: dict[str, Any]) -> str:
        family = str(step.get("action_family") or "").strip().lower()
        if family:
            return family
        info = step.get("action_info") if isinstance(step.get("action_info"), dict) else {}
        semantic = info.get("semantic") if isinstance(info.get("semantic"), dict) else {}
        return str(semantic.get("family") or "").strip().lower()

    @staticmethod
    def _step_is_combat(step: dict[str, Any]) -> bool:
        domain = str(step.get("decision_domain") or step.get("semantic_domain") or "").strip().lower()
        family = MuZeroTrainer._step_family(step)
        phase = str(step.get("phase") or "").strip().lower()
        surface = str(step.get("surface") or "").strip().lower()
        return (
            domain == "combat"
            or phase in {"combat", "actions", "card_selection"}
            or surface == "combat"
            or family in {"play_card", "use_potion", "discard_potion", "end_turn", "combat_select"}
        )

    def _emit_boss_episode_diagnostics(
        self,
        trajectory: GameTrajectory,
        *,
        boss_entry: dict[str, float] | None,
        final_potion_count: int,
    ) -> dict[str, Any]:
        metadata = trajectory.metadata if isinstance(trajectory.metadata, dict) else {}
        encounter_id = str(metadata.get("encounter_id") or "").strip()
        encounter_tier = str(metadata.get("encounter_tier") or infer_encounter_tier(encounter_id)).strip().lower()
        boss_episode = encounter_tier == "boss" or int(metadata.get("boss_rooms_seen", 0) or 0) > 0
        if not boss_episode:
            return {}

        steps = [step for step in trajectory.steps if isinstance(step, dict) and self._step_is_combat(step)]
        total_steps = len(steps)
        family_counts: Counter[str] = Counter(self._step_family(step) or "unknown" for step in steps)
        end_turn_count = int(family_counts.get("end_turn", 0))
        use_potion_count = int(family_counts.get("use_potion", 0) + family_counts.get("discard_potion", 0))
        wasteful_end_turn_count = int(sum(1 for step in steps if bool(step.get("wasteful_end_turn"))))

        search_values: dict[str, list[float]] = defaultdict(list)
        for step in steps:
            search_stats = step.get("search_stats") if isinstance(step.get("search_stats"), dict) else {}
            for key in (
                "search_mode_direct_policy",
                "search_mode_direct_rollout_planner",
                "num_simulations",
                "direct_rollout_objective_q_mean",
                "direct_rollout_uncertainty_mean",
                "direct_rollout_uncertainty_bias_abs_mean",
                "direct_rollout_branch_disagreement_mean",
                "direct_rollout_risk_q_mean",
                "mean_predicted_legal_count",
                "combat_quality_bias_applied",
                "combat_quality_wasteful_end_turn_bias_applied",
                "combat_quality_wasteful_end_turn_available",
                "combat_quality_wasteful_end_turn_selected",
                "combat_quality_true_wasteful_end_turn_available",
                "combat_quality_true_wasteful_end_turn_selected",
                "combat_quality_strategic_defer_available",
                "combat_quality_strategic_defer_end_turn_selected",
                "combat_quality_urgent_positive_action_count",
                "combat_quality_deferable_positive_action_count",
                "combat_quality_deferable_exhaust_card_count",
                "combat_quality_energy_gain_without_followup_count",
                "combat_quality_typed_followup_missing_count",
                "combat_quality_typed_future_penalty_count",
                "combat_quality_typed_no_draw_count",
                "combat_quality_typed_card_state_mutation_count",
                "combat_quality_setup_followup_dependent_count",
                "combat_quality_setup_followup_available_count",
                "combat_quality_potion_available_count",
                "combat_quality_potion_urgent_count",
                "combat_quality_potion_low_urgency_count",
                "combat_quality_potion_save_recommended_count",
                "combat_quality_potion_no_followup_count",
                "combat_quality_potion_lethal_count",
                "combat_quality_potion_prevent_lethal_count",
                "combat_quality_potion_mechanism_count",
                "combat_quality_potion_overkill_count",
                "combat_quality_potion_block_waste_count",
                "combat_quality_potion_use_quality_mean",
                "combat_quality_potion_waste_risk_mean",
                "combat_quality_potion_save_value_mean",
                "combat_quality_potion_hand_context_good_count",
                "combat_quality_potion_hand_context_bad_count",
                "combat_quality_potion_long_term_count",
                "combat_quality_potion_requires_followup_count",
                "combat_quality_potion_selected",
                "combat_quality_potion_selected_when_available",
                "combat_quality_potion_high_urgency_selected",
                "combat_quality_potion_low_urgency_selected",
                "combat_quality_potion_save_recommended_selected",
                "combat_quality_potion_no_followup_selected",
                "combat_quality_potion_lethal_selected",
                "combat_quality_potion_prevent_lethal_selected",
                "combat_quality_potion_mechanism_selected",
                "combat_quality_potion_overkill_selected",
                "combat_quality_potion_block_waste_selected",
                "combat_quality_potion_use_quality_selected",
                "combat_quality_potion_waste_risk_selected",
                "combat_quality_x_cost_available_count",
                "combat_quality_x_cost_selected",
                "combat_quality_x_cost_selected_energy",
                "combat_quality_x_cost_zero_energy_selected",
                "combat_quality_zero_energy_x_cost_count",
                "combat_quality_zero_energy_x_cost_selected",
                "combat_quality_x_cost_bad_count",
                "combat_quality_x_cost_effective_energy_sum",
                "combat_quality_x_cost_effective_energy_mean",
                "combat_quality_x_cost_selected_effective_energy",
                "combat_quality_x_cost_has_non_energy_effect_selected",
                "combat_quality_x_cost_bad_selected",
                "combat_quality_bad_end_turn_available",
                "combat_quality_bad_end_turn_selected",
                "combat_quality_forced_end_turn_available",
                "combat_quality_forced_end_turn_selected",
                "combat_quality_end_turn_unknown_selected",
                "combat_quality_end_turn_selected",
                "combat_quality_energy",
                "combat_quality_positive_action_count",
                "combat_quality_mandatory_positive_action_count",
                "combat_quality_strategic_skip_candidate_count",
                "combat_quality_refund_no_followup_available",
                "combat_quality_refund_no_followup_selected",
                "combat_quality_strategic_skip_selected",
                "combat_quality_enchantment_seen",
                "combat_quality_affliction_seen",
                "combat_quality_kaiser_back_attack_risk",
                "combat_quality_kaiser_defense_candidate_count",
                "combat_quality_kaiser_facing_change_candidate_count",
                "combat_quality_kaiser_pressure_candidate_count",
                "combat_quality_kaiser_facing_change_selected",
                "combat_quality_kaiser_pressure_selected",
                "combat_quality_kaiser_risky_end_turn_selected",
                "combat_quality_ceremonial_one_card_lock",
                "combat_quality_ceremonial_stun_window",
                "combat_quality_ceremonial_low_impact_count",
                "combat_quality_ceremonial_high_impact_count",
                "combat_quality_kaiser_defense_selected",
                "combat_quality_ceremonial_low_impact_selected",
                "combat_quality_ceremonial_high_impact_selected",
            ):
                if key in search_stats:
                    search_values[key].append(self._safe_float(search_stats.get(key)))

        terminated = bool(metadata.get("terminated", False))
        truncated = bool(metadata.get("truncated", False))
        death_floor = self._safe_float(metadata.get("death_floor"))
        loss = (death_floor > 0.0) or (not terminated and not truncated)
        win = terminated and not truncated and death_floor <= 0.0
        entry = boss_entry if isinstance(boss_entry, dict) else {}
        # TASK-A4: subtract used-this-combat from raw final count so a use_potion
        # that succeeded but left a stale slot in the death frame does not falsely
        # inflate potion_unused_on_death.  Diagnostic JSONL captures the raw
        # transition so root-cause analysis can still see it.
        used_potion_count = int(metadata.get("used_potion_count_this_combat", 0) or 0)
        adjusted_unused = max(int(final_potion_count) - used_potion_count, 0)
        potion_unused_on_death = 1.0 if loss and adjusted_unused > 0 else 0.0
        try:
            self._dump_death_final_potions(
                loss=loss,
                final_potion_count=int(final_potion_count),
                used_potion_count=int(used_potion_count),
                adjusted_unused=int(adjusted_unused),
                encounter=encounter_id,
                tier=encounter_tier,
                potion_dump=metadata.get("final_potion_dump") if isinstance(metadata.get("final_potion_dump"), list) else None,
            )
        except Exception:
            pass

        metrics = {
            "boss/attempt_count": 1.0,
            "boss/win": 1.0 if win else 0.0,
            "boss/loss": 1.0 if loss else 0.0,
            "boss/reward": self._safe_float(metadata.get("episode_total_reward", metadata.get("episode_reward"))),
            "episode/boss_entry_hp": self._safe_float(entry.get("hp")),
            "episode/boss_entry_hp_ratio": self._safe_float(entry.get("hp_ratio")),
            "episode/boss_entry_potion_count": float(max(int(final_potion_count), 0)),
            "boss_combat/decision_count": float(total_steps),
            "boss_combat/family_end_turn_rate": self._safe_rate(end_turn_count, total_steps),
            "boss_combat/family_potion_rate": self._safe_rate(use_potion_count, total_steps),
            "boss_combat/wasteful_end_turn_rate": self._safe_rate(wasteful_end_turn_count, total_steps),
            "boss_combat/potion_unused_on_death_rate": potion_unused_on_death,
            "boss_combat/potion_unused_on_death_raw_rate": 1.0 if loss and int(final_potion_count) > 0 else 0.0,
            "boss_combat/used_potion_count_this_combat_mean": float(used_potion_count),
            "boss_combat/final_potion_count_mean": float(int(final_potion_count)),
            "boss_combat/playable_cards_left_mean": float(np.mean(search_values.get("mean_predicted_legal_count", [0.0]))),
            "boss_combat/direct_policy_used_rate": float(np.mean(search_values.get("search_mode_direct_policy", [0.0]))),
            "boss_combat/search_mode_direct_rollout_planner": float(np.mean(search_values.get("search_mode_direct_rollout_planner", [0.0]))),
            "boss_combat/num_simulations": float(np.mean(search_values.get("num_simulations", [0.0]))),
            "boss_combat/direct_rollout_objective_q_mean": float(np.mean(search_values.get("direct_rollout_objective_q_mean", [0.0]))),
            "boss_combat/direct_rollout_uncertainty_mean": float(np.mean(search_values.get("direct_rollout_uncertainty_mean", [0.0]))),
            "boss_combat/direct_rollout_uncertainty_bias_abs_mean": float(np.mean(search_values.get("direct_rollout_uncertainty_bias_abs_mean", [0.0]))),
            "boss_combat/direct_rollout_branch_disagreement_mean": float(np.mean(search_values.get("direct_rollout_branch_disagreement_mean", [0.0]))),
            "boss_combat/direct_rollout_risk_q_mean": float(np.mean(search_values.get("direct_rollout_risk_q_mean", [0.0]))),
            "boss_combat/quality_bias_applied_rate": float(np.mean(search_values.get("combat_quality_bias_applied", [0.0]))),
            "boss_combat/wasteful_end_turn_bias_applied_rate": float(np.mean(search_values.get("combat_quality_wasteful_end_turn_bias_applied", [0.0]))),
            "boss_combat/wasteful_end_turn_available_rate": float(np.mean(search_values.get("combat_quality_wasteful_end_turn_available", [0.0]))),
            "boss_combat/wasteful_end_turn_selected_rate": float(np.mean(search_values.get("combat_quality_wasteful_end_turn_selected", [0.0]))),
            "boss_combat/true_wasteful_end_turn_available_rate": float(np.mean(search_values.get("combat_quality_true_wasteful_end_turn_available", [0.0]))),
            "boss_combat/true_wasteful_end_turn_selected_rate": float(np.mean(search_values.get("combat_quality_true_wasteful_end_turn_selected", [0.0]))),
            "boss_combat/strategic_defer_available_rate": float(np.mean(search_values.get("combat_quality_strategic_defer_available", [0.0]))),
            "boss_combat/strategic_defer_end_turn_selected_rate": float(np.mean(search_values.get("combat_quality_strategic_defer_end_turn_selected", [0.0]))),
            "boss_combat/bad_end_turn_available_rate": float(np.mean(search_values.get("combat_quality_bad_end_turn_available", [0.0]))),
            "boss_combat/bad_end_turn_selected_rate": float(np.mean(search_values.get("combat_quality_bad_end_turn_selected", [0.0]))),
            "boss_combat/forced_end_turn_available_rate": float(np.mean(search_values.get("combat_quality_forced_end_turn_available", [0.0]))),
            "boss_combat/forced_end_turn_selected_rate": float(np.mean(search_values.get("combat_quality_forced_end_turn_selected", [0.0]))),
            "boss_combat/end_turn_unknown_selected_rate": float(np.mean(search_values.get("combat_quality_end_turn_unknown_selected", [0.0]))),
            "boss_combat/urgent_positive_action_count_mean": float(np.mean(search_values.get("combat_quality_urgent_positive_action_count", [0.0]))),
            "boss_combat/deferable_positive_action_count_mean": float(np.mean(search_values.get("combat_quality_deferable_positive_action_count", [0.0]))),
            "boss_combat/deferable_exhaust_card_count_mean": float(np.mean(search_values.get("combat_quality_deferable_exhaust_card_count", [0.0]))),
            "boss_combat/energy_gain_without_followup_count_mean": float(np.mean(search_values.get("combat_quality_energy_gain_without_followup_count", [0.0]))),
            "boss_combat/typed_followup_missing_count_mean": float(np.mean(search_values.get("combat_quality_typed_followup_missing_count", [0.0]))),
            "boss_combat/typed_future_penalty_count_mean": float(np.mean(search_values.get("combat_quality_typed_future_penalty_count", [0.0]))),
            "boss_combat/typed_no_draw_count_mean": float(np.mean(search_values.get("combat_quality_typed_no_draw_count", [0.0]))),
            "boss_combat/typed_card_state_mutation_count_mean": float(np.mean(search_values.get("combat_quality_typed_card_state_mutation_count", [0.0]))),
            "boss_combat/setup_followup_dependent_count_mean": float(np.mean(search_values.get("combat_quality_setup_followup_dependent_count", [0.0]))),
            "boss_combat/setup_followup_available_count_mean": float(np.mean(search_values.get("combat_quality_setup_followup_available_count", [0.0]))),
            "boss_combat/potion_available_count_mean": float(np.mean(search_values.get("combat_quality_potion_available_count", [0.0]))),
            "boss_combat/potion_urgent_available_mean": float(np.mean(search_values.get("combat_quality_potion_urgent_count", [0.0]))),
            "boss_combat/potion_low_urgency_available_mean": float(np.mean(search_values.get("combat_quality_potion_low_urgency_count", [0.0]))),
            "boss_combat/potion_save_recommended_available_mean": float(np.mean(search_values.get("combat_quality_potion_save_recommended_count", [0.0]))),
            "boss_combat/potion_no_followup_available_mean": float(np.mean(search_values.get("combat_quality_potion_no_followup_count", [0.0]))),
            "boss_combat/potion_lethal_available_mean": float(np.mean(search_values.get("combat_quality_potion_lethal_count", [0.0]))),
            "boss_combat/potion_prevent_lethal_available_mean": float(np.mean(search_values.get("combat_quality_potion_prevent_lethal_count", [0.0]))),
            "boss_combat/potion_mechanism_available_mean": float(np.mean(search_values.get("combat_quality_potion_mechanism_count", [0.0]))),
            "boss_combat/potion_overkill_available_mean": float(np.mean(search_values.get("combat_quality_potion_overkill_count", [0.0]))),
            "boss_combat/potion_block_waste_available_mean": float(np.mean(search_values.get("combat_quality_potion_block_waste_count", [0.0]))),
            "boss_combat/potion_use_quality_available_mean": float(np.mean(search_values.get("combat_quality_potion_use_quality_mean", [0.0]))),
            "boss_combat/potion_waste_risk_available_mean": float(np.mean(search_values.get("combat_quality_potion_waste_risk_mean", [0.0]))),
            "boss_combat/potion_selected_rate": float(np.mean(search_values.get("combat_quality_potion_selected", [0.0]))),
            "boss_combat/potion_selected_when_available_rate": float(np.mean(search_values.get("combat_quality_potion_selected_when_available", [0.0]))),
            "boss_combat/potion_high_urgency_selected_rate": float(np.mean(search_values.get("combat_quality_potion_high_urgency_selected", [0.0]))),
            "boss_combat/potion_low_urgency_selected_rate": float(np.mean(search_values.get("combat_quality_potion_low_urgency_selected", [0.0]))),
            "boss_combat/potion_save_recommended_selected_rate": float(np.mean(search_values.get("combat_quality_potion_save_recommended_selected", [0.0]))),
            "boss_combat/potion_no_followup_selected_rate": float(np.mean(search_values.get("combat_quality_potion_no_followup_selected", [0.0]))),
            "boss_combat/potion_lethal_selected_rate": float(np.mean(search_values.get("combat_quality_potion_lethal_selected", [0.0]))),
            "boss_combat/potion_prevent_lethal_selected_rate": float(np.mean(search_values.get("combat_quality_potion_prevent_lethal_selected", [0.0]))),
            "boss_combat/potion_mechanism_selected_rate": float(np.mean(search_values.get("combat_quality_potion_mechanism_selected", [0.0]))),
            "boss_combat/potion_overkill_selected_rate": float(np.mean(search_values.get("combat_quality_potion_overkill_selected", [0.0]))),
            "boss_combat/potion_block_waste_selected_rate": float(np.mean(search_values.get("combat_quality_potion_block_waste_selected", [0.0]))),
            "boss_combat/potion_use_quality_selected_mean": (
                float(np.sum(search_values.get("combat_quality_potion_use_quality_selected", [0.0])))
                / max(float(np.sum(search_values.get("combat_quality_potion_selected", [0.0]))), 1.0)
            ),
            "boss_combat/potion_waste_risk_selected_mean": (
                float(np.sum(search_values.get("combat_quality_potion_waste_risk_selected", [0.0])))
                / max(float(np.sum(search_values.get("combat_quality_potion_selected", [0.0]))), 1.0)
            ),
            # Phase 5 of potion-timing-modeling-plan.md §11.1: aggregate
            # save_value, hand-context, long-term, requires-followup metrics so
            # TensorBoard can spot regressions in the new structured timing.
            "boss_combat/potion_save_value_available_mean": float(np.mean(search_values.get("combat_quality_potion_save_value_mean", [0.0]))),
            "boss_combat/potion_hand_context_good_mean": float(np.mean(search_values.get("combat_quality_potion_hand_context_good_count", [0.0]))),
            "boss_combat/potion_hand_context_bad_mean": float(np.mean(search_values.get("combat_quality_potion_hand_context_bad_count", [0.0]))),
            "boss_combat/potion_long_term_mean": float(np.mean(search_values.get("combat_quality_potion_long_term_count", [0.0]))),
            "boss_combat/potion_requires_followup_mean": float(np.mean(search_values.get("combat_quality_potion_requires_followup_count", [0.0]))),
            "boss_combat/x_cost_available_count_mean": float(np.mean(search_values.get("combat_quality_x_cost_available_count", [0.0]))),
            "boss_combat/x_cost_selected_rate": float(np.mean(search_values.get("combat_quality_x_cost_selected", [0.0]))),
            "boss_combat/x_cost_selected_energy_mean": (
                float(np.sum(search_values.get("combat_quality_x_cost_selected_energy", [0.0])))
                / max(float(np.sum(search_values.get("combat_quality_x_cost_selected", [0.0]))), 1.0)
            ),
            "boss_combat/x_cost_zero_energy_selected_rate": float(np.mean(search_values.get("combat_quality_x_cost_zero_energy_selected", [0.0]))),
            "boss_combat/zero_energy_x_cost_available_mean": float(np.mean(search_values.get("combat_quality_zero_energy_x_cost_count", [0.0]))),
            "boss_combat/zero_energy_x_cost_selected_rate": float(np.mean(search_values.get("combat_quality_zero_energy_x_cost_selected", [0.0]))),
            "boss_combat/x_cost_bad_available_count_mean": float(np.mean(search_values.get("combat_quality_x_cost_bad_count", [0.0]))),
            "boss_combat/x_cost_effective_energy_mean": float(np.mean(search_values.get("combat_quality_x_cost_effective_energy_mean", [0.0]))),
            "boss_combat/x_cost_selected_effective_energy_mean": float(np.mean(search_values.get("combat_quality_x_cost_selected_effective_energy", [0.0]))),
            "boss_combat/x_cost_has_non_energy_effect_selected_rate": float(np.mean(search_values.get("combat_quality_x_cost_has_non_energy_effect_selected", [0.0]))),
            "boss_combat/x_cost_bad_selected_rate": float(np.mean(search_values.get("combat_quality_x_cost_bad_selected", [0.0]))),
            "boss_combat/direct_end_turn_selected_rate": float(np.mean(search_values.get("combat_quality_end_turn_selected", [0.0]))),
            "boss_combat/energy_mean": float(np.mean(search_values.get("combat_quality_energy", [0.0]))),
            "boss_combat/positive_action_count_mean": float(np.mean(search_values.get("combat_quality_positive_action_count", [0.0]))),
            "boss_combat/mandatory_positive_action_count_mean": float(np.mean(search_values.get("combat_quality_mandatory_positive_action_count", [0.0]))),
            "boss_combat/strategic_skip_candidate_count_mean": float(np.mean(search_values.get("combat_quality_strategic_skip_candidate_count", [0.0]))),
            "boss_combat/refund_no_followup_available_mean": float(np.mean(search_values.get("combat_quality_refund_no_followup_available", [0.0]))),
            "boss_combat/refund_no_followup_selected_rate": float(np.mean(search_values.get("combat_quality_refund_no_followup_selected", [0.0]))),
            "boss_combat/strategic_skip_selected_rate": float(np.mean(search_values.get("combat_quality_strategic_skip_selected", [0.0]))),
            "boss_combat/enchantment_seen_rate": float(np.mean(search_values.get("combat_quality_enchantment_seen", [0.0]))),
            "boss_combat/affliction_seen_rate": float(np.mean(search_values.get("combat_quality_affliction_seen", [0.0]))),
            "boss_combat/kaiser_back_attack_risk_mean": float(np.mean(search_values.get("combat_quality_kaiser_back_attack_risk", [0.0]))),
            "boss_combat/kaiser_defense_candidate_count_mean": float(np.mean(search_values.get("combat_quality_kaiser_defense_candidate_count", [0.0]))),
            "boss_combat/kaiser_facing_change_candidate_count_mean": float(np.mean(search_values.get("combat_quality_kaiser_facing_change_candidate_count", [0.0]))),
            "boss_combat/kaiser_pressure_candidate_count_mean": float(np.mean(search_values.get("combat_quality_kaiser_pressure_candidate_count", [0.0]))),
            "boss_combat/kaiser_facing_change_selected_rate": float(np.mean(search_values.get("combat_quality_kaiser_facing_change_selected", [0.0]))),
            "boss_combat/kaiser_pressure_selected_rate": float(np.mean(search_values.get("combat_quality_kaiser_pressure_selected", [0.0]))),
            "boss_combat/kaiser_risky_end_turn_selected_rate": float(np.mean(search_values.get("combat_quality_kaiser_risky_end_turn_selected", [0.0]))),
            "boss_combat/ceremonial_one_card_lock_rate": float(np.mean(search_values.get("combat_quality_ceremonial_one_card_lock", [0.0]))),
            "boss_combat/ceremonial_stun_window_rate": float(np.mean(search_values.get("combat_quality_ceremonial_stun_window", [0.0]))),
            "boss_combat/ceremonial_low_impact_count_mean": float(np.mean(search_values.get("combat_quality_ceremonial_low_impact_count", [0.0]))),
            "boss_combat/ceremonial_high_impact_count_mean": float(np.mean(search_values.get("combat_quality_ceremonial_high_impact_count", [0.0]))),
            "boss_combat/kaiser_defense_selected_rate": float(np.mean(search_values.get("combat_quality_kaiser_defense_selected", [0.0]))),
            "boss_combat/ceremonial_low_impact_selected_rate": float(np.mean(search_values.get("combat_quality_ceremonial_low_impact_selected", [0.0]))),
            "boss_combat/ceremonial_high_impact_selected_rate": float(np.mean(search_values.get("combat_quality_ceremonial_high_impact_selected", [0.0]))),
        }
        for family, count in sorted(family_counts.items()):
            safe_family = family.replace("/", "_").replace(" ", "_") or "unknown"
            metrics[f"boss_combat/family_{safe_family}_rate"] = self._safe_rate(count, total_steps)

        # Boss-combat aggregate metrics are useful for trend watching, but Kaiser /
        # Ceremonial diagnosis gets diluted by other bosses.  Mirror every
        # boss_combat/* scalar under boss_combat/<encounter_id>/* so TensorBoard can
        # answer mechanism-specific questions without post-processing.
        self._mirror_metrics_per_encounter(metrics, encounter_id, prefix="boss_combat/")

        for tag, value in metrics.items():
            self.writer.add_scalar(tag, float(value), self.episode_count)
        return metrics

    @staticmethod
    def _safe_encounter_namespaces(encounter_id: str) -> list[str]:
        """Map an encounter id to TensorBoard-safe namespace token(s)."""
        raw = str(encounter_id or "").strip().lower()
        if not raw:
            return []
        encounter_base = re.sub(r"^encounter[\._:/-]+", "", raw)
        safe = re.sub(r"[^0-9a-zA-Z]+", "_", encounter_base).strip("_")
        raw_safe = re.sub(r"[^0-9a-zA-Z]+", "_", raw).strip("_")
        return list(dict.fromkeys(x for x in (safe, raw_safe) if x))

    @classmethod
    def _mirror_metrics_per_encounter(
        cls,
        metrics: dict[str, float],
        encounter_id: str,
        *,
        prefix: str = "boss_combat/",
    ) -> dict[str, float]:
        """Mirror every ``<prefix><metric>`` scalar under ``<prefix><encounter>/<metric>``.

        Without this mirror, an aggregate boss metric averaged across encounters
        hides per-boss regressions (e.g. Kaiser facing failures averaged with
        Construct Menagerie wins).  Mutates ``metrics`` in place and also returns it.
        """
        for namespace in cls._safe_encounter_namespaces(encounter_id):
            for tag, value in list(metrics.items()):
                if tag.startswith(prefix):
                    rest = tag[len(prefix):]
                    if "/" in rest and rest.split("/", 1)[0] == namespace:
                        # Already mirrored.
                        continue
                    metrics[f"{prefix}{namespace}/{rest}"] = value
        return metrics

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
            if title in {"", "[empty]", "empty", "none", "null"}:
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
            or "璺宠繃鑽按" in text
            or "璺宠繃濂栧姳" in text
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
        if "gold" in text or "閲戝竵" in text:
            return "gold"
        if "potion" in text or "鑽按" in text:
            return "potion"
        if "card reward" in text or "card_reward" in text or "鍗＄墝濂栧姳" in text:
            return "card"
        if "relic" in text or "閬楃墿" in text:
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
        domain_override_counts: dict[str, int] = defaultdict(int)
        combat_like_decision_count = 0
        direct_policy_eligible_count = 0
        direct_policy_used_count = 0
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
            decision_domain, encoded_decision_domain, domain_overridden = self._resolve_acting_decision_domain(
                obs,
                legal_actions,
                prev_info,
            )
            if decision_domain not in decision_counts:
                decision_domain = "build"
            if decision_domain == "combat":
                combat_like_decision_count += 1
                obs = self._with_decision_domain(obs, "combat") or obs
            if domain_overridden:
                domain_override_counts[f"{encoded_decision_domain}->{decision_domain}"] += 1
            decision_counts[decision_domain] += 1
            num_simulations = self._num_simulations_for_domain(decision_domain)
            _lat_legal_count = int((np.asarray(action_mask, dtype=np.float32) > 0).sum())
            _lat_t0 = time.perf_counter()
            obs_batch = dict_obs_to_torch(obs, device=self.device)
            _lat_obs_ms = (time.perf_counter() - _lat_t0) * 1000.0
            _lat_t0 = time.perf_counter()
            with torch.no_grad(), self._amp_autocast():
                initial = self.network.initial_inference(obs_batch)
                root_value = initial.value.squeeze(0).item()
                root_value_components = initial.value_components.squeeze(0).detach().cpu().numpy()
            _lat_initial_ms = (time.perf_counter() - _lat_t0) * 1000.0
            if self.combat_direct_policy and decision_domain == "combat":
                direct_policy_eligible_count += 1

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
            elif self.combat_direct_policy and decision_domain == "combat":
                action_mask_tensor = torch.as_tensor(
                    action_mask,
                    device=initial.policy_logits.device,
                    dtype=initial.policy_logits.dtype,
                )
                masked_logits = initial.policy_logits.squeeze(0).detach().clone()
                rollout_action_mask = action_mask_tensor.unsqueeze(0) > 0
                direct_policy_used_count += 1
                with torch.no_grad(), self._amp_autocast():
                    rollout = self.network.action_rollout_planner(
                        initial.hidden_state,
                        initial.action_embeddings,
                        action_mask=rollout_action_mask,
                        objective_context=obs_batch.get("objective_context"),
                        decision_domain=obs_batch.get("decision_domain"),
                        discount=self.discount,
                        rollout_steps=self.combat_rollout_steps,
                        continuation_beam_width=self.combat_rollout_beam_width,
                        continuation_legal_logit_scale=self.combat_rollout_legal_logit_scale,
                        uncertainty_surprise_weight=self.combat_rollout_uncertainty_surprise_weight,
                        uncertainty_surface_entropy_weight=self.combat_rollout_uncertainty_surface_weight,
                        uncertainty_latent_drift_weight=self.combat_rollout_uncertainty_latent_weight,
                        uncertainty_branch_disagreement_weight=self.combat_rollout_uncertainty_disagreement_weight,
                        continuation_uncertainty_penalty=self.combat_rollout_continuation_uncertainty_penalty,
                    )
                rollout_q = rollout.planner_q.squeeze(0).detach()
                rollout_objective_q = rollout.planner_objective_q.squeeze(0).detach()
                rollout_risk_q = rollout.planner_risk_q.squeeze(0).detach()
                rollout_uncertainty = rollout.planner_uncertainty.squeeze(0).detach()
                rollout_surprise = rollout.planner_surprise.squeeze(0).detach()
                rollout_surface_entropy = rollout.planner_surface_entropy.squeeze(0).detach()
                rollout_latent_drift = rollout.planner_latent_drift.squeeze(0).detach()
                rollout_branch_disagreement = rollout.planner_branch_disagreement.squeeze(0).detach()
                rollout_mask = rollout.action_mask.squeeze(0)
                rollout_q_bias = self._normalize_action_values(
                    rollout_q.unsqueeze(0),
                    rollout_mask.unsqueeze(0),
                ).squeeze(0)
                rollout_objective_q_bias = self._normalize_action_values(
                    rollout_objective_q.unsqueeze(0),
                    rollout_mask.unsqueeze(0),
                ).squeeze(0)
                rollout_risk_q_bias = self._normalize_action_values(
                    rollout_risk_q.unsqueeze(0),
                    rollout_mask.unsqueeze(0),
                ).squeeze(0)
                rollout_uncertainty_bias = self._normalize_action_values(
                    rollout_uncertainty.unsqueeze(0),
                    rollout_mask.unsqueeze(0),
                ).squeeze(0)
                # The supervised surprise/uncertainty heads are raw-scale signals; in boss fights
                # their absolute values can sit around 50+ while Q is around [-3, 1].  We only
                # want uncertainty to rank actions relative to siblings, not dominate policy logits
                # through an outlier z-score, so all rollout side-channel biases are clipped to
                # a bounded logit prior range before blending.
                rollout_q_bias = rollout_q_bias.clamp(-2.5, 2.5)
                rollout_objective_q_bias = rollout_objective_q_bias.clamp(-2.5, 2.5)
                rollout_risk_q_bias = rollout_risk_q_bias.clamp(-2.5, 2.5)
                rollout_uncertainty_bias = rollout_uncertainty_bias.clamp(-2.0, 2.0)
                masked_logits = (
                    masked_logits
                    + self.combat_rollout_q_blend * rollout_q_bias
                    + self.combat_rollout_objective_q_blend * rollout_objective_q_bias
                    + self.combat_rollout_risk_blend * rollout_risk_q_bias
                    - self.combat_rollout_uncertainty_blend * rollout_uncertainty_bias
                )
                quality_bias_np, quality_stats, zero_energy_x_indices = self._combat_action_quality_bias(
                    obs,
                    np.asarray(action_mask, dtype=np.float32),
                    legal_actions if isinstance(legal_actions, list) else [],
                )
                quality_bias = torch.as_tensor(
                    quality_bias_np,
                    device=masked_logits.device,
                    dtype=masked_logits.dtype,
                )
                masked_logits = masked_logits + quality_bias
                masked_logits = masked_logits.masked_fill(action_mask_tensor <= 0, -1e9)
                safe_temperature = max(float(temperature), 1e-3)
                direct_probs = torch.softmax(masked_logits / safe_temperature, dim=0)
                if not torch.isfinite(direct_probs).all() or float(direct_probs.sum().item()) <= 0.0:
                    legal_mask = action_mask_tensor > 0
                    direct_probs = legal_mask.float()
                    direct_probs = direct_probs / direct_probs.sum().clamp(min=1.0)
                direct_probs = direct_probs / direct_probs.sum().clamp(min=1e-8)
                if safe_temperature <= 0.05:
                    action_idx = int(direct_probs.argmax().item())
                else:
                    action_idx = int(torch.multinomial(direct_probs, 1).item())
                search_policy = direct_probs.detach().cpu().numpy().astype(np.float32)
                entropy = -(direct_probs.clamp(min=1e-8) * direct_probs.clamp(min=1e-8).log()).sum().item()
                rollout_valid_count = rollout_mask.float().sum().clamp(min=1.0)
                rollout_q_mean = torch.where(rollout_mask, rollout_q, torch.zeros_like(rollout_q)).sum() / rollout_valid_count
                rollout_objective_q_mean = (
                    torch.where(rollout_mask, rollout_objective_q, torch.zeros_like(rollout_objective_q)).sum()
                    / rollout_valid_count
                )
                rollout_risk_q_mean = (
                    torch.where(rollout_mask, rollout_risk_q, torch.zeros_like(rollout_risk_q)).sum()
                    / rollout_valid_count
                )
                rollout_uncertainty_mean = (
                    torch.where(rollout_mask, rollout_uncertainty, torch.zeros_like(rollout_uncertainty)).sum()
                    / rollout_valid_count
                )
                rollout_surprise_mean = (
                    torch.where(rollout_mask, rollout_surprise, torch.zeros_like(rollout_surprise)).sum()
                    / rollout_valid_count
                )
                rollout_surface_entropy_mean = (
                    torch.where(rollout_mask, rollout_surface_entropy, torch.zeros_like(rollout_surface_entropy)).sum()
                    / rollout_valid_count
                )
                rollout_latent_drift_mean = (
                    torch.where(rollout_mask, rollout_latent_drift, torch.zeros_like(rollout_latent_drift)).sum()
                    / rollout_valid_count
                )
                rollout_branch_disagreement_mean = (
                    torch.where(rollout_mask, rollout_branch_disagreement, torch.zeros_like(rollout_branch_disagreement)).sum()
                    / rollout_valid_count
                )
                rollout_uncertainty_bias_abs_mean = (
                    torch.where(rollout_mask, rollout_uncertainty_bias.abs(), torch.zeros_like(rollout_uncertainty_bias)).sum()
                    / rollout_valid_count
                )
                selected_ceremonial_low_impact = 0.0
                selected_ceremonial_high_impact = 0.0
                if (
                    isinstance(legal_actions, list)
                    and 0 <= int(action_idx) < len(legal_actions)
                    and quality_stats.get("combat_quality_ceremonial_one_card_lock", 0.0) > 0.05
                ):
                    low_impact, high_impact = self._ceremonial_action_timing_flags(
                        legal_actions[int(action_idx)],
                        int(action_idx),
                        obs,
                        self._current_raw_combat_obs(),
                        legal_actions,
                        np.asarray(action_mask, dtype=np.float32).reshape(-1),
                        self._combat_energy(obs, self._current_raw_combat_obs()),
                    )
                    selected_ceremonial_low_impact = 1.0 if low_impact else 0.0
                    selected_ceremonial_high_impact = 1.0 if high_impact else 0.0
                search_stats = {
                    "num_simulations": 0.0,
                    "root_top1_visit_share": float(direct_probs.max().item()),
                    "root_visit_entropy": float(entropy),
                    "mean_predicted_legal_count": float((action_mask_tensor > 0).float().sum().item()),
                    "search_mode_direct_policy": 1.0,
                    "search_mode_direct_rollout_planner": 1.0,
                    "direct_rollout_q_mean": float(rollout_q_mean.item()),
                    "direct_rollout_objective_q_mean": float(rollout_objective_q_mean.item()),
                    "direct_rollout_risk_q_mean": float(rollout_risk_q_mean.item()),
                    "direct_rollout_uncertainty_mean": float(rollout_uncertainty_mean.item()),
                    "direct_rollout_uncertainty_bias_abs_mean": float(rollout_uncertainty_bias_abs_mean.item()),
                    "direct_rollout_surprise_mean": float(rollout_surprise_mean.item()),
                    "direct_rollout_surface_entropy_mean": float(rollout_surface_entropy_mean.item()),
                    "direct_rollout_latent_drift_mean": float(rollout_latent_drift_mean.item()),
                    "direct_rollout_branch_disagreement_mean": float(rollout_branch_disagreement_mean.item()),
                    "direct_rollout_steps_used": float(rollout.rollout_steps_used),
                    "direct_rollout_branch_count_mean": float(rollout.rollout_branch_count_mean),
                    "direct_rollout_root_valid_count": float(rollout.rollout_root_valid_count),
                    "direct_rollout_root_bucket_size": float(rollout.rollout_root_bucket_size),
                    "direct_rollout_bucket_padding_ratio": float(rollout.rollout_bucket_padding_ratio),
                    "direct_rollout_max_branch_bucket_size": float(rollout.rollout_max_branch_bucket_size),
                    "direct_rollout_branch_padding_ratio": float(rollout.rollout_branch_padding_ratio),
                    **quality_stats,
                    "combat_quality_zero_energy_x_cost_selected": 1.0 if int(action_idx) in zero_energy_x_indices else 0.0,
                    "combat_quality_end_turn_selected": 1.0 if self._semantic_family(legal_actions[int(action_idx)] if isinstance(legal_actions, list) and 0 <= int(action_idx) < len(legal_actions) else {}) == "end_turn" else 0.0,
                    "combat_quality_wasteful_end_turn_selected": (
                        1.0
                        if (
                            self._semantic_family(legal_actions[int(action_idx)] if isinstance(legal_actions, list) and 0 <= int(action_idx) < len(legal_actions) else {}) == "end_turn"
                            and bool(quality_stats.get("combat_quality_wasteful_end_turn_available", 0.0) > 0.5)
                        )
                        else 0.0
                    ),
                    "combat_quality_kaiser_defense_selected": (
                        1.0
                        if (
                            isinstance(legal_actions, list)
                            and 0 <= int(action_idx) < len(legal_actions)
                            and quality_stats.get("combat_quality_kaiser_back_attack_risk", 0.0) > 0.05
                            and self._is_kaiser_risk_handling_action(legal_actions[int(action_idx)], self._current_raw_combat_obs())
                        )
                        else 0.0
                    ),
                    "combat_quality_kaiser_facing_change_selected": (
                        1.0
                        if (
                            isinstance(legal_actions, list)
                            and 0 <= int(action_idx) < len(legal_actions)
                            and quality_stats.get("combat_quality_kaiser_back_attack_risk", 0.0) > 0.05
                            and self._is_kaiser_facing_change_action(legal_actions[int(action_idx)], self._current_raw_combat_obs())
                        )
                        else 0.0
                    ),
                    "combat_quality_kaiser_pressure_selected": (
                        1.0
                        if (
                            isinstance(legal_actions, list)
                            and 0 <= int(action_idx) < len(legal_actions)
                            and quality_stats.get("combat_quality_kaiser_back_attack_risk", 0.0) > 0.05
                            and self._is_kaiser_pressure_action(legal_actions[int(action_idx)])
                        )
                        else 0.0
                    ),
                    "combat_quality_kaiser_risky_end_turn_selected": (
                        1.0
                        if (
                            isinstance(legal_actions, list)
                            and 0 <= int(action_idx) < len(legal_actions)
                            and quality_stats.get("combat_quality_kaiser_back_attack_risk", 0.0) > 0.05
                            and self._semantic_family(legal_actions[int(action_idx)]) == "end_turn"
                        )
                        else 0.0
                    ),
                    "combat_quality_ceremonial_low_impact_selected": selected_ceremonial_low_impact,
                    "combat_quality_ceremonial_high_impact_selected": selected_ceremonial_high_impact,
                }
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
            if not action_family and isinstance(chosen_action, dict):
                action_family = self._semantic_family(chosen_action)
                if action_family:
                    selected_family_counts[decision_domain][action_family] += 1
            if action_family == "end_turn":
                end_turn_context: dict[str, Any] = {}
                try:
                    end_turn_context = self._raw_end_turn_context(
                        obs,
                        np.asarray(action_mask, dtype=np.float32).reshape(-1),
                        legal_actions,
                    )
                    wasteful_end_turn = bool(end_turn_context.get("wasteful"))
                except Exception:
                    end_turn_context = {}
                    wasteful_end_turn = False
                if decision_domain == "combat":
                    try:
                        raw_obs_for_dump = self._current_raw_combat_obs()
                        encounter_for_dump = ""
                        if isinstance(raw_obs_for_dump, dict):
                            try:
                                boss_ctx_dump = build_boss_mechanics_context(raw_obs_for_dump)
                                encounter_for_dump = str(boss_ctx_dump.get("encounter_key") or "").lower()
                            except Exception:
                                encounter_for_dump = ""
                        action_diag_pre = info.get("action_diagnostics") if isinstance(info, dict) else None
                        self._dump_selected_end_turn_context(
                            encoded_obs=obs,
                            raw_obs=raw_obs_for_dump,
                            action_mask=np.asarray(action_mask, dtype=np.float32),
                            legal_actions=legal_actions if isinstance(legal_actions, list) else [],
                            chosen_idx=int(action_idx),
                            context=end_turn_context,
                            search_policy=search_policy,
                            search_stats=search_stats if isinstance(search_stats, dict) else None,
                            action_diagnostics=action_diag_pre if isinstance(action_diag_pre, dict) else None,
                            encounter=encounter_for_dump,
                            tier=str((info.get("tier") if isinstance(info, dict) else "") or ""),
                        )
                    except Exception:
                        pass
            wasteful_proceed = self._wasteful_proceed_flag(
                chosen_signature,
                decision_domain=decision_domain,
                phase=phase,
            )

            if decision_domain == "combat" and isinstance(search_stats, dict):
                search_stats.update(
                    self._selected_combat_quality_stats(
                        obs,
                        int(action_idx),
                        legal_actions if isinstance(legal_actions, list) else [],
                        search_stats,
                    )
                )
                try:
                    raw_obs_off = self._current_raw_combat_obs()
                    encounter_off = ""
                    if isinstance(raw_obs_off, dict):
                        try:
                            boss_ctx_off = build_boss_mechanics_context(raw_obs_off)
                            encounter_off = str(boss_ctx_off.get("encounter_key") or "").lower()
                        except Exception:
                            encounter_off = ""
                    offender_types = self._classify_action_offenders(
                        search_stats=search_stats,
                        encounter=encounter_off,
                        family=action_family,
                    )
                    for offender_type in offender_types:
                        global_key = f"boss_combat/{offender_type}_count"
                        search_stats[f"offender/{offender_type}"] = 1.0
                    if offender_types:
                        self._dump_action_offender(
                            encoded_obs=obs,
                            raw_obs=raw_obs_off,
                            action_mask=np.asarray(action_mask, dtype=np.float32),
                            legal_actions=legal_actions if isinstance(legal_actions, list) else [],
                            chosen_idx=int(action_idx),
                            chosen_action=chosen_action,
                            offender_types=offender_types,
                            search_stats=search_stats,
                            encounter=encounter_off,
                            tier=str((info.get("tier") if isinstance(info, dict) else "") or ""),
                        )
                except Exception:
                    pass

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
                search_stats={
                    **self._compact_search_stats(search_stats),
                    "combat_quality_wasteful_end_turn_selected": 1.0 if wasteful_end_turn else float((search_stats or {}).get("combat_quality_wasteful_end_turn_selected", 0.0) or 0.0),
                },
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
            potion_transition_record = info.get("potion_transition") if isinstance(info, dict) else None
            if isinstance(potion_transition_record, dict):
                self._dump_potion_transition(potion_transition_record)
            action_diagnostics = info.get("action_diagnostics") if isinstance(info, dict) else None
            if isinstance(action_diagnostics, dict) and trajectory.steps:
                diag_stats = trajectory.steps[-1].setdefault("search_stats", {})
                diag_key_map = {
                    "energy": "combat_quality_energy",
                    "positive_action_count": "combat_quality_positive_action_count",
                    "mandatory_positive_action_count": "combat_quality_mandatory_positive_action_count",
                    "strategic_skip_candidate_count": "combat_quality_strategic_skip_candidate_count",
                    "wasteful_end_turn_available": "combat_quality_true_wasteful_end_turn_available",
                    "wasteful_end_turn_selected": "combat_quality_true_wasteful_end_turn_selected",
                    "zero_energy_x_cost_available": "combat_quality_zero_energy_x_cost_count",
                    "zero_energy_x_cost_selected": "combat_quality_zero_energy_x_cost_selected",
                    "refund_no_followup_available": "combat_quality_refund_no_followup_available",
                    "refund_no_followup_selected": "combat_quality_refund_no_followup_selected",
                    "strategic_skip_selected": "combat_quality_strategic_skip_selected",
                    "typed_followup_missing_count": "combat_quality_typed_followup_missing_count",
                    "typed_future_penalty_count": "combat_quality_typed_future_penalty_count",
                    "typed_no_draw_count": "combat_quality_typed_no_draw_count",
                    "typed_card_state_mutation_count": "combat_quality_typed_card_state_mutation_count",
                    "setup_followup_dependent_count": "combat_quality_setup_followup_dependent_count",
                    "setup_followup_available_count": "combat_quality_setup_followup_available_count",
                    "enchantment_seen": "combat_quality_enchantment_seen",
                    "affliction_seen": "combat_quality_affliction_seen",
                    "wasteful_end_turn_penalty_applied": "combat_quality_wasteful_end_turn_penalty_applied",
                }
                for diag_key, stat_key in diag_key_map.items():
                    if diag_key not in action_diagnostics:
                        continue
                    try:
                        diag_stats[stat_key] = float(action_diagnostics.get(diag_key) or 0.0)
                    except (TypeError, ValueError):
                        continue
                if float(diag_stats.get("combat_quality_true_wasteful_end_turn_available", 0.0) or 0.0) > 0.5:
                    diag_stats["combat_quality_wasteful_end_turn_available"] = 1.0
                if float(diag_stats.get("combat_quality_true_wasteful_end_turn_selected", 0.0) or 0.0) > 0.5:
                    diag_stats["combat_quality_wasteful_end_turn_selected"] = 1.0
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
        encounter_id = initial_info.get("encounter_id")
        encounter_tier = infer_encounter_tier(encounter_id)
        boss_entry_snapshot = self._boss_entry_snapshot_from_episode(
            initial_info,
            progress_snapshots,
            encounter_tier=encounter_tier,
        )
        final_potion_count = self._potion_count_from_info(final_info)
        used_potion_count = int((final_info or {}).get("used_potion_count_this_combat", 0) or 0)
        trajectory.metadata = {
            "episode_mode": initial_info.get("episode_mode"),
            "encounter_id": encounter_id,
            "encounter_tier": encounter_tier,
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
            "boss_entry_hp": float(boss_entry_snapshot.get("hp", 0.0)) if boss_entry_snapshot else 0.0,
            "boss_entry_max_hp": float(boss_entry_snapshot.get("max_hp", 0.0)) if boss_entry_snapshot else 0.0,
            "boss_entry_hp_ratio": float(boss_entry_snapshot.get("hp_ratio", 0.0)) if boss_entry_snapshot else 0.0,
            "final_potion_count": int(final_potion_count),
            "used_potion_count_this_combat": int(used_potion_count),
            "act1_boss_seen": bool(act1_boss_seen),
            "act1_clear": bool(act1_clear),
            "decision_counts": {domain: int(count) for domain, count in decision_counts.items()},
            "decision_domain_overrides": {key: int(value) for key, value in domain_override_counts.items()},
            "combat_like_decision_count": int(combat_like_decision_count),
            "direct_policy_eligible_count": int(direct_policy_eligible_count),
            "direct_policy_used_count": int(direct_policy_used_count),
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
        self.writer.add_scalar(
            "decision/domain_override_count",
            float(sum(domain_override_counts.values())),
            self.episode_count,
        )
        self.writer.add_scalar("decision/combat_like_count", float(combat_like_decision_count), self.episode_count)
        self.writer.add_scalar(
            "decision/direct_policy_eligible_count",
            float(direct_policy_eligible_count),
            self.episode_count,
        )
        self.writer.add_scalar(
            "decision/direct_policy_used_count",
            float(direct_policy_used_count),
            self.episode_count,
        )
        self.writer.add_scalar(
            "decision/direct_policy_used_rate",
            float(direct_policy_used_count) / float(max(direct_policy_eligible_count, 1)),
            self.episode_count,
        )
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
                "search_mode_direct_policy": "search_mode_direct_policy",
                "search_mode_direct_rollout_planner": "search_mode_direct_rollout_planner",
                "direct_rollout_q_mean": "direct_rollout_q_mean",
                "direct_rollout_objective_q_mean": "direct_rollout_objective_q_mean",
                "direct_rollout_risk_q_mean": "direct_rollout_risk_q_mean",
                "direct_rollout_uncertainty_mean": "direct_rollout_uncertainty_mean",
                "direct_rollout_surprise_mean": "direct_rollout_surprise_mean",
                "direct_rollout_surface_entropy_mean": "direct_rollout_surface_entropy_mean",
                "direct_rollout_latent_drift_mean": "direct_rollout_latent_drift_mean",
                "direct_rollout_branch_disagreement_mean": "direct_rollout_branch_disagreement_mean",
                "direct_rollout_steps_used": "direct_rollout_steps_used",
                "direct_rollout_branch_count_mean": "direct_rollout_branch_count_mean",
                "direct_rollout_root_valid_count": "direct_rollout_root_valid_count",
                "direct_rollout_root_bucket_size": "direct_rollout_root_bucket_size",
                "direct_rollout_bucket_padding_ratio": "direct_rollout_bucket_padding_ratio",
                "direct_rollout_max_branch_bucket_size": "direct_rollout_max_branch_bucket_size",
                "direct_rollout_branch_padding_ratio": "direct_rollout_branch_padding_ratio",
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
        boss_diagnostics = self._emit_boss_episode_diagnostics(
            trajectory,
            boss_entry=boss_entry_snapshot,
            final_potion_count=final_potion_count,
        )

        self.last_episode_metrics = {
            "decision_counts": {domain: int(count) for domain, count in decision_counts.items()},
            "decision_total": int(total_decisions),
            "decision_domain_overrides": {key: int(value) for key, value in domain_override_counts.items()},
            "combat_like_decision_count": int(combat_like_decision_count),
            "direct_policy_eligible_count": int(direct_policy_eligible_count),
            "direct_policy_used_count": int(direct_policy_used_count),
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
            "boss_diagnostics": boss_diagnostics,
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

        # Materialize observation batches lazily but cache per-step tensors / EMA
        # teacher encodings so multi-step future rollouts can reuse them.
        obs_torch_cache: dict[int, dict[str, torch.Tensor]] = {}
        teacher_token_cache: dict[int, Any] = {}

        def get_obs_step(step_index: int) -> dict[str, torch.Tensor]:
            cached = obs_torch_cache.get(step_index)
            if cached is None:
                cached = self._obs_list_to_torch([sequence[step_index] for sequence in obs_sequence_batch])
                obs_torch_cache[step_index] = cached
            return cached

        def get_teacher_token_step(step_index: int) -> Any:
            if not self.network.is_token_mode:
                return None
            if step_index not in teacher_token_cache:
                teacher_token_cache[step_index] = self._encode_target_token_obs(get_obs_step(step_index))
            return teacher_token_cache[step_index]

        current_obs_torch = get_obs_step(0)
        current_teacher_token_encoded = (
            get_teacher_token_step(0)
            if (
                self.network.is_token_mode
                and (self.future_world_aux_weight > 0.0 or self.future_world_rollout_weight > 0.0)
            )
            else None
        )
        batch_indices = torch.arange(batch_size_actual, device=self.device)

        with self._amp_autocast():
            # Initial inference
            initial = self.network.initial_inference(current_obs_torch)
            hidden_state = initial.hidden_state
            policy_logits = initial.policy_logits
            value_logits = initial.value_logits
    
            # Losses
            total_loss = 0.0
            policy_loss_sum = 0.0
            latent_policy_loss_sum = 0.0
            planner_q_loss_sum = 0.0
            planner_objective_q_loss_sum = 0.0
            planner_q_terms = 0.0
            planner_objective_q_terms = 0.0
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
            latent_gaussian_reg_loss_sum = 0.0
            latent_gaussian_reg_terms = 0.0
            latent_reg_mean_abs_sum = 0.0
            latent_reg_var_mean_sum = 0.0
            latent_reg_var_std_sum = 0.0
            latent_reg_cov_offdiag_sum = 0.0
            latent_reg_slot_var_mean_sum = 0.0
            surprise_loss_sum = 0.0
            surprise_terms = 0.0
            surprise_target_mean_sum = 0.0
            surprise_pred_mean_sum = 0.0
            surprise_mae_sum = 0.0
            surprise_target_offset_sum = 0.0
            future_world_aux_loss_sum = 0.0
            future_bank_state_loss_sum = 0.0
            latent_policy_distill_kl_sum = 0.0
            latent_policy_distill_agreement_sum = 0.0
            planner_q_mae_sum = 0.0
            planner_objective_q_mae_sum = 0.0
            planner_risk_q_mae_sum = 0.0
            teacher_entropy_sum = 0.0
            student_entropy_sum = 0.0
            state_consistency_cosine_sum = 0.0
            state_consistency_mse_sum = 0.0
            future_bank_state_cosine_sum = 0.0
            future_bank_state_mse_sum = 0.0
            future_bank_delta_loss_sum = 0.0
            future_bank_delta_mae_sum = 0.0
            future_bank_occupancy_loss_sum = 0.0
            future_bank_occupancy_mae_sum = 0.0
            future_bank_token_presence_loss_sum = 0.0
            future_bank_token_presence_mae_sum = 0.0
            future_bank_token_distribution_loss_sum = 0.0
            future_bank_token_distribution_kl_sum = 0.0
            future_bank_token_distribution_mae_sum = 0.0
            future_bank_token_slot_state_loss_sum = 0.0
            future_bank_token_slot_state_cosine_sum = 0.0
            future_bank_token_slot_state_mse_sum = 0.0
            future_bank_token_slot_mask_loss_sum = 0.0
            future_bank_token_slot_mask_mae_sum = 0.0
            future_bank_token_slot_type_loss_sum = 0.0
            future_bank_token_slot_type_acc_sum = 0.0
            future_bank_token_slot_zone_loss_sum = 0.0
            future_bank_token_slot_zone_acc_sum = 0.0
            future_bank_token_slot_source_loss_sum = 0.0
            future_bank_token_slot_source_acc_sum = 0.0
            future_world_rollout_aux_loss_sum = 0.0
            future_world_rollout_horizon_sum = 0.0
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
            if (
                self.planner_q_loss_weight > 0.0
                and initial.planner_q_logits is not None
                and action_batch.shape[1] > 0
                and reward_target.shape[1] > 0
                and value_target.shape[1] > 1
            ):
                root_q_target = reward_target[:, 0] + self.discount * value_target[:, 1]
                root_valid_mask = action_mask_batch[:, 0].sum(dim=-1) > 0
                planner_q_loss, planner_q_mae = self._planner_q_loss(
                    initial.planner_q_logits,
                    action_batch[:, 0],
                    root_q_target,
                    valid_mask=root_valid_mask,
                )
                planner_q_supervised = True
            else:
                planner_q_loss = value_logits.new_zeros(())
                planner_q_mae = 0.0
                planner_q_supervised = False
            if (
                self.planner_objective_q_loss_weight > 0.0
                and initial.planner_q_component_logits is not None
                and action_batch.shape[1] > 0
                and reward_component_target.shape[1] > 0
                and value_component_target.shape[1] > 1
            ):
                root_objective_q_target = reward_component_target[:, 0] + self.discount * value_component_target[:, 1]
                root_valid_mask = action_mask_batch[:, 0].sum(dim=-1) > 0
                planner_objective_q_loss, planner_objective_q_mae, planner_risk_q_mae = self._planner_objective_q_loss(
                    initial.planner_q_component_logits,
                    action_batch[:, 0],
                    root_objective_q_target,
                    valid_mask=root_valid_mask,
                )
                planner_objective_q_supervised = True
            else:
                planner_objective_q_loss = value_logits.new_zeros(())
                planner_objective_q_mae = 0.0
                planner_risk_q_mae = 0.0
                planner_objective_q_supervised = False
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
            latent_gaussian_reg_loss, latent_reg_metrics = self._latent_regularization_loss(
                initial.hidden_state,
                dynamics=False,
            )
            loss = (
                policy_loss
                + value_loss
                + self.objective_value_weight * objective_value_loss
                + self.planner_q_loss_weight * planner_q_loss
                + self.planner_objective_q_loss_weight * planner_objective_q_loss
                + self.semantic_policy_weight * semantic_policy_loss
                + self.semantic_value_weight * (semantic_value_loss + semantic_objective_value_loss)
                + self.objective_diversity_weight * objective_diversity_loss
                + self.latent_policy_target_weight * latent_policy_loss
                + self.latent_policy_distill_weight * latent_policy_distill_loss
                + self.latent_gaussian_reg_weight * latent_gaussian_reg_loss
            )
            total_loss += loss / (unroll_steps + 1)
            policy_loss_sum += policy_loss.item()
            latent_policy_loss_sum += latent_policy_loss.item()
            planner_q_loss_sum += planner_q_loss.item()
            planner_objective_q_loss_sum += planner_objective_q_loss.item()
            planner_q_terms += 1.0 if planner_q_supervised else 0.0
            planner_objective_q_terms += 1.0 if planner_objective_q_supervised else 0.0
            value_loss_sum += value_loss.item()
            objective_value_loss_sum += objective_value_loss.item()
            semantic_policy_loss_sum += semantic_policy_loss.item()
            semantic_value_loss_sum += (semantic_value_loss.item() + semantic_objective_value_loss.item())
            objective_diversity_loss_sum += objective_diversity_loss.item()
            latent_policy_distill_loss_sum += latent_policy_distill_loss.item()
            latent_gaussian_reg_loss_sum += latent_gaussian_reg_loss.item()
            latent_gaussian_reg_terms += 1.0
            latent_reg_mean_abs_sum += latent_reg_metrics["mean_abs"]
            latent_reg_var_mean_sum += latent_reg_metrics["var_mean"]
            latent_reg_var_std_sum += latent_reg_metrics["var_std"]
            latent_reg_cov_offdiag_sum += latent_reg_metrics["cov_offdiag"]
            latent_reg_slot_var_mean_sum += latent_reg_metrics["slot_var_mean"]
            latent_policy_distill_kl_sum += latent_policy_distill_metrics["kl"]
            latent_policy_distill_agreement_sum += latent_policy_distill_metrics["top1_agreement"]
            planner_q_mae_sum += planner_q_mae
            planner_objective_q_mae_sum += planner_objective_q_mae
            planner_risk_q_mae_sum += planner_risk_q_mae
            teacher_entropy_sum += latent_policy_distill_metrics["teacher_entropy"]
            student_entropy_sum += latent_policy_distill_metrics["student_entropy"]
    
            # Unrolled steps
            for step_k in range(unroll_steps):
                next_obs_torch = get_obs_step(step_k + 1)
                next_teacher_token_encoded = get_teacher_token_step(step_k + 1)
                action_embeddings = self.network.encode_actions(current_obs_torch)  # [B, 80, 64]
    
                # Select action embeddings
                action_indices = action_batch[:, step_k]  # [B]
                action_emb = action_embeddings[batch_indices, action_indices]  # [B, 64]
    
                # Recurrent inference
                recurrent = self.network.recurrent_inference(
                    hidden_state,
                    action_emb,
                    current_obs=current_obs_torch if (self.network.is_token_mode and self.future_bank_token_slot_source_weight > 0.0) else None,
                    teacher_current_encoded=current_teacher_token_encoded,
                    next_obs=next_obs_torch,
                    teacher_next_encoded=next_teacher_token_encoded,
                )
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
                if (
                    self.planner_q_loss_weight > 0.0
                    and recurrent.planner_q_logits is not None
                    and (step_k + 1) < action_batch.shape[1]
                    and (step_k + 1) < reward_target.shape[1]
                    and (step_k + 2) < value_target.shape[1]
                ):
                    planner_next_action = action_batch[:, step_k + 1]
                    planner_q_target = reward_target[:, step_k + 1] + self.discount * value_target[:, step_k + 2]
                    planner_valid_mask = action_mask_batch[:, step_k + 1].sum(dim=-1) > 0
                    planner_q_loss, planner_q_mae = self._planner_q_loss(
                        recurrent.planner_q_logits,
                        planner_next_action,
                        planner_q_target,
                        valid_mask=planner_valid_mask,
                    )
                    planner_q_supervised = True
                else:
                    planner_q_loss = value_logits.new_zeros(())
                    planner_q_mae = 0.0
                    planner_q_supervised = False
                if (
                    self.planner_objective_q_loss_weight > 0.0
                    and recurrent.planner_q_component_logits is not None
                    and (step_k + 1) < action_batch.shape[1]
                    and (step_k + 1) < reward_component_target.shape[1]
                    and (step_k + 2) < value_component_target.shape[1]
                ):
                    planner_objective_next_action = action_batch[:, step_k + 1]
                    planner_objective_q_target = (
                        reward_component_target[:, step_k + 1]
                        + self.discount * value_component_target[:, step_k + 2]
                    )
                    planner_objective_valid_mask = action_mask_batch[:, step_k + 1].sum(dim=-1) > 0
                    (
                        planner_objective_q_loss,
                        planner_objective_q_mae,
                        planner_risk_q_mae,
                    ) = self._planner_objective_q_loss(
                        recurrent.planner_q_component_logits,
                        planner_objective_next_action,
                        planner_objective_q_target,
                        valid_mask=planner_objective_valid_mask,
                    )
                    planner_objective_q_supervised = True
                else:
                    planner_objective_q_loss = value_logits.new_zeros(())
                    planner_objective_q_mae = 0.0
                    planner_risk_q_mae = 0.0
                    planner_objective_q_supervised = False
                teacher_hidden = recurrent.teacher_hidden_state
                if teacher_hidden is None:
                    with torch.no_grad():
                        if self.network.is_token_mode:
                            teacher_hidden = self.network.token_encoder(next_obs_torch).hidden_state
                        else:
                            teacher_hidden = self.network.representation(next_obs_torch)
                with torch.no_grad():
                    teacher_semantic = self.network.project_to_semantic_latent(teacher_hidden)
                state_consistency_loss, state_consistency_cosine, state_consistency_mse = self._state_consistency_loss(
                    recurrent.next_hidden_state,
                    teacher_hidden,
                )
                surprise_target = self._hidden_surprise_target(
                    recurrent.next_hidden_state,
                    teacher_hidden,
                )
                latent_gaussian_reg_loss, latent_reg_metrics = self._latent_regularization_loss(
                    recurrent.next_hidden_state,
                    dynamics=True,
                )
                future_aux_terms = self._future_world_aux_terms(
                    recurrent,
                    zero_ref=value_logits,
                )
                future_bank_state_loss = future_aux_terms["future_bank_state_loss"]
                future_bank_state_cosine = float(future_aux_terms["future_bank_state_cosine"])
                future_bank_state_mse = float(future_aux_terms["future_bank_state_mse"])
                future_bank_delta_loss = future_aux_terms["future_bank_delta_loss"]
                future_bank_delta_mae = float(future_aux_terms["future_bank_delta_mae"])
                future_bank_occupancy_loss = future_aux_terms["future_bank_occupancy_loss"]
                future_bank_occupancy_mae = float(future_aux_terms["future_bank_occupancy_mae"])
                future_bank_token_presence_loss = future_aux_terms["future_bank_token_presence_loss"]
                future_bank_token_presence_mae = float(future_aux_terms["future_bank_token_presence_mae"])
                future_bank_token_distribution_loss = future_aux_terms["future_bank_token_distribution_loss"]
                future_bank_token_distribution_kl = float(future_aux_terms["future_bank_token_distribution_kl"])
                future_bank_token_distribution_mae = float(future_aux_terms["future_bank_token_distribution_mae"])
                future_bank_token_slot_state_loss = future_aux_terms["future_bank_token_slot_state_loss"]
                future_bank_token_slot_state_cosine = float(future_aux_terms["future_bank_token_slot_state_cosine"])
                future_bank_token_slot_state_mse = float(future_aux_terms["future_bank_token_slot_state_mse"])
                future_bank_token_slot_mask_loss = future_aux_terms["future_bank_token_slot_mask_loss"]
                future_bank_token_slot_mask_mae = float(future_aux_terms["future_bank_token_slot_mask_mae"])
                future_bank_token_slot_type_loss = future_aux_terms["future_bank_token_slot_type_loss"]
                future_bank_token_slot_type_acc = float(future_aux_terms["future_bank_token_slot_type_acc"])
                future_bank_token_slot_zone_loss = future_aux_terms["future_bank_token_slot_zone_loss"]
                future_bank_token_slot_zone_acc = float(future_aux_terms["future_bank_token_slot_zone_acc"])
                future_bank_token_slot_source_loss = future_aux_terms["future_bank_token_slot_source_loss"]
                future_bank_token_slot_source_acc = float(future_aux_terms["future_bank_token_slot_source_acc"])
                future_world_aux_loss = future_aux_terms["loss_total"]
    
                future_world_rollout_aux_loss = value_logits.new_zeros(())
                rollout_applied_horizons = 0.0
                if (
                    self.network.is_token_mode
                    and self.future_world_rollout_steps > 0
                    and self.future_world_rollout_weight > 0.0
                ):
                    rollout_hidden_state = recurrent.next_hidden_state
                    rollout_current_teacher_encoded = next_teacher_token_encoded
                    rollout_weight_sum = 0.0
                    rollout_loss_terms: list[torch.Tensor] = []
                    for rollout_offset in range(1, self.future_world_rollout_steps + 1):
                        rollout_current_index = step_k + rollout_offset
                        rollout_next_index = rollout_current_index + 1
                        if rollout_next_index > unroll_steps:
                            break
                        rollout_current_obs = get_obs_step(rollout_current_index)
                        rollout_next_obs = get_obs_step(rollout_next_index)
                        rollout_next_teacher_encoded = get_teacher_token_step(rollout_next_index)
                        rollout_action_embeddings = self.network.encode_actions(rollout_current_obs)
                        rollout_action_indices = action_batch[:, rollout_current_index]
                        rollout_action_emb = rollout_action_embeddings[batch_indices, rollout_action_indices]
                        rollout_recurrent = self.network.recurrent_inference(
                            rollout_hidden_state,
                            rollout_action_emb,
                            current_obs=rollout_current_obs if (self.network.is_token_mode and self.future_bank_token_slot_source_weight > 0.0) else None,
                            teacher_current_encoded=rollout_current_teacher_encoded,
                            next_obs=rollout_next_obs,
                            teacher_next_encoded=rollout_next_teacher_encoded,
                        )
                        rollout_terms = self._future_world_aux_terms(
                            rollout_recurrent,
                            zero_ref=value_logits,
                        )
                        rollout_weight = self.future_world_rollout_decay ** float(rollout_offset - 1)
                        rollout_loss_terms.append(rollout_terms["loss_total"] * rollout_weight)
                        rollout_weight_sum += rollout_weight
                        rollout_applied_horizons += 1.0
                        rollout_hidden_state = rollout_recurrent.next_hidden_state
                        rollout_current_teacher_encoded = rollout_next_teacher_encoded
                    if rollout_loss_terms and rollout_weight_sum > 0.0:
                        future_world_rollout_aux_loss = sum(rollout_loss_terms) / rollout_weight_sum
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
                surprise_target, surprise_target_offset = self._augment_surprise_target(
                    surprise_target,
                    future_world_aux_loss=future_world_aux_loss,
                    surface_mask_loss=surface_mask_loss,
                    surface_count_loss=surface_count_loss,
                    surface_domain_loss=surface_domain_loss,
                    surface_phase_loss=surface_phase_loss,
                )
                surprise_loss, surprise_target_mean, surprise_pred_mean, surprise_mae = self._surprise_loss(
                    recurrent.surprise,
                    surprise_target,
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
                    + self.planner_q_loss_weight * planner_q_loss
                    + self.planner_objective_q_loss_weight * planner_objective_q_loss
                    + self.state_consistency_weight * state_consistency_loss
                    + self.latent_gaussian_reg_weight * latent_gaussian_reg_loss
                    + self.surprise_loss_weight * surprise_loss
                    + self.future_world_aux_weight * (
                        future_world_aux_loss
                        + self.future_world_rollout_weight * future_world_rollout_aux_loss
                    )
                    + self.surface_mask_weight * surface_mask_loss
                    + self.surface_count_weight * surface_count_loss
                    + self.surface_domain_weight * surface_domain_loss
                    + self.surface_phase_weight * surface_phase_loss
                ) / (unroll_steps + 1)
                total_loss += loss
                policy_loss_sum += policy_loss.item()
                latent_policy_loss_sum += latent_policy_loss.item()
                planner_q_loss_sum += planner_q_loss.item()
                planner_objective_q_loss_sum += planner_objective_q_loss.item()
                planner_q_terms += 1.0 if planner_q_supervised else 0.0
                planner_objective_q_terms += 1.0 if planner_objective_q_supervised else 0.0
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
                latent_gaussian_reg_loss_sum += latent_gaussian_reg_loss.item()
                latent_gaussian_reg_terms += 1.0
                latent_reg_mean_abs_sum += latent_reg_metrics["mean_abs"]
                latent_reg_var_mean_sum += latent_reg_metrics["var_mean"]
                latent_reg_var_std_sum += latent_reg_metrics["var_std"]
                latent_reg_cov_offdiag_sum += latent_reg_metrics["cov_offdiag"]
                latent_reg_slot_var_mean_sum += latent_reg_metrics["slot_var_mean"]
                surprise_loss_sum += surprise_loss.item()
                surprise_terms += 1.0
                surprise_target_mean_sum += surprise_target_mean
                surprise_pred_mean_sum += surprise_pred_mean
                surprise_mae_sum += surprise_mae
                surprise_target_offset_sum += surprise_target_offset
                future_world_aux_loss_sum += future_world_aux_loss.item()
                future_bank_state_loss_sum += future_bank_state_loss.item()
                future_bank_delta_loss_sum += future_bank_delta_loss.item()
                future_bank_occupancy_loss_sum += future_bank_occupancy_loss.item()
                surface_mask_loss_sum += surface_mask_loss.item()
                surface_count_loss_sum += surface_count_loss.item()
                surface_domain_loss_sum += surface_domain_loss.item()
                surface_phase_loss_sum += surface_phase_loss.item()
                latent_policy_distill_kl_sum += latent_policy_distill_metrics["kl"]
                latent_policy_distill_agreement_sum += latent_policy_distill_metrics["top1_agreement"]
                planner_q_mae_sum += planner_q_mae
                planner_objective_q_mae_sum += planner_objective_q_mae
                planner_risk_q_mae_sum += planner_risk_q_mae
                teacher_entropy_sum += latent_policy_distill_metrics["teacher_entropy"]
                student_entropy_sum += latent_policy_distill_metrics["student_entropy"]
                state_consistency_cosine_sum += state_consistency_cosine
                state_consistency_mse_sum += state_consistency_mse
                future_bank_state_cosine_sum += future_bank_state_cosine
                future_bank_state_mse_sum += future_bank_state_mse
                future_bank_delta_mae_sum += future_bank_delta_mae
                future_bank_occupancy_mae_sum += future_bank_occupancy_mae
                future_bank_token_presence_loss_sum += future_bank_token_presence_loss.item()
                future_bank_token_presence_mae_sum += future_bank_token_presence_mae
                future_bank_token_distribution_loss_sum += future_bank_token_distribution_loss.item()
                future_bank_token_distribution_kl_sum += future_bank_token_distribution_kl
                future_bank_token_distribution_mae_sum += future_bank_token_distribution_mae
                future_bank_token_slot_state_loss_sum += future_bank_token_slot_state_loss.item()
                future_bank_token_slot_state_cosine_sum += future_bank_token_slot_state_cosine
                future_bank_token_slot_state_mse_sum += future_bank_token_slot_state_mse
                future_bank_token_slot_mask_loss_sum += future_bank_token_slot_mask_loss.item()
                future_bank_token_slot_mask_mae_sum += future_bank_token_slot_mask_mae
                future_bank_token_slot_type_loss_sum += future_bank_token_slot_type_loss.item()
                future_bank_token_slot_type_acc_sum += future_bank_token_slot_type_acc
                future_bank_token_slot_zone_loss_sum += future_bank_token_slot_zone_loss.item()
                future_bank_token_slot_zone_acc_sum += future_bank_token_slot_zone_acc
                future_bank_token_slot_source_loss_sum += future_bank_token_slot_source_loss.item()
                future_bank_token_slot_source_acc_sum += future_bank_token_slot_source_acc
                future_world_rollout_aux_loss_sum += future_world_rollout_aux_loss.item()
                future_world_rollout_horizon_sum += rollout_applied_horizons
                surface_precision_sum += surface_metrics["legal_precision"]
                surface_recall_sum += surface_metrics["legal_recall"]
                surface_f1_sum += surface_metrics["legal_f1"]
                surface_count_mae_sum += surface_metrics["legal_count_mae"]
                surface_domain_acc_sum += surface_metrics["decision_domain_acc"]
                surface_phase_acc_sum += surface_metrics["phase_acc"]
    
                hidden_state = recurrent.next_hidden_state
                current_obs_torch = next_obs_torch
                current_teacher_token_encoded = next_teacher_token_encoded
    
        # Backward pass
        self.optimizer.zero_grad(set_to_none=True)
        if self.amp_scaler_enabled:
            self.amp_grad_scaler.scale(total_loss).backward()
            self.amp_grad_scaler.unscale_(self.optimizer)
            torch.nn.utils.clip_grad_norm_(self.network.parameters(), self.max_grad_norm)
            self.amp_grad_scaler.step(self.optimizer)
            self.amp_grad_scaler.update()
        else:
            total_loss.backward()
            torch.nn.utils.clip_grad_norm_(self.network.parameters(), self.max_grad_norm)
            self.optimizer.step()
        self._sync_token_target_encoder()

        policy_terms = float(unroll_steps + 1)
        planner_terms = float(max(planner_q_terms, 1.0))
        planner_objective_terms = float(max(planner_objective_q_terms, 1.0))
        reward_terms = float(max(unroll_steps, 1))
        surface_terms = float(max(unroll_steps, 1))
        latent_reg_terms = float(max(latent_gaussian_reg_terms, 1.0))
        surprise_metric_terms = float(max(surprise_terms, 1.0))
        return {
            "loss/total": total_loss.item(),
            "amp/enabled": 1.0 if self.amp_enabled else 0.0,
            "amp/scaler_enabled": 1.0 if self.amp_scaler_enabled else 0.0,
            "amp/scaler_scale": (
                float(self.amp_grad_scaler.get_scale())
                if bool(self.amp_grad_scaler.is_enabled())
                else 1.0
            ),
            **{f"memory/{key}": value for key, value in self._device_memory_stats().items()},
            "loss/policy": policy_loss_sum / policy_terms,
            "loss/semantic_policy": semantic_policy_loss_sum / policy_terms,
            "loss/latent_policy_target": latent_policy_loss_sum / policy_terms,
            "loss/planner_q": planner_q_loss_sum / planner_terms,
            "loss/planner_objective_q": planner_objective_q_loss_sum / planner_objective_terms,
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
            "loss/jepa_next_hidden": state_consistency_loss_sum / surface_terms,
            "loss/latent_gaussian_reg": latent_gaussian_reg_loss_sum / latent_reg_terms,
            "loss/surprise": surprise_loss_sum / surprise_metric_terms,
            "loss/future_world_aux": future_world_aux_loss_sum / surface_terms,
            "loss/future_bank_state": future_bank_state_loss_sum / surface_terms,
            "loss/future_bank_delta": future_bank_delta_loss_sum / surface_terms,
            "loss/future_bank_occupancy": future_bank_occupancy_loss_sum / surface_terms,
            "loss/future_bank_token_presence": future_bank_token_presence_loss_sum / surface_terms,
            "loss/future_bank_token_distribution": future_bank_token_distribution_loss_sum / surface_terms,
            "loss/future_bank_token_slot_state": future_bank_token_slot_state_loss_sum / surface_terms,
            "loss/future_bank_token_slot_mask": future_bank_token_slot_mask_loss_sum / surface_terms,
            "loss/future_bank_token_slot_type": future_bank_token_slot_type_loss_sum / surface_terms,
            "loss/future_bank_token_slot_zone": future_bank_token_slot_zone_loss_sum / surface_terms,
            "loss/future_bank_token_slot_source": future_bank_token_slot_source_loss_sum / surface_terms,
            "loss/future_world_rollout_aux": future_world_rollout_aux_loss_sum / surface_terms,
            "loss/surface_mask": surface_mask_loss_sum / surface_terms,
            "loss/surface_count": surface_count_loss_sum / surface_terms,
            "loss/surface_domain": surface_domain_loss_sum / surface_terms,
            "loss/surface_phase": surface_phase_loss_sum / surface_terms,
            "metric/latent_policy_distill_kl": latent_policy_distill_kl_sum / policy_terms,
            "metric/latent_policy_distill_top1_agreement": latent_policy_distill_agreement_sum / policy_terms,
            "metric/planner_q_mae": planner_q_mae_sum / planner_terms,
            "metric/planner_objective_q_mae": planner_objective_q_mae_sum / planner_objective_terms,
            "metric/planner_risk_q_mae": planner_risk_q_mae_sum / planner_objective_terms,
            "metric/teacher_policy_entropy": teacher_entropy_sum / policy_terms,
            "metric/student_policy_entropy": student_entropy_sum / policy_terms,
            "metric/state_consistency_cosine": state_consistency_cosine_sum / surface_terms,
            "metric/state_consistency_mse": state_consistency_mse_sum / surface_terms,
            "metric/jepa_next_hidden_cosine": state_consistency_cosine_sum / surface_terms,
            "metric/jepa_next_hidden_mse": state_consistency_mse_sum / surface_terms,
            "metric/latent_reg_mean_abs": latent_reg_mean_abs_sum / latent_reg_terms,
            "metric/latent_reg_var_mean": latent_reg_var_mean_sum / latent_reg_terms,
            "metric/latent_reg_var_std": latent_reg_var_std_sum / latent_reg_terms,
            "metric/latent_reg_cov_offdiag": latent_reg_cov_offdiag_sum / latent_reg_terms,
            "metric/latent_reg_slot_var_mean": latent_reg_slot_var_mean_sum / latent_reg_terms,
            "metric/surprise_target_mean": surprise_target_mean_sum / surprise_metric_terms,
            "metric/surprise_pred_mean": surprise_pred_mean_sum / surprise_metric_terms,
            "metric/surprise_mae": surprise_mae_sum / surprise_metric_terms,
            "metric/surprise_target_offset": surprise_target_offset_sum / surprise_metric_terms,
            "metric/future_bank_state_cosine": future_bank_state_cosine_sum / surface_terms,
            "metric/future_bank_state_mse": future_bank_state_mse_sum / surface_terms,
            "metric/future_bank_delta_mae": future_bank_delta_mae_sum / surface_terms,
            "metric/future_bank_occupancy_mae": future_bank_occupancy_mae_sum / surface_terms,
            "metric/future_bank_token_presence_mae": future_bank_token_presence_mae_sum / surface_terms,
            "metric/future_bank_token_distribution_kl": future_bank_token_distribution_kl_sum / surface_terms,
            "metric/future_bank_token_distribution_mae": future_bank_token_distribution_mae_sum / surface_terms,
            "metric/future_bank_token_slot_state_cosine": future_bank_token_slot_state_cosine_sum / surface_terms,
            "metric/future_bank_token_slot_state_mse": future_bank_token_slot_state_mse_sum / surface_terms,
            "metric/future_bank_token_slot_mask_mae": future_bank_token_slot_mask_mae_sum / surface_terms,
            "metric/future_bank_token_slot_type_acc": future_bank_token_slot_type_acc_sum / surface_terms,
            "metric/future_bank_token_slot_zone_acc": future_bank_token_slot_zone_acc_sum / surface_terms,
            "metric/future_bank_token_slot_source_acc": future_bank_token_slot_source_acc_sum / surface_terms,
            "metric/future_world_rollout_horizons": future_world_rollout_horizon_sum / surface_terms,
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

    def _planner_q_loss(
        self,
        planner_q_logits: torch.Tensor,
        action_indices: torch.Tensor,
        q_targets: torch.Tensor,
        *,
        valid_mask: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, float]:
        batch_size = planner_q_logits.shape[0]
        gather_index = action_indices.long().view(batch_size, 1, 1).expand(-1, 1, planner_q_logits.shape[-1])
        chosen_logits = planner_q_logits.gather(1, gather_index).squeeze(1)
        targets = q_targets.detach().float()
        if valid_mask is not None:
            valid_rows = valid_mask.detach().bool()
        else:
            valid_rows = torch.ones((batch_size,), dtype=torch.bool, device=planner_q_logits.device)
        if not valid_rows.any():
            zero = planner_q_logits.new_zeros(())
            return zero, 0.0
        chosen_logits = chosen_logits[valid_rows]
        targets = targets[valid_rows]
        support_targets = scalar_to_support(
            targets.reshape(-1),
            support_size=self.network.support_size,
        ).to(chosen_logits.device)
        log_probs = torch.log_softmax(chosen_logits, dim=-1)
        loss = -(support_targets * log_probs).sum(dim=-1).mean()
        pred_q = support_to_scalar(chosen_logits, self.network.support_size).reshape(-1)
        mae = (pred_q - targets.reshape(-1)).abs().mean().item()
        return loss, float(mae)

    def _planner_objective_q_loss(
        self,
        planner_q_component_logits: torch.Tensor,
        action_indices: torch.Tensor,
        component_targets: torch.Tensor,
        *,
        valid_mask: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, float, float]:
        batch_size = planner_q_component_logits.shape[0]
        gather_index = action_indices.long().view(batch_size, 1, 1, 1).expand(
            -1,
            1,
            planner_q_component_logits.shape[2],
            planner_q_component_logits.shape[3],
        )
        chosen_logits = planner_q_component_logits.gather(1, gather_index).squeeze(1)
        targets = component_targets.detach().float()
        if valid_mask is not None:
            valid_rows = valid_mask.detach().bool()
        else:
            valid_rows = torch.ones((batch_size,), dtype=torch.bool, device=planner_q_component_logits.device)
        if not valid_rows.any():
            zero = planner_q_component_logits.new_zeros(())
            return zero, 0.0, 0.0
        chosen_logits = chosen_logits[valid_rows]
        targets = targets[valid_rows]
        flat_logits = chosen_logits.reshape(-1, chosen_logits.shape[-1])
        flat_targets = targets.reshape(-1)
        support_targets = scalar_to_support(
            flat_targets,
            support_size=self.network.support_size,
        ).to(flat_logits.device)
        log_probs = torch.log_softmax(flat_logits, dim=-1)
        loss = -(support_targets * log_probs).sum(dim=-1).mean()
        pred_components = support_tensor_to_scalar(
            chosen_logits,
            self.network.support_size,
        )
        component_mae = (pred_components - targets).abs().mean().item()
        risk_dim = min(2, pred_components.shape[-1])
        if risk_dim > 0:
            pred_risk = pred_components[:, :risk_dim].mean(dim=-1)
            target_risk = targets[:, :risk_dim].mean(dim=-1)
            risk_mae = (pred_risk - target_risk).abs().mean().item()
        else:
            risk_mae = 0.0
        return loss, float(component_mae), float(risk_mae)

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

    def _bank_state_consistency_loss(
        self,
        student_bank_states: torch.Tensor,
        teacher_bank_states: torch.Tensor,
    ) -> tuple[torch.Tensor, float, float]:
        """Align predicted future world-bank states with next-observation bank summaries."""
        teacher = teacher_bank_states.detach()
        valid_mask = teacher.abs().sum(dim=-1) > 1e-6

        losses: list[torch.Tensor] = []
        cosine_value = 0.0
        mse_value = 0.0

        if valid_mask.any():
            student_valid = student_bank_states[valid_mask]
            teacher_valid = teacher[valid_mask]
            student_norm = torch.nn.functional.normalize(student_valid, dim=-1)
            teacher_norm = torch.nn.functional.normalize(teacher_valid, dim=-1)
            cosine = (student_norm * teacher_norm).sum(dim=-1)
            mse = torch.nn.functional.mse_loss(student_norm, teacher_norm, reduction="none").mean(dim=-1)
            losses.append((1.0 - cosine).mean() + 0.25 * mse.mean())
            cosine_value = float(cosine.mean().item())
            mse_value = float(mse.mean().item())

        empty_mask = ~valid_mask
        if empty_mask.any():
            losses.append(0.1 * student_bank_states[empty_mask].pow(2).mean())

        if not losses:
            zero = student_bank_states.new_zeros(())
            return zero, 0.0, 0.0

        return sum(losses), cosine_value, mse_value

    def _bank_delta_consistency_loss(
        self,
        student_bank_delta: torch.Tensor,
        target_bank_delta: torch.Tensor,
    ) -> tuple[torch.Tensor, float]:
        target = target_bank_delta.detach()
        valid_mask = target.abs().sum(dim=-1) > 1e-6
        if not valid_mask.any():
            zero = student_bank_delta.new_zeros(())
            return zero, 0.0
        student_valid = student_bank_delta[valid_mask]
        target_valid = target[valid_mask]
        loss = torch.nn.functional.smooth_l1_loss(student_valid, target_valid)
        mae = (student_valid - target_valid).abs().mean().item()
        return loss, float(mae)

    def _bank_occupancy_loss(
        self,
        logits: torch.Tensor,
        targets: torch.Tensor,
    ) -> tuple[torch.Tensor, float]:
        target_values = torch.clamp(targets.detach().float(), min=0.0, max=1.0)
        loss = torch.nn.functional.binary_cross_entropy_with_logits(logits, target_values)
        probs = torch.sigmoid(logits)
        mae = (probs - target_values).abs().mean().item()
        return loss, float(mae)

    def _bank_token_presence_loss(
        self,
        logits: torch.Tensor,
        targets: torch.Tensor,
    ) -> tuple[torch.Tensor, float]:
        target_values = torch.clamp(targets.detach().float(), min=0.0, max=1.0)
        loss = torch.nn.functional.binary_cross_entropy_with_logits(logits, target_values)
        probs = torch.sigmoid(logits)
        mae = (probs - target_values).abs().mean().item()
        return loss, float(mae)

    def _bank_token_distribution_loss(
        self,
        logits: torch.Tensor,
        targets: torch.Tensor,
    ) -> tuple[torch.Tensor, float, float]:
        target_values = targets.detach().float()
        valid_rows = target_values.sum(dim=-1) > 1e-6
        if not valid_rows.any():
            zero = logits.new_zeros(())
            return zero, 0.0, 0.0
        logits_valid = logits[valid_rows]
        target_valid = target_values[valid_rows]
        log_probs = torch.nn.functional.log_softmax(logits_valid, dim=-1)
        loss = torch.nn.functional.kl_div(log_probs, target_valid, reduction="batchmean")
        probs = log_probs.exp()
        mae = (probs - target_valid).abs().mean().item()
        return loss, float(loss.item()), float(mae)

    def _bank_token_slot_state_loss(
        self,
        student_slot_states: torch.Tensor,
        teacher_slot_states: torch.Tensor,
        slot_mask: torch.Tensor,
    ) -> tuple[torch.Tensor, float, float]:
        teacher = teacher_slot_states.detach()
        valid_mask = slot_mask.detach().bool()

        losses: list[torch.Tensor] = []
        cosine_value = 0.0
        mse_value = 0.0

        if valid_mask.any():
            student_valid = student_slot_states[valid_mask]
            teacher_valid = teacher[valid_mask]
            student_norm = torch.nn.functional.normalize(student_valid, dim=-1)
            teacher_norm = torch.nn.functional.normalize(teacher_valid, dim=-1)
            cosine = (student_norm * teacher_norm).sum(dim=-1)
            mse = torch.nn.functional.mse_loss(student_norm, teacher_norm, reduction="none").mean(dim=-1)
            losses.append((1.0 - cosine).mean() + 0.25 * mse.mean())
            cosine_value = float(cosine.mean().item())
            mse_value = float(mse.mean().item())

        invalid_mask = ~valid_mask
        if invalid_mask.any():
            losses.append(0.05 * student_slot_states[invalid_mask].pow(2).mean())

        if not losses:
            zero = student_slot_states.new_zeros(())
            return zero, 0.0, 0.0

        return sum(losses), cosine_value, mse_value

    def _bank_token_slot_mask_loss(
        self,
        logits: torch.Tensor,
        targets: torch.Tensor,
    ) -> tuple[torch.Tensor, float]:
        target_values = torch.clamp(targets.detach().float(), min=0.0, max=1.0)
        loss = torch.nn.functional.binary_cross_entropy_with_logits(logits, target_values)
        probs = torch.sigmoid(logits)
        mae = (probs - target_values).abs().mean().item()
        return loss, float(mae)

    def _bank_token_slot_type_loss(
        self,
        logits: torch.Tensor,
        targets: torch.Tensor,
        slot_mask: torch.Tensor,
    ) -> tuple[torch.Tensor, float]:
        valid_mask = slot_mask.detach().bool()
        if not valid_mask.any():
            zero = logits.new_zeros(())
            return zero, 0.0
        logits_valid = logits[valid_mask]
        target_valid = targets.detach().long()[valid_mask]
        loss = torch.nn.functional.cross_entropy(logits_valid, target_valid)
        acc = (logits_valid.argmax(dim=-1) == target_valid).float().mean().item()
        return loss, float(acc)

    def _bank_token_slot_zone_loss(
        self,
        logits: torch.Tensor,
        targets: torch.Tensor,
        slot_mask: torch.Tensor,
    ) -> tuple[torch.Tensor, float]:
        valid_mask = slot_mask.detach().bool()
        if not valid_mask.any():
            zero = logits.new_zeros(())
            return zero, 0.0
        logits_valid = logits[valid_mask]
        target_valid = targets.detach().long()[valid_mask]
        loss = torch.nn.functional.cross_entropy(logits_valid, target_valid)
        acc = (logits_valid.argmax(dim=-1) == target_valid).float().mean().item()
        return loss, float(acc)

    def _bank_token_slot_source_loss(
        self,
        logits: torch.Tensor,
        targets: torch.Tensor,
        slot_mask: torch.Tensor,
    ) -> tuple[torch.Tensor, float]:
        valid_mask = slot_mask.detach().bool()
        if not valid_mask.any():
            zero = logits.new_zeros(())
            return zero, 0.0
        logits_valid = logits[valid_mask]
        target_valid = targets.detach().long()[valid_mask]
        loss = torch.nn.functional.cross_entropy(logits_valid, target_valid)
        acc = (logits_valid.argmax(dim=-1) == target_valid).float().mean().item()
        return loss, float(acc)

    def _future_world_aux_terms(
        self,
        recurrent: RecurrentMuZeroOutput,
        *,
        zero_ref: torch.Tensor,
    ) -> dict[str, torch.Tensor | float]:
        zero = zero_ref.new_zeros(())

        if (
            self.network.is_token_mode
            and self.future_bank_state_weight > 0.0
            and recurrent.next_world_bank_state_pred is not None
            and recurrent.next_world_bank_state_target is not None
        ):
            future_bank_state_loss, future_bank_state_cosine, future_bank_state_mse = self._bank_state_consistency_loss(
                recurrent.next_world_bank_state_pred,
                recurrent.next_world_bank_state_target,
            )
        else:
            future_bank_state_loss = zero
            future_bank_state_cosine = 0.0
            future_bank_state_mse = 0.0

        if (
            self.network.is_token_mode
            and self.future_bank_delta_weight > 0.0
            and recurrent.next_world_bank_delta_pred is not None
            and recurrent.next_world_bank_delta_target is not None
        ):
            future_bank_delta_loss, future_bank_delta_mae = self._bank_delta_consistency_loss(
                recurrent.next_world_bank_delta_pred,
                recurrent.next_world_bank_delta_target,
            )
        else:
            future_bank_delta_loss = zero
            future_bank_delta_mae = 0.0

        if (
            self.network.is_token_mode
            and self.future_bank_occupancy_weight > 0.0
            and recurrent.next_world_bank_occupancy_logits is not None
            and recurrent.next_world_bank_occupancy_target is not None
        ):
            future_bank_occupancy_loss, future_bank_occupancy_mae = self._bank_occupancy_loss(
                recurrent.next_world_bank_occupancy_logits,
                recurrent.next_world_bank_occupancy_target,
            )
        else:
            future_bank_occupancy_loss = zero
            future_bank_occupancy_mae = 0.0

        if (
            self.network.is_token_mode
            and self.future_bank_token_presence_weight > 0.0
            and recurrent.next_world_bank_token_presence_logits is not None
            and recurrent.next_world_bank_token_presence_target is not None
        ):
            future_bank_token_presence_loss, future_bank_token_presence_mae = self._bank_token_presence_loss(
                recurrent.next_world_bank_token_presence_logits,
                recurrent.next_world_bank_token_presence_target,
            )
        else:
            future_bank_token_presence_loss = zero
            future_bank_token_presence_mae = 0.0

        if (
            self.network.is_token_mode
            and self.future_bank_token_distribution_weight > 0.0
            and recurrent.next_world_bank_token_distribution_logits is not None
            and recurrent.next_world_bank_token_distribution_target is not None
        ):
            (
                future_bank_token_distribution_loss,
                future_bank_token_distribution_kl,
                future_bank_token_distribution_mae,
            ) = self._bank_token_distribution_loss(
                recurrent.next_world_bank_token_distribution_logits,
                recurrent.next_world_bank_token_distribution_target,
            )
        else:
            future_bank_token_distribution_loss = zero
            future_bank_token_distribution_kl = 0.0
            future_bank_token_distribution_mae = 0.0

        if (
            self.network.is_token_mode
            and self.future_bank_token_slot_state_weight > 0.0
            and recurrent.next_world_bank_token_slot_state_pred is not None
            and recurrent.next_world_bank_token_slot_state_target is not None
            and recurrent.next_world_bank_token_slot_mask_target is not None
        ):
            (
                future_bank_token_slot_state_loss,
                future_bank_token_slot_state_cosine,
                future_bank_token_slot_state_mse,
            ) = self._bank_token_slot_state_loss(
                recurrent.next_world_bank_token_slot_state_pred,
                recurrent.next_world_bank_token_slot_state_target,
                recurrent.next_world_bank_token_slot_mask_target,
            )
        else:
            future_bank_token_slot_state_loss = zero
            future_bank_token_slot_state_cosine = 0.0
            future_bank_token_slot_state_mse = 0.0

        if (
            self.network.is_token_mode
            and self.future_bank_token_slot_mask_weight > 0.0
            and recurrent.next_world_bank_token_slot_mask_logits is not None
            and recurrent.next_world_bank_token_slot_mask_target is not None
        ):
            future_bank_token_slot_mask_loss, future_bank_token_slot_mask_mae = self._bank_token_slot_mask_loss(
                recurrent.next_world_bank_token_slot_mask_logits,
                recurrent.next_world_bank_token_slot_mask_target,
            )
        else:
            future_bank_token_slot_mask_loss = zero
            future_bank_token_slot_mask_mae = 0.0

        if (
            self.network.is_token_mode
            and self.future_bank_token_slot_type_weight > 0.0
            and recurrent.next_world_bank_token_slot_type_logits is not None
            and recurrent.next_world_bank_token_slot_type_target is not None
            and recurrent.next_world_bank_token_slot_mask_target is not None
        ):
            future_bank_token_slot_type_loss, future_bank_token_slot_type_acc = self._bank_token_slot_type_loss(
                recurrent.next_world_bank_token_slot_type_logits,
                recurrent.next_world_bank_token_slot_type_target,
                recurrent.next_world_bank_token_slot_mask_target,
            )
        else:
            future_bank_token_slot_type_loss = zero
            future_bank_token_slot_type_acc = 0.0

        if (
            self.network.is_token_mode
            and self.future_bank_token_slot_zone_weight > 0.0
            and recurrent.next_world_bank_token_slot_zone_logits is not None
            and recurrent.next_world_bank_token_slot_zone_target is not None
            and recurrent.next_world_bank_token_slot_mask_target is not None
        ):
            future_bank_token_slot_zone_loss, future_bank_token_slot_zone_acc = self._bank_token_slot_zone_loss(
                recurrent.next_world_bank_token_slot_zone_logits,
                recurrent.next_world_bank_token_slot_zone_target,
                recurrent.next_world_bank_token_slot_mask_target,
            )
        else:
            future_bank_token_slot_zone_loss = zero
            future_bank_token_slot_zone_acc = 0.0

        if (
            self.network.is_token_mode
            and self.future_bank_token_slot_source_weight > 0.0
            and recurrent.next_world_bank_token_slot_source_logits is not None
            and recurrent.next_world_bank_token_slot_source_target is not None
            and recurrent.next_world_bank_token_slot_mask_target is not None
        ):
            future_bank_token_slot_source_loss, future_bank_token_slot_source_acc = self._bank_token_slot_source_loss(
                recurrent.next_world_bank_token_slot_source_logits,
                recurrent.next_world_bank_token_slot_source_target,
                recurrent.next_world_bank_token_slot_mask_target,
            )
        else:
            future_bank_token_slot_source_loss = zero
            future_bank_token_slot_source_acc = 0.0

        loss_total = (
            self.future_bank_state_weight * future_bank_state_loss
            + self.future_bank_delta_weight * future_bank_delta_loss
            + self.future_bank_occupancy_weight * future_bank_occupancy_loss
            + self.future_bank_token_presence_weight * future_bank_token_presence_loss
            + self.future_bank_token_distribution_weight * future_bank_token_distribution_loss
            + self.future_bank_token_slot_state_weight * future_bank_token_slot_state_loss
            + self.future_bank_token_slot_mask_weight * future_bank_token_slot_mask_loss
            + self.future_bank_token_slot_type_weight * future_bank_token_slot_type_loss
            + self.future_bank_token_slot_zone_weight * future_bank_token_slot_zone_loss
            + self.future_bank_token_slot_source_weight * future_bank_token_slot_source_loss
        )

        return {
            "loss_total": loss_total,
            "future_bank_state_loss": future_bank_state_loss,
            "future_bank_state_cosine": future_bank_state_cosine,
            "future_bank_state_mse": future_bank_state_mse,
            "future_bank_delta_loss": future_bank_delta_loss,
            "future_bank_delta_mae": future_bank_delta_mae,
            "future_bank_occupancy_loss": future_bank_occupancy_loss,
            "future_bank_occupancy_mae": future_bank_occupancy_mae,
            "future_bank_token_presence_loss": future_bank_token_presence_loss,
            "future_bank_token_presence_mae": future_bank_token_presence_mae,
            "future_bank_token_distribution_loss": future_bank_token_distribution_loss,
            "future_bank_token_distribution_kl": future_bank_token_distribution_kl,
            "future_bank_token_distribution_mae": future_bank_token_distribution_mae,
            "future_bank_token_slot_state_loss": future_bank_token_slot_state_loss,
            "future_bank_token_slot_state_cosine": future_bank_token_slot_state_cosine,
            "future_bank_token_slot_state_mse": future_bank_token_slot_state_mse,
            "future_bank_token_slot_mask_loss": future_bank_token_slot_mask_loss,
            "future_bank_token_slot_mask_mae": future_bank_token_slot_mask_mae,
            "future_bank_token_slot_type_loss": future_bank_token_slot_type_loss,
            "future_bank_token_slot_type_acc": future_bank_token_slot_type_acc,
            "future_bank_token_slot_zone_loss": future_bank_token_slot_zone_loss,
            "future_bank_token_slot_zone_acc": future_bank_token_slot_zone_acc,
            "future_bank_token_slot_source_loss": future_bank_token_slot_source_loss,
            "future_bank_token_slot_source_acc": future_bank_token_slot_source_acc,
        }

    def _surface_mask_loss(
        self,
        logits: torch.Tensor,
        target_mask: torch.Tensor,
    ) -> torch.Tensor:
        targets = target_mask.float()
        # Adaptive pos_weight: penalise false-positives proportionally to
        # the actual legal/illegal ratio in this batch.  Typical combat has
        # ~8 legal out of 80 slots, so neg/pos 鈮?9 鈫?false positives get
        # 9脳 the gradient of false negatives.  Clamped to [1, 15].
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
        if self.token_target_encoder is not None:
            torch.save(self.token_target_encoder.state_dict(), checkpoint_path / "token_target_encoder.pt")
        torch.save(self.optimizer.state_dict(), checkpoint_path / "optimizer.pt")
        if self.amp_grad_scaler is not None and bool(self.amp_grad_scaler.is_enabled()):
            torch.save(self.amp_grad_scaler.state_dict(), checkpoint_path / "amp_scaler.pt")
        with (checkpoint_path / "replay_buffer.pkl").open("wb") as handle:
            pickle.dump(self.buffer.state_dict(), handle, protocol=pickle.HIGHEST_PROTOCOL)

        metadata = {
            "format": "muzero-v1",
            "total_steps": int(self.total_steps),
            "episode_count": int(self.episode_count),
            "buffer_size": len(self.buffer),
            "replay_buffer_saved": True,
            "token_target_encoder_saved": bool(self.token_target_encoder is not None),
            "mixed_precision": self.mixed_precision,
            "amp_enabled": bool(self.amp_enabled),
            "amp_device_type": str(self.amp_device_type),
            "amp_dtype": str(self.amp_dtype),
            "amp_scaler_enabled": bool(self.amp_scaler_enabled),
            "pytorch_alloc_conf": os.environ.get("PYTORCH_ALLOC_CONF"),
            "pytorch_hip_alloc_conf": os.environ.get("PYTORCH_HIP_ALLOC_CONF"),
            "network": self.network.constructor_spec() if hasattr(self.network, "constructor_spec") else {},
            "env": {
                "mode": "combat_sandbox" if hasattr(self.env, "sandbox_supports_potions") else "full_run",
                "potion_mechanics_available": bool(getattr(self.env, "sandbox_supports_potions", True)),
                "obs_mode": getattr(self.network, "obs_mode", "dense_v2"),
                "model_arch": getattr(self.network, "model_arch", "dense_v1"),
            },
            "latent_policy_distill_weight": float(self.latent_policy_distill_weight),
            "latent_policy_target_weight": float(self.latent_policy_target_weight),
            "planner_q_loss_weight": float(self.planner_q_loss_weight),
            "planner_objective_q_loss_weight": float(self.planner_objective_q_loss_weight),
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
            "future_world_aux_weight": float(self.future_world_aux_weight),
            "future_bank_state_weight": float(self.future_bank_state_weight),
            "future_bank_delta_weight": float(self.future_bank_delta_weight),
            "future_bank_occupancy_weight": float(self.future_bank_occupancy_weight),
            "future_bank_token_presence_weight": float(self.future_bank_token_presence_weight),
            "future_bank_token_distribution_weight": float(self.future_bank_token_distribution_weight),
            "future_bank_token_slot_state_weight": float(self.future_bank_token_slot_state_weight),
            "future_bank_token_slot_mask_weight": float(self.future_bank_token_slot_mask_weight),
            "future_bank_token_slot_type_weight": float(self.future_bank_token_slot_type_weight),
            "future_bank_token_slot_zone_weight": float(self.future_bank_token_slot_zone_weight),
            "future_bank_token_slot_source_weight": float(self.future_bank_token_slot_source_weight),
            "token_teacher_ema_decay": float(self.token_teacher_ema_decay),
            "future_world_rollout_weight": float(self.future_world_rollout_weight),
            "future_world_rollout_steps": int(self.future_world_rollout_steps),
            "future_world_rollout_decay": float(self.future_world_rollout_decay),
            "latent_gaussian_reg_weight": float(self.latent_gaussian_reg_weight),
            "latent_gaussian_reg_projections": int(self.latent_gaussian_reg_projections),
            "latent_gaussian_reg_slot_weight": float(self.latent_gaussian_reg_slot_weight),
            "latent_gaussian_reg_dynamics_weight": float(self.latent_gaussian_reg_dynamics_weight),
            "latent_gaussian_reg_cov_weight": float(self.latent_gaussian_reg_cov_weight),
            "surprise_loss_weight": float(self.surprise_loss_weight),
            "surprise_hidden_scale": float(self.surprise_hidden_scale),
            "surprise_surface_scale": float(self.surprise_surface_scale),
            "surprise_future_aux_scale": float(self.surprise_future_aux_scale),
            "surprise_target_cap": float(self.surprise_target_cap),
            "surface_mask_weight": float(self.surface_mask_weight),
            "surface_count_weight": float(self.surface_count_weight),
            "surface_domain_weight": float(self.surface_domain_weight),
            "surface_phase_weight": float(self.surface_phase_weight),
            "combat_direct_policy": bool(self.combat_direct_policy),
            "combat_rollout_q_blend": float(self.combat_rollout_q_blend),
            "combat_rollout_objective_q_blend": float(self.combat_rollout_objective_q_blend),
            "combat_rollout_risk_blend": float(self.combat_rollout_risk_blend),
            "combat_rollout_steps": int(self.combat_rollout_steps),
            "combat_rollout_beam_width": int(self.combat_rollout_beam_width),
            "combat_rollout_legal_logit_scale": float(self.combat_rollout_legal_logit_scale),
            "combat_rollout_uncertainty_blend": float(self.combat_rollout_uncertainty_blend),
            "combat_rollout_uncertainty_surprise_weight": float(self.combat_rollout_uncertainty_surprise_weight),
            "combat_rollout_uncertainty_surface_weight": float(self.combat_rollout_uncertainty_surface_weight),
            "combat_rollout_uncertainty_latent_weight": float(self.combat_rollout_uncertainty_latent_weight),
            "combat_rollout_uncertainty_disagreement_weight": float(self.combat_rollout_uncertainty_disagreement_weight),
            "combat_rollout_continuation_uncertainty_penalty": float(self.combat_rollout_continuation_uncertainty_penalty),
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
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--unroll-steps", type=int, default=3)
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
    parser.add_argument("--resume-without-optimizer", action="store_true", default=False,
                        help="Resume model weights only; re-initialize the optimizer (Adam moments). "
                             "Use this when the aux-target schema changed mid-run and the old momentum "
                             "is steering away from the new objective.")
    parser.add_argument("--latent-policy-distill-weight", type=float, default=0.25,
                        help="Distill latent search policy toward observation-conditioned policy.")
    parser.add_argument("--latent-policy-target-weight", type=float, default=0.5,
                        help="Direct supervised target loss on latent-policy logits.")
    parser.add_argument("--planner-q-loss-weight", type=float, default=0.75,
                        help="Loss weight for action-conditioned planner Q / lookahead value supervision.")
    parser.add_argument("--planner-objective-q-loss-weight", type=float, default=0.75,
                        help="Loss weight for multi-head planner Q decomposition (survival/hp/build/resource).")
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
    parser.add_argument("--jepa-next-hidden-weight", type=float, default=None,
                        help="Alias/override for --state-consistency-weight; names the JEPA next-hidden prediction loss explicitly.")
    parser.add_argument("--future-world-aux-weight", type=float, default=1.0,
                        help="Global multiplier applied to the full future token-world auxiliary supervision stack.")
    parser.add_argument("--future-bank-state-weight", type=float, default=1.0,
                        help="Weight for token-mode future world-bank consistency between latent rollout and next observation bank summaries.")
    parser.add_argument("--future-bank-delta-weight", type=float, default=0.5,
                        help="Weight for future world-bank delta prediction relative to the current latent bank reference.")
    parser.add_argument("--future-bank-occupancy-weight", type=float, default=0.25,
                        help="Weight for future world-bank occupancy prediction.")
    parser.add_argument("--future-bank-token-presence-weight", type=float, default=0.25,
                        help="Weight for future world-bank token-family presence prediction.")
    parser.add_argument("--future-bank-token-distribution-weight", type=float, default=0.2,
                        help="Weight for future world-bank token-family distribution prediction.")
    parser.add_argument("--future-bank-token-slot-state-weight", type=float, default=0.5,
                        help="Weight for future world-bank top-k token-slot state reconstruction.")
    parser.add_argument("--future-bank-token-slot-mask-weight", type=float, default=0.15,
                        help="Weight for future world-bank top-k token-slot occupancy prediction.")
    parser.add_argument("--future-bank-token-slot-type-weight", type=float, default=0.2,
                        help="Weight for future world-bank top-k token-slot type prediction.")
    parser.add_argument("--future-bank-token-slot-zone-weight", type=float, default=0.15,
                        help="Weight for future world-bank top-k token-slot zone / pile prediction.")
    parser.add_argument("--future-bank-token-slot-source-weight", type=float, default=0.2,
                        help="Weight for future world-bank token migration/copy source prediction.")
    parser.add_argument("--future-world-rollout-weight", type=float, default=0.35,
                        help="Extra multiplier for multi-step future-world rollout supervision beyond the one-step target.")
    parser.add_argument("--future-world-rollout-steps", type=int, default=2,
                        help="Number of additional future rollout horizons (beyond t+1) supervised inside training.")
    parser.add_argument("--future-world-rollout-decay", type=float, default=0.7,
                        help="Geometric decay applied across extra future rollout horizons.")
    parser.add_argument("--token-teacher-ema-decay", type=float, default=0.995,
                        help="EMA decay for the token-world target encoder used for future-world teacher targets.")
    parser.add_argument("--latent-gaussian-reg-weight", type=float, default=0.005,
                        help="Global SIGReg/JEPA-style Gaussian latent regularization weight.")
    parser.add_argument("--latent-gaussian-reg-projections", type=int, default=64,
                        help="Random projection count for Gaussian latent regularization.")
    parser.add_argument("--latent-gaussian-reg-slot-weight", type=float, default=0.25,
                        help="Extra slot-level latent Gaussian regularization weight in token_memory_v1.")
    parser.add_argument("--latent-gaussian-reg-dynamics-weight", type=float, default=0.5,
                        help="Multiplier for latent Gaussian regularization on dynamics-produced next hidden states.")
    parser.add_argument("--latent-gaussian-reg-cov-weight", type=float, default=0.05,
                        help="Off-diagonal covariance penalty weight inside latent Gaussian regularization.")
    parser.add_argument("--surprise-loss-weight", type=float, default=0.25,
                        help="Loss weight for trainable dynamics surprise / latent prediction error head.")
    parser.add_argument("--surprise-hidden-scale", type=float, default=100.0,
                        help="Scale applied to normalized JEPA hidden prediction error before surprise supervision.")
    parser.add_argument("--surprise-surface-scale", type=float, default=5.0,
                        help="Batch-level surface prediction error scale added to surprise supervision target.")
    parser.add_argument("--surprise-future-aux-scale", type=float, default=1.0,
                        help="Batch-level future-world auxiliary error scale added to surprise supervision target.")
    parser.add_argument("--surprise-target-cap", type=float, default=50.0,
                        help="Optional cap on the final surprise target after hidden/surface/future-world terms; 0 disables.")
    parser.add_argument("--surface-mask-weight", type=float, default=1.5,
                        help="Weight for next-step legal-mask prediction loss.")
    parser.add_argument("--surface-count-weight", type=float, default=0.25,
                        help="Weight for predicted legal-count calibration loss.")
    parser.add_argument("--surface-domain-weight", type=float, default=0.15,
                        help="Weight for next-step decision-domain prediction loss.")
    parser.add_argument("--surface-phase-weight", type=float, default=0.1,
                        help="Weight for next-step phase prediction loss.")
    parser.add_argument("--combat-direct-policy", action="store_true", default=False,
                        help="Search-free combat self-play: bypass combat MCTS and sample from policy logits blended with latent rollout Q/uncertainty.")
    parser.add_argument("--combat-policy-mode", type=str, default="auto", choices=["auto", "direct", "mcts"],
                        help="Combat policy selector. auto enables search-free direct policy for token combat sandbox, direct always bypasses combat MCTS, mcts forces legacy combat MCTS.")
    parser.add_argument("--combat-rollout-q-blend", type=float, default=0.75,
                        help="Extra direct-policy blend weight for explicit one-step rollout Q bias in combat.")
    parser.add_argument("--combat-rollout-objective-q-blend", type=float, default=0.5,
                        help="Extra direct-policy blend weight for explicit one-step rollout objective-Q bias in combat.")
    parser.add_argument("--combat-rollout-risk-blend", type=float, default=0.35,
                        help="Extra direct-policy blend weight for explicit one-step rollout survival/HP risk bias in combat.")
    parser.add_argument("--combat-rollout-steps", type=int, default=2,
                        help="Search-free rollout depth for combat direct-policy. 1 = one-step Q, 2+ adds latent continuation.")
    parser.add_argument("--combat-rollout-beam-width", type=int, default=2,
                        help="Per-root latent beam width used by multi-step combat direct-policy rollout.")
    parser.add_argument("--combat-rollout-legal-logit-scale", type=float, default=0.75,
                        help="Scale applied to predicted next-step legal-mask logits when ranking latent continuation actions.")
    parser.add_argument("--combat-rollout-uncertainty-blend", type=float, default=0.35,
                        help="Direct-policy penalty weight for rollout surprise/uncertainty bias.")
    parser.add_argument("--combat-rollout-uncertainty-surprise-weight", type=float, default=1.0,
                        help="Internal planner uncertainty weight for trainable dynamics surprise.")
    parser.add_argument("--combat-rollout-uncertainty-surface-weight", type=float, default=0.10,
                        help="Internal planner uncertainty weight for next-surface entropy.")
    parser.add_argument("--combat-rollout-uncertainty-latent-weight", type=float, default=0.05,
                        help="Internal planner uncertainty weight for latent norm drift.")
    parser.add_argument("--combat-rollout-uncertainty-disagreement-weight", type=float, default=0.25,
                        help="Internal planner uncertainty weight for branch Q disagreement.")
    parser.add_argument("--combat-rollout-continuation-uncertainty-penalty", type=float, default=0.25,
                        help="Penalty applied while pruning latent continuation beams.")
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
    parser.add_argument("--obs-mode", type=str, default="dense_v2",
                        choices=["dense_v2", "token_v3"],
                        help="Observation encoder mode. token_v3 enables token-world MuZero inputs.")
    parser.add_argument("--model-arch", type=str, default="dense_v1",
                        choices=["dense_v1", "token_memory_v1"],
                        help="MuZero model path. token_memory_v1 uses token-world encoder + latent memory slots.")
    parser.add_argument("--token-d-model", type=int, default=128,
                        help="Token-world backbone width for token_memory_v1.")
    parser.add_argument("--token-n-heads", type=int, default=4,
                        help="Attention heads for token_memory_v1.")
    parser.add_argument("--token-ffn-dim", type=int, default=512,
                        help="FFN width for token_memory_v1.")
    parser.add_argument("--token-world-layers", type=int, default=4,
                        help="World self-attention layers for token_memory_v1.")
    parser.add_argument("--token-local-layers", type=int, default=1,
                        help="Candidate-local self-attention layers for token_memory_v1.")
    parser.add_argument("--token-decoder-layers", type=int, default=2,
                        help="Banked candidate-to-world decoder layers for token_memory_v1.")
    parser.add_argument("--token-candidate-set-layers", type=int, default=1,
                        help="Post-bank candidate-set self-attention layers for token_memory_v1.")
    parser.add_argument("--token-memory-slots", type=int, default=8,
                        help="Latent memory slots for token_memory_v1; hidden_dim becomes token_d_model * token_memory_slots.")
    parser.add_argument("--token-bank-token-slots", type=int, default=4,
                        help="Top-k per-bank world token slots used for token_memory_v1 future-world reconstruction.")
    parser.add_argument("--token-world-bank-top-k", type=int, default=3,
                        help="Top-k world banks each candidate may attend in token_memory_v1.")
    parser.add_argument("--token-slot-source-same-bank-bias", type=float, default=0.35,
                        help="Transport prior logit bias for copying from the same world bank.")
    parser.add_argument("--token-slot-source-same-slot-bias", type=float, default=0.2,
                        help="Transport prior logit bias for copying from the same slot position inside a bank.")
    parser.add_argument("--token-slot-source-type-match-scale", type=float, default=0.5,
                        help="Transport prior scale for preferring source slots whose current type matches the predicted future type.")
    parser.add_argument("--token-slot-source-zone-transport-scale", type=float, default=0.35,
                        help="Transport prior scale for zone / pile-aware copy transitions such as draw->hand and hand->discard.")
    parser.add_argument("--token-internal-planner-blend", type=float, default=0.7,
                        help="Blend weight for the token-mode internal action-imagination planner added on top of candidate policy logits.")
    parser.add_argument("--token-internal-planner-q-blend", type=float, default=0.5,
                        help="Blend weight for planner Q / lookahead value bias injected into token-mode policy logits.")
    parser.add_argument("--token-internal-planner-objective-q-blend", type=float, default=0.35,
                        help="Blend weight for multi-objective planner Q scalarization injected into token-mode policy logits.")
    parser.add_argument("--token-internal-planner-risk-blend", type=float, default=0.25,
                        help="Blend weight for planner survival/HP risk bias injected into token-mode policy logits.")
    parser.add_argument("--token-dropout", type=float, default=0.0,
                        help="Dropout for token_memory_v1 attention blocks.")
    parser.add_argument("--action-rollout-buckets", type=str, default="8,16,32,64,80",
                        help="Comma-separated bucket sizes for search-free action_rollout_planner padding; larger counts round to multiples of the largest bucket.")

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
    parser.add_argument(
        "--mixed-precision",
        type=str,
        default="auto",
        choices=["auto", "off", "fp32", "bf16", "fp16"],
        help=(
            "Training/inference autocast mode. auto uses CUDA bf16 when supported, "
            "otherwise CUDA fp16+GradScaler; CPU auto stays fp32/off."
        ),
    )
    parser.add_argument(
        "--amp-init-scale",
        type=float,
        default=65536.0,
        help="Initial GradScaler scale used for fp16 mixed precision.",
    )
    parser.add_argument(
        "--activation-checkpointing",
        type=str,
        default="auto",
        choices=["auto", "on", "off"],
        help=(
            "Activation checkpointing for token-memory MuZero. "
            "auto enables it for token_v3/token_memory_v1 training and keeps dense models off."
        ),
    )

    # Discount factors
    parser.add_argument("--discount", type=float, default=0.997)
    parser.add_argument("--n-step-return", type=int, default=10)

    args = parser.parse_args()
    activation_checkpointing_mode = str(args.activation_checkpointing or "auto").strip().lower()
    args.activation_checkpointing_enabled = bool(
        activation_checkpointing_mode == "on"
        or (
            activation_checkpointing_mode == "auto"
            and (
                str(args.obs_mode).strip().lower() == "token_v3"
                or str(args.model_arch).strip().lower() == "token_memory_v1"
            )
        )
    )
    if args.jepa_next_hidden_weight is not None:
        args.state_consistency_weight = float(args.jepa_next_hidden_weight)
    if args.combat_policy_mode == "direct":
        args.combat_direct_policy = True
    elif args.combat_policy_mode == "mcts":
        args.combat_direct_policy = False
    elif not args.combat_direct_policy and args.combat_sandbox and args.model_arch == "token_memory_v1":
        args.combat_direct_policy = True
    args.device = resolve_torch_device(args.device)
    args.log_dir = normalize_path_str(args.log_dir) or args.log_dir
    args.checkpoint_dir = normalize_path_str(args.checkpoint_dir) or args.checkpoint_dir
    args.resume_from = normalize_path_str(args.resume_from)
    args.combat_snapshot_dataset = normalize_path_str(args.combat_snapshot_dataset)
    args.session_file = normalize_path_str(args.session_file)
    if args.obs_mode == "token_v3" and args.model_arch != "token_memory_v1":
        raise ValueError("--obs-mode token_v3 requires --model-arch token_memory_v1.")
    if args.model_arch == "token_memory_v1" and args.obs_mode != "token_v3":
        raise ValueError("--model-arch token_memory_v1 requires --obs-mode token_v3.")

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
            obs_mode=args.obs_mode,
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
    network_kwargs = {
        "obs_mode": args.obs_mode,
        "model_arch": args.model_arch,
        "token_d_model": args.token_d_model,
        "token_n_heads": args.token_n_heads,
        "token_ffn_dim": args.token_ffn_dim,
        "token_world_layers": args.token_world_layers,
        "token_local_layers": args.token_local_layers,
        "token_decoder_layers": args.token_decoder_layers,
        "token_candidate_set_layers": args.token_candidate_set_layers,
        "token_memory_slots": args.token_memory_slots,
        "token_bank_token_slots": args.token_bank_token_slots,
        "token_world_bank_top_k": args.token_world_bank_top_k,
        "token_slot_source_same_bank_bias": args.token_slot_source_same_bank_bias,
        "token_slot_source_same_slot_bias": args.token_slot_source_same_slot_bias,
        "token_slot_source_type_match_scale": args.token_slot_source_type_match_scale,
        "token_slot_source_zone_transport_scale": args.token_slot_source_zone_transport_scale,
        "token_internal_planner_blend": args.token_internal_planner_blend,
        "token_internal_planner_q_blend": args.token_internal_planner_q_blend,
        "token_internal_planner_objective_q_blend": args.token_internal_planner_objective_q_blend,
        "token_internal_planner_risk_blend": args.token_internal_planner_risk_blend,
        "token_dropout": args.token_dropout,
        "action_rollout_buckets": args.action_rollout_buckets,
        "activation_checkpointing": args.activation_checkpointing_enabled,
    }
    network = MuZeroNetwork(**network_kwargs)
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
        planner_q_loss_weight=args.planner_q_loss_weight,
        planner_objective_q_loss_weight=args.planner_objective_q_loss_weight,
        objective_value_weight=args.objective_value_weight,
        objective_reward_weight=args.objective_reward_weight,
        semantic_policy_weight=0.0 if args.disable_semantic_training else args.semantic_policy_weight,
        semantic_value_weight=0.0 if args.disable_semantic_training else args.semantic_value_weight,
        semantic_reward_weight=0.0 if args.disable_semantic_training else args.semantic_reward_weight,
        semantic_state_consistency_weight=0.0 if args.disable_semantic_training else args.semantic_state_consistency_weight,
        objective_diversity_weight=0.0 if args.disable_semantic_training else args.objective_diversity_weight,
        semantic_policy_label_smoothing=args.semantic_policy_label_smoothing,
        state_consistency_weight=args.state_consistency_weight,
        future_world_aux_weight=args.future_world_aux_weight,
        future_bank_state_weight=args.future_bank_state_weight,
        future_bank_delta_weight=args.future_bank_delta_weight,
        future_bank_occupancy_weight=args.future_bank_occupancy_weight,
        future_bank_token_presence_weight=args.future_bank_token_presence_weight,
        future_bank_token_distribution_weight=args.future_bank_token_distribution_weight,
        future_bank_token_slot_state_weight=args.future_bank_token_slot_state_weight,
        future_bank_token_slot_mask_weight=args.future_bank_token_slot_mask_weight,
        future_bank_token_slot_type_weight=args.future_bank_token_slot_type_weight,
        future_bank_token_slot_zone_weight=args.future_bank_token_slot_zone_weight,
        future_bank_token_slot_source_weight=args.future_bank_token_slot_source_weight,
        token_teacher_ema_decay=args.token_teacher_ema_decay,
        future_world_rollout_weight=args.future_world_rollout_weight,
        future_world_rollout_steps=args.future_world_rollout_steps,
        future_world_rollout_decay=args.future_world_rollout_decay,
        latent_gaussian_reg_weight=args.latent_gaussian_reg_weight,
        latent_gaussian_reg_projections=args.latent_gaussian_reg_projections,
        latent_gaussian_reg_slot_weight=args.latent_gaussian_reg_slot_weight,
        latent_gaussian_reg_dynamics_weight=args.latent_gaussian_reg_dynamics_weight,
        latent_gaussian_reg_cov_weight=args.latent_gaussian_reg_cov_weight,
        surprise_loss_weight=args.surprise_loss_weight,
        surprise_hidden_scale=args.surprise_hidden_scale,
        surprise_surface_scale=args.surprise_surface_scale,
        surprise_future_aux_scale=args.surprise_future_aux_scale,
        surprise_target_cap=args.surprise_target_cap,
        surface_mask_weight=args.surface_mask_weight,
        surface_count_weight=args.surface_count_weight,
        surface_domain_weight=args.surface_domain_weight,
        surface_phase_weight=args.surface_phase_weight,
        combat_direct_policy=args.combat_direct_policy,
        combat_rollout_q_blend=args.combat_rollout_q_blend,
        combat_rollout_objective_q_blend=args.combat_rollout_objective_q_blend,
        combat_rollout_risk_blend=args.combat_rollout_risk_blend,
        combat_rollout_steps=args.combat_rollout_steps,
        combat_rollout_beam_width=args.combat_rollout_beam_width,
        combat_rollout_legal_logit_scale=args.combat_rollout_legal_logit_scale,
        combat_rollout_uncertainty_blend=args.combat_rollout_uncertainty_blend,
        combat_rollout_uncertainty_surprise_weight=args.combat_rollout_uncertainty_surprise_weight,
        combat_rollout_uncertainty_surface_weight=args.combat_rollout_uncertainty_surface_weight,
        combat_rollout_uncertainty_latent_weight=args.combat_rollout_uncertainty_latent_weight,
        combat_rollout_uncertainty_disagreement_weight=args.combat_rollout_uncertainty_disagreement_weight,
        combat_rollout_continuation_uncertainty_penalty=args.combat_rollout_continuation_uncertainty_penalty,
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
        mixed_precision=args.mixed_precision,
        amp_init_scale=args.amp_init_scale,
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
            load_optimizer=not args.resume_without_optimizer,
            token_target_encoder=trainer.token_target_encoder,
            amp_grad_scaler=trainer.amp_grad_scaler,
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
        "[setup] Network: "
        f"obs_mode={args.obs_mode} "
        f"model_arch={args.model_arch} "
        f"hidden_dim={network.hidden_dim} "
        f"action_embed_dim={network.action_embed_dim}"
    )
    if args.model_arch == "token_memory_v1":
        print(
            "[setup] Token-memory: "
            f"d_model={args.token_d_model} "
            f"slots={args.token_memory_slots} "
            f"bank_token_slots={args.token_bank_token_slots} "
            f"world_layers={args.token_world_layers} "
            f"decoder_layers={args.token_decoder_layers} "
            f"bank_top_k={args.token_world_bank_top_k} "
            f"planner_blend={args.token_internal_planner_blend:.2f} "
            f"planner_q_blend={args.token_internal_planner_q_blend:.2f} "
            f"planner_obj_q_blend={args.token_internal_planner_objective_q_blend:.2f} "
            f"planner_risk_blend={args.token_internal_planner_risk_blend:.2f} "
            f"rollout_buckets={network.action_rollout_buckets} "
            f"source_prior(bank={args.token_slot_source_same_bank_bias:.2f},"
            f"slot={args.token_slot_source_same_slot_bias:.2f},"
            f"type={args.token_slot_source_type_match_scale:.2f},"
            f"zone={args.token_slot_source_zone_transport_scale:.2f})"
        )
        print(
            "[setup] Future-world rollout: "
            f"weight={args.future_world_rollout_weight:.2f} "
            f"steps={args.future_world_rollout_steps} "
            f"decay={args.future_world_rollout_decay:.2f}"
        )
        print(
            "[setup] JEPA latent regularization: "
            f"weight={args.latent_gaussian_reg_weight:.4f} "
            f"proj={args.latent_gaussian_reg_projections} "
            f"slot={args.latent_gaussian_reg_slot_weight:.2f} "
            f"dyn={args.latent_gaussian_reg_dynamics_weight:.2f} "
            f"cov={args.latent_gaussian_reg_cov_weight:.2f} | "
            f"surprise_weight={args.surprise_loss_weight:.2f} "
            f"surprise_hidden_scale={args.surprise_hidden_scale:.1f} "
            f"surprise_surface_scale={args.surprise_surface_scale:.1f} "
            f"surprise_future_aux_scale={args.surprise_future_aux_scale:.1f} "
            f"surprise_cap={args.surprise_target_cap:.1f}"
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
        "[setup] Mixed precision: "
        f"mode={trainer.mixed_precision}, "
        f"enabled={trainer.amp_enabled}, "
        f"device_type={trainer.amp_device_type}, "
        f"dtype={trainer.amp_dtype}, "
        f"scaler={trainer.amp_scaler_enabled}"
    )
    print(
        "[setup] Activation checkpointing: "
        f"cli={args.activation_checkpointing} "
        f"enabled={args.activation_checkpointing_enabled}"
    )
    print(
        "[setup] Allocator config: "
        f"PYTORCH_ALLOC_CONF={os.environ.get('PYTORCH_ALLOC_CONF')!r}, "
        f"PYTORCH_HIP_ALLOC_CONF={os.environ.get('PYTORCH_HIP_ALLOC_CONF')!r}"
    )
    print(
        "[setup] Combat direct policy: "
        f"{'on' if args.combat_direct_policy else 'off'} "
        f"(mode={args.combat_policy_mode}, "
        f"rollout_q={args.combat_rollout_q_blend:.2f}, "
        f"rollout_obj_q={args.combat_rollout_objective_q_blend:.2f}, "
        f"rollout_risk={args.combat_rollout_risk_blend:.2f}, "
        f"uncertainty={args.combat_rollout_uncertainty_blend:.2f}, "
        f"steps={args.combat_rollout_steps}, "
        f"beam={args.combat_rollout_beam_width}, "
        f"legal_scale={args.combat_rollout_legal_logit_scale:.2f}, "
        f"u_weights=(surprise:{args.combat_rollout_uncertainty_surprise_weight:.2f},"
        f"surface:{args.combat_rollout_uncertainty_surface_weight:.2f},"
        f"latent:{args.combat_rollout_uncertainty_latent_weight:.2f},"
        f"disagree:{args.combat_rollout_uncertainty_disagreement_weight:.2f}), "
        f"beam_uncertainty_penalty={args.combat_rollout_continuation_uncertainty_penalty:.2f})"
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
        actor_network = MuZeroNetwork(**network_kwargs)
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
            planner_q_loss_weight=args.planner_q_loss_weight,
            planner_objective_q_loss_weight=args.planner_objective_q_loss_weight,
            objective_value_weight=args.objective_value_weight,
            objective_reward_weight=args.objective_reward_weight,
            semantic_policy_weight=0.0 if args.disable_semantic_training else args.semantic_policy_weight,
            semantic_value_weight=0.0 if args.disable_semantic_training else args.semantic_value_weight,
            semantic_reward_weight=0.0 if args.disable_semantic_training else args.semantic_reward_weight,
            semantic_state_consistency_weight=0.0 if args.disable_semantic_training else args.semantic_state_consistency_weight,
            objective_diversity_weight=0.0 if args.disable_semantic_training else args.objective_diversity_weight,
            semantic_policy_label_smoothing=args.semantic_policy_label_smoothing,
            state_consistency_weight=args.state_consistency_weight,
            future_world_aux_weight=args.future_world_aux_weight,
            future_bank_state_weight=args.future_bank_state_weight,
            future_bank_delta_weight=args.future_bank_delta_weight,
            future_bank_occupancy_weight=args.future_bank_occupancy_weight,
            future_bank_token_presence_weight=args.future_bank_token_presence_weight,
            future_bank_token_distribution_weight=args.future_bank_token_distribution_weight,
            future_bank_token_slot_state_weight=args.future_bank_token_slot_state_weight,
            future_bank_token_slot_mask_weight=args.future_bank_token_slot_mask_weight,
            future_bank_token_slot_type_weight=args.future_bank_token_slot_type_weight,
            future_bank_token_slot_zone_weight=args.future_bank_token_slot_zone_weight,
            future_bank_token_slot_source_weight=args.future_bank_token_slot_source_weight,
            token_teacher_ema_decay=args.token_teacher_ema_decay,
            future_world_rollout_weight=args.future_world_rollout_weight,
            future_world_rollout_steps=args.future_world_rollout_steps,
            future_world_rollout_decay=args.future_world_rollout_decay,
            latent_gaussian_reg_weight=args.latent_gaussian_reg_weight,
            latent_gaussian_reg_projections=args.latent_gaussian_reg_projections,
            latent_gaussian_reg_slot_weight=args.latent_gaussian_reg_slot_weight,
            latent_gaussian_reg_dynamics_weight=args.latent_gaussian_reg_dynamics_weight,
            latent_gaussian_reg_cov_weight=args.latent_gaussian_reg_cov_weight,
            surprise_loss_weight=args.surprise_loss_weight,
            surprise_hidden_scale=args.surprise_hidden_scale,
            surprise_surface_scale=args.surprise_surface_scale,
            surprise_future_aux_scale=args.surprise_future_aux_scale,
            surprise_target_cap=args.surprise_target_cap,
            surface_mask_weight=args.surface_mask_weight,
            surface_count_weight=args.surface_count_weight,
            surface_domain_weight=args.surface_domain_weight,
            surface_phase_weight=args.surface_phase_weight,
            combat_direct_policy=args.combat_direct_policy,
            combat_rollout_q_blend=args.combat_rollout_q_blend,
            combat_rollout_objective_q_blend=args.combat_rollout_objective_q_blend,
            combat_rollout_risk_blend=args.combat_rollout_risk_blend,
            combat_rollout_steps=args.combat_rollout_steps,
            combat_rollout_beam_width=args.combat_rollout_beam_width,
            combat_rollout_legal_logit_scale=args.combat_rollout_legal_logit_scale,
            combat_rollout_uncertainty_blend=args.combat_rollout_uncertainty_blend,
            combat_rollout_uncertainty_surprise_weight=args.combat_rollout_uncertainty_surprise_weight,
            combat_rollout_uncertainty_surface_weight=args.combat_rollout_uncertainty_surface_weight,
            combat_rollout_uncertainty_latent_weight=args.combat_rollout_uncertainty_latent_weight,
            combat_rollout_uncertainty_disagreement_weight=args.combat_rollout_uncertainty_disagreement_weight,
            combat_rollout_continuation_uncertainty_penalty=args.combat_rollout_continuation_uncertainty_penalty,
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
            mixed_precision=args.mixed_precision,
            amp_init_scale=args.amp_init_scale,
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
        def _to_float(value: Any, default: float = 0.0) -> float:
            try:
                return float(value)
            except (TypeError, ValueError):
                return float(default)

        def _safe_tag(value: Any) -> str:
            return re.sub(r"[^0-9a-zA-Z]+", "_", str(value or "").strip().lower()).strip("_") or "unknown"

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
                trainer.writer.add_scalar(writer_key, _to_float(episode_metrics.get(metric_key)), trainer.episode_count)
        for metric_key, writer_key in (
            ("act1_boss_seen", "episode/act1_boss_seen"),
            ("act1_clear", "episode/act1_clear"),
        ):
            if metric_key in episode_metrics:
                trainer.writer.add_scalar(writer_key, 1.0 if bool(episode_metrics.get(metric_key)) else 0.0, trainer.episode_count)

        # Async actors use NullSummaryWriter while collecting episodes, so replay the
        # per-episode diagnostics on the learner writer here. Without this, combat
        # sandbox actor-learner runs silently lose decision/search/boss namespaces.
        decision_counts = episode_metrics.get("decision_counts") if isinstance(episode_metrics.get("decision_counts"), dict) else {}
        total_decisions = int(episode_metrics.get("decision_total", 0) or 0)
        if total_decisions > 0:
            domain_overrides = episode_metrics.get("decision_domain_overrides") if isinstance(episode_metrics.get("decision_domain_overrides"), dict) else {}
            combat_like_count = int(episode_metrics.get("combat_like_decision_count", 0) or 0)
            eligible_count = int(episode_metrics.get("direct_policy_eligible_count", 0) or 0)
            used_count = int(episode_metrics.get("direct_policy_used_count", 0) or 0)
            fast_path_counts = episode_metrics.get("fast_path_counts") if isinstance(episode_metrics.get("fast_path_counts"), dict) else {}
            fast_path_total = int(episode_metrics.get("fast_path_total", 0) or 0)
            fast_path_reason_counts = episode_metrics.get("fast_path_reason_counts") if isinstance(episode_metrics.get("fast_path_reason_counts"), dict) else {}

            trainer.writer.add_scalar("decision/domain_override_count", float(sum(int(v or 0) for v in domain_overrides.values())), trainer.episode_count)
            trainer.writer.add_scalar("decision/combat_like_count", float(combat_like_count), trainer.episode_count)
            trainer.writer.add_scalar("decision/direct_policy_eligible_count", float(eligible_count), trainer.episode_count)
            trainer.writer.add_scalar("decision/direct_policy_used_count", float(used_count), trainer.episode_count)
            trainer.writer.add_scalar("decision/direct_policy_used_rate", float(used_count) / float(max(eligible_count, 1)), trainer.episode_count)
            trainer.writer.add_scalar("decision/fast_path_count", float(fast_path_total), trainer.episode_count)
            trainer.writer.add_scalar("decision/fast_path_rate", float(fast_path_total) / float(max(total_decisions, 1)), trainer.episode_count)

            for domain in DECISION_DOMAINS:
                domain_count = int(decision_counts.get(domain, 0) or 0)
                trainer.writer.add_scalar(f"decision/{domain}_count", float(domain_count), trainer.episode_count)
                trainer.writer.add_scalar(f"decision/{domain}_share", float(domain_count) / float(max(total_decisions, 1)), trainer.episode_count)
                if domain_count > 0:
                    domain_fast_paths = int(fast_path_counts.get(domain, 0) or 0)
                    trainer.writer.add_scalar(f"decision/{domain}/fast_path_rate", float(domain_fast_paths) / float(domain_count), trainer.episode_count)

            if fast_path_total > 0:
                for reason, count in fast_path_reason_counts.items():
                    trainer.writer.add_scalar(
                        f"decision/fast_path/reason_{_safe_tag(reason)}_rate",
                        float(int(count or 0)) / float(fast_path_total),
                        trainer.episode_count,
                    )

        search_suffix_map = {
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
            "search_mode_direct_policy": "search_mode_direct_policy",
            "search_mode_direct_rollout_planner": "search_mode_direct_rollout_planner",
            "direct_rollout_q_mean": "direct_rollout_q_mean",
            "direct_rollout_objective_q_mean": "direct_rollout_objective_q_mean",
            "direct_rollout_risk_q_mean": "direct_rollout_risk_q_mean",
            "direct_rollout_uncertainty_mean": "direct_rollout_uncertainty_mean",
            "direct_rollout_uncertainty_bias_abs_mean": "direct_rollout_uncertainty_bias_abs_mean",
            "direct_rollout_surprise_mean": "direct_rollout_surprise_mean",
            "direct_rollout_surface_entropy_mean": "direct_rollout_surface_entropy_mean",
            "direct_rollout_latent_drift_mean": "direct_rollout_latent_drift_mean",
            "direct_rollout_branch_disagreement_mean": "direct_rollout_branch_disagreement_mean",
            "direct_rollout_steps_used": "direct_rollout_steps_used",
            "direct_rollout_branch_count_mean": "direct_rollout_branch_count_mean",
            "direct_rollout_root_valid_count": "direct_rollout_root_valid_count",
            "direct_rollout_root_bucket_size": "direct_rollout_root_bucket_size",
            "direct_rollout_bucket_padding_ratio": "direct_rollout_bucket_padding_ratio",
            "direct_rollout_max_branch_bucket_size": "direct_rollout_max_branch_bucket_size",
            "direct_rollout_branch_padding_ratio": "direct_rollout_branch_padding_ratio",
            "combat_quality_bias_applied": "combat_quality_bias_applied",
            "combat_quality_bias_abs_mean": "combat_quality_bias_abs_mean",
            "combat_quality_wasteful_end_turn_bias_applied": "combat_quality_wasteful_end_turn_bias_applied",
            "combat_quality_wasteful_end_turn_available": "combat_quality_wasteful_end_turn_available",
            "combat_quality_wasteful_end_turn_selected": "combat_quality_wasteful_end_turn_selected",
            "combat_quality_end_turn_penalty_max": "combat_quality_end_turn_penalty_max",
            "combat_quality_energy": "combat_quality_energy",
            "combat_quality_positive_action_count": "combat_quality_positive_action_count",
            "combat_quality_playable_action_count": "combat_quality_playable_action_count",
            "combat_quality_end_turn_severity": "combat_quality_end_turn_severity",
            "combat_quality_potion_available_count": "combat_quality_potion_available_count",
            "combat_quality_potion_urgent_count": "combat_quality_potion_urgent_count",
            "combat_quality_potion_low_urgency_count": "combat_quality_potion_low_urgency_count",
            "combat_quality_potion_save_recommended_count": "combat_quality_potion_save_recommended_count",
            "combat_quality_potion_no_followup_count": "combat_quality_potion_no_followup_count",
            "combat_quality_potion_lethal_count": "combat_quality_potion_lethal_count",
            "combat_quality_potion_prevent_lethal_count": "combat_quality_potion_prevent_lethal_count",
            "combat_quality_potion_mechanism_count": "combat_quality_potion_mechanism_count",
            "combat_quality_potion_overkill_count": "combat_quality_potion_overkill_count",
            "combat_quality_potion_block_waste_count": "combat_quality_potion_block_waste_count",
            "combat_quality_potion_use_quality_mean": "combat_quality_potion_use_quality_mean",
            "combat_quality_potion_waste_risk_mean": "combat_quality_potion_waste_risk_mean",
            "combat_quality_potion_selected": "combat_quality_potion_selected",
            "combat_quality_potion_selected_when_available": "combat_quality_potion_selected_when_available",
            "combat_quality_potion_high_urgency_selected": "combat_quality_potion_high_urgency_selected",
            "combat_quality_potion_low_urgency_selected": "combat_quality_potion_low_urgency_selected",
            "combat_quality_potion_save_recommended_selected": "combat_quality_potion_save_recommended_selected",
            "combat_quality_potion_no_followup_selected": "combat_quality_potion_no_followup_selected",
            "combat_quality_potion_lethal_selected": "combat_quality_potion_lethal_selected",
            "combat_quality_potion_prevent_lethal_selected": "combat_quality_potion_prevent_lethal_selected",
            "combat_quality_potion_mechanism_selected": "combat_quality_potion_mechanism_selected",
            "combat_quality_potion_overkill_selected": "combat_quality_potion_overkill_selected",
            "combat_quality_potion_block_waste_selected": "combat_quality_potion_block_waste_selected",
            "combat_quality_potion_use_quality_selected": "combat_quality_potion_use_quality_selected",
            "combat_quality_potion_waste_risk_selected": "combat_quality_potion_waste_risk_selected",
            "combat_quality_zero_energy_x_cost_count": "combat_quality_zero_energy_x_cost_count",
            "combat_quality_x_cost_available_count": "combat_quality_x_cost_available_count",
            "combat_quality_x_cost_selected": "combat_quality_x_cost_selected",
            "combat_quality_x_cost_selected_energy": "combat_quality_x_cost_selected_energy",
            "combat_quality_x_cost_zero_energy_selected": "combat_quality_x_cost_zero_energy_selected",
            "combat_quality_zero_energy_x_cost_selected": "combat_quality_zero_energy_x_cost_selected",
            "combat_quality_end_turn_selected": "combat_quality_end_turn_selected",
            "combat_quality_kaiser_back_attack_risk": "combat_quality_kaiser_back_attack_risk",
            "combat_quality_kaiser_defense_candidate_count": "combat_quality_kaiser_defense_candidate_count",
            "combat_quality_kaiser_defense_selected": "combat_quality_kaiser_defense_selected",
            "combat_quality_kaiser_facing_change_candidate_count": "combat_quality_kaiser_facing_change_candidate_count",
            "combat_quality_kaiser_pressure_candidate_count": "combat_quality_kaiser_pressure_candidate_count",
            "combat_quality_kaiser_facing_change_selected": "combat_quality_kaiser_facing_change_selected",
            "combat_quality_kaiser_pressure_selected": "combat_quality_kaiser_pressure_selected",
            "combat_quality_kaiser_risky_end_turn_selected": "combat_quality_kaiser_risky_end_turn_selected",
            "combat_quality_ceremonial_one_card_lock": "combat_quality_ceremonial_one_card_lock",
            "combat_quality_ceremonial_stun_window": "combat_quality_ceremonial_stun_window",
            "combat_quality_ceremonial_low_impact_count": "combat_quality_ceremonial_low_impact_count",
            "combat_quality_ceremonial_high_impact_count": "combat_quality_ceremonial_high_impact_count",
            "combat_quality_ceremonial_low_impact_selected": "combat_quality_ceremonial_low_impact_selected",
            "combat_quality_ceremonial_high_impact_selected": "combat_quality_ceremonial_high_impact_selected",
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
            "zero_energy_x_cost_guard_applied": "zero_energy_x_cost_guard_applied",
            "zero_energy_x_cost_guard_forced_alternative": "zero_energy_x_cost_guard_forced_alternative",
            "objective_prior_applied": "objective_prior_applied",
            "combat_grounded_root_enabled": "combat_grounded_root_enabled",
            "q_value_ucb_enabled": "q_value_ucb_enabled",
        }
        domain_search_means = episode_metrics.get("domain_search_means") if isinstance(episode_metrics.get("domain_search_means"), dict) else {}
        for domain, stats in domain_search_means.items():
            if not isinstance(stats, dict):
                continue
            safe_domain = _safe_tag(domain)
            for stat_key, value in stats.items():
                suffix = search_suffix_map.get(str(stat_key), _safe_tag(stat_key))
                trainer.writer.add_scalar(f"search/{safe_domain}/{suffix}", _to_float(value), trainer.episode_count)

        domain_family_rates = episode_metrics.get("domain_family_rates") if isinstance(episode_metrics.get("domain_family_rates"), dict) else {}
        for domain, rates in domain_family_rates.items():
            if not isinstance(rates, dict):
                continue
            safe_domain = _safe_tag(domain)
            for family, value in rates.items():
                trainer.writer.add_scalar(f"decision/{safe_domain}/family_{_safe_tag(family)}_rate", _to_float(value), trainer.episode_count)

        boss_diagnostics = episode_metrics.get("boss_diagnostics") if isinstance(episode_metrics.get("boss_diagnostics"), dict) else {}
        for tag, value in boss_diagnostics.items():
            trainer.writer.add_scalar(str(tag), _to_float(value), trainer.episode_count)

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
                    total_loss_value = float(losses.get("loss/total", 0.0) or 0.0)
                    if total_loss_value != 0.0:
                        aux_keys = (
                            "loss/planner_q",
                            "loss/planner_objective_q",
                            "loss/future_world_aux",
                            "loss/state_consistency",
                            "loss/semantic_state_consistency",
                            "loss/latent_policy_distill",
                            "loss/surface_mask",
                            "loss/surface_count",
                            "loss/surface_domain",
                            "loss/surface_phase",
                        )
                        aux_total = float(sum(float(losses.get(key, 0.0) or 0.0) for key in aux_keys))
                        for key in ("loss/policy", "loss/value", "loss/reward", *aux_keys):
                            safe_suffix = key.split("/", 1)[-1]
                            trainer.writer.add_scalar(
                                f"loss_ratio/{safe_suffix}",
                                float(losses.get(key, 0.0) or 0.0) / total_loss_value,
                                trainer.total_steps,
                            )
                        trainer.writer.add_scalar("loss_ratio/aux_total", aux_total / total_loss_value, trainer.total_steps)
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
