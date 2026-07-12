"""MuZero trainer core for Slay the Spire 2.

This module owns the ``MuZeroTrainer`` compatibility class and the public
default constants used by the CLI.  The executable entrypoint lives in
``muzero.train`` and the launch/orchestration loop lives in
``muzero.training.cli_main``.

New strategy, combat-quality, route-heuristic, search, diagnostics, and CLI
logic should continue to live in focused modules instead of growing this file.
"""
from __future__ import annotations

import argparse
import copy
from contextlib import nullcontext
import json
import math
import os
import queue
import re
import signal
import threading
import time
from typing import Any, Callable

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import gymnasium as gym
from torch.utils.tensorboard import SummaryWriter

from muzero.combat_quality import (
    COMBAT_QUALITY_CARD_BLOCK_SEARCH_SUFFIXES,
    apply_card_block_waste_bias,
    boss_card_block_waste_metrics,
    card_block_waste_profile as compute_card_block_waste_profile,
)
from muzero.combat_quality.hard_guard_orchestrator import CombatHardGuardMixin
from muzero.combat_quality.potion_timing import PotionTimingMixin
from muzero.combat_quality.trainer_quality import CombatActionQualityMixin
from muzero.diagnostics.episode_metrics import EpisodeMetricsMixin
from muzero.diagnostics.trainer_dumps import DiagnosticDumpMixin
from muzero.strategy import action_features as action_feature_policy
from muzero.strategy.encounters import insatiable as insatiable_strategy
from muzero.strategy.encounters import kaiser as kaiser_strategy
from muzero.training.checkpointing import CheckpointingMixin, dict_obs_to_torch, load_resume_checkpoint
from muzero.training.cli_parsing import (
    parse_encounter_pool,
    parse_encounter_tiers,
    parse_encounter_weights,
    parse_int_list,
    parse_session_files,
    parse_tier_weights,
)
from muzero.training.env_factory import (
    _get_live_supported_encounter_ids,
    build_train_env,
    mask_fn,
    resolve_training_session_files,
)
from muzero.training.build_route_hard_guards import BuildRouteHardGuardMixin
from muzero.training.combat_runtime_features import CombatRuntimeFeatureMixin
from muzero.training.decision_constants import (
    BUILD_ROUTE_SETTLEMENT_FAMILIES,
    TRIVIAL_BUILD_FAST_PATH_COMPLEX_FAMILIES,
    WASTEFUL_REWARD_PHASES,
)
from muzero.training.losses import TrainingLossMixin
from muzero.training.monitoring import EpisodeCaptureBuffer, NullSummaryWriter, RecentCombatMonitor
from muzero.training.paths import PolicyModulePaths, RunPaths
from muzero.training.self_play import SelfPlayMixin
from muzero.training.train_step import TrainStepMixin
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
)
from muzero.sts2_env.latent_regularizers import (
    latent_gaussian_regularizer,
    slot_latent_gaussian_regularizer,
)
from sts2_env.objective_heads import (
    HEAD_HP_PRESERVATION,
    NUM_OBJECTIVE_HEADS,
    compute_transition_objective_rewards,
)
from sts2_env.observation_v2 import DECISION_DOMAINS, DictObservationEncoder, MAX_ACTIONS
from sts2_env.semantic_action import SEMANTIC_ACTION_FAMILIES, SEMANTIC_ROLE_NAMES
from sts2_env.observation_v3 import WorldTokenObservationEncoder
from sts2_env.potion_profiles import (
    DEFAULT_EFFECT_PROFILE as _POTION_EFFECT_DEFAULT,
    all_potion_ids as _all_potion_ids,
    get_potion_profile as _get_potion_profile,
)
from sts2_env.path_utils import normalize_path_str, resolve_torch_device, running_in_wsl
from muzero.sts2_env.semantic_rollout import (
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

# Training monitors, parsing helpers, environment factories, and checkpoint
# loading live in ``muzero.training.*``. Keep this module as the legacy
# trainer/orchestrator until the remaining MuZeroTrainer methods are split.

class MuZeroTrainer(
    CombatHardGuardMixin,
    BuildRouteHardGuardMixin,
    CombatRuntimeFeatureMixin,
    PotionTimingMixin,
    CombatActionQualityMixin,
    DiagnosticDumpMixin,
    EpisodeMetricsMixin,
    SelfPlayMixin,
    TrainStepMixin,
    TrainingLossMixin,
    CheckpointingMixin,
):
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
        combat_hp_preservation_aux_weight: float = 0.0,
        human_demo_policy_alignment: Any | None = None,
        human_demo_alignment_weight: float = 0.0,
        human_demo_alignment_shadow_only: bool = True,
        human_demo_alignment_every_n_train_steps: int = 4,
        offline_policy_alignment: Any | None = None,
        offline_alignment_weight: float = 0.0,
        offline_alignment_shadow_only: bool = True,
        offline_alignment_every_n_train_steps: int = 4,
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
        potion_reward_fast_path: bool = True,
        recent_tail_windows: tuple[int, ...] = DEFAULT_RECENT_TAIL_WINDOWS,
        recent_tail_tracked_encounters: list[str] | None = None,
        recent_tail_min_samples: int = 4,
        route_heuristic_bias: float = 0.0,
        route_safety_guard: bool = False,
        combat_hard_guard_policy: str = "off",
        build_hard_guard_policy: str = "off",
        hard_guard_target_rewrite: str | bool = "off",
        log_dir: str | None = None,
        checkpoint_dir: str | None = None,
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
        self.combat_hp_preservation_aux_weight = max(float(combat_hp_preservation_aux_weight), 0.0)
        if self.combat_hp_preservation_aux_weight > 0.0 and HEAD_HP_PRESERVATION >= NUM_OBJECTIVE_HEADS:
            raise ValueError(
                "HP-preservation objective head index is outside reward_component_logits; "
                "refusing to enable combat_hp_preservation_aux_weight."
            )
        self.human_demo_policy_alignment = human_demo_policy_alignment
        self.human_demo_alignment_weight = max(float(human_demo_alignment_weight), 0.0)
        self.human_demo_alignment_shadow_only = bool(human_demo_alignment_shadow_only)
        self.human_demo_alignment_every_n_train_steps = max(
            int(human_demo_alignment_every_n_train_steps),
            1,
        )
        self.offline_policy_alignment = offline_policy_alignment
        self.offline_alignment_weight = max(float(offline_alignment_weight), 0.0)
        self.offline_alignment_shadow_only = bool(offline_alignment_shadow_only)
        self.offline_alignment_every_n_train_steps = max(
            int(offline_alignment_every_n_train_steps),
            1,
        )
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
        self.potion_reward_fast_path = bool(potion_reward_fast_path)
        self.recent_tail_windows = tuple(sorted({max(int(window), 1) for window in recent_tail_windows}))
        self.recent_tail_min_samples = max(int(recent_tail_min_samples), 1)
        # Phase 3 (recovery 2026-05-09): route-heuristic prior bias weight.
        # Wired via constructor kwarg ``route_heuristic_bias`` (default 0.0
        # = disabled). Trainer pushes this scaled by score_normalized into
        # MCTS at every map decision; non-route decisions push None.
        self.route_heuristic_bias_weight = max(0.0, float(route_heuristic_bias)) if isinstance(route_heuristic_bias, (int, float)) else 0.0
        # Act1 recovery (2026-05-10): optional hard guard for route
        # decisions.  Unlike Phase 3's soft prior bias, this only fires when
        # the chosen route is high-risk and a low-risk alternative is
        # positionally aligned and legal.  Default off for reproducibility;
        # launch scripts opt in with ``--route-safety-guard``.
        self.route_safety_guard_enabled = bool(route_safety_guard)
        valid_guard_policies = {"full", "emergency", "off"}
        self.combat_hard_guard_policy = str(combat_hard_guard_policy or "off").strip().lower()
        if self.combat_hard_guard_policy not in valid_guard_policies:
            self.combat_hard_guard_policy = "off"
        self.build_hard_guard_policy = str(build_hard_guard_policy or "off").strip().lower()
        if self.build_hard_guard_policy not in valid_guard_policies:
            self.build_hard_guard_policy = "off"
        # RC-4: target-rewrite is independent of which guard SET runs. Default off so a guard
        # override no longer rewrites the policy target to a one-hot (keeps learning on-policy).
        self.hard_guard_target_rewrite = (
            bool(hard_guard_target_rewrite)
            if isinstance(hard_guard_target_rewrite, bool)
            else str(hard_guard_target_rewrite or "off").strip().lower() in {"on", "true", "1", "yes"}
        )
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

        # File-governance refactor boundary (2026-05-10):
        # runtime artifacts and policy/source module roots are centralized in
        # ``muzero.training.paths``.  Keep this as thin wiring only; new
        # strategy/search code should not add ad-hoc Path(...) construction to
        # this legacy monolith.
        self.run_paths = RunPaths.from_args(
            argparse.Namespace(log_dir=log_dir, checkpoint_dir=checkpoint_dir, resume_from=None)
        )
        self.run_paths.ensure_dirs()
        self.policy_module_paths = PolicyModulePaths.from_package_root(self.run_paths.package_root)
        self.policy_module_paths.ensure_package_dirs()
        self.log_dir = str(self.run_paths.log_dir)
        self.checkpoint_dir = str(self.run_paths.checkpoint_dir)
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

        self.writer = SummaryWriter(log_dir=self.log_dir)
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
        self._end_turn_pre_dispatch_audit_count = 0
        try:
            self._end_turn_pre_dispatch_audit_max = int(
                os.environ.get("MUZERO_END_TURN_PRE_DISPATCH_AUDIT_MAX", "100000")
            )
        except (TypeError, ValueError):
            self._end_turn_pre_dispatch_audit_max = 100000
        self._end_turn_pre_dispatch_audit_disabled = (
            os.environ.get("MUZERO_END_TURN_PRE_DISPATCH_AUDIT", "1").strip() == "0"
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

    # Combat/action feature adapters and encounter wrappers live in
    # ``muzero.training.combat_runtime_features.CombatRuntimeFeatureMixin``.


    # Search-stat compaction and episode metrics live in ``muzero.diagnostics.episode_metrics``.
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
            # Act1 recovery: the progress-only settlement made late Act1
            # deaths (floor 13/14 normal) look positive because floor_norm was
            # high and the early-death penalty had decayed to nearly zero.
            # Mirror the EnvV2 terminal death penalty at the value-target
            # level: non-clear deaths should not be attractive merely because
            # they happened late in the act.
            signal -= 0.65 * floor_norm
            # HLB/Act1 recovery: reaching the boss but dying used to receive
            # almost the same light-settlement target as a successful Act1
            # trajectory because floor_norm saturates at floor 16 and the
            # +boss_seen bonus remained.  That taught "reach boss, deal damage,
            # die" as an acceptable local optimum.  Keep boss reach positive,
            # but create a large gap to act1_clear so the value head must
            # prefer surviving into Act2.
            if act1_boss_seen:
                signal -= 0.35
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
                elif (
                    reward_type == "potion"
                    and self.potion_reward_fast_path
                    and empty_potion_slots > 0
                ):
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

    # Self-play episode orchestration lives in ``muzero.training.self_play``.
    # Optimizer-step implementation lives in ``muzero.training.train_step``.
    # Tensor/objective/surface loss helpers live in ``muzero.training.losses``.
    # Checkpoint save/prune helpers live in ``muzero.training.checkpointing``.
