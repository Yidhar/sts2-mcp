"""Checkpoint and observation tensor helpers for MuZero training."""

from __future__ import annotations

from pathlib import Path
from typing import Any
import json
import os
import pickle
import shutil

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

from muzero.sts2_env.muzero_buffer import MuZeroReplayBuffer, replay_schema_metadata
from muzero.sts2_env.muzero_model import MuZeroNetwork


def _model_schema_version(network: MuZeroNetwork) -> str:
    """Return a compact model schema id for checkpoint/replay compatibility."""

    spec = network.constructor_spec() if hasattr(network, "constructor_spec") else {}
    obs_mode = str(spec.get("obs_mode") or getattr(network, "obs_mode", "dense_v2"))
    model_arch = str(spec.get("model_arch") or getattr(network, "model_arch", "dense_v1"))
    slot_layout = str(spec.get("token_memory_slot_layout") or "legacy")
    memory_slots = int(spec.get("token_memory_slots") or 0)
    d_model = int(spec.get("token_d_model") or 0)
    action_dim = int(spec.get("action_embed_dim") or getattr(network, "action_embed_dim", 0))
    support_size = int(spec.get("support_size") or getattr(network, "support_size", 0))
    return (
        f"{obs_mode}:{model_arch}:slot_layout={slot_layout}:"
        f"slots={memory_slots}:d={d_model}:action={action_dim}:support={support_size}"
    )


def _checkpoint_schema_metadata(network: MuZeroNetwork) -> dict[str, Any]:
    """Schema metadata persisted in every checkpoint."""

    replay_schema = replay_schema_metadata()
    return {
        "format": "muzero-v2",
        "model_schema_version": _model_schema_version(network),
        **replay_schema,
    }


def _metadata_replay_compatible(saved_metadata: dict[str, Any], current_schema: dict[str, Any]) -> tuple[bool, list[str]]:
    """Check whether a checkpoint's replay buffer can be safely loaded."""

    mismatches: list[str] = []
    for key in (
        "checkpoint_compatibility_version",
        "obs_schema_version",
        "replay_schema_version",
        "observation_shape_caps",
    ):
        if saved_metadata.get(key) != current_schema.get(key):
            mismatches.append(key)
    return (not mismatches, mismatches)


def _replay_state_compatible(replay_state: dict[str, Any], current_schema: dict[str, Any]) -> tuple[bool, list[str]]:
    """Check schema metadata embedded inside replay_buffer.pkl."""

    mismatches: list[str] = []
    if replay_state.get("schema_version") != current_schema.get("replay_schema_version"):
        mismatches.append("replay_buffer.schema_version")
    if replay_state.get("checkpoint_compatibility_version") != current_schema.get("checkpoint_compatibility_version"):
        mismatches.append("replay_buffer.checkpoint_compatibility_version")
    if replay_state.get("observation_shape_caps") != current_schema.get("observation_shape_caps"):
        mismatches.append("replay_buffer.observation_shape_caps")
    return (not mismatches, mismatches)


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
    """Load network/optimizer/(optional) replay buffer from a checkpoint dir."""

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
    current_schema = _checkpoint_schema_metadata(network)
    replay_schema_ok, replay_schema_mismatches = _metadata_replay_compatible(metadata, current_schema)
    allow_exact_resume = not (skipped_keys or missing_keys)

    if allow_exact_resume and load_optimizer:
        try:
            optimizer.load_state_dict(optimizer_state)
        except Exception as exc:
            print(f"[resume] Optimizer state not loaded ({exc}); starting with fresh optimizer.")
    elif allow_exact_resume and not load_optimizer:
        print("[resume] Skipping optimizer warm-start by request; starting with fresh optimizer.")
    else:
        print(
            "[resume] Skipping optimizer warm-start because network state was only partially restored; "
            "starting with fresh optimizer."
        )

    allow_replay_buffer_load = allow_exact_resume and load_buffer and replay_schema_ok
    if replay_buffer_path.exists() and allow_replay_buffer_load:
        with replay_buffer_path.open("rb") as handle:
            replay_buffer_state = pickle.load(handle)
        replay_state_ok, replay_state_mismatches = _replay_state_compatible(replay_buffer_state, current_schema)
        if replay_state_ok:
            buffer.load_state_dict(replay_buffer_state)
        else:
            print(
                "[resume] Skipping replay buffer warm-start because replay_buffer.pkl schema is incompatible "
                f"({', '.join(replay_state_mismatches)}); starting with empty buffer."
            )
    elif replay_buffer_path.exists() and not load_buffer:
        print("[resume] Skipping replay buffer warm-start by request; starting with empty buffer.")
    elif replay_buffer_path.exists() and not replay_schema_ok:
        print(
            "[resume] Skipping replay buffer warm-start because checkpoint replay schema is incompatible "
            f"({', '.join(replay_schema_mismatches)}); starting with empty buffer."
        )
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


class CheckpointingMixin:
    """Checkpoint save/prune helpers for ``MuZeroTrainer``.

    Loading and observation tensor conversion remain module-level functions;
    saving lives here so the legacy train entrypoint does not own artifact
    layout and checkpoint retention logic.
    """

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
            **_checkpoint_schema_metadata(self.network),
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
            "potion_reward_fast_path": bool(self.potion_reward_fast_path),
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

