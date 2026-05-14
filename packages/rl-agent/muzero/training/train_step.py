"""Single optimizer-step implementation for MuZeroTrainer.

Loss functions live in ``training.losses``; this mixin owns batch sampling,
unroll execution, optimization, and train-step metric aggregation.  Keeping it
separate prevents the CLI/orchestrator module from growing with every new head.
"""

from __future__ import annotations

import os
from typing import Any

import torch


class TrainStepMixin:
    """One training/optimization step mixed into MuZeroTrainer."""

    def _maybe_empty_cuda_cache_after_train_step(self) -> float:
        """Optionally release inactive CUDA/HIP allocator blocks after a train step.

        ROCm exposes its caching allocator through ``torch.cuda``.  Large token
        models can leave a high reserved-minus-allocated gap after backward,
        especially when SDPA uses the math backend for stability.  The default is
        off to avoid a throughput tax; low-memory launch scripts can enable this
        with ``STS2_CUDA_EMPTY_CACHE_EVERY_TRAIN_STEPS=N``.

        Returns:
            1.0 when ``torch.cuda.empty_cache()`` was called, else 0.0.  The
            scalar is written into TensorBoard as a heartbeat/debug signal.
        """

        raw_every = os.environ.get("STS2_CUDA_EMPTY_CACHE_EVERY_TRAIN_STEPS", "0")
        try:
            every = max(0, int(str(raw_every).strip() or "0"))
        except ValueError:
            every = 0
        if every <= 0:
            return 0.0
        try:
            device = torch.device(str(getattr(self, "device", "cpu")))
        except (TypeError, RuntimeError):
            return 0.0
        if device.type != "cuda" or not torch.cuda.is_available():
            return 0.0

        count = int(getattr(self, "_cuda_empty_cache_train_step_count", 0)) + 1
        self._cuda_empty_cache_train_step_count = count
        if (count % every) != 0:
            return 0.0

        try:
            torch.cuda.empty_cache()
            if os.environ.get("STS2_CUDA_EMPTY_CACHE_VERBOSE", "0").strip().lower() in {"1", "true", "yes", "on"}:
                print(
                    f"[memory] torch.cuda.empty_cache() after train_step={count}",
                    flush=True,
                )
            return 1.0
        except Exception:
            return 0.0

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
        # P0-2 (recovery 2026-05-06): when ``--replay-tier-quota`` is on the
        # buffer ships per-batch quota statistics (target counts, fallback
        # count, cap-hit / unfilled-min flags). When the quota path is off
        # this is None and the metrics block emits zeros to keep TB tag
        # cardinality stable.
        tier_quota_info = batch.get("tier_quota_info")

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
            # P0-7 hardening (recovery 2026-05-07): hard-skip optimizer step
            # when ANY unroll-step's loss components cross the spike threshold.
            # Previously we wrote a dump but still let the gradient through
            # (clipped to max_grad_norm), which polluted parameters and caused
            # the formal-training boss_win regression. With this flag we
            # zero_grad and skip optimizer.step entirely on contaminated batches.
            spike_detected_in_step = False
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
                _future_world_aux_value = future_world_aux_loss.item()
                _future_bank_state_value = future_bank_state_loss.item()
                _future_bank_delta_value = future_bank_delta_loss.item()
                future_world_aux_loss_sum += _future_world_aux_value
                future_bank_state_loss_sum += _future_bank_state_value
                future_bank_delta_loss_sum += _future_bank_delta_value
                _spike_extra_losses = {
                    "future_bank_occupancy_loss": float(future_bank_occupancy_loss.item()),
                    "future_bank_token_presence_loss": float(future_bank_token_presence_loss.item()),
                    "future_bank_token_distribution_loss": float(future_bank_token_distribution_loss.item()),
                    "future_bank_token_slot_state_loss": float(future_bank_token_slot_state_loss.item()),
                    "future_bank_token_slot_mask_loss": float(future_bank_token_slot_mask_loss.item()),
                    "future_bank_token_slot_type_loss": float(future_bank_token_slot_type_loss.item()),
                    "future_bank_token_slot_zone_loss": float(future_bank_token_slot_zone_loss.item()),
                    "future_bank_token_slot_source_loss": float(future_bank_token_slot_source_loss.item()),
                    "policy_loss": float(policy_loss.item()) if torch.is_tensor(policy_loss) else float(policy_loss),
                    "value_loss": float(value_loss.item()) if torch.is_tensor(value_loss) else float(value_loss),
                    "reward_loss": float(reward_loss.item()) if torch.is_tensor(reward_loss) else float(reward_loss),
                    "surprise_loss": float(surprise_loss.item()) if torch.is_tensor(surprise_loss) else float(surprise_loss),
                }
                _spike_tier_flags = {
                    "weak": batch.get("sample_weak_flag"),
                    "normal": batch.get("sample_normal_flag"),
                    "elite": batch.get("sample_elite_flag"),
                    "boss": batch.get("sample_boss_flag"),
                }
                _spike_now = self._dump_loss_spike(
                    step_k=step_k,
                    future_world_aux_value=_future_world_aux_value,
                    future_bank_state_value=_future_bank_state_value,
                    future_bank_delta_value=_future_bank_delta_value,
                    batch_size=int(action_indices.shape[0]) if hasattr(action_indices, "shape") else None,
                    extra_losses=_spike_extra_losses,
                    action_indices=action_indices,
                    sample_tier_flags=_spike_tier_flags,
                )
                if _spike_now:
                    spike_detected_in_step = True
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
        # P0-7 hardening (recovery 2026-05-07): hard-skip the optimizer step
        # when any unroll-step's primary auxiliary loss crossed the spike
        # threshold. Without this, even with grad clip, the contaminated
        # gradient direction still updates parameters and degrades the policy.
        if spike_detected_in_step:
            if not getattr(self, "_loss_spike_skip_count", None):
                self._loss_spike_skip_count = 0
            self._loss_spike_skip_count += 1
            print(
                f"[loss_spike] SKIP optimizer step #{self._loss_spike_skip_count} "
                f"total_steps={int(getattr(self, 'total_steps', 0))} "
                f"total_loss={float(total_loss.item()) if torch.is_tensor(total_loss) else float(total_loss):.4g}",
                flush=True,
            )
        elif self.amp_scaler_enabled:
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
        empty_cache_called = self._maybe_empty_cuda_cache_after_train_step()

        policy_terms = float(unroll_steps + 1)
        planner_terms = float(max(planner_q_terms, 1.0))
        planner_objective_terms = float(max(planner_objective_q_terms, 1.0))
        reward_terms = float(max(unroll_steps, 1))
        surface_terms = float(max(unroll_steps, 1))
        latent_reg_terms = float(max(latent_gaussian_reg_terms, 1.0))
        surprise_metric_terms = float(max(surprise_terms, 1.0))
        metrics: dict[str, float] = {
            "loss/total": total_loss.item(),
            "amp/enabled": 1.0 if self.amp_enabled else 0.0,
            "amp/scaler_enabled": 1.0 if self.amp_scaler_enabled else 0.0,
            "amp/scaler_scale": (
                float(self.amp_grad_scaler.get_scale())
                if bool(self.amp_grad_scaler.is_enabled())
                else 1.0
            ),
            "memory/empty_cache_called": empty_cache_called,
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
        # P0-2 (recovery 2026-05-06): emit per-batch tier-quota diagnostics so
        # TensorBoard shows whether the hard quota actually held boss<=cap and
        # delivered the configured normal/elite minimums.
        bs_for_quota = max(1, int(batch_size_actual))
        if isinstance(tier_quota_info, dict):
            quotas = dict(tier_quota_info.get("quotas") or {})
            target_counts = dict(tier_quota_info.get("target_counts") or {})
            metrics["buffer/quota_boss_rate"] = float(quotas.get("boss", 0)) / bs_for_quota
            metrics["buffer/quota_elite_rate"] = float(quotas.get("elite", 0)) / bs_for_quota
            metrics["buffer/quota_normal_rate"] = float(quotas.get("normal", 0)) / bs_for_quota
            metrics["buffer/quota_weak_rate"] = float(quotas.get("weak", 0)) / bs_for_quota
            metrics["buffer/quota_fallback_rate"] = float(quotas.get("flex", 0)) / bs_for_quota
            metrics["buffer/sample_tier_target_boss"] = float(target_counts.get("boss", 0)) / bs_for_quota
            metrics["buffer/sample_tier_target_elite"] = float(target_counts.get("elite", 0)) / bs_for_quota
            metrics["buffer/sample_tier_target_normal"] = float(target_counts.get("normal", 0)) / bs_for_quota
            l1 = (
                abs(metrics["buffer/quota_boss_rate"] - metrics["buffer/sample_tier_target_boss"])
                + abs(metrics["buffer/quota_elite_rate"] - metrics["buffer/sample_tier_target_elite"])
                + abs(metrics["buffer/quota_normal_rate"] - metrics["buffer/sample_tier_target_normal"])
            )
            metrics["buffer/sample_tier_l1_to_target"] = l1
            metrics["buffer/boss_cap_hit_rate"] = 1.0 if tier_quota_info.get("boss_cap_hit") else 0.0
            metrics["buffer/normal_min_unfilled_rate"] = 1.0 if tier_quota_info.get("normal_min_unfilled") else 0.0
            metrics["buffer/elite_min_unfilled_rate"] = 1.0 if tier_quota_info.get("elite_min_unfilled") else 0.0
        else:
            # Keep tag cardinality stable when the quota path is disabled.
            for tag in (
                "buffer/quota_boss_rate",
                "buffer/quota_elite_rate",
                "buffer/quota_normal_rate",
                "buffer/quota_weak_rate",
                "buffer/quota_fallback_rate",
                "buffer/sample_tier_target_boss",
                "buffer/sample_tier_target_elite",
                "buffer/sample_tier_target_normal",
                "buffer/sample_tier_l1_to_target",
                "buffer/boss_cap_hit_rate",
                "buffer/normal_min_unfilled_rate",
                "buffer/elite_min_unfilled_rate",
            ):
                metrics[tag] = 0.0
        return metrics
