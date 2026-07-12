"""Loss helpers mixed into the MuZero trainer.

This module intentionally keeps tensor/objective/surface loss formulas out of
``muzero.train``.  It should stay policy-agnostic: no combat heuristics, route
biases, or environment side effects belong here.
"""

from __future__ import annotations

import torch

from muzero.sts2_env.muzero_model import (
    RecurrentMuZeroOutput,
    scalar_to_support,
    support_tensor_to_scalar,
    support_to_scalar,
)
from sts2_env.objective_heads import HEAD_HP_PRESERVATION, NUM_OBJECTIVE_HEADS
from sts2_env.observation_v2 import NUM_PHASES
from muzero.sts2_env.semantic_rollout import SEMANTIC_ROLLOUT_SIZE


class TrainingLossMixin:
    """Tensor loss and metric helpers for ``MuZeroTrainer``.

    The mixin expects the trainer to expose the same attributes used by the
    legacy implementation: ``network``, ``device``, future-bank weights,
    semantic smoothing, and related training hyperparameters.
    """

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

    def _objective_reward_head_loss(
        self,
        logits: torch.Tensor,
        targets: torch.Tensor,
        head_index: int,
    ) -> tuple[torch.Tensor, dict[str, float]]:
        """Single objective-reward-head CE plus scalar calibration metrics.

        This is intentionally a thin weighting hook over the existing
        objective reward decomposition.  It lets us emphasize one semantic
        transition target (currently HP preservation) without adding a new
        network head or changing checkpoint compatibility.
        """
        if logits.ndim != 3 or targets.ndim != 2:
            zero = logits.new_zeros(())
            return zero, {
                "mae": 0.0,
                "pred_mean": 0.0,
                "target_mean": 0.0,
                "active_rate": 0.0,
            }
        head_count = int(min(logits.shape[1], targets.shape[1]))
        if head_index < 0 or head_index >= head_count:
            zero = logits.new_zeros(())
            return zero, {
                "mae": 0.0,
                "pred_mean": 0.0,
                "target_mean": 0.0,
                "active_rate": 0.0,
            }

        head_logits = logits[:, head_index, :]
        head_targets = targets[:, head_index].detach().float()
        valid = torch.isfinite(head_targets)
        if not valid.any():
            zero = logits.new_zeros(())
            return zero, {
                "mae": 0.0,
                "pred_mean": 0.0,
                "target_mean": 0.0,
                "active_rate": 0.0,
            }

        head_logits = head_logits[valid]
        head_targets = head_targets[valid]
        support_targets = scalar_to_support(
            head_targets.reshape(-1),
            support_size=self.network.support_size,
        ).to(head_logits.device)
        log_probs = torch.log_softmax(head_logits, dim=-1)
        loss = -(support_targets * log_probs).sum(dim=-1).mean()
        with torch.no_grad():
            pred = support_tensor_to_scalar(head_logits, self.network.support_size).reshape(-1)
            mae = (pred - head_targets.reshape(-1)).abs().mean().item()
            pred_mean = pred.mean().item()
            target_mean = head_targets.mean().item()
            active_rate = valid.float().mean().item()
        return loss, {
            "mae": float(mae),
            "pred_mean": float(pred_mean),
            "target_mean": float(target_mean),
            "active_rate": float(active_rate),
        }

    def _hp_preservation_reward_aux_loss(
        self,
        logits: torch.Tensor,
        targets: torch.Tensor,
    ) -> tuple[torch.Tensor, dict[str, float]]:
        """Extra trainable pressure on the HP-preservation transition head."""
        return self._objective_reward_head_loss(
            logits,
            targets,
            HEAD_HP_PRESERVATION,
        )

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
