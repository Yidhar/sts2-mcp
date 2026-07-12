"""Explicit action-conditioned latent rollout planner.

The mixin has no constructor or registered modules.  It keeps the public
``MuZeroNetwork`` class and state-dict layout in their historical module while
separating the large planner algorithm from model construction/inference.
"""

from __future__ import annotations

import math
from typing import NamedTuple

import torch

from sts2_env.objective_heads import (
    HEAD_HP_PRESERVATION,
    HEAD_SURVIVAL,
    NUM_OBJECTIVE_HEADS,
    scalarize_objective_components_torch,
)
from sts2_env.observation_v2 import MAX_ACTIONS, NUM_DOMAINS, NUM_PHASES

RISK_OBJECTIVE_HEAD_INDICES = (HEAD_SURVIVAL, HEAD_HP_PRESERVATION)
DEFAULT_ACTION_ROLLOUT_BUCKETS = (8, 16, 32, 64, MAX_ACTIONS)


class ActionRolloutPlannerOutput(NamedTuple):
    """Explicit action-conditioned one-step latent rollout for search-free planning."""

    next_hidden_states: torch.Tensor
    reward_logits: torch.Tensor
    reward: torch.Tensor
    reward_component_logits: torch.Tensor
    reward_components: torch.Tensor
    next_value_logits: torch.Tensor
    next_value: torch.Tensor
    next_value_component_logits: torch.Tensor
    next_value_components: torch.Tensor
    next_objective_value: torch.Tensor
    next_action_mask_logits: torch.Tensor
    next_decision_domain_logits: torch.Tensor
    next_phase_logits: torch.Tensor
    planner_q: torch.Tensor
    planner_q_components: torch.Tensor
    planner_objective_q: torch.Tensor
    planner_risk_q: torch.Tensor
    planner_uncertainty: torch.Tensor
    planner_surprise: torch.Tensor
    planner_surface_entropy: torch.Tensor
    planner_latent_drift: torch.Tensor
    planner_branch_disagreement: torch.Tensor
    action_mask: torch.Tensor
    rollout_steps_used: int
    rollout_branch_count_mean: float
    rollout_root_valid_count: int
    rollout_root_bucket_size: int
    rollout_bucket_padding_ratio: float
    rollout_max_branch_bucket_size: int
    rollout_branch_padding_ratio: float


class ActionRolloutPlannerMixin:
    @staticmethod
    def _normalize_action_rollout_buckets(
        raw_buckets: str | tuple[int, ...] | list[int] | None,
    ) -> tuple[int, ...]:
        if raw_buckets is None:
            values = list(DEFAULT_ACTION_ROLLOUT_BUCKETS)
        elif isinstance(raw_buckets, str):
            parts = [
                part.strip()
                for part in raw_buckets.replace(";", ",").split(",")
                if part.strip()
            ]
            values = [int(part) for part in parts]
        else:
            values = [int(value) for value in raw_buckets]

        normalized = sorted({int(value) for value in values if int(value) > 0})
        if not normalized:
            normalized = list(DEFAULT_ACTION_ROLLOUT_BUCKETS)
        if normalized[-1] < MAX_ACTIONS:
            normalized.append(MAX_ACTIONS)
        return tuple(normalized)

    def _action_rollout_bucket_size(self, active_count: int) -> int:
        active_count = max(int(active_count), 0)
        if active_count <= 0:
            return 0
        for bucket in self.action_rollout_buckets:
            if active_count <= int(bucket):
                return int(bucket)
        largest = max(int(self.action_rollout_buckets[-1]), 1)
        return int(math.ceil(active_count / float(largest)) * largest)

    @staticmethod
    def _pad_first_dim_to_bucket(
        tensor: torch.Tensor,
        bucket_size: int,
        *,
        fill_value: float | int | bool = 0,
    ) -> torch.Tensor:
        active_count = int(tensor.shape[0])
        bucket_size = int(bucket_size)
        if active_count == bucket_size:
            return tensor
        if active_count > bucket_size:
            return tensor[:bucket_size]
        pad_shape = (bucket_size - active_count, *tensor.shape[1:])
        padding = tensor.new_full(pad_shape, fill_value)
        return torch.cat([tensor, padding], dim=0)

    def _planner_chunk_size(self, total_rows: int) -> int:
        total_rows = max(int(total_rows), 0)
        chunk = int(getattr(self, "action_rollout_chunk_size", 0) or 0)
        if chunk <= 0 or total_rows <= chunk:
            return max(total_rows, 1)
        return max(chunk, 1)

    def _planner_dynamics_surface_value(
        self,
        hidden: torch.Tensor,
        actions: torch.Tensor,
        *,
        objective_context: torch.Tensor | None = None,
    ) -> tuple[
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
    ]:
        total_rows = int(hidden.shape[0])
        chunk_size = self._planner_chunk_size(total_rows)
        if total_rows <= chunk_size:
            (
                next_hidden,
                reward_logits,
                reward,
                reward_component_logits,
                reward_components,
                surprise_logits,
                surprise,
            ) = self.dynamics(hidden, actions)
            (
                action_mask_logits,
                decision_domain_logits,
                phase_logits,
            ) = self.transition_surface(next_hidden)
            decision_domain = torch.softmax(decision_domain_logits, dim=-1)
            (
                value_logits,
                value,
                value_component_logits,
                value_components,
                objective_value,
            ) = self.prediction.value_only(
                next_hidden,
                decision_domain=decision_domain,
                objective_context=objective_context,
            )
            return (
                next_hidden,
                reward_logits,
                reward,
                reward_component_logits,
                reward_components,
                surprise_logits,
                surprise,
                action_mask_logits,
                decision_domain_logits,
                phase_logits,
                value_logits,
                value,
                value_component_logits,
                value_components,
                objective_value,
            )

        chunks: list[list[torch.Tensor]] = [[] for _ in range(15)]
        for start in range(0, total_rows, chunk_size):
            end = min(start + chunk_size, total_rows)
            context_chunk = None if objective_context is None else objective_context[start:end]
            outputs = self._planner_dynamics_surface_value(
                hidden[start:end],
                actions[start:end],
                objective_context=context_chunk,
            )
            for index, value in enumerate(outputs):
                chunks[index].append(value)
        return tuple(torch.cat(values, dim=0) for values in chunks)  # type: ignore[return-value]

    def _planner_value_only(
        self,
        hidden: torch.Tensor,
        *,
        decision_domain: torch.Tensor | None = None,
        objective_context: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        total_rows = int(hidden.shape[0])
        chunk_size = self._planner_chunk_size(total_rows)
        if total_rows <= chunk_size:
            return self.prediction.value_only(
                hidden,
                decision_domain=decision_domain,
                objective_context=objective_context,
            )
        chunks: list[list[torch.Tensor]] = [[] for _ in range(5)]
        for start in range(0, total_rows, chunk_size):
            end = min(start + chunk_size, total_rows)
            decision_chunk = None if decision_domain is None else decision_domain[start:end]
            context_chunk = None if objective_context is None else objective_context[start:end]
            outputs = self.prediction.value_only(
                hidden[start:end],
                decision_domain=decision_chunk,
                objective_context=context_chunk,
            )
            for index, value in enumerate(outputs):
                chunks[index].append(value)
        return tuple(torch.cat(values, dim=0) for values in chunks)  # type: ignore[return-value]

    def _planner_latent_policy_only(
        self,
        hidden: torch.Tensor,
        *,
        decision_domain: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        total_rows = int(hidden.shape[0])
        chunk_size = self._planner_chunk_size(total_rows)
        if total_rows <= chunk_size:
            return self.prediction.latent_policy_only(hidden, decision_domain=decision_domain)
        logits_chunks: list[torch.Tensor] = []
        embedding_chunks: list[torch.Tensor] = []
        for start in range(0, total_rows, chunk_size):
            end = min(start + chunk_size, total_rows)
            decision_chunk = None if decision_domain is None else decision_domain[start:end]
            logits, embeddings = self.prediction.latent_policy_only(
                hidden[start:end],
                decision_domain=decision_chunk,
            )
            logits_chunks.append(logits)
            embedding_chunks.append(embeddings)
        return torch.cat(logits_chunks, dim=0), torch.cat(embedding_chunks, dim=0)

    def action_rollout_planner(
        self,
        hidden_state: torch.Tensor,
        action_embeddings: torch.Tensor,
        *,
        action_mask: torch.Tensor | None = None,
        objective_context: torch.Tensor | None = None,
        decision_domain: torch.Tensor | None = None,
        discount: float = 0.997,
        rollout_steps: int = 1,
        continuation_beam_width: int = 2,
        continuation_legal_logit_scale: float = 0.75,
        uncertainty_surprise_weight: float = 1.0,
        uncertainty_surface_entropy_weight: float = 0.10,
        uncertainty_latent_drift_weight: float = 0.05,
        uncertainty_branch_disagreement_weight: float = 0.25,
        continuation_uncertainty_penalty: float = 0.25,
    ) -> ActionRolloutPlannerOutput:
        """Roll shared dynamics for every candidate action, optionally with deeper latent continuation.

        This is the explicit search-free planner path:
        - generate next latent state for each candidate action
        - predict next-step value/objective value from the shared value heads
        - optionally continue with latent-action beam rollout using predicted legality/surface
        - combine discounted rewards + terminal bootstrap into Q-like action scores
        """

        batch_size, action_count, _ = action_embeddings.shape
        device = hidden_state.device
        rollout_steps = max(int(rollout_steps), 1)
        continuation_beam_width = max(int(continuation_beam_width), 1)
        if action_mask is None:
            action_mask = action_embeddings.abs().sum(dim=-1) > 0
        else:
            action_mask = action_mask.to(device=device, dtype=torch.bool)
            if action_mask.dim() == 1:
                action_mask = action_mask.unsqueeze(0)
        action_mask = action_mask.reshape(batch_size, action_count)
        flat_valid = action_mask.reshape(-1)
        valid_indices = torch.nonzero(flat_valid, as_tuple=False).flatten()
        num_valid = int(valid_indices.numel())
        root_bucket_size = self._action_rollout_bucket_size(num_valid)
        root_bucket_padding_ratio = (
            float(root_bucket_size - num_valid) / float(root_bucket_size)
            if root_bucket_size > 0
            else 0.0
        )
        num_bins = 2 * self.support_size + 1

        def _zeros(*shape: int, dtype: torch.dtype | None = None) -> torch.Tensor:
            return torch.zeros(*shape, device=device, dtype=dtype or hidden_state.dtype)

        if num_valid <= 0:
            return ActionRolloutPlannerOutput(
                next_hidden_states=_zeros(batch_size, action_count, self.hidden_dim),
                reward_logits=_zeros(batch_size, action_count, num_bins),
                reward=_zeros(batch_size, action_count),
                reward_component_logits=_zeros(batch_size, action_count, NUM_OBJECTIVE_HEADS, num_bins),
                reward_components=_zeros(batch_size, action_count, NUM_OBJECTIVE_HEADS),
                next_value_logits=_zeros(batch_size, action_count, num_bins),
                next_value=_zeros(batch_size, action_count),
                next_value_component_logits=_zeros(batch_size, action_count, NUM_OBJECTIVE_HEADS, num_bins),
                next_value_components=_zeros(batch_size, action_count, NUM_OBJECTIVE_HEADS),
                next_objective_value=_zeros(batch_size, action_count),
                next_action_mask_logits=_zeros(batch_size, action_count, MAX_ACTIONS),
                next_decision_domain_logits=_zeros(batch_size, action_count, NUM_DOMAINS),
                next_phase_logits=_zeros(batch_size, action_count, NUM_PHASES),
                planner_q=_zeros(batch_size, action_count),
                planner_q_components=_zeros(batch_size, action_count, NUM_OBJECTIVE_HEADS),
                planner_objective_q=_zeros(batch_size, action_count),
                planner_risk_q=_zeros(batch_size, action_count),
                planner_uncertainty=_zeros(batch_size, action_count),
                planner_surprise=_zeros(batch_size, action_count),
                planner_surface_entropy=_zeros(batch_size, action_count),
                planner_latent_drift=_zeros(batch_size, action_count),
                planner_branch_disagreement=_zeros(batch_size, action_count),
                action_mask=action_mask,
                rollout_steps_used=0,
                rollout_branch_count_mean=0.0,
                rollout_root_valid_count=0,
                rollout_root_bucket_size=0,
                rollout_bucket_padding_ratio=0.0,
                rollout_max_branch_bucket_size=0,
                rollout_branch_padding_ratio=0.0,
            )

        hidden_expanded = hidden_state.unsqueeze(1).expand(-1, action_count, -1)
        root_active_mask = torch.arange(root_bucket_size, device=device) < num_valid
        root_active_float = root_active_mask.to(dtype=hidden_state.dtype)
        safe_valid_indices = self._pad_first_dim_to_bucket(valid_indices, root_bucket_size, fill_value=0)
        flat_hidden = hidden_expanded.reshape(batch_size * action_count, self.hidden_dim).index_select(
            0,
            safe_valid_indices,
        )
        flat_actions = action_embeddings.reshape(batch_size * action_count, -1).index_select(0, safe_valid_indices)
        flat_hidden = flat_hidden * root_active_float.unsqueeze(-1)
        flat_actions = flat_actions * root_active_float.unsqueeze(-1)

        repeated_objective_context: torch.Tensor | None = None
        bucket_objective_context: torch.Tensor | None = None
        if objective_context is not None:
            if objective_context.dim() == 1:
                objective_context = objective_context.unsqueeze(0)
            bucket_objective_context = (
                objective_context
                .unsqueeze(1)
                .expand(-1, action_count, -1)
                .reshape(batch_size * action_count, -1)
                .index_select(0, safe_valid_indices)
            )
            bucket_objective_context = bucket_objective_context * root_active_float.unsqueeze(-1)
            repeated_objective_context = bucket_objective_context[:num_valid]

        (
            next_hidden_valid,
            reward_logits_valid,
            reward_valid,
            reward_component_logits_valid,
            reward_components_valid,
            surprise_logits_valid,
            surprise_valid,
            next_action_mask_logits_valid,
            next_decision_domain_logits_valid,
            next_phase_logits_valid,
            next_value_logits_valid,
            next_value_valid,
            next_value_component_logits_valid,
            next_value_components_valid,
            next_objective_value_valid,
        ) = self._planner_dynamics_surface_value(
            flat_hidden,
            flat_actions,
            objective_context=bucket_objective_context,
        )
        if reward_valid.dim() > 1 and reward_valid.shape[-1] == 1:
            reward_valid = reward_valid.squeeze(-1)
        if next_value_valid.dim() > 1 and next_value_valid.shape[-1] == 1:
            next_value_valid = next_value_valid.squeeze(-1)
        if next_objective_value_valid.dim() > 1 and next_objective_value_valid.shape[-1] == 1:
            next_objective_value_valid = next_objective_value_valid.squeeze(-1)
        if surprise_valid.dim() > 1 and surprise_valid.shape[-1] == 1:
            surprise_valid = surprise_valid.squeeze(-1)

        next_hidden_valid = next_hidden_valid[:num_valid]
        reward_logits_valid = reward_logits_valid[:num_valid]
        reward_valid = reward_valid[:num_valid]
        reward_component_logits_valid = reward_component_logits_valid[:num_valid]
        reward_components_valid = reward_components_valid[:num_valid]
        surprise_valid = surprise_valid[:num_valid]
        next_action_mask_logits_valid = next_action_mask_logits_valid[:num_valid]
        next_decision_domain_logits_valid = next_decision_domain_logits_valid[:num_valid]
        next_phase_logits_valid = next_phase_logits_valid[:num_valid]
        next_value_logits_valid = next_value_logits_valid[:num_valid]
        next_value_valid = next_value_valid[:num_valid]
        next_value_component_logits_valid = next_value_component_logits_valid[:num_valid]
        next_value_components_valid = next_value_components_valid[:num_valid]
        next_objective_value_valid = next_objective_value_valid[:num_valid]

        planner_q_components_valid = reward_components_valid + float(discount) * next_value_components_valid
        planner_q_valid = reward_valid + float(discount) * next_value_valid
        planner_objective_q_valid = scalarize_objective_components_torch(
            planner_q_components_valid,
            repeated_objective_context,
        )
        planner_risk_q_valid = planner_q_components_valid[..., list(RISK_OBJECTIVE_HEAD_INDICES)].mean(dim=-1)
        action_surface_prob = torch.sigmoid(next_action_mask_logits_valid)
        surface_entropy_valid = -(
            action_surface_prob * action_surface_prob.clamp(min=1e-6).log()
            + (1.0 - action_surface_prob) * (1.0 - action_surface_prob).clamp(min=1e-6).log()
        ).mean(dim=-1)
        expected_norm = math.sqrt(float(max(self.hidden_dim, 1)))
        latent_drift_valid = (next_hidden_valid.norm(dim=-1) / expected_norm - 1.0).abs()
        planner_surprise_valid = surprise_valid
        planner_surface_entropy_valid = surface_entropy_valid
        planner_latent_drift_valid = latent_drift_valid
        planner_branch_disagreement_valid = planner_q_valid.new_zeros(planner_q_valid.shape)
        planner_uncertainty_valid = (
            float(uncertainty_surprise_weight) * planner_surprise_valid
            + float(uncertainty_surface_entropy_weight) * planner_surface_entropy_valid
            + float(uncertainty_latent_drift_weight) * planner_latent_drift_valid
            + float(uncertainty_branch_disagreement_weight) * planner_branch_disagreement_valid
        )

        rollout_steps_used = 1
        rollout_branch_count_mean = 1.0
        rollout_max_branch_bucket_size = 0
        rollout_branch_bucket_total = 0.0
        rollout_branch_padding_total = 0.0

        if rollout_steps > 1:
            branch_root_index = torch.arange(num_valid, device=device, dtype=torch.long)
            branch_hidden = next_hidden_valid
            branch_action_mask_logits = next_action_mask_logits_valid
            branch_decision_domain_logits = next_decision_domain_logits_valid
            branch_return = reward_valid
            branch_components = reward_components_valid
            branch_surprise = planner_surprise_valid
            branch_surface_entropy = planner_surface_entropy_valid
            branch_latent_drift = planner_latent_drift_valid
            branch_weight = torch.ones_like(branch_return)
            branch_discount_power = torch.full_like(branch_return, float(discount))
            accumulated_branch_states = float(branch_root_index.numel())

            for depth in range(1, rollout_steps):
                if branch_hidden.numel() == 0:
                    break
                branch_active_count = int(branch_hidden.shape[0])
                branch_bucket_size = self._action_rollout_bucket_size(branch_active_count)
                rollout_max_branch_bucket_size = max(rollout_max_branch_bucket_size, int(branch_bucket_size))
                rollout_branch_bucket_total += float(branch_bucket_size)
                rollout_branch_padding_total += float(max(branch_bucket_size - branch_active_count, 0))
                branch_hidden_bucket = self._pad_first_dim_to_bucket(branch_hidden, branch_bucket_size)
                branch_decision_domain_logits_bucket = self._pad_first_dim_to_bucket(
                    branch_decision_domain_logits,
                    branch_bucket_size,
                )
                current_decision_domain = torch.softmax(branch_decision_domain_logits_bucket, dim=-1)
                latent_policy_logits, latent_action_embeddings = self._planner_latent_policy_only(
                    branch_hidden_bucket,
                    decision_domain=current_decision_domain,
                )
                latent_policy_logits = latent_policy_logits[:branch_active_count]
                latent_action_embeddings = latent_action_embeddings[:branch_active_count]
                legal_mask = branch_action_mask_logits > 0.0
                any_legal = legal_mask.any(dim=-1, keepdim=True)
                fallback_mask = torch.ones_like(legal_mask, dtype=torch.bool)
                effective_mask = torch.where(any_legal, legal_mask, fallback_mask)
                effective_scores = latent_policy_logits + float(continuation_legal_logit_scale) * branch_action_mask_logits
                effective_scores = effective_scores.masked_fill(~effective_mask, -1e9)
                branch_k = min(continuation_beam_width, effective_scores.shape[1])
                top_scores, top_indices = torch.topk(effective_scores, k=branch_k, dim=-1)
                top_probs = torch.softmax(top_scores, dim=-1)
                top_probs = torch.where(
                    torch.isfinite(top_probs),
                    top_probs,
                    torch.full_like(top_probs, 1.0 / max(branch_k, 1)),
                )
                gathered_indices = top_indices.unsqueeze(-1).expand(-1, -1, latent_action_embeddings.shape[-1])
                top_action_embeddings = latent_action_embeddings.gather(1, gathered_indices)

                branch_count = branch_hidden.shape[0]
                expanded_hidden = (
                    branch_hidden.unsqueeze(1)
                    .expand(-1, branch_k, -1)
                    .reshape(branch_count * branch_k, self.hidden_dim)
                )
                expanded_actions = top_action_embeddings.reshape(branch_count * branch_k, self.action_embed_dim)
                expanded_root_index = (
                    branch_root_index.unsqueeze(1)
                    .expand(-1, branch_k)
                    .reshape(branch_count * branch_k)
                )
                expanded_weight = (
                    branch_weight.unsqueeze(1)
                    .expand(-1, branch_k)
                    .reshape(branch_count * branch_k)
                    * top_probs.reshape(branch_count * branch_k)
                )
                expanded_discount_power = (
                    branch_discount_power.unsqueeze(1)
                    .expand(-1, branch_k)
                    .reshape(branch_count * branch_k)
                )
                expanded_return = (
                    branch_return.unsqueeze(1)
                    .expand(-1, branch_k)
                    .reshape(branch_count * branch_k)
                )
                expanded_components = (
                    branch_components.unsqueeze(1)
                    .expand(-1, branch_k, -1)
                    .reshape(branch_count * branch_k, NUM_OBJECTIVE_HEADS)
                )
                expanded_surprise = (
                    branch_surprise.unsqueeze(1)
                    .expand(-1, branch_k)
                    .reshape(branch_count * branch_k)
                )
                expanded_surface_entropy = (
                    branch_surface_entropy.unsqueeze(1)
                    .expand(-1, branch_k)
                    .reshape(branch_count * branch_k)
                )
                expanded_latent_drift = (
                    branch_latent_drift.unsqueeze(1)
                    .expand(-1, branch_k)
                    .reshape(branch_count * branch_k)
                )
                expanded_objective_context: torch.Tensor | None = None
                if repeated_objective_context is not None:
                    expanded_objective_context = repeated_objective_context[expanded_root_index]
                expanded_active_count = int(expanded_hidden.shape[0])
                expanded_bucket_size = self._action_rollout_bucket_size(expanded_active_count)
                rollout_max_branch_bucket_size = max(rollout_max_branch_bucket_size, int(expanded_bucket_size))
                rollout_branch_bucket_total += float(expanded_bucket_size)
                rollout_branch_padding_total += float(max(expanded_bucket_size - expanded_active_count, 0))
                expanded_hidden_bucket = self._pad_first_dim_to_bucket(expanded_hidden, expanded_bucket_size)
                expanded_actions_bucket = self._pad_first_dim_to_bucket(expanded_actions, expanded_bucket_size)
                expanded_objective_context_bucket: torch.Tensor | None = None
                if expanded_objective_context is not None:
                    expanded_objective_context_bucket = self._pad_first_dim_to_bucket(
                        expanded_objective_context,
                        expanded_bucket_size,
                    )

                (
                    continued_hidden,
                    _continued_reward_logits,
                    continued_reward,
                    _continued_reward_component_logits,
                    continued_reward_components,
                    _continued_surprise_logits,
                    continued_surprise,
                    continued_action_mask_logits,
                    continued_decision_domain_logits,
                    _continued_phase_logits,
                    _continued_value_logits,
                    continued_value,
                    _continued_value_component_logits,
                    continued_value_components,
                    _continued_objective_value,
                ) = self._planner_dynamics_surface_value(
                    expanded_hidden_bucket,
                    expanded_actions_bucket,
                    objective_context=expanded_objective_context_bucket,
                )
                if continued_reward.dim() > 1 and continued_reward.shape[-1] == 1:
                    continued_reward = continued_reward.squeeze(-1)
                if continued_value.dim() > 1 and continued_value.shape[-1] == 1:
                    continued_value = continued_value.squeeze(-1)
                if continued_surprise.dim() > 1 and continued_surprise.shape[-1] == 1:
                    continued_surprise = continued_surprise.squeeze(-1)
                continued_hidden = continued_hidden[:expanded_active_count]
                continued_reward = continued_reward[:expanded_active_count]
                continued_reward_components = continued_reward_components[:expanded_active_count]
                continued_surprise = continued_surprise[:expanded_active_count]
                continued_action_mask_logits = continued_action_mask_logits[:expanded_active_count]
                continued_decision_domain_logits = continued_decision_domain_logits[:expanded_active_count]
                continued_value = continued_value[:expanded_active_count]
                continued_value_components = continued_value_components[:expanded_active_count]

                continued_return = expanded_return + expanded_discount_power * continued_reward
                continued_components = (
                    expanded_components
                    + expanded_discount_power.unsqueeze(-1) * continued_reward_components
                )
                continued_surface_prob = torch.sigmoid(continued_action_mask_logits)
                continued_surface_entropy = -(
                    continued_surface_prob * continued_surface_prob.clamp(min=1e-6).log()
                    + (1.0 - continued_surface_prob)
                    * (1.0 - continued_surface_prob).clamp(min=1e-6).log()
                ).mean(dim=-1)
                continued_latent_drift = (continued_hidden.norm(dim=-1) / expected_norm - 1.0).abs()
                continued_surprise_total = expanded_surprise + expanded_discount_power * continued_surprise
                continued_surface_entropy_total = (
                    expanded_surface_entropy + expanded_discount_power * continued_surface_entropy
                )
                continued_latent_drift_total = expanded_latent_drift + expanded_discount_power * continued_latent_drift
                continued_uncertainty = (
                    float(uncertainty_surprise_weight) * continued_surprise_total
                    + float(uncertainty_surface_entropy_weight) * continued_surface_entropy_total
                    + float(uncertainty_latent_drift_weight) * continued_latent_drift_total
                )
                next_discount_power = expanded_discount_power * float(discount)
                prune_q_components = (
                    continued_components
                    + next_discount_power.unsqueeze(-1) * continued_value_components
                )
                prune_scores = scalarize_objective_components_torch(
                    prune_q_components,
                    expanded_objective_context,
                )
                prune_scores = prune_scores - float(continuation_uncertainty_penalty) * continued_uncertainty

                # Scheme-B: keep continuation pruning in the same allocator-stable,
                # root-major layout as the heavy dynamics/value calls above.
                #
                # Invariant maintained by this planner:
                #   - branch tensors are root-major: all paths for root 0, then root 1, ...
                #   - every active branch expands the same branch_k latent actions.  When the
                #     predicted legal surface is empty we deliberately use a full fallback mask,
                #     so a root never disappears mid-rollout.
                #
                # Therefore candidates per root are regular and can be reshaped to
                # [num_roots, candidates_per_root].  The previous implementation used
                # per-root torch.nonzero + Python list + torch.cat, producing many small
                # dynamic tensors on ROCm/HIP.  Here topk runs on a root-bucket-padded
                # score matrix and only the active rows are flattened back to indices.
                if expanded_active_count <= 0:
                    break
                candidates_per_root = expanded_active_count // max(num_valid, 1)
                if candidates_per_root <= 0 or candidates_per_root * num_valid != expanded_active_count:
                    raise RuntimeError(
                        "action_rollout_planner expected root-major regular continuation "
                        f"layout, got expanded_active_count={expanded_active_count}, "
                        f"num_valid={num_valid}, branch_count={branch_count}, branch_k={branch_k}"
                    )
                root_keep_k = min(continuation_beam_width, candidates_per_root)
                grouped_prune_scores = prune_scores.reshape(num_valid, candidates_per_root)
                grouped_prune_scores_bucket = prune_scores.new_full(
                    (root_bucket_size, candidates_per_root),
                    -1e9,
                )
                grouped_prune_scores_bucket[:num_valid] = grouped_prune_scores
                _, local_keep_bucket = torch.topk(
                    grouped_prune_scores_bucket,
                    k=root_keep_k,
                    dim=-1,
                )
                local_keep = local_keep_bucket[:num_valid]
                root_offsets = (
                    torch.arange(num_valid, device=device, dtype=torch.long).unsqueeze(1)
                    * candidates_per_root
                )
                keep_indices = (root_offsets + local_keep).reshape(num_valid * root_keep_k)

                branch_root_index = expanded_root_index[keep_indices]
                branch_hidden = continued_hidden[keep_indices]
                branch_action_mask_logits = continued_action_mask_logits[keep_indices]
                branch_decision_domain_logits = continued_decision_domain_logits[keep_indices]
                branch_return = continued_return[keep_indices]
                branch_components = continued_components[keep_indices]
                branch_surprise = continued_surprise_total[keep_indices]
                branch_surface_entropy = continued_surface_entropy_total[keep_indices]
                branch_latent_drift = continued_latent_drift_total[keep_indices]
                branch_weight = expanded_weight[keep_indices]
                branch_discount_power = next_discount_power[keep_indices]
                rollout_steps_used = depth + 1
                accumulated_branch_states += float(branch_root_index.numel())

            if branch_hidden.numel() > 0:
                final_branch_count = int(branch_hidden.shape[0])
                final_bucket_size = self._action_rollout_bucket_size(final_branch_count)
                rollout_max_branch_bucket_size = max(rollout_max_branch_bucket_size, int(final_bucket_size))
                rollout_branch_bucket_total += float(final_bucket_size)
                rollout_branch_padding_total += float(max(final_bucket_size - final_branch_count, 0))
                branch_hidden_bucket = self._pad_first_dim_to_bucket(branch_hidden, final_bucket_size)
                branch_decision_domain_logits_bucket = self._pad_first_dim_to_bucket(
                    branch_decision_domain_logits,
                    final_bucket_size,
                )
                branch_decision_domain = torch.softmax(branch_decision_domain_logits_bucket, dim=-1)
                branch_objective_context: torch.Tensor | None = None
                if repeated_objective_context is not None:
                    branch_objective_context = repeated_objective_context[branch_root_index]
                    branch_objective_context = self._pad_first_dim_to_bucket(
                        branch_objective_context,
                        final_bucket_size,
                    )
                (
                    _final_value_logits,
                    final_branch_value,
                    _final_value_component_logits,
                    final_branch_value_components,
                    _final_objective_value,
                ) = self._planner_value_only(
                    branch_hidden_bucket,
                    decision_domain=branch_decision_domain,
                    objective_context=branch_objective_context,
                )
                if final_branch_value.dim() > 1 and final_branch_value.shape[-1] == 1:
                    final_branch_value = final_branch_value.squeeze(-1)
                final_branch_value = final_branch_value[:final_branch_count]
                final_branch_value_components = final_branch_value_components[:final_branch_count]
                final_branch_q = branch_return + branch_discount_power * final_branch_value
                final_branch_q_components = (
                    branch_components
                    + branch_discount_power.unsqueeze(-1) * final_branch_value_components
                )
                aggregated_q = final_branch_q.new_zeros((num_valid,))
                aggregated_q_components = final_branch_q_components.new_zeros((num_valid, NUM_OBJECTIVE_HEADS))
                aggregated_q_second_moment = final_branch_q.new_zeros((num_valid,))
                aggregated_surprise = final_branch_q.new_zeros((num_valid,))
                aggregated_surface_entropy = final_branch_q.new_zeros((num_valid,))
                aggregated_latent_drift = final_branch_q.new_zeros((num_valid,))
                aggregated_weights = final_branch_q.new_zeros((num_valid,))
                aggregated_q.index_add_(0, branch_root_index, branch_weight * final_branch_q)
                aggregated_q_second_moment.index_add_(
                    0,
                    branch_root_index,
                    branch_weight * final_branch_q.square(),
                )
                aggregated_q_components.index_add_(
                    0,
                    branch_root_index,
                    branch_weight.unsqueeze(-1) * final_branch_q_components,
                )
                aggregated_surprise.index_add_(0, branch_root_index, branch_weight * branch_surprise)
                aggregated_surface_entropy.index_add_(
                    0,
                    branch_root_index,
                    branch_weight * branch_surface_entropy,
                )
                aggregated_latent_drift.index_add_(0, branch_root_index, branch_weight * branch_latent_drift)
                aggregated_weights.index_add_(0, branch_root_index, branch_weight)
                normalized_weights = aggregated_weights.clamp(min=1e-6)
                planner_q_valid = aggregated_q / normalized_weights
                planner_q_components_valid = aggregated_q_components / normalized_weights.unsqueeze(-1)
                branch_q_second_moment_valid = aggregated_q_second_moment / normalized_weights
                planner_branch_disagreement_valid = (
                    branch_q_second_moment_valid - planner_q_valid.square()
                ).clamp(min=1e-8).sqrt()
                planner_surprise_valid = aggregated_surprise / normalized_weights
                planner_surface_entropy_valid = aggregated_surface_entropy / normalized_weights
                planner_latent_drift_valid = aggregated_latent_drift / normalized_weights
                planner_uncertainty_valid = (
                    float(uncertainty_surprise_weight) * planner_surprise_valid
                    + float(uncertainty_surface_entropy_weight) * planner_surface_entropy_valid
                    + float(uncertainty_latent_drift_weight) * planner_latent_drift_valid
                    + float(uncertainty_branch_disagreement_weight) * planner_branch_disagreement_valid
                )
                planner_objective_q_valid = scalarize_objective_components_torch(
                    planner_q_components_valid,
                    repeated_objective_context,
                )
                planner_risk_q_valid = planner_q_components_valid[..., list(RISK_OBJECTIVE_HEAD_INDICES)].mean(dim=-1)
                rollout_branch_count_mean = accumulated_branch_states / max(float(num_valid * rollout_steps_used), 1.0)

        rollout_branch_padding_ratio = (
            rollout_branch_padding_total / rollout_branch_bucket_total
            if rollout_branch_bucket_total > 0.0
            else 0.0
        )

        def _scatter(values: torch.Tensor) -> torch.Tensor:
            out = values.new_zeros((batch_size * action_count, *values.shape[1:]))
            out[flat_valid] = values
            return out.reshape(batch_size, action_count, *values.shape[1:])

        return ActionRolloutPlannerOutput(
            next_hidden_states=_scatter(next_hidden_valid),
            reward_logits=_scatter(reward_logits_valid),
            reward=_scatter(reward_valid),
            reward_component_logits=_scatter(reward_component_logits_valid),
            reward_components=_scatter(reward_components_valid),
            next_value_logits=_scatter(next_value_logits_valid),
            next_value=_scatter(next_value_valid),
            next_value_component_logits=_scatter(next_value_component_logits_valid),
            next_value_components=_scatter(next_value_components_valid),
            next_objective_value=_scatter(next_objective_value_valid),
            next_action_mask_logits=_scatter(next_action_mask_logits_valid),
            next_decision_domain_logits=_scatter(next_decision_domain_logits_valid),
            next_phase_logits=_scatter(next_phase_logits_valid),
            planner_q=_scatter(planner_q_valid),
            planner_q_components=_scatter(planner_q_components_valid),
            planner_objective_q=_scatter(planner_objective_q_valid),
            planner_risk_q=_scatter(planner_risk_q_valid),
            planner_uncertainty=_scatter(planner_uncertainty_valid),
            planner_surprise=_scatter(planner_surprise_valid),
            planner_surface_entropy=_scatter(planner_surface_entropy_valid),
            planner_latent_drift=_scatter(planner_latent_drift_valid),
            planner_branch_disagreement=_scatter(planner_branch_disagreement_valid),
            action_mask=action_mask,
            rollout_steps_used=rollout_steps_used,
            rollout_branch_count_mean=float(rollout_branch_count_mean),
            rollout_root_valid_count=int(num_valid),
            rollout_root_bucket_size=int(root_bucket_size),
            rollout_bucket_padding_ratio=float(root_bucket_padding_ratio),
            rollout_max_branch_bucket_size=int(rollout_max_branch_bucket_size),
            rollout_branch_padding_ratio=float(rollout_branch_padding_ratio),
        )
