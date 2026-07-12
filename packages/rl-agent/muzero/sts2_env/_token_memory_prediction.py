"""Slot-aware policy, value, and planner prediction network."""

from __future__ import annotations

import torch
import torch.nn as nn

from muzero.sts2_env.semantic_rollout import SEMANTIC_ROLLOUT_SIZE
from sts2_env.attention_blocks import (
    CrossAttentionBlock,
    EntityPooling,
    TransformerEncoderBlock,
)
from sts2_env.objective_heads import (
    NUM_OBJECTIVE_HEADS,
    scalarize_objective_components_torch,
)
from sts2_env.observation_v2 import MAX_ACTIONS, NUM_DOMAINS

from ._token_memory_shared import (
    MEMORY_BANK_NAMES,
    MEMORY_SLOT_LAYOUT_LEGACY,
    RISK_OBJECTIVE_HEAD_INDICES,
    _run_cross_block,
    _run_encoder_block,
    _run_function_with_checkpoint,
    build_memory_slot_bank_ids,
    normalize_memory_slot_layout,
)
from .value_support import support_tensor_to_scalar, support_to_scalar


class TokenPredictionNetwork(nn.Module):
    """Slot-aware prediction head for token-memory MuZero."""

    def __init__(
        self,
        *,
        hidden_dim: int,
        action_embed_dim: int,
        d_model: int,
        num_memory_slots: int,
        memory_slot_layout: str = MEMORY_SLOT_LAYOUT_LEGACY,
        support_size: int = 25,
        internal_planner_blend: float = 0.7,
        internal_planner_q_blend: float = 0.5,
        internal_planner_objective_q_blend: float = 0.35,
        internal_planner_risk_blend: float = 0.25,
        n_heads: int = 4,
        ffn_dim: int = 512,
        dropout: float = 0.0,
        activation_checkpointing: bool = False,
        planner_action_chunk_size: int = 0,
    ) -> None:
        super().__init__()
        self.hidden_dim = int(hidden_dim)
        self.action_embed_dim = int(action_embed_dim)
        self.d_model = int(d_model)
        self.num_memory_slots = int(num_memory_slots)
        self.memory_slot_layout = normalize_memory_slot_layout(memory_slot_layout)
        self.support_size = int(support_size)
        self.num_bins = 2 * self.support_size + 1
        self.internal_planner_blend = float(internal_planner_blend)
        self.internal_planner_q_blend = float(internal_planner_q_blend)
        self.internal_planner_objective_q_blend = float(internal_planner_objective_q_blend)
        self.internal_planner_risk_blend = float(internal_planner_risk_blend)
        self.n_heads = max(1, int(n_heads))
        self.ffn_dim = int(ffn_dim)
        self.activation_checkpointing = bool(activation_checkpointing)
        self.planner_action_chunk_size = max(int(planner_action_chunk_size or 0), 0)
        if self.hidden_dim != self.d_model * self.num_memory_slots:
            raise ValueError(
                f"TokenPredictionNetwork expects hidden_dim == d_model * num_memory_slots, "
                f"got {self.hidden_dim} vs {self.d_model} * {self.num_memory_slots}."
            )

        self.num_slot_banks = len(MEMORY_BANK_NAMES)
        self.register_buffer(
            "_slot_bank_ids",
            torch.as_tensor(
                build_memory_slot_bank_ids(self.num_memory_slots, layout=self.memory_slot_layout),
                dtype=torch.long,
            ),
            persistent=False,
        )
        self.slot_index_embedding = nn.Embedding(self.num_memory_slots, self.d_model)
        self.slot_bank_embedding = nn.Embedding(self.num_slot_banks, self.d_model)
        self.latent_type_embeddings = nn.Embedding(2, self.d_model)
        self.slot_blocks = nn.ModuleList(
            [TransformerEncoderBlock(self.d_model, self.n_heads, self.ffn_dim, dropout=dropout) for _ in range(2)]
        )
        self.slot_pool = EntityPooling(self.d_model, self.n_heads, dropout=dropout)
        self.bank_pool = EntityPooling(self.d_model, self.n_heads, dropout=dropout)
        self.bank_state_refine = TransformerEncoderBlock(self.d_model, self.n_heads, self.ffn_dim, dropout=dropout)

        self.domain_seed = nn.Parameter(torch.randn(1, NUM_DOMAINS, self.d_model) * 0.02)
        self.domain_cross = CrossAttentionBlock(self.d_model, self.n_heads, self.ffn_dim, dropout=dropout)
        self.domain_refine = TransformerEncoderBlock(self.d_model, self.n_heads, self.ffn_dim, dropout=dropout)
        self.domain_gate = nn.Sequential(
            nn.LayerNorm(self.d_model),
            nn.Linear(self.d_model, self.d_model),
            nn.GELU(),
            nn.Linear(self.d_model, NUM_DOMAINS),
        )

        self.candidate_in_proj = nn.Sequential(
            nn.LayerNorm(self.action_embed_dim),
            nn.Linear(self.action_embed_dim, self.d_model),
            nn.GELU(),
            nn.Linear(self.d_model, self.d_model),
        )
        self.candidate_cross = CrossAttentionBlock(self.d_model, self.n_heads, self.ffn_dim, dropout=dropout)
        self.candidate_bank_cross = CrossAttentionBlock(self.d_model, self.n_heads, self.ffn_dim, dropout=dropout)
        self.candidate_refine = TransformerEncoderBlock(self.d_model, self.n_heads, self.ffn_dim, dropout=dropout)
        self.candidate_score_heads = nn.ModuleList(
            [
                nn.Sequential(
                    nn.LayerNorm(self.d_model),
                    nn.Linear(self.d_model, self.d_model),
                    nn.GELU(),
                    nn.Linear(self.d_model, 1),
                )
                for _ in range(NUM_DOMAINS)
            ]
        )
        self.planner_bank_seed = nn.Parameter(torch.randn(1, self.num_slot_banks, self.d_model) * 0.02)
        self.planner_slot_cross = CrossAttentionBlock(self.d_model, self.n_heads, self.ffn_dim, dropout=dropout)
        self.planner_bank_cross = CrossAttentionBlock(self.d_model, self.n_heads, self.ffn_dim, dropout=dropout)
        self.planner_refine = TransformerEncoderBlock(self.d_model, self.n_heads, self.ffn_dim, dropout=dropout)
        self.planner_pool = EntityPooling(self.d_model, self.n_heads, dropout=dropout)
        self.planner_score_heads = nn.ModuleList(
            [
                nn.Sequential(
                    nn.LayerNorm(self.d_model),
                    nn.Linear(self.d_model, self.d_model),
                    nn.GELU(),
                    nn.Linear(self.d_model, 1),
                )
                for _ in range(NUM_DOMAINS)
            ]
        )
        self.planner_q_heads = nn.ModuleList(
            [
                nn.Sequential(
                    nn.LayerNorm(self.d_model),
                    nn.Linear(self.d_model, self.d_model),
                    nn.GELU(),
                    nn.Linear(self.d_model, self.num_bins),
                )
                for _ in range(NUM_DOMAINS)
            ]
        )
        self.planner_q_component_heads = nn.ModuleList(
            [
                nn.Sequential(
                    nn.LayerNorm(self.d_model),
                    nn.Linear(self.d_model, self.d_model),
                    nn.GELU(),
                    nn.Linear(self.d_model, self.num_bins * NUM_OBJECTIVE_HEADS),
                )
                for _ in range(NUM_DOMAINS)
            ]
        )

        self.latent_bank_queries = nn.Parameter(torch.randn(NUM_DOMAINS, MAX_ACTIONS, self.d_model) * 0.02)
        self.latent_bank_cross = nn.ModuleList(
            [CrossAttentionBlock(self.d_model, self.n_heads, self.ffn_dim, dropout=dropout) for _ in range(NUM_DOMAINS)]
        )
        self.latent_bank_bank_cross = nn.ModuleList(
            [CrossAttentionBlock(self.d_model, self.n_heads, self.ffn_dim, dropout=dropout) for _ in range(NUM_DOMAINS)]
        )
        self.latent_bank_refine = nn.ModuleList(
            [TransformerEncoderBlock(self.d_model, self.n_heads, self.ffn_dim, dropout=dropout) for _ in range(NUM_DOMAINS)]
        )
        self.latent_policy_score_heads = nn.ModuleList(
            [
                nn.Sequential(
                    nn.LayerNorm(self.d_model),
                    nn.Linear(self.d_model, self.d_model),
                    nn.GELU(),
                    nn.Linear(self.d_model, 1),
                )
                for _ in range(NUM_DOMAINS)
            ]
        )
        self.latent_action_proj = nn.Sequential(
            nn.LayerNorm(self.d_model),
            nn.Linear(self.d_model, self.d_model),
            nn.GELU(),
            nn.Linear(self.d_model, self.action_embed_dim),
        )

        self.value_heads = nn.ModuleList(
            [
                nn.Sequential(
                    nn.LayerNorm(self.d_model),
                    nn.Linear(self.d_model, self.d_model),
                    nn.GELU(),
                    nn.Linear(self.d_model, self.num_bins),
                )
                for _ in range(NUM_DOMAINS)
            ]
        )
        self.value_component_heads = nn.ModuleList(
            [
                nn.Sequential(
                    nn.LayerNorm(self.d_model),
                    nn.Linear(self.d_model, self.d_model),
                    nn.GELU(),
                    nn.Linear(self.d_model, self.num_bins * NUM_OBJECTIVE_HEADS),
                )
                for _ in range(NUM_DOMAINS)
            ]
        )
        self.semantic_policy_head = nn.Sequential(
            nn.LayerNorm(self.d_model),
            nn.Linear(self.d_model, self.d_model),
            nn.GELU(),
            nn.Linear(self.d_model, SEMANTIC_ROLLOUT_SIZE),
        )

    def _unflatten(self, hidden_state: torch.Tensor) -> torch.Tensor:
        return hidden_state.reshape(hidden_state.shape[0], self.num_memory_slots, self.d_model)

    def _slot_mask(self, batch_size: int, device: torch.device) -> torch.Tensor:
        return torch.ones((batch_size, self.num_memory_slots), dtype=torch.bool, device=device)

    def _slot_bank_id_tensor(self, device: torch.device) -> torch.Tensor:
        return self._slot_bank_ids.to(device=device)

    def _slot_states(self, hidden_state: torch.Tensor, *, latent_type_id: int) -> torch.Tensor:
        slots = self._unflatten(hidden_state)
        batch_size = slots.shape[0]
        slot_positions = torch.arange(self.num_memory_slots, device=slots.device, dtype=torch.long)
        slots = (
            slots
            + self.slot_index_embedding(slot_positions).unsqueeze(0)
            + self.slot_bank_embedding(self._slot_bank_id_tensor(slots.device)).unsqueeze(0)
        )
        latent_type = self.latent_type_embeddings(
            torch.full((batch_size,), int(latent_type_id), dtype=torch.long, device=slots.device)
        ).unsqueeze(1)
        slots = slots + latent_type
        slot_mask = self._slot_mask(batch_size, slots.device)
        for block in self.slot_blocks:
            slots = _run_encoder_block(self.activation_checkpointing, block, slots, mask=slot_mask)
        return slots

    def _bank_states(self, slots: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        batch_size = slots.shape[0]
        slot_bank_ids = self._slot_bank_id_tensor(slots.device)
        bank_states = []
        bank_mask_rows = []
        for bank_index in range(self.num_slot_banks):
            bank_slot_mask = slot_bank_ids.eq(bank_index).unsqueeze(0).expand(batch_size, -1)
            bank_states.append(self.bank_pool(slots, mask=bank_slot_mask))
            bank_mask_rows.append(bool(bank_slot_mask[0].any().item()))
        bank_states_tensor = torch.stack(bank_states, dim=1)
        bank_mask = torch.as_tensor(bank_mask_rows, dtype=torch.bool, device=slots.device).unsqueeze(0).expand(batch_size, -1)
        bank_states_tensor = _run_encoder_block(
            self.activation_checkpointing,
            self.bank_state_refine,
            bank_states_tensor,
            mask=bank_mask,
        )
        return bank_states_tensor, bank_mask

    @staticmethod
    def _routing_weights(
        *,
        domain_weights: torch.Tensor,
        decision_domain: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if decision_domain is None or decision_domain.dim() != 2 or decision_domain.shape[1] < NUM_DOMAINS:
            return domain_weights
        routing = decision_domain[:, :NUM_DOMAINS].float()
        routing_sum = routing.sum(dim=-1, keepdim=True)
        normalized = routing / routing_sum.clamp(min=1.0)
        valid = routing_sum > 0
        return torch.where(valid, normalized, domain_weights)

    def _planner_chunk_size(self, action_count: int) -> int:
        action_count = max(int(action_count), 0)
        if action_count <= 0:
            return 1
        chunk = max(int(getattr(self, "planner_action_chunk_size", 0) or 0), 0)
        if chunk <= 0 or action_count <= chunk:
            return action_count
        return max(chunk, 1)

    def _domain_states(
        self,
        slots: torch.Tensor,
        *,
        bank_states: torch.Tensor | None = None,
        bank_mask: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        batch_size = slots.shape[0]
        slot_mask = self._slot_mask(batch_size, slots.device)
        if bank_states is None or bank_mask is None:
            bank_states, bank_mask = self._bank_states(slots)
        domain_queries = self.domain_seed.expand(batch_size, -1, -1)
        domain_mask = torch.ones((batch_size, NUM_DOMAINS), dtype=torch.bool, device=slots.device)
        domain_states = _run_cross_block(
            self.activation_checkpointing,
            self.domain_cross,
            domain_queries,
            bank_states,
            query_mask=domain_mask,
            memory_mask=bank_mask,
        )
        domain_states = _run_encoder_block(
            self.activation_checkpointing,
            self.domain_refine,
            domain_states,
            mask=domain_mask,
        )
        pooled = self.slot_pool(slots, mask=slot_mask)
        pooled = pooled + self.bank_pool(bank_states, mask=bank_mask)
        domain_logits = self.domain_gate(pooled)
        return pooled, domain_states, domain_logits

    def _candidate_tokens(
        self,
        action_embeddings: torch.Tensor,
        slots: torch.Tensor,
        bank_states: torch.Tensor,
        bank_mask: torch.Tensor,
    ) -> torch.Tensor:
        batch_size = slots.shape[0]
        slot_mask = self._slot_mask(batch_size, slots.device)
        action_mask = action_embeddings.abs().sum(dim=-1) > 0
        candidate_x = self.candidate_in_proj(action_embeddings)
        candidate_x = _run_cross_block(
            self.activation_checkpointing,
            self.candidate_cross,
            candidate_x,
            slots,
            query_mask=action_mask,
            memory_mask=slot_mask,
        )
        candidate_x = _run_cross_block(
            self.activation_checkpointing,
            self.candidate_bank_cross,
            candidate_x,
            bank_states,
            query_mask=action_mask,
            memory_mask=bank_mask,
        )
        candidate_x = _run_encoder_block(
            self.activation_checkpointing,
            self.candidate_refine,
            candidate_x,
            mask=action_mask,
        )
        return candidate_x * action_mask.unsqueeze(-1).float()

    def _candidate_tokens_chunked(
        self,
        action_embeddings: torch.Tensor,
        slots: torch.Tensor,
        bank_states: torch.Tensor,
        bank_mask: torch.Tensor,
    ) -> torch.Tensor:
        """Build candidate tokens with optional action-axis chunking.

        The slot/bank cross-attention stages are independent for each action,
        so chunking them preserves semantics while avoiding large ``B*A``
        activation blocks.  The final candidate self-attention still runs over
        the full action set to keep candidate-set ranking interactions intact.
        """

        batch_size, action_count, _ = action_embeddings.shape
        if action_count <= 0:
            return action_embeddings.new_zeros((batch_size, 0, self.d_model))

        chunk_size = self._planner_chunk_size(action_count)
        if action_count <= chunk_size:
            return _run_function_with_checkpoint(
                self.activation_checkpointing,
                lambda action_embeddings_, slots_, bank_states_, bank_mask_: self._candidate_tokens(
                    action_embeddings_,
                    slots_,
                    bank_states_,
                    bank_mask_,
                ),
                action_embeddings,
                slots,
                bank_states,
                bank_mask,
            )

        slot_mask = self._slot_mask(batch_size, slots.device)
        action_mask = action_embeddings.abs().sum(dim=-1) > 0
        candidate_chunks: list[torch.Tensor] = []
        for start in range(0, action_count, chunk_size):
            end = min(start + chunk_size, action_count)
            action_slice = action_embeddings[:, start:end, :]
            mask_slice = action_mask[:, start:end]
            candidate_x = self.candidate_in_proj(action_slice)
            candidate_x = _run_cross_block(
                self.activation_checkpointing,
                self.candidate_cross,
                candidate_x,
                slots,
                query_mask=mask_slice,
                memory_mask=slot_mask,
            )
            candidate_x = _run_cross_block(
                self.activation_checkpointing,
                self.candidate_bank_cross,
                candidate_x,
                bank_states,
                query_mask=mask_slice,
                memory_mask=bank_mask,
            )
            candidate_chunks.append(candidate_x)

        candidate_x = torch.cat(candidate_chunks, dim=1)
        candidate_x = _run_encoder_block(
            self.activation_checkpointing,
            self.candidate_refine,
            candidate_x,
            mask=action_mask,
        )
        return candidate_x * action_mask.unsqueeze(-1).float()

    def _score_candidates(
        self,
        candidate_tokens: torch.Tensor,
        routing_weights: torch.Tensor,
        domain_states: torch.Tensor,
    ) -> torch.Tensor:
        domain_logits = []
        for domain_idx, head in enumerate(self.candidate_score_heads):
            conditioned = candidate_tokens + domain_states[:, domain_idx : domain_idx + 1, :]
            domain_logits.append(head(conditioned).squeeze(-1))
        stacked = torch.stack(domain_logits, dim=1)
        return (stacked * routing_weights.unsqueeze(-1)).sum(dim=1)

    def _planner_features(
        self,
        candidate_tokens: torch.Tensor,
        slots: torch.Tensor,
        bank_states: torch.Tensor,
        bank_mask: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        batch_size, action_count, _ = candidate_tokens.shape
        action_mask = candidate_tokens.abs().sum(dim=-1) > 0
        if not action_mask.any():
            return candidate_tokens.new_zeros((batch_size, action_count, self.d_model)), action_mask

        slot_mask = self._slot_mask(batch_size, slots.device)
        planner_queries = (
            self.planner_bank_seed.view(1, 1, self.num_slot_banks, self.d_model)
            + candidate_tokens.unsqueeze(2)
            + bank_states.unsqueeze(1)
        )
        flat_queries = planner_queries.reshape(batch_size * action_count, self.num_slot_banks, self.d_model)
        flat_query_mask = action_mask.reshape(batch_size * action_count, 1).expand(-1, self.num_slot_banks)
        flat_slots = slots.unsqueeze(1).expand(batch_size, action_count, self.num_memory_slots, self.d_model).reshape(
            batch_size * action_count,
            self.num_memory_slots,
            self.d_model,
        )
        flat_slot_mask = slot_mask.unsqueeze(1).expand(batch_size, action_count, self.num_memory_slots).reshape(
            batch_size * action_count,
            self.num_memory_slots,
        )
        flat_bank_states = bank_states.unsqueeze(1).expand(batch_size, action_count, self.num_slot_banks, self.d_model).reshape(
            batch_size * action_count,
            self.num_slot_banks,
            self.d_model,
        )
        flat_bank_mask = bank_mask.unsqueeze(1).expand(batch_size, action_count, self.num_slot_banks).reshape(
            batch_size * action_count,
            self.num_slot_banks,
        )
        planner_states = _run_cross_block(
            self.activation_checkpointing,
            self.planner_slot_cross,
            flat_queries,
            flat_slots,
            query_mask=flat_query_mask,
            memory_mask=flat_slot_mask,
        )
        planner_states = _run_cross_block(
            self.activation_checkpointing,
            self.planner_bank_cross,
            planner_states,
            flat_bank_states,
            query_mask=flat_query_mask,
            memory_mask=flat_bank_mask,
        )
        planner_states = _run_encoder_block(
            self.activation_checkpointing,
            self.planner_refine,
            planner_states,
            mask=flat_query_mask,
        )
        planner_tokens = self.planner_pool(planner_states, mask=flat_query_mask).reshape(batch_size, action_count, self.d_model)
        return planner_tokens * action_mask.unsqueeze(-1).float(), action_mask

    def _planner_policy_logits(
        self,
        planner_tokens: torch.Tensor,
        candidate_tokens: torch.Tensor,
        routing_weights: torch.Tensor,
        domain_states: torch.Tensor,
    ) -> torch.Tensor:
        batch_size, action_count, _ = candidate_tokens.shape
        action_mask = candidate_tokens.abs().sum(dim=-1) > 0

        domain_logits = []
        for domain_idx, head in enumerate(self.planner_score_heads):
            conditioned = planner_tokens + candidate_tokens + domain_states[:, domain_idx : domain_idx + 1, :]
            domain_logits.append(head(conditioned).squeeze(-1))
        stacked = torch.stack(domain_logits, dim=1)
        planner_logits = (stacked * routing_weights.unsqueeze(-1)).sum(dim=1)
        return planner_logits * action_mask.float()

    def _planner_q_outputs(
        self,
        planner_tokens: torch.Tensor,
        candidate_tokens: torch.Tensor,
        routing_weights: torch.Tensor,
        domain_states: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        batch_size, action_count, _ = candidate_tokens.shape
        action_mask = candidate_tokens.abs().sum(dim=-1) > 0
        domain_q_logits = []
        for domain_idx, head in enumerate(self.planner_q_heads):
            conditioned = planner_tokens + candidate_tokens + domain_states[:, domain_idx : domain_idx + 1, :]
            domain_q_logits.append(head(conditioned))
        stacked = torch.stack(domain_q_logits, dim=1)
        mixed_q_logits = (
            stacked * routing_weights.unsqueeze(-1).unsqueeze(-1)
        ).sum(dim=1)
        flat_q = support_to_scalar(mixed_q_logits.reshape(batch_size * action_count, self.num_bins), self.support_size)
        planner_q = flat_q.reshape(batch_size, action_count)
        planner_q = planner_q * action_mask.float()
        return mixed_q_logits, planner_q

    def _planner_objective_q_outputs(
        self,
        planner_tokens: torch.Tensor,
        candidate_tokens: torch.Tensor,
        routing_weights: torch.Tensor,
        domain_states: torch.Tensor,
        *,
        objective_context: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        batch_size, action_count, _ = candidate_tokens.shape
        action_mask = candidate_tokens.abs().sum(dim=-1) > 0
        domain_component_logits = []
        for domain_idx, head in enumerate(self.planner_q_component_heads):
            conditioned = planner_tokens + candidate_tokens + domain_states[:, domain_idx : domain_idx + 1, :]
            domain_component_logits.append(
                head(conditioned).view(batch_size, action_count, NUM_OBJECTIVE_HEADS, self.num_bins)
            )
        stacked = torch.stack(domain_component_logits, dim=1)
        mixed_component_logits = (
            stacked * routing_weights.unsqueeze(-1).unsqueeze(-1).unsqueeze(-1)
        ).sum(dim=1)
        planner_q_components = support_tensor_to_scalar(
            mixed_component_logits.reshape(batch_size * action_count, NUM_OBJECTIVE_HEADS, self.num_bins),
            self.support_size,
        ).reshape(batch_size, action_count, NUM_OBJECTIVE_HEADS)
        planner_objective_q = scalarize_objective_components_torch(
            planner_q_components,
            objective_context,
        )
        planner_q_components = planner_q_components * action_mask.unsqueeze(-1).float()
        planner_objective_q = planner_objective_q * action_mask.float()
        return mixed_component_logits, planner_q_components, planner_objective_q

    def _planner_outputs_chunked(
        self,
        candidate_tokens: torch.Tensor,
        slots: torch.Tensor,
        bank_states: torch.Tensor,
        bank_mask: torch.Tensor,
        routing_weights: torch.Tensor,
        domain_states: torch.Tensor,
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
    ]:
        """Run action-conditioned planner heads with optional action chunking.

        ``_planner_features`` expands slot/bank tensors across the action axis
        before flattening to ``B * A`` rows.  That expansion is one of the
        peak-memory hotspots for pass-large configs.  The planner feature and
        Q/policy heads are action-local after ``candidate_tokens`` are formed,
        so slicing the action axis preserves action order and tensor semantics
        while bounding the largest intermediate to ``B * chunk_size`` rows.
        """

        batch_size, action_count, _ = candidate_tokens.shape
        if action_count <= 0:
            action_mask = torch.zeros((batch_size, 0), dtype=torch.bool, device=candidate_tokens.device)
            q_logits = candidate_tokens.new_zeros((batch_size, 0, self.num_bins))
            q = candidate_tokens.new_zeros((batch_size, 0))
            component_logits = candidate_tokens.new_zeros((batch_size, 0, NUM_OBJECTIVE_HEADS, self.num_bins))
            components = candidate_tokens.new_zeros((batch_size, 0, NUM_OBJECTIVE_HEADS))
            return (
                action_mask,
                candidate_tokens.new_zeros((batch_size, 0)),
                q_logits,
                q,
                component_logits,
                components,
                candidate_tokens.new_zeros((batch_size, 0)),
                candidate_tokens.new_zeros((batch_size, 0, self.d_model)),
            )

        chunk_size = self._planner_chunk_size(action_count)

        def _planner_feature_fn(candidate_tokens_: torch.Tensor, slots_: torch.Tensor, bank_states_: torch.Tensor, bank_mask_: torch.Tensor):
            return self._planner_features(
                candidate_tokens_,
                slots_,
                bank_states_,
                bank_mask_,
            )

        def _planner_policy_fn(
            planner_tokens_: torch.Tensor,
            candidate_tokens_: torch.Tensor,
            routing_weights_: torch.Tensor,
            domain_states_: torch.Tensor,
        ):
            return self._planner_policy_logits(
                planner_tokens_,
                candidate_tokens_,
                routing_weights_,
                domain_states_,
            )

        def _planner_q_fn(
            planner_tokens_: torch.Tensor,
            candidate_tokens_: torch.Tensor,
            routing_weights_: torch.Tensor,
            domain_states_: torch.Tensor,
        ):
            return self._planner_q_outputs(
                planner_tokens_,
                candidate_tokens_,
                routing_weights_,
                domain_states_,
            )

        def _planner_objective_q_fn(
            planner_tokens_: torch.Tensor,
            candidate_tokens_: torch.Tensor,
            routing_weights_: torch.Tensor,
            domain_states_: torch.Tensor,
        ):
            return self._planner_objective_q_outputs(
                planner_tokens_,
                candidate_tokens_,
                routing_weights_,
                domain_states_,
                objective_context=objective_context,
            )

        if action_count <= chunk_size:
            planner_tokens, planner_action_mask = _run_function_with_checkpoint(
                self.activation_checkpointing,
                _planner_feature_fn,
                candidate_tokens,
                slots,
                bank_states,
                bank_mask,
            )
            planner_policy_logits = _run_function_with_checkpoint(
                self.activation_checkpointing,
                _planner_policy_fn,
                planner_tokens,
                candidate_tokens,
                routing_weights,
                domain_states,
            )
            planner_q_logits, planner_q = _run_function_with_checkpoint(
                self.activation_checkpointing,
                _planner_q_fn,
                planner_tokens,
                candidate_tokens,
                routing_weights,
                domain_states,
            )
            (
                planner_q_component_logits,
                planner_q_components,
                planner_objective_q,
            ) = _run_function_with_checkpoint(
                self.activation_checkpointing,
                _planner_objective_q_fn,
                planner_tokens,
                candidate_tokens,
                routing_weights,
                domain_states,
            )
            return (
                planner_action_mask,
                planner_policy_logits,
                planner_q_logits,
                planner_q,
                planner_q_component_logits,
                planner_q_components,
                planner_objective_q,
                planner_tokens,
            )

        planner_action_mask_chunks: list[torch.Tensor] = []
        planner_policy_chunks: list[torch.Tensor] = []
        planner_q_logits_chunks: list[torch.Tensor] = []
        planner_q_chunks: list[torch.Tensor] = []
        planner_q_component_logits_chunks: list[torch.Tensor] = []
        planner_q_components_chunks: list[torch.Tensor] = []
        planner_objective_q_chunks: list[torch.Tensor] = []
        planner_token_chunks: list[torch.Tensor] = []

        for start in range(0, action_count, chunk_size):
            end = min(start + chunk_size, action_count)
            candidate_slice = candidate_tokens[:, start:end, :]
            planner_tokens, planner_action_mask = _run_function_with_checkpoint(
                self.activation_checkpointing,
                _planner_feature_fn,
                candidate_slice,
                slots,
                bank_states,
                bank_mask,
            )
            planner_policy_logits = _run_function_with_checkpoint(
                self.activation_checkpointing,
                _planner_policy_fn,
                planner_tokens,
                candidate_slice,
                routing_weights,
                domain_states,
            )
            planner_q_logits, planner_q = _run_function_with_checkpoint(
                self.activation_checkpointing,
                _planner_q_fn,
                planner_tokens,
                candidate_slice,
                routing_weights,
                domain_states,
            )
            (
                planner_q_component_logits,
                planner_q_components,
                planner_objective_q,
            ) = _run_function_with_checkpoint(
                self.activation_checkpointing,
                _planner_objective_q_fn,
                planner_tokens,
                candidate_slice,
                routing_weights,
                domain_states,
            )
            planner_action_mask_chunks.append(planner_action_mask)
            planner_policy_chunks.append(planner_policy_logits)
            planner_q_logits_chunks.append(planner_q_logits)
            planner_q_chunks.append(planner_q)
            planner_q_component_logits_chunks.append(planner_q_component_logits)
            planner_q_components_chunks.append(planner_q_components)
            planner_objective_q_chunks.append(planner_objective_q)
            planner_token_chunks.append(planner_tokens)

        return (
            torch.cat(planner_action_mask_chunks, dim=1),
            torch.cat(planner_policy_chunks, dim=1),
            torch.cat(planner_q_logits_chunks, dim=1),
            torch.cat(planner_q_chunks, dim=1),
            torch.cat(planner_q_component_logits_chunks, dim=1),
            torch.cat(planner_q_components_chunks, dim=1),
            torch.cat(planner_objective_q_chunks, dim=1),
            torch.cat(planner_token_chunks, dim=1),
        )

    @staticmethod
    def _normalize_action_values(
        values: torch.Tensor,
        action_mask: torch.Tensor,
    ) -> torch.Tensor:
        masked_values = torch.where(action_mask, values, torch.zeros_like(values))
        valid_count = action_mask.float().sum(dim=-1, keepdim=True).clamp(min=1.0)
        mean = masked_values.sum(dim=-1, keepdim=True) / valid_count
        centered = torch.where(action_mask, values - mean, torch.zeros_like(values))
        variance = centered.pow(2).sum(dim=-1, keepdim=True) / valid_count
        std = variance.clamp(min=1e-6).sqrt()
        normalized = centered / std
        return torch.where(action_mask, normalized, torch.zeros_like(normalized))

    def _latent_policy_logits(
        self,
        slots: torch.Tensor,
        routing_weights: torch.Tensor,
        domain_states: torch.Tensor,
        bank_states: torch.Tensor,
        bank_mask: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        batch_size = slots.shape[0]
        slot_mask = self._slot_mask(batch_size, slots.device)
        action_mask = torch.ones((batch_size, MAX_ACTIONS), dtype=torch.bool, device=slots.device)
        domain_logits = []
        domain_embeddings = []
        for domain_idx in range(NUM_DOMAINS):
            latent_tokens = self.latent_bank_queries[domain_idx].unsqueeze(0).expand(batch_size, -1, -1)
            latent_tokens = latent_tokens + domain_states[:, domain_idx : domain_idx + 1, :]
            latent_tokens = _run_cross_block(
                self.activation_checkpointing,
                self.latent_bank_cross[domain_idx],
                latent_tokens,
                slots,
                query_mask=action_mask,
                memory_mask=slot_mask,
            )
            latent_tokens = _run_cross_block(
                self.activation_checkpointing,
                self.latent_bank_bank_cross[domain_idx],
                latent_tokens,
                bank_states,
                query_mask=action_mask,
                memory_mask=bank_mask,
            )
            latent_tokens = _run_encoder_block(
                self.activation_checkpointing,
                self.latent_bank_refine[domain_idx],
                latent_tokens,
                mask=action_mask,
            )
            domain_logits.append(self.latent_policy_score_heads[domain_idx](latent_tokens).squeeze(-1))
            domain_embeddings.append(self.latent_action_proj(latent_tokens))
        logits = torch.stack(domain_logits, dim=1)
        embeddings = torch.stack(domain_embeddings, dim=1)
        mixed_logits = (logits * routing_weights.unsqueeze(-1)).sum(dim=1)
        mixed_embeddings = (embeddings * routing_weights.unsqueeze(-1).unsqueeze(-1)).sum(dim=1)
        return mixed_logits, mixed_embeddings

    def _value_outputs(
        self,
        slots: torch.Tensor,
        *,
        routing_weights: torch.Tensor,
        objective_context: torch.Tensor | None = None,
        domain_states: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        if domain_states is None:
            _pooled, domain_states, _domain_logits = self._domain_states(slots)
        value_stack = torch.stack([head(domain_states[:, idx, :]) for idx, head in enumerate(self.value_heads)], dim=1)
        component_stack = torch.stack(
            [
                head(domain_states[:, idx, :]).view(slots.shape[0], NUM_OBJECTIVE_HEADS, self.num_bins)
                for idx, head in enumerate(self.value_component_heads)
            ],
            dim=1,
        )
        value_logits = (value_stack * routing_weights.unsqueeze(-1)).sum(dim=1)
        value_component_logits = (
            component_stack * routing_weights.unsqueeze(-1).unsqueeze(-1)
        ).sum(dim=1)
        value = support_to_scalar(value_logits, self.support_size)
        value_components = support_tensor_to_scalar(value_component_logits, self.support_size)
        objective_value = scalarize_objective_components_torch(value_components, objective_context)
        return value_logits, value, value_component_logits, value_components, objective_value

    def value_only(
        self,
        hidden_state: torch.Tensor,
        *,
        decision_domain: torch.Tensor | None = None,
        objective_context: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Cheap value-only path for explicit action-conditioned rollout planning.

        This skips candidate scoring / planner-policy heads and only evaluates the
        latent state through the shared domain/value heads.
        """
        slots = self._slot_states(hidden_state, latent_type_id=0)
        bank_states, bank_mask = self._bank_states(slots)
        def _domain_features(slots_: torch.Tensor, bank_states_: torch.Tensor, bank_mask_: torch.Tensor):
            return self._domain_states(
                slots_,
                bank_states=bank_states_,
                bank_mask=bank_mask_,
            )

        _pooled, domain_states, domain_logits = _run_function_with_checkpoint(
            self.activation_checkpointing,
            _domain_features,
            slots,
            bank_states,
            bank_mask,
        )
        domain_weights = torch.softmax(domain_logits, dim=-1)
        routing_weights = self._routing_weights(
            domain_weights=domain_weights,
            decision_domain=decision_domain,
        )
        def _value_head(slots_: torch.Tensor, routing_weights_: torch.Tensor, domain_states_: torch.Tensor):
            return self._value_outputs(
                slots_,
                routing_weights=routing_weights_,
                objective_context=objective_context,
                domain_states=domain_states_,
            )

        return _run_function_with_checkpoint(
            self.activation_checkpointing,
            _value_head,
            slots,
            routing_weights,
            domain_states,
        )

    def latent_policy_only(
        self,
        hidden_state: torch.Tensor,
        *,
        decision_domain: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Cheap latent-action prior path for multi-step search-free rollout."""
        slots = self._slot_states(hidden_state, latent_type_id=0)
        bank_states, bank_mask = self._bank_states(slots)
        def _domain_features(slots_: torch.Tensor, bank_states_: torch.Tensor, bank_mask_: torch.Tensor):
            return self._domain_states(
                slots_,
                bank_states=bank_states_,
                bank_mask=bank_mask_,
            )

        _pooled, domain_states, domain_logits = _run_function_with_checkpoint(
            self.activation_checkpointing,
            _domain_features,
            slots,
            bank_states,
            bank_mask,
        )
        domain_weights = torch.softmax(domain_logits, dim=-1)
        routing_weights = self._routing_weights(
            domain_weights=domain_weights,
            decision_domain=decision_domain,
        )
        def _latent_policy(
            slots_: torch.Tensor,
            routing_weights_: torch.Tensor,
            domain_states_: torch.Tensor,
            bank_states_: torch.Tensor,
            bank_mask_: torch.Tensor,
        ):
            return self._latent_policy_logits(
                slots_,
                routing_weights_,
                domain_states_,
                bank_states_,
                bank_mask_,
            )

        return _run_function_with_checkpoint(
            self.activation_checkpointing,
            _latent_policy,
            slots,
            routing_weights,
            domain_states,
            bank_states,
            bank_mask,
        )

    def forward(
        self,
        hidden_state: torch.Tensor,
        *,
        action_embeddings: torch.Tensor | None = None,
        decision_domain: torch.Tensor | None = None,
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
    ]:
        slots = self._slot_states(hidden_state, latent_type_id=0)
        bank_states, bank_mask = self._bank_states(slots)
        def _domain_features(slots_: torch.Tensor, bank_states_: torch.Tensor, bank_mask_: torch.Tensor):
            return self._domain_states(
                slots_,
                bank_states=bank_states_,
                bank_mask=bank_mask_,
            )

        pooled, domain_states, domain_logits = _run_function_with_checkpoint(
            self.activation_checkpointing,
            _domain_features,
            slots,
            bank_states,
            bank_mask,
        )
        domain_weights = torch.softmax(domain_logits, dim=-1)
        routing_weights = self._routing_weights(domain_weights=domain_weights, decision_domain=decision_domain)

        def _latent_policy(
            slots_: torch.Tensor,
            routing_weights_: torch.Tensor,
            domain_states_: torch.Tensor,
            bank_states_: torch.Tensor,
            bank_mask_: torch.Tensor,
        ):
            return self._latent_policy_logits(
                slots_,
                routing_weights_,
                domain_states_,
                bank_states_,
                bank_mask_,
            )

        latent_policy_logits, latent_action_embeddings = _run_function_with_checkpoint(
            self.activation_checkpointing,
            _latent_policy,
            slots,
            routing_weights,
            domain_states,
            bank_states,
            bank_mask,
        )
        policy_action_embeddings = action_embeddings if action_embeddings is not None else latent_action_embeddings
        candidate_tokens = self._candidate_tokens_chunked(
            policy_action_embeddings,
            slots,
            bank_states,
            bank_mask,
        )
        base_policy_logits = _run_function_with_checkpoint(
            self.activation_checkpointing,
            lambda candidate_tokens_, routing_weights_, domain_states_: self._score_candidates(
                candidate_tokens_,
                routing_weights_,
                domain_states_,
            ),
            candidate_tokens,
            routing_weights,
            domain_states,
        )
        (
            planner_action_mask,
            planner_policy_logits,
            planner_q_logits,
            planner_q,
            planner_q_component_logits,
            planner_q_components,
            planner_objective_q,
            planner_tokens,
        ) = self._planner_outputs_chunked(
            candidate_tokens,
            slots,
            bank_states,
            bank_mask,
            routing_weights,
            domain_states,
            objective_context=objective_context,
        )
        planner_q_bias = self._normalize_action_values(planner_q, planner_action_mask)
        planner_objective_q_bias = self._normalize_action_values(planner_objective_q, planner_action_mask)
        planner_risk_q = planner_q_components[..., list(RISK_OBJECTIVE_HEAD_INDICES)].mean(dim=-1)
        planner_risk_bias = self._normalize_action_values(planner_risk_q, planner_action_mask)
        policy_logits = (
            base_policy_logits
            + self.internal_planner_blend * planner_policy_logits
            + self.internal_planner_q_blend * planner_q_bias
            + self.internal_planner_objective_q_blend * planner_objective_q_bias
            + self.internal_planner_risk_blend * planner_risk_bias
        )

        del pooled
        value_logits, value, value_component_logits, value_components, objective_value = _run_function_with_checkpoint(
            self.activation_checkpointing,
            lambda slots_, routing_weights_, domain_states_: self._value_outputs(
                slots_,
                routing_weights=routing_weights_,
                objective_context=objective_context,
                domain_states=domain_states_,
            ),
            slots,
            routing_weights,
            domain_states,
        )
        return (
            policy_logits,
            value_logits,
            value,
            latent_policy_logits,
            latent_action_embeddings,
            value_component_logits,
            value_components,
            objective_value,
            planner_q_logits,
            planner_q,
            planner_objective_q,
            planner_q_component_logits,
            planner_q_components,
        )

    def semantic_forward(
        self,
        hidden_state: torch.Tensor,
        *,
        decision_domain: torch.Tensor | None = None,
        objective_context: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        slots = self._slot_states(hidden_state, latent_type_id=1)
        bank_states, bank_mask = self._bank_states(slots)
        def _domain_features(slots_: torch.Tensor, bank_states_: torch.Tensor, bank_mask_: torch.Tensor):
            return self._domain_states(
                slots_,
                bank_states=bank_states_,
                bank_mask=bank_mask_,
            )

        pooled, _domain_states, domain_logits = _run_function_with_checkpoint(
            self.activation_checkpointing,
            _domain_features,
            slots,
            bank_states,
            bank_mask,
        )
        domain_weights = torch.softmax(domain_logits, dim=-1)
        routing_weights = self._routing_weights(domain_weights=domain_weights, decision_domain=decision_domain)
        semantic_policy_logits = self.semantic_policy_head(pooled)
        value_logits, value, value_component_logits, value_components, objective_value = _run_function_with_checkpoint(
            self.activation_checkpointing,
            lambda slots_, routing_weights_, domain_states_: self._value_outputs(
                slots_,
                routing_weights=routing_weights_,
                objective_context=objective_context,
                domain_states=domain_states_,
            ),
            slots,
            routing_weights,
            _domain_states,
        )
        return (
            semantic_policy_logits,
            value_logits,
            value,
            value_component_logits,
            value_components,
            objective_value,
        )
