"""Latent dynamics, projection, and transition-surface token networks."""

from __future__ import annotations

import torch
import torch.nn as nn

from sts2_env.attention_blocks import (
    CrossAttentionBlock,
    EntityPooling,
    TransformerEncoderBlock,
)
from sts2_env.objective_heads import NUM_OBJECTIVE_HEADS
from sts2_env.observation_v2 import MAX_ACTIONS, NUM_DOMAINS, NUM_PHASES

from ._token_memory_shared import (
    MEMORY_BANK_NAMES,
    MEMORY_SLOT_LAYOUT_LEGACY,
    _run_cross_block,
    _run_encoder_block,
    build_memory_slot_bank_ids,
    normalize_memory_slot_layout,
)
from .value_support import support_tensor_to_scalar, support_to_scalar


class TokenDynamicsNetwork(nn.Module):
    """Action-conditioned latent slot transition for token-memory MuZero."""

    def __init__(
        self,
        *,
        hidden_dim: int,
        action_embed_dim: int,
        d_model: int,
        num_memory_slots: int,
        memory_slot_layout: str = MEMORY_SLOT_LAYOUT_LEGACY,
        support_size: int = 25,
        num_transition_layers: int = 2,
        dropout: float = 0.0,
        activation_checkpointing: bool = False,
    ) -> None:
        super().__init__()
        self.hidden_dim = int(hidden_dim)
        self.action_embed_dim = int(action_embed_dim)
        self.d_model = int(d_model)
        self.num_memory_slots = int(num_memory_slots)
        self.memory_slot_layout = normalize_memory_slot_layout(memory_slot_layout)
        self.support_size = int(support_size)
        self.num_bins = 2 * self.support_size + 1
        self.activation_checkpointing = bool(activation_checkpointing)

        if self.hidden_dim != self.d_model * self.num_memory_slots:
            raise ValueError(
                f"TokenDynamicsNetwork expects hidden_dim == d_model * num_memory_slots, "
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
        self.transition_type_embedding = nn.Parameter(torch.randn(1, 1, self.d_model) * 0.02)
        self.action_token_proj = nn.Sequential(
            nn.Linear(self.action_embed_dim, self.d_model),
            nn.GELU(),
            nn.Linear(self.d_model, self.d_model),
        )
        self.action_gate = nn.Sequential(
            nn.Linear(self.action_embed_dim, self.d_model),
            nn.Sigmoid(),
        )
        self.slot_action_gate = nn.Sequential(
            nn.LayerNorm(self.d_model + self.action_embed_dim),
            nn.Linear(self.d_model + self.action_embed_dim, self.d_model),
            nn.Sigmoid(),
        )
        self.slot_action_write = nn.Sequential(
            nn.LayerNorm(self.d_model + self.action_embed_dim),
            nn.Linear(self.d_model + self.action_embed_dim, 1),
            nn.Sigmoid(),
        )
        self.slot_action_cross = CrossAttentionBlock(self.d_model, max(1, min(4, self.d_model // 32)), self.d_model * 4, dropout=dropout)
        self.transition_blocks = nn.ModuleList(
            [
                TransformerEncoderBlock(
                    self.d_model,
                    max(1, min(4, self.d_model // 32)),
                    self.d_model * 4,
                    dropout=dropout,
                )
                for _ in range(max(int(num_transition_layers), 1))
            ]
        )
        self.slot_pool = EntityPooling(self.d_model, max(1, min(4, self.d_model // 32)), dropout=dropout)
        reward_hidden_dim = max(self.d_model * 2, self.action_embed_dim * 2)
        self.reward_net = nn.Sequential(
            nn.LayerNorm(self.d_model + self.action_embed_dim),
            nn.Linear(self.d_model + self.action_embed_dim, reward_hidden_dim),
            nn.GELU(),
            nn.Linear(reward_hidden_dim, self.num_bins),
        )
        self.reward_component_net = nn.Sequential(
            nn.LayerNorm(self.d_model + self.action_embed_dim),
            nn.Linear(self.d_model + self.action_embed_dim, reward_hidden_dim),
            nn.GELU(),
            nn.Linear(reward_hidden_dim, self.num_bins * NUM_OBJECTIVE_HEADS),
        )
        self.surprise_net = nn.Sequential(
            nn.LayerNorm(self.d_model + self.action_embed_dim),
            nn.Linear(self.d_model + self.action_embed_dim, reward_hidden_dim),
            nn.GELU(),
            nn.Linear(reward_hidden_dim, 1),
        )
        self.output_norm = nn.LayerNorm(self.d_model)

    def _unflatten(self, hidden_state: torch.Tensor) -> torch.Tensor:
        return hidden_state.reshape(hidden_state.shape[0], self.num_memory_slots, self.d_model)

    def _flatten(self, slots: torch.Tensor) -> torch.Tensor:
        return slots.reshape(slots.shape[0], -1)

    def _slot_bank_id_tensor(self, device: torch.device) -> torch.Tensor:
        return self._slot_bank_ids.to(device=device)

    def forward(self, hidden_state: torch.Tensor, action_embedding: torch.Tensor):
        slots = self._unflatten(hidden_state)
        batch_size = slots.shape[0]
        slot_positions = torch.arange(self.num_memory_slots, device=slots.device, dtype=torch.long)
        slot_bank_ids = self._slot_bank_id_tensor(slots.device)
        slots = (
            slots
            + self.slot_index_embedding(slot_positions).unsqueeze(0)
            + self.slot_bank_embedding(slot_bank_ids).unsqueeze(0)
            + self.transition_type_embedding
        )
        slot_mask = torch.ones((batch_size, self.num_memory_slots), dtype=torch.bool, device=slots.device)
        action_token = self.action_token_proj(action_embedding).unsqueeze(1)
        action_mask = torch.ones((batch_size, 1), dtype=torch.bool, device=slots.device)
        gate = self.action_gate(action_embedding).unsqueeze(1)
        slot_gate_input = torch.cat(
            [
                slots,
                action_embedding.unsqueeze(1).expand(-1, self.num_memory_slots, -1),
            ],
            dim=-1,
        )
        slot_gate = self.slot_action_gate(slot_gate_input)
        slot_write = self.slot_action_write(slot_gate_input)

        slots = slots + slot_write * slot_gate * (gate * action_token)
        slots = _run_cross_block(
            self.activation_checkpointing,
            self.slot_action_cross,
            slots,
            action_token,
            query_mask=slot_mask,
            memory_mask=action_mask,
        )
        for block in self.transition_blocks:
            slots = _run_encoder_block(self.activation_checkpointing, block, slots, mask=slot_mask)
        slots = self.output_norm(slots)

        pooled = self.slot_pool(slots, mask=slot_mask)
        reward_input = torch.cat([pooled, action_embedding], dim=-1)
        reward_logits = self.reward_net(reward_input)
        reward = support_to_scalar(reward_logits, self.support_size)
        reward_component_logits = self.reward_component_net(reward_input).view(
            reward_input.shape[0],
            NUM_OBJECTIVE_HEADS,
            self.num_bins,
        )
        reward_components = support_tensor_to_scalar(reward_component_logits, self.support_size)
        surprise_logits = self.surprise_net(reward_input).squeeze(-1)
        surprise = torch.nn.functional.softplus(surprise_logits)
        return (
            self._flatten(slots),
            reward_logits,
            reward,
            reward_component_logits,
            reward_components,
            surprise_logits,
            surprise,
        )


class TokenLatentProjector(nn.Module):
    """Token-aware latent projector that preserves the slot structure."""

    def __init__(
        self,
        *,
        hidden_dim: int,
        d_model: int,
        num_memory_slots: int,
        memory_slot_layout: str = MEMORY_SLOT_LAYOUT_LEGACY,
        n_heads: int = 4,
        ffn_dim: int = 512,
        num_layers: int = 2,
        latent_type_vocab: int = 4,
        dropout: float = 0.0,
        activation_checkpointing: bool = False,
    ) -> None:
        super().__init__()
        self.hidden_dim = int(hidden_dim)
        self.d_model = int(d_model)
        self.num_memory_slots = int(num_memory_slots)
        self.memory_slot_layout = normalize_memory_slot_layout(memory_slot_layout)
        self.n_heads = max(1, int(n_heads))
        self.ffn_dim = int(ffn_dim)
        self.activation_checkpointing = bool(activation_checkpointing)
        if self.hidden_dim != self.d_model * self.num_memory_slots:
            raise ValueError(
                f"TokenLatentProjector expects hidden_dim == d_model * num_memory_slots, "
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
        self.latent_type_embeddings = nn.Embedding(max(int(latent_type_vocab), 1), self.d_model)
        self.input_proj = nn.Sequential(
            nn.LayerNorm(self.d_model),
            nn.Linear(self.d_model, self.d_model),
            nn.GELU(),
            nn.Linear(self.d_model, self.d_model),
        )
        self.slot_blocks = nn.ModuleList(
            [
                TransformerEncoderBlock(self.d_model, self.n_heads, self.ffn_dim, dropout=dropout)
                for _ in range(max(int(num_layers), 1))
            ]
        )
        self.output_norm = nn.LayerNorm(self.d_model)

    def _unflatten(self, hidden_state: torch.Tensor) -> torch.Tensor:
        return hidden_state.reshape(hidden_state.shape[0], self.num_memory_slots, self.d_model)

    def _flatten(self, slots: torch.Tensor) -> torch.Tensor:
        return slots.reshape(slots.shape[0], -1)

    def _slot_mask(self, batch_size: int, device: torch.device) -> torch.Tensor:
        return torch.ones((batch_size, self.num_memory_slots), dtype=torch.bool, device=device)

    def _slot_bank_id_tensor(self, device: torch.device) -> torch.Tensor:
        return self._slot_bank_ids.to(device=device)

    def forward(self, hidden_state: torch.Tensor, *, latent_type_id: int = 0) -> torch.Tensor:
        slots = self._unflatten(hidden_state)
        batch_size = slots.shape[0]
        slot_positions = torch.arange(self.num_memory_slots, device=slots.device, dtype=torch.long)
        slots = slots + self.input_proj(slots)
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
        return self._flatten(self.output_norm(slots))


class TokenTransitionSurfaceHead(nn.Module):
    """Slot-aware auxiliary head for next legal surface prediction."""

    def __init__(
        self,
        *,
        hidden_dim: int,
        d_model: int,
        num_memory_slots: int,
        memory_slot_layout: str = MEMORY_SLOT_LAYOUT_LEGACY,
        n_heads: int = 4,
        ffn_dim: int = 512,
        dropout: float = 0.0,
        activation_checkpointing: bool = False,
        planner_action_chunk_size: int = 0,
    ) -> None:
        super().__init__()
        self.hidden_dim = int(hidden_dim)
        self.d_model = int(d_model)
        self.num_memory_slots = int(num_memory_slots)
        self.memory_slot_layout = normalize_memory_slot_layout(memory_slot_layout)
        self.n_heads = max(1, int(n_heads))
        self.ffn_dim = int(ffn_dim)
        self.activation_checkpointing = bool(activation_checkpointing)
        self.planner_action_chunk_size = max(int(planner_action_chunk_size or 0), 0)
        if self.hidden_dim != self.d_model * self.num_memory_slots:
            raise ValueError(
                f"TokenTransitionSurfaceHead expects hidden_dim == d_model * num_memory_slots, "
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
        self.surface_type_embedding = nn.Parameter(torch.randn(1, 1, self.d_model) * 0.02)
        self.slot_blocks = nn.ModuleList(
            [TransformerEncoderBlock(self.d_model, self.n_heads, self.ffn_dim, dropout=dropout) for _ in range(2)]
        )
        self.slot_pool = EntityPooling(self.d_model, self.n_heads, dropout=dropout)

        self.action_queries = nn.Parameter(torch.randn(1, MAX_ACTIONS, self.d_model) * 0.02)
        self.action_cross = CrossAttentionBlock(self.d_model, self.n_heads, self.ffn_dim, dropout=dropout)
        self.action_refine = TransformerEncoderBlock(self.d_model, self.n_heads, self.ffn_dim, dropout=dropout)
        self.action_mask_head = nn.Sequential(
            nn.LayerNorm(self.d_model),
            nn.Linear(self.d_model, self.d_model),
            nn.GELU(),
            nn.Linear(self.d_model, 1),
        )
        self.decision_domain_head = nn.Sequential(
            nn.LayerNorm(self.d_model),
            nn.Linear(self.d_model, self.d_model),
            nn.GELU(),
            nn.Linear(self.d_model, NUM_DOMAINS),
        )
        self.phase_head = nn.Sequential(
            nn.LayerNorm(self.d_model),
            nn.Linear(self.d_model, self.d_model),
            nn.GELU(),
            nn.Linear(self.d_model, NUM_PHASES),
        )

        nn.init.constant_(self.action_mask_head[-1].bias, -2.0)

    def _unflatten(self, hidden_state: torch.Tensor) -> torch.Tensor:
        return hidden_state.reshape(hidden_state.shape[0], self.num_memory_slots, self.d_model)

    def _slot_mask(self, batch_size: int, device: torch.device) -> torch.Tensor:
        return torch.ones((batch_size, self.num_memory_slots), dtype=torch.bool, device=device)

    def _slot_bank_id_tensor(self, device: torch.device) -> torch.Tensor:
        return self._slot_bank_ids.to(device=device)

    def _slot_states(self, hidden_state: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        slots = self._unflatten(hidden_state)
        batch_size = slots.shape[0]
        slot_positions = torch.arange(self.num_memory_slots, device=slots.device, dtype=torch.long)
        slots = (
            slots
            + self.slot_index_embedding(slot_positions).unsqueeze(0)
            + self.slot_bank_embedding(self._slot_bank_id_tensor(slots.device)).unsqueeze(0)
            + self.surface_type_embedding
        )
        slot_mask = self._slot_mask(batch_size, slots.device)
        for block in self.slot_blocks:
            slots = _run_encoder_block(self.activation_checkpointing, block, slots, mask=slot_mask)
        return slots, slot_mask

    def _planner_chunk_size(self, action_count: int) -> int:
        action_count = max(int(action_count), 0)
        if action_count <= 0:
            return 1
        chunk = max(int(getattr(self, "planner_action_chunk_size", 0) or 0), 0)
        if chunk <= 0 or action_count <= chunk:
            return action_count
        return max(chunk, 1)

    def forward(self, hidden_state: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        slots, slot_mask = self._slot_states(hidden_state)
        batch_size = slots.shape[0]
        pooled = self.slot_pool(slots, mask=slot_mask)
        action_tokens = self.action_queries.expand(batch_size, -1, -1)
        action_mask = torch.ones((batch_size, MAX_ACTIONS), dtype=torch.bool, device=slots.device)
        chunk_size = self._planner_chunk_size(MAX_ACTIONS)
        if MAX_ACTIONS <= chunk_size:
            action_tokens = _run_cross_block(
                self.activation_checkpointing,
                self.action_cross,
                action_tokens,
                slots,
                query_mask=action_mask,
                memory_mask=slot_mask,
            )
        else:
            action_chunks = []
            for start in range(0, MAX_ACTIONS, chunk_size):
                end = min(start + chunk_size, MAX_ACTIONS)
                action_chunks.append(
                    _run_cross_block(
                        self.activation_checkpointing,
                        self.action_cross,
                        action_tokens[:, start:end, :],
                        slots,
                        query_mask=action_mask[:, start:end],
                        memory_mask=slot_mask,
                    )
                )
            action_tokens = torch.cat(action_chunks, dim=1)
        action_tokens = _run_encoder_block(
            self.activation_checkpointing,
            self.action_refine,
            action_tokens,
            mask=action_mask,
        )
        return (
            self.action_mask_head(action_tokens).squeeze(-1),
            self.decision_domain_head(pooled),
            self.phase_head(pooled),
        )
