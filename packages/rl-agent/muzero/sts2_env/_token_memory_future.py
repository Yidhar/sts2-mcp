"""Future world-bank reconstruction head for token-memory MuZero."""

from __future__ import annotations

import math

import torch
import torch.nn as nn

from sts2_env.attention_blocks import (
    CrossAttentionBlock,
    EntityPooling,
    TransformerEncoderBlock,
)
from sts2_env.observation_v3 import MAX_ZONE_ID, NUM_TOKEN_TYPES

from ._token_memory_shared import (
    MEMORY_BANK_NAMES,
    MEMORY_SLOT_LAYOUT_LEGACY,
    WORLD_BANK_NAMES,
    _run_cross_block,
    _run_encoder_block,
    build_memory_slot_bank_ids,
    build_zone_transport_prior,
    normalize_memory_slot_layout,
)


class TokenFutureWorldBankHead(nn.Module):
    """Predict future bank-level world states from bank-aware latent slots."""

    def __init__(
        self,
        *,
        hidden_dim: int,
        d_model: int,
        num_memory_slots: int,
        memory_slot_layout: str = MEMORY_SLOT_LAYOUT_LEGACY,
        bank_token_slots: int = 4,
        slot_source_same_bank_bias: float = 0.35,
        slot_source_same_slot_bias: float = 0.2,
        slot_source_type_match_scale: float = 0.5,
        slot_source_zone_transport_scale: float = 0.35,
        n_heads: int = 4,
        ffn_dim: int = 512,
        dropout: float = 0.0,
        activation_checkpointing: bool = False,
    ) -> None:
        super().__init__()
        self.hidden_dim = int(hidden_dim)
        self.d_model = int(d_model)
        self.num_memory_slots = int(num_memory_slots)
        self.memory_slot_layout = normalize_memory_slot_layout(memory_slot_layout)
        self.bank_token_slots = max(1, int(bank_token_slots))
        self.slot_source_same_bank_bias = float(slot_source_same_bank_bias)
        self.slot_source_same_slot_bias = float(slot_source_same_slot_bias)
        self.slot_source_type_match_scale = float(slot_source_type_match_scale)
        self.slot_source_zone_transport_scale = float(slot_source_zone_transport_scale)
        self.n_heads = max(1, int(n_heads))
        self.ffn_dim = int(ffn_dim)
        self.num_slot_banks = len(MEMORY_BANK_NAMES)
        self.activation_checkpointing = bool(activation_checkpointing)
        if self.hidden_dim != self.d_model * self.num_memory_slots:
            raise ValueError(
                f"TokenFutureWorldBankHead expects hidden_dim == d_model * num_memory_slots, "
                f"got {self.hidden_dim} vs {self.d_model} * {self.num_memory_slots}."
            )

        self.register_buffer(
            "_slot_bank_ids",
            torch.as_tensor(
                build_memory_slot_bank_ids(self.num_memory_slots, layout=self.memory_slot_layout),
                dtype=torch.long,
            ),
            persistent=False,
        )
        self.register_buffer(
            "_zone_transport_prior",
            build_zone_transport_prior(),
            persistent=False,
        )
        self.slot_index_embedding = nn.Embedding(self.num_memory_slots, self.d_model)
        self.slot_bank_embedding = nn.Embedding(self.num_slot_banks, self.d_model)
        self.current_type_embedding = nn.Parameter(torch.randn(1, 1, self.d_model) * 0.02)
        self.future_type_embedding = nn.Parameter(torch.randn(1, 1, self.d_model) * 0.02)
        self.slot_blocks = nn.ModuleList(
            [TransformerEncoderBlock(self.d_model, self.n_heads, self.ffn_dim, dropout=dropout) for _ in range(2)]
        )
        self.bank_pool = EntityPooling(self.d_model, self.n_heads, dropout=dropout)
        self.bank_refine = TransformerEncoderBlock(self.d_model, self.n_heads, self.ffn_dim, dropout=dropout)
        self.bank_out_proj = nn.Sequential(
            nn.LayerNorm(self.d_model),
            nn.Linear(self.d_model, self.d_model),
            nn.GELU(),
            nn.Linear(self.d_model, self.d_model),
        )
        self.bank_occupancy_head = nn.Sequential(
            nn.LayerNorm(self.d_model),
            nn.Linear(self.d_model, self.d_model),
            nn.GELU(),
            nn.Linear(self.d_model, 1),
        )
        self.bank_token_presence_head = nn.Sequential(
            nn.LayerNorm(self.d_model),
            nn.Linear(self.d_model, self.d_model),
            nn.GELU(),
            nn.Linear(self.d_model, NUM_TOKEN_TYPES),
        )
        self.bank_token_distribution_head = nn.Sequential(
            nn.LayerNorm(self.d_model),
            nn.Linear(self.d_model, self.d_model),
            nn.GELU(),
            nn.Linear(self.d_model, NUM_TOKEN_TYPES),
        )
        self.bank_token_slot_seed = nn.Parameter(
            torch.randn(1, len(WORLD_BANK_NAMES), self.bank_token_slots, self.d_model) * 0.02
        )
        self.bank_token_slot_embedding = nn.Embedding(self.bank_token_slots, self.d_model)
        self.bank_token_slot_cross = CrossAttentionBlock(self.d_model, self.n_heads, self.ffn_dim, dropout=dropout)
        self.bank_token_slot_copy_cross = CrossAttentionBlock(self.d_model, self.n_heads, self.ffn_dim, dropout=dropout)
        self.bank_token_slot_refine = TransformerEncoderBlock(self.d_model, self.n_heads, self.ffn_dim, dropout=dropout)
        self.bank_token_slot_copy_refine = TransformerEncoderBlock(self.d_model, self.n_heads, self.ffn_dim, dropout=dropout)
        self.bank_token_slot_out_proj = nn.Sequential(
            nn.LayerNorm(self.d_model),
            nn.Linear(self.d_model, self.d_model),
            nn.GELU(),
            nn.Linear(self.d_model, self.d_model),
        )
        self.bank_token_slot_mask_head = nn.Sequential(
            nn.LayerNorm(self.d_model),
            nn.Linear(self.d_model, self.d_model),
            nn.GELU(),
            nn.Linear(self.d_model, 1),
        )
        self.bank_token_slot_type_head = nn.Sequential(
            nn.LayerNorm(self.d_model),
            nn.Linear(self.d_model, self.d_model),
            nn.GELU(),
            nn.Linear(self.d_model, NUM_TOKEN_TYPES),
        )
        self.bank_token_slot_zone_head = nn.Sequential(
            nn.LayerNorm(self.d_model),
            nn.Linear(self.d_model, self.d_model),
            nn.GELU(),
            nn.Linear(self.d_model, MAX_ZONE_ID + 1),
        )
        self.bank_token_slot_source_q = nn.Sequential(
            nn.LayerNorm(self.d_model),
            nn.Linear(self.d_model, self.d_model),
        )
        self.bank_token_slot_source_k = nn.Sequential(
            nn.LayerNorm(self.d_model),
            nn.Linear(self.d_model, self.d_model),
        )
        self.bank_token_slot_new_head = nn.Sequential(
            nn.LayerNorm(self.d_model),
            nn.Linear(self.d_model, self.d_model),
            nn.GELU(),
            nn.Linear(self.d_model, 1),
        )

    def _unflatten(self, hidden_state: torch.Tensor) -> torch.Tensor:
        return hidden_state.reshape(hidden_state.shape[0], self.num_memory_slots, self.d_model)

    def _slot_mask(self, batch_size: int, device: torch.device) -> torch.Tensor:
        return torch.ones((batch_size, self.num_memory_slots), dtype=torch.bool, device=device)

    def _slot_bank_id_tensor(self, device: torch.device) -> torch.Tensor:
        return self._slot_bank_ids.to(device=device)

    def _slot_states(self, hidden_state: torch.Tensor, *, future_mode: bool) -> torch.Tensor:
        slots = self._unflatten(hidden_state)
        batch_size = slots.shape[0]
        slot_positions = torch.arange(self.num_memory_slots, device=slots.device, dtype=torch.long)
        type_embedding = self.future_type_embedding if future_mode else self.current_type_embedding
        slots = (
            slots
            + self.slot_index_embedding(slot_positions).unsqueeze(0)
            + self.slot_bank_embedding(self._slot_bank_id_tensor(slots.device)).unsqueeze(0)
            + type_embedding
        )
        slot_mask = self._slot_mask(batch_size, slots.device)
        for block in self.slot_blocks:
            slots = _run_encoder_block(self.activation_checkpointing, block, slots, mask=slot_mask)
        return slots

    def _pool_bank_states(self, slots: torch.Tensor) -> torch.Tensor:
        batch_size = slots.shape[0]
        slot_bank_ids = self._slot_bank_id_tensor(slots.device)
        bank_states = []
        bank_mask_rows = []
        for bank_index in range(len(WORLD_BANK_NAMES)):
            bank_slot_mask = slot_bank_ids.eq(bank_index).unsqueeze(0).expand(batch_size, -1)
            bank_states.append(self.bank_pool(slots, mask=bank_slot_mask))
            bank_mask_rows.append(bool(bank_slot_mask[0].any().item()))
        bank_states_tensor = torch.stack(bank_states, dim=1)
        bank_mask = torch.as_tensor(bank_mask_rows, dtype=torch.bool, device=slots.device).unsqueeze(0).expand(batch_size, -1)
        bank_states_tensor = _run_encoder_block(
            self.activation_checkpointing,
            self.bank_refine,
            bank_states_tensor,
            mask=bank_mask,
        )
        return bank_states_tensor

    def read_current_bank_states(self, hidden_state: torch.Tensor) -> torch.Tensor:
        slots = self._slot_states(hidden_state, future_mode=False)
        return self.bank_out_proj(self._pool_bank_states(slots))

    def _predict_bank_token_slots(
        self,
        bank_states_tensor: torch.Tensor,
        slots: torch.Tensor,
        *,
        future_mode: bool,
        current_bank_token_slot_states: torch.Tensor | None = None,
        current_bank_token_slot_mask: torch.Tensor | None = None,
        current_bank_token_slot_type_ids: torch.Tensor | None = None,
        current_bank_token_slot_zone_ids: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor | None]:
        batch_size = slots.shape[0]
        bank_count = len(WORLD_BANK_NAMES)
        slot_positions = torch.arange(self.bank_token_slots, device=slots.device, dtype=torch.long)
        token_queries = self.bank_token_slot_seed.expand(batch_size, -1, -1, -1)
        token_queries = token_queries + bank_states_tensor.unsqueeze(2)
        token_queries = token_queries + self.bank_token_slot_embedding(slot_positions).view(
            1,
            1,
            self.bank_token_slots,
            self.d_model,
        )
        token_query_type = self.future_type_embedding if future_mode else self.current_type_embedding
        token_queries = token_queries + token_query_type.unsqueeze(2)
        flat_queries = token_queries.reshape(batch_size, bank_count * self.bank_token_slots, self.d_model)
        flat_query_mask = torch.ones(
            (batch_size, bank_count * self.bank_token_slots),
            dtype=torch.bool,
            device=slots.device,
        )
        flat_queries = _run_cross_block(
            self.activation_checkpointing,
            self.bank_token_slot_cross,
            flat_queries,
            slots,
            query_mask=flat_query_mask,
            memory_mask=self._slot_mask(batch_size, slots.device),
        )
        flat_queries = _run_encoder_block(
            self.activation_checkpointing,
            self.bank_token_slot_refine,
            flat_queries,
            mask=flat_query_mask,
        )
        source_logits: torch.Tensor | None = None
        if current_bank_token_slot_states is not None:
            flat_current_slot_states = current_bank_token_slot_states.reshape(batch_size, -1, self.d_model)
            if current_bank_token_slot_mask is None:
                flat_current_slot_mask = torch.ones(
                    (batch_size, flat_current_slot_states.shape[1]),
                    dtype=torch.bool,
                    device=slots.device,
                )
            else:
                flat_current_slot_mask = current_bank_token_slot_mask.reshape(batch_size, -1).bool()
            copy_queries = _run_cross_block(
                self.activation_checkpointing,
                self.bank_token_slot_copy_cross,
                flat_queries,
                flat_current_slot_states,
                query_mask=flat_query_mask,
                memory_mask=flat_current_slot_mask,
            )
            copy_queries = _run_encoder_block(
                self.activation_checkpointing,
                self.bank_token_slot_copy_refine,
                copy_queries,
                mask=flat_query_mask,
            )
            transport_type_logits = self.bank_token_slot_type_head(copy_queries).reshape(
                batch_size,
                bank_count * self.bank_token_slots,
                NUM_TOKEN_TYPES,
            )
            transport_zone_logits = self.bank_token_slot_zone_head(copy_queries).reshape(
                batch_size,
                bank_count * self.bank_token_slots,
                MAX_ZONE_ID + 1,
            )
            source_q = self.bank_token_slot_source_q(copy_queries)
            source_k = self.bank_token_slot_source_k(flat_current_slot_states)
            source_memory_logits = torch.einsum("bqd,bkd->bqk", source_q, source_k) / math.sqrt(float(self.d_model))
            source_memory_logits = source_memory_logits.masked_fill(~flat_current_slot_mask.unsqueeze(1), -1e4)
            query_bank_ids = (
                torch.arange(bank_count, device=slots.device, dtype=torch.long)
                .view(1, bank_count, 1)
                .expand(batch_size, bank_count, self.bank_token_slots)
                .reshape(batch_size, -1)
            )
            query_slot_ids = (
                torch.arange(self.bank_token_slots, device=slots.device, dtype=torch.long)
                .view(1, 1, self.bank_token_slots)
                .expand(batch_size, bank_count, self.bank_token_slots)
                .reshape(batch_size, -1)
            )
            current_bank_ids = (
                torch.arange(bank_count, device=slots.device, dtype=torch.long)
                .view(1, bank_count, 1)
                .expand(batch_size, bank_count, self.bank_token_slots)
                .reshape(batch_size, -1)
            )
            current_slot_ids = (
                torch.arange(self.bank_token_slots, device=slots.device, dtype=torch.long)
                .view(1, 1, self.bank_token_slots)
                .expand(batch_size, bank_count, self.bank_token_slots)
                .reshape(batch_size, -1)
            )
            if self.slot_source_same_bank_bias != 0.0:
                same_bank = query_bank_ids.unsqueeze(-1).eq(current_bank_ids.unsqueeze(1))
                source_memory_logits = source_memory_logits + self.slot_source_same_bank_bias * same_bank.float()
            if self.slot_source_same_slot_bias != 0.0:
                same_slot = query_bank_ids.unsqueeze(-1).eq(current_bank_ids.unsqueeze(1)) & query_slot_ids.unsqueeze(-1).eq(
                    current_slot_ids.unsqueeze(1)
                )
                source_memory_logits = source_memory_logits + self.slot_source_same_slot_bias * same_slot.float()
            if current_bank_token_slot_type_ids is not None and self.slot_source_type_match_scale != 0.0:
                flat_transport_type_probs = torch.softmax(transport_type_logits, dim=-1)
                flat_current_type_ids = current_bank_token_slot_type_ids.reshape(batch_size, -1).clamp(
                    min=0,
                    max=NUM_TOKEN_TYPES - 1,
                )
                gathered_type_match = flat_transport_type_probs.gather(
                    -1,
                    flat_current_type_ids.unsqueeze(1).expand(-1, bank_count * self.bank_token_slots, -1),
                )
                source_memory_logits = source_memory_logits + self.slot_source_type_match_scale * gathered_type_match
            if current_bank_token_slot_zone_ids is not None and self.slot_source_zone_transport_scale != 0.0:
                flat_current_zone_ids = current_bank_token_slot_zone_ids.reshape(batch_size, -1).clamp(
                    min=0,
                    max=MAX_ZONE_ID,
                )
                current_zone_one_hot = torch.nn.functional.one_hot(
                    flat_current_zone_ids,
                    num_classes=MAX_ZONE_ID + 1,
                ).float()
                transport_zone_probs = torch.softmax(transport_zone_logits, dim=-1)
                zone_bias = torch.einsum(
                    "bqz,zc,bkc->bqk",
                    transport_zone_probs,
                    self._zone_transport_prior.to(device=slots.device),
                    current_zone_one_hot,
                )
                source_memory_logits = source_memory_logits + self.slot_source_zone_transport_scale * zone_bias
            new_token_logits = self.bank_token_slot_new_head(copy_queries)
            source_logits = torch.cat([source_memory_logits, new_token_logits], dim=-1)
            source_probs = torch.softmax(source_logits, dim=-1)
            source_probs = torch.nan_to_num(source_probs, nan=0.0, posinf=0.0, neginf=0.0)
            copy_probs = source_probs[..., :-1]
            new_prob = source_probs[..., -1:].clamp(min=0.0, max=1.0)
            copied_state = torch.einsum("bqk,bkd->bqd", copy_probs, flat_current_slot_states)
            flat_queries = new_prob * flat_queries + (1.0 - new_prob) * copied_state
        token_slot_states = self.bank_token_slot_out_proj(flat_queries).reshape(
            batch_size,
            bank_count,
            self.bank_token_slots,
            self.d_model,
        )
        token_slot_mask_logits = self.bank_token_slot_mask_head(flat_queries).reshape(
            batch_size,
            bank_count,
            self.bank_token_slots,
        )
        token_slot_type_logits = self.bank_token_slot_type_head(flat_queries).reshape(
            batch_size,
            bank_count,
            self.bank_token_slots,
            NUM_TOKEN_TYPES,
        )
        token_slot_zone_logits = self.bank_token_slot_zone_head(flat_queries).reshape(
            batch_size,
            bank_count,
            self.bank_token_slots,
            MAX_ZONE_ID + 1,
        )
        if source_logits is not None:
            source_logits = source_logits.reshape(
                batch_size,
                bank_count,
                self.bank_token_slots,
                bank_count * self.bank_token_slots + 1,
            )
        return token_slot_states, token_slot_mask_logits, token_slot_type_logits, token_slot_zone_logits, source_logits

    def read_current_bank_token_slots(
        self,
        hidden_state: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        slots = self._slot_states(hidden_state, future_mode=False)
        bank_states_tensor = self._pool_bank_states(slots)
        (
            token_slot_states,
            token_slot_mask_logits,
            token_slot_type_logits,
            token_slot_zone_logits,
            _source_logits,
        ) = self._predict_bank_token_slots(
            bank_states_tensor,
            slots,
            future_mode=False,
        )
        return token_slot_states, token_slot_mask_logits, token_slot_type_logits, token_slot_zone_logits

    def forward(
        self,
        hidden_state: torch.Tensor,
        *,
        current_bank_token_slot_states: torch.Tensor | None = None,
        current_bank_token_slot_mask: torch.Tensor | None = None,
        current_bank_token_slot_type_ids: torch.Tensor | None = None,
        current_bank_token_slot_zone_ids: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor | None]:
        slots = self._slot_states(hidden_state, future_mode=True)
        bank_states_tensor = self._pool_bank_states(slots)
        bank_state_pred = self.bank_out_proj(bank_states_tensor)
        bank_occupancy_logits = self.bank_occupancy_head(bank_states_tensor).squeeze(-1)
        bank_token_presence_logits = self.bank_token_presence_head(bank_states_tensor)
        bank_token_distribution_logits = self.bank_token_distribution_head(bank_states_tensor)
        (
            bank_token_slot_states,
            bank_token_slot_mask_logits,
            bank_token_slot_type_logits,
            bank_token_slot_zone_logits,
            bank_token_slot_source_logits,
        ) = self._predict_bank_token_slots(
            bank_states_tensor,
            slots,
            future_mode=True,
            current_bank_token_slot_states=current_bank_token_slot_states,
            current_bank_token_slot_mask=current_bank_token_slot_mask,
            current_bank_token_slot_type_ids=current_bank_token_slot_type_ids,
            current_bank_token_slot_zone_ids=current_bank_token_slot_zone_ids,
        )
        return (
            bank_state_pred,
            bank_occupancy_logits,
            bank_token_presence_logits,
            bank_token_distribution_logits,
            bank_token_slot_states,
            bank_token_slot_mask_logits,
            bank_token_slot_type_logits,
            bank_token_slot_zone_logits,
            bank_token_slot_source_logits,
        )
