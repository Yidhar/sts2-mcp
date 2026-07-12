"""World-token encoder that produces MuZero latent memory slots."""

from __future__ import annotations

import math

import torch
import torch.nn as nn

from sts2_env.attention_blocks import (
    CrossAttentionBlock,
    EntityPooling,
    RelationBias,
    TransformerEncoderBlock,
)
from sts2_env.observation_v2 import NUM_DOMAINS
from sts2_env.observation_v3 import (
    ENTITY_HASH_BUCKETS,
    MAX_CANDIDATE_LOCAL_TOKENS,
    MAX_ORDER_ID,
    MAX_OWNER_ID,
    MAX_ROLE_ID,
    MAX_WORLD_TOKENS,
    MAX_ZONE_ID,
    NUM_TOKEN_TYPES,
    TOKEN_FEAT_DIM,
    TOKEN_NUMERIC_DIM,
    TOKEN_TEXT_DIM,
)

from ._token_memory_shared import (
    _RELATION_BIAS_KW,
    GLOBAL_MEMORY_BANK_INDEX,
    MEMORY_BANK_NAMES,
    MEMORY_SLOT_LAYOUT_LEGACY,
    WORLD_BANK_NAMES,
    WORLD_BANK_ROLE_IDS,
    WORLD_BANK_ZONE_IDS,
    TokenMemoryEncoderOutput,
    _bucket_effective_len,
    _run_cross_block,
    _run_encoder_block,
    _run_function_with_checkpoint,
    _token_encoder_trace,
    build_memory_slot_bank_ids,
    infer_token_decision_domain,
    normalize_memory_slot_layout,
)


class EntityTokenEmbedder(nn.Module):
    """Standalone token embedder copied out of the mainline omni policy."""

    def __init__(
        self,
        *,
        d_model: int,
        num_token_types: int,
        max_owner_id: int,
        max_role_id: int,
        max_zone_id: int,
        max_order_id: int,
        entity_hash_buckets: int,
        use_internal_numeric_proj: bool = True,
        use_internal_text_proj: bool = True,
    ) -> None:
        super().__init__()
        self.num_token_types = int(num_token_types)
        self.max_owner_id = int(max_owner_id)
        self.max_role_id = int(max_role_id)
        self.max_zone_id = int(max_zone_id)
        self.max_order_id = int(max_order_id)
        self.entity_hash_buckets = int(entity_hash_buckets)
        half = d_model // 2
        self.numeric_width = half
        self.text_width = d_model - half
        self.numeric_proj = (
            nn.Sequential(
                nn.Linear(TOKEN_NUMERIC_DIM, self.numeric_width),
                nn.GELU(),
                nn.Linear(self.numeric_width, self.numeric_width),
            )
            if use_internal_numeric_proj
            else None
        )
        self.text_proj = (
            nn.Sequential(
                nn.Linear(TOKEN_TEXT_DIM, self.text_width),
                nn.GELU(),
                nn.Linear(self.text_width, self.text_width),
            )
            if use_internal_text_proj
            else None
        )
        self.type_embedding = nn.Embedding(self.num_token_types, d_model)
        self.role_embedding = nn.Embedding(self.max_role_id + 1, d_model)
        self.owner_embedding = nn.Embedding(self.max_owner_id + 1, d_model)
        self.zone_embedding = nn.Embedding(self.max_zone_id + 1, d_model)
        self.order_embedding = nn.Embedding(self.max_order_id + 1, d_model)
        self.entity_embedding = nn.Embedding(self.entity_hash_buckets, d_model)
        self.norm = nn.LayerNorm(d_model)

    def forward(
        self,
        token_features: torch.Tensor,
        token_type_ids: torch.Tensor,
        role_ids: torch.Tensor,
        owner_ids: torch.Tensor,
        entity_ids: torch.Tensor,
        zone_ids: torch.Tensor,
        order_ids: torch.Tensor,
        *,
        target_owner_ids: torch.Tensor | None = None,
        target_entity_ids: torch.Tensor | None = None,
        projected_numeric: torch.Tensor | None = None,
        projected_text: torch.Tensor | None = None,
    ) -> torch.Tensor:
        numeric = token_features[..., :TOKEN_NUMERIC_DIM]
        text = token_features[..., TOKEN_NUMERIC_DIM : TOKEN_NUMERIC_DIM + TOKEN_TEXT_DIM]
        token_type_ids = token_type_ids.clamp(min=0, max=self.num_token_types - 1)
        role_ids = role_ids.clamp(min=0, max=self.max_role_id)
        owner_ids = owner_ids.clamp(min=0, max=self.max_owner_id)
        zone_ids = zone_ids.clamp(min=0, max=self.max_zone_id)
        order_ids = order_ids.clamp(min=0, max=self.max_order_id)
        entity_ids = entity_ids.clamp(min=0, max=self.entity_hash_buckets - 1)
        if projected_numeric is None:
            if self.numeric_proj is None:
                raise RuntimeError("projected_numeric must be provided when use_internal_numeric_proj=False")
            projected_numeric = self.numeric_proj(numeric)
        if projected_text is None:
            if self.text_proj is None:
                raise RuntimeError("projected_text must be provided when use_internal_text_proj=False")
            projected_text = self.text_proj(text)
        x = torch.cat([projected_numeric, projected_text], dim=-1)
        x = (
            x
            + self.type_embedding(token_type_ids)
            + self.role_embedding(role_ids)
            + self.owner_embedding(owner_ids)
            + self.zone_embedding(zone_ids)
            + self.order_embedding(order_ids)
            + self.entity_embedding(entity_ids)
        )
        if target_owner_ids is not None:
            target_owner_ids = target_owner_ids.clamp(min=0, max=self.max_owner_id)
            x = x + 0.5 * self.owner_embedding(target_owner_ids)
        if target_entity_ids is not None:
            target_entity_ids = target_entity_ids.clamp(min=0, max=self.entity_hash_buckets - 1)
            x = x + 0.5 * self.entity_embedding(target_entity_ids)
        return self.norm(x)


class TokenMemoryEncoder(nn.Module):
    """Main token-world observation encoder that emits MuZero latent memory slots."""

    def __init__(
        self,
        *,
        d_model: int = 128,
        n_heads: int = 4,
        ffn_dim: int = 512,
        world_layers: int = 4,
        local_layers: int = 1,
        decoder_layers: int = 2,
        candidate_set_layers: int = 1,
        world_bank_top_k: int = 3,
        bank_token_slots: int = 4,
        num_memory_slots: int = 8,
        memory_slot_layout: str = MEMORY_SLOT_LAYOUT_LEGACY,
        action_embed_dim: int = 64,
        dropout: float = 0.0,
        activation_checkpointing: bool = False,
    ) -> None:
        super().__init__()
        self.d_model = int(d_model)
        self.n_heads = int(n_heads)
        self.ffn_dim = int(ffn_dim)
        self.world_layers = int(world_layers)
        self.local_layers = int(local_layers)
        self.decoder_layers = int(decoder_layers)
        self.candidate_set_layers = int(candidate_set_layers)
        self.world_bank_top_k = max(1, min(int(world_bank_top_k), len(WORLD_BANK_NAMES)))
        self.bank_token_slots = max(1, min(int(bank_token_slots), MAX_WORLD_TOKENS))
        self.num_memory_slots = max(int(num_memory_slots), 1)
        self.memory_slot_layout = normalize_memory_slot_layout(memory_slot_layout)
        self.action_embed_dim = int(action_embed_dim)
        self.hidden_dim = self.num_memory_slots * self.d_model
        self.activation_checkpointing = bool(activation_checkpointing)
        self.projection_reuse_max_rows = 16_384
        self.num_slot_banks = len(MEMORY_BANK_NAMES)
        self.global_memory_bank_index = GLOBAL_MEMORY_BANK_INDEX

        self.shared_numeric_width = self.d_model // 2
        self.shared_text_width = self.d_model - self.shared_numeric_width
        self.shared_numeric_trunk = nn.Sequential(
            nn.Linear(TOKEN_NUMERIC_DIM, self.shared_numeric_width),
            nn.GELU(),
            nn.Linear(self.shared_numeric_width, self.shared_numeric_width),
        )
        self.shared_text_trunk = nn.Sequential(
            nn.Linear(TOKEN_TEXT_DIM, self.shared_text_width),
            nn.GELU(),
            nn.Linear(self.shared_text_width, self.shared_text_width),
        )
        self.world_embedder = EntityTokenEmbedder(
            d_model=self.d_model,
            num_token_types=NUM_TOKEN_TYPES,
            max_owner_id=MAX_OWNER_ID,
            max_role_id=MAX_ROLE_ID,
            max_zone_id=MAX_ZONE_ID,
            max_order_id=MAX_ORDER_ID,
            entity_hash_buckets=ENTITY_HASH_BUCKETS,
            use_internal_numeric_proj=False,
            use_internal_text_proj=False,
        )
        self.query_embedder = EntityTokenEmbedder(
            d_model=self.d_model,
            num_token_types=NUM_TOKEN_TYPES,
            max_owner_id=MAX_OWNER_ID,
            max_role_id=MAX_ROLE_ID,
            max_zone_id=MAX_ZONE_ID,
            max_order_id=MAX_ORDER_ID,
            entity_hash_buckets=ENTITY_HASH_BUCKETS,
            use_internal_numeric_proj=False,
            use_internal_text_proj=False,
        )
        self.local_embedder = EntityTokenEmbedder(
            d_model=self.d_model,
            num_token_types=NUM_TOKEN_TYPES,
            max_owner_id=MAX_OWNER_ID,
            max_role_id=MAX_ROLE_ID,
            max_zone_id=MAX_ZONE_ID,
            max_order_id=MAX_ORDER_ID,
            entity_hash_buckets=ENTITY_HASH_BUCKETS,
            use_internal_numeric_proj=False,
            use_internal_text_proj=False,
        )

        self.world_relation_bias = RelationBias(n_heads=self.n_heads, **_RELATION_BIAS_KW)
        self.local_relation_bias = RelationBias(n_heads=self.n_heads, **_RELATION_BIAS_KW)
        self.query_local_relation_bias = RelationBias(n_heads=self.n_heads, **_RELATION_BIAS_KW)
        self.candidate_set_relation_bias = RelationBias(n_heads=self.n_heads, **_RELATION_BIAS_KW)
        self.world_bank_relation_bias = nn.ModuleList(
            [RelationBias(n_heads=self.n_heads, **_RELATION_BIAS_KW) for _ in WORLD_BANK_NAMES]
        )

        self.world_blocks = nn.ModuleList(
            [TransformerEncoderBlock(self.d_model, self.n_heads, self.ffn_dim, dropout=dropout) for _ in range(self.world_layers)]
        )
        self.local_blocks = nn.ModuleList(
            [TransformerEncoderBlock(self.d_model, self.n_heads, self.ffn_dim, dropout=dropout) for _ in range(self.local_layers)]
        )
        self.query_local_bridge = CrossAttentionBlock(self.d_model, self.n_heads, self.ffn_dim, dropout=dropout)
        self.decoder_self_blocks = nn.ModuleList(
            [TransformerEncoderBlock(self.d_model, self.n_heads, self.ffn_dim, dropout=dropout) for _ in range(self.decoder_layers)]
        )
        self.world_bank_cross_blocks = nn.ModuleList(
            [
                nn.ModuleList(
                    [CrossAttentionBlock(self.d_model, self.n_heads, self.ffn_dim, dropout=dropout) for _ in WORLD_BANK_NAMES]
                )
                for _ in range(self.decoder_layers)
            ]
        )
        self.candidate_set_blocks = nn.ModuleList(
            [TransformerEncoderBlock(self.d_model, self.n_heads, self.ffn_dim, dropout=dropout) for _ in range(self.candidate_set_layers)]
        )
        self.world_pool = EntityPooling(self.d_model, self.n_heads, dropout=dropout)
        self.local_pool = EntityPooling(self.d_model, self.n_heads, dropout=dropout)
        self.world_bank_poolers = nn.ModuleList(
            [EntityPooling(self.d_model, self.n_heads, dropout=dropout) for _ in WORLD_BANK_NAMES]
        )
        self.world_bank_router_q = nn.Sequential(nn.LayerNorm(self.d_model), nn.Linear(self.d_model, self.d_model))
        self.world_bank_router_k = nn.Sequential(nn.LayerNorm(self.d_model), nn.Linear(self.d_model, self.d_model))
        self.world_bank_router_bias = nn.Parameter(torch.zeros(len(WORLD_BANK_NAMES)))

        self.register_buffer(
            "_slot_bank_ids",
            torch.as_tensor(
                build_memory_slot_bank_ids(self.num_memory_slots, layout=self.memory_slot_layout),
                dtype=torch.long,
            ),
            persistent=False,
        )
        self.memory_seed = nn.Parameter(torch.randn(1, self.num_memory_slots, self.d_model) * 0.02)
        self.memory_slot_bank_embedding = nn.Embedding(self.num_slot_banks, self.d_model)
        self.memory_bank_summary_proj = nn.Sequential(
            nn.LayerNorm(self.d_model),
            nn.Linear(self.d_model, self.d_model),
            nn.GELU(),
            nn.Linear(self.d_model, self.d_model),
        )
        self.memory_from_world_banks = nn.ModuleList(
            [CrossAttentionBlock(self.d_model, self.n_heads, self.ffn_dim, dropout=dropout) for _ in WORLD_BANK_NAMES]
        )
        self.memory_from_world = CrossAttentionBlock(self.d_model, self.n_heads, self.ffn_dim, dropout=dropout)
        self.memory_from_candidates = CrossAttentionBlock(self.d_model, self.n_heads, self.ffn_dim, dropout=dropout)
        self.memory_blocks = nn.ModuleList(
            [TransformerEncoderBlock(self.d_model, self.n_heads, self.ffn_dim, dropout=dropout) for _ in range(2)]
        )
        self.memory_norm = nn.LayerNorm(self.d_model)
        self.world_to_memory_gate = nn.Sequential(
            nn.LayerNorm(self.d_model),
            nn.Linear(self.d_model, self.num_memory_slots),
        )
        self.domain_head = nn.Sequential(
            nn.LayerNorm(self.d_model),
            nn.Linear(self.d_model, max(self.d_model // 2, NUM_DOMAINS * 2)),
            nn.GELU(),
            nn.Linear(max(self.d_model // 2, NUM_DOMAINS * 2), NUM_DOMAINS),
        )
        self.action_out_proj = nn.Sequential(
            nn.LayerNorm(self.d_model),
            nn.Linear(self.d_model, self.d_model),
            nn.GELU(),
            nn.Linear(self.d_model, self.action_embed_dim),
        )

        for bank_name in WORLD_BANK_NAMES:
            self.register_buffer(
                f"_bank_role_ids_{bank_name}",
                torch.as_tensor(WORLD_BANK_ROLE_IDS.get(bank_name, ()), dtype=torch.long),
                persistent=False,
            )
            self.register_buffer(
                f"_bank_zone_ids_{bank_name}",
                torch.as_tensor(WORLD_BANK_ZONE_IDS.get(bank_name, ()), dtype=torch.long),
                persistent=False,
            )

    def _project_shared_modal_trunk_with_reuse(self, trunk, feature_batches, *, output_dim: int):
        flattened_batches = [batch.reshape(-1, batch.shape[-1]) for batch in feature_batches]
        total_rows = sum(batch.shape[0] for batch in flattened_batches)
        if total_rows == 0:
            return [batch.new_zeros(*batch.shape[:-1], output_dim) for batch in feature_batches]
        if self.projection_reuse_max_rows > 0 and total_rows > self.projection_reuse_max_rows:
            return [trunk(batch).reshape(*original.shape[:-1], output_dim) for batch, original in zip(flattened_batches, feature_batches, strict=False)]
        merged_features = torch.cat(flattened_batches, dim=0)
        unique_features, inverse_indices = torch.unique(merged_features, dim=0, return_inverse=True)
        projected_unique = trunk(unique_features)
        projected_merged = projected_unique.index_select(0, inverse_indices)
        projected_batches = []
        row_offset = 0
        for batch, flattened in zip(feature_batches, flattened_batches, strict=False):
            row_count = flattened.shape[0]
            projected_batches.append(
                projected_merged[row_offset : row_offset + row_count].reshape(*batch.shape[:-1], output_dim)
            )
            row_offset += row_count
        return projected_batches

    @staticmethod
    def _float_tensor(value: torch.Tensor | object, *, device: torch.device) -> torch.Tensor:
        tensor = torch.as_tensor(value, dtype=torch.float32, device=device)
        if tensor.ndim == 1:
            tensor = tensor.unsqueeze(0)
        return tensor

    @staticmethod
    def _long_tensor(value: torch.Tensor | object, *, device: torch.device) -> torch.Tensor:
        tensor = torch.as_tensor(value, dtype=torch.long, device=device)
        if tensor.ndim == 1:
            tensor = tensor.unsqueeze(0)
        return tensor

    @staticmethod
    def _bool_tensor(value: torch.Tensor | object, *, device: torch.device) -> torch.Tensor:
        tensor = torch.as_tensor(value, dtype=torch.bool, device=device)
        if tensor.ndim == 1:
            tensor = tensor.unsqueeze(0)
        return tensor

    @staticmethod
    def _effective_mask_len(mask: torch.Tensor, *, dim: int, min_len: int = 1) -> int:
        """Return max active length along ``dim`` for dynamic attention packing.

        Observation tensors are padded to generous pass-large maxima.  Attention
        cost is quadratic in sequence length, so trimming purely padded suffixes
        before world/local attention substantially lowers peak memory while
        preserving every masked token that can affect the encoder output.
        """

        if mask.numel() == 0:
            return max(int(min_len), 1)
        if dim < 0:
            dim += mask.dim()
        if dim < 0 or dim >= mask.dim():
            raise ValueError(f"Invalid dim={dim} for mask shape={tuple(mask.shape)}")
        active = mask.to(dtype=torch.bool)
        reduce_dims = [axis for axis in range(active.dim()) if axis != dim]
        for axis in sorted(reduce_dims, reverse=True):
            active = active.any(dim=axis)
        active_indices = torch.nonzero(active, as_tuple=False).flatten()
        if active_indices.numel() == 0:
            return max(int(min_len), 1)
        return max(int(active_indices.max().item()) + 1, int(min_len), 1)

    def _build_world_bank_masks(self, world_role_ids, world_zone_ids, world_mask):
        world_mask = torch.as_tensor(world_mask, dtype=torch.bool, device=world_role_ids.device)
        if world_mask.ndim == 1:
            world_mask = world_mask.unsqueeze(0)
        bank_masks = []
        for bank_name in WORLD_BANK_NAMES:
            bank_role_ids = getattr(self, f"_bank_role_ids_{bank_name}")
            bank_zone_ids = getattr(self, f"_bank_zone_ids_{bank_name}")
            role_mask = torch.zeros_like(world_mask)
            zone_mask = torch.zeros_like(world_mask)
            if bank_role_ids.numel():
                role_mask = torch.isin(world_role_ids, bank_role_ids)
            if bank_zone_ids.numel():
                zone_mask = torch.isin(world_zone_ids, bank_zone_ids)
            bank_masks.append((role_mask | zone_mask) & world_mask)
        stacked = torch.stack(bank_masks, dim=1)
        unmatched = world_mask & ~stacked.any(dim=1)
        if unmatched.any():
            stacked = stacked.clone()
            stacked[:, 0, :] = stacked[:, 0, :] | unmatched
        return stacked

    def _compute_world_bank_summaries(self, world_x, world_bank_masks):
        summaries = []
        for bank_index, pooler in enumerate(self.world_bank_poolers):
            summaries.append(pooler(world_x, mask=world_bank_masks[:, bank_index, :]))
        return torch.stack(summaries, dim=1)

    @staticmethod
    def _compute_world_bank_occupancy(world_bank_masks: torch.Tensor, world_mask: torch.Tensor) -> torch.Tensor:
        bank_counts = world_bank_masks.sum(dim=-1).float()
        total_tokens = world_mask.sum(dim=-1, keepdim=True).float().clamp(min=1.0)
        return torch.clamp(bank_counts / total_tokens, min=0.0, max=1.0)

    @staticmethod
    def _compute_world_bank_token_signatures(
        world_bank_masks: torch.Tensor,
        world_type_ids: torch.Tensor,
        world_mask: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Summarise which token families each world bank contains.

        Returns:
            token_presence: [batch, num_banks, num_token_types] binary indicator
            token_distribution: [batch, num_banks, num_token_types] normalized counts
        """

        masked_world_types = world_type_ids.clamp(min=0, max=NUM_TOKEN_TYPES - 1)
        type_one_hot = torch.nn.functional.one_hot(masked_world_types, num_classes=NUM_TOKEN_TYPES).float()
        type_one_hot = type_one_hot * world_mask.unsqueeze(-1).float()
        bank_counts = torch.einsum("bkt,btc->bkc", world_bank_masks.float(), type_one_hot)
        token_presence = (bank_counts > 0.0).float()
        bank_totals = bank_counts.sum(dim=-1, keepdim=True)
        safe_totals = bank_totals.clamp(min=1.0)
        token_distribution = bank_counts / safe_totals
        token_distribution = torch.where(bank_totals > 0.0, token_distribution, torch.zeros_like(token_distribution))
        return token_presence, token_distribution

    def _compute_world_bank_token_slots(
        self,
        world_x: torch.Tensor,
        world_bank_masks: torch.Tensor,
        world_type_ids: torch.Tensor,
        world_zone_ids: torch.Tensor,
        world_entity_ids: torch.Tensor,
        world_order_ids: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Extract a stable top-k token view per bank for finer future-world reconstruction."""

        batch_size, world_token_count, _ = world_x.shape
        device = world_x.device
        select_k = min(self.bank_token_slots, world_token_count)
        token_positions = torch.arange(world_token_count, device=device, dtype=torch.long).unsqueeze(0).expand(batch_size, -1)
        normalized_order = torch.where(
            world_order_ids > 0,
            world_order_ids,
            (MAX_ORDER_ID + 1) + token_positions,
        )
        rank = normalized_order * (world_token_count + 1) + token_positions
        invalid_rank = torch.full_like(rank, (MAX_ORDER_ID + world_token_count + 2) * (world_token_count + 1))

        selected_states: list[torch.Tensor] = []
        selected_masks: list[torch.Tensor] = []
        selected_type_ids: list[torch.Tensor] = []
        selected_zone_ids: list[torch.Tensor] = []
        selected_entity_ids: list[torch.Tensor] = []
        selected_order_ids: list[torch.Tensor] = []

        for bank_index in range(len(WORLD_BANK_NAMES)):
            bank_mask = world_bank_masks[:, bank_index, :]
            bank_rank = torch.where(bank_mask, rank, invalid_rank)
            topk_idx = bank_rank.topk(select_k, dim=-1, largest=False).indices
            gathered_mask = bank_mask.gather(1, topk_idx)
            gathered_states = world_x.gather(1, topk_idx.unsqueeze(-1).expand(-1, -1, self.d_model))
            gathered_states = gathered_states * gathered_mask.unsqueeze(-1).float()
            gathered_type_ids = world_type_ids.gather(1, topk_idx)
            gathered_type_ids = torch.where(gathered_mask, gathered_type_ids, torch.zeros_like(gathered_type_ids))
            gathered_zone_ids = world_zone_ids.gather(1, topk_idx)
            gathered_zone_ids = torch.where(gathered_mask, gathered_zone_ids, torch.zeros_like(gathered_zone_ids))
            gathered_entity_ids = world_entity_ids.gather(1, topk_idx)
            gathered_entity_ids = torch.where(gathered_mask, gathered_entity_ids, torch.zeros_like(gathered_entity_ids))
            gathered_order_ids = world_order_ids.gather(1, topk_idx)
            gathered_order_ids = torch.where(gathered_mask, gathered_order_ids, torch.zeros_like(gathered_order_ids))
            if select_k < self.bank_token_slots:
                pad_slots = self.bank_token_slots - select_k
                gathered_states = torch.cat(
                    [gathered_states, gathered_states.new_zeros((batch_size, pad_slots, self.d_model))],
                    dim=1,
                )
                gathered_mask = torch.cat(
                    [gathered_mask, torch.zeros((batch_size, pad_slots), dtype=torch.bool, device=device)],
                    dim=1,
                )
                gathered_type_ids = torch.cat(
                    [gathered_type_ids, torch.zeros((batch_size, pad_slots), dtype=torch.long, device=device)],
                    dim=1,
                )
                gathered_zone_ids = torch.cat(
                    [gathered_zone_ids, torch.zeros((batch_size, pad_slots), dtype=torch.long, device=device)],
                    dim=1,
                )
                gathered_entity_ids = torch.cat(
                    [gathered_entity_ids, torch.zeros((batch_size, pad_slots), dtype=torch.long, device=device)],
                    dim=1,
                )
                gathered_order_ids = torch.cat(
                    [gathered_order_ids, torch.zeros((batch_size, pad_slots), dtype=torch.long, device=device)],
                    dim=1,
                )
            selected_states.append(gathered_states)
            selected_masks.append(gathered_mask)
            selected_type_ids.append(gathered_type_ids)
            selected_zone_ids.append(gathered_zone_ids)
            selected_entity_ids.append(gathered_entity_ids)
            selected_order_ids.append(gathered_order_ids)

        return (
            torch.stack(selected_states, dim=1),
            torch.stack(selected_masks, dim=1),
            torch.stack(selected_type_ids, dim=1),
            torch.stack(selected_zone_ids, dim=1),
            torch.stack(selected_entity_ids, dim=1),
            torch.stack(selected_order_ids, dim=1),
        )

    def _slot_bank_id_tensor(self, device: torch.device) -> torch.Tensor:
        return self._slot_bank_ids.to(device=device)

    def _slot_bank_mask(self, slot_bank_ids: torch.Tensor, bank_index: int, batch_size: int) -> torch.Tensor:
        return slot_bank_ids.eq(int(bank_index)).unsqueeze(0).expand(batch_size, -1)

    def _slot_bank_summary_context(
        self,
        slot_bank_ids: torch.Tensor,
        *,
        world_bank_summaries: torch.Tensor,
        world_pool: torch.Tensor,
    ) -> torch.Tensor:
        summary_table = torch.cat([world_bank_summaries, world_pool.unsqueeze(1)], dim=1)
        return summary_table[:, slot_bank_ids, :]

    def _build_world_bank_biases(
        self,
        *,
        candidate_query_type_ids,
        candidate_query_role_ids,
        candidate_query_owner_ids,
        candidate_query_entity_ids,
        candidate_query_zone_ids,
        candidate_query_order_ids,
        candidate_query_target_owner_ids,
        candidate_query_target_entity_ids,
        world_type_ids,
        world_role_ids,
        world_owner_ids,
        world_entity_ids,
        world_zone_ids,
        world_order_ids,
    ):
        return [
            relation_bias(
                candidate_query_type_ids,
                world_type_ids,
                candidate_query_owner_ids,
                world_owner_ids,
                candidate_query_entity_ids,
                world_entity_ids,
                candidate_query_role_ids,
                world_role_ids,
                candidate_query_zone_ids,
                world_zone_ids,
                candidate_query_order_ids,
                world_order_ids,
                candidate_query_target_owner_ids,
                candidate_query_target_entity_ids,
            )
            for relation_bias in self.world_bank_relation_bias
        ]

    def _compute_world_bank_routing(self, candidate_x, candidate_mask, world_bank_summaries, world_bank_masks):
        bank_available = world_bank_masks.any(dim=-1)
        router_q = self.world_bank_router_q(candidate_x)
        router_k = self.world_bank_router_k(world_bank_summaries)
        router_logits = torch.einsum("bad,bkd->bak", router_q, router_k) / math.sqrt(float(self.d_model))
        router_logits = router_logits + self.world_bank_router_bias.view(1, 1, -1)
        router_logits = router_logits.masked_fill(~bank_available.unsqueeze(1), -1e4)

        top_k = min(self.world_bank_top_k, len(WORLD_BANK_NAMES))
        if top_k < len(WORLD_BANK_NAMES):
            top_indices = router_logits.topk(top_k, dim=-1).indices
            selected = torch.zeros_like(router_logits, dtype=torch.bool)
            selected.scatter_(-1, top_indices, True)
            selected = selected & bank_available.unsqueeze(1)
        else:
            selected = bank_available.unsqueeze(1).expand_as(router_logits)

        routed_logits = router_logits.masked_fill(~selected, -1e4)
        bank_weights = torch.softmax(routed_logits, dim=-1)
        bank_weights = torch.nan_to_num(bank_weights, nan=0.0, posinf=0.0, neginf=0.0)
        has_selected_bank = selected.any(dim=-1, keepdim=True)
        bank_weights = torch.where(has_selected_bank, bank_weights, torch.zeros_like(bank_weights))
        bank_weights = bank_weights * candidate_mask.unsqueeze(-1).float()
        selected = selected & candidate_mask.unsqueeze(-1)
        return bank_weights, selected

    def _apply_banked_world_cross_attention(
        self,
        *,
        layer_index: int,
        candidate_x: torch.Tensor,
        candidate_mask: torch.Tensor,
        world_x: torch.Tensor,
        world_bank_masks: torch.Tensor,
        world_bank_summaries: torch.Tensor,
        world_bank_biases: list[torch.Tensor],
    ) -> torch.Tensor:
        bank_weights, bank_selected = self._compute_world_bank_routing(
            candidate_x,
            candidate_mask,
            world_bank_summaries,
            world_bank_masks,
        )
        fused_delta = torch.zeros_like(candidate_x)
        for bank_index, _bank_name in enumerate(WORLD_BANK_NAMES):
            bank_query_mask = bank_selected[..., bank_index]
            if not bool(bank_query_mask.any().item()):
                continue
            bank_output = _run_cross_block(
                self.activation_checkpointing,
                self.world_bank_cross_blocks[layer_index][bank_index],
                candidate_x,
                world_x,
                query_mask=bank_query_mask,
                memory_mask=world_bank_masks[:, bank_index, :],
                attn_bias=world_bank_biases[bank_index],
            )
            bank_output = torch.where(bank_query_mask.unsqueeze(-1), bank_output, candidate_x)
            fused_delta = fused_delta + bank_weights[..., bank_index].unsqueeze(-1) * (bank_output - candidate_x)
        return candidate_x + fused_delta

    def _memory_slots(
        self,
        *,
        world_x: torch.Tensor,
        world_mask: torch.Tensor,
        world_bank_masks: torch.Tensor,
        world_bank_summaries: torch.Tensor,
        candidate_x: torch.Tensor,
        candidate_mask: torch.Tensor,
        world_pool: torch.Tensor,
    ) -> torch.Tensor:
        batch_size = world_x.shape[0]
        slot_bank_ids = self._slot_bank_id_tensor(world_x.device)
        memory = self.memory_seed.expand(batch_size, -1, -1)
        memory = memory + self.memory_slot_bank_embedding(slot_bank_ids).unsqueeze(0)
        slot_mask = torch.ones((batch_size, self.num_memory_slots), dtype=torch.bool, device=world_x.device)

        gate_logits = self.world_to_memory_gate(world_pool)
        gate = torch.sigmoid(gate_logits).unsqueeze(-1)
        slot_bank_context = self.memory_bank_summary_proj(
            self._slot_bank_summary_context(
                slot_bank_ids,
                world_bank_summaries=world_bank_summaries,
                world_pool=world_pool,
            )
        )
        memory = memory + gate * slot_bank_context
        for bank_index, bank_cross in enumerate(self.memory_from_world_banks):
            bank_slot_mask = self._slot_bank_mask(slot_bank_ids, bank_index, batch_size)
            if not bool(bank_slot_mask.any().item()):
                continue
            bank_output = _run_cross_block(
                self.activation_checkpointing,
                bank_cross,
                memory,
                world_x,
                query_mask=bank_slot_mask,
                memory_mask=world_bank_masks[:, bank_index, :],
            )
            memory = torch.where(bank_slot_mask.unsqueeze(-1), bank_output, memory)
        global_slot_mask = self._slot_bank_mask(slot_bank_ids, self.global_memory_bank_index, batch_size)
        if bool(global_slot_mask.any().item()):
            global_output = _run_cross_block(
                self.activation_checkpointing,
                self.memory_from_world,
                memory,
                world_x,
                query_mask=global_slot_mask,
                memory_mask=world_mask,
            )
            memory = torch.where(global_slot_mask.unsqueeze(-1), global_output, memory)
        memory = _run_cross_block(
            self.activation_checkpointing,
            self.memory_from_candidates,
            memory,
            candidate_x,
            query_mask=slot_mask,
            memory_mask=candidate_mask,
        )
        for block in self.memory_blocks:
            memory = _run_encoder_block(self.activation_checkpointing, block, memory, mask=slot_mask)
        return self.memory_norm(memory)

    def _flatten_hidden(self, memory_slots: torch.Tensor) -> torch.Tensor:
        return memory_slots.reshape(memory_slots.shape[0], -1)

    def forward(self, obs: dict[str, torch.Tensor]) -> TokenMemoryEncoderOutput:
        _token_encoder_trace("forward enter")
        device = None
        for value in obs.values():
            if isinstance(value, torch.Tensor):
                device = value.device
                break
        if device is None:
            device = torch.device("cpu")

        world_tokens = self._float_tensor(obs["world_tokens"], device=device)
        world_mask = self._bool_tensor(obs["world_token_mask"], device=device)
        world_type_ids = self._long_tensor(obs["world_token_type_ids"], device=device)
        world_role_ids = self._long_tensor(obs["world_token_role_ids"], device=device)
        world_owner_ids = self._long_tensor(obs["world_entity_owner_ids"], device=device)
        world_entity_ids = self._long_tensor(obs["world_token_entity_ids"], device=device)
        world_zone_ids = self._long_tensor(obs["world_token_zone_ids"], device=device)
        world_order_ids = self._long_tensor(obs["world_token_order_ids"], device=device)
        world_type_ids = world_type_ids.clamp(min=0, max=NUM_TOKEN_TYPES - 1)
        world_role_ids = world_role_ids.clamp(min=0, max=MAX_ROLE_ID)
        world_owner_ids = world_owner_ids.clamp(min=0, max=MAX_OWNER_ID)
        world_entity_ids = world_entity_ids.clamp(min=0, max=ENTITY_HASH_BUCKETS - 1)
        world_zone_ids = world_zone_ids.clamp(min=0, max=MAX_ZONE_ID)
        world_order_ids = world_order_ids.clamp(min=0, max=MAX_ORDER_ID)
        world_active_len = self._effective_mask_len(world_mask, dim=1, min_len=1)
        world_len = _bucket_effective_len(
            world_active_len,
            full_len=int(world_tokens.shape[1]),
            env_name="STS2_WORLD_TOKEN_LENGTH_BUCKETS",
            default_buckets=(64, 96, 128, 160, 192, 256, 320, 384, 512, MAX_WORLD_TOKENS),
        )
        if world_len < int(world_tokens.shape[1]):
            world_tokens = world_tokens[:, :world_len, :]
            world_mask = world_mask[:, :world_len]
            world_type_ids = world_type_ids[:, :world_len]
            world_role_ids = world_role_ids[:, :world_len]
            world_owner_ids = world_owner_ids[:, :world_len]
            world_entity_ids = world_entity_ids[:, :world_len]
            world_zone_ids = world_zone_ids[:, :world_len]
            world_order_ids = world_order_ids[:, :world_len]
        _token_encoder_trace(
            f"world tensors ready active_len={world_active_len} packed_len={world_len} full_len={int(obs['world_tokens'].shape[-2])}"
        )

        candidate_query_tokens = self._float_tensor(obs["candidate_query_tokens"], device=device)
        candidate_query_type_ids = self._long_tensor(obs["candidate_query_type_ids"], device=device)
        candidate_query_role_ids = self._long_tensor(obs["candidate_query_role_ids"], device=device)
        candidate_query_owner_ids = self._long_tensor(obs["candidate_query_owner_ids"], device=device)
        candidate_query_entity_ids = self._long_tensor(obs["candidate_query_entity_ids"], device=device)
        candidate_query_zone_ids = self._long_tensor(obs["candidate_query_zone_ids"], device=device)
        candidate_query_order_ids = self._long_tensor(obs["candidate_query_order_ids"], device=device)
        candidate_query_target_owner_ids = self._long_tensor(obs["candidate_query_target_owner_ids"], device=device)
        candidate_query_target_entity_ids = self._long_tensor(obs["candidate_query_target_entity_ids"], device=device)
        candidate_query_type_ids = candidate_query_type_ids.clamp(min=0, max=NUM_TOKEN_TYPES - 1)
        candidate_query_role_ids = candidate_query_role_ids.clamp(min=0, max=MAX_ROLE_ID)
        candidate_query_owner_ids = candidate_query_owner_ids.clamp(min=0, max=MAX_OWNER_ID)
        candidate_query_entity_ids = candidate_query_entity_ids.clamp(min=0, max=ENTITY_HASH_BUCKETS - 1)
        candidate_query_zone_ids = candidate_query_zone_ids.clamp(min=0, max=MAX_ZONE_ID)
        candidate_query_order_ids = candidate_query_order_ids.clamp(min=0, max=MAX_ORDER_ID)
        candidate_query_target_owner_ids = candidate_query_target_owner_ids.clamp(min=0, max=MAX_OWNER_ID)
        candidate_query_target_entity_ids = candidate_query_target_entity_ids.clamp(min=0, max=ENTITY_HASH_BUCKETS - 1)
        candidate_mask = self._bool_tensor(obs["action_mask"], device=device)

        candidate_local_tokens = self._float_tensor(obs["candidate_local_tokens"], device=device)
        candidate_local_masks = self._bool_tensor(obs["candidate_local_masks"], device=device)
        candidate_local_type_ids = self._long_tensor(obs["candidate_local_type_ids"], device=device)
        candidate_local_role_ids = self._long_tensor(obs["candidate_local_role_ids"], device=device)
        candidate_local_owner_ids = self._long_tensor(obs["candidate_local_owner_ids"], device=device)
        candidate_local_entity_ids = self._long_tensor(obs["candidate_local_entity_ids"], device=device)
        candidate_local_zone_ids = self._long_tensor(obs["candidate_local_zone_ids"], device=device)
        candidate_local_order_ids = self._long_tensor(obs["candidate_local_order_ids"], device=device)
        candidate_local_type_ids = candidate_local_type_ids.clamp(min=0, max=NUM_TOKEN_TYPES - 1)
        candidate_local_role_ids = candidate_local_role_ids.clamp(min=0, max=MAX_ROLE_ID)
        candidate_local_owner_ids = candidate_local_owner_ids.clamp(min=0, max=MAX_OWNER_ID)
        candidate_local_entity_ids = candidate_local_entity_ids.clamp(min=0, max=ENTITY_HASH_BUCKETS - 1)
        candidate_local_zone_ids = candidate_local_zone_ids.clamp(min=0, max=MAX_ZONE_ID)
        candidate_local_order_ids = candidate_local_order_ids.clamp(min=0, max=MAX_ORDER_ID)
        local_active_len = self._effective_mask_len(candidate_local_masks, dim=2, min_len=1)
        local_len = _bucket_effective_len(
            local_active_len,
            full_len=int(candidate_local_tokens.shape[2]),
            env_name="STS2_CANDIDATE_LOCAL_LENGTH_BUCKETS",
            default_buckets=(8, 16, 24, 32, MAX_CANDIDATE_LOCAL_TOKENS),
        )
        if local_len < int(candidate_local_tokens.shape[2]):
            candidate_local_tokens = candidate_local_tokens[:, :, :local_len, :]
            candidate_local_masks = candidate_local_masks[:, :, :local_len]
            candidate_local_type_ids = candidate_local_type_ids[:, :, :local_len]
            candidate_local_role_ids = candidate_local_role_ids[:, :, :local_len]
            candidate_local_owner_ids = candidate_local_owner_ids[:, :, :local_len]
            candidate_local_entity_ids = candidate_local_entity_ids[:, :, :local_len]
            candidate_local_zone_ids = candidate_local_zone_ids[:, :, :local_len]
            candidate_local_order_ids = candidate_local_order_ids[:, :, :local_len]
        _token_encoder_trace(
            f"local tensors ready active_len={local_active_len} packed_len={local_len} actions={int(candidate_mask.shape[-1])}"
        )

        batch_size, action_count, local_count, _ = candidate_local_tokens.shape
        flat_local_tokens = candidate_local_tokens.reshape(batch_size * action_count, local_count, TOKEN_FEAT_DIM)
        flat_local_masks = candidate_local_masks.reshape(batch_size * action_count, local_count)
        flat_local_type_ids = candidate_local_type_ids.reshape(batch_size * action_count, local_count)
        flat_local_role_ids = candidate_local_role_ids.reshape(batch_size * action_count, local_count)
        flat_local_owner_ids = candidate_local_owner_ids.reshape(batch_size * action_count, local_count)
        flat_local_entity_ids = candidate_local_entity_ids.reshape(batch_size * action_count, local_count)
        flat_local_zone_ids = candidate_local_zone_ids.reshape(batch_size * action_count, local_count)
        flat_local_order_ids = candidate_local_order_ids.reshape(batch_size * action_count, local_count)

        numeric_slice = slice(0, TOKEN_NUMERIC_DIM)
        text_slice = slice(TOKEN_NUMERIC_DIM, TOKEN_NUMERIC_DIM + TOKEN_TEXT_DIM)
        _token_encoder_trace("before shared_numeric_trunk")
        world_numeric, query_numeric, flat_local_numeric = self._project_shared_modal_trunk_with_reuse(
            self.shared_numeric_trunk,
            [
                world_tokens[..., numeric_slice],
                candidate_query_tokens[..., numeric_slice],
                flat_local_tokens[..., numeric_slice],
            ],
            output_dim=self.shared_numeric_width,
        )
        _token_encoder_trace("after shared_numeric_trunk")
        _token_encoder_trace("before shared_text_trunk")
        world_text, query_text, flat_local_text = self._project_shared_modal_trunk_with_reuse(
            self.shared_text_trunk,
            [
                world_tokens[..., text_slice],
                candidate_query_tokens[..., text_slice],
                flat_local_tokens[..., text_slice],
            ],
            output_dim=self.shared_text_width,
        )
        _token_encoder_trace("after shared_text_trunk")

        _token_encoder_trace("before world_embedder")
        world_x = self.world_embedder(
            world_tokens,
            world_type_ids,
            world_role_ids,
            world_owner_ids,
            world_entity_ids,
            world_zone_ids,
            world_order_ids,
            projected_numeric=world_numeric,
            projected_text=world_text,
        )
        _token_encoder_trace("after world_embedder")
        _token_encoder_trace("before world_relation_bias")
        world_bias = self.world_relation_bias(
            world_type_ids,
            world_type_ids,
            world_owner_ids,
            world_owner_ids,
            world_entity_ids,
            world_entity_ids,
            world_role_ids,
            world_role_ids,
            world_zone_ids,
            world_zone_ids,
            world_order_ids,
            world_order_ids,
        )
        _token_encoder_trace("after world_relation_bias")
        for block_index, block in enumerate(self.world_blocks):
            _token_encoder_trace(f"before world_block[{block_index}]")
            world_x = _run_encoder_block(
                self.activation_checkpointing,
                block,
                world_x,
                mask=world_mask,
                attn_bias=world_bias,
            )
            _token_encoder_trace(f"after world_block[{block_index}]")
        _token_encoder_trace("before world_pool")
        world_pool = self.world_pool(world_x, mask=world_mask)
        _token_encoder_trace("after world_pool")

        _token_encoder_trace("before query_embedder")
        query_x = self.query_embedder(
            candidate_query_tokens,
            candidate_query_type_ids,
            candidate_query_role_ids,
            candidate_query_owner_ids,
            candidate_query_entity_ids,
            candidate_query_zone_ids,
            candidate_query_order_ids,
            target_owner_ids=candidate_query_target_owner_ids,
            target_entity_ids=candidate_query_target_entity_ids,
            projected_numeric=query_numeric,
            projected_text=query_text,
        )
        _token_encoder_trace("after query_embedder")
        _token_encoder_trace("before local_embedder")
        local_x = self.local_embedder(
            flat_local_tokens,
            flat_local_type_ids,
            flat_local_role_ids,
            flat_local_owner_ids,
            flat_local_entity_ids,
            flat_local_zone_ids,
            flat_local_order_ids,
            projected_numeric=flat_local_numeric,
            projected_text=flat_local_text,
        )
        _token_encoder_trace("after local_embedder")
        _token_encoder_trace("before local_relation_bias")
        local_bias = self.local_relation_bias(
            flat_local_type_ids,
            flat_local_type_ids,
            flat_local_owner_ids,
            flat_local_owner_ids,
            flat_local_entity_ids,
            flat_local_entity_ids,
            flat_local_role_ids,
            flat_local_role_ids,
            flat_local_zone_ids,
            flat_local_zone_ids,
            flat_local_order_ids,
            flat_local_order_ids,
        )
        _token_encoder_trace("after local_relation_bias")
        for block_index, block in enumerate(self.local_blocks):
            _token_encoder_trace(f"before local_block[{block_index}]")
            local_x = _run_encoder_block(
                self.activation_checkpointing,
                block,
                local_x,
                mask=flat_local_masks,
                attn_bias=local_bias,
            )
            _token_encoder_trace(f"after local_block[{block_index}]")
        _token_encoder_trace("before local_pool")
        local_pool = self.local_pool(local_x, mask=flat_local_masks).reshape(batch_size, action_count, self.d_model)
        _token_encoder_trace("after local_pool")

        flat_query_x = query_x.reshape(batch_size * action_count, 1, self.d_model)
        flat_query_masks = candidate_mask.reshape(batch_size * action_count, 1)
        flat_query_type_ids = candidate_query_type_ids.reshape(batch_size * action_count, 1)
        flat_query_role_ids = candidate_query_role_ids.reshape(batch_size * action_count, 1)
        flat_query_owner_ids = candidate_query_owner_ids.reshape(batch_size * action_count, 1)
        flat_query_entity_ids = candidate_query_entity_ids.reshape(batch_size * action_count, 1)
        flat_query_zone_ids = candidate_query_zone_ids.reshape(batch_size * action_count, 1)
        flat_query_order_ids = candidate_query_order_ids.reshape(batch_size * action_count, 1)
        flat_query_target_owner_ids = candidate_query_target_owner_ids.reshape(batch_size * action_count, 1)
        flat_query_target_entity_ids = candidate_query_target_entity_ids.reshape(batch_size * action_count, 1)
        _token_encoder_trace("before query_local_relation_bias")
        query_local_bias = self.query_local_relation_bias(
            flat_query_type_ids,
            flat_local_type_ids,
            flat_query_owner_ids,
            flat_local_owner_ids,
            flat_query_entity_ids,
            flat_local_entity_ids,
            flat_query_role_ids,
            flat_local_role_ids,
            flat_query_zone_ids,
            flat_local_zone_ids,
            flat_query_order_ids,
            flat_local_order_ids,
            flat_query_target_owner_ids,
            flat_query_target_entity_ids,
        )
        _token_encoder_trace("after query_local_relation_bias")
        _token_encoder_trace("before query_local_bridge")
        bridged_query_x = _run_cross_block(
            self.activation_checkpointing,
            self.query_local_bridge,
            flat_query_x,
            local_x,
            query_mask=flat_query_masks,
            memory_mask=flat_local_masks,
            attn_bias=query_local_bias,
        )
        _token_encoder_trace("after query_local_bridge")
        has_local_context = flat_local_masks.any(dim=1, keepdim=True).unsqueeze(-1)
        bridged_query_x = torch.where(has_local_context, bridged_query_x, flat_query_x)
        candidate_x = bridged_query_x.reshape(batch_size, action_count, self.d_model) + local_pool

        _token_encoder_trace("before candidate_set_relation_bias")
        set_bias = self.candidate_set_relation_bias(
            candidate_query_type_ids,
            candidate_query_type_ids,
            candidate_query_owner_ids,
            candidate_query_owner_ids,
            candidate_query_entity_ids,
            candidate_query_entity_ids,
            candidate_query_role_ids,
            candidate_query_role_ids,
            candidate_query_zone_ids,
            candidate_query_zone_ids,
            candidate_query_order_ids,
            candidate_query_order_ids,
        )
        _token_encoder_trace("after candidate_set_relation_bias")
        _token_encoder_trace("before world_bank_masks")
        world_bank_masks = self._build_world_bank_masks(world_role_ids, world_zone_ids, world_mask)
        _token_encoder_trace("after world_bank_masks")
        _token_encoder_trace("before world_bank_summaries")
        world_bank_summaries = self._compute_world_bank_summaries(world_x, world_bank_masks)
        _token_encoder_trace("after world_bank_summaries")
        _token_encoder_trace("before world_bank_biases")
        world_bank_biases = self._build_world_bank_biases(
            candidate_query_type_ids=candidate_query_type_ids,
            candidate_query_role_ids=candidate_query_role_ids,
            candidate_query_owner_ids=candidate_query_owner_ids,
            candidate_query_entity_ids=candidate_query_entity_ids,
            candidate_query_zone_ids=candidate_query_zone_ids,
            candidate_query_order_ids=candidate_query_order_ids,
            candidate_query_target_owner_ids=candidate_query_target_owner_ids,
            candidate_query_target_entity_ids=candidate_query_target_entity_ids,
            world_type_ids=world_type_ids,
            world_role_ids=world_role_ids,
            world_owner_ids=world_owner_ids,
            world_entity_ids=world_entity_ids,
            world_zone_ids=world_zone_ids,
            world_order_ids=world_order_ids,
        )
        _token_encoder_trace("after world_bank_biases")
        for layer_index in range(self.decoder_layers):
            _token_encoder_trace(f"before decoder_self[{layer_index}]")
            candidate_x = _run_encoder_block(
                self.activation_checkpointing,
                self.decoder_self_blocks[layer_index],
                candidate_x,
                mask=candidate_mask,
                attn_bias=set_bias,
            )
            _token_encoder_trace(f"after decoder_self[{layer_index}]")
            def _apply_bank_cross(
                candidate_x_: torch.Tensor,
                world_x_: torch.Tensor,
                layer_index_: int = layer_index,
            ) -> torch.Tensor:
                return self._apply_banked_world_cross_attention(
                    layer_index=layer_index_,
                    candidate_x=candidate_x_,
                    candidate_mask=candidate_mask,
                    world_x=world_x_,
                    world_bank_masks=world_bank_masks,
                    world_bank_summaries=world_bank_summaries,
                    world_bank_biases=world_bank_biases,
                )

            _token_encoder_trace(f"before bank_cross[{layer_index}]")
            candidate_x = _run_function_with_checkpoint(
                self.activation_checkpointing,
                _apply_bank_cross,
                candidate_x,
                world_x,
            )
            _token_encoder_trace(f"after bank_cross[{layer_index}]")
        for block_index, block in enumerate(self.candidate_set_blocks):
            _token_encoder_trace(f"before candidate_set_block[{block_index}]")
            candidate_x = _run_encoder_block(
                self.activation_checkpointing,
                block,
                candidate_x,
                mask=candidate_mask,
                attn_bias=set_bias,
            )
            _token_encoder_trace(f"after candidate_set_block[{block_index}]")

        _token_encoder_trace("before world_bank_token_signatures")
        world_bank_token_presence, world_bank_token_distribution = self._compute_world_bank_token_signatures(
            world_bank_masks=world_bank_masks,
            world_type_ids=world_type_ids,
            world_mask=world_mask,
        )
        _token_encoder_trace("after world_bank_token_signatures")
        _token_encoder_trace("before world_bank_token_slots")
        (
            world_bank_token_slot_states,
            world_bank_token_slot_mask,
            world_bank_token_slot_type_ids,
            world_bank_token_slot_zone_ids,
            world_bank_token_slot_entity_ids,
            world_bank_token_slot_order_ids,
        ) = self._compute_world_bank_token_slots(
            world_x=world_x,
            world_bank_masks=world_bank_masks,
            world_type_ids=world_type_ids,
            world_zone_ids=world_zone_ids,
            world_entity_ids=world_entity_ids,
            world_order_ids=world_order_ids,
        )
        _token_encoder_trace("after world_bank_token_slots")

        _token_encoder_trace("before memory_slots")
        memory_slots = self._memory_slots(
            world_x=world_x,
            world_mask=world_mask,
            world_bank_masks=world_bank_masks,
            world_bank_summaries=world_bank_summaries,
            candidate_x=candidate_x,
            candidate_mask=candidate_mask,
            world_pool=world_pool,
        )
        _token_encoder_trace("after memory_slots")
        hidden_state = self._flatten_hidden(memory_slots)
        action_embeddings = self.action_out_proj(candidate_x) * candidate_mask.unsqueeze(-1).float()
        decision_domain = infer_token_decision_domain(obs, device=device)
        if decision_domain is None:
            domain_logits = self.domain_head(world_pool)
            decision_domain = torch.softmax(domain_logits, dim=-1)
        _token_encoder_trace("forward exit")
        return TokenMemoryEncoderOutput(
            hidden_state=hidden_state,
            action_embeddings=action_embeddings,
            world_pool=world_pool,
            world_bank_summaries=world_bank_summaries,
            world_bank_occupancy=self._compute_world_bank_occupancy(world_bank_masks, world_mask),
            world_bank_token_presence=world_bank_token_presence,
            world_bank_token_distribution=world_bank_token_distribution,
            world_bank_token_slot_states=world_bank_token_slot_states,
            world_bank_token_slot_mask=world_bank_token_slot_mask,
            world_bank_token_slot_type_ids=world_bank_token_slot_type_ids,
            world_bank_token_slot_zone_ids=world_bank_token_slot_zone_ids,
            world_bank_token_slot_entity_ids=world_bank_token_slot_entity_ids,
            world_bank_token_slot_order_ids=world_bank_token_slot_order_ids,
            decision_domain=decision_domain,
        )
