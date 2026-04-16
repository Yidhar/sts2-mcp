"""Reusable attention blocks for the search-free omni-attention policy."""

from __future__ import annotations

import math

import torch
import torch.nn as nn


def _ensure_mask(mask: torch.Tensor | None, *, device: torch.device, batch_size: int, length: int) -> torch.Tensor:
    if mask is None:
        return torch.ones((batch_size, length), dtype=torch.bool, device=device)
    mask = torch.as_tensor(mask, dtype=torch.bool, device=device)
    if mask.ndim == 1:
        mask = mask.unsqueeze(0)
    return mask


class RelationBias(nn.Module):
    """Typed relation bias for self/cross attention.

    The bias is intentionally lightweight:
      - learned pair bias between token types
      - learned owner-pair prior plus same-owner bonus
      - learned same-entity bonus for non-zero entity ids
      - learned role-pair prior plus same-role bonus
    """

    def __init__(
        self,
        *,
        num_token_types: int,
        n_heads: int,
        max_owner_id: int = 127,
        max_role_id: int = 31,
        max_zone_id: int = 15,
        max_order_id: int = 63,
        max_order_offset: int = 16,
    ):
        super().__init__()
        self.num_token_types = int(num_token_types)
        self.n_heads = int(n_heads)
        self.max_owner_id = int(max_owner_id)
        self.max_role_id = int(max_role_id)
        self.max_zone_id = int(max_zone_id)
        self.max_order_id = int(max_order_id)
        self.max_order_offset = int(max_order_offset)
        self.type_pair_bias = nn.Embedding(self.num_token_types * self.num_token_types, self.n_heads)
        self.owner_pair_bias = nn.Embedding((self.max_owner_id + 1) * (self.max_owner_id + 1), self.n_heads)
        self.role_pair_bias = nn.Embedding((self.max_role_id + 1) * (self.max_role_id + 1), self.n_heads)
        self.zone_pair_bias = nn.Embedding((self.max_zone_id + 1) * (self.max_zone_id + 1), self.n_heads)
        self.relative_order_bias = nn.Embedding(2 * self.max_order_offset + 1, self.n_heads)
        self.target_owner_pair_bias = nn.Embedding((self.max_owner_id + 1) * (self.max_owner_id + 1), self.n_heads)
        self.same_owner_bias = nn.Parameter(torch.zeros(self.n_heads))
        self.same_entity_bias = nn.Parameter(torch.zeros(self.n_heads))
        self.same_role_bias = nn.Parameter(torch.zeros(self.n_heads))
        self.same_zone_bias = nn.Parameter(torch.zeros(self.n_heads))
        self.same_target_entity_bias = nn.Parameter(torch.zeros(self.n_heads))

    def forward(
        self,
        query_type_ids: torch.Tensor,
        key_type_ids: torch.Tensor,
        query_owner_ids: torch.Tensor | None = None,
        key_owner_ids: torch.Tensor | None = None,
        query_entity_ids: torch.Tensor | None = None,
        key_entity_ids: torch.Tensor | None = None,
        query_role_ids: torch.Tensor | None = None,
        key_role_ids: torch.Tensor | None = None,
        query_zone_ids: torch.Tensor | None = None,
        key_zone_ids: torch.Tensor | None = None,
        query_order_ids: torch.Tensor | None = None,
        key_order_ids: torch.Tensor | None = None,
        query_target_owner_ids: torch.Tensor | None = None,
        query_target_entity_ids: torch.Tensor | None = None,
    ) -> torch.Tensor:
        query_type_ids = torch.as_tensor(query_type_ids, dtype=torch.long)
        key_type_ids = torch.as_tensor(key_type_ids, dtype=torch.long, device=query_type_ids.device)
        if query_type_ids.ndim == 1:
            query_type_ids = query_type_ids.unsqueeze(0)
        if key_type_ids.ndim == 1:
            key_type_ids = key_type_ids.unsqueeze(0)

        pair_index = query_type_ids.unsqueeze(-1) * self.num_token_types + key_type_ids.unsqueeze(-2)
        bias = self.type_pair_bias(pair_index).permute(0, 3, 1, 2)

        if query_owner_ids is not None and key_owner_ids is not None:
            query_owner_ids = torch.as_tensor(query_owner_ids, dtype=torch.long, device=bias.device)
            key_owner_ids = torch.as_tensor(key_owner_ids, dtype=torch.long, device=bias.device)
            if query_owner_ids.ndim == 1:
                query_owner_ids = query_owner_ids.unsqueeze(0)
            if key_owner_ids.ndim == 1:
                key_owner_ids = key_owner_ids.unsqueeze(0)
            query_owner_ids = query_owner_ids.clamp(min=0, max=self.max_owner_id)
            key_owner_ids = key_owner_ids.clamp(min=0, max=self.max_owner_id)
            owner_pair_index = query_owner_ids.unsqueeze(-1) * (self.max_owner_id + 1) + key_owner_ids.unsqueeze(-2)
            bias = bias + self.owner_pair_bias(owner_pair_index).permute(0, 3, 1, 2)
            same_owner = (
                (query_owner_ids.unsqueeze(-1) == key_owner_ids.unsqueeze(-2))
                & (query_owner_ids.unsqueeze(-1) > 0)
                & (key_owner_ids.unsqueeze(-2) > 0)
            )
            bias = bias + same_owner.unsqueeze(1).float() * self.same_owner_bias.view(1, -1, 1, 1)

        if query_entity_ids is not None and key_entity_ids is not None:
            query_entity_ids = torch.as_tensor(query_entity_ids, dtype=torch.long, device=bias.device)
            key_entity_ids = torch.as_tensor(key_entity_ids, dtype=torch.long, device=bias.device)
            if query_entity_ids.ndim == 1:
                query_entity_ids = query_entity_ids.unsqueeze(0)
            if key_entity_ids.ndim == 1:
                key_entity_ids = key_entity_ids.unsqueeze(0)
            same_entity = (
                (query_entity_ids.unsqueeze(-1) == key_entity_ids.unsqueeze(-2))
                & (query_entity_ids.unsqueeze(-1) > 0)
                & (key_entity_ids.unsqueeze(-2) > 0)
            )
            bias = bias + same_entity.unsqueeze(1).float() * self.same_entity_bias.view(1, -1, 1, 1)

        if query_role_ids is not None and key_role_ids is not None:
            query_role_ids = torch.as_tensor(query_role_ids, dtype=torch.long, device=bias.device)
            key_role_ids = torch.as_tensor(key_role_ids, dtype=torch.long, device=bias.device)
            if query_role_ids.ndim == 1:
                query_role_ids = query_role_ids.unsqueeze(0)
            if key_role_ids.ndim == 1:
                key_role_ids = key_role_ids.unsqueeze(0)
            query_role_ids = query_role_ids.clamp(min=0, max=self.max_role_id)
            key_role_ids = key_role_ids.clamp(min=0, max=self.max_role_id)
            role_pair_index = query_role_ids.unsqueeze(-1) * (self.max_role_id + 1) + key_role_ids.unsqueeze(-2)
            bias = bias + self.role_pair_bias(role_pair_index).permute(0, 3, 1, 2)
            same_role = (
                (query_role_ids.unsqueeze(-1) == key_role_ids.unsqueeze(-2))
                & (query_role_ids.unsqueeze(-1) > 0)
                & (key_role_ids.unsqueeze(-2) > 0)
            )
            bias = bias + same_role.unsqueeze(1).float() * self.same_role_bias.view(1, -1, 1, 1)

        same_zone = None
        if query_zone_ids is not None and key_zone_ids is not None:
            query_zone_ids = torch.as_tensor(query_zone_ids, dtype=torch.long, device=bias.device)
            key_zone_ids = torch.as_tensor(key_zone_ids, dtype=torch.long, device=bias.device)
            if query_zone_ids.ndim == 1:
                query_zone_ids = query_zone_ids.unsqueeze(0)
            if key_zone_ids.ndim == 1:
                key_zone_ids = key_zone_ids.unsqueeze(0)
            query_zone_ids = query_zone_ids.clamp(min=0, max=self.max_zone_id)
            key_zone_ids = key_zone_ids.clamp(min=0, max=self.max_zone_id)
            zone_pair_index = query_zone_ids.unsqueeze(-1) * (self.max_zone_id + 1) + key_zone_ids.unsqueeze(-2)
            bias = bias + self.zone_pair_bias(zone_pair_index).permute(0, 3, 1, 2)
            same_zone = (
                (query_zone_ids.unsqueeze(-1) == key_zone_ids.unsqueeze(-2))
                & (query_zone_ids.unsqueeze(-1) > 0)
                & (key_zone_ids.unsqueeze(-2) > 0)
            )
            bias = bias + same_zone.unsqueeze(1).float() * self.same_zone_bias.view(1, -1, 1, 1)

        if same_zone is not None and query_order_ids is not None and key_order_ids is not None:
            query_order_ids = torch.as_tensor(query_order_ids, dtype=torch.long, device=bias.device)
            key_order_ids = torch.as_tensor(key_order_ids, dtype=torch.long, device=bias.device)
            if query_order_ids.ndim == 1:
                query_order_ids = query_order_ids.unsqueeze(0)
            if key_order_ids.ndim == 1:
                key_order_ids = key_order_ids.unsqueeze(0)
            query_order_ids = query_order_ids.clamp(min=0, max=self.max_order_id)
            key_order_ids = key_order_ids.clamp(min=0, max=self.max_order_id)
            rel_order = (query_order_ids.unsqueeze(-1) - key_order_ids.unsqueeze(-2)).clamp(
                min=-self.max_order_offset,
                max=self.max_order_offset,
            ) + self.max_order_offset
            valid_order = same_zone & (query_order_ids.unsqueeze(-1) > 0) & (key_order_ids.unsqueeze(-2) > 0)
            bias = bias + self.relative_order_bias(rel_order).permute(0, 3, 1, 2) * valid_order.unsqueeze(1).float()

        if query_target_owner_ids is not None and key_owner_ids is not None:
            query_target_owner_ids = torch.as_tensor(query_target_owner_ids, dtype=torch.long, device=bias.device)
            if query_target_owner_ids.ndim == 1:
                query_target_owner_ids = query_target_owner_ids.unsqueeze(0)
            query_target_owner_ids = query_target_owner_ids.clamp(min=0, max=self.max_owner_id)
            key_owner_ids = torch.as_tensor(key_owner_ids, dtype=torch.long, device=bias.device)
            if key_owner_ids.ndim == 1:
                key_owner_ids = key_owner_ids.unsqueeze(0)
            key_owner_ids = key_owner_ids.clamp(min=0, max=self.max_owner_id)
            target_owner_pair_index = query_target_owner_ids.unsqueeze(-1) * (self.max_owner_id + 1) + key_owner_ids.unsqueeze(-2)
            bias = bias + self.target_owner_pair_bias(target_owner_pair_index).permute(0, 3, 1, 2)

        if query_target_entity_ids is not None and key_entity_ids is not None:
            query_target_entity_ids = torch.as_tensor(query_target_entity_ids, dtype=torch.long, device=bias.device)
            if query_target_entity_ids.ndim == 1:
                query_target_entity_ids = query_target_entity_ids.unsqueeze(0)
            key_entity_ids = torch.as_tensor(key_entity_ids, dtype=torch.long, device=bias.device)
            if key_entity_ids.ndim == 1:
                key_entity_ids = key_entity_ids.unsqueeze(0)
            same_target_entity = (
                (query_target_entity_ids.unsqueeze(-1) == key_entity_ids.unsqueeze(-2))
                & (query_target_entity_ids.unsqueeze(-1) > 0)
                & (key_entity_ids.unsqueeze(-2) > 0)
            )
            bias = bias + same_target_entity.unsqueeze(1).float() * self.same_target_entity_bias.view(1, -1, 1, 1)

        return bias


class MultiheadAttentionWithBias(nn.Module):
    """Small custom MHA with additive relation bias support."""

    def __init__(self, d_model: int, n_heads: int, dropout: float = 0.0):
        super().__init__()
        if d_model % n_heads != 0:
            raise ValueError(f"d_model={d_model} must be divisible by n_heads={n_heads}")

        self.d_model = int(d_model)
        self.n_heads = int(n_heads)
        self.head_dim = self.d_model // self.n_heads

        self.q_proj = nn.Linear(self.d_model, self.d_model)
        self.k_proj = nn.Linear(self.d_model, self.d_model)
        self.v_proj = nn.Linear(self.d_model, self.d_model)
        self.out_proj = nn.Linear(self.d_model, self.d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(
        self,
        query: torch.Tensor,
        key_value: torch.Tensor,
        *,
        query_mask: torch.Tensor | None = None,
        key_mask: torch.Tensor | None = None,
        attn_bias: torch.Tensor | None = None,
    ) -> torch.Tensor:
        batch_size, query_len, _ = query.shape
        key_len = key_value.shape[1]
        device = query.device

        query_mask = _ensure_mask(query_mask, device=device, batch_size=batch_size, length=query_len)
        key_mask = _ensure_mask(key_mask, device=device, batch_size=batch_size, length=key_len)

        q = self.q_proj(query).view(batch_size, query_len, self.n_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(key_value).view(batch_size, key_len, self.n_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(key_value).view(batch_size, key_len, self.n_heads, self.head_dim).transpose(1, 2)

        scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.head_dim)
        if attn_bias is not None:
            scores = scores + attn_bias.to(device=device, dtype=scores.dtype)

        key_mask_expanded = key_mask.unsqueeze(1).unsqueeze(2)
        scores = scores.masked_fill(~key_mask_expanded, -1e9)

        attn = torch.softmax(scores, dim=-1)
        attn = torch.nan_to_num(attn, nan=0.0, posinf=0.0, neginf=0.0)
        attn = self.dropout(attn)

        out = torch.matmul(attn, v).transpose(1, 2).contiguous().view(batch_size, query_len, self.d_model)
        out = self.out_proj(out)
        out = out * query_mask.unsqueeze(-1).float()
        return out


class FeedForwardBlock(nn.Module):
    def __init__(self, d_model: int, ffn_dim: int, dropout: float = 0.0):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(d_model, ffn_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(ffn_dim, d_model),
            nn.Dropout(dropout),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class TransformerEncoderBlock(nn.Module):
    def __init__(self, d_model: int, n_heads: int, ffn_dim: int, dropout: float = 0.0):
        super().__init__()
        self.norm1 = nn.LayerNorm(d_model)
        self.attn = MultiheadAttentionWithBias(d_model, n_heads, dropout=dropout)
        self.norm2 = nn.LayerNorm(d_model)
        self.ffn = FeedForwardBlock(d_model, ffn_dim, dropout=dropout)

    def forward(
        self,
        x: torch.Tensor,
        *,
        mask: torch.Tensor | None = None,
        attn_bias: torch.Tensor | None = None,
    ) -> torch.Tensor:
        normed = self.norm1(x)
        attn_out = self.attn(normed, normed, query_mask=mask, key_mask=mask, attn_bias=attn_bias)
        x = x + attn_out
        x = x + self.ffn(self.norm2(x))
        if mask is not None:
            x = x * torch.as_tensor(mask, dtype=torch.bool, device=x.device).unsqueeze(-1).float()
        return x


class CrossAttentionBlock(nn.Module):
    def __init__(self, d_model: int, n_heads: int, ffn_dim: int, dropout: float = 0.0):
        super().__init__()
        self.query_norm = nn.LayerNorm(d_model)
        self.memory_norm = nn.LayerNorm(d_model)
        self.cross_attn = MultiheadAttentionWithBias(d_model, n_heads, dropout=dropout)
        self.ffn_norm = nn.LayerNorm(d_model)
        self.ffn = FeedForwardBlock(d_model, ffn_dim, dropout=dropout)

    def forward(
        self,
        query: torch.Tensor,
        memory: torch.Tensor,
        *,
        query_mask: torch.Tensor | None = None,
        memory_mask: torch.Tensor | None = None,
        attn_bias: torch.Tensor | None = None,
    ) -> torch.Tensor:
        attn_out = self.cross_attn(
            self.query_norm(query),
            self.memory_norm(memory),
            query_mask=query_mask,
            key_mask=memory_mask,
            attn_bias=attn_bias,
        )
        query = query + attn_out
        query = query + self.ffn(self.ffn_norm(query))
        if query_mask is not None:
            query = query * torch.as_tensor(query_mask, dtype=torch.bool, device=query.device).unsqueeze(-1).float()
        return query


class CandidateDecoderBlock(nn.Module):
    """Local self-attn + world cross-attn + FFN for candidate decoding."""

    def __init__(self, d_model: int, n_heads: int, ffn_dim: int, dropout: float = 0.0):
        super().__init__()
        self.self_block = TransformerEncoderBlock(d_model, n_heads, ffn_dim, dropout=dropout)
        self.cross_block = CrossAttentionBlock(d_model, n_heads, ffn_dim, dropout=dropout)

    def forward(
        self,
        candidate_tokens: torch.Tensor,
        world_memory: torch.Tensor,
        *,
        candidate_mask: torch.Tensor | None = None,
        world_mask: torch.Tensor | None = None,
        self_bias: torch.Tensor | None = None,
        cross_bias: torch.Tensor | None = None,
    ) -> torch.Tensor:
        candidate_tokens = self.self_block(candidate_tokens, mask=candidate_mask, attn_bias=self_bias)
        candidate_tokens = self.cross_block(
            candidate_tokens,
            world_memory,
            query_mask=candidate_mask,
            memory_mask=world_mask,
            attn_bias=cross_bias,
        )
        return candidate_tokens


class EntityPooling(nn.Module):
    """Learned-query pooling over a masked token set."""

    def __init__(self, d_model: int, n_heads: int, dropout: float = 0.0):
        super().__init__()
        self.seed = nn.Parameter(torch.zeros(1, 1, d_model))
        nn.init.normal_(self.seed, mean=0.0, std=0.02)
        self.attn = MultiheadAttentionWithBias(d_model, n_heads, dropout=dropout)
        self.norm = nn.LayerNorm(d_model)

    def forward(
        self,
        x: torch.Tensor,
        *,
        mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        batch_size, _, d_model = x.shape
        device = x.device
        if mask is None:
            mask = torch.ones((batch_size, x.shape[1]), dtype=torch.bool, device=device)
        else:
            mask = torch.as_tensor(mask, dtype=torch.bool, device=device)
        if mask.ndim == 1:
            mask = mask.unsqueeze(0)

        pooled = self.attn(
            self.seed.expand(batch_size, -1, -1),
            x,
            query_mask=torch.ones((batch_size, 1), dtype=torch.bool, device=device),
            key_mask=mask,
            attn_bias=None,
        ).squeeze(1)
        pooled = self.norm(pooled)
        has_tokens = mask.any(dim=1, keepdim=False).unsqueeze(-1)
        return torch.where(has_tokens, pooled, torch.zeros((batch_size, d_model), device=device))
