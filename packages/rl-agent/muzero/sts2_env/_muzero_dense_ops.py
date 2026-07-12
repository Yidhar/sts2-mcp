"""Pure dense-observation attention helpers for the MuZero representation."""

from __future__ import annotations

import torch
import torch.nn as nn

from sts2_env.observation_v2 import NUM_DOMAINS


def _safe_self_attn(
    attn: nn.MultiheadAttention,
    x: torch.Tensor,
    mask: torch.Tensor
) -> torch.Tensor:
    """Apply self-attention with safe handling of empty sequences.

    Args:
        attn: MultiheadAttention module
        x: [batch, seq_len, dim]
        mask: [batch, seq_len] bool tensor (True = valid)

    Returns:
        attention output of same shape as x
    """
    if not mask.any():
        return torch.zeros_like(x)

    empty = ~mask.any(dim=1)
    safe_mask = mask.clone()
    safe_mask[empty, 0] = True

    out, _ = attn(x, x, x, key_padding_mask=~safe_mask)
    return out * mask.unsqueeze(-1).float()


def _safe_cross_attn(
    attn: nn.MultiheadAttention,
    query: torch.Tensor,
    key_value: torch.Tensor,
    query_mask: torch.Tensor,
    key_value_mask: torch.Tensor,
) -> torch.Tensor:
    """Apply cross-attention with safe handling of empty sequences.

    Args:
        attn: MultiheadAttention module
        query: [batch, query_len, dim]
        key_value: [batch, kv_len, dim]
        query_mask: [batch, query_len] bool tensor
        key_value_mask: [batch, kv_len] bool tensor

    Returns:
        attention output matching query shape
    """
    if not query_mask.any() or not key_value_mask.any():
        return torch.zeros_like(query)

    empty = ~key_value_mask.any(dim=1)
    safe_mask = key_value_mask.clone()
    safe_mask[empty, 0] = True

    out, _ = attn(query, key_value, key_value, key_padding_mask=~safe_mask)
    out = out * query_mask.unsqueeze(-1).float()
    out[empty] = 0.0
    return out


def _safe_pool(
    attn: nn.MultiheadAttention,
    seed: torch.Tensor,
    x: torch.Tensor,
    mask: torch.Tensor,
) -> torch.Tensor:
    """Pool sequence using attention with learnable seed.

    Args:
        attn: MultiheadAttention module
        seed: [1, 1, dim]
        x: [batch, seq_len, dim]
        mask: [batch, seq_len] bool tensor

    Returns:
        pooled output [batch, dim]
    """
    batch_size, _, dim = x.shape
    if not mask.any():
        return torch.zeros(batch_size, dim, device=x.device)

    empty = ~mask.any(dim=1)
    safe_mask = mask.clone()
    safe_mask[empty, 0] = True

    out, _ = attn(seed.expand(batch_size, -1, -1), x, x, key_padding_mask=~safe_mask)
    pooled = out.squeeze(1)
    pooled[empty] = 0.0
    return pooled


def _masked_mean(x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    """Compute masked mean across sequence dimension.

    Args:
        x: [batch, seq_len, dim]
        mask: [batch, seq_len] bool tensor

    Returns:
        mean of shape [batch, dim]
    """
    batch_size, _, dim = x.shape
    if not mask.any():
        return torch.zeros(batch_size, dim, device=x.device)

    weights = mask.unsqueeze(-1).float()
    return (x * weights).sum(dim=1) / weights.sum(dim=1).clamp(min=1.0)


def _domain_masks_from_obs(obs: dict[str, torch.Tensor], device: torch.device) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Build hard routing masks from decision_domain one-hot vectors.

    Expected layout:
      0 = combat
      1 = build
      2 = route
    """
    domain = obs["decision_domain"]
    if domain.dim() != 2 or domain.shape[1] < NUM_DOMAINS:
        batch = domain.shape[0] if domain.dim() > 0 else 1
        default = torch.zeros(batch, dtype=torch.bool, device=device)
        return default, ~default, default

    domain_idx = torch.argmax(domain, dim=1)
    combat_mask = domain_idx == 0
    build_mask = domain_idx == 1
    route_mask = domain_idx == 2

    # Safety fallback: rows with invalid/empty domain route to build.
    unresolved = ~(combat_mask | build_mask | route_mask)
    if unresolved.any():
        build_mask = build_mask | unresolved

    return combat_mask, build_mask, route_mask
