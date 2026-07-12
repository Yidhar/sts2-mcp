"""Categorical value-support transforms shared by MuZero networks.

This leaf module intentionally has no dependency on the model or token-memory
modules, which keeps their import graph acyclic.
"""

from __future__ import annotations

import torch


def scalar_to_support(value: torch.Tensor, support_size: int = 25) -> torch.Tensor:
    """Convert scalar value to categorical support representation.

    Args:
        value: scalar tensor of shape [batch]
        support_size: number of support points on each side (2*support_size+1 total)

    Returns:
        soft categorical target of shape [batch, 2*support_size+1]
    """
    batch_size = value.shape[0]
    device = value.device

    # Clamp value to support range [-support_size, support_size]
    value = torch.clamp(value, -support_size, support_size)

    # Compute lower and upper indices
    lower = torch.floor(value).long()
    upper = lower + 1

    # Compute interpolation weights
    weights_upper = (value - lower.float()).clamp(0, 1)
    weights_lower = 1.0 - weights_upper

    # Create target distribution
    num_bins = 2 * support_size + 1
    target = torch.zeros(batch_size, num_bins, device=device, dtype=torch.float32)

    # Offset indices to [0, num_bins)
    lower_offset = lower + support_size
    upper_offset = upper + support_size

    # Clip to valid range
    lower_offset = torch.clamp(lower_offset, 0, num_bins - 1)
    upper_offset = torch.clamp(upper_offset, 0, num_bins - 1)

    # Scatter weights
    target.scatter_(1, lower_offset.unsqueeze(1), weights_lower.unsqueeze(1))
    target.scatter_add_(1, upper_offset.unsqueeze(1), weights_upper.unsqueeze(1))

    return target


def support_to_scalar(logits: torch.Tensor, support_size: int = 25) -> torch.Tensor:
    """Convert categorical support logits to scalar value.

    Args:
        logits: shape [batch, 2*support_size+1]
        support_size: number of support points on each side

    Returns:
        scalar tensor of shape [batch, 1]
    """
    device = logits.device

    # Compute probabilities
    probs = torch.softmax(logits, dim=-1)

    # Create support values [-support_size, ..., support_size]
    support_values = torch.arange(
        -support_size, support_size + 1, dtype=torch.float32, device=device
    )

    # Compute expected value
    value = (probs * support_values.unsqueeze(0)).sum(dim=1, keepdim=True)
    return value


def support_tensor_to_scalar(logits: torch.Tensor, support_size: int = 25) -> torch.Tensor:
    """Vectorized support-to-scalar for tensors with an arbitrary leading shape."""
    if logits.dim() == 2:
        return support_to_scalar(logits, support_size)
    flat = logits.reshape(-1, logits.shape[-1])
    scalars = support_to_scalar(flat, support_size)
    return scalars.reshape(*logits.shape[:-1])
