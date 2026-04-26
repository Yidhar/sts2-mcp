"""Latent-space regularizers for JEPA-style MuZero world models.

These utilities implement a lightweight SIGReg-inspired Gaussian latent
regularizer for structured STS2 latents.  The goal is not to force exact
normality, but to keep representation/dynamics latents non-collapsed,
well-scaled, and decorrelated enough that multi-step latent rollout stays on a
usable manifold.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch


@dataclass(frozen=True)
class LatentRegularizerMetrics:
    mean_abs: float
    var_mean: float
    var_std: float
    cov_offdiag: float
    proj_var_mean: float


def _flatten_latents(z: torch.Tensor) -> torch.Tensor:
    """Flatten arbitrary leading axes into samples and keep the last dim."""
    if z.dim() < 2:
        z = z.reshape(1, -1)
    return z.reshape(-1, z.shape[-1]).float()


def _offdiag(x: torch.Tensor) -> torch.Tensor:
    if x.shape[0] <= 1 or x.shape[1] <= 1:
        return x.new_zeros((0,), dtype=x.dtype)
    mask = ~torch.eye(x.shape[0], dtype=torch.bool, device=x.device)
    return x[mask]


def latent_gaussian_regularizer(
    z: torch.Tensor,
    *,
    projection_count: int = 64,
    cov_weight: float = 0.05,
    eps: float = 1e-6,
) -> tuple[torch.Tensor, LatentRegularizerMetrics]:
    """Return a small Gaussian/isotropy regularizer plus diagnostic metrics.

    The loss combines per-dimension mean/variance, random-projection
    mean/variance, and a weak covariance off-diagonal penalty.  It accepts
    hidden states ``[B, H]`` as well as token slots ``[B, S, D]`` or
    bank/slot tensors ``[B, Bank, Slot, D]``.
    """

    flat = _flatten_latents(z)
    if flat.numel() == 0:
        zero = z.new_zeros(())
        return zero, LatentRegularizerMetrics(0.0, 0.0, 0.0, 0.0, 0.0)

    sample_count, dim = flat.shape
    centered = flat - flat.mean(dim=0, keepdim=True)
    mean = flat.mean(dim=0)
    var = centered.pow(2).mean(dim=0)

    mean_loss = mean.pow(2).mean()
    var_loss = (var - 1.0).pow(2).mean()

    proj_count = min(max(int(projection_count), 0), max(dim, 1))
    if proj_count > 0:
        random_dirs = torch.randn(dim, proj_count, device=flat.device, dtype=flat.dtype)
        random_dirs = torch.nn.functional.normalize(random_dirs, dim=0, eps=eps)
        proj = flat @ random_dirs
        proj_centered = proj - proj.mean(dim=0, keepdim=True)
        proj_mean_loss = proj.mean(dim=0).pow(2).mean()
        proj_var = proj_centered.pow(2).mean(dim=0)
        proj_var_loss = (proj_var - 1.0).pow(2).mean()
        proj_loss = proj_mean_loss + proj_var_loss
        proj_var_mean = float(proj_var.detach().mean().item())
    else:
        proj_loss = flat.new_zeros(())
        proj_var_mean = 0.0

    if sample_count > 1 and dim > 1 and cov_weight > 0.0:
        normed = centered / var.clamp(min=eps).sqrt().unsqueeze(0)
        cov = normed.transpose(0, 1) @ normed / float(max(sample_count, 1))
        cov_offdiag = _offdiag(cov)
        cov_loss = cov_offdiag.pow(2).mean() if cov_offdiag.numel() else flat.new_zeros(())
    else:
        cov_loss = flat.new_zeros(())

    loss = mean_loss + var_loss + proj_loss + float(cov_weight) * cov_loss
    metrics = LatentRegularizerMetrics(
        mean_abs=float(mean.detach().abs().mean().item()),
        var_mean=float(var.detach().mean().item()),
        var_std=float(var.detach().std(unbiased=False).item()) if var.numel() > 1 else 0.0,
        cov_offdiag=float(cov_loss.detach().item()),
        proj_var_mean=proj_var_mean,
    )
    return loss, metrics


def slot_latent_gaussian_regularizer(
    slots: torch.Tensor,
    *,
    projection_count: int = 64,
    cov_weight: float = 0.05,
) -> tuple[torch.Tensor, LatentRegularizerMetrics]:
    """Alias for slot/bank tensors; kept separate for clearer call sites."""

    return latent_gaussian_regularizer(
        slots,
        projection_count=projection_count,
        cov_weight=cov_weight,
    )
