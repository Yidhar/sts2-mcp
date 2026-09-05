"""Prior-preserving, full-root policy improvement without a learned Q head.

Every input row is one legal semantic candidate of the SAME reconstructed
root. Rollout replicas must first be averaged per candidate by the caller.
Do not pass only a sampled/completed subset and renormalize it as a full root.
The caller averages the resulting actor losses over roots, not candidates or
replicas. Branch multiplicities are deliberately not part of the policy law.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import torch
from torch import Tensor


@dataclass(frozen=True, slots=True)
class MacroPolicyProjection:
    target_probabilities: Tensor
    teacher_probabilities: Tensor
    # Critic evaluates the frozen TEACHER, not its improved posterior q.
    value_target: Tensor
    # None denotes the unchanged prior (infinite temperature). Zero denotes
    # its conditional distribution on maximizing actions, when KL permits.
    temperature: float | None
    kl: float
    expected_improvement: float
    actor_enabled: bool


def _validate_inputs(action_values: Tensor, teacher_log_probs: Tensor, max_kl: float) -> None:
    for name, value in (("action_value_targets", action_values), ("teacher_log_probs", teacher_log_probs)):
        if not isinstance(value, Tensor) or value.ndim != 1 or not torch.is_floating_point(value):
            raise ValueError(f"{name} must be a rank-1 floating tensor")
        if value.numel() == 0 or not bool(torch.isfinite(value).all().item()):
            raise ValueError(f"{name} must be non-empty and finite")
    if (
        action_values.shape != teacher_log_probs.shape
        or action_values.dtype != teacher_log_probs.dtype
        or action_values.device != teacher_log_probs.device
    ):
        raise ValueError("teacher_log_probs must match action_value_targets shape, dtype and device")
    if bool((teacher_log_probs > 0).any().item()):
        raise ValueError("teacher_log_probs must be <= 0")
    tolerance = max(1e-6, 4 * torch.finfo(teacher_log_probs.dtype).eps)
    if abs(float(torch.logsumexp(teacher_log_probs.double(), dim=0).item())) > tolerance:
        raise ValueError("teacher_log_probs must describe the normalized full legal candidate distribution")
    if isinstance(max_kl, bool) or not math.isfinite(max_kl) or max_kl < 0:
        raise ValueError("max_kl must be finite and non-negative")


def _distribution(log_prior: Tensor, centered_values: Tensor, inverse_temperature: float) -> tuple[Tensor, float]:
    log_posterior = torch.log_softmax(log_prior + centered_values * inverse_temperature, dim=0)
    posterior = log_posterior.exp()
    positive = posterior > 0
    divergence = (posterior[positive] * (log_posterior[positive] - log_prior[positive])).sum()
    return posterior, max(0.0, float(divergence.item()))


@torch.no_grad()
def project_macro_policy(
    *, action_value_targets: Tensor, teacher_log_probs: Tensor, max_kl: float
) -> MacroPolicyProjection:
    """Solve a per-root MPO-style KL-constrained discrete policy projection.

    q_eta(a) is proportional to pi_teacher(a) exp(Y(a)/eta), and eta is chosen
    so KL(q_eta || pi_teacher) <= max_kl. Y is a detached conditional-action
    lambda-return estimate, not a learned IQL Q prediction. This helper does
    not claim an unbiased return estimator or a guaranteed improvement of the
    environment policy after neural-network fitting.

    Both teacher probabilities and all candidate outcomes must refer to one
    frozen stochastic continuation policy/version. Exact greedy diagnostic
    continuations are not training labels for this stochastic teacher.

    Equal Y keeps the prior and DISABLES the actor term rather than applying
    a cloning CE. No branch-count correction is inserted into q: doing that
    would change the existing flat policy even when outcomes are all equal.
    Root value supervision is sum_a pi_teacher(a) Y(a), never sum_a q(a) Y(a).

    The dual is solved in float64 after centering and scaling Y. Outputs keep
    the input device/dtype and are detached. Finite log probabilities provide
    full support without introducing an artificial probability floor; a tiny
    teacher prior can nevertheless limit practical recovery and must be
    monitored by the caller. Incomplete roots, insufficient replicas, policy
    age and confidence gates are the caller's explicit data-contract concerns.
    """
    _validate_inputs(action_value_targets, teacher_log_probs, max_kl)
    values = action_value_targets.detach().double()
    log_prior = teacher_log_probs.detach().double()
    log_prior = log_prior - torch.logsumexp(log_prior, dim=0)
    prior = log_prior.exp()
    value_target = (prior * values).sum()
    span = float((values.max() - values.min()).item())
    if not math.isfinite(span) or not bool(torch.isfinite(value_target).item()):
        raise FloatingPointError("macro policy projection values exceed finite numerical range")

    posterior = prior
    divergence = 0.0
    temperature: float | None = None
    if span > 0.0 and max_kl > 0.0:
        maximizing = values == values.max()
        greedy_kl = -float(torch.logsumexp(log_prior[maximizing], dim=0).item())
        if greedy_kl <= max_kl:
            posterior = torch.zeros_like(prior)
            posterior[maximizing] = torch.softmax(log_prior[maximizing], dim=0)
            divergence = max(0.0, greedy_kl)
            temperature = 0.0
        else:
            # beta = span / eta: centering prevents exp overflow and scaling
            # makes the solution invariant to reward translation/units.
            centered = (values - values.max()) / span
            low = 0.0
            high = 1.0
            for _ in range(128):
                _, high_kl = _distribution(log_prior, centered, high)
                if high_kl >= max_kl:
                    break
                high *= 2.0
            else:
                raise FloatingPointError("could not bracket macro policy KL dual")
            for _ in range(80):
                middle = (low + high) / 2.0
                candidate, candidate_kl = _distribution(log_prior, centered, middle)
                if candidate_kl <= max_kl:
                    low = middle
                    posterior = candidate
                    divergence = candidate_kl
                else:
                    high = middle
            if low > 0.0:
                temperature = span / low
                if not math.isfinite(temperature):
                    raise FloatingPointError("macro policy temperature is non-finite")

    improvement = float(((posterior - prior) * (values - values.max())).sum().item())
    # Only a positive outcome-directed update is eligible. In particular,
    # all-equal outcomes and a zero KL budget do not become cloning losses.
    target = posterior.to(dtype=action_value_targets.dtype).detach()
    teacher = prior.to(dtype=action_value_targets.dtype).detach()
    actor_enabled = improvement > 0.0 and not torch.equal(target, teacher)
    if not actor_enabled:
        target = teacher.clone()
        divergence = 0.0
        improvement = 0.0
        temperature = None
    if not bool(torch.isfinite(target).all().item()):
        raise FloatingPointError("macro policy probabilities are non-finite")
    return MacroPolicyProjection(
        target_probabilities=target,
        teacher_probabilities=teacher,
        value_target=value_target.to(dtype=action_value_targets.dtype).detach(),
        temperature=temperature,
        kl=divergence,
        expected_improvement=max(0.0, improvement),
        actor_enabled=actor_enabled,
    )


__all__ = ["MacroPolicyProjection", "project_macro_policy"]
