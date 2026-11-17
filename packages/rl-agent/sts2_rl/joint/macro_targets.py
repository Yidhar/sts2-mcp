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
    # Optional unattenuated conditional target distributions (candidate_indices, cond_probs)
    # for multi-candidate deterministic branches under hierarchical MPO.
    within_branch_targets: tuple[tuple[tuple[int, ...], Tensor], ...] = ()


def _validate_inputs(
    action_values: Tensor,
    teacher_log_probs: Tensor,
    max_kl: float,
    min_temperature: float = 0.0,
) -> None:
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
    if isinstance(min_temperature, bool) or not math.isfinite(min_temperature) or min_temperature < 0:
        raise ValueError("min_temperature must be finite and non-negative")


def _distribution(log_prior: Tensor, centered_values: Tensor, inverse_temperature: float) -> tuple[Tensor, float]:
    log_posterior = torch.log_softmax(log_prior + centered_values * inverse_temperature, dim=0)
    posterior = log_posterior.exp()
    positive = posterior > 0
    divergence = (posterior[positive] * (log_posterior[positive] - log_prior[positive])).sum()
    return posterior, max(0.0, float(divergence.item()))


@torch.no_grad()
def project_macro_policy(
    *,
    action_value_targets: Tensor,
    teacher_log_probs: Tensor,
    max_kl: float,
    min_temperature: float = 0.0,
) -> MacroPolicyProjection:
    """Solve a per-root MPO-style KL-constrained discrete policy projection.

    q_eta(a) is proportional to pi_teacher(a) exp(Y(a)/eta), and eta >= min_temperature
    is chosen so KL(q_eta || pi_teacher) <= max_kl. When ``min_temperature > 0``, this
    solves ``max_q E_q[Y] - min_temperature * KL(q || pi_teacher)`` subject to
    ``KL(q || pi_teacher) <= max_kl``, damping sub-threshold critic noise proportionally
    instead of rescaling tiny noise spans up to the full KL boundary.
    """
    _validate_inputs(action_value_targets, teacher_log_probs, max_kl, min_temperature)
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
        centered = (values - values.max()) / span
        if min_temperature > 0.0:
            beta_cap = span / float(min_temperature)
            cap_posterior, cap_kl = _distribution(log_prior, centered, beta_cap)
            if cap_kl <= max_kl:
                posterior = cap_posterior
                divergence = cap_kl
                temperature = float(min_temperature)
            else:
                low = 0.0
                high = beta_cap
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
        else:
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


def _resolve_deterministic_subset(
    det_action_values: Tensor,
    teacher_log_probs: Tensor,
    deterministic_indices: tuple[int, ...] | None,
) -> tuple[int, ...]:
    if not isinstance(det_action_values, Tensor) or det_action_values.ndim != 1 or not torch.is_floating_point(det_action_values):
        raise ValueError("det_action_values must be a rank-1 floating tensor")
    if det_action_values.numel() < 2 or not bool(torch.isfinite(det_action_values).all().item()):
        raise ValueError("det_action_values must have >= 2 finite entries")
    if not isinstance(teacher_log_probs, Tensor) or teacher_log_probs.ndim != 1 or not torch.is_floating_point(teacher_log_probs):
        raise ValueError("teacher_log_probs must be a rank-1 floating tensor")
    if (
        det_action_values.dtype != teacher_log_probs.dtype
        or det_action_values.device != teacher_log_probs.device
    ):
        raise ValueError("det_action_values must match teacher_log_probs dtype and device")
    action_count = int(teacher_log_probs.numel())
    det_count = int(det_action_values.numel())
    if det_count > action_count:
        raise ValueError("det_action_values cannot exceed teacher_log_probs length")
    if bool((teacher_log_probs > 0).any().item()):
        raise ValueError("teacher_log_probs must be <= 0")
    tolerance = max(1e-6, 4 * torch.finfo(teacher_log_probs.dtype).eps)
    if abs(float(torch.logsumexp(teacher_log_probs.double(), dim=0).item())) > tolerance:
        raise ValueError("teacher_log_probs must describe the normalized full legal candidate distribution")
    if deterministic_indices is None:
        if det_count != action_count:
            raise ValueError("partial det_action_values requires explicit deterministic_indices")
        return tuple(range(action_count))
    if len(deterministic_indices) != det_count:
        raise ValueError("deterministic_indices length must match det_action_values")
    prev = -1
    for idx in deterministic_indices:
        if isinstance(idx, bool) or not isinstance(idx, int) or idx <= prev or idx >= action_count:
            raise ValueError("deterministic_indices must be strictly increasing valid candidate indices")
        prev = idx
    return tuple(deterministic_indices)


@torch.no_grad()
def project_subset_macro_policy(
    *,
    det_action_values: Tensor,
    teacher_log_probs: Tensor,
    deterministic_indices: tuple[int, ...] | None = None,
    max_kl: float,
    min_temperature: float = 0.0,
) -> MacroPolicyProjection:
    """Flat MPO projection restricted to verified deterministic candidates ``D``.

    Non-deterministic candidates ``j not in D`` preserve ``q(j) == pi_teacher(j)``
    identically while redistributing ``sum_{i in D} pi_teacher(i)`` within ``D``.
    """
    det_idx = _resolve_deterministic_subset(det_action_values, teacher_log_probs, deterministic_indices)
    if len(det_idx) == int(teacher_log_probs.numel()):
        return project_macro_policy(
            action_value_targets=det_action_values,
            teacher_log_probs=teacher_log_probs,
            max_kl=max_kl,
            min_temperature=min_temperature,
        )
    if isinstance(max_kl, bool) or not math.isfinite(max_kl) or max_kl < 0:
        raise ValueError("max_kl must be finite and non-negative")
    if isinstance(min_temperature, bool) or not math.isfinite(min_temperature) or min_temperature < 0:
        raise ValueError("min_temperature must be finite and non-negative")
    log_prior = teacher_log_probs.detach().double()
    log_prior = log_prior - torch.logsumexp(log_prior, dim=0)
    prior = log_prior.exp()
    det_idx_t = torch.tensor(list(det_idx), device=teacher_log_probs.device, dtype=torch.long)
    det_values = det_action_values.detach().double()
    log_p_det = float(torch.logsumexp(log_prior[det_idx_t], dim=0).item())
    p_det = math.exp(log_p_det) if math.isfinite(log_p_det) else 0.0
    teacher = prior.to(dtype=det_action_values.dtype).detach()
    if p_det <= 1e-12 or max_kl == 0.0:
        val_t = det_values.mean() if p_det <= 1e-12 else (prior[det_idx_t] * det_values).sum() / p_det
        return MacroPolicyProjection(
            target_probabilities=teacher.clone(),
            teacher_probabilities=teacher,
            value_target=val_t.to(dtype=det_action_values.dtype).detach(),
            temperature=None,
            kl=0.0,
            expected_improvement=0.0,
            actor_enabled=False,
        )
    cond_log_prior = (log_prior[det_idx_t] - log_p_det).clamp(max=0.0)
    cond_log_prior = (cond_log_prior - torch.logsumexp(cond_log_prior, dim=0)).clamp(max=0.0)
    # KL(q || pi) = P_det * KL(q_D || pi_D), so conditional budget max_kl / P_det bounds full KL <= max_kl.
    sub_proj = project_macro_policy(
        action_value_targets=det_values,
        teacher_log_probs=cond_log_prior,
        max_kl=max_kl / p_det,
        min_temperature=min_temperature,
    )
    if not sub_proj.actor_enabled:
        val_t = (prior[det_idx_t] * det_values).sum() / p_det
        return MacroPolicyProjection(
            target_probabilities=teacher.clone(),
            teacher_probabilities=teacher,
            value_target=val_t.to(dtype=det_action_values.dtype).detach(),
            temperature=None,
            kl=0.0,
            expected_improvement=0.0,
            actor_enabled=False,
        )
    posterior = prior.clone()
    det_post = sub_proj.target_probabilities.double()
    det_sum = float(det_post.sum().item())
    if det_sum > 0.0:
        det_post = det_post * (float(prior[det_idx_t].sum().item()) / det_sum)
    posterior[det_idx_t] = det_post
    pos = posterior > 0
    log_post = torch.where(pos, posterior.clamp_min(1e-300).log(), 0.0)
    divergence = max(0.0, float((posterior[pos] * (log_post[pos] - log_prior[pos])).sum().item()))
    improvement = float(((posterior[det_idx_t] - prior[det_idx_t]) * (det_values - det_values.max())).sum().item())
    target = posterior.to(dtype=det_action_values.dtype).detach()
    actor_enabled = improvement > 0.0 and not torch.equal(target, teacher)
    if not actor_enabled:
        target = teacher.clone()
        divergence = 0.0
        improvement = 0.0
    val_t = (prior[det_idx_t] * det_values).sum() / p_det
    return MacroPolicyProjection(
        target_probabilities=target,
        teacher_probabilities=teacher,
        value_target=val_t.to(dtype=det_action_values.dtype).detach(),
        temperature=sub_proj.temperature if actor_enabled else None,
        kl=divergence,
        expected_improvement=max(0.0, improvement),
        actor_enabled=actor_enabled,
    )


@torch.no_grad()
def project_hierarchical_macro_policy(
    *,
    det_action_values: Tensor,
    teacher_log_probs: Tensor,
    policy_branch_ids: Tensor,
    deterministic_indices: tuple[int, ...] | None = None,
    branch_max_kl: float,
    within_max_kl: float,
    min_temperature: float = 0.0,
    det_action_values_twin: Tensor | None = None,
    pessimism_coef: float = 0.0,
) -> MacroPolicyProjection:
    """Two-level hierarchical MPO matching ``branch_policy_head + within_log_probs``.

    1. Within each deterministic branch ``D_b`` of size >= 2, solves a conditional
       MPO projection ``pi*(. | D_b)`` under ``within_max_kl`` and ``min_temperature``,
       yielding post-improvement branch value ``Q_branch(b)`` (cross-evaluated across
       twin heads when ``det_action_values_twin`` is supplied to eliminate multi-card
       Winner's Curse maximization bias, minus ``pessimism_coef * |V1 - V2|``)
       plus unattenuated conditional targets ``within_branch_targets`` so minority
       branches (e.g. ``shop:remove``, ``rest:smith``) learn accurate target polarity
       even when current branch probability ``P(b)`` is small.
    2. Across deterministic branches ``b``, solves a branch-marginal MPO projection
       ``P_tilde*(b)`` under ``branch_max_kl`` and ``min_temperature`` using ``Q_branch(b)``,
       preserving ``q(j) == pi_teacher(j)`` for all non-deterministic ``j not in D``.
    """
    det_idx = _resolve_deterministic_subset(det_action_values, teacher_log_probs, deterministic_indices)
    for name, kl_val in (("branch_max_kl", branch_max_kl), ("within_max_kl", within_max_kl)):
        if isinstance(kl_val, bool) or not math.isfinite(kl_val) or kl_val < 0:
            raise ValueError(f"{name} must be finite and non-negative")
    if isinstance(min_temperature, bool) or not math.isfinite(min_temperature) or min_temperature < 0:
        raise ValueError("min_temperature must be finite and non-negative")
    if isinstance(pessimism_coef, bool) or not math.isfinite(pessimism_coef) or pessimism_coef < 0:
        raise ValueError("pessimism_coef must be finite and non-negative")
    if det_action_values_twin is not None:
        if not isinstance(det_action_values_twin, Tensor) or tuple(det_action_values_twin.shape) != tuple(
            det_action_values.shape
        ):
            raise ValueError("det_action_values_twin must match det_action_values shape")
        if not bool(torch.isfinite(det_action_values_twin).all().item()):
            raise FloatingPointError("det_action_values_twin must be finite")
    if not isinstance(policy_branch_ids, Tensor) or policy_branch_ids.ndim != 1:
        raise ValueError("policy_branch_ids must be a rank-1 tensor")
    if int(policy_branch_ids.numel()) < int(teacher_log_probs.numel()):
        raise ValueError("policy_branch_ids must cover all teacher_log_probs candidates")

    log_prior = teacher_log_probs.detach().double()
    log_prior = log_prior - torch.logsumexp(log_prior, dim=0)
    prior = log_prior.exp()
    det_idx_t = torch.tensor(list(det_idx), device=teacher_log_probs.device, dtype=torch.long)
    det_v1 = det_action_values.detach().double()
    if det_action_values_twin is not None:
        det_v2 = det_action_values_twin.detach().double()
        det_disagree = (det_v1 - det_v2).abs()
        det_mean = 0.5 * (det_v1 + det_v2)
        det_cons = det_mean - pessimism_coef * det_disagree
    else:
        det_v2 = None
        det_disagree = torch.zeros_like(det_v1)
        det_mean = det_v1
        det_cons = det_v1

    log_p_det = float(torch.logsumexp(log_prior[det_idx_t], dim=0).item())
    p_det = math.exp(log_p_det) if math.isfinite(log_p_det) else 0.0
    teacher = prior.to(dtype=det_action_values.dtype).detach()
    if p_det <= 1e-12 or (branch_max_kl == 0.0 and within_max_kl == 0.0):
        val_t = det_mean.mean() if p_det <= 1e-12 else (prior[det_idx_t] * det_mean).sum() / p_det
        return MacroPolicyProjection(
            target_probabilities=teacher.clone(),
            teacher_probabilities=teacher,
            value_target=val_t.to(dtype=det_action_values.dtype).detach(),
            temperature=None,
            kl=0.0,
            expected_improvement=0.0,
            actor_enabled=False,
        )

    branch_ids_cpu = policy_branch_ids[: teacher_log_probs.numel()].detach().cpu().tolist()
    branch_groups: dict[int, list[int]] = {}
    for local_u, cand_i in enumerate(det_idx):
        b_id = int(branch_ids_cpu[cand_i])
        branch_groups.setdefault(b_id, []).append(local_u)
    ordered_branches = sorted(branch_groups.keys())

    log_masses: list[float] = []
    q_branches: list[float] = []
    within_stars: list[Tensor] = []
    within_targets_out: list[tuple[tuple[int, ...], Tensor]] = []
    rep_temp: float | None = None
    det_eff_values = det_cons.clone()

    for b_id in ordered_branches:
        local_pos = branch_groups[b_id]
        full_pos = tuple(det_idx[u] for u in local_pos)
        full_pos_t = torch.tensor(list(full_pos), device=teacher_log_probs.device, dtype=torch.long)
        log_mass_b = float(torch.logsumexp(log_prior[full_pos_t], dim=0).item())
        log_masses.append(log_mass_b)
        vals_cons_b = det_cons[local_pos]
        if len(local_pos) == 1:
            within_stars.append(torch.ones((1,), device=teacher_log_probs.device, dtype=torch.float64))
            q_branches.append(float(vals_cons_b[0].item()))
        elif math.isfinite(log_mass_b):
            log_within_prior_b = (log_prior[full_pos_t] - log_mass_b).clamp(max=0.0)
            log_within_prior_b = (log_within_prior_b - torch.logsumexp(log_within_prior_b, dim=0)).clamp(max=0.0)
            within_proj = project_macro_policy(
                action_value_targets=vals_cons_b,
                teacher_log_probs=log_within_prior_b,
                max_kl=within_max_kl,
                min_temperature=min_temperature,
            )
            w_star = within_proj.target_probabilities.double()
            within_stars.append(w_star)
            if det_v2 is not None:
                v1_b = det_v1[local_pos]
                v2_b = det_v2[local_pos]
                dis_b = det_disagree[local_pos]
                proj_1 = project_macro_policy(
                    action_value_targets=v1_b,
                    teacher_log_probs=log_within_prior_b,
                    max_kl=within_max_kl,
                    min_temperature=min_temperature,
                )
                proj_2 = project_macro_policy(
                    action_value_targets=v2_b,
                    teacher_log_probs=log_within_prior_b,
                    max_kl=within_max_kl,
                    min_temperature=min_temperature,
                )
                w1 = proj_1.target_probabilities.double()
                w2 = proj_2.target_probabilities.double()
                q_cross = 0.5 * float(((w1 * v2_b).sum() + (w2 * v1_b).sum()).item())
                w_bar = 0.5 * (w1 + w2)
                q_b = q_cross - pessimism_coef * float((w_bar * dis_b).sum().item())
                delta_b = q_b - float((w_star * vals_cons_b).sum().item())
                det_eff_values[local_pos] = vals_cons_b + delta_b
                q_branches.append(q_b)
            else:
                q_branches.append(float((w_star * vals_cons_b).sum().item()))
            if within_proj.actor_enabled:
                within_targets_out.append((full_pos, w_star.to(dtype=det_action_values.dtype).detach()))
                if rep_temp is None and within_proj.temperature is not None:
                    rep_temp = within_proj.temperature
        else:
            unif = torch.full(
                (len(local_pos),),
                1.0 / len(local_pos),
                device=teacher_log_probs.device,
                dtype=torch.float64,
            )
            within_stars.append(unif)
            q_branches.append(float(vals_cons_b.mean().item()))

    if len(ordered_branches) >= 2 and branch_max_kl > 0.0:
        log_branch_prior = torch.tensor(
            [lm - log_p_det for lm in log_masses],
            device=teacher_log_probs.device,
            dtype=torch.float64,
        )
        log_branch_prior = (log_branch_prior - torch.logsumexp(log_branch_prior, dim=0)).clamp(max=0.0)
        q_branch_vec = torch.tensor(q_branches, device=teacher_log_probs.device, dtype=torch.float64)
        branch_proj = project_macro_policy(
            action_value_targets=q_branch_vec,
            teacher_log_probs=log_branch_prior,
            max_kl=branch_max_kl / p_det,
            min_temperature=min_temperature,
        )
        branch_star = branch_proj.target_probabilities.double()
        if branch_proj.temperature is not None:
            rep_temp = branch_proj.temperature
    elif len(ordered_branches) >= 2:
        log_branch_prior = torch.tensor(
            [lm - log_p_det for lm in log_masses],
            device=teacher_log_probs.device,
            dtype=torch.float64,
        )
        branch_star = torch.softmax(log_branch_prior, dim=0)
    else:
        branch_star = torch.ones((1,), device=teacher_log_probs.device, dtype=torch.float64)

    posterior = prior.clone()
    det_post = torch.zeros((len(det_idx),), device=teacher_log_probs.device, dtype=torch.float64)
    for b_idx, b_id in enumerate(ordered_branches):
        local_pos = branch_groups[b_id]
        det_post[local_pos] = branch_star[b_idx] * within_stars[b_idx]
    det_sum = float(det_post.sum().item())
    if det_sum > 0.0:
        det_post = det_post * (float(prior[det_idx_t].sum().item()) / det_sum)
    posterior[det_idx_t] = det_post

    pos = posterior > 0
    log_post = torch.where(pos, posterior.clamp_min(1e-300).log(), 0.0)
    divergence = max(0.0, float((posterior[pos] * (log_post[pos] - log_prior[pos])).sum().item()))
    improvement = float(
        ((posterior[det_idx_t] - prior[det_idx_t]) * (det_eff_values - det_eff_values.max())).sum().item()
    )
    target = posterior.to(dtype=det_action_values.dtype).detach()
    joint_changed = improvement > 0.0 and not torch.equal(target, teacher)
    actor_enabled = joint_changed or bool(within_targets_out)
    if not joint_changed:
        target = teacher.clone()
        divergence = 0.0
        improvement = 0.0
        if not actor_enabled:
            rep_temp = None
    if not bool(torch.isfinite(target).all().item()):
        raise FloatingPointError("macro policy probabilities are non-finite")
    val_t = (prior[det_idx_t] * det_mean).sum() / p_det
    return MacroPolicyProjection(
        target_probabilities=target,
        teacher_probabilities=teacher,
        value_target=val_t.to(dtype=det_action_values.dtype).detach(),
        temperature=rep_temp,
        kl=divergence,
        expected_improvement=max(0.0, improvement),
        actor_enabled=actor_enabled,
        within_branch_targets=tuple(within_targets_out),
    )


def project_macro_policy_from_shadow_price(
    *,
    shadow_action_values: Tensor,
    teacher_log_probs: Tensor,
    max_kl: float,
) -> MacroPolicyProjection:
    """Project macro policy directly using combat resource shadow price targets."""
    return project_macro_policy(
        action_value_targets=shadow_action_values,
        teacher_log_probs=teacher_log_probs,
        max_kl=max_kl,
    )


__all__ = [
    "MacroPolicyProjection",
    "project_hierarchical_macro_policy",
    "project_macro_policy",
    "project_macro_policy_from_shadow_price",
    "project_subset_macro_policy",
]
