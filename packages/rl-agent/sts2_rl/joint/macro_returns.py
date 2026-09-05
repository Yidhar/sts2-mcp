"""Detached long-horizon macro returns over factual mixed-domain sequences.

The physical discount is supplied by the durable-floor clock.  Lambda decays
only when an edge reaches the NEXT macro state, not at intervening combat
decisions.  Off-policy correction still uses every mixed-domain action ratio.
No model, learned Q head, replay store or optimizer is owned by this module.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Literal

import torch
from torch import Tensor

ControlDomain = Literal["macro", "combat"]


@dataclass(frozen=True, slots=True)
class MixedMacroReturnTargets:
    value_targets: Tensor
    # Conditional-action lambda targets: the action in EACH corresponding row
    # is intervened/forced. Its own rho/c factors are excluded; subsequent
    # mixed-domain actions remain corrected by their recorded behavior law.
    action_value_targets: Tensor
    log_importance_ratios: Tensor
    clipped_rhos: Tensor
    trace_coefficients: Tensor
    trace_lambdas: Tensor


@dataclass(frozen=True, slots=True)
class MacroSMDPEdges:
    rewards: Tensor
    discounts: Tensor
    start_indices: tuple[int, ...]
    stop_indices: tuple[int, ...]


def _lambda(value: float) -> float:
    if isinstance(value, bool) or not math.isfinite(value) or not 0.0 <= value <= 1.0:
        raise ValueError("trace_lambda must be finite and in [0, 1]")
    return float(value)


def _vector(value: Tensor, *, name: str, reference: Tensor | None = None) -> None:
    if not isinstance(value, Tensor) or value.ndim != 1 or not torch.is_floating_point(value):
        raise ValueError(f"{name} must be a rank-1 floating tensor")
    if value.numel() == 0 or not bool(torch.isfinite(value).all().item()):
        raise ValueError(f"{name} must be non-empty and finite")
    if reference is not None and (
        value.shape != reference.shape or value.dtype != reference.dtype or value.device != reference.device
    ):
        raise ValueError(f"{name} must match rewards shape, dtype and device")


def _scalar(value: Tensor, *, name: str, reference: Tensor) -> Tensor:
    if not isinstance(value, Tensor) or value.ndim != 0 or not torch.is_floating_point(value):
        raise ValueError(f"{name} must be a scalar floating tensor")
    if value.dtype != reference.dtype or value.device != reference.device:
        raise ValueError(f"{name} must match rewards dtype and device")
    if not bool(torch.isfinite(value).item()):
        raise ValueError(f"{name} must be finite")
    return value.detach()


def _facts(rewards: Tensor, discounts: Tensor, domains: Sequence[ControlDomain]) -> None:
    _vector(rewards, name="rewards")
    _vector(discounts, name="discounts", reference=rewards)
    if bool(((discounts < 0) | (discounts > 1)).any().item()):
        raise ValueError("discounts must be in [0, 1]")
    if bool((discounts[:-1] == 0).any().item()):
        raise ValueError("a sequence cannot continue after a terminal discount")
    if len(domains) != rewards.numel() or any(domain not in {"macro", "combat"} for domain in domains):
        raise ValueError("control_domains must specify macro/combat for every mixed decision")


@torch.no_grad()
def mixed_vtrace_lambda_targets(
    *,
    rewards: Tensor,
    discounts: Tensor,
    values: Tensor,
    target_log_probs: Tensor,
    behavior_log_probs: Tensor,
    control_domains: Sequence[ControlDomain],
    trace_lambda: float,
    bootstrap_value: Tensor | None = None,
    bootstrap_domain: ControlDomain | None = None,
    tail_value_target: Tensor | None = None,
    rho_clip: float = 1.0,
    c_clip: float = 1.0,
) -> MixedMacroReturnTargets:
    """Compute mixed V-trace with lambda only at macro-successor boundaries.

    ``values`` and ``target_log_probs`` must come from ONE frozen evaluation
    policy/version, with its recurrent prefix reconstructed under those same
    parameters. ``behavior_log_probs`` are exact collector probabilities,
    including any exploration mixture, for EVERY macro and combat action.

    A zero final discount is a factual terminal and forbids all bootstrap
    arguments. A non-terminal/horizon-censored tail requires the real next
    state's value AND domain. It must never be silently given zero bootstrap.

    For reverse chunking, supply the suffix's first ``value_targets`` element
    as ``tail_value_target``, while ``bootstrap_value`` remains that state's
    ordinary V prediction. This reproduces an unsplit return calculation.

    Let l_t=lambda if state t+1 is macro, else 1. The recursion is
      v_t = V_t + rho_t (r_t + d_t V_next - V_t)
                  + d_t l_t c_t (v_next - V_next).
    Conditional-action targets instead use
      y_t = r_t + d_t [V_next + l_t (v_next - V_next)].
    The latter removes ALL importance factors for the forced root action,
    without losing a lambda boundary when its next state is already macro.

    Clipping rho changes the evaluated policy (as in ordinary V-trace); this
    is not an unbiased Q^pi assertion. Log ratios are returned without an
    artificial floor. Clipped coefficients are formed in log space to avoid
    overflow even for a very small but correctly recorded behavior probability.
    """
    _facts(rewards, discounts, control_domains)
    lam = _lambda(trace_lambda)
    for name, value in (("values", values), ("target_log_probs", target_log_probs), ("behavior_log_probs", behavior_log_probs)):
        _vector(value, name=name, reference=rewards)
    if bool((target_log_probs > 0).any().item()) or bool((behavior_log_probs > 0).any().item()):
        raise ValueError("action log probabilities must be <= 0")
    if any(isinstance(value, bool) or not math.isfinite(value) or value <= 0 for value in (rho_clip, c_clip)):
        raise ValueError("rho_clip and c_clip must be positive and finite")
    if c_clip > rho_clip:
        raise ValueError("c_clip must not exceed rho_clip")

    terminal = float(discounts[-1].item()) == 0.0
    if terminal:
        if bootstrap_value is not None or bootstrap_domain is not None or tail_value_target is not None:
            raise ValueError("terminal sequences must not bootstrap")
        next_value = rewards.new_zeros(())
        next_target = next_value
    else:
        if bootstrap_value is None or bootstrap_domain not in {"macro", "combat"}:
            raise ValueError("non-terminal tails require bootstrap_value and bootstrap_domain")
        next_value = _scalar(bootstrap_value, name="bootstrap_value", reference=rewards)
        next_target = (
            next_value if tail_value_target is None
            else _scalar(tail_value_target, name="tail_value_target", reference=rewards)
        )

    log_ratios = (target_log_probs - behavior_log_probs).detach()
    if not bool(torch.isfinite(log_ratios).all().item()):
        raise ValueError("log importance ratios must be finite")
    clipped_rhos = torch.exp(log_ratios.clamp(max=math.log(rho_clip)))
    clipped_cs = torch.exp(log_ratios.clamp(max=math.log(c_clip)))
    successors = (*control_domains[1:], bootstrap_domain)
    trace_lambdas = rewards.new_tensor([lam if domain == "macro" else 1.0 for domain in successors])
    coefficients = trace_lambdas * clipped_cs
    targets = torch.empty_like(values)
    action_targets = torch.empty_like(values)
    for index in range(rewards.numel() - 1, -1, -1):
        continuation_correction = next_target - next_value
        action_targets[index] = rewards[index] + discounts[index] * (
            next_value + trace_lambdas[index] * continuation_correction
        )
        delta = clipped_rhos[index] * (rewards[index] + discounts[index] * next_value - values[index])
        targets[index] = values[index] + delta + discounts[index] * coefficients[index] * continuation_correction
        next_value = values[index]
        next_target = targets[index]
    if not bool(torch.isfinite(targets).all().item()) or not bool(torch.isfinite(action_targets).all().item()):
        raise FloatingPointError("macro return targets are non-finite")
    return MixedMacroReturnTargets(
        value_targets=targets.detach(),
        action_value_targets=action_targets.detach(),
        log_importance_ratios=log_ratios,
        clipped_rhos=clipped_rhos,
        trace_coefficients=coefficients,
        trace_lambdas=trace_lambdas,
    )


@torch.no_grad()
def fold_macro_smdp_edges(
    *, rewards: Tensor, discounts: Tensor, control_domains: Sequence[ControlDomain]
) -> MacroSMDPEdges:
    """Fold mixed edges once; NEVER apply a second duration/native-step discount.

    R_m=sum_t (prod_{u=start}^{t-1} d_u) r_t and Gamma_m=prod_t d_t.
    Leading combat before the first macro is not a macro edge. The final edge
    may end at a genuine terminal or a declared, bootstrappable horizon tail.
    """
    _facts(rewards, discounts, control_domains)
    starts = tuple(index for index, domain in enumerate(control_domains) if domain == "macro")
    stops = (*starts[1:], rewards.numel()) if starts else ()
    macro_rewards: list[Tensor] = []
    macro_discounts: list[Tensor] = []
    for start, stop in zip(starts, stops, strict=True):
        total = rewards.new_zeros(())
        weight = rewards.new_ones(())
        for index in range(start, stop):
            total = total + weight * rewards[index]
            weight = weight * discounts[index]
        if not bool(torch.isfinite(total).item()) or not bool(torch.isfinite(weight).item()):
            raise FloatingPointError("folded macro edges are non-finite")
        macro_rewards.append(total)
        macro_discounts.append(weight)
    return MacroSMDPEdges(
        rewards=torch.stack(macro_rewards) if starts else rewards.new_empty(0),
        discounts=torch.stack(macro_discounts) if starts else rewards.new_empty(0),
        start_indices=starts,
        stop_indices=tuple(stops),
    )


@torch.no_grad()
def on_policy_macro_lambda_returns(
    edges: MacroSMDPEdges,
    *,
    successor_values: Tensor,
    trace_lambda: float,
    bootstrap_domain: ControlDomain | None = None,
    tail_value_target: Tensor | None = None,
) -> Tensor:
    """Independent folded-edge oracle for the on-policy special case.

    Each successor_values[m] is V at the end of that macro edge (zero for a
    factual terminal). For a censored final edge, bootstrap_domain declares
    whether that real tail is a macro boundary. This is an on-policy oracle,
    NOT an off-policy replacement that drops intervening combat ratios.
    """
    lam = _lambda(trace_lambda)
    _vector(edges.rewards, name="macro rewards")
    _vector(edges.discounts, name="macro discounts", reference=edges.rewards)
    _vector(successor_values, name="successor_values", reference=edges.rewards)
    if bool(((edges.discounts < 0) | (edges.discounts > 1)).any().item()):
        raise ValueError("macro discounts must be in [0, 1]")
    if bool((edges.discounts[:-1] == 0).any().item()):
        raise ValueError("macro edges cannot continue after terminal")
    terminal = float(edges.discounts[-1].item()) == 0.0
    if terminal:
        if bootstrap_domain is not None or tail_value_target is not None or float(successor_values[-1].item()) != 0.0:
            raise ValueError("terminal macro edges must not bootstrap")
        next_target = edges.rewards.new_zeros(())
    else:
        if bootstrap_domain not in {"macro", "combat"}:
            raise ValueError("non-terminal macro edges require bootstrap_domain")
        next_target = (
            successor_values[-1] if tail_value_target is None
            else _scalar(tail_value_target, name="tail_value_target", reference=edges.rewards)
        )
    targets = torch.empty_like(edges.rewards)
    for index in range(edges.rewards.numel() - 1, -1, -1):
        edge_lambda = lam if index < edges.rewards.numel() - 1 or bootstrap_domain == "macro" else 1.0
        targets[index] = edges.rewards[index] + edges.discounts[index] * (
            successor_values[index] + edge_lambda * (next_target - successor_values[index])
        )
        next_target = targets[index]
    if not bool(torch.isfinite(targets).all().item()):
        raise FloatingPointError("folded macro lambda returns are non-finite")
    return targets.detach()


__all__ = [
    "ControlDomain",
    "MacroSMDPEdges",
    "MixedMacroReturnTargets",
    "fold_macro_smdp_edges",
    "mixed_vtrace_lambda_targets",
    "on_policy_macro_lambda_returns",
]
