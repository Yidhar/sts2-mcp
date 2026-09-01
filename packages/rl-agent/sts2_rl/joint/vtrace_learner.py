"""Fresh FIFO recurrent V-trace learner for the hybrid semantic policy.

Every unroll is consumed once.  V-trace corrects the exact mixture between
the model policy and branch-balanced exploration recorded by the collector.
Macro and combat rows share one return recursion, then their policy/value/
entropy losses are normalized independently and averaged 1:1 so sparse macro
decisions are not numerically buried under combat traffic.
"""

from __future__ import annotations

import math
from collections.abc import Callable
from dataclasses import asdict, dataclass
from typing import Any, Final, TypeAlias

import torch
from torch import Tensor, nn
from torch.nn import functional as F

from .transitions import HybridVTraceStep, HybridVTraceUnroll

HYBRID_VTRACE_LEARNER_VERSION: Final = "sts2-hybrid-recurrent-vtrace-v1"
HybridActorCriticForward: TypeAlias = tuple[Tensor, Tensor, Tensor]


@dataclass(frozen=True, slots=True)
class HybridVTraceLearnerConfig:
    learning_rate: float = 1.0e-4
    shared_learning_rate_scale: float = 0.5
    policy_weight: float = 1.0
    value_weight: float = 0.5
    entropy_weight: float = 0.02
    entropy_weight_end: float = 0.004
    entropy_decay_updates: int = 2_000
    vtrace_rho_clip: float = 1.0
    vtrace_c_clip: float = 1.0
    policy_rho_clip: float = 1.0
    gradient_clip_norm: float = 1.0
    gradient_diagnostics_interval: int = 50

    def __post_init__(self) -> None:
        for name in ("entropy_decay_updates", "gradient_diagnostics_interval"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
                raise ValueError(f"hybrid V-trace {name} must be positive")
        for name in (
            "learning_rate",
            "shared_learning_rate_scale",
            "policy_weight",
            "value_weight",
            "entropy_weight",
            "vtrace_rho_clip",
            "vtrace_c_clip",
            "policy_rho_clip",
            "gradient_clip_norm",
        ):
            value = float(getattr(self, name))
            if not math.isfinite(value) or value <= 0.0:
                raise ValueError(f"hybrid V-trace {name} must be positive and finite")
        if not math.isfinite(self.entropy_weight_end) or self.entropy_weight_end < 0.0:
            raise ValueError("hybrid V-trace entropy_weight_end must be finite and non-negative")
        if self.entropy_weight_end > self.entropy_weight:
            raise ValueError("hybrid V-trace entropy schedule cannot rise")


@dataclass(slots=True)
class HybridVTraceMetrics:
    updates: int = 0
    unrolls: int = 0
    decisions: int = 0
    macro_steps: int = 0
    combat_steps: int = 0
    loss: float = 0.0
    macro_loss: float = 0.0
    combat_loss: float = 0.0
    macro_policy_loss: float = 0.0
    combat_policy_loss: float = 0.0
    macro_value_loss: float = 0.0
    combat_value_loss: float = 0.0
    macro_entropy: float = 0.0
    combat_entropy: float = 0.0
    entropy_weight: float = 0.0
    mean_importance_ratio: float = 0.0
    clipped_importance_fraction: float = 0.0
    mean_policy_lag: float = 0.0
    maximum_policy_lag: int = 0
    gradient_norm: float = 0.0
    macro_shared_gradient_norm: float = 0.0
    combat_shared_gradient_norm: float = 0.0
    shared_gradient_cosine: float = 0.0

    def as_mapping(self) -> dict[str, Any]:
        return {"version": HYBRID_VTRACE_LEARNER_VERSION, **asdict(self)}


@dataclass(frozen=True, slots=True)
class _DomainTerms:
    total: Tensor
    policy: Tensor
    value: Tensor
    entropy: Tensor
    steps: int


class HybridVTraceLearner:
    """One dense recurrent actor-critic objective over mixed semantic data."""

    def __init__(
        self,
        *,
        model: nn.Module,
        forward: Callable[[Any, Tensor | None], HybridActorCriticForward],
        config: HybridVTraceLearnerConfig | None = None,
    ) -> None:
        self.model = model
        self.forward = forward
        self.config = config or HybridVTraceLearnerConfig()
        local_tokens = (
            "macro_stream",
            "combat_stream",
            "macro_stream_norm",
            "combat_stream_norm",
            "candidate_policy_head",
            "state_value_head",
        )
        local: list[nn.Parameter] = []
        shared: list[nn.Parameter] = []
        for name, parameter in model.named_parameters():
            (local if name.startswith(local_tokens) else shared).append(parameter)
        groups: list[dict[str, Any]] = []
        if local:
            groups.append({"params": local, "lr": self.config.learning_rate})
        if shared:
            groups.append(
                {
                    "params": shared,
                    "lr": self.config.learning_rate * self.config.shared_learning_rate_scale,
                }
            )
        self.optimizer = torch.optim.AdamW(groups)
        self._parameters = [*local, *shared]
        self._shared_parameters = shared
        self.metrics = HybridVTraceMetrics()
        # Collection also disables dropout.  Keeping the learner in eval mode
        # makes current-policy likelihoods comparable to recorded behavior
        # while retaining ordinary autograd.
        self.model.eval()

    @property
    def policy_version(self) -> int:
        return self.metrics.updates

    def entropy_weight(self) -> float:
        progress = min(self.metrics.updates / self.config.entropy_decay_updates, 1.0)
        return float(
            self.config.entropy_weight
            + (self.config.entropy_weight_end - self.config.entropy_weight) * progress
        )

    @staticmethod
    def _selected_policy(
        step: HybridVTraceStep,
        logits: Tensor,
    ) -> tuple[Tensor, Tensor, bool]:
        count = len(step.snapshot.action_mask)
        legal = torch.tensor(
            step.snapshot.action_mask,
            dtype=torch.bool,
            device=logits.device,
        )
        legal_logits = logits[:count].masked_fill(~legal, float("-inf"))
        log_policy = torch.log_softmax(legal_logits, dim=0)
        probabilities = log_policy.exp()
        entropy = -(probabilities[legal] * log_policy[legal]).sum()
        return log_policy[step.action_index], entropy, bool(int(legal.sum().item()) > 1)

    def _unroll_terms(
        self,
        unroll: HybridVTraceUnroll,
    ) -> tuple[list[tuple[str, Tensor, Tensor, Tensor, bool]], Tensor, Tensor]:
        parameter = next(self.model.parameters())
        hidden = torch.tensor(
            unroll.initial_recurrent_state,
            dtype=parameter.dtype,
            device=parameter.device,
        ).unsqueeze(0)
        log_probabilities: list[Tensor] = []
        values: list[Tensor] = []
        entropies: list[Tensor] = []
        policy_decisions: list[bool] = []
        for step in unroll.steps:
            logits, value, hidden = self.forward(step.snapshot, hidden)
            selected, entropy, policy_decision = self._selected_policy(step, logits)
            log_probabilities.append(selected)
            values.append(value.reshape(()))
            entropies.append(entropy)
            policy_decisions.append(policy_decision)
        if unroll.bootstrap_snapshot is None:
            bootstrap = values[-1].new_zeros(())
        else:
            with torch.no_grad():
                _, bootstrap_value, _ = self.forward(unroll.bootstrap_snapshot, hidden.detach())
            bootstrap = bootstrap_value.reshape(()).detach()

        log_probs = torch.stack(log_probabilities)
        baselines = torch.stack(values)
        behavior = baselines.new_tensor(
            [step.behavior_log_probability for step in unroll.steps]
        )
        rewards = baselines.new_tensor([step.reward for step in unroll.steps])
        discounts = baselines.new_tensor([step.discount for step in unroll.steps])
        ratios = torch.exp((log_probs - behavior).clamp(-20.0, 20.0))
        rho = ratios.clamp(max=self.config.vtrace_rho_clip)
        trace_c = ratios.clamp(max=self.config.vtrace_c_clip)
        policy_rho = ratios.clamp(max=self.config.policy_rho_clip)

        targets = torch.zeros_like(baselines)
        next_targets = torch.zeros_like(baselines)
        next_baseline = bootstrap
        next_vtrace = bootstrap
        for index in range(len(unroll.steps) - 1, -1, -1):
            next_targets[index] = next_vtrace
            delta = rho[index] * (
                rewards[index] + discounts[index] * next_baseline - baselines[index]
            )
            target = (
                baselines[index]
                + delta
                + discounts[index] * trace_c[index] * (next_vtrace - next_baseline)
            )
            targets[index] = target
            next_baseline = baselines[index]
            next_vtrace = target
        advantages = policy_rho * (
            rewards + discounts * next_targets - baselines
        )

        rows: list[tuple[str, Tensor, Tensor, Tensor, bool]] = []
        for index, step in enumerate(unroll.steps):
            rows.append(
                (
                    step.control_domain,
                    -log_probs[index] * advantages[index].detach(),
                    F.smooth_l1_loss(baselines[index], targets[index].detach()),
                    entropies[index],
                    policy_decisions[index],
                )
            )
        return rows, ratios.detach(), targets.detach()

    def _domain_terms(
        self,
        rows: list[tuple[str, Tensor, Tensor, Tensor, bool]],
        domain: str,
        *,
        entropy_weight: float,
    ) -> _DomainTerms:
        selected = [row for row in rows if row[0] == domain]
        if not selected:
            raise ValueError(f"hybrid V-trace batch has no {domain} decisions")
        value = torch.stack([row[2] for row in selected]).mean()
        policy_rows = [row for row in selected if row[4]]
        if policy_rows:
            policy = torch.stack([row[1] for row in policy_rows]).mean()
            entropy = torch.stack([row[3] for row in policy_rows]).mean()
        else:
            policy = value.new_zeros(())
            entropy = value.new_zeros(())
        total = (
            self.config.policy_weight * policy
            + self.config.value_weight * value
            - entropy_weight * entropy
        )
        return _DomainTerms(
            total=total,
            policy=policy,
            value=value,
            entropy=entropy,
            steps=len(selected),
        )

    @staticmethod
    def _gradient_vector(loss: Tensor, parameters: list[nn.Parameter]) -> Tensor:
        gradients = torch.autograd.grad(
            loss,
            parameters,
            retain_graph=True,
            allow_unused=True,
        )
        pieces = [
            torch.zeros_like(parameter).reshape(-1)
            if gradient is None
            else gradient.detach().reshape(-1)
            for parameter, gradient in zip(parameters, gradients, strict=True)
        ]
        return torch.cat(pieces) if pieces else loss.new_zeros(1)

    def update(self, unrolls: tuple[HybridVTraceUnroll, ...]) -> dict[str, Any]:
        if not unrolls:
            return self.as_mapping()
        rows: list[tuple[str, Tensor, Tensor, Tensor, bool]] = []
        ratios: list[Tensor] = []
        for unroll in unrolls:
            unroll_rows, unroll_ratios, _ = self._unroll_terms(unroll)
            rows.extend(unroll_rows)
            ratios.append(unroll_ratios)
        entropy_weight = self.entropy_weight()
        macro = self._domain_terms(rows, "macro", entropy_weight=entropy_weight)
        combat = self._domain_terms(rows, "combat", entropy_weight=entropy_weight)
        loss = 0.5 * (macro.total + combat.total)
        if not bool(torch.isfinite(loss).item()):
            raise FloatingPointError("hybrid V-trace loss is non-finite")

        self.optimizer.zero_grad(set_to_none=True)
        if (
            self.metrics.updates % self.config.gradient_diagnostics_interval == 0
            and self._shared_parameters
        ):
            macro_gradient = self._gradient_vector(macro.total, self._shared_parameters)
            combat_gradient = self._gradient_vector(combat.total, self._shared_parameters)
            macro_norm = torch.linalg.vector_norm(macro_gradient)
            combat_norm = torch.linalg.vector_norm(combat_gradient)
            denominator = macro_norm * combat_norm
            cosine = (
                torch.dot(macro_gradient, combat_gradient) / denominator
                if float(denominator.item()) > 0.0
                else denominator.new_zeros(())
            )
            self.metrics.macro_shared_gradient_norm = float(macro_norm.item())
            self.metrics.combat_shared_gradient_norm = float(combat_norm.item())
            self.metrics.shared_gradient_cosine = float(cosine.item())
        loss.backward()  # type: ignore[no-untyped-call]
        gradient_norm = torch.nn.utils.clip_grad_norm_(
            self._parameters,
            self.config.gradient_clip_norm,
        )
        if not bool(torch.isfinite(gradient_norm).item()):
            raise FloatingPointError("hybrid V-trace gradient norm is non-finite")
        self.optimizer.step()

        ratio_values = torch.cat(ratios)
        policy_lags = [
            max(0, self.metrics.updates - unroll.policy_version) for unroll in unrolls
        ]
        self.metrics.updates += 1
        self.metrics.unrolls += len(unrolls)
        self.metrics.decisions += len(rows)
        self.metrics.macro_steps += macro.steps
        self.metrics.combat_steps += combat.steps
        self.metrics.loss = float(loss.detach().item())
        self.metrics.macro_loss = float(macro.total.detach().item())
        self.metrics.combat_loss = float(combat.total.detach().item())
        self.metrics.macro_policy_loss = float(macro.policy.detach().item())
        self.metrics.combat_policy_loss = float(combat.policy.detach().item())
        self.metrics.macro_value_loss = float(macro.value.detach().item())
        self.metrics.combat_value_loss = float(combat.value.detach().item())
        self.metrics.macro_entropy = float(macro.entropy.detach().item())
        self.metrics.combat_entropy = float(combat.entropy.detach().item())
        self.metrics.entropy_weight = entropy_weight
        self.metrics.mean_importance_ratio = float(ratio_values.mean().item())
        self.metrics.clipped_importance_fraction = float(
            (ratio_values > self.config.vtrace_rho_clip).float().mean().item()
        )
        self.metrics.mean_policy_lag = sum(policy_lags) / len(policy_lags)
        self.metrics.maximum_policy_lag = max(policy_lags)
        self.metrics.gradient_norm = float(gradient_norm)
        return self.as_mapping()

    def as_mapping(self) -> dict[str, Any]:
        return self.metrics.as_mapping()

    def state_dict(self) -> dict[str, Any]:
        return {
            "version": HYBRID_VTRACE_LEARNER_VERSION,
            "optimizer": self.optimizer.state_dict(),
            "metrics": asdict(self.metrics),
        }

    def load_state_dict(self, state: dict[str, Any]) -> None:
        if state.get("version") != HYBRID_VTRACE_LEARNER_VERSION:
            raise ValueError("hybrid V-trace learner state version differs")
        self.optimizer.load_state_dict(state["optimizer"])
        metrics = state.get("metrics")
        if not isinstance(metrics, dict):
            raise ValueError("hybrid V-trace learner state has no metrics")
        self.metrics = HybridVTraceMetrics(**metrics)


__all__ = [
    "HYBRID_VTRACE_LEARNER_VERSION",
    "HybridActorCriticForward",
    "HybridVTraceLearner",
    "HybridVTraceLearnerConfig",
    "HybridVTraceMetrics",
]
