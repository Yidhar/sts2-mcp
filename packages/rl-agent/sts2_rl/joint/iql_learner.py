"""Balanced recurrent IQL/AWAC learner for the mixed semantic graph.

This is the predefined successor to the hybrid Double-Q controller after the
unsupported-candidate-max failure was observed in production.  Q bootstraps
from a candidate-independent target value; no learner target contains an
argmax over legal candidates.  The actor is fitted only to factual executed
actions with an advantage-weighted cross-entropy derivative that remains
non-zero even when its current probability is tiny.
"""

from __future__ import annotations

import math
from collections.abc import Callable
from dataclasses import asdict, dataclass, field
from typing import Any, Final, TypeAlias

import torch
from torch import Tensor, nn

from .death_price import (
    AbsoluteDeathPriceController,
    DeathPriceEpisodeOutcome,
    death_price_course,
    validate_discounted_outcome_margin,
)
from .replay import DomainView, HybridRunReplay
from .transitions import HybridRun, HybridStep, next_domain_index

HYBRID_IQL_LEARNER_VERSION: Final = "sts2-hybrid-iql-awac-v1"
HybridIQLForward: TypeAlias = tuple[Tensor, Tensor, Tensor, Tensor]


@dataclass(frozen=True, slots=True)
class HybridIQLLearnerConfig:
    learning_rate: float = 1.0e-4
    shared_learning_rate_scale: float = 0.5
    target_update_interval: int = 25
    huber_delta: float = 1.0
    gradient_clip_norm: float = 1.0
    run_batch_size: int = 4
    combat_n_step: int = 8
    gradient_diagnostics_interval: int = 50
    expectile: float = 0.7
    advantage_temperature: float = 0.5
    maximum_advantage_weight: float = 20.0
    revival_budget: int = 64

    def __post_init__(self) -> None:
        if (
            min(
                self.target_update_interval,
                self.run_batch_size,
                self.combat_n_step,
                self.gradient_diagnostics_interval,
            )
            <= 0
        ):
            raise ValueError("hybrid IQL integer bounds must be positive")
        for name in (
            "learning_rate",
            "shared_learning_rate_scale",
            "huber_delta",
            "gradient_clip_norm",
            "advantage_temperature",
            "maximum_advantage_weight",
        ):
            value = float(getattr(self, name))
            if not math.isfinite(value) or value <= 0.0:
                raise ValueError(f"hybrid IQL {name} must be positive and finite")
        if not math.isfinite(self.expectile) or not 0.5 < self.expectile < 1.0:
            raise ValueError("hybrid IQL expectile must be in (0.5, 1.0)")
        death_price_course(self.revival_budget)


@dataclass(slots=True)
class HybridIQLMetrics:
    updates: int = 0
    macro_steps: int = 0
    combat_steps: int = 0
    macro_loss: float = 0.0
    combat_loss: float = 0.0
    loss: float = 0.0
    macro_q_loss: float = 0.0
    combat_q_loss: float = 0.0
    macro_value_loss: float = 0.0
    combat_value_loss: float = 0.0
    macro_actor_loss: float = 0.0
    combat_actor_loss: float = 0.0
    macro_td_abs: float = 0.0
    combat_td_abs: float = 0.0
    mean_advantage_weight: float = 0.0
    maximum_advantage_weight: float = 0.0
    mean_policy_entropy: float = 0.0
    critic_gradient_norm: float = 0.0
    actor_gradient_norm: float = 0.0
    gradient_norm: float = 0.0
    macro_shared_gradient_norm: float = 0.0
    combat_shared_gradient_norm: float = 0.0
    shared_gradient_cosine: float = 0.0
    mean_policy_lag: float = 0.0
    maximum_policy_lag: int = 0
    target_syncs: int = 0
    last_target_sync_update: int = 0
    domain_surface_counts: dict[str, int] = field(default_factory=dict)

    def as_mapping(self) -> dict[str, Any]:
        return {"version": HYBRID_IQL_LEARNER_VERSION, **asdict(self)}


@dataclass(frozen=True, slots=True)
class _DomainLoss:
    total: Tensor
    q: Tensor
    value: Tensor
    actor: Tensor
    td_abs: Tensor
    weights: Tensor
    entropies: Tensor
    steps: int


class HybridIQLLearner:
    """One IQL/AWAC algorithm over separately reduced macro/combat views."""

    def __init__(
        self,
        *,
        online_model: nn.Module,
        target_model: nn.Module,
        replay: HybridRunReplay,
        forward_online: Callable[[HybridStep, Tensor | None], HybridIQLForward],
        forward_target: Callable[[HybridStep, Tensor | None], HybridIQLForward],
        initial_state: Callable[[], Tensor],
        config: HybridIQLLearnerConfig | None = None,
    ) -> None:
        self.online_model = online_model
        self.target_model = target_model
        self.replay = replay
        self.forward_online = forward_online
        self.forward_target = forward_target
        self.initial_state = initial_state
        self.config = config or HybridIQLLearnerConfig()
        course = death_price_course(self.config.revival_budget)
        validate_discounted_outcome_margin(course)
        self.death_price = AbsoluteDeathPriceController(course)

        local_tokens = (
            "macro_stream",
            "combat_stream",
            "macro_stream_norm",
            "combat_stream_norm",
            "candidate_q_head",
            "candidate_policy_head",
            "state_value_head",
        )
        shared: list[nn.Parameter] = []
        local: list[nn.Parameter] = []
        actor: list[nn.Parameter] = []
        critic: list[nn.Parameter] = []
        for name, parameter in online_model.named_parameters():
            (local if name.startswith(local_tokens) else shared).append(parameter)
            (actor if name.startswith("candidate_policy_head") else critic).append(parameter)
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
        self._critic_parameters = critic
        self._actor_parameters = actor
        self._shared_parameters = shared
        self.metrics = HybridIQLMetrics()
        self.online_model.eval()
        self.target_model.eval()
        for parameter in self.target_model.parameters():
            parameter.requires_grad_(False)

    @staticmethod
    def _gradient_vector(loss: Tensor, parameters: list[nn.Parameter]) -> Tensor:
        gradients = torch.autograd.grad(
            loss,
            parameters,
            retain_graph=True,
            allow_unused=True,
        )
        parts = [
            (
                torch.zeros_like(parameter).reshape(-1)
                if gradient is None
                else gradient.detach().reshape(-1)
            )
            for parameter, gradient in zip(parameters, gradients, strict=True)
        ]
        return torch.cat(parts) if parts else loss.new_zeros(1)

    def _prefix_states(self, view: DomainView) -> tuple[Tensor, Tensor]:
        online_state = self.initial_state()
        target_state = self.initial_state()
        with torch.no_grad():
            for index in view.prefix_indices:
                *_, online_state = self.forward_online(view.run.steps[index], online_state)
                *_, target_state = self.forward_target(view.run.steps[index], target_state)
        return online_state.detach(), target_state.detach()

    def _state_before(self, run: HybridRun, index: int) -> tuple[Tensor, Tensor]:
        macro_prefix = tuple(candidate for candidate in run.indices("macro") if candidate < index)
        if run.steps[index].control_domain == "combat":
            encounter = run.steps[index].encounter_id
            combat_prefix = tuple(
                candidate
                for candidate in run.indices("combat")
                if candidate < index and run.steps[candidate].encounter_id == encounter
            )
            prefix = tuple(sorted((*macro_prefix, *combat_prefix)))
        else:
            prefix = macro_prefix
        online_state = self.initial_state()
        target_state = self.initial_state()
        with torch.no_grad():
            for prefix_index in prefix:
                *_, online_state = self.forward_online(run.steps[prefix_index], online_state)
                *_, target_state = self.forward_target(run.steps[prefix_index], target_state)
        return online_state.detach(), target_state.detach()

    def _states_and_outputs(
        self,
        view: DomainView,
    ) -> tuple[
        dict[int, Tensor],
        dict[int, Tensor],
        dict[int, Tensor],
        dict[int, Tensor],
        dict[int, Tensor],
    ]:
        online_state, target_state = self._prefix_states(view)
        online_q: dict[int, Tensor] = {}
        online_value: dict[int, Tensor] = {}
        online_logits: dict[int, Tensor] = {}
        target_q: dict[int, Tensor] = {}
        target_value: dict[int, Tensor] = {}
        for index in view.learn_indices:
            step = view.run.steps[index]
            q, value, logits, online_state = self.forward_online(step, online_state)
            with torch.no_grad():
                q_target, value_target, _, target_state = self.forward_target(step, target_state)
            online_q[index] = q
            online_value[index] = value
            online_logits[index] = logits
            target_q[index] = q_target
            target_value[index] = value_target
        return online_q, online_value, online_logits, target_q, target_value

    def _bootstrap_value(self, run: HybridRun, index: int) -> Tensor:
        _, target_state = self._state_before(run, index)
        with torch.no_grad():
            _, value, _, _ = self.forward_target(run.steps[index], target_state)
        return value.detach()

    @staticmethod
    def _edge(
        run: HybridRun,
        start: int,
        stop_exclusive: int,
        *,
        lambda_value: float = 0.0,
    ) -> tuple[float, float]:
        value = 0.0
        weight = 1.0
        for index in range(start, stop_exclusive):
            step = run.steps[index]
            value += weight * (step.reward - float(lambda_value) * step.revivals_delta)
            weight *= step.discount
            if weight == 0.0:
                break
        return value, weight

    def _node_losses(
        self,
        *,
        step: HybridStep,
        q: Tensor,
        value: Tensor,
        logits: Tensor,
        target_q: Tensor,
        target_value: Tensor,
        factual_target: Tensor,
    ) -> tuple[Tensor, Tensor, Tensor, Tensor, Tensor, Tensor]:
        executed_q = q[step.action_index]
        q_loss = torch.nn.functional.huber_loss(
            executed_q,
            factual_target,
            delta=self.config.huber_delta,
        )
        q_reference = target_q[step.action_index].detach()
        expectile_error = q_reference - value
        expectile_weight = torch.where(
            expectile_error >= 0.0,
            expectile_error.new_tensor(self.config.expectile),
            expectile_error.new_tensor(1.0 - self.config.expectile),
        )
        value_loss = expectile_weight * expectile_error.square()

        count = len(step.snapshot.action_mask)
        legal = torch.tensor(
            step.snapshot.action_mask.tolist(),
            dtype=torch.bool,
            device=logits.device,
        )
        legal_logits = logits[:count].masked_fill(~legal, float("-inf"))
        log_policy = torch.log_softmax(legal_logits, dim=0)
        advantage = (q_reference - target_value.detach()) / self.config.advantage_temperature
        actor_weight = advantage.exp().clamp(max=self.config.maximum_advantage_weight)
        actor_loss = -actor_weight * log_policy[step.action_index]
        probabilities = log_policy.exp()
        entropy = -(probabilities[legal] * log_policy[legal]).sum()
        return (
            q_loss,
            value_loss,
            actor_loss,
            (executed_q - factual_target).detach().abs(),
            actor_weight.detach(),
            entropy.detach(),
        )

    def _macro_loss(self, view: DomainView, *, lambda_value: float) -> _DomainLoss:
        q, value, logits, target_q, target_value = self._states_and_outputs(view)
        rows: list[tuple[Tensor, Tensor, Tensor, Tensor, Tensor, Tensor]] = []
        run = view.run
        for index in view.learn_indices:
            successor = next_domain_index(run, index, "macro")
            stop = successor if successor is not None else len(run.steps)
            factual, discount = self._edge(run, index, stop, lambda_value=lambda_value)
            target = q[index].new_tensor(factual)
            if successor is not None and discount != 0.0:
                bootstrap = (
                    target_value[successor].detach()
                    if successor in target_value
                    else self._bootstrap_value(run, successor)
                )
                target = target + discount * bootstrap
            rows.append(
                self._node_losses(
                    step=run.steps[index],
                    q=q[index],
                    value=value[index],
                    logits=logits[index],
                    target_q=target_q[index],
                    target_value=target_value[index],
                    factual_target=target,
                )
            )
        return self._reduce_rows(rows)

    def _combat_loss(self, view: DomainView, *, lambda_value: float) -> _DomainLoss:
        q, value, logits, target_q, target_value = self._states_and_outputs(view)
        rows: list[tuple[Tensor, Tensor, Tensor, Tensor, Tensor, Tensor]] = []
        run = view.run
        for index in view.learn_indices:
            encounter_id = run.steps[index].encounter_id
            cursor = index
            combat_seen = 0
            while cursor < len(run.steps):
                current = run.steps[cursor]
                if current.control_domain == "combat":
                    if current.encounter_id != encounter_id:
                        break
                    combat_seen += 1
                elif cursor > index:
                    break
                cursor += 1
                if combat_seen >= self.config.combat_n_step:
                    break
            successor = cursor
            factual, discount = self._edge(run, index, successor, lambda_value=lambda_value)
            target = q[index].new_tensor(factual)
            if successor < len(run.steps) and discount != 0.0:
                bootstrap = (
                    target_value[successor].detach()
                    if successor in target_value and successor > index
                    else self._bootstrap_value(run, successor)
                )
                target = target + discount * bootstrap
            rows.append(
                self._node_losses(
                    step=run.steps[index],
                    q=q[index],
                    value=value[index],
                    logits=logits[index],
                    target_q=target_q[index],
                    target_value=target_value[index],
                    factual_target=target,
                )
            )
        return self._reduce_rows(rows)

    @staticmethod
    def _reduce_rows(
        rows: list[tuple[Tensor, Tensor, Tensor, Tensor, Tensor, Tensor]],
    ) -> _DomainLoss:
        if not rows:
            raise ValueError("hybrid IQL domain view has no learning nodes")
        columns = tuple(torch.stack(items) for items in zip(*rows, strict=True))
        q, value, actor, td_abs, weights, entropies = columns
        q_mean = q.mean()
        value_mean = value.mean()
        actor_mean = actor.mean()
        return _DomainLoss(
            total=q_mean + value_mean + actor_mean,
            q=q_mean,
            value=value_mean,
            actor=actor_mean,
            td_abs=td_abs.mean(),
            weights=weights,
            entropies=entropies,
            steps=len(rows),
        )

    @staticmethod
    def _require_finite(label: str, *values: Tensor) -> None:
        if not all(bool(torch.isfinite(value).all().item()) for value in values):
            raise FloatingPointError(f"hybrid IQL produced non-finite {label}")

    def update(self) -> dict[str, Any]:
        pairs = self.replay.sample_pairs(self.config.run_batch_size)
        if not pairs:
            return self.as_mapping()
        price = self.death_price.lambda_value
        macro_rows = [self._macro_loss(pair.macro, lambda_value=price) for pair in pairs]
        combat_rows = [self._combat_loss(pair.combat, lambda_value=price) for pair in pairs]

        def mean(name: str, rows: list[_DomainLoss]) -> Tensor:
            return torch.stack([getattr(row, name) for row in rows]).mean()

        macro = _DomainLoss(
            total=mean("total", macro_rows),
            q=mean("q", macro_rows),
            value=mean("value", macro_rows),
            actor=mean("actor", macro_rows),
            td_abs=mean("td_abs", macro_rows),
            weights=torch.cat([row.weights for row in macro_rows]),
            entropies=torch.cat([row.entropies for row in macro_rows]),
            steps=sum(row.steps for row in macro_rows),
        )
        combat = _DomainLoss(
            total=mean("total", combat_rows),
            q=mean("q", combat_rows),
            value=mean("value", combat_rows),
            actor=mean("actor", combat_rows),
            td_abs=mean("td_abs", combat_rows),
            weights=torch.cat([row.weights for row in combat_rows]),
            entropies=torch.cat([row.entropies for row in combat_rows]),
            steps=sum(row.steps for row in combat_rows),
        )
        loss = 0.5 * (macro.total + combat.total)
        self._require_finite(
            "loss or target",
            macro.total,
            combat.total,
            macro.q,
            combat.q,
            macro.value,
            combat.value,
            macro.actor,
            combat.actor,
            loss,
        )

        self.optimizer.zero_grad(set_to_none=True)
        if self.metrics.updates % self.config.gradient_diagnostics_interval == 0 and self._shared_parameters:
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
            self._require_finite("shared-gradient diagnostic", macro_norm, combat_norm, cosine)
            self.metrics.macro_shared_gradient_norm = float(macro_norm.item())
            self.metrics.combat_shared_gradient_norm = float(combat_norm.item())
            self.metrics.shared_gradient_cosine = float(cosine.item())
        loss.backward()  # type: ignore[no-untyped-call]
        critic_norm = torch.nn.utils.clip_grad_norm_(
            self._critic_parameters,
            self.config.gradient_clip_norm,
        )
        actor_norm = torch.nn.utils.clip_grad_norm_(
            self._actor_parameters,
            self.config.gradient_clip_norm,
        )
        self._require_finite("gradient norm", critic_norm, actor_norm)
        self.optimizer.step()
        if any(
            parameter.is_floating_point() and not bool(torch.isfinite(parameter).all().item())
            for parameter in self._parameters
        ):
            raise FloatingPointError("hybrid IQL optimizer produced non-finite parameters")

        self.metrics.updates += 1
        self.metrics.macro_steps += macro.steps
        self.metrics.combat_steps += combat.steps
        self.metrics.macro_loss = float(macro.total.detach().item())
        self.metrics.combat_loss = float(combat.total.detach().item())
        self.metrics.loss = float(loss.detach().item())
        self.metrics.macro_q_loss = float(macro.q.detach().item())
        self.metrics.combat_q_loss = float(combat.q.detach().item())
        self.metrics.macro_value_loss = float(macro.value.detach().item())
        self.metrics.combat_value_loss = float(combat.value.detach().item())
        self.metrics.macro_actor_loss = float(macro.actor.detach().item())
        self.metrics.combat_actor_loss = float(combat.actor.detach().item())
        self.metrics.macro_td_abs = float(macro.td_abs.detach().item())
        self.metrics.combat_td_abs = float(combat.td_abs.detach().item())
        weights = torch.cat([macro.weights, combat.weights])
        entropies = torch.cat([macro.entropies, combat.entropies])
        self.metrics.mean_advantage_weight = float(weights.mean().item())
        self.metrics.maximum_advantage_weight = float(weights.max().item())
        self.metrics.mean_policy_entropy = float(entropies.mean().item())
        self.metrics.critic_gradient_norm = float(critic_norm)
        self.metrics.actor_gradient_norm = float(actor_norm)
        self.metrics.gradient_norm = float(torch.maximum(critic_norm, actor_norm))

        policy_lags = [
            max(0, self.metrics.updates - int(view.run.steps[index].policy_version))
            for pair in pairs
            for view in (pair.macro, pair.combat)
            for index in view.learn_indices
        ]
        self.metrics.mean_policy_lag = sum(policy_lags) / len(policy_lags) if policy_lags else 0.0
        self.metrics.maximum_policy_lag = max(policy_lags, default=0)
        for pair in pairs:
            for view in (pair.macro, pair.combat):
                for index in view.learn_indices:
                    step = view.run.steps[index]
                    key = f"{view.domain}:{step.surface}:{step.branch}"
                    self.metrics.domain_surface_counts[key] = self.metrics.domain_surface_counts.get(key, 0) + 1
        if self.metrics.updates - self.metrics.last_target_sync_update >= self.config.target_update_interval:
            self.sync_target()
        return self.as_mapping()

    def observe_episode_outcome(
        self,
        *,
        act1_cleared: bool,
        run_success: bool,
        act1_revivals: int | None,
        run_revivals: int,
    ) -> bool:
        return self.death_price.observe_episode(
            DeathPriceEpisodeOutcome(
                act1_cleared=act1_cleared,
                run_success=run_success,
                act1_revivals=act1_revivals,
                run_revivals=run_revivals,
            )
        )

    def initialize_death_price(self, value: float) -> None:
        inherited = float(value)
        if not math.isfinite(inherited) or inherited < 0.0:
            raise ValueError("inherited death price must be finite and non-negative")
        self.death_price = AbsoluteDeathPriceController(
            self.death_price.course,
            initial_lambda=min(inherited, self.death_price.course.lambda_max),
        )

    def observe_heldout(self, *, act1_rate: float, run_rate: float, policy_version: int) -> None:
        self.death_price.observe_heldout(
            act1_rate=act1_rate,
            run_rate=run_rate,
            policy_version=policy_version,
        )

    def as_mapping(self) -> dict[str, Any]:
        return {**self.metrics.as_mapping(), "death_price": self.death_price.as_mapping()}

    def sync_target(self) -> None:
        self.target_model.load_state_dict(self.online_model.state_dict(), strict=True)
        self.metrics.target_syncs += 1
        self.metrics.last_target_sync_update = self.metrics.updates

    def state_dict(self) -> dict[str, Any]:
        return {
            "version": HYBRID_IQL_LEARNER_VERSION,
            "config": asdict(self.config),
            "optimizer": self.optimizer.state_dict(),
            "metrics": self.metrics.as_mapping(),
            "death_price": self.death_price.state_dict(),
        }

    def load_state_dict(self, state: dict[str, Any]) -> None:
        if state.get("version") != HYBRID_IQL_LEARNER_VERSION:
            raise ValueError("hybrid IQL learner state version differs")
        if dict(state.get("config", {})) != asdict(self.config):
            raise ValueError("hybrid IQL learner config differs")
        self.optimizer.load_state_dict(state["optimizer"])
        metrics = dict(state["metrics"])
        metrics.pop("version", None)
        self.metrics = HybridIQLMetrics(**metrics)
        self.death_price.load_state_dict(dict(state["death_price"]))


__all__ = [
    "HYBRID_IQL_LEARNER_VERSION",
    "HybridIQLForward",
    "HybridIQLLearner",
    "HybridIQLLearnerConfig",
    "HybridIQLMetrics",
]
