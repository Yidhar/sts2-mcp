"""Balanced recurrent Double-Q learner for the mixed decision graph.

There is one preference objective: Huber regression of the executed semantic
candidate to a factual Double-Q target.  Macro and combat views use different
factual horizons, then contribute equal *domain means* to one optimizer step.
No probability ratio, imitation label, completion CE or operation-specific
bonus sits between an executed rare action and its Q gradient.
"""

from __future__ import annotations

import math
from collections.abc import Callable
from dataclasses import asdict, dataclass, field
from typing import Any, Final

import torch
from torch import Tensor, nn

from .replay import DomainView, HybridRunReplay
from .transitions import HybridRun, HybridStep, next_domain_index

HYBRID_Q_LEARNER_VERSION: Final = "sts2-hybrid-double-q-v1"


@dataclass(frozen=True, slots=True)
class HybridQConfig:
    learning_rate: float = 1.0e-4
    shared_learning_rate_scale: float = 0.5
    target_update_interval: int = 200
    huber_delta: float = 1.0
    gradient_clip_norm: float = 1.0
    run_batch_size: int = 4
    combat_n_step: int = 8
    gradient_diagnostics_interval: int = 50

    def __post_init__(self) -> None:
        if min(
            self.target_update_interval,
            self.run_batch_size,
            self.combat_n_step,
            self.gradient_diagnostics_interval,
        ) <= 0:
            raise ValueError("hybrid Q integer bounds must be positive")
        for name in (
            "learning_rate",
            "shared_learning_rate_scale",
            "huber_delta",
            "gradient_clip_norm",
        ):
            value = float(getattr(self, name))
            if not math.isfinite(value) or value <= 0.0:
                raise ValueError(f"hybrid Q {name} must be positive and finite")


@dataclass(slots=True)
class HybridQMetrics:
    updates: int = 0
    macro_steps: int = 0
    combat_steps: int = 0
    macro_loss: float = 0.0
    combat_loss: float = 0.0
    loss: float = 0.0
    macro_td_abs: float = 0.0
    combat_td_abs: float = 0.0
    gradient_norm: float = 0.0
    macro_shared_gradient_norm: float = 0.0
    combat_shared_gradient_norm: float = 0.0
    shared_gradient_cosine: float = 0.0
    mean_policy_lag: float = 0.0
    maximum_policy_lag: int = 0
    target_syncs: int = 0
    domain_surface_counts: dict[str, int] = field(default_factory=dict)

    def as_mapping(self) -> dict[str, Any]:
        return {
            "version": HYBRID_Q_LEARNER_VERSION,
            **asdict(self),
        }


class HybridQLearner:
    """One optimizer and one Double-Q objective over balanced domain views."""

    def __init__(
        self,
        *,
        online_model: nn.Module,
        target_model: nn.Module,
        replay: HybridRunReplay,
        forward_online: Callable[[HybridStep, Tensor | None], tuple[Tensor, Tensor]],
        forward_target: Callable[[HybridStep, Tensor | None], tuple[Tensor, Tensor]],
        initial_state: Callable[[], Tensor],
        config: HybridQConfig | None = None,
    ) -> None:
        self.online_model = online_model
        self.target_model = target_model
        self.replay = replay
        self.forward_online = forward_online
        self.forward_target = forward_target
        self.initial_state = initial_state
        self.config = config or HybridQConfig()
        local_tokens = (
            "macro_stream",
            "combat_stream",
            "macro_stream_norm",
            "combat_stream_norm",
            "candidate_q_head",
        )
        shared: list[nn.Parameter] = []
        local: list[nn.Parameter] = []
        for name, parameter in online_model.named_parameters():
            (local if name.startswith(local_tokens) else shared).append(parameter)
        groups: list[dict[str, Any]] = []
        if local:
            groups.append({"params": local, "lr": self.config.learning_rate})
        if shared:
            groups.append(
                {
                    "params": shared,
                    "lr": self.config.learning_rate
                    * self.config.shared_learning_rate_scale,
                }
            )
        self.optimizer = torch.optim.AdamW(groups)
        self._parameters = [*local, *shared]
        self._shared_parameters = shared
        self.metrics = HybridQMetrics()
        # Recurrent replay must be deterministic under current parameters;
        # exploration belongs to collection, not dropout in burn-in prefixes.
        self.online_model.eval()
        self.target_model.eval()
        for parameter in self.target_model.parameters():
            parameter.requires_grad_(False)

    @staticmethod
    def _gradient_vector(
        loss: Tensor,
        parameters: list[nn.Parameter],
    ) -> Tensor:
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
        if not parts:
            return loss.new_zeros(1)
        return torch.cat(parts)

    @staticmethod
    def _masked_argmax(q_values: Tensor, step: HybridStep) -> int:
        count = len(step.snapshot.action_mask)
        q = q_values[:count]
        mask = torch.tensor(
            step.snapshot.action_mask.tolist(), dtype=torch.bool, device=q.device
        )
        return int(q.masked_fill(~mask, float("-inf")).argmax().item())

    def _prefix_states(self, view: DomainView) -> tuple[Tensor, Tensor]:
        online_state = self.initial_state()
        target_state = self.initial_state()
        with torch.no_grad():
            for index in view.prefix_indices:
                _, online_state = self.forward_online(view.run.steps[index], online_state)
                _, target_state = self.forward_target(view.run.steps[index], target_state)
        return online_state.detach(), target_state.detach()

    def _state_before(self, run: HybridRun, index: int) -> tuple[Tensor, Tensor]:
        """Reconstruct the exact sparse memory prefix for one graph node."""

        macro_prefix = tuple(
            candidate for candidate in run.indices("macro") if candidate < index
        )
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
        state = self.initial_state()
        target_state = self.initial_state()
        with torch.no_grad():
            for prefix_index in prefix:
                _, state = self.forward_online(run.steps[prefix_index], state)
                _, target_state = self.forward_target(
                    run.steps[prefix_index], target_state
                )
        return state.detach(), target_state.detach()

    def _states_and_q(
        self,
        view: DomainView,
    ) -> tuple[
        dict[int, Tensor],
        dict[int, Tensor],
        dict[int, Tensor],
        dict[int, Tensor],
    ]:
        online_state, target_state = self._prefix_states(view)
        online_q: dict[int, Tensor] = {}
        target_q: dict[int, Tensor] = {}
        online_before: dict[int, Tensor] = {}
        target_before: dict[int, Tensor] = {}
        # The learning nodes are sequential within their sparse domain view.
        # Macro views deliberately skip tactical nodes; their edge target
        # folds those factual rewards separately. Combat views stay inside one
        # encounter and therefore retain tactical recurrence.
        for index in view.learn_indices:
            step = view.run.steps[index]
            online_before[index] = online_state
            target_before[index] = target_state
            q_online, online_state = self.forward_online(step, online_state)
            with torch.no_grad():
                q_target, target_state = self.forward_target(step, target_state)
            online_q[index] = q_online
            target_q[index] = q_target
        return online_q, target_q, online_before, target_before

    def _bootstrap(
        self,
        run: HybridRun,
        index: int,
        *,
        online_state: Tensor,
        target_state: Tensor,
    ) -> Tensor:
        step = run.steps[index]
        q_online, _ = self.forward_online(step, online_state)
        with torch.no_grad():
            q_target, _ = self.forward_target(step, target_state)
        selected = self._masked_argmax(q_online.detach(), step)
        return q_target[selected].detach()

    @staticmethod
    def _edge(
        run: HybridRun,
        start: int,
        stop_exclusive: int,
    ) -> tuple[float, float]:
        value = 0.0
        weight = 1.0
        for index in range(start, stop_exclusive):
            step = run.steps[index]
            value += weight * step.reward
            weight *= step.discount
            if weight == 0.0:
                break
        return value, weight

    def _macro_loss(self, view: DomainView) -> tuple[Tensor, Tensor, int]:
        online_q, target_q, _, _ = self._states_and_q(view)
        losses: list[Tensor] = []
        errors: list[Tensor] = []
        run = view.run
        for index in view.learn_indices:
            step = run.steps[index]
            successor = next_domain_index(run, index, "macro")
            stop = successor if successor is not None else len(run.steps)
            factual, discount = self._edge(run, index, stop)
            target = torch.as_tensor(
                factual, dtype=online_q[index].dtype, device=online_q[index].device
            )
            if successor is not None and discount != 0.0:
                if successor in online_q:
                    # The sparse macro view already reconstructed this exact
                    # successor under current weights. Reuse it instead of
                    # replaying the whole run prefix once per learned node.
                    selected = self._masked_argmax(
                        online_q[successor].detach(), run.steps[successor]
                    )
                    bootstrap = target_q[successor][selected].detach()
                else:
                    # A successor beyond the sampled suffix still needs one
                    # exact sparse-prefix reconstruction.
                    online_state, target_state = self._state_before(run, successor)
                    bootstrap = self._bootstrap(
                        run,
                        successor,
                        online_state=online_state,
                        target_state=target_state,
                    )
                target = target + discount * bootstrap
            executed = online_q[index][step.action_index]
            errors.append((executed - target).detach().abs())
            losses.append(
                torch.nn.functional.huber_loss(
                    executed, target, delta=self.config.huber_delta
                )
            )
        return torch.stack(losses).mean(), torch.stack(errors).mean(), len(losses)

    def _combat_loss(self, view: DomainView) -> tuple[Tensor, Tensor, int]:
        online_q, target_q, _, _ = self._states_and_q(view)
        losses: list[Tensor] = []
        errors: list[Tensor] = []
        run = view.run
        indices = view.learn_indices
        for index in indices:
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
            factual, discount = self._edge(run, index, successor)
            target = torch.as_tensor(
                factual, dtype=online_q[index].dtype, device=online_q[index].device
            )
            if successor < len(run.steps) and discount != 0.0:
                if successor in target_q and successor > index:
                    selected = self._masked_argmax(
                        online_q[successor].detach(), run.steps[successor]
                    )
                    bootstrap = target_q[successor][selected].detach()
                else:
                    # Current encounter prefix + macro prefix has already been
                    # reconstructed; advance the factual nodes between this
                    # decision and its successor without gradient.
                    online_state, target_state = self._state_before(run, successor)
                    bootstrap = self._bootstrap(
                        run,
                        successor,
                        online_state=online_state,
                        target_state=target_state,
                    )
                target = target + discount * bootstrap
            executed = online_q[index][run.steps[index].action_index]
            errors.append((executed - target).detach().abs())
            losses.append(
                torch.nn.functional.huber_loss(
                    executed, target, delta=self.config.huber_delta
                )
            )
        return torch.stack(losses).mean(), torch.stack(errors).mean(), len(losses)

    def update(self) -> dict[str, Any]:
        pairs = self.replay.sample_pairs(self.config.run_batch_size)
        if not pairs:
            return self.metrics.as_mapping()
        macro_rows = [self._macro_loss(pair.macro) for pair in pairs]
        combat_rows = [self._combat_loss(pair.combat) for pair in pairs]
        macro_loss = torch.stack([row[0] for row in macro_rows]).mean()
        combat_loss = torch.stack([row[0] for row in combat_rows]).mean()
        loss = 0.5 * (macro_loss + combat_loss)
        self.optimizer.zero_grad(set_to_none=True)
        if (
            self.metrics.updates % self.config.gradient_diagnostics_interval == 0
            and self._shared_parameters
        ):
            macro_gradient = self._gradient_vector(
                macro_loss,
                self._shared_parameters,
            )
            combat_gradient = self._gradient_vector(
                combat_loss,
                self._shared_parameters,
            )
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
            self._parameters, self.config.gradient_clip_norm
        )
        self.optimizer.step()

        self.metrics.updates += 1
        self.metrics.macro_steps += sum(row[2] for row in macro_rows)
        self.metrics.combat_steps += sum(row[2] for row in combat_rows)
        self.metrics.macro_loss = float(macro_loss.detach().item())
        self.metrics.combat_loss = float(combat_loss.detach().item())
        self.metrics.loss = float(loss.detach().item())
        self.metrics.macro_td_abs = float(
            torch.stack([row[1] for row in macro_rows]).mean().item()
        )
        self.metrics.combat_td_abs = float(
            torch.stack([row[1] for row in combat_rows]).mean().item()
        )
        self.metrics.gradient_norm = float(gradient_norm)
        policy_lags = [
            max(0, self.metrics.updates - int(view.run.steps[index].policy_version))
            for pair in pairs
            for view in (pair.macro, pair.combat)
            for index in view.learn_indices
        ]
        self.metrics.mean_policy_lag = (
            sum(policy_lags) / len(policy_lags) if policy_lags else 0.0
        )
        self.metrics.maximum_policy_lag = max(policy_lags, default=0)
        for pair in pairs:
            for view in (pair.macro, pair.combat):
                for index in view.learn_indices:
                    step = view.run.steps[index]
                    key = f"{view.domain}:{step.surface}:{step.branch}"
                    self.metrics.domain_surface_counts[key] = (
                        self.metrics.domain_surface_counts.get(key, 0) + 1
                    )
        if self.metrics.updates % self.config.target_update_interval == 0:
            self.sync_target()
        return self.metrics.as_mapping()

    def sync_target(self) -> None:
        """Copy online weights to the target and account for the sync.

        The normal clock is learner updates. A runner that deliberately
        changes the update-to-fresh-run ratio can additionally call this on
        its factual run clock, preserving the same target-data cadence.
        """

        self.target_model.load_state_dict(self.online_model.state_dict(), strict=True)
        self.metrics.target_syncs += 1

    def state_dict(self) -> dict[str, Any]:
        return {
            "version": HYBRID_Q_LEARNER_VERSION,
            "config": asdict(self.config),
            "optimizer": self.optimizer.state_dict(),
            "metrics": self.metrics.as_mapping(),
        }

    def load_state_dict(self, state: dict[str, Any]) -> None:
        if state.get("version") != HYBRID_Q_LEARNER_VERSION:
            raise ValueError("hybrid learner state version differs")
        if state.get("config") != asdict(self.config):
            raise ValueError("hybrid learner config differs")
        self.optimizer.load_state_dict(state["optimizer"])
        metrics = dict(state["metrics"])
        metrics.pop("version", None)
        self.metrics = HybridQMetrics(**metrics)


__all__ = [
    "HYBRID_Q_LEARNER_VERSION",
    "HybridQConfig",
    "HybridQLearner",
    "HybridQMetrics",
]
