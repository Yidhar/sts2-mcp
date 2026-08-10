"""Recurrent candidate Double-Q learner for the isolated macro domain.

One domain, one primary preference-learning objective: n-step Double-Q TD on
executed macro candidates.  The online network selects the bootstrap argmax,
the slowly-updated target network evaluates it, and an executed action always
receives a full Q-regression gradient regardless of its greedy rank — the
structural answer to the softmax-absorption pathology this migration retires.

The learner owns NO other objective: no imitation, no completion CE, no
support corridor, no liveness actor.  Model instances are injected together
with the collate/initial-state callables so the learner never imports the
legacy training stack.
"""

from __future__ import annotations

import math
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from typing import Any, Final

import torch
from torch import Tensor

from .replay import MacroSequenceReplay, MacroWindow
from .transitions import MacroStep, n_step_targets

MACRO_Q_LEARNER_VERSION: Final = "sts2-macro-double-q-v1"


@dataclass(frozen=True, slots=True)
class MacroQConfig:
    n_step: int = 3
    learning_rate: float = 1.0e-4
    target_update_interval: int = 200
    huber_delta: float = 1.0
    gradient_clip_norm: float = 1.0
    sample_windows: int = 4
    coverage_k_min: int = 8

    def __post_init__(self) -> None:
        if self.n_step <= 0 or self.target_update_interval <= 0 or self.sample_windows <= 0:
            raise ValueError("macro Q config bounds must be positive")
        for label, value in (
            ("learning_rate", self.learning_rate),
            ("huber_delta", self.huber_delta),
            ("gradient_clip_norm", self.gradient_clip_norm),
        ):
            if not math.isfinite(value) or value <= 0.0:
                raise ValueError(f"macro Q config {label} must be positive and finite")


@dataclass(slots=True)
class MacroQMetrics:
    updates: int = 0
    windows_trained: int = 0
    steps_trained: int = 0
    loss: float = 0.0
    td_error_mean: float = 0.0
    td_error_max: float = 0.0
    target_syncs: int = 0
    executed_counts: dict[str, int] = field(default_factory=dict)

    def as_mapping(self) -> dict[str, Any]:
        return {
            "version": MACRO_Q_LEARNER_VERSION,
            "updates": self.updates,
            "windows_trained": self.windows_trained,
            "steps_trained": self.steps_trained,
            "loss": self.loss,
            "td_error_mean": self.td_error_mean,
            "td_error_max": self.td_error_max,
            "target_syncs": self.target_syncs,
            "executed_counts": dict(sorted(self.executed_counts.items())),
        }


class MacroQLearner:
    """n-step Double-Q over replayed macro windows.

    ``forward`` must run one collated snapshot through a model and return
    ``(q_values_1d, next_recurrent_state)``; ``initial_state`` returns the
    domain's initial recurrent state.  Injecting these keeps the learner
    independent from any concrete model class (§7 ownership).
    """

    def __init__(
        self,
        *,
        online_parameters: Sequence[torch.nn.Parameter],
        forward_online: Callable[[MacroStep, Any], tuple[Tensor, Any]],
        forward_target: Callable[[MacroStep, Any], tuple[Tensor, Any]],
        sync_target: Callable[[], None],
        initial_state: Callable[[], Any],
        replay: MacroSequenceReplay,
        config: MacroQConfig | None = None,
    ) -> None:
        self.config = config or MacroQConfig()
        self.forward_online = forward_online
        self.forward_target = forward_target
        self.sync_target = sync_target
        self.initial_state = initial_state
        self.replay = replay
        parameters = list(online_parameters)
        if not parameters:
            raise ValueError("macro Q learner requires trainable parameters")
        self.optimizer = torch.optim.Adam(parameters, lr=self.config.learning_rate)
        self._parameters = parameters
        self.metrics = MacroQMetrics()

    def _window_values(
        self,
        window: MacroWindow,
    ) -> tuple[Tensor, tuple[float, ...], tuple[float, ...], tuple[float, ...], list[MacroStep]]:
        steps = window.episode.steps
        begin, end = window.learn_slice
        hidden = self.initial_state()
        target_hidden = self.initial_state()
        with torch.no_grad():
            for step in steps[window.start : begin]:
                _, hidden = self.forward_online(step, hidden)
                _, target_hidden = self.forward_target(step, target_hidden)

        executed_q: list[Tensor] = []
        online_argmax: list[int] = []
        target_values: list[float] = []
        learn_steps = list(steps[begin:end])
        for step in learn_steps:
            q_online, hidden = self.forward_online(step, hidden)
            with torch.no_grad():
                q_target, target_hidden = self.forward_target(step, target_hidden)
            mask = torch.as_tensor(
                step.snapshot.action_mask,
                dtype=torch.bool,
                device=q_online.device,
            )
            masked_online = q_online.masked_fill(~mask, float("-inf"))
            executed_q.append(q_online[step.action_index])
            argmax_index = int(masked_online.argmax().item())
            online_argmax.append(argmax_index)
            target_values.append(float(q_target[argmax_index].item()))

        # Double-Q bootstrap for transition t consumes the NEXT decision's
        # value; the final transition in the window bootstraps from itself
        # only through its own clock discount (terminal cuts to zero).
        bootstraps = tuple(
            target_values[index + 1] if index + 1 < len(target_values) else 0.0
            for index in range(len(learn_steps))
        )
        rewards = tuple(step.reward for step in learn_steps)
        discounts = tuple(step.discount for step in learn_steps)
        return (
            torch.stack(executed_q),
            rewards,
            discounts,
            bootstraps,
            learn_steps,
        )

    def update(self) -> dict[str, Any]:
        windows = self.replay.sample(self.config.sample_windows)
        if not windows:
            return self.metrics.as_mapping()
        losses: list[Tensor] = []
        td_abs: list[float] = []
        steps_trained = 0
        for window in windows:
            executed_q, rewards, discounts, bootstraps, learn_steps = (
                self._window_values(window)
            )
            targets = torch.as_tensor(
                n_step_targets(
                    rewards,
                    discounts,
                    bootstraps,
                    n_step=self.config.n_step,
                ),
                dtype=executed_q.dtype,
                device=executed_q.device,
            )
            td = executed_q - targets
            losses.append(
                torch.nn.functional.huber_loss(
                    executed_q,
                    targets,
                    delta=self.config.huber_delta,
                )
            )
            window_td = float(td.detach().abs().mean().item())
            td_abs.append(window_td)
            self.replay.update_priority(window.window_id, window_td)
            steps_trained += len(learn_steps)
            for step in learn_steps:
                key = f"{step.surface}:{step.branch}"
                self.metrics.executed_counts[key] = (
                    self.metrics.executed_counts.get(key, 0) + 1
                )

        loss = torch.stack(losses).mean()
        self.optimizer.zero_grad(set_to_none=True)
        loss.backward()  # type: ignore[no-untyped-call]
        torch.nn.utils.clip_grad_norm_(
            self._parameters,
            self.config.gradient_clip_norm,
        )
        self.optimizer.step()

        self.metrics.updates += 1
        self.metrics.windows_trained += len(windows)
        self.metrics.steps_trained += steps_trained
        self.metrics.loss = float(loss.detach().item())
        self.metrics.td_error_mean = sum(td_abs) / len(td_abs)
        self.metrics.td_error_max = max(td_abs)
        if self.metrics.updates % self.config.target_update_interval == 0:
            self.sync_target()
            self.metrics.target_syncs += 1
        return self.metrics.as_mapping()


__all__ = [
    "MACRO_Q_LEARNER_VERSION",
    "MacroQConfig",
    "MacroQLearner",
    "MacroQMetrics",
]
