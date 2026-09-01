"""Single collection authority for the hybrid combat/macro controller."""

from __future__ import annotations

import math
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from typing import Any, Final, Literal

import numpy as np
import torch
from torch import Tensor

from sts2_rl.macro.authority import MacroCollectionAuthority
from sts2_rl.semantics.clock import decision_discount
from sts2_rl.semantics.forward import ForwardDecision, SemanticCandidate

from .transitions import (
    HybridRun,
    HybridStep,
    HybridVTraceStep,
    HybridVTraceUnroll,
)

HYBRID_AUTHORITY_VERSION: Final = "sts2-hybrid-authority-v3"
HYBRID_VTRACE_AUTHORITY_VERSION: Final = "sts2-hybrid-vtrace-authority-v1"


@dataclass(frozen=True, slots=True)
class DomainExplorationSchedule:
    start: float = 0.15
    end: float = 0.05
    decay_decisions: int = 100_000

    def value(self, decisions: int) -> float:
        progress = min(max(int(decisions), 0) / max(self.decay_decisions, 1), 1.0)
        return float(self.start + (self.end - self.start) * progress)


@dataclass(frozen=True, slots=True)
class FactualCoverageSchedule:
    """Temporary branch support for sparse strategic macro surfaces.

    This schedule changes only which legal factual action collection executes.
    It never enters model logits, replay rewards, learner targets, or held-out
    evaluation.  A surface qualifies generically when it exposes more than
    one semantic branch; no operation receives a preferred label.
    """

    start: float = 0.50
    end: float = 0.20
    decay_decisions: int = 50_000

    def value(self, decisions: int) -> float:
        progress = min(max(int(decisions), 0) / max(self.decay_decisions, 1), 1.0)
        return float(self.start + (self.end - self.start) * progress)


class HybridCollectionAuthority(MacroCollectionAuthority):
    """Own every meaningful decision with one recurrent state.

    The reviewed semantic compiler/executor from the stage-two authority is
    reused, but the two isolated episodes and their scalar bridge disappear.
    Replay stores no behavior-network hidden state: current online and target
    memories are rebuilt from the factual sparse prefix.
    """

    def __init__(
        self,
        *args: Any,
        macro_exploration: DomainExplorationSchedule | None = None,
        combat_exploration: DomainExplorationSchedule | None = None,
        factual_coverage: FactualCoverageSchedule | None = None,
        **kwargs: Any,
    ) -> None:
        kwargs["control_domain"] = "joint"
        super().__init__(*args, **kwargs)
        self.exploration = {
            "macro": macro_exploration or DomainExplorationSchedule(),
            "combat": combat_exploration or DomainExplorationSchedule(),
        }
        self.factual_coverage = factual_coverage or FactualCoverageSchedule()
        self.domain_decisions = {"macro": 0, "combat": 0}
        self.policy_version = 0
        self._open_policy_version = 0
        self._open_encounter_id: int | None = None
        self._open_revivals_delta = 0
        self._hybrid_steps: list[HybridStep] = []
        self._terminal_observed = False
        self._encounter_id = 0
        self._was_in_combat = False
        self._coverage_eligible_decisions = 0
        self._coverage_exploratory_decisions = 0
        self._coverage_branch_counts: dict[str, int] = {}

    @property
    def exclusive_control(self) -> bool:
        """The joint Candidate-Q model owns every compiled semantic choice."""

        return True

    @property
    def terminal_observed(self) -> bool:
        """Whether the collector supplied a factual terminal transition."""

        return self._terminal_observed

    def exploration_state_dict(self) -> dict[str, Any]:
        """Return the two decision clocks carried by trainer publication."""

        return {
            "domain_decisions": dict(self.domain_decisions),
            "policy_version": int(self.policy_version),
        }

    def load_exploration_state_dict(self, state: Mapping[str, Any]) -> None:
        decisions = state.get("domain_decisions", {})
        if not isinstance(decisions, Mapping):
            raise TypeError("hybrid domain decision clocks must be a mapping")
        for domain in ("macro", "combat"):
            value = int(decisions.get(domain, 0))
            if value < 0:
                raise ValueError("hybrid domain decision clocks cannot be negative")
            # A collector can advance beyond a slightly older learner
            # publication.  Reloading weights must never rewind exploration.
            self.domain_decisions[domain] = max(
                self.domain_decisions[domain],
                value,
            )
        version = int(state.get("policy_version", 0))
        if version < 0:
            raise ValueError("hybrid policy version cannot be negative")
        self.policy_version = max(self.policy_version, version)

    def begin_episode(self, episode_id: str) -> None:
        super().begin_episode(episode_id)
        self._hybrid_steps = []
        self._terminal_observed = False
        self._encounter_id = 0
        self._was_in_combat = False
        self._open_revivals_delta = 0
        self._coverage_eligible_decisions = 0
        self._coverage_exploratory_decisions = 0
        self._coverage_branch_counts = {}

    def _activate_domain_epsilon(self, observation: Mapping[str, Any]) -> None:
        combat = observation.get("combat")
        domain: Literal["macro", "combat"] = (
            "combat" if isinstance(combat, Mapping) and combat.get("in_progress") is True else "macro"
        )
        self.epsilon = (
            0.0 if self.evaluation_ownership else self.exploration[domain].value(self.domain_decisions[domain])
        )

    def _prepare_semantic_exploration(
        self,
        *,
        observation: Mapping[str, Any],
        decision: ForwardDecision,
        candidates: Sequence[SemanticCandidate],
        control_domain: Literal["macro", "combat"],
    ) -> None:
        del observation
        if self.evaluation_ownership or control_domain != "macro" or decision.surface == "selection":
            return
        branches = {candidate.branch for candidate in candidates}
        if len(branches) <= 1:
            return
        self._coverage_eligible_decisions += 1
        floor = self.factual_coverage.value(self.domain_decisions["macro"])
        if floor > self.epsilon:
            self.epsilon = floor

    def _record_semantic_choice(
        self,
        *,
        observation: Mapping[str, Any],
        decision: ForwardDecision,
        candidates: Sequence[SemanticCandidate],
        selected_index: int,
        control_domain: Literal["macro", "combat"],
    ) -> None:
        del observation
        if self.evaluation_ownership or control_domain != "macro" or decision.surface == "selection":
            return
        branches = {candidate.branch for candidate in candidates}
        if len(branches) <= 1:
            return
        branch = candidates[selected_index].branch
        key = f"{decision.surface}:{branch}"
        self._coverage_branch_counts[key] = self._coverage_branch_counts.get(key, 0) + 1
        if self.last_choice_exploratory:
            self._coverage_exploratory_decisions += 1

    def observe_observation(self, observation: Mapping[str, Any]) -> None:
        combat = observation.get("combat")
        active = isinstance(combat, Mapping) and combat.get("in_progress") is True
        if active and not self._was_in_combat:
            self._encounter_id += 1
        self._was_in_combat = active

    def observe_step(
        self,
        *,
        reward: float,
        floor: int | None,
        terminal: bool,
        revivals_delta: int = 0,
    ) -> None:
        if isinstance(revivals_delta, bool) or not isinstance(revivals_delta, int) or revivals_delta < 0:
            raise ValueError("hybrid revivals_delta must be a non-negative integer")
        if self._open is not None:
            self._open_revivals_delta += revivals_delta
        super().observe_step(reward=reward, floor=floor, terminal=terminal)
        self._terminal_observed = self._terminal_observed or bool(terminal)

    def _open_transition(
        self,
        *,
        snapshot: Any,
        action_index: int,
        surface: str,
        branch: str,
        target_key: str | None,
        floor: int,
        recurrent_reset: bool,
        control_domain: Literal["macro", "combat"] | None = None,
    ) -> None:
        super()._open_transition(
            snapshot=snapshot,
            action_index=action_index,
            surface=surface,
            branch=branch,
            target_key=target_key,
            floor=floor,
            recurrent_reset=False,
            control_domain=control_domain,
        )
        self._open_policy_version = int(self.policy_version)
        self._open_revivals_delta = 0
        assert control_domain is not None
        self._open_encounter_id = self._encounter_id if control_domain == "combat" else None
        self.domain_decisions[control_domain] += 1

    def _close_open(
        self,
        *,
        terminal: bool,
        floor: int | None = None,
        bridge_snapshot: Any = None,
        bridge_value: float | None = None,
    ) -> None:
        del bridge_snapshot, bridge_value
        if self._open is None:
            return
        opened = self._open
        closing_floor = self._last_floor if floor is None else floor
        self._hybrid_steps.append(
            HybridStep(
                snapshot=opened.snapshot,
                action_index=opened.action_index,
                reward=opened.reward_accumulator,
                revivals_delta=self._open_revivals_delta,
                discount=decision_discount(
                    floor_before=opened.floor,
                    floor_after=max(closing_floor, opened.floor),
                    terminal=terminal,
                ),
                terminal=terminal,
                surface=opened.surface,
                branch=opened.branch,
                control_domain=opened.control_domain,
                encounter_id=self._open_encounter_id,
                target_key=opened.target_key,
                behavior_epsilon=opened.behavior_epsilon,
                policy_version=self._open_policy_version,
            )
        )
        self._open = None
        self._open_encounter_id = None
        self._open_revivals_delta = 0

    def finish_run(
        self,
        run_id: str | None = None,
        *,
        environment_steps: int = 0,
    ) -> HybridRun | None:
        """Finish the mixed-domain run only after a factual terminal."""

        if self._episode_replay_invalid:
            self._episode_id = None
            self._hybrid_steps = []
            self._open = None
            self._open_revivals_delta = 0
            self._pending_plan = None
            return None
        if not self._terminal_observed:
            # A collector/backend interruption is a censored suffix, not a
            # factual game terminal.  Completed earlier transitions do not
            # acquire a made-up zero bootstrap.
            self._episode_id = None
            self._hybrid_steps = []
            self._open = None
            self._open_revivals_delta = 0
            self._pending_plan = None
            return None
        self._close_open(terminal=True)
        if not self._hybrid_steps:
            self._episode_id = None
            return None
        run = HybridRun(
            run_id=run_id or self._episode_id or "hybrid-run",
            steps=tuple(self._hybrid_steps),
            environment_steps=environment_steps,
        )
        self._episode_id = None
        self._hybrid_steps = []
        return run

    def discard_run(self) -> None:
        """Discard an infrastructure-interrupted run without inventing a terminal."""

        self._episode_replay_invalid = True
        self.finish_run()
        self._terminal_observed = False
        self._open_encounter_id = None
        self._open_revivals_delta = 0
        self._was_in_combat = False

    def set_epsilon(self, domain: Literal["macro", "combat"], value: float) -> None:
        """Set the active domain's exploration just before a decision.

        The public method exists for :class:`DomainExplorationClock`; the
        branch-balanced sampler itself remains the reviewed authority code.
        """

        if not math.isfinite(value) or not 0.0 <= value <= 1.0:
            raise ValueError("domain epsilon must be in [0, 1]")
        current = self.exploration[domain]
        self.exploration[domain] = DomainExplorationSchedule(
            start=float(value),
            end=float(value),
            decay_decisions=current.decay_decisions,
        )

    def metrics(self) -> dict[str, Any]:
        return {
            **super().metrics(),
            "version": HYBRID_AUTHORITY_VERSION,
            "policy_version": self.policy_version,
            "domain_decisions": dict(self.domain_decisions),
            "domain_epsilon": {
                domain: schedule.value(self.domain_decisions[domain]) for domain, schedule in self.exploration.items()
            },
            "recorded_steps": len(self._hybrid_steps),
            "factual_coverage": {
                "epsilon": self.factual_coverage.value(self.domain_decisions["macro"]),
                "eligible_decisions": self._coverage_eligible_decisions,
                "exploratory_decisions": self._coverage_exploratory_decisions,
                "branch_counts": dict(self._coverage_branch_counts),
            },
            "last_decline": self.last_executor_failure,
        }


class HybridVTraceCollectionAuthority(HybridCollectionAuthority):
    """Stochastic semantic policy that streams fresh mixed-domain unrolls."""

    def __init__(
        self,
        *args: Any,
        unroll_length: int,
        unroll_sink: Callable[[HybridVTraceUnroll], None],
        **kwargs: Any,
    ) -> None:
        if isinstance(unroll_length, bool) or not isinstance(unroll_length, int):
            raise TypeError("hybrid V-trace unroll length must be an integer")
        if unroll_length <= 0:
            raise ValueError("hybrid V-trace unroll length must be positive")
        super().__init__(*args, **kwargs)
        self.unroll_length = unroll_length
        self.unroll_sink = unroll_sink
        self.last_behavior_log_probability = 0.0
        self.last_model_log_probability = 0.0
        self.last_policy_probabilities: np.typing.NDArray[np.float64] | None = None
        self._last_hidden_before: Tensor | None = None
        self._last_semantic_snapshot: Any = None
        self._open_vtrace_initial_state: np.typing.NDArray[np.float32] | None = None
        self._open_vtrace_behavior_log_probability = 0.0
        self._vtrace_pending: list[HybridVTraceStep] = []
        self._vtrace_initial_state: np.typing.NDArray[np.float32] | None = None
        self._vtrace_start_decision = 0
        self._vtrace_decisions = 0
        self._vtrace_emitted = 0

    def begin_episode(self, episode_id: str) -> None:
        super().begin_episode(episode_id)
        self._vtrace_pending = []
        self._vtrace_initial_state = None
        self._open_vtrace_initial_state = None
        self._vtrace_start_decision = 0
        self._vtrace_decisions = 0
        self.last_behavior_log_probability = 0.0
        self.last_model_log_probability = 0.0
        self.last_policy_probabilities = None

    def _q_values(self, snapshot: Any) -> Tensor:
        hidden_before = self._hidden
        values = super()._q_values(snapshot)
        if not torch.is_tensor(hidden_before):
            raise TypeError("hybrid V-trace recurrent state must be a tensor")
        self._last_hidden_before = hidden_before.detach().clone()
        self._last_semantic_snapshot = snapshot
        return values

    def _semantic_choice(
        self,
        *,
        candidates: Sequence[SemanticCandidate],
        snapshot: Any,
    ) -> int:
        logits = self._q_values(snapshot)
        values = logits.detach().double().cpu().numpy()
        if values.shape != (len(candidates),) or not bool(np.isfinite(values).all()):
            raise FloatingPointError("hybrid V-trace policy logits are non-finite")
        shifted = values - float(values.max())
        policy = np.exp(shifted)
        policy /= float(policy.sum())
        branches = sorted({candidate.branch for candidate in candidates})
        support = np.zeros_like(policy)
        for branch in branches:
            members = [index for index, item in enumerate(candidates) if item.branch == branch]
            support[members] = 1.0 / (len(branches) * len(members))
        behavior = (1.0 - self.epsilon) * policy + self.epsilon * support

        self.last_q_values = values.astype(np.float32)
        self.last_policy_probabilities = policy.copy()
        self.last_decision_value = float(values.max())
        if self.evaluation_ownership:
            selected = int(np.argmax(values))
            self.last_choice_exploratory = False
            self.last_behavior_log_probability = 0.0
        elif self._rng.random() < self.epsilon:
            self.last_choice_exploratory = True
            branch = branches[int(self._rng.integers(len(branches)))]
            pool = [index for index, item in enumerate(candidates) if item.branch == branch]
            selected = int(pool[int(self._rng.integers(len(pool)))])
            self.last_behavior_log_probability = float(math.log(behavior[selected]))
        else:
            self.last_choice_exploratory = False
            selected = int(self._rng.choice(len(candidates), p=policy))
            self.last_behavior_log_probability = float(math.log(behavior[selected]))
        self.last_model_log_probability = float(math.log(policy[selected]))
        return selected

    def _open_transition(
        self,
        *,
        snapshot: Any,
        action_index: int,
        surface: str,
        branch: str,
        target_key: str | None,
        floor: int,
        recurrent_reset: bool,
        control_domain: Literal["macro", "combat"] | None = None,
    ) -> None:
        super()._open_transition(
            snapshot=snapshot,
            action_index=action_index,
            surface=surface,
            branch=branch,
            target_key=target_key,
            floor=floor,
            recurrent_reset=recurrent_reset,
            control_domain=control_domain,
        )
        if self._last_hidden_before is None:
            raise RuntimeError("hybrid V-trace choice has no initial recurrent state")
        initial = self._last_hidden_before.detach().float().cpu()
        if initial.ndim == 2 and initial.shape[0] == 1:
            initial = initial[0]
        self._open_vtrace_initial_state = initial.numpy().copy()
        self._open_vtrace_behavior_log_probability = self.last_behavior_log_probability

    def _emit_pending(self, *, bootstrap_snapshot: Any = None) -> None:
        if not self._vtrace_pending:
            return
        if self._vtrace_initial_state is None:
            raise RuntimeError("hybrid V-trace pending unroll lost its initial state")
        unroll = HybridVTraceUnroll(
            episode_id=self._episode_id or "hybrid-vtrace",
            start_decision=self._vtrace_start_decision,
            policy_version=self._vtrace_pending[0].policy_version,
            initial_recurrent_state=self._vtrace_initial_state,
            steps=tuple(self._vtrace_pending),
            bootstrap_snapshot=bootstrap_snapshot,
        )
        self.unroll_sink(unroll)
        self._vtrace_start_decision += len(self._vtrace_pending)
        self._vtrace_emitted += 1
        self._vtrace_pending = []
        self._vtrace_initial_state = None

    def _close_open(
        self,
        *,
        terminal: bool,
        floor: int | None = None,
        bridge_snapshot: Any = None,
        bridge_value: float | None = None,
    ) -> None:
        if self._open is None:
            return
        initial = self._open_vtrace_initial_state
        behavior_log_probability = self._open_vtrace_behavior_log_probability
        before = len(self._hybrid_steps)
        super()._close_open(
            terminal=terminal,
            floor=floor,
            bridge_snapshot=bridge_snapshot,
            bridge_value=bridge_value,
        )
        if len(self._hybrid_steps) != before + 1 or initial is None:
            raise RuntimeError("hybrid V-trace transition close lost its factual step")
        factual = self._hybrid_steps[-1]
        if not self._vtrace_pending:
            self._vtrace_initial_state = initial
        self._vtrace_pending.append(
            HybridVTraceStep(
                snapshot=factual.snapshot,
                action_index=factual.action_index,
                behavior_log_probability=behavior_log_probability,
                reward=factual.reward,
                discount=factual.discount,
                terminal=factual.terminal,
                surface=factual.surface,
                branch=factual.branch,
                control_domain=factual.control_domain,
                policy_version=factual.policy_version,
            )
        )
        self._vtrace_decisions += 1
        self._open_vtrace_initial_state = None
        if terminal:
            self._emit_pending(bootstrap_snapshot=None)
        elif len(self._vtrace_pending) >= self.unroll_length:
            if self._last_semantic_snapshot is None:
                raise RuntimeError("hybrid V-trace unroll has no bootstrap snapshot")
            self._emit_pending(bootstrap_snapshot=self._last_semantic_snapshot)

    def finish_run(
        self,
        run_id: str | None = None,
        *,
        environment_steps: int = 0,
    ) -> HybridRun | None:
        run = super().finish_run(run_id, environment_steps=environment_steps)
        if run is None:
            self._vtrace_pending = []
            self._vtrace_initial_state = None
        return run

    def metrics(self) -> dict[str, Any]:
        return {
            **super().metrics(),
            "version": HYBRID_VTRACE_AUTHORITY_VERSION,
            "vtrace_unrolls_emitted": self._vtrace_emitted,
            "vtrace_decisions": self._vtrace_decisions,
            "vtrace_pending_steps": len(self._vtrace_pending),
        }


__all__ = [
    "HYBRID_AUTHORITY_VERSION",
    "HYBRID_VTRACE_AUTHORITY_VERSION",
    "DomainExplorationSchedule",
    "FactualCoverageSchedule",
    "HybridCollectionAuthority",
    "HybridVTraceCollectionAuthority",
]
