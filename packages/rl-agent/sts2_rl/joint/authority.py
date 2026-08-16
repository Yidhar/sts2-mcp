"""Single collection authority for the hybrid combat/macro controller."""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any, Final, Literal

from sts2_rl.macro.authority import MacroCollectionAuthority
from sts2_rl.semantics.clock import decision_discount
from sts2_rl.semantics.forward import ForwardDecision, SemanticCandidate

from .transitions import HybridRun, HybridStep

HYBRID_AUTHORITY_VERSION: Final = "sts2-hybrid-authority-v3"


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


__all__ = [
    "HYBRID_AUTHORITY_VERSION",
    "DomainExplorationSchedule",
    "FactualCoverageSchedule",
    "HybridCollectionAuthority",
]
