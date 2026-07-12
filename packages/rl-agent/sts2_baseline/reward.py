"""Fixed normalized reward for the restarted baseline.

There is deliberately one immutable specification.  Changing any coefficient
requires a new version in source and therefore a new replay/checkpoint lineage.
No legacy objective vector, settlement backfill, or backend-computed reward is
accepted by this API.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass, field
from typing import Any, Literal

from .transition import BaselineTransition, PotentialState

RewardObjective = Literal["combat", "run"]


@dataclass(frozen=True, slots=True)
class BaselineRewardSpec:
    """The immutable ``sts2-baseline-reward-v1`` specification."""

    version: str = field(default="sts2-baseline-reward-v1", init=False)
    discount: float = field(default=0.997, init=False)
    combat_win: float = field(default=1.0, init=False)
    combat_loss: float = field(default=-1.0, init=False)
    run_win: float = field(default=1.0, init=False)
    run_loss: float = field(default=-1.0, init=False)
    collector_horizon_terminal_reward: float = field(default=0.0, init=False)
    transport_truncation_policy: str = field(default="discard", init=False)
    combat_player_hp_potential_weight: float = field(default=0.10, init=False)
    combat_enemy_progress_potential_weight: float = field(default=0.10, init=False)
    run_player_hp_potential_weight: float = field(default=0.05, init=False)
    run_enemy_progress_potential_weight: float = field(default=0.05, init=False)
    run_progress_potential_weight: float = field(default=0.10, init=False)
    dense_reward_abs_cap: float = field(default=0.25, init=False)


BASELINE_REWARD_SPEC = BaselineRewardSpec()


@dataclass(frozen=True, slots=True)
class BaselineTransitionProjectionSpec:
    """Identity of the raw-fact projection consumed by the reward calculator.

    Reward compatibility is wider than the coefficients above: floor
    normalization, terminal-result projection and truncation handling decide
    which state and outcome reach the calculator. Keeping them in the same
    fingerprint prevents replay/checkpoint reuse after projection drift.
    """

    version: str = field(default="sts2-baseline-transition-projection-v1", init=False)
    run_progress_floor_cap: float = field(default=60.0, init=False)
    combat_result_none: Literal["none"] = field(default="none", init=False)
    combat_result_victory: Literal["win"] = field(default="win", init=False)
    combat_result_defeat: Literal["loss"] = field(default="loss", init=False)
    combat_result_escaped: Literal["loss"] = field(default="loss", init=False)
    terminal_outcome_source: str = field(
        default="typed_transition_facts.combat_result+environment_result.terminated",
        init=False,
    )
    require_terminal_reason_equality: bool = field(default=True, init=False)
    terminal_missing_player_policy: str = field(
        default="allow_zero_pair_only_when_observation.terminated=true",
        init=False,
    )
    collector_horizon_truncation_kind: str = field(
        default="collector_horizon",
        init=False,
    )
    transport_truncation_policy: str = field(default="discard", init=False)

    def combat_result_map(self) -> dict[str, Literal["none", "win", "loss"]]:
        return {
            "none": self.combat_result_none,
            "victory": self.combat_result_victory,
            "defeat": self.combat_result_defeat,
            "escaped": self.combat_result_escaped,
        }


BASELINE_TRANSITION_PROJECTION_SPEC = BaselineTransitionProjectionSpec()


def baseline_reward_identity() -> dict[str, Any]:
    """Return the canonical calculator and transition-projection identity."""

    identity: dict[str, Any] = {
        "version": (
            f"{BASELINE_REWARD_SPEC.version}+"
            f"{BASELINE_TRANSITION_PROJECTION_SPEC.version}"
        ),
        "calculator": asdict(BASELINE_REWARD_SPEC),
        "transition_projection": asdict(BASELINE_TRANSITION_PROJECTION_SPEC),
    }
    serialized = json.dumps(identity, sort_keys=True, separators=(",", ":"))
    identity["fingerprint"] = serialized
    identity["fingerprint_sha256"] = hashlib.sha256(serialized.encode("utf-8")).hexdigest()
    return identity


@dataclass(frozen=True, slots=True)
class RewardBreakdown:
    total: float
    terminal: float
    potential: float
    before_potential: float
    after_potential: float
    objective: RewardObjective
    spec_version: str


class BaselineRewardCalculator:
    """Evaluate the one fixed reward spec for combat or run learning."""

    def __init__(self, objective: RewardObjective) -> None:
        if objective not in {"combat", "run"}:
            raise ValueError("reward objective must be 'combat' or 'run'")
        self.objective: RewardObjective = objective
        self.spec = BASELINE_REWARD_SPEC

    def _potential(self, state: PotentialState) -> float:
        if self.objective == "combat":
            return (
                self.spec.combat_player_hp_potential_weight * state.player_hp_ratio
                + self.spec.combat_enemy_progress_potential_weight * state.enemy_progress_ratio
            )
        return (
            self.spec.run_player_hp_potential_weight * state.player_hp_ratio
            + self.spec.run_enemy_progress_potential_weight * state.enemy_progress_ratio
            + self.spec.run_progress_potential_weight * float(state.run_progress)
        )

    def _terminal_reward(self, transition: BaselineTransition) -> tuple[float, bool]:
        if transition.truncated:
            truncation_kind = transition.metadata.get("truncation_kind")
            if truncation_kind == "collector_horizon":
                # A collector horizon is a censored continuation, not evidence
                # that the player lost.  The current v1 learner uses a zero
                # bootstrap at this boundary, but must never manufacture the
                # game-loss terminal reward previously assigned to every
                # transport truncation.
                return 0.0, False
            raise ValueError(
                "truncated transitions require truncation_kind='collector_horizon'; "
                "transport/outcome-unknown truncations must be discarded"
            )
        result = transition.combat_result if self.objective == "combat" else transition.run_result
        if result == "win":
            return (self.spec.combat_win if self.objective == "combat" else self.spec.run_win), True
        if result == "loss":
            return (self.spec.combat_loss if self.objective == "combat" else self.spec.run_loss), True
        return 0.0, False

    def evaluate(self, transition: BaselineTransition) -> RewardBreakdown:
        terminal_reward, task_terminal = self._terminal_reward(transition)
        before_potential = self._potential(transition.before)
        # A finite-horizon potential must be zero at the task terminal.  This
        # keeps shaping telescoping and prevents terminal-state HP scale from
        # becoming an extra outcome reward.
        after_potential = 0.0 if task_terminal else self._potential(transition.after)
        potential_reward = self.spec.discount * after_potential - before_potential
        potential_reward = min(
            max(potential_reward, -self.spec.dense_reward_abs_cap),
            self.spec.dense_reward_abs_cap,
        )
        return RewardBreakdown(
            total=float(terminal_reward + potential_reward),
            terminal=float(terminal_reward),
            potential=float(potential_reward),
            before_potential=float(before_potential),
            after_potential=float(after_potential),
            objective=self.objective,
            spec_version=self.spec.version,
        )
