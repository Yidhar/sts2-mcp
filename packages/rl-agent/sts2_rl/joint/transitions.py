"""Unified factual decision graph for the hybrid candidate-Q controller.

The old stage-two pipeline projected one physical run into two independent
episodes and joined them with a learned scalar bridge.  A :class:`HybridRun`
keeps the original ordering instead: every meaningful semantic decision is a
node, combat and macro decisions may alternate, and rewards/discounts are the
facts observed until the next node.  Domain views are indices over this one
store, never copied trajectories with different returns.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Final, Literal

from sts2_rl.encoding import EncodedDecisionSnapshot

HYBRID_TRANSITION_VERSION: Final = "sts2-hybrid-transition-v2"


@dataclass(frozen=True, slots=True)
class HybridStep:
    snapshot: EncodedDecisionSnapshot
    action_index: int
    # Stable factual reward.  Transition v2 explicitly excludes revival
    # preference; learner v3 applies one absolute price to ``revivals_delta``.
    reward: float
    revivals_delta: int
    discount: float
    terminal: bool
    surface: str
    branch: str
    control_domain: Literal["macro", "combat"]
    encounter_id: int | None = None
    target_key: str | None = None
    behavior_epsilon: float = 0.0
    policy_version: int = 0
    version: str = HYBRID_TRANSITION_VERSION

    def __post_init__(self) -> None:
        if not isinstance(self.snapshot, EncodedDecisionSnapshot):
            raise TypeError("hybrid step snapshot has the wrong type")
        if isinstance(self.action_index, bool) or not isinstance(self.action_index, int):
            raise TypeError("hybrid step action_index must be an integer")
        if not 0 <= self.action_index < len(self.snapshot.action_mask):
            raise ValueError("hybrid step action_index is outside its candidate set")
        if not bool(self.snapshot.action_mask[self.action_index]):
            raise ValueError("hybrid step executed an action the mask forbids")
        if not math.isfinite(self.reward):
            raise ValueError("hybrid step reward must be finite")
        if isinstance(self.revivals_delta, bool) or not isinstance(self.revivals_delta, int) or self.revivals_delta < 0:
            raise ValueError("hybrid step revivals_delta must be a non-negative integer")
        if not math.isfinite(self.discount) or not 0.0 <= self.discount <= 1.0:
            raise ValueError("hybrid step discount must be in [0, 1]")
        if self.terminal and self.discount != 0.0:
            raise ValueError("terminal hybrid steps never bootstrap")
        if not self.surface or not self.branch:
            raise ValueError("hybrid step requires surface and branch")
        if self.control_domain not in {"macro", "combat"}:
            raise ValueError("hybrid step control domain must be macro or combat")
        if self.control_domain == "combat" and self.encounter_id is None:
            raise ValueError("combat hybrid step requires an encounter id")
        if self.encounter_id is not None and self.encounter_id <= 0:
            raise ValueError("hybrid encounter id must be positive")
        if self.control_domain == "macro" and self.encounter_id is not None:
            raise ValueError("macro hybrid step cannot carry an encounter id")
        if not math.isfinite(self.behavior_epsilon) or not 0.0 <= self.behavior_epsilon <= 1.0:
            raise ValueError("hybrid behavior_epsilon must be in [0, 1]")
        if isinstance(self.policy_version, bool) or self.policy_version < 0:
            raise ValueError("hybrid policy_version must be non-negative")
        if self.version != HYBRID_TRANSITION_VERSION:
            raise ValueError("hybrid step transition version differs")


@dataclass(frozen=True, slots=True)
class HybridRun:
    run_id: str
    steps: tuple[HybridStep, ...]
    data_partition: str = "training"
    environment_steps: int = 0

    def __post_init__(self) -> None:
        if not self.run_id:
            raise ValueError("hybrid run requires an id")
        if not self.steps:
            raise ValueError("hybrid run requires at least one decision")
        if self.data_partition != "training":
            raise ValueError("hybrid replay stores training runs only")
        if isinstance(self.environment_steps, bool) or self.environment_steps < 0:
            raise ValueError("hybrid run environment_steps must be non-negative")
        if not self.steps[-1].terminal:
            raise ValueError("hybrid run must end at the factual run terminal")
        if any(step.terminal for step in self.steps[:-1]):
            raise ValueError("terminal hybrid step must be final")
        if any(step.version != HYBRID_TRANSITION_VERSION for step in self.steps):
            raise ValueError("hybrid run contains an incompatible transition")

    def indices(self, domain: Literal["macro", "combat"]) -> tuple[int, ...]:
        return tuple(index for index, step in enumerate(self.steps) if step.control_domain == domain)

    @property
    def macro_steps(self) -> int:
        return sum(step.control_domain == "macro" for step in self.steps)

    @property
    def combat_steps(self) -> int:
        return len(self.steps) - self.macro_steps


def next_domain_index(
    run: HybridRun,
    start: int,
    domain: Literal["macro", "combat"],
) -> int | None:
    """Return the next state in ``domain`` after ``start``."""

    for index in range(start + 1, len(run.steps)):
        if run.steps[index].control_domain == domain:
            return index
    return None


__all__ = [
    "HYBRID_TRANSITION_VERSION",
    "HybridRun",
    "HybridStep",
    "next_domain_index",
]
