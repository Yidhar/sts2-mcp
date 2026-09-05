"""Training-only CPU facts for one fresh-teacher macro-credit cohort.

This is not an experience replay buffer. A cohort is consumed by exactly one
optimizer update under the teacher version that collected it. Diagnostics,
greedy continuations, partial candidate sets, unknown truncations and saved
behavior hidden states are deliberately outside this data contract.
"""

from __future__ import annotations

import math
from collections.abc import Iterable
from dataclasses import dataclass
from typing import Final, Literal

import numpy as np

from sts2_rl.encoding import EncodedDecisionSnapshot
from sts2_rl.semantics.clock import decision_discount
from sts2_rl.training.seeding import SIGNED_INT32_MAX, TRAINING_SEED_PARITY

from .transitions import HybridVTraceStep

MACRO_CREDIT_DATA_VERSION: Final = "sts2-fresh-macro-credit-cohort-v1"


@dataclass(frozen=True, slots=True)
class MacroRewardFacts:
    """Edge-local native-prefix-discounted rewards; factual counters stay raw.

    The trace retains each unweighted native reward. These five components
    instead have already been folded in native order within a semantic edge;
    the step discount applies only to the next semantic edge, not again here.
    """

    terminal_reward: float
    progress_reward: float
    revival_penalty: float
    pace_penalty: float
    hp_loss_penalty: float
    revivals_used_delta: int
    player_hp_lost_delta: float
    floor_before: int
    floor_after: int
    native_steps: int

    def __post_init__(self) -> None:
        for name in (
            "terminal_reward",
            "progress_reward",
            "revival_penalty",
            "pace_penalty",
            "hp_loss_penalty",
            "player_hp_lost_delta",
        ):
            if not math.isfinite(getattr(self, name)):
                raise ValueError("macro reward facts must be finite")
        for name in ("revivals_used_delta", "floor_before", "floor_after", "native_steps"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int) or value < 0:
                raise ValueError(f"macro reward {name} must be a non-negative integer")
        if self.floor_after < self.floor_before:
            raise ValueError("macro reward durable floor cannot regress")
        if self.player_hp_lost_delta < 0 or self.native_steps == 0:
            raise ValueError("macro reward facts require real native steps and non-negative HP loss")

    @property
    def reward(self) -> float:
        return (
            self.terminal_reward
            + self.progress_reward
            + self.revival_penalty
            + self.pace_penalty
            + self.hp_loss_penalty
        )


def training_seed(seed: int) -> None:
    if isinstance(seed, bool) or not isinstance(seed, int) or not 0 <= seed <= SIGNED_INT32_MAX:
        raise ValueError("macro credit requires a non-negative signed-int32 seed")
    if seed % 2 != TRAINING_SEED_PARITY:
        raise ValueError("macro credit refuses held-out/diagnostic seeds")


def snapshot_equal(left: EncodedDecisionSnapshot, right: EncodedDecisionSnapshot) -> bool:
    """Exact sparse-input equality without inventing a second hash identity."""
    if left is right:
        return True
    for name in ("config", "version", "encoding_fingerprint", "domain_id", "macro_economic_surface_id"):
        if getattr(left, name) != getattr(right, name):
            return False
    for name in ("action_mask", "local_offsets"):
        if not np.array_equal(getattr(left, name), getattr(right, name)):
            return False
    for table_name in ("world", "candidates", "locals"):
        a, b = getattr(left, table_name), getattr(right, table_name)
        for name in ("ids", "feature_indptr", "feature_indices", "feature_values"):
            if not np.array_equal(getattr(a, name), getattr(b, name)):
                return False
    return True


@dataclass(frozen=True, slots=True)
class MacroCreditTrajectory:
    trajectory_id: str
    seed: int
    teacher_version: int
    prefix: tuple[EncodedDecisionSnapshot, ...]
    steps: tuple[HybridVTraceStep, ...]
    bootstrap_snapshot: EncodedDecisionSnapshot | None
    bootstrap_domain: Literal["macro", "combat"] | None
    ending: Literal["factual_terminal", "macro_horizon"]
    reward_facts: tuple[MacroRewardFacts, ...]
    # True only for an actual simulator intervention, whose first action has
    # behavior probability one. It never enters ordinary sampled-policy loss.
    root_intervened: bool = False
    data_partition: str = "training"
    continuation_policy: str = "frozen_stochastic"

    def __post_init__(self) -> None:
        training_seed(self.seed)
        if not self.trajectory_id:
            raise ValueError("macro trajectory needs an id")
        if (
            isinstance(self.teacher_version, bool)
            or not isinstance(self.teacher_version, int)
            or self.teacher_version < 0
        ):
            raise ValueError("macro trajectory needs a non-negative teacher version")
        if self.data_partition != "training" or self.continuation_policy != "frozen_stochastic":
            raise ValueError("only training-partition frozen stochastic continuations are learner inputs")
        if not isinstance(self.root_intervened, bool):
            raise ValueError("root_intervened must be boolean")
        if not self.steps or not all(isinstance(step, HybridVTraceStep) for step in self.steps):
            raise ValueError("macro trajectory requires semantic steps")
        if not all(isinstance(snapshot, EncodedDecisionSnapshot) for snapshot in self.prefix):
            raise ValueError("macro trajectory prefix must contain CPU semantic snapshots, not hidden states")
        if any(step.policy_version != self.teacher_version for step in self.steps):
            raise ValueError("macro trajectory mixes teacher versions")
        if any(step.terminal for step in self.steps[:-1]):
            raise ValueError("macro trajectory has an interior terminal")
        if len(self.reward_facts) != len(self.steps) or not all(
            isinstance(facts, MacroRewardFacts) for facts in self.reward_facts
        ):
            raise ValueError("every semantic edge must retain its factual reward components")
        for step, facts in zip(self.steps, self.reward_facts, strict=True):
            if not math.isclose(step.reward, facts.reward, rel_tol=1e-10, abs_tol=1e-12):
                raise ValueError("semantic reward differs from factual components")
            gamma = decision_discount(
                floor_before=facts.floor_before, floor_after=facts.floor_after, terminal=step.terminal
            )
            if not math.isclose(step.discount, gamma, rel_tol=0, abs_tol=1e-12):
                raise ValueError("semantic discount differs from its durable-floor edge")
        if self.root_intervened and (
            self.steps[0].control_domain != "macro" or self.steps[0].behavior_log_probability != 0.0
        ):
            raise ValueError("forced macro root must record its actual probability-one intervention")
        if self.ending == "factual_terminal":
            if not self.steps[-1].terminal or self.bootstrap_snapshot is not None or self.bootstrap_domain is not None:
                raise ValueError("factual terminal must cut bootstrap")
        elif self.ending == "macro_horizon":
            if self.steps[-1].terminal or self.bootstrap_snapshot is None or self.bootstrap_domain != "macro":
                raise ValueError("macro horizon needs the real next macro snapshot and nonzero bootstrap")
        else:
            raise ValueError("unknown/truncated/budget-exhausted trajectories are not training data")
        first = self.steps[0].snapshot
        for snapshot in (
            *self.prefix,
            *(step.snapshot for step in self.steps),
            *((self.bootstrap_snapshot,) if self.bootstrap_snapshot else ()),
        ):
            snapshot.validate(expected_config=first.config, expected_fingerprint=first.encoding_fingerprint)


@dataclass(frozen=True, slots=True)
class MacroCounterfactualRoot:
    root_id: str
    reference: MacroCreditTrajectory
    root_step_index: int
    # Ordered exactly as the authoritative executable semantic candidate set.
    candidates: tuple[MacroCreditTrajectory, ...]
    candidate_branches: tuple[str, ...]
    teacher_log_probabilities: tuple[float, ...]
    same_action_gate: str
    evidence_directory: str

    def __post_init__(self) -> None:
        if not self.root_id or not self.evidence_directory or self.same_action_gate != "matched":
            raise ValueError("counterfactual root requires its own successful same-action control evidence")
        if self.reference.prefix or self.reference.root_intervened:
            raise ValueError("natural reference must begin at run reset without an intervention")
        if (
            isinstance(self.root_step_index, bool)
            or not isinstance(self.root_step_index, int)
            or not 0 <= self.root_step_index < len(self.reference.steps)
        ):
            raise ValueError("root index is outside the natural reference")
        root = self.reference.steps[self.root_step_index]
        if root.control_domain != "macro":
            raise ValueError("counterfactual control is macro only")
        count = root.snapshot.candidate_count
        if count < 2 or not bool(root.snapshot.action_mask.all()):
            raise ValueError("root requires the complete executable semantic set")
        if any(
            len(items) != count for items in (self.candidates, self.candidate_branches, self.teacher_log_probabilities)
        ):
            raise ValueError("partial or duplicated candidate groups cannot be learned")
        if any(not branch for branch in self.candidate_branches):
            raise ValueError("each candidate needs its factual semantic branch")
        if any(not math.isfinite(value) or value > 1e-7 for value in self.teacher_log_probabilities):
            raise ValueError("root teacher probabilities must be finite legal log probabilities")
        total = sum(math.exp(value) for value in self.teacher_log_probabilities)
        if not math.isclose(total, 1.0, rel_tol=1e-5, abs_tol=1e-6):
            raise ValueError("root teacher probabilities are not normalized")
        if not math.isclose(
            root.behavior_log_probability, self.teacher_log_probabilities[root.action_index], rel_tol=0, abs_tol=1e-7
        ):
            raise ValueError("root teacher prior differs from natural behavior probability")
        prefix = tuple(step.snapshot for step in self.reference.steps[: self.root_step_index])
        for index, trajectory in enumerate(self.candidates):
            if trajectory.seed != self.reference.seed or trajectory.teacher_version != self.reference.teacher_version:
                raise ValueError("root candidate seed/teacher differs")
            first = trajectory.steps[0]
            if first.action_index != index or first.branch != self.candidate_branches[index]:
                raise ValueError("candidate trajectory does not execute its declared root action")
            if trajectory.root_intervened != (index != root.action_index):
                raise ValueError("only the natural reference action may be reused without an intervention")
            if not snapshot_equal(first.snapshot, root.snapshot):
                raise ValueError("candidate trajectory root input differs")
            if len(trajectory.prefix) != len(prefix) or any(
                not snapshot_equal(a, b) for a, b in zip(prefix, trajectory.prefix, strict=True)
            ):
                raise ValueError("candidate trajectory must reconstruct the exact same semantic prefix")
            if index == root.action_index:
                expected_steps = self.reference.steps[self.root_step_index :]
                if (
                    len(trajectory.steps) != len(expected_steps)
                    or trajectory.reward_facts != self.reference.reward_facts[self.root_step_index :]
                ):
                    raise ValueError("natural candidate must reuse the reference suffix facts")
                for actual, expected in zip(trajectory.steps, expected_steps, strict=True):
                    if not snapshot_equal(actual.snapshot, expected.snapshot) or any(
                        getattr(actual, name) != getattr(expected, name)
                        for name in (
                            "action_index",
                            "behavior_log_probability",
                            "reward",
                            "discount",
                            "terminal",
                            "surface",
                            "branch",
                            "control_domain",
                            "policy_version",
                            "version",
                        )
                    ):
                        raise ValueError("natural candidate must reuse the reference suffix steps")
                if (
                    trajectory.ending != self.reference.ending
                    or trajectory.bootstrap_domain != self.reference.bootstrap_domain
                ):
                    raise ValueError("natural candidate must reuse the reference ending")
                a, b = trajectory.bootstrap_snapshot, self.reference.bootstrap_snapshot
                if (a is None) != (b is None) or (a is not None and b is not None and not snapshot_equal(a, b)):
                    raise ValueError("natural candidate must reuse the reference bootstrap")


@dataclass(frozen=True, slots=True)
class MacroCreditCohort:
    cohort_id: str
    teacher_version: int
    roots: tuple[MacroCounterfactualRoot, ...]
    version: str = MACRO_CREDIT_DATA_VERSION

    def __post_init__(self) -> None:
        if self.version != MACRO_CREDIT_DATA_VERSION or not self.cohort_id or not self.roots:
            raise ValueError("macro cohort identity or roots are missing")
        if (
            isinstance(self.teacher_version, bool)
            or not isinstance(self.teacher_version, int)
            or self.teacher_version < 0
        ):
            raise ValueError("macro cohort teacher version must be a non-negative integer")
        if any(root.reference.teacher_version != self.teacher_version for root in self.roots):
            raise ValueError("macro cohort mixes teacher versions")
        if len({root.root_id for root in self.roots}) != len(self.roots):
            raise ValueError("macro cohort duplicates a root")
        if len({root.reference.seed for root in self.roots}) != len(self.roots):
            raise ValueError("cohort roots must use distinct training seeds")

    def storage_nbytes(self) -> int:
        """Unique CPU array payload; shared prefixes are counted only once."""
        return trajectory_storage_nbytes(
            trajectory for root in self.roots for trajectory in (root.reference, *root.candidates)
        )


def trajectory_storage_nbytes(trajectories: Iterable[MacroCreditTrajectory]) -> int:
    """Unique retained sparse arrays, not headers, model memory or process RSS."""
    seen: set[int] = set()
    total = 0
    for trajectory in trajectories:
        snapshots = (*trajectory.prefix, *(step.snapshot for step in trajectory.steps))
        if trajectory.bootstrap_snapshot is not None:
            snapshots = (*snapshots, trajectory.bootstrap_snapshot)
        for snapshot in snapshots:
            if id(snapshot) not in seen:
                seen.add(id(snapshot))
                total += snapshot.storage_nbytes()
    return total


__all__ = [
    "MACRO_CREDIT_DATA_VERSION",
    "MacroCounterfactualRoot",
    "MacroCreditCohort",
    "MacroCreditTrajectory",
    "MacroRewardFacts",
    "snapshot_equal",
    "training_seed",
    "trajectory_storage_nbytes",
]
