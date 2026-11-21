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
import torch
from torch import Tensor

from sts2_rl.encoding import EncodedDecisionSnapshot, EncodedWorldSnapshot
from sts2_rl.models.grounded_candidate import COMBAT_DOMAIN_ID
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


def macro_mixture_log_probs(
    model_log_probs: Tensor,
    snapshot: EncodedDecisionSnapshot,
    epsilon: float,
) -> Tensor:
    """Compute exact branch-uniform exploration mixture log-probabilities.

    When ``epsilon == 0.0``, ``snapshot.domain_id == COMBAT_DOMAIN_ID``, or
    fewer than two legal actions exist, returns ``model_log_probs`` unchanged.
    """
    if isinstance(epsilon, bool) or not math.isfinite(epsilon) or not 0.0 <= epsilon < 1.0:
        raise ValueError("macro_branch_epsilon must be finite and in [0.0, 1.0)")
    if epsilon == 0.0 or snapshot.domain_id == COMBAT_DOMAIN_ID:
        return model_log_probs
    mask_np = np.asarray(snapshot.action_mask, dtype=np.bool_)
    legal_count = int(mask_np.sum())
    if legal_count <= 1:
        return model_log_probs
    mask = torch.tensor(mask_np.tolist(), device=model_log_probs.device, dtype=torch.bool)
    active_len = int(mask.numel())
    branch_ids_np = np.asarray(snapshot.candidates.ids[:active_len, 13], dtype=np.int64)
    legal_branches = branch_ids_np[mask_np]
    unique_branches, counts = np.unique(legal_branches, return_counts=True)
    branch_count = len(unique_branches)
    branch_sizes = dict(zip(unique_branches.tolist(), counts.tolist(), strict=True))
    uniform_np = np.zeros((active_len,), dtype=np.float64)
    for idx in range(active_len):
        if mask_np[idx]:
            uniform_np[idx] = 1.0 / float(branch_count * branch_sizes[int(branch_ids_np[idx])])
    uniform = torch.tensor(uniform_np.tolist(), device=model_log_probs.device, dtype=torch.float64)

    head = model_log_probs[:active_len].double().masked_fill(~mask, -torch.inf)
    head = head - torch.logsumexp(head, dim=0)
    model_probs = head.exp().masked_fill(~mask, 0.0)
    mixed_probs = (1.0 - float(epsilon)) * model_probs + float(epsilon) * uniform
    mixed_logs = torch.full((active_len,), -torch.inf, device=model_log_probs.device, dtype=torch.float64)
    mixed_logs[mask] = torch.log(mixed_probs[mask])
    mixed_logs[mask] = mixed_logs[mask] - torch.logsumexp(mixed_logs[mask], dim=0)
    converted = mixed_logs.to(dtype=model_log_probs.dtype)
    if model_log_probs.numel() > active_len:
        tail = torch.full(
            (model_log_probs.numel() - active_len,),
            -torch.inf,
            device=model_log_probs.device,
            dtype=model_log_probs.dtype,
        )
        return torch.cat([converted, tail], dim=0)
    return converted


@dataclass(frozen=True, slots=True)
class MacroAfterstateNode:
    """Verified deterministic post-state worlds for legal candidates at one macro step."""

    step_index: int
    surface: str
    candidate_branches: tuple[str, ...]
    post_worlds: tuple[EncodedWorldSnapshot, ...]
    candidate_indices: tuple[int, ...] | None = None
    is_partial: bool = False
    crn_candidate_indices: tuple[int, ...] | None = None
    crn_candidate_returns: tuple[float, ...] | None = None

    def __post_init__(self) -> None:
        if isinstance(self.step_index, bool) or not isinstance(self.step_index, int) or self.step_index < 0:
            raise ValueError("afterstate step_index must be a non-negative integer")
        if not isinstance(self.surface, str) or not self.surface:
            raise ValueError("afterstate surface must be non-empty")
        if not isinstance(self.is_partial, bool):
            raise ValueError("afterstate is_partial must be boolean")
        if len(self.post_worlds) < 2 or len(self.candidate_branches) != len(self.post_worlds):
            raise ValueError("afterstate node requires complete multi-candidate post_worlds and branches")
        if any(not isinstance(branch, str) or not branch for branch in self.candidate_branches):
            raise ValueError("afterstate candidate_branches must be non-empty strings")
        if not all(isinstance(world, EncodedWorldSnapshot) for world in self.post_worlds):
            raise ValueError("afterstate post_worlds must be EncodedWorldSnapshot instances")
        if self.candidate_indices is None:
            if self.is_partial:
                raise ValueError("partial afterstate node requires explicit candidate_indices")
            object.__setattr__(self, "candidate_indices", tuple(range(len(self.post_worlds))))
        indices = self.candidate_indices
        assert indices is not None
        if len(indices) != len(self.post_worlds):
            raise ValueError("afterstate candidate_indices must match post_worlds length")
        prev_idx = -1
        for idx in indices:
            if isinstance(idx, bool) or not isinstance(idx, int) or idx <= prev_idx:
                raise ValueError("afterstate candidate_indices must be strictly increasing non-negative ints")
            prev_idx = idx
        if not self.is_partial and indices != tuple(range(len(self.post_worlds))):
            raise ValueError("non-partial afterstate node must cover contiguous 0..K-1 candidate indices")
        if (self.crn_candidate_indices is None) != (self.crn_candidate_returns is None):
            raise ValueError("crn_candidate_indices and crn_candidate_returns must both be set or both None")
        if self.crn_candidate_indices is not None:
            assert self.crn_candidate_returns is not None
            if len(self.crn_candidate_indices) < 2 or len(self.crn_candidate_indices) != len(
                self.crn_candidate_returns
            ):
                raise ValueError("crn_candidate_indices and crn_candidate_returns must match with length >= 2")
            allowed_idx = set(indices)
            prev_c = -1
            for c_idx in self.crn_candidate_indices:
                if isinstance(c_idx, bool) or not isinstance(c_idx, int) or c_idx <= prev_c or c_idx not in allowed_idx:
                    raise ValueError("crn_candidate_indices must be strictly increasing subset of candidate_indices")
                prev_c = c_idx
            for c_ret in self.crn_candidate_returns:
                if isinstance(c_ret, bool) or not isinstance(c_ret, int | float) or not math.isfinite(float(c_ret)):
                    raise ValueError("crn_candidate_returns must be finite numbers")

    @property
    def resolved_candidate_indices(self) -> tuple[int, ...]:
        assert self.candidate_indices is not None
        return self.candidate_indices

    def post_world_for_action(self, action_index: int) -> EncodedWorldSnapshot | None:
        indices = self.resolved_candidate_indices
        if len(indices) == len(self.post_worlds) and indices[-1] == len(self.post_worlds) - 1:
            if 0 <= action_index < len(self.post_worlds):
                return self.post_worlds[action_index]
            return None
        try:
            pos = indices.index(action_index)
        except ValueError:
            return None
        return self.post_worlds[pos]

    def storage_nbytes(self) -> int:
        seen: set[int] = set()
        total = 0
        for world in self.post_worlds:
            if id(world) not in seen:
                seen.add(id(world))
                total += world.storage_nbytes()
        return total


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
    afterstate_nodes: tuple[MacroAfterstateNode, ...] = ()
    macro_branch_epsilon: float = 0.0

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
        if (
            isinstance(self.macro_branch_epsilon, bool)
            or not math.isfinite(self.macro_branch_epsilon)
            or not 0.0 <= self.macro_branch_epsilon < 1.0
        ):
            raise ValueError("macro_branch_epsilon must be finite and in [0.0, 1.0)")
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
        previous_step_index = -1
        for node in self.afterstate_nodes:
            if not isinstance(node, MacroAfterstateNode):
                raise ValueError("afterstate_nodes must contain MacroAfterstateNode instances")
            node.__post_init__()
            if node.step_index <= previous_step_index or node.step_index >= len(self.steps):
                raise ValueError("afterstate_nodes must reference strictly increasing valid step indices")
            previous_step_index = node.step_index
            target_step = self.steps[node.step_index]
            if target_step.control_domain != "macro" or target_step.surface != node.surface:
                raise ValueError("afterstate node must align with a macro step of the same surface")
            if not bool(target_step.snapshot.action_mask.all()):
                raise ValueError("afterstate node requires an unmasked legal candidate snapshot")
            det_indices = node.resolved_candidate_indices
            if node.is_partial:
                if det_indices[-1] >= target_step.snapshot.candidate_count:
                    raise ValueError("partial afterstate node candidate_indices exceed step candidate_count")
                if target_step.action_index in det_indices:
                    det_pos = det_indices.index(target_step.action_index)
                    if target_step.branch != node.candidate_branches[det_pos]:
                        raise ValueError("partial afterstate node disagrees with selected action branch")
            elif (
                target_step.snapshot.candidate_count != len(node.post_worlds)
                or target_step.branch != node.candidate_branches[target_step.action_index]
            ):
                raise ValueError("afterstate node must cover the complete legal candidate set of its step")
            for world in node.post_worlds:
                world.validate(expected_config=first.config, expected_fingerprint=first.encoding_fingerprint)


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
    # A game can genuinely end before reaching the configured reward root.
    # Keep its natural evidence instead of censoring the worst failures.
    natural_only: tuple[MacroCreditTrajectory, ...] = ()

    def __post_init__(self) -> None:
        if self.version != MACRO_CREDIT_DATA_VERSION or not self.cohort_id or not (self.roots or self.natural_only):
            raise ValueError("macro cohort identity or roots are missing")
        if (
            isinstance(self.teacher_version, bool)
            or not isinstance(self.teacher_version, int)
            or self.teacher_version < 0
        ):
            raise ValueError("macro cohort teacher version must be a non-negative integer")
        if any(ref.teacher_version != self.teacher_version for ref in self.references):
            raise ValueError("macro cohort mixes teacher versions")
        if len({root.root_id for root in self.roots}) != len(self.roots):
            raise ValueError("macro cohort duplicates a root")
        if len({ref.seed for ref in self.references}) != len(self.references):
            raise ValueError("cohort roots must use distinct training seeds")
        for ref in self.natural_only:
            ref.__post_init__()
            if ref.ending != "factual_terminal" or ref.root_intervened or ref.prefix:
                raise ValueError("rootless natural data must be a complete, uninterrupted factual game")

    @property
    def references(self) -> tuple[MacroCreditTrajectory, ...]:
        return tuple(root.reference for root in self.roots) + self.natural_only

    def storage_nbytes(self) -> int:
        """Unique CPU array payload; shared prefixes are counted only once."""
        return trajectory_storage_nbytes((
            *(trajectory for root in self.roots for trajectory in (root.reference, *root.candidates)),
            *self.natural_only,
        ))


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
        for node in trajectory.afterstate_nodes:
            for world in node.post_worlds:
                if id(world) not in seen:
                    seen.add(id(world))
                    total += world.storage_nbytes()
    return total


__all__ = [
    "MACRO_CREDIT_DATA_VERSION",
    "MacroAfterstateNode",
    "MacroCounterfactualRoot",
    "MacroCreditCohort",
    "MacroCreditTrajectory",
    "MacroRewardFacts",
    "macro_mixture_log_probs",
    "snapshot_equal",
    "training_seed",
    "trajectory_storage_nbytes",
]
