"""Fresh stochastic-teacher collection, reusing the verified B1 simulator path.

Only this explicit training collector can produce MacroCounterfactualRoot.
The diagnostic CLI remains greedy, read-only with respect to training, and its
finite-prefix return summaries are never imported as learning labels.
"""

from __future__ import annotations

import json
import math
import time
from collections.abc import Callable, Mapping, Sequence
from contextlib import ExitStack
from dataclasses import replace
from pathlib import Path
from typing import Any, Literal, TextIO

import numpy as np
import torch

from sts2_rl.artifacts import resolve_artifact_path
from sts2_rl.contracts import EnvironmentBackend
from sts2_rl.encoding import EncodedDecisionSnapshot, GroundedObservationEncoder
from sts2_rl.macro_branch_diagnostic import BranchAuthority, BranchLimits, _MacroHorizonReached, run_branch_pass
from sts2_rl.macro_root_diagnostic import (
    EvidenceTrace,
    FrozenPolicy,
    ProbeAuthority,
    ProbeBudgetExhausted,
    validate_course,
)
from sts2_rl.semantics.clock import decision_discount
from sts2_rl.semantics.forward import ForwardDecision, SemanticCandidate
from sts2_rl.training.config import TrainingConfig

from .macro_credit_data import (
    MacroCounterfactualRoot,
    MacroCreditTrajectory,
    MacroRewardFacts,
    snapshot_equal,
    training_seed,
    trajectory_storage_nbytes,
)
from .transitions import HybridVTraceStep

COMPONENTS = ("terminal_reward", "progress_reward", "revival_penalty", "pace_penalty", "hp_loss_penalty")


class StochasticBranchAuthority(BranchAuthority):
    """Exactly pi_teacher sampling; no epsilon mixture or hand-written bias."""

    policy_mode = "frozen_stochastic"

    def __init__(
        self,
        policy: FrozenPolicy,
        seed: int,
        limits: BranchLimits,
        forced_candidate: int | None,
        *,
        teacher_version: int,
        config: TrainingConfig,
    ) -> None:
        super().__init__(policy, seed, limits, forced_candidate)
        training_seed(seed)
        self.teacher_version = teacher_version
        self._sampling_rng = np.random.default_rng(np.random.SeedSequence([seed, 0x4D414352]))
        self._encoder = GroundedObservationEncoder(config.model.to_encoding_config())
        self.current_snapshot: EncodedDecisionSnapshot | None = None
        self.bootstrap_snapshot: EncodedDecisionSnapshot | None = None
        self.root_teacher_log_probs: tuple[float, ...] | None = None
        self._behavior_log_probability = 0.0

    def _prepare_semantic_exploration(
        self,
        *,
        observation: Mapping[str, Any],
        decision: ForwardDecision,
        candidates: Sequence[SemanticCandidate],
        control_domain: Literal["macro", "combat"],
    ) -> None:
        try:
            super()._prepare_semantic_exploration(
                observation=observation, decision=decision, candidates=candidates, control_domain=control_domain
            )
        except _MacroHorizonReached:
            self.bootstrap_snapshot = self._encoder.encode(
                observation, [candidate.semantic_action for candidate in candidates]
            ).snapshot
            raise

    def _semantic_choice(self, *, candidates: Sequence[SemanticCandidate], snapshot: EncodedDecisionSnapshot) -> int:
        # ProbeAuthority advances the production model's memory and records its
        # unmodified logits. Sampling changes only the authoritative choice.
        greedy = ProbeAuthority._semantic_choice(self, candidates=candidates, snapshot=snapshot)
        assert self.last_q_values is not None
        self.current_snapshot = snapshot
        logits = torch.tensor(self.last_q_values, dtype=torch.float64)
        log_probs = torch.log_softmax(logits, dim=0).numpy()
        probabilities = np.exp(log_probs)
        draw = float(self._sampling_rng.random())  # Burn one draw even at the forced root.
        selected = min(int(np.searchsorted(np.cumsum(probabilities), draw, side="right")), len(candidates) - 1)
        if self._current_is_root:
            self.greedy_root_index = greedy
            self.root_teacher_log_probs = tuple(float(value) for value in log_probs)
            if self.forced_candidate is not None:
                if isinstance(self.forced_candidate, bool) or not 0 <= self.forced_candidate < len(candidates):
                    raise ValueError("forced root action is not in the complete legal candidate set")
                selected = self.forced_candidate
        self._behavior_log_probability = (
            0.0 if self._current_is_root and self.forced_candidate is not None else float(log_probs[selected])
        )
        return selected

    def _record_semantic_choice(
        self,
        *,
        observation: Mapping[str, Any],
        decision: ForwardDecision,
        candidates: Sequence[SemanticCandidate],
        selected_index: int,
        control_domain: Literal["macro", "combat"],
    ) -> None:
        super()._record_semantic_choice(
            observation=observation,
            decision=decision,
            candidates=candidates,
            selected_index=selected_index,
            control_domain=control_domain,
        )
        assert self.event is not None
        self.event.update(
            behavior_log_probability=self._behavior_log_probability,
            teacher_version=self.teacher_version,
            continuation_policy=self.policy_mode,
            root_intervened=self._current_is_root and self.forced_candidate is not None,
        )


class _GuardedEvidenceTrace(EvidenceTrace):
    def __init__(self, *args: Any, resource_guard: Callable[[], None] | None = None, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self.resource_guard = resource_guard

    def record(self, kind: str, facts: Any, policy: Any = None) -> None:
        if self.resource_guard is not None:
            self.resource_guard()
        super().record(kind, facts, policy)


class TrainingEvidenceTrace(_GuardedEvidenceTrace):
    """Observe validated simulator events; retain CPU sparse inputs, not hidden."""

    def __init__(
        self,
        output: TextIO,
        *,
        authority: StochasticBranchAuthority,
        limits: BranchLimits,
        maximum_snapshot_bytes: int,
        resource_guard: Callable[[], None] | None = None,
    ) -> None:
        super().__init__(output, limits.trace_limits(), resource_guard=resource_guard)
        self.authority = authority
        self.maximum_snapshot_bytes = maximum_snapshot_bytes
        self.snapshot_bytes = 0
        self.steps: list[HybridVTraceStep] = []
        self.reward_facts: list[MacroRewardFacts] = []
        self.root_index: int | None = None
        self.pending: dict[str, Any] | None = None
        self.sums = dict.fromkeys(COMPONENTS, 0.0)
        self.revivals = 0
        self.hp_lost = 0.0
        self.native_steps = 0
        self.edge_weight = 1.0
        self.last_native_floor: int | None = None

    def _close_edge(self, floor: int, terminal: bool) -> None:
        if self.pending is None:
            return
        pending = self.pending
        discount = decision_discount(floor_before=pending["floor"], floor_after=floor, terminal=terminal)
        if self.last_native_floor != floor or not math.isclose(self.edge_weight, discount, rel_tol=0, abs_tol=1e-12):
            raise ValueError("native edge sequence disagrees with semantic durable-floor discount")
        facts = MacroRewardFacts(
            **self.sums,
            revivals_used_delta=self.revivals,
            player_hp_lost_delta=self.hp_lost,
            floor_before=pending["floor"],
            floor_after=floor,
            native_steps=self.native_steps,
        )
        self.steps.append(
            HybridVTraceStep(
                snapshot=pending["snapshot"],
                action_index=pending["selected_index"],
                behavior_log_probability=pending["behavior_log_probability"],
                reward=facts.reward,
                discount=discount,
                terminal=terminal,
                surface=pending["surface"],
                branch=pending["branch"],
                control_domain=pending["domain"],
                policy_version=self.authority.teacher_version,
            )
        )
        self.reward_facts.append(facts)
        self.pending = None
        self.sums = dict.fromkeys(COMPONENTS, 0.0)
        self.revivals = 0
        self.hp_lost = 0.0
        self.native_steps = 0
        self.edge_weight = 1.0
        self.last_native_floor = None

    def record(self, kind: str, facts: Any, policy: Any = None) -> None:
        super().record(kind, facts, policy)
        if kind == "decision" and facts["semantic"] is not None:
            event = facts["semantic"]
            floor = facts["state"]["observation"]["run"]["floor"]
            self._close_edge(floor, False)
            snapshot = self.authority.current_snapshot
            if snapshot is None or event.get("continuation_policy") != "frozen_stochastic":
                raise ValueError("training input was not collected by a frozen stochastic semantic teacher")
            self.snapshot_bytes += snapshot.storage_nbytes()
            if self.snapshot_bytes > self.maximum_snapshot_bytes:
                raise ProbeBudgetExhausted("maximum_snapshot_bytes")
            if event["is_root"]:
                self.root_index = len(self.steps)
            self.pending = {
                "snapshot": snapshot,
                "floor": floor,
                "selected_index": event["selected_index"],
                "behavior_log_probability": event["behavior_log_probability"],
                "surface": event["surface"],
                "branch": event["candidates"][event["selected_index"]]["branch"],
                "domain": event["domain"],
            }
            self.last_native_floor = floor
        elif kind == "macro_horizon":
            self._close_edge(facts["state"]["observation"]["run"]["floor"], False)
            if self.authority.bootstrap_snapshot is None:
                raise ValueError("macro horizon lacks its actual next input")
            self.snapshot_bytes += self.authority.bootstrap_snapshot.storage_nbytes()
            if self.snapshot_bytes > self.maximum_snapshot_bytes:
                raise ProbeBudgetExhausted("maximum_snapshot_bytes")
        elif kind == "transition" and self.pending is not None:
            breakdown = facts["reward_breakdown"]
            native = facts["native_discount"]
            before, after = native["floor_before"], native["floor_after"]
            if before != self.last_native_floor or after < before:
                raise ValueError("training native durable-floor sequence regressed or is discontinuous")
            terminal = breakdown["task_terminal"]
            discount = decision_discount(floor_before=before, floor_after=after, terminal=terminal)
            if native["terminal"] != terminal or not math.isclose(
                native["discount"], discount, rel_tol=0, abs_tol=1e-12
            ):
                raise ValueError("training native discount evidence differs")
            for component in COMPONENTS:
                self.sums[component] += self.edge_weight * float(breakdown[component])
            self.revivals += int(breakdown["revivals_used_delta"])
            self.hp_lost += float(breakdown["player_hp_lost_delta"])
            self.native_steps += 1
            self.edge_weight *= discount
            self.last_native_floor = after
            if breakdown["task_terminal"]:
                self._close_edge(facts["state"]["observation"]["run"]["floor"], True)

    def trajectory(
        self,
        *,
        trajectory_id: str,
        seed: int,
        ending: str,
        root_only: bool = False,
        reference: MacroCreditTrajectory | None = None,
    ) -> MacroCreditTrajectory:
        if self.pending is not None or self.root_index is None or ending not in {"factual_terminal", "macro_horizon"}:
            raise ValueError("unfinished or rootless trajectory is not a training sample")
        start = self.root_index if root_only else 0
        prefix = tuple(step.snapshot for step in self.steps[:start])
        steps = tuple(self.steps[start:])
        if reference is not None:
            shared = tuple(step.snapshot for step in reference.steps[:start])
            if len(prefix) != len(shared) or any(not snapshot_equal(a, b) for a, b in zip(prefix, shared, strict=True)):
                raise ValueError("actual collected candidate prefix differs from verified reference")
            prefix = shared
            if not snapshot_equal(steps[0].snapshot, reference.steps[start].snapshot):
                raise ValueError("actual collected candidate root differs from verified reference")
            steps = (replace(steps[0], snapshot=reference.steps[start].snapshot), *steps[1:])
        return MacroCreditTrajectory(
            trajectory_id=trajectory_id,
            seed=seed,
            teacher_version=self.authority.teacher_version,
            prefix=prefix,
            steps=steps,
            bootstrap_snapshot=self.authority.bootstrap_snapshot,
            bootstrap_domain="macro" if ending == "macro_horizon" else None,
            ending="macro_horizon" if ending == "macro_horizon" else "factual_terminal",
            reward_facts=tuple(self.reward_facts[start:]),
            root_intervened=root_only and self.authority.forced_candidate is not None,
        )


def collect_macro_credit_root(
    *,
    backend_factory: Callable[[], EnvironmentBackend],
    policy: FrozenPolicy,
    config: TrainingConfig,
    seed: int,
    teacher_version: int,
    limits: BranchLimits,
    output_dir: Path,
    maximum_snapshot_bytes: int = 256 * 1024 * 1024,
    maximum_group_seconds: float = 1800.0,
    artifact_root: Path | None = None,
    resource_guard: Callable[[], None] | None = None,
) -> tuple[MacroCounterfactualRoot, dict[str, Any]]:
    """Collect a complete root or raise; no partial group can reach an optimizer."""
    training_seed(seed)
    validate_course(config)
    if isinstance(teacher_version, bool) or not isinstance(teacher_version, int) or teacher_version < 0:
        raise ValueError("teacher_version must be a non-negative integer")
    if (
        isinstance(maximum_snapshot_bytes, bool)
        or not isinstance(maximum_snapshot_bytes, int)
        or maximum_snapshot_bytes <= 0
    ):
        raise ValueError("maximum_snapshot_bytes must be a positive integer")
    if (
        isinstance(maximum_group_seconds, bool)
        or not math.isfinite(maximum_group_seconds)
        or maximum_group_seconds <= 0
    ):
        raise ValueError("maximum_group_seconds must be finite and positive")
    output_dir = resolve_artifact_path(output_dir, root=artifact_root)
    output_dir.mkdir(parents=True, exist_ok=False)
    started = time.monotonic()
    results: list[dict[str, Any]] = []
    report: dict[str, Any] = {
        "format": "sts2-macro-credit-collection-v1",
        "status": "collecting",
        "seed": seed,
        "teacher_version": teacher_version,
        "passes": results,
        "data_partition": "training",
        "continuation_policy": "frozen_stochastic",
    }

    def execute(
        name: str, forced: int | None, *, control: bool = False, available_snapshot_bytes: int = maximum_snapshot_bytes
    ) -> tuple[TrainingEvidenceTrace | None, dict[str, Any]]:
        if resource_guard is not None:
            resource_guard()
        remaining = maximum_group_seconds - (time.monotonic() - started)
        if remaining <= 0:
            raise ProbeBudgetExhausted("maximum_group_seconds")
        if available_snapshot_bytes <= 0:
            raise ProbeBudgetExhausted("maximum_root_snapshot_bytes")
        actual_limits = replace(limits, timeout_seconds=min(remaining, limits.timeout_seconds))
        authority = StochasticBranchAuthority(
            policy, seed, actual_limits, forced, teacher_version=teacher_version, config=config
        )
        with ExitStack() as stack:
            output = stack.enter_context((output_dir / f"{name}.jsonl").open("x", encoding="utf-8", newline="\n"))
            prefix_output = stack.enter_context(
                (output_dir / f"{name}.prefix.jsonl").open("x", encoding="utf-8", newline="\n")
            )
            expected_prefix = (
                stack.enter_context((output_dir / "reference.prefix.jsonl").open(encoding="utf-8"))
                if name != "reference"
                else None
            )
            expected_full = (
                stack.enter_context((output_dir / "reference.jsonl").open(encoding="utf-8")) if control else None
            )
            capture = (
                None
                if control
                else TrainingEvidenceTrace(
                    output,
                    authority=authority,
                    limits=actual_limits,
                    maximum_snapshot_bytes=available_snapshot_bytes,
                    resource_guard=resource_guard,
                )
            )
            trace = (
                _GuardedEvidenceTrace(
                    output, actual_limits.trace_limits(), expected_full, resource_guard=resource_guard
                )
                if control
                else capture
            )
            assert trace is not None
            backend = backend_factory()
            stack.callback(backend.close)
            result = run_branch_pass(
                backend=backend,
                policy=policy,
                config=config,
                seed=seed,
                limits=actual_limits,
                trace=trace,
                prefix=_GuardedEvidenceTrace(
                    prefix_output, actual_limits.trace_limits(), expected_prefix, resource_guard=resource_guard
                ),
                forced_candidate=forced,
                authority=authority,
            )
            if not result["root_found"] or result["ending"] not in {"factual_terminal", "macro_horizon"}:
                raise ProbeBudgetExhausted("no_root_or_unknown_outcome")
            results.append({"name": name, **result})
            if resource_guard is not None:
                resource_guard()
            return capture, result

    try:
        capture, result = execute("reference", None)
        assert capture is not None and capture.root_index is not None
        reference = capture.trajectory(trajectory_id="reference", seed=seed, ending=result["ending"])
        root_index = capture.root_index
        root_action = reference.steps[root_index].action_index
        log_probs = capture.authority.root_teacher_log_probs
        assert log_probs is not None
        root_event = result["root"]
        execute("same-action-control", None, control=True)
        report["same_action_gate"] = "matched"
        candidates: list[MacroCreditTrajectory] = []
        for index in range(reference.steps[root_index].snapshot.candidate_count):
            if index == root_action:
                candidate = capture.trajectory(
                    trajectory_id=f"candidate-{index}",
                    seed=seed,
                    ending=result["ending"],
                    root_only=True,
                    reference=reference,
                )
            else:
                retained_bytes = trajectory_storage_nbytes((reference, *candidates))
                branch_capture, branch_result = execute(
                    f"candidate-{index:03d}",
                    index,
                    available_snapshot_bytes=maximum_snapshot_bytes - retained_bytes,
                )
                assert branch_capture is not None
                candidate = branch_capture.trajectory(
                    trajectory_id=f"candidate-{index}",
                    seed=seed,
                    ending=branch_result["ending"],
                    root_only=True,
                    reference=reference,
                )
                # Drop the independently encoded replay prefix before another
                # pass. The candidate now shares only the verified reference.
                del branch_capture
            candidates.append(candidate)
            if trajectory_storage_nbytes((reference, *candidates)) > maximum_snapshot_bytes:
                raise ProbeBudgetExhausted("maximum_root_snapshot_bytes")
        root = MacroCounterfactualRoot(
            root_id=f"seed-{seed}:teacher-{teacher_version}:{limits.root_surface}-{limits.root_occurrence}",
            reference=reference,
            root_step_index=root_index,
            candidates=tuple(candidates),
            candidate_branches=tuple(candidate["branch"] for candidate in root_event["candidates"]),
            teacher_log_probabilities=log_probs,
            same_action_gate="matched",
            evidence_directory=str(output_dir),
        )
        report.update(
            status="complete",
            root_id=root.root_id,
            candidate_count=len(candidates),
            simulator_steps=sum(int(item["native_steps"]) for item in results),
        )
        return root, report
    except BaseException as exc:
        report.update(status="aborted", error_type=type(exc).__name__, error=str(exc))
        raise
    finally:
        (output_dir / "collection.json").write_text(
            json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8"
        )


__all__ = ["StochasticBranchAuthority", "TrainingEvidenceTrace", "collect_macro_credit_root"]
