"""Fresh stochastic-teacher collection, reusing the verified B1 simulator path.

Only this explicit training collector can produce MacroCounterfactualRoot.
The diagnostic CLI remains greedy, read-only with respect to training, and its
finite-prefix return summaries are never imported as learning labels.
"""

from __future__ import annotations

import hashlib
import io
import json
import math
import threading
import time
from collections.abc import Callable, Mapping, Sequence
from concurrent.futures import ThreadPoolExecutor
from contextlib import ExitStack
from dataclasses import replace
from pathlib import Path
from typing import Any, Literal, TextIO

import numpy as np
import torch

from sts2_rl.artifacts import resolve_artifact_path
from sts2_rl.contracts import EnvironmentBackend
from sts2_rl.encoding import EncodedDecisionSnapshot, EncodedWorldSnapshot, GroundedObservationEncoder
from sts2_rl.macro_branch_diagnostic import BranchAuthority, BranchLimits, _MacroHorizonReached, run_branch_pass
from sts2_rl.macro_root_diagnostic import (
    EvidenceTrace,
    FrozenPolicy,
    ProbeAuthority,
    ProbeBudgetExhausted,
    validate_course,
)
from sts2_rl.macro_root_evidence import canonical_facts
from sts2_rl.semantics.clock import decision_discount
from sts2_rl.semantics.forward import ForwardDecision, SemanticCandidate
from sts2_rl.training.config import TrainingConfig

from .afterstate import synthesize_deterministic_afterstates
from .macro_credit_data import (
    MacroAfterstateNode,
    MacroCounterfactualRoot,
    MacroCreditCohort,
    MacroCreditTrajectory,
    MacroRewardFacts,
    macro_mixture_log_probs,
    snapshot_equal,
    training_seed,
)
from .transitions import HybridVTraceStep

COMPONENTS = ("terminal_reward", "progress_reward", "revival_penalty", "pace_penalty", "hp_loss_penalty")
_RECOVERABLE_EPISODE_BUDGET_REASONS: frozenset[str] = frozenset(
    {
        "max_native_steps",
        "timeout_seconds",
        "incomplete_durable_episode",
    }
)


def is_recoverable_episode_exception(exc: BaseException) -> bool:
    """Return whether a single-episode failure should resample a replacement seed."""
    if getattr(exc, "recoverable", False) is True:
        return True
    return isinstance(exc, ProbeBudgetExhausted) and str(exc) in _RECOVERABLE_EPISODE_BUDGET_REASONS


class StochasticBranchAuthority(BranchAuthority):
    """Stochastic teacher sampling with optional branch-uniform exploration and afterstate capture."""

    policy_mode = "frozen_stochastic"
    macro_branch_epsilon: float = 0.0
    collect_afterstates: bool = False

    def __init__(
        self,
        policy: FrozenPolicy,
        seed: int,
        limits: BranchLimits,
        forced_candidate: int | None,
        *,
        teacher_version: int,
        config: TrainingConfig,
        macro_branch_epsilon: float = 0.0,
        collect_afterstates: bool = False,
    ) -> None:
        super().__init__(policy, seed, limits, forced_candidate)
        training_seed(seed)
        if (
            isinstance(macro_branch_epsilon, bool)
            or not math.isfinite(macro_branch_epsilon)
            or not 0.0 <= macro_branch_epsilon < 1.0
        ):
            raise ValueError("macro_branch_epsilon must be finite and in [0.0, 1.0)")
        if not isinstance(collect_afterstates, bool):
            raise ValueError("collect_afterstates must be boolean")
        self.teacher_version = teacher_version
        self.macro_branch_epsilon = float(macro_branch_epsilon)
        self.collect_afterstates = collect_afterstates
        self._sampling_rng = np.random.default_rng(np.random.SeedSequence([seed, 0x4D414352]))
        self._encoder = GroundedObservationEncoder(config.model.to_encoding_config())
        self.current_snapshot: EncodedDecisionSnapshot | None = None
        self.current_afterstates: tuple[EncodedWorldSnapshot, ...] | None = None
        self.current_afterstate_branches: tuple[str, ...] | None = None
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
        self.current_afterstates = None
        self.current_afterstate_branches = None
        if self.collect_afterstates and self.branch_limits.terminal_only and self.forced_candidate is None:
            self._current_is_root = False
            return
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
        mask = torch.tensor(snapshot.action_mask.tolist(), dtype=torch.bool)
        teacher_logs = torch.log_softmax(logits[: mask.numel()].masked_fill(~mask, -torch.inf), dim=0)
        behavior_logs = macro_mixture_log_probs(teacher_logs, snapshot, self.macro_branch_epsilon)
        probabilities = behavior_logs.exp().numpy()
        draw = float(self._sampling_rng.random())  # Burn one draw even at the forced root.
        selected = min(int(np.searchsorted(np.cumsum(probabilities), draw, side="right")), len(candidates) - 1)
        if self._current_is_root:
            self.greedy_root_index = greedy
            self.root_teacher_log_probs = tuple(float(value) for value in teacher_logs.tolist())
            if self.forced_candidate is not None:
                if isinstance(self.forced_candidate, bool) or not 0 <= self.forced_candidate < len(candidates):
                    raise ValueError("forced root action is not in the complete legal candidate set")
                selected = self.forced_candidate
        self._behavior_log_probability = (
            0.0
            if self._current_is_root and self.forced_candidate is not None
            else float(behavior_logs[selected].item())
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
        if self.collect_afterstates and control_domain == "macro" and len(candidates) >= 2:
            post_worlds = synthesize_deterministic_afterstates(
                encoder=self._encoder,
                observation=observation,
                decision=decision,
                candidates=candidates,
            )
            if post_worlds is not None:
                self.current_afterstates = post_worlds
                self.current_afterstate_branches = tuple(candidate.branch for candidate in candidates)
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
            macro_branch_epsilon=self.macro_branch_epsilon,
            has_afterstate=self.current_afterstates is not None,
        )


class _GuardedEvidenceTrace(EvidenceTrace):
    def __init__(self, *args: Any, resource_guard: Callable[[], None] | None = None, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self.resource_guard = resource_guard

    def record(self, kind: str, facts: Any, policy: Any = None) -> None:
        if self.resource_guard is not None:
            self.resource_guard()
        super().record(kind, facts, policy)


class _DiscardEvidenceTrace(_GuardedEvidenceTrace):
    """No-op prefix sink for full-episode runs without a forced counterfactual root."""

    def record(self, kind: str, facts: Any, policy: Any = None) -> None:
        del kind, facts, policy
        if self.resource_guard is not None:
            self.resource_guard()

    def finish(self) -> None:
        return


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
        compact_evidence: bool = False,
    ) -> None:
        super().__init__(output, limits.trace_limits(), resource_guard=resource_guard)
        self.authority = authority
        self.maximum_snapshot_bytes = maximum_snapshot_bytes
        self.snapshot_bytes = 0
        self.steps: list[HybridVTraceStep] = []
        self.reward_facts: list[MacroRewardFacts] = []
        self.afterstate_nodes: list[MacroAfterstateNode] = []
        self.macro_surface_branch_counts: dict[str, int] = {}
        self.root_index: int | None = None
        self.pending: dict[str, Any] | None = None
        self.sums = dict.fromkeys(COMPONENTS, 0.0)
        self.revivals = 0
        self.hp_lost = 0.0
        self.native_steps = 0
        self.edge_weight = 1.0
        self.last_native_floor: int | None = None
        self.compact_evidence = compact_evidence
        self.last_state: dict[str, Any] | None = None
        self.episode_return = 0.0
        self.episode_weight = 1.0
        self.episode_native_steps = 0
        self.revival_events: list[dict[str, Any]] = []
        self.max_floor = 0
        self.previous_run: dict[str, Any] = {}
        self.last_combat_run: dict[str, Any] = {}

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
        if isinstance(facts, dict) and isinstance(facts.get("state"), dict):
            old_observation = (self.last_state or {}).get("observation", {})
            self.previous_run = dict(old_observation.get("run") or {})
            if old_observation.get("phase") == "combat":
                self.last_combat_run = dict(self.previous_run)
            self.last_state = facts["state"]
            assert self.last_state is not None
            current_run = (self.last_state.get("observation") or {}).get("run") or {}
            self.max_floor = max(self.max_floor, int(current_run.get("floor") or 0))
        if kind == "transition":
            self.episode_return += self.episode_weight * sum(float(facts["reward_breakdown"][key]) for key in COMPONENTS)
            self.episode_weight *= float(facts["native_discount"]["discount"])
            self.episode_native_steps += 1
            deaths = int(facts["reward_breakdown"].get("revivals_used_delta", 0))
            if deaths:
                counters = facts["state"]["observation"].get("_training", {})
                self.revival_events.append({
                    "native_step": self.episode_native_steps, "count": deaths,
                    "act": self.previous_run.get("act"), "floor": self.previous_run.get("floor"),
                    "room_model_id": self.previous_run.get("room_model_id"),
                    "revivals_used_after": counters.get("revivals_used"),
                })
        written = facts
        if self.compact_evidence and isinstance(facts, dict):
            # Learner inputs below remain full sparse snapshots in memory. The
            # log avoids duplicating large raw observations/hidden vectors on
            # every native action of a full run; no output limit is increased.
            written = {k: v for k, v in facts.items() if k not in {"state", "strict_groups", "semantic", "observation", "semantic_actions"}}
            if kind == "semantic_snapshot":
                written["observation_sha256"] = hashlib.sha256(json.dumps(facts["observation"], sort_keys=True).encode()).hexdigest()
                written["log_scope"] = "observation hash; full sparse learner input retained in current cohort"
            if isinstance(facts.get("state"), dict):
                state = facts["state"]
                obs = state.get("observation", {})
                written["state"] = {"observation": {"run": obs.get("run"), "_training": obs.get("_training")},
                                    "terminated": state.get("terminated"), "truncated": state.get("truncated"),
                                    "terminal_reason": state.get("terminal_reason"), "transition": state.get("transition")}
            if isinstance(facts.get("semantic"), dict):
                event = facts["semantic"]
                written["semantic"] = {k: v for k, v in event.items() if k not in {"candidates", "plan", "snapshot"}}
                written["semantic"]["candidate_count"] = len(event.get("candidates", []))
        if self.compact_evidence and self.reference is None:
            if self.resource_guard is not None:
                self.resource_guard()
            row = {"kind": kind, "facts": canonical_facts(written), "policy": None}
            encoded = json.dumps(row, ensure_ascii=False, allow_nan=False, separators=(",", ":")) + "\n"
            size = len(encoded.encode("utf-8"))
            if self.bytes_written + size > self.limits.max_trace_bytes:
                raise ProbeBudgetExhausted("max_trace_bytes")
            self.output.write(encoded)
            self.bytes_written += size
            self.rows += 1
        else:
            super().record(kind, written, None if self.compact_evidence else policy)
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
            if self.authority.current_afterstates is not None:
                assert self.authority.current_afterstate_branches is not None
                node = MacroAfterstateNode(
                    step_index=len(self.steps),
                    surface=event["surface"],
                    candidate_branches=self.authority.current_afterstate_branches,
                    post_worlds=self.authority.current_afterstates,
                )
                self.snapshot_bytes += node.storage_nbytes()
                self.authority.current_afterstates = None
                self.authority.current_afterstate_branches = None
                if self.snapshot_bytes > self.maximum_snapshot_bytes:
                    raise ProbeBudgetExhausted("maximum_snapshot_bytes")
                self.afterstate_nodes.append(node)
            if event["domain"] == "macro":
                selected_branch = event["candidates"][event["selected_index"]]["branch"]
                branch_key = f"{event['surface']}:{selected_branch}"
                self.macro_surface_branch_counts[branch_key] = (
                    self.macro_surface_branch_counts.get(branch_key, 0) + 1
                )
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

    def finish(self) -> None:
        self.output.flush()
        super().finish()

    def episode_statistics(self, *, seed: int, ending: str) -> dict[str, Any]:
        state = self.last_state or {}
        obs = state.get("observation", {})
        run, counters = obs.get("run", {}), obs.get("_training", {})
        player = obs.get("player") if isinstance(obs.get("player"), Mapping) else {}
        raw_deck = player.get("deck_cards", player.get("deck"))
        final_deck_size: int | None = None
        if isinstance(raw_deck, int) and not isinstance(raw_deck, bool):
            final_deck_size = raw_deck
        elif isinstance(raw_deck, Sequence) and not isinstance(raw_deck, str | bytes):
            final_deck_size = len(raw_deck)
        outcome = (state.get("transition") or {}).get("facts", {}).get("run_result")
        complete = (ending == "factual_terminal" and state.get("terminated") is True
                    and state.get("truncated") is False and outcome in {"victory", "defeat"})
        return {"seed": seed, "status": "complete" if complete else "incomplete", "ending": ending,
                "scope": "natural stochastic training episode; reused for gradient update, not held-out evaluation",
                "teacher_version": self.authority.teacher_version, "run_result": outcome,
                "terminated": state.get("terminated"), "truncated": state.get("truncated"),
                "revival_budget": counters.get("revival_budget"), "revivals_used": counters.get("revivals_used"),
                "player_hp_lost": counters.get("player_hp_lost"), "ending_act": run.get("act"),
                "ending_floor": run.get("floor"), "room_model_id": run.get("room_model_id"),
                "max_floor": self.max_floor, "terminal_reason": state.get("terminal_reason"),
                "final_deck_size": final_deck_size,
                "afterstate_step_count": len(self.afterstate_nodes),
                "macro_surface_branch_counts": dict(sorted(self.macro_surface_branch_counts.items())),
                "death_location": ({"act": self.last_combat_run.get("act", run.get("act")),
                                    "floor": self.last_combat_run.get("floor", run.get("floor")),
                                    "room_model_id": self.last_combat_run.get("room_model_id", run.get("room_model_id"))}
                                   if outcome == "defeat" else None),
                "revival_events": self.revival_events,
                "observed_discounted_return": self.episode_return, "native_steps": self.episode_native_steps}

    def trajectory(
        self,
        *,
        trajectory_id: str,
        seed: int,
        ending: str,
        root_only: bool = False,
        reference: MacroCreditTrajectory | None = None,
    ) -> MacroCreditTrajectory:
        if (self.pending is not None or ending not in {"factual_terminal", "macro_horizon"}
                or (self.root_index is None and (root_only or ending != "factual_terminal"))):
            raise ValueError("unfinished or rootless trajectory is not a training sample")
        start = self.root_index if root_only else 0
        assert start is not None
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
        afterstate_nodes = tuple(
            MacroAfterstateNode(
                step_index=node.step_index - start,
                surface=node.surface,
                candidate_branches=node.candidate_branches,
                post_worlds=node.post_worlds,
            )
            for node in self.afterstate_nodes
            if node.step_index >= start
        )
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
            afterstate_nodes=afterstate_nodes,
            macro_branch_epsilon=self.authority.macro_branch_epsilon,
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
) -> tuple[MacroCounterfactualRoot | MacroCreditTrajectory, dict[str, Any]]:
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
                    compact_evidence=limits.terminal_only,
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
            if name == "reference" and capture is not None:
                report["training_episode"] = capture.episode_statistics(seed=seed, ending=result["ending"])
            rootless_complete = (limits.terminal_only and not result["root_found"]
                                 and report.get("training_episode", {}).get("status") == "complete")
            if (not result["root_found"] and not rootless_complete) or result["ending"] not in {"factual_terminal", "macro_horizon"}:
                raise ProbeBudgetExhausted("no_root_or_unknown_outcome")
            results.append({"name": name, **result})
            if resource_guard is not None:
                resource_guard()
            return capture, result

    try:
        capture, result = execute("reference", None)
        assert capture is not None
        reference = capture.trajectory(trajectory_id="reference", seed=seed, ending=result["ending"])
        if capture.root_index is None:
            report.update(status="complete", root_found=False, candidate_count=0,
                          training_scope="complete natural game; no counterfactual root reached",
                          simulator_steps=result["native_steps"])
            return reference, report
        root_index = capture.root_index
        root_action = reference.steps[root_index].action_index
        log_probs = capture.authority.root_teacher_log_probs
        assert log_probs is not None
        root_event = result["root"]
        report["same_action_gate"] = "matched"
        candidates: list[MacroCreditTrajectory] = []
        root_step = reference.steps[root_index]
        # An eventual terminal at the end of the natural run does not make an
        # unexecuted alternative action at the earlier root terminal.
        is_terminal = root_step.terminal
        report["alternative_candidate_scope"] = "shadow-value proxies, not factual alternative rollouts or game outcomes"
        for index in range(root_step.snapshot.candidate_count):
            if index == root_action:
                candidate = capture.trajectory(
                    trajectory_id=f"candidate-{index}",
                    seed=seed,
                    ending=result["ending"],
                    root_only=True,
                    reference=reference,
                )
            else:
                candidate_step = HybridVTraceStep(
                    snapshot=root_step.snapshot,
                    action_index=index,
                    behavior_log_probability=0.0,
                    reward=0.0,
                    discount=0.0 if is_terminal else 1.0,
                    terminal=is_terminal,
                    surface=root_step.surface,
                    branch=root_event["candidates"][index]["branch"],
                    control_domain="macro",
                    policy_version=capture.authority.teacher_version,
                )
                candidate_facts = MacroRewardFacts(
                    terminal_reward=0.0,
                    progress_reward=0.0,
                    revival_penalty=0.0,
                    pace_penalty=0.0,
                    hp_loss_penalty=0.0,
                    revivals_used_delta=0,
                    player_hp_lost_delta=0.0,
                    floor_before=reference.reward_facts[root_index].floor_before,
                    floor_after=reference.reward_facts[root_index].floor_before,
                    native_steps=1,
                )
                candidate = MacroCreditTrajectory(
                    trajectory_id=f"candidate-{index}",
                    seed=seed,
                    teacher_version=capture.authority.teacher_version,
                    prefix=tuple(step.snapshot for step in reference.steps[:root_index]),
                    steps=(candidate_step,),
                    bootstrap_snapshot=None if is_terminal else (capture.authority.bootstrap_snapshot or root_step.snapshot),
                    bootstrap_domain=None if is_terminal else "macro",
                    ending="factual_terminal" if is_terminal else "macro_horizon",
                    reward_facts=(candidate_facts,),
                    root_intervened=True,
                )
            candidates.append(candidate)
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


def collect_durable_afterstate_episode(
    *,
    backend_factory: Callable[[], EnvironmentBackend],
    policy: FrozenPolicy,
    config: TrainingConfig,
    seed: int,
    teacher_version: int,
    limits: BranchLimits,
    output_dir: Path,
    macro_branch_epsilon: float = 0.0,
    maximum_snapshot_bytes: int = 256 * 1024 * 1024,
    maximum_group_seconds: float = 1800.0,
    artifact_root: Path | None = None,
    resource_guard: Callable[[], None] | None = None,
) -> tuple[MacroCreditTrajectory, dict[str, Any]]:
    """Collect one complete full-run episode with multi-node deterministic afterstate synthesis."""
    training_seed(seed)
    validate_course(config)
    if not limits.terminal_only:
        raise ValueError("durable_afterstate_v2 episode collection requires limits.terminal_only=True")
    if isinstance(teacher_version, bool) or not isinstance(teacher_version, int) or teacher_version < 0:
        raise ValueError("teacher_version must be a non-negative integer")
    if (
        isinstance(macro_branch_epsilon, bool)
        or not math.isfinite(macro_branch_epsilon)
        or not 0.0 <= macro_branch_epsilon < 1.0
    ):
        raise ValueError("macro_branch_epsilon must be finite and in [0.0, 1.0)")
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
    report: dict[str, Any] = {
        "format": "sts2-durable-afterstate-collection-v2",
        "status": "collecting",
        "seed": seed,
        "teacher_version": teacher_version,
        "macro_branch_epsilon": float(macro_branch_epsilon),
        "data_partition": "training",
        "continuation_policy": "frozen_stochastic",
    }
    try:
        if resource_guard is not None:
            resource_guard()
        remaining = maximum_group_seconds - (time.monotonic() - started)
        if remaining <= 0:
            raise ProbeBudgetExhausted("maximum_group_seconds")
        actual_limits = replace(limits, timeout_seconds=min(remaining, limits.timeout_seconds))
        authority = StochasticBranchAuthority(
            policy,
            seed,
            actual_limits,
            None,
            teacher_version=teacher_version,
            config=config,
            macro_branch_epsilon=macro_branch_epsilon,
            collect_afterstates=True,
        )
        with ExitStack() as stack:
            output = stack.enter_context((output_dir / "episode.jsonl").open("x", encoding="utf-8", newline="\n"))
            capture = TrainingEvidenceTrace(
                output,
                authority=authority,
                limits=actual_limits,
                maximum_snapshot_bytes=maximum_snapshot_bytes,
                resource_guard=resource_guard,
                compact_evidence=True,
            )
            backend = backend_factory()
            stack.callback(backend.close)
            result = run_branch_pass(
                backend=backend,
                policy=policy,
                config=config,
                seed=seed,
                limits=actual_limits,
                trace=capture,
                prefix=_DiscardEvidenceTrace(
                    io.StringIO(), actual_limits.trace_limits(), None, resource_guard=resource_guard
                ),
                forced_candidate=None,
                authority=authority,
            )
            episode_stats = capture.episode_statistics(seed=seed, ending=result["ending"])
            report["training_episode"] = episode_stats
            if result["ending"] != "factual_terminal" or episode_stats.get("status") != "complete":
                raise ProbeBudgetExhausted("incomplete_durable_episode")
            trajectory = capture.trajectory(
                trajectory_id=f"seed-{seed}:teacher-{teacher_version}:durable-episode",
                seed=seed,
                ending=result["ending"],
            )
            report.update(
                status="complete",
                trajectory_id=trajectory.trajectory_id,
                semantic_steps=len(trajectory.steps),
                macro_steps=sum(step.control_domain == "macro" for step in trajectory.steps),
                combat_steps=sum(step.control_domain == "combat" for step in trajectory.steps),
                afterstate_nodes=len(trajectory.afterstate_nodes),
                simulator_steps=int(result["native_steps"]),
            )
            if resource_guard is not None:
                resource_guard()
            return trajectory, report
    except BaseException as exc:
        report.update(status="aborted", error_type=type(exc).__name__, error=str(exc))
        raise
    finally:
        (output_dir / "collection.json").write_text(
            json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8"
        )


def collect_durable_afterstate_cohort(
    *,
    backend_factory: Callable[[], EnvironmentBackend],
    policy: FrozenPolicy,
    config: TrainingConfig,
    seeds: Sequence[int],
    teacher_version: int,
    limits: BranchLimits,
    cohort_dir: Path,
    cohort_id: str,
    macro_branch_epsilon: float = 0.0,
    parallel_collectors: int = 1,
    replacement_seed_fn: Callable[[int], int] | None = None,
    max_recoverable_incidents: int = 4,
    maximum_episode_snapshot_bytes: int = 256 * 1024 * 1024,
    maximum_cohort_snapshot_bytes: int = 512 * 1024 * 1024,
    maximum_episode_seconds: float = 3600.0,
    maximum_cohort_seconds: float = 7200.0,
    artifact_root: Path | None = None,
    resource_guard: Callable[[], None] | None = None,
    worker_pool: Any | None = None,
) -> tuple[MacroCreditCohort, list[dict[str, Any]], list[dict[str, Any]]]:
    """Collect a full-episode cohort in deterministic seed order, optionally in parallel."""
    if not seeds:
        raise ValueError("durable afterstate cohort requires at least one seed")
    if isinstance(parallel_collectors, bool) or not isinstance(parallel_collectors, int) or parallel_collectors < 1:
        raise ValueError("parallel_collectors must be a positive integer")
    if (
        isinstance(max_recoverable_incidents, bool)
        or not isinstance(max_recoverable_incidents, int)
        or max_recoverable_incidents < 0
    ):
        raise ValueError("max_recoverable_incidents must be a non-negative integer")
    for seed in seeds:
        training_seed(seed)
    cohort_dir = resolve_artifact_path(cohort_dir, root=artifact_root)
    if worker_pool is not None:
        result_tuple: tuple[MacroCreditCohort, list[dict[str, Any]], list[dict[str, Any]]] = (
            worker_pool.collect_cohort(
                config=config,
                seeds=seeds,
                teacher_version=teacher_version,
                limits=limits,
                cohort_dir=cohort_dir,
                cohort_id=cohort_id,
                macro_branch_epsilon=macro_branch_epsilon,
                replacement_seed_fn=replacement_seed_fn,
                max_recoverable_incidents=max_recoverable_incidents,
                maximum_episode_snapshot_bytes=maximum_episode_snapshot_bytes,
                maximum_cohort_snapshot_bytes=maximum_cohort_snapshot_bytes,
                maximum_episode_seconds=maximum_episode_seconds,
                maximum_cohort_seconds=maximum_cohort_seconds,
                artifact_root=artifact_root or cohort_dir.parent,
                resource_guard=resource_guard,
            )
        )
        return result_tuple
    started = time.monotonic()
    guard_lock = threading.Lock()
    recovery_lock = threading.Lock()
    next_attempt_offset = len(seeds)
    recoverable_incidents: list[dict[str, Any]] = []

    def thread_safe_guard() -> None:
        if resource_guard is None:
            return
        with guard_lock:
            resource_guard()

    def run_one(initial_offset: int, initial_seed: int) -> tuple[MacroCreditTrajectory, dict[str, Any]]:
        nonlocal next_attempt_offset
        current_offset = initial_offset
        current_seed = initial_seed
        while True:
            remaining_seconds = maximum_cohort_seconds - (time.monotonic() - started)
            if remaining_seconds <= 0:
                raise ProbeBudgetExhausted("maximum_cohort_seconds")
            try:
                return collect_durable_afterstate_episode(
                    backend_factory=backend_factory,
                    policy=policy,
                    config=config,
                    seed=current_seed,
                    teacher_version=teacher_version,
                    limits=limits,
                    output_dir=cohort_dir / f"episode-{current_offset:03d}",
                    macro_branch_epsilon=macro_branch_epsilon,
                    maximum_snapshot_bytes=min(maximum_episode_snapshot_bytes, maximum_cohort_snapshot_bytes),
                    maximum_group_seconds=min(maximum_episode_seconds, remaining_seconds),
                    artifact_root=artifact_root,
                    resource_guard=thread_safe_guard,
                )
            except BaseException as exc:
                if not is_recoverable_episode_exception(exc) or replacement_seed_fn is None:
                    raise
                with recovery_lock:
                    recoverable_incidents.append(
                        {
                            "attempt_offset": current_offset,
                            "seed": current_seed,
                            "error_type": type(exc).__name__,
                            "error": str(exc),
                            "fingerprint": getattr(exc, "fingerprint", None),
                            "quarantine_path": getattr(exc, "quarantine_path", None),
                        }
                    )
                    if len(recoverable_incidents) > max_recoverable_incidents:
                        raise
                    current_offset = next_attempt_offset
                    next_attempt_offset += 1
                    candidate_seed = replacement_seed_fn(current_offset)
                    training_seed(candidate_seed)
                    current_seed = candidate_seed

    collected_pairs: list[tuple[MacroCreditTrajectory, dict[str, Any]]]
    if parallel_collectors == 1 or len(seeds) == 1:
        collected_pairs = [run_one(offset, seed) for offset, seed in enumerate(seeds)]
    else:
        workers = min(parallel_collectors, len(seeds))
        with ThreadPoolExecutor(max_workers=workers) as executor:
            futures = [executor.submit(run_one, offset, seed) for offset, seed in enumerate(seeds)]
            collected_pairs = [future.result() for future in futures]

    trajectories = tuple(pair[0] for pair in collected_pairs)
    reports = [pair[1] for pair in collected_pairs]
    if recoverable_incidents and reports:
        reports[0]["recoverable_incidents"] = list(recoverable_incidents)
    training_episodes = [
        report["training_episode"]
        for report in reports
        if isinstance(report.get("training_episode"), dict)
    ]
    cohort = MacroCreditCohort(cohort_id, teacher_version, (), natural_only=trajectories)
    if cohort.storage_nbytes() > maximum_cohort_snapshot_bytes:
        raise ProbeBudgetExhausted("maximum_cohort_snapshot_bytes")
    thread_safe_guard()
    return cohort, training_episodes, reports


__all__ = [
    "StochasticBranchAuthority",
    "TrainingEvidenceTrace",
    "collect_durable_afterstate_cohort",
    "collect_durable_afterstate_episode",
    "collect_macro_credit_root",
    "is_recoverable_episode_exception",
]
