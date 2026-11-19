"""B1 fixed-budget simulator interventions with a fresh same-action control gate.

Every branch rebuilds its full prefix and compares it before root dispatch.
Only the root semantic choice is intervened upon. No learner, value bootstrap,
replay publication, action rule, adaptive allocation or statistical resampling
is implemented here. JSONL policy tensors are audit evidence, never training
hidden-state inputs; semantic observations/candidates permit fresh re-encoding.
"""

from __future__ import annotations

import argparse
import json
import math
import time
from collections.abc import Callable, Mapping, Sequence
from contextlib import ExitStack
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Literal
from uuid import uuid4

import numpy as np

from sts2_baseline import ActFrontloadedDeathRewardCalculator, act_frontloaded_death_reward_identity
from sts2_env.headless_sim_bridge_client import resolve_headless_sim_exe
from sts2_rl.backends import HeadlessBackend
from sts2_rl.contracts import EnvironmentBackend, ResetRequest
from sts2_rl.encoding import EncodedDecisionSnapshot, GroundedObservationEncoder, grounding_encoding_identity
from sts2_rl.macro_root_diagnostic import (
    EvidenceTrace,
    FrozenPolicy,
    ProbeAuthority,
    ProbeBudgetExhausted,
    ProbeLimits,
    ProbeMismatch,
    _floor,
    _state_facts,
    dispatch_request,
    new_output_dir,
    validate_course,
    validate_seed,
)
from sts2_rl.macro_root_evidence import reward_state
from sts2_rl.macro_root_policy import FrozenMacroRootPolicy
from sts2_rl.semantics.clock import DECISION_CLOCK_CONTRACT_VERSION, decision_discount
from sts2_rl.semantics.forward import FORWARD_COMPILER_CONTRACT_VERSION, ForwardDecision, SemanticCandidate
from sts2_rl.simulator_identity import verify_headless_simulator
from sts2_rl.training.config import TrainingConfig, load_training_config

SCHEMA = "sts2-macro-branch-b1-v1"
SURFACES = ("reward", "rest", "shop", "map", "event")
REWARD_COMPONENTS = (
    "reward",
    "terminal_reward",
    "potential_reward",
    "revival_penalty",
    "pace_penalty",
    "hp_loss_penalty",
    "progress_reward",
)


@dataclass(frozen=True)
class BranchLimits:
    root_surface: str = "reward"
    root_occurrence: int = 1
    continuation_macro_decisions: int = 4
    terminal_only: bool = False
    max_native_steps: int = 4096
    max_candidates: int = 64
    timeout_seconds: float = 300.0
    max_trace_bytes: int = 128 * 1024 * 1024
    policy_atol: float = 1e-6
    policy_rtol: float = 1e-5

    def __post_init__(self) -> None:
        if self.root_surface not in SURFACES:
            raise ValueError(f"root_surface must be one of {SURFACES}")
        if not isinstance(self.terminal_only, bool):
            raise ValueError("terminal_only must be boolean")
        for name in ("root_occurrence", "max_native_steps", "max_candidates", "max_trace_bytes"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
                raise ValueError(f"{name} must be a positive integer")
        count = self.continuation_macro_decisions
        if isinstance(count, bool) or not isinstance(count, int) or count < 0:
            raise ValueError("continuation_macro_decisions must be a non-negative integer")
        self.trace_limits()

    def trace_limits(self) -> ProbeLimits:
        return ProbeLimits(
            max_native_steps=self.max_native_steps,
            root_occurrence=self.root_occurrence,
            continuation_decisions=self.continuation_macro_decisions,
            timeout_seconds=self.timeout_seconds,
            max_trace_bytes=self.max_trace_bytes,
            policy_atol=self.policy_atol,
            policy_rtol=self.policy_rtol,
        )


class _MacroHorizonReached(Exception):
    pass


class BranchAuthority(ProbeAuthority):
    """Production executor, one root intervention, then frozen greedy control."""

    policy_mode = "greedy"

    def __init__(
        self, policy: FrozenPolicy, seed: int, limits: BranchLimits, forced_candidate: int | None = None
    ) -> None:
        super().__init__(policy, seed, limits.trace_limits())
        self._branch_policy = policy
        self._branch_seed = seed
        self.branch_limits = limits
        self.forced_candidate = forced_candidate
        self.root_count = 0
        self.macro_count = 0
        self.combat_count = 0
        self.post_root_macro_count = 0
        self.post_root_combat_count = 0
        self.greedy_root_index: int | None = None
        self.root_policy_logits: list[float] | None = None
        self._current_is_root = False

    def validate_binding(
        self, *, policy: FrozenPolicy, seed: int, limits: BranchLimits, forced_candidate: int | None
    ) -> None:
        """Prevent a custom sampler from silently using another pass identity."""
        if (
            self._branch_policy is not policy
            or self._branch_seed != seed
            or self.branch_limits != limits
            or self.forced_candidate != forced_candidate
        ):
            raise ValueError("injected branch authority does not match policy/seed/limits/intervention")
        if self.semantic_count or self.root_count or self.root_ordinal is not None:
            raise ValueError("injected branch authority must be fresh for this reset/replay pass")
        if not isinstance(self.policy_mode, str) or not self.policy_mode:
            raise ValueError("injected branch authority must declare its policy_mode")

    def _prepare_semantic_exploration(
        self,
        *,
        observation: Mapping[str, Any],
        decision: ForwardDecision,
        candidates: Sequence[SemanticCandidate],
        control_domain: Literal["macro", "combat"],
    ) -> None:
        del observation
        self._current_is_root = False
        if self.root_ordinal is not None:
            if (
                not self.branch_limits.terminal_only
                and control_domain == "macro"
                and self.post_root_macro_count >= self.branch_limits.continuation_macro_decisions
            ):
                raise _MacroHorizonReached
            return
        eligible = (
            control_domain == "macro" and decision.surface == self.branch_limits.root_surface and len(candidates) >= 2
        )
        if decision.surface == "reward":
            eligible = eligible and {"take", "skip"}.issubset({candidate.branch for candidate in candidates})
        if eligible:
            self.root_count += 1
            self._current_is_root = self.root_count == self.branch_limits.root_occurrence
            if self._current_is_root and len(candidates) > self.branch_limits.max_candidates:
                raise ProbeBudgetExhausted("root_candidate_count_exceeds_fixed_all_candidate_budget")

    def _semantic_choice(self, *, candidates: Sequence[SemanticCandidate], snapshot: EncodedDecisionSnapshot) -> int:
        greedy = super()._semantic_choice(candidates=candidates, snapshot=snapshot)
        if self._current_is_root:
            self.greedy_root_index = greedy
            if self.forced_candidate is not None:
                if (
                    isinstance(self.forced_candidate, bool)
                    or not isinstance(self.forced_candidate, int)
                    or not 0 <= self.forced_candidate < len(candidates)
                ):
                    raise ValueError("forced root candidate is outside the verified executable set")
                return self.forced_candidate
        return greedy

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
        self.semantic_count += 1
        if control_domain == "macro":
            self.macro_count += 1
        else:
            self.combat_count += 1
        if self._current_is_root:
            self.root_ordinal = self.semantic_count
        elif self.root_ordinal is not None:
            if control_domain == "macro":
                self.post_root_macro_count += 1
            else:
                self.post_root_combat_count += 1
        self.event = {
            "semantic_ordinal": self.semantic_count,
            "surface": decision.surface,
            "domain": control_domain,
            "candidates": [asdict(candidate) for candidate in candidates],
            "selected_index": selected_index,
            "is_root": self._current_is_root,
        }
        if self._current_is_root:
            self.root_evidence = dict(self.event)
            logits = self.policy_event.get("policy_logits") if self.policy_event else None
            if not isinstance(logits, list) or len(logits) != len(candidates):
                raise RuntimeError("root policy evidence must retain all candidate policy logits")
            self.root_policy_logits = [float(value) for value in logits]


class _SemanticReturn:
    """Compose native discounted rewards without losing intra-edge floor ticks.

    An edge's reward is sum_k(prod_{j<k} Gamma_j * r_k), not the raw sum
    whenever a mechanical suffix crosses a floor before its final reward.
    """

    def __init__(self) -> None:
        self.weight = 1.0
        self.discounted = dict.fromkeys(REWARD_COMPONENTS, 0.0)
        self.undiscounted = dict.fromkeys(REWARD_COMPONENTS, 0.0)
        self.edge_components = dict.fromkeys(REWARD_COMPONENTS, 0.0)
        self.edge_raw_components = dict.fromkeys(REWARD_COMPONENTS, 0.0)
        self.edge_floor: int | None = None
        self.edge_current_floor: int | None = None
        self.edge_discount = 1.0
        self.edge_terminal = False
        self.edge_steps = 0
        self.edges = 0

    def open(self, floor: int) -> None:
        if self.edge_floor is not None:
            raise ValueError("cannot overwrite an open semantic return edge")
        if isinstance(floor, bool) or not isinstance(floor, int) or floor < 0:
            raise ValueError("semantic return floor must be a non-negative integer")
        self.edge_floor = floor
        self.edge_current_floor = floor

    def add(self, breakdown: Mapping[str, Any], *, floor_before: int, floor_after: int, terminal: bool = False) -> None:
        if self.edge_floor is None:
            return
        if floor_before != self.edge_current_floor or floor_after < floor_before:
            raise ValueError("native semantic return floors must be continuous and nondecreasing")
        if self.edge_terminal:
            raise ValueError("cannot add rewards after a factual terminal")
        gamma = decision_discount(floor_before=floor_before, floor_after=floor_after, terminal=terminal)
        values = {name: float(breakdown[name]) for name in REWARD_COMPONENTS}
        if not all(math.isfinite(value) for value in values.values()):
            raise ValueError("non-finite factual reward component")
        for name, value in values.items():
            self.edge_components[name] += self.edge_discount * value
            self.edge_raw_components[name] += value
            self.undiscounted[name] += value
        self.edge_discount *= gamma
        self.edge_current_floor = floor_after
        self.edge_terminal = terminal
        self.edge_steps += 1

    def close(self, floor: int, terminal: bool, trace: EvidenceTrace) -> None:
        if self.edge_floor is None:
            return
        if floor != self.edge_current_floor or terminal != self.edge_terminal:
            raise ValueError("semantic edge close must match its last native floor and terminal flag")
        gamma = self.edge_discount
        expected_gamma = decision_discount(floor_before=self.edge_floor, floor_after=floor, terminal=terminal)
        if not math.isclose(gamma, expected_gamma, rel_tol=1e-12, abs_tol=1e-12):
            raise ValueError("native discount product does not match durable floor edge")
        trace.record(
            "semantic_edge",
            {
                "edge_index": self.edges,
                "floor_before": self.edge_floor,
                "floor_after": floor,
                "native_steps": self.edge_steps,
                "reward_components": self.edge_components,
                "reward_components_semantics": "edge-local native-prefix-discounted",
                "raw_reward_components": self.edge_raw_components,
                "prefix_discount": self.weight,
                "discount": gamma,
                "factual_terminal": terminal,
            },
        )
        for name, value in self.edge_components.items():
            self.discounted[name] += self.weight * value
        self.weight *= gamma
        self.edge_components = dict.fromkeys(REWARD_COMPONENTS, 0.0)
        self.edge_raw_components = dict.fromkeys(REWARD_COMPONENTS, 0.0)
        self.edge_floor = None
        self.edge_current_floor = None
        self.edge_discount = 1.0
        self.edge_terminal = False
        self.edge_steps = 0
        self.edges += 1


def _surface(encoded: Any) -> tuple[dict[str, Any], ...]:
    return tuple(
        {
            "surface_kind": "strict_action_group",
            "prototype": group.prototype,
            "multiplicity": group.reference.multiplicity,
            "equivalence_fingerprint": group.reference.equivalence_fingerprint,
            "enabled": group.reference.enabled,
        }
        for group in encoded.semantic_groups
    )


def run_branch_pass(
    *,
    backend: EnvironmentBackend,
    policy: FrozenPolicy,
    config: TrainingConfig,
    seed: int,
    limits: BranchLimits,
    trace: EvidenceTrace,
    prefix: EvidenceTrace,
    forced_candidate: int | None = None,
    authority: BranchAuthority | None = None,
    allow_zero_budget: bool = False,
    allow_zero_budget_training: bool = False,
    allow_act_objectives: bool = False,
) -> dict[str, Any]:
    """One fresh full-prefix execution. The caller must close its backend.

    An explicitly injected authority owns sampling, and must be bound to this
    exact policy object, seed, limits and intervention. Its declared policy_mode
    is reported rather than pretending a custom sampler is greedy. This hook
    alone does not make evidence training-eligible or validate policy sampling.
    """
    validate_seed(seed)
    if allow_zero_budget and (not limits.terminal_only or authority is not None or forced_candidate is not None):
        raise ValueError("zero-budget override is only for natural terminal-only evaluation")
    if allow_zero_budget_training and (not limits.terminal_only or forced_candidate is not None):
        raise ValueError("zero-budget training requires natural terminal-only episode collection")
    budget = validate_course(
        config,
        allow_zero_budget=(allow_zero_budget or allow_zero_budget_training),
        allow_act_objectives=allow_act_objectives,
    )
    started = time.monotonic()

    def remaining_ms(cap: int = 20_000) -> int:
        remaining = limits.timeout_seconds - (time.monotonic() - started)
        if remaining <= 0:
            raise ProbeBudgetExhausted("timeout_seconds")
        return max(1, min(cap, int(remaining * 1000)))

    if authority is None:
        authority = BranchAuthority(policy, seed, limits, forced_candidate)
    authority.validate_binding(policy=policy, seed=seed, limits=limits, forced_candidate=forced_candidate)
    encoder = GroundedObservationEncoder(config.model.to_encoding_config())
    reward = ActFrontloadedDeathRewardCalculator(
        objective=config.curriculum.reward_objective,
        discount=config.optimization.discount,
        maximum_episode_steps=config.environment.max_episode_steps,
        revival_budget=budget,
    )
    reward.begin_episode()
    revision = backend.get_state()["state_version"]
    state = backend.reset(
        ResetRequest(
            request_id=str(uuid4()),
            session_id=backend.session_id,
            scenario="full-run",
            expected_state_version=revision,
            character=config.environment.character,
            seed=seed,
            force_fresh=True,
            defensive_buffs=False,
            additional_relics=None,
            training_revival_budget=budget,
            timeout_ms=remaining_ms(45_000),
        )
    )
    authority.begin_episode(state.episode_id)
    reset_evidence = {"state": _state_facts(state), "reward_state": reward_state(reward)}
    prefix.record("reset", reset_evidence)
    trace.record("reset", reset_evidence)
    native_steps = 0
    root_native_step: int | None = None
    root_floor: int | None = None
    combats_started = 0
    combats_exited = 0
    combats_completed = 0
    root_verified = False
    returns = _SemanticReturn()
    ending = "no_root"
    while True:
        remaining_ms()
        if state.terminated or state.truncated:
            ending = "factual_terminal" if state.terminated and not state.truncated else "environment_truncated"
            break
        encoded = encoder.encode(state.observation, state.legal_actions)
        semantic_actions = _surface(encoded)
        valid = encoded.snapshot.action_mask.astype(bool, copy=False)
        authority.event = None
        authority.policy_event = None
        frontier = False
        try:
            selected = authority.choose(
                observation=state.observation, semantic_actions=semantic_actions, snapshot=encoded.snapshot, valid=valid
            )
        except _MacroHorizonReached:
            frontier = True
            selected = None
        remaining_ms()
        if authority.last_executor_failure or authority.semantic_encode_collisions:
            raise RuntimeError(f"semantic authority failed closed: {authority.last_executor_failure}")
        if selected is None and not frontier:
            legal = np.flatnonzero(valid)
            if len(legal) != 1:
                raise RuntimeError("authority declined a non-forced action surface")
            selected = int(legal[0])
        event = authority.event
        if returns.edge_floor is not None and (event is not None or frontier):
            returns.close(_floor(state), False, trace)
        reference = encoded.action(selected) if selected is not None else None
        facts = {
            "native_steps": native_steps,
            "state": _state_facts(state),
            "strict_groups": semantic_actions,
            "semantic": event,
            "selected_group": selected,
            "selected_reference": asdict(reference) if reference is not None else None,
            "reward_state": reward_state(reward),
        }
        if event is not None and event["is_root"]:
            root = {
                "native_steps": native_steps,
                "state": _state_facts(state),
                "strict_groups": semantic_actions,
                "semantic_ordinal": event["semantic_ordinal"],
                "surface": event["surface"],
                "candidates": event["candidates"],
                "greedy_index": authority.greedy_root_index,
                "reward_state": reward_state(reward),
            }
            # This exact root checkpoint is compared before the forced native
            # mutation. Chosen intervention is deliberately not a root fact.
            prefix.record("root", root, authority.policy_event)
            prefix.finish()
            root_verified = True
            root_native_step = native_steps
            root_floor = _floor(state)
        elif not root_verified:
            prefix.record("decision", facts, authority.policy_event)
        trace.record("macro_horizon" if frontier else "decision", facts, authority.policy_event)
        if frontier:
            ending = "macro_horizon"
            break
        if native_steps >= limits.max_native_steps:
            raise ProbeBudgetExhausted("max_native_steps")
        assert selected is not None
        if event is not None:
            if root_verified:
                returns.open(_floor(state))
            # Reconstruction source, not a hidden-state replay target. Raw
            # observations are re-encoded under a future learner's identity.
            trace.record(
                "semantic_snapshot",
                {
                    "observation": _state_facts(state)["observation"],
                    "semantic_actions": [candidate["semantic_action"] for candidate in event["candidates"]],
                    "selected_index": event["selected_index"],
                    "domain": event["domain"],
                    "surface": event["surface"],
                    "is_root": event["is_root"],
                    "semantic_ordinal": event["semantic_ordinal"],
                    "encoding_fingerprint": encoded.encoding_fingerprint,
                },
            )
        after = backend.step(dispatch_request(backend, state, encoded, selected, timeout_ms=remaining_ms()))
        native_steps += 1
        if after.truncated:
            trace.record("environment_truncated", {"state": _state_facts(after), "reward_not_evaluated": True})
            state = after
            ending = "environment_truncated"
            break
        breakdown = reward.evaluate(before=state, after=after, deadlock=False, horizon_exhausted=False)
        if root_verified:
            before_payload = state.observation.get("combat")
            after_payload = after.observation.get("combat")
            before_combat = isinstance(before_payload, Mapping) and before_payload.get("in_progress") is True
            after_combat = isinstance(after_payload, Mapping) and after_payload.get("in_progress") is True
            combats_started += int(after_combat and not before_combat)
            combats_exited += int(before_combat and not after_combat)
            combat_result = after.transition.facts.get("combat_result") if after.transition else None
            combats_completed += int(
                before_combat and not after_combat and combat_result in {"victory", "defeat", "escaped"}
            )
        components = asdict(breakdown)
        transition = {
            "native_steps": native_steps,
            "state": _state_facts(after),
            "reward_breakdown": components,
            "reward_state": reward_state(reward),
            "native_discount": {
                "floor_before": _floor(state),
                "floor_after": _floor(after),
                "terminal": breakdown.task_terminal,
                "discount": decision_discount(
                    floor_before=_floor(state), floor_after=_floor(after), terminal=breakdown.task_terminal
                ),
            },
        }
        if not root_verified:
            prefix.record("transition", transition)
        trace.record("transition", transition)
        returns.add(components, floor_before=_floor(state), floor_after=_floor(after), terminal=breakdown.task_terminal)
        state = after
        remaining_ms()
        if breakdown.task_terminal:
            returns.close(_floor(state), True, trace)
            ending = "factual_terminal"
            break
    result = {
        "policy_mode": authority.policy_mode,
        "ending": ending,
        "root_found": root_verified,
        "root": authority.root_evidence,
        "greedy_root_index": authority.greedy_root_index,
        "root_policy_logits": authority.root_policy_logits,
        "selected_root_index": authority.root_evidence["selected_index"] if authority.root_evidence else None,
        "native_steps": native_steps,
        "prefix_native_steps": root_native_step,
        "branch_native_steps": native_steps - root_native_step if root_native_step is not None else 0,
        "root_floor": root_floor,
        "ending_floor": _floor(state),
        "post_root_floor_span": _floor(state) - root_floor if root_floor is not None else 0,
        "post_root_combats_started": combats_started,
        "post_root_combats_exited": combats_exited,
        "post_root_combats_completed": combats_completed,
        "semantic_decisions": authority.semantic_count,
        "macro_decisions": authority.macro_count,
        "combat_decisions": authority.combat_count,
        "continuation_macro_decisions": authority.post_root_macro_count,
        "continuation_combat_decisions": authority.post_root_combat_count,
        "mechanical_dispatches": authority.mechanical_dispatches,
        "semantic_edges": returns.edges,
        "observed_discounted_return": returns.discounted["reward"],
        "discounted_reward_components": returns.discounted,
        "raw_reward_components": returns.undiscounted,
        "tail_discount": returns.weight,
        "tail_value_used": False,
        "training_label_eligible": False,
        "reward_state": reward_state(reward),
    }
    trace.record("end", result)
    trace.finish()
    return result


def run_branch_probe(
    *,
    backend_factory: Callable[[], EnvironmentBackend],
    policy: FrozenPolicy,
    config: TrainingConfig,
    seed: int,
    limits: BranchLimits,
    output_dir: Path,
) -> dict[str, Any]:
    """Reference, identical-action control, then all remaining root candidates."""
    summary: dict[str, Any] = {
        "schema": SCHEMA,
        "status": "failed",
        "seed": seed,
        "limits": asdict(limits),
        "same_action_gate": "not_run",
        "passes": [],
        "comparisons": None,
        "training_writes": False,
        "training_label_eligible": False,
        "tail_value_used": False,
        "independent_root_samples": 1,
        "hidden_rng_clone_verified": False,
        "seed_partition": "consumed_diagnostic_not_final_audit",
        "scope": "simulator-conditional frozen-policy finite-stopping-rule returns; not Q-star or expected regret",
        "control_scores": "policy_logits_not_Q",
        "combat_coverage_definitions": {
            "started": "observed in_progress false-to-true",
            "exited": "observed in_progress true-to-false; no victory/defeat inference",
            "authoritative_completed": "observed exit with typed combat_result victory/defeat/escaped",
        },
        "reward_identity": act_frontloaded_death_reward_identity(revival_budget=validate_course(config)),
        "source_policy": getattr(policy, "provenance", {"source_kind": "injected_test_policy"}),
        "encoding_identity": grounding_encoding_identity(),
        "forward_compiler_version": FORWARD_COMPILER_CONTRACT_VERSION,
        "decision_clock_version": DECISION_CLOCK_CONTRACT_VERSION,
    }

    def execute(name: str, candidate: int | None, *, compare_full: bool = False) -> dict[str, Any]:
        summary["active_pass"] = name
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
                stack.enter_context((output_dir / "reference.jsonl").open(encoding="utf-8")) if compare_full else None
            )
            backend = backend_factory()
            stack.callback(backend.close)
            result = run_branch_pass(
                backend=backend,
                policy=policy,
                config=config,
                seed=seed,
                limits=limits,
                trace=EvidenceTrace(output, limits.trace_limits(), expected_full),
                prefix=EvidenceTrace(prefix_output, limits.trace_limits(), expected_prefix),
                forced_candidate=candidate,
            )
            summary["passes"].append(
                {
                    "name": name,
                    "trace_file": str(output_dir / f"{name}.jsonl"),
                    "prefix_file": str(output_dir / f"{name}.prefix.jsonl"),
                    **result,
                }
            )
            if not result["root_found"] or result["ending"] not in {"factual_terminal", "macro_horizon"}:
                raise ProbeBudgetExhausted("incomplete_root_group:" + result["ending"])
            return result

    try:
        validate_seed(seed)
        reference = execute("reference", None)
        baseline_index = reference["selected_root_index"]
        control = execute("same-action-control", baseline_index, compare_full=True)
        summary["same_action_gate"] = "matched"
        candidates = reference["root"]["candidates"]
        results = {baseline_index: control}
        for index in range(len(candidates)):
            if index != baseline_index:
                results[index] = execute(f"candidate-{index:03d}", index)
        # Do not publish partial-group rankings. A terminal is the absorbing
        # early end of the SAME terminal-or-macro-horizon stopping rule.
        baseline = control["observed_discounted_return"]
        summary["comparisons"] = [
            {
                "candidate_index": index,
                "candidate": candidate,
                "policy_logit": reference["root_policy_logits"][index],
                "observed_discounted_return": results[index]["observed_discounted_return"],
                "difference_from_greedy_control": results[index]["observed_discounted_return"] - baseline,
                "ending": results[index]["ending"],
                "tail_discount": results[index]["tail_discount"],
            }
            for index, candidate in enumerate(candidates)
        ]
        summary.update(
            status="complete", complete_candidate_count=len(candidates), all_candidates_same_fixed_stopping_rule=True
        )
    except ProbeMismatch as exc:
        if summary["same_action_gate"] != "matched":
            summary["same_action_gate"] = "mismatch"
        summary.update(status="mismatch", difference=exc.detail)
    except ProbeBudgetExhausted as exc:
        summary.update(status="inconclusive", reason=str(exc))
    except Exception as exc:
        summary.update(status="failed", error_type=type(exc).__name__, error=str(exc))
    return summary


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    model = parser.add_mutually_exclusive_group(required=True)
    model.add_argument("--model-path", type=Path)
    model.add_argument("--initialize-from-iql", type=Path)
    parser.add_argument("--sim-exe", type=Path)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--device", choices=("cpu", "cuda"), required=True)
    parser.add_argument("--root-surface", choices=SURFACES, default="reward")
    parser.add_argument("--root-occurrence", type=int, default=1)
    parser.add_argument("--continuation-macro-decisions", type=int, default=4)
    parser.add_argument("--terminal-only", action="store_true")
    parser.add_argument("--max-native-steps", type=int, default=4096)
    parser.add_argument("--max-candidates", type=int, default=64)
    parser.add_argument("--timeout-seconds", type=float, default=300.0)
    parser.add_argument("--max-trace-bytes", type=int, default=128 * 1024 * 1024)
    args = parser.parse_args(argv)
    limits = BranchLimits(
        **{
            name: getattr(args, name)
            for name in (
                "root_surface",
                "root_occurrence",
                "continuation_macro_decisions",
                "terminal_only",
                "max_native_steps",
                "max_candidates",
                "timeout_seconds",
                "max_trace_bytes",
            )
        }
    )
    config = load_training_config(profile="preheat", config_path=args.config)
    validate_course(config)
    validate_seed(args.seed)
    output = new_output_dir(args.output_dir)
    try:
        simulator = verify_headless_simulator(resolve_headless_sim_exe(args.sim_exe or config.environment.sim_exe_path))
        policy = FrozenMacroRootPolicy(
            args.initialize_from_iql or args.model_path,
            config,
            args.device,
            initialize_from_iql=args.initialize_from_iql is not None,
        )
        summary = run_branch_probe(
            backend_factory=lambda: HeadlessBackend(
                exe_path=simulator.executable, protocol_quarantine_dir=output / "protocol-incidents"
            ),
            policy=policy,
            config=config,
            seed=args.seed,
            limits=limits,
            output_dir=output,
        )
        summary.update(simulator=simulator.to_mapping(), config_path=str(args.config.resolve()))
    except Exception as exc:
        summary = {"schema": SCHEMA, "status": "failed", "error_type": type(exc).__name__, "error": str(exc)}
    (output / "summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False, allow_nan=False) + "\n", encoding="utf-8"
    )
    print(json.dumps({"status": summary["status"], "summary_path": str(output / "summary.json")}, ensure_ascii=False))
    return 0 if summary["status"] == "complete" else 2


if __name__ == "__main__":
    raise SystemExit(main())
