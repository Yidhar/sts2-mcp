"""B0: bounded full-prefix reconstruction, without a learner or branch targets.

Run with ``python -m sts2_rl.macro_root_diagnostic --help``. Two isolated
HeadlessSim processes execute the same seed and frozen greedy policy. Every
replayed action is compared BEFORE dispatch; reward-v9 and recurrent state
are rebuilt from reset, never copied from a mid-run observation. This tests
observable reproducibility on one prefix, not hidden RNG cloning or learning.
"""

from __future__ import annotations

import argparse
import json
import math
import time
from collections.abc import Callable, Mapping, Sequence
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Literal, Protocol, TextIO
from uuid import uuid4

import numpy as np
import torch
from torch import Tensor

from sts2_baseline import ActFrontloadedDeathRewardCalculator
from sts2_env.headless_sim_bridge_client import resolve_headless_sim_exe
from sts2_rl.artifacts import artifact_root
from sts2_rl.backends import HeadlessBackend
from sts2_rl.contracts import EnvironmentBackend, EnvironmentResult, ResetRequest, StepRequest
from sts2_rl.encoding import EncodedDecisionSnapshot, GroundedObservationEncoder, grounding_encoding_identity
from sts2_rl.joint.authority import HybridCollectionAuthority
from sts2_rl.macro_root_evidence import TRANSPORT_IDENTITY_KEYS, canonical_facts, first_difference, reward_state
from sts2_rl.macro_root_policy import FrozenMacroRootPolicy
from sts2_rl.models.hybrid_actor_critic import hierarchical_greedy_action_index
from sts2_rl.semantics.clock import DECISION_CLOCK_CONTRACT_VERSION, decision_discount
from sts2_rl.semantics.forward import FORWARD_COMPILER_CONTRACT_VERSION, ForwardDecision, SemanticCandidate
from sts2_rl.simulator_identity import verify_headless_simulator
from sts2_rl.training.config import TrainingConfig, load_training_config


class FrozenPolicy(Protocol):
    def initial_state(self) -> Tensor: ...

    def forward(self, snapshot: EncodedDecisionSnapshot, hidden: Tensor | None) -> tuple[Tensor, Tensor]: ...


@dataclass(frozen=True)
class ProbeLimits:
    max_native_steps: int = 512
    continuation_decisions: int = 2
    root_occurrence: int = 1
    timeout_seconds: float = 180.0
    max_trace_bytes: int = 64 * 1024 * 1024
    policy_atol: float = 1e-6
    policy_rtol: float = 1e-5

    def __post_init__(self) -> None:
        for name in ("max_native_steps", "root_occurrence", "max_trace_bytes"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
                raise ValueError(f"{name} must be a positive integer")
        if (
            isinstance(self.continuation_decisions, bool)
            or not isinstance(self.continuation_decisions, int)
            or self.continuation_decisions < 0
        ):
            raise ValueError("continuation_decisions must be a non-negative integer")
        for name in ("timeout_seconds", "policy_atol", "policy_rtol"):
            value = getattr(self, name)
            if not math.isfinite(value) or value < 0 or (name == "timeout_seconds" and value == 0):
                raise ValueError(f"invalid {name}")


class ProbeMismatch(RuntimeError):
    def __init__(self, detail: dict[str, Any]) -> None:
        self.detail = detail
        super().__init__(json.dumps(detail, ensure_ascii=False))


class ProbeBudgetExhausted(RuntimeError):
    """An administrative censoring boundary, not a task death or reward."""


class _FrontierReached(Exception):
    pass


class EvidenceTrace:
    """Stream evidence with bounded disk usage and one expected row in memory."""

    def __init__(self, output: TextIO, limits: ProbeLimits, reference: TextIO | None = None) -> None:
        self.output = output
        self.reference = reference
        self.limits = limits
        self.rows = 0
        self.bytes_written = 0

    def record(self, kind: str, facts: Any, policy: Any = None) -> None:
        row = {"kind": kind, "facts": canonical_facts(facts), "policy": canonical_facts(policy)}
        encoded = json.dumps(row, ensure_ascii=False, allow_nan=False, separators=(",", ":")) + "\n"
        size = len(encoded.encode("utf-8"))
        if self.bytes_written + size > self.limits.max_trace_bytes:
            raise ProbeBudgetExhausted("max_trace_bytes")
        # Retain the actual mismatching evidence as well as a compact summary.
        self.output.write(encoded)
        self.output.flush()
        self.bytes_written += size
        if self.reference is not None:
            line = self.reference.readline()
            if not line:
                raise ProbeMismatch({"row": self.rows, "path": "$", "kind": "unexpected_extra_row"})
            expected = json.loads(line)
            for section in ("kind", "facts", "policy"):
                tolerances = (
                    {"float_atol": self.limits.policy_atol, "float_rtol": self.limits.policy_rtol}
                    if section == "policy"
                    else {}
                )
                difference = first_difference(expected[section], row[section], **tolerances)
                if difference is not None:
                    raise ProbeMismatch({"row": self.rows, "section": section, **difference})
        self.rows += 1

    def finish(self) -> None:
        if self.reference is not None and self.reference.readline():
            raise ProbeMismatch({"row": self.rows, "path": "$", "kind": "missing_expected_row"})


def _tensor_evidence(value: Tensor | None) -> Any:
    if value is None:
        return None
    if not bool(torch.isfinite(value).all().item()):
        raise FloatingPointError("non-finite recurrent evidence")
    return {"shape": list(value.shape), "dtype": str(value.dtype), "values": value.detach().cpu().tolist()}


class ProbeAuthority(HybridCollectionAuthority):
    """Use the production compiler/executor and recurrence, but retain no replay."""

    def __init__(self, policy: FrozenPolicy, seed: int, limits: ProbeLimits) -> None:
        super().__init__(
            forward_q=policy.forward,
            initial_state=policy.initial_state,
            epsilon=0.0,
            seed=seed,
            evaluation_ownership=True,
        )
        self._policy_ref = policy
        self.limits = limits
        self.semantic_count = 0
        self.reward_roots_seen = 0
        self.root_ordinal: int | None = None
        self.root_evidence: dict[str, Any] | None = None
        self.event: dict[str, Any] | None = None
        self.policy_event: dict[str, Any] | None = None

    def _semantic_choice(self, *, candidates: Sequence[SemanticCandidate], snapshot: EncodedDecisionSnapshot) -> int:
        before = _tensor_evidence(self._hidden)
        selected = super()._semantic_choice(candidates=candidates, snapshot=snapshot)
        if self.forward_retries:
            raise FloatingPointError("diagnostic refuses recovered non-finite policy forwards")
        assert self.last_q_values is not None
        if getattr(self._policy_ref, "hierarchical_branch_greedy", False) and self.epsilon == 0.0:
            selected = hierarchical_greedy_action_index(
                torch.tensor(self.last_q_values.tolist(), dtype=torch.float32),
                torch.tensor(snapshot.candidates.ids[:, 13].tolist(), dtype=torch.int64),
                torch.tensor(snapshot.action_mask.tolist(), dtype=torch.bool),
                domain_id=snapshot.domain_id,
            )
        self.policy_event = {
            "hidden_before": before,
            "hidden_after": _tensor_evidence(self._hidden),
            "policy_logits": self.last_q_values.tolist(),
        }
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
        del observation
        self.semantic_count += 1
        is_reward_root = (
            control_domain == "macro"
            and decision.surface == "reward"
            and {"take", "skip"}.issubset({candidate.branch for candidate in candidates})
        )
        if is_reward_root:
            self.reward_roots_seen += 1
            if self.reward_roots_seen == self.limits.root_occurrence:
                self.root_ordinal = self.semantic_count
        self.event = {
            "semantic_ordinal": self.semantic_count,
            "surface": decision.surface,
            "domain": control_domain,
            "candidates": [asdict(candidate) for candidate in candidates],
            "selected_index": selected_index,
            "is_root": self.root_ordinal == self.semantic_count,
        }
        if self.event["is_root"]:
            self.root_evidence = self.event
        # Stop BEFORE dispatch at the next meaningful frontier, after the root
        # action plus N following semantic decisions and all their suffixes.
        if (
            self.root_ordinal is not None
            and self.semantic_count > self.root_ordinal + self.limits.continuation_decisions
        ):
            raise _FrontierReached

    def _open_transition(self, **kwargs: Any) -> None:
        del kwargs  # No learner/replay transition is opened in a diagnostic.

    def _close_open(self, **kwargs: Any) -> None:
        del kwargs
        self._open = None


def _state_facts(state: EnvironmentResult) -> dict[str, Any]:
    if not state.ok:
        raise RuntimeError("environment returned ok=false")
    if state.transition is None or state.info.get("reward_authority") != "external-rl":
        raise RuntimeError("diagnostic requires typed external-rl transition facts")

    def envelope(value: Mapping[str, Any]) -> dict[str, Any]:
        # Only these explicitly selected DTO envelope levels are transport.
        # Never recursively strip similarly named fields from an entity,
        # modifier, _training block, transition facts, or action candidate.
        return {key: item for key, item in value.items() if key not in TRANSPORT_IDENTITY_KEYS}

    observation = envelope(state.observation)
    if isinstance(observation.get("_sim_raw"), Mapping):
        observation["_sim_raw"] = envelope(observation["_sim_raw"])
    return {
        "step_index": state.step_index,
        "observation": observation,
        "legal_actions": state.legal_actions,
        "transition": envelope(state.transition.to_mapping()),
        "terminated": state.terminated,
        "truncated": state.truncated,
        "terminal_reason": state.terminal_reason,
    }


def _floor(state: EnvironmentResult) -> int:
    run = state.observation.get("run")
    floor = run.get("floor") if isinstance(run, Mapping) else None
    if isinstance(floor, bool) or not isinstance(floor, int) or floor < 0:
        raise ValueError("diagnostic requires a factual non-negative durable floor")
    return floor


def validate_course(config: TrainingConfig, *, allow_zero_budget: bool = False) -> int:
    if config.environment.backend != "headless" or config.environment.scenario != "full-run":
        raise ValueError("B0 requires the isolated headless full-run course")
    if config.curriculum.mode != "native-revival-preheat" or config.curriculum.reward_objective != "run":
        raise ValueError("B0 requires the native-revival-preheat run objective")
    budget = config.curriculum.revival_budget
    if isinstance(budget, bool) or not isinstance(budget, int) or budget < 0 or (budget == 0 and not allow_zero_budget):
        raise ValueError("B0 reward-v9 requires a positive revival budget")
    return budget


def validate_seed(seed: int) -> None:
    if isinstance(seed, bool) or not isinstance(seed, int) or not 0 <= seed <= 2_147_483_647:
        raise ValueError("seed must be a signed-int32 non-negative integer")


def dispatch_request(
    backend: EnvironmentBackend, state: EnvironmentResult, encoded: Any, selected_group: int, *, timeout_ms: int
) -> StepRequest:
    """Resolve a CURRENT strict-group representative, never a prior-run handle."""
    if not 0 <= selected_group < len(encoded.snapshot.action_mask) or not bool(
        encoded.snapshot.action_mask[selected_group]
    ):
        raise ValueError("selected strict group is not enabled")
    reference = encoded.action(selected_group)
    if not 0 <= reference.position < len(state.legal_actions):
        raise ValueError("strict-group representative is outside current native surface")
    current = state.legal_actions[reference.position]
    raw_handle = current.get("action_handle", current.get("action_id"))
    handle = str(raw_handle) if raw_handle is not None and str(raw_handle) else None
    if reference.handle != handle:
        raise ValueError("strict-group handle differs from current native representative")
    if not reference.enabled or not bool(current.get("is_enabled", current.get("enabled", True))):
        raise ValueError("current native representative is disabled")
    return StepRequest(
        request_id=str(uuid4()),
        session_id=backend.session_id,
        episode_id=state.episode_id,
        expected_step_index=state.step_index,
        action_id=reference.handle or None,
        action_index=None if reference.handle else reference.position,
        timeout_ms=timeout_ms,
    )


def run_pass(
    *,
    backend: EnvironmentBackend,
    policy: FrozenPolicy,
    config: TrainingConfig,
    seed: int,
    limits: ProbeLimits,
    trace: EvidenceTrace,
) -> dict[str, Any]:
    """One reset-to-root replay; caller owns backend lifetime (always close it)."""
    validate_seed(seed)
    revival_budget = validate_course(config)
    started = time.monotonic()

    def remaining_ms(cap: int) -> int:
        remaining = limits.timeout_seconds - (time.monotonic() - started)
        if remaining <= 0:
            raise ProbeBudgetExhausted("timeout_seconds")
        return max(1, min(cap, int(remaining * 1000)))

    authority = ProbeAuthority(policy, seed, limits)
    encoder = GroundedObservationEncoder(config.model.to_encoding_config())
    reward = ActFrontloadedDeathRewardCalculator(
        objective=config.curriculum.reward_objective,
        discount=config.optimization.discount,
        maximum_episode_steps=config.environment.max_episode_steps,
        revival_budget=revival_budget,
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
            training_revival_budget=config.curriculum.revival_budget,
            timeout_ms=remaining_ms(45_000),
        )
    )
    authority.begin_episode(state.episode_id)
    trace.record("reset", {"state": _state_facts(state), "reward_state": reward_state(reward)})
    native_steps = 0
    total_reward = 0.0
    end = ""
    while True:
        remaining_ms(20_000)
        if state.terminated or state.truncated:
            end = "factual_terminal" if state.terminated and not state.truncated else "environment_truncated"
            break
        if native_steps >= limits.max_native_steps:
            raise ProbeBudgetExhausted("max_native_steps")
        encoded = encoder.encode(state.observation, state.legal_actions)
        valid = encoded.snapshot.action_mask.astype(bool, copy=False)
        semantic_actions = tuple(
            {
                "surface_kind": "strict_action_group",
                "prototype": group.prototype,
                "multiplicity": group.reference.multiplicity,
                "equivalence_fingerprint": group.reference.equivalence_fingerprint,
                "enabled": group.reference.enabled,
            }
            for group in encoded.semantic_groups
        )
        authority.event = None
        authority.policy_event = None
        frontier = False
        try:
            selected = authority.choose(
                observation=state.observation, semantic_actions=semantic_actions, snapshot=encoded.snapshot, valid=valid
            )
        except _FrontierReached:
            selected = None
            frontier = True
        remaining_ms(20_000)
        if authority.last_executor_failure or authority.semantic_encode_collisions:
            raise RuntimeError(f"semantic authority failed closed: {authority.last_executor_failure}")
        if selected is None and not frontier:
            indices = np.flatnonzero(valid)
            if len(indices) != 1:
                raise RuntimeError("authority declined a non-forced action surface")
            selected = int(indices[0])
        reference = encoded.action(selected) if selected is not None else None
        trace.record(
            "frontier" if frontier else "decision",
            {
                "native_steps": native_steps,
                "state": _state_facts(state),
                "strict_groups": semantic_actions,
                "semantic": authority.event,
                "selected_group": selected,
                "selected_reference": asdict(reference) if reference is not None else None,
                "reward_state": reward_state(reward),
            },
            authority.policy_event,
        )
        if frontier:
            end = "continuation_frontier"
            break
        assert selected is not None
        request = dispatch_request(backend, state, encoded, selected, timeout_ms=remaining_ms(20_000))
        after = backend.step(request)
        if after.truncated:
            # An unknown transport outcome must not be evaluated as task
            # failure or assigned a reward/terminal bootstrap mask.
            trace.record(
                "environment_truncated",
                {
                    "native_steps": native_steps + 1,
                    "state": _state_facts(after),
                    "reward_state": reward_state(reward),
                    "reward_not_evaluated": True,
                },
            )
            native_steps += 1
            end = "environment_truncated"
            break
        breakdown = reward.evaluate(before=state, after=after, deadlock=False, horizon_exhausted=False)
        trace.record(
            "transition",
            {
                "native_steps": native_steps + 1,
                "state": _state_facts(after),
                "reward_breakdown": asdict(breakdown),
                "reward_state": reward_state(reward),
                "native_floor_discount": decision_discount(
                    floor_before=_floor(state), floor_after=_floor(after), terminal=breakdown.task_terminal
                ),
            },
        )
        total_reward += breakdown.reward
        native_steps += 1
        state = after
        remaining_ms(20_000)
        if breakdown.task_terminal:
            end = "factual_terminal"
            break
    outcome = {
        "end": end,
        "root_found": authority.root_ordinal is not None,
        "root": authority.root_evidence,
        "native_steps": native_steps,
        "semantic_forwards": authority.semantic_count,
        "reward_total": total_reward,
        "reward_state": reward_state(reward),
        "trace_rows": trace.rows + 1,  # Include the end row written immediately below.
        "mechanical_dispatches": authority.mechanical_dispatches,
    }
    trace.record("end", outcome)
    trace.finish()
    return outcome


def run_probe(
    *,
    backend_factory: Callable[[], EnvironmentBackend],
    policy: FrozenPolicy,
    config: TrainingConfig,
    seed: int,
    limits: ProbeLimits,
    output_dir: Path,
) -> dict[str, Any]:
    """Write reference/replay streams; fail on first divergence without retry."""
    summary: dict[str, Any] = {
        "schema": "sts2-macro-root-b0-v1",
        "seed": seed,
        "limits": asdict(limits),
        "status": "failed",
        "passes": [],
        "training_writes": False,
        "hidden_rng_clone_verified": False,
        "independent_statistical_samples": 1,
        "scope": "same-action observable reproducibility; no branch ranking or learning target",
        "transport_keys_excluded": sorted(TRANSPORT_IDENTITY_KEYS),
        "transport_exclusion_scope": ["state.observation", "state.observation._sim_raw", "state.transition"],
        "seed_partition": "consumed_diagnostic_not_final_audit",
        "encoding_identity": grounding_encoding_identity(),
        "forward_compiler_version": FORWARD_COMPILER_CONTRACT_VERSION,
        "decision_clock_version": DECISION_CLOCK_CONTRACT_VERSION,
        "effective_reset": {
            "scenario": "full-run",
            "character": config.environment.character,
            "seed": seed,
            "force_fresh": True,
            "defensive_buffs": False,
            "additional_relics": None,
            "training_revival_budget": config.curriculum.revival_budget,
        },
    }
    try:
        validate_seed(seed)
        validate_course(config)
        for name in ("reference", "replay"):
            with (output_dir / f"{name}.jsonl").open("x", encoding="utf-8", newline="\n") as output:
                reference = (output_dir / "reference.jsonl").open(encoding="utf-8") if name == "replay" else None
                backend: EnvironmentBackend | None = None
                try:
                    backend = backend_factory()
                    trace = EvidenceTrace(output, limits, reference)
                    result = run_pass(
                        backend=backend, policy=policy, config=config, seed=seed, limits=limits, trace=trace
                    )
                    summary["passes"].append({"name": name, **result})
                finally:
                    if reference is not None:
                        reference.close()
                    if backend is not None:
                        backend.close()
            if not result["root_found"] or result["end"] == "environment_truncated":
                summary.update(status="inconclusive", reason="no_root_or_environment_truncation")
                break
        else:
            summary["status"] = "matched"
    except ProbeMismatch as exc:
        summary.update(status="mismatch", difference=exc.detail)
    except ProbeBudgetExhausted as exc:
        summary.update(status="inconclusive", reason=str(exc))
    except Exception as exc:
        summary.update(status="failed", error_type=type(exc).__name__, error=str(exc))
    return summary


def new_output_dir(path: Path) -> Path:
    resolved = path.expanduser().resolve()
    root = artifact_root().resolve()
    checkout = Path(__file__).resolve().parents[3]
    if resolved == root or not resolved.is_relative_to(root) or resolved.is_relative_to(checkout):
        raise ValueError("output must be a NEW child of STS2_ARTIFACT_ROOT, outside the checkout")
    resolved.mkdir(parents=True, exist_ok=False)
    return resolved


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    model = parser.add_mutually_exclusive_group(required=True)
    model.add_argument("--model-path", type=Path, help="frozen V-trace publication")
    model.add_argument("--initialize-from-iql", type=Path, help="explicit in-memory initialization, never resume")
    parser.add_argument("--sim-exe", type=Path)
    parser.add_argument("--seed", type=int, required=True, help="diagnostic-only seed; NOT a final-audit sample")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--device", choices=("cpu", "cuda"), required=True)
    parser.add_argument("--max-native-steps", type=int, default=512)
    parser.add_argument("--continuation-decisions", type=int, default=2)
    parser.add_argument("--root-occurrence", type=int, default=1)
    parser.add_argument("--timeout-seconds", type=float, default=180.0, help="per pass, excluding process startup")
    parser.add_argument("--max-trace-bytes", type=int, default=64 * 1024 * 1024)
    args = parser.parse_args(argv)
    limits = ProbeLimits(
        max_native_steps=args.max_native_steps,
        continuation_decisions=args.continuation_decisions,
        root_occurrence=args.root_occurrence,
        timeout_seconds=args.timeout_seconds,
        max_trace_bytes=args.max_trace_bytes,
    )
    config = load_training_config(profile="preheat", config_path=args.config)
    validate_course(config)
    validate_seed(args.seed)
    output = new_output_dir(args.output_dir)
    summary: dict[str, Any]
    try:
        simulator = verify_headless_simulator(resolve_headless_sim_exe(args.sim_exe or config.environment.sim_exe_path))
        policy = FrozenMacroRootPolicy(
            args.initialize_from_iql or args.model_path,
            config,
            args.device,
            initialize_from_iql=args.initialize_from_iql is not None,
        )
        summary = run_probe(
            backend_factory=lambda: HeadlessBackend(
                exe_path=simulator.executable, protocol_quarantine_dir=output / "protocol-incidents"
            ),
            policy=policy,
            config=config,
            seed=args.seed,
            limits=limits,
            output_dir=output,
        )
        summary.update(
            policy=policy.provenance, simulator=simulator.to_mapping(), config_path=str(args.config.resolve())
        )
    except Exception as exc:
        summary = {
            "schema": "sts2-macro-root-b0-v1",
            "status": "failed",
            "error_type": type(exc).__name__,
            "error": str(exc),
        }
    (output / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8"
    )
    print(json.dumps({"status": summary["status"], "summary_path": str(output / "summary.json")}, ensure_ascii=False))
    return 0 if summary["status"] == "matched" else 2


if __name__ == "__main__":
    raise SystemExit(main())
