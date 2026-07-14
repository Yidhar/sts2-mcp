"""Typed recurrent actor collecting fixed-length v2 sequence unrolls."""

from __future__ import annotations

import math
import time
from collections.abc import Callable, Mapping
from copy import deepcopy
from dataclasses import dataclass
from typing import Literal, Protocol
from uuid import uuid4

import numpy as np
import torch
from numpy.typing import NDArray

from sts2_baseline import (
    RolloutStep,
    SequenceUnroll,
    TaskReward,
    TaskRewardCalculator,
)
from sts2_rl.contracts import (
    CombatResetRequest,
    EnvironmentBackend,
    EnvironmentResult,
    ResetRequest,
    StepRequest,
)
from sts2_rl.encoding import EncodedDecisionSnapshot, GroundedObservationEncoder
from sts2_rl.models import RecurrentCandidateModel

from .seeding import (
    EVALUATION_SEED_PARITY,
    SIGNED_INT32_MAX,
    training_seed_start,
)
from .trajectory import (
    SemanticDeadlockDetector,
    TrajectoryJournal,
    semantic_projection,
)


class CollectionProtocolError(RuntimeError):
    """The environment exposed no dispatchable legal candidate."""


class RewardCalculator(Protocol):
    def evaluate(
        self,
        before: EnvironmentResult,
        after: EnvironmentResult,
        *,
        deadlock: bool = False,
        horizon_exhausted: bool = False,
    ) -> TaskReward: ...


@dataclass(frozen=True, slots=True)
class EpisodeMetrics:
    episode_id: str
    reset_seed: int
    steps: int
    reward_total: float
    terminal_reason: str | None
    truncated: bool
    run_won: bool
    combat_won: bool
    act1_cleared: bool
    max_act: int
    max_floor: int
    policy_decisions: int
    forced_decisions: int
    deadlocked: bool
    revivals_used: int
    revival_free_combat_win: bool
    revival_free_act1_clear: bool
    revival_free_run_win: bool
    player_hp_lost: float


@dataclass(frozen=True, slots=True)
class CollectedEpisode:
    unrolls: tuple[SequenceUnroll, ...]
    metrics: EpisodeMetrics
    timings: CollectorTimings | None = None


@dataclass(frozen=True, slots=True)
class CollectorStageTiming:
    count: int
    total_ms: float
    min_ms: float
    max_ms: float

    def to_mapping(self) -> dict[str, float | int]:
        return {
            "count": self.count,
            "total_ms": self.total_ms,
            "mean_ms": self.total_ms / self.count,
            "min_ms": self.min_ms,
            "max_ms": self.max_ms,
        }


@dataclass(frozen=True, slots=True)
class CollectorTimings:
    """Episode-aggregated collector timings; never written per environment step."""

    stages: dict[str, CollectorStageTiming]

    def to_mapping(self) -> dict[str, dict[str, float | int]]:
        return {
            name: timing.to_mapping()
            for name, timing in sorted(self.stages.items())
        }


@dataclass(frozen=True, slots=True)
class _ActionChoice:
    action_index: int
    behavior_log_probability: float
    valid_count: int
    snapshot: EncodedDecisionSnapshot
    recurrent_state: torch.Tensor
    policy: NDArray[np.float32]
    value: float
    encoding_ms: float
    policy_forward_ms: float


@dataclass(slots=True)
class _MutableStageTiming:
    count: int = 0
    total_ms: float = 0.0
    min_ms: float = float("inf")
    max_ms: float = 0.0

    def add(self, duration_ms: float) -> None:
        duration_ms = max(0.0, float(duration_ms))
        self.count += 1
        self.total_ms += duration_ms
        self.min_ms = min(self.min_ms, duration_ms)
        self.max_ms = max(self.max_ms, duration_ms)


class _CollectorTimingAccumulator:
    def __init__(self) -> None:
        self._stages: dict[str, _MutableStageTiming] = {}

    def add(self, stage: str, duration_ms: float) -> None:
        self._stages.setdefault(stage, _MutableStageTiming()).add(duration_ms)

    def record(self, stage: str, started_ns: int) -> None:
        self.add(stage, (time.perf_counter_ns() - started_ns) / 1_000_000.0)

    def snapshot(self) -> CollectorTimings:
        return CollectorTimings(
            stages={
                name: CollectorStageTiming(
                    count=timing.count,
                    total_ms=timing.total_ms,
                    min_ms=timing.min_ms,
                    max_ms=timing.max_ms,
                )
                for name, timing in self._stages.items()
            }
        )


def _number(value: object, default: float = 0.0) -> float:
    if not isinstance(value, str | int | float):
        return float(default)
    try:
        return float(value)
    except (TypeError, ValueError):
        return float(default)


def _run_position(observation: Mapping[str, object]) -> tuple[int, int]:
    raw_run = observation.get("run")
    run = raw_run if isinstance(raw_run, Mapping) else {}
    act = int(max(0.0, _number(run.get("act", observation.get("act")))))
    floor = int(max(0.0, _number(run.get("floor", observation.get("floor")))))
    return act, floor


class GroundedCollector:
    """Collect policy trajectories with no MCTS, guard, or action rewrite."""

    def __init__(
        self,
        *,
        model: RecurrentCandidateModel,
        encoder: GroundedObservationEncoder,
        backend: EnvironmentBackend,
        scenario: Literal["full-run", "combat"],
        objective: Literal["combat", "act1", "run"],
        discount: float,
        max_episode_steps: int,
        character: str | None = None,
        encounter_id: str | None = None,
        seed: int = 0,
        unroll_length: int = 64,
        deadlock_window: int = 128,
        deadlock_repeat_threshold: int = 8,
        journal_policy_topk: int = 5,
        reward_calculator: RewardCalculator | None = None,
        additional_relics: tuple[str, ...] = (),
        revival_relic_id: str | None = None,
        training_revival_budget: int | None = None,
        horizon_as_failure: bool = False,
    ) -> None:
        if scenario not in {"full-run", "combat"}:
            raise ValueError("scenario must be full-run or combat")
        if objective not in {"combat", "act1", "run"}:
            raise ValueError("objective must be combat, act1, or run")
        if scenario == "combat" and objective != "combat":
            raise ValueError("combat scenario requires the combat objective")
        if scenario == "full-run" and objective == "combat":
            raise ValueError("full-run scenario requires the act1 or run objective")
        if (
            isinstance(discount, bool)
            or not isinstance(discount, int | float)
            or not math.isfinite(float(discount))
            or not 0.0 < float(discount) <= 1.0
        ):
            raise ValueError("discount must be in (0, 1]")
        if (
            isinstance(max_episode_steps, bool)
            or not isinstance(max_episode_steps, int)
            or max_episode_steps <= 0
        ):
            raise ValueError("max_episode_steps must be positive")
        if isinstance(seed, bool) or not isinstance(seed, int) or seed < 0:
            raise ValueError("collector seed must be a non-negative integer")
        if isinstance(unroll_length, bool) or not isinstance(unroll_length, int):
            raise TypeError("unroll_length must be an integer")
        if unroll_length <= 0:
            raise ValueError("unroll_length must be positive")
        if isinstance(journal_policy_topk, bool) or not isinstance(
            journal_policy_topk, int
        ):
            raise TypeError("journal_policy_topk must be an integer")
        if journal_policy_topk <= 0:
            raise ValueError("journal_policy_topk must be positive")
        self.model = model
        self.encoder = encoder
        self.backend = backend
        self.scenario = scenario
        self.objective = objective
        self.discount = float(discount)
        self.max_episode_steps = int(max_episode_steps)
        self.character = character
        self.encounter_id = encounter_id
        self.unroll_length = unroll_length
        self.journal_policy_topk = journal_policy_topk
        self.additional_relics = tuple(str(item) for item in additional_relics)
        self.revival_relic_id = (
            revival_relic_id.strip().upper()
            if isinstance(revival_relic_id, str) and revival_relic_id.strip()
            else None
        )
        self.training_revival_budget = training_revival_budget
        self.horizon_as_failure = bool(horizon_as_failure)
        if self.revival_relic_id is not None and self.revival_relic_id not in {
            item.upper() for item in self.additional_relics
        }:
            raise ValueError("revival_relic_id must be one of the injected additional relics")
        if self.training_revival_budget is not None:
            if self.revival_relic_id is None:
                raise ValueError(
                    "training_revival_budget requires an injected revival relic"
                )
            if self.training_revival_budget < -1:
                raise ValueError("training_revival_budget must be -1 or non-negative")
        self._rng = np.random.default_rng(int(seed))
        self._episode_seed = training_seed_start(int(seed))
        self._active_state_version: int | None = None
        self.reward_calculator = reward_calculator or TaskRewardCalculator(
            objective,
            discount=discount,
        )
        self.deadlock_detector = SemanticDeadlockDetector(
            window_size=deadlock_window,
            repeat_threshold=deadlock_repeat_threshold,
        )

    @property
    def device(self) -> torch.device:
        return next(self.model.parameters()).device

    def state_dict(self) -> dict[str, object]:
        """Return every collector-owned stochastic continuation input.

        Checkpoints are only published between episodes, so no live environment
        state belongs here.  The next reset seed and the epsilon-action RNG are
        nevertheless part of an exact continuation and must not silently reset.
        """

        return {
            "version": "sts2-recurrent-collector-state-v3",
            "episode_seed": self._episode_seed,
            "rng_state": deepcopy(self._rng.bit_generator.state),
        }

    def load_state_dict(self, payload: Mapping[str, object]) -> None:
        """Restore a fail-closed collector continuation state."""

        expected_keys = {"version", "episode_seed", "rng_state"}
        actual_keys = set(payload)
        if actual_keys != expected_keys:
            raise ValueError(
                "collector state keys mismatch: "
                f"missing={sorted(expected_keys - actual_keys)} "
                f"unknown={sorted(actual_keys - expected_keys)}"
            )
        if payload["version"] != "sts2-recurrent-collector-state-v3":
            raise ValueError("unsupported collector checkpoint state")
        episode_seed = payload["episode_seed"]
        if isinstance(episode_seed, bool) or not isinstance(episode_seed, int):
            raise TypeError("collector episode_seed must be an integer")
        if episode_seed < 0:
            raise ValueError("collector episode_seed must be non-negative")
        if episode_seed > SIGNED_INT32_MAX or episode_seed % 2 != 0:
            raise ValueError("collector episode_seed must be an even signed 32-bit seed")
        rng_state = payload["rng_state"]
        if not isinstance(rng_state, Mapping):
            raise TypeError("collector rng_state must be a mapping")
        candidate_rng = np.random.default_rng()
        try:
            candidate_rng.bit_generator.state = dict(deepcopy(rng_state))
        except (TypeError, ValueError) as exc:
            raise ValueError("collector rng_state is invalid") from exc
        self._episode_seed = episode_seed
        self._rng = candidate_rng

    def _state_version(self) -> int:
        state = self.backend.get_state()
        if not isinstance(state, Mapping) or state.get("ok") is not True:
            raise CollectionProtocolError("backend state read was not explicitly successful")
        revision = state.get("state_version")
        if isinstance(revision, bool) or not isinstance(revision, int) or revision < 0:
            raise CollectionProtocolError(
                "backend state_version must be an exact non-negative integer"
            )
        return revision

    @staticmethod
    def _validate_common_result(result: EnvironmentResult, *, surface: str) -> None:
        if result.ok is not True:
            raise CollectionProtocolError(f"{surface} result was not explicitly successful")
        if not isinstance(result.episode_id, str) or not result.episode_id.strip():
            raise CollectionProtocolError(f"{surface} result has no episode identity")
        if (
            isinstance(result.step_index, bool)
            or not isinstance(result.step_index, int)
            or result.step_index < 0
        ):
            raise CollectionProtocolError(
                f"{surface} result step_index must be an exact non-negative integer"
            )
        if not isinstance(result.terminated, bool) or not isinstance(result.truncated, bool):
            raise CollectionProtocolError(
                f"{surface} terminated/truncated flags must be booleans"
            )
        if result.terminated and result.truncated:
            raise CollectionProtocolError(
                f"{surface} result cannot be both terminated and truncated"
            )
        if result.terminal_reason is not None and not isinstance(
            result.terminal_reason, str
        ):
            raise CollectionProtocolError(
                f"{surface} terminal_reason must be a string or null"
            )
        if not isinstance(result.info, Mapping):
            raise CollectionProtocolError(f"{surface} info must be an object")

    def _validate_reset_result(
        self,
        result: EnvironmentResult,
        *,
        before_state_version: int,
        after_state_version: int,
    ) -> None:
        self._validate_common_result(result, surface="reset")
        if result.step_index != 0:
            raise CollectionProtocolError("fresh reset result must start at step_index=0")
        if result.terminated or result.truncated:
            raise CollectionProtocolError("fresh reset returned a terminal/truncated episode")
        if not result.legal_actions:
            raise CollectionProtocolError("fresh reset returned zero legal actions")
        if result.info.get("reward_authority") != "external-rl":
            raise CollectionProtocolError(
                "reset result must delegate reward authority to external-rl"
            )
        transition = result.transition
        if transition is None:
            raise CollectionProtocolError("reset result has no typed transition")
        if transition.episode_id != result.episode_id or transition.step_index != 0:
            raise CollectionProtocolError("reset transition identity does not match result")
        if (
            transition.before_state_version != before_state_version
            or transition.after_state_version != after_state_version
        ):
            raise CollectionProtocolError("reset transition revision identity is inconsistent")
        if not isinstance(transition.facts, Mapping):
            raise CollectionProtocolError("reset transition facts must be an object")

    def _validate_step_result(
        self,
        before: EnvironmentResult,
        after: EnvironmentResult,
    ) -> None:
        self._validate_common_result(after, surface="step")
        if after.episode_id != before.episode_id:
            raise CollectionProtocolError(
                "step result episode_id differs from the active episode"
            )
        if after.step_index != before.step_index + 1:
            raise CollectionProtocolError(
                "step result must advance step_index by exactly one"
            )
        transition = after.transition
        if transition is None:
            raise CollectionProtocolError("step result has no typed transition")
        if (
            transition.episode_id != after.episode_id
            or transition.step_index != after.step_index
        ):
            raise CollectionProtocolError(
                "step transition episode/step identity differs from result"
            )
        if not isinstance(transition.facts, Mapping):
            raise CollectionProtocolError("step transition facts must be an object")
        active_revision = self._active_state_version
        if active_revision is None:
            raise CollectionProtocolError("collector has no active state revision")
        before_revision = transition.before_state_version
        after_revision = transition.after_state_version
        if (
            isinstance(before_revision, bool)
            or not isinstance(before_revision, int)
            or isinstance(after_revision, bool)
            or not isinstance(after_revision, int)
            or before_revision < 0
            or after_revision < 0
        ):
            raise CollectionProtocolError(
                "step transition revisions must be exact non-negative integers"
            )
        if before_revision != active_revision or after_revision != before_revision + 1:
            raise CollectionProtocolError(
                "step transition revision chain is stale or non-contiguous"
            )
        if after.info.get("reward_authority") != "external-rl":
            raise CollectionProtocolError(
                "step result must delegate reward authority to external-rl"
            )
        if not after.terminated and not after.truncated and not after.legal_actions:
            raise CollectionProtocolError(
                "non-terminal step result returned zero legal actions"
            )
        if (after.terminated or after.truncated) and after.legal_actions:
            raise CollectionProtocolError(
                "terminal/truncated step result returned legal actions"
            )
        self._active_state_version = after_revision

    def reset(self, *, evaluation_seed: int | None = None) -> EnvironmentResult:
        request_id = str(uuid4())
        expected_state_version = self._state_version()
        if evaluation_seed is None:
            seed = self._episode_seed
        else:
            if isinstance(evaluation_seed, bool) or not isinstance(evaluation_seed, int):
                raise TypeError("evaluation seed must be an integer")
            if (
                evaluation_seed < 0
                or evaluation_seed > SIGNED_INT32_MAX
                or evaluation_seed % 2 != EVALUATION_SEED_PARITY
            ):
                raise ValueError("evaluation seed must be an odd signed 32-bit seed")
            seed = evaluation_seed
        if self.scenario == "combat":
            result = self.backend.combat_reset(
                CombatResetRequest(
                    request_id=request_id,
                    session_id=self.backend.session_id,
                    expected_state_version=expected_state_version,
                    character=self.character,
                    encounter_id=self.encounter_id,
                    seed=seed,
                    additional_relics=self.additional_relics or None,
                    training_revival_budget=self.training_revival_budget,
                )
            )
        else:
            result = self.backend.reset(
                ResetRequest(
                    request_id=request_id,
                    session_id=self.backend.session_id,
                    scenario="full-run",
                    expected_state_version=expected_state_version,
                    character=self.character,
                    seed=seed,
                    force_fresh=True,
                    additional_relics=self.additional_relics or None,
                    training_revival_budget=self.training_revival_budget,
                )
            )
        committed_state_version = self._state_version()
        if committed_state_version != expected_state_version + 1:
            raise CollectionProtocolError(
                "reset did not advance state_version by exactly one"
            )
        self._validate_reset_result(
            result,
            before_state_version=expected_state_version,
            after_state_version=committed_state_version,
        )
        self._active_state_version = committed_state_version
        if evaluation_seed is None:
            self._episode_seed += 2
        return result

    def _choose_action(
        self,
        state: EnvironmentResult,
        recurrent_state: torch.Tensor,
        *,
        epsilon: float,
        deterministic: bool,
    ) -> _ActionChoice:
        normalized_epsilon = float(epsilon)
        if not math.isfinite(normalized_epsilon) or not 0.0 <= normalized_epsilon <= 1.0:
            raise ValueError("exploration epsilon must be finite and in [0, 1]")
        encoding_started_ns = time.perf_counter_ns()
        encoded = self.encoder.encode(
            state.observation,
            state.legal_actions,
            device=self.device,
        )
        if self.device.type == "cuda":
            torch.cuda.synchronize(self.device)
        encoding_ms = (time.perf_counter_ns() - encoding_started_ns) / 1_000_000.0
        policy_started_ns = time.perf_counter_ns()
        was_training = self.model.training
        self.model.eval()
        try:
            with torch.no_grad():
                output = self.model(
                    encoded.batch,
                    recurrent_state,
                    validate=False,
                )
                policy = output.policy_probabilities()[0].float().cpu().numpy()
                valid = output.action_mask[0].cpu().numpy().astype(bool)
                next_recurrent_state = output.recurrent_state.detach()
                value = float(output.value[0].item())
        finally:
            self.model.train(was_training)
        policy_forward_ms = (time.perf_counter_ns() - policy_started_ns) / 1_000_000.0
        valid_indices = np.flatnonzero(valid)
        if valid_indices.size == 0:
            raise CollectionProtocolError(
                f"episode={state.episode_id!r} step={state.step_index} has no enabled legal action"
            )
        valid_count = int(valid_indices.size)
        valid_policy = policy[valid_indices]
        if not np.all(np.isfinite(valid_policy)) or np.any(valid_policy < 0.0):
            raise CollectionProtocolError("model produced a non-finite policy distribution")
        policy_mass = float(valid_policy.sum())
        if not math.isfinite(policy_mass) or policy_mass <= 0.0:
            raise CollectionProtocolError("model produced zero/non-finite legal policy mass")
        if deterministic:
            selected = int(valid_indices[int(np.argmax(valid_policy))])
            return _ActionChoice(
                action_index=selected,
                behavior_log_probability=float(
                    math.log(max(float(valid_policy.max() / policy_mass), 1e-30))
                ),
                valid_count=valid_count,
                snapshot=encoded.snapshot,
                recurrent_state=next_recurrent_state,
                policy=policy,
                value=value,
                encoding_ms=encoding_ms,
                policy_forward_ms=policy_forward_ms,
            )

        behavior = np.zeros_like(policy, dtype=np.float64)
        behavior[valid_indices] = (
            (1.0 - normalized_epsilon) * valid_policy / policy_mass
            + normalized_epsilon / valid_count
        )
        behavior /= behavior.sum()
        if not np.all(np.isfinite(behavior)) or np.any(behavior < 0.0):
            raise CollectionProtocolError("collector produced an invalid behavior policy")
        selected = int(self._rng.choice(len(behavior), p=behavior))
        return _ActionChoice(
            action_index=selected,
            behavior_log_probability=float(
                math.log(max(float(behavior[selected]), 1e-30))
            ),
            valid_count=valid_count,
            snapshot=encoded.snapshot,
            recurrent_state=next_recurrent_state,
            policy=policy,
            value=value,
            encoding_ms=encoding_ms,
            policy_forward_ms=policy_forward_ms,
        )

    def _step(
        self,
        state: EnvironmentResult,
        *,
        action_index: int,
    ) -> tuple[EnvironmentResult, str]:
        action = state.legal_actions[action_index]
        handle_value = action.get("action_handle", action.get("action_id"))
        handle = str(handle_value) if handle_value is not None and str(handle_value) else ""
        request = StepRequest(
            request_id=str(uuid4()),
            session_id=self.backend.session_id,
            episode_id=state.episode_id,
            expected_step_index=state.step_index,
            action_id=handle or None,
            action_index=None if handle else action_index,
        )
        result = self.backend.step(request)
        self._validate_step_result(state, result)
        return result, handle or f"index:{action_index}"

    def collect_episode(
        self,
        *,
        epsilon: float = 0.0,
        deterministic: bool = False,
        record: bool = True,
        evaluation_seed: int | None = None,
        policy_version: int = 0,
        trajectory_journal: TrajectoryJournal | None = None,
        maximum_steps: int | None = None,
        unroll_sink: Callable[[SequenceUnroll], None] | None = None,
    ) -> CollectedEpisode:
        if isinstance(policy_version, bool) or not isinstance(policy_version, int):
            raise TypeError("policy_version must be an integer")
        if policy_version < 0:
            raise ValueError("policy_version must be non-negative")
        episode_limit = self.max_episode_steps
        if maximum_steps is not None:
            if isinstance(maximum_steps, bool) or not isinstance(maximum_steps, int):
                raise TypeError("maximum_steps must be an integer or null")
            if maximum_steps <= 0:
                raise ValueError("maximum_steps must be positive")
            episode_limit = min(episode_limit, maximum_steps)
        timings = _CollectorTimingAccumulator()
        reset_seed = self._episode_seed if evaluation_seed is None else evaluation_seed
        reset_started_ns = time.perf_counter_ns()
        state = self.reset(evaluation_seed=evaluation_seed)
        timings.record("reset", reset_started_ns)
        self.deadlock_detector.reset()
        recurrent_state = self.model.initial_state(1, device=self.device)
        segment_initial_state = (
            recurrent_state[0].detach().float().cpu().numpy().copy()
        )
        segment_start_step = state.step_index
        segment_steps: list[RolloutStep] = []
        unrolls: list[SequenceUnroll] = []
        reward_total = 0.0
        max_act, max_floor = _run_position(state.observation)
        policy_decisions = 0
        forced_decisions = 0
        steps_taken = 0
        final_outcome = "ongoing"
        deadlocked = False
        forced_horizon = False
        curriculum_horizon = False
        revivals_used = 0
        player_hp_lost = 0.0

        for step_offset in range(episode_limit):
            if state.terminated or state.truncated:
                break
            if not state.legal_actions:
                raise CollectionProtocolError(
                    f"episode={state.episode_id!r} step={state.step_index} returned zero legal actions"
                )
            choice = self._choose_action(
                state,
                recurrent_state,
                epsilon=epsilon,
                deterministic=deterministic,
            )
            timings.add("observation_encoding", choice.encoding_ms)
            timings.add("policy_forward", choice.policy_forward_ms)
            policy_decisions += int(choice.valid_count > 1)
            forced_decisions += int(choice.valid_count == 1)
            selected_action = state.legal_actions[choice.action_index]
            deadlock_evidence = self.deadlock_detector.observe(
                step_index=state.step_index,
                observation=state.observation,
                legal_actions=state.legal_actions,
                selected_action=selected_action,
            )
            sim_step_started_ns = time.perf_counter_ns()
            next_state, _ = self._step(state, action_index=choice.action_index)
            timings.record("sim_step", sim_step_started_ns)
            steps_taken += 1
            if next_state.truncated:
                raise CollectionProtocolError(
                    "transport/outcome-unknown truncation discarded before rollout"
                )
            forced_horizon = (
                step_offset + 1 >= episode_limit
                and not next_state.terminated
                and not next_state.truncated
            )
            # ``maximum_steps`` may be a runtime's remaining global budget,
            # which can cut an otherwise healthy episode after only one or a
            # few decisions. Only the configured task horizon is a semantic
            # failure. A shorter collection-budget cut keeps a positive
            # discount and a bootstrap snapshot instead of fabricating a loss.
            curriculum_horizon = bool(
                forced_horizon and episode_limit >= self.max_episode_steps
            )
            reward_started_ns = time.perf_counter_ns()
            breakdown = self.reward_calculator.evaluate(
                state,
                next_state,
                deadlock=deadlock_evidence is not None,
                horizon_exhausted=bool(
                    curriculum_horizon and self.horizon_as_failure
                ),
            )
            reward_total += breakdown.reward
            revivals_used += breakdown.revivals_used_delta
            player_hp_lost += breakdown.player_hp_lost_delta
            final_outcome = breakdown.outcome
            deadlocked = breakdown.outcome == "deadlock"
            if record:
                segment_steps.append(
                    RolloutStep(
                        snapshot=choice.snapshot,
                        action_index=choice.action_index,
                        behavior_log_probability=choice.behavior_log_probability,
                        reward=breakdown.reward,
                        discount=breakdown.discount,
                        policy_decision=choice.valid_count > 1,
                    )
                )
            if trajectory_journal is not None:
                valid_indices = np.flatnonzero(choice.snapshot.action_mask)
                ranked = sorted(
                    valid_indices.tolist(),
                    key=lambda index: float(choice.policy[index]),
                    reverse=True,
                )[: self.journal_policy_topk]
                trajectory_journal.write(
                    {
                        "event": "decision",
                        "episode_id": state.episode_id,
                        "reset_seed": reset_seed,
                        "step_index": state.step_index,
                        "observation": semantic_projection(state.observation),
                        "legal_actions": semantic_projection(state.legal_actions),
                        "selected_index": choice.action_index,
                        "selected_action": semantic_projection(selected_action),
                        "policy_topk": [
                            {
                                "index": index,
                                "probability": float(choice.policy[index]),
                            }
                            for index in ranked
                        ],
                        "value": choice.value,
                        "reward": breakdown.reward,
                        "terminal_reward": breakdown.terminal_reward,
                        "potential_reward": breakdown.potential_reward,
                        "revival_penalty": breakdown.revival_penalty,
                        "pace_penalty": breakdown.pace_penalty,
                        "hp_loss_penalty": breakdown.hp_loss_penalty,
                        "player_hp_lost": player_hp_lost,
                        "revivals_used": revivals_used,
                        "outcome": breakdown.outcome,
                        "deadlock": (
                            deadlock_evidence.to_mapping()
                            if deadlock_evidence is not None
                            else None
                        ),
                    }
                )
            timings.record("reward_and_diagnostics", reward_started_ns)
            recurrent_state = choice.recurrent_state
            state = next_state
            act, floor = _run_position(state.observation)
            max_act = max(max_act, act)
            max_floor = max(max_floor, floor)

            flush_segment = bool(
                record
                and segment_steps
                and (
                    len(segment_steps) >= self.unroll_length
                    or breakdown.task_terminal
                    or forced_horizon
                )
            )
            if flush_segment:
                bootstrap_snapshot: EncodedDecisionSnapshot | None = None
                if segment_steps[-1].discount > 0.0:
                    bootstrap_started_ns = time.perf_counter_ns()
                    bootstrap_snapshot = self.encoder.encode(
                        state.observation,
                        state.legal_actions,
                        device="cpu",
                    ).snapshot
                    timings.record("bootstrap_encoding", bootstrap_started_ns)
                completed_unroll = SequenceUnroll(
                    episode_id=state.episode_id,
                    start_step=segment_start_step,
                    policy_version=policy_version,
                    initial_recurrent_state=segment_initial_state,
                    steps=tuple(segment_steps),
                    bootstrap_snapshot=bootstrap_snapshot,
                )
                if unroll_sink is None:
                    unrolls.append(completed_unroll)
                else:
                    unroll_sink(completed_unroll)
                segment_steps = []
                segment_initial_state = (
                    recurrent_state[0].detach().float().cpu().numpy().copy()
                )
                segment_start_step = state.step_index

            if state.terminated or breakdown.task_terminal or forced_horizon:
                break

        if segment_steps:
            raise RuntimeError("collector exited with an unflushed rollout segment")
        run_won = self.objective == "run" and final_outcome == "success"
        combat_won = self.objective == "combat" and final_outcome == "success"
        return CollectedEpisode(
            unrolls=tuple(unrolls),
            metrics=EpisodeMetrics(
                episode_id=state.episode_id,
                reset_seed=reset_seed,
                steps=steps_taken,
                reward_total=reward_total,
                terminal_reason=(
                    "semantic_deadlock"
                    if deadlocked
                    else "curriculum_horizon"
                    if curriculum_horizon and self.horizon_as_failure
                    else "collection_budget"
                    if forced_horizon
                    else state.terminal_reason
                ),
                truncated=forced_horizon,
                run_won=run_won,
                combat_won=combat_won,
                act1_cleared=bool(max_act >= 2),
                max_act=max_act,
                max_floor=max_floor,
                policy_decisions=policy_decisions,
                forced_decisions=forced_decisions,
                deadlocked=deadlocked,
                revivals_used=revivals_used,
                revival_free_combat_win=bool(combat_won and revivals_used == 0),
                revival_free_act1_clear=bool(max_act >= 2 and revivals_used == 0),
                revival_free_run_win=bool(run_won and revivals_used == 0),
                player_hp_lost=player_hp_lost,
            ),
            timings=timings.snapshot(),
        )


__all__ = [
    "CollectedEpisode",
    "CollectionProtocolError",
    "EpisodeMetrics",
    "GroundedCollector",
]
