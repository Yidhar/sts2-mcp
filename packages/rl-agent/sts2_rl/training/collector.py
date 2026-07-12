"""Direct typed-backend collector for the grounded baseline."""

from __future__ import annotations

import math
from collections.abc import Mapping
from copy import deepcopy
from dataclasses import dataclass
from typing import Literal
from uuid import uuid4

import numpy as np
import torch

from sts2_baseline import (
    BaselineRewardCalculator,
    BaselineTargets,
    BaselineTransition,
    ReplaySample,
    ReplayStratum,
)
from sts2_rl.contracts import (
    CombatResetRequest,
    EnvironmentBackend,
    EnvironmentResult,
    ResetRequest,
    StepRequest,
)
from sts2_rl.encoding import GroundedObservationEncoder
from sts2_rl.models import GroundedCandidateModel

from .experience import DecisionExperience, baseline_transition, compact_decision, replay_stratum
from .seeding import (
    EVALUATION_SEED_PARITY,
    SIGNED_INT32_MAX,
    training_seed_start,
)


class CollectionProtocolError(RuntimeError):
    """The environment exposed no dispatchable legal candidate."""


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


@dataclass(frozen=True, slots=True)
class CollectedEpisode:
    samples: tuple[ReplaySample, ...]
    metrics: EpisodeMetrics


@dataclass(frozen=True, slots=True)
class _Pending:
    transition: BaselineTransition
    reward: float
    experience: DecisionExperience
    stratum: ReplayStratum
    objective_terminal: bool


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
        model: GroundedCandidateModel,
        encoder: GroundedObservationEncoder,
        backend: EnvironmentBackend,
        scenario: Literal["full-run", "combat"],
        objective: Literal["combat", "run"],
        discount: float,
        max_episode_steps: int,
        character: str | None = None,
        encounter_id: str | None = None,
        seed: int = 0,
    ) -> None:
        if scenario not in {"full-run", "combat"}:
            raise ValueError("scenario must be full-run or combat")
        if objective not in {"combat", "run"}:
            raise ValueError("objective must be combat or run")
        if (scenario == "combat") != (objective == "combat"):
            raise ValueError("combat scenario and combat objective must be selected together")
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
        self.model = model
        self.encoder = encoder
        self.backend = backend
        self.scenario = scenario
        self.objective = objective
        self.discount = float(discount)
        self.max_episode_steps = int(max_episode_steps)
        self.character = character
        self.encounter_id = encounter_id
        self._rng = np.random.default_rng(int(seed))
        self._episode_seed = training_seed_start(int(seed))
        self._active_state_version: int | None = None
        self.reward_calculator = BaselineRewardCalculator(objective)

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
            "version": "sts2-grounded-collector-state-v2",
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
        if payload["version"] != "sts2-grounded-collector-state-v2":
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
        *,
        epsilon: float,
        deterministic: bool,
    ) -> tuple[int, float, int]:
        normalized_epsilon = float(epsilon)
        if not math.isfinite(normalized_epsilon) or not 0.0 <= normalized_epsilon <= 1.0:
            raise ValueError("exploration epsilon must be finite and in [0, 1]")
        encoded = self.encoder.encode(
            state.observation,
            state.legal_actions,
            device=self.device,
        )
        was_training = self.model.training
        self.model.eval()
        try:
            with torch.no_grad():
                output = self.model(encoded.batch, validate=False)
                policy = output.policy_probabilities()[0].float().cpu().numpy()
                valid = output.action_mask[0].cpu().numpy().astype(bool)
        finally:
            self.model.train(was_training)
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
            return selected, 0.0, valid_count

        behavior = np.zeros_like(policy, dtype=np.float64)
        behavior[valid_indices] = (
            (1.0 - normalized_epsilon) * valid_policy / policy_mass
            + normalized_epsilon / valid_count
        )
        behavior /= behavior.sum()
        if not np.all(np.isfinite(behavior)) or np.any(behavior < 0.0):
            raise CollectionProtocolError("collector produced an invalid behavior policy")
        selected = int(self._rng.choice(len(behavior), p=behavior))
        return selected, float(math.log(max(float(behavior[selected]), 1e-30))), valid_count

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
    ) -> CollectedEpisode:
        reset_seed = self._episode_seed if evaluation_seed is None else evaluation_seed
        state = self.reset(evaluation_seed=evaluation_seed)
        pending: list[_Pending] = []
        reward_total = 0.0
        max_act, max_floor = _run_position(state.observation)
        policy_decisions = 0
        forced_decisions = 0
        final_transition: BaselineTransition | None = None
        steps_taken = 0

        for step_offset in range(self.max_episode_steps):
            if state.terminated or state.truncated:
                break
            if not state.legal_actions:
                raise CollectionProtocolError(
                    f"episode={state.episode_id!r} step={state.step_index} returned zero legal actions"
                )
            action_index, behavior_log_probability, valid_count = self._choose_action(
                state,
                epsilon=epsilon,
                deterministic=deterministic,
            )
            policy_decisions += int(valid_count > 1)
            forced_decisions += int(valid_count == 1)
            next_state, action_handle = self._step(state, action_index=action_index)
            steps_taken += 1
            if next_state.truncated:
                raise CollectionProtocolError(
                    "transport/outcome-unknown truncation discarded before replay"
                )
            forced_truncation = (
                step_offset + 1 >= self.max_episode_steps
                and not next_state.terminated
                and not next_state.truncated
            )
            transition = baseline_transition(
                before=state,
                after=next_state,
                action_handle=action_handle,
                objective=self.objective,
                forced_truncation=forced_truncation,
            )
            breakdown = self.reward_calculator.evaluate(transition)
            reward_total += breakdown.total
            task_terminal = (
                transition.combat_result != "none"
                if self.objective == "combat"
                else transition.run_result != "none"
            )
            terminal_class = 1 if task_terminal else int(
                bool(next_state.info.get("chance_boundary", False))
            ) * 2
            if record:
                experience = compact_decision(
                    state.observation,
                    state.legal_actions,
                    action_index=action_index,
                    behavior_log_probability=behavior_log_probability,
                    terminal_class=terminal_class,
                    objective=self.objective,
                )
                pending.append(
                    _Pending(
                        transition=transition,
                        reward=breakdown.total,
                        experience=experience,
                        stratum=replay_stratum(state.observation, transition),
                        objective_terminal=task_terminal,
                    )
                )
            final_transition = transition
            state = next_state
            act, floor = _run_position(state.observation)
            max_act = max(max_act, act)
            max_floor = max(max_floor, floor)
            if state.terminated or task_terminal or forced_truncation:
                break

        samples: list[ReplaySample] = []
        running_return = 0.0
        for item in reversed(pending):
            if item.objective_terminal:
                running_return = 0.0
            running_return = float(item.reward) + self.discount * running_return
            samples.append(
                ReplaySample(
                    transition=item.transition,
                    targets=BaselineTargets(reward=float(item.reward), value=running_return),
                    stratum=item.stratum,
                    payload=item.experience,
                )
            )
        samples.reverse()

        run_won = bool(final_transition is not None and final_transition.run_result == "win")
        combat_won = bool(
            final_transition is not None and final_transition.combat_result == "win"
        )
        truncated = bool(final_transition is not None and final_transition.truncated)
        return CollectedEpisode(
            samples=tuple(samples),
            metrics=EpisodeMetrics(
                episode_id=state.episode_id,
                reset_seed=reset_seed,
                steps=steps_taken,
                reward_total=reward_total,
                terminal_reason=state.terminal_reason,
                truncated=truncated,
                run_won=run_won,
                combat_won=combat_won,
                act1_cleared=bool(run_won or max_act >= 2),
                max_act=max_act,
                max_floor=max_floor,
                policy_decisions=policy_decisions,
                forced_decisions=forced_decisions,
            ),
        )


__all__ = [
    "CollectedEpisode",
    "CollectionProtocolError",
    "EpisodeMetrics",
    "GroundedCollector",
]
