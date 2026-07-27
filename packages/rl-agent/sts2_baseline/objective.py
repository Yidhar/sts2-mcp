"""Auditable outcome, survival-efficiency, and run-distance rewards.

The v3 objective deliberately contains no reward for damage dealt, enemy HP
change, cards played, or any hand-written action preference.  Engine-bailout
preheat runs on either a combat or full-run horizon and learns from the task
outcome, monotonic run progress, exact player-HP loss, and exact training-revival
counters. Simulator-only counters remain underscore-prefixed observation facts
and are never model features.
"""

from __future__ import annotations

import hashlib
import json
import math
from collections.abc import Mapping
from dataclasses import asdict, dataclass, field
from typing import Any, Final, Literal

from sts2_rl.contracts import EnvironmentResult

TaskObjective = Literal["combat", "act1", "run"]
TaskOutcome = Literal["ongoing", "success", "failure", "deadlock", "horizon"]


@dataclass(frozen=True, slots=True)
class TaskRewardSpec:
    """Normal-task reward: terminal outcome plus bounded forward distance."""

    version: str = field(default="sts2-task-reward-v4", init=False)
    discount: float = field(default=0.997, init=False)
    success_reward: float = field(default=1.0, init=False)
    failure_reward: float = field(default=-1.0, init=False)
    deadlock_reward: float = field(default=-1.0, init=False)
    horizon_reward: float = field(default=-1.0, init=False)
    run_progress_reward_weight: float = field(default=0.50, init=False)
    act1_success_act: int = field(default=2, init=False)
    fallback_run_floor_cap: float = field(default=60.0, init=False)


TASK_REWARD_SPEC: Final = TaskRewardSpec()


@dataclass(frozen=True, slots=True)
class RevivalEfficiencyRewardSpec:
    """Bounded, undiscounted training-revival preference.

    Each efficiency term is a delta of a monotonic bounded score.  Across a
    complete episode their combined magnitude is strictly below 1.0, so
    every victory remains better than every failure.  Within the same outcome,
    the weighted survival objective jointly prefers less cumulative HP loss,
    fewer training revivals, and fewer decisions; it does not reward damage.
    """

    version: str = field(default="sts2-run-survival-efficiency-v4", init=False)
    required_discount: float = field(default=1.0, init=False)
    hp_loss_weight: float = field(default=0.55, init=False)
    revival_weight: float = field(default=0.20, init=False)
    pace_budget: float = field(default=0.05, init=False)
    hp_loss_scale: float = field(default=80.0, init=False)


REVIVAL_EFFICIENCY_REWARD_SPEC: Final = RevivalEfficiencyRewardSpec()


def task_reward_identity() -> dict[str, Any]:
    payload: dict[str, Any] = {
        "version": TASK_REWARD_SPEC.version,
        "spec": asdict(TASK_REWARD_SPEC),
        "outcome_source": (
            "environment_result.terminated+typed_transition.facts.combat_result+"
            "typed_transition.facts.run_result+typed_transition.facts.terminal_reason+"
            "observation.run.act"
        ),
        "progress_source": "positive_delta(observation.run.progress_or_floor)",
        "forbidden_shaping": "enemy_hp_delta+damage_dealt+cards_played",
        "terminal_reason_policy": "transition/result exact equality+objective-scoped canonical result",
        "transport_truncation": "reject",
        "collector_horizon": "explicit_horizon_outcome_when_configured",
    }
    serialized = json.dumps(payload, sort_keys=True, separators=(",", ":"))
    payload["fingerprint"] = serialized
    payload["fingerprint_sha256"] = hashlib.sha256(
        serialized.encode("utf-8")
    ).hexdigest()
    return payload


def revival_efficiency_reward_identity() -> dict[str, Any]:
    payload: dict[str, Any] = {
        "version": REVIVAL_EFFICIENCY_REWARD_SPEC.version,
        "base": task_reward_identity(),
        "spec": asdict(REVIVAL_EFFICIENCY_REWARD_SPEC),
        "revival_event_source": "observation._training.revivals_used exact counter",
        "hp_loss_source": "observation._training.player_hp_lost exact counter",
        "ordering": "task_outcome+run_progress>bounded_survival_efficiency>decisions",
        "forbidden_shaping": "enemy_hp_delta+damage_dealt+cards_played",
    }
    serialized = json.dumps(payload, sort_keys=True, separators=(",", ":"))
    payload["fingerprint"] = serialized
    payload["fingerprint_sha256"] = hashlib.sha256(
        serialized.encode("utf-8")
    ).hexdigest()
    return payload


def _mapping(value: Any) -> Mapping[str, Any]:
    return value if isinstance(value, Mapping) else {}


def _number(value: Any, *, default: float = 0.0) -> float:
    if isinstance(value, bool) or not isinstance(value, str | int | float):
        return default
    try:
        result = float(value)
    except (TypeError, ValueError):
        return default
    return result if math.isfinite(result) else default


def _run_position(observation: Mapping[str, Any]) -> tuple[int, int, float]:
    run = _mapping(observation.get("run"))
    act = int(max(0.0, _number(run.get("act", observation.get("act")))))
    floor = int(max(0.0, _number(run.get("floor", observation.get("floor")))))
    raw_progress = run.get("progress", observation.get("run_progress"))
    if raw_progress is None:
        progress = min(1.0, floor / TASK_REWARD_SPEC.fallback_run_floor_cap)
    else:
        progress = min(max(_number(raw_progress), 0.0), 1.0)
    return act, floor, progress


def _training_counter(observation: Mapping[str, Any], key: str) -> float:
    training = _mapping(observation.get("_training"))
    return max(0.0, _number(training.get(key)))


def _bounded_resource_score(amount: float, scale: float) -> float:
    normalized_amount = max(0.0, float(amount))
    normalized_scale = max(1.0, float(scale))
    return normalized_amount / (normalized_amount + normalized_scale)


def _typed_terminal_result(result: EnvironmentResult, *, objective: TaskObjective) -> str:
    transition = result.transition
    if transition is None or not isinstance(transition.facts, Mapping):
        if result.terminated:
            raise ValueError("terminal task result has no typed transition facts")
        return "none"
    fact_name = "combat_result" if objective == "combat" else "run_result"
    supported = (
        {"none", "victory", "defeat", "escaped"}
        if objective == "combat"
        else {"none", "victory", "defeat"}
    )
    raw = transition.facts.get(fact_name)
    if not isinstance(raw, str) or raw not in supported or (result.terminated and raw == "none"):
        raise ValueError(f"typed {fact_name} is missing or unsupported")
    if "terminal_reason" not in transition.facts:
        raise ValueError("typed terminal_reason is missing")
    fact_reason = transition.facts["terminal_reason"]
    if fact_reason is not None and not isinstance(fact_reason, str):
        raise ValueError("typed terminal_reason must be text or null")
    if fact_reason != result.terminal_reason:
        raise ValueError("transition and result terminal_reason identities differ")
    expected_reason = f"{'combat' if objective == 'combat' else 'run'}_{raw}"
    if fact_reason != expected_reason:
        raise ValueError(f"typed {fact_name} disagrees with terminal_reason")
    return raw


@dataclass(frozen=True, slots=True)
class TaskReward:
    reward: float
    discount: float
    terminal_reward: float
    potential_reward: float
    task_terminal: bool
    outcome: TaskOutcome
    revival_penalty: float = 0.0
    pace_penalty: float = 0.0
    hp_loss_penalty: float = 0.0
    progress_reward: float = 0.0
    revivals_used_delta: int = 0
    player_hp_lost_delta: float = 0.0


class TaskRewardCalculator:
    def __init__(self, objective: TaskObjective, *, discount: float = 0.997) -> None:
        if objective not in {"combat", "act1", "run"}:
            raise ValueError("objective must be combat, act1, or run")
        normalized_discount = float(discount)
        if not math.isfinite(normalized_discount) or not 0.0 < normalized_discount <= 1.0:
            raise ValueError("discount must be finite and in (0, 1]")
        self.objective = objective
        self.discount = normalized_discount

    def _outcome(
        self,
        after: EnvironmentResult,
        *,
        deadlock: bool,
        horizon_exhausted: bool,
    ) -> TaskOutcome:
        if deadlock:
            return "deadlock"
        if horizon_exhausted:
            return "horizon"
        act, _, _ = _run_position(after.observation)
        if self.objective == "act1" and act >= TASK_REWARD_SPEC.act1_success_act:
            return "success"
        if not after.terminated:
            return "ongoing"
        if self.objective == "act1":
            return "failure"
        terminal_result = _typed_terminal_result(after, objective=self.objective)
        return "success" if terminal_result == "victory" else "failure"

    def evaluate(
        self,
        before: EnvironmentResult,
        after: EnvironmentResult,
        *,
        deadlock: bool = False,
        horizon_exhausted: bool = False,
    ) -> TaskReward:
        if after.truncated:
            raise ValueError(
                "transport/outcome-unknown truncation cannot become training data"
            )
        outcome = self._outcome(
            after,
            deadlock=deadlock,
            horizon_exhausted=horizon_exhausted,
        )
        task_terminal = outcome != "ongoing"
        terminal_reward = {
            "ongoing": 0.0,
            "success": TASK_REWARD_SPEC.success_reward,
            "failure": TASK_REWARD_SPEC.failure_reward,
            "deadlock": TASK_REWARD_SPEC.deadlock_reward,
            "horizon": TASK_REWARD_SPEC.horizon_reward,
        }[outcome]
        progress_reward = 0.0
        if self.objective in {"act1", "run"}:
            before_progress = _run_position(before.observation)[2]
            after_progress = _run_position(after.observation)[2]
            progress_reward = TASK_REWARD_SPEC.run_progress_reward_weight * max(
                0.0, after_progress - before_progress
            )
        return TaskReward(
            reward=float(terminal_reward + progress_reward),
            discount=0.0 if task_terminal else self.discount,
            terminal_reward=float(terminal_reward),
            potential_reward=float(progress_reward),
            task_terminal=task_terminal,
            outcome=outcome,
            progress_reward=float(progress_reward),
        )


class RevivalEfficiencyRewardCalculator:
    """Revival preheat reward from task progress and exact run counters."""

    def __init__(
        self,
        *,
        maximum_episode_steps: int,
        objective: TaskObjective = "combat",
        discount: float = 1.0,
    ) -> None:
        if objective not in {"combat", "act1", "run"}:
            raise ValueError("objective must be combat, act1, or run")
        if isinstance(maximum_episode_steps, bool) or maximum_episode_steps <= 0:
            raise ValueError("maximum_episode_steps must be a positive integer")
        if float(discount) != REVIVAL_EFFICIENCY_REWARD_SPEC.required_discount:
            raise ValueError("survival preheat requires an undiscounted return (discount=1)")
        self.maximum_episode_steps = int(maximum_episode_steps)
        self.base = TaskRewardCalculator(objective, discount=discount)

    def evaluate(
        self,
        before: EnvironmentResult,
        after: EnvironmentResult,
        *,
        deadlock: bool = False,
        horizon_exhausted: bool = False,
    ) -> TaskReward:
        base = self.base.evaluate(
            before,
            after,
            deadlock=deadlock,
            horizon_exhausted=horizon_exhausted,
        )
        before_revivals = _training_counter(before.observation, "revivals_used")
        after_revivals = _training_counter(after.observation, "revivals_used")
        before_hp_lost = _training_counter(before.observation, "player_hp_lost")
        after_hp_lost = _training_counter(after.observation, "player_hp_lost")
        if after_revivals < before_revivals or after_hp_lost < before_hp_lost:
            raise ValueError("training efficiency counters must be monotonic")

        revivals_used_delta = int(after_revivals - before_revivals)
        player_hp_lost_delta = after_hp_lost - before_hp_lost
        hp_loss_penalty = -REVIVAL_EFFICIENCY_REWARD_SPEC.hp_loss_weight * (
            _bounded_resource_score(
                after_hp_lost,
                REVIVAL_EFFICIENCY_REWARD_SPEC.hp_loss_scale,
            )
            - _bounded_resource_score(
                before_hp_lost,
                REVIVAL_EFFICIENCY_REWARD_SPEC.hp_loss_scale,
            )
        )
        revival_penalty = -REVIVAL_EFFICIENCY_REWARD_SPEC.revival_weight * (
            _bounded_resource_score(after_revivals, 1.0)
            - _bounded_resource_score(before_revivals, 1.0)
        )
        pace_penalty = -REVIVAL_EFFICIENCY_REWARD_SPEC.pace_budget / float(
            self.maximum_episode_steps
        )
        return TaskReward(
            reward=float(
                base.reward + hp_loss_penalty + revival_penalty + pace_penalty
            ),
            discount=base.discount,
            terminal_reward=base.terminal_reward,
            potential_reward=base.potential_reward,
            task_terminal=base.task_terminal,
            outcome=base.outcome,
            revival_penalty=float(revival_penalty),
            pace_penalty=float(pace_penalty),
            hp_loss_penalty=float(hp_loss_penalty),
            progress_reward=base.progress_reward,
            revivals_used_delta=revivals_used_delta,
            player_hp_lost_delta=float(player_hp_lost_delta),
        )


__all__ = [
    "REVIVAL_EFFICIENCY_REWARD_SPEC",
    "TASK_REWARD_SPEC",
    "RevivalEfficiencyRewardCalculator",
    "RevivalEfficiencyRewardSpec",
    "TaskObjective",
    "TaskOutcome",
    "TaskReward",
    "TaskRewardCalculator",
    "TaskRewardSpec",
    "revival_efficiency_reward_identity",
    "task_reward_identity",
]
