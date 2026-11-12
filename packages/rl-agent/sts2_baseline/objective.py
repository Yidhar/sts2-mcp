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
class AbsoluteDeathPriceTaskRewardSpec:
    """Outcome scale for the absolute-death-price hybrid course.

    This is intentionally a separate reward identity rather than a mutation of
    the archived v4/v7 lineages.  The larger terminal margin is verified in
    durable-floor discounted Q space against every course's maximum secondary
    cost span.
    """

    version: str = field(default="sts2-task-reward-v5", init=False)
    discount: float = field(default=0.997, init=False)
    success_reward: float = field(default=2.0, init=False)
    failure_reward: float = field(default=-2.0, init=False)
    deadlock_reward: float = field(default=-2.0, init=False)
    horizon_reward: float = field(default=-2.0, init=False)
    run_progress_reward_weight: float = field(default=0.50, init=False)
    act1_success_act: int = field(default=2, init=False)
    fallback_run_floor_cap: float = field(default=60.0, init=False)


ABSOLUTE_DEATH_PRICE_TASK_REWARD_SPEC: Final = AbsoluteDeathPriceTaskRewardSpec()


@dataclass(frozen=True, slots=True)
class RevivalEfficiencyRewardSpec:
    """Bounded, undiscounted training-revival preference.

    Each efficiency term is a delta of a monotonic bounded score.  Across a
    complete episode their combined magnitude is strictly below 1.0, so
    every victory remains better than every failure.  Within the same outcome,
    the weighted survival objective jointly prefers less cumulative HP loss,
    fewer training revivals, and fewer decisions; it does not reward damage.
    """

    version: str = field(default="sts2-run-survival-efficiency-v7", init=False)
    required_discount: float = field(default=1.0, init=False)
    hp_loss_weight: float = field(default=0.55, init=False)
    # V6 reached this cap at revival 17, so every revival from 18 through a
    # later 32/64-revival course had exactly zero scalar marginal cost.  V7
    # spends the same bounded cost budget uniformly across the *configured*
    # finite course.  Thus every permitted revival remains distinguishable,
    # while tightening the curriculum automatically raises the unit price.
    revival_cost_cap: float = field(default=0.40, init=False)
    revival_reference_budget: int = field(default=64, init=False)
    pace_budget: float = field(default=0.04, init=False)
    hp_loss_scale: float = field(default=80.0, init=False)


REVIVAL_EFFICIENCY_REWARD_SPEC: Final = RevivalEfficiencyRewardSpec()


@dataclass(frozen=True, slots=True)
class AbsoluteDeathPriceRewardSpec:
    """Stable factual reward carried by hybrid transition v2.

    Revival preference is deliberately absent here.  The transition exports
    the exact counter delta and the Candidate-Q learner applies one absolute
    price to every sample in an optimizer update.
    """

    version: str = field(default="sts2-run-survival-efficiency-v8", init=False)
    required_discount: float = field(default=1.0, init=False)
    hp_loss_weight: float = field(default=0.55, init=False)
    pace_budget: float = field(default=0.04, init=False)
    hp_loss_scale: float = field(default=80.0, init=False)


ABSOLUTE_DEATH_PRICE_REWARD_SPEC: Final = AbsoluteDeathPriceRewardSpec()


@dataclass(frozen=True, slots=True)
class ActFrontloadedDeathRewardSpec:
    """Reward-v9 death cost used by the recurrent hybrid policy.

    The first deaths in *each* Act are deliberately expensive, matching the
    useful pressure of reward-v4, while a fixed tail price keeps every later
    revival distinguishable.  ``act_frontload_cap * k / (k + 1)`` is applied
    independently to each Act and ``tail_budget / revival_budget`` is charged
    for every revival in the run.

    The calculator owns no adaptive price or training feedback loop.  A
    revival-budget change is a course boundary; within one course the reward
    is stationary and can be learned by an ordinary recurrent actor-critic.
    """

    version: str = field(default="sts2-run-survival-efficiency-v9", init=False)
    required_discount: float = field(default=1.0, init=False)
    hp_loss_weight: float = field(default=0.55, init=False)
    pace_budget: float = field(default=0.04, init=False)
    hp_loss_scale: float = field(default=80.0, init=False)
    act_frontload_cap: float = field(default=0.20, init=False)
    tail_budget: float = field(default=0.80, init=False)


ACT_FRONTLOADED_DEATH_REWARD_SPEC: Final = ActFrontloadedDeathRewardSpec()


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
    payload["fingerprint_sha256"] = hashlib.sha256(serialized.encode("utf-8")).hexdigest()
    return payload


def revival_efficiency_reward_identity() -> dict[str, Any]:
    payload: dict[str, Any] = {
        "version": REVIVAL_EFFICIENCY_REWARD_SPEC.version,
        "base": task_reward_identity(),
        "spec": asdict(REVIVAL_EFFICIENCY_REWARD_SPEC),
        "revival_event_source": "observation._training.revivals_used exact counter",
        "revival_budget_source": (
            "positive observation._training.revival_budget; otherwise " "spec.revival_reference_budget"
        ),
        "revival_cost_schedule": "cap*min(revivals_used/revival_budget,1)",
        "hp_loss_source": "observation._training.player_hp_lost exact counter",
        "ordering": "task_outcome+run_progress>bounded_survival_efficiency>decisions",
        "forbidden_shaping": "enemy_hp_delta+damage_dealt+cards_played",
    }
    serialized = json.dumps(payload, sort_keys=True, separators=(",", ":"))
    payload["fingerprint"] = serialized
    payload["fingerprint_sha256"] = hashlib.sha256(serialized.encode("utf-8")).hexdigest()
    return payload


def absolute_death_price_reward_identity() -> dict[str, Any]:
    """Return the v8 factual-reward identity used by hybrid transition v2."""

    task_spec = asdict(ABSOLUTE_DEATH_PRICE_TASK_REWARD_SPEC)
    task_payload: dict[str, Any] = {
        "version": ABSOLUTE_DEATH_PRICE_TASK_REWARD_SPEC.version,
        "spec": task_spec,
        "outcome_source": (
            "environment_result.terminated+typed_transition.facts.combat_result+"
            "typed_transition.facts.run_result+typed_transition.facts.terminal_reason+"
            "observation.run.act"
        ),
        "progress_source": "positive_delta(observation.run.progress_or_floor)",
    }
    payload: dict[str, Any] = {
        "version": ABSOLUTE_DEATH_PRICE_REWARD_SPEC.version,
        "base": task_payload,
        "spec": asdict(ABSOLUTE_DEATH_PRICE_REWARD_SPEC),
        "revival_event_source": "observation._training.revivals_used exact counter",
        "revival_pricing": "learner_absolute_lambda_times_transition_delta",
        "hp_loss_source": "observation._training.player_hp_lost exact counter",
        "ordering_check": "durable-floor-discounted-terminal-margin-v1",
        "forbidden_shaping": "enemy_hp_delta+damage_dealt+cards_played",
    }
    return payload


def act_frontloaded_death_reward_identity(*, revival_budget: int) -> dict[str, Any]:
    """Return the stationary reward-v9 identity for one budget course."""

    if isinstance(revival_budget, bool) or not isinstance(revival_budget, int):
        raise TypeError("reward-v9 revival budget must be an integer")
    if revival_budget < 0:
        raise ValueError("reward-v9 revival budget must be non-negative")
    return {
        "version": ACT_FRONTLOADED_DEATH_REWARD_SPEC.version,
        "base": {
            "version": ABSOLUTE_DEATH_PRICE_TASK_REWARD_SPEC.version,
            "spec": asdict(ABSOLUTE_DEATH_PRICE_TASK_REWARD_SPEC),
        },
        "spec": asdict(ACT_FRONTLOADED_DEATH_REWARD_SPEC),
        "revival_budget": revival_budget,
        "revival_event_source": "observation._training.revivals_used exact counter",
        "revival_pricing": (
            "per_act 0.20*k/(k+1) plus per_run (0.80/revival_budget)*k"
            if revival_budget > 0 else "permadeath: no revival permitted; revival cost is zero"
        ),
        "act_reset": "observation.run.act boundary",
        "hp_loss_source": "observation._training.player_hp_lost exact counter",
        "forbidden_shaping": "enemy_hp_delta+damage_dealt+cards_played",
    }


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


def _training_revival_budget(observation: Mapping[str, Any]) -> int:
    """Return the positive finite course budget used by reward v7.

    Historical/unlimited observations encode the budget as ``-1`` (and some
    fixtures omit it).  Those observations use the reviewed 64-revival
    reference horizon rather than silently collapsing the denominator to one.
    A positive non-integral budget is malformed and fails closed.
    """

    training = _mapping(observation.get("_training"))
    raw = training.get("revival_budget")
    if raw is None:
        return REVIVAL_EFFICIENCY_REWARD_SPEC.revival_reference_budget
    value = _number(raw, default=float("nan"))
    if not math.isfinite(value):
        raise ValueError("training revival budget must be finite")
    if value <= 0.0:
        return REVIVAL_EFFICIENCY_REWARD_SPEC.revival_reference_budget
    if not value.is_integer():
        raise ValueError("training revival budget must be an integer")
    return int(value)


def _revival_efficiency_cost(amount: float, *, revival_budget: int) -> float:
    """Return the versioned, bounded cumulative v7 revival cost.

    ``0.40 * min(k / B, 1)`` preserves a non-zero marginal penalty for every
    revival admitted by finite course budget ``B``.  At B=64 the unit price is
    0.00625; at B=32 it is 0.0125; at B=16 it is 0.025.  The complete revival,
    HP-loss and pace preference remains bounded by 0.99, strictly below one
    terminal-outcome unit.
    """

    if isinstance(revival_budget, bool) or revival_budget <= 0:
        raise ValueError("revival_budget must be a positive integer")
    normalized_amount = max(0.0, float(amount))
    normalized_fraction = min(normalized_amount / float(revival_budget), 1.0)
    return REVIVAL_EFFICIENCY_REWARD_SPEC.revival_cost_cap * normalized_fraction


def _typed_terminal_result(result: EnvironmentResult, *, objective: TaskObjective) -> str:
    transition = result.transition
    if transition is None or not isinstance(transition.facts, Mapping):
        if result.terminated:
            raise ValueError("terminal task result has no typed transition facts")
        return "none"
    fact_name = "combat_result" if objective == "combat" else "run_result"
    supported = {"none", "victory", "defeat", "escaped"} if objective == "combat" else {"none", "victory", "defeat"}
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
    def __init__(
        self,
        objective: TaskObjective,
        *,
        discount: float = 0.997,
        spec: TaskRewardSpec | AbsoluteDeathPriceTaskRewardSpec = TASK_REWARD_SPEC,
    ) -> None:
        if objective not in {"combat", "act1", "run"}:
            raise ValueError("objective must be combat, act1, or run")
        normalized_discount = float(discount)
        if not math.isfinite(normalized_discount) or not 0.0 < normalized_discount <= 1.0:
            raise ValueError("discount must be finite and in (0, 1]")
        self.objective = objective
        self.discount = normalized_discount
        self.spec = spec

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
        if self.objective == "act1" and act >= self.spec.act1_success_act:
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
            raise ValueError("transport/outcome-unknown truncation cannot become training data")
        outcome = self._outcome(
            after,
            deadlock=deadlock,
            horizon_exhausted=horizon_exhausted,
        )
        task_terminal = outcome != "ongoing"
        terminal_reward = {
            "ongoing": 0.0,
            "success": self.spec.success_reward,
            "failure": self.spec.failure_reward,
            "deadlock": self.spec.deadlock_reward,
            "horizon": self.spec.horizon_reward,
        }[outcome]
        progress_reward = 0.0
        if self.objective in {"act1", "run"}:
            before_progress = _run_position(before.observation)[2]
            after_progress = _run_position(after.observation)[2]
            progress_reward = self.spec.run_progress_reward_weight * max(0.0, after_progress - before_progress)
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
        before_revival_budget = _training_revival_budget(before.observation)
        after_revival_budget = _training_revival_budget(after.observation)
        before_hp_lost = _training_counter(before.observation, "player_hp_lost")
        after_hp_lost = _training_counter(after.observation, "player_hp_lost")
        if after_revival_budget != before_revival_budget:
            raise ValueError("training revival budget must remain constant within an episode")
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
        revival_penalty = -(
            _revival_efficiency_cost(
                after_revivals,
                revival_budget=after_revival_budget,
            )
            - _revival_efficiency_cost(
                before_revivals,
                revival_budget=before_revival_budget,
            )
        )
        pace_penalty = -REVIVAL_EFFICIENCY_REWARD_SPEC.pace_budget / float(self.maximum_episode_steps)
        return TaskReward(
            reward=float(base.reward + hp_loss_penalty + revival_penalty + pace_penalty),
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


class AbsoluteDeathPriceRewardCalculator:
    """V8 factual reward; revival price is owned by the hybrid learner.

    The calculator still validates and exports the exact monotonic revival
    delta.  Its scalar ``reward`` contains terminal/progress, bounded HP-loss,
    and pace terms only, so factual replay can be repriced as ``lambda`` moves
    without recollection.
    """

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
        if float(discount) != ABSOLUTE_DEATH_PRICE_REWARD_SPEC.required_discount:
            raise ValueError("absolute-death-price preheat requires discount=1")
        self.maximum_episode_steps = int(maximum_episode_steps)
        self.base = TaskRewardCalculator(
            objective,
            discount=discount,
            spec=ABSOLUTE_DEATH_PRICE_TASK_REWARD_SPEC,
        )

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
        before_revival_budget = _training_revival_budget(before.observation)
        after_revival_budget = _training_revival_budget(after.observation)
        before_hp_lost = _training_counter(before.observation, "player_hp_lost")
        after_hp_lost = _training_counter(after.observation, "player_hp_lost")
        if after_revival_budget != before_revival_budget:
            raise ValueError("training revival budget must remain constant within an episode")
        if after_revivals < before_revivals or after_hp_lost < before_hp_lost:
            raise ValueError("training efficiency counters must be monotonic")

        raw_revival_delta = after_revivals - before_revivals
        if not float(raw_revival_delta).is_integer():
            raise ValueError("training revival delta must be an integer")
        revivals_used_delta = int(raw_revival_delta)
        player_hp_lost_delta = after_hp_lost - before_hp_lost
        hp_loss_penalty = -ABSOLUTE_DEATH_PRICE_REWARD_SPEC.hp_loss_weight * (
            _bounded_resource_score(
                after_hp_lost,
                ABSOLUTE_DEATH_PRICE_REWARD_SPEC.hp_loss_scale,
            )
            - _bounded_resource_score(
                before_hp_lost,
                ABSOLUTE_DEATH_PRICE_REWARD_SPEC.hp_loss_scale,
            )
        )
        pace_penalty = -ABSOLUTE_DEATH_PRICE_REWARD_SPEC.pace_budget / float(self.maximum_episode_steps)
        return TaskReward(
            reward=float(base.reward + hp_loss_penalty + pace_penalty),
            discount=base.discount,
            terminal_reward=base.terminal_reward,
            potential_reward=base.potential_reward,
            task_terminal=base.task_terminal,
            outcome=base.outcome,
            # Explicitly zero: transition v2 carries the delta and learner v3
            # applies the current absolute price inside the TD target.
            revival_penalty=0.0,
            pace_penalty=float(pace_penalty),
            hp_loss_penalty=float(hp_loss_penalty),
            progress_reward=base.progress_reward,
            revivals_used_delta=revivals_used_delta,
            player_hp_lost_delta=float(player_hp_lost_delta),
        )


class ActFrontloadedDeathRewardCalculator:
    """Reward-v9 with a fresh high marginal death price in every Act.

    ``begin_episode`` is intentionally explicit because the Act-local counter
    is an environment-course fact, not learner state.  Hybrid collectors call
    it immediately before resetting/collecting a run.  The total revival and
    HP-loss counters remain authoritative checks on every transition.
    """

    def __init__(
        self,
        *,
        maximum_episode_steps: int,
        revival_budget: int,
        objective: TaskObjective = "combat",
        discount: float = 1.0,
    ) -> None:
        if objective not in {"combat", "act1", "run"}:
            raise ValueError("objective must be combat, act1, or run")
        if isinstance(maximum_episode_steps, bool) or maximum_episode_steps <= 0:
            raise ValueError("maximum_episode_steps must be a positive integer")
        if isinstance(revival_budget, bool) or not isinstance(revival_budget, int):
            raise TypeError("reward-v9 revival budget must be an integer")
        if revival_budget < 0:
            raise ValueError("reward-v9 revival budget must be non-negative")
        if float(discount) != ACT_FRONTLOADED_DEATH_REWARD_SPEC.required_discount:
            raise ValueError("act-frontloaded death reward requires discount=1")
        self.maximum_episode_steps = int(maximum_episode_steps)
        self.revival_budget = int(revival_budget)
        self.base = TaskRewardCalculator(
            objective,
            discount=discount,
            spec=ABSOLUTE_DEATH_PRICE_TASK_REWARD_SPEC,
        )
        self._episode_started = False
        self._current_act = 1
        self._act_revivals = 0
        self._last_total_revivals = 0

    def begin_episode(self) -> None:
        self._episode_started = True
        self._current_act = 1
        self._act_revivals = 0
        self._last_total_revivals = 0

    def _revival_penalty(self, count: int) -> float:
        if self.revival_budget == 0:
            if count != 0:
                raise ValueError("permadeath forbids any revival")
            return 0.0
        penalty = 0.0
        tail = ACT_FRONTLOADED_DEATH_REWARD_SPEC.tail_budget / self.revival_budget
        for ordinal in range(self._act_revivals + 1, self._act_revivals + count + 1):
            penalty += tail + (
                ACT_FRONTLOADED_DEATH_REWARD_SPEC.act_frontload_cap
                / float(ordinal * (ordinal + 1))
            )
        self._act_revivals += count
        return -penalty

    def evaluate(
        self,
        before: EnvironmentResult,
        after: EnvironmentResult,
        *,
        deadlock: bool = False,
        horizon_exhausted: bool = False,
    ) -> TaskReward:
        if not self._episode_started:
            self.begin_episode()
        base = self.base.evaluate(
            before,
            after,
            deadlock=deadlock,
            horizon_exhausted=horizon_exhausted,
        )
        before_revivals = _training_counter(before.observation, "revivals_used")
        after_revivals = _training_counter(after.observation, "revivals_used")
        if self.revival_budget == 0:
            # Never use the legacy v7 fallback (0 -> reference budget 64).
            # A zero-budget evaluation needs explicit simulator evidence.
            for observation in (before.observation, after.observation):
                counters = _mapping(observation.get("_training"))
                for name in ("revival_budget", "revivals_used"):
                    value = counters.get(name)
                    if isinstance(value, bool) or not isinstance(value, int) or value != 0:
                        raise ValueError(f"permadeath requires explicit {name}=0")
            before_budget = after_budget = 0
        else:
            before_budget = _training_revival_budget(before.observation)
            after_budget = _training_revival_budget(after.observation)
        before_hp_lost = _training_counter(before.observation, "player_hp_lost")
        after_hp_lost = _training_counter(after.observation, "player_hp_lost")
        if before_budget != self.revival_budget or after_budget != self.revival_budget:
            raise ValueError("environment revival budget differs from reward-v9 course")
        if after_revivals < before_revivals or after_hp_lost < before_hp_lost:
            raise ValueError("training efficiency counters must be monotonic")

        raw_delta = after_revivals - before_revivals
        if not float(raw_delta).is_integer():
            raise ValueError("training revival delta must be an integer")
        revivals_used_delta = int(raw_delta)
        before_act = max(_run_position(before.observation)[0], 1)
        after_act = max(_run_position(after.observation)[0], 1)
        # A fresh episode or a newly entered Act gets a fresh front-loaded
        # curve.  The run-wide tail remains in force through the fixed unit
        # price and never refunds earlier Act costs.
        if before_revivals < self._last_total_revivals or before_act < self._current_act:
            self.begin_episode()
        if before_act != self._current_act:
            self._current_act = before_act
            self._act_revivals = 0
        revival_penalty = self._revival_penalty(revivals_used_delta)
        self._last_total_revivals = int(after_revivals)
        if after_act != before_act:
            self._current_act = after_act
            self._act_revivals = 0

        player_hp_lost_delta = after_hp_lost - before_hp_lost
        hp_loss_penalty = -ACT_FRONTLOADED_DEATH_REWARD_SPEC.hp_loss_weight * (
            _bounded_resource_score(
                after_hp_lost,
                ACT_FRONTLOADED_DEATH_REWARD_SPEC.hp_loss_scale,
            )
            - _bounded_resource_score(
                before_hp_lost,
                ACT_FRONTLOADED_DEATH_REWARD_SPEC.hp_loss_scale,
            )
        )
        pace_penalty = -ACT_FRONTLOADED_DEATH_REWARD_SPEC.pace_budget / float(
            self.maximum_episode_steps
        )
        return TaskReward(
            reward=float(base.reward + hp_loss_penalty + revival_penalty + pace_penalty),
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
    "ABSOLUTE_DEATH_PRICE_REWARD_SPEC",
    "ABSOLUTE_DEATH_PRICE_TASK_REWARD_SPEC",
    "ACT_FRONTLOADED_DEATH_REWARD_SPEC",
    "REVIVAL_EFFICIENCY_REWARD_SPEC",
    "TASK_REWARD_SPEC",
    "AbsoluteDeathPriceRewardCalculator",
    "AbsoluteDeathPriceRewardSpec",
    "AbsoluteDeathPriceTaskRewardSpec",
    "ActFrontloadedDeathRewardCalculator",
    "ActFrontloadedDeathRewardSpec",
    "RevivalEfficiencyRewardCalculator",
    "RevivalEfficiencyRewardSpec",
    "TaskObjective",
    "TaskOutcome",
    "TaskReward",
    "TaskRewardCalculator",
    "TaskRewardSpec",
    "absolute_death_price_reward_identity",
    "act_frontloaded_death_reward_identity",
    "revival_efficiency_reward_identity",
    "task_reward_identity",
]
