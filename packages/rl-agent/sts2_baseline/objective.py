"""Auditable, versioned task rewards for the recurrent v2 training line.

The normal reward contains no card, boss, encounter, deck, route, or
action-specific rules.  The optional native-revival preheat reward adds only an
exact relic-consumption event and a bounded per-step pace cost.  Neither reward
uses backend-provided reward scalars or inferred healing/HP-jump heuristics.
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
TaskOutcome = Literal["ongoing", "success", "failure", "deadlock"]


@dataclass(frozen=True, slots=True)
class TaskRewardSpec:
    version: str = field(default="sts2-task-reward-v2", init=False)
    discount: float = field(default=0.997, init=False)
    success_reward: float = field(default=1.0, init=False)
    failure_reward: float = field(default=-1.0, init=False)
    deadlock_reward: float = field(default=-1.0, init=False)
    player_hp_potential_weight: float = field(default=0.05, init=False)
    enemy_progress_potential_weight: float = field(default=0.05, init=False)
    run_progress_potential_weight: float = field(default=0.10, init=False)
    potential_delta_abs_cap: float = field(default=0.20, init=False)
    act1_success_act: int = field(default=2, init=False)
    fallback_run_floor_cap: float = field(default=60.0, init=False)


TASK_REWARD_SPEC: Final = TaskRewardSpec()


@dataclass(frozen=True, slots=True)
class RevivalEfficiencyRewardSpec:
    """Bounded lexicographic curriculum layered over the normal combat task.

    For the preheat profile the episode horizon is at most 512 steps.  The
    terminal margin dominates every possible revival/pace cost, and one native
    revival costs more than the entire pace budget.  The resulting preference
    is therefore: win first, then avoid revival, then finish in fewer steps.
    """

    version: str = field(default="sts2-native-revival-efficiency-v1", init=False)
    terminal_margin: float = field(default=3.0, init=False)
    native_revival_penalty: float = field(default=-1.0, init=False)
    pace_penalty_per_step: float = field(default=-1.0 / 2048.0, init=False)
    maximum_episode_steps: int = field(default=512, init=False)


REVIVAL_EFFICIENCY_REWARD_SPEC: Final = RevivalEfficiencyRewardSpec()


def task_reward_identity() -> dict[str, Any]:
    payload: dict[str, Any] = {
        "version": TASK_REWARD_SPEC.version,
        "spec": asdict(TASK_REWARD_SPEC),
        "outcome_source": (
            "environment_result.terminated+typed_transition.facts.combat_result+"
            "typed_transition.facts.terminal_reason+observation.run.act"
        ),
        "terminal_reason_policy": "transition/result exact equality",
        "transport_truncation": "reject",
        "collector_horizon": "bootstrap",
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
        "revival_event_source": "typed_transition.facts.relics_used",
        "ordering": "task_outcome>native_revival_count>environment_steps",
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


def _ratio(current: Any, maximum: Any) -> float:
    maximum_value = _number(maximum)
    current_value = _number(current)
    if maximum_value <= 0.0:
        return 0.0
    return min(max(current_value / maximum_value, 0.0), 1.0)


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


def _potential(observation: Mapping[str, Any], objective: TaskObjective) -> float:
    player = _mapping(observation.get("player"))
    player_hp = player.get("hp", player.get("current_hp"))
    player_max_hp = player.get("max_hp", player.get("maximum_hp"))
    player_ratio = _ratio(player_hp, player_max_hp)

    combat = _mapping(observation.get("combat"))
    enemies = combat.get("enemies", ())
    current_enemy_hp = 0.0
    maximum_enemy_hp = 0.0
    if isinstance(enemies, list | tuple):
        for raw_enemy in enemies:
            enemy = _mapping(raw_enemy)
            current_enemy_hp += max(0.0, _number(enemy.get("hp")))
            maximum_enemy_hp += max(
                0.0,
                _number(enemy.get("max_hp", enemy.get("maximum_hp"))),
            )
    enemy_progress = (
        1.0 - min(current_enemy_hp / maximum_enemy_hp, 1.0)
        if maximum_enemy_hp > 0.0
        else 0.0
    )
    _, _, run_progress = _run_position(observation)
    result = TASK_REWARD_SPEC.player_hp_potential_weight * player_ratio
    if objective in {"combat", "act1", "run"}:
        result += TASK_REWARD_SPEC.enemy_progress_potential_weight * enemy_progress
    if objective in {"act1", "run"}:
        result += TASK_REWARD_SPEC.run_progress_potential_weight * run_progress
    return result


def _typed_combat_result(result: EnvironmentResult) -> str:
    transition = result.transition
    if transition is None or not isinstance(transition.facts, Mapping):
        if result.terminated:
            raise ValueError("terminal task result has no typed transition facts")
        return "none"
    raw = transition.facts.get("combat_result")
    if not isinstance(raw, str) or raw not in {
        "none",
        "victory",
        "defeat",
        "escaped",
    }:
        raise ValueError("typed combat_result is missing or unsupported")
    if "terminal_reason" not in transition.facts:
        raise ValueError("typed terminal_reason is missing")
    fact_reason = transition.facts["terminal_reason"]
    if fact_reason is not None and not isinstance(fact_reason, str):
        raise ValueError("typed terminal_reason must be text or null")
    if fact_reason != result.terminal_reason:
        raise ValueError("transition and result terminal_reason identities differ")
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
    ) -> TaskOutcome:
        if deadlock:
            return "deadlock"
        act, _, _ = _run_position(after.observation)
        if self.objective == "act1" and act >= TASK_REWARD_SPEC.act1_success_act:
            return "success"
        if not after.terminated:
            return "ongoing"
        combat_result = _typed_combat_result(after)
        if self.objective == "combat":
            return "success" if combat_result == "victory" else "failure"
        if self.objective == "act1":
            return "failure"
        return "success" if combat_result == "victory" else "failure"

    def evaluate(
        self,
        before: EnvironmentResult,
        after: EnvironmentResult,
        *,
        deadlock: bool = False,
    ) -> TaskReward:
        if after.truncated:
            raise ValueError(
                "transport/outcome-unknown truncation cannot become training data"
            )
        outcome = self._outcome(after, deadlock=deadlock)
        task_terminal = outcome != "ongoing"
        terminal_reward = {
            "ongoing": 0.0,
            "success": TASK_REWARD_SPEC.success_reward,
            "failure": TASK_REWARD_SPEC.failure_reward,
            "deadlock": TASK_REWARD_SPEC.deadlock_reward,
        }[outcome]
        before_potential = _potential(before.observation, self.objective)
        after_potential = (
            0.0 if task_terminal else _potential(after.observation, self.objective)
        )
        potential_reward = self.discount * after_potential - before_potential
        cap = TASK_REWARD_SPEC.potential_delta_abs_cap
        potential_reward = min(max(potential_reward, -cap), cap)
        return TaskReward(
            reward=float(terminal_reward + potential_reward),
            discount=0.0 if task_terminal else self.discount,
            terminal_reward=float(terminal_reward),
            potential_reward=float(potential_reward),
            task_terminal=task_terminal,
            outcome=outcome,
        )


class RevivalEfficiencyRewardCalculator:
    """Combat preheat reward using exact native relic-consumption events."""

    def __init__(
        self,
        *,
        revival_relic_id: str,
        discount: float = 0.997,
        maximum_episode_steps: int = 512,
    ) -> None:
        if not isinstance(revival_relic_id, str) or not revival_relic_id.strip():
            raise TypeError("revival_relic_id must be non-empty text")
        if maximum_episode_steps > REVIVAL_EFFICIENCY_REWARD_SPEC.maximum_episode_steps:
            raise ValueError(
                "native revival preheat horizon exceeds the reward ordering proof"
            )
        self.revival_relic_id = revival_relic_id.strip().upper()
        self.base = TaskRewardCalculator("combat", discount=discount)

    def evaluate(
        self,
        before: EnvironmentResult,
        after: EnvironmentResult,
        *,
        deadlock: bool = False,
    ) -> TaskReward:
        base = self.base.evaluate(before, after, deadlock=deadlock)
        facts = after.transition.facts if after.transition is not None else {}
        raw_used = facts.get("relics_used", ()) if isinstance(facts, Mapping) else ()
        used = {
            str(item).upper()
            for item in raw_used
        } if isinstance(raw_used, list | tuple) else set()
        revival_penalty = (
            REVIVAL_EFFICIENCY_REWARD_SPEC.native_revival_penalty
            if self.revival_relic_id in used
            else 0.0
        )
        pace_penalty = REVIVAL_EFFICIENCY_REWARD_SPEC.pace_penalty_per_step
        terminal_margin = {
            "ongoing": 0.0,
            "success": REVIVAL_EFFICIENCY_REWARD_SPEC.terminal_margin,
            "failure": -REVIVAL_EFFICIENCY_REWARD_SPEC.terminal_margin,
            "deadlock": -REVIVAL_EFFICIENCY_REWARD_SPEC.terminal_margin,
        }[base.outcome]
        return TaskReward(
            reward=float(base.reward + terminal_margin + revival_penalty + pace_penalty),
            discount=base.discount,
            terminal_reward=float(base.terminal_reward + terminal_margin),
            potential_reward=base.potential_reward,
            task_terminal=base.task_terminal,
            outcome=base.outcome,
            revival_penalty=float(revival_penalty),
            pace_penalty=float(pace_penalty),
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
