from __future__ import annotations

import pytest

from sts2_baseline import (
    REVIVAL_EFFICIENCY_REWARD_SPEC,
    RevivalEfficiencyRewardCalculator,
    TaskRewardCalculator,
    revival_efficiency_reward_identity,
    task_reward_identity,
)
from sts2_rl.contracts import EnvironmentResult, EnvironmentTransition
from sts2_rl.transitions import derive_transition_facts


def _result(
    *,
    step: int,
    act: int = 1,
    floor: int = 1,
    hp: int = 50,
    enemy_hp: int = 10,
    terminated: bool = False,
    combat_result: str = "none",
    truncated: bool = False,
    relics_used: tuple[str, ...] = (),
) -> EnvironmentResult:
    return EnvironmentResult(
        episode_id="episode",
        step_index=step,
        observation={
            "player": {"hp": hp, "max_hp": 50},
            "run": {"act": act, "floor": floor},
            "combat": {"enemies": [{"hp": enemy_hp, "max_hp": 20}]},
        },
        transition=EnvironmentTransition(
            episode_id="episode",
            step_index=step,
            before_state_version=step,
            after_state_version=step + 1,
            facts={
                "combat_result": combat_result,
                "terminal_reason": "terminal" if terminated else None,
                "relics_used": list(relics_used),
            },
        ),
        terminated=terminated,
        truncated=truncated,
        terminal_reason="terminal" if terminated else None,
    )


def test_reward_identity_is_versioned_and_hashed() -> None:
    identity = task_reward_identity()
    assert identity["version"] == "sts2-task-reward-v2"
    assert len(identity["fingerprint_sha256"]) == 64
    assert "act1_success_act" in identity["fingerprint"]


def test_act1_success_is_the_only_positive_task_terminal() -> None:
    calculator = TaskRewardCalculator("act1")
    reward = calculator.evaluate(_result(step=0), _result(step=1, act=2))
    assert reward.task_terminal
    assert reward.outcome == "success"
    assert reward.terminal_reward == 1.0
    assert reward.discount == 0.0


def test_act1_death_and_semantic_deadlock_are_failures() -> None:
    calculator = TaskRewardCalculator("act1")
    death = calculator.evaluate(
        _result(step=0),
        _result(step=1, hp=0, terminated=True, combat_result="defeat"),
    )
    deadlock = calculator.evaluate(
        _result(step=0),
        _result(step=1),
        deadlock=True,
    )
    assert death.outcome == "failure"
    assert deadlock.outcome == "deadlock"
    assert death.terminal_reward == deadlock.terminal_reward == -1.0
    assert death.discount == deadlock.discount == 0.0


def test_ongoing_reward_is_bounded_potential_delta_and_bootstraps() -> None:
    calculator = TaskRewardCalculator("act1", discount=0.997)
    reward = calculator.evaluate(
        _result(step=0, floor=1, enemy_hp=20),
        _result(step=1, floor=2, enemy_hp=10),
    )
    assert reward.outcome == "ongoing"
    assert not reward.task_terminal
    assert reward.discount == 0.997
    assert abs(reward.potential_reward) <= 0.20
    assert reward.reward == reward.potential_reward


def test_transport_truncation_never_becomes_training_reward() -> None:
    with pytest.raises(ValueError, match="transport"):
        TaskRewardCalculator("act1").evaluate(
            _result(step=0),
            _result(step=1, truncated=True),
        )


def test_combat_objective_uses_typed_terminal_outcome() -> None:
    calculator = TaskRewardCalculator("combat")
    victory = calculator.evaluate(
        _result(step=0),
        _result(step=1, terminated=True, combat_result="victory", enemy_hp=0),
    )
    defeat = calculator.evaluate(
        _result(step=0),
        _result(step=1, terminated=True, combat_result="defeat", hp=0),
    )
    assert victory.outcome == "success"
    assert defeat.outcome == "failure"


def test_native_revival_reward_is_exact_and_lexicographically_bounded() -> None:
    identity = revival_efficiency_reward_identity()
    assert identity["version"] == "sts2-native-revival-efficiency-v1"
    calculator = RevivalEfficiencyRewardCalculator(
        revival_relic_id="RELIC.LIZARD_TAIL",
        maximum_episode_steps=512,
    )
    ordinary = calculator.evaluate(_result(step=0), _result(step=1))
    revived = calculator.evaluate(
        _result(step=0),
        _result(step=1, relics_used=("RELIC.LIZARD_TAIL",)),
    )
    assert ordinary.revival_penalty == 0.0
    assert revived.revival_penalty == -1.0
    assert ordinary.pace_penalty == revived.pace_penalty < 0.0
    assert ordinary.reward - revived.reward == pytest.approx(1.0)
    assert abs(512 * REVIVAL_EFFICIENCY_REWARD_SPEC.pace_penalty_per_step) < abs(
        REVIVAL_EFFICIENCY_REWARD_SPEC.native_revival_penalty
    )


def test_native_revival_preheat_still_prioritizes_winning() -> None:
    calculator = RevivalEfficiencyRewardCalculator(
        revival_relic_id="RELIC.LIZARD_TAIL",
        maximum_episode_steps=512,
    )
    victory = calculator.evaluate(
        _result(step=0),
        _result(
            step=1,
            terminated=True,
            combat_result="victory",
            enemy_hp=0,
            relics_used=("RELIC.LIZARD_TAIL",),
        ),
    )
    defeat = calculator.evaluate(
        _result(step=0),
        _result(step=1, terminated=True, combat_result="defeat", hp=0),
    )
    assert victory.outcome == "success"
    assert defeat.outcome == "failure"
    assert victory.reward > defeat.reward


def test_relic_consumption_fact_uses_native_used_state_not_hp_heuristics() -> None:
    before = {
        "player": {
            "hp": 1,
            "max_hp": 80,
            "relics": [
                {"id": "RELIC.LIZARD_TAIL", "is_used_up": False},
            ],
        }
    }
    after = {
        "player": {
            "hp": 40,
            "max_hp": 80,
            "relics": [
                {"id": "RELIC.LIZARD_TAIL", "is_used_up": True},
            ],
        }
    }
    assert derive_transition_facts(before, after).relics_used == (
        "RELIC.LIZARD_TAIL",
    )
    after["player"]["relics"] = []  # type: ignore[index]
    assert derive_transition_facts(before, after).relics_used == ()
