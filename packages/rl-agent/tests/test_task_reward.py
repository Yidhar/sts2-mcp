from __future__ import annotations

import pytest

from sts2_baseline import (
    RevivalEfficiencyRewardCalculator,
    TaskRewardCalculator,
    revival_efficiency_reward_identity,
    task_reward_identity,
)
from sts2_rl.contracts import EnvironmentResult, EnvironmentTransition
from sts2_rl.preheat_gate import validate_counter_transition
from sts2_rl.transitions import derive_transition_facts


def _result(
    *,
    step: int,
    act: int = 1,
    floor: int = 1,
    hp: int = 50,
    max_hp: int = 50,
    enemy_hp: int = 10,
    terminated: bool = False,
    combat_result: str = "none",
    terminal_reason: str | None = None,
    truncated: bool = False,
    relics_used: tuple[str, ...] = (),
    revivals_used: int = 0,
    player_hp_lost: float = 0.0,
    revivals_used_delta: int = 0,
    player_hp_lost_delta: float = 0.0,
) -> EnvironmentResult:
    reason = terminal_reason
    if terminated and reason is None:
        reason = "combat_victory" if combat_result == "victory" else "combat_defeat"
    return EnvironmentResult(
        episode_id="episode",
        step_index=step,
        observation={
            "player": {"hp": hp, "max_hp": max_hp},
            "run": {"act": act, "floor": floor},
            "combat": {"enemies": [{"hp": enemy_hp, "max_hp": 20}]},
            "_training": {
                "revival_budget": -1,
                "revivals_used": revivals_used,
                "player_hp_lost": player_hp_lost,
            },
        },
        transition=EnvironmentTransition(
            episode_id="episode",
            step_index=step,
            before_state_version=step,
            after_state_version=step + 1,
            facts={
                "combat_result": combat_result,
                "terminal_reason": reason,
                "relics_used": list(relics_used),
                "revivals_used_delta": revivals_used_delta,
                "player_hp_lost_delta": player_hp_lost_delta,
            },
        ),
        terminated=terminated,
        truncated=truncated,
        terminal_reason=reason,
    )


def test_reward_identity_is_versioned_hashed_and_forbids_damage_shaping() -> None:
    identity = task_reward_identity()
    assert identity["version"] == "sts2-task-reward-v3"
    assert len(identity["fingerprint_sha256"]) == 64
    assert identity["forbidden_shaping"] == "enemy_hp_delta+damage_dealt+cards_played"


def test_act1_success_is_a_positive_task_terminal() -> None:
    calculator = TaskRewardCalculator("act1")
    reward = calculator.evaluate(_result(step=0), _result(step=1, act=2))
    assert reward.task_terminal
    assert reward.outcome == "success"
    assert reward.terminal_reward == 1.0
    assert reward.discount == 0.0


def test_act1_death_deadlock_and_collector_horizon_are_failures() -> None:
    calculator = TaskRewardCalculator("act1")
    death = calculator.evaluate(
        _result(step=0),
        _result(step=1, hp=0, terminated=True, combat_result="defeat"),
    )
    deadlock = calculator.evaluate(_result(step=0), _result(step=1), deadlock=True)
    horizon = calculator.evaluate(
        _result(step=0),
        _result(step=1),
        horizon_exhausted=True,
    )
    assert death.outcome == "failure"
    assert deadlock.outcome == "deadlock"
    assert horizon.outcome == "horizon"
    assert death.terminal_reward == deadlock.terminal_reward == horizon.terminal_reward == -1.0
    assert death.discount == deadlock.discount == horizon.discount == 0.0


def test_run_distance_rewards_only_monotonic_forward_progress() -> None:
    calculator = TaskRewardCalculator("act1", discount=0.997)
    forward = calculator.evaluate(
        _result(step=0, floor=1, enemy_hp=20),
        _result(step=1, floor=2, enemy_hp=10),
    )
    backward = calculator.evaluate(
        _result(step=1, floor=2),
        _result(step=2, floor=1),
    )
    assert forward.outcome == "ongoing"
    assert not forward.task_terminal
    assert forward.discount == 0.997
    assert forward.progress_reward > 0.0
    assert forward.reward == forward.progress_reward
    assert backward.progress_reward == 0.0


def test_enemy_damage_and_cards_are_not_reward_signals() -> None:
    calculator = TaskRewardCalculator("combat")
    no_damage = calculator.evaluate(
        _result(step=0, enemy_hp=20),
        _result(step=1, enemy_hp=20),
    )
    large_damage = calculator.evaluate(
        _result(step=0, enemy_hp=20),
        _result(step=1, enemy_hp=1),
    )
    assert no_damage.reward == 0.0
    assert large_damage.reward == 0.0


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


def test_preheat_reward_uses_exact_hp_loss_and_revival_counters() -> None:
    identity = revival_efficiency_reward_identity()
    assert identity["version"] == "sts2-run-survival-efficiency-v3"
    calculator = RevivalEfficiencyRewardCalculator(
        revival_relic_id="RELIC.LIZARD_TAIL",
        maximum_episode_steps=512,
    )
    before = _result(step=0)
    ordinary = calculator.evaluate(before, _result(step=1))
    costly = calculator.evaluate(
        before,
        _result(
            step=1,
            hp=40,
            revivals_used=1,
            player_hp_lost=60,
            revivals_used_delta=1,
            player_hp_lost_delta=60,
        ),
    )
    assert ordinary.hp_loss_penalty == ordinary.revival_penalty == 0.0
    assert ordinary.pace_penalty < 0.0
    assert costly.hp_loss_penalty < 0.0
    assert costly.revival_penalty < 0.0
    assert costly.reward < ordinary.reward
    assert costly.revivals_used_delta == 1
    assert costly.player_hp_lost_delta == 60


def test_healing_does_not_erase_exact_hp_loss_cost() -> None:
    calculator = RevivalEfficiencyRewardCalculator(
        revival_relic_id="RELIC.LIZARD_TAIL",
        maximum_episode_steps=512,
    )
    reward = calculator.evaluate(
        _result(step=0, hp=1, player_hp_lost=49),
        _result(
            step=1,
            hp=40,
            player_hp_lost=50,
            player_hp_lost_delta=1,
        ),
    )
    assert reward.player_hp_lost_delta == 1
    assert reward.hp_loss_penalty < 0.0


def test_full_run_preheat_combines_forward_progress_with_run_scoped_costs() -> None:
    calculator = RevivalEfficiencyRewardCalculator(
        revival_relic_id="RELIC.LIZARD_TAIL",
        objective="run",
        maximum_episode_steps=10_000,
    )
    reward = calculator.evaluate(
        _result(step=0, floor=1),
        _result(
            step=1,
            floor=2,
            revivals_used=1,
            player_hp_lost=40,
            revivals_used_delta=1,
            player_hp_lost_delta=40,
        ),
    )
    assert reward.outcome == "ongoing"
    assert reward.progress_reward > 0.0
    assert reward.hp_loss_penalty < 0.0
    assert reward.revival_penalty < 0.0
    assert reward.pace_penalty < 0.0
    assert reward.reward == pytest.approx(
        reward.progress_reward
        + reward.hp_loss_penalty
        + reward.revival_penalty
        + reward.pace_penalty
    )


def test_full_run_preheat_progresses_past_act1_instead_of_terminating() -> None:
    calculator = RevivalEfficiencyRewardCalculator(
        revival_relic_id="RELIC.LIZARD_TAIL",
        objective="run",
        maximum_episode_steps=10_000,
    )
    act2 = calculator.evaluate(
        _result(step=0, act=1, floor=17),
        _result(step=1, act=2, floor=18),
    )
    assert act2.outcome == "ongoing"
    assert not act2.task_terminal
    assert act2.discount == 1.0


def test_survival_preheat_still_makes_every_win_better_than_every_loss() -> None:
    calculator = RevivalEfficiencyRewardCalculator(
        revival_relic_id="RELIC.LIZARD_TAIL",
        maximum_episode_steps=512,
    )
    costly_victory = calculator.evaluate(
        _result(step=0),
        _result(
            step=1,
            terminated=True,
            combat_result="victory",
            enemy_hp=0,
            revivals_used=1_000_000,
            player_hp_lost=1_000_000,
        ),
    )
    clean_defeat = calculator.evaluate(
        _result(step=0),
        _result(step=1, terminated=True, combat_result="defeat", hp=0),
    )
    assert costly_victory.outcome == "success"
    assert clean_defeat.outcome == "failure"
    assert costly_victory.reward > clean_defeat.reward


def test_preheat_requires_undiscounted_efficiency_telescoping() -> None:
    with pytest.raises(ValueError, match="undiscounted"):
        RevivalEfficiencyRewardCalculator(
            revival_relic_id="RELIC.LIZARD_TAIL",
            maximum_episode_steps=512,
            discount=0.997,
        )


def test_counter_gate_agrees_with_typed_transition_facts() -> None:
    before = _result(step=0, revivals_used=2, player_hp_lost=90)
    after = _result(
        step=1,
        revivals_used=3,
        player_hp_lost=130,
        revivals_used_delta=1,
        player_hp_lost_delta=40,
    )
    validate_counter_transition(before, after)
    bad = _result(
        step=1,
        revivals_used=3,
        player_hp_lost=130,
        revivals_used_delta=0,
        player_hp_lost_delta=40,
    )
    with pytest.raises(RuntimeError, match="revival delta"):
        validate_counter_transition(before, bad)


def test_relic_consumption_fact_uses_native_used_state_not_hp_heuristics() -> None:
    before = {
        "player": {
            "hp": 1,
            "max_hp": 80,
            "relics": [{"id": "RELIC.LIZARD_TAIL", "is_used_up": False}],
        }
    }
    after = {
        "player": {
            "hp": 40,
            "max_hp": 80,
            "relics": [{"id": "RELIC.LIZARD_TAIL", "is_used_up": True}],
        }
    }
    assert derive_transition_facts(before, after).relics_used == (
        "RELIC.LIZARD_TAIL",
    )
    after["player"]["relics"] = []  # type: ignore[index]
    assert derive_transition_facts(before, after).relics_used == ()


def test_explicit_combat_victory_reason_precedes_missing_terminal_player_hp() -> None:
    facts = derive_transition_facts(
        {"player": {"hp": 20}},
        {},
        terminated=True,
        terminal_reason="combat_victory",
    )
    assert facts.combat_result == "victory"
