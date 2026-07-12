from __future__ import annotations

import math

import pytest

from sts2_baseline import (
    BaselineRewardCalculator,
    BaselineTransition,
    PotentialState,
    baseline_reward_identity,
)


def test_reward_identity_covers_fact_projection_and_terminal_mapping() -> None:
    identity = baseline_reward_identity()
    projection = identity["transition_projection"]
    assert projection["run_progress_floor_cap"] == 60.0
    assert projection["combat_result_victory"] == "win"
    assert projection["combat_result_defeat"] == "loss"
    assert projection["combat_result_escaped"] == "loss"
    assert projection["collector_horizon_truncation_kind"] == "collector_horizon"
    assert len(identity["fingerprint_sha256"]) == 64


def _combat_transition(*, enemy_max_hp: float, enemy_after: float, result: str = "none") -> BaselineTransition:
    return BaselineTransition(
        episode_id="combat-episode",
        step_index=0,
        action_handle="action:attack",
        before=PotentialState(
            player_hp=80,
            player_max_hp=80,
            enemy_hp=enemy_max_hp,
            enemy_max_hp=enemy_max_hp,
            in_combat=True,
        ),
        after=PotentialState(
            player_hp=0 if result == "loss" else 80,
            player_max_hp=80,
            enemy_hp=enemy_after,
            enemy_max_hp=enemy_max_hp,
            in_combat=result == "none",
        ),
        combat_result=result,  # type: ignore[arg-type]
    )


def test_enemy_damage_shaping_is_invariant_to_boss_hp_scale() -> None:
    calculator = BaselineRewardCalculator("combat")

    small = calculator.evaluate(_combat_transition(enemy_max_hp=100, enemy_after=50))
    huge = calculator.evaluate(_combat_transition(enemy_max_hp=10_000, enemy_after=5_000))

    assert small.total == pytest.approx(huge.total)
    assert small.potential == pytest.approx(calculator.spec.discount * 0.15 - 0.10)


def test_dealing_all_boss_hp_then_losing_cannot_be_positive_or_scale_with_hp() -> None:
    calculator = BaselineRewardCalculator("combat")

    small = calculator.evaluate(_combat_transition(enemy_max_hp=100, enemy_after=0, result="loss"))
    huge = calculator.evaluate(_combat_transition(enemy_max_hp=100_000, enemy_after=0, result="loss"))

    assert small.total < 0.0
    assert small.total == pytest.approx(huge.total)
    assert small.terminal == -1.0
    assert abs(small.potential) <= calculator.spec.dense_reward_abs_cap


def test_combat_and_run_terminal_rewards_are_separate() -> None:
    transition = BaselineTransition(
        episode_id="full-run",
        step_index=12,
        action_handle="action:finish-combat",
        before=PotentialState(
            player_hp=50,
            player_max_hp=80,
            enemy_hp=1,
            enemy_max_hp=100,
            run_progress=0.25,
            in_combat=True,
        ),
        after=PotentialState(player_hp=50, player_max_hp=80, run_progress=0.25),
        combat_result="win",
        run_result="none",
    )

    combat = BaselineRewardCalculator("combat").evaluate(transition)
    run = BaselineRewardCalculator("run").evaluate(transition)

    assert combat.terminal == 1.0
    assert run.terminal == 0.0
    assert math.isfinite(run.total)


def test_collector_horizon_is_censored_not_a_game_loss() -> None:
    transition = BaselineTransition(
        episode_id="horizon",
        step_index=100,
        action_handle="action:continue",
        before=PotentialState(player_hp=60, player_max_hp=80),
        after=PotentialState(player_hp=60, player_max_hp=80),
        truncated=True,
        metadata={"truncation_kind": "collector_horizon"},
    )

    breakdown = BaselineRewardCalculator("run").evaluate(transition)

    assert breakdown.terminal == 0.0
    assert breakdown.total > -1.0
    assert abs(breakdown.potential) <= breakdown.before_potential


def test_untyped_or_transport_truncation_cannot_become_a_loss_label() -> None:
    transition = BaselineTransition(
        episode_id="transport-timeout",
        step_index=7,
        action_handle="action:unknown",
        before=PotentialState(player_hp=60, player_max_hp=80),
        after=PotentialState(player_hp=60, player_max_hp=80),
        truncated=True,
    )

    with pytest.raises(ValueError, match="must be discarded"):
        BaselineRewardCalculator("run").evaluate(transition)
