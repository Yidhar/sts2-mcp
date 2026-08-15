from __future__ import annotations

import pytest

pytest.importorskip("torch")

from sts2_rl.joint import (
    AbsoluteDeathPriceController,
    DeathPriceEpisodeOutcome,
    death_price_course,
    validate_discounted_outcome_margin,
)


def _outcome(
    *,
    act1: bool,
    won: bool = False,
    act1_revivals: int | None = None,
    run_revivals: int = 0,
) -> DeathPriceEpisodeOutcome:
    return DeathPriceEpisodeOutcome(
        act1_cleared=act1,
        run_success=won,
        act1_revivals=act1_revivals,
        run_revivals=run_revivals,
    )


def _observe_block(
    controller: AbsoluteDeathPriceController,
    rows: list[DeathPriceEpisodeOutcome],
) -> None:
    assert len(rows) == controller.course.support_window
    closed = [controller.observe_episode(row) for row in rows]
    assert closed[-1]
    assert not any(closed[:-1])


def _gate(controller: AbsoluteDeathPriceController, *, policy_version: int = 0) -> None:
    controller.observe_heldout(
        act1_rate=0.8,
        run_rate=0.25,
        policy_version=policy_version,
    )


def test_lambda_uses_only_success_conditioned_revival_counts() -> None:
    controller = AbsoluteDeathPriceController(death_price_course(64))
    _gate(controller)
    # Forty-two early failures report zero run revivals. They are not cheap
    # examples: only the eight factual Act-1 clears define K_act1|clear.
    rows = [_outcome(act1=False) for _ in range(42)] + [
        _outcome(act1=True, act1_revivals=10, run_revivals=30) for _ in range(8)
    ]
    _observe_block(controller, rows)
    assert controller.metrics.last_support_successes == 8
    assert controller.metrics.act1_ema == 10.0
    assert controller.lambda_value == pytest.approx(0.00625 + 0.00025 * 4.0)


def test_lambda_freezes_without_numeric_success_support() -> None:
    controller = AbsoluteDeathPriceController(death_price_course(64))
    _gate(controller)
    original = controller.lambda_value
    rows = [_outcome(act1=False) for _ in range(43)] + [_outcome(act1=True, act1_revivals=20) for _ in range(7)]
    _observe_block(controller, rows)
    assert controller.lambda_value == original
    assert controller.metrics.freeze_reason == "insufficient_success_support"


def test_lambda_freezes_when_latest_heldout_falls_below_half_baseline() -> None:
    controller = AbsoluteDeathPriceController(death_price_course(64))
    _gate(controller)
    controller.observe_heldout(act1_rate=0.39, run_rate=0.20, policy_version=10)
    original = controller.lambda_value
    _observe_block(
        controller,
        [_outcome(act1=True, act1_revivals=20) for _ in range(50)],
    )
    assert controller.lambda_value == original
    assert controller.metrics.freeze_reason == "heldout_below_course_floor"


def test_lambda_is_a_ratchet_and_never_decreases_below_target() -> None:
    controller = AbsoluteDeathPriceController(death_price_course(64))
    _gate(controller)
    _observe_block(
        controller,
        [_outcome(act1=True, act1_revivals=10) for _ in range(50)],
    )
    raised = controller.lambda_value
    low_revival_block = [_outcome(act1=True, act1_revivals=1) for _ in range(50)]
    # The EMA still remembers the high-revival block, so the price can keep
    # rising briefly. Once the EMA is below target, another good block must
    # leave the ratchet unchanged rather than lowering it.
    for policy_version in range(1, 4):
        _gate(controller, policy_version=policy_version)
        _observe_block(controller, low_revival_block)
    below_target_price = controller.lambda_value
    assert below_target_price >= raised
    assert controller.metrics.act1_ema is not None
    assert controller.metrics.act1_ema < controller.course.act1_target

    _gate(controller, policy_version=4)
    _observe_block(controller, low_revival_block)
    assert controller.lambda_value == below_target_price
    assert controller.metrics.freeze_reason == "at_or_below_target"


def test_act1_statistic_hands_off_to_successful_run_statistic() -> None:
    controller = AbsoluteDeathPriceController(death_price_course(64))
    _gate(controller)
    for policy_version in range(2):
        if policy_version:
            _gate(controller, policy_version=policy_version)
        _observe_block(
            controller,
            [
                _outcome(
                    act1=True,
                    won=index < 8,
                    act1_revivals=2,
                    run_revivals=40,
                )
                for index in range(50)
            ],
        )
    assert controller.metrics.active_statistic == "run"
    before = controller.lambda_value
    _gate(controller, policy_version=2)
    _observe_block(
        controller,
        [
            _outcome(
                act1=True,
                won=index < 8,
                act1_revivals=2,
                run_revivals=40,
            )
            for index in range(50)
        ],
    )
    assert controller.metrics.last_support_successes == 8
    assert controller.lambda_value > before


def test_lambda_requires_a_fresh_heldout_gate_for_each_supported_block() -> None:
    controller = AbsoluteDeathPriceController(death_price_course(64))
    _gate(controller)
    rows = [_outcome(act1=True, act1_revivals=10) for _ in range(50)]
    _observe_block(controller, rows)
    raised = controller.lambda_value

    _observe_block(controller, rows)
    assert controller.lambda_value == raised
    assert controller.metrics.freeze_reason == "stale_heldout_gate"

    _gate(controller, policy_version=1)
    _observe_block(controller, rows)
    assert controller.lambda_value > raised


def test_controller_state_roundtrip_preserves_partial_window_and_gate() -> None:
    course = death_price_course(64)
    controller = AbsoluteDeathPriceController(course)
    _gate(controller)
    for _ in range(17):
        assert not controller.observe_episode(_outcome(act1=True, act1_revivals=5))
    restored = AbsoluteDeathPriceController(course)
    restored.load_state_dict(controller.state_dict())
    assert restored.as_mapping() == controller.as_mapping()


@pytest.mark.parametrize("budget", [64, 32, 16, 8, 4, 2, 0])
def test_each_course_preserves_discounted_outcome_first_margin(budget: int) -> None:
    discounted, required = validate_discounted_outcome_margin(death_price_course(budget))
    assert discounted >= required
