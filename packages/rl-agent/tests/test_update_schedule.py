from __future__ import annotations

import pytest

from sts2_rl.training import advance_update_credit


def test_discard_policy_does_not_accrue_credit_before_replay_warmup() -> None:
    assert advance_update_credit(
        current_credit=900,
        collected_steps=100,
        replay_size_before=900,
        replay_size_after=1000,
        minimum_replay_size=1024,
        warmup_policy="discard",
    ) == 0


def test_discard_policy_only_credits_steps_beyond_crossed_boundary() -> None:
    assert advance_update_credit(
        current_credit=1000,
        collected_steps=50,
        replay_size_before=1000,
        replay_size_after=1050,
        minimum_replay_size=1024,
        warmup_policy="discard",
    ) == 26


def test_discard_policy_uses_normal_credit_accumulation_after_warmup() -> None:
    assert advance_update_credit(
        current_credit=3,
        collected_steps=41,
        replay_size_before=1024,
        replay_size_after=1065,
        minimum_replay_size=1024,
        warmup_policy="discard",
    ) == 44


def test_accrue_policy_preserves_legacy_warmup_debt() -> None:
    assert advance_update_credit(
        current_credit=900,
        collected_steps=124,
        replay_size_before=900,
        replay_size_after=1024,
        minimum_replay_size=1024,
        warmup_policy="accrue",
    ) == 1024


@pytest.mark.parametrize(
    ("override", "error"),
    (
        ({"collected_steps": -1}, "non-negative"),
        ({"replay_size_after": 9}, "cannot be smaller"),
        ({"minimum_replay_size": 0}, "must be positive"),
        ({"warmup_policy": "unknown"}, "discard.*accrue"),
    ),
)
def test_update_credit_inputs_fail_closed(
    override: dict[str, object],
    error: str,
) -> None:
    arguments: dict[str, object] = {
        "current_credit": 0,
        "collected_steps": 10,
        "replay_size_before": 10,
        "replay_size_after": 20,
        "minimum_replay_size": 100,
        "warmup_policy": "discard",
    }
    arguments.update(override)
    with pytest.raises((TypeError, ValueError), match=error):
        advance_update_credit(**arguments)  # type: ignore[arg-type]
