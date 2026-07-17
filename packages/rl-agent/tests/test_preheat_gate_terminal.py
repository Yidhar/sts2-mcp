from __future__ import annotations

import pytest

from sts2_rl.contracts import EnvironmentResult, EnvironmentTransition
from sts2_rl.preheat_gate import _validated_full_run_result


def _result(*, terminated: bool, run_result: object, reason: str | None) -> EnvironmentResult:
    return EnvironmentResult(
        episode_id="episode",
        step_index=1,
        observation={},
        transition=EnvironmentTransition(
            episode_id="episode",
            step_index=1,
            before_state_version=0,
            after_state_version=1,
            facts={
                "combat_result": "none",
                "run_result": run_result,
                "terminal_reason": reason,
            },
        ),
        terminated=terminated,
        terminal_reason=reason,
    )


@pytest.mark.parametrize("outcome", ["victory", "defeat"])
def test_preheat_accepts_typed_canonical_run_terminal(outcome: str) -> None:
    reason = f"run_{outcome}"
    assert _validated_full_run_result(
        _result(terminated=True, run_result=outcome, reason=reason)
    ) == outcome


@pytest.mark.parametrize(
    ("run_result", "reason"),
    [
        ("none", None),
        (None, None),
        ("victory", "run_defeat"),
        ("defeat", "run_victory"),
    ],
)
def test_preheat_rejects_missing_or_inconsistent_run_terminal(
    run_result: object,
    reason: str | None,
) -> None:
    with pytest.raises(RuntimeError):
        _validated_full_run_result(
            _result(terminated=True, run_result=run_result, reason=reason)
        )


def test_preheat_rejects_nonterminal_final_run_result() -> None:
    with pytest.raises(RuntimeError, match="non-terminal"):
        _validated_full_run_result(
            _result(terminated=False, run_result="victory", reason=None)
        )
