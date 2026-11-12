import io
from pathlib import Path

import pytest

from sts2_rl import macro_episode_evaluation as evaluation
from sts2_rl.macro_branch_diagnostic import BranchLimits
from sts2_rl.macro_root_diagnostic import ProbeBudgetExhausted


def _facts(*, terminal: bool, outcome: str | None = None) -> dict:
    return {"native_steps": 123, "state": {"observation": {
        "run": {"floor": 12, "act": 1}, "player": {"hp": 51, "max_hp": 91},
        "_training": {"revivals_used": 3, "player_hp_lost": 211}},
        "terminated": terminal, "truncated": False, "terminal_reason": "win" if terminal else None,
        "transition": {"facts": {"run_result": outcome}}},
        "reward_breakdown": {"reward": 0.125}, "native_discount": {"discount": 0.0 if terminal else 0.997}}


def test_full_return_counts_prefix_rewards_once() -> None:
    trace = evaluation.EpisodeTrace(io.StringIO(), BranchLimits().trace_limits())
    trace.record("transition", _facts(terminal=False))
    trace.record("transition", _facts(terminal=True, outcome="victory"))
    metrics = trace.metrics()
    assert metrics["observed_discounted_return"] == pytest.approx(0.125 * (1 + 0.997))
    assert metrics["revivals_used"] == 3
    assert metrics["hp"] == 51


@pytest.mark.parametrize("ending,outcome,expected", [
    ("factual_terminal", "victory", "complete"), ("factual_terminal", None, "failed"),
    ("environment_truncated", None, "censored"), ("budget", None, "censored"),
    ("error", None, "failed")])
def test_terminal_censoring_and_errors_are_separate(tmp_path: Path, monkeypatch, ending, outcome, expected) -> None:
    def run(**kwargs):
        kwargs["trace"].record("transition", _facts(terminal=ending == "factual_terminal", outcome=outcome))
        if ending == "budget":
            raise ProbeBudgetExhausted("max_native_steps")
        if ending == "error":
            raise ValueError("injected encoder error")
        return {"ending": ending}
    monkeypatch.setattr(evaluation, "run_branch_pass", run)
    result = evaluation.evaluate_episode(backend=None, policy=None, config=None, seed=101,
                                        output=tmp_path / "episode", limits=BranchLimits(terminal_only=True))
    assert result["status"] == expected
    assert result["revivals_used"] == 3
    assert (tmp_path / "episode/final-state.json").is_file()


@pytest.mark.parametrize("budget,revivals,verified", [(0, 0, True), (1, 0, False), (None, 0, False), (0, 1, False)])
def test_permadeath_requires_explicit_simulator_evidence(tmp_path, monkeypatch, budget, revivals, verified):
    from types import SimpleNamespace
    def run(**kwargs):
        assert kwargs["allow_zero_budget"] is True
        facts = _facts(terminal=True, outcome="victory")
        facts["state"]["observation"]["_training"].update(revival_budget=budget, revivals_used=revivals)
        kwargs["trace"].record("transition", facts)
        return {"ending": "factual_terminal"}
    monkeypatch.setattr(evaluation, "run_branch_pass", run)
    result = evaluation.evaluate_episode(backend=None, policy=None,
        config=SimpleNamespace(curriculum=SimpleNamespace(revival_budget=0)), seed=101,
        output=tmp_path / "episode", limits=BranchLimits(terminal_only=True))
    assert result["zero_revival_verified"] is verified
    assert result["status"] == ("complete" if verified else "failed")
