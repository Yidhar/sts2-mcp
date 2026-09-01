from __future__ import annotations

import math

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from sts2_rl.joint import (  # noqa: E402
    HybridVTraceLearner,
    HybridVTraceLearnerConfig,
    HybridVTraceStep,
    HybridVTraceUnroll,
)
from tests.test_macro_authority import _snapshot  # noqa: E402


class _TinyActorCritic(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.logits = torch.nn.Parameter(torch.tensor([2.0, -2.0]))
        self.baseline = torch.nn.Parameter(torch.tensor(0.0))

    def call(self, snapshot, hidden):
        del snapshot
        return self.logits, self.baseline, hidden


def _unroll(*, reward: float = 0.0, action: int = 1) -> HybridVTraceUnroll:
    snapshot = _snapshot(candidate_count=2)
    steps = (
        HybridVTraceStep(
            snapshot=snapshot,
            action_index=action,
            behavior_log_probability=math.log(0.5),
            reward=reward,
            discount=1.0,
            terminal=False,
            surface="rest",
            branch="forge",
            control_domain="macro",
            policy_version=0,
        ),
        HybridVTraceStep(
            snapshot=snapshot,
            action_index=action,
            behavior_log_probability=math.log(0.5),
            reward=reward,
            discount=0.0,
            terminal=True,
            surface="combat",
            branch="play",
            control_domain="combat",
            policy_version=0,
        ),
    )
    return HybridVTraceUnroll(
        episode_id="tiny",
        start_decision=0,
        policy_version=0,
        initial_recurrent_state=np.zeros(1, dtype=np.float32),
        steps=steps,
        bootstrap_snapshot=None,
    )


def test_entropy_is_part_of_the_dense_policy_objective() -> None:
    model = _TinyActorCritic()
    learner = HybridVTraceLearner(
        model=model,
        forward=model.call,
        config=HybridVTraceLearnerConfig(
            learning_rate=0.05,
            entropy_weight=0.1,
            entropy_weight_end=0.1,
        ),
    )
    before = float((model.logits[0] - model.logits[1]).abs().item())
    metrics = learner.update((_unroll(),))
    after = float((model.logits[0] - model.logits[1]).abs().item())
    assert after < before
    assert metrics["macro_steps"] == 1
    assert metrics["combat_steps"] == 1
    assert metrics["entropy_weight"] == pytest.approx(0.1)


def test_explored_low_probability_action_keeps_a_policy_gradient() -> None:
    model = _TinyActorCritic()
    model.logits.data.copy_(torch.tensor([4.0, -4.0]))
    learner = HybridVTraceLearner(
        model=model,
        forward=model.call,
        config=HybridVTraceLearnerConfig(
            learning_rate=0.01,
            entropy_weight=1.0e-8,
            entropy_weight_end=1.0e-8,
        ),
    )
    before = model.logits.detach().clone()
    metrics = learner.update((_unroll(reward=1.0, action=1),))
    assert not torch.equal(model.logits.detach(), before)
    assert metrics["mean_importance_ratio"] > 0.0
    assert metrics["gradient_norm"] > 0.0
