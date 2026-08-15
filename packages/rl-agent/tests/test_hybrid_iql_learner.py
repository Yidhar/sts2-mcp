from __future__ import annotations

import copy

import pytest

torch = pytest.importorskip("torch")

import sts2_rl.joint.iql_learner as iql_module  # noqa: E402
from sts2_rl.joint import (  # noqa: E402
    HybridIQLLearner,
    HybridIQLLearnerConfig,
    HybridRunReplay,
)
from tests.test_hybrid_replay import _run  # noqa: E402


class _TinyIQL(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.candidate_q_head = torch.nn.Linear(1, 3, bias=False)
        self.state_value_head = torch.nn.Linear(1, 1, bias=False)
        self.candidate_policy_head = torch.nn.Linear(1, 3, bias=False)
        self.run_recurrent = torch.nn.GRUCell(1, 2)


def _forward(model: _TinyIQL):
    def call(step, hidden):
        if hidden is None:
            hidden = torch.zeros((1, 2))
        signal = torch.ones((1, 1)) * (1.0 if step.control_domain == "macro" else -1.0)
        next_hidden = model.run_recurrent(signal, hidden)
        one = torch.ones((1, 1))
        q = model.candidate_q_head(one)[0]
        value = model.state_value_head(one)[0, 0]
        logits = model.candidate_policy_head(one)[0]
        return q, value, logits, next_hidden

    return call


def _learner(online: _TinyIQL, target: _TinyIQL) -> HybridIQLLearner:
    replay = HybridRunReplay(capacity_runs=4, macro_unroll=8, combat_unroll=8, seed=2)
    replay.put(_run())
    return HybridIQLLearner(
        online_model=online,
        target_model=target,
        replay=replay,
        forward_online=_forward(online),
        forward_target=_forward(target),
        initial_state=lambda: torch.zeros((1, 2)),
        config=HybridIQLLearnerConfig(
            run_batch_size=1,
            combat_n_step=2,
            target_update_interval=20,
            gradient_diagnostics_interval=1,
        ),
    )


def test_iql_q_target_does_not_max_over_unexecuted_successor_candidates() -> None:
    online_a = _TinyIQL()
    target_a = copy.deepcopy(online_a)
    learner_a = _learner(online_a, target_a)
    pair_a = learner_a.replay.sample_pairs(1)[0]
    q_loss_a = learner_a._macro_loss(pair_a.macro, lambda_value=0.0).q

    online_b = copy.deepcopy(online_a)
    target_b = copy.deepcopy(target_a)
    with torch.no_grad():
        target_b.candidate_q_head.weight[2, 0] = 10_000.0
    learner_b = _learner(online_b, target_b)
    pair_b = learner_b.replay.sample_pairs(1)[0]
    q_loss_b = learner_b._macro_loss(pair_b.macro, lambda_value=0.0).q

    torch.testing.assert_close(q_loss_a, q_loss_b)


def test_awac_recovers_an_executed_action_from_tiny_policy_probability() -> None:
    online = _TinyIQL()
    with torch.no_grad():
        online.candidate_q_head.weight[:, 0] = torch.tensor([1.0, 0.0, 0.0])
        online.state_value_head.weight.fill_(0.0)
        online.candidate_policy_head.weight[:, 0] = torch.tensor([-20.0, 0.0, 0.0])
    target = copy.deepcopy(online)
    learner = _learner(online, target)
    before = online.candidate_policy_head.weight.detach().clone()
    metrics = learner.update()
    after = online.candidate_policy_head.weight.detach()
    assert metrics["updates"] == 1
    assert after[0, 0] > before[0, 0]
    assert (after[0, 0] - after[1, 0]) > (before[0, 0] - before[1, 0])
    assert metrics["mean_advantage_weight"] > 1.0
    assert metrics["actor_gradient_norm"] > 0.0


def test_nonfinite_batch_stops_before_optimizer_step() -> None:
    online = _TinyIQL()
    target = copy.deepcopy(online)
    with torch.no_grad():
        target.state_value_head.weight.fill_(float("nan"))
    learner = _learner(online, target)
    before = copy.deepcopy(online.state_dict())
    with pytest.raises(FloatingPointError, match="non-finite"):
        learner.update()
    assert learner.metrics.updates == 0
    for key, value in online.state_dict().items():
        torch.testing.assert_close(value, before[key])


def test_transient_nonfinite_forward_retries_the_same_batch_once() -> None:
    online = _TinyIQL()
    target = copy.deepcopy(online)
    learner = _learner(online, target)
    stable_domain_losses = learner._domain_losses
    pair_objects: list[object] = []

    def flaky_domain_losses(pairs, *, price):
        pair_objects.append(pairs)
        if len(pair_objects) == 1:
            raise iql_module._NonFiniteForwardPass("injected transient")
        return stable_domain_losses(pairs, price=price)

    learner._domain_losses = flaky_domain_losses  # type: ignore[method-assign]
    metrics = learner.update()
    assert metrics["updates"] == 1
    assert metrics["forward_retries"] == 1
    assert pair_objects[0] is pair_objects[1]
