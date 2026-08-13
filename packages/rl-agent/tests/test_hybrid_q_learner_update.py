from __future__ import annotations

import copy

import pytest

torch = pytest.importorskip("torch")

from sts2_rl.joint import (  # noqa: E402
    HybridQConfig,
    HybridQLearner,
    HybridRunReplay,
)
from tests.test_hybrid_replay import _run  # noqa: E402


class _TinyQ(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.bias = torch.nn.Parameter(torch.zeros(3))
        self.run_recurrent = torch.nn.GRUCell(1, 2)


def _forward(model: _TinyQ):
    def call(step, hidden):
        if hidden is None:
            hidden = torch.zeros((1, 2))
        signal = torch.ones((1, 1)) * (1.0 if step.control_domain == "macro" else -1.0)
        next_hidden = model.run_recurrent(signal, hidden)
        return model.bias + next_hidden.sum(), next_hidden

    return call


def test_balanced_update_gives_executed_q_a_direct_gradient() -> None:
    online = _TinyQ()
    target = copy.deepcopy(online)
    replay = HybridRunReplay(capacity_runs=4, macro_unroll=2, combat_unroll=2, seed=2)
    replay.put(_run())
    learner = HybridQLearner(
        online_model=online,
        target_model=target,
        replay=replay,
        forward_online=_forward(online),
        forward_target=_forward(target),
        initial_state=lambda: torch.zeros((1, 2)),
        config=HybridQConfig(
            run_batch_size=1,
            combat_n_step=2,
            target_update_interval=10,
            gradient_diagnostics_interval=1,
        ),
    )
    before = online.bias.detach().clone()
    metrics = learner.update()
    assert metrics["updates"] == 1
    assert metrics["macro_steps"] > 0
    assert metrics["combat_steps"] > 0
    assert not torch.equal(before, online.bias.detach())
    assert metrics["loss"] == pytest.approx(
        0.5 * (metrics["macro_loss"] + metrics["combat_loss"])
    )
    assert metrics["macro_shared_gradient_norm"] > 0.0
    assert metrics["combat_shared_gradient_norm"] > 0.0
    assert -1.0 <= metrics["shared_gradient_cosine"] <= 1.0


def test_macro_loss_reuses_successor_already_in_sparse_view(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    online = _TinyQ()
    target = copy.deepcopy(online)
    replay = HybridRunReplay(capacity_runs=4, macro_unroll=8, combat_unroll=2, seed=5)
    replay.put(_run())
    learner = HybridQLearner(
        online_model=online,
        target_model=target,
        replay=replay,
        forward_online=_forward(online),
        forward_target=_forward(target),
        initial_state=lambda: torch.zeros((1, 2)),
        config=HybridQConfig(run_batch_size=1),
    )
    pair = replay.sample_pairs(1)[0]

    def unexpected_prefix_replay(*_args, **_kwargs):
        raise AssertionError("in-view macro successor must not replay the full prefix")

    monkeypatch.setattr(learner, "_state_before", unexpected_prefix_replay)
    loss, _, count = learner._macro_loss(pair.macro)
    assert count == len(pair.macro.learn_indices)
    assert torch.isfinite(loss)


def test_explicit_target_sync_updates_weights_and_metric() -> None:
    online = _TinyQ()
    target = copy.deepcopy(online)
    replay = HybridRunReplay(capacity_runs=4, macro_unroll=2, combat_unroll=2)
    learner = HybridQLearner(
        online_model=online,
        target_model=target,
        replay=replay,
        forward_online=_forward(online),
        forward_target=_forward(target),
        initial_state=lambda: torch.zeros((1, 2)),
    )
    with torch.no_grad():
        online.bias.add_(1.0)
    assert not torch.equal(online.bias, target.bias)
    learner.sync_target()
    assert torch.equal(online.bias, target.bias)
    assert learner.metrics.target_syncs == 1
