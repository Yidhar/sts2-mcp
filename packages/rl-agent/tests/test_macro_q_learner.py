from __future__ import annotations

import numpy as np
import pytest
import torch

from sts2_rl.encoding import EncodedDecisionSnapshot, GroundedEncodingConfig
from sts2_rl.encoding.snapshot import sparse_token_table
from sts2_rl.macro import (
    MacroEpisode,
    MacroQConfig,
    MacroQLearner,
    MacroSequenceReplay,
    MacroStep,
    n_step_targets,
    summarize_counts,
)


def _snapshot(*, candidate_count: int = 3) -> EncodedDecisionSnapshot:
    config = GroundedEncodingConfig(
        max_world_tokens=4,
        max_candidates=4,
        max_candidate_local_tokens=2,
    )
    feature_dim = config.feature_dim
    world = tuple([1.0] + [0.0] * (feature_dim - 1))
    candidate_features = tuple(
        tuple([0.0] * (index + 1) + [1.0] + [0.0] * (feature_dim - index - 2))
        for index in range(candidate_count)
    )
    candidate_ids = tuple(
        (2 + index, 2, 2, 3, 3, 3, 3, 2, 2, 4, 4, 4, 4) for index in range(candidate_count)
    )
    return EncodedDecisionSnapshot(
        config=config,
        encoding_fingerprint="0" * 64,
        world=sparse_token_table(
            features=(world,),
            ids=((2, 2, 2, 2, 2, 2, 2, 2, 0),),
            feature_dim=feature_dim,
            id_width=9,
        ),
        candidates=sparse_token_table(
            features=candidate_features,
            ids=candidate_ids,
            feature_dim=feature_dim,
            id_width=13,
        ),
        locals=sparse_token_table(
            features=(),
            ids=(),
            feature_dim=feature_dim,
            id_width=9,
        ),
        local_offsets=np.zeros(candidate_count + 1, dtype=np.uint32),
        action_mask=np.ones(candidate_count, dtype=np.bool_),
        domain_id=1,
    )


def _step(
    *,
    action_index: int = 0,
    reward: float = 0.0,
    discount: float = 1.0,
    terminal: bool = False,
    surface: str = "rest",
    branch: str = "rest",
) -> MacroStep:
    return MacroStep(
        snapshot=_snapshot(),
        action_index=action_index,
        reward=reward,
        discount=discount,
        terminal=terminal,
        surface=surface,
        branch=branch,
    )


def test_transition_contract_rejects_illegal_and_terminal_bootstrap() -> None:
    with pytest.raises(ValueError, match="outside its candidate set"):
        _step(action_index=7)
    with pytest.raises(ValueError, match="never bootstrap"):
        MacroStep(
            snapshot=_snapshot(),
            action_index=0,
            reward=0.0,
            discount=0.5,
            terminal=True,
            surface="rest",
            branch="rest",
        )
    episode = MacroEpisode(
        episode_id="ep-1",
        steps=(_step(branch="smith"), _step(terminal=True, discount=0.0)),
    )
    counts = summarize_counts((episode,))
    assert counts["executed_counts"] == {"rest:rest": 1, "rest:smith": 1}


def test_n_step_targets_respect_clock_and_terminal_cut() -> None:
    rewards = (1.0, 2.0, 4.0)
    discounts = (0.5, 0.0, 1.0)
    bootstraps = (10.0, 20.0, 30.0)
    targets = n_step_targets(rewards, discounts, bootstraps, n_step=3)
    # t=0: 1.0 + 0.5*2.0, then discount hits 0.0 at t=1 -> terminal cut.
    assert targets[0] == pytest.approx(2.0)
    # t=1: terminal transition -> reward only.
    assert targets[1] == pytest.approx(2.0)
    # t=2: last transition bootstraps through its own discount.
    assert targets[2] == pytest.approx(4.0 + 30.0)


def test_replay_bounds_priorities_and_eviction() -> None:
    replay = MacroSequenceReplay(capacity_episodes=2, burn_in=1, window_length=4, seed=7)
    for index in range(3):
        replay.put(
            MacroEpisode(
                episode_id=f"ep-{index}",
                steps=tuple(_step() for _ in range(6)),
            )
        )
    assert len(replay) == 2  # oldest evicted
    windows = replay.sample(4)
    assert windows
    replay.update_priority(windows[0].window_id, 3.0)
    metrics = replay.metrics()
    assert metrics["episodes"] == 2
    assert metrics["prioritized_windows"] == 1
    with pytest.raises(ValueError, match="already stored"):
        replay.put(MacroEpisode(episode_id="ep-2", steps=(_step(),)))


class _FakeQ:
    """Minimal candidate-Q pair: online trainable, target snapshot."""

    def __init__(self) -> None:
        self.online = torch.nn.Parameter(torch.zeros(4))
        self.target = torch.zeros(4)

    def forward_online(self, step: MacroStep, hidden: object) -> tuple[torch.Tensor, object]:
        count = len(step.snapshot.action_mask)
        return self.online[:count], hidden

    def forward_target(self, step: MacroStep, hidden: object) -> tuple[torch.Tensor, object]:
        count = len(step.snapshot.action_mask)
        return self.target[:count], hidden

    def sync(self) -> None:
        self.target = self.online.detach().clone()


def test_double_q_learner_converges_and_reports_ec4_counts() -> None:
    replay = MacroSequenceReplay(capacity_episodes=8, burn_in=0, window_length=2, seed=3)
    # One-decision episodes: action 1 pays +1 and terminates.
    for index in range(4):
        replay.put(
            MacroEpisode(
                episode_id=f"ep-{index}",
                steps=(
                    _step(
                        action_index=1,
                        reward=1.0,
                        discount=0.0,
                        terminal=True,
                        surface="reward",
                        branch="skip",
                    ),
                ),
            )
        )
    fake = _FakeQ()
    learner = MacroQLearner(
        online_parameters=[fake.online],
        forward_online=fake.forward_online,
        forward_target=fake.forward_target,
        sync_target=fake.sync,
        initial_state=lambda: None,
        replay=replay,
        config=MacroQConfig(
            n_step=3,
            learning_rate=0.2,
            target_update_interval=5,
            sample_windows=2,
        ),
    )
    for _ in range(60):
        metrics = learner.update()
    assert float(fake.online[1].item()) == pytest.approx(1.0, abs=0.05)
    # The executed action got its gradient even though it was never greedy
    # at initialization — the anti-absorption property, by construction.
    assert metrics["executed_counts"]["reward:skip"] > 0
    assert metrics["target_syncs"] >= 1
    assert metrics["loss"] < 0.1


def test_window_tail_bootstraps_from_the_successor_beyond_the_window() -> None:
    """A window that ends mid-episode must bootstrap its final transition
    from the actual successor state's Double-Q value, not a biased zero."""

    replay = MacroSequenceReplay(capacity_episodes=4, burn_in=0, window_length=2, seed=5)
    # Three-step episode, all zero reward, unit discount: with the target
    # network fixed at Q(candidate 1) = 1, every learn step's exact n-step
    # target is 1.0 ONLY if the final window transition sees its successor.
    replay.put(
        MacroEpisode(
            episode_id="ep-tail",
            steps=(
                _step(action_index=0),
                _step(action_index=0),
                _step(action_index=0, terminal=True, discount=0.0),
            ),
        )
    )
    values = torch.tensor([0.0, 1.0, 0.0])

    def forward(step: object, hidden: object) -> tuple[torch.Tensor, object]:
        return values.clone().requires_grad_(True), hidden

    dummy = torch.nn.Parameter(torch.zeros(1))
    learner = MacroQLearner(
        online_parameters=[dummy],
        forward_online=forward,
        forward_target=forward,
        sync_target=lambda: None,
        initial_state=lambda: None,
        replay=replay,
        config=MacroQConfig(n_step=1, learning_rate=0.01, sample_windows=8),
    )
    # First window covers steps [0, 2): its tail (step 1) is non-terminal
    # and its successor (step 2) lives beyond the window.
    windows = [w for w in replay._windows() if w.learn_slice == (0, 2)]
    assert windows, "expected a window ending mid-episode"
    _, rewards, discounts, bootstraps, learn_steps = learner._window_values(windows[0])
    assert len(learn_steps) == 2
    assert rewards == (0.0, 0.0) and discounts == (1.0, 1.0)
    # Both transitions bootstrap from a successor whose masked argmax is
    # candidate 1 with target value 1.0 — including the window tail.
    assert bootstraps == (1.0, 1.0)


def test_trunk_loading_inherits_compatible_and_refuses_drift() -> None:
    """load_trunk_state inherits shape-compatible tensors, restarts tolerated
    head groups on shape change, and refuses trunk drift."""

    from sts2_rl.macro import load_trunk_state

    trunk = torch.nn.Linear(4, 4)
    head_old = torch.nn.Linear(4, 1)
    head_new = torch.nn.Linear(6, 1)

    class _Old(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.trunk = trunk
            self.transaction_q_head = head_old

    class _New(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.trunk = torch.nn.Linear(4, 4)
            self.transaction_q_head = head_new

    old_state = _Old().state_dict()
    new_model = _New()
    report = load_trunk_state(new_model, dict(old_state))
    assert any("transaction_q_head" in key for key in report["dropped"])
    assert torch.equal(new_model.trunk.weight, trunk.weight)

    class _Drifted(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.trunk = torch.nn.Linear(5, 5)  # trunk shape drift: refuse
            self.transaction_q_head = torch.nn.Linear(6, 1)

    with pytest.raises(RuntimeError, match="drift beyond tolerated"):
        load_trunk_state(_Drifted(), dict(old_state))


def test_batched_lockstep_path_matches_semantics_and_converges() -> None:
    """The lockstep-batched update trains the same objective: executed
    action converges to its factual return, window tails bootstrap from
    successors, and burn-in rows produce no learn targets."""

    replay = MacroSequenceReplay(capacity_episodes=8, burn_in=1, window_length=2, seed=7)
    for index in range(4):
        replay.put(
            MacroEpisode(
                episode_id=f"bep-{index}",
                steps=(
                    _step(action_index=0),
                    _step(action_index=1, reward=1.0),
                    _step(action_index=0, terminal=True, discount=0.0),
                ),
            )
        )
    online = torch.nn.Parameter(torch.zeros(4))
    target = torch.zeros(4)

    def batch_forward_online(snapshots: object, hidden: object) -> tuple[torch.Tensor, object]:
        count = len(snapshots)  # type: ignore[arg-type]
        return online[:3].unsqueeze(0).expand(count, -1), torch.zeros(count, 1)

    def batch_forward_target(snapshots: object, hidden: object) -> tuple[torch.Tensor, object]:
        count = len(snapshots)  # type: ignore[arg-type]
        return target[:3].unsqueeze(0).expand(count, -1), torch.zeros(count, 1)

    def unused(step: object, hidden: object) -> tuple[torch.Tensor, object]:
        raise AssertionError("sequential path must not run when batch forwards exist")

    def sync() -> None:
        nonlocal target
        target = online.detach().clone()

    learner = MacroQLearner(
        online_parameters=[online],
        forward_online=unused,
        forward_target=unused,
        sync_target=sync,
        initial_state=lambda: None,
        replay=replay,
        config=MacroQConfig(
            n_step=2, learning_rate=0.2, target_update_interval=5, sample_windows=4
        ),
        forward_online_batch=batch_forward_online,
        forward_target_batch=batch_forward_target,
    )
    metrics = {}
    for _ in range(80):
        metrics = learner.update()
    # Action 1 pays +1 at the middle step; with terminal cut afterwards its
    # Q must converge to 1 regardless of initialization.
    assert float(online[1].item()) == pytest.approx(1.0, abs=0.08)
    assert metrics["steps_trained"] > 0
