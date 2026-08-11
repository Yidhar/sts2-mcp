from __future__ import annotations

from collections.abc import Iterator
from contextlib import contextmanager, nullcontext

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
    MacroWindow,
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
    control_domain: str = "macro",
    recurrent_reset: bool = False,
) -> MacroStep:
    return MacroStep(
        snapshot=_snapshot(),
        action_index=action_index,
        reward=reward,
        discount=discount,
        terminal=terminal,
        surface=surface,
        branch=branch,
        control_domain=control_domain,  # type: ignore[arg-type]
        recurrent_reset=recurrent_reset,
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
            control_domain="macro",
        )
    with pytest.raises(ValueError, match="must end with a terminal step"):
        MacroEpisode(episode_id="unfinished", steps=(_step(),))
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


def test_replay_covers_episode_from_step_zero_and_evicts_uniformly() -> None:
    replay = MacroSequenceReplay(capacity_episodes=2, window_length=4, seed=7)
    for index in range(3):
        replay.put(
            MacroEpisode(
                episode_id=f"ep-{index}",
                steps=tuple(
                    _step(
                        terminal=step_index == 5,
                        discount=0.0 if step_index == 5 else 1.0,
                    )
                    for step_index in range(6)
                ),
            )
        )
    assert len(replay) == 2  # oldest evicted
    windows = replay.sample(4)
    assert windows
    metrics = replay.metrics()
    assert metrics["episodes"] == 2
    assert metrics["sampling"] == "uniform"
    for episode in replay._episodes:
        episode_windows = [
            window for window in replay._windows() if window.episode is episode
        ]
        assert episode_windows[0].learn_slice[0] == 0
        covered = {
            index
            for window in episode_windows
            for index in range(*window.learn_slice)
        }
        assert covered == set(range(len(episode.steps)))
        for window in episode_windows[1:]:
            # Without a factual reset, exact-history mode still replays the
            # real episode prefix rather than fabricating a zero state.
            assert window.start == 0
            assert window.burn_in == window.learn_slice[0]
    with pytest.raises(ValueError, match="already stored"):
        replay.put(
            MacroEpisode(
                episode_id="ep-2",
                steps=(_step(terminal=True, discount=0.0),),
            )
        )


def test_long_combat_replay_starts_each_window_at_latest_recurrent_reset() -> None:
    replay = MacroSequenceReplay(
        capacity_episodes=1,
        window_length=16,
        seed=9,
        control_domain="combat",
    )
    reset_indices = (0, 63, 128, 207, 401, 520)
    step_count = 540
    replay.put(
        MacroEpisode(
            episode_id="long-combat",
            steps=tuple(
                _step(
                    reward=float((index % 7) + 1) / 10.0,
                    discount=0.0 if index == step_count - 1 else 1.0,
                    terminal=index == step_count - 1,
                    surface="combat",
                    branch="play_card",
                    control_domain="combat",
                    recurrent_reset=index in reset_indices,
                )
                for index in range(step_count)
            ),
        )
    )

    windows = replay._windows()
    covered = {
        index
        for window in windows
        for index in range(*window.learn_slice)
    }
    assert covered == set(range(step_count))
    for window in windows:
        learn_start, _ = window.learn_slice
        expected_start = max(
            (index for index in reset_indices if index <= learn_start),
            default=0,
        )
        assert window.start == expected_start
        assert window.burn_in == learn_start - expected_start

    last_window = windows[-1]
    assert last_window.learn_slice == (528, 540)
    assert last_window.start == 520
    assert last_window.burn_in == 8


def test_reset_aware_burn_in_is_exactly_equivalent_to_full_episode_prefix() -> None:
    replay = MacroSequenceReplay(
        capacity_episodes=1,
        window_length=16,
        seed=13,
        control_domain="combat",
    )
    step_count = 540
    reset_indices = {0, 63, 128, 207, 401, 520}
    replay.put(
        MacroEpisode(
            episode_id="reset-equivalence",
            steps=tuple(
                _step(
                    reward=float((index % 7) + 1) / 10.0,
                    discount=0.0 if index == step_count - 1 else 1.0,
                    terminal=index == step_count - 1,
                    surface="combat",
                    branch="play_card",
                    control_domain="combat",
                    recurrent_reset=index in reset_indices,
                )
                for index in range(step_count)
            ),
        )
    )
    reset_window = replay._windows()[-1]
    learn_start, _ = reset_window.learn_slice
    full_prefix_window = MacroWindow(
        episode=reset_window.episode,
        start=0,
        burn_in=learn_start,
        length=reset_window.length,
        window_id="full-prefix-reference",
    )
    parameter = torch.nn.Parameter(torch.tensor(0.25))

    def online(step: MacroStep, hidden: object) -> tuple[torch.Tensor, object]:
        state = parameter.new_zeros(()) if hidden is None else hidden
        assert isinstance(state, torch.Tensor)
        q_values = torch.stack((state + parameter, state - parameter, state * 0.5))
        return q_values, state + parameter * 0.1 + step.reward

    def target(step: MacroStep, hidden: object) -> tuple[torch.Tensor, object]:
        state = torch.tensor(0.0) if hidden is None else hidden
        assert isinstance(state, torch.Tensor)
        q_values = torch.stack((state + 0.5, state - 0.5, state * 0.5))
        return q_values, state + 0.025 + step.reward

    learner = MacroQLearner(
        online_parameters=[parameter],
        forward_online=online,
        forward_target=target,
        sync_target=lambda: None,
        initial_state=lambda: None,
        replay=replay,
        config=MacroQConfig(n_step=3, sample_windows=1),
    )

    full_values = learner._window_values(full_prefix_window)
    reset_values = learner._window_values(reset_window)
    torch.testing.assert_close(full_values[0], reset_values[0])
    assert full_values[1:4] == reset_values[1:4]
    assert full_values[4] == reset_values[4]


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
    replay = MacroSequenceReplay(capacity_episodes=8, window_length=2, seed=3)
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
        target_evaluation_context=nullcontext,
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

    replay = MacroSequenceReplay(capacity_episodes=4, window_length=2, seed=5)
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
        target_evaluation_context=nullcontext,
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


def test_exact_history_burn_in_and_learning_suffix_bptt() -> None:
    """A later window reconstructs the complete prefix under no-grad, then a
    later TD loss can train a recurrent write made earlier in its suffix."""

    replay = MacroSequenceReplay(capacity_episodes=2, window_length=2, seed=11)
    replay.put(
        MacroEpisode(
            episode_id="history",
            steps=(
                _step(reward=1.0),
                _step(reward=2.0),
                _step(reward=3.0),
                _step(reward=4.0, terminal=True, discount=0.0),
            ),
        )
    )
    window = next(window for window in replay._windows() if window.learn_slice == (2, 4))
    write = torch.nn.Parameter(torch.tensor(1.0))
    seen_online: list[tuple[float, bool]] = []
    target_context_active = False

    @contextmanager
    def target_evaluation() -> Iterator[None]:
        nonlocal target_context_active
        target_context_active = True
        try:
            yield
        finally:
            target_context_active = False

    def online(step: MacroStep, hidden: object) -> tuple[torch.Tensor, object]:
        seen_online.append((step.reward, torch.is_grad_enabled()))
        state = write.new_zeros(()) if hidden is None else hidden
        assert isinstance(state, torch.Tensor)
        q_values = torch.stack((state, state * 0.0, state * 0.0))
        return q_values, state + write * step.reward

    def target(step: MacroStep, hidden: object) -> tuple[torch.Tensor, object]:
        assert target_context_active
        assert not torch.is_grad_enabled()
        state = torch.tensor(0.0) if hidden is None else hidden
        assert isinstance(state, torch.Tensor)
        return torch.zeros(3), state + step.reward

    learner = MacroQLearner(
        online_parameters=[write],
        forward_online=online,
        forward_target=target,
        sync_target=lambda: None,
        initial_state=lambda: None,
        replay=replay,
        config=MacroQConfig(n_step=1, sample_windows=1),
        target_evaluation_context=target_evaluation,
    )
    executed_q, _, _, _, _ = learner._window_values(window)
    # The exact prefix is steps 0 and 1; only steps 2 and 3 are trainable.
    assert seen_online == [
        (1.0, False),
        (2.0, False),
        (3.0, True),
        (4.0, True),
    ]
    executed_q[-1].backward()
    # Q at step 3 depends on the write made while processing step 2. If the
    # callback or learner detached every step, this gradient would be zero.
    assert write.grad is not None
    assert float(write.grad.item()) == pytest.approx(3.0)


def test_learner_rejects_detached_state_inside_learning_suffix() -> None:
    replay = MacroSequenceReplay(capacity_episodes=1, window_length=2, seed=13)
    replay.put(
        MacroEpisode(
            episode_id="detached",
            steps=(_step(), _step(terminal=True, discount=0.0)),
        )
    )
    parameter = torch.nn.Parameter(torch.tensor(0.0))

    def detached_online(step: MacroStep, hidden: object) -> tuple[torch.Tensor, object]:
        return parameter.expand(3), torch.zeros(1)

    learner = MacroQLearner(
        online_parameters=[parameter],
        forward_online=detached_online,
        forward_target=lambda step, hidden: (torch.zeros(3), torch.zeros(1)),
        sync_target=lambda: None,
        initial_state=lambda: None,
        target_evaluation_context=nullcontext,
        replay=replay,
        config=MacroQConfig(n_step=1, sample_windows=1),
    )
    with pytest.raises(RuntimeError, match="detached the recurrent state"):
        learner._window_values(replay._windows()[0])


def test_replay_and_learner_state_round_trip() -> None:
    replay = MacroSequenceReplay(capacity_episodes=4, window_length=2, seed=17)
    for index in range(3):
        replay.put(
            MacroEpisode(
                episode_id=f"resume-{index}",
                steps=(_step(), _step(terminal=True, discount=0.0)),
            )
        )
    replay.sample(1)  # advance the replay RNG before publication
    replay_state = replay.state_dict()
    expected_next = tuple(window.window_id for window in replay.sample(2))
    restored_replay = MacroSequenceReplay(
        capacity_episodes=4, window_length=2, seed=999
    )
    restored_replay.load_state_dict(replay_state)
    assert tuple(window.window_id for window in restored_replay.sample(2)) == expected_next
    assert restored_replay.metrics() == replay.metrics()

    fake = _FakeQ()
    config = MacroQConfig(n_step=1, learning_rate=0.1, sample_windows=2)
    learner = MacroQLearner(
        online_parameters=[fake.online],
        forward_online=fake.forward_online,
        forward_target=fake.forward_target,
        sync_target=fake.sync,
        initial_state=lambda: None,
        target_evaluation_context=nullcontext,
        replay=replay,
        config=config,
    )
    learner.update()
    learner_state = learner.state_dict()
    restored_fake = _FakeQ()
    restored_learner = MacroQLearner(
        online_parameters=[restored_fake.online],
        forward_online=restored_fake.forward_online,
        forward_target=restored_fake.forward_target,
        sync_target=restored_fake.sync,
        initial_state=lambda: None,
        target_evaluation_context=nullcontext,
        replay=restored_replay,
        config=config,
    )
    restored_learner.load_state_dict(learner_state)
    assert restored_learner.metrics.as_mapping() == learner.metrics.as_mapping()
    assert restored_learner.optimizer.state_dict()["state"]


def test_replay_refuses_cross_domain_episodes() -> None:
    replay = MacroSequenceReplay(
        capacity_episodes=2,
        window_length=2,
        control_domain="combat",
    )
    with pytest.raises(ValueError, match="another domain"):
        replay.put(
            MacroEpisode(
                episode_id="macro",
                steps=(_step(terminal=True, discount=0.0),),
            )
        )
    replay.put(
        MacroEpisode(
            episode_id="combat",
            steps=(
                _step(
                    surface="combat",
                    branch="end_turn",
                    control_domain="combat",
                    recurrent_reset=True,
                    terminal=True,
                    discount=0.0,
                ),
            ),
        )
    )
    assert replay.metrics()["control_domain"] == "combat"


def test_recurrent_state_resets_at_each_combat_boundary() -> None:
    replay = MacroSequenceReplay(
        capacity_episodes=2,
        window_length=3,
        control_domain="combat",
    )
    replay.put(
        MacroEpisode(
            episode_id="two-combats",
            steps=(
                _step(control_domain="combat", recurrent_reset=True),
                _step(control_domain="combat"),
                _step(
                    control_domain="combat",
                    recurrent_reset=True,
                    terminal=True,
                    discount=0.0,
                ),
            ),
        )
    )
    parameter = torch.nn.Parameter(torch.tensor(1.0))
    seen: list[float] = []

    def online(step: MacroStep, hidden: object) -> tuple[torch.Tensor, object]:
        state = parameter * 0.0 if hidden is None else hidden
        assert isinstance(state, torch.Tensor)
        seen.append(float(state.detach().item()))
        return parameter.expand(3), state + 1.0

    def target(step: MacroStep, hidden: object) -> tuple[torch.Tensor, object]:
        state = torch.tensor(0.0) if hidden is None else hidden
        assert isinstance(state, torch.Tensor)
        return torch.zeros(3), state + 1.0

    learner = MacroQLearner(
        online_parameters=[parameter],
        forward_online=online,
        forward_target=target,
        sync_target=lambda: None,
        initial_state=lambda: None,
        replay=replay,
        config=MacroQConfig(n_step=1, sample_windows=1),
    )
    learner._window_values(replay._windows()[0])
    assert seen == [0.0, 1.0, 0.0]


def test_batched_recurrence_handles_partial_combat_resets() -> None:
    replay = MacroSequenceReplay(
        capacity_episodes=2,
        window_length=3,
        control_domain="combat",
    )
    replay.put(
        MacroEpisode(
            episode_id="reset-at-two",
            steps=(
                _step(control_domain="combat", recurrent_reset=True),
                _step(control_domain="combat"),
                _step(
                    control_domain="combat",
                    recurrent_reset=True,
                    terminal=True,
                    discount=0.0,
                ),
            ),
        )
    )
    replay.put(
        MacroEpisode(
            episode_id="one-combat",
            steps=(
                _step(control_domain="combat", recurrent_reset=True),
                _step(control_domain="combat"),
                _step(
                    control_domain="combat",
                    terminal=True,
                    discount=0.0,
                ),
            ),
        )
    )
    parameter = torch.nn.Parameter(torch.tensor(0.0))

    def batch_online(snapshots: object, hidden: object) -> tuple[torch.Tensor, object]:
        count = len(snapshots)  # type: ignore[arg-type]
        state = (
            parameter.new_zeros((count, 1))
            if hidden is None
            else hidden
        )
        assert isinstance(state, torch.Tensor)
        q = parameter.expand(count, 3)
        return q, state + parameter * 0.0 + 1.0

    def batch_target(snapshots: object, hidden: object) -> tuple[torch.Tensor, object]:
        count = len(snapshots)  # type: ignore[arg-type]
        state = torch.zeros(count, 1) if hidden is None else hidden
        assert isinstance(state, torch.Tensor)
        return torch.zeros(count, 3), state + 1.0

    learner = MacroQLearner(
        online_parameters=[parameter],
        forward_online=lambda step, hidden: (parameter.expand(3), hidden),
        forward_target=lambda step, hidden: (torch.zeros(3), hidden),
        sync_target=lambda: None,
        initial_state=lambda: None,
        replay=replay,
        config=MacroQConfig(n_step=1, sample_windows=2),
        forward_online_batch=batch_online,
        forward_target_batch=batch_target,
    )
    values = learner._batched_window_values(replay._windows())
    assert len(values) == 2
    assert all(len(result[-1]) == 3 for result in values)


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

    replay = MacroSequenceReplay(capacity_episodes=8, window_length=2, seed=7)
    for index in range(4):
        replay.put(
            MacroEpisode(
                episode_id=f"bep-{index}",
                steps=(
                    _step(action_index=0),
                    # Cut this synthetic transition's bootstrap. The fake Q
                    # below is intentionally state-independent, so allowing it
                    # to bootstrap from an unexecuted action at the successor
                    # would create an unsupported self-target unrelated to the
                    # lockstep batching behavior under test.
                    _step(action_index=1, reward=1.0, discount=0.0),
                    _step(action_index=0, terminal=True, discount=0.0),
                ),
            )
        )
    online = torch.nn.Parameter(torch.zeros(4))
    target = torch.zeros(4)

    def batch_forward_online(snapshots: object, hidden: object) -> tuple[torch.Tensor, object]:
        count = len(snapshots)  # type: ignore[arg-type]
        attached_hidden = online[0].reshape(1, 1).expand(count, -1) * 0.0
        return online[:3].unsqueeze(0).expand(count, -1), attached_hidden

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
        target_evaluation_context=nullcontext,
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
