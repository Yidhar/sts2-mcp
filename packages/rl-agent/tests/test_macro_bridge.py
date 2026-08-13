"""Cross-domain bootstrap bridge (bridge doc §2/§3): transitions, learner,
authority, and router semantics for combat-domain encounter boundaries."""

from __future__ import annotations

from contextlib import nullcontext
from typing import Any

import numpy as np
import pytest
import torch

from sts2_rl.encoding import (
    EncodedDecisionSnapshot,
    GroundedEncodingConfig,
    GroundedObservationEncoder,
)
from sts2_rl.encoding.snapshot import sparse_token_table
from sts2_rl.macro import (
    MACRO_TRANSITION_CONTRACT_VERSION,
    JoinedCollectionAuthority,
    MacroCollectionAuthority,
    MacroEpisode,
    MacroQConfig,
    MacroQLearner,
    MacroSequenceReplay,
    MacroStep,
    n_step_targets,
)
from sts2_rl.semantics.clock import DECISION_CLOCK_BASE


def _snapshot(*, candidate_count: int = 4) -> EncodedDecisionSnapshot:
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


def _combat_step(
    *,
    action_index: int = 0,
    reward: float = 0.0,
    discount: float = 1.0,
    terminal: bool = False,
    recurrent_reset: bool = False,
    bridge_snapshot: EncodedDecisionSnapshot | None = None,
) -> MacroStep:
    return MacroStep(
        snapshot=_snapshot(),
        action_index=action_index,
        reward=reward,
        discount=discount,
        terminal=terminal,
        surface="combat",
        branch="play_card",
        control_domain="combat",
        recurrent_reset=recurrent_reset,
        bridge_snapshot=bridge_snapshot,
    )


# ---------------------------------------------------------------- transitions


def test_macro_step_bridge_validation_and_contract_version() -> None:
    assert MACRO_TRANSITION_CONTRACT_VERSION == "sts2-macro-transition-v5"
    bridge = _snapshot()
    with pytest.raises(ValueError, match="combat-domain"):
        MacroStep(
            snapshot=_snapshot(),
            action_index=0,
            reward=0.0,
            discount=1.0,
            terminal=False,
            surface="rest",
            branch="rest",
            control_domain="macro",
            bridge_snapshot=bridge,
        )
    with pytest.raises(ValueError, match="never terminal"):
        _combat_step(terminal=True, discount=0.0, bridge_snapshot=bridge)
    step = _combat_step(discount=0.5, bridge_snapshot=bridge)
    assert step.bridge_snapshot is bridge


def test_episode_may_end_at_a_domain_terminal_bridge_step() -> None:
    bridged = MacroEpisode(
        episode_id="bridged-tail",
        steps=(
            _combat_step(recurrent_reset=True),
            _combat_step(discount=0.5, bridge_snapshot=_snapshot()),
        ),
    )
    assert bridged.steps[-1].terminal is False
    with pytest.raises(ValueError, match="terminal or bridged"):
        MacroEpisode(episode_id="dangling", steps=(_combat_step(),))


def test_n_step_targets_forced_truncation_consumes_bridge_bootstrap() -> None:
    rewards = (1.0, 2.0, 4.0)
    discounts = (1.0, 0.9, 1.0)
    bootstraps = (0.0, 7.0, 5.0)
    forced = (False, True, False)
    targets = n_step_targets(
        rewards, discounts, bootstraps, n_step=3, forced_truncations=forced
    )
    # t=0 runs through the bridge step and stops there: 1 + 2 + 0.9*7.
    assert targets[0] == pytest.approx(9.3)
    # t=1 is the bridge step itself: reward + clock discount * bridge value;
    # the following transition's reward never leaks across the boundary.
    assert targets[1] == pytest.approx(2.0 + 0.9 * 7.0)
    # t=2 belongs to the next encounter and bootstraps normally.
    assert targets[2] == pytest.approx(4.0 + 5.0)
    unforced = n_step_targets(rewards, discounts, bootstraps, n_step=3)
    assert unforced[0] == pytest.approx(1.0 + 2.0 + 0.9 * 4.0 + 0.9 * 5.0)
    with pytest.raises(ValueError, match="misaligned"):
        n_step_targets(
            rewards, discounts, bootstraps, n_step=3, forced_truncations=(True,)
        )


# -------------------------------------------------------------------- learner


def _bridged_replay() -> tuple[MacroSequenceReplay, EncodedDecisionSnapshot]:
    bridge = _snapshot()
    replay = MacroSequenceReplay(
        capacity_episodes=2, window_length=4, seed=3, control_domain="combat"
    )
    replay.put(
        MacroEpisode(
            episode_id="two-encounters",
            steps=(
                _combat_step(action_index=0, recurrent_reset=True),
                _combat_step(action_index=1, reward=0.5, discount=0.5, bridge_snapshot=bridge),
                _combat_step(action_index=2, recurrent_reset=True),
                _combat_step(action_index=3, reward=1.0, discount=0.0, terminal=True),
            ),
        )
    )
    return replay, bridge


class _FakeQ:
    def __init__(self) -> None:
        self.online = torch.nn.Parameter(torch.zeros(4))
        self.target = torch.zeros(4)

    def forward_online(self, step: MacroStep, hidden: object) -> tuple[torch.Tensor, object]:
        return self.online[: len(step.snapshot.action_mask)], hidden

    def forward_target(self, step: MacroStep, hidden: object) -> tuple[torch.Tensor, object]:
        return self.target[: len(step.snapshot.action_mask)], hidden


def test_sequential_learner_trains_executed_q_toward_bridge_target() -> None:
    replay, bridge = _bridged_replay()
    fake = _FakeQ()
    seen_bridges: list[EncodedDecisionSnapshot] = []

    def bridge_value(snapshot: EncodedDecisionSnapshot) -> float:
        assert not torch.is_grad_enabled()
        seen_bridges.append(snapshot)
        return 10.0

    learner = MacroQLearner(
        online_parameters=[fake.online],
        forward_online=fake.forward_online,
        forward_target=fake.forward_target,
        sync_target=lambda: None,
        initial_state=lambda: None,
        target_evaluation_context=nullcontext,
        replay=replay,
        config=MacroQConfig(
            n_step=8,
            learning_rate=0.2,
            target_update_interval=10_000,
            sample_windows=1,
        ),
        bridge_value=bridge_value,
    )
    for _ in range(400):
        learner.update()
    # The bridge step's executed Q trains toward reward + Gamma * bridge value
    # (0.5 + 0.5*10), the step before it chains through the bridge and stops,
    # and the following encounter's steps stay at their own factual return —
    # untouched by the previous encounter's partner value.
    assert float(fake.online[0].item()) == pytest.approx(5.5, abs=0.1)
    assert float(fake.online[1].item()) == pytest.approx(5.5, abs=0.1)
    assert float(fake.online[2].item()) == pytest.approx(1.0, abs=0.1)
    assert float(fake.online[3].item()) == pytest.approx(1.0, abs=0.1)
    assert seen_bridges and all(snapshot is bridge for snapshot in seen_bridges)


def test_batched_learner_path_produces_the_same_bridge_targets() -> None:
    replay, bridge = _bridged_replay()
    online = torch.nn.Parameter(torch.zeros(4))

    def batch_online(snapshots: Any, hidden: Any) -> tuple[torch.Tensor, Any]:
        count = len(snapshots)
        state = online.new_zeros((count, 1)) if hidden is None else hidden
        assert isinstance(state, torch.Tensor)
        return online.unsqueeze(0).expand(count, -1), state + online[0] * 0.0 + 1.0

    def batch_target(snapshots: Any, hidden: Any) -> tuple[torch.Tensor, Any]:
        count = len(snapshots)
        state = torch.zeros(count, 1) if hidden is None else hidden
        assert isinstance(state, torch.Tensor)
        return torch.zeros(count, 4), state + 1.0

    def unused(step: object, hidden: object) -> tuple[torch.Tensor, object]:
        raise AssertionError("sequential path must not run when batch forwards exist")

    learner = MacroQLearner(
        online_parameters=[online],
        forward_online=unused,
        forward_target=unused,
        sync_target=lambda: None,
        initial_state=lambda: None,
        target_evaluation_context=nullcontext,
        replay=replay,
        config=MacroQConfig(n_step=8, sample_windows=1, target_update_interval=10_000),
        forward_online_batch=batch_online,
        forward_target_batch=batch_target,
        bridge_value=lambda snapshot: 10.0,
    )
    (executed_q, rewards, discounts, bootstraps, learn_steps) = (
        learner._batched_window_values(list(replay._windows()))[0]
    )
    adjusted, forced = learner._bridge_bootstraps(bootstraps, learn_steps)
    assert forced == (False, True, False, False)
    targets = n_step_targets(
        rewards, discounts, adjusted, n_step=8, forced_truncations=forced
    )
    assert targets == pytest.approx((5.5, 5.5, 1.0, 1.0))
    assert len(executed_q) == 4
    metrics = learner.update()
    assert metrics["steps_trained"] == 4


def test_learner_refuses_bridge_steps_without_a_partner() -> None:
    replay, _ = _bridged_replay()
    fake = _FakeQ()
    learner = MacroQLearner(
        online_parameters=[fake.online],
        forward_online=fake.forward_online,
        forward_target=fake.forward_target,
        sync_target=lambda: None,
        initial_state=lambda: None,
        target_evaluation_context=nullcontext,
        replay=replay,
        config=MacroQConfig(n_step=8, sample_windows=1),
    )
    with pytest.raises(RuntimeError, match="bridge"):
        learner.update()


# ------------------------------------------------------------------ authority

_COMBAT_ACTIONS = [{"kind": "end_turn", "model_action_kind": "end_turn"}]

_MAP_ACTIONS = [
    {
        "kind": "choose_map_node",
        "model_action_kind": "map",
        "map_node": {"row": 3, "col": 0},
    }
]


def _combat_observation(floor: int) -> dict[str, Any]:
    return {
        "phase": "combat",
        "combat": {"in_progress": True},
        "run": {"floor": floor},
    }


def _map_observation(floor: int) -> dict[str, Any]:
    return {
        "phase": "map",
        "combat": {"in_progress": False},
        "run": {"floor": floor},
    }


def _encode(observation: dict[str, Any], actions: list[dict[str, Any]]) -> Any:
    return GroundedObservationEncoder().encode(observation, actions).snapshot


def _combat_authority() -> MacroCollectionAuthority:
    return MacroCollectionAuthority(
        forward_q=lambda snapshot, hidden: (
            torch.zeros(snapshot.candidate_count),
            hidden,
        ),
        initial_state=lambda: None,
        epsilon=0.0,
        seed=7,
        control_domain="combat",
    )


def test_encounter_end_holds_the_open_transition_until_the_bridge_closes() -> None:
    authority = _combat_authority()
    authority.begin_episode("enc-bridge")
    observation = _combat_observation(5)
    assert (
        authority.choose(
            observation=observation,
            semantic_actions=_COMBAT_ACTIONS,
            snapshot=_encode(observation, _COMBAT_ACTIONS),
            valid=np.ones(1, dtype=np.bool_),
        )
        == 0
    )
    authority.observe_step(reward=0.25, floor=5, terminal=False)
    assert authority.has_pending_bridge is False
    authority.observe_observation(_map_observation(5))
    assert authority.has_pending_bridge is True
    # Boundary rewards between the last combat decision and the next macro
    # decision keep folding into the still-open transition.
    authority.observe_step(reward=0.05, floor=6, terminal=False)
    bridge = _encode(_map_observation(6), _MAP_ACTIONS)
    authority.close_encounter(bridge)
    assert authority.has_pending_bridge is False
    episode = authority.finish_episode()
    assert episode is not None and len(episode.steps) == 1
    step = episode.steps[0]
    assert step.bridge_snapshot is bridge
    assert step.terminal is False
    assert step.reward == pytest.approx(0.30)
    assert step.discount == pytest.approx(DECISION_CLOCK_BASE)  # one durable floor


def test_run_terminal_close_stays_terminal_without_a_bridge() -> None:
    authority = _combat_authority()
    authority.begin_episode("enc-terminal")
    observation = _combat_observation(5)
    assert (
        authority.choose(
            observation=observation,
            semantic_actions=_COMBAT_ACTIONS,
            snapshot=_encode(observation, _COMBAT_ACTIONS),
            valid=np.ones(1, dtype=np.bool_),
        )
        == 0
    )
    authority.observe_observation(_map_observation(5))
    assert authority.has_pending_bridge is True
    authority.observe_step(reward=0.1, floor=6, terminal=True)
    assert authority.has_pending_bridge is False
    episode = authority.finish_episode()
    assert episode is not None and len(episode.steps) == 1
    step = episode.steps[0]
    assert step.terminal is True
    assert step.discount == 0.0  # run terminal keeps bootstrap 0 via the clock
    assert step.bridge_snapshot is None


def test_missing_bridge_capture_marks_the_episode_replay_invalid() -> None:
    authority = _combat_authority()
    authority.begin_episode("enc-miss")
    first = _combat_observation(5)
    assert (
        authority.choose(
            observation=first,
            semantic_actions=_COMBAT_ACTIONS,
            snapshot=_encode(first, _COMBAT_ACTIONS),
            valid=np.ones(1, dtype=np.bool_),
        )
        == 0
    )
    authority.observe_observation(_map_observation(5))
    assert authority.has_pending_bridge is True
    # A new encounter starts without any macro decision snapshot having been
    # captured: fail-closed, the whole episode leaves combat replay.
    second = _combat_observation(6)
    assert (
        authority.choose(
            observation=second,
            semantic_actions=_COMBAT_ACTIONS,
            snapshot=_encode(second, _COMBAT_ACTIONS),
            valid=np.ones(1, dtype=np.bool_),
        )
        is None
    )
    metrics = authority.metrics()
    assert metrics["episode_replay_invalid"] is True
    assert metrics["bridge_misses"] == 1
    assert authority.finish_episode() is None


def test_close_encounter_contract_misuse_raises() -> None:
    combat = _combat_authority()
    combat.begin_episode("enc-misuse")
    with pytest.raises(RuntimeError, match="awaits a bridge"):
        combat.close_encounter(_snapshot())
    macro = MacroCollectionAuthority(
        forward_q=lambda snapshot, hidden: (
            torch.zeros(snapshot.candidate_count),
            hidden,
        ),
        initial_state=lambda: None,
        control_domain="macro",
    )
    with pytest.raises(RuntimeError, match="combat authority"):
        macro.close_encounter(_snapshot())


# --------------------------------------------------------------------- router


def test_router_passes_the_macro_decision_snapshot_into_the_pending_bridge() -> None:
    macro = MacroCollectionAuthority(
        forward_q=lambda snapshot, hidden: (
            torch.zeros(snapshot.candidate_count),
            hidden,
        ),
        initial_state=lambda: None,
        epsilon=0.0,
        control_domain="macro",
    )
    combat = _combat_authority()
    router = JoinedCollectionAuthority(macro=macro, combat=combat)
    router.begin_episode("bridge-route")

    combat_observation = _combat_observation(2)
    assert (
        router.choose(
            observation=combat_observation,
            semantic_actions=_COMBAT_ACTIONS,
            snapshot=_encode(combat_observation, _COMBAT_ACTIONS),
            valid=np.ones(1, dtype=np.bool_),
        )
        == 0
    )
    router.observe_step(reward=0.1, floor=3, terminal=False)

    macro_observation = _map_observation(3)
    macro_snapshot = _encode(macro_observation, _MAP_ACTIONS)
    assert (
        router.choose(
            observation=macro_observation,
            semantic_actions=_MAP_ACTIONS,
            snapshot=macro_snapshot,
            valid=np.ones(1, dtype=np.bool_),
        )
        == 0
    )
    assert combat.has_pending_bridge is False
    router.observe_step(reward=0.0, floor=3, terminal=True)
    episodes = router.finish_episodes("bridge-route")
    combat_episode = episodes["combat"]
    assert combat_episode is not None and len(combat_episode.steps) == 1
    step = combat_episode.steps[0]
    assert step.bridge_snapshot is macro_snapshot
    assert step.terminal is False
    assert step.reward == pytest.approx(0.1)
    assert step.discount == pytest.approx(DECISION_CLOCK_BASE)  # floor 2 -> 3
    macro_episode = episodes["macro"]
    assert macro_episode is not None
    assert macro_episode.steps[-1].terminal is True
