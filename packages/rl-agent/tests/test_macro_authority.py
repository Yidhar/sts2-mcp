from __future__ import annotations

from typing import Any

import numpy as np
import torch

from sts2_rl.macro.authority import MacroCollectionAuthority
from sts2_rl.semantics.clock import DECISION_CLOCK_BASE
from tests.test_macro_q_learner import _snapshot


def _authority(*, epsilon: float = 0.0, q: list[float] | None = None) -> MacroCollectionAuthority:
    values = torch.tensor(q or [0.0, 1.0, 0.0], dtype=torch.float32)

    def forward(snapshot: Any, hidden: Any) -> tuple[torch.Tensor, Any]:
        return values[: len(snapshot.action_mask)], hidden

    return MacroCollectionAuthority(
        forward_q=forward,
        initial_state=lambda: None,
        epsilon=epsilon,
        seed=11,
    )


def _rest_observation(floor: int = 7) -> dict[str, Any]:
    return {
        "phase": "rest",
        "combat": {"in_progress": False},
        "run": {"floor": floor},
        "player": {
            "deck": [
                {"id": "CARD.BASH", "is_upgradable": True, "is_removable": True},
            ]
        },
        "rest_site": {"options": []},
    }


_REST_ACTIONS = [
    {"kind": "choose_rest_option", "idx": 0, "option": {"type": "heal"}},
    {"kind": "choose_rest_option", "idx": 1, "option": {"type": "smith"}},
]

_PICKER_ACTIONS = [
    {
        "kind": "select_card",
        "model_action_kind": "card_selection",
        "selection_operation": "select",
        "card": {"id": "CARD.BASH", "instance_id": "CARD.BASH-1"},
    },
    {
        "kind": "confirm_selection",
        "model_action_kind": "card_selection",
        "selection_operation": "confirm",
    },
    {
        "kind": "cancel_selection",
        "model_action_kind": "card_selection",
        "selection_operation": "cancel_prompt",
    },
]


def test_smith_flow_decomposes_into_two_q_decisions_and_mechanical_confirm() -> None:
    authority = _authority(q=[0.0, 5.0, 0.0])  # greedy prefers smith entry
    authority.begin_episode("ep-smith")
    snapshot = _snapshot(candidate_count=2)
    valid = np.ones(2, dtype=np.bool_)

    entry = authority.choose(
        observation=_rest_observation(floor=7),
        semantic_actions=_REST_ACTIONS,
        snapshot=snapshot,
        valid=valid,
    )
    assert entry == 1  # smith entry chosen by Q

    picker_snapshot = _snapshot(candidate_count=3)
    picker_valid = np.ones(3, dtype=np.bool_)
    target = authority.choose(
        observation={"phase": "card_selection", "run": {"floor": 7}},
        semantic_actions=_PICKER_ACTIONS,
        snapshot=picker_snapshot,
        valid=picker_valid,
    )
    assert target == 0  # the matching select_card, never cancel/deselect

    confirm = authority.choose(
        observation={"phase": "card_selection", "run": {"floor": 7}},
        semantic_actions=_PICKER_ACTIONS,
        snapshot=picker_snapshot,
        valid=picker_valid,
    )
    assert confirm == 1  # mechanical confirm dispatch
    assert authority.mechanical_dispatches == 1

    authority.observe_step(reward=0.05, floor=8, terminal=False)
    authority.observe_step(reward=0.0, floor=8, terminal=True)
    episode = authority.finish_episode()
    assert episode is not None
    assert len(episode.steps) == 2  # entry decision + target decision
    entry_step, target_step = episode.steps
    assert entry_step.surface == "rest" and entry_step.branch == "smith"
    # Entry -> picker happens on the same floor: Gamma = 1.
    assert entry_step.discount == 1.0
    assert target_step.surface == "picker"
    assert target_step.terminal is True and target_step.discount == 0.0
    assert target_step.reward == 0.05


def test_non_macro_surfaces_are_declined_to_the_champion() -> None:
    authority = _authority()
    authority.begin_episode("ep-combat")
    combat_observation = {"phase": "combat", "combat": {"in_progress": True}}
    index = authority.choose(
        observation=combat_observation,
        semantic_actions=[{"kind": "play_card"}, {"kind": "end_turn"}],
        snapshot=_snapshot(candidate_count=2),
        valid=np.ones(2, dtype=np.bool_),
    )
    assert index is None
    assert authority.declined == 1
    assert authority.finish_episode() is None


def test_floor_advance_applies_the_durable_clock_between_macro_decisions() -> None:
    authority = _authority(q=[5.0, 0.0])
    authority.begin_episode("ep-clock")
    snapshot = _snapshot(candidate_count=2)
    valid = np.ones(2, dtype=np.bool_)
    first = authority.choose(
        observation=_rest_observation(floor=7),
        semantic_actions=_REST_ACTIONS,
        snapshot=snapshot,
        valid=valid,
    )
    assert first == 0  # heal
    authority.observe_step(reward=0.01, floor=8, terminal=False)
    second = authority.choose(
        observation=_rest_observation(floor=8),
        semantic_actions=_REST_ACTIONS,
        snapshot=snapshot,
        valid=valid,
    )
    assert second == 0
    authority.observe_step(reward=0.0, floor=8, terminal=True)
    episode = authority.finish_episode()
    assert episode is not None and len(episode.steps) == 2
    assert episode.steps[0].discount == DECISION_CLOCK_BASE  # one floor advanced
    assert episode.steps[0].reward == 0.01


def test_authority_drives_a_real_collector_episode_end_to_end() -> None:
    """Stage-two integration: the authority overrides macro decisions inside
    a real GroundedCollector episode on the synthetic rest-forge backend and
    yields a recorded MacroEpisode; collection stays legal throughout."""

    import torch as _torch

    from sts2_rl.training import build_training_resources
    from tests.test_v33_recovery_semantics import (
        _recovery_config,
        _RestForgeSelectionSuccessBackend,
    )

    config = _recovery_config(max_steps=8, repeat_threshold=8)
    resources = build_training_resources(
        config,
        backend=_RestForgeSelectionSuccessBackend(),
    )
    try:
        values = _torch.zeros(64)

        def forward(snapshot: Any, hidden: Any) -> tuple[_torch.Tensor, Any]:
            return values[: len(snapshot.action_mask)], hidden

        authority = MacroCollectionAuthority(
            forward_q=forward,
            initial_state=lambda: None,
            epsilon=1.0,  # pure branch-balanced exploration
            seed=5,
        )
        resources.collector.macro_authority = authority
        episode = resources.collector.collect_episode(
            epsilon=0.05,
            deterministic=False,
            record=True,
        )
        assert episode is not None
        macro_episode = authority.finish_episode("stage2-e2e")
        assert authority.overrides > 0
        assert macro_episode is not None
        assert macro_episode.episode_id == "stage2-e2e"
        assert all(step.discount >= 0.0 for step in macro_episode.steps)
        surfaces = {step.surface for step in macro_episode.steps}
        assert surfaces & {"rest", "picker"}
    finally:
        resources.close()


def test_stage2_runner_loop_with_real_model_on_synthetic_backend() -> None:
    """The full stage-two loop: real champion model frozen, real macro Q
    clone training via Double-Q, authority-driven collection on the
    synthetic backend — two episodes, two learner updates, no crash."""

    import copy as _copy
    import importlib.util as _ilu
    from pathlib import Path as _Path

    import torch as _torch

    from sts2_rl.macro import MacroQConfig, MacroQLearner, MacroSequenceReplay
    from sts2_rl.training import build_training_resources
    from tests.test_v33_recovery_semantics import (
        _RestForgeSelectionSuccessBackend,
    )
    from tests.test_v34_transaction_competence import _lifecycle_learning_config

    spec = _ilu.spec_from_file_location(
        "stage2_runner",
        _Path(__file__).resolve().parents[1] / "scripts" / "run_stage2_isolated_macro.py",
    )
    assert spec is not None and spec.loader is not None
    runner = _ilu.module_from_spec(spec)
    spec.loader.exec_module(runner)

    config = _lifecycle_learning_config(max_steps=8)
    resources = build_training_resources(
        config,
        backend=_RestForgeSelectionSuccessBackend(),
    )
    try:
        for parameter in resources.model.parameters():
            parameter.requires_grad_(False)
        macro_online = _copy.deepcopy(resources.model)
        for parameter in macro_online.parameters():
            parameter.requires_grad_(True)
        macro_target = _copy.deepcopy(macro_online)
        for parameter in macro_target.parameters():
            parameter.requires_grad_(False)
        device = resources.device
        forward_online = runner._forward_factory(
            macro_online, resources.encoder, device
        )
        forward_target = runner._forward_factory(
            macro_target, resources.encoder, device
        )
        replay = MacroSequenceReplay(capacity_episodes=8, burn_in=0, window_length=4)
        learner = MacroQLearner(
            online_parameters=[
                parameter
                for parameter in macro_online.parameters()
                if parameter.requires_grad
            ],
            forward_online=forward_online,
            forward_target=forward_target,
            sync_target=lambda: macro_target.load_state_dict(
                macro_online.state_dict()
            ),
            initial_state=lambda: None,
            replay=replay,
            config=MacroQConfig(sample_windows=2, target_update_interval=2),
        )
        authority = MacroCollectionAuthority(
            forward_q=forward_online,
            initial_state=lambda: None,
            epsilon=0.5,
            seed=9,
        )
        resources.collector.macro_authority = authority
        resources.collector.bind_failure_credit_run_id("stage2-runner-loop")
        stored = 0
        for episode_index in range(2):
            resources.collector.collect_episode(
                epsilon=0.0,
                deterministic=False,
                record=True,
            )
            macro_episode = authority.finish_episode(f"stage2-loop-{episode_index}")
            if macro_episode is not None:
                replay.put(macro_episode)
                stored += 1
        assert stored >= 1
        metrics = {}
        with _torch.autograd.set_detect_anomaly(False):
            for _ in range(2):
                metrics = learner.update()
        assert metrics["updates"] >= 1
        assert metrics["steps_trained"] >= 1
    finally:
        resources.close()
