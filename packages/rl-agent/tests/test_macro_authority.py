from __future__ import annotations

from typing import Any

import numpy as np
import torch

from sts2_env._sim_translate_actions import _translate_legal_actions
from sts2_env._sim_translate_entities import _translate_card
from sts2_rl.encoding import GroundedObservationEncoder
from sts2_rl.macro.authority import MacroCollectionAuthority
from sts2_rl.semantics.clock import DECISION_CLOCK_BASE


def _snapshot(*, candidate_count: int = 3) -> Any:
    return GroundedObservationEncoder().encode(
        {"phase": "combat", "combat": {"in_progress": True}},
        [
            {
                "kind": "end_turn",
                "model_action_kind": "end_turn",
                "action_handle": f"test-{index}",
            }
            for index in range(candidate_count)
        ],
    ).snapshot


def _authority(
    *,
    epsilon: float = 0.0,
    q: list[float] | None = None,
    control_domain: str = "macro",
    forward_calls: list[int] | None = None,
) -> MacroCollectionAuthority:
    values = torch.tensor(q or [0.0, 1.0, 0.0], dtype=torch.float32)

    def forward(snapshot: Any, hidden: Any) -> tuple[torch.Tensor, Any]:
        if forward_calls is not None:
            forward_calls.append(len(snapshot.action_mask))
        return values[: len(snapshot.action_mask)], hidden

    return MacroCollectionAuthority(
        forward_q=forward,
        initial_state=lambda: None,
        epsilon=epsilon,
        seed=11,
        control_domain=control_domain,  # type: ignore[arg-type]
    )


def _rest_observation(
    floor: int = 7,
    *,
    deck: list[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    return {
        "phase": "rest",
        "combat": {"in_progress": False},
        "run": {"floor": floor},
        "player": {
            "deck": deck
            or [
                {
                    "id": "CARD.BASH",
                    "instance_id": "CARD.BASH-1",
                    "index": 0,
                    "is_upgradable": True,
                    "is_removable": True,
                },
            ]
        },
        "rest_site": {"options": []},
    }


_REST_ACTIONS = [
    {
        "kind": "choose_rest_option",
        "model_action_kind": "rest_site",
        "model_action_variant": "rest",
        "idx": 0,
        "option": {"type": "heal"},
    },
    {
        "kind": "choose_rest_option",
        "model_action_kind": "rest_site",
        "model_action_variant": "forge",
        "idx": 1,
        "option": {"type": "smith"},
    },
]

_PICKER_ACTIONS = [
    {
        "kind": "select_card",
        "model_action_kind": "card_selection",
        "selection_operation": "select",
        "card": {
            "id": "CARD.BASH",
            "instance_id": "CARD.BASH-1",
            "index": 0,
            "is_upgradable": True,
            "is_removable": True,
        },
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


def test_smith_is_one_semantic_q_decision_with_mechanical_suffix() -> None:
    forward_calls: list[int] = []
    authority = _authority(q=[0.0, 5.0, 0.0], forward_calls=forward_calls)
    authority.begin_episode("ep-smith")
    snapshot = _snapshot(candidate_count=2)
    valid = np.ones(2, dtype=np.bool_)

    entry = authority.choose(
        observation=_rest_observation(floor=7),
        semantic_actions=_REST_ACTIONS,
        snapshot=snapshot,
        valid=valid,
    )
    assert entry == 1  # native forge entry chosen by semantic Q

    picker_snapshot = _snapshot(candidate_count=3)
    picker_valid = np.ones(3, dtype=np.bool_)
    picker_groups = GroundedObservationEncoder().semantic_action_groups(_PICKER_ACTIONS)
    picker_surface = [{"prototype": group.prototype} for group in picker_groups]
    target = authority.choose(
        observation={"phase": "card_selection", "run": {"floor": 7}},
        semantic_actions=picker_surface,
        snapshot=picker_snapshot,
        valid=picker_valid,
    )
    assert target == 0  # matching target, dispatched mechanically

    confirm = authority.choose(
        observation={"phase": "card_selection", "run": {"floor": 7}},
        semantic_actions=_PICKER_ACTIONS,
        snapshot=picker_snapshot,
        valid=picker_valid,
    )
    assert confirm == 1  # mechanical confirm dispatch
    assert authority.mechanical_dispatches == 2
    assert forward_calls == [2]  # picker and confirm never run Q

    authority.observe_step(reward=0.05, floor=8, terminal=False)
    authority.observe_step(reward=0.0, floor=8, terminal=True)
    episode = authority.finish_episode()
    assert episode is not None
    assert len(episode.steps) == 1
    step = episode.steps[0]
    assert step.surface == "rest" and step.branch == "smith"
    assert step.terminal is True and step.discount == 0.0
    assert step.reward == 0.05


def test_real_nested_upgrade_preview_smith_target_executes_atomically() -> None:
    native_card = _native_card_with_upgrade_preview()
    deck_card = _translate_card(native_card, pile="Deck")
    picker_actions = _translated_picker_actions(native_card)
    picker_surface = _grouped_surface(picker_actions)
    authority = _authority(q=[0.0, 5.0, 0.0])
    authority.begin_episode("ep-real-preview-smith")

    entry = authority.choose(
        observation=_rest_observation(deck=[deck_card]),
        semantic_actions=_REST_ACTIONS,
        snapshot=_snapshot(candidate_count=2),
        valid=np.ones(2, dtype=np.bool_),
    )
    assert entry == 1
    assert authority.choose(
        observation={"phase": "card_selection", "run": {"floor": 7}},
        semantic_actions=picker_surface,
        snapshot=_snapshot(candidate_count=len(picker_surface)),
        valid=np.ones(len(picker_surface), dtype=np.bool_),
    ) == 0
    assert authority.choose(
        observation={"phase": "card_selection", "run": {"floor": 7}},
        semantic_actions=picker_surface,
        snapshot=_snapshot(candidate_count=len(picker_surface)),
        valid=np.ones(len(picker_surface), dtype=np.bool_),
    ) == 1

    authority.observe_step(reward=0.1, floor=7, terminal=True)
    episode = authority.finish_episode()
    assert episode is not None and len(episode.steps) == 1
    assert episode.steps[0].branch == "smith"
    assert authority.mechanical_dispatches == 2


def test_real_nested_upgrade_preview_remove_target_executes_atomically() -> None:
    native_card = _native_card_with_upgrade_preview()
    deck_card = _translate_card(native_card, pile="Deck")
    picker_actions = _translated_picker_actions(native_card)
    picker_surface = _grouped_surface(picker_actions)
    shop_actions = [
        {
            "kind": "shop_purchase",
            "model_action_kind": "shop",
            "model_action_variant": "purchase",
            "idx": 0,
            "item": {
                "category": "card_removal",
                "cost": 75,
                "can_afford": True,
            },
        },
        {
            "kind": "proceed",
            "model_action_kind": "shop",
            "model_action_variant": "leave",
            "idx": 1,
        },
    ]
    observation = {
        "phase": "shop",
        "combat": {"in_progress": False},
        "run": {"floor": 8},
        "player": {"deck": [deck_card]},
        "shop": {"is_open": True},
    }
    authority = _authority(q=[5.0, 0.0, 0.0])
    authority.begin_episode("ep-real-preview-remove")

    assert authority.choose(
        observation=observation,
        semantic_actions=shop_actions,
        snapshot=_snapshot(candidate_count=2),
        valid=np.ones(2, dtype=np.bool_),
    ) == 0
    assert authority.choose(
        observation={"phase": "card_selection", "run": {"floor": 8}},
        semantic_actions=picker_surface,
        snapshot=_snapshot(candidate_count=len(picker_surface)),
        valid=np.ones(len(picker_surface), dtype=np.bool_),
    ) == 0
    assert authority.choose(
        observation={"phase": "card_selection", "run": {"floor": 8}},
        semantic_actions=picker_surface,
        snapshot=_snapshot(candidate_count=len(picker_surface)),
        valid=np.ones(len(picker_surface), dtype=np.bool_),
    ) == 1

    authority.observe_step(reward=0.1, floor=8, terminal=True)
    episode = authority.finish_episode()
    assert episode is not None and len(episode.steps) == 1
    assert episode.steps[0].branch == "remove"
    assert authority.mechanical_dispatches == 2


def test_semantic_snapshot_scores_smith_targets_and_replay_uses_that_index() -> None:
    deck = [
        {
            "id": "CARD.BASH",
            "instance_id": "bash-copy",
            "index": 0,
            "is_upgradable": True,
            "is_removable": True,
        },
        {
            "id": "CARD.ANGER",
            "instance_id": "anger-copy",
            "index": 1,
            "is_upgradable": True,
            "is_removable": True,
        },
    ]
    authority = _authority(q=[0.0, 1.0, 9.0])
    authority.begin_episode("ep-target-q")
    native = authority.choose(
        observation=_rest_observation(deck=deck),
        semantic_actions=_REST_ACTIONS,
        snapshot=_snapshot(candidate_count=2),
        valid=np.ones(2, dtype=np.bool_),
    )
    # Both semantic targets enter native action 1; Q selected Anger at semantic
    # index 2 rather than aliasing both cards to the parent action index.
    assert native == 1
    picker = [
        {
            "kind": "select_card",
            "model_action_kind": "card_selection",
            "model_action_variant": "select",
            "selection_operation": "select",
            "card": deck[0],
        },
        {
            "kind": "select_card",
            "model_action_kind": "card_selection",
            "model_action_variant": "select",
            "selection_operation": "select",
            "card": deck[1],
        },
        {
            "kind": "confirm_selection",
            "model_action_kind": "card_selection",
            "model_action_variant": "confirm",
        },
    ]
    assert (
        authority.choose(
            observation={"phase": "card_selection", "run": {"floor": 7}},
            semantic_actions=picker,
            snapshot=_snapshot(candidate_count=3),
            valid=np.ones(3, dtype=np.bool_),
        )
        == 1
    )
    authority.observe_step(reward=0.0, floor=7, terminal=True)
    episode = authority.finish_episode()
    assert episode is not None and len(episode.steps) == 1
    step = episode.steps[0]
    assert len(step.snapshot.action_mask) == 3
    assert step.snapshot.action_mask.tolist() == [True, True, True]
    assert step.action_index == 2


def test_picker_target_mismatch_never_selects_an_arbitrary_copy() -> None:
    authority = _authority(q=[0.0, 5.0])
    authority.begin_episode("ep-target-mismatch")
    assert (
        authority.choose(
            observation=_rest_observation(),
            semantic_actions=_REST_ACTIONS,
            snapshot=_snapshot(candidate_count=2),
            valid=np.ones(2, dtype=np.bool_),
        )
        == 1
    )
    wrong_copy = [
        {
            "kind": "select_card",
            "model_action_kind": "card_selection",
            "model_action_variant": "select",
            "selection_operation": "select",
            "card": {
                "id": "CARD.ANGER",
                "instance_id": "CARD.ANGER-OTHER",
                "index": 9,
                "is_upgradable": True,
                "is_removable": True,
            },
        }
    ]
    assert (
        authority.choose(
            observation={"phase": "card_selection", "run": {"floor": 7}},
            semantic_actions=wrong_copy,
            snapshot=_snapshot(candidate_count=1),
            valid=np.ones(1, dtype=np.bool_),
        )
        is None
    )
    assert authority.finish_episode() is None


def test_composite_executor_mismatch_invalidates_the_whole_replay_episode() -> None:
    forward_calls = 0

    def forward(snapshot: Any, hidden: Any) -> tuple[torch.Tensor, Any]:
        nonlocal forward_calls
        forward_calls += 1
        values = [5.0, 0.0] if forward_calls == 1 else [0.0, 5.0]
        return torch.tensor(values[: len(snapshot.action_mask)]), hidden

    authority = MacroCollectionAuthority(
        forward_q=forward,
        initial_state=lambda: None,
        epsilon=0.0,
        seed=4,
    )
    authority.begin_episode("ep-invalid-composite")
    assert authority.choose(
        observation=_rest_observation(floor=6),
        semantic_actions=_REST_ACTIONS,
        snapshot=_snapshot(candidate_count=2),
        valid=np.ones(2, dtype=np.bool_),
    ) == 0
    authority.observe_step(reward=0.2, floor=7, terminal=False)
    assert authority.choose(
        observation=_rest_observation(floor=7),
        semantic_actions=_REST_ACTIONS,
        snapshot=_snapshot(candidate_count=2),
        valid=np.ones(2, dtype=np.bool_),
    ) == 1

    wrong_picker = [
        {
            "kind": "select_card",
            "model_action_kind": "card_selection",
            "model_action_variant": "select",
            "selection_operation": "select",
            "card": {
                "id": "CARD.ANGER",
                "is_upgradable": True,
                "is_removable": True,
            },
        }
    ]
    assert authority.choose(
        observation={"phase": "card_selection", "run": {"floor": 7}},
        semantic_actions=wrong_picker,
        snapshot=_snapshot(candidate_count=1),
        valid=np.ones(1, dtype=np.bool_),
    ) is None
    # The earlier valid Rest step cannot be replayed after the behavior hidden
    # state consumed a Smith action whose mechanical suffix never completed.
    assert authority.metrics()["episode_replay_invalid"] is True
    assert authority.metrics()["executor_failures"] == 1
    assert authority.metrics()["recorded_steps"] == 0
    assert authority.choose(
        observation=_rest_observation(floor=8),
        semantic_actions=_REST_ACTIONS,
        snapshot=_snapshot(candidate_count=2),
        valid=np.ones(2, dtype=np.bool_),
    ) is None
    assert forward_calls == 2
    authority.observe_step(reward=1.0, floor=8, terminal=True)
    assert authority.finish_episode() is None


def test_picker_uses_stable_member_of_one_strict_equal_action_group() -> None:
    target = {
        "id": "CARD.BASH",
        "cost": 2,
        "is_upgradable": True,
        "is_removable": True,
    }
    authority = _authority(q=[0.0, 5.0])
    authority.begin_episode("ep-equal-picker-copies")
    assert (
        authority.choose(
            observation=_rest_observation(deck=[dict(target), dict(target)]),
            semantic_actions=_REST_ACTIONS,
            snapshot=_snapshot(candidate_count=2),
            valid=np.ones(2, dtype=np.bool_),
        )
        == 1
    )
    equal_picker_actions = [
        {
            "kind": "select_card",
            "model_action_kind": "card_selection",
            "model_action_variant": "select",
            "selection_operation": "select",
            "action_handle": f"copy-{index}",
            "card": {
                **target,
                "instance_id": f"copy-{index}",
                "index": index,
            },
        }
        for index in range(2)
    ]
    assert (
        authority.choose(
            observation={"phase": "card_selection", "run": {"floor": 7}},
            semantic_actions=equal_picker_actions,
            snapshot=_snapshot(candidate_count=2),
            valid=np.ones(2, dtype=np.bool_),
        )
        == 0
    )
    assert authority.mechanical_dispatches == 1


def test_auto_confirm_processes_the_current_new_macro_decision() -> None:
    calls = 0

    def forward(snapshot: Any, hidden: Any) -> tuple[torch.Tensor, Any]:
        nonlocal calls
        calls += 1
        values = [0.0, 5.0] if calls == 1 else [5.0, 0.0]
        return torch.tensor(values[: len(snapshot.action_mask)]), hidden

    authority = MacroCollectionAuthority(
        forward_q=forward,
        initial_state=lambda: None,
        epsilon=0.0,
        seed=4,
    )
    authority.begin_episode("ep-auto-confirm")
    assert (
        authority.choose(
            observation=_rest_observation(floor=7),
            semantic_actions=_REST_ACTIONS,
            snapshot=_snapshot(candidate_count=2),
            valid=np.ones(2, dtype=np.bool_),
        )
        == 1
    )
    assert (
        authority.choose(
            observation={"phase": "card_selection", "run": {"floor": 7}},
            semantic_actions=_PICKER_ACTIONS,
            snapshot=_snapshot(candidate_count=3),
            valid=np.ones(3, dtype=np.bool_),
        )
        == 0
    )
    # The engine auto-confirms and immediately publishes the next rest-site
    # decision.  The same choose() call consumes the absent confirm and owns
    # that next decision; it is not silently ceded to the champion.
    assert (
        authority.choose(
            observation=_rest_observation(floor=8),
            semantic_actions=_REST_ACTIONS,
            snapshot=_snapshot(candidate_count=2),
            valid=np.ones(2, dtype=np.bool_),
        )
        == 0
    )
    assert calls == 2
    authority.observe_step(reward=0.0, floor=8, terminal=True)
    episode = authority.finish_episode()
    assert episode is not None and len(episode.steps) == 2


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
            macro_online, resources.encoder, device, detach_hidden=False
        )
        forward_target = runner._forward_factory(
            macro_target, resources.encoder, device, detach_hidden=True
        )
        forward_behavior = runner._forward_factory(
            macro_online, resources.encoder, device, detach_hidden=True
        )
        replay = MacroSequenceReplay(capacity_episodes=8, window_length=4)
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
            forward_q=forward_behavior,
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


def test_evaluation_ownership_extends_into_deterministic_collection() -> None:
    """Stage-three joined evaluation: with evaluation_ownership the authority
    owns macro surfaces even in deterministic held-out collection; without it
    deterministic collection stays entirely with the champion."""

    import torch as _torch

    from sts2_rl.training import build_training_resources
    from tests.test_v33_recovery_semantics import (
        _recovery_config,
        _RestForgeSelectionSuccessBackend,
    )

    for ownership, expects_overrides in ((True, True), (False, False)):
        config = _recovery_config(max_steps=8, repeat_threshold=8)
        resources = build_training_resources(
            config,
            backend=_RestForgeSelectionSuccessBackend(),
        )
        try:
            values = _torch.zeros(64)

            def forward(
                snapshot: Any, hidden: Any, values: _torch.Tensor = values
            ) -> tuple[_torch.Tensor, Any]:
                return values[: len(snapshot.action_mask)], hidden

            authority = MacroCollectionAuthority(
                forward_q=forward,
                initial_state=lambda: None,
                epsilon=0.0,
                seed=3,
                evaluation_ownership=ownership,
            )
            resources.collector.macro_authority = authority
            resources.collector.collect_episode(
                epsilon=0.0,
                deterministic=True,
                record=False,
            )
            if expects_overrides:
                assert authority.overrides > 0
            else:
                assert authority.overrides == 0
        finally:
            resources.close()


_COMBAT_OBSERVATION = {
    "phase": "combat",
    "combat": {"in_progress": True},
    "run": {"floor": 5},
}

_COMBAT_ACTIONS = [
    {
        "kind": "play_card",
        "model_action_kind": "play_card",
        "card": {"id": "CARD.STRIKE", "instance_id": "CARD.STRIKE-1", "cost": 1},
        "target": {"id": "MONSTER.CULTIST", "index": 0},
    },
    {
        "kind": "play_card",
        "model_action_kind": "play_card",
        "card": {"id": "CARD.DEFEND", "instance_id": "CARD.DEFEND-1", "cost": 1},
    },
    {"kind": "end_turn", "model_action_kind": "end_turn"},
]


def _native_card_with_upgrade_preview() -> dict[str, Any]:
    return {
        "id": "ANGER",
        "instance_id": "source-card-instance",
        "index": 0,
        "source_pile": "Deck",
        "upgrade_level": 0,
        "is_upgradable": True,
        "is_removable": True,
        "enchantments": [{"id": "ENCHANT.TEST", "amount": 2}],
        "dynamic_vars": [{"name": "damage", "current_value": 6}],
        "upgrade_preview": {
            "id": "ANGER",
            "instance_id": "detached-preview-instance",
            "index": 0,
            "source_pile": "None",
            "upgrade_level": 1,
            "enchantments": [{"id": "ENCHANT.TEST", "amount": 2}],
            "dynamic_vars": [{"name": "damage", "current_value": 9}],
        },
    }


def _translated_picker_actions(native_card: dict[str, Any]) -> list[dict[str, Any]]:
    return _translate_legal_actions(
        (
            {"action": "select_card", "index": 0},
            {"action": "confirm_selection"},
            {"action": "cancel_selection"},
        ),
        sim_player={},
        battle={},
        map_state={},
        event={},
        rest_site={},
        shop={},
        rewards={},
        card_reward={},
        card_select={"cards": [native_card]},
        treasure={},
        relic_select={},
    )


def _grouped_surface(actions: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [
        {"prototype": group.prototype}
        for group in GroundedObservationEncoder().semantic_action_groups(actions)
    ]


def test_combat_view_is_declined_by_default_and_owned_with_the_flag() -> None:
    """Stage-4: forward compilation exposes the native-atomic combat view
    only when the challenger opts in; the stage-2/3 authority still declines
    combat to the champion."""

    from sts2_rl.semantics.forward import forward_decision

    assert forward_decision(_COMBAT_OBSERVATION, _COMBAT_ACTIONS) is None
    decision = forward_decision(
        _COMBAT_OBSERVATION, _COMBAT_ACTIONS, control_domain="combat"
    )
    assert decision is not None and decision.surface == "combat"
    assert [c.branch for c in decision.candidates] == [
        "play_card",
        "play_card",
        "end_turn",
    ]
    assert decision.candidates[0].native_index == 0
    assert decision.candidates[0].target_key != decision.candidates[1].target_key

    declining = _authority(q=[0.0, 0.0, 5.0])
    declining.begin_episode("ep-decline")
    assert (
        declining.choose(
            observation=_COMBAT_OBSERVATION,
            semantic_actions=_COMBAT_ACTIONS,
            snapshot=_snapshot(candidate_count=3),
            valid=np.ones(3, dtype=np.bool_),
        )
        is None
    )
    assert declining.declined == 1

    values = torch.tensor([0.0, 0.0, 5.0], dtype=torch.float32)

    def forward(snapshot: Any, hidden: Any) -> tuple[torch.Tensor, Any]:
        return values[: len(snapshot.action_mask)], hidden

    challenger = MacroCollectionAuthority(
        forward_q=forward,
        initial_state=lambda: None,
        epsilon=0.0,
        seed=7,
        control_domain="combat",
    )
    challenger.begin_episode("ep-challenger")
    index = challenger.choose(
        observation=_COMBAT_OBSERVATION,
        semantic_actions=_COMBAT_ACTIONS,
        snapshot=_snapshot(candidate_count=3),
        valid=np.ones(3, dtype=np.bool_),
    )
    assert index == 2  # greedy Q picks end_turn
    challenger.observe_step(reward=0.01, floor=5, terminal=True)
    episode = challenger.finish_episode()
    assert episode is not None and len(episode.steps) == 1
    step = episode.steps[0]
    assert step.surface == "combat" and step.branch == "end_turn"
    assert step.terminal is True and step.discount == 0.0


def test_card_matching_is_container_invariant() -> None:
    """Deck aggregates and picker cards render different views of the same
    card (measured on real journals: deck carries quantity/description/
    rarity, pickers carry upgrade_preview); executor binding must match on
    the mutually rendered keys and still separate distinct cards."""

    from sts2_rl.macro.authority import _card_matches

    deck_view = {
        "id": "CARD.DEFEND_IRONCLAD",
        "cost": 1,
        "is_upgraded": True,
        "quantity": 2,
        "description": "DEFEND_IRONCLAD.description",
        "rarity": "Basic",
        "upgrade_preview": None,
        "tags": ["Defend"],
    }
    picker_same = {
        "id": "CARD.DEFEND_IRONCLAD",
        "cost": 1,
        "is_upgraded": True,
        "quantity": None,
        "description": None,
        "rarity": None,
        "upgrade_preview": {"cost": 1, "is_upgraded": True},
        "tags": ["Defend"],
    }
    picker_unupgraded = {**picker_same, "is_upgraded": False}
    picker_other = {**picker_same, "id": "CARD.BASH"}

    assert _card_matches({"card": picker_same}, deck_view)
    assert not _card_matches({"card": picker_unupgraded}, deck_view)
    assert not _card_matches({"card": picker_other}, deck_view)
    assert not _card_matches({"card": None}, deck_view)
    assert _card_matches({"card": picker_same}, None)


def test_role_vocabulary_collision_fails_closed_to_the_champion() -> None:
    """A candidate set the configured role vocabulary cannot represent
    distinctly (measured live: co-occurring reward-claim branches hashing to
    one role) must decline to the champion and count, never crash."""

    from unittest.mock import patch

    authority = _authority(q=[0.0, 1.0])
    authority.begin_episode("ep-collision")
    with patch(
        "sts2_rl.macro.authority.GroundedObservationEncoder"
    ) as encoder_class:
        encoder_class.return_value.encode.side_effect = ValueError(
            "co-occurring semantic action branches collide in the configured "
            "role vocabulary"
        )
        index = authority.choose(
            observation=_rest_observation(floor=7),
            semantic_actions=_REST_ACTIONS,
            snapshot=_snapshot(candidate_count=2),
            valid=np.ones(2, dtype=np.bool_),
        )
    assert index is None
    assert authority.declined == 1
    assert authority.semantic_encode_collisions == 1
    assert authority.last_decision_value is None
