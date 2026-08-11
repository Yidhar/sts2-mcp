from __future__ import annotations

from typing import Any

import numpy as np
import pytest
import torch

from sts2_rl.encoding import GroundedObservationEncoder
from sts2_rl.macro import JoinedCollectionAuthority, MacroCollectionAuthority


def _snapshot(
    observation: dict[str, Any],
    actions: list[dict[str, Any]],
) -> Any:
    return GroundedObservationEncoder().encode(observation, actions).snapshot


def test_joined_router_keeps_models_and_recurrent_states_domain_local() -> None:
    macro_seen: list[int] = []
    combat_seen: list[int] = []

    def macro_forward(snapshot: Any, hidden: Any) -> tuple[torch.Tensor, Any]:
        macro_seen.append(int(hidden))
        return torch.zeros(snapshot.candidate_count), int(hidden) + 1

    def combat_forward(snapshot: Any, hidden: Any) -> tuple[torch.Tensor, Any]:
        combat_seen.append(int(hidden))
        return torch.zeros(snapshot.candidate_count), int(hidden) + 1

    macro = MacroCollectionAuthority(
        forward_q=macro_forward,
        initial_state=lambda: 100,
        epsilon=0.0,
        evaluation_ownership=True,
        control_domain="macro",
    )
    combat = MacroCollectionAuthority(
        forward_q=combat_forward,
        initial_state=lambda: 200,
        epsilon=0.0,
        evaluation_ownership=True,
        control_domain="combat",
    )
    router = JoinedCollectionAuthority(macro=macro, combat=combat)
    router.begin_episode("joined")

    rest_observation = {
        "phase": "rest",
        "combat": {"in_progress": False},
        "run": {"floor": 5},
        "player": {"deck": []},
        "rest_site": {"options": []},
    }
    rest_actions = [
        {
            "kind": "choose_rest_option",
            "model_action_kind": "rest_site",
            "model_action_variant": "rest",
            "option": {"type": "heal"},
        }
    ]
    assert (
        router.choose(
            observation=rest_observation,
            semantic_actions=rest_actions,
            snapshot=_snapshot(rest_observation, rest_actions),
            valid=np.ones(1, dtype=np.bool_),
        )
        == 0
    )
    router.observe_step(reward=0.1, floor=5, terminal=False)

    combat_observation = {
        "phase": "combat",
        "combat": {"in_progress": True},
        "run": {"floor": 6},
    }
    combat_actions = [{"kind": "end_turn", "model_action_kind": "end_turn"}]
    assert (
        router.choose(
            observation=combat_observation,
            semantic_actions=combat_actions,
            snapshot=_snapshot(combat_observation, combat_actions),
            valid=np.ones(1, dtype=np.bool_),
        )
        == 0
    )
    router.observe_step(reward=-0.2, floor=6, terminal=True)
    episodes = router.finish_episodes("joined")

    assert macro_seen == [100]
    assert combat_seen == [200]
    assert episodes["macro"] is not None
    assert episodes["combat"] is not None
    assert episodes["macro"].control_domain == "macro"
    assert episodes["combat"].control_domain == "combat"
    assert episodes["combat"].steps[0].recurrent_reset is True


def test_joined_router_rejects_two_authorities_for_one_domain() -> None:
    authority = MacroCollectionAuthority(
        forward_q=lambda snapshot, hidden: (
            torch.zeros(snapshot.candidate_count),
            hidden,
        ),
        initial_state=lambda: None,
        control_domain="macro",
    )
    with pytest.raises(ValueError, match="combat authority"):
        JoinedCollectionAuthority(macro=authority, combat=authority)


def test_joined_router_resets_combat_recurrence_after_macro_interval() -> None:
    combat_seen: list[int] = []

    macro = MacroCollectionAuthority(
        forward_q=lambda snapshot, hidden: (
            torch.zeros(snapshot.candidate_count),
            hidden,
        ),
        initial_state=lambda: 100,
        epsilon=0.0,
        evaluation_ownership=True,
        control_domain="macro",
    )

    def combat_forward(snapshot: Any, hidden: Any) -> tuple[torch.Tensor, Any]:
        combat_seen.append(int(hidden))
        return torch.zeros(snapshot.candidate_count), int(hidden) + 1

    combat = MacroCollectionAuthority(
        forward_q=combat_forward,
        initial_state=lambda: 200,
        epsilon=0.0,
        evaluation_ownership=True,
        control_domain="combat",
    )
    router = JoinedCollectionAuthority(macro=macro, combat=combat)
    router.begin_episode("two-combats")

    combat_actions = [{"kind": "end_turn", "model_action_kind": "end_turn"}]
    for floor in (2, 4):
        observation = {
            "phase": "combat",
            "combat": {"in_progress": True},
            "run": {"floor": floor},
        }
        assert (
            router.choose(
                observation=observation,
                semantic_actions=combat_actions,
                snapshot=_snapshot(observation, combat_actions),
                valid=np.ones(1, dtype=np.bool_),
            )
            == 0
        )
        router.observe_step(reward=0.1, floor=floor, terminal=False)
        if floor == 2:
            macro_observation = {
                "phase": "map",
                "combat": {"in_progress": False},
                "run": {"floor": 3},
            }
            macro_actions = [
                {
                    "kind": "choose_map_node",
                    "model_action_kind": "map",
                    "map_node": {"row": 3, "col": 0},
                }
            ]
            assert (
                router.choose(
                    observation=macro_observation,
                    semantic_actions=macro_actions,
                    snapshot=_snapshot(macro_observation, macro_actions),
                    valid=np.ones(1, dtype=np.bool_),
                )
                == 0
            )
            router.observe_step(reward=0.0, floor=3, terminal=False)

    router.observe_step(reward=0.0, floor=4, terminal=True)
    episodes = router.finish_episodes("two-combats")
    assert combat_seen == [200, 200]
    assert episodes["combat"] is not None
    assert [step.recurrent_reset for step in episodes["combat"].steps] == [True, True]
