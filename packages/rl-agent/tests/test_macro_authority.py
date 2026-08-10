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
