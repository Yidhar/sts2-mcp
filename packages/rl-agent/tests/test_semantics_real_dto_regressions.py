from __future__ import annotations

from copy import deepcopy
from typing import Any

import pytest

from sts2_rl.semantics import DecisionSemantics, DecisionSemanticsKernel


def _observation(
    *,
    phase: str,
    state_type: str,
) -> dict[str, Any]:
    """HeadlessSim-shaped observation with deliberately present inert DTOs."""

    return {
        "phase": phase,
        "state_type": state_type,
        "decision_domain": "combat" if phase == "combat" else "build",
        "run": {"act": 1, "floor": 7},
        "room": {
            "room_type": state_type,
            "room_model_id": f"ROOM.{state_type.upper()}",
            "coordinate": {"row": 7, "column": 2},
        },
        # These containers are always present in real rich snapshots.  Empty
        # containers must never steal surface resolution from state/action
        # evidence.
        "event": {"options": []},
        "combat": {"enemies": [], "in_progress": phase == "combat"},
        "rewards": {},
        "shop": {},
        "rest_site": {},
        "map": {},
        "player": {"hp": 70, "max_hp": 80},
    }


def _identified(
    *,
    phase: str,
    state_type: str,
    actions: tuple[dict[str, Any], ...],
) -> DecisionSemantics:
    return DecisionSemanticsKernel().identify(
        observation=_observation(phase=phase, state_type=state_type),
        legal_actions=actions,
    )


def _assert_injective(result: DecisionSemantics, expected_count: int) -> None:
    assert len(result.actions) == expected_count
    assert len({action.identities.loop.digest for action in result.actions}) == expected_count


@pytest.mark.parametrize(
    ("phase", "state_type", "expected_surface", "actions"),
    (
        (
            "actions",
            "combat_rewards",
            "reward",
            (
                {
                    "action": "claim_reward",
                    "action_index": 0,
                    "idx": 0,
                    "index": 0,
                    "kind": "claim_reward",
                    "label": "COMBAT_REWARD_GOLD",
                    "model_action_kind": "reward",
                    "reward": {
                        "index": 0,
                        "label": "COMBAT_REWARD_GOLD",
                        "reward_key": "gold|35|combat_reward_gold",
                        "reward_source": "combat_end",
                        "slot_index": 0,
                        "type": "gold",
                    },
                    "reward_key": "gold|35|combat_reward_gold",
                    "reward_type": "gold",
                },
                {
                    "action": "claim_reward",
                    "action_index": 1,
                    "idx": 1,
                    "index": 1,
                    "kind": "claim_reward",
                    "label": "VAMBRACE.title",
                    "model_action_kind": "reward",
                    "reward": {
                        "index": 1,
                        "label": "VAMBRACE.title",
                        "reward_key": "relic|relic_reward|vambrace.title",
                        "reward_source": "combat_end",
                        "slot_index": 1,
                        "type": "relic",
                    },
                    "reward_key": "relic|relic_reward|vambrace.title",
                    "reward_type": "relic",
                },
            ),
        ),
        (
            "combat",
            "monster",
            "combat",
            (
                {
                    "action": "use_potion",
                    "action_index": 4,
                    "idx": 4,
                    "kind": "use_potion",
                    "label": "WEAK_POTION.title",
                    "model_action_kind": "use_potion",
                    "potion": {
                        "id": "POTION.WEAK_POTION",
                        "slot": 0,
                        "slot_index": 0,
                        "target_type": "AnyEnemy",
                    },
                    "slot": 0,
                    "target_id": 3,
                },
                {
                    "action": "use_potion",
                    "action_index": 5,
                    "idx": 5,
                    "kind": "use_potion",
                    "label": "POTION_SHAPED_ROCK.title",
                    "model_action_kind": "use_potion",
                    "potion": {
                        "id": "POTION.POTION_SHAPED_ROCK",
                        "slot": 1,
                        "slot_index": 1,
                        "target_type": "AnyEnemy",
                    },
                    "slot": 1,
                    "target_id": 3,
                },
            ),
        ),
        (
            "actions",
            "map",
            "map",
            (
                {
                    "action": "choose_map_node",
                    "action_index": 0,
                    "col": 2,
                    "idx": 0,
                    "index": 0,
                    "kind": "choose_map_node",
                    "label": "monster",
                    "map_node": {
                        "coord": {"x": 2, "y": 8},
                        "index": 0,
                        "point_type": "monster",
                    },
                    "model_action_kind": "map",
                    "row": 8,
                },
                {
                    "action": "choose_map_node",
                    "action_index": 1,
                    "col": 4,
                    "idx": 1,
                    "index": 1,
                    "kind": "choose_map_node",
                    "label": "monster",
                    "map_node": {
                        "coord": {"x": 4, "y": 8},
                        "index": 1,
                        "point_type": "monster",
                    },
                    "model_action_kind": "map",
                    "row": 8,
                },
            ),
        ),
        (
            "actions",
            "card_reward",
            "reward",
            (
                {
                    "action": "select_card_reward",
                    "action_index": 0,
                    "card": {
                        "id": "CARD.VICIOUS",
                        "index": 0,
                        "is_upgraded": False,
                        "enchantments": [],
                        "afflictions": [],
                    },
                    "card_id": "VICIOUS",
                    "card_index": 0,
                    "idx": 0,
                    "index": 0,
                    "kind": "select_card_reward",
                    "label": "VICIOUS.title",
                    "model_action_kind": "card_reward",
                },
                {
                    "action": "select_card_reward",
                    "action_index": 1,
                    "card": {
                        "id": "CARD.FORGOTTEN_RITUAL",
                        "index": 1,
                        "is_upgraded": True,
                        "enchantments": [],
                        "afflictions": [],
                    },
                    "card_id": "FORGOTTEN_RITUAL",
                    "card_index": 1,
                    "idx": 1,
                    "index": 1,
                    "kind": "select_card_reward",
                    "label": "FORGOTTEN_RITUAL.title+",
                    "model_action_kind": "card_reward",
                },
            ),
        ),
        (
            "actions",
            "shop",
            "shop",
            (
                {
                    "action": "shop_purchase",
                    "action_index": 0,
                    "idx": 0,
                    "index": 0,
                    "item": {
                        "card": {
                            "id": "CARD.THUNDERCLAP",
                            "is_upgraded": True,
                        },
                        "card_id": "THUNDERCLAP",
                        "category": "card",
                        "cost": 25,
                        "index": 0,
                        "slot_index": 0,
                        "type": "card",
                    },
                    "kind": "shop_purchase",
                    "label": "THUNDERCLAP.title+",
                    "model_action_kind": "shop",
                    "model_action_variant": "buy",
                },
                {
                    "action": "shop_purchase",
                    "action_index": 1,
                    "idx": 1,
                    "index": 1,
                    "item": {
                        "category": "relic",
                        "cost": 143,
                        "index": 1,
                        "relic_id": "RELIC.VAMBRACE",
                        "slot_index": 1,
                        "type": "relic",
                    },
                    "kind": "shop_purchase",
                    "label": "VAMBRACE.title",
                    "model_action_kind": "shop",
                    "model_action_variant": "buy",
                },
            ),
        ),
    ),
)
def test_real_dto_surface_candidates_have_injective_loop_identity(
    phase: str,
    state_type: str,
    expected_surface: str,
    actions: tuple[dict[str, Any], ...],
) -> None:
    result = _identified(
        phase=phase,
        state_type=state_type,
        actions=actions,
    )

    assert result.scopes.root.spec_id == expected_surface
    _assert_injective(result, len(actions))


def test_selection_identity_retains_permanent_copy_distinctions() -> None:
    shared = {
        "action": "select_card",
        "kind": "select_card",
        "label": "TWIN_STRIKE.title",
        "model_action_kind": "card_selection",
        "model_action_variant": "select",
        "selection_operation": "select",
        "selection": {
            "operation_type": "select",
            "prompt_id": "card_selection.TO_REMOVE",
            "source_zone": "Deck",
        },
    }
    actions = (
        {
            **shared,
            "action_index": 11,
            "card_index": 11,
            "card": {
                "id": "CARD.TWIN_STRIKE",
                "index": 11,
                "floor_added_to_deck": 3,
                "is_upgraded": False,
                "enchantments": [],
                "afflictions": [],
                "source_pile": "Deck",
            },
        },
        {
            **shared,
            "action_index": 12,
            "card_index": 12,
            "card": {
                "id": "CARD.TWIN_STRIKE",
                "index": 12,
                "floor_added_to_deck": 5,
                "is_upgraded": False,
                "enchantments": [],
                "afflictions": [],
                "source_pile": "Deck",
            },
        },
    )
    result = _identified(
        phase="card_selection",
        state_type="card_select",
        actions=actions,
    )

    assert result.scopes.active.spec_id == "selection"
    _assert_injective(result, 2)


def test_root_transport_reindexing_does_not_change_control_identity() -> None:
    action = {
        "action": "choose_map_node",
        "action_handle": "request-a",
        "action_index": 0,
        "idx": 0,
        "index": 0,
        "kind": "choose_map_node",
        "map_node": {
            "coord": {"x": 2, "y": 8},
            "index": 7,
            "point_type": "monster",
        },
        "model_action_kind": "map",
    }
    reindexed = deepcopy(action)
    reindexed.update(
        {
            "action_handle": "request-b",
            "action_index": 103,
            "idx": 103,
            "index": 103,
        }
    )

    before = _identified(
        phase="actions",
        state_type="map",
        actions=(action,),
    )
    after = _identified(
        phase="actions",
        state_type="map",
        actions=(reindexed,),
    )

    assert before.actions[0].identities.loop == after.actions[0].identities.loop
    assert before.node.loop == after.node.loop


def test_realized_cost_ledger_noise_does_not_change_control_identity() -> None:
    action = {
        "action": "event_option",
        "action_index": 0,
        "current_hp": 60,
        "damage_taken": 0,
        "kind": "event_option",
        "model_action_kind": "event_option",
        "option": {
            "effect_contract": {"gain_gold": 35},
            "option_id": "TAKE_GOLD",
            "revival_count": 0,
        },
    }
    after_cost = deepcopy(action)
    after_cost["current_hp"] = 1
    after_cost["damage_taken"] = 59
    after_cost["option"]["revival_count"] = 4

    before = _identified(
        phase="event",
        state_type="event",
        actions=(action,),
    )
    after = _identified(
        phase="event",
        state_type="event",
        actions=(after_cost,),
    )

    assert before.actions[0].identities.loop == after.actions[0].identities.loop
    assert before.actions[0].identities.exact != after.actions[0].identities.exact
