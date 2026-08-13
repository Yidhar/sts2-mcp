from __future__ import annotations

import json
import math
from dataclasses import asdict, replace
from pathlib import Path
from typing import cast

import numpy as np
import pytest
import torch

from sts2_baseline import RolloutStep, SequenceUnroll
from sts2_env._sim_translate_actions import _translate_legal_actions
from sts2_env._sim_translate_decisions import _translate_card_sel_block
from sts2_env._sim_translate_entities import _translate_card
from sts2_rl.checkpoints import ValidatedResumeCheckpoint
from sts2_rl.contracts import EnvironmentBackend
from sts2_rl.encoding import (
    EncodedDecisionSnapshot,
    GroundedEncodingConfig,
    GroundedObservationEncoder,
    grounding_encoding_identity,
)
from sts2_rl.encoding.snapshot import sparse_token_table
from sts2_rl.models import GroundedCandidateConfig, RecurrentCandidateModel
from sts2_rl.training import (
    BoundedTransactionReplay,
    TrainingConfig,
    TrainingState,
    TransactionEffect,
    TransactionLearningConfig,
    TransactionLifecycleEvidence,
    TransactionLifecycleOutcome,
    TransactionOutcome,
    TransactionStep,
    TransactionTrace,
    backfill_factual_monte_carlo_returns,
    build_training_resources,
    initialize_model_from_checkpoint,
    load_training_checkpoint,
    observed_outcome_pairs,
    save_training_checkpoint,
)
from sts2_rl.training import checkpointing as checkpointing_module
from sts2_rl.training.collector import (
    _classify_transaction_transition,
    _deadlock_semantic_observation,
    _semantic_action_surface,
    _transaction_node_key,
    _transaction_option_universe,
    _transaction_surface_key,
)
from sts2_rl.training.learner import VTraceLearner
from sts2_rl.training.runtime import run_training
from sts2_rl.training.trajectory import (
    semantic_action_fingerprint,
    semantic_decision_fingerprint,
)


class _NoopBackend:
    def close(self) -> None:
        pass


def test_native_upgrade_preview_translation_is_exact_and_relation_bound() -> None:
    native_card = {
        "id": "ANGER",
        "instance_id": "source-card-instance",
        "source_pile": "Deck",
        "upgrade_level": 0,
        "enchantments": [{"id": "ENCHANT.TEST", "amount": 2}],
        "dynamic_vars": [{"name": "damage", "current_value": 6}],
        "upgrade_preview": {
            "id": "ANGER",
            # A detached native clone may own a different UUID. Translation
            # must bind the alternative state to the physical source card.
            "instance_id": "detached-preview-instance",
            # Detached native card clones are not in a physical pile.  The
            # translator must rebind this alternative state to its source.
            "source_pile": "None",
            "upgrade_level": 1,
            "enchantments": [{"id": "ENCHANT.TEST", "amount": 2}],
            "dynamic_vars": [{"name": "damage", "current_value": 9}],
        },
    }

    translated = _translate_card(native_card, selection_membership="selectable")
    preview = translated["upgrade_preview"]
    assert translated["id"] == preview["id"] == "CARD.ANGER"
    assert translated["instance_id"] == preview["instance_id"] == (
        "source-card-instance"
    )
    assert translated["source_pile"] == preview["source_pile"] == "Deck"
    assert translated["pile"] == preview["pile"] == "Deck"
    assert preview["upgrade_level"] == 1
    assert preview["enchantments"] == native_card["upgrade_preview"]["enchantments"]
    assert preview["dynamic_vars"] == native_card["upgrade_preview"]["dynamic_vars"]

    translated_actions = _translate_legal_actions(
        ({"action": "select_card", "index": 0},),
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
    assert translated_actions[0]["upgrade_preview"] == preview
    assert translated_actions[0]["card"]["upgrade_preview"] == preview
    assert translated_actions[0]["upgrade_preview"] is not (
        translated_actions[0]["card"]["upgrade_preview"]
    )

    recursive = {
        **native_card,
        "upgrade_preview": {
            **native_card["upgrade_preview"],
            "upgrade_preview": {"id": "ANGER"},
        },
    }
    with pytest.raises(ValueError, match="cannot contain another"):
        _translate_card(recursive)


def test_strict_group_surface_removes_physical_selection_instance_noise() -> None:
    encoder = GroundedObservationEncoder(GroundedEncodingConfig(max_candidates=4))
    raw_actions = tuple(
        {
            "action_handle": f"select-{index}",
            "kind": "combat_select_card",
            "model_action_kind": "card_selection",
            "model_action_variant": "select",
            "selection_operation": "select",
            "card_index": index,
            "card": {
                "id": "CARD.WOUND",
                "instance_uuid": f"wound-{index}",
                "source_pile": "Discard",
                "cost": -2,
            },
        }
        for index in range(2_040)
    )
    raw_cards = [
        {
            "id": "CARD.WOUND",
            "instance_uuid": f"wound-{index}",
            "source_pile": "Discard",
            "cost": -2,
        }
        for index in range(2_040)
    ]
    observation = {
        "run": {"act": 3, "floor": 46, "room_type": "combat"},
        "card_selection": {
            "mode": "SimpleGrid",
            "prompt_id": "card.HEADBUTT.selection",
            "operation_type": "select",
            "source_zone": "Discard",
            "selected_count": 0,
            "min_select": 1,
            "max_select": 1,
            "cards": raw_cards,
            "selected_cards": [],
        },
    }

    groups = encoder.semantic_action_groups(raw_actions)
    assert len(groups) == 1
    assert groups[0].multiplicity == 2_040
    semantic_actions = _semantic_action_surface(groups)
    universe = _transaction_option_universe(
        observation["card_selection"],
        semantic_actions,
    )
    assert len(universe) == 1
    assert json.loads(universe[0]) == {
        "identity": {
            "definition": "CARD.WOUND",
            "physical_source": "Discard",
            "facts": {"cost": -2},
        },
        "multiplicity": 2_040,
    }
    assert "wound-" not in universe[0]

    surface = _transaction_surface_key(observation, semantic_actions)
    assert surface is not None
    node = _transaction_node_key(
        surface,
        observation,
        semantic_actions,
    )
    deadlock_observation = _deadlock_semantic_observation(
        observation,
        semantic_actions,
    )
    decision = semantic_decision_fingerprint(
        deadlock_observation,
        semantic_actions,
    )

    reversed_actions = tuple(reversed(raw_actions))
    reversed_observation = {
        **observation,
        "card_selection": {
            **observation["card_selection"],
            "cards": list(reversed(raw_cards)),
        },
    }
    reversed_semantic_actions = _semantic_action_surface(encoder.semantic_action_groups(reversed_actions))
    assert (
        _transaction_surface_key(
            reversed_observation,
            reversed_semantic_actions,
        )
        == surface
    )
    assert (
        _transaction_node_key(
            surface,
            reversed_observation,
            reversed_semantic_actions,
        )
        == node
    )
    assert (
        semantic_decision_fingerprint(
            _deadlock_semantic_observation(
                reversed_observation,
                reversed_semantic_actions,
            ),
            reversed_semantic_actions,
        )
        == decision
    )
    assert semantic_action_fingerprint(reversed_semantic_actions[0]) == semantic_action_fingerprint(semantic_actions[0])
    assert len(json.dumps(deadlock_observation, sort_keys=True)) < 2_000


def test_transaction_identity_never_merges_same_definition_with_different_facts() -> None:
    encoder = GroundedObservationEncoder(GroundedEncodingConfig(max_candidates=8))
    base_card = {
        "id": "CARD.STRIKE",
        "source_pile": "Discard",
        "cost": 1,
    }
    actions = (
        {
            "action_handle": "select-normal",
            "kind": "select_card",
            "model_action_kind": "card_selection",
            "model_action_variant": "select",
            "selection_operation": "select",
            "card_index": 0,
            "card": {
                **base_card,
                "instance_uuid": "normal-instance",
                "is_upgraded": False,
                "enchantments": [],
            },
        },
        {
            "action_handle": "select-enchanted",
            "kind": "select_card",
            "model_action_kind": "card_selection",
            "model_action_variant": "select",
            "selection_operation": "select",
            "card_index": 1,
            "card": {
                **base_card,
                "instance_uuid": "enchanted-instance",
                "is_upgraded": True,
                "enchantments": [{"id": "ENCHANTMENT.TEST", "amount": 2}],
            },
        },
    )
    groups = encoder.semantic_action_groups(actions)
    assert len(groups) == 2
    semantic_actions = _semantic_action_surface(groups)
    selection = {
        "mode": "SimpleGrid",
        "prompt_id": "strict-facts",
        "operation_type": "select",
        "source_zone": "Discard",
        "selected_count": 0,
        "min_select": 1,
        "max_select": 1,
    }
    universe = _transaction_option_universe(selection, semantic_actions)
    assert len(universe) == 2
    identities = [json.loads(item)["identity"] for item in universe]
    assert {identity["facts"]["is_upgraded"] for identity in identities} == {
        False,
        True,
    }
    assert any(identity["facts"].get("enchantments") for identity in identities)

    observation = {
        "run": {"act": 1, "floor": 7, "room_type": "combat"},
        "card_selection": selection,
    }
    surface = _transaction_surface_key(observation, semantic_actions)
    assert surface is not None
    normal_selected = {
        **observation,
        "card_selection": {
            **selection,
            "selected_count": 1,
            "selected_cards": [actions[0]["card"]],
        },
    }
    enchanted_selected = {
        **observation,
        "card_selection": {
            **selection,
            "selected_count": 1,
            "selected_cards": [actions[1]["card"]],
        },
    }
    assert _transaction_node_key(
        surface,
        normal_selected,
        semantic_actions,
    ) != _transaction_node_key(
        surface,
        enchanted_selected,
        semantic_actions,
    )


def test_transaction_surface_and_node_are_order_independent_but_membership_sensitive() -> None:
    first_observation = {
        "run": {"act": 1, "floor": 7, "room_type": "combat"},
        "card_selection": {
            "mode": "SimpleGrid",
            "prompt_id": "card.TEST.discard",
            "operation_type": "discard",
            "source_zone": "Hand",
            "min_select": 1,
            "max_select": 2,
            "selected_count": 1,
            "remaining_picks": 1,
            "can_confirm": True,
            "options": [
                {
                    "option_index": 0,
                    "is_selected": True,
                    "preview": {"damage": 9},
                    "card": {
                        "id": "CARD.STRIKE",
                        "instance_id": "strike-1",
                        "pile": "Selected",
                        "cost": 0,
                    },
                },
                {
                    "option_index": 1,
                    "is_selected": False,
                    "preview": {"damage": 100},
                    "card": {
                        "id": "CARD.DEFEND",
                        "instance_id": "defend-1",
                        "pile": "Hand",
                        "cost": 1,
                    },
                },
            ],
        },
    }
    first_actions = (
        {
            "action_handle": "volatile-a",
            "kind": "deselect_card",
            "model_action_kind": "card_selection",
            "model_action_variant": "deselect",
            "card": {
                "id": "CARD.STRIKE",
                "instance_id": "strike-1",
                "pile": "Selected",
                "cost": 0,
            },
        },
        {
            "action_handle": "volatile-b",
            "kind": "select_card",
            "model_action_kind": "card_selection",
            "model_action_variant": "select",
            "card": {
                "id": "CARD.DEFEND",
                "instance_id": "defend-1",
                "pile": "Hand",
                "cost": 1,
            },
        },
        {
            "action_handle": "volatile-c",
            "kind": "confirm_selection",
            "model_action_kind": "card_selection",
            "model_action_variant": "confirm",
        },
    )
    reordered_observation = {
        **first_observation,
        "card_selection": {
            **first_observation["card_selection"],
            "options": list(reversed(first_observation["card_selection"]["options"])),
        },
    }
    reordered_actions = tuple(reversed(first_actions))

    first_surface = _transaction_surface_key(first_observation, first_actions)
    reordered_surface = _transaction_surface_key(reordered_observation, reordered_actions)
    assert first_surface is not None
    assert reordered_surface == first_surface
    assert _transaction_node_key(first_surface, first_observation, first_actions) == _transaction_node_key(
        reordered_surface,
        reordered_observation,
        reordered_actions,
    )

    changed_membership = {
        **first_observation,
        "card_selection": {
            **first_observation["card_selection"],
            "options": [
                {
                    **first_observation["card_selection"]["options"][0],
                    "is_selected": False,
                    "card": {
                        **first_observation["card_selection"]["options"][0]["card"],
                        "pile": "Hand",
                        "cost": 2,
                    },
                },
                {
                    **first_observation["card_selection"]["options"][1],
                    "is_selected": True,
                    "card": {
                        **first_observation["card_selection"]["options"][1]["card"],
                        "pile": "Selected",
                        "cost": 0,
                    },
                },
            ],
        },
    }
    changed_surface = _transaction_surface_key(changed_membership, reordered_actions)
    assert changed_surface == first_surface
    assert _transaction_node_key(
        changed_surface,
        changed_membership,
        reordered_actions,
    ) != _transaction_node_key(first_surface, first_observation, first_actions)


def test_transaction_node_separates_reward_relevant_world_context() -> None:
    actions = (
        {
            "kind": "select_card",
            "model_action_kind": "card_selection",
            "model_action_variant": "select",
            "card": {"id": "CARD.STRIKE", "instance_id": "strike-1"},
        },
    )
    selection = {
        "mode": "SimpleGrid",
        "prompt_id": "test.discard",
        "operation_type": "discard",
        "min_select": 1,
        "max_select": 1,
        "cards": [{"id": "CARD.STRIKE", "instance_id": "strike-1"}],
        "selected_cards": [],
    }
    low_hp = {
        "run": {"act": 1, "floor": 3, "room_type": "combat"},
        "player": {"hp": 10, "deck_cards": [{"id": "CARD.STRIKE"}]},
        "_training": {"revivals_used": 2, "player_hp_lost": 90},
        "card_selection": selection,
    }
    high_hp = {
        **low_hp,
        "player": {"hp": 80, "deck_cards": [{"id": "CARD.BASH"}]},
        "_training": {"revivals_used": 0, "player_hp_lost": 0},
    }

    surface = _transaction_surface_key(low_hp, actions)
    assert surface == _transaction_surface_key(high_hp, actions)
    assert surface is not None
    assert _transaction_node_key(surface, low_hp, actions) != _transaction_node_key(
        surface,
        high_hp,
        actions,
    )


def test_action_only_selection_surface_is_captured_without_observation_dto() -> None:
    observation = {
        "phase": "combat",
        "decision_domain": "combat",
        "run": {"act": 1, "floor": 3, "room_type": "combat"},
    }
    actions = (
        {
            "action_handle": "select-1",
            "kind": "select_card",
            "model_action_kind": "card_selection",
            "model_action_variant": "select",
            "prompt_id": "legacy.discard",
            "operation_type": "discard",
            "source_zone": "Discard",
            "card": {
                "id": "CARD.STRIKE",
                "instance_id": "strike-1",
                "pile": "Discard",
                "selection_membership": "selectable",
            },
        },
        {
            "action_handle": "select-2",
            "kind": "select_card",
            "model_action_kind": "card_selection",
            "model_action_variant": "select",
            "prompt_id": "legacy.discard",
            "operation_type": "discard",
            "source_zone": "Discard",
            "card": {
                "id": "CARD.DEFEND",
                "instance_id": "defend-1",
                "pile": "Discard",
                "selection_membership": "selectable",
            },
        },
    )

    surface = _transaction_surface_key(observation, actions)
    assert surface is not None
    assert _transaction_surface_key(observation, tuple(reversed(actions))) == surface


def test_real_translated_selection_membership_keeps_surface_and_changes_node() -> None:
    """Selectable/selected movement is one transaction, not a false completion."""

    selection_contract = {
        "mode": "SimpleGrid",
        "prompt_id": "card.HEADBUTT.selection",
        "operation_type": "select",
        "source_zone": "Discard",
        "min_select": 1,
        "max_select": 2,
        "selected_count": 1,
        "remaining_picks": 1,
        "can_confirm": True,
    }
    strike = {
        "id": "STRIKE",
        "instance_id": "strike-1",
        "source_pile": "Discard",
    }
    defend = {
        "id": "DEFEND",
        "instance_id": "defend-1",
        "source_pile": "Discard",
    }
    before_selection = _translate_card_sel_block(
        {
            **selection_contract,
            # Real translation marks both aliases selectable. The explicit
            # field is authoritative so exposing both must not double-count it.
            "cards": [defend],
            "selectable_cards": [defend],
            "selected_cards": [strike],
        },
        {},
        None,
    )
    after_selection = _translate_card_sel_block(
        {
            **selection_contract,
            "cards": [defend, strike],
            "selectable_cards": [defend, strike],
            "selected_cards": [],
            "selected_count": 0,
            "remaining_picks": 2,
            "can_confirm": False,
        },
        {},
        None,
    )
    assert before_selection["cards"][0]["selection_membership"] == "selectable"
    assert before_selection["selectable_cards"][0]["selection_membership"] == "selectable"
    assert before_selection["selected_cards"][0]["selection_membership"] == "selected"

    before = {
        "run": {"act": 1, "floor": 3, "room_type": "combat"},
        "card_selection": before_selection,
    }
    after = {
        "run": before["run"],
        "card_selection": after_selection,
    }
    before_actions = (
        {
            "kind": "select_card",
            "model_action_kind": "card_selection",
            "model_action_variant": "select",
            "card": before_selection["selectable_cards"][0],
        },
        {
            "kind": "deselect_card",
            "model_action_kind": "card_selection",
            "model_action_variant": "deselect",
            "card": before_selection["selected_cards"][0],
        },
        {
            "kind": "confirm_selection",
            "model_action_kind": "card_selection",
            "model_action_variant": "confirm",
        },
    )
    after_actions = tuple(
        {
            "kind": "select_card",
            "model_action_kind": "card_selection",
            "model_action_variant": "select",
            "card": card,
        }
        for card in after_selection["selectable_cards"]
    )

    before_surface = _transaction_surface_key(before, before_actions)
    after_surface = _transaction_surface_key(after, after_actions)
    assert before_surface is not None
    assert after_surface == before_surface
    assert _transaction_node_key(before_surface, before, before_actions) != _transaction_node_key(
        after_surface,
        after,
        after_actions,
    )


def test_inactive_empty_card_selection_dto_is_not_a_transaction() -> None:
    observation = {
        "phase": "combat",
        "decision_domain": "combat",
        "card_selection": {},
    }
    actions = (
        {
            "action": "end_turn",
            "model_action_kind": "end_turn",
        },
    )

    assert _transaction_surface_key(observation, actions) is None


@pytest.mark.parametrize(
    ("operation", "prompt_id"),
    (
        ("upgrade", "card_selection.TO_UPGRADE"),
        ("remove", "shop.REMOVE_CARD"),
    ),
)
def test_real_max_one_selection_journal_semantics_keep_one_surface_and_return_node(
    operation: str,
    prompt_id: str,
) -> None:
    """Upgrade/remove grids hide unselected cards after a toggle.

    This mirrors the held-out journal shape that previously split select and
    deselect into two separately completed transaction surfaces.
    """

    cards = [
        {
            "id": "CARD.STRIKE_IRONCLAD",
            "instance_id": "strike-1",
            "source_pile": "Deck",
            "cost": 1,
            "is_upgraded": False,
        },
        {
            "id": "CARD.DEFEND_IRONCLAD",
            "instance_id": "defend-1",
            "source_pile": "Deck",
            "cost": 1,
            "is_upgraded": False,
        },
    ]

    def observation(*, selected: bool, state_hash: str) -> dict[str, object]:
        return {
            "phase": "selection",
            "decision_domain": "build",
            "state_hash": state_hash,
            "semantic_state_hash": f"semantic-{state_hash}",
            "run": {
                "act": 1,
                "floor": 8,
                "room_type": "shop" if operation == "remove" else "rest",
            },
            "player": {
                "hp": 42,
                "max_hp": 68,
                "gold": 120,
                "deck": cards,
            },
            "_training": {"revivals_used": 0, "player_hp_lost": 7},
            "card_selection": {
                "screen_type": "DeckUpgrade" if operation == "upgrade" else "DeckRemove",
                "prompt_id": prompt_id,
                "operation_type": operation,
                "source_zone": "Deck",
                "destination_zone": "Deck",
                "min_select": 1,
                "max_select": 1,
                "selected_count": int(selected),
                "remaining_picks": int(not selected),
                "requires_manual_confirmation": True,
                "can_confirm": selected,
                # Real max-one screens hide all other selectable cards while
                # the selected card awaits confirmation.
                "cards": [] if selected else cards,
                "selected_cards": [cards[1]] if selected else [],
            },
        }

    initial_actions = tuple(
        {
            "action_handle": f"volatile-select-{index}",
            "action": "select_card",
            "kind": "select_card",
            "model_action_kind": "card_selection",
            "model_action_variant": "select",
            "selection_operation": "select",
            "card": card,
        }
        for index, card in enumerate(cards)
    )
    selected_actions = (
        {
            "action_handle": "volatile-deselect",
            "action": "deselect_card",
            "kind": "deselect_card",
            "model_action_kind": "card_selection",
            "model_action_variant": "deselect",
            "selection_operation": "deselect",
            "card": {
                **cards[1],
                "is_selected": True,
                "selection_membership": "selected",
            },
        },
        {
            "action_handle": "volatile-confirm",
            "action": "confirm_selection",
            "kind": "confirm_selection",
            "model_action_kind": "card_selection",
            "model_action_variant": "confirm",
            "selection_operation": "confirm",
        },
        {
            "action_handle": "volatile-cancel",
            "action": "cancel_selection",
            "kind": "cancel_selection",
            "model_action_kind": "card_selection",
            "model_action_variant": "cancel_prompt",
            "selection_operation": "cancel_prompt",
        },
    )
    initial = observation(selected=False, state_hash="before")
    selected = observation(selected=True, state_hash="selected")
    returned = observation(selected=False, state_hash="returned")

    initial_surface = _transaction_surface_key(initial, initial_actions)
    selected_surface = _transaction_surface_key(selected, selected_actions)
    returned_surface = _transaction_surface_key(
        returned,
        tuple(reversed(initial_actions)),
    )
    assert initial_surface is not None
    assert initial_surface == selected_surface == returned_surface

    initial_node = _transaction_node_key(
        initial_surface,
        initial,
        initial_actions,
    )
    selected_node = _transaction_node_key(
        selected_surface,
        selected,
        selected_actions,
    )
    returned_node = _transaction_node_key(
        returned_surface,
        returned,
        tuple(reversed(initial_actions)),
    )
    assert selected_node != initial_node
    assert returned_node == initial_node


def test_mandatory_multiselect_deselect_returns_to_same_semantic_node() -> None:
    cards = tuple(
        {
            "id": f"CARD.TEST_{name}",
            "instance_id": f"{name.lower()}-instance",
            "source_pile": "Hand",
            "cost": 1,
        }
        for name in ("A", "B", "C")
    )

    def state(selected_count: int) -> tuple[dict[str, object], tuple[dict[str, object], ...]]:
        selected = cards[:selected_count]
        selectable = cards[selected_count:]
        observation: dict[str, object] = {
            "phase": "selection",
            "decision_domain": "combat",
            "state_hash": f"state-{selected_count}",
            "semantic_state_hash": f"semantic-{selected_count}",
            "run": {"act": 2, "floor": 27, "room_type": "combat"},
            "combat": {"in_progress": True, "turn": 4},
            "card_selection": {
                "prompt_id": "card.TEST.mandatory_discard",
                "operation_type": "discard",
                "source_zone": "Hand",
                "destination_zone": "Discard",
                "min_select": 2,
                "max_select": 2,
                "selected_count": selected_count,
                "remaining_picks": 2 - selected_count,
                "requires_manual_confirmation": True,
                "can_skip": False,
                "cancelable": False,
                "can_confirm": selected_count == 2,
                "cards": list(selectable),
                "selected_cards": list(selected),
            },
        }
        actions: list[dict[str, object]] = [
            {
                "action": "select_card",
                "kind": "select_card",
                "model_action_kind": "card_selection",
                "model_action_variant": "select",
                "selection_operation": "select",
                "card": card,
            }
            for card in selectable
        ]
        actions.extend(
            {
                "action": "deselect_card",
                "kind": "deselect_card",
                "model_action_kind": "card_selection",
                "model_action_variant": "deselect",
                "selection_operation": "deselect",
                "card": {
                    **card,
                    "is_selected": True,
                    "selection_membership": "selected",
                },
            }
            for card in selected
        )
        if selected_count == 2:
            actions.append(
                {
                    "action": "confirm_selection",
                    "kind": "confirm_selection",
                    "model_action_kind": "card_selection",
                    "model_action_variant": "confirm",
                    "selection_operation": "confirm",
                }
            )
        return observation, tuple(actions)

    one_selected, one_actions = state(1)
    two_selected, two_actions = state(2)
    returned, returned_actions = state(1)
    returned["state_hash"] = "transport-returned"
    returned["semantic_state_hash"] = "transport-semantic-returned"

    surfaces = (
        _transaction_surface_key(one_selected, one_actions),
        _transaction_surface_key(two_selected, two_actions),
        _transaction_surface_key(returned, tuple(reversed(returned_actions))),
    )
    assert surfaces[0] is not None
    assert surfaces[0] == surfaces[1] == surfaces[2]
    first_node = _transaction_node_key(surfaces[0], one_selected, one_actions)
    assert (
        _transaction_node_key(
            surfaces[1],
            two_selected,
            two_actions,
        )
        != first_node
    )
    assert (
        _transaction_node_key(
            surfaces[2],
            returned,
            tuple(reversed(returned_actions)),
        )
        == first_node
    )


def test_transaction_exit_is_not_mislabeled_as_selection_teardown_delta() -> None:
    effect, delta = _classify_transaction_transition(
        current_surface="selection-surface",
        next_surface=None,
        current_node="selected-one",
        next_node="reward-screen",
        seen_nodes={"selected-zero", "selected-one"},
        before_selected_count=1,
        after_selected_count=0,
    )
    assert effect is TransactionEffect.EXIT
    assert delta == 0


def _model_config() -> GroundedCandidateConfig:
    return GroundedCandidateConfig(
        token_feature_dim=224,
        d_model=16,
        n_heads=4,
        ffn_dim=32,
        world_layers=1,
        latent_slots=2,
        latent_layers=1,
        local_layers=1,
        candidate_layers=1,
        recurrent_hidden_dim=32,
        dropout=0.0,
        domain_count=8,
        type_vocab_size=16,
        role_vocab_size=12,
        owner_vocab_size=16,
        entity_vocab_size=64,
        zone_vocab_size=10,
        order_vocab_size=16,
    )


def _snapshot(config: GroundedEncodingConfig) -> EncodedDecisionSnapshot:
    feature_dim = config.feature_dim
    world_feature = tuple([1.0] + [0.0] * (feature_dim - 1))
    candidate_a = tuple([0.0, 1.0] + [0.0] * (feature_dim - 2))
    candidate_b = tuple([0.0, 0.0, 1.0] + [0.0] * (feature_dim - 3))
    return EncodedDecisionSnapshot(
        config=config,
        encoding_fingerprint=grounding_encoding_identity()["fingerprint_sha256"],
        world=sparse_token_table(
            features=(world_feature,),
            ids=((2, 2, 2, 2, 2, 2, 2, 2, 0),),
            feature_dim=feature_dim,
            id_width=9,
        ),
        candidates=sparse_token_table(
            features=(candidate_a, candidate_b),
            ids=(
                (2, 2, 2, 3, 3, 3, 3, 2, 2, 4, 4, 4, 4),
                (3, 3, 2, 4, 4, 4, 4, 2, 2, 3, 3, 3, 3),
            ),
            feature_dim=feature_dim,
            id_width=13,
        ),
        locals=sparse_token_table(
            features=(),
            ids=(),
            feature_dim=feature_dim,
            id_width=9,
        ),
        local_offsets=np.asarray([0, 0, 0], dtype=np.uint32),
        action_mask=np.asarray([True, True], dtype=np.bool_),
        domain_id=1,
    )


def _trace(
    snapshot: EncodedDecisionSnapshot,
    *,
    trace_id: str,
    action_index: int,
    action_fingerprint: str,
    effect: TransactionEffect,
    transaction_return: float | None,
    node_key: str = "same-node",
    partition: str = "training",
    outcome: TransactionOutcome = TransactionOutcome.CENSORED,
) -> TransactionTrace:
    return TransactionTrace(
        trace_id=trace_id,
        episode_id=f"episode-{trace_id}",
        surface_key="selection-surface",
        start_step=10,
        policy_version=3,
        initial_recurrent_state=np.zeros(32, dtype=np.float32),
        steps=(
            TransactionStep(
                snapshot=snapshot,
                action_index=action_index,
                node_key=node_key,
                next_node_key=f"next-{trace_id}",
                action_fingerprint=action_fingerprint,
                effect=effect,
                selected_count_delta=0,
                transaction_return=transaction_return,
                return_steps=1 if transaction_return is not None else None,
            ),
        ),
        outcome=outcome,
        data_partition=partition,
    )


def _sequence_trace(
    snapshot: EncodedDecisionSnapshot,
    *,
    trace_id: str,
    outcome: TransactionOutcome,
    steps: tuple[tuple[str, str, str, int, TransactionEffect], ...],
) -> TransactionTrace:
    return TransactionTrace(
        trace_id=trace_id,
        episode_id=f"episode-{trace_id}",
        surface_key=f"surface-{trace_id}",
        start_step=0,
        policy_version=0,
        initial_recurrent_state=np.zeros(32, dtype=np.float32),
        steps=tuple(
            TransactionStep(
                snapshot=snapshot,
                action_index=action_index,
                node_key=node_key,
                next_node_key=next_node_key,
                action_fingerprint=action_fingerprint,
                effect=effect,
                selected_count_delta=0,
                transaction_return=None,
                return_steps=None,
            )
            for (
                node_key,
                next_node_key,
                action_fingerprint,
                action_index,
                effect,
            ) in steps
        ),
        outcome=outcome,
    )


@pytest.mark.parametrize(
    ("policy_node_key", "policy_action_fingerprint"),
    (("coarse-node", None), (None, "coarse-action")),
)
def test_transaction_liveness_policy_identity_is_an_atomic_pair(
    policy_node_key: str | None,
    policy_action_fingerprint: str | None,
) -> None:
    encoding = GroundedEncodingConfig.from_model_config(
        _model_config(),
        max_world_tokens=8,
        max_candidates=256,
        max_candidate_local_tokens=4,
    )
    step = _trace(
        _snapshot(encoding),
        trace_id="policy-pair",
        action_index=0,
        action_fingerprint="exact-action",
        effect=TransactionEffect.MOVE,
        transaction_return=None,
    ).steps[0]

    with pytest.raises(ValueError, match="both present or both absent"):
        replace(
            step,
            policy_node_key=policy_node_key,
            policy_action_fingerprint=policy_action_fingerprint,
        )


def test_forced_recurrent_deadlock_keeps_value_credit() -> None:
    config = _model_config()
    encoding = GroundedEncodingConfig.from_model_config(
        config,
        max_world_tokens=8,
        max_candidates=256,
        max_candidate_local_tokens=4,
    )
    forced_snapshot = replace(
        _snapshot(encoding),
        action_mask=np.asarray([True, False], dtype=np.bool_),
    )
    trace = _sequence_trace(
        forced_snapshot,
        trace_id="forced-event-cycle",
        outcome=TransactionOutcome.DEADLOCK,
        steps=(
            ("forced-page", "forced-page", "forced:proceed", 0, TransactionEffect.STAY),
            ("forced-page", "forced-page", "forced:proceed", 0, TransactionEffect.STAY),
        ),
    )

    credited = backfill_factual_monte_carlo_returns(
        trace,
        episode_rewards=(-0.1, -1.0),
        episode_discounts=(1.0, 0.0),
        authoritative_outcome=True,
    )
    # The failure remains a factual return/Q target even though the actor had
    # no alternative legal candidate.
    assert all(step.q_observed for step in credited.steps)


def test_transaction_heads_are_opt_in_and_candidate_equivariant() -> None:
    config = _model_config()
    disabled = RecurrentCandidateModel(config)
    assert not any("transaction" in key or "effect_head" in key for key in disabled.state_dict())

    enabled = RecurrentCandidateModel(config, enable_transaction_heads=True).eval()
    encoding = GroundedEncodingConfig.from_model_config(
        config,
        max_world_tokens=8,
        max_candidates=8,
        max_candidate_local_tokens=4,
    )
    encoder = GroundedObservationEncoder(encoding)
    batch = encoder.collate_snapshots((_snapshot(encoding),))
    output = enabled(batch)

    assert output.candidate_effect_logits is not None
    assert output.selection_delta_logits is not None
    assert output.transaction_q_values is not None
    assert output.candidate_effect_logits.shape == (1, 2, 4)
    assert output.selection_delta_logits.shape == (1, 2, 3)
    assert output.transaction_q_values.shape == (1, 2)

    permutation = torch.tensor([1, 0])
    permuted = enabled(batch.permute_candidates(permutation))
    torch.testing.assert_close(
        permuted.candidate_effect_logits,
        output.candidate_effect_logits[:, permutation],
    )
    torch.testing.assert_close(
        permuted.selection_delta_logits,
        output.selection_delta_logits[:, permutation],
    )
    torch.testing.assert_close(
        permuted.transaction_q_values,
        output.transaction_q_values[:, permutation],
    )


def test_replay_is_bounded_checkpointable_and_rejects_heldout() -> None:
    config = _model_config()
    encoding = GroundedEncodingConfig.from_model_config(
        config,
        max_world_tokens=8,
        max_candidates=8,
        max_candidate_local_tokens=4,
    )
    snapshot = _snapshot(encoding)
    first = _trace(
        snapshot,
        trace_id="first",
        action_index=0,
        action_fingerprint="select-a",
        effect=TransactionEffect.EXIT,
        transaction_return=1.0,
    )
    second = _trace(
        snapshot,
        trace_id="second",
        action_index=1,
        action_fingerprint="deselect-a",
        effect=TransactionEffect.REVISIT,
        transaction_return=-1.0,
    )
    with pytest.raises(ValueError, match="held-out"):
        _trace(
            snapshot,
            trace_id="heldout",
            action_index=0,
            action_fingerprint="confirm",
            effect=TransactionEffect.EXIT,
            transaction_return=1.0,
            partition="evaluation",
        )

    replay = BoundedTransactionReplay(
        capacity=1,
        byte_capacity=max(first.storage_nbytes(), second.storage_nbytes()) + 128,
        seed=7,
    )
    assert replay.put(first)
    assert not replay.put(first)
    assert replay.put(second)
    assert replay.snapshot() == (second,)
    assert replay.metrics()["eviction_count"] == 1

    payload = replay.state_dict()
    restored = BoundedTransactionReplay(
        capacity=1,
        byte_capacity=replay.byte_capacity,
        seed=999,
    )
    restored.load_state_dict(payload)
    assert restored.snapshot() == replay.snapshot()
    assert restored.metrics() == replay.metrics()
    assert restored.sample(1)[0].trace_id == "second"

    legacy_payload = dict(payload)
    legacy_payload["version"] = "sts2-transaction-replay-v1"
    with pytest.raises(ValueError, match="unsupported transaction replay"):
        restored.load_state_dict(legacy_payload)

    inconsistent_payload = dict(payload)
    inconsistent_payload["put_count"] += 1
    restored_before = restored.state_dict()
    with pytest.raises(ValueError, match="accounting is inconsistent"):
        restored.load_state_dict(inconsistent_payload)
    assert restored.state_dict() == restored_before


def test_replay_retains_and_samples_sparse_deadlock_traces_first() -> None:
    config = _model_config()
    encoding = GroundedEncodingConfig.from_model_config(
        config,
        max_world_tokens=8,
        max_candidates=8,
        max_candidate_local_tokens=4,
    )
    snapshot = _snapshot(encoding)
    deadlock = _trace(
        snapshot,
        trace_id="deadlock",
        action_index=0,
        action_fingerprint="deselect",
        effect=TransactionEffect.REVISIT,
        transaction_return=-1.0,
        outcome=TransactionOutcome.DEADLOCK,
    )
    completed = tuple(
        _trace(
            snapshot,
            trace_id=f"completed-{index}",
            action_index=index % 2,
            action_fingerprint=f"confirm-{index}",
            effect=TransactionEffect.EXIT,
            transaction_return=1.0,
            outcome=TransactionOutcome.COMPLETED,
        )
        for index in range(3)
    )
    replay = BoundedTransactionReplay(
        capacity=3,
        byte_capacity=10_000_000,
        seed=19,
    )

    assert replay.put(deadlock)
    assert all(replay.put(trace) for trace in completed)
    retained_ids = {trace.trace_id for trace in replay.snapshot()}
    assert "deadlock" in retained_ids
    assert "completed-0" not in retained_ids
    assert replay.metrics()["deadlock_size"] == 1

    for _ in range(8):
        sampled_ids = {trace.trace_id for trace in replay.sample(2)}
        assert "deadlock" in sampled_ids


def test_replay_reserves_one_committed_lifecycle_per_operation_after_restore() -> None:
    config = _model_config()
    encoding = GroundedEncodingConfig.from_model_config(
        config,
        max_world_tokens=8,
        max_candidates=8,
        max_candidate_local_tokens=4,
    )
    snapshot = _snapshot(encoding)

    def lifecycle_trace(operation: str) -> TransactionTrace:
        action_fingerprint = f"entry:{operation}"
        trace = _sequence_trace(
            snapshot,
            trace_id=f"lifecycle-{operation}",
            outcome=TransactionOutcome.COMPLETED,
            steps=(("entry", "exit", action_fingerprint, 0, TransactionEffect.EXIT),),
        )
        return replace(
            trace,
            lifecycle=TransactionLifecycleEvidence(
                operation=operation,
                entry_step_index=0,
                exit_step_index=0,
                entry_behavior_log_probability=math.log(0.5),
                entry_model_probability=0.0,
                entry_policy_version=0,
                entry_action_fingerprint=action_fingerprint,
                outcome=TransactionLifecycleOutcome.COMMITTED,
                effect_verified=True,
                post_terminal=True,
                option_return=1.0,
                option_discount=0.0,
                option_steps=1,
                option_boundary="transaction_exit",
            ),
        )

    replay = BoundedTransactionReplay(
        capacity=16,
        byte_capacity=50_000_000,
        seed=31,
    )
    for index in range(8):
        assert replay.put(
            _trace(
                snapshot,
                trace_id=f"ordinary-lifecycle-control-{index}",
                action_index=index % 2,
                action_fingerprint=f"ordinary:{index}",
                effect=TransactionEffect.EXIT,
                transaction_return=1.0,
                outcome=TransactionOutcome.COMPLETED,
            )
        )
    assert replay.put(lifecycle_trace("upgrade"))
    assert replay.put(lifecycle_trace("remove"))
    assert replay.put(lifecycle_trace("rest"))
    assert replay.metrics()["committed_lifecycle_size"] == 3
    assert replay.metrics()["committed_upgrade_lifecycle_size"] == 1
    assert replay.metrics()["committed_remove_lifecycle_size"] == 1
    assert replay.metrics()["committed_rest_lifecycle_size"] == 1
    assert {trace.lifecycle.operation for trace in replay.sample(3) if trace.lifecycle} == {
        "upgrade",
        "remove",
        "rest",
    }

    restored = BoundedTransactionReplay(
        capacity=replay.capacity,
        byte_capacity=replay.byte_capacity,
        seed=999,
    )
    restored.load_state_dict(replay.state_dict())
    assert restored.metrics() == replay.metrics()
    assert {
        trace.lifecycle.operation
        for trace in restored.sample(3)
        if trace.lifecycle is not None
    } == {"upgrade", "remove", "rest"}


def test_replay_restore_preserves_exact_sampling_stream_after_churn() -> None:
    config = _model_config()
    encoding = GroundedEncodingConfig.from_model_config(
        config,
        max_world_tokens=8,
        max_candidates=8,
        max_candidate_local_tokens=4,
    )
    snapshot = _snapshot(encoding)
    templates = tuple(
        _sequence_trace(
            snapshot,
            trace_id=f"template-{index}",
            outcome=(TransactionOutcome.DEADLOCK if index == 0 else TransactionOutcome.COMPLETED),
            steps=(
                ("empty", "one", f"select:{index}", 0, TransactionEffect.MOVE),
                ("one", "exit", "confirm", 1, TransactionEffect.EXIT),
            ),
        )
        for index in range(3)
    )

    # Reconstructing derived committed-lifecycle sets after eviction must not
    # change the mapping from checkpointed RNG indices to replay items, so a
    # restored replay must reproduce the exact sampling stream.
    churned = BoundedTransactionReplay(
        capacity=16,
        byte_capacity=50_000_000,
        seed=31,
    )
    for index in range(100):
        template = templates[index % len(templates)]
        assert churned.put(
            replace(
                template,
                trace_id=f"churn-{index}",
                episode_id=f"episode-churn-{index}",
                surface_key=f"surface-churn-{index}",
            )
        )
    assert churned.metrics()["eviction_count"] == 84
    churned_restored = BoundedTransactionReplay(
        capacity=churned.capacity,
        byte_capacity=churned.byte_capacity,
        seed=999,
    )
    churned_restored.load_state_dict(churned.state_dict())
    for _ in range(12):
        expected_ids = tuple(trace.trace_id for trace in churned.sample(8))
        restored_ids = tuple(trace.trace_id for trace in churned_restored.sample(8))
        assert restored_ids == expected_ids


def test_replay_load_refuses_retired_v7_protection_counter_payload() -> None:
    config = _model_config()
    encoding = GroundedEncodingConfig.from_model_config(
        config,
        max_world_tokens=8,
        max_candidates=8,
        max_candidate_local_tokens=4,
    )
    snapshot = _snapshot(encoding)
    replay = BoundedTransactionReplay(
        capacity=4,
        byte_capacity=10_000_000,
        seed=7,
    )
    assert replay.put(
        _trace(
            snapshot,
            trace_id="live",
            action_index=0,
            action_fingerprint="confirm",
            effect=TransactionEffect.EXIT,
            transaction_return=1.0,
            outcome=TransactionOutcome.COMPLETED,
        )
    )
    payload = replay.state_dict()
    assert payload["version"] == "sts2-transaction-replay-v8"

    target = BoundedTransactionReplay(
        capacity=replay.capacity,
        byte_capacity=replay.byte_capacity,
        seed=99,
    )
    # A pre-v8 payload carries the retired replay version string; it must
    # fail closed rather than be reinterpreted under the new protection
    # semantics.
    legacy_version = {**payload, "version": "sts2-transaction-replay-v7"}
    with pytest.raises(ValueError, match="unsupported transaction replay checkpoint schema"):
        target.load_state_dict(legacy_version)
    # Extra retired counter keys change the payload shape and must also fail
    # closed instead of being silently ignored.
    legacy_shape = {**payload, "actionable_avoid_trace_ids": ("live",)}
    with pytest.raises(ValueError, match="unsupported transaction replay checkpoint schema"):
        target.load_state_dict(legacy_shape)
    target.load_state_dict(payload)
    assert {trace.trace_id for trace in target.snapshot()} == {"live"}


def test_pairwise_labels_require_same_node_distinct_factual_actions() -> None:
    config = _model_config()
    encoding = GroundedEncodingConfig.from_model_config(
        config,
        max_world_tokens=8,
        max_candidates=8,
        max_candidate_local_tokens=4,
    )
    snapshot = _snapshot(encoding)
    better = _trace(
        snapshot,
        trace_id="better",
        action_index=0,
        action_fingerprint="actual-a",
        effect=TransactionEffect.EXIT,
        transaction_return=2.0,
    )
    worse = _trace(
        snapshot,
        trace_id="worse",
        action_index=1,
        action_fingerprint="actual-b",
        effect=TransactionEffect.REVISIT,
        transaction_return=-1.0,
    )
    censored = _trace(
        snapshot,
        trace_id="censored",
        action_index=1,
        action_fingerprint="actual-c",
        effect=TransactionEffect.MOVE,
        transaction_return=None,
    )
    other_node = _trace(
        snapshot,
        trace_id="other-node",
        action_index=1,
        action_fingerprint="actual-d",
        effect=TransactionEffect.EXIT,
        transaction_return=10.0,
        node_key="different-node",
    )

    pairs = observed_outcome_pairs((better, worse, censored, other_node))
    assert len(pairs) == 1
    assert pairs[0].better_trace == 0
    assert pairs[0].worse_trace == 1


def test_factual_mc_backfill_masks_infrastructure_aborts() -> None:
    config = _model_config()
    encoding = GroundedEncodingConfig.from_model_config(
        config,
        max_world_tokens=8,
        max_candidates=8,
        max_candidate_local_tokens=4,
    )
    pending = _trace(
        _snapshot(encoding),
        trace_id="pending",
        action_index=0,
        action_fingerprint="factual-action",
        effect=TransactionEffect.EXIT,
        transaction_return=None,
    )
    rewards = (0.0,) * 10 + (0.25, 1.0)
    discounts = (0.9,) * 11 + (0.0,)

    completed = backfill_factual_monte_carlo_returns(
        pending,
        episode_rewards=rewards,
        episode_discounts=discounts,
        authoritative_outcome=True,
    )
    assert completed.steps[0].transaction_return == pytest.approx(1.15)
    assert completed.steps[0].return_steps == 2

    infrastructure_abort = backfill_factual_monte_carlo_returns(
        completed,
        episode_rewards=rewards,
        episode_discounts=discounts,
        authoritative_outcome=False,
    )
    assert infrastructure_abort.steps[0].transaction_return is None
    assert infrastructure_abort.steps[0].return_steps is None


def test_extended_lifecycle_option_uses_factual_next_macro_boundary_without_bootstrap() -> None:
    encoding = GroundedEncodingConfig.from_model_config(
        _model_config(),
        max_world_tokens=8,
        max_candidates=8,
        max_candidate_local_tokens=4,
    )
    snapshot = _snapshot(encoding)
    trace = _sequence_trace(
        snapshot,
        trace_id="extended-option",
        outcome=TransactionOutcome.COMPLETED,
        steps=(("entry", "exit", "smith", 0, TransactionEffect.EXIT),),
    )
    trace = replace(
        trace,
        lifecycle=TransactionLifecycleEvidence(
            operation="upgrade",
            entry_step_index=0,
            exit_step_index=0,
            entry_behavior_log_probability=math.log(0.5),
            entry_model_probability=0.5,
            entry_policy_version=0,
            entry_action_fingerprint="smith",
            outcome=TransactionLifecycleOutcome.COMMITTED,
            effect_verified=True,
            post_snapshot=snapshot,
        ),
    )
    rewards = (0.1, 0.2, 0.3, 0.4, 9.0)
    discounts = (1.0, 1.0, 1.0, 1.0, 0.0)

    extended = backfill_factual_monte_carlo_returns(
        trace,
        episode_rewards=rewards,
        episode_discounts=discounts,
        authoritative_outcome=True,
        option_horizon_end=3,
        option_horizon_boundary="next_rest_site",
        require_extended_option_horizon=True,
    )
    assert extended.lifecycle is not None
    assert extended.lifecycle.option_return == pytest.approx(1.0)
    assert extended.lifecycle.option_discount == 0.0
    assert extended.lifecycle.option_steps == 4
    assert extended.lifecycle.option_boundary == "next_rest_site"

    censored_before_boundary = backfill_factual_monte_carlo_returns(
        trace,
        episode_rewards=rewards,
        episode_discounts=discounts,
        authoritative_outcome=False,
        require_extended_option_horizon=True,
    )
    assert censored_before_boundary.lifecycle is not None
    assert not censored_before_boundary.lifecycle.option_target_observed


def test_selection_option_targets_start_at_each_selected_card_not_macro_entry() -> None:
    encoding = GroundedEncodingConfig.from_model_config(
        _model_config(),
        max_world_tokens=8,
        max_candidates=8,
        max_candidate_local_tokens=4,
    )
    snapshot = _snapshot(encoding)
    trace = _sequence_trace(
        snapshot,
        trace_id="selection-origin-option",
        outcome=TransactionOutcome.COMPLETED,
        steps=(
            ("rest-entry", "grid-empty", "smith", 0, TransactionEffect.MOVE),
            ("grid-empty", "grid-a", "select:a", 0, TransactionEffect.MOVE),
            ("grid-a", "grid-empty", "deselect:a", 1, TransactionEffect.REVISIT),
            ("grid-empty", "grid-b", "select:b", 1, TransactionEffect.MOVE),
            ("grid-b", "exit", "confirm", 0, TransactionEffect.EXIT),
        ),
    )
    trace = replace(
        trace,
        steps=tuple(
            replace(
                step,
                selected_count_delta=(1 if index in {1, 3} else -1 if index == 2 else 0),
                behavior_log_probability=math.log(0.5),
                model_probability=0.5,
            )
            for index, step in enumerate(trace.steps)
        ),
        lifecycle=TransactionLifecycleEvidence(
            operation="upgrade",
            entry_step_index=0,
            exit_step_index=4,
            entry_behavior_log_probability=math.log(0.5),
            entry_model_probability=0.5,
            entry_policy_version=0,
            entry_action_fingerprint="smith",
            outcome=TransactionLifecycleOutcome.COMMITTED,
            effect_verified=True,
            post_snapshot=snapshot,
        ),
    )
    rewards = (1.0, 2.0, 4.0, 8.0, 16.0, 32.0)
    discounts = (1.0, 1.0, 1.0, 1.0, 1.0, 0.0)

    credited = backfill_factual_monte_carlo_returns(
        trace,
        episode_rewards=rewards,
        episode_discounts=discounts,
        authoritative_outcome=True,
        option_horizon_end=5,
        option_horizon_boundary="next_rest_site",
        require_extended_option_horizon=True,
    )

    assert credited.lifecycle is not None
    assert credited.lifecycle.option_return == pytest.approx(63.0)
    assert credited.lifecycle.option_steps == 6
    first_selection = credited.steps[1]
    second_selection = credited.steps[3]
    assert first_selection.option_return == pytest.approx(62.0)
    assert first_selection.option_steps == 5
    assert second_selection.option_return == pytest.approx(56.0)
    assert second_selection.option_steps == 3
    assert first_selection.option_return != credited.lifecycle.option_return
    assert second_selection.option_return != first_selection.option_return
    assert credited.steps[0].option_return is None
    assert credited.steps[2].option_return is None
    assert credited.steps[4].option_return is None



def test_transaction_burn_in_recomputes_context_and_excludes_it_from_labels() -> None:
    model_config = _model_config()
    encoding_config = GroundedEncodingConfig.from_model_config(
        model_config,
        max_world_tokens=8,
        max_candidates=8,
        max_candidate_local_tokens=4,
    )
    snapshot = _snapshot(encoding_config)
    model = RecurrentCandidateModel(model_config, enable_transaction_heads=True)
    learner = VTraceLearner(
        model=model,
        encoder=GroundedObservationEncoder(encoding_config),
        optimizer=torch.optim.AdamW(model.parameters(), lr=1.0e-3),
        config=TrainingConfig().optimization,
        maximum_unroll_length=4,
        maximum_policy_lag=10,
        transaction_config=TransactionLearningConfig(
            enabled=True,
            replay_byte_capacity=10_000_000,
            burn_in_steps=1,
        ),
    )
    context = TransactionStep(
        snapshot=snapshot,
        action_index=0,
        node_key="context-node",
        next_node_key="transaction-node",
        action_fingerprint="context-factual-action",
        effect=TransactionEffect.MOVE,
        selected_count_delta=0,
        transaction_return=None,
        return_steps=None,
    )
    factual = TransactionStep(
        snapshot=snapshot,
        action_index=1,
        node_key="transaction-node",
        next_node_key="exited",
        action_fingerprint="transaction-factual-action",
        effect=TransactionEffect.EXIT,
        selected_count_delta=0,
        transaction_return=1.0,
        return_steps=1,
    )
    trace = TransactionTrace(
        trace_id="burn-in",
        episode_id="burn-in-episode",
        surface_key="burn-in-surface",
        start_step=7,
        policy_version=0,
        initial_recurrent_state=np.zeros(32, dtype=np.float32),
        steps=(context, factual),
        burn_in_steps=1,
    )
    forward_calls = 0

    def count_forward(_module: torch.nn.Module, _inputs: object, _output: object) -> None:
        nonlocal forward_calls
        forward_calls += 1

    handle = model.register_forward_hook(count_forward)
    try:
        losses = learner._transaction_losses((trace,))
    finally:
        handle.remove()

    assert forward_calls == 2
    assert losses.effect_labels == 1
    assert losses.q_labels == 1
    with pytest.raises(ValueError, match="zero initial state"):
        replace(
            trace,
            initial_recurrent_state=np.ones(32, dtype=np.float32),
        )


def test_learner_uses_factual_heads_and_pairwise_targets() -> None:
    model_config = _model_config()
    encoding_config = GroundedEncodingConfig.from_model_config(
        model_config,
        max_world_tokens=8,
        max_candidates=8,
        max_candidate_local_tokens=4,
    )
    snapshot = _snapshot(encoding_config)
    model = RecurrentCandidateModel(model_config, enable_transaction_heads=True)
    encoder = GroundedObservationEncoder(encoding_config)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1.0e-3)
    transaction_config = TransactionLearningConfig(
        enabled=True,
        replay_byte_capacity=10_000_000,
        burn_in_steps=0,
        pairwise_ranking_weight=0.10,
    )
    learner = VTraceLearner(
        model=model,
        encoder=encoder,
        optimizer=optimizer,
        config=TrainingConfig().optimization,
        maximum_unroll_length=4,
        maximum_policy_lag=10,
        transaction_config=transaction_config,
    )
    # A zero-burn lineage is intentional and valid; it must not silently
    # inherit the default burn-in requirement from another configuration.
    assert transaction_config.burn_in_steps == 0
    unroll = SequenceUnroll(
        episode_id="main-vtrace",
        start_step=0,
        policy_version=0,
        initial_recurrent_state=np.zeros(32, dtype=np.float32),
        steps=(
            RolloutStep(
                snapshot=snapshot,
                action_index=0,
                behavior_log_probability=math.log(0.5),
                reward=0.0,
                discount=0.0,
                policy_decision=True,
            ),
        ),
        bootstrap_snapshot=None,
    )
    better = _trace(
        snapshot,
        trace_id="better",
        action_index=0,
        action_fingerprint="observed-a",
        effect=TransactionEffect.EXIT,
        transaction_return=1.0,
        outcome=TransactionOutcome.COMPLETED,
    )
    worse = _trace(
        snapshot,
        trace_id="worse",
        action_index=1,
        action_fingerprint="observed-b",
        effect=TransactionEffect.REVISIT,
        transaction_return=-1.0,
    )
    censored = _trace(
        snapshot,
        trace_id="censored",
        action_index=1,
        action_fingerprint="observed-c",
        effect=TransactionEffect.MOVE,
        transaction_return=None,
        node_key="censored-node",
    )

    metrics = learner.update(
        (unroll,),
        current_policy_version=0,
        current_learner_update=0,
        transaction_traces=(better, worse, censored),
    )

    assert metrics.transaction_traces == 3
    assert metrics.transaction_effect_labels == 3
    assert metrics.transaction_q_labels == 2
    assert metrics.transaction_pairs == 1
    assert metrics.transaction_effect_loss > 0.0
    assert metrics.transaction_delta_loss > 0.0
    assert metrics.transaction_q_loss > 0.0
    assert metrics.transaction_pairwise_ranking_loss > 0.0


def test_training_collector_emits_factual_trace_but_evaluation_does_not() -> None:
    from tests.test_v2_training_pipeline import (  # local import avoids fixture coupling
        DynamicPreviewLoopBackend,
        _event_loop_config,
    )

    base = _event_loop_config(durable_window=4)
    config = replace(
        base,
        transaction_learning=TransactionLearningConfig(
            enabled=True,
            replay_capacity=8,
            replay_byte_capacity=10_000_000,
            sample_traces=2,
            burn_in_steps=2,
        ),
    )
    resources = build_training_resources(
        config,
        backend=DynamicPreviewLoopBackend(alternate_ui_actions=True),
    )
    try:
        training_episode = resources.collector.collect_episode(record=True)
        assert training_episode.metrics.noncombat_progress_stalled
        assert not training_episode.metrics.trusted_policy_failure
        assert len(training_episode.transaction_traces) == 1
        trace = training_episode.transaction_traces[0]
        assert trace.data_partition == "training"
        assert trace.start_step == 0
        assert trace.burn_in_steps == 0
        assert trace.outcome is TransactionOutcome.CENSORED
        assert all(not step.q_observed for step in trace.learn_steps)

        evaluation_episode = resources.collector.collect_episode(
            record=False,
            evaluation_seed=9,
            deterministic=True,
        )
        assert evaluation_episode.transaction_traces == ()
    finally:
        resources.close()


def test_training_only_greedy_liveness_probe_records_delta_behavior_without_false_action_credit() -> None:
    from tests.test_v2_training_pipeline import (
        DynamicPreviewLoopBackend,
        _event_loop_config,
    )

    class MultiActionLoopBackend(DynamicPreviewLoopBackend):
        def _actions(self) -> tuple[dict[str, object], ...]:
            return (
                {
                    "action_handle": f"hold-a:{self._step}",
                    "action": "choose_event_option",
                    "kind": "choose_event_option",
                    "model_action_kind": "event_option",
                    "index": 0,
                    "label": "EVENT.LOOP.options.HOLD_A",
                },
                {
                    "action_handle": f"hold-b:{self._step}",
                    "action": "choose_event_option",
                    "kind": "choose_event_option",
                    "model_action_kind": "event_option",
                    "index": 1,
                    "label": "EVENT.LOOP.options.HOLD_B",
                },
            )

    base = _event_loop_config(durable_window=40)
    config = replace(
        base,
        environment=replace(base.environment, max_episode_steps=80),
        transaction_learning=TransactionLearningConfig(
            enabled=True,
            replay_capacity=8,
            replay_byte_capacity=10_000_000,
            sample_traces=2,
            burn_in_steps=4,
        ),
    )
    resources = build_training_resources(
        config,
        backend=MultiActionLoopBackend(),
    )
    try:
        episode = resources.collector.collect_episode(
            # The explicit probe contract overrides ordinary exploration.
            epsilon=1.0,
            deterministic=False,
            record=True,
            liveness_probe=True,
        )
        assert episode.metrics.reset_seed % 2 == 0
        assert episode.metrics.deadlocked
        assert not episode.metrics.trusted_policy_failure
        assert episode.transaction_traces
        trace = episode.transaction_traces[-1]
        assert trace.outcome is TransactionOutcome.CENSORED
        assert len(trace.learn_steps) <= 32
        assert len(trace.steps) <= 36
        rollout_steps = tuple(step for unroll in episode.unrolls for step in unroll.steps)
        assert all(step.behavior_log_probability == pytest.approx(0.0) for step in rollout_steps)
        assert any(step.policy_decision for step in rollout_steps[:-1])
        assert rollout_steps[-1].discount == 0.0
        assert rollout_steps[-1].policy_decision is False
        # A training probe still records the bounded diagnostic trace, but a
        # unique MOVE-only window supplies no factual Q authority.
        assert all(step.effect is TransactionEffect.MOVE for step in trace.learn_steps)
        assert all(not step.q_observed for step in trace.learn_steps)

        with pytest.raises(ValueError, match="recorded training"):
            resources.collector.collect_episode(
                record=False,
                liveness_probe=True,
            )
        with pytest.raises(ValueError, match="held-out"):
            resources.collector.collect_episode(
                record=True,
                evaluation_seed=9,
                liveness_probe=True,
            )
    finally:
        resources.close()


def test_unique_liveness_trace_keeps_value_prefix_without_invented_policy_blame() -> None:
    from tests.test_v2_training_pipeline import (
        DynamicPreviewLoopBackend,
        _event_loop_config,
    )

    class PolicyChoiceThenForcedLoopBackend(DynamicPreviewLoopBackend):
        def _actions(self) -> tuple[dict[str, object], ...]:
            primary = super()._actions()
            if self._step:
                return primary
            return (
                primary[0],
                {
                    "action_handle": "alternate:0",
                    "action": "choose_event_option",
                    "kind": "choose_event_option",
                    "model_action_kind": "event_option",
                    "index": 2,
                    "label": "EVENT.LOOP.options.ALTERNATE",
                },
            )

    base = _event_loop_config(durable_window=40)
    config = replace(
        base,
        environment=replace(base.environment, max_episode_steps=80),
        transaction_learning=TransactionLearningConfig(
            enabled=True,
            replay_capacity=8,
            replay_byte_capacity=10_000_000,
            sample_traces=2,
            burn_in_steps=4,
        ),
    )
    resources = build_training_resources(
        config,
        backend=PolicyChoiceThenForcedLoopBackend(),
    )
    try:
        episode = resources.collector.collect_episode(
            record=True,
            deterministic=True,
        )
        assert episode.metrics.noncombat_progress_stalled
        assert not episode.metrics.trusted_policy_failure
        trace = episode.transaction_traces[-1]
        assert trace.outcome is TransactionOutcome.CENSORED
        assert trace.start_step == 0
        assert trace.burn_in_steps == 0
        assert len(trace.steps) == 1
        assert np.count_nonzero(trace.steps[0].snapshot.action_mask) == 2
        # This preserves the last learnable prefix for value/Q reconstruction;
        # MOVE-only delayed-window evidence carries no factual Q authority.
        assert all(not step.q_observed for step in trace.learn_steps)
    finally:
        resources.close()


def test_long_selection_failure_trace_is_bounded_to_burn_in_plus_learning_tail() -> None:
    from tests.test_v2_training_pipeline import (
        DynamicPreviewLoopBackend,
        _event_loop_config,
    )

    class MultiActionSelectionLoopBackend(DynamicPreviewLoopBackend):
        def _actions(self) -> tuple[dict[str, object], ...]:
            primary = super()._actions()
            return (
                *primary,
                {
                    "action_handle": f"alternate:{self._step}",
                    "action": "cancel_selection",
                    "kind": "cancel_selection",
                    "model_action_kind": "card_selection",
                    "model_action_variant": "cancel_prompt",
                    "selection_operation": "cancel_prompt",
                },
            )

    base = _event_loop_config(durable_window=60)
    config = replace(
        base,
        environment=replace(base.environment, max_episode_steps=100),
        transaction_learning=TransactionLearningConfig(
            enabled=True,
            replay_capacity=8,
            replay_byte_capacity=20_000_000,
            sample_traces=2,
            burn_in_steps=4,
        ),
    )
    resources = build_training_resources(
        config,
        backend=MultiActionSelectionLoopBackend(alternate_ui_actions=True),
    )
    try:
        episode = resources.collector.collect_episode(
            record=True,
            deterministic=True,
        )
        assert episode.metrics.noncombat_progress_stalled
        assert not episode.metrics.trusted_policy_failure
        assert len(episode.transaction_traces) == 1
        trace = episode.transaction_traces[0]
        assert trace.outcome is TransactionOutcome.CENSORED
        assert trace.burn_in_steps <= 4
        assert len(trace.learn_steps) <= 32
        assert len(trace.steps) <= 36
        # The bounded trace is still retained without factual Q authority.
        assert all(not step.q_observed for step in trace.learn_steps)
    finally:
        resources.close()


def test_external_policy_burn_in_does_not_make_forced_selection_cycle_authoritative() -> None:
    from tests.test_v2_training_pipeline import (  # local import avoids fixture coupling
        DynamicPreviewLoopBackend,
        _event_loop_config,
    )

    class ExternalChoiceThenForcedSelectionLoopBackend(DynamicPreviewLoopBackend):
        """One real choice precedes an unrelated, entirely forced selection loop."""

        def _actions(self) -> tuple[dict[str, object], ...]:
            if self._step == 0:
                return (
                    {
                        "action_handle": "enter-selection:0",
                        "action": "choose_event_option",
                        "kind": "choose_event_option",
                        "model_action_kind": "event_option",
                        "transport_kind": "event_option",
                        "index": 0,
                    },
                    {
                        "action_handle": "alternate:0",
                        "action": "choose_event_option",
                        "kind": "choose_event_option",
                        "model_action_kind": "event_option",
                        "transport_kind": "event_option",
                        "index": 1,
                    },
                )
            selected = bool((self._step - 1) % 2)
            return (
                {
                    "action_handle": f"forced-toggle:{self._step}",
                    "action": "deselect_card" if selected else "select_card",
                    "kind": "deselect_card" if selected else "select_card",
                    "model_action_kind": "card_selection",
                    "model_action_variant": "deselect" if selected else "select",
                    "selection_operation": "deselect" if selected else "select",
                    "card": {
                        "id": "CARD.STRIKE",
                        "instance_id": "forced-strike",
                        "source_pile": "Discard",
                        "selection_membership": "selected" if selected else "selectable",
                        "is_selected": selected,
                    },
                },
            )

        def _observation(self, *, terminal: bool = False) -> dict[str, object]:
            observation = super()._observation(terminal=terminal)
            observation["phase"] = "event" if self._step == 0 else "selection"
            observation["state_type"] = "event" if self._step == 0 else "card_select"
            observation["screen"] = "EVENT" if self._step == 0 else "SELECTION"
            event = observation["event"]
            assert isinstance(event, dict)
            event["description_key"] = "EVENT.FORCED_SELECTION.pages.LOOP"
            event["dynamic_vars"] = []
            if self._step == 0:
                observation.pop("card_selection", None)
                return observation

            selected = bool((self._step - 1) % 2)
            card = {
                "id": "CARD.STRIKE",
                "instance_id": "forced-strike",
                "source_pile": "Discard",
            }
            observation["card_selection"] = {
                "mode": "SimpleGrid",
                "prompt_id": "event.FORCED_SELECTION.selection",
                "operation_type": "select",
                "source_zone": "Discard",
                "min_select": 0,
                "max_select": 1,
                "cards": [card],
                "selected_cards": [card] if selected else [],
                "selected_count": int(selected),
                "remaining_picks": 1 - int(selected),
                "requires_manual_confirmation": False,
                "can_confirm": False,
            }
            return observation

    base = _event_loop_config(durable_window=40)
    config = replace(
        base,
        environment=replace(base.environment, max_episode_steps=40),
        diagnostics=replace(
            base.diagnostics,
            deadlock_window=8,
            deadlock_repeat_threshold=3,
            noncombat_durable_progress_window=40,
        ),
        transaction_learning=TransactionLearningConfig(
            enabled=True,
            replay_capacity=8,
            replay_byte_capacity=10_000_000,
            sample_traces=2,
            burn_in_steps=4,
        ),
        episodic_learning=replace(base.episodic_learning, enabled=True),
    )
    resources = build_training_resources(
        config,
        backend=ExternalChoiceThenForcedSelectionLoopBackend(),
    )
    try:
        with torch.no_grad():
            for parameter in resources.model.parameters():
                parameter.zero_()
        episode = resources.collector.collect_episode(record=True, deterministic=True)

        assert episode.metrics.deadlocked
        assert episode.metrics.terminal_reason == "semantic_deadlock"
        assert not episode.metrics.selection_action_cycle
        assert not episode.metrics.trusted_policy_failure
        completed = episode.completed_episode
        assert completed is not None
        assert not completed.completion.authoritative
        assert completed.completion.won is None

        assert len(episode.transaction_traces) == 1
        trace = episode.transaction_traces[0]
        assert trace.burn_in_steps == 1
        assert np.count_nonzero(trace.steps[0].snapshot.action_mask) == 2
        assert all(np.count_nonzero(step.snapshot.action_mask) == 1 for step in trace.learn_steps)
        assert trace.outcome is TransactionOutcome.CENSORED
        assert all(not step.q_observed for step in trace.steps)
    finally:
        resources.close()


@pytest.mark.parametrize("novel_exit", [False, True])
def test_post_step_deadlock_agrees_with_transaction_exit_outcome(novel_exit: bool) -> None:
    from tests.test_v2_training_pipeline import (  # local import avoids fixture coupling
        TerminalWithoutObservationFlagsBackend,
        _event_loop_config,
    )

    class SelectionCycleThenExitBackend(TerminalWithoutObservationFlagsBackend):
        def __init__(self, *, exit_at_threshold: bool) -> None:
            super().__init__(terminal_step=6)
            self.exit_at_threshold = exit_at_threshold

        def _exited(self) -> bool:
            return bool(self.exit_at_threshold and self._step >= 5)

        def _actions(self) -> tuple[dict[str, object], ...]:
            if self._exited():
                return (
                    {
                        "action": "choose_event_option",
                        "kind": "choose_event_option",
                        "model_action_kind": "event_option",
                        "index": 0,
                        "label": "EVENT.AFTER_SELECTION.options.CONTINUE",
                    },
                )
            selected = bool(self._step % 2)
            return (
                {
                    "action": "deselect_card" if selected else "select_card",
                    "kind": "deselect_card" if selected else "select_card",
                    "model_action_kind": "card_selection",
                    "model_action_variant": "deselect" if selected else "select",
                    "card": {
                        "id": "CARD.STRIKE",
                        "instance_id": "strike-1",
                        "source_pile": "Discard",
                        "selection_membership": "selected" if selected else "selectable",
                        "is_selected": selected,
                    },
                },
                {
                    "action": "select_card",
                    "kind": "select_card",
                    "model_action_kind": "card_selection",
                    "model_action_variant": "select",
                    "card": {
                        "id": "CARD.DEFEND",
                        "instance_id": "defend-1",
                        "source_pile": "Discard",
                        "selection_membership": "selectable",
                        "is_selected": False,
                    },
                },
            )

        def _observation(self, *, terminal: bool = False) -> dict[str, object]:
            player: dict[str, object] = {
                "character": "IRONCLAD",
                "hp": 33,
                "max_hp": 67,
                "gold": 99,
                "open_potion_slots": 2,
                "deck": [
                    {"id": "CARD.STRIKE", "is_upgraded": False},
                    {"id": "CARD.DEFEND", "is_upgraded": False},
                ],
                "relics": [],
                "potions": [],
            }
            observation: dict[str, object] = {
                "phase": "event" if self._exited() else "selection",
                "decision_domain": "build",
                "state_type": "event" if self._exited() else "card_select",
                "screen": "EVENT" if self._exited() else "SELECTION",
                "player": player,
                "combat": {"in_progress": False, "enemies": []},
                "run": {
                    "active": not terminal,
                    "act": 1,
                    "floor": 9,
                    "room_type": "event",
                    "room_model_id": "EVENT.TRANSACTION_TEST",
                },
            }
            if self._exited():
                observation["event"] = {
                    "event_id": "EVENT.AFTER_SELECTION",
                    "description_key": "EVENT.AFTER_SELECTION.pages.DONE",
                    "is_finished": False,
                    "dynamic_vars": [],
                    "options": [
                        {
                            "index": 0,
                            "text_key": "EVENT.AFTER_SELECTION.options.CONTINUE",
                            "is_locked": False,
                            "is_chosen": False,
                            "is_proceed": True,
                        }
                    ],
                }
                return observation

            selected = bool(self._step % 2)
            card = {
                "id": "CARD.STRIKE",
                "instance_id": "strike-1",
                "source_pile": "Discard",
            }
            alternative = {
                "id": "CARD.DEFEND",
                "instance_id": "defend-1",
                "source_pile": "Discard",
            }
            observation["card_selection"] = {
                "mode": "SimpleGrid",
                "prompt_id": "card.TRANSACTION_TEST.selection",
                "operation_type": "select",
                "source_zone": "Discard",
                "min_select": 2,
                "max_select": 2,
                "cards": [alternative] if selected else [card, alternative],
                "selected_cards": [card] if selected else [],
                "selected_count": int(selected),
                "remaining_picks": 2 - int(selected),
                "requires_manual_confirmation": False,
                "can_confirm": False,
            }
            return observation

    base = _event_loop_config(durable_window=20)
    config = replace(
        base,
        diagnostics=replace(
            base.diagnostics,
            deadlock_window=8,
            deadlock_repeat_threshold=3,
            noncombat_durable_progress_window=20,
        ),
        transaction_learning=TransactionLearningConfig(
            enabled=True,
            replay_capacity=8,
            replay_byte_capacity=10_000_000,
            sample_traces=2,
            burn_in_steps=0,
        ),
    )
    resources = build_training_resources(
        config,
        backend=SelectionCycleThenExitBackend(exit_at_threshold=novel_exit),
    )
    try:
        with torch.no_grad():
            for parameter in resources.model.parameters():
                parameter.zero_()
        episode = resources.collector.collect_episode(record=True, deterministic=True)
        assert len(episode.transaction_traces) == 1
        trace = episode.transaction_traces[0]
        if novel_exit:
            assert episode.metrics.steps == 6
            assert episode.metrics.run_won
            assert not episode.metrics.deadlocked
            assert episode.metrics.terminal_reason == "run_victory"
            assert trace.outcome is TransactionOutcome.COMPLETED
            assert trace.steps[-1].effect is TransactionEffect.EXIT
        else:
            assert episode.metrics.steps == 5
            assert not episode.metrics.run_won
            assert episode.metrics.deadlocked
            assert episode.metrics.terminal_reason == "selection_action_cycle"
            assert episode.metrics.selection_action_cycle
            assert episode.metrics.trusted_policy_failure
            assert trace.outcome is TransactionOutcome.DEADLOCK
            assert trace.steps[-1].effect is TransactionEffect.REVISIT
    finally:
        resources.close()


def test_model_initialization_adds_only_new_heads_and_replay_roundtrips(
    tmp_path: Path,
) -> None:
    base = TrainingConfig()
    base = replace(
        base,
        model=replace(
            base.model,
            d_model=16,
            n_heads=4,
            ffn_dim=32,
            world_layers=1,
            latent_slots=2,
            latent_layers=1,
            local_layers=1,
            candidate_layers=1,
            recurrent_hidden_dim=32,
        ),
        runtime=replace(base.runtime, device="cpu", collector_device="cpu"),
    )
    source = build_training_resources(
        base,
        backend=cast(EnvironmentBackend, _NoopBackend()),
    )
    try:
        source_state = {key: value.detach().clone() for key, value in source.model.state_dict().items()}
        checkpoint = save_training_checkpoint(
            tmp_path / "v10-source",
            config=base,
            resources=source,
            state=TrainingState(environment_steps=70_679, policy_version=1_111),
            checkpoint_load_mode="fresh",
        )
    finally:
        source.close()

    v11 = replace(
        base,
        transaction_learning=replace(
            base.transaction_learning,
            enabled=True,
            replay_capacity=8,
            replay_byte_capacity=10_000_000,
            sample_traces=2,
        ),
    )
    target = build_training_resources(
        v11,
        backend=cast(EnvironmentBackend, _NoopBackend()),
    )
    try:
        new_head_before = {
            key: value.detach().clone() for key, value in target.model.state_dict().items() if key not in source_state
        }
        parent = initialize_model_from_checkpoint(
            checkpoint,
            config=v11,
            resources=target,
        )
        assert parent == checkpoint.resolve()
        assert len(target.optimizer.state) == 0
        assert target.transaction_replay is not None
        for key, expected in source_state.items():
            assert torch.equal(target.model.state_dict()[key], expected), key
        for key, expected in new_head_before.items():
            assert torch.equal(target.model.state_dict()[key], expected), key

        snapshot = _snapshot(v11.model.to_encoding_config())
        target.transaction_replay.put(
            TransactionTrace(
                trace_id="checkpoint-trace",
                episode_id="checkpoint-episode",
                surface_key="checkpoint-surface",
                start_step=0,
                policy_version=0,
                initial_recurrent_state=np.zeros(32, dtype=np.float32),
                steps=(
                    TransactionStep(
                        snapshot=snapshot,
                        action_index=0,
                        node_key="checkpoint-node",
                        next_node_key="checkpoint-next",
                        action_fingerprint="factual-action",
                        effect=TransactionEffect.EXIT,
                        selected_count_delta=0,
                        transaction_return=1.0,
                        return_steps=1,
                    ),
                ),
            )
        )
        migrated = save_training_checkpoint(
            tmp_path / "v11-lineage",
            config=v11,
            resources=target,
            state=TrainingState(),
            parent_checkpoint=checkpoint,
            checkpoint_load_mode="model_initialization",
            parent_relation="model_parameter_initialization",
        )
        assert (migrated / "transaction_replay.pkl").is_file()
        metadata = json.loads((migrated / "metadata.json").read_text("utf-8"))
        assert metadata["training_state"] == asdict(TrainingState())
        assert metadata["provenance"]["checkpoint_load_mode"] == "model_initialization"
        parent_metadata = metadata["provenance"]["parent_checkpoint"]
        assert parent_metadata["relation"] == "model_parameter_initialization"
        assert parent_metadata["training_state"]["policy_version"] == 1_111
    finally:
        target.close()

    restored = build_training_resources(
        v11,
        backend=cast(EnvironmentBackend, _NoopBackend()),
    )
    try:
        loaded = load_training_checkpoint(migrated, config=v11, resources=restored)
        assert loaded == TrainingState()
        assert restored.transaction_replay is not None
        assert restored.transaction_replay.snapshot()[0].trace_id == "checkpoint-trace"
    finally:
        restored.close()


def test_parameter_initialization_overlay_is_strict_except_for_new_heads() -> None:
    config = _model_config()
    source = RecurrentCandidateModel(config).state_dict()
    target = RecurrentCandidateModel(
        config,
        enable_transaction_heads=True,
    ).state_dict()

    migrated = checkpointing_module._model_parameter_initialization_state(
        source,
        target_state=target,
        allow_missing_transaction_heads=True,
    )
    assert set(migrated) == set(target)
    assert all(torch.equal(migrated[key], value) for key, value in source.items())

    partial_heads = dict(target)
    del partial_heads["transaction_q_head.3.bias"]
    with pytest.raises(ValueError, match="either all or none"):
        checkpointing_module._model_parameter_initialization_state(
            partial_heads,
            target_state=target,
            allow_missing_transaction_heads=True,
        )

    shared_key = "policy_head.0.weight"
    missing_shared = dict(source)
    del missing_shared[shared_key]
    with pytest.raises(ValueError, match="missing shared tensors"):
        checkpointing_module._model_parameter_initialization_state(
            missing_shared,
            target_state=target,
            allow_missing_transaction_heads=True,
        )

    unexpected = dict(source)
    unexpected["unsupported.weight"] = torch.zeros(1)
    with pytest.raises(ValueError, match="unsupported tensors"):
        checkpointing_module._model_parameter_initialization_state(
            unexpected,
            target_state=target,
            allow_missing_transaction_heads=True,
        )

    wrong_shape = dict(source)
    wrong_shape[shared_key] = source[shared_key][:-1]
    with pytest.raises(ValueError, match="tensor ABI differs"):
        checkpointing_module._model_parameter_initialization_state(
            wrong_shape,
            target_state=target,
            allow_missing_transaction_heads=True,
        )


def test_legacy_lineage_version_requires_new_lineage_but_keeps_model_initialization_compatible(
    tmp_path: Path,
) -> None:
    config = TrainingConfig()
    enabled = replace(
        config,
        transaction_learning=replace(config.transaction_learning, enabled=True),
    )
    legacy_lineage = enabled.lineage_mapping()
    legacy_lineage["version"] = "sts2-relational-curriculum-config-v3"
    validated = ValidatedResumeCheckpoint(
        root=tmp_path,
        manifest={},
        metadata={
            "format": "sts2-recurrent-vtrace-checkpoint-v5",
            "model_config": asdict(enabled.model.to_model_config()),
            "encoding_contract": grounding_encoding_identity(),
            "model_state_spec": {},
            "actor_model_state_spec": {},
            "training_state": asdict(TrainingState()),
            "lineage_config": legacy_lineage,
            "resolved_device": "cpu",
            "resolved_collector_device": "cpu",
            "queue_spec": {},
            "decision_semantics_abi": checkpointing_module._decision_semantics_abi(),
            "failure_credit_abi": checkpointing_module._failure_credit_abi(),
            "failure_credit_mode": enabled.failure_credit.mode,
            "liveness_cost_heads_enabled": False,
            "failure_credit_replay_enabled": False,
            "failure_credit_replay_spec": None,
            "transaction_heads_enabled": True,
            "transaction_replay_spec": {},
        },
    )

    with pytest.raises(ValueError, match="identical immutable training lineage"):
        checkpointing_module._validate_metadata(
            validated,
            config=enabled,
            resolved_device="cpu",
            resolved_collector_device="cpu",
            model_only=False,
        )
    checkpointing_module._validate_metadata(
        validated,
        config=enabled,
        resolved_device=None,
        resolved_collector_device=None,
        model_only=True,
    )

    source = RecurrentCandidateModel(
        enabled.model.to_model_config(),
        enable_transaction_heads=True,
    ).state_dict()
    target = RecurrentCandidateModel(
        enabled.model.to_model_config(),
        enable_transaction_heads=True,
    ).state_dict()
    migrated = checkpointing_module._model_parameter_initialization_state(
        source,
        target_state=target,
        allow_missing_transaction_heads=True,
    )
    assert set(migrated) == set(target)
    assert all(torch.equal(migrated[key], value) for key, value in source.items())


def test_legacy_v10_lineage_without_transaction_section_exact_resumes_only_disabled(
    tmp_path: Path,
) -> None:
    config = TrainingConfig()
    legacy_lineage = config.lineage_mapping()
    del legacy_lineage["transaction_learning"]
    validated = ValidatedResumeCheckpoint(
        root=tmp_path,
        manifest={},
        metadata={
            "format": "sts2-recurrent-vtrace-checkpoint-v5",
            "model_config": asdict(config.model.to_model_config()),
            "encoding_contract": grounding_encoding_identity(),
            "model_state_spec": {},
            "actor_model_state_spec": {},
            "training_state": asdict(TrainingState()),
            "lineage_config": legacy_lineage,
            "resolved_device": "cpu",
            "resolved_collector_device": "cpu",
            "queue_spec": {},
            "decision_semantics_abi": checkpointing_module._decision_semantics_abi(),
            "failure_credit_abi": checkpointing_module._failure_credit_abi(),
            "failure_credit_mode": config.failure_credit.mode,
            "liveness_cost_heads_enabled": False,
            "failure_credit_replay_enabled": False,
            "failure_credit_replay_spec": None,
            "long_horizon_value_head_abi": "sts2-long-horizon-value-heads-v1",
            "combat_hp_loss_value_head_abi": "sts2-combat-hp-loss-value-head-v1",
            "episodic_replay_enabled": False,
        },
    )

    checkpointing_module._validate_metadata(
        validated,
        config=config,
        resolved_device="cpu",
        resolved_collector_device="cpu",
        model_only=False,
    )

    enabled = replace(
        config,
        transaction_learning=replace(config.transaction_learning, enabled=True),
    )
    with pytest.raises(ValueError, match="identical immutable training lineage"):
        checkpointing_module._validate_metadata(
            validated,
            config=enabled,
            resolved_device="cpu",
            resolved_collector_device="cpu",
            model_only=False,
        )


def test_runtime_commits_completed_traces_then_samples_and_checkpoints_replay(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from tests.test_v2_training_pipeline import (
        DynamicPreviewLoopBackend,
        _event_loop_config,
    )

    monkeypatch.setenv("STS2_ARTIFACT_ROOT", str(tmp_path))
    base = _event_loop_config(durable_window=4)
    config = replace(
        base,
        transaction_learning=TransactionLearningConfig(
            enabled=True,
            replay_capacity=16,
            replay_byte_capacity=10_000_000,
            sample_traces=4,
            burn_in_steps=2,
        ),
        runtime=replace(
            base.runtime,
            total_environment_steps=8,
            log_dir="runs/transaction-runtime",
            checkpoint_dir="checkpoints/transaction-runtime",
            checkpoint_interval_steps=100,
            evaluation_steps=(),
            evaluation_episodes=0,
        ),
    )

    state = run_training(
        config,
        backend=DynamicPreviewLoopBackend(alternate_ui_actions=True),
    )

    assert state.environment_steps == 8
    assert state.episodes == 2
    metrics_path = next((tmp_path / "runs" / "transaction-runtime").glob("run-*/metrics.jsonl"))
    events = [json.loads(line) for line in metrics_path.read_text(encoding="utf-8").splitlines()]
    train_episodes = [event for event in events if event["event"] == "train_episode"]
    assert len(train_episodes) == 2
    assert all(event["transaction_traces_emitted"] == 1 for event in train_episodes)
    assert all(event["transaction_traces_stored"] == 1 for event in train_episodes)
    learner_updates = [event for event in events if event["event"] == "learner_update"]
    assert any(event["transaction_traces"] >= 1 for event in learner_updates)
    assert any(event["transaction_effect_labels"] >= 1 for event in learner_updates)

    checkpoint = next((tmp_path / "checkpoints" / "transaction-runtime").glob("run-*/final-*"))
    metadata = json.loads((checkpoint / "metadata.json").read_text(encoding="utf-8"))
    assert metadata["transaction_heads_enabled"] is True
    assert metadata["transaction_replay_spec"]["size"] == 2
    assert (checkpoint / "transaction_replay.pkl").is_file()
