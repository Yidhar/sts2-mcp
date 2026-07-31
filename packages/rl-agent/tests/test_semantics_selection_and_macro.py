from __future__ import annotations

from typing import Any

from sts2_rl.semantics import (
    DecisionSemanticsKernel,
    ForcedTransition,
    MacroEdgeBuilder,
    MacroEdgeOutcome,
    ProgressKind,
    ProgressReceipt,
    SurfaceRole,
    strict_action_groups,
)


def _event_root() -> dict[str, Any]:
    return {
        "phase": "event",
        "run": {"act": 1, "floor": 8},
        "room": {
            "room_type": "event",
            "room_model_id": "ROOM_FULL_OF_CHEESE",
        },
        "event": {
            "event_id": "LINGER9",
            "page_id": "LINGER9",
        },
        "player": {"hp": 60, "max_hp": 80},
    }


def _duplicate_select_actions() -> tuple[dict[str, Any], ...]:
    shared = {
        "model_action_kind": "card_selection",
        "model_action_variant": "select",
        "selection_operation": "select",
        "kind": "select_card",
        "source_zone": "hand",
    }
    return (
        {
            **shared,
            "action_handle": "copy-1",
            "action_index": 4,
            "card": {
                "card_id": "BURNING_PACT",
                "card_instance_id": "instance-1",
                "upgrade_level": 0,
                "enchantment": "NONE",
            },
        },
        {
            **shared,
            "action_handle": "copy-2",
            "action_index": 17,
            "card": {
                "card_id": "BURNING_PACT",
                "card_instance_id": "instance-2",
                "upgrade_level": 0,
                "enchantment": "NONE",
            },
        },
    )


def test_strict_duplicate_grouping_merges_only_exact_card_copies() -> None:
    actions = (
        *_duplicate_select_actions(),
        {
            **_duplicate_select_actions()[0],
            "action_handle": "upgraded-copy",
            "action_index": 23,
            "card": {
                "card_id": "BURNING_PACT",
                "card_instance_id": "instance-3",
                "upgrade_level": 1,
                "enchantment": "NONE",
            },
        },
        {
            **_duplicate_select_actions()[0],
            "action_handle": "deselect-copy",
            "action_index": 24,
            "kind": "deselect_card",
            "selection_operation": "deselect",
            "model_action_variant": "deselect",
        },
    )
    groups = strict_action_groups(actions)

    assert len(groups) == 3
    assert groups[0].multiplicity == 2
    assert groups[0].member_positions == (0, 1)
    assert groups[0].equivalence_fingerprint is not None
    assert groups[1].multiplicity == 1
    assert groups[2].multiplicity == 1


def test_unknown_card_action_fact_prevents_strict_duplicate_grouping() -> None:
    first, second = _duplicate_select_actions()
    first = {**first, "future_semantic_rule": {"mode": "A"}}
    second = {**second, "future_semantic_rule": {"mode": "B"}}

    groups = strict_action_groups((first, second))

    assert len(groups) == 2
    assert all(group.multiplicity == 1 for group in groups)


def test_selection_overlay_inherits_parent_event_scope() -> None:
    kernel = DecisionSemanticsKernel()
    root = kernel.identify(
        observation=_event_root(),
        legal_actions=(
            {
                "model_action_kind": "event_option",
                "option_id": "OPEN_SELECTION",
            },
            {
                "model_action_kind": "event_option",
                "option_id": "LEAVE",
            },
        ),
    )
    selection_only_observation = {
        "phase": "card_selection",
        "selection": {
            "prompt_id": "CHOOSE_TWO",
            "min_select": 2,
            "max_select": 2,
            "remaining_select": 2,
            "selected_cards": [],
        },
        "player": {"hp": 60, "max_hp": 80},
    }
    nested = kernel.identify(
        observation=selection_only_observation,
        legal_actions=_duplicate_select_actions(),
        parent_scopes=root.scopes,
    )

    assert nested.anchor == root.anchor
    assert nested.scopes.root.spec_id == "event"
    assert nested.scopes.root.role is SurfaceRole.ROOT
    assert nested.scopes.active.spec_id == "selection"
    assert nested.scopes.active.role is SurfaceRole.OVERLAY
    assert len(nested.scopes.scopes) == 2
    assert len(nested.actions) == 1
    assert nested.actions[0].group.multiplicity == 2


def test_selection_transaction_change_is_reviewed_control_move() -> None:
    kernel = DecisionSemanticsKernel()
    before_observation = {
        "phase": "combat",
        "state_type": "monster",
        "combat": {
            "in_progress": True,
            "round": 3,
            "enemies": [{"model_id": "MONSTER.CULTIST", "hp": 48}],
        },
        "card_selection": {
            "prompt_id": "potions.GAMBLERS_BREW.selectionScreenPrompt",
            "min_select": 0,
            "max_select": 99,
            "remaining_select": 2,
            "selected_count": 0,
            "selected_cards": [],
        },
        "player": {"hp": 60, "max_hp": 80},
    }
    after_observation = {
        **before_observation,
        "card_selection": {
            **before_observation["card_selection"],
            "remaining_select": 1,
            "selected_count": 1,
            "selected_cards": [{"id": "CARD.BURNING_PACT"}],
        },
    }
    before = kernel.identify(
        observation=before_observation,
        legal_actions=_duplicate_select_actions(),
    )
    after = kernel.identify(
        observation=after_observation,
        legal_actions=(
            {
                **_duplicate_select_actions()[0],
                "selection_operation": "deselect",
                "model_action_variant": "deselect",
                "kind": "deselect_card",
                "action": "deselect_card",
            },
        ),
        parent_scopes=before.scopes,
    )

    receipt = kernel.classify_transition(
        before=before,
        after=after,
        before_observation=before_observation,
        after_observation=after_observation,
    )

    assert before.scopes.root.spec_id == after.scopes.root.spec_id == "combat"
    assert before.scopes.active.spec_id == after.scopes.active.spec_id == "selection"
    assert receipt.kind is ProgressKind.CONTROL_MOVE


def test_selection_loop_identity_is_canonical_under_mapping_and_set_order() -> None:
    kernel = DecisionSemanticsKernel()
    first_observation = {
        "phase": "card_selection",
        "selection": {
            "prompt_id": "ORDER_INVARIANT",
            "min_select": 0,
            "max_select": 2,
            "selected_cards": [
                {
                    "card_id": "CARD.DEFEND",
                    "upgrade_level": 0,
                    "enchantments": [
                        {"id": "ENCHANT.A", "amount": 1},
                        {"id": "ENCHANT.B", "amount": 2},
                    ],
                },
                {
                    "upgrade_level": 0,
                    "card_id": "CARD.STRIKE",
                },
            ],
        },
    }
    second_observation = {
        "selection": {
            "selected_cards": [
                {
                    "card_id": "CARD.STRIKE",
                    "upgrade_level": 0,
                },
                {
                    "enchantments": [
                        {"amount": 1, "id": "ENCHANT.A"},
                        {"amount": 2, "id": "ENCHANT.B"},
                    ],
                    "upgrade_level": 0,
                    "card_id": "CARD.DEFEND",
                },
            ],
            "max_select": 2,
            "min_select": 0,
            "prompt_id": "ORDER_INVARIANT",
        },
        "phase": "card_selection",
    }
    first_actions = (
        {
            "model_action_kind": "card_selection",
            "selection_operation": "select",
            "model_action_variant": "select",
            "kind": "select_card",
            "card": {
                "card_id": "CARD.STRIKE",
                "upgrade_level": 0,
            },
        },
        {
            "model_action_kind": "card_selection",
            "selection_operation": "select",
            "model_action_variant": "select",
            "kind": "select_card",
            "card": {
                "card_id": "CARD.DEFEND",
                "upgrade_level": 0,
            },
        },
    )
    second_actions = (
        {
            "card": {
                "upgrade_level": 0,
                "card_id": "CARD.DEFEND",
            },
            "kind": "select_card",
            "model_action_variant": "select",
            "selection_operation": "select",
            "model_action_kind": "card_selection",
        },
        {
            "card": {
                "upgrade_level": 0,
                "card_id": "CARD.STRIKE",
            },
            "kind": "select_card",
            "model_action_variant": "select",
            "selection_operation": "select",
            "model_action_kind": "card_selection",
        },
    )

    parent = kernel.identify(
        observation=_event_root(),
        legal_actions=(
            {
                "model_action_kind": "event_option",
                "option_id": "OPEN_SELECTION",
            },
            {
                "model_action_kind": "event_option",
                "option_id": "LEAVE",
            },
        ),
    )
    first = kernel.identify(
        observation=first_observation,
        legal_actions=first_actions,
        parent_scopes=parent.scopes,
    )
    second = kernel.identify(
        observation=second_observation,
        legal_actions=second_actions,
        parent_scopes=parent.scopes,
    )

    assert first.node.loop.canonical_payload == second.node.loop.canonical_payload
    assert first.node.loop.digest == second.node.loop.digest


def test_rest_choice_entering_card_selection_preserves_parent_scope() -> None:
    kernel = DecisionSemanticsKernel()
    rest_observation = {
        "phase": "actions",
        "state_type": "rest_site",
        "run": {"act": 1, "floor": 8},
        "room": {"room_model_id": "ROOM.REST_SITE"},
        "rest_site": {
            "can_proceed": False,
            "options": [{"id": "SMITH", "is_enabled": True}],
        },
        "player": {"hp": 60, "max_hp": 80},
    }
    selection_observation = {
        "phase": "card_selection",
        "state_type": "card_select",
        "run": {"act": 1, "floor": 8},
        "room": {"room_model_id": "ROOM.REST_SITE"},
        "card_selection": {
            "prompt_id": "card_selection.TO_UPGRADE",
            "operation_type": "upgrade",
            "min_select": 1,
            "max_select": 1,
            "remaining_select": 1,
            "selected_count": 0,
            "selected_cards": [],
        },
        "player": {"hp": 60, "max_hp": 80},
    }
    before = kernel.identify(
        observation=rest_observation,
        legal_actions=(
            {
                "action": "choose_rest_option",
                "kind": "choose_rest_option",
                "model_action_kind": "rest_site",
                "option": {"id": "SMITH"},
            },
        ),
    )
    after = kernel.identify(
        observation=selection_observation,
        legal_actions=_duplicate_select_actions(),
        parent_scopes=before.scopes,
    )

    receipt = kernel.classify_transition(
        before=before,
        after=after,
        before_observation=rest_observation,
        after_observation=selection_observation,
    )

    assert before.scopes.root.spec_id == after.scopes.root.spec_id == "rest"
    assert after.scopes.active.spec_id == "selection"
    assert receipt.kind is ProgressKind.CONTROL_MOVE


def test_combat_round_advance_is_reviewed_control_move() -> None:
    kernel = DecisionSemanticsKernel()
    before_observation = {
        "phase": "combat",
        "state_type": "boss",
        "combat": {
            "in_progress": True,
            "round": 262,
            "enemies": [
                {
                    "model_id": "MONSTER.INSATIABLE",
                    "hp": 300,
                    "next_move_id": "COUNTDOWN",
                }
            ],
        },
        "player": {"hp": 60, "max_hp": 80, "hand": []},
    }
    after_observation = {
        **before_observation,
        "combat": {
            **before_observation["combat"],
            "round": 263,
        },
    }
    end_turn = (
        {
            "action": "end_turn",
            "kind": "end_turn",
            "model_action_kind": "end_turn",
        },
    )
    before = kernel.identify(
        observation=before_observation,
        legal_actions=end_turn,
    )
    after = kernel.identify(
        observation=after_observation,
        legal_actions=end_turn,
    )

    receipt = kernel.classify_transition(
        before=before,
        after=after,
        before_observation=before_observation,
        after_observation=after_observation,
    )

    assert receipt.kind is ProgressKind.CONTROL_MOVE
    assert receipt.source == "combat:control_node_changed"


def test_forced_suffix_keeps_actor_credit_on_initiating_policy_choice() -> None:
    kernel = DecisionSemanticsKernel()
    source = kernel.identify(
        observation=_event_root(),
        legal_actions=(
            {
                "model_action_kind": "event_option",
                "option_id": "LOOP",
            },
            {
                "model_action_kind": "event_option",
                "option_id": "EXIT",
            },
        ),
    )
    warning_observation = {
        **_event_root(),
        "event": {
            "event_id": "LINGER9",
            "page_id": "DEATH_WARNING",
        },
        "player": {"hp": 0, "max_hp": 80},
        "_training": {"revivals_used": 1, "player_hp_lost": 60},
    }
    warning = kernel.identify(
        observation=warning_observation,
        legal_actions=(
            {
                "model_action_kind": "proceed",
                "kind": "proceed",
            },
        ),
    )
    destination_observation = {
        **_event_root(),
        "_training": {"revivals_used": 1, "player_hp_lost": 60},
    }
    destination = kernel.identify(
        observation=destination_observation,
        legal_actions=(
            {
                "model_action_kind": "event_option",
                "option_id": "LOOP",
            },
            {
                "model_action_kind": "event_option",
                "option_id": "EXIT",
            },
        ),
    )
    builder = MacroEdgeBuilder(
        anchor=source.anchor,
        source_node=source.node,
        chosen_action=source.actions[0].identities,
        policy_step_id=100,
        legal_candidate_count=2,
    )
    builder.append_forced(
        ForcedTransition(
            step_id=101,
            node=warning.node,
            action=warning.actions[0].identities,
            receipt=ProgressReceipt(
                kind=ProgressKind.CONTROL_MOVE,
                source="event:death_warning",
            ),
        )
    )
    edge = builder.close(
        outcome=MacroEdgeOutcome.NEXT_POLICY,
        closing_step_id=102,
        destination_node=destination.node,
        destination_anchor=destination.anchor,
    )

    assert edge.anchor == destination.anchor
    assert edge.actor_credit_step_id == 100
    assert len(edge.forced_suffix) == 1
    assert edge.forced_suffix[0].step_id == 101
    assert edge.destination_node is not None
    assert edge.destination_node.loop == source.node.loop
