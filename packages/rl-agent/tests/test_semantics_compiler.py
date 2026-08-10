from __future__ import annotations

from typing import Any

from sts2_rl.semantics import DECISION_CLOCK_BASE
from sts2_rl.semantics.compiler import (
    CompiledKind,
    SemanticActionCompiler,
)


def _rest_observation() -> dict[str, Any]:
    return {
        "phase": "rest",
        "state_type": "rest_site",
        "rest_site": {"id": "REST-7", "state": "menu"},
        "run": {"floor": 7},
    }


def _picker_observation(*, max_select: int = 1, selected: list[dict[str, Any]] | None = None) -> dict[str, Any]:
    return {
        "phase": "card_selection",
        "state_type": "card_select",
        "selection": {
            "prompt_id": "smith-prompt",
            "min_select": 1,
            "max_select": max_select,
            "selected_cards": selected or [],
        },
        "run": {"floor": 7},
    }


def _rest_actions() -> list[dict[str, Any]]:
    return [
        {"kind": "rest_site", "action": "choose_rest_option", "idx": 0, "option": {"type": "rest"}},
        {"kind": "rest_site", "action": "choose_rest_option", "idx": 1, "option": {"type": "smith"}},
    ]


def _select(card_id: str) -> dict[str, Any]:
    return {
        "kind": "select_card",
        "action": "select_card",
        "model_action_kind": "card_selection",
        "selection_operation": "select",
        "card": {"id": card_id, "is_upgraded": False},
    }


def _deselect(card_id: str) -> dict[str, Any]:
    return {
        "kind": "deselect_card",
        "action": "deselect_card",
        "model_action_kind": "card_selection",
        "selection_operation": "deselect",
        "card": {"id": card_id, "is_upgraded": False},
    }


def _confirm() -> dict[str, Any]:
    return {
        "kind": "confirm_selection",
        "action": "confirm_selection",
        "model_action_kind": "card_selection",
        "selection_operation": "confirm",
    }


def _cancel() -> dict[str, Any]:
    return {
        "kind": "cancel_selection",
        "action": "cancel_selection",
        "model_action_kind": "card_selection",
        "selection_operation": "cancel_prompt",
    }


def _picker_actions(*card_ids: str) -> list[dict[str, Any]]:
    return [*(_select(card) for card in card_ids), _confirm(), _cancel()]


def test_single_target_smith_flow_compiles_to_one_composite_commit() -> None:
    compiler = SemanticActionCompiler()

    entry = compiler.feed(
        step_index=0,
        observation=_rest_observation(),
        legal_actions=_rest_actions(),
        selected_action=_rest_actions()[1],
        floor=7,
    )
    assert entry.kind is CompiledKind.SEMANTIC_DECISION
    assert entry.root_spec == "rest"

    picking = compiler.feed(
        step_index=1,
        observation=_picker_observation(),
        legal_actions=_picker_actions("CARD.BASH", "CARD.DEFEND_IRONCLAD"),
        selected_action=_select("CARD.BASH"),
        floor=7,
    )
    # Parent-scope inheritance: the picker keeps the rest root instead of
    # falling to the opaque adapter (stage-1 defect 2).
    assert picking.root_spec == "rest"
    assert picking.kind is CompiledKind.MECHANICAL_FOLDED

    commit = compiler.feed(
        step_index=2,
        observation=_picker_observation(selected=[{"id": "CARD.BASH"}]),
        legal_actions=_picker_actions("CARD.BASH", "CARD.DEFEND_IRONCLAD"),
        selected_action=_confirm(),
        floor=7,
    )
    assert commit.kind is CompiledKind.COMPOSITE_COMMIT
    assert commit.root_spec == "rest"
    assert commit.folded_steps == 2
    assert "CARD.BASH" in (commit.target_identity or "")
    assert commit.clock is not None and commit.clock.discount == 1.0

    summary = compiler.summary()
    assert summary["kinds"][CompiledKind.COMPOSITE_COMMIT.value] == 1
    # The select toggle never became a decision of its own (stage-1 defect 1).
    assert CompiledKind.COMPOSITE_TARGET.value not in summary["kinds"]


def test_select_deselect_cycle_folds_into_churn_not_decisions() -> None:
    compiler = SemanticActionCompiler()
    compiler.feed(
        step_index=0,
        observation=_rest_observation(),
        legal_actions=_rest_actions(),
        selected_action=_rest_actions()[1],
        floor=7,
    )
    actions = _picker_actions("CARD.BASH", "CARD.DEFEND_IRONCLAD")
    for index, chosen in enumerate(
        (_select("CARD.BASH"), _deselect("CARD.BASH"), _select("CARD.BASH")),
        start=1,
    ):
        event = compiler.feed(
            step_index=index,
            observation=_picker_observation(),
            legal_actions=actions,
            selected_action=chosen,
            floor=7,
        )
        assert event.kind is CompiledKind.MECHANICAL_FOLDED
    refusal = compiler.feed(
        step_index=4,
        observation=_picker_observation(),
        legal_actions=actions,
        selected_action=_cancel(),
        floor=7,
    )
    assert refusal.kind is CompiledKind.COMPOSITE_REFUSAL
    assert refusal.folded_steps == 4
    assert refusal.deselect_churn == 1
    summary = compiler.summary()
    assert summary["deselect_churn"] == 1
    assert summary["kinds"].get(CompiledKind.SEMANTIC_DECISION.value) == 1


def test_multi_select_uses_monotone_add_and_commit_set() -> None:
    compiler = SemanticActionCompiler()
    compiler.feed(
        step_index=0,
        observation=_rest_observation(),
        legal_actions=_rest_actions(),
        selected_action=_rest_actions()[1],
        floor=7,
    )
    actions = _picker_actions("CARD.A", "CARD.B", "CARD.C")
    first = compiler.feed(
        step_index=1,
        observation=_picker_observation(max_select=2),
        legal_actions=actions,
        selected_action=_select("CARD.A"),
        floor=7,
    )
    assert first.kind is CompiledKind.COMPOSITE_TARGET
    assert first.operation == "add"
    second = compiler.feed(
        step_index=2,
        observation=_picker_observation(max_select=2),
        legal_actions=actions,
        selected_action=_select("CARD.B"),
        floor=7,
    )
    assert second.kind is CompiledKind.COMPOSITE_TARGET
    commit = compiler.feed(
        step_index=3,
        observation=_picker_observation(max_select=2),
        legal_actions=actions,
        selected_action=_confirm(),
        floor=7,
    )
    assert commit.kind is CompiledKind.COMPOSITE_COMMIT
    assert commit.operation == "commit_set"
    assert "CARD.A" in (commit.target_identity or "")
    assert "CARD.B" in (commit.target_identity or "")


def test_reveal_boundary_and_floor_clock() -> None:
    compiler = SemanticActionCompiler()
    compiler.feed(
        step_index=0,
        observation=_rest_observation(),
        legal_actions=_rest_actions(),
        selected_action=_rest_actions()[1],
        floor=7,
    )
    compiler.feed(
        step_index=1,
        observation=_picker_observation(),
        legal_actions=_picker_actions("CARD.A", "CARD.B"),
        selected_action=_select("CARD.A"),
        floor=7,
    )
    # The selectable set changes while the overlay stays open: reveal boundary.
    revealed = compiler.feed(
        step_index=2,
        observation=_picker_observation(),
        legal_actions=_picker_actions("CARD.X", "CARD.Y"),
        selected_action=_select("CARD.X"),
        floor=7,
    )
    assert revealed.reveal_boundary is True

    # Floor advance applies the durable clock exactly once on the next event.
    compiler.feed(
        step_index=3,
        observation=_picker_observation(),
        legal_actions=_picker_actions("CARD.X", "CARD.Y"),
        selected_action=_confirm(),
        floor=8,
    )
    commit = compiler.events[-1]
    assert commit.clock is not None
    assert commit.clock.discount == DECISION_CLOCK_BASE


def test_opaque_root_passes_through_untouched() -> None:
    compiler = SemanticActionCompiler()
    event = compiler.feed(
        step_index=0,
        observation={"phase": "??unknown??", "run": {"floor": 3}},
        legal_actions=[{"kind": "mystery_a"}, {"kind": "mystery_b"}],
        selected_action={"kind": "mystery_a"},
        floor=3,
    )
    assert event.kind is CompiledKind.OPAQUE_PASSTHROUGH
    assert event.root_spec == "opaque"
