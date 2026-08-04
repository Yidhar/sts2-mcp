from __future__ import annotations

from copy import deepcopy
from typing import Any

import pytest

from sts2_rl.semantics import (
    CoarseActionCollisionError,
    DecisionSemanticsKernel,
    ProgressKind,
    SemanticKeyIndex,
    SurfaceRegistry,
    default_surface_registry,
)


def _event_observation(
    *,
    page: str = "LINGER9",
    hp: int = 60,
    revivals: int = 0,
    hp_lost: int = 0,
    floor: int = 12,
) -> dict[str, Any]:
    return {
        "phase": "event",
        "decision_domain": "build",
        "run": {"act": 1, "floor": floor},
        "room": {
            "room_type": "event",
            "room_model_id": "ROOM_FULL_OF_CHEESE",
            "coordinate": {"row": 12, "column": 2},
        },
        "event": {
            "event_id": "LINGER9",
            "page_id": page,
            "stage_id": "main",
        },
        "player": {
            "hp": hp,
            "max_hp": 80,
            "is_dead": hp <= 0,
            "gold": 99,
            "deck": [{"card_id": "STRIKE", "upgrade_level": 0}],
            "relics": [{"relic_id": "STARTER"}],
            "potions": [],
        },
        "_training": {
            "revivals_used": revivals,
            "player_hp_lost": hp_lost,
        },
    }


def _event_actions() -> tuple[dict[str, Any], ...]:
    return (
        {
            "action_handle": "volatile-a",
            "model_action_kind": "event_option",
            "kind": "event_option",
            "option_id": "TAKE_BITE",
        },
        {
            "action_handle": "volatile-b",
            "model_action_kind": "event_option",
            "kind": "event_option",
            "option_id": "LEAVE",
        },
    )


def test_hp_death_and_revival_are_cost_only_not_durable_progress() -> None:
    kernel = DecisionSemanticsKernel()
    before_observation = _event_observation()
    after_observation = _event_observation(
        hp=0,
        revivals=3,
        hp_lost=60,
    )
    before = kernel.identify(
        observation=before_observation,
        legal_actions=_event_actions(),
    )
    after = kernel.identify(
        observation=after_observation,
        legal_actions=_event_actions(),
    )

    assert before.anchor == after.anchor
    assert before.node.loop == after.node.loop
    assert before.node.exact != after.node.exact
    assert before.node.comparison != after.node.comparison

    receipt = kernel.classify_transition(
        before=before,
        after=after,
        before_observation=before_observation,
        after_observation=after_observation,
    )
    assert receipt.kind is ProgressKind.COST_ONLY
    assert not receipt.flow_advanced
    assert not receipt.durable_committed


def test_durable_commit_does_not_claim_flow_advance() -> None:
    kernel = DecisionSemanticsKernel()
    before_observation = _event_observation()
    after_observation = deepcopy(before_observation)
    after_observation["player"]["deck"].append(  # type: ignore[index]
        {"card_id": "DEFEND", "upgrade_level": 0}
    )
    before = kernel.identify(
        observation=before_observation,
        legal_actions=_event_actions(),
    )
    after = kernel.identify(
        observation=after_observation,
        legal_actions=_event_actions(),
    )

    receipt = kernel.classify_transition(
        before=before,
        after=after,
        before_observation=before_observation,
        after_observation=after_observation,
    )
    assert receipt.kind is ProgressKind.DURABLE_COMMIT
    assert receipt.durable_committed
    assert not receipt.flow_advanced


def test_screen_shaped_sim_raw_deck_relocation_is_transport_not_commit() -> None:
    """Cancel teardown must not fabricate a deck mutation from raw view paths."""

    kernel = DecisionSemanticsKernel()
    before_observation = _event_observation()
    after_observation = deepcopy(before_observation)
    deck = deepcopy(before_observation["player"]["deck"])  # type: ignore[index]
    before_observation["_sim_raw"] = {
        "card_select": {"player": {"deck": deck}},
    }
    after_observation["_sim_raw"] = {
        "rest": {"player": {"deck": deepcopy(deck)}},
    }
    before = kernel.identify(
        observation=before_observation,
        legal_actions=_event_actions(),
    )
    after = kernel.identify(
        observation=after_observation,
        legal_actions=_event_actions(),
    )

    assert before.node.exact == after.node.exact
    receipt = kernel.classify_transition(
        before=before,
        after=after,
        before_observation=before_observation,
        after_observation=after_observation,
    )
    assert receipt.kind is ProgressKind.NONE
    assert not receipt.durable_committed


def test_linger9_anchor_survives_death_warning_and_revival_churn() -> None:
    kernel = DecisionSemanticsKernel()
    linger = _event_observation(page="LINGER9")
    warning = _event_observation(
        page="DEATH_WARNING",
        hp=0,
        revivals=1,
        hp_lost=60,
    )
    linger_again = _event_observation(
        page="LINGER9",
        hp=80,
        revivals=1,
        hp_lost=60,
    )

    linger_identity = kernel.identify(
        observation=linger,
        legal_actions=_event_actions(),
    )
    warning_identity = kernel.identify(
        observation=warning,
        legal_actions=(
            {
                "action_handle": "forced-warning",
                "model_action_kind": "proceed",
                "kind": "proceed",
            },
        ),
    )
    linger_again_identity = kernel.identify(
        observation=linger_again,
        legal_actions=_event_actions(),
    )

    assert linger_identity.anchor == warning_identity.anchor
    assert warning_identity.anchor == linger_again_identity.anchor
    assert linger_identity.node.loop != warning_identity.node.loop
    assert linger_identity.node.loop == linger_again_identity.node.loop


def test_exact_loop_and_comparison_views_have_distinct_noise_contracts() -> None:
    kernel = DecisionSemanticsKernel()
    before_observation = _event_observation()
    before_observation["event"]["localized_text"] = "old rendering"
    before_observation["request_id"] = "transport-1"
    after_observation = deepcopy(before_observation)
    after_observation["event"]["localized_text"] = "new rendering"  # type: ignore[index]
    after_observation["request_id"] = "transport-2"
    before = kernel.identify(
        observation=before_observation,
        legal_actions=_event_actions(),
    )
    after = kernel.identify(
        observation=after_observation,
        legal_actions=_event_actions(),
    )

    assert before.anchor == after.anchor
    assert before.node.exact != after.node.exact
    assert before.node.loop == after.node.loop
    assert before.node.comparison == after.node.comparison


def test_floor_or_room_locus_change_is_flow_advance() -> None:
    kernel = DecisionSemanticsKernel()
    before_observation = _event_observation(floor=12)
    after_observation = _event_observation(floor=13)
    before = kernel.identify(
        observation=before_observation,
        legal_actions=_event_actions(),
    )
    after = kernel.identify(
        observation=after_observation,
        legal_actions=_event_actions(),
    )

    receipt = kernel.classify_transition(
        before=before,
        after=after,
        before_observation=before_observation,
        after_observation=after_observation,
    )
    assert receipt.kind is ProgressKind.FLOW_ADVANCE
    assert receipt.flow_advanced


def test_action_coarsening_collision_fails_closed() -> None:
    kernel = DecisionSemanticsKernel()
    actions = (
        {
            "action_handle": "first",
            "action_index": 4,
            "model_action_kind": "event_option",
            "option_id": "SAME_OPTION",
        },
        {
            "action_handle": "second",
            "action_index": 9,
            "model_action_kind": "event_option",
            "option_id": "SAME_OPTION",
        },
    )

    with pytest.raises(
        CoarseActionCollisionError,
        match="share one loop action identity",
    ):
        kernel.identify(
            observation=_event_observation(),
            legal_actions=actions,
        )


def test_unknown_action_semantics_prevent_unsafe_coarse_aliasing() -> None:
    kernel = DecisionSemanticsKernel()
    result = kernel.identify(
        observation=_event_observation(),
        legal_actions=(
            {
                "action_handle": "first",
                "model_action_kind": "event_option",
                "option_id": "SAME_OPTION",
                "effect_contract": {"gain_gold": 5},
            },
            {
                "action_handle": "second",
                "model_action_kind": "event_option",
                "option_id": "SAME_OPTION",
                "effect_contract": {"gain_gold": 50},
            },
        ),
    )

    assert result.actions[0].identities.loop != result.actions[1].identities.loop


def test_unreviewed_transition_retains_bounded_changed_path_diagnostics() -> None:
    kernel = DecisionSemanticsKernel()
    before_observation = _event_observation()
    after_observation = deepcopy(before_observation)
    after_observation["event"]["future_unreviewed_counter"] = 1  # type: ignore[index]
    before = kernel.identify(
        observation=before_observation,
        legal_actions=_event_actions(),
    )
    after = kernel.identify(
        observation=after_observation,
        legal_actions=_event_actions(),
    )

    receipt = kernel.classify_transition(
        before=before,
        after=after,
        before_observation=before_observation,
        after_observation=after_observation,
    )

    assert receipt.kind is ProgressKind.UNKNOWN
    assert receipt.changed_paths == ("event.future_unreviewed_counter",)


@pytest.mark.parametrize(
    ("observation", "action", "expected_surface"),
    (
        (
            {"phase": "combat", "combat": {"in_progress": True}},
            {"model_action_kind": "end_turn"},
            "combat",
        ),
        (
            {"phase": "event", "event": {"event_id": "E"}},
            {"model_action_kind": "event_option", "option_id": "O"},
            "event",
        ),
        (
            {"phase": "rest_site", "rest_site": {"id": "R"}},
            {"model_action_kind": "rest_site", "option_id": "SMITH"},
            "rest",
        ),
        (
            {"phase": "shop", "shop": {"id": "S"}},
            {"model_action_kind": "shop", "item_id": "I"},
            "shop",
        ),
        (
            {"phase": "reward", "reward": {"reward_id": "R"}},
            {"model_action_kind": "reward", "reward_id": "GOLD"},
            "reward",
        ),
        (
            {"phase": "map", "map": {"current_node_id": "M"}},
            {"model_action_kind": "map", "map_node_id": "N"},
            "map",
        ),
        (
            {"phase": "future-unreviewed-surface"},
            {"model_action_kind": "future-action"},
            "opaque",
        ),
    ),
)
def test_versioned_registry_resolves_all_root_surface_adapters(
    observation: dict[str, Any],
    action: dict[str, Any],
    expected_surface: str,
) -> None:
    registry = default_surface_registry()
    adapter = registry.resolve_root(observation, (action,))

    assert adapter.spec.spec_id == expected_surface
    assert adapter.spec.version.endswith("-v1")
    manifest = registry.manifest_payload()
    assert manifest["contract_version"] == "sts2-surface-registry-v1"
    assert any(item["spec_id"] == expected_surface for item in manifest["adapters"])


def test_registry_manifest_is_independent_of_registration_order() -> None:
    first = default_surface_registry()
    second = SurfaceRegistry(tuple(reversed(first.adapters)))
    key_index = SemanticKeyIndex()

    assert first.manifest_payload() == second.manifest_payload()
    assert first.manifest_key(key_index) == second.manifest_key(key_index)
