from __future__ import annotations

import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


from muzero.diagnostics.deck_upgrade_metrics import (
    DeckUpgradeEpisodeTracker,
    build_deck_upgrade_choice_payload,
    build_smith_upgrade_transition_payload,
    complete_deck_upgrade_choice_payload,
    deck_upgrade_available_summary,
    is_deck_upgrade_action,
    is_deck_upgrade_terminal_action,
)


def _strike(*, upgraded: bool = False) -> dict:
    return {
        "id": "CARD.STRIKE_IRONCLAD",
        "title": "打击+" if upgraded else "打击",
        "type": "Attack",
        "cost": 1,
        "energy_cost": 1,
        "upgraded": upgraded,
        "upgrade_level": 1 if upgraded else 0,
        "card_effect_profile": {
            "semantic_signals": {"damage": 9 if upgraded else 6},
            "semantic_tags": ["damage", "attack"],
        },
    }


def _defend(*, upgraded: bool = False) -> dict:
    return {
        "id": "CARD.DEFEND_IRONCLAD",
        "title": "防御+" if upgraded else "防御",
        "type": "Skill",
        "cost": 1,
        "energy_cost": 1,
        "upgraded": upgraded,
        "upgrade_level": 1 if upgraded else 0,
        "card_effect_profile": {
            "semantic_signals": {"block": 8 if upgraded else 5},
            "semantic_tags": ["block", "skill"],
        },
    }


def _obs(upgraded_count: int = 0) -> dict:
    return {
        "player": {
            "deck_cards": [
                _strike(upgraded=upgraded_count >= 1),
                _defend(upgraded=upgraded_count >= 2),
                _strike(upgraded=False),
            ]
        }
    }


def test_is_deck_upgrade_action_recognizes_bridge_and_semantic_shapes() -> None:
    assert is_deck_upgrade_action({"kind": "deck_upgrade", "card": {"title": "打击"}})
    assert is_deck_upgrade_action({"action_id": "deck_upgrade:0", "title": "打击"})
    assert is_deck_upgrade_action({"action_id": "upgrade_card:2", "title": "防御"})
    assert is_deck_upgrade_action(
        {
            "kind": "generic",
            "semantic": {"domain": "build", "family": "deck_upgrade"},
        }
    )
    assert not is_deck_upgrade_action({"kind": "rest_site", "title": "锻造"})


def test_deck_upgrade_close_is_terminal_not_upgrade_target() -> None:
    close = {"kind": "deck_upgrade", "action_id": "deck_upgrade:close", "title": "关闭"}
    wrapped_close = {"payload": {"kind": "deck_upgrade", "action_id": "sim:upgrade_card:close"}}

    assert is_deck_upgrade_terminal_action(close)
    assert is_deck_upgrade_terminal_action(wrapped_close)
    assert not is_deck_upgrade_action(close)
    assert not is_deck_upgrade_action(wrapped_close)
    assert deck_upgrade_available_summary(
        [
            close,
            {"kind": "deck_upgrade", "action_id": "deck_upgrade:0", "card": {"title": "痛击"}},
        ]
    ) == {
        "deck_upgrade_available": True,
        "deck_upgrade_action_count": 1.0,
    }


def test_build_and_complete_deck_upgrade_choice_payload_records_upgrade_delta() -> None:
    upgrade_action = {
        "kind": "deck_upgrade",
        "action_id": "deck_upgrade:0",
        "card": {"id": "CARD.STRIKE_IRONCLAD", "title": "打击"},
    }
    non_upgrade_action = {"kind": "cancel", "action_id": "cancel"}

    payload = build_deck_upgrade_choice_payload(
        decision_domain="build",
        phase="deck_upgrade",
        legal_actions=[upgrade_action, non_upgrade_action],
        chosen_action=upgrade_action,
        chosen_signature=upgrade_action,
        selected_index=0,
        progress={"floor": 8, "act_id": 1, "room_type": "rest"},
        raw_obs=_obs(upgraded_count=0),
        search_policy=[0.8, 0.2],
    )

    assert payload is not None
    assert payload["selected_action_kind"] == "deck_upgrade"
    assert payload["deck_upgrade_available"] is True
    assert payload["deck_upgrade_action_count"] == 1.0
    assert payload["deck_context_present"] is True
    assert payload["upgraded_count_before"] == 0.0
    assert payload["legal_deck_upgrade_actions"][0]["title"] == "打击"

    completed = complete_deck_upgrade_choice_payload(
        payload,
        post_obs=_obs(upgraded_count=1),
        post_info={"phase": "map", "decision_domain": "route"},
    )

    assert completed is not None
    assert completed["upgraded_count_after"] == 1.0
    assert completed["upgraded_delta"] == 1.0
    assert completed["upgrade_applied"] is True


def test_smith_upgrade_transition_detects_followup_upgrade_surface() -> None:
    rest_payload = {
        "decision_domain": "build",
        "phase": "rest_site",
        "selected_action_kind": "smith",
        "floor": 8,
        "act_id": 1,
        "selected_action": {"kind": "rest_site", "title": "锻造"},
    }
    post_actions = [
        {"kind": "deck_upgrade", "action_id": "deck_upgrade:0", "card": {"title": "打击"}},
        {"kind": "deck_upgrade", "action_id": "deck_upgrade:1", "card": {"title": "防御"}},
    ]

    payload = build_smith_upgrade_transition_payload(
        rest_site_payload=rest_payload,
        post_legal_actions=post_actions,
        pre_obs=_obs(upgraded_count=0),
        post_obs=_obs(upgraded_count=0),
        post_info={"phase": "deck_upgrade", "decision_domain": "build"},
    )

    assert payload is not None
    assert payload["smith_to_upgrade_seen"] is True
    assert payload["deck_upgrade_action_count"] == 2.0
    assert payload["deck_context_present"] is True


def test_deck_upgrade_tracker_metadata_rates() -> None:
    tracker = DeckUpgradeEpisodeTracker()
    tracker.update_smith_transition({"smith_to_upgrade_seen": True})
    tracker.update_smith_transition({"smith_to_upgrade_seen": False})
    tracker.update_deck_upgrade_choice(
        {
            "selected_action_kind": "deck_upgrade",
            "upgrade_applied": True,
            "deck_upgrade_action_count": 2.0,
            "upgraded_delta": 1.0,
            "deck_context_present": True,
            "upgraded_count_before": 0.0,
            "upgraded_count_after": 1.0,
        }
    )
    tracker.update_deck_upgrade_choice(
        {
            "selected_action_kind": "non_deck_upgrade",
            "upgrade_applied": False,
            "deck_upgrade_action_count": 1.0,
            "upgraded_delta": 0.0,
            "deck_context_present": True,
            "upgraded_count_before": 1.0,
            "upgraded_count_after": 1.0,
        }
    )

    meta = tracker.as_metadata()
    assert meta["deck_upgrade_smith_selected_count"] == 2.0
    assert meta["deck_upgrade_smith_to_upgrade_seen_rate"] == 0.5
    assert meta["deck_upgrade_smith_no_upgrade_surface_rate"] == 0.5
    assert meta["deck_upgrade_seen_count"] == 2.0
    assert meta["deck_upgrade_selected_rate"] == 0.5
    assert meta["deck_upgrade_applied_rate"] == 1.0
    assert meta["deck_upgrade_target_count_mean"] == 1.5
    assert meta["deck_upgrade_applied_delta_mean"] == 0.5
    assert meta["deck_upgrade_context_present_rate"] == 1.0


def test_deck_upgrade_available_summary_tolerates_missing_actions() -> None:
    assert deck_upgrade_available_summary([]) == {
        "deck_upgrade_available": False,
        "deck_upgrade_action_count": 0.0,
    }
