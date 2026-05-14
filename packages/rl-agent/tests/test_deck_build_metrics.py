from __future__ import annotations

import math
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from muzero.diagnostics.deck_build_metrics import (
    CardRewardEpisodeTracker,
    compact_deck_cards,
    compute_deck_quality_summary,
    extract_deck_cards_from_obs_like,
)
from sts2_env.deck_quality import DECK_QUALITY_V2_KEYS


def _strike() -> dict:
    return {
        "id": "CARD.STRIKE_IRONCLAD",
        "title": "打击",
        "type": "Attack",
        "cost": 1,
        "energy_cost": 1,
        "card_effect_profile": {
            "semantic_signals": {"damage": 6},
            "semantic_tags": ["damage", "attack"],
        },
    }


def _defend() -> dict:
    return {
        "id": "CARD.DEFEND_IRONCLAD",
        "title": "防御",
        "type": "Skill",
        "cost": 1,
        "energy_cost": 1,
        "upgraded": True,
        "upgrade_level": 1,
        "card_effect_profile": {
            "semantic_signals": {"block": 8},
            "semantic_tags": ["block", "skill"],
        },
    }


def _draw_card() -> dict:
    return {
        "id": "CARD.BATTLE_TRANCE",
        "title": "战斗专注",
        "type": "Skill",
        "cost": 0,
        "energy_cost": 0,
        "card_effect_profile": {
            "semantic_signals": {"draw": 3},
            "semantic_tags": ["draw", "card_draw"],
        },
    }


def test_extract_deck_cards_from_common_payload_shapes() -> None:
    deck = [_strike(), _defend()]

    assert extract_deck_cards_from_obs_like({"player": {"deck_cards": deck}}) == deck
    assert extract_deck_cards_from_obs_like({"transition_state": {"player": {"deck_cards": deck}}}) == deck
    assert extract_deck_cards_from_obs_like({"raw_obs": {"player": {"deck_cards": deck}}}) == deck
    assert extract_deck_cards_from_obs_like({"wrapper": {"state": {"player": {"deck_cards": deck}}}}) == deck
    assert extract_deck_cards_from_obs_like(None) == []


def test_compact_deck_cards_preserves_identity_upgrade_and_effect_signals() -> None:
    compact = compact_deck_cards([_strike(), _defend(), _draw_card()], limit=3)

    assert compact[0]["id"] == "CARD.STRIKE_IRONCLAD"
    assert compact[0]["title"] == "打击"
    assert compact[0]["type"] == "Attack"
    assert compact[0]["cost"] == 1
    assert compact[0]["damage"] == 6.0
    assert compact[1]["upgraded"] is True
    assert compact[1]["upgrade_level"] == 1
    assert compact[1]["block"] == 8.0
    assert compact[2]["draw"] == 3.0


def test_compute_deck_quality_summary_exposes_finite_deck_quality_keys() -> None:
    summary = compute_deck_quality_summary([_strike() for _ in range(5)] + [_defend() for _ in range(5)] + [_draw_card()])

    assert set(summary.keys()) == set(DECK_QUALITY_V2_KEYS)
    for key, value in summary.items():
        assert isinstance(value, float), key
        assert math.isfinite(value), key
    assert summary["raw_avg_damage_per_energy"] > 0.0
    assert summary["raw_avg_block_per_energy"] > 0.0
    assert summary["raw_expected_cards_seen_per_turn"] >= 5.0
    assert summary["expected_hand_attack_cards"] > 0.0
    assert summary["expected_hand_draw_cards"] > 0.0
    assert summary["expected_hand_engine_cards"] > 0.0
    assert summary["expected_hand_block_per_turn"] > 0.0
    assert 0.0 <= summary["expected_hand_useful_quality_score"] <= 1.0
    assert summary["raw_expected_playable_cards_per_turn"] > 0.0
    assert summary["expected_energy_utilization_score"] > 0.0
    assert summary["cost_curve_one_share"] > 0.0
    assert 0.0 <= summary["combo_option_value_score"] <= 1.0
    assert 0.0 <= summary["combo_unmet_dependency_score"] <= 1.0


def test_card_reward_tracker_counts_pick_skip_and_consecutive_skips() -> None:
    pick_action = {
        "kind": "card_reward",
        "selection": "pick",
        "surface": "card_reward",
        "card": {"id": "CARD.STRIKE_IRONCLAD", "title": "打击", "type": "Attack", "cost": 1},
    }
    skip_action = {
        "kind": "skip_card_reward",
        "selection": "skip",
        "surface": "card_reward",
        "action_id": "reward.skip_card_reward",
    }

    tracker = CardRewardEpisodeTracker()
    tracker.update(
        decision_domain="build",
        phase="card_reward",
        legal_actions=[pick_action, skip_action],
        chosen_action=skip_action,
        chosen_signature=skip_action,
        selected_family="card_reward",
        selection="skip",
    )
    tracker.update(
        decision_domain="build",
        phase="card_reward",
        legal_actions=[pick_action, skip_action],
        chosen_action=skip_action,
        chosen_signature=skip_action,
        selected_family="card_reward",
        selection="skip",
    )
    tracker.update(
        decision_domain="build",
        phase="card_reward",
        legal_actions=[pick_action, skip_action],
        chosen_action=pick_action,
        chosen_signature=pick_action,
        selected_family="card_reward",
        selection="pick",
    )

    # Non-card-reward decisions should not pollute reward skip accounting.
    tracker.update(
        decision_domain="combat",
        phase="combat",
        legal_actions=[],
        chosen_action={"kind": "play_card", "selection": "play"},
        chosen_signature={"kind": "play_card", "selection": "play"},
        selected_family="attack",
        selection="play",
    )

    for _ in range(4):
        tracker.update(
            decision_domain="build",
            phase="card_reward",
            legal_actions=[pick_action, skip_action],
            chosen_action=skip_action,
            chosen_signature=skip_action,
            selected_family="card_reward",
            selection="skip",
        )

    meta = tracker.as_metadata()
    assert meta["card_reward_seen_count"] == 7.0
    assert meta["card_reward_pick_count"] == 1.0
    assert meta["card_reward_skip_count"] == 6.0
    assert meta["card_reward_consecutive_skip_current"] == 4.0
    assert meta["card_reward_consecutive_skip_max"] == 4.0
    assert abs(meta["card_reward_pick_rate"] - (1.0 / 7.0)) < 1e-9
    assert abs(meta["card_reward_skip_rate"] - (6.0 / 7.0)) < 1e-9
