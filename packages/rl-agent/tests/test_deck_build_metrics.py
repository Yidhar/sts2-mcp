from __future__ import annotations

import math
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from muzero.diagnostics.deck_build_metrics import (
    CardRewardEpisodeTracker,
    DECK_COMPOSITION_KEYS,
    compact_deck_cards,
    compute_deck_composition_summary,
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

    assert set(summary.keys()) == set(DECK_QUALITY_V2_KEYS) | set(DECK_COMPOSITION_KEYS)
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
    assert summary["starter_count"] == 10.0
    assert summary["nonstarter_count"] == 1.0
    assert summary["upgraded_count"] == 5.0


def test_compute_deck_composition_summary_counts_starters_upgrades_and_nonstarters() -> None:
    deck = [
        {"id": "CARD.STRIKE_IRONCLAD", "title": "打击"},
        {"id": "CARD.STRIKE_IRONCLAD", "title": "打击+", "upgraded": True},
        {"id": "CARD.DEFEND_IRONCLAD", "title": "防御", "upgrade_level": 1},
        {"id": "CARD.BASH", "title": "痛击"},
        {"id": "CARD.BATTLE_TRANCE", "title": "战斗专注"},
    ]

    summary = compute_deck_composition_summary(deck)

    assert summary["starter_count"] == 3.0
    assert summary["starter_ratio"] == 3.0 / 5.0
    assert summary["nonstarter_count"] == 2.0
    assert summary["strike_count"] == 2.0
    assert summary["defend_count"] == 1.0
    assert summary["upgraded_count"] == 2.0
    assert summary["upgraded_ratio"] == 2.0 / 5.0
    assert summary["starter_upgrade_count"] == 2.0
    assert summary["starter_upgrade_ratio"] == 2.0 / 3.0


def test_compute_deck_composition_summary_recognises_english_and_upgraded_titles() -> None:
    deck = [
        {"card_id": "strike", "name": "Strike+"},
        {"card_id": "defend", "name": "Defend＋"},
        {"card_id": "CARD.STRIKE_IRONCLAD", "localized_title": "打击2"},
        {"card_id": "CARD.SHRUG_IT_OFF", "title": "耸肩无视", "upgrades": 1},
    ]

    summary = compute_deck_composition_summary(deck)

    assert summary["starter_count"] == 3.0
    assert summary["strike_count"] == 2.0
    assert summary["defend_count"] == 1.0
    assert summary["nonstarter_count"] == 1.0
    assert summary["upgraded_count"] == 3.0
    assert summary["starter_upgrade_count"] == 2.0


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


def test_card_reward_tracker_treats_compact_positional_reward_id_as_pick() -> None:
    """Post-guard replay may only keep ``card_reward:0`` for the final pick.

    The full bridge action contains the card payload, but compact replay/tracker
    paths can see only the positional action id.  That must still count as the
    final executed pick rather than as an ambiguous ``other`` folded into skip.
    """

    compact_pick = {"action_id": "card_reward:0"}
    compact_skip = {"action_id": "card_reward:skip"}

    tracker = CardRewardEpisodeTracker()
    tracker.update(
        decision_domain="build",
        phase="card_reward",
        legal_actions=[compact_pick, compact_skip],
        chosen_action=compact_pick,
        chosen_signature=compact_pick,
        selected_family="",
        selection="",
    )
    tracker.update(
        decision_domain="build",
        phase="card_reward",
        legal_actions=[compact_pick, compact_skip],
        chosen_action=compact_skip,
        chosen_signature=compact_skip,
        selected_family="",
        selection="",
    )

    meta = tracker.as_metadata()
    assert meta["card_reward_seen_count"] == 2.0
    assert meta["card_reward_pick_count"] == 1.0
    assert meta["card_reward_skip_count"] == 1.0
    assert meta["card_reward_other_count"] == 0.0
    assert meta["card_reward_consecutive_skip_current"] == 1.0
    assert meta["card_reward_consecutive_skip_max"] == 1.0


def test_card_reward_tracker_ignores_non_card_reward_claims_on_mixed_reward_surface() -> None:
    """Gold/relic/potion claims beside card rewards are not card skips."""

    card_pick = {"action_id": "card_reward:0", "kind": "card_reward", "selection": "pick"}
    card_skip = {"action_id": "card_reward:skip", "kind": "skip_card_reward", "selection": "skip"}
    claim_gold = {
        "action_id": "reward:gold",
        "kind": "claim_reward",
        "selection": "claim",
        "surface": "room_reward",
        "reward": {"type": "gold", "amount": 32},
    }

    tracker = CardRewardEpisodeTracker()
    tracker.update(
        decision_domain="build",
        phase="room_reward",
        legal_actions=[claim_gold, card_pick, card_skip],
        chosen_action=claim_gold,
        chosen_signature=claim_gold,
        selected_family="reward_pick",
        selection="claim",
    )

    meta = tracker.as_metadata()
    assert meta["card_reward_seen_count"] == 0.0
    assert meta["card_reward_pick_count"] == 0.0
    assert meta["card_reward_skip_count"] == 0.0


def test_card_reward_tracker_ignores_non_card_reward_claims_even_when_phase_is_card_reward() -> None:
    """Bridge phases can stay labelled card_reward while claiming safe rewards.

    The active full-run logs showed ``card_reward_other_count`` tracking almost
    exactly with ``skip_count`` because reward-cleanup actions were counted as
    card skips when ``phase == card_reward``.  This test locks the desired
    contract: only explicit card picks/skips count; gold/relic/potion cleanup
    does not.
    """

    card_pick = {"action_id": "card_reward:0", "kind": "card_reward", "selection": "pick"}
    card_skip = {"action_id": "card_reward:skip", "kind": "skip_card_reward", "selection": "skip"}
    claim_gold = {
        "action_id": "reward:gold",
        "kind": "claim_reward",
        "selection": "claim",
        "surface": "room_reward",
        "reward": {"type": "gold", "amount": 32},
    }
    claim_relic = {
        "action_id": "reward:relic",
        "kind": "claim_reward",
        "selection": "claim",
        "surface": "room_reward",
        "reward": {"type": "relic", "id": "RELIC.TEST"},
    }

    tracker = CardRewardEpisodeTracker()
    for action in (claim_gold, claim_relic):
        tracker.update(
            decision_domain="build",
            phase="card_reward",
            legal_actions=[action, card_pick, card_skip],
            chosen_action=action,
            chosen_signature=action,
            selected_family="reward_pick",
            selection="claim",
        )

    meta = tracker.as_metadata()
    assert meta["card_reward_seen_count"] == 0.0
    assert meta["card_reward_pick_count"] == 0.0
    assert meta["card_reward_skip_count"] == 0.0
    assert meta["card_reward_other_count"] == 0.0


def test_card_reward_tracker_ignores_post_card_reward_proceed_when_no_card_choice_left() -> None:
    """Proceed/close after the card reward was resolved is not another skip."""

    proceed = {
        "action_id": "reward:proceed",
        "kind": "proceed",
        "selection": "proceed",
        "surface": "room_reward",
    }

    tracker = CardRewardEpisodeTracker()
    tracker.update(
        decision_domain="build",
        phase="card_reward",
        legal_actions=[proceed],
        chosen_action=proceed,
        chosen_signature=proceed,
        selected_family="proceed",
        selection="proceed",
    )

    meta = tracker.as_metadata()
    assert meta["card_reward_seen_count"] == 0.0
    assert meta["card_reward_pick_count"] == 0.0
    assert meta["card_reward_skip_count"] == 0.0
    assert meta["card_reward_other_count"] == 0.0


def test_card_reward_tracker_counts_explicit_skip_on_mixed_reward_surface() -> None:
    card_pick = {"action_id": "card_reward:0", "kind": "card_reward", "selection": "pick"}
    card_skip = {"action_id": "card_reward:skip", "kind": "skip_card_reward", "selection": "skip"}

    tracker = CardRewardEpisodeTracker()
    tracker.update(
        decision_domain="build",
        phase="room_reward",
        legal_actions=[card_pick, card_skip],
        chosen_action=card_skip,
        chosen_signature=card_skip,
        selected_family="skip",
        selection="skip",
    )

    meta = tracker.as_metadata()
    assert meta["card_reward_seen_count"] == 1.0
    assert meta["card_reward_pick_count"] == 0.0
    assert meta["card_reward_skip_count"] == 1.0
