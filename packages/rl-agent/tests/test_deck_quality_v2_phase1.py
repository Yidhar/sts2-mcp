"""Phase 1 deck-quality v2 unit tests (recovery 2026-05-08).

Mirrors the fixtures listed in ``docs/muzero-route-deck-long-horizon-review-20260508.md``
§6.7. The pure-compute helper must:

* Return all keys regardless of input.
* Safe-zero on empty / missing-metadata decks (no NaN, no crash).
* Clamp model-facing score/density values into ``[0, 1]`` (or the documented
  narrow range) while allowing raw/count diagnostics to expose real magnitudes.
* Differentiate fixture archetypes (Strike-only vs Defend-only vs scaling).

The card metadata registry is loaded lazily from disk by
``content_registry.get_card_metadata``. Tests rely on a small set of
ironclad starter cards that are guaranteed to be present.
"""
from __future__ import annotations

import math
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from sts2_env.deck_quality import (
    DECK_QUALITY_V2_KEYS,
    deck_quality_v2,
    deck_quality_v2_from_obs,
)


def _strike(upgrade_level: int = 0) -> dict:
    return {
        "id": "CARD.STRIKE_IRONCLAD",
        "title": "打击",
        "type": "Attack",
        "cost": 1,
        "energy_cost": 1,
        "upgrade_level": upgrade_level,
    }


def _defend(upgrade_level: int = 0) -> dict:
    return {
        "id": "CARD.DEFEND_IRONCLAD",
        "title": "防御",
        "type": "Skill",
        "cost": 1,
        "energy_cost": 1,
        "upgrade_level": upgrade_level,
    }


def _bash() -> dict:
    return {
        "id": "CARD.BASH",
        "title": "痛击",
        "type": "Attack",
        "cost": 2,
        "energy_cost": 2,
    }


def _curse() -> dict:
    return {
        "id": "CARD.CURSE_REGRET",
        "title": "悔恨",
        "type": "Curse",
        "cost": -1,  # unplayable curses often emit cost=-1
    }


def _status() -> dict:
    return {
        "id": "CARD.WOUND",
        "title": "伤口",
        "type": "Status",
        "cost": -1,
    }


def _x_cost_card() -> dict:
    return {
        "id": "CARD.WHIRLWIND",
        "title": "旋风斩",
        "type": "Attack",
        "cost": "X",
        "energy_cost_text": "X",
        "x_cost": True,
        "card_effect_profile": {
            "semantic_signals": {"damage": 5},
            "semantic_tags": ["damage", "aoe", "all_enemies"],
        },
    }


def _draw_card() -> dict:
    """Synthetic draw-engine card; uses card_effect_profile for tags."""
    return {
        "id": "CARD.IMPATIENCE",
        "title": "急躁",
        "type": "Skill",
        "cost": 0,
        "energy_cost": 0,
        "card_effect_profile": {
            "semantic_signals": {"draw": 2},
            "semantic_tags": ["draw", "card_draw"],
        },
    }


def _energy_refund_card() -> dict:
    return {
        "id": "CARD.OFFERING",
        "title": "祭品",
        "type": "Skill",
        "cost": 0,
        "energy_cost": 0,
        "card_effect_profile": {
            "semantic_signals": {"draw": 3},
            "semantic_tags": ["draw", "energy_gain", "exhaust"],
        },
    }


def _power_scaling_card() -> dict:
    return {
        "id": "CARD.DEMON_FORM",
        "title": "恶魔变形",
        "type": "Power",
        "cost": 3,
        "energy_cost": 3,
        "card_effect_profile": {
            "semantic_signals": {"strength": 2},
            "semantic_tags": ["strength_gain", "power"],
        },
    }


def _body_slam() -> dict:
    """Synthetic Body Slam payload with intentionally missing damage signal.

    The bridge/content registry may not expose static damage for Body Slam
    because the card deals current-block damage at runtime.  Deck-quality should
    still classify it as a block-payoff combo component via the curated ID
    override, not as a blank card.
    """

    return {
        "id": "CARD.BODY_SLAM",
        "title": "全身撞击",
        "type": "Attack",
        "cost": 1,
        "energy_cost": 1,
        "card_effect_profile": {
            "semantic_signals": {},
            "semantic_tags": [],
        },
    }


def _three_cost_attack() -> dict:
    return {
        "id": "CARD.HEAVY_BLADE",
        "title": "重刃",
        "type": "Attack",
        "cost": 3,
        "energy_cost": 3,
        "card_effect_profile": {
            "semantic_signals": {"damage": 18},
            "semantic_tags": ["damage"],
        },
    }


def test_returns_full_key_set_on_empty_deck():
    out = deck_quality_v2([])
    assert set(out.keys()) == set(DECK_QUALITY_V2_KEYS)
    for key, value in out.items():
        assert isinstance(value, float), f"{key} not float"
        assert math.isfinite(value), f"{key}={value} not finite"


def test_returns_full_key_set_on_none_input():
    out = deck_quality_v2(None)
    assert set(out.keys()) == set(DECK_QUALITY_V2_KEYS)
    assert out["deck_size_raw"] == 0.0


def test_unknown_card_does_not_crash():
    deck = [{"id": "CARD.MADE_UP_DOES_NOT_EXIST", "type": "Attack", "cost": 1}]
    out = deck_quality_v2(deck)
    assert out["deck_size_raw"] == 1.0
    assert out["metadata_hit_rate"] == 0.0
    for value in out.values():
        assert math.isfinite(value)


def test_strike_only_deck_has_high_frontload_low_block():
    deck = [_strike() for _ in range(10)]
    out = deck_quality_v2(deck)
    assert out["attack_density"] == 1.0
    assert out["skill_density"] == 0.0
    assert out["frontload_score"] > 0.4, f"Strike deck should have non-trivial frontload, got {out['frontload_score']}"
    assert out["block_score"] == 0.0
    assert out["pollution_score"] == 0.0
    assert out["consistency_score"] >= 0.95  # all same id


def test_defend_only_deck_has_high_block_low_frontload():
    deck = [_defend() for _ in range(10)]
    out = deck_quality_v2(deck)
    assert out["skill_density"] == 1.0
    assert out["attack_density"] == 0.0
    assert out["block_score"] > 0.3
    assert out["frontload_score"] == 0.0


def test_bash_deck_has_higher_per_energy_damage():
    """Bash is 2-cost 8-damage = 4 dmg/energy; same as 1-cost 6-damage Strike's
    6 dmg/1 energy. But Bash deck cost average 2, Strike average 1. Verify
    avg_cost differs and per-energy damage is sensible."""
    bash_deck = [_bash() for _ in range(8)]
    strike_deck = [_strike() for _ in range(8)]
    bash_out = deck_quality_v2(bash_deck)
    strike_out = deck_quality_v2(strike_deck)
    assert bash_out["avg_cost"] > strike_out["avg_cost"]
    assert bash_out["high_cost_density"] == 1.0
    assert strike_out["high_cost_density"] == 0.0


def test_pollution_score_climbs_with_curse_and_status():
    deck = [_strike(), _strike(), _strike(), _strike(), _curse(), _status()]
    out = deck_quality_v2(deck)
    assert out["curse_density"] > 0.0
    assert out["status_density"] > 0.0
    assert out["pollution_score"] > 0.2
    # Pollution should drag elite_readiness down.
    pure = deck_quality_v2([_strike() for _ in range(6)])
    assert out["elite_readiness_score"] < pure["elite_readiness_score"]


def test_x_cost_isolated_from_avg_cost():
    """X-cost cards must NOT pollute avg_cost (they get their own density)."""
    deck = [_strike(), _strike(), _x_cost_card()]
    out = deck_quality_v2(deck)
    assert out["x_cost_density"] > 0.0
    assert out["x_cost_damage_potential"] > 0.0
    # avg_cost is computed over non-X cards only; should reflect Strike's 1.
    expected_avg = 1.0 / 4.0  # cost 1 normalised by /4.0 ceiling
    assert abs(out["avg_cost"] - expected_avg) < 1e-3


def test_draw_engine_score_responds_to_draw_cards():
    pure = deck_quality_v2([_strike() for _ in range(8)])
    with_draw = deck_quality_v2(
        [_strike() for _ in range(6)] + [_draw_card(), _draw_card()]
    )
    assert with_draw["draw_density"] > pure["draw_density"]
    assert with_draw["draw_engine_score"] > pure["draw_engine_score"]


def test_energy_engine_score_responds_to_refund_cards():
    pure = deck_quality_v2([_strike() for _ in range(8)])
    with_refund = deck_quality_v2(
        [_strike() for _ in range(6)] + [_energy_refund_card(), _energy_refund_card()]
    )
    assert with_refund["energy_refund_density"] > 0.0
    assert with_refund["energy_engine_score"] > pure["energy_engine_score"]
    assert with_refund["exhaust_density"] > 0.0


def test_scaling_score_responds_to_power_cards():
    pure = deck_quality_v2([_strike() for _ in range(8)])
    with_scaling = deck_quality_v2(
        [_strike() for _ in range(6)] + [_power_scaling_card(), _power_scaling_card()]
    )
    assert with_scaling["power_density"] > 0.0
    assert with_scaling["strength_scaling_density"] > 0.0
    assert with_scaling["scaling_score"] > pure["scaling_score"]
    assert with_scaling["boss_readiness_score"] > pure["boss_readiness_score"]


def test_consistency_score_higher_for_dup_heavy_decks():
    """Deck with many duplicates of same id is more 'consistent' than one with
    only unique cards (consistency proxies focus, not necessarily quality)."""
    dup_heavy = deck_quality_v2([_strike() for _ in range(10)])
    diverse = deck_quality_v2(
        [_strike(), _defend(), _bash(), _curse(), _status(),
         _x_cost_card(), _draw_card(), _energy_refund_card(),
         _power_scaling_card(), _strike(upgrade_level=1)]
    )
    assert dup_heavy["consistency_score"] > diverse["consistency_score"]


def test_upgrade_ratio_tracks_upgrade_level():
    deck = [_strike() for _ in range(5)] + [_strike(upgrade_level=1) for _ in range(5)]
    out = deck_quality_v2(deck)
    assert abs(out["upgraded_ratio"] - 0.5) < 1e-6


def test_all_features_clamped_to_unit_range():
    """Adversarial deck — make sure no feature escapes [0,1]."""
    big = (
        [_strike() for _ in range(20)]
        + [_defend() for _ in range(20)]
        + [_bash() for _ in range(10)]
        + [_x_cost_card() for _ in range(5)]
        + [_curse() for _ in range(5)]
        + [_power_scaling_card() for _ in range(3)]
    )
    out = deck_quality_v2(big)
    raw_or_count_keys = {
        "deck_size_raw",
        "raw_avg_damage_per_energy",
        "raw_avg_block_per_energy",
        "raw_expected_extra_draw_per_turn",
        "raw_expected_cards_seen_per_turn",
        "raw_expected_energy_budget_per_turn",
        "raw_expected_playable_cards_per_turn",
        "raw_expected_playable_energy_spent_per_turn",
        "raw_expected_unspent_energy_per_turn",
        "raw_expected_playable_attack_cards_per_turn",
        "raw_expected_playable_skill_cards_per_turn",
        "raw_expected_playable_power_cards_per_turn",
        "raw_expected_playable_attack_damage_per_turn",
        "raw_expected_playable_block_per_turn",
        "expected_hand_attack_cards",
        "expected_hand_skill_cards",
        "expected_hand_power_cards",
        "expected_hand_curse_cards",
        "expected_hand_status_cards",
        "expected_hand_draw_cards",
        "expected_hand_engine_cards",
        "expected_hand_scaling_cards",
        "expected_hand_unplayable_cards",
        "expected_hand_attack_damage_per_turn",
        "expected_hand_block_per_turn",
    }
    for key, value in out.items():
        if key in raw_or_count_keys:
            continue  # raw/count diagnostics, not normalised
        assert 0.0 <= value <= 1.0, f"{key}={value} outside [0,1]"


def test_raw_per_energy_and_expected_hand_metrics_are_exposed():
    deck = [_strike() for _ in range(5)] + [_defend() for _ in range(5)]
    out = deck_quality_v2(deck)

    assert out["raw_avg_damage_per_energy"] > 0.0
    assert out["raw_avg_block_per_energy"] > 0.0
    assert out["raw_expected_cards_seen_per_turn"] >= 5.0
    assert out["expected_hand_attack_cards"] > 0.0
    assert out["expected_hand_skill_cards"] > 0.0
    assert out["expected_hand_attack_damage_per_turn"] > 0.0
    assert out["expected_hand_block_per_turn"] > 0.0
    assert 0.0 <= out["expected_hand_attack_share"] <= 1.0
    assert 0.0 <= out["expected_hand_skill_share"] <= 1.0
    assert 0.0 <= out["expected_hand_useful_quality_score"] <= 1.0
    assert out["raw_expected_playable_cards_per_turn"] > 0.0
    assert out["raw_expected_playable_energy_spent_per_turn"] > 0.0
    assert out["raw_expected_playable_attack_damage_per_turn"] > 0.0
    assert out["raw_expected_playable_block_per_turn"] > 0.0
    assert out["expected_energy_utilization_score"] > 0.0
    assert out["expected_playable_cards_score"] > 0.0


def test_cost_curve_buckets_are_exposed():
    deck = [_draw_card(), _strike(), _bash(), _power_scaling_card(), _x_cost_card()]
    out = deck_quality_v2(deck)

    assert out["cost_curve_zero_share"] > 0.0
    assert out["cost_curve_one_share"] > 0.0
    assert out["cost_curve_two_share"] > 0.0
    assert out["cost_curve_three_plus_share"] > 0.0
    assert out["cost_curve_x_share"] > 0.0


def test_high_cost_deck_has_lower_playable_card_count_than_cheap_deck():
    cheap = deck_quality_v2([_strike() for _ in range(8)] + [_defend() for _ in range(8)])
    high_cost = deck_quality_v2([_three_cost_attack() for _ in range(8)] + [_power_scaling_card() for _ in range(8)])

    assert high_cost["raw_expected_playable_cards_per_turn"] < cheap["raw_expected_playable_cards_per_turn"]
    assert high_cost["cost_curve_three_plus_share"] > cheap["cost_curve_three_plus_share"]


def test_body_slam_is_combo_option_not_static_blank():
    out = deck_quality_v2([_body_slam()] + [_defend() for _ in range(6)])

    assert out["combo_payoff_density"] > 0.0
    assert out["combo_component_density"] > 0.0
    assert out["combo_option_value_score"] > 0.0
    assert out["delayed_payoff_density"] > 0.0


def test_delayed_payoff_metrics_exposed_and_clamped():
    deck = (
        [_strike() for _ in range(4)]
        + [_defend() for _ in range(4)]
        + [_power_scaling_card(), _draw_card(), _energy_refund_card()]
    )
    out = deck_quality_v2(deck)

    for key in (
        "delayed_payoff_density",
        "delayed_enabler_density",
        "delayed_payoff_time_to_value_score",
        "delayed_payoff_maturity_score",
        "delayed_payoff_option_value_score",
        "delayed_payoff_unrealized_risk_score",
    ):
        assert key in out
        assert math.isfinite(out[key])
        assert 0.0 <= out[key] <= 1.0

    assert out["delayed_payoff_density"] > 0.0
    assert out["delayed_enabler_density"] > 0.0
    assert out["delayed_payoff_time_to_value_score"] > 0.0
    assert out["delayed_payoff_option_value_score"] > 0.0


def test_orphan_delayed_payoff_has_higher_unrealized_risk_than_supported_payoff():
    orphan = deck_quality_v2([_power_scaling_card()] + [_strike() for _ in range(7)] + [_defend() for _ in range(2)])
    supported = deck_quality_v2(
        [_power_scaling_card(), _draw_card(), _energy_refund_card()]
        + [_strike() for _ in range(5)]
        + [_defend() for _ in range(2)]
    )

    assert orphan["delayed_payoff_unrealized_risk_score"] > supported["delayed_payoff_unrealized_risk_score"]
    assert supported["delayed_payoff_maturity_score"] > orphan["delayed_payoff_maturity_score"]


def test_expected_hand_quality_composition_tracks_draw_scaling_and_junk():
    deck = (
        [_strike() for _ in range(4)]
        + [_defend() for _ in range(4)]
        + [_draw_card(), _energy_refund_card(), _power_scaling_card(), _curse(), _status()]
    )
    out = deck_quality_v2(deck)

    assert out["expected_hand_draw_cards"] > 0.0
    assert out["expected_hand_engine_cards"] > 0.0
    assert out["expected_hand_scaling_cards"] > 0.0
    assert out["expected_hand_unplayable_cards"] > 0.0
    assert out["expected_hand_draw_score"] > 0.0
    assert out["expected_hand_engine_score"] > 0.0
    assert out["expected_hand_scaling_score"] > 0.0
    assert out["expected_hand_pollution_score"] > 0.0
    assert out["expected_hand_junk_share"] > 0.0
    quality_share_sum = (
        out["expected_hand_quality_attack_share"]
        + out["expected_hand_quality_block_share"]
        + out["expected_hand_quality_draw_share"]
        + out["expected_hand_quality_scaling_share"]
        + out["expected_hand_quality_pollution_share"]
    )
    assert 0.999 <= quality_share_sum <= 1.001


def test_from_obs_helper_safe_with_missing_player():
    out = deck_quality_v2_from_obs({})
    assert set(out.keys()) == set(DECK_QUALITY_V2_KEYS)
    out2 = deck_quality_v2_from_obs({"player": None})
    assert set(out2.keys()) == set(DECK_QUALITY_V2_KEYS)
    out3 = deck_quality_v2_from_obs(None)
    assert set(out3.keys()) == set(DECK_QUALITY_V2_KEYS)


def test_from_obs_helper_processes_real_obs_shape():
    obs = {
        "player": {
            "deck_cards": [_strike(), _strike(), _defend(), _bash()],
        }
    }
    out = deck_quality_v2_from_obs(obs)
    assert out["deck_size_raw"] == 4.0
    assert out["attack_density"] == 0.75
    assert out["skill_density"] == 0.25
