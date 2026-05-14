"""Phase 2 route heuristic unit tests (recovery 2026-05-08).

Mirrors the validation scenarios in
``docs/muzero-route-deck-long-horizon-review-20260508.md`` §7.5–7.11:

* Low HP + elite + no rest → score low.
* Low HP + rest before elite → score higher than no-rest variant.
* High gold + shop → score higher than 0-gold same fork.
* Strong deck + elite → penalty smaller than weak deck same fork.
* Weak deck + elite → penalty larger.

Plus pure schema robustness:

* Missing route_summary returns safe-zero.
* Non-dict action and non-map action return None when ranked.
* All output keys present regardless of input.
* Score is always finite.
"""
from __future__ import annotations

import math
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from sts2_env.route_heuristic import (
    REST_URGENCY_HP_RATIO,
    ROUTE_LOW_HP_RATIO,
    compute_elite_risk,
    count_non_empty_potions,
    rank_legal_route_actions,
    score_route_action,
)
from sts2_env.deck_quality import deck_quality_v2


_OUTPUT_KEYS = {
    "score", "score_normalized",
    "boss_progress", "rest_value", "shop_value",
    "treasure_value", "event_value", "branch_value",
    "unsafe_elite_penalty", "forced_elite_penalty",
    "no_rest_before_elite_penalty", "low_hp_monster_chain_penalty",
    "elite_risk_factor", "low_hp_elite_flag", "low_hp_flag",
    "rest_before_elite_available", "summary_used",
    # Phase 3 prep (review §P1-2): elite count split.
    "forced_elite_count", "immediate_elite_count",
    "optional_elite_count", "subtree_elite_count",
    # Recovery route hard-guard prep.
    "risk_class", "risk_reason",
}


def _summary(**overrides) -> dict:
    base = {
        "reachable_node_count": 8,
        "max_depth": 6,
        "direct_child_count": 2,
        "forced_path_steps_before_branch": 0,
        "count_monster": 3,
        "count_elite": 0,
        "count_boss": 1,
        "count_event": 1,
        "count_question_mark": 0,
        "count_rest_site": 1,
        "count_shop": 0,
        "count_treasure": 0,
        "next_elite_steps": 99,
        "next_rest_steps": 4,
        "next_shop_steps": 99,
        "next_event_steps": 2,
        "next_question_mark_steps": 99,
        "next_treasure_steps": 99,
        "can_reach_rest_site_before_elite": False,
        "can_reach_elite_then_rest_site": False,
    }
    base.update(overrides)
    return base


def _strong_deck() -> dict[str, float]:
    """Synthetic deck that scores high frontload + block."""
    cards = [
        {"id": "CARD.STRIKE_IRONCLAD", "type": "Attack", "cost": 1, "energy_cost": 1},
        {"id": "CARD.STRIKE_IRONCLAD", "type": "Attack", "cost": 1, "energy_cost": 1},
        {"id": "CARD.STRIKE_IRONCLAD", "type": "Attack", "cost": 1, "energy_cost": 1},
        {"id": "CARD.STRIKE_IRONCLAD", "type": "Attack", "cost": 1, "energy_cost": 1},
        {"id": "CARD.DEFEND_IRONCLAD", "type": "Skill", "cost": 1, "energy_cost": 1},
        {"id": "CARD.DEFEND_IRONCLAD", "type": "Skill", "cost": 1, "energy_cost": 1},
        {"id": "CARD.DEFEND_IRONCLAD", "type": "Skill", "cost": 1, "energy_cost": 1},
        {"id": "CARD.DEFEND_IRONCLAD", "type": "Skill", "cost": 1, "energy_cost": 1},
        {"id": "CARD.BASH", "type": "Attack", "cost": 2, "energy_cost": 2},
        {"id": "CARD.BASH", "type": "Attack", "cost": 2, "energy_cost": 2},
    ]
    return deck_quality_v2(cards)


def _weak_polluted_deck() -> dict[str, float]:
    """Deck dragged down by curses / few attacks."""
    cards = [
        {"id": "CARD.STRIKE_IRONCLAD", "type": "Attack", "cost": 1, "energy_cost": 1},
        {"id": "CARD.STRIKE_IRONCLAD", "type": "Attack", "cost": 1, "energy_cost": 1},
        {"id": "CARD.DEFEND_IRONCLAD", "type": "Skill", "cost": 1, "energy_cost": 1},
        {"id": "CARD.CURSE_REGRET", "type": "Curse", "cost": -1},
        {"id": "CARD.CURSE_REGRET", "type": "Curse", "cost": -1},
        {"id": "CARD.WOUND", "type": "Status", "cost": -1},
        {"id": "CARD.WOUND", "type": "Status", "cost": -1},
        {"id": "CARD.WOUND", "type": "Status", "cost": -1},
    ]
    return deck_quality_v2(cards)


def test_returns_full_key_set_on_missing_summary():
    out = score_route_action(
        route_summary=None,
        deck_quality=None,
        hp=70, max_hp=80, gold=100, potion_count=2,
    )
    assert set(out.keys()) == _OUTPUT_KEYS
    assert out["summary_used"] is False
    assert out["score"] == 0.0


def test_non_dict_summary_returns_safe_zero():
    out = score_route_action(
        route_summary="not a dict",  # type: ignore[arg-type]
        deck_quality=None,
        hp=70, max_hp=80, gold=100, potion_count=2,
    )
    assert out["summary_used"] is False


def test_all_outputs_finite_under_adversarial_summary():
    weird = _summary(
        reachable_node_count=999, count_elite=99, count_rest_site=99,
        next_elite_steps=0, can_reach_rest_site_before_elite=True,
    )
    out = score_route_action(
        route_summary=weird, deck_quality=_strong_deck(),
        hp=1, max_hp=80, gold=999, potion_count=99,
    )
    for value in out.values():
        if isinstance(value, (int, float)):
            assert math.isfinite(value), value


def test_low_hp_no_rest_elite_is_worse_than_rest_before_elite():
    """Rest-before-elite available → smaller no_rest penalty + bigger
    rest_value → final score higher."""
    risky = _summary(count_elite=1, next_elite_steps=2, can_reach_rest_site_before_elite=False, count_rest_site=0)
    safe = _summary(count_elite=1, next_elite_steps=2, can_reach_rest_site_before_elite=True, count_rest_site=1, next_rest_steps=1)
    risky_score = score_route_action(
        route_summary=risky, deck_quality=_strong_deck(),
        hp=24, max_hp=80, gold=100, potion_count=1,
    )
    safe_score = score_route_action(
        route_summary=safe, deck_quality=_strong_deck(),
        hp=24, max_hp=80, gold=100, potion_count=1,
    )
    assert safe_score["score"] > risky_score["score"]
    # Hole F watch: low_hp_flag should trigger on both.
    assert risky_score["low_hp_flag"] == 1.0
    assert safe_score["low_hp_flag"] == 1.0
    # low_hp_elite_flag should also trigger on both since both have an elite.
    assert risky_score["low_hp_elite_flag"] == 1.0
    assert safe_score["low_hp_elite_flag"] == 1.0
    # The no_rest penalty must vanish on the safe path.
    assert risky_score["no_rest_before_elite_penalty"] > 0.0
    assert safe_score["no_rest_before_elite_penalty"] == 0.0


def test_strong_deck_elite_penalty_smaller_than_weak_deck():
    summary = _summary(count_elite=1, next_elite_steps=2, can_reach_rest_site_before_elite=False)
    strong = score_route_action(
        route_summary=summary, deck_quality=_strong_deck(),
        hp=70, max_hp=80, gold=100, potion_count=2,
    )
    weak = score_route_action(
        route_summary=summary, deck_quality=_weak_polluted_deck(),
        hp=70, max_hp=80, gold=100, potion_count=2,
    )
    assert strong["unsafe_elite_penalty"] < weak["unsafe_elite_penalty"]
    assert strong["elite_risk_factor"] < weak["elite_risk_factor"]
    assert strong["score"] > weak["score"]


def test_high_gold_makes_shop_more_valuable():
    summary = _summary(count_shop=1, count_rest_site=0, count_elite=0)
    rich = score_route_action(
        route_summary=summary, deck_quality=_strong_deck(),
        hp=70, max_hp=80, gold=300, potion_count=2,
    )
    poor = score_route_action(
        route_summary=summary, deck_quality=_strong_deck(),
        hp=70, max_hp=80, gold=20, potion_count=2,
    )
    assert rich["shop_value"] > poor["shop_value"]
    assert rich["score"] > poor["score"]


def test_pollution_deck_increases_shop_value():
    summary = _summary(count_shop=1, count_elite=0)
    polluted = score_route_action(
        route_summary=summary, deck_quality=_weak_polluted_deck(),
        hp=70, max_hp=80, gold=200, potion_count=2,
    )
    clean = score_route_action(
        route_summary=summary, deck_quality=_strong_deck(),
        hp=70, max_hp=80, gold=200, potion_count=2,
    )
    # Shop value adds pollution_score directly — polluted > clean here.
    assert polluted["shop_value"] > clean["shop_value"]


def test_forced_elite_penalty_only_when_path_locked():
    branched = _summary(
        count_elite=1, forced_path_steps_before_branch=0, next_elite_steps=2,
    )
    forced = _summary(
        count_elite=1, forced_path_steps_before_branch=2, next_elite_steps=1,
    )
    branched_out = score_route_action(
        route_summary=branched, deck_quality=_strong_deck(),
        hp=70, max_hp=80, gold=100, potion_count=2,
    )
    forced_out = score_route_action(
        route_summary=forced, deck_quality=_strong_deck(),
        hp=70, max_hp=80, gold=100, potion_count=2,
    )
    assert forced_out["forced_elite_penalty"] > 0.0
    assert branched_out["forced_elite_penalty"] == 0.0


def test_low_hp_monster_chain_penalty_kicks_in():
    summary = _summary(count_monster=4, count_rest_site=0, count_elite=0)
    healthy = score_route_action(
        route_summary=summary, deck_quality=_strong_deck(),
        hp=70, max_hp=80, gold=100, potion_count=2,
    )
    bleeding = score_route_action(
        route_summary=summary, deck_quality=_strong_deck(),
        hp=15, max_hp=80, gold=100, potion_count=2,
    )
    assert bleeding["low_hp_monster_chain_penalty"] > 0.0
    assert healthy["low_hp_monster_chain_penalty"] == 0.0


def test_rank_legal_route_actions_skips_non_map_actions():
    legal = [
        {"kind": "play_card", "card": {"id": "CARD.STRIKE_IRONCLAD"}},
        {
            "kind": "map",
            "route_summary": _summary(count_elite=1, next_elite_steps=2),
        },
        {
            "kind": "map",
            "route_summary": _summary(count_rest_site=2, count_elite=0, next_rest_steps=1),
        },
        "not a dict",
    ]
    ranked = rank_legal_route_actions(
        legal_actions=legal, deck_quality=_strong_deck(),
        hp=24, max_hp=80, gold=100, potion_count=1,
    )
    assert ranked[0] is None  # play_card
    assert ranked[1] is not None  # map with elite
    assert ranked[2] is not None  # map with rest
    assert ranked[3] is None  # bad input
    # Rest action should outrank elite action when low HP.
    assert ranked[2]["score"] > ranked[1]["score"]


def test_compute_elite_risk_within_bounds():
    """Risk must clamp into [0.1, 2.0]."""
    extreme_low = compute_elite_risk(
        deck_quality=_weak_polluted_deck(), hp_ratio=0.0,
        potion_count=0, rest_before_elite=False, next_elite_steps=1,
    )
    extreme_high = compute_elite_risk(
        deck_quality=_strong_deck(), hp_ratio=1.0,
        potion_count=5, rest_before_elite=True, next_elite_steps=10,
    )
    assert 0.1 <= extreme_low <= 2.0
    assert 0.1 <= extreme_high <= 2.0
    assert extreme_low > extreme_high


def test_optional_elite_penalty_lighter_than_forced():
    """P1-2 split: a candidate with an optional elite (next_elite > forced
    branch point) should incur a smaller penalty than one with the same
    elite forced onto the path."""
    optional = _summary(
        count_elite=1,
        next_elite_steps=5,
        forced_path_steps_before_branch=2,  # branch happens before elite
    )
    forced = _summary(
        count_elite=1,
        next_elite_steps=2,
        forced_path_steps_before_branch=2,  # elite is forced onto the path
    )
    optional_out = score_route_action(
        route_summary=optional, deck_quality=_strong_deck(),
        hp=70, max_hp=80, gold=100, potion_count=2,
    )
    forced_out = score_route_action(
        route_summary=forced, deck_quality=_strong_deck(),
        hp=70, max_hp=80, gold=100, potion_count=2,
    )
    assert forced_out["forced_elite_count"] >= 1.0
    assert optional_out["forced_elite_count"] == 0.0
    assert optional_out["optional_elite_count"] >= 1.0
    assert optional_out["unsafe_elite_penalty"] < forced_out["unsafe_elite_penalty"]
    # immediate flag fires only when next_elite_steps <= 1.
    assert optional_out["immediate_elite_count"] == 0.0
    assert forced_out["immediate_elite_count"] == 0.0


def test_immediate_elite_flag_set_when_next_step_is_elite():
    summary = _summary(count_elite=1, next_elite_steps=1, forced_path_steps_before_branch=2)
    out = score_route_action(
        route_summary=summary, deck_quality=_strong_deck(),
        hp=70, max_hp=80, gold=100, potion_count=2,
    )
    assert out["immediate_elite_count"] == 1.0


def test_branch_value_capped_under_huge_subtree():
    """P1-3 cap: branch_value must NOT exceed 0.8 even with a 30-node
    subtree and 4 direct children."""
    huge = _summary(
        reachable_node_count=30,
        direct_child_count=4,
        forced_path_steps_before_branch=0,
        count_elite=0,
        count_rest_site=0,
        count_shop=0,
    )
    out = score_route_action(
        route_summary=huge, deck_quality=_strong_deck(),
        hp=70, max_hp=80, gold=100, potion_count=2,
    )
    assert out["branch_value"] <= 0.8 + 1e-9


def test_subtree_elite_count_separate_from_forced():
    summary = _summary(
        count_elite=3,
        next_elite_steps=4,
        forced_path_steps_before_branch=2,
    )
    out = score_route_action(
        route_summary=summary, deck_quality=_strong_deck(),
        hp=70, max_hp=80, gold=100, potion_count=2,
    )
    assert out["subtree_elite_count"] == 3.0
    assert out["forced_elite_count"] == 0.0  # elite is past branch
    assert out["optional_elite_count"] == 3.0


def test_count_non_empty_potions_handles_all_encodings():
    """Bridge slot encoding is heterogeneous — exclude all empty markers."""
    potions = [
        "[empty]",                                # empty marker string
        "",                                        # blank
        "Empty",                                   # case-insensitive
        "POTION.NONE",                             # canonical empty id
        "POTION.BLOOD_VIAL",                       # real potion (string form)
        {"empty": True, "title": "should be ignored"},
        {"title": "[empty]"},
        {"title": "潜能药水", "id": "POTION.POTENCY"},
        {"id": "POTION.SWIFT"},
        {"name": "explosive"},
        {},                                        # nothing usable
        None,                                      # not a slot
    ]
    assert count_non_empty_potions(potions) == 4
    assert count_non_empty_potions(None) == 0
    assert count_non_empty_potions("not a list") == 0
    assert count_non_empty_potions([]) == 0


def test_low_hp_thresholds_documented():
    assert ROUTE_LOW_HP_RATIO == 0.40
    assert REST_URGENCY_HP_RATIO == 0.50
    assert ROUTE_LOW_HP_RATIO < REST_URGENCY_HP_RATIO


def test_forced_elite_penalty_handles_none_forced_steps():
    """Linear chain to elite — bridge sets forced_path_steps_before_branch=None."""
    summary = _summary(
        count_elite=1,
        next_elite_steps=2,
        forced_path_steps_before_branch=None,  # type: ignore[arg-type]
    )
    out = score_route_action(
        route_summary=summary, deck_quality=_strong_deck(),
        hp=70, max_hp=80, gold=100, potion_count=2,
    )
    # Linear-chain elite must be flagged as forced.
    assert out["forced_elite_penalty"] > 0.0


def test_score_normalized_clamped():
    summary = _summary(
        count_elite=99, count_rest_site=99, count_shop=99,
        count_treasure=99, count_boss=10, count_event=99,
        reachable_node_count=999,
    )
    out = score_route_action(
        route_summary=summary, deck_quality=_strong_deck(),
        hp=70, max_hp=80, gold=999, potion_count=99,
    )
    assert -1.0 <= out["score_normalized"] <= 1.0
