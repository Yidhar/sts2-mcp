from __future__ import annotations

import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def _card(
    card_id: str,
    *,
    title: str | None = None,
    type_: str = "Attack",
    cost: int = 1,
    damage: float = 0.0,
    block: float = 0.0,
    draw: float = 0.0,
    tags: list[str] | None = None,
) -> dict:
    return {
        "id": card_id,
        "title": title or card_id,
        "type": type_,
        "cost": cost,
        "card_effect_profile": {
            "semantic_signals": {"damage": damage, "block": block, "draw": draw},
            "semantic_tags": list(tags or []),
        },
    }


def _thin_low_block_deck() -> list[dict]:
    return [_card(f"strike_{idx}", type_="Attack", damage=6) for idx in range(5)] + [
        _card("weak_defend", type_="Skill", block=5)
    ]


def _thin_low_attack_deck() -> list[dict]:
    return [_card(f"defend_{idx}", type_="Skill", block=10) for idx in range(8)]


def _healthy_deck() -> list[dict]:
    return (
        [_card(f"attack_{idx}", type_="Attack", damage=14) for idx in range(8)]
        + [_card(f"block_{idx}", type_="Skill", block=10) for idx in range(8)]
        + [_card(f"draw_{idx}", type_="Skill", draw=2, tags=["draw"]) for idx in range(5)]
        + [_card(f"scaling_{idx}", type_="Power", tags=["strength"]) for idx in range(4)]
    )


def _raw(deck: list[dict], *, floor: int = 4, act_id: int = 0) -> dict:
    return {
        "run": {"current_floor": floor, "act_id": act_id},
        "player": {"hp": 70, "max_hp": 80, "deck_cards": deck},
    }


def _raw_with_map_nodes(deck: list[dict], *, floor: int, nodes: list[dict], act_id: int = 0) -> dict:
    raw = _raw(deck, floor=floor, act_id=act_id)
    raw["_sim_raw"] = {"map": {"nodes": nodes}}
    return raw


def _skip_action() -> dict:
    return {
        "kind": "skip_card_reward",
        "selection": "skip",
        "surface": "card_reward",
        "action_id": "reward.skip_card_reward",
    }


def _pick_action(card: dict, idx: int = 0) -> dict:
    return {
        "kind": "card_reward",
        "selection": "pick",
        "surface": "card_reward",
        "action_id": f"reward.pick_card:{idx}",
        "card": card,
    }


def _compact_pick_action(card: dict, idx: int = 0) -> dict:
    return {
        "surface": "card_reward",
        "action_id": f"card_reward:{idx}",
        "choice_index": idx,
        "card_id": card["id"],
        "card_title": card.get("title"),
        "title": card.get("title"),
        "card_type": card.get("type"),
        "card_cost": card.get("cost"),
        "card_effect_profile": card.get("card_effect_profile"),
    }


def _compact_skip_action() -> dict:
    return {
        "surface": "card_reward",
        "action_id": "card_reward:skip",
        "selection": "skip",
    }


def test_thin_low_block_deck_blocks_skip_to_best_block_card():
    from muzero.training.card_reward_guard import apply_card_reward_guard

    block_card = _card("good_block", type_="Skill", block=12)
    attack_card = _card("ok_attack", type_="Attack", damage=6)
    actions = [_skip_action(), _pick_action(attack_card, 0), _pick_action(block_card, 1)]
    stats: dict = {}

    new_idx = apply_card_reward_guard(
        action_idx=0,
        legal_actions=actions,
        full_legal_actions=actions,
        action_mask=np.array([1, 1, 1], dtype=np.float32),
        raw_obs=_raw(_thin_low_block_deck()),
        search_stats=stats,
    )

    assert new_idx == 2
    assert stats["card_reward_guard_context"] == 1.0
    assert stats["card_reward_guard_applicable"] == 1.0
    assert stats["card_reward_guard_override"] == 1.0
    assert stats["card_reward_guard_low_block"] == 1.0
    assert stats["card_reward_guard_block_per_energy"] >= 0.0
    assert stats["card_reward_guard_expected_hand_block"] >= 0.0
    assert stats["card_reward_guard_delayed_payoff_option_value"] >= 0.0
    assert stats["card_reward_guard_delayed_payoff_unrealized_risk"] >= 0.0
    assert stats["card_reward_guard_delayed_payoff_maturity"] >= 0.0
    assert stats["card_reward_guard_delayed_payoff_time_to_value"] >= 0.0


def test_thin_low_attack_deck_blocks_skip_to_attack_card():
    from muzero.training.card_reward_guard import apply_card_reward_guard

    attack_card = _card("big_attack", type_="Attack", damage=12)
    block_card = _card("small_block", type_="Skill", block=5)
    actions = [_skip_action(), _pick_action(block_card, 0), _pick_action(attack_card, 1)]
    stats: dict = {}

    new_idx = apply_card_reward_guard(
        action_idx=0,
        legal_actions=actions,
        full_legal_actions=actions,
        action_mask=np.array([1, 1, 1], dtype=np.float32),
        raw_obs=_raw(_thin_low_attack_deck()),
        search_stats=stats,
    )

    assert new_idx == 2
    assert stats["card_reward_guard_low_attack"] == 1.0
    assert stats["card_reward_guard_damage_per_energy"] >= 0.0


def test_low_draw_deck_blocks_skip_to_draw_card_when_draw_is_only_useful_candidate():
    from muzero.training.card_reward_guard import apply_card_reward_guard

    draw_card = _card("draw_two", type_="Skill", draw=2, tags=["draw"])
    actions = [_skip_action(), _pick_action(draw_card, 0)]
    stats: dict = {}

    new_idx = apply_card_reward_guard(
        action_idx=0,
        legal_actions=actions,
        full_legal_actions=actions,
        action_mask=np.array([1, 1], dtype=np.float32),
        raw_obs=_raw(_thin_low_block_deck()),
        search_stats=stats,
    )

    assert new_idx == 1
    assert stats["card_reward_guard_low_draw"] == 1.0
    assert stats["card_reward_guard_expected_extra_draw"] >= 0.0
    assert stats["card_reward_guard_expected_cards_seen"] >= 0.0


def test_body_slam_without_block_partner_does_not_force_skip():
    from muzero.training.card_reward_guard import apply_card_reward_guard

    body_slam = _card(
        "CARD.BODY_SLAM",
        title="全身撞击",
        type_="Attack",
        damage=0,
        tags=[],
    )
    actions = [_skip_action(), _pick_action(body_slam, 0)]
    stats: dict = {}

    new_idx = apply_card_reward_guard(
        action_idx=0,
        legal_actions=actions,
        full_legal_actions=actions,
        action_mask=np.array([1, 1], dtype=np.float32),
        raw_obs=_raw([_card(f"strike_{idx}", type_="Attack", damage=6) for idx in range(8)], floor=4),
        search_stats=stats,
    )

    assert new_idx == 0
    assert stats["card_reward_guard_pick_available"] == 1.0
    assert stats["card_reward_guard_no_useful_candidate"] == 1.0
    assert stats["card_reward_guard_best_score"] < 0.45
    assert stats["card_reward_guard_future_combo_window"] == 1.0
    assert stats["card_reward_guard_best_combo_orphan_risk"] > 0.50
    assert stats["card_reward_guard_orphan_reject"] == 1.0
    assert stats["card_reward_guard_best_combo_current_fit"] < 0.20
    assert stats["card_reward_guard_best_combo_final_option_value"] <= 0.0


def test_early_thin_deck_dynamic_threshold_forces_moderate_card():
    from muzero.training.card_reward_guard import apply_card_reward_guard

    # This card is below the historical fixed 0.45 threshold but still above
    # the dynamic early-thin-deck threshold.  The guard should prevent the
    # pathological "skip everything until boss" behavior without requiring a
    # premium standalone card in every reward.
    moderate_block = _card("costly_block", type_="Skill", cost=3, block=5)
    deck = [_card(f"strike_{idx}", type_="Attack", damage=6) for idx in range(5)] + [
        _card(f"defend_{idx}", type_="Skill", block=5) for idx in range(5)
    ]
    actions = [_skip_action(), _pick_action(moderate_block, 0)]
    stats: dict = {}

    new_idx = apply_card_reward_guard(
        action_idx=0,
        legal_actions=actions,
        full_legal_actions=actions,
        action_mask=np.array([1, 1], dtype=np.float32),
        raw_obs=_raw(deck, floor=3),
        search_stats=stats,
    )

    assert stats["card_reward_guard_best_score"] < 0.45
    assert stats["card_reward_guard_dynamic_min_useful_score"] < 0.45
    assert stats["card_reward_guard_threshold_relaxed"] == 1.0
    assert stats["card_reward_guard_best_score"] >= stats["card_reward_guard_dynamic_min_useful_score"]
    assert new_idx == 1


def test_body_slam_with_block_partner_is_useful_combo_candidate():
    from muzero.training.card_reward_guard import apply_card_reward_guard

    body_slam = _card(
        "CARD.BODY_SLAM",
        title="全身撞击",
        type_="Attack",
        damage=0,
        tags=[],
    )
    actions = [_skip_action(), _pick_action(body_slam, 0)]
    stats: dict = {}

    new_idx = apply_card_reward_guard(
        action_idx=0,
        legal_actions=actions,
        full_legal_actions=actions,
        action_mask=np.array([1, 1], dtype=np.float32),
        raw_obs=_raw(_thin_low_attack_deck(), floor=4),
        search_stats=stats,
    )

    assert new_idx == 1
    assert stats["card_reward_guard_override"] == 1.0
    assert stats["card_reward_guard_expected_playable_block"] >= 10.0
    assert stats["card_reward_guard_combo_option_value"] >= 0.0
    assert stats["card_reward_guard_delayed_payoff_option_value"] >= 0.0
    assert stats["card_reward_guard_best_combo_current_fit"] > stats["card_reward_guard_best_combo_orphan_risk"]
    assert stats["card_reward_guard_best_combo_final_option_value"] > 0.0


def test_candidate_combo_fit_rewards_missing_enabler_for_existing_payoff():
    from muzero.training.card_reward_guard import apply_card_reward_guard

    body_slam = _card(
        "CARD.BODY_SLAM",
        title="全身撞击",
        type_="Attack",
        damage=0,
        tags=[],
    )
    draw_engine = _card("draw_two", type_="Skill", draw=2, tags=["draw"])
    deck = [body_slam] + [_card(f"strike_{idx}", type_="Attack", damage=6) for idx in range(5)]
    actions = [_skip_action(), _pick_action(draw_engine, 0)]
    stats: dict = {}

    new_idx = apply_card_reward_guard(
        action_idx=0,
        legal_actions=actions,
        full_legal_actions=actions,
        action_mask=np.array([1, 1], dtype=np.float32),
        raw_obs=_raw(deck, floor=4),
        search_stats=stats,
    )

    assert new_idx == 1
    assert stats["card_reward_guard_combo_unmet_dependency"] > 0.0
    assert stats["card_reward_guard_best_combo_missing_piece_fit"] > 0.50
    assert stats["card_reward_guard_best_combo_orphan_risk"] < 0.25
    assert stats["card_reward_guard_best_combo_final_option_value"] > 0.0


def test_future_combo_option_window_decays_near_act1_boss():
    from muzero.training.card_reward_guard import apply_card_reward_guard

    combo_card = _card("CARD.BODY_SLAM", title="全身撞击", type_="Attack", damage=0, tags=[])
    actions = [_skip_action(), _pick_action(combo_card, 0)]
    early_stats: dict = {}
    late_stats: dict = {}

    apply_card_reward_guard(
        action_idx=0,
        legal_actions=actions,
        full_legal_actions=actions,
        action_mask=np.array([1, 1], dtype=np.float32),
        raw_obs=_raw(_thin_low_attack_deck(), floor=4),
        search_stats=early_stats,
    )
    apply_card_reward_guard(
        action_idx=0,
        legal_actions=actions,
        full_legal_actions=actions,
        action_mask=np.array([1, 1], dtype=np.float32),
        raw_obs=_raw(_thin_low_attack_deck(), floor=15),
        search_stats=late_stats,
    )

    assert early_stats["card_reward_guard_future_combo_window"] > late_stats["card_reward_guard_future_combo_window"]
    assert late_stats["card_reward_guard_future_combo_window"] == 0.10


def test_late_act1_orphan_combo_piece_is_rejected_and_threshold_raised():
    from muzero.training.card_reward_guard import apply_card_reward_guard

    body_slam = _card("CARD.BODY_SLAM", title="全身撞击", type_="Attack", damage=0, tags=[])
    deck = [_card(f"strike_{idx}", type_="Attack", damage=6) for idx in range(8)]
    actions = [_skip_action(), _pick_action(body_slam, 0)]
    stats: dict = {}

    new_idx = apply_card_reward_guard(
        action_idx=0,
        legal_actions=actions,
        full_legal_actions=actions,
        action_mask=np.array([1, 1], dtype=np.float32),
        raw_obs=_raw(deck, floor=15),
        search_stats=stats,
    )

    assert new_idx == 0
    assert stats["card_reward_guard_orphan_reject"] == 1.0
    assert stats["card_reward_guard_threshold_raised_near_boss"] == 1.0
    assert stats["card_reward_guard_no_useful_candidate"] == 1.0


def test_future_combo_window_uses_map_reward_runway_when_available():
    from muzero.training.card_reward_guard import apply_card_reward_guard

    combo_card = _card("CARD.BODY_SLAM", title="全身撞击", type_="Attack", damage=0, tags=[])
    actions = [_skip_action(), _pick_action(combo_card, 0)]
    stats: dict = {}
    nodes = [
        {"row": 13, "point_type": "Monster"},
        {"row": 14, "point_type": "Elite"},
        {"row": 15, "point_type": "Event"},
        {"row": 16, "point_type": "Shop"},
        {"row": 17, "point_type": "Boss"},
    ]

    apply_card_reward_guard(
        action_idx=0,
        legal_actions=actions,
        full_legal_actions=actions,
        action_mask=np.array([1, 1], dtype=np.float32),
        raw_obs=_raw_with_map_nodes(_thin_low_attack_deck(), floor=12, nodes=nodes),
        search_stats=stats,
    )

    assert stats["card_reward_guard_future_runway_observed"] == 1.0
    assert stats["card_reward_guard_future_combo_floor_window"] == 0.25
    assert stats["card_reward_guard_future_combo_route_window"] > 0.25
    assert stats["card_reward_guard_future_combo_window"] > 0.25
    assert stats["card_reward_guard_future_reward_opportunity"] > 0.0


def test_future_combo_window_uses_run_route_snapshot_on_reward_surface():
    from muzero.training.card_reward_guard import apply_card_reward_guard

    combo_card = _card("CARD.BODY_SLAM", title="全身撞击", type_="Attack", damage=0, tags=[])
    actions = [_skip_action(), _pick_action(combo_card, 0)]
    stats: dict = {}
    raw = _raw(_thin_low_attack_deck(), floor=12)
    raw["_run_route_snapshot"] = {
        "current_floor": 11,
        "act_id": 0,
        "route_summaries": [
            {
                "count_monster": 2,
                "count_elite": 1,
                "count_event": 1,
                "count_shop": 1,
                "count_treasure": 0,
                "count_question_mark": 0,
                "count_boss": 1,
                "reachable_node_count": 6,
                "max_depth": 5,
                "next_boss_steps": 5,
            }
        ],
        "route_nodes": [
            {"row": 13, "point_type": "Monster"},
            {"row": 14, "point_type": "Elite"},
            {"row": 15, "point_type": "Event"},
            {"row": 16, "point_type": "Shop"},
            {"row": 17, "point_type": "Boss"},
        ],
    }

    apply_card_reward_guard(
        action_idx=0,
        legal_actions=actions,
        full_legal_actions=actions,
        action_mask=np.array([1, 1], dtype=np.float32),
        raw_obs=raw,
        search_stats=stats,
    )

    assert stats["card_reward_guard_future_runway_observed"] == 1.0
    assert stats["card_reward_guard_future_combo_floor_window"] == 0.25
    assert stats["card_reward_guard_future_combo_route_window"] > 0.25
    assert stats["card_reward_guard_future_reward_opportunity"] > 0.0


def test_run_memory_attaches_last_route_snapshot_to_later_reward_obs():
    from sts2_env.run_memory import RunMemoryTracker

    tracker = RunMemoryTracker()
    map_obs = {"run": {"floor": 8, "act_id": 0}, "player": {"hp": 60, "max_hp": 80}}
    legal_actions = [
        {
            "kind": "map",
            "action_id": "map:0",
            "route_summary": {
                "count_monster": 2,
                "count_shop": 1,
                "count_boss": 1,
                "reachable_node_count": 5,
                "max_depth": 5,
                "next_boss_steps": 5,
            },
            "route_nodes": [
                {"row": 9, "point_type": "Monster", "extra_heavy_field": {"drop": "me"}},
                {"row": 10, "point_type": "Shop"},
                {"row": 13, "point_type": "Boss"},
            ],
        }
    ]

    tracker.reset(map_obs, legal_actions)
    reward_obs = {"run": {"floor": 9, "act_id": 0}, "player": {"hp": 55, "max_hp": 80}}
    tracker.attach_route_snapshot_to_obs(reward_obs)

    snapshot = reward_obs.get("_run_route_snapshot")
    assert isinstance(snapshot, dict)
    assert snapshot["route_summaries"]
    assert snapshot["route_nodes"]
    assert snapshot["map"]["nodes"]
    assert "extra_heavy_field" not in snapshot["route_nodes"][0]


def test_future_combo_window_drops_when_route_says_boss_is_close():
    from muzero.training.card_reward_guard import apply_card_reward_guard

    combo_card = _card("CARD.BODY_SLAM", title="全身撞击", type_="Attack", damage=0, tags=[])
    actions = [_skip_action(), _pick_action(combo_card, 0)]
    stats: dict = {}
    nodes = [
        {"row": 5, "point_type": "RestSite"},
        {"row": 6, "point_type": "Boss"},
    ]

    apply_card_reward_guard(
        action_idx=0,
        legal_actions=actions,
        full_legal_actions=actions,
        action_mask=np.array([1, 1], dtype=np.float32),
        raw_obs=_raw_with_map_nodes(_thin_low_attack_deck(), floor=4, nodes=nodes),
        search_stats=stats,
    )

    assert stats["card_reward_guard_future_runway_observed"] == 1.0
    assert stats["card_reward_guard_future_combo_floor_window"] == 1.0
    assert stats["card_reward_guard_future_combo_window"] < 0.50


def test_delayed_payoff_guard_metrics_are_registered_for_tb_reduction():
    from muzero.training.card_reward_guard import (
        CARD_REWARD_GUARD_SEARCH_SUFFIXES,
        card_reward_guard_metric_keys,
    )

    keys = set(card_reward_guard_metric_keys())
    assert "card_reward_guard_delayed_payoff_option_value" in keys
    assert "card_reward_guard_delayed_payoff_unrealized_risk" in keys
    assert "card_reward_guard_delayed_payoff_maturity" in keys
    assert "card_reward_guard_delayed_payoff_time_to_value" in keys
    assert "card_reward_guard_future_combo_floor_window" in keys
    assert "card_reward_guard_future_combo_route_window" in keys
    assert "card_reward_guard_future_reward_opportunity" in keys
    assert "card_reward_guard_future_runway_observed" in keys
    assert "card_reward_guard_best_combo_current_fit" in keys
    assert "card_reward_guard_best_combo_missing_piece_fit" in keys
    assert "card_reward_guard_best_combo_speculative_option" in keys
    assert "card_reward_guard_best_combo_orphan_risk" in keys
    assert "card_reward_guard_best_combo_final_option_value" in keys
    assert "card_reward_guard_dynamic_min_useful_score" in keys
    assert "card_reward_guard_threshold_relaxed" in keys
    assert "card_reward_guard_threshold_raised_near_boss" in keys
    assert "card_reward_guard_orphan_reject" in keys
    assert "card_reward_guard_speculative_option_accepted" in keys
    assert (
        CARD_REWARD_GUARD_SEARCH_SUFFIXES["card_reward_guard_delayed_payoff_option_value"]
        == "card_reward_guard_delayed_payoff_option_value_mean"
    )
    assert (
        CARD_REWARD_GUARD_SEARCH_SUFFIXES["card_reward_guard_delayed_payoff_unrealized_risk"]
        == "card_reward_guard_delayed_payoff_unrealized_risk_mean"
    )
    assert (
        CARD_REWARD_GUARD_SEARCH_SUFFIXES["card_reward_guard_future_runway_observed"]
        == "card_reward_guard_future_runway_observed_rate"
    )
    assert (
        CARD_REWARD_GUARD_SEARCH_SUFFIXES["card_reward_guard_best_combo_missing_piece_fit"]
        == "card_reward_guard_best_combo_missing_piece_fit_mean"
    )
    assert (
        CARD_REWARD_GUARD_SEARCH_SUFFIXES["card_reward_guard_dynamic_min_useful_score"]
        == "card_reward_guard_dynamic_min_useful_score_mean"
    )
    assert (
        CARD_REWARD_GUARD_SEARCH_SUFFIXES["card_reward_guard_threshold_relaxed"]
        == "card_reward_guard_threshold_relaxed_rate"
    )
    assert (
        CARD_REWARD_GUARD_SEARCH_SUFFIXES["card_reward_guard_threshold_raised_near_boss"]
        == "card_reward_guard_threshold_raised_near_boss_rate"
    )
    assert (
        CARD_REWARD_GUARD_SEARCH_SUFFIXES["card_reward_guard_orphan_reject"]
        == "card_reward_guard_orphan_reject_rate"
    )
    assert (
        CARD_REWARD_GUARD_SEARCH_SUFFIXES["card_reward_guard_speculative_option_accepted"]
        == "card_reward_guard_speculative_option_accepted_rate"
    )


def test_card_reward_guard_emits_choice_diagnostic_payload():
    from muzero.training.card_reward_guard import apply_card_reward_guard

    block_card = _card("good_block", type_="Skill", block=12)
    attack_card = _card("ok_attack", type_="Attack", damage=6)
    actions = [_skip_action(), _pick_action(attack_card, 0), _pick_action(block_card, 1)]
    stats: dict = {}

    new_idx = apply_card_reward_guard(
        action_idx=0,
        legal_actions=actions,
        full_legal_actions=actions,
        action_mask=np.array([1, 1, 1], dtype=np.float32),
        raw_obs=_raw(_thin_low_block_deck()),
        search_stats=stats,
    )

    payload = stats.get("_card_reward_choice_diagnostic")
    assert new_idx == 2
    assert isinstance(payload, dict)
    assert payload["skip_blocked"] is True
    assert payload["original_selected_is_skip"] is True
    assert payload["final_selected_is_skip"] is None
    # The raw guard payload is created before self_play knows the final
    # post-guard action; self_play rewrites selected_is_skip to final semantics
    # before JSONL dump.
    assert payload["selected_is_skip"] is True
    assert payload["decision_reason"] == "skip_blocked"
    assert payload["best_score"] == stats["card_reward_guard_best_score"]
    assert payload["dynamic_min_useful_score"] == stats["card_reward_guard_dynamic_min_useful_score"]
    assert payload["deck_quality_before"]["damage_per_energy"] >= 0.0
    assert payload["deck_quality_before"]["block_per_energy"] >= 0.0
    assert payload["deficits"]["low_block"] is True
    assert len(payload["offered_cards"]) == 2
    assert payload["best_card"]["id"] == "good_block"


def test_junk_only_candidates_do_not_override_skip():
    from muzero.training.card_reward_guard import apply_card_reward_guard

    curse = _card("bad_curse", type_="Curse")
    actions = [_skip_action(), _pick_action(curse, 0)]
    stats: dict = {}

    new_idx = apply_card_reward_guard(
        action_idx=0,
        legal_actions=actions,
        full_legal_actions=actions,
        action_mask=np.array([1, 1], dtype=np.float32),
        raw_obs=_raw(_thin_low_block_deck()),
        search_stats=stats,
    )

    assert new_idx == 0
    assert stats["card_reward_guard_pick_available"] == 1.0
    assert stats["card_reward_guard_no_useful_candidate"] == 1.0
    assert stats["card_reward_guard_override"] == 0.0


def test_healthy_large_deck_does_not_force_more_cards():
    from muzero.training.card_reward_guard import apply_card_reward_guard

    actions = [_skip_action(), _pick_action(_card("extra_attack", type_="Attack", damage=8), 0)]
    stats: dict = {}

    new_idx = apply_card_reward_guard(
        action_idx=0,
        legal_actions=actions,
        full_legal_actions=actions,
        action_mask=np.array([1, 1], dtype=np.float32),
        raw_obs=_raw(_healthy_deck()),
        search_stats=stats,
    )

    assert new_idx == 0
    assert stats["card_reward_guard_context"] == 1.0
    assert stats["card_reward_guard_applicable"] == 0.0
    assert stats["card_reward_guard_expected_hand_useful_quality"] > 0.0


def test_non_card_reward_context_and_existing_pick_are_noops():
    from muzero.training.card_reward_guard import apply_card_reward_guard

    non_reward = [{"kind": "map", "action_id": "map:0"}, {"kind": "map", "action_id": "map:1"}]
    stats: dict = {}
    assert (
        apply_card_reward_guard(
            action_idx=0,
            legal_actions=non_reward,
            full_legal_actions=non_reward,
            action_mask=np.array([1, 1], dtype=np.float32),
            raw_obs=_raw(_thin_low_block_deck()),
            search_stats=stats,
        )
        == 0
    )
    assert stats["card_reward_guard_context"] == 0.0

    pick_actions = [_skip_action(), _pick_action(_card("good_block", type_="Skill", block=12), 0)]
    stats = {}
    assert (
        apply_card_reward_guard(
            action_idx=1,
            legal_actions=pick_actions,
            full_legal_actions=pick_actions,
            action_mask=np.array([1, 1], dtype=np.float32),
            raw_obs=_raw(_thin_low_block_deck()),
            search_stats=stats,
        )
        == 1
    )
    assert stats["card_reward_guard_selected_pick"] == 1.0


def test_compact_only_card_reward_skip_overrides_to_positional_pick():
    from muzero.training.card_reward_guard import apply_card_reward_guard

    block_card = _card("good_block", type_="Skill", block=12)
    attack_card = _card("ok_attack", type_="Attack", damage=6)
    actions = [
        _compact_skip_action(),
        _compact_pick_action(attack_card, 0),
        _compact_pick_action(block_card, 1),
    ]
    stats: dict = {}

    new_idx = apply_card_reward_guard(
        action_idx=0,
        legal_actions=actions,
        full_legal_actions=None,
        action_mask=np.array([1, 1, 1], dtype=np.float32),
        raw_obs=_raw(_thin_low_block_deck()),
        search_stats=stats,
    )

    assert new_idx == 2
    assert stats["card_reward_guard_context"] == 1.0
    assert stats["card_reward_guard_pick_available"] == 1.0
    assert stats["card_reward_guard_skip_blocked"] == 1.0
    assert stats["card_reward_guard_override"] == 1.0
    assert stats["_card_reward_choice_diagnostic"]["best_card"]["id"] == "good_block"


def test_compact_positional_card_reward_pick_is_noop_not_skip():
    from muzero.training.card_reward_guard import apply_card_reward_guard

    block_card = _card("good_block", type_="Skill", block=12)
    actions = [_compact_skip_action(), _compact_pick_action(block_card, 0)]
    stats: dict = {}

    new_idx = apply_card_reward_guard(
        action_idx=1,
        legal_actions=actions,
        full_legal_actions=None,
        action_mask=np.array([1, 1], dtype=np.float32),
        raw_obs=_raw(_thin_low_block_deck()),
        search_stats=stats,
    )

    assert new_idx == 1
    assert stats["card_reward_guard_context"] == 1.0
    assert stats["card_reward_guard_selected_pick"] == 1.0
    assert stats["card_reward_guard_selected_skip"] == 0.0
    assert stats["card_reward_guard_override"] == 0.0


def test_alignment_error_fails_open():
    from muzero.training.card_reward_guard import apply_card_reward_guard

    actions = [_skip_action(), _pick_action(_card("good_block", type_="Skill", block=12), 0)]
    stats: dict = {}

    new_idx = apply_card_reward_guard(
        action_idx=1,
        legal_actions=actions,
        full_legal_actions=actions[:1],
        action_mask=np.array([1, 1], dtype=np.float32),
        raw_obs=_raw(_thin_low_block_deck()),
        search_stats=stats,
    )

    assert new_idx == 1
    assert stats["card_reward_guard_alignment_error"] == 1.0


def test_build_guard_integration_runs_card_reward_before_hp_rest_guard():
    from muzero.train import MuZeroTrainer

    actions = [_skip_action(), _pick_action(_card("good_block", type_="Skill", block=12), 0)]
    trainer = MuZeroTrainer.__new__(MuZeroTrainer)
    trainer.env = SimpleNamespace(
        unwrapped=SimpleNamespace(
            _last_obs_raw=_raw(_thin_low_block_deck()),
            _legal_actions=actions,
        )
    )
    stats: dict = {}

    new_idx = trainer._apply_build_action_hard_guards(
        action_idx=0,
        legal_actions=actions,
        action_mask=np.array([1, 1], dtype=np.float32),
        search_stats=stats,
    )

    assert new_idx == 1
    assert stats["card_reward_guard_applied"] == 1.0
    assert stats["card_reward_guard_override"] == 1.0
