from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
from typing import Any


RL_AGENT_ROOT = Path(__file__).resolve().parents[1]
MONITOR_SRC = RL_AGENT_ROOT / "scripts" / "monitor_fullrun_act1_gate.py"
_spec = importlib.util.spec_from_file_location("monitor_fullrun_act1_gate_under_test", str(MONITOR_SRC))
assert _spec is not None
monitor = importlib.util.module_from_spec(_spec)
sys.modules["monitor_fullrun_act1_gate_under_test"] = monitor
assert _spec.loader is not None
_spec.loader.exec_module(monitor)


def _metric(
    last: float,
    *,
    avg_tail: float | None = None,
    min_tail: float | None = None,
    max_tail: float | None = None,
    n: int = 80,
    step: int = 100,
) -> dict[str, Any]:
    value = float(last)
    return {
        "n": int(n),
        "step": int(step),
        "last": value,
        "avg_tail": float(value if avg_tail is None else avg_tail),
        "min_tail": float(value if min_tail is None else min_tail),
        "max_tail": float(value if max_tail is None else max_tail),
    }


def _passing_summary(*, n: int = 80, act1_pass64: float = 0.0) -> dict[str, dict[str, Any]]:
    return {
        "buffer/size": _metric(5000, n=n),
        "episode/reward": _metric(4.0, n=n),
        "episode/max_floor": _metric(11.0, n=n),
        "episode/length": _metric(80.0, n=n),
        "episode/act1_boss_seen": _metric(1.0, n=n),
        "recent_tail/64/act1_boss_seen_rate": _metric(0.35, n=n),
        "recent_tail/64/act1_pass_rate": _metric(act1_pass64, n=n),
        "recent_tail/256/act1_pass_rate": _metric(act1_pass64, n=n),
        "recent_tail/64/max_floor_mean": _metric(10.5, n=n),
        "recent_tail/64/reached_floor_10_rate": _metric(0.50, n=n),
        "recent_tail/64/reached_floor_12_rate": _metric(0.25, n=n),
        "loss/total": _metric(12.0, max_tail=12.0, n=n),
        "loss/future_world_aux": _metric(0.2, max_tail=0.2, n=n),
        "loss/future_bank_state": _metric(0.1, max_tail=0.1, n=n),
        "loss/future_bank_delta": _metric(0.1, max_tail=0.1, n=n),
        "loss/future_bank_token_slot_source": _metric(0.0, max_tail=0.0, n=n),
        "memory/reserved_gb": _metric(8.0, max_tail=8.0, n=n),
        "memory/max_allocated_gb": _metric(7.0, max_tail=7.0, n=n),
        "memory/empty_cache_called": _metric(0.0, avg_tail=0.0, n=n),
        "combat_quality/hp_cost_self_lethal_selected_rate": _metric(0.0, n=n),
        "combat_quality/zero_energy_x_cost_selected_rate": _metric(0.0, n=n),
        "combat_quality/hp_cost_low_margin_selected_rate": _metric(0.0, n=n),
        "combat_quality/wasteful_end_turn_rate": _metric(0.0, n=n),
        "combat_quality/refund_no_followup_with_progress_selected_rate": _metric(0.0, n=n),
        "combat_quality/potion_low_urgency_selected_rate": _metric(0.0, n=n),
    }


def _evaluate(
    summary: dict[str, dict[str, Any]],
    *,
    mode: str = "early",
    min_episodes: int = 64,
    min_buffer: float = 3000.0,
) -> dict[str, Any]:
    return monitor.evaluate_gate(
        summary,
        mode=mode,
        min_episodes=min_episodes,
        min_buffer=min_buffer,
        max_reserved_gb=22.0,
        max_peak_allocated_gb=20.0,
        max_empty_cache_rate=0.2,
        max_wasteful_end_turn_rate=0.002,
        max_refund_no_followup_with_progress_rate=0.005,
        max_potion_low_urgency_rate=0.01,
    )


def _find_check(gate: dict[str, Any], tag: str) -> dict[str, Any]:
    matches = [check for check in gate["checks"] if check["tag"] == tag]
    assert matches, f"missing check for {tag}"
    return matches[0]


def test_early_gate_passes_with_boss_seen_and_floor_progress() -> None:
    gate = _evaluate(_passing_summary())

    assert gate["verdict"] == "PASS"
    assert _find_check(gate, "recent_tail/64/act1_boss_seen_rate")["status"] == "pass"


def test_wait_episodes_overrides_red_progress_metrics() -> None:
    summary = _passing_summary(n=12)
    summary["recent_tail/64/act1_boss_seen_rate"] = _metric(0.0, n=12)

    gate = _evaluate(summary, min_episodes=64)

    assert gate["verdict"] == "WAIT_EPISODES"
    assert _find_check(gate, "episode_count")["status"] == "fail"
    assert _find_check(gate, "recent_tail/64/act1_boss_seen_rate")["status"] == "fail"


def test_wait_buffer_after_enough_episodes() -> None:
    summary = _passing_summary(n=80)
    summary["buffer/size"] = _metric(500.0, n=80)

    gate = _evaluate(summary, min_buffer=3000.0)

    assert gate["verdict"] == "WAIT_BUFFER"
    assert _find_check(gate, "buffer/size")["status"] == "fail"


def test_strong_gate_requires_nonzero_act1_pass() -> None:
    summary = _passing_summary(n=300, act1_pass64=0.0)
    summary["recent_tail/64/act1_boss_seen_rate"] = _metric(0.70, n=300)
    summary["recent_tail/64/max_floor_mean"] = _metric(13.0, n=300)
    summary["recent_tail/64/reached_floor_10_rate"] = _metric(0.80, n=300)
    summary["recent_tail/64/reached_floor_12_rate"] = _metric(0.50, n=300)

    gate = _evaluate(summary, mode="strong", min_episodes=256)

    assert gate["verdict"] == "FAIL"
    assert _find_check(gate, "recent_tail/64/act1_pass_rate")["status"] == "fail"


def test_memory_redline_fails_after_data_ready() -> None:
    summary = _passing_summary(n=80)
    summary["memory/reserved_gb"] = _metric(24.0, max_tail=24.0, n=80)

    gate = _evaluate(summary)

    assert gate["verdict"] == "FAIL"
    assert _find_check(gate, "memory/reserved_gb")["status"] == "fail"


def test_lucky_skip_on_boss_death_is_redline_when_present() -> None:
    summary = _passing_summary(n=80)
    summary["boss/lucky_skip_on_boss_death_rate"] = _metric(1.0, avg_tail=1.0, n=80)

    gate = _evaluate(summary)

    assert gate["verdict"] == "FAIL"
    assert _find_check(gate, "boss/lucky_skip_on_boss_death_rate")["status"] == "fail"


def test_final_lucky_unused_on_boss_death_is_redline_when_present() -> None:
    summary = _passing_summary(n=80)
    summary["boss/final_lucky_unused_on_boss_death_rate"] = _metric(1.0, avg_tail=1.0, n=80)

    gate = _evaluate(summary)

    assert gate["verdict"] == "FAIL"
    assert _find_check(gate, "boss/final_lucky_unused_on_boss_death_rate")["status"] == "fail"


def test_insufficient_block_selected_is_redline_when_present() -> None:
    summary = _passing_summary(n=80)
    summary["combat_quality/insufficient_block_selected_rate"] = _metric(0.02, avg_tail=0.02, n=80)

    gate = _evaluate(summary)

    assert gate["verdict"] == "FAIL"
    assert _find_check(gate, "combat_quality/insufficient_block_selected_rate")["status"] == "fail"


def test_deck_build_tags_are_exposed() -> None:
    assert "deck/final_size" in monitor.DECK_BUILD_TAGS
    assert "deck/final_starter_count" in monitor.DECK_BUILD_TAGS
    assert "deck/final_starter_ratio" in monitor.DECK_BUILD_TAGS
    assert "deck/final_nonstarter_count" in monitor.DECK_BUILD_TAGS
    assert "deck/final_upgraded_count" in monitor.DECK_BUILD_TAGS
    assert "deck/final_raw_avg_damage_per_energy" in monitor.DECK_BUILD_TAGS
    assert "deck/final_raw_expected_cards_seen_per_turn" in monitor.DECK_BUILD_TAGS
    assert "deck/final_expected_hand_useful_quality_score" in monitor.DECK_BUILD_TAGS
    assert "deck/final_expected_hand_quality_draw_share" in monitor.DECK_BUILD_TAGS
    assert "deck/final_raw_expected_playable_cards_per_turn" in monitor.DECK_BUILD_TAGS
    assert "deck/final_combo_option_value_score" in monitor.DECK_BUILD_TAGS
    assert "deck/final_delayed_payoff_option_value_score" in monitor.DECK_BUILD_TAGS
    assert "deck/final_delayed_payoff_unrealized_risk_score" in monitor.DECK_BUILD_TAGS
    assert "death_deck/starter_count" in monitor.DECK_BUILD_TAGS
    assert "death_deck/starter_ratio" in monitor.DECK_BUILD_TAGS
    assert "death_deck/nonstarter_count" in monitor.DECK_BUILD_TAGS
    assert "death_deck/upgraded_count" in monitor.DECK_BUILD_TAGS
    assert "death_deck/raw_avg_block_per_energy" in monitor.DECK_BUILD_TAGS
    assert "death_deck/expected_hand_junk_share" in monitor.DECK_BUILD_TAGS
    assert "death_deck/raw_expected_playable_cards_per_turn" in monitor.DECK_BUILD_TAGS
    assert "death_deck/combo_unmet_dependency_score" in monitor.DECK_BUILD_TAGS
    assert "death_deck/delayed_payoff_maturity_score" in monitor.DECK_BUILD_TAGS
    assert "death_deck/delayed_payoff_unrealized_risk_score" in monitor.DECK_BUILD_TAGS
    assert "build/card_reward_consecutive_skip_max" in monitor.DECK_BUILD_TAGS
    assert "search/build/card_reward_guard_combo_option_value_mean" in monitor.DECK_BUILD_TAGS
    assert "search/build/card_reward_guard_future_combo_window_mean" in monitor.DECK_BUILD_TAGS
    assert "search/build/card_reward_guard_delayed_payoff_option_value_mean" in monitor.DECK_BUILD_TAGS
    assert "search/build/card_reward_guard_delayed_payoff_unrealized_risk_mean" in monitor.DECK_BUILD_TAGS
    assert "search/build/post_search_hard_guard_policy_retargeted_rate" in monitor.DECK_BUILD_TAGS
    assert "search/build/card_reward_pick_quality_guard_applied_rate" in monitor.DECK_BUILD_TAGS
    assert "search/build/card_reward_pick_quality_guard_override_rate" in monitor.DECK_BUILD_TAGS
    assert (
        "search/build/card_reward_pick_quality_guard_best_minus_selected_mean"
        in monitor.DECK_BUILD_TAGS
    )
    assert "search/build/card_reward_pick_quality_guard_low_block_rate" in monitor.DECK_BUILD_TAGS
    assert "search/build/card_reward_pick_quality_guard_low_draw_rate" in monitor.DECK_BUILD_TAGS


def test_campfire_rest_tags_are_exposed() -> None:
    assert "decision/build/family_rest_rate" in monitor.CAMPFIRE_REST_TAGS
    assert "route_heuristic/rest_before_elite_available_rate" in monitor.CAMPFIRE_REST_TAGS
    assert "route_heuristic/no_rest_before_elite_selected_rate" in monitor.CAMPFIRE_REST_TAGS
    assert "env/rest_site_encounters" in monitor.CAMPFIRE_REST_TAGS
    assert "env/rest_heal_chosen" in monitor.CAMPFIRE_REST_TAGS
    assert "env/rest_smith_chosen" in monitor.CAMPFIRE_REST_TAGS
    assert "env/rest_skip_heal_chosen" in monitor.CAMPFIRE_REST_TAGS
    assert "env/rest_skip_heal_at_low_hp" in monitor.CAMPFIRE_REST_TAGS
    assert "env/rest_heal_exposure_forced" in monitor.CAMPFIRE_REST_TAGS
    assert "env/rest_heal_exposure_miss" in monitor.CAMPFIRE_REST_TAGS
    assert "search/build/rest_site_smith_guard_applied_rate" in monitor.CAMPFIRE_REST_TAGS
    assert "search/build/rest_site_smith_guard_override_rate" in monitor.CAMPFIRE_REST_TAGS
    assert "search/build/rest_site_smith_guard_hp_ratio_mean" in monitor.CAMPFIRE_REST_TAGS
    assert "build/deck_upgrade_smith_selected" in monitor.CAMPFIRE_REST_TAGS
    assert "build/deck_upgrade_smith_to_upgrade_seen_rate" in monitor.CAMPFIRE_REST_TAGS
    assert "build/deck_upgrade_smith_no_upgrade_surface_rate" in monitor.CAMPFIRE_REST_TAGS
    assert "build/deck_upgrade_seen" in monitor.CAMPFIRE_REST_TAGS
    assert "build/deck_upgrade_selected_rate" in monitor.CAMPFIRE_REST_TAGS
    assert "build/deck_upgrade_applied_rate" in monitor.CAMPFIRE_REST_TAGS
    assert "build/deck_upgrade_target_count_mean" in monitor.CAMPFIRE_REST_TAGS
    assert "env/rest_smith_chosen" in monitor.ALL_TAGS
    assert "build/deck_upgrade_applied_rate" in monitor.ALL_TAGS
    assert "search/build/rest_site_smith_guard_applied_rate" in monitor.ALL_TAGS


def test_deck_build_warnings_are_advisory_not_gate_failures() -> None:
    summary = _passing_summary(n=80)
    summary["build/card_reward_seen"] = _metric(5.0, avg_tail=5.0, n=80)
    summary["build/card_reward_skip_rate"] = _metric(0.8, avg_tail=0.8, n=80)
    summary["build/card_reward_consecutive_skip_max"] = _metric(4.0, max_tail=4.0, n=80)

    warnings = monitor.deck_build_warnings(summary)
    gate = _evaluate(summary)

    assert warnings
    assert gate["verdict"] == "PASS"


def test_card_reward_policy_retarget_missing_is_advisory_warning() -> None:
    summary = _passing_summary(n=80)
    summary["search/build/card_reward_guard_skip_blocked_rate"] = _metric(0.08, avg_tail=0.08, n=80)

    warnings = monitor.deck_build_warnings(summary)
    kinds = {warning["kind"] for warning in warnings}
    gate = _evaluate(summary)

    assert "card_reward_policy_retarget_missing" in kinds
    assert gate["verdict"] == "PASS"


def test_card_reward_policy_retarget_low_is_advisory_warning() -> None:
    summary = _passing_summary(n=80)
    summary["search/build/card_reward_guard_skip_blocked_rate"] = _metric(0.10, avg_tail=0.10, n=80)
    summary["search/build/post_search_hard_guard_policy_retargeted_rate"] = _metric(0.02, avg_tail=0.02, n=80)

    warnings = monitor.deck_build_warnings(summary)
    kinds = {warning["kind"] for warning in warnings}
    gate = _evaluate(summary)

    assert "card_reward_policy_retarget_low" in kinds
    assert gate["verdict"] == "PASS"


def test_shop_metrics_missing_warning_when_coarse_shop_signals_exist() -> None:
    summary = _passing_summary(n=80)
    summary["decision/build/family_shop_rate"] = _metric(0.08, avg_tail=0.08, n=80)
    summary["route_heuristic/shop_with_gold_available_rate"] = _metric(0.7, avg_tail=0.7, n=80)

    warnings = monitor.deck_build_warnings(summary)
    kinds = {warning["kind"] for warning in warnings}
    gate = _evaluate(summary)

    assert "shop_metrics_missing" in kinds
    assert gate["verdict"] == "PASS"


def test_starter_heavy_deck_warnings_are_advisory() -> None:
    summary = _passing_summary(n=80)
    summary["episode/max_floor"] = _metric(9.0, avg_tail=9.0, n=80)
    summary["deck/final_starter_count"] = _metric(8.0, avg_tail=8.0, n=80)
    summary["deck/final_starter_ratio"] = _metric(0.55, avg_tail=0.55, n=80)
    summary["deck/final_nonstarter_count"] = _metric(5.0, avg_tail=5.0, n=80)
    summary["episode/death_floor"] = _metric(9.0, avg_tail=9.0, n=80)
    summary["death_deck/starter_count"] = _metric(8.0, avg_tail=8.0, n=80)
    summary["death_deck/starter_ratio"] = _metric(0.55, avg_tail=0.55, n=80)
    summary["death_deck/nonstarter_count"] = _metric(5.0, avg_tail=5.0, n=80)

    warnings = monitor.deck_build_warnings(summary)
    kinds = {warning["kind"] for warning in warnings}
    gate = _evaluate(summary)

    assert "starter_heavy_final_deck" in kinds
    assert "low_final_nonstarter_count" in kinds
    assert "starter_heavy_death_deck" in kinds
    assert "low_death_nonstarter_count" in kinds
    assert gate["verdict"] == "PASS"


def test_death_deck_playability_and_orphan_combo_warnings_are_advisory() -> None:
    summary = _passing_summary(n=80)
    summary["episode/death_floor"] = _metric(9.0, avg_tail=9.0, n=80)
    summary["death_deck/size"] = _metric(17.0, avg_tail=17.0, n=80)
    summary["death_deck/raw_expected_playable_cards_per_turn"] = _metric(1.4, avg_tail=1.4, n=80)
    summary["death_deck/raw_expected_playable_attack_damage_per_turn"] = _metric(9.0, avg_tail=9.0, n=80)
    summary["death_deck/raw_expected_playable_block_per_turn"] = _metric(6.0, avg_tail=6.0, n=80)
    summary["death_deck/expected_energy_utilization_score"] = _metric(0.35, avg_tail=0.35, n=80)
    summary["death_deck/cost_curve_three_plus_share"] = _metric(0.42, avg_tail=0.42, n=80)
    summary["death_deck/raw_expected_extra_draw_per_turn"] = _metric(0.2, avg_tail=0.2, n=80)
    summary["death_deck/combo_component_density"] = _metric(0.25, avg_tail=0.25, n=80)
    summary["death_deck/combo_unmet_dependency_score"] = _metric(0.70, avg_tail=0.70, n=80)
    summary["death_deck/delayed_payoff_density"] = _metric(0.25, avg_tail=0.25, n=80)
    summary["death_deck/delayed_payoff_unrealized_risk_score"] = _metric(0.70, avg_tail=0.70, n=80)
    summary["death_deck/delayed_payoff_maturity_score"] = _metric(0.20, avg_tail=0.20, n=80)

    warnings = monitor.deck_build_warnings(summary)
    kinds = {warning["kind"] for warning in warnings}
    gate = _evaluate(summary)

    assert "low_death_playable_cards" in kinds
    assert "low_death_playable_attack" in kinds
    assert "low_death_playable_block" in kinds
    assert "low_death_energy_utilization" in kinds
    assert "death_cost_curve_bloat" in kinds
    assert "death_orphan_combo_components" in kinds
    assert "death_unrealized_delayed_payoff" in kinds
    assert gate["verdict"] == "PASS"
