#!/usr/bin/env python3
"""Monitor full-run random-seed MuZero Act1 progress.

This script is the full-run counterpart to ``monitor_combat_sandbox_gate.py``.
The combat sandbox gate answers whether weak/normal hallway execution is clean
enough to leave the sandbox.  This gate answers whether a *real full run* is
actually progressing through Act1 under random map/reward/shop/build
distributions.

It is intentionally read-only and operator-facing:

* no training orchestration lives here;
* no checkpoint mutation happens here;
* verdicts are conservative and data-gated so early noisy windows do not cause
  unnecessary restarts.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path
from statistics import mean
from typing import Any


SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

try:
    import monitor_combat_sandbox_gate as sandbox_monitor
except Exception as exc:  # pragma: no cover - operator-facing failure
    raise SystemExit(
        "Could not import scripts/monitor_combat_sandbox_gate.py helpers. "
        "Run from packages/rl-agent or keep both monitor scripts together. "
        f"Original error: {exc!r}"
    )

try:
    from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
except Exception as exc:  # pragma: no cover - operator-facing failure
    raise SystemExit(
        "TensorBoard EventAccumulator import failed. Run this from the "
        "packages/rl-agent venv. Original error: " + repr(exc)
    )


DEFAULT_ROOT = Path(__file__).resolve().parents[1] / "logs_muzero"


PROGRESS_TAGS: tuple[str, ...] = (
    "buffer/size",
    "episode/reward",
    "episode/reward_augmented",
    "episode/length",
    "episode/max_floor",
    "episode/max_act_id",
    "episode/rooms_seen",
    "episode/death_floor",
    "episode/elite_rooms_seen",
    "episode/act1_boss_seen",
    "episode/act1_clear",
    "recent_tail/64/act1_boss_seen_rate",
    "recent_tail/64/act1_pass_rate",
    "recent_tail/64/max_floor_mean",
    "recent_tail/64/max_floor_median",
    "recent_tail/64/max_floor_p25",
    "recent_tail/64/max_floor_p75",
    "recent_tail/64/death_floor_mean",
    "recent_tail/64/reached_floor_07_rate",
    "recent_tail/64/reached_floor_10_rate",
    "recent_tail/64/reached_floor_12_rate",
    "recent_tail/64/reached_floor_15_rate",
    "recent_tail/64/reached_floor_17_rate",
    "recent_tail/256/act1_boss_seen_rate",
    "recent_tail/256/act1_pass_rate",
    "recent_tail/256/max_floor_mean",
    "recent_tail/256/max_floor_median",
    "recent_tail/256/death_floor_mean",
    "recent_tail/256/reached_floor_10_rate",
    "recent_tail/256/reached_floor_12_rate",
    "recent_tail/256/reached_floor_15_rate",
    "recent_tail/256/reached_floor_17_rate",
)


DECISION_TAGS: tuple[str, ...] = (
    "decision/combat_count",
    "decision/combat_share",
    "decision/build_count",
    "decision/build_share",
    "decision/route_count",
    "decision/route_share",
    "decision/direct_policy_eligible_count",
    "decision/direct_policy_used_count",
    "decision/direct_policy_used_rate",
    "decision/fast_path_count",
    "decision/fast_path_rate",
    "search/route/root_top1_visit_share",
    "search/route/root_visit_entropy",
    "search/build/root_top1_visit_share",
    "search/build/root_visit_entropy",
    "search/combat/direct_rollout_steps_used",
    "search/combat/direct_rollout_branch_count_mean",
    "search/combat/direct_rollout_root_bucket_size",
    "search/combat/direct_rollout_bucket_padding_ratio",
    "search/combat/direct_rollout_max_branch_bucket_size",
    "search/combat/direct_rollout_branch_padding_ratio",
)


LOSS_TAGS: tuple[str, ...] = (
    "loss/total",
    "loss/policy",
    "loss/value",
    "loss/reward",
    "loss/future_world_aux",
    "loss/future_bank_state",
    "loss/future_bank_delta",
    "loss/future_bank_token_slot_source",
    "loss/jepa_next_hidden",
    "loss/surprise",
)


MEMORY_TAGS: tuple[str, ...] = (
    "memory/allocated_gb",
    "memory/max_allocated_gb",
    "memory/reserved_gb",
    "memory/reserved_minus_allocated_gb",
    "memory/empty_cache_called",
)


TACTICAL_REDLINE_TAGS: tuple[str, ...] = (
    "combat_quality/bad_pure_block_selected_rate",
    "combat_quality/insufficient_block_selected_rate",
    "combat_quality/card_block_waste_with_progress_selected_rate",
    "combat_quality/card_no_damage_pressure_with_progress_selected_rate",
    "combat_quality/wasteful_end_turn_rate",
    "combat_quality/zero_energy_x_cost_selected_rate",
    "combat_quality/hp_cost_low_margin_selected_rate",
    "combat_quality/hp_cost_self_lethal_selected_rate",
    "combat_quality/refund_no_followup_with_progress_selected_rate",
    "combat_quality/potion_low_urgency_selected_rate",
)

POTION_SURVIVAL_TAGS: tuple[str, ...] = (
    "boss/lucky_seen_rate",
    "boss/lucky_legal_rate",
    "boss/lucky_seen_anywhere_on_boss_death_rate",
    "boss/final_lucky_potion_count_mean",
    "boss/final_lucky_unused_on_boss_death_rate",
    "boss/lucky_skip_on_boss_death_rate",
    "boss/lucky_used_on_boss_win_rate",
    "boss_combat/potion_unused_on_death_rate",
    "boss_combat/potion_unused_on_death_raw_rate",
    "boss_combat/boss_survival_potion_guard_applied_rate",
    "boss_combat/boss_survival_potion_guard_override_rate",
    "boss_combat/boss_survival_potion_guard_no_alternative_rate",
)


SHOP_ECONOMY_TAGS: tuple[str, ...] = (
    "decision/build/family_shop_rate",
    "route_heuristic/shop_with_gold_available_rate",
    "route_heuristic/shop_selected_high_gold_rate",
    "build/shop_seen",
    "build/shop_open_rate",
    "build/shop_buy_any_rate",
    "build/shop_buy_card_rate",
    "build/shop_buy_relic_rate",
    "build/shop_buy_potion_rate",
    "build/shop_remove_rate",
    "build/shop_leave_rate",
    "build/shop_back_rate",
    "build/shop_remove_available_rate",
    "build/shop_remove_affordable_rate",
    "build/shop_leave_with_gold_ge_100_rate",
    "build/shop_leave_with_remove_affordable_rate",
    "build/shop_deck_context_present_rate",
    "build/shop_deck_size_mean",
    "build/shop_starter_count_mean",
    "build/shop_starter_ratio_mean",
    "build/shop_nonstarter_count_mean",
    "build/shop_starter_heavy_rate",
    "build/shop_buy_with_affordable_remove_rate",
    "build/shop_buy_blocks_affordable_remove_rate",
    "build/shop_buy_blocks_affordable_remove_starter_heavy_rate",
    "build/shop_gold_before_mean",
    "build/shop_selected_cost_mean",
    "build/shop_affordable_item_count_mean",
    "search/build/shop_action_guard_shop_surface_rate",
    "search/build/shop_action_guard_selected_open_rate",
    "search/build/shop_action_guard_selected_buy_rate",
    "search/build/shop_action_guard_selected_remove_rate",
    "search/build/shop_action_guard_selected_leave_rate",
    "search/build/shop_action_guard_selected_back_rate",
    "search/build/shop_action_guard_open_available_rate",
    "search/build/shop_action_guard_remove_affordable_available_rate",
    "search/build/shop_action_guard_gold_mean",
    "search/build/shop_action_guard_starter_count_mean",
    "search/build/shop_action_guard_junk_count_mean",
    "search/build/shop_action_guard_open_applicable_rate",
    "search/build/shop_action_guard_open_applied_rate",
    "search/build/shop_action_guard_remove_applicable_rate",
    "search/build/shop_action_guard_remove_applied_rate",
    "search/build/shop_action_guard_buy_blocks_remove_applicable_rate",
    "search/build/shop_action_guard_buy_blocks_remove_applied_rate",
    "search/build/shop_action_guard_buy_blocks_remove_premium_allow_rate",
    "search/build/shop_action_guard_selected_buy_cost_mean",
    "search/build/shop_action_guard_remove_cost_min_mean",
    "search/build/shop_action_guard_gold_after_selected_buy_mean",
    "search/build/shop_action_guard_remove_selection_surface_rate",
    "search/build/shop_action_guard_remove_selection_selected_score_mean",
    "search/build/shop_action_guard_remove_selection_best_score_mean",
    "search/build/shop_action_guard_remove_selection_applied_rate",
    "search/build/shop_action_guard_alignment_error_rate",
    "search/build/shop_action_guard_invalid_obs_rate",
)


CAMPFIRE_REST_TAGS: tuple[str, ...] = (
    "decision/build/family_rest_rate",
    "route_heuristic/rest_before_elite_available_rate",
    "route_heuristic/rest_before_elite_selected_rate",
    "route_heuristic/no_rest_before_elite_selected_rate",
    "route_heuristic/rest_selected_low_hp_rate",
    "route_heuristic/rest_selected_when_low_hp_rate",
    "env/rest_site_encounters",
    "env/rest_heal_chosen",
    "env/rest_smith_chosen",
    "env/rest_skip_heal_chosen",
    "env/rest_skip_heal_at_low_hp",
    "env/rest_penalty_total",
    "env/rest_heal_exposure_low_hp",
    "env/rest_heal_exposure_heal_available",
    "env/rest_heal_exposure_forced",
    "env/rest_heal_exposure_miss",
    "search/build/rest_site_smith_guard_enabled",
    "search/build/rest_site_smith_guard_applicable_rate",
    "search/build/rest_site_smith_guard_smith_available_rate",
    "search/build/rest_site_smith_guard_selected_heal_safe_hp_rate",
    "search/build/rest_site_smith_guard_applied_rate",
    "search/build/rest_site_smith_guard_override_rate",
    "search/build/rest_site_smith_guard_invalid_obs_rate",
    "search/build/rest_site_smith_guard_alignment_error_rate",
    "search/build/rest_site_smith_guard_hp_ratio_mean",
    "search/build/rest_site_smith_guard_hp_threshold",
    "build/deck_upgrade_smith_selected",
    "build/deck_upgrade_smith_to_upgrade_seen_rate",
    "build/deck_upgrade_smith_no_upgrade_surface_rate",
    "build/deck_upgrade_seen",
    "build/deck_upgrade_selected_rate",
    "build/deck_upgrade_applied_rate",
    "build/deck_upgrade_target_count_mean",
    "build/deck_upgrade_applied_delta_mean",
    "build/deck_upgrade_context_present_rate",
    "build/deck_upgrade_upgraded_before_mean",
    "build/deck_upgrade_upgraded_after_mean",
)


DECK_BUILD_TAGS: tuple[str, ...] = (
    "deck/final_size",
    "deck/final_starter_count",
    "deck/final_starter_ratio",
    "deck/final_nonstarter_count",
    "deck/final_strike_count",
    "deck/final_defend_count",
    "deck/final_upgraded_count",
    "deck/final_upgraded_ratio",
    "deck/final_starter_upgrade_count",
    "deck/final_starter_upgrade_ratio",
    "deck/final_raw_avg_damage_per_energy",
    "deck/final_raw_avg_block_per_energy",
    "deck/final_raw_expected_extra_draw_per_turn",
    "deck/final_raw_expected_cards_seen_per_turn",
    "deck/final_attack_density",
    "deck/final_skill_density",
    "deck/final_power_density",
    "deck/final_curse_density",
    "deck/final_status_density",
    "deck/final_frontload_score",
    "deck/final_block_score",
    "deck/final_draw_engine_score",
    "deck/final_scaling_score",
    "deck/final_elite_readiness_score",
    "deck/final_boss_readiness_score",
    "deck/final_pollution_score",
    "deck/final_metadata_hit_rate",
    "deck/final_expected_hand_attack_cards",
    "deck/final_expected_hand_skill_cards",
    "deck/final_expected_hand_power_cards",
    "deck/final_expected_hand_curse_cards",
    "deck/final_expected_hand_status_cards",
    "deck/final_expected_hand_draw_cards",
    "deck/final_expected_hand_engine_cards",
    "deck/final_expected_hand_scaling_cards",
    "deck/final_expected_hand_unplayable_cards",
    "deck/final_expected_hand_attack_damage_per_turn",
    "deck/final_expected_hand_block_per_turn",
    "deck/final_expected_hand_attack_share",
    "deck/final_expected_hand_skill_share",
    "deck/final_expected_hand_power_share",
    "deck/final_expected_hand_junk_share",
    "deck/final_expected_hand_frontload_score",
    "deck/final_expected_hand_block_score",
    "deck/final_expected_hand_draw_score",
    "deck/final_expected_hand_engine_score",
    "deck/final_expected_hand_scaling_score",
    "deck/final_expected_hand_pollution_score",
    "deck/final_expected_hand_useful_quality_score",
    "deck/final_expected_hand_quality_attack_share",
    "deck/final_expected_hand_quality_block_share",
    "deck/final_expected_hand_quality_draw_share",
    "deck/final_expected_hand_quality_scaling_share",
    "deck/final_expected_hand_quality_pollution_share",
    "deck/final_raw_expected_energy_budget_per_turn",
    "deck/final_raw_expected_playable_cards_per_turn",
    "deck/final_raw_expected_playable_energy_spent_per_turn",
    "deck/final_raw_expected_unspent_energy_per_turn",
    "deck/final_raw_expected_playable_attack_cards_per_turn",
    "deck/final_raw_expected_playable_skill_cards_per_turn",
    "deck/final_raw_expected_playable_power_cards_per_turn",
    "deck/final_raw_expected_playable_attack_damage_per_turn",
    "deck/final_raw_expected_playable_block_per_turn",
    "deck/final_expected_energy_utilization_score",
    "deck/final_expected_playable_cards_score",
    "deck/final_expected_unspent_energy_score",
    "deck/final_expected_playable_frontload_score",
    "deck/final_expected_playable_block_score",
    "deck/final_cost_curve_zero_share",
    "deck/final_cost_curve_one_share",
    "deck/final_cost_curve_two_share",
    "deck/final_cost_curve_three_plus_share",
    "deck/final_cost_curve_x_share",
    "deck/final_combo_component_density",
    "deck/final_combo_enabler_density",
    "deck/final_combo_payoff_density",
    "deck/final_combo_option_value_score",
    "deck/final_combo_unmet_dependency_score",
    "deck/final_scaling_option_value_score",
    "deck/final_delayed_payoff_density",
    "deck/final_delayed_enabler_density",
    "deck/final_delayed_payoff_time_to_value_score",
    "deck/final_delayed_payoff_maturity_score",
    "deck/final_delayed_payoff_option_value_score",
    "deck/final_delayed_payoff_unrealized_risk_score",
    "death_deck/size",
    "death_deck/starter_count",
    "death_deck/starter_ratio",
    "death_deck/nonstarter_count",
    "death_deck/strike_count",
    "death_deck/defend_count",
    "death_deck/upgraded_count",
    "death_deck/upgraded_ratio",
    "death_deck/starter_upgrade_count",
    "death_deck/starter_upgrade_ratio",
    "death_deck/raw_avg_damage_per_energy",
    "death_deck/raw_avg_block_per_energy",
    "death_deck/raw_expected_extra_draw_per_turn",
    "death_deck/raw_expected_cards_seen_per_turn",
    "death_deck/frontload_score",
    "death_deck/block_score",
    "death_deck/draw_engine_score",
    "death_deck/scaling_score",
    "death_deck/elite_readiness_score",
    "death_deck/boss_readiness_score",
    "death_deck/pollution_score",
    "death_deck/metadata_hit_rate",
    "death_deck/expected_hand_attack_cards",
    "death_deck/expected_hand_skill_cards",
    "death_deck/expected_hand_power_cards",
    "death_deck/expected_hand_curse_cards",
    "death_deck/expected_hand_status_cards",
    "death_deck/expected_hand_draw_cards",
    "death_deck/expected_hand_engine_cards",
    "death_deck/expected_hand_scaling_cards",
    "death_deck/expected_hand_unplayable_cards",
    "death_deck/expected_hand_attack_damage_per_turn",
    "death_deck/expected_hand_block_per_turn",
    "death_deck/expected_hand_attack_share",
    "death_deck/expected_hand_skill_share",
    "death_deck/expected_hand_power_share",
    "death_deck/expected_hand_junk_share",
    "death_deck/expected_hand_frontload_score",
    "death_deck/expected_hand_block_score",
    "death_deck/expected_hand_draw_score",
    "death_deck/expected_hand_engine_score",
    "death_deck/expected_hand_scaling_score",
    "death_deck/expected_hand_pollution_score",
    "death_deck/expected_hand_useful_quality_score",
    "death_deck/expected_hand_quality_attack_share",
    "death_deck/expected_hand_quality_block_share",
    "death_deck/expected_hand_quality_draw_share",
    "death_deck/expected_hand_quality_scaling_share",
    "death_deck/expected_hand_quality_pollution_share",
    "death_deck/raw_expected_energy_budget_per_turn",
    "death_deck/raw_expected_playable_cards_per_turn",
    "death_deck/raw_expected_playable_energy_spent_per_turn",
    "death_deck/raw_expected_unspent_energy_per_turn",
    "death_deck/raw_expected_playable_attack_cards_per_turn",
    "death_deck/raw_expected_playable_skill_cards_per_turn",
    "death_deck/raw_expected_playable_power_cards_per_turn",
    "death_deck/raw_expected_playable_attack_damage_per_turn",
    "death_deck/raw_expected_playable_block_per_turn",
    "death_deck/expected_energy_utilization_score",
    "death_deck/expected_playable_cards_score",
    "death_deck/expected_unspent_energy_score",
    "death_deck/expected_playable_frontload_score",
    "death_deck/expected_playable_block_score",
    "death_deck/cost_curve_zero_share",
    "death_deck/cost_curve_one_share",
    "death_deck/cost_curve_two_share",
    "death_deck/cost_curve_three_plus_share",
    "death_deck/cost_curve_x_share",
    "death_deck/combo_component_density",
    "death_deck/combo_enabler_density",
    "death_deck/combo_payoff_density",
    "death_deck/combo_option_value_score",
    "death_deck/combo_unmet_dependency_score",
    "death_deck/scaling_option_value_score",
    "death_deck/delayed_payoff_density",
    "death_deck/delayed_enabler_density",
    "death_deck/delayed_payoff_time_to_value_score",
    "death_deck/delayed_payoff_maturity_score",
    "death_deck/delayed_payoff_option_value_score",
    "death_deck/delayed_payoff_unrealized_risk_score",
    "build/card_reward_seen",
    "build/card_reward_pick",
    "build/card_reward_skip",
    "build/card_reward_pick_rate",
    "build/card_reward_skip_rate",
    "build/card_reward_consecutive_skip_current",
    "build/card_reward_consecutive_skip_max",
    "search/build/card_reward_guard_context_rate",
    "search/build/card_reward_guard_applicable_rate",
    "search/build/card_reward_guard_selected_skip_rate",
    "search/build/card_reward_guard_selected_pick_rate",
    "search/build/card_reward_guard_pick_available_rate",
    "search/build/card_reward_guard_useful_candidate_available_rate",
    "search/build/card_reward_guard_no_useful_candidate_rate",
    "search/build/card_reward_guard_applied_rate",
    "search/build/card_reward_guard_override_rate",
    "search/build/card_reward_guard_skip_blocked_rate",
    "search/build/post_search_hard_guard_policy_retargeted_rate",
    "search/build/post_search_hard_guard_original_action_idx_mean",
    "search/build/post_search_hard_guard_final_action_idx_mean",
    "search/build/card_reward_pick_quality_guard_context_rate",
    "search/build/card_reward_pick_quality_guard_applicable_rate",
    "search/build/card_reward_pick_quality_guard_selected_pick_rate",
    "search/build/card_reward_pick_quality_guard_pick_available_rate",
    "search/build/card_reward_pick_quality_guard_accepted_available_rate",
    "search/build/card_reward_pick_quality_guard_applied_rate",
    "search/build/card_reward_pick_quality_guard_override_rate",
    "search/build/card_reward_pick_quality_guard_alignment_error_rate",
    "search/build/card_reward_pick_quality_guard_invalid_obs_rate",
    "search/build/card_reward_pick_quality_guard_not_act1_rate",
    "search/build/card_reward_pick_quality_guard_no_deficit_rate",
    "search/build/card_reward_pick_quality_guard_selected_rejected_rate",
    "search/build/card_reward_pick_quality_guard_best_score_mean",
    "search/build/card_reward_pick_quality_guard_selected_score_mean",
    "search/build/card_reward_pick_quality_guard_best_minus_selected_mean",
    "search/build/card_reward_pick_quality_guard_dynamic_min_useful_score_mean",
    "search/build/card_reward_pick_quality_guard_deck_size_mean",
    "search/build/card_reward_pick_quality_guard_low_block_rate",
    "search/build/card_reward_pick_quality_guard_low_draw_rate",
    "search/build/card_reward_pick_quality_guard_low_attack_rate",
    "search/build/card_reward_pick_quality_guard_low_scaling_rate",
    "search/build/card_reward_pick_quality_guard_thin_deck_rate",
    "search/build/card_reward_guard_alignment_error_rate",
    "search/build/card_reward_guard_invalid_obs_rate",
    "search/build/card_reward_guard_not_act1_rate",
    "search/build/card_reward_guard_deck_size_mean",
    "search/build/card_reward_guard_best_score_mean",
    "search/build/card_reward_guard_selected_score_mean",
    "search/build/card_reward_guard_best_standalone_value_mean",
    "search/build/card_reward_guard_best_combo_current_fit_mean",
    "search/build/card_reward_guard_best_combo_missing_piece_fit_mean",
    "search/build/card_reward_guard_best_combo_speculative_option_mean",
    "search/build/card_reward_guard_best_combo_orphan_risk_mean",
    "search/build/card_reward_guard_best_combo_final_option_value_mean",
    "search/build/card_reward_guard_damage_per_energy_mean",
    "search/build/card_reward_guard_block_per_energy_mean",
    "search/build/card_reward_guard_expected_extra_draw_mean",
    "search/build/card_reward_guard_expected_cards_seen_mean",
    "search/build/card_reward_guard_expected_hand_attack_damage_mean",
    "search/build/card_reward_guard_expected_hand_block_mean",
    "search/build/card_reward_guard_expected_hand_attack_share_mean",
    "search/build/card_reward_guard_expected_hand_skill_share_mean",
    "search/build/card_reward_guard_expected_hand_power_share_mean",
    "search/build/card_reward_guard_expected_hand_junk_share_mean",
    "search/build/card_reward_guard_expected_hand_useful_quality_mean",
    "search/build/card_reward_guard_expected_playable_cards_mean",
    "search/build/card_reward_guard_expected_playable_attack_damage_mean",
    "search/build/card_reward_guard_expected_playable_block_mean",
    "search/build/card_reward_guard_expected_energy_utilization_mean",
    "search/build/card_reward_guard_combo_option_value_mean",
    "search/build/card_reward_guard_combo_unmet_dependency_mean",
    "search/build/card_reward_guard_future_combo_window_mean",
    "search/build/card_reward_guard_future_combo_floor_window_mean",
    "search/build/card_reward_guard_future_combo_route_window_mean",
    "search/build/card_reward_guard_future_reward_opportunity_mean",
    "search/build/card_reward_guard_future_runway_observed_rate",
    "search/build/card_reward_guard_delayed_payoff_option_value_mean",
    "search/build/card_reward_guard_delayed_payoff_unrealized_risk_mean",
    "search/build/card_reward_guard_delayed_payoff_maturity_mean",
    "search/build/card_reward_guard_delayed_payoff_time_to_value_mean",
)


ALL_TAGS: tuple[str, ...] = (
    *PROGRESS_TAGS,
    *DECISION_TAGS,
    *LOSS_TAGS,
    *MEMORY_TAGS,
    *TACTICAL_REDLINE_TAGS,
    *POTION_SURVIVAL_TAGS,
    *SHOP_ECONOMY_TAGS,
    *CAMPFIRE_REST_TAGS,
    *DECK_BUILD_TAGS,
)


def latest_fullrun(root: Path) -> Path:
    pointer = root / "latest_pass_large_fullrun_run_id.txt"
    if pointer.exists():
        run_id = pointer.read_text(encoding="utf-8").strip()
        if run_id:
            candidate = root / run_id
            if candidate.exists():
                return candidate
    return sandbox_monitor.latest_run(root)


def _finite(values: list[float]) -> list[float]:
    return [float(value) for value in values if math.isfinite(float(value))]


def summarize_scalar(ea: EventAccumulator, tag: str, tail: int) -> dict[str, Any]:
    events = ea.Scalars(tag)
    values = _finite([float(event.value) for event in events])
    if not events or not values:
        return {
            "n": 0,
            "step": None,
            "last": None,
            "avg_tail": None,
            "min_tail": None,
            "max_tail": None,
        }
    tail_values = values[-min(int(tail), len(values)) :]
    return {
        "n": len(events),
        "step": int(events[-1].step),
        "last": float(values[-1]),
        "avg_tail": float(mean(tail_values)),
        "min_tail": float(min(tail_values)),
        "max_tail": float(max(tail_values)),
    }


def load_scalars(run_dir: Path, tail: int) -> tuple[dict[str, dict[str, Any]], set[str]]:
    ea = EventAccumulator(str(run_dir), size_guidance={"scalars": 0})
    ea.Reload()
    tags = set(ea.Tags().get("scalars", []))
    summaries: dict[str, dict[str, Any]] = {}
    for tag in ALL_TAGS:
        if tag in tags:
            summaries[tag] = summarize_scalar(ea, tag, tail)
    return summaries, tags


def value(summary: dict[str, dict[str, Any]], tag: str, field: str = "last") -> float | None:
    item = summary.get(tag)
    if not item:
        return None
    raw = item.get(field)
    return float(raw) if isinstance(raw, (int, float)) and math.isfinite(float(raw)) else None


def has_metric(summary: dict[str, dict[str, Any]], tag: str) -> bool:
    item = summary.get(tag)
    return bool(item and int(item.get("n") or 0) > 0)


def deck_build_warnings(summary: dict[str, dict[str, Any]]) -> list[dict[str, Any]]:
    """Return operator-facing deck/build warnings without changing gate verdict.

    These are deliberately advisory.  They answer whether the current Act1
    failures look like under-building (thin deck / repeated reward skips) or low
    final/death deck quality, but they should not stop a run by themselves.
    """

    warnings: list[dict[str, Any]] = []

    def add(
        kind: str,
        tag: str,
        *,
        field: str,
        actual: float | None,
        threshold: float,
        message: str,
    ) -> None:
        warnings.append(
            {
                "kind": kind,
                "tag": tag,
                "field": field,
                "actual": actual,
                "threshold": float(threshold),
                "message": message,
            }
        )

    seen_avg = value(summary, "build/card_reward_seen", "avg_tail")
    skip_rate_avg = value(summary, "build/card_reward_skip_rate", "avg_tail")
    consecutive_skip_max = value(summary, "build/card_reward_consecutive_skip_max", "max_tail")
    if seen_avg is not None and skip_rate_avg is not None and seen_avg >= 3.0 and skip_rate_avg > 0.50:
        add(
            "card_reward_skip_rate",
            "build/card_reward_skip_rate",
            field="avg_tail",
            actual=skip_rate_avg,
            threshold=0.50,
            message="card rewards are being skipped too often in the recent tail",
        )
    if consecutive_skip_max is not None and consecutive_skip_max >= 4.0:
        add(
            "card_reward_consecutive_skip",
            "build/card_reward_consecutive_skip_max",
            field="max_tail",
            actual=consecutive_skip_max,
            threshold=4.0,
            message="observed 4+ consecutive card-reward skips in at least one episode",
        )
    skip_blocked_rate = value(summary, "search/build/card_reward_guard_skip_blocked_rate", "avg_tail")
    policy_retargeted_rate = value(
        summary,
        "search/build/post_search_hard_guard_policy_retargeted_rate",
        "avg_tail",
    )
    if skip_blocked_rate is not None and skip_blocked_rate > 0.01:
        if policy_retargeted_rate is None:
            add(
                "card_reward_policy_retarget_missing",
                "search/build/post_search_hard_guard_policy_retargeted_rate",
                field="avg_tail",
                actual=None,
                threshold=skip_blocked_rate,
                message=(
                    "card-reward skip guard is firing but post-search policy-retarget telemetry is missing; "
                    "new-run validation cannot prove replay learns the final guarded pick"
                ),
            )
        elif policy_retargeted_rate + 1e-9 < 0.80 * skip_blocked_rate:
            add(
                "card_reward_policy_retarget_low",
                "search/build/post_search_hard_guard_policy_retargeted_rate",
                field="avg_tail",
                actual=policy_retargeted_rate,
                threshold=0.80 * skip_blocked_rate,
                message=(
                    "card-reward skip guard is firing but policy target rewrite rate is much lower; "
                    "replay may still reinforce skipped rewards"
                ),
            )

    shop_seen_avg = value(summary, "build/shop_seen", "avg_tail")
    if shop_seen_avg is None:
        # Older/self-play processes can still expose coarse build/route shop
        # signals while missing the richer shop telemetry tags added for the
        # Act1 build-quality work.  Surface that as an advisory warning so a
        # run does not silently look "clean" just because the detailed shop
        # tracker is not writing yet.
        shop_family_rate = value(summary, "decision/build/family_shop_rate", "avg_tail")
        shop_with_gold_rate = value(summary, "route_heuristic/shop_with_gold_available_rate", "avg_tail")
        shop_high_gold_selected_rate = value(summary, "route_heuristic/shop_selected_high_gold_rate", "avg_tail")
        if any(
            rate is not None and rate > 0.0
            for rate in (shop_family_rate, shop_with_gold_rate, shop_high_gold_selected_rate)
        ):
            add(
                "shop_metrics_missing",
                "build/shop_seen",
                field="avg_tail",
                actual=None,
                threshold=1.0,
                message=(
                    "coarse shop/build route signals exist but detailed build/shop_* telemetry is missing; "
                    "new-run validation cannot judge shop open/buy/remove behavior yet"
                ),
            )
    if shop_seen_avg is not None and shop_seen_avg >= 1.0:
        shop_open_rate = value(summary, "build/shop_open_rate", "avg_tail")
        shop_buy_rate = value(summary, "build/shop_buy_any_rate", "avg_tail")
        shop_remove_affordable_rate = value(summary, "build/shop_remove_affordable_rate", "avg_tail")
        shop_remove_rate = value(summary, "build/shop_remove_rate", "avg_tail")
        shop_leave_gold_rate = value(summary, "build/shop_leave_with_gold_ge_100_rate", "avg_tail")
        shop_leave_remove_rate = value(summary, "build/shop_leave_with_remove_affordable_rate", "avg_tail")
        if shop_open_rate is not None and shop_open_rate < 0.35:
            add(
                "low_shop_open_rate",
                "build/shop_open_rate",
                field="avg_tail",
                actual=shop_open_rate,
                threshold=0.35,
                message="shop surfaces are seen but the agent rarely opens the merchant/inventory",
            )
        if shop_buy_rate is not None and shop_buy_rate < 0.20:
            add(
                "low_shop_buy_rate",
                "build/shop_buy_any_rate",
                field="avg_tail",
                actual=shop_buy_rate,
                threshold=0.20,
                message="shop surfaces are seen but purchases are rare",
            )
        if (
            shop_remove_affordable_rate is not None
            and shop_remove_rate is not None
            and shop_remove_affordable_rate >= 0.10
            and shop_remove_rate < 0.05
        ):
            add(
                "low_shop_remove_rate",
                "build/shop_remove_rate",
                field="avg_tail",
                actual=shop_remove_rate,
                threshold=0.05,
                message="card removal is affordable in shops but is almost never selected",
            )
        if shop_leave_gold_rate is not None and shop_leave_gold_rate > 0.35:
            add(
                "shop_leave_with_gold",
                "build/shop_leave_with_gold_ge_100_rate",
                field="avg_tail",
                actual=shop_leave_gold_rate,
                threshold=0.35,
                message="agent often leaves shops while still holding at least 100 gold",
            )
        if shop_leave_remove_rate is not None and shop_leave_remove_rate > 0.10:
            add(
                "shop_leave_with_remove_affordable",
                "build/shop_leave_with_remove_affordable_rate",
                field="avg_tail",
                actual=shop_leave_remove_rate,
                threshold=0.10,
                message="agent leaves shops while card removal is affordable",
            )

    max_floor_avg = value(summary, "episode/max_floor", "avg_tail")
    final_size_avg = value(summary, "deck/final_size", "avg_tail")
    if final_size_avg is not None and max_floor_avg is not None and max_floor_avg >= 7.0 and final_size_avg <= 12.0:
        add(
            "thin_final_deck",
            "deck/final_size",
            field="avg_tail",
            actual=final_size_avg,
            threshold=12.0,
            message="episodes are reaching mid Act1 with a very small final deck",
        )

    final_starter_ratio_avg = value(summary, "deck/final_starter_ratio", "avg_tail")
    final_starter_count_avg = value(summary, "deck/final_starter_count", "avg_tail")
    if (
        max_floor_avg is not None
        and max_floor_avg >= 7.0
        and final_starter_ratio_avg is not None
        and final_starter_ratio_avg >= 0.45
    ):
        add(
            "starter_heavy_final_deck",
            "deck/final_starter_ratio",
            field="avg_tail",
            actual=final_starter_ratio_avg,
            threshold=0.45,
            message="episodes reach mid Act1 with starter cards still making up too much of the deck",
        )
    final_nonstarter_avg = value(summary, "deck/final_nonstarter_count", "avg_tail")
    if (
        max_floor_avg is not None
        and max_floor_avg >= 7.0
        and final_nonstarter_avg is not None
        and final_nonstarter_avg <= 6.0
        and (final_starter_count_avg is None or final_starter_count_avg >= 6.0)
    ):
        add(
            "low_final_nonstarter_count",
            "deck/final_nonstarter_count",
            field="avg_tail",
            actual=final_nonstarter_avg,
            threshold=6.0,
            message="episodes reach mid Act1 without adding enough non-starter cards",
        )

    death_floor_avg = value(summary, "episode/death_floor", "avg_tail")
    death_size_avg = value(summary, "death_deck/size", "avg_tail")
    if death_size_avg is not None and death_floor_avg is not None and death_floor_avg >= 7.0 and death_size_avg <= 12.0:
        add(
            "thin_death_deck",
            "death_deck/size",
            field="avg_tail",
            actual=death_size_avg,
            threshold=12.0,
            message="death episodes are still dying with starter-sized decks",
        )

    death_starter_ratio_avg = value(summary, "death_deck/starter_ratio", "avg_tail")
    death_starter_count_avg = value(summary, "death_deck/starter_count", "avg_tail")
    if (
        death_floor_avg is not None
        and death_floor_avg >= 7.0
        and death_starter_ratio_avg is not None
        and death_starter_count_avg is not None
        and death_starter_count_avg >= 6.0
        and death_starter_ratio_avg >= 0.45
    ):
        add(
            "starter_heavy_death_deck",
            "death_deck/starter_ratio",
            field="avg_tail",
            actual=death_starter_ratio_avg,
            threshold=0.45,
            message="death decks are still dominated by Strike/Defend starters",
        )
    death_nonstarter_avg = value(summary, "death_deck/nonstarter_count", "avg_tail")
    if (
        death_floor_avg is not None
        and death_floor_avg >= 7.0
        and death_nonstarter_avg is not None
        and death_nonstarter_avg <= 7.0
    ):
        add(
            "low_death_nonstarter_count",
            "death_deck/nonstarter_count",
            field="avg_tail",
            actual=death_nonstarter_avg,
            threshold=7.0,
            message="death decks contain too few non-starter cards for mid/late Act1",
        )

    death_atk_avg = value(summary, "death_deck/expected_hand_attack_damage_per_turn", "avg_tail")
    if death_atk_avg is not None and death_floor_avg is not None and death_floor_avg >= 7.0 and death_atk_avg < 12.0:
        add(
            "low_death_hand_attack",
            "death_deck/expected_hand_attack_damage_per_turn",
            field="avg_tail",
            actual=death_atk_avg,
            threshold=12.0,
            message="death decks have low expected attack output in a typical hand",
        )

    death_block_avg = value(summary, "death_deck/expected_hand_block_per_turn", "avg_tail")
    if death_block_avg is not None and death_floor_avg is not None and death_floor_avg >= 7.0 and death_block_avg < 10.0:
        add(
            "low_death_hand_block",
            "death_deck/expected_hand_block_per_turn",
            field="avg_tail",
            actual=death_block_avg,
            threshold=10.0,
            message="death decks have low expected block in a typical hand",
        )

    cards_seen_avg = value(summary, "deck/final_raw_expected_cards_seen_per_turn", "avg_tail")
    deck_size_avg = value(summary, "deck/final_size", "avg_tail")
    if cards_seen_avg is not None and deck_size_avg is not None and deck_size_avg >= 14.0 and cards_seen_avg <= 5.1:
        add(
            "low_rotation",
            "deck/final_raw_expected_cards_seen_per_turn",
            field="avg_tail",
            actual=cards_seen_avg,
            threshold=5.1,
            message="deck has grown but expected cards seen per turn remains near the base hand size",
        )

    death_playable_cards = value(summary, "death_deck/raw_expected_playable_cards_per_turn", "avg_tail")
    if death_playable_cards is not None and death_floor_avg is not None and death_floor_avg >= 7.0 and death_playable_cards < 2.2:
        add(
            "low_death_playable_cards",
            "death_deck/raw_expected_playable_cards_per_turn",
            field="avg_tail",
            actual=death_playable_cards,
            threshold=2.2,
            message="death decks cannot convert a typical hand into enough played cards under energy budget",
        )

    death_playable_attack = value(summary, "death_deck/raw_expected_playable_attack_damage_per_turn", "avg_tail")
    if death_playable_attack is not None and death_floor_avg is not None and death_floor_avg >= 7.0 and death_playable_attack < 14.0:
        add(
            "low_death_playable_attack",
            "death_deck/raw_expected_playable_attack_damage_per_turn",
            field="avg_tail",
            actual=death_playable_attack,
            threshold=14.0,
            message="death decks have low attack that can actually be spent under the energy budget",
        )

    death_playable_block = value(summary, "death_deck/raw_expected_playable_block_per_turn", "avg_tail")
    if death_playable_block is not None and death_floor_avg is not None and death_floor_avg >= 7.0 and death_playable_block < 10.0:
        add(
            "low_death_playable_block",
            "death_deck/raw_expected_playable_block_per_turn",
            field="avg_tail",
            actual=death_playable_block,
            threshold=10.0,
            message="death decks have low block that can actually be spent under the energy budget",
        )

    death_energy_util = value(summary, "death_deck/expected_energy_utilization_score", "avg_tail")
    if (
        death_energy_util is not None
        and death_size_avg is not None
        and death_size_avg >= 14.0
        and death_energy_util < 0.65
    ):
        add(
            "low_death_energy_utilization",
            "death_deck/expected_energy_utilization_score",
            field="avg_tail",
            actual=death_energy_util,
            threshold=0.65,
            message="death deck has cards but cannot spend energy into useful effects reliably",
        )

    death_high_cost = value(summary, "death_deck/cost_curve_three_plus_share", "avg_tail")
    death_draw = value(summary, "death_deck/raw_expected_extra_draw_per_turn", "avg_tail")
    if (
        death_high_cost is not None
        and death_draw is not None
        and death_high_cost > 0.25
        and death_draw < 1.0
    ):
        add(
            "death_cost_curve_bloat",
            "death_deck/cost_curve_three_plus_share",
            field="avg_tail",
            actual=death_high_cost,
            threshold=0.25,
            message="death deck has too many expensive cards without enough draw/engine support",
        )

    death_combo_unmet = value(summary, "death_deck/combo_unmet_dependency_score", "avg_tail")
    death_combo_density = value(summary, "death_deck/combo_component_density", "avg_tail")
    if (
        death_combo_unmet is not None
        and death_combo_density is not None
        and death_combo_unmet > 0.50
        and death_combo_density > 0.15
    ):
        add(
            "death_orphan_combo_components",
            "death_deck/combo_unmet_dependency_score",
            field="avg_tail",
            actual=death_combo_unmet,
            threshold=0.50,
            message="death deck contains combo components whose dependencies were not assembled",
        )

    death_delayed_density = value(summary, "death_deck/delayed_payoff_density", "avg_tail")
    death_delayed_risk = value(summary, "death_deck/delayed_payoff_unrealized_risk_score", "avg_tail")
    death_delayed_maturity = value(summary, "death_deck/delayed_payoff_maturity_score", "avg_tail")
    if (
        death_delayed_density is not None
        and death_delayed_risk is not None
        and death_delayed_maturity is not None
        and death_delayed_density > 0.12
        and death_delayed_risk > 0.55
        and death_delayed_maturity < 0.35
    ):
        add(
            "death_unrealized_delayed_payoff",
            "death_deck/delayed_payoff_unrealized_risk_score",
            field="avg_tail",
            actual=death_delayed_risk,
            threshold=0.55,
            message="death deck invested in multi-turn payoff cards but lacked enough enablers/rotation to realize them",
        )

    return warnings


def episode_events(summary: dict[str, dict[str, Any]]) -> int:
    candidates = []
    for tag in ("episode/max_floor", "episode/reward", "episode/length", "episode/act1_boss_seen"):
        item = summary.get(tag)
        if item:
            candidates.append(int(item.get("n") or 0))
    return max(candidates, default=0)


def check_ge(
    summary: dict[str, dict[str, Any]],
    tag: str,
    threshold: float,
    *,
    field: str = "last",
    required: bool = True,
) -> dict[str, Any]:
    actual = value(summary, tag, field)
    if actual is None:
        return {
            "tag": tag,
            "field": field,
            "op": ">=",
            "threshold": float(threshold),
            "actual": None,
            "status": "missing" if required else "skipped",
        }
    return {
        "tag": tag,
        "field": field,
        "op": ">=",
        "threshold": float(threshold),
        "actual": actual,
        "status": "pass" if actual >= threshold else "fail",
    }


def check_le(
    summary: dict[str, dict[str, Any]],
    tag: str,
    threshold: float,
    *,
    field: str = "avg_tail",
    required: bool = True,
) -> dict[str, Any]:
    actual = value(summary, tag, field)
    if actual is None:
        return {
            "tag": tag,
            "field": field,
            "op": "<=",
            "threshold": float(threshold),
            "actual": None,
            "status": "missing" if required else "skipped",
        }
    return {
        "tag": tag,
        "field": field,
        "op": "<=",
        "threshold": float(threshold),
        "actual": actual,
        "status": "pass" if actual <= threshold else "fail",
    }


def _progress_thresholds(mode: str) -> dict[str, float]:
    if mode == "strong":
        return {
            "boss_seen_64": 0.60,
            "pass_64": 1.0 / 64.0,
            "pass_256": 1.0 / 256.0,
            "max_floor_mean_64": 12.0,
            "reach_10_64": 0.70,
            "reach_12_64": 0.45,
        }
    return {
        "boss_seen_64": 0.30,
        "pass_64": 0.0,
        "pass_256": 0.0,
        "max_floor_mean_64": 9.0,
        "reach_10_64": 0.45,
        "reach_12_64": 0.20,
    }


def evaluate_gate(
    summary: dict[str, dict[str, Any]],
    *,
    mode: str,
    min_episodes: int,
    min_buffer: float,
    max_reserved_gb: float,
    max_peak_allocated_gb: float,
    max_empty_cache_rate: float,
    max_wasteful_end_turn_rate: float,
    max_refund_no_followup_with_progress_rate: float,
    max_potion_low_urgency_rate: float,
) -> dict[str, Any]:
    mode = "strong" if str(mode).lower().strip() == "strong" else "early"
    thresholds = _progress_thresholds(mode)
    checks: list[dict[str, Any]] = []

    eps = episode_events(summary)
    buffer_last = value(summary, "buffer/size", "last")

    checks.append(
        {
            "tag": "episode_count",
            "field": "n",
            "op": ">=",
            "threshold": float(min_episodes),
            "actual": float(eps),
            "status": "pass" if eps >= int(min_episodes) else "fail",
        }
    )
    checks.append(check_ge(summary, "buffer/size", min_buffer, field="last"))

    # Act1 progression checks.  The early mode is for "is this worth continuing
    # tonight"; the strong mode is the stricter "start calling this an Act1
    # candidate" gate.
    checks.append(
        check_ge(
            summary,
            "recent_tail/64/act1_boss_seen_rate",
            thresholds["boss_seen_64"],
            field="last",
        )
    )
    checks.append(
        check_ge(
            summary,
            "recent_tail/64/act1_pass_rate",
            thresholds["pass_64"],
            field="last",
        )
    )
    if mode == "strong":
        checks.append(
            check_ge(
                summary,
                "recent_tail/256/act1_pass_rate",
                thresholds["pass_256"],
                field="last",
            )
        )
    else:
        checks.append(check_ge(summary, "recent_tail/256/act1_pass_rate", 0.0, field="last", required=False))
    checks.append(
        check_ge(
            summary,
            "recent_tail/64/max_floor_mean",
            thresholds["max_floor_mean_64"],
            field="last",
        )
    )
    checks.append(
        check_ge(
            summary,
            "recent_tail/64/reached_floor_10_rate",
            thresholds["reach_10_64"],
            field="last",
        )
    )
    checks.append(
        check_ge(
            summary,
            "recent_tail/64/reached_floor_12_rate",
            thresholds["reach_12_64"],
            field="last",
        )
    )

    # Loss sanity.  These are loose spike sentinels.
    checks.append(check_le(summary, "loss/total", 100.0, field="max_tail"))
    checks.append(check_le(summary, "loss/future_world_aux", 10.0, field="max_tail"))
    checks.append(check_le(summary, "loss/future_bank_state", 10.0, field="max_tail"))
    checks.append(check_le(summary, "loss/future_bank_delta", 5.0, field="max_tail"))
    checks.append(check_le(summary, "loss/future_bank_token_slot_source", 0.0, field="max_tail", required=False))

    # Memory sanity for Pass-Large low-VRAM runs.
    checks.append(check_le(summary, "memory/reserved_gb", max_reserved_gb, field="max_tail", required=False))
    if has_metric(summary, "memory/max_allocated_gb"):
        checks.append(
            check_le(
                summary,
                "memory/max_allocated_gb",
                max_peak_allocated_gb,
                field="max_tail",
                required=False,
            )
        )
    else:
        checks.append(check_le(summary, "memory/allocated_gb", max_peak_allocated_gb, field="max_tail", required=False))
    checks.append(check_le(summary, "memory/empty_cache_called", max_empty_cache_rate, field="avg_tail", required=False))

    # Tactical regressions should remain low even in full-run.  Use optional
    # checks so old checkpoints without the tags can still be inspected.
    checks.append(check_le(summary, "combat_quality/hp_cost_self_lethal_selected_rate", 0.0, field="avg_tail", required=False))
    checks.append(check_le(summary, "combat_quality/zero_energy_x_cost_selected_rate", 0.005, field="avg_tail", required=False))
    checks.append(check_le(summary, "combat_quality/hp_cost_low_margin_selected_rate", 0.005, field="avg_tail", required=False))
    checks.append(check_le(summary, "combat_quality/insufficient_block_selected_rate", 0.005, field="avg_tail", required=False))
    checks.append(
        check_le(
            summary,
            "combat_quality/wasteful_end_turn_rate",
            max_wasteful_end_turn_rate,
            field="avg_tail",
            required=False,
        )
    )
    checks.append(
        check_le(
            summary,
            "combat_quality/refund_no_followup_with_progress_selected_rate",
            max_refund_no_followup_with_progress_rate,
            field="avg_tail",
            required=False,
        )
    )
    checks.append(
        check_le(
            summary,
            "combat_quality/potion_low_urgency_selected_rate",
            max_potion_low_urgency_rate,
            field="avg_tail",
            required=False,
        )
    )

    # Lucky Tonic / 幸运药剂 is a boss survival tool.  A nonzero skip-on-death
    # rate means the potion was seen/legal in a boss fight that ended in death
    # but was never selected, so surface it as a hard red line when the tag is
    # present.  The tag is optional because it only appears after boss samples.
    checks.append(check_le(summary, "boss/lucky_skip_on_boss_death_rate", 0.0, field="avg_tail", required=False))
    checks.append(
        check_le(
            summary,
            "boss/final_lucky_unused_on_boss_death_rate",
            0.0,
            field="avg_tail",
            required=False,
        )
    )

    if eps < int(min_episodes):
        verdict = "WAIT_EPISODES"
    elif buffer_last is None or buffer_last < float(min_buffer):
        verdict = "WAIT_BUFFER"
    elif any(check["status"] == "missing" for check in checks):
        verdict = "MISSING_TAGS"
    elif any(check["status"] == "fail" for check in checks):
        verdict = "FAIL"
    else:
        verdict = "PASS"

    return {
        "verdict": verdict,
        "mode": mode,
        "episode_events": eps,
        "buffer_last": buffer_last,
        "checks": checks,
    }


def _fmt(value_: Any) -> str:
    if value_ is None:
        return "None"
    if isinstance(value_, (int, float)):
        return f"{float(value_):.6g}"
    return str(value_)


def print_human(payload: dict[str, Any]) -> None:
    print(f"run_dir: {payload['run_dir']}")
    print(f"scalar_tag_count: {payload['scalar_tag_count']}")
    print(f"tail: {payload['tail']}")
    print(f"mode: {payload['gate']['mode']}")
    print(f"verdict: {payload['gate']['verdict']}")
    print()

    sections = (
        ("PROGRESS", PROGRESS_TAGS),
        ("DECISIONS / PLANNER", DECISION_TAGS),
        ("LOSS", LOSS_TAGS),
        ("MEMORY", MEMORY_TAGS),
        ("TACTICAL REDLINES", TACTICAL_REDLINE_TAGS),
        ("POTION / BOSS SURVIVAL", POTION_SURVIVAL_TAGS),
        ("SHOP / ECONOMY", SHOP_ECONOMY_TAGS),
        ("CAMPFIRE / REST-SMITH", CAMPFIRE_REST_TAGS),
        ("DECK / BUILD QUALITY", DECK_BUILD_TAGS),
    )
    for title, tags in sections:
        print(title)
        print("-" * len(title))
        for tag in tags:
            item = payload["metrics"].get(tag)
            if item is None:
                print(f"{tag:<62} MISSING")
                continue
            print(
                f"{tag:<62} "
                f"last={_fmt(item['last'])} avg={_fmt(item['avg_tail'])} "
                f"min={_fmt(item['min_tail'])} max={_fmt(item['max_tail'])} "
                f"step={item['step']} n={item['n']}"
            )
        print()

    warnings = payload.get("deck_build_warnings") or []
    if warnings:
        print("DECK / BUILD WARNINGS")
        print("---------------------")
        for warning in warnings:
            print(
                f"{warning.get('kind')}: {warning.get('tag')}[{warning.get('field')}] "
                f"actual={_fmt(warning.get('actual'))} "
                f"threshold={_fmt(warning.get('threshold'))} — {warning.get('message')}"
            )
        print()

    print("GATE CHECKS")
    print("-----------")
    for check in payload["gate"]["checks"]:
        print(
            f"{check['status']:<8} {check['tag']}[{check['field']}] "
            f"{check['op']} {check['threshold']} actual={_fmt(check['actual'])}"
        )

    diagnostics = payload.get("diagnostics")
    if isinstance(diagnostics, dict):
        print()
        print("DIAGNOSTICS")
        print("-----------")
        if not diagnostics.get("available"):
            print(f"diagnostics_dir: {diagnostics.get('diagnostics_dir')} MISSING")
            return
        offenders = diagnostics.get("offenders") or {}
        deaths = diagnostics.get("deaths") or {}
        print(
            "offenders: "
            f"rows={offenders.get('rows')} "
            f"recent_window={diagnostics.get('window')} "
            f"recent_rows={offenders.get('recent_rows')} "
            f"max_step={offenders.get('max_step')}"
        )
        print("top offender types:")
        for item in offenders.get("top_types") or []:
            print(f"  {item['name']}: {item['count']}")
        print("top offender encounters:")
        for item in offenders.get("top_encounters") or []:
            print(f"  {item['name']}: {item['count']}")
        print(
            "deaths: "
            f"rows={deaths.get('rows')} "
            f"recent_window={diagnostics.get('window')} "
            f"recent_rows={deaths.get('recent_rows')} "
            f"max_step={deaths.get('max_step')}"
        )
        print("top recent death encounters:")
        for item in deaths.get("top_encounters") or []:
            print(f"  {item['name']}: {item['count']}")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=DEFAULT_ROOT, help="logs_muzero root")
    parser.add_argument("--run-dir", type=Path, default=None, help="specific TensorBoard run directory")
    parser.add_argument("--tail", type=int, default=50, help="tail window for averages/min/max")
    parser.add_argument("--mode", choices=("early", "strong"), default="early")
    parser.add_argument("--min-episodes", type=int, default=64)
    parser.add_argument("--min-buffer", type=float, default=3000.0)
    parser.add_argument("--max-reserved-gb", type=float, default=22.0)
    parser.add_argument("--max-peak-allocated-gb", type=float, default=20.0)
    parser.add_argument("--max-empty-cache-rate", type=float, default=0.2)
    parser.add_argument("--max-wasteful-end-turn-rate", type=float, default=0.002)
    parser.add_argument("--max-refund-no-followup-with-progress-rate", type=float, default=0.005)
    parser.add_argument("--max-potion-low-urgency-rate", type=float, default=0.01)
    parser.add_argument("--diagnostics-window", type=int, default=2000)
    parser.add_argument("--no-diagnostics", action="store_true")
    parser.add_argument("--json", action="store_true", help="emit JSON only")
    parser.add_argument("--fail-on-red", action="store_true", help="exit nonzero on FAIL/MISSING_TAGS")
    args = parser.parse_args()

    run_dir = args.run_dir or latest_fullrun(args.root)
    summaries, tags = load_scalars(run_dir, tail=max(int(args.tail), 1))
    payload: dict[str, Any] = {
        "run_dir": str(run_dir),
        "scalar_tag_count": len(tags),
        "tail": int(args.tail),
        "metrics": summaries,
        "missing_progress_tags": [tag for tag in PROGRESS_TAGS if tag not in tags],
        "missing_decision_tags": [tag for tag in DECISION_TAGS if tag not in tags],
        "missing_loss_tags": [tag for tag in LOSS_TAGS if tag not in tags],
        "missing_memory_tags": [tag for tag in MEMORY_TAGS if tag not in tags],
        "missing_tactical_redline_tags": [tag for tag in TACTICAL_REDLINE_TAGS if tag not in tags],
        "missing_potion_survival_tags": [tag for tag in POTION_SURVIVAL_TAGS if tag not in tags],
        "missing_shop_economy_tags": [tag for tag in SHOP_ECONOMY_TAGS if tag not in tags],
        "missing_campfire_rest_tags": [tag for tag in CAMPFIRE_REST_TAGS if tag not in tags],
        "missing_deck_build_tags": [tag for tag in DECK_BUILD_TAGS if tag not in tags],
        "deck_build_warnings": deck_build_warnings(summaries),
        "gate": evaluate_gate(
            summaries,
            mode=str(args.mode),
            min_episodes=int(args.min_episodes),
            min_buffer=float(args.min_buffer),
            max_reserved_gb=float(args.max_reserved_gb),
            max_peak_allocated_gb=float(args.max_peak_allocated_gb),
            max_empty_cache_rate=float(args.max_empty_cache_rate),
            max_wasteful_end_turn_rate=float(args.max_wasteful_end_turn_rate),
            max_refund_no_followup_with_progress_rate=float(args.max_refund_no_followup_with_progress_rate),
            max_potion_low_urgency_rate=float(args.max_potion_low_urgency_rate),
        ),
    }
    if not args.no_diagnostics:
        payload["diagnostics"] = sandbox_monitor.load_diagnostics_summary(
            run_dir,
            window=int(args.diagnostics_window),
        )

    if args.json:
        print(json.dumps(payload, indent=2, ensure_ascii=False))
    else:
        print_human(payload)

    if args.fail_on_red and payload["gate"]["verdict"] in {"FAIL", "MISSING_TAGS"}:
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
