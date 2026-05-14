"""Episode, search-stat, and boss-combat metric helpers.

This mixin owns post-episode aggregation and compact metric normalization for
MuZero training.  It intentionally does not make policy decisions; tactical
biases and hard guards should stay in ``muzero.combat_quality`` / ``strategy``.
"""

from __future__ import annotations

from collections import Counter, defaultdict
import re
from typing import Any

import numpy as np
import torch

from combat_snapshot_dataset import infer_encounter_tier
from muzero.combat_quality.metrics import (
    COMBAT_QUALITY_CARD_BLOCK_SEARCH_SUFFIXES,
    COMBAT_QUALITY_GUARD_SEARCH_SUFFIXES,
    COMBAT_QUALITY_TACTICAL_SEARCH_SUFFIXES,
    boss_card_block_waste_metrics,
    card_block_waste_metrics,
    combat_tactical_issue_metrics,
)

COMBAT_QUALITY_EPISODE_SEARCH_SUFFIXES = {
    **COMBAT_QUALITY_CARD_BLOCK_SEARCH_SUFFIXES,
    **COMBAT_QUALITY_GUARD_SEARCH_SUFFIXES,
    **COMBAT_QUALITY_TACTICAL_SEARCH_SUFFIXES,
}


class EpisodeMetricsMixin:
    """Search-stat compaction and episode-level diagnostic metric helpers."""

    @staticmethod
    def _boss_episode_lucky_text_match(value: Any) -> bool:
        """Return True when a potion/action/debug row looks like Lucky Tonic.

        Boss-level TensorBoard metrics are the first place we look during live
        full-run monitoring.  They should not depend only on the explicit
        ``lucky_*`` metadata flags, because older bridge/runtime paths can leave
        those flags unset while the final death frame still contains the actual
        Lucky potion object.
        """

        try:
            if isinstance(value, (dict, list, tuple)):
                import json

                text = json.dumps(value, ensure_ascii=False, default=str)
            else:
                text = str(value or "")
        except Exception:
            text = str(value or "")
        text_u = text.upper()
        return bool(
            "LUCKY_TONIC" in text_u
            or "LUCKY TONIC" in text_u
            or "POTION.LUCK" in text_u
            or "FORTUNE" in text_u
            or "幸运药剂" in text
            or "幸运补剂" in text
            or "幸运药" in text
            or "幸运补" in text
            or "幸運" in text
        )

    @classmethod
    def _boss_episode_lucky_count(cls, values: Any) -> int:
        if values is None:
            return 0
        if isinstance(values, dict):
            return 1 if cls._boss_episode_lucky_text_match(values) else 0
        if isinstance(values, (list, tuple)):
            return sum(1 for item in values if cls._boss_episode_lucky_text_match(item))
        return 1 if cls._boss_episode_lucky_text_match(values) else 0

    @classmethod
    def _boss_episode_lucky_used_in_rows(cls, values: Any) -> bool:
        rows = values if isinstance(values, (list, tuple)) else ([values] if values is not None else [])
        for row in rows:
            if not cls._boss_episode_lucky_text_match(row):
                continue
            if isinstance(row, dict):
                action_text = " ".join(
                    str(row.get(key) or "")
                    for key in ("action_id", "action_type", "family", "kind", "event")
                ).lower()
                if "use_potion" in action_text or "used" in action_text or "potion_used" in action_text:
                    return True
            else:
                text = str(row).lower()
                if "use_potion" in text or "used" in text or "potion_used" in text:
                    return True
        return False

    @staticmethod
    def _compact_search_stats(search_stats: dict[str, Any]) -> dict[str, Any]:
        if not search_stats:
            return {}
        compact: dict[str, Any] = {}
        for key in (
            "num_simulations",
            "root_candidates",
            "root_selectable_children_mean",
            "mean_expanded_children",
            "mean_predicted_legal_count",
            "mean_surface_keep_count",
            "max_search_depth",
            "mean_leaf_depth",
            "mean_concrete_leaf_depth",
            "depth_ge_2_rate",
            "depth_ge_3_rate",
            "root_top1_visit_share",
            "root_visit_entropy",
            "action_hard_guard_dispatch_error",
            "end_turn_bias_applied",
            "end_turn_guard_applied",
            "end_turn_guard_forced_alternative",
            "zero_energy_x_cost_guard_applied",
            "zero_energy_x_cost_guard_forced_alternative",
            "objective_prior_applied",
            "objective_weight_survival",
            "objective_weight_hp",
            "objective_weight_build",
            "objective_weight_resource",
            "root_objective_value",
            "root_bias_scale",
            "root_bias_nonzero",
            "root_bias_abs_mean",
            "root_bias_max_abs",
            "root_bias_changed_top1",
            "root_bias_selected_action_delta",
            "root_bias_suppressed_by_gate",
            "root_bias_scale_effective",
            "semantic_switch_depth",
            "semantic_rollout_enabled",
            "semantic_expansion_rate",
            "semantic_switch_rate",
            "semantic_chain_steps_mean",
            "semantic_drill_rate",
            "combat_grounded_root_enabled",
            "q_value_ucb_enabled",
            "search_mode_direct_policy",
            "search_mode_direct_rollout_planner",
            "direct_rollout_q_mean",
            "direct_rollout_objective_q_mean",
            "direct_rollout_risk_q_mean",
            "direct_rollout_uncertainty_mean",
            "direct_rollout_uncertainty_bias_abs_mean",
            "direct_rollout_surprise_mean",
            "direct_rollout_surface_entropy_mean",
            "direct_rollout_latent_drift_mean",
            "direct_rollout_branch_disagreement_mean",
            "direct_rollout_steps_used",
            "direct_rollout_branch_count_mean",
            "direct_rollout_root_valid_count",
            "direct_rollout_root_bucket_size",
            "direct_rollout_bucket_padding_ratio",
            "direct_rollout_max_branch_bucket_size",
            "direct_rollout_branch_padding_ratio",
            "combat_quality_bias_applied",
            "combat_quality_bias_abs_mean",
            "combat_quality_wasteful_end_turn_bias_applied",
            "combat_quality_wasteful_end_turn_available",
            "combat_quality_wasteful_end_turn_selected",
            "combat_quality_true_wasteful_end_turn_available",
            "combat_quality_true_wasteful_end_turn_selected",
            "combat_quality_strategic_defer_available",
            "combat_quality_strategic_defer_end_turn_selected",
            "combat_quality_end_turn_penalty_max",
            "combat_quality_energy",
            "combat_quality_positive_action_count",
            "combat_quality_mandatory_positive_action_count",
            "combat_quality_strategic_skip_candidate_count",
            "combat_quality_refund_no_followup_available",
            "combat_quality_refund_no_followup_selected",
            "combat_quality_refund_no_followup_with_progress_selected",
            "combat_quality_refund_no_followup_progress_alternative_selected",
            "combat_quality_refund_no_followup_no_alternative_selected",
            "combat_quality_refund_no_followup_progress_alternative_count",
            "combat_quality_refund_no_followup_guard_available",
            "combat_quality_refund_no_followup_guard_applied",
            "combat_quality_refund_no_followup_guard_override",
            "combat_quality_refund_no_followup_guard_no_alternative",
            "combat_quality_refund_no_followup_guard_end_turn_fallback",
            "combat_quality_refund_no_followup_guard_lethal_exemption",
            "combat_quality_refund_no_followup_guard_candidate_count",
            "combat_quality_refund_no_followup_guard_lethal_candidate",
            "combat_quality_strategic_skip_selected",
            "combat_quality_enchantment_seen",
            "combat_quality_affliction_seen",
            "combat_quality_urgent_positive_action_count",
            "combat_quality_deferable_positive_action_count",
            "combat_quality_deferable_exhaust_card_count",
            "combat_quality_energy_gain_without_followup_count",
            "combat_quality_typed_followup_missing_count",
            "combat_quality_typed_future_penalty_count",
            "combat_quality_typed_no_draw_count",
            "combat_quality_typed_card_state_mutation_count",
            "combat_quality_card_block_waste_count",
            "combat_quality_card_pure_block_count",
            "combat_quality_card_no_damage_pressure_count",
            "combat_quality_card_block_waste_bias_count",
            "combat_quality_card_block_waste_bias_min",
            "combat_quality_card_block_waste_hard_bias_applied",
            "combat_quality_card_block_waste_progress_alternative",
            "combat_quality_card_block_waste_progress_bonus_count",
            "combat_quality_card_block_waste_progress_bonus_max",
            "combat_quality_card_block_waste_selected",
            "combat_quality_card_block_waste_with_progress_selected",
            "combat_quality_card_pure_block_selected",
            "combat_quality_bad_pure_block_selected",
            "combat_quality_insufficient_block_selected",
            "combat_quality_pure_block_survival_justified_selected",
            "combat_quality_pure_block_progress_alternative_selected",
            "combat_quality_pure_block_no_alternative_selected",
            "combat_quality_pure_block_low_value_pressure_selected",
            "combat_quality_card_no_damage_pressure_selected",
            "combat_quality_card_no_damage_pressure_with_progress_selected",
            "combat_quality_setup_followup_dependent_count",
            "combat_quality_setup_followup_available_count",
            "combat_quality_potion_available_count",
            "combat_quality_potion_urgent_count",
            "combat_quality_potion_low_urgency_count",
            "combat_quality_potion_save_recommended_count",
            "combat_quality_potion_no_followup_count",
            "combat_quality_potion_lethal_count",
            "combat_quality_potion_prevent_lethal_count",
            "combat_quality_potion_mechanism_count",
            "combat_quality_potion_overkill_count",
            "combat_quality_potion_block_waste_count",
            "combat_quality_potion_use_quality_mean",
            "combat_quality_potion_waste_risk_mean",
            "combat_quality_potion_save_value_mean",
            "combat_quality_potion_hand_context_good_count",
            "combat_quality_potion_hand_context_bad_count",
            "combat_quality_potion_long_term_count",
            "combat_quality_potion_requires_followup_count",
            "combat_quality_bad_potion_bias_count",
            "combat_quality_bad_potion_bias_min",
            "combat_quality_potion_selected",
            "combat_quality_potion_selected_when_available",
            "combat_quality_potion_high_urgency_selected",
            "combat_quality_potion_low_urgency_selected",
            "combat_quality_potion_save_recommended_selected",
            "combat_quality_potion_no_followup_selected",
            "combat_quality_potion_lethal_selected",
            "combat_quality_potion_prevent_lethal_selected",
            "combat_quality_potion_mechanism_selected",
            "combat_quality_potion_overkill_selected",
            "combat_quality_potion_block_waste_selected",
            "combat_quality_potion_use_quality_selected",
            "combat_quality_potion_waste_risk_selected",
            "combat_quality_kaiser_back_attack_risk",
            "combat_quality_kaiser_defense_candidate_count",
            "combat_quality_kaiser_facing_change_candidate_count",
            "combat_quality_kaiser_pressure_candidate_count",
            "combat_quality_kaiser_facing_change_selected",
            "combat_quality_kaiser_pressure_selected",
            "combat_quality_kaiser_risky_end_turn_selected",
            # P0-3 (recovery 2026-05-06): Kaiser facing change hard guard
            # observability. Whitelisted here so the per-step values
            # persist into ``trajectory.steps[*]["search_stats"]`` and
            # subsequently into the boss_combat aggregation.
            "combat_quality_kaiser_facing_guard_available",
            "combat_quality_kaiser_facing_guard_applied",
            "combat_quality_kaiser_facing_guard_override",
            "combat_quality_kaiser_facing_guard_lethal_exemption",
            "combat_quality_kaiser_nonfacing_nonlethal_selected",
            "combat_quality_kaiser_end_turn_under_risk_with_candidate",
            # P0-4 (recovery 2026-05-06): Insatiable Frantic Escape hard
            # force observability — same persistence reason as P0-3.
            "combat_quality_insatiable_escape_force_available",
            "combat_quality_insatiable_escape_force_applied",
            "combat_quality_insatiable_escape_force_override",
            "combat_quality_insatiable_escape_force_lethal_exemption",
            "combat_quality_insatiable_non_escape_at1_blocked",
            # P1-2 (recovery 2026-05-07): X-cost zero-energy hard invalid
            # observability.
            "combat_quality_x_cost_zero_guard_available",
            "combat_quality_x_cost_zero_guard_applied",
            "combat_quality_x_cost_zero_guard_override",
            "combat_quality_x_cost_zero_guard_no_alternative",
            "combat_quality_x_cost_zero_guard_end_turn_fallback",
            # P1-3 (recovery 2026-05-07): HP-cost survival-margin hard guard.
            "combat_quality_hp_cost_margin_guard_available",
            "combat_quality_hp_cost_margin_guard_applied",
            "combat_quality_hp_cost_margin_guard_override",
            "combat_quality_hp_cost_margin_guard_lethal_exemption",
            "combat_quality_hp_cost_margin_guard_no_alternative",
            # P1-1 (recovery 2026-05-07): Potion bad-use hard guard.
            "combat_quality_potion_bad_guard_available",
            "combat_quality_potion_bad_guard_applied",
            "combat_quality_potion_bad_guard_override",
            "combat_quality_potion_bad_guard_lethal_exemption",
            "combat_quality_potion_bad_guard_no_alternative",
            "combat_quality_potion_bad_guard_invalid_obs",
            "combat_quality_potion_bad_guard_critical_hp_survival_skip",
            "combat_quality_potion_bad_guard_critical_hp_idle_waste_not_skipped",
            "combat_quality_potion_bad_guard_end_turn_fallback_blocked_unsafe",
            "combat_quality_potion_bad_guard_forced_end_turn_hopeless",
            "combat_quality_potion_bad_guard_boss_race_skip",
            "combat_quality_potion_bad_guard_late_normal_race_skip",
            "combat_quality_potion_bad_hopeless_guard_late_normal_race_skip",
            "combat_quality_potion_bad_hopeless_guard_boss_survival_skip",
            "combat_quality_potion_bad_hopeless_guard_lagavulin_setup_skip",
            # Act1 recovery 2026-05-10: forced potion-overflow discard
            # priority guard.  Persist the signal so diagnostics can tell
            # whether late hallway deaths were preceded by throwing away a
            # survival potion.
            "combat_quality_potion_discard_guard_available",
            "combat_quality_potion_discard_guard_applied",
            "combat_quality_potion_discard_guard_override",
            "combat_quality_potion_discard_guard_candidate_count",
            "combat_quality_potion_discard_guard_saved_survival",
            "combat_quality_elite_boss_lethal_end_turn_guard_available",
            "combat_quality_elite_boss_lethal_end_turn_guard_applied",
            "combat_quality_elite_boss_lethal_end_turn_guard_override",
            # Global no-pressure EndTurn -> meaningful damage/progress guard.
            # This is not boss-only, but the legacy boss_combat aggregate is
            # still used by existing dashboards; keep raw keys persisted here.
            "combat_quality_meaningful_damage_endturn_guard_available",
            "combat_quality_meaningful_damage_endturn_guard_applied",
            "combat_quality_meaningful_damage_endturn_guard_override",
            "combat_quality_meaningful_damage_endturn_guard_candidate_count",
            "combat_quality_meaningful_damage_endturn_guard_no_alternative",
            "combat_quality_meaningful_damage_endturn_guard_lethal_candidate",
            "combat_quality_meaningful_damage_endturn_guard_pressure_skip",
            "combat_quality_no_pressure_block_guard_available",
            "combat_quality_no_pressure_block_guard_applied",
            "combat_quality_no_pressure_block_guard_override",
            "combat_quality_no_pressure_block_guard_candidate_count",
            "combat_quality_no_pressure_block_guard_no_alternative",
            "combat_quality_no_pressure_block_guard_pressure_skip",
            "combat_quality_no_pressure_block_guard_pressure_attack_window",
            "combat_quality_no_pressure_block_guard_low_value_pressure",
            "combat_quality_no_pressure_block_guard_trivial_pressure",
            "combat_quality_no_pressure_block_guard_lethal_candidate",
            "combat_quality_boss_survival_potion_guard_available",
            "combat_quality_boss_survival_potion_guard_applied",
            "combat_quality_boss_survival_potion_guard_override",
            "combat_quality_boss_survival_potion_guard_lethal_exemption",
            "combat_quality_boss_survival_potion_guard_no_alternative",
            "combat_quality_boss_race_potion_guard_available",
            "combat_quality_boss_race_potion_guard_applied",
            "combat_quality_boss_race_potion_guard_override",
            "combat_quality_boss_race_potion_guard_lethal_exemption",
            "combat_quality_boss_race_potion_guard_no_alternative",
            "combat_quality_boss_race_potion_guard_lagavulin_setup_escape",
            "combat_quality_boss_survival_block_guard_available",
            "combat_quality_boss_survival_block_guard_applied",
            "combat_quality_boss_survival_block_guard_override",
            "combat_quality_boss_survival_block_guard_lethal_exemption",
            "combat_quality_boss_survival_block_guard_no_alternative",
            "combat_quality_boss_survival_block_guard_insufficient_candidate",
            # P1-6 (act1 recovery 2026-05-10): late-Act1 normal
            # combat guard persistence.
            "combat_quality_late_normal_lethal_end_turn_guard_available",
            "combat_quality_late_normal_lethal_end_turn_guard_applied",
            "combat_quality_late_normal_lethal_end_turn_guard_override",
            "combat_quality_late_normal_survival_guard_available",
            "combat_quality_late_normal_survival_guard_applied",
            "combat_quality_late_normal_survival_guard_override",
            "combat_quality_late_normal_survival_guard_lethal_exemption",
            "combat_quality_late_normal_survival_guard_no_alternative",
            "combat_quality_late_normal_survival_guard_candidate_count",
            "combat_quality_late_normal_survival_guard_insufficient_candidate",
            "combat_quality_late_normal_race_potion_guard_available",
            "combat_quality_late_normal_race_potion_guard_applied",
            "combat_quality_late_normal_race_potion_guard_override",
            "combat_quality_late_normal_race_potion_guard_lethal_exemption",
            "combat_quality_late_normal_race_potion_guard_no_alternative",
            "combat_quality_late_normal_race_potion_guard_candidate_count",
            # P1-8 (act1 recovery 2026-05-10): high-pressure selected
            # non-EndTurn survival correction.  The guard is created after
            # generic search_stats collection, so whitelist it here as well
            # as in the domain post-append path.
            "combat_quality_survival_non_endturn_guard_available",
            "combat_quality_survival_non_endturn_guard_applied",
            "combat_quality_survival_non_endturn_guard_override",
            "combat_quality_survival_non_endturn_guard_lethal_exemption",
            "combat_quality_survival_non_endturn_guard_progress_exemption",
            "combat_quality_survival_non_endturn_guard_scaling_enemy_exemption",
            "combat_quality_survival_non_endturn_guard_scaling_enemy_candidate",
            "combat_quality_survival_non_endturn_guard_no_alternative",
            "combat_quality_survival_non_endturn_guard_candidate_count",
            "combat_quality_survival_non_endturn_guard_insufficient_candidate",
            "combat_quality_hard_guard_override_any",
            "combat_quality_hard_guard_policy_target_rewrite",
            # P2-2 (recovery 2026-05-07): Selection-loop dead-loop guard.
            "combat_quality_selection_loop_screen_active",
            "combat_quality_selection_loop_detected",
            "combat_quality_selection_loop_applied",
            "combat_quality_selection_loop_auto_confirm",
            "combat_quality_selection_loop_auto_cancel",
            "combat_quality_selection_loop_alt_pick",
            "combat_quality_selection_loop_no_alternative",
            "combat_quality_selection_repeated_same_option",
            # P2-1 (recovery 2026-05-07): runtime card-state presence flags.
            "combat_quality_card_runtime_instance_uuid_present",
            "combat_quality_card_runtime_modified_cost_present",
            "combat_quality_card_runtime_exhaust_flag_present",
            "combat_quality_card_runtime_ethereal_flag_present",
            "combat_quality_card_runtime_retain_flag_present",
            "combat_quality_card_runtime_enchantment_present",
            "combat_quality_card_runtime_replay_flag_present",
            "combat_quality_card_runtime_selection_effect_present",
            "combat_quality_ceremonial_one_card_lock",
            "combat_quality_ceremonial_stun_window",
            "combat_quality_ceremonial_low_impact_count",
            "combat_quality_ceremonial_high_impact_count",
            "combat_quality_kaiser_defense_selected",
            "combat_quality_ceremonial_low_impact_selected",
            "combat_quality_ceremonial_high_impact_selected",
            "combat_quality_insatiable_sandpit_countdown",
            "combat_quality_insatiable_sandpit_active",
            "combat_quality_insatiable_sandpit_lt3",
            "combat_quality_insatiable_sandpit_1",
            "combat_quality_insatiable_frantic_escape_hand_count",
            "combat_quality_insatiable_frantic_escape_draw_count",
            "combat_quality_insatiable_frantic_escape_discard_count",
            "combat_quality_insatiable_frantic_escape_exhaust_count",
            "combat_quality_insatiable_frantic_escape_total_count",
            "combat_quality_insatiable_frantic_escape_candidate_count",
            "combat_quality_insatiable_frantic_escape_available",
            "combat_quality_insatiable_frantic_escape_urgency",
            "combat_quality_insatiable_frantic_escape_bonus_applied",
            "combat_quality_insatiable_non_escape_at1_penalty_count",
            "combat_quality_insatiable_escape_cycle_risk",
            "combat_quality_insatiable_lethal_candidate_count",
            "combat_quality_insatiable_frantic_escape_selected",
            "combat_quality_insatiable_frantic_escape_missed_lt3",
            "combat_quality_insatiable_frantic_escape_missed_at_1",
            "combat_quality_insatiable_non_escape_at_1_selected",
            "combat_quality_playable_action_count",
            "combat_quality_end_turn_severity",
            "combat_quality_x_cost_available_count",
            "combat_quality_x_cost_selected",
            "combat_quality_x_cost_selected_energy",
            "combat_quality_x_cost_zero_energy_selected",
            "combat_quality_zero_energy_x_cost_count",
            "combat_quality_zero_energy_x_cost_selected",
            "combat_quality_x_cost_bad_count",
            "combat_quality_x_cost_effective_energy_sum",
            "combat_quality_x_cost_effective_energy_mean",
            "combat_quality_x_cost_selected_effective_energy",
            "combat_quality_x_cost_has_non_energy_effect_selected",
            "combat_quality_x_cost_bad_selected",
            "combat_quality_bad_end_turn_available",
            "combat_quality_bad_end_turn_selected",
            "combat_quality_forced_end_turn_available",
            "combat_quality_forced_end_turn_selected",
            "combat_quality_end_turn_unknown_selected",
            "combat_quality_end_turn_selected",
        ):
            value = search_stats.get(key)
            if value is None:
                continue
            try:
                compact[key] = float(value)
            except (TypeError, ValueError):
                continue
        return compact

    @staticmethod
    def _normalize_action_values(values: torch.Tensor, action_mask: torch.Tensor) -> torch.Tensor:
        action_mask = action_mask.to(device=values.device, dtype=torch.bool)
        masked_values = torch.where(action_mask, values, torch.zeros_like(values))
        valid_count = action_mask.float().sum(dim=-1, keepdim=True).clamp(min=1.0)
        mean = masked_values.sum(dim=-1, keepdim=True) / valid_count
        centered = torch.where(action_mask, values - mean, torch.zeros_like(values))
        variance = centered.pow(2).sum(dim=-1, keepdim=True) / valid_count
        std = variance.clamp(min=1e-6).sqrt()
        normalized = centered / std
        return torch.where(action_mask, normalized, torch.zeros_like(normalized))

    def record_recent_combat_episode(self, metadata: dict[str, Any] | None) -> dict[int, dict[str, Any]]:
        snapshot = self.recent_combat_monitor.record_episode(metadata)
        if snapshot:
            self.recent_combat_monitor.log_to_writer(self.writer, self.episode_count)
        return snapshot

    def _num_simulations_for_domain(self, decision_domain: str) -> int:
        return int(self.domain_num_simulations.get(decision_domain, self.mcts.num_simulations))

    @staticmethod
    def _parse_progress_act_id(value: Any, run: dict[str, Any] | None = None) -> float:
        """Parse act id for episode progress telemetry.

        EnvV2 now normalizes transition_state.run.act_id, but this parser is
        intentionally duplicated here so replayed/actor payloads and older
        transition states do not collapse live enum strings such as
        ``ACT.UNDERDOCKS`` to 0.  Unknown values remain 0.
        """
        def _from_value(raw: Any) -> float:
            if raw is None or isinstance(raw, bool):
                return 0.0
            if isinstance(raw, (int, float)):
                return float(raw)
            text = str(raw).strip()
            if not text:
                return 0.0
            try:
                return float(text)
            except (TypeError, ValueError):
                pass
            for char in reversed(text):
                if char.isdigit():
                    return float(char)
            upper = text.upper()
            for fragment, act_id in (
                ("UNDERDOCKS", 1.0),
                ("UNDERDOCK", 1.0),
                ("ACT_ONE", 1.0),
                ("ACT_1", 1.0),
                ("ACT1", 1.0),
                ("HIVE", 2.0),
                ("ACT_TWO", 2.0),
                ("ACT_2", 2.0),
                ("ACT2", 2.0),
                ("GLORY", 3.0),
                ("ACT_THREE", 3.0),
                ("ACT_3", 3.0),
                ("ACT3", 3.0),
            ):
                if fragment in upper:
                    return float(act_id)
            return 0.0

        parsed = _from_value(value)
        if parsed > 0.0:
            return parsed
        if not isinstance(run, dict):
            return 0.0
        for key in ("act_id", "act_id_raw", "act", "act_number", "current_act", "act_name"):
            parsed = _from_value(run.get(key))
            if parsed > 0.0:
                return parsed
        for key in ("current_act_index", "act_index"):
            if key not in run:
                continue
            try:
                index = float(run.get(key))
            except (TypeError, ValueError):
                continue
            if index >= 0.0:
                return float(index + 1.0)
        return 0.0

    @staticmethod
    def _episode_progress_snapshot(info: dict[str, Any] | None) -> dict[str, Any]:
        transition_state = info.get("transition_state") if isinstance(info, dict) else None
        run = transition_state.get("run") if isinstance(transition_state, dict) else None
        player = transition_state.get("player") if isinstance(transition_state, dict) else None
        if not isinstance(run, dict):
            return {}
        room_type = str(run.get("room_type") or run.get("state_type") or "")
        room_model = str(
            run.get("room_model")
            or run.get("encounter_id")
            or run.get("encounter")
            or run.get("room_encounter")
            or run.get("room")
            or ""
        )
        try:
            floor = float(run.get("floor", run.get("total_floor")) or 0.0)
        except (TypeError, ValueError):
            floor = 0.0
        act_id = EpisodeMetricsMixin._parse_progress_act_id(run.get("act_id"), run)
        try:
            hp = float(player.get("hp") or 0.0) if isinstance(player, dict) else 0.0
        except (TypeError, ValueError):
            hp = 0.0
        try:
            max_hp = float(player.get("max_hp") or 0.0) if isinstance(player, dict) else 0.0
        except (TypeError, ValueError):
            max_hp = 0.0
        return {
            "floor": floor,
            "act_id": act_id,
            "act_id_raw": run.get("act_id_raw", run.get("act_id")),
            "current_act_index": run.get("current_act_index"),
            "room_type": room_type,
            "room_type_lower": room_type.lower(),
            "room_model": room_model,
            "room_model_lower": room_model.lower(),
            "encounter_id": room_model,
            "encounter_id_upper": room_model.upper(),
            "hp": hp,
            "max_hp": max_hp,
        }

    @staticmethod
    def _safe_float(value: Any, default: float = 0.0) -> float:
        try:
            return float(value)
        except (TypeError, ValueError):
            return float(default)

    @staticmethod
    def _safe_rate(numerator: float, denominator: float) -> float:
        return float(numerator) / float(denominator) if float(denominator) > 0.0 else 0.0

    @staticmethod
    def _potion_count_from_info(info: dict[str, Any] | None) -> int:
        transition_state = info.get("transition_state") if isinstance(info, dict) else None
        player = transition_state.get("player") if isinstance(transition_state, dict) else None
        potions = player.get("potions") if isinstance(player, dict) else None
        if not isinstance(potions, list):
            return 0
        count = 0
        # IMPORTANT: STS2 bridge serializes empty potion slots as
        # {title:"[empty]", id:null, ...} — note the BRACKETS.  An earlier
        # exclusion set of {"empty","none","null"} did NOT match the bracket
        # form, so every empty slot was being counted as a usable potion.
        # That made `boss_combat/<boss>/potion_unused_on_death_rate` track
        # the boss loss rate (70-85%) instead of actual leftover potions, and
        # repeatedly misled the curriculum diagnostic.
        EMPTY_TITLES = {"empty", "[empty]", "none", "null", ""}
        for potion in potions:
            if isinstance(potion, dict):
                if bool(potion.get("empty")):
                    continue
                # A used potion can remain in PotionSlots briefly as queued or
                # removed-from-state.  Do not count those as "unused on death".
                if bool(potion.get("is_queued")) or bool(potion.get("has_been_removed_from_state")):
                    continue
                if potion.get("is_usable") is False:
                    continue
                title = str(potion.get("title") or potion.get("id") or "").strip().lower()
            else:
                title = str(potion or "").strip().lower()
            if title and title not in EMPTY_TITLES:
                count += 1
        return count

    @classmethod
    def _boss_entry_snapshot_from_episode(
        cls,
        initial_info: dict[str, Any] | None,
        progress_snapshots: list[dict[str, Any]],
        *,
        encounter_tier: str,
    ) -> dict[str, float]:
        candidates: list[dict[str, Any]] = []
        if str(encounter_tier or "").strip().lower() == "boss":
            first = cls._episode_progress_snapshot(initial_info)
            first_text = f"{first.get('room_type_lower', '')} {first.get('room_model_lower', '')}" if first else ""
            if first and "boss" in first_text:
                candidates.append(first)
        candidates.extend(
            snapshot for snapshot in progress_snapshots
            if "boss" in str(snapshot.get("room_type_lower") or "")
            or "boss" in str(snapshot.get("room_model_lower") or "")
        )
        if not candidates:
            return {}
        entry = candidates[0]
        hp = cls._safe_float(entry.get("hp"))
        max_hp = cls._safe_float(entry.get("max_hp"))
        return {
            "hp": hp,
            "max_hp": max_hp,
            "hp_ratio": hp / max_hp if max_hp > 0.0 else 0.0,
        }

    @staticmethod
    def _step_family(step: dict[str, Any]) -> str:
        family = str(step.get("action_family") or "").strip().lower()
        if family:
            return family
        info = step.get("action_info") if isinstance(step.get("action_info"), dict) else {}
        semantic = info.get("semantic") if isinstance(info.get("semantic"), dict) else {}
        return str(semantic.get("family") or "").strip().lower()

    @staticmethod
    def _step_is_combat(step: dict[str, Any]) -> bool:
        domain = str(step.get("decision_domain") or step.get("semantic_domain") or "").strip().lower()
        family = EpisodeMetricsMixin._step_family(step)
        phase = str(step.get("phase") or "").strip().lower()
        surface = str(step.get("surface") or "").strip().lower()
        return (
            domain == "combat"
            or phase in {"combat", "actions", "card_selection"}
            or surface == "combat"
            or family in {"play_card", "use_potion", "discard_potion", "end_turn", "combat_select"}
        )

    def _collect_search_values_from_steps(
        self,
        steps: list[dict[str, Any]],
        keys: Any,
    ) -> dict[str, list[float]]:
        values: dict[str, list[float]] = defaultdict(list)
        for step in steps:
            search_stats = step.get("search_stats") if isinstance(step.get("search_stats"), dict) else {}
            for key in keys:
                if key in search_stats:
                    values[key].append(self._safe_float(search_stats.get(key)))
        return values

    def _infer_step_encounter_tier(self, step: dict[str, Any]) -> str:
        tier = str(step.get("encounter_tier") or "").strip().lower()
        if tier in {"weak", "normal", "elite", "boss"}:
            return tier
        encounter_id = str(step.get("encounter_id") or "").strip()
        room_type = str(step.get("room_type") or "").strip()
        inferred = str(infer_encounter_tier(encounter_id, room_type=room_type) or "").strip().lower()
        return inferred if inferred in {"weak", "normal", "elite", "boss"} else ""

    def _combat_quality_family_metrics(
        self,
        steps: list[dict[str, Any]],
        *,
        prefix: str,
    ) -> dict[str, float]:
        normalized_prefix = str(prefix or "")
        if normalized_prefix and not normalized_prefix.endswith("/"):
            normalized_prefix = f"{normalized_prefix}/"
        total_steps = len(steps)
        family_counts: Counter[str] = Counter(self._step_family(step) or "unknown" for step in steps)
        return {
            f"{normalized_prefix}decision_count": float(total_steps),
            f"{normalized_prefix}family_end_turn_rate": self._safe_rate(family_counts.get("end_turn", 0), total_steps),
            f"{normalized_prefix}family_play_card_rate": self._safe_rate(family_counts.get("play_card", 0), total_steps),
            f"{normalized_prefix}family_potion_rate": self._safe_rate(
                family_counts.get("use_potion", 0) + family_counts.get("discard_potion", 0),
                total_steps,
            ),
            f"{normalized_prefix}wasteful_end_turn_rate": self._safe_rate(
                sum(1 for step in steps if bool(step.get("wasteful_end_turn"))),
                total_steps,
            ),
        }

    @staticmethod
    def _combat_quality_guard_activity_metrics(
        search_values: dict[str, list[float]],
        *,
        prefix: str,
    ) -> dict[str, float]:
        """Aggregate global/tier hard-guard activity from per-decision stats."""

        normalized_prefix = str(prefix or "")
        if normalized_prefix and not normalized_prefix.endswith("/"):
            normalized_prefix = f"{normalized_prefix}/"

        def mean(key: str) -> float:
            return float(np.mean(search_values.get(key, [0.0])))

        return {
            f"{normalized_prefix}meaningful_damage_endturn_guard_available_rate": mean("combat_quality_meaningful_damage_endturn_guard_available"),
            f"{normalized_prefix}meaningful_damage_endturn_guard_applied_rate": mean("combat_quality_meaningful_damage_endturn_guard_applied"),
            f"{normalized_prefix}meaningful_damage_endturn_guard_override_rate": mean("combat_quality_meaningful_damage_endturn_guard_override"),
            f"{normalized_prefix}meaningful_damage_endturn_guard_candidate_count_mean": mean("combat_quality_meaningful_damage_endturn_guard_candidate_count"),
            f"{normalized_prefix}meaningful_damage_endturn_guard_no_alternative_rate": mean("combat_quality_meaningful_damage_endturn_guard_no_alternative"),
            f"{normalized_prefix}meaningful_damage_endturn_guard_lethal_candidate_rate": mean("combat_quality_meaningful_damage_endturn_guard_lethal_candidate"),
            f"{normalized_prefix}meaningful_damage_endturn_guard_pressure_skip_rate": mean("combat_quality_meaningful_damage_endturn_guard_pressure_skip"),
            f"{normalized_prefix}no_pressure_block_guard_available_rate": mean("combat_quality_no_pressure_block_guard_available"),
            f"{normalized_prefix}no_pressure_block_guard_applied_rate": mean("combat_quality_no_pressure_block_guard_applied"),
            f"{normalized_prefix}no_pressure_block_guard_override_rate": mean("combat_quality_no_pressure_block_guard_override"),
            f"{normalized_prefix}no_pressure_block_guard_candidate_count_mean": mean("combat_quality_no_pressure_block_guard_candidate_count"),
            f"{normalized_prefix}no_pressure_block_guard_no_alternative_rate": mean("combat_quality_no_pressure_block_guard_no_alternative"),
            f"{normalized_prefix}no_pressure_block_guard_pressure_skip_rate": mean("combat_quality_no_pressure_block_guard_pressure_skip"),
            f"{normalized_prefix}no_pressure_block_guard_pressure_attack_window_rate": mean("combat_quality_no_pressure_block_guard_pressure_attack_window"),
            f"{normalized_prefix}no_pressure_block_guard_low_value_pressure_rate": mean("combat_quality_no_pressure_block_guard_low_value_pressure"),
            f"{normalized_prefix}no_pressure_block_guard_trivial_pressure_rate": mean("combat_quality_no_pressure_block_guard_trivial_pressure"),
            f"{normalized_prefix}no_pressure_block_guard_lethal_candidate_rate": mean("combat_quality_no_pressure_block_guard_lethal_candidate"),
            f"{normalized_prefix}boss_survival_block_guard_insufficient_candidate_rate": mean("combat_quality_boss_survival_block_guard_insufficient_candidate"),
            f"{normalized_prefix}late_normal_survival_guard_insufficient_candidate_rate": mean("combat_quality_late_normal_survival_guard_insufficient_candidate"),
            f"{normalized_prefix}survival_non_endturn_guard_insufficient_candidate_rate": mean("combat_quality_survival_non_endturn_guard_insufficient_candidate"),
        }

    def _emit_combat_quality_episode_diagnostics(self, trajectory: GameTrajectory) -> dict[str, Any]:
        """Emit global and encounter-tier combat-quality diagnostics.

        Boss-specific metrics already have a richer dedicated emitter.  This
        method fills the previous blind spot: weak/normal/elite hallway combat
        and all-combat aggregate trends for useless block / no-pressure action
        choices.
        """

        steps = [step for step in trajectory.steps if isinstance(step, dict) and self._step_is_combat(step)]
        if not steps:
            return {}

        # Combat sandbox regressions currently hide in normal/hard-normal
        # losses, while the older death-slice hook only ran from the boss
        # diagnostic path.  Emit bounded per-encounter death slices for any
        # losing combat episode so hard hallway blockers (e.g. Ovicopter or
        # Slumbering Beetle) can be inspected without restarting with a custom
        # logger.  The dump helper is best-effort and capped per encounter.
        metadata = trajectory.metadata if isinstance(getattr(trajectory, "metadata", None), dict) else {}
        loss = float(metadata.get("death_floor", 0.0) or 0.0) > 0.0
        if loss and hasattr(self, "_dump_death_slice"):
            last_combat_step = steps[-1] if steps else {}
            encounter_id = str(
                last_combat_step.get("encounter_id")
                or metadata.get("encounter_id")
                or metadata.get("final_room_model")
                or ""
            ).strip()
            encounter_tier = str(
                last_combat_step.get("encounter_tier")
                or metadata.get("encounter_tier")
                or infer_encounter_tier(encounter_id)
                or ""
            ).strip().lower()
            try:
                self._dump_death_slice(
                    trajectory=trajectory,
                    encounter_id=encounter_id,
                    encounter_tier=encounter_tier,
                    loss=True,
                    steps=steps,
                    watch_only=False,
                    tail_len=8,
                    reason="combat_quality_episode_loss",
                )
            except Exception:
                pass

        metrics: dict[str, float] = {}
        all_values = self._collect_search_values_from_steps(
            steps,
            COMBAT_QUALITY_EPISODE_SEARCH_SUFFIXES,
        )
        metrics.update(card_block_waste_metrics(all_values, prefix="combat_quality/"))
        metrics.update(self._combat_quality_guard_activity_metrics(all_values, prefix="combat_quality/"))
        metrics.update(combat_tactical_issue_metrics(all_values, prefix="combat_quality/"))
        # Mirror hard-guard activity under a domain-neutral namespace for
        # dashboards and live smoke checks.  Previously ``combat_guards/*`` was
        # only emitted from the boss-combat diagnostic path, so an early Act 1
        # run that died before the boss could look as if the new guards were not
        # wired even when the all-combat ``combat_quality/*`` tags were present.
        metrics.update(self._combat_quality_guard_activity_metrics(all_values, prefix="combat_guards/"))
        metrics.update(self._combat_quality_family_metrics(steps, prefix="combat_quality/"))

        for tier, prefix in (
            ("weak", "weak_combat/"),
            ("normal", "normal_combat/"),
            ("elite", "elite_combat/"),
        ):
            tier_steps = [step for step in steps if self._infer_step_encounter_tier(step) == tier]
            if not tier_steps:
                continue
            tier_values = self._collect_search_values_from_steps(
                tier_steps,
                COMBAT_QUALITY_EPISODE_SEARCH_SUFFIXES,
            )
            metrics.update(card_block_waste_metrics(tier_values, prefix=prefix))
            metrics.update(self._combat_quality_guard_activity_metrics(tier_values, prefix=prefix))
            metrics.update(combat_tactical_issue_metrics(tier_values, prefix=prefix))
            metrics.update(self._combat_quality_family_metrics(tier_steps, prefix=prefix))

        for tag, value in metrics.items():
            self.writer.add_scalar(tag, float(value), self.episode_count)
        return metrics

    def _emit_boss_episode_diagnostics(
        self,
        trajectory: GameTrajectory,
        *,
        boss_entry: dict[str, float] | None,
        final_potion_count: int,
    ) -> dict[str, Any]:
        metadata = trajectory.metadata if isinstance(trajectory.metadata, dict) else {}
        encounter_id = str(
            metadata.get("boss_encounter_id")
            or metadata.get("final_room_model")
            or metadata.get("encounter_id")
            or ""
        ).strip()
        if not encounter_id:
            progress_meta = metadata.get("progress_snapshots")
            for snapshot in (progress_meta if isinstance(progress_meta, list) else []):
                if not isinstance(snapshot, dict):
                    continue
                candidate = str(snapshot.get("room_model") or snapshot.get("encounter_id") or "").strip()
                text = f"{snapshot.get('room_type_lower', '')} {snapshot.get('room_model_lower', '')}".lower()
                if candidate and "boss" in text:
                    encounter_id = candidate
                    break
        encounter_tier = str(metadata.get("encounter_tier") or infer_encounter_tier(encounter_id)).strip().lower()
        boss_episode = encounter_tier == "boss" or int(metadata.get("boss_rooms_seen", 0) or 0) > 0
        if boss_episode and encounter_tier != "boss" and encounter_id:
            encounter_tier = str(infer_encounter_tier(encounter_id)).strip().lower()
        if boss_episode and encounter_tier != "boss" and int(metadata.get("boss_rooms_seen", 0) or 0) > 0:
            encounter_tier = "boss"
        if not boss_episode:
            return {}

        all_steps = [step for step in trajectory.steps if isinstance(step, dict) and self._step_is_combat(step)]
        boss_steps = [step for step in all_steps if self._infer_step_encounter_tier(step) == "boss"]
        steps = boss_steps if boss_steps else all_steps
        total_steps = len(steps)
        family_counts: Counter[str] = Counter(self._step_family(step) or "unknown" for step in steps)
        end_turn_count = int(family_counts.get("end_turn", 0))
        use_potion_count = int(family_counts.get("use_potion", 0) + family_counts.get("discard_potion", 0))
        wasteful_end_turn_count = int(sum(1 for step in steps if bool(step.get("wasteful_end_turn"))))

        search_values: dict[str, list[float]] = defaultdict(list)
        for step in steps:
            search_stats = step.get("search_stats") if isinstance(step.get("search_stats"), dict) else {}
            for key in (
                "search_mode_direct_policy",
                "search_mode_direct_rollout_planner",
                "num_simulations",
                "direct_rollout_objective_q_mean",
                "direct_rollout_uncertainty_mean",
                "direct_rollout_uncertainty_bias_abs_mean",
                "direct_rollout_branch_disagreement_mean",
                "direct_rollout_risk_q_mean",
                "root_bias_scale",
                "root_bias_nonzero",
                "root_bias_abs_mean",
                "root_bias_max_abs",
                "root_bias_changed_top1",
                "root_bias_selected_action_delta",
                "root_bias_suppressed_by_gate",
                "root_bias_scale_effective",
                "mean_predicted_legal_count",
                "combat_quality_bias_applied",
                "combat_quality_bias_abs_mean",
                "combat_quality_wasteful_end_turn_bias_applied",
                "combat_quality_wasteful_end_turn_available",
                "combat_quality_wasteful_end_turn_selected",
                "combat_quality_true_wasteful_end_turn_available",
                "combat_quality_true_wasteful_end_turn_selected",
                "combat_quality_strategic_defer_available",
                "combat_quality_strategic_defer_end_turn_selected",
                "combat_quality_urgent_positive_action_count",
                "combat_quality_deferable_positive_action_count",
                "combat_quality_deferable_exhaust_card_count",
                "combat_quality_energy_gain_without_followup_count",
                "combat_quality_typed_followup_missing_count",
                "combat_quality_typed_future_penalty_count",
                "combat_quality_typed_no_draw_count",
                "combat_quality_typed_card_state_mutation_count",
                "combat_quality_card_block_waste_count",
                "combat_quality_card_pure_block_count",
                "combat_quality_card_no_damage_pressure_count",
                "combat_quality_card_block_waste_bias_count",
                "combat_quality_card_block_waste_bias_min",
                "combat_quality_card_block_waste_hard_bias_applied",
                "combat_quality_card_block_waste_progress_alternative",
                "combat_quality_card_block_waste_progress_bonus_count",
                "combat_quality_card_block_waste_progress_bonus_max",
                "combat_quality_card_block_waste_selected",
                "combat_quality_card_block_waste_with_progress_selected",
                "combat_quality_card_pure_block_selected",
                "combat_quality_bad_pure_block_selected",
                "combat_quality_insufficient_block_selected",
                "combat_quality_pure_block_survival_justified_selected",
                "combat_quality_pure_block_progress_alternative_selected",
                "combat_quality_pure_block_no_alternative_selected",
                "combat_quality_pure_block_low_value_pressure_selected",
                "combat_quality_card_no_damage_pressure_selected",
                "combat_quality_card_no_damage_pressure_with_progress_selected",
                "combat_quality_setup_followup_dependent_count",
                "combat_quality_setup_followup_available_count",
                "combat_quality_potion_available_count",
                "combat_quality_potion_urgent_count",
                "combat_quality_potion_low_urgency_count",
                "combat_quality_potion_save_recommended_count",
                "combat_quality_potion_no_followup_count",
                "combat_quality_potion_lethal_count",
                "combat_quality_potion_prevent_lethal_count",
                "combat_quality_potion_mechanism_count",
                "combat_quality_potion_overkill_count",
                "combat_quality_potion_block_waste_count",
                "combat_quality_potion_use_quality_mean",
                "combat_quality_potion_waste_risk_mean",
                "combat_quality_potion_save_value_mean",
                "combat_quality_potion_hand_context_good_count",
                "combat_quality_potion_hand_context_bad_count",
                "combat_quality_potion_long_term_count",
                "combat_quality_potion_requires_followup_count",
                "combat_quality_bad_potion_bias_count",
                "combat_quality_bad_potion_bias_min",
                "combat_quality_potion_selected",
                "combat_quality_potion_selected_when_available",
                "combat_quality_potion_high_urgency_selected",
                "combat_quality_potion_low_urgency_selected",
                "combat_quality_potion_save_recommended_selected",
                "combat_quality_potion_no_followup_selected",
                "combat_quality_potion_lethal_selected",
                "combat_quality_potion_prevent_lethal_selected",
                "combat_quality_potion_mechanism_selected",
                "combat_quality_potion_overkill_selected",
                "combat_quality_potion_block_waste_selected",
                "combat_quality_potion_use_quality_selected",
                "combat_quality_potion_waste_risk_selected",
                "combat_quality_x_cost_available_count",
                "combat_quality_x_cost_selected",
                "combat_quality_x_cost_selected_energy",
                "combat_quality_x_cost_zero_energy_selected",
                "combat_quality_zero_energy_x_cost_count",
                "combat_quality_zero_energy_x_cost_selected",
                "combat_quality_x_cost_bad_count",
                "combat_quality_x_cost_effective_energy_sum",
                "combat_quality_x_cost_effective_energy_mean",
                "combat_quality_x_cost_selected_effective_energy",
                "combat_quality_x_cost_has_non_energy_effect_selected",
                "combat_quality_x_cost_bad_selected",
                "combat_quality_bad_end_turn_available",
                "combat_quality_bad_end_turn_selected",
                "combat_quality_forced_end_turn_available",
                "combat_quality_forced_end_turn_selected",
                "combat_quality_end_turn_unknown_selected",
                "combat_quality_end_turn_selected",
                "combat_quality_energy",
                "combat_quality_positive_action_count",
                "combat_quality_mandatory_positive_action_count",
                "combat_quality_strategic_skip_candidate_count",
                "combat_quality_refund_no_followup_available",
                "combat_quality_refund_no_followup_selected",
                "combat_quality_refund_no_followup_with_progress_selected",
                "combat_quality_refund_no_followup_progress_alternative_selected",
                "combat_quality_refund_no_followup_no_alternative_selected",
                "combat_quality_refund_no_followup_progress_alternative_count",
                "combat_quality_refund_no_followup_guard_available",
                "combat_quality_refund_no_followup_guard_applied",
                "combat_quality_refund_no_followup_guard_override",
                "combat_quality_refund_no_followup_guard_no_alternative",
                "combat_quality_refund_no_followup_guard_end_turn_fallback",
                "combat_quality_refund_no_followup_guard_lethal_exemption",
                "combat_quality_refund_no_followup_guard_candidate_count",
                "combat_quality_refund_no_followup_guard_lethal_candidate",
                "combat_quality_strategic_skip_selected",
                "combat_quality_enchantment_seen",
                "combat_quality_affliction_seen",
                "combat_quality_kaiser_back_attack_risk",
                "combat_quality_kaiser_defense_candidate_count",
                "combat_quality_kaiser_facing_change_candidate_count",
                "combat_quality_kaiser_pressure_candidate_count",
                "combat_quality_kaiser_facing_change_selected",
                "combat_quality_kaiser_pressure_selected",
                "combat_quality_kaiser_risky_end_turn_selected",
                "combat_quality_ceremonial_one_card_lock",
                "combat_quality_ceremonial_stun_window",
                "combat_quality_ceremonial_low_impact_count",
                "combat_quality_ceremonial_high_impact_count",
                "combat_quality_kaiser_defense_selected",
                "combat_quality_ceremonial_low_impact_selected",
                "combat_quality_ceremonial_high_impact_selected",
                "combat_quality_insatiable_sandpit_countdown",
                "combat_quality_insatiable_sandpit_active",
                "combat_quality_insatiable_sandpit_lt3",
                "combat_quality_insatiable_sandpit_1",
                "combat_quality_insatiable_frantic_escape_hand_count",
                "combat_quality_insatiable_frantic_escape_draw_count",
                "combat_quality_insatiable_frantic_escape_discard_count",
                "combat_quality_insatiable_frantic_escape_exhaust_count",
                "combat_quality_insatiable_frantic_escape_total_count",
                "combat_quality_insatiable_frantic_escape_candidate_count",
                "combat_quality_insatiable_frantic_escape_available",
                "combat_quality_insatiable_frantic_escape_urgency",
                "combat_quality_insatiable_frantic_escape_bonus_applied",
                "combat_quality_insatiable_non_escape_at1_penalty_count",
                "combat_quality_insatiable_escape_cycle_risk",
                "combat_quality_insatiable_lethal_candidate_count",
                "combat_quality_insatiable_frantic_escape_selected",
                "combat_quality_insatiable_frantic_escape_missed_lt3",
                "combat_quality_insatiable_frantic_escape_missed_at_1",
                "combat_quality_insatiable_non_escape_at_1_selected",
                # P0 hardening hooks: HP-cost / X-cost / selection / identity /
                # transient leak — sourced from typed helper outputs in
                # combat_env.step diagnostics via diag_key_map.
                "combat_quality_hp_cost_self_lethal_selected",
                "combat_quality_hp_cost_low_margin_selected",
                "combat_quality_hp_cost_unblockable_value",
                "combat_quality_x_cost_zero_bad_selected",
                "combat_quality_x_cost_zero_selected",
                "combat_quality_x_cost_energy_value",
                "combat_quality_x_cost_star_value",
                "combat_quality_star_x_selected",
                "combat_quality_selection_text_fallback_selected",
                "combat_quality_selection_runtime_internal_selected",
                "combat_quality_card_identity_text_fallback_selected",
                "combat_quality_card_identity_runtime_internal_selected",
                "combat_quality_transient_leaked_selected",
                "combat_quality_prior_transient_only_end_turn",
                # P0-3 / P0-4 (recovery 2026-05-06): hard-guard observability.
                # Each rate captures whether the override hook fired this
                # decision so TensorBoard can show the gate independent of
                # the soft kaiser/insatiable bias signals.
                "combat_quality_kaiser_facing_guard_available",
                "combat_quality_kaiser_facing_guard_applied",
                "combat_quality_kaiser_facing_guard_override",
                "combat_quality_kaiser_facing_guard_lethal_exemption",
                "combat_quality_kaiser_nonfacing_nonlethal_selected",
                "combat_quality_kaiser_end_turn_under_risk_with_candidate",
                "combat_quality_insatiable_escape_force_available",
                "combat_quality_insatiable_escape_force_applied",
                "combat_quality_insatiable_escape_force_override",
                "combat_quality_insatiable_escape_force_lethal_exemption",
                "combat_quality_insatiable_non_escape_at1_blocked",
                # P1-2 (recovery 2026-05-07): X-cost zero-energy hard invalid.
                "combat_quality_x_cost_zero_guard_available",
                "combat_quality_x_cost_zero_guard_applied",
                "combat_quality_x_cost_zero_guard_override",
                "combat_quality_x_cost_zero_guard_no_alternative",
                "combat_quality_x_cost_zero_guard_end_turn_fallback",
                # P1-3 (recovery 2026-05-07): HP-cost margin hard guard.
                "combat_quality_hp_cost_margin_guard_available",
                "combat_quality_hp_cost_margin_guard_applied",
                "combat_quality_hp_cost_margin_guard_override",
                "combat_quality_hp_cost_margin_guard_lethal_exemption",
                "combat_quality_hp_cost_margin_guard_no_alternative",
                # P1-1 (recovery 2026-05-07): Potion bad-use hard guard.
                "combat_quality_potion_bad_guard_available",
                "combat_quality_potion_bad_guard_applied",
                "combat_quality_potion_bad_guard_override",
                "combat_quality_potion_bad_guard_lethal_exemption",
                "combat_quality_potion_bad_guard_no_alternative",
                "combat_quality_potion_bad_guard_invalid_obs",
                "combat_quality_potion_bad_guard_critical_hp_survival_skip",
                "combat_quality_potion_bad_guard_critical_hp_idle_waste_not_skipped",
                "combat_quality_potion_bad_guard_end_turn_fallback_blocked_unsafe",
                "combat_quality_potion_bad_guard_forced_end_turn_hopeless",
                "combat_quality_potion_bad_guard_boss_race_skip",
                "combat_quality_potion_bad_guard_late_normal_race_skip",
                "combat_quality_potion_bad_hopeless_guard_late_normal_race_skip",
                "combat_quality_potion_bad_hopeless_guard_boss_survival_skip",
                "combat_quality_potion_bad_hopeless_guard_lagavulin_setup_skip",
                "combat_quality_potion_discard_guard_available",
                "combat_quality_potion_discard_guard_applied",
                "combat_quality_potion_discard_guard_override",
                "combat_quality_potion_discard_guard_candidate_count",
                "combat_quality_potion_discard_guard_saved_survival",
                "combat_quality_elite_boss_lethal_end_turn_guard_available",
                "combat_quality_elite_boss_lethal_end_turn_guard_applied",
                "combat_quality_elite_boss_lethal_end_turn_guard_override",
                "combat_quality_meaningful_damage_endturn_guard_available",
                "combat_quality_meaningful_damage_endturn_guard_applied",
                "combat_quality_meaningful_damage_endturn_guard_override",
                "combat_quality_meaningful_damage_endturn_guard_candidate_count",
                "combat_quality_meaningful_damage_endturn_guard_no_alternative",
                "combat_quality_meaningful_damage_endturn_guard_lethal_candidate",
                "combat_quality_meaningful_damage_endturn_guard_pressure_skip",
                "combat_quality_no_pressure_block_guard_available",
                "combat_quality_no_pressure_block_guard_applied",
                "combat_quality_no_pressure_block_guard_override",
                "combat_quality_no_pressure_block_guard_candidate_count",
                "combat_quality_no_pressure_block_guard_no_alternative",
                "combat_quality_no_pressure_block_guard_pressure_skip",
                "combat_quality_no_pressure_block_guard_pressure_attack_window",
                "combat_quality_no_pressure_block_guard_low_value_pressure",
                "combat_quality_no_pressure_block_guard_trivial_pressure",
                "combat_quality_no_pressure_block_guard_lethal_candidate",
                "combat_quality_boss_survival_potion_guard_available",
                "combat_quality_boss_survival_potion_guard_applied",
                "combat_quality_boss_survival_potion_guard_override",
                "combat_quality_boss_survival_potion_guard_lethal_exemption",
                "combat_quality_boss_survival_potion_guard_no_alternative",
                "combat_quality_boss_race_potion_guard_available",
                "combat_quality_boss_race_potion_guard_applied",
                "combat_quality_boss_race_potion_guard_override",
                "combat_quality_boss_race_potion_guard_lethal_exemption",
                "combat_quality_boss_race_potion_guard_no_alternative",
                "combat_quality_boss_race_potion_guard_lagavulin_setup_escape",
                "combat_quality_boss_survival_block_guard_available",
                "combat_quality_boss_survival_block_guard_applied",
                "combat_quality_boss_survival_block_guard_override",
                "combat_quality_boss_survival_block_guard_lethal_exemption",
                "combat_quality_boss_survival_block_guard_no_alternative",
                "combat_quality_boss_survival_block_guard_insufficient_candidate",
                "combat_quality_late_normal_lethal_end_turn_guard_available",
                "combat_quality_late_normal_lethal_end_turn_guard_applied",
                "combat_quality_late_normal_lethal_end_turn_guard_override",
                "combat_quality_late_normal_survival_guard_available",
                "combat_quality_late_normal_survival_guard_applied",
                "combat_quality_late_normal_survival_guard_override",
                "combat_quality_late_normal_survival_guard_lethal_exemption",
                "combat_quality_late_normal_survival_guard_no_alternative",
                "combat_quality_late_normal_survival_guard_candidate_count",
                "combat_quality_late_normal_survival_guard_insufficient_candidate",
                "combat_quality_late_normal_race_potion_guard_available",
                "combat_quality_late_normal_race_potion_guard_applied",
                "combat_quality_late_normal_race_potion_guard_override",
                "combat_quality_late_normal_race_potion_guard_lethal_exemption",
                "combat_quality_late_normal_race_potion_guard_no_alternative",
                "combat_quality_late_normal_race_potion_guard_candidate_count",
                "combat_quality_survival_non_endturn_guard_available",
                "combat_quality_survival_non_endturn_guard_applied",
                "combat_quality_survival_non_endturn_guard_override",
                "combat_quality_survival_non_endturn_guard_lethal_exemption",
                "combat_quality_survival_non_endturn_guard_progress_exemption",
                "combat_quality_survival_non_endturn_guard_scaling_enemy_exemption",
                "combat_quality_survival_non_endturn_guard_scaling_enemy_candidate",
                "combat_quality_survival_non_endturn_guard_no_alternative",
                "combat_quality_survival_non_endturn_guard_candidate_count",
                "combat_quality_survival_non_endturn_guard_insufficient_candidate",
                "combat_quality_hard_guard_override_any",
                "combat_quality_hard_guard_policy_target_rewrite",
                # P2-2 (recovery 2026-05-07): Selection-loop dead-loop guard.
                "combat_quality_selection_loop_screen_active",
                "combat_quality_selection_loop_detected",
                "combat_quality_selection_loop_applied",
                "combat_quality_selection_loop_auto_confirm",
                "combat_quality_selection_loop_auto_cancel",
                "combat_quality_selection_loop_alt_pick",
                "combat_quality_selection_loop_no_alternative",
                "combat_quality_selection_repeated_same_option",
                # P2-1 (recovery 2026-05-07): runtime card-state presence.
                "combat_quality_card_runtime_instance_uuid_present",
                "combat_quality_card_runtime_modified_cost_present",
                "combat_quality_card_runtime_exhaust_flag_present",
                "combat_quality_card_runtime_ethereal_flag_present",
                "combat_quality_card_runtime_retain_flag_present",
                "combat_quality_card_runtime_enchantment_present",
                "combat_quality_card_runtime_replay_flag_present",
                "combat_quality_card_runtime_selection_effect_present",
            ):
                if key in search_stats:
                    search_values[key].append(self._safe_float(search_stats.get(key)))

        terminated = bool(metadata.get("terminated", False))
        truncated = bool(metadata.get("truncated", False))
        death_floor = self._safe_float(metadata.get("death_floor"))
        loss = (death_floor > 0.0) or (not terminated and not truncated)
        win = terminated and not truncated and death_floor <= 0.0
        entry = boss_entry if isinstance(boss_entry, dict) else {}
        # TASK-A4: subtract used-this-combat from raw final count so a use_potion
        # that succeeded but left a stale slot in the death frame does not falsely
        # inflate potion_unused_on_death.  Diagnostic JSONL captures the raw
        # transition so root-cause analysis can still see it.
        used_potion_count = int(metadata.get("used_potion_count_this_combat", 0) or 0)
        adjusted_unused = max(int(final_potion_count) - used_potion_count, 0)
        potion_unused_on_death = 1.0 if loss and adjusted_unused > 0 else 0.0
        final_lucky_potion_count = self._boss_episode_lucky_count(metadata.get("final_potion_dump"))
        lucky_seen_anywhere = bool(
            bool(metadata.get("lucky_seen_this_combat"))
            or bool(metadata.get("lucky_legal_this_combat"))
            or final_lucky_potion_count > 0
            or self._boss_episode_lucky_count(metadata.get("last_seen_potions_this_combat")) > 0
            or self._boss_episode_lucky_count(metadata.get("last_seen_legal_potion_actions_this_combat")) > 0
            or self._boss_episode_lucky_count(metadata.get("selected_potion_actions_this_combat")) > 0
            or self._boss_episode_lucky_count(metadata.get("potion_use_transitions_this_combat")) > 0
        )
        lucky_selected_or_used = bool(
            bool(metadata.get("lucky_selected_this_combat"))
            or self._boss_episode_lucky_used_in_rows(metadata.get("selected_potion_actions_this_combat"))
            or self._boss_episode_lucky_used_in_rows(metadata.get("potion_use_transitions_this_combat"))
        )
        lucky_unused_on_boss_death = bool(loss and lucky_seen_anywhere and not lucky_selected_or_used)
        final_lucky_unused_on_boss_death = bool(loss and final_lucky_potion_count > 0 and not lucky_selected_or_used)
        try:
            self._dump_death_final_potions(
                loss=loss,
                final_potion_count=int(final_potion_count),
                used_potion_count=int(used_potion_count),
                adjusted_unused=int(adjusted_unused),
                encounter=encounter_id,
                tier=encounter_tier,
                potion_dump=metadata.get("final_potion_dump") if isinstance(metadata.get("final_potion_dump"), list) else None,
            )
        except Exception:
            pass
        # P2-3 (recovery 2026-05-07): death-slice writer for the encounters
        # the recovery doc identified as biggest blockers. We capture the
        # last few decisions before death + mechanic context so a human can
        # drill into root causes without re-running the run.
        try:
            self._dump_death_slice(
                trajectory=trajectory,
                encounter_id=encounter_id,
                encounter_tier=encounter_tier,
                loss=loss,
                steps=steps,
            )
        except Exception:
            pass

        metrics = {
            "boss/attempt_count": 1.0,
            "boss/win": 1.0 if win else 0.0,
            "boss/loss": 1.0 if loss else 0.0,
            "boss/reward": self._safe_float(metadata.get("episode_total_reward", metadata.get("episode_reward"))),
            # Lucky Tonic / 幸运药剂 visibility audit.  These are episode-level
            # boss metrics so a death where the agent saw a usable Lucky but
            # skipped it is visible in TensorBoard instead of requiring JSONL
            # greps through death_slices.
            "boss/lucky_seen_rate": 1.0 if bool(metadata.get("lucky_seen_this_combat")) else 0.0,
            "boss/lucky_legal_rate": 1.0 if bool(metadata.get("lucky_legal_this_combat")) else 0.0,
            "boss/lucky_skip_on_boss_death_rate": 1.0
            if lucky_unused_on_boss_death
            else 0.0,
            "boss/lucky_seen_anywhere_on_boss_death_rate": 1.0 if (loss and lucky_seen_anywhere) else 0.0,
            "boss/final_lucky_potion_count_mean": float(final_lucky_potion_count),
            "boss/final_lucky_unused_on_boss_death_rate": 1.0 if final_lucky_unused_on_boss_death else 0.0,
            "boss/lucky_used_on_boss_win_rate": 1.0
            if (win and lucky_selected_or_used)
            else 0.0,
            "episode/boss_entry_hp": self._safe_float(entry.get("hp")),
            "episode/boss_entry_hp_ratio": self._safe_float(entry.get("hp_ratio")),
            "episode/boss_entry_potion_count": float(max(int(final_potion_count), 0)),
            "boss_combat/decision_count": float(total_steps),
            "boss_combat/family_end_turn_rate": self._safe_rate(end_turn_count, total_steps),
            "boss_combat/family_potion_rate": self._safe_rate(use_potion_count, total_steps),
            "boss_combat/wasteful_end_turn_rate": self._safe_rate(wasteful_end_turn_count, total_steps),
            "boss_combat/potion_unused_on_death_rate": potion_unused_on_death,
            "boss_combat/potion_unused_on_death_raw_rate": 1.0 if loss and int(final_potion_count) > 0 else 0.0,
            "boss_combat/used_potion_count_this_combat_mean": float(used_potion_count),
            "boss_combat/final_potion_count_mean": float(int(final_potion_count)),
            "boss_combat/playable_cards_left_mean": float(np.mean(search_values.get("mean_predicted_legal_count", [0.0]))),
            "boss_combat/direct_policy_used_rate": float(np.mean(search_values.get("search_mode_direct_policy", [0.0]))),
            "boss_combat/search_mode_direct_rollout_planner": float(np.mean(search_values.get("search_mode_direct_rollout_planner", [0.0]))),
            "boss_combat/num_simulations": float(np.mean(search_values.get("num_simulations", [0.0]))),
            "boss_combat/direct_rollout_objective_q_mean": float(np.mean(search_values.get("direct_rollout_objective_q_mean", [0.0]))),
            "boss_combat/direct_rollout_uncertainty_mean": float(np.mean(search_values.get("direct_rollout_uncertainty_mean", [0.0]))),
            "boss_combat/direct_rollout_uncertainty_bias_abs_mean": float(np.mean(search_values.get("direct_rollout_uncertainty_bias_abs_mean", [0.0]))),
            "boss_combat/direct_rollout_branch_disagreement_mean": float(np.mean(search_values.get("direct_rollout_branch_disagreement_mean", [0.0]))),
            "boss_combat/direct_rollout_risk_q_mean": float(np.mean(search_values.get("direct_rollout_risk_q_mean", [0.0]))),
            "boss_combat/quality_bias_applied_rate": float(np.mean(search_values.get("combat_quality_bias_applied", [0.0]))),
            "boss_combat/quality_bias_abs_mean": float(np.mean(search_values.get("combat_quality_bias_abs_mean", [0.0]))),
            "boss_combat/root_bias_scale_mean": float(np.mean(search_values.get("root_bias_scale", [0.0]))),
            "boss_combat/root_bias_nonzero_rate": float(np.mean(search_values.get("root_bias_nonzero", [0.0]))),
            "boss_combat/root_bias_abs_mean": float(np.mean(search_values.get("root_bias_abs_mean", [0.0]))),
            "boss_combat/root_bias_max_abs": float(np.max(search_values.get("root_bias_max_abs", [0.0]))),
            "boss_combat/root_bias_changed_top1_rate": float(np.mean(search_values.get("root_bias_changed_top1", [0.0]))),
            "boss_combat/root_bias_selected_action_delta_mean": float(np.mean(search_values.get("root_bias_selected_action_delta", [0.0]))),
            "boss_combat/root_bias_suppressed_by_gate_rate": float(np.mean(search_values.get("root_bias_suppressed_by_gate", [0.0]))),
            "boss_combat/root_bias_scale_effective_mean": float(np.mean(search_values.get("root_bias_scale_effective", [0.0]))),
            "boss_combat/wasteful_end_turn_bias_applied_rate": float(np.mean(search_values.get("combat_quality_wasteful_end_turn_bias_applied", [0.0]))),
            "boss_combat/wasteful_end_turn_available_rate": float(np.mean(search_values.get("combat_quality_wasteful_end_turn_available", [0.0]))),
            "boss_combat/wasteful_end_turn_selected_rate": float(np.mean(search_values.get("combat_quality_wasteful_end_turn_selected", [0.0]))),
            "boss_combat/true_wasteful_end_turn_available_rate": float(np.mean(search_values.get("combat_quality_true_wasteful_end_turn_available", [0.0]))),
            "boss_combat/true_wasteful_end_turn_selected_rate": float(np.mean(search_values.get("combat_quality_true_wasteful_end_turn_selected", [0.0]))),
            "boss_combat/strategic_defer_available_rate": float(np.mean(search_values.get("combat_quality_strategic_defer_available", [0.0]))),
            "boss_combat/strategic_defer_end_turn_selected_rate": float(np.mean(search_values.get("combat_quality_strategic_defer_end_turn_selected", [0.0]))),
            "boss_combat/bad_end_turn_available_rate": float(np.mean(search_values.get("combat_quality_bad_end_turn_available", [0.0]))),
            "boss_combat/bad_end_turn_selected_rate": float(np.mean(search_values.get("combat_quality_bad_end_turn_selected", [0.0]))),
            "boss_combat/forced_end_turn_available_rate": float(np.mean(search_values.get("combat_quality_forced_end_turn_available", [0.0]))),
            "boss_combat/forced_end_turn_selected_rate": float(np.mean(search_values.get("combat_quality_forced_end_turn_selected", [0.0]))),
            "boss_combat/end_turn_unknown_selected_rate": float(np.mean(search_values.get("combat_quality_end_turn_unknown_selected", [0.0]))),
            "boss_combat/urgent_positive_action_count_mean": float(np.mean(search_values.get("combat_quality_urgent_positive_action_count", [0.0]))),
            "boss_combat/deferable_positive_action_count_mean": float(np.mean(search_values.get("combat_quality_deferable_positive_action_count", [0.0]))),
            "boss_combat/deferable_exhaust_card_count_mean": float(np.mean(search_values.get("combat_quality_deferable_exhaust_card_count", [0.0]))),
            "boss_combat/energy_gain_without_followup_count_mean": float(np.mean(search_values.get("combat_quality_energy_gain_without_followup_count", [0.0]))),
            "boss_combat/typed_followup_missing_count_mean": float(np.mean(search_values.get("combat_quality_typed_followup_missing_count", [0.0]))),
            "boss_combat/typed_future_penalty_count_mean": float(np.mean(search_values.get("combat_quality_typed_future_penalty_count", [0.0]))),
            "boss_combat/typed_no_draw_count_mean": float(np.mean(search_values.get("combat_quality_typed_no_draw_count", [0.0]))),
            "boss_combat/typed_card_state_mutation_count_mean": float(np.mean(search_values.get("combat_quality_typed_card_state_mutation_count", [0.0]))),
            "boss_combat/setup_followup_dependent_count_mean": float(np.mean(search_values.get("combat_quality_setup_followup_dependent_count", [0.0]))),
            "boss_combat/setup_followup_available_count_mean": float(np.mean(search_values.get("combat_quality_setup_followup_available_count", [0.0]))),
            "boss_combat/potion_available_count_mean": float(np.mean(search_values.get("combat_quality_potion_available_count", [0.0]))),
            "boss_combat/potion_urgent_available_mean": float(np.mean(search_values.get("combat_quality_potion_urgent_count", [0.0]))),
            "boss_combat/potion_low_urgency_available_mean": float(np.mean(search_values.get("combat_quality_potion_low_urgency_count", [0.0]))),
            "boss_combat/potion_save_recommended_available_mean": float(np.mean(search_values.get("combat_quality_potion_save_recommended_count", [0.0]))),
            "boss_combat/potion_no_followup_available_mean": float(np.mean(search_values.get("combat_quality_potion_no_followup_count", [0.0]))),
            "boss_combat/potion_lethal_available_mean": float(np.mean(search_values.get("combat_quality_potion_lethal_count", [0.0]))),
            "boss_combat/potion_prevent_lethal_available_mean": float(np.mean(search_values.get("combat_quality_potion_prevent_lethal_count", [0.0]))),
            "boss_combat/potion_mechanism_available_mean": float(np.mean(search_values.get("combat_quality_potion_mechanism_count", [0.0]))),
            "boss_combat/potion_overkill_available_mean": float(np.mean(search_values.get("combat_quality_potion_overkill_count", [0.0]))),
            "boss_combat/potion_block_waste_available_mean": float(np.mean(search_values.get("combat_quality_potion_block_waste_count", [0.0]))),
            "boss_combat/potion_use_quality_available_mean": float(np.mean(search_values.get("combat_quality_potion_use_quality_mean", [0.0]))),
            "boss_combat/potion_waste_risk_available_mean": float(np.mean(search_values.get("combat_quality_potion_waste_risk_mean", [0.0]))),
            "boss_combat/potion_selected_rate": float(np.mean(search_values.get("combat_quality_potion_selected", [0.0]))),
            "boss_combat/potion_selected_when_available_rate": float(np.mean(search_values.get("combat_quality_potion_selected_when_available", [0.0]))),
            "boss_combat/potion_high_urgency_selected_rate": float(np.mean(search_values.get("combat_quality_potion_high_urgency_selected", [0.0]))),
            "boss_combat/potion_low_urgency_selected_rate": float(np.mean(search_values.get("combat_quality_potion_low_urgency_selected", [0.0]))),
            "boss_combat/potion_save_recommended_selected_rate": float(np.mean(search_values.get("combat_quality_potion_save_recommended_selected", [0.0]))),
            "boss_combat/potion_no_followup_selected_rate": float(np.mean(search_values.get("combat_quality_potion_no_followup_selected", [0.0]))),
            "boss_combat/potion_lethal_selected_rate": float(np.mean(search_values.get("combat_quality_potion_lethal_selected", [0.0]))),
            "boss_combat/potion_prevent_lethal_selected_rate": float(np.mean(search_values.get("combat_quality_potion_prevent_lethal_selected", [0.0]))),
            "boss_combat/potion_mechanism_selected_rate": float(np.mean(search_values.get("combat_quality_potion_mechanism_selected", [0.0]))),
            "boss_combat/potion_overkill_selected_rate": float(np.mean(search_values.get("combat_quality_potion_overkill_selected", [0.0]))),
            "boss_combat/potion_block_waste_selected_rate": float(np.mean(search_values.get("combat_quality_potion_block_waste_selected", [0.0]))),
            "boss_combat/potion_use_quality_selected_mean": (
                float(np.sum(search_values.get("combat_quality_potion_use_quality_selected", [0.0])))
                / max(float(np.sum(search_values.get("combat_quality_potion_selected", [0.0]))), 1.0)
            ),
            "boss_combat/potion_waste_risk_selected_mean": (
                float(np.sum(search_values.get("combat_quality_potion_waste_risk_selected", [0.0])))
                / max(float(np.sum(search_values.get("combat_quality_potion_selected", [0.0]))), 1.0)
            ),
            # Phase 5 of potion-timing-modeling-plan.md §11.1: aggregate
            # save_value, hand-context, long-term, requires-followup metrics so
            # TensorBoard can spot regressions in the new structured timing.
            "boss_combat/potion_save_value_available_mean": float(np.mean(search_values.get("combat_quality_potion_save_value_mean", [0.0]))),
            "boss_combat/potion_hand_context_good_mean": float(np.mean(search_values.get("combat_quality_potion_hand_context_good_count", [0.0]))),
            "boss_combat/potion_hand_context_bad_mean": float(np.mean(search_values.get("combat_quality_potion_hand_context_bad_count", [0.0]))),
            "boss_combat/potion_long_term_mean": float(np.mean(search_values.get("combat_quality_potion_long_term_count", [0.0]))),
            "boss_combat/potion_requires_followup_mean": float(np.mean(search_values.get("combat_quality_potion_requires_followup_count", [0.0]))),
            "boss_combat/x_cost_available_count_mean": float(np.mean(search_values.get("combat_quality_x_cost_available_count", [0.0]))),
            "boss_combat/x_cost_selected_rate": float(np.mean(search_values.get("combat_quality_x_cost_selected", [0.0]))),
            "boss_combat/x_cost_selected_energy_mean": (
                float(np.sum(search_values.get("combat_quality_x_cost_selected_energy", [0.0])))
                / max(float(np.sum(search_values.get("combat_quality_x_cost_selected", [0.0]))), 1.0)
            ),
            "boss_combat/x_cost_zero_energy_selected_rate": float(np.mean(search_values.get("combat_quality_x_cost_zero_energy_selected", [0.0]))),
            "boss_combat/zero_energy_x_cost_available_mean": float(np.mean(search_values.get("combat_quality_zero_energy_x_cost_count", [0.0]))),
            "boss_combat/zero_energy_x_cost_selected_rate": float(np.mean(search_values.get("combat_quality_zero_energy_x_cost_selected", [0.0]))),
            "boss_combat/x_cost_bad_available_count_mean": float(np.mean(search_values.get("combat_quality_x_cost_bad_count", [0.0]))),
            "boss_combat/x_cost_effective_energy_mean": float(np.mean(search_values.get("combat_quality_x_cost_effective_energy_mean", [0.0]))),
            "boss_combat/x_cost_selected_effective_energy_mean": float(np.mean(search_values.get("combat_quality_x_cost_selected_effective_energy", [0.0]))),
            "boss_combat/x_cost_has_non_energy_effect_selected_rate": float(np.mean(search_values.get("combat_quality_x_cost_has_non_energy_effect_selected", [0.0]))),
            "boss_combat/x_cost_bad_selected_rate": float(np.mean(search_values.get("combat_quality_x_cost_bad_selected", [0.0]))),
            "boss_combat/direct_end_turn_selected_rate": float(np.mean(search_values.get("combat_quality_end_turn_selected", [0.0]))),
            "boss_combat/energy_mean": float(np.mean(search_values.get("combat_quality_energy", [0.0]))),
            "boss_combat/positive_action_count_mean": float(np.mean(search_values.get("combat_quality_positive_action_count", [0.0]))),
            "boss_combat/mandatory_positive_action_count_mean": float(np.mean(search_values.get("combat_quality_mandatory_positive_action_count", [0.0]))),
            "boss_combat/strategic_skip_candidate_count_mean": float(np.mean(search_values.get("combat_quality_strategic_skip_candidate_count", [0.0]))),
            "boss_combat/refund_no_followup_available_mean": float(np.mean(search_values.get("combat_quality_refund_no_followup_available", [0.0]))),
            "boss_combat/refund_no_followup_selected_rate": float(np.mean(search_values.get("combat_quality_refund_no_followup_selected", [0.0]))),
            "boss_combat/refund_no_followup_with_progress_selected_rate": float(np.mean(search_values.get("combat_quality_refund_no_followup_with_progress_selected", [0.0]))),
            "boss_combat/refund_no_followup_progress_alternative_selected_rate": float(np.mean(search_values.get("combat_quality_refund_no_followup_progress_alternative_selected", [0.0]))),
            "boss_combat/refund_no_followup_no_alternative_selected_rate": float(np.mean(search_values.get("combat_quality_refund_no_followup_no_alternative_selected", [0.0]))),
            "boss_combat/refund_no_followup_progress_alternative_count_mean": float(np.mean(search_values.get("combat_quality_refund_no_followup_progress_alternative_count", [0.0]))),
            "boss_combat/refund_no_followup_guard_available_rate": float(np.mean(search_values.get("combat_quality_refund_no_followup_guard_available", [0.0]))),
            "boss_combat/refund_no_followup_guard_applied_rate": float(np.mean(search_values.get("combat_quality_refund_no_followup_guard_applied", [0.0]))),
            "boss_combat/refund_no_followup_guard_override_rate": float(np.mean(search_values.get("combat_quality_refund_no_followup_guard_override", [0.0]))),
            "boss_combat/refund_no_followup_guard_no_alternative_rate": float(np.mean(search_values.get("combat_quality_refund_no_followup_guard_no_alternative", [0.0]))),
            "boss_combat/refund_no_followup_guard_end_turn_fallback_rate": float(np.mean(search_values.get("combat_quality_refund_no_followup_guard_end_turn_fallback", [0.0]))),
            "boss_combat/strategic_skip_selected_rate": float(np.mean(search_values.get("combat_quality_strategic_skip_selected", [0.0]))),
            "boss_combat/enchantment_seen_rate": float(np.mean(search_values.get("combat_quality_enchantment_seen", [0.0]))),
            "boss_combat/affliction_seen_rate": float(np.mean(search_values.get("combat_quality_affliction_seen", [0.0]))),
            "boss_combat/kaiser_back_attack_risk_mean": float(np.mean(search_values.get("combat_quality_kaiser_back_attack_risk", [0.0]))),
            "boss_combat/kaiser_defense_candidate_count_mean": float(np.mean(search_values.get("combat_quality_kaiser_defense_candidate_count", [0.0]))),
            "boss_combat/kaiser_facing_change_candidate_count_mean": float(np.mean(search_values.get("combat_quality_kaiser_facing_change_candidate_count", [0.0]))),
            "boss_combat/kaiser_pressure_candidate_count_mean": float(np.mean(search_values.get("combat_quality_kaiser_pressure_candidate_count", [0.0]))),
            "boss_combat/kaiser_facing_change_selected_rate": float(np.mean(search_values.get("combat_quality_kaiser_facing_change_selected", [0.0]))),
            "boss_combat/kaiser_pressure_selected_rate": float(np.mean(search_values.get("combat_quality_kaiser_pressure_selected", [0.0]))),
            "boss_combat/kaiser_risky_end_turn_selected_rate": float(np.mean(search_values.get("combat_quality_kaiser_risky_end_turn_selected", [0.0]))),
            "boss_combat/ceremonial_one_card_lock_rate": float(np.mean(search_values.get("combat_quality_ceremonial_one_card_lock", [0.0]))),
            "boss_combat/ceremonial_stun_window_rate": float(np.mean(search_values.get("combat_quality_ceremonial_stun_window", [0.0]))),
            "boss_combat/ceremonial_low_impact_count_mean": float(np.mean(search_values.get("combat_quality_ceremonial_low_impact_count", [0.0]))),
            "boss_combat/ceremonial_high_impact_count_mean": float(np.mean(search_values.get("combat_quality_ceremonial_high_impact_count", [0.0]))),
            "boss_combat/kaiser_defense_selected_rate": float(np.mean(search_values.get("combat_quality_kaiser_defense_selected", [0.0]))),
            "boss_combat/ceremonial_low_impact_selected_rate": float(np.mean(search_values.get("combat_quality_ceremonial_low_impact_selected", [0.0]))),
            "boss_combat/ceremonial_high_impact_selected_rate": float(np.mean(search_values.get("combat_quality_ceremonial_high_impact_selected", [0.0]))),
            "boss_combat/insatiable_sandpit_countdown_mean": float(np.mean(search_values.get("combat_quality_insatiable_sandpit_countdown", [0.0]))),
            "boss_combat/insatiable_sandpit_active_rate": float(np.mean(search_values.get("combat_quality_insatiable_sandpit_active", [0.0]))),
            "boss_combat/insatiable_sandpit_lt3_rate": float(np.mean(search_values.get("combat_quality_insatiable_sandpit_lt3", [0.0]))),
            "boss_combat/insatiable_sandpit_1_rate": float(np.mean(search_values.get("combat_quality_insatiable_sandpit_1", [0.0]))),
            "boss_combat/insatiable_frantic_escape_hand_count_mean": float(np.mean(search_values.get("combat_quality_insatiable_frantic_escape_hand_count", [0.0]))),
            "boss_combat/insatiable_frantic_escape_draw_count_mean": float(np.mean(search_values.get("combat_quality_insatiable_frantic_escape_draw_count", [0.0]))),
            "boss_combat/insatiable_frantic_escape_discard_count_mean": float(np.mean(search_values.get("combat_quality_insatiable_frantic_escape_discard_count", [0.0]))),
            "boss_combat/insatiable_frantic_escape_exhaust_count_mean": float(np.mean(search_values.get("combat_quality_insatiable_frantic_escape_exhaust_count", [0.0]))),
            "boss_combat/insatiable_frantic_escape_total_count_mean": float(np.mean(search_values.get("combat_quality_insatiable_frantic_escape_total_count", [0.0]))),
            "boss_combat/insatiable_frantic_escape_available_rate": float(np.mean(search_values.get("combat_quality_insatiable_frantic_escape_available", [0.0]))),
            "boss_combat/insatiable_frantic_escape_candidate_count_mean": float(np.mean(search_values.get("combat_quality_insatiable_frantic_escape_candidate_count", [0.0]))),
            "boss_combat/insatiable_frantic_escape_selected_rate": float(np.mean(search_values.get("combat_quality_insatiable_frantic_escape_selected", [0.0]))),
            "boss_combat/insatiable_frantic_escape_missed_lt3_rate": float(np.mean(search_values.get("combat_quality_insatiable_frantic_escape_missed_lt3", [0.0]))),
            "boss_combat/insatiable_frantic_escape_missed_at_1_rate": float(np.mean(search_values.get("combat_quality_insatiable_frantic_escape_missed_at_1", [0.0]))),
            "boss_combat/insatiable_frantic_escape_urgency_mean": float(np.mean(search_values.get("combat_quality_insatiable_frantic_escape_urgency", [0.0]))),
            "boss_combat/insatiable_frantic_escape_bonus_applied_rate": float(np.mean(search_values.get("combat_quality_insatiable_frantic_escape_bonus_applied", [0.0]))),
            "boss_combat/insatiable_non_escape_at1_penalty_count_mean": float(np.mean(search_values.get("combat_quality_insatiable_non_escape_at1_penalty_count", [0.0]))),
            "boss_combat/insatiable_escape_cycle_risk_mean": float(np.mean(search_values.get("combat_quality_insatiable_escape_cycle_risk", [0.0]))),
            "boss_combat/insatiable_lethal_candidate_count_mean": float(np.mean(search_values.get("combat_quality_insatiable_lethal_candidate_count", [0.0]))),
            # P0 hardening metrics — sourced from combat_env P0 helper
            # diagnostics (HP-cost safety, X-cost / Star-X, selection
            # typed contract, card identity, transient end_turn leak).
            # All are episode-level rate / mean averages for the §11
            # smoke gate.
            "boss_combat/hp_cost_self_lethal_selected_rate": float(np.mean(search_values.get("combat_quality_hp_cost_self_lethal_selected", [0.0]))),
            "boss_combat/hp_cost_low_margin_selected_rate": float(np.mean(search_values.get("combat_quality_hp_cost_low_margin_selected", [0.0]))),
            "boss_combat/hp_cost_unblockable_value_mean": float(np.mean(search_values.get("combat_quality_hp_cost_unblockable_value", [0.0]))),
            "boss_combat/x_cost_zero_bad_selected_rate_p0": float(np.mean(search_values.get("combat_quality_x_cost_zero_bad_selected", [0.0]))),
            "boss_combat/x_cost_zero_selected_rate_p0": float(np.mean(search_values.get("combat_quality_x_cost_zero_selected", [0.0]))),
            "boss_combat/x_cost_energy_value_mean_p0": float(np.mean(search_values.get("combat_quality_x_cost_energy_value", [0.0]))),
            "boss_combat/x_cost_star_value_mean_p0": float(np.mean(search_values.get("combat_quality_x_cost_star_value", [0.0]))),
            "boss_combat/star_x_selected_rate": float(np.mean(search_values.get("combat_quality_star_x_selected", [0.0]))),
            "boss_combat/selection_text_fallback_selected_rate": float(np.mean(search_values.get("combat_quality_selection_text_fallback_selected", [0.0]))),
            "boss_combat/selection_runtime_internal_selected_rate": float(np.mean(search_values.get("combat_quality_selection_runtime_internal_selected", [0.0]))),
            "boss_combat/card_identity_text_fallback_selected_rate": float(np.mean(search_values.get("combat_quality_card_identity_text_fallback_selected", [0.0]))),
            "boss_combat/card_identity_runtime_internal_selected_rate": float(np.mean(search_values.get("combat_quality_card_identity_runtime_internal_selected", [0.0]))),
            "boss_combat/transient_leaked_selected_rate": float(np.mean(search_values.get("combat_quality_transient_leaked_selected", [0.0]))),
            "boss_combat/prior_transient_only_end_turn_rate": float(np.mean(search_values.get("combat_quality_prior_transient_only_end_turn", [0.0]))),
            # P0-3 (recovery 2026-05-06): Kaiser facing change hard-guard
            # observability. ``available`` is the rate at which the guard
            # had a candidate to override to; ``applied``/``override`` is the
            # rate at which the guard actually fired.
            "boss_combat/kaiser_facing_guard_available_rate": float(np.mean(search_values.get("combat_quality_kaiser_facing_guard_available", [0.0]))),
            "boss_combat/kaiser_facing_guard_applied_rate": float(np.mean(search_values.get("combat_quality_kaiser_facing_guard_applied", [0.0]))),
            "boss_combat/kaiser_facing_guard_override_rate": float(np.mean(search_values.get("combat_quality_kaiser_facing_guard_override", [0.0]))),
            "boss_combat/kaiser_facing_guard_lethal_exemption_rate": float(np.mean(search_values.get("combat_quality_kaiser_facing_guard_lethal_exemption", [0.0]))),
            "boss_combat/kaiser_nonfacing_nonlethal_selected_rate": float(np.mean(search_values.get("combat_quality_kaiser_nonfacing_nonlethal_selected", [0.0]))),
            "boss_combat/kaiser_end_turn_under_risk_with_candidate_rate": float(np.mean(search_values.get("combat_quality_kaiser_end_turn_under_risk_with_candidate", [0.0]))),
            # P0-4 (recovery 2026-05-06): Insatiable Frantic Escape hard
            # force on countdown <=1. Mirrors the Kaiser observability set.
            "boss_combat/insatiable_escape_force_available_rate": float(np.mean(search_values.get("combat_quality_insatiable_escape_force_available", [0.0]))),
            "boss_combat/insatiable_escape_force_applied_rate": float(np.mean(search_values.get("combat_quality_insatiable_escape_force_applied", [0.0]))),
            "boss_combat/insatiable_escape_force_override_rate": float(np.mean(search_values.get("combat_quality_insatiable_escape_force_override", [0.0]))),
            "boss_combat/insatiable_escape_force_lethal_exemption_rate": float(np.mean(search_values.get("combat_quality_insatiable_escape_force_lethal_exemption", [0.0]))),
            "boss_combat/insatiable_non_escape_at1_blocked_rate": float(np.mean(search_values.get("combat_quality_insatiable_non_escape_at1_blocked", [0.0]))),
            # P1-2 (recovery 2026-05-07): X-cost zero-energy hard invalid.
            "boss_combat/x_cost_zero_guard_available_rate": float(np.mean(search_values.get("combat_quality_x_cost_zero_guard_available", [0.0]))),
            "boss_combat/x_cost_zero_guard_applied_rate": float(np.mean(search_values.get("combat_quality_x_cost_zero_guard_applied", [0.0]))),
            "boss_combat/x_cost_zero_guard_override_rate": float(np.mean(search_values.get("combat_quality_x_cost_zero_guard_override", [0.0]))),
            "boss_combat/x_cost_zero_guard_no_alternative_rate": float(np.mean(search_values.get("combat_quality_x_cost_zero_guard_no_alternative", [0.0]))),
            "boss_combat/x_cost_zero_guard_end_turn_fallback_rate": float(np.mean(search_values.get("combat_quality_x_cost_zero_guard_end_turn_fallback", [0.0]))),
            # P1-3 (recovery 2026-05-07): HP-cost survival-margin guard.
            "boss_combat/hp_cost_margin_guard_available_rate": float(np.mean(search_values.get("combat_quality_hp_cost_margin_guard_available", [0.0]))),
            "boss_combat/hp_cost_margin_guard_applied_rate": float(np.mean(search_values.get("combat_quality_hp_cost_margin_guard_applied", [0.0]))),
            "boss_combat/hp_cost_margin_guard_override_rate": float(np.mean(search_values.get("combat_quality_hp_cost_margin_guard_override", [0.0]))),
            "boss_combat/hp_cost_margin_guard_lethal_exemption_rate": float(np.mean(search_values.get("combat_quality_hp_cost_margin_guard_lethal_exemption", [0.0]))),
            "boss_combat/hp_cost_margin_guard_no_alternative_rate": float(np.mean(search_values.get("combat_quality_hp_cost_margin_guard_no_alternative", [0.0]))),
            # P1-1 (recovery 2026-05-07): Potion bad-use hard guard.
            "boss_combat/potion_bad_guard_available_rate": float(np.mean(search_values.get("combat_quality_potion_bad_guard_available", [0.0]))),
            "boss_combat/potion_bad_guard_applied_rate": float(np.mean(search_values.get("combat_quality_potion_bad_guard_applied", [0.0]))),
            "boss_combat/potion_bad_guard_override_rate": float(np.mean(search_values.get("combat_quality_potion_bad_guard_override", [0.0]))),
            "boss_combat/potion_bad_guard_lethal_exemption_rate": float(np.mean(search_values.get("combat_quality_potion_bad_guard_lethal_exemption", [0.0]))),
            "boss_combat/potion_bad_guard_no_alternative_rate": float(np.mean(search_values.get("combat_quality_potion_bad_guard_no_alternative", [0.0]))),
            "boss_combat/potion_bad_guard_invalid_obs_rate": float(np.mean(search_values.get("combat_quality_potion_bad_guard_invalid_obs", [0.0]))),
            "boss_combat/potion_bad_guard_critical_hp_survival_skip_rate": float(np.mean(search_values.get("combat_quality_potion_bad_guard_critical_hp_survival_skip", [0.0]))),
            "boss_combat/potion_bad_guard_critical_hp_idle_waste_not_skipped_rate": float(np.mean(search_values.get("combat_quality_potion_bad_guard_critical_hp_idle_waste_not_skipped", [0.0]))),
            "boss_combat/potion_bad_guard_end_turn_fallback_blocked_unsafe_rate": float(np.mean(search_values.get("combat_quality_potion_bad_guard_end_turn_fallback_blocked_unsafe", [0.0]))),
            "boss_combat/potion_bad_guard_forced_end_turn_hopeless_rate": float(np.mean(search_values.get("combat_quality_potion_bad_guard_forced_end_turn_hopeless", [0.0]))),
            "boss_combat/potion_bad_guard_boss_race_skip_rate": float(np.mean(search_values.get("combat_quality_potion_bad_guard_boss_race_skip", [0.0]))),
            "boss_combat/potion_bad_guard_late_normal_race_skip_rate": float(np.mean(search_values.get("combat_quality_potion_bad_guard_late_normal_race_skip", [0.0]))),
            "boss_combat/potion_bad_hopeless_guard_late_normal_race_skip_rate": float(np.mean(search_values.get("combat_quality_potion_bad_hopeless_guard_late_normal_race_skip", [0.0]))),
            "boss_combat/potion_bad_hopeless_guard_boss_survival_skip_rate": float(np.mean(search_values.get("combat_quality_potion_bad_hopeless_guard_boss_survival_skip", [0.0]))),
            "boss_combat/potion_bad_hopeless_guard_lagavulin_setup_skip_rate": float(np.mean(search_values.get("combat_quality_potion_bad_hopeless_guard_lagavulin_setup_skip", [0.0]))),
            "boss_combat/potion_discard_guard_available_rate": float(np.mean(search_values.get("combat_quality_potion_discard_guard_available", [0.0]))),
            "boss_combat/potion_discard_guard_applied_rate": float(np.mean(search_values.get("combat_quality_potion_discard_guard_applied", [0.0]))),
            "boss_combat/potion_discard_guard_override_rate": float(np.mean(search_values.get("combat_quality_potion_discard_guard_override", [0.0]))),
            "boss_combat/potion_discard_guard_candidate_count_mean": float(np.mean(search_values.get("combat_quality_potion_discard_guard_candidate_count", [0.0]))),
            "boss_combat/potion_discard_guard_saved_survival_rate": float(np.mean(search_values.get("combat_quality_potion_discard_guard_saved_survival", [0.0]))),
            "boss_combat/lethal_end_turn_guard_available_rate": float(np.mean(search_values.get("combat_quality_elite_boss_lethal_end_turn_guard_available", [0.0]))),
            "boss_combat/lethal_end_turn_guard_applied_rate": float(np.mean(search_values.get("combat_quality_elite_boss_lethal_end_turn_guard_applied", [0.0]))),
            "boss_combat/lethal_end_turn_guard_override_rate": float(np.mean(search_values.get("combat_quality_elite_boss_lethal_end_turn_guard_override", [0.0]))),
            "boss_combat/meaningful_damage_endturn_guard_available_rate": float(np.mean(search_values.get("combat_quality_meaningful_damage_endturn_guard_available", [0.0]))),
            "boss_combat/meaningful_damage_endturn_guard_applied_rate": float(np.mean(search_values.get("combat_quality_meaningful_damage_endturn_guard_applied", [0.0]))),
            "boss_combat/meaningful_damage_endturn_guard_override_rate": float(np.mean(search_values.get("combat_quality_meaningful_damage_endturn_guard_override", [0.0]))),
            "boss_combat/meaningful_damage_endturn_guard_candidate_count_mean": float(np.mean(search_values.get("combat_quality_meaningful_damage_endturn_guard_candidate_count", [0.0]))),
            "boss_combat/meaningful_damage_endturn_guard_no_alternative_rate": float(np.mean(search_values.get("combat_quality_meaningful_damage_endturn_guard_no_alternative", [0.0]))),
            "boss_combat/meaningful_damage_endturn_guard_lethal_candidate_rate": float(np.mean(search_values.get("combat_quality_meaningful_damage_endturn_guard_lethal_candidate", [0.0]))),
            "boss_combat/meaningful_damage_endturn_guard_pressure_skip_rate": float(np.mean(search_values.get("combat_quality_meaningful_damage_endturn_guard_pressure_skip", [0.0]))),
            "boss_combat/no_pressure_block_guard_available_rate": float(np.mean(search_values.get("combat_quality_no_pressure_block_guard_available", [0.0]))),
            "boss_combat/no_pressure_block_guard_applied_rate": float(np.mean(search_values.get("combat_quality_no_pressure_block_guard_applied", [0.0]))),
            "boss_combat/no_pressure_block_guard_override_rate": float(np.mean(search_values.get("combat_quality_no_pressure_block_guard_override", [0.0]))),
            "boss_combat/no_pressure_block_guard_candidate_count_mean": float(np.mean(search_values.get("combat_quality_no_pressure_block_guard_candidate_count", [0.0]))),
            "boss_combat/no_pressure_block_guard_no_alternative_rate": float(np.mean(search_values.get("combat_quality_no_pressure_block_guard_no_alternative", [0.0]))),
            "boss_combat/no_pressure_block_guard_pressure_skip_rate": float(np.mean(search_values.get("combat_quality_no_pressure_block_guard_pressure_skip", [0.0]))),
            "boss_combat/no_pressure_block_guard_pressure_attack_window_rate": float(np.mean(search_values.get("combat_quality_no_pressure_block_guard_pressure_attack_window", [0.0]))),
            "boss_combat/no_pressure_block_guard_low_value_pressure_rate": float(np.mean(search_values.get("combat_quality_no_pressure_block_guard_low_value_pressure", [0.0]))),
            "boss_combat/no_pressure_block_guard_trivial_pressure_rate": float(np.mean(search_values.get("combat_quality_no_pressure_block_guard_trivial_pressure", [0.0]))),
            "boss_combat/no_pressure_block_guard_lethal_candidate_rate": float(np.mean(search_values.get("combat_quality_no_pressure_block_guard_lethal_candidate", [0.0]))),
            "boss_combat/boss_survival_potion_guard_available_rate": float(np.mean(search_values.get("combat_quality_boss_survival_potion_guard_available", [0.0]))),
            "boss_combat/boss_survival_potion_guard_applied_rate": float(np.mean(search_values.get("combat_quality_boss_survival_potion_guard_applied", [0.0]))),
            "boss_combat/boss_survival_potion_guard_override_rate": float(np.mean(search_values.get("combat_quality_boss_survival_potion_guard_override", [0.0]))),
            "boss_combat/boss_survival_potion_guard_lethal_exemption_rate": float(np.mean(search_values.get("combat_quality_boss_survival_potion_guard_lethal_exemption", [0.0]))),
            "boss_combat/boss_survival_potion_guard_no_alternative_rate": float(np.mean(search_values.get("combat_quality_boss_survival_potion_guard_no_alternative", [0.0]))),
            "boss_combat/boss_race_potion_guard_available_rate": float(np.mean(search_values.get("combat_quality_boss_race_potion_guard_available", [0.0]))),
            "boss_combat/boss_race_potion_guard_applied_rate": float(np.mean(search_values.get("combat_quality_boss_race_potion_guard_applied", [0.0]))),
            "boss_combat/boss_race_potion_guard_override_rate": float(np.mean(search_values.get("combat_quality_boss_race_potion_guard_override", [0.0]))),
            "boss_combat/boss_race_potion_guard_lethal_exemption_rate": float(np.mean(search_values.get("combat_quality_boss_race_potion_guard_lethal_exemption", [0.0]))),
            "boss_combat/boss_race_potion_guard_no_alternative_rate": float(np.mean(search_values.get("combat_quality_boss_race_potion_guard_no_alternative", [0.0]))),
            "boss_combat/boss_race_potion_guard_lagavulin_setup_escape_rate": float(np.mean(search_values.get("combat_quality_boss_race_potion_guard_lagavulin_setup_escape", [0.0]))),
            "boss_combat/boss_survival_block_guard_available_rate": float(np.mean(search_values.get("combat_quality_boss_survival_block_guard_available", [0.0]))),
            "boss_combat/boss_survival_block_guard_applied_rate": float(np.mean(search_values.get("combat_quality_boss_survival_block_guard_applied", [0.0]))),
            "boss_combat/boss_survival_block_guard_override_rate": float(np.mean(search_values.get("combat_quality_boss_survival_block_guard_override", [0.0]))),
            "boss_combat/boss_survival_block_guard_lethal_exemption_rate": float(np.mean(search_values.get("combat_quality_boss_survival_block_guard_lethal_exemption", [0.0]))),
            "boss_combat/boss_survival_block_guard_no_alternative_rate": float(np.mean(search_values.get("combat_quality_boss_survival_block_guard_no_alternative", [0.0]))),
            "boss_combat/boss_survival_block_guard_insufficient_candidate_rate": float(np.mean(search_values.get("combat_quality_boss_survival_block_guard_insufficient_candidate", [0.0]))),
            # P1-6 (act1 recovery 2026-05-10): late normal hallway
            # survival guard.  Kept under both boss_combat (legacy
            # aggregate used by existing dashboards) and combat_guards
            # (domain-neutral, because this is explicitly not boss-only).
            "boss_combat/late_normal_lethal_end_turn_guard_available_rate": float(np.mean(search_values.get("combat_quality_late_normal_lethal_end_turn_guard_available", [0.0]))),
            "boss_combat/late_normal_lethal_end_turn_guard_applied_rate": float(np.mean(search_values.get("combat_quality_late_normal_lethal_end_turn_guard_applied", [0.0]))),
            "boss_combat/late_normal_lethal_end_turn_guard_override_rate": float(np.mean(search_values.get("combat_quality_late_normal_lethal_end_turn_guard_override", [0.0]))),
            "boss_combat/late_normal_survival_guard_available_rate": float(np.mean(search_values.get("combat_quality_late_normal_survival_guard_available", [0.0]))),
            "boss_combat/late_normal_survival_guard_applied_rate": float(np.mean(search_values.get("combat_quality_late_normal_survival_guard_applied", [0.0]))),
            "boss_combat/late_normal_survival_guard_override_rate": float(np.mean(search_values.get("combat_quality_late_normal_survival_guard_override", [0.0]))),
            "boss_combat/late_normal_survival_guard_lethal_exemption_rate": float(np.mean(search_values.get("combat_quality_late_normal_survival_guard_lethal_exemption", [0.0]))),
            "boss_combat/late_normal_survival_guard_no_alternative_rate": float(np.mean(search_values.get("combat_quality_late_normal_survival_guard_no_alternative", [0.0]))),
            "boss_combat/late_normal_survival_guard_candidate_count_mean": float(np.mean(search_values.get("combat_quality_late_normal_survival_guard_candidate_count", [0.0]))),
            "boss_combat/late_normal_survival_guard_insufficient_candidate_rate": float(np.mean(search_values.get("combat_quality_late_normal_survival_guard_insufficient_candidate", [0.0]))),
            "boss_combat/late_normal_race_potion_guard_available_rate": float(np.mean(search_values.get("combat_quality_late_normal_race_potion_guard_available", [0.0]))),
            "boss_combat/late_normal_race_potion_guard_applied_rate": float(np.mean(search_values.get("combat_quality_late_normal_race_potion_guard_applied", [0.0]))),
            "boss_combat/late_normal_race_potion_guard_override_rate": float(np.mean(search_values.get("combat_quality_late_normal_race_potion_guard_override", [0.0]))),
            "boss_combat/late_normal_race_potion_guard_lethal_exemption_rate": float(np.mean(search_values.get("combat_quality_late_normal_race_potion_guard_lethal_exemption", [0.0]))),
            "boss_combat/late_normal_race_potion_guard_no_alternative_rate": float(np.mean(search_values.get("combat_quality_late_normal_race_potion_guard_no_alternative", [0.0]))),
            "boss_combat/late_normal_race_potion_guard_candidate_count_mean": float(np.mean(search_values.get("combat_quality_late_normal_race_potion_guard_candidate_count", [0.0]))),
            "boss_combat/survival_non_endturn_guard_available_rate": float(np.mean(search_values.get("combat_quality_survival_non_endturn_guard_available", [0.0]))),
            "boss_combat/survival_non_endturn_guard_applied_rate": float(np.mean(search_values.get("combat_quality_survival_non_endturn_guard_applied", [0.0]))),
            "boss_combat/survival_non_endturn_guard_override_rate": float(np.mean(search_values.get("combat_quality_survival_non_endturn_guard_override", [0.0]))),
            "boss_combat/survival_non_endturn_guard_lethal_exemption_rate": float(np.mean(search_values.get("combat_quality_survival_non_endturn_guard_lethal_exemption", [0.0]))),
            "boss_combat/survival_non_endturn_guard_progress_exemption_rate": float(np.mean(search_values.get("combat_quality_survival_non_endturn_guard_progress_exemption", [0.0]))),
            "boss_combat/survival_non_endturn_guard_scaling_enemy_exemption_rate": float(np.mean(search_values.get("combat_quality_survival_non_endturn_guard_scaling_enemy_exemption", [0.0]))),
            "boss_combat/survival_non_endturn_guard_scaling_enemy_candidate_rate": float(np.mean(search_values.get("combat_quality_survival_non_endturn_guard_scaling_enemy_candidate", [0.0]))),
            "boss_combat/survival_non_endturn_guard_no_alternative_rate": float(np.mean(search_values.get("combat_quality_survival_non_endturn_guard_no_alternative", [0.0]))),
            "boss_combat/survival_non_endturn_guard_candidate_count_mean": float(np.mean(search_values.get("combat_quality_survival_non_endturn_guard_candidate_count", [0.0]))),
            "boss_combat/survival_non_endturn_guard_insufficient_candidate_rate": float(np.mean(search_values.get("combat_quality_survival_non_endturn_guard_insufficient_candidate", [0.0]))),
            "boss_combat/hard_guard_override_any_rate": float(np.mean(search_values.get("combat_quality_hard_guard_override_any", [0.0]))),
            "boss_combat/hard_guard_policy_target_rewrite_rate": float(np.mean(search_values.get("combat_quality_hard_guard_policy_target_rewrite", [0.0]))),
            "combat_guards/meaningful_damage_endturn_guard_available_rate": float(np.mean(search_values.get("combat_quality_meaningful_damage_endturn_guard_available", [0.0]))),
            "combat_guards/meaningful_damage_endturn_guard_applied_rate": float(np.mean(search_values.get("combat_quality_meaningful_damage_endturn_guard_applied", [0.0]))),
            "combat_guards/meaningful_damage_endturn_guard_override_rate": float(np.mean(search_values.get("combat_quality_meaningful_damage_endturn_guard_override", [0.0]))),
            "combat_guards/meaningful_damage_endturn_guard_candidate_count_mean": float(np.mean(search_values.get("combat_quality_meaningful_damage_endturn_guard_candidate_count", [0.0]))),
            "combat_guards/meaningful_damage_endturn_guard_no_alternative_rate": float(np.mean(search_values.get("combat_quality_meaningful_damage_endturn_guard_no_alternative", [0.0]))),
            "combat_guards/meaningful_damage_endturn_guard_lethal_candidate_rate": float(np.mean(search_values.get("combat_quality_meaningful_damage_endturn_guard_lethal_candidate", [0.0]))),
            "combat_guards/meaningful_damage_endturn_guard_pressure_skip_rate": float(np.mean(search_values.get("combat_quality_meaningful_damage_endturn_guard_pressure_skip", [0.0]))),
            "combat_guards/no_pressure_block_guard_available_rate": float(np.mean(search_values.get("combat_quality_no_pressure_block_guard_available", [0.0]))),
            "combat_guards/no_pressure_block_guard_applied_rate": float(np.mean(search_values.get("combat_quality_no_pressure_block_guard_applied", [0.0]))),
            "combat_guards/no_pressure_block_guard_override_rate": float(np.mean(search_values.get("combat_quality_no_pressure_block_guard_override", [0.0]))),
            "combat_guards/no_pressure_block_guard_candidate_count_mean": float(np.mean(search_values.get("combat_quality_no_pressure_block_guard_candidate_count", [0.0]))),
            "combat_guards/no_pressure_block_guard_no_alternative_rate": float(np.mean(search_values.get("combat_quality_no_pressure_block_guard_no_alternative", [0.0]))),
            "combat_guards/no_pressure_block_guard_pressure_skip_rate": float(np.mean(search_values.get("combat_quality_no_pressure_block_guard_pressure_skip", [0.0]))),
            "combat_guards/no_pressure_block_guard_pressure_attack_window_rate": float(np.mean(search_values.get("combat_quality_no_pressure_block_guard_pressure_attack_window", [0.0]))),
            "combat_guards/no_pressure_block_guard_low_value_pressure_rate": float(np.mean(search_values.get("combat_quality_no_pressure_block_guard_low_value_pressure", [0.0]))),
            "combat_guards/no_pressure_block_guard_trivial_pressure_rate": float(np.mean(search_values.get("combat_quality_no_pressure_block_guard_trivial_pressure", [0.0]))),
            "combat_guards/no_pressure_block_guard_lethal_candidate_rate": float(np.mean(search_values.get("combat_quality_no_pressure_block_guard_lethal_candidate", [0.0]))),
            "combat_guards/boss_survival_block_guard_insufficient_candidate_rate": float(np.mean(search_values.get("combat_quality_boss_survival_block_guard_insufficient_candidate", [0.0]))),
            "combat_guards/late_normal_lethal_end_turn_guard_available_rate": float(np.mean(search_values.get("combat_quality_late_normal_lethal_end_turn_guard_available", [0.0]))),
            "combat_guards/late_normal_lethal_end_turn_guard_applied_rate": float(np.mean(search_values.get("combat_quality_late_normal_lethal_end_turn_guard_applied", [0.0]))),
            "combat_guards/late_normal_survival_guard_available_rate": float(np.mean(search_values.get("combat_quality_late_normal_survival_guard_available", [0.0]))),
            "combat_guards/late_normal_survival_guard_applied_rate": float(np.mean(search_values.get("combat_quality_late_normal_survival_guard_applied", [0.0]))),
            "combat_guards/late_normal_survival_guard_no_alternative_rate": float(np.mean(search_values.get("combat_quality_late_normal_survival_guard_no_alternative", [0.0]))),
            "combat_guards/late_normal_survival_guard_insufficient_candidate_rate": float(np.mean(search_values.get("combat_quality_late_normal_survival_guard_insufficient_candidate", [0.0]))),
            "combat_guards/late_normal_race_potion_guard_available_rate": float(np.mean(search_values.get("combat_quality_late_normal_race_potion_guard_available", [0.0]))),
            "combat_guards/late_normal_race_potion_guard_applied_rate": float(np.mean(search_values.get("combat_quality_late_normal_race_potion_guard_applied", [0.0]))),
            "combat_guards/late_normal_race_potion_guard_no_alternative_rate": float(np.mean(search_values.get("combat_quality_late_normal_race_potion_guard_no_alternative", [0.0]))),
            "combat_guards/survival_non_endturn_guard_available_rate": float(np.mean(search_values.get("combat_quality_survival_non_endturn_guard_available", [0.0]))),
            "combat_guards/survival_non_endturn_guard_applied_rate": float(np.mean(search_values.get("combat_quality_survival_non_endturn_guard_applied", [0.0]))),
            "combat_guards/survival_non_endturn_guard_progress_exemption_rate": float(np.mean(search_values.get("combat_quality_survival_non_endturn_guard_progress_exemption", [0.0]))),
            "combat_guards/survival_non_endturn_guard_scaling_enemy_exemption_rate": float(np.mean(search_values.get("combat_quality_survival_non_endturn_guard_scaling_enemy_exemption", [0.0]))),
            "combat_guards/survival_non_endturn_guard_scaling_enemy_candidate_rate": float(np.mean(search_values.get("combat_quality_survival_non_endturn_guard_scaling_enemy_candidate", [0.0]))),
            "combat_guards/survival_non_endturn_guard_no_alternative_rate": float(np.mean(search_values.get("combat_quality_survival_non_endturn_guard_no_alternative", [0.0]))),
            "combat_guards/survival_non_endturn_guard_insufficient_candidate_rate": float(np.mean(search_values.get("combat_quality_survival_non_endturn_guard_insufficient_candidate", [0.0]))),
            "combat_guards/potion_bad_guard_end_turn_fallback_blocked_unsafe_rate": float(np.mean(search_values.get("combat_quality_potion_bad_guard_end_turn_fallback_blocked_unsafe", [0.0]))),
            "combat_guards/potion_bad_guard_critical_hp_idle_waste_not_skipped_rate": float(np.mean(search_values.get("combat_quality_potion_bad_guard_critical_hp_idle_waste_not_skipped", [0.0]))),
            "combat_guards/potion_bad_guard_forced_end_turn_hopeless_rate": float(np.mean(search_values.get("combat_quality_potion_bad_guard_forced_end_turn_hopeless", [0.0]))),
            "combat_guards/potion_bad_guard_boss_race_skip_rate": float(np.mean(search_values.get("combat_quality_potion_bad_guard_boss_race_skip", [0.0]))),
            "combat_guards/potion_bad_guard_late_normal_race_skip_rate": float(np.mean(search_values.get("combat_quality_potion_bad_guard_late_normal_race_skip", [0.0]))),
            "combat_guards/potion_bad_hopeless_guard_late_normal_race_skip_rate": float(np.mean(search_values.get("combat_quality_potion_bad_hopeless_guard_late_normal_race_skip", [0.0]))),
            "combat_guards/potion_bad_hopeless_guard_boss_survival_skip_rate": float(np.mean(search_values.get("combat_quality_potion_bad_hopeless_guard_boss_survival_skip", [0.0]))),
            "combat_guards/potion_bad_hopeless_guard_lagavulin_setup_skip_rate": float(np.mean(search_values.get("combat_quality_potion_bad_hopeless_guard_lagavulin_setup_skip", [0.0]))),
            "combat_guards/boss_race_potion_guard_applied_rate": float(np.mean(search_values.get("combat_quality_boss_race_potion_guard_applied", [0.0]))),
            "combat_guards/boss_race_potion_guard_no_alternative_rate": float(np.mean(search_values.get("combat_quality_boss_race_potion_guard_no_alternative", [0.0]))),
            "combat_guards/boss_race_potion_guard_lagavulin_setup_escape_rate": float(np.mean(search_values.get("combat_quality_boss_race_potion_guard_lagavulin_setup_escape", [0.0]))),
            "combat_guards/potion_discard_guard_available_rate": float(np.mean(search_values.get("combat_quality_potion_discard_guard_available", [0.0]))),
            "combat_guards/potion_discard_guard_applied_rate": float(np.mean(search_values.get("combat_quality_potion_discard_guard_applied", [0.0]))),
            "combat_guards/potion_discard_guard_saved_survival_rate": float(np.mean(search_values.get("combat_quality_potion_discard_guard_saved_survival", [0.0]))),
            "combat_guards/hard_guard_override_any_rate": float(np.mean(search_values.get("combat_quality_hard_guard_override_any", [0.0]))),
            "combat_guards/hard_guard_policy_target_rewrite_rate": float(np.mean(search_values.get("combat_quality_hard_guard_policy_target_rewrite", [0.0]))),
            # P2-2 (recovery 2026-05-07): Selection-loop dead-loop guard.
            "boss_combat/selection_loop_screen_active_rate": float(np.mean(search_values.get("combat_quality_selection_loop_screen_active", [0.0]))),
            "boss_combat/selection_loop_detected_rate": float(np.mean(search_values.get("combat_quality_selection_loop_detected", [0.0]))),
            "boss_combat/selection_loop_applied_rate": float(np.mean(search_values.get("combat_quality_selection_loop_applied", [0.0]))),
            "boss_combat/selection_loop_auto_confirm_rate": float(np.mean(search_values.get("combat_quality_selection_loop_auto_confirm", [0.0]))),
            "boss_combat/selection_loop_auto_cancel_rate": float(np.mean(search_values.get("combat_quality_selection_loop_auto_cancel", [0.0]))),
            "boss_combat/selection_loop_alt_pick_rate": float(np.mean(search_values.get("combat_quality_selection_loop_alt_pick", [0.0]))),
            "boss_combat/selection_loop_no_alternative_rate": float(np.mean(search_values.get("combat_quality_selection_loop_no_alternative", [0.0]))),
            "boss_combat/selection_repeated_same_option_rate": float(np.mean(search_values.get("combat_quality_selection_repeated_same_option", [0.0]))),
            # P2-1 (recovery 2026-05-07): runtime card-state presence rates.
            # Each tag is the per-decision indicator that the bridge exposed
            # the corresponding runtime modifier on the chosen card. Low
            # rates highlight a bridge gap, not a model bug.
            "card_runtime/instance_uuid_present_rate": float(np.mean(search_values.get("combat_quality_card_runtime_instance_uuid_present", [0.0]))),
            "card_runtime/modified_cost_present_rate": float(np.mean(search_values.get("combat_quality_card_runtime_modified_cost_present", [0.0]))),
            "card_runtime/exhaust_flag_present_rate": float(np.mean(search_values.get("combat_quality_card_runtime_exhaust_flag_present", [0.0]))),
            "card_runtime/ethereal_flag_present_rate": float(np.mean(search_values.get("combat_quality_card_runtime_ethereal_flag_present", [0.0]))),
            "card_runtime/retain_flag_present_rate": float(np.mean(search_values.get("combat_quality_card_runtime_retain_flag_present", [0.0]))),
            "card_runtime/enchantment_present_rate": float(np.mean(search_values.get("combat_quality_card_runtime_enchantment_present", [0.0]))),
            "card_runtime/replay_flag_present_rate": float(np.mean(search_values.get("combat_quality_card_runtime_replay_flag_present", [0.0]))),
            "card_runtime/selection_effect_present_rate": float(np.mean(search_values.get("combat_quality_card_runtime_selection_effect_present", [0.0]))),
        }
        metrics.update(boss_card_block_waste_metrics(search_values))
        for family, count in sorted(family_counts.items()):
            safe_family = family.replace("/", "_").replace(" ", "_") or "unknown"
            metrics[f"boss_combat/family_{safe_family}_rate"] = self._safe_rate(count, total_steps)

        # Boss-combat aggregate metrics are useful for trend watching, but Kaiser /
        # Ceremonial diagnosis gets diluted by other bosses.  Mirror every
        # boss_combat/* scalar under boss_combat/<encounter_id>/* so TensorBoard can
        # answer mechanism-specific questions without post-processing.
        self._mirror_metrics_per_encounter(metrics, encounter_id, prefix="boss_combat/")

        for tag, value in metrics.items():
            self.writer.add_scalar(tag, float(value), self.episode_count)
        return metrics

    @staticmethod
    def _safe_encounter_namespaces(encounter_id: str) -> list[str]:
        """Map an encounter id to TensorBoard-safe namespace token(s)."""
        raw = str(encounter_id or "").strip().lower()
        if not raw:
            return []
        encounter_base = re.sub(r"^encounter[\._:/-]+", "", raw)
        safe = re.sub(r"[^0-9a-zA-Z]+", "_", encounter_base).strip("_")
        raw_safe = re.sub(r"[^0-9a-zA-Z]+", "_", raw).strip("_")
        return list(dict.fromkeys(x for x in (safe, raw_safe) if x))

    @classmethod
    def _mirror_metrics_per_encounter(
        cls,
        metrics: dict[str, float],
        encounter_id: str,
        *,
        prefix: str = "boss_combat/",
    ) -> dict[str, float]:
        """Mirror every ``<prefix><metric>`` scalar under ``<prefix><encounter>/<metric>``.

        Without this mirror, an aggregate boss metric averaged across encounters
        hides per-boss regressions (e.g. Kaiser facing failures averaged with
        Construct Menagerie wins).  Mutates ``metrics`` in place and also returns it.
        """
        for namespace in cls._safe_encounter_namespaces(encounter_id):
            for tag, value in list(metrics.items()):
                if tag.startswith(prefix):
                    rest = tag[len(prefix):]
                    if "/" in rest and rest.split("/", 1)[0] == namespace:
                        # Already mirrored.
                        continue
                    metrics[f"{prefix}{namespace}/{rest}"] = value
        return metrics
