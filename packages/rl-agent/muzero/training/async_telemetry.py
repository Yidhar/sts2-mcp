"""Async actor telemetry replay for MuZero training.

Async combat actors collect with ``NullSummaryWriter``.  The learner replays
per-episode metrics through this module so TensorBoard tags stay centralized
without growing the CLI/training loop file.
"""

from __future__ import annotations

import re
from typing import Any, Sequence

from muzero.combat_quality import (
    COMBAT_QUALITY_CARD_BLOCK_SEARCH_SUFFIXES,
    COMBAT_QUALITY_GUARD_SEARCH_SUFFIXES,
)
from muzero.diagnostics.deck_build_metrics import (
    CARD_REWARD_TB_KEYS,
    DEATH_DECK_QUALITY_TB_KEYS,
    FINAL_DECK_QUALITY_TB_KEYS,
)
from muzero.diagnostics.deck_upgrade_metrics import DECK_UPGRADE_TB_KEYS
from muzero.diagnostics.rest_site_metrics import REST_SITE_TB_KEYS
from muzero.diagnostics.shop_metrics import SHOP_TB_KEYS
from muzero.diagnostics.summoner_targeting import SUMMONER_TARGETING_TB_KEYS
from muzero.diagnostics.target_priority import TARGET_PRIORITY_TB_KEYS
from muzero.combat_quality.summoner_target_guard import SUMMONER_TARGET_GUARD_SEARCH_SUFFIXES
from muzero.combat_quality.target_priority_guard import TARGET_PRIORITY_GUARD_SEARCH_SUFFIXES
from muzero.training.card_reward_guard import CARD_REWARD_GUARD_SEARCH_SUFFIXES
from muzero.training.card_reward_pick_quality_guard import CARD_REWARD_PICK_QUALITY_GUARD_SEARCH_SUFFIXES
from muzero.training.post_search_policy_retarget import POST_SEARCH_HARD_GUARD_SEARCH_SUFFIXES
from muzero.training.rest_site_smith_guard import REST_SITE_SMITH_GUARD_SEARCH_SUFFIXES
from muzero.training.shop_action_guard import SHOP_ACTION_GUARD_SEARCH_SUFFIXES
from sts2_env.observation_v2 import DECISION_DOMAINS


def _to_float(value: Any, default: float = 0.0) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return float(default)


def _safe_tag(value: Any) -> str:
    return re.sub(r"[^0-9a-zA-Z]+", "_", str(value or "").strip().lower()).strip("_") or "unknown"


def log_environment_episode_telemetry(
    *,
    writer: Any,
    episode_index: int,
    episode_telemetry: dict[str, Any] | None,
) -> None:
    """Replay env_v2 per-episode counters into ``env/*`` TensorBoard tags.

    Full-run Act1 decisions depend on sparse campfire/shop/boss surfaces.  The
    env already attaches these counters to terminal ``info``; this helper keeps
    sync and async self-play on the same tag path instead of relying on ad-hoc
    event-log parsing.
    """

    if not isinstance(episode_telemetry, dict):
        return
    for key, value in episode_telemetry.items():
        writer.add_scalar(f"env/{_safe_tag(key)}", _to_float(value), int(episode_index))


def log_async_episode_scalars(*, trainer: Any, actor_index: int, episode_metrics: dict[str, Any], actor_completed_episodes: Sequence[int]) -> None:
    metric_map = {
        "max_floor": "episode/max_floor",
        "max_act_id": "episode/max_act_id",
        "rooms_seen": "episode/rooms_seen",
        "death_floor": "episode/death_floor",
        "elite_rooms_seen": "episode/elite_rooms_seen",
        "settlement_signal": "episode/settlement_signal",
        "settlement_applied_steps": "episode/settlement_applied_steps",
        "settlement_total_bonus": "episode/settlement_total_bonus",
        "episode_reward_augmented": "episode/reward_augmented",
    }
    for metric_key, writer_key in metric_map.items():
        if metric_key in episode_metrics:
            trainer.writer.add_scalar(writer_key, _to_float(episode_metrics.get(metric_key)), trainer.episode_count)
    for metric_key, writer_key in (
        ("act1_boss_seen", "episode/act1_boss_seen"),
        ("act1_clear", "episode/act1_clear"),
    ):
        if metric_key in episode_metrics:
            trainer.writer.add_scalar(writer_key, 1.0 if bool(episode_metrics.get(metric_key)) else 0.0, trainer.episode_count)

    episode_telemetry = episode_metrics.get("episode_telemetry")
    log_environment_episode_telemetry(
        writer=trainer.writer,
        episode_index=trainer.episode_count,
        episode_telemetry=episode_telemetry if isinstance(episode_telemetry, dict) else None,
    )

    # Async actors use NullSummaryWriter while collecting episodes, so replay the
    # per-episode diagnostics on the learner writer here. Without this, combat
    # sandbox actor-learner runs silently lose decision/search/boss namespaces.
    decision_counts = episode_metrics.get("decision_counts") if isinstance(episode_metrics.get("decision_counts"), dict) else {}
    total_decisions = int(episode_metrics.get("decision_total", 0) or 0)
    if total_decisions > 0:
        domain_overrides = episode_metrics.get("decision_domain_overrides") if isinstance(episode_metrics.get("decision_domain_overrides"), dict) else {}
        combat_like_count = int(episode_metrics.get("combat_like_decision_count", 0) or 0)
        eligible_count = int(episode_metrics.get("direct_policy_eligible_count", 0) or 0)
        used_count = int(episode_metrics.get("direct_policy_used_count", 0) or 0)
        fast_path_counts = episode_metrics.get("fast_path_counts") if isinstance(episode_metrics.get("fast_path_counts"), dict) else {}
        fast_path_total = int(episode_metrics.get("fast_path_total", 0) or 0)
        fast_path_reason_counts = episode_metrics.get("fast_path_reason_counts") if isinstance(episode_metrics.get("fast_path_reason_counts"), dict) else {}

        trainer.writer.add_scalar("decision/domain_override_count", float(sum(int(v or 0) for v in domain_overrides.values())), trainer.episode_count)
        trainer.writer.add_scalar("decision/combat_like_count", float(combat_like_count), trainer.episode_count)
        trainer.writer.add_scalar("decision/direct_policy_eligible_count", float(eligible_count), trainer.episode_count)
        trainer.writer.add_scalar("decision/direct_policy_used_count", float(used_count), trainer.episode_count)
        trainer.writer.add_scalar("decision/direct_policy_used_rate", float(used_count) / float(max(eligible_count, 1)), trainer.episode_count)
        trainer.writer.add_scalar("decision/fast_path_count", float(fast_path_total), trainer.episode_count)
        trainer.writer.add_scalar("decision/fast_path_rate", float(fast_path_total) / float(max(total_decisions, 1)), trainer.episode_count)

        for domain in DECISION_DOMAINS:
            domain_count = int(decision_counts.get(domain, 0) or 0)
            trainer.writer.add_scalar(f"decision/{domain}_count", float(domain_count), trainer.episode_count)
            trainer.writer.add_scalar(f"decision/{domain}_share", float(domain_count) / float(max(total_decisions, 1)), trainer.episode_count)
            if domain_count > 0:
                domain_fast_paths = int(fast_path_counts.get(domain, 0) or 0)
                trainer.writer.add_scalar(f"decision/{domain}/fast_path_rate", float(domain_fast_paths) / float(domain_count), trainer.episode_count)

        if fast_path_total > 0:
            for reason, count in fast_path_reason_counts.items():
                trainer.writer.add_scalar(
                    f"decision/fast_path/reason_{_safe_tag(reason)}_rate",
                    float(int(count or 0)) / float(fast_path_total),
                    trainer.episode_count,
                )

    search_suffix_map = {
        "root_candidates": "root_candidates_mean",
        "root_selectable_children_mean": "root_selectable_children_mean",
        "num_simulations": "num_simulations",
        "mean_expanded_children": "expanded_children_mean",
        "mean_predicted_legal_count": "predicted_legal_mean",
        "mean_surface_keep_count": "surface_keep_mean",
        "max_search_depth": "max_depth_mean",
        "mean_leaf_depth": "leaf_depth_mean",
        "mean_concrete_leaf_depth": "concrete_leaf_depth_mean",
        "depth_ge_2_rate": "depth_ge_2_rate",
        "depth_ge_3_rate": "depth_ge_3_rate",
        "root_top1_visit_share": "root_top1_visit_share",
        "root_visit_entropy": "root_visit_entropy",
        "search_mode_direct_policy": "search_mode_direct_policy",
        "search_mode_direct_rollout_planner": "search_mode_direct_rollout_planner",
        "direct_rollout_q_mean": "direct_rollout_q_mean",
        "direct_rollout_objective_q_mean": "direct_rollout_objective_q_mean",
        "direct_rollout_risk_q_mean": "direct_rollout_risk_q_mean",
        "direct_rollout_uncertainty_mean": "direct_rollout_uncertainty_mean",
        "direct_rollout_uncertainty_bias_abs_mean": "direct_rollout_uncertainty_bias_abs_mean",
        "direct_rollout_surprise_mean": "direct_rollout_surprise_mean",
        "direct_rollout_surface_entropy_mean": "direct_rollout_surface_entropy_mean",
        "direct_rollout_latent_drift_mean": "direct_rollout_latent_drift_mean",
        "direct_rollout_branch_disagreement_mean": "direct_rollout_branch_disagreement_mean",
        "direct_rollout_steps_used": "direct_rollout_steps_used",
        "direct_rollout_branch_count_mean": "direct_rollout_branch_count_mean",
        "direct_rollout_root_valid_count": "direct_rollout_root_valid_count",
        "direct_rollout_root_bucket_size": "direct_rollout_root_bucket_size",
        "direct_rollout_bucket_padding_ratio": "direct_rollout_bucket_padding_ratio",
        "direct_rollout_max_branch_bucket_size": "direct_rollout_max_branch_bucket_size",
        "direct_rollout_branch_padding_ratio": "direct_rollout_branch_padding_ratio",
        "combat_quality_bias_applied": "combat_quality_bias_applied",
        "combat_quality_bias_abs_mean": "combat_quality_bias_abs_mean",
        "combat_quality_wasteful_end_turn_bias_applied": "combat_quality_wasteful_end_turn_bias_applied",
        "combat_quality_wasteful_end_turn_available": "combat_quality_wasteful_end_turn_available",
        "combat_quality_wasteful_end_turn_selected": "combat_quality_wasteful_end_turn_selected",
        "combat_quality_end_turn_penalty_max": "combat_quality_end_turn_penalty_max",
        "combat_quality_energy": "combat_quality_energy",
        "combat_quality_positive_action_count": "combat_quality_positive_action_count",
        "combat_quality_playable_action_count": "combat_quality_playable_action_count",
        "combat_quality_end_turn_severity": "combat_quality_end_turn_severity",
        "combat_quality_potion_available_count": "combat_quality_potion_available_count",
        "combat_quality_potion_urgent_count": "combat_quality_potion_urgent_count",
        "combat_quality_potion_low_urgency_count": "combat_quality_potion_low_urgency_count",
        "combat_quality_potion_save_recommended_count": "combat_quality_potion_save_recommended_count",
        "combat_quality_potion_no_followup_count": "combat_quality_potion_no_followup_count",
        "combat_quality_potion_lethal_count": "combat_quality_potion_lethal_count",
        "combat_quality_potion_prevent_lethal_count": "combat_quality_potion_prevent_lethal_count",
        "combat_quality_potion_mechanism_count": "combat_quality_potion_mechanism_count",
        "combat_quality_potion_overkill_count": "combat_quality_potion_overkill_count",
        "combat_quality_potion_block_waste_count": "combat_quality_potion_block_waste_count",
        "combat_quality_card_block_waste_count": "combat_quality_card_block_waste_count",
        "combat_quality_card_pure_block_count": "combat_quality_card_pure_block_count",
        "combat_quality_card_no_damage_pressure_count": "combat_quality_card_no_damage_pressure_count",
        "combat_quality_card_block_waste_bias_count": "combat_quality_card_block_waste_bias_count",
        "combat_quality_card_block_waste_bias_min": "combat_quality_card_block_waste_bias_min",
        "combat_quality_card_block_waste_hard_bias_applied": "combat_quality_card_block_waste_hard_bias_applied",
        "combat_quality_card_block_waste_progress_alternative": "combat_quality_card_block_waste_progress_alternative",
        "combat_quality_card_block_waste_progress_bonus_count": "combat_quality_card_block_waste_progress_bonus_count",
        "combat_quality_card_block_waste_progress_bonus_max": "combat_quality_card_block_waste_progress_bonus_max",
        "combat_quality_card_block_waste_selected": "combat_quality_card_block_waste_selected",
        "combat_quality_card_pure_block_selected": "combat_quality_card_pure_block_selected",
        "combat_quality_card_no_damage_pressure_selected": "combat_quality_card_no_damage_pressure_selected",
        "combat_quality_potion_use_quality_mean": "combat_quality_potion_use_quality_mean",
        "combat_quality_potion_waste_risk_mean": "combat_quality_potion_waste_risk_mean",
        "combat_quality_potion_selected": "combat_quality_potion_selected",
        "combat_quality_potion_selected_when_available": "combat_quality_potion_selected_when_available",
        "combat_quality_potion_high_urgency_selected": "combat_quality_potion_high_urgency_selected",
        "combat_quality_potion_low_urgency_selected": "combat_quality_potion_low_urgency_selected",
        "combat_quality_potion_save_recommended_selected": "combat_quality_potion_save_recommended_selected",
        "combat_quality_potion_no_followup_selected": "combat_quality_potion_no_followup_selected",
        "combat_quality_potion_lethal_selected": "combat_quality_potion_lethal_selected",
        "combat_quality_potion_prevent_lethal_selected": "combat_quality_potion_prevent_lethal_selected",
        "combat_quality_potion_mechanism_selected": "combat_quality_potion_mechanism_selected",
        "combat_quality_potion_overkill_selected": "combat_quality_potion_overkill_selected",
        "combat_quality_potion_block_waste_selected": "combat_quality_potion_block_waste_selected",
        "combat_quality_potion_use_quality_selected": "combat_quality_potion_use_quality_selected",
        "combat_quality_potion_waste_risk_selected": "combat_quality_potion_waste_risk_selected",
        "combat_quality_zero_energy_x_cost_count": "combat_quality_zero_energy_x_cost_count",
        "combat_quality_x_cost_available_count": "combat_quality_x_cost_available_count",
        "combat_quality_x_cost_selected": "combat_quality_x_cost_selected",
        "combat_quality_x_cost_selected_energy": "combat_quality_x_cost_selected_energy",
        "combat_quality_x_cost_zero_energy_selected": "combat_quality_x_cost_zero_energy_selected",
        "combat_quality_zero_energy_x_cost_selected": "combat_quality_zero_energy_x_cost_selected",
        "combat_quality_end_turn_selected": "combat_quality_end_turn_selected",
        "combat_quality_kaiser_back_attack_risk": "combat_quality_kaiser_back_attack_risk",
        "combat_quality_kaiser_defense_candidate_count": "combat_quality_kaiser_defense_candidate_count",
        "combat_quality_kaiser_defense_selected": "combat_quality_kaiser_defense_selected",
        "combat_quality_kaiser_facing_change_candidate_count": "combat_quality_kaiser_facing_change_candidate_count",
        "combat_quality_kaiser_pressure_candidate_count": "combat_quality_kaiser_pressure_candidate_count",
        "combat_quality_kaiser_facing_change_selected": "combat_quality_kaiser_facing_change_selected",
        "combat_quality_kaiser_pressure_selected": "combat_quality_kaiser_pressure_selected",
        "combat_quality_kaiser_risky_end_turn_selected": "combat_quality_kaiser_risky_end_turn_selected",
        "combat_quality_ceremonial_one_card_lock": "combat_quality_ceremonial_one_card_lock",
        "combat_quality_ceremonial_stun_window": "combat_quality_ceremonial_stun_window",
        "combat_quality_ceremonial_low_impact_count": "combat_quality_ceremonial_low_impact_count",
        "combat_quality_ceremonial_high_impact_count": "combat_quality_ceremonial_high_impact_count",
        "combat_quality_ceremonial_low_impact_selected": "combat_quality_ceremonial_low_impact_selected",
        "combat_quality_ceremonial_high_impact_selected": "combat_quality_ceremonial_high_impact_selected",
        "combat_quality_insatiable_sandpit_countdown": "combat_quality_insatiable_sandpit_countdown",
        "combat_quality_insatiable_sandpit_active": "combat_quality_insatiable_sandpit_active",
        "combat_quality_insatiable_sandpit_lt3": "combat_quality_insatiable_sandpit_lt3",
        "combat_quality_insatiable_sandpit_1": "combat_quality_insatiable_sandpit_1",
        "combat_quality_insatiable_frantic_escape_hand_count": "combat_quality_insatiable_frantic_escape_hand_count",
        "combat_quality_insatiable_frantic_escape_draw_count": "combat_quality_insatiable_frantic_escape_draw_count",
        "combat_quality_insatiable_frantic_escape_discard_count": "combat_quality_insatiable_frantic_escape_discard_count",
        "combat_quality_insatiable_frantic_escape_exhaust_count": "combat_quality_insatiable_frantic_escape_exhaust_count",
        "combat_quality_insatiable_frantic_escape_total_count": "combat_quality_insatiable_frantic_escape_total_count",
        "combat_quality_insatiable_frantic_escape_candidate_count": "combat_quality_insatiable_frantic_escape_candidate_count",
        "combat_quality_insatiable_frantic_escape_available": "combat_quality_insatiable_frantic_escape_available",
        "combat_quality_insatiable_frantic_escape_urgency": "combat_quality_insatiable_frantic_escape_urgency",
        "combat_quality_insatiable_frantic_escape_bonus_applied": "combat_quality_insatiable_frantic_escape_bonus_applied",
        "combat_quality_insatiable_non_escape_at1_penalty_count": "combat_quality_insatiable_non_escape_at1_penalty_count",
        "combat_quality_insatiable_escape_cycle_risk": "combat_quality_insatiable_escape_cycle_risk",
        "combat_quality_insatiable_lethal_candidate_count": "combat_quality_insatiable_lethal_candidate_count",
        "combat_quality_insatiable_frantic_escape_selected": "combat_quality_insatiable_frantic_escape_selected",
        "combat_quality_insatiable_frantic_escape_missed_lt3": "combat_quality_insatiable_frantic_escape_missed_lt3",
        "combat_quality_insatiable_frantic_escape_missed_at_1": "combat_quality_insatiable_frantic_escape_missed_at_1",
        "combat_quality_insatiable_non_escape_at_1_selected": "combat_quality_insatiable_non_escape_at_1_selected",
        "semantic_switch_rate": "semantic_switch_rate",
        "semantic_expansion_rate": "semantic_expansion_rate",
        "semantic_chain_steps_mean": "semantic_chain_steps_mean",
        "semantic_drill_rate": "semantic_drill_rate",
        "root_bias_scale": "root_bias_scale",
        "root_bias_nonzero": "root_bias_nonzero_rate",
        "root_bias_abs_mean": "root_bias_abs_mean",
        "root_bias_max_abs": "root_bias_max_abs",
        "root_bias_changed_top1": "root_bias_changed_top1_rate",
        "root_bias_selected_action_delta": "root_bias_selected_action_delta_mean",
        "root_bias_suppressed_by_gate": "root_bias_suppressed_by_gate_rate",
        "root_bias_scale_effective": "root_bias_scale_effective_mean",
        "objective_weight_survival": "objective_weight_survival",
        "objective_weight_hp": "objective_weight_hp",
        "objective_weight_build": "objective_weight_build",
        "objective_weight_resource": "objective_weight_resource",
        "root_objective_value": "root_objective_value",
        "end_turn_bias_applied": "end_turn_bias_applied",
        "end_turn_guard_applied": "end_turn_guard_applied",
        "end_turn_guard_forced_alternative": "end_turn_guard_forced_alternative",
        "zero_energy_x_cost_guard_applied": "zero_energy_x_cost_guard_applied",
        "zero_energy_x_cost_guard_forced_alternative": "zero_energy_x_cost_guard_forced_alternative",
        "objective_prior_applied": "objective_prior_applied",
        # Phase 3 (recovery 2026-05-09): route-heuristic prior bias.
        "route_heuristic_bias_applied": "route_heuristic_bias_applied_rate",
        "route_heuristic_bias_abs_mean": "route_heuristic_bias_abs_mean",
        "route_heuristic_bias_max_abs": "route_heuristic_bias_max_abs",
        "route_safety_guard_enabled": "route_safety_guard_enabled",
        "route_safety_guard_applicable": "route_safety_guard_applicable_rate",
        "route_safety_guard_safe_available": "route_safety_guard_safe_available_rate",
        "route_safety_guard_lower_risk_available": "route_safety_guard_lower_risk_available_rate",
        "route_safety_guard_applied": "route_safety_guard_applied_rate",
        "route_safety_guard_override": "route_safety_guard_override_rate",
        "route_safety_guard_alignment_error": "route_safety_guard_alignment_error_rate",
        "route_safety_guard_selected_risk_class": "route_safety_guard_selected_risk_class_mean",
        "route_safety_guard_final_risk_class": "route_safety_guard_final_risk_class_mean",
        "route_safety_guard_selected_forced_elite": "route_safety_guard_selected_forced_elite_rate",
        "route_safety_guard_selected_immediate_elite": "route_safety_guard_selected_immediate_elite_rate",
        "route_safety_guard_low_hp_forced": "route_safety_guard_low_hp_forced_rate",
        "route_safety_guard_invalid_obs": "route_safety_guard_invalid_obs_rate",
        "build_hard_guard_policy_full": "build_hard_guard_policy_full",
        "build_hard_guard_policy_emergency": "build_hard_guard_policy_emergency",
        "build_hard_guard_policy_off": "build_hard_guard_policy_off",
        "build_safety_guard_enabled": "build_safety_guard_enabled",
        "build_safety_guard_rest_low_hp_applicable": "build_safety_guard_rest_low_hp_applicable_rate",
        "build_safety_guard_rest_available": "build_safety_guard_rest_available_rate",
        "build_safety_guard_rest_applied": "build_safety_guard_rest_applied_rate",
        "build_safety_guard_rest_override": "build_safety_guard_rest_override_rate",
        "build_safety_guard_rest_selected_non_heal_low_hp": "build_safety_guard_rest_selected_non_heal_low_hp_rate",
        "build_safety_guard_rest_selected_heal": "build_safety_guard_rest_selected_heal_rate",
        "build_safety_guard_invalid_obs": "build_safety_guard_invalid_obs_rate",
        "build_safety_guard_alignment_error": "build_safety_guard_alignment_error_rate",
        "build_safety_guard_hp_ratio": "build_safety_guard_hp_ratio_mean",
        "build_safety_guard_hp_threshold": "build_safety_guard_hp_threshold",
        "combat_grounded_root_enabled": "combat_grounded_root_enabled",
        "q_value_ucb_enabled": "q_value_ucb_enabled",
    }
    search_suffix_map.update(CARD_REWARD_GUARD_SEARCH_SUFFIXES)
    search_suffix_map.update(CARD_REWARD_PICK_QUALITY_GUARD_SEARCH_SUFFIXES)
    search_suffix_map.update(SHOP_ACTION_GUARD_SEARCH_SUFFIXES)
    search_suffix_map.update(POST_SEARCH_HARD_GUARD_SEARCH_SUFFIXES)
    search_suffix_map.update(REST_SITE_SMITH_GUARD_SEARCH_SUFFIXES)
    search_suffix_map.update(SUMMONER_TARGET_GUARD_SEARCH_SUFFIXES)
    search_suffix_map.update(TARGET_PRIORITY_GUARD_SEARCH_SUFFIXES)
    search_suffix_map.update(COMBAT_QUALITY_CARD_BLOCK_SEARCH_SUFFIXES)
    search_suffix_map.update(COMBAT_QUALITY_GUARD_SEARCH_SUFFIXES)
    domain_search_means = episode_metrics.get("domain_search_means") if isinstance(episode_metrics.get("domain_search_means"), dict) else {}
    for domain, stats in domain_search_means.items():
        if not isinstance(stats, dict):
            continue
        safe_domain = _safe_tag(domain)
        for stat_key, value in stats.items():
            suffix = search_suffix_map.get(str(stat_key), _safe_tag(stat_key))
            trainer.writer.add_scalar(f"search/{safe_domain}/{suffix}", _to_float(value), trainer.episode_count)

    domain_family_rates = episode_metrics.get("domain_family_rates") if isinstance(episode_metrics.get("domain_family_rates"), dict) else {}
    for domain, rates in domain_family_rates.items():
        if not isinstance(rates, dict):
            continue
        safe_domain = _safe_tag(domain)
        for family, value in rates.items():
            trainer.writer.add_scalar(f"decision/{safe_domain}/family_{_safe_tag(family)}_rate", _to_float(value), trainer.episode_count)

    combat_quality_diagnostics = (
        episode_metrics.get("combat_quality_diagnostics")
        if isinstance(episode_metrics.get("combat_quality_diagnostics"), dict)
        else {}
    )
    for tag, value in combat_quality_diagnostics.items():
        trainer.writer.add_scalar(str(tag), _to_float(value), trainer.episode_count)

    boss_diagnostics = episode_metrics.get("boss_diagnostics") if isinstance(episode_metrics.get("boss_diagnostics"), dict) else {}
    for tag, value in boss_diagnostics.items():
        trainer.writer.add_scalar(str(tag), _to_float(value), trainer.episode_count)

    # Async actors collect with NullSummaryWriter, so episode-level deck/build
    # diagnostics must be replayed here too.  Keep this in lock-step with the
    # synchronous writer in self_play.py; otherwise multi-actor runs lose the
    # exact metrics we need for Act1 debugging (reward skip, shop use, death
    # deck quality).
    deck_quality = episode_metrics.get("deck_quality_v2")
    if isinstance(deck_quality, dict):
        for tag_suffix, quality_key in FINAL_DECK_QUALITY_TB_KEYS:
            trainer.writer.add_scalar(
                f"deck/final_{tag_suffix}",
                _to_float(deck_quality.get(quality_key)),
                trainer.episode_count,
            )

    card_reward_metrics = episode_metrics.get("card_reward_metrics")
    if isinstance(card_reward_metrics, dict):
        for tag_suffix, meta_key in CARD_REWARD_TB_KEYS:
            trainer.writer.add_scalar(
                f"build/card_reward_{tag_suffix}",
                _to_float(card_reward_metrics.get(meta_key)),
                trainer.episode_count,
            )

    shop_metrics = episode_metrics.get("shop_metrics")
    if isinstance(shop_metrics, dict):
        for tag_suffix, meta_key in SHOP_TB_KEYS:
            trainer.writer.add_scalar(
                f"build/shop_{tag_suffix}",
                _to_float(shop_metrics.get(meta_key)),
                trainer.episode_count,
            )

    rest_site_metrics = episode_metrics.get("rest_site_metrics")
    if isinstance(rest_site_metrics, dict):
        for tag_suffix, meta_key in REST_SITE_TB_KEYS:
            trainer.writer.add_scalar(
                f"build/rest_site_{tag_suffix}",
                _to_float(rest_site_metrics.get(meta_key)),
                trainer.episode_count,
            )

    deck_upgrade_metrics = episode_metrics.get("deck_upgrade_metrics")
    if isinstance(deck_upgrade_metrics, dict):
        for tag_suffix, meta_key in DECK_UPGRADE_TB_KEYS:
            trainer.writer.add_scalar(
                f"build/deck_upgrade_{tag_suffix}",
                _to_float(deck_upgrade_metrics.get(meta_key)),
                trainer.episode_count,
            )

    summoner_targeting_metrics = episode_metrics.get("summoner_targeting_metrics")
    if isinstance(summoner_targeting_metrics, dict):
        for tag_suffix, meta_key in SUMMONER_TARGETING_TB_KEYS:
            trainer.writer.add_scalar(
                f"combat/summoner_targeting_{tag_suffix}",
                _to_float(summoner_targeting_metrics.get(meta_key)),
                trainer.episode_count,
            )

    target_priority_metrics = episode_metrics.get("target_priority_metrics")
    if isinstance(target_priority_metrics, dict):
        for tag_suffix, meta_key in TARGET_PRIORITY_TB_KEYS:
            trainer.writer.add_scalar(
                f"combat/target_priority_{tag_suffix}",
                _to_float(target_priority_metrics.get(meta_key)),
                trainer.episode_count,
            )

    if _to_float(episode_metrics.get("death_floor"), 0.0) > 0.0 and isinstance(deck_quality, dict):
        for tag_suffix, quality_key in DEATH_DECK_QUALITY_TB_KEYS:
            trainer.writer.add_scalar(
                f"death_deck/{tag_suffix}",
                _to_float(deck_quality.get(quality_key)),
                trainer.episode_count,
            )

    trainer.writer.add_scalar(f"env/{actor_index}_episodes", float(actor_completed_episodes[actor_index]), trainer.total_steps)
