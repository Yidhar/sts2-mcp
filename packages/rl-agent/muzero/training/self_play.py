"""Self-play episode orchestration for MuZeroTrainer.

This module owns rollout collection only.  Tactical policy heuristics, route
heuristics, diagnostics, and checkpoint/loss code live in sibling packages so the
legacy ``muzero.train`` entrypoint does not grow again.
"""

from __future__ import annotations

from collections import defaultdict
import time
from typing import Any

import gymnasium as gym
import numpy as np
import torch

from combat_snapshot_dataset import infer_encounter_tier
from muzero.diagnostics.deck_build_metrics import (
    CARD_REWARD_TB_KEYS,
    DEATH_DECK_QUALITY_TB_KEYS,
    FINAL_DECK_QUALITY_TB_KEYS,
    CardRewardEpisodeTracker,
    compact_deck_cards,
    compute_deck_quality_summary,
    extract_deck_cards_from_obs_like,
)
from muzero.diagnostics.shop_metrics import (
    SHOP_TB_KEYS,
    ShopEpisodeTracker,
    build_shop_choice_payload,
    dump_shop_choice_diagnostic,
)
from muzero.diagnostics.rest_site_metrics import (
    REST_SITE_TB_KEYS,
    RestSiteEpisodeTracker,
    build_rest_site_choice_payload,
    dump_rest_site_choice_diagnostic,
)
from muzero.diagnostics.deck_upgrade_metrics import (
    DECK_UPGRADE_TB_KEYS,
    DeckUpgradeEpisodeTracker,
    build_deck_upgrade_choice_payload,
    build_smith_upgrade_transition_payload,
    complete_deck_upgrade_choice_payload,
    dump_deck_upgrade_choice_diagnostic,
    dump_smith_upgrade_transition_diagnostic,
)
from muzero.diagnostics.summoner_targeting import (
    SUMMONER_TARGETING_TB_KEYS,
    SummonerTargetingEpisodeTracker,
    build_summoner_targeting_payload,
    dump_summoner_targeting_diagnostic,
)
from muzero.diagnostics.target_priority import (
    TARGET_PRIORITY_TB_KEYS,
    TargetPriorityEpisodeTracker,
    build_target_priority_payload,
    dump_target_priority_diagnostic,
)
from muzero.diagnostics.intent_combat_quality import (
    INTENT_COMBAT_QUALITY_TB_KEYS,
    IntentCombatQualityEpisodeTracker,
    build_intent_combat_quality_payload,
    dump_intent_combat_quality_diagnostic,
)
from muzero.diagnostics.end_turn_pre_dispatch import dump_end_turn_pre_dispatch_audit
from muzero.sts2_env.muzero_buffer import GameTrajectory, MuZeroReplayBuffer
from muzero.training.action_hard_guard_dispatch import ActionHardGuardDispatchMixin
from muzero.training.checkpointing import dict_obs_to_torch
from muzero.training.card_reward_guard import CARD_REWARD_GUARD_SEARCH_SUFFIXES
from muzero.training.card_reward_pick_quality_guard import CARD_REWARD_PICK_QUALITY_GUARD_SEARCH_SUFFIXES
from muzero.training.decision_constants import TRIVIAL_BUILD_FAST_PATH_REASONS
from muzero.training.async_telemetry import log_environment_episode_telemetry
from muzero.training.post_search_policy_retarget import (
    POST_SEARCH_HARD_GUARD_SEARCH_SUFFIXES,
    annotate_card_reward_final_selection,
    resolve_hard_guard_policy_target,
)
from muzero.training.rest_site_smith_guard import REST_SITE_SMITH_GUARD_SEARCH_SUFFIXES
from muzero.training.shop_action_guard import SHOP_ACTION_GUARD_SEARCH_SUFFIXES
from muzero.training.route_heuristic_telemetry import RouteHeuristicTelemetryMixin
from muzero.training.self_play_diagnostics import SelfPlayDiagnosticsMixin
from muzero.combat_quality import COMBAT_QUALITY_GUARD_SEARCH_SUFFIXES
from muzero.combat_quality.summoner_target_guard import SUMMONER_TARGET_GUARD_SEARCH_SUFFIXES
from muzero.combat_quality.target_priority_guard import TARGET_PRIORITY_GUARD_SEARCH_SUFFIXES
from sts2_env.boss_mechanics import build_boss_mechanics_context
from sts2_env.objective_heads import NUM_OBJECTIVE_HEADS, compute_transition_objective_rewards
from sts2_env.observation_v2 import DECISION_DOMAINS, MAX_ACTIONS
from muzero.sts2_env.semantic_rollout import aggregate_concrete_policy_to_semantic, semantic_rollout_index
from muzero.training.action_diagnostics_merge import merge_action_diagnostics_into_search_stats


class SelfPlayMixin(SelfPlayDiagnosticsMixin, ActionHardGuardDispatchMixin, RouteHeuristicTelemetryMixin):
    """Self-play rollout methods mixed into MuZeroTrainer."""


    def _obs_list_to_torch(self, obs_list: list[dict[str, Any]]) -> dict[str, torch.Tensor]:
        """Convert a list of dict observations into a batched torch dict."""
        obs_torch_batch: dict[str, torch.Tensor] = {}
        obs_numpy_batch = MuZeroReplayBuffer.batch_observations(obs_list)
        for key, value in obs_numpy_batch.items():
            obs_torch_batch[key] = torch.from_numpy(value).to(self.device)
        return obs_torch_batch

    def compute_temperature(self, step: int, total_steps: int) -> float:
        """Compute temperature schedule: linear decay from 1.0 to 0.1."""
        progress = min(step / max(total_steps, 1), 1.0)
        return 1.0 - progress * 0.9

    def self_play_episode_on_env(self, env: gym.Env, *, temperature: float = 1.0) -> tuple[float, int]:
        """Run one self-play episode on the provided environment.

        This is used by the combat-sandbox multi-env harness so the trainer can
        round-robin across multiple live bridge sessions without changing the
        full-run single-env path.
        """
        previous_env = self.env
        self.env = env
        try:
            return self.self_play_episode(temperature=temperature)
        finally:
            self.env = previous_env

    def self_play_episode(self, temperature: float = 1.0) -> tuple[float, int]:
        """Run one self-play episode using MCTS action selection.

        Args:
            temperature: Temperature for action selection.

        Returns:
            (episode_reward, episode_length)
        """
        obs, info = self.env.reset()
        initial_info = dict(info)
        trajectory = GameTrajectory()

        episode_reward = 0.0
        episode_length = 0
        progress_snapshots: list[dict[str, Any]] = []
        seen_floors: set[int] = set()
        elite_floors: set[int] = set()
        boss_floors: set[int] = set()
        death_floor = 0.0
        used_potion_count_episode = 0
        used_potion_count_by_encounter: defaultdict[str, int] = defaultdict(int)
        used_potion_count_by_room: defaultdict[str, int] = defaultdict(int)
        potion_transition_records_episode: list[dict[str, Any]] = []
        search_root_candidates: list[float] = []
        search_root_selectable_children: list[float] = []
        search_expanded_children: list[float] = []
        search_predicted_legal: list[float] = []
        search_surface_keep: list[float] = []
        search_max_depths: list[float] = []
        search_depths: list[float] = []
        search_concrete_depths: list[float] = []
        search_depth_ge_2: list[float] = []
        search_depth_ge_3: list[float] = []
        search_semantic_expansion: list[float] = []
        search_semantic_switch: list[float] = []
        search_semantic_chain_steps: list[float] = []
        search_semantic_drill: list[float] = []
        search_root_top1_visit_share: list[float] = []
        search_root_visit_entropy: list[float] = []
        search_num_simulations: list[float] = []
        search_root_bias_scale: list[float] = []
        search_root_bias_nonzero: list[float] = []
        search_root_bias_abs_mean: list[float] = []
        search_root_bias_max_abs: list[float] = []
        search_root_bias_changed_top1: list[float] = []
        search_root_bias_selected_delta: list[float] = []
        search_root_bias_suppressed_by_gate: list[float] = []
        search_root_bias_scale_effective: list[float] = []
        search_end_turn_guard_applied: list[float] = []
        search_end_turn_guard_forced_alt: list[float] = []
        domain_search_stats: dict[str, dict[str, list[float]]] = {
            domain: defaultdict(list) for domain in DECISION_DOMAINS
        }
        decision_counts: dict[str, int] = {domain: 0 for domain in DECISION_DOMAINS}
        fast_path_counts: dict[str, int] = {domain: 0 for domain in DECISION_DOMAINS}
        fast_path_reason_counts: dict[str, int] = defaultdict(int)
        selected_family_counts: dict[str, dict[str, int]] = {
            domain: defaultdict(int) for domain in DECISION_DOMAINS
        }
        domain_override_counts: dict[str, int] = defaultdict(int)
        combat_like_decision_count = 0
        direct_policy_eligible_count = 0
        direct_policy_used_count = 0
        card_reward_tracker = CardRewardEpisodeTracker()
        shop_tracker = ShopEpisodeTracker()
        rest_site_tracker = RestSiteEpisodeTracker()
        deck_upgrade_tracker = DeckUpgradeEpisodeTracker()
        summoner_targeting_tracker = SummonerTargetingEpisodeTracker()
        target_priority_tracker = TargetPriorityEpisodeTracker()
        intent_combat_quality_tracker = IntentCombatQualityEpisodeTracker()
        route_dry_run_records: list[dict[str, Any]] = []
        route_dry_run_error_count = 0
        terminated = False
        truncated = False
        initial_progress = self._episode_progress_snapshot(initial_info)
        if initial_progress:
            progress_snapshots.append(initial_progress)

        while True:
            prev_info = dict(info)
            action_mask = info.get("action_mask", np.ones(MAX_ACTIONS, dtype=bool))
            legal_actions = (
                info.get("legal_actions_compact")
                or info.get("legal_actions")
                or getattr(self.env.unwrapped, "get_compact_legal_actions", lambda: [])()
            )
            planner_context = prev_info.get("planner_context") if isinstance(prev_info.get("planner_context"), dict) else {}
            semantic_actions = planner_context.get("semantic_actions") if isinstance(planner_context.get("semantic_actions"), list) else []
            decision_domain, encoded_decision_domain, domain_overridden = self._resolve_acting_decision_domain(
                obs,
                legal_actions,
                prev_info,
            )
            if decision_domain not in decision_counts:
                decision_domain = "build"
            if decision_domain == "combat":
                combat_like_decision_count += 1
                obs = self._with_decision_domain(obs, "combat") or obs
            if domain_overridden:
                domain_override_counts[f"{encoded_decision_domain}->{decision_domain}"] += 1
            decision_counts[decision_domain] += 1
            num_simulations = self._num_simulations_for_domain(decision_domain)
            _lat_legal_count = int((np.asarray(action_mask, dtype=np.float32) > 0).sum())
            _lat_t0 = time.perf_counter()
            obs_batch = dict_obs_to_torch(obs, device=self.device)
            _lat_obs_ms = (time.perf_counter() - _lat_t0) * 1000.0
            _lat_t0 = time.perf_counter()
            with torch.no_grad(), self._amp_autocast():
                initial = self.network.initial_inference(obs_batch)
                root_value = initial.value.squeeze(0).item()
                root_value_components = initial.value_components.squeeze(0).detach().cpu().numpy()
            _lat_initial_ms = (time.perf_counter() - _lat_t0) * 1000.0
            if self.combat_direct_policy and decision_domain == "combat":
                direct_policy_eligible_count += 1

            fast_path_choice = self._trivial_build_fast_path_choice(
                decision_domain=decision_domain,
                action_mask=np.asarray(action_mask, dtype=np.float32),
                legal_actions=legal_actions,
            )
            if fast_path_choice is None:
                fast_path_choice = self._combat_card_selection_fast_path_choice(
                    decision_domain=decision_domain,
                    action_mask=np.asarray(action_mask, dtype=np.float32),
                    legal_actions=legal_actions,
                    policy_logits=initial.policy_logits.squeeze(0),
                )
            fast_path_reason = ""
            if fast_path_choice is not None:
                action_idx = int(fast_path_choice["action_idx"])
                fast_path_reason = str(fast_path_choice.get("reason") or "").strip().lower()
                search_policy = np.zeros(MAX_ACTIONS, dtype=np.float32)
                if 0 <= action_idx < MAX_ACTIONS:
                    search_policy[action_idx] = 1.0
                fast_path_counts[decision_domain] = fast_path_counts.get(decision_domain, 0) + 1
                if fast_path_reason:
                    fast_path_reason_counts[fast_path_reason] += 1
                search_stats: dict[str, Any] = {}
            elif self.combat_direct_policy and decision_domain == "combat":
                action_mask_tensor = torch.as_tensor(
                    action_mask,
                    device=initial.policy_logits.device,
                    dtype=initial.policy_logits.dtype,
                )
                masked_logits = initial.policy_logits.squeeze(0).detach().clone()
                rollout_action_mask = action_mask_tensor.unsqueeze(0) > 0
                direct_policy_used_count += 1
                with torch.no_grad(), self._amp_autocast():
                    rollout = self.network.action_rollout_planner(
                        initial.hidden_state,
                        initial.action_embeddings,
                        action_mask=rollout_action_mask,
                        objective_context=obs_batch.get("objective_context"),
                        decision_domain=obs_batch.get("decision_domain"),
                        discount=self.discount,
                        rollout_steps=self.combat_rollout_steps,
                        continuation_beam_width=self.combat_rollout_beam_width,
                        continuation_legal_logit_scale=self.combat_rollout_legal_logit_scale,
                        uncertainty_surprise_weight=self.combat_rollout_uncertainty_surprise_weight,
                        uncertainty_surface_entropy_weight=self.combat_rollout_uncertainty_surface_weight,
                        uncertainty_latent_drift_weight=self.combat_rollout_uncertainty_latent_weight,
                        uncertainty_branch_disagreement_weight=self.combat_rollout_uncertainty_disagreement_weight,
                        continuation_uncertainty_penalty=self.combat_rollout_continuation_uncertainty_penalty,
                    )
                rollout_q = rollout.planner_q.squeeze(0).detach()
                rollout_objective_q = rollout.planner_objective_q.squeeze(0).detach()
                rollout_risk_q = rollout.planner_risk_q.squeeze(0).detach()
                rollout_uncertainty = rollout.planner_uncertainty.squeeze(0).detach()
                rollout_surprise = rollout.planner_surprise.squeeze(0).detach()
                rollout_surface_entropy = rollout.planner_surface_entropy.squeeze(0).detach()
                rollout_latent_drift = rollout.planner_latent_drift.squeeze(0).detach()
                rollout_branch_disagreement = rollout.planner_branch_disagreement.squeeze(0).detach()
                rollout_mask = rollout.action_mask.squeeze(0)
                rollout_q_bias = self._normalize_action_values(
                    rollout_q.unsqueeze(0),
                    rollout_mask.unsqueeze(0),
                ).squeeze(0)
                rollout_objective_q_bias = self._normalize_action_values(
                    rollout_objective_q.unsqueeze(0),
                    rollout_mask.unsqueeze(0),
                ).squeeze(0)
                rollout_risk_q_bias = self._normalize_action_values(
                    rollout_risk_q.unsqueeze(0),
                    rollout_mask.unsqueeze(0),
                ).squeeze(0)
                rollout_uncertainty_bias = self._normalize_action_values(
                    rollout_uncertainty.unsqueeze(0),
                    rollout_mask.unsqueeze(0),
                ).squeeze(0)
                # The supervised surprise/uncertainty heads are raw-scale signals; in boss fights
                # their absolute values can sit around 50+ while Q is around [-3, 1].  We only
                # want uncertainty to rank actions relative to siblings, not dominate policy logits
                # through an outlier z-score, so all rollout side-channel biases are clipped to
                # a bounded logit prior range before blending.
                rollout_q_bias = rollout_q_bias.clamp(-2.5, 2.5)
                rollout_objective_q_bias = rollout_objective_q_bias.clamp(-2.5, 2.5)
                rollout_risk_q_bias = rollout_risk_q_bias.clamp(-2.5, 2.5)
                rollout_uncertainty_bias = rollout_uncertainty_bias.clamp(-2.0, 2.0)
                masked_logits = (
                    masked_logits
                    + self.combat_rollout_q_blend * rollout_q_bias
                    + self.combat_rollout_objective_q_blend * rollout_objective_q_bias
                    + self.combat_rollout_risk_blend * rollout_risk_q_bias
                    - self.combat_rollout_uncertainty_blend * rollout_uncertainty_bias
                )
                # Root-quality bias instrumentation: keep the exact pre-bias
                # root logits so TensorBoard can prove whether the hard
                # mechanism prior (Kaiser facing, end_turn, X-cost, potion
                # timing, HP-cost, etc.) actually changes the root ordering.
                pre_quality_masked_logits = masked_logits.masked_fill(action_mask_tensor <= 0, -1e9)
                quality_bias_np, quality_stats, zero_energy_x_indices = self._combat_action_quality_bias(
                    obs,
                    np.asarray(action_mask, dtype=np.float32),
                    legal_actions if isinstance(legal_actions, list) else [],
                )
                quality_bias = torch.as_tensor(
                    quality_bias_np,
                    device=masked_logits.device,
                    dtype=masked_logits.dtype,
                )
                masked_logits = masked_logits + quality_bias
                masked_logits = masked_logits.masked_fill(action_mask_tensor <= 0, -1e9)
                legal_root_bias = quality_bias.masked_select(action_mask_tensor > 0)
                root_bias_nonzero = (
                    1.0
                    if legal_root_bias.numel() > 0 and bool(torch.any(legal_root_bias.abs() > 1e-6).item())
                    else 0.0
                )
                if bool((action_mask_tensor > 0).any().item()):
                    pre_quality_top1 = int(pre_quality_masked_logits.argmax().item())
                    post_quality_top1 = int(masked_logits.argmax().item())
                else:
                    pre_quality_top1 = -1
                    post_quality_top1 = -1
                root_bias_abs_mean = float(legal_root_bias.abs().mean().item()) if legal_root_bias.numel() > 0 else 0.0
                root_bias_max_abs = float(legal_root_bias.abs().max().item()) if legal_root_bias.numel() > 0 else 0.0
                root_bias_changed_top1 = 1.0 if pre_quality_top1 >= 0 and pre_quality_top1 != post_quality_top1 else 0.0
                safe_temperature = max(float(temperature), 1e-3)
                direct_probs = torch.softmax(masked_logits / safe_temperature, dim=0)
                if not torch.isfinite(direct_probs).all() or float(direct_probs.sum().item()) <= 0.0:
                    legal_mask = action_mask_tensor > 0
                    direct_probs = legal_mask.float()
                    direct_probs = direct_probs / direct_probs.sum().clamp(min=1.0)
                direct_probs = direct_probs / direct_probs.sum().clamp(min=1e-8)
                if safe_temperature <= 0.05:
                    action_idx = int(direct_probs.argmax().item())
                else:
                    action_idx = int(torch.multinomial(direct_probs, 1).item())
                search_policy = direct_probs.detach().cpu().numpy().astype(np.float32)
                entropy = -(direct_probs.clamp(min=1e-8) * direct_probs.clamp(min=1e-8).log()).sum().item()
                rollout_valid_count = rollout_mask.float().sum().clamp(min=1.0)
                rollout_q_mean = torch.where(rollout_mask, rollout_q, torch.zeros_like(rollout_q)).sum() / rollout_valid_count
                rollout_objective_q_mean = (
                    torch.where(rollout_mask, rollout_objective_q, torch.zeros_like(rollout_objective_q)).sum()
                    / rollout_valid_count
                )
                rollout_risk_q_mean = (
                    torch.where(rollout_mask, rollout_risk_q, torch.zeros_like(rollout_risk_q)).sum()
                    / rollout_valid_count
                )
                rollout_uncertainty_mean = (
                    torch.where(rollout_mask, rollout_uncertainty, torch.zeros_like(rollout_uncertainty)).sum()
                    / rollout_valid_count
                )
                rollout_surprise_mean = (
                    torch.where(rollout_mask, rollout_surprise, torch.zeros_like(rollout_surprise)).sum()
                    / rollout_valid_count
                )
                rollout_surface_entropy_mean = (
                    torch.where(rollout_mask, rollout_surface_entropy, torch.zeros_like(rollout_surface_entropy)).sum()
                    / rollout_valid_count
                )
                rollout_latent_drift_mean = (
                    torch.where(rollout_mask, rollout_latent_drift, torch.zeros_like(rollout_latent_drift)).sum()
                    / rollout_valid_count
                )
                rollout_branch_disagreement_mean = (
                    torch.where(rollout_mask, rollout_branch_disagreement, torch.zeros_like(rollout_branch_disagreement)).sum()
                    / rollout_valid_count
                )
                rollout_uncertainty_bias_abs_mean = (
                    torch.where(rollout_mask, rollout_uncertainty_bias.abs(), torch.zeros_like(rollout_uncertainty_bias)).sum()
                    / rollout_valid_count
                )
                selected_ceremonial_low_impact = 0.0
                selected_ceremonial_high_impact = 0.0
                if (
                    isinstance(legal_actions, list)
                    and 0 <= int(action_idx) < len(legal_actions)
                    and quality_stats.get("combat_quality_ceremonial_one_card_lock", 0.0) > 0.05
                ):
                    low_impact, high_impact = self._ceremonial_action_timing_flags(
                        legal_actions[int(action_idx)],
                        int(action_idx),
                        obs,
                        self._current_raw_combat_obs(),
                        legal_actions,
                        np.asarray(action_mask, dtype=np.float32).reshape(-1),
                        self._combat_energy(obs, self._current_raw_combat_obs()),
                    )
                    selected_ceremonial_low_impact = 1.0 if low_impact else 0.0
                    selected_ceremonial_high_impact = 1.0 if high_impact else 0.0
                raw_for_selected = self._current_raw_combat_obs()
                selected_is_kaiser_encounter = False
                try:
                    if isinstance(raw_for_selected, dict):
                        selected_is_kaiser_encounter = self._is_kaiser_encounter_context(
                            build_boss_mechanics_context(raw_for_selected),
                            raw_for_selected,
                        )
                except Exception:
                    selected_is_kaiser_encounter = False
                search_stats = {
                    "num_simulations": 0.0,
                    "root_top1_visit_share": float(direct_probs.max().item()),
                    "root_visit_entropy": float(entropy),
                    "mean_predicted_legal_count": float((action_mask_tensor > 0).float().sum().item()),
                    "search_mode_direct_policy": 1.0,
                    "search_mode_direct_rollout_planner": 1.0,
                    "direct_rollout_q_mean": float(rollout_q_mean.item()),
                    "direct_rollout_objective_q_mean": float(rollout_objective_q_mean.item()),
                    "direct_rollout_risk_q_mean": float(rollout_risk_q_mean.item()),
                    "direct_rollout_uncertainty_mean": float(rollout_uncertainty_mean.item()),
                    "direct_rollout_uncertainty_bias_abs_mean": float(rollout_uncertainty_bias_abs_mean.item()),
                    "direct_rollout_surprise_mean": float(rollout_surprise_mean.item()),
                    "direct_rollout_surface_entropy_mean": float(rollout_surface_entropy_mean.item()),
                    "direct_rollout_latent_drift_mean": float(rollout_latent_drift_mean.item()),
                    "direct_rollout_branch_disagreement_mean": float(rollout_branch_disagreement_mean.item()),
                    "direct_rollout_steps_used": float(rollout.rollout_steps_used),
                    "direct_rollout_branch_count_mean": float(rollout.rollout_branch_count_mean),
                    "direct_rollout_root_valid_count": float(rollout.rollout_root_valid_count),
                    "direct_rollout_root_bucket_size": float(rollout.rollout_root_bucket_size),
                    "direct_rollout_bucket_padding_ratio": float(rollout.rollout_bucket_padding_ratio),
                    "direct_rollout_max_branch_bucket_size": float(rollout.rollout_max_branch_bucket_size),
                    "direct_rollout_branch_padding_ratio": float(rollout.rollout_branch_padding_ratio),
                    # Direct-rollout mode does not go through MCTS
                    # ``_objective_prior_bias``; the equivalent root prior is
                    # ``_combat_action_quality_bias``.  Report it under the
                    # root_bias namespace too, otherwise search/root_bias_scale
                    # looks permanently zero in search-free training even when
                    # the prior is actively changing logits.
                    "root_bias_scale": 1.0,
                    "root_bias_nonzero": float(root_bias_nonzero),
                    "root_bias_abs_mean": float(root_bias_abs_mean),
                    "root_bias_max_abs": float(root_bias_max_abs),
                    "root_bias_changed_top1": float(root_bias_changed_top1),
                    "root_bias_selected_action_delta": float(
                        quality_bias_np[int(action_idx)]
                        if 0 <= int(action_idx) < min(len(quality_bias_np), MAX_ACTIONS)
                        else 0.0
                    ),
                    "root_bias_suppressed_by_gate": 0.0,
                    "root_bias_scale_effective": 1.0,
                    **quality_stats,
                    "combat_quality_zero_energy_x_cost_selected": 1.0 if int(action_idx) in zero_energy_x_indices else 0.0,
                    "combat_quality_end_turn_selected": 1.0 if self._semantic_family(legal_actions[int(action_idx)] if isinstance(legal_actions, list) and 0 <= int(action_idx) < len(legal_actions) else {}) == "end_turn" else 0.0,
                    "combat_quality_wasteful_end_turn_selected": (
                        1.0
                        if (
                            self._semantic_family(legal_actions[int(action_idx)] if isinstance(legal_actions, list) and 0 <= int(action_idx) < len(legal_actions) else {}) == "end_turn"
                            and bool(quality_stats.get("combat_quality_wasteful_end_turn_available", 0.0) > 0.5)
                        )
                        else 0.0
                    ),
                    "combat_quality_kaiser_defense_selected": (
                        1.0
                        if (
                            selected_is_kaiser_encounter
                            and
                            isinstance(legal_actions, list)
                            and 0 <= int(action_idx) < len(legal_actions)
                            and quality_stats.get("combat_quality_kaiser_back_attack_risk", 0.0) > 0.05
                            and self._is_kaiser_risk_handling_action(legal_actions[int(action_idx)], raw_for_selected)
                        )
                        else 0.0
                    ),
                    "combat_quality_kaiser_facing_change_selected": (
                        1.0
                        if (
                            selected_is_kaiser_encounter
                            and
                            isinstance(legal_actions, list)
                            and 0 <= int(action_idx) < len(legal_actions)
                            and quality_stats.get("combat_quality_kaiser_back_attack_risk", 0.0) > 0.05
                            and self._is_kaiser_facing_change_action(legal_actions[int(action_idx)], raw_for_selected)
                        )
                        else 0.0
                    ),
                    "combat_quality_kaiser_pressure_selected": (
                        1.0
                        if (
                            selected_is_kaiser_encounter
                            and
                            isinstance(legal_actions, list)
                            and 0 <= int(action_idx) < len(legal_actions)
                            and quality_stats.get("combat_quality_kaiser_back_attack_risk", 0.0) > 0.05
                            and self._is_kaiser_pressure_action(legal_actions[int(action_idx)])
                        )
                        else 0.0
                    ),
                    "combat_quality_kaiser_risky_end_turn_selected": (
                        1.0
                        if (
                            selected_is_kaiser_encounter
                            and
                            isinstance(legal_actions, list)
                            and 0 <= int(action_idx) < len(legal_actions)
                            and quality_stats.get("combat_quality_kaiser_back_attack_risk", 0.0) > 0.05
                            and self._semantic_family(legal_actions[int(action_idx)]) == "end_turn"
                        )
                        else 0.0
                    ),
                    "combat_quality_ceremonial_low_impact_selected": selected_ceremonial_low_impact,
                    "combat_quality_ceremonial_high_impact_selected": selected_ceremonial_high_impact,
                }
            else:
                # Run MCTS to get action and policy
                with torch.no_grad():
                    self.mcts.set_training_step(self.total_steps)
                    self.mcts.set_root_bias_enabled(True)
                    self.mcts.set_semantic_rollout_enabled(self.mcts._semantic_rollout_enabled)
                    phase3_bias_vec = self._compute_route_heuristic_bias_vector(
                        decision_domain=decision_domain,
                        legal_actions=legal_actions if isinstance(legal_actions, list) else None,
                    )
                    if hasattr(self.mcts, "set_route_heuristic_bias"):
                        self.mcts.set_route_heuristic_bias(phase3_bias_vec)
                    action_idx, search_policy = self.mcts.run(
                        self.network,
                        obs,
                        action_mask,
                        num_simulations=num_simulations,
                        temperature=temperature,
                        decision_domain=decision_domain,
                    )
                search_stats = getattr(self.mcts, "last_run_stats", {}) or {}
            pre_guard_action_idx = int(action_idx)
            action_idx = self._apply_post_search_action_hard_guards(
                decision_domain=decision_domain,
                action_idx=int(action_idx),
                legal_actions=legal_actions if isinstance(legal_actions, list) else None,
                action_mask=action_mask,
                obs=obs if isinstance(obs, dict) else None,
                info=prev_info if isinstance(prev_info, dict) else None,
                search_stats=search_stats if isinstance(search_stats, dict) else {},
            )
            if isinstance(search_stats, dict):
                # RC-4: by default (rewrite_target off) the guard stays a behavior/safety wrapper --
                # the guard action is still EXECUTED and stored as the replay 'action', but the
                # policy/value target keeps the model's OWN pre-guard soft search distribution so
                # credit assignment stays on-policy. --hard-guard-target-rewrite on restores the
                # legacy one-hot rewrite for A/B comparison.
                search_policy, policy_retargeted, guard_override = resolve_hard_guard_policy_target(
                    search_policy,
                    original_action_idx=pre_guard_action_idx,
                    final_action_idx=int(action_idx),
                    rewrite_target=getattr(self, "hard_guard_target_rewrite", False),
                    max_actions=MAX_ACTIONS,
                )
                search_stats["post_search_hard_guard_override_applied"] = 1.0 if guard_override else 0.0
                search_stats["post_search_hard_guard_policy_retargeted"] = 1.0 if policy_retargeted else 0.0
                search_stats["post_search_hard_guard_original_action_idx"] = float(pre_guard_action_idx)
                search_stats["post_search_hard_guard_final_action_idx"] = float(action_idx)
                if policy_retargeted and decision_domain == "combat":
                    search_stats["combat_quality_hard_guard_policy_target_rewrite"] = 1.0
            if decision_domain == "route":
                route_dry_run_error_count = self._record_route_heuristic_dry_run(
                    records=route_dry_run_records,
                    error_count=route_dry_run_error_count,
                    action_idx=int(action_idx),
                    legal_actions=legal_actions if isinstance(legal_actions, list) else None,
                )

            if search_stats:
                search_num_simulations.append(float(search_stats.get("num_simulations", num_simulations)))
                search_root_candidates.append(float(search_stats.get("root_candidates", 0.0)))
                search_root_selectable_children.append(float(search_stats.get("root_selectable_children_mean", 0.0)))
                search_expanded_children.append(float(search_stats.get("mean_expanded_children", 0.0)))
                search_predicted_legal.append(float(search_stats.get("mean_predicted_legal_count", 0.0)))
                search_surface_keep.append(float(search_stats.get("mean_surface_keep_count", 0.0)))
                search_max_depths.append(float(search_stats.get("max_search_depth", 0.0)))
                search_depths.append(float(search_stats.get("mean_leaf_depth", 0.0)))
                search_concrete_depths.append(float(search_stats.get("mean_concrete_leaf_depth", 0.0)))
                search_depth_ge_2.append(float(search_stats.get("depth_ge_2_rate", 0.0)))
                search_depth_ge_3.append(float(search_stats.get("depth_ge_3_rate", 0.0)))
                search_semantic_expansion.append(float(search_stats.get("semantic_expansion_rate", 0.0)))
                search_semantic_switch.append(float(search_stats.get("semantic_switch_rate", 0.0)))
                search_semantic_chain_steps.append(float(search_stats.get("semantic_chain_steps_mean", 0.0)))
                search_semantic_drill.append(float(search_stats.get("semantic_drill_rate", 0.0)))
                search_root_top1_visit_share.append(float(search_stats.get("root_top1_visit_share", 0.0)))
                search_root_visit_entropy.append(float(search_stats.get("root_visit_entropy", 0.0)))
                search_root_bias_scale.append(float(search_stats.get("root_bias_scale", 0.0)))
                search_root_bias_nonzero.append(float(search_stats.get("root_bias_nonzero", 0.0)))
                search_root_bias_abs_mean.append(float(search_stats.get("root_bias_abs_mean", 0.0)))
                search_root_bias_max_abs.append(float(search_stats.get("root_bias_max_abs", 0.0)))
                search_root_bias_changed_top1.append(float(search_stats.get("root_bias_changed_top1", 0.0)))
                search_root_bias_selected_delta.append(float(search_stats.get("root_bias_selected_action_delta", 0.0)))
                search_root_bias_suppressed_by_gate.append(float(search_stats.get("root_bias_suppressed_by_gate", 0.0)))
                search_root_bias_scale_effective.append(float(search_stats.get("root_bias_scale_effective", 0.0)))
                search_end_turn_guard_applied.append(float(search_stats.get("end_turn_guard_applied", 0.0)))
                search_end_turn_guard_forced_alt.append(float(search_stats.get("end_turn_guard_forced_alternative", 0.0)))
                for metric_key, metric_value in search_stats.items():
                    try:
                        domain_search_stats[decision_domain][metric_key].append(float(metric_value))
                    except (TypeError, ValueError):
                        continue

            chosen_action = legal_actions[action_idx] if action_idx < len(legal_actions) else None
            chosen_semantic = (
                semantic_actions[action_idx]
                if action_idx < len(semantic_actions) and isinstance(semantic_actions[action_idx], dict)
                else None
            )
            semantic_candidate_indices = [
                semantic_rollout_index(signature)
                for signature in semantic_actions[:MAX_ACTIONS]
            ]
            semantic_policy = aggregate_concrete_policy_to_semantic(
                search_policy,
                semantic_candidate_indices,
            )
            semantic_action = semantic_rollout_index(chosen_semantic or chosen_action)
            chosen_signature = compact_action_signature(chosen_action)
            action_family = ""
            semantic_domain = ""
            phase = str(info.get("phase") or "").strip().lower()
            surface = ""
            selection = ""
            wasteful_end_turn = False
            wasteful_proceed = False
            if chosen_signature:
                chosen_signature = {
                    **chosen_signature,
                    "selected_index": int(action_idx),
                    "phase": info.get("phase"),
                    "legal_action_count": int(info.get("legal_action_count", 0) or 0),
                    "selection": chosen_action.get("selection") if isinstance(chosen_action, dict) else None,
                    "fast_path_applied": bool(fast_path_reason),
                    "fast_path_reason": fast_path_reason or None,
                    "reward_type": (
                        (chosen_action.get("reward") or {}).get("type")
                        if isinstance(chosen_action, dict) and isinstance(chosen_action.get("reward"), dict)
                        else None
                    ),
                    "shop_action": (
                        chosen_action.get("shop_action")
                        if isinstance(chosen_action, dict)
                        else None
                    ),
                }
                surface = str(chosen_signature.get("surface") or "").strip().lower()
                selection = str(chosen_signature.get("selection") or "").strip().lower()
                semantic_compact = chosen_signature.get("semantic")
                if isinstance(semantic_compact, dict):
                    family = str(semantic_compact.get("family") or "").strip().lower()
                    action_family = family
                    semantic_domain = str(semantic_compact.get("domain") or "").strip().lower()
                    if family:
                        selected_family_counts[decision_domain][family] += 1
            if not action_family and isinstance(chosen_action, dict):
                action_family = self._semantic_family(chosen_action)
                if action_family:
                    selected_family_counts[decision_domain][action_family] += 1
            if action_family == "end_turn":
                end_turn_context: dict[str, Any] = {}
                try:
                    end_turn_context = self._raw_end_turn_context(
                        obs,
                        np.asarray(action_mask, dtype=np.float32).reshape(-1),
                        legal_actions,
                    )
                    wasteful_end_turn = bool(end_turn_context.get("wasteful"))
                except Exception:
                    end_turn_context = {}
                    wasteful_end_turn = False
                if decision_domain == "combat":
                    try:
                        raw_obs_for_dump = self._current_raw_combat_obs()
                        encounter_for_dump = ""
                        if isinstance(raw_obs_for_dump, dict):
                            try:
                                boss_ctx_dump = build_boss_mechanics_context(raw_obs_for_dump)
                                encounter_for_dump = str(boss_ctx_dump.get("encounter_key") or "").lower()
                            except Exception:
                                encounter_for_dump = ""
                        action_diag_pre = info.get("action_diagnostics") if isinstance(info, dict) else None
                        self._dump_selected_end_turn_context(
                            encoded_obs=obs,
                            raw_obs=raw_obs_for_dump,
                            action_mask=np.asarray(action_mask, dtype=np.float32),
                            legal_actions=legal_actions if isinstance(legal_actions, list) else [],
                            chosen_idx=int(action_idx),
                            context=end_turn_context,
                            search_policy=search_policy,
                            search_stats=search_stats if isinstance(search_stats, dict) else None,
                            action_diagnostics=action_diag_pre if isinstance(action_diag_pre, dict) else None,
                            encounter=encounter_for_dump,
                            tier=str((info.get("tier") if isinstance(info, dict) else "") or ""),
                            pre_step_info=info if isinstance(info, dict) else None,
                        )
                    except Exception:
                        pass
            wasteful_proceed = self._wasteful_proceed_flag(
                chosen_signature,
                decision_domain=decision_domain,
                phase=phase,
            )
            card_reward_tracker.update(
                decision_domain=decision_domain,
                phase=phase,
                legal_actions=legal_actions if isinstance(legal_actions, list) else [],
                chosen_action=chosen_action if isinstance(chosen_action, dict) else None,
                chosen_signature=chosen_signature if isinstance(chosen_signature, dict) else None,
                selected_family=action_family,
                selection=selection,
            )
            if isinstance(search_stats, dict):
                payload = search_stats.get("_card_reward_choice_diagnostic")
                if isinstance(payload, dict):
                    try:
                        annotate_card_reward_final_selection(
                            payload,
                            chosen_action=chosen_action,
                            final_action_idx=int(action_idx),
                            final_selected_family=action_family,
                            phase=phase,
                            decision_domain=decision_domain,
                            policy_retargeted=bool(
                                float(search_stats.get("post_search_hard_guard_policy_retargeted", 0.0) or 0.0)
                            ),
                        )
                        self._dump_card_reward_choice_diagnostic(payload)
                    except Exception:
                        pass

            pre_step_raw_combat_obs = None
            pre_step_encounter_off = ""
            pre_step_tier_off = str((info.get("tier") if isinstance(info, dict) else "") or "")
            pre_step_offender_types: set[str] = set()

            if decision_domain == "combat" and isinstance(search_stats, dict):
                search_stats.update(
                    self._selected_combat_quality_stats(
                        obs,
                        int(action_idx),
                        legal_actions if isinstance(legal_actions, list) else [],
                        search_stats,
                        action_mask=np.asarray(action_mask, dtype=np.float32),
                    )
                )
                try:
                    pre_step_raw_combat_obs = self._current_raw_combat_obs()
                    if isinstance(pre_step_raw_combat_obs, dict):
                        try:
                            boss_ctx_off = build_boss_mechanics_context(pre_step_raw_combat_obs)
                            pre_step_encounter_off = str(boss_ctx_off.get("encounter_key") or "").lower()
                        except Exception:
                            pre_step_encounter_off = ""
                    offender_types = self._classify_action_offenders(
                        search_stats=search_stats,
                        encounter=pre_step_encounter_off,
                        family=action_family,
                    )
                    pre_step_offender_types = set(offender_types)
                    for offender_type in offender_types:
                        global_key = f"boss_combat/{offender_type}_count"
                        search_stats[f"offender/{offender_type}"] = 1.0
                    if offender_types:
                        self._dump_action_offender(
                            encoded_obs=obs,
                            raw_obs=pre_step_raw_combat_obs,
                            action_mask=np.asarray(action_mask, dtype=np.float32),
                            legal_actions=legal_actions if isinstance(legal_actions, list) else [],
                            chosen_idx=int(action_idx),
                            chosen_action=chosen_action,
                            offender_types=offender_types,
                            search_stats=search_stats,
                            encounter=pre_step_encounter_off,
                            tier=pre_step_tier_off,
                        )
                except Exception:
                    pass

            # Store transition
            decision_progress = self._episode_progress_snapshot(prev_info)
            decision_room_type = str(decision_progress.get("room_type") or "")
            decision_encounter_id = str(decision_progress.get("encounter_id") or "")
            decision_encounter_tier = str(
                infer_encounter_tier(decision_encounter_id, room_type=decision_room_type) or ""
            ).strip().lower()
            decision_floor = decision_progress.get("floor", 0.0)
            decision_act_id = decision_progress.get("act_id", 0.0)
            decision_diagnostics: dict[str, Any] = {}
            shop_payload = None
            rest_site_payload = None
            deck_upgrade_payload = None
            rest_raw_context: Any = prev_info
            try:
                transition_state = prev_info.get("transition_state") if isinstance(prev_info, dict) else None
                env_unwrap_for_shop = getattr(self.env, "unwrapped", self.env)
                raw_obs_for_shop = getattr(env_unwrap_for_shop, "_last_obs_raw", None)
                player_state = transition_state.get("player") if isinstance(transition_state, dict) else None
                gold_before = player_state.get("gold") if isinstance(player_state, dict) else prev_info.get("gold")
                if gold_before is None and isinstance(raw_obs_for_shop, dict):
                    raw_shop_player = raw_obs_for_shop.get("player")
                    if isinstance(raw_shop_player, dict):
                        gold_before = raw_shop_player.get("gold")
                    if gold_before is None:
                        gold_before = raw_obs_for_shop.get("gold")
                shop_raw_context: Any
                if isinstance(raw_obs_for_shop, dict) and isinstance(transition_state, dict):
                    # ``transition_state`` is the compact info payload used for
                    # replay metadata; on shop surfaces it can omit
                    # ``player.deck_cards``.  The hard shop guard already uses
                    # the env's raw pre-step obs, so include the same source
                    # here as a fallback.  This keeps episode-level
                    # build/shop_deck_* telemetry aligned with the guard
                    # without changing replay schemas or model inputs.
                    shop_raw_context = {"transition_state": transition_state, "raw_obs": raw_obs_for_shop}
                elif isinstance(raw_obs_for_shop, dict):
                    shop_raw_context = raw_obs_for_shop
                elif isinstance(transition_state, dict):
                    shop_raw_context = transition_state
                else:
                    shop_raw_context = prev_info
                shop_payload = build_shop_choice_payload(
                    decision_domain=decision_domain,
                    phase=phase,
                    legal_actions=legal_actions if isinstance(legal_actions, list) else [],
                    chosen_action=chosen_action if isinstance(chosen_action, dict) else None,
                    chosen_signature=chosen_signature if isinstance(chosen_signature, dict) else None,
                    selected_index=int(action_idx),
                    progress=decision_progress,
                    gold=gold_before,
                    raw_obs=shop_raw_context,
                    search_policy=search_policy,
                    max_topk=8,
                )
                if isinstance(shop_payload, dict):
                    shop_tracker.update(shop_payload)
                    dump_shop_choice_diagnostic(self, shop_payload)
            except Exception:
                shop_payload = None
            try:
                transition_state = prev_info.get("transition_state") if isinstance(prev_info, dict) else None
                env_unwrap_for_rest = getattr(self.env, "unwrapped", self.env)
                raw_obs_for_rest = getattr(env_unwrap_for_rest, "_last_obs_raw", None)
                if isinstance(raw_obs_for_rest, dict) and isinstance(transition_state, dict):
                    rest_raw_context = {"transition_state": transition_state, "raw_obs": raw_obs_for_rest}
                elif isinstance(raw_obs_for_rest, dict):
                    rest_raw_context = raw_obs_for_rest
                elif isinstance(transition_state, dict):
                    rest_raw_context = transition_state
                else:
                    rest_raw_context = prev_info
                raw_legal_actions = (
                    prev_info.get("raw_legal_actions_compact")
                    if isinstance(prev_info.get("raw_legal_actions_compact"), list)
                    else None
                )
                rest_site_payload = build_rest_site_choice_payload(
                    decision_domain=decision_domain,
                    phase=phase,
                    legal_actions=legal_actions if isinstance(legal_actions, list) else [],
                    raw_legal_actions=raw_legal_actions,
                    chosen_action=chosen_action if isinstance(chosen_action, dict) else None,
                    chosen_signature=chosen_signature if isinstance(chosen_signature, dict) else None,
                    selected_index=int(action_idx),
                    progress=decision_progress,
                    raw_obs=rest_raw_context,
                    search_policy=search_policy,
                    max_topk=8,
                )
                if isinstance(rest_site_payload, dict):
                    rest_site_tracker.update(rest_site_payload)
                    dump_rest_site_choice_diagnostic(self, rest_site_payload)
            except Exception:
                rest_site_payload = None
            try:
                deck_upgrade_payload = build_deck_upgrade_choice_payload(
                    decision_domain=decision_domain,
                    phase=phase,
                    legal_actions=legal_actions if isinstance(legal_actions, list) else [],
                    chosen_action=chosen_action if isinstance(chosen_action, dict) else None,
                    chosen_signature=chosen_signature if isinstance(chosen_signature, dict) else None,
                    selected_index=int(action_idx),
                    progress=decision_progress,
                    raw_obs=rest_raw_context,
                    search_policy=search_policy,
                    max_topk=8,
                )
            except Exception:
                deck_upgrade_payload = None
            if decision_domain == "combat":
                try:
                    raw_obs_for_pre_step = pre_step_raw_combat_obs
                    if raw_obs_for_pre_step is None:
                        raw_obs_for_pre_step = self._current_raw_combat_obs()
                    decision_diagnostics = self._compact_pre_step_combat_diagnostics(
                        encoded_obs=obs,
                        raw_obs=raw_obs_for_pre_step if isinstance(raw_obs_for_pre_step, dict) else None,
                        legal_actions=legal_actions if isinstance(legal_actions, list) else [],
                        action_mask=np.asarray(action_mask, dtype=np.float32),
                        selected_idx=int(action_idx),
                        encounter_tier=decision_encounter_tier,
                        action_family=action_family,
                    )
                except Exception:
                    decision_diagnostics = {}
                try:
                    summoner_payload = build_summoner_targeting_payload(
                        raw_obs=raw_obs_for_pre_step if isinstance(raw_obs_for_pre_step, dict) else None,
                        legal_actions=legal_actions if isinstance(legal_actions, list) else [],
                        action_mask=np.asarray(action_mask, dtype=np.float32),
                        selected_idx=int(action_idx),
                        search_policy=search_policy,
                        progress=decision_progress,
                        encounter_id=decision_encounter_id,
                        encounter_tier=decision_encounter_tier,
                        max_candidates=16,
                    )
                    if isinstance(summoner_payload, dict):
                        decision_diagnostics.setdefault("summoner_targeting", summoner_payload)
                        summoner_targeting_tracker.update(summoner_payload)
                        dump_summoner_targeting_diagnostic(self, summoner_payload)
                except Exception:
                    pass
                try:
                    intent_combat_quality_payload = build_intent_combat_quality_payload(
                        raw_obs=raw_obs_for_pre_step if isinstance(raw_obs_for_pre_step, dict) else None,
                        legal_actions=legal_actions if isinstance(legal_actions, list) else [],
                        action_mask=np.asarray(action_mask, dtype=np.float32),
                        selected_idx=int(action_idx),
                        search_policy=search_policy,
                        progress=decision_progress,
                        encounter_id=decision_encounter_id,
                        encounter_tier=decision_encounter_tier,
                        max_candidates=12,
                    )
                    if isinstance(intent_combat_quality_payload, dict):
                        decision_diagnostics.setdefault("intent_combat_quality", intent_combat_quality_payload)
                        intent_combat_quality_tracker.update(intent_combat_quality_payload)
                        dump_intent_combat_quality_diagnostic(self, intent_combat_quality_payload)
                except Exception:
                    pass
                try:
                    target_priority_payload = build_target_priority_payload(
                        raw_obs=raw_obs_for_pre_step if isinstance(raw_obs_for_pre_step, dict) else None,
                        legal_actions=legal_actions if isinstance(legal_actions, list) else [],
                        action_mask=np.asarray(action_mask, dtype=np.float32),
                        selected_idx=int(action_idx),
                        search_policy=search_policy,
                        progress=decision_progress,
                        encounter_id=decision_encounter_id,
                        encounter_tier=decision_encounter_tier,
                        max_candidates=32,
                    )
                    if isinstance(target_priority_payload, dict):
                        decision_diagnostics.setdefault("target_priority", target_priority_payload)
                        target_priority_tracker.update(target_priority_payload)
                        dump_target_priority_diagnostic(self, target_priority_payload)
                except Exception:
                    pass
            if isinstance(shop_payload, dict):
                decision_diagnostics.setdefault("shop_choice", shop_payload)
            if isinstance(rest_site_payload, dict):
                decision_diagnostics.setdefault("rest_site_choice", rest_site_payload)
            if isinstance(deck_upgrade_payload, dict):
                decision_diagnostics.setdefault("deck_upgrade_choice", deck_upgrade_payload)
            if decision_domain == "combat" and action_family == "end_turn":
                try:
                    raw_obs_for_audit = pre_step_raw_combat_obs
                    if raw_obs_for_audit is None:
                        raw_obs_for_audit = self._current_raw_combat_obs()
                    if not isinstance(search_stats, dict):
                        search_stats = {}
                    audit_payload = dump_end_turn_pre_dispatch_audit(
                        self,
                        encoded_obs=obs if isinstance(obs, dict) else None,
                        raw_obs=raw_obs_for_audit if isinstance(raw_obs_for_audit, dict) else None,
                        action_mask=np.asarray(action_mask, dtype=np.float32),
                        legal_actions=legal_actions if isinstance(legal_actions, list) else [],
                        chosen_idx=int(action_idx),
                        search_policy=search_policy,
                        search_stats=search_stats,
                        pre_step_info=prev_info if isinstance(prev_info, dict) else None,
                        encounter=pre_step_encounter_off or decision_encounter_id,
                        tier=pre_step_tier_off or decision_encounter_tier,
                    )
                    if isinstance(audit_payload, dict):
                        decision_diagnostics.setdefault("end_turn_pre_dispatch", audit_payload)
                        flags = audit_payload.get("flags")
                        counts = audit_payload.get("counts")
                        player = audit_payload.get("player")
                        flags = flags if isinstance(flags, dict) else {}
                        counts = counts if isinstance(counts, dict) else {}
                        player = player if isinstance(player, dict) else {}

                        def _audit_float(src: dict[str, Any], key: str, default: float = 0.0) -> float:
                            try:
                                return float(src.get(key, default) or default)
                            except (TypeError, ValueError):
                                return float(default)

                        search_stats["combat_quality_end_turn_pre_dispatch_full_energy_like"] = (
                            1.0 if bool(flags.get("full_energy_like")) else 0.0
                        )
                        search_stats["combat_quality_end_turn_pre_dispatch_full_energy_skip_suspect"] = (
                            1.0 if bool(flags.get("full_energy_skip_suspect")) else 0.0
                        )
                        search_stats["combat_quality_end_turn_pre_dispatch_legal_generation_gap"] = (
                            1.0 if bool(flags.get("legal_generation_gap_suspect")) else 0.0
                        )
                        search_stats["combat_quality_end_turn_pre_dispatch_raw_hand_legal_surface_mismatch"] = (
                            1.0 if bool(flags.get("raw_hand_legal_surface_mismatch")) else 0.0
                        )
                        search_stats["combat_quality_end_turn_pre_dispatch_singleton_frontier_suspect"] = (
                            1.0 if bool(flags.get("singleton_frontier_suspect")) else 0.0
                        )
                        search_stats["combat_quality_end_turn_pre_dispatch_pressure_skip_suspect"] = (
                            1.0 if bool(flags.get("pressure_skip_suspect")) else 0.0
                        )
                        search_stats["combat_quality_end_turn_pre_dispatch_full_energy_skip_with_playable_hand"] = (
                            1.0 if bool(flags.get("full_energy_skip_with_playable_hand")) else 0.0
                        )
                        search_stats["combat_quality_end_turn_pre_dispatch_safe_progress_candidate_count"] = _audit_float(
                            counts, "safe_progress_candidate_count"
                        )
                        search_stats["combat_quality_end_turn_pre_dispatch_ui_affordable_hand_card_count"] = _audit_float(
                            counts, "ui_affordable_hand_card_count"
                        )
                        search_stats["combat_quality_end_turn_pre_dispatch_legal_play_card_action_count"] = _audit_float(
                            counts, "legal_play_card_action_count"
                        )
                        search_stats["combat_quality_end_turn_pre_dispatch_affordable_play_card_action_count"] = _audit_float(
                            counts, "affordable_play_card_action_count"
                        )
                        search_stats["combat_quality_end_turn_pre_dispatch_mask_legal_count"] = _audit_float(
                            counts, "mask_legal_count"
                        )
                        search_stats["combat_quality_end_turn_pre_dispatch_raw_hand_card_count"] = _audit_float(
                            counts, "raw_hand_card_count"
                        )
                        search_stats["combat_quality_end_turn_pre_dispatch_energy"] = _audit_float(player, "energy")
                        search_stats["combat_quality_end_turn_pre_dispatch_energy_ratio"] = _audit_float(
                            player, "energy_ratio"
                        )
                except Exception:
                    pass
            trajectory.add_step(
                obs=obs,
                action=action_idx,
                reward=0.0,  # Reward is assigned after step
                reward_components=np.zeros(NUM_OBJECTIVE_HEADS, dtype=np.float32),
                action_mask=action_mask,
                search_policy=search_policy,
                root_value=root_value,
                root_value_components=root_value_components,
                objective_context=obs.get("objective_context"),
                semantic_action=semantic_action,
                semantic_policy=semantic_policy,
                action_info=chosen_signature,
                search_stats={
                    **self._compact_search_stats(search_stats),
                    "combat_quality_wasteful_end_turn_selected": 1.0 if wasteful_end_turn else float((search_stats or {}).get("combat_quality_wasteful_end_turn_selected", 0.0) or 0.0),
                },
                decision_diagnostics=decision_diagnostics,
                decision_domain=decision_domain,
                phase=phase,
                action_family=action_family,
                semantic_domain=semantic_domain,
                surface=surface,
                selection=selection,
                wasteful_end_turn=wasteful_end_turn,
                wasteful_proceed=wasteful_proceed,
                room_type=decision_room_type,
                encounter_id=decision_encounter_id,
                encounter_tier=decision_encounter_tier,
                floor=decision_floor,
                act_id=decision_act_id,
            )

            # Take action in environment
            obs, reward, terminated, truncated, info = self.env.step(int(action_idx))
            try:
                env_unwrap_after = getattr(self.env, "unwrapped", self.env)
                post_raw_obs = getattr(env_unwrap_after, "_last_obs_raw", None)
                post_transition_state = (
                    info.get("transition_state")
                    if isinstance(info, dict) and isinstance(info.get("transition_state"), dict)
                    else None
                )
                post_obs_context: Any
                if isinstance(post_raw_obs, dict) and isinstance(post_transition_state, dict):
                    post_obs_context = {
                        "transition_state": post_transition_state,
                        "raw_obs": post_raw_obs,
                        "info": info if isinstance(info, dict) else {},
                    }
                elif isinstance(post_raw_obs, dict):
                    post_obs_context = {
                        "raw_obs": post_raw_obs,
                        "info": info if isinstance(info, dict) else {},
                    }
                elif isinstance(post_transition_state, dict):
                    post_obs_context = {
                        "transition_state": post_transition_state,
                        "info": info if isinstance(info, dict) else {},
                    }
                else:
                    post_obs_context = info

                post_legal_actions: Any = []
                if isinstance(info, dict):
                    post_legal_actions = (
                        info.get("legal_actions_compact")
                        if isinstance(info.get("legal_actions_compact"), list)
                        else info.get("legal_actions")
                    )
                if not isinstance(post_legal_actions, list):
                    compact_getter = getattr(env_unwrap_after, "get_compact_legal_actions", None)
                    post_legal_actions = compact_getter() if callable(compact_getter) else []
                if not isinstance(post_legal_actions, list):
                    post_legal_actions = []

                smith_transition_payload = build_smith_upgrade_transition_payload(
                    rest_site_payload=rest_site_payload,
                    post_legal_actions=post_legal_actions,
                    pre_obs=rest_raw_context,
                    post_obs=post_obs_context,
                    post_info=info if isinstance(info, dict) else None,
                )
                if isinstance(smith_transition_payload, dict):
                    deck_upgrade_tracker.update_smith_transition(smith_transition_payload)
                    dump_smith_upgrade_transition_diagnostic(self, smith_transition_payload)
                    if trajectory.steps:
                        trajectory.steps[-1].setdefault("decision_diagnostics", {}).setdefault(
                            "smith_upgrade_transition",
                            smith_transition_payload,
                        )

                completed_deck_upgrade_payload = complete_deck_upgrade_choice_payload(
                    deck_upgrade_payload,
                    post_obs=post_obs_context,
                    post_info=info if isinstance(info, dict) else None,
                )
                if isinstance(completed_deck_upgrade_payload, dict):
                    deck_upgrade_tracker.update_deck_upgrade_choice(completed_deck_upgrade_payload)
                    dump_deck_upgrade_choice_diagnostic(self, completed_deck_upgrade_payload)
                    if trajectory.steps:
                        trajectory.steps[-1].setdefault("decision_diagnostics", {})[
                            "deck_upgrade_choice"
                        ] = completed_deck_upgrade_payload
            except Exception:
                pass
            potion_transition_record = info.get("potion_transition") if isinstance(info, dict) else None
            if isinstance(potion_transition_record, dict):
                self._dump_potion_transition(potion_transition_record)
                compact_transition = {
                    "event": potion_transition_record.get("event"),
                    "action_id": potion_transition_record.get("action_id"),
                    "potion_id_before": potion_transition_record.get("potion_id_before"),
                    "potion_title_before": potion_transition_record.get("potion_title_before"),
                    "potion_count_before": potion_transition_record.get("potion_count_before"),
                    "potion_count_after": potion_transition_record.get("potion_count_after"),
                    "execute_ok": potion_transition_record.get("execute_ok"),
                    "state_version_before": potion_transition_record.get("state_version_before"),
                    "state_version_after": potion_transition_record.get("state_version_after"),
                    "floor": potion_transition_record.get("floor"),
                    "room_type": potion_transition_record.get("room_type"),
                    "room_model": potion_transition_record.get("room_model"),
                    "encounter_id": potion_transition_record.get("encounter_id"),
                }
                potion_transition_records_episode.append(
                    {k: v for k, v in compact_transition.items() if v not in (None, "")}
                )
                if bool(potion_transition_record.get("execute_ok", True)):
                    used_potion_count_episode += 1
                    encounter_key = str(
                        potion_transition_record.get("room_model")
                        or potion_transition_record.get("encounter_id")
                        or potion_transition_record.get("encounter")
                        or ""
                    ).strip()
                    room_type_key = str(potion_transition_record.get("room_type") or "").strip()
                    if encounter_key:
                        used_potion_count_by_encounter[encounter_key.upper()] += 1
                    if room_type_key:
                        used_potion_count_by_room[room_type_key.lower()] += 1
            frontier_trace_record = info.get("frontier_trace") if isinstance(info, dict) else None
            if isinstance(frontier_trace_record, dict):
                self._dump_frontier_trace(frontier_trace_record)
            action_diagnostics = info.get("action_diagnostics") if isinstance(info, dict) else None
            if isinstance(action_diagnostics, dict) and trajectory.steps:
                diag_stats = trajectory.steps[-1].setdefault("search_stats", {})
                merge_action_diagnostics_into_search_stats(diag_stats, action_diagnostics)
                # Bridge-only diagnostics (notably HP-cost runtime safety) are
                # merged after the env step.  The primary offender dump above
                # runs before ``env.step`` so it cannot see those selected-side
                # flags.  Emit only newly-visible offenders here; this keeps the
                # JSONL actionable without duplicating the richer pre-step
                # trainer/search offenders.
                if decision_domain == "combat":
                    try:
                        post_offender_types = self._classify_action_offenders(
                            search_stats=diag_stats,
                            encounter=pre_step_encounter_off,
                            family=action_family,
                        )
                        post_offender_types = [
                            offender_type
                            for offender_type in post_offender_types
                            if offender_type not in pre_step_offender_types
                        ]
                        for offender_type in post_offender_types:
                            diag_stats[f"offender/{offender_type}"] = 1.0
                        if post_offender_types:
                            step_record = trajectory.steps[-1]
                            self._dump_action_offender(
                                encoded_obs=step_record.get("obs") if isinstance(step_record, dict) else obs,
                                raw_obs=pre_step_raw_combat_obs,
                                action_mask=np.asarray(
                                    step_record.get("action_mask", action_mask) if isinstance(step_record, dict) else action_mask,
                                    dtype=np.float32,
                                ),
                                legal_actions=legal_actions if isinstance(legal_actions, list) else [],
                                chosen_idx=int(action_idx),
                                chosen_action=chosen_action,
                                offender_types=post_offender_types,
                                search_stats=diag_stats,
                                encounter=pre_step_encounter_off,
                                tier=pre_step_tier_off,
                            )
                    except Exception:
                        pass
            episode_reward += reward
            episode_length += 1
            progress_snapshot = self._episode_progress_snapshot(info)
            if progress_snapshot:
                progress_snapshots.append(progress_snapshot)
                floor_int = int(progress_snapshot["floor"])
                if floor_int > 0:
                    seen_floors.add(floor_int)
                room_type_lower = str(progress_snapshot.get("room_type_lower") or "")
                if "elite" in room_type_lower:
                    elite_floors.add(floor_int)
                if "boss" in room_type_lower:
                    boss_floors.add(floor_int)
                if progress_snapshot.get("hp", 0.0) <= 0.0:
                    death_floor = max(death_floor, float(progress_snapshot.get("floor", 0.0)))

            reward_components = compute_transition_objective_rewards(
                prev_info.get("transition_state") if isinstance(prev_info.get("transition_state"), dict) else None,
                chosen_semantic or chosen_action,
                info.get("transition_state") if isinstance(info.get("transition_state"), dict) else None,
                prev_planner_context=planner_context,
                next_planner_context=info.get("planner_context") if isinstance(info.get("planner_context"), dict) else None,
                terminated=terminated,
                truncated=truncated,
            )

            # Update last step's reward
            trajectory.steps[-1]["reward"] = reward
            trajectory.steps[-1]["reward_components"] = reward_components.astype(np.float32, copy=False)

            if terminated or truncated:
                break

        final_info = dict(info)
        episode_telemetry = final_info.get("episode_telemetry") if isinstance(final_info, dict) else None
        episode_telemetry = episode_telemetry if isinstance(episode_telemetry, dict) else {}
        max_floor = max((float(snapshot.get("floor", 0.0)) for snapshot in progress_snapshots), default=0.0)
        max_act_id = max((float(snapshot.get("act_id", 0.0)) for snapshot in progress_snapshots), default=0.0)
        rooms_seen = len(seen_floors)
        act1_boss_seen = any(
            float(snapshot.get("act_id", 0.0)) <= 1.0 and "boss" in str(snapshot.get("room_type_lower") or "")
            for snapshot in progress_snapshots
        )
        act1_clear = max_act_id >= 2.0
        total_decisions = max(sum(decision_counts.values()), 1)
        encounter_id = initial_info.get("encounter_id")
        encounter_tier = infer_encounter_tier(encounter_id)
        boss_entry_snapshot = self._boss_entry_snapshot_from_episode(
            initial_info,
            progress_snapshots,
            encounter_tier=encounter_tier,
        )
        final_potion_count = self._potion_count_from_info(final_info)
        final_potion_count_int = int(self._diag_float(final_potion_count, 0.0))
        final_transition_state = final_info.get("transition_state") if isinstance(final_info, dict) else None
        env_unwrapped = getattr(self.env, "unwrapped", self.env)
        final_raw_obs = getattr(env_unwrapped, "_last_obs_raw", None)
        final_deck_cards = (
            extract_deck_cards_from_obs_like(final_transition_state)
            or extract_deck_cards_from_obs_like(final_info)
            or extract_deck_cards_from_obs_like(final_raw_obs)
        )
        final_deck_quality = compute_deck_quality_summary(final_deck_cards)
        final_deck_compact = compact_deck_cards(final_deck_cards, limit=80)
        card_reward_meta = card_reward_tracker.as_metadata()
        shop_meta = shop_tracker.as_metadata()
        rest_site_meta = rest_site_tracker.as_metadata()
        deck_upgrade_meta = deck_upgrade_tracker.as_metadata()
        summoner_targeting_meta = summoner_targeting_tracker.as_metadata()
        target_priority_meta = target_priority_tracker.as_metadata()
        intent_combat_quality_meta = intent_combat_quality_tracker.as_metadata()
        final_potion_dump = (
            self._compact_raw_potion_inventory(
                final_transition_state if isinstance(final_transition_state, dict) else None,
                limit=5,
            )
            or self._compact_raw_potion_inventory(
                final_info if isinstance(final_info, dict) else None,
                limit=5,
            )
        )
        if not final_potion_dump and final_potion_count_int > 0:
            final_potion_dump = [
                {"slot": int(slot), "empty": False, "source": "count_only"}
                for slot in range(min(final_potion_count_int, 5))
            ]
        used_potion_count_final_info = int((final_info or {}).get("used_potion_count_this_combat", 0) or 0)
        final_progress = progress_snapshots[-1] if progress_snapshots else self._episode_progress_snapshot(final_info)
        final_encounter_upper = str(final_progress.get("encounter_id_upper") or "").strip().upper()
        final_room_type_lower = str(final_progress.get("room_type_lower") or "").strip().lower()
        used_potion_count_transition_current = (
            int(used_potion_count_by_encounter.get(final_encounter_upper, 0))
            if final_encounter_upper
            else 0
        )
        if used_potion_count_transition_current <= 0 and final_room_type_lower:
            used_potion_count_transition_current = int(used_potion_count_by_room.get(final_room_type_lower, 0))
        used_potion_count = max(used_potion_count_final_info, used_potion_count_transition_current)
        potion_history = self._episode_potion_history_from_steps(
            trajectory.steps,
            encounter_id=final_encounter_upper,
            tail_limit=8,
        )
        potion_transitions_this_combat: list[dict[str, Any]] = []
        for record in potion_transition_records_episode:
            rec_encounter = str(
                record.get("room_model") or record.get("encounter_id") or ""
            ).strip().upper()
            rec_room = str(record.get("room_type") or "").strip().lower()
            if (final_encounter_upper and rec_encounter == final_encounter_upper) or (
                final_room_type_lower and rec_room == final_room_type_lower
            ):
                potion_transitions_this_combat.append(record)

        def _transition_sync_suspect(record: dict[str, Any]) -> bool:
            if not bool(record.get("execute_ok", True)):
                return False
            before_count = record.get("potion_count_before")
            after_count = record.get("potion_count_after")
            before_version = record.get("state_version_before")
            after_version = record.get("state_version_after")
            return bool(
                (
                    before_count is not None
                    and after_count is not None
                    and before_count == after_count
                )
                or (
                    before_version is not None
                    and after_version is not None
                    and before_version == after_version
                )
            )

        potion_transition_sync_suspect = any(
            _transition_sync_suspect(record) for record in potion_transitions_this_combat
        )
        trajectory.metadata = {
            "episode_mode": initial_info.get("episode_mode"),
            "encounter_id": encounter_id,
            "encounter_tier": encounter_tier,
            "encounter_pool": list(initial_info.get("encounter_pool") or []),
            "snapshot_sample_id": initial_info.get("snapshot_sample_id"),
            "snapshot_run_id": initial_info.get("snapshot_run_id"),
            "snapshot_floor_number": initial_info.get("snapshot_floor_number"),
            "snapshot_build_id": initial_info.get("snapshot_build_id"),
            "temperature": float(temperature),
            "semantic_rollout_enabled": bool(self.mcts._semantic_rollout_enabled),
            "semantic_switch_depth": int(self.mcts.semantic_switch_depth),
            "episode_total_reward": float(episode_reward),
            "episode_length": int(episode_length),
            "terminated": bool(terminated),
            "truncated": bool(truncated),
            "initial_phase": initial_info.get("phase"),
            "final_phase": final_info.get("phase"),
            "negative_reward_episode": bool(episode_reward < 0.0),
            "final_deck_cards_compact": final_deck_compact,
            "final_deck_quality_v2": final_deck_quality,
            **card_reward_meta,
            **shop_meta,
            **rest_site_meta,
            **deck_upgrade_meta,
            **summoner_targeting_meta,
            **target_priority_meta,
            **intent_combat_quality_meta,
            "max_floor": float(max_floor),
            "max_act_id": float(max_act_id),
            "rooms_seen": int(rooms_seen),
            "death_floor": float(death_floor),
            "elite_rooms_seen": int(len(elite_floors)),
            "boss_rooms_seen": int(len(boss_floors)),
            "boss_entry_hp": float(boss_entry_snapshot.get("hp", 0.0)) if boss_entry_snapshot else 0.0,
            "boss_entry_max_hp": float(boss_entry_snapshot.get("max_hp", 0.0)) if boss_entry_snapshot else 0.0,
            "boss_entry_hp_ratio": float(boss_entry_snapshot.get("hp_ratio", 0.0)) if boss_entry_snapshot else 0.0,
            "final_potion_count": int(final_potion_count_int),
            "final_potion_dump": list(final_potion_dump),
            "used_potion_count_this_combat": int(used_potion_count),
            "used_potion_count_final_info": int(used_potion_count_final_info),
            "used_potion_count_transition_current": int(used_potion_count_transition_current),
            "used_potion_count_episode": int(used_potion_count_episode),
            **potion_history,
            "potion_use_transitions_this_combat": list(potion_transitions_this_combat[-8:]),
            "potion_transition_sync_suspect_this_combat": bool(potion_transition_sync_suspect),
            "act1_boss_seen": bool(act1_boss_seen),
            "act1_clear": bool(act1_clear),
            "decision_counts": {domain: int(count) for domain, count in decision_counts.items()},
            "decision_domain_overrides": {key: int(value) for key, value in domain_override_counts.items()},
            "combat_like_decision_count": int(combat_like_decision_count),
            "direct_policy_eligible_count": int(direct_policy_eligible_count),
            "direct_policy_used_count": int(direct_policy_used_count),
            "episode_telemetry": dict(episode_telemetry),
        }

        settlement_signal = self._episode_settlement_signal(
            max_floor=max_floor,
            death_floor=death_floor,
            elite_rooms_seen=len(elite_floors),
            act1_boss_seen=act1_boss_seen,
            act1_clear=act1_clear,
            terminated=terminated,
        )
        settlement_stats = trajectory.apply_light_episode_settlement(
            settlement_signal,
            settlement_weight=self.settlement_weight,
            decay=self.settlement_decay,
            max_steps=self.settlement_max_steps,
        )
        augmented_episode_reward = float(episode_reward + settlement_stats.get("total_bonus", 0.0))
        trajectory.metadata.update({
            "settlement_weight": float(self.settlement_weight),
            "settlement_decay": float(self.settlement_decay),
            "settlement_max_steps": int(self.settlement_max_steps),
            "episode_augmented_total_reward": augmented_episode_reward,
        })

        self.buffer.save_episode(
            trajectory,
            discount=self.discount,
            n_steps=self.n_step_return,
        )
        self.episode_count += 1
        log_environment_episode_telemetry(
            writer=self.writer,
            episode_index=self.episode_count,
            episode_telemetry=episode_telemetry,
        )
        recent_tail_snapshot = self.record_recent_combat_episode(trajectory.metadata)

        if search_root_candidates:
            self.writer.add_scalar(
                "search/max_depth_mean",
                float(np.mean(search_max_depths)) if search_max_depths else 0.0,
                self.episode_count,
            )
            self.writer.add_scalar(
                "search/num_simulations",
                float(np.mean(search_num_simulations)) if search_num_simulations else 0.0,
                self.episode_count,
            )
            self.writer.add_scalar(
                "search/root_candidates_mean",
                float(np.mean(search_root_candidates)),
                self.episode_count,
            )
            self.writer.add_scalar(
                "search/root_selectable_children_mean",
                float(np.mean(search_root_selectable_children)) if search_root_selectable_children else 0.0,
                self.episode_count,
            )
            self.writer.add_scalar(
                "search/expanded_children_mean",
                float(np.mean(search_expanded_children)),
                self.episode_count,
            )
            self.writer.add_scalar(
                "search/predicted_legal_mean",
                float(np.mean(search_predicted_legal)),
                self.episode_count,
            )
            self.writer.add_scalar(
                "search/surface_keep_mean",
                float(np.mean(search_surface_keep)) if search_surface_keep else 0.0,
                self.episode_count,
            )
            self.writer.add_scalar(
                "search/leaf_depth_mean",
                float(np.mean(search_depths)),
                self.episode_count,
            )
            self.writer.add_scalar(
                "search/concrete_leaf_depth_mean",
                float(np.mean(search_concrete_depths)) if search_concrete_depths else 0.0,
                self.episode_count,
            )
            self.writer.add_scalar(
                "search/depth_ge_2_rate",
                float(np.mean(search_depth_ge_2)) if search_depth_ge_2 else 0.0,
                self.episode_count,
            )
            self.writer.add_scalar(
                "search/depth_ge_3_rate",
                float(np.mean(search_depth_ge_3)) if search_depth_ge_3 else 0.0,
                self.episode_count,
            )
            self.writer.add_scalar(
                "search/semantic_expansion_rate",
                float(np.mean(search_semantic_expansion)) if search_semantic_expansion else 0.0,
                self.episode_count,
            )
            self.writer.add_scalar(
                "search/semantic_switch_rate",
                float(np.mean(search_semantic_switch)) if search_semantic_switch else 0.0,
                self.episode_count,
            )
            self.writer.add_scalar(
                "search/semantic_chain_steps_mean",
                float(np.mean(search_semantic_chain_steps)) if search_semantic_chain_steps else 0.0,
                self.episode_count,
            )
            self.writer.add_scalar(
                "search/semantic_drill_rate",
                float(np.mean(search_semantic_drill)) if search_semantic_drill else 0.0,
                self.episode_count,
            )
            self.writer.add_scalar(
                "search/root_top1_visit_share",
                float(np.mean(search_root_top1_visit_share)) if search_root_top1_visit_share else 0.0,
                self.episode_count,
            )
            self.writer.add_scalar(
                "search/root_visit_entropy",
                float(np.mean(search_root_visit_entropy)) if search_root_visit_entropy else 0.0,
                self.episode_count,
            )
            self.writer.add_scalar(
                "search/root_bias_scale",
                float(np.mean(search_root_bias_scale)) if search_root_bias_scale else 0.0,
                self.episode_count,
            )
            self.writer.add_scalar(
                "search/root_bias_nonzero_rate",
                float(np.mean(search_root_bias_nonzero)) if search_root_bias_nonzero else 0.0,
                self.episode_count,
            )
            self.writer.add_scalar(
                "search/root_bias_abs_mean",
                float(np.mean(search_root_bias_abs_mean)) if search_root_bias_abs_mean else 0.0,
                self.episode_count,
            )
            self.writer.add_scalar(
                "search/root_bias_max_abs",
                float(np.max(search_root_bias_max_abs)) if search_root_bias_max_abs else 0.0,
                self.episode_count,
            )
            self.writer.add_scalar(
                "search/root_bias_changed_top1_rate",
                float(np.mean(search_root_bias_changed_top1)) if search_root_bias_changed_top1 else 0.0,
                self.episode_count,
            )
            self.writer.add_scalar(
                "search/root_bias_selected_action_delta_mean",
                float(np.mean(search_root_bias_selected_delta)) if search_root_bias_selected_delta else 0.0,
                self.episode_count,
            )
            self.writer.add_scalar(
                "search/root_bias_suppressed_by_gate_rate",
                float(np.mean(search_root_bias_suppressed_by_gate)) if search_root_bias_suppressed_by_gate else 0.0,
                self.episode_count,
            )
            self.writer.add_scalar(
                "search/root_bias_scale_effective_mean",
                float(np.mean(search_root_bias_scale_effective)) if search_root_bias_scale_effective else 0.0,
                self.episode_count,
            )
            self.writer.add_scalar(
                "search/end_turn_guard_applied",
                float(np.mean(search_end_turn_guard_applied)) if search_end_turn_guard_applied else 0.0,
                self.episode_count,
            )
            self.writer.add_scalar(
                "search/end_turn_guard_forced_alternative",
                float(np.mean(search_end_turn_guard_forced_alt)) if search_end_turn_guard_forced_alt else 0.0,
                self.episode_count,
            )
            objective_weight_keys = (
                ("search/objective_weight_survival", "objective_weight_survival"),
                ("search/objective_weight_hp", "objective_weight_hp"),
                ("search/objective_weight_build", "objective_weight_build"),
                ("search/objective_weight_resource", "objective_weight_resource"),
                ("search/root_objective_value", "root_objective_value"),
                ("search/end_turn_bias_applied", "end_turn_bias_applied"),
                ("search/objective_prior_applied", "objective_prior_applied"),
            )
            for writer_key, stat_key in objective_weight_keys:
                values = [
                    float(step.get("search_stats", {}).get(stat_key, 0.0))
                    for step in trajectory.steps
                    if isinstance(step.get("search_stats"), dict)
                ]
                if values:
                    self.writer.add_scalar(writer_key, float(np.mean(values)), self.episode_count)

        domain_search_means: dict[str, dict[str, float]] = {}
        domain_family_rates: dict[str, dict[str, float]] = {}
        total_fast_paths = int(sum(fast_path_counts.values()))

        self.writer.add_scalar("decision/total_count", float(total_decisions), self.episode_count)
        self.writer.add_scalar(
            "decision/domain_override_count",
            float(sum(domain_override_counts.values())),
            self.episode_count,
        )
        self.writer.add_scalar("decision/combat_like_count", float(combat_like_decision_count), self.episode_count)
        self.writer.add_scalar(
            "decision/direct_policy_eligible_count",
            float(direct_policy_eligible_count),
            self.episode_count,
        )
        self.writer.add_scalar(
            "decision/direct_policy_used_count",
            float(direct_policy_used_count),
            self.episode_count,
        )
        self.writer.add_scalar(
            "decision/direct_policy_used_rate",
            float(direct_policy_used_count) / float(max(direct_policy_eligible_count, 1)),
            self.episode_count,
        )
        self.writer.add_scalar("decision/fast_path_count", float(total_fast_paths), self.episode_count)
        self.writer.add_scalar(
            "decision/fast_path_rate",
            float(total_fast_paths) / float(total_decisions),
            self.episode_count,
        )
        for domain in DECISION_DOMAINS:
            domain_count = decision_counts.get(domain, 0)
            self.writer.add_scalar(f"decision/{domain}_count", float(domain_count), self.episode_count)
            self.writer.add_scalar(f"decision/{domain}_share", float(domain_count) / float(total_decisions), self.episode_count)
            domain_fast_paths = fast_path_counts.get(domain, 0)
            if domain_count > 0:
                self.writer.add_scalar(
                    f"decision/{domain}/fast_path_rate",
                    float(domain_fast_paths) / float(domain_count),
                    self.episode_count,
                )
            metric_lists = domain_search_stats.get(domain) or {}
            if not metric_lists:
                continue
            metric_name_map: dict[str, str] = {}
            metric_name_map.update(CARD_REWARD_PICK_QUALITY_GUARD_SEARCH_SUFFIXES)
            metric_name_map = {
                **metric_name_map,
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
                # Phase 3 route-heuristic prior bias telemetry.  The bias is
                # default-off, but both sync and async writers must expose the
                # same keys when an experiment opts in.
                "route_heuristic_bias_applied": "route_heuristic_bias_applied_rate",
                "route_heuristic_bias_abs_mean": "route_heuristic_bias_abs_mean",
                "route_heuristic_bias_max_abs": "route_heuristic_bias_max_abs",
                # Route hard-guard telemetry.  Keep this synchronized with the
                # async search_suffix_map in train.py while the async writer is
                # still hosted there.
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
                # Build hard-guard policy/telemetry.  Emergency mode keeps only
                # low-HP campfire survival protection; full-only card/shop/rest
                # overrides should stay at zero under that mode.
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
                "objective_weight_survival": "objective_weight_survival",
                "objective_weight_hp": "objective_weight_hp",
                "objective_weight_build": "objective_weight_build",
                "objective_weight_resource": "objective_weight_resource",
                "root_objective_value": "root_objective_value",
                "end_turn_bias_applied": "end_turn_bias_applied",
                "end_turn_guard_applied": "end_turn_guard_applied",
                "end_turn_guard_forced_alternative": "end_turn_guard_forced_alternative",
                "objective_prior_applied": "objective_prior_applied",
                "combat_grounded_root_enabled": "combat_grounded_root_enabled",
                "q_value_ucb_enabled": "q_value_ucb_enabled",
                **CARD_REWARD_GUARD_SEARCH_SUFFIXES,
                **SHOP_ACTION_GUARD_SEARCH_SUFFIXES,
                **POST_SEARCH_HARD_GUARD_SEARCH_SUFFIXES,
                **REST_SITE_SMITH_GUARD_SEARCH_SUFFIXES,
                **COMBAT_QUALITY_GUARD_SEARCH_SUFFIXES,
                **SUMMONER_TARGET_GUARD_SEARCH_SUFFIXES,
                **TARGET_PRIORITY_GUARD_SEARCH_SUFFIXES,
            }
            for stat_key, writer_suffix in metric_name_map.items():
                values = metric_lists.get(stat_key)
                if values:
                    domain_search_means.setdefault(domain, {})[stat_key] = float(np.mean(values))
                    self.writer.add_scalar(
                        f"search/{domain}/{writer_suffix}",
                        float(np.mean(values)),
                        self.episode_count,
                    )

            family_counts = selected_family_counts.get(domain) or {}
            if family_counts and domain_count > 0:
                for family, count in sorted(family_counts.items()):
                    safe_family = family.replace("/", "_").replace(" ", "_")
                    domain_family_rates.setdefault(domain, {})[family] = float(count) / float(domain_count)
                    self.writer.add_scalar(
                        f"decision/{domain}/family_{safe_family}_rate",
                        float(count) / float(domain_count),
                        self.episode_count,
                    )
        if total_fast_paths > 0:
            for reason in TRIVIAL_BUILD_FAST_PATH_REASONS:
                safe_reason = reason.replace("/", "_").replace(" ", "_")
                count = int(fast_path_reason_counts.get(reason, 0))
                self.writer.add_scalar(
                    f"decision/fast_path/reason_{safe_reason}_rate",
                    float(count) / float(total_fast_paths),
                    self.episode_count,
                )

        self.writer.add_scalar("episode/max_floor", float(max_floor), self.episode_count)
        self.writer.add_scalar("episode/max_act_id", float(max_act_id), self.episode_count)
        self.writer.add_scalar("episode/rooms_seen", float(rooms_seen), self.episode_count)
        self.writer.add_scalar("episode/death_floor", float(death_floor), self.episode_count)
        self.writer.add_scalar("episode/elite_rooms_seen", float(len(elite_floors)), self.episode_count)
        self.writer.add_scalar("episode/act1_boss_seen", 1.0 if act1_boss_seen else 0.0, self.episode_count)
        self.writer.add_scalar("episode/act1_clear", 1.0 if act1_clear else 0.0, self.episode_count)
        self.writer.add_scalar("episode/settlement_signal", float(settlement_stats.get("signal", 0.0)), self.episode_count)
        self.writer.add_scalar("episode/settlement_applied_steps", float(settlement_stats.get("applied_steps", 0.0)), self.episode_count)
        self.writer.add_scalar("episode/settlement_total_bonus", float(settlement_stats.get("total_bonus", 0.0)), self.episode_count)
        self.writer.add_scalar("episode/reward_augmented", float(augmented_episode_reward), self.episode_count)
        for tag_suffix, quality_key in FINAL_DECK_QUALITY_TB_KEYS:
            self.writer.add_scalar(
                f"deck/final_{tag_suffix}",
                float(final_deck_quality.get(quality_key, 0.0)),
                self.episode_count,
            )
        for tag_suffix, meta_key in CARD_REWARD_TB_KEYS:
            self.writer.add_scalar(
                f"build/card_reward_{tag_suffix}",
                float(card_reward_meta.get(meta_key, 0.0)),
                self.episode_count,
            )
        for tag_suffix, meta_key in SHOP_TB_KEYS:
            self.writer.add_scalar(
                f"build/shop_{tag_suffix}",
                float(shop_meta.get(meta_key, 0.0)),
                self.episode_count,
            )
        for tag_suffix, meta_key in REST_SITE_TB_KEYS:
            self.writer.add_scalar(
                f"build/rest_site_{tag_suffix}",
                float(rest_site_meta.get(meta_key, 0.0)),
                self.episode_count,
            )
        for tag_suffix, meta_key in DECK_UPGRADE_TB_KEYS:
            self.writer.add_scalar(
                f"build/deck_upgrade_{tag_suffix}",
                float(deck_upgrade_meta.get(meta_key, 0.0)),
                self.episode_count,
            )
        for tag_suffix, meta_key in SUMMONER_TARGETING_TB_KEYS:
            self.writer.add_scalar(
                f"combat/summoner_targeting_{tag_suffix}",
                float(summoner_targeting_meta.get(meta_key, 0.0)),
                self.episode_count,
            )
        for tag_suffix, meta_key in TARGET_PRIORITY_TB_KEYS:
            self.writer.add_scalar(
                f"combat/target_priority_{tag_suffix}",
                float(target_priority_meta.get(meta_key, 0.0)),
                self.episode_count,
            )
        for tag_suffix, meta_key in INTENT_COMBAT_QUALITY_TB_KEYS:
            self.writer.add_scalar(
                f"combat/intent_quality_{tag_suffix}",
                float(intent_combat_quality_meta.get(meta_key, 0.0)),
                self.episode_count,
            )
        if float(death_floor) > 0.0:
            for tag_suffix, quality_key in DEATH_DECK_QUALITY_TB_KEYS:
                self.writer.add_scalar(
                    f"death_deck/{tag_suffix}",
                    float(final_deck_quality.get(quality_key, 0.0)),
                    self.episode_count,
                )
            self._dump_death_deck_summary(
                final_info=final_info,
                final_progress=final_progress,
                final_deck_cards=final_deck_cards,
                final_deck_quality=final_deck_quality,
                final_deck_compact=final_deck_compact,
                card_reward_meta=card_reward_meta,
                episode_reward=float(episode_reward),
                episode_length=int(episode_length),
                max_floor=float(max_floor),
                death_floor=float(death_floor),
                act1_boss_seen=bool(act1_boss_seen),
                act1_clear=bool(act1_clear),
                progress_snapshots=progress_snapshots,
            )
        self._emit_route_heuristic_dry_run_metrics(
            records=route_dry_run_records,
            error_count=route_dry_run_error_count,
        )
        combat_quality_diagnostics = self._emit_combat_quality_episode_diagnostics(trajectory)
        boss_diagnostics = self._emit_boss_episode_diagnostics(
            trajectory,
            boss_entry=boss_entry_snapshot,
            final_potion_count=final_potion_count,
        )

        self.last_episode_metrics = {
            "decision_counts": {domain: int(count) for domain, count in decision_counts.items()},
            "decision_total": int(total_decisions),
            "decision_domain_overrides": {key: int(value) for key, value in domain_override_counts.items()},
            "combat_like_decision_count": int(combat_like_decision_count),
            "direct_policy_eligible_count": int(direct_policy_eligible_count),
            "direct_policy_used_count": int(direct_policy_used_count),
            "fast_path_counts": {domain: int(count) for domain, count in fast_path_counts.items()},
            "fast_path_total": int(total_fast_paths),
            "fast_path_reason_counts": {
                reason: int(fast_path_reason_counts.get(reason, 0))
                for reason in TRIVIAL_BUILD_FAST_PATH_REASONS
                if int(fast_path_reason_counts.get(reason, 0)) > 0
            },
            "domain_search_means": domain_search_means,
            "domain_family_rates": domain_family_rates,
            "combat_quality_diagnostics": combat_quality_diagnostics,
            "deck_quality_v2": final_deck_quality,
            "final_deck_cards_compact": final_deck_compact,
            "card_reward_metrics": card_reward_meta,
            "shop_metrics": shop_meta,
            "rest_site_metrics": rest_site_meta,
            "deck_upgrade_metrics": deck_upgrade_meta,
            "summoner_targeting_metrics": summoner_targeting_meta,
            "target_priority_metrics": target_priority_meta,
            "intent_combat_quality_metrics": intent_combat_quality_meta,
            "episode_telemetry": dict(episode_telemetry),
            "episode_reward": float(episode_reward),
            "episode_length": int(episode_length),
            "max_floor": float(max_floor),
            "max_act_id": float(max_act_id),
            "rooms_seen": int(rooms_seen),
            "death_floor": float(death_floor),
            "elite_rooms_seen": int(len(elite_floors)),
            "act1_boss_seen": bool(act1_boss_seen),
            "act1_clear": bool(act1_clear),
            "settlement_signal": float(settlement_stats.get("signal", 0.0)),
            "settlement_applied_steps": int(settlement_stats.get("applied_steps", 0.0)),
            "settlement_total_bonus": float(settlement_stats.get("total_bonus", 0.0)),
            "episode_reward_augmented": float(augmented_episode_reward),
            "boss_diagnostics": boss_diagnostics,
        }
        if recent_tail_snapshot:
            self.last_episode_metrics["recent_tail"] = recent_tail_snapshot

        return episode_reward, episode_length
