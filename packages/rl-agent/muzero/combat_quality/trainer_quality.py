"""Trainer adapter for combat-action quality heuristics.

This module owns tactical combat quality classification and search-free prior
bias helpers that were historically embedded in ``muzero.train``.  Keep it
adapter-light: methods may call back into ``MuZeroTrainer`` for already-defined
action feature helpers, but new tactical heuristics should live in
``muzero.combat_quality`` modules rather than growing the trainer monolith.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import torch

from muzero.combat_quality.action_bias import apply_card_block_waste_bias
from muzero.combat_quality.progress_candidates import collect_safe_progress_candidates
from muzero.combat_quality.survival_math import protection_outcome
from sts2_env.boss_mechanics import build_boss_mechanics_context
from sts2_env.observation_v2 import MAX_ACTIONS
from sts2_env.semantic_action import SEMANTIC_ACTION_FAMILIES, SEMANTIC_ROLE_NAMES


class CombatActionQualityMixin:
    """Combat action-quality diagnostics and bias helpers for MuZeroTrainer."""

    @staticmethod
    def _classify_end_turn_action(
        context: dict[str, Any],
        action_diagnostics: dict[str, Any] | None = None,
        boss_signals: dict[str, Any] | None = None,
    ) -> tuple[str, dict[str, bool]]:
        """Single classifier for selected end_turn actions.

        Used by both the bias-side stats and the selected-action JSONL dumper so
        ``bad_end_turn``/``forced_end_turn``/``strategic_defer_end_turn`` rates
        cannot drift between detector and tracker.  Priority is strict:
        ``bad > transient(forced) > forced > strategic_defer > unknown``.

        ``boss_signals`` carries already-computed mechanism flags (Kaiser back
        attack + facing candidate, Ceremonial stun window, etc.) so end_turn
        chosen *under* those windows is upgraded to ``bad_end_turn`` even when
        the static positive/urgent counters say nothing was urgently playable.
        """

        end_turn_indices = list(context.get("end_turn_indices") or [])
        wasteful = bool(context.get("wasteful", False))
        strategic_defer_available = bool(context.get("strategic_defer_available", False))
        positive_count = int(context.get("positive_progress_count", 0) or 0)
        urgent_count = int(context.get("urgent_positive_count", 0) or 0)
        deferable_count = int(context.get("deferable_positive_count", 0) or 0)
        energy = float(context.get("energy", 0.0) or 0.0)
        diag = action_diagnostics if isinstance(action_diagnostics, dict) else {}
        transient = bool(diag.get("transient_only_end_turn", False))
        boss = boss_signals if isinstance(boss_signals, dict) else {}
        kaiser_risk = float(boss.get("kaiser_back_attack_risk", 0.0) or 0.0)
        kaiser_facing_cands = float(boss.get("kaiser_facing_change_candidate_count", 0.0) or 0.0)
        kaiser_defense_cands = float(boss.get("kaiser_defense_candidate_count", 0.0) or 0.0)
        ceremonial_stun = float(boss.get("ceremonial_stun_window", 0.0) or 0.0)
        ceremonial_high_impact = float(boss.get("ceremonial_high_impact_count", 0.0) or 0.0)
        kaiser_pressure_window = (
            kaiser_risk > 0.05 and (kaiser_facing_cands >= 1.0 or kaiser_defense_cands >= 1.0)
        )
        ceremonial_open_window = ceremonial_stun > 0.05 and ceremonial_high_impact >= 1.0
        boss_window_open = bool(kaiser_pressure_window or ceremonial_open_window)
        flags = {
            "no_legal_positive_action": positive_count == 0,
            "transient_only_end_turn": transient,
            "has_energy_and_positive_action": energy > 0.05 and positive_count > 0,
            "has_urgent_or_mandatory_action": urgent_count > 0,
            "has_strategic_defer_reason": strategic_defer_available,
            "has_deferable_action": deferable_count > 0,
            "kaiser_pressure_window_open": kaiser_pressure_window,
            "ceremonial_open_window": ceremonial_open_window,
            "boss_window_open": boss_window_open,
        }
        if not end_turn_indices:
            return "unknown", flags
        # Transient (bridge handed us only end_turn this frame) is treated as a
        # forced selection, not bad — boss-window override does not apply.
        if transient:
            return "forced_end_turn", flags
        if wasteful or boss_window_open:
            return "bad_end_turn", flags
        if positive_count == 0:
            return "forced_end_turn", flags
        if strategic_defer_available:
            return "strategic_defer_end_turn", flags
        return "unknown", flags

    def _raw_end_turn_context(
        self,
        encoded_obs: dict[str, Any] | None,
        action_mask: np.ndarray,
        legal_actions: list[Any] | None,
    ) -> dict[str, Any]:
        mask_np = np.asarray(action_mask, dtype=np.float32).reshape(-1)
        raw_obs = self._current_raw_combat_obs()
        energy = self._combat_energy(encoded_obs, raw_obs)
        end_turn_indices: list[int] = []
        positive_indices: list[int] = []
        urgent_positive_indices: list[int] = []
        deferable_positive_indices: list[int] = []
        deferable_exhaust_indices: list[int] = []
        ethereal_urgent_indices: list[int] = []
        energy_gain_without_followup_indices: list[int] = []
        typed_followup_missing_indices: list[int] = []
        typed_future_penalty_indices: list[int] = []
        typed_no_draw_indices: list[int] = []
        typed_card_state_mutation_indices: list[int] = []
        setup_followup_dependent_indices: list[int] = []
        setup_followup_available_indices: list[int] = []
        potion_available_indices: list[int] = []
        potion_urgent_indices: list[int] = []
        potion_low_urgency_indices: list[int] = []
        potion_save_recommended_indices: list[int] = []
        potion_no_followup_indices: list[int] = []
        potion_lethal_indices: list[int] = []
        potion_prevent_lethal_indices: list[int] = []
        potion_mechanism_indices: list[int] = []
        potion_overkill_indices: list[int] = []
        potion_block_waste_indices: list[int] = []
        card_block_waste_indices: list[int] = []
        card_pure_block_indices: list[int] = []
        card_no_damage_pressure_indices: list[int] = []
        potion_use_quality_values: list[float] = []
        potion_waste_risk_values: list[float] = []
        potion_save_value_values: list[float] = []
        potion_hand_context_good_indices: list[int] = []
        potion_hand_context_bad_indices: list[int] = []
        potion_long_term_indices: list[int] = []
        potion_requires_followup_indices: list[int] = []
        potion_family_counts: dict[str, int] = {}
        potion_id_present: dict[str, int] = {}
        setup_scaling_indices: list[int] = []
        zero_cost_positive = False
        zero_cost_urgent = False
        legal_count = min(len(legal_actions or []), MAX_ACTIONS, mask_np.shape[0])
        for idx in range(legal_count):
            if mask_np[idx] <= 0:
                continue
            action = (legal_actions or [])[idx]
            family = self._semantic_family(action)
            if family == "end_turn":
                end_turn_indices.append(int(idx))
                continue
            classification = self._classify_positive_combat_action(
                action,
                idx,
                encoded_obs,
                raw_obs,
                legal_actions,
                mask_np,
                energy,
            )
            if bool(classification.get("card_block_waste", False)):
                card_block_waste_indices.append(int(idx))
            if bool(classification.get("card_pure_block", False)):
                card_pure_block_indices.append(int(idx))
            if bool(classification.get("card_no_damage_pressure", False)):
                card_no_damage_pressure_indices.append(int(idx))
            if bool(classification.get("potion_available", False)):
                potion_available_indices.append(int(idx))
                potion_use_quality_values.append(float(classification.get("potion_use_quality", 0.0) or 0.0))
                potion_waste_risk_values.append(float(classification.get("potion_waste_risk", 0.0) or 0.0))
                potion_save_value_values.append(float(classification.get("potion_save_value", 0.0) or 0.0))
                if bool(classification.get("potion_hand_context_good", False)):
                    potion_hand_context_good_indices.append(int(idx))
                if bool(classification.get("potion_hand_context_bad", False)):
                    potion_hand_context_bad_indices.append(int(idx))
                if bool(classification.get("potion_long_term_value", False)):
                    potion_long_term_indices.append(int(idx))
                if bool(classification.get("potion_requires_followup", False)):
                    potion_requires_followup_indices.append(int(idx))
                fam_list = classification.get("potion_effect_family") or []
                for fam in fam_list:
                    fam_key = str(fam).lower()
                    if fam_key:
                        potion_family_counts[fam_key] = potion_family_counts.get(fam_key, 0) + 1
                pid_str = str(classification.get("potion_id") or "")
                if pid_str:
                    potion_id_present[pid_str] = potion_id_present.get(pid_str, 0) + 1
            if bool(classification.get("potion_urgent", False)):
                potion_urgent_indices.append(int(idx))
            if bool(classification.get("potion_low_urgency", False)):
                potion_low_urgency_indices.append(int(idx))
            if bool(classification.get("potion_save_recommended", False)):
                potion_save_recommended_indices.append(int(idx))
            if bool(classification.get("potion_no_followup", False)):
                potion_no_followup_indices.append(int(idx))
            if bool(classification.get("potion_lethal", False)):
                potion_lethal_indices.append(int(idx))
            if bool(classification.get("potion_prevent_lethal", False)):
                potion_prevent_lethal_indices.append(int(idx))
            if bool(classification.get("potion_mechanism_answer", False)):
                potion_mechanism_indices.append(int(idx))
            if bool(classification.get("potion_overkill", False)):
                potion_overkill_indices.append(int(idx))
            if bool(classification.get("potion_block_waste", False)):
                potion_block_waste_indices.append(int(idx))
            if classification["positive"]:
                positive_indices.append(int(idx))
                roles = self._action_roles(action)
                if roles.intersection({"setup", "scaling", "power"}):
                    setup_scaling_indices.append(int(idx))
                if family == "play_card" and self._is_zero_cost_action(action):
                    zero_cost_positive = True
                if classification["urgent"]:
                    urgent_positive_indices.append(int(idx))
                    if family == "play_card" and self._is_zero_cost_action(action):
                        zero_cost_urgent = True
                if classification["deferable"]:
                    deferable_positive_indices.append(int(idx))
                if classification["deferable_exhaust"]:
                    deferable_exhaust_indices.append(int(idx))
                if classification["ethereal_urgent"]:
                    ethereal_urgent_indices.append(int(idx))
                if classification["energy_without_followup"]:
                    energy_gain_without_followup_indices.append(int(idx))
                if bool(classification.get("followup_missing", False)):
                    typed_followup_missing_indices.append(int(idx))
                if bool(classification.get("typed_future_penalty", False)):
                    typed_future_penalty_indices.append(int(idx))
                if bool(classification.get("typed_no_draw", False)):
                    typed_no_draw_indices.append(int(idx))
                if bool(classification.get("typed_card_state_mutation", False)):
                    typed_card_state_mutation_indices.append(int(idx))
                if bool(classification.get("setup_followup_dependent", False)):
                    setup_followup_dependent_indices.append(int(idx))
                if bool(classification.get("setup_followup_available", False)):
                    setup_followup_available_indices.append(int(idx))
        positive_progress_count = len(positive_indices)
        urgent_positive_count = len(urgent_positive_indices)
        deferable_positive_count = len(deferable_positive_indices)
        safe_progress_candidate_count = 0
        if isinstance(raw_obs, dict) and isinstance(legal_actions, list) and end_turn_indices:
            try:
                safe_candidates, _ = collect_safe_progress_candidates(
                    self,
                    selected_idx=int(end_turn_indices[0]),
                    legal_count=int(legal_count),
                    legal_actions=legal_actions,
                    mask_np=mask_np,
                    raw_obs=raw_obs,
                    current_energy=float(energy),
                    use_mask=True,
                    include_debug=False,
                )
                safe_progress_candidate_count = len(safe_candidates)
            except Exception:
                safe_progress_candidate_count = 0
        # True waste is now gated on urgent progress.  Exhaust/retain/HP-cost
        # resource cards can be legal and positive but strategically correct to
        # let flow to discard/retain instead of consuming the combat loop.
        wasteful = bool(end_turn_indices) and urgent_positive_count > 0 and (energy > 0.05 or zero_cost_urgent)
        strategic_defer_available = (
            bool(end_turn_indices)
            and safe_progress_candidate_count > 0
            and urgent_positive_count == 0
            and deferable_positive_count > 0
        )
        severity = 0.0
        if wasteful:
            severity = (
                1.0
                + 0.45 * min(max(float(energy), 0.0), 3.0)
                + 0.35 * min(float(urgent_positive_count), 4.0)
                + (0.75 if zero_cost_urgent else 0.0)
                + (0.35 if setup_scaling_indices else 0.0)
                + (0.25 if ethereal_urgent_indices else 0.0)
            )
        return {
            "wasteful": wasteful,
            "true_wasteful": wasteful,
            "strategic_defer_available": strategic_defer_available,
            "energy": float(energy),
            "end_turn_indices": end_turn_indices,
            "positive_indices": positive_indices,
            "urgent_positive_indices": urgent_positive_indices,
            "deferable_positive_indices": deferable_positive_indices,
            "deferable_exhaust_indices": deferable_exhaust_indices,
            "ethereal_urgent_indices": ethereal_urgent_indices,
            "energy_gain_without_followup_indices": energy_gain_without_followup_indices,
            "typed_followup_missing_indices": typed_followup_missing_indices,
            "typed_future_penalty_indices": typed_future_penalty_indices,
            "typed_no_draw_indices": typed_no_draw_indices,
            "typed_card_state_mutation_indices": typed_card_state_mutation_indices,
            "setup_followup_dependent_indices": setup_followup_dependent_indices,
            "setup_followup_available_indices": setup_followup_available_indices,
            "potion_available_indices": potion_available_indices,
            "potion_urgent_indices": potion_urgent_indices,
            "potion_low_urgency_indices": potion_low_urgency_indices,
            "potion_save_recommended_indices": potion_save_recommended_indices,
            "potion_no_followup_indices": potion_no_followup_indices,
            "potion_lethal_indices": potion_lethal_indices,
            "potion_prevent_lethal_indices": potion_prevent_lethal_indices,
            "potion_mechanism_indices": potion_mechanism_indices,
            "potion_overkill_indices": potion_overkill_indices,
            "potion_block_waste_indices": potion_block_waste_indices,
            "card_block_waste_indices": card_block_waste_indices,
            "card_pure_block_indices": card_pure_block_indices,
            "card_no_damage_pressure_indices": card_no_damage_pressure_indices,
            "setup_scaling_indices": setup_scaling_indices,
            "positive_progress_count": positive_progress_count,
            "safe_progress_candidate_count": safe_progress_candidate_count,
            "urgent_positive_count": urgent_positive_count,
            "deferable_positive_count": deferable_positive_count,
            "deferable_exhaust_count": len(deferable_exhaust_indices),
            "ethereal_urgent_count": len(ethereal_urgent_indices),
            "energy_gain_without_followup_count": len(energy_gain_without_followup_indices),
            "typed_followup_missing_count": len(typed_followup_missing_indices),
            "typed_future_penalty_count": len(typed_future_penalty_indices),
            "typed_no_draw_count": len(typed_no_draw_indices),
            "typed_card_state_mutation_count": len(typed_card_state_mutation_indices),
            "setup_followup_dependent_count": len(setup_followup_dependent_indices),
            "setup_followup_available_count": len(setup_followup_available_indices),
            "potion_available_count": len(potion_available_indices),
            "potion_urgent_count": len(potion_urgent_indices),
            "potion_low_urgency_count": len(potion_low_urgency_indices),
            "potion_save_recommended_count": len(potion_save_recommended_indices),
            "potion_no_followup_count": len(potion_no_followup_indices),
            "potion_lethal_count": len(potion_lethal_indices),
            "potion_prevent_lethal_count": len(potion_prevent_lethal_indices),
            "potion_mechanism_count": len(potion_mechanism_indices),
            "potion_overkill_count": len(potion_overkill_indices),
            "potion_block_waste_count": len(potion_block_waste_indices),
            "card_block_waste_count": len(card_block_waste_indices),
            "card_pure_block_count": len(card_pure_block_indices),
            "card_no_damage_pressure_count": len(card_no_damage_pressure_indices),
            "potion_use_quality_mean": float(np.mean(potion_use_quality_values)) if potion_use_quality_values else 0.0,
            "potion_waste_risk_mean": float(np.mean(potion_waste_risk_values)) if potion_waste_risk_values else 0.0,
            "potion_save_value_mean": float(np.mean(potion_save_value_values)) if potion_save_value_values else 0.0,
            "potion_hand_context_good_indices": potion_hand_context_good_indices,
            "potion_hand_context_bad_indices": potion_hand_context_bad_indices,
            "potion_long_term_indices": potion_long_term_indices,
            "potion_requires_followup_indices": potion_requires_followup_indices,
            "potion_hand_context_good_count": len(potion_hand_context_good_indices),
            "potion_hand_context_bad_count": len(potion_hand_context_bad_indices),
            "potion_long_term_count": len(potion_long_term_indices),
            "potion_requires_followup_count": len(potion_requires_followup_indices),
            "potion_family_counts": potion_family_counts,
            "potion_id_present": potion_id_present,
            "zero_cost_positive": zero_cost_positive,
            "zero_cost_urgent": zero_cost_urgent,
            "severity": float(severity),
        }

    def _obs_semantic_role_active(self, obs: dict[str, Any] | None, index: int, role: str) -> bool:
        if not isinstance(obs, dict) or index < 0:
            return False
        semantic_actions = obs.get("semantic_actions")
        if semantic_actions is None:
            return False
        try:
            semantic_np = semantic_actions.detach().cpu().numpy() if isinstance(semantic_actions, torch.Tensor) else np.asarray(semantic_actions)
            if semantic_np.ndim != 2 or index >= semantic_np.shape[0]:
                return False
            role_offset = len(SEMANTIC_ACTION_FAMILIES) + 0  # target scopes added below dynamically for import-stability
            # Avoid importing target-scope count into every caller; the vector layout is families + target_scopes + roles.
            from sts2_env.semantic_action import SEMANTIC_TARGET_SCOPES
            role_offset = len(SEMANTIC_ACTION_FAMILIES) + len(SEMANTIC_TARGET_SCOPES)
            role_index = SEMANTIC_ROLE_NAMES.index(role)
            vector_index = role_offset + role_index
            return bool(semantic_np[index].shape[0] > vector_index and semantic_np[index][vector_index] > 0.5)
        except Exception:
            return False

    def _is_x_cost_action(self, obs: dict[str, Any] | None, index: int, action: Any) -> bool:
        if self._obs_semantic_role_active(obs, index, "x_cost"):
            return True
        if not isinstance(action, dict):
            return False
        semantic = action.get("semantic") if isinstance(action.get("semantic"), dict) else {}
        if "x_cost" in self._action_roles(action):
            return True
        try:
            if float(semantic.get("x_cost_value") or 0.0) > 0.0:
                return True
        except (TypeError, ValueError):
            pass
        card = action.get("card") if isinstance(action.get("card"), dict) else {}
        if bool(card.get("x_cost") or card.get("costs_x") or action.get("x_cost") or action.get("costs_x")):
            return True
        cost = action.get("card_cost", card.get("cost"))
        return str(cost).strip().upper() == "X"

    @classmethod
    def _x_cost_has_non_energy_effect(cls, action: Any) -> bool:
        """Whether an X-cost play produces value independent of current energy.

        StS2 X-cost cards usually scale with the energy spent, so a 0-energy X
        play is dominated.  But X-cost pile-manipulation/exhaust/transform/
        retain/keyword cards still mutate state at 0 energy — those plays must
        not be flagged as ``zero_energy_x_cost_selected`` offenders.
        """

        if not isinstance(action, dict):
            return False
        semantic = action.get("semantic") if isinstance(action.get("semantic"), dict) else {}
        if bool(semantic.get("x_cost_has_non_energy_effect", False)):
            return True
        for key in (
            "typed_modifies_hand",
            "typed_upgrade_hand",
            "typed_exhaust_cards",
            "typed_discard_cards",
            "typed_transform_cards",
            "typed_copy_cards",
            "typed_add_modifier",
            "typed_add_keyword",
            "typed_set_replay",
            "typed_retain_cards",
            "typed_card_state_mutation",
        ):
            if bool(semantic.get(key, False)):
                return True
        if cls._action_roles(action).intersection(
            {"facing_change", "stun", "artifact_strip", "lock", "mechanism"}
        ):
            return True
        return False

    @classmethod
    def _x_cost_diagnostic(cls, action: Any, current_energy: float) -> dict[str, float]:
        """Static-state X-cost view for diagnostics: effective energy + non-energy effect."""

        is_x = False
        if isinstance(action, dict):
            semantic = action.get("semantic") if isinstance(action.get("semantic"), dict) else {}
            card = action.get("card") if isinstance(action.get("card"), dict) else {}
            try:
                if float(semantic.get("x_cost_value") or 0.0) > 0.0:
                    is_x = True
            except (TypeError, ValueError):
                pass
            if not is_x:
                if "x_cost" in cls._action_roles(action):
                    is_x = True
            if not is_x:
                if bool(semantic.get("is_x_cost") or card.get("x_cost") or card.get("costs_x")
                        or action.get("x_cost") or action.get("costs_x")):
                    is_x = True
            if not is_x:
                cost_text = str(action.get("card_cost") or card.get("cost") or "").strip().upper()
                if cost_text == "X":
                    is_x = True
        if not is_x:
            return {
                "is_x_cost": 0.0,
                "x_cost_effective_energy": 0.0,
                "x_cost_has_non_energy_effect": 0.0,
                "x_cost_bad": 0.0,
            }
        effective_energy = max(float(current_energy), 0.0)
        non_energy = cls._x_cost_has_non_energy_effect(action)
        bad = effective_energy <= 0.05 and not non_energy
        return {
            "is_x_cost": 1.0,
            "x_cost_effective_energy": effective_energy,
            "x_cost_has_non_energy_effect": 1.0 if non_energy else 0.0,
            "x_cost_bad": 1.0 if bad else 0.0,
        }

    def _combat_action_quality_bias(
        self,
        obs: dict[str, Any],
        action_mask: np.ndarray,
        legal_actions: list[Any],
    ) -> tuple[np.ndarray, dict[str, float], set[int]]:
        """Hard combat action-quality prior for search-free policy selection.

        This is not a learning target replacement.  It prevents the direct rollout
        planner from repeatedly sampling actions that are mechanically dominated:
        ending turn while playable progress remains, and spending an X-card at
        zero energy.  MCTS already had an end-turn guard; direct policy did not.
        """

        mask_np = np.asarray(action_mask, dtype=np.float32).reshape(-1)
        bias = np.zeros(MAX_ACTIONS, dtype=np.float32)
        raw_obs = self._current_raw_combat_obs()
        energy = self._combat_energy(obs, raw_obs)
        zero_energy_x_indices: set[int] = set()
        playable_indices: set[int] = set()
        end_turn_indices: set[int] = set()
        x_cost_indices: set[int] = set()
        x_cost_bad_indices: set[int] = set()
        x_cost_effective_energy_sum = 0.0
        card_selection_confirm_ready_available = False
        card_selection_confirm_ready_confirm_count = 0
        card_selection_confirm_ready_select_count = 0
        card_selection_deselect_candidate_count = 0
        card_selection_remaining_zero_select_count = 0
        card_selection_not_ready_select_count = 0

        def _action_bool(value: Any) -> bool:
            if isinstance(value, bool):
                return value
            if isinstance(value, (int, float)):
                return float(value) != 0.0
            if isinstance(value, str):
                return value.strip().lower() in {"1", "true", "yes", "y", "on"}
            return False

        legal_count = min(len(legal_actions or []), MAX_ACTIONS, mask_np.shape[0])
        for idx in range(legal_count):
            if mask_np[idx] <= 0:
                continue
            action = legal_actions[idx]
            family = self._semantic_family(action)
            if family == "end_turn":
                end_turn_indices.add(idx)
            elif family == "play_card":
                playable_indices.add(idx)
            if family == "play_card" and self._is_x_cost_action(obs, idx, action):
                x_cost_indices.add(idx)
                x_diag = self._x_cost_diagnostic(action, float(energy))
                x_cost_effective_energy_sum += float(x_diag.get("x_cost_effective_energy", 0.0))
                if energy <= 0.05:
                    zero_energy_x_indices.add(idx)
                if float(x_diag.get("x_cost_bad", 0.0)) > 0.5:
                    x_cost_bad_indices.add(idx)

        context: dict[str, Any]
        try:
            context = self._raw_end_turn_context(obs, mask_np, legal_actions)
        except Exception:
            context = {}
        wasteful = bool(context.get("wasteful", False))
        positive_indices = {int(i) for i in context.get("positive_indices", []) if 0 <= int(i) < MAX_ACTIONS}
        urgent_positive_indices = {int(i) for i in context.get("urgent_positive_indices", []) if 0 <= int(i) < MAX_ACTIONS}
        deferable_positive_indices = {int(i) for i in context.get("deferable_positive_indices", []) if 0 <= int(i) < MAX_ACTIONS}
        deferable_exhaust_indices = {int(i) for i in context.get("deferable_exhaust_indices", []) if 0 <= int(i) < MAX_ACTIONS}
        energy_gain_without_followup_indices = {int(i) for i in context.get("energy_gain_without_followup_indices", []) if 0 <= int(i) < MAX_ACTIONS}
        typed_followup_missing_indices = {int(i) for i in context.get("typed_followup_missing_indices", []) if 0 <= int(i) < MAX_ACTIONS}
        typed_future_penalty_indices = {int(i) for i in context.get("typed_future_penalty_indices", []) if 0 <= int(i) < MAX_ACTIONS}
        typed_no_draw_indices = {int(i) for i in context.get("typed_no_draw_indices", []) if 0 <= int(i) < MAX_ACTIONS}
        typed_card_state_mutation_indices = {int(i) for i in context.get("typed_card_state_mutation_indices", []) if 0 <= int(i) < MAX_ACTIONS}
        setup_followup_dependent_indices = {int(i) for i in context.get("setup_followup_dependent_indices", []) if 0 <= int(i) < MAX_ACTIONS}
        setup_followup_available_indices = {int(i) for i in context.get("setup_followup_available_indices", []) if 0 <= int(i) < MAX_ACTIONS}
        potion_available_indices = {int(i) for i in context.get("potion_available_indices", []) if 0 <= int(i) < MAX_ACTIONS}
        potion_urgent_indices = {int(i) for i in context.get("potion_urgent_indices", []) if 0 <= int(i) < MAX_ACTIONS}
        potion_low_urgency_indices = {int(i) for i in context.get("potion_low_urgency_indices", []) if 0 <= int(i) < MAX_ACTIONS}
        potion_save_recommended_indices = {int(i) for i in context.get("potion_save_recommended_indices", []) if 0 <= int(i) < MAX_ACTIONS}
        potion_no_followup_indices = {int(i) for i in context.get("potion_no_followup_indices", []) if 0 <= int(i) < MAX_ACTIONS}
        potion_lethal_indices = {int(i) for i in context.get("potion_lethal_indices", []) if 0 <= int(i) < MAX_ACTIONS}
        potion_prevent_lethal_indices = {int(i) for i in context.get("potion_prevent_lethal_indices", []) if 0 <= int(i) < MAX_ACTIONS}
        potion_mechanism_indices = {int(i) for i in context.get("potion_mechanism_indices", []) if 0 <= int(i) < MAX_ACTIONS}
        potion_overkill_indices = {int(i) for i in context.get("potion_overkill_indices", []) if 0 <= int(i) < MAX_ACTIONS}
        potion_block_waste_indices = {int(i) for i in context.get("potion_block_waste_indices", []) if 0 <= int(i) < MAX_ACTIONS}
        card_block_waste_indices = {int(i) for i in context.get("card_block_waste_indices", []) if 0 <= int(i) < MAX_ACTIONS}
        card_pure_block_indices = {int(i) for i in context.get("card_pure_block_indices", []) if 0 <= int(i) < MAX_ACTIONS}
        card_no_damage_pressure_indices = {int(i) for i in context.get("card_no_damage_pressure_indices", []) if 0 <= int(i) < MAX_ACTIONS}
        setup_indices = {int(i) for i in context.get("setup_scaling_indices", []) if 0 <= int(i) < MAX_ACTIONS}
        end_turn_indices.update(int(i) for i in context.get("end_turn_indices", []) if 0 <= int(i) < MAX_ACTIONS)
        if not positive_indices and not urgent_positive_indices:
            # Fallback is intentionally weaker than the old detector: do not
            # label every legal play_card as urgent, because exhaust/resource
            # cards may be strategically deferred into the discard loop.
            positive_indices = set(playable_indices)
            fallback_playable = set(playable_indices) - card_block_waste_indices
            urgent_positive_indices = {idx for idx in fallback_playable if not self._is_exhausting_action(legal_actions[idx])}
            wasteful = energy > 0.05 and bool(end_turn_indices) and bool(urgent_positive_indices)
        boost_indices = urgent_positive_indices if urgent_positive_indices else positive_indices
        severity = max(float(context.get("severity", 0.0) or 0.0), 1.0 if wasteful else 0.0)

        if wasteful and end_turn_indices:
            end_turn_penalty = min(5.5, 2.25 + 0.70 * severity + 0.25 * min(energy, 4.0))
            positive_bonus = min(1.00, 0.25 + 0.14 * severity)
            setup_bonus = 0.15 if setup_indices else 0.0
            for idx in end_turn_indices:
                if 0 <= idx < MAX_ACTIONS and idx < mask_np.shape[0] and mask_np[idx] > 0:
                    bias[idx] -= end_turn_penalty
            for idx in boost_indices:
                if 0 <= idx < MAX_ACTIONS and idx < mask_np.shape[0] and mask_np[idx] > 0:
                    bias[idx] += positive_bonus + (setup_bonus if idx in setup_indices else 0.0)
        else:
            end_turn_penalty = 0.0

        if zero_energy_x_indices:
            for idx in zero_energy_x_indices:
                if 0 <= idx < MAX_ACTIONS and idx < mask_np.shape[0] and mask_np[idx] > 0:
                    bias[idx] -= 4.5

        for idx in typed_followup_missing_indices:
            if 0 <= idx < MAX_ACTIONS and idx < mask_np.shape[0] and mask_np[idx] > 0:
                # Production/Bloodletting/Bullet-Time-like cards are legal but
                # dominated when the current hand cannot convert the generated
                # energy/cost rule/no-draw tradeoff this turn.  Keep this softer
                # than zero-energy X, because sometimes retaining/setting up a
                # card-state mutation is still a legitimate long-horizon choice.
                bias[idx] -= 0.65
        for idx in typed_future_penalty_indices | typed_no_draw_indices:
            if idx not in setup_followup_available_indices and 0 <= idx < MAX_ACTIONS and idx < mask_np.shape[0] and mask_np[idx] > 0:
                bias[idx] -= 0.25

        card_block_waste_bias_stats = apply_card_block_waste_bias(
            bias,
            action_mask=mask_np,
            card_block_waste_indices=card_block_waste_indices,
            positive_indices=positive_indices,
            urgent_positive_indices=urgent_positive_indices,
            end_turn_indices=end_turn_indices,
            max_actions=MAX_ACTIONS,
        )
        card_block_waste_bias_count = int(
            card_block_waste_bias_stats.get("card_block_waste_bias_count", 0.0)
        )
        card_block_waste_bias_min = float(
            card_block_waste_bias_stats.get("card_block_waste_bias_min", 0.0)
        )

        for idx in potion_urgent_indices:
            if 0 <= idx < MAX_ACTIONS and idx < mask_np.shape[0] and mask_np[idx] > 0:
                bias[idx] += 0.85
        for idx in potion_lethal_indices | potion_prevent_lethal_indices:
            if 0 <= idx < MAX_ACTIONS and idx < mask_np.shape[0] and mask_np[idx] > 0:
                bias[idx] += 1.15
        for idx in potion_mechanism_indices:
            if 0 <= idx < MAX_ACTIONS and idx < mask_np.shape[0] and mask_np[idx] > 0:
                bias[idx] += 0.90
        for idx in potion_low_urgency_indices:
            if 0 <= idx < MAX_ACTIONS and idx < mask_np.shape[0] and mask_np[idx] > 0:
                bias[idx] -= 0.55
        for idx in potion_save_recommended_indices:
            if 0 <= idx < MAX_ACTIONS and idx < mask_np.shape[0] and mask_np[idx] > 0:
                bias[idx] -= 0.65
        for idx in potion_no_followup_indices:
            if 0 <= idx < MAX_ACTIONS and idx < mask_np.shape[0] and mask_np[idx] > 0:
                bias[idx] -= 0.85
        for idx in potion_overkill_indices:
            if 0 <= idx < MAX_ACTIONS and idx < mask_np.shape[0] and mask_np[idx] > 0:
                bias[idx] -= 0.45
        for idx in potion_block_waste_indices:
            if 0 <= idx < MAX_ACTIONS and idx < mask_np.shape[0] and mask_np[idx] > 0:
                bias[idx] -= 0.45

        potion_protected_indices = (
            potion_urgent_indices
            | potion_lethal_indices
            | potion_prevent_lethal_indices
            | potion_mechanism_indices
        )
        potion_bad_indices = (
            potion_low_urgency_indices
            | potion_save_recommended_indices
            | potion_no_followup_indices
            | potion_overkill_indices
            | potion_block_waste_indices
        ) - potion_protected_indices
        bad_potion_bias_count = 0
        bad_potion_bias_min = 0.0
        for idx in potion_bad_indices:
            if 0 <= idx < MAX_ACTIONS and idx < mask_np.shape[0] and mask_np[idx] > 0:
                # The earlier per-symptom penalties are deliberately soft so
                # urgent / tactical potions can still win.  When several bad
                # timing symptoms combine (live offender pattern: 0-energy
                # Strength/Speed/Fortifier with only End Turn left), add a
                # stronger suppression unless the potion is protected by an
                # urgent/lethal/mechanism label.
                delta = -1.25
                if energy <= 0.05:
                    delta -= 0.75
                if idx in potion_no_followup_indices:
                    delta -= 0.50
                bias[idx] += delta
                bad_potion_bias_count += 1
                bad_potion_bias_min = min(bad_potion_bias_min, float(delta))

        for idx in range(legal_count):
            if mask_np[idx] <= 0:
                continue
            action = legal_actions[idx]
            if self._semantic_family(action) != "card_selection":
                continue
            selection = str(action.get("selection_action") or action.get("selection") or "").strip().lower()
            confirm_ready = _action_bool(action.get("confirm_ready"))
            is_selected = _action_bool(action.get("is_selected"))
            selection_ready = True if action.get("selection_ready") is None else _action_bool(action.get("selection_ready"))
            remaining_select: int | None = None
            try:
                if action.get("remaining_select") is not None:
                    remaining_select = int(action.get("remaining_select"))
            except (TypeError, ValueError):
                remaining_select = None
            selected_count: int | None = None
            try:
                if action.get("selected_count") is not None:
                    selected_count = int(action.get("selected_count"))
            except (TypeError, ValueError):
                selected_count = None
            should_close_selection = bool(confirm_ready) and (
                # New bridge contract: after at least one selected card, stop
                # toggling and prefer the terminal confirm.  When an optional
                # screen first opens with selected_count=0 and confirm already
                # enabled, do NOT force a zero-pick confirm; allow the policy to
                # select a useful card first.
                (selected_count is not None and selected_count > 0)
                or remaining_select == 0
                # Back-compat for old replay / old bridge payloads that lack
                # selected_count/remaining_select: keep the old confirm-ready
                # closeout behavior rather than leaving loops unguarded.
                or (selected_count is None and remaining_select is None)
            )

            if confirm_ready:
                card_selection_confirm_ready_available = True

            if selection in {"confirm", "confirm_selection"} and should_close_selection:
                # State-driven closeout for multi-pick card-selection screens:
                # once the UI says confirmation is legal *and* at least one
                # card has been selected (or max picks are exhausted), favor the
                # terminal action instead of continuing to toggle cards.
                bias[idx] += 2.25
                card_selection_confirm_ready_confirm_count += 1
            elif selection == "select":
                if should_close_selection:
                    bias[idx] -= 1.10
                    card_selection_confirm_ready_select_count += 1
                if not selection_ready:
                    # Diagnostic/light guard only.  The bridge fast-forwards
                    # the NChooseACard open guard before executing a select,
                    # so this should be rare; if it rises, we know the model is
                    # still seeing a transient selection surface.
                    bias[idx] -= 0.20
                    card_selection_not_ready_select_count += 1
                if is_selected:
                    # Clicking an already-selected card is a deselect toggle.
                    # This is exactly the Purity/净化 loop failure mode.
                    bias[idx] -= 2.75
                    card_selection_deselect_candidate_count += 1
                if remaining_select == 0:
                    bias[idx] -= 2.00
                    card_selection_remaining_zero_select_count += 1

        kaiser_back_attack_risk = 0.0
        ceremonial_one_card_lock = 0.0
        ceremonial_stun_window = 0.0
        ceremonial_low_impact_count = 0
        ceremonial_high_impact_count = 0
        kaiser_defense_candidate_count = 0
        kaiser_facing_change_candidate_count = 0
        kaiser_pressure_candidate_count = 0
        encounter = ""
        is_kaiser_encounter = False
        is_insatiable_encounter = False
        boss_ctx: dict[str, Any] = {}
        insatiable_sandpit_turns = 0.0
        insatiable_frantic_hand_count = 0.0
        insatiable_frantic_draw_count = 0.0
        insatiable_frantic_discard_count = 0.0
        insatiable_frantic_exhaust_count = 0.0
        insatiable_frantic_total_count = 0.0
        insatiable_escape_cycle_risk = 0.0
        insatiable_frantic_escape_candidate_count = 0
        insatiable_frantic_escape_bonus_applied = False
        insatiable_frantic_escape_urgency = 0.0
        insatiable_lethal_candidate_count = 0
        insatiable_non_escape_at1_penalty_count = 0
        if isinstance(raw_obs, dict):
            try:
                boss_ctx = build_boss_mechanics_context(raw_obs)
                encounter = str(boss_ctx.get("encounter_key") or "").lower()
                is_kaiser_encounter = self._is_kaiser_encounter_context(boss_ctx, raw_obs)
                is_insatiable_encounter = self._is_insatiable_encounter_context(boss_ctx, raw_obs)
                kaiser_back_attack_risk = self._kaiser_back_attack_risk_from_context(boss_ctx, raw_obs)
                ceremonial_one_card_lock = self._boss_context_max(boss_ctx, "one_card_lock")
                ceremonial_stun_window = self._boss_context_max(boss_ctx, "stun_window")
                if is_insatiable_encounter:
                    insatiable_sandpit_turns = self._insatiable_sandpit_turns_from_context(boss_ctx, raw_obs)
                    insatiable_frantic_hand_count = self._boss_context_max(boss_ctx, "frantic_escape_hand_count")
                    insatiable_frantic_draw_count = self._boss_context_max(boss_ctx, "frantic_escape_draw_count")
                    insatiable_frantic_discard_count = self._boss_context_max(boss_ctx, "frantic_escape_discard_count")
                    insatiable_frantic_exhaust_count = self._boss_context_max(boss_ctx, "frantic_escape_exhaust_count")
                    insatiable_frantic_total_count = self._boss_context_max(boss_ctx, "frantic_escape_total_count")
                    insatiable_escape_cycle_risk = self._insatiable_escape_cycle_risk(
                        insatiable_sandpit_turns,
                        insatiable_frantic_hand_count,
                        insatiable_frantic_draw_count,
                        insatiable_frantic_discard_count,
                        insatiable_frantic_total_count,
                    )
            except Exception:
                encounter = ""
                is_kaiser_encounter = False
                is_insatiable_encounter = False

        if is_insatiable_encounter and insatiable_sandpit_turns > 0.0:
            sandpit = float(insatiable_sandpit_turns)
            if sandpit <= 1.0:
                insatiable_frantic_escape_urgency = 1.0
            elif sandpit < 3.0:
                insatiable_frantic_escape_urgency = 0.85
            elif sandpit < 4.0:
                insatiable_frantic_escape_urgency = min(1.0, 0.35 + 0.25 * insatiable_escape_cycle_risk)
            else:
                insatiable_frantic_escape_urgency = 0.0

            lethal_candidate_indices: set[int] = set()
            frantic_indices: set[int] = set()
            for idx in range(legal_count):
                if mask_np[idx] <= 0:
                    continue
                action = legal_actions[idx]
                family = self._semantic_family(action)
                if family in {"play_card", "use_potion", "potion"} and self._is_action_confirmed_lethal(action, raw_obs):
                    lethal_candidate_indices.add(idx)
                if family == "play_card" and self._is_frantic_escape_action(action):
                    frantic_indices.add(idx)

            insatiable_lethal_candidate_count = len(lethal_candidate_indices)
            insatiable_frantic_escape_candidate_count = len(frantic_indices)

            for idx in frantic_indices:
                if 0 <= idx < MAX_ACTIONS and idx < mask_np.shape[0] and mask_np[idx] > 0:
                    if sandpit <= 1.0:
                        bias[idx] += 5.0
                    elif sandpit < 3.0:
                        bias[idx] += 3.25
                    elif insatiable_frantic_escape_urgency > 0.0:
                        bias[idx] += 0.75 * insatiable_frantic_escape_urgency
                    insatiable_frantic_escape_bonus_applied = True

            if sandpit <= 1.0 and frantic_indices:
                for idx in range(legal_count):
                    if mask_np[idx] <= 0 or idx in frantic_indices or idx in lethal_candidate_indices:
                        continue
                    # Sandpit 0 is terminal; at 1 the only non-lethal priority
                    # should be Frantic Escape when the action is legal.
                    bias[idx] -= 3.75
                    insatiable_non_escape_at1_penalty_count += 1
                for idx in end_turn_indices:
                    if idx not in frantic_indices and idx not in lethal_candidate_indices:
                        bias[idx] -= 5.0
            elif sandpit < 3.0 and frantic_indices:
                for idx in end_turn_indices:
                    if idx not in lethal_candidate_indices:
                        bias[idx] -= 1.10

        if is_kaiser_encounter and kaiser_back_attack_risk > 0.05:
            risk = min(1.0, float(kaiser_back_attack_risk))
            for idx in range(legal_count):
                if mask_np[idx] <= 0:
                    continue
                action = legal_actions[idx]
                family = self._semantic_family(action)
                roles = self._action_roles(action)
                block = self._action_metric(action, "block")
                if family == "end_turn":
                    bias[idx] -= 1.1 * risk
                elif self._is_kaiser_facing_change_action(action, raw_obs):
                    # Surrounded facing changes implicitly by resolving any targeted
                    # card/potion toward an enemy on the opposite side.  This is not a
                    # guessed "turn around" action-name field: it is derived from
                    # combat.facing + target_combat_id and the target enemy's
                    # BACK_ATTACK_LEFT/RIGHT_POWER marker.
                    bias[idx] += 1.10 * risk
                    kaiser_facing_change_candidate_count += 1
                    kaiser_defense_candidate_count += 1
                elif self._is_kaiser_risk_handling_action(action, raw_obs):
                    bias[idx] += 0.55 * risk
                    kaiser_defense_candidate_count += 1
                elif self._is_kaiser_pressure_action(action):
                    # Strike/Bash-only hands are not "defense" candidates, but they are
                    # useful to diagnose snapshot composition and can still be correct if
                    # they kill or push a phase before the back attack lands.
                    bias[idx] += 0.15 * risk
                    kaiser_pressure_candidate_count += 1
            if kaiser_facing_change_candidate_count == 0:
                self._dump_kaiser_facing_diagnostic(raw_obs, legal_actions, mask_np, encounter, risk)

        if ceremonial_one_card_lock > 0.05 and ("ceremonial" in encounter or not encounter):
            for idx in range(legal_count):
                if mask_np[idx] <= 0:
                    continue
                action = legal_actions[idx]
                family = self._semantic_family(action)
                impact = self._action_immediate_impact(action)
                roles = self._action_roles(action)
                if family == "end_turn":
                    bias[idx] -= 1.25
                elif family in {"use_potion", "potion"}:
                    low_impact, high_impact = self._ceremonial_action_timing_flags(
                        action,
                        idx,
                        obs,
                        raw_obs,
                        legal_actions,
                        mask_np,
                        energy,
                    )
                    if high_impact:
                        bias[idx] += 0.65 + (0.30 if ceremonial_stun_window > 0.05 else 0.0)
                        ceremonial_high_impact_count += 1
                    elif low_impact:
                        bias[idx] -= 0.85
                        ceremonial_low_impact_count += 1
                elif family == "play_card":
                    if impact >= 12.0:
                        bias[idx] += 0.75 + (0.35 if ceremonial_stun_window > 0.05 else 0.0)
                        ceremonial_high_impact_count += 1
                    elif impact <= 2.0 and not roles.intersection({"draw", "energy", "scaling", "power"}):
                        bias[idx] -= 0.95
                        ceremonial_low_impact_count += 1

        legal_mask_fixed = np.zeros(MAX_ACTIONS, dtype=bool)
        valid_len = min(mask_np.shape[0], MAX_ACTIONS)
        if valid_len > 0:
            legal_mask_fixed[:valid_len] = mask_np[:valid_len] > 0
        applied_mask = legal_mask_fixed & (np.abs(bias[:MAX_ACTIONS]) > 1e-6)
        stats = {
            "combat_quality_bias_applied": 1.0 if bool(applied_mask.any()) else 0.0,
            "combat_quality_bias_abs_mean": float(np.mean(np.abs(bias[:MAX_ACTIONS][legal_mask_fixed]))) if bool(legal_mask_fixed.any()) else 0.0,
            # State-level: an end_turn candidate exists while positive progress is still available.
            # This is an availability/bias-applied signal, not necessarily the selected action.
            "combat_quality_wasteful_end_turn_bias_applied": 1.0 if wasteful and bool(end_turn_indices) else 0.0,
            "combat_quality_wasteful_end_turn_available": 1.0 if wasteful and bool(end_turn_indices) else 0.0,
            "combat_quality_end_turn_penalty_max": float(end_turn_penalty),
            "combat_quality_energy": float(energy),
            "combat_quality_positive_action_count": float(len(positive_indices)),
            "combat_quality_urgent_positive_action_count": float(len(urgent_positive_indices)),
            "combat_quality_deferable_positive_action_count": float(len(deferable_positive_indices)),
            "combat_quality_safe_progress_candidate_count": float(context.get("safe_progress_candidate_count", 0) or 0),
            "combat_quality_deferable_exhaust_card_count": float(len(deferable_exhaust_indices)),
            "combat_quality_energy_gain_without_followup_count": float(len(energy_gain_without_followup_indices)),
            "combat_quality_typed_followup_missing_count": float(len(typed_followup_missing_indices)),
            "combat_quality_typed_future_penalty_count": float(len(typed_future_penalty_indices)),
            "combat_quality_typed_no_draw_count": float(len(typed_no_draw_indices)),
            "combat_quality_typed_card_state_mutation_count": float(len(typed_card_state_mutation_indices)),
            "combat_quality_card_block_waste_count": float(len(card_block_waste_indices)),
            "combat_quality_card_pure_block_count": float(len(card_pure_block_indices)),
            "combat_quality_card_no_damage_pressure_count": float(len(card_no_damage_pressure_indices)),
            "combat_quality_card_block_waste_bias_count": float(card_block_waste_bias_count),
            "combat_quality_card_block_waste_bias_min": float(card_block_waste_bias_min),
            "combat_quality_card_block_waste_hard_bias_applied": float(card_block_waste_bias_stats.get("card_block_waste_hard_bias_applied", 0.0)),
            "combat_quality_card_block_waste_progress_alternative": float(card_block_waste_bias_stats.get("card_block_waste_progress_alternative", 0.0)),
            "combat_quality_card_block_waste_progress_bonus_count": float(card_block_waste_bias_stats.get("card_block_waste_progress_bonus_count", 0.0)),
            "combat_quality_card_block_waste_progress_bonus_max": float(card_block_waste_bias_stats.get("card_block_waste_progress_bonus_max", 0.0)),
            "combat_quality_setup_followup_dependent_count": float(len(setup_followup_dependent_indices)),
            "combat_quality_setup_followup_available_count": float(len(setup_followup_available_indices)),
            "combat_quality_potion_available_count": float(len(potion_available_indices)),
            "combat_quality_potion_urgent_count": float(len(potion_urgent_indices)),
            "combat_quality_potion_low_urgency_count": float(len(potion_low_urgency_indices)),
            "combat_quality_potion_save_recommended_count": float(len(potion_save_recommended_indices)),
            "combat_quality_potion_no_followup_count": float(len(potion_no_followup_indices)),
            "combat_quality_potion_lethal_count": float(len(potion_lethal_indices)),
            "combat_quality_potion_prevent_lethal_count": float(len(potion_prevent_lethal_indices)),
            "combat_quality_potion_mechanism_count": float(len(potion_mechanism_indices)),
            "combat_quality_potion_overkill_count": float(len(potion_overkill_indices)),
            "combat_quality_potion_block_waste_count": float(len(potion_block_waste_indices)),
            "combat_quality_potion_use_quality_mean": float(context.get("potion_use_quality_mean", 0.0) or 0.0),
            "combat_quality_potion_waste_risk_mean": float(context.get("potion_waste_risk_mean", 0.0) or 0.0),
            "combat_quality_potion_save_value_mean": float(context.get("potion_save_value_mean", 0.0) or 0.0),
            "combat_quality_potion_hand_context_good_count": float(context.get("potion_hand_context_good_count", 0) or 0),
            "combat_quality_potion_hand_context_bad_count": float(context.get("potion_hand_context_bad_count", 0) or 0),
            "combat_quality_potion_long_term_count": float(context.get("potion_long_term_count", 0) or 0),
            "combat_quality_potion_requires_followup_count": float(context.get("potion_requires_followup_count", 0) or 0),
            "combat_quality_bad_potion_bias_count": float(bad_potion_bias_count),
            "combat_quality_bad_potion_bias_min": float(bad_potion_bias_min),
            "combat_quality_card_selection_confirm_ready_available": 1.0 if card_selection_confirm_ready_available else 0.0,
            "combat_quality_card_selection_confirm_ready_confirm_count": float(card_selection_confirm_ready_confirm_count),
            "combat_quality_card_selection_confirm_ready_select_count": float(card_selection_confirm_ready_select_count),
            "combat_quality_card_selection_deselect_candidate_count": float(card_selection_deselect_candidate_count),
            "combat_quality_card_selection_remaining_zero_select_count": float(card_selection_remaining_zero_select_count),
            "combat_quality_card_selection_not_ready_select_count": float(card_selection_not_ready_select_count),
            "combat_quality_strategic_defer_available": 1.0 if (
                bool(context.get("strategic_defer_available", False))
                and bool(end_turn_indices)
                and float(context.get("safe_progress_candidate_count", 0) or 0) > 0.5
            ) else 0.0,
            "combat_quality_true_wasteful_end_turn_available": 1.0 if wasteful and bool(end_turn_indices) else 0.0,
            "combat_quality_bad_end_turn_available": 1.0 if (wasteful or (
                bool(end_turn_indices)
                and (is_kaiser_encounter and kaiser_back_attack_risk > 0.05 and kaiser_facing_change_candidate_count >= 1)
            )) else 0.0,
            "combat_quality_forced_end_turn_available": 1.0 if (
                bool(end_turn_indices) and not wasteful and len(positive_indices) == 0
            ) else 0.0,
            "combat_quality_playable_action_count": float(len(playable_indices)),
            "combat_quality_end_turn_severity": float(severity),
            "combat_quality_x_cost_available_count": float(sum(
                1
                for _idx in range(legal_count)
                if mask_np[_idx] > 0 and self._semantic_family(legal_actions[_idx]) == "play_card" and self._is_x_cost_action(obs, _idx, legal_actions[_idx])
            )),
            "combat_quality_zero_energy_x_cost_count": float(len(zero_energy_x_indices)),
            "combat_quality_x_cost_bad_count": float(len(x_cost_bad_indices)),
            "combat_quality_x_cost_effective_energy_sum": float(x_cost_effective_energy_sum),
            "combat_quality_x_cost_effective_energy_mean": (
                float(x_cost_effective_energy_sum / max(len(x_cost_indices), 1))
                if x_cost_indices
                else 0.0
            ),
            "combat_quality_kaiser_back_attack_risk": float(kaiser_back_attack_risk),
            "combat_quality_kaiser_defense_candidate_count": float(kaiser_defense_candidate_count),
            "combat_quality_kaiser_facing_change_candidate_count": float(kaiser_facing_change_candidate_count),
            "combat_quality_kaiser_pressure_candidate_count": float(kaiser_pressure_candidate_count),
            "combat_quality_ceremonial_one_card_lock": float(ceremonial_one_card_lock),
            "combat_quality_ceremonial_stun_window": float(ceremonial_stun_window),
            "combat_quality_ceremonial_low_impact_count": float(ceremonial_low_impact_count),
            "combat_quality_ceremonial_high_impact_count": float(ceremonial_high_impact_count),
            "combat_quality_insatiable_sandpit_countdown": float(insatiable_sandpit_turns),
            "combat_quality_insatiable_sandpit_active": 1.0 if insatiable_sandpit_turns > 0.0 else 0.0,
            "combat_quality_insatiable_sandpit_lt3": 1.0 if 0.0 < insatiable_sandpit_turns < 3.0 else 0.0,
            "combat_quality_insatiable_sandpit_1": 1.0 if 0.0 < insatiable_sandpit_turns <= 1.0 else 0.0,
            "combat_quality_insatiable_frantic_escape_hand_count": float(insatiable_frantic_hand_count),
            "combat_quality_insatiable_frantic_escape_draw_count": float(insatiable_frantic_draw_count),
            "combat_quality_insatiable_frantic_escape_discard_count": float(insatiable_frantic_discard_count),
            "combat_quality_insatiable_frantic_escape_exhaust_count": float(insatiable_frantic_exhaust_count),
            "combat_quality_insatiable_frantic_escape_total_count": float(insatiable_frantic_total_count),
            "combat_quality_insatiable_frantic_escape_candidate_count": float(insatiable_frantic_escape_candidate_count),
            "combat_quality_insatiable_frantic_escape_available": 1.0 if insatiable_frantic_escape_candidate_count > 0 else 0.0,
            "combat_quality_insatiable_frantic_escape_urgency": float(insatiable_frantic_escape_urgency),
            "combat_quality_insatiable_frantic_escape_bonus_applied": 1.0 if insatiable_frantic_escape_bonus_applied else 0.0,
            "combat_quality_insatiable_non_escape_at1_penalty_count": float(insatiable_non_escape_at1_penalty_count),
            "combat_quality_insatiable_escape_cycle_risk": float(insatiable_escape_cycle_risk),
            "combat_quality_insatiable_lethal_candidate_count": float(insatiable_lethal_candidate_count),
        }
        return bias, stats, zero_energy_x_indices

    def _ceremonial_action_timing_flags(
        self,
        action: Any,
        action_idx: int,
        obs: dict[str, Any] | None,
        raw_obs: dict[str, Any] | None,
        legal_actions: list[Any] | None,
        mask_np: np.ndarray,
        energy: float,
    ) -> tuple[bool, bool]:
        """Classify one-card-lock actions as low/high timing quality.

        Ceremonial Beast's one-card-lock punishes spending the single allowed
        card/action on a low-impact move.  Potion actions must use the same
        timing-aware profile as the direct planner; otherwise diagnostics would
        regress to the old "potion immediate impact" view and again label almost
        every potion as a reasonable one-card-lock spend.

        Returns ``(low_impact, high_impact)``.
        """

        family = self._semantic_family(action)
        impact = self._action_immediate_impact(action)
        if family in {"use_potion", "potion"}:
            profile = self._potion_timing_profile(action, action_idx, obs, raw_obs, legal_actions, mask_np, energy)
            use_quality = float(profile.get("use_quality", 0.0) or 0.0)
            high_impact = bool(profile.get("urgent", False) or use_quality >= 0.55)
            low_impact = bool(
                profile.get("low_urgency", False)
                or profile.get("save_recommended", False)
                or profile.get("no_followup", False)
                or profile.get("block_waste", False)
                or (use_quality < 0.25 and impact <= 2.0)
            )
            return low_impact, high_impact

        if family == "play_card":
            roles = self._action_roles(action)
            high_impact = bool(impact >= 12.0)
            low_impact = bool(impact <= 2.0 and not roles.intersection({"draw", "energy", "scaling", "power"}))
            return low_impact, high_impact

        return False, False

    def _pure_block_selected_quality_flags(
        self,
        *,
        action_idx: int,
        action: Any,
        raw_obs: dict[str, Any] | None,
        legal_actions: list[Any] | None,
        energy: float,
        card_block_profile: dict[str, Any],
        action_mask: Any | None = None,
    ) -> dict[str, float]:
        """Split selected pure-block cards into bad vs survival/no-alt buckets.

        ``card_pure_block_selected`` is intentionally broad: a Defend-like card
        selected under 26 incoming damage is very different from Defend under a
        buffing/no-damage enemy.  Full-run monitoring was overreacting to the
        broad rate, so expose a narrower actionable signal:

        * bad_pure_block_selected: selected pure block was not answering a
          meaningful pressure window and a safe play-card progress alternative
          existed;
        * survival_justified: block answers lethal/high/low-HP/elite pressure;
        * no_alternative: no safe play-card progress alternative was legal.

        This is diagnostic only; hard rewrites remain in
        ``no_pressure_block_guard``.
        """

        zero = {
            "bad": 0.0,
            "insufficient": 0.0,
            "survival": 0.0,
            "progress_alt": 0.0,
            "no_alt": 0.0,
            "low_value_pressure": 0.0,
        }
        if not bool(card_block_profile.get("pure_block", False)):
            return zero
        if not isinstance(raw_obs, dict) or not isinstance(legal_actions, list):
            return zero

        incoming, current_block, current_hp = self._incoming_damage_pressure(raw_obs)
        threat_gap = max(0.0, float(incoming) - float(current_block))
        encounter_tier = self._combat_encounter_tier_from_raw(raw_obs)
        selected_block = max(
            self._action_metric(action, "block"),
            self._action_metric(action, "total_block"),
            self._action_numeric_value(action, ("block", "total_block", "preview_block", "expected_block")),
            0.0,
        )
        low_value_pressure = False
        try:
            low_value_pressure = bool(
                self._no_pressure_block_guard_low_value_pressure_window(
                    selected_block=selected_block,
                    threat_gap=threat_gap,
                    current_hp=float(current_hp),
                    encounter_tier=encounter_tier,
                )
            )
        except Exception:
            low_value_pressure = False
        outcome = protection_outcome(
            hp=float(current_hp),
            threat_gap=float(threat_gap),
            block=float(selected_block),
        )
        insufficient = bool(outcome.insufficient)
        survival_justified = bool(
            self._is_meaningful_block_urgent(
                block=selected_block,
                threat_gap=threat_gap,
                current_hp=float(current_hp),
                incoming=float(incoming),
                encounter_tier=encounter_tier,
            )
            and not low_value_pressure
            and not insufficient
        )

        mask_np = np.ones(MAX_ACTIONS, dtype=np.float32)
        use_mask = False
        if action_mask is not None:
            try:
                candidate_mask = np.asarray(action_mask, dtype=np.float32).reshape(-1)
                if candidate_mask.size > 0:
                    mask_np = candidate_mask
                    use_mask = True
            except Exception:
                mask_np = np.ones(MAX_ACTIONS, dtype=np.float32)
                use_mask = False
        legal_count = min(len(legal_actions), MAX_ACTIONS)
        progress_candidates, _debug_counts = collect_safe_progress_candidates(
            self,
            selected_idx=int(action_idx),
            legal_count=legal_count,
            legal_actions=legal_actions,
            mask_np=mask_np,
            raw_obs=raw_obs,
            current_energy=float(energy),
            # Prefer the real post-search mask when threaded from self-play so
            # selected-side diagnostics and hard guards see the same candidate
            # surface.  Older/unit-test call sites can omit action_mask and keep
            # the previous semantic-only fallback.
            use_mask=use_mask,
            include_debug=False,
        )

        progress_alt = len(progress_candidates) > 0
        return {
            "bad": 1.0 if (progress_alt and not survival_justified) else 0.0,
            "insufficient": 1.0 if insufficient else 0.0,
            "survival": 1.0 if survival_justified else 0.0,
            "progress_alt": 1.0 if progress_alt else 0.0,
            "no_alt": 1.0 if not progress_alt else 0.0,
            "low_value_pressure": 1.0 if low_value_pressure else 0.0,
        }

    def _selected_combat_quality_stats(
        self,
        obs: dict[str, Any] | None,
        action_idx: int,
        legal_actions: list[Any] | None,
        search_stats: dict[str, Any] | None,
        action_mask: Any | None = None,
    ) -> dict[str, float]:
        """Selection-side combat diagnostics that do not depend on planner mode.

        MCTS, direct-policy, and direct-rollout all eventually choose one legal action.
        These metrics classify that chosen action against the *current* raw combat energy
        and the already-computed root availability flags, so zero-energy X-cost and
        strategic-defer/true-wasteful rates have one consistent definition.
        """
        if not isinstance(legal_actions, list) or action_idx < 0 or action_idx >= len(legal_actions):
            return {}
        action = legal_actions[action_idx]
        stats = search_stats if isinstance(search_stats, dict) else {}
        raw_obs = self._current_raw_combat_obs()
        energy = self._combat_energy(obs, raw_obs)
        mask_np = np.ones(MAX_ACTIONS, dtype=np.float32)
        mask_is_real = False
        if action_mask is not None:
            try:
                candidate_mask = np.asarray(action_mask, dtype=np.float32).reshape(-1)
                if candidate_mask.size > 0:
                    mask_np = candidate_mask
                    mask_is_real = True
            except Exception:
                mask_np = np.ones(MAX_ACTIONS, dtype=np.float32)
                mask_is_real = False
        family = self._semantic_family(action)
        end_turn_selected = family == "end_turn"
        x_selected = bool(family == "play_card" and self._is_x_cost_action(obs, action_idx, action))
        zero_x_selected = bool(x_selected and energy <= 0.05)
        x_diag = self._x_cost_diagnostic(action, float(energy)) if x_selected else {}
        x_has_non_energy = float(x_diag.get("x_cost_has_non_energy_effect", 0.0) or 0.0) > 0.5
        x_bad_selected = bool(x_selected and float(x_diag.get("x_cost_bad", 0.0) or 0.0) > 0.5)
        true_waste_available = float(stats.get("combat_quality_true_wasteful_end_turn_available", stats.get("combat_quality_wasteful_end_turn_available", 0.0)) or 0.0) > 0.5
        safe_progress_available = float(stats.get("combat_quality_safe_progress_candidate_count", 0.0) or 0.0) > 0.5
        strategic_defer_available = (
            float(stats.get("combat_quality_strategic_defer_available", 0.0) or 0.0) > 0.5
            and safe_progress_available
        )
        potion_selected = family in {"use_potion", "potion"}
        potion_available = float(stats.get("combat_quality_potion_available_count", 0.0) or 0.0) > 0.0
        is_frantic_selected = bool(family == "play_card" and self._is_frantic_escape_action(action))
        insat_available = float(stats.get("combat_quality_insatiable_frantic_escape_available", 0.0) or 0.0) > 0.5
        sandpit_lt3 = float(stats.get("combat_quality_insatiable_sandpit_lt3", 0.0) or 0.0) > 0.5
        sandpit_at1 = float(stats.get("combat_quality_insatiable_sandpit_1", 0.0) or 0.0) > 0.5
        lethal_selected = self._is_action_confirmed_lethal(action, raw_obs)
        card_block_profile: dict[str, Any] = {}
        if family == "play_card":
            try:
                card_block_profile = self._card_block_waste_profile(action, raw_obs)
            except Exception:
                card_block_profile = {}
        pure_block_quality = self._pure_block_selected_quality_flags(
            action_idx=int(action_idx),
            action=action,
            raw_obs=raw_obs,
            legal_actions=legal_actions,
            energy=float(energy),
            card_block_profile=card_block_profile,
            action_mask=mask_np if mask_is_real else None,
        )
        refund_bad_selected = False
        refund_lethal_exemption = False
        refund_progress_alt_count = 0
        if family == "play_card" and isinstance(raw_obs, dict) and isinstance(legal_actions, list):
            try:
                refund_profile = self._classify_positive_combat_action(
                    action,
                    int(action_idx),
                    obs,
                    raw_obs,
                    legal_actions,
                    mask_np,
                    float(energy),
                )
            except Exception:
                refund_profile = {}
            try:
                refund_bad_selected = bool(self._refund_no_followup_guard_selected_bad(refund_profile))
            except Exception:
                refund_bad_selected = False
            refund_lethal_exemption = bool(refund_bad_selected and lethal_selected)
            if refund_bad_selected and not refund_lethal_exemption:
                try:
                    refund_progress_alt_count = len(
                        self._refund_no_followup_progress_candidates(
                            selected_idx=int(action_idx),
                            legal_count=min(len(legal_actions), MAX_ACTIONS),
                            legal_actions=legal_actions,
                            mask_np=mask_np,
                            raw_obs=raw_obs,
                            current_energy=float(energy),
                        )
                    )
                except Exception:
                    refund_progress_alt_count = 0
        potion_profile: dict[str, Any] = {}
        if potion_selected:
            potion_profile = self._potion_timing_profile(action, action_idx, obs, raw_obs, legal_actions, mask_np, energy)
        # TASK-B1: re-classify the selected end_turn through the central
        # taxonomy so the selected-side rate uses the same priority as the
        # bias-side context.  Reconstruct a minimal context dict from the
        # already-computed bias stats — avoids re-walking legal_actions and
        # keeps the taxonomy frame-consistent with the bias judgment.
        end_turn_class = "unknown"
        end_turn_class_flags: dict[str, bool] = {}
        if end_turn_selected:
            taxonomy_context = {
                "end_turn_indices": [int(action_idx)],
                "wasteful": bool(true_waste_available),
                "strategic_defer_available": bool(strategic_defer_available),
                "positive_progress_count": int(float(stats.get("combat_quality_positive_action_count", 0.0) or 0.0)),
                "urgent_positive_count": int(float(stats.get("combat_quality_urgent_positive_action_count", 0.0) or 0.0)),
                "deferable_positive_count": int(float(stats.get("combat_quality_deferable_positive_action_count", 0.0) or 0.0)),
                "energy": float(stats.get("combat_quality_energy", energy) or energy),
            }
            boss_signals = {
                "kaiser_back_attack_risk": float(stats.get("combat_quality_kaiser_back_attack_risk", 0.0) or 0.0),
                "kaiser_facing_change_candidate_count": float(stats.get("combat_quality_kaiser_facing_change_candidate_count", 0.0) or 0.0),
                "kaiser_defense_candidate_count": float(stats.get("combat_quality_kaiser_defense_candidate_count", 0.0) or 0.0),
                "ceremonial_stun_window": float(stats.get("combat_quality_ceremonial_stun_window", 0.0) or 0.0),
                "ceremonial_high_impact_count": float(stats.get("combat_quality_ceremonial_high_impact_count", 0.0) or 0.0),
            }
            end_turn_class, end_turn_class_flags = self._classify_end_turn_action(
                taxonomy_context,
                None,
                boss_signals,
            )
        bad_selected = end_turn_selected and end_turn_class == "bad_end_turn"
        forced_selected = end_turn_selected and end_turn_class == "forced_end_turn"
        defer_selected_taxonomy = end_turn_selected and end_turn_class == "strategic_defer_end_turn"
        unknown_selected = end_turn_selected and end_turn_class == "unknown"
        result = {
            "combat_quality_x_cost_selected": 1.0 if x_selected else 0.0,
            "combat_quality_x_cost_selected_energy": float(energy) if x_selected else 0.0,
            "combat_quality_x_cost_zero_energy_selected": 1.0 if zero_x_selected else 0.0,
            "combat_quality_zero_energy_x_cost_selected": 1.0 if zero_x_selected else 0.0,
            "combat_quality_x_cost_selected_effective_energy": float(x_diag.get("x_cost_effective_energy", 0.0) or 0.0) if x_selected else 0.0,
            "combat_quality_x_cost_has_non_energy_effect_selected": 1.0 if (x_selected and x_has_non_energy) else 0.0,
            "combat_quality_x_cost_bad_selected": 1.0 if x_bad_selected else 0.0,
            "combat_quality_end_turn_selected": 1.0 if end_turn_selected else 0.0,
            # Legacy alias kept so existing dashboards stay readable.  The
            # taxonomy view (bad/forced/strategic_defer) is the new source of
            # truth — bad_selected is wider than wasteful because it folds in
            # boss-pressure windows.
            "combat_quality_true_wasteful_end_turn_selected": 1.0 if (end_turn_selected and true_waste_available) else 0.0,
            "combat_quality_wasteful_end_turn_selected": 1.0 if bad_selected else (1.0 if (end_turn_selected and true_waste_available) else 0.0),
            "combat_quality_bad_end_turn_selected": 1.0 if bad_selected else 0.0,
            "combat_quality_forced_end_turn_selected": 1.0 if forced_selected else 0.0,
            "combat_quality_strategic_defer_end_turn_selected": 1.0 if (defer_selected_taxonomy or (end_turn_selected and strategic_defer_available and not bad_selected)) else 0.0,
            "combat_quality_end_turn_unknown_selected": 1.0 if unknown_selected else 0.0,
            "combat_quality_end_turn_class": end_turn_class,
        }
        card_no_damage_pressure_context = bool(card_block_profile.get("no_damage_pressure", False))
        card_no_damage_pressure_bad = bool(
            card_no_damage_pressure_context and bool(card_block_profile.get("pure_block", False))
        )
        block_waste_selected = bool(card_block_profile.get("block_waste", False))
        has_progress_alt = float(pure_block_quality.get("progress_alt", 0.0) or 0.0) > 0.5
        result.update(
            {
                "combat_quality_card_block_waste_selected": 1.0 if block_waste_selected else 0.0,
                # Narrow actionable versions used by sandbox gates.  The broad
                # block/no-pressure gauges intentionally remain visible, but
                # they include forced "Defend + End Turn only" hands.  These
                # narrow metrics only fire when the selected block had a real
                # progress alternative, so they track policy mistakes rather
                # than legal-action starvation.
                "combat_quality_card_block_waste_with_progress_selected": 1.0
                if (block_waste_selected and has_progress_alt)
                else 0.0,
                "combat_quality_card_pure_block_selected": 1.0 if bool(card_block_profile.get("pure_block", False)) else 0.0,
                "combat_quality_bad_pure_block_selected": float(pure_block_quality.get("bad", 0.0) or 0.0),
                "combat_quality_insufficient_block_selected": float(pure_block_quality.get("insufficient", 0.0) or 0.0),
                "combat_quality_pure_block_survival_justified_selected": float(pure_block_quality.get("survival", 0.0) or 0.0),
                "combat_quality_pure_block_progress_alternative_selected": float(pure_block_quality.get("progress_alt", 0.0) or 0.0),
                "combat_quality_pure_block_no_alternative_selected": float(pure_block_quality.get("no_alt", 0.0) or 0.0),
                "combat_quality_pure_block_low_value_pressure_selected": float(pure_block_quality.get("low_value_pressure", 0.0) or 0.0),
                "combat_quality_card_no_damage_pressure_selected": 1.0 if card_no_damage_pressure_bad else 0.0,
                "combat_quality_card_no_damage_pressure_with_progress_selected": 1.0
                if (card_no_damage_pressure_bad and has_progress_alt)
                else 0.0,
                # Narrow actionable refund/setup metric.  The broad
                # ``refund_no_followup_selected`` flag is still kept because it
                # describes card semantics, but sandbox gates should fail only
                # when a safe immediate-progress alternative existed.
                "combat_quality_refund_no_followup_with_progress_selected": 1.0
                if (refund_bad_selected and refund_progress_alt_count > 0 and not refund_lethal_exemption)
                else 0.0,
                "combat_quality_refund_no_followup_progress_alternative_selected": 1.0
                if (refund_bad_selected and refund_progress_alt_count > 0 and not refund_lethal_exemption)
                else 0.0,
                "combat_quality_refund_no_followup_no_alternative_selected": 1.0
                if (refund_bad_selected and refund_progress_alt_count <= 0 and not refund_lethal_exemption)
                else 0.0,
                "combat_quality_refund_no_followup_progress_alternative_count": float(refund_progress_alt_count),
                "combat_quality_potion_selected": 1.0 if potion_selected else 0.0,
                "combat_quality_potion_selected_when_available": 1.0 if (potion_selected and potion_available) else 0.0,
                "combat_quality_potion_high_urgency_selected": 1.0 if (potion_selected and bool(potion_profile.get("urgent", False))) else 0.0,
                "combat_quality_potion_low_urgency_selected": 1.0 if (potion_selected and bool(potion_profile.get("low_urgency", False))) else 0.0,
                "combat_quality_potion_save_recommended_selected": 1.0 if (potion_selected and bool(potion_profile.get("save_recommended", False))) else 0.0,
                "combat_quality_potion_no_followup_selected": 1.0 if (potion_selected and bool(potion_profile.get("no_followup", False))) else 0.0,
                "combat_quality_potion_lethal_selected": 1.0 if (potion_selected and bool(potion_profile.get("lethal", False))) else 0.0,
                "combat_quality_potion_prevent_lethal_selected": 1.0 if (potion_selected and bool(potion_profile.get("prevent_lethal", False))) else 0.0,
                "combat_quality_potion_mechanism_selected": 1.0 if (potion_selected and bool(potion_profile.get("mechanism_answer", False))) else 0.0,
                "combat_quality_potion_overkill_selected": 1.0 if (potion_selected and bool(potion_profile.get("overkill", False))) else 0.0,
                "combat_quality_potion_block_waste_selected": 1.0 if (potion_selected and bool(potion_profile.get("block_waste", False))) else 0.0,
                "combat_quality_potion_use_quality_selected": float(potion_profile.get("use_quality", 0.0) or 0.0) if potion_selected else 0.0,
                "combat_quality_potion_waste_risk_selected": float(potion_profile.get("waste_risk", 0.0) or 0.0) if potion_selected else 0.0,
            }
        )
        result.update(
            {
                "combat_quality_insatiable_frantic_escape_selected": 1.0 if is_frantic_selected else 0.0,
                "combat_quality_insatiable_frantic_escape_missed_lt3": 1.0 if (
                    insat_available and sandpit_lt3 and not is_frantic_selected and not lethal_selected
                ) else 0.0,
                "combat_quality_insatiable_frantic_escape_missed_at_1": 1.0 if (
                    insat_available and sandpit_at1 and not is_frantic_selected and not lethal_selected
                ) else 0.0,
                "combat_quality_insatiable_non_escape_at_1_selected": 1.0 if (
                    insat_available and sandpit_at1 and not is_frantic_selected and not lethal_selected
                ) else 0.0,
            }
        )
        return result
