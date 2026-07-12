# ruff: noqa: RUF003
"""Deprecated-v1 combat reward and transition diagnostics.

Canonical v2 backends own reward through ``EnvironmentRuntimeMixin``; this
mixin remains an isolated compatibility implementation.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from ._combat_env_values import _float
from .boss_mechanics import build_boss_mechanics_context
from .end_turn_quality import strict_end_turn_waste_context
from .potion_timing import compute_potion_timing
from .reward_constants import (
    BOSS_COMBAT_LOSS_DAMAGE_UNDO_PERCENT_SCALE,
    BOSS_COMBAT_LOSS_PENALTY_BASE,
    BOSS_COMBAT_LOSS_PENALTY_MISSING_HP_SCALE,
    BOSS_COMBAT_WIN_BONUS_BASE,
    BOSS_COMBAT_WIN_BONUS_HP_SCALE,
    BOSS_ENEMY_HP_DELTA_PERCENT_SCALE,
    CEREMONIAL_ONE_CARD_END_TURN_PENALTY,
    CEREMONIAL_ONE_CARD_HIGH_IMPACT_BONUS,
    CEREMONIAL_ONE_CARD_LOW_IMPACT_PENALTY,
    CEREMONIAL_STUN_DAMAGE_MULTIPLIER,
    CEREMONIAL_STUN_WINDOW_ENTER_BONUS,
    CEREMONIAL_THRESHOLD_PROGRESS_BONUS,
    ENEMY_HP_DELTA_REWARD_MAX_ABS,
    ENEMY_HP_DELTA_REWARD_SCALE,
    ENEMY_HP_SENTINEL_THRESHOLD,
    HP_PRESERVE_WIN_BONUS_TIER_SCALE,
    KAISER_BACK_ATTACK_DEFENSE_BONUS,
    KAISER_BACK_ATTACK_END_TURN_PENALTY,
    KAISER_BACK_ATTACK_HI_THREAT_DMG,
    KAISER_BACK_ATTACK_HI_THREAT_EXTRA,
    KAISER_BACK_ATTACK_HP_LOSS_PENALTY_SCALE,
    KAISER_BACK_ATTACK_RISK_REDUCTION_BONUS,
    KAISER_FACING_CHANGE_BONUS,
    KAISER_FACING_CHANGE_BONUS_BASE,
    KAISER_FACING_INTENT_DMG_REF,
    KAISER_FACING_INTENT_DMG_SCALE_MAX,
    KAISER_NO_RESPONSE_PENALTY_SOFTEN,
    KAISER_PRESSURE_KILL_BONUS,
    KNOWLEDGE_DEMON_BAD_CURSE_PICK_PENALTY,
    KNOWLEDGE_DEMON_END_TURN_PENALTY,
    KNOWLEDGE_DEMON_GOOD_CURSE_PICK_BONUS,
    OUTCOME_TIER_SCALE,
    PLAYER_HP_LOSS_REWARD_SCALE,
    PLAYER_HP_LOSS_TIER_SCALE,
    POTION_HOARDING_MAX_PENALTY_ABS,
    POTION_HOARDING_PENALTY_PER_POTION,
    POTION_TIMING_QUALITY_SCALE,
    POTION_TIMING_WASTE_SCALE,
    POTION_USE_BOSS_BONUS,
    POTION_USE_ELITE_BONUS,
    POTION_USE_MONSTER_BONUS,
    POTION_USE_MONSTER_PENALTY,
    SELECTION_DESELECT_PENALTY,
    SELECTION_EARLY_CONFIRM_BONUS,
    SELECTION_LOOP_PENALTY,
    SELECTION_OVER_CAP_PENALTY,
    SELECTION_PICK_CAP,
    SELECTION_REENTRY_BUDGET,
    SELECTION_REENTRY_PENALTY,
    SENTINEL_COMBAT_LOSS_PENALTY_BASE,
    SENTINEL_COMBAT_LOSS_PENALTY_SCALE,
    SENTINEL_COMBAT_WIN_BONUS_BASE,
    SENTINEL_COMBAT_WIN_BONUS_SCALE,
    SENTINEL_DEATH_DAMAGE_POWER_KEYWORDS,
    TURN_EFFICIENCY_PENALTY_PER_END_TURN_TIER,
    WASTEFUL_END_TURN_TIER_MULTIPLIER,
)
from .reward_constants import (
    COMBAT_SANDBOX_WASTE_BASE as END_TURN_WASTE_BASE_PENALTY,
)
from .reward_constants import (
    COMBAT_SANDBOX_WASTE_ENERGY as END_TURN_WASTE_ENERGY_PENALTY,
)
from .reward_constants import (
    COMBAT_SANDBOX_WASTE_EXTRA_ACTION as END_TURN_WASTE_EXTRA_ACTION_PENALTY,
)
from .reward_constants import (
    COMBAT_SANDBOX_WASTE_ZERO_COST as END_TURN_WASTE_ZERO_COST_BONUS_PENALTY,
)


class LegacyCombatRewardMixin:
    @staticmethod
    def _potion_slots_dump(obs: dict[str, Any] | None) -> list[dict[str, Any]]:
        """Return a normalised view of every potion slot for diagnostic dumps."""
        player = (obs or {}).get("player") if isinstance(obs, dict) else {}
        potions = player.get("potions") if isinstance(player, dict) else None
        if not isinstance(potions, list):
            return []
        EMPTY_NAMES = {"empty", "[empty]", "none", "null", ""}
        out: list[dict[str, Any]] = []
        for idx, potion in enumerate(potions):
            if isinstance(potion, dict):
                name = str(potion.get("name") or potion.get("id") or potion.get("title") or "").strip().lower()
                empty = bool(potion.get("empty")) or (not name) or (name in EMPTY_NAMES)
                out.append({
                    "slot": idx,
                    "id": potion.get("id"),
                    "title": potion.get("title") or potion.get("name"),
                    "empty": empty,
                    "is_usable": bool(potion.get("is_usable", not empty)),
                    "is_queued": bool(potion.get("is_queued", False)),
                })
            else:
                name = str(potion or "").strip().lower()
                out.append({
                    "slot": idx,
                    "id": None,
                    "title": str(potion) if potion is not None else None,
                    "empty": (not name) or (name in EMPTY_NAMES),
                    "is_usable": False,
                    "is_queued": False,
                })
        return out

    def _build_potion_transition_record(
        self,
        *,
        action: dict[str, Any] | None,
        prev_obs: dict[str, Any] | None,
        after_obs: dict[str, Any] | None,
        bridge_result: dict[str, Any] | None,
        bridge_error: str | None,
    ) -> dict[str, Any] | None:
        """Assemble a use_potion transition record for the diagnostics JSONL.

        Returns None for non-potion actions.  When ``bridge_error`` is set the
        record is still returned so we can post-mortem failed potion uses.
        """
        if self._action_family(action) not in {"use_potion", "potion"}:
            return None
        result = bridge_result if isinstance(bridge_result, dict) else {}
        info_block = result.get("info") if isinstance(result.get("info"), dict) else {}
        execute_ok = bridge_error is None and not bool(info_block.get("error"))
        # State versions: bridge serialises an integer state version per result; if
        # missing, fall back to obs.state_version from the raw frame.
        def _state_version(obs: dict[str, Any] | None) -> int:
            if not isinstance(obs, dict):
                return 0
            for container in (obs.get("meta"), obs):
                if not isinstance(container, dict):
                    continue
                for key in ("state_version", "stateVersion"):
                    if key in container:
                        try:
                            return int(container.get(key) or 0)
                        except (TypeError, ValueError):
                            return 0
            return 0
        slot_index = -1
        target_block = action.get("target") if isinstance(action.get("target"), dict) else {}
        for key in ("slot_index", "potion_slot", "slot"):
            for source in (action, target_block, action.get("potion") if isinstance(action.get("potion"), dict) else {}):
                if isinstance(source, dict) and key in source:
                    try:
                        slot_index = int(source.get(key))
                        break
                    except (TypeError, ValueError):
                        continue
            if slot_index >= 0:
                break
        potion_block = action.get("potion") if isinstance(action.get("potion"), dict) else {}
        before_dump = self._potion_slots_dump(prev_obs)
        after_dump = self._potion_slots_dump(after_obs)
        before_slot = next((slot for slot in before_dump if slot.get("slot") == slot_index), None)
        after_slot = next((slot for slot in after_dump if slot.get("slot") == slot_index), None)
        return {
            "event": "use_potion_transition",
            "action_id": action.get("action_id"),
            "potion_slot": slot_index,
            "potion_id_before": (before_slot or {}).get("id") if before_slot else potion_block.get("id"),
            "potion_title_before": (before_slot or {}).get("title") if before_slot else potion_block.get("title"),
            "execute_ok": bool(execute_ok),
            "bridge_error": bridge_error,
            "state_version_before": _state_version(prev_obs),
            "state_version_after": _state_version(after_obs),
            "potion_slot_after": after_slot,
            "potion_slots_after": after_dump,
        }

    @staticmethod
    def _nonempty_potion_count(obs: dict[str, Any] | None) -> int:
        player = (obs or {}).get("player") if isinstance(obs, dict) else {}
        potions = player.get("potions") if isinstance(player, dict) else None
        if not isinstance(potions, list):
            return 0
        count = 0
        # STS2 bridge serializes empty potion slots as title="[empty]" — note
        # the BRACKETS.  An earlier exclusion set of {"empty","none","null"}
        # missed that form, so every empty slot was counted as a usable
        # potion.  That misled the potion-hoarding terminal reward and the
        # potion-use diagnostics.
        EMPTY_NAMES = {"empty", "[empty]", "none", "null", ""}
        for potion in potions:
            if not potion:
                continue
            if isinstance(potion, dict):
                if bool(potion.get("empty")):
                    continue
                name = str(potion.get("name") or potion.get("id") or potion.get("title") or "").strip().lower()
                if name and name not in EMPTY_NAMES:
                    count += 1
            else:
                name = str(potion or "").strip().lower()
                if name and name not in EMPTY_NAMES:
                    count += 1
        return count

    def _encounter_potion_use_reward(self, action: dict[str, Any] | None) -> float:
        if self._action_family(action) not in {"use_potion", "potion"}:
            return 0.0
        tier = self._current_encounter_tier()
        if tier == "boss":
            return float(POTION_USE_BOSS_BONUS)
        if tier == "elite":
            return float(POTION_USE_ELITE_BONUS)
        return float(POTION_USE_MONSTER_BONUS + POTION_USE_MONSTER_PENALTY)

    def _potion_hoarding_terminal_reward(self, after_obs: dict[str, Any] | None, terminated: bool, truncated: bool) -> float:
        if not (terminated or truncated):
            return 0.0
        unused = self._nonempty_potion_count(after_obs)
        if unused <= 0:
            return 0.0
        raw = float(POTION_HOARDING_PENALTY_PER_POTION) * float(unused)
        return float(np.clip(raw, -abs(float(POTION_HOARDING_MAX_PENALTY_ABS)), abs(float(POTION_HOARDING_MAX_PENALTY_ABS))))

    def _potion_timing_step_reward(
        self,
        action: dict[str, Any] | None,
        before_obs: dict[str, Any] | None,
        legal_actions: list[Any] | None,
    ) -> float:
        """Phase 4b of docs/potion-timing-modeling-plan.md (§2).

        Convert the timing profile (use_quality / waste_risk) into per-step
        reward shaping so the policy gradient actually learns "don't dump
        potions turn 1".  Penalty > bonus by design — model should prefer
        hoarding over bad use.
        """
        if self._action_family(action) not in {"use_potion", "potion"}:
            return 0.0
        if not isinstance(before_obs, dict):
            return 0.0
        try:
            energy = float(((before_obs.get("player") or {}).get("energy")) or 0.0)
        except (TypeError, ValueError):
            energy = 0.0
        encounter_tier = self._current_encounter_tier()
        try:
            profile = compute_potion_timing(
                action,
                before_obs,
                legal_actions,
                None,
                energy,
                encounter_tier=encounter_tier,
            )
        except Exception:
            return 0.0
        if not profile.get("is_potion"):
            return 0.0
        use_q = float(profile.get("use_quality") or 0.0)
        waste = float(profile.get("waste_risk") or 0.0)
        urgent = bool(
            profile.get("urgent")
            or profile.get("lethal")
            or profile.get("prevent_lethal")
            or profile.get("prevent_major_loss")
            or profile.get("mechanism_answer")
        )
        bad_timing = bool(
            profile.get("low_urgency")
            or profile.get("save_recommended")
            or profile.get("no_followup")
            or profile.get("block_waste")
            or profile.get("overkill")
        )
        reward = 0.0
        # The shared timing model has a small baseline use_quality.  Treating
        # any positive value as reward made the policy learn "use potion when
        # legal".  Reward only urgent / genuinely high-quality timing; convert
        # low-urgency, no-followup, overkill, or block-waste uses into waste.
        if urgent or (use_q >= 0.45 and not bad_timing):
            reward += float(POTION_TIMING_QUALITY_SCALE) * use_q
            self._potion_timing_quality_events += 1
        elif bad_timing:
            effective_waste = max(float(waste), 0.35)
            reward -= float(POTION_TIMING_WASTE_SCALE) * effective_waste
            self._potion_timing_waste_events += 1
        elif waste > 0.0:
            reward -= float(POTION_TIMING_WASTE_SCALE) * waste
            self._potion_timing_waste_events += 1
        return reward

    def _card_selection_step_reward(self, action: dict[str, Any] | None) -> float:
        """Anti-loop + early-confirm shaping for multi-pick burn cards.

        See §4 of docs/kaiser-and-potion-fixes-todo.md.

        Detects three failure modes:
          * Pick→replace cycle on SAME card (A→A): SELECTION_LOOP_PENALTY.
          * Oscillation across multiple cards (A→B→C→A): caught by
            SELECTION_DESELECT_PENALTY — each pick whose `is_selected=True`
            is a deselect, every deselect after the first pays the penalty.
          * Total-picks death loop: SELECTION_PICK_CAP=12 caps a single
            selection round; each pick beyond pays SELECTION_OVER_CAP_PENALTY.

        Tracks per-selection state in self._selection_last_picked_id /
        self._selection_flip_count / self._selection_pick_count /
        self._selection_deselect_count, which reset when the family
        transitions away from card_selection or on confirm/cancel/skip.
        """
        if not isinstance(action, dict):
            return 0.0
        family = self._action_family(action)
        sel_action = str(action.get("selection_action") or action.get("selection") or "").strip().lower()
        action_id = str(action.get("action_id") or "")

        def _reset_state() -> None:
            self._selection_last_picked_id = ""
            self._selection_flip_count = 0
            self._selection_pick_count = 0
            self._selection_deselect_count = 0

        # §4 v3: detect selection-screen entry/exit edges. Pay re-entry penalty
        # when the model bounces in and out of selection screens within the
        # same episode (the_insatiable frantic_escape pattern).
        reentry_penalty = 0.0
        if family == "card_selection" and not self._selection_screen_active:
            self._selection_screen_active = True
            self._selection_screen_entries += 1
            budget = int(SELECTION_REENTRY_BUDGET)
            if self._selection_screen_entries > budget:
                excess = self._selection_screen_entries - budget
                reentry_penalty = -float(SELECTION_REENTRY_PENALTY) * float(excess)
                self._selection_reentry_events += 1
        elif family != "card_selection" and self._selection_screen_active:
            self._selection_screen_active = False

        if family != "card_selection":
            if (self._selection_pick_count > 0
                or self._selection_flip_count > 0
                or self._selection_deselect_count > 0):
                _reset_state()
            return 0.0

        if sel_action == "confirm":
            picked = max(int(self._selection_pick_count), 0)
            max_picks = int(
                action.get("max_pick")
                or action.get("selection_max")
                or action.get("max_select")
                or action.get("max")
                or 0
            )
            if max_picks <= 0:
                max_picks = int(action.get("selection_pick_limit") or 0)
            reward = reentry_penalty
            if max_picks > 0 and picked < max_picks:
                ratio = float(max_picks - picked) / float(max_picks)
                reward += float(SELECTION_EARLY_CONFIRM_BONUS) * ratio
                self._selection_early_confirm_events += 1
            _reset_state()
            return reward

        if sel_action in {"cancel", "close", "skip"}:
            _reset_state()
            return reentry_penalty

        # Pick path: detect repeat-same, deselect-pattern, and over-cap.
        picked_id = ""
        card = action.get("card") if isinstance(action.get("card"), dict) else None
        if isinstance(card, dict):
            picked_id = str(card.get("id") or card.get("title") or "")
        if not picked_id and ":" in action_id:
            picked_id = action_id

        # Bridge marks `is_selected=True` on actions that toggle a card OFF
        # (the click would deselect it). Counting these directly catches
        # the A→B→A→B oscillation pattern that the legacy id-equality check
        # missed.
        raw_is_selected = action.get("is_selected")
        if isinstance(raw_is_selected, bool):
            is_deselect = raw_is_selected
        elif isinstance(raw_is_selected, int | float):
            is_deselect = float(raw_is_selected) != 0.0
        elif isinstance(raw_is_selected, str):
            is_deselect = raw_is_selected.strip().lower() in {"1", "true", "yes", "y", "on"}
        else:
            is_deselect = False

        reward = reentry_penalty
        # 1) Same-id repeat (legacy A→A→A check).
        if picked_id:
            if picked_id == self._selection_last_picked_id:
                self._selection_flip_count += 1
                if self._selection_flip_count >= 2:
                    reward -= float(SELECTION_LOOP_PENALTY) * float(self._selection_flip_count)
                    self._selection_loop_events += 1
            else:
                self._selection_flip_count = 0
            self._selection_last_picked_id = picked_id

        # 2) Deselect detection (oscillation across cards).
        if is_deselect:
            self._selection_deselect_count += 1
            if self._selection_deselect_count >= 2:
                # Linear escalation: 2nd deselect = -0.40, 3rd = -0.80, 4th = -1.20...
                reward -= float(SELECTION_DESELECT_PENALTY) * float(self._selection_deselect_count - 1)
                self._selection_loop_events += 1

        self._selection_pick_count += 1

        # 3) Hard pick-cap (kills the 1700-step death loop).
        if self._selection_pick_count > int(SELECTION_PICK_CAP):
            reward -= float(SELECTION_OVER_CAP_PENALTY)
            self._selection_over_cap_events += 1

        return reward


    @staticmethod
    def _boss_context_max(context: dict[str, Any], key: str) -> float:
        if not isinstance(context, dict):
            return 0.0
        vals: list[float] = []
        player_state = context.get("player_state") if isinstance(context.get("player_state"), dict) else {}
        if key in player_state:
            vals.append(_float(player_state.get(key)))
        enemy_states = context.get("enemy_states_by_index")
        if isinstance(enemy_states, list):
            vals.extend(_float((state or {}).get(key)) for state in enemy_states if isinstance(state, dict))
        return max(vals) if vals else 0.0

    @staticmethod
    def _action_semantic(action: dict[str, Any] | None) -> dict[str, Any]:
        return action.get("semantic") if isinstance(action, dict) and isinstance(action.get("semantic"), dict) else {}

    @classmethod
    def _action_roles(cls, action: dict[str, Any] | None) -> set[str]:
        semantic = cls._action_semantic(action)
        roles = semantic.get("roles")
        if not isinstance(roles, list):
            return set()
        return {str(role).strip().lower() for role in roles if str(role).strip()}

    @classmethod
    def _action_metric(cls, action: dict[str, Any] | None, key: str) -> float:
        semantic = cls._action_semantic(action)
        if key in semantic:
            return _float(semantic.get(key))
        if isinstance(action, dict):
            if key in action:
                return _float(action.get(key))
            card = action.get("card") if isinstance(action.get("card"), dict) else {}
            preview = card.get("preview") if isinstance(card.get("preview"), dict) else {}
            for source in (card, preview):
                if key in source:
                    return _float(source.get(key))
        return 0.0

    @classmethod
    def _action_immediate_impact(cls, action: dict[str, Any] | None) -> float:
        roles = cls._action_roles(action)
        damage = cls._action_metric(action, "damage")
        block = cls._action_metric(action, "block")
        hits = max(cls._action_metric(action, "hits"), 1.0 if damage > 0.0 else 0.0)
        debuff_bonus = 8.0 if roles.intersection({"debuff", "weak", "vulnerable", "poison", "exhaust", "discard"}) else 0.0
        scaling_bonus = 6.0 if roles.intersection({"scaling", "power", "draw", "energy", "retain"}) else 0.0
        return float(damage + 0.75 * block + 1.5 * max(hits - 1.0, 0.0) + debuff_bonus + scaling_bonus)

    def _boss_mechanic_reward(
        self,
        before_obs: dict[str, Any] | None,
        after_obs: dict[str, Any] | None,
        action: dict[str, Any] | None,
    ) -> float:
        """Dense tactical shaping for boss-only mechanics (Kaiser / Ceremonial / Knowledge Demon)."""
        if self._current_encounter_tier() != "boss":
            return 0.0
        encounter = str(self._current_encounter_id or "").lower()
        if not ("kaiser" in encounter or "ceremonial" in encounter or "knowledge_demon" in encounter):
            return 0.0
        try:
            before_ctx = build_boss_mechanics_context(before_obs)
            after_ctx = build_boss_mechanics_context(after_obs)
        except Exception:
            return 0.0

        reward = 0.0
        before_hp, _ = self._player_hp_and_max(before_obs)
        after_hp, _ = self._player_hp_and_max(after_obs)
        hp_loss = max(before_hp - after_hp, 0.0)
        enemy_hp_delta = max(self._combat_enemy_total_hp(before_obs) - self._combat_enemy_total_hp(after_obs), 0.0)
        family = self._action_family(action)
        roles = self._action_roles(action)
        impact = self._action_immediate_impact(action)

        if "kaiser" in encounter:
            # H25: switch the Kaiser branch to the PRIMARY-threat back-attack
            # signal.  Old code used max(...) across all enemies for risk —
            # but Kaiser has two parts both flagging back_attack_active=1
            # most turns, so the metric was pinned at 1.0 and facing_change
            # 1→0 detection never fired.  primary_back_attack_active reads
            # only the highest-intent-damage enemy's status, so when the
            # player correctly faces the high-damage attacker it flips 1→0
            # even if the low-damage attacker still has multiplier=1.5.
            before_primary = self._boss_context_max(before_ctx, "primary_back_attack_active")
            after_primary = self._boss_context_max(after_ctx, "primary_back_attack_active")
            before_risk = max(
                self._boss_context_max(before_ctx, "primary_back_attack_risk"),
                before_primary,
            )
            after_risk = max(
                self._boss_context_max(after_ctx, "primary_back_attack_risk"),
                after_primary,
            )
            # §12 soften factor: don't fully penalize if the agent had no
            # mechanically valid response available this frame.
            defense_candidates = self._boss_context_max(before_ctx, "kaiser_defense_candidate_count")
            facing_change_candidates = self._boss_context_max(before_ctx, "kaiser_facing_change_candidate_count")
            pressure_candidates = self._boss_context_max(before_ctx, "kaiser_pressure_candidate_count")
            no_response_avail = (
                defense_candidates < 0.5
                and facing_change_candidates < 0.5
                and pressure_candidates < 0.5
            )
            soften = float(KAISER_NO_RESPONSE_PENALTY_SOFTEN) if no_response_avail else 1.0

            # 2026-04-28 §1C: surface the *primary* (highest-intent-damage)
            # threat damage so we can scale facing bonus and high-threat
            # back-attack penalty by intent magnitude.
            primary_intent_dmg = self._boss_context_max(before_ctx, "primary_threat_intent_damage")

            if before_risk > 0.05:
                reward -= hp_loss * float(KAISER_BACK_ATTACK_HP_LOSS_PENALTY_SCALE) * (1.0 + before_risk) * soften
                # §1C: extra penalty if the threat we ignored was a HIGH-damage
                # attacker (≥ KAISER_BACK_ATTACK_HI_THREAT_DMG). Scaled by hp_loss/max_hp
                # so it stays balanced across boss HP variance.
                _, max_hp_back = self._player_hp_and_max(before_obs)
                if (
                    primary_intent_dmg >= float(KAISER_BACK_ATTACK_HI_THREAT_DMG)
                    and hp_loss > 0.0
                    and max_hp_back > 0.0
                ):
                    reward -= float(KAISER_BACK_ATTACK_HI_THREAT_EXTRA) * (hp_loss / max_hp_back) * soften
                if family == "end_turn":
                    reward += float(KAISER_BACK_ATTACK_END_TURN_PENALTY) * min(1.0, before_risk) * soften
                if family in {"play_card", "use_potion", "potion"} and (
                    "block" in roles or "debuff" in roles or "weak" in roles or self._action_metric(action, "block") > 0.0
                ):
                    reward += float(KAISER_BACK_ATTACK_DEFENSE_BONUS) * min(1.0, before_risk)
            risk_drop = max(before_risk - after_risk, 0.0)
            if risk_drop > 0.05:
                reward += float(KAISER_BACK_ATTACK_RISK_REDUCTION_BONUS) * min(1.0, risk_drop)

            # §12 missing positive signals �?facing change + pressure kill.
            # back_attack_active flipping from 1 �?0 means the player
            # successfully re-faced (took an action that turned the boss
            # so the back enemy is no longer active threat).  We only fire
            # this on play_card / use_potion (not end_turn).
            # Use primary-threat active flag (set above) so facing_change
            # detects "now correctly facing the high-damage attacker".
            facing_changed = before_primary > 0.5 and after_primary <= 0.5
            if facing_changed and family in {"play_card", "use_potion", "potion"}:
                # 2026-04-28 §1A: scale the facing bonus by the threat magnitude
                # the agent just faced. Refacing toward a 30 dmg attacker pays
                # 1.5x; refacing toward a 10 dmg one pays ~0.5x.  This kills
                # the failure mode where model would target the cheap claw
                # for "free" facing bonus.
                ref = max(float(KAISER_FACING_INTENT_DMG_REF), 1.0)
                threat_scale = float(np.clip(
                    primary_intent_dmg / ref, 0.0,
                    float(KAISER_FACING_INTENT_DMG_SCALE_MAX),
                ))
                bonus = float(KAISER_FACING_CHANGE_BONUS_BASE) * threat_scale
                # Cap with the legacy flat bonus to avoid over-shooting prior
                # calibration on tiny-intent dummy fights.
                bonus = min(bonus, float(KAISER_FACING_CHANGE_BONUS) * float(KAISER_FACING_INTENT_DMG_SCALE_MAX))
                reward += bonus
                self._kaiser_facing_change_count += 1

            # Pressure kill: dealt damage AND back-attack risk dropped
            # meaningfully in the same step (proxy for "killed the back side
            # part / enemy without re-facing").  Differentiated from facing
            # change: facing_changed=True covers refacing; pressure_kill is
            # the alternative win condition where you just out-DPS the back.
            if (
                not facing_changed
                and family == "play_card"
                and enemy_hp_delta > 5.0
                and risk_drop > 0.20
                and before_risk > 0.20
            ):
                reward += float(KAISER_PRESSURE_KILL_BONUS)
                self._kaiser_pressure_kill_count += 1

            # Lightweight stdout breadcrumb for visibility (every 25 new events).
            # Guard on count *change* — modulo would fire on every subsequent step
            # once the sum lands on a multiple of 25 and falsely look like a hang.
            current_total = self._kaiser_facing_change_count + self._kaiser_pressure_kill_count
            last_printed = getattr(self, "_kaiser_response_last_print_total", 0)
            if current_total > 0 and current_total != last_printed and current_total % 25 == 0:
                print(
                    f"[combat_env] kaiser_response facing_change={self._kaiser_facing_change_count} "
                    f"pressure_kill={self._kaiser_pressure_kill_count}",
                    flush=True,
                )
                self._kaiser_response_last_print_total = current_total

        if "ceremonial" in encounter:
            before_one = self._boss_context_max(before_ctx, "one_card_lock")
            after_one = self._boss_context_max(after_ctx, "one_card_lock")
            before_stun = self._boss_context_max(before_ctx, "stun_window")
            after_stun = self._boss_context_max(after_ctx, "stun_window")
            before_pending = self._boss_context_max(before_ctx, "transform_pending")
            after_threshold = self._boss_context_max(after_ctx, "threshold_active")
            if before_stun <= 0.05 and after_stun > 0.05:
                reward += float(CEREMONIAL_STUN_WINDOW_ENTER_BONUS)
            if before_pending > 0.05 and after_threshold > 0.05:
                reward += float(CEREMONIAL_THRESHOLD_PROGRESS_BONUS)
            if before_stun > 0.05 and enemy_hp_delta > 0.0:
                reward += min(0.75, enemy_hp_delta * float(CEREMONIAL_STUN_DAMAGE_MULTIPLIER))
            one_card_lock = max(before_one, after_one)
            if one_card_lock > 0.05:
                if family == "end_turn":
                    reward += float(CEREMONIAL_ONE_CARD_END_TURN_PENALTY)
                elif family in {"play_card", "use_potion", "potion"}:
                    if impact >= 12.0:
                        reward += float(CEREMONIAL_ONE_CARD_HIGH_IMPACT_BONUS)
                    elif impact <= 2.0 and not roles.intersection({"draw", "energy", "scaling", "power"}):
                        reward += float(CEREMONIAL_ONE_CARD_LOW_IMPACT_PENALTY)

        if "knowledge_demon" in encounter:
            # Knowledge Demon (知识恶魔) curse-selection shaping per user
            # strategy guidance:
            #   Curse 1 (no prior 瓦解 stack)  → prefer Option B "draw -1"
            #     (status / debuff card-selection, NOT damage_per_turn).
            #   Curse 2 (some 瓦解 stack)      → prefer Option A "+7 damage,
            #     blockable" (HP-loss easier to mitigate than max-3-cards).
            #   Curse 3 (heavy 瓦解 stack)     → prefer Option A
            #     (energy-loss is crippling).
            # We infer "which curse" from the cumulative 瓦解 / disintegrate
            # damage already on the player (read via _power_amount needles).
            # We detect the curse-selection event by checking the action's
            # surface/family/text keywords for damage-per-turn clauses.
            family_lc = family or ""
            action_text = ""
            if isinstance(action, dict):
                for key in ("title", "label", "name"):
                    val = action.get(key)
                    if val:
                        action_text += " " + str(val)
                card = action.get("card") if isinstance(action.get("card"), dict) else {}
                for key in ("title", "description", "effect", "canonical_text"):
                    val = card.get(key) if isinstance(card, dict) else None
                    if val:
                        action_text += " " + str(val)
            action_text_l = action_text.lower()
            is_curse_event = any(
                kw in action_text_l
                for kw in ("disintegrate", "瓦解", "card_selection:select", "event_option", "card_reward:skip")
            )
            picks_disintegrate = any(
                kw in action_text_l
                for kw in ("disintegrate", "瓦解", "受到 6 点", "受到 7 点", "受到 8 点", "每回合收到", "每回合受到")
            )
            picks_draw_loss = any(
                kw in action_text_l
                for kw in ("少抽 1 张", "少抽一张", "draw 1 fewer", "draw -1", "fewer card")
            )
            picks_play_cap = any(
                kw in action_text_l
                for kw in ("最多打出 3 张", "最多打 3 张", "max 3 cards", "play 3 cards")
            )
            picks_energy_loss = any(
                kw in action_text_l
                for kw in ("减少 1 点能量", "失去 1 点能量", "lose 1 energy", "-1 energy", "energy -1")
            )
            # Estimate which curse number we're choosing using the
            # current Disintegrate stack (see boss_mechanics if it's there;
            # fall back to scanning player_powers text).
            disintegrate_stack = 0.0
            before_player = before_obs.get("player") if isinstance(before_obs, dict) else None
            if isinstance(before_player, dict):
                powers = before_player.get("powers") if isinstance(before_player.get("powers"), list) else []
                for power in powers:
                    if not isinstance(power, dict):
                        continue
                    text = " ".join(
                        str(power.get(k) or "") for k in ("id", "title", "description")
                    ).lower()
                    if any(kw in text for kw in ("disintegrate", "瓦解")):
                        disintegrate_stack = max(disintegrate_stack, float(power.get("amount") or power.get("display_amount") or 0))
            curse_index = (
                1 if disintegrate_stack < 5.5
                else 2 if disintegrate_stack < 12.5
                else 3
            )

            if is_curse_event:
                if curse_index == 1:
                    # Prefer Option B (draw_loss).  A is the bad pick now.
                    if picks_draw_loss:
                        reward += float(KNOWLEDGE_DEMON_GOOD_CURSE_PICK_BONUS)
                    elif picks_disintegrate:
                        reward += float(KNOWLEDGE_DEMON_BAD_CURSE_PICK_PENALTY)
                elif curse_index == 2:
                    # Prefer Option A (disintegrate +7, blockable).
                    if picks_disintegrate:
                        reward += float(KNOWLEDGE_DEMON_GOOD_CURSE_PICK_BONUS)
                    elif picks_play_cap:
                        reward += float(KNOWLEDGE_DEMON_BAD_CURSE_PICK_PENALTY)
                else:  # curse_index == 3
                    # Prefer Option A (energy_loss is crippling).
                    if picks_disintegrate:
                        reward += float(KNOWLEDGE_DEMON_GOOD_CURSE_PICK_BONUS)
                    elif picks_energy_loss:
                        reward += float(KNOWLEDGE_DEMON_BAD_CURSE_PICK_PENALTY)
            # Speed-kill incentive: every end_turn lets the boss tick another
            # round of disintegrate + advance toward the next (worse) curse.
            if family_lc == "end_turn":
                reward += float(KNOWLEDGE_DEMON_END_TURN_PENALTY)

        return float(np.clip(reward, -1.25, 1.25))

    def _boss_terminal_reward(
        self,
        before_obs: dict[str, Any] | None,
        after_obs: dict[str, Any] | None,
        terminated: bool,
        truncated: bool,
    ) -> float:
        if self._current_encounter_tier() != "boss" or not (terminated or truncated):
            return 0.0
        after_hp, after_max_hp = self._player_hp_and_max(after_obs)
        before_hp, before_max_hp = self._player_hp_and_max(before_obs)
        max_hp = max(after_max_hp, before_max_hp, 1.0)
        if terminated and (not truncated) and after_hp > 0.0:
            return float(BOSS_COMBAT_WIN_BONUS_BASE + BOSS_COMBAT_WIN_BONUS_HP_SCALE * np.clip(after_hp / max_hp, 0.0, 1.0))
        missing_ratio = 1.0 - float(np.clip(max(after_hp, 0.0) / max_hp, 0.0, 1.0))
        # H22 v3 damage-undo (percent-based): on loss, undo the per-step
        # damage shaping that paid out during this episode by subtracting
        # BOSS_COMBAT_LOSS_DAMAGE_UNDO_PERCENT_SCALE × damage_dealt_ratio.
        # UNDO_SCALE (7.0) > per-step PERCENT_SCALE (5.0) so net damage
        # contribution is mildly negative even on a "deal everything but
        # die" loss, regardless of boss size.
        end_enemy_total = float(self._combat_enemy_total_hp(after_obs))
        base_hp = float(getattr(self, "_episode_start_boss_total_hp", 0.0) or 0.0)
        if base_hp > 0.0:
            damage_dealt_ratio = max(0.0, (base_hp - end_enemy_total) / base_hp)
            damage_undo = damage_dealt_ratio * float(BOSS_COMBAT_LOSS_DAMAGE_UNDO_PERCENT_SCALE)
        else:
            damage_undo = 0.0
        return -float(
            BOSS_COMBAT_LOSS_PENALTY_BASE
            + BOSS_COMBAT_LOSS_PENALTY_MISSING_HP_SCALE * missing_ratio
            + damage_undo
        )

    # ----- R_hp_efficiency §6.2 -----
    def _hp_preserve_win_bonus(
        self,
        after_obs: dict[str, Any] | None,
        terminated: bool,
        truncated: bool,
    ) -> float:
        """Terminal bonus for winning a non-boss combat with HP left.

        Uses sqrt(hp_end / max_hp) so the marginal value of each additional
        preserved HP tapers �?the first 20% preserved is worth more than
        the last 20%.  Boss tier gets zero weight because non-A10 bosses
        restore HP post-combat.
        """
        if not (terminated and not truncated):
            return 0.0
        after_hp, after_max_hp = self._player_hp_and_max(after_obs)
        if after_hp <= 0.0:
            return 0.0
        tier = self._current_encounter_tier()
        scale = float(HP_PRESERVE_WIN_BONUS_TIER_SCALE.get(tier, 0.0))
        if scale <= 0.0:
            return 0.0
        ratio = float(np.clip(after_hp / max(after_max_hp, 1.0), 0.0, 1.0))
        return scale * float(np.sqrt(ratio))

    # ----- R_turn_efficiency §8.2 -----
    def _turn_efficiency_penalty(self, action: dict[str, Any] | None) -> float:
        """Small tier-aware per-end_turn penalty to counter defend-forever."""
        if self._action_family(action) != "end_turn":
            return 0.0
        tier = self._current_encounter_tier()
        return float(TURN_EFFICIENCY_PENALTY_PER_END_TURN_TIER.get(tier, 0.0))

    # ----- R_outcome §5 �?tier-weighted outcome scaling -----
    def _tier_outcome_reward(
        self,
        after_obs: dict[str, Any] | None,
        terminated: bool,
        truncated: bool,
    ) -> float:
        """Symmetric win/loss bonus scaled by tier.

        Runs AFTER `_boss_terminal_reward` so boss-specific terminal shaping
        already carries its own magnitude; this function only supplements
        non-boss tiers (where there is no corresponding terminal bonus).
        Net effect: normal/elite wins and losses get a fixed ±(tier_scale)
        multiplier on top of the sparse bridge-side outcome reward.
        """
        if not (terminated or truncated):
            return 0.0
        tier = self._current_encounter_tier()
        if tier == "boss":
            # Boss already gets boss-specific terminal shaping; don't double-dip.
            return 0.0
        scale = float(OUTCOME_TIER_SCALE.get(tier, 1.0))
        if scale <= 0.0:
            return 0.0
        after_hp, _ = self._player_hp_and_max(after_obs)
        win = terminated and (not truncated) and after_hp > 0.0
        return (scale if win else -scale)

    # ----- Curriculum bookkeeping (§13) -----
    def _record_terminal_outcome_for_curriculum(
        self,
        after_obs: dict[str, Any] | None,
        terminated: bool,
        truncated: bool,
    ) -> None:
        """Feed win/loss into the CurriculumTracker and emit a one-line
        phase-switch annotation when the encounter crosses a boundary."""
        if not (terminated or truncated):
            return
        encounter = str(self._current_encounter_id or "").strip()
        if not encounter:
            return
        after_hp, _ = self._player_hp_and_max(after_obs)
        win = bool(terminated and (not truncated) and after_hp > 0.0)
        self._curriculum_tracker.record(encounter, win)
        switch = self._curriculum_tracker.check_phase_switch(encounter)
        if switch is not None:
            old_phase, new_phase = switch
            wr, n = self._curriculum_tracker.win_rate(encounter)
            print(
                f"[curriculum] encounter={encounter} phase {old_phase}->{new_phase} "
                f"win_rate_128={wr:.3f} n={n} (phase=P{new_phase})",
                flush=True,
            )
        # Periodic full-state dump so encounters that haven't crossed a phase
        # boundary still surface in the operator log.  Every 200 terminal
        # events feels right at ~3.6k steps/h × ~30 steps/ep �?120 ep/h �?
        # i.e. a dump every ~1.7h, slightly more often than the hourly cron.
        self._curriculum_episode_count += 1
        if self._curriculum_episode_count % 200 == 0:
            print(
                f"[curriculum/state] dump @ episodes={self._curriculum_episode_count}\n"
                + self._curriculum_tracker.dump_all_state(),
                flush=True,
            )

    def _enemy_hp_delta_reward(self, before_obs: dict[str, Any] | None, after_obs: dict[str, Any] | None) -> float:
        before_total = self._combat_enemy_total_hp(before_obs)
        after_total = self._combat_enemy_total_hp(after_obs)
        if before_total <= 0.0 and after_total <= 0.0:
            return 0.0

        # Guard against bridge clearing enemies list at terminal step when the
        # player DIED. Both live bridge mod and sim drop `combat.enemies` to
        # an empty list the moment the combat ends regardless of outcome �?
        # if we naively credit (before_total - 0) as "damage dealt", every
        # loss emits a positive shaping reward equal to the still-alive
        # enemies' total HP × 0.01. On a 566-HP terminal clear that's +5.66,
        # which drowns the bridge's loss penalty (-3.5 live, -1.0 sim) and
        # causes every short loss to be misclassified as a win under the
        # eval's `reward_sum > 0` heuristic. Skip the delta when the
        # after-state shows empty enemies AND the player is dead.
        after_player_hp = 0.0
        if isinstance(after_obs, dict):
            player = after_obs.get("player") if isinstance(after_obs.get("player"), dict) else {}
            after_player_hp = _float((player or {}).get("hp"))
        # Both "enemies is missing key (sim: combat={in_progress:False})" and
        # "enemies is empty list (live: combat={...,enemies:[]})" are terminal
        # transitions. Catch both by checking after_total==0 (we already have
        # that via _combat_enemy_total_hp == 0 when enemies not-a-list) AND
        # player_hp<=0 (defeat).
        if (
            before_total > 0.0
            and after_total <= 0.0
            and after_player_hp <= 0.0
        ):
            return 0.0

        delta = before_total - after_total
        # Boss tier: percent-of-boss-HP based shaping (H22 v3).  Per-step reward
        # is the FRACTION of the boss's initial total HP killed this step times
        # BOSS_ENEMY_HP_DELTA_PERCENT_SCALE (5.0), so a full-boss kill across
        # the whole episode sums to +5.0 regardless of whether the boss is
        # 200 HP or 900 HP across phase changes.  Negative deltas (boss heal)
        # use the same percent-based scale.  Falls back to raw if we lost the
        # initial-HP snapshot (defensive).
        tier = self._current_encounter_tier()
        if tier == "boss":
            base_hp = float(getattr(self, "_episode_start_boss_total_hp", 0.0) or 0.0)
            if base_hp > 0.0:
                ratio = delta / base_hp
                raw = ratio * float(BOSS_ENEMY_HP_DELTA_PERCENT_SCALE)
            else:
                raw = delta * ENEMY_HP_DELTA_REWARD_SCALE
        else:
            raw = delta * ENEMY_HP_DELTA_REWARD_SCALE
        if raw > ENEMY_HP_DELTA_REWARD_MAX_ABS:
            return ENEMY_HP_DELTA_REWARD_MAX_ABS
        if raw < -ENEMY_HP_DELTA_REWARD_MAX_ABS:
            return -ENEMY_HP_DELTA_REWARD_MAX_ABS
        return raw

    def _player_hp_delta_reward(self, before_obs: dict[str, Any] | None, after_obs: dict[str, Any] | None) -> float:
        before_player = before_obs.get("player") if isinstance(before_obs, dict) else {}
        after_player = after_obs.get("player") if isinstance(after_obs, dict) else {}
        before_hp = _float((before_player or {}).get("hp"))
        after_hp = _float((after_player or {}).get("hp"))
        if before_hp <= 0.0 and after_hp <= 0.0:
            return 0.0
        delta = after_hp - before_hp
        # Tier-aware loss multiplier (combat-reward-curriculum.md §3, §6).  Boss
        # fights (non-A10) restore HP post-combat, so heavy HP-loss penalties
        # distort boss play toward "turtle forever" instead of "win efficiently
        # using mechanics".  HP *gain* keeps full weight in all tiers �?healing
        # is equally valuable regardless of what encounter triggered it.
        #
        # Curriculum layer (§13.1): the tier scale is further multiplied by a
        # progress-lerped weight from the CurriculumTracker so Phase 0 policies
        # see minimal HP pressure (they just need to win first) and Phase 3
        # policies see full pressure.
        if delta < 0.0:
            tier = self._current_encounter_tier()
            tier_scale = float(PLAYER_HP_LOSS_TIER_SCALE.get(tier, 1.0))
            progress_weight = self._curriculum_tracker.hp_weight(
                self._current_encounter_id or "", tier
            )
            multiplier = tier_scale * progress_weight
        else:
            multiplier = 1.0
        return delta * PLAYER_HP_LOSS_REWARD_SCALE * multiplier

    @staticmethod
    def _find_sentinel_enemy(obs: dict[str, Any] | None) -> dict[str, Any] | None:
        if not isinstance(obs, dict):
            return None
        combat = obs.get("combat")
        if not isinstance(combat, dict):
            return None
        enemies = combat.get("enemies")
        if not isinstance(enemies, list):
            return None
        for enemy in enemies:
            if not isinstance(enemy, dict):
                continue
            hp = _float(enemy.get("hp", enemy.get("current_hp")))
            if hp > ENEMY_HP_SENTINEL_THRESHOLD:
                return enemy
        return None

    @staticmethod
    def _player_hp_and_max(obs: dict[str, Any] | None) -> tuple[float, float]:
        if not isinstance(obs, dict):
            return 0.0, 0.0
        player = obs.get("player") or {}
        return _float(player.get("hp")), _float(player.get("max_hp"))

    @staticmethod
    def _estimate_sentinel_death_damage(enemy: dict[str, Any] | None) -> float:
        """Upper-bound estimate of the on-death damage a sentinel enemy will deal.

        Takes the max of any matching on-death-flavored buff stack count and
        the currently announced intent damage, so the overshoot calculation is
        conservative (larger penalty if either signal is high).
        """
        if not isinstance(enemy, dict):
            return 0.0
        best = 0.0
        powers = enemy.get("powers")
        if isinstance(powers, list):
            for power in powers:
                if not isinstance(power, dict):
                    continue
                title = str(power.get("title") or "").lower()
                if not any(kw in title for kw in SENTINEL_DEATH_DAMAGE_POWER_KEYWORDS):
                    continue
                amount = _float(power.get("amount"))
                if amount > best:
                    best = amount
        intent = enemy.get("intent")
        if isinstance(intent, dict):
            intent_damage = _float(intent.get("total_damage"))
            if intent_damage > best:
                best = intent_damage
        return best

    def _sentinel_terminal_reward(
        self,
        prev_obs: dict[str, Any] | None,
        after_obs: dict[str, Any] | None,
        terminated: bool,
        truncated: bool,
    ) -> float:
        if not self._sentinel_combat_active or not (terminated or truncated):
            return 0.0

        after_hp, after_max = self._player_hp_and_max(after_obs)
        ref_max = self._sentinel_combat_start_max_hp or after_max
        if ref_max <= 0.0:
            ref_max = 1.0

        victory = terminated and (not truncated) and after_hp > 0.0
        if victory:
            hp_fraction = max(0.0, min(after_hp / ref_max, 1.0))
            return SENTINEL_COMBAT_WIN_BONUS_BASE + SENTINEL_COMBAT_WIN_BONUS_SCALE * hp_fraction

        # Loss branch: scale penalty by how much the expected death damage
        # overshot the player's (block + hp) buffer right before the terminal step.
        sentinel_before = self._find_sentinel_enemy(prev_obs)
        expected_death_damage = self._estimate_sentinel_death_damage(sentinel_before)
        prev_player = prev_obs.get("player") if isinstance(prev_obs, dict) else None
        prev_block = _float((prev_player or {}).get("block"))
        prev_hp = _float((prev_player or {}).get("hp"))
        overshoot = max(0.0, expected_death_damage - (prev_block + prev_hp))
        overshoot_fraction = max(0.0, min(overshoot / ref_max, 1.0))
        return -(
            SENTINEL_COMBAT_LOSS_PENALTY_BASE
            + SENTINEL_COMBAT_LOSS_PENALTY_SCALE * overshoot_fraction
        )

    def _end_turn_waste_penalty(
        self,
        obs: dict[str, Any] | None,
        legal_actions: list[dict[str, Any]],
        chosen_action: dict[str, Any],
    ) -> float:
        if str(chosen_action.get("action_id") or "") != "end_turn":
            return 0.0

        combat = obs.get("combat") if isinstance(obs, dict) else None
        if not isinstance(combat, dict):
            return 0.0
        energy = float(combat.get("energy") or 0.0)
        if energy <= 0.0:
            return 0.0

        diagnostics = self._action_quality_diagnostics(obs, legal_actions, chosen_action)
        positive_actions = int(diagnostics.get("urgent_positive_action_count", 0.0))
        if positive_actions <= 0:
            return 0.0

        strict_context = strict_end_turn_waste_context(
            obs,
            legal_actions,
            chosen_action,
            positive_score_fn=self._action_positive_score,
            strategic_skip_fn=lambda action, current_energy, actions: self._is_strategic_skip_candidate(
                dict(action),
                energy=current_energy,
                legal_actions=[dict(item) for item in actions],
            ),
        )
        urgent_indices = {int(idx) for idx in strict_context.get("urgent_positive_indices", [])}
        has_zero_cost_positive = any(
            isinstance(action, dict)
            and idx in urgent_indices
            and str(action.get("action_id") or "") != "end_turn"
            and isinstance(action.get("card"), dict)
            and self._card_cost(action.get("card")) <= 0.0
            for idx, action in enumerate(legal_actions)
        )

        penalty = END_TURN_WASTE_BASE_PENALTY
        penalty += END_TURN_WASTE_ENERGY_PENALTY * min(energy, 3.0)
        if has_zero_cost_positive:
            penalty += END_TURN_WASTE_ZERO_COST_BONUS_PENALTY
        penalty += END_TURN_WASTE_EXTRA_ACTION_PENALTY * min(max(positive_actions - 1, 0), 2)
        # Tier multiplier (combat-reward-curriculum.md §9.1).  Base stack maxes
        # at �?.10; tier multipliers (1.5/1.5/2.5/3.0) take it to the doc's
        # �?.15/�?.25/�?.30 targets without altering the detector logic.
        tier = self._current_encounter_tier()
        tier_mult = float(WASTEFUL_END_TURN_TIER_MULTIPLIER.get(tier, 1.5))
        penalty *= tier_mult
        # Lightweight stdout breadcrumb so the operator can see waste actually
        # happening between hourly tfevents reports.  Throttled to one print
        # every WASTE_PRINT_INTERVAL events to avoid log spam.
        self._wasteful_end_turn_count += 1
        if self._wasteful_end_turn_count % 25 == 0:
            print(
                f"[combat_env] wasteful_end_turn count={self._wasteful_end_turn_count} "
                f"tier={tier} energy={energy:.0f} positive_actions={positive_actions} "
                f"penalty={penalty:.3f}",
                flush=True,
            )
        return float(penalty)
