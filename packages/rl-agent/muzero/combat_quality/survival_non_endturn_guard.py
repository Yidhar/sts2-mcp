"""Selected non-EndTurn combat survival hard guard for MuZero training."""

from __future__ import annotations

from typing import Any

import numpy as np

from muzero.combat_quality import boss_race_potion_traits as _boss_race_potion_traits
from muzero.combat_quality.survival_math import protection_outcome


class NonEndTurnSurvivalGuardMixin:
    def _apply_survival_non_endturn_guard(self, *, action_idx: int, legal_count: int, legal_actions: list[Any], mask_np: Any, raw_obs: Any | None, encounter: str, search_stats: dict[str, Any]) -> int:
        # P1-8 (act1 recovery 2026-05-10): selected non-EndTurn survival guard.
        #
        # The prior recovery guards mostly corrected explicit End Turn choices.
        # Recent traces also show high-pressure turns where the policy chooses a
        # non-lethal, low-survival action (typically a Strike/setup card) while
        # legal Defend/block/survival-potion actions are available.  In those
        # windows the model is not "空过", but the executed action still creates
        # deterministic HP loss or death.  Keep the guard narrow:
        #   - never touch card-selection/discard-potion/end-turn surfaces;
        #   - never override a confirmed lethal selected action;
        #   - never override an already protective card/potion;
        #   - under non-critical pressure, do not turn a meaningful
        #     boss/late-hallway race/progress card into passive block;
        #   - boss/elite always eligible; normal/weak only late Act1 or critical;
        #   - choose confirmed lethal first, then affordable block/heal cards,
        #     then survival potions.
        if 0 <= int(action_idx) < legal_count and isinstance(legal_actions[int(action_idx)], dict):
            selected = legal_actions[int(action_idx)]
            selected_family = self._semantic_family(selected)
            if selected_family not in {"end_turn", "card_selection", "discard_potion"}:
                hp, max_hp, hp_valid = self._player_hp_values(raw_obs if isinstance(raw_obs, dict) else None)
                if hp_valid:
                    encounter_tier = self._combat_encounter_tier_from_raw(raw_obs if isinstance(raw_obs, dict) else None)
                    encounter_text = self._combat_encounter_text(
                        raw_obs if isinstance(raw_obs, dict) else None,
                        encounter,
                    )
                    encounter_l = encounter_text.lower()
                    hard_normal_profile = bool(self._is_hard_normal_race_encounter(encounter_l))
                    is_normal_hallway = self._is_normal_or_weak_hallway_encounter(
                        encounter_tier,
                        encounter_l,
                    )
                    floor_value = float(self._combat_floor_value(raw_obs if isinstance(raw_obs, dict) else None))
                    hp_ratio = float(np.clip(hp / max(max_hp, 1.0), 0.0, 1.0))
                    encounter_eligible = bool(
                        encounter_tier in {"elite", "boss"}
                        or (is_normal_hallway and (floor_value >= 8.0 or hp_ratio <= 0.35 or hard_normal_profile))
                    )
                    if encounter_eligible:
                        incoming, current_block, _current_hp = self._incoming_damage_pressure(
                            raw_obs if isinstance(raw_obs, dict) else None
                        )
                        threat_gap = max(0.0, float(incoming) - float(current_block))
                        survival_window = bool(
                            threat_gap >= max(6.0, 0.25 * max(hp, 1.0))
                            or threat_gap >= max(1.0, hp - 1.0)
                            or (hp_ratio <= 0.25 and threat_gap >= 4.0)
                            # Mirror the EndTurn survival windows for the common
                            # failure mode where the policy does *something* (Strike
                            # or setup) but still ignores medium incoming pressure.
                            or (
                                encounter_tier == "boss"
                                and (
                                    (hp_ratio <= 0.60 and threat_gap >= 6.0)
                                    or (hp_ratio <= 0.45 and threat_gap >= 4.0)
                                    or (hp <= 35.0 and threat_gap >= 8.0)
                                )
                            )
                            or (is_normal_hallway and floor_value >= 12.0 and threat_gap >= 6.0)
                            or (is_normal_hallway and floor_value >= 12.0 and hp_ratio <= 0.45 and threat_gap >= 4.0)
                            or (is_normal_hallway and floor_value >= 13.0 and hp_ratio <= 0.55 and threat_gap >= 5.0)
                            or (is_normal_hallway and floor_value >= 14.0 and hp_ratio <= 0.60 and threat_gap >= 4.0)
                            or (is_normal_hallway and hard_normal_profile and threat_gap >= 6.0)
                            or (is_normal_hallway and hard_normal_profile and hp_ratio <= 0.45 and threat_gap >= 4.0)
                            or (is_normal_hallway and hard_normal_profile and hp_ratio <= 0.55 and threat_gap >= 5.0)
                            or (is_normal_hallway and hp_ratio <= 0.30 and threat_gap >= 2.0)
                        )
                        if survival_window:
                            if self._is_action_confirmed_lethal(selected, raw_obs):
                                search_stats["combat_quality_survival_non_endturn_guard_available"] = 1.0
                                search_stats["combat_quality_survival_non_endturn_guard_lethal_exemption"] = 1.0
                            else:
                                current_energy = float(self._combat_energy(None, raw_obs))

                                def _card_block_heal_value(action: Any) -> tuple[float, float, float]:
                                    block_value = max(
                                        self._action_metric(action, "block"),
                                        self._action_numeric_value(
                                            action,
                                            ("block", "total_block", "preview_block", "typed_block_amount"),
                                        ),
                                    )
                                    heal_value = max(
                                        self._action_metric(action, "heal"),
                                        self._action_numeric_value(
                                            action,
                                            ("heal", "healing", "hp_gain", "typed_heal_amount"),
                                        ),
                                    )
                                    hp_cost = max(
                                        self._action_metric(action, "hp_loss"),
                                        self._action_metric(action, "hp_cost"),
                                        self._action_numeric_value(
                                            action,
                                            ("hp_loss", "hp_cost", "typed_hp_loss"),
                                        ),
                                    )
                                    roles = self._action_roles(action)
                                    source = self._action_source(action)
                                    identity_text = " ".join(
                                        str(x or "")
                                        for x in (
                                            action.get("action_id") if isinstance(action, dict) else "",
                                            action.get("label") if isinstance(action, dict) else "",
                                            source.get("id") if isinstance(source, dict) else "",
                                            source.get("title") if isinstance(source, dict) else "",
                                            source.get("name") if isinstance(source, dict) else "",
                                        )
                                    ).lower()
                                    if block_value <= 0.0 and (
                                        roles.intersection({"block", "defense", "defend"})
                                        or "defend" in identity_text
                                        or "防御" in identity_text
                                    ):
                                        block_value = 5.0
                                    return float(block_value), float(heal_value), float(hp_cost)

                                def _enemy_hp_for_action(action: Any) -> float:
                                    if not isinstance(raw_obs, dict):
                                        return 0.0
                                    combat_obj = raw_obs.get("combat")
                                    if not isinstance(combat_obj, dict):
                                        return 0.0
                                    enemies = combat_obj.get("enemies")
                                    if not isinstance(enemies, list):
                                        return 0.0
                                    target_combat_id = self._action_target_combat_id(action)
                                    all_hp: list[float] = []
                                    target_hp: list[float] = []
                                    for enemy in enemies:
                                        if not isinstance(enemy, dict):
                                            continue
                                        hp_value = 0.0
                                        for key in ("current_hp", "hp", "health", "currentHealth"):
                                            try:
                                                hp_value = max(hp_value, float(enemy.get(key) or 0.0))
                                            except (TypeError, ValueError):
                                                pass
                                        if hp_value <= 0.0:
                                            continue
                                        all_hp.append(float(hp_value))
                                        if target_combat_id is not None:
                                            for key in ("combat_id", "id", "target_combat_id"):
                                                try:
                                                    if str(int(enemy.get(key))).strip() == str(int(target_combat_id)):
                                                        target_hp.append(float(hp_value))
                                                        break
                                                except (TypeError, ValueError):
                                                    continue
                                    if target_hp:
                                        return float(max(target_hp))
                                    if all_hp:
                                        return float(max(all_hp))
                                    return 0.0

                                def _scaling_enemy_profile_for_action(action: Any) -> dict[str, Any]:
                                    """Return target profile for Act1 scaling hallways.

                                    The non-EndTurn survival guard previously treated late
                                    hallway fights as pure HP-preservation problems.  Live
                                    traces show DAMP_CULTIST/RITUAL and RAVENOUS targets
                                    become worse every turn; blocking a non-critical turn
                                    while skipping a kill/high-impact attack can be the
                                    losing move.  This profile is target-aware and only
                                    activates on explicit scaling cues from bridge powers or
                                    monster ids.
                                    """
                                    result: dict[str, Any] = {
                                        "active": False,
                                        "enemy": None,
                                        "hp": 0.0,
                                        "max_hp": 0.0,
                                        "incoming": 0.0,
                                        "alive_count": 0,
                                        "scaling_score": 0.0,
                                    }
                                    if not isinstance(raw_obs, dict) or not isinstance(action, dict):
                                        return result
                                    enemies = self._combat_enemies_from_raw(raw_obs)
                                    alive_enemies: list[dict[str, Any]] = []
                                    scaling_enemies: list[dict[str, Any]] = []

                                    def _enemy_hp(enemy: dict[str, Any]) -> float:
                                        return max(
                                            self._safe_float(enemy.get("hp")),
                                            self._safe_float(enemy.get("current_hp")),
                                            self._safe_float(enemy.get("health")),
                                            self._safe_float(enemy.get("currentHealth")),
                                        )

                                    def _enemy_max_hp(enemy: dict[str, Any]) -> float:
                                        return max(
                                            self._safe_float(enemy.get("max_hp")),
                                            self._safe_float(enemy.get("maxHealth")),
                                            self._safe_float(enemy.get("max_health")),
                                            _enemy_hp(enemy),
                                        )

                                    def _enemy_incoming(enemy: dict[str, Any]) -> float:
                                        intent = enemy.get("intent") if isinstance(enemy.get("intent"), dict) else {}
                                        return max(
                                            self._safe_float(enemy.get("intent_damage")),
                                            self._safe_float(enemy.get("total_damage")),
                                            self._safe_float(intent.get("total_damage")),
                                            self._safe_float(intent.get("damage")),
                                            self._safe_float(intent.get("attack_damage")),
                                        )

                                    def _enemy_scaling_score(enemy: dict[str, Any]) -> float:
                                        text = " ".join(
                                            str(enemy.get(key) or "")
                                            for key in (
                                                "id",
                                                "name",
                                                "title",
                                                "model_id",
                                                "modelId",
                                                "monster_id",
                                                "monsterId",
                                            )
                                        ).upper()
                                        powers = enemy.get("powers")
                                        ritual = 0.0
                                        strength = 0.0
                                        ravenous = 0.0
                                        if isinstance(powers, list):
                                            for power in powers:
                                                if not isinstance(power, dict):
                                                    continue
                                                pid = str(power.get("id") or power.get("power_id") or power.get("name") or "").upper()
                                                amount = max(
                                                    self._safe_float(power.get("amount")),
                                                    self._safe_float(power.get("value")),
                                                )
                                                if "RITUAL" in pid:
                                                    ritual = max(ritual, amount if amount > 0.0 else 1.0)
                                                if "STRENGTH" in pid:
                                                    strength = max(strength, amount)
                                                if "RAVENOUS" in pid:
                                                    ravenous = max(ravenous, amount if amount > 0.0 else 1.0)
                                        score = 0.0
                                        if "DAMP_CULTIST" in text or "CULTIST" in text or ritual > 0.0:
                                            score += 4.0 + ritual + 0.35 * max(strength, 0.0)
                                        if "CORPSE_SLUG" in text or ravenous > 0.0:
                                            score += 3.0 + 0.75 * ravenous + 0.25 * max(strength, 0.0)
                                        if strength >= 10.0:
                                            score += 2.0 + 0.10 * strength
                                        return float(score)

                                    for enemy in enemies:
                                        if not isinstance(enemy, dict):
                                            continue
                                        if bool(enemy.get("is_dead") or enemy.get("dead")) or enemy.get("alive") is False:
                                            continue
                                        hp_value = _enemy_hp(enemy)
                                        if hp_value <= 0.0:
                                            continue
                                        alive_enemies.append(enemy)
                                        if _enemy_scaling_score(enemy) > 0.0:
                                            scaling_enemies.append(enemy)

                                    result["alive_count"] = len(alive_enemies)
                                    if not scaling_enemies:
                                        return result

                                    target_id = self._action_target_combat_id(action)
                                    target_name = ""
                                    target = action.get("target") if isinstance(action.get("target"), dict) else {}
                                    if isinstance(target, dict):
                                        target_name = str(target.get("name") or target.get("title") or target.get("target_name") or "").strip().lower()
                                    chosen: dict[str, Any] | None = None
                                    if target_id is not None:
                                        for enemy in scaling_enemies:
                                            for key in ("combat_id", "id", "target_combat_id"):
                                                try:
                                                    if enemy.get(key) is not None and int(enemy.get(key)) == int(target_id):
                                                        chosen = enemy
                                                        break
                                                except (TypeError, ValueError):
                                                    continue
                                            if chosen is not None:
                                                break
                                    if chosen is None and target_name:
                                        for enemy in scaling_enemies:
                                            enemy_name = str(enemy.get("name") or enemy.get("title") or "").strip().lower()
                                            if enemy_name and enemy_name == target_name:
                                                chosen = enemy
                                                break
                                    # Untargeted/AoE attacks can still be real progress; if there is
                                    # exactly one scaling enemy alive, attach the profile to it.
                                    if chosen is None and target_id is None and len(scaling_enemies) == 1:
                                        chosen = scaling_enemies[0]
                                    if chosen is None:
                                        return result

                                    result.update(
                                        {
                                            "active": True,
                                            "enemy": chosen,
                                            "hp": float(_enemy_hp(chosen)),
                                            "max_hp": float(_enemy_max_hp(chosen)),
                                            "incoming": float(_enemy_incoming(chosen)),
                                            "scaling_score": float(_enemy_scaling_score(chosen)),
                                        }
                                    )
                                    return result

                                def _scaling_enemy_attack_value(
                                    action: Any,
                                ) -> tuple[dict[str, Any], float, float, float, bool, bool, bool]:
                                    profile = _scaling_enemy_profile_for_action(action)
                                    damage_value, impact_value, hp_cost, attackish = _card_damage_progress_value(action)
                                    target_hp = float(profile.get("hp", 0.0) or 0.0)
                                    if not bool(profile.get("active", False)) or not attackish:
                                        return profile, damage_value, impact_value, hp_cost, attackish, False, False
                                    kill_like = bool(target_hp > 0.0 and damage_value >= target_hp - 1e-6)
                                    high_damage_threshold = max(6.0, min(10.0, 0.18 * target_hp if target_hp > 0.0 else 8.0))
                                    high_damage = bool(
                                        damage_value >= high_damage_threshold
                                        or (damage_value >= 5.0 and impact_value >= 10.0)
                                    )
                                    return profile, damage_value, impact_value, hp_cost, attackish, kill_like, high_damage

                                def _scaling_enemy_kill_stabilizes(profile: dict[str, Any]) -> bool:
                                    if not bool(profile.get("active", False)):
                                        return False
                                    if int(profile.get("alive_count", 0) or 0) <= 1:
                                        return True
                                    target_incoming = float(profile.get("incoming", 0.0) or 0.0)
                                    remaining_gap = max(0.0, float(threat_gap) - target_incoming)
                                    return bool(remaining_gap < max(1.0, hp - 1.0))

                                def _card_damage_progress_value(action: Any) -> tuple[float, float, float, bool]:
                                    damage_value = max(
                                        self._action_metric(action, "damage"),
                                        self._action_metric(action, "total_damage"),
                                        self._action_numeric_value(
                                            action,
                                            (
                                                "damage",
                                                "total_damage",
                                                "attack_damage",
                                                "preview_damage",
                                                "expected_damage",
                                                "typed_damage_amount",
                                                "typed_damage",
                                            ),
                                        ),
                                    )
                                    hp_cost = max(
                                        self._action_metric(action, "hp_loss"),
                                        self._action_metric(action, "hp_cost"),
                                        self._action_numeric_value(
                                            action,
                                            ("hp_loss", "hp_cost", "typed_hp_loss"),
                                        ),
                                    )
                                    impact_value = float(self._action_immediate_impact(action))
                                    roles = self._action_roles(action)
                                    source = self._action_source(action)
                                    identity_text = " ".join(
                                        str(x or "")
                                        for x in (
                                            action.get("action_id") if isinstance(action, dict) else "",
                                            action.get("label") if isinstance(action, dict) else "",
                                            source.get("id") if isinstance(source, dict) else "",
                                            source.get("title") if isinstance(source, dict) else "",
                                            source.get("name") if isinstance(source, dict) else "",
                                        )
                                    ).lower()
                                    attackish = bool(
                                        damage_value > 0.0
                                        or roles.intersection({"attack", "damage", "offense", "strike"})
                                        or any(token in identity_text for token in ("strike", "slash", "attack", "打击", "攻击"))
                                    )
                                    return float(damage_value), float(impact_value), float(hp_cost), attackish

                                def _is_meaningful_race_progress_card(action: Any, action_index: int | None = None) -> bool:
                                    if not isinstance(action, dict):
                                        return False
                                    if self._semantic_family(action) != "play_card":
                                        return False
                                    check_idx = int(action_idx if action_index is None else action_index)
                                    if self._is_x_cost_action(None, check_idx, action) and current_energy <= 0.05:
                                        return False
                                    cost = self._action_cost_value(action)
                                    if cost > current_energy + 1e-6:
                                        return False
                                    damage_value, impact_value, hp_cost, attackish = _card_damage_progress_value(action)
                                    if hp_cost > 0.0 and hp - hp_cost <= 1.0:
                                        return False
                                    # This exemption is deliberately race-oriented, not
                                    # a blanket "setup is good" rule.  Pure energy/draw/
                                    # scaling cards with no immediate damage can still be
                                    # overridden when the turn is dangerous.
                                    if not attackish:
                                        return False
                                    target_hp = _enemy_hp_for_action(action)
                                    boss_damage_threshold = max(10.0, min(16.0, 0.10 * target_hp if target_hp > 0.0 else 10.0))
                                    normal_damage_threshold = max(10.0, min(14.0, 0.18 * target_hp if target_hp > 0.0 else 10.0))
                                    if encounter_tier == "boss":
                                        return bool(
                                            damage_value >= boss_damage_threshold
                                            or (damage_value >= 6.0 and impact_value >= 14.0)
                                        )
                                    if is_normal_hallway and floor_value >= 11.0:
                                        return bool(
                                            damage_value >= normal_damage_threshold
                                            or (damage_value >= 8.0 and impact_value >= 12.0)
                                        )
                                    return False

                                critical_survival = bool(
                                    threat_gap >= max(1.0, hp - 1.0)
                                    or (hp_ratio <= 0.20 and threat_gap >= 2.0)
                                    or (hp_ratio <= 0.25 and threat_gap >= 4.0)
                                    or (hp_ratio <= 0.30 and threat_gap >= max(6.0, 0.35 * max(hp, 1.0)))
                                )

                                no_pressure_progress_lock = False
                                try:
                                    no_pressure_progress_lock = bool(
                                        float(
                                            search_stats.get(
                                                "combat_quality_no_pressure_block_guard_progress_override_lock",
                                                0.0,
                                            )
                                            or 0.0
                                        )
                                        > 0.5
                                    )
                                except (TypeError, ValueError):
                                    no_pressure_progress_lock = False
                                locked_progress_idx: int | None = None
                                if no_pressure_progress_lock:
                                    try:
                                        locked_progress_idx = int(
                                            round(
                                                float(
                                                    search_stats.get(
                                                        "combat_quality_no_pressure_block_guard_progress_override_idx",
                                                        -1.0,
                                                    )
                                                    or -1.0
                                                )
                                            )
                                        )
                                    except (TypeError, ValueError):
                                        locked_progress_idx = None
                                if no_pressure_progress_lock and not critical_survival:
                                    search_stats["combat_quality_survival_non_endturn_guard_available"] = 1.0
                                    search_stats[
                                        "combat_quality_survival_non_endturn_guard_no_pressure_lock_skip"
                                    ] = 1.0
                                    if (
                                        locked_progress_idx is not None
                                        and 0 <= int(locked_progress_idx) < legal_count
                                        and mask_np[int(locked_progress_idx)] > 0
                                    ):
                                        return int(locked_progress_idx)
                                    return int(action_idx)
                                if no_pressure_progress_lock and locked_progress_idx == int(action_idx):
                                    search_stats["combat_quality_survival_non_endturn_guard_available"] = 1.0
                                    search_stats[
                                        "combat_quality_survival_non_endturn_guard_no_pressure_lock_critical_override"
                                    ] = 1.0

                                selected_already_protective = False
                                selected_race_progress_exempt = False
                                selected_scaling_enemy_exempt = False
                                selected_safe_hallway_progress_exempt = False
                                selected_low_value_block_exempt = False
                                selected_critical_low_value_block_exempt = False
                                # Diagnostics / decision scalars for the normal-hallway
                                # low-value-block skip.  Initialise outside the narrow
                                # play-card branch so any later guard dump can include
                                # them without risking an unbound local during training.
                                post_hit_hp = float(hp) - float(threat_gap)
                                best_card_protection = 0.0
                                block_solves_materially = False
                                relaxed_post_hit_ok = False
                                critical_low_value_post_hit_ok = False
                                selected_progressish = False
                                has_scaling_enemy_candidate = False
                                if selected_family == "play_card":
                                    sel_block, sel_heal, sel_hp_cost = _card_block_heal_value(selected)
                                    sel_has_direct_protection = bool(sel_block > 0.0 or sel_heal > 0.0)
                                    sel_outcome = protection_outcome(
                                        hp=float(hp),
                                        threat_gap=float(threat_gap),
                                        block=float(sel_block),
                                        heal=float(sel_heal),
                                        hp_cost=float(sel_hp_cost),
                                    )
                                    selected_already_protective = bool(
                                        sel_has_direct_protection and sel_outcome.survives
                                        if critical_survival
                                        else (
                                            (sel_has_direct_protection and sel_outcome.survives)
                                            or sel_heal > 0.0
                                            or sel_block >= min(max(threat_gap, 1.0), 5.0)
                                        )
                                    )
                                    selected_race_progress_exempt = bool(
                                        not critical_survival and _is_meaningful_race_progress_card(selected, int(action_idx))
                                    )
                                    (
                                        sel_scaling_profile,
                                        _sel_scaling_damage,
                                        _sel_scaling_impact,
                                        sel_scaling_hp_cost,
                                        _sel_scaling_attackish,
                                        sel_scaling_kill,
                                        sel_scaling_high_damage,
                                    ) = _scaling_enemy_attack_value(selected)
                                    selected_scaling_enemy_exempt = bool(
                                        bool(sel_scaling_profile.get("active", False))
                                        and not (sel_scaling_hp_cost > 0.0 and hp - sel_scaling_hp_cost <= 1.0)
                                        and (
                                            (sel_scaling_kill and (not critical_survival or _scaling_enemy_kill_stabilizes(sel_scaling_profile)))
                                            or ((not critical_survival) and sel_scaling_high_damage)
                                        )
                                    )
                                    # Normal hallway survival guard should not turn a safe
                                    # high-HP progress card into passive block/potion just
                                    # because incoming damage crosses a broad pressure
                                    # threshold.  The guard is for correcting survival
                                    # blunders, not for overriding every attack under
                                    # non-critical hallway pressure.  This addresses live
                                    # weak/normal sandbox traces where HP stayed comfortable
                                    # after the hit, but survival_non_endturn repeatedly
                                    # rewrote targeted damage into Defend/self/potion plays.
                                    if is_normal_hallway:
                                        post_hit_hp = float(hp) - float(threat_gap)

                                        def _has_affordable_followup_protection_after_selected() -> bool:
                                            """Return true if this attack can still be followed by defense.

                                            Survival guards operate one action at a time.  In high-energy
                                            hallway turns the model may correctly open with a Strike/Bash-like
                                            progress card and still have enough energy to play Defend after the
                                            bridge re-evaluates the next decision.  The old guard treated the
                                            *current* incoming damage as if the selected card ended the turn and
                                            rewrote those progress openers into passive block.  That created the
                                            observed normal/hard-normal attrition loops.  Keep the exemption
                                            narrow: require an actually legal, affordable block/heal follow-up and
                                            require post-hit HP to stay outside the danger band.
                                            """

                                            if threat_gap <= 0.05:
                                                return False
                                            selected_cost = max(0.0, float(self._action_cost_value(selected)))
                                            remaining_energy = float(current_energy) - selected_cost
                                            if remaining_energy < 0.95:
                                                return False
                                            required_protection = min(
                                                max(float(threat_gap) * 0.20, 4.0),
                                                max(float(threat_gap), 1.0),
                                            )
                                            for follow_idx in range(legal_count):
                                                if follow_idx == int(action_idx) or mask_np[follow_idx] <= 0:
                                                    continue
                                                follow_action = legal_actions[follow_idx]
                                                if not isinstance(follow_action, dict):
                                                    continue
                                                if self._semantic_family(follow_action) != "play_card":
                                                    continue
                                                if self._is_x_cost_action(None, int(follow_idx), follow_action) and remaining_energy <= 0.05:
                                                    continue
                                                follow_cost = max(0.0, float(self._action_cost_value(follow_action)))
                                                if follow_cost > remaining_energy + 1e-6:
                                                    continue
                                                follow_block, follow_heal, follow_hp_cost = _card_block_heal_value(
                                                    follow_action
                                                )
                                                if follow_hp_cost > 0.0 and hp - follow_hp_cost <= 1.0:
                                                    continue
                                                if follow_block + follow_heal >= required_protection:
                                                    return True
                                            return False

                                        if not critical_survival:
                                            can_followup_protect = _has_affordable_followup_protection_after_selected()
                                            comfortable_after_hit = bool(
                                                threat_gap < max(1.0, hp - 1.0)
                                                and post_hit_hp >= max(18.0, 0.45 * max_hp)
                                            )
                                            sequence_safe_after_hit = bool(
                                                can_followup_protect
                                                and threat_gap < max(1.0, hp - 1.0)
                                                and post_hit_hp >= max(18.0, 0.25 * max_hp, 0.35 * hp)
                                            )
                                            safe_after_hit = bool(comfortable_after_hit or sequence_safe_after_hit)
                                            if safe_after_hit:
                                                sel_damage, sel_impact, sel_progress_hp_cost, sel_attackish = _card_damage_progress_value(
                                                    selected
                                                )
                                                selected_safe_hallway_progress_exempt = bool(
                                                    not (
                                                        sel_progress_hp_cost > 0.0
                                                        and hp - sel_progress_hp_cost <= 1.0
                                                    )
                                                    and (
                                                        sel_attackish
                                                        or sel_damage > 0.0
                                                        or (
                                                            sel_impact >= 6.0
                                                            and sel_block <= 0.0
                                                            and sel_heal <= 0.0
                                                        )
                                                    )
                                                )
                                        # Live hard-normal traces also show a different failure:
                                        # the turn is not "comfortable" by the older 45% max-HP
                                        # threshold, but a single 5-block Defend still leaves most
                                        # of the threat unsolved.  Rewriting an attack/setup opener
                                        # to that low-value pure block just stretches the fight and
                                        # increases total HP loss.  In non-critical normal hallways,
                                        # keep meaningful progress when post-hit HP remains outside
                                        # the immediate danger band and the best affordable card
                                        # block cannot cover a material share of the threat.
                                        if not selected_safe_hallway_progress_exempt:
                                            sel_damage, sel_impact, sel_progress_hp_cost, sel_attackish = _card_damage_progress_value(
                                                selected
                                            )
                                            sel_roles = self._action_roles(selected)
                                            progress_roles = {
                                                "attack",
                                                "damage",
                                                "offense",
                                                "strike",
                                                "scaling",
                                                "draw",
                                                "card_draw",
                                                "resource",
                                                "setup",
                                                "exhaust",
                                                "generate",
                                                "buff",
                                                "debuff",
                                            }
                                            selected_progressish = bool(
                                                not (sel_progress_hp_cost > 0.0 and hp - sel_progress_hp_cost <= 1.0)
                                                and (
                                                    sel_attackish
                                                    or sel_damage > 0.0
                                                    or bool(sel_roles.intersection(progress_roles))
                                                    or (sel_impact >= 6.0 and sel_block <= 0.0 and sel_heal <= 0.0)
                                                )
                                            )
                                            best_card_protection = 0.0
                                            has_scaling_enemy_candidate = False
                                            for alt_idx in range(legal_count):
                                                if alt_idx == int(action_idx) or mask_np[alt_idx] <= 0:
                                                    continue
                                                block_alt = legal_actions[alt_idx]
                                                if not isinstance(block_alt, dict):
                                                    continue
                                                if self._semantic_family(block_alt) != "play_card":
                                                    continue
                                                if self._is_x_cost_action(None, int(alt_idx), block_alt) and current_energy <= 0.05:
                                                    continue
                                                block_cost = self._action_cost_value(block_alt)
                                                if block_cost > current_energy + 1e-6:
                                                    continue
                                                block_value, heal_value, block_hp_cost = _card_block_heal_value(block_alt)
                                                if block_hp_cost > 0.0 and hp - block_hp_cost <= 1.0:
                                                    continue
                                                (
                                                    alt_scaling_profile,
                                                    _alt_scaling_damage,
                                                    _alt_scaling_impact,
                                                    _alt_scaling_hp_cost,
                                                    _alt_scaling_attackish,
                                                    alt_scaling_kill,
                                                    alt_scaling_high_damage,
                                                ) = _scaling_enemy_attack_value(block_alt)
                                                if bool(alt_scaling_profile.get("active", False)) and (
                                                    (
                                                        alt_scaling_kill
                                                        and (
                                                            not critical_survival
                                                            or _scaling_enemy_kill_stabilizes(alt_scaling_profile)
                                                        )
                                                    )
                                                    or ((not critical_survival) and alt_scaling_high_damage)
                                                ):
                                                    has_scaling_enemy_candidate = True
                                                best_card_protection = max(
                                                    best_card_protection,
                                                    float(block_value) + float(heal_value),
                                                )
                                            relaxed_post_hit_ok = bool(
                                                threat_gap < max(1.0, hp - 1.0)
                                                and post_hit_hp >= max(10.0, 0.20 * max_hp, 0.25 * hp)
                                            )
                                            block_solves_materially = bool(
                                                best_card_protection >= max(8.0, 0.45 * max(threat_gap, 1.0))
                                                or max(0.0, float(threat_gap) - float(best_card_protection))
                                                <= max(3.0, 0.20 * float(threat_gap))
                                            )
                                            selected_low_value_block_exempt = bool(
                                                (not critical_survival)
                                                and selected_progressish
                                                and relaxed_post_hit_ok
                                                and threat_gap >= 6.0
                                                and not has_scaling_enemy_candidate
                                                and not block_solves_materially
                                            )
                                            critical_low_value_post_hit_ok = bool(
                                                critical_survival
                                                and threat_gap < max(1.0, hp - 1.0)
                                                and post_hit_hp >= max(8.0, 0.12 * max_hp)
                                            )
                                            selected_critical_low_value_block_exempt = bool(
                                                selected_progressish
                                                and critical_low_value_post_hit_ok
                                                and threat_gap >= 6.0
                                                and not has_scaling_enemy_candidate
                                                and not block_solves_materially
                                                and best_card_protection <= max(5.0, 0.35 * max(threat_gap, 1.0))
                                            )
                                            search_stats[
                                                "combat_quality_survival_non_endturn_guard_best_card_protection"
                                            ] = float(best_card_protection)
                                            search_stats[
                                                "combat_quality_survival_non_endturn_guard_block_solves_materially"
                                            ] = 1.0 if block_solves_materially else 0.0
                                            search_stats[
                                                "combat_quality_survival_non_endturn_guard_relaxed_post_hit_ok"
                                            ] = 1.0 if relaxed_post_hit_ok else 0.0
                                            search_stats[
                                                "combat_quality_survival_non_endturn_guard_selected_progressish"
                                            ] = 1.0 if selected_progressish else 0.0
                                            search_stats[
                                                "combat_quality_survival_non_endturn_guard_has_scaling_enemy_candidate"
                                            ] = 1.0 if has_scaling_enemy_candidate else 0.0
                                            search_stats[
                                                "combat_quality_survival_non_endturn_guard_critical_low_value_post_hit_ok"
                                            ] = 1.0 if critical_low_value_post_hit_ok else 0.0
                                            search_stats[
                                                "combat_quality_survival_non_endturn_guard_critical_low_value_block_skip"
                                            ] = 1.0 if selected_critical_low_value_block_exempt else 0.0
                                elif selected_family in {"use_potion", "potion"}:
                                    sel_profile = self._potion_timing_profile(
                                        selected,
                                        int(action_idx),
                                        None,
                                        raw_obs,
                                        legal_actions,
                                        mask_np,
                                        current_energy,
                                    )
                                    sel_tags = {
                                        str(x).strip().lower()
                                        for key in ("effect_family", "semantic_tags", "timing_tags", "training_tags")
                                        for x in (sel_profile.get(key) or [])
                                        if str(x).strip()
                                    }
                                    sel_traits = _boss_race_potion_traits(selected, sel_profile, raw_obs)
                                    selected_already_protective = bool(
                                        bool(sel_profile.get("urgent", False))
                                        or bool(sel_profile.get("prevent_lethal", False))
                                        or bool(sel_profile.get("prevent_major_loss", False))
                                        or bool(sel_profile.get("resource_survival_tool", False))
                                        or bool(sel_profile.get("critical_hp_survival_tool", False))
                                        or float(sel_profile.get("block", 0.0) or 0.0) > 0.0
                                        or float(sel_profile.get("heal", 0.0) or 0.0) > 0.0
                                        or "survival" in sel_tags
                                        or "prevent_lethal_tool" in sel_tags
                                        or (
                                            bool(sel_traits.get("candidate", False))
                                            and not bool(sel_traits.get("invalid_context", False))
                                            and (
                                                bool(sel_traits.get("setup_like", False))
                                                or (
                                                    bool(sel_traits.get("damage_like", False))
                                                    and threat_gap > 0.05
                                                )
                                            )
                                        )
                                    )

                                if selected_scaling_enemy_exempt:
                                    search_stats["combat_quality_survival_non_endturn_guard_available"] = 1.0
                                    search_stats[
                                        "combat_quality_survival_non_endturn_guard_scaling_enemy_exemption"
                                    ] = 1.0
                                    search_stats[
                                        "combat_quality_survival_non_endturn_guard_progress_exemption"
                                    ] = 1.0
                                elif selected_race_progress_exempt:
                                    search_stats["combat_quality_survival_non_endturn_guard_available"] = 1.0
                                    search_stats[
                                        "combat_quality_survival_non_endturn_guard_progress_exemption"
                                    ] = 1.0
                                elif selected_safe_hallway_progress_exempt:
                                    search_stats["combat_quality_survival_non_endturn_guard_available"] = 1.0
                                    search_stats[
                                        "combat_quality_survival_non_endturn_guard_progress_exemption"
                                    ] = 1.0
                                elif selected_low_value_block_exempt:
                                    search_stats["combat_quality_survival_non_endturn_guard_available"] = 1.0
                                    search_stats[
                                        "combat_quality_survival_non_endturn_guard_progress_exemption"
                                    ] = 1.0
                                    search_stats[
                                        "combat_quality_survival_non_endturn_guard_low_value_block_skip"
                                    ] = 1.0
                                elif selected_critical_low_value_block_exempt:
                                    search_stats["combat_quality_survival_non_endturn_guard_available"] = 1.0
                                    search_stats[
                                        "combat_quality_survival_non_endturn_guard_progress_exemption"
                                    ] = 1.0
                                    search_stats[
                                        "combat_quality_survival_non_endturn_guard_low_value_block_skip"
                                    ] = 1.0
                                    search_stats[
                                        "combat_quality_survival_non_endturn_guard_critical_low_value_block_skip"
                                    ] = 1.0
                                elif not selected_already_protective:
                                    candidates: list[tuple[float, int]] = []
                                    for idx in range(legal_count):
                                        if idx == int(action_idx) or mask_np[idx] <= 0:
                                            continue
                                        alt = legal_actions[idx]
                                        if not isinstance(alt, dict):
                                            continue
                                        family = self._semantic_family(alt)
                                        if family in {"end_turn", "card_selection", "discard_potion"}:
                                            continue
                                        if self._is_action_confirmed_lethal(alt, raw_obs):
                                            score = max(
                                                self._action_metric(alt, "damage"),
                                                self._action_metric(alt, "total_damage"),
                                                self._action_numeric_value(
                                                    alt,
                                                    (
                                                        "damage",
                                                        "total_damage",
                                                        "attack_damage",
                                                        "preview_damage",
                                                        "expected_damage",
                                                    ),
                                                ),
                                                self._action_immediate_impact(alt),
                                            )
                                            candidates.append((100.0 + float(score), int(idx)))
                                            continue

                                        if family == "play_card":
                                            if self._is_x_cost_action(None, int(idx), alt) and current_energy <= 0.05:
                                                continue
                                            cost = self._action_cost_value(alt)
                                            if cost > current_energy + 1e-6:
                                                continue
                                            block_value, heal_value, hp_cost = _card_block_heal_value(alt)
                                            if hp_cost > 0.0 and hp - hp_cost <= 1.0:
                                                continue
                                            (
                                                scaling_profile,
                                                scaling_damage,
                                                scaling_impact,
                                                _scaling_hp_cost,
                                                _scaling_attackish,
                                                scaling_kill,
                                                scaling_high_damage,
                                            ) = _scaling_enemy_attack_value(alt)
                                            if bool(scaling_profile.get("active", False)) and (
                                                (scaling_kill and (not critical_survival or _scaling_enemy_kill_stabilizes(scaling_profile)))
                                                or ((not critical_survival) and scaling_high_damage)
                                            ):
                                                scaling_score = float(scaling_profile.get("scaling_score", 0.0) or 0.0)
                                                candidate_score = (
                                                    (105.0 if scaling_kill else 70.0)
                                                    + min(max(scaling_damage, 0.0), 35.0)
                                                    + 0.15 * min(max(scaling_impact, 0.0), 40.0)
                                                    + min(max(scaling_score, 0.0), 20.0)
                                                )
                                                candidates.append((float(candidate_score), int(idx)))
                                                search_stats[
                                                    "combat_quality_survival_non_endturn_guard_scaling_enemy_candidate"
                                                ] = 1.0
                                                continue
                                            if (not critical_survival) and _is_meaningful_race_progress_card(alt, int(idx)):
                                                damage_value, impact_value, _progress_hp_cost, _attackish = _card_damage_progress_value(alt)
                                                race_score_base = 60.0 if encounter_tier == "boss" else 52.0
                                                candidates.append(
                                                    (
                                                        float(
                                                            race_score_base
                                                            + min(max(damage_value, 0.0), 30.0)
                                                            + 0.15 * min(max(impact_value, 0.0), 40.0)
                                                        ),
                                                        int(idx),
                                                    )
                                                )
                                                continue
                                            if block_value <= 0.0 and heal_value <= 0.0:
                                                continue
                                            outcome = protection_outcome(
                                                hp=float(hp),
                                                threat_gap=float(threat_gap),
                                                block=float(block_value),
                                                heal=float(heal_value),
                                                hp_cost=float(hp_cost),
                                            )
                                            if not outcome.survives:
                                                search_stats[
                                                    "combat_quality_survival_non_endturn_guard_insufficient_candidate"
                                                ] = (
                                                    float(
                                                        search_stats.get(
                                                            "combat_quality_survival_non_endturn_guard_insufficient_candidate",
                                                            0.0,
                                                        )
                                                        or 0.0
                                                    )
                                                    + 1.0
                                                )
                                                continue
                                            score = (
                                                40.0
                                                + min(block_value, threat_gap)
                                                + min(heal_value, max_hp - hp if max_hp > 0.0 else heal_value)
                                                + 0.10 * self._action_immediate_impact(alt)
                                                - 0.50 * min(max(hp_cost, 0.0), 6.0)
                                            )
                                            candidates.append((float(score), int(idx)))
                                            continue

                                        if family in {"use_potion", "potion"}:
                                            profile = self._potion_timing_profile(
                                                alt,
                                                int(idx),
                                                None,
                                                raw_obs,
                                                legal_actions,
                                                mask_np,
                                                current_energy,
                                            )
                                            if not bool(profile.get("hp_valid", False)):
                                                continue
                                            effect_tags = {
                                                str(x).strip().lower()
                                                for key in ("effect_family", "semantic_tags", "timing_tags", "training_tags")
                                                for x in (profile.get(key) or [])
                                                if str(x).strip()
                                            }
                                            potion_id = str(profile.get("potion_id") or "").upper()
                                            retrieve_has_target = bool(profile.get("retrieve_has_target", True))
                                            retrieve_tool = bool(
                                                retrieve_has_target
                                                and (
                                                    bool(profile.get("retrieve_from_discard_like", False))
                                                    or float(profile.get("retrieve_from_discard", 0.0) or 0.0) > 0.0
                                                    or "LIQUID_MEMORIES" in potion_id
                                                    or "tutor" in effect_tags
                                                )
                                            )
                                            survival_tool = bool(
                                                bool(profile.get("urgent", False))
                                                or bool(profile.get("prevent_lethal", False))
                                                or bool(profile.get("prevent_major_loss", False))
                                                or bool(profile.get("resource_survival_tool", False))
                                                or bool(profile.get("critical_hp_survival_tool", False))
                                                or float(profile.get("heal", 0.0) or 0.0) > 0.0
                                                or float(profile.get("block", 0.0) or 0.0) > 0.0
                                                or retrieve_tool
                                                or "survival" in effect_tags
                                                or "prevent_lethal_tool" in effect_tags
                                            )
                                            if not survival_tool:
                                                continue
                                            score = 20.0 + float(profile.get("use_quality", profile.get("urgency", 0.0)) or 0.0)
                                            score += min(0.50, float(profile.get("block", 0.0) or 0.0) / max(threat_gap, 1.0))
                                            score += min(0.50, float(profile.get("heal", 0.0) or 0.0) / max(max_hp - hp, 1.0))
                                            if bool(profile.get("prevent_lethal", False)):
                                                score += 0.75
                                            if bool(profile.get("resource_survival_tool", False)):
                                                score += 0.35
                                            if retrieve_tool:
                                                score += 0.35
                                            candidates.append((float(score), int(idx)))

                                    search_stats["combat_quality_survival_non_endturn_guard_available"] = 1.0
                                    search_stats["combat_quality_survival_non_endturn_guard_candidate_count"] = float(len(candidates))
                                    if candidates:
                                        candidates.sort(key=lambda item: (-item[0], item[1]))
                                        override_idx = int(candidates[0][1])

                                        def _candidate_debug_rows() -> list[dict[str, Any]]:
                                            rows: list[dict[str, Any]] = []
                                            for cand_score, cand_idx in candidates[:5]:
                                                if not (0 <= int(cand_idx) < len(legal_actions)):
                                                    continue
                                                cand_action = legal_actions[int(cand_idx)]
                                                if not isinstance(cand_action, dict):
                                                    continue
                                                cand_card = cand_action.get("card") if isinstance(cand_action.get("card"), dict) else {}
                                                cand_potion = (
                                                    cand_action.get("potion")
                                                    if isinstance(cand_action.get("potion"), dict)
                                                    else {}
                                                )
                                                try:
                                                    cand_roles = sorted(str(x) for x in self._action_roles(cand_action))
                                                except Exception:
                                                    cand_roles = []
                                                rows.append(
                                                    {
                                                        "score": float(cand_score),
                                                        "idx": int(cand_idx),
                                                        "family": self._semantic_family(cand_action),
                                                        "kind": cand_action.get("kind"),
                                                        "title": cand_action.get("title")
                                                        or cand_card.get("title")
                                                        or cand_card.get("name")
                                                        or cand_potion.get("title")
                                                        or cand_potion.get("name")
                                                        or cand_action.get("label"),
                                                        "cost": float(self._action_cost_value(cand_action)),
                                                        "damage": float(self._action_metric(cand_action, "damage")),
                                                        "total_damage": float(self._action_metric(cand_action, "total_damage")),
                                                        "block": float(self._action_metric(cand_action, "block")),
                                                        "total_block": float(self._action_metric(cand_action, "total_block")),
                                                        "heal": float(self._action_metric(cand_action, "heal")),
                                                        "impact": float(self._action_immediate_impact(cand_action)),
                                                        "roles": cand_roles,
                                                    }
                                                )
                                            return rows

                                        self._dump_combat_hard_guard_record(
                                            kind="survival_non_endturn",
                                            raw_obs=raw_obs,
                                            legal_actions=legal_actions,
                                            original_idx=int(action_idx),
                                            override_idx=override_idx,
                                            risk=float(threat_gap),
                                            countdown=None,
                                            encounter=encounter,
                                            lethal_exemption=False,
                                            extra={
                                                "hp": float(hp),
                                                "max_hp": float(max_hp),
                                                "hp_ratio": float(hp_ratio),
                                                "incoming": float(incoming),
                                                "current_block": float(current_block),
                                                "threat_gap": float(threat_gap),
                                                "post_hit_hp": float(post_hit_hp),
                                                "floor": float(floor_value),
                                                "encounter_tier": encounter_tier,
                                                "is_normal_hallway": bool(is_normal_hallway),
                                                "hard_normal_profile": bool(hard_normal_profile),
                                                "critical_survival": bool(critical_survival),
                                                "selected_already_protective": bool(selected_already_protective),
                                                "selected_scaling_enemy_exempt": bool(selected_scaling_enemy_exempt),
                                                "selected_race_progress_exempt": bool(selected_race_progress_exempt),
                                                "selected_safe_hallway_progress_exempt": bool(
                                                    selected_safe_hallway_progress_exempt
                                                ),
                                                "selected_low_value_block_exempt": bool(selected_low_value_block_exempt),
                                                "selected_critical_low_value_block_exempt": bool(
                                                    selected_critical_low_value_block_exempt
                                                ),
                                                "selected_progressish": bool(selected_progressish),
                                                "best_card_protection": float(best_card_protection),
                                                "block_solves_materially": bool(block_solves_materially),
                                                "relaxed_post_hit_ok": bool(relaxed_post_hit_ok),
                                                "critical_low_value_post_hit_ok": bool(
                                                    critical_low_value_post_hit_ok
                                                ),
                                                "candidate_count": int(len(candidates)),
                                                "candidate_top": _candidate_debug_rows(),
                                            },
                                        )
                                        action_idx = override_idx
                                        search_stats["combat_quality_survival_non_endturn_guard_applied"] = 1.0
                                        search_stats["combat_quality_survival_non_endturn_guard_override"] = 1.0
                                        search_stats["combat_quality_hard_guard_override_any"] = 1.0
                                        search_stats["combat_quality_wasteful_end_turn_selected"] = 0.0
                                        search_stats["combat_quality_end_turn_selected"] = 0.0
                                    else:
                                        search_stats["combat_quality_survival_non_endturn_guard_no_alternative"] = 1.0
        return int(action_idx)
