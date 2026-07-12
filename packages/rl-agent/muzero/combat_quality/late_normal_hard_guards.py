"""Late-Act1 normal-combat hard guards for MuZero training."""

from __future__ import annotations

from typing import Any

import numpy as np

from muzero.combat_quality import boss_race_potion_traits as _boss_race_potion_traits
from muzero.combat_quality.survival_math import protection_outcome


class LateNormalHardGuardMixin:
    def _apply_late_normal_lethal_endturn_guard(self, *, action_idx: int, legal_count: int, legal_actions: list[Any], mask_np: Any, raw_obs: Any | None, encounter: str, search_stats: dict[str, Any]) -> int:
        # P1-6 (act1 recovery 2026-05-10): late-Act1 normal combat
        # EndTurn survival guard.  The latest full-run failures are not
        # bosses: they die on floor 13/14 NORMAL encounters after spending
        # potions early.  The existing survival hard guards only cover
        # elite/boss, so they do not protect the final hallway before the
        # Act1 boss.  Keep this deliberately narrow:
        #   - selected action is End Turn,
        #   - encounter is a normal hallway,
        #   - late Act1 (floor >= 11) OR critical HP,
        #   - meaningful incoming threat,
        #   - prefer confirmed lethal, otherwise affordable block/survival
        #     potion; no override if there is no concrete alternative.
        if 0 <= int(action_idx) < legal_count and isinstance(legal_actions[int(action_idx)], dict):
            selected = legal_actions[int(action_idx)]
            if self._semantic_family(selected) == "end_turn":
                hp, max_hp, hp_valid = self._player_hp_values(raw_obs if isinstance(raw_obs, dict) else None)
                encounter_tier = self._combat_encounter_tier_from_raw(raw_obs if isinstance(raw_obs, dict) else None)
                encounter_text = self._combat_encounter_text(raw_obs if isinstance(raw_obs, dict) else None, encounter)
                encounter_l = encounter_text.lower()
                hard_normal_profile = bool(self._is_hard_normal_race_encounter(encounter_l))
                is_normal_hallway = self._is_normal_or_weak_hallway_encounter(
                    encounter_tier,
                    encounter_l,
                )
                if hp_valid and is_normal_hallway:
                    floor_value = float(self._combat_floor_value(raw_obs if isinstance(raw_obs, dict) else None))
                    hp_ratio = self._player_hp_ratio_from_values(hp, max_hp)
                    late_or_critical = bool(
                        floor_value >= 11.0
                        or hp_ratio <= 0.35
                        or (hard_normal_profile and hp_ratio <= 0.50)
                    )
                    if late_or_critical:
                        # Lethal is the cleanest survival play and should be
                        # taken even if the incoming window is not classified
                        # as lethal.  This catches cases like enemy at 1 HP
                        # where End Turn wastes a guaranteed hallway clear.
                        lethal_candidates: list[tuple[float, int]] = []
                        for idx in range(legal_count):
                            if idx == int(action_idx) or mask_np[idx] <= 0:
                                continue
                            alt = legal_actions[idx]
                            if not isinstance(alt, dict):
                                continue
                            if self._semantic_family(alt) == "end_turn":
                                continue
                            if not self._is_action_confirmed_lethal(alt, raw_obs):
                                continue
                            score = max(
                                self._action_metric(alt, "damage"),
                                self._action_metric(alt, "total_damage"),
                                self._action_numeric_value(
                                    alt,
                                    ("damage", "total_damage", "attack_damage", "preview_damage", "expected_damage"),
                                ),
                                self._action_immediate_impact(alt),
                            )
                            lethal_candidates.append((float(score), int(idx)))
                        if lethal_candidates:
                            search_stats["combat_quality_late_normal_lethal_end_turn_guard_available"] = 1.0
                            lethal_candidates.sort(key=lambda item: (-item[0], item[1]))
                            override_idx = int(lethal_candidates[0][1])
                            self._dump_combat_hard_guard_record(
                                kind="late_normal_lethal_end_turn",
                                raw_obs=raw_obs,
                                legal_actions=legal_actions,
                                original_idx=int(action_idx),
                                override_idx=override_idx,
                                risk=0.0,
                                countdown=None,
                                encounter=encounter,
                                lethal_exemption=False,
                            )
                            action_idx = override_idx
                            search_stats["combat_quality_late_normal_lethal_end_turn_guard_applied"] = 1.0
                            search_stats["combat_quality_late_normal_lethal_end_turn_guard_override"] = 1.0
                            search_stats["combat_quality_wasteful_end_turn_selected"] = 0.0
                            search_stats["combat_quality_end_turn_selected"] = 0.0

                # Re-check after the lethal override: if the action is no
                # longer End Turn, the survival-card/potion guard should not
                # run in the same step.
        return int(action_idx)

    def _apply_late_normal_survival_guard(self, *, action_idx: int, legal_count: int, legal_actions: list[Any], mask_np: Any, raw_obs: Any | None, encounter: str, search_stats: dict[str, Any]) -> int:
        if 0 <= int(action_idx) < legal_count and isinstance(legal_actions[int(action_idx)], dict):
            selected = legal_actions[int(action_idx)]
            if self._semantic_family(selected) == "end_turn":
                hp, max_hp, hp_valid = self._player_hp_values(raw_obs if isinstance(raw_obs, dict) else None)
                encounter_tier = self._combat_encounter_tier_from_raw(raw_obs if isinstance(raw_obs, dict) else None)
                encounter_text = self._combat_encounter_text(raw_obs if isinstance(raw_obs, dict) else None, encounter)
                encounter_l = encounter_text.lower()
                hard_normal_profile = bool(self._is_hard_normal_race_encounter(encounter_l))
                is_normal_hallway = self._is_normal_or_weak_hallway_encounter(
                    encounter_tier,
                    encounter_l,
                )
                if hp_valid and is_normal_hallway:
                    floor_value = float(self._combat_floor_value(raw_obs if isinstance(raw_obs, dict) else None))
                    hp_ratio = self._player_hp_ratio_from_values(hp, max_hp)
                    late_or_critical = bool(floor_value >= 11.0 or hp_ratio <= 0.35 or hard_normal_profile)
                    if late_or_critical:
                        incoming, current_block, _current_hp = self._incoming_damage_pressure(raw_obs if isinstance(raw_obs, dict) else None)
                        threat_gap = max(0.0, float(incoming) - float(current_block))
                        survival_window = bool(
                            threat_gap >= max(6.0, 0.20 * max(hp, 1.0))
                            or threat_gap >= max(1.0, hp - 1.0)
                            or (hp_ratio <= 0.25 and threat_gap >= 4.0)
                            # Tonight Act1-clear oracle patch: floor 13-15 normals
                            # were killing the run before the boss.  These medium
                            # hits are not single-turn lethal, but preserving HP is
                            # required because the Act1 boss immediately follows.
                            # Apply only in late hallway windows and only if a real
                            # defensive/survival alternative exists below.
                            or (floor_value >= 12.0 and threat_gap >= 6.0)
                            or (floor_value >= 12.0 and hp_ratio <= 0.45 and threat_gap >= 4.0)
                            or (floor_value >= 13.0 and hp_ratio <= 0.55 and threat_gap >= 5.0)
                            or (floor_value >= 14.0 and hp_ratio <= 0.60 and threat_gap >= 4.0)
                            or (hard_normal_profile and threat_gap >= 6.0)
                            or (hard_normal_profile and hp_ratio <= 0.45 and threat_gap >= 4.0)
                            or (hard_normal_profile and hp_ratio <= 0.55 and threat_gap >= 5.0)
                            or (hp_ratio <= 0.30 and threat_gap >= 2.0)
                        )
                        if survival_window:
                            search_stats["combat_quality_late_normal_survival_guard_available"] = 1.0
                            # Defensive guard must never block a kill.  The
                            # lethal-end-turn guard above should already have
                            # taken such a candidate; keep this diagnostic as
                            # a backstop if ordering changes later.
                            lethal_available = False
                            for idx in range(legal_count):
                                if idx == int(action_idx) or mask_np[idx] <= 0:
                                    continue
                                alt = legal_actions[idx]
                                if isinstance(alt, dict) and self._is_action_confirmed_lethal(alt, raw_obs):
                                    lethal_available = True
                                    break
                            if lethal_available:
                                search_stats["combat_quality_late_normal_survival_guard_lethal_exemption"] = 1.0
                            else:
                                current_energy = float(self._combat_energy(None, raw_obs))
                                candidates: list[tuple[float, int]] = []
                                for idx in range(legal_count):
                                    if idx == int(action_idx) or mask_np[idx] <= 0:
                                        continue
                                    alt = legal_actions[idx]
                                    if not isinstance(alt, dict):
                                        continue
                                    family = self._semantic_family(alt)
                                    if family == "play_card":
                                        if self._is_x_cost_action(None, int(idx), alt) and current_energy <= 0.05:
                                            continue
                                        cost = self._action_cost_value(alt)
                                        if cost > current_energy + 1e-6:
                                            continue
                                        block_value = max(
                                            self._action_metric(alt, "block"),
                                            self._action_numeric_value(alt, ("block", "total_block", "preview_block")),
                                        )
                                        roles = self._action_roles(alt)
                                        source = self._action_source(alt)
                                        identity_text = " ".join(
                                            str(x or "")
                                            for x in (
                                                alt.get("action_id"),
                                                alt.get("label"),
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
                                        if block_value <= 0.0:
                                            continue
                                        hp_cost = max(
                                            self._action_metric(alt, "hp_loss"),
                                            self._action_metric(alt, "hp_cost"),
                                            self._action_numeric_value(alt, ("hp_loss", "hp_cost", "typed_hp_loss")),
                                        )
                                        if hp_cost > 0.0 and hp - hp_cost <= 1.0:
                                            continue
                                        outcome = protection_outcome(
                                            hp=float(hp),
                                            threat_gap=float(threat_gap),
                                            block=float(block_value),
                                            hp_cost=float(hp_cost),
                                        )
                                        if not outcome.survives:
                                            search_stats[
                                                "combat_quality_late_normal_survival_guard_insufficient_candidate"
                                            ] = (
                                                float(
                                                    search_stats.get(
                                                        "combat_quality_late_normal_survival_guard_insufficient_candidate",
                                                        0.0,
                                                    )
                                                    or 0.0
                                                )
                                                + 1.0
                                            )
                                            continue
                                        score = (
                                            min(block_value, threat_gap)
                                            + 0.10 * self._action_immediate_impact(alt)
                                            - 0.50 * min(max(hp_cost, 0.0), 6.0)
                                        )
                                        candidates.append((float(score), int(idx)))
                                    elif family in {"use_potion", "potion"}:
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
                                        effect_tags = {str(x).strip().lower() for x in profile.get("effect_family", []) or []}
                                        semantic_tags = {str(x).strip().lower() for x in profile.get("semantic_tags", []) or []}
                                        timing_tags = {str(x).strip().lower() for x in profile.get("timing_tags", []) or []}
                                        potion_id = str(profile.get("potion_id") or "").upper()
                                        retrieve_has_target = bool(profile.get("retrieve_has_target", True))
                                        retrieve_tool = bool(
                                            retrieve_has_target
                                            and (
                                                profile.get("retrieve_from_discard_like", False)
                                                or float(profile.get("retrieve_from_discard", 0.0) or 0.0) > 0.0
                                                or "LIQUID_MEMORIES" in potion_id
                                                or "tutor" in effect_tags
                                                or "tutor" in semantic_tags
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
                                            or "survival" in semantic_tags
                                            or "prevent_lethal_tool" in timing_tags
                                        )
                                        if not survival_tool:
                                            continue
                                        score = float(profile.get("use_quality", profile.get("urgency", 0.0)) or 0.0)
                                        score += min(0.35, float(profile.get("heal", 0.0) or 0.0) / max(max_hp, 1.0))
                                        score += min(0.35, float(profile.get("block", 0.0) or 0.0) / max(threat_gap, 1.0))
                                        if retrieve_tool:
                                            score += 0.30
                                        if bool(profile.get("prevent_lethal", False)):
                                            score += 0.50
                                        if bool(profile.get("resource_survival_tool", False)):
                                            score += 0.35
                                        candidates.append((float(score), int(idx)))
                                search_stats["combat_quality_late_normal_survival_guard_candidate_count"] = float(len(candidates))
                                if candidates:
                                    candidates.sort(key=lambda item: (-item[0], item[1]))
                                    override_idx = int(candidates[0][1])
                                    self._dump_combat_hard_guard_record(
                                        kind="late_normal_survival",
                                        raw_obs=raw_obs,
                                        legal_actions=legal_actions,
                                        original_idx=int(action_idx),
                                        override_idx=override_idx,
                                        risk=float(threat_gap),
                                        countdown=None,
                                        encounter=encounter,
                                        lethal_exemption=False,
                                    )
                                    action_idx = override_idx
                                    search_stats["combat_quality_late_normal_survival_guard_applied"] = 1.0
                                    search_stats["combat_quality_late_normal_survival_guard_override"] = 1.0
                                    search_stats["combat_quality_wasteful_end_turn_selected"] = 0.0
                                    search_stats["combat_quality_end_turn_selected"] = 0.0
                                else:
                                    search_stats["combat_quality_late_normal_survival_guard_no_alternative"] = 1.0
        return int(action_idx)

    def _apply_late_normal_race_potion_guard(self, *, action_idx: int, legal_count: int, legal_actions: list[Any], mask_np: Any, raw_obs: Any | None, encounter: str, search_stats: dict[str, Any]) -> int:
        # P1-7 (act1 recovery 2026-05-10): late-Act1 normal
        # race/setup potion guard.  The survival guard above only forces
        # block/heal/retrieve-style tools.  Current diagnostics also show
        # floor-13 normal deaths where the policy has 0 energy, meaningful
        # incoming pressure, and a Strength/Speed/direct-damage/burst-draw
        # potion legal, but chooses End Turn.  This is a narrow
        # "beyond-gradient" patch: only late normal hallway, only 0 energy,
        # only when no useful affordable card is exposed, and never over a
        # confirmed lethal candidate.
        if 0 <= int(action_idx) < legal_count and isinstance(legal_actions[int(action_idx)], dict):
            selected = legal_actions[int(action_idx)]
            if self._semantic_family(selected) == "end_turn":
                hp, max_hp, hp_valid = self._player_hp_values(raw_obs if isinstance(raw_obs, dict) else None)
                encounter_tier = self._combat_encounter_tier_from_raw(raw_obs if isinstance(raw_obs, dict) else None)
                encounter_text = self._combat_encounter_text(raw_obs if isinstance(raw_obs, dict) else None, encounter)
                encounter_l = encounter_text.lower()
                hard_normal_profile = bool(self._is_hard_normal_race_encounter(encounter_l))
                is_normal_hallway = self._is_normal_or_weak_hallway_encounter(
                    encounter_tier,
                    encounter_l,
                )
                if hp_valid and is_normal_hallway:
                    floor_value = float(self._combat_floor_value(raw_obs if isinstance(raw_obs, dict) else None))
                    hp_ratio = self._player_hp_ratio_from_values(hp, max_hp)
                    current_energy = float(self._combat_energy(None, raw_obs))
                    incoming, current_block, _current_hp = self._incoming_damage_pressure(raw_obs if isinstance(raw_obs, dict) else None)
                    threat_gap = max(0.0, float(incoming) - float(current_block))
                    late_normal_race_window = bool(
                        (floor_value >= 8.0 or hard_normal_profile)
                        and current_energy <= 0.05
                        and (
                            threat_gap >= 8.0
                            or ((floor_value >= 11.0 or hard_normal_profile) and threat_gap >= 6.0)
                            or hp_ratio <= 0.45
                            or ((floor_value >= 11.0 or hard_normal_profile) and hp_ratio <= 0.55)
                        )
                    )
                    if late_normal_race_window:
                        lethal_available = False
                        useful_non_potion_alt = False
                        for idx in range(legal_count):
                            if idx == int(action_idx) or mask_np[idx] <= 0:
                                continue
                            alt = legal_actions[idx]
                            if not isinstance(alt, dict):
                                continue
                            family = self._semantic_family(alt)
                            if family == "end_turn":
                                continue
                            if self._is_action_confirmed_lethal(alt, raw_obs):
                                lethal_available = True
                                break
                            if family != "play_card":
                                continue
                            if self._is_x_cost_action(None, int(idx), alt) and current_energy <= 0.05:
                                continue
                            cost = self._action_cost_value(alt)
                            if cost > current_energy + 1e-6:
                                continue
                            block_value = max(
                                self._action_metric(alt, "block"),
                                self._action_numeric_value(alt, ("block", "total_block", "preview_block")),
                            )
                            damage_value = max(
                                self._action_metric(alt, "damage"),
                                self._action_metric(alt, "total_damage"),
                                self._action_numeric_value(
                                    alt,
                                    ("damage", "total_damage", "attack_damage", "preview_damage", "expected_damage"),
                                ),
                            )
                            impact_value = self._action_immediate_impact(alt)
                            roles = self._action_roles(alt)
                            if (
                                block_value > 0.0
                                or damage_value > 0.0
                                or impact_value > 0.25
                                or bool(roles.intersection({"block", "defense", "defend", "attack", "damage"}))
                            ):
                                useful_non_potion_alt = True
                                break

                        if not useful_non_potion_alt:
                            candidates: list[tuple[float, int]] = []
                            saw_potion_alt = False
                            for idx in range(legal_count):
                                if idx == int(action_idx) or mask_np[idx] <= 0:
                                    continue
                                alt = legal_actions[idx]
                                if not isinstance(alt, dict):
                                    continue
                                if self._semantic_family(alt) not in {"use_potion", "potion"}:
                                    continue
                                saw_potion_alt = True
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
                                traits = _boss_race_potion_traits(alt, profile, raw_obs)
                                if not bool(traits.get("candidate", False)):
                                    continue
                                # Do not burn arbitrary damage/no-op potions on an
                                # idle hallway turn.  For low HP with no incoming,
                                # allow only setup/burst/retrieve tools; under
                                # actual incoming pressure direct damage is also a
                                # valid race tool.
                                if threat_gap <= 0.05 and not bool(traits.get("setup_like", False)):
                                    continue
                                score = float(profile.get("use_quality", profile.get("urgency", 0.0)) or 0.0)
                                if bool(traits.get("strength_like", False)):
                                    score += 0.45
                                if bool(traits.get("dex_like", False)):
                                    score += 0.35
                                if bool(traits.get("burst_draw_like", False)):
                                    score += 0.40
                                if bool(traits.get("damage_like", False)):
                                    score += 0.25 + min(0.35, float(profile.get("damage", 0.0) or 0.0) / 35.0)
                                if bool(traits.get("retrieve_tool", False)):
                                    score += 0.25
                                if threat_gap >= 6.0:
                                    score += min(0.35, threat_gap / max(hp, 1.0))
                                if hp_ratio <= 0.55:
                                    score += 0.15
                                candidates.append((float(score), int(idx)))

                            if saw_potion_alt:
                                search_stats["combat_quality_late_normal_race_potion_guard_available"] = 1.0
                                search_stats["combat_quality_late_normal_race_potion_guard_candidate_count"] = float(len(candidates))
                                if lethal_available:
                                    search_stats["combat_quality_late_normal_race_potion_guard_lethal_exemption"] = 1.0
                                elif candidates:
                                    candidates.sort(key=lambda item: (-item[0], item[1]))
                                    override_idx = int(candidates[0][1])
                                    self._dump_combat_hard_guard_record(
                                        kind="late_normal_race_potion",
                                        raw_obs=raw_obs,
                                        legal_actions=legal_actions,
                                        original_idx=int(action_idx),
                                        override_idx=override_idx,
                                        risk=float(threat_gap),
                                        countdown=None,
                                        encounter=encounter,
                                        lethal_exemption=False,
                                    )
                                    action_idx = override_idx
                                    search_stats["combat_quality_late_normal_race_potion_guard_applied"] = 1.0
                                    search_stats["combat_quality_late_normal_race_potion_guard_override"] = 1.0
                                    search_stats["combat_quality_wasteful_end_turn_selected"] = 0.0
                                    search_stats["combat_quality_end_turn_selected"] = 0.0
                                    search_stats["combat_quality_potion_selected"] = 1.0
                                else:
                                    search_stats["combat_quality_late_normal_race_potion_guard_no_alternative"] = 1.0
        return int(action_idx)
