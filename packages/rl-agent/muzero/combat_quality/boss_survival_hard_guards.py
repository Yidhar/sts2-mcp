"""Elite/boss combat survival hard guards for MuZero training."""

from __future__ import annotations

from typing import Any

import numpy as np

from muzero.combat_quality import (
    boss_race_potion_traits as _boss_race_potion_traits,
    boss_zero_energy_block_potion_escape as _boss_zero_energy_block_potion_escape,
    boss_zero_energy_liquid_escape as _boss_zero_energy_liquid_escape,
    is_lucky_survival_potion_for_guard as _is_lucky_survival_potion_for_guard,
    lagavulin_setup_liquid_escape as _lagavulin_setup_liquid_escape,
    potion_identity_text_for_guard as _potion_identity_text_for_guard,
)
from muzero.combat_quality.survival_math import protection_outcome


class BossSurvivalHardGuardMixin:
    def _apply_boss_race_potion_guard(self, *, action_idx: int, legal_count: int, legal_actions: list[Any], mask_np: Any, raw_obs: Any | None, encounter: str, search_stats: dict[str, Any]) -> int:
        # P1-4b (act1 recovery 2026-05-10): boss race/setup potion guard.
        # The latest Act1 run reaches the boss but repeatedly dies with
        # Strength/Glowing-Water style potions still legal on 0-energy End Turn
        # decisions.  Waiting for RL to discover this sparse long-horizon timing
        # is too slow for the recovery target, so add a narrow boss-only
        # heuristic: if End Turn is selected in a boss setup/race window and the
        # only meaningful progress is a high-confidence race/setup potion, spend
        # it.  This explicitly excludes Fortifier-at-zero-block and empty-discard
        # Liquid Memories so it does not reintroduce the earlier potion waste
        # loops.
        if 0 <= int(action_idx) < legal_count and isinstance(legal_actions[int(action_idx)], dict):
            selected = legal_actions[int(action_idx)]
            if self._semantic_family(selected) == "end_turn":
                hp, max_hp, hp_valid = self._player_hp_values(raw_obs if isinstance(raw_obs, dict) else None)
                encounter_tier = self._combat_encounter_tier_from_raw(raw_obs if isinstance(raw_obs, dict) else None)
                if hp_valid and encounter_tier == "boss":
                    incoming, current_block, _current_hp = self._incoming_damage_pressure(raw_obs if isinstance(raw_obs, dict) else None)
                    threat_gap = max(0.0, float(incoming) - float(current_block))
                    hp_ratio = float(np.clip(hp / max(max_hp, 1.0), 0.0, 1.0))
                    current_energy = float(self._combat_energy(None, raw_obs))
                    boss_setup_window = bool(
                        threat_gap <= 0.05
                        and current_energy <= 0.05
                        and hp_ratio >= 0.45
                    )
                    boss_race_window = bool(threat_gap >= 8.0 or hp_ratio <= 0.65)
                    boss_critical_survival_window = bool(
                        hp_ratio <= 0.15
                        or threat_gap >= max(1.0, hp - 1.0)
                        or (hp_ratio <= 0.35 and threat_gap >= max(6.0, 0.25 * max(hp, 1.0)))
                        or (hp_ratio <= 0.45 and threat_gap >= max(6.0, 0.50 * max(hp, 1.0)))
                    )
                    boss_survival_race_window = bool(
                        boss_critical_survival_window
                        or (current_energy <= 0.05 and hp_ratio <= 0.60 and threat_gap >= 6.0)
                    )
                    if boss_setup_window or boss_race_window or boss_survival_race_window:
                        lethal_available = False
                        useful_non_potion_alt = False
                        for idx in range(legal_count):
                            if idx == int(action_idx) or mask_np[idx] <= 0:
                                continue
                            alt = legal_actions[idx]
                            if isinstance(alt, dict) and self._is_action_confirmed_lethal(alt, raw_obs):
                                lethal_available = True
                                break
                            if isinstance(alt, dict):
                                family = self._semantic_family(alt)
                                if family not in {"end_turn", "use_potion", "potion", "discard_potion"}:
                                    useful_non_potion_alt = True
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
                            liquid_escape = _boss_zero_energy_liquid_escape(
                                alt,
                                profile,
                                raw_obs,
                                encounter_tier=encounter_tier,
                                hp=float(hp),
                                max_hp=float(max_hp),
                                hp_ratio=float(hp_ratio),
                                threat_gap=float(threat_gap),
                                current_energy=float(current_energy),
                                no_non_potion_alt=not useful_non_potion_alt,
                            )
                            lagavulin_setup_escape = _lagavulin_setup_liquid_escape(
                                alt,
                                profile,
                                raw_obs,
                                encounter_tier=encounter_tier,
                                encounter_hint=encounter,
                                threat_gap=float(threat_gap),
                                current_energy=float(current_energy),
                                no_non_potion_alt=not useful_non_potion_alt,
                            )
                            block_escape = _boss_zero_energy_block_potion_escape(
                                alt,
                                profile,
                                raw_obs,
                                encounter_tier=encounter_tier,
                                threat_gap=float(threat_gap),
                                current_energy=float(current_energy),
                                no_non_potion_alt=not useful_non_potion_alt,
                            )
                            if not bool(
                                traits.get("candidate", False)
                                or liquid_escape
                                or lagavulin_setup_escape
                                or block_escape
                            ):
                                continue
                            # On a true idle/setup boss turn, spend long-horizon
                            # setup/burst tools, not arbitrary damage/no-op
                            # potions.  Under pressure we also allow direct damage
                            # because Act1 boss deaths are currently race losses.
                            if boss_setup_window and not bool(
                                traits.get("setup_like", False)
                                or liquid_escape
                                or lagavulin_setup_escape
                                or block_escape
                            ):
                                continue
                            if (
                                bool(traits.get("buffer_like", False))
                                and float(threat_gap) <= 0.05
                                and not boss_survival_race_window
                            ):
                                # Buffer/Lucky Tonic is a survival reserve, not a
                                # generic idle setup potion.  Spend it only once
                                # there is visible pressure or a critical boss
                                # survival window; otherwise keep it for the
                                # next damaging cycle.
                                continue
                            if not (boss_setup_window or boss_race_window or boss_survival_race_window):
                                continue
                            score = float(profile.get("use_quality", profile.get("urgency", 0.0)) or 0.0)
                            if bool(traits.get("strength_like", False)):
                                score += 0.55
                            if bool(traits.get("dex_like", False)):
                                score += 0.35
                            if bool(traits.get("burst_draw_like", False)):
                                score += 0.45
                            if bool(traits.get("damage_like", False)):
                                score += 0.35 + min(0.35, float(profile.get("damage", 0.0) or 0.0) / 40.0)
                            if bool(traits.get("retrieve_tool", False)):
                                score += 0.25
                            if bool(traits.get("buffer_like", False)):
                                score += 0.65 + min(0.25, float(threat_gap) / max(float(hp), 1.0))
                            if bool(traits.get("survival_like", False)):
                                score += 0.25
                            if bool(profile.get("prevent_major_loss", False)):
                                score += 0.20
                            if liquid_escape:
                                score += 0.60 + min(0.25, float(threat_gap) / max(float(hp), 1.0))
                            if lagavulin_setup_escape:
                                score += 0.85
                                search_stats[
                                    "combat_quality_boss_race_potion_guard_lagavulin_setup_escape"
                                ] = 1.0
                            if block_escape:
                                score += 0.55 + min(
                                    0.35,
                                    float(profile.get("block", 0.0) or 0.0) / max(float(threat_gap), 1.0),
                                )
                            if boss_race_window:
                                score += 0.25
                            if boss_survival_race_window:
                                score += 0.20
                            if boss_setup_window and bool(traits.get("setup_like", False)):
                                score += 0.25
                            candidates.append((float(score), int(idx)))

                        if saw_potion_alt:
                            search_stats["combat_quality_boss_race_potion_guard_available"] = 1.0
                            # The race/setup guard runs before the dedicated
                            # survival-potion guard.  Since the recovery patch
                            # intentionally lets boss critical-survival
                            # windows use the same candidate scan, mirror the
                            # survival metrics here as well; otherwise the
                            # action is fixed but TensorBoard still reports
                            # boss_survival_potion_guard_* as missing/zero.
                            if boss_survival_race_window:
                                search_stats["combat_quality_boss_survival_potion_guard_available"] = 1.0
                            if lethal_available:
                                search_stats["combat_quality_boss_race_potion_guard_lethal_exemption"] = 1.0
                                if boss_survival_race_window:
                                    search_stats["combat_quality_boss_survival_potion_guard_lethal_exemption"] = 1.0
                            elif candidates:
                                candidates.sort(key=lambda item: (-item[0], item[1]))
                                override_idx = int(candidates[0][1])
                                self._dump_combat_hard_guard_record(
                                    kind="boss_race_potion",
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
                                search_stats["combat_quality_boss_race_potion_guard_applied"] = 1.0
                                search_stats["combat_quality_boss_race_potion_guard_override"] = 1.0
                                if boss_survival_race_window:
                                    search_stats["combat_quality_boss_survival_potion_guard_applied"] = 1.0
                                    search_stats["combat_quality_boss_survival_potion_guard_override"] = 1.0
                                search_stats["combat_quality_wasteful_end_turn_selected"] = 0.0
                                search_stats["combat_quality_end_turn_selected"] = 0.0
                                search_stats["combat_quality_potion_selected"] = 1.0
                            else:
                                search_stats["combat_quality_boss_race_potion_guard_no_alternative"] = 1.0
                                if boss_survival_race_window:
                                    search_stats["combat_quality_boss_survival_potion_guard_no_alternative"] = 1.0
        return int(action_idx)

    def _boss_survival_potion_tool_score(
        self,
        *,
        action: Any,
        action_index: int,
        raw_obs: Any | None,
        legal_actions: list[Any],
        mask_np: Any,
        current_energy: float,
        encounter_tier: str,
        hp: float,
        max_hp: float,
        hp_ratio: float,
        threat_gap: float,
        useful_non_potion_alt: bool,
    ) -> float | None:
        """Return a survival-potion score, or ``None`` when the potion is not a tool.

        Keep the survival-potion classification in one place so the boss guard
        treats the *already-selected* potion and alternative potion candidates
        identically.  This matters for Lucky Tonic/幸运药剂 because the bridge may
        expose it as a buffer-like title-only action rather than as block/heal.
        """

        profile = self._potion_timing_profile(
            action,
            int(action_index),
            None,
            raw_obs,
            legal_actions,
            mask_np,
            current_energy,
        )
        if not isinstance(profile, dict) or not bool(profile.get("hp_valid", False)):
            return None

        effect_tags = {str(x).strip().lower() for x in profile.get("effect_family", []) or []}
        semantic_tags = {str(x).strip().lower() for x in profile.get("semantic_tags", []) or []}
        timing_tags = {str(x).strip().lower() for x in profile.get("timing_tags", []) or []}
        potion_id = str(profile.get("potion_id") or "").upper()
        identity_text = _potion_identity_text_for_guard(action, profile, raw_obs)
        prevent_damage = float(profile.get("prevent_damage", 0.0) or 0.0)
        lucky_tool = _is_lucky_survival_potion_for_guard(action, profile, raw_obs)
        buffer_tool = bool(
            prevent_damage > 0.0
            or bool(profile.get("buffer_like", False))
            or bool(profile.get("prevent_damage_like", False))
            or "buffer" in effect_tags
            or "buffer" in semantic_tags
            or "prevent_damage" in effect_tags
            or "prevent_damage" in semantic_tags
            or "survival_tool" in timing_tags
            or "boss_survival_tool" in timing_tags
            or lucky_tool
        )
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
        liquid_escape = _boss_zero_energy_liquid_escape(
            action,
            profile,
            raw_obs,
            encounter_tier=encounter_tier,
            hp=float(hp),
            max_hp=float(max_hp),
            hp_ratio=float(hp_ratio),
            threat_gap=float(threat_gap),
            current_energy=float(current_energy),
            no_non_potion_alt=not useful_non_potion_alt,
        )
        survival_tool = bool(
            bool(profile.get("urgent", False))
            or bool(profile.get("prevent_lethal", False))
            or bool(profile.get("prevent_major_loss", False))
            or bool(profile.get("resource_survival_tool", False))
            or float(profile.get("heal", 0.0) or 0.0) > 0.0
            or float(profile.get("block", 0.0) or 0.0) > 0.0
            or buffer_tool
            or retrieve_tool
            or liquid_escape
            or "survival" in semantic_tags
            or "prevent_lethal_tool" in timing_tags
            or "prevent_major_loss_tool" in timing_tags
            or "survival_tool" in timing_tags
            or "boss_survival_tool" in timing_tags
        )
        if not survival_tool:
            return None

        score = float(profile.get("use_quality", profile.get("urgency", 0.0)) or 0.0)
        score += min(0.35, float(profile.get("heal", 0.0) or 0.0) / max(max_hp, 1.0))
        score += min(0.35, float(profile.get("block", 0.0) or 0.0) / max(threat_gap, 1.0))
        if retrieve_tool:
            score += 0.30
        if buffer_tool:
            score += 0.70 + min(0.25, float(threat_gap) / max(float(hp), 1.0))
        if lucky_tool:
            # Lucky/Buffer is a scarce but decisive boss survival tool; give it
            # a small tie-breaker above generic buffer tags so slot-only bridge
            # payloads do not lose to weaker utility potions in the same window.
            score += 0.20
        if liquid_escape:
            score += 0.60 + min(0.25, float(threat_gap) / max(float(hp), 1.0))
        if bool(profile.get("prevent_lethal", False)):
            score += 0.50
        if bool(profile.get("resource_survival_tool", False)):
            score += 0.35
        return float(score)

    def _boss_survival_selected_action_prevents_loss(
        self,
        *,
        selected: Any,
        selected_index: int,
        selected_family: str,
        raw_obs: Any | None,
        legal_actions: list[Any],
        mask_np: Any,
        current_energy: float,
        encounter_tier: str,
        hp: float,
        max_hp: float,
        hp_ratio: float,
        threat_gap: float,
        useful_non_potion_alt: bool,
    ) -> bool:
        """Return true when the already-selected action is an adequate survival play.

        The any-action potion takeover must not replace a winning hit or a block
        card that already keeps the player alive.  It should only replace
        non-lethal, non-defensive actions in a boss/elite emergency with a legal
        survival potion such as Lucky Tonic.
        """

        if not isinstance(selected, dict):
            return False
        if selected_family in {"use_potion", "potion"}:
            return self._boss_survival_potion_tool_score(
                action=selected,
                action_index=int(selected_index),
                raw_obs=raw_obs,
                legal_actions=legal_actions,
                mask_np=mask_np,
                current_energy=current_energy,
                encounter_tier=encounter_tier,
                hp=hp,
                max_hp=max_hp,
                hp_ratio=hp_ratio,
                threat_gap=threat_gap,
                useful_non_potion_alt=useful_non_potion_alt,
            ) is not None
        if selected_family != "play_card":
            return False

        cost = float(self._action_cost_value(selected))
        if cost > float(current_energy) + 1e-6:
            return False

        block_value = max(
            self._action_metric(selected, "block"),
            self._action_numeric_value(selected, ("block", "total_block", "preview_block")),
        )
        heal_value = max(
            self._action_metric(selected, "heal"),
            self._action_metric(selected, "hp_gain"),
            self._action_numeric_value(selected, ("heal", "hp_gain", "preview_heal")),
        )
        prevent_damage = max(
            self._action_metric(selected, "prevent_damage"),
            self._action_numeric_value(selected, ("prevent_damage", "buffer", "prevent_damage_count")),
        )
        roles = self._action_roles(selected)
        identity_text = " ".join(
            str(x or "")
            for x in (
                selected.get("action_id"),
                selected.get("label"),
                selected.get("title"),
                selected.get("name"),
            )
        ).lower()
        if block_value <= 0.0 and (
            roles.intersection({"block", "defense", "defend"})
            or "defend" in identity_text
            or "防御" in identity_text
        ):
            block_value = 5.0

        hp_cost = max(
            self._action_metric(selected, "hp_loss"),
            self._action_metric(selected, "hp_cost"),
            self._action_numeric_value(selected, ("hp_loss", "hp_cost", "typed_hp_loss")),
        )
        if hp_cost > 0.0 and float(hp) - float(hp_cost) <= 0.0:
            return False

        # A non-lethal attack/utility card that merely leaves the player alive
        # after taking damage is not a "survival play".  This distinction is
        # important for boss moderate-pressure windows: Lucky Tonic/幸运药剂 is
        # often valuable before the hit is strictly lethal, while a low-value
        # Strike does nothing to reduce that boss-race damage.  Only exempt the
        # selected card when it actually contributes block/heal/prevent-damage
        # (or was already handled above as a potion / confirmed lethal).
        if max(float(block_value), float(heal_value), float(prevent_damage)) <= 0.0:
            return False

        hp_after_cost_and_heal = max(0.0, float(hp) - max(0.0, float(hp_cost))) + max(0.0, float(heal_value))
        remaining_gap = max(0.0, float(threat_gap) - max(0.0, float(block_value)) - max(0.0, float(prevent_damage)))
        return bool(remaining_gap < hp_after_cost_and_heal)

    def _has_legal_lucky_survival_potion(
        self,
        *,
        legal_actions: list[Any],
        mask_np: Any,
        legal_count: int,
        raw_obs: Any | None,
        current_energy: float,
    ) -> bool:
        """Detect whether a legal Lucky/幸运药剂 action exists now.

        This is a cheap pre-scan used only to decide whether a *moderate* boss
        pressure window may override a non-EndTurn action.  The final candidate
        choice still goes through ``_boss_survival_potion_tool_score`` so lethal
        alternatives and already-sufficient block cards remain protected.
        """

        for idx in range(int(legal_count)):
            try:
                if idx >= len(mask_np) or float(mask_np[idx]) <= 0.0:
                    continue
            except Exception:
                continue
            action = legal_actions[idx] if idx < len(legal_actions) else None
            if not isinstance(action, dict):
                continue
            if self._semantic_family(action) not in {"use_potion", "potion"}:
                continue
            try:
                profile = self._potion_timing_profile(
                    action,
                    int(idx),
                    None,
                    raw_obs,
                    legal_actions,
                    mask_np,
                    current_energy,
                )
            except Exception:
                profile = {}
            if _is_lucky_survival_potion_for_guard(action, profile if isinstance(profile, dict) else {}, raw_obs):
                return True
        return False

    def _apply_boss_survival_potion_guard(self, *, action_idx: int, legal_count: int, legal_actions: list[Any], mask_np: Any, raw_obs: Any | None, encounter: str, search_stats: dict[str, Any]) -> int:
        # P1-4 (act1 recovery 2026-05-10): critical elite/boss survival
        # potion guard.  Originally this only rescued fatal End Turn choices.
        # Act1 boss diagnostics later showed a second miss mode: the model can
        # play a low-value non-lethal card in the same death window while Lucky
        # Tonic/幸运药剂 remains legal.  Keep the broad early-boss window for
        # End Turn, but require a stricter emergency window before overriding a
        # non-EndTurn action.
        if 0 <= int(action_idx) < legal_count and isinstance(legal_actions[int(action_idx)], dict):
            selected = legal_actions[int(action_idx)]
            selected_family = self._semantic_family(selected)
            hp, max_hp, hp_valid = self._player_hp_values(raw_obs if isinstance(raw_obs, dict) else None)
            encounter_tier = self._combat_encounter_tier_from_raw(raw_obs if isinstance(raw_obs, dict) else None)
            if hp_valid and encounter_tier in {"elite", "boss"}:
                incoming, current_block, _current_hp = self._incoming_damage_pressure(raw_obs if isinstance(raw_obs, dict) else None)
                threat_gap = max(0.0, float(incoming) - float(current_block))
                hp_ratio = float(np.clip(hp / max(max_hp, 1.0), 0.0, 1.0))
                survival_window = bool(
                    hp_ratio <= 0.15
                    or threat_gap >= max(1.0, hp - 1.0)
                    or (hp_ratio <= 0.35 and threat_gap >= max(6.0, 0.25 * max(hp, 1.0)))
                    # Act1 recovery 2026-05-10: diagnostics show boss entries
                    # around 40-45 HP dying across a few "medium" hits.  The
                    # old boss branch only treated a hit as potion-worthy when
                    # it was half the current HP, which is too late for tonight's
                    # Act1-clear oracle target.  Start considering survival
                    # potions earlier for End Turn only; non-EndTurn takeover is
                    # stricter below to avoid spending Lucky over a good card.
                    or (
                        encounter_tier == "boss"
                        and (
                            (hp_ratio <= 0.60 and threat_gap >= 6.0)
                            or (hp_ratio <= 0.45 and threat_gap >= 4.0)
                            or (hp <= 35.0 and threat_gap >= 8.0)
                            or (hp <= 40.0 and threat_gap >= 3.0)
                        )
                    )
                )
                if survival_window:
                    current_energy = float(self._combat_energy(None, raw_obs))
                    non_endturn_takeover_window = bool(
                        hp_ratio <= 0.15
                        or threat_gap >= max(1.0, hp - 1.0)
                        or (
                            encounter_tier == "boss"
                            and (
                                (hp <= 35.0 and threat_gap >= 8.0)
                                or (hp_ratio <= 0.35 and threat_gap >= max(8.0, 0.35 * max(hp, 1.0)))
                            )
                        )
                        or (
                            encounter_tier == "elite"
                            and hp_ratio <= 0.25
                            and threat_gap >= max(8.0, 0.45 * max(hp, 1.0))
                        )
                    )
                    if selected_family != "end_turn" and not non_endturn_takeover_window:
                        lucky_moderate_takeover_window = bool(
                            encounter_tier == "boss"
                            and (
                                (hp <= 40.0 and threat_gap >= 3.0)
                                or (hp_ratio <= 0.45 and threat_gap >= 3.0)
                                or (hp_ratio <= 0.60 and threat_gap >= 6.0)
                            )
                            and self._has_legal_lucky_survival_potion(
                                legal_actions=legal_actions,
                                mask_np=mask_np,
                                legal_count=legal_count,
                                raw_obs=raw_obs,
                                current_energy=current_energy,
                            )
                        )
                        if not lucky_moderate_takeover_window:
                            return int(action_idx)
                        search_stats["combat_quality_boss_survival_lucky_moderate_takeover_window"] = 1.0

                    search_stats["combat_quality_boss_survival_potion_guard_available"] = 1.0
                    if selected_family != "end_turn":
                        search_stats["combat_quality_boss_survival_potion_any_action_guard_available"] = 1.0

                    if isinstance(selected, dict) and self._is_action_confirmed_lethal(selected, raw_obs):
                        search_stats["combat_quality_boss_survival_potion_guard_lethal_exemption"] = 1.0
                        return int(action_idx)

                    lethal_alternative_idx: int | None = None
                    useful_non_potion_alt = False
                    for idx in range(legal_count):
                        if idx == int(action_idx) or mask_np[idx] <= 0:
                            continue
                        alt = legal_actions[idx]
                        if not isinstance(alt, dict):
                            continue
                        if lethal_alternative_idx is None and self._is_action_confirmed_lethal(alt, raw_obs):
                            lethal_alternative_idx = int(idx)
                        family = self._semantic_family(alt)
                        if family not in {"end_turn", "use_potion", "potion", "discard_potion"}:
                            useful_non_potion_alt = True
                    if lethal_alternative_idx is not None:
                        # A confirmed kill is the best survival tool.  The old
                        # guard only exempted End Turn when a lethal card was
                        # legal, which still allowed two bad edge cases:
                        #   1) low-value non-lethal card -> Lucky, missing kill
                        #   2) selected Lucky -> keep Lucky, missing kill
                        # Prefer the legal lethal action over burning a scarce
                        # boss/elite survival potion.  This is deliberately
                        # checked before the "selected action prevents loss"
                        # exemption so a selected Lucky/Buffer does not mask a
                        # legal kill.
                        action_idx = int(lethal_alternative_idx)
                        search_stats["combat_quality_boss_survival_potion_guard_lethal_exemption"] = 1.0
                        search_stats["combat_quality_boss_survival_potion_guard_lethal_alternative_override"] = 1.0
                        if selected_family != "end_turn":
                            search_stats["combat_quality_boss_survival_potion_any_action_guard_lethal_alternative_override"] = 1.0
                        search_stats["combat_quality_wasteful_end_turn_selected"] = 0.0
                        search_stats["combat_quality_end_turn_selected"] = 0.0
                        search_stats["combat_quality_potion_selected"] = 0.0
                        return int(action_idx)

                    if selected_family != "end_turn" and self._boss_survival_selected_action_prevents_loss(
                        selected=selected,
                        selected_index=int(action_idx),
                        selected_family=selected_family,
                        raw_obs=raw_obs,
                        legal_actions=legal_actions,
                        mask_np=mask_np,
                        current_energy=current_energy,
                        encounter_tier=encounter_tier,
                        hp=float(hp),
                        max_hp=float(max_hp),
                        hp_ratio=float(hp_ratio),
                        threat_gap=float(threat_gap),
                        useful_non_potion_alt=useful_non_potion_alt,
                    ):
                        search_stats["combat_quality_boss_survival_potion_guard_selected_survival_exemption"] = 1.0
                        return int(action_idx)

                    candidates: list[tuple[float, int]] = []
                    for idx in range(legal_count):
                        if idx == int(action_idx) or mask_np[idx] <= 0:
                            continue
                        alt = legal_actions[idx]
                        if not isinstance(alt, dict):
                            continue
                        if self._semantic_family(alt) not in {"use_potion", "potion"}:
                            continue
                        score = self._boss_survival_potion_tool_score(
                            action=alt,
                            action_index=int(idx),
                            raw_obs=raw_obs,
                            legal_actions=legal_actions,
                            mask_np=mask_np,
                            current_energy=current_energy,
                            encounter_tier=encounter_tier,
                            hp=float(hp),
                            max_hp=float(max_hp),
                            hp_ratio=float(hp_ratio),
                            threat_gap=float(threat_gap),
                            useful_non_potion_alt=useful_non_potion_alt,
                        )
                        if score is not None:
                            candidates.append((float(score), int(idx)))
                    if candidates:
                        candidates.sort(key=lambda item: (-item[0], item[1]))
                        override_idx = int(candidates[0][1])
                        self._dump_combat_hard_guard_record(
                            kind="boss_survival_potion_any_action" if selected_family != "end_turn" else "boss_survival_potion",
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
                        search_stats["combat_quality_boss_survival_potion_guard_applied"] = 1.0
                        search_stats["combat_quality_boss_survival_potion_guard_override"] = 1.0
                        if selected_family != "end_turn":
                            search_stats["combat_quality_boss_survival_potion_any_action_guard_applied"] = 1.0
                            search_stats["combat_quality_boss_survival_potion_any_action_guard_override"] = 1.0
                        search_stats["combat_quality_wasteful_end_turn_selected"] = 0.0
                        search_stats["combat_quality_end_turn_selected"] = 0.0
                        search_stats["combat_quality_potion_selected"] = 1.0
                    else:
                        search_stats["combat_quality_boss_survival_potion_guard_no_alternative"] = 1.0
        return int(action_idx)

    def _apply_boss_survival_block_guard(self, *, action_idx: int, legal_count: int, legal_actions: list[Any], mask_np: Any, raw_obs: Any | None, encounter: str, search_stats: dict[str, Any]) -> int:
        # P1-5 (act1 recovery 2026-05-10): elite/boss EndTurn survival
        # block guard.  Recent full-run deaths reached the Act1 boss but
        # ended turns into 12-21 incoming damage with 0 block despite legal
        # Defend-like cards.  Keep this narrow and tactical: only End Turn,
        # elite/boss, valid HP, meaningful incoming damage, and no confirmed
        # lethal action available.
        if 0 <= int(action_idx) < legal_count and isinstance(legal_actions[int(action_idx)], dict):
            selected = legal_actions[int(action_idx)]
            if self._semantic_family(selected) == "end_turn":
                hp, max_hp, hp_valid = self._player_hp_values(raw_obs if isinstance(raw_obs, dict) else None)
                encounter_tier = self._combat_encounter_tier_from_raw(raw_obs if isinstance(raw_obs, dict) else None)
                if hp_valid and encounter_tier in {"elite", "boss"}:
                    incoming, current_block, _current_hp = self._incoming_damage_pressure(raw_obs if isinstance(raw_obs, dict) else None)
                    threat_gap = max(0.0, float(incoming) - float(current_block))
                    hp_ratio = float(np.clip(hp / max(max_hp, 1.0), 0.0, 1.0))
                    survival_window = bool(
                        threat_gap >= max(6.0, 0.20 * max(hp, 1.0))
                        or threat_gap >= max(1.0, hp - 1.0)
                        # Same medium-threat boss window as the potion guard: if a
                        # legal block card is exposed, rewrite End Turn before the
                        # model bleeds the boss-entry HP reserve away.
                        or (
                            encounter_tier == "boss"
                            and (
                                (hp_ratio <= 0.60 and threat_gap >= 6.0)
                                or (hp_ratio <= 0.45 and threat_gap >= 4.0)
                                or (hp <= 35.0 and threat_gap >= 8.0)
                            )
                        )
                    )
                    if survival_window:
                        search_stats["combat_quality_boss_survival_block_guard_available"] = 1.0
                        lethal_available = False
                        for idx in range(legal_count):
                            if idx == int(action_idx) or mask_np[idx] <= 0:
                                continue
                            alt = legal_actions[idx]
                            if isinstance(alt, dict) and self._is_action_confirmed_lethal(alt, raw_obs):
                                lethal_available = True
                                break
                        if lethal_available:
                            search_stats["combat_quality_boss_survival_block_guard_lethal_exemption"] = 1.0
                        else:
                            current_energy = float(self._combat_energy(None, raw_obs))
                            candidates: list[tuple[float, int]] = []
                            for idx in range(legal_count):
                                if idx == int(action_idx) or mask_np[idx] <= 0:
                                    continue
                                alt = legal_actions[idx]
                                if not isinstance(alt, dict):
                                    continue
                                if self._semantic_family(alt) != "play_card":
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
                                        "combat_quality_boss_survival_block_guard_insufficient_candidate"
                                    ] = (
                                        float(
                                            search_stats.get(
                                                "combat_quality_boss_survival_block_guard_insufficient_candidate",
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
                            if candidates:
                                candidates.sort(key=lambda item: (-item[0], item[1]))
                                override_idx = int(candidates[0][1])
                                self._dump_combat_hard_guard_record(
                                    kind="boss_survival_block",
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
                                search_stats["combat_quality_boss_survival_block_guard_applied"] = 1.0
                                search_stats["combat_quality_boss_survival_block_guard_override"] = 1.0
                                search_stats["combat_quality_wasteful_end_turn_selected"] = 0.0
                                search_stats["combat_quality_end_turn_selected"] = 0.0
                            else:
                                search_stats["combat_quality_boss_survival_block_guard_no_alternative"] = 1.0
        return int(action_idx)
