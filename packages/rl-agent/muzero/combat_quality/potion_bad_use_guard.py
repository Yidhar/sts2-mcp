"""Potion bad-use hard guard for MuZero training."""

from __future__ import annotations

from typing import Any

import numpy as np

from muzero.combat_quality import (
    boss_race_potion_traits as _boss_race_potion_traits,
    boss_zero_energy_liquid_escape as _boss_zero_energy_liquid_escape,
    lagavulin_setup_liquid_escape as _lagavulin_setup_liquid_escape,
)


class PotionBadUseGuardMixin:
    def _apply_potion_bad_use_guard(self, *, action_idx: int, legal_count: int, legal_actions: list[Any], mask_np: Any, raw_obs: Any | None, encounter: str, search_stats: dict[str, Any]) -> int:
        # P1-1 (recovery 2026-05-07): Potion bad-use hard guard.
        # Re-classify the post-search selected potion *inside* the guard.
        # Do not rely on ``combat_quality_potion_*_selected`` in
        # ``search_stats`` here: those selected-side diagnostics are computed
        # later in the training loop, after this hard-override hook.  The old
        # implementation therefore silently no-oped on exactly the bad-use
        # cases it was meant to block (low-urgency potion + only End Turn as
        # alternative).  Keep the taxonomy identical to selected diagnostics
        # by calling ``_potion_timing_profile`` directly.
        if 0 <= int(action_idx) < legal_count and isinstance(legal_actions[int(action_idx)], dict):
            selected = legal_actions[int(action_idx)]
            if self._semantic_family(selected) in {"use_potion", "potion"}:
                current_energy = float(self._combat_energy(None, raw_obs))
                potion_profile = self._potion_timing_profile(
                    selected,
                    int(action_idx),
                    None,
                    raw_obs,
                    legal_actions,
                    mask_np,
                    current_energy,
                )
                if not bool(potion_profile.get("hp_valid", False)):
                    # Missing HP/maxHP is an observability failure.  Fail open:
                    # record it, but do not convert a potion into End Turn or
                    # another action based on a fabricated hp_ratio.
                    search_stats["combat_quality_potion_bad_guard_invalid_obs"] = 1.0
                else:
                    potion_urgent = bool(
                        potion_profile.get("urgent", False)
                        or potion_profile.get("lethal", False)
                        or potion_profile.get("prevent_lethal", False)
                        or potion_profile.get("mechanism_answer", False)
                    )
                    bad_potion = (
                        not potion_urgent
                        and (
                            bool(potion_profile.get("low_urgency", False))
                            or bool(potion_profile.get("save_recommended", False))
                            or bool(potion_profile.get("no_followup", False))
                            or bool(potion_profile.get("block_waste", False))
                            or bool(potion_profile.get("overkill", False))
                        )
                    )
                    if bad_potion:
                        encounter_tier = self._combat_encounter_tier_from_raw(raw_obs if isinstance(raw_obs, dict) else None)
                        hp_ratio = float(potion_profile.get("hp_ratio", 0.0) or 0.0)
                        potion_id = str(potion_profile.get("potion_id") or "").upper()
                        profile_tags = {
                            str(x).strip().lower()
                            for x in (
                                list(potion_profile.get("effect_family", []) or [])
                                + list(potion_profile.get("semantic_tags", []) or [])
                                + list(potion_profile.get("timing_tags", []) or [])
                                + list(potion_profile.get("training_tags", []) or [])
                            )
                            if str(x).strip()
                        }
                        retrieve_tool = bool(
                            bool(potion_profile.get("retrieve_has_target", True))
                            and (
                                bool(potion_profile.get("retrieve_from_discard_like", False))
                                or float(potion_profile.get("retrieve_from_discard", 0.0) or 0.0) > 0.0
                                or "LIQUID_MEMORIES" in potion_id
                                or ("discard_pile" in profile_tags and "tutor" in profile_tags)
                            )
                        )
                        survival_skip_hp = 0.50 if (
                            encounter_tier == "boss"
                            and (
                                retrieve_tool
                                or bool(potion_profile.get("resource_survival_tool", False))
                            )
                        ) else 0.35
                        critical_hp_survival_skip = bool(
                            encounter_tier in {"elite", "boss"}
                            and hp_ratio <= survival_skip_hp
                            and (
                                float(potion_profile.get("heal", 0.0) or 0.0) > 0.0
                                or float(potion_profile.get("block", 0.0) or 0.0) > 0.0
                                or retrieve_tool
                                or bool(potion_profile.get("resource_survival_tool", False))
                                or bool(potion_profile.get("critical_hp_survival_tool", False))
                                or "survival" in profile_tags
                                or "prevent_lethal_tool" in profile_tags
                            )
                        )
                        threat_gap_for_skip = float(potion_profile.get("threat_gap", 0.0) or 0.0)
                        hp_for_skip = float(potion_profile.get("hp", 0.0) or 0.0)
                        max_hp_for_skip = float(potion_profile.get("max_hp", 0.0) or 0.0)
                        if hp_for_skip <= 0.0 and max_hp_for_skip > 0.0:
                            hp_for_skip = hp_ratio * max_hp_for_skip
                        real_current_survival_need = bool(
                            bool(potion_profile.get("prevent_lethal", False))
                            or bool(potion_profile.get("prevent_major_loss", False))
                            or threat_gap_for_skip >= max(6.0, 0.25 * max(hp_for_skip, 1.0))
                        )
                        idle_waste_potion = bool(
                            threat_gap_for_skip <= 0.05
                            and (
                                bool(potion_profile.get("low_urgency", False))
                                or bool(potion_profile.get("save_recommended", False))
                                or bool(potion_profile.get("no_followup", False))
                                or bool(potion_profile.get("block_waste", False))
                                or bool(potion_profile.get("overkill", False))
                            )
                        )
                        if critical_hp_survival_skip and real_current_survival_need and not idle_waste_potion:
                            # Fail open: at critical elite/boss HP, a survival
                            # potion being slightly mis-scored is safer than
                            # converting it to End Turn.  This specifically
                            # covers Liquid Memories, whose free-card follow-up
                            # only becomes visible after using the potion.
                            bad_potion = False
                            search_stats["combat_quality_potion_bad_guard_critical_hp_survival_skip"] = 1.0
                        elif critical_hp_survival_skip and idle_waste_potion:
                            search_stats["combat_quality_potion_bad_guard_critical_hp_idle_waste_not_skipped"] = 1.0
                        # Act1 recovery 2026-05-10: if the immediately
                        # preceding boss-race guard intentionally rewrote a
                        # 0-energy End Turn into a high-confidence setup/race
                        # potion (Strength/Flex/Glowing Water/direct-damage,
                        # or Liquid Memories with an actual discard target),
                        # do not let the generic "bad potion" guard undo that
                        # rewrite just because the follow-up cards are only
                        # exposed after the potion is consumed.  Keep this
                        # boss-only and 0-energy-only so the older rule still
                        # blocks low-value Strength when a real card
                        # alternative exists.
                        if bad_potion and encounter_tier == "boss" and current_energy <= 0.05:
                            boss_race_traits = _boss_race_potion_traits(selected, potion_profile, raw_obs)
                            boss_race_setup_potion = bool(
                                boss_race_traits.get("candidate", False)
                                and not boss_race_traits.get("invalid_context", False)
                                and (
                                    boss_race_traits.get("setup_like", False)
                                    or boss_race_traits.get("damage_like", False)
                                )
                            )
                            if boss_race_setup_potion:
                                bad_potion = False
                                search_stats["combat_quality_potion_bad_guard_boss_race_skip"] = 1.0
                        # Same fail-open principle for the late-Act1 normal
                        # race/setup guard above.  Without this, the guard can
                        # rewrite 0-energy End Turn into a Strength/Speed/damage
                        # potion and the generic bad-use rule immediately turns
                        # it back into End Turn because the potion's follow-up
                        # value is only visible after consumption.
                        if bad_potion and current_energy <= 0.05:
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
                            useful_non_potion_alt = False
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
                            late_normal_race_skip_window = bool(
                                is_normal_hallway
                                and (floor_value >= 8.0 or hard_normal_profile)
                                and not useful_non_potion_alt
                                and (
                                    threat_gap_for_skip >= 8.0
                                    or ((floor_value >= 11.0 or hard_normal_profile) and threat_gap_for_skip >= 6.0)
                                    or hp_ratio <= 0.45
                                    or ((floor_value >= 11.0 or hard_normal_profile) and hp_ratio <= 0.55)
                                )
                            )
                            if late_normal_race_skip_window:
                                late_normal_traits = _boss_race_potion_traits(selected, potion_profile, raw_obs)
                                late_normal_setup_potion = bool(
                                    late_normal_traits.get("candidate", False)
                                    and not late_normal_traits.get("invalid_context", False)
                                    and (
                                        late_normal_traits.get("setup_like", False)
                                        or (
                                            late_normal_traits.get("damage_like", False)
                                            and threat_gap_for_skip > 0.05
                                        )
                                    )
                                )
                                if late_normal_setup_potion:
                                    bad_potion = False
                                    search_stats["combat_quality_potion_bad_guard_late_normal_race_skip"] = 1.0
                    if bad_potion:
                        search_stats["combat_quality_potion_bad_guard_available"] = 1.0
                        lethal = bool(self._is_action_confirmed_lethal(selected, raw_obs))
                        if lethal:
                            search_stats["combat_quality_potion_bad_guard_lethal_exemption"] = 1.0
                        else:
                            non_potion_alt: list[int] = []
                            end_turn_alt: int = -1
                            for idx in range(legal_count):
                                if idx == int(action_idx) or mask_np[idx] <= 0:
                                    continue
                                alt = legal_actions[idx]
                                if not isinstance(alt, dict):
                                    continue
                                family = self._semantic_family(alt)
                                if family == "end_turn":
                                    if end_turn_alt < 0:
                                        end_turn_alt = int(idx)
                                    continue
                                if family in {"use_potion", "potion"}:
                                    continue
                                if family == "play_card":
                                    try:
                                        alt_diag = self._x_cost_diagnostic(alt, current_energy)
                                    except Exception:
                                        alt_diag = {}
                                    if float(alt_diag.get("x_cost_bad", 0.0) or 0.0) > 0.5:
                                        search_stats[
                                            "combat_quality_potion_bad_guard_skipped_bad_x_cost_alt"
                                        ] = 1.0
                                        continue
                                non_potion_alt.append(idx)
                            threat_gap = float(potion_profile.get("threat_gap", 0.0) or 0.0)
                            hp_ratio = float(potion_profile.get("hp_ratio", 0.0) or 0.0)
                            hp = float(potion_profile.get("hp", 0.0) or 0.0)
                            max_hp = float(potion_profile.get("max_hp", 0.0) or 0.0)
                            if hp <= 0.0 and max_hp > 0.0:
                                hp = hp_ratio * max_hp
                            safe_idle_turn = bool(threat_gap <= 0.05)
                            amplify_block_noop = bool(potion_profile.get("amplify_block_noop", False))
                            bad_no_followup_resource = bool(
                                bool(potion_profile.get("no_followup", False))
                                and bool(potion_profile.get("resource_like", False))
                            )
                            bad_no_followup_dependent = bool(
                                bool(potion_profile.get("no_followup", False))
                                and bool(potion_profile.get("requires_followup", False))
                            )
                            survival_margin = float(hp - threat_gap)
                            safe_to_end = bool(
                                hp_ratio >= 0.45
                                and survival_margin >= max(6.0, 0.15 * max(max_hp, 1.0))
                            )
                            moderate_safe_to_end = bool(
                                hp_ratio >= 0.30
                                and survival_margin >= max(10.0, 0.20 * max(max_hp, 1.0))
                            )
                            clearly_waste_idle = bool(
                                safe_idle_turn
                                and (
                                    bool(potion_profile.get("low_urgency", False))
                                    or bool(potion_profile.get("save_recommended", False))
                                    or bool(potion_profile.get("no_followup", False))
                                    or bool(potion_profile.get("block_waste", False))
                                    or bool(potion_profile.get("overkill", False))
                                )
                            )

                            def _non_potion_fallback_index() -> tuple[int | None, str]:
                                """Rank non-potion fallbacks instead of taking the first legal action.

                                The previous fallback used ``non_potion_alt[0]``.  Live low-memory
                                sandbox diagnostics showed that a low-urgency potion on an idle turn
                                could therefore be rewritten to the first legal card, often a pure
                                Defend at zero incoming damage, creating the bad-pure-block target
                                we were trying to remove.  Prefer actual progress/setup cards, use
                                meaningful block only under real pressure, and avoid idle pure block.
                                """

                                progress_roles = {
                                    "attack",
                                    "damage",
                                    "offense",
                                    "strike",
                                    "draw",
                                    "card_draw",
                                    "scaling",
                                    "resource",
                                    "setup",
                                    "exhaust",
                                    "generate",
                                    "buff",
                                    "debuff",
                                    "vulnerable",
                                    "weak",
                                }

                                def _identity_text(action: Any) -> str:
                                    source = self._action_source(action)
                                    parts = [
                                        action.get("action_id") if isinstance(action, dict) else "",
                                        action.get("label") if isinstance(action, dict) else "",
                                    ]
                                    if isinstance(source, dict):
                                        parts.extend(
                                            [
                                                source.get("id"),
                                                source.get("title"),
                                                source.get("name"),
                                                source.get("type"),
                                            ]
                                        )
                                    return " ".join(str(x or "") for x in parts).lower()

                                def _card_block_value(action: Any) -> float:
                                    block_value = max(
                                        self._action_metric(action, "block"),
                                        self._action_numeric_value(
                                            action,
                                            ("block", "total_block", "preview_block", "typed_block_amount"),
                                        ),
                                    )
                                    text = _identity_text(action)
                                    roles = self._action_roles(action)
                                    if block_value <= 0.0 and (
                                        roles.intersection({"block", "defense", "defend"})
                                        or "defend" in text
                                        or "防御" in text
                                    ):
                                        block_value = 5.0
                                    return float(block_value)

                                scored: list[tuple[float, int, str]] = []
                                for alt_idx in non_potion_alt:
                                    if not (0 <= int(alt_idx) < legal_count):
                                        continue
                                    alt = legal_actions[int(alt_idx)]
                                    if not isinstance(alt, dict):
                                        continue
                                    family = self._semantic_family(alt)
                                    if self._is_action_confirmed_lethal(alt, raw_obs):
                                        scored.append((1000.0, int(alt_idx), "progress"))
                                        continue
                                    if family != "play_card":
                                        scored.append((5.0, int(alt_idx), "other"))
                                        continue
                                    if self._is_x_cost_action(None, int(alt_idx), alt) and current_energy <= 0.05:
                                        continue
                                    cost = self._action_cost_value(alt)
                                    if cost > current_energy + 1e-6:
                                        continue
                                    hp_cost = max(
                                        self._action_metric(alt, "hp_loss"),
                                        self._action_metric(alt, "hp_cost"),
                                        self._action_numeric_value(
                                            alt,
                                            ("hp_loss", "hp_cost", "typed_hp_loss"),
                                        ),
                                    )
                                    if hp_cost > 0.0 and hp - hp_cost <= 1.0:
                                        continue
                                    damage_value = max(
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
                                                "typed_damage_amount",
                                                "typed_damage",
                                            ),
                                        ),
                                    )
                                    block_value = _card_block_value(alt)
                                    heal_value = max(
                                        self._action_metric(alt, "heal"),
                                        self._action_numeric_value(
                                            alt,
                                            ("heal", "healing", "hp_gain", "typed_heal_amount"),
                                        ),
                                    )
                                    impact_value = float(self._action_immediate_impact(alt))
                                    roles = self._action_roles(alt)
                                    text = _identity_text(alt)
                                    attackish = bool(
                                        damage_value > 0.0
                                        or roles.intersection({"attack", "damage", "offense", "strike"})
                                        or any(token in text for token in ("strike", "slash", "attack", "打击", "攻击"))
                                    )
                                    progressish = bool(
                                        attackish
                                        or damage_value > 0.0
                                        or roles.intersection(progress_roles)
                                        or (impact_value >= 4.0 and block_value <= 0.0 and heal_value <= 0.0)
                                    )
                                    pure_block = bool(
                                        (block_value + heal_value) > 0.0
                                        and not attackish
                                        and damage_value <= 0.05
                                        and not roles.intersection(progress_roles)
                                    )
                                    score = 0.0
                                    kind = "other"
                                    if progressish:
                                        score += 50.0 + min(max(damage_value, 0.0), 30.0) + 0.15 * min(
                                            max(impact_value, 0.0),
                                            40.0,
                                        )
                                        kind = "progress"
                                    else:
                                        score += 5.0
                                    protection = float(block_value) + float(heal_value)
                                    if protection > 0.0:
                                        if threat_gap > 0.05:
                                            required = min(max(4.0, 0.25 * threat_gap), max(threat_gap, 1.0))
                                            if protection >= required:
                                                score += 20.0 + min(protection, threat_gap)
                                                if kind != "progress":
                                                    kind = "block"
                                            elif pure_block:
                                                score -= 20.0
                                        elif pure_block:
                                            # Zero-pressure Defend is the exact fallback pattern
                                            # that produced bad_pure_block_selected offenders.
                                            score -= 100.0
                                            kind = "idle_pure_block"
                                    score -= 0.50 * min(max(float(hp_cost), 0.0), 6.0)
                                    scored.append((float(score), int(alt_idx), kind))

                                if not scored:
                                    return None, "none"
                                scored.sort(key=lambda item: (-item[0], item[1]))
                                best_score, best_idx, best_kind = scored[0]
                                if best_kind == "idle_pure_block" and safe_idle_turn:
                                    search_stats[
                                        "combat_quality_potion_bad_guard_avoided_idle_pure_block"
                                    ] = 1.0
                                    return None, "idle_pure_block"
                                # Do not manufacture a bad non-potion fallback when every legal
                                # alternative is worse than simply failing open or safely ending.
                                if best_score < 0.0:
                                    return None, best_kind
                                return int(best_idx), best_kind

                            ranked_non_potion_alt, ranked_non_potion_kind = _non_potion_fallback_index()
                            allow_end_turn_fallback = bool(
                                # Resource/free-card potions often expose no
                                # immediate follow-up at 0 energy because the
                                # follow-up appears only after use.  Do not
                                # blindly rewrite them to End Turn in danger;
                                # only use End Turn as a fallback when the
                                # current turn is demonstrably safe.
                                (bad_no_followup_resource and (safe_to_end or clearly_waste_idle))
                                or (bad_no_followup_dependent and safe_to_end)
                                or clearly_waste_idle
                                or (bool(potion_profile.get("low_urgency", False)) and safe_to_end)
                                or (bool(potion_profile.get("save_recommended", False)) and moderate_safe_to_end)
                            )
                            hopeless_end_turn_potion = bool(
                                end_turn_alt >= 0
                                and not non_potion_alt
                                and current_energy <= 0.05
                                and (safe_idle_turn or amplify_block_noop)
                                and (
                                    # Observed Act1 failure: Fortifier/triple
                                    # block at 0 current block (even with
                                    # incoming damage) is a bridge-visible
                                    # pure no-op: it consumes a potion and
                                    # cannot reduce this turn's damage.  Do
                                    # not keep it fail-open just because End
                                    # Turn is unsafe; if it is the only
                                    # non-progress alternative, End Turn is
                                    # strictly better than burning the potion.
                                    bool(amplify_block_noop)
                                    or
                                    # Fortifier/triple block at 0 current
                                    # block and 0 incoming
                                    # is a bridge no-op that may remain legal
                                    # forever.  Keeping it "fail-open" just
                                    # teaches a repeat-use loop; End Turn is
                                    # the only progress action.
                                    bool(potion_profile.get("block_waste", False))
                                    or (
                                        bool(potion_profile.get("no_followup", False))
                                        and (
                                            bool(potion_profile.get("requires_followup", False))
                                            or bool(potion_profile.get("resource_like", False))
                                        )
                                    )
                                    or (
                                        bool(potion_profile.get("low_urgency", False))
                                        and float(potion_profile.get("use_quality", 0.0) or 0.0) <= 0.05
                                    )
                                )
                            )
                            hopeless_guard_fail_open = False
                            if end_turn_alt >= 0 and hopeless_end_turn_potion:
                                # Act1 recovery 2026-05-10: the "hopeless
                                # potion -> End Turn" exception is correct for
                                # true no-ops (Fortifier at 0 block, empty
                                # Liquid Memories), but diagnostics showed it
                                # also fires when the bridge/action surface does
                                # not expose the potion identity and the timing
                                # profile incorrectly looks idle.  Recompute a
                                # raw threat floor from the visible enemy intents
                                # and fail open for meaningful survival/race
                                # potions in late Act1 hallway or elite/boss
                                # danger windows.  This follows the current
                                # "learning beyond gradients" recovery style:
                                # never let a diagnostic guard manufacture a
                                # deterministic death target.
                                raw_incoming, raw_block, raw_hp = self._incoming_damage_pressure(
                                    raw_obs if isinstance(raw_obs, dict) else None
                                )
                                raw_threat_gap = max(0.0, float(raw_incoming) - float(raw_block))
                                effective_threat_gap = max(float(threat_gap), float(raw_threat_gap))
                                effective_hp = float(hp if hp > 0.0 else raw_hp)
                                effective_max_hp = float(max_hp)
                                if effective_max_hp <= 0.0 and hp_ratio > 0.0 and effective_hp > 0.0:
                                    effective_max_hp = effective_hp / max(float(hp_ratio), 1e-6)
                                effective_hp_ratio = float(hp_ratio)
                                if effective_hp_ratio <= 0.0 and effective_hp > 0.0 and effective_max_hp > 0.0:
                                    effective_hp_ratio = effective_hp / max(effective_max_hp, 1.0)

                                encounter_tier = self._combat_encounter_tier_from_raw(
                                    raw_obs if isinstance(raw_obs, dict) else None
                                )
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

                                profile_tags = {
                                    str(x).strip().lower()
                                    for key in ("effect_family", "semantic_tags", "timing_tags", "training_tags")
                                    for x in (potion_profile.get(key) or [])
                                    if str(x).strip()
                                }
                                potion_id = str(potion_profile.get("potion_id") or "").upper()
                                traits = _boss_race_potion_traits(selected, potion_profile, raw_obs)
                                retrieve_like = bool(
                                    bool(potion_profile.get("retrieve_from_discard_like", False))
                                    or float(potion_profile.get("retrieve_from_discard", 0.0) or 0.0) > 0.0
                                    or "LIQUID_MEMORIES" in potion_id
                                    or ("discard_pile" in profile_tags and "tutor" in profile_tags)
                                )
                                empty_retrieve = bool(retrieve_like and not bool(potion_profile.get("retrieve_has_target", False)))
                                block_value = float(potion_profile.get("block", 0.0) or 0.0)
                                heal_value = float(potion_profile.get("heal", 0.0) or 0.0)
                                damage_value = float(potion_profile.get("damage", 0.0) or 0.0)
                                boss_liquid_escape = _boss_zero_energy_liquid_escape(
                                    selected,
                                    potion_profile,
                                    raw_obs,
                                    encounter_tier=encounter_tier,
                                    hp=float(effective_hp),
                                    max_hp=float(effective_max_hp),
                                    hp_ratio=float(effective_hp_ratio),
                                    threat_gap=float(effective_threat_gap),
                                    current_energy=float(current_energy),
                                    no_non_potion_alt=not non_potion_alt,
                                )
                                lagavulin_setup_escape = _lagavulin_setup_liquid_escape(
                                    selected,
                                    potion_profile,
                                    raw_obs,
                                    encounter_tier=encounter_tier,
                                    encounter_hint=encounter,
                                    threat_gap=float(effective_threat_gap),
                                    current_energy=float(current_energy),
                                    no_non_potion_alt=not non_potion_alt,
                                )
                                invalid_noop = bool(
                                    amplify_block_noop
                                    or (empty_retrieve and not boss_liquid_escape and not lagavulin_setup_escape)
                                    or (
                                        bool(potion_profile.get("block_waste", False))
                                        and block_value <= 0.05
                                        and heal_value <= 0.05
                                        and damage_value <= 0.05
                                        and not bool(traits.get("candidate", False))
                                    )
                                        or (
                                            bool(potion_profile.get("block_waste", False))
                                            and effective_threat_gap <= 0.05
                                            and heal_value <= 0.05
                                            and damage_value <= 0.05
                                            and not bool(traits.get("candidate", False))
                                        )
                                    )
                                survival_tags = {
                                    "survival",
                                    "prevent_lethal_tool",
                                    "block",
                                    "heal",
                                    "healing",
                                    "defense",
                                    "defensive",
                                    "weak",
                                    "vulnerable",
                                    "debuff",
                                    "resource_survival_tool",
                                }
                                meaningful_survival_tool = bool(
                                    heal_value > 0.05
                                    or block_value > 0.05
                                    or bool(potion_profile.get("resource_survival_tool", False))
                                    or bool(potion_profile.get("critical_hp_survival_tool", False))
                                    or bool(potion_profile.get("prevent_lethal", False))
                                    or bool(potion_profile.get("prevent_major_loss", False))
                                    or bool(profile_tags.intersection(survival_tags))
                                    or boss_liquid_escape
                                    or lagavulin_setup_escape
                                )
                                meaningful_race_tool = bool(
                                    boss_liquid_escape
                                    or lagavulin_setup_escape
                                    or (
                                        traits.get("candidate", False)
                                        and not traits.get("invalid_context", False)
                                        and (
                                            traits.get("setup_like", False)
                                            or (
                                                traits.get("damage_like", False)
                                                and (effective_threat_gap > 0.05 or encounter_tier in {"elite", "boss"})
                                            )
                                        )
                                    )
                                )
                                meaningful_potion = bool(meaningful_race_tool or meaningful_survival_tool)
                                late_normal_window = bool(
                                    is_normal_hallway
                                    and (floor_value >= 8.0 or hard_normal_profile)
                                    and current_energy <= 0.05
                                    and not non_potion_alt
                                    and (
                                        effective_threat_gap >= 8.0
                                        or ((floor_value >= 11.0 or hard_normal_profile) and effective_threat_gap >= 6.0)
                                        or effective_hp_ratio <= 0.45
                                        or ((floor_value >= 11.0 or hard_normal_profile) and effective_hp_ratio <= 0.55)
                                        or raw_threat_gap >= 8.0
                                        or ((floor_value >= 11.0 or hard_normal_profile) and raw_threat_gap >= 6.0)
                                    )
                                )
                                boss_or_elite_window = bool(
                                    encounter_tier in {"elite", "boss"}
                                    and current_energy <= 0.05
                                    and not non_potion_alt
                                        and (
                                            effective_threat_gap >= 6.0
                                            or raw_threat_gap >= 6.0
                                            or bool(potion_profile.get("prevent_lethal", False))
                                            or bool(potion_profile.get("prevent_major_loss", False))
                                            or meaningful_race_tool
                                            or (
                                                effective_hp_ratio <= (0.55 if encounter_tier == "boss" else 0.45)
                                                and effective_threat_gap > 0.05
                                            )
                                        )
                                    )
                                if (not invalid_noop) and meaningful_potion and late_normal_window:
                                    hopeless_guard_fail_open = True
                                    hopeless_end_turn_potion = False
                                    allow_end_turn_fallback = False
                                    search_stats[
                                        "combat_quality_potion_bad_hopeless_guard_late_normal_race_skip"
                                    ] = 1.0
                                elif (not invalid_noop) and meaningful_potion and lagavulin_setup_escape:
                                    hopeless_guard_fail_open = True
                                    hopeless_end_turn_potion = False
                                    allow_end_turn_fallback = False
                                    search_stats[
                                        "combat_quality_potion_bad_hopeless_guard_lagavulin_setup_skip"
                                    ] = 1.0
                                elif (not invalid_noop) and meaningful_potion and boss_or_elite_window:
                                    hopeless_guard_fail_open = True
                                    hopeless_end_turn_potion = False
                                    allow_end_turn_fallback = False
                                    search_stats[
                                        "combat_quality_potion_bad_hopeless_guard_boss_survival_skip"
                                    ] = 1.0

                            if (not hopeless_guard_fail_open) and end_turn_alt >= 0 and hopeless_end_turn_potion:
                                override_idx = int(end_turn_alt)
                                self._dump_combat_hard_guard_record(
                                    kind="potion_bad_use_hopeless_end_turn",
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
                                search_stats["combat_quality_potion_bad_guard_applied"] = 1.0
                                search_stats["combat_quality_potion_bad_guard_override"] = 1.0
                                search_stats["combat_quality_potion_bad_guard_forced_end_turn_hopeless"] = 1.0
                                # Refresh post-override potion flags.
                                search_stats["combat_quality_potion_low_urgency_selected"] = 0.0
                                search_stats["combat_quality_potion_save_recommended_selected"] = 0.0
                                search_stats["combat_quality_potion_no_followup_selected"] = 0.0
                                search_stats["combat_quality_potion_block_waste_selected"] = 0.0
                                search_stats["combat_quality_potion_overkill_selected"] = 0.0
                                search_stats["combat_quality_potion_selected"] = 0.0
                            elif (not hopeless_guard_fail_open) and (
                                ranked_non_potion_alt is not None
                                or (end_turn_alt >= 0 and allow_end_turn_fallback)
                            ):
                                # Prefer doing another useful non-potion action.
                                # Only fall back to End Turn when the potion is
                                # specifically a no-followup resource spend or
                                # the turn is clearly safe/low-threat; otherwise
                                # do not create end-turn spam from uncertain
                                # potion timing scores.
                                override_idx = (
                                    int(ranked_non_potion_alt)
                                    if ranked_non_potion_alt is not None
                                    else int(end_turn_alt)
                                )
                                self._dump_combat_hard_guard_record(
                                    kind="potion_bad_use",
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
                                search_stats["combat_quality_potion_bad_guard_applied"] = 1.0
                                search_stats["combat_quality_potion_bad_guard_override"] = 1.0
                                if ranked_non_potion_alt is not None:
                                    if ranked_non_potion_kind == "progress":
                                        search_stats[
                                            "combat_quality_potion_bad_guard_progress_fallback"
                                        ] = 1.0
                                    elif ranked_non_potion_kind == "block":
                                        search_stats[
                                            "combat_quality_potion_bad_guard_block_fallback"
                                        ] = 1.0
                                # Refresh post-override potion flags.
                                search_stats["combat_quality_potion_low_urgency_selected"] = 0.0
                                search_stats["combat_quality_potion_save_recommended_selected"] = 0.0
                                search_stats["combat_quality_potion_no_followup_selected"] = 0.0
                                search_stats["combat_quality_potion_block_waste_selected"] = 0.0
                                search_stats["combat_quality_potion_overkill_selected"] = 0.0
                                search_stats["combat_quality_potion_selected"] = 0.0
                            else:
                                if (
                                    not hopeless_guard_fail_open
                                    and end_turn_alt >= 0
                                    and ranked_non_potion_alt is None
                                    and current_energy <= 0.05
                                ):
                                    search_stats["combat_quality_potion_bad_guard_end_turn_fallback_blocked_unsafe"] = 1.0
                                if not hopeless_guard_fail_open:
                                    search_stats["combat_quality_potion_bad_guard_no_alternative"] = 1.0
        return int(action_idx)
