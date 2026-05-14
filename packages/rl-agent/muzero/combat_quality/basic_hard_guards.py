"""Basic combat hard-guard mixins for MuZero training.

This module owns the small, early hard-guard families that used to live inside
``muzero.train.MuZeroTrainer._apply_combat_action_hard_guards``.  Keep each
family as a narrow method so future mechanism fixes do not regrow train.py.
"""

from __future__ import annotations

from typing import Any

from sts2_env.hp_cost_safety import hp_cost_safety_view


class BasicCombatHardGuardMixin:
    def _apply_potion_discard_priority_guard(self, *, action_idx: int, legal_count: int, legal_actions: list[Any], mask_np: Any, raw_obs: Any | None, encounter: str, search_stats: dict[str, Any]) -> int:
        # Potion-specific hard-guard helpers live in
        # muzero.combat_quality.potion_guard; keep this method as wiring only.

        # Act1 recovery: discard-potion priority guard.  Recent full-run
        # diagnostics showed forced overflow discarding POTION.FORTIFIER before
        # floor-13/14 deaths.  This is a policy-target poison source: the
        # actual modal must choose *some* potion, but treating all discard
        # choices as equivalent teaches the agent to throw away survival tools.
        #
        # This guard is intentionally limited to discard_potion actions and
        # only rewrites to another legal discard_potion candidate.  It never
        # blocks the modal itself and therefore cannot create illegal stalling.
        if 0 <= int(action_idx) < legal_count and isinstance(legal_actions[int(action_idx)], dict):
            selected = legal_actions[int(action_idx)]
            if self._semantic_family(selected) == "discard_potion":
                discard_candidates: list[tuple[float, int]] = []
                for idx in range(legal_count):
                    if mask_np[idx] <= 0:
                        continue
                    alt = legal_actions[idx]
                    if not isinstance(alt, dict):
                        continue
                    if self._semantic_family(alt) != "discard_potion":
                        continue
                    keep_value = float(self._discard_potion_keep_value(alt, raw_obs))
                    discard_candidates.append((keep_value, int(idx)))
                search_stats["combat_quality_potion_discard_guard_candidate_count"] = float(len(discard_candidates))
                if len(discard_candidates) >= 2:
                    search_stats["combat_quality_potion_discard_guard_available"] = 1.0
                    selected_keep = next((value for value, idx in discard_candidates if idx == int(action_idx)), 0.0)
                    # Lower keep_value is safer to discard.  Stable tie-break
                    # by action index keeps the guard deterministic.
                    discard_candidates.sort(key=lambda item: (item[0], item[1]))
                    best_keep, best_idx = discard_candidates[0]
                    if int(best_idx) != int(action_idx) and selected_keep >= best_keep + 0.30:
                        self._dump_combat_hard_guard_record(
                            kind="potion_discard_priority",
                            raw_obs=raw_obs,
                            legal_actions=legal_actions,
                            original_idx=int(action_idx),
                            override_idx=int(best_idx),
                            risk=float(selected_keep - best_keep),
                            countdown=None,
                            encounter=encounter,
                            lethal_exemption=False,
                        )
                        action_idx = int(best_idx)
                        search_stats["combat_quality_potion_discard_guard_applied"] = 1.0
                        search_stats["combat_quality_potion_discard_guard_override"] = 1.0
                        if selected_keep >= 0.85:
                            search_stats["combat_quality_potion_discard_guard_saved_survival"] = 1.0
        return int(action_idx)

    def _apply_kaiser_facing_guard(self, *, action_idx: int, legal_count: int, legal_actions: list[Any], mask_np: Any, raw_obs: Any | None, boss_ctx: Any | None, encounter: str, search_stats: dict[str, Any]) -> int:
        # P0-3: Kaiser facing change hard override.
        is_kaiser = bool(isinstance(boss_ctx, dict) and self._is_kaiser_encounter_context(boss_ctx, raw_obs))
        if is_kaiser:
            risk = float(self._kaiser_back_attack_risk_from_context(boss_ctx, raw_obs))
            if risk > 0.05:
                facing_indices = self._find_kaiser_facing_candidates(legal_actions, mask_np, raw_obs)
                if facing_indices:
                    search_stats["combat_quality_kaiser_facing_guard_available"] = 1.0
                    selected = legal_actions[int(action_idx)]
                    selected_is_facing = bool(
                        isinstance(selected, dict)
                        and self._is_kaiser_facing_change_action(selected, raw_obs)
                    )
                    if not selected_is_facing:
                        search_stats["combat_quality_kaiser_nonfacing_nonlethal_selected"] = 1.0
                        if self._semantic_family(selected) == "end_turn":
                            search_stats["combat_quality_kaiser_end_turn_under_risk_with_candidate"] = 1.0
                        lethal = bool(
                            isinstance(selected, dict)
                            and self._is_action_confirmed_lethal(selected, raw_obs)
                        )
                        if lethal:
                            search_stats["combat_quality_kaiser_facing_guard_lethal_exemption"] = 1.0
                        else:
                            override_idx = int(facing_indices[0])
                            self._dump_combat_hard_guard_record(
                                kind="kaiser_facing",
                                raw_obs=raw_obs,
                                legal_actions=legal_actions,
                                original_idx=int(action_idx),
                                override_idx=override_idx,
                                risk=risk,
                                countdown=None,
                                encounter=encounter,
                                lethal_exemption=False,
                            )
                            action_idx = override_idx
                            search_stats["combat_quality_kaiser_facing_guard_applied"] = 1.0
                            search_stats["combat_quality_kaiser_facing_guard_override"] = 1.0
                            # Refresh post-override behavior flags so the
                            # downstream `boss_combat/kaiser_*_selected_rate`
                            # tags reflect the action that will actually be
                            # taken.
                            search_stats["combat_quality_kaiser_facing_change_selected"] = 1.0
                            search_stats["combat_quality_kaiser_risky_end_turn_selected"] = 0.0
                            search_stats["combat_quality_kaiser_nonfacing_nonlethal_selected"] = 0.0
                            search_stats["combat_quality_kaiser_end_turn_under_risk_with_candidate"] = 0.0
        return int(action_idx)

    def _apply_insatiable_escape_guard(self, *, action_idx: int, legal_count: int, legal_actions: list[Any], mask_np: Any, raw_obs: Any | None, boss_ctx: Any | None, encounter: str, search_stats: dict[str, Any]) -> int:
        # P0-4: Insatiable Frantic Escape hard force on countdown <= 1.
        is_insatiable = bool(
            isinstance(boss_ctx, dict)
            and self._is_insatiable_encounter_context(boss_ctx, raw_obs)
        )
        if is_insatiable:
            countdown = self._insatiable_sandpit_countdown_from_context(boss_ctx, raw_obs)
            if countdown is not None and countdown <= 1.0:
                escape_indices = self._find_insatiable_frantic_escape_candidates(legal_actions, mask_np)
                if escape_indices:
                    search_stats["combat_quality_insatiable_escape_force_available"] = 1.0
                    selected = legal_actions[int(action_idx)]
                    selected_is_escape = bool(
                        isinstance(selected, dict) and int(action_idx) in escape_indices
                    )
                    if not selected_is_escape:
                        search_stats["combat_quality_insatiable_non_escape_at1_blocked"] = 1.0
                        lethal = bool(
                            isinstance(selected, dict)
                            and self._is_action_confirmed_lethal(selected, raw_obs)
                        )
                        if lethal:
                            search_stats["combat_quality_insatiable_escape_force_lethal_exemption"] = 1.0
                        else:
                            override_idx = int(escape_indices[0])
                            self._dump_combat_hard_guard_record(
                                kind="insatiable_escape",
                                raw_obs=raw_obs,
                                legal_actions=legal_actions,
                                original_idx=int(action_idx),
                                override_idx=override_idx,
                                risk=0.0,
                                countdown=float(countdown),
                                encounter=encounter,
                                lethal_exemption=False,
                            )
                            action_idx = override_idx
                            search_stats["combat_quality_insatiable_escape_force_applied"] = 1.0
                            search_stats["combat_quality_insatiable_escape_force_override"] = 1.0
                            search_stats["combat_quality_insatiable_frantic_escape_selected"] = 1.0
                            search_stats["combat_quality_insatiable_frantic_escape_missed_at_1"] = 0.0
                            search_stats["combat_quality_insatiable_frantic_escape_missed_lt3"] = 0.0
                            search_stats["combat_quality_insatiable_non_escape_at1_blocked"] = 0.0
        return int(action_idx)

    def _apply_x_cost_zero_guard(self, *, action_idx: int, legal_count: int, legal_actions: list[Any], mask_np: Any, raw_obs: Any | None, encounter: str, search_stats: dict[str, Any]) -> int:
        # P1-2 (recovery 2026-05-07): X-cost zero-energy hard invalid.
        # If the post-search action is an X-cost card with effective_energy
        # at or below zero AND no zero-energy effect, override to the first
        # legal non-X-cost / non-end_turn alternative.  If the only legal
        # escape is End Turn, take it instead of spending/discarding a no-op
        # X-card; live diagnostics showed 0-energy ``倾泻+`` staying selected
        # when the action set was {bad X-card, End Turn}.  The guard only
        # consumes the diagnostic helper that already powers
        # ``combat_quality_x_cost_bad_selected`` so detection here matches
        # the existing soft-bias path; only the override is new.
        if 0 <= int(action_idx) < legal_count and isinstance(legal_actions[int(action_idx)], dict):
            current_energy = float(self._combat_energy(None, raw_obs))
            selected = legal_actions[int(action_idx)]
            x_diag = self._x_cost_diagnostic(selected, current_energy)
            if float(x_diag.get("x_cost_bad", 0.0)) > 0.5:
                search_stats["combat_quality_x_cost_zero_guard_available"] = 1.0
                non_x_alt: list[int] = []
                end_turn_alt: list[int] = []
                for idx in range(legal_count):
                    if idx == int(action_idx) or mask_np[idx] <= 0:
                        continue
                    alt = legal_actions[idx]
                    if not isinstance(alt, dict):
                        continue
                    if self._semantic_family(alt) == "end_turn":
                        # Prefer any real legal action first, but keep End Turn
                        # as the final safe fallback.  A zero-energy no-op
                        # X-card is never better than ending the turn when no
                        # non-X play exists.
                        end_turn_alt.append(idx)
                        continue
                    alt_diag = self._x_cost_diagnostic(alt, current_energy)
                    if float(alt_diag.get("is_x_cost", 0.0)) > 0.5:
                        continue  # alt also X-cost, may still be zero-bad
                    non_x_alt.append(idx)
                if non_x_alt or end_turn_alt:
                    override_idx = int(non_x_alt[0] if non_x_alt else end_turn_alt[0])
                    self._dump_combat_hard_guard_record(
                        kind="x_cost_zero",
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
                    search_stats["combat_quality_x_cost_zero_guard_applied"] = 1.0
                    search_stats["combat_quality_x_cost_zero_guard_override"] = 1.0
                    search_stats["combat_quality_hard_guard_override_any"] = 1.0
                    if not non_x_alt:
                        search_stats["combat_quality_x_cost_zero_guard_end_turn_fallback"] = 1.0
                    # Refresh selected-action flags so downstream
                    # ``boss_combat/x_cost_zero_*_selected_rate`` tags
                    # reflect the action that will actually be taken.
                    search_stats["combat_quality_x_cost_selected"] = 0.0
                    search_stats["combat_quality_x_cost_selected_energy"] = 0.0
                    search_stats["combat_quality_x_cost_selected_effective_energy"] = 0.0
                    search_stats["combat_quality_x_cost_has_non_energy_effect_selected"] = 0.0
                    search_stats["combat_quality_x_cost_bad_selected"] = 0.0
                    search_stats["combat_quality_x_cost_zero_bad_selected"] = 0.0
                    search_stats["combat_quality_x_cost_zero_selected"] = 0.0
                    search_stats["combat_quality_zero_energy_x_cost_selected"] = 0.0
                    search_stats["combat_quality_x_cost_zero_energy_selected"] = 0.0
                else:
                    # No non-X alternative — leave the action and record so
                    # we know how often the guard is forced to no-op.
                    search_stats["combat_quality_x_cost_zero_guard_no_alternative"] = 1.0
        return int(action_idx)

    def _apply_hp_cost_margin_guard(self, *, action_idx: int, legal_count: int, legal_actions: list[Any], mask_np: Any, raw_obs: Any | None, encounter: str, search_stats: dict[str, Any]) -> int:
        # P1-3 (recovery 2026-05-07): HP-cost survival-margin hard guard.
        # Reads bridge typed safety + this turn's incoming damage estimate
        # to decide whether the action would leave the player dead AFTER
        # incoming resolves. ``hp_cost_safety_view`` only covers self-cost;
        # this guard adds the incoming-damage axis the spec demands.
        if 0 <= int(action_idx) < legal_count and isinstance(legal_actions[int(action_idx)], dict):
            selected = legal_actions[int(action_idx)]
            safety = hp_cost_safety_view(selected, raw_obs)
            hp_cost_unblockable = float(safety.get("hp_loss_unblockable", 0.0) or 0.0)
            if hp_cost_unblockable > 0.0:
                search_stats["combat_quality_hp_cost_margin_guard_available"] = 1.0
                try:
                    incoming, current_block, current_hp = self._incoming_damage_pressure(raw_obs)
                except Exception:
                    incoming, current_block, current_hp = 0.0, 0.0, 0.0
                player = raw_obs.get("player") if isinstance(raw_obs, dict) else None
                max_hp = 0.0
                if isinstance(player, dict):
                    try:
                        max_hp = float(player.get("max_hp") or 0.0)
                    except (TypeError, ValueError):
                        max_hp = 0.0
                if max_hp <= 0.0:
                    max_hp = max(float(current_hp), 1.0)
                after_self_hp = max(0.0, float(current_hp) - hp_cost_unblockable)
                projected_damage = max(0.0, float(incoming) - float(current_block))
                survival_margin = after_self_hp - projected_damage
                low_margin_after_cost = bool(safety.get("low_hp_margin_after_cost"))
                risky_after_cost = bool(survival_margin <= 0.0 or low_margin_after_cost)
                if risky_after_cost and current_hp > 0.0:
                    lethal = bool(self._is_action_confirmed_lethal(selected, raw_obs))
                    if lethal:
                        search_stats["combat_quality_hp_cost_margin_guard_lethal_exemption"] = 1.0
                    else:
                        # Find a non-end_turn alternative without HP cost.
                        non_hp_alt: list[tuple[float, int]] = []
                        for idx in range(legal_count):
                            if idx == int(action_idx) or mask_np[idx] <= 0:
                                continue
                            alt = legal_actions[idx]
                            if not isinstance(alt, dict):
                                continue
                            if self._semantic_family(alt) == "end_turn":
                                continue
                            alt_safety = hp_cost_safety_view(alt, raw_obs)
                            if float(alt_safety.get("hp_loss_unblockable", 0.0) or 0.0) > 0.0:
                                continue
                            score = max(
                                self._action_immediate_impact(alt),
                                self._action_metric(alt, "damage"),
                                self._action_metric(alt, "total_damage"),
                                self._action_metric(alt, "block"),
                                0.01,
                            )
                            non_hp_alt.append((float(score), int(idx)))
                        if non_hp_alt:
                            non_hp_alt.sort(key=lambda item: (-item[0], item[1]))
                            override_idx = int(non_hp_alt[0][1])
                            self._dump_combat_hard_guard_record(
                                kind="hp_cost_margin",
                                raw_obs=raw_obs,
                                legal_actions=legal_actions,
                                original_idx=int(action_idx),
                                override_idx=override_idx,
                                risk=float(min(survival_margin, float(safety.get("hp_after_self_cost", 0.0) or 0.0))),
                                countdown=None,
                                encounter=encounter,
                                lethal_exemption=False,
                            )
                            action_idx = override_idx
                            search_stats["combat_quality_hp_cost_margin_guard_applied"] = 1.0
                            search_stats["combat_quality_hp_cost_margin_guard_override"] = 1.0
                            # Refresh post-override safety flags.
                            search_stats["combat_quality_hp_cost_self_lethal_selected"] = 0.0
                            search_stats["combat_quality_hp_cost_low_margin_selected"] = 0.0
                        else:
                            search_stats["combat_quality_hp_cost_margin_guard_no_alternative"] = 1.0
        return int(action_idx)

    def _apply_elite_boss_lethal_endturn_guard(self, *, action_idx: int, legal_count: int, legal_actions: list[Any], mask_np: Any, raw_obs: Any | None, encounter: str, search_stats: dict[str, Any]) -> int:
        # P1-4a (act1 recovery 2026-05-10): elite/boss lethal EndTurn guard.
        # The survival guards below deliberately avoid replacing a kill with a
        # defensive potion/card, but merely "exempting" would still leave the
        # original End Turn selected.  When a legal action is already confirmed
        # lethal, taking it is the safest narrow override for tonight's Act1
        # bottleneck.
        if 0 <= int(action_idx) < legal_count and isinstance(legal_actions[int(action_idx)], dict):
            selected = legal_actions[int(action_idx)]
            if self._semantic_family(selected) == "end_turn":
                encounter_tier = self._combat_encounter_tier_from_raw(raw_obs if isinstance(raw_obs, dict) else None)
                if encounter_tier in {"elite", "boss"}:
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
                        search_stats["combat_quality_elite_boss_lethal_end_turn_guard_available"] = 1.0
                        lethal_candidates.sort(key=lambda item: (-item[0], item[1]))
                        override_idx = int(lethal_candidates[0][1])
                        self._dump_combat_hard_guard_record(
                            kind="elite_boss_lethal_end_turn",
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
                        search_stats["combat_quality_elite_boss_lethal_end_turn_guard_applied"] = 1.0
                        search_stats["combat_quality_elite_boss_lethal_end_turn_guard_override"] = 1.0
                        search_stats["combat_quality_wasteful_end_turn_selected"] = 0.0
                        search_stats["combat_quality_end_turn_selected"] = 0.0
        return int(action_idx)

