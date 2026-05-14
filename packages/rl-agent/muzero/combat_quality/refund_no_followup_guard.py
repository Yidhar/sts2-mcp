"""Hard guard for refund/setup cards that have no useful follow-up.

Some cards are only good when they unlock a real follow-up in the same turn:
energy refunds, temporary cost modification, hand mutation, replay/retain
setup, etc.  If the selected action is classified as "refund/no-followup",
playing it spends a card and often exhausts or mutates resources for no combat
progress.  The policy has been observed to keep taking these legal-but-bad
plays in combat sandbox.

This guard is deliberately narrow:

* it only considers the *selected* play_card action;
* it uses the same central classifier flags that power
  ``combat_quality_refund_no_followup_selected``;
* it never overrides a confirmed lethal selected action;
* it first rewrites to an affordable, safe, immediate progress card;
* when no progress card exists, it may fall back to End Turn for a selected
  HP-cost / no-real-progress setup card, because playing ``放血``-style
  no-followup cards is strictly worse than ending the turn.

Keep this outside ``muzero.train`` so tactical combat fixes do not grow the
training entrypoint further.
"""

from __future__ import annotations

from typing import Any

from sts2_env.hp_cost_safety import hp_cost_safety_view


class RefundNoFollowupGuardMixin:
    """Rewrite bad refund/setup plays to safe progress alternatives."""

    @staticmethod
    def _refund_no_followup_guard_selected_bad(profile: dict[str, Any]) -> bool:
        """Return true when the selected profile matches this guard.

        The broad selected metric is driven by the central taxonomy.  Mirror
        those flags here instead of matching names/text so future card metadata
        improvements automatically feed the guard.
        """

        if not isinstance(profile, dict):
            return False
        if bool(profile.get("energy_without_followup", False)):
            return True
        if bool(profile.get("followup_missing", False)):
            return True
        if (
            bool(profile.get("setup_followup_dependent", False))
            and not bool(profile.get("setup_followup_available", False))
        ):
            return True
        return False

    @staticmethod
    def _refund_no_followup_guard_reset_selected_stats(search_stats: dict[str, Any]) -> None:
        """Clear selected-side no-followup gauges after a successful rewrite."""

        for key in (
            "combat_quality_refund_no_followup_selected",
            "combat_quality_refund_no_followup_with_progress_selected",
            "combat_quality_refund_no_followup_progress_alternative_selected",
            "combat_quality_refund_no_followup_no_alternative_selected",
            "combat_quality_refund_no_followup_progress_alternative_count",
            "combat_quality_strategic_skip_selected",
        ):
            search_stats[key] = 0.0

    def _refund_no_followup_progress_candidates(
        self,
        *,
        selected_idx: int,
        legal_count: int,
        legal_actions: list[Any],
        mask_np: Any,
        raw_obs: dict[str, Any],
        current_energy: float,
    ) -> list[tuple[tuple[float, float, float, float, float], int, float, float, bool]]:
        """Return safe immediate-progress alternatives for a no-followup play.

        This is the single predicate used by both the hard guard and the narrow
        selected-side metric.  Keeping them shared prevents the monitor from
        failing on broad refund/no-followup cases where the player truly had no
        better legal card to play.
        """

        candidates: list[tuple[tuple[float, float, float, float, float], int, float, float, bool]] = []
        safe_legal_count = max(0, min(int(legal_count), len(legal_actions)))
        for idx in range(safe_legal_count):
            if idx == int(selected_idx):
                continue
            try:
                if mask_np[idx] <= 0:
                    continue
            except Exception:
                continue

            alt = legal_actions[idx]
            if not isinstance(alt, dict) or self._semantic_family(alt) != "play_card":
                continue

            cost = float(self._action_cost_value(alt))
            if cost > float(current_energy) + 1e-6:
                continue

            x_diag = self._x_cost_diagnostic(alt, float(current_energy))
            if float(x_diag.get("x_cost_bad", 0.0) or 0.0) > 0.5:
                continue

            lethal = bool(self._is_action_confirmed_lethal(alt, raw_obs))
            safety = hp_cost_safety_view(alt, raw_obs)
            hp_loss = float(safety.get("hp_loss_unblockable", 0.0) or 0.0)
            if hp_loss > 0.0 and not lethal:
                continue

            profile = self._classify_positive_combat_action(
                alt,
                int(idx),
                None,
                raw_obs,
                legal_actions,
                mask_np,
                float(current_energy),
            )
            if not bool(profile.get("positive", False)):
                continue
            if bool(profile.get("card_block_waste", False)) or bool(profile.get("card_pure_block", False)):
                continue

            damage = max(
                self._action_metric(alt, "damage"),
                self._action_metric(alt, "total_damage"),
                self._action_numeric_value(
                    alt,
                    ("damage", "total_damage", "attack_damage", "preview_damage", "expected_damage"),
                ),
            )
            impact = float(self._action_immediate_impact(alt))
            roles = self._action_roles(alt)
            progress_roles = bool(
                roles.intersection(
                    {
                        "attack",
                        "damage",
                        "debuff",
                        "weak",
                        "vulnerable",
                        "poison",
                        "stun",
                        "artifact_strip",
                        "lock",
                        "mechanism",
                        "facing_change",
                        "scaling",
                        "power",
                        "draw",
                        "discard",
                        "resource",
                        "energy",
                    }
                )
            )
            high_damage = bool(damage >= 8.0 or impact >= 8.0)
            meaningful_progress = bool(lethal or damage > 0.0 or impact >= 3.0 or progress_roles)
            if not meaningful_progress:
                continue

            if bool(profile.get("deferable", False)) and not (lethal or high_damage):
                continue
            if bool(profile.get("followup_missing", False)) and not (lethal or high_damage):
                continue
            if bool(profile.get("energy_without_followup", False)) and not (lethal or high_damage):
                continue
            if (
                bool(profile.get("setup_followup_dependent", False))
                and not bool(profile.get("setup_followup_available", False))
                and not (lethal or high_damage)
            ):
                continue

            score = (
                1.0 if lethal else 0.0,
                float(damage),
                float(impact),
                -float(cost),
                -float(idx),
            )
            candidates.append((score, int(idx), float(damage), float(impact), lethal))
        return candidates

    def _refund_no_followup_end_turn_fallback_idx(
        self,
        *,
        selected: dict[str, Any],
        selected_profile: dict[str, Any],
        selected_idx: int,
        legal_count: int,
        legal_actions: list[Any],
        mask_np: Any,
        raw_obs: dict[str, Any],
    ) -> int | None:
        """Return a legal End Turn fallback for harmful no-followup setup.

        This is intentionally narrower than the progress-candidate path.  It
        exists for live death-slice patterns such as ``放血``/``放血+`` at the
        end of the playable turn: the card is tagged as resource/setup, no
        follow-up is available, and the only remaining legal alternative is End
        Turn.  Keeping the no-followup card selected spends HP / mutates the
        hand for no immediate combat effect; End Turn preserves HP and is the
        safer target for replay.
        """

        end_turn_idx: int | None = None
        safe_legal_count = max(0, min(int(legal_count), len(legal_actions)))
        for idx in range(safe_legal_count):
            if idx == int(selected_idx):
                continue
            try:
                if mask_np[idx] <= 0:
                    continue
            except Exception:
                continue
            alt = legal_actions[idx]
            if isinstance(alt, dict) and self._semantic_family(alt) == "end_turn":
                end_turn_idx = int(idx)
                break
        if end_turn_idx is None:
            return None

        safety = hp_cost_safety_view(selected, raw_obs)
        hp_loss = float(safety.get("hp_loss_unblockable", 0.0) or 0.0)

        damage = max(
            self._action_metric(selected, "damage"),
            self._action_metric(selected, "total_damage"),
            self._action_numeric_value(
                selected,
                ("damage", "total_damage", "attack_damage", "preview_damage", "expected_damage"),
            ),
        )
        block = max(
            self._action_metric(selected, "block"),
            self._action_metric(selected, "total_block"),
            self._action_numeric_value(
                selected,
                ("block", "total_block", "preview_block", "expected_block"),
            ),
        )
        # Resource/setup classifiers may assign a positive heuristic impact to
        # energy generation, so use *real* damage/block/heal for the fallback
        # safety check.  Confirmed lethal was already exempted by the caller.
        heal = max(
            self._action_metric(selected, "heal"),
            self._action_numeric_value(selected, ("heal", "healing", "hp_gain")),
        )
        has_real_combat_progress = bool(float(damage) > 0.0 or float(block) > 0.0 or float(heal) > 0.0)
        if has_real_combat_progress:
            return None

        # Primary live offender: self-damage resource/setup with no follow-up.
        if hp_loss > 0.0:
            return end_turn_idx

        # Secondary narrow case: a selected no-followup card with literally no
        # immediate effect should also not be preferred over End Turn when the
        # turn has no progress card left.
        immediate_impact = float(self._action_immediate_impact(selected))
        roles = self._action_roles(selected)
        only_deferable_setup = bool(
            roles.issubset({"resource", "setup", "energy", "draw", "discard"}) if roles else True
        )
        if immediate_impact <= 0.5 and only_deferable_setup and bool(selected_profile.get("deferable", True)):
            return end_turn_idx
        return None

    def _apply_refund_no_followup_guard(
        self,
        *,
        action_idx: int,
        legal_count: int,
        legal_actions: list[Any],
        mask_np: Any,
        raw_obs: Any | None,
        encounter: str,
        search_stats: dict[str, Any],
    ) -> int:
        """Rewrite a selected no-followup refund/setup card when possible."""

        original_idx = int(action_idx)
        if not (0 <= original_idx < int(legal_count)):
            return original_idx
        selected = legal_actions[original_idx]
        if not isinstance(selected, dict) or self._semantic_family(selected) != "play_card":
            return original_idx
        if not isinstance(raw_obs, dict):
            return original_idx

        current_energy = float(self._combat_energy(None, raw_obs))
        selected_profile = self._classify_positive_combat_action(
            selected,
            original_idx,
            None,
            raw_obs,
            legal_actions,
            mask_np,
            current_energy,
        )
        if not self._refund_no_followup_guard_selected_bad(selected_profile):
            return original_idx

        search_stats["combat_quality_refund_no_followup_guard_available"] = 1.0

        if bool(self._is_action_confirmed_lethal(selected, raw_obs)):
            search_stats["combat_quality_refund_no_followup_guard_lethal_exemption"] = 1.0
            return original_idx

        candidates = self._refund_no_followup_progress_candidates(
            selected_idx=original_idx,
            legal_count=int(legal_count),
            legal_actions=legal_actions,
            mask_np=mask_np,
            raw_obs=raw_obs,
            current_energy=current_energy,
        )

        search_stats["combat_quality_refund_no_followup_guard_candidate_count"] = float(len(candidates))
        if not candidates:
            search_stats["combat_quality_refund_no_followup_guard_no_alternative"] = 1.0
            end_turn_idx = self._refund_no_followup_end_turn_fallback_idx(
                selected=selected,
                selected_profile=selected_profile,
                selected_idx=original_idx,
                legal_count=int(legal_count),
                legal_actions=legal_actions,
                mask_np=mask_np,
                raw_obs=raw_obs,
            )
            if end_turn_idx is not None:
                self._dump_combat_hard_guard_record(
                    kind="refund_no_followup",
                    raw_obs=raw_obs,
                    legal_actions=legal_actions,
                    original_idx=original_idx,
                    override_idx=int(end_turn_idx),
                    risk=0.0,
                    countdown=None,
                    encounter=encounter,
                    lethal_exemption=False,
                )
                search_stats["combat_quality_refund_no_followup_guard_applied"] = 1.0
                search_stats["combat_quality_refund_no_followup_guard_override"] = 1.0
                search_stats["combat_quality_refund_no_followup_guard_end_turn_fallback"] = 1.0
                search_stats["combat_quality_hard_guard_override_any"] = 1.0
                self._refund_no_followup_guard_reset_selected_stats(search_stats)
                return int(end_turn_idx)
            return original_idx

        candidates.sort(key=lambda item: item[0], reverse=True)
        _score, best_idx, best_damage, best_impact, best_lethal = candidates[0]
        if best_lethal:
            search_stats["combat_quality_refund_no_followup_guard_lethal_candidate"] = 1.0

        if int(best_idx) != original_idx:
            self._dump_combat_hard_guard_record(
                kind="refund_no_followup",
                raw_obs=raw_obs,
                legal_actions=legal_actions,
                original_idx=original_idx,
                override_idx=int(best_idx),
                risk=float(max(best_damage, best_impact)),
                countdown=None,
                encounter=encounter,
                lethal_exemption=False,
            )
            search_stats["combat_quality_refund_no_followup_guard_applied"] = 1.0
            search_stats["combat_quality_refund_no_followup_guard_override"] = 1.0
            search_stats["combat_quality_hard_guard_override_any"] = 1.0
            self._refund_no_followup_guard_reset_selected_stats(search_stats)
            return int(best_idx)

        return original_idx
