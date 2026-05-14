"""Global no/trivial-pressure pure-block hard guard for combat quality.

The model can legally play Defend-like cards even when enemies are buffing,
stunned, or dealing only tiny non-urgent damage.  That behavior is costly in
Act 1 because it spends energy that should shorten the fight, which raises
future HP loss and lowers campfire upgrade opportunities.

This guard is intentionally conservative:

* it only acts after the policy/search selected a ``play_card`` action that is
  classified as pure block or block waste;
* it skips meaningful pressure windows where block is actually urgent;
* it only rewrites to a safe, affordable, immediate damage/progress card;
* it refuses non-lethal HP-cost cards, bad zero-energy X-cost plays, pure block
  alternatives, and setup/refund cards that have no follow-up.

Keep this module separate from ``muzero.train`` and from boss-specific guards:
the pathology is global across weak/normal/elite/boss combats.
"""

from __future__ import annotations

import json
import time
from typing import Any

import numpy as np

from muzero.combat_quality.progress_candidates import collect_safe_progress_candidates


class NoPressureBlockGuardMixin:
    """Rewrite useless pure-block choices into safe fight-progress cards."""

    @staticmethod
    def _no_pressure_block_guard_reset_block_stats(search_stats: dict[str, Any]) -> None:
        """Clear selected pure-block gauges after a successful rewrite."""

        for key in (
            "combat_quality_card_block_waste_selected",
            "combat_quality_card_block_waste_with_progress_selected",
            "combat_quality_card_pure_block_selected",
            "combat_quality_card_no_damage_pressure_selected",
            "combat_quality_card_no_damage_pressure_with_progress_selected",
            # Narrow pure-block buckets are computed before hard-guard
            # rewrites and mirrored into TensorBoard.  If this guard replaces
            # the selected Defend-like card, the selected-side narrow gauges
            # must describe the *post-override* action as well; otherwise the
            # sandbox gate keeps seeing stale bad-pure-block selections even
            # when execution was corrected.
            "combat_quality_bad_pure_block_selected",
            "combat_quality_insufficient_block_selected",
            "combat_quality_pure_block_progress_alternative_selected",
            "combat_quality_pure_block_survival_justified_selected",
            "combat_quality_pure_block_no_alternative_selected",
            "combat_quality_pure_block_low_value_pressure_selected",
        ):
            search_stats[key] = 0.0

    @staticmethod
    def _no_pressure_block_guard_trivial_threshold(current_hp: float) -> float:
        """Tiny uncovered damage that should not force low-value blocking.

        The threshold intentionally stays small.  At high HP, taking 1-3 damage
        to shorten the fight can be correct; at low HP or meaningful pressure,
        survival guards should retain control.
        """

        try:
            hp = max(float(current_hp or 0.0), 0.0)
        except (TypeError, ValueError):
            hp = 0.0
        if hp <= 0.0:
            return 0.0
        return float(min(3.0, max(2.0, 0.05 * hp)))

    @staticmethod
    def _no_pressure_block_guard_low_value_pressure_window(
        *,
        selected_block: float,
        threat_gap: float,
        current_hp: float,
        encounter_tier: str | None,
    ) -> bool:
        """Return true for safe hallway pressure where low block is a tempo trap.

        This is intentionally narrower than ``_is_meaningful_block_urgent``.
        It does *not* say "ignore damage"; it says that spending energy on a
        tiny pure-block card is likely worse than racing when all of these are
        true:

        * the room is not elite/boss;
        * HP is still safe;
        * uncovered damage is not near-lethal/high pressure;
        * the selected block covers only a small fraction of the threat.
        """

        tier = str(encounter_tier or "").strip().lower()
        if tier in {"elite", "boss"}:
            return False
        try:
            hp = float(current_hp or 0.0)
            threat = float(threat_gap or 0.0)
            block = float(selected_block or 0.0)
        except (TypeError, ValueError):
            return False
        if hp < 30.0:
            return False
        if threat <= 0.05:
            return False
        if threat >= hp - 8.0:
            return False
        post_hit_hp = hp - threat

        # Very small hallway pressure is also a tempo trap even when the
        # selected block card over-covers the hit.  This covers the live
        # offender class seen in sandbox logs:
        #
        #   HP 71/80, incoming 5, energy 1, selected Defend+ for 8 block,
        #   with two legal Strikes for 9 damage.
        #
        # The previous ratio gate below required ``block <= max(7, .5*threat)``
        # and therefore missed this case solely because Defend+ had 8 block.
        # For weak/normal hallway fights, spending the last energy on pure
        # block against 5-6 damage at high HP lengthens the fight and usually
        # increases future attrition.  Keep the post-hit margin high so this
        # does not fire around low HP.
        low_absolute_pressure = bool(
            threat <= max(6.0, min(10.0, 0.12 * hp))
            and post_hit_hp >= max(30.0, 0.45 * hp)
        )
        if low_absolute_pressure:
            return True

        # Old rule used ``threat <= max(14, 0.35*hp)``.  That was too narrow
        # for exactly the Act1 hallway race failures we are now seeing:
        # e.g. HP 68 / incoming 27 / selected block 6 is very safe after hit
        # (41 HP remains) but ``27 > 0.35*68`` made the guard stand down.  Use
        # post-hit margin instead: if the player is still comfortably above a
        # floor after taking the attack, spending energy on tiny block is a
        # tempo trap when a progress card is legal.
        if post_hit_hp < max(18.0, 0.25 * hp):
            return False
        if block > max(7.0, 0.50 * threat):
            return False
        return True

    @staticmethod
    def _no_pressure_block_guard_json_scalar(value: Any) -> Any:
        """Return a compact JSON-friendly scalar for guard eval dumps."""

        try:
            if isinstance(value, np.generic):
                value = value.item()
        except Exception:
            pass
        if isinstance(value, bool):
            return bool(value)
        if isinstance(value, int):
            return int(value)
        if isinstance(value, float):
            try:
                if not np.isfinite(value):
                    return None
            except Exception:
                pass
            return float(value)
        if value is None or isinstance(value, str):
            return value
        return str(value)

    @staticmethod
    def _no_pressure_block_guard_mask_value(mask_np: Any, idx: int) -> float:
        try:
            arr = np.asarray(mask_np, dtype=np.float32).reshape(-1)
            if 0 <= int(idx) < int(arr.shape[0]):
                return float(arr[int(idx)])
        except Exception:
            pass
        return 0.0

    def _no_pressure_block_guard_action_summary(
        self,
        *,
        idx: int,
        action: Any,
        mask_np: Any,
        raw_obs: dict[str, Any],
        legal_actions: list[Any],
        current_energy: float,
    ) -> dict[str, Any]:
        """Summarize one action as seen by the no-pressure block guard."""

        if not isinstance(action, dict):
            return {"idx": int(idx), "missing": True, "mask": self._no_pressure_block_guard_mask_value(mask_np, idx)}
        card = action.get("card") if isinstance(action.get("card"), dict) else {}
        potion = action.get("potion") if isinstance(action.get("potion"), dict) else {}
        try:
            roles = sorted(str(role) for role in self._action_roles(action))
        except Exception:
            roles = []
        try:
            profile = self._classify_positive_combat_action(
                action,
                int(idx),
                None,
                raw_obs,
                legal_actions,
                mask_np,
                float(current_energy),
            )
        except Exception:
            profile = {}
        profile_keep: dict[str, Any] = {}
        if isinstance(profile, dict):
            for key, value in profile.items():
                if isinstance(value, (bool, int, float, str, np.generic)) or value is None:
                    profile_keep[str(key)] = self._no_pressure_block_guard_json_scalar(value)

        def metric(name: str) -> float:
            try:
                return float(self._action_metric(action, name))
            except Exception:
                return 0.0

        try:
            impact = float(self._action_immediate_impact(action))
        except Exception:
            impact = 0.0
        try:
            cost = float(self._action_cost_value(action))
        except Exception:
            cost = 0.0
        try:
            target_combat_id = self._action_target_combat_id(action)
        except Exception:
            target_combat_id = None
        try:
            lethal = bool(self._is_action_confirmed_lethal(action, raw_obs))
        except Exception:
            lethal = False

        return {
            "idx": int(idx),
            "mask": self._no_pressure_block_guard_mask_value(mask_np, idx),
            "family": self._semantic_family(action),
            "kind": action.get("kind"),
            "action_id": action.get("action_id"),
            "label": action.get("label"),
            "title": action.get("title") or card.get("title") or card.get("name") or potion.get("title") or potion.get("name"),
            "card_id": card.get("id"),
            "potion_id": potion.get("id"),
            "target_combat_id": target_combat_id,
            "cost": cost,
            "damage": max(metric("damage"), metric("total_damage")),
            "block": max(metric("block"), metric("total_block")),
            "impact": impact,
            "roles": roles,
            "lethal": lethal,
            "profile": profile_keep,
        }

    def _dump_no_pressure_block_guard_eval(
        self,
        *,
        raw_obs: dict[str, Any],
        legal_actions: list[Any],
        legal_count: int,
        mask_np: Any,
        original_idx: int,
        encounter: str,
        current_energy: float,
        selected_block: float,
        threat_gap: float,
        current_hp: float,
        incoming: float,
        current_block: float,
        no_damage_pressure: bool,
        trivial_pressure: bool,
        pressure_attack_window: bool,
        selected_profile: dict[str, Any],
        candidates: list[Any],
        rejection_counts: dict[str, int],
        meaningful_block_urgent: bool = False,
        survival_justified: bool = False,
        mid_pressure_rewrite_attempt: bool = False,
    ) -> None:
        """Bounded exact-view dump for candidate/metric mismatch debugging."""

        try:
            dump_count = int(getattr(self, "_no_pressure_block_eval_dump_count", 0) or 0)
        except Exception:
            dump_count = 0
        if dump_count >= 1000:
            return
        try:
            path_getter = getattr(self, "_diagnostic_jsonl_path", None)
            if not callable(path_getter):
                return
            path = path_getter("no_pressure_block_eval_guard.jsonl")
            if path is None:
                return
            path.parent.mkdir(parents=True, exist_ok=True)
            count = min(max(0, int(legal_count)), len(legal_actions), 32)
            actions = [
                self._no_pressure_block_guard_action_summary(
                    idx=int(idx),
                    action=legal_actions[idx],
                    mask_np=mask_np,
                    raw_obs=raw_obs,
                    legal_actions=legal_actions,
                    current_energy=float(current_energy),
                )
                for idx in range(count)
            ]
            selected_summary = (
                actions[int(original_idx)]
                if 0 <= int(original_idx) < len(actions)
                else self._no_pressure_block_guard_action_summary(
                    idx=int(original_idx),
                    action=legal_actions[int(original_idx)] if 0 <= int(original_idx) < len(legal_actions) else None,
                    mask_np=mask_np,
                    raw_obs=raw_obs,
                    legal_actions=legal_actions,
                    current_energy=float(current_energy),
                )
            )
            selected_profile_keep = {
                str(k): self._no_pressure_block_guard_json_scalar(v)
                for k, v in (selected_profile or {}).items()
                if isinstance(v, (bool, int, float, str, np.generic)) or v is None
            }
            payload = {
                "time": time.time(),
                "kind": "no_pressure_block_eval",
                "episode_count": int(getattr(self, "episode_count", 0)),
                "total_steps": int(getattr(self, "total_steps", 0)),
                "episode_id": int(getattr(self, "episode_count", 0)),
                "global_step": int(getattr(self, "total_steps", 0)),
                "encounter": encounter,
                "original_idx": int(original_idx),
                "legal_count": int(legal_count),
                "energy": float(current_energy),
                "pressure": {
                    "selected_block": float(selected_block),
                    "threat_gap": float(threat_gap),
                    "current_hp": float(current_hp),
                    "incoming": float(incoming),
                    "current_block": float(current_block),
                    "no_damage_pressure": bool(no_damage_pressure),
                    "trivial_pressure": bool(trivial_pressure),
                    "pressure_attack_window": bool(pressure_attack_window),
                    "meaningful_block_urgent": bool(meaningful_block_urgent),
                    "survival_justified": bool(survival_justified),
                    "mid_pressure_rewrite_attempt": bool(mid_pressure_rewrite_attempt),
                },
                "selected": selected_summary,
                "selected_profile": selected_profile_keep,
                "candidates": [
                    {
                        "idx": int(getattr(candidate, "index", -1)),
                        "title": str(getattr(candidate, "title", "")),
                        "damage": float(getattr(candidate, "damage", 0.0)),
                        "impact": float(getattr(candidate, "impact", 0.0)),
                        "cost": float(getattr(candidate, "cost", 0.0)),
                        "lethal": bool(getattr(candidate, "lethal", False)),
                    }
                    for candidate in list(candidates or [])[:12]
                ],
                "rejection_counts": dict(rejection_counts or {}),
                "actions": actions,
            }
            with path.open("a", encoding="utf-8") as fh:
                fh.write(json.dumps(payload, ensure_ascii=False, default=str) + "\n")
            try:
                setattr(self, "_no_pressure_block_eval_dump_count", dump_count + 1)
            except Exception:
                pass
        except Exception:
            return

    def _apply_no_pressure_block_guard(
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
        """Rewrite no/trivial-pressure pure block to a safe progress card."""

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
        # Keep the guard's "is the selected action a bad pure-block candidate?"
        # gate byte-for-byte aligned with the selected-side trainer metric.  The
        # richer classifier above may mark some cards as mechanism-urgent (for
        # example Kaiser-facing/back-attack contexts), which intentionally turns
        # pure_block off for tactical reasoning.  The TensorBoard quality metric
        # uses the plain block-waste profile instead; if this guard gates on the
        # richer classifier it can skip a Defend-like card while the selected-side
        # metric still records ``bad_pure_block_selected=1``.  Use the same plain
        # profile here, then let the later survival/pressure checks decide whether
        # rewriting is actually safe.
        selected_block_profile = self._card_block_waste_profile(selected, raw_obs)
        if not (
            bool(selected_block_profile.get("block_waste", False))
            or bool(selected_block_profile.get("pure_block", False))
        ):
            search_stats["combat_quality_no_pressure_block_guard_profile_skip"] = 1.0
            return original_idx

        incoming, current_block, current_hp = self._incoming_damage_pressure(raw_obs)
        threat_gap = max(0.0, float(incoming) - float(current_block))
        encounter_tier = self._combat_encounter_tier_from_raw(raw_obs)
        selected_block = max(
            self._action_metric(selected, "block"),
            self._action_metric(selected, "total_block"),
            self._action_numeric_value(selected, ("block", "total_block", "preview_block", "expected_block")),
            0.0,
        )
        pressure_attack_window = self._no_pressure_block_guard_low_value_pressure_window(
            selected_block=selected_block,
            threat_gap=threat_gap,
            current_hp=current_hp,
            encounter_tier=encounter_tier,
        )

        meaningful_block_urgent = bool(
            self._is_meaningful_block_urgent(
                block=selected_block,
                threat_gap=threat_gap,
                current_hp=current_hp,
                incoming=incoming,
                encounter_tier=encounter_tier,
            )
        )
        survival_justified = bool(meaningful_block_urgent and not pressure_attack_window)

        # Keep this guard's "should I try to rewrite?" gate aligned with the
        # selected-side bad-pure-block metric in ``trainer_quality``.  The old
        # implementation had a second, narrower pressure gate below and
        # therefore missed a large middle band:
        #
        #   not urgent enough to be survival-justified,
        #   but not tiny/no-damage enough to be tagged no/trivial/low-value.
        #
        # TensorBoard then correctly reported ``bad_pure_block_selected=1`` for
        # the post-guard action while this guard had returned early without
        # even collecting the same safe-progress candidates.  From here onward
        # survival justification is the only pressure skip; the shared
        # candidate filter still enforces legality, affordability, HP-cost,
        # X-cost, follow-up, and immediate-progress safety.
        if survival_justified:
            search_stats["combat_quality_no_pressure_block_guard_pressure_skip"] = 1.0
            search_stats["combat_quality_no_pressure_block_guard_survival_justified"] = 1.0
            return original_idx

        no_damage_pressure = bool(
            selected_profile.get("card_no_damage_pressure_context", False)
            or selected_profile.get("card_no_damage_pressure", False)
            or threat_gap <= 0.05
        )
        trivial_threshold = self._no_pressure_block_guard_trivial_threshold(current_hp)
        trivial_pressure = bool(
            not no_damage_pressure
            and threat_gap > 0.05
            and trivial_threshold > 0.0
            and threat_gap <= trivial_threshold
        )
        mid_pressure_rewrite_attempt = bool(
            not no_damage_pressure and not trivial_pressure and not pressure_attack_window
        )

        search_stats["combat_quality_no_pressure_block_guard_available"] = 1.0
        if trivial_pressure:
            search_stats["combat_quality_no_pressure_block_guard_trivial_pressure"] = 1.0
        if pressure_attack_window:
            search_stats["combat_quality_no_pressure_block_guard_pressure_attack_window"] = 1.0
            search_stats["combat_quality_no_pressure_block_guard_low_value_pressure"] = 1.0
        if mid_pressure_rewrite_attempt:
            search_stats["combat_quality_no_pressure_block_guard_mid_pressure_attempt"] = 1.0

        # The old pressure-only early-return lived here:
        #
        #   if not (no_damage_pressure or trivial_pressure or pressure_attack_window):
        #       return original_idx
        #
        # Do not reintroduce it unless the selected-side bad-pure-block metric
        # is changed at the same time; otherwise training will again execute
        # Defend-like cards that the quality gate marks as actionable failures.

        candidates, rejection_counts = collect_safe_progress_candidates(
            self,
            selected_idx=original_idx,
            legal_count=int(legal_count),
            legal_actions=legal_actions,
            mask_np=mask_np,
            raw_obs=raw_obs,
            current_energy=current_energy,
            use_mask=True,
            include_debug=True,
        )
        self._dump_no_pressure_block_guard_eval(
            raw_obs=raw_obs,
            legal_actions=legal_actions,
            legal_count=int(legal_count),
            mask_np=mask_np,
            original_idx=original_idx,
            encounter=encounter,
            current_energy=float(current_energy),
            selected_block=float(selected_block),
            threat_gap=float(threat_gap),
            current_hp=float(current_hp),
            incoming=float(incoming),
            current_block=float(current_block),
            no_damage_pressure=bool(no_damage_pressure),
            trivial_pressure=bool(trivial_pressure),
            pressure_attack_window=bool(pressure_attack_window),
            selected_profile=selected_profile if isinstance(selected_profile, dict) else {},
            candidates=candidates,
            rejection_counts=rejection_counts,
            meaningful_block_urgent=bool(meaningful_block_urgent),
            survival_justified=bool(survival_justified),
            mid_pressure_rewrite_attempt=bool(mid_pressure_rewrite_attempt),
        )

        search_stats["combat_quality_no_pressure_block_guard_candidate_count"] = float(len(candidates))
        if not candidates:
            search_stats["combat_quality_no_pressure_block_guard_no_alternative"] = 1.0
            for reason, count in rejection_counts.items():
                if reason == "accepted":
                    continue
                search_stats[f"combat_quality_no_pressure_block_guard_reject_{reason}"] = float(count)
            try:
                dump_count = int(getattr(self, "_no_pressure_block_no_candidate_dump_count", 0) or 0)
            except Exception:
                dump_count = 0
            if dump_count < 500:
                self._dump_combat_hard_guard_record(
                    kind="no_pressure_block_no_candidate",
                    raw_obs=raw_obs,
                    legal_actions=legal_actions,
                    original_idx=original_idx,
                    override_idx=original_idx,
                    risk=float(threat_gap),
                    countdown=None,
                    encounter=encounter,
                    lethal_exemption=False,
                    extra={
                        "rejection_counts": rejection_counts,
                        "selected_block": float(selected_block),
                        "threat_gap": float(threat_gap),
                        "current_hp": float(current_hp),
                        "incoming": float(incoming),
                        "current_block": float(current_block),
                        "no_damage_pressure": bool(no_damage_pressure),
                        "trivial_pressure": bool(trivial_pressure),
                        "pressure_attack_window": bool(pressure_attack_window),
                        "meaningful_block_urgent": bool(meaningful_block_urgent),
                        "survival_justified": bool(survival_justified),
                        "mid_pressure_rewrite_attempt": bool(mid_pressure_rewrite_attempt),
                    },
                )
                try:
                    setattr(self, "_no_pressure_block_no_candidate_dump_count", dump_count + 1)
                except Exception:
                    pass
            return original_idx

        best = candidates[0]
        best_idx = int(best.index)
        best_damage = float(best.damage)
        best_impact = float(best.impact)
        best_lethal = bool(best.lethal)
        if best_lethal:
            search_stats["combat_quality_no_pressure_block_guard_lethal_candidate"] = 1.0

        if int(best_idx) != original_idx:
            self._dump_combat_hard_guard_record(
                kind="no_pressure_block",
                raw_obs=raw_obs,
                legal_actions=legal_actions,
                original_idx=original_idx,
                override_idx=int(best_idx),
                risk=float(max(best_damage, best_impact)),
                countdown=None,
                encounter=encounter,
                lethal_exemption=False,
            )
            search_stats["combat_quality_no_pressure_block_guard_applied"] = 1.0
            search_stats["combat_quality_no_pressure_block_guard_override"] = 1.0
            # Downstream survival guards run after this guard.  Live sandbox
            # traces showed a ping-pong pattern:
            #
            #   Defend under safe/low-value pressure
            #     -> no_pressure_block rewrites to Strike/progress
            #     -> survival_non_endturn rewrites back to Defend
            #
            # Mark the post-override progress action so later guards can
            # preserve it unless the current turn is truly critical/near
            # lethal.  Keep the explicit lock bit separate from the index so
            # action 0 remains representable.
            search_stats["combat_quality_no_pressure_block_guard_progress_override_idx"] = float(best_idx)
            search_stats["combat_quality_no_pressure_block_guard_progress_override_lock"] = 1.0
            search_stats["combat_quality_hard_guard_override_any"] = 1.0
            self._no_pressure_block_guard_reset_block_stats(search_stats)
            return int(best_idx)

        return original_idx
