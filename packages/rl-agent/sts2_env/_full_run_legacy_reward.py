# ruff: noqa: RUF002, RUF003
"""Deprecated-v1 full-run reward shaping and progress diagnostics.

Canonical v2 backends own reward through ``EnvironmentRuntimeMixin``; this
mixin remains an isolated compatibility implementation.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from ._full_run_values import (
    DISCARD_POTION_ACTION_KIND,
    EMPTY_POTION_NAMES,
    _float,
)
from .end_turn_quality import strict_end_turn_waste_context
from .potion_timing import compute_potion_timing
from .reward_constants import (
    BOSS_ACT_FLOORS,
    BOSS_COMBAT_LOSS_DAMAGE_UNDO_PERCENT_SCALE,
    BOSS_COMBAT_LOSS_PENALTY_BASE,
    BOSS_COMBAT_LOSS_PENALTY_MISSING_HP_SCALE,
    BOSS_ENEMY_HP_DELTA_PERCENT_SCALE,
    BOSS_FLOOR_ENTRY_BONUS,
    ENEMY_HP_DELTA_REWARD_MAX_ABS,
    ENEMY_HP_DELTA_REWARD_SCALE,
    FLOOR_CLEAR_BONUS_PER_FLOOR,
    FLOOR_CLEAR_MIN_FLOOR,
    FULL_RUN_DEATH_PENALTY_BASE,
    FULL_RUN_DEATH_PENALTY_LATE_ACT_SCALE,
    FULL_RUN_DEATH_PENALTY_MISSING_HP_SCALE,
    PLAYER_HP_LOSS_REWARD_SCALE,
    POTION_HOARDING_MAX_PENALTY_ABS,
    POTION_HOARDING_PENALTY_PER_POTION,
    POTION_TIMING_QUALITY_SCALE,
    POTION_TIMING_WASTE_SCALE,
    POTION_USE_BOSS_BONUS,
    POTION_USE_ELITE_BONUS,
    POTION_USE_MONSTER_BONUS,
    POTION_USE_MONSTER_PENALTY,
    REST_SITE_SKIP_HEAL_HP_THRESHOLD,
    REST_SITE_SKIP_HEAL_PENALTY,
)
from .reward_constants import (
    FULL_RUN_WASTE_BASE as END_TURN_WASTE_BASE_PENALTY,
)
from .reward_constants import (
    FULL_RUN_WASTE_ENERGY as END_TURN_WASTE_ENERGY_PENALTY,
)
from .reward_constants import (
    FULL_RUN_WASTE_EXTRA_ACTION as END_TURN_WASTE_EXTRA_ACTION_PENALTY,
)
from .reward_constants import (
    FULL_RUN_WASTE_ZERO_COST as END_TURN_WASTE_ZERO_COST_BONUS_PENALTY,
)


class LegacyFullRunRewardMixin:
    def _enemy_hp_delta_reward(self, before_obs: dict[str, Any] | None, after_obs: dict[str, Any] | None) -> float:
        before_total = self._combat_enemy_total_hp(before_obs)
        after_total = self._combat_enemy_total_hp(after_obs)
        if before_total <= 0.0 and after_total <= 0.0:
            return 0.0

        # Bridge empties `combat.enemies` the moment a combat ends (both
        # death transitions and victory transitions). If the player died
        # with enemies still alive, crediting (before_total - 0) emits a
        # false "+5.66 kill reward" on the defeat step. Guard: skip the
        # delta when after-state has empty enemies AND player is dead.
        # Full-run episode doesn't usually terminate on player death (run
        # ends), but combat-end transitions still drop enemies to [].
        after_player_hp = 0.0
        if isinstance(after_obs, dict):
            player = after_obs.get("player") if isinstance(after_obs.get("player"), dict) else {}
            after_player_hp = _float((player or {}).get("hp"))
        # Both "enemies key missing (sim)" and "enemies=[] (live)" reduce
        # to after_total==0 through _combat_enemy_total_hp. On player-death
        # transitions we see that AND player_hp<=0; skip the delta.
        if (
            before_total > 0.0
            and after_total <= 0.0
            and after_player_hp <= 0.0
        ):
            return 0.0

        raw = (before_total - after_total) * ENEMY_HP_DELTA_REWARD_SCALE
        if raw > ENEMY_HP_DELTA_REWARD_MAX_ABS:
            return ENEMY_HP_DELTA_REWARD_MAX_ABS
        if raw < -ENEMY_HP_DELTA_REWARD_MAX_ABS:
            return -ENEMY_HP_DELTA_REWARD_MAX_ABS
        return raw

    def _is_boss_encounter(self, obs: dict[str, Any] | None) -> bool:
        """True when the observation represents a boss-room combat.

        Primary signal is ``run.state_type == "boss"`` (sim emits it, real
        bridge emits a compatible ``room_type``). Falls back to the
        canonical act-boss floor list (17/34/51) for environments that
        don't populate the string tag. Having two independent signals
        keeps the bonus from over-firing on mis-tagged rooms — both
        must at least not contradict the boss-ness judgment.
        """
        if not isinstance(obs, dict):
            return False
        run = obs.get("run") if isinstance(obs.get("run"), dict) else {}
        state_type = str(run.get("state_type") or run.get("room_type") or "").strip().lower()
        if state_type == "boss":
            return True
        floor_val = run.get("floor")
        try:
            floor = int(floor_val) if floor_val is not None else 0
        except (TypeError, ValueError):
            floor = 0
        if floor in BOSS_ACT_FLOORS:
            # Only treat as boss if we're actually IN combat (avoids
            # awarding bonus for walking onto the boss tile without the
            # encounter starting yet).
            combat = obs.get("combat") if isinstance(obs.get("combat"), dict) else None
            if isinstance(combat, dict) and combat.get("enemies"):
                return True
        return False

    def _encounter_tier_from_obs(self, obs: dict[str, Any] | None) -> str:
        """Best-effort encounter tier for shared potion-timing shaping.

        Full-run EnvV2 sees build / route / combat screens, while the
        shared potion evaluator only needs a coarse combat tier to decide
        whether a potion is worth saving.  Prefer the bridge/sim room tag
        when present, fall back to the stricter boss helper so act-boss
        floor snapshots without a string tag still get boss weighting.
        """
        if self._is_boss_encounter(obs):
            return "boss"
        if not isinstance(obs, dict):
            return "normal"
        run = obs.get("run") if isinstance(obs.get("run"), dict) else {}
        state_type = str(run.get("state_type") or run.get("room_type") or "").strip().lower()
        if state_type in {"elite", "miniboss"}:
            return "elite"
        if state_type == "weak":
            return "weak"
        return "normal"

    def _boss_damage_bonus_reward(
        self,
        before_obs: dict[str, Any] | None,
        after_obs: dict[str, Any] | None,
    ) -> float:
        """Additive bonus on damage dealt during boss encounters.

        Full-run previously used raw HP × multiplier here, so a high-HP boss
        could hand out far more positive shaping than the terminal loss could
        undo.  Mirror CombatSandbox's percent-based shaping: cumulative boss
        damage bonus is bounded by ``BOSS_ENEMY_HP_DELTA_PERCENT_SCALE`` for a
        full kill, independent of absolute boss HP.  Sign-asymmetric: only
        POSITIVE damage (enemy losing HP) is rewarded.
        """
        if not self._is_boss_encounter(before_obs) and not self._is_boss_encounter(after_obs):
            return 0.0
        # Count any step where one side of the transition was a boss
        # encounter, even if no damage this tick (captures block / setup
        # turns so we can see "boss engagement density" not just dmg).
        self._episode_telemetry["boss_encounter_steps"] += 1.0
        before_total = self._combat_enemy_total_hp(before_obs)
        after_total = self._combat_enemy_total_hp(after_obs)
        if before_total <= 0.0:
            return 0.0
        # Same bridge contract edge case as _enemy_hp_delta_reward():
        # combat.enemies is cleared to [] on both victory and defeat.  On a
        # defeat step with enemies still alive, interpreting [] as 0 HP would
        # award a full boss-damage bonus (often 200+ raw HP), teaching the
        # policy that dying on boss is equivalent to killing it.  If the
        # after-state has no enemy HP AND the player is dead, count the guard
        # for diagnostics but emit no reward / no boss_damage_dealt_raw.
        after_player_hp = 0.0
        if isinstance(after_obs, dict):
            player = after_obs.get("player") if isinstance(after_obs.get("player"), dict) else {}
            after_player_hp = _float((player or {}).get("hp"))
        if before_total > 0.0 and after_total <= 0.0 and after_player_hp <= 0.0:
            self._episode_telemetry["boss_damage_death_clear_guarded"] += 1.0
            # When the player dies, the bridge can clear the enemy list in the
            # same post-action snapshot.  Treating ``before_total -> 0`` as
            # damage would grant false boss-kill credit and pollute Act1
            # diagnostics.  Keep the legacy raw-damage field at zero and expose
            # the suppressed remaining HP under an explicitly named metric.
            self._episode_telemetry["boss_damage_death_clear_guarded_remaining_hp_raw"] += float(
                before_total
            )
            self._episode_telemetry["boss_damage_death_clear_guarded_raw"] += 0.0
            return 0.0
        raw_damage = max(before_total - after_total, 0.0)
        if raw_damage <= 0.0:
            return 0.0
        damage_dealt_before = max(_float(self._episode_telemetry.get("boss_damage_dealt_raw")), 0.0)
        self._episode_telemetry["boss_damage_dealt_raw"] += float(raw_damage)
        boss_hp_estimate = max(damage_dealt_before + before_total, before_total, raw_damage)
        damage_ratio = (
            float(np.clip(raw_damage / boss_hp_estimate, 0.0, 1.0))
            if boss_hp_estimate > 0.0
            else 0.0
        )
        bonus = damage_ratio * float(BOSS_ENEMY_HP_DELTA_PERCENT_SCALE)
        self._episode_telemetry["boss_damage_bonus_total"] += float(bonus)
        return bonus

    def _boss_death_terminal_penalty(
        self,
        before_obs: dict[str, Any] | None,
        after_obs: dict[str, Any] | None,
        *,
        terminated: bool,
        truncated: bool,
    ) -> float:
        """One-shot full-run penalty for dying in a boss fight.

        Dense boss damage shaping made "reach boss, deal damage, die" highly
        positive in full-run training.  CombatSandbox already has boss terminal
        loss shaping, but EnvV2 did not.  Apply a terminal-only loss penalty
        here so Act1 boss death is clearly worse than living into Act2.
        """
        if not terminated or truncated:
            return 0.0
        if not self._is_boss_encounter(before_obs) and not self._is_boss_encounter(after_obs):
            return 0.0
        player = after_obs.get("player") if isinstance(after_obs, dict) and isinstance(after_obs.get("player"), dict) else {}
        hp = _float((player or {}).get("hp"))
        max_hp = max(_float((player or {}).get("max_hp"), 1.0), 1.0)
        if hp > 0.0:
            return 0.0

        missing_hp_ratio = float(np.clip((max_hp - max(hp, 0.0)) / max_hp, 0.0, 1.0))
        damage_dealt = max(_float(self._episode_telemetry.get("boss_damage_dealt_raw")), 0.0)
        remaining_guarded = max(
            _float(self._episode_telemetry.get("boss_damage_death_clear_guarded_remaining_hp_raw")),
            0.0,
        )
        # If the final bridge snapshot did not trigger the death-clear guard,
        # fall back to the pre-step enemy HP as remaining boss HP.
        if remaining_guarded <= 0.0:
            remaining_guarded = max(self._combat_enemy_total_hp(before_obs), 0.0)
        boss_hp_estimate = max(damage_dealt + remaining_guarded, damage_dealt, remaining_guarded)
        damage_ratio = (
            float(np.clip(damage_dealt / boss_hp_estimate, 0.0, 1.0))
            if boss_hp_estimate > 0.0
            else 0.0
        )

        magnitude = (
            float(BOSS_COMBAT_LOSS_PENALTY_BASE)
            + float(BOSS_COMBAT_LOSS_PENALTY_MISSING_HP_SCALE) * missing_hp_ratio
            + float(BOSS_COMBAT_LOSS_DAMAGE_UNDO_PERCENT_SCALE) * damage_ratio
        )
        penalty = -magnitude
        self._episode_telemetry["boss_death_terminal_penalty_events"] += 1.0
        self._episode_telemetry["boss_death_terminal_penalty_total"] += float(penalty)
        self._episode_telemetry["boss_death_terminal_missing_hp_ratio"] = float(missing_hp_ratio)
        self._episode_telemetry["boss_death_terminal_damage_ratio"] = float(damage_ratio)
        return penalty

    def _rest_site_skip_heal_penalty(
        self,
        before_obs: dict[str, Any] | None,
        action: dict[str, Any] | None,
    ) -> float:
        """Penalize picking a non-HEAL rest-site option when low on HP.

        Magnitude stays strictly below FLOOR_CLEAR_BONUS_PER_FLOOR so
        the policy can never prefer "skip the campfire tile entirely
        on the map" over "take the campfire tile and pick SMITH". The
        within-campfire contrast between HEAL and non-HEAL is what we
        want biased, not the campfire-vs-monster decision at map time.

        Activates only when:
          1. action kind is a rest-site choice
          2. the chosen option isn't HEAL/REST
          3. player HP ratio (pre-step) is below the threshold
        """
        if not isinstance(action, dict):
            return 0.0
        if not self._is_rest_site_choice_action(action):
            return 0.0
        smith_picked = self._is_rest_smith_choice_action(action)
        heal_picked = (not smith_picked) and self._is_rest_heal_choice_action(action)
        # Telemetry: always count rest-site encounters + the HEAL/non-HEAL
        # split, regardless of HP threshold. Helps diagnose "is the
        # policy even reaching campfires" vs "is it choosing correctly".
        self._episode_telemetry["rest_site_encounters"] += 1.0
        if heal_picked:
            self._episode_telemetry["rest_heal_chosen"] += 1.0
            return 0.0
        if smith_picked:
            self._episode_telemetry["rest_smith_chosen"] += 1.0
        self._episode_telemetry["rest_skip_heal_chosen"] += 1.0

        hp_ratio, hp_valid = self._hp_ratio_from_obs(before_obs)
        if not hp_valid:
            return 0.0
        if hp_ratio >= REST_SITE_SKIP_HEAL_HP_THRESHOLD:
            return 0.0
        penalty = float(REST_SITE_SKIP_HEAL_PENALTY)
        self._episode_telemetry["rest_skip_heal_at_low_hp"] += 1.0
        self._episode_telemetry["rest_penalty_total"] += penalty
        return penalty

    def _potion_use_bonus(
        self,
        before_obs: dict[str, Any] | None,
        action: dict[str, Any] | None,
    ) -> float:
        """Encourage use_potion actions, with context-dependent scaling.

        STS potions are single-use combat consumables that, by the
        800k baseline, the policy had learned to hoard almost
        indefinitely. Base hp/damage rewards didn't distinguish them
        enough from card plays to overcome the implicit "save it for
        later" bias. This flat bonus lifts the expected value of
        use_potion slightly above an equivalent-damage card play, and
        the boss/elite multipliers concentrate the bias where potions
        actually matter.

        Only fires for ``use_potion`` — ``discard_potion`` gets
        nothing (discarding is itself a waste signal).
        """
        if not isinstance(action, dict):
            return 0.0
        kind = str(action.get("kind") or "").lower()
        if kind == "discard_potion":
            self._episode_telemetry["potion_discard_count"] += 1.0
            return 0.0
        if kind != "use_potion":
            return 0.0
        self._episode_telemetry["potion_use_count"] += 1.0
        # Encounter-scoped absolute bonuses. Unlike the earlier
        # base×multiplier scheme, non-boss / non-elite use gets
        # MONSTER_BONUS (default 0) or MONSTER_PENALTY (default 0)
        # — there's no longer an unconditional positive gradient for
        # use_potion in ordinary combat. First-run telemetry showed
        # the previous base bonus caused the policy to burn all
        # potions on floor 3-7 monsters before reaching the boss.
        if self._is_boss_encounter(before_obs):
            bonus = float(POTION_USE_BOSS_BONUS)
            self._episode_telemetry["potion_use_boss_count"] += 1.0
            self._episode_telemetry["potion_use_bonus_total"] += bonus
            return bonus
        run = before_obs.get("run") if isinstance(before_obs, dict) and isinstance(before_obs.get("run"), dict) else {}
        state_type = str(run.get("state_type") or run.get("room_type") or "").strip().lower()
        if state_type in {"elite", "miniboss"}:
            bonus = float(POTION_USE_ELITE_BONUS)
            self._episode_telemetry["potion_use_elite_count"] += 1.0
            self._episode_telemetry["potion_use_bonus_total"] += bonus
            return bonus
        # Monster-fight use — net signal depends on whether the
        # optional penalty is configured. Default config gives 0.
        monster_signal = float(POTION_USE_MONSTER_BONUS) + float(POTION_USE_MONSTER_PENALTY)
        self._episode_telemetry["potion_use_bonus_total"] += monster_signal
        return monster_signal

    def _potion_timing_step_reward(
        self,
        action: dict[str, Any] | None,
        before_obs: dict[str, Any] | None,
        legal_actions: list[dict[str, Any]] | None,
    ) -> float:
        """Reward good potion timing and penalize waste in full-run EnvV2.

        `combat_env.py` already uses the shared `compute_potion_timing`
        evaluator; the full-run path was only counting potion use, so
        value targets could not distinguish "Blood Potion saves a lethal
        turn" from "energy potion with no follow-up on a safe hallway".
        This wiring gives the full-run replay the same dense timing signal
        without adding a hard block that could forbid legitimate saves.
        """
        if not isinstance(action, dict):
            return 0.0
        kind = str(action.get("kind") or "").lower()
        action_id = str(action.get("action_id") or "").lower()
        if kind not in {"use_potion", "potion"} and not action_id.startswith("use_potion:"):
            return 0.0
        if not isinstance(before_obs, dict):
            return 0.0

        player = before_obs.get("player") if isinstance(before_obs.get("player"), dict) else {}
        combat = before_obs.get("combat") if isinstance(before_obs.get("combat"), dict) else {}
        energy = _float(player.get("energy", combat.get("energy", 0.0)))

        try:
            profile = compute_potion_timing(
                action,
                before_obs,
                legal_actions or [],
                None,
                energy,
                encounter_tier=self._encounter_tier_from_obs(before_obs),
            )
        except Exception:
            # Reward shaping must never turn an otherwise valid bridge
            # transition into a training crash.  The planner still has its
            # own timing metrics; this path is best-effort value shaping.
            return 0.0
        if not profile.get("is_potion"):
            return 0.0

        use_quality = _float(profile.get("use_quality"))
        waste_risk = _float(profile.get("waste_risk"))
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
        # Do not let the 0.08 baseline / weakly-positive profile reinforce
        # burning potions in hallway fights when the only alternative is End
        # Turn.  Only urgent or clearly high-quality use should be positive;
        # explicitly bad timing becomes negative even if use_quality was
        # clipped just above zero.
        if urgent or (use_quality >= 0.45 and not bad_timing):
            reward += float(POTION_TIMING_QUALITY_SCALE) * use_quality
            self._episode_telemetry["potion_timing_quality_events"] += 1.0
        elif bad_timing:
            effective_waste = max(float(waste_risk), 0.35)
            reward -= float(POTION_TIMING_WASTE_SCALE) * effective_waste
            self._episode_telemetry["potion_timing_waste_events"] += 1.0
        elif waste_risk > 0.0:
            reward -= float(POTION_TIMING_WASTE_SCALE) * waste_risk
            self._episode_telemetry["potion_timing_waste_events"] += 1.0
        self._episode_telemetry["potion_timing_reward_total"] += float(reward)
        return float(reward)

    @staticmethod
    def _count_nonempty_potions(obs: dict[str, Any] | None) -> int:
        """How many real potions are currently in the inventory.

        STS2 represents empty potion slots as the string "[empty]" (or
        a dict with that title). Only count actual potions. Returns 0
        on malformed obs.
        """
        if not isinstance(obs, dict):
            return 0
        player = obs.get("player") if isinstance(obs.get("player"), dict) else None
        if not isinstance(player, dict):
            return 0
        potions = player.get("potions")
        if not isinstance(potions, list):
            return 0
        count = 0
        for potion in potions:
            if isinstance(potion, str):
                s = potion.strip()
                if s.lower() not in EMPTY_POTION_NAMES:
                    count += 1
            elif isinstance(potion, dict):
                if bool(potion.get("empty")):
                    continue
                title = str(potion.get("title") or potion.get("id") or "").strip()
                if title.lower() not in EMPTY_POTION_NAMES:
                    count += 1
        return count

    @staticmethod
    def _count_empty_potion_slots(obs: dict[str, Any] | None) -> int:
        """How many potion slots are observably empty.

        Return 0 for missing/malformed payloads so the filter fails open and
        does not hide a real overflow modal. If the bridge explicitly reports
        any empty slot, a singleton ``discard_potion`` frontier is not a true
        overflow requirement and should be treated as transient/fake.
        """
        if not isinstance(obs, dict):
            return 0
        player = obs.get("player") if isinstance(obs.get("player"), dict) else None
        if not isinstance(player, dict):
            return 0
        potions = player.get("potions")
        if not isinstance(potions, list):
            return 0

        count = 0
        for potion in potions:
            if isinstance(potion, str):
                if potion.strip().lower() in EMPTY_POTION_NAMES:
                    count += 1
            elif isinstance(potion, dict):
                if bool(potion.get("empty")):
                    count += 1
                    continue
                title = str(
                    potion.get("title")
                    or potion.get("id")
                    or potion.get("name")
                    or ""
                ).strip()
                if title.lower() in EMPTY_POTION_NAMES:
                    count += 1
        return int(count)

    @staticmethod
    def _potion_slots_dump(obs: dict[str, Any] | None) -> list[dict[str, Any]]:
        """Compact potion-slot dump for full-run diagnostics.

        Full-run EnvV2 historically only surfaced aggregate
        ``potion_discard_count`` / ``potion_use_count``.  That made it
        impossible to tell whether the 4-6 discards per Act1 attempt were
        forced overflow cleanup, optional policy waste, or stale bridge slot
        state.  Keep this payload small and JSON-friendly so MuZeroTrainer can
        append it directly to diagnostics/potion_transitions.jsonl.
        """
        if not isinstance(obs, dict):
            return []
        player = obs.get("player") if isinstance(obs.get("player"), dict) else {}
        potions = player.get("potions") if isinstance(player, dict) else None
        if not isinstance(potions, list):
            return []

        out: list[dict[str, Any]] = []
        for idx, potion in enumerate(potions):
            slot = idx
            potion_id: str | None = None
            title: str | None = None
            empty = False
            is_usable: Any = None
            is_queued: Any = None
            raw_kind = type(potion).__name__
            if isinstance(potion, dict):
                try:
                    slot = int(potion.get("slot", potion.get("slot_index", idx)))
                except (TypeError, ValueError):
                    slot = idx
                potion_id = str(
                    potion.get("id")
                    or potion.get("potion_id")
                    or potion.get("internal_id")
                    or ""
                ).strip() or None
                title = str(
                    potion.get("title")
                    or potion.get("name")
                    or potion.get("label")
                    or potion_id
                    or ""
                ).strip() or None
                empty = bool(potion.get("empty"))
                is_usable = potion.get("is_usable", potion.get("usable"))
                is_queued = potion.get("is_queued", potion.get("queued"))
            elif potion is None:
                empty = True
            else:
                title = str(potion).strip() or None
                potion_id = title

            marker = str(title or potion_id or "").strip().lower()
            if marker in EMPTY_POTION_NAMES:
                empty = True
            out.append(
                {
                    "slot": int(slot),
                    "id": potion_id,
                    "title": title,
                    "empty": bool(empty),
                    "is_usable": is_usable,
                    "is_queued": is_queued,
                    "raw_kind": raw_kind,
                }
            )
        return out

    @staticmethod
    def _state_version_from_obs(obs: dict[str, Any] | None) -> int:
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

    @staticmethod
    def _potion_action_slot(action: dict[str, Any] | None) -> int:
        if not isinstance(action, dict):
            return -1
        def _bounded_int(value: Any) -> int | None:
            try:
                if value is None or str(value).strip() == "":
                    return None
                slot = int(value)
                if 0 <= slot < 10:
                    return slot
            except (TypeError, ValueError):
                return None
            return None

        sources: list[dict[str, Any]] = [action]
        for key in ("target", "potion", "payload"):
            value = action.get(key)
            if isinstance(value, dict):
                sources.append(value)
        for source in sources:
            for key in ("potion_slot", "slot", "slot_index", "potion_index", "potion_idx", "index"):
                if key not in source:
                    continue
                slot = _bounded_int(source.get(key))
                if slot is not None:
                    return slot
        action_id = str(action.get("action_id") or "").strip().lower()
        if action_id.startswith("use_potion:") or action_id.startswith("discard_potion:"):
            parts = action_id.split(":")[1:]
            leading_numbers: list[int] = []
            for part in parts:
                if not part.isdigit():
                    break
                try:
                    leading_numbers.append(int(part))
                except (TypeError, ValueError):
                    break
            # Live bridge ids are {kind}:{playerIndex}:{slotIndex}[:target].
            # Legacy tests/logs also use {kind}:{slotIndex}.  Prefer slotIndex
            # when both player and slot are present so diagnostics point to the
            # actual potion changed instead of always player 0.
            if len(leading_numbers) >= 2:
                slot = _bounded_int(leading_numbers[1])
                if slot is not None:
                    return slot
            if len(leading_numbers) == 1:
                slot = _bounded_int(leading_numbers[0])
                if slot is not None:
                    return slot
        return -1

    def _build_potion_transition_record(
        self,
        *,
        action: dict[str, Any] | None,
        prev_obs: dict[str, Any] | None,
        after_obs: dict[str, Any] | None,
        legal_actions_before: list[dict[str, Any]] | None,
        bridge_info: dict[str, Any] | None,
        reward: float,
        terminated: bool,
        truncated: bool,
    ) -> dict[str, Any] | None:
        """Emit full-run use/discard potion diagnostics for MuZeroTrainer.

        combat_env already emits use-potion transitions, but the full-run
        EnvV2 path did not.  Act1 recovery currently observes many
        ``discard_potion`` selections and no per-action explanation; this
        record makes the selected action, inventory before/after, and
        "forced singleton discard" status visible without changing policy.
        """
        if not isinstance(action, dict):
            return None
        kind = str(action.get("kind") or "").strip().lower()
        action_id = str(action.get("action_id") or "").strip()
        action_id_lower = action_id.lower()
        if kind not in {"use_potion", "potion", "discard_potion"} and not (
            action_id_lower.startswith("use_potion:") or action_id_lower.startswith("discard_potion:")
        ):
            return None

        before_dump = self._potion_slots_dump(prev_obs)
        after_dump = self._potion_slots_dump(after_obs)
        before_count = sum(1 for slot in before_dump if not bool(slot.get("empty")))
        after_count = sum(1 for slot in after_dump if not bool(slot.get("empty")))
        slot_index = self._potion_action_slot(action)
        before_slot = next((slot for slot in before_dump if int(slot.get("slot", -1)) == slot_index), None)
        after_slot = next((slot for slot in after_dump if int(slot.get("slot", -1)) == slot_index), None)

        legal_before = legal_actions_before if isinstance(legal_actions_before, list) else []
        unblocked_before = [
            candidate
            for candidate in legal_before
            if isinstance(candidate, dict) and not self._state_action_is_blocked(candidate, legal_before)
        ]
        non_discard_unblocked = [
            candidate
            for candidate in unblocked_before
            if str(candidate.get("kind") or "").strip().lower() != DISCARD_POTION_ACTION_KIND
        ]

        before_run = prev_obs.get("run") if isinstance(prev_obs, dict) and isinstance(prev_obs.get("run"), dict) else {}
        after_run = after_obs.get("run") if isinstance(after_obs, dict) and isinstance(after_obs.get("run"), dict) else {}
        run = before_run if before_run else after_run
        room_model = (
            run.get("room_model")
            or run.get("encounter_id")
            or run.get("encounter")
            or run.get("room_encounter")
            or run.get("room")
            or ""
        )
        info = bridge_info if isinstance(bridge_info, dict) else {}
        execute_ok = not bool(info.get("error") or info.get("action_error"))

        return {
            "event": "discard_potion_transition" if kind == DISCARD_POTION_ACTION_KIND or action_id_lower.startswith("discard_potion:") else "use_potion_transition",
            "episode_id": self._episode_id,
            "action_id": action_id,
            "kind": kind,
            "potion_slot": int(slot_index),
            "potion_id_before": (before_slot or {}).get("id"),
            "potion_title_before": (before_slot or {}).get("title"),
            "potion_slot_before": before_slot,
            "potion_slot_after": after_slot,
            "potion_slots_before": before_dump,
            "potion_slots_after": after_dump,
            "potion_count_before": int(before_count),
            "potion_count_after": int(after_count),
            "legal_action_count_before": int(len(legal_before)),
            "unblocked_action_count_before": int(len(unblocked_before)),
            "non_discard_unblocked_action_count_before": int(len(non_discard_unblocked)),
            "forced_singleton_discard": bool(
                (kind == DISCARD_POTION_ACTION_KIND or action_id_lower.startswith("discard_potion:"))
                and len(unblocked_before) == 1
                and not non_discard_unblocked
            ),
            "optional_discard_with_alternative": bool(
                (kind == DISCARD_POTION_ACTION_KIND or action_id_lower.startswith("discard_potion:"))
                and len(non_discard_unblocked) > 0
            ),
            "execute_ok": bool(execute_ok),
            "terminated": bool(terminated),
            "truncated": bool(truncated),
            "reward_after_shaping": float(reward),
            "state_version_before": self._state_version_from_obs(prev_obs),
            "state_version_after": self._state_version_from_obs(after_obs),
            "floor": _float(run.get("floor", run.get("total_floor", run.get("act_floor")))),
            "room_type": run.get("room_type") or run.get("state_type"),
            "room_model": room_model,
        }

    def _potion_hoarding_penalty(
        self,
        final_obs: dict[str, Any] | None,
        *,
        terminated: bool,
        truncated: bool,
    ) -> float:
        """One-shot penalty at episode end per unused potion in inventory.

        Fires on both natural termination (death/victory) and watchdog
        truncation — any end-of-episode unused potion is a wasted
        resource regardless of cause. Capped at
        POTION_HOARDING_MAX_PENALTY_ABS so the total penalty can never
        exceed one floor-clear bonus, preserving the "policy should
        still prefer to have potions over not having them" invariant.
        """
        if not (terminated or truncated):
            return 0.0
        unused = self._count_nonempty_potions(final_obs)
        if unused <= 0:
            return 0.0
        raw = unused * float(POTION_HOARDING_PENALTY_PER_POTION)
        cap = float(POTION_HOARDING_MAX_PENALTY_ABS)
        # Keep the sign; clip magnitude.
        penalty = max(raw, -cap) if raw < 0 else min(raw, cap)
        self._episode_telemetry["potion_hoarding_unused_at_end"] = float(unused)
        self._episode_telemetry["potion_hoarding_penalty_total"] += penalty
        return penalty

    def _full_run_death_terminal_penalty(
        self,
        before_obs: dict[str, Any] | None,
        after_obs: dict[str, Any] | None,
        *,
        terminated: bool,
        truncated: bool,
    ) -> float:
        """One-shot penalty for non-boss full-run deaths.

        The late-Act floor-clear ladder is intentionally positive so the
        policy values reaching Act 1 boss.  Without a matching terminal death
        loss, however, floor 13/14 normal-combat deaths can still end with a
        positive episode return.  Apply a terminal-only, floor-scaled penalty
        to make "got deep and died" clearly worse than surviving to the boss.

        Boss fights have a separate damage-undo terminal penalty; do not stack
        this generic term on top of it.
        """
        if not terminated or truncated:
            return 0.0
        if self._is_boss_encounter(before_obs) or self._is_boss_encounter(after_obs):
            return 0.0
        player = (
            after_obs.get("player")
            if isinstance(after_obs, dict) and isinstance(after_obs.get("player"), dict)
            else {}
        )
        hp = _float((player or {}).get("hp"))
        max_hp = max(_float((player or {}).get("max_hp"), 1.0), 1.0)
        if hp > 0.0:
            return 0.0

        missing_hp_ratio = float(np.clip((max_hp - max(hp, 0.0)) / max_hp, 0.0, 1.0))

        def _floor_from(obs: dict[str, Any] | None) -> float:
            run = obs.get("run") if isinstance(obs, dict) and isinstance(obs.get("run"), dict) else {}
            for key in ("act_floor", "floor", "total_floor"):
                val = run.get(key)
                if val is None:
                    continue
                try:
                    return float(val)
                except (TypeError, ValueError):
                    continue
            return 0.0

        # Act-local floor is ideal.  If only total floor is present, capping at
        # 17 still gives a full late-act penalty in later acts, which is safer
        # than letting Act2/3 deaths look like floor-1 deaths.
        floor_value = max(
            _floor_from(before_obs),
            _floor_from(after_obs),
            float(getattr(self, "_max_floor_reached", 0) or 0),
        )
        floor_norm = float(np.clip(floor_value / 17.0, 0.0, 1.0))
        magnitude = (
            float(FULL_RUN_DEATH_PENALTY_BASE)
            + float(FULL_RUN_DEATH_PENALTY_MISSING_HP_SCALE) * missing_hp_ratio
            + float(FULL_RUN_DEATH_PENALTY_LATE_ACT_SCALE) * floor_norm
        )
        penalty = -magnitude
        self._episode_telemetry["full_run_death_terminal_penalty_events"] += 1.0
        self._episode_telemetry["full_run_death_terminal_penalty_total"] += float(penalty)
        self._episode_telemetry["full_run_death_terminal_floor_norm"] = float(floor_norm)
        self._episode_telemetry["full_run_death_terminal_missing_hp_ratio"] = float(missing_hp_ratio)
        return penalty

    def _floor_clear_reward(
        self,
        before_obs: dict[str, Any] | None,
        after_obs: dict[str, Any] | None,
    ) -> float:
        """One-shot bonus on every floor advancement past FLOOR_CLEAR_MIN_FLOOR.

        Two-tier:
          - Reaching a boss floor (BOSS_ACT_FLOORS) → BOSS_FLOOR_ENTRY_BONUS
          - Reaching any other qualifying floor → FLOOR_CLEAR_BONUS_PER_FLOOR

        Fires exactly once per floor transition (when floor strictly
        increases), not every step. Decays back to zero on same-floor
        combats. Low-floor advancement (floor 0-10) gives nothing —
        those are the easy half of Act 1 where the policy already
        regularly reaches, and rewarding them would be wasteful
        shaping on solved states.
        """
        if not isinstance(before_obs, dict) or not isinstance(after_obs, dict):
            return 0.0
        before_run = before_obs.get("run") if isinstance(before_obs.get("run"), dict) else {}
        after_run = after_obs.get("run") if isinstance(after_obs.get("run"), dict) else {}
        try:
            before_floor = int(before_run.get("floor") or 0)
            after_floor = int(after_run.get("floor") or 0)
        except (TypeError, ValueError):
            return 0.0
        if after_floor <= before_floor:
            return 0.0
        if after_floor < FLOOR_CLEAR_MIN_FLOOR:
            return 0.0
        if after_floor in BOSS_ACT_FLOORS:
            reward = float(BOSS_FLOOR_ENTRY_BONUS)
            self._episode_telemetry["boss_floor_entry_events"] += 1.0
            self._episode_telemetry["floor_clear_reward_total"] += reward
            return reward
        reward = float(FLOOR_CLEAR_BONUS_PER_FLOOR)
        self._episode_telemetry["floor_clear_events"] += 1.0
        self._episode_telemetry["floor_clear_reward_total"] += reward
        return reward

    def _progress_fingerprint(self) -> tuple:
        """Coarse snapshot of 'has the world meaningfully advanced?' signals.

        Two consecutive steps sharing the same fingerprint means the agent
        chose an action that left the visible game state identical — no
        floor change, no combat round tick, no damage dealt, no hp loss,
        no card selection progress. A handful of these can happen
        legitimately (0-cost card draws, null-effect selections). Hundreds
        in a row means the policy is stuck in a no-op loop and should be
        truncated.

        Phase 8.1: extended with (selected_count, card_selection_prompt,
        can_confirm) so multi-step NEOW / card_reward / campfire selection
        flows don't get falsely flagged. These screens keep the basic 5-
        tuple constant for dozens of legit choice steps (no combat, no hp
        change, no floor change) — previously caused 21/21 floor-1 stuck
        cases in the Phase 8 smoke to all land on exactly stuck_steps=400.
        """
        obs = self._last_obs_raw or {}
        run = obs.get("run") if isinstance(obs, dict) else None
        combat = obs.get("combat") if isinstance(obs, dict) else None
        player = obs.get("player") if isinstance(obs, dict) else None
        floor = 0
        if isinstance(run, dict):
            for key in ("floor", "total_floor", "act_floor"):
                val = run.get(key)
                if val is not None:
                    try:
                        floor = int(val)
                        break
                    except (TypeError, ValueError):
                        continue
        combat_round = 0
        enemy_hp_total = 0
        if isinstance(combat, dict):
            try:
                combat_round = int(combat.get("round") or 0)
            except (TypeError, ValueError):
                combat_round = 0
            enemies = combat.get("enemies") if isinstance(combat.get("enemies"), list) else []
            for enemy in enemies:
                if not isinstance(enemy, dict):
                    continue
                try:
                    enemy_hp_total += int(enemy.get("hp") or 0)
                except (TypeError, ValueError):
                    continue
        player_hp = 0
        if isinstance(player, dict):
            try:
                player_hp = int(player.get("hp") or 0)
            except (TypeError, ValueError):
                player_hp = 0
        phase = str(obs.get("phase") or "")
        # Phase 8.1 additions: selection-aware fields.
        selected_count = 0
        selection_prompt = ""
        can_confirm = False
        decision = obs.get("decision") if isinstance(obs, dict) else None
        if isinstance(decision, dict):
            try:
                selected_count = int(decision.get("selected_count") or 0)
            except (TypeError, ValueError):
                selected_count = 0
        card_selection = obs.get("card_selection") if isinstance(obs, dict) else None
        if isinstance(card_selection, dict):
            selection_prompt = str(card_selection.get("prompt") or "")
            can_confirm = bool(card_selection.get("can_confirm"))
        return (
            phase, floor, combat_round, enemy_hp_total, player_hp,
            selected_count, selection_prompt, bool(can_confirm),
        )

    def _check_stuck_watchdog(self, bridge_info: Any) -> tuple[bool, Any]:
        """Increment stuck counter; truncate if fingerprint stable too long.

        Returns ``(truncated, bridge_info)``. When truncation fires, the
        returned bridge_info has ``truncation_reason="phase_stuck_watchdog"``
        plus ``stuck_phase`` / ``stuck_floor`` / ``stuck_steps`` fields so
        the async collector's ``episode_terminal`` event captures the
        reason. The bridge_info passed in is treated as the existing dict
        we should append to (not replaced).
        """
        fingerprint = self._progress_fingerprint()
        if fingerprint == self._stuck_fingerprint:
            self._stuck_steps += 1
        else:
            self._stuck_fingerprint = fingerprint
            self._stuck_steps = 1
        if self._stuck_steps < self.stuck_watchdog_steps:
            return False, bridge_info
        # Stuck — truncate and annotate.
        info_out: dict[str, Any] = dict(bridge_info) if isinstance(bridge_info, dict) else {}
        info_out["truncation_reason"] = "phase_stuck_watchdog"
        info_out["stuck_phase"] = fingerprint[0]
        info_out["stuck_floor"] = fingerprint[1]
        info_out["stuck_combat_round"] = fingerprint[2]
        info_out["stuck_enemy_hp_total"] = fingerprint[3]
        info_out["stuck_player_hp"] = fingerprint[4]
        info_out["stuck_steps"] = int(self._stuck_steps)
        return True, info_out

    def _player_hp_delta_reward(self, before_obs: dict[str, Any] | None, after_obs: dict[str, Any] | None) -> float:
        before_player = before_obs.get("player") if isinstance(before_obs, dict) else {}
        after_player = after_obs.get("player") if isinstance(after_obs, dict) else {}
        before_hp = _float((before_player or {}).get("hp"))
        after_hp = _float((after_player or {}).get("hp"))
        if before_hp <= 0.0 and after_hp <= 0.0:
            return 0.0
        # Symmetric shaping: positive for HP gain (rest site, heal potion,
        # heal event, lifesteal cards), negative for HP loss. Previously
        # we only penalized loss, which made rest-site decisions invisible
        # to PPO (0 reward whether agent rests or skips) and left HP
        # management as a distant-future credit-assignment problem.
        return (after_hp - before_hp) * PLAYER_HP_LOSS_REWARD_SCALE


    def _is_positive_progress_action(self, action: dict[str, Any]) -> bool:
        kind = str(action.get("kind") or "").strip()
        if kind not in ("play_card", "use_potion"):
            return False

        source = action.get("card") if kind == "play_card" else action.get("potion")
        if not isinstance(source, dict):
            return False

        if kind == "play_card" and str(source.get("type") or "").strip().lower() == "power":
            return True

        for key in ("damage", "block", "draw", "weak", "vulnerable", "heal", "strength", "dexterity", "summon"):
            if self._source_preview_metric(source, key) > 0.0:
                return True
        return False

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

        strict_context = strict_end_turn_waste_context(
            obs,
            legal_actions,
            chosen_action,
        )
        positive_actions = int(strict_context.get("urgent_positive_action_count", 0.0) or 0.0)
        urgent_indices = {int(idx) for idx in strict_context.get("urgent_positive_indices", [])}
        has_zero_cost_positive = False
        for idx, action in enumerate(legal_actions):
            if idx not in urgent_indices or not isinstance(action, dict):
                continue
            card = action.get("card")
            if not isinstance(card, dict):
                continue
            try:
                cost_raw = str(card.get("cost", card.get("resolved_energy_cost", 0.0))).strip().upper()
                cost = 0.0 if cost_raw == "X" else float(cost_raw or 0.0)
                if cost <= 0.0:
                    has_zero_cost_positive = True
            except (TypeError, ValueError):
                pass

        if positive_actions <= 0:
            return 0.0

        penalty = END_TURN_WASTE_BASE_PENALTY
        penalty += END_TURN_WASTE_ENERGY_PENALTY * min(energy, 3.0)
        if has_zero_cost_positive:
            penalty += END_TURN_WASTE_ZERO_COST_BONUS_PENALTY
        penalty += END_TURN_WASTE_EXTRA_ACTION_PENALTY * min(max(positive_actions - 1, 0), 2)
        return float(penalty)
