"""Combat hard-guard orchestrator for MuZero training.

The tactical guard implementations are intentionally split by responsibility
so ``muzero.train`` remains an orchestration entrypoint instead of a growing
policy/heuristic monolith.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from sts2_env.observation_v2 import MAX_ACTIONS

from muzero.combat_quality import COMBAT_HARD_GUARD_DEFAULT_KEYS
from muzero.combat_quality.basic_hard_guards import BasicCombatHardGuardMixin
from muzero.combat_quality.boss_survival_hard_guards import BossSurvivalHardGuardMixin
from muzero.combat_quality.late_normal_hard_guards import LateNormalHardGuardMixin
from muzero.combat_quality.meaningful_damage_guard import MeaningfulDamageEndTurnGuardMixin
from muzero.combat_quality.no_pressure_block_guard import NoPressureBlockGuardMixin
from muzero.combat_quality.potion_bad_use_guard import PotionBadUseGuardMixin
from muzero.combat_quality.refund_no_followup_guard import RefundNoFollowupGuardMixin
from muzero.combat_quality.selection_loop_guard import SelectionLoopGuardMixin
from muzero.combat_quality.strategic_defer_endturn_guard import StrategicDeferEndTurnGuardMixin
from muzero.combat_quality.strategic_skip_guard import StrategicSkipGuardMixin
from muzero.combat_quality.survival_non_endturn_guard import NonEndTurnSurvivalGuardMixin
from muzero.combat_quality.urgent_endturn_guard import UrgentEndTurnGuardMixin


class CombatHardGuardMixin(
    BasicCombatHardGuardMixin,
    BossSurvivalHardGuardMixin,
    LateNormalHardGuardMixin,
    MeaningfulDamageEndTurnGuardMixin,
    NoPressureBlockGuardMixin,
    RefundNoFollowupGuardMixin,
    StrategicSkipGuardMixin,
    StrategicDeferEndTurnGuardMixin,
    UrgentEndTurnGuardMixin,
    NonEndTurnSurvivalGuardMixin,
    PotionBadUseGuardMixin,
    SelectionLoopGuardMixin,
):
    _HARD_NORMAL_RACE_ENCOUNTER_MARKERS = (
        "slumbering_beetle_normal",
        "slimed_berserker_normal",
        "fogmog_normal",
        "scrolls_of_biting_normal",
        "the_obscura_normal",
        "construct_menagerie_normal",
        "ovicopter_normal",
    )

    @staticmethod
    def _is_normal_or_weak_hallway_encounter(encounter_tier: Any, encounter_text: Any) -> bool:
        """Return true for non-elite, non-boss hallway combats.

        The bridge/logging stack uses both explicit tiers (``normal``/``weak``)
        and encounter ids such as ``*_NORMAL`` / ``*_WEAK``.  Several combat
        guards share this hallway classifier, so keep it with the combat guard
        mixin instead of the legacy trainer entrypoint.
        """

        tier = str(encounter_tier or "").strip().lower()
        text = str(encounter_text or "").strip().lower()
        if "boss" in text or "elite" in text:
            return False
        if tier in {"normal", "weak"}:
            return True
        return bool("_normal" in text or "_weak" in text or "normal" in text or "weak" in text)

    @staticmethod
    def _combat_encounter_text(raw_obs: Any | None, encounter_hint: Any = "") -> str:
        """Return a conservative joined encounter string from bridge/sandbox shapes.

        Live bridge observations, full-run planner context, and combat-sandbox
        snapshots do not expose encounter ids in one stable location.  Hard
        guards only need a best-effort classifier string, so collect the common
        top-level/run/combat fields without mutating the observation.
        """

        parts: list[str] = []
        if str(encounter_hint or "").strip():
            parts.append(str(encounter_hint))
        if isinstance(raw_obs, dict):
            for key in (
                "encounter",
                "encounter_id",
                "room_model",
                "roomModel",
                "snapshot_encounter_id",
                "snapshot_sample_id",
            ):
                value = raw_obs.get(key)
                if value is not None and str(value).strip():
                    parts.append(str(value))
            for parent_key in ("run", "combat", "map", "planner_context", "combat_snapshot"):
                parent = raw_obs.get(parent_key)
                if not isinstance(parent, dict):
                    continue
                for key in (
                    "encounter",
                    "encounter_id",
                    "room_model",
                    "roomModel",
                    "room_type",
                    "roomType",
                    "sample_id",
                ):
                    value = parent.get(key)
                    if value is not None and str(value).strip():
                        parts.append(str(value))
        return " ".join(parts)

    @classmethod
    def _is_hard_normal_race_encounter(cls, encounter_text: Any) -> bool:
        """Return true for curated hard-normal hallway fights.

        These fights repeatedly appear in sandbox death slices as long/race
        normal encounters.  The classifier is intentionally separate from the
        generic ``normal`` hallway detector so weak fights do not inherit
        overly aggressive late-normal potion/survival rules.
        """

        text = str(encounter_text or "").strip().lower()
        return any(marker in text for marker in cls._HARD_NORMAL_RACE_ENCOUNTER_MARKERS)

    @staticmethod
    def _combat_floor_value(raw_obs: Any | None) -> float:
        """Read the effective combat floor, preferring sandbox snapshot floor.

        Combat sandbox injects deck/relic/potion/card-state context from a real
        pre-combat snapshot, but bridge ``run.floor`` can remain the fake live
        combat room floor.  Guards that model late-Act1 hallway pressure must
        prefer the injected snapshot floor while leaving ``run.floor`` intact
        for components that intentionally consume the bridge value.
        """

        if not isinstance(raw_obs, dict):
            return 0.0

        def _as_positive_float(value: Any) -> float:
            try:
                number = float(value)
            except (TypeError, ValueError):
                return 0.0
            if not np.isfinite(number) or number <= 0.0:
                return 0.0
            return float(number)

        snapshot_sources: list[dict[str, Any]] = [raw_obs]
        for parent_key in ("run", "combat_snapshot", "planner_context", "map"):
            parent = raw_obs.get(parent_key)
            if isinstance(parent, dict):
                snapshot_sources.append(parent)
        for src in snapshot_sources:
            for key in ("snapshot_floor_number", "floor_number"):
                floor_value = _as_positive_float(src.get(key))
                if floor_value > 0.0:
                    return floor_value

        ordinary_sources: list[dict[str, Any]] = [raw_obs]
        for parent_key in ("run", "map"):
            parent = raw_obs.get(parent_key)
            if isinstance(parent, dict):
                ordinary_sources.append(parent)
        for src in ordinary_sources:
            for key in ("floor", "current_floor", "floor_num", "room_floor", "floor_number"):
                floor_value = _as_positive_float(src.get(key))
                if floor_value > 0.0:
                    return floor_value
        return 0.0

    def _apply_combat_action_hard_guards(
        self,
        *,
        action_idx: int,
        legal_actions: list[Any] | None,
        action_mask: Any,
        raw_obs: Any | None,
        boss_ctx: Any | None,
        encounter: str,
        search_stats: dict[str, Any],
    ) -> int:
        """Apply ordered combat hard guards and emit stable metrics.

        When the post-search ``action_idx`` violates one of the configured
        tactical invariants, replace it with a safe candidate and refresh the
        per-step ``combat_quality_*`` flags so downstream logging reflects the
        post-override action.
        """
        if isinstance(raw_obs, dict) and str(encounter or "").strip():
            raw_has_encounter = any(
                str(raw_obs.get(key) or "").strip()
                for key in ("encounter_id", "encounter", "room_model", "roomModel")
            )
            run = raw_obs.get("run") if isinstance(raw_obs.get("run"), dict) else {}
            if isinstance(run, dict):
                raw_has_encounter = raw_has_encounter or any(
                    str(run.get(key) or "").strip()
                    for key in ("encounter_id", "encounter", "room_model", "roomModel")
                )
            if not raw_has_encounter:
                raw_obs = dict(raw_obs)
                raw_obs["encounter_id"] = str(encounter)

        for key in COMBAT_HARD_GUARD_DEFAULT_KEYS:
            search_stats.setdefault(key, 0.0)

        if not isinstance(legal_actions, list) or len(legal_actions) == 0:
            return int(action_idx)

        try:
            mask_np = np.asarray(action_mask, dtype=np.float32).reshape(-1)
        except Exception:
            return int(action_idx)
        legal_count = min(len(legal_actions), int(mask_np.shape[0]) if mask_np.size else 0, MAX_ACTIONS)
        if not (0 <= int(action_idx) < legal_count):
            return int(action_idx)
        if mask_np[int(action_idx)] <= 0:
            return int(action_idx)

        original_action_idx = int(action_idx)

        action_idx = self._apply_potion_discard_priority_guard(
            action_idx=int(action_idx),
            legal_count=legal_count,
            legal_actions=legal_actions,
            mask_np=mask_np,
            raw_obs=raw_obs,
            encounter=encounter,
            search_stats=search_stats,
        )
        action_idx = self._apply_kaiser_facing_guard(
            action_idx=int(action_idx),
            legal_count=legal_count,
            legal_actions=legal_actions,
            mask_np=mask_np,
            raw_obs=raw_obs,
            boss_ctx=boss_ctx,
            encounter=encounter,
            search_stats=search_stats,
        )
        action_idx = self._apply_insatiable_escape_guard(
            action_idx=int(action_idx),
            legal_count=legal_count,
            legal_actions=legal_actions,
            mask_np=mask_np,
            raw_obs=raw_obs,
            boss_ctx=boss_ctx,
            encounter=encounter,
            search_stats=search_stats,
        )
        action_idx = self._apply_x_cost_zero_guard(
            action_idx=int(action_idx),
            legal_count=legal_count,
            legal_actions=legal_actions,
            mask_np=mask_np,
            raw_obs=raw_obs,
            encounter=encounter,
            search_stats=search_stats,
        )
        action_idx = self._apply_refund_no_followup_guard(
            action_idx=int(action_idx),
            legal_count=legal_count,
            legal_actions=legal_actions,
            mask_np=mask_np,
            raw_obs=raw_obs,
            encounter=encounter,
            search_stats=search_stats,
        )
        action_idx = self._apply_strategic_skip_guard(
            action_idx=int(action_idx),
            legal_count=legal_count,
            legal_actions=legal_actions,
            mask_np=mask_np,
            raw_obs=raw_obs,
            encounter=encounter,
            search_stats=search_stats,
        )
        action_idx = self._apply_hp_cost_margin_guard(
            action_idx=int(action_idx),
            legal_count=legal_count,
            legal_actions=legal_actions,
            mask_np=mask_np,
            raw_obs=raw_obs,
            encounter=encounter,
            search_stats=search_stats,
        )
        action_idx = self._apply_elite_boss_lethal_endturn_guard(
            action_idx=int(action_idx),
            legal_count=legal_count,
            legal_actions=legal_actions,
            mask_np=mask_np,
            raw_obs=raw_obs,
            encounter=encounter,
            search_stats=search_stats,
        )
        action_idx = self._apply_meaningful_damage_endturn_guard(
            action_idx=int(action_idx),
            legal_count=legal_count,
            legal_actions=legal_actions,
            mask_np=mask_np,
            raw_obs=raw_obs,
            encounter=encounter,
            search_stats=search_stats,
        )
        action_idx = self._apply_strategic_defer_endturn_guard(
            action_idx=int(action_idx),
            legal_count=legal_count,
            legal_actions=legal_actions,
            mask_np=mask_np,
            raw_obs=raw_obs,
            encounter=encounter,
            search_stats=search_stats,
        )
        action_idx = self._apply_no_pressure_block_guard(
            action_idx=int(action_idx),
            legal_count=legal_count,
            legal_actions=legal_actions,
            mask_np=mask_np,
            raw_obs=raw_obs,
            encounter=encounter,
            search_stats=search_stats,
        )
        action_idx = self._apply_boss_race_potion_guard(
            action_idx=int(action_idx),
            legal_count=legal_count,
            legal_actions=legal_actions,
            mask_np=mask_np,
            raw_obs=raw_obs,
            encounter=encounter,
            search_stats=search_stats,
        )
        action_idx = self._apply_boss_survival_potion_guard(
            action_idx=int(action_idx),
            legal_count=legal_count,
            legal_actions=legal_actions,
            mask_np=mask_np,
            raw_obs=raw_obs,
            encounter=encounter,
            search_stats=search_stats,
        )
        action_idx = self._apply_boss_survival_block_guard(
            action_idx=int(action_idx),
            legal_count=legal_count,
            legal_actions=legal_actions,
            mask_np=mask_np,
            raw_obs=raw_obs,
            encounter=encounter,
            search_stats=search_stats,
        )
        action_idx = self._apply_late_normal_lethal_endturn_guard(
            action_idx=int(action_idx),
            legal_count=legal_count,
            legal_actions=legal_actions,
            mask_np=mask_np,
            raw_obs=raw_obs,
            encounter=encounter,
            search_stats=search_stats,
        )
        action_idx = self._apply_late_normal_survival_guard(
            action_idx=int(action_idx),
            legal_count=legal_count,
            legal_actions=legal_actions,
            mask_np=mask_np,
            raw_obs=raw_obs,
            encounter=encounter,
            search_stats=search_stats,
        )
        action_idx = self._apply_late_normal_race_potion_guard(
            action_idx=int(action_idx),
            legal_count=legal_count,
            legal_actions=legal_actions,
            mask_np=mask_np,
            raw_obs=raw_obs,
            encounter=encounter,
            search_stats=search_stats,
        )
        action_idx = self._apply_survival_non_endturn_guard(
            action_idx=int(action_idx),
            legal_count=legal_count,
            legal_actions=legal_actions,
            mask_np=mask_np,
            raw_obs=raw_obs,
            encounter=encounter,
            search_stats=search_stats,
        )
        # Generic urgent EndTurn rescue runs after the boss/late-normal
        # specialist guards so it does not steal their more specific telemetry.
        # It only patches remaining exploration-tail EndTurn samples when the
        # same stable frontier exposes an urgent safe non-EndTurn alternative.
        action_idx = self._apply_urgent_endturn_guard(
            action_idx=int(action_idx),
            legal_count=legal_count,
            legal_actions=legal_actions,
            mask_np=mask_np,
            raw_obs=raw_obs,
            encounter=encounter,
            search_stats=search_stats,
        )
        action_idx = self._apply_potion_bad_use_guard(
            action_idx=int(action_idx),
            legal_count=legal_count,
            legal_actions=legal_actions,
            mask_np=mask_np,
            raw_obs=raw_obs,
            encounter=encounter,
            search_stats=search_stats,
        )
        action_idx = self._apply_selection_loop_guard(
            action_idx=int(action_idx),
            legal_count=legal_count,
            legal_actions=legal_actions,
            mask_np=mask_np,
            raw_obs=raw_obs,
            encounter=encounter,
            search_stats=search_stats,
        )

        if int(action_idx) != int(original_action_idx):
            search_stats["combat_quality_hard_guard_override_any"] = 1.0
        return int(action_idx)
