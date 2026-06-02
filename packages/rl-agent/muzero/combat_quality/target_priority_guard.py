"""Hard guard for source-vs-summon target-priority errors.

This guard handles the non-lethal half of the Fogmog/Living Fog failure mode.
The older ``summoner_target_guard`` already catches immediate lethal windows on
the summoner/source.  Here we only retarget when the selected single-target card
is aimed at a low/zero-pressure summon while the *same card* has a legal target
on the attacking source enemy.

The rule is intentionally conservative:

* no override for AoE;
* no override for lethal/high-pressure summons;
* no cross-card replacement;
* no potion/card substitution;
* only applies when the source is clearly the higher current pressure target.
"""

from __future__ import annotations

from collections.abc import Mapping
from types import MappingProxyType
from typing import Any

import numpy as np

from muzero.diagnostics.target_priority import (
    best_source_pressure_candidate,
    build_target_priority_payload,
)


TARGET_PRIORITY_GUARD_SEARCH_SUFFIXES: Mapping[str, str] = MappingProxyType(
    {
        "combat_quality_target_priority_source_pressure_guard_applicable": (
            "combat_quality_target_priority_source_pressure_guard_applicable_rate"
        ),
        "combat_quality_target_priority_source_pressure_guard_available": (
            "combat_quality_target_priority_source_pressure_guard_available_rate"
        ),
        "combat_quality_target_priority_source_pressure_guard_applied": (
            "combat_quality_target_priority_source_pressure_guard_applied_rate"
        ),
        "combat_quality_target_priority_source_pressure_guard_override": (
            "combat_quality_target_priority_source_pressure_guard_override_rate"
        ),
        "combat_quality_target_priority_source_pressure_guard_exception": (
            "combat_quality_target_priority_source_pressure_guard_exception_rate"
        ),
        "combat_quality_target_priority_source_pressure_guard_candidate_count": (
            "combat_quality_target_priority_source_pressure_guard_candidate_count"
        ),
        "combat_quality_target_priority_source_pressure_guard_original_is_summon": (
            "combat_quality_target_priority_source_pressure_guard_original_is_summon_rate"
        ),
        "combat_quality_target_priority_source_pressure_guard_zero_intent_summon": (
            "combat_quality_target_priority_source_pressure_guard_zero_intent_summon_rate"
        ),
        "combat_quality_target_priority_source_pressure_guard_source_intent": (
            "combat_quality_target_priority_source_pressure_guard_source_intent"
        ),
        "combat_quality_target_priority_source_pressure_guard_summon_intent": (
            "combat_quality_target_priority_source_pressure_guard_summon_intent"
        ),
        "combat_quality_target_priority_source_pressure_guard_original_idx": (
            "combat_quality_target_priority_source_pressure_guard_original_idx"
        ),
        "combat_quality_target_priority_source_pressure_guard_final_idx": (
            "combat_quality_target_priority_source_pressure_guard_final_idx"
        ),
    }
)


TARGET_PRIORITY_GUARD_DEFAULT_KEYS: tuple[str, ...] = tuple(
    TARGET_PRIORITY_GUARD_SEARCH_SUFFIXES.keys()
)


def _safe_float(value: Any, default: float = 0.0) -> float:
    try:
        out = float(value)
    except (TypeError, ValueError):
        return float(default)
    return out if np.isfinite(out) else float(default)


def _safe_int(value: Any, default: int = -1) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return int(default)


class TargetPriorityGuardMixin:
    """Narrow post-search retarget from low-pressure summon to source."""

    def _apply_source_pressure_target_guard(
        self,
        *,
        action_idx: int,
        legal_count: int,
        legal_actions: list[Any],
        mask_np: np.ndarray,
        raw_obs: Any | None,
        encounter: str,
        search_stats: dict[str, Any],
    ) -> int:
        for key in TARGET_PRIORITY_GUARD_DEFAULT_KEYS:
            search_stats.setdefault(key, 0.0)

        original_idx = int(action_idx)
        if not isinstance(raw_obs, dict) or not isinstance(legal_actions, list):
            return original_idx
        if not (0 <= original_idx < int(legal_count)):
            return original_idx

        try:
            payload = build_target_priority_payload(
                raw_obs=raw_obs,
                legal_actions=legal_actions,
                action_mask=mask_np,
                selected_idx=original_idx,
                search_policy=None,
                progress=None,
                encounter_id=str(encounter or ""),
                max_candidates=max(int(legal_count), 1),
            )
        except Exception:
            return original_idx

        if not isinstance(payload, dict):
            return original_idx

        selected = payload.get("selected_action") if isinstance(payload.get("selected_action"), dict) else {}
        best_source = (
            payload.get("best_source_pressure_candidate")
            if isinstance(payload.get("best_source_pressure_candidate"), dict)
            else {}
        )
        search_stats["combat_quality_target_priority_source_pressure_guard_applicable"] = 1.0
        search_stats["combat_quality_target_priority_source_pressure_guard_candidate_count"] = float(
            payload.get("source_pressure_candidate_count", 0.0) or 0.0
        )
        search_stats["combat_quality_target_priority_source_pressure_guard_original_idx"] = float(original_idx)
        search_stats["combat_quality_target_priority_source_pressure_guard_final_idx"] = float(original_idx)
        search_stats["combat_quality_target_priority_source_pressure_guard_source_intent"] = _safe_float(
            best_source.get("target_intent_damage"), 0.0
        )
        search_stats["combat_quality_target_priority_source_pressure_guard_summon_intent"] = _safe_float(
            selected.get("target_intent_damage"), 0.0
        )
        if bool(payload.get("selected_summon")):
            search_stats["combat_quality_target_priority_source_pressure_guard_original_is_summon"] = 1.0
        if bool(payload.get("selected_zero_intent_summon_over_attacking_source")):
            search_stats["combat_quality_target_priority_source_pressure_guard_zero_intent_summon"] = 1.0

        if str(payload.get("source_pressure_exception_reason") or "").strip():
            search_stats["combat_quality_target_priority_source_pressure_guard_exception"] = 1.0
            return original_idx

        if not bool(payload.get("source_pressure_available")):
            return original_idx
        search_stats["combat_quality_target_priority_source_pressure_guard_available"] = 1.0

        candidates_raw = payload.get("candidate_attacks")
        candidates = [c for c in candidates_raw if isinstance(c, dict)] if isinstance(candidates_raw, list) else []
        replacement = best_source_pressure_candidate(selected=selected, candidates=candidates)
        if not isinstance(replacement, dict):
            return original_idx

        new_idx = _safe_int(replacement.get("index"), original_idx)
        if new_idx == original_idx or not (0 <= new_idx < int(legal_count)):
            return original_idx
        if new_idx >= int(mask_np.shape[0]) or float(mask_np[new_idx]) <= 0.0:
            return original_idx

        dump_record = getattr(self, "_dump_combat_hard_guard_record", None)
        if callable(dump_record):
            try:
                dump_record(
                    kind="source_pressure_target",
                    raw_obs=raw_obs,
                    legal_actions=legal_actions,
                    original_idx=original_idx,
                    override_idx=int(new_idx),
                    risk=float(_safe_float(best_source.get("target_intent_damage"), 0.0)),
                    countdown=None,
                    encounter=encounter,
                    lethal_exemption=False,
                )
            except Exception:
                pass

        search_stats["combat_quality_target_priority_source_pressure_guard_applied"] = 1.0
        search_stats["combat_quality_target_priority_source_pressure_guard_override"] = 1.0
        search_stats["combat_quality_target_priority_source_pressure_guard_final_idx"] = float(new_idx)
        search_stats["combat_quality_hard_guard_override_any"] = 1.0
        return int(new_idx)


__all__ = [
    "TARGET_PRIORITY_GUARD_DEFAULT_KEYS",
    "TARGET_PRIORITY_GUARD_SEARCH_SUFFIXES",
    "TargetPriorityGuardMixin",
]
