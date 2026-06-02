"""Hard guard for summon/summoner target-priority mistakes.

This guard is deliberately narrow.  It only rewrites a combat target when the
currently selected attack is aimed at a summon/minion (or another non-summoner
enemy) while a legal attack can kill the summoner/source enemy immediately.

The goal is to patch the observed Fogmog/Living Fog failure mode without
turning target selection into a broad scripted policy:

* no override if the selected action is already a lethal summoner hit;
* no override for selected AoE actions;
* no override when killing the summon is a lethal-incoming exception;
* prefer card attacks over potion attacks when several lethal summoner choices
  exist, so the guard does not waste potions just to satisfy target priority.
"""

from __future__ import annotations

from collections.abc import Mapping
from types import MappingProxyType
from typing import Any

import numpy as np

from muzero.diagnostics.summoner_targeting import build_summoner_targeting_payload


SUMMONER_TARGET_GUARD_SEARCH_SUFFIXES: Mapping[str, str] = MappingProxyType(
    {
        "combat_quality_summoner_lethal_retarget_guard_applicable": (
            "combat_quality_summoner_lethal_retarget_guard_applicable_rate"
        ),
        "combat_quality_summoner_lethal_retarget_guard_available": (
            "combat_quality_summoner_lethal_retarget_guard_available_rate"
        ),
        "combat_quality_summoner_lethal_retarget_guard_applied": (
            "combat_quality_summoner_lethal_retarget_guard_applied_rate"
        ),
        "combat_quality_summoner_lethal_retarget_guard_override": (
            "combat_quality_summoner_lethal_retarget_guard_override_rate"
        ),
        "combat_quality_summoner_lethal_retarget_guard_exception": (
            "combat_quality_summoner_lethal_retarget_guard_exception_rate"
        ),
        "combat_quality_summoner_lethal_retarget_guard_candidate_count": (
            "combat_quality_summoner_lethal_retarget_guard_candidate_count"
        ),
        "combat_quality_summoner_lethal_retarget_guard_original_is_summon": (
            "combat_quality_summoner_lethal_retarget_guard_original_is_summon_rate"
        ),
        "combat_quality_summoner_lethal_retarget_guard_original_is_non_summoner": (
            "combat_quality_summoner_lethal_retarget_guard_original_is_non_summoner_rate"
        ),
        "combat_quality_summoner_lethal_retarget_guard_original_idx": (
            "combat_quality_summoner_lethal_retarget_guard_original_idx"
        ),
        "combat_quality_summoner_lethal_retarget_guard_final_idx": (
            "combat_quality_summoner_lethal_retarget_guard_final_idx"
        ),
    }
)


SUMMONER_TARGET_GUARD_DEFAULT_KEYS: tuple[str, ...] = tuple(
    SUMMONER_TARGET_GUARD_SEARCH_SUFFIXES.keys()
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


def _action_source(action: Any) -> dict[str, Any]:
    if not isinstance(action, dict):
        return {}
    source = action.get("card")
    if isinstance(source, dict):
        return source
    source = action.get("potion")
    if isinstance(source, dict):
        return source
    source = action.get("source")
    if isinstance(source, dict):
        return source
    return {}


def _candidate_cost(action: Any) -> float:
    if not isinstance(action, dict):
        return 99.0
    source = _action_source(action)
    semantic = action.get("semantic") if isinstance(action.get("semantic"), dict) else {}
    best: float | None = None
    for container in (action, source, semantic):
        if not isinstance(container, dict):
            continue
        for key in ("cost", "energy_cost", "base_cost", "card_cost"):
            if key not in container:
                continue
            value = _safe_float(container.get(key), 99.0)
            if value < -0.5:
                continue
            best = value if best is None else min(best, value)
    return float(best if best is not None else 99.0)


def _is_card_candidate(candidate: dict[str, Any], action: Any) -> bool:
    family = str(candidate.get("family") or "").strip().lower()
    if family == "play_card":
        return True
    if isinstance(action, dict) and isinstance(action.get("card"), dict):
        return True
    return False


def _best_lethal_summoner_candidate(
    *,
    payload: dict[str, Any],
    legal_actions: list[Any],
    legal_count: int,
    mask_np: np.ndarray,
) -> dict[str, Any] | None:
    candidates = payload.get("candidate_attacks")
    if not isinstance(candidates, list):
        return None

    valid: list[tuple[tuple[float, float, float, float, int], dict[str, Any]]] = []
    for candidate in candidates:
        if not isinstance(candidate, dict):
            continue
        idx = _safe_int(candidate.get("index"), -1)
        if idx < 0 or idx >= int(legal_count) or idx >= int(mask_np.shape[0]):
            continue
        if float(mask_np[idx]) <= 0.0:
            continue
        if not bool(candidate.get("target_is_summoner")) or not bool(candidate.get("kills_target")):
            continue
        action = legal_actions[idx] if idx < len(legal_actions) else None
        # Prefer cards over potions, lower cost, then higher policy if present,
        # then lower overkill.  The final stable tie-breaker is index.
        is_card = 1.0 if _is_card_candidate(candidate, action) else 0.0
        cost = _candidate_cost(action)
        policy = _safe_float(candidate.get("policy"), 0.0)
        damage = _safe_float(candidate.get("damage"), 0.0)
        hp = _safe_float(candidate.get("target_hp"), 0.0)
        overkill = max(0.0, damage - hp)
        score = (-is_card, cost, -policy, overkill, idx)
        valid.append((score, candidate))

    if not valid:
        return None
    valid.sort(key=lambda item: item[0])
    return valid[0][1]


class SummonerTargetGuardMixin:
    """Narrow post-search retarget for lethal summoner/source windows."""

    def _apply_summoner_lethal_retarget_guard(
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
        for key in SUMMONER_TARGET_GUARD_DEFAULT_KEYS:
            search_stats.setdefault(key, 0.0)

        if not isinstance(raw_obs, dict) or not isinstance(legal_actions, list):
            return int(action_idx)
        if not (0 <= int(action_idx) < int(legal_count)):
            return int(action_idx)

        try:
            payload = build_summoner_targeting_payload(
                raw_obs=raw_obs,
                legal_actions=legal_actions,
                action_mask=mask_np,
                selected_idx=int(action_idx),
                search_policy=None,
                progress=None,
                encounter_id=str(encounter or ""),
                max_candidates=max(int(legal_count), 1),
            )
        except Exception:
            return int(action_idx)

        if not isinstance(payload, dict):
            return int(action_idx)

        search_stats["combat_quality_summoner_lethal_retarget_guard_applicable"] = 1.0
        selected = payload.get("selected_action") if isinstance(payload.get("selected_action"), dict) else {}
        search_stats["combat_quality_summoner_lethal_retarget_guard_original_idx"] = float(action_idx)
        search_stats["combat_quality_summoner_lethal_retarget_guard_final_idx"] = float(action_idx)
        search_stats["combat_quality_summoner_lethal_retarget_guard_candidate_count"] = float(
            payload.get("lethal_summoner_candidate_count", 0.0) or 0.0
        )
        if bool(selected.get("target_is_summon")):
            search_stats["combat_quality_summoner_lethal_retarget_guard_original_is_summon"] = 1.0
        if bool(payload.get("selected_non_summoner_over_lethal_summoner")):
            search_stats["combat_quality_summoner_lethal_retarget_guard_original_is_non_summoner"] = 1.0

        if not bool(payload.get("lethal_summoner_available")):
            return int(action_idx)
        search_stats["combat_quality_summoner_lethal_retarget_guard_available"] = 1.0

        if bool(payload.get("selected_summoner_when_lethal_available")):
            return int(action_idx)

        if str(payload.get("exception_reason") or "").strip():
            search_stats["combat_quality_summoner_lethal_retarget_guard_exception"] = 1.0
            return int(action_idx)

        selected_bad_target = bool(payload.get("selected_summon_over_lethal_summoner")) or bool(
            payload.get("selected_non_summoner_over_lethal_summoner")
        )
        if not selected_bad_target:
            return int(action_idx)

        replacement = _best_lethal_summoner_candidate(
            payload=payload,
            legal_actions=legal_actions,
            legal_count=int(legal_count),
            mask_np=mask_np,
        )
        if not isinstance(replacement, dict):
            return int(action_idx)

        new_idx = _safe_int(replacement.get("index"), int(action_idx))
        if new_idx == int(action_idx) or not (0 <= new_idx < int(legal_count)):
            return int(action_idx)

        search_stats["combat_quality_summoner_lethal_retarget_guard_applied"] = 1.0
        search_stats["combat_quality_summoner_lethal_retarget_guard_override"] = 1.0
        search_stats["combat_quality_summoner_lethal_retarget_guard_final_idx"] = float(new_idx)
        return int(new_idx)


__all__ = [
    "SUMMONER_TARGET_GUARD_DEFAULT_KEYS",
    "SUMMONER_TARGET_GUARD_SEARCH_SUFFIXES",
    "SummonerTargetGuardMixin",
]
