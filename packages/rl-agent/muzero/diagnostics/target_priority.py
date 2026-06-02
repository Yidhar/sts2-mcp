"""Target-priority diagnostics for source-vs-summon combat mistakes.

This module is intentionally narrower than a full tactical policy.  It records
whether the chosen single-target attack spent damage on a summon/minion while a
legal same-card target existed for the summoner/source enemy that was applying
substantially more pressure.

The first production use case is Act1 Fogmog/Living Fog style fights:

* source/summoner is attacking or otherwise the dangerous root of the fight;
* summon/minion has zero or much lower current intent damage;
* policy selects the summon with very high confidence;
* the same card could have hit the source instead.

The payload is diagnostic-first and is also consumed by a narrow hard guard in
``muzero.combat_quality.target_priority_guard``.  Keeping it here prevents the
combat trainer and ``train.py`` from accumulating another block of ad-hoc
target-selection logic.
"""

from __future__ import annotations

from dataclasses import dataclass
import json
import math
import time
from typing import Any

import numpy as np

from muzero.diagnostics.summoner_targeting import build_summoner_targeting_payload


TARGET_PRIORITY_TB_KEYS: tuple[tuple[str, str], ...] = (
    ("seen", "target_priority_seen_count"),
    ("source_present_rate", "target_priority_source_present_rate"),
    ("summon_present_rate", "target_priority_summon_present_rate"),
    ("selected_source_rate", "target_priority_selected_source_rate"),
    ("selected_summon_rate", "target_priority_selected_summon_rate"),
    ("source_pressure_available_rate", "target_priority_source_pressure_available_rate"),
    (
        "selected_summon_over_source_pressure_rate",
        "target_priority_selected_summon_over_source_pressure_rate",
    ),
    (
        "selected_zero_intent_summon_over_attacking_source_rate",
        "target_priority_selected_zero_intent_summon_over_attacking_source_rate",
    ),
    ("source_pressure_exception_rate", "target_priority_source_pressure_exception_rate"),
    ("source_pressure_candidate_count_mean", "target_priority_source_pressure_candidate_count_mean"),
    ("source_damage_lost_mean", "target_priority_source_damage_lost_mean"),
    ("summon_overkill_mean", "target_priority_summon_overkill_mean"),
    (
        "cross_card_lethal_source_available_rate",
        "target_priority_cross_card_lethal_source_available_rate",
    ),
    (
        "selected_summon_over_cross_card_lethal_source_rate",
        "target_priority_selected_summon_over_cross_card_lethal_source_rate",
    ),
)


def _safe_float(value: Any, default: float = 0.0) -> float:
    try:
        out = float(value)
    except (TypeError, ValueError):
        return float(default)
    return out if math.isfinite(out) else float(default)


def _safe_int(value: Any, default: int = -1) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return int(default)


def _text(value: Any) -> str:
    return str(value or "").strip().lower()


def _same_card_identity(a: dict[str, Any], b: dict[str, Any]) -> bool:
    """Return true when two candidate payloads are the same card/action source.

    Combat legal actions usually expose one candidate per target for the same
    card instance.  Retargeting across different cards would become broad policy
    scripting, so both diagnostics and guard use this conservative same-source
    predicate.
    """

    if not isinstance(a, dict) or not isinstance(b, dict):
        return False
    family_a = _text(a.get("family"))
    family_b = _text(b.get("family"))
    if family_a and family_b and family_a != family_b:
        return False
    card_id_a = _text(a.get("card_id"))
    card_id_b = _text(b.get("card_id"))
    if card_id_a and card_id_b:
        return card_id_a == card_id_b
    title_a = _text(a.get("card_title"))
    title_b = _text(b.get("card_title"))
    if title_a and title_b:
        return title_a == title_b
    # Last-resort fallback for compact test payloads that only expose a family.
    return bool(family_a and family_a == family_b)


def _is_card_candidate(candidate: dict[str, Any]) -> bool:
    return _text(candidate.get("family")) == "play_card"


def _selected_summon_lethal_incoming(base_payload: dict[str, Any]) -> bool:
    return _text(base_payload.get("exception_reason")) == "summon_lethal_incoming"


def _selected_aoe(base_payload: dict[str, Any], selected: dict[str, Any]) -> bool:
    return bool(selected.get("aoe")) or _text(base_payload.get("exception_reason")) == "selected_aoe"


def _source_pressure_candidate_rows(
    *,
    selected: dict[str, Any],
    candidates: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for candidate in candidates:
        if not isinstance(candidate, dict):
            continue
        if not bool(candidate.get("target_is_summoner")):
            continue
        if bool(candidate.get("aoe")):
            continue
        if not _same_card_identity(selected, candidate):
            continue
        rows.append(candidate)
    return rows


def best_source_pressure_candidate(
    *,
    selected: dict[str, Any],
    candidates: list[dict[str, Any]],
) -> dict[str, Any] | None:
    """Pick the best same-card source candidate for a source-pressure retarget."""

    rows = _source_pressure_candidate_rows(selected=selected, candidates=candidates)
    if not rows:
        return None
    rows.sort(
        key=lambda c: (
            bool(c.get("kills_target")),
            _safe_float(c.get("target_intent_damage"), 0.0),
            _safe_float(c.get("damage"), 0.0),
            _safe_float(c.get("policy"), 0.0),
            -_safe_int(c.get("index"), 9999),
        ),
        reverse=True,
    )
    return rows[0]


def source_pressure_decision(
    *,
    base_payload: dict[str, Any],
) -> dict[str, Any]:
    """Compute summon-over-source pressure labels from a summoner payload."""

    selected = base_payload.get("selected_action") if isinstance(base_payload.get("selected_action"), dict) else {}
    candidates_raw = base_payload.get("candidate_attacks")
    candidates = [c for c in candidates_raw if isinstance(c, dict)] if isinstance(candidates_raw, list) else []

    source_candidates = [c for c in candidates if bool(c.get("target_is_summoner"))]
    summon_candidates = [c for c in candidates if bool(c.get("target_is_summon"))]
    best_source = best_source_pressure_candidate(selected=selected, candidates=candidates)

    selected_is_summon = bool(selected.get("target_is_summon"))
    selected_is_source = bool(selected.get("target_is_summoner"))
    selected_intent = _safe_float(selected.get("target_intent_damage"), 0.0)
    selected_damage = _safe_float(selected.get("damage"), 0.0)
    selected_hp = _safe_float(selected.get("target_hp"), 0.0)
    selected_overkill = max(0.0, selected_damage - selected_hp) if selected_hp > 0.0 else 0.0

    best_source_intent = _safe_float(best_source.get("target_intent_damage"), 0.0) if isinstance(best_source, dict) else 0.0
    best_source_damage = _safe_float(best_source.get("damage"), 0.0) if isinstance(best_source, dict) else 0.0
    damage_lost = max(0.0, best_source_damage - selected_damage) if isinstance(best_source, dict) else 0.0
    if isinstance(best_source, dict):
        # Most same-card retargets have identical damage.  In that common case,
        # "damage lost" should reflect useful damage diverted away from the
        # dangerous source, not literal action damage difference.
        damage_lost = max(damage_lost, min(selected_damage, max(_safe_float(best_source.get("target_hp"), 0.0), 0.0)))

    exception_reason = ""
    if _selected_aoe(base_payload, selected):
        exception_reason = "selected_aoe"
    elif _selected_summon_lethal_incoming(base_payload):
        exception_reason = "summon_lethal_incoming"
    elif selected_is_summon and best_source is None:
        exception_reason = "no_same_card_source_candidate"
    elif selected_is_summon and selected_intent > 0.0 and selected_intent >= max(1.0, best_source_intent - 4.0):
        exception_reason = "summon_has_comparable_pressure"

    # Deliberately narrow threshold: only call it a source-pressure mistake when
    # the source is clearly attacking harder than the summon, or the summon is a
    # zero-intent body while the source attacks.  This prevents rewriting valid
    # kills on dangerous bombs/minions.
    pressure_gap = best_source_intent - selected_intent
    source_pressure_available = bool(
        selected_is_summon
        and isinstance(best_source, dict)
        and best_source_intent > 0.0
        and (
            (selected_intent <= 0.0 and best_source_intent >= 4.0)
            or pressure_gap >= 6.0
        )
        and not exception_reason
    )
    selected_summon_over_source_pressure = bool(selected_is_summon and source_pressure_available)
    selected_zero_intent_summon_over_attacking_source = bool(
        selected_summon_over_source_pressure and selected_intent <= 0.0 and best_source_intent > 0.0
    )

    return {
        "selected_source": bool(selected_is_source),
        "selected_summon": bool(selected_is_summon),
        "source_candidate_count": int(len(source_candidates)),
        "summon_candidate_count": int(len(summon_candidates)),
        "source_pressure_candidate_count": int(len(_source_pressure_candidate_rows(selected=selected, candidates=candidates))),
        "source_pressure_available": bool(source_pressure_available),
        "selected_summon_over_source_pressure": bool(selected_summon_over_source_pressure),
        "selected_zero_intent_summon_over_attacking_source": bool(
            selected_zero_intent_summon_over_attacking_source
        ),
        "source_pressure_exception_reason": exception_reason,
        "best_source_pressure_candidate": best_source or {},
        "source_damage_lost": float(damage_lost if selected_summon_over_source_pressure else 0.0),
        "summon_overkill": float(selected_overkill if selected_is_summon else 0.0),
    }


def build_target_priority_payload(
    *,
    raw_obs: Any,
    legal_actions: list[Any],
    action_mask: Any,
    selected_idx: int,
    search_policy: Any = None,
    progress: dict[str, Any] | None = None,
    encounter_id: str = "",
    encounter_tier: str = "",
    max_candidates: int = 32,
) -> dict[str, Any] | None:
    """Return a compact target-priority payload or ``None`` for irrelevant fights."""

    base = build_summoner_targeting_payload(
        raw_obs=raw_obs,
        legal_actions=legal_actions,
        action_mask=action_mask,
        selected_idx=selected_idx,
        search_policy=search_policy,
        progress=progress,
        encounter_id=encounter_id,
        encounter_tier=encounter_tier,
        max_candidates=max(max_candidates, len(legal_actions) if isinstance(legal_actions, list) else 1),
    )
    if not isinstance(base, dict):
        return None
    if not (bool(base.get("summoner_present")) and bool(base.get("summon_present"))):
        return None

    decision = source_pressure_decision(base_payload=base)
    payload = {
        "schema": "target_priority_v1",
        "floor": base.get("floor"),
        "act_id": base.get("act_id"),
        "room_type": base.get("room_type"),
        "encounter_id": base.get("encounter_id"),
        "encounter_tier": base.get("encounter_tier"),
        "source_present": bool(base.get("summoner_present")),
        "summon_present": bool(base.get("summon_present")),
        "enemy_count": int(base.get("enemy_count", 0) or 0),
        "enemies": base.get("enemies") if isinstance(base.get("enemies"), list) else [],
        "combat": base.get("combat") if isinstance(base.get("combat"), dict) else {},
        "selected_action": base.get("selected_action") if isinstance(base.get("selected_action"), dict) else {},
        "candidate_attack_count": int(base.get("candidate_attack_count", 0) or 0),
        "candidate_attacks": base.get("candidate_attacks") if isinstance(base.get("candidate_attacks"), list) else [],
        "lethal_source_available": bool(base.get("lethal_summoner_available")),
        "cross_card_lethal_source_available": bool(base.get("cross_card_lethal_summoner_available")),
        "best_lethal_source_candidate": base.get("best_lethal_summoner_candidate")
        if isinstance(base.get("best_lethal_summoner_candidate"), dict)
        else {},
        "best_cross_card_lethal_source_candidate": base.get("best_cross_card_lethal_summoner_candidate")
        if isinstance(base.get("best_cross_card_lethal_summoner_candidate"), dict)
        else {},
        "selected_non_source_over_lethal_source": bool(
            base.get("selected_summon_over_lethal_summoner")
            or base.get("selected_non_summoner_over_lethal_summoner")
        ),
        "selected_summon_over_cross_card_lethal_source": bool(
            base.get("selected_summon_over_cross_card_lethal_summoner")
        ),
        "selected_non_source_over_cross_card_lethal_source": bool(
            base.get("selected_summon_over_cross_card_lethal_summoner")
            or base.get("selected_non_summoner_over_cross_card_lethal_summoner")
        ),
        **decision,
    }
    return {k: v for k, v in payload.items() if v not in (None, "", [], {})}


@dataclass
class TargetPriorityEpisodeTracker:
    seen: int = 0
    source_present: int = 0
    summon_present: int = 0
    selected_source: int = 0
    selected_summon: int = 0
    source_pressure_available: int = 0
    selected_summon_over_source_pressure: int = 0
    selected_zero_intent_summon_over_attacking_source: int = 0
    source_pressure_exception: int = 0
    source_pressure_candidate_count_sum: float = 0.0
    source_damage_lost_sum: float = 0.0
    summon_overkill_sum: float = 0.0
    cross_card_lethal_source_available: int = 0
    selected_summon_over_cross_card_lethal_source: int = 0

    def update(self, payload: dict[str, Any] | None) -> None:
        if not isinstance(payload, dict):
            return
        self.seen += 1
        if bool(payload.get("source_present")):
            self.source_present += 1
        if bool(payload.get("summon_present")):
            self.summon_present += 1
        if bool(payload.get("selected_source")):
            self.selected_source += 1
        if bool(payload.get("selected_summon")):
            self.selected_summon += 1
        if bool(payload.get("source_pressure_available")):
            self.source_pressure_available += 1
        if bool(payload.get("selected_summon_over_source_pressure")):
            self.selected_summon_over_source_pressure += 1
        if bool(payload.get("selected_zero_intent_summon_over_attacking_source")):
            self.selected_zero_intent_summon_over_attacking_source += 1
        if bool(payload.get("cross_card_lethal_source_available")):
            self.cross_card_lethal_source_available += 1
        if bool(payload.get("selected_summon_over_cross_card_lethal_source")):
            self.selected_summon_over_cross_card_lethal_source += 1
        if _text(payload.get("source_pressure_exception_reason")):
            self.source_pressure_exception += 1
        self.source_pressure_candidate_count_sum += _safe_float(
            payload.get("source_pressure_candidate_count"), 0.0
        )
        self.source_damage_lost_sum += _safe_float(payload.get("source_damage_lost"), 0.0)
        self.summon_overkill_sum += _safe_float(payload.get("summon_overkill"), 0.0)

    def as_metadata(self) -> dict[str, float]:
        seen_safe = max(int(self.seen), 1)
        pressure_safe = max(int(self.source_pressure_available), 1)
        cross_card_lethal_safe = max(int(self.cross_card_lethal_source_available), 1)
        return {
            "target_priority_seen_count": float(self.seen),
            "target_priority_source_present_rate": float(self.source_present) / float(seen_safe),
            "target_priority_summon_present_rate": float(self.summon_present) / float(seen_safe),
            "target_priority_selected_source_rate": float(self.selected_source) / float(seen_safe),
            "target_priority_selected_summon_rate": float(self.selected_summon) / float(seen_safe),
            "target_priority_source_pressure_available_rate": (
                float(self.source_pressure_available) / float(seen_safe)
            ),
            "target_priority_selected_summon_over_source_pressure_rate": (
                float(self.selected_summon_over_source_pressure) / float(pressure_safe)
            ),
            "target_priority_selected_zero_intent_summon_over_attacking_source_rate": (
                float(self.selected_zero_intent_summon_over_attacking_source) / float(pressure_safe)
            ),
            "target_priority_source_pressure_exception_rate": (
                float(self.source_pressure_exception) / float(seen_safe)
            ),
            "target_priority_source_pressure_candidate_count_mean": (
                float(self.source_pressure_candidate_count_sum) / float(seen_safe)
            ),
            "target_priority_source_damage_lost_mean": float(self.source_damage_lost_sum) / float(seen_safe),
            "target_priority_summon_overkill_mean": float(self.summon_overkill_sum) / float(seen_safe),
            "target_priority_cross_card_lethal_source_available_rate": (
                float(self.cross_card_lethal_source_available) / float(seen_safe)
            ),
            "target_priority_selected_summon_over_cross_card_lethal_source_rate": (
                float(self.selected_summon_over_cross_card_lethal_source)
                / float(cross_card_lethal_safe)
            ),
        }


def dump_target_priority_diagnostic(trainer: Any, payload: dict[str, Any] | None) -> None:
    """Append one bounded ``target_priority.jsonl`` record using trainer paths."""

    if not isinstance(payload, dict):
        return
    if getattr(trainer, "_target_priority_dump_disabled", False):
        return
    cap = int(getattr(trainer, "_target_priority_dump_max", 100000))
    count = int(getattr(trainer, "_target_priority_dump_count", 0) or 0)
    if cap > 0 and count >= cap:
        return
    path_getter = getattr(trainer, "_diagnostic_jsonl_path", None)
    if not callable(path_getter):
        return
    try:
        record = {
            "time": time.time(),
            "global_step": int(getattr(trainer, "total_steps", 0)),
            "episode_id": int(getattr(trainer, "episode_count", 0)),
            **payload,
        }
        path = path_getter("target_priority.jsonl")
        if path is None:
            return
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("a", encoding="utf-8") as fh:
            fh.write(json.dumps(record, ensure_ascii=False, default=str) + "\n")
        trainer._target_priority_dump_count = count + 1
    except Exception:
        return


__all__ = [
    "TARGET_PRIORITY_TB_KEYS",
    "TargetPriorityEpisodeTracker",
    "best_source_pressure_candidate",
    "build_target_priority_payload",
    "dump_target_priority_diagnostic",
    "source_pressure_decision",
]
