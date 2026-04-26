"""Static potion timing profiles — Phase 1 of potion-timing-modeling-plan.md.

Loads the merged registry (auto-generated base + curated overrides) and
exposes a dict-like API that the rest of the codebase consumes:

    >>> from sts2_env.potion_profiles import (
    ...     get_potion_profile, get_potion_metadata,
    ...     potion_is_enabled_for_training, all_potion_ids,
    ... )

Design goals (per docs/potion-timing-modeling-plan.md §3, §4):
  * Single source of truth for the structured potion profile.  Bridge
    `BuildPotionPayload()` (Phase 2) and Python `_potion_timing_profile()`
    (Phase 4) BOTH consume this so the two sides cannot drift.
  * Registry covers all 64 potions.  POTION.DEPRECATED_POTION is marked
    `enabled_for_training=False` so the trainer ignores it as a positive
    example.
  * Schema is stable — adding new effect_profile slots is additive; old
    consumers tolerate missing keys via DEFAULT_EFFECT_PROFILE merging.

This module does NOT compute per-step timing (`use_quality / waste_risk
/ save_value`) — that is Phase 4 in `_potion_timing_profile()`.
"""

from __future__ import annotations

import json
import os
from functools import lru_cache
from typing import Any

_HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_CONTENT_DIR = os.path.join(_HERE, "content")

GENERATED_PATH = os.path.join(_CONTENT_DIR, "potions.timing.generated.json")
OVERRIDES_PATH = os.path.join(_CONTENT_DIR, "potions.timing.overrides.json")
STATIC_POTIONS_PATH = os.path.join(_CONTENT_DIR, "potions.static.generated.json")


DEFAULT_EFFECT_PROFILE: dict[str, Any] = {
    "damage": 0.0,
    "block": 0.0,
    "draw": 0.0,
    "energy_gain": 0.0,
    "heal": 0.0,
    "weak": 0.0,
    "vulnerable": 0.0,
    "poison": 0.0,
    "strength": 0.0,
    "dexterity": 0.0,
    "intangible": 0.0,
    "prevent_damage": 0.0,
    "generate_card_count": 0.0,
    "discover_count": 0.0,
    "upgrade_hand": 0.0,
    "duplicate_next": 0.0,
    "retrieve_from_discard": 0.0,
    "replace_or_transform_hand": 0.0,
    "aoe": False,
    "single_target": False,
    "random_target": False,
    "target_required": False,
    "can_change_facing_if_targeted_enemy": False,
    "requires_followup": False,
    "long_term_value": False,
    "passive_or_triggered": False,
}


DEFAULT_PROFILE_ENTRY: dict[str, Any] = {
    "id": "",
    "title": "",
    "rarity": "Unknown",
    "target_scope": "Self",
    "enabled_for_training": True,
    "effect_family": [],
    "effect_profile": DEFAULT_EFFECT_PROFILE.copy(),
    "semantic_tags": [],
    "timing_tags": [],
    "training_tags": [],
}


def _load_json_if_exists(path: str) -> dict[str, Any]:
    if not os.path.exists(path):
        return {}
    try:
        with open(path, "r", encoding="utf-8") as fh:
            data = json.load(fh)
    except Exception:
        return {}
    return data if isinstance(data, dict) else {}


def _merge_effect_profile(base: dict[str, Any], override: dict[str, Any] | None) -> dict[str, Any]:
    merged = dict(DEFAULT_EFFECT_PROFILE)
    merged.update(base or {})
    if isinstance(override, dict):
        merged.update(override)
    return merged


def _merge_entry(static_meta: dict[str, Any], gen_entry: dict[str, Any], override_entry: dict[str, Any]) -> dict[str, Any]:
    out = dict(DEFAULT_PROFILE_ENTRY)
    out["effect_profile"] = dict(DEFAULT_EFFECT_PROFILE)
    if isinstance(static_meta, dict):
        for key in ("id", "title", "rarity"):
            if static_meta.get(key):
                out[key] = static_meta[key]
    for source in (gen_entry, override_entry):
        if not isinstance(source, dict):
            continue
        for key, val in source.items():
            if key == "effect_profile":
                out["effect_profile"] = _merge_effect_profile(out["effect_profile"], val)
            else:
                out[key] = val
    if not out.get("id") and static_meta.get("id"):
        out["id"] = static_meta["id"]
    return out


@lru_cache(maxsize=1)
def _build_registry() -> dict[str, dict[str, Any]]:
    static = _load_json_if_exists(STATIC_POTIONS_PATH)
    generated = _load_json_if_exists(GENERATED_PATH)
    overrides = _load_json_if_exists(OVERRIDES_PATH)
    out: dict[str, dict[str, Any]] = {}
    seen_ids: set[str] = set()
    for source in (static, generated, overrides):
        for key in source.keys():
            if not key.startswith("POTION."):
                continue
            seen_ids.add(key)
    for pid in sorted(seen_ids):
        out[pid] = _merge_entry(
            static.get(pid, {}) if isinstance(static, dict) else {},
            generated.get(pid, {}) if isinstance(generated, dict) else {},
            overrides.get(pid, {}) if isinstance(overrides, dict) else {},
        )
    return out


def get_potion_profile(potion_id: str | None) -> dict[str, Any]:
    """Return the merged profile entry for a potion id (case-insensitive ish)."""
    if not potion_id:
        return DEFAULT_PROFILE_ENTRY.copy()
    pid = str(potion_id).strip()
    registry = _build_registry()
    if pid in registry:
        return registry[pid]
    upper = pid.upper()
    if upper in registry:
        return registry[upper]
    return DEFAULT_PROFILE_ENTRY.copy()


def get_potion_metadata(potion_id: str | None) -> dict[str, Any]:
    """Lightweight surface for code paths that only need title/rarity/family."""
    profile = get_potion_profile(potion_id)
    return {
        "id": profile.get("id", ""),
        "title": profile.get("title", ""),
        "rarity": profile.get("rarity", "Unknown"),
        "target_scope": profile.get("target_scope", "Self"),
        "enabled_for_training": bool(profile.get("enabled_for_training", True)),
        "effect_family": list(profile.get("effect_family", []) or []),
        "semantic_tags": list(profile.get("semantic_tags", []) or []),
        "timing_tags": list(profile.get("timing_tags", []) or []),
    }


def potion_is_enabled_for_training(potion_id: str | None) -> bool:
    profile = get_potion_profile(potion_id)
    return bool(profile.get("enabled_for_training", True))


def all_potion_ids() -> list[str]:
    return sorted(_build_registry().keys())


def reload_registry() -> None:
    """Clear the LRU cache; next call rebuilds from disk.  Used by tests."""
    _build_registry.cache_clear()
