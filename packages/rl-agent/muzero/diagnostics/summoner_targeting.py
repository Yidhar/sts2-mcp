"""Summoner/minion target-priority diagnostics for combat decisions.

This module is intentionally diagnostic-first.  Recent full-run traces suggest
the policy can waste lethal windows by attacking summons/minions while the
summoner/source enemy is killable.  Death slices currently show the selected
target, but not the full candidate set needed to prove whether a lethal
summoner alternative existed.  The helpers here produce compact JSONL payloads
for those decisions without adding more logic to ``self_play.py``.
"""

from __future__ import annotations

from dataclasses import dataclass
import json
import math
import time
from typing import Any

import numpy as np

from muzero.strategy import action_features
from muzero.strategy.encounters.kaiser import action_target_combat_id


SUMMONER_TARGETING_TB_KEYS: tuple[tuple[str, str], ...] = (
    ("seen", "summoner_targeting_seen_count"),
    ("summoner_present_rate", "summoner_targeting_summoner_present_rate"),
    ("summon_present_rate", "summoner_targeting_summon_present_rate"),
    ("selected_summoner_rate", "summoner_targeting_selected_summoner_rate"),
    ("selected_summon_rate", "summoner_targeting_selected_summon_rate"),
    ("lethal_summoner_available_rate", "summoner_targeting_lethal_summoner_available_rate"),
    (
        "selected_summoner_when_lethal_available_rate",
        "summoner_targeting_selected_summoner_when_lethal_available_rate",
    ),
    (
        "selected_summon_over_lethal_summoner_rate",
        "summoner_targeting_selected_summon_over_lethal_summoner_rate",
    ),
    (
        "selected_non_summoner_over_lethal_summoner_rate",
        "summoner_targeting_selected_non_summoner_over_lethal_summoner_rate",
    ),
    (
        "cross_card_lethal_summoner_available_rate",
        "summoner_targeting_cross_card_lethal_summoner_available_rate",
    ),
    (
        "selected_summon_over_cross_card_lethal_summoner_rate",
        "summoner_targeting_selected_summon_over_cross_card_lethal_summoner_rate",
    ),
    (
        "selected_non_summoner_over_cross_card_lethal_summoner_rate",
        "summoner_targeting_selected_non_summoner_over_cross_card_lethal_summoner_rate",
    ),
    ("exception_rate", "summoner_targeting_exception_rate"),
    ("candidate_attack_count_mean", "summoner_targeting_candidate_attack_count_mean"),
)


SUMMONER_TOKENS = (
    "fogmog",
    "living_fog",
    "living fog",
    "flyconid",
    "summoner",
    "spawner",
    "雾菇",
    "活雾",
    "飞蝇菌子",
    "召唤师",
    "召唤者",
)

SUMMON_TOKENS = (
    "gas bomb",
    "gaseous bomb",
    "toothed eye",
    "fanged eye",
    "leaf slime",
    "spawn",
    "minion",
    "summon",
    "bomb",
    "气态炸弹",
    "利齿之眼",
    "树叶史莱姆",
    "召唤物",
    "小怪",
    "炸弹",
)


def _safe_float(value: Any, default: float = 0.0) -> float:
    try:
        out = float(value)
    except (TypeError, ValueError):
        return float(default)
    return out if math.isfinite(out) else float(default)


def _safe_int(value: Any) -> int | None:
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _safe_bool(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        return bool(value)
    if isinstance(value, str):
        return value.strip().lower() in {"1", "true", "yes", "y", "on"}
    return bool(value)


def _lower(value: Any) -> str:
    return str(value or "").strip().lower()


def _compact_dict(d: dict[str, Any]) -> dict[str, Any]:
    return {k: v for k, v in d.items() if v not in (None, "", [], {})}


def _nested_dict(obj: Any, key: str) -> dict[str, Any]:
    if isinstance(obj, dict) and isinstance(obj.get(key), dict):
        return obj.get(key)  # type: ignore[return-value]
    return {}


def _text_blob(obj: Any) -> str:
    if not isinstance(obj, dict):
        return ""
    parts: list[str] = []
    containers = [obj]
    for key in ("enemy", "monster", "target", "card", "semantic"):
        nested = obj.get(key)
        if isinstance(nested, dict):
            containers.append(nested)
    for container in containers:
        for key in (
            "id",
            "enemy_id",
            "model_id",
            "monster_id",
            "name",
            "title",
            "label",
            "localized_name",
            "encounter_id",
            "kind",
            "type",
            "role",
            "tags",
        ):
            value = container.get(key)
            if isinstance(value, list):
                parts.extend(str(x or "") for x in value)
            elif value is not None:
                parts.append(str(value))
    return " ".join(parts).strip().lower()


def _has_any_token(text: str, tokens: tuple[str, ...]) -> bool:
    return any(token in text for token in tokens)


def is_summoner_enemy(enemy: Any) -> bool:
    """Best-effort source/summoner classifier.

    The bridge does not currently expose a stable "summoner" flag for every
    encounter.  Prefer explicit booleans when available and fall back to a
    conservative name/id token table for known Act1 summon-style fights.
    """

    if not isinstance(enemy, dict):
        return False
    for key in ("is_summoner", "summoner", "spawns_minions", "can_summon"):
        if key in enemy and _safe_bool(enemy.get(key)):
            return True
    return _has_any_token(_text_blob(enemy), SUMMONER_TOKENS)


def is_summon_enemy(enemy: Any) -> bool:
    """Best-effort summon/minion classifier."""

    if not isinstance(enemy, dict):
        return False
    for key in ("is_summon", "is_minion", "summoned", "spawned", "minion"):
        if key in enemy and _safe_bool(enemy.get(key)):
            return True
    return _has_any_token(_text_blob(enemy), SUMMON_TOKENS)


def _combat(raw_obs: Any) -> dict[str, Any]:
    if not isinstance(raw_obs, dict):
        return {}
    combat = raw_obs.get("combat")
    return combat if isinstance(combat, dict) else {}


def _combat_enemies(raw_obs: Any) -> list[dict[str, Any]]:
    combat = _combat(raw_obs)
    for key in ("enemies", "monsters", "creatures"):
        enemies = combat.get(key)
        if isinstance(enemies, list):
            return [enemy for enemy in enemies if isinstance(enemy, dict)]
    return []


def _enemy_combat_id(enemy: dict[str, Any]) -> int | None:
    for key in ("combat_id", "id", "target_combat_id", "monster_index", "index"):
        if enemy.get(key) is None:
            continue
        parsed = _safe_int(enemy.get(key))
        if parsed is not None:
            return parsed
    return None


def _enemy_name(enemy: dict[str, Any]) -> str:
    for key in ("name", "title", "label", "localized_name", "id", "enemy_id", "model_id"):
        value = enemy.get(key)
        if value:
            return str(value)
    return ""


def _enemy_hp(enemy: dict[str, Any]) -> float:
    return max(
        _safe_float(enemy.get("hp")),
        _safe_float(enemy.get("current_hp")),
        _safe_float(enemy.get("health")),
        _safe_float(enemy.get("currentHealth")),
    )


def _enemy_block(enemy: dict[str, Any]) -> float:
    return max(
        _safe_float(enemy.get("block")),
        _safe_float(enemy.get("current_block")),
        _safe_float(enemy.get("armor")),
    )


def _enemy_intent_damage(enemy: dict[str, Any]) -> float:
    intent = _nested_dict(enemy, "intent")
    return max(
        _safe_float(enemy.get("intent_damage")),
        _safe_float(enemy.get("total_damage")),
        _safe_float(enemy.get("attack_damage")),
        _safe_float(intent.get("total_damage")),
        _safe_float(intent.get("damage")),
        _safe_float(intent.get("attack_damage")),
    )


def _enemy_alive(enemy: dict[str, Any]) -> bool:
    if _safe_bool(enemy.get("is_dead")) or _safe_bool(enemy.get("dead")):
        return False
    if enemy.get("alive") is False:
        return False
    return _enemy_hp(enemy) > 0.0


def _target_name(action: Any) -> str:
    if not isinstance(action, dict):
        return ""
    target = _nested_dict(action, "target")
    for container in (action, target):
        for key in ("target_name", "target_title", "name", "title", "label"):
            value = container.get(key)
            if value:
                return str(value)
    return ""


def _target_enemy(action: Any, enemies: list[dict[str, Any]]) -> dict[str, Any] | None:
    if not isinstance(action, dict):
        return None
    target_id = action_target_combat_id(action)
    target_name = _lower(_target_name(action))
    if target_id is not None:
        for enemy in enemies:
            enemy_id = _enemy_combat_id(enemy)
            if enemy_id is not None and int(enemy_id) == int(target_id):
                return enemy
    if target_name:
        for enemy in enemies:
            enemy_name = _lower(_enemy_name(enemy))
            if enemy_name and (target_name == enemy_name or target_name in enemy_name or enemy_name in target_name):
                return enemy
    return None


def _action_source(action: Any) -> dict[str, Any]:
    return action_features.action_source(action)


def _action_title(action: Any) -> str:
    if not isinstance(action, dict):
        return ""
    source = _action_source(action)
    card = action.get("card") if isinstance(action.get("card"), dict) else {}
    # Prefer card/source semantic labels over the compact top-level action_id
    # (e.g. ``play_card:1:1``).  The action id is useful as a last-resort
    # fallback, but treating it as the title hides the actual card in
    # summoner/source lethal diagnostics.
    for container in (card, source, action):
        if not isinstance(container, dict):
            continue
        for key in ("title", "card_title", "name", "label"):
            value = container.get(key)
            if value:
                return str(value)
    for container in (card, source, action):
        if not isinstance(container, dict):
            continue
        for key in ("id", "card_id"):
            value = container.get(key)
            if value:
                return str(value)
    value = action.get("action_id")
    if value:
        return str(value)
    return ""


def _action_card_id(action: Any) -> str:
    if not isinstance(action, dict):
        return ""
    source = _action_source(action)
    for container in (action, source):
        if not isinstance(container, dict):
            continue
        for key in ("card_id", "id", "uuid", "model_id"):
            value = container.get(key)
            if value:
                return str(value)
    return ""


def _numeric_from_action(action: Any, keys: tuple[str, ...]) -> float:
    if not isinstance(action, dict):
        return 0.0
    source = _action_source(action)
    semantic = _nested_dict(action, "semantic")
    profile = _nested_dict(action, "card_effect_profile")
    source_profile = _nested_dict(source, "card_effect_profile")
    semantic_profile = _nested_dict(semantic, "card_effect_profile")
    best = 0.0
    for container in (action, semantic, profile, source, source_profile, semantic_profile):
        if not isinstance(container, dict):
            continue
        for key in keys:
            if key not in container:
                continue
            value = container.get(key)
            if isinstance(value, (dict, list)):
                continue
            best = max(best, _safe_float(value, 0.0))
    return float(best)


def _action_damage(action: Any) -> float:
    return max(
        action_features.action_metric(action, "damage"),
        _numeric_from_action(
            action,
            (
                "damage",
                "total_damage",
                "preview_damage",
                "expected_damage",
                "typed_damage_amount",
                "damage_amount",
            ),
        ),
    )


def _is_aoe_action(action: Any) -> bool:
    if not isinstance(action, dict):
        return False
    source = _action_source(action)
    roles = action_features.action_roles(action)
    text = " ".join(
        str(x or "")
        for x in (
            action.get("target_type"),
            action.get("target"),
            source.get("target_type"),
            source.get("target"),
            action.get("action_id"),
            action.get("label"),
        )
    ).lower()
    return bool(
        roles.intersection({"aoe", "all_enemies"})
        or "all_enem" in text
        or "allenemy" in text
        or "all enemy" in text
        or "all enemies" in text
        or "所有敌" in text
    )


def _is_attack_candidate(action: Any) -> bool:
    if not isinstance(action, dict):
        return False
    family = action_features.semantic_family(action)
    if family not in {"play_card", "use_potion", "combat", "attack"}:
        return False
    source = _action_source(action)
    card_type = _lower(action.get("card_type") or source.get("type"))
    roles = action_features.action_roles(action)
    return bool(
        _action_damage(action) > 0.0
        or card_type == "attack"
        or roles.intersection({"attack", "damage"})
    )


def _legal_mask_array(action_mask: Any, legal_count: int) -> np.ndarray:
    try:
        mask_np = np.asarray(action_mask, dtype=np.float32).reshape(-1)
    except Exception:
        mask_np = np.ones((legal_count,), dtype=np.float32)
    if mask_np.shape[0] < legal_count:
        padded = np.zeros((legal_count,), dtype=np.float32)
        padded[: mask_np.shape[0]] = mask_np
        return padded
    return mask_np[:legal_count]


def _policy_prob(search_policy: Any, idx: int) -> float | None:
    try:
        arr = np.asarray(search_policy, dtype=np.float32).reshape(-1)
    except Exception:
        return None
    if 0 <= int(idx) < arr.shape[0]:
        return float(arr[int(idx)])
    return None


def _player_summary(raw_obs: Any) -> dict[str, Any]:
    obs = raw_obs if isinstance(raw_obs, dict) else {}
    combat = _combat(raw_obs)
    player = obs.get("player") if isinstance(obs.get("player"), dict) else {}
    hp = max(_safe_float(player.get("hp")), _safe_float(player.get("current_hp")), _safe_float(combat.get("hp")))
    max_hp = max(_safe_float(player.get("max_hp")), _safe_float(player.get("maxHealth")), hp)
    block = max(_safe_float(player.get("block")), _safe_float(player.get("current_block")), _safe_float(combat.get("block")))
    energy = max(_safe_float(combat.get("energy")), _safe_float(player.get("energy")))
    incoming = max(_safe_float(combat.get("incoming_damage")), _safe_float(combat.get("total_incoming_damage")))
    return {
        "hp": float(hp),
        "max_hp": float(max_hp),
        "block": float(block),
        "energy": float(energy),
        "incoming": float(incoming),
    }


def _enemy_payload(enemy: dict[str, Any]) -> dict[str, Any]:
    return _compact_dict(
        {
            "combat_id": _enemy_combat_id(enemy),
            "name": _enemy_name(enemy),
            "hp": float(_enemy_hp(enemy)),
            "block": float(_enemy_block(enemy)),
            "intent_damage": float(_enemy_intent_damage(enemy)),
            "is_summoner": bool(is_summoner_enemy(enemy)),
            "is_summon": bool(is_summon_enemy(enemy)),
            "alive": bool(_enemy_alive(enemy)),
        }
    )


def _candidate_payload(
    *,
    idx: int,
    action: Any,
    target: dict[str, Any] | None,
    search_policy: Any,
) -> dict[str, Any]:
    damage = float(_action_damage(action))
    target_hp = float(_enemy_hp(target)) if isinstance(target, dict) else 0.0
    target_block = float(_enemy_block(target)) if isinstance(target, dict) else 0.0
    kills_hp_only = bool(target_hp > 0.0 and damage >= max(target_hp, 0.0) - 1e-6)
    kills_block_aware = bool(target_hp > 0.0 and damage >= max(target_hp + target_block, target_hp) - 1e-6)
    target_is_summoner = bool(is_summoner_enemy(target)) if isinstance(target, dict) else False
    target_is_summon = bool(is_summon_enemy(target)) if isinstance(target, dict) else False
    return _compact_dict(
        {
            "index": int(idx),
            "policy": _policy_prob(search_policy, idx),
            "family": action_features.semantic_family(action),
            "card_id": _action_card_id(action),
            "card_title": _action_title(action),
            "damage": damage,
            "aoe": bool(_is_aoe_action(action)),
            "target_combat_id": _enemy_combat_id(target) if isinstance(target, dict) else action_target_combat_id(action),
            "target_name": _enemy_name(target) if isinstance(target, dict) else _target_name(action),
            "target_hp": target_hp,
            "target_block": target_block,
            "kills_target_hp_only": kills_hp_only,
            "kills_target_block_aware": kills_block_aware,
            # For now use the hp-only criterion for visibility, because bridge
            # preview damage is often already post-block.  The block-aware flag is
            # included so later hard guards can choose the stricter condition.
            "kills_target": kills_hp_only,
            "target_is_summoner": target_is_summoner,
            "target_is_summon": target_is_summon,
            "target_intent_damage": float(_enemy_intent_damage(target)) if isinstance(target, dict) else 0.0,
        }
    )


def _candidate_identity(candidate: dict[str, Any] | None) -> tuple[str, str]:
    """Return a stable best-effort card/action-source identity.

    Target-priority guards intentionally stay conservative and often require a
    same-card retarget.  Diagnostics need the opposite visibility too: user
    traces can show "played a summon attack while a *different* card could kill
    the summoner".  This helper lets us split lethal-summoner opportunities into
    same-source and cross-card buckets without relying on raw action indices.
    """

    if not isinstance(candidate, dict):
        return "", ""
    family = _lower(candidate.get("family"))
    card_id = _lower(candidate.get("card_id"))
    title = _lower(candidate.get("card_title"))
    if card_id:
        return family, f"id:{card_id}"
    if title:
        return family, f"title:{title}"
    return family, ""


def _same_candidate_source(a: dict[str, Any] | None, b: dict[str, Any] | None) -> bool:
    fam_a, src_a = _candidate_identity(a)
    fam_b, src_b = _candidate_identity(b)
    if fam_a and fam_b and fam_a != fam_b:
        return False
    if src_a and src_b:
        return src_a == src_b
    # If either source is missing, do not call it "same"; false positives here
    # would hide the cross-card mistake bucket we are trying to expose.
    return False


def _best_candidate(candidates: list[dict[str, Any]]) -> dict[str, Any]:
    if not candidates:
        return {}
    rows = list(candidates)
    rows.sort(
        key=lambda c: (
            bool(c.get("kills_target")),
            bool(c.get("target_is_summoner")),
            float(c.get("damage") or 0.0),
            float(c.get("policy") or 0.0),
            -int(c.get("index", 9999) or 9999),
        ),
        reverse=True,
    )
    return rows[0]


def build_summoner_targeting_payload(
    *,
    raw_obs: Any,
    legal_actions: list[Any],
    action_mask: Any,
    selected_idx: int,
    search_policy: Any = None,
    progress: dict[str, Any] | None = None,
    encounter_id: str = "",
    encounter_tier: str = "",
    max_candidates: int = 16,
) -> dict[str, Any] | None:
    """Return a compact diagnostic payload for summon/summoner target choices.

    The function returns ``None`` for ordinary single-enemy or non-summon fights
    to keep logs small.  It does not mutate policy/action selection.
    """

    if not isinstance(legal_actions, list) or not isinstance(raw_obs, dict):
        return None
    enemies = [enemy for enemy in _combat_enemies(raw_obs) if _enemy_alive(enemy)]
    if len(enemies) < 2:
        return None
    enemy_rows = [_enemy_payload(enemy) for enemy in enemies]
    summoner_present = any(row.get("is_summoner") for row in enemy_rows)
    summon_present = any(row.get("is_summon") for row in enemy_rows)
    encounter_text = _lower(encounter_id or (_combat(raw_obs).get("encounter_id") or _combat(raw_obs).get("encounter")))
    encounter_summon_related = _has_any_token(encounter_text, SUMMONER_TOKENS + SUMMON_TOKENS)
    if not (summoner_present or summon_present or encounter_summon_related):
        return None

    legal_count = min(len(legal_actions), max(int(max_candidates), 1) if max_candidates > 0 else len(legal_actions))
    mask_np = _legal_mask_array(action_mask, len(legal_actions))
    candidates: list[dict[str, Any]] = []
    for idx, action in enumerate(legal_actions):
        if idx >= mask_np.shape[0] or float(mask_np[idx]) <= 0.0:
            continue
        if not _is_attack_candidate(action):
            continue
        target = _target_enemy(action, enemies)
        payload = _candidate_payload(idx=idx, action=action, target=target, search_policy=search_policy)
        candidates.append(payload)

    if not candidates:
        return None

    selected_action = legal_actions[int(selected_idx)] if 0 <= int(selected_idx) < len(legal_actions) else None
    selected_target = _target_enemy(selected_action, enemies) if isinstance(selected_action, dict) else None
    selected_candidate = None
    for candidate in candidates:
        if int(candidate.get("index", -1)) == int(selected_idx):
            selected_candidate = candidate
            break
    if selected_candidate is None and isinstance(selected_action, dict):
        selected_candidate = _candidate_payload(
            idx=int(selected_idx),
            action=selected_action,
            target=selected_target,
            search_policy=search_policy,
        )

    lethal_summoner_candidates = [
        c
        for c in candidates
        if bool(c.get("target_is_summoner")) and bool(c.get("kills_target"))
    ]
    lethal_summoner_available = bool(lethal_summoner_candidates)

    selected_is_attack = bool(isinstance(selected_candidate, dict) and _is_attack_candidate(selected_action))
    selected_target_is_summoner = bool((selected_candidate or {}).get("target_is_summoner"))
    selected_target_is_summon = bool((selected_candidate or {}).get("target_is_summon"))
    selected_kills_target = bool((selected_candidate or {}).get("kills_target"))
    selected_aoe = bool((selected_candidate or {}).get("aoe"))
    cross_card_lethal_summoner_candidates = [
        c
        for c in lethal_summoner_candidates
        if not _same_candidate_source(selected_candidate, c)
    ]
    cross_card_lethal_summoner_available = bool(cross_card_lethal_summoner_candidates)

    player = _player_summary(raw_obs)
    selected_summon_lethal_incoming = False
    if selected_target_is_summon and selected_kills_target and isinstance(selected_target, dict):
        threat_after_block = max(0.0, _enemy_intent_damage(selected_target) - float(player.get("block", 0.0)))
        selected_summon_lethal_incoming = bool(threat_after_block >= max(float(player.get("hp", 0.0)), 1.0))

    exception_reason = ""
    if selected_aoe:
        exception_reason = "selected_aoe"
    elif selected_summon_lethal_incoming:
        exception_reason = "summon_lethal_incoming"

    selected_summoner_when_lethal_available = bool(
        lethal_summoner_available and selected_target_is_summoner and selected_kills_target
    )
    selected_summon_over_lethal_summoner = bool(
        lethal_summoner_available
        and selected_target_is_summon
        and not selected_summoner_when_lethal_available
        and not exception_reason
    )
    selected_non_summoner_over_lethal_summoner = bool(
        lethal_summoner_available
        and selected_is_attack
        and not selected_target_is_summoner
        and not selected_target_is_summon
        and not exception_reason
    )
    selected_summon_over_cross_card_lethal_summoner = bool(
        cross_card_lethal_summoner_available
        and selected_target_is_summon
        and not selected_summoner_when_lethal_available
        and not exception_reason
    )
    selected_non_summoner_over_cross_card_lethal_summoner = bool(
        cross_card_lethal_summoner_available
        and selected_is_attack
        and not selected_target_is_summoner
        and not selected_target_is_summon
        and not exception_reason
    )

    candidates_sorted = sorted(
        candidates,
        key=lambda c: (
            not bool(c.get("target_is_summoner")),
            not bool(c.get("kills_target")),
            -float(c.get("policy") or 0.0),
            int(c.get("index", 9999)),
        ),
    )

    payload = {
        "schema": "summoner_targeting_v1",
        "floor": (progress or {}).get("floor"),
        "act_id": (progress or {}).get("act_id"),
        "room_type": (progress or {}).get("room_type"),
        "encounter_id": encounter_id or encounter_text,
        "encounter_tier": encounter_tier,
        "summoner_present": bool(summoner_present),
        "summon_present": bool(summon_present),
        "enemy_count": int(len(enemy_rows)),
        "enemies": enemy_rows[:8],
        "combat": player,
        "selected_action": selected_candidate or {"index": int(selected_idx)},
        "candidate_attack_count": int(len(candidates)),
        "candidate_attacks": candidates_sorted[: max(int(max_candidates), 1)],
        "lethal_summoner_available": bool(lethal_summoner_available),
        "lethal_summoner_candidate_count": int(len(lethal_summoner_candidates)),
        "cross_card_lethal_summoner_available": bool(cross_card_lethal_summoner_available),
        "cross_card_lethal_summoner_candidate_count": int(len(cross_card_lethal_summoner_candidates)),
        "best_lethal_summoner_candidate": _best_candidate(lethal_summoner_candidates),
        "best_cross_card_lethal_summoner_candidate": _best_candidate(cross_card_lethal_summoner_candidates),
        "selected_summoner_when_lethal_available": bool(selected_summoner_when_lethal_available),
        "selected_summon_over_lethal_summoner": bool(selected_summon_over_lethal_summoner),
        "selected_non_summoner_over_lethal_summoner": bool(selected_non_summoner_over_lethal_summoner),
        "selected_summon_over_cross_card_lethal_summoner": bool(
            selected_summon_over_cross_card_lethal_summoner
        ),
        "selected_non_summoner_over_cross_card_lethal_summoner": bool(
            selected_non_summoner_over_cross_card_lethal_summoner
        ),
        "exception_reason": exception_reason,
    }
    return _compact_dict(payload)


@dataclass
class SummonerTargetingEpisodeTracker:
    seen: int = 0
    summoner_present: int = 0
    summon_present: int = 0
    selected_summoner: int = 0
    selected_summon: int = 0
    lethal_summoner_available: int = 0
    selected_summoner_when_lethal_available: int = 0
    selected_summon_over_lethal_summoner: int = 0
    selected_non_summoner_over_lethal_summoner: int = 0
    cross_card_lethal_summoner_available: int = 0
    selected_summon_over_cross_card_lethal_summoner: int = 0
    selected_non_summoner_over_cross_card_lethal_summoner: int = 0
    exception_count: int = 0
    candidate_attack_count_sum: float = 0.0

    def update(self, payload: dict[str, Any] | None) -> None:
        if not isinstance(payload, dict):
            return
        self.seen += 1
        if _safe_bool(payload.get("summoner_present")):
            self.summoner_present += 1
        if _safe_bool(payload.get("summon_present")):
            self.summon_present += 1
        selected = payload.get("selected_action") if isinstance(payload.get("selected_action"), dict) else {}
        if _safe_bool(selected.get("target_is_summoner")):
            self.selected_summoner += 1
        if _safe_bool(selected.get("target_is_summon")):
            self.selected_summon += 1
        if _safe_bool(payload.get("lethal_summoner_available")):
            self.lethal_summoner_available += 1
        if _safe_bool(payload.get("selected_summoner_when_lethal_available")):
            self.selected_summoner_when_lethal_available += 1
        if _safe_bool(payload.get("selected_summon_over_lethal_summoner")):
            self.selected_summon_over_lethal_summoner += 1
        if _safe_bool(payload.get("selected_non_summoner_over_lethal_summoner")):
            self.selected_non_summoner_over_lethal_summoner += 1
        if _safe_bool(payload.get("cross_card_lethal_summoner_available")):
            self.cross_card_lethal_summoner_available += 1
        if _safe_bool(payload.get("selected_summon_over_cross_card_lethal_summoner")):
            self.selected_summon_over_cross_card_lethal_summoner += 1
        if _safe_bool(payload.get("selected_non_summoner_over_cross_card_lethal_summoner")):
            self.selected_non_summoner_over_cross_card_lethal_summoner += 1
        if str(payload.get("exception_reason") or "").strip():
            self.exception_count += 1
        self.candidate_attack_count_sum += _safe_float(payload.get("candidate_attack_count"), 0.0)

    def as_metadata(self) -> dict[str, float]:
        seen_safe = max(int(self.seen), 1)
        lethal_safe = max(int(self.lethal_summoner_available), 1)
        cross_card_lethal_safe = max(int(self.cross_card_lethal_summoner_available), 1)
        return {
            "summoner_targeting_seen_count": float(self.seen),
            "summoner_targeting_summoner_present_rate": float(self.summoner_present) / float(seen_safe),
            "summoner_targeting_summon_present_rate": float(self.summon_present) / float(seen_safe),
            "summoner_targeting_selected_summoner_rate": float(self.selected_summoner) / float(seen_safe),
            "summoner_targeting_selected_summon_rate": float(self.selected_summon) / float(seen_safe),
            "summoner_targeting_lethal_summoner_available_rate": (
                float(self.lethal_summoner_available) / float(seen_safe)
            ),
            "summoner_targeting_selected_summoner_when_lethal_available_rate": (
                float(self.selected_summoner_when_lethal_available) / float(lethal_safe)
            ),
            "summoner_targeting_selected_summon_over_lethal_summoner_rate": (
                float(self.selected_summon_over_lethal_summoner) / float(lethal_safe)
            ),
            "summoner_targeting_selected_non_summoner_over_lethal_summoner_rate": (
                float(self.selected_non_summoner_over_lethal_summoner) / float(lethal_safe)
            ),
            "summoner_targeting_cross_card_lethal_summoner_available_rate": (
                float(self.cross_card_lethal_summoner_available) / float(seen_safe)
            ),
            "summoner_targeting_selected_summon_over_cross_card_lethal_summoner_rate": (
                float(self.selected_summon_over_cross_card_lethal_summoner)
                / float(cross_card_lethal_safe)
            ),
            "summoner_targeting_selected_non_summoner_over_cross_card_lethal_summoner_rate": (
                float(self.selected_non_summoner_over_cross_card_lethal_summoner)
                / float(cross_card_lethal_safe)
            ),
            "summoner_targeting_exception_rate": float(self.exception_count) / float(seen_safe),
            "summoner_targeting_candidate_attack_count_mean": (
                float(self.candidate_attack_count_sum) / float(seen_safe)
            ),
        }


def dump_summoner_targeting_diagnostic(trainer: Any, payload: dict[str, Any] | None) -> None:
    """Append one bounded ``summoner_targeting.jsonl`` record using trainer paths."""

    if not isinstance(payload, dict):
        return
    if getattr(trainer, "_summoner_targeting_dump_disabled", False):
        return
    cap = int(getattr(trainer, "_summoner_targeting_dump_max", 100000))
    count = int(getattr(trainer, "_summoner_targeting_dump_count", 0) or 0)
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
        path = path_getter("summoner_targeting.jsonl")
        if path is None:
            return
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("a", encoding="utf-8") as fh:
            fh.write(json.dumps(record, ensure_ascii=False, default=str) + "\n")
        trainer._summoner_targeting_dump_count = count + 1
    except Exception:
        return
