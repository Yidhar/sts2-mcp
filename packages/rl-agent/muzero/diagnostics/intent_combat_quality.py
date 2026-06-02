"""Intent-aware combat decision diagnostics.

This module is diagnostic-first.  The current Act1 bottleneck is no longer a
simple "selected EndTurn with full energy" bug: live traces show the model often
spends the *previous* actions badly (ignoring incoming damage, over-blocking
zero-intent turns, or missing lethal), which makes later forced EndTurn frames
look like "空过".  The helpers here record compact per-decision evidence without
turning the trainer or ``train.py`` into another heuristic monolith.
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


INTENT_COMBAT_QUALITY_TB_KEYS: tuple[tuple[str, str], ...] = (
    ("seen", "intent_combat_quality_seen_count"),
    ("high_pressure_rate", "intent_combat_quality_high_pressure_rate"),
    ("no_pressure_rate", "intent_combat_quality_no_pressure_rate"),
    ("lethal_available_rate", "intent_combat_quality_lethal_available_rate"),
    ("selected_lethal_rate", "intent_combat_quality_selected_lethal_rate"),
    ("missed_lethal_rate", "intent_combat_quality_missed_lethal_rate"),
    ("survival_candidate_available_rate", "intent_combat_quality_survival_candidate_available_rate"),
    (
        "selected_nonprotective_under_pressure_rate",
        "intent_combat_quality_selected_nonprotective_under_pressure_rate",
    ),
    ("no_pressure_pure_block_rate", "intent_combat_quality_no_pressure_pure_block_rate"),
    ("selected_end_turn_rate", "intent_combat_quality_selected_end_turn_rate"),
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


def _lower(value: Any) -> str:
    return str(value or "").strip().lower()


def _compact_dict(d: dict[str, Any]) -> dict[str, Any]:
    return {k: v for k, v in d.items() if v not in (None, "", [], {})}


def _nested_dict(obj: Any, key: str) -> dict[str, Any]:
    if isinstance(obj, dict) and isinstance(obj.get(key), dict):
        return obj.get(key)  # type: ignore[return-value]
    return {}


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
    if bool(enemy.get("is_dead")) or bool(enemy.get("dead")):
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
    # Prefer semantic card titles/names over compact action ids.  In compact
    # payloads the top-level ``action_id`` is often just ``play_card:1:0``;
    # using it as the candidate title makes diagnostics unreadable and can
    # break same-card/cross-card attribution tests.  Keep action_id only as a
    # final fallback for payloads that truly have no card/source metadata.
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


def _action_block(action: Any) -> float:
    roles = action_features.action_roles(action)
    value = max(
        action_features.action_metric(action, "block"),
        _numeric_from_action(action, ("block", "total_block", "preview_block", "typed_block_amount")),
    )
    if value <= 0.0 and roles.intersection({"block", "defense", "defend"}):
        value = 5.0
    identity = " ".join(
        str(x or "")
        for x in (
            action.get("action_id") if isinstance(action, dict) else "",
            action.get("label") if isinstance(action, dict) else "",
            _action_title(action),
        )
    ).lower()
    if value <= 0.0 and ("defend" in identity or "防御" in identity):
        value = 5.0
    return float(value)


def _action_heal(action: Any) -> float:
    return max(
        action_features.action_metric(action, "heal"),
        _numeric_from_action(action, ("heal", "healing", "hp_gain", "typed_heal_amount")),
    )


def _action_hp_cost(action: Any) -> float:
    return max(
        action_features.action_metric(action, "hp_loss"),
        action_features.action_metric(action, "hp_cost"),
        _numeric_from_action(action, ("hp_loss", "hp_cost", "typed_hp_loss")),
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
        or "all enemy" in text
        or "all enemies" in text
        or "所有敌" in text
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


def _player_summary(raw_obs: Any) -> dict[str, float]:
    obs = raw_obs if isinstance(raw_obs, dict) else {}
    combat = _combat(raw_obs)
    player = obs.get("player") if isinstance(obs.get("player"), dict) else {}
    hp = max(_safe_float(player.get("hp")), _safe_float(player.get("current_hp")), _safe_float(combat.get("hp")))
    max_hp = max(_safe_float(player.get("max_hp")), _safe_float(player.get("maxHealth")), hp)
    block = max(_safe_float(player.get("block")), _safe_float(player.get("current_block")), _safe_float(combat.get("block")))
    energy = max(_safe_float(combat.get("energy")), _safe_float(player.get("energy")))
    incoming = max(_safe_float(combat.get("incoming_damage")), _safe_float(combat.get("total_incoming_damage")))
    if incoming <= 0.0:
        incoming = sum(_enemy_intent_damage(enemy) for enemy in _combat_enemies(raw_obs) if _enemy_alive(enemy))
    threat_gap = max(0.0, incoming - block)
    hp_ratio = hp / max(max_hp, 1.0)
    return {
        "hp": float(hp),
        "max_hp": float(max_hp),
        "hp_ratio": float(hp_ratio),
        "block": float(block),
        "energy": float(energy),
        "incoming": float(incoming),
        "threat_gap": float(threat_gap),
    }


def _confirmed_lethal(action: Any, enemies: list[dict[str, Any]]) -> bool:
    damage = _action_damage(action)
    if damage <= 0.0:
        return False
    target = _target_enemy(action, enemies)
    if isinstance(target, dict):
        return bool(_enemy_hp(target) > 0.0 and damage >= _enemy_hp(target) - 1e-6)
    if _is_aoe_action(action):
        return any(_enemy_hp(enemy) > 0.0 and damage >= _enemy_hp(enemy) - 1e-6 for enemy in enemies)
    return False


def _action_summary(idx: int, action: Any, enemies: list[dict[str, Any]], search_policy: Any) -> dict[str, Any]:
    family = action_features.semantic_family(action)
    roles = sorted(str(role) for role in action_features.action_roles(action))
    target = _target_enemy(action, enemies)
    damage = _action_damage(action)
    block = _action_block(action)
    heal = _action_heal(action)
    hp_cost = _action_hp_cost(action)
    return _compact_dict(
        {
            "index": int(idx),
            "policy": _policy_prob(search_policy, idx),
            "family": family,
            "roles": roles[:8],
            "card_id": _action_card_id(action),
            "card_title": _action_title(action),
            "damage": float(damage),
            "block": float(block),
            "heal": float(heal),
            "hp_cost": float(hp_cost),
            "aoe": bool(_is_aoe_action(action)),
            "target_combat_id": _enemy_combat_id(target) if isinstance(target, dict) else action_target_combat_id(action),
            "target_name": _enemy_name(target) if isinstance(target, dict) else _target_name(action),
            "target_hp": float(_enemy_hp(target)) if isinstance(target, dict) else 0.0,
            "target_intent_damage": float(_enemy_intent_damage(target)) if isinstance(target, dict) else 0.0,
            "confirmed_lethal": bool(_confirmed_lethal(action, enemies)),
            "is_end_turn": bool(family == "end_turn" or str(action.get("action_id", "") if isinstance(action, dict) else "") == "end_turn"),
        }
    )


def _best_by(rows: list[dict[str, Any]], key_name: str) -> dict[str, Any]:
    if not rows:
        return {}
    if key_name == "lethal":
        key = lambda c: (
            bool(c.get("confirmed_lethal")),
            float(c.get("damage") or 0.0),
            float(c.get("target_hp") or 0.0),
            float(c.get("policy") or 0.0),
            -int(c.get("index", 9999) or 9999),
        )
    elif key_name == "survival":
        key = lambda c: (
            float(c.get("block") or 0.0) + float(c.get("heal") or 0.0) - float(c.get("hp_cost") or 0.0),
            float(c.get("policy") or 0.0),
            -int(c.get("index", 9999) or 9999),
        )
    else:
        key = lambda c: (float(c.get("policy") or 0.0), -int(c.get("index", 9999) or 9999))
    return sorted(rows, key=key, reverse=True)[0]


def build_intent_combat_quality_payload(
    *,
    raw_obs: Any,
    legal_actions: list[Any],
    action_mask: Any,
    selected_idx: int,
    search_policy: Any = None,
    progress: dict[str, Any] | None = None,
    encounter_id: str = "",
    encounter_tier: str = "",
    max_candidates: int = 12,
) -> dict[str, Any] | None:
    """Build a compact intent-vs-action diagnostic payload for one combat step."""

    if not isinstance(raw_obs, dict) or not isinstance(legal_actions, list) or not legal_actions:
        return None
    enemies = [enemy for enemy in _combat_enemies(raw_obs) if _enemy_alive(enemy)]
    player = _player_summary(raw_obs)
    mask_np = _legal_mask_array(action_mask, len(legal_actions))
    candidates: list[dict[str, Any]] = []
    for idx, action in enumerate(legal_actions):
        if idx >= mask_np.shape[0] or float(mask_np[idx]) <= 0.0:
            continue
        if not isinstance(action, dict):
            continue
        family = action_features.semantic_family(action)
        if family not in {"play_card", "use_potion", "combat", "attack", "end_turn"}:
            continue
        candidates.append(_action_summary(idx, action, enemies, search_policy))

    selected_action = legal_actions[int(selected_idx)] if 0 <= int(selected_idx) < len(legal_actions) else None
    selected = _action_summary(int(selected_idx), selected_action, enemies, search_policy) if isinstance(selected_action, dict) else {"index": int(selected_idx)}
    lethal_candidates = [c for c in candidates if bool(c.get("confirmed_lethal"))]
    survival_candidates = [
        c
        for c in candidates
        if (float(c.get("block") or 0.0) + float(c.get("heal") or 0.0) - float(c.get("hp_cost") or 0.0)) > 0.0
    ]
    damage_candidates = [c for c in candidates if float(c.get("damage") or 0.0) > 0.0]

    threat_gap = float(player.get("threat_gap", 0.0))
    hp = float(player.get("hp", 0.0))
    hp_ratio = float(player.get("hp_ratio", 0.0))
    high_pressure = bool(
        threat_gap >= max(6.0, 0.25 * max(hp, 1.0))
        or threat_gap >= max(1.0, hp - 1.0)
        or (hp_ratio <= 0.35 and threat_gap >= 4.0)
    )
    no_pressure = bool(threat_gap <= 0.05)
    selected_lethal = bool(selected.get("confirmed_lethal"))
    selected_protective = bool(
        (float(selected.get("block") or 0.0) + float(selected.get("heal") or 0.0) - float(selected.get("hp_cost") or 0.0)) > 0.0
    )
    selected_end_turn = bool(selected.get("is_end_turn"))
    lethal_available = bool(lethal_candidates)
    survival_available = bool(survival_candidates)
    selected_nonprotective_under_pressure = bool(
        high_pressure
        and survival_available
        and not selected_protective
        and not selected_lethal
        and not selected_end_turn
    )
    selected_end_turn_under_pressure = bool(high_pressure and survival_available and selected_end_turn)
    no_pressure_pure_block = bool(
        no_pressure
        and selected_protective
        and float(selected.get("damage") or 0.0) <= 0.0
        and bool(damage_candidates)
        and not selected_lethal
    )
    missed_lethal = bool(lethal_available and not selected_lethal and not bool(selected.get("aoe")))

    candidates_sorted = sorted(
        candidates,
        key=lambda c: (
            bool(c.get("confirmed_lethal")),
            float(c.get("block") or 0.0) + float(c.get("heal") or 0.0),
            float(c.get("damage") or 0.0),
            float(c.get("policy") or 0.0),
        ),
        reverse=True,
    )
    payload = {
        "schema": "intent_combat_quality_v1",
        "floor": (progress or {}).get("floor"),
        "act_id": (progress or {}).get("act_id"),
        "room_type": (progress or {}).get("room_type"),
        "encounter_id": encounter_id,
        "encounter_tier": encounter_tier,
        "combat": player,
        "enemy_count": int(len(enemies)),
        "selected_action": selected,
        "candidate_count": int(len(candidates)),
        "candidates": candidates_sorted[: max(int(max_candidates), 1)],
        "high_pressure": bool(high_pressure),
        "no_pressure": bool(no_pressure),
        "lethal_available": bool(lethal_available),
        "selected_lethal": bool(selected_lethal),
        "missed_lethal": bool(missed_lethal),
        "survival_candidate_available": bool(survival_available),
        "selected_protective": bool(selected_protective),
        "selected_nonprotective_under_pressure": bool(selected_nonprotective_under_pressure),
        "selected_end_turn_under_pressure": bool(selected_end_turn_under_pressure),
        "no_pressure_pure_block": bool(no_pressure_pure_block),
        "best_lethal_candidate": _best_by(lethal_candidates, "lethal"),
        "best_survival_candidate": _best_by(survival_candidates, "survival"),
        "damage_candidate_count": int(len(damage_candidates)),
    }
    return _compact_dict(payload)


@dataclass
class IntentCombatQualityEpisodeTracker:
    seen: int = 0
    high_pressure: int = 0
    no_pressure: int = 0
    lethal_available: int = 0
    selected_lethal: int = 0
    missed_lethal: int = 0
    survival_candidate_available: int = 0
    selected_nonprotective_under_pressure: int = 0
    no_pressure_pure_block: int = 0
    selected_end_turn: int = 0

    def update(self, payload: dict[str, Any] | None) -> None:
        if not isinstance(payload, dict):
            return
        self.seen += 1
        selected = payload.get("selected_action") if isinstance(payload.get("selected_action"), dict) else {}
        if bool(payload.get("high_pressure")):
            self.high_pressure += 1
        if bool(payload.get("no_pressure")):
            self.no_pressure += 1
        if bool(payload.get("lethal_available")):
            self.lethal_available += 1
        if bool(payload.get("selected_lethal")):
            self.selected_lethal += 1
        if bool(payload.get("missed_lethal")):
            self.missed_lethal += 1
        if bool(payload.get("survival_candidate_available")):
            self.survival_candidate_available += 1
        if bool(payload.get("selected_nonprotective_under_pressure")) or bool(payload.get("selected_end_turn_under_pressure")):
            self.selected_nonprotective_under_pressure += 1
        if bool(payload.get("no_pressure_pure_block")):
            self.no_pressure_pure_block += 1
        if bool(selected.get("is_end_turn")):
            self.selected_end_turn += 1

    def as_metadata(self) -> dict[str, float]:
        seen_safe = max(int(self.seen), 1)
        lethal_safe = max(int(self.lethal_available), 1)
        survival_safe = max(int(self.survival_candidate_available), 1)
        no_pressure_safe = max(int(self.no_pressure), 1)
        return {
            "intent_combat_quality_seen_count": float(self.seen),
            "intent_combat_quality_high_pressure_rate": float(self.high_pressure) / float(seen_safe),
            "intent_combat_quality_no_pressure_rate": float(self.no_pressure) / float(seen_safe),
            "intent_combat_quality_lethal_available_rate": float(self.lethal_available) / float(seen_safe),
            "intent_combat_quality_selected_lethal_rate": float(self.selected_lethal) / float(lethal_safe),
            "intent_combat_quality_missed_lethal_rate": float(self.missed_lethal) / float(lethal_safe),
            "intent_combat_quality_survival_candidate_available_rate": (
                float(self.survival_candidate_available) / float(seen_safe)
            ),
            "intent_combat_quality_selected_nonprotective_under_pressure_rate": (
                float(self.selected_nonprotective_under_pressure) / float(survival_safe)
            ),
            "intent_combat_quality_no_pressure_pure_block_rate": (
                float(self.no_pressure_pure_block) / float(no_pressure_safe)
            ),
            "intent_combat_quality_selected_end_turn_rate": float(self.selected_end_turn) / float(seen_safe),
        }


def dump_intent_combat_quality_diagnostic(trainer: Any, payload: dict[str, Any] | None) -> None:
    """Append one bounded ``intent_combat_quality.jsonl`` record using trainer paths."""

    if not isinstance(payload, dict):
        return
    if getattr(trainer, "_intent_combat_quality_dump_disabled", False):
        return
    cap = int(getattr(trainer, "_intent_combat_quality_dump_max", 100000))
    count = int(getattr(trainer, "_intent_combat_quality_dump_count", 0) or 0)
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
        path = path_getter("intent_combat_quality.jsonl")
        if path is None:
            return
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("a", encoding="utf-8") as fh:
            fh.write(json.dumps(record, ensure_ascii=False, default=str) + "\n")
        trainer._intent_combat_quality_dump_count = count + 1
    except Exception:
        return


__all__ = [
    "INTENT_COMBAT_QUALITY_TB_KEYS",
    "IntentCombatQualityEpisodeTracker",
    "build_intent_combat_quality_payload",
    "dump_intent_combat_quality_diagnostic",
]
