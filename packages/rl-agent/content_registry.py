"""Lightweight semantic text registry hooks for offline STS2 training.

The registry merges optional layers under ``packages/rl-agent/content``:

1. generated metadata, e.g. ``cards.generated.json``
2. static extracted metadata, e.g. ``cards.static.generated.json``
3. curated overrides, e.g. ``cards.json``

Later layers win when both files define the same id.
"""

from __future__ import annotations

import json
import re
from functools import lru_cache
from pathlib import Path
from typing import Any


_CONTENT_DIR = Path(__file__).with_name("content")
_CARD_SIGNAL_ORDER = (
    "damage",
    "hits",
    "block",
    "draw",
    "discard",
    "energyGain",
    "starGain",
    "strengthGain",
    "dexterityGain",
    "focusGain",
    "thornsGain",
    "intangibleGain",
    "heal",
    "hpLoss",
    "weak",
    "vulnerable",
    "poison",
    "calamity",
    "summon",
    "forge",
    "scry",
    "potionGain",
    "orbGeneration",
    "cardsToHand",
)
_CARD_SIGNAL_ALIASES = {
    "damage": "dmg",
    "hits": "hits",
    "block": "blk",
    "draw": "draw",
    "discard": "disc",
    "energyGain": "eng+",
    "starGain": "star+",
    "strengthGain": "str+",
    "dexterityGain": "dex+",
    "vigorGain": "vigor+",
    "focusGain": "focus+",
    "thornsGain": "thorn+",
    "intangibleGain": "intang+",
    "heal": "heal",
    "hpLoss": "hp-",
    "weak": "weak",
    "vulnerable": "vuln",
    "poison": "pois",
    "calamity": "calam",
    "summon": "summon",
    "forge": "forge",
    "scry": "scry",
    "potionGain": "pot+",
    "orbGeneration": "orb+",
    "cardsToHand": "hand+",
}
_ENTITY_SIGNAL_ALIASES = {
    "damage": "dmg",
    "hits": "hits",
    "block": "blk",
    "draw": "draw",
    "discard": "disc",
    "energyGain": "eng+",
    "starGain": "star+",
    "strengthGain": "str+",
    "dexterityGain": "dex+",
    "vigorGain": "vigor+",
    "focusGain": "focus+",
    "thornsGain": "thorn+",
    "intangibleGain": "intang+",
    "heal": "heal",
    "hpLoss": "hp-",
    "weak": "weak",
    "vulnerable": "vuln",
    "poison": "pois",
    "calamity": "calam",
    "summon": "summon",
    "forge": "forge",
    "scry": "scry",
    "potionGain": "pot+",
    "orbGeneration": "orb+",
    "cardsToHand": "hand+",
}
_INTENT_TYPE_TAGS = {
    "attack": ("attack",),
    "singleattackintent": ("attack",),
    "multiattackintent": ("attack", "multi_hit"),
    "attackdebuff": ("attack", "debuff"),
    "attackbuff": ("attack", "buff"),
    "attackdefend": ("attack", "block"),
    "block": ("block",),
    "defend": ("block",),
    "buff": ("buff",),
    "debuff": ("debuff",),
    "statuscard": ("status_card",),
    "status": ("status_card",),
    "summon": ("summon",),
    "sleep": ("sleep",),
    "escape": ("escape",),
    "stun": ("stun",),
}


def _deep_merge_dict(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    merged: dict[str, Any] = dict(base)
    for key, value in override.items():
        existing = merged.get(key)
        if isinstance(existing, dict) and isinstance(value, dict):
            merged[key] = _deep_merge_dict(existing, value)
        else:
            merged[key] = value
    return merged


def humanize_game_id(entity_id: str | None) -> str:
    raw = str(entity_id or "").strip()
    if not raw:
        return "[unknown]"
    tail = raw.split(".", 1)[-1]
    return tail.replace("_", " ").strip() or raw


@lru_cache(maxsize=None)
def _load_registry(kind: str) -> dict[str, dict[str, Any]]:
    merged: dict[str, dict[str, Any]] = {}
    for filename in (
        f"{kind}.generated.json",
        f"{kind}.static.generated.json",
        f"{kind}.json",
    ):
        path = _CONTENT_DIR / filename
        if not path.exists():
            continue
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except Exception:
            continue
        if not isinstance(payload, dict):
            continue
        for entity_id, metadata in payload.items():
            if not entity_id or str(entity_id).startswith("__"):
                continue
            if isinstance(metadata, dict):
                entity_key = str(entity_id)
                existing = merged.get(entity_key)
                if isinstance(existing, dict):
                    merged[entity_key] = _deep_merge_dict(existing, metadata)
                else:
                    merged[entity_key] = metadata
    return merged


def _get_metadata(kind: str, entity_id: str | None) -> dict[str, Any] | None:
    if not entity_id:
        return None
    return _load_registry(kind).get(str(entity_id))


def get_card_metadata(card_id: str | None) -> dict[str, Any] | None:
    return _get_metadata("cards", card_id)


def get_relic_metadata(relic_id: str | None) -> dict[str, Any] | None:
    return _get_metadata("relics", relic_id)


def get_potion_metadata(potion_id: str | None) -> dict[str, Any] | None:
    return _get_metadata("potions", potion_id)


def get_enemy_metadata(enemy_id: str | None) -> dict[str, Any] | None:
    return _get_metadata("enemies", enemy_id)


def _entity_title(kind: str, entity_id: str | None) -> str:
    metadata = {
        "card": get_card_metadata,
        "relic": get_relic_metadata,
        "potion": get_potion_metadata,
        "enemy": get_enemy_metadata,
    }.get(kind, lambda _value: None)(entity_id)
    if metadata:
        title = str(metadata.get("title") or "").strip()
        if title:
            return title
    return humanize_game_id(entity_id)


def _task_summary(metadata: dict[str, Any] | None, task: str | None) -> str:
    if not metadata:
        return ""
    if task:
        for key in ("task_summaries", "task_summary"):
            mapping = metadata.get(key)
            if isinstance(mapping, dict):
                text = str(mapping.get(task) or "").strip()
                if text:
                    return text
    for key in ("summary", "effect", "description"):
        text = str(metadata.get(key) or "").strip()
        if text:
            return text
    tags = metadata.get("tags")
    if isinstance(tags, list):
        compact = ", ".join(str(item).strip() for item in tags if str(item).strip())
        if compact:
            return compact
    return ""


def _format_compact_number(value: Any) -> str:
    if isinstance(value, bool):
        return "1" if value else "0"
    if isinstance(value, int):
        return str(value)
    if isinstance(value, float):
        rounded = int(value) if value.is_integer() else round(value, 3)
        return str(rounded)
    return str(value).strip()


def _normalize_compact_text(value: Any) -> str:
    text = str(value or "").strip()
    if not text:
        return ""
    return " ".join(text.split())


def _preferred_live_entity_title(
    kind: str,
    entity_id: str | None,
    runtime_title: Any,
    *,
    metadata: dict[str, Any] | None = None,
    upgrade_level: float = 0.0,
) -> str:
    metadata_title = _normalize_compact_text((metadata or {}).get("title")) if isinstance(metadata, dict) else ""
    if metadata_title:
        if kind == "card":
            return f"{metadata_title}{'+' * max(int(upgrade_level), 0)}"
        return metadata_title

    runtime = _normalize_compact_text(runtime_title)
    if runtime:
        return runtime

    if entity_id:
        if kind == "card":
            return build_card_label(entity_id, upgrade_level)
        return _entity_title(kind, entity_id)

    return ""


def _extract_semantic_hints_from_text(text: Any) -> tuple[dict[str, Any], list[str]]:
    normalized = _normalize_compact_text(text)
    if not normalized:
        return {}, []

    signals: dict[str, Any] = {}
    tags: list[str] = []

    def add_tag(tag: str) -> None:
        tag_value = str(tag).strip()
        if tag_value and tag_value not in tags:
            tags.append(tag_value)

    def add_signal(key: str, value: Any) -> None:
        if value in (None, "", False):
            return
        signals[key] = value

    numeric_rules = (
        (r"造成(\d+)点伤害", "damage", "damage"),
        (r"获得(\d+)点格挡", "block", "block"),
        (r"抽(\d+)张牌", "draw", "draw"),
        (r"回复(\d+)点生命", "heal", "heal"),
        (r"失去(\d+)点生命", "hpLoss", "hp_loss"),
        (r"获得(\d+)点力量", "strengthGain", "gain_strength"),
        (r"获得(\d+)点敏捷", "dexterityGain", "gain_dexterity"),
        (r"获得(\d+)点活力", "vigorGain", "gain_vigor"),
        (r"给予(\d+)层虚弱", "weak", "apply_weak"),
        (r"给予(\d+)层易伤", "vulnerable", "apply_vulnerable"),
        (r"给予(\d+)层中毒", "poison", "apply_poison"),
    )
    for pattern, signal_key, tag in numeric_rules:
        match = re.search(pattern, normalized)
        if match:
            add_signal(signal_key, int(match.group(1)))
            add_tag(tag)

    phrase_tags = (
        ("拾起时", "on_pickup"),
        ("每场战斗开始时", "combat_start"),
        ("在战斗结束时", "combat_end"),
        ("每回合", "per_turn"),
        ("免费打出", "cost_zero"),
        ("状态牌", "status_card"),
        ("变化", "transform_card"),
        ("升级", "upgrade"),
        ("药水栏位", "potion_slots"),
        ("药水", "potion"),
        ("随机", "random"),
        ("消耗", "exhaust"),
        ("从3张随机", "discover"),
        ("加入你的手牌", "add_to_hand"),
    )
    for needle, tag in phrase_tags:
        if needle in normalized:
            add_tag(tag)

    return signals, tags


def _resolve_card_variant_metadata(
    metadata: dict[str, Any] | None,
    *,
    upgrade_level: float = 0.0,
) -> dict[str, Any] | None:
    if not metadata:
        return None

    variants = metadata.get("upgrade_level_texts")
    if not isinstance(variants, dict) or not variants:
        return metadata

    requested_level = max(int(upgrade_level), 0)
    available_levels = []
    for key, value in variants.items():
        try:
            level = int(str(key))
        except (TypeError, ValueError):
            continue
        if isinstance(value, dict):
            available_levels.append(level)

    if not available_levels:
        return metadata

    chosen_level = requested_level if requested_level in available_levels else max(
        (level for level in available_levels if level <= requested_level),
        default=min(available_levels),
    )
    variant = variants.get(str(chosen_level))
    if not isinstance(variant, dict):
        return metadata
    return _deep_merge_dict(metadata, variant)


def _infer_upgrade_level_from_card_payload(card_payload: dict[str, Any] | None) -> int:
    if not isinstance(card_payload, dict):
        return 0

    raw_level = card_payload.get("upgrade_level")
    try:
        return max(int(raw_level), 0)
    except (TypeError, ValueError):
        pass

    title = str(card_payload.get("title") or "").strip()
    if not title:
        return 0

    if "+" not in title:
        return 0

    plus_suffix = title.rsplit("+", 1)[-1]
    if plus_suffix.isdigit():
        return max(int(plus_suffix), 0)
    if title.endswith("+"):
        return 1
    return title.count("+")


def _compact_semantic_signals(
    metadata: dict[str, Any] | None,
    *,
    signal_order: tuple[str, ...] | None = None,
    signal_aliases: dict[str, str] | None = None,
) -> str:
    if not metadata:
        return ""
    raw = metadata.get("semantic_signals")
    if not isinstance(raw, dict) or not raw:
        summary_text = metadata.get("summary") or metadata.get("effect") or metadata.get("description")
        fallback_signals, _ = _extract_semantic_hints_from_text(summary_text)
        raw = fallback_signals
    if not isinstance(raw, dict) or not raw:
        return ""

    ordered_keys = tuple(signal_order or ())
    aliases = signal_aliases or _ENTITY_SIGNAL_ALIASES

    parts: list[str] = []
    seen_keys: set[str] = set()
    for key in ordered_keys:
        value = raw.get(key)
        if value in (None, "", False):
            continue
        seen_keys.add(key)
        alias = aliases.get(key, key)
        parts.append(f"{alias}={_format_compact_number(value)}")

    for key in sorted(str(name) for name in raw.keys() if str(name) not in seen_keys):
        value = raw.get(key)
        if value in (None, "", False):
            continue
        alias = aliases.get(key, key)
        parts.append(f"{alias}={_format_compact_number(value)}")

    if not parts:
        return ""
    return "sig " + " ".join(parts)


def _compact_semantic_tags(metadata: dict[str, Any] | None) -> str:
    if not metadata:
        return ""
    raw = metadata.get("semantic_tags")
    if not isinstance(raw, list) or not raw:
        summary_text = metadata.get("summary") or metadata.get("effect") or metadata.get("description")
        _, fallback_tags = _extract_semantic_hints_from_text(summary_text)
        raw = fallback_tags
    if not isinstance(raw, list) or not raw:
        return ""
    tags = [str(tag).strip() for tag in raw if str(tag).strip()]
    if not tags:
        return ""
    return "tag " + " ".join(tags)


def _compact_card_semantic_signals(metadata: dict[str, Any] | None) -> str:
    return _compact_semantic_signals(
        metadata,
        signal_order=_CARD_SIGNAL_ORDER,
        signal_aliases=_CARD_SIGNAL_ALIASES,
    )


def _compact_card_semantic_tags(metadata: dict[str, Any] | None) -> str:
    return _compact_semantic_tags(metadata)


def _compact_runtime_card_summary(card_payload: dict[str, Any] | None) -> str:
    if not isinstance(card_payload, dict):
        return ""

    tokens: list[str] = []

    cost = card_payload.get("cost")
    if card_payload.get("x_cost"):
        tokens.append("Xe")
    elif isinstance(cost, (int, float)):
        tokens.append(f"{_format_compact_number(cost)}e")

    star = card_payload.get("star")
    if card_payload.get("star_x"):
        tokens.append("Xs")
    elif isinstance(star, (int, float)) and float(star) > 0:
        tokens.append(f"{_format_compact_number(star)}s")

    card_type = _normalize_compact_text(card_payload.get("type"))
    if card_type:
        tokens.append(card_type)

    target = _normalize_compact_text(card_payload.get("target"))
    if target:
        tokens.append(target)

    effect = _normalize_compact_text(card_payload.get("effect") or card_payload.get("description"))

    parts: list[str] = []
    if tokens:
        parts.append(" ".join(tokens))
    if effect:
        parts.append(effect)
    return " | ".join(parts)


def _compact_runtime_potion_summary(potion_payload: dict[str, Any] | None) -> str:
    if not isinstance(potion_payload, dict):
        return ""

    tokens: list[str] = []
    rarity = _normalize_compact_text(potion_payload.get("rarity"))
    if rarity:
        tokens.append(rarity)

    target = _normalize_compact_text(potion_payload.get("target"))
    if target:
        tokens.append(target)

    preview_parts: list[str] = []
    preview_aliases = (
        ("damage", "dmg"),
        ("block", "blk"),
        ("draw", "draw"),
        ("weak", "weak"),
        ("vulnerable", "vuln"),
        ("heal", "heal"),
        ("hp_loss", "hp-"),
        ("strength", "str+"),
        ("dexterity", "dex+"),
        ("summon", "summon"),
    )
    for key, alias in preview_aliases:
        value = potion_payload.get(key)
        if value in (None, "", False):
            continue
        preview_parts.append(f"{alias}={_format_compact_number(value)}")

    effect = _normalize_compact_text(potion_payload.get("description"))
    canonical_text = _normalize_compact_text(potion_payload.get("canonical_text"))

    parts: list[str] = []
    if tokens:
        parts.append(" ".join(tokens))
    if preview_parts:
        parts.append("sig " + " ".join(preview_parts))
    if effect:
        parts.append(effect)
    elif canonical_text:
        parts.append(canonical_text)
    return " | ".join(parts)


def _compact_entity_static_summary(kind: str, metadata: dict[str, Any] | None) -> str:
    if not metadata:
        return ""

    parts: list[str] = []
    rarity = _normalize_compact_text(metadata.get("rarity"))
    if rarity:
        parts.append(rarity)

    semantic_signals = _compact_semantic_signals(metadata)
    if semantic_signals:
        parts.append(semantic_signals)

    semantic_tags = _compact_semantic_tags(metadata)
    if semantic_tags:
        parts.append(semantic_tags)

    summary = _normalize_compact_text(
        metadata.get("summary")
        or metadata.get("effect")
        or metadata.get("description")
    )
    if summary:
        parts.append(summary)

    if kind == "relic":
        pools = metadata.get("pools")
        if isinstance(pools, list):
            pool_names = []
            for pool in pools[:2]:
                if not isinstance(pool, dict):
                    continue
                pool_name = _normalize_compact_text(pool.get("name") or pool.get("id"))
                if pool_name:
                    pool_names.append(pool_name)
            if pool_names:
                parts.append("pool " + ",".join(pool_names))

    return " | ".join(part for part in parts if part)


def _compact_card_static_summary(
    metadata: dict[str, Any] | None,
    *,
    upgrade_level: float = 0.0,
) -> str:
    if not metadata:
        return ""

    metadata = _resolve_card_variant_metadata(metadata, upgrade_level=upgrade_level)
    if not metadata:
        return ""

    tokens: list[str] = []

    energy_cost = metadata.get("energy_cost")
    if isinstance(energy_cost, (int, float)):
        energy_value = int(energy_cost) if float(energy_cost).is_integer() else float(energy_cost)
        tokens.append(f"{energy_value}e")
    elif metadata.get("energy_cost_x"):
        tokens.append("Xe")
    else:
        energy_text = str(metadata.get("energy_cost_text") or "").strip()
        if energy_text:
            tokens.append(f"{energy_text}e")

    star_cost = metadata.get("star_cost")
    if isinstance(star_cost, (int, float)) and float(star_cost) > 0:
        star_value = int(star_cost) if float(star_cost).is_integer() else float(star_cost)
        tokens.append(f"{star_value}s")
    elif metadata.get("star_cost_x"):
        tokens.append("Xs")
    else:
        star_text = str(metadata.get("star_cost_text") or "").strip()
        if star_text and star_text != "0":
            tokens.append(f"{star_text}s")

    card_type = str(metadata.get("type") or "").strip()
    if card_type:
        tokens.append(card_type)

    target = str(metadata.get("target") or "").strip()
    if target:
        tokens.append(target)

    effect = str(metadata.get("effect") or "").strip()
    upgrade_note = str(metadata.get("upgrade_note") or "").strip()
    semantic_signals = _compact_card_semantic_signals(metadata)
    semantic_tags = _compact_card_semantic_tags(metadata)

    parts: list[str] = []
    if tokens:
        parts.append(" ".join(tokens))
    if semantic_signals:
        parts.append(semantic_signals)
    if semantic_tags:
        parts.append(semantic_tags)
    if effect:
        parts.append(effect)
    if upgrade_level <= 0 and upgrade_note and upgrade_note != effect:
        parts.append(f"upg {upgrade_note}")
    return " | ".join(parts)


def get_card_task_summary(card_id: str | None, *, task: str | None = None) -> str:
    metadata = get_card_metadata(card_id)
    if not metadata:
        return ""
    if task:
        for key in ("task_summaries", "task_summary"):
            mapping = metadata.get(key)
            if isinstance(mapping, dict):
                text = str(mapping.get(task) or "").strip()
                if text:
                    return text
    summary = str(metadata.get("summary") or "").strip()
    if summary:
        return summary
    tags = metadata.get("tags")
    if isinstance(tags, list):
        compact = ", ".join(str(item).strip() for item in tags if str(item).strip())
        if compact:
            return compact
    return ""


def _entity_summary(kind: str, entity_id: str | None, *, task: str | None = None) -> str:
    metadata = {
        "card": get_card_metadata,
        "relic": get_relic_metadata,
        "potion": get_potion_metadata,
    }.get(kind, lambda _value: None)(entity_id)
    return _task_summary(metadata, task)


def build_card_label(card_id: str | None, upgrade_level: float = 0.0) -> str:
    suffix = "+" * max(int(upgrade_level), 0)
    return f"{_entity_title('card', card_id)}{suffix}"


def build_card_semantic_text(
    card_id: str | None,
    *,
    upgrade_level: float = 0.0,
    task: str | None = None,
) -> str:
    label = build_card_label(card_id, upgrade_level)
    metadata = get_card_metadata(card_id)
    static_summary = _compact_card_static_summary(metadata, upgrade_level=upgrade_level)
    summary = get_card_task_summary(card_id, task=task)
    parts = [label]
    if static_summary:
        parts.append(static_summary)
    if summary:
        parts.append(summary)
    return " | ".join(part for part in parts if part)


def build_live_card_semantic_text(card_payload: dict[str, Any] | None) -> str:
    if not isinstance(card_payload, dict):
        return ""

    card_id = str(card_payload.get("id") or "").strip()
    upgrade_level = _infer_upgrade_level_from_card_payload(card_payload)
    metadata = _resolve_card_variant_metadata(get_card_metadata(card_id), upgrade_level=upgrade_level)

    title = _preferred_live_entity_title(
        "card",
        card_id,
        card_payload.get("title"),
        metadata=metadata,
        upgrade_level=upgrade_level,
    )
    if not title:
        title = "[unknown]"

    runtime_summary = _compact_runtime_card_summary(card_payload)
    semantic_signals = _compact_card_semantic_signals(metadata)
    semantic_tags = _compact_card_semantic_tags(metadata)

    parts = [title]
    if runtime_summary:
        parts.append(runtime_summary)
    if semantic_signals:
        parts.append(semantic_signals)
    if semantic_tags:
        parts.append(semantic_tags)
    return " | ".join(part for part in parts if part)


def build_live_relic_semantic_text(relic_payload: dict[str, Any] | None) -> str:
    if not isinstance(relic_payload, dict):
        return ""

    relic_id = str(relic_payload.get("id") or "").strip()
    metadata = get_relic_metadata(relic_id)

    title = _preferred_live_entity_title("relic", relic_id, relic_payload.get("title"), metadata=metadata)
    if not title:
        title = "[unknown relic]"

    runtime_rarity = _normalize_compact_text(relic_payload.get("rarity"))
    static_summary = _compact_entity_static_summary("relic", metadata)
    canonical_text = _normalize_compact_text(relic_payload.get("canonical_text"))

    parts = [title]
    if runtime_rarity and not static_summary.startswith(runtime_rarity):
        parts.append(runtime_rarity)
    if static_summary:
        parts.append(static_summary)
    elif canonical_text:
        parts.append(canonical_text)
    return " | ".join(part for part in parts if part)


def build_live_potion_semantic_text(potion_payload: dict[str, Any] | None) -> str:
    if not isinstance(potion_payload, dict):
        return ""

    potion_id = str(potion_payload.get("id") or "").strip()
    metadata = get_potion_metadata(potion_id)

    title = _preferred_live_entity_title("potion", potion_id, potion_payload.get("title"), metadata=metadata)
    if not title:
        title = "[unknown potion]"

    runtime_summary = _compact_runtime_potion_summary(potion_payload)
    static_summary = _compact_entity_static_summary("potion", metadata)

    parts = [title]
    if runtime_summary:
        parts.append(runtime_summary)
    if static_summary and static_summary != runtime_summary:
        parts.append(static_summary)
    return " | ".join(part for part in parts if part)


def _normalize_intent_type(intent_type: Any) -> str:
    raw = _normalize_compact_text(intent_type)
    if not raw:
        return ""
    return raw.replace("Intent", "")


def build_enemy_intent_semantic_text(intent_payload: dict[str, Any] | None) -> str:
    if not isinstance(intent_payload, dict):
        return ""

    title = _normalize_compact_text(intent_payload.get("title"))
    intent_type_raw = _normalize_compact_text(intent_payload.get("intent_type"))
    intent_type = _normalize_intent_type(intent_type_raw)
    description = _normalize_compact_text(intent_payload.get("description") or intent_payload.get("text"))

    signal_parts: list[str] = []
    total_damage = intent_payload.get("total_damage")
    if total_damage not in (None, "", False):
        signal_parts.append(f"dmg={_format_compact_number(total_damage)}")
    repeats = intent_payload.get("repeats")
    if repeats not in (None, "", False) and int(float(repeats)) > 1:
        signal_parts.append(f"hits={_format_compact_number(repeats)}")
    damage_per_hit = intent_payload.get("damage_per_hit")
    if damage_per_hit not in (None, "", False):
        signal_parts.append(f"dph={_format_compact_number(damage_per_hit)}")
    candidate_count = intent_payload.get("candidate_count")
    if candidate_count not in (None, "", False) and int(float(candidate_count)) > 1:
        signal_parts.append(f"cand={_format_compact_number(candidate_count)}")

    tag_parts: list[str] = []
    normalized_lookup = intent_type_raw.lower().replace(" ", "")
    for key, tags in _INTENT_TYPE_TAGS.items():
        if key in normalized_lookup:
            for tag in tags:
                if tag not in tag_parts:
                    tag_parts.append(tag)

    parts: list[str] = []
    if title:
        parts.append(title)
    if intent_type:
        parts.append(f"type {intent_type}")
    if signal_parts:
        parts.append("sig " + " ".join(signal_parts))
    if tag_parts:
        parts.append("tag " + " ".join(tag_parts))
    if description:
        parts.append(description)
    return " | ".join(part for part in parts if part)


def _coerce_enemy_trait_entries(
    values: Any,
    *,
    category: str | None = None,
) -> list[dict[str, Any]]:
    entries: list[dict[str, Any]] = []
    for item in values or []:
        if not isinstance(item, dict):
            continue
        payload = dict(item)
        if category and not payload.get("category"):
            payload["category"] = category
        trait_name = _normalize_compact_text(
            payload.get("trait")
            or payload.get("effect_type")
            or payload.get("state")
            or payload.get("condition")
        )
        if not trait_name:
            continue
        payload["trait"] = trait_name
        description = _normalize_compact_text(payload.get("description"))
        if not description:
            description = trait_name
        payload["description"] = description
        entries.append(payload)
    return entries



def _dedupe_enemy_trait_entries(entries: list[dict[str, Any]]) -> list[dict[str, Any]]:
    deduped: list[dict[str, Any]] = []
    seen: set[str] = set()
    for entry in entries:
        key = "|".join(
            _normalize_compact_text(entry.get(field))
            for field in (
                "category",
                "trait",
                "trigger_type",
                "condition",
                "effect_type",
                "state",
                "threshold",
                "effect_amount",
            )
        )
        if key in seen:
            continue
        seen.add(key)
        deduped.append(entry)
    return deduped


def _reactive(
    trigger_type: str | None,
    condition: str | None,
    effect_type: str | None,
    description: str | None,
    *,
    trait: str | None = None,
    threshold: Any = None,
    state: str | None = None,
    effect_amount: Any = None,
    severity: str | None = None,
) -> dict[str, Any]:
    normalized_trait = _normalize_compact_text(trait or effect_type or condition or description)
    return {
        "category": "reactive",
        "trait": normalized_trait,
        "description": _normalize_compact_text(description) or normalized_trait,
        "trigger_type": _normalize_compact_text(trigger_type),
        "condition": _normalize_compact_text(condition),
        "effect_type": _normalize_compact_text(effect_type),
        "threshold": threshold,
        "state": _normalize_compact_text(state),
        "effect_amount": effect_amount,
        "severity": _normalize_compact_text(severity),
    }


def _phase_rule(
    trigger_type: str | None,
    condition: str | None,
    effect_type: str | None,
    description: str | None,
    *,
    trait: str | None = None,
    threshold: Any = None,
    state: str | None = None,
    effect_amount: Any = None,
    severity: str | None = None,
) -> dict[str, Any]:
    normalized_trait = _normalize_compact_text(trait or effect_type or condition or description)
    return {
        "category": "phase",
        "trait": normalized_trait,
        "description": _normalize_compact_text(description) or normalized_trait,
        "trigger_type": _normalize_compact_text(trigger_type),
        "condition": _normalize_compact_text(condition),
        "effect_type": _normalize_compact_text(effect_type),
        "threshold": threshold,
        "state": _normalize_compact_text(state),
        "effect_amount": effect_amount,
        "severity": _normalize_compact_text(severity),
    }


def _compact_enemy_trait_entry(entry: dict[str, Any]) -> str:
    trait = _normalize_compact_text(entry.get("trait"))
    description = _normalize_compact_text(entry.get("description"))
    trigger_type = _normalize_compact_text(entry.get("trigger_type"))
    condition = _normalize_compact_text(entry.get("condition"))
    effect_type = _normalize_compact_text(entry.get("effect_type"))
    state = _normalize_compact_text(entry.get("state"))
    amount = entry.get("effect_amount") or entry.get("amount")
    threshold = entry.get("threshold")

    compact_bits: list[str] = []
    if trigger_type:
        compact_bits.append(trigger_type)
    if condition:
        compact_bits.append(condition)
    if effect_type and effect_type != trait:
        compact_bits.append(effect_type)
    if state:
        compact_bits.append(state)
    if threshold not in (None, "", False):
        compact_bits.append(f"th={_format_compact_number(threshold)}")
    if amount not in (None, "", False):
        compact_bits.append(f"amt={_format_compact_number(amount)}")

    label = description or trait
    if compact_bits:
        return f"{label} ({', '.join(compact_bits)})"
    return label



def _compact_enemy_trait_text(entries: list[dict[str, Any]], *, limit: int = 4) -> str:
    compact: list[str] = []
    seen: set[str] = set()
    for entry in entries:
        rendered = _compact_enemy_trait_entry(entry)
        if not rendered or rendered in seen:
            continue
        seen.add(rendered)
        compact.append(rendered)
        if len(compact) >= limit:
            break
    if not compact:
        return ""
    return "trait " + " || ".join(compact)



def _compact_enemy_danger_profile(danger_profile: dict[str, Any] | None) -> str:
    if not isinstance(danger_profile, dict):
        return ""
    aliases = {
        "burst": "burst",
        "attrition": "attr",
        "scaling": "scale",
        "retaliation": "retal",
        "summon_pressure": "summon",
        "debuff_pressure": "debuff",
        "phase_complexity": "phase",
        "volatility": "vol",
        "target_priority": "prio",
    }
    parts: list[str] = []
    for key in (
        "burst",
        "attrition",
        "scaling",
        "retaliation",
        "summon_pressure",
        "debuff_pressure",
        "phase_complexity",
        "volatility",
        "target_priority",
    ):
        value = danger_profile.get(key)
        if value in (None, "", False):
            continue
        try:
            numeric = float(value)
        except (TypeError, ValueError):
            continue
        if numeric <= 0:
            continue
        parts.append(f"{aliases[key]}={_format_compact_number(numeric)}")
    notes = _normalize_compact_text(danger_profile.get("notes"))
    if notes:
        parts.append(notes)
    if not parts:
        return ""
    return "danger " + " ".join(parts)



def _enemy_metadata_from_payload(enemy_payload: dict[str, Any] | None) -> tuple[str, dict[str, Any] | None]:
    if not isinstance(enemy_payload, dict):
        return "", None
    enemy_id = _normalize_compact_text(enemy_payload.get("model_id") or enemy_payload.get("id"))
    return enemy_id, get_enemy_metadata(enemy_id)



def _collect_enemy_trait_entries(
    enemy_payload: dict[str, Any] | None,
    *,
    metadata: dict[str, Any] | None = None,
) -> list[dict[str, Any]]:
    entries: list[dict[str, Any]] = []
    if metadata:
        entries.extend(_coerce_enemy_trait_entries(metadata.get("static_traits"), category="static"))
        entries.extend(_coerce_enemy_trait_entries(metadata.get("reactive_triggers"), category="reactive"))
        entries.extend(_coerce_enemy_trait_entries(metadata.get("phase_rules"), category="phase"))
        entries.extend(_coerce_enemy_trait_entries(metadata.get("trait_tokens")))
    if isinstance(enemy_payload, dict):
        entries.extend(_coerce_enemy_trait_entries(enemy_payload.get("static_traits"), category="static"))
        entries.extend(_coerce_enemy_trait_entries(enemy_payload.get("reactive_triggers"), category="reactive"))
        entries.extend(_coerce_enemy_trait_entries(enemy_payload.get("phase_rules"), category="phase"))
        for power in enemy_payload.get("powers") or []:
            if not isinstance(power, dict):
                continue
            title = _normalize_compact_text(power.get("title") or power.get("description"))
            lowered = title.lower()
            if "thorn" in lowered or "spike" in lowered:
                entries.append(
                    _reactive(
                        "on_hit",
                        "contact",
                        "retaliate",
                        title or "thorns retaliation",
                        effect_amount=power.get("amount") or power.get("display_amount"),
                    )
                )
            if "retali" in lowered or "contact" in lowered:
                entries.append(
                    _reactive(
                        "on_hit",
                        "contact",
                        "retaliate",
                        title or "contact retaliation",
                        effect_amount=power.get("amount") or power.get("display_amount"),
                    )
                )
            if "intang" in lowered:
                entries.append(
                    _phase_rule(
                        "on_turn_start",
                        "intangible_window",
                        "gain_intangible",
                        title or "intangible window",
                    )
                )
            if "split" in lowered:
                entries.append(
                    _phase_rule(
                        "on_hp_threshold",
                        "threshold_crossed",
                        "split",
                        title or "split threshold",
                    )
                )
        name_text = " ".join(
            part
            for part in (
                _normalize_compact_text(enemy_payload.get("name")),
                _normalize_compact_text(enemy_payload.get("model_id")),
                _normalize_compact_text((enemy_payload.get("intent") or {}).get("description") if isinstance(enemy_payload.get("intent"), dict) else None),
            )
            if part
        ).lower()
        if "split" in name_text:
            entries.append(_phase_rule("on_hp_threshold", "threshold_crossed", "split", "split_on_threshold"))
        if "phase" in name_text or "threshold" in name_text:
            entries.append(_phase_rule("on_hp_threshold", "threshold_crossed", "phase_shift", "hp_threshold_phase_shift"))
    return _dedupe_enemy_trait_entries(entries)



def enemy_trait_vector(enemy_payload_or_metadata: dict[str, Any] | None) -> dict[str, float]:
    payload = enemy_payload_or_metadata if isinstance(enemy_payload_or_metadata, dict) and any(
        key in enemy_payload_or_metadata for key in ("name", "model_id", "intent", "powers")
    ) else None
    metadata = enemy_payload_or_metadata if isinstance(enemy_payload_or_metadata, dict) and payload is None else None
    if payload is not None:
        _enemy_id, payload_metadata = _enemy_metadata_from_payload(payload)
        metadata = payload_metadata or metadata
    entries = _collect_enemy_trait_entries(payload, metadata=metadata)
    danger_profile = {}
    if isinstance(metadata, dict):
        danger_profile = metadata.get("danger_profile") or {}
    if isinstance(payload, dict) and isinstance(payload.get("danger_profile"), dict):
        danger_profile = payload.get("danger_profile") or danger_profile

    text_blob = " ".join(_compact_enemy_trait_entry(entry) for entry in entries).lower()
    return {
        "retaliation": float("retali" in text_blob or "thorn" in text_blob),
        "summon": float("summon" in text_blob or "spawn" in text_blob),
        "phase_shift": float("phase" in text_blob or "split" in text_blob),
        "threshold": float("threshold" in text_blob or "th=" in text_blob),
        "debuff": float("debuff" in text_blob or "bind" in text_blob or "vulnerable" in text_blob),
        "intangible": float("intang" in text_blob),
        "burst": float(min(float(danger_profile.get("burst") or 0), 5.0) / 5.0),
        "attrition": float(min(float(danger_profile.get("attrition") or 0), 5.0) / 5.0),
        "target_priority": float(min(float(danger_profile.get("target_priority") or 0), 5.0) / 5.0),
    }



def build_enemy_semantic_text(enemy_id: str | None) -> str:
    enemy_key = _normalize_compact_text(enemy_id)
    title = _entity_title("enemy", enemy_key) if enemy_key else "[unknown enemy]"
    metadata = get_enemy_metadata(enemy_key)
    if not metadata:
        return title
    trait_text = _compact_enemy_trait_text(_collect_enemy_trait_entries(None, metadata=metadata))
    danger_text = _compact_enemy_danger_profile(metadata.get("danger_profile"))
    summary = _normalize_compact_text(metadata.get("summary"))
    parts = [title]
    if trait_text:
        parts.append(trait_text)
    if danger_text:
        parts.append(danger_text)
    if summary:
        parts.append(summary)
    return " | ".join(part for part in parts if part)



def build_live_enemy_trait_text(enemy_payload: dict[str, Any] | None) -> str:
    if not isinstance(enemy_payload, dict):
        return ""
    _enemy_id, metadata = _enemy_metadata_from_payload(enemy_payload)
    return _compact_enemy_trait_text(_collect_enemy_trait_entries(enemy_payload, metadata=metadata))



def build_live_enemy_semantic_text(enemy_payload: dict[str, Any] | None) -> str:
    if not isinstance(enemy_payload, dict):
        return ""

    enemy_id, metadata = _enemy_metadata_from_payload(enemy_payload)
    name = _normalize_compact_text(enemy_payload.get("name"))
    if not name:
        name = _entity_title("enemy", enemy_id) if enemy_id else "[unknown enemy]"
    intent_text = build_enemy_intent_semantic_text(enemy_payload.get("intent"))
    trait_text = build_live_enemy_trait_text(enemy_payload)
    danger_text = _compact_enemy_danger_profile(
        enemy_payload.get("danger_profile") if isinstance(enemy_payload.get("danger_profile"), dict) else (metadata.get("danger_profile") if isinstance(metadata, dict) else None)
    )

    runtime_parts: list[str] = []
    current_hp = enemy_payload.get("current_hp")
    max_hp = enemy_payload.get("max_hp")
    block = enemy_payload.get("block")
    if current_hp not in (None, "") and max_hp not in (None, ""):
        runtime_parts.append(f"hp={_format_compact_number(current_hp)}/{_format_compact_number(max_hp)}")
    if block not in (None, "", False):
        runtime_parts.append(f"blk={_format_compact_number(block)}")

    power_parts: list[str] = []
    for power in (enemy_payload.get("powers") or [])[:3]:
        if not isinstance(power, dict):
            continue
        power_title = _normalize_compact_text(power.get("title"))
        amount = power.get("amount") or power.get("display_amount")
        if not power_title:
            continue
        if amount in (None, "", False):
            power_parts.append(power_title)
        else:
            power_parts.append(f"{power_title}:{_format_compact_number(amount)}")

    static_summary = _normalize_compact_text(metadata.get("summary")) if isinstance(metadata, dict) else ""

    parts = [name]
    if runtime_parts:
        parts.append("state " + " ".join(runtime_parts))
    if intent_text:
        parts.append(intent_text)
    if trait_text:
        parts.append(trait_text)
    if danger_text:
        parts.append(danger_text)
    if power_parts:
        parts.append("pow " + " ".join(power_parts))
    if static_summary and static_summary not in parts:
        parts.append(static_summary)
    return " | ".join(part for part in parts if part)


def build_entity_text(kind: str, entity_id: str | None, *, task: str | None = None) -> str:
    title = _entity_title(kind, entity_id)
    summary = _entity_summary(kind, entity_id, task=task)
    if summary:
        return f"{title} | {summary}"
    return title


def build_candidate_semantic_text(
    *,
    task: str,
    option_kind: str,
    candidate_id: str,
    count: float = 1.0,
    upgrade_level: float = 0.0,
) -> str:
    if option_kind == "card":
        entity_text = build_card_semantic_text(candidate_id, upgrade_level=upgrade_level, task=task)
    elif option_kind == "relic":
        entity_text = build_entity_text("relic", candidate_id, task=task)
    elif option_kind == "potion":
        entity_text = build_entity_text("potion", candidate_id, task=task)
    else:
        entity_text = humanize_game_id(candidate_id)

    prefix = {
        "regular_card_reward": "card reward",
        "event_card_bundle": "event card",
        "ancient_choice": "ancient",
        "relic_choice_step": "relic",
        "potion_choice_step": "potion",
        "smith_target": "smith",
        "remove_card_step": "remove",
        "transform_card_step": "transform",
        "shop_relic_pick_step": "shop relic",
        "shop_potion_pick_step": "shop potion",
        "shop_remove_target_step": "shop remove",
    }.get(task, option_kind or "choice")
    count_suffix = f" x{int(count)}" if count > 1 else ""
    return f"{prefix} {entity_text}{count_suffix}".strip()


def summarize_entity_ids(kind: str, entity_ids: list[str] | None, *, limit: int = 3) -> str:
    values = [str(value) for value in (entity_ids or []) if value]
    if not values:
        return "none"
    labels = [
        build_card_label(value) if kind == "card" else build_entity_text(kind, value)
        for value in values[:limit]
    ]
    if len(values) > limit:
        labels.append(f"+{len(values) - limit} more")
    return ", ".join(labels)
