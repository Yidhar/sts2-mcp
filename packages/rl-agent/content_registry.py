"""Lightweight semantic text registry hooks for offline STS2 training.

The registry merges optional layers under ``packages/rl-agent/content``:

1. generated metadata, e.g. ``cards.generated.json``
2. static extracted metadata, e.g. ``cards.static.generated.json``
3. curated overrides, e.g. ``cards.json``

Later layers win when both files define the same id.
"""

from __future__ import annotations

import json
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


def _entity_title(kind: str, entity_id: str | None) -> str:
    metadata = {
        "card": get_card_metadata,
        "relic": get_relic_metadata,
        "potion": get_potion_metadata,
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


def _compact_card_semantic_signals(metadata: dict[str, Any] | None) -> str:
    if not metadata:
        return ""
    raw = metadata.get("semantic_signals")
    if not isinstance(raw, dict):
        return ""

    parts: list[str] = []
    seen_keys: set[str] = set()
    for key in _CARD_SIGNAL_ORDER:
        value = raw.get(key)
        if value in (None, "", False):
            continue
        seen_keys.add(key)
        alias = _CARD_SIGNAL_ALIASES.get(key, key)
        parts.append(f"{alias}={_format_compact_number(value)}")

    for key in sorted(str(name) for name in raw.keys() if str(name) not in seen_keys):
        value = raw.get(key)
        if value in (None, "", False):
            continue
        parts.append(f"{key}={_format_compact_number(value)}")

    if not parts:
        return ""
    return "sig " + " ".join(parts)


def _compact_card_semantic_tags(metadata: dict[str, Any] | None) -> str:
    if not metadata:
        return ""
    raw = metadata.get("semantic_tags")
    if not isinstance(raw, list):
        return ""
    tags = [str(tag).strip() for tag in raw if str(tag).strip()]
    if not tags:
        return ""
    return "tag " + " ".join(tags)


def _compact_runtime_card_summary(card_payload: dict[str, Any] | None) -> str:
    if not isinstance(card_payload, dict):
        return ""

    tokens: list[str] = []

    cost = card_payload.get("cost")
    if isinstance(cost, (int, float)):
        tokens.append(f"{_format_compact_number(cost)}e")
    elif card_payload.get("x_cost"):
        tokens.append("Xe")

    star = card_payload.get("star")
    if isinstance(star, (int, float)) and float(star) > 0:
        tokens.append(f"{_format_compact_number(star)}s")
    elif card_payload.get("star_x"):
        tokens.append("Xs")

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

    title = _normalize_compact_text(card_payload.get("title"))
    if not title and card_id:
        title = build_card_label(card_id, upgrade_level)
    elif not title:
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
