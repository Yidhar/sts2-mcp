"""Import and normalize ``sts2-exporter`` items.json into local project assets.

This script is the bridge between OceanUwU/sts2-exporter and our own RL/content
pipeline. It reads the exporter-produced ``items.json`` and writes two kinds of
outputs:

1. Shared generated game data consumed by ``content_registry.py``:
   - ``game-data/generated/cards.static.generated.json``
   - ``game-data/generated/relics.static.generated.json``
   - ``game-data/generated/potions.static.generated.json``
2. Normalized dataset-side exports for future offline tasks:
   - ``datasets/static_export/cards.normalized.json``
   - ``datasets/static_export/relics.normalized.json``
   - ``datasets/static_export/potions.normalized.json``
   - ``datasets/static_export/events.normalized.json``
   - ``datasets/static_export/creatures.normalized.json``
   - ``datasets/static_export/keywords.normalized.json``
   - ``datasets/static_export/manifest.json``

The upstream exporter emits one entry per card upgrade level. We collapse those
rows back into a single base-card record with structured per-upgrade variants.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import shutil
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

from sts2_rl.artifacts import artifact_root, resolve_artifact_path
from sts2_rl.game_data import resolve_generated_game_data_output

PROJECT_ROOT = Path(__file__).resolve().parents[2]
GAME_DATA_ROOT = Path(os.environ.get("STS2_GAME_DATA_ROOT", PROJECT_ROOT / "game-data")).expanduser()
DEFAULT_CONTENT_DIR = GAME_DATA_ROOT / "generated"
ARTIFACT_ROOT = artifact_root()
DEFAULT_DATASET_DIR = resolve_artifact_path("datasets/static_export", root=ARTIFACT_ROOT)


def _common_export_paths() -> tuple[Path, ...]:
    candidates: list[Path] = []
    configured = os.environ.get("STS2_EXPORT_ITEMS")
    if configured:
        candidates.append(Path(configured).expanduser())
    candidates.extend((Path.cwd() / "export" / "items.json", PROJECT_ROOT / "tmp" / "export" / "items.json"))
    for variable in ("ProgramFiles(x86)", "ProgramFiles"):
        base = os.environ.get(variable)
        if base:
            candidates.append(Path(base) / "Steam" / "steamapps" / "common" / "Slay the Spire 2" / "export" / "items.json")
    return tuple(dict.fromkeys(candidates))


COMMON_EXPORT_PATHS = _common_export_paths()


def _normalize_text(value: Any) -> str:
    if not isinstance(value, str):
        return ""
    text = value.replace("\r\n", "\n").replace("\r", "\n").strip()
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text


def _compact_text(value: Any) -> str:
    text = _normalize_text(value)
    if not text:
        return ""
    return " / ".join(part.strip() for part in text.split("\n") if part.strip())


def _normalize_value(value: Any) -> Any:
    if isinstance(value, str):
        return _normalize_text(value)
    if isinstance(value, list):
        normalized = [_normalize_value(item) for item in value]
        return [item for item in normalized if not _is_effectively_empty(item)]
    if isinstance(value, dict):
        normalized = {str(key): _normalize_value(val) for key, val in value.items()}
        return {key: val for key, val in normalized.items() if not _is_effectively_empty(val)}
    if isinstance(value, (int, float, bool)) or value is None:
        return value
    return _normalize_text(value)


def _parse_energy_cost(raw_cost: Any) -> dict[str, Any]:
    text = str(raw_cost or "").strip()
    payload: dict[str, Any] = {
        "energy_cost_text": text,
        "energy_cost_x": False,
    }
    if not text:
        return payload
    if text.upper() == "X":
        payload["energy_cost_x"] = True
        return payload
    try:
        payload["energy_cost"] = int(text)
    except ValueError:
        payload["energy_cost_parse_error"] = text
    return payload


def _parse_star_cost(raw_star_cost: Any) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "star_cost_text": "",
        "star_cost_x": False,
    }
    if raw_star_cost is None:
        return payload
    try:
        value = int(raw_star_cost)
    except (TypeError, ValueError):
        payload["star_cost_parse_error"] = raw_star_cost
        payload["star_cost_text"] = str(raw_star_cost)
        return payload

    if value == -1:
        payload["star_cost_text"] = "X"
        payload["star_cost_x"] = True
        return payload

    payload["star_cost"] = value
    payload["star_cost_text"] = str(value)
    return payload


def _safe_int(value: Any, *, default: int = 0) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes((json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n").encode("utf-8"))


def _is_effectively_empty(value: Any) -> bool:
    if value is None or value is False:
        return True
    if isinstance(value, str) and value == "":
        return True
    if isinstance(value, (dict, list)) and len(value) == 0:
        return True
    return False


def _portable_source_label(_path: Path) -> str:
    """Return the logical input identity, never its invocation-time location.

    The exporter contract always calls this payload ``items.json``.  Recording
    the path used to reach it would make identical exporter bytes differ across
    checkouts and developer machines; the concrete path remains visible in the
    command's console output instead.
    """
    return "sts2-export/items.json"


def _resolve_items_path(explicit_path: str | None) -> Path:
    if explicit_path:
        path = Path(explicit_path).expanduser()
        if path.exists():
            return path
        raise FileNotFoundError(f"items.json not found: {path}")

    for candidate in COMMON_EXPORT_PATHS:
        if candidate.exists():
            return candidate

    candidates = "\n  - ".join(str(path) for path in COMMON_EXPORT_PATHS)
    raise FileNotFoundError(
        "Could not auto-detect sts2-exporter items.json.\n"
        "Pass --items explicitly, for example:\n"
        "  python packages/rl-agent/import_sts2_exporter_items.py --items "
        "\"<PATH_TO_STS2>\\export\\items.json\"\n"
        f"Auto-detect paths checked:\n  - {candidates}"
    )


def _load_items(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"Expected top-level object in {path}, got {type(payload).__name__}.")
    return payload


def _build_card_variant(entry: dict[str, Any]) -> dict[str, Any]:
    variant: dict[str, Any] = {
        "upgrade_level": _safe_int(entry.get("upgrades")),
        "description": _normalize_text(entry.get("description")),
        "effect": _compact_text(entry.get("effect") or entry.get("description")),
        "keywords": [str(keyword) for keyword in entry.get("keywords") or [] if str(keyword).strip()],
        "keyword_details": _normalize_value(entry.get("keywordDetails") or []),
        "canonical_text": _normalize_text(entry.get("canonicalText")),
    }
    semantic_tags = [str(tag) for tag in entry.get("semanticTags") or [] if str(tag).strip()]
    semantic_signals = _normalize_value(entry.get("semanticSignals") or {})
    if semantic_tags:
        variant["semantic_tags"] = semantic_tags
    if isinstance(semantic_signals, dict) and semantic_signals:
        variant["semantic_signals"] = semantic_signals
    variant.update(_parse_energy_cost(entry.get("cost")))
    variant.update(_parse_star_cost(entry.get("starCost")))
    return variant


def _variant_changed(base: dict[str, Any], other: dict[str, Any]) -> bool:
    keys = (
        "description",
        "effect",
        "energy_cost",
        "energy_cost_text",
        "energy_cost_x",
        "star_cost",
        "star_cost_text",
        "star_cost_x",
    )
    return any(base.get(key) != other.get(key) for key in keys)


def _pick_upgrade_variant(variants: list[dict[str, Any]]) -> dict[str, Any] | None:
    if len(variants) <= 1:
        return None
    base = variants[0]
    for variant in variants[1:]:
        if _variant_changed(base, variant):
            return variant
    return variants[-1]


def _normalize_cards(raw_cards: list[dict[str, Any]], *, source_path: Path, mod_info: dict[str, Any] | None) -> dict[str, Any]:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for entry in raw_cards:
        card_id = str(entry.get("id") or "").strip()
        if not card_id:
            continue
        grouped[card_id].append(entry)

    normalized: dict[str, Any] = {
        "__meta__": {
            "generator": "import_sts2_exporter_items.py",
            "source": "sts2-exporter",
            "source_items_json": _portable_source_label(source_path),
            "mod": mod_info or {},
            "entity_count": len(grouped),
        }
    }

    for card_id in sorted(grouped):
        entries = sorted(grouped[card_id], key=lambda item: _safe_int(item.get("upgrades")))
        base_entry = entries[0]
        variants = [_build_card_variant(entry) for entry in entries]
        upgrade_variant = _pick_upgrade_variant(variants)
        max_upgrade_level = max((variant["upgrade_level"] for variant in variants), default=0)

        record: dict[str, Any] = {
            "id": card_id,
            "title": _normalize_text(base_entry.get("name")),
            "color": _normalize_text(base_entry.get("color")),
            "rarity": _normalize_text(base_entry.get("rarity")),
            "type": _normalize_text(base_entry.get("type")),
            "target": _normalize_text(base_entry.get("target")),
            "description": variants[0]["description"],
            "effect": variants[0]["effect"],
            "keywords": variants[0].get("keywords") or [],
            "keyword_details": variants[0].get("keyword_details") or [],
            "semantic_tags": variants[0].get("semantic_tags") or [],
            "semantic_signals": variants[0].get("semantic_signals") or {},
            "canonical_text": variants[0].get("canonical_text") or "",
            "upgrade_levels": max_upgrade_level,
            "upgrade_level_texts": {
                str(variant["upgrade_level"]): {
                    key: value
                    for key, value in variant.items()
                    if key != "upgrade_level" and value not in ("", None, False)
                }
                for variant in variants
            },
        }
        record.update(_parse_energy_cost(base_entry.get("cost")))
        record.update(_parse_star_cost(base_entry.get("starCost")))

        if upgrade_variant is not None:
            upgrade_note = upgrade_variant.get("effect") or upgrade_variant.get("description") or ""
            if upgrade_note and upgrade_note != record.get("effect"):
                record["upgrade_note"] = upgrade_note
            record["upgrade_description"] = upgrade_variant.get("description") or ""
            raw_upgrade_diff = entries[1].get("upgradeDiffFromPrevious") if len(entries) > 1 else None
            if isinstance(raw_upgrade_diff, dict):
                record["upgrade_diff"] = _normalize_value(raw_upgrade_diff)
            record["upgrade_changes"] = {
                key: upgrade_variant.get(key)
                for key in (
                    "energy_cost",
                    "energy_cost_text",
                    "energy_cost_x",
                    "star_cost",
                    "star_cost_text",
                    "star_cost_x",
                )
                if upgrade_variant.get(key) not in ("", None, False)
                and upgrade_variant.get(key) != record.get(key)
            }

        normalized[card_id] = {
            key: value
            for key, value in record.items()
            if not _is_effectively_empty(value)
        }

    return normalized


def _normalize_relics(raw_relics: list[dict[str, Any]], *, source_path: Path, mod_info: dict[str, Any] | None) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "__meta__": {
            "generator": "import_sts2_exporter_items.py",
            "source": "sts2-exporter",
            "source_items_json": _portable_source_label(source_path),
            "mod": mod_info or {},
            "entity_count": 0,
        }
    }
    count = 0
    for entry in raw_relics:
        relic_id = str(entry.get("id") or "").strip()
        if not relic_id:
            continue
        count += 1
        payload[relic_id] = {
            "id": relic_id,
            "title": _normalize_text(entry.get("name")),
            "pool": _normalize_text(entry.get("pool")),
            "ancient": _normalize_text(entry.get("ancient")),
            "rarity": _normalize_text(entry.get("rarity") or entry.get("tier")),
            "description": _normalize_text(entry.get("description")),
            "summary": _compact_text(entry.get("description")),
            "flavor_text": _normalize_text(entry.get("flavorText")),
            "flavor": _normalize_text(entry.get("flavor")),
            "canonical_text": _normalize_text(entry.get("canonicalText")),
            "merchant_cost": entry.get("merchantCost"),
            "is_tradable": entry.get("isTradable"),
            "has_upon_pickup_effect": entry.get("hasUponPickupEffect"),
            "spawns_pets": entry.get("spawnsPets"),
            "adds_pet": entry.get("addsPet"),
            "is_stackable": entry.get("isStackable"),
            "pools": _normalize_value(entry.get("pools") or []),
            "character_owners": _normalize_value(entry.get("characterOwners") or []),
        }
        payload[relic_id] = {
            key: value for key, value in payload[relic_id].items() if not _is_effectively_empty(value)
        }
    payload["__meta__"]["entity_count"] = count
    return payload


def _normalize_potions(raw_potions: list[dict[str, Any]], *, source_path: Path, mod_info: dict[str, Any] | None) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "__meta__": {
            "generator": "import_sts2_exporter_items.py",
            "source": "sts2-exporter",
            "source_items_json": _portable_source_label(source_path),
            "mod": mod_info or {},
            "entity_count": 0,
        }
    }
    count = 0
    for entry in raw_potions:
        potion_id = str(entry.get("id") or "").strip()
        if not potion_id:
            continue
        count += 1
        payload[potion_id] = {
            "id": potion_id,
            "title": _normalize_text(entry.get("name")),
            "color": _normalize_text(entry.get("color")),
            "rarity": _normalize_text(entry.get("rarity")),
            "description": _normalize_text(entry.get("description")),
            "summary": _compact_text(entry.get("description")),
        }
        payload[potion_id] = {
            key: value for key, value in payload[potion_id].items() if not _is_effectively_empty(value)
        }
    payload["__meta__"]["entity_count"] = count
    return payload


def _normalize_simple_map(
    raw_entries: list[dict[str, Any]],
    *,
    id_key: str = "id",
    field_map: dict[str, str],
    source_path: Path,
    mod_info: dict[str, Any] | None,
    compact_fields: set[str] | None = None,
) -> dict[str, Any]:
    compact_fields = compact_fields or set()
    payload: dict[str, Any] = {
        "__meta__": {
            "generator": "import_sts2_exporter_items.py",
            "source": "sts2-exporter",
            "source_items_json": _portable_source_label(source_path),
            "mod": mod_info or {},
            "entity_count": 0,
        }
    }

    count = 0
    for entry in raw_entries:
        entity_id = str(entry.get(id_key) or "").strip()
        if not entity_id:
            continue
        count += 1
        record: dict[str, Any] = {"id": entity_id}
        for target_key, source_key in field_map.items():
            value = entry.get(source_key)
            if target_key in compact_fields and isinstance(value, str):
                text_value = _compact_text(value)
            else:
                text_value = _normalize_value(value)
            if text_value not in ("", None, []):
                record[target_key] = text_value
        payload[entity_id] = {
            key: value for key, value in record.items() if not _is_effectively_empty(value)
        }

    payload["__meta__"]["entity_count"] = count
    return payload


def _write_content_outputs(content_dir: Path, cards: dict[str, Any], relics: dict[str, Any], potions: dict[str, Any]) -> list[Path]:
    outputs = [
        content_dir / "cards.static.generated.json",
        content_dir / "relics.static.generated.json",
        content_dir / "potions.static.generated.json",
    ]
    for path, payload in zip(outputs, (cards, relics, potions)):
        _write_json(path, payload)
    return outputs


def _write_dataset_outputs(
    dataset_dir: Path,
    *,
    raw_items_path: Path,
    cards: dict[str, Any],
    relics: dict[str, Any],
    potions: dict[str, Any],
    events: dict[str, Any],
    creatures: dict[str, Any],
    keywords: dict[str, Any],
    enchantments: dict[str, Any],
    afflictions: dict[str, Any],
    mod_info: dict[str, Any] | None,
    copy_raw: bool,
) -> list[Path]:
    outputs = [
        dataset_dir / "cards.normalized.json",
        dataset_dir / "relics.normalized.json",
        dataset_dir / "potions.normalized.json",
        dataset_dir / "events.normalized.json",
        dataset_dir / "creatures.normalized.json",
        dataset_dir / "keywords.normalized.json",
        dataset_dir / "enchantments.normalized.json",
        dataset_dir / "afflictions.normalized.json",
    ]
    payloads = [cards, relics, potions, events, creatures, keywords, enchantments, afflictions]
    for path, payload in zip(outputs, payloads):
        _write_json(path, payload)

    manifest = {
        "generator": "import_sts2_exporter_items.py",
        "source": "sts2-exporter",
        "source_items_json": _portable_source_label(raw_items_path),
        "mod": mod_info or {},
        "counts": {
            "cards": max(len(cards) - 1, 0),
            "relics": max(len(relics) - 1, 0),
            "potions": max(len(potions) - 1, 0),
            "events": max(len(events) - 1, 0),
            "creatures": max(len(creatures) - 1, 0),
            "keywords": max(len(keywords) - 1, 0),
            "enchantments": max(len(enchantments) - 1, 0),
            "afflictions": max(len(afflictions) - 1, 0),
        },
    }
    manifest_path = dataset_dir / "manifest.json"
    _write_json(manifest_path, manifest)
    outputs.append(manifest_path)

    if copy_raw:
        raw_copy_path = dataset_dir / "items.raw.json"
        raw_copy_path.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(raw_items_path, raw_copy_path)
        outputs.append(raw_copy_path)

    return outputs


def build_outputs(items_path: Path) -> dict[str, dict[str, Any]]:
    raw = _load_items(items_path)
    mod_info = raw.get("mod") if isinstance(raw.get("mod"), dict) else None

    cards = _normalize_cards(list(raw.get("cards") or []), source_path=items_path, mod_info=mod_info)
    relics = _normalize_relics(list(raw.get("relics") or []), source_path=items_path, mod_info=mod_info)
    potions = _normalize_potions(list(raw.get("potions") or []), source_path=items_path, mod_info=mod_info)
    events = _normalize_simple_map(
        list(raw.get("events") or []),
        field_map={
            "title": "name",
            "description": "description",
            "options": "options",
            "layout_type": "layoutType",
            "is_shared": "isShared",
            "is_deterministic": "isDeterministic",
            "canonical_encounter_id": "canonicalEncounterId",
            "option_records": "optionRecords",
            "generated_initial_options": "generatedInitialOptions",
        },
        source_path=items_path,
        mod_info=mod_info,
    )
    creatures = _normalize_simple_map(
        list(raw.get("creatures") or []),
        field_map={
            "title": "name",
            "type": "type",
            "min_hp": "minHP",
            "max_hp": "maxHP",
            "min_hp_ascension": "minHPA",
            "max_hp_ascension": "maxHPA",
            "starting_gold": "startingGold",
            "max_energy": "maxEnergy",
            "card_pool_id": "cardPoolId",
            "relic_pool_id": "relicPoolId",
            "potion_pool_id": "potionPoolId",
            "starting_deck_ids": "startingDeckIds",
            "starting_relic_ids": "startingRelicIds",
            "starting_potion_ids": "startingPotionIds",
            "move_templates": "moveTemplates",
            "state_machine": "stateMachine",
        },
        source_path=items_path,
        mod_info=mod_info,
    )
    keywords = _normalize_simple_map(
        list(raw.get("keywords") or []),
        field_map={
            "title": "name",
            "description": "description",
            "type": "type",
            "has_icon": "hasIcon",
        },
        source_path=items_path,
        mod_info=mod_info,
    )
    enchantments = _normalize_simple_map(
        list(raw.get("enchantments") or []),
        field_map={
            "title": "name",
            "description": "description",
        },
        source_path=items_path,
        mod_info=mod_info,
    )
    afflictions = _normalize_simple_map(
        list(raw.get("afflictions") or []),
        field_map={
            "title": "name",
            "description": "description",
        },
        source_path=items_path,
        mod_info=mod_info,
    )

    return {
        "cards": cards,
        "relics": relics,
        "potions": potions,
        "events": events,
        "creatures": creatures,
        "keywords": keywords,
        "enchantments": enchantments,
        "afflictions": afflictions,
        "mod": mod_info or {},
    }


def parse_args(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--items", type=str, default=None, help="Path to sts2-exporter export/items.json")
    parser.add_argument("--content-dir", type=str, default=str(DEFAULT_CONTENT_DIR))
    parser.add_argument("--dataset-dir", type=str, default=str(DEFAULT_DATASET_DIR))
    parser.add_argument("--no-content", action="store_true", help="Skip writing game-data/generated/*.static.generated.json")
    parser.add_argument("--no-datasets", action="store_true", help="Skip writing datasets/static_export outputs")
    parser.add_argument("--no-copy-raw", action="store_true", help="Do not copy raw items.json into dataset export directory")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv or sys.argv[1:])
    items_path = _resolve_items_path(args.items)
    outputs = build_outputs(items_path)

    written_paths: list[Path] = []
    if not args.no_content:
        written_paths.extend(
            _write_content_outputs(
                resolve_generated_game_data_output(args.content_dir),
                outputs["cards"],
                outputs["relics"],
                outputs["potions"],
            )
        )

    if not args.no_datasets:
        written_paths.extend(
            _write_dataset_outputs(
                resolve_artifact_path(args.dataset_dir),
                raw_items_path=items_path,
                cards=outputs["cards"],
                relics=outputs["relics"],
                potions=outputs["potions"],
                events=outputs["events"],
                creatures=outputs["creatures"],
                keywords=outputs["keywords"],
                enchantments=outputs["enchantments"],
                afflictions=outputs["afflictions"],
                mod_info=outputs["mod"],
                copy_raw=not args.no_copy_raw,
            )
        )

    print(f"[sts2-exporter] source: {items_path}")
    print(f"[sts2-exporter] cards: {max(len(outputs['cards']) - 1, 0)}")
    print(f"[sts2-exporter] relics: {max(len(outputs['relics']) - 1, 0)}")
    print(f"[sts2-exporter] potions: {max(len(outputs['potions']) - 1, 0)}")
    print(f"[sts2-exporter] events: {max(len(outputs['events']) - 1, 0)}")
    print(f"[sts2-exporter] creatures: {max(len(outputs['creatures']) - 1, 0)}")
    print(f"[sts2-exporter] keywords: {max(len(outputs['keywords']) - 1, 0)}")
    print(f"[sts2-exporter] enchantments: {max(len(outputs['enchantments']) - 1, 0)}")
    print(f"[sts2-exporter] afflictions: {max(len(outputs['afflictions']) - 1, 0)}")
    for path in written_paths:
        print(f"[sts2-exporter] wrote: {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
