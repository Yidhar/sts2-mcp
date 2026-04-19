"""Combat snapshot dataset loading and sampling utilities.

This module bridges exported run-history combat snapshots into live combat
sandbox resets. The goal is intentionally narrow:

- load ``combat_snapshot_samples`` from JSONL or Parquet
- apply lightweight filters useful for training
- sample either uniformly by row or balanced by encounter

The sampled rows can then be converted directly into ``/env/combat_reset``
payload fields such as encounter_id, HP, deck, relics, and gold.
"""

from __future__ import annotations

import json
from collections import Counter
from functools import lru_cache
from pathlib import Path
from typing import Any

import numpy as np

from content_registry import (
    get_card_metadata,
    get_enemy_metadata,
    get_potion_metadata,
    get_relic_metadata,
)

VALID_ENCOUNTER_TIERS = {"weak", "normal", "elite", "boss"}

# Deck-stage is orthogonal to encounter tier. It classifies how mature the
# player's deck is at snapshot time. "starter_early" is the critical-coverage
# bucket: snapshots where the player still has ~starter deck on the first
# 1-3 floors. Full-run training crashes in act 1 because the sandbox-trained
# policy never sees this state.
STARTER_EARLY_MAX_FLOOR = 3
STARTER_EARLY_MAX_DECK_SIZE = 13
VALID_DECK_STAGES = {"starter_early", "rest"}
DEFAULT_EXCLUDED_COMBAT_CHARACTERS = frozenset({
    "CHARACTER.WATCHER",
})
VALID_CURATED_COMBINED_SUBSETS = {
    "human_only",
    "human_only_weak_normal",
    "human_only_minus_combat_reset_failures",
    "human_only_weak_normal_minus_combat_reset_failures",
    "human_only_roomwin_only",
    "human_only_roomwin_only_minus_combat_reset_failures",
    "human_only_weak_normal_roomwin_only",
    "human_only_weak_normal_roomwin_only_minus_combat_reset_failures",
    "local_act1clear_only",
    "local_act1clear_only_minus_combat_reset_failures",
    "local_act1clear_only_roomwin_only",
    "local_act1clear_only_roomwin_only_minus_combat_reset_failures",
    "bootstrap_human_plus_local_act1clear",
    "bootstrap_human_plus_local_act1clear_minus_combat_reset_failures",
    "bootstrap_human_plus_local_act1clear_weak_normal",
    "bootstrap_human_plus_local_act1clear_weak_normal_minus_combat_reset_failures",
    "bootstrap_human_plus_local_act1clear_roomwin_only",
    "bootstrap_human_plus_local_act1clear_roomwin_only_minus_combat_reset_failures",
    "bootstrap_human_plus_local_act1clear_weak_normal_roomwin_only",
    "bootstrap_human_plus_local_act1clear_weak_normal_roomwin_only_minus_combat_reset_failures",
    # "all"-quality variants: include local_history runs regardless of cleared_act1,
    # so starter-deck floor 1-3 snapshots from failed runs survive. Required for
    # bridging sandbox -> full_run training after act 1 death distribution shift.
    "bootstrap_human_plus_local_all",
    "bootstrap_human_plus_local_all_minus_combat_reset_failures",
    "bootstrap_human_plus_local_all_weak_normal",
    "bootstrap_human_plus_local_all_weak_normal_minus_combat_reset_failures",
    "bootstrap_human_plus_local_all_roomwin_only",
    "bootstrap_human_plus_local_all_roomwin_only_minus_combat_reset_failures",
    "bootstrap_human_plus_local_all_weak_normal_roomwin_only",
    "bootstrap_human_plus_local_all_weak_normal_roomwin_only_minus_combat_reset_failures",
}
DEFAULT_CURATED_COMBINED_SUBSET = "bootstrap_human_plus_local_act1clear_roomwin_only"
_UNRESOLVED_CARD_TEMPLATE_MARKERS = (
    "{CardType:",
    "{Damage:",
    "{Block:",
    "{Violence:",
    "{HasRider:",
    "{Sapping:",
    "{Choking:",
    ":choose(",
    ":diff()",
)


def _resolve_card_max_upgrade_level(card_id: str) -> int | None:
    metadata = get_card_metadata(card_id)
    if not isinstance(metadata, dict):
        return None

    raw_upgrade_levels = metadata.get("upgrade_levels")
    try:
        return max(int(raw_upgrade_levels), 0)
    except (TypeError, ValueError):
        pass

    raw_max_upgrade_level = metadata.get("max_upgrade_level")
    try:
        return max(int(raw_max_upgrade_level), 0)
    except (TypeError, ValueError):
        pass

    upgrade_level_texts = metadata.get("upgrade_level_texts")
    if isinstance(upgrade_level_texts, dict) and upgrade_level_texts:
        available_levels: list[int] = []
        for key in upgrade_level_texts.keys():
            try:
                available_levels.append(max(int(str(key)), 0))
            except (TypeError, ValueError):
                continue
        if available_levels:
            return max(available_levels)

    observations = metadata.get("observations")
    if isinstance(observations, dict):
        raw_seen = observations.get("max_upgrade_level_seen")
        try:
            return max(int(raw_seen), 0)
        except (TypeError, ValueError):
            pass

    raw_seen = metadata.get("max_upgrade_level_seen")
    try:
        return max(int(raw_seen), 0)
    except (TypeError, ValueError):
        return None


def _clamp_card_upgrade_level(card_id: str, upgrade_level: int) -> int:
    max_upgrade_level = _resolve_card_max_upgrade_level(card_id)
    if max_upgrade_level is None:
        return max(upgrade_level, 0)
    return min(max(upgrade_level, 0), max_upgrade_level)


def _contains_unresolved_card_template(value: Any) -> bool:
    text = str(value or "").strip()
    if not text:
        return False
    return any(marker in text for marker in _UNRESOLVED_CARD_TEMPLATE_MARKERS)


@lru_cache(maxsize=None)
def _snapshot_deck_card_is_supported(card_id: str) -> bool:
    metadata = get_card_metadata(card_id)
    if not isinstance(metadata, dict):
        return True

    card_type = str(metadata.get("type") or "").strip().lower()
    if card_type == "none":
        return False

    for key in ("description", "canonical_text", "upgrade_description", "effect"):
        if _contains_unresolved_card_template(metadata.get(key)):
            return False

    upgrade_level_texts = metadata.get("upgrade_level_texts")
    if isinstance(upgrade_level_texts, dict):
        for payload in upgrade_level_texts.values():
            if not isinstance(payload, dict):
                continue
            for key in ("description", "canonical_text", "effect"):
                if _contains_unresolved_card_template(payload.get(key)):
                    return False

    return True


def _sanitize_snapshot_deck_card_ids(card_ids: list[str]) -> list[str]:
    return [
        card_id
        for card_id in card_ids
        if card_id and _snapshot_deck_card_is_supported(card_id)
    ]


def _normalize_character_id(value: Any) -> str:
    return str(value or "").strip().upper()


def _resolve_excluded_characters(
    excluded_characters: set[str] | list[str] | tuple[str, ...] | None,
) -> set[str]:
    if excluded_characters is None:
        values = DEFAULT_EXCLUDED_COMBAT_CHARACTERS
    else:
        values = excluded_characters
    return {
        _normalize_character_id(value)
        for value in values
        if _normalize_character_id(value)
    }


def _resolve_supported_characters(
    supported_characters: set[str] | list[str] | tuple[str, ...] | None,
) -> set[str] | None:
    if supported_characters is None:
        return None
    resolved = {
        _normalize_character_id(value)
        for value in supported_characters
        if _normalize_character_id(value)
    }
    return resolved or None


def infer_encounter_tier(
    encounter_id: str | None = None,
    *,
    room_type: str | None = None,
) -> str:
    """Infer a coarse encounter tier for curriculum filtering."""
    normalized_room_type = str(room_type or "").strip().lower()
    if normalized_room_type == "boss":
        return "boss"
    if normalized_room_type == "elite":
        return "elite"

    enc = str(encounter_id or "").strip().upper()
    if enc.endswith("_WEAK"):
        return "weak"
    if enc.endswith("_ELITE"):
        return "elite"
    if enc.endswith("_BOSS"):
        return "boss"
    if enc.endswith("_NORMAL") or enc.endswith("_NORMAL_ALT"):
        return "normal"

    return "normal"


def infer_encounter_tier_from_row(row: dict[str, Any]) -> str:
    return infer_encounter_tier(
        row.get("encounter_id"),
        room_type=row.get("room_type"),
    )


def infer_deck_stage(
    *,
    floor_number: int | None,
    deck_size: int | None,
) -> str:
    """Classify how mature the player's deck is at snapshot time.

    "starter_early" means the deck is close to pristine starter composition
    on an early floor — the state the model crashes in at act 1 when moved
    from combat sandbox to full-run training. Rows without a usable floor
    or deck size default to "rest" so they don't silently leak into the
    starter-early bucket.
    """
    if not isinstance(floor_number, int) or floor_number < 1:
        return "rest"
    if not isinstance(deck_size, int) or deck_size < 1:
        return "rest"
    if floor_number <= STARTER_EARLY_MAX_FLOOR and deck_size <= STARTER_EARLY_MAX_DECK_SIZE:
        return "starter_early"
    return "rest"


def infer_deck_stage_from_row(row: dict[str, Any]) -> str:
    deck_ids = row.get("deck_card_ids")
    deck_size = len(deck_ids) if isinstance(deck_ids, list) else None
    return infer_deck_stage(
        floor_number=row.get("floor_number"),
        deck_size=deck_size,
    )


def is_failed_combat_room_snapshot(row: dict[str, Any]) -> bool:
    """Return True when a combat snapshot is the run-ending failed room itself."""

    killed_by_encounter = str(row.get("source_killed_by_encounter") or "").strip()
    encounter_id = str(row.get("encounter_id") or "").strip()
    floor_number = row.get("floor_number")
    source_run_path_point_count = row.get("source_run_path_point_count")

    if bool(row.get("source_run_win")):
        return False
    if not killed_by_encounter or killed_by_encounter == "NONE.NONE":
        return False
    if encounter_id != killed_by_encounter:
        return False
    if not isinstance(floor_number, int) or not isinstance(source_run_path_point_count, int):
        return False
    return floor_number == source_run_path_point_count


def is_room_win_combat_snapshot_row(row: dict[str, Any]) -> bool:
    """Return True when the extracted combat room itself was won."""

    return not is_failed_combat_room_snapshot(row)


def resolve_combat_snapshot_dataset_path(
    path: str | Path,
    *,
    curated_subset: str | None = None,
) -> Path:
    """Resolve a user-provided path to a concrete combat snapshot dataset file.

    Accepted inputs:
    - direct JSONL / Parquet file path
    - dataset root containing ``combat_snapshot_samples.jsonl``
    - dataset root containing ``parquet/combat_snapshot_samples.parquet``
    - partition dir such as ``.../by_build_family/v0.99.1``
    - curated combat root / combined dir containing bootstrap subsets
    """

    input_path = Path(path)
    if input_path.is_file():
        return input_path

    if not input_path.exists():
        raise FileNotFoundError(f"Combat snapshot dataset path does not exist: {input_path}")

    preferred_subset = str(curated_subset or "").strip()
    if preferred_subset:
        if preferred_subset not in VALID_CURATED_COMBINED_SUBSETS:
            raise ValueError(
                f"Unsupported curated combat subset {preferred_subset!r}; "
                f"expected one of {sorted(VALID_CURATED_COMBINED_SUBSETS)}"
            )

    curated_candidates: list[Path] = []
    subsets_to_try: list[str] = []
    if preferred_subset:
        subsets_to_try.append(preferred_subset)
    if (input_path / "combined").exists() or (input_path / "curated_runs_summary.jsonl").exists():
        if DEFAULT_CURATED_COMBINED_SUBSET not in subsets_to_try:
            subsets_to_try.append(DEFAULT_CURATED_COMBINED_SUBSET)

    for subset in subsets_to_try:
        curated_candidates.extend(
            [
                input_path / "combined" / f"{subset}.parquet",
                input_path / "combined" / f"{subset}.jsonl",
                input_path / f"{subset}.parquet",
                input_path / f"{subset}.jsonl",
            ]
        )

    for candidate in curated_candidates:
        if candidate.exists():
            return candidate

    candidates = [
        input_path / "combat_snapshot_samples.parquet",
        input_path / "combat_snapshot_samples.jsonl",
        input_path / "combat_snapshot_samples_all.parquet",
        input_path / "combat_snapshot_samples_all.jsonl",
        input_path / "parquet" / "combat_snapshot_samples.parquet",
        input_path / "parquet" / "combat_snapshot_samples.jsonl",
        input_path / "parquet" / "combat_snapshot_samples_all.parquet",
        input_path / "parquet" / "combat_snapshot_samples_all.jsonl",
    ]
    for candidate in candidates:
        if candidate.exists():
            return candidate

    raise FileNotFoundError(
        f"Could not resolve combat_snapshot_samples dataset under: {input_path}"
    )


def _load_jsonl_rows(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
    return rows


def _load_parquet_rows(path: Path) -> list[dict[str, Any]]:
    import pyarrow.parquet as pq

    table = pq.read_table(path, memory_map=True)
    rows = table.to_pylist()
    for row in rows:
        for key, value in list(row.items()):
            if isinstance(value, str) and value and value[:1] in ("{", "["):
                try:
                    row[key] = json.loads(value)
                except Exception:
                    pass
    return rows


@lru_cache(maxsize=None)
def _card_id_is_known(card_id: str) -> bool:
    return isinstance(get_card_metadata(card_id), dict)


@lru_cache(maxsize=None)
def _relic_id_is_known(relic_id: str) -> bool:
    return isinstance(get_relic_metadata(relic_id), dict)


@lru_cache(maxsize=None)
def _potion_id_is_known(potion_id: str) -> bool:
    return isinstance(get_potion_metadata(potion_id), dict)


@lru_cache(maxsize=None)
def _monster_id_is_known(monster_id: str) -> bool:
    return isinstance(get_enemy_metadata(monster_id), dict)


def _row_non_vanilla_id_kinds(row: dict[str, Any]) -> list[str]:
    """Return the content kinds that carry at least one id not in the static
    registry. Empty list means the row is vanilla as far as we can tell.

    The static registries are generated from the game's /static catalog, so
    any id missing here is either modded content or a near-future-game-version
    entry. Per project decision we treat both as "non-vanilla" and drop the
    row to keep training distribution stable.
    """
    offending: list[str] = []

    deck_ids = row.get("deck_card_ids") or []
    if isinstance(deck_ids, list):
        for card_id in deck_ids:
            if isinstance(card_id, str) and card_id and not _card_id_is_known(card_id):
                offending.append("card")
                break

    deck_entries = row.get("deck_entries") or []
    if isinstance(deck_entries, list) and "card" not in offending:
        for entry in deck_entries:
            if not isinstance(entry, dict):
                continue
            card_id = entry.get("id")
            if isinstance(card_id, str) and card_id and not _card_id_is_known(card_id):
                offending.append("card")
                break

    for field_key, is_known in (
        ("relic_ids_before", _relic_id_is_known),
        ("potion_ids_before", _potion_id_is_known),
        ("monster_ids", _monster_id_is_known),
    ):
        values = row.get(field_key) or []
        if not isinstance(values, list):
            continue
        kind = field_key.split("_", 1)[0]  # relic / potion / monster
        if any(
            isinstance(value, str) and value and not is_known(value)
            for value in values
        ):
            offending.append(kind)

    return offending


def validate_combat_snapshot_row(row: dict[str, Any]) -> list[str]:
    """Return a list of reject reasons for an unusable combat snapshot row.

    The goal here is intentionally pragmatic: keep rows that can be turned into a
    stable ``/env/combat_reset`` opening state with the current bridge contract,
    and reject rows that are structurally corrupted or obviously under-specified.
    """

    return validate_combat_snapshot_row_against(row)


def validate_combat_snapshot_row_against(
    row: dict[str, Any],
    *,
    supported_encounter_ids: set[str] | None = None,
    supported_characters: set[str] | None = None,
    excluded_characters: set[str] | None = None,
    reject_non_vanilla: bool = False,
) -> list[str]:
    reasons: list[str] = []
    if reject_non_vanilla:
        offending_kinds = _row_non_vanilla_id_kinds(row)
        if offending_kinds:
            # One rejection reason per kind so Counter reports are useful.
            for kind in offending_kinds:
                reasons.append(f"non_vanilla_{kind}")
    sample_id = row.get("sample_id")
    if not isinstance(sample_id, str) or not sample_id:
        reasons.append("missing_sample_id")

    character = row.get("character")
    if not isinstance(character, str) or not character:
        reasons.append("missing_character")
    else:
        normalized_character = _normalize_character_id(character)
        resolved_supported_characters = _resolve_supported_characters(supported_characters)
        resolved_excluded_characters = _resolve_excluded_characters(excluded_characters)
        if resolved_supported_characters is not None and normalized_character not in resolved_supported_characters:
            reasons.append("unsupported_character")
        elif normalized_character in resolved_excluded_characters:
            reasons.append("excluded_character")

    room_type = str(row.get("room_type") or "").lower()
    if room_type not in {"monster", "elite", "boss"}:
        reasons.append("invalid_room_type")

    encounter_id = row.get("encounter_id")
    if not isinstance(encounter_id, str) or not encounter_id:
        reasons.append("missing_encounter_id")
    elif supported_encounter_ids is not None and encounter_id not in supported_encounter_ids:
        reasons.append("unsupported_encounter_id")

    room_model_id = row.get("room_model_id")
    if isinstance(room_model_id, str) and room_model_id and isinstance(encounter_id, str) and encounter_id:
        if room_model_id != encounter_id:
            reasons.append("room_model_mismatch")

    floor_number = row.get("floor_number")
    if not isinstance(floor_number, int) or floor_number < 1:
        reasons.append("invalid_floor_number")

    turns_taken = row.get("turns_taken")
    if not isinstance(turns_taken, int) or turns_taken <= 0:
        reasons.append("invalid_turns_taken")

    current_hp = row.get("snapshot_current_hp")
    max_hp = row.get("snapshot_max_hp")
    if not isinstance(current_hp, int) or not isinstance(max_hp, int) or current_hp <= 0 or max_hp <= 0 or current_hp > max_hp:
        reasons.append("invalid_hp")

    hp_ratio = row.get("snapshot_hp_ratio")
    if not isinstance(hp_ratio, (int, float)) or hp_ratio <= 0 or hp_ratio > 1.0:
        reasons.append("invalid_hp_ratio")

    max_energy = row.get("snapshot_max_energy")
    if not isinstance(max_energy, int) or max_energy <= 0 or max_energy > 10:
        reasons.append("invalid_max_energy")

    snapshot_gold = row.get("snapshot_gold")
    if snapshot_gold is not None and (not isinstance(snapshot_gold, int) or snapshot_gold < 0):
        reasons.append("invalid_gold")

    monster_ids = row.get("monster_ids")
    if not isinstance(monster_ids, list) or not monster_ids or not all(isinstance(value, str) and value for value in monster_ids):
        reasons.append("invalid_monster_ids")

    deck_ids = row.get("deck_card_ids")
    if not isinstance(deck_ids, list) or len(deck_ids) < 5:
        reasons.append("invalid_deck_card_ids")
    elif not all(isinstance(value, str) and value for value in deck_ids):
        reasons.append("invalid_deck_card_ids")

    deck_entries = row.get("deck_entries")
    if deck_entries is not None:
        if not isinstance(deck_entries, list) or len(deck_entries) != len(deck_ids or []):
            reasons.append("invalid_deck_entries")
        else:
            for entry in deck_entries:
                if not isinstance(entry, dict):
                    reasons.append("invalid_deck_entries")
                    break
                card_id = entry.get("id")
                upgrade_level = entry.get("upgrade_level")
                if not isinstance(card_id, str) or not card_id:
                    reasons.append("invalid_deck_entries")
                    break
                if not isinstance(upgrade_level, int) or upgrade_level < 0:
                    reasons.append("invalid_deck_entries")
                    break

    return reasons


def is_playable_combat_snapshot_row(row: dict[str, Any]) -> bool:
    return not validate_combat_snapshot_row_against(row)


def clean_combat_snapshot_rows(
    rows: list[dict[str, Any]],
    *,
    dedupe_by_sample_id: bool = True,
    supported_encounter_ids: set[str] | None = None,
    supported_characters: set[str] | None = None,
    excluded_characters: set[str] | None = None,
    reject_non_vanilla: bool = False,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Filter combat snapshot rows down to the smallest reliable playable set."""

    cleaned: list[dict[str, Any]] = []
    reject_reasons: Counter[str] = Counter()
    dropped_examples: dict[str, dict[str, Any]] = {}
    seen_sample_ids: set[str] = set()
    duplicate_sample_ids = 0

    for row in rows:
        sample_id = str(row.get("sample_id") or "")
        if dedupe_by_sample_id and sample_id:
            if sample_id in seen_sample_ids:
                duplicate_sample_ids += 1
                reject_reasons["duplicate_sample_id"] += 1
                dropped_examples.setdefault("duplicate_sample_id", row)
                continue
            seen_sample_ids.add(sample_id)

        reasons = validate_combat_snapshot_row_against(
            row,
            supported_encounter_ids=supported_encounter_ids,
            supported_characters=supported_characters,
            excluded_characters=excluded_characters,
            reject_non_vanilla=reject_non_vanilla,
        )
        if reasons:
            for reason in reasons:
                reject_reasons[reason] += 1
                dropped_examples.setdefault(reason, row)
            continue

        cleaned.append(row)

    report = {
        "input_rows": len(rows),
        "kept_rows": len(cleaned),
        "dropped_rows": len(rows) - len(cleaned),
        "duplicate_sample_ids": duplicate_sample_ids,
        "reject_reasons": dict(sorted(reject_reasons.items())),
        "dropped_examples": {
            reason: {
                "sample_id": example.get("sample_id"),
                "run_id": example.get("run_id"),
                "character": example.get("character"),
                "encounter_id": example.get("encounter_id"),
                "floor_number": example.get("floor_number"),
            }
            for reason, example in sorted(dropped_examples.items())
        },
    }
    return cleaned, report


def load_combat_snapshot_rows(
    path: str | Path,
    *,
    curated_subset: str | None = None,
    split: str | None = None,
    character: str | None = None,
    build_id: str | None = None,
    encounter_ids: list[str] | None = None,
    encounter_tiers: list[str] | None = None,
    min_floor: int | None = None,
    max_floor: int | None = None,
    max_rows: int | None = None,
    strict_playable_only: bool = True,
    supported_encounter_ids: set[str] | None = None,
    supported_characters: set[str] | None = None,
    excluded_characters: set[str] | None = None,
) -> list[dict[str, Any]]:
    dataset_path = resolve_combat_snapshot_dataset_path(path, curated_subset=curated_subset)
    suffix = dataset_path.suffix.lower()
    if suffix == ".jsonl":
        rows = _load_jsonl_rows(dataset_path)
    elif suffix == ".parquet":
        rows = _load_parquet_rows(dataset_path)
    else:
        raise ValueError(f"Unsupported combat snapshot dataset format: {dataset_path}")

    encounter_filter = {str(value) for value in (encounter_ids or []) if value}
    encounter_tier_filter = {str(value).strip().lower() for value in (encounter_tiers or []) if value}
    invalid_tiers = sorted(encounter_tier_filter.difference(VALID_ENCOUNTER_TIERS))
    if invalid_tiers:
        raise ValueError(f"Unsupported encounter tiers: {invalid_tiers}")
    filtered: list[dict[str, Any]] = []
    resolved_supported_characters = _resolve_supported_characters(supported_characters)
    resolved_excluded_characters = _resolve_excluded_characters(excluded_characters)
    for row in rows:
        if split and str(row.get("split") or "") != split:
            continue
        if character and str(row.get("character") or "") != character:
            continue
        row_character = _normalize_character_id(row.get("character"))
        if resolved_supported_characters is not None and row_character not in resolved_supported_characters:
            continue
        if row_character in resolved_excluded_characters:
            continue
        if build_id and str(row.get("build_id") or "") != build_id:
            continue
        if encounter_filter and str(row.get("encounter_id") or "") not in encounter_filter:
            continue
        if encounter_tier_filter and infer_encounter_tier_from_row(row) not in encounter_tier_filter:
            continue

        floor_number = row.get("floor_number")
        if min_floor is not None:
            if not isinstance(floor_number, int) or floor_number < min_floor:
                continue
        if max_floor is not None:
            if not isinstance(floor_number, int) or floor_number > max_floor:
                continue

        if strict_playable_only and validate_combat_snapshot_row_against(
            row,
            supported_encounter_ids=supported_encounter_ids,
            supported_characters=resolved_supported_characters,
            excluded_characters=resolved_excluded_characters,
        ):
            continue

        filtered.append(row)

    if max_rows is not None and max_rows > 0:
        filtered = filtered[:max_rows]
    return filtered


class CombatSnapshotPool:
    """In-memory sampler for combat snapshot rows."""

    def __init__(
        self,
        rows: list[dict[str, Any]],
        *,
        sample_mode: str = "encounter_balanced",
        tier_weights: dict[str, float] | None = None,
        encounter_weights: dict[str, float] | None = None,
        starter_early_boost: float = 0.0,
    ) -> None:
        if not rows:
            raise ValueError("CombatSnapshotPool requires at least one row")
        if sample_mode not in {"row_uniform", "encounter_balanced", "tier_weighted_encounter_balanced"}:
            raise ValueError(f"Unsupported sample_mode: {sample_mode}")
        if not isinstance(starter_early_boost, (int, float)):
            raise ValueError("starter_early_boost must be numeric")
        starter_early_boost_f = float(starter_early_boost)
        if not 0.0 <= starter_early_boost_f <= 1.0:
            raise ValueError(
                f"starter_early_boost must be in [0, 1], got {starter_early_boost_f}"
            )

        self.rows = rows
        self.sample_mode = sample_mode
        self.starter_early_boost = starter_early_boost_f
        self.tier_weights = {
            str(tier).strip().lower(): float(weight)
            for tier, weight in (tier_weights or {}).items()
            if float(weight) > 0.0
        }
        self.encounter_weights = {
            str(encounter_id).strip(): float(weight)
            for encounter_id, weight in (encounter_weights or {}).items()
            if float(weight) > 0.0
        }
        self._rows_by_encounter: dict[str, list[dict[str, Any]]] = {}
        self._rows_by_tier_encounter: dict[str, dict[str, list[dict[str, Any]]]] = {}
        self._starter_early_rows_by_encounter: dict[str, list[dict[str, Any]]] = {}
        for row in rows:
            encounter_id = str(row.get("encounter_id") or "")
            if not encounter_id:
                continue
            self._rows_by_encounter.setdefault(encounter_id, []).append(row)
            tier = infer_encounter_tier_from_row(row)
            self._rows_by_tier_encounter.setdefault(tier, {}).setdefault(encounter_id, []).append(row)
            if infer_deck_stage_from_row(row) == "starter_early":
                self._starter_early_rows_by_encounter.setdefault(encounter_id, []).append(row)
        self._encounter_ids = sorted(self._rows_by_encounter.keys())
        self._tier_ids = sorted(self._rows_by_tier_encounter.keys())
        self._starter_early_encounter_ids = sorted(self._starter_early_rows_by_encounter.keys())
        if not self._encounter_ids:
            raise ValueError("CombatSnapshotPool found no usable encounter_id rows")
        if sample_mode == "tier_weighted_encounter_balanced" and not self._tier_ids:
            raise ValueError("CombatSnapshotPool found no encounter tiers for weighted sampling")
        if starter_early_boost_f > 0.0 and not self._starter_early_encounter_ids:
            raise ValueError(
                "CombatSnapshotPool starter_early_boost > 0 but no starter_early rows in the pool "
                f"(floor<={STARTER_EARLY_MAX_FLOOR} AND deck_size<={STARTER_EARLY_MAX_DECK_SIZE})."
            )

        self._normalized_tier_weights = self._build_normalized_tier_weights()

    @classmethod
    def from_path(
        cls,
        path: str | Path,
        *,
        curated_subset: str | None = None,
        split: str | None = None,
        character: str | None = None,
        build_id: str | None = None,
        encounter_ids: list[str] | None = None,
        encounter_tiers: list[str] | None = None,
        min_floor: int | None = None,
        max_floor: int | None = None,
        max_rows: int | None = None,
        sample_mode: str = "encounter_balanced",
        tier_weights: dict[str, float] | None = None,
        encounter_weights: dict[str, float] | None = None,
        starter_early_boost: float = 0.0,
        supported_encounter_ids: set[str] | None = None,
        supported_characters: set[str] | list[str] | tuple[str, ...] | None = None,
        excluded_characters: set[str] | list[str] | tuple[str, ...] | None = None,
    ) -> "CombatSnapshotPool":
        rows = load_combat_snapshot_rows(
            path,
            curated_subset=curated_subset,
            split=split,
            character=character,
            build_id=build_id,
            encounter_ids=encounter_ids,
            encounter_tiers=encounter_tiers,
            min_floor=min_floor,
            max_floor=max_floor,
            max_rows=max_rows,
            supported_encounter_ids=supported_encounter_ids,
            supported_characters=supported_characters,
            excluded_characters=excluded_characters,
        )
        return cls(
            rows,
            sample_mode=sample_mode,
            tier_weights=tier_weights,
            encounter_weights=encounter_weights,
            starter_early_boost=starter_early_boost,
        )

    def __len__(self) -> int:
        return len(self.rows)

    @property
    def encounter_count(self) -> int:
        return len(self._encounter_ids)

    def summary(self) -> dict[str, Any]:
        row_count = len(self.rows)
        characters = sorted({str(row.get("character") or "unknown") for row in self.rows})
        character_counts = Counter(str(row.get("character") or "unknown") for row in self.rows)
        build_ids = sorted({str(row.get("build_id") or "unknown") for row in self.rows})
        floors = [int(row.get("floor_number")) for row in self.rows if isinstance(row.get("floor_number"), int)]
        tier_counts = Counter(infer_encounter_tier_from_row(row) for row in self.rows)
        deck_stage_counts = Counter(infer_deck_stage_from_row(row) for row in self.rows)
        starter_early_row_count = sum(
            len(rows) for rows in self._starter_early_rows_by_encounter.values()
        )
        encounter_sizes = {encounter_id: len(rows) for encounter_id, rows in self._rows_by_encounter.items()}
        top_encounters = sorted(
            encounter_sizes.items(),
            key=lambda item: (-item[1], item[0]),
        )[:10]
        return {
            "row_count": row_count,
            "encounter_count": self.encounter_count,
            "characters": characters,
            "character_counts": dict(sorted(character_counts.items())),
            "build_ids": build_ids,
            "min_floor": min(floors) if floors else None,
            "max_floor": max(floors) if floors else None,
            "sample_mode": self.sample_mode,
            "tier_counts": dict(sorted(tier_counts.items())),
            "tier_sampling_weights": dict(sorted(self._normalized_tier_weights.items())),
            "deck_stage_counts": dict(sorted(deck_stage_counts.items())),
            "starter_early_boost": self.starter_early_boost,
            "starter_early_row_count": starter_early_row_count,
            "starter_early_encounter_count": len(self._starter_early_encounter_ids),
            "encounter_weight_overrides": dict(sorted(self.encounter_weights.items())[:10]),
            "top_encounters": top_encounters,
        }

    def sample(self, rng) -> dict[str, Any]:
        if (
            self.starter_early_boost > 0.0
            and self._starter_early_encounter_ids
            and float(rng.random()) < self.starter_early_boost
        ):
            return self._sample_starter_early(rng)

        if self.sample_mode == "row_uniform":
            index = int(rng.integers(len(self.rows)))
            return self.rows[index]

        if self.sample_mode == "tier_weighted_encounter_balanced":
            tier_probs = np.asarray([self._normalized_tier_weights[tier] for tier in self._tier_ids], dtype=np.float64)
            tier_index = int(rng.choice(len(self._tier_ids), p=tier_probs))
            tier_id = self._tier_ids[tier_index]
            encounter_ids = sorted(self._rows_by_tier_encounter[tier_id].keys())
            encounter_probs = self._encounter_probabilities(encounter_ids)
            encounter_index = int(rng.choice(len(encounter_ids), p=encounter_probs))
            encounter_id = encounter_ids[encounter_index]
            encounter_rows = self._rows_by_tier_encounter[tier_id][encounter_id]
            row_index = int(rng.integers(len(encounter_rows)))
            return encounter_rows[row_index]

        encounter_probs = self._encounter_probabilities(self._encounter_ids)
        encounter_index = int(rng.choice(len(self._encounter_ids), p=encounter_probs))
        encounter_id = self._encounter_ids[encounter_index]
        encounter_rows = self._rows_by_encounter[encounter_id]
        row_index = int(rng.integers(len(encounter_rows)))
        return encounter_rows[row_index]

    def _sample_starter_early(self, rng) -> dict[str, Any]:
        """Encounter-balanced sample over the starter_early bucket.

        Using encounter-balanced regardless of the outer sample_mode keeps
        the boost's intent crisp: we want fair coverage over *which weak
        monster fight* the starter deck is facing, not the underlying
        pool's tier/encounter distribution which is already skewed toward
        later-game rows.
        """
        encounter_ids = self._starter_early_encounter_ids
        encounter_probs = self._encounter_probabilities(encounter_ids)
        encounter_index = int(rng.choice(len(encounter_ids), p=encounter_probs))
        encounter_id = encounter_ids[encounter_index]
        encounter_rows = self._starter_early_rows_by_encounter[encounter_id]
        row_index = int(rng.integers(len(encounter_rows)))
        return encounter_rows[row_index]

    def _build_normalized_tier_weights(self) -> dict[str, float]:
        if not self._tier_ids:
            return {}

        if not self.tier_weights:
            uniform = 1.0 / float(len(self._tier_ids))
            return {tier: uniform for tier in self._tier_ids}

        weights: dict[str, float] = {}
        total = 0.0
        for tier in self._tier_ids:
            weight = float(self.tier_weights.get(tier, 0.0))
            if weight > 0.0:
                weights[tier] = weight
                total += weight
        if total <= 0.0:
            raise ValueError(
                "CombatSnapshotPool tier_weighted_encounter_balanced needs at least one positive tier weight "
                f"among available tiers {self._tier_ids!r}."
            )
        return {tier: weight / total for tier, weight in weights.items()}

    def _encounter_probabilities(self, encounter_ids: list[str]) -> np.ndarray:
        if not encounter_ids:
            raise ValueError("CombatSnapshotPool cannot sample from an empty encounter id list")

        if not self.encounter_weights:
            return np.full(len(encounter_ids), 1.0 / float(len(encounter_ids)), dtype=np.float64)

        weights = np.asarray(
            [max(float(self.encounter_weights.get(encounter_id, 1.0)), 0.0) for encounter_id in encounter_ids],
            dtype=np.float64,
        )
        total = float(weights.sum())
        if total <= 0.0:
            return np.full(len(encounter_ids), 1.0 / float(len(encounter_ids)), dtype=np.float64)
        return weights / total


def snapshot_row_to_reset_kwargs(
    row: dict[str, Any],
    *,
    include_potions: bool = False,
) -> dict[str, Any]:
    """Convert a combat snapshot row into ``combat_reset`` kwargs."""

    deck_entries: list[dict[str, Any]] | None = None
    sanitized_deck_ids_from_entries: list[str] | None = None
    raw_deck_entries = row.get("deck_entries")
    if isinstance(raw_deck_entries, list):
        normalized_entries: list[dict[str, Any]] = []
        for entry in raw_deck_entries:
            if not isinstance(entry, dict):
                continue
            card_id = str(entry.get("id") or "").strip()
            if not card_id:
                continue
            if not _snapshot_deck_card_is_supported(card_id):
                continue
            try:
                upgrade_level = max(int(entry.get("upgrade_level") or 0), 0)
            except (TypeError, ValueError):
                upgrade_level = 0
            upgrade_level = _clamp_card_upgrade_level(card_id, upgrade_level)
            normalized_entries.append(
                {
                    "id": card_id,
                    "upgrade_level": upgrade_level,
                }
            )
        sanitized_deck_ids_from_entries = [str(entry["id"]) for entry in normalized_entries]
        if normalized_entries:
            deck_entries = normalized_entries

    raw_deck_ids = [str(value) for value in (row.get("deck_card_ids") or []) if value]
    sanitized_deck_ids = (
        sanitized_deck_ids_from_entries
        if sanitized_deck_ids_from_entries is not None
        else _sanitize_snapshot_deck_card_ids(raw_deck_ids)
    )

    potions = None
    if include_potions and row.get("potion_state_known"):
        raw_potions = row.get("potion_ids_before")
        if isinstance(raw_potions, list):
            potions = [str(value) for value in raw_potions if value]

    # Combat sandbox training should start from a clean tactical state.
    # We keep the historical max HP / build snapshot, but default the live
    # current HP to full so the policy learns encounter handling instead of
    # inheriting arbitrary prior-run chip damage.
    current_hp = row.get("snapshot_max_hp")
    if not isinstance(current_hp, int) or current_hp <= 0:
        current_hp = row.get("snapshot_current_hp")

    return {
        "character": row.get("character"),
        "encounter_id": row.get("encounter_id"),
        "current_hp": current_hp,
        "max_hp": row.get("snapshot_max_hp"),
        "max_energy": row.get("snapshot_max_energy"),
        "deck": sanitized_deck_ids,
        "deck_entries": deck_entries,
        "relics": [str(value) for value in (row.get("relic_ids_before") or []) if value],
        "potions": potions,
        "gold": row.get("snapshot_gold"),
    }
