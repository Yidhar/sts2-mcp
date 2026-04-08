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
from pathlib import Path
from typing import Any


def resolve_combat_snapshot_dataset_path(path: str | Path) -> Path:
    """Resolve a user-provided path to a concrete combat snapshot dataset file.

    Accepted inputs:
    - direct JSONL / Parquet file path
    - dataset root containing ``combat_snapshot_samples.jsonl``
    - dataset root containing ``parquet/combat_snapshot_samples.parquet``
    - partition dir such as ``.../by_build_family/v0.99.1``
    """

    input_path = Path(path)
    if input_path.is_file():
        return input_path

    if not input_path.exists():
        raise FileNotFoundError(f"Combat snapshot dataset path does not exist: {input_path}")

    candidates = [
        input_path / "combat_snapshot_samples.parquet",
        input_path / "combat_snapshot_samples.jsonl",
        input_path / "parquet" / "combat_snapshot_samples.parquet",
        input_path / "parquet" / "combat_snapshot_samples.jsonl",
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
) -> list[str]:
    reasons: list[str] = []
    sample_id = row.get("sample_id")
    if not isinstance(sample_id, str) or not sample_id:
        reasons.append("missing_sample_id")

    character = row.get("character")
    if not isinstance(character, str) or not character:
        reasons.append("missing_character")

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
    split: str | None = None,
    character: str | None = None,
    build_id: str | None = None,
    encounter_ids: list[str] | None = None,
    min_floor: int | None = None,
    max_floor: int | None = None,
    max_rows: int | None = None,
    strict_playable_only: bool = True,
    supported_encounter_ids: set[str] | None = None,
) -> list[dict[str, Any]]:
    dataset_path = resolve_combat_snapshot_dataset_path(path)
    suffix = dataset_path.suffix.lower()
    if suffix == ".jsonl":
        rows = _load_jsonl_rows(dataset_path)
    elif suffix == ".parquet":
        rows = _load_parquet_rows(dataset_path)
    else:
        raise ValueError(f"Unsupported combat snapshot dataset format: {dataset_path}")

    encounter_filter = {str(value) for value in (encounter_ids or []) if value}
    filtered: list[dict[str, Any]] = []
    for row in rows:
        if split and str(row.get("split") or "") != split:
            continue
        if character and str(row.get("character") or "") != character:
            continue
        if build_id and str(row.get("build_id") or "") != build_id:
            continue
        if encounter_filter and str(row.get("encounter_id") or "") not in encounter_filter:
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
    ) -> None:
        if not rows:
            raise ValueError("CombatSnapshotPool requires at least one row")
        if sample_mode not in {"row_uniform", "encounter_balanced"}:
            raise ValueError(f"Unsupported sample_mode: {sample_mode}")

        self.rows = rows
        self.sample_mode = sample_mode
        self._rows_by_encounter: dict[str, list[dict[str, Any]]] = {}
        for row in rows:
            encounter_id = str(row.get("encounter_id") or "")
            if not encounter_id:
                continue
            self._rows_by_encounter.setdefault(encounter_id, []).append(row)
        self._encounter_ids = sorted(self._rows_by_encounter.keys())
        if not self._encounter_ids:
            raise ValueError("CombatSnapshotPool found no usable encounter_id rows")

    @classmethod
    def from_path(
        cls,
        path: str | Path,
        *,
        split: str | None = None,
        character: str | None = None,
        build_id: str | None = None,
        encounter_ids: list[str] | None = None,
        min_floor: int | None = None,
        max_floor: int | None = None,
        max_rows: int | None = None,
        sample_mode: str = "encounter_balanced",
    ) -> "CombatSnapshotPool":
        rows = load_combat_snapshot_rows(
            path,
            split=split,
            character=character,
            build_id=build_id,
            encounter_ids=encounter_ids,
            min_floor=min_floor,
            max_floor=max_floor,
            max_rows=max_rows,
        )
        return cls(rows, sample_mode=sample_mode)

    def __len__(self) -> int:
        return len(self.rows)

    @property
    def encounter_count(self) -> int:
        return len(self._encounter_ids)

    def summary(self) -> dict[str, Any]:
        row_count = len(self.rows)
        characters = sorted({str(row.get("character") or "unknown") for row in self.rows})
        build_ids = sorted({str(row.get("build_id") or "unknown") for row in self.rows})
        floors = [int(row.get("floor_number")) for row in self.rows if isinstance(row.get("floor_number"), int)]
        encounter_sizes = {encounter_id: len(rows) for encounter_id, rows in self._rows_by_encounter.items()}
        top_encounters = sorted(
            encounter_sizes.items(),
            key=lambda item: (-item[1], item[0]),
        )[:10]
        return {
            "row_count": row_count,
            "encounter_count": self.encounter_count,
            "characters": characters,
            "build_ids": build_ids,
            "min_floor": min(floors) if floors else None,
            "max_floor": max(floors) if floors else None,
            "sample_mode": self.sample_mode,
            "top_encounters": top_encounters,
        }

    def sample(self, rng) -> dict[str, Any]:
        if self.sample_mode == "row_uniform":
            index = int(rng.integers(len(self.rows)))
            return self.rows[index]

        encounter_index = int(rng.integers(len(self._encounter_ids)))
        encounter_id = self._encounter_ids[encounter_index]
        encounter_rows = self._rows_by_encounter[encounter_id]
        row_index = int(rng.integers(len(encounter_rows)))
        return encounter_rows[row_index]


def snapshot_row_to_reset_kwargs(row: dict[str, Any]) -> dict[str, Any]:
    """Convert a combat snapshot row into ``combat_reset`` kwargs."""

    potions = None
    if row.get("potion_state_known"):
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
        "deck": [str(value) for value in (row.get("deck_card_ids") or []) if value],
        "relics": [str(value) for value in (row.get("relic_ids_before") or []) if value],
        "potions": potions,
        "gold": row.get("snapshot_gold"),
    }
