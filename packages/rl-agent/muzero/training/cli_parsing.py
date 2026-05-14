"""Small CLI parsing helpers for MuZero training entrypoints."""

from __future__ import annotations

from combat_snapshot_dataset import (
    VALID_ENCOUNTER_TIERS,
    infer_encounter_tier,
    normalize_encounter_id,
)
from sts2_env.path_utils import normalize_path_str


def parse_encounter_pool(raw: str | None) -> list[str]:
    """Parse comma-separated encounter pool."""
    if not raw:
        return []
    return [
        normalize_encounter_id(entry)
        for entry in raw.split(",")
        if normalize_encounter_id(entry)
    ]


def resolve_combat_sandbox_encounter_pool(
    *,
    explicit_pool: str | None,
    combat_snapshot_dataset: str | None,
    encounter_tiers: list[str],
    default_pool: str,
) -> list[str]:
    """Resolve the encounter pool used by combat sandbox resets.

    Important distinction:
    - Without a snapshot dataset, the bridge needs a concrete fallback pool,
      so we use the historical weak-fight default.
    - With a snapshot dataset, the rows already carry their own encounter_id.
      Unless the caller explicitly passes ``--encounter-pool``, do not apply
      the historical fallback pool as an implicit filter; doing so silently
      collapses a weak+normal curated dataset down to only the four default
      weak encounters.

    ``encounter_tiers`` still filters any explicit/default concrete pool.
    Snapshot-only tier filtering is handled by CombatSnapshotPool.from_path.
    """

    if explicit_pool:
        pool = parse_encounter_pool(explicit_pool)
    elif combat_snapshot_dataset:
        pool = []
    else:
        pool = parse_encounter_pool(default_pool)

    if encounter_tiers and pool:
        tier_filter = {str(tier).strip().lower() for tier in encounter_tiers if str(tier).strip()}
        pool = [
            encounter_id
            for encounter_id in pool
            if infer_encounter_tier(encounter_id) in tier_filter
        ]
    return pool


def parse_session_files(raw: str | None) -> list[str]:
    """Parse comma-separated session files."""
    if not raw:
        return []
    return [
        normalize_path_str(entry.strip()) or entry.strip()
        for entry in raw.split(",")
        if entry.strip()
    ]


def parse_encounter_tiers(raw: str | None) -> list[str]:
    """Parse comma-separated encounter tiers."""
    if not raw:
        return []
    tiers = [entry.strip().lower() for entry in raw.split(",") if entry.strip()]
    invalid = sorted(set(tiers).difference(VALID_ENCOUNTER_TIERS))
    if invalid:
        raise ValueError(f"Unsupported --combat-encounter-tiers values: {invalid}")
    return tiers


def parse_tier_weights(raw: str | None) -> dict[str, float]:
    """Parse comma-separated tier weights like ``weak=0.6,normal=0.4``."""
    if not raw:
        return {}

    weights: dict[str, float] = {}
    for chunk in raw.split(","):
        entry = chunk.strip()
        if not entry:
            continue
        if "=" not in entry:
            raise ValueError(
                f"Invalid --combat-tier-weights entry {entry!r}; expected tier=weight pairs."
            )
        tier_raw, weight_raw = entry.split("=", 1)
        tier = tier_raw.strip().lower()
        if tier not in VALID_ENCOUNTER_TIERS:
            raise ValueError(f"Unsupported encounter tier in --combat-tier-weights: {tier!r}")
        try:
            weight = float(weight_raw)
        except ValueError as exc:
            raise ValueError(
                f"Invalid weight {weight_raw!r} for encounter tier {tier!r}."
            ) from exc
        if weight < 0.0:
            raise ValueError(f"Encounter tier weight must be >= 0 for {tier!r}, got {weight}.")
        weights[tier] = weight
    return weights


def parse_encounter_weights(raw: str | None) -> dict[str, float]:
    """Parse comma-separated encounter weights like ``ENCOUNTER.X=3.0``."""
    if not raw:
        return {}

    weights: dict[str, float] = {}
    for chunk in raw.split(","):
        entry = chunk.strip()
        if not entry:
            continue
        if "=" not in entry:
            raise ValueError(
                f"Invalid --combat-encounter-weights entry {entry!r}; expected encounter=weight pairs."
            )
        encounter_raw, weight_raw = entry.split("=", 1)
        encounter_id = normalize_encounter_id(encounter_raw)
        if not encounter_id:
            raise ValueError("Encounter id in --combat-encounter-weights cannot be empty.")
        try:
            weight = float(weight_raw)
        except ValueError as exc:
            raise ValueError(
                f"Invalid weight {weight_raw!r} for encounter {encounter_id!r}."
            ) from exc
        if weight < 0.0:
            raise ValueError(
                f"Encounter weight must be >= 0 for {encounter_id!r}, got {weight}."
            )
        weights[encounter_id] = weight
    return weights


def parse_int_list(raw: str | None) -> tuple[int, ...]:
    """Parse comma-separated positive integers."""
    if not raw:
        return ()
    values: list[int] = []
    for chunk in raw.split(","):
        entry = chunk.strip()
        if not entry:
            continue
        try:
            value = int(entry)
        except ValueError as exc:
            raise ValueError(f"Invalid integer entry {entry!r}.") from exc
        if value <= 0:
            raise ValueError(f"Expected positive integer in list, got {value}.")
        values.append(value)
    return tuple(values)
