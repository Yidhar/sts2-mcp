from __future__ import annotations

from muzero.training.cli_parsing import (
    parse_encounter_pool,
    parse_encounter_weights,
    resolve_combat_sandbox_encounter_pool,
)


DEFAULT_POOL = (
    "ENCOUNTER.SLIMES_WEAK,"
    "ENCOUNTER.SHRINKER_BEETLE_WEAK,"
    "ENCOUNTER.CONSTRUCT_MENAGERIE_NORMAL"
)


def test_snapshot_dataset_without_explicit_pool_does_not_apply_default_pool() -> None:
    pool = resolve_combat_sandbox_encounter_pool(
        explicit_pool=None,
        combat_snapshot_dataset="/tmp/curated_combat_dataset",
        encounter_tiers=["weak", "normal"],
        default_pool=DEFAULT_POOL,
    )

    assert pool == []


def test_no_snapshot_dataset_uses_default_pool_and_tier_filter() -> None:
    pool = resolve_combat_sandbox_encounter_pool(
        explicit_pool=None,
        combat_snapshot_dataset=None,
        encounter_tiers=["normal"],
        default_pool=DEFAULT_POOL,
    )

    assert pool == ["ENCOUNTER.CONSTRUCT_MENAGERIE_NORMAL"]


def test_explicit_pool_still_filters_snapshot_dataset_when_requested() -> None:
    pool = resolve_combat_sandbox_encounter_pool(
        explicit_pool="ENCOUNTER.SLIMES_WEAK,ENCOUNTER.CONSTRUCT_MENAGERIE_NORMAL",
        combat_snapshot_dataset="/tmp/curated_combat_dataset",
        encounter_tiers=["normal"],
        default_pool=DEFAULT_POOL,
    )

    assert pool == ["ENCOUNTER.CONSTRUCT_MENAGERIE_NORMAL"]


def test_encounter_pool_accepts_lowercase_and_bare_aliases() -> None:
    pool = parse_encounter_pool("encounter.ovicopter_normal, fabricator_normal")

    assert pool == ["ENCOUNTER.OVICOPTER_NORMAL", "ENCOUNTER.FABRICATOR_NORMAL"]


def test_encounter_weights_are_canonicalized() -> None:
    weights = parse_encounter_weights(
        "encounter.ovicopter_normal=8.0, fabricator_normal=7.5"
    )

    assert weights == {
        "ENCOUNTER.OVICOPTER_NORMAL": 8.0,
        "ENCOUNTER.FABRICATOR_NORMAL": 7.5,
    }
