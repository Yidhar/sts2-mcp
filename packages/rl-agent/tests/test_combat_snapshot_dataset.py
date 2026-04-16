from __future__ import annotations

from combat_snapshot_dataset import snapshot_row_to_reset_kwargs


def test_snapshot_row_to_reset_kwargs_filters_unresolved_template_card_from_deck_entries() -> None:
    row = {
        "character": "CHARACTER.IRONCLAD",
        "encounter_id": "ENCOUNTER.LOUSE_WEAK",
        "snapshot_max_hp": 80,
        "snapshot_max_energy": 3,
        "snapshot_gold": 99,
        "deck_entries": [
            {"id": "CARD.MAD_SCIENCE", "upgrade_level": 1},
            {"id": "CARD.STRIKE_IRONCLAD", "upgrade_level": 0},
            {"id": "CARD.DEFEND_IRONCLAD", "upgrade_level": 1},
        ],
        "deck_card_ids": [
            "CARD.MAD_SCIENCE",
            "CARD.STRIKE_IRONCLAD",
            "CARD.DEFEND_IRONCLAD",
        ],
        "relic_ids_before": ["RELIC.BURNING_BLOOD"],
    }

    result = snapshot_row_to_reset_kwargs(row)

    assert result["deck"] == [
        "CARD.STRIKE_IRONCLAD",
        "CARD.DEFEND_IRONCLAD",
    ]
    assert result["deck_entries"] == [
        {"id": "CARD.STRIKE_IRONCLAD", "upgrade_level": 0},
        {"id": "CARD.DEFEND_IRONCLAD", "upgrade_level": 1},
    ]


def test_snapshot_row_to_reset_kwargs_keeps_well_formed_event_attack_cards() -> None:
    row = {
        "character": "CHARACTER.IRONCLAD",
        "encounter_id": "ENCOUNTER.LOUSE_WEAK",
        "snapshot_max_hp": 80,
        "snapshot_max_energy": 3,
        "snapshot_gold": 99,
        "deck_entries": [
            {"id": "CARD.BYRD_SWOOP", "upgrade_level": 0},
            {"id": "CARD.STRIKE_IRONCLAD", "upgrade_level": 0},
        ],
        "deck_card_ids": [
            "CARD.BYRD_SWOOP",
            "CARD.STRIKE_IRONCLAD",
        ],
        "relic_ids_before": ["RELIC.BURNING_BLOOD"],
    }

    result = snapshot_row_to_reset_kwargs(row)

    assert result["deck"] == [
        "CARD.BYRD_SWOOP",
        "CARD.STRIKE_IRONCLAD",
    ]
    assert result["deck_entries"] == [
        {"id": "CARD.BYRD_SWOOP", "upgrade_level": 0},
        {"id": "CARD.STRIKE_IRONCLAD", "upgrade_level": 0},
    ]


def test_snapshot_row_to_reset_kwargs_filters_unresolved_template_card_from_deck_ids_without_entries() -> None:
    row = {
        "character": "CHARACTER.IRONCLAD",
        "encounter_id": "ENCOUNTER.LOUSE_WEAK",
        "snapshot_max_hp": 80,
        "snapshot_max_energy": 3,
        "snapshot_gold": 99,
        "deck_card_ids": [
            "CARD.MAD_SCIENCE",
            "CARD.STRIKE_IRONCLAD",
        ],
        "relic_ids_before": ["RELIC.BURNING_BLOOD"],
    }

    result = snapshot_row_to_reset_kwargs(row)

    assert result["deck"] == ["CARD.STRIKE_IRONCLAD"]
    assert result["deck_entries"] is None
