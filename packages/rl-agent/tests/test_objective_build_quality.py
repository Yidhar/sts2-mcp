"""Regression tests for objective-head build-quality shaping."""

from __future__ import annotations

import sys
import unittest
from pathlib import Path


RL_AGENT_ROOT = Path(__file__).resolve().parents[1]
if str(RL_AGENT_ROOT) not in sys.path:
    sys.path.insert(0, str(RL_AGENT_ROOT))

from sts2_env.objective_heads import _build_quality_score


def _attack(card_id: str, *, damage: int = 6, cost: int = 1) -> dict:
    return {
        "id": card_id,
        "type": "attack",
        "cost": cost,
        "effect_preview": {"damage": damage},
    }


def _block(card_id: str, *, block: int = 5, cost: int = 1) -> dict:
    return {
        "id": card_id,
        "type": "skill",
        "cost": cost,
        "effect_preview": {"block": block},
    }


def _obs(deck_cards: list[dict]) -> dict:
    return {"player": {"deck_cards": deck_cards}}


class ObjectiveBuildQualityTests(unittest.TestCase):
    def test_mediocre_attack_bloat_after_twelve_cards_lowers_build_quality(self):
        compact_act1_deck = [
            *[_attack(f"CARD.STRIKE_{i}") for i in range(5)],
            *[_block(f"CARD.DEFEND_{i}") for i in range(4)],
            _attack("CARD.BASH", damage=8, cost=2),
            _attack("CARD.POMMEL_STRIKE", damage=9),
            _block("CARD.SHRUG_IT_OFF", block=8),
        ]
        bloated_attack_deck = [
            *compact_act1_deck,
            *[_attack(f"CARD.MEDIOCRE_EXTRA_ATTACK_{i}", damage=6) for i in range(6)],
        ]

        self.assertLess(
            _build_quality_score(_obs(bloated_attack_deck)),
            _build_quality_score(_obs(compact_act1_deck)),
        )


if __name__ == "__main__":
    unittest.main()
