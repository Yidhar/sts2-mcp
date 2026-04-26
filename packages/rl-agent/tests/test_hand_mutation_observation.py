from __future__ import annotations

import sys
import unittest
from pathlib import Path

RL_AGENT_ROOT = Path(__file__).resolve().parents[1]
if str(RL_AGENT_ROOT) not in sys.path:
    sys.path.insert(0, str(RL_AGENT_ROOT))

from sts2_env.observation_v3 import TOKEN_TYPE_TO_ID, WorldTokenObservationEncoder


def _card(
    title: str,
    *,
    card_type: str = "Attack",
    cost: int = 1,
    damage: int = 0,
    block: int = 0,
    text: str = "",
    upgrade_level: int = 0,
) -> dict[str, object]:
    return {
        "id": title.upper().replace(" ", "_"),
        "title": title,
        "type": card_type,
        "cost": cost,
        "damage": damage,
        "block": block,
        "preview_damage": damage,
        "preview_block": block,
        "description": text,
        "text": text,
        "upgrade_level": upgrade_level,
        "can_upgrade": upgrade_level == 0,
        "can_play": True,
    }


def _obs_with_action(source: dict[str, object], hand: list[dict[str, object]]) -> tuple[dict[str, object], list[dict[str, object]]]:
    if source not in hand:
        hand = [source] + hand
    obs = {
        "phase": "combat",
        "player": {"hp": 50, "max_hp": 80, "block": 0, "relics": [], "potions": []},
        "combat": {
            "round": 1,
            "energy": 3,
            "max_energy": 3,
            "player_powers": [],
            "hand": hand,
            "draw_pile": [],
            "discard_pile": [],
            "exhaust_pile": [],
            "enemies": [],
        },
    }
    return obs, [{"kind": "play_card", "card": source}]


def _locals(encoded: dict[str, object], token_type: str):
    wanted = TOKEN_TYPE_TO_ID[token_type]
    rows = []
    for i, current in enumerate(encoded["candidate_local_type_ids"][0]):
        if encoded["candidate_local_masks"][0][i] > 0 and int(current) == wanted:
            rows.append(encoded["candidate_local_tokens"][0, i, :96])
    return rows


class HandMutationObservationTest(unittest.TestCase):
    def test_upgrade_one_hand_effect_is_generic_not_armaments_specific(self) -> None:
        source = _card("Practice Smith", card_type="Skill", cost=1, text="Upgrade a card in your hand.")
        strike = _card("Strike", damage=6)
        defend = _card("Defend", card_type="Skill", block=5)
        obs, actions = _obs_with_action(source, [source, strike, defend])

        encoded = WorldTokenObservationEncoder(use_text=False).encode(obs, actions)
        summary = _locals(encoded, "HAND_MUTATION_LOCAL")[0]
        targets = _locals(encoded, "HAND_MUTATION_TARGET_LOCAL")

        self.assertEqual(float(summary[0]), 1.0)  # will_mutate_hand
        self.assertEqual(float(summary[1]), 1.0)  # upgrade_hand
        self.assertEqual(float(summary[2]), 1.0)  # upgrade_one
        self.assertEqual(float(summary[3]), 0.0)  # not upgrade_all
        self.assertGreater(float(summary[19]), 0.0)  # upgradeable_count_norm
        self.assertTrue(any(float(t[3]) == 1.0 for t in targets))  # would_upgrade target

    def test_upgrade_all_hand_effect_surfaces_multiple_targets(self) -> None:
        source = _card("Battle Prep", card_type="Skill", cost=1, text="Upgrade all cards in your hand.", upgrade_level=1)
        strike = _card("Strike", damage=6)
        defend = _card("Defend", card_type="Skill", block=5)
        obs, actions = _obs_with_action(source, [source, strike, defend])

        encoded = WorldTokenObservationEncoder(use_text=False).encode(obs, actions)
        summary = _locals(encoded, "HAND_MUTATION_LOCAL")[0]
        targets = _locals(encoded, "HAND_MUTATION_TARGET_LOCAL")

        self.assertEqual(float(summary[1]), 1.0)
        self.assertEqual(float(summary[3]), 1.0)  # upgrade_all
        self.assertGreaterEqual(sum(1 for t in targets if float(t[3]) == 1.0), 2)
        post = _locals(encoded, "POST_HAND_PREVIEW_LOCAL")[0]
        self.assertGreater(float(post[1]), 0.0)  # post upgraded count

    def test_cost_to_zero_hand_effect_emits_cost_delta(self) -> None:
        source = _card("Tactical Discount", card_type="Skill", cost=1, text="Choose a card in your hand. Set its cost to 0 this turn.")
        bash = _card("Bash", damage=8, cost=2)
        obs, actions = _obs_with_action(source, [source, bash])

        encoded = WorldTokenObservationEncoder(use_text=False).encode(obs, actions)
        summary = _locals(encoded, "HAND_MUTATION_LOCAL")[0]
        targets = _locals(encoded, "HAND_MUTATION_TARGET_LOCAL")

        self.assertEqual(float(summary[5]), 1.0)  # cost_modify_hand
        self.assertEqual(float(summary[6]), 1.0)  # set_cost_zero
        self.assertLess(float(summary[25]), 0.0)  # expected cost delta signed
        self.assertTrue(any(float(t[4]) == 1.0 and float(t[12]) < 0.0 for t in targets))

    def test_exhaust_and_draw_hand_mutations_share_same_abstraction(self) -> None:
        exhaust_source = _card("Feed Flames", card_type="Skill", cost=1, text="Exhaust a card in your hand.")
        obs, actions = _obs_with_action(exhaust_source, [exhaust_source, _card("Strike", damage=6)])
        encoded = WorldTokenObservationEncoder(use_text=False).encode(obs, actions)
        summary = _locals(encoded, "HAND_MUTATION_LOCAL")[0]
        self.assertEqual(float(summary[11]), 1.0)  # exhaust_from_hand
        self.assertTrue(any(float(t[6]) == 1.0 for t in _locals(encoded, "HAND_MUTATION_TARGET_LOCAL")))

        draw_source = _card("Quick Study", card_type="Skill", cost=1, text="Draw 2 cards.")
        obs, actions = _obs_with_action(draw_source, [draw_source])
        encoded = WorldTokenObservationEncoder(use_text=False).encode(obs, actions)
        summary = _locals(encoded, "HAND_MUTATION_LOCAL")[0]
        self.assertEqual(float(summary[7]), 1.0)  # draw_to_hand
        self.assertGreater(float(summary[18]), 0.0)  # affected/post count exposure


if __name__ == "__main__":
    unittest.main()
