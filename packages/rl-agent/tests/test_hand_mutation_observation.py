from __future__ import annotations

import sys
import unittest
from pathlib import Path

RL_AGENT_ROOT = Path(__file__).resolve().parents[1]
if str(RL_AGENT_ROOT) not in sys.path:
    sys.path.insert(0, str(RL_AGENT_ROOT))

from sts2_env.observation_v3 import TOKEN_TYPE_TO_ID, WorldTokenObservationEncoder
from sts2_env.hand_mutation import infer_hand_mutation


def _card(
    title: str,
    *,
    card_type: str = "Attack",
    cost: int = 1,
    x_cost: bool = False,
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
        "x_cost": x_cost,
        "costs_x": x_cost,
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
    def test_typed_upgrade_one_from_hand_does_not_need_text(self) -> None:
        source = _card("Practice Smith", card_type="Skill", cost=1, text="")
        source["card_effect_profile"] = {
            "operations": [
                {
                    "op": "upgrade_card",
                    "source_zone": "hand",
                    "selection": "choice",
                    "scope": "one",
                    "target_filter": ["is_upgradable"],
                    "count": 1,
                }
            ]
        }
        strike = _card("Strike", damage=6, text="")
        defend = _card("Defend", card_type="Skill", block=5, text="")
        obs, actions = _obs_with_action(source, [source, strike, defend])

        encoded = WorldTokenObservationEncoder(use_text=False).encode(obs, actions)
        summary = _locals(encoded, "HAND_MUTATION_LOCAL")[0]
        targets = _locals(encoded, "HAND_MUTATION_TARGET_LOCAL")

        self.assertEqual(float(summary[0]), 1.0)  # will_mutate_hand
        self.assertEqual(float(summary[1]), 1.0)  # upgrade_hand
        self.assertEqual(float(summary[2]), 1.0)  # upgrade_one
        self.assertEqual(float(summary[3]), 0.0)  # not upgrade_all
        self.assertTrue(any(float(t[3]) == 1.0 for t in targets))

    def test_typed_upgraded_override_promotes_upgrade_to_all_hand(self) -> None:
        source = _card("Armaments Like", card_type="Skill", cost=1, text="", upgrade_level=1)
        source["card_effect_profile"] = {
            "operations": [
                {
                    "op": "upgrade_card",
                    "source_zone": "hand",
                    "selection": "choice",
                    "scope": "one",
                    "target_filter": ["is_upgradable"],
                    "count": 1,
                    "upgraded_override": {"selection": "all", "scope": "all"},
                }
            ]
        }
        strike = _card("Strike", damage=6, text="")
        defend = _card("Defend", card_type="Skill", block=5, text="")
        obs, actions = _obs_with_action(source, [source, strike, defend])

        encoded = WorldTokenObservationEncoder(use_text=False).encode(obs, actions)
        summary = _locals(encoded, "HAND_MUTATION_LOCAL")[0]
        targets = _locals(encoded, "HAND_MUTATION_TARGET_LOCAL")

        self.assertEqual(float(summary[1]), 1.0)
        self.assertEqual(float(summary[3]), 1.0)  # upgrade_all
        self.assertEqual(float(summary[28]), 1.0)  # source_upgraded_scope_bonus
        self.assertGreaterEqual(sum(1 for t in targets if float(t[3]) == 1.0), 2)

    def test_typed_exhaust_modify_cost_copy_and_retain_do_not_need_text(self) -> None:
        strike = _card("Strike", damage=6, text="")
        defend = _card("Defend", card_type="Skill", block=5, text="")

        exhaust_source = _card("Purity Like", card_type="Skill", cost=0, text="")
        exhaust_source["card_effect_profile"] = {
            "operations": [
                {
                    "op": "exhaust_card",
                    "source_zone": "hand",
                    "selection": "choice",
                    "max_count": 3,
                    "target_filter": ["not_status", "not_curse"],
                }
            ]
        }
        obs, actions = _obs_with_action(exhaust_source, [exhaust_source, strike, defend])
        encoded = WorldTokenObservationEncoder(use_text=False).encode(obs, actions)
        summary = _locals(encoded, "HAND_MUTATION_LOCAL")[0]
        self.assertEqual(float(summary[11]), 1.0)  # exhaust_from_hand
        self.assertTrue(any(float(t[6]) == 1.0 for t in _locals(encoded, "HAND_MUTATION_TARGET_LOCAL")))

        cost_source = _card("Enlightenment Like", card_type="Skill", cost=0, text="")
        cost_source["card_effect_profile"] = {
            "operations": [
                {
                    "op": "modify_cost",
                    "source_zone": "hand",
                    "scope": "all",
                    "set_cost": 0,
                    "duration": "this_turn_or_until_played",
                }
            ]
        }
        bash = _card("Bash", damage=8, cost=2, text="")
        obs, actions = _obs_with_action(cost_source, [cost_source, bash])
        encoded = WorldTokenObservationEncoder(use_text=False).encode(obs, actions)
        summary = _locals(encoded, "HAND_MUTATION_LOCAL")[0]
        self.assertEqual(float(summary[5]), 1.0)  # cost_modify_hand
        self.assertEqual(float(summary[6]), 1.0)  # set_cost_zero
        self.assertEqual(float(summary[17]), 1.0)  # temporary
        self.assertLess(float(summary[25]), 0.0)

        copy_source = _card("Dual Wield Like", card_type="Skill", cost=1, text="")
        copy_source["card_effect_profile"] = {
            "operations": [
                {
                    "op": "copy_card",
                    "source_zone": "hand",
                    "destination_zone": "hand",
                    "selection": "choice",
                    "target_filter": ["type_attack", "type_power"],
                    "copy_count": 2,
                }
            ]
        }
        obs, actions = _obs_with_action(copy_source, [copy_source, strike, defend])
        encoded = WorldTokenObservationEncoder(use_text=False).encode(obs, actions)
        summary = _locals(encoded, "HAND_MUTATION_LOCAL")[0]
        self.assertEqual(float(summary[8]), 1.0)  # add_to_hand
        self.assertEqual(float(summary[13]), 1.0)  # copy_hand
        self.assertTrue(any(float(t[8]) == 1.0 for t in _locals(encoded, "HAND_MUTATION_TARGET_LOCAL")))

        retain_source = _card("Retain Like", card_type="Skill", cost=0, text="")
        retain_source["card_effect_profile"] = {
            "operations": [
                {
                    "op": "add_keyword",
                    "source_zone": "hand",
                    "scope": "all",
                    "keyword": "retain",
                    "duration": "this_combat",
                }
            ]
        }
        obs, actions = _obs_with_action(retain_source, [retain_source, strike])
        encoded = WorldTokenObservationEncoder(use_text=False).encode(obs, actions)
        summary = _locals(encoded, "HAND_MUTATION_LOCAL")[0]
        self.assertEqual(float(summary[14]), 1.0)  # modifier_hand
        self.assertTrue(any(float(t[9]) == 1.0 for t in _locals(encoded, "HAND_MUTATION_TARGET_LOCAL")))

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

    def test_typed_not_x_cost_filter_excludes_x_cost_cards(self) -> None:
        source = _card("Bullet Time Like", card_type="Skill", cost=3, text="")
        source["card_effect_profile"] = {
            "operations": [
                {
                    "op": "modify_cost",
                    "source_zone": "hand",
                    "scope": "all",
                    "set_cost": 0,
                    "duration": "this_turn",
                    "target_filter": ["not_x_cost"],
                }
            ]
        }
        bash = _card("Bash", damage=8, cost=2, text="")
        whirlwind = _card("Whirlwind", cost=0, x_cost=True, damage=5, text="")

        plan = infer_hand_mutation(source, [source, bash, whirlwind], current_energy=3)
        affected_titles = {
            str(target.card.get("title"))
            for target in plan.targets
            if target.affected and target.would_cost_modify
        }

        self.assertIn("Bash", affected_titles)
        self.assertNotIn("Whirlwind", affected_titles)

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
