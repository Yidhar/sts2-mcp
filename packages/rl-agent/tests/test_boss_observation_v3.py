from __future__ import annotations

import sys
import unittest
from pathlib import Path

RL_AGENT_ROOT = Path(__file__).resolve().parents[1]
if str(RL_AGENT_ROOT) not in sys.path:
    sys.path.insert(0, str(RL_AGENT_ROOT))

from sts2_env.observation_v3 import TOKEN_TYPE_TO_ID, WorldTokenObservationEncoder


def _card(title: str, *, card_type: str = "Attack", cost: int = 1, damage: int = 6) -> dict[str, object]:
    return {
        "id": title.upper().replace(" ", "_"),
        "title": title,
        "type": card_type,
        "cost": cost,
        "damage": damage,
        "preview_damage": damage,
        "can_play": True,
        "index": 0,
    }


def _enemy(
    *,
    enemy_id: int,
    model_id: str,
    name: str,
    hp: int,
    max_hp: int,
    incoming_damage_multiplier: float = 1.0,
    powers: list[dict[str, object]] | None = None,
) -> dict[str, object]:
    return {
        "id": enemy_id,
        "combat_id": enemy_id,
        "model_id": model_id,
        "name": name,
        "hp": hp,
        "max_hp": max_hp,
        "block": 0,
        "incoming_damage_multiplier": incoming_damage_multiplier,
        "intent": {"intent_type": "attack", "title": "Attack", "total_damage": 18},
        "powers": powers or [],
        "is_alive": True,
    }


def _combat_obs(enemy: dict[str, object], *, extra_combat: dict[str, object] | None = None, run: dict[str, object] | None = None) -> tuple[dict[str, object], list[dict[str, object]]]:
    strike = _card("Strike")
    combat = {
        "round": 1,
        "energy": 3,
        "max_energy": 3,
        "player_powers": [],
        "hand": [strike],
        "draw_pile": [],
        "discard_pile": [],
        "exhaust_pile": [],
        "enemies": [enemy],
    }
    if extra_combat:
        combat.update(extra_combat)
    obs = {
        "phase": "combat",
        "run": run or {},
        "player": {"hp": 50, "max_hp": 80, "block": 0, "relics": [], "potions": []},
        "combat": combat,
    }
    action = {"kind": "play_card", "card": strike, "target": {"combat_id": enemy["combat_id"], "name": enemy["name"]}}
    return obs, [action]


def _first_local_numeric(encoded: dict[str, object], token_type: str):
    type_id = TOKEN_TYPE_TO_ID[token_type]
    type_ids = encoded["candidate_local_type_ids"][0]
    masks = encoded["candidate_local_masks"][0]
    for index, current in enumerate(type_ids):
        if masks[index] > 0 and int(current) == type_id:
            return encoded["candidate_local_tokens"][0, index, :96]
    raise AssertionError(f"missing local token {token_type}")


class BossObservationV3IntegrationTest(unittest.TestCase):
    def test_back_attack_and_vantom_damage_cap_reach_local_tokens(self) -> None:
        obs, actions = _combat_obs(
            _enemy(
                enemy_id=11,
                model_id="MONSTER.VANTOM",
                name="Vantom",
                hp=300,
                max_hp=300,
                incoming_damage_multiplier=1.5,
                powers=[
                    {"id": "BackAttackLeftPower", "title": "Back Attack Left", "amount": 1},
                    {"id": "SLIPPERY_POWER", "title": "Slippery", "amount": 1},
                ],
            ),
            extra_combat={"facing": "right"},
            run={"room_model": "MONSTER.VANTOM"},
        )
        encoded = WorldTokenObservationEncoder(use_text=False).encode(obs, actions)

        player = _first_local_numeric(encoded, "PLAYER_STATE_LOCAL")
        energy = _first_local_numeric(encoded, "ENERGY_CONTEXT_LOCAL")
        reaction = _first_local_numeric(encoded, "TARGET_REACTION_LOCAL")

        self.assertEqual(float(player[30]), 1.0)  # facing_right
        self.assertGreater(float(player[47]), 0.0)  # back_attack_risk
        self.assertGreater(float(energy[21]), 0.0)  # energy/action local sees back-attack pressure
        self.assertEqual(float(reaction[19]), 1.0)  # back_attack
        self.assertEqual(float(reaction[20]), 1.0)  # damage_cap_active
        self.assertGreater(float(reaction[21]), 0.0)  # damage_cap_value

    def test_waterfall_giant_deathburst_reaches_target_reaction(self) -> None:
        obs, actions = _combat_obs(
            _enemy(
                enemy_id=41,
                model_id="MONSTER.WATERFALL_GIANT",
                name="Waterfall Giant",
                hp=180,
                max_hp=400,
            ),
            run={"room_model": "MONSTER.WATERFALL_GIANT"},
        )
        reaction = _first_local_numeric(WorldTokenObservationEncoder(use_text=False).encode(obs, actions), "TARGET_REACTION_LOCAL")
        self.assertEqual(float(reaction[22]), 1.0)
        self.assertGreater(float(reaction[23]), 0.0)

    def test_insatiable_escape_tax_and_countdown_reach_player_and_target_tokens(self) -> None:
        enemy = _enemy(enemy_id=21, model_id="MONSTER.THE_INSATIABLE", name="The Insatiable", hp=240, max_hp=340)
        obs, actions = _combat_obs(
            enemy,
            extra_combat={
                "round": 4,
                "player_powers": [{"id": "SANDPIT_POWER", "title": "Sandpit", "amount": 3}],
                "hand": [_card("Frantic Escape", card_type="Skill", cost=0, damage=0), _card("Strike")],
                "draw_pile": [_card("Frantic Escape", card_type="Skill", cost=0, damage=0)],
                "discard_pile": [_card("Frantic Escape", card_type="Skill", cost=0, damage=0)],
            },
            run={"room_model": "MONSTER.THE_INSATIABLE"},
        )
        # target the first hand card after overriding hand above
        actions[0]["card"] = obs["combat"]["hand"][1]
        encoded = WorldTokenObservationEncoder(use_text=False).encode(obs, actions)
        player = _first_local_numeric(encoded, "PLAYER_STATE_LOCAL")
        reaction = _first_local_numeric(encoded, "TARGET_REACTION_LOCAL")

        self.assertEqual(float(player[31]), 1.0)  # sandpit_active
        self.assertGreater(float(player[43]), 0.0)  # frantic_escape_total_norm
        self.assertEqual(float(player[50]), 1.0)  # escape_card_tax
        self.assertEqual(float(reaction[31]), 1.0)  # escape_card_tax
        self.assertEqual(float(reaction[32]), 1.0)  # countdown

    def test_queen_binding_and_linked_support_reach_local_tokens(self) -> None:
        queen = _enemy(enemy_id=31, model_id="MONSTER.QUEEN", name="Queen", hp=260, max_hp=260)
        torch = _enemy(enemy_id=32, model_id="MONSTER.TORCH_HEAD_AMALGAM", name="Torch Head Amalgam", hp=90, max_hp=90)
        obs, actions = _combat_obs(
            queen,
            extra_combat={
                "round": 2,
                "player_powers": [
                    {"id": "CHAINS_OF_BINDING", "title": "Chains of Binding", "amount": 1},
                    {"id": "BOUND", "title": "Bound", "amount": 1},
                ],
                "enemies": [queen, torch],
            },
            run={"room_model": "MONSTER.QUEEN"},
        )
        encoded = WorldTokenObservationEncoder(use_text=False).encode(obs, actions)
        player = _first_local_numeric(encoded, "PLAYER_STATE_LOCAL")
        reaction = _first_local_numeric(encoded, "TARGET_REACTION_LOCAL")

        self.assertEqual(float(player[35]), 1.0)  # chains_active
        self.assertEqual(float(player[36]), 1.0)  # bound_active
        self.assertEqual(float(player[48]), 1.0)  # linked support present
        self.assertEqual(float(reaction[26]), 1.0)  # linked_support_alive
        self.assertEqual(float(reaction[34]), 1.0)  # binding_control


if __name__ == "__main__":
    unittest.main()
