from __future__ import annotations

import importlib.util
import sys
import types
import unittest
from pathlib import Path


RL_AGENT_ROOT = Path(__file__).resolve().parents[1]
if str(RL_AGENT_ROOT) not in sys.path:
    sys.path.insert(0, str(RL_AGENT_ROOT))


def _stub(module_name: str, **attrs) -> types.ModuleType:
    module = types.ModuleType(module_name)
    for key, value in attrs.items():
        setattr(module, key, value)
    sys.modules[module_name] = module
    return module


def _load(module_name: str, file_path: Path) -> types.ModuleType:
    spec = importlib.util.spec_from_file_location(module_name, file_path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


_sts2_env_pkg = sys.modules.get("sts2_env")
if _sts2_env_pkg is None:
    _sts2_env_pkg = types.ModuleType("sts2_env")
    _sts2_env_pkg.__path__ = [str(RL_AGENT_ROOT / "sts2_env")]
    sys.modules["sts2_env"] = _sts2_env_pkg


_load("content_registry", RL_AGENT_ROOT / "content_registry.py")
_stub("sts2_env.text_encoder", TEXT_DIM=512)
_load("sts2_env.semantic_action", RL_AGENT_ROOT / "sts2_env" / "semantic_action.py")
_load("sts2_env.run_memory", RL_AGENT_ROOT / "sts2_env" / "run_memory.py")
_load("sts2_env.observation_common", RL_AGENT_ROOT / "sts2_env" / "observation_common.py")
boss_mechanics = _load("sts2_env.boss_mechanics", RL_AGENT_ROOT / "sts2_env" / "boss_mechanics.py")


def _card(title: str) -> dict[str, object]:
    return {"title": title, "id": title.upper().replace(" ", "_")}


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
        "model_id": model_id,
        "name": name,
        "hp": hp,
        "max_hp": max_hp,
        "block": 0,
        "incoming_damage_multiplier": incoming_damage_multiplier,
        "intent": {"intent_type": "attack", "title": "Attack", "description": "Deal damage", "total_damage": 18},
        "powers": list(powers or []),
        "is_alive": hp > 0,
    }


def _sandpit_power(amount: int | float) -> dict[str, object]:
    return {
        "id": "POWER.SANDPIT_POWER",
        "model_id": "POWER.SANDPIT_POWER",
        "class_name": "SandpitPower",
        "kind": "SandpitPower",
        "title": "Sandpit",
        "amount": amount,
        "display_amount": amount,
        "stack_type": "Counter",
    }


class BossMechanicsTest(unittest.TestCase):
    def test_kaiser_style_back_attack_and_vantom_damage_cap_detected(self) -> None:
        obs = {
            "run": {"room_model": "MONSTER.VANTOM"},
            "combat": {
                "round": 1,
                "facing": "right",
                "player_powers": [],
                "hand": [],
                "draw_pile": [],
                "discard_pile": [],
                "exhaust_pile": [],
                "enemies": [
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
                    )
                ],
            },
        }

        context = boss_mechanics.build_boss_mechanics_context(obs)
        player_state = context["player_state"]
        enemy_state = context["enemy_states_by_index"][0]

        self.assertEqual(player_state["facing_right"], 1.0)
        self.assertGreater(player_state["back_attack_risk"], 0.0)
        self.assertEqual(enemy_state["back_attack_active"], 1.0)
        self.assertEqual(enemy_state["damage_cap_active"], 1.0)

    def test_insatiable_countdown_and_frantic_escape_piles_detected(self) -> None:
        obs = {
            "encounter_id": "MONSTER.THE_INSATIABLE",
            "combat": {
                "round": 4,
                "facing": None,
                "player_powers": [
                    {"id": "SANDPIT_POWER", "title": "Sandpit", "amount": 3},
                ],
                "hand": [_card("Frantic Escape"), _card("Strike")],
                "draw_pile": [_card("Frantic Escape"), _card("Defend")],
                "discard_pile": [_card("Frantic Escape")],
                "exhaust_pile": [],
                "enemies": [
                    _enemy(
                        enemy_id=21,
                        model_id="MONSTER.THE_INSATIABLE",
                        name="The Insatiable",
                        hp=240,
                        max_hp=340,
                    )
                ],
            },
        }

        context = boss_mechanics.build_boss_mechanics_context(obs)
        player_state = context["player_state"]
        enemy_state = context["enemy_states_by_index"][0]

        self.assertEqual(player_state["sandpit_active"], 1.0)
        self.assertEqual(player_state["sandpit_turns"], 3.0)
        self.assertGreater(player_state["frantic_escape_hand_norm"], 0.0)
        self.assertGreater(player_state["frantic_escape_total_norm"], 0.0)
        self.assertEqual(player_state["frantic_escape_hand_count"], 1.0)
        self.assertEqual(player_state["frantic_escape_draw_count"], 1.0)
        self.assertEqual(player_state["frantic_escape_discard_count"], 1.0)
        self.assertEqual(player_state["frantic_escape_exhaust_count"], 0.0)
        self.assertEqual(player_state["frantic_escape_total_count"], 3.0)
        self.assertEqual(enemy_state["countdown_active"], 1.0)
        self.assertEqual(enemy_state["escape_card_tax"], 1.0)

    def test_insatiable_real_sandpit_power_id_detected(self) -> None:
        obs = {
            "encounter_id": "MONSTER.THE_INSATIABLE",
            "combat": {
                "round": 2,
                "player_powers": [
                    {
                        "id": "POWER.SANDPIT_POWER",
                        "model_id": "POWER.SANDPIT_POWER",
                        "class_name": "SandpitPower",
                        "kind": "SandpitPower",
                        "title": "Sandpit",
                        "amount": 2,
                        "display_amount": 2,
                    },
                ],
                "hand": [],
                "draw_pile": [],
                "discard_pile": [],
                "exhaust_pile": [],
                "enemies": [
                    _enemy(
                        enemy_id=22,
                        model_id="MONSTER.THE_INSATIABLE",
                        name="The Insatiable",
                        hp=240,
                        max_hp=340,
                    )
                ],
            },
        }

        context = boss_mechanics.build_boss_mechanics_context(obs)
        player_state = context["player_state"]

        self.assertEqual(player_state["sandpit_active"], 1.0)
        self.assertEqual(player_state["sandpit_turns"], 2.0)

    def test_insatiable_real_sandpit_power_detected_from_enemy_owner(self) -> None:
        """The live game owns SandpitPower on The Insatiable, not the player.

        The power targets the player internally, but the bridge's enemy power
        payload is where the countdown is visible.  This test guards against
        regressing to player-only scans, which pins all sandpit metrics at 0.
        """
        obs = {
            "encounter_id": "MONSTER.THE_INSATIABLE",
            "combat": {
                "round": 3,
                "player_powers": [],
                "hand": [
                    {
                        "id": "CARD.FRANTIC_ESCAPE",
                        "model_id": "CARD.FRANTIC_ESCAPE",
                        "normalized_id": "frantic_escape",
                        "class_name": "FranticEscape",
                        "title": "狂乱逃离",
                    }
                ],
                "draw_pile": [],
                "discard_pile": [],
                "exhaust_pile": [],
                "enemies": [
                    _enemy(
                        enemy_id=25,
                        model_id="MONSTER.THE_INSATIABLE",
                        name="The Insatiable",
                        hp=240,
                        max_hp=340,
                        powers=[_sandpit_power(2)],
                    )
                ],
            },
        }

        context = boss_mechanics.build_boss_mechanics_context(obs)
        player_state = context["player_state"]
        enemy_state = context["enemy_states_by_index"][0]

        self.assertEqual(player_state["sandpit_active"], 1.0)
        self.assertEqual(player_state["sandpit_turns"], 2.0)
        self.assertEqual(player_state["frantic_escape_hand_count"], 1.0)
        self.assertEqual(player_state["escape_card_available"], 1.0)
        self.assertEqual(enemy_state["countdown_active"], 1.0)
        self.assertEqual(enemy_state["escape_card_tax"], 1.0)

    def test_insatiable_sandpit_detected_from_live_bridge_nested_player_payload(self) -> None:
        obs = {
            "encounter_id": "MONSTER.THE_INSATIABLE",
            "combat": {
                "round": 2,
                # Empty top-level payloads must not mask the real live bridge
                # creature/player payload.  This mirrors BuildPilePayload shape.
                "player_powers": [],
                "hand": [],
                "draw_pile": [],
                "discard_pile": [],
                "exhaust_pile": [],
                "player_creatures": [
                    {
                        "powers": [
                            {
                                "id": "POWER.SANDPIT_POWER",
                                "model_id": "POWER.SANDPIT_POWER",
                                "class_name": "SandpitPower",
                                "kind": "SandpitPower",
                                "title": "Sandpit",
                                "amount": 1,
                                "display_amount": 1,
                            }
                        ]
                    }
                ],
                "enemies": [
                    _enemy(
                        enemy_id=23,
                        model_id="MONSTER.THE_INSATIABLE",
                        name="The Insatiable",
                        hp=240,
                        max_hp=340,
                    )
                ],
            },
            "players": [
                {
                    "creature": {
                        "powers": [
                            {
                                "id": "POWER.SANDPIT_POWER",
                                "model_id": "POWER.SANDPIT_POWER",
                                "class_name": "SandpitPower",
                                "kind": "SandpitPower",
                                "title": "Sandpit",
                                "amount": 1,
                                "display_amount": 1,
                            }
                        ]
                    },
                    "combat": {
                        "hand": {
                            "pile_type": "Hand",
                            "count": 1,
                            "cards": [
                                {
                                    "id": "CARD.FRANTIC_ESCAPE",
                                    "model_id": "CARD.FRANTIC_ESCAPE",
                                    "normalized_id": "frantic_escape",
                                    "class_name": "FranticEscape",
                                    "title": "狂乱逃离",
                                }
                            ],
                        },
                        "draw_pile": {"pile_type": "Draw", "count": 0, "cards": []},
                        "discard_pile": {
                            "pile_type": "Discard",
                            "count": 1,
                            "cards": [
                                {
                                    "id": "CARD.FRANTIC_ESCAPE",
                                    "model_id": "CARD.FRANTIC_ESCAPE",
                                    "normalized_id": "frantic_escape",
                                    "class_name": "FranticEscape",
                                }
                            ],
                        },
                        "exhaust_pile": {"pile_type": "Exhaust", "count": 0, "cards": []},
                    },
                }
            ],
        }

        context = boss_mechanics.build_boss_mechanics_context(obs)
        player_state = context["player_state"]

        self.assertEqual(player_state["sandpit_active"], 1.0)
        self.assertEqual(player_state["sandpit_turns"], 1.0)
        self.assertEqual(player_state["frantic_escape_hand_count"], 1.0)
        self.assertEqual(player_state["frantic_escape_draw_count"], 0.0)
        self.assertEqual(player_state["frantic_escape_discard_count"], 1.0)
        self.assertEqual(player_state["frantic_escape_exhaust_count"], 0.0)
        self.assertEqual(player_state["frantic_escape_total_count"], 2.0)

    def test_insatiable_frantic_escape_detected_from_class_metadata_only(self) -> None:
        obs = {
            "encounter_id": "MONSTER.THE_INSATIABLE",
            "combat": {
                "round": 2,
                "player_powers": [
                    {
                        "id": "POWER.SANDPIT_POWER",
                        "model_id": "POWER.SANDPIT_POWER",
                        "class_name": "SandpitPower",
                        "kind": "SandpitPower",
                        "title": "Sandpit",
                        "amount": 2,
                        "display_amount": 2,
                    }
                ],
                "hand": [{"class_name": "FranticEscape"}],
                "draw_pile": {"pile_type": "Draw", "count": 1, "cards": [{"kind": "FranticEscape"}]},
                "discard_pile": [],
                "exhaust_pile": [],
                "enemies": [
                    _enemy(
                        enemy_id=24,
                        model_id="MONSTER.THE_INSATIABLE",
                        name="The Insatiable",
                        hp=240,
                        max_hp=340,
                    )
                ],
            },
        }

        context = boss_mechanics.build_boss_mechanics_context(obs)
        player_state = context["player_state"]

        self.assertEqual(player_state["sandpit_active"], 1.0)
        self.assertEqual(player_state["sandpit_turns"], 2.0)
        self.assertEqual(player_state["frantic_escape_hand_count"], 1.0)
        self.assertEqual(player_state["frantic_escape_draw_count"], 1.0)
        self.assertEqual(player_state["frantic_escape_total_count"], 2.0)

    def test_queen_binding_and_linked_support_detected(self) -> None:
        obs = {
            "run": {"room_model": "MONSTER.QUEEN"},
            "combat": {
                "round": 2,
                "player_powers": [
                    {"id": "CHAINS_OF_BINDING", "title": "Chains of Binding", "amount": 1},
                    {"id": "BOUND", "title": "Bound", "amount": 1},
                ],
                "hand": [],
                "draw_pile": [],
                "discard_pile": [],
                "exhaust_pile": [],
                "enemies": [
                    _enemy(
                        enemy_id=31,
                        model_id="MONSTER.QUEEN",
                        name="Queen",
                        hp=260,
                        max_hp=260,
                    ),
                    _enemy(
                        enemy_id=32,
                        model_id="MONSTER.TORCH_HEAD_AMALGAM",
                        name="Torch Head Amalgam",
                        hp=90,
                        max_hp=90,
                    ),
                ],
            },
        }

        context = boss_mechanics.build_boss_mechanics_context(obs)
        queen_state = context["enemy_states_by_index"][0]

        self.assertEqual(queen_state["binding_control"], 1.0)
        self.assertEqual(queen_state["linked_support_alive"], 1.0)

    def test_waterfall_giant_deathburst_detected(self) -> None:
        obs = {
            "run": {"room_model": "MONSTER.WATERFALL_GIANT"},
            "combat": {
                "round": 5,
                "player_powers": [],
                "hand": [],
                "draw_pile": [],
                "discard_pile": [],
                "exhaust_pile": [],
                "enemies": [
                    _enemy(
                        enemy_id=41,
                        model_id="MONSTER.WATERFALL_GIANT",
                        name="Waterfall Giant",
                        hp=180,
                        max_hp=400,
                    )
                ],
            },
        }

        context = boss_mechanics.build_boss_mechanics_context(obs)
        enemy_state = context["enemy_states_by_index"][0]
        self.assertEqual(enemy_state["deathburst"], 1.0)
        self.assertGreater(enemy_state["deathburst_damage"], 0.0)


if __name__ == "__main__":
    unittest.main()
