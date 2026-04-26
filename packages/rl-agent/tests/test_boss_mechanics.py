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
        self.assertGreater(player_state["frantic_escape_hand_norm"], 0.0)
        self.assertGreater(player_state["frantic_escape_total_norm"], 0.0)
        self.assertEqual(enemy_state["countdown_active"], 1.0)
        self.assertEqual(enemy_state["escape_card_tax"], 1.0)

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
