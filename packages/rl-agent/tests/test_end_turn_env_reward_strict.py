from __future__ import annotations

from sts2_env.combat_env import CombatSandboxEnv
from sts2_env.env_v2 import SlayTheSpire2EnvV2


def _obs(*, energy: int = 2, hp: int = 50, block: int = 0, incoming: int = 0, enemy_hp: int = 20) -> dict:
    return {
        "player": {"hp": hp, "block": block},
        "combat": {
            "energy": energy,
            "enemies": [
                {
                    "id": "e1",
                    "combat_id": "e1",
                    "hp": enemy_hp,
                    "intent": {"total_damage": incoming},
                }
            ],
        },
    }


def _end_turn() -> dict:
    return {"action_id": "end_turn", "kind": "end_turn"}


def _card(title: str, *, cost: int = 1, damage: int = 0, block: int = 0, self_damage: int = 0) -> dict:
    effect_preview = {}
    if damage:
        effect_preview["damage"] = damage
        effect_preview["total_damage"] = damage
    if block:
        effect_preview["block"] = block
    if self_damage:
        effect_preview["self_damage"] = self_damage
    return {
        "action_id": f"play:{title}",
        "kind": "play_card",
        "target": {"combat_id": "e1"},
        "card": {
            "title": title,
            "name": title,
            "cost": cost,
            "effect_preview": effect_preview,
        },
    }


def _potion(title: str, *, damage: int = 0, block: int = 0, heal: int = 0) -> dict:
    effect_preview = {}
    if damage:
        effect_preview["damage"] = damage
        effect_preview["total_damage"] = damage
    if block:
        effect_preview["block"] = block
    if heal:
        effect_preview["heal"] = heal
    return {
        "action_id": f"potion:{title}",
        "kind": "use_potion",
        "target": {"combat_id": "e1"},
        "potion": {"title": title, "effect_preview": effect_preview},
    }


def _combat_env() -> CombatSandboxEnv:
    env = CombatSandboxEnv.__new__(CombatSandboxEnv)
    env._current_encounter_id = "unit_normal"
    env._last_obs_raw = {}
    env._wasteful_end_turn_count = 0
    return env


def _fullrun_env() -> SlayTheSpire2EnvV2:
    return SlayTheSpire2EnvV2.__new__(SlayTheSpire2EnvV2)


def test_combat_env_does_not_penalize_empty_hand_leftover_energy() -> None:
    env = _combat_env()
    obs = _obs(energy=3, incoming=18)
    assert env._end_turn_waste_penalty(obs, [_end_turn()], _end_turn()) == 0.0
    diag = env._action_quality_diagnostics(obs, [_end_turn()], _end_turn())
    assert diag["benign_leftover_energy"] == 1.0
    assert diag["wasteful_end_turn_selected"] == 0.0


def test_combat_env_does_not_penalize_block_when_enemy_not_attacking() -> None:
    env = _combat_env()
    obs = _obs(energy=2, incoming=0)
    legal = [_card("防御", block=5), _end_turn()]
    assert env._end_turn_waste_penalty(obs, legal, _end_turn()) == 0.0
    diag = env._action_quality_diagnostics(obs, legal, _end_turn())
    assert diag["positive_action_count"] == 1.0  # legacy/broad signal remains visible
    assert diag["urgent_positive_action_count"] == 0.0
    assert diag["wasteful_end_turn_selected"] == 0.0


def test_combat_env_penalizes_skipping_defense_against_real_damage() -> None:
    env = _combat_env()
    obs = _obs(energy=2, incoming=12, block=0)
    legal = [_card("防御", block=5), _end_turn()]
    assert env._end_turn_waste_penalty(obs, legal, _end_turn()) < 0.0
    diag = env._action_quality_diagnostics(obs, legal, _end_turn())
    assert diag["urgent_positive_action_count"] == 1.0
    assert diag["wasteful_end_turn_selected"] == 1.0


def test_combat_env_penalizes_skipping_lethal_attack_even_without_incoming_damage() -> None:
    env = _combat_env()
    obs = _obs(energy=1, incoming=0, enemy_hp=6)
    legal = [_card("打击", damage=6), _end_turn()]
    assert env._end_turn_waste_penalty(obs, legal, _end_turn()) < 0.0


def test_fullrun_env_does_not_penalize_optional_potion_left_after_cards() -> None:
    env = _fullrun_env()
    obs = _obs(energy=2, hp=50, incoming=0)
    legal = [_potion("铁心药水", block=10), _end_turn()]
    assert env._end_turn_waste_penalty(obs, legal, _end_turn()) == 0.0


def test_fullrun_env_penalizes_potion_that_prevents_lethal() -> None:
    env = _fullrun_env()
    obs = _obs(energy=2, hp=8, block=0, incoming=14)
    legal = [_potion("铁心药水", block=10), _end_turn()]
    assert env._end_turn_waste_penalty(obs, legal, _end_turn()) < 0.0
