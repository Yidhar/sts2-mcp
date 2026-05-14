from __future__ import annotations

import importlib.util
from pathlib import Path
import sys
import types

import numpy as np


RL_AGENT_ROOT = Path(__file__).resolve().parents[1]
STS2_ENV_ROOT = RL_AGENT_ROOT / "sts2_env"


def _load_sts2_env_modules():
    if str(RL_AGENT_ROOT) not in sys.path:
        sys.path.insert(0, str(RL_AGENT_ROOT))

    pkg_name = "sts2_env"
    if pkg_name not in sys.modules:
        pkg = types.ModuleType(pkg_name)
        pkg.__path__ = [str(STS2_ENV_ROOT)]
        sys.modules[pkg_name] = pkg

    loaded = {}
    for name in ("text_encoder", "semantic_action", "run_memory", "observation_common", "observation_v3"):
        qualified = f"{pkg_name}.{name}"
        if qualified not in sys.modules:
            path = STS2_ENV_ROOT / f"{name}.py"
            spec = importlib.util.spec_from_file_location(qualified, path)
            if spec is None or spec.loader is None:
                raise RuntimeError(f"Failed to load module spec for {qualified}")
            module = importlib.util.module_from_spec(spec)
            module.__package__ = pkg_name
            sys.modules[qualified] = module
            spec.loader.exec_module(module)
        loaded[name] = sys.modules[qualified]
    return loaded


def _body_slam_card() -> dict:
    return {
        "id": "CARD.BODY_SLAM",
        "model_id": "CARD.BODY_SLAM",
        "class_name": "BodySlam",
        "title": "全身撞击",
        "type": "Attack",
        "cost": 1,
        "target": "AnyEnemy",
        "target_type": "AnyEnemy",
        # Intentionally no damage/effect_preview fields: this reproduces the
        # bridge/static underexposure where Body Slam looked like 0 damage.
        "description": "造成你当前格挡值的伤害。",
    }


def _obs_and_actions(player_block: int, enemy_hp: int = 12) -> tuple[dict, list[dict]]:
    body_slam = _body_slam_card()
    obs = {
        "phase": "combat",
        "player": {
            "hp": 50,
            "max_hp": 80,
            "block": player_block,
            "gold": 0,
            "deck_cards": [body_slam],
            "relics": [],
            "potions": [],
        },
        "combat": {
            "in_combat": True,
            "energy": 3,
            "max_energy": 3,
            "hand": [body_slam],
            "draw_pile": {"cards": []},
            "discard_pile": {"cards": []},
            "exhaust_pile": {"cards": []},
            "play_pile": {"cards": []},
            "enemies": [
                {
                    "name": "Dummy",
                    "model_id": "MONSTER.DUMMY",
                    "combat_id": 1,
                    "current_hp": enemy_hp,
                    "max_hp": 30,
                    "block": 0,
                    "powers": [],
                    "intent": {"intent_type": "buff", "total_damage": 0, "damage_per_hit": 0, "repeats": 0},
                }
            ],
        },
    }
    legal_actions = [
        {
            "action_id": "play:body_slam:1",
            "kind": "play_card",
            "card": body_slam,
            "target": {"name": "Dummy", "combat_id": 1, "side": "Enemy"},
        }
    ]
    return obs, legal_actions


def test_body_slam_missing_preview_uses_current_block_for_action_and_lethal_context() -> None:
    modules = _load_sts2_env_modules()
    obs_common = modules["observation_common"]
    obs_v3 = modules["observation_v3"]

    encoder = obs_v3.WorldTokenObservationEncoder(use_text=False)
    obs, legal_actions = _obs_and_actions(player_block=17, enemy_hp=12)
    encoded = encoder.encode(obs, legal_actions=legal_actions)

    expected_damage_200 = obs_common._log_norm(17, obs_common._LOG1P_200)
    expected_damage_100 = obs_common._log_norm(17, obs_common._LOG1P_100)

    # Query token is copied from ACTION_FEAT_DIM; slot 22 is preview_damage.
    np.testing.assert_allclose(encoded["candidate_query_tokens"][0, 22], expected_damage_200, rtol=1e-5)
    np.testing.assert_allclose(encoded["candidate_query_tokens"][0, 40], expected_damage_200, rtol=1e-5)
    np.testing.assert_allclose(encoded["candidate_query_tokens"][0, 42], expected_damage_100, rtol=1e-5)
    assert encoded["candidate_query_tokens"][0, 38] == 1.0

    local_types = encoded["candidate_local_type_ids"][0]
    local_tokens = encoded["candidate_local_tokens"][0]
    source_idx = int(np.where(local_types == obs_v3.TOKEN_TYPE_TO_ID["SOURCE_CARD_LOCAL"])[0][0])
    np.testing.assert_allclose(local_tokens[source_idx, 22], expected_damage_200, rtol=1e-5)

    reaction_idx = int(np.where(local_types == obs_v3.TOKEN_TYPE_TO_ID["TARGET_REACTION_LOCAL"])[0][0])
    assert local_tokens[reaction_idx, 9] == 1.0  # expected_damage lethal vs 12 HP

    hand_idx = int(np.where(encoded["world_token_type_ids"] == obs_v3.TOKEN_TYPE_TO_ID["HAND_CARD"])[0][0])
    np.testing.assert_allclose(encoded["world_tokens"][hand_idx, 14], expected_damage_200, rtol=1e-5)


def test_body_slam_zero_block_does_not_invent_damage() -> None:
    modules = _load_sts2_env_modules()
    obs_v3 = modules["observation_v3"]

    encoder = obs_v3.WorldTokenObservationEncoder(use_text=False)
    obs, legal_actions = _obs_and_actions(player_block=0, enemy_hp=12)
    encoded = encoder.encode(obs, legal_actions=legal_actions)

    assert encoded["candidate_query_tokens"][0, 22] == 0.0
    assert encoded["candidate_query_tokens"][0, 40] == 0.0
    local_types = encoded["candidate_local_type_ids"][0]
    local_tokens = encoded["candidate_local_tokens"][0]
    reaction_idx = int(np.where(local_types == obs_v3.TOKEN_TYPE_TO_ID["TARGET_REACTION_LOCAL"])[0][0])
    assert local_tokens[reaction_idx, 9] == 0.0
