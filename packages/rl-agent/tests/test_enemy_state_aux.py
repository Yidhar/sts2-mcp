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


# aux_targets.py pulls the full observation_common + run_memory + objective_heads
# import chain, and the objective pathway references torch-dependent modules
# indirectly through the package __init__. Load aux_targets in isolation.
_sts2_env_pkg = sys.modules.get("sts2_env")
if _sts2_env_pkg is None:
    _sts2_env_pkg = types.ModuleType("sts2_env")
    _sts2_env_pkg.__path__ = [str(RL_AGENT_ROOT / "sts2_env")]
    sys.modules["sts2_env"] = _sts2_env_pkg


def _load(module_name: str, file_path: Path) -> types.ModuleType:
    spec = importlib.util.spec_from_file_location(module_name, file_path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


# Load the real content_registry — it is a flat top-level module with no torch deps.
_load("content_registry", RL_AGENT_ROOT / "content_registry.py")


# Stub torch just enough for objective_heads' isinstance(x, torch.Tensor) check.
class _TorchTensorStub:  # pragma: no cover
    pass


class _TorchDtypeStub:  # pragma: no cover
    pass


_stub(
    "torch",
    Tensor=_TorchTensorStub,
    dtype=_TorchDtypeStub,
    float32=_TorchDtypeStub(),
    long=_TorchDtypeStub(),
    bool=_TorchDtypeStub(),
)

# Stub text_encoder
_stub("sts2_env.text_encoder", TEXT_DIM=512)

# Load semantic_action for real (pure python)
_load("sts2_env.semantic_action", RL_AGENT_ROOT / "sts2_env" / "semantic_action.py")
# Load run_memory for real
_load("sts2_env.run_memory", RL_AGENT_ROOT / "sts2_env" / "run_memory.py")
# Load observation_common for real
_load("sts2_env.observation_common", RL_AGENT_ROOT / "sts2_env" / "observation_common.py")
# Load objective_heads for real
_load("sts2_env.objective_heads", RL_AGENT_ROOT / "sts2_env" / "objective_heads.py")
# Now aux_targets
aux_targets = _load("sts2_env.aux_targets", RL_AGENT_ROOT / "sts2_env" / "aux_targets.py")


def _make_obs(player_hp, enemies, *, self_inflicted_cum=0.0):
    return {
        "combat": {
            "encounter_id": "ENC.TEST",
            "enemies": list(enemies),
            "self_inflicted_hp_loss_cumulative": self_inflicted_cum,
        },
        "player": {"hp": player_hp, "max_hp": 80, "block": 0, "energy": 3},
    }


def _enemy(eid, hp, max_hp=100, damage=10):
    return {
        "id": eid,
        "hp": hp,
        "max_hp": max_hp,
        "block": 0,
        "intent": {"intent_type": "attack", "total_damage": damage,
                   "damage_per_hit": damage, "repeats": 1},
        "powers": [],
    }


class EnemyStateAuxTargetTest(unittest.TestCase):
    def test_shape_and_names(self) -> None:
        self.assertEqual(aux_targets.NUM_ENEMY_STATE_FIELDS, 3)
        self.assertEqual(aux_targets.ENEMY_STATE_SLOT_COUNT, 5)
        self.assertEqual(
            aux_targets.ENEMY_STATE_FIELD_NAMES,
            ("next_hp_delta_ratio",
             "attributable_player_hp_loss_ratio",
             "alive_next"),
        )

    def test_hp_delta_ratio_and_alive_flag(self) -> None:
        prev = _make_obs(80, [_enemy(1, 50, max_hp=50, damage=10)])
        nxt = _make_obs(70, [_enemy(1, 30, max_hp=50, damage=10)])
        targets, mask = aux_targets.compute_enemy_state_targets(prev, nxt)
        self.assertEqual(targets.shape, (5, 3))
        self.assertEqual(mask.shape, (5,))
        # Slot 0: enemy 1 lost 20/50 HP → -0.4
        self.assertAlmostEqual(float(targets[0, 0]), -0.4, places=5)
        # Attribution: predicted 10, actual 10 → 10/80
        self.assertAlmostEqual(float(targets[0, 1]), 10.0 / 80.0, places=5)
        # Alive next
        self.assertEqual(float(targets[0, 2]), 1.0)
        self.assertEqual(float(mask[0]), 1.0)
        # Other slots zeroed
        self.assertEqual(float(mask[1]), 0.0)

    def test_on_death_burst_marks_dead(self) -> None:
        prev = _make_obs(80, [_enemy(9, 20, max_hp=100, damage=5)])
        nxt = _make_obs(40, [])  # enemy vanished, player lost 40 from 5 predicted
        targets, mask = aux_targets.compute_enemy_state_targets(prev, nxt)
        # Slot 0: hp went from 20 to 0 → -0.2 (20/100)
        self.assertAlmostEqual(float(targets[0, 0]), -0.2, places=5)
        # alive_next = 0
        self.assertEqual(float(targets[0, 2]), 0.0)
        self.assertEqual(float(mask[0]), 1.0)

    def test_multiple_enemies_attribution_shared(self) -> None:
        prev = _make_obs(
            80,
            [_enemy(1, 50, damage=10), _enemy(2, 50, damage=30)],
        )
        nxt = _make_obs(60, [_enemy(1, 50, damage=10), _enemy(2, 50, damage=30)])
        # Player lost 20. Predicted total 40. Attribution: enemy1 gets 10/40*20=5,
        # enemy2 gets 30/40*20=15.
        targets, _mask = aux_targets.compute_enemy_state_targets(prev, nxt)
        self.assertAlmostEqual(float(targets[0, 1]), 5.0 / 80.0, places=5)
        self.assertAlmostEqual(float(targets[1, 1]), 15.0 / 80.0, places=5)

    def test_self_damage_stripped_before_attribution(self) -> None:
        """Offering self-damage must not inflate enemy attribution target."""
        # prev: 80 HP, enemy intending 10
        prev = _make_obs(80, [_enemy(1, 50, damage=10)], self_inflicted_cum=0.0)
        # next: lost 13 HP (3 from Offering + 10 enemy). Self cum advanced by 3.
        nxt = _make_obs(67, [_enemy(1, 50, damage=10)], self_inflicted_cum=3.0)
        targets, mask = aux_targets.compute_enemy_state_targets(prev, nxt)
        # Enemy attribution should be 10 / 80 (stripped), not 13 / 80.
        self.assertAlmostEqual(float(targets[0, 1]), 10.0 / 80.0, places=5)
        self.assertEqual(float(mask[0]), 1.0)

    def test_self_damage_only_zero_attribution(self) -> None:
        prev = _make_obs(80, [_enemy(2, 50, damage=10)], self_inflicted_cum=0.0)
        # Enemy fully blocked (0 damage got through), only Offering (3) hit player.
        nxt = _make_obs(77, [_enemy(2, 50, damage=10)], self_inflicted_cum=3.0)
        targets, _mask = aux_targets.compute_enemy_state_targets(prev, nxt)
        self.assertEqual(float(targets[0, 1]), 0.0)

    def test_mask_zero_for_empty_prev(self) -> None:
        prev = _make_obs(80, [])
        nxt = _make_obs(80, [])
        targets, mask = aux_targets.compute_enemy_state_targets(prev, nxt)
        self.assertTrue((targets == 0.0).all())
        self.assertTrue((mask == 0.0).all())

    def test_build_aux_targets_includes_enemy_state(self) -> None:
        prev = _make_obs(80, [_enemy(1, 50, damage=10)])
        nxt = _make_obs(70, [_enemy(1, 30, damage=10)])
        action = {"kind": "play_card", "action_id": "play_card:0",
                  "card": {"title": "Strike", "cost": 1}}
        result = aux_targets.build_aux_targets(prev, action, nxt)
        self.assertIn("enemy_state", result)
        self.assertIn("enemy_state_mask", result)
        self.assertEqual(result["enemy_state"].shape, (5, 3))
        self.assertEqual(result["enemy_state_mask"].shape, (5,))
        self.assertEqual(result["version"], 3)


if __name__ == "__main__":
    unittest.main()
