"""Tests for the future-world card-lifecycle aux head (TASK-D3).

Verifies the spec acceptance shapes:

1. Plain Strike → hand-1, discard+1, card_moved_to_discard_prob=1.
2. Exhaust card → hand-1, exhaust+1, card_moved_to_exhaust_prob=1.
3. Draw card → drawn_card_count > 0 and hand size delta consistent with draw.
4. Armaments-style upgrade → hand_upgraded_count delta > 0.
5. Facing-target action → next_kaiser_facing reflects observed enemy facing.

The aux target is self-supervised — it only reads (prev_obs, action, next_obs)
diffs and never depends on hand-labeled future predictions.
"""

from __future__ import annotations

import sys
import unittest
from pathlib import Path
from typing import Any

import importlib.util
import types


RL_AGENT_ROOT = Path(__file__).resolve().parents[1]
if str(RL_AGENT_ROOT) not in sys.path:
    sys.path.insert(0, str(RL_AGENT_ROOT))


# Mirror the lightweight loader pattern from test_enemy_state_aux so this
# module does not need the full sts2_env package import (which pulls torch,
# stable-baselines, etc.).  We only need pure-python aux_targets primitives.
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


class _TorchTensorStub:  # pragma: no cover
    pass


class _TorchDtypeStub:  # pragma: no cover
    pass


_real_torch = sys.modules.get("torch")
_stub(
    "torch",
    Tensor=_TorchTensorStub,
    dtype=_TorchDtypeStub,
    float32=_TorchDtypeStub(),
    long=_TorchDtypeStub(),
    bool=_TorchDtypeStub(),
)

_real_text_encoder = sys.modules.get("sts2_env.text_encoder")
_stub("sts2_env.text_encoder", TEXT_DIM=512)

_load("sts2_env.semantic_action", RL_AGENT_ROOT / "sts2_env" / "semantic_action.py")
_load("sts2_env.run_memory", RL_AGENT_ROOT / "sts2_env" / "run_memory.py")
_load("sts2_env.observation_common", RL_AGENT_ROOT / "sts2_env" / "observation_common.py")
_load("sts2_env.objective_heads", RL_AGENT_ROOT / "sts2_env" / "objective_heads.py")
aux_targets = _load("sts2_env.aux_targets", RL_AGENT_ROOT / "sts2_env" / "aux_targets.py")

if _real_torch is not None:
    sys.modules["torch"] = _real_torch
else:
    sys.modules.pop("torch", None)
if _real_text_encoder is not None:
    sys.modules["sts2_env.text_encoder"] = _real_text_encoder
else:
    sys.modules.pop("sts2_env.text_encoder", None)


def _card(uid: str, title: str, *, cost: int = 1, upgraded: bool = False, current_cost: int | None = None) -> dict[str, Any]:
    base = {
        "uid": uid,
        "id": f"CARD.{title.upper()}",
        "title": title + ("+" if upgraded else ""),
        "cost": cost,
        "is_upgraded": upgraded,
    }
    if current_cost is not None:
        base["current_cost"] = current_cost
    return base


def _enemy(eid: str, hp: int = 50, intent_damage: int = 10, facing: str = "player", powers: list[Any] | None = None) -> dict[str, Any]:
    return {
        "id": eid,
        "hp": hp,
        "max_hp": 100,
        "block": 0,
        "intent": {"intent_type": "attack", "total_damage": intent_damage, "damage_per_hit": intent_damage, "repeats": 1},
        "facing": facing,
        "powers": powers or [],
    }


def _obs(*, hand: list[dict[str, Any]], draw: list[dict[str, Any]], discard: list[dict[str, Any]], exhaust: list[dict[str, Any]], energy: int = 3, block: int = 0, enemies: list[dict[str, Any]] | None = None) -> dict[str, Any]:
    return {
        "combat": {
            "encounter_id": "ENC.TEST",
            "hand": hand,
            "draw_pile": draw,
            "discard_pile": discard,
            "exhaust_pile": exhaust,
            "enemies": enemies if enemies is not None else [_enemy("E1")],
            "self_inflicted_hp_loss_cumulative": 0.0,
        },
        "player": {"hp": 70, "max_hp": 80, "block": block, "energy": energy},
    }


def _play(card: dict[str, Any]) -> dict[str, Any]:
    return {
        "kind": "play_card",
        "action_id": f"play:{card['id']}",
        "card": card,
    }


class FutureLifecycleStrikeTests(unittest.TestCase):
    def test_plain_strike_lands_in_discard(self):
        strike = _card("u1", "Strike")
        prev = _obs(hand=[strike, _card("u2", "Defend")], draw=[], discard=[], exhaust=[])
        nxt = _obs(hand=[_card("u2", "Defend")], draw=[], discard=[strike], exhaust=[])
        target = aux_targets.compute_future_lifecycle_targets(prev, _play(strike), nxt)
        names = aux_targets.FUTURE_LIFECYCLE_HEAD_NAMES
        idx = {name: i for i, name in enumerate(names)}
        self.assertAlmostEqual(target[idx["card_moved_to_discard_prob"]], 1.0)
        self.assertAlmostEqual(target[idx["card_moved_to_exhaust_prob"]], 0.0)
        self.assertAlmostEqual(target[idx["card_retained_prob"]], 0.0)


class FutureLifecycleExhaustTests(unittest.TestCase):
    def test_exhaust_card_lands_in_exhaust_pile(self):
        apparition = _card("u9", "Apparition")
        prev = _obs(hand=[apparition, _card("u2", "Defend")], draw=[], discard=[], exhaust=[])
        nxt = _obs(hand=[_card("u2", "Defend")], draw=[], discard=[], exhaust=[apparition])
        target = aux_targets.compute_future_lifecycle_targets(prev, _play(apparition), nxt)
        names = aux_targets.FUTURE_LIFECYCLE_HEAD_NAMES
        idx = {name: i for i, name in enumerate(names)}
        self.assertAlmostEqual(target[idx["card_moved_to_exhaust_prob"]], 1.0)
        self.assertAlmostEqual(target[idx["card_moved_to_discard_prob"]], 0.0)
        self.assertGreater(target[idx["next_exhaust_count_ratio"]], 0.0)


class FutureLifecycleDrawTests(unittest.TestCase):
    def test_draw_card_increases_drawn_count(self):
        cycle = _card("u3", "Cycle")
        prev = _obs(hand=[cycle], draw=[_card("u4", "Strike"), _card("u5", "Defend")], discard=[], exhaust=[])
        # After playing Cycle, hand has the two new cards (Cycle exhausted in
        # this fictional version going to discard); drawn 2 cards so hand
        # size goes from 1 -> 2.
        nxt = _obs(hand=[_card("u4", "Strike"), _card("u5", "Defend")], draw=[], discard=[cycle], exhaust=[])
        target = aux_targets.compute_future_lifecycle_targets(prev, _play(cycle), nxt)
        names = aux_targets.FUTURE_LIFECYCLE_HEAD_NAMES
        idx = {name: i for i, name in enumerate(names)}
        # hand_count delta = 2 - 1 = 1 → drawn_proxy = max(0, hand_after - (hand_before - 1)) = max(0, 2 - 0) = 2
        self.assertGreaterEqual(target[idx["drawn_card_count"]], 1.0)


class FutureLifecycleUpgradeTests(unittest.TestCase):
    def test_armaments_increases_hand_upgraded_count(self):
        armaments = _card("u8", "Armaments")
        target_card_pre = _card("u6", "Strike", upgraded=False)
        target_card_post = _card("u6", "Strike", upgraded=True)
        prev = _obs(hand=[armaments, target_card_pre], draw=[], discard=[], exhaust=[])
        nxt = _obs(hand=[target_card_post], draw=[], discard=[armaments], exhaust=[])
        target = aux_targets.compute_future_lifecycle_targets(prev, _play(armaments), nxt)
        names = aux_targets.FUTURE_LIFECYCLE_HEAD_NAMES
        idx = {name: i for i, name in enumerate(names)}
        self.assertEqual(target[idx["hand_upgraded_count"]], 1.0)


class FutureLifecycleFacingTests(unittest.TestCase):
    def test_facing_target_action_updates_kaiser_facing(self):
        kaiser_strike = _card("u11", "KaiserSpin")
        prev = _obs(hand=[kaiser_strike], draw=[], discard=[], exhaust=[],
                    enemies=[_enemy("kaiser", facing="player")])
        nxt = _obs(hand=[], draw=[], discard=[kaiser_strike], exhaust=[],
                   enemies=[_enemy("kaiser", facing="back")])
        target = aux_targets.compute_future_lifecycle_targets(prev, _play(kaiser_strike), nxt)
        names = aux_targets.FUTURE_LIFECYCLE_HEAD_NAMES
        idx = {name: i for i, name in enumerate(names)}
        self.assertAlmostEqual(target[idx["next_kaiser_facing"]], 0.0)
        # Back-attack risk = 1 when enemy now facing back AND has attack intent.
        self.assertAlmostEqual(target[idx["next_back_attack_risk"]], 1.0)


class FutureLifecycleCeremonialTests(unittest.TestCase):
    def test_ceremonial_lock_state_reflects_enemy_power(self):
        spin = _card("u12", "BlockUp")
        prev = _obs(hand=[spin], draw=[], discard=[], exhaust=[],
                    enemies=[_enemy("ceremonial_beast")])
        nxt = _obs(hand=[], draw=[], discard=[spin], exhaust=[],
                   enemies=[_enemy("ceremonial_beast", powers=[{"id": "CeremonialOneCardLockPower"}])])
        target = aux_targets.compute_future_lifecycle_targets(prev, _play(spin), nxt)
        names = aux_targets.FUTURE_LIFECYCLE_HEAD_NAMES
        idx = {name: i for i, name in enumerate(names)}
        self.assertAlmostEqual(target[idx["next_ceremonial_lock_state"]], 1.0)


class FutureLifecycleBuildAuxTargetsIntegrationTest(unittest.TestCase):
    def test_build_aux_targets_includes_future_lifecycle_v5(self):
        strike = _card("u1", "Strike")
        prev = _obs(hand=[strike], draw=[], discard=[], exhaust=[])
        nxt = _obs(hand=[], draw=[], discard=[strike], exhaust=[])
        result = aux_targets.build_aux_targets(prev, _play(strike), nxt)
        self.assertIn("future_lifecycle", result)
        self.assertIn("future_lifecycle_mask", result)
        self.assertIn("future_lifecycle_names", result)
        self.assertEqual(result["future_lifecycle"].shape, (aux_targets.NUM_FUTURE_LIFECYCLE_HEADS,))
        self.assertEqual(result["version"], 5)


if __name__ == "__main__":
    unittest.main()
