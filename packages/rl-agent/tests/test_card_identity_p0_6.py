"""Tests for the stable card-instance identity helper (P0-6).

The future_world_aux head uses card identity to mark Bernoulli destination
targets (``card_moved_to_exhaust_prob``, ``card_moved_to_discard_prob``,
``card_retained_prob``).  Without a stable UUID, two copies of the same
card (or a transformed card whose title changed) would silently false-
match on the title or definition id.  This test suite locks down the
contract that:

* ``runtime_internal`` (instance UUID) is the only confidence allowed to
  drive lifecycle aux Bernoulli targets.
* ``static_export`` (definition id) is OK as a coarse cluster id but
  MUST NOT be used to declare per-instance destinations.
* ``text_fallback`` (title) is diagnostics-only.
"""

from __future__ import annotations

import importlib.util
import sys
import types
import unittest
from pathlib import Path
from typing import Any


RL_AGENT_ROOT = Path(__file__).resolve().parents[1]
if str(RL_AGENT_ROOT) not in sys.path:
    sys.path.insert(0, str(RL_AGENT_ROOT))


def _stub(name: str, **attrs) -> types.ModuleType:
    module = types.ModuleType(name)
    for k, v in attrs.items():
        setattr(module, k, v)
    sys.modules[name] = module
    return module


_pkg = sys.modules.get("sts2_env")
if _pkg is None:
    _pkg = types.ModuleType("sts2_env")
    _pkg.__path__ = [str(RL_AGENT_ROOT / "sts2_env")]
    sys.modules["sts2_env"] = _pkg


def _load(qualified: str, path: Path) -> types.ModuleType:
    spec = importlib.util.spec_from_file_location(qualified, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[qualified] = module
    spec.loader.exec_module(module)
    return module


# Light-weight load of card_identity (pure python, zero deps) and aux_targets
# along the same loader chain used by other lightweight tests.
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
card_identity_mod = _load("sts2_env.card_identity", RL_AGENT_ROOT / "sts2_env" / "card_identity.py")
aux_targets = _load("sts2_env.aux_targets", RL_AGENT_ROOT / "sts2_env" / "aux_targets.py")

if _real_torch is not None:
    sys.modules["torch"] = _real_torch
else:
    sys.modules.pop("torch", None)
if _real_text_encoder is not None:
    sys.modules["sts2_env.text_encoder"] = _real_text_encoder
else:
    sys.modules.pop("sts2_env.text_encoder", None)


def _card_with_uuid(uuid: str, title: str = "Strike") -> dict[str, Any]:
    return {"instance_uuid": uuid, "id": "CARD.STRIKE", "title": title}


def _card_id_only(card_id: str = "CARD.STRIKE", title: str = "Strike") -> dict[str, Any]:
    return {"id": card_id, "title": title}


def _card_title_only(title: str = "Strike") -> dict[str, Any]:
    return {"title": title}


class CardIdentityResolutionTests(unittest.TestCase):
    def test_instance_uuid_runtime_internal(self):
        ident = card_identity_mod.card_identity(_card_with_uuid("uuid-123"))
        self.assertEqual(ident["confidence"], "runtime_internal")
        self.assertEqual(ident["source"], "instance_uuid")
        self.assertEqual(ident["key"], "uuid:uuid-123")

    def test_compact_like_action_card_instance_uuid_runtime_internal(self):
        action = {
            "kind": "play_card",
            "card": {
                "id": "CARD.STRIKE",
                "title": "Strike",
                "instance_uuid": "ABCD",
            },
        }
        ident = card_identity_mod.card_identity(action["card"])
        self.assertEqual(ident["confidence"], "runtime_internal")
        self.assertEqual(ident["source"], "instance_uuid")
        self.assertEqual(ident["key"], "uuid:ABCD")

    def test_alias_uid_runtime_internal(self):
        ident = card_identity_mod.card_identity({"uid": "u1"})
        self.assertEqual(ident["confidence"], "runtime_internal")
        self.assertEqual(ident["source"], "uid")

    def test_combat_uuid_runtime_internal(self):
        ident = card_identity_mod.card_identity({"combat_uuid": "c-7"})
        self.assertEqual(ident["confidence"], "runtime_internal")
        self.assertEqual(ident["source"], "combat_uuid")

    def test_id_only_falls_to_static_export(self):
        ident = card_identity_mod.card_identity(_card_id_only())
        self.assertEqual(ident["confidence"], "static_export")
        self.assertEqual(ident["key"], "id:CARD.STRIKE")

    def test_title_only_falls_to_text_fallback(self):
        ident = card_identity_mod.card_identity(_card_title_only("Reaper"))
        self.assertEqual(ident["confidence"], "text_fallback")
        self.assertEqual(ident["source"], "title")

    def test_empty_returns_none(self):
        self.assertEqual(card_identity_mod.card_identity({})["confidence"], "none")
        self.assertEqual(card_identity_mod.card_identity(None)["confidence"], "none")

    def test_namespacing_prevents_uuid_id_collision(self):
        a = card_identity_mod.card_identity_key({"uid": "abc"})
        b = card_identity_mod.card_identity_key({"id": "abc"})
        self.assertNotEqual(a, b)


class CardsMatchTests(unittest.TestCase):
    def test_uuid_match(self):
        self.assertTrue(card_identity_mod.cards_match(_card_with_uuid("u1"), _card_with_uuid("u1")))

    def test_uuid_mismatch(self):
        self.assertFalse(card_identity_mod.cards_match(_card_with_uuid("u1"), _card_with_uuid("u2")))

    def test_two_unknown_identity_cards_dont_match(self):
        # Both empty → returns False (don't match unknowns).
        self.assertFalse(card_identity_mod.cards_match({}, {}))

    def test_strict_uuid_rejects_id_only(self):
        # Same id but no UUID → cards_match returns True (id matches),
        # cards_match_strict_uuid returns False (no UUID).
        a = _card_id_only()
        b = _card_id_only()
        self.assertTrue(card_identity_mod.cards_match(a, b))
        self.assertFalse(card_identity_mod.cards_match_strict_uuid(a, b))

    def test_strict_uuid_accepts_uuid_match(self):
        a = _card_with_uuid("u1")
        b = _card_with_uuid("u1")
        self.assertTrue(card_identity_mod.cards_match_strict_uuid(a, b))

    def test_strict_uuid_rejects_mixed_uuid_and_id(self):
        a = _card_with_uuid("u1")
        b = _card_id_only()
        self.assertFalse(card_identity_mod.cards_match_strict_uuid(a, b))


class FutureLifecycleAuxIdentityGate(unittest.TestCase):
    """Lifecycle aux Bernoulli targets MUST require runtime_internal UUID."""

    def _obs(self, hand=(), draw=(), discard=(), exhaust=()) -> dict[str, Any]:
        return {
            "combat": {
                "encounter_id": "ENC.TEST",
                "hand": list(hand),
                "draw_pile": list(draw),
                "discard_pile": list(discard),
                "exhaust_pile": list(exhaust),
                "enemies": [{"id": "e1", "hp": 30, "max_hp": 50, "block": 0,
                             "intent": {"intent_type": "attack", "total_damage": 5},
                             "powers": []}],
                "self_inflicted_hp_loss_cumulative": 0.0,
            },
            "player": {"hp": 70, "max_hp": 80, "block": 0, "energy": 3},
        }

    def test_played_card_with_uuid_emits_destination_target(self):
        played = _card_with_uuid("u-strike-1")
        prev = self._obs(hand=[played, _card_with_uuid("u-defend-1", "Defend")])
        nxt = self._obs(hand=[_card_with_uuid("u-defend-1", "Defend")], discard=[played])
        action = {"kind": "play_card", "action_id": "play:u-strike-1", "card": played}
        target = aux_targets.compute_future_lifecycle_targets(prev, action, nxt)
        names = aux_targets.FUTURE_LIFECYCLE_HEAD_NAMES
        idx = {n: i for i, n in enumerate(names)}
        self.assertAlmostEqual(target[idx["card_moved_to_discard_prob"]], 1.0)
        self.assertAlmostEqual(target[idx["card_moved_to_exhaust_prob"]], 0.0)

    def test_played_card_id_only_does_not_emit_destination_target(self):
        # Two copies of Strike (same definition id, no instance UUID) — old
        # code would false-match on title/id and emit a confident destination
        # target.  P0-6 says: don't claim — leave at 0 (head not supervised).
        played = _card_id_only("CARD.STRIKE")
        twin = _card_id_only("CARD.STRIKE")
        prev = self._obs(hand=[played, twin])
        nxt = self._obs(hand=[twin], discard=[played])
        action = {"kind": "play_card", "action_id": "play:strike", "card": played}
        target = aux_targets.compute_future_lifecycle_targets(prev, action, nxt)
        names = aux_targets.FUTURE_LIFECYCLE_HEAD_NAMES
        idx = {n: i for i, n in enumerate(names)}
        self.assertAlmostEqual(target[idx["card_moved_to_discard_prob"]], 0.0)
        self.assertAlmostEqual(target[idx["card_moved_to_exhaust_prob"]], 0.0)
        self.assertAlmostEqual(target[idx["card_retained_prob"]], 0.0)

    def test_played_card_title_only_does_not_emit_destination_target(self):
        played = _card_title_only("Strike")
        prev = self._obs(hand=[played, _card_title_only("Defend")])
        nxt = self._obs(hand=[_card_title_only("Defend")], discard=[played])
        action = {"kind": "play_card", "action_id": "play:strike", "card": played}
        target = aux_targets.compute_future_lifecycle_targets(prev, action, nxt)
        names = aux_targets.FUTURE_LIFECYCLE_HEAD_NAMES
        idx = {n: i for i, n in enumerate(names)}
        self.assertAlmostEqual(target[idx["card_moved_to_discard_prob"]], 0.0)


if __name__ == "__main__":
    unittest.main()
