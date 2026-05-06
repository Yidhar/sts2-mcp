"""Tests for the typed selection / mutation contract helper (P0-5)."""

from __future__ import annotations

import sys
import unittest
from pathlib import Path
from typing import Any


RL_AGENT_ROOT = Path(__file__).resolve().parents[1]
if str(RL_AGENT_ROOT) not in sys.path:
    sys.path.insert(0, str(RL_AGENT_ROOT))

from sts2_env.selection_typed import (
    is_text_fallback_only,
    selection_confidence,
    selection_operation_type,
    selection_view,
)


def _bridge_typed_action(operation: str, **selection_extra: Any) -> dict[str, Any]:
    """Build an action whose bridge ``selection`` block is fully typed."""
    selection = {
        "operation_type": operation,
        "screen_type": "card_selection",
        "source": "card",
        "source_zone": "hand",
        "destination_zone": "discard" if operation == "discard" else "exhaust",
        "min_count": 1,
        "max_count": 1,
        "selection_required": True,
        "modifier_id": "",
        "confidence": "runtime_internal",
    }
    selection.update(selection_extra)
    return {
        "kind": "play_card",
        "action_id": f"play:typed_{operation}",
        "selection": selection,
    }


def _profile_only_action(op_name: str) -> dict[str, Any]:
    """Action without ``selection`` block but with a typed card_effect_profile."""
    return {
        "kind": "play_card",
        "action_id": f"play:profile_{op_name}",
        "card": {
            "id": f"CARD.{op_name.upper()}",
            "title": "ProfileOnly",
            "type": "Skill",
            "card_effect_profile": {
                "operations": [{"op": op_name, "source_zone": "hand"}],
                "semantic_tags": [],
                "training_tags": [],
            },
        },
    }


def _text_only_action(prompt: str) -> dict[str, Any]:
    """Action without typed selection or profile — only localized text."""
    return {
        "kind": "play_card",
        "action_id": "play:text_only",
        "selection_prompt": prompt,
        "card": {"id": "X", "title": "X"},
    }


class TypedBridgeBlockPath(unittest.TestCase):
    def test_runtime_internal_discard(self):
        view = selection_view(_bridge_typed_action("discard"))
        self.assertEqual(view["operation_type"], "discard")
        self.assertEqual(view["confidence"], "runtime_internal")
        self.assertTrue(view["selection_required"])
        self.assertEqual(view["source_zone"], "hand")

    def test_runtime_internal_retain(self):
        self.assertEqual(selection_operation_type(_bridge_typed_action("retain")), "retain")

    def test_runtime_internal_transform(self):
        self.assertEqual(selection_operation_type(_bridge_typed_action("transform")), "transform")

    def test_runtime_internal_upgrade(self):
        self.assertEqual(selection_operation_type(_bridge_typed_action("upgrade")), "upgrade")

    def test_runtime_internal_exhaust(self):
        self.assertEqual(selection_operation_type(_bridge_typed_action("exhaust")), "exhaust")

    def test_runtime_internal_from_typed_selection_key(self):
        action = _bridge_typed_action("exhaust")
        action["typed_selection"] = action.pop("selection")
        action["selection"] = "select"

        view = selection_view(action)
        self.assertEqual(view["operation_type"], "exhaust")
        self.assertEqual(view["confidence"], "runtime_internal")

    def test_runtime_internal_from_selection_typed_alias(self):
        action = _bridge_typed_action("discard")
        action["selection_typed"] = action.pop("selection")
        action["selection"] = "select"

        view = selection_view(action)
        self.assertEqual(view["operation_type"], "discard")
        self.assertEqual(view["confidence"], "runtime_internal")

    def test_runtime_internal_from_nested_card_selection_block(self):
        action = {"kind": "play_card", "selection": "select", "card": {}}
        action["card"]["selection"] = {
            "operation_type": "retain",
            "screen_type": "card_selection",
            "source": "card",
            "source_zone": "hand",
            "destination_zone": "hand",
            "min_count": 1,
            "max_count": 1,
            "selection_required": True,
            "modifier_id": "",
            "confidence": "runtime_internal",
        }

        view = selection_view(action)
        self.assertEqual(view["operation_type"], "retain")
        self.assertEqual(view["confidence"], "runtime_internal")

    def test_unknown_operation_falls_through_to_profile(self):
        # Bridge wrote an unrecognized operation_type — we must not trust it.
        action = _bridge_typed_action("does_not_exist")
        action["card"] = _profile_only_action("upgrade_card")["card"]
        view = selection_view(action)
        self.assertEqual(view["operation_type"], "upgrade")
        self.assertEqual(view["confidence"], "static_export")


class TypedProfileFallbackPath(unittest.TestCase):
    def test_profile_upgrade(self):
        view = selection_view(_profile_only_action("upgrade_card"))
        self.assertEqual(view["operation_type"], "upgrade")
        self.assertEqual(view["confidence"], "static_export")

    def test_profile_exhaust(self):
        self.assertEqual(selection_operation_type(_profile_only_action("exhaust_card")), "exhaust")

    def test_profile_transform(self):
        self.assertEqual(selection_operation_type(_profile_only_action("transform_card")), "transform")

    def test_profile_discard(self):
        self.assertEqual(selection_operation_type(_profile_only_action("discard_card")), "discard")

    def test_profile_retain(self):
        self.assertEqual(selection_operation_type(_profile_only_action("retain_card")), "retain")

    def test_profile_modify_cost_is_enchant(self):
        self.assertEqual(selection_operation_type(_profile_only_action("modify_cost")), "enchant")


class TextFallbackPath(unittest.TestCase):
    def test_text_only_resolves_with_low_confidence(self):
        view = selection_view(_text_only_action("Discard a card from your hand."))
        self.assertEqual(view["operation_type"], "discard")
        self.assertEqual(view["confidence"], "text_fallback")
        self.assertTrue(is_text_fallback_only(_text_only_action("Discard a card.")))

    def test_text_only_upgrade(self):
        view = selection_view(_text_only_action("Upgrade a card in your hand."))
        self.assertEqual(view["operation_type"], "upgrade")
        self.assertEqual(view["confidence"], "text_fallback")

    def test_text_only_exhaust(self):
        view = selection_view(_text_only_action("Exhaust the chosen card."))
        self.assertEqual(view["operation_type"], "exhaust")
        self.assertEqual(view["confidence"], "text_fallback")


class NoSignalPath(unittest.TestCase):
    def test_unknown_action_returns_empty_op(self):
        view = selection_view({"kind": "play_card", "action_id": "play:noop"})
        self.assertEqual(view["operation_type"], "")
        self.assertEqual(view["confidence"], "none")
        self.assertFalse(is_text_fallback_only({"kind": "play_card", "action_id": "play:noop"}))

    def test_empty_runtime_card_selection_metadata_does_not_leak_confidence(self):
        # Full bridge card payloads can carry a nested ``card.selection`` block
        # whose operation_type is empty metadata.  That must not inflate
        # selection_runtime_internal_selected_rate for ordinary play_card
        # actions.
        action = {
            "kind": "play_card",
            "action_id": "play:ordinary_strike",
            "card": {
                "id": "CARD.STRIKE",
                "title": "Strike",
                "selection": {
                    "operation_type": "",
                    "screen_type": "",
                    "source": "",
                    "confidence": "runtime_internal",
                },
            },
        }
        view = selection_view(action)
        self.assertEqual(view["operation_type"], "")
        self.assertEqual(view["confidence"], "none")

    def test_unknown_runtime_selection_without_fallback_does_not_leak_confidence(self):
        action = {
            "kind": "card_selection",
            "selection": "select",
            "typed_selection": {
                "operation_type": "future_unknown_op",
                "screen_type": "card_selection",
                "confidence": "runtime_internal",
            },
        }
        view = selection_view(action)
        self.assertEqual(view["operation_type"], "")
        self.assertEqual(view["confidence"], "none")

    def test_none_action(self):
        view = selection_view(None)
        self.assertEqual(view["operation_type"], "")
        self.assertEqual(view["confidence"], "none")


class ConfidencePriorityOrdering(unittest.TestCase):
    def test_typed_block_wins_over_card_profile(self):
        # Action carries BOTH bridge typed (transform) and card profile (upgrade).
        action = _bridge_typed_action("transform")
        action["card"] = _profile_only_action("upgrade_card")["card"]
        view = selection_view(action)
        self.assertEqual(view["operation_type"], "transform")
        self.assertEqual(view["confidence"], "runtime_internal")

    def test_profile_wins_over_text(self):
        action = _profile_only_action("exhaust_card")
        action["selection_prompt"] = "Discard a card."
        view = selection_view(action)
        self.assertEqual(view["operation_type"], "exhaust")
        self.assertEqual(view["confidence"], "static_export")
        self.assertFalse(is_text_fallback_only(action))


if __name__ == "__main__":
    unittest.main()
