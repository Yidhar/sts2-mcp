"""P2-1 runtime card-state reader tests.

Validates the typed reader at ``sts2_env/card_runtime_state.py``: alias
chain resolution, presence-flag derivation, and confidence labelling.
"""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from sts2_env.card_runtime_state import (
    card_runtime_confidence,
    card_runtime_field_present,
    card_runtime_presence_flags,
    card_runtime_state,
)


def test_runtime_state_recognises_runtime_internal_uuid():
    card = {"id": "CARD.STRIKE", "instance_uuid": "abc-123", "cost": 1}
    state = card_runtime_state(card)
    assert state["confidence"] == "runtime_internal"
    assert state["identity"]["card_id"] == "CARD.STRIKE"
    assert state["identity"]["instance_uuid"] == "abc-123"
    assert state["identity"]["cost"] == 1
    assert state["presence"]["identity_count"] >= 3


def test_runtime_state_falls_back_to_static_export():
    card = {"id": "CARD.STRIKE"}
    state = card_runtime_state(card)
    assert state["confidence"] == "static_export"


def test_runtime_state_falls_back_to_text():
    card = {"title": "Strike"}
    state = card_runtime_state(card)
    assert state["confidence"] == "text_fallback"


def test_runtime_state_resolves_alias_for_modifier():
    """Alias chain: ``ethereal`` should be read as ``ethereal_this_combat``
    so older bridge payloads don't drop the flag."""
    card = {"id": "CARD.X", "ethereal": True}
    state = card_runtime_state(card)
    assert state["modifiers"]["ethereal_this_combat"] is True
    assert card_runtime_field_present(card, "ethereal_this_combat") is True


def test_presence_flags_set_when_modifier_present():
    card = {
        "id": "CARD.X",
        "instance_uuid": "u",
        "exhaust_this_combat": True,
        "ethereal": True,
        "retain_this_turn": True,
        "enchanted": True,
        "replay_count": 2,
        "requires_card_selection": True,
        "cost": 2,
        "cost_for_turn": 0,  # cost modified
    }
    flags = card_runtime_presence_flags(card)
    assert flags["instance_uuid_present"] == 1.0
    assert flags["modified_cost_present"] == 1.0
    assert flags["exhaust_flag_present"] == 1.0
    assert flags["ethereal_flag_present"] == 1.0
    assert flags["retain_flag_present"] == 1.0
    assert flags["enchantment_present"] == 1.0
    assert flags["replay_flag_present"] == 1.0
    assert flags["selection_effect_present"] == 1.0


def test_presence_flags_zero_when_modifier_absent():
    card = {"id": "CARD.X", "cost": 2, "cost_for_turn": 2}  # cost not modified
    flags = card_runtime_presence_flags(card)
    assert flags["instance_uuid_present"] == 0.0
    assert flags["modified_cost_present"] == 0.0
    assert flags["exhaust_flag_present"] == 0.0
    assert flags["ethereal_flag_present"] == 0.0
    assert flags["retain_flag_present"] == 0.0
    assert flags["enchantment_present"] == 0.0
    assert flags["replay_flag_present"] == 0.0
    assert flags["selection_effect_present"] == 0.0


def test_presence_flags_handle_non_dict_input():
    flags = card_runtime_presence_flags(None)
    for key, value in flags.items():
        assert value == 0.0, f"{key} should be 0.0 for non-dict input"


def test_runtime_confidence_alias():
    assert card_runtime_confidence({"instance_uuid": "u"}) == "runtime_internal"
    assert card_runtime_confidence({"id": "CARD.X"}) == "static_export"
    assert card_runtime_confidence({"title": "Strike"}) == "text_fallback"
    assert card_runtime_confidence(None) == "none"


def test_selection_effect_modifier_aliases():
    """``selection_required`` is the older bridge alias for the canonical
    ``requires_card_selection`` field — verify the alias resolves."""
    card = {"id": "CARD.X", "selection_required": True, "min_count": 1, "max_count": 3}
    state = card_runtime_state(card)
    assert state["selection_effect"]["requires_card_selection"] is True
    assert state["selection_effect"]["min_select"] == 1
    assert state["selection_effect"]["max_select"] == 3
