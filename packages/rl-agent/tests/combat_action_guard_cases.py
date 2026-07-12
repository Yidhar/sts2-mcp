"""Shared deterministic payload builders for combat hard-guard tests."""

from __future__ import annotations


def _boss_race_strength_profile(**overrides):
    profile = {
        "low_urgency": True,
        "save_recommended": True,
        "no_followup": True,
        "requires_followup": True,
        "block_waste": False,
        "overkill": False,
        "urgent": False,
        "lethal": False,
        "prevent_lethal": False,
        "mechanism_answer": False,
        "hp_valid": True,
        "hp": 70.0,
        "max_hp": 80.0,
        "hp_ratio": 70.0 / 80.0,
        "threat_gap": 19.0,
        "resource_like": False,
        "potion_id": "POTION.STRENGTH_POTION",
        "effect_family": ["buff", "strength"],
        "semantic_tags": ["buff", "scaling", "strength"],
        "timing_tags": ["scaling_tool", "requires_followup"],
        "training_tags": ["combat_immediate"],
        "retrieve_from_discard": 0.0,
        "retrieve_from_discard_like": False,
        "retrieve_has_target": False,
        "resource_survival_tool": False,
        "critical_hp_survival_tool": False,
        "heal": 0.0,
        "block": 0.0,
        "damage": 0.0,
        "use_quality": 0.0,
    }
    profile.update(overrides)
    return profile
