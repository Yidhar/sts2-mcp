"""Potion timing, discard, and bad-use guard regressions.

The guard helpers are method-on-`MuZeroTrainer`, but the override decision
only depends on a small subset of state (``raw_obs`` + boss context dict +
legal_actions + mask + search_stats). To avoid spinning up the full trainer
for a unit test, we instantiate a minimal stub that mirrors only the
helpers the guard reaches for, then call the guard directly.
"""
from __future__ import annotations

import sys
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from muzero.combat_quality.potion_guard import potion_slot_from_action_for_guard  # noqa: E402


@pytest.fixture
def trainer_stub():
    """Construct a MuZeroTrainer with the absolute minimum config so that
    the guard helper can run without instantiating bridge / network /
    snapshot pool. We patch ``__init__`` to skip heavy setup."""
    from muzero.train import MuZeroTrainer

    trainer = MuZeroTrainer.__new__(MuZeroTrainer)
    trainer.combat_hard_guard_policy = "full"
    trainer.log_dir = None  # skip diagnostics writes
    trainer.episode_count = 0
    trainer.total_steps = 0
    return trainer

def test_potion_bad_use_guard_reclassifies_selected_before_selected_stats(trainer_stub):
    """The potion hard guard runs before selected-side combat_quality stats.

    Regression test: a low-urgency/save/no-followup potion must still be
    blocked even when ``search_stats`` does not yet contain
    ``combat_quality_potion_*_selected`` flags.  The previous implementation
    read those later-populated flags and therefore no-oped in live training.
    """
    raw_obs = {
        "encounter": "ENCOUNTER.SEAPUNK_WEAK",
        "combat": {"energy": 0},
        "player": {"current_hp": 80, "max_hp": 91},
    }
    legal_actions = [
        {"action_id": "use_potion:0", "kind": "use_potion", "potion": {"id": "POTION.STRENGTH", "title": "力量药水"}},
        {"action_id": "end_turn", "kind": "end_turn", "label": "End Turn"},
    ]
    mask = np.array([1, 1], dtype=np.float32)
    search_stats: dict = {}

    with patch.object(trainer_stub, "_is_kaiser_encounter_context", return_value=False), patch.object(
        trainer_stub, "_is_insatiable_encounter_context", return_value=False
    ), patch.object(
        trainer_stub, "_combat_energy", return_value=0.0
    ), patch.object(
        trainer_stub, "_semantic_family", side_effect=lambda action: action.get("kind") or "unknown"
    ), patch.object(
        trainer_stub,
        "_potion_timing_profile",
        return_value={
            "low_urgency": True,
            "save_recommended": True,
            "no_followup": True,
            "block_waste": False,
            "overkill": False,
            "urgent": False,
            "lethal": False,
            "prevent_lethal": False,
            "mechanism_answer": False,
            "hp_valid": True,
            "hp_ratio": 0.90,
            "threat_gap": 0.0,
            "resource_like": True,
        },
    ), patch.object(
        trainer_stub, "_is_action_confirmed_lethal", return_value=False
    ):
        new_idx = trainer_stub._apply_combat_action_hard_guards(
            action_idx=0,
            legal_actions=legal_actions,
            action_mask=mask,
            raw_obs=raw_obs,
            boss_ctx={},
            encounter="ENCOUNTER.SEAPUNK_WEAK",
            search_stats=search_stats,
        )

    assert new_idx == 1
    assert search_stats["combat_quality_potion_bad_guard_available"] == 1.0
    assert search_stats["combat_quality_potion_bad_guard_applied"] == 1.0
    assert search_stats["combat_quality_potion_bad_guard_override"] == 1.0
    assert search_stats["combat_quality_potion_bad_guard_no_alternative"] == 0.0
    assert search_stats["combat_quality_potion_selected"] == 0.0


def test_potion_bad_use_guard_falls_back_to_end_turn_on_safe_idle_low_urgency(trainer_stub):
    """Regression for live Lagavulin stun-turn waste.

    Some bridge potion actions expose only a localized title and no id, so the
    selected-side profile can be merely "low urgency" rather than explicitly
    resource/no-followup.  If the turn is safe and the only alternative is End
    Turn, the guard should still block wasting the potion; the old hp>=0.65
    gate missed common mid-HP boss states such as 47/91.
    """
    raw_obs = {
        "encounter": "ENCOUNTER.LAGAVULIN_MATRIARCH_BOSS",
        "combat": {"energy": 0},
        "player": {"current_hp": 47, "max_hp": 91},
    }
    legal_actions = [
        {"action_id": "use_potion:0", "kind": "use_potion", "potion": {"title": "液态记忆"}},
        {"action_id": "end_turn", "kind": "end_turn", "label": "End Turn"},
    ]
    mask = np.array([1, 1], dtype=np.float32)
    search_stats: dict = {}

    with patch.object(trainer_stub, "_is_kaiser_encounter_context", return_value=False), patch.object(
        trainer_stub, "_is_insatiable_encounter_context", return_value=False
    ), patch.object(
        trainer_stub, "_combat_energy", return_value=0.0
    ), patch.object(
        trainer_stub, "_semantic_family", side_effect=lambda action: action.get("kind") or "unknown"
    ), patch.object(
        trainer_stub,
        "_potion_timing_profile",
        return_value={
            "low_urgency": True,
            "save_recommended": False,
            "no_followup": False,
            "block_waste": False,
            "overkill": False,
            "urgent": False,
            "lethal": False,
            "prevent_lethal": False,
            "mechanism_answer": False,
            "hp_valid": True,
            "hp_ratio": 47 / 91,
            "threat_gap": 0.0,
            "resource_like": False,
        },
    ), patch.object(
        trainer_stub, "_is_action_confirmed_lethal", return_value=False
    ):
        new_idx = trainer_stub._apply_combat_action_hard_guards(
            action_idx=0,
            legal_actions=legal_actions,
            action_mask=mask,
            raw_obs=raw_obs,
            boss_ctx={},
            encounter="ENCOUNTER.LAGAVULIN_MATRIARCH_BOSS",
            search_stats=search_stats,
        )

    assert new_idx == 1


def test_potion_bad_use_guard_prefers_progress_over_idle_defend_fallback(trainer_stub):
    """Low-urgency potion fallback must not choose the first idle Defend.

    Regression from low-memory sandbox v4: Ovicopter/Frog Knight zero-pressure
    turns rewrote a bad potion into the first legal non-potion card, often pure
    Defend, even though a draw/setup/progress card was available.  That creates
    bad_pure_block_selected targets and teaches low-tempo hallway play.
    """
    raw_obs = {
        "encounter": "ENCOUNTER.OVICOPTER_NORMAL",
        "combat": {"energy": 2, "enemies": [{"intent": {"total_damage": 0}}]},
        "player": {"hp": 80, "current_hp": 80, "max_hp": 80, "block": 0},
    }
    legal_actions = [
        {"action_id": "use_potion:0", "kind": "use_potion", "potion": {"id": "POTION.STRENGTH"}},
        {
            "action_id": "play_card_defend",
            "kind": "play_card",
            "card": {
                "id": "CARD.DEFEND",
                "title": "防御",
                "type": "Skill",
                "cost": 1,
                "block": 5,
                "card_effect_profile": {"semantic_tags": ["block"], "training_tags": []},
            },
            "semantic": {"block": 5, "roles": ["block"]},
        },
        {
            "action_id": "play_card_shrug",
            "kind": "play_card",
            "card": {
                "id": "CARD.SHRUG_IT_OFF",
                "title": "耸肩无视",
                "type": "Skill",
                "cost": 1,
                "block": 8,
                "card_effect_profile": {
                    "semantic_tags": ["block", "draw", "setup"],
                    "training_tags": ["draw"],
                },
            },
            "semantic": {"block": 8, "roles": ["block", "draw", "setup"], "immediate_impact": 14},
        },
        {"action_id": "end_turn", "kind": "end_turn", "label": "End Turn"},
    ]
    mask = np.array([1, 1, 1, 1], dtype=np.float32)
    search_stats: dict = {}

    with patch.object(trainer_stub, "_is_kaiser_encounter_context", return_value=False), patch.object(
        trainer_stub, "_is_insatiable_encounter_context", return_value=False
    ), patch.object(
        trainer_stub, "_combat_energy", return_value=2.0
    ), patch.object(
        trainer_stub, "_combat_encounter_tier_from_raw", return_value="normal"
    ), patch.object(
        trainer_stub, "_semantic_family", side_effect=lambda action: action.get("kind") or "unknown"
    ), patch.object(
        trainer_stub,
        "_potion_timing_profile",
        return_value={
            "low_urgency": True,
            "save_recommended": True,
            "no_followup": False,
            "block_waste": False,
            "overkill": False,
            "urgent": False,
            "lethal": False,
            "prevent_lethal": False,
            "mechanism_answer": False,
            "hp_valid": True,
            "hp": 80.0,
            "max_hp": 80.0,
            "hp_ratio": 1.0,
            "threat_gap": 0.0,
            "resource_like": False,
            "potion_id": "POTION.STRENGTH",
            "effect_family": ["buff"],
            "semantic_tags": ["strength"],
            "timing_tags": [],
            "training_tags": [],
        },
    ), patch.object(
        trainer_stub, "_is_action_confirmed_lethal", return_value=False
    ):
        new_idx = trainer_stub._apply_combat_action_hard_guards(
            action_idx=0,
            legal_actions=legal_actions,
            action_mask=mask,
            raw_obs=raw_obs,
            boss_ctx={},
            encounter="ENCOUNTER.OVICOPTER_NORMAL",
            search_stats=search_stats,
        )

    assert new_idx == 2
    assert search_stats["combat_quality_potion_bad_guard_available"] == 1.0
    assert search_stats["combat_quality_potion_bad_guard_applied"] == 1.0
    assert search_stats["combat_quality_potion_bad_guard_progress_fallback"] == 1.0
    assert search_stats["combat_quality_potion_bad_guard_avoided_idle_pure_block"] == 0.0
    assert search_stats["combat_quality_potion_bad_guard_available"] == 1.0
    assert search_stats["combat_quality_potion_bad_guard_applied"] == 1.0
    assert search_stats["combat_quality_potion_bad_guard_override"] == 1.0


def test_potion_bad_use_guard_falls_back_to_end_turn_on_safe_high_hp_nonidle_low_urgency(trainer_stub):
    """Live offender regression: high-HP hallway, 0 energy, only End Turn.

    Strength/Speed-style potions need a follow-up turn/card window.  When the
    player is safe enough to simply take the current hit, the hard guard should
    rewrite the bad potion to End Turn even if the turn is not fully idle.
    """
    raw_obs = {
        "encounter": "ENCOUNTER.CORPSE_SLUGS_NORMAL",
        "combat": {"energy": 0},
        "player": {"hp": 78, "current_hp": 78, "max_hp": 91, "block": 0},
    }
    legal_actions = [
        {"action_id": "use_potion:0", "kind": "use_potion", "potion": {"id": "POTION.SPEED_POTION", "title": "速度药水"}},
        {"action_id": "end_turn", "kind": "end_turn", "label": "End Turn"},
    ]
    mask = np.array([1, 1], dtype=np.float32)
    search_stats: dict = {}

    with patch.object(trainer_stub, "_is_kaiser_encounter_context", return_value=False), patch.object(
        trainer_stub, "_is_insatiable_encounter_context", return_value=False
    ), patch.object(
        trainer_stub, "_combat_energy", return_value=0.0
    ), patch.object(
        trainer_stub, "_semantic_family", side_effect=lambda action: action.get("kind") or "unknown"
    ), patch.object(
        trainer_stub,
        "_potion_timing_profile",
        return_value={
            "low_urgency": True,
            "save_recommended": False,
            "no_followup": True,
            "requires_followup": True,
            "block_waste": False,
            "overkill": False,
            "urgent": False,
            "lethal": False,
            "prevent_lethal": False,
            "mechanism_answer": False,
            "hp_valid": True,
            "hp_ratio": 78 / 91,
            "hp": 78.0,
            "max_hp": 91.0,
            "threat_gap": 14.0,
            "resource_like": False,
        },
    ), patch.object(
        trainer_stub, "_is_action_confirmed_lethal", return_value=False
    ):
        new_idx = trainer_stub._apply_combat_action_hard_guards(
            action_idx=0,
            legal_actions=legal_actions,
            action_mask=mask,
            raw_obs=raw_obs,
            boss_ctx={},
            encounter="ENCOUNTER.CORPSE_SLUGS_NORMAL",
            search_stats=search_stats,
        )

    assert new_idx == 1
    assert search_stats["combat_quality_potion_bad_guard_available"] == 1.0
    assert search_stats["combat_quality_potion_bad_guard_applied"] == 1.0
    assert search_stats["combat_quality_potion_bad_guard_override"] == 1.0


def test_potion_bad_use_guard_fails_open_when_hp_invalid(trainer_stub):
    """Bad-potion suppression must not invent an End Turn fallback when HP is
    missing from the bridge observation.  Missing HP is telemetry, not grounds
    for an override."""
    raw_obs = {"encounter": "ENCOUNTER.SEAPUNK_WEAK", "combat": {"energy": 0}}
    legal_actions = [
        {"action_id": "use_potion:0", "kind": "use_potion", "potion": {"id": "POTION.STRENGTH"}},
        {"action_id": "strike", "kind": "play_card", "card": {"id": "CARD.STRIKE"}},
        {"action_id": "end_turn", "kind": "end_turn"},
    ]
    mask = np.array([1, 1, 1], dtype=np.float32)
    search_stats: dict = {}

    with patch.object(trainer_stub, "_is_kaiser_encounter_context", return_value=False), patch.object(
        trainer_stub, "_is_insatiable_encounter_context", return_value=False
    ), patch.object(
        trainer_stub, "_combat_energy", return_value=0.0
    ), patch.object(
        trainer_stub, "_semantic_family", side_effect=lambda action: action.get("kind") or "unknown"
    ), patch.object(
        trainer_stub,
        "_potion_timing_profile",
        return_value={
            "low_urgency": True,
            "save_recommended": True,
            "no_followup": True,
            "urgent": False,
            "lethal": False,
            "prevent_lethal": False,
            "mechanism_answer": False,
            "hp_valid": False,
        },
    ), patch.object(
        trainer_stub, "_is_action_confirmed_lethal", return_value=False
    ):
        new_idx = trainer_stub._apply_combat_action_hard_guards(
            action_idx=0,
            legal_actions=legal_actions,
            action_mask=mask,
            raw_obs=raw_obs,
            boss_ctx={},
            encounter="ENCOUNTER.SEAPUNK_WEAK",
            search_stats=search_stats,
        )

    assert new_idx == 0
    assert search_stats["combat_quality_potion_bad_guard_invalid_obs"] == 1.0
    assert search_stats["combat_quality_potion_bad_guard_applied"] == 0.0


def test_potion_bad_use_guard_does_not_fallback_to_end_turn_when_not_safe(trainer_stub):
    """If the only non-potion alternative is End Turn, the guard should only
    use it for explicitly safe/resource-like cases.  This prevents potion guard
    fixes from becoming a new end-turn spam source."""
    raw_obs = {
        "encounter": "ENCOUNTER.SEAPUNK_WEAK",
        "combat": {"energy": 0},
        "player": {"current_hp": 20, "max_hp": 80},
    }
    legal_actions = [
        {"action_id": "use_potion:0", "kind": "use_potion", "potion": {"id": "POTION.STRENGTH"}},
        {"action_id": "end_turn", "kind": "end_turn"},
    ]
    mask = np.array([1, 1], dtype=np.float32)
    search_stats: dict = {}

    with patch.object(trainer_stub, "_is_kaiser_encounter_context", return_value=False), patch.object(
        trainer_stub, "_is_insatiable_encounter_context", return_value=False
    ), patch.object(
        trainer_stub, "_combat_energy", return_value=0.0
    ), patch.object(
        trainer_stub, "_semantic_family", side_effect=lambda action: action.get("kind") or "unknown"
    ), patch.object(
        trainer_stub,
        "_potion_timing_profile",
        return_value={
            "low_urgency": True,
            "save_recommended": True,
            "no_followup": True,
            "block_waste": False,
            "overkill": False,
            "urgent": False,
            "lethal": False,
            "prevent_lethal": False,
            "mechanism_answer": False,
            "hp_valid": True,
            "hp_ratio": 0.25,
            "threat_gap": 0.30,
            "resource_like": False,
        },
    ), patch.object(
        trainer_stub, "_is_action_confirmed_lethal", return_value=False
    ):
        new_idx = trainer_stub._apply_combat_action_hard_guards(
            action_idx=0,
            legal_actions=legal_actions,
            action_mask=mask,
            raw_obs=raw_obs,
            boss_ctx={},
            encounter="ENCOUNTER.SEAPUNK_WEAK",
            search_stats=search_stats,
        )

    assert new_idx == 0
    assert search_stats["combat_quality_potion_bad_guard_available"] == 1.0
    assert search_stats["combat_quality_potion_bad_guard_no_alternative"] == 1.0
    assert search_stats["combat_quality_potion_bad_guard_applied"] == 0.0
    assert search_stats["combat_quality_potion_bad_guard_end_turn_fallback_blocked_unsafe"] == 1.0


def test_potion_bad_use_guard_keeps_late_normal_race_resource_no_followup(trainer_stub):
    """0-energy resource/follow-up potions must not be rewritten to End Turn
    when End Turn itself is unsafe.

    In the late-Act1 race window this is now an explicit race/setup-potion
    fail-open: if the policy wants to use Speed/Strength at 0 energy in a
    dangerous hallway and the bridge exposes no useful non-potion alternative,
    the generic bad-use guard should not create a new End Turn death source.
    """
    raw_obs = {
        "encounter": "ENCOUNTER.CORPSE_SLUGS_NORMAL",
        "run": {"floor": 13},
        "combat": {"energy": 0, "enemies": [{"intent": {"total_damage": 18}}]},
        "player": {"hp": 18, "current_hp": 18, "max_hp": 91, "block": 0},
    }
    legal_actions = [
        {"action_id": "use_potion:0", "kind": "use_potion", "potion": {"id": "POTION.SPEED_POTION", "title": "速度药水"}},
        {"action_id": "end_turn", "kind": "end_turn"},
    ]
    mask = np.array([1, 1], dtype=np.float32)
    search_stats: dict = {}

    with patch.object(trainer_stub, "_is_kaiser_encounter_context", return_value=False), patch.object(
        trainer_stub, "_is_insatiable_encounter_context", return_value=False
    ), patch.object(
        trainer_stub, "_combat_energy", return_value=0.0
    ), patch.object(
        trainer_stub, "_semantic_family", side_effect=lambda action: action.get("kind") or "unknown"
    ), patch.object(
        trainer_stub,
        "_potion_timing_profile",
        return_value={
            "low_urgency": True,
            "save_recommended": True,
            "no_followup": True,
            "requires_followup": True,
            "resource_like": True,
            "block_waste": False,
            "overkill": False,
            "urgent": False,
            "lethal": False,
            "prevent_lethal": False,
            "mechanism_answer": False,
            "hp_valid": True,
            "hp_ratio": 18 / 91,
            "hp": 18.0,
            "max_hp": 91.0,
            "threat_gap": 12.0,
        },
    ), patch.object(
        trainer_stub, "_is_action_confirmed_lethal", return_value=False
    ):
        new_idx = trainer_stub._apply_combat_action_hard_guards(
            action_idx=0,
            legal_actions=legal_actions,
            action_mask=mask,
            raw_obs=raw_obs,
            boss_ctx={},
            encounter="ENCOUNTER.CORPSE_SLUGS_NORMAL",
            search_stats=search_stats,
        )

    assert new_idx == 0
    assert search_stats["combat_quality_potion_bad_guard_late_normal_race_skip"] == 1.0
    assert search_stats["combat_quality_potion_bad_guard_available"] == 0.0
    assert search_stats["combat_quality_potion_bad_guard_applied"] == 0.0
    assert search_stats["combat_quality_potion_bad_guard_no_alternative"] == 0.0
    assert search_stats["combat_quality_potion_bad_guard_end_turn_fallback_blocked_unsafe"] == 0.0


def test_potion_bad_use_guard_forces_end_turn_for_hopeless_fortifier_noop(trainer_stub):
    """0-energy Fortifier at zero block / zero incoming is a pure no-op.

    The previous guard intentionally failed open when End Turn was considered
    unsafe at low HP, but that creates a repeat-use bridge loop for potions
    that do not get consumed and cannot affect the current turn.  This narrow
    exception converts only the provably-hopeless idle no-op into End Turn.
    """
    raw_obs = {
        "encounter": "ENCOUNTER.HAUNTED_SHIP_NORMAL",
        "run": {"floor": 14},
        "combat": {"energy": 0, "enemies": [{"intent": {"total_damage": 0}}]},
        "player": {"hp": 11, "current_hp": 11, "max_hp": 91, "block": 0},
    }
    legal_actions = [
        {"action_id": "use_potion:0", "kind": "use_potion", "potion": {"id": "POTION.FORTIFIER", "title": "固化药水"}},
        {"action_id": "end_turn", "kind": "end_turn"},
    ]
    mask = np.array([1, 1], dtype=np.float32)
    search_stats: dict = {}

    with patch.object(trainer_stub, "_is_kaiser_encounter_context", return_value=False), patch.object(
        trainer_stub, "_is_insatiable_encounter_context", return_value=False
    ), patch.object(
        trainer_stub, "_combat_energy", return_value=0.0
    ), patch.object(
        trainer_stub, "_semantic_family", side_effect=lambda action: action.get("kind") or "unknown"
    ), patch.object(
        trainer_stub,
        "_potion_timing_profile",
        return_value={
            "low_urgency": True,
            "save_recommended": False,
            "no_followup": False,
            "requires_followup": False,
            "resource_like": False,
            "block_waste": True,
            "overkill": False,
            "urgent": False,
            "lethal": False,
            "prevent_lethal": False,
            "mechanism_answer": False,
            "hp_valid": True,
            "hp_ratio": 11 / 91,
            "hp": 11.0,
            "max_hp": 91.0,
            "threat_gap": 0.0,
            "use_quality": 0.0,
        },
    ), patch.object(
        trainer_stub, "_is_action_confirmed_lethal", return_value=False
    ):
        new_idx = trainer_stub._apply_combat_action_hard_guards(
            action_idx=0,
            legal_actions=legal_actions,
            action_mask=mask,
            raw_obs=raw_obs,
            boss_ctx={},
            encounter="ENCOUNTER.HAUNTED_SHIP_NORMAL",
            search_stats=search_stats,
        )

    assert new_idx == 1
    assert search_stats["combat_quality_potion_bad_guard_available"] == 1.0
    assert search_stats["combat_quality_potion_bad_guard_applied"] == 1.0
    assert search_stats["combat_quality_potion_bad_guard_override"] == 1.0
    assert search_stats["combat_quality_potion_bad_guard_forced_end_turn_hopeless"] == 1.0
    assert search_stats["combat_quality_potion_bad_guard_end_turn_fallback_blocked_unsafe"] == 0.0


def test_potion_timing_fortifier_zero_block_is_noop_under_incoming():
    """Fortifier at 0 current block must be classified as no-op, not defense.

    This is the exact late-Act1 failure mode: legal actions are often only
    {Fortifier, End Turn} after energy is spent.  If current block is 0,
    Fortifier cannot reduce incoming damage and should not be rewarded as a
    survival action.
    """
    from sts2_env.potion_timing import compute_potion_timing

    raw_obs = {
        "encounter": "ENCOUNTER.HAUNTED_SHIP_NORMAL",
        "combat": {"energy": 0, "enemies": [{"intent": {"total_damage": 10}}]},
        "player": {"hp": 25, "current_hp": 25, "max_hp": 80, "block": 0},
    }
    action = {
        "action_id": "use_potion:0",
        "kind": "use_potion",
        "potion": {"id": "POTION.FORTIFIER", "title": "固化药水", "description": "将你的格挡变为三倍。"},
    }
    legal_actions = [action, {"action_id": "end_turn", "kind": "end_turn"}]
    profile = compute_potion_timing(action, raw_obs, legal_actions, np.array([1, 1], dtype=np.float32), energy=0.0)

    assert profile["amplify_block_like"] is True
    assert profile["amplify_block_noop"] is True
    assert profile["amplify_block_added"] == 0.0
    assert profile["block"] == 0.0
    assert profile["block_waste"] is True
    assert profile["prevent_major_loss"] is False
    assert profile["urgent"] is False
    assert profile["low_urgency"] is True


def test_potion_timing_fortifier_existing_block_counts_state_dependent_block():
    """Fortifier with current block should estimate added block = 2x current block."""
    from sts2_env.potion_timing import compute_potion_timing

    raw_obs = {
        "encounter": "ENCOUNTER.HAUNTED_SHIP_NORMAL",
        "combat": {"energy": 0, "enemies": [{"intent": {"total_damage": 14}}]},
        "player": {"hp": 25, "current_hp": 25, "max_hp": 80, "block": 5},
    }
    action = {
        "action_id": "use_potion:0",
        "kind": "use_potion",
        "potion": {"id": "POTION.FORTIFIER", "title": "固化药水", "description": "将你的格挡变为三倍。"},
    }
    legal_actions = [action, {"action_id": "end_turn", "kind": "end_turn"}]
    profile = compute_potion_timing(action, raw_obs, legal_actions, np.array([1, 1], dtype=np.float32), energy=0.0)

    assert profile["amplify_block_like"] is True
    assert profile["amplify_block_noop"] is False
    assert profile["amplify_block_added"] == 10.0
    assert profile["block"] == 10.0
    assert profile["block_waste"] is False
    assert profile["no_followup"] is False
    assert profile["prevent_major_loss"] is True
    assert profile["urgent"] is True


def test_potion_bad_use_guard_forces_end_turn_for_fortifier_noop_under_incoming(trainer_stub):
    """Even when End Turn is unsafe, Fortifier at 0 block is strictly worse.

    If the only legal alternatives are a no-op Fortifier and End Turn, the
    guard should preserve the potion slot by ending the turn instead of letting
    the model burn the potion and learn a bad habit.
    """
    raw_obs = {
        "encounter": "ENCOUNTER.HAUNTED_SHIP_NORMAL",
        "run": {"floor": 14},
        "combat": {"energy": 0, "enemies": [{"intent": {"total_damage": 10}}]},
        "player": {"hp": 11, "current_hp": 11, "max_hp": 91, "block": 0},
    }
    legal_actions = [
        {"action_id": "use_potion:0", "kind": "use_potion", "potion": {"id": "POTION.FORTIFIER", "title": "固化药水"}},
        {"action_id": "end_turn", "kind": "end_turn"},
    ]
    mask = np.array([1, 1], dtype=np.float32)
    search_stats: dict = {}

    with patch.object(trainer_stub, "_is_kaiser_encounter_context", return_value=False), patch.object(
        trainer_stub, "_is_insatiable_encounter_context", return_value=False
    ), patch.object(
        trainer_stub, "_combat_energy", return_value=0.0
    ), patch.object(
        trainer_stub, "_semantic_family", side_effect=lambda action: action.get("kind") or "unknown"
    ), patch.object(
        trainer_stub, "_is_action_confirmed_lethal", return_value=False
    ):
        new_idx = trainer_stub._apply_combat_action_hard_guards(
            action_idx=0,
            legal_actions=legal_actions,
            action_mask=mask,
            raw_obs=raw_obs,
            boss_ctx={},
            encounter="ENCOUNTER.HAUNTED_SHIP_NORMAL",
            search_stats=search_stats,
        )

    assert new_idx == 1
    assert search_stats["combat_quality_potion_bad_guard_available"] == 1.0
    assert search_stats["combat_quality_potion_bad_guard_applied"] == 1.0
    assert search_stats["combat_quality_potion_bad_guard_override"] == 1.0
    assert search_stats["combat_quality_potion_bad_guard_forced_end_turn_hopeless"] == 1.0


def test_potion_slot_parser_uses_bridge_slot_not_player_index():
    """Bridge potion ids encode player first, then slot: kind:player:slot."""

    assert potion_slot_from_action_for_guard({"action_id": "discard_potion:0:1", "kind": "discard_potion"}) == 1
    assert potion_slot_from_action_for_guard({"action_id": "use_potion:0:2:self", "kind": "use_potion"}) == 2
    assert potion_slot_from_action_for_guard({"action_id": "discard_potion:1", "kind": "discard_potion"}) == 1


def test_potion_discard_guard_keeps_fortifier_discards_strength(trainer_stub):
    """Forced potion overflow should discard the low-value potion first.

    Observed failure: policy discarded Fortifier before late Act1 hallway
    deaths.  The guard should preserve Fortifier and throw away Strength when
    both discard choices are legal.
    """
    raw_obs = {
        "encounter": "ENCOUNTER.HAUNTED_SHIP_NORMAL",
        "player": {
            "current_hp": 35,
            "max_hp": 91,
            "potions": [
                {"id": "POTION.FORTIFIER", "title": "固化药水", "description": "将你的格挡变为三倍。"},
                {"id": "POTION.STRENGTH_POTION", "title": "力量药水", "description": "获得2点力量。"},
            ],
        },
        "combat": {"energy": 0},
    }
    legal_actions = [
        {"action_id": "discard_potion:0", "kind": "discard_potion"},
        {"action_id": "discard_potion:1", "kind": "discard_potion"},
    ]
    mask = np.array([1, 1], dtype=np.float32)
    search_stats: dict = {}

    with patch.object(trainer_stub, "_is_kaiser_encounter_context", return_value=False), patch.object(
        trainer_stub, "_is_insatiable_encounter_context", return_value=False
    ), patch.object(
        trainer_stub, "_semantic_family", side_effect=lambda action: action.get("kind") or "unknown"
    ), patch.object(
        trainer_stub, "_is_action_confirmed_lethal", return_value=False
    ):
        new_idx = trainer_stub._apply_combat_action_hard_guards(
            action_idx=0,
            legal_actions=legal_actions,
            action_mask=mask,
            raw_obs=raw_obs,
            boss_ctx={},
            encounter="ENCOUNTER.HAUNTED_SHIP_NORMAL",
            search_stats=search_stats,
        )

    assert new_idx == 1
    assert search_stats["combat_quality_potion_discard_guard_available"] == 1.0
    assert search_stats["combat_quality_potion_discard_guard_applied"] == 1.0
    assert search_stats["combat_quality_potion_discard_guard_override"] == 1.0
    assert search_stats["combat_quality_potion_discard_guard_candidate_count"] == 2.0
    assert search_stats["combat_quality_potion_discard_guard_saved_survival"] == 1.0
    assert search_stats["combat_quality_hard_guard_override_any"] == 1.0


def test_potion_discard_guard_keeps_lucky_discards_skill_slot_only_bridge_id(trainer_stub):
    """Regression: live bridge discard_potion:{player}:{slot} must not drop Lucky.

    If the parser reads ``discard_potion:0:1`` as slot 0, both discard choices
    appear to be the Lucky Tonic in raw_obs and the priority guard cannot
    override.  The correct discard is the lower-value Skill Potion in slot 1.
    """

    raw_obs = {
        "encounter": "ENCOUNTER.TWO_TAILED_RATS_NORMAL",
        "player": {
            "current_hp": 35,
            "max_hp": 80,
            "potions": [
                {"slot": 0, "id": "POTION.LUCKY_TONIC", "title": "幸运补剂"},
                {"slot": 1, "id": "POTION.SKILL_POTION", "title": "技能药水"},
            ],
        },
        "combat": {"energy": 0},
    }
    legal_actions = [
        {"action_id": "discard_potion:0:0", "kind": "discard_potion"},
        {"action_id": "discard_potion:0:1", "kind": "discard_potion"},
    ]
    mask = np.array([1, 1], dtype=np.float32)
    search_stats: dict = {}

    with patch.object(trainer_stub, "_is_kaiser_encounter_context", return_value=False), patch.object(
        trainer_stub, "_is_insatiable_encounter_context", return_value=False
    ), patch.object(
        trainer_stub, "_semantic_family", side_effect=lambda action: action.get("kind") or "unknown"
    ), patch.object(
        trainer_stub, "_is_action_confirmed_lethal", return_value=False
    ):
        new_idx = trainer_stub._apply_combat_action_hard_guards(
            action_idx=0,
            legal_actions=legal_actions,
            action_mask=mask,
            raw_obs=raw_obs,
            boss_ctx={},
            encounter="ENCOUNTER.TWO_TAILED_RATS_NORMAL",
            search_stats=search_stats,
        )

    assert new_idx == 1
    assert search_stats["combat_quality_potion_discard_guard_applied"] == 1.0
    assert search_stats["combat_quality_potion_discard_guard_saved_survival"] == 1.0


def test_potion_discard_guard_keeps_lucky_title_only_zh(trainer_stub):
    """Lucky/幸运 title-only raw potion still has survival keep priority."""

    raw_obs = {
        "encounter": "ENCOUNTER.TWO_TAILED_RATS_NORMAL",
        "player": {
            "current_hp": 35,
            "max_hp": 80,
            "potions": [
                {"slot": 0, "title": "幸运药剂"},
                {"slot": 1, "id": "POTION.SKILL_POTION", "title": "技能药水"},
            ],
        },
        "combat": {"energy": 0},
    }
    legal_actions = [
        {"action_id": "discard_potion:0:0", "kind": "discard_potion"},
        {"action_id": "discard_potion:0:1", "kind": "discard_potion"},
    ]
    mask = np.array([1, 1], dtype=np.float32)
    search_stats: dict = {}

    with patch.object(trainer_stub, "_is_kaiser_encounter_context", return_value=False), patch.object(
        trainer_stub, "_is_insatiable_encounter_context", return_value=False
    ), patch.object(
        trainer_stub, "_semantic_family", side_effect=lambda action: action.get("kind") or "unknown"
    ), patch.object(
        trainer_stub, "_is_action_confirmed_lethal", return_value=False
    ):
        new_idx = trainer_stub._apply_combat_action_hard_guards(
            action_idx=0,
            legal_actions=legal_actions,
            action_mask=mask,
            raw_obs=raw_obs,
            boss_ctx={},
            encounter="ENCOUNTER.TWO_TAILED_RATS_NORMAL",
            search_stats=search_stats,
        )

    assert new_idx == 1
    assert search_stats["combat_quality_potion_discard_guard_applied"] == 1.0
    assert search_stats["combat_quality_potion_discard_guard_saved_survival"] == 1.0


def test_potion_discard_guard_keeps_blood_discards_strength(trainer_stub):
    """Blood Potion is a late-Act1 survival tool and should be protected."""
    raw_obs = {
        "encounter": "ENCOUNTER.CORPSE_SLUGS_NORMAL",
        "player": {
            "current_hp": 20,
            "max_hp": 91,
            "potions": [
                {"id": "POTION.BLOOD_POTION", "title": "鲜血药水", "description": "回复你最大生命值的20%。"},
                {"id": "POTION.STRENGTH_POTION", "title": "力量药水", "description": "获得2点力量。"},
            ],
        },
        "combat": {"energy": 0},
    }
    legal_actions = [
        {"action_id": "discard_potion:0", "kind": "discard_potion"},
        {"action_id": "discard_potion:1", "kind": "discard_potion"},
    ]
    mask = np.array([1, 1], dtype=np.float32)
    search_stats: dict = {}

    with patch.object(trainer_stub, "_is_kaiser_encounter_context", return_value=False), patch.object(
        trainer_stub, "_is_insatiable_encounter_context", return_value=False
    ), patch.object(
        trainer_stub, "_semantic_family", side_effect=lambda action: action.get("kind") or "unknown"
    ), patch.object(
        trainer_stub, "_is_action_confirmed_lethal", return_value=False
    ):
        new_idx = trainer_stub._apply_combat_action_hard_guards(
            action_idx=0,
            legal_actions=legal_actions,
            action_mask=mask,
            raw_obs=raw_obs,
            boss_ctx={},
            encounter="ENCOUNTER.CORPSE_SLUGS_NORMAL",
            search_stats=search_stats,
        )

    assert new_idx == 1
    assert search_stats["combat_quality_potion_discard_guard_applied"] == 1.0
    assert search_stats["combat_quality_potion_discard_guard_saved_survival"] == 1.0


def test_potion_discard_guard_does_not_override_equal_low_value(trainer_stub):
    """Equal/near-equal low-value discards remain policy-controlled."""
    raw_obs = {
        "encounter": "ENCOUNTER.SEAPUNK_WEAK",
        "player": {
            "current_hp": 70,
            "max_hp": 91,
            "potions": [
                {"id": "POTION.STRENGTH_POTION", "title": "力量药水"},
                {"id": "POTION.SPEED_POTION", "title": "速度药水"},
            ],
        },
        "combat": {"energy": 0},
    }
    legal_actions = [
        {"action_id": "discard_potion:0", "kind": "discard_potion"},
        {"action_id": "discard_potion:1", "kind": "discard_potion"},
    ]
    mask = np.array([1, 1], dtype=np.float32)
    search_stats: dict = {}

    with patch.object(trainer_stub, "_is_kaiser_encounter_context", return_value=False), patch.object(
        trainer_stub, "_is_insatiable_encounter_context", return_value=False
    ), patch.object(
        trainer_stub, "_semantic_family", side_effect=lambda action: action.get("kind") or "unknown"
    ), patch.object(
        trainer_stub, "_is_action_confirmed_lethal", return_value=False
    ):
        new_idx = trainer_stub._apply_combat_action_hard_guards(
            action_idx=0,
            legal_actions=legal_actions,
            action_mask=mask,
            raw_obs=raw_obs,
            boss_ctx={},
            encounter="ENCOUNTER.SEAPUNK_WEAK",
            search_stats=search_stats,
        )

    assert new_idx == 0
    assert search_stats["combat_quality_potion_discard_guard_available"] == 1.0
    assert search_stats["combat_quality_potion_discard_guard_applied"] == 0.0


def test_potion_bad_use_guard_keeps_urgent_potion(trainer_stub):
    """Urgent/lethal/prevent-lethal/mechanism potions are not blocked."""
    raw_obs = {"encounter": "ENCOUNTER.ELITE", "combat": {"energy": 0}, "player": {"current_hp": 5, "max_hp": 80}}
    legal_actions = [
        {"action_id": "use_potion:0", "kind": "use_potion", "potion": {"id": "POTION.BLOCK"}},
        {"action_id": "end_turn", "kind": "end_turn"},
    ]
    mask = np.array([1, 1], dtype=np.float32)
    search_stats: dict = {}

    with patch.object(trainer_stub, "_is_kaiser_encounter_context", return_value=False), patch.object(
        trainer_stub, "_is_insatiable_encounter_context", return_value=False
    ), patch.object(
        trainer_stub, "_combat_energy", return_value=0.0
    ), patch.object(
        trainer_stub, "_semantic_family", side_effect=lambda action: action.get("kind") or "unknown"
    ), patch.object(
        trainer_stub,
        "_potion_timing_profile",
        return_value={
            "low_urgency": False,
            "save_recommended": False,
            "no_followup": False,
            "block_waste": False,
            "overkill": False,
            "urgent": True,
            "lethal": False,
            "prevent_lethal": True,
            "mechanism_answer": False,
        },
    ):
        new_idx = trainer_stub._apply_combat_action_hard_guards(
            action_idx=0,
            legal_actions=legal_actions,
            action_mask=mask,
            raw_obs=raw_obs,
            boss_ctx={},
            encounter="ENCOUNTER.ELITE",
            search_stats=search_stats,
        )

    assert new_idx == 0
    assert search_stats["combat_quality_potion_bad_guard_available"] == 0.0
    assert search_stats["combat_quality_potion_bad_guard_applied"] == 0.0


def test_potion_timing_profile_resolves_title_only_blood_potion_fractional_heal(trainer_stub):
    """Bridge may omit potion id and expose only localized title.

    Blood Potion's curated profile uses heal=0.20 to mean 20% max HP.  The
    timing profile must resolve "鲜血药水" by title and convert the fraction
    to an absolute heal amount; otherwise the guard mislabels it as a
    low-quality potion under incoming boss damage.
    """
    raw_obs = {
        "encounter": "ENCOUNTER.LAGAVULIN_MATRIARCH_BOSS",
        "combat": {
            "energy": 0,
            "enemies": [{"hp": 200, "intent": {"total_damage": 12}}],
        },
        "player": {"current_hp": 24, "max_hp": 91, "block": 0},
    }
    action = {
        "action_id": "use_potion:0:0:self",
        "kind": "use_potion",
        "potion": {"title": "鲜血药水"},
    }
    profile = trainer_stub._potion_timing_profile(
        action,
        0,
        None,
        raw_obs,
        [action, {"action_id": "end_turn", "kind": "end_turn"}],
        np.array([1, 1], dtype=np.float32),
        0.0,
    )

    assert profile["potion_id"] == "POTION.BLOOD_POTION"
    assert profile["heal_fraction_of_max_hp"] is True
    assert profile["heal"] == pytest.approx(18.2)
    assert profile["prevent_major_loss"] is True
    assert profile["urgent"] is True
    assert profile["positive"] is True
    assert profile["low_urgency"] is False


def test_potion_timing_profile_marks_strength_requires_followup_no_followup(trainer_stub):
    raw_obs = {
        "encounter": "ENCOUNTER.CORPSE_SLUGS_NORMAL",
        "combat": {
            "energy": 0,
            "enemies": [{"hp": 40, "intent": {"total_damage": 14}}],
        },
        "player": {"hp": 78, "current_hp": 78, "max_hp": 91, "block": 0},
    }
    action = {
        "action_id": "use_potion:0:self",
        "kind": "use_potion",
        "potion": {
            "id": "POTION.STRENGTH_POTION",
            "title": "力量药水",
            "effect_profile": {"strength": 2, "requires_followup": True},
            "timing_tags": ["requires_followup"],
        },
    }

    with patch.object(trainer_stub, "_is_kaiser_facing_change_action", return_value=False):
        profile = trainer_stub._potion_timing_profile(
            action,
            0,
            None,
            raw_obs,
            [action, {"action_id": "end_turn", "kind": "end_turn"}],
            np.array([1, 1], dtype=np.float32),
            0.0,
        )

    assert profile["requires_followup"] is True
    assert profile["followup_available"] is False
    assert profile["no_followup"] is True
    assert profile["low_urgency"] is True
    assert profile["urgent"] is False


def test_potion_bad_use_guard_skips_liquid_memories_at_critical_hp_boss(trainer_stub):
    """Liquid Memories exposes its real payoff only after use.

    Regression: in the Act1 boss death slice the root picked Liquid Memories,
    but the bad-use guard saw no current legal follow-up at 0 energy and
    rewrote it to End Turn.  Critical elite/boss survival/retrieve potions must
    fail open instead of being converted into a deterministic death.
    """
    raw_obs = {
        "encounter": "ENCOUNTER.LAGAVULIN_MATRIARCH_BOSS",
        "combat": {"energy": 0, "enemies": [{"intent": {"total_damage": 21}}]},
        "player": {"hp": 4, "current_hp": 4, "max_hp": 91, "block": 0},
    }
    legal_actions = [
        {
            "action_id": "use_potion:0:0:self",
            "kind": "use_potion",
            "potion": {"id": "POTION.LIQUID_MEMORIES", "title": "液态记忆"},
        },
        {"action_id": "end_turn", "kind": "end_turn", "label": "End Turn"},
    ]
    mask = np.array([1, 1], dtype=np.float32)
    search_stats: dict = {}

    with patch.object(trainer_stub, "_is_kaiser_encounter_context", return_value=False), patch.object(
        trainer_stub, "_is_insatiable_encounter_context", return_value=False
    ), patch.object(
        trainer_stub, "_combat_energy", return_value=0.0
    ), patch.object(
        trainer_stub, "_combat_encounter_tier_from_raw", return_value="boss"
    ), patch.object(
        trainer_stub, "_semantic_family", side_effect=lambda action: action.get("kind") or "unknown"
    ), patch.object(
        trainer_stub,
        "_potion_timing_profile",
        return_value={
            "low_urgency": True,
            "save_recommended": False,
            "no_followup": True,
            "block_waste": False,
            "overkill": False,
            "urgent": False,
            "lethal": False,
            "prevent_lethal": False,
            "mechanism_answer": False,
            "hp_valid": True,
            "hp_ratio": 4 / 91,
            "threat_gap": 21.0,
            "resource_like": True,
            "potion_id": "POTION.LIQUID_MEMORIES",
            "effect_family": ["discard_pile", "tutor"],
            "semantic_tags": ["resource", "tutor"],
            "timing_tags": [],
            "training_tags": [],
            "retrieve_from_discard": 1.0,
            "retrieve_from_discard_like": True,
            "resource_survival_tool": True,
            "critical_hp_survival_tool": True,
            "heal": 0.0,
            "block": 0.0,
        },
    ), patch.object(
        trainer_stub, "_is_action_confirmed_lethal", return_value=False
    ):
        new_idx = trainer_stub._apply_combat_action_hard_guards(
            action_idx=0,
            legal_actions=legal_actions,
            action_mask=mask,
            raw_obs=raw_obs,
            boss_ctx={},
            encounter="ENCOUNTER.LAGAVULIN_MATRIARCH_BOSS",
            search_stats=search_stats,
        )

    assert new_idx == 0
    assert search_stats["combat_quality_potion_bad_guard_critical_hp_survival_skip"] == 1.0
    assert search_stats["combat_quality_potion_bad_guard_applied"] == 0.0
    assert search_stats["combat_quality_potion_bad_guard_override"] == 0.0


def test_boss_survival_guard_overrides_end_turn_to_liquid_memories(trainer_stub):
    """At critical boss HP, End Turn + legal Liquid Memories should become
    Liquid Memories.  This covers the exact observed action set
    {Liquid Memories, End Turn} after the hand spent its energy.
    """
    raw_obs = {
        "encounter": "ENCOUNTER.LAGAVULIN_MATRIARCH_BOSS",
        "combat": {
            "energy": 0,
            "discard_pile": [{"id": "CARD.STRIKE", "title": "Strike"}],
            "enemies": [{"current_hp": 4, "intent": {"total_damage": 21}}],
        },
        "player": {"hp": 4, "current_hp": 4, "max_hp": 91, "block": 0},
    }
    legal_actions = [
        {
            "action_id": "use_potion:0:0:self",
            "kind": "use_potion",
            "potion": {"id": "POTION.LIQUID_MEMORIES", "title": "液态记忆"},
        },
        {"action_id": "end_turn", "kind": "end_turn", "label": "End Turn"},
    ]
    mask = np.array([1, 1], dtype=np.float32)
    search_stats: dict = {}

    with patch.object(trainer_stub, "_is_kaiser_encounter_context", return_value=False), patch.object(
        trainer_stub, "_is_insatiable_encounter_context", return_value=False
    ), patch.object(
        trainer_stub, "_combat_energy", return_value=0.0
    ), patch.object(
        trainer_stub, "_semantic_family", side_effect=lambda action: action.get("kind") or "unknown"
    ), patch.object(
        trainer_stub, "_is_action_confirmed_lethal", return_value=False
    ):
        new_idx = trainer_stub._apply_combat_action_hard_guards(
            action_idx=1,
            legal_actions=legal_actions,
            action_mask=mask,
            raw_obs=raw_obs,
            boss_ctx={},
            encounter="ENCOUNTER.LAGAVULIN_MATRIARCH_BOSS",
            search_stats=search_stats,
        )

    assert new_idx == 0
    assert search_stats["combat_quality_boss_survival_potion_guard_available"] == 1.0
    assert search_stats["combat_quality_boss_survival_potion_guard_applied"] == 1.0
    assert search_stats["combat_quality_boss_survival_potion_guard_override"] == 1.0
    assert search_stats["combat_quality_wasteful_end_turn_selected"] == 0.0


def test_potion_timing_profile_blood_potion_low_hp_idle_boss_is_urgent(trainer_stub):
    """Blood Potion should be urgent at critical elite/boss HP even on an
    idle turn: it buys survival margin for the next boss cycle.
    """
    raw_obs = {
        "encounter": "ENCOUNTER.LAGAVULIN_MATRIARCH_BOSS",
        "combat": {
            "energy": 0,
            "enemies": [{"hp": 160, "intent": {"total_damage": 0}}],
        },
        "player": {"current_hp": 15, "max_hp": 91, "block": 0},
    }
    action = {
        "action_id": "use_potion:0:0:self",
        "kind": "use_potion",
        "potion": {"id": "POTION.BLOOD_POTION", "title": "鲜血药水"},
    }
    profile = trainer_stub._potion_timing_profile(
        action,
        0,
        None,
        raw_obs,
        [action, {"action_id": "end_turn", "kind": "end_turn"}],
        np.array([1, 1], dtype=np.float32),
        0.0,
    )

    assert profile["heal"] == pytest.approx(18.2)
    assert profile["critical_hp_survival_tool"] is True
    assert profile["prevent_major_loss"] is True
    assert profile["urgent"] is True
    assert profile["low_urgency"] is False


def test_potion_bad_use_guard_can_still_block_non_survival_strength_potion(trainer_stub):
    """The critical survival fail-open is not a blanket potion exemption.

    A non-survival potion such as Strength at low boss HP can still be
    redirected to a concrete non-potion alternative; only End Turn fallback is
    restricted in dangerous windows.
    """
    raw_obs = {
        "encounter": "ENCOUNTER.LAGAVULIN_MATRIARCH_BOSS",
        "combat": {"energy": 1, "enemies": [{"intent": {"total_damage": 0}}]},
        "player": {"hp": 4, "current_hp": 4, "max_hp": 91, "block": 0},
    }
    legal_actions = [
        {"action_id": "use_potion:0", "kind": "use_potion", "potion": {"id": "POTION.STRENGTH", "title": "力量药水"}},
        {"action_id": "play_card_strike", "kind": "play_card", "card": {"id": "CARD.STRIKE"}},
        {"action_id": "end_turn", "kind": "end_turn", "label": "End Turn"},
    ]
    mask = np.array([1, 1, 1], dtype=np.float32)
    search_stats: dict = {}

    with patch.object(trainer_stub, "_is_kaiser_encounter_context", return_value=False), patch.object(
        trainer_stub, "_is_insatiable_encounter_context", return_value=False
    ), patch.object(
        trainer_stub, "_combat_energy", return_value=1.0
    ), patch.object(
        trainer_stub, "_combat_encounter_tier_from_raw", return_value="boss"
    ), patch.object(
        trainer_stub, "_semantic_family", side_effect=lambda action: action.get("kind") or "unknown"
    ), patch.object(
        trainer_stub,
        "_potion_timing_profile",
        return_value={
            "low_urgency": True,
            "save_recommended": True,
            "no_followup": False,
            "block_waste": False,
            "overkill": False,
            "urgent": False,
            "lethal": False,
            "prevent_lethal": False,
            "mechanism_answer": False,
            "hp_valid": True,
            "hp_ratio": 4 / 91,
            "threat_gap": 0.0,
            "resource_like": False,
            "potion_id": "POTION.STRENGTH",
            "effect_family": ["buff"],
            "semantic_tags": ["strength"],
            "timing_tags": [],
            "training_tags": [],
            "retrieve_from_discard": 0.0,
            "retrieve_from_discard_like": False,
            "resource_survival_tool": False,
            "critical_hp_survival_tool": False,
            "heal": 0.0,
            "block": 0.0,
        },
    ), patch.object(
        trainer_stub, "_is_action_confirmed_lethal", return_value=False
    ):
        new_idx = trainer_stub._apply_combat_action_hard_guards(
            action_idx=0,
            legal_actions=legal_actions,
            action_mask=mask,
            raw_obs=raw_obs,
            boss_ctx={},
            encounter="ENCOUNTER.LAGAVULIN_MATRIARCH_BOSS",
            search_stats=search_stats,
        )

    assert new_idx == 1
    assert search_stats["combat_quality_potion_bad_guard_critical_hp_survival_skip"] == 0.0
    assert search_stats["combat_quality_potion_bad_guard_applied"] == 1.0
    assert search_stats["combat_quality_potion_selected"] == 0.0


def test_potion_timing_profile_resolves_liquid_memories_from_raw_obs_slot(trainer_stub):
    """Bridge can expose use_potion actions with potion_id=None.

    The actual identity is still in raw_obs.player.potions[slot].  Liquid
    Memories must be resolved there; otherwise it is mis-scored as a generic
    no-followup potion and the bad-use guard may replace it with End Turn.
    """
    raw_obs = {
        "encounter": "ENCOUNTER.LAGAVULIN_MATRIARCH_BOSS",
        "combat": {
            "energy": 0,
            "discard_pile": [{"id": "CARD.STRIKE", "title": "Strike"}],
            "enemies": [{"current_hp": 160, "intent": {"total_damage": 12}}],
        },
        "player": {
            "hp": 40,
            "current_hp": 40,
            "max_hp": 91,
            "block": 0,
            "potions": [{"id": "POTION.LIQUID_MEMORIES", "title": "液态记忆"}],
        },
    }
    action = {
        "action_id": "use_potion:0:0:self",
        "kind": "use_potion",
        "potion_id": None,
        "potion_title": None,
    }
    profile = trainer_stub._potion_timing_profile(
        action,
        0,
        None,
        raw_obs,
        [action, {"action_id": "end_turn", "kind": "end_turn"}],
        np.array([1, 1], dtype=np.float32),
        0.0,
    )

    assert profile["potion_id"] == "POTION.LIQUID_MEMORIES"
    assert profile["retrieve_from_discard_like"] is True
    assert profile["discard_count"] == 1
    assert profile["retrieve_has_target"] is True
    assert profile["free_play_like"] is True
    assert profile["resource_like"] is True
    assert profile["followup_available"] is True
    assert profile["no_followup"] is False


def test_potion_timing_profile_liquid_memories_empty_discard_has_no_followup(trainer_stub):
    """Liquid Memories is not a boss/elite follow-up if discard is empty.

    Regression: boss turn 1 often exposes {Liquid Memories, Fortifier,
    End Turn} after all cards are spent.  The previous profile treated Liquid
    Memories as having a hidden follow-up purely because the encounter was a
    boss, even when the discard pile had no selectable card.
    """
    raw_obs = {
        "encounter": "ENCOUNTER.LAGAVULIN_MATRIARCH_BOSS",
        "combat": {
            "energy": 0,
            "discard_pile": [],
            "enemies": [{"current_hp": 160, "intent": {"total_damage": 0}}],
        },
        "player": {
            "hp": 78,
            "current_hp": 78,
            "max_hp": 91,
            "block": 0,
            "potions": [{"id": "POTION.LIQUID_MEMORIES", "title": "液态记忆"}],
        },
    }
    action = {
        "action_id": "use_potion:0:0:self",
        "kind": "use_potion",
        "potion_id": None,
        "potion_title": None,
    }
    profile = trainer_stub._potion_timing_profile(
        action,
        0,
        None,
        raw_obs,
        [action, {"action_id": "end_turn", "kind": "end_turn"}],
        np.array([1, 1], dtype=np.float32),
        0.0,
    )

    assert profile["potion_id"] == "POTION.LIQUID_MEMORIES"
    assert profile["retrieve_from_discard_like"] is True
    assert profile["discard_count"] == 0
    assert profile["retrieve_has_target"] is False
    assert profile["followup_available"] is False
    assert profile["resource_survival_tool"] is False
    assert profile["critical_hp_survival_tool"] is False
    assert profile["no_followup"] is True


def test_env_potion_timing_entropic_brew_near_death_is_urgent():
    """Pure env timing must spend stochastic resource potions near death.

    Regression from full-run death slice: 23 HP facing 22 incoming kept
    Entropic Brew/混沌药水 as low-urgency save value and died with it.
    """
    from sts2_env.potion_timing import compute_potion_timing

    raw_obs = {
        "encounter": "ENCOUNTER.NIBBITS_NORMAL",
        "combat": {
            "energy": 0,
            "enemies": [{"hp": 40, "current_hp": 40, "intent": {"total_damage": 22}}],
        },
        "player": {"hp": 23, "current_hp": 23, "max_hp": 80, "block": 0},
    }
    action = {
        "action_id": "use_potion:0:0:self",
        "kind": "use_potion",
        "potion": {"id": "POTION.ENTROPIC_BREW", "title": "混沌药水"},
    }
    profile = compute_potion_timing(
        action,
        raw_obs,
        [action, {"action_id": "end_turn", "kind": "end_turn"}],
        np.array([1, 1], dtype=np.float32),
        energy=0.0,
        encounter_tier="normal",
    )

    assert profile["near_death_after_incoming"] is True
    assert profile["random_potion_resource_like"] is True
    assert profile["new_option_resource_like"] is True
    assert profile["resource_survival_tool"] is True
    assert profile["prevent_major_loss"] is True
    assert profile["urgent"] is True
    assert profile["low_urgency"] is False
    assert profile["save_recommended"] is False
    assert profile["no_followup"] is False


def test_trainer_potion_timing_entropic_brew_near_death_is_urgent(trainer_stub):
    """Trainer-side timing must match the pure env near-death rule."""
    raw_obs = {
        "encounter": "ENCOUNTER.NIBBITS_NORMAL",
        "combat": {
            "energy": 0,
            "enemies": [{"hp": 40, "current_hp": 40, "intent": {"total_damage": 22}}],
        },
        "player": {"hp": 23, "current_hp": 23, "max_hp": 80, "block": 0},
    }
    action = {
        "action_id": "use_potion:0:0:self",
        "kind": "use_potion",
        "potion": {"id": "POTION.ENTROPIC_BREW", "title": "混沌药水"},
    }
    profile = trainer_stub._potion_timing_profile(
        action,
        0,
        None,
        raw_obs,
        [action, {"action_id": "end_turn", "kind": "end_turn"}],
        np.array([1, 1], dtype=np.float32),
        0.0,
    )

    assert profile["near_death_after_incoming"] is True
    assert profile["random_potion_resource_like"] is True
    assert profile["new_option_resource_like"] is True
    assert profile["resource_survival_tool"] is True
    assert profile["prevent_major_loss"] is True
    assert profile["urgent"] is True
    assert profile["low_urgency"] is False
    assert profile["save_recommended"] is False
    assert profile["no_followup"] is False


@pytest.mark.parametrize(
    ("potion_id", "title"),
    [
        ("POTION.ATTACK_POTION", "攻击药水"),
        ("POTION.SKILL_POTION", "技能药水"),
        ("POTION.POWER_POTION", "能力药水"),
        ("POTION.COLORLESS_POTION", "无色药水"),
    ],
)
def test_trainer_potion_timing_discovery_potions_near_death_are_urgent(
    trainer_stub,
    potion_id,
    title,
):
    """Card-discovery potions are valid death-margin rolls, not save-only."""
    raw_obs = {
        "encounter": "ENCOUNTER.TERROR_EEL_ELITE",
        "combat": {
            "energy": 0,
            "enemies": [{"hp": 70, "current_hp": 70, "intent": {"total_damage": 33}}],
        },
        "player": {"hp": 22, "current_hp": 22, "max_hp": 80, "block": 0},
    }
    action = {
        "action_id": "use_potion:0:0:self",
        "kind": "use_potion",
        "potion": {"id": potion_id, "title": title},
    }
    profile = trainer_stub._potion_timing_profile(
        action,
        0,
        None,
        raw_obs,
        [action, {"action_id": "end_turn", "kind": "end_turn"}],
        np.array([1, 1], dtype=np.float32),
        0.0,
    )

    assert profile["near_death_after_incoming"] is True
    assert profile["new_option_resource_like"] is True
    assert profile["resource_survival_tool"] is True
    assert profile["prevent_major_loss"] is True
    assert profile["urgent"] is True
    assert profile["low_urgency"] is False
    assert profile["no_followup"] is False


def test_potion_bad_use_guard_forces_end_turn_for_boss_idle_empty_liquid_memories(trainer_stub):
    """At low boss HP but zero incoming, empty-discard Liquid Memories is waste.

    This is the observed Act1 boss pattern: the policy spent Liquid Memories
    at 25/91 HP, 0 incoming, 0 energy, and no discard target, then also burned
    Fortifier.  The critical-HP fail-open must not protect that idle waste.
    """
    raw_obs = {
        "encounter": "ENCOUNTER.LAGAVULIN_MATRIARCH_BOSS",
        "combat": {
            "energy": 0,
            "discard_pile": [],
            "enemies": [{"current_hp": 160, "intent": {"total_damage": 0}}],
        },
        "player": {
            "hp": 25,
            "current_hp": 25,
            "max_hp": 91,
            "block": 5,
            "potions": [{"id": "POTION.LIQUID_MEMORIES", "title": "液态记忆"}],
        },
    }
    legal_actions = [
        {
            "action_id": "use_potion:0:0:self",
            "kind": "use_potion",
            "potion_id": None,
            "potion_title": None,
        },
        {"action_id": "end_turn", "kind": "end_turn", "label": "End Turn"},
    ]
    mask = np.array([1, 1], dtype=np.float32)
    search_stats: dict = {}

    with patch.object(trainer_stub, "_is_kaiser_encounter_context", return_value=False), patch.object(
        trainer_stub, "_is_insatiable_encounter_context", return_value=False
    ), patch.object(
        trainer_stub, "_combat_energy", return_value=0.0
    ), patch.object(
        trainer_stub, "_semantic_family", side_effect=lambda action: action.get("kind") or "unknown"
    ), patch.object(
        trainer_stub, "_is_action_confirmed_lethal", return_value=False
    ):
        new_idx = trainer_stub._apply_combat_action_hard_guards(
            action_idx=0,
            legal_actions=legal_actions,
            action_mask=mask,
            raw_obs=raw_obs,
            boss_ctx={},
            encounter="ENCOUNTER.LAGAVULIN_MATRIARCH_BOSS",
            search_stats=search_stats,
    )

    assert new_idx == 1
    assert search_stats["combat_quality_potion_bad_guard_critical_hp_survival_skip"] == 0.0
    assert search_stats["combat_quality_potion_bad_guard_applied"] == 1.0
    assert search_stats["combat_quality_potion_bad_guard_override"] == 1.0
    assert search_stats["combat_quality_potion_selected"] == 0.0


def test_potion_bad_use_guard_forces_end_turn_for_boss_idle_fortifier_waste(trainer_stub):
    """Fortifier with existing block is still waste when there is no threat."""
    raw_obs = {
        "encounter": "ENCOUNTER.LAGAVULIN_MATRIARCH_BOSS",
        "combat": {
            "energy": 0,
            "enemies": [{"current_hp": 160, "intent": {"total_damage": 0}}],
        },
        "player": {
            "hp": 25,
            "current_hp": 25,
            "max_hp": 91,
            "block": 5,
            "potions": [{"id": "POTION.FORTIFIER", "title": "固化药水", "description": "将你的格挡变为三倍。"}],
        },
    }
    legal_actions = [
        {
            "action_id": "use_potion:0:0:self",
            "kind": "use_potion",
            "potion_id": None,
            "potion_title": None,
        },
        {"action_id": "end_turn", "kind": "end_turn", "label": "End Turn"},
    ]
    mask = np.array([1, 1], dtype=np.float32)
    search_stats: dict = {}

    with patch.object(trainer_stub, "_is_kaiser_encounter_context", return_value=False), patch.object(
        trainer_stub, "_is_insatiable_encounter_context", return_value=False
    ), patch.object(
        trainer_stub, "_combat_energy", return_value=0.0
    ), patch.object(
        trainer_stub, "_semantic_family", side_effect=lambda action: action.get("kind") or "unknown"
    ), patch.object(
        trainer_stub, "_is_action_confirmed_lethal", return_value=False
    ):
        new_idx = trainer_stub._apply_combat_action_hard_guards(
            action_idx=0,
            legal_actions=legal_actions,
            action_mask=mask,
            raw_obs=raw_obs,
            boss_ctx={},
            encounter="ENCOUNTER.LAGAVULIN_MATRIARCH_BOSS",
            search_stats=search_stats,
        )

    assert new_idx == 1
    assert search_stats["combat_quality_potion_bad_guard_critical_hp_idle_waste_not_skipped"] == 1.0
    assert search_stats["combat_quality_potion_bad_guard_applied"] == 1.0
    assert search_stats["combat_quality_potion_bad_guard_override"] == 1.0
    assert search_stats["combat_quality_potion_selected"] == 0.0
