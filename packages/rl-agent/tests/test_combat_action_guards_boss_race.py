"""Boss race/setup potion and survival-block guard regressions.

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

from tests.combat_action_guard_cases import _boss_race_strength_profile  # noqa: E402


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



def test_boss_race_potion_guard_overrides_end_turn_to_strength_under_pressure(trainer_stub):
    """Boss race guard should spend setup/race potions instead of 0-energy End Turn.

    This is the Act1 failure seen in the latest run: the agent reaches the boss,
    has a Strength-style potion legal, then ends the turn and dies before the
    sparse long-horizon value can learn the timing.
    """
    raw_obs = {
        "encounter": "ENCOUNTER.LAGAVULIN_MATRIARCH_BOSS",
        "run": {"room_type": "boss", "floor": 17},
        "combat": {"energy": 0, "enemies": [{"current_hp": 200, "intent": {"total_damage": 19}}]},
        "player": {"hp": 70, "current_hp": 70, "max_hp": 80, "block": 0},
    }
    legal_actions = [
        {"action_id": "end_turn", "kind": "end_turn"},
        {
            "action_id": "use_potion:0",
            "kind": "use_potion",
            "potion": {"id": "POTION.STRENGTH_POTION", "title": "Strength Potion"},
        },
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
        trainer_stub, "_potion_timing_profile", return_value=_boss_race_strength_profile()
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
    assert search_stats["combat_quality_boss_race_potion_guard_available"] == 1.0
    assert search_stats["combat_quality_boss_race_potion_guard_applied"] == 1.0
    assert search_stats["combat_quality_boss_race_potion_guard_override"] == 1.0
    # The generic potion-bad guard must not undo the boss-race override.
    assert search_stats["combat_quality_potion_bad_guard_boss_race_skip"] == 1.0
    assert search_stats["combat_quality_potion_bad_guard_applied"] == 0.0
    assert search_stats["combat_quality_end_turn_selected"] == 0.0
    assert search_stats["combat_quality_potion_selected"] == 1.0


def test_boss_setup_potion_guard_overrides_idle_end_turn_to_strength(trainer_stub):
    """0-energy boss setup turns with no incoming still should use scaling."""
    raw_obs = {
        "encounter": "ENCOUNTER.LAGAVULIN_MATRIARCH_BOSS",
        "run": {"room_type": "boss", "floor": 17},
        "combat": {"energy": 0, "enemies": [{"current_hp": 200, "intent": {"total_damage": 0}}]},
        "player": {"hp": 70, "current_hp": 70, "max_hp": 80, "block": 0},
    }
    legal_actions = [
        {"action_id": "end_turn", "kind": "end_turn"},
        {
            "action_id": "use_potion:0",
            "kind": "use_potion",
            "potion": {"id": "POTION.STRENGTH_POTION", "title": "力量药水"},
        },
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
        return_value=_boss_race_strength_profile(threat_gap=0.0, low_urgency=True, save_recommended=True),
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
    assert search_stats["combat_quality_boss_race_potion_guard_available"] == 1.0
    assert search_stats["combat_quality_boss_race_potion_guard_applied"] == 1.0
    assert search_stats["combat_quality_potion_bad_guard_boss_race_skip"] == 1.0
    assert search_stats["combat_quality_potion_bad_guard_applied"] == 0.0


def test_boss_race_potion_guard_rejects_fortifier_zero_block_noop(trainer_stub):
    """Fortifier at zero block is a no-op and must not be treated as boss race."""
    raw_obs = {
        "encounter": "ENCOUNTER.LAGAVULIN_MATRIARCH_BOSS",
        "run": {"room_type": "boss", "floor": 17},
        "combat": {"energy": 0, "enemies": [{"current_hp": 200, "intent": {"total_damage": 19}}]},
        "player": {"hp": 70, "current_hp": 70, "max_hp": 80, "block": 0},
    }
    legal_actions = [
        {"action_id": "end_turn", "kind": "end_turn"},
        {
            "action_id": "use_potion:0",
            "kind": "use_potion",
            "potion": {"id": "POTION.FORTIFIER", "title": "固化药水"},
        },
    ]
    mask = np.array([1, 1], dtype=np.float32)
    search_stats: dict = {}
    fortifier_profile = {
        "hp_valid": True,
        "hp": 70.0,
        "max_hp": 80.0,
        "hp_ratio": 70.0 / 80.0,
        "threat_gap": 19.0,
        "potion_id": "POTION.FORTIFIER",
        "effect_family": ["block", "amplify_block"],
        "semantic_tags": ["block"],
        "timing_tags": ["requires_block_in_play"],
        "training_tags": [],
        "block": 0.0,
        "damage": 0.0,
        "amplify_block_noop": True,
        "block_waste": True,
    }

    with patch.object(trainer_stub, "_is_kaiser_encounter_context", return_value=False), patch.object(
        trainer_stub, "_is_insatiable_encounter_context", return_value=False
    ), patch.object(
        trainer_stub, "_combat_energy", return_value=0.0
    ), patch.object(
        trainer_stub, "_combat_encounter_tier_from_raw", return_value="boss"
    ), patch.object(
        trainer_stub, "_semantic_family", side_effect=lambda action: action.get("kind") or "unknown"
    ), patch.object(
        trainer_stub, "_potion_timing_profile", return_value=fortifier_profile
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
    assert search_stats["combat_quality_boss_race_potion_guard_available"] == 1.0
    assert search_stats["combat_quality_boss_race_potion_guard_no_alternative"] == 1.0
    assert search_stats["combat_quality_boss_race_potion_guard_applied"] == 0.0


def test_boss_race_potion_guard_rejects_empty_discard_liquid_memories(trainer_stub):
    """Liquid Memories without a discard target must not be forced as race setup."""
    raw_obs = {
        "encounter": "ENCOUNTER.LAGAVULIN_MATRIARCH_BOSS",
        "run": {"room_type": "boss", "floor": 17},
        "combat": {
            "energy": 0,
            "discard_pile": [],
            "enemies": [{"current_hp": 200, "intent": {"total_damage": 19}}],
        },
        "player": {"hp": 70, "current_hp": 70, "max_hp": 80, "block": 0},
    }
    legal_actions = [
        {"action_id": "end_turn", "kind": "end_turn"},
        {
            "action_id": "use_potion:0",
            "kind": "use_potion",
            "potion": {"id": "POTION.LIQUID_MEMORIES", "title": "液态记忆"},
        },
    ]
    mask = np.array([1, 1], dtype=np.float32)
    search_stats: dict = {}
    liquid_empty_profile = {
        "hp_valid": True,
        "hp": 70.0,
        "max_hp": 80.0,
        "hp_ratio": 70.0 / 80.0,
        "threat_gap": 19.0,
        "potion_id": "POTION.LIQUID_MEMORIES",
        "effect_family": ["discard_pile", "tutor", "set_cost_zero"],
        "semantic_tags": ["discard", "tutor"],
        "timing_tags": ["requires_discard_context"],
        "training_tags": [],
        "retrieve_from_discard": 1.0,
        "retrieve_from_discard_like": True,
        "retrieve_has_target": False,
        "block": 0.0,
        "damage": 0.0,
    }

    with patch.object(trainer_stub, "_is_kaiser_encounter_context", return_value=False), patch.object(
        trainer_stub, "_is_insatiable_encounter_context", return_value=False
    ), patch.object(
        trainer_stub, "_combat_energy", return_value=0.0
    ), patch.object(
        trainer_stub, "_combat_encounter_tier_from_raw", return_value="boss"
    ), patch.object(
        trainer_stub, "_semantic_family", side_effect=lambda action: action.get("kind") or "unknown"
    ), patch.object(
        trainer_stub, "_potion_timing_profile", return_value=liquid_empty_profile
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
    assert search_stats["combat_quality_boss_race_potion_guard_available"] == 1.0
    assert search_stats["combat_quality_boss_race_potion_guard_no_alternative"] == 1.0
    assert search_stats["combat_quality_boss_race_potion_guard_applied"] == 0.0


def test_boss_race_potion_guard_uses_lagavulin_setup_liquid_memories(trainer_stub):
    """Lagavulin asleep/vulnerable opening is a race window even at 0 incoming.

    Diagnostics showed legal actions reduced to End Turn + empty-looking Liquid
    Memories after spending the hand at 0 energy.  Generic empty-discard
    filtering must not force an End Turn through Lagavulin's opening/stun
    damage window.
    """
    raw_obs = {
        "encounter": "ENCOUNTER.LAGAVULIN_MATRIARCH_BOSS",
        "run": {"room_type": "boss", "floor": 17},
        "combat": {
            "energy": 0,
            "discard_pile": [],
            "enemies": [
                {
                    "id": "MONSTER.LAGAVULIN_MATRIARCH",
                    "model_id": "MONSTER.LAGAVULIN_MATRIARCH",
                    "current_hp": 222,
                    "intent": {"total_damage": 0, "intent": "Sleep"},
                    "powers": [
                        {"id": "PLATING_POWER", "amount": 12},
                        {"id": "ASLEEP_POWER", "amount": 3},
                        {"id": "VULNERABLE_POWER", "amount": 2},
                    ],
                }
            ],
        },
        "player": {"hp": 47, "current_hp": 47, "max_hp": 91, "block": 0},
    }
    legal_actions = [
        {"action_id": "end_turn", "kind": "end_turn"},
        {
            "action_id": "use_potion:0:0:self",
            "kind": "use_potion",
            "potion": {"id": "POTION.LIQUID_MEMORIES", "title": "液态记忆"},
        },
    ]
    mask = np.array([1, 1], dtype=np.float32)
    search_stats: dict = {}
    liquid_empty_profile = {
        "low_urgency": True,
        "save_recommended": False,
        "no_followup": True,
        "requires_followup": False,
        "resource_like": True,
        "block_waste": False,
        "overkill": False,
        "urgent": False,
        "lethal": False,
        "prevent_lethal": False,
        "mechanism_answer": False,
        "hp_valid": True,
        "hp": 47.0,
        "max_hp": 91.0,
        "hp_ratio": 47.0 / 91.0,
        "threat_gap": 0.0,
        "potion_id": "POTION.LIQUID_MEMORIES",
        "effect_family": ["discard_pile", "tutor", "set_cost_zero"],
        "semantic_tags": ["discard", "tutor", "resource"],
        "timing_tags": ["requires_discard_context"],
        "training_tags": [],
        "retrieve_from_discard": 1.0,
        "retrieve_from_discard_like": True,
        "retrieve_has_target": False,
        "block": 0.0,
        "damage": 0.0,
        "heal": 0.0,
        "use_quality": 0.0,
    }

    with patch.object(trainer_stub, "_is_kaiser_encounter_context", return_value=False), patch.object(
        trainer_stub, "_is_insatiable_encounter_context", return_value=False
    ), patch.object(
        trainer_stub, "_combat_energy", return_value=0.0
    ), patch.object(
        trainer_stub, "_combat_encounter_tier_from_raw", return_value="boss"
    ), patch.object(
        trainer_stub, "_semantic_family", side_effect=lambda action: action.get("kind") or "unknown"
    ), patch.object(
        trainer_stub, "_potion_timing_profile", return_value=liquid_empty_profile
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
    assert search_stats["combat_quality_boss_race_potion_guard_available"] == 1.0
    assert search_stats["combat_quality_boss_race_potion_guard_applied"] == 1.0
    assert search_stats["combat_quality_boss_race_potion_guard_lagavulin_setup_escape"] == 1.0
    assert search_stats["combat_quality_potion_bad_guard_applied"] == 0.0


def test_potion_bad_guard_keeps_lagavulin_setup_liquid_memories(trainer_stub):
    """The selected-potion bad-use guard must not undo the Lagavulin setup escape."""
    raw_obs = {
        "encounter": "ENCOUNTER.LAGAVULIN_MATRIARCH_BOSS",
        "run": {"room_type": "boss", "floor": 17},
        "combat": {
            "energy": 0,
            "discard_pile": [],
            "enemies": [
                {
                    "id": "MONSTER.LAGAVULIN_MATRIARCH",
                    "model_id": "MONSTER.LAGAVULIN_MATRIARCH",
                    "current_hp": 216,
                    "intent": {"total_damage": 0, "intent": "Stun"},
                    "powers": [{"id": "VULNERABLE_POWER", "amount": 2}],
                }
            ],
        },
        "player": {
            "hp": 47,
            "current_hp": 47,
            "max_hp": 91,
            "block": 0,
            "potions": [{"id": "POTION.LIQUID_MEMORIES", "title": "液态记忆"}],
        },
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
    liquid_empty_profile = {
        "low_urgency": True,
        "save_recommended": False,
        "no_followup": True,
        "requires_followup": False,
        "resource_like": True,
        "block_waste": False,
        "overkill": False,
        "urgent": False,
        "lethal": False,
        "prevent_lethal": False,
        "mechanism_answer": False,
        "hp_valid": True,
        "hp": 47.0,
        "max_hp": 91.0,
        "hp_ratio": 47.0 / 91.0,
        "threat_gap": 0.0,
        "potion_id": "POTION.LIQUID_MEMORIES",
        "effect_family": ["discard_pile", "tutor", "set_cost_zero"],
        "semantic_tags": ["discard", "tutor", "resource"],
        "timing_tags": ["requires_discard_context"],
        "training_tags": [],
        "retrieve_from_discard": 1.0,
        "retrieve_from_discard_like": True,
        "retrieve_has_target": False,
        "block": 0.0,
        "damage": 0.0,
        "heal": 0.0,
        "use_quality": 0.0,
    }

    with patch.object(trainer_stub, "_is_kaiser_encounter_context", return_value=False), patch.object(
        trainer_stub, "_is_insatiable_encounter_context", return_value=False
    ), patch.object(
        trainer_stub, "_combat_energy", return_value=0.0
    ), patch.object(
        trainer_stub, "_combat_encounter_tier_from_raw", return_value="boss"
    ), patch.object(
        trainer_stub, "_semantic_family", side_effect=lambda action: action.get("kind") or "unknown"
    ), patch.object(
        trainer_stub, "_potion_timing_profile", return_value=liquid_empty_profile
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
    assert search_stats["combat_quality_potion_bad_hopeless_guard_lagavulin_setup_skip"] == 1.0
    assert search_stats["combat_quality_potion_bad_guard_applied"] == 0.0


def test_potion_bad_guard_reads_live_lagavulin_intent_type_stun(trainer_stub):
    """Regression for the live bridge shape observed in v4 diagnostics.

    The bridge reports Lagavulin's no-action setup turn as
    ``intent: {"intent_type": "Stun", "title": "击晕"}`` and may omit
    ASLEEP_POWER/VULNERABLE_POWER from ``powers``.  The Lagavulin setup escape
    must read those fields; otherwise the generic empty-discard Liquid Memories
    guard rewrites the potion to End Turn and wastes the boss race window.
    """
    raw_obs = {
        "run": {"room_type": "Boss", "floor": 17, "room_model": "ENCOUNTER.LAGAVULIN_MATRIARCH_BOSS"},
        "combat": {
            "energy": 0,
            "discard_pile": [],
            "enemies": [
                {
                    "id": 1,
                    "combat_id": 1,
                    "name": "乐加维林族母",
                    "model_id": "MONSTER.LAGAVULIN_MATRIARCH",
                    "hp": 209,
                    "max_hp": 222,
                    "block": 0,
                    "intent": {
                        "intent_type": "Stun",
                        "title": "击晕",
                        "description": "这个敌人在其下一回合无法行动。",
                        "total_damage": None,
                        "repeats": 1,
                    },
                    "powers": [],
                }
            ],
        },
        "player": {
            "hp": 39,
            "current_hp": 39,
            "max_hp": 91,
            "block": 5,
            "potions": [{"id": "POTION.LIQUID_MEMORIES", "title": "液态记忆"}],
        },
    }
    legal_actions = [
        {
            "action_id": "use_potion:0:0:self",
            "kind": "use_potion",
            # Live action payloads can carry the localized title at top level
            # rather than under a nested potion object.
            "title": "液态记忆",
            "target": {"title": "铁甲战士"},
        },
        {"action_id": "end_turn", "kind": "end_turn", "label": "End Turn"},
    ]
    mask = np.array([1, 1], dtype=np.float32)
    search_stats: dict = {}
    liquid_empty_profile = {
        "low_urgency": True,
        "save_recommended": False,
        "no_followup": True,
        "requires_followup": False,
        "resource_like": True,
        "block_waste": False,
        "overkill": False,
        "urgent": False,
        "lethal": False,
        "prevent_lethal": False,
        "mechanism_answer": False,
        "hp_valid": True,
        "hp": 39.0,
        "max_hp": 91.0,
        "hp_ratio": 39.0 / 91.0,
        "threat_gap": 0.0,
        "potion_id": "POTION.LIQUID_MEMORIES",
        "effect_family": ["discard_pile", "tutor", "set_cost_zero"],
        "semantic_tags": ["discard", "tutor", "resource"],
        "timing_tags": ["requires_discard_context"],
        "training_tags": [],
        "retrieve_from_discard": 1.0,
        "retrieve_from_discard_like": True,
        "retrieve_has_target": False,
        "block": 0.0,
        "damage": 0.0,
        "heal": 0.0,
        "use_quality": 0.0,
    }

    with patch.object(trainer_stub, "_is_kaiser_encounter_context", return_value=False), patch.object(
        trainer_stub, "_is_insatiable_encounter_context", return_value=False
    ), patch.object(
        trainer_stub, "_combat_energy", return_value=0.0
    ), patch.object(
        trainer_stub, "_semantic_family", side_effect=lambda action: action.get("kind") or "unknown"
    ), patch.object(
        trainer_stub, "_potion_timing_profile", return_value=liquid_empty_profile
    ), patch.object(
        trainer_stub, "_is_action_confirmed_lethal", return_value=False
    ):
        new_idx = trainer_stub._apply_combat_action_hard_guards(
            action_idx=0,
            legal_actions=legal_actions,
            action_mask=mask,
            raw_obs=raw_obs,
            boss_ctx={},
            encounter="",
            search_stats=search_stats,
        )

    assert new_idx == 0
    assert search_stats["combat_quality_potion_bad_hopeless_guard_lagavulin_setup_skip"] == 1.0
    assert search_stats["combat_quality_potion_bad_guard_applied"] == 0.0


def test_boss_race_potion_guard_uses_liquid_memories_escape_under_low_hp_pressure(trainer_stub):
    """Bridge can under-report the discard pile during boss pressure windows.

    At 0 energy with no playable cards, low/mid HP, and visible incoming
    damage, empty-looking Liquid Memories is still a better Act1-boss survival
    target than End Turn.  High/full HP empty Liquid Memories remains rejected
    by the previous test.
    """
    raw_obs = {
        "encounter": "ENCOUNTER.LAGAVULIN_MATRIARCH_BOSS",
        "run": {"room_type": "boss", "floor": 17},
        "combat": {
            "energy": 0,
            "discard_pile": [],
            "enemies": [{"current_hp": 160, "intent": {"total_damage": 21}}],
        },
        "player": {"hp": 40, "current_hp": 40, "max_hp": 91, "block": 0},
    }
    legal_actions = [
        {"action_id": "end_turn", "kind": "end_turn"},
        {
            "action_id": "use_potion:0:0:self",
            "kind": "use_potion",
            "potion": {"id": "POTION.LIQUID_MEMORIES", "title": "液态记忆"},
        },
    ]
    mask = np.array([1, 1], dtype=np.float32)
    search_stats: dict = {}
    liquid_empty_profile = {
        "low_urgency": True,
        "save_recommended": False,
        "no_followup": True,
        "requires_followup": False,
        "resource_like": True,
        "block_waste": False,
        "overkill": False,
        "urgent": False,
        "lethal": False,
        "prevent_lethal": False,
        "mechanism_answer": False,
        "hp_valid": True,
        "hp": 40.0,
        "max_hp": 91.0,
        "hp_ratio": 40.0 / 91.0,
        "threat_gap": 21.0,
        "potion_id": "POTION.LIQUID_MEMORIES",
        "effect_family": ["discard_pile", "tutor", "set_cost_zero"],
        "semantic_tags": ["discard", "tutor", "resource"],
        "timing_tags": ["requires_discard_context"],
        "training_tags": [],
        "retrieve_from_discard": 1.0,
        "retrieve_from_discard_like": True,
        "retrieve_has_target": False,
        "block": 0.0,
        "damage": 0.0,
        "heal": 0.0,
        "use_quality": 0.0,
    }

    with patch.object(trainer_stub, "_is_kaiser_encounter_context", return_value=False), patch.object(
        trainer_stub, "_is_insatiable_encounter_context", return_value=False
    ), patch.object(
        trainer_stub, "_combat_energy", return_value=0.0
    ), patch.object(
        trainer_stub, "_combat_encounter_tier_from_raw", return_value="boss"
    ), patch.object(
        trainer_stub, "_semantic_family", side_effect=lambda action: action.get("kind") or "unknown"
    ), patch.object(
        trainer_stub, "_potion_timing_profile", return_value=liquid_empty_profile
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
    assert search_stats["combat_quality_boss_race_potion_guard_available"] == 1.0
    assert search_stats["combat_quality_boss_race_potion_guard_applied"] == 1.0
    assert search_stats["combat_quality_boss_race_potion_guard_override"] == 1.0
    assert search_stats["combat_quality_potion_bad_guard_applied"] == 0.0


def test_boss_race_potion_guard_uses_fortifier_with_existing_block_under_pressure(trainer_stub):
    """Fortifier is waste at 0 block, but concrete survival with existing block."""
    raw_obs = {
        "encounter": "ENCOUNTER.LAGAVULIN_MATRIARCH_BOSS",
        "run": {"room_type": "boss", "floor": 17},
        "combat": {"energy": 0, "enemies": [{"current_hp": 160, "intent": {"total_damage": 19}}]},
        "player": {"hp": 72, "current_hp": 72, "max_hp": 91, "block": 8},
    }
    legal_actions = [
        {"action_id": "end_turn", "kind": "end_turn"},
        {
            "action_id": "use_potion:0:0:self",
            "kind": "use_potion",
            "potion": {"id": "POTION.FORTIFIER", "title": "固化药水"},
        },
    ]
    mask = np.array([1, 1], dtype=np.float32)
    search_stats: dict = {}
    fortifier_profile = {
        "low_urgency": False,
        "save_recommended": False,
        "no_followup": False,
        "requires_followup": False,
        "resource_like": False,
        "block_waste": False,
        "overkill": False,
        "urgent": False,
        "lethal": False,
        "prevent_lethal": False,
        "mechanism_answer": False,
        "hp_valid": True,
        "hp": 72.0,
        "max_hp": 91.0,
        "hp_ratio": 72.0 / 91.0,
        "threat_gap": 11.0,
        "potion_id": "POTION.FORTIFIER",
        "effect_family": ["block", "amplify_block"],
        "semantic_tags": ["block"],
        "timing_tags": ["requires_block_in_play"],
        "training_tags": [],
        "retrieve_from_discard": 0.0,
        "retrieve_from_discard_like": False,
        "retrieve_has_target": False,
        "block": 16.0,
        "damage": 0.0,
        "heal": 0.0,
        "amplify_block_noop": False,
        "use_quality": 0.0,
    }

    with patch.object(trainer_stub, "_is_kaiser_encounter_context", return_value=False), patch.object(
        trainer_stub, "_is_insatiable_encounter_context", return_value=False
    ), patch.object(
        trainer_stub, "_combat_energy", return_value=0.0
    ), patch.object(
        trainer_stub, "_combat_encounter_tier_from_raw", return_value="boss"
    ), patch.object(
        trainer_stub, "_semantic_family", side_effect=lambda action: action.get("kind") or "unknown"
    ), patch.object(
        trainer_stub, "_potion_timing_profile", return_value=fortifier_profile
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
    assert search_stats["combat_quality_boss_race_potion_guard_available"] == 1.0
    assert search_stats["combat_quality_boss_race_potion_guard_applied"] == 1.0
    assert search_stats["combat_quality_boss_race_potion_guard_override"] == 1.0
    assert search_stats["combat_quality_potion_bad_guard_applied"] == 0.0


def test_potion_bad_use_guard_does_not_undo_boss_race_strength_potion(trainer_stub):
    """Generic bad-use guard must keep 0-energy boss race/setup potions."""
    raw_obs = {
        "encounter": "ENCOUNTER.LAGAVULIN_MATRIARCH_BOSS",
        "run": {"room_type": "boss", "floor": 17},
        "combat": {"energy": 0, "enemies": [{"current_hp": 200, "intent": {"total_damage": 19}}]},
        "player": {"hp": 70, "current_hp": 70, "max_hp": 80, "block": 0},
    }
    legal_actions = [
        {
            "action_id": "use_potion:0",
            "kind": "use_potion",
            "potion": {"id": "POTION.STRENGTH_POTION", "title": "Strength Potion"},
        },
        {"action_id": "end_turn", "kind": "end_turn"},
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
        trainer_stub, "_potion_timing_profile", return_value=_boss_race_strength_profile()
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
    assert search_stats["combat_quality_potion_bad_guard_boss_race_skip"] == 1.0
    assert search_stats["combat_quality_potion_bad_guard_applied"] == 0.0
    assert search_stats.get("combat_quality_potion_selected", 0.0) == 0.0
