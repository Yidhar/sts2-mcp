"""Floor-sensitive race-potion and fallback guard regressions.

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

def test_floor8_normal_race_potion_guard_overrides_end_turn_to_strength(trainer_stub):
    """Floor >= 8 high-risk normal hallways should spend race/setup potion at 0 energy."""
    raw_obs = {
        "encounter": "ENCOUNTER.CORPSE_SLUGS_NORMAL",
        "run": {"floor": 8},
        "combat": {"energy": 0, "enemies": [{"current_hp": 40, "intent": {"total_damage": 14}}]},
        "player": {"hp": 53, "current_hp": 53, "max_hp": 91, "block": 0},
    }
    legal_actions = [
        {"action_id": "end_turn", "kind": "end_turn"},
        {"action_id": "use_potion:0", "kind": "use_potion", "potion": {"id": "POTION.STRENGTH_POTION"}},
    ]
    mask = np.array([1, 1], dtype=np.float32)
    search_stats: dict = {}

    with patch.object(trainer_stub, "_is_kaiser_encounter_context", return_value=False), patch.object(
        trainer_stub, "_is_insatiable_encounter_context", return_value=False
    ), patch.object(
        trainer_stub, "_combat_encounter_tier_from_raw", return_value="normal"
    ), patch.object(
        trainer_stub, "_combat_energy", return_value=0.0
    ), patch.object(
        trainer_stub, "_semantic_family", side_effect=lambda action: action.get("kind") or "unknown"
    ), patch.object(
        trainer_stub,
        "_potion_timing_profile",
        return_value=_boss_race_strength_profile(hp=53.0, max_hp=91.0, hp_ratio=53.0 / 91.0, threat_gap=14.0),
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
    assert search_stats["combat_quality_late_normal_race_potion_guard_available"] == 1.0
    assert search_stats["combat_quality_late_normal_race_potion_guard_applied"] == 1.0


def test_floor8_weak_race_potion_guard_overrides_end_turn_to_strength(trainer_stub):
    """The floor-8 broadening also covers _WEAK hallway ids."""
    raw_obs = {
        "encounter": "ENCOUNTER.GREMLIN_MERC_WEAK",
        "run": {"floor": 8},
        "combat": {"energy": 0, "enemies": [{"current_hp": 32, "intent": {"total_damage": 14}}]},
        "player": {"hp": 53, "current_hp": 53, "max_hp": 91, "block": 0},
    }
    legal_actions = [
        {"action_id": "end_turn", "kind": "end_turn"},
        {"action_id": "use_potion:0", "kind": "use_potion", "potion": {"id": "POTION.STRENGTH_POTION"}},
    ]
    mask = np.array([1, 1], dtype=np.float32)
    search_stats: dict = {}

    with patch.object(trainer_stub, "_is_kaiser_encounter_context", return_value=False), patch.object(
        trainer_stub, "_is_insatiable_encounter_context", return_value=False
    ), patch.object(
        trainer_stub, "_combat_encounter_tier_from_raw", return_value="weak"
    ), patch.object(
        trainer_stub, "_combat_energy", return_value=0.0
    ), patch.object(
        trainer_stub, "_semantic_family", side_effect=lambda action: action.get("kind") or "unknown"
    ), patch.object(
        trainer_stub,
        "_potion_timing_profile",
        return_value=_boss_race_strength_profile(hp=53.0, max_hp=91.0, hp_ratio=53.0 / 91.0, threat_gap=14.0),
    ), patch.object(
        trainer_stub, "_is_action_confirmed_lethal", return_value=False
    ):
        new_idx = trainer_stub._apply_combat_action_hard_guards(
            action_idx=0,
            legal_actions=legal_actions,
            action_mask=mask,
            raw_obs=raw_obs,
            boss_ctx={},
            encounter="ENCOUNTER.GREMLIN_MERC_WEAK",
            search_stats=search_stats,
        )

    assert new_idx == 1
    assert search_stats["combat_quality_late_normal_race_potion_guard_available"] == 1.0
    assert search_stats["combat_quality_late_normal_race_potion_guard_applied"] == 1.0


def test_floor8_race_potion_guard_dormant_below_pressure_threshold(trainer_stub):
    """Floor 8 is not enough by itself; high HP + only 6 incoming stays dormant."""
    raw_obs = {
        "encounter": "ENCOUNTER.CORPSE_SLUGS_NORMAL",
        "run": {"floor": 8},
        "combat": {"energy": 0, "enemies": [{"current_hp": 40, "intent": {"total_damage": 6}}]},
        "player": {"hp": 70, "current_hp": 70, "max_hp": 91, "block": 0},
    }
    legal_actions = [
        {"action_id": "end_turn", "kind": "end_turn"},
        {"action_id": "use_potion:0", "kind": "use_potion", "potion": {"id": "POTION.STRENGTH_POTION"}},
    ]
    mask = np.array([1, 1], dtype=np.float32)
    search_stats: dict = {}

    with patch.object(trainer_stub, "_is_kaiser_encounter_context", return_value=False), patch.object(
        trainer_stub, "_is_insatiable_encounter_context", return_value=False
    ), patch.object(
        trainer_stub, "_combat_encounter_tier_from_raw", return_value="normal"
    ), patch.object(
        trainer_stub, "_combat_energy", return_value=0.0
    ), patch.object(
        trainer_stub, "_semantic_family", side_effect=lambda action: action.get("kind") or "unknown"
    ), patch.object(
        trainer_stub,
        "_potion_timing_profile",
        return_value=_boss_race_strength_profile(hp=70.0, max_hp=91.0, hp_ratio=70.0 / 91.0, threat_gap=6.0),
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
    assert search_stats["combat_quality_late_normal_race_potion_guard_available"] == 0.0
    assert search_stats["combat_quality_late_normal_race_potion_guard_applied"] == 0.0


def test_floor4_race_potion_guard_still_dormant(trainer_stub):
    """The floor-8 broadening must not leak into early Act1."""
    raw_obs = {
        "encounter": "ENCOUNTER.CORPSE_SLUGS_NORMAL",
        "run": {"floor": 4},
        "combat": {"energy": 0, "enemies": [{"current_hp": 40, "intent": {"total_damage": 14}}]},
        "player": {"hp": 53, "current_hp": 53, "max_hp": 91, "block": 0},
    }
    legal_actions = [
        {"action_id": "end_turn", "kind": "end_turn"},
        {"action_id": "use_potion:0", "kind": "use_potion", "potion": {"id": "POTION.STRENGTH_POTION"}},
    ]
    mask = np.array([1, 1], dtype=np.float32)
    search_stats: dict = {}

    with patch.object(trainer_stub, "_is_kaiser_encounter_context", return_value=False), patch.object(
        trainer_stub, "_is_insatiable_encounter_context", return_value=False
    ), patch.object(
        trainer_stub, "_combat_encounter_tier_from_raw", return_value="normal"
    ), patch.object(
        trainer_stub, "_combat_energy", return_value=0.0
    ), patch.object(
        trainer_stub, "_semantic_family", side_effect=lambda action: action.get("kind") or "unknown"
    ), patch.object(
        trainer_stub,
        "_potion_timing_profile",
        return_value=_boss_race_strength_profile(hp=53.0, max_hp=91.0, hp_ratio=53.0 / 91.0, threat_gap=14.0),
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
    assert search_stats["combat_quality_late_normal_race_potion_guard_available"] == 0.0
    assert search_stats["combat_quality_late_normal_race_potion_guard_applied"] == 0.0


def test_combat_floor_value_prefers_sandbox_snapshot_floor(trainer_stub):
    """Combat sandbox must use the injected snapshot floor for tactical guards.

    Bridge ``run.floor`` can be a synthetic combat-room value such as 1 while
    the injected snapshot came from a late Act1/Act2 hallway.  Do not overwrite
    the bridge field, but hard guards must prefer the explicit snapshot floor.
    """

    raw_obs = {"run": {"floor": 1}, "snapshot_floor_number": 25}

    assert trainer_stub._combat_floor_value(raw_obs) == 25.0


def test_combat_floor_value_reads_run_snapshot_floor(trainer_stub):
    """The sandbox decorator also mirrors snapshot_floor_number under run."""

    raw_obs = {"run": {"floor": 1, "snapshot_floor_number": 46}}

    assert trainer_stub._combat_floor_value(raw_obs) == 46.0


def test_late_normal_race_potion_guard_handles_hard_normal_with_fake_sandbox_floor(trainer_stub):
    """Hard-normal curated sandbox fights should trigger race-potion logic even
    when the live bridge floor is fake/low.

    This is the death-slice regression for slumbering_beetle/fogmog/etc.:
    the snapshot is a hard normal hallway, current energy is 0, End Turn is
    unsafe-ish, and a Strength-style setup potion is legal.
    """

    raw_obs = {
        "run": {"floor": 1},
        "encounter_id": "ENCOUNTER.SLUMBERING_BEETLE_NORMAL",
        "combat": {"energy": 0, "enemies": [{"current_hp": 40, "intent": {"total_damage": 8}}]},
        "player": {"hp": 35, "current_hp": 35, "max_hp": 70, "block": 0},
    }
    legal_actions = [
        {"action_id": "end_turn", "kind": "end_turn", "label": "End Turn"},
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
        trainer_stub, "_combat_encounter_tier_from_raw", return_value="normal"
    ), patch.object(
        trainer_stub, "_combat_energy", return_value=0.0
    ), patch.object(
        trainer_stub, "_semantic_family", side_effect=lambda action: action.get("kind") or "unknown"
    ), patch.object(
        trainer_stub,
        "_potion_timing_profile",
        return_value=_boss_race_strength_profile(
            hp=35.0,
            max_hp=70.0,
            hp_ratio=0.5,
            threat_gap=8.0,
        ),
    ), patch.object(
        trainer_stub, "_is_action_confirmed_lethal", return_value=False
    ):
        new_idx = trainer_stub._apply_combat_action_hard_guards(
            action_idx=0,
            legal_actions=legal_actions,
            action_mask=mask,
            raw_obs=raw_obs,
            boss_ctx={},
            encounter="ENCOUNTER.SLUMBERING_BEETLE_NORMAL",
            search_stats=search_stats,
        )

    assert new_idx == 1
    assert search_stats["combat_quality_late_normal_race_potion_guard_available"] == 1.0
    assert search_stats["combat_quality_late_normal_race_potion_guard_applied"] == 1.0
    assert search_stats["combat_quality_late_normal_race_potion_guard_override"] == 1.0
    assert search_stats["combat_quality_potion_bad_guard_late_normal_race_skip"] == 1.0
    assert search_stats["combat_quality_potion_bad_guard_applied"] == 0.0


def test_potion_bad_guard_keeps_hard_normal_race_potion_with_fake_sandbox_floor(trainer_stub):
    """The generic bad-potion guard must not undo a hard-normal race potion.

    The selected action is already the potion.  Even though the timing profile
    marks it low-urgency/no-followup, hard-normal + 0 energy + no useful card
    should fail open instead of rewriting to End Turn.
    """

    raw_obs = {
        "run": {"floor": 1},
        "snapshot_sample_id": "run42/f25/slumbering_beetle_normal",
        "encounter_id": "ENCOUNTER.SLUMBERING_BEETLE_NORMAL",
        "combat": {"energy": 0, "enemies": [{"current_hp": 40, "intent": {"total_damage": 8}}]},
        "player": {"hp": 35, "current_hp": 35, "max_hp": 70, "block": 0},
    }
    legal_actions = [
        {
            "action_id": "use_potion:0",
            "kind": "use_potion",
            "potion": {"id": "POTION.STRENGTH_POTION", "title": "力量药水"},
        },
        {"action_id": "end_turn", "kind": "end_turn", "label": "End Turn"},
    ]
    mask = np.array([1, 1], dtype=np.float32)
    search_stats: dict = {}

    with patch.object(trainer_stub, "_is_kaiser_encounter_context", return_value=False), patch.object(
        trainer_stub, "_is_insatiable_encounter_context", return_value=False
    ), patch.object(
        trainer_stub, "_combat_encounter_tier_from_raw", return_value="normal"
    ), patch.object(
        trainer_stub, "_combat_energy", return_value=0.0
    ), patch.object(
        trainer_stub, "_semantic_family", side_effect=lambda action: action.get("kind") or "unknown"
    ), patch.object(
        trainer_stub,
        "_potion_timing_profile",
        return_value=_boss_race_strength_profile(
            hp=35.0,
            max_hp=70.0,
            hp_ratio=0.5,
            threat_gap=8.0,
        ),
    ), patch.object(
        trainer_stub, "_is_action_confirmed_lethal", return_value=False
    ):
        new_idx = trainer_stub._apply_combat_action_hard_guards(
            action_idx=0,
            legal_actions=legal_actions,
            action_mask=mask,
            raw_obs=raw_obs,
            boss_ctx={},
            encounter="ENCOUNTER.SLUMBERING_BEETLE_NORMAL",
            search_stats=search_stats,
        )

    assert new_idx == 0
    assert search_stats["combat_quality_potion_bad_guard_late_normal_race_skip"] == 1.0
    assert search_stats["combat_quality_potion_bad_guard_applied"] == 0.0
    assert search_stats.get("combat_quality_potion_bad_guard_override", 0.0) == 0.0


def test_potion_bad_guard_skips_bad_zero_energy_x_cost_fallback(trainer_stub):
    """Bad-potion fallback must not choose a known-bad 0-energy X-cost card.

    Regression chain: X-cost hard guard correctly prevents 0-energy X cards,
    then potion_bad guard tries to replace a bad potion and must not fall back
    into the same bad X-cost action.
    """

    raw_obs = {
        "encounter": "ENCOUNTER.CORPSE_SLUGS_NORMAL",
        "run": {"floor": 4},
        "combat": {"energy": 0, "enemies": [{"current_hp": 20, "intent": {"total_damage": 0}}]},
        "player": {"hp": 70, "current_hp": 70, "max_hp": 80, "block": 0},
    }
    legal_actions = [
        {
            "action_id": "play_card_x",
            "kind": "play_card",
            "card": {"id": "CARD.WHIRLWIND", "title": "Whirlwind", "cost": "X"},
            "card_cost": "X",
            "semantic": {"is_x_cost": True, "x_cost_value": 1},
        },
        {
            "action_id": "use_potion:0",
            "kind": "use_potion",
            "potion": {"id": "POTION.STRENGTH_POTION", "title": "力量药水"},
        },
        {"action_id": "end_turn", "kind": "end_turn", "label": "End Turn"},
    ]
    mask = np.array([1, 1, 1], dtype=np.float32)
    search_stats: dict = {}

    with patch.object(trainer_stub, "_is_kaiser_encounter_context", return_value=False), patch.object(
        trainer_stub, "_is_insatiable_encounter_context", return_value=False
    ), patch.object(
        trainer_stub, "_combat_encounter_tier_from_raw", return_value="normal"
    ), patch.object(
        trainer_stub, "_combat_energy", return_value=0.0
    ), patch.object(
        trainer_stub, "_semantic_family", side_effect=lambda action: action.get("kind") or "unknown"
    ), patch.object(
        trainer_stub,
        "_potion_timing_profile",
        return_value=_boss_race_strength_profile(
            hp=70.0,
            max_hp=80.0,
            hp_ratio=70.0 / 80.0,
            threat_gap=0.0,
        ),
    ), patch.object(
        trainer_stub,
        "_x_cost_diagnostic",
        side_effect=lambda action, current_energy: (
            {"x_cost_bad": 1.0, "is_x_cost": 1.0}
            if isinstance(action, dict) and action.get("action_id") == "play_card_x"
            else {"x_cost_bad": 0.0, "is_x_cost": 0.0}
        ),
    ), patch.object(
        trainer_stub, "_is_action_confirmed_lethal", return_value=False
    ):
        new_idx = trainer_stub._apply_combat_action_hard_guards(
            action_idx=1,
            legal_actions=legal_actions,
            action_mask=mask,
            raw_obs=raw_obs,
            boss_ctx={},
            encounter="ENCOUNTER.CORPSE_SLUGS_NORMAL",
            search_stats=search_stats,
        )

    assert search_stats["combat_quality_potion_bad_guard_available"] == 1.0
    assert search_stats["combat_quality_potion_bad_guard_skipped_bad_x_cost_alt"] == 1.0
    assert new_idx != 0
    assert new_idx == 2
