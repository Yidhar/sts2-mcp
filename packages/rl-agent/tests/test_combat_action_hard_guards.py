"""Smoke tests for the P0-3 / P0-4 combat action hard guards.

The guard helpers are method-on-`MuZeroTrainer`, but the override decision
only depends on a small subset of state (``raw_obs`` + boss context dict +
legal_actions + mask + search_stats). To avoid spinning up the full trainer
for a unit test, we instantiate a minimal stub that mirrors only the
helpers the guard reaches for, then call the guard directly.
"""
from __future__ import annotations

import sys
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from muzero.combat_quality.potion_guard import potion_slot_from_action_for_guard


@pytest.fixture
def trainer_stub():
    """Construct a MuZeroTrainer with the absolute minimum config so that
    the guard helper can run without instantiating bridge / network /
    snapshot pool. We patch ``__init__`` to skip heavy setup."""
    from muzero.train import MuZeroTrainer

    trainer = MuZeroTrainer.__new__(MuZeroTrainer)
    trainer.log_dir = None  # skip diagnostics writes
    trainer.episode_count = 0
    trainer.total_steps = 0
    return trainer


def test_urgent_endturn_guard_overrides_sampled_tail_when_urgent_defense_exists(trainer_stub):
    """Leftover energy is fine; sampled EndTurn with urgent legal defense is not.

    Regression target from sandbox diagnostics: policy placed almost all mass on
    non-EndTurn actions, but multinomial sampling selected End Turn while an
    affordable urgent Defend was legal under incoming damage.
    """

    raw_obs = {
        "encounter": "ENCOUNTER.CORPSE_SLUGS_NORMAL",
        "combat": {"energy": 1, "enemies": [{"intent": {"total_damage": 14}}]},
        "player": {"hp": 66, "current_hp": 66, "max_hp": 80, "block": 0},
    }
    legal_actions = [
        {"action_id": "defend", "kind": "play_card", "card": {"cost": 1}, "block": 5},
        {"action_id": "end_turn", "kind": "end_turn"},
    ]
    mask = np.array([1, 1], dtype=np.float32)
    search_stats = {"combat_quality_bad_end_turn_selected": 1.0, "combat_quality_end_turn_selected": 1.0}

    with patch.object(trainer_stub, "_semantic_family", side_effect=lambda action: action.get("kind")), patch.object(
        trainer_stub, "_combat_energy", return_value=1.0
    ), patch.object(
        trainer_stub, "_incoming_damage_pressure", return_value=(14.0, 0.0, 66.0)
    ), patch.object(
        trainer_stub, "_combat_encounter_tier_from_raw", return_value="normal"
    ), patch.object(
        trainer_stub, "_action_cost_value", side_effect=lambda action: action.get("card", {}).get("cost", 0)
    ), patch.object(
        trainer_stub, "_x_cost_diagnostic", return_value={"x_cost_bad": 0.0}
    ), patch.object(
        trainer_stub,
        "_classify_positive_combat_action",
        return_value={"positive": True, "urgent": True},
    ), patch.object(
        trainer_stub, "_is_action_confirmed_lethal", return_value=False
    ), patch.object(
        trainer_stub, "_action_metric", side_effect=lambda action, key: float(action.get(key, 0.0) or 0.0)
    ), patch.object(
        trainer_stub, "_action_numeric_value", return_value=0.0
    ), patch.object(
        trainer_stub, "_action_immediate_impact", side_effect=lambda action: float(action.get("block", 0.0) or 0.0)
    ), patch.object(
        trainer_stub, "_is_meaningful_block_urgent", return_value=True
    ), patch.object(
        trainer_stub, "_dump_combat_hard_guard_record", return_value=None
    ):
        new_idx = trainer_stub._apply_urgent_endturn_guard(
            action_idx=1,
            legal_count=2,
            legal_actions=legal_actions,
            mask_np=mask,
            raw_obs=raw_obs,
            encounter="ENCOUNTER.CORPSE_SLUGS_NORMAL",
            search_stats=search_stats,
        )

    assert new_idx == 0
    assert search_stats["combat_quality_urgent_endturn_guard_available"] == 1.0
    assert search_stats["combat_quality_urgent_endturn_guard_applied"] == 1.0
    assert search_stats["combat_quality_urgent_endturn_guard_override"] == 1.0
    assert search_stats["combat_quality_urgent_endturn_guard_candidate_count"] == 1.0
    # EndTurn-selected gauges are reset after the hard override so downstream
    # diagnostics reflect the post-guard action rather than the sampled tail.
    assert search_stats["combat_quality_bad_end_turn_selected"] == 0.0
    assert search_stats["combat_quality_end_turn_selected"] == 0.0


def test_urgent_endturn_guard_allows_true_forced_leftover_energy_endturn(trainer_stub):
    """Remaining energy with no legal non-EndTurn action is a normal forced pass."""

    raw_obs = {
        "encounter": "ENCOUNTER.OWL_MAGISTRATE_NORMAL",
        "combat": {"energy": 1, "enemies": [{"intent": {"total_damage": 24}}]},
        "player": {"hp": 49, "current_hp": 49, "max_hp": 80, "block": 5},
    }
    legal_actions = [{"action_id": "end_turn", "kind": "end_turn"}]
    search_stats: dict = {}

    with patch.object(trainer_stub, "_semantic_family", side_effect=lambda action: action.get("kind")):
        new_idx = trainer_stub._apply_urgent_endturn_guard(
            action_idx=0,
            legal_count=1,
            legal_actions=legal_actions,
            mask_np=np.array([1], dtype=np.float32),
            raw_obs=raw_obs,
            encounter="ENCOUNTER.OWL_MAGISTRATE_NORMAL",
            search_stats=search_stats,
        )

    assert new_idx == 0
    assert search_stats["combat_quality_urgent_endturn_guard_forced_skip"] == 1.0
    assert search_stats.get("combat_quality_urgent_endturn_guard_applied", 0.0) == 0.0


def test_urgent_endturn_guard_allows_leftover_energy_without_positive_alternative(trainer_stub):
    """Do not treat energy>0 as bad when the only alternative is non-positive."""

    raw_obs = {
        "encounter": "ENCOUNTER.STATUS_ONLY_NORMAL",
        "combat": {"energy": 2, "enemies": [{"intent": {"total_damage": 0}}]},
        "player": {"hp": 70, "current_hp": 70, "max_hp": 80, "block": 0},
    }
    legal_actions = [
        {"action_id": "play_status", "kind": "play_card", "card": {"cost": 0}},
        {"action_id": "end_turn", "kind": "end_turn"},
    ]
    search_stats: dict = {}

    with patch.object(trainer_stub, "_semantic_family", side_effect=lambda action: action.get("kind")), patch.object(
        trainer_stub, "_combat_energy", return_value=2.0
    ), patch.object(
        trainer_stub, "_incoming_damage_pressure", return_value=(0.0, 0.0, 70.0)
    ), patch.object(
        trainer_stub, "_combat_encounter_tier_from_raw", return_value="normal"
    ), patch.object(
        trainer_stub, "_action_cost_value", return_value=0.0
    ), patch.object(
        trainer_stub, "_x_cost_diagnostic", return_value={"x_cost_bad": 0.0}
    ), patch.object(
        trainer_stub, "_classify_positive_combat_action", return_value={"positive": False}
    ):
        new_idx = trainer_stub._apply_urgent_endturn_guard(
            action_idx=1,
            legal_count=2,
            legal_actions=legal_actions,
            mask_np=np.array([1, 1], dtype=np.float32),
            raw_obs=raw_obs,
            encounter="ENCOUNTER.STATUS_ONLY_NORMAL",
            search_stats=search_stats,
        )

    assert new_idx == 1
    assert search_stats["combat_quality_urgent_endturn_guard_candidate_count"] == 0.0
    assert search_stats["combat_quality_urgent_endturn_guard_no_alternative"] == 1.0
    assert search_stats.get("combat_quality_urgent_endturn_guard_applied", 0.0) == 0.0


def test_urgent_endturn_guard_does_not_spend_low_urgency_potion(trainer_stub):
    """Low-pressure/resource potions are not urgent EndTurn replacements."""

    raw_obs = {
        "encounter": "ENCOUNTER.BOWLBUGS_WEAK",
        "combat": {"energy": 1, "enemies": [{"intent": {"total_damage": 0}}]},
        "player": {"hp": 76, "current_hp": 76, "max_hp": 80, "block": 0},
    }
    legal_actions = [
        {"action_id": "use_potion:0", "kind": "use_potion", "potion": {"id": "POTION.STRENGTH"}},
        {"action_id": "end_turn", "kind": "end_turn"},
    ]
    search_stats: dict = {}

    with patch.object(trainer_stub, "_semantic_family", side_effect=lambda action: action.get("kind")), patch.object(
        trainer_stub, "_combat_energy", return_value=1.0
    ), patch.object(
        trainer_stub, "_incoming_damage_pressure", return_value=(0.0, 0.0, 76.0)
    ), patch.object(
        trainer_stub, "_combat_encounter_tier_from_raw", return_value="weak"
    ), patch.object(
        trainer_stub,
        "_classify_positive_combat_action",
        return_value={"positive": True, "potion_low_urgency": True},
    ), patch.object(
        trainer_stub, "_is_action_confirmed_lethal", return_value=False
    ), patch.object(
        trainer_stub, "_action_metric", return_value=0.0
    ), patch.object(
        trainer_stub, "_action_numeric_value", return_value=0.0
    ), patch.object(
        trainer_stub, "_action_immediate_impact", return_value=0.0
    ), patch.object(
        trainer_stub, "_is_meaningful_block_urgent", return_value=False
    ):
        new_idx = trainer_stub._apply_urgent_endturn_guard(
            action_idx=1,
            legal_count=2,
            legal_actions=legal_actions,
            mask_np=np.array([1, 1], dtype=np.float32),
            raw_obs=raw_obs,
            encounter="ENCOUNTER.BOWLBUGS_WEAK",
            search_stats=search_stats,
        )

    assert new_idx == 1
    assert search_stats["combat_quality_urgent_endturn_guard_candidate_count"] == 0.0
    assert search_stats["combat_quality_urgent_endturn_guard_no_alternative"] == 1.0
    assert search_stats.get("combat_quality_urgent_endturn_guard_applied", 0.0) == 0.0


def test_combat_tier_inference_handles_live_bridge_boss_shapes(trainer_stub):
    """Live combat raw_obs can omit top-level encounter_id.

    Boss/elite guards must still classify route/run room metadata and
    distinctive boss monster ids correctly; otherwise Act1 boss hard guards
    silently degrade to normal-combat behavior.
    """
    assert trainer_stub._combat_encounter_tier_from_raw({"run": {"room_type": "Boss"}}) == "boss"
    assert trainer_stub._combat_encounter_tier_from_raw({"run": {"room_type": "Elite"}}) == "elite"
    assert (
        trainer_stub._combat_encounter_tier_from_raw(
            {"combat": {"enemies": [{"model_id": "MONSTER.LAGAVULIN_MATRIARCH"}]}}
        )
        == "boss"
    )
    assert trainer_stub._combat_encounter_tier_from_raw({"run": {"room_model": "MONSTER.WATERFALL_GIANT"}}) == "boss"
    assert trainer_stub._combat_encounter_tier_from_raw({"run": {"room_model": "MONSTER.QUEEN"}}) == "boss"
    assert (
        trainer_stub._combat_encounter_tier_from_raw(
            {"combat": {"enemies": [{"model_id": "MONSTER.SOUL_FYSH"}]}}
        )
        == "boss"
    )
    assert (
        trainer_stub._combat_encounter_tier_from_raw(
            {"combat": {"enemies": [{"model_id": "MONSTER.TEST_SUBJECT"}]}}
        )
        == "boss"
    )


def test_boss_potion_guard_uses_callsite_encounter_hint_when_raw_obs_omits_it(trainer_stub):
    """Regression for live diagnostics where end_turn contexts had tier=''.

    The caller already passes encounter='ENCOUNTER.*_BOSS'.  If raw_obs lacks
    that field, the guard should still inject the hint and override a lethal-ish
    0-energy boss End Turn to Liquid Memories.
    """
    raw_obs = {
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
    assert search_stats["combat_quality_boss_survival_potion_guard_applied"] == 1.0


def test_boss_race_potion_guard_spends_lucky_tonic_under_lethal_pressure(trainer_stub):
    """Regression: Act1 boss death had legal End Turn + Lucky Tonic only.

    Lucky Tonic/幸运补剂 grants Buffer.  It is not block and used to be tagged as
    low-urgency/save, so both boss potion guards reported "no alternative" and
    accepted a fatal EndTurn.  In a boss survival window it must be a candidate.
    """

    raw_obs = {
        "encounter": "ENCOUNTER.LAGAVULIN_MATRIARCH_BOSS",
        "combat": {"energy": 0, "enemies": [{"current_hp": 80, "intent": {"total_damage": 18}}]},
        "player": {"hp": 12, "current_hp": 12, "max_hp": 80, "block": 0},
    }
    legal_actions = [
        {"action_id": "end_turn", "kind": "end_turn"},
        {
            "action_id": "use_potion:0:self",
            "kind": "use_potion",
            "potion": {"id": "POTION.LUCKY_TONIC", "title": "幸运补剂"},
        },
    ]
    lucky_profile = {
        "hp_valid": True,
        "hp": 12.0,
        "max_hp": 80.0,
        "hp_ratio": 0.15,
        "threat_gap": 18.0,
        "potion_id": "POTION.LUCKY_TONIC",
        "effect_family": ["buff", "buffer", "prevent_damage"],
        "semantic_tags": ["buff", "defense", "survival", "buffer"],
        "timing_tags": ["prevent_major_loss_tool", "survival_tool", "boss_survival_tool"],
        "training_tags": ["combat_immediate"],
        "prevent_damage": 1.0,
        "use_quality": 0.08,
    }
    search_stats: dict = {}

    with patch.object(trainer_stub, "_semantic_family", side_effect=lambda action: action.get("kind") or "unknown"), patch.object(
        trainer_stub, "_player_hp_values", return_value=(12.0, 80.0, True)
    ), patch.object(
        trainer_stub, "_combat_encounter_tier_from_raw", return_value="boss"
    ), patch.object(
        trainer_stub, "_incoming_damage_pressure", return_value=(18.0, 0.0, 12.0)
    ), patch.object(
        trainer_stub, "_combat_energy", return_value=0.0
    ), patch.object(
        trainer_stub, "_potion_timing_profile", return_value=lucky_profile
    ), patch.object(
        trainer_stub, "_is_action_confirmed_lethal", return_value=False
    ), patch.object(
        trainer_stub, "_dump_combat_hard_guard_record", return_value=None
    ):
        new_idx = trainer_stub._apply_boss_race_potion_guard(
            action_idx=0,
            legal_count=2,
            legal_actions=legal_actions,
            mask_np=np.array([1, 1], dtype=np.float32),
            raw_obs=raw_obs,
            encounter="ENCOUNTER.LAGAVULIN_MATRIARCH_BOSS",
            search_stats=search_stats,
        )

    assert new_idx == 1
    assert search_stats["combat_quality_boss_race_potion_guard_available"] == 1.0
    assert search_stats["combat_quality_boss_race_potion_guard_applied"] == 1.0
    assert search_stats["combat_quality_boss_race_potion_guard_override"] == 1.0
    assert search_stats["combat_quality_boss_survival_potion_guard_applied"] == 1.0
    assert search_stats["combat_quality_potion_selected"] == 1.0


def test_boss_survival_potion_guard_spends_lucky_tonic_title_only_zh(trainer_stub):
    """Regression hardening: compact bridge action may only expose 幸运药剂 title."""

    raw_obs = {
        "encounter": "ENCOUNTER.LAGAVULIN_MATRIARCH_BOSS",
        "combat": {"energy": 0, "enemies": [{"current_hp": 80, "intent": {"total_damage": 18}}]},
        "player": {"hp": 12, "current_hp": 12, "max_hp": 80, "block": 0},
    }
    legal_actions = [
        {"action_id": "end_turn", "kind": "end_turn"},
        {
            "action_id": "use_potion:0:self",
            "kind": "use_potion",
            "potion": {"title": "幸运药剂"},
        },
    ]
    title_only_profile = {
        "hp_valid": True,
        "hp": 12.0,
        "max_hp": 80.0,
        "hp_ratio": 0.15,
        "threat_gap": 18.0,
        "effect_family": [],
        "semantic_tags": [],
        "timing_tags": [],
        "training_tags": [],
        "buffer_like": True,
        "use_quality": 0.08,
    }
    search_stats: dict = {}

    with patch.object(trainer_stub, "_semantic_family", side_effect=lambda action: action.get("kind") or "unknown"), patch.object(
        trainer_stub, "_player_hp_values", return_value=(12.0, 80.0, True)
    ), patch.object(
        trainer_stub, "_combat_encounter_tier_from_raw", return_value="boss"
    ), patch.object(
        trainer_stub, "_incoming_damage_pressure", return_value=(18.0, 0.0, 12.0)
    ), patch.object(
        trainer_stub, "_combat_energy", return_value=0.0
    ), patch.object(
        trainer_stub, "_potion_timing_profile", return_value=title_only_profile
    ), patch.object(
        trainer_stub, "_is_action_confirmed_lethal", return_value=False
    ), patch.object(
        trainer_stub, "_dump_combat_hard_guard_record", return_value=None
    ):
        new_idx = trainer_stub._apply_boss_survival_potion_guard(
            action_idx=0,
            legal_count=2,
            legal_actions=legal_actions,
            mask_np=np.array([1, 1], dtype=np.float32),
            raw_obs=raw_obs,
            encounter="ENCOUNTER.LAGAVULIN_MATRIARCH_BOSS",
            search_stats=search_stats,
        )

    assert new_idx == 1
    assert search_stats["combat_quality_boss_survival_potion_guard_available"] == 1.0
    assert search_stats["combat_quality_boss_survival_potion_guard_applied"] == 1.0
    assert search_stats["combat_quality_boss_survival_potion_guard_override"] == 1.0


def test_boss_survival_potion_guard_resolves_new_boss_marker_from_raw_obs(trainer_stub):
    """Raw bridge monster ids such as SOUL_FYSH must still enable boss Lucky guard."""

    raw_obs = {
        "run": {"room_model": "MONSTER.SOUL_FYSH"},
        "combat": {"energy": 0, "enemies": [{"current_hp": 120, "intent": {"total_damage": 8}}]},
        "player": {
            "hp": 7,
            "current_hp": 7,
            "max_hp": 80,
            "block": 0,
            "potions": [{"id": "POTION.LUCKY_TONIC", "title": "幸运药剂"}],
        },
    }
    legal_actions = [
        {"action_id": "end_turn", "kind": "end_turn"},
        {
            "action_id": "use_potion:0:self",
            "kind": "use_potion",
            "potion": {"id": "POTION.LUCKY_TONIC", "title": "幸运药剂"},
        },
    ]
    lucky_profile = {
        "hp_valid": True,
        "hp": 7.0,
        "max_hp": 80.0,
        "hp_ratio": 7.0 / 80.0,
        "threat_gap": 8.0,
        "potion_id": "POTION.LUCKY_TONIC",
        "effect_family": ["buff", "buffer"],
        "semantic_tags": ["survival", "buffer"],
        "timing_tags": ["survival_tool", "boss_survival_tool"],
        "prevent_damage": 1.0,
        "buffer_like": True,
        "use_quality": 0.20,
    }
    search_stats: dict = {}

    with patch.object(trainer_stub, "_semantic_family", side_effect=lambda action: action.get("kind") or "unknown"), patch.object(
        trainer_stub, "_player_hp_values", return_value=(7.0, 80.0, True)
    ), patch.object(
        trainer_stub, "_incoming_damage_pressure", return_value=(8.0, 0.0, 7.0)
    ), patch.object(
        trainer_stub, "_combat_energy", return_value=0.0
    ), patch.object(
        trainer_stub, "_potion_timing_profile", return_value=lucky_profile
    ), patch.object(
        trainer_stub, "_is_action_confirmed_lethal", return_value=False
    ), patch.object(
        trainer_stub, "_dump_combat_hard_guard_record", return_value=None
    ):
        new_idx = trainer_stub._apply_boss_survival_potion_guard(
            action_idx=0,
            legal_count=2,
            legal_actions=legal_actions,
            mask_np=np.array([1, 1], dtype=np.float32),
            raw_obs=raw_obs,
            encounter="MONSTER.SOUL_FYSH",
            search_stats=search_stats,
        )

    assert trainer_stub._combat_encounter_tier_from_raw(raw_obs) == "boss"
    assert new_idx == 1
    assert search_stats["combat_quality_boss_survival_potion_guard_available"] == 1.0
    assert search_stats["combat_quality_boss_survival_potion_guard_applied"] == 1.0
    assert search_stats["combat_quality_boss_survival_potion_guard_override"] == 1.0


def test_boss_survival_potion_guard_overrides_low_value_card_to_lucky_tonic(trainer_stub):
    """Boss emergency: Lucky must rescue more than just End Turn.

    The observed miss mode was "death with Lucky unused"; the selected action
    does not have to be End Turn.  If the model plays a non-lethal low-value
    card in a lethal boss window, prefer a legal Buffer/Lucky survival potion.
    """

    raw_obs = {
        "encounter": "ENCOUNTER.LAGAVULIN_MATRIARCH_BOSS",
        "combat": {"energy": 1, "enemies": [{"current_hp": 80, "intent": {"total_damage": 24}}]},
        "player": {"hp": 20, "current_hp": 20, "max_hp": 80, "block": 0},
    }
    legal_actions = [
        {"action_id": "play_card:0:0", "kind": "play_card", "card": {"id": "Strike", "cost": 1}, "damage": 6},
        {
            "action_id": "use_potion:0:self",
            "kind": "use_potion",
            "potion": {"id": "POTION.LUCKY_TONIC", "title": "幸运补剂"},
        },
    ]
    lucky_profile = {
        "hp_valid": True,
        "hp": 20.0,
        "max_hp": 80.0,
        "hp_ratio": 0.25,
        "threat_gap": 24.0,
        "potion_id": "POTION.LUCKY_TONIC",
        "effect_family": ["buff", "buffer", "prevent_damage"],
        "semantic_tags": ["buff", "defense", "survival", "buffer"],
        "timing_tags": ["survival_tool", "boss_survival_tool"],
        "prevent_damage": 1.0,
        "buffer_like": True,
        "use_quality": 0.20,
    }
    search_stats: dict = {}

    with patch.object(trainer_stub, "_semantic_family", side_effect=lambda action: action.get("kind") or "unknown"), patch.object(
        trainer_stub, "_player_hp_values", return_value=(20.0, 80.0, True)
    ), patch.object(
        trainer_stub, "_combat_encounter_tier_from_raw", return_value="boss"
    ), patch.object(
        trainer_stub, "_incoming_damage_pressure", return_value=(24.0, 0.0, 20.0)
    ), patch.object(
        trainer_stub, "_combat_energy", return_value=1.0
    ), patch.object(
        trainer_stub, "_potion_timing_profile", return_value=lucky_profile
    ), patch.object(
        trainer_stub, "_is_action_confirmed_lethal", return_value=False
    ), patch.object(
        trainer_stub, "_action_cost_value", side_effect=lambda action: float(action.get("card", {}).get("cost", action.get("cost", 0)) or 0)
    ), patch.object(
        trainer_stub, "_action_metric", side_effect=lambda action, key: float(action.get(key, 0.0) or 0.0)
    ), patch.object(
        trainer_stub,
        "_action_numeric_value",
        side_effect=lambda action, keys: max(float(action.get(k, 0.0) or 0.0) for k in keys),
    ), patch.object(
        trainer_stub, "_action_roles", side_effect=lambda action: set(action.get("roles", []) or [])
    ), patch.object(
        trainer_stub, "_dump_combat_hard_guard_record", return_value=None
    ):
        new_idx = trainer_stub._apply_boss_survival_potion_guard(
            action_idx=0,
            legal_count=2,
            legal_actions=legal_actions,
            mask_np=np.array([1, 1], dtype=np.float32),
            raw_obs=raw_obs,
            encounter="ENCOUNTER.LAGAVULIN_MATRIARCH_BOSS",
            search_stats=search_stats,
        )

    assert new_idx == 1
    assert search_stats["combat_quality_boss_survival_potion_guard_available"] == 1.0
    assert search_stats["combat_quality_boss_survival_potion_guard_applied"] == 1.0
    assert search_stats["combat_quality_boss_survival_potion_guard_override"] == 1.0
    assert search_stats["combat_quality_boss_survival_potion_any_action_guard_available"] == 1.0
    assert search_stats["combat_quality_boss_survival_potion_any_action_guard_applied"] == 1.0
    assert search_stats["combat_quality_boss_survival_potion_any_action_guard_override"] == 1.0
    assert search_stats["combat_quality_potion_selected"] == 1.0


def test_boss_survival_potion_guard_lucky_takeover_moderate_boss_pressure(trainer_stub):
    """Boss medium-pressure: low-value card should not spend HP while Lucky is legal.

    This covers the miss mode where the model does *something* instead of End
    Turn, but that action does not prevent boss-race damage.  The window is not
    immediately lethal (38 HP, 4 unblocked incoming), so it only activates for a
    legal Lucky/幸运药剂 survival potion and should not be confused with the
    stricter any-action lethal/near-lethal takeover.
    """

    raw_obs = {
        "encounter": "ENCOUNTER.WATERFALL_GIANT_BOSS",
        "combat": {"energy": 1, "enemies": [{"current_hp": 80, "intent": {"total_damage": 10}}]},
        "player": {"hp": 38, "current_hp": 38, "max_hp": 80, "block": 6},
    }
    legal_actions = [
        {"action_id": "play_card:0:0", "kind": "play_card", "card": {"id": "Strike", "cost": 1}, "damage": 6},
        {
            "action_id": "use_potion:0:self",
            "kind": "use_potion",
            "potion": {"id": "POTION.LUCKY_TONIC", "title": "幸运补剂"},
        },
    ]
    lucky_profile = {
        "hp_valid": True,
        "hp": 38.0,
        "max_hp": 80.0,
        "hp_ratio": 38.0 / 80.0,
        "threat_gap": 4.0,
        "potion_id": "POTION.LUCKY_TONIC",
        "effect_family": ["buff"],
        "semantic_tags": [],
        "timing_tags": [],
        "buffer_like": True,
        "use_quality": 0.10,
    }
    search_stats: dict = {}

    with patch.object(trainer_stub, "_semantic_family", side_effect=lambda action: action.get("kind") or "unknown"), patch.object(
        trainer_stub, "_player_hp_values", return_value=(38.0, 80.0, True)
    ), patch.object(
        trainer_stub, "_combat_encounter_tier_from_raw", return_value="boss"
    ), patch.object(
        trainer_stub, "_incoming_damage_pressure", return_value=(10.0, 6.0, 38.0)
    ), patch.object(
        trainer_stub, "_combat_energy", return_value=1.0
    ), patch.object(
        trainer_stub, "_potion_timing_profile", return_value=lucky_profile
    ), patch.object(
        trainer_stub, "_is_action_confirmed_lethal", return_value=False
    ), patch.object(
        trainer_stub, "_action_cost_value", side_effect=lambda action: float(action.get("card", {}).get("cost", action.get("cost", 0)) or 0)
    ), patch.object(
        trainer_stub, "_action_metric", side_effect=lambda action, key: float(action.get(key, 0.0) or 0.0)
    ), patch.object(
        trainer_stub,
        "_action_numeric_value",
        side_effect=lambda action, keys: max(float(action.get(k, 0.0) or 0.0) for k in keys),
    ), patch.object(
        trainer_stub, "_action_roles", side_effect=lambda action: set(action.get("roles", []) or [])
    ), patch.object(
        trainer_stub, "_dump_combat_hard_guard_record", return_value=None
    ):
        new_idx = trainer_stub._apply_boss_survival_potion_guard(
            action_idx=0,
            legal_count=2,
            legal_actions=legal_actions,
            mask_np=np.array([1, 1], dtype=np.float32),
            raw_obs=raw_obs,
            encounter="ENCOUNTER.WATERFALL_GIANT_BOSS",
            search_stats=search_stats,
        )

    assert new_idx == 1
    assert search_stats["combat_quality_boss_survival_lucky_moderate_takeover_window"] == 1.0
    assert search_stats["combat_quality_boss_survival_potion_guard_available"] == 1.0
    assert search_stats["combat_quality_boss_survival_potion_guard_applied"] == 1.0
    assert search_stats["combat_quality_boss_survival_potion_guard_override"] == 1.0
    assert search_stats["combat_quality_boss_survival_potion_any_action_guard_available"] == 1.0
    assert search_stats["combat_quality_boss_survival_potion_any_action_guard_applied"] == 1.0
    assert search_stats["combat_quality_potion_selected"] == 1.0


def test_boss_survival_potion_guard_does_not_replace_confirmed_lethal_card(trainer_stub):
    """Do not burn Lucky if the currently selected card kills the boss."""

    raw_obs = {
        "encounter": "ENCOUNTER.LAGAVULIN_MATRIARCH_BOSS",
        "combat": {"energy": 1, "enemies": [{"current_hp": 6, "intent": {"total_damage": 24}}]},
        "player": {"hp": 20, "current_hp": 20, "max_hp": 80, "block": 0},
    }
    legal_actions = [
        {"action_id": "play_card:0:0", "kind": "play_card", "card": {"id": "Strike", "cost": 1}, "lethal": True},
        {"action_id": "use_potion:0:self", "kind": "use_potion", "potion": {"id": "POTION.LUCKY_TONIC"}},
    ]
    search_stats: dict = {}

    with patch.object(trainer_stub, "_semantic_family", side_effect=lambda action: action.get("kind") or "unknown"), patch.object(
        trainer_stub, "_player_hp_values", return_value=(20.0, 80.0, True)
    ), patch.object(
        trainer_stub, "_combat_encounter_tier_from_raw", return_value="boss"
    ), patch.object(
        trainer_stub, "_incoming_damage_pressure", return_value=(24.0, 0.0, 20.0)
    ), patch.object(
        trainer_stub, "_is_action_confirmed_lethal", side_effect=lambda action, raw: bool(action.get("lethal", False))
    ):
        new_idx = trainer_stub._apply_boss_survival_potion_guard(
            action_idx=0,
            legal_count=2,
            legal_actions=legal_actions,
            mask_np=np.array([1, 1], dtype=np.float32),
            raw_obs=raw_obs,
            encounter="ENCOUNTER.LAGAVULIN_MATRIARCH_BOSS",
            search_stats=search_stats,
        )

    assert new_idx == 0
    assert search_stats["combat_quality_boss_survival_potion_guard_available"] == 1.0
    assert search_stats["combat_quality_boss_survival_potion_any_action_guard_available"] == 1.0
    assert search_stats["combat_quality_boss_survival_potion_guard_lethal_exemption"] == 1.0
    assert search_stats.get("combat_quality_boss_survival_potion_guard_applied", 0.0) == 0.0


def test_boss_survival_potion_any_action_guard_prefers_lethal_alternative_over_lucky(trainer_stub):
    """If a kill is legal, do not burn Lucky over a non-lethal selected card."""

    raw_obs = {
        "encounter": "ENCOUNTER.LAGAVULIN_MATRIARCH_BOSS",
        "combat": {"energy": 1, "enemies": [{"current_hp": 6, "intent": {"total_damage": 24}}]},
        "player": {"hp": 20, "current_hp": 20, "max_hp": 80, "block": 0},
    }
    legal_actions = [
        {"action_id": "play_card:0:0", "kind": "play_card", "card": {"id": "Strike", "cost": 1}, "damage": 3},
        {"action_id": "play_card:1:0", "kind": "play_card", "card": {"id": "Strike+", "cost": 1}, "lethal": True},
        {"action_id": "use_potion:0:self", "kind": "use_potion", "potion": {"id": "POTION.LUCKY_TONIC"}},
    ]
    search_stats: dict = {}

    with patch.object(trainer_stub, "_semantic_family", side_effect=lambda action: action.get("kind") or "unknown"), patch.object(
        trainer_stub, "_player_hp_values", return_value=(20.0, 80.0, True)
    ), patch.object(
        trainer_stub, "_combat_encounter_tier_from_raw", return_value="boss"
    ), patch.object(
        trainer_stub, "_incoming_damage_pressure", return_value=(24.0, 0.0, 20.0)
    ), patch.object(
        trainer_stub, "_is_action_confirmed_lethal", side_effect=lambda action, raw: bool(action.get("lethal", False))
    ):
        new_idx = trainer_stub._apply_boss_survival_potion_guard(
            action_idx=0,
            legal_count=3,
            legal_actions=legal_actions,
            mask_np=np.array([1, 1, 1], dtype=np.float32),
            raw_obs=raw_obs,
            encounter="ENCOUNTER.LAGAVULIN_MATRIARCH_BOSS",
            search_stats=search_stats,
        )

    assert new_idx == 1
    assert search_stats["combat_quality_boss_survival_potion_guard_available"] == 1.0
    assert search_stats["combat_quality_boss_survival_potion_any_action_guard_available"] == 1.0
    assert search_stats["combat_quality_boss_survival_potion_guard_lethal_exemption"] == 1.0
    assert search_stats["combat_quality_boss_survival_potion_guard_lethal_alternative_override"] == 1.0
    assert search_stats["combat_quality_boss_survival_potion_any_action_guard_lethal_alternative_override"] == 1.0
    assert search_stats.get("combat_quality_boss_survival_potion_guard_applied", 0.0) == 0.0
    assert search_stats.get("combat_quality_potion_selected", 0.0) == 0.0


def test_boss_survival_potion_guard_selected_lucky_yields_to_lethal_alternative(trainer_stub):
    """Selected Lucky is a survival play, but a confirmed kill is better."""

    raw_obs = {
        "encounter": "ENCOUNTER.LAGAVULIN_MATRIARCH_BOSS",
        "combat": {"energy": 1, "enemies": [{"current_hp": 6, "intent": {"total_damage": 24}}]},
        "player": {"hp": 20, "current_hp": 20, "max_hp": 80, "block": 0},
    }
    legal_actions = [
        {"action_id": "use_potion:0:self", "kind": "use_potion", "potion": {"id": "POTION.LUCKY_TONIC"}},
        {"action_id": "play_card:1:0", "kind": "play_card", "card": {"id": "Strike+", "cost": 1}, "lethal": True},
    ]
    search_stats: dict = {"combat_quality_potion_selected": 1.0}

    with patch.object(trainer_stub, "_semantic_family", side_effect=lambda action: action.get("kind") or "unknown"), patch.object(
        trainer_stub, "_player_hp_values", return_value=(20.0, 80.0, True)
    ), patch.object(
        trainer_stub, "_combat_encounter_tier_from_raw", return_value="boss"
    ), patch.object(
        trainer_stub, "_incoming_damage_pressure", return_value=(24.0, 0.0, 20.0)
    ), patch.object(
        trainer_stub, "_is_action_confirmed_lethal", side_effect=lambda action, raw: bool(action.get("lethal", False))
    ):
        new_idx = trainer_stub._apply_boss_survival_potion_guard(
            action_idx=0,
            legal_count=2,
            legal_actions=legal_actions,
            mask_np=np.array([1, 1], dtype=np.float32),
            raw_obs=raw_obs,
            encounter="ENCOUNTER.LAGAVULIN_MATRIARCH_BOSS",
            search_stats=search_stats,
        )

    assert new_idx == 1
    assert search_stats["combat_quality_boss_survival_potion_guard_available"] == 1.0
    assert search_stats["combat_quality_boss_survival_potion_any_action_guard_available"] == 1.0
    assert search_stats["combat_quality_boss_survival_potion_guard_lethal_exemption"] == 1.0
    assert search_stats["combat_quality_boss_survival_potion_guard_lethal_alternative_override"] == 1.0
    assert search_stats["combat_quality_boss_survival_potion_any_action_guard_lethal_alternative_override"] == 1.0
    assert search_stats["combat_quality_potion_selected"] == 0.0
    assert search_stats.get("combat_quality_boss_survival_potion_guard_applied", 0.0) == 0.0


def test_boss_survival_potion_guard_does_not_replace_sufficient_block_card(trainer_stub):
    """Do not replace a selected card that already survives incoming damage."""

    raw_obs = {
        "encounter": "ENCOUNTER.LAGAVULIN_MATRIARCH_BOSS",
        "combat": {"energy": 1, "enemies": [{"current_hp": 80, "intent": {"total_damage": 24}}]},
        "player": {"hp": 20, "current_hp": 20, "max_hp": 80, "block": 0},
    }
    legal_actions = [
        {"action_id": "play_card:0:self", "kind": "play_card", "card": {"id": "Defend", "cost": 1}, "block": 24, "roles": ["block"]},
        {"action_id": "use_potion:0:self", "kind": "use_potion", "potion": {"id": "POTION.LUCKY_TONIC"}},
    ]
    search_stats: dict = {}

    with patch.object(trainer_stub, "_semantic_family", side_effect=lambda action: action.get("kind") or "unknown"), patch.object(
        trainer_stub, "_player_hp_values", return_value=(20.0, 80.0, True)
    ), patch.object(
        trainer_stub, "_combat_encounter_tier_from_raw", return_value="boss"
    ), patch.object(
        trainer_stub, "_incoming_damage_pressure", return_value=(24.0, 0.0, 20.0)
    ), patch.object(
        trainer_stub, "_combat_energy", return_value=1.0
    ), patch.object(
        trainer_stub, "_is_action_confirmed_lethal", return_value=False
    ), patch.object(
        trainer_stub, "_action_cost_value", side_effect=lambda action: float(action.get("card", {}).get("cost", action.get("cost", 0)) or 0)
    ), patch.object(
        trainer_stub, "_action_metric", side_effect=lambda action, key: float(action.get(key, 0.0) or 0.0)
    ), patch.object(
        trainer_stub,
        "_action_numeric_value",
        side_effect=lambda action, keys: max(float(action.get(k, 0.0) or 0.0) for k in keys),
    ), patch.object(
        trainer_stub, "_action_roles", side_effect=lambda action: set(action.get("roles", []) or [])
    ):
        new_idx = trainer_stub._apply_boss_survival_potion_guard(
            action_idx=0,
            legal_count=2,
            legal_actions=legal_actions,
            mask_np=np.array([1, 1], dtype=np.float32),
            raw_obs=raw_obs,
            encounter="ENCOUNTER.LAGAVULIN_MATRIARCH_BOSS",
            search_stats=search_stats,
        )

    assert new_idx == 0
    assert search_stats["combat_quality_boss_survival_potion_guard_available"] == 1.0
    assert search_stats["combat_quality_boss_survival_potion_any_action_guard_available"] == 1.0
    assert search_stats["combat_quality_boss_survival_potion_guard_selected_survival_exemption"] == 1.0
    assert search_stats.get("combat_quality_boss_survival_potion_guard_applied", 0.0) == 0.0


def test_boss_survival_potion_guard_overrides_low_value_card_to_title_only_lucky_zh(trainer_stub):
    """Title-only 幸运药剂 must still be recognized in the any-action guard."""

    raw_obs = {
        "encounter": "ENCOUNTER.LAGAVULIN_MATRIARCH_BOSS",
        "combat": {"energy": 1, "enemies": [{"current_hp": 80, "intent": {"total_damage": 24}}]},
        "player": {"hp": 20, "current_hp": 20, "max_hp": 80, "block": 0},
    }
    legal_actions = [
        {"action_id": "play_card:0:0", "kind": "play_card", "card": {"id": "Strike", "cost": 1}, "damage": 6},
        {"action_id": "use_potion:0:self", "kind": "use_potion", "potion": {"title": "幸运药剂"}},
    ]
    title_only_profile = {
        "hp_valid": True,
        "hp": 20.0,
        "max_hp": 80.0,
        "hp_ratio": 0.25,
        "threat_gap": 24.0,
        "effect_family": [],
        "semantic_tags": [],
        "timing_tags": [],
        "buffer_like": True,
        "use_quality": 0.05,
    }
    search_stats: dict = {}

    with patch.object(trainer_stub, "_semantic_family", side_effect=lambda action: action.get("kind") or "unknown"), patch.object(
        trainer_stub, "_player_hp_values", return_value=(20.0, 80.0, True)
    ), patch.object(
        trainer_stub, "_combat_encounter_tier_from_raw", return_value="boss"
    ), patch.object(
        trainer_stub, "_incoming_damage_pressure", return_value=(24.0, 0.0, 20.0)
    ), patch.object(
        trainer_stub, "_combat_energy", return_value=1.0
    ), patch.object(
        trainer_stub, "_potion_timing_profile", return_value=title_only_profile
    ), patch.object(
        trainer_stub, "_is_action_confirmed_lethal", return_value=False
    ), patch.object(
        trainer_stub, "_action_cost_value", side_effect=lambda action: float(action.get("card", {}).get("cost", action.get("cost", 0)) or 0)
    ), patch.object(
        trainer_stub, "_action_metric", side_effect=lambda action, key: float(action.get(key, 0.0) or 0.0)
    ), patch.object(
        trainer_stub,
        "_action_numeric_value",
        side_effect=lambda action, keys: max(float(action.get(k, 0.0) or 0.0) for k in keys),
    ), patch.object(
        trainer_stub, "_action_roles", side_effect=lambda action: set(action.get("roles", []) or [])
    ), patch.object(
        trainer_stub, "_dump_combat_hard_guard_record", return_value=None
    ):
        new_idx = trainer_stub._apply_boss_survival_potion_guard(
            action_idx=0,
            legal_count=2,
            legal_actions=legal_actions,
            mask_np=np.array([1, 1], dtype=np.float32),
            raw_obs=raw_obs,
            encounter="ENCOUNTER.LAGAVULIN_MATRIARCH_BOSS",
            search_stats=search_stats,
        )

    assert new_idx == 1
    assert search_stats["combat_quality_boss_survival_potion_any_action_guard_applied"] == 1.0
    assert search_stats["combat_quality_potion_selected"] == 1.0


def test_boss_survival_potion_guard_resolves_slot_only_lucky_from_raw_obs(trainer_stub):
    """Slot-only use_potion action must resolve Lucky/幸运药剂 from raw inventory.

    This is the boss-death miss mode observed in full-run diagnostics: the legal
    action surface says only ``use_potion:0`` while the actual potion identity is
    carried by ``raw_obs.player.potions[0]``.  A fatal boss EndTurn must be
    rewritten to that potion even when the timing profile has no id/tags.
    """

    raw_obs = {
        "encounter": "ENCOUNTER.LAGAVULIN_MATRIARCH_BOSS",
        "combat": {"energy": 0, "enemies": [{"current_hp": 80, "intent": {"total_damage": 18}}]},
        "player": {
            "hp": 12,
            "current_hp": 12,
            "max_hp": 80,
            "block": 0,
            "potions": [{"id": "POTION.LUCKY_TONIC", "title": "幸运药剂"}],
        },
    }
    legal_actions = [
        {"action_id": "end_turn", "kind": "end_turn"},
        {"action_id": "use_potion:0:self", "kind": "use_potion"},
    ]
    low_identity_profile = {
        "hp_valid": True,
        "hp": 12.0,
        "max_hp": 80.0,
        "hp_ratio": 0.15,
        "threat_gap": 18.0,
        "effect_family": [],
        "semantic_tags": [],
        "timing_tags": [],
        "training_tags": [],
        "use_quality": 0.08,
        "low_urgency": True,
    }
    search_stats: dict = {}

    with patch.object(trainer_stub, "_semantic_family", side_effect=lambda action: action.get("kind") or "unknown"), patch.object(
        trainer_stub, "_player_hp_values", return_value=(12.0, 80.0, True)
    ), patch.object(
        trainer_stub, "_combat_encounter_tier_from_raw", return_value="boss"
    ), patch.object(
        trainer_stub, "_incoming_damage_pressure", return_value=(18.0, 0.0, 12.0)
    ), patch.object(
        trainer_stub, "_combat_energy", return_value=0.0
    ), patch.object(
        trainer_stub, "_potion_timing_profile", return_value=low_identity_profile
    ), patch.object(
        trainer_stub, "_is_action_confirmed_lethal", return_value=False
    ), patch.object(
        trainer_stub, "_dump_combat_hard_guard_record", return_value=None
    ):
        new_idx = trainer_stub._apply_boss_survival_potion_guard(
            action_idx=0,
            legal_count=2,
            legal_actions=legal_actions,
            mask_np=np.array([1, 1], dtype=np.float32),
            raw_obs=raw_obs,
            encounter="ENCOUNTER.LAGAVULIN_MATRIARCH_BOSS",
            search_stats=search_stats,
        )

    assert new_idx == 1
    assert search_stats["combat_quality_boss_survival_potion_guard_available"] == 1.0
    assert search_stats["combat_quality_boss_survival_potion_guard_applied"] == 1.0
    assert search_stats["combat_quality_boss_survival_potion_guard_override"] == 1.0
    assert search_stats["combat_quality_potion_selected"] == 1.0


def test_boss_race_potion_guard_resolves_slot_only_lucky_from_raw_obs(trainer_stub):
    """The earlier boss-race guard also has to see slot-only Lucky identity."""

    raw_obs = {
        "encounter": "ENCOUNTER.LAGAVULIN_MATRIARCH_BOSS",
        "combat": {"energy": 0, "enemies": [{"current_hp": 80, "intent": {"total_damage": 18}}]},
        "player": {
            "hp": 12,
            "current_hp": 12,
            "max_hp": 80,
            "block": 0,
            "potions": [{"id": "POTION.LUCKY_TONIC", "title": "幸运药剂"}],
        },
    }
    legal_actions = [
        {"action_id": "end_turn", "kind": "end_turn"},
        {"action_id": "use_potion:0:self", "kind": "use_potion"},
    ]
    low_identity_profile = {
        "hp_valid": True,
        "hp": 12.0,
        "max_hp": 80.0,
        "hp_ratio": 0.15,
        "threat_gap": 18.0,
        "effect_family": [],
        "semantic_tags": [],
        "timing_tags": [],
        "training_tags": [],
        "use_quality": 0.08,
        "low_urgency": True,
    }
    search_stats: dict = {}

    with patch.object(trainer_stub, "_semantic_family", side_effect=lambda action: action.get("kind") or "unknown"), patch.object(
        trainer_stub, "_player_hp_values", return_value=(12.0, 80.0, True)
    ), patch.object(
        trainer_stub, "_combat_encounter_tier_from_raw", return_value="boss"
    ), patch.object(
        trainer_stub, "_incoming_damage_pressure", return_value=(18.0, 0.0, 12.0)
    ), patch.object(
        trainer_stub, "_combat_energy", return_value=0.0
    ), patch.object(
        trainer_stub, "_potion_timing_profile", return_value=low_identity_profile
    ), patch.object(
        trainer_stub, "_is_action_confirmed_lethal", return_value=False
    ), patch.object(
        trainer_stub, "_dump_combat_hard_guard_record", return_value=None
    ):
        new_idx = trainer_stub._apply_boss_race_potion_guard(
            action_idx=0,
            legal_count=2,
            legal_actions=legal_actions,
            mask_np=np.array([1, 1], dtype=np.float32),
            raw_obs=raw_obs,
            encounter="ENCOUNTER.LAGAVULIN_MATRIARCH_BOSS",
            search_stats=search_stats,
        )

    assert new_idx == 1
    assert search_stats["combat_quality_boss_race_potion_guard_available"] == 1.0
    assert search_stats["combat_quality_boss_race_potion_guard_applied"] == 1.0
    assert search_stats["combat_quality_boss_race_potion_guard_override"] == 1.0
    assert search_stats["combat_quality_boss_survival_potion_guard_applied"] == 1.0


def test_boss_race_potion_guard_saves_lucky_tonic_on_idle_setup_turn(trainer_stub):
    """Do not spend Buffer as generic Lagavulin setup when there is no pressure."""

    raw_obs = {
        "encounter": "ENCOUNTER.LAGAVULIN_MATRIARCH_BOSS",
        "combat": {"energy": 0, "enemies": [{"current_hp": 180, "intent": {"total_damage": 0}}]},
        "player": {"hp": 70, "current_hp": 70, "max_hp": 80, "block": 0},
    }
    legal_actions = [
        {"action_id": "end_turn", "kind": "end_turn"},
        {
            "action_id": "use_potion:0:self",
            "kind": "use_potion",
            "potion": {"id": "POTION.LUCKY_TONIC", "title": "幸运补剂"},
        },
    ]
    lucky_profile = {
        "hp_valid": True,
        "hp": 70.0,
        "max_hp": 80.0,
        "hp_ratio": 70.0 / 80.0,
        "threat_gap": 0.0,
        "potion_id": "POTION.LUCKY_TONIC",
        "effect_family": ["buff", "buffer", "prevent_damage"],
        "semantic_tags": ["buff", "defense", "survival", "buffer"],
        "timing_tags": ["prevent_major_loss_tool", "survival_tool", "boss_survival_tool"],
        "training_tags": ["combat_immediate"],
        "prevent_damage": 1.0,
        "use_quality": 0.08,
    }
    search_stats: dict = {}

    with patch.object(trainer_stub, "_semantic_family", side_effect=lambda action: action.get("kind") or "unknown"), patch.object(
        trainer_stub, "_player_hp_values", return_value=(70.0, 80.0, True)
    ), patch.object(
        trainer_stub, "_combat_encounter_tier_from_raw", return_value="boss"
    ), patch.object(
        trainer_stub, "_incoming_damage_pressure", return_value=(0.0, 0.0, 70.0)
    ), patch.object(
        trainer_stub, "_combat_energy", return_value=0.0
    ), patch.object(
        trainer_stub, "_potion_timing_profile", return_value=lucky_profile
    ), patch.object(
        trainer_stub, "_is_action_confirmed_lethal", return_value=False
    ):
        new_idx = trainer_stub._apply_boss_race_potion_guard(
            action_idx=0,
            legal_count=2,
            legal_actions=legal_actions,
            mask_np=np.array([1, 1], dtype=np.float32),
            raw_obs=raw_obs,
            encounter="ENCOUNTER.LAGAVULIN_MATRIARCH_BOSS",
            search_stats=search_stats,
        )

    assert new_idx == 0
    assert search_stats["combat_quality_boss_race_potion_guard_available"] == 1.0
    assert search_stats["combat_quality_boss_race_potion_guard_no_alternative"] == 1.0
    assert search_stats.get("combat_quality_boss_race_potion_guard_applied", 0.0) == 0.0


def test_env_potion_timing_damage_potion_killing_lethal_attacker_is_prevent_lethal():
    """A damage potion can be the defensive answer when it kills the attacker.

    Previous timing logic only treated block/heal/debuff as ``prevent_lethal``.
    That made a lethal Fire Potion look optional in the same class of death
    frames as the missed Lucky Tonic report.
    """

    from sts2_env.potion_timing import compute_potion_timing

    raw_obs = {
        "encounter": "ENCOUNTER.HAUNTED_SHIP_NORMAL",
        "combat": {
            "energy": 0,
            "enemies": [{"id": 1, "hp": 20, "current_hp": 20, "intent": {"total_damage": 12}}],
        },
        "player": {"hp": 10, "current_hp": 10, "max_hp": 80, "block": 0},
    }
    action = {
        "action_id": "use_potion:0:1",
        "kind": "use_potion",
        "target": {"id": 1},
        "potion": {
            "id": "POTION.FIRE_POTION",
            "title": "火焰药水",
            "effect_profile": {"damage": 20},
            "target_scope": "Enemy",
        },
    }
    legal_actions = [action, {"action_id": "end_turn", "kind": "end_turn"}]

    profile = compute_potion_timing(
        action,
        raw_obs,
        legal_actions,
        np.array([1, 1], dtype=np.float32),
        energy=0.0,
    )

    assert profile["lethal"] is True
    assert profile["lethal_attacker_killable"] is True
    assert profile["prevent_lethal"] is True
    assert profile["urgent"] is True
    assert profile["low_urgency"] is False


def test_trainer_potion_timing_damage_potion_killing_lethal_attacker_is_prevent_lethal(trainer_stub):
    """Trainer-side timing mirror must match the env pure function."""

    raw_obs = {
        "encounter": "ENCOUNTER.HAUNTED_SHIP_NORMAL",
        "combat": {
            "energy": 0,
            "enemies": [{"id": 1, "hp": 20, "current_hp": 20, "intent": {"total_damage": 12}}],
        },
        "player": {"hp": 10, "current_hp": 10, "max_hp": 80, "block": 0},
    }
    action = {
        "action_id": "use_potion:0:1",
        "kind": "use_potion",
        "target": {"id": 1},
        "potion": {
            "id": "POTION.FIRE_POTION",
            "title": "火焰药水",
            "effect_profile": {"damage": 20},
            "target_scope": "Enemy",
        },
    }
    legal_actions = [action, {"action_id": "end_turn", "kind": "end_turn"}]

    with patch.object(trainer_stub, "_is_kaiser_facing_change_action", return_value=False):
        profile = trainer_stub._potion_timing_profile(
            action,
            0,
            None,
            raw_obs,
            legal_actions,
            np.array([1, 1], dtype=np.float32),
            0.0,
        )

    assert profile["lethal"] is True
    assert profile["lethal_attacker_killable"] is True
    assert profile["prevent_lethal"] is True
    assert profile["urgent"] is True
    assert profile["low_urgency"] is False


def test_env_potion_timing_title_only_lucky_tonic_under_lethal_pressure_is_urgent():
    """Bridge payloads may expose only the localized title 幸运药剂."""

    from sts2_env.potion_timing import compute_potion_timing

    raw_obs = {
        "encounter": "ENCOUNTER.SOUL_FYSH_BOSS",
        "combat": {"energy": 0, "enemies": [{"hp": 120, "intent": {"total_damage": 8}}]},
        "player": {"hp": 7, "current_hp": 7, "max_hp": 80, "block": 0},
    }
    action = {
        "action_id": "use_potion:0:self",
        "kind": "use_potion",
        "potion": {"title": "幸运药剂"},
    }
    legal_actions = [action, {"action_id": "end_turn", "kind": "end_turn"}]

    profile = compute_potion_timing(
        action,
        raw_obs,
        legal_actions,
        np.array([1, 1], dtype=np.float32),
        energy=0.0,
        encounter_tier="boss",
    )

    assert profile["buffer_like"] is True
    assert profile["prevent_lethal"] is True
    assert profile["critical_hp_usable_survival_potion"] is True
    assert profile["urgent"] is True
    assert profile["low_urgency"] is False


def test_kaiser_facing_guard_overrides_non_facing_under_risk(trainer_stub):
    """Selected = end_turn under back-attack risk with a facing-change
    candidate must be overridden to the facing change."""
    raw_obs = {
        "encounter": "ENCOUNTER.KAISER_CRAB_BOSS",
        "combat": {
            "facing": "left",
            "enemies": [
                {
                    "combat_id": "enemy_left",
                    "current_hp": 80,
                    "powers": [{"id": "BACK_ATTACK_LEFT_POWER", "amount": 1}],
                },
                {
                    "combat_id": "enemy_right",
                    "current_hp": 80,
                    "powers": [{"id": "BACK_ATTACK_RIGHT_POWER", "amount": 1}],
                },
            ],
        },
    }
    legal_actions = [
        {"action_id": "end_turn", "kind": "end_turn", "label": "结束回合"},
        # Targeted strike at the enemy on the *opposite* side of the
        # current facing — facing is "left" so right-side target = facing
        # change candidate.
        {
            "action_id": "play_card_strike_right",
            "kind": "play_card",
            "label": "Strike",
            "card": {"id": "CARD.STRIKE", "title": "Strike"},
            "target": {"combat_id": "enemy_right"},
            "target_combat_id": "enemy_right",
        },
    ]
    mask = np.array([1, 1], dtype=np.float32)
    search_stats: dict = {}

    boss_ctx = {"primary_back_attack_active": 1.0, "primary_back_attack_risk": 1.0, "encounter_id": "ENCOUNTER.KAISER_CRAB_BOSS"}

    with patch.object(trainer_stub, "_is_kaiser_encounter_context", return_value=True), patch.object(
        trainer_stub, "_kaiser_back_attack_risk_from_context", return_value=1.0
    ), patch.object(
        trainer_stub, "_is_action_confirmed_lethal", return_value=False
    ), patch.object(
        trainer_stub,
        "_is_kaiser_facing_change_action",
        side_effect=lambda action, raw_obs=None: action.get("action_id") == "play_card_strike_right",
    ), patch.object(
        trainer_stub, "_semantic_family", side_effect=lambda action: action.get("kind") or "unknown",
    ), patch.object(
        trainer_stub, "_is_insatiable_encounter_context", return_value=False,
    ):
        new_idx = trainer_stub._apply_combat_action_hard_guards(
            action_idx=0,  # end_turn
            legal_actions=legal_actions,
            action_mask=mask,
            raw_obs=raw_obs,
            boss_ctx=boss_ctx,
            encounter="ENCOUNTER.KAISER_CRAB_BOSS",
            search_stats=search_stats,
        )

    assert new_idx == 1  # overridden to the strike-right facing change
    assert search_stats["combat_quality_kaiser_facing_guard_available"] == 1.0
    assert search_stats["combat_quality_kaiser_facing_guard_applied"] == 1.0
    assert search_stats["combat_quality_kaiser_facing_guard_override"] == 1.0
    assert search_stats["combat_quality_kaiser_facing_guard_lethal_exemption"] == 0.0
    # Post-override flags refresh: the kaiser_*_selected gauges must show
    # the override succeeded so the boss_combat/* aggregations are accurate.
    assert search_stats["combat_quality_kaiser_facing_change_selected"] == 1.0
    assert search_stats["combat_quality_kaiser_risky_end_turn_selected"] == 0.0


def test_kaiser_facing_guard_lethal_exemption(trainer_stub):
    """If the original action is a confirmed lethal, the guard must
    record the lethal exemption and NOT override."""
    raw_obs = {"encounter": "ENCOUNTER.KAISER_CRAB_BOSS", "combat": {"facing": "left", "enemies": []}}
    legal_actions = [
        {"action_id": "lethal_strike", "kind": "play_card"},
        {"action_id": "facing_change", "kind": "play_card"},
    ]
    mask = np.array([1, 1], dtype=np.float32)
    search_stats: dict = {}

    with patch.object(trainer_stub, "_is_kaiser_encounter_context", return_value=True), patch.object(
        trainer_stub, "_kaiser_back_attack_risk_from_context", return_value=1.0
    ), patch.object(
        trainer_stub,
        "_is_action_confirmed_lethal",
        side_effect=lambda action, raw_obs=None: action.get("action_id") == "lethal_strike",
    ), patch.object(
        trainer_stub,
        "_is_kaiser_facing_change_action",
        side_effect=lambda action, raw_obs=None: action.get("action_id") == "facing_change",
    ), patch.object(
        trainer_stub, "_semantic_family", side_effect=lambda action: action.get("kind") or "unknown",
    ), patch.object(
        trainer_stub, "_is_insatiable_encounter_context", return_value=False,
    ):
        new_idx = trainer_stub._apply_combat_action_hard_guards(
            action_idx=0,  # lethal_strike
            legal_actions=legal_actions,
            action_mask=mask,
            raw_obs=raw_obs,
            boss_ctx={"primary_back_attack_active": 1.0},
            encounter="ENCOUNTER.KAISER_CRAB_BOSS",
            search_stats=search_stats,
        )

    assert new_idx == 0  # NOT overridden
    assert search_stats["combat_quality_kaiser_facing_guard_available"] == 1.0
    assert search_stats["combat_quality_kaiser_facing_guard_applied"] == 0.0
    assert search_stats["combat_quality_kaiser_facing_guard_lethal_exemption"] == 1.0


def test_insatiable_escape_force_overrides_at_countdown_1(trainer_stub):
    """countdown<=1 with Frantic Escape legal AND selected=non-escape AND
    not lethal → override to Frantic Escape."""
    raw_obs = {"encounter": "ENCOUNTER.THE_INSATIABLE_BOSS"}
    legal_actions = [
        {"action_id": "anger", "kind": "play_card", "card": {"id": "CARD.ANGER"}},
        {
            "action_id": "frantic_escape",
            "kind": "play_card",
            "card": {"id": "CARD.FRANTIC_ESCAPE", "title": "Frantic Escape"},
        },
    ]
    mask = np.array([1, 1], dtype=np.float32)
    search_stats: dict = {}

    boss_ctx = {"insatiable_sandpit_countdown": 1.0}

    with patch.object(trainer_stub, "_is_kaiser_encounter_context", return_value=False), patch.object(
        trainer_stub, "_is_insatiable_encounter_context", return_value=True
    ), patch.object(
        trainer_stub, "_is_action_confirmed_lethal", return_value=False
    ), patch.object(
        trainer_stub, "_semantic_family", side_effect=lambda action: action.get("kind") or "unknown",
    ):
        new_idx = trainer_stub._apply_combat_action_hard_guards(
            action_idx=0,  # anger
            legal_actions=legal_actions,
            action_mask=mask,
            raw_obs=raw_obs,
            boss_ctx=boss_ctx,
            encounter="ENCOUNTER.THE_INSATIABLE_BOSS",
            search_stats=search_stats,
        )

    assert new_idx == 1  # overridden to frantic escape
    assert search_stats["combat_quality_insatiable_escape_force_available"] == 1.0
    assert search_stats["combat_quality_insatiable_escape_force_applied"] == 1.0
    assert search_stats["combat_quality_insatiable_escape_force_override"] == 1.0
    assert search_stats["combat_quality_insatiable_frantic_escape_selected"] == 1.0
    assert search_stats["combat_quality_insatiable_frantic_escape_missed_at_1"] == 0.0


def test_insatiable_escape_force_skips_when_countdown_above_1(trainer_stub):
    """countdown=2 must NOT trigger hard force (soft bias is the only
    P0-4 lever for countdown=2, per spec §P0-4)."""
    raw_obs = {"encounter": "ENCOUNTER.THE_INSATIABLE_BOSS"}
    legal_actions = [
        {"action_id": "anger", "kind": "play_card", "card": {"id": "CARD.ANGER"}},
        {"action_id": "frantic_escape", "kind": "play_card", "card": {"id": "CARD.FRANTIC_ESCAPE"}},
    ]
    mask = np.array([1, 1], dtype=np.float32)
    search_stats: dict = {}

    boss_ctx = {"insatiable_sandpit_countdown": 2.0}

    with patch.object(trainer_stub, "_is_kaiser_encounter_context", return_value=False), patch.object(
        trainer_stub, "_is_insatiable_encounter_context", return_value=True
    ), patch.object(
        trainer_stub, "_is_action_confirmed_lethal", return_value=False
    ), patch.object(
        trainer_stub, "_semantic_family", side_effect=lambda action: action.get("kind") or "unknown",
    ):
        new_idx = trainer_stub._apply_combat_action_hard_guards(
            action_idx=0,
            legal_actions=legal_actions,
            action_mask=mask,
            raw_obs=raw_obs,
            boss_ctx=boss_ctx,
            encounter="ENCOUNTER.THE_INSATIABLE_BOSS",
            search_stats=search_stats,
        )

    assert new_idx == 0  # NOT overridden
    assert search_stats["combat_quality_insatiable_escape_force_available"] == 0.0
    assert search_stats["combat_quality_insatiable_escape_force_applied"] == 0.0


def test_x_cost_zero_guard_overrides_when_alternative_exists(trainer_stub):
    """Selected = X-cost card with effective_energy=0 and no zero-energy
    effect must be overridden to the first non-X-cost / non-end_turn legal
    alternative."""
    raw_obs = {
        "combat": {"energy": 0},
        "player": {"current_hp": 50, "max_hp": 80},
    }
    legal_actions = [
        # Bad X-cost card at 0 energy
        {
            "action_id": "play_card_whirlwind",
            "kind": "play_card",
            "label": "Whirlwind",
            "card": {"id": "CARD.WHIRLWIND", "title": "Whirlwind", "cost": "X"},
            "card_cost": "X",
            "semantic": {"is_x_cost": True, "x_cost_value": 1},
        },
        # Non-X alternative (block card)
        {
            "action_id": "play_card_defend",
            "kind": "play_card",
            "label": "Defend",
            "card": {"id": "CARD.DEFEND", "title": "Defend", "cost": 1},
            "semantic": {"block": 5},
        },
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
    ):
        new_idx = trainer_stub._apply_combat_action_hard_guards(
            action_idx=0,  # bad X-cost
            legal_actions=legal_actions,
            action_mask=mask,
            raw_obs=raw_obs,
            boss_ctx={},
            encounter="ENCOUNTER.X",
            search_stats=search_stats,
        )

    assert new_idx == 1  # overridden to non-X alternative (Defend)
    assert search_stats["combat_quality_x_cost_zero_guard_available"] == 1.0
    assert search_stats["combat_quality_x_cost_zero_guard_applied"] == 1.0
    assert search_stats["combat_quality_x_cost_zero_guard_override"] == 1.0
    assert search_stats.get("combat_quality_x_cost_zero_bad_selected") == 0.0


def test_x_cost_zero_guard_dormant_when_energy_present(trainer_stub):
    """X-cost card with positive energy is fine — guard must not fire."""
    raw_obs = {"combat": {"energy": 3}, "player": {"current_hp": 50}}
    legal_actions = [
        {
            "action_id": "play_card_whirlwind",
            "kind": "play_card",
            "card": {"id": "CARD.WHIRLWIND", "cost": "X"},
            "card_cost": "X",
            "semantic": {"is_x_cost": True, "x_cost_value": 1},
        },
    ]
    mask = np.array([1], dtype=np.float32)
    search_stats: dict = {}

    with patch.object(trainer_stub, "_is_kaiser_encounter_context", return_value=False), patch.object(
        trainer_stub, "_is_insatiable_encounter_context", return_value=False
    ), patch.object(
        trainer_stub, "_combat_energy", return_value=3.0
    ), patch.object(
        trainer_stub, "_semantic_family", side_effect=lambda action: action.get("kind") or "unknown"
    ):
        new_idx = trainer_stub._apply_combat_action_hard_guards(
            action_idx=0,
            legal_actions=legal_actions,
            action_mask=mask,
            raw_obs=raw_obs,
            boss_ctx={},
            encounter="ENCOUNTER.X",
            search_stats=search_stats,
        )

    assert new_idx == 0
    assert search_stats["combat_quality_x_cost_zero_guard_available"] == 0.0
    assert search_stats["combat_quality_x_cost_zero_guard_applied"] == 0.0


def test_x_cost_zero_guard_falls_back_to_end_turn_when_only_safe_alternative(trainer_stub):
    """Regression for live ``倾泻+`` offenders.

    When a bad zero-energy X-card and End Turn are the only legal actions,
    the guard must still rewrite.  Previously it refused End Turn as an
    alternative and left the no-op X-card selected.
    """

    raw_obs = {"combat": {"energy": 0}, "player": {"current_hp": 64, "max_hp": 80}}
    legal_actions = [
        {
            "action_id": "play_card_downpour",
            "kind": "play_card",
            "label": "倾泻+",
            "card": {"id": "CARD.DOWNPOUR", "title": "倾泻+", "cost": "X"},
            "card_cost": "X",
            "semantic": {"is_x_cost": True, "x_cost_value": 1},
        },
        {"action_id": "end_turn", "kind": "end_turn", "label": "End Turn"},
    ]
    mask = np.array([1, 1], dtype=np.float32)
    search_stats: dict = {
        "combat_quality_x_cost_selected": 1.0,
        "combat_quality_x_cost_bad_selected": 1.0,
        "combat_quality_x_cost_zero_bad_selected": 1.0,
        "combat_quality_zero_energy_x_cost_selected": 1.0,
        "combat_quality_x_cost_zero_energy_selected": 1.0,
    }

    with patch.object(trainer_stub, "_combat_energy", return_value=0.0), patch.object(
        trainer_stub, "_semantic_family", side_effect=lambda action: action.get("kind") or "unknown"
    ):
        new_idx = trainer_stub._apply_x_cost_zero_guard(
            action_idx=0,
            legal_count=len(legal_actions),
            legal_actions=legal_actions,
            mask_np=mask,
            raw_obs=raw_obs,
            encounter="ENCOUNTER.OVICOPTER_NORMAL",
            search_stats=search_stats,
        )

    assert new_idx == 1
    assert search_stats["combat_quality_x_cost_zero_guard_available"] == 1.0
    assert search_stats["combat_quality_x_cost_zero_guard_applied"] == 1.0
    assert search_stats["combat_quality_x_cost_zero_guard_override"] == 1.0
    assert search_stats["combat_quality_x_cost_zero_guard_end_turn_fallback"] == 1.0
    assert search_stats["combat_quality_hard_guard_override_any"] == 1.0
    assert search_stats["combat_quality_x_cost_selected"] == 0.0
    assert search_stats["combat_quality_x_cost_bad_selected"] == 0.0
    assert search_stats["combat_quality_zero_energy_x_cost_selected"] == 0.0


def test_refund_no_followup_guard_falls_back_to_end_turn_for_hp_cost_no_followup(trainer_stub):
    """Regression for live ``放血`` / ``放血+`` no-followup offenders.

    If a HP-cost resource/setup card has no same-turn follow-up and no progress
    card alternative, End Turn is strictly safer than spending HP for no real
    damage/block/heal.
    """

    raw_obs = {
        "encounter": "ENCOUNTER.FABRICATOR_NORMAL",
        "combat": {"energy": 1, "enemies": [{"intent": {"total_damage": 13}}]},
        "player": {"hp": 65, "current_hp": 65, "max_hp": 80, "block": 0},
    }
    legal_actions = [
        {
            "action_id": "play_card_bloodletting",
            "kind": "play_card",
            "label": "放血",
            "card": {"id": "CARD.BLOODLETTING", "title": "放血", "cost": 0},
            "safety": {"hp_loss_unblockable": 3.0},
        },
        {"action_id": "end_turn", "kind": "end_turn", "label": "End Turn"},
    ]
    mask = np.array([1, 1], dtype=np.float32)
    selected_profile = {
        "positive": True,
        "deferable": True,
        "followup_missing": True,
        "energy_without_followup": True,
        "setup_followup_dependent": True,
        "setup_followup_available": False,
    }
    search_stats: dict = {
        "combat_quality_refund_no_followup_selected": 1.0,
        "combat_quality_refund_no_followup_no_alternative_selected": 1.0,
        "combat_quality_strategic_skip_selected": 1.0,
    }

    with patch.object(trainer_stub, "_combat_energy", return_value=1.0), patch.object(
        trainer_stub, "_semantic_family", side_effect=lambda action: action.get("kind") or "unknown"
    ), patch.object(
        trainer_stub, "_classify_positive_combat_action", return_value=selected_profile
    ), patch.object(
        trainer_stub, "_is_action_confirmed_lethal", return_value=False
    ), patch.object(
        trainer_stub, "_action_metric", return_value=0.0
    ), patch.object(
        trainer_stub, "_action_numeric_value", return_value=0.0
    ), patch.object(
        trainer_stub, "_action_immediate_impact", return_value=8.5
    ), patch.object(
        trainer_stub, "_action_roles", return_value={"resource", "setup"}
    ):
        new_idx = trainer_stub._apply_refund_no_followup_guard(
            action_idx=0,
            legal_count=len(legal_actions),
            legal_actions=legal_actions,
            mask_np=mask,
            raw_obs=raw_obs,
            encounter="ENCOUNTER.FABRICATOR_NORMAL",
            search_stats=search_stats,
        )

    assert new_idx == 1
    assert search_stats["combat_quality_refund_no_followup_guard_available"] == 1.0
    assert search_stats["combat_quality_refund_no_followup_guard_no_alternative"] == 1.0
    assert search_stats["combat_quality_refund_no_followup_guard_applied"] == 1.0
    assert search_stats["combat_quality_refund_no_followup_guard_override"] == 1.0
    assert search_stats["combat_quality_refund_no_followup_guard_end_turn_fallback"] == 1.0
    assert search_stats["combat_quality_hard_guard_override_any"] == 1.0
    assert search_stats["combat_quality_refund_no_followup_selected"] == 0.0
    assert search_stats["combat_quality_refund_no_followup_no_alternative_selected"] == 0.0
    assert search_stats["combat_quality_strategic_skip_selected"] == 0.0


def test_hp_cost_margin_guard_overrides_low_margin_nonlethal_hp_cost(trainer_stub):
    """Non-lethal HP-cost cards should still be blocked at razor-thin HP.

    Example: Bloodletting at 5 HP leaves 3 HP.  It is legal, but in hallway
    training this teaches bad self-damage habits unless it is a confirmed kill
    or no safe non-HP alternative exists.
    """
    raw_obs = {
        "encounter": "ENCOUNTER.FROG_KNIGHT_NORMAL",
        "combat": {"energy": 1, "enemies": [{"intent": {"total_damage": 0}}]},
        "player": {"hp": 5, "current_hp": 5, "max_hp": 80, "block": 0},
    }
    legal_actions = [
        {
            "action_id": "play_card_bloodletting",
            "kind": "play_card",
            "card": {"id": "CARD.BLOODLETTING", "title": "Bloodletting", "cost": 0},
            "safety": {
                "hp_loss_unblockable": 2.0,
                "hp_before": 5.0,
                "hp_after_self_cost": 3.0,
                "low_hp_margin_after_cost": True,
                "source_confidence": "runtime_internal",
            },
        },
        {
            "action_id": "play_card_strike",
            "kind": "play_card",
            "damage": 6,
            "card": {"id": "CARD.STRIKE", "title": "Strike", "type": "Attack", "damage": 6},
        },
        {"action_id": "end_turn", "kind": "end_turn"},
    ]
    mask = np.array([1, 1, 1], dtype=np.float32)
    search_stats: dict = {"combat_quality_hp_cost_low_margin_selected": 1.0}

    new_idx = trainer_stub._apply_hp_cost_margin_guard(
        action_idx=0,
        legal_count=len(legal_actions),
        legal_actions=legal_actions,
        mask_np=mask,
        raw_obs=raw_obs,
        encounter="ENCOUNTER.FROG_KNIGHT_NORMAL",
        search_stats=search_stats,
    )

    assert new_idx == 1
    assert search_stats["combat_quality_hp_cost_margin_guard_available"] == 1.0
    assert search_stats["combat_quality_hp_cost_margin_guard_applied"] == 1.0
    assert search_stats["combat_quality_hp_cost_margin_guard_override"] == 1.0
    assert search_stats["combat_quality_hp_cost_low_margin_selected"] == 0.0


def test_hp_cost_margin_guard_overrides_title_only_bloodletting(trainer_stub):
    """Regression for live title-only Bloodletting actions.

    The selected action can arrive pre-step without a compact card id/safety
    block, while bridge diagnostics only mark hp_cost_low_margin after the
    damage is already done.  The guard must use static title fallback before
    env.step and rewrite to a non-HP progress action.
    """
    raw_obs = {
        "encounter": "ENCOUNTER.BOWLBUGS_NORMAL",
        "combat": {"energy": 0, "enemies": [{"intent": {"total_damage": 0}}]},
        "player": {"hp": 4, "current_hp": 4, "max_hp": 70, "block": 0},
    }
    legal_actions = [
        {
            "action_id": "play_card_0",
            "kind": "play_card",
            "card": {"title": "放血+", "cost": 0},
        },
        {
            "action_id": "play_card_1",
            "kind": "play_card",
            "damage": 5,
            "card": {"id": "CARD.STRIKE", "title": "打击", "type": "Attack"},
        },
        {"action_id": "end_turn", "kind": "end_turn"},
    ]
    search_stats: dict = {"combat_quality_hp_cost_low_margin_selected": 1.0}

    new_idx = trainer_stub._apply_hp_cost_margin_guard(
        action_idx=0,
        legal_count=len(legal_actions),
        legal_actions=legal_actions,
        mask_np=np.array([1, 1, 1], dtype=np.float32),
        raw_obs=raw_obs,
        encounter="ENCOUNTER.BOWLBUGS_NORMAL",
        search_stats=search_stats,
    )

    assert new_idx == 1
    assert search_stats["combat_quality_hp_cost_margin_guard_available"] == 1.0
    assert search_stats["combat_quality_hp_cost_margin_guard_applied"] == 1.0
    assert search_stats["combat_quality_hp_cost_margin_guard_override"] == 1.0
    assert search_stats["combat_quality_hp_cost_low_margin_selected"] == 0.0


def test_hp_cost_margin_guard_keeps_confirmed_lethal_low_margin_hp_cost(trainer_stub):
    raw_obs = {
        "combat": {"energy": 1, "enemies": [{"intent": {"total_damage": 0}}]},
        "player": {"current_hp": 5, "max_hp": 80, "block": 0},
    }
    legal_actions = [
        {
            "action_id": "play_card_blood_for_kill",
            "kind": "play_card",
            "card": {"id": "CARD.BLOODLETTING", "title": "Bloodletting", "cost": 0},
            "safety": {
                "hp_loss_unblockable": 2.0,
                "hp_before": 5.0,
                "hp_after_self_cost": 3.0,
                "low_hp_margin_after_cost": True,
                "source_confidence": "runtime_internal",
            },
        },
        {"action_id": "play_card_strike", "kind": "play_card", "damage": 6},
    ]
    search_stats: dict = {"combat_quality_hp_cost_low_margin_selected": 1.0}

    with patch.object(trainer_stub, "_is_action_confirmed_lethal", return_value=True):
        new_idx = trainer_stub._apply_hp_cost_margin_guard(
            action_idx=0,
            legal_count=len(legal_actions),
            legal_actions=legal_actions,
            mask_np=np.array([1, 1], dtype=np.float32),
            raw_obs=raw_obs,
            encounter="ENCOUNTER.FROG_KNIGHT_NORMAL",
            search_stats=search_stats,
        )

    assert new_idx == 0
    assert search_stats["combat_quality_hp_cost_margin_guard_available"] == 1.0
    assert search_stats["combat_quality_hp_cost_margin_guard_lethal_exemption"] == 1.0


def test_guards_dormant_when_not_kaiser_or_insatiable(trainer_stub):
    """For non-Kaiser, non-Insatiable encounters the guard must leave
    action_idx untouched and emit zero metrics (gauge presence still
    enforced for cardinality)."""
    raw_obs = {"encounter": "ENCOUNTER.SOUL_NEXUS_ELITE"}
    legal_actions = [{"action_id": "strike", "kind": "play_card"}]
    mask = np.array([1], dtype=np.float32)
    search_stats: dict = {}

    with patch.object(trainer_stub, "_is_kaiser_encounter_context", return_value=False), patch.object(
        trainer_stub, "_is_insatiable_encounter_context", return_value=False
    ), patch.object(
        trainer_stub, "_semantic_family", side_effect=lambda action: action.get("kind") or "unknown",
    ):
        new_idx = trainer_stub._apply_combat_action_hard_guards(
            action_idx=0,
            legal_actions=legal_actions,
            action_mask=mask,
            raw_obs=raw_obs,
            boss_ctx={},
            encounter="ENCOUNTER.SOUL_NEXUS_ELITE",
            search_stats=search_stats,
        )

    assert new_idx == 0
    # All guard keys must be present at 0 so TB tag cardinality is stable.
    for key in (
        "combat_quality_kaiser_facing_guard_available",
        "combat_quality_kaiser_facing_guard_applied",
        "combat_quality_kaiser_facing_guard_override",
        "combat_quality_kaiser_facing_guard_lethal_exemption",
        "combat_quality_insatiable_escape_force_available",
        "combat_quality_insatiable_escape_force_applied",
        "combat_quality_insatiable_escape_force_override",
        "combat_quality_insatiable_escape_force_lethal_exemption",
    ):
        assert search_stats.get(key) == 0.0, f"{key} should be 0.0, got {search_stats.get(key)}"


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


def test_boss_survival_block_guard_overrides_end_turn_to_defend(trainer_stub):
    """At boss low/mid HP with incoming damage, End Turn should be replaced
    by an affordable block card when no lethal action is legal."""
    raw_obs = {
        "encounter": "ENCOUNTER.LAGAVULIN_MATRIARCH_BOSS",
        "combat": {"energy": 1, "enemies": [{"current_hp": 200, "intent": {"total_damage": 18}}]},
        "player": {"hp": 24, "current_hp": 24, "max_hp": 91, "block": 0},
    }
    legal_actions = [
        {"action_id": "end_turn", "kind": "end_turn"},
        {
            "action_id": "play_card_defend",
            "kind": "play_card",
            "card": {"id": "CARD.DEFEND", "title": "Defend", "cost": 1},
            "semantic": {"block": 5},
        },
    ]
    mask = np.array([1, 1], dtype=np.float32)
    search_stats: dict = {}

    with patch.object(trainer_stub, "_is_kaiser_encounter_context", return_value=False), patch.object(
        trainer_stub, "_is_insatiable_encounter_context", return_value=False
    ), patch.object(
        trainer_stub, "_combat_energy", return_value=1.0
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
    assert search_stats["combat_quality_boss_survival_block_guard_available"] == 1.0
    assert search_stats["combat_quality_boss_survival_block_guard_applied"] == 1.0
    assert search_stats["combat_quality_boss_survival_block_guard_override"] == 1.0
    assert search_stats["combat_quality_wasteful_end_turn_selected"] == 0.0


def test_boss_survival_block_guard_skips_insufficient_defend(trainer_stub):
    """Do not teach the replay that a Defend is correct if it still dies.

    Observed boss tail: hp=13, incoming=25, two forced Defends still left
    lethal damage.  Survival candidates must satisfy hp_after > remaining_gap;
    damage equal to current HP is death.
    """
    raw_obs = {
        "encounter": "ENCOUNTER.LAGAVULIN_MATRIARCH_BOSS",
        "combat": {"energy": 1, "enemies": [{"current_hp": 200, "intent": {"total_damage": 25}}]},
        "player": {"hp": 13, "current_hp": 13, "max_hp": 91, "block": 0},
    }
    legal_actions = [
        {"action_id": "end_turn", "kind": "end_turn"},
        {
            "action_id": "play_card_defend",
            "kind": "play_card",
            "label": "Defend",
            "card": {"id": "CARD.DEFEND", "title": "Defend", "cost": 1},
            "semantic": {"block": 5},
        },
    ]
    mask = np.array([1, 1], dtype=np.float32)
    search_stats: dict = {}

    with patch.object(trainer_stub, "_is_kaiser_encounter_context", return_value=False), patch.object(
        trainer_stub, "_is_insatiable_encounter_context", return_value=False
    ), patch.object(
        trainer_stub, "_combat_energy", return_value=1.0
    ), patch.object(
        trainer_stub, "_is_action_confirmed_lethal", return_value=False
    ):
        new_idx = trainer_stub._apply_boss_survival_block_guard(
            action_idx=0,
            legal_count=len(legal_actions),
            legal_actions=legal_actions,
            mask_np=mask,
            raw_obs=raw_obs,
            encounter="ENCOUNTER.LAGAVULIN_MATRIARCH_BOSS",
            search_stats=search_stats,
        )

    assert new_idx == 0
    assert search_stats["combat_quality_boss_survival_block_guard_available"] == 1.0
    assert search_stats["combat_quality_boss_survival_block_guard_insufficient_candidate"] == 1.0
    assert search_stats["combat_quality_boss_survival_block_guard_no_alternative"] == 1.0
    assert search_stats.get("combat_quality_boss_survival_block_guard_applied", 0.0) == 0.0


def test_boss_survival_block_guard_catches_mid_hp_medium_threat(trainer_stub):
    """Regression for Act1 boss deaths after entering around 40-45 HP.

    A 6-damage exposed hit at ~47% HP is not one-turn lethal and did not meet
    the older ``0.20 * hp`` threshold, but repeated medium hits were bleeding
    boss-entry HP to zero.  The oracle guard should spend the visible Defend.
    """
    raw_obs = {
        "encounter": "ENCOUNTER.LAGAVULIN_MATRIARCH_BOSS",
        "run": {"floor": 17, "room_type": "boss"},
        "combat": {"energy": 1, "enemies": [{"current_hp": 150, "intent": {"total_damage": 6}}]},
        "player": {"hp": 43, "current_hp": 43, "max_hp": 91, "block": 0},
    }
    legal_actions = [
        {"action_id": "end_turn", "kind": "end_turn"},
        {
            "action_id": "play_card_defend",
            "kind": "play_card",
            "card": {"id": "CARD.DEFEND", "title": "Defend", "cost": 1},
            "semantic": {"block": 5},
        },
    ]
    mask = np.array([1, 1], dtype=np.float32)
    search_stats: dict = {}

    with patch.object(trainer_stub, "_is_kaiser_encounter_context", return_value=False), patch.object(
        trainer_stub, "_is_insatiable_encounter_context", return_value=False
    ), patch.object(
        trainer_stub, "_combat_energy", return_value=1.0
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
    assert search_stats["combat_quality_boss_survival_block_guard_available"] == 1.0
    assert search_stats["combat_quality_boss_survival_block_guard_applied"] == 1.0
    assert search_stats["combat_quality_boss_survival_block_guard_override"] == 1.0


def test_elite_boss_lethal_end_turn_guard_overrides_before_block(trainer_stub):
    """If End Turn is selected while a legal boss/elite kill exists, take it.

    This prevents the later survival potion/block guards from spending tempo
    defensively when the correct survival action is simply ending the fight.
    """
    raw_obs = {
        "encounter": "ENCOUNTER.LAGAVULIN_MATRIARCH_BOSS",
        "combat": {"energy": 1, "enemies": [{"current_hp": 4, "intent": {"total_damage": 18}}]},
        "player": {"hp": 24, "current_hp": 24, "max_hp": 91, "block": 0},
    }
    legal_actions = [
        {"action_id": "end_turn", "kind": "end_turn"},
        {
            "action_id": "play_card_defend",
            "kind": "play_card",
            "card": {"id": "CARD.DEFEND", "title": "Defend", "cost": 1},
            "semantic": {"block": 5},
        },
        {
            "action_id": "play_card_strike",
            "kind": "play_card",
            "card": {"id": "CARD.STRIKE", "title": "Strike", "cost": 1},
            "semantic": {"damage": 6},
        },
    ]
    mask = np.array([1, 1, 1], dtype=np.float32)
    search_stats: dict = {}

    with patch.object(trainer_stub, "_is_kaiser_encounter_context", return_value=False), patch.object(
        trainer_stub, "_is_insatiable_encounter_context", return_value=False
    ), patch.object(
        trainer_stub, "_combat_energy", return_value=1.0
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

    assert new_idx == 2
    assert search_stats["combat_quality_elite_boss_lethal_end_turn_guard_available"] == 1.0
    assert search_stats["combat_quality_elite_boss_lethal_end_turn_guard_applied"] == 1.0
    assert search_stats["combat_quality_elite_boss_lethal_end_turn_guard_override"] == 1.0
    assert search_stats["combat_quality_boss_survival_block_guard_applied"] == 0.0


def test_late_normal_survival_guard_overrides_end_turn_to_defend(trainer_stub):
    """Late Act1 hallway deaths are the current recovery bottleneck.

    On floor 13 normal combat, selecting End Turn into a large incoming hit
    should be rewritten to an affordable block action when no lethal is
    available.  This is intentionally narrower than the generic wasteful-end
    guard: it only protects late/critical normal hallways.
    """
    raw_obs = {
        "encounter": "ENCOUNTER.CORPSE_SLUGS_NORMAL",
        "run": {"floor": 13},
        "combat": {"energy": 1, "enemies": [{"current_hp": 40, "intent": {"total_damage": 18}}]},
        "player": {"hp": 22, "current_hp": 22, "max_hp": 91, "block": 0},
    }
    legal_actions = [
        {"action_id": "end_turn", "kind": "end_turn", "label": "End Turn"},
        {
            "action_id": "play_card_defend",
            "kind": "play_card",
            "label": "Defend",
            "card": {"id": "CARD.DEFEND", "title": "Defend", "cost": 1},
            "semantic": {"block": 5},
        },
    ]
    mask = np.array([1, 1], dtype=np.float32)
    search_stats: dict = {}

    with patch.object(trainer_stub, "_is_kaiser_encounter_context", return_value=False), patch.object(
        trainer_stub, "_is_insatiable_encounter_context", return_value=False
    ), patch.object(
        trainer_stub, "_combat_encounter_tier_from_raw", return_value="normal"
    ), patch.object(
        trainer_stub, "_combat_energy", return_value=1.0
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
    assert search_stats["combat_quality_late_normal_survival_guard_available"] == 1.0
    assert search_stats["combat_quality_late_normal_survival_guard_applied"] == 1.0
    assert search_stats["combat_quality_late_normal_survival_guard_override"] == 1.0
    assert search_stats["combat_quality_late_normal_survival_guard_candidate_count"] == 1.0
    assert search_stats["combat_quality_late_normal_survival_guard_no_alternative"] == 0.0
    assert search_stats["combat_quality_hard_guard_override_any"] == 1.0


def test_late_normal_survival_guard_skips_insufficient_defend(trainer_stub):
    """Late-normal EndTurn guard must not force pure block that still dies."""
    raw_obs = {
        "encounter": "ENCOUNTER.CORPSE_SLUGS_NORMAL",
        "run": {"floor": 13},
        "combat": {"energy": 1, "enemies": [{"current_hp": 40, "intent": {"total_damage": 18}}]},
        "player": {"hp": 10, "current_hp": 10, "max_hp": 91, "block": 0},
    }
    legal_actions = [
        {"action_id": "end_turn", "kind": "end_turn", "label": "End Turn"},
        {
            "action_id": "play_card_defend",
            "kind": "play_card",
            "label": "Defend",
            "card": {"id": "CARD.DEFEND", "title": "Defend", "cost": 1},
            "semantic": {"block": 5},
        },
    ]
    mask = np.array([1, 1], dtype=np.float32)
    search_stats: dict = {}

    with patch.object(trainer_stub, "_is_kaiser_encounter_context", return_value=False), patch.object(
        trainer_stub, "_is_insatiable_encounter_context", return_value=False
    ), patch.object(
        trainer_stub, "_combat_encounter_tier_from_raw", return_value="normal"
    ), patch.object(
        trainer_stub, "_combat_energy", return_value=1.0
    ), patch.object(
        trainer_stub, "_is_action_confirmed_lethal", return_value=False
    ):
        new_idx = trainer_stub._apply_late_normal_survival_guard(
            action_idx=0,
            legal_count=len(legal_actions),
            legal_actions=legal_actions,
            mask_np=mask,
            raw_obs=raw_obs,
            encounter="ENCOUNTER.CORPSE_SLUGS_NORMAL",
            search_stats=search_stats,
        )

    assert new_idx == 0
    assert search_stats["combat_quality_late_normal_survival_guard_available"] == 1.0
    assert search_stats["combat_quality_late_normal_survival_guard_insufficient_candidate"] == 1.0
    assert search_stats["combat_quality_late_normal_survival_guard_candidate_count"] == 0.0
    assert search_stats["combat_quality_late_normal_survival_guard_no_alternative"] == 1.0
    assert search_stats.get("combat_quality_late_normal_survival_guard_applied", 0.0) == 0.0


def test_late_normal_survival_guard_catches_floor14_medium_threat(trainer_stub):
    """Floor-14 hallway HP preservation is part of the Act1 pass objective.

    The older survival threshold ignored 5 incoming damage at ~50% HP; this is
    acceptable for a single combat but not for a run that still needs to enter
    the boss with enough HP reserve.
    """
    raw_obs = {
        "encounter": "ENCOUNTER.GREMLIN_MERC_NORMAL",
        "run": {"floor": 14},
        "combat": {"energy": 1, "enemies": [{"current_hp": 35, "intent": {"total_damage": 5}}]},
        "player": {"hp": 45, "current_hp": 45, "max_hp": 91, "block": 0},
    }
    legal_actions = [
        {"action_id": "end_turn", "kind": "end_turn", "label": "End Turn"},
        {
            "action_id": "play_card_defend",
            "kind": "play_card",
            "label": "Defend",
            "card": {"id": "CARD.DEFEND", "title": "Defend", "cost": 1},
            "semantic": {"block": 5},
        },
    ]
    mask = np.array([1, 1], dtype=np.float32)
    search_stats: dict = {}

    with patch.object(trainer_stub, "_is_kaiser_encounter_context", return_value=False), patch.object(
        trainer_stub, "_is_insatiable_encounter_context", return_value=False
    ), patch.object(
        trainer_stub, "_combat_encounter_tier_from_raw", return_value="normal"
    ), patch.object(
        trainer_stub, "_combat_energy", return_value=1.0
    ), patch.object(
        trainer_stub, "_is_action_confirmed_lethal", return_value=False
    ):
        new_idx = trainer_stub._apply_combat_action_hard_guards(
            action_idx=0,
            legal_actions=legal_actions,
            action_mask=mask,
            raw_obs=raw_obs,
            boss_ctx={},
            encounter="ENCOUNTER.GREMLIN_MERC_NORMAL",
            search_stats=search_stats,
        )

    assert new_idx == 1
    assert search_stats["combat_quality_late_normal_survival_guard_available"] == 1.0
    assert search_stats["combat_quality_late_normal_survival_guard_applied"] == 1.0
    assert search_stats["combat_quality_late_normal_survival_guard_override"] == 1.0


def test_late_normal_lethal_end_turn_guard_overrides_to_kill(trainer_stub):
    """Late normal End Turn must take a confirmed kill before defense.

    This prevents the recovery guard from becoming over-defensive when the
    correct hallway survival play is simply ending the combat.
    """
    raw_obs = {
        "encounter": "ENCOUNTER.CORPSE_SLUGS_NORMAL",
        "run": {"floor": 13},
        "combat": {"energy": 1, "enemies": [{"current_hp": 4, "intent": {"total_damage": 18}}]},
        "player": {"hp": 22, "current_hp": 22, "max_hp": 91, "block": 0},
    }
    legal_actions = [
        {"action_id": "end_turn", "kind": "end_turn", "label": "End Turn"},
        {
            "action_id": "play_card_defend",
            "kind": "play_card",
            "card": {"id": "CARD.DEFEND", "title": "Defend", "cost": 1},
            "semantic": {"block": 5},
        },
        {
            "action_id": "play_card_strike",
            "kind": "play_card",
            "card": {"id": "CARD.STRIKE", "title": "Strike", "cost": 1},
            "semantic": {"damage": 6},
        },
    ]
    mask = np.array([1, 1, 1], dtype=np.float32)
    search_stats: dict = {}

    with patch.object(trainer_stub, "_is_kaiser_encounter_context", return_value=False), patch.object(
        trainer_stub, "_is_insatiable_encounter_context", return_value=False
    ), patch.object(
        trainer_stub, "_combat_encounter_tier_from_raw", return_value="normal"
    ), patch.object(
        trainer_stub, "_combat_energy", return_value=1.0
    ), patch.object(
        trainer_stub,
        "_is_action_confirmed_lethal",
        side_effect=lambda action, raw_obs=None: isinstance(action, dict)
        and action.get("action_id") == "play_card_strike",
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

    assert new_idx == 2
    assert search_stats["combat_quality_late_normal_lethal_end_turn_guard_available"] == 1.0
    assert search_stats["combat_quality_late_normal_lethal_end_turn_guard_applied"] == 1.0
    assert search_stats["combat_quality_late_normal_lethal_end_turn_guard_override"] == 1.0
    assert search_stats["combat_quality_late_normal_survival_guard_applied"] == 0.0
    assert search_stats["combat_quality_hard_guard_override_any"] == 1.0


def test_late_normal_survival_guard_dormant_on_early_safe_normal(trainer_stub):
    """Early safe normal fights must stay policy-controlled.

    The guard is a late-Act1/critical-HP recovery guard, not a universal
    script that blocks every End Turn in hallways.
    """
    raw_obs = {
        "encounter": "ENCOUNTER.CORPSE_SLUGS_NORMAL",
        "run": {"floor": 5},
        "combat": {"energy": 1, "enemies": [{"current_hp": 40, "intent": {"total_damage": 4}}]},
        "player": {"hp": 80, "current_hp": 80, "max_hp": 91, "block": 0},
    }
    legal_actions = [
        {"action_id": "end_turn", "kind": "end_turn", "label": "End Turn"},
        {
            "action_id": "play_card_defend",
            "kind": "play_card",
            "card": {"id": "CARD.DEFEND", "title": "Defend", "cost": 1},
            "semantic": {"block": 5},
        },
    ]
    mask = np.array([1, 1], dtype=np.float32)
    search_stats: dict = {}

    with patch.object(trainer_stub, "_is_kaiser_encounter_context", return_value=False), patch.object(
        trainer_stub, "_is_insatiable_encounter_context", return_value=False
    ), patch.object(
        trainer_stub, "_combat_encounter_tier_from_raw", return_value="normal"
    ), patch.object(
        trainer_stub, "_combat_energy", return_value=1.0
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
    assert search_stats["combat_quality_late_normal_lethal_end_turn_guard_available"] == 0.0
    assert search_stats["combat_quality_late_normal_lethal_end_turn_guard_applied"] == 0.0
    assert search_stats["combat_quality_late_normal_survival_guard_available"] == 0.0
    assert search_stats["combat_quality_late_normal_survival_guard_applied"] == 0.0
    assert search_stats["combat_quality_hard_guard_override_any"] == 0.0


def test_late_normal_survival_guard_records_no_alternative(trainer_stub):
    """If the bridge exposes only End Turn, fail open and log it.

    This distinguishes "guard did not exist" from "no legal survival action
    was actually available" in tonight's TensorBoard readout.
    """
    raw_obs = {
        "encounter": "ENCOUNTER.CORPSE_SLUGS_NORMAL",
        "run": {"floor": 13},
        "combat": {"energy": 0, "enemies": [{"current_hp": 40, "intent": {"total_damage": 18}}]},
        "player": {"hp": 10, "current_hp": 10, "max_hp": 91, "block": 0},
    }
    legal_actions = [{"action_id": "end_turn", "kind": "end_turn", "label": "End Turn"}]
    mask = np.array([1], dtype=np.float32)
    search_stats: dict = {}

    with patch.object(trainer_stub, "_is_kaiser_encounter_context", return_value=False), patch.object(
        trainer_stub, "_is_insatiable_encounter_context", return_value=False
    ), patch.object(
        trainer_stub, "_combat_encounter_tier_from_raw", return_value="normal"
    ), patch.object(
        trainer_stub, "_combat_energy", return_value=0.0
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
    assert search_stats["combat_quality_late_normal_survival_guard_available"] == 1.0
    assert search_stats["combat_quality_late_normal_survival_guard_applied"] == 0.0
    assert search_stats["combat_quality_late_normal_survival_guard_no_alternative"] == 1.0
    assert search_stats["combat_quality_late_normal_survival_guard_candidate_count"] == 0.0
    assert search_stats["combat_quality_hard_guard_override_any"] == 0.0


def test_late_normal_race_potion_guard_overrides_end_turn_to_strength(trainer_stub):
    """Late Act1 normal, 0 energy, meaningful incoming: End Turn should spend
    a high-confidence race/setup potion when no useful card is currently legal.
    """
    raw_obs = {
        "encounter": "ENCOUNTER.CORPSE_SLUGS_NORMAL",
        "run": {"floor": 13},
        "combat": {"energy": 0, "enemies": [{"current_hp": 40, "intent": {"total_damage": 14}}]},
        "player": {"hp": 53, "current_hp": 53, "max_hp": 91, "block": 0},
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
            hp=53.0,
            max_hp=91.0,
            hp_ratio=53.0 / 91.0,
            threat_gap=14.0,
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
            encounter="ENCOUNTER.CORPSE_SLUGS_NORMAL",
            search_stats=search_stats,
        )

    assert new_idx == 1
    assert search_stats["combat_quality_late_normal_race_potion_guard_available"] == 1.0
    assert search_stats["combat_quality_late_normal_race_potion_guard_applied"] == 1.0
    assert search_stats["combat_quality_late_normal_race_potion_guard_override"] == 1.0
    assert search_stats["combat_quality_late_normal_race_potion_guard_candidate_count"] == 1.0
    # The generic potion-bad guard must not undo the race-potion override.
    assert search_stats["combat_quality_potion_bad_guard_late_normal_race_skip"] == 1.0
    assert search_stats["combat_quality_potion_bad_guard_applied"] == 0.0
    assert search_stats["combat_quality_end_turn_selected"] == 0.0
    assert search_stats["combat_quality_potion_selected"] == 1.0


def test_late_normal_race_potion_guard_dormant_before_floor_11(trainer_stub):
    """The race-potion guard is a late-Act1 recovery patch, not an early
    hallway script.
    """
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
        trainer_stub, "_potion_timing_profile", return_value=_boss_race_strength_profile(threat_gap=14.0)
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


def test_late_normal_race_potion_guard_rejects_fortifier_zero_block_noop(trainer_stub):
    """Fortifier at zero block cannot help the current 0-energy hallway race."""
    raw_obs = {
        "encounter": "ENCOUNTER.CORPSE_SLUGS_NORMAL",
        "run": {"floor": 13},
        "combat": {"energy": 0, "enemies": [{"current_hp": 40, "intent": {"total_damage": 14}}]},
        "player": {"hp": 53, "current_hp": 53, "max_hp": 91, "block": 0},
    }
    legal_actions = [
        {"action_id": "end_turn", "kind": "end_turn"},
        {"action_id": "use_potion:0", "kind": "use_potion", "potion": {"id": "POTION.FORTIFIER", "title": "固化药水"}},
    ]
    mask = np.array([1, 1], dtype=np.float32)
    search_stats: dict = {}
    fortifier_profile = {
        "hp_valid": True,
        "hp": 53.0,
        "max_hp": 91.0,
        "hp_ratio": 53.0 / 91.0,
        "threat_gap": 14.0,
        "potion_id": "POTION.FORTIFIER",
        "effect_family": ["block", "amplify_block"],
        "semantic_tags": ["block"],
        "timing_tags": ["requires_block_in_play"],
        "training_tags": [],
        "amplify_block_noop": True,
        "block_waste": True,
        "block": 0.0,
        "damage": 0.0,
    }

    with patch.object(trainer_stub, "_is_kaiser_encounter_context", return_value=False), patch.object(
        trainer_stub, "_is_insatiable_encounter_context", return_value=False
    ), patch.object(
        trainer_stub, "_combat_encounter_tier_from_raw", return_value="normal"
    ), patch.object(
        trainer_stub, "_combat_energy", return_value=0.0
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
            encounter="ENCOUNTER.CORPSE_SLUGS_NORMAL",
            search_stats=search_stats,
        )

    assert new_idx == 0
    assert search_stats["combat_quality_late_normal_race_potion_guard_available"] == 1.0
    assert search_stats["combat_quality_late_normal_race_potion_guard_no_alternative"] == 1.0
    assert search_stats["combat_quality_late_normal_race_potion_guard_applied"] == 0.0


def test_late_normal_race_potion_guard_does_not_override_when_useful_card_exists(trainer_stub):
    """If a useful zero-cost/non-potion action is already legal, leave the
    policy-controlled End Turn decision to the older card/end-turn guards.
    """
    raw_obs = {
        "encounter": "ENCOUNTER.CORPSE_SLUGS_NORMAL",
        "run": {"floor": 13},
        "combat": {"energy": 0, "enemies": [{"current_hp": 40, "intent": {"total_damage": 14}}]},
        "player": {"hp": 53, "current_hp": 53, "max_hp": 91, "block": 0},
    }
    legal_actions = [
        {"action_id": "end_turn", "kind": "end_turn"},
        {"action_id": "use_potion:0", "kind": "use_potion", "potion": {"id": "POTION.STRENGTH_POTION"}},
        {
            "action_id": "play_card_zero_defend",
            "kind": "play_card",
            "card": {"id": "CARD.TEST_ZERO_DEFEND", "cost": 0},
            "semantic": {"block": 4},
        },
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
        trainer_stub, "_potion_timing_profile", return_value=_boss_race_strength_profile(threat_gap=14.0)
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

    assert new_idx == 2
    assert search_stats["combat_quality_late_normal_race_potion_guard_applied"] == 0.0
    assert search_stats["combat_quality_late_normal_race_potion_guard_override"] == 0.0


def test_potion_bad_use_guard_does_not_undo_late_normal_race_strength_potion(trainer_stub):
    """Selected Strength/Speed-style potion in the late-normal race window
    must fail open instead of being rewritten back to End Turn.
    """
    raw_obs = {
        "encounter": "ENCOUNTER.CORPSE_SLUGS_NORMAL",
        "run": {"floor": 13},
        "combat": {"energy": 0, "enemies": [{"current_hp": 40, "intent": {"total_damage": 14}}]},
        "player": {"hp": 53, "current_hp": 53, "max_hp": 91, "block": 0},
    }
    legal_actions = [
        {"action_id": "use_potion:0", "kind": "use_potion", "potion": {"id": "POTION.STRENGTH_POTION", "title": "力量药水"}},
        {"action_id": "end_turn", "kind": "end_turn"},
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
            hp=53.0,
            max_hp=91.0,
            hp_ratio=53.0 / 91.0,
            threat_gap=14.0,
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
            encounter="ENCOUNTER.CORPSE_SLUGS_NORMAL",
            search_stats=search_stats,
        )

    assert new_idx == 0
    assert search_stats["combat_quality_potion_bad_guard_late_normal_race_skip"] == 1.0
    assert search_stats["combat_quality_potion_bad_guard_applied"] == 0.0
    assert search_stats.get("combat_quality_potion_bad_guard_override", 0.0) == 0.0


def test_hopeless_potion_guard_keeps_late_normal_block_potion_when_profile_misses_threat(trainer_stub):
    """The hopeless-potion exception must not turn a useful late hallway potion
    into End Turn when raw enemy intents show danger.

    This covers the deployed failure mode: the potion timing profile may look
    idle (threat_gap=0 -> block_waste=True), but the raw bridge observation still
    contains an Attack intent.  In that case the diagnostic guard must fail open
    unless the potion is a proven no-op such as Fortifier at 0 current block.
    """
    raw_obs = {
        "encounter": "ENCOUNTER.GREMLIN_MERC_NORMAL",
        "run": {"floor": 13},
        "combat": {"energy": 0, "enemies": [{"current_hp": 15, "intent": {"total_damage": 12}}]},
        "player": {"hp": 23, "current_hp": 23, "max_hp": 91, "block": 0},
    }
    legal_actions = [
        {"action_id": "use_potion:0:0:self", "kind": "use_potion", "potion": {"id": "POTION.BLOCK_POTION"}},
        {"action_id": "end_turn", "kind": "end_turn"},
    ]
    mask = np.array([1, 1], dtype=np.float32)
    search_stats: dict = {}
    profile = {
        "low_urgency": True,
        "save_recommended": True,
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
        "hp": 23.0,
        "max_hp": 91.0,
        "hp_ratio": 23.0 / 91.0,
        # Simulate the broken/too-conservative timing profile; the new guard
        # must recover from raw_obs instead of trusting this zero threat.
        "threat_gap": 0.0,
        "potion_id": "POTION.BLOCK_POTION",
        "effect_family": ["block"],
        "semantic_tags": ["block", "survival"],
        "timing_tags": [],
        "training_tags": ["combat_immediate"],
        "amplify_block_noop": False,
        "retrieve_from_discard_like": False,
        "retrieve_has_target": False,
        "block": 12.0,
        "damage": 0.0,
        "heal": 0.0,
        "use_quality": 0.0,
    }

    with patch.object(trainer_stub, "_is_kaiser_encounter_context", return_value=False), patch.object(
        trainer_stub, "_is_insatiable_encounter_context", return_value=False
    ), patch.object(
        trainer_stub, "_combat_encounter_tier_from_raw", return_value="normal"
    ), patch.object(
        trainer_stub, "_combat_energy", return_value=0.0
    ), patch.object(
        trainer_stub, "_semantic_family", side_effect=lambda action: action.get("kind") or "unknown"
    ), patch.object(
        trainer_stub, "_potion_timing_profile", return_value=profile
    ), patch.object(
        trainer_stub, "_is_action_confirmed_lethal", return_value=False
    ):
        new_idx = trainer_stub._apply_combat_action_hard_guards(
            action_idx=0,
            legal_actions=legal_actions,
            action_mask=mask,
            raw_obs=raw_obs,
            boss_ctx={},
            encounter="ENCOUNTER.GREMLIN_MERC_NORMAL",
            search_stats=search_stats,
        )

    assert new_idx == 0
    assert search_stats["combat_quality_potion_bad_guard_available"] == 1.0
    assert search_stats["combat_quality_potion_bad_hopeless_guard_late_normal_race_skip"] == 1.0
    assert search_stats["combat_quality_potion_bad_guard_forced_end_turn_hopeless"] == 0.0
    assert search_stats["combat_quality_potion_bad_guard_applied"] == 0.0
    assert search_stats["combat_quality_potion_bad_guard_no_alternative"] == 0.0


def test_hopeless_potion_guard_keeps_boss_survival_potion_when_profile_misses_threat(trainer_stub):
    """Boss/elite survival potion use should also fail open under raw danger."""
    raw_obs = {
        "encounter": "ENCOUNTER.LAGAVULIN_MATRIARCH_BOSS",
        "run": {"floor": 17, "room_type": "boss"},
        "combat": {"energy": 0, "enemies": [{"current_hp": 103, "intent": {"total_damage": 19}}]},
        "player": {"hp": 25, "current_hp": 25, "max_hp": 91, "block": 0},
    }
    legal_actions = [
        {"action_id": "use_potion:0:0:self", "kind": "use_potion", "potion": {"id": "POTION.BLOCK_POTION"}},
        {"action_id": "end_turn", "kind": "end_turn"},
    ]
    mask = np.array([1, 1], dtype=np.float32)
    search_stats: dict = {}
    profile = {
        "low_urgency": True,
        "save_recommended": True,
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
        "hp": 25.0,
        "max_hp": 91.0,
        "hp_ratio": 25.0 / 91.0,
        "threat_gap": 0.0,
        "potion_id": "POTION.BLOCK_POTION",
        "effect_family": ["block"],
        "semantic_tags": ["block", "survival"],
        "timing_tags": [],
        "training_tags": ["combat_immediate"],
        "amplify_block_noop": False,
        "retrieve_from_discard_like": False,
        "retrieve_has_target": False,
        "block": 12.0,
        "damage": 0.0,
        "heal": 0.0,
        "use_quality": 0.0,
    }

    with patch.object(trainer_stub, "_is_kaiser_encounter_context", return_value=False), patch.object(
        trainer_stub, "_is_insatiable_encounter_context", return_value=False
    ), patch.object(
        trainer_stub, "_combat_encounter_tier_from_raw", return_value="boss"
    ), patch.object(
        trainer_stub, "_combat_energy", return_value=0.0
    ), patch.object(
        trainer_stub, "_semantic_family", side_effect=lambda action: action.get("kind") or "unknown"
    ), patch.object(
        trainer_stub, "_potion_timing_profile", return_value=profile
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
    assert search_stats["combat_quality_potion_bad_guard_available"] == 1.0
    assert search_stats["combat_quality_potion_bad_hopeless_guard_boss_survival_skip"] == 1.0
    assert search_stats["combat_quality_potion_bad_guard_forced_end_turn_hopeless"] == 0.0
    assert search_stats["combat_quality_potion_bad_guard_applied"] == 0.0


def test_no_pressure_progress_lock_prevents_survival_guard_ping_pong(trainer_stub):
    """Regression for live weak/normal sandbox over-defense.

    A safe hallway Defend under moderate pressure should be rewritten to a
    progress card by the no-pressure block guard.  The later non-EndTurn
    survival guard must not immediately rewrite that progress card back to
    Defend unless the hit is truly critical/near-lethal.
    """
    raw_obs = {
        "encounter": "ENCOUNTER.CONSTRUCT_MENAGERIE_NORMAL",
        "run": {"floor": 12},
        "combat": {
            "energy": 1,
            "enemies": [
                {
                    "combat_id": 1,
                    "current_hp": 40,
                    "max_hp": 40,
                    "intent": {"total_damage": 23, "damage": 23, "intent_type": "Attack"},
                }
            ],
        },
        "player": {"hp": 73, "current_hp": 73, "max_hp": 91, "block": 0},
    }
    legal_actions = [
        {
            "action_id": "play_card_defend",
            "kind": "play_card",
            "card": {
                "id": "CARD.DEFEND",
                "title": "Defend",
                "type": "Skill",
                "cost": 1,
                "block": 5,
                "total_block": 5,
                "preview_block": 5,
                "card_effect_profile": {"semantic_tags": ["block"], "training_tags": []},
            },
            "semantic": {"block": 5, "roles": ["block"]},
        },
        {
            "action_id": "play_card_strike",
            "kind": "play_card",
            "target_combat_id": 1,
            "card": {
                "id": "CARD.STRIKE",
                "title": "Strike",
                "type": "Attack",
                "cost": 1,
                "damage": 6,
                "total_damage": 6,
                "preview_damage": 6,
                "card_effect_profile": {"semantic_tags": ["attack", "damage"], "training_tags": []},
            },
            "semantic": {"damage": 6, "roles": ["attack", "damage"]},
        },
    ]
    mask = np.array([1, 1], dtype=np.float32)
    search_stats: dict = {"combat_quality_bad_pure_block_selected": 1.0}

    with patch.object(trainer_stub, "_is_kaiser_encounter_context", return_value=False), patch.object(
        trainer_stub, "_is_insatiable_encounter_context", return_value=False
    ), patch.object(
        trainer_stub, "_combat_encounter_tier_from_raw", return_value="normal"
    ), patch.object(
        trainer_stub, "_combat_energy", return_value=1.0
    ), patch.object(
        trainer_stub, "_is_action_confirmed_lethal", return_value=False
    ):
        new_idx = trainer_stub._apply_combat_action_hard_guards(
            action_idx=0,
            legal_actions=legal_actions,
            action_mask=mask,
            raw_obs=raw_obs,
            boss_ctx={},
            encounter="ENCOUNTER.CONSTRUCT_MENAGERIE_NORMAL",
            search_stats=search_stats,
        )

    assert new_idx == 1
    assert search_stats["combat_quality_no_pressure_block_guard_applied"] == 1.0
    assert search_stats["combat_quality_no_pressure_block_guard_progress_override_idx"] == 1.0
    assert search_stats["combat_quality_no_pressure_block_guard_progress_override_lock"] == 1.0
    assert search_stats["combat_quality_survival_non_endturn_guard_available"] == 1.0
    assert search_stats["combat_quality_survival_non_endturn_guard_no_pressure_lock_skip"] == 1.0
    assert search_stats["combat_quality_survival_non_endturn_guard_applied"] == 0.0
    assert search_stats["combat_quality_bad_pure_block_selected"] == 0.0


def test_survival_non_endturn_guard_preserves_safe_normal_hallway_progress(trainer_stub):
    """Regression for weak/normal sandbox over-defense.

    The non-EndTurn survival guard may correct real survival blunders, but it
    must not turn an already-selected attack into Defend when the player is at
    high HP and would still be comfortable after the hit.  Otherwise normal
    hallway fights drift into attrition loops: the model spends energy on
    low-value block, fights run longer, and full-run Act1 loses upgrade/rest
    tempo.
    """
    raw_obs = {
        "encounter": "ENCOUNTER.THE_LOST_AND_FORGOTTEN_NORMAL",
        "run": {"floor": 12},
        "combat": {
            "energy": 1,
            "enemies": [
                {
                    "combat_id": 1,
                    "current_hp": 84,
                    "max_hp": 93,
                    "intent": {"total_damage": 12, "damage": 12, "intent_type": "Attack"},
                },
                {
                    "combat_id": 2,
                    "current_hp": 76,
                    "max_hp": 106,
                    "intent": {"total_damage": 15, "damage": 15, "intent_type": "Attack"},
                },
            ],
        },
        "player": {"hp": 93, "current_hp": 93, "max_hp": 93, "block": 0},
    }
    legal_actions = [
        {
            "action_id": "play_card_strike",
            "kind": "play_card",
            "target_combat_id": 1,
            "card": {
                "id": "CARD.STRIKE",
                "title": "Strike",
                "type": "Attack",
                "cost": 1,
                "damage": 6,
                "total_damage": 6,
                "preview_damage": 6,
                "card_effect_profile": {"semantic_tags": ["attack", "damage"], "training_tags": []},
            },
            "semantic": {"damage": 6, "roles": ["attack", "damage"]},
        },
        {
            "action_id": "play_card_defend",
            "kind": "play_card",
            "card": {
                "id": "CARD.DEFEND",
                "title": "Defend",
                "type": "Skill",
                "cost": 1,
                "block": 5,
                "total_block": 5,
                "preview_block": 5,
                "card_effect_profile": {"semantic_tags": ["block"], "training_tags": []},
            },
            "semantic": {"block": 5, "roles": ["block"]},
        },
    ]
    mask = np.array([1, 1], dtype=np.float32)
    search_stats: dict = {}

    with patch.object(trainer_stub, "_is_kaiser_encounter_context", return_value=False), patch.object(
        trainer_stub, "_is_insatiable_encounter_context", return_value=False
    ), patch.object(
        trainer_stub, "_combat_encounter_tier_from_raw", return_value="normal"
    ), patch.object(
        trainer_stub, "_combat_energy", return_value=1.0
    ), patch.object(
        trainer_stub, "_is_action_confirmed_lethal", return_value=False
    ):
        new_idx = trainer_stub._apply_combat_action_hard_guards(
            action_idx=0,
            legal_actions=legal_actions,
            action_mask=mask,
            raw_obs=raw_obs,
            boss_ctx={},
            encounter="ENCOUNTER.THE_LOST_AND_FORGOTTEN_NORMAL",
            search_stats=search_stats,
        )

    assert new_idx == 0
    assert search_stats["combat_quality_survival_non_endturn_guard_available"] == 1.0
    assert search_stats["combat_quality_survival_non_endturn_guard_progress_exemption"] == 1.0
    assert search_stats["combat_quality_survival_non_endturn_guard_applied"] == 0.0


def test_survival_non_endturn_guard_skips_low_value_block_when_threat_remains_unsolved(trainer_stub):
    """Do not replace progress with 5 block when the hit remains largely unsolved.

    Regression for Owl/Fogmog/Bowlbugs hard-normal traces: high-HP, non-critical
    hallway turns selected an attack/setup card, but survival_non_endturn rewrote
    it to Defend.  A 5-block card did not solve 22-33 incoming and only removed
    tempo, increasing total fight damage.  The guard should still block critical
    deaths, but in this non-critical shape it must keep progress.
    """
    raw_obs = {
        "encounter": "ENCOUNTER.OWL_MAGISTRATE_NORMAL",
        "run": {"floor": 12},
        "combat": {
            "energy": 5,
            "enemies": [
                {
                    "combat_id": 1,
                    "current_hp": 96,
                    "max_hp": 96,
                    "intent": {"total_damage": 33, "damage": 33, "intent_type": "Attack"},
                }
            ],
        },
        "player": {"hp": 66, "current_hp": 66, "max_hp": 80, "block": 0},
    }
    legal_actions = [
        {
            "action_id": "play_card_pommel",
            "kind": "play_card",
            "target_combat_id": 1,
            "card": {
                "id": "CARD.POMMEL_STRIKE",
                "title": "痛击",
                "type": "Attack",
                "cost": 2,
                "damage": 6,
                "total_damage": 6,
                "preview_damage": 6,
                "card_effect_profile": {"semantic_tags": ["attack", "damage", "setup"], "training_tags": []},
            },
            "semantic": {"damage": 6, "roles": ["attack", "damage", "setup"], "immediate_impact": 12},
        },
        {
            "action_id": "play_card_defend",
            "kind": "play_card",
            "card": {
                "id": "CARD.DEFEND",
                "title": "防御",
                "type": "Skill",
                "cost": 1,
                "block": 5,
                "total_block": 5,
                "preview_block": 5,
                "card_effect_profile": {"semantic_tags": ["block"], "training_tags": []},
            },
            "semantic": {"block": 5, "roles": ["block"]},
        },
    ]
    mask = np.array([1, 1], dtype=np.float32)
    search_stats: dict = {}

    with patch.object(trainer_stub, "_is_kaiser_encounter_context", return_value=False), patch.object(
        trainer_stub, "_is_insatiable_encounter_context", return_value=False
    ), patch.object(
        trainer_stub, "_combat_encounter_tier_from_raw", return_value="normal"
    ), patch.object(
        trainer_stub, "_combat_energy", return_value=5.0
    ), patch.object(
        trainer_stub, "_is_action_confirmed_lethal", return_value=False
    ):
        new_idx = trainer_stub._apply_combat_action_hard_guards(
            action_idx=0,
            legal_actions=legal_actions,
            action_mask=mask,
            raw_obs=raw_obs,
            boss_ctx={},
            encounter="ENCOUNTER.OWL_MAGISTRATE_NORMAL",
            search_stats=search_stats,
        )

    assert new_idx == 0
    assert search_stats["combat_quality_survival_non_endturn_guard_available"] == 1.0
    assert search_stats["combat_quality_survival_non_endturn_guard_progress_exemption"] == 1.0
    assert search_stats["combat_quality_survival_non_endturn_guard_low_value_block_skip"] == 1.0
    assert search_stats["combat_quality_survival_non_endturn_guard_applied"] == 0.0


def test_survival_non_endturn_guard_preserves_progress_when_followup_block_is_affordable(trainer_stub):
    """Do not treat a high-energy hallway opener as if it ended the turn.

    Regression for Mytes/Fabricator/Ovicopter-style sandbox traces where the
    policy selected an affordable attack at high HP with enough energy left to
    defend afterward, but survival_non_endturn rewrote the opener to Defend.
    That teaches low-tempo pure block and makes hard-normal fights drag.
    """

    raw_obs = {
        "encounter": "ENCOUNTER.MYTES_NORMAL",
        "run": {"floor": 10},
        "combat": {
            "energy": 4,
            "enemies": [
                {
                    "combat_id": 1,
                    "current_hp": 48,
                    "max_hp": 48,
                    "intent": {"total_damage": 19, "damage": 19, "intent_type": "Attack"},
                }
            ],
        },
        "player": {"hp": 53, "current_hp": 53, "max_hp": 80, "block": 0},
    }
    legal_actions = [
        {
            "action_id": "play_card_brawl",
            "kind": "play_card",
            "target_combat_id": 1,
            "card": {
                "id": "CARD.BRAWL",
                "title": "Brawl",
                "type": "Attack",
                "cost": 1,
                "damage": 7,
                "total_damage": 7,
                "preview_damage": 7,
                "card_effect_profile": {"semantic_tags": ["attack", "damage"], "training_tags": []},
            },
            "semantic": {"damage": 7, "roles": ["attack", "damage"]},
        },
        {
            "action_id": "play_card_defend",
            "kind": "play_card",
            "card": {
                "id": "CARD.DEFEND",
                "title": "Defend",
                "type": "Skill",
                "cost": 1,
                "block": 5,
                "total_block": 5,
                "preview_block": 5,
                "card_effect_profile": {"semantic_tags": ["block"], "training_tags": []},
            },
            "semantic": {"block": 5, "roles": ["block"]},
        },
    ]
    mask = np.array([1, 1], dtype=np.float32)
    search_stats: dict = {}

    with patch.object(trainer_stub, "_is_kaiser_encounter_context", return_value=False), patch.object(
        trainer_stub, "_is_insatiable_encounter_context", return_value=False
    ), patch.object(
        trainer_stub, "_combat_encounter_tier_from_raw", return_value="normal"
    ), patch.object(
        trainer_stub, "_combat_energy", return_value=4.0
    ), patch.object(
        trainer_stub, "_is_action_confirmed_lethal", return_value=False
    ):
        new_idx = trainer_stub._apply_combat_action_hard_guards(
            action_idx=0,
            legal_actions=legal_actions,
            action_mask=mask,
            raw_obs=raw_obs,
            boss_ctx={},
            encounter="ENCOUNTER.MYTES_NORMAL",
            search_stats=search_stats,
        )

    assert new_idx == 0
    assert search_stats["combat_quality_survival_non_endturn_guard_available"] == 1.0
    assert search_stats["combat_quality_survival_non_endturn_guard_progress_exemption"] == 1.0
    assert search_stats["combat_quality_survival_non_endturn_guard_applied"] == 0.0


def test_survival_non_endturn_guard_overrides_boss_strike_to_defend(trainer_stub):
    """Boss high-pressure turns must not spend the action on a non-lethal Strike
    when an affordable Defend is legal.
    """
    raw_obs = {
        "encounter": "ENCOUNTER.LAGAVULIN_MATRIARCH_BOSS",
        "run": {"floor": 17, "room_type": "boss"},
        "combat": {"energy": 1, "enemies": [{"current_hp": 80, "intent": {"total_damage": 18}}]},
        # Defend(5) must be enough to turn a lethal hit into survival.  A
        # previous regression rewrote hp=13/incoming=25 to Defend even though
        # the player still died; keep this positive case survivable.
        "player": {"hp": 14, "current_hp": 14, "max_hp": 91, "block": 0},
    }
    legal_actions = [
        {
            "action_id": "play_card_strike",
            "kind": "play_card",
            "card": {"id": "CARD.STRIKE", "title": "Strike", "cost": 1},
            "semantic": {"damage": 6},
        },
        {
            "action_id": "play_card_defend",
            "kind": "play_card",
            "label": "Defend",
            "card": {"id": "CARD.DEFEND", "title": "Defend", "cost": 1},
            "semantic": {"block": 5},
        },
    ]
    mask = np.array([1, 1], dtype=np.float32)
    search_stats: dict = {}

    with patch.object(trainer_stub, "_is_kaiser_encounter_context", return_value=False), patch.object(
        trainer_stub, "_is_insatiable_encounter_context", return_value=False
    ), patch.object(
        trainer_stub, "_combat_encounter_tier_from_raw", return_value="boss"
    ), patch.object(
        trainer_stub, "_combat_energy", return_value=1.0
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
    assert search_stats["combat_quality_survival_non_endturn_guard_available"] == 1.0
    assert search_stats["combat_quality_survival_non_endturn_guard_applied"] == 1.0
    assert search_stats["combat_quality_survival_non_endturn_guard_override"] == 1.0
    assert search_stats["combat_quality_survival_non_endturn_guard_candidate_count"] == 1.0
    assert search_stats["combat_quality_hard_guard_override_any"] == 1.0


def test_survival_non_endturn_guard_skips_insufficient_block_candidate(trainer_stub):
    """Non-EndTurn survival guard must not convert attacks into dead Defends."""
    raw_obs = {
        "encounter": "ENCOUNTER.LAGAVULIN_MATRIARCH_BOSS",
        "run": {"floor": 17, "room_type": "boss"},
        "combat": {"energy": 1, "enemies": [{"current_hp": 80, "intent": {"total_damage": 25}}]},
        "player": {"hp": 13, "current_hp": 13, "max_hp": 91, "block": 0},
    }
    legal_actions = [
        {
            "action_id": "play_card_strike",
            "kind": "play_card",
            "card": {"id": "CARD.STRIKE", "title": "Strike", "cost": 1},
            "semantic": {"damage": 6},
        },
        {
            "action_id": "play_card_defend",
            "kind": "play_card",
            "label": "Defend",
            "card": {"id": "CARD.DEFEND", "title": "Defend", "cost": 1},
            "semantic": {"block": 5},
        },
    ]
    mask = np.array([1, 1], dtype=np.float32)
    search_stats: dict = {}

    with patch.object(trainer_stub, "_is_kaiser_encounter_context", return_value=False), patch.object(
        trainer_stub, "_is_insatiable_encounter_context", return_value=False
    ), patch.object(
        trainer_stub, "_combat_encounter_tier_from_raw", return_value="boss"
    ), patch.object(
        trainer_stub, "_combat_energy", return_value=1.0
    ), patch.object(
        trainer_stub, "_is_action_confirmed_lethal", return_value=False
    ):
        new_idx = trainer_stub._apply_survival_non_endturn_guard(
            action_idx=0,
            legal_count=len(legal_actions),
            legal_actions=legal_actions,
            mask_np=mask,
            raw_obs=raw_obs,
            encounter="ENCOUNTER.LAGAVULIN_MATRIARCH_BOSS",
            search_stats=search_stats,
        )

    assert new_idx == 0
    assert search_stats["combat_quality_survival_non_endturn_guard_available"] == 1.0
    assert search_stats["combat_quality_survival_non_endturn_guard_insufficient_candidate"] == 1.0
    assert search_stats["combat_quality_survival_non_endturn_guard_candidate_count"] == 0.0
    assert search_stats["combat_quality_survival_non_endturn_guard_no_alternative"] == 1.0
    assert search_stats.get("combat_quality_survival_non_endturn_guard_applied", 0.0) == 0.0


def test_survival_non_endturn_guard_catches_boss_mid_hp_medium_threat(trainer_stub):
    """Medium boss threat must also rewrite non-lethal attacks, not only End Turn."""
    raw_obs = {
        "encounter": "ENCOUNTER.LAGAVULIN_MATRIARCH_BOSS",
        "run": {"floor": 17, "room_type": "boss"},
        "combat": {"energy": 1, "enemies": [{"current_hp": 120, "intent": {"total_damage": 6}}]},
        "player": {"hp": 43, "current_hp": 43, "max_hp": 91, "block": 0},
    }
    legal_actions = [
        {
            "action_id": "play_card_strike",
            "kind": "play_card",
            "card": {"id": "CARD.STRIKE", "title": "Strike", "cost": 1},
            "semantic": {"damage": 6},
        },
        {
            "action_id": "play_card_defend",
            "kind": "play_card",
            "label": "Defend",
            "card": {"id": "CARD.DEFEND", "title": "Defend", "cost": 1},
            "semantic": {"block": 5},
        },
    ]
    mask = np.array([1, 1], dtype=np.float32)
    search_stats: dict = {}

    with patch.object(trainer_stub, "_is_kaiser_encounter_context", return_value=False), patch.object(
        trainer_stub, "_is_insatiable_encounter_context", return_value=False
    ), patch.object(
        trainer_stub, "_combat_encounter_tier_from_raw", return_value="boss"
    ), patch.object(
        trainer_stub, "_combat_energy", return_value=1.0
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
    assert search_stats["combat_quality_survival_non_endturn_guard_available"] == 1.0
    assert search_stats["combat_quality_survival_non_endturn_guard_applied"] == 1.0
    assert search_stats["combat_quality_survival_non_endturn_guard_override"] == 1.0


def test_survival_non_endturn_guard_overrides_late_weak_strike_to_defend(trainer_stub):
    """Weak hallways are hallway combats too; floor-9 weak deaths need the same
    non-EndTurn survival correction as normal hallways.
    """
    raw_obs = {
        "encounter": "ENCOUNTER.GREMLIN_MERC_WEAK",
        "run": {"floor": 9},
        "combat": {"energy": 1, "enemies": [{"current_hp": 28, "intent": {"total_damage": 14}}]},
        "player": {"hp": 14, "current_hp": 14, "max_hp": 80, "block": 0},
    }
    legal_actions = [
        {
            "action_id": "play_card_strike",
            "kind": "play_card",
            "card": {"id": "CARD.STRIKE", "title": "Strike", "cost": 1},
            "semantic": {"damage": 6},
        },
        {
            "action_id": "play_card_defend",
            "kind": "play_card",
            "card": {"id": "CARD.DEFEND", "title": "Defend", "cost": 1},
            "semantic": {"block": 5},
        },
    ]
    mask = np.array([1, 1], dtype=np.float32)
    search_stats: dict = {}

    with patch.object(trainer_stub, "_is_kaiser_encounter_context", return_value=False), patch.object(
        trainer_stub, "_is_insatiable_encounter_context", return_value=False
    ), patch.object(
        trainer_stub, "_combat_encounter_tier_from_raw", return_value="weak"
    ), patch.object(
        trainer_stub, "_combat_energy", return_value=1.0
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
    assert search_stats["combat_quality_survival_non_endturn_guard_available"] == 1.0
    assert search_stats["combat_quality_survival_non_endturn_guard_applied"] == 1.0
    assert search_stats["combat_quality_survival_non_endturn_guard_override"] == 1.0


def test_survival_non_endturn_guard_catches_floor14_medium_threat(trainer_stub):
    """Late hallway non-lethal attacks should preserve HP before the Act1 boss."""
    raw_obs = {
        "encounter": "ENCOUNTER.GREMLIN_MERC_NORMAL",
        "run": {"floor": 14},
        "combat": {"energy": 1, "enemies": [{"current_hp": 35, "intent": {"total_damage": 5}}]},
        "player": {"hp": 45, "current_hp": 45, "max_hp": 91, "block": 0},
    }
    legal_actions = [
        {
            "action_id": "play_card_strike",
            "kind": "play_card",
            "card": {"id": "CARD.STRIKE", "title": "Strike", "cost": 1},
            "semantic": {"damage": 6},
        },
        {
            "action_id": "play_card_defend",
            "kind": "play_card",
            "card": {"id": "CARD.DEFEND", "title": "Defend", "cost": 1},
            "semantic": {"block": 5},
        },
    ]
    mask = np.array([1, 1], dtype=np.float32)
    search_stats: dict = {}

    with patch.object(trainer_stub, "_is_kaiser_encounter_context", return_value=False), patch.object(
        trainer_stub, "_is_insatiable_encounter_context", return_value=False
    ), patch.object(
        trainer_stub, "_combat_encounter_tier_from_raw", return_value="normal"
    ), patch.object(
        trainer_stub, "_combat_energy", return_value=1.0
    ), patch.object(
        trainer_stub, "_is_action_confirmed_lethal", return_value=False
    ):
        new_idx = trainer_stub._apply_combat_action_hard_guards(
            action_idx=0,
            legal_actions=legal_actions,
            action_mask=mask,
            raw_obs=raw_obs,
            boss_ctx={},
            encounter="ENCOUNTER.GREMLIN_MERC_NORMAL",
            search_stats=search_stats,
        )

    assert new_idx == 1
    assert search_stats["combat_quality_survival_non_endturn_guard_available"] == 1.0
    assert search_stats["combat_quality_survival_non_endturn_guard_applied"] == 1.0
    assert search_stats["combat_quality_survival_non_endturn_guard_override"] == 1.0


def test_survival_non_endturn_guard_keeps_boss_high_damage_progress_card(trainer_stub):
    """Medium boss pressure should not convert a meaningful race card into
    passive block.  This is the Act1-clear regression that produced boss
    traces with very low damage dealt despite survival guards firing.
    """
    raw_obs = {
        "encounter": "ENCOUNTER.LAGAVULIN_MATRIARCH_BOSS",
        "run": {"floor": 17, "room_type": "boss"},
        "combat": {"energy": 2, "enemies": [{"current_hp": 120, "intent": {"total_damage": 6}}]},
        "player": {"hp": 45, "current_hp": 45, "max_hp": 91, "block": 0},
    }
    legal_actions = [
        {
            "action_id": "play_card_heavy_blade",
            "kind": "play_card",
            "card": {"id": "CARD.HEAVY_BLADE", "title": "Heavy Blade", "cost": 2},
            "semantic": {"damage": 18, "roles": ["attack", "damage"]},
        },
        {
            "action_id": "play_card_defend",
            "kind": "play_card",
            "card": {"id": "CARD.DEFEND", "title": "Defend", "cost": 1},
            "semantic": {"block": 5, "roles": ["block"]},
        },
    ]
    mask = np.array([1, 1], dtype=np.float32)
    search_stats: dict = {}

    with patch.object(trainer_stub, "_is_kaiser_encounter_context", return_value=False), patch.object(
        trainer_stub, "_is_insatiable_encounter_context", return_value=False
    ), patch.object(
        trainer_stub, "_combat_encounter_tier_from_raw", return_value="boss"
    ), patch.object(
        trainer_stub, "_combat_energy", return_value=2.0
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
    assert search_stats["combat_quality_survival_non_endturn_guard_available"] == 1.0
    assert search_stats["combat_quality_survival_non_endturn_guard_progress_exemption"] == 1.0
    assert search_stats["combat_quality_survival_non_endturn_guard_applied"] == 0.0


def test_survival_non_endturn_guard_still_blocks_critical_boss_high_damage(trainer_stub):
    """The race-card exemption is non-critical only; near-lethal incoming still
    gets rewritten to an available defensive action.
    """
    raw_obs = {
        "encounter": "ENCOUNTER.LAGAVULIN_MATRIARCH_BOSS",
        "run": {"floor": 17, "room_type": "boss"},
        "combat": {"energy": 2, "enemies": [{"current_hp": 120, "intent": {"total_damage": 14}}]},
        "player": {"hp": 10, "current_hp": 10, "max_hp": 91, "block": 0},
    }
    legal_actions = [
        {
            "action_id": "play_card_heavy_blade",
            "kind": "play_card",
            "card": {"id": "CARD.HEAVY_BLADE", "title": "Heavy Blade", "cost": 2},
            "semantic": {"damage": 18, "roles": ["attack", "damage"]},
        },
        {
            "action_id": "play_card_defend_plus",
            "kind": "play_card",
            "card": {"id": "CARD.DEFEND", "title": "Defend+", "cost": 1},
            "semantic": {"block": 8, "roles": ["block"]},
        },
    ]
    mask = np.array([1, 1], dtype=np.float32)
    search_stats: dict = {}

    with patch.object(trainer_stub, "_is_kaiser_encounter_context", return_value=False), patch.object(
        trainer_stub, "_is_insatiable_encounter_context", return_value=False
    ), patch.object(
        trainer_stub, "_combat_encounter_tier_from_raw", return_value="boss"
    ), patch.object(
        trainer_stub, "_combat_energy", return_value=2.0
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
    assert search_stats["combat_quality_survival_non_endturn_guard_progress_exemption"] == 0.0
    assert search_stats["combat_quality_survival_non_endturn_guard_applied"] == 1.0


def test_survival_non_endturn_guard_prefers_boss_race_attack_candidate(trainer_stub):
    """If the selected boss action is weak but a meaningful race attack is legal,
    the oracle correction should progress the kill instead of always selecting
    Defend under medium, non-critical pressure.
    """
    raw_obs = {
        "encounter": "ENCOUNTER.LAGAVULIN_MATRIARCH_BOSS",
        "run": {"floor": 17, "room_type": "boss"},
        "combat": {"energy": 2, "enemies": [{"current_hp": 120, "intent": {"total_damage": 6}}]},
        "player": {"hp": 43, "current_hp": 43, "max_hp": 91, "block": 0},
    }
    legal_actions = [
        {
            "action_id": "play_card_strike",
            "kind": "play_card",
            "card": {"id": "CARD.STRIKE", "title": "Strike", "cost": 1},
            "semantic": {"damage": 6, "roles": ["attack", "damage"]},
        },
        {
            "action_id": "play_card_heavy_blade",
            "kind": "play_card",
            "card": {"id": "CARD.HEAVY_BLADE", "title": "Heavy Blade", "cost": 2},
            "semantic": {"damage": 18, "roles": ["attack", "damage"]},
        },
        {
            "action_id": "play_card_defend",
            "kind": "play_card",
            "card": {"id": "CARD.DEFEND", "title": "Defend", "cost": 1},
            "semantic": {"block": 5, "roles": ["block"]},
        },
    ]
    mask = np.array([1, 1, 1], dtype=np.float32)
    search_stats: dict = {}

    with patch.object(trainer_stub, "_is_kaiser_encounter_context", return_value=False), patch.object(
        trainer_stub, "_is_insatiable_encounter_context", return_value=False
    ), patch.object(
        trainer_stub, "_combat_encounter_tier_from_raw", return_value="boss"
    ), patch.object(
        trainer_stub, "_combat_energy", return_value=2.0
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
    assert search_stats["combat_quality_survival_non_endturn_guard_applied"] == 1.0
    assert search_stats["combat_quality_survival_non_endturn_guard_candidate_count"] == 2.0


def test_survival_non_endturn_guard_keeps_floor14_high_damage_progress_card(trainer_stub):
    """Late normal fights need HP preservation, but not at the cost of throwing
    away a meaningful race card under non-critical pressure.
    """
    raw_obs = {
        "encounter": "ENCOUNTER.GREMLIN_MERC_NORMAL",
        "run": {"floor": 14},
        "combat": {"energy": 2, "enemies": [{"current_hp": 35, "intent": {"total_damage": 5}}]},
        "player": {"hp": 45, "current_hp": 45, "max_hp": 91, "block": 0},
    }
    legal_actions = [
        {
            "action_id": "play_card_big_attack",
            "kind": "play_card",
            "card": {"id": "CARD.BIG_ATTACK", "title": "Big Attack", "cost": 2},
            "semantic": {"damage": 12, "roles": ["attack", "damage"]},
        },
        {
            "action_id": "play_card_defend",
            "kind": "play_card",
            "card": {"id": "CARD.DEFEND", "title": "Defend", "cost": 1},
            "semantic": {"block": 5, "roles": ["block"]},
        },
    ]
    mask = np.array([1, 1], dtype=np.float32)
    search_stats: dict = {}

    with patch.object(trainer_stub, "_is_kaiser_encounter_context", return_value=False), patch.object(
        trainer_stub, "_is_insatiable_encounter_context", return_value=False
    ), patch.object(
        trainer_stub, "_combat_encounter_tier_from_raw", return_value="normal"
    ), patch.object(
        trainer_stub, "_combat_energy", return_value=2.0
    ), patch.object(
        trainer_stub, "_is_action_confirmed_lethal", return_value=False
    ):
        new_idx = trainer_stub._apply_combat_action_hard_guards(
            action_idx=0,
            legal_actions=legal_actions,
            action_mask=mask,
            raw_obs=raw_obs,
            boss_ctx={},
            encounter="ENCOUNTER.GREMLIN_MERC_NORMAL",
            search_stats=search_stats,
        )

    assert new_idx == 0
    assert search_stats["combat_quality_survival_non_endturn_guard_available"] == 1.0
    assert search_stats["combat_quality_survival_non_endturn_guard_progress_exemption"] == 1.0
    assert search_stats["combat_quality_survival_non_endturn_guard_applied"] == 0.0


def test_survival_non_endturn_guard_keeps_scaling_enemy_finish(trainer_stub):
    """Late Act1 scaling enemies must not turn a kill into passive block.

    Damp Cultist/Ritual traces showed the non-EndTurn survival guard rewriting
    kill-capable attacks into self-defense.  That preserves a few HP on the
    current turn but lets Ritual/Strength snowball and causes deterministic
    late hallway deaths before the boss.
    """
    raw_obs = {
        "encounter": "ENCOUNTER.DAMP_CULTIST_NORMAL",
        "run": {"floor": 13},
        "combat": {
            "energy": 1,
            "enemies": [
                {
                    "combat_id": 2,
                    "model_id": "MONSTER.DAMP_CULTIST",
                    "current_hp": 9,
                    "max_hp": 52,
                    "intent": {"total_damage": 16},
                    "powers": [
                        {"id": "RITUAL_POWER", "amount": 5},
                        {"id": "STRENGTH_POWER", "amount": 15},
                    ],
                }
            ],
        },
        "player": {"hp": 24, "current_hp": 24, "max_hp": 91, "block": 0},
    }
    legal_actions = [
        {
            "action_id": "play_card_target_2",
            "kind": "play_card",
            "target_combat_id": 2,
            "card": {"id": "CARD.STRIKE", "title": "Strike", "cost": 1},
            "semantic": {"damage": 9, "roles": ["attack", "damage"]},
        },
        {
            "action_id": "play_card_self",
            "kind": "play_card",
            "card": {"id": "CARD.DEFEND", "title": "Defend", "cost": 1},
            "semantic": {"block": 5, "roles": ["block"]},
        },
    ]
    mask = np.array([1, 1], dtype=np.float32)
    search_stats: dict = {}

    with patch.object(trainer_stub, "_is_kaiser_encounter_context", return_value=False), patch.object(
        trainer_stub, "_is_insatiable_encounter_context", return_value=False
    ), patch.object(
        trainer_stub, "_combat_encounter_tier_from_raw", return_value="normal"
    ), patch.object(
        trainer_stub, "_combat_energy", return_value=1.0
    ), patch.object(
        trainer_stub, "_is_action_confirmed_lethal", return_value=False
    ):
        new_idx = trainer_stub._apply_combat_action_hard_guards(
            action_idx=0,
            legal_actions=legal_actions,
            action_mask=mask,
            raw_obs=raw_obs,
            boss_ctx={},
            encounter="ENCOUNTER.DAMP_CULTIST_NORMAL",
            search_stats=search_stats,
        )

    assert new_idx == 0
    assert search_stats["combat_quality_survival_non_endturn_guard_available"] == 1.0
    assert search_stats["combat_quality_survival_non_endturn_guard_scaling_enemy_exemption"] == 1.0
    assert search_stats["combat_quality_survival_non_endturn_guard_progress_exemption"] == 1.0
    assert search_stats["combat_quality_survival_non_endturn_guard_applied"] == 0.0


def test_survival_non_endturn_guard_prefers_scaling_enemy_finish_candidate(trainer_stub):
    """When selected action is weak, pick the legal kill on a scaling enemy.

    This is the oracle-correction target rewrite we want replay to imitate:
    do not teach the policy that a low-impact setup/block choice was correct
    while a Ritual enemy could be finished.
    """
    raw_obs = {
        "encounter": "ENCOUNTER.DAMP_CULTIST_NORMAL",
        "run": {"floor": 13},
        "combat": {
            "energy": 1,
            "enemies": [
                {
                    "combat_id": 2,
                    "model_id": "MONSTER.DAMP_CULTIST",
                    "current_hp": 9,
                    "max_hp": 52,
                    "intent": {"total_damage": 16},
                    "powers": [
                        {"id": "RITUAL_POWER", "amount": 5},
                        {"id": "STRENGTH_POWER", "amount": 15},
                    ],
                }
            ],
        },
        "player": {"hp": 35, "current_hp": 35, "max_hp": 91, "block": 0},
    }
    legal_actions = [
        {
            "action_id": "play_card_low_impact_setup",
            "kind": "play_card",
            "card": {"id": "CARD.SETUP_TEST", "title": "Setup", "cost": 1},
            "semantic": {"draw": 1, "roles": ["draw"]},
        },
        {
            "action_id": "play_card_target_2",
            "kind": "play_card",
            "target_combat_id": 2,
            "card": {"id": "CARD.STRIKE", "title": "Strike", "cost": 1},
            "semantic": {"damage": 9, "roles": ["attack", "damage"]},
        },
        {
            "action_id": "play_card_self",
            "kind": "play_card",
            "card": {"id": "CARD.DEFEND", "title": "Defend", "cost": 1},
            "semantic": {"block": 5, "roles": ["block"]},
        },
    ]
    mask = np.array([1, 1, 1], dtype=np.float32)
    search_stats: dict = {}

    with patch.object(trainer_stub, "_is_kaiser_encounter_context", return_value=False), patch.object(
        trainer_stub, "_is_insatiable_encounter_context", return_value=False
    ), patch.object(
        trainer_stub, "_combat_encounter_tier_from_raw", return_value="normal"
    ), patch.object(
        trainer_stub, "_combat_energy", return_value=1.0
    ), patch.object(
        trainer_stub, "_is_action_confirmed_lethal", return_value=False
    ):
        new_idx = trainer_stub._apply_combat_action_hard_guards(
            action_idx=0,
            legal_actions=legal_actions,
            action_mask=mask,
            raw_obs=raw_obs,
            boss_ctx={},
            encounter="ENCOUNTER.DAMP_CULTIST_NORMAL",
            search_stats=search_stats,
        )

    assert new_idx == 1
    assert search_stats["combat_quality_survival_non_endturn_guard_available"] == 1.0
    assert search_stats["combat_quality_survival_non_endturn_guard_scaling_enemy_candidate"] == 1.0
    assert search_stats["combat_quality_survival_non_endturn_guard_applied"] == 1.0


def test_survival_non_endturn_guard_still_blocks_scaling_enemy_when_attack_not_enough_and_critical(trainer_stub):
    """Scaling-enemy race logic must not override true near-lethal survival.

    If the attack cannot kill/stabilize and HP is critically low, the guard
    should still choose available block rather than blindly racing.
    """
    raw_obs = {
        "encounter": "ENCOUNTER.DAMP_CULTIST_NORMAL",
        "run": {"floor": 13},
        "combat": {
            "energy": 1,
            "enemies": [
                {
                    "combat_id": 2,
                    "model_id": "MONSTER.DAMP_CULTIST",
                    "current_hp": 20,
                    "max_hp": 52,
                    "intent": {"total_damage": 21},
                    "powers": [
                        {"id": "RITUAL_POWER", "amount": 5},
                        {"id": "STRENGTH_POWER", "amount": 20},
                    ],
                }
            ],
        },
        # Defend+(8) leaves 13 incoming; hp must be strictly > 13 because
        # damage equal to current HP is death.
        "player": {"hp": 14, "current_hp": 14, "max_hp": 91, "block": 0},
    }
    legal_actions = [
        {
            "action_id": "play_card_target_2",
            "kind": "play_card",
            "target_combat_id": 2,
            "card": {"id": "CARD.STRIKE", "title": "Strike", "cost": 1},
            "semantic": {"damage": 6, "roles": ["attack", "damage"]},
        },
        {
            "action_id": "play_card_self",
            "kind": "play_card",
            "card": {"id": "CARD.DEFEND", "title": "Defend+", "cost": 1},
            "semantic": {"block": 8, "roles": ["block"]},
        },
    ]
    mask = np.array([1, 1], dtype=np.float32)
    search_stats: dict = {}

    with patch.object(trainer_stub, "_is_kaiser_encounter_context", return_value=False), patch.object(
        trainer_stub, "_is_insatiable_encounter_context", return_value=False
    ), patch.object(
        trainer_stub, "_combat_encounter_tier_from_raw", return_value="normal"
    ), patch.object(
        trainer_stub, "_combat_energy", return_value=1.0
    ), patch.object(
        trainer_stub, "_is_action_confirmed_lethal", return_value=False
    ):
        new_idx = trainer_stub._apply_combat_action_hard_guards(
            action_idx=0,
            legal_actions=legal_actions,
            action_mask=mask,
            raw_obs=raw_obs,
            boss_ctx={},
            encounter="ENCOUNTER.DAMP_CULTIST_NORMAL",
            search_stats=search_stats,
        )

    assert new_idx == 1
    assert search_stats["combat_quality_survival_non_endturn_guard_scaling_enemy_exemption"] == 0.0
    assert search_stats["combat_quality_survival_non_endturn_guard_applied"] == 1.0


def test_survival_non_endturn_guard_keeps_selected_lethal(trainer_stub):
    """The guard must never turn a confirmed kill into defense."""
    raw_obs = {
        "encounter": "ENCOUNTER.LAGAVULIN_MATRIARCH_BOSS",
        "run": {"floor": 17, "room_type": "boss"},
        "combat": {"energy": 1, "enemies": [{"current_hp": 4, "intent": {"total_damage": 18}}]},
        "player": {"hp": 14, "current_hp": 14, "max_hp": 91, "block": 0},
    }
    legal_actions = [
        {"action_id": "play_card_strike", "kind": "play_card", "card": {"id": "CARD.STRIKE", "cost": 1}, "semantic": {"damage": 6}},
        {"action_id": "play_card_defend", "kind": "play_card", "card": {"id": "CARD.DEFEND", "cost": 1}, "semantic": {"block": 5}},
    ]
    mask = np.array([1, 1], dtype=np.float32)
    search_stats: dict = {}

    with patch.object(trainer_stub, "_is_kaiser_encounter_context", return_value=False), patch.object(
        trainer_stub, "_is_insatiable_encounter_context", return_value=False
    ), patch.object(
        trainer_stub, "_combat_encounter_tier_from_raw", return_value="boss"
    ), patch.object(
        trainer_stub, "_combat_energy", return_value=1.0
    ), patch.object(
        trainer_stub,
        "_is_action_confirmed_lethal",
        side_effect=lambda action, raw_obs=None: isinstance(action, dict)
        and action.get("action_id") == "play_card_strike",
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
    assert search_stats["combat_quality_survival_non_endturn_guard_available"] == 1.0
    assert search_stats["combat_quality_survival_non_endturn_guard_lethal_exemption"] == 1.0
    assert search_stats["combat_quality_survival_non_endturn_guard_applied"] == 0.0
    assert search_stats["combat_quality_hard_guard_override_any"] == 0.0


def test_survival_non_endturn_guard_keeps_already_protective_defend(trainer_stub):
    """Do not churn a good defensive choice into another action."""
    raw_obs = {
        "encounter": "ENCOUNTER.LAGAVULIN_MATRIARCH_BOSS",
        "run": {"floor": 17, "room_type": "boss"},
        "combat": {"energy": 1, "enemies": [{"current_hp": 80, "intent": {"total_damage": 18}}]},
        "player": {"hp": 14, "current_hp": 14, "max_hp": 91, "block": 0},
    }
    legal_actions = [
        {"action_id": "play_card_defend", "kind": "play_card", "card": {"id": "CARD.DEFEND", "cost": 1}, "semantic": {"block": 5}},
        {"action_id": "play_card_strike", "kind": "play_card", "card": {"id": "CARD.STRIKE", "cost": 1}, "semantic": {"damage": 6}},
    ]
    mask = np.array([1, 1], dtype=np.float32)
    search_stats: dict = {}

    with patch.object(trainer_stub, "_is_kaiser_encounter_context", return_value=False), patch.object(
        trainer_stub, "_is_insatiable_encounter_context", return_value=False
    ), patch.object(
        trainer_stub, "_combat_encounter_tier_from_raw", return_value="boss"
    ), patch.object(
        trainer_stub, "_combat_energy", return_value=1.0
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
    assert search_stats["combat_quality_survival_non_endturn_guard_applied"] == 0.0
    assert search_stats["combat_quality_survival_non_endturn_guard_override"] == 0.0
    assert search_stats["combat_quality_hard_guard_override_any"] == 0.0


def test_survival_non_endturn_guard_dormant_on_early_safe_normal(trainer_stub):
    """Early safe hallway attacks stay policy-controlled."""
    raw_obs = {
        "encounter": "ENCOUNTER.CORPSE_SLUGS_NORMAL",
        "run": {"floor": 5},
        "combat": {"energy": 1, "enemies": [{"current_hp": 40, "intent": {"total_damage": 4}}]},
        "player": {"hp": 70, "current_hp": 70, "max_hp": 91, "block": 0},
    }
    legal_actions = [
        {"action_id": "play_card_strike", "kind": "play_card", "card": {"id": "CARD.STRIKE", "cost": 1}, "semantic": {"damage": 6}},
        {"action_id": "play_card_defend", "kind": "play_card", "card": {"id": "CARD.DEFEND", "cost": 1}, "semantic": {"block": 5}},
    ]
    mask = np.array([1, 1], dtype=np.float32)
    search_stats: dict = {}

    with patch.object(trainer_stub, "_is_kaiser_encounter_context", return_value=False), patch.object(
        trainer_stub, "_is_insatiable_encounter_context", return_value=False
    ), patch.object(
        trainer_stub, "_combat_encounter_tier_from_raw", return_value="normal"
    ), patch.object(
        trainer_stub, "_combat_energy", return_value=1.0
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
    assert search_stats["combat_quality_survival_non_endturn_guard_available"] == 0.0
    assert search_stats["combat_quality_survival_non_endturn_guard_applied"] == 0.0
    assert search_stats["combat_quality_hard_guard_override_any"] == 0.0


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
