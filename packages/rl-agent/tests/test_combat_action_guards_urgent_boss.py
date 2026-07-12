"""Urgent end-turn, boss-potion, Kaiser, and Insatiable guard regressions.

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
