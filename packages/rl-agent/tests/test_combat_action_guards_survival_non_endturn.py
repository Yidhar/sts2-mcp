"""Non-end-turn survival and progress-lock guard regressions.

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


def test_survival_non_endturn_guard_preserves_critical_hallway_progress_when_block_does_not_solve(
    trainer_stub,
):
    """Critical hallway does not mean every attack should be rewritten to Defend.

    Regression for full-run traces where normal hallway fights at moderate HP
    repeatedly rewrote Strike/Pommel-like progress into a 5-block Defend even
    though the original play survived the hit and Defend did not materially
    solve the incoming damage.  This is the over-defense shape that drags
    hallway fights out and increases total HP loss before the Act1 boss.
    """
    raw_obs = {
        "encounter": "ENCOUNTER.FOGMOG_NORMAL",
        "run": {"floor": 8},
        "combat": {
            "energy": 2,
            "enemies": [{"current_hp": 74, "intent": {"total_damage": 9, "damage": 9}}],
        },
        "player": {"hp": 20, "current_hp": 20, "max_hp": 80, "block": 0},
    }
    legal_actions = [
        {
            "action_id": "play_card_strike",
            "kind": "play_card",
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
            encounter="ENCOUNTER.FOGMOG_NORMAL",
            search_stats=search_stats,
        )

    assert new_idx == 0
    assert search_stats["combat_quality_survival_non_endturn_guard_available"] == 1.0
    assert search_stats["combat_quality_survival_non_endturn_guard_progress_exemption"] == 1.0
    assert search_stats["combat_quality_survival_non_endturn_guard_low_value_block_skip"] == 1.0
    assert search_stats["combat_quality_survival_non_endturn_guard_critical_low_value_block_skip"] == 1.0
    assert search_stats["combat_quality_survival_non_endturn_guard_applied"] == 0.0


def test_survival_non_endturn_guard_preserves_floor9_tempo_when_five_block_does_not_solve(
    trainer_stub,
):
    """Regression for the live full-run floor-8/9 normal over-defense loop.

    Recent diagnostics showed hp=27/80, incoming=14, selected Strike, and the
    survival_non_endturn guard rewriting it to a 5-block Defend.  The original
    play survives at 13 HP, while Defend still leaves 9 damage and removes all
    tempo.  Keep the attack so the hallway fight can end instead of locking into
    repeated low-value block.
    """
    raw_obs = {
        "encounter": "ENCOUNTER.FOGMOG_NORMAL",
        "run": {"floor": 9},
        "combat": {
            "energy": 1,
            "enemies": [{"current_hp": 32, "intent": {"total_damage": 14, "damage": 14}}],
        },
        "player": {"hp": 27, "current_hp": 27, "max_hp": 80, "block": 0},
    }
    legal_actions = [
        {
            "action_id": "play_card_strike",
            "kind": "play_card",
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
            encounter="ENCOUNTER.FOGMOG_NORMAL",
            search_stats=search_stats,
        )

    assert new_idx == 0
    assert search_stats["combat_quality_survival_non_endturn_guard_available"] == 1.0
    assert search_stats["combat_quality_survival_non_endturn_guard_progress_exemption"] == 1.0
    assert search_stats["combat_quality_survival_non_endturn_guard_low_value_block_skip"] == 1.0
    assert search_stats["combat_quality_survival_non_endturn_guard_normal_hallway_tempo_candidate"] == 1.0
    assert search_stats["combat_quality_survival_non_endturn_guard_tempo_exempt"] == 1.0
    assert search_stats["combat_quality_survival_non_endturn_guard_applied"] == 0.0


def test_survival_non_endturn_guard_still_blocks_true_lethal_hallway_attack(trainer_stub):
    """The low-value block exemption must not reopen real self-lethal attacks."""
    raw_obs = {
        "encounter": "ENCOUNTER.FLYCONID_NORMAL",
        "run": {"floor": 8},
        "combat": {"energy": 1, "enemies": [{"current_hp": 40, "intent": {"total_damage": 12}}]},
        "player": {"hp": 11, "current_hp": 11, "max_hp": 80, "block": 0},
    }
    legal_actions = [
        {
            "action_id": "play_card_strike",
            "kind": "play_card",
            "card": {"id": "CARD.STRIKE", "title": "Strike", "type": "Attack", "cost": 1},
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
            encounter="ENCOUNTER.FLYCONID_NORMAL",
            search_stats=search_stats,
        )

    assert new_idx == 1
    assert search_stats["combat_quality_survival_non_endturn_guard_available"] == 1.0
    assert search_stats["combat_quality_survival_non_endturn_guard_applied"] == 1.0
    assert search_stats["combat_quality_survival_non_endturn_guard_override"] == 1.0


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
