"""Boss, elite, and late-normal survival/race guard regressions.

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
