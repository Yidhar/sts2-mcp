"""X-cost, refund, and HP-cost safety guard regressions.

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
