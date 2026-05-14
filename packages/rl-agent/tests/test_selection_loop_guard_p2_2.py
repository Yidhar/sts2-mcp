"""P2-2 selection-loop guard tests.

The guard fires when the same selection screen signature + same picked
option + same selection_action recur for ``_SELECTION_LOOP_STREAK_THRESHOLD``
consecutive trainer steps.  Three override paths in priority order:

1. ``confirm`` legal → force confirm.
2. else ``cancel`` / ``skip`` / ``close`` legal → force cancel.
3. else any other legal pick → alt_pick.

Tests construct a minimal ``MuZeroTrainer`` stub and drive
``_apply_combat_action_hard_guards`` 4+ times with the same selection
state to assert the override fires only on the streak-threshold-th call.
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
    from muzero.train import MuZeroTrainer

    trainer = MuZeroTrainer.__new__(MuZeroTrainer)
    trainer.log_dir = None
    trainer.episode_count = 0
    trainer.total_steps = 0
    return trainer


def _build_pick_actions() -> list[dict]:
    return [
        {
            "action_id": "select_card_a",
            "kind": "card_selection",
            "label": "Select",
            "card": {"id": "CARD.A"},
            "selection_action": "pick",
        },
        {
            "action_id": "select_card_b",
            "kind": "card_selection",
            "label": "Select",
            "card": {"id": "CARD.B"},
            "selection_action": "pick",
        },
        {
            "action_id": "selection_confirm",
            "kind": "card_selection",
            "label": "Confirm",
            "selection_action": "confirm",
        },
        {
            "action_id": "selection_cancel",
            "kind": "card_selection",
            "label": "Cancel",
            "selection_action": "cancel",
        },
    ]


def _drive_guard(
    trainer_stub,
    *,
    action_idx: int,
    legal_actions: list[dict],
    iterations: int,
):
    """Drive the guard ``iterations`` times with the same input and return
    (final_action_idx, last_search_stats)."""
    mask = np.ones(len(legal_actions), dtype=np.float32)
    last_idx = action_idx
    last_stats: dict = {}
    raw_obs = {"combat": {"energy": 0}, "player": {"current_hp": 50, "max_hp": 80}}
    with patch.object(
        trainer_stub, "_is_kaiser_encounter_context", return_value=False
    ), patch.object(
        trainer_stub, "_is_insatiable_encounter_context", return_value=False
    ), patch.object(
        trainer_stub, "_combat_energy", return_value=0.0
    ), patch.object(
        trainer_stub, "_semantic_family", side_effect=lambda action: action.get("kind") or "unknown"
    ):
        for _ in range(iterations):
            stats: dict = {}
            last_idx = trainer_stub._apply_combat_action_hard_guards(
                action_idx=action_idx,
                legal_actions=legal_actions,
                action_mask=mask,
                raw_obs=raw_obs,
                boss_ctx={},
                encounter="ENCOUNTER.SELECTION_TEST",
                search_stats=stats,
            )
            last_stats = stats
    return last_idx, last_stats


def test_selection_loop_guard_dormant_for_first_three_picks(trainer_stub):
    """Streak-threshold defaults to 4 so the first three identical picks
    must NOT override (legitimate exploration, e.g., re-clicking the same
    card to deselect)."""
    actions = _build_pick_actions()
    final_idx, stats = _drive_guard(
        trainer_stub, action_idx=0, legal_actions=actions, iterations=3
    )
    assert final_idx == 0
    assert stats["combat_quality_selection_loop_screen_active"] == 1.0
    assert stats["combat_quality_selection_loop_detected"] == 0.0
    assert stats["combat_quality_selection_loop_applied"] == 0.0


def test_selection_loop_guard_force_confirm_at_streak_threshold(trainer_stub):
    """Once the same (signature, picked_card, selection_action) tuple repeats
    threshold (=4) times, the guard must override to ``confirm`` since it
    is legal in this fixture."""
    actions = _build_pick_actions()
    final_idx, stats = _drive_guard(
        trainer_stub, action_idx=0, legal_actions=actions, iterations=4
    )
    assert final_idx == 2  # confirm
    assert stats["combat_quality_selection_loop_detected"] == 1.0
    assert stats["combat_quality_selection_loop_applied"] == 1.0
    assert stats["combat_quality_selection_loop_auto_confirm"] == 1.0
    assert stats["combat_quality_selection_repeated_same_option"] == 1.0


def test_selection_loop_guard_force_cancel_when_no_confirm(trainer_stub):
    """If confirm isn't legal (min_select not met), guard prefers cancel."""
    actions = _build_pick_actions()
    # Drop the confirm action; cancel is still available.
    actions = [a for a in actions if a["selection_action"] != "confirm"]
    final_idx, stats = _drive_guard(
        trainer_stub, action_idx=0, legal_actions=actions, iterations=4
    )
    # actions order: pick_a, pick_b, cancel
    assert final_idx == 2  # cancel
    assert stats["combat_quality_selection_loop_auto_cancel"] == 1.0
    assert stats["combat_quality_selection_loop_auto_confirm"] == 0.0


def test_selection_loop_guard_force_alt_pick_when_no_confirm_or_cancel(trainer_stub):
    """No confirm / cancel — guard picks the first alternative card with a
    different card_id from the repeated pick."""
    actions = [
        {
            "action_id": "select_card_a",
            "kind": "card_selection",
            "card": {"id": "CARD.A"},
            "selection_action": "pick",
        },
        {
            "action_id": "select_card_b",
            "kind": "card_selection",
            "card": {"id": "CARD.B"},
            "selection_action": "pick",
        },
    ]
    final_idx, stats = _drive_guard(
        trainer_stub, action_idx=0, legal_actions=actions, iterations=4
    )
    assert final_idx == 1  # CARD.B alt pick
    assert stats["combat_quality_selection_loop_alt_pick"] == 1.0
    assert stats["combat_quality_selection_loop_auto_confirm"] == 0.0
    assert stats["combat_quality_selection_loop_auto_cancel"] == 0.0


def test_selection_loop_guard_resets_when_screen_exits(trainer_stub):
    """Leaving the selection screen must clear the streak so a *later*
    selection round starts fresh (no spurious override on the 1st pick of
    the new round)."""
    pick_actions = _build_pick_actions()
    non_selection_actions = [
        {"action_id": "play_card_strike", "kind": "play_card", "card": {"id": "CARD.STRIKE"}},
    ]
    raw_obs = {"combat": {"energy": 0}, "player": {"current_hp": 50, "max_hp": 80}}
    mask_pick = np.ones(len(pick_actions), dtype=np.float32)
    mask_play = np.ones(len(non_selection_actions), dtype=np.float32)

    with patch.object(
        trainer_stub, "_is_kaiser_encounter_context", return_value=False
    ), patch.object(
        trainer_stub, "_is_insatiable_encounter_context", return_value=False
    ), patch.object(
        trainer_stub, "_combat_energy", return_value=0.0
    ), patch.object(
        trainer_stub, "_semantic_family", side_effect=lambda action: action.get("kind") or "unknown"
    ):
        # 3 selection picks (no override yet)
        for _ in range(3):
            stats: dict = {}
            trainer_stub._apply_combat_action_hard_guards(
                action_idx=0,
                legal_actions=pick_actions,
                action_mask=mask_pick,
                raw_obs=raw_obs,
                boss_ctx={},
                encounter="ENCOUNTER.X",
                search_stats=stats,
            )
        # Leave selection screen → reset.
        stats: dict = {}
        trainer_stub._apply_combat_action_hard_guards(
            action_idx=0,
            legal_actions=non_selection_actions,
            action_mask=mask_play,
            raw_obs=raw_obs,
            boss_ctx={},
            encounter="ENCOUNTER.X",
            search_stats=stats,
        )
        # Re-enter selection — first pick should NOT trigger override.
        stats: dict = {}
        new_idx = trainer_stub._apply_combat_action_hard_guards(
            action_idx=0,
            legal_actions=pick_actions,
            action_mask=mask_pick,
            raw_obs=raw_obs,
            boss_ctx={},
            encounter="ENCOUNTER.X",
            search_stats=stats,
        )

    assert new_idx == 0
    assert stats["combat_quality_selection_loop_detected"] == 0.0
    assert stats["combat_quality_selection_loop_applied"] == 0.0
