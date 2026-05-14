from __future__ import annotations

import sys
from pathlib import Path
from unittest.mock import patch

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from muzero.train import MuZeroTrainer


def _trainer() -> MuZeroTrainer:
    trainer = MuZeroTrainer.__new__(MuZeroTrainer)
    trainer.log_dir = None
    trainer.episode_count = 0
    trainer.total_steps = 0
    trainer._dump_combat_hard_guard_record = lambda **_kwargs: None  # type: ignore[method-assign]
    return trainer


def _card(
    title: str,
    *,
    card_type: str = "Skill",
    cost: int | str = 1,
    damage: int = 0,
    block: int = 0,
    operations: list[dict[str, object]] | None = None,
    semantic_tags: list[str] | None = None,
) -> dict[str, object]:
    return {
        "id": title.upper().replace(" ", "_"),
        "title": title,
        "name": title,
        "type": card_type,
        "cost": cost,
        "damage": damage,
        "total_damage": damage,
        "preview_damage": damage,
        "block": block,
        "total_block": block,
        "preview_block": block,
        "can_play": True,
        "card_effect_profile": {
            "operations": operations or [],
            "semantic_tags": semantic_tags or [],
            "training_tags": [],
        },
    }


def _play(card: dict[str, object]) -> dict[str, object]:
    return {
        "kind": "play_card",
        "action_id": f"play_card:{card['id']}",
        "card": card,
    }


def _raw_combat_obs(
    *,
    hp: int = 50,
    max_hp: int = 80,
    block: int = 0,
    energy: int = 3,
    incoming: int = 0,
    intent_type: str = "Buff",
    room_type: str = "Monster",
) -> dict[str, object]:
    return {
        "phase": "combat",
        "run": {"room_type": room_type},
        "player": {"hp": hp, "max_hp": max_hp, "block": block, "relics": [], "potions": []},
        "combat": {
            "energy": energy,
            "max_energy": 3,
            "round": 1,
            "hand": [],
            "draw_pile": [],
            "discard_pile": [],
            "exhaust_pile": [],
            "enemies": [
                {
                    "id": "enemy.test",
                    "combat_id": "enemy-1",
                    "hp": 40,
                    "current_hp": 40,
                    "max_hp": 40,
                    "intent": {
                        "intent_type": intent_type,
                        "type": intent_type,
                        "total_damage": incoming,
                        "damage": incoming,
                    },
                }
            ],
        },
    }


def _apply(
    trainer: MuZeroTrainer,
    *,
    legal_actions: list[dict[str, object]],
    raw_obs: dict[str, object],
    action_idx: int = 0,
    search_stats: dict[str, object] | None = None,
    mask_np: np.ndarray | None = None,
) -> tuple[int, dict[str, object]]:
    stats = search_stats if search_stats is not None else {}
    new_idx = trainer._apply_no_pressure_block_guard(
        action_idx=action_idx,
        legal_count=len(legal_actions),
        legal_actions=legal_actions,
        mask_np=mask_np if mask_np is not None else np.ones(len(legal_actions), dtype=np.float32),
        raw_obs=raw_obs,
        encounter="ENCOUNTER.TEST_NORMAL",
        search_stats=stats,
    )
    return new_idx, stats


def test_selected_defend_under_buff_is_overridden_to_strike() -> None:
    trainer = _trainer()
    legal_actions = [
        _play(_card("Defend", card_type="Skill", block=5, semantic_tags=["block"])),
        _play(_card("Strike", card_type="Attack", damage=6, semantic_tags=["attack", "damage"])),
        {"kind": "end_turn", "action_id": "end_turn"},
    ]

    prefilled_stats = {
        "combat_quality_bad_pure_block_selected": 1.0,
        "combat_quality_pure_block_progress_alternative_selected": 1.0,
        "combat_quality_pure_block_low_value_pressure_selected": 1.0,
        "combat_quality_pure_block_no_alternative_selected": 1.0,
        "combat_quality_pure_block_survival_justified_selected": 1.0,
        "combat_quality_insufficient_block_selected": 1.0,
    }

    new_idx, stats = _apply(
        trainer,
        legal_actions=legal_actions,
        raw_obs=_raw_combat_obs(),
        search_stats=prefilled_stats,
    )

    assert new_idx == 1
    assert stats["combat_quality_no_pressure_block_guard_available"] == 1.0
    assert stats["combat_quality_no_pressure_block_guard_applied"] == 1.0
    assert stats["combat_quality_no_pressure_block_guard_override"] == 1.0
    assert stats["combat_quality_no_pressure_block_guard_progress_override_idx"] == 1.0
    assert stats["combat_quality_no_pressure_block_guard_progress_override_lock"] == 1.0
    assert stats["combat_quality_no_pressure_block_guard_candidate_count"] == 1.0
    assert stats["combat_quality_card_block_waste_selected"] == 0.0
    assert stats["combat_quality_card_block_waste_with_progress_selected"] == 0.0
    assert stats["combat_quality_card_pure_block_selected"] == 0.0
    assert stats["combat_quality_card_no_damage_pressure_selected"] == 0.0
    assert stats["combat_quality_card_no_damage_pressure_with_progress_selected"] == 0.0
    assert stats["combat_quality_bad_pure_block_selected"] == 0.0
    assert stats["combat_quality_pure_block_progress_alternative_selected"] == 0.0
    assert stats["combat_quality_pure_block_low_value_pressure_selected"] == 0.0
    assert stats["combat_quality_pure_block_no_alternative_selected"] == 0.0
    assert stats["combat_quality_pure_block_survival_justified_selected"] == 0.0
    assert stats["combat_quality_insufficient_block_selected"] == 0.0


def test_selected_defend_against_meaningful_incoming_is_not_overridden() -> None:
    trainer = _trainer()
    legal_actions = [
        _play(_card("Defend", card_type="Skill", block=5, semantic_tags=["block"])),
        _play(_card("Strike", card_type="Attack", damage=6, semantic_tags=["attack", "damage"])),
    ]

    new_idx, stats = _apply(
        trainer,
        legal_actions=legal_actions,
        raw_obs=_raw_combat_obs(hp=18, incoming=10, intent_type="Attack"),
    )

    assert new_idx == 0
    assert stats["combat_quality_no_pressure_block_guard_pressure_skip"] == 1.0
    assert "combat_quality_no_pressure_block_guard_applied" not in stats


def test_selected_low_value_defend_under_safe_pressure_is_overridden_to_strike() -> None:
    trainer = _trainer()
    legal_actions = [
        _play(_card("Defend", card_type="Skill", block=5, semantic_tags=["block"])),
        _play(_card("Strike", card_type="Attack", damage=6, semantic_tags=["attack", "damage"])),
    ]

    new_idx, stats = _apply(
        trainer,
        legal_actions=legal_actions,
        raw_obs=_raw_combat_obs(hp=50, incoming=10, intent_type="Attack"),
    )

    assert new_idx == 1
    assert stats["combat_quality_no_pressure_block_guard_available"] == 1.0
    assert stats["combat_quality_no_pressure_block_guard_pressure_attack_window"] == 1.0
    assert stats["combat_quality_no_pressure_block_guard_low_value_pressure"] == 1.0
    assert stats["combat_quality_no_pressure_block_guard_applied"] == 1.0
    assert stats["combat_quality_no_pressure_block_guard_override"] == 1.0


def test_selected_defend_under_moderate_safe_hallway_pressure_is_overridden_to_progress() -> None:
    trainer = _trainer()
    legal_actions = [
        _play(_card("Defend+", card_type="Skill", block=6, semantic_tags=["block"])),
        _play(_card("Ash Strike+", card_type="Attack", damage=10, semantic_tags=["attack", "damage"])),
    ]

    new_idx, stats = _apply(
        trainer,
        legal_actions=legal_actions,
        raw_obs=_raw_combat_obs(hp=68, incoming=27, intent_type="Attack"),
    )

    assert new_idx == 1
    assert stats["combat_quality_no_pressure_block_guard_available"] == 1.0
    assert stats["combat_quality_no_pressure_block_guard_pressure_attack_window"] == 1.0
    assert stats["combat_quality_no_pressure_block_guard_low_value_pressure"] == 1.0
    assert stats["combat_quality_no_pressure_block_guard_applied"] == 1.0


def test_selected_defend_frog_knight_like_pressure_prefers_progress() -> None:
    trainer = _trainer()
    legal_actions = [
        _play(_card("Defend", card_type="Skill", block=5, semantic_tags=["block"])),
        _play(
            _card(
                "Kindling Source+",
                card_type="Skill",
                cost=2,
                operations=[{"op": "gain_energy_next_turn", "amount": 1}],
                semantic_tags=["resource", "scaling", "setup"],
            )
        ),
        _play(_card("Strike", card_type="Attack", damage=7, semantic_tags=["attack", "damage"])),
    ]

    new_idx, stats = _apply(
        trainer,
        legal_actions=legal_actions,
        raw_obs=_raw_combat_obs(hp=83, max_hp=84, incoming=13, intent_type="Attack"),
    )

    assert new_idx == 2
    assert stats["combat_quality_no_pressure_block_guard_candidate_count"] >= 1.0
    assert stats["combat_quality_no_pressure_block_guard_applied"] == 1.0


def test_selected_defend_mytes_like_safe_pressure_prefers_attack() -> None:
    trainer = _trainer()
    legal_actions = [
        _play(_card("Defend", card_type="Skill", block=5, semantic_tags=["block"])),
        _play(_card("Strike+", card_type="Attack", damage=9, semantic_tags=["attack", "damage"])),
        _play(
            _card(
                "Automation+",
                card_type="Skill",
                cost=1,
                operations=[{"op": "draw_card", "count": 1}],
                semantic_tags=["draw", "resource"],
            )
        ),
    ]

    new_idx, stats = _apply(
        trainer,
        legal_actions=legal_actions,
        raw_obs=_raw_combat_obs(hp=55, incoming=19, intent_type="Attack"),
    )

    assert new_idx == 1
    assert stats["combat_quality_no_pressure_block_guard_pressure_attack_window"] == 1.0
    assert stats["combat_quality_no_pressure_block_guard_applied"] == 1.0


def test_selected_defend_with_one_energy_safe_pressure_prefers_attack() -> None:
    trainer = _trainer()
    legal_actions = [
        _play(_card("Defend", card_type="Skill", block=5, semantic_tags=["block"])),
        _play(_card("Strike+", card_type="Attack", damage=9, semantic_tags=["attack", "damage"])),
        _play(_card("Big Attack", card_type="Attack", cost=2, damage=16, semantic_tags=["attack", "damage"])),
    ]

    new_idx, stats = _apply(
        trainer,
        legal_actions=legal_actions,
        raw_obs=_raw_combat_obs(hp=46, energy=1, incoming=6, intent_type="Attack"),
    )

    assert new_idx == 1
    assert stats["combat_quality_no_pressure_block_guard_candidate_count"] == 1.0
    assert stats["combat_quality_no_pressure_block_guard_applied"] == 1.0


def test_selected_defend_plus_one_energy_low_absolute_pressure_prefers_attack() -> None:
    """Regression for live sandbox offender: HP safe, 5 incoming, Defend+ wastes tempo."""

    trainer = _trainer()
    legal_actions = [
        _play(_card("Defend+", card_type="Skill", block=8, semantic_tags=["block"])),
        _play(_card("Strike", card_type="Attack", damage=9, semantic_tags=["attack", "damage"])),
        _play(_card("Strike", card_type="Attack", damage=9, semantic_tags=["attack", "damage"])),
        {"kind": "end_turn", "action_id": "end_turn"},
    ]

    new_idx, stats = _apply(
        trainer,
        legal_actions=legal_actions,
        raw_obs=_raw_combat_obs(hp=71, max_hp=80, energy=1, incoming=5, intent_type="Attack"),
    )

    assert new_idx in {1, 2}
    assert stats["combat_quality_no_pressure_block_guard_available"] == 1.0
    assert stats["combat_quality_no_pressure_block_guard_pressure_attack_window"] == 1.0
    assert stats["combat_quality_no_pressure_block_guard_low_value_pressure"] == 1.0
    assert stats["combat_quality_no_pressure_block_guard_candidate_count"] == 2.0
    assert stats["combat_quality_no_pressure_block_guard_applied"] == 1.0


def test_selected_defend_mid_pressure_metric_band_prefers_attack() -> None:
    """Regression for guard/metric drift: selected-side bad=1 but old guard skipped.

    HP 31, incoming 5, block 0, Defend+ for 8 is not a no-damage/trivial
    context and the low-value-pressure helper returns false because the card
    over-covers more than half the threat and post-hit HP is below the helper's
    high-margin absolute-pressure floor.  It is also not survival-justified:
    taking 5 leaves 26 HP in a normal hallway and Strike is legal progress.
    The guard must still collect candidates and rewrite, otherwise
    ``bad_pure_block_selected`` survives post-guard recomputation.
    """

    trainer = _trainer()
    legal_actions = [
        _play(_card("Defend+", card_type="Skill", block=8, semantic_tags=["block"])),
        _play(_card("Strike", card_type="Attack", damage=6, semantic_tags=["attack", "damage"])),
    ]

    new_idx, stats = _apply(
        trainer,
        legal_actions=legal_actions,
        raw_obs=_raw_combat_obs(hp=31, max_hp=80, energy=1, incoming=5, intent_type="Attack"),
    )

    assert new_idx == 1
    assert stats["combat_quality_no_pressure_block_guard_available"] == 1.0
    assert stats["combat_quality_no_pressure_block_guard_mid_pressure_attempt"] == 1.0
    assert stats["combat_quality_no_pressure_block_guard_candidate_count"] == 1.0
    assert stats["combat_quality_no_pressure_block_guard_applied"] == 1.0


def test_selected_defend_plus_low_absolute_pressure_not_overridden_when_survival_justified() -> None:
    trainer = _trainer()
    legal_actions = [
        _play(_card("Defend+", card_type="Skill", block=8, semantic_tags=["block"])),
        _play(_card("Strike", card_type="Attack", damage=9, semantic_tags=["attack", "damage"])),
    ]

    new_idx, stats = _apply(
        trainer,
        legal_actions=legal_actions,
        raw_obs=_raw_combat_obs(hp=25, max_hp=80, energy=1, incoming=5, intent_type="Attack"),
    )

    assert new_idx == 0
    assert stats["combat_quality_no_pressure_block_guard_pressure_skip"] == 1.0
    assert stats["combat_quality_no_pressure_block_guard_survival_justified"] == 1.0
    assert "combat_quality_no_pressure_block_guard_applied" not in stats


def test_selected_defend_with_masked_progress_alt_is_not_overridden() -> None:
    trainer = _trainer()
    legal_actions = [
        _play(_card("Defend", card_type="Skill", block=5, semantic_tags=["block"])),
        _play(_card("Strike", card_type="Attack", damage=7, semantic_tags=["attack", "damage"])),
        {"kind": "end_turn", "action_id": "end_turn"},
    ]

    new_idx, stats = _apply(
        trainer,
        legal_actions=legal_actions,
        raw_obs=_raw_combat_obs(),
        mask_np=np.asarray([1.0, 0.0, 1.0], dtype=np.float32),
    )

    assert new_idx == 0
    assert stats["combat_quality_no_pressure_block_guard_available"] == 1.0
    assert stats["combat_quality_no_pressure_block_guard_candidate_count"] == 0.0
    assert stats["combat_quality_no_pressure_block_guard_no_alternative"] == 1.0
    assert stats["combat_quality_no_pressure_block_guard_reject_mask"] >= 1.0


def test_selected_low_value_defend_low_hp_pressure_is_not_overridden() -> None:
    trainer = _trainer()
    legal_actions = [
        _play(_card("Defend", card_type="Skill", block=5, semantic_tags=["block"])),
        _play(_card("Strike", card_type="Attack", damage=6, semantic_tags=["attack", "damage"])),
    ]

    new_idx, stats = _apply(
        trainer,
        legal_actions=legal_actions,
        raw_obs=_raw_combat_obs(hp=11, incoming=5, intent_type="Attack"),
    )

    assert new_idx == 0
    assert stats["combat_quality_no_pressure_block_guard_pressure_skip"] == 1.0
    assert "combat_quality_no_pressure_block_guard_applied" not in stats


def test_selected_defend_against_trivial_incoming_is_overridden_to_strike() -> None:
    trainer = _trainer()
    legal_actions = [
        _play(_card("Defend", card_type="Skill", block=5, semantic_tags=["block"])),
        _play(_card("Strike", card_type="Attack", damage=6, semantic_tags=["attack", "damage"])),
    ]

    new_idx, stats = _apply(
        trainer,
        legal_actions=legal_actions,
        raw_obs=_raw_combat_obs(hp=50, incoming=2, intent_type="Attack"),
    )

    assert new_idx == 1
    assert stats["combat_quality_no_pressure_block_guard_available"] == 1.0
    assert stats["combat_quality_no_pressure_block_guard_trivial_pressure"] == 1.0
    assert stats["combat_quality_no_pressure_block_guard_applied"] == 1.0


def test_selected_defend_without_progress_alternative_is_not_overridden() -> None:
    trainer = _trainer()
    legal_actions = [
        _play(_card("Defend", card_type="Skill", block=5, semantic_tags=["block"])),
        {"kind": "end_turn", "action_id": "end_turn"},
    ]

    new_idx, stats = _apply(trainer, legal_actions=legal_actions, raw_obs=_raw_combat_obs())

    assert new_idx == 0
    assert stats["combat_quality_no_pressure_block_guard_available"] == 1.0
    assert stats["combat_quality_no_pressure_block_guard_candidate_count"] == 0.0
    assert stats["combat_quality_no_pressure_block_guard_no_alternative"] == 1.0


def test_block_plus_draw_card_is_not_treated_as_pure_block() -> None:
    trainer = _trainer()
    legal_actions = [
        _play(
            _card(
                "Shrug Like",
                card_type="Skill",
                block=8,
                operations=[{"op": "draw_card", "count": 1}],
                semantic_tags=["block", "draw"],
            )
        ),
        _play(_card("Strike", card_type="Attack", damage=6, semantic_tags=["attack", "damage"])),
    ]

    new_idx, stats = _apply(trainer, legal_actions=legal_actions, raw_obs=_raw_combat_obs())

    assert new_idx == 0
    assert "combat_quality_no_pressure_block_guard_available" not in stats
    assert "combat_quality_no_pressure_block_guard_applied" not in stats


def test_metric_aligned_profile_still_rewrites_when_rich_classifier_skips_pure_block() -> None:
    """Regression for guard/metric drift caused by mechanism-urgent classification.

    The selected-side bad-pure-block TensorBoard metric is based on the plain
    ``_card_block_waste_profile(..., mechanism_urgent=False)`` path.  The richer
    positive-action classifier can mark the same selected Defend as not pure
    block in mechanism-urgent contexts.  The hard guard must use the metric-
    aligned plain profile for its initial gate, otherwise the executed action can
    remain Defend while the selected-side metric reports a failure.
    """

    trainer = _trainer()
    legal_actions = [
        _play(_card("Defend", card_type="Skill", block=5, semantic_tags=["block"])),
        _play(_card("Strike", card_type="Attack", damage=11, semantic_tags=["attack", "damage"])),
        {"kind": "end_turn", "action_id": "end_turn"},
    ]
    rich_classifier_profile = {
        "card_pure_block": False,
        "card_block_waste": False,
        "card_no_damage_pressure": False,
        "card_no_damage_pressure_context": False,
    }
    original_classifier = trainer._classify_positive_combat_action

    def _selected_only_rich_classifier(*args, **kwargs):
        if len(args) >= 2 and int(args[1]) == 0:
            return rich_classifier_profile
        return original_classifier(*args, **kwargs)

    with patch.object(trainer, "_classify_positive_combat_action", side_effect=_selected_only_rich_classifier):
        new_idx, stats = _apply(
            trainer,
            legal_actions=legal_actions,
            raw_obs=_raw_combat_obs(hp=60, max_hp=80, incoming=0, intent_type="Buff"),
        )

    assert new_idx == 1
    assert stats.get("combat_quality_no_pressure_block_guard_profile_skip", 0.0) == 0.0
    assert stats["combat_quality_no_pressure_block_guard_available"] == 1.0
    assert stats["combat_quality_no_pressure_block_guard_applied"] == 1.0
    assert stats["combat_quality_no_pressure_block_guard_override"] == 1.0
    assert stats["combat_quality_no_pressure_block_guard_candidate_count"] >= 1.0


def test_non_pure_block_selected_records_profile_skip_without_rewrite() -> None:
    """Telemetry should explain why a selected play-card skipped the guard gate."""

    trainer = _trainer()
    legal_actions = [
        _play(
            _card(
                "Shrug Like",
                card_type="Skill",
                block=8,
                operations=[{"op": "draw_card", "count": 1}],
                semantic_tags=["block", "draw"],
            )
        ),
        _play(_card("Strike", card_type="Attack", damage=6, semantic_tags=["attack", "damage"])),
    ]

    with patch.object(trainer, "_card_block_waste_profile", return_value={"pure_block": False, "block_waste": False}):
        new_idx, stats = _apply(trainer, legal_actions=legal_actions, raw_obs=_raw_combat_obs())

    assert new_idx == 0
    assert stats["combat_quality_no_pressure_block_guard_profile_skip"] == 1.0
    assert "combat_quality_no_pressure_block_guard_available" not in stats
    assert "combat_quality_no_pressure_block_guard_applied" not in stats
