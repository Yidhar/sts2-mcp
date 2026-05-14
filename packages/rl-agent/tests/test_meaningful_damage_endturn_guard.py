from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest


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
    card_type: str = "Attack",
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


def _play(card: dict[str, object], *, safety: dict[str, object] | None = None) -> dict[str, object]:
    action: dict[str, object] = {
        "kind": "play_card",
        "action_id": f"play_card:{card['id']}",
        "card": card,
    }
    if safety is not None:
        action["safety"] = safety
    return action


def _end_turn() -> dict[str, object]:
    return {"kind": "end_turn", "action_id": "end_turn"}


def _raw_combat_obs(
    *,
    hp: int = 50,
    max_hp: int = 80,
    block: int = 0,
    energy: int = 3,
    incoming: int = 0,
    intent_type: str = "Buff",
    enemy_hp: int = 40,
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
                    "hp": enemy_hp,
                    "current_hp": enemy_hp,
                    "max_hp": enemy_hp,
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
    search_stats: dict[str, object] | None = None,
) -> tuple[int, dict[str, object]]:
    stats = search_stats if search_stats is not None else {}
    new_idx = trainer._apply_meaningful_damage_endturn_guard(
        action_idx=0,
        legal_count=len(legal_actions),
        legal_actions=legal_actions,
        mask_np=np.ones(len(legal_actions), dtype=np.float32),
        raw_obs=raw_obs,
        encounter="ENCOUNTER.TEST_NORMAL",
        search_stats=stats,
    )
    return new_idx, stats


def test_no_pressure_end_turn_overridden_to_safe_strike() -> None:
    trainer = _trainer()
    legal_actions = [
        _end_turn(),
        _play(_card("Strike", damage=6, semantic_tags=["attack", "damage"])),
    ]

    new_idx, stats = _apply(trainer, legal_actions=legal_actions, raw_obs=_raw_combat_obs())

    assert new_idx == 1
    assert stats["combat_quality_meaningful_damage_endturn_guard_available"] == 1.0
    assert stats["combat_quality_meaningful_damage_endturn_guard_applied"] == 1.0
    assert stats["combat_quality_meaningful_damage_endturn_guard_override"] == 1.0
    assert stats["combat_quality_meaningful_damage_endturn_guard_candidate_count"] == 1.0
    assert stats["combat_quality_hard_guard_override_any"] == 1.0


def test_high_incoming_pressure_skips_end_turn_guard() -> None:
    trainer = _trainer()
    legal_actions = [
        _end_turn(),
        _play(_card("Strike", damage=6, semantic_tags=["attack", "damage"])),
    ]

    new_idx, stats = _apply(
        trainer,
        legal_actions=legal_actions,
        raw_obs=_raw_combat_obs(incoming=10, intent_type="Attack"),
    )

    assert new_idx == 0
    assert stats["combat_quality_meaningful_damage_endturn_guard_pressure_skip"] == 1.0
    assert "combat_quality_meaningful_damage_endturn_guard_applied" not in stats


def test_nonlethal_hp_cost_candidate_is_rejected() -> None:
    trainer = _trainer()
    blood_attack = _play(
        _card(
            "Blood Strike",
            damage=6,
            operations=[{"op": "lose_hp", "amount": 3}],
            semantic_tags=["attack", "damage"],
        ),
        safety={"hp_loss_unblockable": 3, "hp_before": 50, "source_confidence": "runtime_internal"},
    )
    legal_actions = [_end_turn(), blood_attack]

    new_idx, stats = _apply(trainer, legal_actions=legal_actions, raw_obs=_raw_combat_obs(hp=50))

    assert new_idx == 0
    assert stats["combat_quality_meaningful_damage_endturn_guard_candidate_count"] == 0.0
    assert stats["combat_quality_meaningful_damage_endturn_guard_no_alternative"] == 1.0


def test_lethal_hp_cost_candidate_is_allowed() -> None:
    trainer = _trainer()
    blood_attack = _play(
        _card(
            "Blood Strike",
            damage=6,
            operations=[{"op": "lose_hp", "amount": 3}],
            semantic_tags=["attack", "damage"],
        ),
        safety={"hp_loss_unblockable": 3, "hp_before": 50, "source_confidence": "runtime_internal"},
    )
    legal_actions = [_end_turn(), blood_attack]

    new_idx, stats = _apply(
        trainer,
        legal_actions=legal_actions,
        raw_obs=_raw_combat_obs(hp=50, enemy_hp=6),
    )

    assert new_idx == 1
    assert stats["combat_quality_meaningful_damage_endturn_guard_lethal_candidate"] == 1.0
    assert stats["combat_quality_meaningful_damage_endturn_guard_applied"] == 1.0


def test_only_pure_block_under_no_pressure_is_not_a_progress_candidate() -> None:
    trainer = _trainer()
    legal_actions = [
        _end_turn(),
        _play(_card("Defend", card_type="Skill", damage=0, block=5, semantic_tags=["block"])),
    ]

    new_idx, stats = _apply(trainer, legal_actions=legal_actions, raw_obs=_raw_combat_obs())

    assert new_idx == 0
    assert stats["combat_quality_meaningful_damage_endturn_guard_candidate_count"] == 0.0
    assert stats["combat_quality_meaningful_damage_endturn_guard_no_alternative"] == 1.0


@pytest.mark.parametrize("masked_idx", [0, 2])
def test_invalid_or_non_end_turn_selection_is_ignored(masked_idx: int) -> None:
    trainer = _trainer()
    legal_actions = [
        _end_turn(),
        _play(_card("Strike", damage=6, semantic_tags=["attack", "damage"])),
    ]
    stats: dict[str, object] = {}
    new_idx = trainer._apply_meaningful_damage_endturn_guard(
        action_idx=masked_idx,
        legal_count=len(legal_actions),
        legal_actions=legal_actions,
        mask_np=np.ones(len(legal_actions), dtype=np.float32),
        raw_obs=_raw_combat_obs(),
        encounter="ENCOUNTER.TEST_NORMAL",
        search_stats=stats,
    )

    if masked_idx == 0:
        assert new_idx == 1
    else:
        assert new_idx == masked_idx
