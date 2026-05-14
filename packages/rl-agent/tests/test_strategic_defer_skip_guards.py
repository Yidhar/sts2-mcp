from __future__ import annotations

import sys
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from muzero.train import MuZeroTrainer


def _trainer(raw_obs: dict[str, object] | None = None) -> MuZeroTrainer:
    trainer = MuZeroTrainer.__new__(MuZeroTrainer)
    trainer.log_dir = None
    trainer.episode_count = 0
    trainer.total_steps = 0
    trainer._dump_combat_hard_guard_record = lambda **_kwargs: None  # type: ignore[method-assign]
    if raw_obs is not None:
        trainer.env = SimpleNamespace(unwrapped=SimpleNamespace(_last_obs_raw=raw_obs))
    else:
        trainer.env = SimpleNamespace(unwrapped=SimpleNamespace(_last_obs_raw=None))
    return trainer


def _card(
    title: str,
    *,
    card_type: str = "Attack",
    cost: int | str = 1,
    damage: int = 0,
    block: int = 0,
    semantic_tags: list[str] | None = None,
    profile: dict[str, object] | None = None,
) -> dict[str, object]:
    card_profile = {
        "operations": [],
        "semantic_tags": semantic_tags or [],
        "training_tags": [],
    }
    if profile:
        card_profile.update(profile)
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
        "card_effect_profile": card_profile,
    }


def _play(
    card: dict[str, object],
    *,
    roles: list[str] | None = None,
    profile: dict[str, object] | None = None,
) -> dict[str, object]:
    action: dict[str, object] = {
        "kind": "play_card",
        "action_id": f"play_card:{card['id']}",
        "card": card,
        "semantic": {
            "family": "play_card",
            "roles": roles or [],
            "card_effect_profile": profile or {},
        },
    }
    return action


def _end_turn() -> dict[str, object]:
    return {"kind": "end_turn", "action_id": "end_turn", "semantic": {"family": "end_turn"}}


def _raw_combat_obs(
    *,
    hp: int = 70,
    max_hp: int = 80,
    block: int = 0,
    energy: int = 3,
    incoming: int = 0,
    intent_type: str = "Buff",
    enemy_hp: int = 40,
    room_type: str = "Monster",
    encounter_id: str = "ENCOUNTER.TEST_NORMAL",
) -> dict[str, object]:
    return {
        "phase": "combat",
        "encounter_id": encounter_id,
        "run": {"room_type": room_type},
        "player": {"hp": hp, "current_hp": hp, "max_hp": max_hp, "block": block, "relics": [], "potions": []},
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


def _strike(damage: int = 6) -> dict[str, object]:
    return _play(
        _card("Strike", card_type="Attack", damage=damage, semantic_tags=["attack", "damage"]),
        roles=["attack", "damage"],
    )


def _setup_card() -> dict[str, object]:
    profile = {"typed_requires_followup": 1.0, "typed_strategic_skip_if_no_followup": 1.0}
    return _play(
        _card("Setup Bell", card_type="Skill", semantic_tags=["setup"], profile=profile),
        roles=["setup"],
        profile=profile,
    )


def test_strategic_defer_endturn_overrides_safe_hallway_pressure_to_strike() -> None:
    raw_obs = _raw_combat_obs(hp=70, incoming=10, intent_type="Attack")
    trainer = _trainer(raw_obs)
    legal_actions = [_end_turn(), _strike(damage=6)]
    stats: dict[str, object] = {}

    new_idx = trainer._apply_strategic_defer_endturn_guard(
        action_idx=0,
        legal_count=len(legal_actions),
        legal_actions=legal_actions,
        mask_np=np.ones(len(legal_actions), dtype=np.float32),
        raw_obs=raw_obs,
        encounter="ENCOUNTER.TEST_NORMAL",
        search_stats=stats,
    )

    assert new_idx == 1
    assert stats["combat_quality_strategic_defer_endturn_guard_available"] == 1.0
    assert stats["combat_quality_strategic_defer_endturn_guard_safe_pressure"] == 1.0
    assert stats["combat_quality_strategic_defer_endturn_guard_applied"] == 1.0
    assert stats["combat_quality_strategic_defer_endturn_guard_override"] == 1.0
    assert stats["combat_quality_strategic_defer_endturn_guard_candidate_count"] == 1.0
    assert stats["combat_quality_hard_guard_override_any"] == 1.0


def test_strategic_defer_endturn_respects_near_lethal_pressure() -> None:
    raw_obs = _raw_combat_obs(hp=10, incoming=14, intent_type="Attack")
    trainer = _trainer(raw_obs)
    legal_actions = [_end_turn(), _strike(damage=6)]
    stats: dict[str, object] = {}

    new_idx = trainer._apply_strategic_defer_endturn_guard(
        action_idx=0,
        legal_count=len(legal_actions),
        legal_actions=legal_actions,
        mask_np=np.ones(len(legal_actions), dtype=np.float32),
        raw_obs=raw_obs,
        encounter="ENCOUNTER.TEST_NORMAL",
        search_stats=stats,
    )

    assert new_idx == 0
    assert stats["combat_quality_strategic_defer_endturn_guard_pressure_skip"] == 1.0
    assert "combat_quality_strategic_defer_endturn_guard_applied" not in stats


def test_strategic_skip_setup_card_overrides_to_safe_progress() -> None:
    raw_obs = _raw_combat_obs(hp=70, incoming=0, intent_type="Buff")
    trainer = _trainer(raw_obs)
    legal_actions = [_setup_card(), _strike(damage=6), _end_turn()]
    stats: dict[str, object] = {}

    new_idx = trainer._apply_strategic_skip_guard(
        action_idx=0,
        legal_count=len(legal_actions),
        legal_actions=legal_actions,
        mask_np=np.ones(len(legal_actions), dtype=np.float32),
        raw_obs=raw_obs,
        encounter="ENCOUNTER.TEST_NORMAL",
        search_stats=stats,
    )

    assert new_idx == 1
    assert stats["combat_quality_strategic_skip_guard_available"] == 1.0
    assert stats["combat_quality_strategic_skip_guard_applied"] == 1.0
    assert stats["combat_quality_strategic_skip_guard_override"] == 1.0
    assert stats["combat_quality_strategic_skip_guard_candidate_count"] == 1.0
    assert stats["combat_quality_hard_guard_override_any"] == 1.0


def test_strategic_skip_keeps_confirmed_lethal_setup_action() -> None:
    raw_obs = _raw_combat_obs(hp=70, incoming=0, intent_type="Buff")
    trainer = _trainer(raw_obs)
    setup = _setup_card()
    legal_actions = [setup, _strike(damage=6), _end_turn()]
    stats: dict[str, object] = {}

    with patch.object(trainer, "_is_action_confirmed_lethal", side_effect=lambda action, raw=None: action is setup):
        new_idx = trainer._apply_strategic_skip_guard(
            action_idx=0,
            legal_count=len(legal_actions),
            legal_actions=legal_actions,
            mask_np=np.ones(len(legal_actions), dtype=np.float32),
            raw_obs=raw_obs,
            encounter="ENCOUNTER.TEST_NORMAL",
            search_stats=stats,
        )

    assert new_idx == 0
    assert stats["combat_quality_strategic_skip_guard_available"] == 1.0
    assert stats["combat_quality_strategic_skip_guard_lethal_exemption"] == 1.0
    assert "combat_quality_strategic_skip_guard_applied" not in stats


def test_endturn_strategic_defer_availability_requires_safe_progress_candidate() -> None:
    raw_obs = _raw_combat_obs(hp=70, incoming=0, intent_type="Buff")
    trainer = _trainer(raw_obs)
    legal_actions = [_end_turn(), _setup_card()]
    mask = np.ones(len(legal_actions), dtype=np.float32)

    context = trainer._raw_end_turn_context(None, mask, legal_actions)
    assert context["safe_progress_candidate_count"] == 0
    assert context["strategic_defer_available"] is False

    _bias, stats, _zero_x = trainer._combat_action_quality_bias(None, mask, legal_actions)
    assert float(stats["combat_quality_safe_progress_candidate_count"]) == 0.0
    assert float(stats["combat_quality_strategic_defer_available"]) == 0.0

    selected = trainer._selected_combat_quality_stats(None, 0, legal_actions, stats, mask)
    assert selected["combat_quality_strategic_defer_end_turn_selected"] == 0.0
