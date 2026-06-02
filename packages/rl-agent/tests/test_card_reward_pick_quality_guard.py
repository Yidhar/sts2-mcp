from __future__ import annotations

import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def _card(
    card_id: str,
    *,
    title: str | None = None,
    type_: str = "Attack",
    cost: int = 1,
    damage: float = 0.0,
    block: float = 0.0,
    draw: float = 0.0,
    tags: list[str] | None = None,
) -> dict:
    return {
        "id": card_id,
        "title": title or card_id,
        "type": type_,
        "cost": cost,
        "card_effect_profile": {
            "semantic_signals": {"damage": damage, "block": block, "draw": draw},
            "semantic_tags": list(tags or []),
        },
    }


def _thin_low_block_deck() -> list[dict]:
    return [_card(f"strike_{idx}", type_="Attack", damage=6) for idx in range(5)] + [
        _card("weak_defend", type_="Skill", block=5)
    ]


def _thin_low_attack_deck() -> list[dict]:
    return [_card(f"defend_{idx}", type_="Skill", block=10) for idx in range(8)]


def _healthy_deck() -> list[dict]:
    return (
        [_card(f"attack_{idx}", type_="Attack", damage=14) for idx in range(8)]
        + [_card(f"block_{idx}", type_="Skill", block=10) for idx in range(8)]
        + [_card(f"draw_{idx}", type_="Skill", draw=2, tags=["draw"]) for idx in range(5)]
        + [_card(f"scaling_{idx}", type_="Power", tags=["strength"]) for idx in range(4)]
    )


def _raw(deck: list[dict], *, floor: int = 4, act_id: int = 0) -> dict:
    return {
        "run": {"current_floor": floor, "act_id": act_id},
        "player": {"hp": 70, "max_hp": 80, "deck_cards": deck},
    }


def _skip_action() -> dict:
    return {
        "kind": "skip_card_reward",
        "selection": "skip",
        "surface": "card_reward",
        "action_id": "reward.skip_card_reward",
    }


def _pick_action(card: dict, idx: int = 0) -> dict:
    return {
        "kind": "card_reward",
        "selection": "pick",
        "surface": "card_reward",
        "action_id": f"reward.pick_card:{idx}",
        "card": card,
    }


def _compact_pick_action(card: dict, idx: int = 0) -> dict:
    return {
        "surface": "card_reward",
        "action_id": f"card_reward:{idx}",
        "choice_index": idx,
        "card_id": card["id"],
        "card_title": card.get("title"),
        "title": card.get("title"),
        "card_type": card.get("type"),
        "card_cost": card.get("cost"),
        "card_effect_profile": card.get("card_effect_profile"),
    }


def test_low_block_bad_pick_overrides_to_clear_block_upgrade():
    from muzero.training.card_reward_pick_quality_guard import apply_card_reward_pick_quality_guard

    weak_attack = _card("weak_attack", type_="Attack", damage=4)
    good_block = _card("good_block", type_="Skill", block=14)
    actions = [_pick_action(weak_attack, 0), _pick_action(good_block, 1)]
    stats: dict = {}

    new_idx = apply_card_reward_pick_quality_guard(
        action_idx=0,
        legal_actions=actions,
        full_legal_actions=actions,
        action_mask=np.array([1, 1], dtype=np.float32),
        raw_obs=_raw(_thin_low_block_deck()),
        search_stats=stats,
    )

    assert new_idx == 1
    assert stats["card_reward_pick_quality_guard_context"] == 1.0
    assert stats["card_reward_pick_quality_guard_selected_pick"] == 1.0
    assert stats["card_reward_pick_quality_guard_applicable"] == 1.0
    assert stats["card_reward_pick_quality_guard_override"] == 1.0
    assert stats["card_reward_pick_quality_guard_best_minus_selected"] >= 0.35
    assert stats["card_reward_pick_quality_guard_low_block"] == 1.0


def test_small_quality_gap_does_not_replace_existing_pick():
    from muzero.training.card_reward_pick_quality_guard import apply_card_reward_pick_quality_guard

    selected_block = _card("selected_block", type_="Skill", block=6)
    slightly_better_block = _card("slightly_better_block", type_="Skill", block=8)
    actions = [_pick_action(selected_block, 0), _pick_action(slightly_better_block, 1)]
    stats: dict = {}

    new_idx = apply_card_reward_pick_quality_guard(
        action_idx=0,
        legal_actions=actions,
        full_legal_actions=actions,
        action_mask=np.array([1, 1], dtype=np.float32),
        raw_obs=_raw(_thin_low_block_deck()),
        search_stats=stats,
    )

    assert new_idx == 0
    assert stats["card_reward_pick_quality_guard_applicable"] == 1.0
    assert 0.0 < stats["card_reward_pick_quality_guard_best_minus_selected"] < 0.35
    assert stats["card_reward_pick_quality_guard_override"] == 0.0


def test_healthy_deck_does_not_force_pick_retarget():
    from muzero.training.card_reward_pick_quality_guard import apply_card_reward_pick_quality_guard

    weak_attack = _card("weak_attack", type_="Attack", damage=4)
    good_block = _card("good_block", type_="Skill", block=14)
    actions = [_pick_action(weak_attack, 0), _pick_action(good_block, 1)]
    stats: dict = {}

    new_idx = apply_card_reward_pick_quality_guard(
        action_idx=0,
        legal_actions=actions,
        full_legal_actions=actions,
        action_mask=np.array([1, 1], dtype=np.float32),
        raw_obs=_raw(_healthy_deck()),
        search_stats=stats,
    )

    assert new_idx == 0
    assert stats["card_reward_pick_quality_guard_context"] == 1.0
    assert stats["card_reward_pick_quality_guard_no_deficit"] == 1.0
    assert stats["card_reward_pick_quality_guard_override"] == 0.0


def test_orphan_body_slam_is_not_selected_over_standalone_pick():
    from muzero.training.card_reward_pick_quality_guard import apply_card_reward_pick_quality_guard

    weak_attack = _card("weak_attack", type_="Attack", damage=4)
    body_slam = _card("CARD.BODY_SLAM", title="全身撞击", type_="Attack", damage=0)
    actions = [_pick_action(weak_attack, 0), _pick_action(body_slam, 1)]
    stats: dict = {}

    new_idx = apply_card_reward_pick_quality_guard(
        action_idx=0,
        legal_actions=actions,
        full_legal_actions=actions,
        action_mask=np.array([1, 1], dtype=np.float32),
        raw_obs=_raw([_card(f"strike_{idx}", type_="Attack", damage=6) for idx in range(8)]),
        search_stats=stats,
    )

    assert new_idx == 0
    assert stats["card_reward_pick_quality_guard_pick_available"] == 1.0
    assert stats["card_reward_pick_quality_guard_accepted_available"] == 1.0
    assert stats["card_reward_pick_quality_guard_override"] == 0.0
    assert stats["card_reward_pick_quality_guard_best_minus_selected"] == 0.0


def test_compact_only_contract_can_retarget_bad_pick():
    from muzero.training.card_reward_pick_quality_guard import apply_card_reward_pick_quality_guard

    weak_attack = _card("weak_attack", type_="Attack", damage=4)
    good_block = _card("good_block", type_="Skill", block=14)
    actions = [_compact_pick_action(weak_attack, 0), _compact_pick_action(good_block, 1)]
    stats: dict = {}

    new_idx = apply_card_reward_pick_quality_guard(
        action_idx=0,
        legal_actions=actions,
        full_legal_actions=None,
        action_mask=np.array([1, 1], dtype=np.float32),
        raw_obs=_raw(_thin_low_block_deck()),
        search_stats=stats,
    )

    assert new_idx == 1
    assert stats["card_reward_pick_quality_guard_context"] == 1.0
    assert stats["card_reward_pick_quality_guard_override"] == 1.0
    assert stats["card_reward_pick_quality_guard_alignment_error"] == 0.0


def test_build_guard_integration_runs_pick_quality_after_skip_guard():
    from muzero.train import MuZeroTrainer

    weak_attack = _card("weak_attack", type_="Attack", damage=4)
    good_block = _card("good_block", type_="Skill", block=14)
    actions = [_skip_action(), _pick_action(weak_attack, 0), _pick_action(good_block, 1)]
    trainer = MuZeroTrainer.__new__(MuZeroTrainer)
    trainer.env = SimpleNamespace(
        unwrapped=SimpleNamespace(
            _last_obs_raw=_raw(_thin_low_block_deck()),
            _legal_actions=actions,
        )
    )
    stats: dict = {}

    new_idx = trainer._apply_build_action_hard_guards(
        action_idx=1,
        legal_actions=actions,
        action_mask=np.array([1, 1, 1], dtype=np.float32),
        search_stats=stats,
    )

    assert new_idx == 2
    assert stats["card_reward_guard_context"] == 1.0
    assert stats["card_reward_guard_selected_pick"] == 1.0
    assert stats["card_reward_guard_override"] == 0.0
    assert stats["card_reward_pick_quality_guard_applied"] == 1.0
    assert stats["card_reward_pick_quality_guard_override"] == 1.0


class _CaptureWriter:
    def __init__(self) -> None:
        self.scalars: list[tuple[str, float, int]] = []

    def add_scalar(self, tag: str, value: float, step: int) -> None:
        self.scalars.append((tag, float(value), int(step)))


def test_pick_quality_metrics_are_registered_for_sync_and_async_tb_paths():
    from muzero.training.async_telemetry import log_async_episode_scalars
    from muzero.training.card_reward_pick_quality_guard import (
        CARD_REWARD_PICK_QUALITY_GUARD_SEARCH_SUFFIXES,
        card_reward_pick_quality_guard_metric_keys,
    )

    keys = set(card_reward_pick_quality_guard_metric_keys())
    assert "card_reward_pick_quality_guard_override" in keys
    assert (
        CARD_REWARD_PICK_QUALITY_GUARD_SEARCH_SUFFIXES["card_reward_pick_quality_guard_override"]
        == "card_reward_pick_quality_guard_override_rate"
    )
    assert (
        CARD_REWARD_PICK_QUALITY_GUARD_SEARCH_SUFFIXES["card_reward_pick_quality_guard_best_minus_selected"]
        == "card_reward_pick_quality_guard_best_minus_selected_mean"
    )

    writer = _CaptureWriter()
    trainer = SimpleNamespace(writer=writer, episode_count=7, total_steps=123)
    log_async_episode_scalars(
        trainer=trainer,
        actor_index=0,
        actor_completed_episodes=[1],
        episode_metrics={
            "decision_total": 1,
            "decision_counts": {"build": 1},
            "domain_search_means": {"build": {"card_reward_pick_quality_guard_override": 1.0}},
        },
    )
    assert ("search/build/card_reward_pick_quality_guard_override_rate", 1.0, 7) in writer.scalars

    self_play_source = (ROOT / "muzero" / "training" / "self_play.py").read_text(encoding="utf-8")
    assert "metric_name_map.update(CARD_REWARD_PICK_QUALITY_GUARD_SEARCH_SUFFIXES)" in self_play_source
