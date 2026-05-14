"""Route safety guard tests (Act1 recovery 2026-05-10).

These tests keep the hard-guard helper pure and deterministic.  The trainer
still owns action-mask and bridge-action alignment checks; this module verifies
that the route heuristic can identify "policy picked a high-risk route while a
safe route exists" without depending on live bridge state.
"""
from __future__ import annotations

import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from sts2_env.route_heuristic import choose_route_safety_override, score_route_action


def _b(*, risk: float, score: float = 0.0, summary: bool = True, forced: float = 0.0, immediate: float = 0.0, low_hp: float = 0.0):
    return {
        "summary_used": summary,
        "risk_class": risk,
        "score": score,
        "forced_elite_count": forced,
        "forced_elite_penalty": 1.0 if forced else 0.0,
        "immediate_elite_count": immediate,
        "no_rest_before_elite_penalty": 0.0,
        "low_hp_flag": low_hp,
    }


def test_choose_route_safety_override_swaps_high_risk_to_safe():
    ranked = [
        _b(risk=4, score=-2.0, forced=1.0, low_hp=1.0),
        _b(risk=1, score=0.5),
        _b(risk=2, score=2.0),
    ]
    out = choose_route_safety_override(ranked, 0)
    assert out["applicable"] is True
    assert out["safe_available"] is True
    assert out["override"] is True
    assert out["override_idx"] == 1
    assert out["selected_risk_class"] == 4.0
    assert out["final_risk_class"] == 1.0
    assert out["low_hp_forced"] is True


def test_choose_route_safety_override_no_safe_downgrades_to_lower_risk():
    ranked = [
        _b(risk=4, score=-2.0, forced=1.0, low_hp=1.0),
        _b(risk=2, score=1.0),
        _b(risk=3, score=2.0),
    ]
    out = choose_route_safety_override(ranked, 0)
    assert out["applicable"] is True
    assert out["safe_available"] is False
    assert out["lower_risk_available"] is True
    assert out["override"] is True
    assert out["override_idx"] == 1
    assert out["final_risk_class"] == 2.0
    assert out["reason"] == "override_high_risk_to_lower_risk"


def test_choose_route_safety_override_no_lower_risk_no_swap():
    ranked = [
        _b(risk=4, score=-2.0, forced=1.0, low_hp=1.0),
        _b(risk=4, score=1.0),
        _b(risk=5, score=2.0),
    ]
    out = choose_route_safety_override(ranked, 0)
    assert out["applicable"] is True
    assert out["safe_available"] is False
    assert out["lower_risk_available"] is False
    assert out["override"] is False
    assert out["override_idx"] is None
    assert out["reason"] == "no_lower_risk_alternative"


def test_choose_route_safety_override_selected_summary_unused_no_swap():
    out = choose_route_safety_override([_b(risk=4, summary=False), _b(risk=0)], 0)
    assert out["applicable"] is False
    assert out["override"] is False
    assert out["reason"] == "selected_unscored"


def test_choose_route_safety_override_picks_lowest_risk_then_highest_score():
    ranked = [
        _b(risk=3, score=-1.0, forced=1.0),
        _b(risk=1, score=0.1),
        _b(risk=0, score=-5.0),
        _b(risk=1, score=9.0),
    ]
    out = choose_route_safety_override(ranked, 0)
    # risk=0 beats a much better-scored risk=1 route; this is a hard safety
    # guard, not a soft heuristic preference.
    assert out["override_idx"] == 2
    assert out["final_risk_class"] == 0.0

    ranked[2]["risk_class"] = 1.0
    out2 = choose_route_safety_override(ranked, 0)
    # Same risk bucket => highest score wins.
    assert out2["override_idx"] == 3


def test_choose_route_safety_override_handles_none_entries():
    ranked = [None, _b(risk=4, score=-1.0, forced=1.0), None, _b(risk=1, score=0.0)]
    out = choose_route_safety_override(ranked, 1)
    assert out["override"] is True
    assert out["override_idx"] == 3


def test_route_safety_guard_cli_and_metric_plumbing_present():
    train_src = (ROOT / "muzero" / "train.py").read_text(encoding="utf-8-sig")
    trainer_src = (ROOT / "muzero" / "training" / "trainer.py").read_text(encoding="utf-8-sig")
    cli_src = (ROOT / "muzero" / "training" / "cli_main.py").read_text(encoding="utf-8-sig")
    cli_args_src = (ROOT / "muzero" / "training" / "cli_args.py").read_text(encoding="utf-8-sig")
    async_telemetry_src = (ROOT / "muzero" / "training" / "async_telemetry.py").read_text(encoding="utf-8-sig")
    self_play_src = (ROOT / "muzero" / "training" / "self_play.py").read_text(encoding="utf-8-sig")
    combined_src = train_src + "\n" + trainer_src + "\n" + cli_src + "\n" + cli_args_src + "\n" + async_telemetry_src + "\n" + self_play_src
    assert "--route-safety-guard" in cli_args_src
    assert "route_safety_guard=args.route_safety_guard" in cli_src
    assert "self.route_safety_guard_enabled = bool(route_safety_guard)" in trainer_src
    for key in (
        "route_safety_guard_applied",
        "route_safety_guard_lower_risk_available",
        "route_safety_guard_alignment_error",
        "route_safety_guard_selected_risk_class",
        "route_safety_guard_final_risk_class",
    ):
        assert key in combined_src
    # Synchronous TB map and async TB map should both expose the metric key.
    assert "route_safety_guard_applied_rate" in self_play_src
    assert "route_safety_guard_applied_rate" in async_telemetry_src
    assert "route_safety_guard_lower_risk_available_rate" in self_play_src
    assert "route_safety_guard_lower_risk_available_rate" in async_telemetry_src
    assert "route_safety_guard_invalid_obs_rate" in self_play_src
    assert "route_safety_guard_invalid_obs_rate" in async_telemetry_src


def test_route_safety_guard_fails_open_on_invalid_hp_obs():
    from muzero.train import MuZeroTrainer

    trainer = MuZeroTrainer.__new__(MuZeroTrainer)
    trainer.route_safety_guard_enabled = True
    raw = {
        "run": {"floor": 7},
        # Missing player.current_hp/max_hp is an observation-contract problem.
        # The guard must record it and leave the model's route choice alone.
        "player": {"gold": 80, "potions": []},
    }
    legal_actions = [
        {
            "action_id": "map:elite",
            "kind": "map",
            "route_summary": {
                "count_elite": 1,
                "count_monster": 1,
                "count_rest_site": 0,
                "can_reach_rest_site_before_elite": False,
                "next_elite_steps": 1,
            },
        },
        {
            "action_id": "map:rest",
            "kind": "map",
            "route_summary": {
                "count_elite": 0,
                "count_monster": 1,
                "count_rest_site": 1,
                "can_reach_rest_site_before_elite": False,
                "next_elite_steps": None,
            },
        },
    ]
    trainer.env = SimpleNamespace(unwrapped=SimpleNamespace(_last_obs_raw=raw, _legal_actions=legal_actions))
    search_stats: dict = {}

    new_idx = trainer._apply_route_action_hard_guards(
        action_idx=0,
        legal_actions=legal_actions,
        action_mask=np.array([1, 1], dtype=np.float32),
        search_stats=search_stats,
    )

    assert new_idx == 0
    assert search_stats["route_safety_guard_enabled"] == 1.0
    assert search_stats["route_safety_guard_invalid_obs"] == 1.0
    assert search_stats["route_safety_guard_applied"] == 0.0


def test_route_heuristic_invalid_hp_does_not_set_low_hp_flags():
    out = score_route_action(
        route_summary={
            "count_elite": 1,
            "count_monster": 1,
            "count_rest_site": 0,
            "can_reach_rest_site_before_elite": False,
            "next_elite_steps": 1,
        },
        deck_quality={},
        hp=0,
        max_hp=0,
        gold=0,
        potion_count=0,
        act=1,
        floor=6,
    )

    assert out["summary_used"] is True
    assert out["low_hp_flag"] == 0.0
    assert out["low_hp_elite_flag"] == 0.0
