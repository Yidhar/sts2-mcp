"""Policy-mode tests for the hard-guard downgrade path.

The Act1 recovery work should stop turning every tactical/build concern into
an unconditional rule.  These tests lock the new split:

* ``full`` keeps legacy overrides by default for backwards compatibility;
* ``emergency`` keeps only narrow survival/protocol guards;
* ``off`` records telemetry but never rewrites the selected action.
"""

from __future__ import annotations

import sys
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def _trainer():
    from muzero.train import MuZeroTrainer

    trainer = MuZeroTrainer.__new__(MuZeroTrainer)
    trainer.log_dir = None
    trainer.episode_count = 0
    trainer.total_steps = 0
    trainer.env = SimpleNamespace(
        unwrapped=SimpleNamespace(
            _last_obs_raw={},
            _legal_actions=[],
        )
    )
    return trainer


def _identity_guard(self, *, action_idx: int, **_: object) -> int:
    return int(action_idx)


def test_cli_accepts_hard_guard_policy_modes():
    from muzero.training.cli_args import build_arg_parser

    args = build_arg_parser().parse_args(
        [
            "--combat-hard-guard-policy",
            "emergency",
            "--build-hard-guard-policy",
            "off",
        ]
    )

    assert args.combat_hard_guard_policy == "emergency"
    assert args.build_hard_guard_policy == "off"


def test_combat_hard_guard_off_returns_original_without_calling_guards():
    trainer = _trainer()
    trainer.combat_hard_guard_policy = "off"
    search_stats: dict[str, float] = {}

    # If the off mode accidentally falls through into guard execution, this
    # early emergency guard would raise.
    with patch.object(trainer, "_apply_x_cost_zero_guard", side_effect=AssertionError("guard called")):
        new_idx = trainer._apply_combat_action_hard_guards(
            action_idx=1,
            legal_actions=[{"kind": "play_card"}, {"kind": "end_turn"}],
            action_mask=np.array([1, 1], dtype=np.float32),
            raw_obs={},
            boss_ctx={},
            encounter="ENCOUNTER.SMOKE",
            search_stats=search_stats,
        )

    assert new_idx == 1
    assert search_stats["combat_hard_guard_policy_off"] == 1.0
    assert search_stats["combat_hard_guard_policy_emergency"] == 0.0
    assert search_stats["combat_hard_guard_policy_full"] == 0.0


def test_combat_hard_guard_emergency_skips_full_only_behavior_guards():
    trainer = _trainer()
    trainer.combat_hard_guard_policy = "emergency"
    search_stats: dict[str, float] = {}
    legal_actions = [{"kind": "play_card", "card": {"cost": 1}}, {"kind": "end_turn"}]
    mask = np.array([1, 1], dtype=np.float32)

    emergency_identity_guards = (
        "_apply_potion_discard_priority_guard",
        "_apply_kaiser_facing_guard",
        "_apply_insatiable_escape_guard",
        "_apply_x_cost_zero_guard",
        "_apply_elite_boss_lethal_endturn_guard",
        "_apply_urgent_endturn_guard",
        "_apply_full_energy_endturn_guard",
        "_apply_selection_loop_guard",
    )
    patchers = [
        patch.object(trainer, name, _identity_guard.__get__(trainer, type(trainer)))
        for name in emergency_identity_guards
    ]
    patchers.append(
        patch.object(
            trainer,
            "_apply_hp_cost_margin_guard",
            side_effect=AssertionError("full-only hp-cost guard called in emergency mode"),
        )
    )
    patchers.append(
        patch.object(
            trainer,
            "_apply_no_pressure_block_guard",
            side_effect=AssertionError("full-only no-pressure-block guard called in emergency mode"),
        )
    )

    try:
        for patcher in patchers:
            patcher.start()
        new_idx = trainer._apply_combat_action_hard_guards(
            action_idx=0,
            legal_actions=legal_actions,
            action_mask=mask,
            raw_obs={},
            boss_ctx={},
            encounter="ENCOUNTER.SMOKE",
            search_stats=search_stats,
        )
    finally:
        for patcher in reversed(patchers):
            patcher.stop()

    assert new_idx == 0
    assert search_stats["combat_hard_guard_policy_emergency"] == 1.0
    assert search_stats["combat_hard_guard_policy_full"] == 0.0
    assert search_stats["combat_hard_guard_policy_off"] == 0.0


def test_build_hard_guard_off_never_overrides_low_hp_campfire():
    trainer = _trainer()
    trainer.build_hard_guard_policy = "off"
    raw_obs = {"player": {"hp": 20, "max_hp": 80}}
    actions = [
        {
            "kind": "rest_site",
            "action_id": "rest_site:smith",
            "option": {"option_type": "SMITH", "title": "Smith"},
        },
        {
            "kind": "rest_site",
            "action_id": "rest_site:rest",
            "option": {"option_type": "REST", "title": "Rest"},
        },
    ]
    trainer.env = SimpleNamespace(unwrapped=SimpleNamespace(_last_obs_raw=raw_obs, _legal_actions=actions))
    search_stats: dict[str, float] = {}

    new_idx = trainer._apply_build_action_hard_guards(
        action_idx=0,
        legal_actions=actions,
        action_mask=np.array([1, 1], dtype=np.float32),
        search_stats=search_stats,
    )

    assert new_idx == 0
    assert search_stats["build_hard_guard_policy_off"] == 1.0
    assert search_stats["build_safety_guard_enabled"] == 0.0
    assert search_stats["build_safety_guard_rest_applied"] == 0.0


def test_build_hard_guard_emergency_keeps_low_hp_rest_only():
    trainer = _trainer()
    trainer.build_hard_guard_policy = "emergency"
    raw_obs = {"player": {"hp": 20, "max_hp": 80}}
    actions = [
        {
            "kind": "rest_site",
            "action_id": "rest_site:smith",
            "option": {"option_type": "SMITH", "title": "Smith"},
        },
        {
            "kind": "rest_site",
            "action_id": "rest_site:rest",
            "option": {"option_type": "REST", "title": "Rest"},
        },
    ]
    trainer.env = SimpleNamespace(unwrapped=SimpleNamespace(_last_obs_raw=raw_obs, _legal_actions=actions))
    search_stats: dict[str, float] = {}

    new_idx = trainer._apply_build_action_hard_guards(
        action_idx=0,
        legal_actions=actions,
        action_mask=np.array([1, 1], dtype=np.float32),
        search_stats=search_stats,
    )

    assert new_idx == 1
    assert search_stats["build_hard_guard_policy_emergency"] == 1.0
    assert search_stats["build_safety_guard_enabled"] == 1.0
    assert search_stats["build_safety_guard_rest_applied"] == 1.0
    assert search_stats["rest_site_smith_guard_applied"] == 0.0


def test_build_hard_guard_emergency_does_not_force_high_hp_smith():
    trainer = _trainer()
    trainer.build_hard_guard_policy = "emergency"
    raw_obs = {"player": {"hp": 70, "max_hp": 80}}
    actions = [
        {
            "kind": "rest_site",
            "action_id": "rest_site:smith",
            "option": {"option_type": "SMITH", "title": "Smith"},
        },
        {
            "kind": "rest_site",
            "action_id": "rest_site:rest",
            "option": {"option_type": "REST", "title": "Rest"},
        },
    ]
    trainer.env = SimpleNamespace(unwrapped=SimpleNamespace(_last_obs_raw=raw_obs, _legal_actions=actions))
    search_stats: dict[str, float] = {}

    new_idx = trainer._apply_build_action_hard_guards(
        action_idx=1,
        legal_actions=actions,
        action_mask=np.array([1, 1], dtype=np.float32),
        search_stats=search_stats,
    )

    assert new_idx == 1
    assert search_stats["build_hard_guard_policy_emergency"] == 1.0
    assert search_stats["rest_site_smith_guard_enabled"] == 0.0
    assert search_stats["rest_site_smith_guard_applied"] == 0.0
    assert search_stats["build_safety_guard_rest_applied"] == 0.0
