"""Build-domain hard-guard tests for Act1 recovery.

The production failure mode was not route selection: after reaching a campfire
at low HP, the policy repeatedly picked non-heal options and entered the Act1
boss around 40% HP.  These tests exercise the small trainer-side override that
keeps low-HP campfires on the concrete REST/HEAL option when it is legal.
"""
from __future__ import annotations

import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def _trainer_with_raw(raw_obs: dict, full_actions: list[dict]):
    from muzero.train import MuZeroTrainer

    trainer = MuZeroTrainer.__new__(MuZeroTrainer)
    trainer.build_hard_guard_policy = "full"
    trainer.env = SimpleNamespace(
        unwrapped=SimpleNamespace(
            _last_obs_raw=raw_obs,
            _legal_actions=full_actions,
        )
    )
    return trainer


def test_low_hp_rest_site_smith_overrides_to_rest():
    raw_obs = {"player": {"hp": 39, "max_hp": 80}}
    full_actions = [
        {
            "kind": "rest_site",
            "action_id": "rest_site:smith",
            "label": "Rest Site: Smith",
            "option": {"option_type": "SMITH", "title": "Smith"},
        },
        {
            "kind": "rest_site",
            "action_id": "rest_site:rest",
            "label": "Rest",
            "option": {"option_type": "REST", "title": "Rest"},
        },
    ]
    # Compact actions intentionally omit the option payload; the guard should
    # read the live full bridge actions by index.
    compact_actions = [{"kind": "rest_site", "action_id": "rest_site:smith"}, {"kind": "rest_site", "action_id": "rest_site:rest"}]
    trainer = _trainer_with_raw(raw_obs, full_actions)
    search_stats: dict = {}

    new_idx = trainer._apply_build_action_hard_guards(
        action_idx=0,
        legal_actions=compact_actions,
        action_mask=np.array([1, 1], dtype=np.float32),
        search_stats=search_stats,
    )

    assert new_idx == 1
    assert search_stats["build_safety_guard_enabled"] == 1.0
    assert search_stats["build_safety_guard_rest_low_hp_applicable"] == 1.0
    assert search_stats["build_safety_guard_rest_available"] == 1.0
    assert search_stats["build_safety_guard_rest_selected_non_heal_low_hp"] == 1.0
    assert search_stats["build_safety_guard_rest_applied"] == 1.0
    assert search_stats["build_safety_guard_rest_override"] == 1.0


def test_high_hp_rest_site_rest_overrides_to_smith():
    raw_obs = {"player": {"hp": 64, "max_hp": 80}}
    full_actions = [
        {
            "kind": "rest_site",
            "action_id": "rest_site:smith",
            "label": "Rest site option 0: 锻造",
            "option": {
                "option_id": "SMITH",
                "option_type": "SmithRestSiteOption",
                "title": "锻造",
                "description": "升级你牌组中的1张牌。",
            },
        },
        {
            "kind": "rest_site",
            "action_id": "rest_site:rest",
            "label": "Rest site option 1: 休息",
            "option": {
                "option_id": "HEAL",
                "option_type": "HealRestSiteOption",
                "title": "休息",
                "description": "回复18点生命值。",
            },
        },
    ]
    # Compact actions may not carry option identity; the guard must use the
    # full live bridge actions by index and still return the compact index.
    compact_actions = [{"kind": "rest_site", "action_id": "rest_site:0"}, {"kind": "rest_site", "action_id": "rest_site:1"}]
    trainer = _trainer_with_raw(raw_obs, full_actions)
    search_stats: dict = {}

    new_idx = trainer._apply_build_action_hard_guards(
        action_idx=1,
        legal_actions=compact_actions,
        action_mask=np.array([1, 1], dtype=np.float32),
        search_stats=search_stats,
    )

    assert new_idx == 0
    assert search_stats["rest_site_smith_guard_applicable"] == 1.0
    assert search_stats["rest_site_smith_guard_selected_heal_safe_hp"] == 1.0
    assert search_stats["rest_site_smith_guard_smith_available"] == 1.0
    assert search_stats["rest_site_smith_guard_applied"] == 1.0
    assert search_stats["rest_site_smith_guard_override"] == 1.0
    # The low-HP survival rest override must not fight the high-HP smith guard.
    assert search_stats["build_safety_guard_rest_low_hp_applicable"] == 0.0
    assert search_stats["build_safety_guard_rest_applied"] == 0.0


def test_mid_hp_rest_site_rest_is_not_forced_to_smith():
    raw_obs = {"player": {"hp": 55, "max_hp": 80}}
    full_actions = [
        {
            "kind": "rest_site",
            "action_id": "rest_site:smith",
            "option": {"option_id": "SMITH", "option_type": "SmithRestSiteOption", "title": "Smith"},
        },
        {
            "kind": "rest_site",
            "action_id": "rest_site:rest",
            "option": {"option_id": "HEAL", "option_type": "HealRestSiteOption", "title": "Rest"},
        },
    ]
    trainer = _trainer_with_raw(raw_obs, full_actions)
    search_stats: dict = {}

    new_idx = trainer._apply_build_action_hard_guards(
        action_idx=1,
        legal_actions=full_actions,
        action_mask=np.array([1, 1], dtype=np.float32),
        search_stats=search_stats,
    )

    assert new_idx == 1
    assert search_stats["rest_site_smith_guard_applied"] == 0.0
    assert search_stats["build_safety_guard_rest_applied"] == 0.0


def test_high_hp_rest_site_selected_smith_is_kept():
    raw_obs = {"player": {"hp": 64, "max_hp": 80}}
    full_actions = [
        {
            "kind": "rest_site",
            "action_id": "rest_site:smith",
            "option": {"option_id": "SMITH", "option_type": "SmithRestSiteOption", "title": "Smith"},
        },
        {
            "kind": "rest_site",
            "action_id": "rest_site:rest",
            "option": {"option_id": "HEAL", "option_type": "HealRestSiteOption", "title": "Rest"},
        },
    ]
    trainer = _trainer_with_raw(raw_obs, full_actions)
    search_stats: dict = {}

    new_idx = trainer._apply_build_action_hard_guards(
        action_idx=0,
        legal_actions=full_actions,
        action_mask=np.array([1, 1], dtype=np.float32),
        search_stats=search_stats,
    )

    assert new_idx == 0
    assert search_stats["rest_site_smith_guard_applicable"] == 1.0
    assert search_stats["rest_site_smith_guard_selected_heal_safe_hp"] == 0.0
    assert search_stats["rest_site_smith_guard_applied"] == 0.0


def test_low_hp_rest_site_live_bridge_heal_option_overrides_indexed_smith():
    raw_obs = {"player": {"hp": 39, "max_hp": 80}}
    full_actions = [
        {
            "kind": "rest_site",
            "action_id": "rest_site:0",
            "label": "Rest site option 0: 锻造",
            "option": {
                "option_id": "SMITH",
                "option_type": "SmithRestSiteOption",
                "title": "锻造",
                "description": "升级你牌组中的1张牌。",
            },
        },
        {
            "kind": "rest_site",
            "action_id": "rest_site:1",
            "label": "Rest site option 1: 休息",
            "option": {
                "option_id": "HEAL",
                "option_type": "HealRestSiteOption",
                "title": "休息",
                "description": "回复18点生命值。",
            },
        },
    ]
    trainer = _trainer_with_raw(raw_obs, full_actions)
    search_stats: dict = {}

    new_idx = trainer._apply_build_action_hard_guards(
        action_idx=0,
        legal_actions=full_actions,
        action_mask=np.array([1, 1], dtype=np.float32),
        search_stats=search_stats,
    )

    assert new_idx == 1
    assert search_stats["build_safety_guard_rest_available"] == 1.0
    assert search_stats["build_safety_guard_rest_applied"] == 1.0


def test_low_hp_rest_site_live_bridge_nested_payload_heal_option_overrides():
    raw_obs = {"player": {"hp": 39, "max_hp": 80}}
    full_actions = [
        {
            "action_id": "wrapped:0",
            "payload": {
                "kind": "rest_site",
                "action_id": "rest_site:0",
                "label": "Rest site option 0: Smith",
                "option": {
                    "option_id": "SMITH",
                    "option_type": "SmithRestSiteOption",
                    "title": "Smith",
                },
            },
        },
        {
            "action_id": "wrapped:1",
            "payload": {
                "kind": "rest_site",
                "action_id": "rest_site:1",
                "label": "Rest site option 1: Rest",
                "option": {
                    "option_id": "HEAL",
                    "option_type": "HealRestSiteOption",
                    "description": "回复18点生命值。",
                },
            },
        },
    ]
    trainer = _trainer_with_raw(raw_obs, full_actions)
    search_stats: dict = {}

    new_idx = trainer._apply_build_action_hard_guards(
        action_idx=0,
        legal_actions=full_actions,
        action_mask=np.array([1, 1], dtype=np.float32),
        search_stats=search_stats,
    )

    assert new_idx == 1
    assert search_stats["build_safety_guard_rest_available"] == 1.0
    assert search_stats["build_safety_guard_rest_applied"] == 1.0


def test_low_hp_rest_site_selected_rest_is_not_overridden():
    raw_obs = {"player": {"hp": 39, "max_hp": 80}}
    full_actions = [
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
    trainer = _trainer_with_raw(raw_obs, full_actions)
    search_stats: dict = {}

    new_idx = trainer._apply_build_action_hard_guards(
        action_idx=1,
        legal_actions=full_actions,
        action_mask=np.array([1, 1], dtype=np.float32),
        search_stats=search_stats,
    )

    assert new_idx == 1
    assert search_stats["build_safety_guard_rest_selected_heal"] == 1.0
    assert search_stats["build_safety_guard_rest_applied"] == 0.0


def test_low_hp_rest_site_proceed_is_not_guard_applicable():
    raw_obs = {"player": {"hp": 39, "max_hp": 80}}
    full_actions = [
        {
            "kind": "rest_site",
            "action_id": "rest_site:proceed",
            "canonical_text": "动作｜营火｜",
        },
    ]
    trainer = _trainer_with_raw(raw_obs, full_actions)
    search_stats: dict = {}

    new_idx = trainer._apply_build_action_hard_guards(
        action_idx=0,
        legal_actions=full_actions,
        action_mask=np.array([1], dtype=np.float32),
        search_stats=search_stats,
    )

    assert new_idx == 0
    assert search_stats["build_safety_guard_enabled"] == 1.0
    assert search_stats["build_safety_guard_rest_low_hp_applicable"] == 0.0
    assert search_stats["build_safety_guard_rest_available"] == 0.0
    assert search_stats["build_safety_guard_rest_selected_non_heal_low_hp"] == 0.0
    assert search_stats["build_safety_guard_rest_applied"] == 0.0


def test_build_rest_guard_fails_open_on_missing_hp():
    raw_obs = {"player": {"gold": 99}}
    full_actions = [
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
    trainer = _trainer_with_raw(raw_obs, full_actions)
    search_stats: dict = {}

    new_idx = trainer._apply_build_action_hard_guards(
        action_idx=0,
        legal_actions=full_actions,
        action_mask=np.array([1, 1], dtype=np.float32),
        search_stats=search_stats,
    )

    assert new_idx == 0
    assert search_stats["build_safety_guard_invalid_obs"] == 1.0
    assert search_stats["build_safety_guard_rest_applied"] == 0.0


def test_build_rest_guard_ignores_route_actions_with_future_rest_sites():
    raw_obs = {"player": {"hp": 30, "max_hp": 80}}
    route_actions = [
        {
            "kind": "map",
            "action_id": "map:to_rest_later",
            "route_summary": {"count_rest_site": 2},
        },
        {
            "kind": "map",
            "action_id": "map:to_monster",
            "route_summary": {"count_rest_site": 0},
        },
    ]
    trainer = _trainer_with_raw(raw_obs, route_actions)
    search_stats: dict = {}

    new_idx = trainer._apply_build_action_hard_guards(
        action_idx=0,
        legal_actions=route_actions,
        action_mask=np.array([1, 1], dtype=np.float32),
        search_stats=search_stats,
    )

    assert new_idx == 0
    assert search_stats["build_safety_guard_rest_low_hp_applicable"] == 0.0
    assert search_stats["build_safety_guard_rest_applied"] == 0.0


def test_deck_upgrade_close_overrides_to_best_upgrade_target():
    raw_obs = {"player": {"hp": 70, "max_hp": 80}}
    full_actions = [
        {
            "kind": "deck_upgrade",
            "action_id": "deck_upgrade:0",
            "card": {"id": "CARD.STRIKE_IRONCLAD", "title": "打击"},
        },
        {
            "kind": "deck_upgrade",
            "action_id": "deck_upgrade:1",
            "card": {"id": "CARD.BASH_IRONCLAD", "title": "痛击"},
        },
        {
            "kind": "deck_upgrade",
            "action_id": "deck_upgrade:close",
            "title": "关闭",
        },
    ]
    compact_actions = [
        {"kind": "deck_upgrade", "action_id": "deck_upgrade:0", "card": {"title": "打击"}},
        {"kind": "deck_upgrade", "action_id": "deck_upgrade:1", "card": {"title": "痛击"}},
        {"kind": "deck_upgrade", "action_id": "deck_upgrade:close", "title": "关闭"},
    ]
    trainer = _trainer_with_raw(raw_obs, full_actions)
    search_stats: dict = {}

    new_idx = trainer._apply_build_action_hard_guards(
        action_idx=2,
        legal_actions=compact_actions,
        action_mask=np.array([1, 1, 1], dtype=np.float32),
        search_stats=search_stats,
    )

    assert new_idx == 1
    assert search_stats["deck_upgrade_target_guard_context"] == 1.0
    assert search_stats["deck_upgrade_target_guard_selected_close"] == 1.0
    assert search_stats["deck_upgrade_target_guard_close_with_upgrade_available"] == 1.0
    assert search_stats["deck_upgrade_target_guard_applied"] == 1.0
    assert search_stats["deck_upgrade_target_guard_close_override"] == 1.0
    assert search_stats["deck_upgrade_target_guard_bash_available"] == 1.0


def test_deck_upgrade_close_without_targets_is_allowed():
    raw_obs = {"player": {"hp": 70, "max_hp": 80}}
    full_actions = [{"kind": "deck_upgrade", "action_id": "deck_upgrade:close", "title": "关闭"}]
    trainer = _trainer_with_raw(raw_obs, full_actions)
    search_stats: dict = {}

    new_idx = trainer._apply_build_action_hard_guards(
        action_idx=0,
        legal_actions=full_actions,
        action_mask=np.array([1], dtype=np.float32),
        search_stats=search_stats,
    )

    assert new_idx == 0
    assert search_stats["deck_upgrade_target_guard_context"] == 0.0
    assert search_stats["deck_upgrade_target_guard_applied"] == 0.0
    assert search_stats["deck_upgrade_target_guard_close_override"] == 0.0
