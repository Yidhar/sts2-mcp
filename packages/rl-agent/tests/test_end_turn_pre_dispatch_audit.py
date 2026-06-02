"""Focused tests for the pre-dispatch EndTurn audit.

The audit is intentionally built from the exact legal frontier immediately
before env.step(EndTurn).  These tests make sure it catches the two cases that
made old logs unreliable: full-energy EndTurn despite legal progress, and raw
hand/playable-card evidence that the legal surface failed to expose play_card.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from muzero.diagnostics.end_turn_pre_dispatch import (
    build_end_turn_pre_dispatch_audit,
    dump_end_turn_pre_dispatch_audit,
)


class AuditOwner:
    def __init__(self, log_dir: Path | None = None) -> None:
        self.log_dir = str(log_dir) if log_dir is not None else None
        self.total_steps = 123
        self.episode_count = 7
        self._end_turn_pre_dispatch_audit_count = 0
        self._end_turn_pre_dispatch_audit_max = 100000
        self._end_turn_pre_dispatch_audit_disabled = False

    def _diagnostic_jsonl_path(self, filename: str):
        if not self.log_dir:
            return None
        return Path(self.log_dir) / "diagnostics" / filename

    def _semantic_family(self, action):
        if not isinstance(action, dict):
            return ""
        semantic = action.get("semantic") if isinstance(action.get("semantic"), dict) else {}
        return str(semantic.get("family") or action.get("kind") or "").lower()

    def _combat_energy(self, _encoded_obs, raw_obs):
        combat = raw_obs.get("combat") if isinstance(raw_obs, dict) and isinstance(raw_obs.get("combat"), dict) else {}
        player = raw_obs.get("player") if isinstance(raw_obs, dict) and isinstance(raw_obs.get("player"), dict) else {}
        return float(combat.get("energy", player.get("energy", 0.0)) or 0.0)

    def _action_cost_value(self, action):
        if not isinstance(action, dict):
            return 0.0
        card = action.get("card") if isinstance(action.get("card"), dict) else {}
        return float(action.get("cost", card.get("cost", card.get("cost_for_turn", 0.0))) or 0.0)

    def _x_cost_diagnostic(self, _action, _energy):
        return {"x_cost_bad": 0.0}

    def _is_action_confirmed_lethal(self, action, _raw_obs):
        return bool(isinstance(action, dict) and action.get("lethal"))

    def _classify_positive_combat_action(self, action, *_args, **_kwargs):
        if isinstance(action, dict) and action.get("positive") is False:
            return {"positive": False}
        return {"positive": True}

    def _action_metric(self, action, key):
        if not isinstance(action, dict):
            return 0.0
        card = action.get("card") if isinstance(action.get("card"), dict) else {}
        return float(action.get(key, card.get(key, 0.0)) or 0.0)

    def _action_numeric_value(self, action, keys):
        if not isinstance(action, dict):
            return 0.0
        card = action.get("card") if isinstance(action.get("card"), dict) else {}
        for key in keys:
            if key in action:
                return float(action.get(key) or 0.0)
            if key in card:
                return float(card.get(key) or 0.0)
        return 0.0

    def _action_immediate_impact(self, action):
        if not isinstance(action, dict):
            return 0.0
        return float(action.get("impact", action.get("damage", 0.0)) or 0.0)

    def _action_roles(self, action):
        if not isinstance(action, dict):
            return set()
        roles = action.get("roles")
        if isinstance(roles, (list, tuple, set)):
            return set(str(role) for role in roles)
        return {"attack"} if float(action.get("damage", 0.0) or 0.0) > 0.0 else set()

    def _incoming_damage_pressure(self, raw_obs):
        combat = raw_obs.get("combat") if isinstance(raw_obs, dict) and isinstance(raw_obs.get("combat"), dict) else {}
        player = raw_obs.get("player") if isinstance(raw_obs, dict) and isinstance(raw_obs.get("player"), dict) else {}
        return (
            float(combat.get("incoming_damage", 0.0) or 0.0),
            float(player.get("block", combat.get("block", 0.0)) or 0.0),
            float(player.get("hp", player.get("current_hp", 70.0)) or 70.0),
        )


def raw_obs(*, energy: float = 3.0, max_energy: float = 3.0, hand: list[dict] | None = None) -> dict:
    return {
        "phase": "combat",
        "run": {"floor": 11, "act_id": 1, "room_type": "Monster"},
        "combat": {
            "energy": energy,
            "max_energy": max_energy,
            "incoming_damage": 0,
            "hand": hand
            if hand is not None
            else [
                {
                    "id": "CARD.STRIKE",
                    "title": "Strike",
                    "type": "Attack",
                    "cost": 1,
                    "cost_for_turn": 1,
                    "is_playable": True,
                    "damage": 6,
                }
            ],
        },
        "player": {"hp": 70, "max_hp": 80, "block": 0, "max_energy": max_energy},
    }


def strike_action() -> dict:
    return {
        "action_id": "play_card:0",
        "kind": "play_card",
        "title": "Strike",
        "card": {"title": "Strike", "cost": 1, "damage": 6},
        "cost": 1,
        "damage": 6,
        "roles": ["attack"],
    }


def end_turn_action() -> dict:
    return {"action_id": "end_turn", "kind": "end_turn", "title": "End Turn"}


def build(owner: AuditOwner, *, raw=None, actions=None, mask=None, chosen_idx=1):
    if raw is None:
        raw = raw_obs()
    if actions is None:
        actions = [strike_action(), end_turn_action()]
    if mask is None:
        mask = np.ones(len(actions), dtype=np.float32)
    return build_end_turn_pre_dispatch_audit(
        owner,
        encoded_obs={},
        raw_obs=raw,
        action_mask=mask,
        legal_actions=actions,
        chosen_idx=chosen_idx,
        search_policy=np.array([0.25, 0.75], dtype=np.float32),
        search_stats={},
        pre_step_info={"phase": "combat", "raw_legal_action_count": len(actions)},
        encounter="ENCOUNTER.LIVING_FOG",
        tier="normal",
    )


def test_full_energy_legal_safe_progress_endturn_is_flagged() -> None:
    payload = build(AuditOwner())

    assert payload["selected_family"] == "end_turn"
    assert payload["player"]["energy"] == 3.0
    assert payload["player"]["energy_ratio"] == 1.0
    assert payload["counts"]["mask_legal_count"] == 2
    assert payload["counts"]["legal_play_card_action_count"] == 1
    assert payload["counts"]["affordable_play_card_action_count"] == 1
    assert payload["counts"]["ui_affordable_hand_card_count"] == 1
    assert payload["counts"]["safe_progress_candidate_count"] == 1
    assert payload["flags"]["full_energy_like"] is True
    assert payload["flags"]["full_energy_skip_suspect"] is True
    assert payload["flags"]["full_energy_skip_with_playable_hand"] is True
    assert payload["safe_progress_candidates"][0]["index"] == 0


def test_raw_hand_playable_but_no_legal_play_card_marks_legal_generation_gap() -> None:
    actions = [end_turn_action(), {"action_id": "wait", "kind": "wait"}]
    payload = build(AuditOwner(), actions=actions, chosen_idx=0, mask=np.array([1, 1], dtype=np.float32))

    assert payload["counts"]["ui_affordable_hand_card_count"] == 1
    assert payload["counts"]["legal_play_card_action_count"] == 0
    assert payload["flags"]["legal_generation_gap_suspect"] is True
    assert payload["flags"]["raw_hand_legal_surface_mismatch"] is True
    assert payload["flags"]["full_energy_skip_suspect"] is True


def test_singleton_endturn_with_playable_raw_hand_is_marked_as_frontier_suspect() -> None:
    payload = build(
        AuditOwner(),
        actions=[end_turn_action()],
        chosen_idx=0,
        mask=np.array([1], dtype=np.float32),
    )

    assert payload["counts"]["mask_legal_count"] == 1
    assert payload["counts"]["ui_affordable_hand_card_count"] == 1
    assert payload["flags"]["singleton_frontier_suspect"] is True
    assert payload["flags"]["legal_generation_gap_suspect"] is True


def test_status_and_curse_hand_does_not_false_positive_as_playable_gap() -> None:
    status_hand = [
        {"id": "STATUS.DAZED", "title": "Dazed", "type": "Status", "cost": 0, "is_playable": False},
        {"id": "CURSE.PAIN", "title": "Pain", "type": "Curse", "cost": 0, "is_playable": False},
    ]
    payload = build(
        AuditOwner(),
        raw=raw_obs(hand=status_hand),
        actions=[end_turn_action()],
        chosen_idx=0,
        mask=np.array([1], dtype=np.float32),
    )

    assert payload["counts"]["raw_hand_card_count"] == 2
    assert payload["counts"]["ui_affordable_hand_card_count"] == 0
    assert payload["flags"]["legal_generation_gap_suspect"] is False
    assert payload["flags"]["full_energy_skip_suspect"] is False
    assert payload["flags"]["singleton_frontier_suspect"] is False


def test_full_energy_threshold_is_not_loose_two_of_three_energy() -> None:
    payload = build(AuditOwner(), raw=raw_obs(energy=2.0, max_energy=3.0))

    assert payload["player"]["energy"] == 2.0
    assert 0.66 < payload["player"]["energy_ratio"] < 0.67
    assert payload["flags"]["full_energy_like"] is False
    assert payload["flags"]["full_energy_skip_suspect"] is False


def test_dump_writes_jsonl_and_honors_disabled_flag(tmp_path: Path) -> None:
    owner = AuditOwner(log_dir=tmp_path)

    payload = dump_end_turn_pre_dispatch_audit(
        owner,
        encoded_obs={},
        raw_obs=raw_obs(),
        action_mask=np.array([1, 1], dtype=np.float32),
        legal_actions=[strike_action(), end_turn_action()],
        chosen_idx=1,
        search_policy=np.array([0.2, 0.8], dtype=np.float32),
        search_stats={},
        pre_step_info={},
        encounter="ENCOUNTER.LIVING_FOG",
        tier="normal",
    )

    assert payload is not None
    path = tmp_path / "diagnostics" / "end_turn_pre_dispatch_audit.jsonl"
    assert path.exists()
    rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]
    assert len(rows) == 1
    assert rows[0]["flags"]["full_energy_skip_suspect"] is True
    assert owner._end_turn_pre_dispatch_audit_count == 1

    owner._end_turn_pre_dispatch_audit_disabled = True
    assert dump_end_turn_pre_dispatch_audit(
        owner,
        encoded_obs={},
        raw_obs=raw_obs(),
        action_mask=np.array([1], dtype=np.float32),
        legal_actions=[end_turn_action()],
        chosen_idx=0,
        search_policy=None,
        search_stats={},
        pre_step_info={},
    ) is None
    assert owner._end_turn_pre_dispatch_audit_count == 1
