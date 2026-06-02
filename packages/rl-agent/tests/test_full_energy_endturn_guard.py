"""Focused tests for the full-energy EndTurn hard guard.

These tests cover the live failure mode reported from full-run play: the final
post-search action is EndTurn while the same legal frontier still has full
energy and a safe playable card.  The guard must rewrite only when the legal
surface exposes a safe action, and it must diagnose (not hide) bridge/legal
surface gaps when the raw hand looks playable but no play_card action is legal.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from muzero.combat_quality.full_energy_endturn_guard import FullEnergyEndTurnGuardMixin
from muzero.combat_quality.progress_candidates import ProgressCandidate


class GuardStub(FullEnergyEndTurnGuardMixin):
    def __init__(self) -> None:
        self.records: list[dict[str, object]] = []

    def _semantic_family(self, action):
        if not isinstance(action, dict):
            return ""
        semantic = action.get("semantic") if isinstance(action.get("semantic"), dict) else {}
        return str(semantic.get("family") or action.get("kind") or "").lower()

    def _combat_energy(self, _encoded_obs, raw_obs):
        if isinstance(raw_obs, dict):
            combat = raw_obs.get("combat") if isinstance(raw_obs.get("combat"), dict) else {}
            player = raw_obs.get("player") if isinstance(raw_obs.get("player"), dict) else {}
            return float(combat.get("energy", player.get("energy", 0.0)) or 0.0)
        return 0.0

    def _action_cost_value(self, action):
        if not isinstance(action, dict):
            return 0.0
        card = action.get("card") if isinstance(action.get("card"), dict) else {}
        return float(action.get("cost", card.get("cost", card.get("cost_for_turn", 0.0))) or 0.0)

    def _x_cost_diagnostic(self, _action, _energy):
        return {"x_cost_bad": 0.0}

    def _classify_positive_combat_action(self, action, *_args, **_kwargs):
        if isinstance(action, dict) and action.get("positive") is False:
            return {"positive": False}
        if isinstance(action, dict) and action.get("pure_block"):
            return {"positive": True, "card_pure_block": True}
        return {"positive": True}

    def _is_action_confirmed_lethal(self, action, _raw_obs):
        return bool(isinstance(action, dict) and action.get("lethal"))

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
        if roles is None and isinstance(action.get("semantic"), dict):
            roles = action["semantic"].get("roles")
        if isinstance(roles, (list, tuple, set)):
            return set(str(role) for role in roles)
        kind = self._semantic_family(action)
        return {"attack"} if kind == "play_card" and float(action.get("damage", 0.0) or 0.0) > 0.0 else set()

    def _dump_combat_hard_guard_record(self, **kwargs):
        self.records.append(kwargs)

    def _incoming_damage_pressure(self, raw_obs):
        combat = raw_obs.get("combat") if isinstance(raw_obs, dict) and isinstance(raw_obs.get("combat"), dict) else {}
        player = raw_obs.get("player") if isinstance(raw_obs, dict) and isinstance(raw_obs.get("player"), dict) else {}
        return (
            float(combat.get("incoming_damage", 0.0) or 0.0),
            float(player.get("block", combat.get("block", 0.0)) or 0.0),
            float(player.get("hp", player.get("current_hp", 70.0)) or 70.0),
        )


def raw_obs(*, energy: float = 3.0, max_energy: float = 3.0, phase: str = "combat") -> dict:
    return {
        "phase": phase,
        "combat": {
            "energy": energy,
            "max_energy": max_energy,
            "hand": [
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
            "incoming_damage": 0,
        },
        "player": {"hp": 70, "max_hp": 80, "block": 0, "max_energy": max_energy},
    }


def strike_action(action_id: str = "strike") -> dict:
    return {
        "action_id": action_id,
        "kind": "play_card",
        "card": {"title": "Strike", "cost": 1, "damage": 6},
        "title": "Strike",
        "cost": 1,
        "damage": 6,
        "roles": ["attack"],
    }


def end_turn_action() -> dict:
    return {"action_id": "end_turn", "kind": "end_turn", "title": "End Turn"}


def test_full_energy_endturn_overrides_to_legal_safe_progress_card() -> None:
    stub = GuardStub()
    legal_actions = [strike_action(), end_turn_action()]
    stats = {
        "combat_quality_end_turn_selected": 1.0,
        "combat_quality_full_energy_nonurgent_end_turn_selected": 1.0,
        "combat_quality_safe_progress_skip_selected": 1.0,
    }

    new_idx = stub._apply_full_energy_endturn_guard(
        action_idx=1,
        legal_count=2,
        legal_actions=legal_actions,
        mask_np=np.array([1, 1], dtype=np.float32),
        raw_obs=raw_obs(),
        encounter="ENCOUNTER.LIVING_FOG",
        search_stats=stats,
    )

    assert new_idx == 0
    assert stats["combat_quality_full_energy_endturn_guard_full_energy_selected"] == 1.0
    assert stats["combat_quality_full_energy_endturn_guard_available"] == 1.0
    assert stats["combat_quality_full_energy_endturn_guard_applied"] == 1.0
    assert stats["combat_quality_full_energy_endturn_guard_override"] == 1.0
    assert stats["combat_quality_hard_guard_override_any"] == 1.0
    assert stats["combat_quality_full_energy_endturn_guard_candidate_count"] == 1.0
    assert stats["combat_quality_end_turn_selected"] == 0.0
    assert stats["combat_quality_full_energy_nonurgent_end_turn_selected"] == 0.0
    assert stats["combat_quality_safe_progress_skip_selected"] == 0.0
    assert stub.records and stub.records[0]["override_idx"] == 0


def test_raw_hand_playable_but_no_legal_play_card_marks_gap_without_fabricating_action() -> None:
    stub = GuardStub()
    stats: dict[str, float] = {}

    new_idx = stub._apply_full_energy_endturn_guard(
        action_idx=0,
        legal_count=2,
        legal_actions=[end_turn_action(), {"action_id": "inspect", "kind": "inspect"}],
        mask_np=np.array([1, 1], dtype=np.float32),
        raw_obs=raw_obs(),
        encounter="ENCOUNTER.LIVING_FOG",
        search_stats=stats,
    )

    assert new_idx == 0
    assert stats["combat_quality_full_energy_endturn_guard_legal_generation_gap_suspect"] == 1.0
    assert stats["combat_quality_full_energy_endturn_guard_legal_surface_mismatch_skip"] == 1.0
    assert stats["combat_quality_full_energy_endturn_guard_ui_affordable_count"] == 1.0
    assert stats["combat_quality_full_energy_endturn_guard_legal_play_card_count"] == 0.0
    assert stats.get("combat_quality_full_energy_endturn_guard_applied", 0.0) == 0.0
    assert not stub.records


def test_low_energy_endturn_is_not_rewritten_by_full_energy_guard() -> None:
    stub = GuardStub()
    stats: dict[str, float] = {}

    new_idx = stub._apply_full_energy_endturn_guard(
        action_idx=1,
        legal_count=2,
        legal_actions=[strike_action(), end_turn_action()],
        mask_np=np.array([1, 1], dtype=np.float32),
        raw_obs=raw_obs(energy=1.0, max_energy=3.0),
        encounter="ENCOUNTER.LIVING_FOG",
        search_stats=stats,
    )

    assert new_idx == 1
    assert stats["combat_quality_full_energy_endturn_guard_full_energy_selected"] == 0.0
    assert stats.get("combat_quality_full_energy_endturn_guard_applied", 0.0) == 0.0


def test_selection_surface_endturn_is_not_rewritten() -> None:
    stub = GuardStub()
    stats: dict[str, float] = {}

    new_idx = stub._apply_full_energy_endturn_guard(
        action_idx=1,
        legal_count=2,
        legal_actions=[strike_action(), end_turn_action()],
        mask_np=np.array([1, 1], dtype=np.float32),
        raw_obs=raw_obs(phase="card_selection"),
        encounter="ENCOUNTER.LIVING_FOG",
        search_stats=stats,
    )

    assert new_idx == 1
    assert stats["combat_quality_full_energy_endturn_guard_selection_screen_skip"] == 1.0
    assert stats.get("combat_quality_full_energy_endturn_guard_applied", 0.0) == 0.0


def test_singleton_endturn_frontier_is_not_rewritten() -> None:
    stub = GuardStub()
    stats: dict[str, float] = {}

    new_idx = stub._apply_full_energy_endturn_guard(
        action_idx=0,
        legal_count=1,
        legal_actions=[end_turn_action()],
        mask_np=np.array([1], dtype=np.float32),
        raw_obs=raw_obs(),
        encounter="ENCOUNTER.LIVING_FOG",
        search_stats=stats,
    )

    assert new_idx == 0
    assert stats["combat_quality_full_energy_endturn_guard_forced_skip"] == 1.0
    assert stats.get("combat_quality_full_energy_endturn_guard_applied", 0.0) == 0.0


def test_candidate_returned_by_collector_is_rechecked_against_mask(monkeypatch) -> None:
    import muzero.combat_quality.full_energy_endturn_guard as guard_module

    stub = GuardStub()
    stats: dict[str, float] = {}
    legal_actions = [strike_action("illegal_strike"), strike_action("legal_strike"), end_turn_action()]

    def fake_collect(*_args, **_kwargs):
        return [
            ProgressCandidate(
                score=(1.0, 99.0, 99.0, -1.0, 0.0),
                index=0,
                damage=99.0,
                impact=99.0,
                cost=1.0,
                lethal=True,
                title="Illegal Strike",
            )
        ], {"accepted": 1}

    monkeypatch.setattr(guard_module, "collect_safe_progress_candidates", fake_collect)

    new_idx = stub._apply_full_energy_endturn_guard(
        action_idx=2,
        legal_count=3,
        legal_actions=legal_actions,
        mask_np=np.array([0, 1, 1], dtype=np.float32),
        raw_obs=raw_obs(),
        encounter="ENCOUNTER.LIVING_FOG",
        search_stats=stats,
    )

    assert new_idx == 2
    assert stats["combat_quality_full_energy_endturn_guard_illegal_candidate_reject"] == 1.0
    assert stats.get("combat_quality_full_energy_endturn_guard_applied", 0.0) == 0.0
