from __future__ import annotations

import json
import tempfile
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from muzero.diagnostics.episode_metrics import EpisodeMetricsMixin
from muzero.diagnostics.trainer_dumps import DiagnosticDumpMixin


class _Writer:
    def __init__(self) -> None:
        self.scalars: list[tuple[str, float, int]] = []

    def add_scalar(self, tag: str, value: float, step: int) -> None:
        self.scalars.append((tag, float(value), int(step)))


class _Harness(EpisodeMetricsMixin):
    def __init__(self) -> None:
        self.writer = _Writer()
        self.episode_count = 7


class _DeathSliceHarness(_Harness):
    def __init__(self) -> None:
        super().__init__()
        self.death_slice_calls: list[dict[str, object]] = []

    def _dump_death_slice(self, **kwargs: object) -> None:
        self.death_slice_calls.append(dict(kwargs))


class _DeathSliceWriterHarness(EpisodeMetricsMixin, DiagnosticDumpMixin):
    _DEATH_SLICE_TARGETS = ("ENCOUNTER.KAISER_CRAB_BOSS",)
    _DEATH_SLICE_PER_ENCOUNTER_CAP = 200

    def __init__(self, log_dir: str) -> None:
        self.writer = _Writer()
        self.episode_count = 11
        self.total_steps = 123
        self.log_dir = log_dir


def test_emit_combat_quality_episode_diagnostics_splits_global_normal_elite() -> None:
    harness = _Harness()
    trajectory = SimpleNamespace(
        steps=[
            {
                "decision_domain": "combat",
                "action_family": "play_card",
                "encounter_tier": "normal",
                "search_stats": {
                    "combat_quality_card_block_waste_count": 1,
                    "combat_quality_card_pure_block_count": 1,
                    "combat_quality_card_no_damage_pressure_count": 0,
                    "combat_quality_card_block_waste_selected": 1,
                    "combat_quality_card_pure_block_selected": 1,
                    "combat_quality_card_no_damage_pressure_selected": 0,
                },
            },
            {
                "decision_domain": "combat",
                "action_family": "end_turn",
                "encounter_tier": "elite",
                "wasteful_end_turn": True,
                "search_stats": {
                    "combat_quality_card_block_waste_count": 0,
                    "combat_quality_card_pure_block_count": 0,
                    "combat_quality_card_no_damage_pressure_count": 1,
                    "combat_quality_card_block_waste_selected": 0,
                    "combat_quality_card_pure_block_selected": 0,
                    "combat_quality_card_no_damage_pressure_selected": 1,
                },
            },
            {
                "decision_domain": "route",
                "action_family": "path",
                "encounter_tier": "normal",
                "search_stats": {
                    "combat_quality_card_block_waste_selected": 1,
                },
            },
        ],
    )

    metrics = harness._emit_combat_quality_episode_diagnostics(trajectory)

    assert metrics["combat_quality/decision_count"] == 2.0
    assert metrics["combat_quality/family_play_card_rate"] == 0.5
    assert metrics["combat_quality/family_end_turn_rate"] == 0.5
    assert metrics["combat_quality/wasteful_end_turn_rate"] == 0.5
    assert metrics["combat_quality/card_block_waste_selected_rate"] == 0.5
    assert metrics["combat_quality/card_no_damage_pressure_selected_rate"] == 0.5

    assert metrics["normal_combat/decision_count"] == 1.0
    assert metrics["normal_combat/card_block_waste_selected_rate"] == 1.0
    assert metrics["normal_combat/card_no_damage_pressure_selected_rate"] == 0.0

    assert metrics["elite_combat/decision_count"] == 1.0
    assert metrics["elite_combat/card_block_waste_selected_rate"] == 0.0
    assert metrics["elite_combat/card_no_damage_pressure_selected_rate"] == 1.0
    assert metrics["elite_combat/wasteful_end_turn_rate"] == 1.0

    written = {tag: value for tag, value, step in harness.writer.scalars if step == 7}
    assert written["combat_quality/card_block_waste_selected_rate"] == pytest.approx(0.5)
    assert "boss_combat/card_block_waste_selected_rate" not in written


def test_emit_combat_quality_episode_diagnostics_dumps_non_boss_death_slice() -> None:
    harness = _DeathSliceHarness()
    trajectory = SimpleNamespace(
        metadata={
            "episode_mode": "combat_sandbox",
            "encounter_id": "ENCOUNTER.OVICOPTER_NORMAL",
            "encounter_tier": "normal",
            "death_floor": 1.0,
            "episode_total_reward": -4.0,
        },
        steps=[
            {
                "decision_domain": "combat",
                "action_family": "play_card",
                "encounter_id": "ENCOUNTER.OVICOPTER_NORMAL",
                "encounter_tier": "normal",
                "search_stats": {
                    "combat_quality_card_block_waste_selected": 0.0,
                },
            }
        ],
    )

    harness._emit_combat_quality_episode_diagnostics(trajectory)

    assert len(harness.death_slice_calls) == 1
    call = harness.death_slice_calls[0]
    assert call["encounter_id"] == "ENCOUNTER.OVICOPTER_NORMAL"
    assert call["encounter_tier"] == "normal"
    assert call["watch_only"] is False
    assert call["tail_len"] == 8
    assert call["reason"] == "combat_quality_episode_loss"


def test_emit_combat_quality_episode_diagnostics_skips_death_slice_on_win() -> None:
    harness = _DeathSliceHarness()
    trajectory = SimpleNamespace(
        metadata={
            "episode_mode": "combat_sandbox",
            "encounter_id": "ENCOUNTER.OVICOPTER_NORMAL",
            "encounter_tier": "normal",
            "death_floor": 0.0,
            "episode_total_reward": 2.0,
        },
        steps=[
            {
                "decision_domain": "combat",
                "action_family": "play_card",
                "encounter_id": "ENCOUNTER.OVICOPTER_NORMAL",
                "encounter_tier": "normal",
                "search_stats": {},
            }
        ],
    )

    harness._emit_combat_quality_episode_diagnostics(trajectory)

    assert harness.death_slice_calls == []


def test_emit_boss_episode_diagnostics_writes_lucky_skip_scalars() -> None:
    harness = _Harness()
    trajectory = SimpleNamespace(
        metadata={
            "encounter_id": "ENCOUNTER.INSATIABLE_BOSS",
            "encounter_tier": "boss",
            "death_floor": 17.0,
            "episode_total_reward": -18.0,
            "lucky_seen_this_combat": True,
            "lucky_legal_this_combat": True,
            "lucky_selected_this_combat": False,
        },
        steps=[
            {
                "decision_domain": "combat",
                "action_family": "end_turn",
                "encounter_id": "ENCOUNTER.INSATIABLE_BOSS",
                "encounter_tier": "boss",
                "search_stats": {},
            }
        ],
    )

    metrics = harness._emit_boss_episode_diagnostics(
        trajectory,
        boss_entry={"hp": 12.0, "hp_ratio": 0.2},
        final_potion_count=1,
    )

    assert metrics["boss/lucky_seen_rate"] == 1.0
    assert metrics["boss/lucky_legal_rate"] == 1.0
    assert metrics["boss/lucky_skip_on_boss_death_rate"] == 1.0
    assert metrics["boss/lucky_used_on_boss_win_rate"] == 0.0
    written = {tag: value for tag, value, step in harness.writer.scalars if step == 7}
    assert written["boss/lucky_skip_on_boss_death_rate"] == 1.0


def test_emit_boss_episode_diagnostics_flags_final_lucky_even_when_flags_missing() -> None:
    harness = _Harness()
    trajectory = SimpleNamespace(
        metadata={
            "encounter_id": "ENCOUNTER.SOUL_FYSH_BOSS",
            "encounter_tier": "boss",
            "death_floor": 17.0,
            "episode_total_reward": -20.0,
            "final_potion_dump": [
                {"slot": 0, "id": "POTION.LUCKY_TONIC", "title": "幸运药剂"},
            ],
            "selected_potion_actions_this_combat": [],
            "potion_use_transitions_this_combat": [],
            "lucky_seen_this_combat": False,
            "lucky_legal_this_combat": False,
            "lucky_selected_this_combat": False,
        },
        steps=[
            {
                "decision_domain": "combat",
                "action_family": "end_turn",
                "encounter_id": "ENCOUNTER.SOUL_FYSH_BOSS",
                "encounter_tier": "boss",
                "search_stats": {},
            }
        ],
    )

    metrics = harness._emit_boss_episode_diagnostics(
        trajectory,
        boss_entry={"hp": 3.0, "hp_ratio": 0.05},
        final_potion_count=1,
    )

    assert metrics["boss/lucky_seen_rate"] == 0.0
    assert metrics["boss/lucky_legal_rate"] == 0.0
    assert metrics["boss/lucky_seen_anywhere_on_boss_death_rate"] == 1.0
    assert metrics["boss/final_lucky_potion_count_mean"] == 1.0
    assert metrics["boss/final_lucky_unused_on_boss_death_rate"] == 1.0
    assert metrics["boss/lucky_skip_on_boss_death_rate"] == 1.0
    written = {tag: value for tag, value, step in harness.writer.scalars if step == 7}
    assert written["boss/final_lucky_unused_on_boss_death_rate"] == 1.0


def test_emit_boss_episode_diagnostics_does_not_flag_final_lucky_when_used() -> None:
    harness = _Harness()
    trajectory = SimpleNamespace(
        metadata={
            "encounter_id": "ENCOUNTER.SOUL_FYSH_BOSS",
            "encounter_tier": "boss",
            "death_floor": 17.0,
            "episode_total_reward": -20.0,
            "final_potion_dump": [
                {"slot": 0, "id": "POTION.LUCKY_TONIC", "title": "幸运药剂"},
            ],
            "selected_potion_actions_this_combat": [
                {"action_id": "use_potion:0:self", "potion_id": "POTION.LUCKY_TONIC"},
            ],
            "potion_use_transitions_this_combat": [],
            "lucky_seen_this_combat": True,
            "lucky_legal_this_combat": True,
            "lucky_selected_this_combat": True,
        },
        steps=[
            {
                "decision_domain": "combat",
                "action_family": "use_potion",
                "encounter_id": "ENCOUNTER.SOUL_FYSH_BOSS",
                "encounter_tier": "boss",
                "search_stats": {},
            }
        ],
    )

    metrics = harness._emit_boss_episode_diagnostics(
        trajectory,
        boss_entry={"hp": 3.0, "hp_ratio": 0.05},
        final_potion_count=1,
    )

    assert metrics["boss/lucky_seen_anywhere_on_boss_death_rate"] == 1.0
    assert metrics["boss/final_lucky_potion_count_mean"] == 1.0
    assert metrics["boss/final_lucky_unused_on_boss_death_rate"] == 0.0
    assert metrics["boss/lucky_skip_on_boss_death_rate"] == 0.0


def test_death_slice_writer_accepts_non_watchlisted_normal_when_requested() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        harness = _DeathSliceWriterHarness(tmp)
        trajectory = SimpleNamespace(
            metadata={
                "episode_mode": "combat_sandbox",
                "death_floor": 1.0,
                "episode_total_reward": -3.0,
                "snapshot_sample_id": "sample-1",
                "potion_history_schema": "episode_potion_history_v1",
                "potion_history_steps_this_combat": 1,
                "last_seen_potions_this_combat": [
                    {"slot": 0, "id": "POTION.LUCKY_TONIC", "title": "幸运补剂"},
                ],
                "last_seen_legal_potion_actions_this_combat": [
                    {"index": 0, "potion_id": "POTION.LUCKY_TONIC", "title": "幸运补剂"},
                ],
                "selected_potion_actions_this_combat": [],
                "lucky_seen_this_combat": True,
                "lucky_legal_this_combat": True,
                "lucky_selected_this_combat": False,
                "potion_use_transitions_this_combat": [
                    {"action_id": "use_potion:0:self", "potion_id_before": "POTION.LUCKY_TONIC"},
                ],
                "potion_transition_sync_suspect_this_combat": True,
            }
        )
        steps = [
            {
                "decision_domain": "combat",
                "action_family": "end_turn",
                "encounter_id": "ENCOUNTER.OVICOPTER_NORMAL",
                "encounter_tier": "normal",
                "action": 2,
                "reward": -1.5,
                "root_value": -0.25,
                "action_mask": np.asarray([1, 0, 1], dtype=np.float32),
                "search_policy": np.asarray([0.1, 0.0, 0.9], dtype=np.float32),
                "action_info": {"title": "End Turn", "kind": "combat"},
                "search_stats": {
                    "root_top1_visit_share": 0.9,
                    "combat_quality_wasteful_end_turn_selected": 1.0,
                    "unrelated_large_debug": 999.0,
                },
                "decision_diagnostics": {
                    "schema": "combat_pre_step_v1",
                    "raw_potion_count": 1,
                    "legal_potion_action_count": 1,
                    "raw_potions": [{"slot": 0, "id": "POTION.LUCKY_TONIC", "title": "幸运补剂"}],
                    "legal_potion_actions": [
                        {
                            "index": 0,
                            "action_id": "use_potion:0:self",
                            "potion_id": "POTION.LUCKY_TONIC",
                            "title": "幸运补剂",
                        }
                    ],
                },
                "obs": {
                    "vector": np.asarray([1.0, 2.0, 3.0], dtype=np.float32),
                    "phase": "combat",
                },
            }
        ]

        harness._dump_death_slice(
            trajectory=trajectory,
            encounter_id="ENCOUNTER.OVICOPTER_NORMAL",
            encounter_tier="normal",
            loss=True,
            steps=steps,
            watch_only=False,
            tail_len=8,
            reason="unit_test",
        )

        path = Path(tmp) / "diagnostics" / "death_slices" / "encounter_ovicopter_normal.jsonl"
        row = json.loads(path.read_text(encoding="utf-8").strip())
        assert row["schema_version"] == 3
        assert row["reason"] == "unit_test"
        assert row["metadata_subset"]["snapshot_sample_id"] == "sample-1"
        assert row["tail_steps"][0]["policy_summary"]["legal_count"] == 2
        assert row["tail_steps"][0]["policy_summary"]["selected_policy_prob"] == pytest.approx(0.9)
        assert row["tail_steps"][0]["search_stats"]["combat_quality_wasteful_end_turn_selected"] == 1.0
        assert "unrelated_large_debug" not in row["tail_steps"][0]["search_stats"]
        assert row["tail_steps"][0]["decision_diagnostics"]["raw_potions"][0]["id"] == "POTION.LUCKY_TONIC"
        assert row["tail_steps"][0]["decision_diagnostics"]["legal_potion_action_count"] == 1
        assert row["tail_steps"][0]["obs_summary"]["vector"]["shape"] == [3]
        assert row["metadata_subset"]["potion_history_schema"] == "episode_potion_history_v1"
        assert row["metadata_subset"]["lucky_seen_this_combat"] is True
        assert row["metadata_subset"]["lucky_legal_this_combat"] is True
        assert row["metadata_subset"]["lucky_selected_this_combat"] is False
        assert row["metadata_subset"]["potion_transition_sync_suspect_this_combat"] is True
        assert row["metadata_subset"]["potion_use_transitions_this_combat"][0]["action_id"] == "use_potion:0:self"


def test_death_slice_writer_watch_only_bypasses_for_lucky_unused_boss_death() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        harness = _DeathSliceWriterHarness(tmp)
        trajectory = SimpleNamespace(
            metadata={
                "episode_mode": "full_run",
                "death_floor": 17.0,
                "episode_total_reward": -18.0,
                "last_seen_potions_this_combat": [
                    {"slot": 0, "id": "POTION.LUCKY_TONIC", "title": "幸运补剂"},
                ],
                "last_seen_legal_potion_actions_this_combat": [
                    {"index": 0, "potion_id": "POTION.LUCKY_TONIC", "title": "幸运补剂"},
                ],
                "selected_potion_actions_this_combat": [],
                "lucky_seen_this_combat": True,
                "lucky_legal_this_combat": True,
                "lucky_selected_this_combat": False,
            }
        )
        steps = [
            {
                "decision_domain": "combat",
                "action_family": "end_turn",
                "encounter_id": "ENCOUNTER.INSATIABLE_BOSS",
                "encounter_tier": "boss",
                "action": 0,
                "action_mask": np.asarray([1], dtype=np.float32),
                "search_policy": np.asarray([1.0], dtype=np.float32),
                "action_info": {"title": "End Turn"},
                "search_stats": {"combat_quality_potion_prevent_lethal_count": 1.0},
            }
        ]

        harness._dump_death_slice(
            trajectory=trajectory,
            encounter_id="ENCOUNTER.INSATIABLE_BOSS",
            encounter_tier="boss",
            loss=True,
            steps=steps,
            watch_only=True,
            tail_len=8,
            reason="lucky_unused_unit_test",
        )

        path = Path(tmp) / "diagnostics" / "death_slices" / "encounter_insatiable_boss.jsonl"
        assert path.exists()
        row = json.loads(path.read_text(encoding="utf-8").strip())
        assert row["reason"] == "lucky_unused_unit_test"
        assert row["metadata_subset"]["lucky_seen_this_combat"] is True
        assert row["metadata_subset"]["lucky_legal_this_combat"] is True
        assert row["metadata_subset"]["lucky_selected_this_combat"] is False
        assert row["tail_steps"][0]["decision_diagnostics"] == {}


def test_death_slice_writer_watch_only_bypasses_for_final_lucky_even_when_tier_normal_and_flags_missing() -> None:
    """A final-inventory Lucky on death must be dumped even if tier/flags are wrong.

    This prevents the exact post-mortem blind spot where the player sees a
    Lucky/幸运药剂 left in the potion belt, but compact metadata did not set the
    lucky_seen/legal flags and the encounter tier was misclassified as normal.
    """

    with tempfile.TemporaryDirectory() as tmp:
        harness = _DeathSliceWriterHarness(tmp)
        trajectory = SimpleNamespace(
            metadata={
                "episode_mode": "full_run",
                "death_floor": 17.0,
                "episode_total_reward": -18.0,
                "final_potion_dump": [
                    {"slot": 0, "id": "POTION.LUCKY_TONIC", "title": "幸运药剂"},
                ],
                "lucky_seen_this_combat": False,
                "lucky_legal_this_combat": False,
                "lucky_selected_this_combat": False,
            }
        )
        steps = [
            {
                "decision_domain": "combat",
                "action_family": "end_turn",
                "encounter_id": "ENCOUNTER.UNKNOWN_NORMAL",
                "encounter_tier": "normal",
                "action": 0,
                "action_mask": np.asarray([1], dtype=np.float32),
                "search_policy": np.asarray([1.0], dtype=np.float32),
                "action_info": {"title": "End Turn"},
                "search_stats": {},
            }
        ]

        harness._dump_death_slice(
            trajectory=trajectory,
            encounter_id="ENCOUNTER.UNKNOWN_NORMAL",
            encounter_tier="normal",
            loss=True,
            steps=steps,
            watch_only=True,
            tail_len=8,
            reason="final_lucky_unused_unit_test",
        )

        path = Path(tmp) / "diagnostics" / "death_slices" / "encounter_unknown_normal.jsonl"
        assert path.exists()
        row = json.loads(path.read_text(encoding="utf-8").strip())
        subset = row["metadata_subset"]
        assert row["reason"] == "final_lucky_unused_unit_test"
        assert subset["lucky_seen_this_combat"] is False
        assert subset["lucky_legal_this_combat"] is False
        assert subset["lucky_selected_this_combat"] is False
        assert subset["lucky_seen_anywhere_on_death"] is True
        assert subset["lucky_selected_or_used_this_combat"] is False
        assert subset["final_lucky_potion_count"] == 1
        assert subset["final_lucky_unused_on_death"] is True
        assert subset["lucky_unused_survival_potion_death"] is True


def test_death_slice_writer_does_not_mark_lucky_unused_when_selected_lucky() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        harness = _DeathSliceWriterHarness(tmp)
        trajectory = SimpleNamespace(
            metadata={
                "episode_mode": "full_run",
                "death_floor": 17.0,
                "episode_total_reward": -18.0,
                "final_potion_dump": [
                    {"slot": 0, "id": "POTION.LUCKY_TONIC", "title": "幸运补剂"},
                ],
                "selected_potion_actions_this_combat": [
                    {"action_id": "use_potion:0:self", "potion_id": "POTION.LUCKY_TONIC"},
                ],
                "lucky_seen_this_combat": True,
                "lucky_legal_this_combat": True,
                "lucky_selected_this_combat": True,
            }
        )
        steps = [
            {
                "decision_domain": "combat",
                "action_family": "use_potion",
                "encounter_id": "ENCOUNTER.INSATIABLE_BOSS",
                "encounter_tier": "boss",
                "action": 0,
                "action_mask": np.asarray([1], dtype=np.float32),
                "search_policy": np.asarray([1.0], dtype=np.float32),
                "action_info": {"title": "幸运补剂"},
                "search_stats": {},
            }
        ]

        harness._dump_death_slice(
            trajectory=trajectory,
            encounter_id="ENCOUNTER.INSATIABLE_BOSS",
            encounter_tier="boss",
            loss=True,
            steps=steps,
            watch_only=False,
            tail_len=8,
            reason="lucky_used_unit_test",
        )

        path = Path(tmp) / "diagnostics" / "death_slices" / "encounter_insatiable_boss.jsonl"
        row = json.loads(path.read_text(encoding="utf-8").strip())
        subset = row["metadata_subset"]
        assert subset["lucky_seen_anywhere_on_death"] is True
        assert subset["lucky_selected_or_used_this_combat"] is True
        assert subset["final_lucky_potion_count"] == 1
        assert subset["final_lucky_unused_on_death"] is False
        assert subset["lucky_unused_survival_potion_death"] is False
