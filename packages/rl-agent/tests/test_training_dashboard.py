from __future__ import annotations

import http.client
import json
import os
import threading
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path
from urllib.parse import urlencode

import pytest

import sts2_rl.monitoring as monitoring_module
from sts2_rl.monitor_dashboard import DashboardHTTPServer, _dashboard_html
from sts2_rl.monitoring import DashboardStore, IncrementalMetrics

RUN_A = "11111111-1111-4111-8111-111111111111"
RUN_B = "22222222-2222-4222-8222-222222222222"
RUN_C = "33333333-3333-4333-8333-333333333333"


def _append_jsonl(path: Path, *events: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("ab") as handle:
        for event in events:
            handle.write(json.dumps(event, ensure_ascii=False, separators=(",", ":")).encode("utf-8"))
            handle.write(b"\n")


def _run_start(
    run_id: str,
    unix_s: float,
    *,
    state: dict[str, object] | None = None,
    checkpoint_load: dict[str, object] | None = None,
    checkpoint_dir: str = "checkpoints/lineage",
) -> dict[str, object]:
    return {
        "event": "run_start",
        "run_id": run_id,
        "unix_s": unix_s,
        "state": state
        or {
            "environment_steps": 0,
            "episodes": 0,
            "learner_updates": 0,
            "policy_version": 0,
            "evaluation_episodes": 0,
        },
        "checkpoint_load": checkpoint_load or {"mode": "fresh"},
        "config": {
            "profile": "test",
            "runtime": {
                "total_environment_steps": 100,
                "checkpoint_dir": checkpoint_dir,
                "device": "cuda",
                "collector_device": "cpu",
            },
            "curriculum": {
                "mode": "native-revival-preheat",
                "revival_budget": -1,
            },
            "model": {
                "architecture": "test-policy",
                "max_candidates": 256,
            },
        },
    }


def _create_run(
    root: Path,
    run_id: str,
    *,
    lineage: str = "lineage",
    unix_s: float = 100.0,
    state: dict[str, object] | None = None,
    checkpoint_load: dict[str, object] | None = None,
    checkpoint_dir: str | None = None,
) -> tuple[Path, Path]:
    run_directory = root / "runs" / lineage / f"run-{run_id}"
    metrics = run_directory / "metrics.jsonl"
    _append_jsonl(
        metrics,
        _run_start(
            run_id,
            unix_s,
            state=state,
            checkpoint_load=checkpoint_load,
            checkpoint_dir=checkpoint_dir or f"checkpoints/{lineage}",
        ),
    )
    return run_directory, metrics


def test_incremental_jsonl_complete_partial_append_bad_line_terminal_and_truncate(tmp_path: Path) -> None:
    metrics = tmp_path / "metrics.jsonl"
    _append_jsonl(metrics, _run_start(RUN_A, 1.0))
    parser = IncrementalMetrics(metrics)

    parser.refresh()
    assert parser.parsed_rows == 1
    assert parser.event_counts == {"run_start": 1}

    encoded = json.dumps(
        {"event": "train_episode", "unix_s": 2.0, "episode_id": "中文局"},
        ensure_ascii=False,
        separators=(",", ":"),
    ).encode("utf-8")
    split = encoded.index("中".encode()) + 1
    with metrics.open("ab") as handle:
        handle.write(encoded[:split])
    parser.refresh()
    assert parser.partial_line is True
    assert parser.event_counts.get("train_episode", 0) == 0

    with metrics.open("ab") as handle:
        handle.write(encoded[split:] + b"\n")
        handle.write(b"{this is not json}\n")
        handle.write(b'{"event":"run_complete","unix_s":3,"environment_steps":10}\n')
    parser.refresh()
    assert parser.event_counts["train_episode"] == 1
    assert parser.event_counts["run_complete"] == 1
    assert parser.latest_by_event["run_complete"]["environment_steps"] == 10
    assert parser.malformed_rows == 1
    assert parser.partial_line is False

    parsed_once = parser.parsed_rows
    parser.refresh()
    assert parser.parsed_rows == parsed_once
    assert parser.event_counts["run_complete"] == 1

    metrics.write_bytes(b'{"event":"run_start","unix_s":4}\n')
    parser.refresh()
    assert parser.parsed_rows == 1
    assert parser.event_counts == {"run_start": 1}
    assert "run_complete" not in parser.latest_by_event
    assert parser.malformed_rows == 0


def test_discovery_uses_run_start_time_and_only_canonical_paths(tmp_path: Path) -> None:
    root = tmp_path / "artifacts"
    _, older_metrics = _create_run(root, RUN_A, unix_s=100.0)
    _, newer_metrics = _create_run(root, RUN_B, unix_s=200.0)
    now = 1_000_000_000
    os.utime(older_metrics, (now, now))
    os.utime(newer_metrics, (now - 1000, now - 1000))

    bad_name = root / "runs" / "lineage" / "run-not-a-uuid" / "metrics.jsonl"
    _append_jsonl(bad_name, _run_start(RUN_C, 300.0))
    mismatched = root / "runs" / "lineage" / f"run-{RUN_C}" / "metrics.jsonl"
    _append_jsonl(mismatched, _run_start(RUN_A, 400.0))
    nested_legacy = root / "runs" / "legacy" / "nested" / f"run-{RUN_C}" / "metrics.jsonl"
    _append_jsonl(nested_legacy, _run_start(RUN_C, 500.0))

    store = DashboardStore(root)
    runs = store.discover(force=True)

    assert [run.run_id for run in runs] == [RUN_B, RUN_A]
    assert all(run.run_directory.parent.parent == root / "runs" for run in runs)
    assert store.list_runs()["selected"] == f"lineage/run-{RUN_B}"


def test_exact_resume_chain_aggregates_series_counters_gate_zero_and_context(tmp_path: Path) -> None:
    root = tmp_path / "artifacts"
    _, parent_metrics = _create_run(root, RUN_A, unix_s=100.0)
    _append_jsonl(
        parent_metrics,
        {
            "event": "learner_update",
            "unix_s": 109.0,
            "environment_steps": 10,
            "policy_version": 2,
            "loss": 0.8,
        },
        {
            "event": "train_episode",
            "unix_s": 110.0,
            "environment_steps": 10,
            "max_floor": 8,
            "revivals_used": 2,
        },
        {
            "event": "evaluation",
            "unix_s": 115.0,
            "evaluation_gate": 0.0,
            "environment_steps": 10,
            "episodes": 2,
            "act1_clear_rate": 0.0,
            "act3_reach_rate": 0.0,
            "run_win_rate": 0.0,
            "mean_max_floor": 4.5,
            "maximum_floor": 8,
            "mean_revivals_used": 3.0,
            "mean_player_hp_lost": 90.0,
            "deadlock_rate": 0.5,
            "maximum_definition_hash_collisions_per_decision": 4,
            "maximum_relation_hash_collisions_per_decision": 5,
            "definition_hash_collisions_total": 8,
            "relation_hash_collisions_total": 9,
        },
        {
            "event": "checkpoint",
            "unix_s": 116.0,
            "environment_steps": 10,
            "episodes": 1,
            "evaluation_episodes": 2,
            "policy_version": 2,
        },
        {
            # This completed after the inherited checkpoint and is not part
            # of the child's actual model lineage.
            "event": "learner_update",
            "unix_s": 117.0,
            "environment_steps": 10,
            "policy_version": 3,
            "loss": 99.0,
        },
    )
    _, child_metrics = _create_run(
        root,
        RUN_B,
        unix_s=200.0,
        state={
            "environment_steps": 10,
            "episodes": 1,
            "learner_updates": 2,
            "policy_version": 2,
            "evaluation_episodes": 2,
            "maximum_observed_candidates": 31,
        },
        checkpoint_load={
            "mode": "exact_resume",
            "parent_checkpoint": (f"/runtime/checkpoints/lineage/run-{RUN_A}/periodic-step-000000010"),
        },
    )
    _append_jsonl(
        child_metrics,
        {
            "event": "train_episode",
            "unix_s": 210.0,
            "environment_steps": 15,
            "episodes": 2,
            "max_floor": 12,
            "revivals_used": 1,
            "maximum_observed_candidates": 111,
            "maximum_definition_hash_collisions_per_decision": 2,
            "maximum_relation_hash_collisions_per_decision": 3,
            "definition_hash_collisions_total": 5,
            "relation_hash_collisions_total": 7,
        },
        {
            "event": "learner_update",
            "unix_s": 211.0,
            "environment_steps": 15,
            "learner_updates": 3,
            "policy_version": 3,
            "loss": 0.5,
        },
    )

    store = DashboardStore(root, now=lambda: 212.0)
    snapshot = store.snapshot(f"lineage/run-{RUN_B}")

    assert snapshot["run"]["chain_run_count"] == 2
    assert [item["run_id"] for item in snapshot["run"]["chain"]] == [RUN_A, RUN_B]
    assert snapshot["progress"]["environment_steps"] == 15
    assert snapshot["progress"]["training_episodes"] == 2
    assert snapshot["progress"]["learner_updates"] == 3
    assert snapshot["progress"]["policy_version"] == 3
    assert snapshot["progress"]["evaluation_episodes"] == 2
    assert snapshot["progress"]["maximum_observed_candidates"] == 111
    assert [episode["max_floor"] for episode in snapshot["episodes"]] == [8, 12]
    assert snapshot["episodes"][-1][
        "maximum_definition_hash_collisions_per_decision"
    ] == 2
    assert snapshot["episodes"][-1][
        "maximum_relation_hash_collisions_per_decision"
    ] == 3
    assert snapshot["episodes"][-1]["definition_hash_collisions_total"] == 5
    assert snapshot["episodes"][-1]["relation_hash_collisions_total"] == 7
    assert len(snapshot["evaluations"]) == 1
    assert snapshot["evaluations"][0]["evaluation_gate"] == 0
    assert snapshot["evaluations"][0]["act_1_success_rate"] == 0.0
    assert snapshot["evaluations"][0][
        "maximum_definition_hash_collisions_per_decision"
    ] == 4
    assert snapshot["evaluations"][0][
        "maximum_relation_hash_collisions_per_decision"
    ] == 5
    assert snapshot["evaluations"][0]["definition_hash_collisions_total"] == 8
    assert snapshot["evaluations"][0]["relation_hash_collisions_total"] == 9
    assert [point["policy_version"] for point in snapshot["learner_series"]] == [2, 3]
    assert [point["loss"] for point in snapshot["learner_series"]] == [0.8, 0.5]
    assert snapshot["context"] == {
        "mode": "native-revival-preheat",
        "revival_budget": -1,
        "standard_game": False,
        "label": "无限隐藏引擎续命预热 · 非标准胜率",
    }


def test_episode_projection_preserves_chronology_and_breaks_rollback_branches(
    tmp_path: Path,
) -> None:
    root = tmp_path / "artifacts"
    _, metrics = _create_run(root, RUN_A, unix_s=100.0)
    for index, environment_steps in enumerate((1_000, 2_000, 400, 900), start=1):
        _append_jsonl(
            metrics,
            {
                "event": "train_episode",
                "unix_s": 100.0 + index,
                "environment_steps": environment_steps,
                "max_floor": index,
                "revivals_used": index * 2,
            },
        )

    snapshot = DashboardStore(root).snapshot()

    assert [episode["environment_steps"] for episode in snapshot["episodes"]] == [
        1_000,
        2_000,
        400,
        900,
    ]
    assert [episode["rollback_generation"] for episode in snapshot["episodes"]] == [
        0,
        0,
        1,
        1,
    ]
    assert [episode["rollback_boundary"] for episode in snapshot["episodes"]] == [
        False,
        False,
        True,
        False,
    ]


def test_explicit_rollback_generation_wins_over_legacy_step_inference() -> None:
    episodes = monitoring_module._annotate_episode_generations(
        [
            {"environment_steps": 100, "rollback_generation": 3},
            {"environment_steps": 200, "rollback_generation": 3},
            {"environment_steps": 50, "rollback_generation": 4},
        ]
    )

    assert [episode["rollback_generation"] for episode in episodes] == [3, 3, 4]
    assert [episode["rollback_boundary"] for episode in episodes] == [False, False, True]


def test_status_completed_overrides_stale_fresh_is_running_and_old_unknown_is_stale(tmp_path: Path) -> None:
    now = 10_000.0

    completed_root = tmp_path / "complete"
    _, completed_metrics = _create_run(completed_root, RUN_A, unix_s=100.0)
    _append_jsonl(
        completed_metrics,
        {"event": "run_complete", "unix_s": 101.0, "environment_steps": 100},
    )
    os.utime(completed_metrics, (100.0, 100.0))
    completed = DashboardStore(completed_root, stale_seconds=100.0, now=lambda: now).snapshot()
    assert completed["status"]["state"] == "completed"
    assert completed["status"]["phase"] == "complete"

    running_root = tmp_path / "running"
    _, running_metrics = _create_run(running_root, RUN_B, unix_s=now - 10)
    os.utime(running_metrics, (now - 10, now - 10))
    running = DashboardStore(running_root, stale_seconds=100.0, now=lambda: now).snapshot()
    assert running["status"]["state"] == "running"

    stale_root = tmp_path / "stale"
    _, stale_metrics = _create_run(stale_root, RUN_C, unix_s=100.0)
    os.utime(stale_metrics, (100.0, 100.0))
    stale = DashboardStore(stale_root, stale_seconds=100.0, now=lambda: now).snapshot()
    assert stale["status"]["state"] == "stale"
    assert stale["status"]["phase"] == "stale_unknown"


@pytest.mark.parametrize(
    ("completion_status", "expected_status"),
    (
        (
            "horizon_complete",
            {
                "state": "completed",
                "phase": "complete",
                "label": "已完成",
            },
        ),
        (
            "evaluation_guard_stopped",
            {
                "state": "guard_stopped",
                "phase": "evaluation_guard_stopped",
                "label": "评估门禁提前停止",
            },
        ),
    ),
)
def test_run_completion_status_is_consistent_across_quick_and_full_paths(
    tmp_path: Path,
    completion_status: str,
    expected_status: dict[str, str],
) -> None:
    root = tmp_path / completion_status
    _, metrics = _create_run(root, RUN_A, unix_s=100.0)
    _append_jsonl(
        metrics,
        {
            "event": "run_complete",
            "unix_s": 101.0,
            "environment_steps": (25 if completion_status == "evaluation_guard_stopped" else 100),
            "completion_status": completion_status,
        },
    )

    store = DashboardStore(root, now=lambda: 102.0)
    cold_quick = store.list_runs()["runs"][0]["status"]
    full = store.snapshot()
    cached_quick = store.list_runs()["runs"][0]["status"]

    assert cold_quick == expected_status
    assert {key: full["status"][key] for key in ("state", "phase", "label")} == expected_status
    assert cached_quick == expected_status
    assert full["lifecycle"]["completion_status"] == completion_status


def test_active_final_audit_is_fresh_evaluating_not_complete_at_training_horizon(
    tmp_path: Path,
) -> None:
    now = 10_000.0
    root = tmp_path / "artifacts"
    run_directory, metrics = _create_run(root, RUN_A, unix_s=100.0)
    _append_jsonl(
        metrics,
        {
            "event": "learner_update",
            "unix_s": 200.0,
            "environment_steps": 100,
            "policy_version": 1,
        },
    )
    os.utime(metrics, (200.0, 200.0))
    final_audit = run_directory / "final-audit-step-000000100.jsonl"
    _append_jsonl(final_audit, {"event": "decision", "reset_seed": 200_000_001})
    os.utime(final_audit, (now - 5.0, now - 5.0))

    store = DashboardStore(root, stale_seconds=100.0, now=lambda: now)
    quick_status = store.list_runs()["runs"][0]["status"]
    snapshot = store.snapshot()

    assert quick_status == {
        "state": "evaluating",
        "phase": "evaluating",
        "label": "最终审计中 · 门 100",
    }
    assert snapshot["status"]["state"] == "evaluating"
    assert snapshot["status"]["label"] == "最终审计中 · 门 100"
    assert "run_complete 尚未持久化" in snapshot["status"]["evidence"][1]
    assert snapshot["progress"]["percent"] == 100.0
    assert snapshot["lifecycle"] == {
        "run_complete_persisted": False,
        "training_horizon_reached": True,
        "completion_status": None,
        "pending_evaluation": {
            "kind": "final_audit",
            "gate": 100,
            "attempt": 1,
            "journal_name": "final-audit-step-000000100.jsonl",
            "modified_at": now - 5.0,
        },
    }


def test_validation_and_early_validation_journals_are_recognized(tmp_path: Path) -> None:
    cases = (
        ("evaluation-step-000000010.jsonl", "validation", "评估中 · 门 10"),
        ("early-validation-step-000000020.jsonl", "early_validation", "早期评估中 · 门 20"),
    )
    for index, (journal_name, expected_kind, expected_label) in enumerate(cases):
        now = 2_000.0 + index
        root = tmp_path / f"artifacts-{index}"
        run_directory, metrics = _create_run(root, RUN_A, unix_s=100.0)
        os.utime(metrics, (100.0, 100.0))
        journal = run_directory / journal_name
        _append_jsonl(journal, {"event": "decision"})
        os.utime(journal, (now - 1.0, now - 1.0))

        snapshot = DashboardStore(root, stale_seconds=100.0, now=lambda now=now: now).snapshot()

        assert snapshot["status"]["label"] == expected_label
        assert snapshot["lifecycle"]["pending_evaluation"]["kind"] == expected_kind


def test_repeated_evaluation_gate_attempt_is_independent_and_visible(
    tmp_path: Path,
) -> None:
    now = 3_000.0
    root = tmp_path / "artifacts"
    run_directory, metrics = _create_run(root, RUN_A, unix_s=100.0)
    _append_jsonl(
        metrics,
        {
            "event": "evaluation",
            "unix_s": 200.0,
            "evaluation_gate": 20,
            "evaluation_attempt": 1,
            "evaluation_guard_rollbacks": 0,
            "gate_kind": "early_validation",
            "environment_steps": 20,
            "episodes": 16,
        },
    )
    os.utime(metrics, (200.0, 200.0))
    journal_name = "early-validation-step-000000020-attempt-002.jsonl"
    journal = run_directory / journal_name
    _append_jsonl(
        journal,
        {
            "event": "evaluation_started",
            "evaluation_gate": 20,
            "evaluation_attempt": 2,
            "evaluation_guard_rollbacks": 1,
            "gate_kind": "early_validation",
            "evaluation_seeds": [7],
        },
    )
    os.utime(journal, (now - 1.0, now - 1.0))

    store = DashboardStore(root, stale_seconds=100.0, now=lambda: now)
    snapshot = store.snapshot()
    run_key = f"lineage/run-{RUN_A}"
    journals = store.heldout_journals(run_key)["journals"]

    assert snapshot["status"]["label"] == "早期评估中 · 门 20 · 尝试 2"
    assert snapshot["status"]["evaluation_attempt"] == 2
    assert snapshot["lifecycle"]["pending_evaluation"]["attempt"] == 2
    assert journals[0]["evaluation_attempt"] == 2
    assert journals[0]["complete"] is False

    _append_jsonl(
        metrics,
        {
            "event": "evaluation",
            "unix_s": now,
            "evaluation_gate": 20,
            "evaluation_attempt": 2,
            "evaluation_guard_rollbacks": 1,
            "gate_kind": "early_validation",
            "environment_steps": 20,
            "episodes": 16,
        },
    )
    os.utime(metrics, (now, now))

    assert store.snapshot()["lifecycle"]["pending_evaluation"] is None
    assert store.heldout_journals(run_key)["journals"][0]["complete"] is True


def test_evaluation_completion_and_projection_are_keyed_by_kind_and_gate(
    tmp_path: Path,
) -> None:
    now = 1_000.0
    root = tmp_path / "artifacts"
    run_directory, metrics = _create_run(root, RUN_A, unix_s=100.0)
    _append_jsonl(
        metrics,
        {
            "event": "evaluation",
            "unix_s": 900.0,
            "evaluation_gate": 100,
            "environment_steps": 100,
            "episodes": 1,
        },
    )
    os.utime(metrics, (900.0, 900.0))
    validation = run_directory / "evaluation-step-000000100.jsonl"
    final_audit = run_directory / "final-audit-step-000000100.jsonl"
    _append_jsonl(validation, {"event": "decision"})
    _append_jsonl(final_audit, {"event": "decision"})
    os.utime(validation, (899.0, 899.0))
    os.utime(final_audit, (995.0, 995.0))

    store = DashboardStore(root, stale_seconds=100.0, now=lambda: now)
    pending = store.snapshot()

    assert pending["lifecycle"]["pending_evaluation"]["kind"] == "final_audit"
    assert pending["lifecycle"]["pending_evaluation"]["gate"] == 100
    assert pending["status"]["label"] == "最终审计中 · 门 100"

    _append_jsonl(
        metrics,
        {
            "event": "evaluation",
            "unix_s": 999.0,
            "evaluation_gate": 100,
            "gate_kind": "final_audit",
            "data_partition": "final_audit",
            "environment_steps": 100,
            "episodes": 1,
            "deadlock_rate": 1.0,
        },
    )
    os.utime(metrics, (999.0, 999.0))
    completed = store.snapshot()

    assert completed["lifecycle"]["pending_evaluation"] is None
    assert [(evaluation["gate_kind"], evaluation["evaluation_gate"]) for evaluation in completed["evaluations"]] == [
        ("validation", 100),
        ("final_audit", 100),
    ]
    assert completed["evaluations"][-1]["gate_kind"] == "final_audit"
    assert any(alert["type"] == "heldout_deadlock" for alert in completed["alerts"])


def test_final_audit_policy_mismatch_is_visible_and_not_certified(
    tmp_path: Path,
) -> None:
    root = tmp_path / "artifacts"
    _, metrics = _create_run(root, RUN_A, unix_s=100.0)
    _append_jsonl(
        metrics,
        {
            "event": "evaluation",
            "unix_s": 200.0,
            "evaluation_gate": 250,
            "gate_kind": "final_audit",
            "data_partition": "final_audit",
            "environment_steps": 250,
            "episodes": 2,
            "evaluation_context": {
                "policy_version": 7,
                "actor_policy_version": 7,
                "policy_model_state_sha256": "a" * 64,
            },
        },
        {
            "event": "run_complete",
            "unix_s": 201.0,
            "environment_steps": 250,
            "policy_version": 8,
            "actor_policy_version": 8,
        },
    )

    snapshot = DashboardStore(root, now=lambda: 202.0).snapshot()

    latest = snapshot["evaluations"][-1]
    assert latest["gate_kind"] == "final_audit"
    assert latest["policy_version"] == 7
    assert latest["actor_policy_version"] == 7
    assert latest["policy_model_state_sha256"] == "a" * 64
    mismatch = [alert for alert in snapshot["alerts"] if alert["type"] == "final_audit_policy_mismatch"]
    assert len(mismatch) == 1
    assert "策略版本 7" in mismatch[0]["message"]
    assert "run_complete 策略版本 8" in mismatch[0]["message"]


def test_evaluation_projection_prefers_reliable_event_time_over_gate_number(
    tmp_path: Path,
) -> None:
    root = tmp_path / "artifacts"
    _, metrics = _create_run(root, RUN_A, unix_s=100.0)
    _append_jsonl(
        metrics,
        {
            "event": "evaluation",
            "unix_s": 200.0,
            "evaluation_gate": 200,
            "gate_kind": "validation",
            "environment_steps": 200,
            "episodes": 1,
        },
        {
            "event": "evaluation",
            "unix_s": 300.0,
            "evaluation_gate": 100,
            "gate_kind": "validation",
            "environment_steps": 300,
            "episodes": 1,
        },
    )

    evaluations = DashboardStore(root, now=lambda: 301.0).snapshot()["evaluations"]

    assert [evaluation["evaluation_gate"] for evaluation in evaluations] == [200, 100]


def test_stale_final_audit_journal_does_not_override_stale_status(tmp_path: Path) -> None:
    now = 1_000.0
    root = tmp_path / "artifacts"
    run_directory, metrics = _create_run(root, RUN_A, unix_s=100.0)
    os.utime(metrics, (100.0, 100.0))
    final_audit = run_directory / "final-audit-step-000000100.jsonl"
    _append_jsonl(final_audit, {"event": "decision"})
    os.utime(final_audit, (200.0, 200.0))

    store = DashboardStore(root, stale_seconds=100.0, now=lambda: now)
    quick_status = store.list_runs()["runs"][0]["status"]
    snapshot = store.snapshot()

    assert quick_status["state"] == "stale"
    assert snapshot["status"]["state"] == "stale"
    assert snapshot["lifecycle"]["pending_evaluation"]["kind"] == "final_audit"


def test_evaluation_journal_scan_is_bounded_and_skips_overlong_gate_names(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    now = 1_000.0
    root = tmp_path / "artifacts"
    run_directory, metrics = _create_run(root, RUN_A, unix_s=100.0)
    os.utime(metrics, (100.0, 100.0))
    overlong_gate = run_directory / f"evaluation-step-{'9' * 40}.jsonl"
    final_audit = run_directory / "final-audit-step-000000100.jsonl"
    _append_jsonl(overlong_gate, {"event": "decision"})
    _append_jsonl(final_audit, {"event": "decision"})
    os.utime(final_audit, (now - 1.0, now - 1.0))

    original_iterdir = Path.iterdir
    scan_limit = monitoring_module._MAX_EVALUATION_JOURNAL_SCAN_ENTRIES

    def guarded_iterdir(path: Path) -> Iterator[Path]:
        if path != run_directory:
            return original_iterdir(path)

        def entries() -> Iterator[Path]:
            yield overlong_gate
            yield final_audit
            for index in range(scan_limit - 2):
                yield run_directory / f"unrelated-{index:06d}.tmp"
            raise AssertionError("evaluation journal scan exceeded its entry bound")

        return entries()

    monkeypatch.setattr(Path, "iterdir", guarded_iterdir)

    snapshot = DashboardStore(root, stale_seconds=100.0, now=lambda: now).snapshot()

    assert snapshot["status"]["label"] == "最终审计中 · 门 100"
    assert snapshot["lifecycle"]["pending_evaluation"]["journal_name"] == final_audit.name


def test_dashboard_copy_names_collection_progress_and_disclaims_run_completion() -> None:
    html = _dashboard_html().decode("utf-8")

    assert '<div class="metric-label">环境采集进度</div>' in html
    assert "采集目标已达" in html
    assert "整个运行尚未完成" in html
    assert "评估门禁提前停止" in html
    assert "采集目标未完成" in html
    assert "const latest = evaluations[evaluations.length - 1]" in html
    assert "Held-out 对局下钻" in html
    assert "拿牌记录" in html
    assert "最终构筑" in html
    assert '["display_name", "card_id", "label", "kind"]' in html
    assert 'const displayName = first(item || {}, ["display_name"], null);' in html
    assert "内部 ID ${id}" in html
    assert 'id="heldout-map-visual"' in html
    assert 'id="heldout-floor-visual"' in html
    assert 'id="heldout-final-deck"' in html
    assert "将由 seed 与已记录动作在后台复现完整地图" in html
    assert "匹配 simulator 中精确复现完整地图" in html
    assert "历史日志未记录完整拓扑时" not in html
    assert "策略回滚处断线" in html
    assert 'xLabel: "训练 Episode' in html
    assert "point.breakBefore !== true" in html
    assert 'rollback_generation: "回滚代"' in html
    assert 'const API_HELDOUT_JOURNALS = "/api/v1/heldout-journals"' in html
    assert 'const API_HELDOUT_EPISODE = "/api/v1/heldout-episode"' in html
    assert 'const API_HELDOUT_REPLAY_MAP = "/api/v1/heldout-replay-map"' in html
    assert "function macroActionsForFloor(episode, floor)" in html
    assert "自动推进" in html and "唯一合法动作" in html
    assert "不代表模型在多个策略中选择" in html
    assert '" · forced"' not in html
    assert "innerHTML" not in html


def test_exact_resume_parent_is_listed_as_continued(tmp_path: Path) -> None:
    root = tmp_path / "artifacts"
    _, parent_metrics = _create_run(root, RUN_A, unix_s=100.0)
    os.utime(parent_metrics, (100.0, 100.0))
    _create_run(
        root,
        RUN_B,
        unix_s=200.0,
        checkpoint_load={
            "mode": "exact_resume",
            "parent_checkpoint": (f"/runtime/checkpoints/lineage/run-{RUN_A}/periodic-step-000000010"),
        },
    )

    store = DashboardStore(root, stale_seconds=10.0, now=lambda: 1_000.0)
    statuses = {item["run_id"]: item["status"] for item in store.list_runs()["runs"]}

    assert statuses[RUN_A] == {
        "state": "continued",
        "phase": "continued",
        "label": "已续跑",
    }
    parent_snapshot = store.snapshot(f"lineage/run-{RUN_A}")
    assert parent_snapshot["status"]["state"] == "continued"
    assert f"lineage/run-{RUN_B}" in parent_snapshot["status"]["evidence"][0]


def test_checkpoint_publication_requires_public_directory_manifest_and_matching_metadata(tmp_path: Path) -> None:
    root = tmp_path / "artifacts"
    _, metrics = _create_run(root, RUN_A, checkpoint_dir="checkpoints/lineage")
    os.utime(metrics, (100.0, 100.0))
    checkpoint_root = root / "checkpoints" / "lineage" / f"run-{RUN_A}"

    published = checkpoint_root / "periodic-step-000000010"
    published.mkdir(parents=True)
    payload = published / "model.bin"
    payload.write_bytes(b"model")
    metadata_text = json.dumps(
        {
            "checkpoint_id": "checkpoint-1",
            "training_state": {"environment_steps": 10},
        }
    )
    (published / "metadata.json").write_text(metadata_text, encoding="utf-8")
    (published / "checkpoint.manifest.json").write_text(
        json.dumps(
            {
                "checkpoint_id": "checkpoint-1",
                "created_unix_s": 110.0,
                "format": "sts2-atomic-checkpoint-v1",
                "files": [
                    {"path": "model.bin", "size_bytes": 5},
                    {
                        "path": "metadata.json",
                        "size_bytes": len(metadata_text.encode("utf-8")),
                    },
                ],
            }
        ),
        encoding="utf-8",
    )

    incomplete = checkpoint_root / ".periodic-step-000000020.incomplete-token"
    incomplete.mkdir()
    (incomplete / "checkpoint.manifest.json").write_text("{}", encoding="utf-8")
    (incomplete / "metadata.json").write_text("{}", encoding="utf-8")

    invalid = checkpoint_root / "final-step-000000030"
    invalid.mkdir()
    (invalid / "checkpoint.manifest.json").write_text(
        json.dumps({"checkpoint_id": "one", "files": []}),
        encoding="utf-8",
    )
    (invalid / "metadata.json").write_text(
        json.dumps({"checkpoint_id": "two", "training_state": {"environment_steps": 29}}),
        encoding="utf-8",
    )

    snapshot = DashboardStore(root, now=lambda: 120.0).snapshot()
    checkpoints = snapshot["checkpoints"]

    assert [item["name"] for item in checkpoints["published"]] == [
        "periodic-step-000000010",
        "final-step-000000030",
    ]
    assert checkpoints["published"][0]["valid"] is True
    assert checkpoints["published"][0]["verification"] == "size_verified"
    assert checkpoints["published"][0]["size_bytes"] == 5 + len(metadata_text.encode("utf-8"))
    assert checkpoints["published"][1]["valid"] is False
    assert "checkpoint_id_mismatch" in checkpoints["published"][1]["issues"]
    assert "environment_step_mismatch" in checkpoints["published"][1]["issues"]
    assert [item["name"] for item in checkpoints["publishing"]] == ["periodic-step-000000020"]
    assert all(".incomplete-" not in item["name"] for item in checkpoints["published"])


def test_empty_checkpoint_objects_are_not_valid_restore_candidates(tmp_path: Path) -> None:
    root = tmp_path / "artifacts"
    _create_run(root, RUN_A, checkpoint_dir="checkpoints/lineage")
    checkpoint = root / "checkpoints" / "lineage" / f"run-{RUN_A}" / "periodic-step-000000010"
    checkpoint.mkdir(parents=True)
    (checkpoint / "checkpoint.manifest.json").write_text("{}", encoding="utf-8")
    (checkpoint / "metadata.json").write_text("{}", encoding="utf-8")

    published = DashboardStore(root, now=lambda: 120.0).snapshot()["checkpoints"]["published"]

    assert len(published) == 1
    assert published[0]["valid"] is False
    assert published[0]["verification"] == "invalid"
    assert "manifest_checkpoint_id_missing" in published[0]["issues"]
    assert "metadata_checkpoint_id_missing" in published[0]["issues"]
    assert "manifest_files_missing" in published[0]["issues"]
    assert "training_state_missing" in published[0]["issues"]


def test_lifetime_candidate_peaks_survive_episode_ring_and_failure_counts_stay_separate(
    tmp_path: Path,
) -> None:
    root = tmp_path / "artifacts"
    _, metrics = _create_run(root, RUN_A, unix_s=1.0)
    for index in range(510):
        _append_jsonl(
            metrics,
            {
                "event": "train_episode",
                "unix_s": float(index + 2),
                "environment_steps": index + 1,
                "maximum_observed_candidates": 300 if index == 0 else 20,
                "maximum_observed_semantic_candidates": 250 if index == 0 else 15,
                "maximum_equivalence_class_size": 200 if index == 0 else 10,
            },
        )
    _append_jsonl(
        metrics,
        {
            "event": "evaluation",
            "unix_s": 600.0,
            "evaluation_gate": 500,
            "environment_steps": 510,
            "episodes": 2,
            "deadlock_rate": 0.5,
            "combat_progress_stall_rate": 0.5,
            "noncombat_progress_stall_rate": 0.0,
            "combat_policy_failure_count": 1,
        },
    )

    snapshot = DashboardStore(root, now=lambda: 601.0).snapshot()
    progress = snapshot["progress"]
    evaluation = snapshot["evaluations"][0]

    assert len(snapshot["episodes"]) == 500
    assert progress["maximum_observed_candidates"] == 300
    assert progress["maximum_observed_semantic_candidates"] == 250
    assert progress["maximum_equivalence_class_size"] == 200
    assert evaluation["deadlock_count"] == 1
    assert evaluation["combat_progress_stall_count"] == 1
    assert evaluation["noncombat_progress_stall_count"] == 0
    assert evaluation["combat_policy_failure_count"] == 1
    assert "stall_failure_count" not in evaluation


def test_live_throughput_uses_fresh_learner_progress_and_expires_old_windows(
    tmp_path: Path,
) -> None:
    root = tmp_path / "fresh"
    _, metrics = _create_run(root, RUN_A, unix_s=1_000.0)
    _append_jsonl(
        metrics,
        {
            "event": "learner_progress",
            "unix_s": 1_500.0,
            "environment_steps": 400,
            "stage": "optimizing",
        },
        {
            "event": "learner_progress",
            "unix_s": 1_750.0,
            "environment_steps": 700,
            "stage": "optimizing",
        },
        {
            "event": "learner_update",
            "unix_s": 2_000.0,
            "environment_steps": 1_000,
            "policy_version": 1,
            "loss": 0.5,
        },
    )
    fresh = DashboardStore(root, now=lambda: 2_001.0).snapshot()
    assert fresh["throughput"]["env_steps_per_s_15m"] == 1.2
    assert fresh["throughput"]["scope"] == "live"

    expired_root = tmp_path / "expired"
    _, expired_metrics = _create_run(expired_root, RUN_B, unix_s=100.0)
    _append_jsonl(
        expired_metrics,
        {"event": "learner_progress", "unix_s": 200.0, "environment_steps": 100},
        {"event": "learner_progress", "unix_s": 300.0, "environment_steps": 200},
        {"event": "learner_progress", "unix_s": 400.0, "environment_steps": 300},
    )
    expired = DashboardStore(expired_root, now=lambda: 2_001.0).snapshot()
    assert expired["throughput"]["env_steps_per_s_15m"] is None


@contextmanager
def _running_server(server: DashboardHTTPServer) -> Iterator[DashboardHTTPServer]:
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        yield server
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=2)


def _request(
    server: DashboardHTTPServer,
    method: str,
    path: str,
    *,
    host_header: str | None = None,
) -> tuple[int, dict[str, str], bytes]:
    host, port = server.server_address[:2]
    connection = http.client.HTTPConnection(host, port, timeout=2)
    headers = {"Host": host_header} if host_header is not None else {}
    connection.request(method, path, headers=headers)
    response = connection.getresponse()
    body = response.read()
    response_headers = {key.lower(): value for key, value in response.getheaders()}
    status = response.status
    connection.close()
    return status, response_headers, body


def test_http_dashboard_routes_are_read_only_loopback_scoped_and_hardened(tmp_path: Path) -> None:
    root = tmp_path / "artifacts"
    _, metrics = _create_run(root, RUN_A, unix_s=100.0)
    _append_jsonl(
        metrics,
        {
            "event": "learner_update",
            "unix_s": 100.5,
            "environment_steps": 1,
            "policy_version": 1,
            "loss": float("nan"),
        },
    )
    store = DashboardStore(root, now=lambda: 101.0)
    server = DashboardHTTPServer(("127.0.0.1", 0), store, b"<!doctype html><title>monitor</title>")

    with _running_server(server):
        index_status, _index_headers, index_body = _request(server, "GET", "/")
        assert index_status == 200
        assert b"<title>monitor</title>" in index_body

        health_status, _, health_body = _request(server, "GET", "/health")
        assert health_status == 200
        health = json.loads(health_body)
        assert health == {
            "status": "ok",
            "schema": "sts2-training-dashboard-v1",
            "read_only": True,
            "run_count": 1,
            "artifact_root": str(root.resolve()),
        }

        runs_status, _, runs_body = _request(server, "GET", "/api/v1/runs")
        assert runs_status == 200
        runs = json.loads(runs_body)
        assert runs["selected"] == f"lineage/run-{RUN_A}"
        assert len(runs["runs"]) == 1

        snapshot_path = "/api/v1/snapshot?" + urlencode({"run": runs["selected"]})
        snapshot_status, snapshot_headers, snapshot_body = _request(server, "GET", snapshot_path)
        assert snapshot_status == 200
        snapshot_payload = json.loads(snapshot_body)
        assert snapshot_payload["run"]["run_id"] == RUN_A
        assert snapshot_payload["latest_learner"]["loss"] is None
        assert snapshot_headers["cache-control"].startswith("no-store")
        assert snapshot_headers["x-content-type-options"] == "nosniff"
        assert snapshot_headers["x-frame-options"] == "DENY"
        assert "default-src 'self'" in snapshot_headers["content-security-policy"]
        assert snapshot_headers["referrer-policy"] == "no-referrer"

        for method in ("POST", "PUT", "PATCH", "DELETE", "OPTIONS"):
            method_status, _, method_body = _request(server, method, "/api/v1/snapshot")
            assert method_status == 405
            assert json.loads(method_body)["message"] == "the training dashboard is read-only"

        bad_host_status, _, _ = _request(
            server,
            "GET",
            "/api/v1/runs",
            host_header="attacker.example",
        )
        assert bad_host_status == 403
        loopback_alias_status, _, _ = _request(
            server,
            "GET",
            "/api/v1/runs",
            host_header="127.0.0.2",
        )
        assert loopback_alias_status == 200

        missing_status, _, missing_body = _request(
            server,
            "GET",
            "/api/v1/snapshot?" + urlencode({"run": f"lineage/run-{RUN_B}"}),
        )
        assert missing_status == 404
        assert json.loads(missing_body)["error"] == "Not Found"


def test_http_heldout_drilldown_is_lazy_key_scoped_and_read_only(tmp_path: Path) -> None:
    root = tmp_path / "artifacts"
    run_directory, metrics = _create_run(root, RUN_A, unix_s=100.0)
    _append_jsonl(
        metrics,
        {
            "event": "evaluation",
            "unix_s": 101.0,
            "gate_kind": "validation",
            "evaluation_gate": 75_000,
            "environment_steps": 75_000,
            "episodes": 1,
        },
    )
    episode_id = "heldout-seed-7-attempt-1:episode"
    journal_name = "evaluation-step-000075000.jsonl"
    _append_jsonl(
        run_directory / journal_name,
        {
            "event": "evaluation_started",
            "evaluation_gate": 75_000,
            "gate_kind": "validation",
            "policy_version": 19,
            "evaluation_seeds": [7],
        },
        {
            "event": "decision",
            "episode_id": episode_id,
            "step_index": 0,
            "reset_seed": 7,
            "player_hp_lost": 0,
            "revivals_used": 0,
            "outcome": "success",
            "selected_candidate_index": 0,
            "selected_action": {"action": "choose_event_option", "index": 0},
            "policy_topk": [],
            "observation_summary": {
                "screen": "EVENT",
                "run": {"act": 1, "floor": 1, "room_type": "event"},
                "player": {"character": "IRONCLAD", "hp": 80, "max_hp": 80, "gold": 99},
                "combat": {"in_progress": False},
            },
        },
        {
            "event": "evaluation_attempt_completed",
            "episode_id": episode_id,
            "evaluation_seed": 7,
            "attempt": 1,
            "steps": 1,
        },
    )
    replay_calls: list[tuple[Path, str, object]] = []

    class FakeReplayCache:
        def load(self, path: Path, *, episode_id: str, detail: object) -> dict[str, object]:
            replay_calls.append((path, episode_id, detail))
            return {
                "schema": "sts2-heldout-episode-replay-v2",
                "episode_id": episode_id,
                "map_topologies": [{"act": 1, "nodes": []}],
                "macro_actions": [
                    {
                        "step": 7,
                        "kind": "choose_rest_option",
                        "semantic_label": "休息回血: 20 → 44 (+24)",
                    }
                ],
                "macro_actions_omitted": 0,
                "reproduction": {"mode": "seed_and_recorded_action_replay"},
                "cache_hit": False,
            }

    store = DashboardStore(root)
    store._map_replay_cache = FakeReplayCache()  # type: ignore[assignment]
    server = DashboardHTTPServer(("127.0.0.1", 0), store, _dashboard_html())
    run_key = f"lineage/run-{RUN_A}"
    with _running_server(server):
        list_path = "/api/v1/heldout-journals?" + urlencode({"run": run_key})
        status, _, body = _request(server, "GET", list_path)
        assert status == 200
        journals = json.loads(body)["journals"]
        assert [journal["key"] for journal in journals] == [journal_name]
        assert journals[0]["complete"] is True

        index_path = "/api/v1/heldout-episodes?" + urlencode({"run": run_key, "journal": journal_name})
        status, _, body = _request(server, "GET", index_path)
        assert status == 200
        episodes = json.loads(body)["episodes"]
        assert episodes[0]["episode_id"] == episode_id

        detail_path = "/api/v1/heldout-episode?" + urlencode(
            {"run": run_key, "journal": journal_name, "episode": episode_id}
        )
        status, _, body = _request(server, "GET", detail_path)
        assert status == 200
        assert json.loads(body)["episode"]["provenance"]["game_version"] is None

        replay_path = "/api/v1/heldout-replay-map?" + urlencode(
            {"run": run_key, "journal": journal_name, "episode": episode_id}
        )
        status, _, body = _request(server, "GET", replay_path)
        assert status == 200
        replay = json.loads(body)
        assert replay["schema"] == "sts2-heldout-episode-replay-v2"
        assert replay["macro_actions"][0]["semantic_label"] == "休息回血: 20 → 44 (+24)"
        assert replay["reproduction"]["mode"] == "seed_and_recorded_action_replay"
        assert replay["run"]["run_id"] == RUN_A
        assert replay_calls[0][0] == run_directory / journal_name
        assert replay_calls[0][1] == episode_id

        unsafe_path = "/api/v1/heldout-episodes?" + urlencode(
            {"run": run_key, "journal": "../evaluation-step-000075000.jsonl"}
        )
        status, _, _ = _request(server, "GET", unsafe_path)
        assert status == 404
        status, _, _ = _request(server, "GET", detail_path + "&extra=1")
        assert status == 400
        status, _, _ = _request(server, "GET", replay_path + "&extra=1")
        assert status == 400
