from __future__ import annotations

import http.client
import json
import os
import threading
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path
from urllib.parse import urlencode

from sts2_rl.monitor_dashboard import DashboardHTTPServer
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
            "parent_checkpoint": (
                f"/runtime/checkpoints/lineage/run-{RUN_A}/periodic-step-000000010"
            ),
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
    assert len(snapshot["evaluations"]) == 1
    assert snapshot["evaluations"][0]["evaluation_gate"] == 0
    assert snapshot["evaluations"][0]["act_1_success_rate"] == 0.0
    assert [point["policy_version"] for point in snapshot["learner_series"]] == [2, 3]
    assert [point["loss"] for point in snapshot["learner_series"]] == [0.8, 0.5]
    assert snapshot["context"] == {
        "mode": "native-revival-preheat",
        "revival_budget": -1,
        "standard_game": False,
        "label": "无限原生复活预热 · 非标准胜率",
    }


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
            "parent_checkpoint": (
                f"/runtime/checkpoints/lineage/run-{RUN_A}/periodic-step-000000010"
            ),
        },
    )

    store = DashboardStore(root, stale_seconds=10.0, now=lambda: 1_000.0)
    statuses = {
        item["run_id"]: item["status"]
        for item in store.list_runs()["runs"]
    }

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
    checkpoint = (
        root
        / "checkpoints"
        / "lineage"
        / f"run-{RUN_A}"
        / "periodic-step-000000010"
    )
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
        index_status, index_headers, index_body = _request(server, "GET", "/")
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
