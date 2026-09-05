from __future__ import annotations

import http.client
import json
import os
import shutil
import subprocess
import threading
from pathlib import Path
from typing import Any

import pytest

from sts2_rl import macro_dashboard as monitor


def _write(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value), encoding="utf-8")


def _run(root: Path) -> Path:
    run = root / "hybrid_runs" / "macro-credit-test"
    _write(run / "course.json", {
        "lineage_id": "macro-credit-test", "reward_identity": {"revival_budget": 64},
        "course_config": {"version": monitor.COURSE_VERSION, "branches": {"root_surface": "reward", "continuation_macro_decisions": 12},
                          "training": {"runtime": {"device": "cuda:0"}}},
    })
    _write(run / "status.json", {"state": "collecting", "committed_update": 2, "next_seed_index": 2,
                                "attempt": "/mnt/e/artifacts/hybrid_runs/macro-credit-test/cohorts/update-000002-attempt-new"})
    for update in (1, 2):
        _write(run / "cohorts" / f"update-{update-1:06d}-attempt-old" / "update.json", {
            "committed_update": update, "next_seed_index": update,
            "learner": {"version": "sts2-fresh-macro-credit-learner-v1", "loss": 0.7 - update * .1,
                        "parameter_delta_norm": .1, "macro_steps": 20, "combat_steps": 100,
                        "peak_cuda_allocated_bytes": 300000000, "process_peak_rss_bytes": 1900000000,
                        "root_targets": [{"teacher_probabilities": [.5, .5], "target_probabilities": [.6, .4]}]},
            "coverage": {"natural_macro_steps": 20, "natural_combat_steps": 100},
        })
    trace = run / "cohorts/update-000002-attempt-new/root-000/reference.jsonl"
    _write(trace, {"kind": "decision", "facts": {"state": {"observation": {"run": {"floor": 3}}},
                                                "semantic": {"surface": "combat", "domain": "combat"}}})
    with trace.open("a", encoding="utf-8") as handle:
        handle.write("\n{\"incomplete\":")
    (run / ".writer.lock").write_text("test", encoding="utf-8")
    return run


def test_new_course_snapshot_has_correct_identity_updates_and_partial_trace(tmp_path: Path) -> None:
    run = _run(tmp_path)
    before = {path: path.read_bytes() for path in run.rglob("*") if path.is_file()}
    snapshot = monitor.MacroDashboardStore(run, artifact_root=tmp_path).snapshot()
    assert snapshot["schema"] == monitor.SCHEMA
    assert snapshot["lineage_id"] == "macro-credit-test"
    assert snapshot["state"] == "collecting"
    assert snapshot["committed_update"] == 2
    assert snapshot["next_training_seed"] == 13400004
    assert [row["update"] for row in snapshot["history"]] == [1, 2]
    assert snapshot["progress"]["floor"] == 3
    assert snapshot["writer_lock_present"] is True
    assert snapshot["process_liveness_verified"] is False
    assert snapshot["measurement_scope"]["quality_certified"] is False
    assert before == {path: path.read_bytes() for path in run.rglob("*") if path.is_file()}


def test_refresh_observes_new_status_without_restart(tmp_path: Path) -> None:
    run = _run(tmp_path)
    store = monitor.MacroDashboardStore(run, artifact_root=tmp_path)
    assert store.snapshot()["committed_update"] == 2
    _write(run / "status.json", {"state": "aborted", "committed_update": 3, "next_seed_index": 3,
                                "error_type": "ProbeBudgetExhausted", "error": "maximum_root_snapshot_bytes"})
    snapshot = store.snapshot()
    assert snapshot["committed_update"] == 3
    assert snapshot["error_type"] == "ProbeBudgetExhausted"
    assert snapshot["error"] == "maximum_root_snapshot_bytes"
    assert any("ahead" in warning for warning in snapshot["warnings"])


def test_wrong_or_changed_course_is_not_silently_used(tmp_path: Path) -> None:
    run = _run(tmp_path)
    store = monitor.MacroDashboardStore(run, artifact_root=tmp_path)
    _write(run / "course.json", {"lineage_id": "old-vtrace", "course_config": {"version": "legacy"}})
    with pytest.raises(ValueError, match="identity changed"):
        store.snapshot()
    with pytest.raises(ValueError, match="not a synchronous"):
        monitor.MacroDashboardStore(run, artifact_root=tmp_path)


def test_invocation_target_bound_to_exact_lineage(tmp_path: Path) -> None:
    run = _run(tmp_path)
    invocation = tmp_path / "analysis/pilot"
    _write(invocation / "process.json", {"argv": ["--lineage-id", "macro-credit-test"], "pid": 271,
                                         "starting_update": 2, "maximum_target_update": 22,
                                         "maximum_additional_updates": 20})
    _write(invocation / "launch.json", {"windows_wsl_pid": 74140})
    _write(invocation / "exit.json", {"returncode": 0, "exited_utc": "2026-09-05T05:00:00Z"})
    snapshot = monitor.MacroDashboardStore(run, artifact_root=tmp_path, invocation_dir=invocation).snapshot()
    assert snapshot["invocation"]["target_update"] == 22
    assert snapshot["invocation"]["wsl_training_pid_recorded"] == 271
    assert snapshot["invocation"]["process_liveness_verified"] is False
    assert snapshot["invocation"]["exit"]["exited_utc"] == "2026-09-05T05:00:00Z"
    _write(invocation / "process.json", {"argv": ["--lineage-id", "unrelated"], "maximum_target_update": 999})
    wrong = monitor.MacroDashboardStore(run, artifact_root=tmp_path, invocation_dir=invocation).snapshot()
    assert wrong["invocation"] == {}
    assert any("lineage differs" in warning for warning in wrong["warnings"])


def test_oversized_and_nonfinite_metadata_are_bounded(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    run = _run(tmp_path)
    store = monitor.MacroDashboardStore(run, artifact_root=tmp_path)
    (run / "status.json").write_text('{"committed_update": NaN}', encoding="utf-8")
    snapshot = store.snapshot()
    assert snapshot["committed_update"] == 2  # Visible committed update record, not invalid status.
    assert snapshot["warnings"]
    monkeypatch.setattr(monitor, "MAX_JSON_BYTES", 1)
    with pytest.raises(ValueError, match="size limit"):
        store.snapshot()


def test_symlink_escape_is_not_read(tmp_path: Path) -> None:
    run = _run(tmp_path)
    store = monitor.MacroDashboardStore(run, artifact_root=tmp_path)
    outside = tmp_path / "outside.json"
    _write(outside, {"secret": "must not read"})
    link = run / "linked.json"
    try:
        link.symlink_to(outside)
    except OSError:
        pytest.skip("symlink creation unavailable")
    with pytest.raises(ValueError, match="boundary"):
        store._read_json(link, run)


@pytest.mark.parametrize("host", ["0.0.0.0", "192.168.2.1", "::", "example.com"])
def test_server_refuses_nonloopback_binding(tmp_path: Path, host: str) -> None:
    with pytest.raises(ValueError, match="loopback"):
        monitor.make_server(tmp_path, artifact_root=tmp_path, host=host, port=0)


def test_http_serves_selected_run_and_rejects_mutations_and_host_attack(tmp_path: Path) -> None:
    run = _run(tmp_path)
    server = monitor.make_server(run, artifact_root=tmp_path, port=0)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    host, port = server.server_address[:2]
    connection = http.client.HTTPConnection(host, port, timeout=5)
    try:
        connection.request("GET", "/api/v1/snapshot")
        response = connection.getresponse()
        assert response.status == 200
        assert response.getheader("Cache-Control") == "no-store"
        payload = json.loads(response.read())
        assert payload["lineage_id"] == "macro-credit-test"
        assert payload["committed_update"] == 2
        connection.request("GET", "/")
        response = connection.getresponse()
        assert "新同步训练" in response.read().decode()
        for method, path, headers, expected in (
            ("POST", "/api/v1/snapshot", {}, 405),
            ("DELETE", "/", {}, 405),
            ("GET", "/api/v1/snapshot?run=old", {}, 400),
            ("GET", "/checkpoint.pt", {}, 404),
            ("GET", "/api/v1/snapshot", {"Host": "attacker.invalid"}, 403),
        ):
            connection.request(method, path, headers=headers)
            response = connection.getresponse()
            assert response.status == expected
            response.read()
    finally:
        connection.close()
        server.shutdown()
        server.server_close()
        thread.join(timeout=5)


def test_ui_has_single_flight_polling_and_unambiguous_metric_labels() -> None:
    html = (Path(monitor.__file__).parent / "dashboard_static/macro-credit.html").read_text(encoding="utf-8")
    assert "setInterval(" not in html
    assert "finally{setTimeout(refresh,2000)}" in html
    assert "λ-return Y" in html
    assert "RSS 历史高水位" in html
    assert "GPU peak 仅为 learner 窗口" in html
    assert "s.error_type" in html


def test_dashboard_read_handle_does_not_block_actual_posix_trainer_replace(tmp_path: Path) -> None:
    if os.name == "nt" and os.environ.get("STS2_RUN_WSL_INTEGRATION") != "1":
        pytest.skip("set STS2_RUN_WSL_INTEGRATION=1 for the Windows-reader / WSL-writer contract")
    target = tmp_path / "status.json"
    replacement = tmp_path / "status.next.json"
    target.write_bytes(b"old complete record")
    replacement.write_bytes(b"new complete record")
    with monitor._open_artifact_read(target) as handle:
        if os.name == "nt":
            if not shutil.which("wsl.exe"):
                pytest.skip("actual Windows-reader / WSL-writer integration requires WSL")
            drive = target.drive.removesuffix(":").lower()
            if len(drive) != 1:
                pytest.skip("WSL integration requires a mounted drive path")
            target_wsl = f"/mnt/{drive}/" + target.as_posix()[3:]
            replacement_wsl = f"/mnt/{drive}/" + replacement.as_posix()[3:]
            result = subprocess.run(
                ["wsl.exe", "-d", "Ubuntu-24.04", "--exec", "/usr/bin/python3", "-c",
                 "import os,sys; os.replace(sys.argv[1],sys.argv[2])", replacement_wsl, target_wsl],
                capture_output=True, text=True, timeout=20, check=False,
            )
            assert result.returncode == 0, result.stderr
        else:
            os.replace(replacement, target)
        assert handle.read() == b"old complete record"
    assert target.read_bytes() == b"new complete record"
