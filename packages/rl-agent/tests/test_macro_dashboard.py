from __future__ import annotations

import copy
import http.client
import json
import os
import shutil
import subprocess
import threading
from pathlib import Path
from types import SimpleNamespace
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
        html = response.read().decode()
        assert "STS2" in html
        assert '<body data-dashboard-mode="macro-credit">' in html
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


def _collection(run: Path, *, update: int = 1, reference_tail: bool = True) -> Path:
    attempt = run / "cohorts" / f"update-{update-1:06d}-attempt-old"
    directory = attempt / "root-000"
    root_id = f"root-{update}"
    source = json.loads((attempt / "update.json").read_text())
    source["learner"].update({"teacher_version": update - 1, "elapsed_seconds": 20.0, "root_targets": [{
        "root_id": root_id, "action_values": [-1.0, -0.5], "teacher_probabilities": [.6, .4],
        "target_probabilities": [.5, .5], "updated_root_probabilities": [.55, .45], "value_target": -.8,
        "kl": .02, "updated_policy_kl_to_teacher": .005, "projection_kl_to_updated_policy": .004,
        "expected_improvement": .05, "actor_enabled": True, "temperature": .2,
    }]})
    _write(attempt / "update.json", source)
    reference = {
        "name": "reference", "policy_mode": "frozen_stochastic", "ending": "macro_horizon",
        "selected_root_index": 0, "root_floor": 3, "ending_floor": 7, "post_root_floor_span": 4,
        "native_steps": 145, "prefix_native_steps": 54, "branch_native_steps": 91,
        "semantic_decisions": 141, "macro_decisions": 20, "combat_decisions": 121,
        "continuation_macro_decisions": 12, "continuation_combat_decisions": 75,
        "post_root_combats_exited": 3, "post_root_combats_completed": 0,
        "observed_discounted_return": -.2, "tail_discount": .99, "tail_value_used": False,
        "raw_reward_components": {"reward": -.2, "progress_reward": .1, "potential_reward": .1,
                                  "revival_penalty": -.1, "hp_loss_penalty": -.2, "terminal_reward": 0.0, "pace_penalty": 0.0},
        "discounted_reward_components": {"reward": -.2, "potential_reward": .1, "hp_loss_penalty": -.3},
        "reward_state": {"last_total_revivals": 3, "act_revivals": 3, "current_act": 1},
        "root": {"surface": "reward", "candidates": [
            {"branch": "take", "target": {"id": "CARD.ANGER"}, "plan": ["huge plan not forwarded"]},
            {"branch": "skip", "target": {}}]},
    }
    control = copy.deepcopy(reference)
    control["name"] = "same-action-control"
    forced = copy.deepcopy(reference)
    forced.update({"name": "candidate-001", "selected_root_index": 1, "ending_floor": 5,
                   "post_root_floor_span": 2, "observed_discounted_return": -.1})
    _write(directory / "collection.json", {
        "format": monitor.COLLECTION_VERSION, "status": "complete", "seed": 13400000 + 2 * (update - 1),
        "teacher_version": update - 1, "root_id": root_id, "candidate_count": 2,
        "continuation_policy": "frozen_stochastic", "data_partition": "training", "same_action_gate": "matched",
        "passes": [reference, control, forced],
    })
    if reference_tail:
        rows = [
            {"kind": "macro_horizon", "facts": {"state": {"terminated": False, "truncated": False,
                "terminal_reason": None, "observation": {"_training": {"player_hp_lost": 19.5, "revivals_used": 3}},
                "transition": {"facts": {"run_result": None}}}}},
            {"kind": "end", "facts": {"ending_floor": 7}},
        ]
        (directory / "reference.jsonl").write_text("\n".join(json.dumps(row) for row in rows) + "\n", encoding="utf-8")
    return directory


def test_gameplay_is_one_natural_window_with_separate_same_root_alternatives(tmp_path: Path) -> None:
    run = _run(tmp_path)
    directory = _collection(run)
    before = {path: path.read_bytes() for path in run.rglob("*") if path.is_file()}
    result = monitor.MacroDashboardStore(run, artifact_root=tmp_path).snapshot()
    gameplay = result["history"][0]["gameplay"]
    assert gameplay["scope"] == "training_root_windows_not_full_episodes"
    assert gameplay["natural_reference_count"] == 1
    assert gameplay["candidate_trajectory_count"] == 2  # Does not count same-action control as an episode.
    root = gameplay["roots"][0]
    assert root["teacher_version"] == 0 and result["history"][0]["update"] == 1
    assert root["seed"] == 13400000 and root["reference_candidate_label"] == "CARD.ANGER"
    natural = root["natural_reference"]
    assert natural["native_steps"] == natural["prefix_native_steps"] + natural["branch_native_steps"] == 145
    assert natural["observed_discounted_return"] == -.2
    assert natural["last_total_revivals"] == 3 and natural["window_revivals"] is None
    assert natural["act_revivals"] == 3
    assert natural["player_hp_lost_since_reset"] == 19.5  # Not reverse-engineered from a reward coefficient.
    assert natural["trace_revivals_since_reset"] == 3 and natural["terminal_reason"] is None
    assert natural["post_root_combats_completed"] == 0 and natural["post_root_combats_exited"] == 3
    assert root["same_action_control"]["name"] == "same-action-control"
    forced = root["candidates"][1]
    assert forced["ordinary_training_sample"] is False
    assert forced["pass"]["ending_floor"] == 5  # Same horizon is not the same final floor.
    assert forced["observed_return_delta_vs_reference"] == pytest.approx(.1)
    assert forced["pass"]["player_hp_lost_since_reset"] is None  # No candidate tail scans.
    assert forced["target_probability_delta"] == pytest.approx(.1)
    assert forced["updated_probability_delta"] == pytest.approx(.05)
    assert root["value_target"] == -.8
    assert root["expected_target_improvement"] == .05
    assert root["updated_target_surrogate_improvement"] == pytest.approx(.025)
    assert gameplay["learner_elapsed_seconds"] == 20 and gameplay["collection_elapsed_seconds"] is None
    assert gameplay["collection_elapsed_source"] == "not_recorded"
    assert "potential_reward" not in natural["raw_reward_components"]
    assert natural["discounted_reward_components"]["progress_reward"] == .1
    assert "win_rate" not in root and "victory" not in natural
    assert "not paired" in gameplay["cross_update_comparison"]
    assert "plan" not in json.dumps(gameplay)
    assert directory.is_dir()
    assert before == {path: path.read_bytes() for path in run.rglob("*") if path.is_file()}


def test_immutable_projection_cache_reuses_reads_and_invalidates_on_metadata_change(tmp_path: Path, monkeypatch) -> None:
    run = _run(tmp_path)
    directory = _collection(run)
    store = monitor.MacroDashboardStore(run, artifact_root=tmp_path)
    original = monitor._open_artifact_read
    opened = []

    def observe(path):
        opened.append(path)
        return original(path)

    monkeypatch.setattr(monitor, "_open_artifact_read", observe)
    first = store.snapshot()
    assert directory / "collection.json" in opened and directory / "reference.jsonl" in opened
    first["history"][0]["root_targets"][0]["value_target"] = 999  # No alias back into cached JSON.
    first["history"][0]["gameplay"]["roots"][0]["natural_reference"]["observed_discounted_return"] = 999
    opened.clear()
    second = store.snapshot()
    assert opened == []
    assert second["history"][0]["root_targets"][0]["value_target"] == -.8
    assert second["history"][0]["gameplay"]["roots"][0]["natural_reference"]["observed_discounted_return"] == -.2
    collection = json.loads((directory / "collection.json").read_text())
    collection["passes"][0]["observed_discounted_return"] = -.3
    _write(directory / "collection.json", collection)
    third = store.snapshot()
    assert opened == [directory / "collection.json"]
    assert third["history"][0]["gameplay"]["roots"][0]["natural_reference"]["observed_discounted_return"] == -.3


def test_cache_signature_does_not_mix_windows_path_and_handle_ctime_semantics():
    path_stat = SimpleNamespace(st_mtime_ns=100, st_size=200, st_ino=300, st_ctime_ns=400)
    handle_stat = SimpleNamespace(st_mtime_ns=100, st_size=200, st_ino=300, st_ctime_ns=999)
    assert monitor._signature(path_stat) == monitor._signature(handle_stat)
    for key in ("st_mtime_ns", "st_size", "st_ino"):
        changed = SimpleNamespace(**vars(path_stat))
        setattr(changed, key, getattr(changed, key) + 1)
        assert monitor._signature(path_stat) != monitor._signature(changed)


def test_real_shared_read_handle_has_same_portable_signature_as_path(tmp_path: Path):
    path = tmp_path / "identity.json"
    path.write_text("{}")
    with monitor._open_artifact_read(path) as handle:
        assert monitor._signature(path.stat()) == monitor._signature(os.fstat(handle.fileno()))


def test_reference_tail_is_bounded_skips_end_and_never_scans_candidate_tapes(tmp_path: Path, monkeypatch) -> None:
    run = _run(tmp_path)
    directory = _collection(run)
    reference = directory / "reference.jsonl"
    reference.write_bytes(b"x" * (monitor.MAX_TAIL_BYTES * 2) + b"\n" + reference.read_bytes())
    (directory / "candidate-001.jsonl").write_bytes(b"do not parse a candidate tape")
    original = monitor._open_artifact_read
    reads = []

    class Reader:
        def __init__(self, path):
            self.path = path
            self.handle = original(path)

        def __enter__(self):
            return self

        def __exit__(self, *args):
            self.handle.close()

        def __getattr__(self, name):
            return getattr(self.handle, name)

        def read(self, size):
            if self.path.suffix == ".jsonl":
                assert size <= monitor.MAX_TAIL_BYTES
                assert self.path.name != "candidate-001.jsonl"
                reads.append((self.path, size))
            return self.handle.read(size)

    monkeypatch.setattr(monitor, "_open_artifact_read", Reader)
    store = monitor.MacroDashboardStore(run, artifact_root=tmp_path)
    row = store.snapshot()["history"][0]["gameplay"]["roots"][0]["natural_reference"]
    assert row["player_hp_lost_since_reset"] == 19.5
    assert row["trace_evidence_record_kind"] == "macro_horizon"
    assert sum(path == reference for path, _ in reads) == 1


def test_cold_tail_budget_defers_older_rows_and_next_poll_fills_them(tmp_path: Path, monkeypatch) -> None:
    run = _run(tmp_path)
    _collection(run, update=1)
    _collection(run, update=2)
    monkeypatch.setattr(monitor, "MAX_REFERENCE_TAIL_READS", 1)
    store = monitor.MacroDashboardStore(run, artifact_root=tmp_path)
    first = [row["gameplay"]["roots"][0]["natural_reference"] for row in store.snapshot()["history"]]
    assert sum(row.get("trace_evidence_pending") is True for row in first) == 1
    second = [row["gameplay"]["roots"][0]["natural_reference"] for row in store.snapshot()["history"]]
    assert all(row["player_hp_lost_since_reset"] == 19.5 for row in second)


@pytest.mark.parametrize("bad_field,bad_value", [("status", "pending"), ("status", "aborted"),
                                                ("teacher_version", 999), ("root_id", "unrelated-root"),
                                                ("data_partition", "heldout"), ("continuation_policy", "greedy"),
                                                ("same_action_gate", "mismatched")])
def test_incompatible_or_uncommitted_collection_not_mixed_into_gameplay(tmp_path: Path, bad_field, bad_value) -> None:
    run = _run(tmp_path)
    directory = _collection(run)
    value = json.loads((directory / "collection.json").read_text())
    value[bad_field] = bad_value
    _write(directory / "collection.json", value)
    snapshot = monitor.MacroDashboardStore(run, artifact_root=tmp_path).snapshot()
    assert snapshot["history"][0]["gameplay"]["roots"] == []
    assert snapshot["warnings"]


def test_collection_without_committed_update_never_becomes_natural_history(tmp_path: Path) -> None:
    run = _run(tmp_path)
    directory = _collection(run)
    (directory.parent / "update.json").unlink()
    snapshot = monitor.MacroDashboardStore(run, artifact_root=tmp_path).snapshot()
    assert all(row["update"] != 1 for row in snapshot["history"])
    assert all(not row["gameplay"]["roots"] for row in snapshot["history"])


def test_missing_exact_counters_and_collection_time_stay_unknown(tmp_path: Path) -> None:
    run = _run(tmp_path)
    directory = _collection(run, reference_tail=False)
    value = json.loads((directory / "collection.json").read_text())
    for row in value["passes"]:
        row["reward_state"] = {}
    _write(directory / "collection.json", value)
    natural = monitor.MacroDashboardStore(run, artifact_root=tmp_path).snapshot()["history"][0]["gameplay"]["roots"][0]["natural_reference"]
    assert natural["last_total_revivals"] is natural["window_revivals"] is None
    assert natural["player_hp_lost_since_reset"] is natural["trace_revivals_since_reset"] is None
    assert natural["terminated"] is natural["truncated"] is natural["run_result"] is None


def test_cache_is_size_bounded_and_symlinked_reference_is_not_read(tmp_path: Path, monkeypatch) -> None:
    run = _run(tmp_path)
    directory = _collection(run)
    outside = tmp_path / "external-state.jsonl"
    outside.write_text('{"private":true}\n')
    reference = directory / "reference.jsonl"
    reference.unlink()
    try:
        reference.symlink_to(outside)
    except OSError:
        pytest.skip("symlink creation unavailable")
    monkeypatch.setattr(monitor, "MAX_CACHE_FILES", 2)
    monkeypatch.setattr(monitor, "MAX_CACHE_BYTES", 4096)
    store = monitor.MacroDashboardStore(run, artifact_root=tmp_path)
    snapshot = store.snapshot()
    assert len(store._cache) <= 2 and store._cache_bytes <= 4096
    assert snapshot["history"][0]["gameplay"]["roots"][0]["natural_reference"]["player_hp_lost_since_reset"] is None
    assert any("tail unavailable" in warning for warning in snapshot["warnings"])
