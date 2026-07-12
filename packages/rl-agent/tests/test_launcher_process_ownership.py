from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import launcher
from sts2_rl.process_identity import (
    ProcessIdentity,
    TerminationResult,
    capture_process_identity,
    same_process,
    terminate_verified_process,
)


class _FakeProcess:
    def __init__(self, pid: int) -> None:
        self.pid = pid
        self.returncode = None

    def poll(self):
        return None

    def wait(self, timeout=None):
        self.returncode = 0
        return 0


class _Response:
    status_code = 200

    def __init__(self, pid: int, session_id: str) -> None:
        self._pid = pid
        self._session_id = session_id

    def json(self):
        return {"process_id": self._pid, "session_id": self._session_id}


def _identity(pid: int = 123) -> ProcessIdentity:
    return ProcessIdentity(
        pid=pid,
        creation_marker="test-start:1",
        executable=str(Path(sys.executable).resolve()),
        started_at_unix_ns=None,
    )


def _record(tmp_path: Path, pid: int = 123) -> launcher.LauncherProcessRecord:
    return launcher.LauncherProcessRecord(
        identity=_identity(pid),
        instance_id=0,
        session_path=str(tmp_path / "session_0.json"),
        launcher_run_id="run",
        launched_at_unix_ns=1,
    )


def test_registry_round_trip_has_complete_process_identity(tmp_path, monkeypatch):
    state = tmp_path / "owned.json"
    monkeypatch.setenv("STS2_LAUNCHER_STATE_FILE", str(state))
    record = _record(tmp_path)

    launcher._write_registry([record])

    loaded = launcher._read_registry()
    assert loaded == [record]
    payload = json.loads(state.read_text(encoding="utf-8"))
    identity = payload["processes"][0]["identity"]
    assert identity["pid"] == 123
    assert identity["creation_marker"] == "test-start:1"
    assert identity["executable"]


def test_kill_from_registry_refuses_reused_pid_and_drops_stale_authority(
    tmp_path, monkeypatch, capsys
):
    monkeypatch.setenv("STS2_LAUNCHER_STATE_FILE", str(tmp_path / "owned.json"))
    launcher._write_registry([_record(tmp_path)])
    monkeypatch.setattr(
        launcher,
        "_terminate_record",
        lambda _record: TerminationResult(False, "identity_mismatch"),
    )

    assert launcher.kill_all() is False

    assert launcher._read_registry() == []
    assert "REFUSED" in capsys.readouterr().out


def test_owned_session_requires_pid_loopback_and_training_capability(tmp_path, monkeypatch):
    record = _record(tmp_path)
    process = _FakeProcess(record.identity.pid)
    instance = launcher.GameInstance(0, process, Path(record.session_path), record)
    monkeypatch.setattr(launcher, "capture_process_identity", lambda _pid: record.identity)

    base = {
        "pid": record.identity.pid,
        "session_id": "session",
        "base_url": "http://127.0.0.1:27100/",
        "capability_tokens": {"training": "scoped-token"},
    }
    Path(record.session_path).write_text(json.dumps(base), encoding="utf-8")
    assert launcher._read_owned_session(instance) == (
        "http://127.0.0.1:27100",
        "scoped-token",
        "session",
    )

    Path(record.session_path).write_text(
        json.dumps({**base, "pid": record.identity.pid + 1}),
        encoding="utf-8",
    )
    assert launcher._read_owned_session(instance) is None

    Path(record.session_path).write_text(
        json.dumps({**base, "base_url": "http://192.168.2.1:27100/"}),
        encoding="utf-8",
    )
    assert launcher._read_owned_session(instance) is None

    Path(record.session_path).write_text(
        json.dumps({**base, "capability_tokens": {"player-control": "wrong-scope"}}),
        encoding="utf-8",
    )
    assert launcher._read_owned_session(instance) is None


def test_wait_for_ready_uses_scoped_v2_health(tmp_path, monkeypatch):
    record = _record(tmp_path)
    instance = launcher.GameInstance(
        0,
        _FakeProcess(record.identity.pid),
        Path(record.session_path),
        record,
    )
    Path(record.session_path).write_text(
        json.dumps(
            {
                "pid": record.identity.pid,
                "session_id": "session",
                "base_url": "http://127.0.0.1:27100/",
                "capability_tokens": {"training": "scoped-token"},
            }
        ),
        encoding="utf-8",
    )
    monkeypatch.setattr(launcher, "capture_process_identity", lambda _pid: record.identity)
    monkeypatch.setattr(launcher, "_upsert_record", lambda _record: None)
    observed = {}

    def fake_get(url, *, headers, timeout):
        observed.update(url=url, headers=headers, timeout=timeout)
        return _Response(record.identity.pid, "session")

    monkeypatch.setattr(launcher.requests, "get", fake_get)

    assert launcher.wait_for_ready(instance, timeout_s=1.0) is True
    assert observed["url"] == "http://127.0.0.1:27100/v2/health"
    assert observed["headers"] == {"Authorization": "Bearer scoped-token"}


def test_process_identity_mismatch_is_fail_closed():
    actual = capture_process_identity(subprocess.os.getpid())
    wrong = ProcessIdentity(
        pid=actual.pid,
        creation_marker=actual.creation_marker + "-reused",
        executable=actual.executable,
        started_at_unix_ns=actual.started_at_unix_ns,
    )

    result = terminate_verified_process(wrong, timeout_s=0.01)

    assert result.terminated is False
    assert result.reason == "identity_mismatch"
    assert same_process(actual, capture_process_identity(actual.pid))


def test_launcher_contains_no_global_image_name_kill():
    source = Path(launcher.__file__).read_text(encoding="utf-8").casefold()
    assert "taskkill" not in source
    assert " /im " not in source
    assert "os.system(" not in source
