from __future__ import annotations

import json
from pathlib import Path

import pytest

from sts2_rl.simulator_identity import (
    IDENTITY_SCHEMA_VERSION,
    SimulatorIdentityError,
    sha256_file,
    simulator_identity_path,
    verify_headless_simulator,
    write_preflight_audit,
)
from sts2_rl.training.checkpointing import TrainingState
from sts2_rl.training.config import TrainingConfig


def _lock(tmp_path: Path) -> tuple[Path, dict[str, str]]:
    payload = {
        "url": "https://example.invalid/sts2-ai.git",
        "commit": "1" * 40,
        "tree": "2" * 40,
        "canonical_headless_project": "STS2AI/ENV/Sim/HeadlessSim/HeadlessSim.csproj",
    }
    path = tmp_path / "sts2-ai.lock.json"
    path.write_text(json.dumps(payload), encoding="utf-8")
    return path, payload


def _simulator(tmp_path: Path) -> tuple[Path, Path, Path, dict[str, object]]:
    lock_path, lock = _lock(tmp_path)
    executable = tmp_path / "HeadlessSim.exe"
    executable.write_bytes(b"pinned-headless-sim")
    managed_assembly = tmp_path / "HeadlessSim.dll"
    managed_assembly.write_bytes(b"pinned-headless-sim-managed-code")
    payload: dict[str, object] = {
        "schema_version": IDENTITY_SCHEMA_VERSION,
        "component": "HeadlessSim",
        "source": {
            "url": lock["url"],
            "commit": lock["commit"],
            "tree": lock["tree"],
            "project": lock["canonical_headless_project"],
            "patches": lock.get("patches", []),
        },
        "build": {
            "configuration": "Release",
            "target_framework": "net9.0",
            "dotnet_sdk": "9.0.308",
        },
        "binary": {
            "file_name": executable.name,
            "size_bytes": executable.stat().st_size,
            "sha256": sha256_file(executable),
        },
        "managed_binary": {
            "file_name": managed_assembly.name,
            "size_bytes": managed_assembly.stat().st_size,
            "sha256": sha256_file(managed_assembly),
        },
    }
    identity_path = simulator_identity_path(executable)
    identity_path.write_text(json.dumps(payload), encoding="utf-8")
    return executable, identity_path, lock_path, payload


def _simulator_for_repository_lock(tmp_path: Path) -> Path:
    from sts2_rl.simulator_identity import load_sts2_ai_lock

    lock = load_sts2_ai_lock()
    executable = tmp_path / "HeadlessSim.exe"
    executable.write_bytes(b"repository-pinned-test-simulator")
    managed_assembly = tmp_path / "HeadlessSim.dll"
    managed_assembly.write_bytes(b"repository-pinned-test-managed-code")
    payload = {
        "schema_version": IDENTITY_SCHEMA_VERSION,
        "component": "HeadlessSim",
        "source": {
            "url": lock["url"],
            "commit": lock["commit"],
            "tree": lock["tree"],
            "project": lock["canonical_headless_project"],
            "patches": lock.get("patches", []),
        },
        "build": {
            "configuration": "Release",
            "target_framework": "net9.0",
            "dotnet_sdk": "test-sdk",
        },
        "binary": {
            "file_name": executable.name,
            "size_bytes": executable.stat().st_size,
            "sha256": sha256_file(executable),
        },
        "managed_binary": {
            "file_name": managed_assembly.name,
            "size_bytes": managed_assembly.stat().st_size,
            "sha256": sha256_file(managed_assembly),
        },
    }
    simulator_identity_path(executable).write_text(json.dumps(payload), encoding="utf-8")
    return executable


def test_verifies_lock_and_exact_binary_bytes(tmp_path: Path) -> None:
    executable, identity_path, lock_path, _ = _simulator(tmp_path)

    identity = verify_headless_simulator(
        executable,
        identity_path=identity_path,
        lock_path=lock_path,
    )

    assert identity.executable == executable.resolve()
    assert identity.source_commit == "1" * 40
    assert identity.binary_sha256 == sha256_file(executable)
    assert identity.managed_assembly_sha256 == sha256_file(
        executable.with_suffix(".dll")
    )
    assert identity.build_configuration == "Release"


def test_missing_sidecar_is_refused_without_path_or_version_fallback(tmp_path: Path) -> None:
    executable = tmp_path / "HeadlessSim.exe"
    executable.write_bytes(b"fake")
    executable.with_suffix(".dll").write_bytes(b"fake-managed-code")

    with pytest.raises(SimulatorIdentityError, match="sidecar is missing.*unverified binaries are refused"):
        verify_headless_simulator(executable, lock_path=_lock(tmp_path)[0])


@pytest.mark.parametrize(
    "mutation",
    ["source", "binary", "managed_binary", "configuration"],
)
def test_wrong_source_binary_or_build_configuration_is_refused(
    tmp_path: Path,
    mutation: str,
) -> None:
    executable, identity_path, lock_path, payload = _simulator(tmp_path)
    if mutation == "source":
        source = payload["source"]
        assert isinstance(source, dict)
        source["commit"] = "f" * 40
        identity_path.write_text(json.dumps(payload), encoding="utf-8")
        expected = "source commit mismatch"
    elif mutation == "binary":
        executable.write_bytes(b"different-binary")
        expected = "size mismatch|SHA-256 mismatch"
    elif mutation == "managed_binary":
        executable.with_suffix(".dll").write_bytes(b"different-managed-binary")
        expected = "managed assembly size mismatch|managed assembly SHA-256 mismatch"
    else:
        build = payload["build"]
        assert isinstance(build, dict)
        build["configuration"] = "Debug"
        identity_path.write_text(json.dumps(payload), encoding="utf-8")
        expected = "configuration must be 'Release'"

    with pytest.raises(SimulatorIdentityError, match=expected):
        verify_headless_simulator(
            executable,
            identity_path=identity_path,
            lock_path=lock_path,
        )


def test_verified_identity_can_be_persisted_as_external_audit(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    executable, identity_path, lock_path, _ = _simulator(tmp_path)
    artifact_root = tmp_path / "artifacts"
    monkeypatch.setenv("STS2_ARTIFACT_ROOT", str(artifact_root))
    identity = verify_headless_simulator(
        executable,
        identity_path=identity_path,
        lock_path=lock_path,
    )

    audit = write_preflight_audit(identity)

    payload = json.loads(audit.read_text(encoding="utf-8"))
    assert audit.parent == artifact_root / "logs" / "simulator-preflight"
    assert payload["event"] == "simulator_identity_verified"
    assert payload["binary"]["sha256"] == identity.binary_sha256
    assert (
        payload["managed_binary"]["sha256"]
        == identity.managed_assembly_sha256
    )


def test_training_cli_gates_and_pins_headless_executable_before_run(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    from sts2_rl import train as train_module

    executable = _simulator_for_repository_lock(tmp_path)
    artifact_root = tmp_path / "artifacts"
    monkeypatch.setenv("STS2_ARTIFACT_ROOT", str(artifact_root))
    received: list[TrainingConfig] = []

    def _run(config: TrainingConfig, **_kwargs: object) -> TrainingState:
        received.append(config)
        return TrainingState()

    monkeypatch.setattr(train_module, "run_training", _run)
    monkeypatch.setattr(
        train_module,
        "run_runtime_mechanics_preflight",
        lambda _executable: {
            "schema": "sts2-runtime-mechanics-audit-v1",
            "runtime_event_checked": True,
            "runtime_combat_checked": True,
        },
    )

    assert train_module.main(["--backend", "headless", "--sim-exe", str(executable), "--steps", "1"]) == 0

    assert len(received) == 1
    config = received[0]
    assert config.environment.sim_exe_path == str(executable.resolve())
    events = [json.loads(line) for line in capsys.readouterr().out.splitlines()]
    assert events[0]["event"] == "simulator_identity_verified"
    assert Path(events[0]["audit_path"]).is_file()
    assert events[1]["event"] == "runtime_mechanics_verified"
    assert Path(events[1]["audit_path"]).is_file()
    assert events[-1]["status"] == "complete"
