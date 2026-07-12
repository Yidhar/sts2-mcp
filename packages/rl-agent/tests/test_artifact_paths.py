from __future__ import annotations

import ast
from pathlib import Path
from types import SimpleNamespace

import pytest

from sts2_env.headless_sim_bridge_client import HeadlessSimBridgeClient
from sts2_rl.artifacts import (
    ARTIFACT_ROOT_ENV,
    artifact_root,
    resolve_artifact_path,
    resolve_external_input_path,
)


def test_artifact_root_defaults_below_home_without_using_checkout(tmp_path: Path) -> None:
    checkout = tmp_path / "checkout"
    root = artifact_root(environ={}, home=tmp_path / "home", source_root=checkout)

    assert root == tmp_path / "home" / ".sts2-artifacts"
    assert not root.is_relative_to(checkout)


def test_artifact_root_accepts_absolute_override_and_rejects_cwd_relative_override(tmp_path: Path) -> None:
    configured = tmp_path / "runtime-artifacts"

    assert artifact_root(environ={ARTIFACT_ROOT_ENV: str(configured)}) == configured
    with pytest.raises(ValueError, match="must be an absolute path"):
        artifact_root(environ={ARTIFACT_ROOT_ENV: "runtime-artifacts"})


def test_relative_artifact_paths_use_root_and_cannot_escape(tmp_path: Path) -> None:
    root = tmp_path / "artifacts"
    source = tmp_path / "checkout"

    assert resolve_artifact_path("runs/one", root=root, source_root=source) == root / "runs" / "one"
    assert resolve_artifact_path(None, default="checkpoints", root=root, source_root=source) == root / "checkpoints"
    absolute_inside_root = root / "explicit" / "run"
    assert resolve_artifact_path(absolute_inside_root, root=root, source_root=source) == absolute_inside_root
    with pytest.raises(ValueError, match="must stay below the artifact root"):
        resolve_artifact_path(tmp_path / "other-output", root=root, source_root=source)
    with pytest.raises(ValueError, match="must stay below"):
        resolve_artifact_path("../checkout/logs", root=root, source_root=source)


def test_artifact_root_and_output_reject_paths_inside_source_checkout(tmp_path: Path) -> None:
    source = tmp_path / "checkout"
    root = tmp_path / "artifacts"

    with pytest.raises(ValueError, match="disjoint from the source checkout"):
        artifact_root(environ={ARTIFACT_ROOT_ENV: str(source / "runtime")}, source_root=source)
    with pytest.raises(ValueError, match="disjoint from the source checkout"):
        resolve_artifact_path(source / "logs", root=root, source_root=source)
    with pytest.raises(ValueError, match="disjoint from the source checkout"):
        artifact_root(environ={ARTIFACT_ROOT_ENV: str(tmp_path)}, source_root=source)


def test_read_only_input_allows_explicit_external_path_but_never_source_checkout(tmp_path: Path) -> None:
    source = tmp_path / "checkout"
    root = tmp_path / "artifacts"
    external = tmp_path / "imported-checkpoints" / "step-1"

    assert resolve_external_input_path(external, root=root, source_root=source) == external
    assert resolve_external_input_path("checkpoints/step-2", root=root, source_root=source) == (
        root / "checkpoints" / "step-2"
    )
    with pytest.raises(ValueError, match="disjoint from the source checkout"):
        resolve_external_input_path(source / "old-checkpoint", root=root, source_root=source)


def test_headless_hang_log_defaults_to_artifact_root(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    artifacts = tmp_path / "external-artifacts"
    monkeypatch.setenv(ARTIFACT_ROOT_ENV, str(artifacts))
    monkeypatch.delenv("STS2_SIM_HANG_LOG_DIR", raising=False)
    client = HeadlessSimBridgeClient.__new__(HeadlessSimBridgeClient)
    client._proc = SimpleNamespace(pid=12345)
    client._startup_stderr_tail = []
    client._hang_log_path = None
    client._hang_log_handle = None

    client._open_hang_log()
    try:
        assert client._hang_log_path == artifacts / "logs" / "headless-sim" / "sim_hang_debug_pid12345.log"
        assert client._hang_log_path.is_file()
    finally:
        if client._hang_log_handle is not None:
            client._hang_log_handle.close()


def test_cli_path_options_cannot_bypass_declared_source_or_artifact_boundaries() -> None:
    package_root = Path(__file__).resolve().parents[1]
    output_options = {
        "--out",
        "--output",
        "--output-dir",
        "--output-root",
        "--output-csv",
        "--report",
        "--report-dir",
        "--dump-samples",
        "--log-dir",
        "--checkpoint-dir",
        "--state-file",
    }
    input_options = {
        "--input",
        "--dataset-root",
        "--runs-summary",
        "--combat-snapshots",
        "--snapshot-pool",
        "--snapshot-dataset",
        "--checkpoint",
        "--checkpoints",
        "--session-file",
        "--auth-session-file",
        "--replay-buffer",
        "--network-state",
    }
    source_paths = set(package_root.glob("*.py"))
    for directory in ("sts2_rl", "sts2_baseline", "sts2_env", "scripts", "tools"):
        source_paths.update((package_root / directory).rglob("*.py"))
    violations: list[str] = []
    for path in sorted(source_paths):
        relative = path.relative_to(package_root).as_posix()
        source = path.read_text(encoding="utf-8-sig")
        tree = ast.parse(source, filename=str(path))
        declared: set[str] = set()
        for node in ast.walk(tree):
            if not (
                isinstance(node, ast.Call)
                and isinstance(node.func, ast.Attribute)
                and node.func.attr == "add_argument"
            ):
                continue
            declared.update(
                argument.value
                for argument in node.args
                if isinstance(argument, ast.Constant)
                and isinstance(argument.value, str)
                and argument.value.startswith("--")
            )
        if declared & output_options and not (
            "resolve_artifact_path" in source
            or "resolve_generated_game_data_output" in source
            or "PATH_BOUNDARY_DELEGATED" in source
        ):
            violations.append(f"{relative}: mutable output path has no declared boundary")
        if declared & input_options and not (
            "resolve_external_input_path" in source
            or "resolve_artifact_path" in source
            or "resolve_generated_game_data_output" in source
            or "PATH_BOUNDARY_DELEGATED" in source
        ):
            violations.append(f"{relative}: runtime input path has no declared boundary")
    assert not violations, "\n".join(violations)
