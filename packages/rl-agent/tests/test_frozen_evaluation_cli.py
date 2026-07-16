from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

from sts2_rl import evaluate_checkpoint as cli
from sts2_rl.training import checkpoint_evaluation as checkpoint_evaluation_module
from sts2_rl.training.checkpoint_evaluation import FrozenEvaluationResult


@dataclass(frozen=True)
class _FakeConfig:
    environment: Any = field(
        default_factory=lambda: SimpleNamespace(sim_exe_path="fake-sim")
    )
    runtime: Any = field(default_factory=lambda: SimpleNamespace(seed=22))

    @staticmethod
    def lineage_mapping() -> dict[str, object]:
        return {
            "version": 10,
            "profile": "paired-cli-test",
            "immutable_semantics": {"candidate_cap": 256},
        }


@dataclass(frozen=True)
class _FakeSimulator:
    executable: Path

    def to_mapping(self) -> dict[str, object]:
        return {"executable": str(self.executable), "sha256": "simulator-sha"}


def test_atomic_directory_publication_retries_transient_permission_error(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    source = tmp_path / "staging"
    destination = tmp_path / "published"
    source.mkdir()
    (source / "complete.txt").write_text("complete\n", encoding="utf-8")
    real_replace = os.replace
    attempts = 0

    def transient_replace(left: str | Path, right: str | Path) -> None:
        nonlocal attempts
        attempts += 1
        if attempts == 1:
            raise PermissionError("injected transient sharing handle")
        real_replace(left, right)

    monkeypatch.setattr(checkpoint_evaluation_module.os, "replace", transient_replace)
    monkeypatch.setattr(checkpoint_evaluation_module.time, "sleep", lambda _delay: None)

    checkpoint_evaluation_module.atomic_publish_directory(
        source,
        destination,
    )

    assert attempts == 2
    assert not source.exists()
    assert (destination / "complete.txt").read_text(encoding="utf-8") == "complete\n"


def test_paired_cli_is_atomic_rerunnable_and_records_resolved_provenance(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("STS2_ARTIFACT_ROOT", str(tmp_path))
    checkpoints = [tmp_path / "checkpoint-one", tmp_path / "checkpoint-two"]
    for checkpoint in checkpoints:
        checkpoint.mkdir()
        (checkpoint / "checkpoint.manifest.json").write_text(
            json.dumps({"checkpoint_id": checkpoint.name}),
            encoding="utf-8",
        )

    simulator = _FakeSimulator(tmp_path / "headless-sim.exe")
    identity_audit = tmp_path / "identity-audit.json"
    mechanics_audit = tmp_path / "mechanics-audit.json"
    identity_audit.write_text("{}\n", encoding="utf-8")
    mechanics_audit.write_text("{}\n", encoding="utf-8")
    monkeypatch.setattr(cli, "checkpoint_training_config", lambda _path: _FakeConfig())
    monkeypatch.setattr(cli, "resolve_headless_sim_exe", lambda _path: simulator.executable)
    monkeypatch.setattr(cli, "verify_headless_simulator", lambda *_args, **_kwargs: simulator)
    monkeypatch.setattr(cli, "write_preflight_audit", lambda _sim: identity_audit)
    monkeypatch.setattr(
        cli,
        "run_runtime_mechanics_preflight",
        lambda _exe: {"contract": "settlement-v1"},
    )
    monkeypatch.setattr(cli, "write_runtime_mechanics_audit", lambda _result: mechanics_audit)
    monkeypatch.setattr(
        cli,
        "validate_resume_checkpoint",
        lambda checkpoint: SimpleNamespace(
            root=Path(checkpoint),
            manifest={"checkpoint_id": Path(checkpoint).name},
        ),
    )

    fail_second_once = True
    evaluator_output_names: list[str] = []

    def fake_evaluate(
        checkpoint: str | Path,
        *,
        output_directory: str | Path,
        episodes: int,
        **_kwargs: object,
    ) -> FrozenEvaluationResult:
        nonlocal fail_second_once
        root = Path(checkpoint)
        evaluator_output_names.append(Path(output_directory).name)
        if root.name == "checkpoint-two" and fail_second_once:
            fail_second_once = False
            raise RuntimeError("injected second-checkpoint infrastructure fault")
        output = Path(output_directory)
        output.mkdir(parents=True)
        journal = output / "trajectory.jsonl"
        audit = output / "evaluation.json"
        journal.write_text("{}\n", encoding="utf-8")
        audit.write_text(
            json.dumps({"evaluation": {"journal": str(journal)}}),
            encoding="utf-8",
        )
        return FrozenEvaluationResult(
            checkpoint=root,
            checkpoint_id=root.name,
            source_environment_steps=100,
            source_policy_version=10,
            output_directory=output,
            audit_path=audit,
            journal_path=journal,
            summary={"episodes": episodes, "mean_max_floor": 3.0},
        )

    monkeypatch.setattr(cli, "evaluate_checkpoint_policy", fake_evaluate)
    argv = [
        "--checkpoint",
        str(checkpoints[0]),
        "--checkpoint",
        str(checkpoints[1]),
        "--episodes",
        "2",
        "--output-root",
        "paired-evaluation",
    ]
    output_root = tmp_path / "paired-evaluation"
    invalid_audit = tmp_path / "paired-evaluation.paired-evaluation.invalid.json"

    with pytest.raises(RuntimeError, match="injected second-checkpoint"):
        cli.main(argv)
    assert not output_root.exists()
    assert invalid_audit.is_file()
    assert not tuple(tmp_path.glob(".paired-evaluation.paired-staging-*"))

    assert cli.main(argv) == 0
    # Pair-level staging deliberately uses compact child names so the nested
    # per-checkpoint staging UUID cannot cross Windows' legacy MAX_PATH limit.
    assert evaluator_output_names == ["c0", "c1", "c0", "c1"]
    assert output_root.is_dir()
    assert not invalid_audit.exists()
    manifest = json.loads(
        (output_root / "paired-evaluation.manifest.json").read_text(encoding="utf-8")
    )
    assert manifest["schema_version"] == "sts2-paired-frozen-evaluation-v2"
    assert manifest["publication_contract"] == "atomic-directory-rename-v1"
    assert manifest["requested_base_seed"] is None
    assert manifest["resolved_base_seed"] == 22
    assert manifest["held_out_seeds"] == [45, 47]
    assert len(manifest["lineage"]["sha256"]) == 64
    assert manifest["simulator_provenance"]["verified_identity"]["sha256"] == "simulator-sha"
    assert len(manifest["simulator_provenance"]["identity_audit_sha256"]) == 64
    assert len(manifest["results"]) == 2
    for item in manifest["results"]:
        published = Path(item["output_directory"])
        assert published.parent == output_root
        assert Path(item["audit_path"]).is_file()
        assert Path(item["journal_path"]).is_file()
        audit = json.loads(Path(item["audit_path"]).read_text(encoding="utf-8"))
        assert Path(audit["evaluation"]["journal"]) == Path(item["journal_path"])
        assert Path(audit["published_output_directory"]) == published
