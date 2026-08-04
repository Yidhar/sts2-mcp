from __future__ import annotations

import json
import os
import threading
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

from sts2_rl import evaluate_checkpoint as cli
from sts2_rl.checkpoints import (
    dependency_lock_metadata as checkpoints_dependency_lock_metadata,
)
from sts2_rl.training import checkpoint_evaluation as checkpoint_evaluation_module
from sts2_rl.training.checkpoint_evaluation import FrozenEvaluationResult


@dataclass(frozen=True)
class _FakeConfig:
    environment: Any = field(default_factory=lambda: SimpleNamespace(sim_exe_path="fake-sim"))
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
        base_seed = int(_kwargs["base_seed"])
        evaluation_seeds = cli.held_out_evaluation_seeds(base_seed, episodes)
        summary = {"episodes": episodes, "mean_max_floor": 3.0}
        journal.write_text("{}\n", encoding="utf-8")
        audit.write_text(
            json.dumps(
                {
                    "evaluation_of": {
                        "checkpoint_id": root.name,
                        "manifest_sha256": cli._sha256(root / "checkpoint.manifest.json"),
                        "training_state": {
                            "environment_steps": 100,
                            "policy_version": 10,
                        },
                    },
                    "policy_source": "network.pt",
                    "learner_updates_performed": 0,
                    "simulator_provenance": _kwargs["simulator_provenance"],
                    "evaluation": {
                        "episodes": episodes,
                        "base_seed": base_seed,
                        "episode_metrics": [{"reset_seed": seed} for seed in evaluation_seeds],
                        "summary": summary,
                        "journal": str(journal),
                    },
                }
            ),
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
            summary=summary,
        )

    monkeypatch.setattr(cli, "evaluate_checkpoint_policy", fake_evaluate)
    monkeypatch.setattr(
        cli,
        "_create_checkpoint_executor",
        lambda workers: ThreadPoolExecutor(max_workers=workers),
    )
    argv = [
        "--checkpoint",
        str(checkpoints[0]),
        "--checkpoint",
        str(checkpoints[1]),
        "--episodes",
        "2",
        "--checkpoint-workers",
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
    durable_children = tuple((tmp_path / ".paired-checkpoint-cache-v1").glob("child-*"))
    assert len(durable_children) == 1
    durable_manifest = json.loads((durable_children[0] / "durable-child.manifest.json").read_text(encoding="utf-8"))
    assert durable_manifest["status"] == "complete"
    assert durable_manifest["contract"]["held_out_seeds"] == [45, 47]

    assert cli.main(argv) == 0
    # The first completed checkpoint survived the failed pair and was reused;
    # only the failed second checkpoint needed another expensive evaluation.
    assert len(evaluator_output_names) == 3
    assert all(name.startswith(".s-") for name in evaluator_output_names)
    assert output_root.is_dir()
    assert not invalid_audit.exists()
    manifest = json.loads((output_root / "paired-evaluation.manifest.json").read_text(encoding="utf-8"))
    assert manifest["schema_version"] == "sts2-paired-frozen-evaluation-v2"
    assert manifest["publication_contract"] == "atomic-directory-rename-v1"
    assert manifest["requested_base_seed"] is None
    assert manifest["resolved_base_seed"] == 22
    assert manifest["checkpoint_workers"] == 2
    assert manifest["held_out_seeds"] == [45, 47]
    assert len(manifest["lineage"]["sha256"]) == 64
    assert manifest["simulator_provenance"]["verified_identity"]["sha256"] == "simulator-sha"
    assert len(manifest["simulator_provenance"]["identity_audit_sha256"]) == 64
    assert manifest["durable_child_cache"]["schema_version"] == ("sts2-durable-frozen-evaluation-child-v1")
    assert len(manifest["results"]) == 2
    assert [item["durable_cache"]["cache_hit"] for item in manifest["results"]] == [
        True,
        False,
    ]
    for item in manifest["results"]:
        published = Path(item["output_directory"])
        assert published.parent == output_root
        assert Path(item["audit_path"]).is_file()
        assert Path(item["journal_path"]).is_file()
        audit = json.loads(Path(item["audit_path"]).read_text(encoding="utf-8"))
        assert Path(audit["evaluation"]["journal"]) == Path(item["journal_path"])
        assert Path(audit["published_output_directory"]) == published
    cached_audit = json.loads((durable_children[0] / "evaluation.json").read_text(encoding="utf-8"))
    assert Path(cached_audit["published_output_directory"]) == durable_children[0]
    # Reusing a durable child must copy, rather than hard-link, the formal
    # journal.  In-place edits on either side may never cross-contaminate the
    # other evidence set.
    cached_journal = durable_children[0] / "trajectory.jsonl"
    published_journal = Path(manifest["results"][0]["journal_path"])
    original_cache_bytes = cached_journal.read_bytes()
    published_journal.write_bytes(published_journal.read_bytes() + b'{"pair_edit":true}\n')
    published_after_edit = published_journal.read_bytes()
    assert cached_journal.read_bytes() == original_cache_bytes
    cached_journal.write_bytes(original_cache_bytes + b'{"cache_edit":true}\n')
    assert published_journal.read_bytes() == published_after_edit
    cached_journal.write_bytes(original_cache_bytes)
    cli._validate_durable_child(
        durable_children[0],
        expected_contract=durable_manifest["contract"],
    )


def test_two_worker_scheduler_runs_independent_children_concurrently(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    barrier = threading.Barrier(2)
    seen: list[str] = []
    executor_sizes: list[int] = []

    def fake_worker(request: dict[str, object]) -> dict[str, object]:
        checkpoint_id = str(request["checkpoint_id"])
        seen.append(checkpoint_id)
        barrier.wait(timeout=2.0)
        return {
            "status": "complete",
            "checkpoint_id": checkpoint_id,
            "cache_output": f"cache/{checkpoint_id}",
            "cache_hit": False,
        }

    def executor_factory(workers: int) -> ThreadPoolExecutor:
        executor_sizes.append(workers)
        return ThreadPoolExecutor(max_workers=workers)

    monkeypatch.setattr(cli, "_evaluate_durable_child_request", fake_worker)
    monkeypatch.setattr(cli, "_create_checkpoint_executor", executor_factory)
    results = cli._execute_durable_child_requests(
        [
            {"checkpoint_id": "checkpoint-one"},
            {"checkpoint_id": "checkpoint-two"},
        ],
        checkpoint_workers=2,
    )

    assert executor_sizes == [2]
    assert set(seen) == {"checkpoint-one", "checkpoint-two"}
    assert {str(item["checkpoint_id"]) for item in results} == set(seen)
    assert all(item["status"] == "complete" for item in results)


def test_durable_child_validation_fails_closed_on_file_corruption(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    root = tmp_path / "child"
    root.mkdir()
    summary = {"episodes": 2}
    simulator_semantics = {
        "verified_identity": {"sha256": "simulator-sha"},
        "runtime_mechanics": {"contract": "settlement-v1"},
    }
    (root / "evaluation.json").write_text(
        json.dumps(
            {
                "evaluation_of": {
                    "checkpoint_id": "checkpoint-one",
                    "manifest_sha256": "manifest-sha",
                    "training_state": {
                        "environment_steps": 100,
                        "policy_version": 10,
                    },
                },
                "policy_source": "network.pt",
                "learner_updates_performed": 0,
                "simulator_provenance": simulator_semantics,
                "evaluation": {
                    "episodes": 2,
                    "base_seed": 22,
                    "episode_metrics": [
                        {"reset_seed": 45},
                        {"reset_seed": 47},
                    ],
                    "summary": summary,
                },
            }
        ),
        encoding="utf-8",
    )
    (root / "trajectory.jsonl").write_text("{}\n", encoding="utf-8")
    contract: dict[str, object] = {
        "checkpoint_id": "checkpoint-one",
        "checkpoint_manifest_sha256": "manifest-sha",
        "held_out_seeds": [45, 47],
        "episodes": 2,
        "resolved_base_seed": 22,
        "simulator_semantics": simulator_semantics,
    }
    manifest = {
        "schema_version": "sts2-durable-frozen-evaluation-child-v1",
        "status": "complete",
        "cache_key": cli._mapping_fingerprint(contract),
        "contract": contract,
        "result": {
            "checkpoint_id": "checkpoint-one",
            "source_environment_steps": 100,
            "source_policy_version": 10,
            "summary": summary,
        },
        "files": [
            cli._durable_file_record(root, "evaluation.json"),
            cli._durable_file_record(root, "trajectory.jsonl"),
        ],
    }
    (root / "durable-child.manifest.json").write_text(
        json.dumps(manifest),
        encoding="utf-8",
    )
    cli._validate_durable_child(root, expected_contract=contract)

    # A child can sit ready for hours while its paired peer runs.  The formal
    # clone must validate again immediately before copying, not trust that old
    # validation result.
    original_journal = (root / "trajectory.jsonl").read_bytes()
    (root / "trajectory.jsonl").write_text('{"changed_while_waiting":true}\n', encoding="utf-8")
    stale_clone = tmp_path / "stale-clone"
    with pytest.raises(ValueError, match="hash mismatch"):
        cli._validated_clone_durable_child(
            root,
            stale_clone,
            expected_contract=contract,
        )
    assert not stale_clone.exists()
    (root / "trajectory.jsonl").write_bytes(original_journal)

    # Also close the smaller validation-to-copy race: each copied payload must
    # still equal the file record from the immediately preceding validation.
    real_copy2 = cli.shutil.copy2

    def tampering_copy2(source: str | Path, destination: str | Path) -> str:
        result = real_copy2(source, destination)
        if Path(source).name == "trajectory.jsonl":
            Path(destination).write_text('{"changed_during_copy":true}\n', encoding="utf-8")
        return str(result)

    monkeypatch.setattr(cli.shutil, "copy2", tampering_copy2)
    raced_clone = tmp_path / "raced-clone"
    with pytest.raises(ValueError, match="changed while copying trajectory.jsonl"):
        cli._validated_clone_durable_child(
            root,
            raced_clone,
            expected_contract=contract,
        )
    assert not raced_clone.exists()
    monkeypatch.setattr(cli.shutil, "copy2", real_copy2)

    (root / "unexpected.tmp").write_text("unexpected\n", encoding="utf-8")
    with pytest.raises(ValueError, match="on-disk file set"):
        cli._validate_durable_child(root, expected_contract=contract)
    (root / "unexpected.tmp").unlink()

    (root / "unexpected-directory").mkdir()
    with pytest.raises(ValueError, match="on-disk file set"):
        cli._validate_durable_child(root, expected_contract=contract)
    (root / "unexpected-directory").rmdir()

    (root / "trajectory.jsonl").write_text('{"corrupt":true}\n', encoding="utf-8")
    with pytest.raises(ValueError, match="hash mismatch"):
        cli._validate_durable_child(root, expected_contract=contract)


def test_durable_child_validation_rejects_symlink_payload(
    tmp_path: Path,
) -> None:
    root = tmp_path / "child"
    root.mkdir()
    external = tmp_path / "external-evaluation.json"
    external.write_text("{}\n", encoding="utf-8")
    try:
        (root / "evaluation.json").symlink_to(external)
    except OSError as exc:
        pytest.skip(f"test environment cannot create file symlinks: {exc}")
    (root / "trajectory.jsonl").write_text("{}\n", encoding="utf-8")
    (root / "durable-child.manifest.json").write_text("{}\n", encoding="utf-8")

    with pytest.raises(ValueError, match="not a regular file"):
        cli._validate_durable_child(root, expected_contract={})


def test_worker_rejects_source_change_after_evaluation(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    runtime_environment = {"runtime": "stable"}
    fingerprints = iter(("implementation-before", "implementation-after"))
    monkeypatch.setattr(
        cli,
        "_evaluation_implementation_fingerprint",
        lambda: next(fingerprints),
    )
    monkeypatch.setattr(cli, "_runtime_environment_contract", lambda: runtime_environment)

    def fake_evaluate(
        checkpoint: str | Path,
        *,
        output_directory: str | Path,
        **_kwargs: object,
    ) -> FrozenEvaluationResult:
        output = Path(output_directory)
        output.mkdir(parents=True)
        audit = output / "evaluation.json"
        journal = output / "trajectory.jsonl"
        audit.write_text("{}\n", encoding="utf-8")
        journal.write_text("{}\n", encoding="utf-8")
        return FrozenEvaluationResult(
            checkpoint=Path(checkpoint),
            checkpoint_id="checkpoint-one",
            source_environment_steps=100,
            source_policy_version=10,
            output_directory=output,
            audit_path=audit,
            journal_path=journal,
            summary={"episodes": 1},
        )

    monkeypatch.setattr(cli, "evaluate_checkpoint_policy", fake_evaluate)
    cache_output = tmp_path / "cache" / "child-key"
    result = cli._evaluate_durable_child_request(
        {
            "checkpoint": str(tmp_path / "checkpoint-one"),
            "checkpoint_id": "checkpoint-one",
            "cache_output": str(cache_output),
            "contract": {
                "checkpoint_id": "checkpoint-one",
                "evaluation_implementation_sha256": "implementation-before",
                "runtime_environment": runtime_environment,
            },
            "episodes": 1,
            "resolved_base_seed": 22,
            "device": "cpu",
            "collector_device": "cpu",
            "sim_exe_path": "fake-sim",
            "simulator_provenance": {},
        }
    )

    assert result["status"] == "failed"
    assert "changed while checkpoint evaluation was running" in str(result["error"])
    assert not cache_output.exists()
    assert not tuple(cache_output.parent.glob(".s-*"))


def test_worker_rejects_parent_runtime_contract_mismatch(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(cli, "_evaluation_implementation_fingerprint", lambda: "implementation")
    monkeypatch.setattr(cli, "_runtime_environment_contract", lambda: {"runtime": "worker"})

    def should_not_evaluate(*_args: object, **_kwargs: object) -> FrozenEvaluationResult:
        raise AssertionError("runtime mismatch must fail before evaluation")

    monkeypatch.setattr(cli, "evaluate_checkpoint_policy", should_not_evaluate)
    cache_output = tmp_path / "cache" / "child-key"
    result = cli._evaluate_durable_child_request(
        {
            "checkpoint": str(tmp_path / "checkpoint-one"),
            "checkpoint_id": "checkpoint-one",
            "cache_output": str(cache_output),
            "contract": {
                "checkpoint_id": "checkpoint-one",
                "evaluation_implementation_sha256": "implementation",
                "runtime_environment": {"runtime": "parent"},
            },
            "episodes": 1,
            "resolved_base_seed": 22,
            "device": "cpu",
            "collector_device": "cpu",
            "sim_exe_path": "fake-sim",
            "simulator_provenance": {},
        }
    )

    assert result["status"] == "failed"
    assert "differs from the parent cache contract" in str(result["error"])
    assert not cache_output.exists()


def test_worker_rejects_runtime_change_after_evaluation(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    runtime_environments = iter(({"runtime": "stable"}, {"runtime": "changed"}))
    monkeypatch.setattr(cli, "_evaluation_implementation_fingerprint", lambda: "implementation")
    monkeypatch.setattr(cli, "_runtime_environment_contract", lambda: next(runtime_environments))

    def fake_evaluate(
        _checkpoint: str | Path,
        *,
        output_directory: str | Path,
        **_kwargs: object,
    ) -> object:
        Path(output_directory).mkdir(parents=True)
        return object()

    monkeypatch.setattr(cli, "evaluate_checkpoint_policy", fake_evaluate)
    cache_output = tmp_path / "cache" / "child-key"
    result = cli._evaluate_durable_child_request(
        {
            "checkpoint": str(tmp_path / "checkpoint-one"),
            "checkpoint_id": "checkpoint-one",
            "cache_output": str(cache_output),
            "contract": {
                "checkpoint_id": "checkpoint-one",
                "evaluation_implementation_sha256": "implementation",
                "runtime_environment": {"runtime": "stable"},
            },
            "episodes": 1,
            "resolved_base_seed": 22,
            "device": "cpu",
            "collector_device": "cpu",
            "sim_exe_path": "fake-sim",
            "simulator_provenance": {},
        }
    )

    assert result["status"] == "failed"
    assert "runtime environment changed while evaluation was running" in str(result["error"])
    assert not cache_output.exists()
    assert not tuple(cache_output.parent.glob(".s-*"))


def test_evaluation_roots_must_be_disjoint(tmp_path: Path) -> None:
    output = tmp_path / "paired"
    cache = tmp_path / "cache"
    cli._require_disjoint_evaluation_roots(output, cache)

    with pytest.raises(ValueError, match="must be disjoint"):
        cli._require_disjoint_evaluation_roots(output, output)
    with pytest.raises(ValueError, match="must be disjoint"):
        cli._require_disjoint_evaluation_roots(output, output / "cache")
    with pytest.raises(ValueError, match="must be disjoint"):
        cli._require_disjoint_evaluation_roots(output / "nested", output)


def test_runtime_environment_contract_binds_dependency_locks() -> None:
    contract = cli._runtime_environment_contract()

    assert contract["python_version"]
    assert contract["python_implementation"]
    assert contract["platform"]
    assert contract["platform_machine"]
    assert contract["numpy_version"]
    assert isinstance(contract["torch_num_threads"], int)
    assert isinstance(contract["torch_num_interop_threads"], int)
    locks = contract["dependency_locks"]
    assert locks == checkpoints_dependency_lock_metadata()
    assert isinstance(locks, list)
    assert {Path(str(record["path"])).name for record in locks} == {
        "requirements.lock",
        "requirements-dev.lock",
        "requirements-wsl-rocm.txt",
    }
    assert all(
        isinstance(record, dict) and isinstance(record.get("size_bytes"), int) and len(str(record.get("sha256"))) == 64
        for record in locks
    )
