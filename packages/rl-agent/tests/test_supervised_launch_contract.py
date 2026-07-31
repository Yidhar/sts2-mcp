from __future__ import annotations

import hashlib
import json
from pathlib import Path
from uuid import uuid4

import pytest

from sts2_rl.checkpoints import ValidatedResumeCheckpoint
from sts2_rl.training.launch_contract import (
    EXTERNAL_SEAL_VERSION,
    FORMAL_REPORT_GENERATION_SOURCE_VERSION,
    RUNTIME_READINESS_REPORT_VERSION,
    SOURCE_AUTHORITY_VERSION,
    SUPERVISED_LAUNCH_CONTRACT_VERSION,
    SupervisedLaunchContractError,
    checkpoint_source_identity,
    load_runtime_readiness_evidence,
    load_supervised_launch_contract,
    trainer_environment_sha256,
    validate_supervised_model_initialization_binding,
    validate_supervised_runtime_readiness,
    validate_supervised_source_authority,
    validate_supervised_trainer_environment,
)

SHA_A = "a" * 64
SHA_B = "b" * 64
SHA_C = "c" * 64
SHA_D = "d" * 64
COMMIT_A = "1" * 40
TREE_A = "2" * 40
COMMIT_B = "3" * 40
TREE_B = "4" * 40


def _trainer_environment() -> dict[str, str]:
    return {
        "STS2_ARTIFACT_ROOT": "/tmp/artifacts",
        "PYTHONPATH": "/tmp/source",
        "PYTHONNOUSERSITE": "1",
        "PYTHONUNBUFFERED": "1",
        "OMP_NUM_THREADS": "4",
        "MKL_NUM_THREADS": "4",
        "OPENBLAS_NUM_THREADS": "4",
        "NUMEXPR_NUM_THREADS": "4",
        "LC_CTYPE": "C.UTF-8",
        "PATH": "/tmp/venv/bin:/usr/bin:/bin",
    }


def _checkpoint(tmp_path: Path) -> ValidatedResumeCheckpoint:
    root = tmp_path / "checkpoint"
    root.mkdir()
    checkpoint_id = str(uuid4())
    manifest = {
        "checkpoint_id": checkpoint_id,
        "files": [],
        "format": "sts2-atomic-checkpoint-v1",
    }
    metadata = {
        "checkpoint_id": checkpoint_id,
        "training_state": {
            "environment_steps": 100_000,
            "policy_version": 1_569,
        },
    }
    (root / "checkpoint.manifest.json").write_text(
        json.dumps(manifest, sort_keys=True),
        encoding="utf-8",
    )
    (root / "metadata.json").write_text(
        json.dumps(metadata, sort_keys=True),
        encoding="utf-8",
    )
    return ValidatedResumeCheckpoint(
        root=root.resolve(),
        manifest=manifest,
        metadata=metadata,
    )


def _runtime_identity() -> dict[str, object]:
    return {
        "accelerator": {
            "device_index": 0,
            "multi_processor_count": 48,
            "name": "AMD Radeon RX 7900 XTX",
            "total_memory_bytes": 25_708_244_992,
        },
        "cuda_available": True,
        "cuda_device_count": 1,
        "environment": {
            "HIP_VISIBLE_DEVICES": None,
            "HSA_OVERRIDE_GFX_VERSION": None,
            "ROCR_VISIBLE_DEVICES": None,
        },
        "platform": "Linux-test-WSL2",
        "python": "3.12.3 test",
        "resolved_device": "cuda",
        "torch_cuda_version": None,
        "torch_git_version": "f" * 40,
        "torch_hip_version": "7.2.test",
        "torch_version": "2.9.1+rocm7.2.1",
    }


def _readiness_report(
    tmp_path: Path,
    checkpoint: ValidatedResumeCheckpoint,
    *,
    generation_source: dict[str, object] | None = None,
    validator_relative_path: str = "code.py",
    validator_sha256: str = SHA_A,
) -> tuple[Path, str, dict[str, object]]:
    config_path = (tmp_path / "config.toml").resolve()
    config_path.write_text("version = 1\n", encoding="utf-8")
    source = checkpoint_source_identity(checkpoint)
    report: dict[str, object] = {
        "active_shape_probe": {},
        "authority": {},
        "config": {
            "calibration_updates": 256,
            "fingerprint_sha256": SHA_A,
            "maximum_candidates": 256,
            "maximum_context_steps": 256,
            "profile": "preheat",
            "risk_actor_start_update": 512,
            "source": str(config_path),
            "tbptt_window_steps": 16,
            "version": "sts2-relational-curriculum-config-v12",
        },
        "durations_seconds": {},
        "gates": {
            "bounded_tbptt_segments": True,
            "calibration_and_mature_phase_contract": True,
            "finite_gradients": True,
            "finite_losses": True,
            "formal_shape_contract": True,
            "matched_outcome_contrast_path": True,
            "no_optimizer_step": True,
            "nonzero_expected_gradients": True,
            "ordinary_rows_not_globally_padded": True,
            "production_manifest_work_accounting": True,
            "stress_forward_uses_active_timestep_batch": True,
        },
        "generation_source": {
            **(
                generation_source
                if generation_source is not None
                else {
                    "checkout_root": str((tmp_path / "checkout").resolve()),
                    "git_object_format": "sha1",
                    "implementation_commit": COMMIT_A,
                    "implementation_tree": TREE_A,
                    "worktree_clean": True,
                }
            ),
            "schema_version": FORMAL_REPORT_GENERATION_SOURCE_VERSION,
            "validator_relative_path": validator_relative_path,
            "validator_sha256": validator_sha256,
        },
        "gpu_peak": {
            "maximum_allocated_bytes": 5_970_000_000,
            "maximum_reserved_bytes": 5_990_000_000,
        },
        "initialization": {
            "authority": "pinned v28 shared model parameters with freshly initialized v29 liveness heads",
            "checkpoint_id": source["checkpoint_id"],
            "checkpoint_path": source["path"],
            "contract_name": "v28-mature-refinement-100k",
            "kind": "frozen-v28-model-init",
            "manifest_sha256": source["manifest_sha256"],
            "metadata_sha256": source["metadata_sha256"],
        },
        "one_shot": True,
        "optimizer_steps": 0,
        "phases": {},
        "runtime_identity": _runtime_identity(),
        "sdpa": {
            "applicability": "rocm_cuda",
            "applied": True,
            "current_flags": {
                "cudnn": False,
                "flash": False,
                "math": True,
                "memory_efficient": False,
            },
            "devices": ["cuda"],
            "effective_backend": "math_only",
            "hip_version": "7.2.test",
            "previous_flags": {
                "cudnn": True,
                "flash": True,
                "math": True,
                "memory_efficient": True,
            },
            "requested_policy": "math",
            "version": "sts2-rocm-sdpa-execution-v1",
        },
        "shape": {
            "candidate_decisions_per_context": 65_536,
            "contexts": 2,
            "expected_tbptt_segments": 32,
            "legal_candidates_per_step": 256,
            "ordinary_probe_candidates": 3,
            "replayed_candidate_decisions": 131_072,
            "replayed_steps": 512,
            "steps_per_context": 256,
            "tbptt_segments_per_context": 16,
        },
        "status": "passed",
        "training_authority": False,
        "version": RUNTIME_READINESS_REPORT_VERSION,
    }
    path = (tmp_path / "reports" / "readiness.json").resolve()
    path.parent.mkdir()
    raw = (json.dumps(report, indent=2, sort_keys=True) + "\n").encode()
    path.write_bytes(raw)
    return path, hashlib.sha256(raw).hexdigest(), report


def _payload(
    checkpoint: ValidatedResumeCheckpoint,
    *,
    tmp_path: Path,
) -> dict[str, object]:
    checkout = (tmp_path / "checkout").resolve()
    checkout.mkdir(exist_ok=True)
    code_path = checkout / "code.py"
    code_path.write_text("VALUE = 1\n", encoding="utf-8")
    generation_source = {
        "checkout_root": str(checkout),
        "git_object_format": "sha1",
        "implementation_commit": COMMIT_A,
        "implementation_tree": TREE_A,
        "worktree_clean": True,
    }
    code_hashes = {"code.py": hashlib.sha256(code_path.read_bytes()).hexdigest()}
    report_path, report_sha256, _ = _readiness_report(
        tmp_path,
        checkpoint,
        generation_source=generation_source,
        validator_relative_path="code.py",
        validator_sha256=code_hashes["code.py"],
    )
    readiness = load_runtime_readiness_evidence(
        report_path,
        expected_sha256=report_sha256,
        artifact_root=tmp_path,
    )
    implementation_source = {
        "checkout_root": str(checkout),
        "git_object_format": "sha1",
        "implementation_commit": COMMIT_B,
        "implementation_tree": TREE_B,
        "worktree_clean": True,
    }
    return {
        "schema_version": SUPERVISED_LAUNCH_CONTRACT_VERSION,
        "launch_id": str(uuid4()),
        "run_name": "formal-v29-model-init",
        "created_at_utc": "2026-07-31T00:00:00Z",
        "created_unix_s": 1_785_427_200.0,
        "config_fingerprint_sha256": SHA_A,
        "effective_config_fingerprint_sha256": SHA_B,
        "trainer_environment_sha256": trainer_environment_sha256(
            _trainer_environment(),
        ),
        "implementation_source": implementation_source,
        "source_authority": {
            "schema_version": SOURCE_AUTHORITY_VERSION,
            "evidence_generation_source": generation_source,
            "validated_code_sha256": code_hashes,
            "validated_code_mapping_sha256": hashlib.sha256(
                json.dumps(code_hashes, separators=(",", ":"), sort_keys=True).encode(),
            ).hexdigest(),
            "allowed_seal_paths": ["seal.json"],
            "observed_seal_paths": ["seal.json"],
        },
        "runtime_readiness_evidence": readiness,
        "seal_provenance": {
            "schema_version": EXTERNAL_SEAL_VERSION,
            "algorithm": "sha256",
            "digest_location": "supervisor_manifest.launch_contract.sha256",
            "self_digest_embedded": False,
        },
        "source_checkpoint": checkpoint_source_identity(checkpoint),
        "shadow_validation": {
            "contract_sha256": SHA_A,
            "semantics_report_sha256": SHA_B,
            "evidence_report_sha256": SHA_C,
            "actor_evidence_report_sha256": SHA_D,
            "validated_code_mapping_sha256": hashlib.sha256(
                json.dumps(code_hashes, separators=(",", ":"), sort_keys=True).encode(),
            ).hexdigest(),
            "evidence_generation_commit": COMMIT_A,
            "evidence_generation_tree": TREE_A,
        },
        "initial_training_state": {
            "environment_steps": 0,
            "learner_updates": 0,
            "episodes": 0,
            "evaluation_episodes": 0,
            "policy_version": 0,
            "actor_policy_version": 0,
            "consumed_unrolls": 0,
            "maximum_observed_candidates": 0,
        },
        "migration_proof_sha256": SHA_C,
    }


def _write_contract(
    tmp_path: Path,
    payload: dict[str, object],
) -> tuple[Path, str]:
    path = (tmp_path / "launch-contract.json").resolve()
    raw = (
        json.dumps(
            payload,
            allow_nan=False,
            indent=2,
            sort_keys=True,
        )
        + "\n"
    ).encode()
    path.write_bytes(raw)
    return path, hashlib.sha256(raw).hexdigest()


def test_launch_contract_binds_exact_config_and_checkpoint(tmp_path: Path) -> None:
    checkpoint = _checkpoint(tmp_path)
    payload = _payload(checkpoint, tmp_path=tmp_path)
    path, digest = _write_contract(tmp_path, payload)

    contract = load_supervised_launch_contract(
        path,
        expected_sha256=digest,
    )

    identity = validate_supervised_model_initialization_binding(
        contract,
        config_fingerprint_sha256=SHA_B,
        checkpoint=checkpoint,
    )
    assert identity == contract.source_checkpoint
    assert contract.provenance_mapping()["contract_sha256"] == digest
    assert contract.provenance_mapping()["trainer_environment_sha256"] == payload["trainer_environment_sha256"]
    assert (
        validate_supervised_trainer_environment(
            contract,
            environment=_trainer_environment(),
        )
        == payload["trainer_environment_sha256"]
    )
    assert set(contract.initial_training_state.values()) == {0}
    assert (
        validate_supervised_source_authority(
            contract,
            observed_implementation_source=payload["implementation_source"],
            observed_seal_paths=["seal.json"],
        )["implementation_source"]
        == payload["implementation_source"]
    )
    readiness = payload["runtime_readiness_evidence"]
    assert isinstance(readiness, dict)
    assert (
        validate_supervised_runtime_readiness(
            contract,
            observed_runtime_identity=_runtime_identity(),
        )["report_sha256"]
        == readiness["report_sha256"]
    )


def test_launch_contract_rejects_tamper_unknown_fields_and_nonzero_state(
    tmp_path: Path,
) -> None:
    checkpoint = _checkpoint(tmp_path)
    payload = _payload(checkpoint, tmp_path=tmp_path)
    path, digest = _write_contract(tmp_path, payload)

    with pytest.raises(SupervisedLaunchContractError, match="SHA-256 mismatch"):
        load_supervised_launch_contract(path, expected_sha256=SHA_A)

    payload["unexpected"] = True
    path, digest = _write_contract(tmp_path, payload)
    with pytest.raises(SupervisedLaunchContractError, match="fields differ"):
        load_supervised_launch_contract(path, expected_sha256=digest)

    payload.pop("unexpected")
    initial = payload["initial_training_state"]
    assert isinstance(initial, dict)
    initial["environment_steps"] = 1
    path, digest = _write_contract(tmp_path, payload)
    with pytest.raises(SupervisedLaunchContractError, match="must be zero"):
        load_supervised_launch_contract(path, expected_sha256=digest)


def test_launch_contract_rejects_config_or_source_mismatch(tmp_path: Path) -> None:
    checkpoint = _checkpoint(tmp_path)
    path, digest = _write_contract(tmp_path, _payload(checkpoint, tmp_path=tmp_path))
    contract = load_supervised_launch_contract(path, expected_sha256=digest)

    with pytest.raises(SupervisedLaunchContractError, match="effective config"):
        validate_supervised_model_initialization_binding(
            contract,
            config_fingerprint_sha256=SHA_A,
            checkpoint=checkpoint,
        )

    checkpoint.metadata["training_state"]["policy_version"] = 1_570
    with pytest.raises(SupervisedLaunchContractError, match="source differs"):
        validate_supervised_model_initialization_binding(
            contract,
            config_fingerprint_sha256=SHA_B,
            checkpoint=checkpoint,
        )


def test_launch_contract_rejects_trainer_environment_drift(
    tmp_path: Path,
) -> None:
    checkpoint = _checkpoint(tmp_path)
    path, digest = _write_contract(tmp_path, _payload(checkpoint, tmp_path=tmp_path))
    contract = load_supervised_launch_contract(
        path,
        expected_sha256=digest,
    )
    for mutate in (
        lambda value: value.__setitem__(
            "LD_PRELOAD",
            "/tmp/injected.so",
        ),
        lambda value: value.__setitem__("OMP_NUM_THREADS", "8"),
        lambda value: value.__setitem__(
            "PATH",
            value["PATH"] + ":/unreviewed",
        ),
        lambda value: value.pop("PYTHONPATH"),
        lambda value: value.__setitem__("PYTHONHOME", "/tmp/injected"),
    ):
        environment = _trainer_environment()
        mutate(environment)
        with pytest.raises(
            SupervisedLaunchContractError,
            match="environment",
        ):
            validate_supervised_trainer_environment(
                contract,
                environment=environment,
            )


def test_runtime_readiness_is_reopened_and_live_runtime_is_exact(
    tmp_path: Path,
) -> None:
    checkpoint = _checkpoint(tmp_path)
    payload = _payload(checkpoint, tmp_path=tmp_path)
    path, digest = _write_contract(tmp_path, payload)
    contract = load_supervised_launch_contract(path, expected_sha256=digest)

    drifted_runtime = _runtime_identity()
    accelerator = drifted_runtime["accelerator"]
    assert isinstance(accelerator, dict)
    accelerator["name"] = "different accelerator"
    with pytest.raises(
        SupervisedLaunchContractError,
        match="runtime differs",
    ):
        validate_supervised_runtime_readiness(
            contract,
            observed_runtime_identity=drifted_runtime,
        )

    evidence = payload["runtime_readiness_evidence"]
    assert isinstance(evidence, dict)
    report_path = Path(str(evidence["report_path"]))
    report_path.write_bytes(report_path.read_bytes() + b"\n")
    with pytest.raises(
        SupervisedLaunchContractError,
        match="SHA-256 mismatch",
    ):
        validate_supervised_runtime_readiness(
            contract,
            observed_runtime_identity=_runtime_identity(),
        )


def test_source_authority_rejects_commit_seal_and_code_drift(
    tmp_path: Path,
) -> None:
    checkpoint = _checkpoint(tmp_path)
    payload = _payload(checkpoint, tmp_path=tmp_path)
    path, digest = _write_contract(tmp_path, payload)
    contract = load_supervised_launch_contract(path, expected_sha256=digest)
    implementation = payload["implementation_source"]
    assert isinstance(implementation, dict)

    changed_commit = dict(implementation)
    changed_commit["implementation_commit"] = "9" * 40
    with pytest.raises(
        SupervisedLaunchContractError,
        match="implementation differs",
    ):
        validate_supervised_source_authority(
            contract,
            observed_implementation_source=changed_commit,
            observed_seal_paths=["seal.json"],
        )

    with pytest.raises(
        SupervisedLaunchContractError,
        match="seal paths differ",
    ):
        validate_supervised_source_authority(
            contract,
            observed_implementation_source=implementation,
            observed_seal_paths=["other-seal.json"],
        )

    checkout = Path(str(implementation["checkout_root"]))
    (checkout / "code.py").write_text("VALUE = 2\n", encoding="utf-8")
    with pytest.raises(
        SupervisedLaunchContractError,
        match="validated code differs",
    ):
        validate_supervised_source_authority(
            contract,
            observed_implementation_source=implementation,
            observed_seal_paths=["seal.json"],
        )
