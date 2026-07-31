from __future__ import annotations

import argparse
import importlib.util
import json
import os
import subprocess
from pathlib import Path
from types import ModuleType
from typing import Any

import pytest

SCRIPT = Path(__file__).parents[1] / "scripts" / "prepare_v29_authority_seal.py"


def _load_module() -> ModuleType:
    spec = importlib.util.spec_from_file_location("prepare_v29_authority_seal", SCRIPT)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


seal = _load_module()


def _git(root: Path, *args: str) -> str:
    completed = subprocess.run(
        ("git", *args),
        cwd=root,
        check=True,
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
    )
    return completed.stdout.strip()


def _write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _runtime_identity() -> dict[str, Any]:
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
        "platform": "test-linux",
        "python": "3.12.test",
        "resolved_device": "cuda",
        "torch_cuda_version": None,
        "torch_git_version": "f" * 40,
        "torch_hip_version": "7.2.test",
        "torch_version": "2.9.1+rocm-test",
    }


def _semantics_report() -> dict[str, Any]:
    return {
        "counts": {"parsed_records": 500_000, "rich_decision_snapshots": 3_986},
        "errors": [],
        "gates": {
            "maximum_unknown_transitions": 0,
            "minimum_rich_records": True,
            "no_contract_errors": True,
            "observed_unknown_transitions": 0,
            "required_minimum_rich_records": 1_000,
            "unknown_transition_budget": True,
        },
        "read_only": True,
        "semantic_manifest": {
            "digest": "ff7b2e40a2aeb6cd29fd5db5ea9f42998a285deadc175d7a2480de50a0a97bf8",
            "namespace": "surface_registry_manifest",
            "payload": {"contract_version": "sts2-surface-registry-v1"},
            "schema_version": "sts2-surface-registry-v1",
        },
        "status": "passed",
        "training_authority": False,
        "version": seal.SEMANTICS_VERSION,
    }


def _evidence_report(checkpoint: Path) -> dict[str, Any]:
    return {
        "config": {
            "collector_device": "cpu",
            "device": "cuda",
            "episodes": 2,
            "fingerprint_sha256": "a" * 64,
            "maximum_steps": 1_000,
            "mode": "shadow",
        },
        "counts": {
            "decisions": 1_000,
            "maximum_episode_completion_storage_nbytes": 1_024,
            "semantic_censored_transitions": 0,
        },
        "direct_policy_targets": {"avoid": 1},
        "gates": {key: True for key in seal.EVIDENCE_GATES},
        "read_only": True,
        "source": {
            "checkpoint": os.fspath(checkpoint.resolve()),
            "checkpoint_id": seal.FIXED_CHECKPOINT_ID,
            "environment_steps": seal.FIXED_CHECKPOINT_STEP,
            "manifest_sha256": seal.FIXED_MANIFEST_SHA256,
            "metadata_sha256": seal.FIXED_METADATA_SHA256,
            "policy_version": seal.FIXED_POLICY_VERSION,
        },
        "status": "passed",
        "training_authority": False,
        "training_state_unchanged": {"model": True, "optimizer": True},
        "version": seal.EVIDENCE_VERSION,
    }


def _actor_report() -> dict[str, Any]:
    return {
        "counts": {
            "actor_actionable_records": 4,
            "cases": 6,
            "direct_targets": {"avoid": 1},
            "risk_sequence_records": 5,
        },
        "fixture_authority": "reviewed-contract-cases-only",
        "gates": {key: True for key in seal.ACTOR_GATES},
        "read_only": True,
        "status": "passed",
        "training_authority": False,
        "version": seal.ACTOR_VERSION,
    }


def _readiness_report(config: Path, checkpoint: Path, identity: dict[str, Any]) -> dict[str, Any]:
    return {
        "active_shape_probe": {},
        "authority": {},
        "config": {
            "calibration_updates": 256,
            "fingerprint_sha256": seal.FIXED_CONFIG_FINGERPRINT_SHA256,
            "maximum_candidates": 256,
            "maximum_context_steps": 256,
            "profile": "preheat",
            "risk_actor_start_update": 512,
            "source": os.fspath(config.resolve()),
            "tbptt_window_steps": 16,
            "version": "sts2-relational-curriculum-config-v12",
        },
        "durations_seconds": {},
        "gates": {key: True for key in seal.READINESS_GATES},
        "gpu_peak": {
            "maximum_allocated_bytes": 5_000_000_000,
            "maximum_reserved_bytes": 6_000_000_000,
        },
        "initialization": {
            "authority": "pinned",
            "checkpoint_id": seal.FIXED_CHECKPOINT_ID,
            "checkpoint_path": os.fspath(checkpoint.resolve()),
            "contract_name": seal.FROZEN_CONTRACT_NAME,
            "kind": "frozen-v28-model-init",
            "manifest_sha256": seal.FIXED_MANIFEST_SHA256,
            "metadata_sha256": seal.FIXED_METADATA_SHA256,
        },
        "one_shot": True,
        "optimizer_steps": 0,
        "phases": {},
        "runtime_identity": identity,
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
            "hip_version": identity["torch_hip_version"],
            "requested_policy": "math",
            "version": "sts2-rocm-sdpa-execution-v1",
        },
        "shape": dict(seal.READINESS_SHAPE),
        "status": "passed",
        "training_authority": False,
        "version": seal.READINESS_VERSION,
    }


@pytest.fixture()
def authority_fixture(tmp_path: Path) -> dict[str, Any]:
    root = tmp_path / "checkout"
    artifact_root = tmp_path / "artifacts"
    external = tmp_path / "external"
    root.mkdir()
    artifact_root.mkdir()
    external.mkdir()
    _git(root, "init")
    _git(root, "config", "user.email", "test@example.com")
    _git(root, "config", "user.name", "Seal Test")
    fixture_closure = set(seal.REQUIRED_PRODUCTION_CLOSURE) | set(seal.REPORT_VALIDATORS.values())
    for relative in fixture_closure:
        path = root / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(f"production:{relative}\n", encoding="utf-8")
    launcher = root / seal.LAUNCHER_RELATIVE
    launcher.parent.mkdir(parents=True, exist_ok=True)
    launcher.write_text(
        "# reviewed launcher generation A\n"
        + "\n".join(
            f"{name} = {json.dumps(0 if name == 'SHADOW_VALIDATED_CODE_COUNT' else '0' * 64)}"
            for name in sorted(seal.LAUNCHER_SEAL_CONSTANT_NAMES)
        )
        + "\n\ndef reviewed_preflight() -> bool:\n    return True\n",
        encoding="utf-8",
    )
    base_contract = root / seal.SHADOW_CONTRACT_RELATIVE
    _write_json(
        base_contract,
        {
            "schema_version": "sts2-shadow-validation-contract-v2",
            "validated_code_sha256": {relative: "0" * 64 for relative in sorted(fixture_closure)},
        },
    )
    config = root / seal.DEFAULT_CONFIG_RELATIVE
    config.parent.mkdir(parents=True, exist_ok=True)
    config.write_text("profile = 'preheat'\n", encoding="utf-8")
    checkpoint = artifact_root / seal.DEFAULT_CHECKPOINT_RELATIVE
    checkpoint.mkdir(parents=True)
    _git(root, "add", ".")
    _git(root, "commit", "-m", "generation A")
    generation_commit = _git(root, "rev-parse", "HEAD")
    generation_source = seal._clean_source(root, expected_commit=generation_commit)
    identity = _runtime_identity()
    identity_path = external / "runtime-identity.json"
    _write_json(identity_path, identity)
    reports = {
        "semantics_historical": _semantics_report(),
        "evidence_live": _evidence_report(checkpoint),
        "evidence_actor": _actor_report(),
        "runtime_readiness": _readiness_report(config, checkpoint, identity),
    }
    for key, report in reports.items():
        validator_relative = seal.REPORT_VALIDATORS[key]
        report["generation_source"] = {
            **generation_source,
            "schema_version": seal.REPORT_GENERATION_SOURCE_VERSION,
            "validator_relative_path": validator_relative,
            "validator_sha256": seal._sha256_file(root / validator_relative),
        }
    report_paths: dict[str, Path] = {}
    for key, report in reports.items():
        report_path = artifact_root / seal.DEFAULT_REPORT_RELATIVES[key]
        _write_json(report_path, report)
        report_paths[key] = report_path
    return {
        "artifact_root": artifact_root,
        "checkpoint": checkpoint,
        "config": config,
        "external": external,
        "generation_commit": generation_commit,
        "identity": identity,
        "identity_path": identity_path,
        "report_paths": report_paths,
        "root": root,
    }


def _prepare_args(fixture: dict[str, Any], **updates: Any) -> argparse.Namespace:
    values: dict[str, Any] = {
        "artifact_root": os.fspath(fixture["artifact_root"]),
        "checkpoint": None,
        "checkout_root": os.fspath(fixture["root"]),
        "config": None,
        "expected_evidence_actor_sha256": None,
        "expected_evidence_live_sha256": None,
        "expected_runtime_readiness_sha256": None,
        "expected_semantics_historical_sha256": None,
        "generation_commit": fixture["generation_commit"],
        "runtime_identity_json": os.fspath(fixture["identity_path"]),
        "runtime_python": None,
    }
    values.update(updates)
    return argparse.Namespace(**values)


def _prepare_external(fixture: dict[str, Any]) -> tuple[dict[str, Any], Path, Path]:
    plan, shadow_bytes = seal._prepare_payload(_prepare_args(fixture))
    plan_path = fixture["external"] / "seal-plan.json"
    shadow_path = fixture["external"] / "shadow-contract.json"
    plan_path.write_bytes(seal._pretty_json_bytes(plan))
    shadow_path.write_bytes(shadow_bytes)
    return plan, plan_path, shadow_path


def _install_seal_b(
    fixture: dict[str, Any],
    plan: dict[str, Any],
    shadow_path: Path,
    *,
    include_launcher: bool = True,
) -> None:
    root = fixture["root"]
    (root / seal.SHADOW_CONTRACT_RELATIVE).write_bytes(shadow_path.read_bytes())
    if include_launcher:
        (root / seal.LAUNCHER_RELATIVE).write_text(
            plan["launcher_seal"]["expected_content_utf8"],
            encoding="utf-8",
            newline="",
        )
    _git(root, "add", ".")
    _git(root, "commit", "-m", "seal B")


def _verify_args(fixture: dict[str, Any], plan_path: Path, shadow_path: Path) -> argparse.Namespace:
    return argparse.Namespace(
        checkout_root=os.fspath(fixture["root"]),
        plan=os.fspath(plan_path),
        runtime_identity_json=os.fspath(fixture["identity_path"]),
        runtime_python=None,
        staged_shadow_contract=os.fspath(shadow_path),
    )


def test_prepare_is_deterministic_non_launching_and_verify_accepts_exact_two_phase(
    authority_fixture: dict[str, Any],
) -> None:
    plan_one, shadow_one = seal._prepare_payload(_prepare_args(authority_fixture))
    plan_two, shadow_two = seal._prepare_payload(_prepare_args(authority_fixture))
    assert seal._pretty_json_bytes(plan_one) == seal._pretty_json_bytes(plan_two)
    assert shadow_one == shadow_two
    assert plan_one["training_authority"] is False
    assert plan_one["launch_allowed"] is False
    plan, plan_path, shadow_path = _prepare_external(authority_fixture)
    _install_seal_b(authority_fixture, plan, shadow_path)
    result = seal._verify(_verify_args(authority_fixture, plan_path, shadow_path))
    assert result["status"] == "verified"
    assert result["training_authority"] is False
    assert result["observed_seal_paths"] == sorted(seal.REVIEWED_SEAL_PATHS)


def test_prepare_refuses_dirty_checkout(authority_fixture: dict[str, Any]) -> None:
    (authority_fixture["root"] / "untracked.txt").write_text("dirty", encoding="utf-8")
    with pytest.raises(seal.SealError, match="clean checkout"):
        seal._prepare_payload(_prepare_args(authority_fixture))


def test_prepare_refuses_missing_or_wrong_generation_commit(authority_fixture: dict[str, Any]) -> None:
    with pytest.raises(seal.SealError, match="must equal clean HEAD"):
        seal._prepare_payload(_prepare_args(authority_fixture, generation_commit="0" * 40))
    with pytest.raises(seal.SealError, match="full object ID"):
        seal._prepare_payload(_prepare_args(authority_fixture, generation_commit="HEAD"))


def test_prepare_refuses_missing_and_stale_reports(authority_fixture: dict[str, Any]) -> None:
    report = authority_fixture["report_paths"]["semantics_historical"]
    report.unlink()
    with pytest.raises(seal.SealError, match="is missing"):
        seal._prepare_payload(_prepare_args(authority_fixture))
    _write_json(report, _semantics_report())
    commit_time = int(_git(authority_fixture["root"], "show", "-s", "--format=%ct", "HEAD"))
    os.utime(report, (commit_time - 10, commit_time - 10))
    with pytest.raises(seal.SealError, match="predates generation commit"):
        seal._prepare_payload(_prepare_args(authority_fixture))


@pytest.mark.parametrize(
    ("key", "mutate", "message"),
    [
        ("semantics_historical", lambda value: value.update(version="old-schema"), "schema/version"),
        ("evidence_actor", lambda value: value.update(status="failed"), "status is not passed"),
        (
            "evidence_live",
            lambda value: value["source"].update(checkpoint_id="00000000-0000-0000-0000-000000000000"),
            "frozen source changed",
        ),
        (
            "runtime_readiness",
            lambda value: value["config"].update(maximum_candidates=128),
            "config/source changed",
        ),
        (
            "semantics_historical",
            lambda value: value["generation_source"].update(implementation_commit="f" * 40),
            "not generated by clean commit A",
        ),
    ],
)
def test_prepare_refuses_wrong_report_schema_status_source_or_config(
    authority_fixture: dict[str, Any],
    key: str,
    mutate: Any,
    message: str,
) -> None:
    path = authority_fixture["report_paths"][key]
    value = json.loads(path.read_text(encoding="utf-8"))
    mutate(value)
    _write_json(path, value)
    with pytest.raises(seal.SealError, match=message):
        seal._prepare_payload(_prepare_args(authority_fixture))


def test_prepare_refuses_wrong_expected_hash_and_runtime_identity(authority_fixture: dict[str, Any]) -> None:
    with pytest.raises(seal.SealError, match="SHA-256 mismatch"):
        seal._prepare_payload(_prepare_args(authority_fixture, expected_evidence_live_sha256="0" * 64))
    identity = dict(authority_fixture["identity"])
    identity["torch_version"] = "different"
    _write_json(authority_fixture["identity_path"], identity)
    with pytest.raises(seal.SealError, match="differs from the live reviewed runtime"):
        seal._prepare_payload(_prepare_args(authority_fixture))


def test_prepare_refuses_output_inside_tracked_checkout(authority_fixture: dict[str, Any]) -> None:
    with pytest.raises(seal.SealError, match="outside the tracked checkout"):
        seal._require_external_output(
            authority_fixture["root"] / "plan.json",
            root=authority_fixture["root"],
            label="plan output",
        )


def test_verify_refuses_report_tamper_after_prepare(authority_fixture: dict[str, Any]) -> None:
    plan, plan_path, shadow_path = _prepare_external(authority_fixture)
    _install_seal_b(authority_fixture, plan, shadow_path)
    report = authority_fixture["report_paths"]["evidence_actor"]
    report.write_bytes(report.read_bytes() + b"\n")
    with pytest.raises(seal.SealError, match="hash changed after prepare"):
        seal._verify(_verify_args(authority_fixture, plan_path, shadow_path))


def test_verify_refuses_nonexact_seal_paths_and_production_drift(authority_fixture: dict[str, Any]) -> None:
    plan, plan_path, shadow_path = _prepare_external(authority_fixture)
    production = authority_fixture["root"] / next(iter(seal.REQUIRED_PRODUCTION_CLOSURE))
    production.write_text("drift\n", encoding="utf-8")
    _install_seal_b(authority_fixture, plan, shadow_path)
    with pytest.raises(seal.SealError, match="exact reviewed seal set"):
        seal._verify(_verify_args(authority_fixture, plan_path, shadow_path))


def test_verify_refuses_incomplete_seal_commit(authority_fixture: dict[str, Any]) -> None:
    plan, plan_path, shadow_path = _prepare_external(authority_fixture)
    _install_seal_b(authority_fixture, plan, shadow_path, include_launcher=False)
    with pytest.raises(seal.SealError, match="exact reviewed seal set"):
        seal._verify(_verify_args(authority_fixture, plan_path, shadow_path))


def test_verify_refuses_any_launcher_edit_beyond_seal_constants(
    authority_fixture: dict[str, Any],
) -> None:
    plan, plan_path, shadow_path = _prepare_external(authority_fixture)
    root = authority_fixture["root"]
    (root / seal.SHADOW_CONTRACT_RELATIVE).write_bytes(shadow_path.read_bytes())
    expected = plan["launcher_seal"]["expected_content_utf8"]
    (root / seal.LAUNCHER_RELATIVE).write_text(
        expected + "\n# unauthorized launcher edit\n",
        encoding="utf-8",
        newline="",
    )
    _git(root, "add", ".")
    _git(root, "commit", "-m", "tampered seal B")
    with pytest.raises(seal.SealError, match="beyond the reviewed constants"):
        seal._verify(_verify_args(authority_fixture, plan_path, shadow_path))


def test_verify_rederives_full_closure_from_generation_commit_a(
    authority_fixture: dict[str, Any],
) -> None:
    plan, plan_path, shadow_path = _prepare_external(authority_fixture)
    mapping = plan["source_authority"]["validated_code_sha256"]
    mapping.pop(next(iter(mapping)))
    plan_path.write_bytes(seal._pretty_json_bytes(plan))
    _install_seal_b(authority_fixture, plan, shadow_path)
    with pytest.raises(seal.SealError, match="differs from generation commit A authority"):
        seal._verify(_verify_args(authority_fixture, plan_path, shadow_path))


def test_verify_requires_seal_b_to_be_direct_single_child_of_a(
    authority_fixture: dict[str, Any],
) -> None:
    plan, plan_path, shadow_path = _prepare_external(authority_fixture)
    _git(authority_fixture["root"], "commit", "--allow-empty", "-m", "intermediate")
    _install_seal_b(authority_fixture, plan, shadow_path)
    with pytest.raises(seal.SealError, match="only direct parent"):
        seal._verify(_verify_args(authority_fixture, plan_path, shadow_path))
