from __future__ import annotations

import hashlib
import importlib.util
import json
import signal
import sys
from collections.abc import Callable
from dataclasses import asdict
from pathlib import Path
from typing import Any

import pytest

from sts2_rl.training import launch_contract as trainer_launch_contract

PACKAGE_ROOT = Path(__file__).resolve().parents[1]
CHECKOUT_ROOT = PACKAGE_ROOT.parents[1]
SCRIPT = PACKAGE_ROOT / "scripts" / "launch_v29_failure_credit_v4.py"


def _load_launcher_module(name: str) -> Any:
    spec = importlib.util.spec_from_file_location(name, SCRIPT)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


launcher = _load_launcher_module("v29_supervised_launcher")

ABI = launcher._expected_abi_contract()
TEST_RUN_ID = "44444444-4444-4444-8444-444444444444"
FORMAL_SHADOW_CONTRACT = json.loads(
    (CHECKOUT_ROOT / "contracts/shadow-validation/v29-failure-credit-v4-bootstrap.json").read_text(encoding="utf-8"),
)
TEST_CODE_HASHES = dict(FORMAL_SHADOW_CONTRACT["validated_code_sha256"])
for index, required_path in enumerate(sorted(launcher.REQUIRED_SHADOW_CODE_PATHS), start=1):
    TEST_CODE_HASHES.setdefault(required_path, f"{index:064x}")
SHADOW = {
    "contract": {
        "name": "v29-failure-credit-v4-bootstrap",
        "path": "/reviewed/contracts/v29-failure-credit-v4-bootstrap.json",
        "schema_version": launcher.SHADOW_CONTRACT_SCHEMA,
        "sha256": launcher.SHADOW_CONTRACT_SHA256,
    },
    "generation_source": {
        "checkout_root": "/reviewed/source",
        "git_object_format": "sha1",
        "implementation_commit": "1" * 40,
        "implementation_tree": "2" * 40,
        "worktree_clean": True,
    },
    "validated_code_sha256": TEST_CODE_HASHES,
    "reports": {
        "semantics_historical": {
            "path": "/reviewed/reports/semantic-shadow-historical-v3.json",
            "sha256": launcher.SEMANTICS_SHADOW_SHA256,
            "status": "passed",
            "parsed_records": 505_344,
            "rich_decision_snapshots": 3_986,
            "unknown_transitions": 0,
            "errors": 0,
        },
        "evidence_live": {
            "path": "/reviewed/reports/evidence-shadow-v28-live.json",
            "sha256": launcher.EVIDENCE_SHADOW_SHA256,
            "status": "passed",
            "decisions": 1_713,
            "semantic_censored_transitions": 0,
            "prefer_targets": 0,
            "source_checkpoint_id": (launcher.FIXED_INITIALIZATION_CHECKPOINT_ID),
            "training_state_unchanged": True,
            "maximum_episode_completion_storage_nbytes": 133_708_949,
            "completion_staging_is_bounded": True,
        },
        "evidence_actor": {
            "path": "/reviewed/reports/evidence-shadow-actor-contract-v2.json",
            "sha256": launcher.ACTOR_EVIDENCE_SHADOW_SHA256,
            "status": "passed",
            "cases": 6,
            "actor_actionable_records": 4,
            "risk_sequence_records": 5,
            "prefer_targets": 0,
            "learner_mask_dry_run_passed": True,
        },
    },
}
launcher.SHADOW_VALIDATED_CODE_COUNT = len(TEST_CODE_HASHES)
launcher.SHADOW_VALIDATED_CODE_MAPPING_SHA256 = launcher._canonical_json_sha256(
    TEST_CODE_HASHES,
)
SHADOW["authority_sha256"] = launcher._canonical_json_sha256(
    launcher._shadow_authority_payload(SHADOW),
)
launcher.SHADOW_AUTHORITY_SHA256 = SHADOW["authority_sha256"]


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _running_in_wsl() -> bool:
    if not sys.platform.startswith("linux"):
        return False
    try:
        release = Path("/proc/sys/kernel/osrelease").read_text(
            encoding="utf-8",
            errors="replace",
        )
    except OSError:
        return False
    return "microsoft" in release.casefold()


def _paths(tmp_path: Path) -> Any:
    checkout = tmp_path / "source"
    package = checkout / "packages" / "rl-agent"
    artifact = tmp_path / "artifacts" / "runtime"
    simulator = artifact / "dependencies" / "HeadlessSim.exe"
    preflight = launcher.v29_preflight.PreflightPaths(
        checkout_root=checkout,
        package_root=package,
        artifact_root=artifact,
        venv_python=artifact / "environments/wsl-rocm/bin/python",
        config_path=package / "config/v29.toml",
        simulator_executable=simulator,
        simulator_identity=simulator.with_name(simulator.name + ".identity.json"),
    )
    return launcher.LaunchPaths(
        preflight=preflight,
        launcher_dir=artifact / "launcher",
        manifest_dir=artifact / "launchers",
        initialization_checkpoint=(artifact / launcher.FIXED_INITIALIZATION_RELATIVE),
    )


def _migration() -> dict[str, Any]:
    return {
        "mode": "model_initialization",
        "source_transaction_head_tensors_dropped": 18,
        "target_transaction_head_tensors": 0,
        "fresh_liveness_head_tensors": 12,
        "shared_network_tensors_inherited": 214,
        "optimizer_state_entries": 0,
        "rollout_queue_items": 0,
        "transaction_replay": "absent",
        "failure_credit_replay": {
            "version": launcher.FAILURE_CREDIT_REPLAY_V4,
            "size": 0,
            "put_count": 0,
        },
        "episodic_replay": {
            "version": "sts2-bounded-episodic-replay-v1",
            "size": 0,
            "put_count": 0,
        },
        "rng_preserved_from_fresh_lineage": True,
        "new_training_state": {
            "environment_steps": 0,
            "learner_updates": 0,
            "episodes": 0,
            "evaluation_episodes": 0,
            "policy_version": 0,
            "actor_policy_version": 0,
            "consumed_unrolls": 0,
            "maximum_observed_candidates": 0,
        },
    }


def _test_runtime_identity() -> dict[str, Any]:
    return {
        "accelerator": {
            "device_index": 0,
            "multi_processor_count": 48,
            "name": "test-gpu",
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
        "torch_version": "test",
    }


def _payload(paths: Any) -> dict[str, Any]:
    environment = launcher.v29_preflight.build_trainer_environment(
        paths.preflight,
        base_environment={"PATH": "/usr/bin"},
    )
    environment_contract = launcher.v29_preflight.trainer_environment_contract(environment)
    exact_environment = launcher._environment_from_contract(
        environment_contract,
    )
    environment_sha256 = launcher._exact_trainer_environment_sha256(
        exact_environment,
    )
    runtime_identity = _test_runtime_identity()
    implementation_source = {
        "checkout_root": str(paths.checkout_root),
        "git_object_format": "sha1",
        "implementation_commit": "3" * 40,
        "implementation_tree": "4" * 40,
        "worktree_clean": True,
    }
    source_authority = {
        "schema_version": launcher.SOURCE_AUTHORITY_SCHEMA,
        "evidence_generation_source": SHADOW["generation_source"],
        "validated_code_sha256": TEST_CODE_HASHES,
        "validated_code_mapping_sha256": launcher._canonical_json_sha256(
            TEST_CODE_HASHES,
        ),
        "allowed_seal_paths": sorted(launcher.REVIEWED_SEAL_PATHS),
        "observed_seal_paths": sorted(launcher.REVIEWED_SEAL_PATHS),
    }
    readiness = {
        "schema_version": launcher.RUNTIME_READINESS_EVIDENCE_SCHEMA,
        "artifact_root": str(paths.artifact_root),
        "report_path": str(
            (paths.artifact_root / launcher.RUNTIME_READINESS_REPORT_RELATIVE).resolve(
                strict=False,
            )
        ),
        "report_sha256": launcher.RUNTIME_READINESS_REPORT_SHA256,
        "report_binding": {
            "config": {
                "fingerprint_sha256": launcher.FIXED_CONFIG_FINGERPRINT_SHA256,
            },
            "initialization": {
                "checkpoint_id": launcher.FIXED_INITIALIZATION_CHECKPOINT_ID,
            },
            "generation_source": {
                **SHADOW["generation_source"],
                "schema_version": launcher.FORMAL_REPORT_GENERATION_SOURCE_SCHEMA,
                "validator_relative_path": launcher.FORMAL_REPORT_VALIDATORS["runtime_readiness"],
                "validator_sha256": TEST_CODE_HASHES[launcher.FORMAL_REPORT_VALIDATORS["runtime_readiness"]],
            },
            "runtime_identity": runtime_identity,
            "status": "passed",
            "training_authority": False,
            "version": launcher.RUNTIME_READINESS_REPORT_SCHEMA,
        },
    }
    runtime: dict[str, Any] = {
        "schema_version": launcher.HERMETIC_RUNTIME_PROBE_SCHEMA,
        "probe_command": list(
            launcher._hermetic_runtime_probe_command(paths.preflight),
        ),
        "timeout_seconds": launcher.HERMETIC_RUNTIME_PROBE_TIMEOUT_SECONDS,
        "trainer_environment_sha256": environment_sha256,
        "torch_version": "test",
        "device_name": "test-gpu",
        "runtime_identity": runtime_identity,
        "simulator": {"schema_version": "test"},
        "runtime_mechanics": {
            "schema": "sts2-runtime-mechanics-audit-v1",
            "runtime_event_checked": True,
            "runtime_combat_checked": True,
        },
        "training_revival": {
            "mechanism": "test-engine-revival",
        },
    }
    runtime["result_sha256"] = launcher._canonical_json_sha256(
        launcher._runtime_probe_result_mapping(runtime),
    )
    return {
        "status": "preflight-passed",
        "run_name": launcher.RUN_NAME,
        "preflight_training_started": False,
        "training_started": False,
        "config_fingerprint_sha256": (launcher.FIXED_CONFIG_FINGERPRINT_SHA256),
        "effective_config_fingerprint_sha256": (launcher.FIXED_RUNTIME_CONFIG_FINGERPRINT_SHA256),
        "abi_contract": ABI,
        "shadow_validation": SHADOW,
        "runtime_readiness_evidence": readiness,
        "git": implementation_source,
        "source_authority": source_authority,
        "initialization": {
            "mode": "model_initialization",
            "checkpoint": str(paths.initialization_checkpoint),
            "checkpoint_contract": launcher.FIXED_FROZEN_CONTRACT_NAME,
            "checkpoint_id": launcher.FIXED_INITIALIZATION_CHECKPOINT_ID,
            "source_training_state": {
                "environment_steps": launcher.FIXED_INITIALIZATION_STEP,
                "policy_version": (launcher.FIXED_INITIALIZATION_POLICY_VERSION),
            },
            "network_parameters_inherited": True,
            "old_transaction_heads_removed": True,
            "new_liveness_heads_fresh": True,
            "optimizer_rollouts_rng_and_counters_reset": True,
            "migration_proof": _migration(),
        },
        "trainer_command": list(
            launcher.v29_preflight.build_trainer_command(
                paths.preflight,
                initialize_from=paths.initialization_checkpoint,
            )
        ),
        "trainer_environment": environment_contract,
        "trainer_environment_sha256": environment_sha256,
        "runtime": runtime,
    }


def _proof(paths: Any) -> dict[str, Any]:
    return {
        "contract_version": (launcher.FROZEN_CHECKPOINT_CONTRACT_VERSION),
        "contract_name": launcher.FIXED_FROZEN_CONTRACT_NAME,
        "exact_resume_permitted": False,
        "path": str(paths.initialization_checkpoint),
        "checkpoint_id": launcher.FIXED_INITIALIZATION_CHECKPOINT_ID,
        "manifest_sha256": (launcher.FIXED_INITIALIZATION_MANIFEST_SHA256),
        "metadata_sha256": (launcher.FIXED_INITIALIZATION_METADATA_SHA256),
        "source_environment_steps": launcher.FIXED_INITIALIZATION_STEP,
        "source_policy_version": (launcher.FIXED_INITIALIZATION_POLICY_VERSION),
        "source_git_commit": (launcher.FIXED_INITIALIZATION_SOURCE_GIT_COMMIT),
    }


def _authorize_manifest(
    paths: Any,
    manifest: dict[str, Any],
    *,
    manifest_path: Path | None = None,
) -> dict[str, Any]:
    launch_id = str(manifest["launch_id"])
    bound_manifest_path = (
        manifest_path
        if manifest_path is not None
        else paths.manifest_dir / f"{launcher.RUN_NAME}-unit-{launch_id}.launch.json"
    ).resolve(strict=False)
    created_at_utc = str(
        manifest.setdefault(
            "created_at_utc",
            "1970-01-01T00:00:01+00:00",
        )
    )
    created_unix_s = float(manifest.setdefault("created_unix_s", 1.0))
    launch_contract = launcher._create_supervised_launch_contract(
        paths,
        preflight=manifest,
        launch_id=launch_id,
        created_at_utc=created_at_utc,
        created_unix_s=created_unix_s,
    )
    manifest["launch_contract"] = launch_contract
    manifest["launch_contract_payload"] = json.loads(
        Path(launch_contract["path"]).read_text(encoding="utf-8"),
    )
    manifest["supervised_trainer_command"] = list(
        launcher._supervised_trainer_command(
            manifest["trainer_command"],
            launch_contract=launch_contract,
        )
    )
    manifest["manifest_path"] = str(bound_manifest_path)
    manifest["supervisor_command"] = list(
        launcher._supervisor_command(
            paths,
            bound_manifest_path,
        )
    )
    training_started = manifest.get("training_started") is True
    manifest.setdefault("trainer_spawn_intent", training_started)
    manifest.setdefault(
        "trainer_spawn_intent_at_utc",
        "1970-01-01T00:00:02+00:00" if training_started else None,
    )
    manifest.setdefault("trainer_spawn_failed", False)
    manifest.setdefault("trainer_spawn_failure_at_utc", None)
    manifest.setdefault("trainer_exit_observed", False)
    manifest.setdefault("trainer_exit_observed_at_utc", None)
    manifest.setdefault("trainer_reaped_returncode", None)
    manifest.setdefault(
        "trainer_bootstrap_protocol",
        launcher.TRAINER_BOOTSTRAP_PROTOCOL,
    )
    manifest.setdefault("trainer_bootstrap_phase", "none")
    manifest.setdefault("trainer_bootstrap_command", None)
    manifest.setdefault("trainer_bootstrap_nonce_sha256", None)
    manifest.setdefault("trainer_bootstrap_process_identity", None)
    manifest.setdefault("trainer_bootstrap_ready_at_utc", None)
    manifest.setdefault("trainer_bootstrap_exec_authorized_at_utc", None)
    manifest.setdefault("trainer_bootstrap_exec_observed_at_utc", None)
    manifest.setdefault("trainer_spawned_pid", None)
    manifest.setdefault("trainer_spawned_at_utc", None)
    manifest.setdefault("trainer_started_at_utc", None)
    if training_started:
        supervisor = _identity_for_command(
            manifest["supervisor_command"],
            pid=101,
            proc_start_ticks=11,
        )
        manifest.setdefault("supervisor_spawned_pid", supervisor.pid)
        manifest.setdefault("supervisor_process_identity", asdict(supervisor))
        nonce = "a" * 64
        bootstrap_command = launcher._trainer_bootstrap_command(
            paths,
            supervisor_identity=supervisor,
            gate_fd=101,
            status_fd=102,
            nonce=nonce,
            trainer_command=manifest["supervised_trainer_command"],
        )
        bootstrap = _identity_for_command(
            bootstrap_command,
            pid=4_242,
            proc_start_ticks=22,
        )
        trainer = _identity_for_command(
            manifest["supervised_trainer_command"],
            pid=4_242,
            proc_start_ticks=22,
        )
        manifest.update(
            {
                "trainer_bootstrap_phase": "exec_observed",
                "trainer_bootstrap_command": list(bootstrap_command),
                "trainer_bootstrap_nonce_sha256": launcher._sha256_bytes(
                    nonce.encode("ascii"),
                ),
                "trainer_bootstrap_process_identity": asdict(bootstrap),
                "trainer_bootstrap_ready_at_utc": "1970-01-01T00:00:03+00:00",
                "trainer_bootstrap_exec_authorized_at_utc": "1970-01-01T00:00:04+00:00",
                "trainer_bootstrap_exec_observed_at_utc": "1970-01-01T00:00:05+00:00",
                "trainer_spawned_pid": trainer.pid,
                "trainer_spawned_at_utc": "1970-01-01T00:00:05+00:00",
                "trainer_started_at_utc": "1970-01-01T00:00:05+00:00",
                "trainer_process_identity": asdict(trainer),
            }
        )
    return manifest


def _identity_for_command(
    command: list[str] | tuple[str, ...],
    *,
    pid: int,
    proc_start_ticks: int,
) -> Any:
    return launcher.ProcessIdentity(
        pid=pid,
        proc_start_ticks=proc_start_ticks,
        command_line_sha256=launcher._command_line_sha256(command),
        executable=str(Path(command[0]).resolve(strict=False)),
    )


def _patch_payload_authorities(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        launcher,
        "_validate_shadow_contract",
        lambda _paths: SHADOW,
    )
    monkeypatch.setattr(
        launcher,
        "SHADOW_AUTHORITY_SHA256",
        SHADOW["authority_sha256"],
    )
    monkeypatch.setattr(
        launcher,
        "_validate_runtime_readiness_evidence",
        lambda paths, *, runtime, shadow: _payload(paths)["runtime_readiness_evidence"],
    )


def _write_event(path: Path, event: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8", newline="\n") as handle:
        handle.write(json.dumps(event) + "\n")


def _run_config() -> dict[str, Any]:
    return {
        "version": "sts2-relational-curriculum-config-v12",
        "transaction_learning": {"enabled": False},
        "failure_credit": {"mode": "learning"},
        "runtime": {
            "seed": 4_000_000,
            "total_environment_steps": 100_000,
        },
    }


def _run_start(
    paths: Any,
    *,
    manifest: dict[str, Any],
) -> dict[str, Any]:
    pin = manifest["launch_contract"]
    contract = json.loads(Path(pin["path"]).read_text(encoding="utf-8"))
    return {
        "event": "run_start",
        "unix_s": 1.0,
        "run_id": TEST_RUN_ID,
        "pipeline": launcher.TRAINING_PIPELINE_V7,
        "state": contract["initial_training_state"],
        "actor_supervisor_state": {
            "version": "sts2-actor-supervisor-state-v1",
            "episode_attempts": 0,
            "consecutive_incidents": 0,
            "incident_fingerprints": {},
            "recent_incident_attempts": [],
        },
        "runtime_provenance": {
            "backend": "headless",
            "training_revival": manifest["runtime"]["training_revival"],
            "simulator_identity": manifest["runtime"]["simulator"],
            "simulator_identity_audit_path": str(
                paths.artifact_root / "logs/simulator-preflight/test.json",
            ),
            "runtime_mechanics": manifest["runtime"]["runtime_mechanics"],
            "runtime_mechanics_audit_path": str(
                paths.artifact_root / "logs/runtime-mechanics-preflight/test.json",
            ),
            "sdpa_backend": {
                "version": "sts2-rocm-sdpa-transition-v1",
                "checkpoint_load_mode": "model_initialization",
                "previous": {
                    "recording_status": "no_parent_checkpoint",
                    "requested_policy": None,
                    "effective_backend": None,
                },
                "previous_label": "no_parent_checkpoint",
                "current": {
                    "version": "sts2-rocm-sdpa-execution-v1",
                    "requested_policy": "math",
                    "devices": ["cuda", "cpu"],
                    "hip_version": "7.2.1",
                    "applicability": "rocm_cuda",
                    "applied": True,
                    "previous_flags": {
                        "flash": True,
                        "memory_efficient": True,
                        "math": True,
                        "cudnn": True,
                    },
                    "current_flags": {
                        "flash": False,
                        "memory_efficient": False,
                        "math": True,
                        "cudnn": False,
                    },
                    "effective_backend": "math_only",
                },
                "current_label": "math_only",
                "changed": False,
                "reason": "configured execution policy at process start",
            },
            "supervised_launch": {
                "contract_path": pin["path"],
                "contract_sha256": pin["sha256"],
                "trainer_environment_sha256": contract["trainer_environment_sha256"],
                "implementation_commit": contract["implementation_source"]["implementation_commit"],
                "implementation_tree": contract["implementation_source"]["implementation_tree"],
                "worktree_clean": True,
                "runtime_readiness_report_sha256": contract["runtime_readiness_evidence"]["report_sha256"],
                "contract": contract,
            },
        },
        "config_fingerprint_sha256": (launcher.FIXED_RUNTIME_CONFIG_FINGERPRINT_SHA256),
        "config": _run_config(),
        "checkpoint_load": {
            "mode": "model_initialization",
            "parent_checkpoint": str(paths.initialization_checkpoint),
            "source_training_state": {
                "environment_steps": launcher.FIXED_INITIALIZATION_STEP,
                "policy_version": (launcher.FIXED_INITIALIZATION_POLICY_VERSION),
            },
            "source_checkpoint": contract["source_checkpoint"],
            "network_parameters_initialized": True,
            "optimizer_rollouts_rng_and_counters_reset": True,
        },
    }


def test_launcher_is_pinned_to_explicit_v28_100k_contract() -> None:
    # These literals are intentionally independent of launcher.FIXED_* so a
    # copied v27 constant cannot make the test self-consistently pass.
    assert launcher.FIXED_FROZEN_CONTRACT_NAME == "v28-mature-refinement-100k"
    assert launcher.FIXED_INITIALIZATION_STEP == 100_000
    assert launcher.FIXED_INITIALIZATION_POLICY_VERSION == 1_569
    assert launcher.FIXED_INITIALIZATION_CHECKPOINT_ID == "f670deda-97d5-46f2-be07-d69a5842eeec"
    assert launcher.FIXED_INITIALIZATION_RELATIVE == Path(
        "checkpoints/full-run-revival-v28-mature-refinement-model-init/"
        "run-071a43f5-120a-485f-a6dc-a67a55a1efc2/"
        "periodic-step-000100000"
    )


@pytest.mark.skipif(
    importlib.util.find_spec("torch") is None,
    reason="runtime ABI probe requires the training Torch environment",
)
def test_replay_v5_runtime_retires_the_historical_v29_launcher_fail_closed() -> None:
    from sts2_rl.checkpoints.frozen import V28_100K_FROZEN
    from sts2_rl.training import checkpointing

    assert launcher._v28_frozen_contract() is V28_100K_FROZEN
    actual = checkpointing._failure_credit_abi()
    assert actual["collector"] == "sts2-failure-credit-collector-v5"
    assert actual["detector"] == "sts2-semantic-macro-cycle-detector-v4"
    assert actual["replay"] == "sts2-failure-evidence-replay-v5"
    with pytest.raises(launcher.LaunchError, match="failure-credit-v4 ABI changed"):
        launcher._abi_contract()


@pytest.mark.skipif(
    not _running_in_wsl(),
    reason="formal shadow integration uses the reviewed WSL artifact root",
)
def test_formal_shadow_contract_matches_current_code_or_retires_fail_closed() -> None:
    # The synthetic unit-test authority above deliberately replaces seal
    # constants on ``launcher``.  Formal integration must instead validate
    # the immutable constants embedded in the tracked launcher itself.  A
    # later production commit is allowed to retire this historical launch
    # authority, but must make the old launcher fail closed rather than let
    # the old reports authorize the changed code.
    formal_launcher = _load_launcher_module("v29_supervised_launcher_formal")
    paths = formal_launcher.validate_layout(formal_launcher.default_paths())
    runtime_relative = "packages/rl-agent/sts2_rl/training/runtime.py"
    current_runtime_sha256 = hashlib.sha256((paths.checkout_root / runtime_relative).read_bytes()).hexdigest()
    if current_runtime_sha256 != TEST_CODE_HASHES[runtime_relative]:
        with pytest.raises(
            formal_launcher.LaunchError,
            match="shadow-validated code hash changed",
        ):
            formal_launcher._validate_shadow_contract(paths)
        return

    proof = formal_launcher._validate_shadow_contract(paths)

    assert proof["contract"]["schema_version"] == formal_launcher.SHADOW_CONTRACT_SCHEMA
    assert proof["contract"]["sha256"] == formal_launcher.SHADOW_CONTRACT_SHA256
    assert len(proof["validated_code_sha256"]) == formal_launcher.SHADOW_VALIDATED_CODE_COUNT
    assert proof["generation_source"]["worktree_clean"] is True
    evidence = proof["reports"]["evidence_live"]
    assert evidence["sha256"] == formal_launcher.EVIDENCE_SHADOW_SHA256
    assert evidence["decisions"] >= 1_000
    assert evidence["semantic_censored_transitions"] == 0
    assert evidence["prefer_targets"] == 0
    assert evidence["training_state_unchanged"] is True
    assert evidence["completion_staging_is_bounded"] is True
    assert 0 <= evidence["maximum_episode_completion_storage_nbytes"] <= 134_217_728
    actor = proof["reports"]["evidence_actor"]
    assert actor["sha256"] == formal_launcher.ACTOR_EVIDENCE_SHADOW_SHA256
    assert actor["cases"] == 6
    assert actor["actor_actionable_records"] == 4
    assert actor["risk_sequence_records"] == 5
    assert actor["prefer_targets"] == 0
    assert actor["learner_mask_dry_run_passed"] is True


def test_reviewed_payload_is_model_initialization_only(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    paths = launcher.validate_layout(
        _paths(tmp_path),
        enforce_active_root=False,
    )
    _patch_payload_authorities(monkeypatch)
    payload = _payload(paths)
    assert launcher._validate_preflight_payload(payload, paths=paths) == payload
    assert "--resume" not in payload["trainer_command"]
    assert payload["trainer_command"].count("--initialize-from") == 1

    changed = {
        **payload,
        "trainer_command": [
            *payload["trainer_command"],
            "--resume",
            "bad",
        ],
    }
    with pytest.raises(
        launcher.LaunchError,
        match=r"never masquerade|differs",
    ):
        launcher._validate_preflight_payload(changed, paths=paths)

    changed = {
        **payload,
        "initialization": {
            **payload["initialization"],
            "checkpoint": str(tmp_path / "v27-periodic-step-000200044"),
        },
    }
    with pytest.raises(launcher.LaunchError, match="checkpoint changed"):
        launcher._validate_preflight_payload(changed, paths=paths)

    changed = {
        **payload,
        "initialization": {
            **payload["initialization"],
            "migration_proof": {
                **payload["initialization"]["migration_proof"],
                "fresh_liveness_head_tensors": 11,
            },
        },
    }
    with pytest.raises(launcher.LaunchError, match="liveness"):
        launcher._validate_preflight_payload(changed, paths=paths)


def test_preflight_payload_validation_reopens_runtime_readiness_report(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    paths = launcher.validate_layout(
        _paths(tmp_path),
        enforce_active_root=False,
    )
    payload = _payload(paths)
    monkeypatch.setattr(
        launcher,
        "_validate_shadow_contract",
        lambda _paths: SHADOW,
    )
    calls: list[tuple[Any, Any, Any]] = []

    def reopen_readiness(
        observed_paths: Any,
        *,
        runtime: Any,
        shadow: Any,
    ) -> dict[str, Any]:
        calls.append((observed_paths, runtime, shadow))
        return payload["runtime_readiness_evidence"]

    monkeypatch.setattr(
        launcher,
        "_validate_runtime_readiness_evidence",
        reopen_readiness,
    )
    assert (
        launcher._validate_preflight_payload(
            payload,
            paths=paths,
            verify_live_shadow=True,
        )
        == payload
    )
    assert calls == [(paths, payload["runtime"], SHADOW)]

    changed_readiness = json.loads(
        json.dumps(payload["runtime_readiness_evidence"]),
    )
    changed_readiness["report_binding"]["status"] = "changed"
    monkeypatch.setattr(
        launcher,
        "_validate_runtime_readiness_evidence",
        lambda _paths, *, runtime, shadow: changed_readiness,
    )
    with pytest.raises(
        launcher.LaunchError,
        match="runtime readiness report changed",
    ):
        launcher._validate_preflight_payload(
            payload,
            paths=paths,
            verify_live_shadow=True,
        )


def test_pre_spawn_reproof_rejects_trainer_environment_drift(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    paths = launcher.validate_layout(
        _paths(tmp_path),
        enforce_active_root=False,
    )
    reviewed = {
        **_payload(paths),
        "selection": _proof(paths),
        "launch_id": "99999999-9999-4999-8999-999999999999",
    }
    monkeypatch.setattr(
        launcher,
        "run_preflight",
        lambda _paths: reviewed,
    )
    mutations: tuple[Callable[[dict[str, Any]], None], ...] = (
        lambda value: value["set"].__setitem__(
            "LD_PRELOAD",
            "/tmp/injected.so",
        ),
        lambda value: value["set"].__setitem__(
            "OMP_NUM_THREADS",
            "8",
        ),
        lambda value: value["set"].__setitem__(
            "PATH",
            value["set"]["PATH"] + ":/unreviewed",
        ),
        lambda value: value.__setitem__(
            "unset",
            list(reversed(value["unset"])),
        ),
        lambda value: value["set"].__delitem__("PYTHONPATH"),
    )
    for mutate in mutations:
        manifest = json.loads(json.dumps(reviewed))
        mutate(manifest["trainer_environment"])
        with pytest.raises(
            launcher.LaunchError,
            match="trainer_environment",
        ):
            launcher._reprove_launch_contract(
                paths,
                manifest,
            )


def test_v29_trainer_environment_is_idempotent_across_supervisor(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    paths = launcher.validate_layout(
        _paths(tmp_path),
        enforce_active_root=False,
    )
    first = launcher.v29_preflight.build_trainer_environment(
        paths.preflight,
        base_environment={
            "PATH": "/first-process/bin",
            "LD_PRELOAD": "/first-process/lib.so",
            "OMP_NUM_THREADS": "99",
        },
    )
    second = launcher.v29_preflight.build_trainer_environment(
        paths.preflight,
        base_environment={
            "PATH": "/second-process/bin",
            "LD_LIBRARY_PATH": "/second-process/lib",
            "PYTHONHOME": "/second-process/python",
        },
    )
    assert second == first
    assert set(first) == set(
        launcher._V29_HERMETIC_TRAINER_ENVIRONMENT_KEYS,
    )
    assert first["LC_CTYPE"] == "C.UTF-8"
    assert first["OMP_NUM_THREADS"] == "4"
    assert first["PATH"] == launcher.os.pathsep.join(
        (
            str(paths.preflight.venv_python.parent),
            *launcher._V29_HERMETIC_SYSTEM_PATH,
        )
    )
    assert "LD_PRELOAD" not in first
    assert "LD_LIBRARY_PATH" not in first
    assert launcher._exact_trainer_environment_sha256(second) == (launcher._exact_trainer_environment_sha256(first))

    # Reconstructing the reviewed contract happens in ``start`` and later in
    # the detached supervisor.  Ambient drift in either invocation must not
    # enter the process environment or change its exact digest.
    contract = launcher.v29_preflight.trainer_environment_contract(first)
    monkeypatch.setenv("LD_PRELOAD", "/ambient/after-preflight.so")
    monkeypatch.setenv("PATH", "/ambient/after-preflight/bin")
    reconstructed = launcher._environment_from_contract(contract)
    assert reconstructed == first
    assert launcher._exact_trainer_environment_sha256(reconstructed) == (
        launcher._exact_trainer_environment_sha256(first)
    )


def test_v29_launcher_and_trainer_environment_contracts_have_exact_parity(
    tmp_path: Path,
) -> None:
    paths = launcher.validate_layout(
        _paths(tmp_path),
        enforce_active_root=False,
    )
    environment = launcher.v29_preflight.build_trainer_environment(
        paths.preflight,
    )
    launcher_contract = launcher.v29_preflight.trainer_environment_contract(
        environment,
    )

    assert launcher._V29_HERMETIC_TRAINER_ENVIRONMENT_KEYS == (
        trainer_launch_contract.V29_HERMETIC_TRAINER_ENVIRONMENT_KEYS
    )
    assert launcher._V29_HERMETIC_TRAINER_ENVIRONMENT_UNSET_KEYS == (
        trainer_launch_contract.V29_HERMETIC_TRAINER_ENVIRONMENT_UNSET_KEYS
    )
    assert tuple(environment) == (trainer_launch_contract.V29_HERMETIC_TRAINER_ENVIRONMENT_KEYS)
    assert tuple(launcher_contract["set"]) == (trainer_launch_contract.V29_HERMETIC_TRAINER_ENVIRONMENT_KEYS)
    assert launcher_contract["unset"] == list(
        trainer_launch_contract.V29_HERMETIC_TRAINER_ENVIRONMENT_UNSET_KEYS,
    )
    assert launcher._exact_trainer_environment_sha256(environment) == (
        trainer_launch_contract.trainer_environment_sha256(environment)
    )


def test_v29_fixed_runtime_probe_uses_only_hermetic_environment_and_command(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    paths = launcher.validate_layout(
        _paths(tmp_path),
        enforce_active_root=False,
    )
    environment = launcher.v29_preflight.build_trainer_environment(
        paths.preflight,
    )
    expected = _payload(paths)["runtime"]
    observed: dict[str, Any] = {}

    def fake_run(command: tuple[str, ...], **kwargs: Any) -> Any:
        observed["command"] = command
        observed.update(kwargs)
        return 0, json.dumps(expected).encode("utf-8"), b""

    monkeypatch.setattr(
        launcher,
        "_run_bounded_runtime_probe_process",
        fake_run,
    )
    assert (
        launcher._verify_runtime_inputs(
            paths.preflight,
            environment=environment,
        )
        == expected
    )
    assert observed["command"] == launcher._hermetic_runtime_probe_command(
        paths.preflight,
    )
    assert observed["cwd"] == paths.package_root
    assert observed["environment"] == environment
    assert set(observed["environment"]) == set(
        launcher._V29_HERMETIC_TRAINER_ENVIRONMENT_KEYS,
    )
    assert observed["timeout_seconds"] == (launcher.HERMETIC_RUNTIME_PROBE_TIMEOUT_SECONDS)
    assert observed["maximum_output_bytes"] == (launcher.HERMETIC_RUNTIME_PROBE_MAX_OUTPUT_BYTES)


@pytest.mark.parametrize(
    ("process_result", "message"),
    (
        ((17, b"", b"probe failed"), r"probe failed \(17\)"),
        ((0, b"{", b""), "malformed JSON"),
        ((0, b"[]", b""), "not an object"),
    ),
)
def test_v29_runtime_probe_rejects_nonzero_malformed_and_non_object(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    process_result: tuple[int, bytes, bytes],
    message: str,
) -> None:
    paths = launcher.validate_layout(
        _paths(tmp_path),
        enforce_active_root=False,
    )
    environment = launcher.v29_preflight.build_trainer_environment(
        paths.preflight,
    )
    monkeypatch.setattr(
        launcher,
        "_run_bounded_runtime_probe_process",
        lambda *args, **kwargs: process_result,
    )
    with pytest.raises(launcher.LaunchError, match=message):
        launcher._verify_runtime_inputs(
            paths.preflight,
            environment=environment,
        )


@pytest.mark.skipif(
    launcher.os.name != "posix",
    reason="process-group cleanup is the Linux/WSL production path",
)
def test_bounded_runtime_probe_timeout_kills_leader_and_descendant_group(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    marker = tmp_path / "descendant.pid"
    program = """
import subprocess, sys, time
child = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(30)"])
with open(sys.argv[1], "w", encoding="ascii") as handle:
    handle.write(str(child.pid))
    handle.flush()
time.sleep(30)
"""
    real_popen = launcher.subprocess.Popen
    observed: list[Any] = []

    def recording_popen(*args: Any, **kwargs: Any) -> Any:
        process = real_popen(*args, **kwargs)
        observed.append(process)
        return process

    monkeypatch.setattr(launcher.subprocess, "Popen", recording_popen)
    with pytest.raises(launcher.LaunchError, match="exceeded"):
        launcher._run_bounded_runtime_probe_process(
            (sys.executable, "-c", program, str(marker)),
            cwd=tmp_path,
            environment=dict(launcher.os.environ),
            timeout_seconds=0.25,
            maximum_output_bytes=4096,
        )
    assert len(observed) == 1
    assert observed[0].poll() is not None
    assert not launcher._runtime_probe_group_exists(observed[0].pid)
    descendant_pid = int(marker.read_text(encoding="ascii"))
    deadline = launcher.time.monotonic() + 2.0
    while (Path("/proc") / str(descendant_pid)).exists() and (launcher.time.monotonic() < deadline):
        launcher.time.sleep(0.01)
    assert not (Path("/proc") / str(descendant_pid)).exists()


@pytest.mark.skipif(
    launcher.os.name != "posix",
    reason="pipe selectors are the Linux/WSL production path",
)
def test_bounded_runtime_probe_overflow_terminates_and_reaps_immediately(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    real_popen = launcher.subprocess.Popen
    observed: list[Any] = []

    def recording_popen(*args: Any, **kwargs: Any) -> Any:
        process = real_popen(*args, **kwargs)
        observed.append(process)
        return process

    monkeypatch.setattr(launcher.subprocess, "Popen", recording_popen)
    with pytest.raises(launcher.LaunchError, match="stdout exceeded"):
        launcher._run_bounded_runtime_probe_process(
            (
                sys.executable,
                "-c",
                "import os, time; os.write(1, b'x' * 8192); time.sleep(30)",
            ),
            cwd=tmp_path,
            environment=dict(launcher.os.environ),
            timeout_seconds=2.0,
            maximum_output_bytes=1024,
        )
    assert len(observed) == 1
    assert observed[0].poll() is not None
    assert not launcher._runtime_probe_group_exists(observed[0].pid)


@pytest.mark.skipif(
    launcher.os.name != "posix",
    reason="process-group cleanup is the Linux/WSL production path",
)
def test_bounded_runtime_probe_success_reaps_session_without_false_positive(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    real_popen = launcher.subprocess.Popen
    observed: list[Any] = []

    def recording_popen(*args: Any, **kwargs: Any) -> Any:
        process = real_popen(*args, **kwargs)
        observed.append(process)
        return process

    monkeypatch.setattr(launcher.subprocess, "Popen", recording_popen)
    result = launcher._run_bounded_runtime_probe_process(
        (
            sys.executable,
            "-c",
            "import os; os.write(1, b'{\"ok\":true}'); os.write(2, b'note')",
        ),
        cwd=tmp_path,
        environment=dict(launcher.os.environ),
        timeout_seconds=2.0,
        maximum_output_bytes=4096,
    )
    assert result == (0, b'{"ok":true}', b"note")
    assert len(observed) == 1
    assert observed[0].poll() == 0
    assert not launcher._runtime_probe_group_exists(observed[0].pid)


@pytest.mark.skipif(
    launcher.os.name != "posix",
    reason="process-group cleanup is the Linux/WSL production path",
)
def test_selector_construction_failure_still_closes_pipes_and_reaps_leader(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    real_popen = launcher.subprocess.Popen
    observed: list[Any] = []

    def recording_popen(*args: Any, **kwargs: Any) -> Any:
        process = real_popen(*args, **kwargs)
        observed.append(process)
        return process

    def exploding_selector() -> Any:
        raise RuntimeError("selector construction failed")

    monkeypatch.setattr(launcher.subprocess, "Popen", recording_popen)
    monkeypatch.setattr(
        launcher.selectors,
        "DefaultSelector",
        exploding_selector,
    )
    with pytest.raises(RuntimeError, match="selector construction failed"):
        launcher._run_bounded_runtime_probe_process(
            (sys.executable, "-c", "import time; time.sleep(30)"),
            cwd=tmp_path,
            environment=dict(launcher.os.environ),
            timeout_seconds=2.0,
            maximum_output_bytes=4096,
        )
    assert len(observed) == 1
    assert observed[0].returncode is not None
    assert observed[0].stdout.closed
    assert observed[0].stderr.closed
    assert not (Path("/proc") / str(observed[0].pid)).exists()


@pytest.mark.skipif(
    launcher.os.name != "posix",
    reason="resource cleanup fault injection is the Linux/WSL production path",
)
def test_selector_and_stream_close_failures_cannot_bypass_group_cleanup(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    close_calls: list[str] = []
    observed: list[Any] = []
    real_popen = launcher.subprocess.Popen
    real_selector = launcher.selectors.DefaultSelector

    class FailingCloseStream:
        def __init__(self, stream: Any, label: str) -> None:
            self._stream = stream
            self._label = label

        def fileno(self) -> int:
            return int(self._stream.fileno())

        def close(self) -> None:
            close_calls.append(self._label)
            self._stream.close()
            raise OSError(f"{self._label} close failed")

    class FailingCloseSelector:
        def __init__(self) -> None:
            self._selector = real_selector()

        def register(self, *args: Any, **kwargs: Any) -> Any:
            return self._selector.register(*args, **kwargs)

        def unregister(self, *args: Any, **kwargs: Any) -> Any:
            return self._selector.unregister(*args, **kwargs)

        def select(self, *args: Any, **kwargs: Any) -> Any:
            return self._selector.select(*args, **kwargs)

        def close(self) -> None:
            close_calls.append("selector")
            self._selector.close()
            raise OSError("selector close failed")

    def wrapped_popen(*args: Any, **kwargs: Any) -> Any:
        process = real_popen(*args, **kwargs)
        process.stdout = FailingCloseStream(process.stdout, "stdout")
        process.stderr = FailingCloseStream(process.stderr, "stderr")
        observed.append(process)
        return process

    monkeypatch.setattr(launcher.subprocess, "Popen", wrapped_popen)
    monkeypatch.setattr(
        launcher.selectors,
        "DefaultSelector",
        FailingCloseSelector,
    )
    with pytest.raises(
        launcher.LaunchError,
        match="resource cleanup failed",
    ):
        launcher._run_bounded_runtime_probe_process(
            (sys.executable, "-c", "import os; os.write(1, b'{}')"),
            cwd=tmp_path,
            environment=dict(launcher.os.environ),
            timeout_seconds=2.0,
            maximum_output_bytes=4096,
        )
    assert close_calls == ["selector", "stdout", "stderr"]
    assert observed[0].returncode == 0
    assert not (Path("/proc") / str(observed[0].pid)).exists()


@pytest.mark.skipif(
    launcher.os.name != "posix",
    reason="signal-before-reap ordering is the Linux/WSL production path",
)
def test_success_cleanup_signals_live_descendant_before_reaping_leader(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    program = """
import os, subprocess, sys
child = subprocess.Popen(
    [sys.executable, "-c", "import time; time.sleep(30)"],
    stdin=subprocess.DEVNULL,
    stdout=subprocess.DEVNULL,
    stderr=subprocess.DEVNULL,
    close_fds=True,
)
os.write(1, str(child.pid).encode("ascii"))
"""
    events: list[tuple[str, int]] = []
    real_popen = launcher.subprocess.Popen
    real_signal_group = launcher._signal_runtime_probe_group

    class RecordingProcess:
        def __init__(self, process: Any) -> None:
            self._process = process

        def __getattr__(self, name: str) -> Any:
            return getattr(self._process, name)

        def wait(self, *args: Any, **kwargs: Any) -> int:
            events.append(("wait", self.pid))
            return int(self._process.wait(*args, **kwargs))

    def wrapped_popen(*args: Any, **kwargs: Any) -> Any:
        return RecordingProcess(real_popen(*args, **kwargs))

    def recording_signal(process_group_id: int, signal_number: int) -> None:
        events.append(("signal", signal_number))
        real_signal_group(process_group_id, signal_number)

    monkeypatch.setattr(launcher.subprocess, "Popen", wrapped_popen)
    monkeypatch.setattr(
        launcher,
        "_signal_runtime_probe_group",
        recording_signal,
    )
    returncode, stdout, stderr = launcher._run_bounded_runtime_probe_process(
        (sys.executable, "-c", program),
        cwd=tmp_path,
        environment=dict(launcher.os.environ),
        timeout_seconds=2.0,
        maximum_output_bytes=4096,
    )
    assert (returncode, stderr) == (0, b"")
    assert int(stdout.decode("ascii")) > 0
    wait_positions = [index for index, event in enumerate(events) if event[0] == "wait"]
    signal_positions = [index for index, event in enumerate(events) if event[0] == "signal"]
    assert len(wait_positions) == 1
    assert signal_positions
    assert max(signal_positions) < wait_positions[0]
    assert wait_positions[0] == len(events) - 1


@pytest.mark.skipif(
    launcher.os.name != "posix",
    reason="zombie process-group inspection requires Linux /proc",
)
def test_zombie_only_runtime_probe_group_has_no_live_members_or_signal(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    process = launcher.subprocess.Popen(
        (sys.executable, "-c", "pass"),
        stdin=launcher.subprocess.DEVNULL,
        stdout=launcher.subprocess.DEVNULL,
        stderr=launcher.subprocess.DEVNULL,
        start_new_session=True,
    )
    stat_path = Path("/proc") / str(process.pid) / "stat"
    deadline = launcher.time.monotonic() + 2.0
    state = ""
    while launcher.time.monotonic() < deadline:
        stat = stat_path.read_text(encoding="utf-8")
        state = stat.rpartition(")")[2].strip().split()[0]
        if state == "Z":
            break
        launcher.time.sleep(0.01)
    assert state == "Z"
    assert launcher._runtime_probe_live_group_members(process.pid) == ()

    signals: list[tuple[int, int]] = []
    monkeypatch.setattr(
        launcher,
        "_signal_runtime_probe_group",
        lambda process_group_id, signal_number: signals.append(
            (process_group_id, signal_number),
        ),
    )
    try:
        assert (
            launcher._terminate_and_reap_runtime_probe_group(
                process,
                process_group_id=process.pid,
            )
            == 0
        )
    finally:
        if process.returncode is None:
            process.wait(timeout=2.0)
    assert signals == []
    assert not stat_path.exists()


def test_internal_runtime_probe_cli_rejects_every_extra_option(
    capsys: pytest.CaptureFixture[str],
) -> None:
    with pytest.raises(SystemExit):
        launcher.build_parser().parse_args(
            [launcher.HERMETIC_RUNTIME_PROBE_ACTION, "--command", "bad"],
        )
    assert (
        launcher.main(
            [
                launcher.HERMETIC_RUNTIME_PROBE_ACTION,
                "--manifest",
                "/tmp/not-accepted.json",
            ]
        )
        == 2
    )
    assert "does not accept --manifest" in capsys.readouterr().err


def test_v29_runtime_probe_binding_rejects_environment_command_and_result_drift(
    tmp_path: Path,
) -> None:
    paths = launcher.validate_layout(
        _paths(tmp_path),
        enforce_active_root=False,
    )
    reviewed = _payload(paths)
    mutations: tuple[Callable[[dict[str, Any]], None], ...] = (
        lambda value: value["runtime"].__setitem__(
            "trainer_environment_sha256",
            "0" * 64,
        ),
        lambda value: value["runtime"]["probe_command"].append(
            "--arbitrary-command",
        ),
        lambda value: value["runtime"].__setitem__(
            "result_sha256",
            "0" * 64,
        ),
    )
    for mutate in mutations:
        changed = json.loads(json.dumps(reviewed))
        mutate(changed)
        with pytest.raises(launcher.LaunchError, match="runtime probe"):
            launcher._validate_preflight_payload(
                changed,
                paths=paths,
                verify_live_shadow=False,
            )


def test_exact_environment_digest_is_checked_when_reports_are_not_reopened(
    tmp_path: Path,
) -> None:
    paths = launcher.validate_layout(
        _paths(tmp_path),
        enforce_active_root=False,
    )
    changed = _payload(paths)
    changed["trainer_environment_sha256"] = "0" * 64
    changed["runtime"]["trainer_environment_sha256"] = "0" * 64
    with pytest.raises(
        launcher.LaunchError,
        match="exact trainer environment changed",
    ):
        launcher._validate_preflight_payload(
            changed,
            paths=paths,
            verify_live_shadow=False,
        )


def test_v29_launcher_rejects_non_hermetic_environment_extras(
    tmp_path: Path,
) -> None:
    paths = launcher.validate_layout(
        _paths(tmp_path),
        enforce_active_root=False,
    )
    environment = launcher.v29_preflight.build_trainer_environment(
        paths.preflight,
    )
    environment["LD_PRELOAD"] = "/unreviewed/lib.so"

    with pytest.raises(
        launcher.PreflightError,
        match="hermetic trainer environment keys changed",
    ):
        launcher.v29_preflight.validate_trainer_environment(
            environment,
            paths=paths.preflight,
        )
    with pytest.raises(
        launcher.PreflightError,
        match="contract keys changed",
    ):
        launcher.v29_preflight.trainer_environment_contract(environment)
    with pytest.raises(
        launcher.LaunchError,
        match="exact v29 hermetic set",
    ):
        launcher._exact_trainer_environment_sha256(environment)


def test_v29_hermetic_environment_survives_python_exec_as_exact_set(
    tmp_path: Path,
) -> None:
    paths = launcher.validate_layout(
        _paths(tmp_path),
        enforce_active_root=False,
    )
    environment = launcher.v29_preflight.build_trainer_environment(
        paths.preflight,
    )
    result = launcher.subprocess.run(
        (
            sys.executable,
            "-c",
            "import json, os; print(json.dumps(dict(os.environ), sort_keys=True))",
        ),
        env=environment,
        check=False,
        capture_output=True,
        text=True,
        encoding="utf-8",
    )
    assert result.returncode == 0, result.stderr
    assert json.loads(result.stdout) == environment


@pytest.mark.skipif(
    launcher.os.name != "posix",
    reason="READY/GO bootstrap exec integration is Linux-specific",
)
def test_ready_go_bootstrap_exec_preserves_exact_v29_ten_key_environment(
    tmp_path: Path,
) -> None:
    paths = launcher.validate_layout(
        _paths(tmp_path),
        enforce_active_root=False,
    )
    environment = launcher.v29_preflight.build_trainer_environment(
        paths.preflight,
    )
    marker = tmp_path / "exec-environment.json"
    gate_read, gate_write = launcher.os.pipe()
    status_read, status_write = launcher.os.pipe()
    nonce = "a" * 64
    trainer_program = (
        "import json, os, sys; json.dump(dict(os.environ), open(sys.argv[1], 'w', encoding='utf-8'), sort_keys=True)"
    )
    command = (
        sys.executable,
        str(PACKAGE_ROOT / "scripts/supervised_trainer_bootstrap.py"),
        "--expected-parent-pid",
        str(launcher.os.getpid()),
        "--expected-parent-start-ticks",
        str(
            launcher._trainer_bootstrap.process_start_ticks(
                launcher.os.getpid(),
            )
        ),
        "--gate-fd",
        str(gate_read),
        "--status-fd",
        str(status_write),
        "--nonce",
        nonce,
        "--timeout-seconds",
        "2",
        "--",
        sys.executable,
        "-c",
        trainer_program,
        str(marker),
    )
    process = launcher.subprocess.Popen(
        command,
        env=environment,
        stdin=launcher.subprocess.DEVNULL,
        stdout=launcher.subprocess.PIPE,
        stderr=launcher.subprocess.PIPE,
        close_fds=True,
        pass_fds=(gate_read, status_write),
    )
    launcher.os.close(gate_read)
    launcher.os.close(status_write)
    try:
        ready = launcher._trainer_bootstrap.read_bounded_frame(
            status_read,
            2.0,
        )
        assert ready["kind"] == launcher._trainer_bootstrap.READY_KIND
        launcher.os.write(
            gate_write,
            launcher._trainer_bootstrap.canonical_frame(
                {
                    "kind": launcher._trainer_bootstrap.GO_KIND,
                    "nonce": nonce,
                    "protocol": launcher.TRAINER_BOOTSTRAP_PROTOCOL,
                }
            ),
        )
        launcher.os.close(gate_write)
        gate_write = -1
        stdout, stderr = process.communicate(timeout=5.0)
        assert process.returncode == 0, (stdout, stderr)
    finally:
        if gate_write >= 0:
            launcher.os.close(gate_write)
        launcher.os.close(status_read)
        if process.poll() is None:
            process.kill()
            process.wait(timeout=2.0)
    observed = json.loads(marker.read_text(encoding="utf-8"))
    assert observed == environment
    assert len(observed) == 10
    assert tuple(observed) == tuple(sorted(environment))
    assert set(observed) == set(
        trainer_launch_contract.V29_HERMETIC_TRAINER_ENVIRONMENT_KEYS,
    )


def _make_shadow_fixture(
    paths: Any,
    monkeypatch: pytest.MonkeyPatch,
) -> tuple[Path, Path, Path, Path]:
    code = paths.checkout_root / "packages/rl-agent/sts2_rl/semantics.py"
    code.parent.mkdir(parents=True, exist_ok=True)
    code.write_text("SEMANTICS = 1\n", encoding="utf-8")
    generation_source = {
        "checkout_root": str(paths.checkout_root),
        "git_object_format": "sha1",
        "implementation_commit": "1" * 40,
        "implementation_tree": "2" * 40,
        "worktree_clean": True,
    }
    validator_paths: dict[str, Path] = {}
    for key, relative in launcher.FORMAL_REPORT_VALIDATORS.items():
        if key == "runtime_readiness":
            continue
        validator = paths.checkout_root / relative
        validator.parent.mkdir(parents=True, exist_ok=True)
        validator.write_text(f"# {key} validator\n", encoding="utf-8")
        validator_paths[key] = validator
    report_root = paths.artifact_root / "reports/failure-credit-v4"
    report_root.mkdir(parents=True, exist_ok=True)
    semantics_path = report_root / "semantic.json"
    evidence_path = report_root / "evidence.json"
    actor_evidence_path = report_root / "actor-evidence.json"
    semantics = {
        "version": "sts2-semantics-shadow-report-v2",
        "status": "passed",
        "read_only": True,
        "training_authority": False,
        "errors": [],
        "counts": {
            "parsed_records": 500_000,
            "rich_decision_snapshots": 3_986,
        },
        "gates": {
            "minimum_rich_records": True,
            "no_contract_errors": True,
            "unknown_transition_budget": True,
            "required_minimum_rich_records": 1_000,
            "maximum_unknown_transitions": 0,
            "observed_unknown_transitions": 0,
        },
        "semantic_manifest": {
            "namespace": "surface_registry_manifest",
            "schema_version": "sts2-surface-registry-v1",
            "digest": ("ff7b2e40a2aeb6cd29fd5db5ea9f42998a285deadc175d7a2480de50a0a97bf8"),
            "payload": {
                "contract_version": "sts2-surface-registry-v1",
            },
        },
    }
    evidence = {
        "version": "sts2-failure-evidence-shadow-report-v2",
        "status": "passed",
        "read_only": True,
        "training_authority": False,
        "gates": {
            "completion_staging_is_bounded": True,
            "minimum_decisions": True,
            "no_replay_or_learner_updates": True,
            "no_unjustified_completion_prefer": True,
            "source_manifest_unchanged": True,
            "source_metadata_unchanged": True,
            "zero_semantic_censored_transitions": True,
        },
        "training_state_unchanged": {
            "collector_model": True,
            "episodic_replay": True,
            "model": True,
            "optimizer": True,
            "rollout_queue": True,
            "transaction_replay": True,
        },
        "counts": {
            "decisions": 1_000,
            "semantic_censored_transitions": 0,
            "maximum_episode_completion_storage_nbytes": 1_024,
        },
        "direct_policy_targets": {},
        "episodes": [
            {
                "shadow": {
                    "completion_controls": 2,
                    "completion_controls_observed": 3,
                    "completion_controls_dropped": 1,
                    "completion_storage_nbytes": 1_024,
                },
            },
        ],
        "source": {
            "checkpoint": str(paths.initialization_checkpoint),
            "checkpoint_id": launcher.FIXED_INITIALIZATION_CHECKPOINT_ID,
            "environment_steps": launcher.FIXED_INITIALIZATION_STEP,
            "policy_version": (launcher.FIXED_INITIALIZATION_POLICY_VERSION),
            "manifest_sha256": (launcher.FIXED_INITIALIZATION_MANIFEST_SHA256),
            "metadata_sha256": (launcher.FIXED_INITIALIZATION_METADATA_SHA256),
        },
        "failure_credit_abi": {
            "schema": launcher.FAILURE_CREDIT_SCHEMA_V4,
            "collector": "sts2-failure-credit-collector-v3",
            "detector": "sts2-semantic-macro-cycle-detector-v3",
            "compiler": "sts2-failure-credit-compiler-v1",
            "replay": launcher.FAILURE_CREDIT_REPLAY_V4,
        },
    }
    mask_counts = {
        "effective_direct_rows": 0,
        "effective_risk_rows": 0,
        "effective_cycle_groups": 0,
        "effective_contrast_groups": 0,
    }

    def mask_summary(**overrides: int) -> dict[str, Any]:
        return {
            "mask_counts": {
                **mask_counts,
                **overrides,
            },
        }

    actor_cases: dict[str, dict[str, Any]] = {}
    for case in (
        "abandoned_cycle_suffix",
        "censored_boundary",
        "forced_only_stall",
        "linger9_death_warning_direct",
        "room_full_of_cheese_multi_edge",
        "unique_unresolved_stall",
    ):
        actor_cases[case] = {
            "calibration_update_0": mask_summary(),
            "mature_risk_start": mask_summary(),
        }
    for phase in ("calibration_update_0", "mature_risk_start"):
        actor_cases["linger9_death_warning_direct"][phase] = mask_summary(effective_direct_rows=1)
        actor_cases["room_full_of_cheese_multi_edge"][phase] = mask_summary(effective_cycle_groups=1)
    actor_cases["unique_unresolved_stall"]["mature_risk_start"] = mask_summary(effective_risk_rows=6)
    actor_evidence = {
        "version": "sts2-failure-actor-evidence-shadow-report-v3",
        "status": "passed",
        "read_only": True,
        "training_authority": False,
        "fixture_authority": "reviewed-contract-cases-only",
        "abi": {
            "schema": launcher.FAILURE_CREDIT_SCHEMA_V4,
            "collector": "sts2-failure-credit-collector-v3",
            "detector": "sts2-semantic-macro-cycle-detector-v3",
            "compiler": "sts2-failure-credit-compiler-v1",
        },
        "gates": {
            "abandoned_cycle_has_zero_direct_or_cycle_blame": True,
            "all_contexts_bounded": True,
            "censored_has_zero_learning_targets": True,
            "direct_avoid_nonzero": True,
            "forced_only_has_zero_actor_blame": True,
            "learner_mask_dry_run_passed": True,
            "multi_edge_cycle_nonzero": True,
            "no_generic_prefer_target": True,
            "no_last_action_fallback": True,
            "unresolved_stall_nonzero": True,
        },
        "counts": {
            "cases": 6,
            "actor_actionable_records": 4,
            "risk_sequence_records": 5,
            "strata": {
                "CENSORED": 1,
                "DIRECT_WITNESS": 1,
                "MULTI_EDGE_CYCLE": 1,
                "RISK_SEQUENCE": 5,
                "UNRESOLVED_STALL": 3,
            },
            "direct_targets": {"avoid": 1},
        },
        "learner_mask_dry_run": {
            "batch_manifests": {},
            "case_manifests": actor_cases,
            "checks": {
                "learner_abandoned_cycle_has_no_effective_direct_or_cycle_blame": True,
                "learner_direct_witness_effective_during_and_after_calibration": True,
                "learner_forced_and_censored_have_zero_effective_actor": True,
                "learner_multi_edge_cycle_effective_during_and_after_calibration": True,
                "learner_phase_boundary_correct": True,
                "learner_unique_risk_effective_at_mature_start": True,
                "learner_unique_risk_suppressed_during_calibration": True,
            },
            "config": {
                "policy_gradient_max_lag": 128,
                "liveness_head_calibration_updates": 256,
                "liveness_risk_actor_start_update": 512,
            },
            "current_policy_version": 11,
            "interface": ("sts2_rl.training.learner.compile_liveness_label_manifest"),
            "model_constructed": False,
            "optimizer_constructed": False,
            "phase_updates": {
                "calibration_update_0": 0,
                "mature_risk_start": 512,
            },
            "pure_dry_run": True,
            "replay_constructed": False,
        },
    }
    for key, report in {
        "semantics_historical": semantics,
        "evidence_live": evidence,
        "evidence_actor": actor_evidence,
    }.items():
        relative = launcher.FORMAL_REPORT_VALIDATORS[key]
        report["generation_source"] = {
            **generation_source,
            "schema_version": launcher.FORMAL_REPORT_GENERATION_SOURCE_SCHEMA,
            "validator_relative_path": relative,
            "validator_sha256": _sha256(validator_paths[key]),
        }
    semantics_path.write_text(json.dumps(semantics), encoding="utf-8")
    evidence_path.write_text(json.dumps(evidence), encoding="utf-8")
    actor_evidence_path.write_text(
        json.dumps(actor_evidence),
        encoding="utf-8",
    )
    contract = {
        "name": "v29-failure-credit-v4-bootstrap",
        "reports": {
            "semantics_historical": {
                "maximum_errors": 0,
                "maximum_unknown_transitions": 0,
                "minimum_parsed_records": 500_000,
                "minimum_rich_decision_snapshots": 3_986,
                "relative_path": ("reports/failure-credit-v4/semantic.json"),
                "sha256": _sha256(semantics_path),
                "status": "passed",
                "version": "sts2-semantics-shadow-report-v2",
            },
            "evidence_live": {
                "maximum_episode_completion_bytes": 134_217_728,
                "maximum_episode_completion_controls": 32,
                "maximum_prefer_targets": 0,
                "maximum_semantic_censored_transitions": 0,
                "minimum_decisions": 1_000,
                "relative_path": ("reports/failure-credit-v4/evidence.json"),
                "require_all_gates": True,
                "require_no_training_authority": True,
                "require_read_only": True,
                "sha256": _sha256(evidence_path),
                "source_checkpoint_id": (launcher.FIXED_INITIALIZATION_CHECKPOINT_ID),
                "status": "passed",
                "version": "sts2-failure-evidence-shadow-report-v2",
            },
            "evidence_actor": {
                "maximum_prefer_targets": 0,
                "minimum_actor_actionable_records": 4,
                "minimum_cases": 6,
                "minimum_risk_sequence_records": 5,
                "relative_path": ("reports/failure-credit-v4/actor-evidence.json"),
                "require_all_gates": True,
                "require_no_training_authority": True,
                "require_read_only": True,
                "sha256": _sha256(actor_evidence_path),
                "status": "passed",
                "version": ("sts2-failure-actor-evidence-shadow-report-v3"),
            },
        },
        "schema_version": launcher.SHADOW_CONTRACT_SCHEMA,
        "source_checkpoint_contract": (launcher.FIXED_FROZEN_CONTRACT_NAME),
        "training_authority": False,
        "validated_code_sha256": {
            str(code.relative_to(paths.checkout_root)).replace("\\", "/"): _sha256(code),
            **{
                relative: _sha256(path)
                for relative, path in (
                    (launcher.FORMAL_REPORT_VALIDATORS[key], validator_paths[key]) for key in validator_paths
                )
            },
        },
        "generation_source": generation_source,
    }
    contract_path = paths.checkout_root / "contracts/shadow-validation/v29-test.json"
    contract_path.parent.mkdir(parents=True, exist_ok=True)
    contract_path.write_text(json.dumps(contract), encoding="utf-8")
    monkeypatch.setattr(
        launcher,
        "SHADOW_CONTRACT_RELATIVE",
        contract_path.relative_to(paths.checkout_root),
    )
    monkeypatch.setattr(
        launcher,
        "SHADOW_CONTRACT_SHA256",
        _sha256(contract_path),
    )
    monkeypatch.setattr(
        launcher,
        "REQUIRED_SHADOW_CODE_PATHS",
        frozenset(contract["validated_code_sha256"]),
    )
    monkeypatch.setattr(
        launcher,
        "SHADOW_VALIDATED_CODE_COUNT",
        len(contract["validated_code_sha256"]),
    )
    monkeypatch.setattr(
        launcher,
        "SHADOW_VALIDATED_CODE_MAPPING_SHA256",
        launcher._canonical_json_sha256(contract["validated_code_sha256"]),
    )
    fixture_proof = {
        "contract": {
            "name": contract["name"],
            "schema_version": contract["schema_version"],
            "sha256": _sha256(contract_path),
        },
        "generation_source": contract["generation_source"],
        "validated_code_sha256": contract["validated_code_sha256"],
        "reports": {
            "semantics_historical": {
                "sha256": contract["reports"]["semantics_historical"]["sha256"],
                "status": "passed",
                "parsed_records": 500_000,
                "rich_decision_snapshots": 3_986,
                "unknown_transitions": 0,
                "errors": 0,
            },
            "evidence_live": {
                "sha256": contract["reports"]["evidence_live"]["sha256"],
                "status": "passed",
                "decisions": 1_000,
                "semantic_censored_transitions": 0,
                "prefer_targets": 0,
                "source_checkpoint_id": (launcher.FIXED_INITIALIZATION_CHECKPOINT_ID),
                "training_state_unchanged": True,
                "maximum_episode_completion_storage_nbytes": 1_024,
                "completion_staging_is_bounded": True,
            },
            "evidence_actor": {
                "sha256": contract["reports"]["evidence_actor"]["sha256"],
                "status": "passed",
                "cases": 6,
                "actor_actionable_records": 4,
                "risk_sequence_records": 5,
                "prefer_targets": 0,
                "learner_mask_dry_run_passed": True,
            },
        },
    }
    monkeypatch.setattr(
        launcher,
        "SHADOW_AUTHORITY_SHA256",
        launcher._canonical_json_sha256(
            launcher._shadow_authority_payload(fixture_proof),
        ),
    )
    return contract_path, code, semantics_path, evidence_path


def test_shadow_contract_rejects_code_and_report_hash_drift(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    paths = launcher.validate_layout(
        _paths(tmp_path),
        enforce_active_root=False,
    )
    _, code, _, evidence = _make_shadow_fixture(paths, monkeypatch)
    proof = launcher._validate_shadow_contract(paths)
    assert proof["reports"]["evidence_live"]["decisions"] == 1_000

    code.write_text("SEMANTICS = 2\n", encoding="utf-8")
    with pytest.raises(launcher.LaunchError, match="code hash changed"):
        launcher._validate_shadow_contract(paths)
    code.write_text("SEMANTICS = 1\n", encoding="utf-8")

    evidence.write_text(
        evidence.read_text(encoding="utf-8") + "\n",
        encoding="utf-8",
    )
    with pytest.raises(launcher.LaunchError, match="report hash changed"):
        launcher._validate_shadow_contract(paths)


def test_embedded_shadow_rejects_valid_looking_code_digest_drift() -> None:
    changed = json.loads(json.dumps(SHADOW))
    first_key = next(iter(changed["validated_code_sha256"]))
    changed["validated_code_sha256"][first_key] = "0" * 64
    changed["authority_sha256"] = launcher._canonical_json_sha256(
        launcher._shadow_authority_payload(changed),
    )

    with pytest.raises(
        launcher.LaunchError,
        match="validated-code authority digest changed",
    ):
        launcher._validate_embedded_shadow_proof(changed)


def test_embedded_shadow_accepts_any_bounded_completion_storage_observation(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    changed = json.loads(json.dumps(SHADOW))
    changed["reports"]["evidence_live"]["maximum_episode_completion_storage_nbytes"] = 1_024
    authority = launcher._canonical_json_sha256(launcher._shadow_authority_payload(changed))
    changed["authority_sha256"] = authority
    monkeypatch.setattr(launcher, "SHADOW_AUTHORITY_SHA256", authority)
    observed = launcher._validate_embedded_shadow_proof(changed)
    assert observed["reports"]["evidence_live"]["maximum_episode_completion_storage_nbytes"] == 1_024

    changed["reports"]["evidence_live"]["maximum_episode_completion_storage_nbytes"] = 134_217_729
    with pytest.raises(launcher.LaunchError, match="failure-evidence shadow proof changed"):
        launcher._validate_embedded_shadow_proof(changed)


@pytest.mark.parametrize("abi_key", ("collector", "detector"))
def test_evidence_shadow_rejects_collector_or_detector_abi_drift(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    abi_key: str,
) -> None:
    paths = launcher.validate_layout(
        _paths(tmp_path),
        enforce_active_root=False,
    )
    contract, _, _, evidence = _make_shadow_fixture(paths, monkeypatch)
    report = json.loads(evidence.read_text(encoding="utf-8"))
    report["failure_credit_abi"][abi_key] = f"tampered-{abi_key}"
    evidence.write_text(json.dumps(report), encoding="utf-8")

    contract_payload = json.loads(contract.read_text(encoding="utf-8"))
    contract_payload["reports"]["evidence_live"]["sha256"] = _sha256(evidence)
    contract.write_text(json.dumps(contract_payload), encoding="utf-8")
    monkeypatch.setattr(
        launcher,
        "SHADOW_CONTRACT_SHA256",
        _sha256(contract),
    )

    with pytest.raises(
        launcher.LaunchError,
        match=rf"failure-evidence shadow {abi_key} ABI changed",
    ):
        launcher._validate_shadow_contract(paths)


@pytest.mark.parametrize(
    "direct_targets",
    (
        {"PREFER": 0},
        {"AVOID": 1},
        {"avoid": 1, "unexpected": 0},
    ),
)
def test_evidence_shadow_rejects_nonproduction_direct_target_keys(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    direct_targets: dict[str, int],
) -> None:
    paths = launcher.validate_layout(
        _paths(tmp_path),
        enforce_active_root=False,
    )
    contract, _, _, evidence = _make_shadow_fixture(
        paths,
        monkeypatch,
    )
    report = json.loads(evidence.read_text(encoding="utf-8"))
    report["direct_policy_targets"] = direct_targets
    evidence.write_text(json.dumps(report), encoding="utf-8")

    contract_payload = json.loads(
        contract.read_text(encoding="utf-8"),
    )
    contract_payload["reports"]["evidence_live"]["sha256"] = _sha256(evidence)
    contract.write_text(json.dumps(contract_payload), encoding="utf-8")
    monkeypatch.setattr(
        launcher,
        "SHADOW_CONTRACT_SHA256",
        _sha256(contract),
    )

    with pytest.raises(
        launcher.LaunchError,
        match="lowercase production enum values",
    ):
        launcher._validate_shadow_contract(paths)


def test_semantics_report_with_missing_gate_is_rejected() -> None:
    report = {
        "version": "sts2-semantics-shadow-report-v2",
        "status": "passed",
        "read_only": True,
        "training_authority": False,
        "errors": [],
        "counts": {
            "parsed_records": 500_000,
            "rich_decision_snapshots": 3_986,
        },
        "gates": {},
    }
    spec = {
        "version": report["version"],
        "status": "passed",
        "maximum_errors": 0,
        "maximum_unknown_transitions": 0,
        "minimum_parsed_records": 500_000,
        "minimum_rich_decision_snapshots": 3_986,
    }
    with pytest.raises(launcher.LaunchError, match="gate failed"):
        launcher._validate_semantics_shadow(report, spec)


def test_successor_binding_requires_full_v29_run_start(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        launcher,
        "FIXED_RUNTIME_CONFIG_FINGERPRINT_SHA256",
        launcher._canonical_json_sha256(_run_config()),
    )
    paths = launcher.validate_layout(
        _paths(tmp_path),
        enforce_active_root=False,
    )
    metrics = paths.artifact_root / "runs" / launcher.RUN_NAME / f"run-{TEST_RUN_ID}" / "metrics.jsonl"
    manifest = {
        **_payload(paths),
        "selection": _proof(paths),
        "launch_id": "33333333-3333-4333-8333-333333333333",
    }
    _authorize_manifest(paths, manifest)
    event = _run_start(paths, manifest=manifest)
    _write_event(metrics, event)
    assert (
        launcher._matching_model_init_run_start(
            metrics,
            paths,
            manifest=manifest,
        )
        == event
    )
    Path(manifest["launch_contract"]["path"]).unlink()
    assert (
        launcher._matching_model_init_run_start(
            metrics,
            paths,
            manifest=manifest,
        )
        == event
    )

    mutations: tuple[Callable[[dict[str, Any]], None], ...] = (
        lambda item: item["checkpoint_load"].__setitem__(
            "mode",
            "exact_resume",
        ),
        lambda item: item["state"].__setitem__("environment_steps", 1),
        lambda item: item["checkpoint_load"]["source_training_state"].__setitem__(
            "policy_version",
            0,
        ),
        lambda item: item["checkpoint_load"]["source_training_state"].__setitem__(
            "episodes",
            1,
        ),
        lambda item: item["checkpoint_load"].__setitem__(
            "unreviewed",
            True,
        ),
        lambda item: item["checkpoint_load"]["source_checkpoint"].__setitem__(
            "checkpoint_id",
            "00000000-0000-4000-8000-000000000000",
        ),
        lambda item: item["state"].__setitem__("learner_updates", 1),
        lambda item: item["actor_supervisor_state"].__setitem__(
            "episode_attempts",
            1,
        ),
        lambda item: item["runtime_provenance"]["supervised_launch"].__setitem__(
            "contract_sha256",
            "0" * 64,
        ),
        lambda item: item.__setitem__("run_id", "not-a-uuid"),
        lambda item: item.__setitem__(
            "run_id",
            "88888888-8888-4888-8888-888888888888",
        ),
        lambda item: item.__setitem__("unix_s", float("nan")),
        lambda item: item["runtime_provenance"].__setitem__(
            "backend",
            "live",
        ),
        lambda item: item["runtime_provenance"].__setitem__(
            "simulator_identity",
            {"schema_version": "another-simulator"},
        ),
        lambda item: item["runtime_provenance"].__setitem__(
            "runtime_mechanics",
            {"schema": "another-mechanics-contract"},
        ),
        lambda item: item["runtime_provenance"]["sdpa_backend"].__setitem__(
            "checkpoint_load_mode",
            "exact_resume",
        ),
        lambda item: item["runtime_provenance"]["sdpa_backend"]["current"].__setitem__(
            "effective_backend",
            "framework_default",
        ),
        lambda item: item.__setitem__("pipeline", "old-pipeline"),
        lambda item: item["config"]["transaction_learning"].__setitem__(
            "enabled",
            True,
        ),
        lambda item: item["config"]["failure_credit"].__setitem__(
            "mode",
            "shadow",
        ),
        lambda item: item["config"]["runtime"].__setitem__(
            "checkpoint_interval_steps",
            12_345,
        ),
    )
    for mutate in mutations:
        changed = json.loads(json.dumps(event))
        mutate(changed)
        metrics.write_text(json.dumps(changed) + "\n", encoding="utf-8")
        assert (
            launcher._matching_model_init_run_start(
                metrics,
                paths,
                manifest=manifest,
            )
            is None
        )

    metrics.write_text(
        json.dumps({"event": "learner_update"}) + "\n" + json.dumps(event) + "\n",
        encoding="utf-8",
    )
    assert (
        launcher._matching_model_init_run_start(
            metrics,
            paths,
            manifest=manifest,
        )
        is None
    )
    metrics.write_text(
        json.dumps(event) + "\n" + json.dumps(event) + "\n",
        encoding="utf-8",
    )
    assert (
        launcher._matching_model_init_run_start(
            metrics,
            paths,
            manifest=manifest,
        )
        is None
    )


def test_torn_metrics_tail_is_repaired_before_watchdog_append(
    tmp_path: Path,
) -> None:
    metrics = tmp_path / "metrics.jsonl"
    _write_event(metrics, {"event": "run_start"})
    with metrics.open("ab") as handle:
        handle.write(b'{"event":"learner_')
    event = {
        "event": "run_failed",
        "source": "persistent-native-exit-watchdog",
        "watchdog_event_id": "event-id",
    }
    assert (
        launcher._append_jsonl_event_once(
            metrics,
            event,
            event_id="event-id",
        )
        is True
    )
    assert launcher._metrics_events(metrics)[-1] == event
    assert (
        launcher._append_jsonl_event_once(
            metrics,
            event,
            event_id="event-id",
        )
        is False
    )
    assert metrics.read_bytes().count(b'"watchdog_event_id":"event-id"') == 1


def test_native_abort_terminalizes_bound_successor(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        launcher,
        "FIXED_RUNTIME_CONFIG_FINGERPRINT_SHA256",
        launcher._canonical_json_sha256(_run_config()),
    )
    paths = launcher.validate_layout(
        _paths(tmp_path),
        enforce_active_root=False,
    )
    _patch_payload_authorities(monkeypatch)
    monkeypatch.setattr(
        launcher,
        "_checkpoint_proof",
        lambda _paths: _proof(paths),
    )
    manifest_path = paths.manifest_dir / (f"{launcher.RUN_NAME}-test-launch.launch.json")
    log_path = paths.launcher_dir / "logs/launch.log"
    log_path.parent.mkdir(parents=True, exist_ok=True)
    log_path.write_text(
        "AqlQueue::HandleInsufficientScratch\n",
        encoding="utf-8",
    )
    manifest = {
        **_payload(paths),
        "schema_version": launcher.MANIFEST_SCHEMA,
        "preflight_status": "preflight-passed",
        "selection": _proof(paths),
        "launch_id": "11111111-1111-4111-8111-111111111111",
        "status": "running",
        "training_started": True,
        "preexisting_run_directories": [],
        "metrics_path": None,
        "log_path": str(log_path),
    }
    _authorize_manifest(
        paths,
        manifest,
        manifest_path=manifest_path,
    )
    launcher._persist_manifest(
        paths,
        manifest_path=manifest_path,
        manifest=manifest,
    )
    state = json.loads(launcher._state_path(paths).read_text(encoding="utf-8"))
    assert state["abi_contract"] == ABI
    assert state["initialization"]["checkpoint_contract"] == launcher.FIXED_FROZEN_CONTRACT_NAME
    assert state["preflight_training_started"] is False
    assert state["training_started"] is True

    metrics = paths.artifact_root / "runs" / launcher.RUN_NAME / f"run-{TEST_RUN_ID}" / "metrics.jsonl"
    run_start = _run_start(paths, manifest=manifest)
    _write_event(metrics, run_start)
    launcher._record_successor_binding(
        paths,
        manifest_path=manifest_path,
        metrics_path=metrics,
        run_start=run_start,
    )
    supervisor = launcher._identity_from_mapping(
        manifest["supervisor_process_identity"],
        label="supervisor",
    )
    trainer = launcher._identity_from_mapping(
        manifest["trainer_process_identity"],
        label="trainer",
    )
    other_metrics = (
        paths.artifact_root / "runs" / launcher.RUN_NAME / "run-88888888-8888-4888-8888-888888888888" / "metrics.jsonl"
    )
    _write_event(
        other_metrics,
        _run_start(paths, manifest=manifest),
    )
    with pytest.raises(
        launcher.LaunchError,
        match="manifest-bound successor",
    ):
        launcher.finalize_supervised_exit(
            paths,
            manifest_path=manifest_path,
            returncode=-signal.SIGABRT,
            supervisor_identity=supervisor,
            trainer_identity=trainer,
            metrics_path=other_metrics,
            enforce_active_root=False,
        )

    terminal = launcher.finalize_supervised_exit(
        paths,
        manifest_path=manifest_path,
        returncode=-signal.SIGABRT,
        supervisor_identity=supervisor,
        trainer_identity=trainer,
        metrics_path=metrics,
        enforce_active_root=False,
    )
    assert terminal["status"] == "failed"
    assert terminal["terminal"]["classification"]["kind"] == "native_abort"
    failures = [event for event in launcher._metrics_events(metrics) if event.get("event") == "run_failed"]
    assert len(failures) == 1

    terminal_manifest = json.loads(
        manifest_path.read_text(encoding="utf-8"),
    )
    terminal_metrics = metrics.read_bytes()
    metrics.write_text(
        json.dumps(run_start) + "\n",
        encoding="utf-8",
    )
    with pytest.raises(
        launcher.LaunchError,
        match="lacks one metrics run_failed event",
    ):
        launcher._validate_manifest(paths, manifest_path)
    metrics.write_bytes(terminal_metrics)

    missing_terminal_field = json.loads(
        json.dumps(terminal_manifest),
    )
    del missing_terminal_field["terminal"]["returncode"]
    launcher.atomic_write_json(
        manifest_path,
        missing_terminal_field,
    )
    with pytest.raises(
        launcher.LaunchError,
        match="terminal supervised manifest fields changed",
    ):
        launcher._validate_manifest(paths, manifest_path)
    launcher.atomic_write_json(
        manifest_path,
        terminal_manifest,
    )
    assert (
        launcher._validate_manifest(
            paths,
            manifest_path,
        )["status"]
        == "failed"
    )
    assert failures[0]["initialization_mode"] == "model_initialization"
    assert failures[0]["environment_steps"] == 0
    assert "resume_checkpoint" not in failures[0]

    launcher.finalize_supervised_exit(
        paths,
        manifest_path=manifest_path,
        returncode=-signal.SIGABRT,
        supervisor_identity=supervisor,
        trainer_identity=trainer,
        metrics_path=metrics,
        enforce_active_root=False,
    )
    failures = [event for event in launcher._metrics_events(metrics) if event.get("event") == "run_failed"]
    assert len(failures) == 1


def test_status_does_not_reopen_live_shadow_and_fails_closed_on_untracked_supervisor(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    paths = launcher.validate_layout(
        _paths(tmp_path),
        enforce_active_root=False,
    )
    _patch_payload_authorities(monkeypatch)
    manifest_path = paths.manifest_dir / (f"{launcher.RUN_NAME}-durable-launch.launch.json")
    log_path = paths.launcher_dir / "logs/launch.log"
    manifest = {
        **_payload(paths),
        "schema_version": launcher.MANIFEST_SCHEMA,
        "preflight_status": "preflight-passed",
        "selection": _proof(paths),
        "launch_id": "22222222-2222-4222-8222-222222222222",
        "created_unix_s": 1.0,
        "created_at_utc": "1970-01-01T00:00:01+00:00",
        "status": "supervisor-launching",
        "preexisting_run_directories": [],
        "metrics_path": None,
        "log_path": str(log_path),
    }
    _authorize_manifest(
        paths,
        manifest,
        manifest_path=manifest_path,
    )
    launcher._persist_manifest(
        paths,
        manifest_path=manifest_path,
        manifest=manifest,
    )
    Path(manifest["launch_contract"]["path"]).unlink()
    state_path = launcher._state_path(paths)
    state_path.unlink()

    def live_shadow_must_not_be_reopened(_paths: Any) -> dict[str, Any]:
        raise AssertionError("lifecycle reopened mutable live shadow evidence")

    monkeypatch.setattr(
        launcher,
        "_validate_shadow_contract",
        live_shadow_must_not_be_reopened,
    )
    launching = launcher.read_status(
        paths,
        enforce_active_root=False,
    )
    assert launching["status"] == "supervisor-launching"
    assert state_path.is_file()
    repaired_state = json.loads(state_path.read_text(encoding="utf-8"))
    assert repaired_state["manifest_path"] == str(manifest_path)
    assert repaired_state["log_path"] == str(log_path)
    assert repaired_state["training_started"] is False

    repaired_state["written_unix_s"] = 0.0
    launcher.atomic_write_json(state_path, repaired_state)
    status = launcher.read_status(
        paths,
        enforce_active_root=False,
    )
    assert status["status"] == "ambiguous-untracked-supervisor"
    assert status["running"] is True
    assert status["process_identity_ambiguity"] == ("supervisor_spawned_pid_missing_or_malformed")
    durable = json.loads(manifest_path.read_text(encoding="utf-8"))
    assert durable["status"] == "supervisor-launching"
    assert durable.get("terminal") is None


def test_supervisor_finalization_failure_is_durable(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    paths = launcher.validate_layout(
        _paths(tmp_path),
        enforce_active_root=False,
    )
    _patch_payload_authorities(monkeypatch)
    manifest_path = paths.manifest_dir / f"{launcher.RUN_NAME}-emergency.launch.json"
    manifest = {
        **_payload(paths),
        "schema_version": launcher.MANIFEST_SCHEMA,
        "preflight_status": "preflight-passed",
        "selection": _proof(paths),
        "launch_id": "55555555-5555-4555-8555-555555555555",
        "created_unix_s": 1.0,
        "created_at_utc": "1970-01-01T00:00:01+00:00",
        "status": "running",
        "training_started": True,
        "preexisting_run_directories": [],
        "metrics_path": None,
        "log_path": str(paths.launcher_dir / "logs/launch.log"),
    }
    _authorize_manifest(
        paths,
        manifest,
        manifest_path=manifest_path,
    )
    launcher._persist_manifest(
        paths,
        manifest_path=manifest_path,
        manifest=manifest,
    )
    fallback = launcher._persist_supervisor_emergency_failure(
        paths,
        manifest_path=manifest_path,
        launch_id=str(manifest["launch_id"]),
        original_error=RuntimeError("trainer crashed"),
        finalization_error=RuntimeError("metrics binding changed"),
        returncode=70,
        supervisor_identity=None,
        trainer_identity=None,
        metrics_path=None,
    )
    assert fallback.is_file()
    payload = json.loads(fallback.read_text(encoding="utf-8"))
    assert payload["event"] == "supervisor_recovery_failed"
    assert payload["finalization_error"]["message"] == ("metrics binding changed")
    repaired = json.loads(manifest_path.read_text(encoding="utf-8"))
    assert repaired["status"] == "failed"
    assert repaired["terminal"]["classification"]["kind"] == ("supervisor_finalization_failure")
    assert repaired["terminal"]["fallback_event_path"] == str(fallback)


def test_manifest_recovery_ignores_only_older_malformed_files(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    paths = launcher.validate_layout(
        _paths(tmp_path),
        enforce_active_root=False,
    )
    _patch_payload_authorities(monkeypatch)
    valid_path = paths.manifest_dir / (
        f"{launcher.RUN_NAME}-20260731-120000-66666666-6666-4666-8666-666666666666.launch.json"
    )
    manifest = {
        **_payload(paths),
        "schema_version": launcher.MANIFEST_SCHEMA,
        "preflight_status": "preflight-passed",
        "selection": _proof(paths),
        "launch_id": "66666666-6666-4666-8666-666666666666",
        "created_unix_s": 1.0,
        "created_at_utc": "1970-01-01T00:00:01+00:00",
        "status": "supervisor-launching",
        "preexisting_run_directories": [],
        "metrics_path": None,
        "log_path": str(paths.launcher_dir / "logs/launch.log"),
    }
    _authorize_manifest(
        paths,
        manifest,
        manifest_path=valid_path,
    )
    launcher._persist_manifest(
        paths,
        manifest_path=valid_path,
        manifest=manifest,
    )
    old_malformed = paths.manifest_dir / f"{launcher.RUN_NAME}-20260730-120000-old.launch.json"
    old_malformed.write_text("{torn", encoding="utf-8")
    launcher._state_path(paths).unlink()

    status = launcher.read_status(
        paths,
        enforce_active_root=False,
    )
    assert status["launch_id"] == manifest["launch_id"]
    assert status["manifest_path"] == str(valid_path)


def test_newest_or_state_referenced_malformed_manifest_fails_closed(
    tmp_path: Path,
) -> None:
    paths = launcher.validate_layout(
        _paths(tmp_path),
        enforce_active_root=False,
    )
    valid_path = paths.manifest_dir / (
        f"{launcher.RUN_NAME}-20260731-120000-77777777-7777-4777-8777-777777777777.launch.json"
    )
    manifest = {
        **_payload(paths),
        "schema_version": launcher.MANIFEST_SCHEMA,
        "preflight_status": "preflight-passed",
        "selection": _proof(paths),
        "launch_id": "77777777-7777-4777-8777-777777777777",
        "created_unix_s": 1.0,
        "created_at_utc": "1970-01-01T00:00:01+00:00",
        "status": "supervisor-launching",
        "preexisting_run_directories": [],
        "metrics_path": None,
        "log_path": str(paths.launcher_dir / "logs/launch.log"),
    }
    _authorize_manifest(
        paths,
        manifest,
        manifest_path=valid_path,
    )
    launcher._persist_manifest(
        paths,
        manifest_path=valid_path,
        manifest=manifest,
    )
    valid_path.write_text("{torn", encoding="utf-8")
    with pytest.raises(
        launcher.LaunchError,
        match="state points at a missing or invalid manifest",
    ):
        launcher.read_status(
            paths,
            enforce_active_root=False,
        )

    launcher._state_path(paths).unlink()
    launcher.atomic_write_json(valid_path, manifest)
    newest_malformed = paths.manifest_dir / (f"{launcher.RUN_NAME}-20260801-120000-new.launch.json")
    newest_malformed.write_text("{torn", encoding="utf-8")
    with pytest.raises(
        launcher.LaunchError,
        match="newest supervised launch manifest is invalid",
    ):
        launcher.read_status(
            paths,
            enforce_active_root=False,
        )


def _active_manifest(
    paths: Any,
    *,
    launch_id: str,
    training_started: bool,
) -> tuple[Path, dict[str, Any]]:
    manifest_path = paths.manifest_dir / (f"{launcher.RUN_NAME}-20260731-120000-{launch_id}.launch.json")
    manifest = {
        **_payload(paths),
        "schema_version": launcher.MANIFEST_SCHEMA,
        "preflight_status": "preflight-passed",
        "selection": _proof(paths),
        "launch_id": launch_id,
        "created_unix_s": 1.0,
        "created_at_utc": "1970-01-01T00:00:01+00:00",
        "status": ("running" if training_started else "supervisor-launching"),
        "training_started": training_started,
        "preexisting_run_directories": [],
        "metrics_path": None,
        "log_path": str(paths.launcher_dir / "logs/launch.log"),
    }
    _authorize_manifest(
        paths,
        manifest,
        manifest_path=manifest_path,
    )
    if training_started:
        assert manifest.get("supervisor_process_identity") is not None
    return manifest_path, manifest


def test_nonzero_exit_after_run_complete_uses_failure_fallback_without_second_metrics_terminal(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        launcher,
        "FIXED_RUNTIME_CONFIG_FINGERPRINT_SHA256",
        launcher._canonical_json_sha256(_run_config()),
    )
    paths = launcher.validate_layout(
        _paths(tmp_path),
        enforce_active_root=False,
    )
    _patch_payload_authorities(monkeypatch)
    manifest_path, manifest = _active_manifest(
        paths,
        launch_id="bbbbbbbb-bbbb-4bbb-8bbb-bbbbbbbbbbbb",
        training_started=True,
    )
    launcher._persist_manifest(
        paths,
        manifest_path=manifest_path,
        manifest=manifest,
    )
    metrics = paths.artifact_root / "runs" / launcher.RUN_NAME / f"run-{TEST_RUN_ID}" / "metrics.jsonl"
    run_start = _run_start(paths, manifest=manifest)
    _write_event(metrics, run_start)
    _write_event(
        metrics,
        {
            "event": "run_complete",
            "unix_s": 2.0,
            "environment_steps": 100_000,
        },
    )
    launcher._record_successor_binding(
        paths,
        manifest_path=manifest_path,
        metrics_path=metrics,
        run_start=run_start,
    )

    terminal = launcher.finalize_supervised_exit(
        paths,
        manifest_path=manifest_path,
        returncode=70,
        supervisor_identity=None,
        trainer_identity=None,
        metrics_path=metrics,
        enforce_active_root=False,
    )

    assert terminal["status"] == "failed"
    assert terminal["terminal"]["classification"]["kind"] == ("nonzero_exit")
    terminal_events = [
        event
        for event in launcher._metrics_events(metrics)
        if event.get("event") in {"run_complete", "interrupt", "run_failed"}
    ]
    assert [event["event"] for event in terminal_events] == [
        "run_complete",
    ]
    fallback = Path(terminal["terminal"]["fallback_event_path"])
    assert fallback.is_file()
    failure = json.loads(fallback.read_text(encoding="utf-8"))
    assert failure["event"] == "run_failed"
    assert failure["returncode"] == 70
    assert failure["watchdog_event_id"] == (terminal["terminal_event_id"])
    assert (
        launcher._validate_manifest(
            paths,
            manifest_path,
        )["status"]
        == "failed"
    )


def test_status_recovers_exec_authorized_bootstrap_handoff(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    paths = launcher.validate_layout(
        _paths(tmp_path),
        enforce_active_root=False,
    )
    _patch_payload_authorities(monkeypatch)
    manifest_path, manifest = _active_manifest(
        paths,
        launch_id="88888888-8888-4888-8888-888888888888",
        training_started=True,
    )
    pid = 42_424
    command = manifest["supervised_trainer_command"]
    recovered = launcher.ProcessIdentity(
        pid=pid,
        proc_start_ticks=123_456,
        command_line_sha256=launcher._command_line_sha256(command),
        executable=str(Path(command[0]).resolve(strict=False)),
    )
    bootstrap_command = manifest["trainer_bootstrap_command"]
    manifest.update(
        {
            "training_started": False,
            "trainer_bootstrap_phase": "exec_authorized",
            "trainer_bootstrap_process_identity": asdict(
                _identity_for_command(
                    bootstrap_command,
                    pid=pid,
                    proc_start_ticks=123_456,
                )
            ),
            "trainer_bootstrap_exec_observed_at_utc": None,
            "trainer_spawned_pid": pid,
            "trainer_spawned_at_utc": None,
            "trainer_started_at_utc": None,
            "trainer_process_identity": None,
        }
    )
    launcher._persist_manifest(
        paths,
        manifest_path=manifest_path,
        manifest=manifest,
    )
    monkeypatch.setattr(
        launcher,
        "capture_process_identity",
        lambda observed_pid: (
            recovered
            if observed_pid == pid
            else (_ for _ in ()).throw(
                launcher.LaunchError("unexpected PID"),
            )
        ),
    )
    monkeypatch.setattr(
        launcher,
        "process_identity_matches",
        lambda identity: identity == recovered,
    )

    status = launcher.read_status(
        paths,
        enforce_active_root=False,
    )

    assert status["status"] == "unwatched-running"
    assert status["running"] is True
    assert status["process_identity_ambiguity"] is None
    assert status["trainer"]["identity"] == {
        "pid": pid,
        "proc_start_ticks": 123_456,
        "command_line_sha256": recovered.command_line_sha256,
        "executable": recovered.executable,
    }
    durable = json.loads(
        manifest_path.read_text(encoding="utf-8"),
    )
    assert durable["trainer_process_identity"] == (status["trainer"]["identity"])
    assert durable["trainer_spawned_pid"] == pid
    assert durable["trainer_bootstrap_phase"] == "exec_observed"
    assert durable["training_started"] is True


@pytest.mark.parametrize("failure_mode", ("mismatch", "unreadable"))
def test_status_treats_unprovable_trainer_pid_as_running_ambiguity(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    failure_mode: str,
) -> None:
    paths = launcher.validate_layout(
        _paths(tmp_path),
        enforce_active_root=False,
    )
    _patch_payload_authorities(monkeypatch)
    manifest_path, manifest = _active_manifest(
        paths,
        launch_id="99999999-9999-4999-8999-999999999999",
        training_started=True,
    )
    pid = 43_434
    manifest.update(
        {
            "training_started": False,
            "trainer_bootstrap_phase": "exec_authorized",
            "trainer_bootstrap_process_identity": asdict(
                _identity_for_command(
                    manifest["trainer_bootstrap_command"],
                    pid=pid,
                    proc_start_ticks=123_456,
                )
            ),
            "trainer_bootstrap_exec_observed_at_utc": None,
            "trainer_spawned_pid": pid,
            "trainer_spawned_at_utc": None,
            "trainer_started_at_utc": None,
            "trainer_process_identity": None,
        }
    )
    launcher._persist_manifest(
        paths,
        manifest_path=manifest_path,
        manifest=manifest,
    )

    if failure_mode == "unreadable":

        def capture(_pid: int) -> launcher.ProcessIdentity:
            raise launcher.LaunchError("proc disappeared")
    else:

        def capture(_pid: int) -> launcher.ProcessIdentity:
            return launcher.ProcessIdentity(
                pid=pid,
                proc_start_ticks=1,
                command_line_sha256="f" * 64,
                executable="/reused/pid",
            )

    monkeypatch.setattr(
        launcher,
        "capture_process_identity",
        capture,
    )

    status = launcher.read_status(
        paths,
        enforce_active_root=False,
    )

    if failure_mode == "mismatch":
        assert status["status"] == "ambiguous-untracked-bootstrap"
        assert status["running"] is True
        assert status["process_identity_ambiguity"] == ("bootstrap_exec_identity_mismatch")
    else:
        assert status["status"] == "failed"
        assert status["running"] is False
        assert status["process_identity_ambiguity"] is None
    durable = json.loads(
        manifest_path.read_text(encoding="utf-8"),
    )
    assert durable["status"] == ("running" if failure_mode == "mismatch" else "failed")
    assert durable.get("trainer_process_identity") is None


def test_status_treats_missing_supervisor_spawn_pid_as_running_ambiguity(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    paths = launcher.validate_layout(
        _paths(tmp_path),
        enforce_active_root=False,
    )
    _patch_payload_authorities(monkeypatch)
    manifest_path, manifest = _active_manifest(
        paths,
        launch_id="aaaaaaaa-aaaa-4aaa-8aaa-aaaaaaaaaaaa",
        training_started=False,
    )
    manifest["supervisor_command"] = [
        str(paths.venv_python),
        str(launcher.SCRIPT_DIR / "launch_v29_failure_credit_v4.py"),
        "supervise",
        "--manifest",
        str(manifest_path),
    ]
    launcher._persist_manifest(
        paths,
        manifest_path=manifest_path,
        manifest=manifest,
    )
    stale_state = json.loads(
        launcher._state_path(paths).read_text(encoding="utf-8"),
    )
    stale_state["written_unix_s"] = 0.0
    launcher.atomic_write_json(
        launcher._state_path(paths),
        stale_state,
    )

    status = launcher.read_status(
        paths,
        enforce_active_root=False,
    )

    assert status["status"] == ("ambiguous-untracked-supervisor")
    assert status["running"] is True
    assert status["process_identity_ambiguity"] == ("supervisor_spawned_pid_missing_or_malformed")
    durable = json.loads(
        manifest_path.read_text(encoding="utf-8"),
    )
    assert durable["status"] == "supervisor-launching"
    assert durable.get("terminal") is None


@pytest.mark.parametrize(
    "mutation,match",
    (
        (
            lambda manifest, _path: manifest.__setitem__(
                "manifest_path",
                str(_path.with_name("other.launch.json")),
            ),
            "path differs",
        ),
        (
            lambda manifest, _path: manifest["supervisor_command"].__setitem__(
                -1,
                str(_path.with_name("other.launch.json")),
            ),
            "supervisor command changed",
        ),
    ),
)
def test_manifest_file_and_supervisor_command_are_exactly_bound(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    mutation: Callable[[dict[str, Any], Path], None],
    match: str,
) -> None:
    paths = launcher.validate_layout(
        _paths(tmp_path),
        enforce_active_root=False,
    )
    _patch_payload_authorities(monkeypatch)
    manifest_path, manifest = _active_manifest(
        paths,
        launch_id="cccccccc-cccc-4ccc-8ccc-cccccccccccc",
        training_started=False,
    )
    mutation(manifest, manifest_path)
    launcher.atomic_write_json(manifest_path, manifest)

    with pytest.raises(launcher.LaunchError, match=match):
        launcher._validate_manifest(paths, manifest_path)


@pytest.mark.parametrize(
    "identity",
    (
        {
            "pid": True,
            "proc_start_ticks": 1,
            "command_line_sha256": "0" * 64,
            "executable": "/python",
        },
        {
            "pid": "1",
            "proc_start_ticks": 1,
            "command_line_sha256": "0" * 64,
            "executable": "/python",
        },
        {
            "pid": 1,
            "proc_start_ticks": 0,
            "command_line_sha256": "0" * 64,
            "executable": "/python",
        },
        {
            "pid": 1,
            "proc_start_ticks": 1,
            "command_line_sha256": "A" * 64,
            "executable": "/python",
        },
        {
            "pid": 1,
            "proc_start_ticks": 1,
            "command_line_sha256": "0" * 64,
            "executable": "relative/python",
        },
        {
            "pid": 1,
            "proc_start_ticks": 1,
            "command_line_sha256": "0" * 64,
            "executable": "/python",
            "unreviewed": True,
        },
    ),
)
def test_process_identity_parser_rejects_coercion_and_extra_fields(
    identity: dict[str, Any],
) -> None:
    with pytest.raises(
        launcher.LaunchError,
        match=r"identity (?:is malformed|fields changed)",
    ):
        launcher._identity_from_mapping(
            identity,
            label="trainer",
        )


@pytest.mark.parametrize("role", ("supervisor", "trainer"))
def test_manifest_rejects_stored_identity_outside_exact_command(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    role: str,
) -> None:
    paths = launcher.validate_layout(
        _paths(tmp_path),
        enforce_active_root=False,
    )
    _patch_payload_authorities(monkeypatch)
    manifest_path, manifest = _active_manifest(
        paths,
        launch_id="dddddddd-dddd-4ddd-8ddd-dddddddddddd",
        training_started=True,
    )
    pid_field = f"{role}_spawned_pid"
    identity_field = f"{role}_process_identity"
    pid = int(manifest[pid_field])
    manifest[identity_field] = {
        "pid": pid,
        "proc_start_ticks": 19,
        "command_line_sha256": "f" * 64,
        "executable": str(paths.venv_python.resolve(strict=False)),
    }
    launcher.atomic_write_json(manifest_path, manifest)

    with pytest.raises(
        launcher.LaunchError,
        match=rf"{role} process identity does not match",
    ):
        launcher._validate_manifest(paths, manifest_path)


def test_trainer_spawn_intent_without_pid_is_safely_reconciled(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    paths = launcher.validate_layout(
        _paths(tmp_path),
        enforce_active_root=False,
    )
    _patch_payload_authorities(monkeypatch)
    manifest_path, manifest = _active_manifest(
        paths,
        launch_id="eeeeeeee-eeee-4eee-8eee-eeeeeeeeeeee",
        training_started=False,
    )
    supervisor = _identity_for_command(
        manifest["supervisor_command"],
        pid=98_765,
        proc_start_ticks=456,
    )
    manifest["supervisor_spawned_pid"] = supervisor.pid
    manifest["supervisor_process_identity"] = asdict(supervisor)
    launcher._persist_manifest(
        paths,
        manifest_path=manifest_path,
        manifest=manifest,
    )
    nonce = "b" * 64
    bootstrap_command = launcher._trainer_bootstrap_command(
        paths,
        supervisor_identity=supervisor,
        gate_fd=101,
        status_fd=102,
        nonce=nonce,
        trainer_command=manifest["supervised_trainer_command"],
    )
    launcher._record_trainer_spawn_intent(
        paths,
        manifest_path=manifest_path,
        bootstrap_command=bootstrap_command,
        nonce_sha256=launcher._sha256_bytes(nonce.encode("ascii")),
    )

    status = launcher.read_status(
        paths,
        enforce_active_root=False,
    )

    assert status["status"] == "failed"
    assert status["running"] is False
    assert status["process_identity_ambiguity"] is None
    durable = json.loads(manifest_path.read_text(encoding="utf-8"))
    assert durable["trainer_spawn_intent"] is True
    assert durable["training_started"] is False
    assert durable.get("terminal") is not None


def test_bootstrap_lifecycle_requires_durable_ready_and_authorization_order(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    paths = launcher.validate_layout(
        _paths(tmp_path),
        enforce_active_root=False,
    )
    _patch_payload_authorities(monkeypatch)
    manifest_path, manifest = _active_manifest(
        paths,
        launch_id="abababab-abab-4bab-8bab-abababababab",
        training_started=False,
    )
    supervisor = _identity_for_command(
        manifest["supervisor_command"],
        pid=77_001,
        proc_start_ticks=901,
    )
    manifest["supervisor_spawned_pid"] = supervisor.pid
    manifest["supervisor_process_identity"] = asdict(supervisor)
    launcher._persist_manifest(
        paths,
        manifest_path=manifest_path,
        manifest=manifest,
    )
    nonce = "c" * 64
    bootstrap_command = launcher._trainer_bootstrap_command(
        paths,
        supervisor_identity=supervisor,
        gate_fd=105,
        status_fd=106,
        nonce=nonce,
        trainer_command=manifest["supervised_trainer_command"],
    )
    launcher._record_trainer_spawn_intent(
        paths,
        manifest_path=manifest_path,
        bootstrap_command=bootstrap_command,
        nonce_sha256=launcher._sha256_bytes(nonce.encode("ascii")),
    )
    bootstrap = _identity_for_command(
        bootstrap_command,
        pid=77_002,
        proc_start_ticks=902,
    )
    with pytest.raises(
        launcher.LaunchError,
        match="exec authorization lacks exact bootstrap authority",
    ):
        launcher._record_trainer_exec_authorized(
            paths,
            manifest_path=manifest_path,
            identity=bootstrap,
        )

    launcher._record_trainer_bootstrap_ready(
        paths,
        manifest_path=manifest_path,
        identity=bootstrap,
    )
    ready = launcher._validate_manifest(paths, manifest_path)
    assert ready["trainer_bootstrap_phase"] == "bootstrap_ready"
    assert ready["training_started"] is False
    assert ready.get("trainer_process_identity") is None

    launcher._record_trainer_exec_authorized(
        paths,
        manifest_path=manifest_path,
        identity=bootstrap,
    )
    authorized = launcher._validate_manifest(paths, manifest_path)
    assert authorized["trainer_bootstrap_phase"] == "exec_authorized"
    assert authorized["training_started"] is False

    trainer = _identity_for_command(
        manifest["supervised_trainer_command"],
        pid=bootstrap.pid,
        proc_start_ticks=bootstrap.proc_start_ticks,
    )
    launcher._record_trainer_exec_observed(
        paths,
        manifest_path=manifest_path,
        bootstrap_identity=bootstrap,
        trainer_identity=trainer,
    )
    observed = launcher._validate_manifest(paths, manifest_path)
    assert observed["trainer_bootstrap_phase"] == "exec_observed"
    assert observed["training_started"] is True
    assert observed["trainer_process_identity"] == asdict(trainer)


@pytest.mark.parametrize(
    "mutation,match",
    (
        (
            lambda manifest: manifest.__setitem__(
                "trainer_bootstrap_nonce_sha256",
                "f" * 64,
            ),
            "command authority changed",
        ),
        (
            lambda manifest: manifest.__setitem__(
                "trainer_bootstrap_phase",
                "bootstrap_ready",
            ),
            "premature exec evidence",
        ),
        (
            lambda manifest: manifest["trainer_bootstrap_command"].__setitem__(
                13,
                "31",
            ),
            "command authority changed",
        ),
    ),
)
def test_bootstrap_manifest_tamper_fails_closed(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    mutation: Callable[[dict[str, Any]], None],
    match: str,
) -> None:
    paths = launcher.validate_layout(
        _paths(tmp_path),
        enforce_active_root=False,
    )
    _patch_payload_authorities(monkeypatch)
    manifest_path, manifest = _active_manifest(
        paths,
        launch_id="cdcdcdcd-cdcd-4dcd-8dcd-cdcdcdcdcdcd",
        training_started=True,
    )
    mutation(manifest)
    launcher.atomic_write_json(manifest_path, manifest)
    with pytest.raises(launcher.LaunchError, match=match):
        launcher._validate_manifest(paths, manifest_path)


def test_terminal_reaped_trainer_does_not_become_running_ambiguity(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    paths = launcher.validate_layout(
        _paths(tmp_path),
        enforce_active_root=False,
    )
    _patch_payload_authorities(monkeypatch)
    manifest_path, manifest = _active_manifest(
        paths,
        launch_id="ffffffff-ffff-4fff-8fff-ffffffffffff",
        training_started=True,
    )
    launcher._persist_manifest(
        paths,
        manifest_path=manifest_path,
        manifest=manifest,
    )
    launcher._record_trainer_exit_observed(
        paths,
        manifest_path=manifest_path,
        pid=int(manifest["trainer_spawned_pid"]),
        returncode=70,
    )
    terminal = launcher.finalize_supervised_exit(
        paths,
        manifest_path=manifest_path,
        returncode=70,
        supervisor_identity=None,
        trainer_identity=None,
        metrics_path=None,
        enforce_active_root=False,
    )
    assert terminal["status"] == "failed"
    monkeypatch.setattr(
        launcher,
        "process_identity_matches",
        lambda _identity: False,
    )

    status = launcher.read_status(
        paths,
        enforce_active_root=False,
    )

    assert status["status"] == "failed"
    assert status["running"] is False
    assert status["process_identity_ambiguity"] is None
    assert status["trainer_exit_observed"] is True
    assert status["trainer_reaped_returncode"] == 70

    incoherent = json.loads(
        manifest_path.read_text(encoding="utf-8"),
    )
    incoherent["terminal"]["returncode"] = 71
    launcher.atomic_write_json(manifest_path, incoherent)
    with pytest.raises(
        launcher.LaunchError,
        match="differs from durable reaped-trainer evidence",
    ):
        launcher._validate_manifest(paths, manifest_path)


@pytest.mark.parametrize(
    ("returncode", "with_run_complete", "expected_status"),
    (
        (0, True, "completed"),
        (70, False, "failed"),
    ),
)
def test_status_reconciles_from_durable_reaped_trainer_returncode(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    returncode: int,
    with_run_complete: bool,
    expected_status: str,
) -> None:
    monkeypatch.setattr(
        launcher,
        "FIXED_RUNTIME_CONFIG_FINGERPRINT_SHA256",
        launcher._canonical_json_sha256(_run_config()),
    )
    paths = launcher.validate_layout(
        _paths(tmp_path),
        enforce_active_root=False,
    )
    _patch_payload_authorities(monkeypatch)
    manifest_path, manifest = _active_manifest(
        paths,
        launch_id="12121212-1212-4212-8212-121212121212",
        training_started=True,
    )
    launcher._persist_manifest(
        paths,
        manifest_path=manifest_path,
        manifest=manifest,
    )
    if with_run_complete:
        metrics = paths.artifact_root / "runs" / launcher.RUN_NAME / f"run-{TEST_RUN_ID}" / "metrics.jsonl"
        run_start = _run_start(paths, manifest=manifest)
        _write_event(metrics, run_start)
        _write_event(
            metrics,
            {
                "event": "run_complete",
                "unix_s": 2.0,
                "environment_steps": 100_000,
            },
        )
        launcher._record_successor_binding(
            paths,
            manifest_path=manifest_path,
            metrics_path=metrics,
            run_start=run_start,
        )
    launcher._record_trainer_exit_observed(
        paths,
        manifest_path=manifest_path,
        pid=int(manifest["trainer_spawned_pid"]),
        returncode=returncode,
    )
    monkeypatch.setattr(
        launcher,
        "process_identity_matches",
        lambda _identity: False,
    )

    status = launcher.read_status(
        paths,
        enforce_active_root=False,
    )

    assert status["status"] == expected_status
    assert status["running"] is False
    assert status["terminal"]["returncode"] == returncode
    assert status["trainer_reaped_returncode"] == returncode
