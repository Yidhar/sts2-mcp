from __future__ import annotations

import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace

PACKAGE_ROOT = Path(__file__).resolve().parents[1]
CHECKOUT_ROOT = PACKAGE_ROOT.parents[1]
SCRIPT = PACKAGE_ROOT / "scripts" / "launch_v29_stall_recovery_350k.py"
CONTRACT = (
    CHECKOUT_ROOT
    / "contracts/exact-resume-checkpoints/v29-failure-credit-v4-170119-stall-recovery.json"
)
SPEC = importlib.util.spec_from_file_location("v29_stall_recovery_launcher", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
launcher = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(launcher)


def test_recovery_contract_is_pinned_to_latest_atomic_checkpoint() -> None:
    contract = json.loads(CONTRACT.read_text(encoding="utf-8"))
    source = contract["source_checkpoint"]
    continuation = contract["continuation"]

    assert contract["expected_load_mode"] == "exact_resume"
    assert contract["usage"] == "exact_resume_only"
    assert source["run_id"] == launcher.RECOVERY_SOURCE_RUN_ID
    assert source["checkpoint_id"] == launcher.RECOVERY_SOURCE_CHECKPOINT_ID
    assert source["environment_steps"] == launcher.RECOVERY_SOURCE_ENVIRONMENT_STEPS
    assert source["policy_version"] == launcher.RECOVERY_SOURCE_POLICY_VERSION
    assert source["learner_updates"] == launcher.RECOVERY_SOURCE_LEARNER_UPDATES
    assert continuation["additional_environment_steps"] == 179_881
    assert continuation["target_total_environment_steps"] == 350_000
    assert continuation["model_initialization"] is False


def test_recovery_supervisor_reenters_recovery_adapter() -> None:
    paths = SimpleNamespace(
        venv_python=Path("/runtime/environments/wsl-rocm/bin/python")
    )
    manifest = Path("/runtime/launchers/recovery.launch.json")
    command = launcher._build_supervisor_command(paths, manifest_path=manifest)

    assert command[1] == str(SCRIPT.resolve())
    assert command[-2:] == ("--manifest", str(manifest))


def test_recovery_has_an_independent_durable_supervisor_identity() -> None:
    assert launcher.RECOVERY_RUN_NAME.endswith("recovery-170119")
    paths = SimpleNamespace(artifact_root=Path("/runtime"))
    assert launcher._recovery_successor_log_root(paths) == (
        Path("/runtime/runs") / launcher.RECOVERY_SUCCESSOR_LOG_NAME
    ).resolve(strict=False)


def test_stall_watchdog_has_a_single_conservative_timeout() -> None:
    assert launcher.RECOVERY_LEARNER_STALL_TIMEOUT_SECONDS == 1800.0
