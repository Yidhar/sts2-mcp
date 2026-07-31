#!/usr/bin/env python3
"""Persistent, fail-closed WSL supervisor for failure-credit-v4 v29.

The public ``start`` action can launch exactly one training command: a
``model_initialization`` from :data:`V28_100K_FROZEN`.  It never accepts
trainer arguments and has no exact-resume path.  Before a process can be
spawned, the launcher authenticates the frozen checkpoint, the repository
shadow-validation contract, both shadow reports and every semantics/evidence
source file named by that contract.  It then proves the reviewed model
migration on fresh CPU resources: transaction-v3 disappears, the liveness
head family stays freshly initialized, and optimizer/replay/RNG/counters all
start a new lineage.

``start`` detaches a Linux supervisor, not a bare trainer.  The supervisor
records native process identities for itself and its child, binds the newly
created v29 ``metrics.jsonl`` by its model-initialization ``run_start``, and
turns every observed child exit into a durable terminal lifecycle.  Therefore
``status`` can reconcile an interrupted supervisor instead of leaving a dead
run as ``stale_unknown``.

This file deliberately has no stop command.  It must be invoked inside WSL::

    python3 scripts/launch_v29_failure_credit_v4.py preflight
    python3 scripts/launch_v29_failure_credit_v4.py start
    python3 scripts/launch_v29_failure_credit_v4.py status
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import re
import secrets
import selectors
import signal
import subprocess
import sys
import time
import uuid
from collections.abc import Mapping, Sequence
from contextlib import contextmanager
from copy import deepcopy
from dataclasses import asdict, dataclass, replace
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, TypeAlias, cast

try:  # pragma: no cover - Windows imports this module for unit tests.
    import fcntl
except ImportError:  # pragma: no cover
    fcntl = None  # type: ignore[assignment]

# ``scripts`` is intentionally not a Python package.  Reuse only the mature
# launcher's path/environment primitives.  The v29 config, frozen lineage,
# shadow evidence and migration proof are independently defined below; v28 is
# not modified and its training contract is never reused.
SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))
import preflight_v28_mature_refinement as _common_preflight  # noqa: E402
import supervised_trainer_bootstrap as _trainer_bootstrap  # noqa: E402

MANIFEST_SCHEMA = "sts2-v29-model-init-supervised-launch-v2"
STATE_SCHEMA = "sts2-v29-model-init-supervised-state-v2"
WATCHDOG_EVENT_SCHEMA = "sts2-native-exit-watchdog-v1"
SUPERVISOR_EMERGENCY_SCHEMA = "sts2-supervisor-emergency-v1"
SUPERVISED_LAUNCH_CONTRACT_SCHEMA = "sts2-supervised-training-launch-contract-v2"
TRAINER_BOOTSTRAP_PROTOCOL = _trainer_bootstrap.PROTOCOL_VERSION
TRAINER_BOOTSTRAP_READY_TIMEOUT_SECONDS = 30.0
TRAINER_BOOTSTRAP_EXEC_TIMEOUT_SECONDS = 30.0
RUN_NAME = "full-run-revival-v29-failure-credit-v4-model-init"

FROZEN_CHECKPOINT_CONTRACT_VERSION = "sts2-frozen-checkpoint-contract-v1"
FIXED_FROZEN_CONTRACT_NAME = "v28-mature-refinement-100k"
FIXED_INITIALIZATION_STEP = 100_000
FIXED_INITIALIZATION_POLICY_VERSION = 1_569
FIXED_INITIALIZATION_CHECKPOINT_ID = "f670deda-97d5-46f2-be07-d69a5842eeec"
FIXED_INITIALIZATION_MANIFEST_SHA256 = "cfd4d23bf20e064e7a1b21939e94ff01a9f5bf48ab7d70690c66d71f9f2e7af3"
FIXED_INITIALIZATION_METADATA_SHA256 = "e1933292cc0fbca95cd3b80e073b0b2cda40c1ea29731b59a4bfa21f4f8b5127"
FIXED_INITIALIZATION_SOURCE_GIT_COMMIT = "129094067b19c450030d7f8aaeb28f8963163137"
FIXED_INITIALIZATION_RELATIVE = Path(
    "checkpoints/full-run-revival-v28-mature-refinement-model-init/"
    "run-071a43f5-120a-485f-a6dc-a67a55a1efc2/"
    "periodic-step-000100000"
)
FIXED_CONFIG_FINGERPRINT_SHA256 = "78a258fa9a4b5b7fb418b013b51ebc603d15ab249191fa21c4986e3e4f5919e7"
FIXED_RUNTIME_CONFIG_FINGERPRINT_SHA256 = "96943e9bec6e554a39eb07f47fbcde5ae74b137e0bbc4b4f9de00256ec2be6b5"

CHECKPOINT_FORMAT_V5 = "sts2-recurrent-vtrace-checkpoint-v5"
FAILURE_CREDIT_SCHEMA_V4 = "sts2-failure-credit-v4"
FAILURE_CREDIT_REPLAY_V4 = "sts2-failure-evidence-replay-v4"
TRAINING_PIPELINE_V7 = "bounded-fifo-async-vtrace-failure-credit-v4-v7"
SHADOW_CONTRACT_SCHEMA = "sts2-shadow-validation-contract-v3"
SHADOW_CONTRACT_RELATIVE = Path("contracts/shadow-validation/v29-failure-credit-v4-bootstrap.json")
SHADOW_CONTRACT_SHA256 = "14714ded0bf29ad0870f86ae09e00d16a07e3bbb3f6a132041bc694b5ca1b486"
SEMANTICS_SHADOW_SHA256 = "7e9cd8f36a1c9cc7801ed57fe08a90c2f5ca5e07b52a7cf86e27bb97a64ed15c"
EVIDENCE_SHADOW_SHA256 = "aa10e4ee6511417590017db3cf307250d01a53f6118ec733cbf80a13f2ac47d8"
ACTOR_EVIDENCE_SHADOW_SHA256 = "781629162663b5be52fbcdc05bebbc4c3d9e30019547fcbd8c25cac4f7492bb7"
SHADOW_VALIDATED_CODE_COUNT = 48
SHADOW_VALIDATED_CODE_MAPPING_SHA256 = "bde478f0bec0930c20a1c08f04cc59e93f2bd92a11ee65d2157bd1f848defa0f"
# Updated together with the formal shadow contract after the final report set
# is frozen.  It authenticates the path-independent proof embedded in every
# supervised launch manifest, so lifecycle reads never need mutable reports.
SHADOW_AUTHORITY_SHA256 = "14cf8ed73ff85b46064ea428b6461cb12dbcec666254d57152ae47ba6e44f7f2"
RUNTIME_READINESS_REPORT_SCHEMA = "sts2-liveness-head-active-shape-stress-v3"
FORMAL_REPORT_GENERATION_SOURCE_SCHEMA = "sts2-formal-report-generation-source-v1"
FORMAL_REPORT_VALIDATORS = {
    "semantics_historical": "packages/rl-agent/scripts/validate_semantics_shadow.py",
    "evidence_live": "packages/rl-agent/scripts/validate_failure_evidence_shadow.py",
    "evidence_actor": "packages/rl-agent/scripts/validate_failure_actor_evidence_shadow.py",
    "runtime_readiness": "packages/rl-agent/scripts/validate_liveness_head_stress.py",
}
RUNTIME_READINESS_EVIDENCE_SCHEMA = "sts2-runtime-readiness-evidence-v1"
RUNTIME_READINESS_REPORT_RELATIVE = Path(
    "reports/failure-credit-v4/liveness-head-stress-v29.json",
)
RUNTIME_READINESS_REPORT_SHA256 = "73733771936b08844e68ed6745d0f5c00b38b2d7f6e98289d44f01af7a05eba3"
SOURCE_AUTHORITY_SCHEMA = "sts2-two-phase-source-authority-v1"
EXTERNAL_SEAL_SCHEMA = "sts2-external-launch-contract-seal-v1"
REVIEWED_SEAL_PATHS = (
    "contracts/shadow-validation/v29-failure-credit-v4-bootstrap.json",
    "packages/rl-agent/scripts/launch_v29_failure_credit_v4.py",
)
REQUIRED_SHADOW_CODE_PATHS = frozenset(
    {
        "packages/rl-agent/scripts/prepare_v29_authority_seal.py",
        "packages/rl-agent/scripts/validate_liveness_head_stress.py",
        "packages/rl-agent/scripts/supervised_trainer_bootstrap.py",
        "packages/rl-agent/sts2_rl/checkpoints/frozen.py",
        "packages/rl-agent/sts2_rl/training/checkpoint_evaluation.py",
        "packages/rl-agent/sts2_rl/training/checkpointing.py",
        "packages/rl-agent/sts2_rl/training/factory.py",
        "packages/rl-agent/sts2_rl/training/failure_credit/actor_eligibility.py",
        "packages/rl-agent/sts2_rl/training/launch_contract.py",
        "packages/rl-agent/sts2_rl/training/runtime.py",
        "packages/rl-agent/sts2_rl/train.py",
    },
)
TERMINAL_STATUSES = frozenset({"completed", "interrupted", "failed"})

# v29 deliberately does not inherit the launcher's ambient environment.  This
# is a process-boundary ABI: preflight, the detached supervisor, the bootstrap
# and the exec'd trainer must all observe this exact key set.  In particular,
# explicitly setting ``LC_CTYPE`` prevents CPython's locale coercion from
# adding it after ``execve(2)`` and thereby changing the trainer-side digest.
_V29_HERMETIC_TRAINER_ENVIRONMENT_KEYS = (
    "STS2_ARTIFACT_ROOT",
    "PYTHONPATH",
    "PYTHONNOUSERSITE",
    "PYTHONUNBUFFERED",
    "OMP_NUM_THREADS",
    "MKL_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "NUMEXPR_NUM_THREADS",
    "LC_CTYPE",
    "PATH",
)
_V29_HERMETIC_TRAINER_ENVIRONMENT_UNSET_KEYS = (
    "PYTHONHOME",
    "VENV_DIR",
    "STS2_HEADLESS_SIM_EXE",
)
_V29_HERMETIC_SYSTEM_PATH = (
    "/usr/bin",
    "/bin",
)
HERMETIC_RUNTIME_PROBE_SCHEMA = "sts2-v29-hermetic-runtime-probe-v1"
HERMETIC_RUNTIME_PROBE_ACTION = "_runtime-probe"
HERMETIC_RUNTIME_PROBE_TIMEOUT_SECONDS = 180.0
HERMETIC_RUNTIME_PROBE_MAX_OUTPUT_BYTES = 4 * 1024 * 1024


class LaunchError(RuntimeError):
    """The reviewed v29 launch contract could not be proven."""


PreflightPaths: TypeAlias = _common_preflight.PreflightPaths
PreflightError: TypeAlias = _common_preflight.PreflightError


class _V29PreflightNamespace:
    """Module-shaped compatibility surface used by lifecycle code and tests."""

    PreflightPaths = PreflightPaths
    PreflightError = PreflightError
    RUN_NAME = RUN_NAME

    @staticmethod
    def default_paths() -> _common_preflight.PreflightPaths:
        script_dir = Path(__file__).resolve().parent
        package_root = script_dir.parent
        checkout_root = script_dir.parents[2]
        artifact_root = _common_preflight.ACTIVE_ARTIFACT_ROOT
        simulator = artifact_root / "dependencies/sts2-ai/STS2AI/ENV/Sim/HeadlessSim/bin/Release/net9.0/HeadlessSim.exe"
        return _common_preflight.PreflightPaths(
            checkout_root=checkout_root,
            package_root=package_root,
            artifact_root=artifact_root,
            venv_python=artifact_root / "environments/wsl-rocm/bin/python",
            config_path=(package_root / "config/experiments/full_run_revival_v29_failure_credit_v4_model_init.toml"),
            simulator_executable=simulator,
            simulator_identity=simulator.with_name(simulator.name + ".identity.json"),
        )

    @staticmethod
    def validate_layout(
        paths: _common_preflight.PreflightPaths,
        *,
        enforce_active_root: bool = True,
    ) -> _common_preflight.PreflightPaths:
        return _common_preflight.validate_layout(
            paths,
            enforce_active_root=enforce_active_root,
        )

    @staticmethod
    def build_trainer_environment(
        paths: _common_preflight.PreflightPaths,
        *,
        base_environment: dict[str, str] | None = None,
    ) -> dict[str, str]:
        """Build the v29-only hermetic trainer environment.

        ``base_environment`` remains in the compatibility surface so tests and
        lifecycle callers can prove independence from two different ambient
        processes.  Its contents are intentionally never copied.  v28 keeps
        its historical ambient-overlay behavior in the common preflight.
        """

        del base_environment
        return {
            "STS2_ARTIFACT_ROOT": os.fspath(paths.artifact_root),
            "PYTHONPATH": os.fspath(paths.package_root),
            "PYTHONNOUSERSITE": "1",
            "PYTHONUNBUFFERED": "1",
            "OMP_NUM_THREADS": "4",
            "MKL_NUM_THREADS": "4",
            "OPENBLAS_NUM_THREADS": "4",
            "NUMEXPR_NUM_THREADS": "4",
            "LC_CTYPE": "C.UTF-8",
            "PATH": os.pathsep.join(
                (
                    os.fspath(paths.venv_python.parent),
                    *_V29_HERMETIC_SYSTEM_PATH,
                )
            ),
        }

    @classmethod
    def validate_trainer_environment(
        cls,
        environment: dict[str, str],
        *,
        paths: _common_preflight.PreflightPaths,
    ) -> dict[str, str]:
        expected = cls.build_trainer_environment(paths)
        actual_keys = set(environment)
        expected_keys = set(_V29_HERMETIC_TRAINER_ENVIRONMENT_KEYS)
        if actual_keys != expected_keys:
            raise PreflightError(
                "v29 hermetic trainer environment keys changed: "
                f"missing={sorted(expected_keys - actual_keys)} "
                f"extra={sorted(actual_keys - expected_keys)}"
            )
        if environment != expected:
            changed = sorted(key for key in _V29_HERMETIC_TRAINER_ENVIRONMENT_KEYS if environment[key] != expected[key])
            raise PreflightError("v29 hermetic trainer environment values changed: " + ", ".join(changed))
        return environment

    @staticmethod
    def trainer_environment_contract(
        environment: dict[str, str],
    ) -> dict[str, Any]:
        actual_keys = set(environment)
        expected_keys = set(_V29_HERMETIC_TRAINER_ENVIRONMENT_KEYS)
        if actual_keys != expected_keys:
            raise PreflightError(
                "v29 hermetic trainer environment contract keys changed: "
                f"missing={sorted(expected_keys - actual_keys)} "
                f"extra={sorted(actual_keys - expected_keys)}"
            )
        return {
            "set": {key: environment[key] for key in _V29_HERMETIC_TRAINER_ENVIRONMENT_KEYS},
            "unset": list(
                _V29_HERMETIC_TRAINER_ENVIRONMENT_UNSET_KEYS,
            ),
        }

    @staticmethod
    def build_trainer_command(
        paths: _common_preflight.PreflightPaths,
        *,
        initialize_from: Path,
    ) -> tuple[str, ...]:
        return (
            str(paths.venv_python),
            "-m",
            "sts2_rl.train",
            "--profile",
            "preheat",
            "--config",
            str(paths.config_path),
            "--device",
            "cuda",
            "--collector-device",
            "cpu",
            "--backend",
            "headless",
            "--sim-exe",
            str(paths.simulator_executable),
            "--sim-identity",
            str(paths.simulator_identity),
            "--initialize-from",
            str(initialize_from),
        )

    @classmethod
    def validate_trainer_command(
        cls,
        command: tuple[str, ...],
        *,
        paths: _common_preflight.PreflightPaths,
        initialize_from: Path,
    ) -> tuple[str, ...]:
        if "--resume" in command:
            raise PreflightError("v29 failure-credit-v4 must never masquerade as exact resume")
        if command.count("--initialize-from") != 1:
            raise PreflightError("v29 trainer command must contain one --initialize-from")
        index = command.index("--initialize-from")
        if index + 1 >= len(command) or command[index + 1] != str(initialize_from):
            raise PreflightError("v29 trainer command initialization source changed")
        expected = cls.build_trainer_command(
            paths,
            initialize_from=initialize_from,
        )
        if command != expected:
            raise PreflightError("trainer command differs from the reviewed v29 command")
        return command

    @staticmethod
    def run_preflight(
        paths: _common_preflight.PreflightPaths,
        *,
        initialize_from: str | Path,
        enforce_active_root: bool = True,
        verify_runtime: bool = True,
    ) -> dict[str, Any]:
        return _run_v29_preflight(
            paths,
            initialize_from=initialize_from,
            enforce_active_root=enforce_active_root,
            verify_runtime=verify_runtime,
        )


v29_preflight = _V29PreflightNamespace()


@dataclass(frozen=True, slots=True)
class LaunchPaths:
    preflight: PreflightPaths
    launcher_dir: Path
    manifest_dir: Path
    initialization_checkpoint: Path

    @property
    def checkout_root(self) -> Path:
        return self.preflight.checkout_root

    @property
    def package_root(self) -> Path:
        return self.preflight.package_root

    @property
    def artifact_root(self) -> Path:
        return self.preflight.artifact_root

    @property
    def venv_python(self) -> Path:
        return self.preflight.venv_python


@dataclass(frozen=True, slots=True)
class ProcessIdentity:
    pid: int
    proc_start_ticks: int
    command_line_sha256: str
    executable: str


def _utc_now() -> str:
    return datetime.now(UTC).isoformat()


def _sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _canonical_json_sha256(value: object) -> str:
    return _sha256_bytes(
        json.dumps(
            value,
            allow_nan=False,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8"),
    )


def _immutable_json_bytes(value: object) -> bytes:
    return (
        json.dumps(
            value,
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
        )
        + "\n"
    ).encode("utf-8")


def _is_within(path: Path, root: Path) -> bool:
    try:
        path.relative_to(root)
    except ValueError:
        return False
    return True


def _absolute_without_symlink_resolution(path: Path) -> Path:
    return Path(os.path.abspath(os.fspath(path.expanduser())))


def _v28_frozen_contract() -> Any:
    """Load and cross-check the repository authority without eager Torch import."""

    from sts2_rl.checkpoints.frozen import (
        FROZEN_CHECKPOINT_CONTRACT_VERSION as repository_version,
    )
    from sts2_rl.checkpoints.frozen import (
        V28_100K_FROZEN,
    )

    expected = {
        "name": FIXED_FROZEN_CONTRACT_NAME,
        "relative_path": FIXED_INITIALIZATION_RELATIVE,
        "checkpoint_id": FIXED_INITIALIZATION_CHECKPOINT_ID,
        "manifest_sha256": FIXED_INITIALIZATION_MANIFEST_SHA256,
        "metadata_sha256": FIXED_INITIALIZATION_METADATA_SHA256,
        "environment_steps": FIXED_INITIALIZATION_STEP,
        "policy_version": FIXED_INITIALIZATION_POLICY_VERSION,
        "source_git_commit": FIXED_INITIALIZATION_SOURCE_GIT_COMMIT,
        "exact_resume_permitted": False,
    }
    actual = {key: getattr(V28_100K_FROZEN, key) for key in expected}
    if repository_version != FROZEN_CHECKPOINT_CONTRACT_VERSION:
        raise LaunchError("frozen checkpoint contract ABI changed")
    if actual != expected:
        raise LaunchError("launcher constants disagree with V28_100K_FROZEN")
    return V28_100K_FROZEN


def default_paths() -> LaunchPaths:
    preflight = v29_preflight.default_paths()
    checkpoint = preflight.artifact_root / FIXED_INITIALIZATION_RELATIVE
    return LaunchPaths(
        preflight=preflight,
        launcher_dir=preflight.artifact_root / "launcher",
        manifest_dir=preflight.artifact_root / "launchers",
        initialization_checkpoint=checkpoint,
    )


def validate_layout(
    paths: LaunchPaths,
    *,
    enforce_active_root: bool = True,
) -> LaunchPaths:
    try:
        preflight = v29_preflight.validate_layout(
            paths.preflight,
            enforce_active_root=enforce_active_root,
        )
    except PreflightError as exc:
        raise LaunchError(str(exc)) from exc
    launcher_dir = paths.launcher_dir.resolve(strict=False)
    manifest_dir = paths.manifest_dir.resolve(strict=False)
    checkpoint = paths.initialization_checkpoint.resolve(strict=False)
    expected_checkpoint = (preflight.artifact_root / FIXED_INITIALIZATION_RELATIVE).resolve(strict=False)
    if checkpoint != expected_checkpoint:
        raise LaunchError(f"v29 launcher is pinned to the V28_100K_FROZEN contract; actual={checkpoint}")
    for candidate in (launcher_dir, manifest_dir, checkpoint):
        if not _is_within(candidate, preflight.artifact_root):
            raise LaunchError(f"v29 runtime path escapes the artifact root: {candidate}")
    return LaunchPaths(
        preflight=preflight,
        launcher_dir=launcher_dir,
        manifest_dir=manifest_dir,
        initialization_checkpoint=checkpoint,
    )


def require_wsl() -> None:
    if os.name != "posix" or not sys.platform.startswith("linux"):
        raise LaunchError("the v29 launcher must run inside WSL")
    version = ""
    for candidate in (Path("/proc/sys/kernel/osrelease"), Path("/proc/version")):
        try:
            version += candidate.read_text(encoding="utf-8", errors="replace")
        except OSError:
            continue
    if "microsoft" not in version.casefold():
        raise LaunchError("the v29 launcher requires a WSL Linux kernel")


def require_exact_artifact_environment(paths: LaunchPaths) -> None:
    raw = os.environ.get("STS2_ARTIFACT_ROOT")
    if raw and Path(raw).expanduser().resolve(strict=False) != paths.artifact_root:
        raise LaunchError("STS2_ARTIFACT_ROOT disagrees with the reviewed v29 runtime")


def atomic_write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.parent / f".contract-{uuid.uuid4().hex}.tmp"
    serialized = (json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n").encode("utf-8")
    descriptor: int | None = None
    try:
        descriptor = os.open(
            temporary,
            os.O_WRONLY | os.O_CREAT | os.O_EXCL,
            0o600,
        )
        with os.fdopen(descriptor, "wb", closefd=True) as handle:
            descriptor = None
            handle.write(serialized)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
        try:
            directory_fd = os.open(path.parent, os.O_RDONLY)
        except OSError:
            directory_fd = None
        if directory_fd is not None:
            try:
                os.fsync(directory_fd)
            finally:
                os.close(directory_fd)
    finally:
        if descriptor is not None:
            os.close(descriptor)
        temporary.unlink(missing_ok=True)


def _write_immutable_json(
    path: Path,
    payload: Mapping[str, Any],
) -> str:
    """Atomically publish one read-only JSON authority without replacement."""

    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        raise LaunchError(f"immutable launch contract already exists: {path}")
    temporary = path.parent / f".contract-{uuid.uuid4().hex}.tmp"
    serialized = _immutable_json_bytes(payload)
    descriptor: int | None = None
    try:
        descriptor = os.open(
            temporary,
            os.O_WRONLY | os.O_CREAT | os.O_EXCL,
            0o600,
        )
        with os.fdopen(descriptor, "wb", closefd=True) as handle:
            descriptor = None
            handle.write(serialized)
            handle.flush()
            os.fsync(handle.fileno())
        try:
            os.link(temporary, path)
        except FileExistsError as exc:
            raise LaunchError(
                f"immutable launch contract raced with another writer: {path}",
            ) from exc
        if os.name == "posix":
            os.chmod(path, 0o400)
        if os.name == "posix":
            directory_fd = os.open(path.parent, os.O_RDONLY)
            try:
                os.fsync(directory_fd)
            finally:
                os.close(directory_fd)
    finally:
        if descriptor is not None:
            os.close(descriptor)
        temporary.unlink(missing_ok=True)
    return _sha256_bytes(serialized)


def _load_json_object(path: Path, *, label: str) -> dict[str, Any]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise LaunchError(f"cannot read {label} {path}: {exc}") from exc
    if not isinstance(payload, dict):
        raise LaunchError(f"{label} must be a JSON object: {path}")
    return payload


def _run_checked(
    command: Sequence[str],
    *,
    cwd: Path,
    environment: Mapping[str, str],
) -> str:
    result = subprocess.run(
        tuple(command),
        cwd=cwd,
        env=dict(environment),
        check=False,
        capture_output=True,
        text=True,
        encoding="utf-8",
    )
    if result.returncode != 0:
        output = (result.stderr or result.stdout).strip()
        raise LaunchError(f"preflight command failed ({result.returncode}): {' '.join(command)}\n{output[-4000:]}")
    return result.stdout


def _environment_from_contract(contract: Mapping[str, Any]) -> dict[str, str]:
    if set(contract) != {"set", "unset"}:
        raise LaunchError(
            "reviewed trainer environment contract fields changed",
        )
    raw_set = contract.get("set")
    raw_unset = contract.get("unset")
    if not isinstance(raw_set, Mapping) or not all(
        isinstance(key, str) and isinstance(value, str) for key, value in raw_set.items()
    ):
        raise LaunchError("reviewed trainer environment has a malformed set contract")
    if not isinstance(raw_unset, list) or not all(isinstance(item, str) for item in raw_unset):
        raise LaunchError("reviewed trainer environment has a malformed unset contract")
    if set(raw_set) != set(_V29_HERMETIC_TRAINER_ENVIRONMENT_KEYS):
        raise LaunchError(
            "reviewed trainer environment set keys changed",
        )
    if raw_unset != list(_V29_HERMETIC_TRAINER_ENVIRONMENT_UNSET_KEYS):
        raise LaunchError(
            "reviewed trainer environment unset keys changed or reordered",
        )
    if set(raw_set) & set(raw_unset):
        raise LaunchError("reviewed trainer environment sets and unsets the same key")
    # Never reconstruct a reviewed process environment by overlaying it onto
    # this invocation's ambient state.  Preflight and start are intentionally
    # allowed to be separate processes; the contract itself is the complete
    # environment passed to supervisor/bootstrap/trainer.
    return {key: str(raw_set[key]) for key in _V29_HERMETIC_TRAINER_ENVIRONMENT_KEYS}


def _exact_trainer_environment_payload(
    environment: Mapping[str, str],
) -> dict[str, Any]:
    """Return the canonical full environment used for process binding."""

    if not all(isinstance(key, str) and isinstance(value, str) for key, value in environment.items()):
        raise LaunchError("trainer environment contains non-text entries")
    actual_keys = set(environment)
    expected_keys = set(_V29_HERMETIC_TRAINER_ENVIRONMENT_KEYS)
    if actual_keys != expected_keys:
        raise LaunchError(
            "trainer environment is not the exact v29 hermetic set: "
            f"missing={sorted(expected_keys - actual_keys)} "
            f"extra={sorted(actual_keys - expected_keys)}",
        )
    return {
        "set": {key: environment[key] for key in _V29_HERMETIC_TRAINER_ENVIRONMENT_KEYS},
        "unset": list(
            _V29_HERMETIC_TRAINER_ENVIRONMENT_UNSET_KEYS,
        ),
    }


def _exact_trainer_environment_sha256(
    environment: Mapping[str, str],
) -> str:
    return _canonical_json_sha256(
        _exact_trainer_environment_payload(environment),
    )


def _expected_abi_contract() -> dict[str, Any]:
    """Return the immutable ABI payload embedded in manifest and state."""

    return {
        "checkpoint_format": CHECKPOINT_FORMAT_V5,
        "frozen_checkpoint_contract": FROZEN_CHECKPOINT_CONTRACT_VERSION,
        "training_pipeline": TRAINING_PIPELINE_V7,
        "encoding": {
            "version": "grounded-relational-runtime-encoding-v13",
            "min_token_feature_dim": 224,
            "feature_abi_end": 215,
            "fingerprint_sha256": ("ac119f0d1fe0de5c09394e091169f3b7712084bce8a90a9d02be4732f60ce5bf"),
        },
        "long_horizon_value_heads": ("sts2-long-horizon-value-heads-v1"),
        "episodic_targets": ("sts2-episodic-task-targets-one-terminal-unit-v2"),
        "failure_credit": {
            "collector": "sts2-failure-credit-collector-v3",
            "compiler": "sts2-failure-credit-compiler-v1",
            "detector": "sts2-semantic-macro-cycle-detector-v3",
            "liveness_heads": "sts2-liveness-cost-heads-v1",
            "replay": FAILURE_CREDIT_REPLAY_V4,
            "schema": FAILURE_CREDIT_SCHEMA_V4,
        },
        "failure_credit_replay": FAILURE_CREDIT_REPLAY_V4,
        "decision_semantics": {
            "decision_identity": "sts2-decision-identity-v1",
            "macro_edge": "sts2-policy-macro-edge-v1",
            "progress_receipt": "sts2-progress-receipt-v1",
            "progress_scope": "sts2-progress-scope-v1",
            "semantic_key": "sts2-semantic-key-v1",
            "surface_registry": "sts2-surface-registry-v1",
        },
    }


def _abi_contract() -> dict[str, Any]:
    """Validate current executable code against the immutable v29 ABI."""

    from sts2_rl.encoding import grounding_encoding_identity
    from sts2_rl.training import checkpointing
    from sts2_rl.training import runtime as training_runtime

    expected = _expected_abi_contract()
    actual_checkpoint = str(checkpointing._CHECKPOINT_FORMAT)
    actual_failure_credit = checkpointing._failure_credit_abi()
    actual_decision_semantics = checkpointing._decision_semantics_abi()
    if actual_checkpoint != expected["checkpoint_format"]:
        raise LaunchError(f"v29 requires checkpoint ABI v5; actual={actual_checkpoint!r}")
    if actual_failure_credit != expected["failure_credit"]:
        raise LaunchError("failure-credit-v4 ABI changed")
    if actual_decision_semantics != expected["decision_semantics"]:
        raise LaunchError("decision-semantics ABI changed")
    if training_runtime._TRAINING_PIPELINE_ABI != expected["training_pipeline"]:
        raise LaunchError("failure-credit-v4 training pipeline ABI changed")
    if grounding_encoding_identity() != expected["encoding"]:
        raise LaunchError("grounded encoding-v13 ABI changed")
    if checkpointing._LONG_HORIZON_VALUE_HEAD_ABI != expected["long_horizon_value_heads"]:
        raise LaunchError("long-horizon value-head ABI changed")
    if checkpointing._EPISODIC_TARGET_ABI != expected["episodic_targets"]:
        raise LaunchError("episodic target ABI changed")
    return expected


def _load_v29_config(paths: PreflightPaths) -> Any:
    from sts2_rl.training.config import load_training_config

    if not paths.config_path.is_file():
        raise LaunchError(f"v29 config was not found: {paths.config_path}")
    config = load_training_config(
        profile="preheat",
        config_path=paths.config_path,
    )
    actual = {
        "architecture": config.model.architecture,
        "transaction_learning_enabled": config.transaction_learning.enabled,
        "failure_credit_mode": config.failure_credit.mode,
        "failure_credit_learning_enabled": (config.failure_credit.learning_enabled),
        "failure_credit_replay_capacity": (config.failure_credit.replay_capacity),
        "failure_credit_replay_byte_capacity": (config.failure_credit.replay_byte_capacity),
        "failure_credit_sample_records": config.failure_credit.sample_records,
        "failure_credit_burn_in_steps": config.failure_credit.burn_in_steps,
        "failure_credit_maximum_context_steps": (config.failure_credit.maximum_context_steps),
        "failure_credit_maximum_episode_completion_controls": (
            config.failure_credit.maximum_episode_completion_controls
        ),
        "failure_credit_maximum_episode_completion_bytes": (config.failure_credit.maximum_episode_completion_bytes),
        "direct_witness_quota": config.failure_credit.direct_witness_quota,
        "multi_edge_cycle_quota": (config.failure_credit.multi_edge_cycle_quota),
        "risk_sequence_quota": config.failure_credit.risk_sequence_quota,
        "unresolved_stall_quota": (config.failure_credit.unresolved_stall_quota),
        "completion_control_quota": (config.failure_credit.completion_control_quota),
        "matched_outcome_pair_quota": (config.failure_credit.matched_outcome_pair_quota),
        "failure_policy_gradient_max_lag": (config.failure_credit.policy_gradient_max_lag),
        "liveness_head_calibration_updates": (config.failure_credit.liveness_head_calibration_updates),
        "liveness_risk_actor_start_update": (config.failure_credit.liveness_risk_actor_start_update),
        "liveness_records_per_autograd_batch": (config.failure_credit.liveness_records_per_autograd_batch),
        "liveness_tbptt_window_steps": (config.failure_credit.liveness_tbptt_window_steps),
        "liveness_maximum_contexts_per_record": (config.failure_credit.liveness_maximum_contexts_per_record),
        "liveness_maximum_replayed_steps_per_update": (
            config.failure_credit.liveness_maximum_replayed_steps_per_update
        ),
        "liveness_maximum_replayed_candidates_per_update": (
            config.failure_credit.liveness_maximum_replayed_candidates_per_update
        ),
        "liveness_value_critic_weight": (config.failure_credit.liveness_value_critic_weight),
        "liveness_cost_critic_weight": (config.failure_credit.liveness_cost_critic_weight),
        "liveness_cost_actor_weight": (config.failure_credit.liveness_cost_actor_weight),
        "liveness_direct_policy_weight": (config.failure_credit.liveness_direct_policy_weight),
        "liveness_cycle_policy_weight": (config.failure_credit.liveness_cycle_policy_weight),
        "liveness_contrast_policy_weight": (config.failure_credit.liveness_contrast_policy_weight),
        "liveness_completion_policy_weight": (config.failure_credit.liveness_completion_policy_weight),
        "liveness_risk_advantage_clip": (config.failure_credit.liveness_risk_advantage_clip),
        "liveness_contrast_margin": (config.failure_credit.liveness_contrast_margin),
        "episodic_learning_enabled": config.episodic_learning.enabled,
        "fresh_policy_sequences": (config.episodic_learning.fresh_policy_sequences),
        "revival_budget": config.curriculum.revival_budget,
        "total_environment_steps": config.runtime.total_environment_steps,
        "seed": config.runtime.seed,
        "rocm_sdpa_backend": config.runtime.rocm_sdpa_backend,
        "evaluation_steps": config.runtime.evaluation_steps,
        "evaluation_episodes": config.runtime.evaluation_episodes,
        "early_evaluation_steps": config.runtime.early_evaluation_steps,
        "early_evaluation_episodes": config.runtime.early_evaluation_episodes,
        "final_audit_steps": config.runtime.final_audit_steps,
        "final_audit_episodes": config.runtime.final_audit_episodes,
        "evaluation_liveness_guard_enabled": (config.runtime.evaluation_liveness_guard_enabled),
    }
    expected = {
        "architecture": "relational_candidate_v3",
        "transaction_learning_enabled": False,
        "failure_credit_mode": "learning",
        "failure_credit_learning_enabled": True,
        "failure_credit_replay_capacity": 4_096,
        "failure_credit_replay_byte_capacity": 536_870_912,
        "failure_credit_sample_records": 8,
        "failure_credit_burn_in_steps": 32,
        "failure_credit_maximum_context_steps": 256,
        "failure_credit_maximum_episode_completion_controls": 32,
        "failure_credit_maximum_episode_completion_bytes": 134_217_728,
        "direct_witness_quota": 1,
        "multi_edge_cycle_quota": 1,
        "risk_sequence_quota": 1,
        "unresolved_stall_quota": 1,
        "completion_control_quota": 1,
        "matched_outcome_pair_quota": 0,
        "failure_policy_gradient_max_lag": 128,
        "liveness_head_calibration_updates": 256,
        "liveness_risk_actor_start_update": 512,
        "liveness_records_per_autograd_batch": 1,
        "liveness_tbptt_window_steps": 16,
        "liveness_maximum_contexts_per_record": 2,
        "liveness_maximum_replayed_steps_per_update": 4_096,
        "liveness_maximum_replayed_candidates_per_update": 1_048_576,
        "liveness_value_critic_weight": 0.10,
        "liveness_cost_critic_weight": 0.25,
        "liveness_cost_actor_weight": 0.10,
        "liveness_direct_policy_weight": 0.25,
        "liveness_cycle_policy_weight": 0.10,
        "liveness_contrast_policy_weight": 0.10,
        "liveness_completion_policy_weight": 0.0,
        "liveness_risk_advantage_clip": 0.25,
        "liveness_contrast_margin": 0.10,
        "episodic_learning_enabled": True,
        "fresh_policy_sequences": 1,
        "revival_budget": -1,
        "total_environment_steps": 100_000,
        "seed": 4_000_000,
        "rocm_sdpa_backend": "math",
        "evaluation_steps": (0, 25_000, 50_000, 75_000),
        "evaluation_episodes": 16,
        "early_evaluation_steps": (5_000, 10_000),
        "early_evaluation_episodes": 8,
        "final_audit_steps": (100_000,),
        "final_audit_episodes": 64,
        "evaluation_liveness_guard_enabled": True,
    }
    if actual != expected:
        raise LaunchError(
            "effective v29 failure-credit-v4 config changed: "
            + json.dumps(
                {"actual": actual, "expected": expected},
                default=list,
                sort_keys=True,
            )
        )
    if config.environment.backend != "headless" or config.environment.scenario != "full-run":
        raise LaunchError("v29 requires the headless full-run environment")
    if not config.runtime.log_dir.endswith(RUN_NAME):
        raise LaunchError("v29 log directory changed")
    if not config.runtime.checkpoint_dir.endswith(RUN_NAME):
        raise LaunchError("v29 checkpoint directory changed")
    fingerprint = config.fingerprint_sha256()
    if fingerprint != FIXED_CONFIG_FINGERPRINT_SHA256:
        raise LaunchError(
            f"v29 config fingerprint changed: expected={FIXED_CONFIG_FINGERPRINT_SHA256} actual={fingerprint}"
        )
    return config


def _effective_v29_config(config: Any, *, paths: PreflightPaths) -> Any:
    """Mirror ``sts2_rl.train._cli_overrides`` for the only allowed command."""

    return replace(
        config,
        runtime=replace(
            config.runtime,
            device="cuda",
            collector_device="cpu",
        ),
        environment=replace(
            config.environment,
            backend="headless",
            sim_exe_path=str(paths.simulator_executable),
        ),
    )


def _expected_effective_config_fingerprint(paths: LaunchPaths) -> str:
    config = _load_v29_config(paths.preflight)
    return cast(
        str,
        _effective_v29_config(
            config,
            paths=paths.preflight,
        ).fingerprint_sha256(),
    )


def _checked_repo_relative(
    raw: object,
    *,
    root: Path,
    label: str,
) -> Path:
    if not isinstance(raw, str) or not raw:
        raise LaunchError(f"{label} must be a non-empty relative path")
    lexical = Path(raw)
    if lexical.is_absolute() or ".." in lexical.parts:
        raise LaunchError(f"{label} escapes its authority root")
    resolved = (root / lexical).resolve(strict=False)
    if not _is_within(resolved, root):
        raise LaunchError(f"{label} escapes its authority root")
    return resolved


def _require_report_header(
    report: Mapping[str, Any],
    spec: Mapping[str, Any],
    *,
    label: str,
) -> None:
    for key in ("status", "version"):
        if report.get(key) != spec.get(key):
            raise LaunchError(f"{label} {key} changed")
    if report.get("read_only") is not True:
        raise LaunchError(f"{label} must be read-only")
    if report.get("training_authority") is not False:
        raise LaunchError(f"{label} must have no training authority")


def _validate_semantics_shadow(
    report: Mapping[str, Any],
    spec: Mapping[str, Any],
) -> dict[str, Any]:
    _require_report_header(
        report,
        spec,
        label="semantics shadow report",
    )
    errors = report.get("errors")
    maximum_errors = spec.get("maximum_errors")
    if (
        not isinstance(errors, list)
        or isinstance(maximum_errors, bool)
        or not isinstance(maximum_errors, int)
        or len(errors) > maximum_errors
    ):
        raise LaunchError("semantics shadow report contains contract errors")
    counts = report.get("counts")
    gates = report.get("gates")
    if not isinstance(counts, Mapping) or not isinstance(gates, Mapping):
        raise LaunchError("semantics shadow report has malformed gates/counts")
    for count_key, contract_key in (
        ("parsed_records", "minimum_parsed_records"),
        ("rich_decision_snapshots", "minimum_rich_decision_snapshots"),
    ):
        value = counts.get(count_key)
        minimum = spec.get(contract_key)
        if (
            isinstance(value, bool)
            or not isinstance(value, int)
            or isinstance(minimum, bool)
            or not isinstance(minimum, int)
            or value < minimum
        ):
            raise LaunchError(f"semantics shadow report failed {contract_key}")
    for key in (
        "minimum_rich_records",
        "no_contract_errors",
        "unknown_transition_budget",
    ):
        if gates.get(key) is not True:
            raise LaunchError(f"semantics shadow gate failed: {key}")
    if gates.get("required_minimum_rich_records") != 1_000:
        raise LaunchError("semantics shadow required-rich-record gate changed")
    maximum_unknown = spec.get("maximum_unknown_transitions")
    observed_unknown = gates.get("observed_unknown_transitions")
    declared_maximum = gates.get("maximum_unknown_transitions")
    if (
        isinstance(maximum_unknown, bool)
        or not isinstance(maximum_unknown, int)
        or isinstance(observed_unknown, bool)
        or not isinstance(observed_unknown, int)
        or isinstance(declared_maximum, bool)
        or not isinstance(declared_maximum, int)
        or observed_unknown != maximum_unknown
        or declared_maximum != maximum_unknown
    ):
        raise LaunchError("semantics shadow has unknown transitions")
    semantic_manifest = report.get("semantic_manifest")
    expected_manifest = {
        "namespace": "surface_registry_manifest",
        "schema_version": "sts2-surface-registry-v1",
        "digest": ("ff7b2e40a2aeb6cd29fd5db5ea9f42998a285deadc175d7a2480de50a0a97bf8"),
    }
    if not isinstance(semantic_manifest, Mapping):
        raise LaunchError("semantics shadow registry manifest is malformed")
    for key, expected in expected_manifest.items():
        if semantic_manifest.get(key) != expected:
            raise LaunchError(f"semantics shadow registry manifest {key} changed")
    manifest_payload = semantic_manifest.get("payload")
    if (
        not isinstance(manifest_payload, Mapping)
        or manifest_payload.get("contract_version") != "sts2-surface-registry-v1"
    ):
        raise LaunchError("semantics shadow registry contract version changed")
    return {
        "status": report["status"],
        "parsed_records": counts["parsed_records"],
        "rich_decision_snapshots": counts["rich_decision_snapshots"],
        "unknown_transitions": observed_unknown,
        "errors": len(errors),
    }


def _validate_evidence_shadow(
    report: Mapping[str, Any],
    spec: Mapping[str, Any],
    *,
    paths: LaunchPaths,
) -> dict[str, Any]:
    _require_report_header(
        report,
        spec,
        label="failure-evidence shadow report",
    )
    gates = report.get("gates")
    if not isinstance(gates, Mapping) or not gates:
        raise LaunchError("failure-evidence shadow report has no gates")
    if (
        spec.get("require_all_gates") is not True
        or spec.get("require_read_only") is not True
        or spec.get("require_no_training_authority") is not True
    ):
        raise LaunchError("failure-evidence contract must require all gates")
    expected_gates = {
        "completion_staging_is_bounded",
        "minimum_decisions",
        "no_replay_or_learner_updates",
        "no_unjustified_completion_prefer",
        "source_manifest_unchanged",
        "source_metadata_unchanged",
        "zero_semantic_censored_transitions",
    }
    if set(gates) != expected_gates:
        raise LaunchError("failure-evidence shadow gate set changed")
    if any(value is not True for value in gates.values()):
        failed = sorted(key for key, value in gates.items() if value is not True)
        raise LaunchError("failure-evidence shadow gates failed: " + ", ".join(failed))
    unchanged = report.get("training_state_unchanged")
    expected_unchanged = {
        "collector_model",
        "episodic_replay",
        "model",
        "optimizer",
        "rollout_queue",
        "transaction_replay",
    }
    if (
        not isinstance(unchanged, Mapping)
        or set(unchanged) != expected_unchanged
        or any(value is not True for value in unchanged.values())
    ):
        raise LaunchError("failure-evidence shadow mutated training state")
    counts = report.get("counts")
    if not isinstance(counts, Mapping):
        raise LaunchError("failure-evidence shadow counts are malformed")
    decisions = counts.get("decisions")
    minimum_decisions = spec.get("minimum_decisions")
    censored = counts.get("semantic_censored_transitions")
    maximum_censored = spec.get("maximum_semantic_censored_transitions")
    if (
        isinstance(decisions, bool)
        or not isinstance(decisions, int)
        or isinstance(minimum_decisions, bool)
        or not isinstance(minimum_decisions, int)
        or decisions < minimum_decisions
    ):
        raise LaunchError("failure-evidence shadow has too few decisions")
    if (
        isinstance(censored, bool)
        or not isinstance(censored, int)
        or isinstance(maximum_censored, bool)
        or not isinstance(maximum_censored, int)
        or censored > maximum_censored
    ):
        raise LaunchError("failure-evidence shadow contains semantic-censored transitions")
    direct_targets = report.get("direct_policy_targets")
    if not isinstance(direct_targets, Mapping):
        raise LaunchError("failure-evidence direct policy targets are malformed")
    if not set(direct_targets).issubset({"avoid", "prefer"}) or any(
        isinstance(count, bool) or not isinstance(count, int) or count < 0 for count in direct_targets.values()
    ):
        raise LaunchError(
            "failure-evidence direct policy targets must use lowercase production enum values",
        )
    prefer = direct_targets.get("prefer", 0)
    maximum_prefer = spec.get("maximum_prefer_targets")
    if (
        isinstance(prefer, bool)
        or not isinstance(prefer, int)
        or isinstance(maximum_prefer, bool)
        or not isinstance(maximum_prefer, int)
        or prefer > maximum_prefer
    ):
        raise LaunchError("failure-evidence shadow contains unjustified PREFER targets")
    maximum_completion_controls = spec.get(
        "maximum_episode_completion_controls",
    )
    maximum_completion_bytes = spec.get(
        "maximum_episode_completion_bytes",
    )
    episodes = report.get("episodes")
    if (
        isinstance(maximum_completion_controls, bool)
        or not isinstance(maximum_completion_controls, int)
        or maximum_completion_controls != 32
        or isinstance(maximum_completion_bytes, bool)
        or not isinstance(maximum_completion_bytes, int)
        or maximum_completion_bytes != 134_217_728
        or not isinstance(episodes, list)
        or not episodes
    ):
        raise LaunchError("failure-evidence completion-staging contract changed")
    observed_episode_completion_bytes = 0
    for index, episode in enumerate(episodes):
        if not isinstance(episode, Mapping):
            raise LaunchError(
                f"failure-evidence episode {index} is malformed",
            )
        shadow = episode.get("shadow")
        if not isinstance(shadow, Mapping):
            raise LaunchError(
                f"failure-evidence episode {index} shadow is malformed",
            )
        retained = shadow.get("completion_controls")
        storage_bytes = shadow.get("completion_storage_nbytes")
        observed = shadow.get("completion_controls_observed")
        dropped = shadow.get("completion_controls_dropped")
        if (
            isinstance(retained, bool)
            or not isinstance(retained, int)
            or not 0 <= retained <= maximum_completion_controls
            or isinstance(storage_bytes, bool)
            or not isinstance(storage_bytes, int)
            or not 0 <= storage_bytes <= maximum_completion_bytes
            or isinstance(observed, bool)
            or not isinstance(observed, int)
            or observed < retained
            or isinstance(dropped, bool)
            or not isinstance(dropped, int)
            or dropped != observed - retained
        ):
            raise LaunchError(
                f"failure-evidence episode {index} completion staging is not bounded",
            )
        observed_episode_completion_bytes = max(
            observed_episode_completion_bytes,
            storage_bytes,
        )
    top_level_maximum_bytes = counts.get(
        "maximum_episode_completion_storage_nbytes",
    )
    if top_level_maximum_bytes != observed_episode_completion_bytes:
        raise LaunchError(
            "failure-evidence maximum episode completion storage changed",
        )
    source = report.get("source")
    if not isinstance(source, Mapping):
        raise LaunchError("failure-evidence shadow source is malformed")
    expected_source = {
        "checkpoint_id": FIXED_INITIALIZATION_CHECKPOINT_ID,
        "environment_steps": FIXED_INITIALIZATION_STEP,
        "policy_version": FIXED_INITIALIZATION_POLICY_VERSION,
        "manifest_sha256": FIXED_INITIALIZATION_MANIFEST_SHA256,
        "metadata_sha256": FIXED_INITIALIZATION_METADATA_SHA256,
    }
    if spec.get("source_checkpoint_id") != FIXED_INITIALIZATION_CHECKPOINT_ID:
        raise LaunchError("failure-evidence contract source checkpoint changed")
    for key, expected in expected_source.items():
        if source.get(key) != expected:
            raise LaunchError(f"failure-evidence shadow source {key} changed")
    checkpoint = source.get("checkpoint")
    if not isinstance(checkpoint, str) or Path(checkpoint).resolve(strict=False) != paths.initialization_checkpoint:
        raise LaunchError("failure-evidence shadow source checkpoint path changed")
    report_abi = report.get("failure_credit_abi")
    current_abi = _expected_abi_contract()["failure_credit"]
    if not isinstance(report_abi, Mapping):
        raise LaunchError("failure-evidence shadow ABI is malformed")
    for key in ("schema", "collector", "detector", "compiler", "replay"):
        if report_abi.get(key) != current_abi[key]:
            raise LaunchError(f"failure-evidence shadow {key} ABI changed")
    return {
        "status": report["status"],
        "decisions": decisions,
        "semantic_censored_transitions": censored,
        "prefer_targets": prefer,
        "maximum_episode_completion_storage_nbytes": (observed_episode_completion_bytes),
        "completion_staging_is_bounded": True,
        "source_checkpoint_id": source["checkpoint_id"],
        "training_state_unchanged": True,
    }


def _validate_actor_evidence_shadow(
    report: Mapping[str, Any],
    spec: Mapping[str, Any],
) -> dict[str, Any]:
    """Validate the reviewed actor-label and production-mask contract."""

    _require_report_header(
        report,
        spec,
        label="failure actor-evidence shadow report",
    )
    if report.get("fixture_authority") != "reviewed-contract-cases-only":
        raise LaunchError("failure actor-evidence fixture authority changed")
    if (
        spec.get("require_all_gates") is not True
        or spec.get("require_read_only") is not True
        or spec.get("require_no_training_authority") is not True
    ):
        raise LaunchError(
            "failure actor-evidence contract must require all gates",
        )
    gates = report.get("gates")
    expected_gates = {
        "abandoned_cycle_has_zero_direct_or_cycle_blame",
        "all_contexts_bounded",
        "censored_has_zero_learning_targets",
        "direct_avoid_nonzero",
        "forced_only_has_zero_actor_blame",
        "learner_mask_dry_run_passed",
        "multi_edge_cycle_nonzero",
        "no_generic_prefer_target",
        "no_last_action_fallback",
        "unresolved_stall_nonzero",
    }
    if not isinstance(gates, Mapping) or set(gates) != expected_gates:
        raise LaunchError("failure actor-evidence gate set changed")
    if any(value is not True for value in gates.values()):
        failed = sorted(key for key, value in gates.items() if value is not True)
        raise LaunchError(
            "failure actor-evidence gates failed: " + ", ".join(failed),
        )

    report_abi = report.get("abi")
    current_abi = _expected_abi_contract()["failure_credit"]
    if not isinstance(report_abi, Mapping) or set(report_abi) != {
        "schema",
        "collector",
        "detector",
        "compiler",
    }:
        raise LaunchError("failure actor-evidence ABI is malformed")
    for key in ("schema", "collector", "detector", "compiler"):
        if report_abi.get(key) != current_abi[key]:
            raise LaunchError(f"failure actor-evidence {key} ABI changed")

    counts = report.get("counts")
    if not isinstance(counts, Mapping) or set(counts) != {
        "cases",
        "actor_actionable_records",
        "risk_sequence_records",
        "strata",
        "direct_targets",
    }:
        raise LaunchError("failure actor-evidence counts are malformed")
    for count_key, spec_key in (
        ("cases", "minimum_cases"),
        ("actor_actionable_records", "minimum_actor_actionable_records"),
        ("risk_sequence_records", "minimum_risk_sequence_records"),
    ):
        observed = counts.get(count_key)
        minimum = spec.get(spec_key)
        if (
            isinstance(observed, bool)
            or not isinstance(observed, int)
            or isinstance(minimum, bool)
            or not isinstance(minimum, int)
            or observed < minimum
        ):
            raise LaunchError(
                f"failure actor-evidence failed {spec_key}",
            )
    direct_targets = counts.get("direct_targets")
    maximum_prefer = spec.get("maximum_prefer_targets")
    if (
        not isinstance(direct_targets, Mapping)
        or not set(direct_targets).issubset({"avoid", "prefer"})
        or any(isinstance(value, bool) or not isinstance(value, int) or value < 0 for value in direct_targets.values())
        or direct_targets.get("avoid", 0) < 1
        or isinstance(maximum_prefer, bool)
        or not isinstance(maximum_prefer, int)
        or direct_targets.get("prefer", 0) > maximum_prefer
    ):
        raise LaunchError(
            "failure actor-evidence direct targets changed",
        )
    strata = counts.get("strata")
    expected_strata = {
        "CENSORED",
        "DIRECT_WITNESS",
        "MULTI_EDGE_CYCLE",
        "RISK_SEQUENCE",
        "UNRESOLVED_STALL",
    }
    if (
        not isinstance(strata, Mapping)
        or set(strata) != expected_strata
        or any(isinstance(value, bool) or not isinstance(value, int) or value < 1 for value in strata.values())
    ):
        raise LaunchError("failure actor-evidence strata changed")

    dry_run = report.get("learner_mask_dry_run")
    if not isinstance(dry_run, Mapping):
        raise LaunchError("failure actor-evidence learner-mask dry run is missing")
    expected_dry_run_keys = {
        "batch_manifests",
        "case_manifests",
        "checks",
        "config",
        "current_policy_version",
        "interface",
        "model_constructed",
        "optimizer_constructed",
        "phase_updates",
        "pure_dry_run",
        "replay_constructed",
    }
    if (
        set(dry_run) != expected_dry_run_keys
        or dry_run.get("interface") != "sts2_rl.training.learner.compile_liveness_label_manifest"
        or dry_run.get("pure_dry_run") is not True
        or dry_run.get("model_constructed") is not False
        or dry_run.get("optimizer_constructed") is not False
        or dry_run.get("replay_constructed") is not False
        or dry_run.get("current_policy_version") != 11
        or dry_run.get("config")
        != {
            "policy_gradient_max_lag": 128,
            "liveness_head_calibration_updates": 256,
            "liveness_risk_actor_start_update": 512,
        }
        or dry_run.get("phase_updates")
        != {
            "calibration_update_0": 0,
            "mature_risk_start": 512,
        }
    ):
        raise LaunchError(
            "failure actor-evidence learner-mask dry-run contract changed",
        )
    expected_checks = {
        "learner_abandoned_cycle_has_no_effective_direct_or_cycle_blame",
        "learner_direct_witness_effective_during_and_after_calibration",
        "learner_forced_and_censored_have_zero_effective_actor",
        "learner_multi_edge_cycle_effective_during_and_after_calibration",
        "learner_phase_boundary_correct",
        "learner_unique_risk_effective_at_mature_start",
        "learner_unique_risk_suppressed_during_calibration",
    }
    checks = dry_run.get("checks")
    if (
        not isinstance(checks, Mapping)
        or set(checks) != expected_checks
        or any(value is not True for value in checks.values())
    ):
        raise LaunchError(
            "failure actor-evidence learner-mask dry-run checks failed",
        )
    case_manifests = dry_run.get("case_manifests")
    expected_cases = {
        "abandoned_cycle_suffix",
        "censored_boundary",
        "forced_only_stall",
        "linger9_death_warning_direct",
        "room_full_of_cheese_multi_edge",
        "unique_unresolved_stall",
    }
    if not isinstance(case_manifests, Mapping) or set(case_manifests) != expected_cases:
        raise LaunchError(
            "failure actor-evidence learner-mask cases changed",
        )
    phase_names = {"calibration_update_0", "mature_risk_start"}
    for case_name, phase_manifests in case_manifests.items():
        if not isinstance(phase_manifests, Mapping) or set(phase_manifests) != phase_names:
            raise LaunchError(
                f"failure actor-evidence learner-mask phases changed for {case_name}",
            )
        for phase_name, summary in phase_manifests.items():
            if not isinstance(summary, Mapping):
                raise LaunchError(
                    f"failure actor-evidence learner-mask summary changed for {case_name}/{phase_name}",
                )
            mask_counts = summary.get("mask_counts")
            if not isinstance(mask_counts, Mapping):
                raise LaunchError(
                    f"failure actor-evidence learner-mask counts changed for {case_name}/{phase_name}",
                )

    def _mask_count(
        case_name: str,
        phase_name: str,
        key: str,
    ) -> int:
        phase_manifests = cast(
            Mapping[str, Any],
            cast(Mapping[str, Any], case_manifests)[case_name],
        )
        summary = cast(Mapping[str, Any], phase_manifests[phase_name])
        mask_counts = cast(Mapping[str, Any], summary["mask_counts"])
        value = mask_counts.get(key)
        if isinstance(value, bool) or not isinstance(value, int):
            raise LaunchError(
                f"failure actor-evidence learner-mask count {case_name}/{phase_name}/{key} changed",
            )
        return value

    for phase_name in phase_names:
        if (
            _mask_count(
                "linger9_death_warning_direct",
                phase_name,
                "effective_direct_rows",
            )
            != 1
            or _mask_count(
                "room_full_of_cheese_multi_edge",
                phase_name,
                "effective_cycle_groups",
            )
            != 1
        ):
            raise LaunchError(
                "failure actor-evidence direct/cycle learner masks changed",
            )
        for zero_case in ("forced_only_stall", "censored_boundary"):
            if any(
                _mask_count(zero_case, phase_name, key) != 0
                for key in (
                    "effective_direct_rows",
                    "effective_risk_rows",
                    "effective_cycle_groups",
                    "effective_contrast_groups",
                )
            ):
                raise LaunchError(
                    "failure actor-evidence forced/censored mask changed",
                )
    if (
        _mask_count(
            "unique_unresolved_stall",
            "calibration_update_0",
            "effective_risk_rows",
        )
        != 0
        or _mask_count(
            "unique_unresolved_stall",
            "mature_risk_start",
            "effective_risk_rows",
        )
        <= 0
    ):
        raise LaunchError(
            "failure actor-evidence risk phase boundary changed",
        )

    return {
        "status": report["status"],
        "cases": counts["cases"],
        "actor_actionable_records": counts["actor_actionable_records"],
        "risk_sequence_records": counts["risk_sequence_records"],
        "prefer_targets": direct_targets.get("prefer", 0),
        "learner_mask_dry_run_passed": True,
    }


def _validate_git_source_mapping(
    value: object,
    *,
    checkout_root: Path | None,
    label: str,
) -> dict[str, Any]:
    if not isinstance(value, Mapping) or set(value) != {
        "checkout_root",
        "git_object_format",
        "implementation_commit",
        "implementation_tree",
        "worktree_clean",
    }:
        raise LaunchError(f"{label} is malformed")
    object_format = value.get("git_object_format")
    expected_length = 40 if object_format == "sha1" else 64 if object_format == "sha256" else 0
    commit = value.get("implementation_commit")
    tree = value.get("implementation_tree")
    raw_root = value.get("checkout_root")
    if not isinstance(raw_root, str) or not Path(raw_root).is_absolute():
        raise LaunchError(f"{label} is malformed")
    normalized_root = Path(raw_root).resolve(strict=False)
    if (checkout_root is not None and normalized_root != checkout_root.resolve(strict=False)) or (
        expected_length == 0
        or not isinstance(commit, str)
        or re.fullmatch(rf"[0-9a-f]{{{expected_length}}}", commit) is None
        or not isinstance(tree, str)
        or re.fullmatch(rf"[0-9a-f]{{{expected_length}}}", tree) is None
        or value.get("worktree_clean") is not True
    ):
        raise LaunchError(f"{label} is malformed")
    return {
        "checkout_root": str(normalized_root),
        "git_object_format": object_format,
        "implementation_commit": commit,
        "implementation_tree": tree,
        "worktree_clean": True,
    }


def _validate_formal_report_generation_source(
    value: object,
    *,
    expected_source: Mapping[str, Any],
    validator_relative: str,
    validated_code: Mapping[str, str],
    checkout_root: Path,
    label: str,
) -> dict[str, Any]:
    expected_fields = set(expected_source) | {
        "schema_version",
        "validator_relative_path",
        "validator_sha256",
    }
    if not isinstance(value, Mapping) or set(value) != expected_fields:
        raise LaunchError(f"{label} generation source is malformed")
    source = _validate_git_source_mapping(
        {key: value.get(key) for key in expected_source},
        checkout_root=checkout_root,
        label=f"{label} generation source",
    )
    validator_digest = value.get("validator_sha256")
    if (
        source != dict(expected_source)
        or value.get("schema_version") != FORMAL_REPORT_GENERATION_SOURCE_SCHEMA
        or value.get("validator_relative_path") != validator_relative
        or not isinstance(validator_digest, str)
        or validated_code.get(validator_relative) != validator_digest
    ):
        raise LaunchError(f"{label} was not generated by the reviewed clean source")
    return dict(value)


def _shadow_authority_payload(value: Mapping[str, Any]) -> dict[str, Any]:
    """Return the path-independent shadow proof authenticated by source."""

    contract = value.get("contract")
    generation_source = value.get("generation_source")
    code_hashes = value.get("validated_code_sha256")
    reports = value.get("reports")
    semantics = reports.get("semantics_historical") if isinstance(reports, Mapping) else None
    evidence = reports.get("evidence_live") if isinstance(reports, Mapping) else None
    actor_evidence = reports.get("evidence_actor") if isinstance(reports, Mapping) else None
    if (
        not isinstance(contract, Mapping)
        or not isinstance(generation_source, Mapping)
        or not isinstance(code_hashes, Mapping)
        or not isinstance(semantics, Mapping)
        or not isinstance(evidence, Mapping)
        or not isinstance(actor_evidence, Mapping)
    ):
        raise LaunchError("shadow authority proof is malformed")
    return {
        "contract": {
            "name": contract.get("name"),
            "schema_version": contract.get("schema_version"),
            "sha256": contract.get("sha256"),
        },
        "generation_source": dict(generation_source),
        "validated_code_sha256": dict(code_hashes),
        "reports": {
            "semantics_historical": {
                "sha256": semantics.get("sha256"),
                "status": semantics.get("status"),
                "parsed_records": semantics.get("parsed_records"),
                "rich_decision_snapshots": semantics.get(
                    "rich_decision_snapshots",
                ),
                "unknown_transitions": semantics.get(
                    "unknown_transitions",
                ),
                "errors": semantics.get("errors"),
            },
            "evidence_live": {
                "sha256": evidence.get("sha256"),
                "status": evidence.get("status"),
                "decisions": evidence.get("decisions"),
                "semantic_censored_transitions": evidence.get(
                    "semantic_censored_transitions",
                ),
                "prefer_targets": evidence.get("prefer_targets"),
                "source_checkpoint_id": evidence.get(
                    "source_checkpoint_id",
                ),
                "training_state_unchanged": evidence.get(
                    "training_state_unchanged",
                ),
                "maximum_episode_completion_storage_nbytes": evidence.get(
                    "maximum_episode_completion_storage_nbytes",
                ),
                "completion_staging_is_bounded": evidence.get(
                    "completion_staging_is_bounded",
                ),
            },
            "evidence_actor": {
                "sha256": actor_evidence.get("sha256"),
                "status": actor_evidence.get("status"),
                "cases": actor_evidence.get("cases"),
                "actor_actionable_records": actor_evidence.get(
                    "actor_actionable_records",
                ),
                "risk_sequence_records": actor_evidence.get(
                    "risk_sequence_records",
                ),
                "prefer_targets": actor_evidence.get("prefer_targets"),
                "learner_mask_dry_run_passed": actor_evidence.get(
                    "learner_mask_dry_run_passed",
                ),
            },
        },
    }


def _validate_shadow_contract(paths: LaunchPaths) -> dict[str, Any]:
    contract_path = _checked_repo_relative(
        str(SHADOW_CONTRACT_RELATIVE),
        root=paths.checkout_root,
        label="shadow-validation contract path",
    )
    if not contract_path.is_file():
        raise LaunchError(f"shadow-validation contract was not found: {contract_path}")
    contract_sha = _sha256_file(contract_path)
    if contract_sha != SHADOW_CONTRACT_SHA256:
        raise LaunchError(
            f"shadow-validation contract hash changed: expected={SHADOW_CONTRACT_SHA256} actual={contract_sha}"
        )
    contract = _load_json_object(
        contract_path,
        label="shadow-validation contract",
    )
    if set(contract) != {
        "name",
        "reports",
        "schema_version",
        "source_checkpoint_contract",
        "training_authority",
        "validated_code_sha256",
        "generation_source",
    }:
        raise LaunchError("shadow-validation contract fields changed")
    if contract.get("schema_version") != SHADOW_CONTRACT_SCHEMA:
        raise LaunchError("shadow-validation contract schema changed")
    if contract.get("name") != "v29-failure-credit-v4-bootstrap":
        raise LaunchError("shadow-validation contract name changed")
    if contract.get("source_checkpoint_contract") != FIXED_FROZEN_CONTRACT_NAME:
        raise LaunchError("shadow-validation contract names another frozen checkpoint")
    if contract.get("training_authority") is not False:
        raise LaunchError("shadow-validation contract must have no training authority")
    generation_source = _validate_git_source_mapping(
        contract.get("generation_source"),
        checkout_root=paths.checkout_root,
        label="shadow-validation evidence generation source",
    )

    code_hashes = contract.get("validated_code_sha256")
    if not isinstance(code_hashes, Mapping) or not code_hashes:
        raise LaunchError("shadow-validation contract has no validated code hashes")
    if not REQUIRED_SHADOW_CODE_PATHS.issubset(code_hashes):
        missing = sorted(REQUIRED_SHADOW_CODE_PATHS - set(code_hashes))
        raise LaunchError(
            "shadow-validation source authority omits required execution closure: " + ", ".join(missing),
        )
    validated_code: dict[str, str] = {}
    resolved_code_paths: set[Path] = set()
    for raw_relative, raw_digest in sorted(code_hashes.items()):
        if not isinstance(raw_relative, str) or not isinstance(raw_digest, str):
            raise LaunchError("shadow-validation code hash entry is malformed")
        candidate = _checked_repo_relative(
            raw_relative,
            root=paths.checkout_root,
            label="shadow-validated code path",
        )
        if candidate in resolved_code_paths:
            raise LaunchError("shadow-validation contract aliases one code file twice")
        resolved_code_paths.add(candidate)
        if not candidate.is_file():
            raise LaunchError(f"shadow-validated code file was not found: {candidate}")
        actual_digest = _sha256_file(candidate)
        if actual_digest != raw_digest:
            raise LaunchError(
                f"shadow-validated code hash changed: {raw_relative} expected={raw_digest} actual={actual_digest}"
            )
        validated_code[raw_relative] = actual_digest

    reports = contract.get("reports")
    if not isinstance(reports, Mapping) or set(reports) != {
        "semantics_historical",
        "evidence_live",
        "evidence_actor",
    }:
        raise LaunchError("shadow-validation report contract changed")
    result: dict[str, Any] = {}
    resolved_report_paths: set[Path] = set()
    for report_key in (
        "semantics_historical",
        "evidence_live",
        "evidence_actor",
    ):
        spec = reports.get(report_key)
        if not isinstance(spec, Mapping):
            raise LaunchError(f"shadow-validation {report_key} spec is malformed")
        report_path = _checked_repo_relative(
            spec.get("relative_path"),
            root=paths.artifact_root,
            label=f"shadow-validation {report_key} report path",
        )
        if report_path in resolved_report_paths:
            raise LaunchError("shadow-validation contract aliases one report twice")
        resolved_report_paths.add(report_path)
        if not report_path.is_file():
            raise LaunchError(f"shadow-validation report was not found: {report_path}")
        expected_digest = spec.get("sha256")
        if not isinstance(expected_digest, str):
            raise LaunchError(f"shadow-validation {report_key} SHA-256 is malformed")
        actual_digest = _sha256_file(report_path)
        if actual_digest != expected_digest:
            raise LaunchError(f"shadow-validation {report_key} report hash changed")
        report = _load_json_object(
            report_path,
            label=f"{report_key} shadow report",
        )
        _validate_formal_report_generation_source(
            report.get("generation_source"),
            expected_source=generation_source,
            validator_relative=FORMAL_REPORT_VALIDATORS[report_key],
            validated_code=validated_code,
            checkout_root=paths.checkout_root,
            label=f"{report_key} shadow report",
        )
        if report_key == "semantics_historical":
            summary = _validate_semantics_shadow(report, spec)
        elif report_key == "evidence_live":
            summary = _validate_evidence_shadow(
                report,
                spec,
                paths=paths,
            )
        else:
            summary = _validate_actor_evidence_shadow(report, spec)
        result[report_key] = {
            "path": str(report_path),
            "sha256": actual_digest,
            **summary,
        }
    proof: dict[str, Any] = {
        "contract": {
            "path": str(contract_path),
            "sha256": contract_sha,
            "schema_version": SHADOW_CONTRACT_SCHEMA,
            "name": contract["name"],
        },
        "validated_code_sha256": validated_code,
        "generation_source": generation_source,
        "reports": result,
    }
    authority_sha256 = _canonical_json_sha256(
        _shadow_authority_payload(proof),
    )
    if authority_sha256 != SHADOW_AUTHORITY_SHA256:
        raise LaunchError(
            "shadow-validation embedded authority digest changed: "
            f"expected={SHADOW_AUTHORITY_SHA256} actual={authority_sha256}",
        )
    proof["authority_sha256"] = authority_sha256
    return proof


def _validate_embedded_shadow_proof(value: object) -> dict[str, Any]:
    """Validate a launch-time shadow proof without reopening live evidence.

    The supervisor revalidates the live contract immediately before spawning
    the trainer.  After that point, lifecycle reconciliation must remain
    available even if the checkout or read-only evidence reports are moved or
    changed.  This validator therefore checks only the immutable proof already
    embedded in the launch manifest.
    """

    if not isinstance(value, Mapping):
        raise LaunchError("reviewed preflight has no embedded shadow proof")
    contract = value.get("contract")
    generation_source = value.get("generation_source")
    code_hashes = value.get("validated_code_sha256")
    reports = value.get("reports")
    if not isinstance(contract, Mapping):
        raise LaunchError("embedded shadow contract proof is malformed")
    generation_source = _validate_git_source_mapping(
        generation_source,
        checkout_root=None,
        label="embedded shadow evidence generation source",
    )
    expected_contract = {
        "sha256": SHADOW_CONTRACT_SHA256,
        "schema_version": SHADOW_CONTRACT_SCHEMA,
        "name": "v29-failure-credit-v4-bootstrap",
    }
    for key, expected in expected_contract.items():
        if contract.get(key) != expected:
            raise LaunchError(f"embedded shadow contract {key} changed")
    if (
        not isinstance(code_hashes, Mapping)
        or len(code_hashes) != SHADOW_VALIDATED_CODE_COUNT
        or any(
            not isinstance(path, str)
            or not path
            or Path(path).is_absolute()
            or ".." in Path(path).parts
            or not isinstance(digest, str)
            or re.fullmatch(r"[0-9a-f]{64}", digest) is None
            for path, digest in code_hashes.items()
        )
    ):
        raise LaunchError("embedded shadow validated-code proof is malformed")
    if not REQUIRED_SHADOW_CODE_PATHS.issubset(code_hashes):
        raise LaunchError("embedded shadow proof omits required execution closure")
    mapping_sha256 = _canonical_json_sha256(dict(code_hashes))
    if mapping_sha256 != SHADOW_VALIDATED_CODE_MAPPING_SHA256:
        raise LaunchError(
            "embedded shadow validated-code authority digest changed",
        )
    if not isinstance(reports, Mapping) or set(reports) != {
        "semantics_historical",
        "evidence_live",
        "evidence_actor",
    }:
        raise LaunchError("embedded shadow report proof is malformed")
    semantics = reports.get("semantics_historical")
    evidence = reports.get("evidence_live")
    actor_evidence = reports.get("evidence_actor")
    if (
        not isinstance(semantics, Mapping)
        or not isinstance(evidence, Mapping)
        or not isinstance(actor_evidence, Mapping)
    ):
        raise LaunchError("embedded shadow report proof is malformed")
    if (
        semantics.get("sha256") != SEMANTICS_SHADOW_SHA256
        or semantics.get("status") != "passed"
        or semantics.get("unknown_transitions") != 0
        or semantics.get("errors") != 0
        or not isinstance(semantics.get("parsed_records"), int)
        or isinstance(semantics.get("parsed_records"), bool)
        or semantics["parsed_records"] < 500_000
        or not isinstance(semantics.get("rich_decision_snapshots"), int)
        or isinstance(semantics.get("rich_decision_snapshots"), bool)
        or semantics["rich_decision_snapshots"] < 3_986
    ):
        raise LaunchError("embedded semantics shadow proof changed")
    maximum_completion_bytes = evidence.get(
        "maximum_episode_completion_storage_nbytes",
    )
    if (
        evidence.get("sha256") != EVIDENCE_SHADOW_SHA256
        or evidence.get("status") != "passed"
        or evidence.get("source_checkpoint_id") != FIXED_INITIALIZATION_CHECKPOINT_ID
        or evidence.get("semantic_censored_transitions") != 0
        or evidence.get("prefer_targets") != 0
        or evidence.get("training_state_unchanged") is not True
        or evidence.get("completion_staging_is_bounded") is not True
        or not isinstance(maximum_completion_bytes, int)
        or isinstance(maximum_completion_bytes, bool)
        or maximum_completion_bytes < 0
        or maximum_completion_bytes > 134_217_728
        or not isinstance(evidence.get("decisions"), int)
        or isinstance(evidence.get("decisions"), bool)
        or evidence["decisions"] < 1_000
    ):
        raise LaunchError("embedded failure-evidence shadow proof changed")
    if (
        actor_evidence.get("sha256") != ACTOR_EVIDENCE_SHADOW_SHA256
        or actor_evidence.get("status") != "passed"
        or actor_evidence.get("cases") != 6
        or actor_evidence.get("actor_actionable_records") != 4
        or actor_evidence.get("risk_sequence_records") != 5
        or actor_evidence.get("prefer_targets") != 0
        or actor_evidence.get("learner_mask_dry_run_passed") is not True
    ):
        raise LaunchError(
            "embedded failure actor-evidence shadow proof changed",
        )
    authority_sha256 = _canonical_json_sha256(
        _shadow_authority_payload(value),
    )
    if value.get("authority_sha256") != authority_sha256 or authority_sha256 != SHADOW_AUTHORITY_SHA256:
        raise LaunchError("embedded shadow authority digest changed")
    return dict(value)


class _MigrationProofBackend:
    """Non-I/O backend used only to compose fresh CPU training resources."""

    def __init__(self) -> None:
        from sts2_rl.contracts import BackendCapabilities

        self._capabilities = BackendCapabilities(
            backend_name="v29-model-init-preflight",
            session_id="v29-model-init-preflight",
        )

    @property
    def capabilities(self) -> Any:
        return self._capabilities

    @property
    def session_id(self) -> str:
        return self._capabilities.session_id

    @property
    def is_connected(self) -> bool:
        return False

    def health(self) -> dict[str, Any]:
        return {"ok": False, "preflight_only": True}

    def get_spec(self) -> dict[str, Any]:
        return {}

    def get_state(self) -> dict[str, Any]:
        raise RuntimeError("model-init preflight backend cannot read state")

    def reset(self, _request: Any) -> Any:
        raise RuntimeError("model-init preflight backend cannot reset")

    def step(self, _request: Any) -> Any:
        raise RuntimeError("model-init preflight backend cannot step")

    def combat_reset(self, _request: Any) -> Any:
        raise RuntimeError("model-init preflight backend cannot reset combat")

    def close(self) -> None:
        return None


def _numpy_rng_equal(left: object, right: object) -> bool:
    import numpy as np

    if not isinstance(left, tuple) or not isinstance(right, tuple):
        return False
    if len(left) != len(right):
        return False
    for left_item, right_item in zip(left, right, strict=True):
        if isinstance(left_item, np.ndarray):
            if not isinstance(right_item, np.ndarray) or not np.array_equal(left_item, right_item):
                return False
        elif left_item != right_item:
            return False
    return True


def _preflight_model_initialization_migration(
    checkpoint: Path,
    *,
    config: Any,
) -> dict[str, Any]:
    """Execute the model-only migration on fresh CPU resources and prove resets."""

    import pickle
    import random

    import numpy as np
    import torch

    from sts2_rl.training.checkpointing import (
        _LIVENESS_HEAD_PREFIXES,
        _TRANSACTION_HEAD_PREFIXES,
        TrainingState,
        initialize_model_from_checkpoint,
        preflight_model_initialization,
    )
    from sts2_rl.training.factory import build_training_resources

    validated = preflight_model_initialization(checkpoint, config=config)
    source_state = torch.load(
        validated.root / "network.pt",
        map_location="cpu",
        weights_only=True,
    )
    if not isinstance(source_state, dict):
        raise LaunchError("frozen checkpoint network payload must be a tensor mapping")
    source_transaction = {
        key for key in source_state if isinstance(key, str) and key.startswith(_TRANSACTION_HEAD_PREFIXES)
    }
    source_transaction_prefixes = {
        prefix for prefix in _TRANSACTION_HEAD_PREFIXES if any(key.startswith(prefix) for key in source_transaction)
    }
    if source_transaction_prefixes != set(_TRANSACTION_HEAD_PREFIXES):
        raise LaunchError("frozen v28 source does not contain the complete transaction-v3 head family")
    if len(source_transaction) != 18:
        raise LaunchError("frozen v28 source transaction-v3 tensor count changed")
    source_liveness = {key for key in source_state if isinstance(key, str) and key.startswith(_LIVENESS_HEAD_PREFIXES)}
    if source_liveness:
        raise LaunchError("frozen v28 source unexpectedly already contains liveness heads")

    cpu_config = replace(
        config,
        runtime=replace(
            config.runtime,
            device="cpu",
            collector_device="cpu",
        ),
    )
    resources = build_training_resources(
        cpu_config,
        backend=_MigrationProofBackend(),
    )
    try:
        target_before = {key: value.detach().cpu().clone() for key, value in resources.model.state_dict().items()}
        target_transaction = {key for key in target_before if key.startswith(_TRANSACTION_HEAD_PREFIXES)}
        if target_transaction:
            raise LaunchError("v29 target still contains retired transaction-v3 heads")
        target_liveness = {key for key in target_before if key.startswith(_LIVENESS_HEAD_PREFIXES)}
        target_liveness_prefixes = {
            prefix for prefix in _LIVENESS_HEAD_PREFIXES if any(key.startswith(prefix) for key in target_liveness)
        }
        if not target_liveness or target_liveness_prefixes != set(_LIVENESS_HEAD_PREFIXES):
            raise LaunchError("v29 target lacks the complete freshly initialized liveness head family")
        if len(target_liveness) != 12:
            raise LaunchError("v29 target liveness tensor count changed")
        if resources.transaction_replay is not None:
            raise LaunchError("transaction-v3 replay must be absent from the v29 lineage")
        if resources.failure_credit_replay is None:
            raise LaunchError("failure-credit-v4 learning requires a replay-v4 instance")
        if resources.episodic_replay is None:
            raise LaunchError("v29 requires the maintained episodic replay")
        if resources.optimizer.state:
            raise LaunchError("fresh v29 optimizer unexpectedly has moments")
        if resources.rollout_queue.snapshot():
            raise LaunchError("fresh v29 rollout queue is not empty")

        optimizer_before = deepcopy(resources.optimizer.state_dict())
        queue_before = resources.rollout_queue.snapshot()
        failure_before = resources.failure_credit_replay.state_dict()
        episodic_before = resources.episodic_replay.state_dict()
        collector_before = resources.collector.state_dict()
        python_rng_before = random.getstate()
        numpy_rng_before = np.random.get_state()
        torch_rng_before = torch.get_rng_state().clone()

        initialized_root = initialize_model_from_checkpoint(
            checkpoint,
            config=cpu_config,
            resources=resources,
        )
        if initialized_root.resolve(strict=False) != checkpoint:
            raise LaunchError("model initialization returned another checkpoint root")

        target_after = resources.model.state_dict()
        if any(key.startswith(_TRANSACTION_HEAD_PREFIXES) for key in target_after):
            raise LaunchError("transaction-v3 heads survived model initialization")
        for key in target_liveness:
            if not torch.equal(
                target_after[key].detach().cpu(),
                target_before[key],
            ):
                raise LaunchError(f"new liveness head was not freshly preserved: {key}")
        inherited = 0
        for key, source_value in source_state.items():
            if key in source_transaction:
                continue
            if key not in target_after or not isinstance(
                source_value,
                torch.Tensor,
            ):
                raise LaunchError(f"shared source tensor was not inherited: {key}")
            if not torch.equal(
                target_after[key].detach().cpu(),
                source_value.detach().cpu(),
            ):
                raise LaunchError(f"shared source tensor changed during migration: {key}")
            inherited += 1

        if resources.optimizer.state_dict() != optimizer_before:
            raise LaunchError("model initialization mutated the fresh optimizer")
        if resources.rollout_queue.snapshot() != queue_before:
            raise LaunchError("model initialization mutated the fresh rollout queue")
        if pickle.dumps(
            resources.failure_credit_replay.state_dict(),
            protocol=5,
        ) != pickle.dumps(failure_before, protocol=5):
            raise LaunchError("model initialization mutated failure replay-v4")
        if pickle.dumps(
            resources.episodic_replay.state_dict(),
            protocol=5,
        ) != pickle.dumps(episodic_before, protocol=5):
            raise LaunchError("model initialization mutated episodic replay")
        if resources.collector.state_dict() != collector_before:
            raise LaunchError("model initialization mutated collector continuation state")
        if random.getstate() != python_rng_before:
            raise LaunchError("model initialization mutated Python RNG state")
        if not _numpy_rng_equal(np.random.get_state(), numpy_rng_before):
            raise LaunchError("model initialization mutated NumPy RNG state")
        if not torch.equal(torch.get_rng_state(), torch_rng_before):
            raise LaunchError("model initialization mutated Torch RNG state")
        failure_metrics = resources.failure_credit_replay.metrics()
        episodic_metrics = resources.episodic_replay.metrics()
        for label, metrics in (
            ("failure-credit", failure_metrics),
            ("episodic", episodic_metrics),
        ):
            for key in ("size", "put_count"):
                if metrics.get(key) != 0:
                    raise LaunchError(f"fresh {label} replay {key} is non-zero")
        fresh_state = asdict(TrainingState())
        if any(value != 0 for value in fresh_state.values()):
            raise LaunchError("fresh v29 TrainingState counters are not zero")
        return {
            "mode": "model_initialization",
            "source_transaction_head_tensors_dropped": len(source_transaction),
            "target_transaction_head_tensors": 0,
            "fresh_liveness_head_tensors": len(target_liveness),
            "shared_network_tensors_inherited": inherited,
            "optimizer_state_entries": 0,
            "rollout_queue_items": 0,
            "transaction_replay": "absent",
            "failure_credit_replay": {
                "version": failure_metrics["version"],
                "size": failure_metrics["size"],
                "put_count": failure_metrics["put_count"],
            },
            "episodic_replay": {
                "version": episodic_metrics["version"],
                "size": episodic_metrics["size"],
                "put_count": episodic_metrics["put_count"],
            },
            "rng_preserved_from_fresh_lineage": True,
            "new_training_state": fresh_state,
        }
    finally:
        resources.close()


def _hermetic_runtime_probe_command(
    paths: PreflightPaths,
) -> tuple[str, ...]:
    """Return the only command allowed to produce v29 runtime proof."""

    launcher = (paths.package_root / "scripts/launch_v29_failure_credit_v4.py").resolve(strict=False)
    return (
        os.fspath(paths.venv_python),
        os.fspath(launcher),
        HERMETIC_RUNTIME_PROBE_ACTION,
    )


def _runtime_probe_result_mapping(
    runtime: Mapping[str, Any],
) -> dict[str, Any]:
    return {
        "runtime_identity": runtime.get("runtime_identity"),
        "simulator": runtime.get("simulator"),
        "runtime_mechanics": runtime.get("runtime_mechanics"),
        "training_revival": runtime.get("training_revival"),
    }


def _validate_hermetic_runtime_probe_payload(
    payload: Mapping[str, Any],
    *,
    paths: PreflightPaths,
    trainer_environment_sha256: str,
) -> dict[str, Any]:
    expected_fields = {
        "schema_version",
        "probe_command",
        "timeout_seconds",
        "trainer_environment_sha256",
        "result_sha256",
        "torch_version",
        "device_name",
        "runtime_identity",
        "simulator",
        "runtime_mechanics",
        "training_revival",
    }
    if set(payload) != expected_fields:
        raise LaunchError(
            "hermetic runtime probe fields changed: "
            f"missing={sorted(expected_fields - set(payload))} "
            f"extra={sorted(set(payload) - expected_fields)}"
        )
    if payload.get("schema_version") != HERMETIC_RUNTIME_PROBE_SCHEMA:
        raise LaunchError("hermetic runtime probe schema changed")
    command = payload.get("probe_command")
    if not isinstance(command, list) or command != list(_hermetic_runtime_probe_command(paths)):
        raise LaunchError("hermetic runtime probe command changed")
    if payload.get("timeout_seconds") != HERMETIC_RUNTIME_PROBE_TIMEOUT_SECONDS:
        raise LaunchError("hermetic runtime probe timeout changed")
    if payload.get("trainer_environment_sha256") != trainer_environment_sha256:
        raise LaunchError("hermetic runtime probe environment binding changed")
    runtime_identity = payload.get("runtime_identity")
    simulator = payload.get("simulator")
    mechanics = payload.get("runtime_mechanics")
    revival = payload.get("training_revival")
    if not isinstance(runtime_identity, Mapping):
        raise LaunchError("hermetic runtime probe has no ROCm runtime identity")
    accelerator = runtime_identity.get("accelerator")
    if (
        not isinstance(accelerator, Mapping)
        or payload.get("torch_version") != runtime_identity.get("torch_version")
        or payload.get("device_name") != accelerator.get("name")
    ):
        raise LaunchError("hermetic runtime probe ROCm summary changed")
    if not isinstance(simulator, Mapping):
        raise LaunchError("hermetic runtime probe has no verified HeadlessSim identity")
    if (
        not isinstance(mechanics, Mapping)
        or mechanics.get("schema") != "sts2-runtime-mechanics-audit-v1"
        or mechanics.get("runtime_event_checked") is not True
        or mechanics.get("runtime_combat_checked") is not True
    ):
        raise LaunchError("hermetic runtime probe mechanics gate is incomplete")
    if not isinstance(revival, Mapping):
        raise LaunchError("hermetic runtime probe has no revival identity")
    result_sha256 = payload.get("result_sha256")
    if (
        not isinstance(result_sha256, str)
        or re.fullmatch(r"[0-9a-f]{64}", result_sha256) is None
        or result_sha256 != _canonical_json_sha256(_runtime_probe_result_mapping(payload))
    ):
        raise LaunchError("hermetic runtime probe result digest changed")
    return dict(payload)


def _execute_hermetic_runtime_probe(paths: PreflightPaths) -> dict[str, Any]:
    """Execute the fixed runtime gate inside an already-hermetic process."""

    from sts2_rl.runtime_mechanics import (
        run_runtime_mechanics_preflight,
    )
    from sts2_rl.simulator_identity import verify_headless_simulator
    from sts2_rl.training.config import engine_revival_identity
    from sts2_rl.training.launch_contract import current_rocm_runtime_identity

    environment = dict(os.environ)
    try:
        v29_preflight.validate_trainer_environment(
            environment,
            paths=paths,
        )
        environment_sha256 = _exact_trainer_environment_sha256(environment)
        runtime_identity = current_rocm_runtime_identity()
        simulator = verify_headless_simulator(
            paths.simulator_executable,
            identity_path=paths.simulator_identity,
        )
        runtime_mechanics = run_runtime_mechanics_preflight(
            simulator.executable,
        )
    except LaunchError:
        raise
    except Exception as exc:
        raise LaunchError(f"fixed hermetic runtime probe failed: {type(exc).__name__}: {exc}") from exc
    runtime: dict[str, Any] = {
        "schema_version": HERMETIC_RUNTIME_PROBE_SCHEMA,
        "probe_command": list(_hermetic_runtime_probe_command(paths)),
        "timeout_seconds": HERMETIC_RUNTIME_PROBE_TIMEOUT_SECONDS,
        "trainer_environment_sha256": environment_sha256,
        "torch_version": runtime_identity["torch_version"],
        "device_name": runtime_identity["accelerator"]["name"],
        "runtime_identity": runtime_identity,
        "simulator": simulator.to_mapping(),
        "runtime_mechanics": runtime_mechanics,
        "training_revival": engine_revival_identity(),
    }
    runtime["result_sha256"] = _canonical_json_sha256(_runtime_probe_result_mapping(runtime))
    return _validate_hermetic_runtime_probe_payload(
        runtime,
        paths=paths,
        trainer_environment_sha256=environment_sha256,
    )


def _verify_runtime_inputs(
    paths: PreflightPaths,
    *,
    environment: dict[str, str],
) -> dict[str, Any]:
    """Run the fixed production probe in a bounded hermetic subprocess."""

    v29_preflight.validate_trainer_environment(
        environment,
        paths=paths,
    )
    environment_sha256 = _exact_trainer_environment_sha256(environment)
    command = _hermetic_runtime_probe_command(paths)
    returncode, stdout_raw, stderr_raw = _run_bounded_runtime_probe_process(
        command,
        cwd=paths.package_root,
        environment=environment,
        timeout_seconds=HERMETIC_RUNTIME_PROBE_TIMEOUT_SECONDS,
        maximum_output_bytes=HERMETIC_RUNTIME_PROBE_MAX_OUTPUT_BYTES,
    )
    if returncode != 0:
        diagnostic = (
            (stderr_raw or stdout_raw)
            .decode(
                "utf-8",
                errors="replace",
            )
            .strip()
        )
        raise LaunchError(f"fixed hermetic runtime probe failed ({returncode}): {diagnostic[-4000:]}")
    try:
        payload = json.loads(stdout_raw.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise LaunchError("fixed hermetic runtime probe returned malformed JSON") from exc
    if not isinstance(payload, Mapping):
        raise LaunchError("fixed hermetic runtime probe result is not an object")
    return _validate_hermetic_runtime_probe_payload(
        payload,
        paths=paths,
        trainer_environment_sha256=environment_sha256,
    )


def _runtime_probe_live_group_members(
    process_group_id: int,
) -> tuple[int, ...]:
    """Return non-zombie Linux processes that still belong to ``pgrp``.

    The probe leader is deliberately left unreaped while this function is
    used, so its PID (and therefore the newly-created process-group ID) cannot
    be reused between inspection and ``killpg(2)``.  ``killpg(..., 0)`` is not
    sufficient here because a zombie-only group still exists in the kernel
    even though it has no process that can execute or retain resources.
    """

    if os.name != "posix" or not Path("/proc").is_dir():
        return ()
    members: list[int] = []
    try:
        entries = tuple(Path("/proc").iterdir())
    except OSError as exc:
        raise LaunchError("cannot enumerate fixed runtime probe process group") from exc
    for entry in entries:
        if not entry.name.isdecimal():
            continue
        try:
            stat = (entry / "stat").read_text(encoding="utf-8")
        except (FileNotFoundError, ProcessLookupError):
            # The process disappeared between /proc enumeration and read.
            continue
        except OSError as exc:
            raise LaunchError(
                "cannot inspect fixed runtime probe process group member",
            ) from exc
        # /proc/<pid>/stat field 2 is parenthesised and may itself contain
        # spaces or ')'.  Everything after the final ')' starts with fields
        # 3=state, 4=ppid and 5=pgrp.
        _separator, found, suffix = stat.rpartition(")")
        fields = suffix.strip().split() if found else []
        if len(fields) < 3:
            raise LaunchError("malformed /proc stat while inspecting runtime probe group")
        state = fields[0]
        try:
            member_group_id = int(fields[2])
        except ValueError as exc:
            raise LaunchError("malformed process group in runtime probe /proc stat") from exc
        if member_group_id == process_group_id and state != "Z":
            members.append(int(entry.name))
    return tuple(sorted(members))


def _runtime_probe_group_exists(process_group_id: int) -> bool:
    """Compatibility predicate meaning the group has a *live* member."""

    return bool(_runtime_probe_live_group_members(process_group_id))


def _signal_runtime_probe_group(
    process_group_id: int,
    signal_number: int,
) -> None:
    """Signal an unreaped probe group, accepting a concurrent clean exit."""

    try:
        os.killpg(process_group_id, signal_number)
    except ProcessLookupError:
        pass
    except PermissionError as exc:
        raise LaunchError("cannot signal fixed runtime probe process group") from exc


def _wait_for_no_live_runtime_probe_group_members(
    process_group_id: int,
    *,
    timeout_seconds: float,
) -> tuple[int, ...]:
    deadline = time.monotonic() + timeout_seconds
    while True:
        members = _runtime_probe_live_group_members(process_group_id)
        if not members or time.monotonic() >= deadline:
            return members
        time.sleep(0.01)


def _terminate_and_reap_runtime_probe_group(
    process: subprocess.Popen[bytes],
    *,
    process_group_id: int | None,
) -> int:
    """Terminate all live members, then reap the still-unique group leader.

    No ``wait``/``poll`` is permitted before all group-directed signals have
    finished.  Keeping the leader unreaped is what makes a raw numeric PGID
    safe from reuse.  Once the final ``process.wait`` begins this function
    never calls ``killpg`` again.
    """

    cleanup_error: BaseException | None = None
    if process_group_id is not None and os.name == "posix":
        try:
            members = _runtime_probe_live_group_members(process_group_id)
            if members:
                _signal_runtime_probe_group(process_group_id, signal.SIGTERM)
                members = _wait_for_no_live_runtime_probe_group_members(
                    process_group_id,
                    timeout_seconds=2.0,
                )
            if members:
                _signal_runtime_probe_group(process_group_id, signal.SIGKILL)
                members = _wait_for_no_live_runtime_probe_group_members(
                    process_group_id,
                    timeout_seconds=2.0,
                )
            if members:
                cleanup_error = LaunchError(
                    f"fixed runtime probe process group retained live members after SIGKILL: {list(members)}",
                )
        except BaseException as exc:
            # Enumeration itself is a proof obligation.  If it fails, make a
            # best-effort TERM+KILL while the unreaped leader still protects
            # the PGID from reuse, then report the original inspection error.
            cleanup_error = exc
            for signal_number in (signal.SIGTERM, signal.SIGKILL):
                try:
                    _signal_runtime_probe_group(
                        process_group_id,
                        signal_number,
                    )
                except BaseException:
                    pass
    else:  # pragma: no cover - production execution is Linux/WSL only.
        try:
            process.terminate()
            time.sleep(0.05)
            process.kill()
        except (ProcessLookupError, OSError):
            pass

    # This is deliberately the first and only operation that may reap the
    # leader.  There must be no PGID-directed operation after this point.
    try:
        returncode = process.wait(timeout=2.0)
    except subprocess.TimeoutExpired as exc:
        raise LaunchError("fixed runtime probe leader could not be reaped") from exc
    if cleanup_error is not None:
        if isinstance(cleanup_error, LaunchError):
            raise cleanup_error
        raise LaunchError("fixed runtime probe group cleanup failed") from cleanup_error
    return returncode


def _close_runtime_probe_resources(
    selector: selectors.BaseSelector | None,
    streams: Sequence[Any],
) -> BaseException | None:
    """Best-effort close every local resource and return the first failure."""

    first_error: BaseException | None = None
    if selector is not None:
        try:
            selector.close()
        except BaseException as exc:
            first_error = exc
    for stream in streams:
        try:
            stream.close()
        except BaseException as exc:
            if first_error is None:
                first_error = exc
    return first_error


def _run_bounded_runtime_probe_process(
    command: tuple[str, ...],
    *,
    cwd: Path,
    environment: dict[str, str],
    timeout_seconds: float,
    maximum_output_bytes: int,
) -> tuple[int, bytes, bytes]:
    """Drain both probe streams with hard bounds and own its process group."""

    if os.name != "posix":
        raise LaunchError("fixed hermetic runtime probe execution requires Linux/WSL")
    process_group_id: int | None = None
    stdout = bytearray()
    stderr = bytearray()
    streams: dict[int, tuple[Any, bytearray, str]] = {}
    all_streams: tuple[Any, ...] = ()
    selector: selectors.BaseSelector | None = None
    returncode: int | None = None
    # All potentially-failing Python-side initialization has happened before
    # spawning.  Consequently the first statement after successful Popen is
    # the cleanup guard itself.
    process = subprocess.Popen(
        command,
        cwd=cwd,
        env=environment,
        stdin=subprocess.DEVNULL,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        start_new_session=True,
        bufsize=0,
    )
    try:
        process_group_id = process.pid
        all_streams = tuple(stream for stream in (process.stdout, process.stderr) if stream is not None)
        if process.stdout is None or process.stderr is None:
            raise LaunchError("fixed runtime probe pipes were not created")
        selector = selectors.DefaultSelector()
        for stream, destination, label in (
            (process.stdout, stdout, "stdout"),
            (process.stderr, stderr, "stderr"),
        ):
            descriptor = stream.fileno()
            streams[descriptor] = (stream, destination, label)
            selector.register(descriptor, selectors.EVENT_READ)
        deadline = time.monotonic() + timeout_seconds
        while streams:
            remaining = deadline - time.monotonic()
            if remaining <= 0.0:
                raise LaunchError(f"fixed hermetic runtime probe exceeded {timeout_seconds:.0f}s")
            events = selector.select(min(remaining, 0.1))
            if not events:
                continue
            for key, _mask in events:
                descriptor = int(key.fd)
                stream, destination, label = streams[descriptor]
                try:
                    chunk = os.read(descriptor, 64 * 1024)
                except InterruptedError:
                    continue
                if not chunk:
                    selector.unregister(descriptor)
                    del streams[descriptor]
                    continue
                if len(destination) + len(chunk) > maximum_output_bytes:
                    raise LaunchError(f"fixed hermetic runtime probe {label} exceeded {maximum_output_bytes} bytes")
                destination.extend(chunk)
        remaining = deadline - time.monotonic()
        if remaining <= 0.0:
            raise LaunchError(f"fixed hermetic runtime probe exceeded {timeout_seconds:.0f}s")
    finally:
        had_primary_error = sys.exc_info()[0] is not None
        resource_error: BaseException | None = None
        try:
            resource_error = _close_runtime_probe_resources(
                selector,
                all_streams,
            )
        finally:
            returncode = _terminate_and_reap_runtime_probe_group(
                process,
                process_group_id=process_group_id,
            )
        if resource_error is not None and not had_primary_error:
            raise LaunchError("fixed runtime probe resource cleanup failed") from resource_error
    if returncode is None:  # pragma: no cover - cleanup always assigns or raises.
        raise LaunchError("fixed runtime probe did not reap its leader")
    return returncode, bytes(stdout), bytes(stderr)


def _validate_runtime_readiness_evidence(
    paths: LaunchPaths,
    *,
    runtime: Mapping[str, Any] | None,
    shadow: Mapping[str, Any],
) -> dict[str, Any]:
    """Authenticate independent ROCm/OOM readiness without granting training authority."""

    from sts2_rl.training.launch_contract import (
        SupervisedLaunchContractError,
        load_runtime_readiness_evidence,
    )

    report_path = (paths.artifact_root / RUNTIME_READINESS_REPORT_RELATIVE).resolve(strict=False)
    try:
        evidence = load_runtime_readiness_evidence(
            report_path,
            expected_sha256=RUNTIME_READINESS_REPORT_SHA256,
            artifact_root=paths.artifact_root,
        )
    except SupervisedLaunchContractError as exc:
        raise LaunchError(f"runtime readiness evidence failed: {exc}") from exc
    if evidence.get("schema_version") != RUNTIME_READINESS_EVIDENCE_SCHEMA:
        raise LaunchError("runtime readiness evidence schema changed")
    binding = evidence.get("report_binding")
    if not isinstance(binding, Mapping):
        raise LaunchError("runtime readiness evidence has no report binding")
    if binding.get("version") != RUNTIME_READINESS_REPORT_SCHEMA:
        raise LaunchError("runtime readiness report schema changed")
    generation_source = shadow.get("generation_source")
    validated_code = shadow.get("validated_code_sha256")
    if not isinstance(generation_source, Mapping) or not isinstance(validated_code, Mapping):
        raise LaunchError("shadow source authority is malformed")
    _validate_formal_report_generation_source(
        binding.get("generation_source"),
        expected_source=generation_source,
        validator_relative=FORMAL_REPORT_VALIDATORS["runtime_readiness"],
        validated_code=cast(Mapping[str, str], validated_code),
        checkout_root=paths.checkout_root,
        label="runtime readiness report",
    )
    config = binding.get("config")
    initialization = binding.get("initialization")
    if not isinstance(config, Mapping) or not isinstance(initialization, Mapping):
        raise LaunchError("runtime readiness source/config binding is malformed")
    if (
        config.get("fingerprint_sha256") != FIXED_CONFIG_FINGERPRINT_SHA256
        or config.get("source") != str(paths.preflight.config_path)
        or config.get("maximum_context_steps") != 256
        or config.get("maximum_candidates") != 256
        or config.get("tbptt_window_steps") != 16
    ):
        raise LaunchError("runtime readiness config differs from reviewed v29 config")
    if (
        initialization.get("contract_name") != FIXED_FROZEN_CONTRACT_NAME
        or initialization.get("checkpoint_path") != str(paths.initialization_checkpoint)
        or initialization.get("checkpoint_id") != FIXED_INITIALIZATION_CHECKPOINT_ID
        or initialization.get("manifest_sha256") != FIXED_INITIALIZATION_MANIFEST_SHA256
        or initialization.get("metadata_sha256") != FIXED_INITIALIZATION_METADATA_SHA256
    ):
        raise LaunchError("runtime readiness source differs from V28_100K_FROZEN")
    if runtime is not None:
        live_identity = runtime.get("runtime_identity")
        if not isinstance(live_identity, Mapping) or binding.get("runtime_identity") != live_identity:
            raise LaunchError("live ROCm runtime differs from readiness report")
    return evidence


def _run_v29_preflight(
    raw_paths: PreflightPaths,
    *,
    initialize_from: str | Path,
    enforce_active_root: bool,
    verify_runtime: bool,
) -> dict[str, Any]:
    try:
        paths = v29_preflight.validate_layout(
            raw_paths,
            enforce_active_root=enforce_active_root,
        )
        selected = Path(initialize_from).expanduser()
        if not selected.is_absolute():
            raise LaunchError("--initialize-from must be an absolute path")
        selected = selected.resolve(strict=False)
        frozen_contract = _v28_frozen_contract()
        expected = frozen_contract.resolve(paths.artifact_root)
        if selected != expected:
            raise LaunchError("v29 initialization must use only V28_100K_FROZEN")
        if not paths.venv_python.is_file():
            raise LaunchError(f"ROCm venv Python was not found: {paths.venv_python}")

        config = _load_v29_config(paths)
        abi = _abi_contract()
        from sts2_rl.checkpoints.frozen import validate_frozen_checkpoint

        frozen = validate_frozen_checkpoint(
            paths.artifact_root,
            contract=frozen_contract,
        )
        if frozen.root.resolve(strict=False) != selected:
            raise LaunchError("frozen checkpoint validator returned another root")
        launch_paths = LaunchPaths(
            preflight=paths,
            launcher_dir=paths.artifact_root / "launcher",
            manifest_dir=paths.artifact_root / "launchers",
            initialization_checkpoint=selected,
        )
        shadow = _validate_shadow_contract(launch_paths)
        migration = _preflight_model_initialization_migration(
            selected,
            config=config,
        )
        effective = _effective_v29_config(config, paths=paths)
        effective_fingerprint = effective.fingerprint_sha256()
        if effective_fingerprint != FIXED_RUNTIME_CONFIG_FINGERPRINT_SHA256:
            raise LaunchError(
                "v29 effective runtime config fingerprint changed: "
                f"expected={FIXED_RUNTIME_CONFIG_FINGERPRINT_SHA256} "
                f"actual={effective_fingerprint}"
            )
        environment = v29_preflight.validate_trainer_environment(
            v29_preflight.build_trainer_environment(paths),
            paths=paths,
        )
        runtime = (
            _verify_runtime_inputs(
                paths,
                environment=environment,
            )
            if verify_runtime
            else None
        )
        runtime_readiness = _validate_runtime_readiness_evidence(
            launch_paths,
            runtime=runtime,
            shadow=shadow,
        )
        command = v29_preflight.validate_trainer_command(
            v29_preflight.build_trainer_command(
                paths,
                initialize_from=selected,
            ),
            paths=paths,
            initialize_from=selected,
        )
        source_state = frozen.metadata.get("training_state")
        return {
            "status": "preflight-passed",
            "run_name": RUN_NAME,
            "initialization": {
                "mode": "model_initialization",
                "checkpoint": str(selected),
                "checkpoint_contract": FIXED_FROZEN_CONTRACT_NAME,
                "checkpoint_id": FIXED_INITIALIZATION_CHECKPOINT_ID,
                "source_training_state": source_state,
                "network_parameters_inherited": True,
                "old_transaction_heads_removed": True,
                "new_liveness_heads_fresh": True,
                "optimizer_rollouts_rng_and_counters_reset": True,
                "migration_proof": migration,
            },
            "abi_contract": abi,
            "shadow_validation": shadow,
            "runtime_readiness_evidence": runtime_readiness,
            "config_fingerprint_sha256": config.fingerprint_sha256(),
            "effective_config_fingerprint_sha256": effective_fingerprint,
            "trainer_command": list(command),
            "trainer_environment": (v29_preflight.trainer_environment_contract(environment)),
            "trainer_environment_sha256": (_exact_trainer_environment_sha256(environment)),
            "runtime": runtime,
            "preflight_training_started": False,
            "training_started": False,
        }
    except PreflightError:
        raise
    except (LaunchError, OSError, RuntimeError, TypeError, ValueError) as exc:
        raise PreflightError(str(exc)) from exc


def _validate_preflight_payload(
    payload: Mapping[str, Any],
    *,
    paths: LaunchPaths,
    verify_live_shadow: bool = True,
) -> dict[str, Any]:
    if payload.get("preflight_status", payload.get("status")) != "preflight-passed":
        raise LaunchError("reviewed preflight did not pass")
    if payload.get("preflight_training_started") is not False:
        raise LaunchError("reviewed preflight must prove a non-launching preflight")
    if "training_started" in payload and not isinstance(
        payload.get("training_started"),
        bool,
    ):
        raise LaunchError("mutable training_started lifecycle flag is malformed")
    if payload.get("run_name") != RUN_NAME:
        raise LaunchError("reviewed preflight names another training lineage")
    if payload.get("config_fingerprint_sha256") != FIXED_CONFIG_FINGERPRINT_SHA256:
        raise LaunchError("reviewed preflight config fingerprint changed")
    if payload.get("effective_config_fingerprint_sha256") != FIXED_RUNTIME_CONFIG_FINGERPRINT_SHA256:
        raise LaunchError("reviewed preflight effective config fingerprint changed")
    if payload.get("abi_contract") != _expected_abi_contract():
        raise LaunchError("reviewed preflight ABI contract changed")
    embedded_shadow = _validate_embedded_shadow_proof(
        payload.get("shadow_validation"),
    )
    if verify_live_shadow and embedded_shadow != _validate_shadow_contract(paths):
        raise LaunchError("reviewed preflight shadow validation changed")
    embedded_readiness = payload.get("runtime_readiness_evidence")
    if not isinstance(embedded_readiness, Mapping) or set(embedded_readiness) != {
        "schema_version",
        "artifact_root",
        "report_path",
        "report_sha256",
        "report_binding",
    }:
        raise LaunchError("reviewed preflight runtime readiness evidence is malformed")
    expected_readiness_path = (paths.artifact_root / RUNTIME_READINESS_REPORT_RELATIVE).resolve(strict=False)
    if (
        embedded_readiness.get("schema_version") != RUNTIME_READINESS_EVIDENCE_SCHEMA
        or embedded_readiness.get("artifact_root") != str(paths.artifact_root)
        or embedded_readiness.get("report_path") != str(expected_readiness_path)
        or embedded_readiness.get("report_sha256") != RUNTIME_READINESS_REPORT_SHA256
    ):
        raise LaunchError("reviewed preflight runtime readiness identity changed")
    initialization = payload.get("initialization")
    if not isinstance(initialization, Mapping):
        raise LaunchError("reviewed preflight has no initialization proof")
    expected_initialization = {
        "mode": "model_initialization",
        "checkpoint": str(paths.initialization_checkpoint),
        "checkpoint_contract": FIXED_FROZEN_CONTRACT_NAME,
        "checkpoint_id": FIXED_INITIALIZATION_CHECKPOINT_ID,
        "network_parameters_inherited": True,
        "old_transaction_heads_removed": True,
        "new_liveness_heads_fresh": True,
        "optimizer_rollouts_rng_and_counters_reset": True,
    }
    for key, expected in expected_initialization.items():
        if initialization.get(key) != expected:
            raise LaunchError(f"reviewed preflight initialization {key} changed")
    source_state = initialization.get("source_training_state")
    if not isinstance(source_state, Mapping):
        raise LaunchError("reviewed preflight has no source training state")
    if source_state.get("environment_steps") != FIXED_INITIALIZATION_STEP:
        raise LaunchError("reviewed preflight source environment step changed")
    if source_state.get("policy_version") != FIXED_INITIALIZATION_POLICY_VERSION:
        raise LaunchError("reviewed preflight source policy version changed")
    migration = initialization.get("migration_proof")
    if not isinstance(migration, Mapping):
        raise LaunchError("reviewed preflight has no migration proof")
    expected_migration = {
        "mode": "model_initialization",
        "source_transaction_head_tensors_dropped": 18,
        "target_transaction_head_tensors": 0,
        "fresh_liveness_head_tensors": 12,
        "shared_network_tensors_inherited": 214,
        "optimizer_state_entries": 0,
        "rollout_queue_items": 0,
        "transaction_replay": "absent",
        "rng_preserved_from_fresh_lineage": True,
    }
    for key, expected in expected_migration.items():
        if migration.get(key) != expected:
            raise LaunchError(f"reviewed preflight migration proof {key} changed")
    new_state = migration.get("new_training_state")
    if not isinstance(new_state, Mapping) or not new_state or any(value != 0 for value in new_state.values()):
        raise LaunchError("reviewed preflight new-lineage counters are not all zero")
    for replay_key in ("failure_credit_replay", "episodic_replay"):
        replay = migration.get(replay_key)
        if not isinstance(replay, Mapping) or replay.get("size") != 0 or replay.get("put_count") != 0:
            raise LaunchError(f"reviewed preflight {replay_key} is not fresh/empty")
    failure_replay = migration["failure_credit_replay"]
    if failure_replay.get("version") != FAILURE_CREDIT_REPLAY_V4:
        raise LaunchError("reviewed preflight failure replay is not replay-v4")

    command = payload.get("trainer_command")
    if not isinstance(command, list) or not all(isinstance(item, str) for item in command):
        raise LaunchError("reviewed preflight has no fixed trainer command")
    try:
        v29_preflight.validate_trainer_command(
            tuple(command),
            paths=paths.preflight,
            initialize_from=paths.initialization_checkpoint,
        )
    except v29_preflight.PreflightError as exc:
        raise LaunchError(str(exc)) from exc
    if "--resume" in command or command.count("--initialize-from") != 1:
        raise LaunchError("v29 launcher permits model initialization only, never exact resume")

    environment_contract = payload.get("trainer_environment")
    if not isinstance(environment_contract, Mapping):
        raise LaunchError("reviewed preflight has no trainer environment contract")
    environment = _environment_from_contract(environment_contract)
    try:
        v29_preflight.validate_trainer_environment(
            environment,
            paths=paths.preflight,
        )
    except v29_preflight.PreflightError as exc:
        raise LaunchError(str(exc)) from exc
    claimed_environment_sha256 = payload.get(
        "trainer_environment_sha256",
    )
    if (
        not isinstance(claimed_environment_sha256, str)
        or re.fullmatch(
            r"[0-9a-f]{64}",
            claimed_environment_sha256,
        )
        is None
    ):
        raise LaunchError(
            "reviewed preflight has no exact trainer environment digest",
        )
    if claimed_environment_sha256 != _exact_trainer_environment_sha256(environment):
        raise LaunchError(
            "reviewed preflight exact trainer environment changed",
        )
    runtime = payload.get("runtime")
    if not isinstance(runtime, Mapping):
        raise LaunchError("reviewed preflight did not prove the GPU/simulator runtime")
    runtime = _validate_hermetic_runtime_probe_payload(
        runtime,
        paths=paths.preflight,
        trainer_environment_sha256=claimed_environment_sha256,
    )
    simulator = runtime.get("simulator")
    mechanics = runtime.get("runtime_mechanics")
    revival = runtime.get("training_revival")
    if (
        not isinstance(runtime.get("device_name"), str)
        or not isinstance(simulator, Mapping)
        or not isinstance(mechanics, Mapping)
        or mechanics.get("schema") != "sts2-runtime-mechanics-audit-v1"
        or mechanics.get("runtime_event_checked") is not True
        or mechanics.get("runtime_combat_checked") is not True
        or not isinstance(revival, Mapping)
    ):
        raise LaunchError("reviewed preflight runtime proof is incomplete")
    binding = embedded_readiness.get("report_binding")
    if not isinstance(binding, Mapping) or binding.get("runtime_identity") != runtime.get("runtime_identity"):
        raise LaunchError("reviewed preflight runtime readiness/live identity changed")
    if verify_live_shadow and embedded_readiness != _validate_runtime_readiness_evidence(
        paths,
        runtime=runtime,
        shadow=embedded_shadow,
    ):
        raise LaunchError("reviewed preflight runtime readiness report changed")
    return dict(payload)


def _expected_checkpoint_proof(paths: LaunchPaths) -> dict[str, Any]:
    return {
        "contract_version": FROZEN_CHECKPOINT_CONTRACT_VERSION,
        "contract_name": FIXED_FROZEN_CONTRACT_NAME,
        "exact_resume_permitted": False,
        "path": str(paths.initialization_checkpoint),
        "checkpoint_id": FIXED_INITIALIZATION_CHECKPOINT_ID,
        "manifest_sha256": FIXED_INITIALIZATION_MANIFEST_SHA256,
        "metadata_sha256": FIXED_INITIALIZATION_METADATA_SHA256,
        "source_environment_steps": FIXED_INITIALIZATION_STEP,
        "source_policy_version": FIXED_INITIALIZATION_POLICY_VERSION,
        "source_git_commit": FIXED_INITIALIZATION_SOURCE_GIT_COMMIT,
    }


def _checkpoint_proof(paths: LaunchPaths) -> dict[str, Any]:
    try:
        from sts2_rl.checkpoints.frozen import validate_frozen_checkpoint

        frozen_contract = _v28_frozen_contract()
        validated = validate_frozen_checkpoint(
            paths.artifact_root,
            contract=frozen_contract,
        )
    except (OSError, RuntimeError, TypeError, ValueError) as exc:
        raise LaunchError(f"V28_100K_FROZEN validation failed: {exc}") from exc
    checkpoint = validated.root.resolve(strict=False)
    if checkpoint != paths.initialization_checkpoint:
        raise LaunchError("frozen checkpoint validator returned another checkpoint")
    state = validated.metadata.get("training_state")
    if (
        not isinstance(state, Mapping)
        or state.get("environment_steps") != FIXED_INITIALIZATION_STEP
        or state.get("policy_version") != FIXED_INITIALIZATION_POLICY_VERSION
    ):
        raise LaunchError("fixed checkpoint training state changed")
    return _expected_checkpoint_proof(paths)


def _launch_contract_path(
    paths: LaunchPaths,
    *,
    launch_id: str,
) -> Path:
    canonical_launch_id = _canonical_uuid_text(
        launch_id,
        label="supervised launch ID",
    )
    return (paths.launcher_dir / "launch-contracts" / f"{RUN_NAME}-{canonical_launch_id}.json").resolve(strict=False)


def _canonical_uuid_text(value: object, *, label: str) -> str:
    if not isinstance(value, str):
        raise LaunchError(f"{label} must be a canonical UUID")
    try:
        canonical = str(uuid.UUID(value))
    except (ValueError, AttributeError) as exc:
        raise LaunchError(f"{label} must be a canonical UUID") from exc
    if value != canonical:
        raise LaunchError(f"{label} must use canonical lowercase UUID text")
    return canonical


def _supervised_launch_contract_payload(
    paths: LaunchPaths,
    *,
    preflight: Mapping[str, Any],
    launch_id: str,
    created_at_utc: str,
    created_unix_s: float,
) -> dict[str, Any]:
    selection = preflight.get("selection")
    initialization = preflight.get("initialization")
    shadow = preflight.get("shadow_validation")
    trainer_environment = preflight.get("trainer_environment")
    implementation_source = preflight.get("git")
    source_authority = preflight.get("source_authority")
    runtime_readiness = preflight.get("runtime_readiness_evidence")
    if (
        not isinstance(selection, Mapping)
        or not isinstance(initialization, Mapping)
        or not isinstance(shadow, Mapping)
        or not isinstance(trainer_environment, Mapping)
        or not isinstance(implementation_source, Mapping)
        or not isinstance(source_authority, Mapping)
        or not isinstance(runtime_readiness, Mapping)
    ):
        raise LaunchError(
            "cannot build supervised launch contract from incomplete preflight",
        )
    migration = initialization.get("migration_proof")
    reports = shadow.get("reports")
    shadow_contract = shadow.get("contract")
    if (
        not isinstance(migration, Mapping)
        or not isinstance(reports, Mapping)
        or not isinstance(shadow_contract, Mapping)
    ):
        raise LaunchError(
            "cannot build supervised launch contract from incomplete proofs",
        )
    semantics = reports.get("semantics_historical")
    evidence = reports.get("evidence_live")
    actor_evidence = reports.get("evidence_actor")
    initial_state = migration.get("new_training_state")
    expected_state_keys = {
        "actor_policy_version",
        "consumed_unrolls",
        "environment_steps",
        "episodes",
        "evaluation_episodes",
        "learner_updates",
        "maximum_observed_candidates",
        "policy_version",
    }
    if (
        not isinstance(semantics, Mapping)
        or not isinstance(evidence, Mapping)
        or not isinstance(actor_evidence, Mapping)
        or not isinstance(initial_state, Mapping)
        or set(initial_state) != expected_state_keys
        or any(isinstance(value, bool) or not isinstance(value, int) or value != 0 for value in initial_state.values())
    ):
        raise LaunchError(
            "supervised launch contract requires the complete fresh TrainingState",
        )
    payload = {
        "schema_version": SUPERVISED_LAUNCH_CONTRACT_SCHEMA,
        "launch_id": launch_id,
        "run_name": RUN_NAME,
        "created_at_utc": created_at_utc,
        "created_unix_s": created_unix_s,
        "config_fingerprint_sha256": preflight.get(
            "config_fingerprint_sha256",
        ),
        "effective_config_fingerprint_sha256": preflight.get(
            "effective_config_fingerprint_sha256",
        ),
        "trainer_environment_sha256": preflight.get(
            "trainer_environment_sha256",
        ),
        "implementation_source": dict(implementation_source),
        "source_authority": dict(source_authority),
        "runtime_readiness_evidence": dict(runtime_readiness),
        "seal_provenance": {
            "schema_version": EXTERNAL_SEAL_SCHEMA,
            "algorithm": "sha256",
            "digest_location": "supervisor_manifest.launch_contract.sha256",
            "self_digest_embedded": False,
        },
        "source_checkpoint": {
            "path": str(paths.initialization_checkpoint),
            "checkpoint_id": selection.get("checkpoint_id"),
            "environment_steps": selection.get(
                "source_environment_steps",
            ),
            "policy_version": selection.get("source_policy_version"),
            "manifest_sha256": selection.get("manifest_sha256"),
            "metadata_sha256": selection.get("metadata_sha256"),
            "usage": "model_parameter_initialization_only",
        },
        "shadow_validation": {
            "contract_sha256": shadow_contract.get("sha256"),
            "semantics_report_sha256": semantics.get("sha256"),
            "evidence_report_sha256": evidence.get("sha256"),
            "actor_evidence_report_sha256": actor_evidence.get(
                "sha256",
            ),
            "validated_code_mapping_sha256": source_authority.get(
                "validated_code_mapping_sha256",
            ),
            "evidence_generation_commit": (
                source_authority.get("evidence_generation_source", {}).get(
                    "implementation_commit",
                )
                if isinstance(source_authority.get("evidence_generation_source"), Mapping)
                else None
            ),
            "evidence_generation_tree": (
                source_authority.get("evidence_generation_source", {}).get(
                    "implementation_tree",
                )
                if isinstance(source_authority.get("evidence_generation_source"), Mapping)
                else None
            ),
        },
        "initial_training_state": dict(initial_state),
        "migration_proof_sha256": _canonical_json_sha256(
            dict(migration),
        ),
    }
    _validate_supervised_launch_contract_payload(
        payload,
        paths=paths,
        preflight=preflight,
        launch_id=launch_id,
    )
    return payload


def _validate_supervised_launch_contract_payload(
    value: object,
    *,
    paths: LaunchPaths,
    preflight: Mapping[str, Any],
    launch_id: str,
) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        raise LaunchError("supervised launch contract is malformed")
    expected_keys = {
        "schema_version",
        "launch_id",
        "run_name",
        "created_at_utc",
        "created_unix_s",
        "config_fingerprint_sha256",
        "effective_config_fingerprint_sha256",
        "trainer_environment_sha256",
        "implementation_source",
        "source_authority",
        "seal_provenance",
        "source_checkpoint",
        "shadow_validation",
        "runtime_readiness_evidence",
        "initial_training_state",
        "migration_proof_sha256",
    }
    if set(value) != expected_keys:
        raise LaunchError("supervised launch contract fields changed")
    if (
        value.get("schema_version") != SUPERVISED_LAUNCH_CONTRACT_SCHEMA
        or value.get("launch_id") != launch_id
        or value.get("run_name") != RUN_NAME
        or value.get("config_fingerprint_sha256") != FIXED_CONFIG_FINGERPRINT_SHA256
        or value.get("effective_config_fingerprint_sha256") != FIXED_RUNTIME_CONFIG_FINGERPRINT_SHA256
        or not isinstance(value.get("created_at_utc"), str)
        or not value["created_at_utc"]
        or isinstance(value.get("created_unix_s"), bool)
        or not isinstance(value.get("created_unix_s"), int | float)
        or not math.isfinite(float(value["created_unix_s"]))
        or float(value["created_unix_s"]) <= 0.0
    ):
        raise LaunchError("supervised launch contract identity changed")
    _canonical_uuid_text(
        value.get("launch_id"),
        label="supervised launch contract launch ID",
    )
    initialization = preflight.get("initialization")
    selection = preflight.get("selection")
    shadow = preflight.get("shadow_validation")
    if not isinstance(initialization, Mapping) or not isinstance(selection, Mapping) or not isinstance(shadow, Mapping):
        raise LaunchError("manifest lacks authorities for launch contract")
    migration = initialization.get("migration_proof")
    reports = shadow.get("reports")
    contract = shadow.get("contract")
    if not isinstance(migration, Mapping) or not isinstance(reports, Mapping) or not isinstance(contract, Mapping):
        raise LaunchError("manifest launch-contract authorities are malformed")
    expected_source = {
        "path": str(paths.initialization_checkpoint),
        "checkpoint_id": FIXED_INITIALIZATION_CHECKPOINT_ID,
        "environment_steps": FIXED_INITIALIZATION_STEP,
        "policy_version": FIXED_INITIALIZATION_POLICY_VERSION,
        "manifest_sha256": FIXED_INITIALIZATION_MANIFEST_SHA256,
        "metadata_sha256": FIXED_INITIALIZATION_METADATA_SHA256,
        "usage": "model_parameter_initialization_only",
    }
    if value.get("source_checkpoint") != expected_source:
        raise LaunchError("supervised launch contract source changed")
    trainer_environment = preflight.get("trainer_environment")
    if not isinstance(trainer_environment, Mapping) or value.get("trainer_environment_sha256") != preflight.get(
        "trainer_environment_sha256"
    ):
        raise LaunchError(
            "supervised launch contract trainer environment changed",
        )
    implementation_source = _validate_git_source_mapping(
        value.get("implementation_source"),
        checkout_root=paths.checkout_root,
        label="supervised launch implementation source",
    )
    if implementation_source != preflight.get("git"):
        raise LaunchError("supervised launch implementation source changed")
    source_authority = value.get("source_authority")
    if not isinstance(source_authority, Mapping) or source_authority != preflight.get("source_authority"):
        raise LaunchError("supervised launch source authority changed")
    expected_seal = {
        "schema_version": EXTERNAL_SEAL_SCHEMA,
        "algorithm": "sha256",
        "digest_location": "supervisor_manifest.launch_contract.sha256",
        "self_digest_embedded": False,
    }
    if value.get("seal_provenance") != expected_seal:
        raise LaunchError("supervised launch external seal provenance changed")
    if value.get("runtime_readiness_evidence") != preflight.get("runtime_readiness_evidence"):
        raise LaunchError("supervised launch runtime readiness evidence changed")
    semantics = reports.get("semantics_historical")
    evidence = reports.get("evidence_live")
    actor_evidence = reports.get("evidence_actor")
    if (
        not isinstance(semantics, Mapping)
        or not isinstance(evidence, Mapping)
        or not isinstance(actor_evidence, Mapping)
    ):
        raise LaunchError("manifest shadow report proof is malformed")
    expected_shadow = {
        "contract_sha256": SHADOW_CONTRACT_SHA256,
        "semantics_report_sha256": SEMANTICS_SHADOW_SHA256,
        "evidence_report_sha256": EVIDENCE_SHADOW_SHA256,
        "actor_evidence_report_sha256": (ACTOR_EVIDENCE_SHADOW_SHA256),
        "validated_code_mapping_sha256": source_authority.get(
            "validated_code_mapping_sha256",
        ),
        "evidence_generation_commit": source_authority.get(
            "evidence_generation_source",
            {},
        ).get("implementation_commit")
        if isinstance(source_authority.get("evidence_generation_source"), Mapping)
        else None,
        "evidence_generation_tree": source_authority.get(
            "evidence_generation_source",
            {},
        ).get("implementation_tree")
        if isinstance(source_authority.get("evidence_generation_source"), Mapping)
        else None,
    }
    if value.get("shadow_validation") != expected_shadow:
        raise LaunchError("supervised launch contract shadow proof changed")
    expected_state = migration.get("new_training_state")
    if value.get("initial_training_state") != expected_state:
        raise LaunchError("supervised launch contract initial state changed")
    if value.get("migration_proof_sha256") != _canonical_json_sha256(
        dict(migration),
    ):
        raise LaunchError("supervised launch contract migration proof changed")
    return dict(value)


def _create_supervised_launch_contract(
    paths: LaunchPaths,
    *,
    preflight: Mapping[str, Any],
    launch_id: str,
    created_at_utc: str,
    created_unix_s: float,
) -> dict[str, Any]:
    payload = _supervised_launch_contract_payload(
        paths,
        preflight=preflight,
        launch_id=launch_id,
        created_at_utc=created_at_utc,
        created_unix_s=created_unix_s,
    )
    path = _launch_contract_path(paths, launch_id=launch_id)
    digest = _write_immutable_json(path, payload)
    return {
        "schema_version": SUPERVISED_LAUNCH_CONTRACT_SCHEMA,
        "launch_id": launch_id,
        "path": str(path),
        "sha256": digest,
    }


def _validate_launch_contract_pin(
    value: object,
    *,
    paths: LaunchPaths,
    launch_id: str,
) -> dict[str, Any]:
    if not isinstance(value, Mapping) or set(value) != {
        "schema_version",
        "launch_id",
        "path",
        "sha256",
    }:
        raise LaunchError("supervised launch-contract pin is malformed")
    expected_path = _launch_contract_path(paths, launch_id=launch_id)
    raw_path = value.get("path")
    digest = value.get("sha256")
    if (
        value.get("schema_version") != SUPERVISED_LAUNCH_CONTRACT_SCHEMA
        or value.get("launch_id") != launch_id
        or not isinstance(raw_path, str)
        or Path(raw_path).resolve(strict=False) != expected_path
        or not isinstance(digest, str)
        or re.fullmatch(r"[0-9a-f]{64}", digest) is None
    ):
        raise LaunchError("supervised launch-contract pin changed")
    return dict(value)


def _validate_launch_contract_file(
    value: object,
    *,
    paths: LaunchPaths,
    preflight: Mapping[str, Any],
    launch_id: str,
) -> dict[str, Any]:
    pin = _validate_launch_contract_pin(
        value,
        paths=paths,
        launch_id=launch_id,
    )
    path = Path(pin["path"])
    if not path.is_file() or _sha256_file(path) != pin["sha256"]:
        raise LaunchError("immutable supervised launch contract hash changed")
    payload = _load_json_object(
        path,
        label="immutable supervised launch contract",
    )
    _validate_supervised_launch_contract_payload(
        payload,
        paths=paths,
        preflight=preflight,
        launch_id=launch_id,
    )
    return pin


def _validate_embedded_launch_contract(
    value: object,
    *,
    pin_value: object,
    paths: LaunchPaths,
    preflight: Mapping[str, Any],
    launch_id: str,
) -> dict[str, Any]:
    """Validate the post-spawn launch authority without reopening its file.

    The immutable file is re-hashed immediately before ``Popen``.  Once the
    child has loaded that file, lifecycle recovery must not depend on the
    small filesystem copy remaining readable: its canonical payload and hash
    are therefore embedded in the durable launch manifest as well.
    """

    pin = _validate_launch_contract_pin(
        pin_value,
        paths=paths,
        launch_id=launch_id,
    )
    payload = _validate_supervised_launch_contract_payload(
        value,
        paths=paths,
        preflight=preflight,
        launch_id=launch_id,
    )
    if _sha256_bytes(_immutable_json_bytes(payload)) != pin["sha256"]:
        raise LaunchError(
            "embedded supervised launch contract hash differs from its pin",
        )
    return payload


def _supervised_trainer_command(
    base_command: Sequence[str],
    *,
    launch_contract: Mapping[str, Any],
) -> tuple[str, ...]:
    if "--launch-contract" in base_command or "--launch-contract-sha256" in base_command:
        raise LaunchError("base trainer command already contains supervised launch arguments")
    return (
        *base_command,
        "--launch-contract",
        str(launch_contract["path"]),
        "--launch-contract-sha256",
        str(launch_contract["sha256"]),
    )


def _supervisor_command(
    paths: LaunchPaths,
    manifest_path: Path,
) -> tuple[str, ...]:
    """Return the one command authorized to supervise ``manifest_path``."""

    return (
        str(paths.venv_python),
        str(Path(__file__).resolve()),
        "supervise",
        "--manifest",
        str(manifest_path.resolve(strict=False)),
    )


def _trainer_bootstrap_script(paths: LaunchPaths) -> Path:
    script = (paths.package_root / "scripts" / "supervised_trainer_bootstrap.py").resolve(strict=False)
    if script.parent != (paths.package_root / "scripts").resolve(strict=False):
        raise LaunchError("trainer bootstrap helper path changed")
    return script


def _trainer_bootstrap_command(
    paths: LaunchPaths,
    *,
    supervisor_identity: ProcessIdentity,
    gate_fd: int,
    status_fd: int,
    nonce: str,
    trainer_command: Sequence[str],
) -> tuple[str, ...]:
    if gate_fd < 3 or status_fd < 3 or gate_fd == status_fd or re.fullmatch(r"[0-9a-f]{64}", nonce) is None:
        raise LaunchError("trainer bootstrap descriptor/nonce authority is malformed")
    if not trainer_command or not all(isinstance(item, str) and item for item in trainer_command):
        raise LaunchError("trainer bootstrap final command is malformed")
    return (
        str(paths.venv_python),
        str(_trainer_bootstrap_script(paths)),
        "--expected-parent-pid",
        str(supervisor_identity.pid),
        "--expected-parent-start-ticks",
        str(supervisor_identity.proc_start_ticks),
        "--gate-fd",
        str(gate_fd),
        "--status-fd",
        str(status_fd),
        "--nonce",
        nonce,
        "--timeout-seconds",
        str(int(TRAINER_BOOTSTRAP_READY_TIMEOUT_SECONDS)),
        "--",
        *trainer_command,
    )


def _validate_trainer_bootstrap_command(
    value: object,
    *,
    paths: LaunchPaths,
    supervisor_identity: ProcessIdentity,
    trainer_command: Sequence[str],
    nonce_sha256: str,
) -> tuple[str, ...]:
    if not isinstance(value, list) or not all(isinstance(item, str) and item for item in value):
        raise LaunchError("trainer bootstrap command is malformed")
    command = tuple(value)
    prefix_length = 15
    if len(command) <= prefix_length or command[14] != "--":
        raise LaunchError("trainer bootstrap command fields changed")
    expected_labels = (
        "--expected-parent-pid",
        "--expected-parent-start-ticks",
        "--gate-fd",
        "--status-fd",
        "--nonce",
        "--timeout-seconds",
    )
    if (
        command[0] != str(paths.venv_python)
        or Path(command[1]).resolve(strict=False) != _trainer_bootstrap_script(paths)
        or command[2:14:2] != expected_labels
        or tuple(command[prefix_length:]) != tuple(trainer_command)
    ):
        raise LaunchError("trainer bootstrap command changed")
    try:
        parent_pid = int(command[3])
        parent_ticks = int(command[5])
        gate_fd = int(command[7])
        status_fd = int(command[9])
    except ValueError as exc:
        raise LaunchError("trainer bootstrap command integer fields are malformed") from exc
    nonce = command[11]
    if (
        parent_pid != supervisor_identity.pid
        or parent_ticks != supervisor_identity.proc_start_ticks
        or gate_fd < 3
        or status_fd < 3
        or gate_fd == status_fd
        or re.fullmatch(r"[0-9a-f]{64}", nonce) is None
        or _sha256_bytes(nonce.encode("ascii")) != nonce_sha256
        or command[13] != str(int(TRAINER_BOOTSTRAP_READY_TIMEOUT_SECONDS))
    ):
        raise LaunchError("trainer bootstrap command authority changed")
    return command


def _git_provenance(paths: LaunchPaths) -> dict[str, Any]:
    environment = _environment_from_contract(
        v29_preflight.trainer_environment_contract(v29_preflight.build_trainer_environment(paths.preflight))
    )
    top_level = _run_checked(
        ("git", "rev-parse", "--show-toplevel"),
        cwd=paths.checkout_root,
        environment=environment,
    ).strip()
    if Path(top_level).resolve(strict=False) != paths.checkout_root:
        raise LaunchError("v29 checkout root is not the Git toplevel")
    object_format = _run_checked(
        ("git", "rev-parse", "--show-object-format"),
        cwd=paths.checkout_root,
        environment=environment,
    ).strip()
    if object_format not in {"sha1", "sha256"}:
        raise LaunchError("git returned an unsupported object format")
    oid_length = 40 if object_format == "sha1" else 64
    commit = _run_checked(
        ("git", "rev-parse", "HEAD"),
        cwd=paths.checkout_root,
        environment=environment,
    ).strip()
    if re.fullmatch(rf"[0-9a-f]{{{oid_length}}}", commit) is None:
        raise LaunchError(f"git returned an invalid full commit: {commit!r}")
    tree = _run_checked(
        ("git", "rev-parse", "HEAD^{tree}"),
        cwd=paths.checkout_root,
        environment=environment,
    ).strip()
    if re.fullmatch(rf"[0-9a-f]{{{oid_length}}}", tree) is None:
        raise LaunchError(f"git returned an invalid full tree: {tree!r}")
    dirty = _run_checked(
        ("git", "status", "--porcelain=v1", "--untracked-files=normal"),
        cwd=paths.checkout_root,
        environment=environment,
    )
    if dirty.strip():
        raise LaunchError("v29 start requires a clean committed checkout")
    return {
        "checkout_root": str(paths.checkout_root),
        "git_object_format": object_format,
        "implementation_commit": commit,
        "implementation_tree": tree,
        "worktree_clean": True,
    }


def _source_authority_proof(
    paths: LaunchPaths,
    *,
    shadow: Mapping[str, Any],
    implementation_source: Mapping[str, Any],
) -> dict[str, Any]:
    generation_source = _validate_git_source_mapping(
        shadow.get("generation_source"),
        checkout_root=paths.checkout_root,
        label="evidence generation source",
    )
    implementation = _validate_git_source_mapping(
        implementation_source,
        checkout_root=paths.checkout_root,
        label="launch implementation source",
    )
    if generation_source["git_object_format"] != implementation["git_object_format"]:
        raise LaunchError("evidence generation and launch commits use different Git object formats")
    generation_commit = str(generation_source["implementation_commit"])
    launch_commit = str(implementation["implementation_commit"])
    if generation_commit == launch_commit:
        raise LaunchError("two-phase evidence seal requires distinct generation commit A and launch commit B")
    environment = _environment_from_contract(
        v29_preflight.trainer_environment_contract(v29_preflight.build_trainer_environment(paths.preflight))
    )
    observed_generation_tree = _run_checked(
        ("git", "rev-parse", f"{generation_commit}^{{tree}}"),
        cwd=paths.checkout_root,
        environment=environment,
    ).strip()
    if observed_generation_tree != generation_source["implementation_tree"]:
        raise LaunchError("evidence generation tree differs from commit A")
    ancestor = subprocess.run(
        ("git", "merge-base", "--is-ancestor", generation_commit, launch_commit),
        cwd=paths.checkout_root,
        env=environment,
        check=False,
        capture_output=True,
        timeout=30.0,
    )
    if ancestor.returncode != 0:
        raise LaunchError("evidence generation commit A is not an ancestor of launch commit B")
    changed = _run_checked(
        (
            "git",
            "diff",
            "--name-only",
            "--no-renames",
            f"{generation_commit}..{launch_commit}",
            "--",
        ),
        cwd=paths.checkout_root,
        environment=environment,
    )
    observed_paths = sorted(line for line in changed.splitlines() if line)
    allowed_paths = sorted(REVIEWED_SEAL_PATHS)
    if observed_paths != allowed_paths:
        raise LaunchError(
            "seal commit B changed files outside the exact reviewed seal set: "
            f"expected={allowed_paths!r} actual={observed_paths!r}",
        )
    code_hashes = shadow.get("validated_code_sha256")
    if not isinstance(code_hashes, Mapping):
        raise LaunchError("shadow proof has no validated code closure")
    return {
        "schema_version": SOURCE_AUTHORITY_SCHEMA,
        "evidence_generation_source": generation_source,
        "validated_code_sha256": dict(code_hashes),
        "validated_code_mapping_sha256": _canonical_json_sha256(dict(code_hashes)),
        "allowed_seal_paths": allowed_paths,
        "observed_seal_paths": observed_paths,
    }


def run_preflight(
    paths: LaunchPaths,
    *,
    reviewed_preflight: Path | None = None,
    enforce_active_root: bool = True,
    verify_runtime: bool = True,
) -> dict[str, Any]:
    paths = validate_layout(paths, enforce_active_root=enforce_active_root)
    require_exact_artifact_environment(paths)
    try:
        live = v29_preflight.run_preflight(
            paths.preflight,
            initialize_from=paths.initialization_checkpoint,
            enforce_active_root=enforce_active_root,
            verify_runtime=verify_runtime,
        )
    except v29_preflight.PreflightError as exc:
        raise LaunchError(str(exc)) from exc
    live = _validate_preflight_payload(live, paths=paths)

    reviewed: dict[str, Any]
    review_source: dict[str, Any]
    if reviewed_preflight is None:
        reviewed = live
        review_source = {"mode": "live-fixed-preflight", "path": None, "sha256": None}
    else:
        review_path = reviewed_preflight.expanduser().resolve(strict=True)
        reviewed = _validate_preflight_payload(
            _load_json_object(review_path, label="reviewed preflight payload"),
            paths=paths,
        )
        for key in (
            "run_name",
            "initialization",
            "config_fingerprint_sha256",
            "effective_config_fingerprint_sha256",
            "abi_contract",
            "shadow_validation",
            "runtime_readiness_evidence",
            "trainer_command",
            "runtime",
        ):
            if reviewed.get(key) != live.get(key):
                raise LaunchError(f"reviewed preflight {key} no longer matches live preflight")
        review_source = {
            "mode": "explicit-reviewed-preflight",
            "path": str(review_path),
            "sha256": _sha256_file(review_path),
        }
    implementation_source = _git_provenance(paths)
    source_authority = _source_authority_proof(
        paths,
        shadow=reviewed["shadow_validation"],
        implementation_source=implementation_source,
    )
    return {
        **reviewed,
        "schema_version": MANIFEST_SCHEMA,
        # The supervised manifest later owns the top-level ``status`` field.
        # Keep the non-launching proof under a non-colliding immutable key.
        "preflight_status": "preflight-passed",
        "selection": _checkpoint_proof(paths),
        "review_source": review_source,
        "git": implementation_source,
        "source_authority": source_authority,
        "config_file_sha256": _sha256_file(paths.preflight.config_path),
        "simulator_identity_sha256": _sha256_file(paths.preflight.simulator_identity),
        "training_started": False,
    }


def capture_process_identity(pid: int) -> ProcessIdentity:
    proc_root = Path("/proc") / str(pid)
    try:
        stat = (proc_root / "stat").read_text(encoding="utf-8")
        command_line = (proc_root / "cmdline").read_bytes()
        executable = os.readlink(proc_root / "exe")
    except OSError as exc:
        raise LaunchError(f"cannot capture Linux process identity for PID {pid}: {exc}") from exc
    close = stat.rfind(")")
    fields = stat[close + 2 :].split() if close >= 0 else []
    if len(fields) < 20 or not command_line:
        raise LaunchError(f"Linux process identity is incomplete for PID {pid}")
    try:
        start_ticks = int(fields[19])
    except ValueError as exc:
        raise LaunchError(f"Linux process start marker is invalid for PID {pid}") from exc
    return ProcessIdentity(
        pid=pid,
        proc_start_ticks=start_ticks,
        command_line_sha256=_sha256_bytes(command_line),
        executable=executable,
    )


def process_identity_matches(identity: ProcessIdentity) -> bool:
    try:
        return capture_process_identity(identity.pid) == identity
    except LaunchError:
        return False


def _command_line_sha256(command: Sequence[str]) -> str:
    if not command or not all(isinstance(item, str) and item for item in command):
        raise LaunchError("spawned process command is malformed")
    raw = b"\0".join(os.fsencode(item) for item in command) + b"\0"
    return _sha256_bytes(raw)


def _identity_matches_command(
    identity: ProcessIdentity,
    command: Sequence[str],
) -> bool:
    try:
        expected_command_sha256 = _command_line_sha256(command)
    except LaunchError:
        return False
    expected_executable = Path(command[0]).resolve(strict=False)
    observed_executable = Path(identity.executable).resolve(strict=False)
    return identity.command_line_sha256 == expected_command_sha256 and observed_executable == expected_executable


def _identity_from_mapping(value: object, *, label: str) -> ProcessIdentity:
    if not isinstance(value, Mapping):
        raise LaunchError(f"{label} process identity is malformed")
    expected_fields = {
        "pid",
        "proc_start_ticks",
        "command_line_sha256",
        "executable",
    }
    if set(value) != expected_fields:
        raise LaunchError(f"{label} process identity fields changed")
    pid = value["pid"]
    start_ticks = value["proc_start_ticks"]
    digest = value["command_line_sha256"]
    executable = value["executable"]
    if (
        isinstance(pid, bool)
        or not isinstance(pid, int)
        or pid <= 0
        or isinstance(start_ticks, bool)
        or not isinstance(start_ticks, int)
        or start_ticks <= 0
        or not isinstance(digest, str)
        or re.fullmatch(r"[0-9a-f]{64}", digest) is None
        or not isinstance(executable, str)
        or not executable
        or not Path(executable).is_absolute()
    ):
        raise LaunchError(f"{label} process identity is malformed")
    return ProcessIdentity(
        pid=pid,
        proc_start_ticks=start_ticks,
        command_line_sha256=digest,
        executable=executable,
    )


def _optional_identity(value: object, *, label: str) -> ProcessIdentity | None:
    return None if value is None else _identity_from_mapping(value, label=label)


def _capture_child(process: subprocess.Popen[bytes]) -> ProcessIdentity:
    last_error: Exception | None = None
    for _ in range(50):
        if process.poll() is not None:
            raise LaunchError(f"child exited before identity capture: {process.returncode}")
        try:
            return capture_process_identity(process.pid)
        except LaunchError as exc:
            last_error = exc
            time.sleep(0.02)
    process.terminate()
    try:
        returncode = process.wait(timeout=5.0)
    except subprocess.TimeoutExpired:
        process.kill()
        returncode = process.wait(timeout=5.0)
    raise LaunchError(
        f"could not capture child process identity; child was reaped with returncode={returncode}: {last_error}",
    )


def _state_path(paths: LaunchPaths) -> Path:
    return paths.launcher_dir / f"{RUN_NAME}.state.json"


def _run_root(paths: LaunchPaths) -> Path:
    return (paths.artifact_root / "runs" / RUN_NAME).resolve(strict=False)


@contextmanager
def _terminal_lock(paths: LaunchPaths, launch_id: str):  # type: ignore[no-untyped-def]
    lock_dir = paths.launcher_dir / "watchdog-locks"
    lock_dir.mkdir(parents=True, exist_ok=True)
    descriptor = os.open(lock_dir / f"{launch_id}.lock", os.O_RDWR | os.O_CREAT, 0o600)
    try:
        if fcntl is not None:
            fcntl.flock(descriptor, fcntl.LOCK_EX)
        yield
    finally:
        if fcntl is not None:
            fcntl.flock(descriptor, fcntl.LOCK_UN)
        os.close(descriptor)


def _state_payload(manifest_path: Path, manifest: Mapping[str, Any]) -> dict[str, Any]:
    try:
        manifest_directory_mtime_ns = manifest_path.parent.stat().st_mtime_ns
    except OSError:
        manifest_directory_mtime_ns = None
    return {
        "schema_version": STATE_SCHEMA,
        "run_name": RUN_NAME,
        "launch_id": manifest.get("launch_id"),
        "manifest_path": str(manifest_path),
        "status": manifest.get("status"),
        "preflight_training_started": manifest.get(
            "preflight_training_started",
        ),
        "training_started": manifest.get("training_started"),
        "trainer_bootstrap_protocol": manifest.get("trainer_bootstrap_protocol"),
        "trainer_bootstrap_phase": manifest.get("trainer_bootstrap_phase"),
        "trainer_bootstrap_process_identity": manifest.get(
            "trainer_bootstrap_process_identity",
        ),
        "trainer_spawn_intent": manifest.get("trainer_spawn_intent"),
        "trainer_spawn_failed": manifest.get("trainer_spawn_failed"),
        "trainer_exit_observed": manifest.get("trainer_exit_observed"),
        "trainer_reaped_returncode": manifest.get(
            "trainer_reaped_returncode",
        ),
        "supervisor_process_identity": manifest.get("supervisor_process_identity"),
        "trainer_process_identity": manifest.get("trainer_process_identity"),
        "metrics_path": manifest.get("metrics_path"),
        "log_path": manifest.get("log_path"),
        "successor": manifest.get("successor"),
        "terminal": manifest.get("terminal"),
        "terminal_event_id": manifest.get("terminal_event_id"),
        "launch_contract": manifest.get("launch_contract"),
        "launch_contract_payload": manifest.get(
            "launch_contract_payload",
        ),
        "initialization": manifest.get("initialization"),
        "selection": manifest.get("selection"),
        "abi_contract": manifest.get("abi_contract"),
        "shadow_validation": manifest.get("shadow_validation"),
        "config_fingerprint_sha256": manifest.get("config_fingerprint_sha256"),
        "effective_config_fingerprint_sha256": manifest.get("effective_config_fingerprint_sha256"),
        "manifest_directory_mtime_ns": manifest_directory_mtime_ns,
        "written_at_utc": _utc_now(),
        "written_unix_s": time.time(),
    }


_STATE_MANIFEST_KEYS = (
    "schema_version",
    "run_name",
    "launch_id",
    "manifest_path",
    "status",
    "preflight_training_started",
    "training_started",
    "trainer_bootstrap_protocol",
    "trainer_bootstrap_phase",
    "trainer_bootstrap_process_identity",
    "trainer_spawn_intent",
    "trainer_spawn_failed",
    "trainer_exit_observed",
    "trainer_reaped_returncode",
    "supervisor_process_identity",
    "trainer_process_identity",
    "metrics_path",
    "log_path",
    "successor",
    "terminal",
    "terminal_event_id",
    "launch_contract",
    "launch_contract_payload",
    "initialization",
    "selection",
    "abi_contract",
    "shadow_validation",
    "config_fingerprint_sha256",
    "effective_config_fingerprint_sha256",
)


def _state_matches_manifest(
    state: Mapping[str, Any],
    *,
    manifest_path: Path,
    manifest: Mapping[str, Any],
) -> bool:
    expected = _state_payload(manifest_path, manifest)
    return all(state.get(key) == expected.get(key) for key in _STATE_MANIFEST_KEYS)


def _persist_manifest(
    paths: LaunchPaths,
    *,
    manifest_path: Path,
    manifest: Mapping[str, Any],
) -> None:
    atomic_write_json(manifest_path, manifest)
    atomic_write_json(_state_path(paths), _state_payload(manifest_path, manifest))


def _latest_authoritative_manifest(
    paths: LaunchPaths,
) -> tuple[Path, dict[str, Any]] | None:
    if not paths.manifest_dir.is_dir():
        return None
    candidates: list[
        tuple[
            tuple[float, int, str],
            Path,
            dict[str, Any] | None,
            str | None,
        ]
    ] = []
    for candidate in paths.manifest_dir.glob(f"{RUN_NAME}-*.launch.json"):
        resolved = candidate.resolve(strict=False)
        timestamp_match = re.match(
            rf"^{re.escape(RUN_NAME)}-(\d{{8}}-\d{{6}})-",
            candidate.name,
        )
        if timestamp_match is not None:
            try:
                filename_unix_s = (
                    datetime.strptime(
                        timestamp_match.group(1),
                        "%Y%m%d-%H%M%S",
                    )
                    .replace(tzinfo=UTC)
                    .timestamp()
                )
            except ValueError:
                filename_unix_s = 0.0
        else:
            filename_unix_s = 0.0
        try:
            mtime_ns = candidate.stat().st_mtime_ns
        except OSError:
            mtime_ns = 0
        ordering = (filename_unix_s, mtime_ns, candidate.name)
        try:
            raw = _load_json_object(
                candidate,
                label="candidate supervised launch manifest",
            )
        except LaunchError as exc:
            candidates.append((ordering, resolved, None, str(exc)))
            continue
        if raw.get("schema_version") != MANIFEST_SCHEMA or raw.get("run_name") != RUN_NAME:
            candidates.append(
                (
                    ordering,
                    resolved,
                    None,
                    "candidate supervised launch manifest has another schema or run name",
                )
            )
            continue
        launch_id = raw.get("launch_id")
        try:
            _canonical_uuid_text(
                launch_id,
                label="candidate supervised launch ID",
            )
        except LaunchError as exc:
            candidates.append((ordering, resolved, None, str(exc)))
            continue
        candidates.append(
            (
                ordering,
                resolved,
                raw,
                None,
            )
        )
    if not candidates:
        return None
    _, manifest_path, selected_raw, error = max(
        candidates,
        key=lambda item: item[0],
    )
    if selected_raw is None:
        raise LaunchError(
            f"newest supervised launch manifest is invalid; refusing to fall back to an older launch: {error}",
        )
    return manifest_path, _validate_manifest(paths, manifest_path)


def _validate_selection_proof(value: object, *, paths: LaunchPaths) -> None:
    if not isinstance(value, Mapping):
        raise LaunchError("supervised manifest has no fixed checkpoint proof")
    expected = _expected_checkpoint_proof(paths)
    if dict(value) != expected:
        raise LaunchError("supervised manifest checkpoint proof changed")


def _nonempty_timestamp(value: object) -> bool:
    return isinstance(value, str) and bool(value)


def _validate_trainer_spawn_lifecycle(
    manifest: Mapping[str, Any],
) -> None:
    if manifest.get("trainer_bootstrap_protocol") != TRAINER_BOOTSTRAP_PROTOCOL:
        raise LaunchError("trainer bootstrap protocol changed")
    phase = manifest.get("trainer_bootstrap_phase")
    phases = {
        "none",
        "intent",
        "bootstrap_ready",
        "exec_authorized",
        "exec_observed",
        "preexec_failed",
        "reaped",
    }
    if phase not in phases:
        raise LaunchError("trainer bootstrap phase is malformed")
    training_started = manifest.get("training_started")
    spawn_intent = manifest.get("trainer_spawn_intent")
    spawn_failed = manifest.get("trainer_spawn_failed")
    exit_observed = manifest.get("trainer_exit_observed")
    if not all(
        isinstance(value, bool)
        for value in (
            training_started,
            spawn_intent,
            spawn_failed,
            exit_observed,
        )
    ):
        raise LaunchError("trainer spawn lifecycle flags are malformed")

    intent_at = manifest.get("trainer_spawn_intent_at_utc")
    failure_at = manifest.get("trainer_spawn_failure_at_utc")
    exit_at = manifest.get("trainer_exit_observed_at_utc")
    returncode = manifest.get("trainer_reaped_returncode")
    raw_pid = manifest.get("trainer_spawned_pid")
    trainer_identity = manifest.get("trainer_process_identity")
    bootstrap_command = manifest.get("trainer_bootstrap_command")
    bootstrap_nonce_sha256 = manifest.get("trainer_bootstrap_nonce_sha256")
    bootstrap_identity = manifest.get("trainer_bootstrap_process_identity")
    bootstrap_ready_at = manifest.get("trainer_bootstrap_ready_at_utc")
    exec_authorized_at = manifest.get("trainer_bootstrap_exec_authorized_at_utc")
    exec_observed_at = manifest.get("trainer_bootstrap_exec_observed_at_utc")
    spawned_at = manifest.get("trainer_spawned_at_utc")

    if spawn_intent is False:
        if (
            phase != "none"
            or bootstrap_command is not None
            or bootstrap_nonce_sha256 is not None
            or bootstrap_identity is not None
            or bootstrap_ready_at is not None
            or exec_authorized_at is not None
            or exec_observed_at is not None
            or spawned_at is not None
            or training_started is True
            or spawn_failed is True
            or exit_observed is True
            or raw_pid is not None
            or trainer_identity is not None
        ):
            raise LaunchError("trainer bootstrap advanced without spawn intent")
        if intent_at is not None or failure_at is not None or exit_at is not None or returncode is not None:
            raise LaunchError("trainer bootstrap timestamps advanced without spawn intent")
        return

    if spawn_intent is True:
        if not _nonempty_timestamp(intent_at):
            raise LaunchError("trainer spawn intent lacks a durable timestamp")
        if phase == "none":
            raise LaunchError("trainer spawn intent lacks a bootstrap phase")
        if (
            not isinstance(bootstrap_command, list)
            or not bootstrap_command
            or not isinstance(bootstrap_nonce_sha256, str)
            or re.fullmatch(r"[0-9a-f]{64}", bootstrap_nonce_sha256) is None
        ):
            raise LaunchError("trainer spawn intent lacks bootstrap command authority")

    if spawn_failed is True:
        if (
            not _nonempty_timestamp(failure_at)
            or training_started is True
            or exit_observed is True
            or trainer_identity is not None
            or phase != "preexec_failed"
        ):
            raise LaunchError("trainer spawn-failure evidence is incoherent")
        if raw_pid is None and bootstrap_identity is not None:
            raise LaunchError("failed bootstrap identity has no PID authority")
        if raw_pid is not None and bootstrap_identity is None:
            raise LaunchError("failed bootstrap PID has no full identity")
    elif failure_at is not None:
        raise LaunchError("trainer spawn-failure timestamp has no failure flag")

    if raw_pid is not None and (
        isinstance(raw_pid, bool) or not isinstance(raw_pid, int) or raw_pid <= 0 or spawn_intent is not True
    ):
        raise LaunchError("trainer spawned PID is malformed or incoherent")
    if bootstrap_identity is not None and (
        raw_pid is None
        or not isinstance(bootstrap_identity, Mapping)
        or phase not in {"bootstrap_ready", "exec_authorized", "exec_observed", "preexec_failed", "reaped"}
    ):
        raise LaunchError("bootstrap process identity lacks durable PID authority")
    if phase in {"bootstrap_ready", "exec_authorized", "exec_observed", "reaped"} and (
        raw_pid is None or bootstrap_identity is None or not _nonempty_timestamp(bootstrap_ready_at)
    ):
        raise LaunchError("bootstrap-ready phase lacks durable process authority")
    if phase == "intent" and (
        raw_pid is not None
        or bootstrap_identity is not None
        or bootstrap_ready_at is not None
        or exec_authorized_at is not None
        or exec_observed_at is not None
    ):
        raise LaunchError("trainer bootstrap intent contains premature child authority")
    if phase in {"exec_authorized", "exec_observed", "reaped"} and not _nonempty_timestamp(
        exec_authorized_at,
    ):
        raise LaunchError("trainer bootstrap exec authorization lacks a durable timestamp")
    if phase == "bootstrap_ready" and (exec_authorized_at is not None or exec_observed_at is not None):
        raise LaunchError("bootstrap-ready phase contains premature exec evidence")
    if training_started is True and (
        spawn_intent is not True
        or spawn_failed is True
        or raw_pid is None
        or phase not in {"exec_observed", "reaped"}
        or not _nonempty_timestamp(spawned_at)
        or not _nonempty_timestamp(exec_observed_at)
        or trainer_identity is None
    ):
        raise LaunchError("training-started lifecycle lacks durable spawn authority")
    if trainer_identity is not None and (
        training_started is not True
        or raw_pid is None
        or phase not in {"exec_observed", "reaped"}
        or not _nonempty_timestamp(spawned_at)
    ):
        raise LaunchError(
            "trainer process identity lacks durable spawn authority",
        )

    if exit_observed is True:
        if (
            spawn_intent is not True
            or spawn_failed is True
            or training_started is not True
            or phase != "reaped"
            or raw_pid is None
            or not _nonempty_timestamp(spawned_at)
            or not _nonempty_timestamp(exit_at)
            or isinstance(returncode, bool)
            or not isinstance(returncode, int)
        ):
            raise LaunchError("trainer reaped-exit evidence is incoherent")
    elif exit_at is not None or returncode is not None:
        raise LaunchError("trainer reaped-exit data has no observation flag")
    if phase == "exec_observed" and exit_observed is True:
        raise LaunchError("exec-observed phase already contains reaped evidence")
    if phase == "reaped" and exit_observed is not True:
        raise LaunchError("reaped bootstrap phase lacks trainer exit evidence")
    if phase in {"intent", "bootstrap_ready", "exec_authorized"} and (
        training_started is True
        or trainer_identity is not None
        or exec_observed_at is not None
        or exit_observed is True
    ):
        raise LaunchError("pre-exec bootstrap phase contains trainer-start evidence")


def _validate_manifest_lifecycle(
    manifest: Mapping[str, Any],
    *,
    paths: LaunchPaths,
) -> None:
    _validate_trainer_spawn_lifecycle(manifest)
    status = manifest.get("status")
    active_statuses = {
        "supervisor-launching",
        "supervisor-running",
        "running",
    }
    if status not in active_statuses | TERMINAL_STATUSES:
        raise LaunchError("supervised manifest has an unknown lifecycle status")
    terminal = manifest.get("terminal")
    if status in active_statuses:
        if terminal is not None:
            raise LaunchError(
                "active supervised manifest contains terminal lifecycle data",
            )
        return
    if not isinstance(terminal, Mapping):
        raise LaunchError(
            "terminal supervised manifest has no terminal payload",
        )
    expected_terminal_keys = {
        "status",
        "classification",
        "returncode",
        "observed_at_utc",
        "watchdog_event_id",
        "watchdog_event_appended",
        "metrics_path",
        "fallback_event_path",
        "reconciliation_reason",
    }
    if set(terminal) != expected_terminal_keys:
        raise LaunchError("terminal supervised manifest fields changed")
    classification = terminal.get("classification")
    if (
        terminal.get("status") != status
        or not isinstance(classification, Mapping)
        or classification.get("status") != status
        or not isinstance(classification.get("kind"), str)
        or not classification["kind"]
        or not isinstance(classification.get("native_abort"), bool)
        or not isinstance(classification.get("append_run_failed"), bool)
        or not isinstance(terminal.get("observed_at_utc"), str)
        or not terminal["observed_at_utc"]
        or not isinstance(terminal.get("watchdog_event_appended"), bool)
    ):
        raise LaunchError("terminal supervised manifest is incoherent")
    returncode = terminal.get("returncode")
    if isinstance(returncode, bool) or (returncode is not None and not isinstance(returncode, int)):
        raise LaunchError("terminal supervised manifest return code is malformed")
    if manifest.get("trainer_exit_observed") is True and returncode != manifest.get("trainer_reaped_returncode"):
        raise LaunchError(
            "terminal return code differs from durable reaped-trainer evidence",
        )
    event_id = terminal.get("watchdog_event_id")
    if event_id is not None and (not isinstance(event_id, str) or re.fullmatch(r"[0-9a-f]{64}", event_id) is None):
        raise LaunchError("terminal supervised watchdog event ID is malformed")
    if manifest.get("terminal_event_id") != event_id:
        raise LaunchError(
            "terminal supervised watchdog event ID differs from manifest",
        )
    metrics = terminal.get("metrics_path")
    fallback = terminal.get("fallback_event_path")
    if metrics != manifest.get("metrics_path"):
        raise LaunchError(
            "terminal supervised metrics path differs from manifest binding",
        )
    metric_terminal_events: list[dict[str, Any]] = []
    if metrics is not None:
        if not isinstance(metrics, str):
            raise LaunchError("terminal supervised metrics path is malformed")
        resolved_metrics, _ = _validate_bound_successor_metrics(
            paths,
            manifest=manifest,
            metrics_path=Path(metrics),
        )
        metric_terminal_events = [
            item
            for item in _metrics_events(resolved_metrics)
            if item.get("event") in {"run_complete", "interrupt", "run_failed"}
        ]
        if len(metric_terminal_events) > 1:
            raise LaunchError(
                "terminal supervised metrics has multiple terminal events",
            )
    fallback_payload: dict[str, Any] | None = None
    if fallback is not None:
        if not isinstance(fallback, str):
            raise LaunchError("terminal fallback event path is malformed")
        resolved_fallback = Path(fallback).resolve(strict=False)
        if (
            not _is_within(
                resolved_fallback,
                (paths.launcher_dir / "terminal-events").resolve(
                    strict=False,
                ),
            )
            or not resolved_fallback.is_file()
        ):
            raise LaunchError(
                "terminal fallback event is outside authority or missing",
            )
        fallback_payload = _load_json_object(
            resolved_fallback,
            label="terminal fallback event",
        )
        launch_id = _canonical_uuid_text(
            manifest.get("launch_id"),
            label="terminal fallback launch ID",
        )
        launch_contract = manifest.get("launch_contract")
        if not isinstance(launch_contract, Mapping):
            raise LaunchError(
                "terminal fallback has no launch-contract authority",
            )
        expected_contract_sha256 = launch_contract.get("sha256")
        common_matches = (
            fallback_payload.get("launch_id") == launch_id
            and fallback_payload.get("run_name") == RUN_NAME
            and fallback_payload.get("launch_contract_sha256") == expected_contract_sha256
            and fallback_payload.get("returncode") == returncode
        )
        if not common_matches:
            raise LaunchError(
                "terminal fallback event differs from launch authority",
            )
        fallback_schema = fallback_payload.get("schema_version")
        if fallback_schema == WATCHDOG_EVENT_SCHEMA:
            if (
                fallback_payload.get("event") != "run_failed"
                or fallback_payload.get("source") != "persistent-native-exit-watchdog"
                or fallback_payload.get("watchdog_event_id") != event_id
                or resolved_fallback.name != f"{launch_id}.run_failed.json"
            ):
                raise LaunchError(
                    "terminal watchdog fallback event is incoherent",
                )
            fallback_classification = fallback_payload.get(
                "classification",
            )
            if (
                not isinstance(fallback_classification, Mapping)
                or fallback_classification.get("status") != "failed"
                or status != "failed"
            ):
                raise LaunchError(
                    "terminal watchdog fallback classification changed",
                )
        elif fallback_schema == SUPERVISOR_EMERGENCY_SCHEMA:
            if (
                fallback_payload.get("event") != "supervisor_recovery_failed"
                or event_id is not None
                or status != "failed"
                or classification.get("kind") != "supervisor_finalization_failure"
                or resolved_fallback.name != f"{launch_id}.supervisor_recovery_failed.json"
            ):
                raise LaunchError(
                    "terminal supervisor-emergency fallback is incoherent",
                )
        else:
            raise LaunchError(
                "terminal fallback event schema is unsupported",
            )
    if status in {"completed", "interrupted"}:
        expected_kind = "clean_completion" if status == "completed" else "runtime_interrupt"
        expected_event = "run_complete" if status == "completed" else "interrupt"
        if (
            classification.get("kind") != expected_kind
            or metrics is None
            or fallback is not None
            or classification.get("append_run_failed") is not False
            or len(metric_terminal_events) != 1
            or metric_terminal_events[0].get("event") != expected_event
        ):
            raise LaunchError(
                "successful/interrupted terminal lifecycle lacks bound evidence",
            )
    elif metrics is None and fallback is None:
        raise LaunchError(
            "failed terminal lifecycle has neither metrics nor fallback evidence",
        )
    elif fallback is None and (
        len(metric_terminal_events) != 1 or metric_terminal_events[0].get("event") != "run_failed"
    ):
        raise LaunchError(
            "failed terminal lifecycle lacks one metrics run_failed event",
        )
    if event_id is not None and metrics is not None and fallback is None:
        matching_watchdog = [
            item
            for item in metric_terminal_events
            if item.get("event") == "run_failed"
            and item.get("schema_version") == WATCHDOG_EVENT_SCHEMA
            and item.get("source") == "persistent-native-exit-watchdog"
            and item.get("watchdog_event_id") == event_id
            and item.get("launch_id") == manifest.get("launch_id")
            and item.get("launch_contract_sha256") == manifest["launch_contract"]["sha256"]
            and item.get("returncode") == returncode
        ]
        if len(matching_watchdog) != 1:
            raise LaunchError(
                "terminal watchdog metrics event differs from manifest",
            )


def _validate_manifest(paths: LaunchPaths, manifest_path: Path) -> dict[str, Any]:
    manifest_path = manifest_path.resolve(strict=False)
    if not _is_within(manifest_path, paths.manifest_dir):
        raise LaunchError("supervised manifest escapes the manifest directory")
    manifest = _load_json_object(manifest_path, label="supervised launch manifest")
    if manifest.get("schema_version") != MANIFEST_SCHEMA:
        raise LaunchError("supervised manifest schema changed")
    if manifest.get("run_name") != RUN_NAME:
        raise LaunchError("supervised manifest names another lineage")
    launch_id = manifest.get("launch_id")
    launch_id = _canonical_uuid_text(
        launch_id,
        label="supervised manifest launch ID",
    )
    if manifest.get("manifest_path") != str(manifest_path):
        raise LaunchError(
            "supervised manifest path differs from its authoritative file",
        )
    expected_supervisor_command = _supervisor_command(
        paths,
        manifest_path,
    )
    raw_supervisor_command = manifest.get("supervisor_command")
    if (
        not isinstance(raw_supervisor_command, list)
        or not all(isinstance(item, str) and item for item in raw_supervisor_command)
        or tuple(raw_supervisor_command) != expected_supervisor_command
    ):
        raise LaunchError("supervisor command changed")
    # Live shadow evidence is re-proven before the trainer is spawned.  Status,
    # successor binding and native-exit reconciliation must not depend on
    # mutable checkout/report files after launch.
    _validate_preflight_payload(
        manifest,
        paths=paths,
        verify_live_shadow=False,
    )
    _validate_selection_proof(manifest.get("selection"), paths=paths)
    launch_contract = _validate_launch_contract_pin(
        manifest.get("launch_contract"),
        paths=paths,
        launch_id=launch_id,
    )
    embedded_launch_contract = _validate_embedded_launch_contract(
        manifest.get("launch_contract_payload"),
        pin_value=launch_contract,
        paths=paths,
        preflight=manifest,
        launch_id=launch_id,
    )
    if manifest.get("created_at_utc") != embedded_launch_contract.get("created_at_utc") or manifest.get(
        "created_unix_s"
    ) != embedded_launch_contract.get("created_unix_s"):
        raise LaunchError(
            "supervised manifest creation identity differs from its immutable launch contract",
        )
    supervised_command = manifest.get("supervised_trainer_command")
    base_command = manifest.get("trainer_command")
    if (
        not isinstance(supervised_command, list)
        or not all(isinstance(item, str) and item for item in supervised_command)
        or not isinstance(base_command, list)
        or supervised_command
        != list(
            _supervised_trainer_command(
                base_command,
                launch_contract=launch_contract,
            )
        )
    ):
        raise LaunchError("supervised trainer command changed")
    supervisor_identity = _optional_identity(
        manifest.get("supervisor_process_identity"),
        label="supervisor",
    )
    trainer_identity = _optional_identity(
        manifest.get("trainer_process_identity"),
        label="trainer",
    )
    bootstrap_identity = _optional_identity(
        manifest.get("trainer_bootstrap_process_identity"),
        label="trainer bootstrap",
    )
    if supervisor_identity is not None and not _identity_matches_command(
        supervisor_identity,
        expected_supervisor_command,
    ):
        raise LaunchError(
            "supervisor process identity does not match its authorized command",
        )
    if trainer_identity is not None and not _identity_matches_command(
        trainer_identity,
        supervised_command,
    ):
        raise LaunchError(
            "trainer process identity does not match its authorized command",
        )
    bootstrap_command: tuple[str, ...] | None = None
    if manifest.get("trainer_spawn_intent") is True:
        if supervisor_identity is None:
            raise LaunchError("trainer bootstrap intent has no supervisor identity")
        raw_nonce_sha256 = manifest.get("trainer_bootstrap_nonce_sha256")
        if not isinstance(raw_nonce_sha256, str):
            raise LaunchError("trainer bootstrap nonce authority is malformed")
        bootstrap_command = _validate_trainer_bootstrap_command(
            manifest.get("trainer_bootstrap_command"),
            paths=paths,
            supervisor_identity=supervisor_identity,
            trainer_command=supervised_command,
            nonce_sha256=raw_nonce_sha256,
        )
    if bootstrap_identity is not None and (
        bootstrap_command is None or not _identity_matches_command(bootstrap_identity, bootstrap_command)
    ):
        raise LaunchError(
            "trainer bootstrap identity does not match its authorized command",
        )
    raw_supervisor_pid = manifest.get("supervisor_spawned_pid")
    if raw_supervisor_pid is not None and (
        isinstance(raw_supervisor_pid, bool) or not isinstance(raw_supervisor_pid, int) or raw_supervisor_pid <= 0
    ):
        raise LaunchError("supervisor spawned PID is malformed")
    if (
        supervisor_identity is not None
        and raw_supervisor_pid is not None
        and supervisor_identity.pid != raw_supervisor_pid
    ):
        raise LaunchError(
            "supervisor process identity differs from its spawned PID",
        )
    raw_trainer_pid = manifest.get("trainer_spawned_pid")
    if bootstrap_identity is not None and raw_trainer_pid is not None and bootstrap_identity.pid != raw_trainer_pid:
        raise LaunchError(
            "trainer bootstrap identity differs from its spawned PID",
        )
    if trainer_identity is not None and raw_trainer_pid is not None and trainer_identity.pid != raw_trainer_pid:
        raise LaunchError(
            "trainer process identity differs from its spawned PID",
        )
    if (
        trainer_identity is not None
        and bootstrap_identity is not None
        and (
            trainer_identity.pid != bootstrap_identity.pid
            or trainer_identity.proc_start_ticks != bootstrap_identity.proc_start_ticks
        )
    ):
        raise LaunchError(
            "bootstrap-to-trainer exec did not preserve PID/start ticks",
        )
    existing = manifest.get("preexisting_run_directories")
    if not isinstance(existing, list) or not all(isinstance(item, str) for item in existing):
        raise LaunchError("supervised manifest has no run-directory snapshot")
    root = _run_root(paths)
    for item in existing:
        candidate = Path(item).resolve(strict=False)
        if not _is_within(candidate, root) or not candidate.name.startswith("run-"):
            raise LaunchError("run-directory snapshot escapes the v29 run root")
    log_path = Path(str(manifest.get("log_path") or "")).resolve(strict=False)
    if not _is_within(log_path, paths.launcher_dir / "logs"):
        raise LaunchError("supervised log path escapes the launcher log root")
    _validate_manifest_lifecycle(manifest, paths=paths)
    return manifest


def _snapshot_run_directories(paths: LaunchPaths) -> list[str]:
    root = _run_root(paths)
    if not root.is_dir():
        return []
    return sorted(str(path.resolve(strict=False)) for path in root.glob("run-*") if path.is_dir())


def _metrics_events(metrics_path: Path) -> list[dict[str, Any]]:
    events: list[dict[str, Any]] = []
    try:
        with metrics_path.open("rb") as handle:
            for line_number, raw in enumerate(handle, start=1):
                if len(raw) > 8 * 1024 * 1024:
                    raise LaunchError(f"metrics line {line_number} exceeds size limit")
                try:
                    payload = json.loads(raw)
                except json.JSONDecodeError:
                    if handle.read(1):
                        raise LaunchError(f"metrics has invalid JSON before EOF at line {line_number}") from None
                    break
                if isinstance(payload, dict):
                    events.append(payload)
    except FileNotFoundError:
        return []
    return events


def _matching_model_init_run_start(
    metrics_path: Path,
    paths: LaunchPaths,
    *,
    manifest: Mapping[str, Any],
) -> dict[str, Any] | None:
    launch_id = _canonical_uuid_text(
        manifest.get("launch_id"),
        label="successor authority launch ID",
    )
    pin = _validate_launch_contract_pin(
        manifest.get("launch_contract"),
        paths=paths,
        launch_id=launch_id,
    )
    contract_payload = _validate_embedded_launch_contract(
        manifest.get("launch_contract_payload"),
        pin_value=pin,
        paths=paths,
        preflight=manifest,
        launch_id=launch_id,
    )
    expected_supervised_provenance = {
        "contract_path": pin["path"],
        "contract_sha256": pin["sha256"],
        "trainer_environment_sha256": contract_payload["trainer_environment_sha256"],
        "implementation_commit": contract_payload["implementation_source"]["implementation_commit"],
        "implementation_tree": contract_payload["implementation_source"]["implementation_tree"],
        "worktree_clean": True,
        "runtime_readiness_report_sha256": contract_payload["runtime_readiness_evidence"]["report_sha256"],
        "contract": contract_payload,
    }
    expected_source = contract_payload["source_checkpoint"]
    expected_state = contract_payload["initial_training_state"]
    initialization = manifest.get("initialization")
    if not isinstance(initialization, Mapping) or not isinstance(
        initialization.get("source_training_state"),
        Mapping,
    ):
        raise LaunchError(
            "successor authority has no complete source training state",
        )
    expected_source_state = dict(
        initialization["source_training_state"],
    )
    expected_actor_supervisor = {
        "version": "sts2-actor-supervisor-state-v1",
        "episode_attempts": 0,
        "consecutive_incidents": 0,
        "incident_fingerprints": {},
        "recent_incident_attempts": [],
    }
    events = _metrics_events(metrics_path)
    run_starts = [event for event in events if event.get("event") == "run_start"]
    if len(run_starts) != 1 or not events or events[0] is not run_starts[0]:
        return None
    event = run_starts[0]
    try:
        run_id = _canonical_uuid_text(
            event.get("run_id"),
            label="successor run ID",
        )
    except LaunchError:
        return None
    if metrics_path.resolve(strict=False).parent.name != f"run-{run_id}":
        return None
    unix_s = event.get("unix_s")
    if (
        isinstance(unix_s, bool)
        or not isinstance(unix_s, int | float)
        or not math.isfinite(float(unix_s))
        or float(unix_s) <= 0.0
    ):
        return None
    reviewed_runtime = manifest.get("runtime")
    if not isinstance(reviewed_runtime, Mapping):
        raise LaunchError("successor authority has no reviewed runtime")
    expected_runtime_keys = {
        "backend",
        "training_revival",
        "simulator_identity",
        "simulator_identity_audit_path",
        "runtime_mechanics",
        "runtime_mechanics_audit_path",
        "sdpa_backend",
        "supervised_launch",
    }
    expected_sdpa_previous = {
        "recording_status": "no_parent_checkpoint",
        "requested_policy": None,
        "effective_backend": None,
    }
    expected_sdpa_flags = {
        "flash": False,
        "memory_efficient": False,
        "math": True,
        "cudnn": False,
    }
    expected_sdpa_keys = {
        "version",
        "checkpoint_load_mode",
        "previous",
        "previous_label",
        "current",
        "current_label",
        "changed",
        "reason",
    }
    expected_sdpa_current_keys = {
        "version",
        "requested_policy",
        "devices",
        "hip_version",
        "applicability",
        "applied",
        "previous_flags",
        "current_flags",
        "effective_backend",
    }
    runtime_provenance = event.get("runtime_provenance")
    if (
        not isinstance(runtime_provenance, Mapping)
        or set(runtime_provenance) != expected_runtime_keys
        or runtime_provenance.get("backend") != "headless"
        or runtime_provenance.get("training_revival") != reviewed_runtime.get("training_revival")
        or runtime_provenance.get("simulator_identity") != reviewed_runtime.get("simulator")
        or runtime_provenance.get("runtime_mechanics") != reviewed_runtime.get("runtime_mechanics")
        or runtime_provenance.get("supervised_launch") != expected_supervised_provenance
    ):
        return None
    audit_roots = {
        "simulator_identity_audit_path": (paths.artifact_root / "logs" / "simulator-preflight"),
        "runtime_mechanics_audit_path": (paths.artifact_root / "logs" / "runtime-mechanics-preflight"),
    }
    if any(
        not isinstance(runtime_provenance.get(key), str)
        or not _is_within(
            Path(str(runtime_provenance[key])).resolve(strict=False),
            root.resolve(strict=False),
        )
        for key, root in audit_roots.items()
    ):
        return None
    sdpa = runtime_provenance.get("sdpa_backend")
    if (
        not isinstance(sdpa, Mapping)
        or set(sdpa) != expected_sdpa_keys
        or sdpa.get("version") != "sts2-rocm-sdpa-transition-v1"
        or sdpa.get("checkpoint_load_mode") != "model_initialization"
        or sdpa.get("previous") != expected_sdpa_previous
        or sdpa.get("previous_label") != "no_parent_checkpoint"
        or sdpa.get("current_label") != "math_only"
        or sdpa.get("changed") is not False
        or sdpa.get("reason") != "configured execution policy at process start"
    ):
        return None
    sdpa_current = sdpa.get("current")
    if (
        not isinstance(sdpa_current, Mapping)
        or set(sdpa_current) != expected_sdpa_current_keys
        or sdpa_current.get("version") != "sts2-rocm-sdpa-execution-v1"
        or sdpa_current.get("requested_policy") != "math"
        or sdpa_current.get("devices") != ["cuda", "cpu"]
        or not isinstance(sdpa_current.get("hip_version"), str)
        or not sdpa_current["hip_version"]
        or sdpa_current.get("applicability") != "rocm_cuda"
        or sdpa_current.get("applied") is not True
        or sdpa_current.get("current_flags") != expected_sdpa_flags
        or sdpa_current.get("effective_backend") != "math_only"
        or not isinstance(sdpa_current.get("previous_flags"), Mapping)
        or set(sdpa_current["previous_flags"]) != set(expected_sdpa_flags)
        or any(not isinstance(value, bool) for value in sdpa_current["previous_flags"].values())
    ):
        return None
    load = event.get("checkpoint_load")
    state = event.get("state")
    config = event.get("config")
    if (
        not isinstance(load, Mapping)
        or set(load)
        != {
            "mode",
            "parent_checkpoint",
            "source_training_state",
            "source_checkpoint",
            "network_parameters_initialized",
            "optimizer_rollouts_rng_and_counters_reset",
        }
        or not isinstance(state, Mapping)
        or not isinstance(config, Mapping)
    ):
        return None
    parent = load.get("parent_checkpoint")
    if not isinstance(parent, str) or not Path(parent).is_absolute():
        return None
    source_state = load.get("source_training_state")
    transaction = config.get("transaction_learning")
    failure_credit = config.get("failure_credit")
    runtime = config.get("runtime")
    if (
        not isinstance(source_state, Mapping)
        or dict(source_state) != expected_source_state
        or not isinstance(transaction, Mapping)
        or not isinstance(failure_credit, Mapping)
        or not isinstance(runtime, Mapping)
    ):
        return None
    try:
        observed_config_sha256 = _canonical_json_sha256(
            dict(config),
        )
    except (TypeError, ValueError):
        return None
    if (
        observed_config_sha256 != FIXED_RUNTIME_CONFIG_FINGERPRINT_SHA256
        or observed_config_sha256 != contract_payload["effective_config_fingerprint_sha256"]
        or event.get("config_fingerprint_sha256") != observed_config_sha256
    ):
        return None
    if (
        load.get("mode") == "model_initialization"
        and Path(parent).resolve(strict=False) == paths.initialization_checkpoint
        and load.get("source_checkpoint") == expected_source
        and source_state.get("environment_steps") == FIXED_INITIALIZATION_STEP
        and source_state.get("policy_version") == FIXED_INITIALIZATION_POLICY_VERSION
        and load.get("network_parameters_initialized") is True
        and load.get("optimizer_rollouts_rng_and_counters_reset") is True
        and dict(state) == expected_state
        and event.get("actor_supervisor_state") == expected_actor_supervisor
        and event.get("pipeline") == TRAINING_PIPELINE_V7
        and config.get("version") == "sts2-relational-curriculum-config-v12"
        and transaction.get("enabled") is False
        and failure_credit.get("mode") == "learning"
        and runtime.get("seed") == 4_000_000
        and runtime.get("total_environment_steps") == 100_000
    ):
        return event
    return None


def discover_successor_metrics(
    paths: LaunchPaths,
    *,
    manifest: Mapping[str, Any],
) -> tuple[Path, dict[str, Any]] | None:
    root = _run_root(paths)
    if not root.is_dir():
        return None
    preexisting_run_directories = manifest.get(
        "preexisting_run_directories",
    )
    if not isinstance(preexisting_run_directories, list) or not all(
        isinstance(item, str) for item in preexisting_run_directories
    ):
        raise LaunchError("manifest run snapshot is malformed")
    previous = {Path(item).resolve(strict=False) for item in preexisting_run_directories}
    matches: list[tuple[Path, dict[str, Any]]] = []
    for run_dir in root.glob("run-*"):
        resolved = run_dir.resolve(strict=False)
        if resolved in previous or not run_dir.is_dir():
            continue
        metrics = (run_dir / "metrics.jsonl").resolve(strict=False)
        if not _is_within(metrics, root) or not metrics.is_file():
            continue
        run_start = _matching_model_init_run_start(
            metrics,
            paths,
            manifest=manifest,
        )
        if run_start is not None:
            matches.append((metrics, run_start))
    if len(matches) > 1:
        raise LaunchError("multiple new runs claim the fixed model-initialization checkpoint")
    return matches[0] if matches else None


def _validate_new_successor_metrics_path(
    paths: LaunchPaths,
    *,
    manifest: Mapping[str, Any],
    metrics_path: Path,
) -> Path:
    resolved = metrics_path.resolve(strict=False)
    root = _run_root(paths)
    run_directory_match = re.fullmatch(
        r"run-([0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-" r"[0-9a-f]{4}-[0-9a-f]{12})",
        resolved.parent.name,
    )
    if (
        resolved.name != "metrics.jsonl"
        or resolved.parent.parent != root
        or run_directory_match is None
        or not _is_within(resolved, root)
    ):
        raise LaunchError(
            "successor metrics path is not a direct v29 run metrics file",
        )
    _canonical_uuid_text(
        run_directory_match.group(1),
        label="successor run-directory ID",
    )
    existing = manifest.get("preexisting_run_directories")
    if not isinstance(existing, list) or not all(isinstance(item, str) for item in existing):
        raise LaunchError("manifest run snapshot is malformed")
    if resolved.parent in {Path(item).resolve(strict=False) for item in existing}:
        raise LaunchError("successor metrics belongs to a preexisting run")
    if not resolved.is_file():
        raise LaunchError("successor metrics file does not exist")
    return resolved


def _successor_binding_payload(
    *,
    launch_id: str,
    launch_contract_sha256: str,
    run_start: Mapping[str, Any],
) -> dict[str, Any]:
    state = run_start.get("state")
    return {
        "run_id": run_start.get("run_id"),
        "unix_s": run_start.get("unix_s"),
        "supervised_launch_id": launch_id,
        "launch_contract_sha256": launch_contract_sha256,
        "initial_environment_steps": (state.get("environment_steps") if isinstance(state, Mapping) else None),
        "checkpoint_load": run_start.get("checkpoint_load"),
    }


def _record_successor_binding(
    paths: LaunchPaths,
    *,
    manifest_path: Path,
    metrics_path: Path,
    run_start: Mapping[str, Any],
) -> dict[str, Any]:
    launch_id = _canonical_uuid_text(
        _load_json_object(
            manifest_path,
            label="manifest",
        ).get("launch_id"),
        label="successor manifest launch ID",
    )
    with _terminal_lock(paths, launch_id):
        manifest = _validate_manifest(paths, manifest_path)
        metrics_path = _validate_new_successor_metrics_path(
            paths,
            manifest=manifest,
            metrics_path=metrics_path,
        )
        matched = _matching_model_init_run_start(
            metrics_path,
            paths,
            manifest=manifest,
        )
        if matched is None or dict(matched) != dict(run_start):
            raise LaunchError(
                "successor run_start no longer matches immutable launch authority",
            )
        prior = manifest.get("metrics_path")
        if prior is not None and Path(str(prior)).resolve(strict=False) != metrics_path:
            raise LaunchError("successor metrics binding changed")
        launch_contract = manifest.get("launch_contract")
        assert isinstance(launch_contract, Mapping)
        manifest.update(
            {
                "metrics_path": str(metrics_path),
                "successor": _successor_binding_payload(
                    launch_id=launch_id,
                    launch_contract_sha256=str(
                        launch_contract["sha256"],
                    ),
                    run_start=run_start,
                ),
                "successor_bound_at_utc": _utc_now(),
            }
        )
        _persist_manifest(paths, manifest_path=manifest_path, manifest=manifest)
        return manifest


def _validate_bound_successor_metrics(
    paths: LaunchPaths,
    *,
    manifest: Mapping[str, Any],
    metrics_path: Path,
) -> tuple[Path, dict[str, Any]]:
    resolved = _validate_new_successor_metrics_path(
        paths,
        manifest=manifest,
        metrics_path=metrics_path,
    )
    recorded_path = manifest.get("metrics_path")
    if not isinstance(recorded_path, str) or Path(recorded_path).resolve(strict=False) != resolved:
        raise LaunchError(
            "watchdog metrics path is not the manifest-bound successor",
        )
    matched = _matching_model_init_run_start(
        resolved,
        paths,
        manifest=manifest,
    )
    if matched is None:
        raise LaunchError(
            "manifest-bound successor no longer has its authoritative run_start",
        )
    launch_id = _canonical_uuid_text(
        manifest.get("launch_id"),
        label="bound successor launch ID",
    )
    launch_contract = manifest.get("launch_contract")
    if not isinstance(launch_contract, Mapping):
        raise LaunchError("bound successor has no launch-contract pin")
    expected = _successor_binding_payload(
        launch_id=launch_id,
        launch_contract_sha256=str(launch_contract.get("sha256")),
        run_start=matched,
    )
    if manifest.get("successor") != expected:
        raise LaunchError(
            "manifest successor binding differs from authoritative run_start",
        )
    return resolved, matched


def _discover_from_manifest(paths: LaunchPaths, manifest: Mapping[str, Any]) -> tuple[Path, dict[str, Any]] | None:
    return discover_successor_metrics(paths, manifest=manifest)


def _append_jsonl_event_once(
    metrics_path: Path,
    event: Mapping[str, Any],
    *,
    event_id: str,
) -> bool:
    if any(item.get("watchdog_event_id") == event_id for item in _metrics_events(metrics_path)):
        return False
    serialized = (json.dumps(event, ensure_ascii=False, sort_keys=True, separators=(",", ":")) + "\n").encode("utf-8")
    # A native abort can leave JsonlMetrics halfway through its final object.
    # The trainer is already dead here, so remove only that unterminated EOF
    # suffix before appending.  Prefixing a newline would preserve malformed
    # JSON as an internal line and break strict reconciliation on the next
    # watchdog pass.
    flags = os.O_RDWR | os.O_APPEND
    if hasattr(os, "O_NOFOLLOW"):
        flags |= os.O_NOFOLLOW
    descriptor = os.open(metrics_path, flags)
    try:
        size = os.lseek(descriptor, 0, os.SEEK_END)
        if size:
            os.lseek(descriptor, size - 1, os.SEEK_SET)
            if os.read(descriptor, 1) != b"\n":
                position = size
                last_newline = -1
                while position > 0 and last_newline < 0:
                    chunk_start = max(0, position - 64 * 1024)
                    os.lseek(descriptor, chunk_start, os.SEEK_SET)
                    chunk = os.read(descriptor, position - chunk_start)
                    relative = chunk.rfind(b"\n")
                    if relative >= 0:
                        last_newline = chunk_start + relative
                        break
                    position = chunk_start
                os.ftruncate(descriptor, last_newline + 1)
                os.fsync(descriptor)
        os.lseek(descriptor, 0, os.SEEK_END)
        if os.write(descriptor, serialized) != len(serialized):
            raise LaunchError("watchdog metrics append was incomplete")
        os.fsync(descriptor)
    finally:
        os.close(descriptor)
    return True


def _log_tail(path: Path, *, maximum_bytes: int = 128 * 1024) -> str:
    try:
        with path.open("rb") as handle:
            handle.seek(0, os.SEEK_END)
            size = handle.tell()
            handle.seek(max(0, size - maximum_bytes))
            return handle.read(maximum_bytes).decode("utf-8", errors="replace")
    except OSError:
        return ""


def _exit_classification(
    *,
    returncode: int | None,
    events: Sequence[Mapping[str, Any]],
    log_tail: str,
) -> dict[str, Any]:
    lifecycle = {
        str(event.get("event")) for event in events if event.get("event") in {"run_complete", "interrupt", "run_failed"}
    }
    if (returncode in (0, None)) and "run_complete" in lifecycle:
        return {"status": "completed", "kind": "clean_completion", "native_abort": False, "append_run_failed": False}
    if "interrupt" in lifecycle:
        return {"status": "interrupted", "kind": "runtime_interrupt", "native_abort": False, "append_run_failed": False}
    if "run_failed" in lifecycle:
        return {
            "status": "failed",
            "kind": "runtime_reported_failure",
            "native_abort": False,
            "append_run_failed": False,
        }
    signal_number = -returncode if returncode is not None and returncode < 0 else None
    if signal_number is None and returncode is not None and returncode >= 128:
        candidate = returncode - 128
        if 0 < candidate < signal.NSIG:
            signal_number = candidate
    try:
        signal_name = signal.Signals(signal_number).name if signal_number is not None else None
    except ValueError:
        signal_name = f"SIGNAL_{signal_number}"
    signatures = (
        "AqlQueue::HandleInsufficientScratch",
        "process compute queue fail",
        "Assertion `",
        "Aborted (core dumped)",
    )
    matched = [item for item in signatures if item in log_tail]
    native_abort = signal_number == signal.SIGABRT or bool(matched)
    kind = (
        "native_abort"
        if native_abort
        else "exit_zero_without_run_complete"
        if returncode == 0
        else "unobserved_exit_without_terminal_event"
        if returncode is None
        else "signal_exit"
        if signal_number is not None
        else "nonzero_exit"
    )
    return {
        "status": "failed",
        "kind": kind,
        "native_abort": native_abort,
        "returncode": returncode,
        "signal_number": signal_number,
        "signal_name": signal_name,
        "matched_log_signatures": matched,
        "append_run_failed": True,
    }


def _latest_environment_steps(events: Sequence[Mapping[str, Any]]) -> int:
    latest = 0
    for event in events:
        for value in (
            event.get("environment_steps"),
            event.get("state", {}).get("environment_steps") if isinstance(event.get("state"), Mapping) else None,
        ):
            if isinstance(value, int) and not isinstance(value, bool):
                latest = max(latest, value)
    return latest


def _persist_supervisor_emergency_failure(
    paths: LaunchPaths,
    *,
    manifest_path: Path,
    launch_id: str,
    original_error: BaseException,
    finalization_error: BaseException,
    returncode: int | None,
    supervisor_identity: ProcessIdentity | None,
    trainer_identity: ProcessIdentity | None,
    metrics_path: Path | None,
) -> Path:
    """Persist a last-resort terminal record when normal finalization fails."""

    launch_id = _canonical_uuid_text(
        launch_id,
        label="emergency launch ID",
    )
    try:
        raw_manifest = _load_json_object(
            manifest_path,
            label="emergency supervised launch manifest",
        )
    except LaunchError:
        raw_manifest = {}
    raw_launch_contract = raw_manifest.get("launch_contract")
    launch_contract_sha256 = raw_launch_contract.get("sha256") if isinstance(raw_launch_contract, Mapping) else None
    fallback = paths.launcher_dir / "terminal-events" / f"{launch_id}.supervisor_recovery_failed.json"
    failure = {
        "event": "supervisor_recovery_failed",
        "schema_version": SUPERVISOR_EMERGENCY_SCHEMA,
        "unix_s": time.time(),
        "observed_at_utc": _utc_now(),
        "launch_id": launch_id,
        "run_name": RUN_NAME,
        "launch_contract_sha256": launch_contract_sha256,
        "returncode": returncode,
        "original_error": {
            "type": type(original_error).__name__,
            "message": str(original_error),
        },
        "finalization_error": {
            "type": type(finalization_error).__name__,
            "message": str(finalization_error),
        },
        "supervisor_process_identity": (asdict(supervisor_identity) if supervisor_identity is not None else None),
        "trainer_process_identity": (asdict(trainer_identity) if trainer_identity is not None else None),
        "metrics_path": (str(metrics_path.resolve(strict=False)) if metrics_path is not None else None),
        "manifest_path": str(manifest_path.resolve(strict=False)),
    }
    atomic_write_json(fallback, failure)

    # The emergency JSON is the irreducible durable record.  If the original
    # manifest remains valid, also terminalize it so the dashboard does not
    # report a dead supervisor as an active or unknown run.
    try:
        manifest = _validate_manifest(paths, manifest_path)
        if manifest.get("status") not in TERMINAL_STATUSES:
            terminal = {
                "status": "failed",
                "classification": {
                    "status": "failed",
                    "kind": "supervisor_finalization_failure",
                    "native_abort": False,
                    "returncode": returncode,
                    "signal_number": None,
                    "signal_name": None,
                    "matched_log_signatures": [],
                    "append_run_failed": False,
                },
                "returncode": returncode,
                "observed_at_utc": failure["observed_at_utc"],
                "watchdog_event_id": None,
                "watchdog_event_appended": False,
                "metrics_path": manifest.get("metrics_path"),
                "fallback_event_path": str(fallback),
                "reconciliation_reason": ("normal_supervisor_finalization_failed"),
            }
            manifest.update(
                {
                    "status": "failed",
                    "terminal": terminal,
                    "terminal_event_id": None,
                    "supervisor_process_identity": (failure["supervisor_process_identity"]),
                    "trainer_process_identity": (failure["trainer_process_identity"]),
                }
            )
            _persist_manifest(
                paths,
                manifest_path=manifest_path,
                manifest=manifest,
            )
    except Exception:
        # Never replace or hide the already fsync'd emergency record with a
        # secondary dashboard-state repair failure.
        pass
    return fallback


def finalize_supervised_exit(
    paths: LaunchPaths,
    *,
    manifest_path: Path,
    returncode: int | None,
    supervisor_identity: ProcessIdentity | None,
    trainer_identity: ProcessIdentity | None,
    metrics_path: Path | None,
    reconciliation_reason: str | None = None,
    enforce_active_root: bool = True,
) -> dict[str, Any]:
    paths = validate_layout(paths, enforce_active_root=enforce_active_root)
    manifest = _validate_manifest(paths, manifest_path)
    launch_id = str(manifest["launch_id"])
    with _terminal_lock(paths, launch_id):
        manifest = _validate_manifest(paths, manifest_path)
        if manifest.get("status") in TERMINAL_STATUSES:
            return manifest
        stored_supervisor = _optional_identity(
            manifest.get("supervisor_process_identity"),
            label="supervisor",
        )
        stored_trainer = _optional_identity(
            manifest.get("trainer_process_identity"),
            label="trainer",
        )
        if (
            stored_supervisor is not None
            and supervisor_identity is not None
            and stored_supervisor != supervisor_identity
        ):
            raise LaunchError(
                "finalizer supervisor identity differs from durable authority",
            )
        if stored_trainer is not None and trainer_identity is not None and stored_trainer != trainer_identity:
            raise LaunchError(
                "finalizer trainer identity differs from durable authority",
            )
        supervisor_identity = supervisor_identity or stored_supervisor
        trainer_identity = trainer_identity or stored_trainer
        if supervisor_identity is not None and not _identity_matches_command(
            supervisor_identity,
            _supervisor_command(paths, manifest_path),
        ):
            raise LaunchError(
                "finalizer supervisor identity is outside launch authority",
            )
        supervised_command = manifest.get("supervised_trainer_command")
        assert isinstance(supervised_command, list)
        if trainer_identity is not None and not _identity_matches_command(
            trainer_identity,
            supervised_command,
        ):
            raise LaunchError(
                "finalizer trainer identity is outside launch authority",
            )
        raw_supervisor_pid = manifest.get("supervisor_spawned_pid")
        if (
            supervisor_identity is not None
            and raw_supervisor_pid is not None
            and supervisor_identity.pid != raw_supervisor_pid
        ):
            raise LaunchError(
                "finalizer supervisor identity differs from spawned PID",
            )
        raw_trainer_pid = manifest.get("trainer_spawned_pid")
        if trainer_identity is not None and raw_trainer_pid is not None and trainer_identity.pid != raw_trainer_pid:
            raise LaunchError(
                "finalizer trainer identity differs from spawned PID",
            )
        if metrics_path is not None:
            metrics_path, _ = _validate_bound_successor_metrics(
                paths,
                manifest=manifest,
                metrics_path=metrics_path,
            )
            events = _metrics_events(metrics_path)
        else:
            events = []
        log_path = Path(str(manifest.get("log_path") or ""))
        existing = next(
            (
                event
                for event in reversed(events)
                if event.get("event") == "run_failed"
                and event.get("source") == "persistent-native-exit-watchdog"
                and event.get("launch_id") == launch_id
            ),
            None,
        )
        if existing is not None and isinstance(existing.get("classification"), Mapping):
            classification = dict(existing["classification"])
            classification["status"] = "failed"
            classification["append_run_failed"] = False
        else:
            classification = _exit_classification(
                returncode=returncode,
                events=events,
                log_tail=_log_tail(log_path),
            )
        event_id = (
            str(existing.get("watchdog_event_id"))
            if existing is not None and isinstance(existing.get("watchdog_event_id"), str)
            else None
        )
        appended = False
        fallback_event_path: str | None = None
        if classification["append_run_failed"] is True:
            event_id = _sha256_bytes(f"{launch_id}:{WATCHDOG_EVENT_SCHEMA}:run_failed".encode())
            failure = {
                "event": "run_failed",
                "unix_s": time.time(),
                "schema_version": WATCHDOG_EVENT_SCHEMA,
                "source": "persistent-native-exit-watchdog",
                "watchdog_event_id": event_id,
                "launch_id": launch_id,
                "run_name": RUN_NAME,
                "launch_contract_sha256": manifest["launch_contract"]["sha256"],
                "environment_steps": _latest_environment_steps(events),
                "initialization_mode": "model_initialization",
                "initialization_checkpoint": str(paths.initialization_checkpoint),
                "initialization_checkpoint_id": FIXED_INITIALIZATION_CHECKPOINT_ID,
                "source_environment_steps": FIXED_INITIALIZATION_STEP,
                "returncode": returncode,
                "classification": classification,
                "reason": classification["kind"],
                "reconciliation_reason": reconciliation_reason,
                "supervisor_process_identity": asdict(supervisor_identity) if supervisor_identity else None,
                "trainer_process_identity": asdict(trainer_identity) if trainer_identity else None,
                "log_path": str(log_path),
            }
            existing_terminal_events = [
                event for event in events if event.get("event") in {"run_complete", "interrupt", "run_failed"}
            ]
            if metrics_path is not None and not existing_terminal_events:
                appended = _append_jsonl_event_once(metrics_path, failure, event_id=event_id)
            else:
                fallback = paths.launcher_dir / "terminal-events" / f"{launch_id}.run_failed.json"
                atomic_write_json(fallback, failure)
                fallback_event_path = str(fallback)
        terminal = {
            "status": classification["status"],
            "classification": classification,
            "returncode": returncode,
            "observed_at_utc": _utc_now(),
            "watchdog_event_id": event_id,
            "watchdog_event_appended": appended,
            "metrics_path": str(metrics_path) if metrics_path is not None else None,
            "fallback_event_path": fallback_event_path,
            "reconciliation_reason": reconciliation_reason,
        }
        manifest.update(
            {
                "status": classification["status"],
                "terminal": terminal,
                "supervisor_process_identity": asdict(supervisor_identity) if supervisor_identity else None,
                "trainer_process_identity": asdict(trainer_identity) if trainer_identity else None,
                "metrics_path": terminal["metrics_path"],
                "terminal_event_id": event_id,
            }
        )
        _persist_manifest(paths, manifest_path=manifest_path, manifest=manifest)
        return manifest


def _recover_spawned_process_identity(
    paths: LaunchPaths,
    *,
    manifest_path: Path,
    manifest: Mapping[str, Any],
    role: str,
) -> tuple[ProcessIdentity | None, str | None]:
    """Recover the durable identity in a Popen-to-capture crash window.

    A bare PID is never accepted as liveness authority: it is converted to the
    full `/proc` identity only when its executable and NUL-delimited command
    line exactly match the immutable manifest command.  Any uncertainty is
    returned as an ambiguity instead of being reconciled as a dead run.
    """

    if role not in {"supervisor", "bootstrap", "trainer"}:
        raise LaunchError("spawned process recovery role is unsupported")
    identity_field = "trainer_bootstrap_process_identity" if role == "bootstrap" else f"{role}_process_identity"
    pid_field = "trainer_spawned_pid" if role in {"bootstrap", "trainer"} else "supervisor_spawned_pid"
    command_field = {
        "supervisor": "supervisor_command",
        "bootstrap": "trainer_bootstrap_command",
        "trainer": "supervised_trainer_command",
    }[role]
    expected_command = tuple(str(item) for item in manifest[command_field])
    existing = manifest.get(identity_field)
    if existing is not None:
        return (
            _identity_from_mapping(existing, label=role),
            None,
        )
    raw_pid = manifest.get(pid_field)
    if isinstance(raw_pid, bool) or not isinstance(raw_pid, int) or raw_pid <= 0:
        return None, f"{role}_spawned_pid_missing_or_malformed"
    try:
        recovered = capture_process_identity(raw_pid)
    except LaunchError as exc:
        return None, f"{role}_identity_unreadable:{exc}"
    if not _identity_matches_command(recovered, expected_command):
        return None, f"{role}_pid_command_identity_mismatch"
    with _terminal_lock(paths, str(manifest["launch_id"])):
        current = _validate_manifest(paths, manifest_path)
        durable = current.get(identity_field)
        if durable is not None:
            return (
                _identity_from_mapping(durable, label=role),
                None,
            )
        if current.get(pid_field) != raw_pid or current.get(
            command_field,
        ) != list(expected_command):
            return None, f"{role}_spawn_authority_changed"
        current[identity_field] = asdict(recovered)
        current[f"{role}_identity_recovered_at_utc"] = _utc_now()
        _persist_manifest(
            paths,
            manifest_path=manifest_path,
            manifest=current,
        )
    return recovered, None


def _record_trainer_spawn_intent(
    paths: LaunchPaths,
    *,
    manifest_path: Path,
    bootstrap_command: Sequence[str],
    nonce_sha256: str,
) -> dict[str, Any]:
    if (
        not bootstrap_command
        or not all(isinstance(item, str) and item for item in bootstrap_command)
        or re.fullmatch(r"[0-9a-f]{64}", nonce_sha256) is None
    ):
        raise LaunchError("trainer bootstrap spawn intent is malformed")
    manifest = _validate_manifest(paths, manifest_path)
    launch_id = str(manifest["launch_id"])
    with _terminal_lock(paths, launch_id):
        manifest = _validate_manifest(paths, manifest_path)
        if manifest.get("status") in TERMINAL_STATUSES:
            raise LaunchError(
                "cannot persist trainer spawn intent after terminalization",
            )
        if manifest.get("trainer_spawn_intent") is True:
            raise LaunchError(
                "trainer spawn intent already exists; refusing a duplicate spawn",
            )
        supervisor_identity = _identity_from_mapping(
            manifest.get("supervisor_process_identity"),
            label="supervisor",
        )
        trainer_command = manifest.get("supervised_trainer_command")
        if not isinstance(trainer_command, list):
            raise LaunchError("trainer spawn intent lacks a final command")
        _validate_trainer_bootstrap_command(
            list(bootstrap_command),
            paths=paths,
            supervisor_identity=supervisor_identity,
            trainer_command=trainer_command,
            nonce_sha256=nonce_sha256,
        )
        manifest.update(
            {
                "trainer_spawn_intent": True,
                "trainer_spawn_intent_at_utc": _utc_now(),
                "trainer_bootstrap_phase": "intent",
                "trainer_bootstrap_command": list(bootstrap_command),
                "trainer_bootstrap_nonce_sha256": nonce_sha256,
            }
        )
        _persist_manifest(
            paths,
            manifest_path=manifest_path,
            manifest=manifest,
        )
    return manifest


def _record_trainer_bootstrap_ready(
    paths: LaunchPaths,
    *,
    manifest_path: Path,
    identity: ProcessIdentity,
) -> dict[str, Any]:
    manifest = _validate_manifest(paths, manifest_path)
    launch_id = str(manifest["launch_id"])
    with _terminal_lock(paths, launch_id):
        manifest = _validate_manifest(paths, manifest_path)
        if (
            manifest.get("status") in TERMINAL_STATUSES
            or manifest.get("trainer_spawn_intent") is not True
            or manifest.get("trainer_spawn_failed") is not False
            or manifest.get("trainer_exit_observed") is not False
            or manifest.get("trainer_bootstrap_phase") != "intent"
        ):
            raise LaunchError(
                "trainer bootstrap-ready authority cannot advance from its current lifecycle",
            )
        command = manifest.get("trainer_bootstrap_command")
        if not isinstance(command, list) or not _identity_matches_command(
            identity,
            command,
        ):
            raise LaunchError("ready bootstrap identity differs from its command")
        manifest.update(
            {
                "trainer_bootstrap_phase": "bootstrap_ready",
                "trainer_bootstrap_ready_at_utc": _utc_now(),
                "trainer_spawned_pid": identity.pid,
                "trainer_bootstrap_process_identity": asdict(identity),
            }
        )
        _persist_manifest(
            paths,
            manifest_path=manifest_path,
            manifest=manifest,
        )
    return manifest


def _record_trainer_exec_authorized(
    paths: LaunchPaths,
    *,
    manifest_path: Path,
    identity: ProcessIdentity,
) -> dict[str, Any]:
    manifest = _validate_manifest(paths, manifest_path)
    launch_id = str(manifest["launch_id"])
    with _terminal_lock(paths, launch_id):
        manifest = _validate_manifest(paths, manifest_path)
        if (
            manifest.get("status") in TERMINAL_STATUSES
            or manifest.get("trainer_bootstrap_phase") != "bootstrap_ready"
            or manifest.get("trainer_bootstrap_process_identity") != asdict(identity)
            or manifest.get("trainer_spawned_pid") != identity.pid
        ):
            raise LaunchError("trainer exec authorization lacks exact bootstrap authority")
        manifest.update(
            {
                "trainer_bootstrap_phase": "exec_authorized",
                "trainer_bootstrap_exec_authorized_at_utc": _utc_now(),
            }
        )
        _persist_manifest(paths, manifest_path=manifest_path, manifest=manifest)
    return manifest


def _record_trainer_exec_observed(
    paths: LaunchPaths,
    *,
    manifest_path: Path,
    bootstrap_identity: ProcessIdentity,
    trainer_identity: ProcessIdentity,
) -> dict[str, Any]:
    if (
        bootstrap_identity.pid != trainer_identity.pid
        or bootstrap_identity.proc_start_ticks != trainer_identity.proc_start_ticks
    ):
        raise LaunchError("bootstrap-to-trainer exec changed PID/start ticks")
    manifest = _validate_manifest(paths, manifest_path)
    launch_id = str(manifest["launch_id"])
    with _terminal_lock(paths, launch_id):
        manifest = _validate_manifest(paths, manifest_path)
        command = manifest.get("supervised_trainer_command")
        if (
            manifest.get("status") in TERMINAL_STATUSES
            or manifest.get("trainer_bootstrap_phase") != "exec_authorized"
            or manifest.get("trainer_bootstrap_process_identity") != asdict(bootstrap_identity)
            or not isinstance(command, list)
            or not _identity_matches_command(trainer_identity, command)
        ):
            raise LaunchError("observed trainer exec lacks exact authorization")
        now = _utc_now()
        manifest.update(
            {
                "status": "running",
                "training_started": True,
                "trainer_bootstrap_phase": "exec_observed",
                "trainer_bootstrap_exec_observed_at_utc": now,
                "trainer_spawned_at_utc": now,
                "trainer_started_at_utc": now,
                "trainer_spawned_pid": trainer_identity.pid,
                "trainer_process_identity": asdict(trainer_identity),
            }
        )
        _persist_manifest(paths, manifest_path=manifest_path, manifest=manifest)
    return manifest


def _record_trainer_spawn_failed(
    paths: LaunchPaths,
    *,
    manifest_path: Path,
) -> dict[str, Any]:
    manifest = _validate_manifest(paths, manifest_path)
    launch_id = str(manifest["launch_id"])
    with _terminal_lock(paths, launch_id):
        manifest = _validate_manifest(paths, manifest_path)
        if manifest.get("trainer_spawn_failed") is True:
            return manifest
        if (
            manifest.get("trainer_spawn_intent") is not True
            or manifest.get("training_started") is not False
            or manifest.get("trainer_exit_observed") is not False
            or manifest.get("trainer_bootstrap_phase") not in {"intent", "bootstrap_ready", "exec_authorized"}
        ):
            raise LaunchError(
                "trainer spawn failure conflicts with durable child authority",
            )
        manifest.update(
            {
                "trainer_spawn_failed": True,
                "trainer_spawn_failure_at_utc": _utc_now(),
                "trainer_bootstrap_phase": "preexec_failed",
            }
        )
        _persist_manifest(
            paths,
            manifest_path=manifest_path,
            manifest=manifest,
        )
    return manifest


def _record_trainer_exit_observed(
    paths: LaunchPaths,
    *,
    manifest_path: Path,
    pid: int,
    returncode: int,
) -> dict[str, Any]:
    if (
        isinstance(pid, bool)
        or not isinstance(pid, int)
        or pid <= 0
        or isinstance(returncode, bool)
        or not isinstance(returncode, int)
    ):
        raise LaunchError("reaped trainer process evidence is malformed")
    manifest = _validate_manifest(paths, manifest_path)
    launch_id = str(manifest["launch_id"])
    with _terminal_lock(paths, launch_id):
        manifest = _validate_manifest(paths, manifest_path)
        if manifest.get("trainer_exit_observed") is True:
            if manifest.get("trainer_spawned_pid") != pid or manifest.get("trainer_reaped_returncode") != returncode:
                raise LaunchError("reaped trainer evidence changed")
            return manifest
        if (
            manifest.get("trainer_spawn_intent") is not True
            or manifest.get("trainer_spawn_failed") is not False
            or manifest.get("trainer_bootstrap_phase") != "exec_observed"
            or manifest.get("training_started") is not True
        ):
            raise LaunchError(
                "reaped trainer evidence has no compatible spawn intent",
            )
        existing_pid = manifest.get("trainer_spawned_pid")
        if existing_pid != pid:
            raise LaunchError("reaped trainer PID differs from spawn authority")
        manifest.update(
            {
                "trainer_bootstrap_phase": "reaped",
                "trainer_exit_observed": True,
                "trainer_exit_observed_at_utc": _utc_now(),
                "trainer_reaped_returncode": returncode,
            }
        )
        _persist_manifest(
            paths,
            manifest_path=manifest_path,
            manifest=manifest,
        )
    return manifest


def _observe_authorized_bootstrap_handoff(
    paths: LaunchPaths,
    *,
    manifest_path: Path,
    manifest: Mapping[str, Any],
) -> tuple[ProcessIdentity | None, ProcessIdentity | None, str | None]:
    """Observe only the two commands authorized for one bootstrap PID.

    During ``exec_authorized`` the same Linux PID/start marker may still show
    the bootstrap command or may already show the final trainer command.  No
    third command, PID reuse or start-marker change is accepted.
    """

    phase = manifest.get("trainer_bootstrap_phase")
    if phase not in {"bootstrap_ready", "exec_authorized"}:
        return None, None, None
    bootstrap = _identity_from_mapping(
        manifest.get("trainer_bootstrap_process_identity"),
        label="trainer bootstrap",
    )
    try:
        observed = capture_process_identity(bootstrap.pid)
    except LaunchError:
        return None, None, None
    if observed == bootstrap:
        return bootstrap, None, None
    if phase != "exec_authorized":
        return None, None, "bootstrap_command_changed_before_exec_authorization"
    command = manifest.get("supervised_trainer_command")
    if (
        not isinstance(command, list)
        or observed.pid != bootstrap.pid
        or observed.proc_start_ticks != bootstrap.proc_start_ticks
        or not _identity_matches_command(observed, command)
    ):
        return None, None, "bootstrap_exec_identity_mismatch"
    _record_trainer_exec_observed(
        paths,
        manifest_path=manifest_path,
        bootstrap_identity=bootstrap,
        trainer_identity=observed,
    )
    return None, observed, None


def _supervised_status(
    paths: LaunchPaths,
    *,
    state: Mapping[str, Any],
    manifest_path: Path,
    allow_reconciliation: bool,
    enforce_active_root: bool,
) -> dict[str, Any]:
    manifest = _validate_manifest(paths, manifest_path)
    if manifest.get("launch_id") != state.get("launch_id"):
        raise LaunchError("launcher state and manifest launch IDs disagree")
    manifest_status = str(manifest.get("status") or "invalid")
    bootstrap_phase = str(manifest.get("trainer_bootstrap_phase") or "invalid")
    supervisor = _optional_identity(manifest.get("supervisor_process_identity"), label="supervisor")
    bootstrap = _optional_identity(
        manifest.get("trainer_bootstrap_process_identity"),
        label="trainer bootstrap",
    )
    trainer = _optional_identity(manifest.get("trainer_process_identity"), label="trainer")
    ambiguity: str | None = None
    written = state.get("written_unix_s")
    supervisor_launch_grace = (
        manifest_status == "supervisor-launching"
        and manifest.get("supervisor_spawned_pid") is None
        and isinstance(written, int | float)
        and not isinstance(written, bool)
        and time.time() - float(written) < 30.0
    )
    if supervisor is None and manifest_status not in TERMINAL_STATUSES and not supervisor_launch_grace:
        supervisor, ambiguity = _recover_spawned_process_identity(
            paths,
            manifest_path=manifest_path,
            manifest=manifest,
            role="supervisor",
        )
    bootstrap_live_identity: ProcessIdentity | None = None
    if ambiguity is None and bootstrap_phase in {"bootstrap_ready", "exec_authorized"}:
        bootstrap_live_identity, observed_trainer, ambiguity = _observe_authorized_bootstrap_handoff(
            paths,
            manifest_path=manifest_path,
            manifest=manifest,
        )
        if observed_trainer is not None:
            trainer = observed_trainer
            bootstrap_phase = "exec_observed"
    if supervisor is not None or bootstrap is not None or trainer is not None:
        manifest = _validate_manifest(paths, manifest_path)
    supervisor_live = process_identity_matches(supervisor) if supervisor else False
    bootstrap_live = bootstrap_live_identity is not None
    trainer_live = process_identity_matches(trainer) if trainer else False
    if ambiguity is not None:
        role = (
            "bootstrap"
            if ambiguity.startswith("bootstrap_")
            else "trainer"
            if ambiguity.startswith("trainer_")
            else "supervisor"
        )
        status, running = f"ambiguous-untracked-{role}", True
    elif manifest_status in TERMINAL_STATUSES and (supervisor_live or bootstrap_live or trainer_live):
        status, running = "terminal-process-still-live", True
    elif manifest_status in TERMINAL_STATUSES:
        status, running = manifest_status, False
    elif supervisor_live and (bootstrap_live or trainer_live):
        status, running = "running", True
    elif supervisor_live and not bootstrap_live and trainer is None:
        status, running = "supervisor-running", True
    elif supervisor_live:
        status, running = "supervisor-finalizing", True
    elif bootstrap_live:
        status, running = "unwatched-bootstrap", True
    elif trainer_live:
        status, running = "unwatched-running", True
    else:
        recent = (
            isinstance(written, int | float)
            and not isinstance(written, bool)
            and time.time() - float(written) < 30.0
            and supervisor is None
            and bootstrap is None
            and trainer is None
        )
        if recent:
            status, running = "supervisor-launching", True
        elif allow_reconciliation:
            metrics: Path | None = None
            raw_metrics = manifest.get("metrics_path")
            if isinstance(raw_metrics, str) and raw_metrics:
                metrics = Path(raw_metrics).resolve(strict=False)
            else:
                discovered = _discover_from_manifest(paths, manifest)
                if discovered is not None:
                    metrics, run_start = discovered
                    _record_successor_binding(
                        paths,
                        manifest_path=manifest_path,
                        metrics_path=metrics,
                        run_start=run_start,
                    )
            reconciliation_returncode: int | None = None
            if manifest.get("trainer_exit_observed") is True:
                raw_returncode = manifest.get("trainer_reaped_returncode")
                if isinstance(raw_returncode, bool) or not isinstance(
                    raw_returncode,
                    int,
                ):
                    raise LaunchError(
                        "durable reaped-trainer return code is malformed",
                    )
                reconciliation_returncode = raw_returncode
            elif manifest.get("trainer_bootstrap_phase") in {
                "intent",
                "bootstrap_ready",
                "exec_authorized",
            }:
                _record_trainer_spawn_failed(
                    paths,
                    manifest_path=manifest_path,
                )
                manifest = _validate_manifest(paths, manifest_path)
            finalize_supervised_exit(
                paths,
                manifest_path=manifest_path,
                returncode=reconciliation_returncode,
                supervisor_identity=supervisor,
                trainer_identity=trainer,
                metrics_path=metrics,
                reconciliation_reason="supervisor_and_trainer_not_live",
                enforce_active_root=enforce_active_root,
            )
            repaired = _load_json_object(_state_path(paths), label="reconciled state")
            return _supervised_status(
                paths,
                state=repaired,
                manifest_path=manifest_path,
                allow_reconciliation=False,
                enforce_active_root=enforce_active_root,
            )
        else:
            status, running = "failed-unreconciled", False
    return {
        "schema_version": STATE_SCHEMA,
        "run_name": RUN_NAME,
        "status": status,
        "running": running,
        "launch_id": manifest.get("launch_id"),
        "manifest_path": str(manifest_path),
        "manifest_status": manifest_status,
        "preflight_training_started": manifest.get(
            "preflight_training_started",
        ),
        "training_started": manifest.get("training_started"),
        "trainer_bootstrap_protocol": manifest.get("trainer_bootstrap_protocol"),
        "trainer_bootstrap_phase": manifest.get("trainer_bootstrap_phase"),
        "trainer_spawn_intent": manifest.get("trainer_spawn_intent"),
        "trainer_spawn_failed": manifest.get("trainer_spawn_failed"),
        "trainer_exit_observed": manifest.get("trainer_exit_observed"),
        "trainer_reaped_returncode": manifest.get(
            "trainer_reaped_returncode",
        ),
        "log_path": manifest.get("log_path"),
        "metrics_path": manifest.get("metrics_path"),
        "terminal": manifest.get("terminal"),
        "launch_contract": manifest.get("launch_contract"),
        "selection": manifest.get("selection"),
        "abi_contract": manifest.get("abi_contract"),
        "shadow_validation": manifest.get("shadow_validation"),
        "process_identity_ambiguity": ambiguity,
        "supervisor": {"identity": asdict(supervisor) if supervisor else None, "identity_matches": supervisor_live},
        "trainer": {"identity": asdict(trainer) if trainer else None, "identity_matches": trainer_live},
        "bootstrap": {
            "identity": asdict(bootstrap) if bootstrap else None,
            "identity_matches": bootstrap_live,
        },
    }


def read_status(
    paths: LaunchPaths,
    *,
    enforce_active_root: bool = True,
) -> dict[str, Any]:
    paths = validate_layout(paths, enforce_active_root=enforce_active_root)
    state_path = _state_path(paths)
    state: dict[str, Any] | None = None
    manifest_path: Path | None = None
    manifest: dict[str, Any] | None = None
    state_needs_recovery = state_path.is_file()
    if state_path.is_file():
        try:
            candidate_state = _load_json_object(
                state_path,
                label="v29 launcher state",
            )
        except LaunchError:
            candidate_state = None
        if (
            candidate_state is not None
            and candidate_state.get(
                "schema_version",
            )
            == STATE_SCHEMA
        ):
            raw_manifest = candidate_state.get("manifest_path")
            if not isinstance(raw_manifest, str) or not raw_manifest:
                raise LaunchError(
                    "schema-valid v29 launcher state has no manifest path",
                )
            referenced = Path(raw_manifest).resolve(strict=False)
            if not _is_within(referenced, paths.manifest_dir):
                raise LaunchError(
                    "schema-valid v29 launcher state points outside the manifest directory",
                )
            try:
                referenced_manifest = _validate_manifest(
                    paths,
                    referenced,
                )
            except LaunchError as exc:
                raise LaunchError(
                    f"schema-valid v29 launcher state points at a missing or invalid manifest: {exc}",
                ) from exc
            try:
                current_directory_mtime_ns = paths.manifest_dir.stat().st_mtime_ns
            except OSError:
                current_directory_mtime_ns = None
            if candidate_state.get(
                "manifest_directory_mtime_ns",
            ) == current_directory_mtime_ns and _state_matches_manifest(
                candidate_state,
                manifest_path=referenced,
                manifest=referenced_manifest,
            ):
                state = candidate_state
                manifest_path = referenced
                manifest = referenced_manifest
                state_needs_recovery = False
    if state_needs_recovery or state is None:
        recovered = _latest_authoritative_manifest(paths)
        if recovered is None:
            if state_path.is_file():
                raise LaunchError(
                    "v29 launcher state exists without an authoritative launch manifest",
                )
            return {
                "schema_version": STATE_SCHEMA,
                "run_name": RUN_NAME,
                "status": "not-started",
                "running": False,
                "state_path": str(state_path),
            }
        manifest_path, manifest = recovered
        state = _state_payload(manifest_path, manifest)
        atomic_write_json(state_path, state)
    assert manifest_path is not None and manifest is not None and state is not None
    return _supervised_status(
        paths,
        state=state,
        manifest_path=manifest_path,
        allow_reconciliation=True,
        enforce_active_root=enforce_active_root,
    )


def _reprove_launch_contract(paths: LaunchPaths, manifest: Mapping[str, Any]) -> None:
    current = run_preflight(paths)
    for key in (
        "initialization",
        "config_fingerprint_sha256",
        "effective_config_fingerprint_sha256",
        "abi_contract",
        "shadow_validation",
        "runtime_readiness_evidence",
        "trainer_command",
        "trainer_environment",
        "trainer_environment_sha256",
        "runtime",
        "selection",
        "git",
        "source_authority",
        "config_file_sha256",
        "simulator_identity_sha256",
    ):
        if current.get(key) != manifest.get(key):
            raise LaunchError(f"v29 launch proof changed after detach: {key}")
    launch_id = manifest.get("launch_id")
    if not isinstance(launch_id, str):
        raise LaunchError("v29 launch manifest has no launch ID")
    _validate_launch_contract_file(
        manifest.get("launch_contract"),
        paths=paths,
        preflight=manifest,
        launch_id=launch_id,
    )


def _validate_bootstrap_ready_frame(
    value: object,
    *,
    nonce: str,
    supervisor_identity: ProcessIdentity,
    process: subprocess.Popen[bytes],
) -> ProcessIdentity:
    expected_fields = {
        "protocol",
        "kind",
        "nonce",
        "pid",
        "ppid",
        "proc_start_ticks",
    }
    if not isinstance(value, Mapping) or set(value) != expected_fields:
        raise LaunchError("trainer bootstrap READY frame fields changed")
    pid = value.get("pid")
    ppid = value.get("ppid")
    start_ticks = value.get("proc_start_ticks")
    if (
        value.get("protocol") != TRAINER_BOOTSTRAP_PROTOCOL
        or value.get("kind") != _trainer_bootstrap.READY_KIND
        or value.get("nonce") != nonce
        or isinstance(pid, bool)
        or not isinstance(pid, int)
        or pid != process.pid
        or isinstance(ppid, bool)
        or not isinstance(ppid, int)
        or ppid != supervisor_identity.pid
        or isinstance(start_ticks, bool)
        or not isinstance(start_ticks, int)
        or start_ticks <= 0
    ):
        raise LaunchError("trainer bootstrap READY frame identity changed")
    identity = capture_process_identity(pid)
    if identity.proc_start_ticks != start_ticks:
        raise LaunchError("trainer bootstrap READY start ticks differ from /proc")
    return identity


def _write_protocol_frame(fd: int, payload: bytes) -> None:
    view = memoryview(payload)
    while view:
        try:
            written = os.write(fd, view)
        except InterruptedError:
            continue
        except OSError as exc:
            raise LaunchError(f"trainer bootstrap protocol write failed: {exc}") from exc
        if written <= 0:
            raise LaunchError("trainer bootstrap protocol write made no progress")
        view = view[written:]


def _await_bootstrap_exec_ack(fd: int, *, timeout_s: float) -> None:
    """Require status-pipe EOF with no post-READY bytes.

    The helper marks the status descriptor ``FD_CLOEXEC`` immediately before
    ``execve``.  Therefore clean EOF is a kernel-level acknowledgement that the
    exact exec call succeeded; any ERROR/additional frame is a refusal.
    """

    deadline = time.monotonic() + timeout_s
    selector = selectors.DefaultSelector()
    try:
        selector.register(fd, selectors.EVENT_READ)
        while True:
            remaining = deadline - time.monotonic()
            if remaining <= 0.0 or not selector.select(remaining):
                raise LaunchError("trainer bootstrap exec acknowledgement timed out")
            try:
                payload = os.read(fd, _trainer_bootstrap.MAX_FRAME_BYTES + 1)
            except InterruptedError:
                continue
            except OSError as exc:
                raise LaunchError(f"trainer bootstrap status read failed: {exc}") from exc
            if not payload:
                return
            try:
                detail = payload.decode("ascii", errors="strict").strip()
            except UnicodeDecodeError:
                detail = payload.hex()
            raise LaunchError(
                f"trainer bootstrap refused exec after READY: {detail[:1024]}",
            )
    finally:
        selector.close()


def _capture_execed_trainer(
    process: subprocess.Popen[bytes],
    *,
    bootstrap_identity: ProcessIdentity,
    trainer_command: Sequence[str],
) -> ProcessIdentity:
    deadline = time.monotonic() + TRAINER_BOOTSTRAP_EXEC_TIMEOUT_SECONDS
    last_observed: ProcessIdentity | None = None
    while time.monotonic() < deadline:
        if process.poll() is not None:
            raise LaunchError(
                "exec-authorized child exited before final trainer identity was durable: "
                f"returncode={process.returncode}",
            )
        try:
            observed = capture_process_identity(process.pid)
        except LaunchError:
            time.sleep(0.01)
            continue
        last_observed = observed
        if observed.pid != bootstrap_identity.pid or observed.proc_start_ticks != bootstrap_identity.proc_start_ticks:
            raise LaunchError("bootstrap-to-trainer exec changed PID/start ticks")
        if _identity_matches_command(observed, trainer_command):
            return observed
        if observed == bootstrap_identity:
            time.sleep(0.01)
            continue
        raise LaunchError("exec-authorized child assumed an unauthorized command identity")
    raise LaunchError(
        "final trainer command identity was not observed before timeout: "
        f"last={asdict(last_observed) if last_observed is not None else None}",
    )


def supervise(paths: LaunchPaths, *, manifest_path: Path) -> dict[str, Any]:
    require_wsl()
    paths = validate_layout(paths)
    require_exact_artifact_environment(paths)
    manifest = _validate_manifest(paths, manifest_path)
    launch_id = str(manifest["launch_id"])
    supervisor_identity: ProcessIdentity | None = None
    bootstrap_identity: ProcessIdentity | None = None
    trainer_identity: ProcessIdentity | None = None
    trainer: subprocess.Popen[bytes] | None = None
    metrics: Path | None = None
    returncode: int | None = None
    spawn_intent_persisted = False
    trainer_exec_observed = False
    gate_read: int | None = None
    gate_write: int | None = None
    status_read: int | None = None
    status_write: int | None = None
    try:
        _reprove_launch_contract(paths, manifest)
        supervisor_identity = capture_process_identity(os.getpid())
        with _terminal_lock(paths, launch_id):
            manifest = _validate_manifest(paths, manifest_path)
            if manifest.get("status") in TERMINAL_STATUSES:
                return manifest
            manifest.update(
                {
                    "status": "supervisor-running",
                    "supervisor_process_identity": asdict(supervisor_identity),
                    "supervisor_started_at_utc": _utc_now(),
                }
            )
            _persist_manifest(paths, manifest_path=manifest_path, manifest=manifest)

        base_command = tuple(str(item) for item in manifest["trainer_command"])
        v29_preflight.validate_trainer_command(
            base_command,
            paths=paths.preflight,
            initialize_from=paths.initialization_checkpoint,
        )
        command = tuple(str(item) for item in manifest["supervised_trainer_command"])
        if command != _supervised_trainer_command(
            base_command,
            launch_contract=manifest["launch_contract"],
        ):
            raise LaunchError("supervised trainer command changed before spawn")
        environment_contract = manifest.get("trainer_environment")
        if not isinstance(environment_contract, Mapping):
            raise LaunchError("manifest trainer environment is malformed")
        environment = _environment_from_contract(environment_contract)
        v29_preflight.validate_trainer_environment(environment, paths=paths.preflight)
        log_path = Path(str(manifest["log_path"])).resolve(strict=False)
        log_path.parent.mkdir(parents=True, exist_ok=True)
        gate_read, gate_write = os.pipe2(os.O_CLOEXEC)
        status_read, status_write = os.pipe2(os.O_CLOEXEC)
        nonce = secrets.token_hex(32)
        bootstrap_command = _trainer_bootstrap_command(
            paths,
            supervisor_identity=supervisor_identity,
            gate_fd=gate_read,
            status_fd=status_write,
            nonce=nonce,
            trainer_command=command,
        )
        _record_trainer_spawn_intent(
            paths,
            manifest_path=manifest_path,
            bootstrap_command=bootstrap_command,
            nonce_sha256=_sha256_bytes(nonce.encode("ascii")),
        )
        spawn_intent_persisted = True
        with log_path.open("ab", buffering=0) as log_handle:
            trainer = subprocess.Popen(
                bootstrap_command,
                cwd=paths.package_root,
                env=environment,
                stdin=subprocess.DEVNULL,
                stdout=log_handle,
                stderr=subprocess.STDOUT,
                pass_fds=(gate_read, status_write),
            )
        os.close(gate_read)
        gate_read = None
        os.close(status_write)
        status_write = None
        try:
            ready = _trainer_bootstrap.read_bounded_frame(
                status_read,
                TRAINER_BOOTSTRAP_READY_TIMEOUT_SECONDS,
            )
        except _trainer_bootstrap.BootstrapError as exc:
            raise LaunchError(f"trainer bootstrap READY failed: {exc}") from exc
        bootstrap_identity = _validate_bootstrap_ready_frame(
            ready,
            nonce=nonce,
            supervisor_identity=supervisor_identity,
            process=trainer,
        )
        if not _identity_matches_command(bootstrap_identity, bootstrap_command):
            raise LaunchError("trainer bootstrap READY command differs from authorization")
        _record_trainer_bootstrap_ready(
            paths,
            manifest_path=manifest_path,
            identity=bootstrap_identity,
        )
        _record_trainer_exec_authorized(
            paths,
            manifest_path=manifest_path,
            identity=bootstrap_identity,
        )
        go = _trainer_bootstrap.canonical_frame(
            {
                "kind": _trainer_bootstrap.GO_KIND,
                "nonce": nonce,
                "protocol": TRAINER_BOOTSTRAP_PROTOCOL,
            },
        )
        _write_protocol_frame(gate_write, go)
        os.close(gate_write)
        gate_write = None
        _await_bootstrap_exec_ack(
            status_read,
            timeout_s=TRAINER_BOOTSTRAP_EXEC_TIMEOUT_SECONDS,
        )
        os.close(status_read)
        status_read = None
        trainer_identity = _capture_execed_trainer(
            trainer,
            bootstrap_identity=bootstrap_identity,
            trainer_command=command,
        )
        _record_trainer_exec_observed(
            paths,
            manifest_path=manifest_path,
            bootstrap_identity=bootstrap_identity,
            trainer_identity=trainer_identity,
        )
        trainer_exec_observed = True

        def forward_signal(number: int, _frame: object) -> None:
            if trainer is not None and trainer.poll() is None:
                trainer.send_signal(number)

        signal.signal(signal.SIGINT, forward_signal)
        signal.signal(signal.SIGTERM, forward_signal)
        while True:
            returncode = trainer.poll()
            if metrics is None:
                current_manifest = _load_json_object(manifest_path, label="manifest")
                discovered = _discover_from_manifest(paths, current_manifest)
                if discovered is not None:
                    metrics, run_start = discovered
                    _record_successor_binding(
                        paths,
                        manifest_path=manifest_path,
                        metrics_path=metrics,
                        run_start=run_start,
                    )
            if returncode is not None:
                break
            time.sleep(1.0)
        _record_trainer_exit_observed(
            paths,
            manifest_path=manifest_path,
            pid=trainer.pid,
            returncode=returncode,
        )
        deadline = time.monotonic() + 5.0
        while metrics is None and time.monotonic() < deadline:
            current_manifest = _load_json_object(manifest_path, label="manifest")
            discovered = _discover_from_manifest(paths, current_manifest)
            if discovered is not None:
                metrics, run_start = discovered
                _record_successor_binding(
                    paths,
                    manifest_path=manifest_path,
                    metrics_path=metrics,
                    run_start=run_start,
                )
                break
            time.sleep(0.2)
        return finalize_supervised_exit(
            paths,
            manifest_path=manifest_path,
            returncode=returncode,
            supervisor_identity=supervisor_identity,
            trainer_identity=trainer_identity,
            metrics_path=metrics,
        )
    except Exception as exc:
        for descriptor in (
            gate_read,
            gate_write,
            status_read,
            status_write,
        ):
            if isinstance(descriptor, int):
                try:
                    os.close(descriptor)
                except OSError:
                    pass
        if trainer is not None and trainer.poll() is None:
            trainer.terminate()
            try:
                returncode = trainer.wait(timeout=15.0)
            except subprocess.TimeoutExpired:
                trainer.kill()
                returncode = trainer.wait(timeout=15.0)
        elif trainer is not None:
            returncode = trainer.poll()
        try:
            durable_phase: object = None
            if spawn_intent_persisted:
                durable_phase = _validate_manifest(
                    paths,
                    manifest_path,
                ).get("trainer_bootstrap_phase")
            exec_was_durable = trainer_exec_observed or durable_phase in {
                "exec_observed",
                "reaped",
            }
            if trainer is not None and returncode is not None and exec_was_durable and durable_phase != "reaped":
                _record_trainer_exit_observed(
                    paths,
                    manifest_path=manifest_path,
                    pid=trainer.pid,
                    returncode=returncode,
                )
            elif spawn_intent_persisted and not exec_was_durable:
                _record_trainer_spawn_failed(
                    paths,
                    manifest_path=manifest_path,
                )
        except Exception:
            # Do not erase the original supervisor failure.  Missing reaped or
            # spawn-failed evidence makes later status recovery ambiguous and
            # therefore fail-closed rather than falsely declaring the child dead.
            pass
        finalization_failure: Exception | None = None
        try:
            current_manifest = _load_json_object(manifest_path, label="manifest")
            if metrics is None:
                discovered = _discover_from_manifest(paths, current_manifest)
                if discovered is not None:
                    metrics, run_start = discovered
                    _record_successor_binding(
                        paths,
                        manifest_path=manifest_path,
                        metrics_path=metrics,
                        run_start=run_start,
                    )
            finalize_supervised_exit(
                paths,
                manifest_path=manifest_path,
                returncode=returncode if returncode is not None else 70,
                supervisor_identity=supervisor_identity,
                trainer_identity=trainer_identity,
                metrics_path=metrics,
                reconciliation_reason=f"supervisor_exception:{type(exc).__name__}:{exc}",
            )
        except Exception as recovery_exc:
            finalization_failure = recovery_exc
            try:
                _persist_supervisor_emergency_failure(
                    paths,
                    manifest_path=manifest_path,
                    launch_id=launch_id,
                    original_error=exc,
                    finalization_error=recovery_exc,
                    returncode=returncode,
                    supervisor_identity=supervisor_identity,
                    trainer_identity=trainer_identity,
                    metrics_path=metrics,
                )
            except Exception as emergency_exc:
                raise LaunchError(
                    "supervisor failed and neither normal finalization nor "
                    "emergency persistence succeeded: "
                    f"original={type(exc).__name__}:{exc}; "
                    "finalization="
                    f"{type(recovery_exc).__name__}:{recovery_exc}; "
                    "emergency="
                    f"{type(emergency_exc).__name__}:{emergency_exc}",
                ) from exc
        if finalization_failure is not None:
            raise LaunchError(
                "supervisor failed; normal finalization also failed, but a "
                "durable supervisor-recovery failure record was persisted: "
                f"original={type(exc).__name__}:{exc}; "
                "finalization="
                f"{type(finalization_failure).__name__}:"
                f"{finalization_failure}",
            ) from exc
        if isinstance(exc, LaunchError | v29_preflight.PreflightError):
            raise LaunchError(str(exc)) from exc
        raise LaunchError(f"supervisor failed: {type(exc).__name__}: {exc}") from exc


def start(
    paths: LaunchPaths,
    *,
    reviewed_preflight: Path | None = None,
) -> dict[str, Any]:
    require_wsl()
    preflight = run_preflight(paths, reviewed_preflight=reviewed_preflight)
    paths = validate_layout(paths)
    paths.launcher_dir.mkdir(parents=True, exist_ok=True)
    paths.manifest_dir.mkdir(parents=True, exist_ok=True)
    with _terminal_lock(paths, f"{RUN_NAME}-start"):
        previous = read_status(paths)
        if previous["running"] is True:
            raise LaunchError(
                "v29 already has an active trainer/supervisor: "
                f"status={previous.get('status')} launch_id={previous.get('launch_id')}"
            )
        launch_id = str(uuid.uuid4())
        timestamp = datetime.now(UTC).strftime("%Y%m%d-%H%M%S")
        created_at_utc = _utc_now()
        created_unix_s = time.time()
        log_path = paths.launcher_dir / "logs" / f"{RUN_NAME}-{timestamp}-{launch_id}.log"
        manifest_path = paths.manifest_dir / f"{RUN_NAME}-{timestamp}-{launch_id}.launch.json"
        supervisor_command = _supervisor_command(
            paths,
            manifest_path,
        )
        launch_contract = _create_supervised_launch_contract(
            paths,
            preflight=preflight,
            launch_id=launch_id,
            created_at_utc=created_at_utc,
            created_unix_s=created_unix_s,
        )
        _validate_launch_contract_file(
            launch_contract,
            paths=paths,
            preflight=preflight,
            launch_id=launch_id,
        )
        launch_contract_payload = _load_json_object(
            Path(str(launch_contract["path"])),
            label="immutable supervised launch contract",
        )
        supervised_trainer_command = _supervised_trainer_command(
            tuple(str(item) for item in preflight["trainer_command"]),
            launch_contract=launch_contract,
        )
        manifest: dict[str, Any] = {
            **preflight,
            "launch_id": launch_id,
            "created_at_utc": created_at_utc,
            "created_unix_s": created_unix_s,
            "status": "supervisor-launching",
            "training_started": False,
            "trainer_bootstrap_protocol": TRAINER_BOOTSTRAP_PROTOCOL,
            "trainer_bootstrap_phase": "none",
            "trainer_bootstrap_command": None,
            "trainer_bootstrap_nonce_sha256": None,
            "trainer_bootstrap_process_identity": None,
            "trainer_bootstrap_ready_at_utc": None,
            "trainer_bootstrap_exec_authorized_at_utc": None,
            "trainer_bootstrap_exec_observed_at_utc": None,
            "trainer_spawn_intent": False,
            "trainer_spawn_intent_at_utc": None,
            "trainer_spawn_failed": False,
            "trainer_spawn_failure_at_utc": None,
            "trainer_spawned_pid": None,
            "trainer_spawned_at_utc": None,
            "trainer_started_at_utc": None,
            "trainer_exit_observed": False,
            "trainer_exit_observed_at_utc": None,
            "trainer_reaped_returncode": None,
            "launch_contract": launch_contract,
            "launch_contract_payload": launch_contract_payload,
            "supervised_trainer_command": list(
                supervised_trainer_command,
            ),
            "supervisor_command": list(supervisor_command),
            "supervisor_process_identity": None,
            "trainer_process_identity": None,
            "preexisting_run_directories": _snapshot_run_directories(paths),
            "metrics_path": None,
            "log_path": str(log_path),
            "manifest_path": str(manifest_path),
        }
        _persist_manifest(paths, manifest_path=manifest_path, manifest=manifest)
        environment_contract = manifest.get("trainer_environment")
        assert isinstance(environment_contract, Mapping)
        environment = _environment_from_contract(environment_contract)
        log_path.parent.mkdir(parents=True, exist_ok=True)
        process: subprocess.Popen[bytes] | None = None
        try:
            with log_path.open("ab", buffering=0) as log_handle:
                process = subprocess.Popen(
                    supervisor_command,
                    cwd=paths.package_root,
                    env=environment,
                    stdin=subprocess.DEVNULL,
                    stdout=log_handle,
                    stderr=subprocess.STDOUT,
                    start_new_session=True,
                )
            with _terminal_lock(paths, launch_id):
                manifest = _validate_manifest(paths, manifest_path)
                manifest["supervisor_spawned_pid"] = process.pid
                manifest["supervisor_spawned_at_utc"] = _utc_now()
                _persist_manifest(
                    paths,
                    manifest_path=manifest_path,
                    manifest=manifest,
                )
            identity = _capture_child(process)
        except (OSError, LaunchError) as exc:
            if process is not None and process.poll() is None:
                process.terminate()
                try:
                    process.wait(timeout=15.0)
                except subprocess.TimeoutExpired:
                    process.kill()
                    process.wait(timeout=15.0)
            finalize_supervised_exit(
                paths,
                manifest_path=manifest_path,
                returncode=70,
                supervisor_identity=None,
                trainer_identity=None,
                metrics_path=None,
                reconciliation_reason=f"supervisor_spawn_failed:{exc}",
            )
            raise LaunchError(f"could not spawn v29 supervisor: {exc}") from exc
        with _terminal_lock(paths, launch_id):
            manifest = _validate_manifest(paths, manifest_path)
            if manifest.get("supervisor_process_identity") is None:
                manifest["supervisor_process_identity"] = asdict(identity)
            _persist_manifest(paths, manifest_path=manifest_path, manifest=manifest)
    return read_status(paths)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "action",
        nargs="?",
        default="status",
        choices=(
            "preflight",
            "start",
            "status",
            "supervise",
            HERMETIC_RUNTIME_PROBE_ACTION,
        ),
    )
    parser.add_argument(
        "--reviewed-preflight",
        type=Path,
        help="explicit operator-reviewed JSON from the non-launching v29 preflight",
    )
    parser.add_argument("--manifest", type=Path, help=argparse.SUPPRESS)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    paths = default_paths()
    try:
        if args.action != "supervise" and args.manifest is not None:
            raise LaunchError(f"{args.action} does not accept --manifest")
        if args.action not in {"preflight", "start"} and args.reviewed_preflight is not None:
            raise LaunchError(f"{args.action} does not accept --reviewed-preflight")
        if args.action == HERMETIC_RUNTIME_PROBE_ACTION:
            require_wsl()
            paths = validate_layout(paths)
            require_exact_artifact_environment(paths)
            payload = _execute_hermetic_runtime_probe(paths.preflight)
        elif args.action == "preflight":
            require_wsl()
            payload = run_preflight(paths, reviewed_preflight=args.reviewed_preflight)
        elif args.action == "start":
            payload = start(paths, reviewed_preflight=args.reviewed_preflight)
        elif args.action == "supervise":
            if args.manifest is None:
                raise LaunchError("internal supervise action requires --manifest")
            payload = supervise(paths, manifest_path=args.manifest)
        else:
            require_wsl()
            require_exact_artifact_environment(validate_layout(paths))
            payload = read_status(paths)
    except (LaunchError, v29_preflight.PreflightError) as exc:
        print(f"[v29-launcher] {exc}", file=sys.stderr)
        return 2
    print(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
