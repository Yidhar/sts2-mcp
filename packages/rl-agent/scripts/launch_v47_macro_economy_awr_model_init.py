#!/usr/bin/env python3
"""Launch the reviewed v47 factual macro-economy AWR lineage.

The immutable network source is v46's healthy validation checkpoint at local
environment step 10,035. V47 introduces a unified factual card-reward/shop/rest
economy horizon, a candidate-independent macro value baseline, and an isolated
residual AWR actor. Shared tactical features are detached from these new losses.
Consequently this is an explicit model-parameter initialization with fresh
optimizer/replay/RNG/rollout state, never an exact resume. Reward-v7 and the
budget-40 curriculum remain unchanged. Preflight revalidates every fixed v46
checkpoint identity and the pinned patched HeadlessSim dependency before a
new process may start.
"""

from __future__ import annotations

import importlib.util
import json
import os
import sys
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any, cast


def _load_base() -> Any:
    path = Path(__file__).resolve().with_name(
        "launch_v30_failure_credit_v5_model_init.py"
    )
    module_name = "_sts2_v47_macro_economy_awr_model_init_base"
    existing = sys.modules.get(module_name)
    if existing is not None:
        return existing
    spec = importlib.util.spec_from_file_location(module_name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load model-init control plane: {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


_base = _load_base()
_base.__doc__ = __doc__

RUN_NAME = "full-run-revival-v47-macro-economy-awr-model-init"
SOURCE_RUN_ID = "9a23c1bc-7a7b-467e-ad97-95af63fe4bec"
SOURCE_ENVIRONMENT_STEPS = 10_035
SOURCE_TOTAL_ENVIRONMENT_STEPS = 250_000
TARGET_ENVIRONMENT_STEPS = 30_000
ADDITIONAL_ENVIRONMENT_STEPS = TARGET_ENVIRONMENT_STEPS
SOURCE_CHECKPOINT_ID = "a497e8e7-00ea-4a89-af41-ce30691997cd"
SOURCE_MANIFEST_SHA256 = (
    "775bf964fb54ffa86166495d80fba39dd6a49de13522032b5c3654e6860cba35"
)
SOURCE_METADATA_SHA256 = (
    "0a2de33f13ce49807065f264ee3d739833c4044602907075ef5284cfc0f144f9"
)
SOURCE_POLICY_VERSION = 164
SOURCE_LEARNER_UPDATES = 164
SOURCE_CHECKPOINT_RELATIVE = Path(
    "checkpoints/full-run-revival-v46-macro-option-actor-model-init"
    f"/run-{SOURCE_RUN_ID}/healthy-validation-step-{SOURCE_ENVIRONMENT_STEPS:09d}"
)
SOURCE_ATTESTATION_RELATIVE = Path(
    "validation/checkpoint-attestations/"
    "v46-healthy-010035-byte-verified-v1.json"
)
SOURCE_ATTESTATION_SHA256 = (
    "a00096dbccde2355471b5cdf093031db53f887357ca4b28265631865aab3d518"
)
ACTIVE_ARTIFACT_ROOT = Path(
    os.environ.get(
        "STS2_ARTIFACT_ROOT",
        str(
            Path(__file__).resolve().parents[3].with_name("sts2_mcp_artifacts")
            / "runtime"
        ),
    )
)
CONFIG_FILE = "full_run_revival_v47_macro_economy_awr_model_init.toml"
LINEAGE_LABEL = "v47"

SCHEMA_VERSION = "sts2-v47-macro-economy-awr-model-init-preflight-v1"
STATE_SCHEMA_VERSION = "sts2-v47-macro-economy-awr-model-init-state-v1"
SUPERVISED_SCHEMA_VERSION = (
    "sts2-v47-macro-economy-awr-model-init-supervised-launch-v1"
)
SUPERVISED_STATE_SCHEMA_VERSION = (
    "sts2-v47-macro-economy-awr-model-init-supervised-state-v1"
)
LEARNER_STALL_TIMEOUT_SECONDS = 30.0 * 60.0

_OVERRIDES = {
    "RUN_NAME": RUN_NAME,
    "SOURCE_RUN_ID": SOURCE_RUN_ID,
    "SOURCE_ENVIRONMENT_STEPS": SOURCE_ENVIRONMENT_STEPS,
    "SOURCE_TOTAL_ENVIRONMENT_STEPS": SOURCE_TOTAL_ENVIRONMENT_STEPS,
    "TARGET_ENVIRONMENT_STEPS": TARGET_ENVIRONMENT_STEPS,
    "ADDITIONAL_ENVIRONMENT_STEPS": ADDITIONAL_ENVIRONMENT_STEPS,
    "SOURCE_CHECKPOINT_ID": SOURCE_CHECKPOINT_ID,
    "SOURCE_MANIFEST_SHA256": SOURCE_MANIFEST_SHA256,
    "SOURCE_METADATA_SHA256": SOURCE_METADATA_SHA256,
    "SOURCE_POLICY_VERSION": SOURCE_POLICY_VERSION,
    "SOURCE_LEARNER_UPDATES": SOURCE_LEARNER_UPDATES,
    "SOURCE_CHECKPOINT_RELATIVE": SOURCE_CHECKPOINT_RELATIVE,
    "ACTIVE_ARTIFACT_ROOT": ACTIVE_ARTIFACT_ROOT,
    "CONFIG_FILE": CONFIG_FILE,
    "LINEAGE_LABEL": LINEAGE_LABEL,
    "SCHEMA_VERSION": SCHEMA_VERSION,
    "STATE_SCHEMA_VERSION": STATE_SCHEMA_VERSION,
    "SUPERVISED_SCHEMA_VERSION": SUPERVISED_SCHEMA_VERSION,
    "SUPERVISED_STATE_SCHEMA_VERSION": SUPERVISED_STATE_SCHEMA_VERSION,
    "LEARNER_STALL_TIMEOUT_SECONDS": LEARNER_STALL_TIMEOUT_SECONDS,
}
for _name, _value in _OVERRIDES.items():
    setattr(_base, _name, _value)


def source_attestation_path(paths: Any) -> Path:
    return (paths.artifact_root / SOURCE_ATTESTATION_RELATIVE).resolve(
        strict=False
    )


def _unchecked_trainer_command(paths: Any) -> tuple[str, ...]:
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
        str(_base.source_checkpoint_path(paths)),
        "--model-initialization-attestation",
        str(source_attestation_path(paths)),
        "--model-initialization-attestation-sha256",
        SOURCE_ATTESTATION_SHA256,
    )


def validate_fixed_model_initialization_command(
    command: Sequence[str],
    *,
    paths: Any,
) -> None:
    values = tuple(command)
    if values != _unchecked_trainer_command(paths):
        raise _base.LaunchError(
            f"{LINEAGE_LABEL} model-initialization trainer command was modified"
        )
    if values.count("--initialize-from") != 1 or "--resume" in values:
        raise _base.LaunchError(
            f"{LINEAGE_LABEL} must initialize model parameters, never exact-resume"
        )
    if values.count("--model-initialization-attestation") != 1:
        raise _base.LaunchError(
            f"{LINEAGE_LABEL} must bind one source-byte attestation"
        )
    if values.count("--model-initialization-attestation-sha256") != 1:
        raise _base.LaunchError(
            f"{LINEAGE_LABEL} must pin the source-byte attestation digest"
        )


def build_resume_trainer_command(paths: Any) -> tuple[str, ...]:
    """Compatibility hook name required by the shared supervisor core."""

    command = _unchecked_trainer_command(paths)
    validate_fixed_model_initialization_command(command, paths=paths)
    return command


def _verify_source_checkpoint(
    paths: Any,
    environment: Mapping[str, str],
) -> dict[str, Any]:
    """Reuse the pinned landing proof; never re-hash 1.8 GiB of sidecars."""

    checkpoint = _base.source_checkpoint_path(paths)
    attestation = source_attestation_path(paths)
    if not checkpoint.is_dir():
        raise _base.LaunchError(f"fixed v46 model source is missing: {checkpoint}")
    if not attestation.is_file():
        raise _base.LaunchError(
            f"fixed v46 source-byte attestation is missing: {attestation}"
        )
    source = r"""
import hashlib
import json
import sys
from dataclasses import replace
from pathlib import Path
from sts2_rl.training.checkpointing import preflight_model_initialization
from sts2_rl.training.config import load_training_config

checkpoint = Path(sys.argv[1]).resolve()
config_path = Path(sys.argv[2]).resolve()
simulator = Path(sys.argv[3]).resolve()
attestation = Path(sys.argv[4]).resolve()
attestation_sha256 = sys.argv[5]
config = load_training_config(profile="preheat", config_path=config_path)
config = replace(
    config,
    runtime=replace(config.runtime, device="cuda", collector_device="cpu"),
    environment=replace(
        config.environment,
        backend="headless",
        sim_exe_path=str(simulator),
    ),
)
validated = preflight_model_initialization(
    checkpoint,
    config=config,
    attestation=attestation,
    attestation_sha256=attestation_sha256,
)
metadata = validated.metadata
provenance = metadata.get("provenance") or {}
source_runtime = (metadata.get("training_config") or {}).get("runtime") or {}
training_state = metadata.get("training_state") or {}
payload = {
    "root": str(validated.root),
    "checkpoint_id": validated.manifest.get("checkpoint_id"),
    "manifest_sha256": hashlib.sha256(
        (checkpoint / "checkpoint.manifest.json").read_bytes()
    ).hexdigest(),
    "metadata_sha256": hashlib.sha256(
        (checkpoint / "metadata.json").read_bytes()
    ).hexdigest(),
    "manifest_files": validated.manifest.get("files"),
    "experiment_run_id": provenance.get("experiment_run_id"),
    "environment_steps": training_state.get("environment_steps"),
    "policy_version": training_state.get("policy_version"),
    "learner_updates": training_state.get("learner_updates"),
    "source_total_environment_steps": source_runtime.get("total_environment_steps"),
    "target_total_environment_steps": config.runtime.total_environment_steps,
    "checkpoint_format": metadata.get("format"),
    "payload_validation": "prior-full-byte-attestation+current-identity-and-size",
    "attestation_path": str(attestation),
    "attestation_sha256": attestation_sha256,
}
print(json.dumps(payload, sort_keys=True))
"""
    output = _base._core._run_checked(
        (
            str(paths.venv_python),
            "-c",
            source,
            str(checkpoint),
            str(paths.config_path),
            str(paths.simulator_executable),
            str(attestation),
            SOURCE_ATTESTATION_SHA256,
        ),
        cwd=paths.package_root,
        environment=environment,
    )
    try:
        payload = json.loads(output.strip().splitlines()[-1])
    except (IndexError, json.JSONDecodeError) as exc:
        raise _base.LaunchError(
            "attested model-initialization preflight returned invalid JSON"
        ) from exc
    if not isinstance(payload, dict):
        raise _base.LaunchError(
            "attested model-initialization preflight must return an object"
        )
    return _base._validate_checkpoint_summary(payload, paths=paths)


def build_supervisor_command(
    paths: Any,
    *,
    manifest_path: Path,
) -> tuple[str, ...]:
    """Re-enter this adapter so the immutable v46 source pins remain bound."""

    return (
        str(paths.venv_python),
        str(Path(__file__).resolve()),
        "supervise",
        "--manifest",
        str(manifest_path),
    )


_base._unchecked_trainer_command = _unchecked_trainer_command
_base.validate_fixed_model_initialization_command = (
    validate_fixed_model_initialization_command
)
_base.build_resume_trainer_command = build_resume_trainer_command
_base._verify_source_checkpoint = _verify_source_checkpoint
_base.build_supervisor_command = build_supervisor_command
_base._configure_supervisor_core()

LaunchError = _base.LaunchError
LaunchPaths = _base.LaunchPaths
default_paths = _base.default_paths
build_environment = _base.build_environment
source_checkpoint_path = _base.source_checkpoint_path
_validate_checkpoint_summary = _base._validate_checkpoint_summary
run_model_initialization_preflight = _base.run_model_initialization_preflight
_validate_supervised_manifest = _base._validate_supervised_manifest
_matching_model_initialization_run_start = (
    _base._matching_model_initialization_run_start
)
initialize = _base.initialize
_core = _base._core


def main(argv: Sequence[str] | None = None) -> int:
    return cast(int, _base.main(argv))


if __name__ == "__main__":
    raise SystemExit(main())
