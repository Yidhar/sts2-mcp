"""CLI for frozen, learner-free held-out checkpoint evaluation."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import shutil
import sys
import time
import traceback
from collections.abc import Sequence
from concurrent.futures import ProcessPoolExecutor, as_completed
from multiprocessing import get_context
from pathlib import Path
from uuid import uuid4

import numpy as np
import torch

from sts2_env.headless_sim_bridge_client import (
    HeadlessSimError,
    resolve_headless_sim_exe,
)

from .artifacts import resolve_artifact_path, resolve_external_input_path
from .checkpoints import validate_resume_checkpoint
from .runtime_mechanics import (
    RuntimeMechanicsAuditError,
    run_runtime_mechanics_preflight,
    write_runtime_mechanics_audit,
)
from .simulator_identity import (
    SimulatorIdentityError,
    verify_headless_simulator,
    write_preflight_audit,
)
from .training.checkpoint_evaluation import (
    FrozenEvaluationResult,
    atomic_publish_directory,
    checkpoint_training_config,
    evaluate_checkpoint_policy,
)
from .training.seeding import held_out_evaluation_seeds

_PAIRED_EVALUATION_SCHEMA = "sts2-paired-frozen-evaluation-v2"
_DURABLE_CHILD_SCHEMA = "sts2-durable-frozen-evaluation-child-v1"
_DURABLE_CHILD_CONTRACT = "checkpoint-config-seeds-simulator-implementation-v2"
_DURABLE_CHILD_MANIFEST = "durable-child.manifest.json"
_DURABLE_CHILD_FILES = frozenset(
    {
        "evaluation.json",
        "trajectory.jsonl",
        _DURABLE_CHILD_MANIFEST,
    }
)


def _atomic_write_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp-{uuid4().hex}")
    temporary.write_text(
        json.dumps(payload, indent=2, sort_keys=True, default=str) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _lineage_fingerprint(lineage: dict[str, object]) -> str:
    payload = json.dumps(
        lineage,
        ensure_ascii=False,
        allow_nan=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _mapping_fingerprint(payload: object) -> str:
    encoded = json.dumps(
        payload,
        ensure_ascii=False,
        allow_nan=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _evaluation_implementation_fingerprint() -> str:
    """Hash the Python policy/evaluator implementation used by durable reuse.

    A manually versioned cache ABI alone is too easy to leave stale while the
    encoder, model, collector, reward, or protocol adapter changes.  Hashing
    the three runtime packages is deliberately conservative: an unrelated
    Python edit may invalidate useful cached work, but stale policy semantics
    can never be silently mixed into a formal paired comparison.
    """

    package_root = Path(__file__).resolve().parent.parent
    digest = hashlib.sha256()
    files = sorted(
        path
        for package_name in ("sts2_baseline", "sts2_env", "sts2_rl")
        for path in (package_root / package_name).rglob("*.py")
        if path.is_file()
    )
    if not files:
        raise RuntimeError("evaluation implementation fingerprint found no Python sources")
    for path in files:
        relative = path.relative_to(package_root).as_posix().encode("utf-8")
        digest.update(len(relative).to_bytes(4, "big"))
        digest.update(relative)
        with path.open("rb") as handle:
            for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(chunk)
    return digest.hexdigest()


def _dependency_lock_fingerprints() -> dict[str, dict[str, object]]:
    """Fingerprint the dependency inputs which define the evaluator runtime."""

    package_root = Path(__file__).resolve().parent.parent
    lock_files = sorted(
        {
            *package_root.glob("requirements*.lock"),
            *package_root.glob("requirements*.txt"),
        }
    )
    if not lock_files:
        raise RuntimeError("evaluation dependency fingerprint found no requirements lock inputs")
    return {
        path.name: {
            "size_bytes": path.stat().st_size,
            "sha256": _sha256(path),
        }
        for path in lock_files
    }


def _runtime_environment_contract() -> dict[str, object]:
    """Return stable process/runtime facts which can change deterministic policy traces."""

    return {
        "python_version": sys.version,
        "python_implementation": platform.python_implementation(),
        "platform": platform.platform(),
        "platform_system": platform.system(),
        "platform_machine": platform.machine(),
        "numpy_version": str(np.__version__),
        "torch_num_threads": torch.get_num_threads(),
        "torch_num_interop_threads": torch.get_num_interop_threads(),
        "dependency_locks": _dependency_lock_fingerprints(),
    }


def _resolved_device_name(requested: str) -> str:
    normalized = str(requested).strip().lower()
    if normalized == "auto":
        return "cuda" if torch.cuda.is_available() else "cpu"
    return str(torch.device(requested))


def _durable_child_contract(
    *,
    checkpoint_id: str,
    checkpoint_manifest_sha256: str,
    lineage_sha256: str,
    simulator_semantics: dict[str, object],
    implementation_sha256: str,
    runtime_environment: dict[str, object],
    episodes: int,
    resolved_base_seed: int,
    evaluation_seeds: Sequence[int],
    device: str,
    collector_device: str,
) -> dict[str, object]:
    return {
        "contract_version": _DURABLE_CHILD_CONTRACT,
        "checkpoint_id": checkpoint_id,
        "checkpoint_manifest_sha256": checkpoint_manifest_sha256,
        "lineage_sha256": lineage_sha256,
        "simulator_semantics": simulator_semantics,
        "simulator_semantics_sha256": _mapping_fingerprint(simulator_semantics),
        "evaluation_implementation_sha256": implementation_sha256,
        "runtime_environment": runtime_environment,
        "episodes": episodes,
        "resolved_base_seed": resolved_base_seed,
        "seed_contract": "even-training/odd-held-out-v1",
        "held_out_seeds": list(evaluation_seeds),
        "data_partition": "validation",
        "epsilon": 0.0,
        "deterministic": True,
        "device": _resolved_device_name(device),
        "collector_device": _resolved_device_name(collector_device),
        "torch_version": str(torch.__version__),
        "torch_hip_version": str(torch.version.hip or ""),
    }


def _durable_file_record(root: Path, name: str) -> dict[str, object]:
    path = root / name
    if path.is_symlink() or not path.is_file():
        raise ValueError(f"durable evaluation child has no regular {name}: {root}")
    return {
        "path": name,
        "size_bytes": path.stat().st_size,
        "sha256": _sha256(path),
    }


def _durable_manifest_file_records(
    manifest: dict[str, object],
    *,
    root: Path,
) -> dict[str, dict[str, object]]:
    files = manifest.get("files")
    if not isinstance(files, list):
        raise ValueError(f"durable evaluation child file manifest is invalid: {root}")
    records: dict[str, dict[str, object]] = {}
    for item in files:
        if not isinstance(item, dict):
            raise ValueError(f"durable evaluation child file manifest is invalid: {root}")
        name = item.get("path")
        if not isinstance(name, str) or name in records:
            raise ValueError(f"durable evaluation child file manifest is invalid: {root}")
        records[name] = item
    if len(files) != 2 or set(records) != {"evaluation.json", "trajectory.jsonl"}:
        raise ValueError(f"durable evaluation child file set is invalid: {root}")
    return records


def _validate_durable_child(
    root: Path,
    *,
    expected_contract: dict[str, object],
) -> tuple[dict[str, object], dict[str, object]]:
    """Validate one complete durable child without trusting directory names."""

    manifest_path = root / _DURABLE_CHILD_MANIFEST
    if root.is_symlink() or not root.is_dir():
        raise ValueError(f"durable evaluation child is incomplete: {root}")
    try:
        actual_entries = {entry.name: entry for entry in root.iterdir()}
    except OSError as exc:
        raise ValueError(f"durable evaluation child directory is unreadable: {root}") from exc
    if set(actual_entries) != _DURABLE_CHILD_FILES:
        raise ValueError(f"durable evaluation child on-disk file set is invalid: {root}")
    for name, entry in actual_entries.items():
        if entry.is_symlink() or not entry.is_file():
            raise ValueError(f"durable evaluation child entry is not a regular file ({name}): {root}")
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"durable evaluation child manifest is unreadable: {root}") from exc
    if not isinstance(manifest, dict):
        raise ValueError(f"durable evaluation child manifest is not an object: {root}")
    expected_key = _mapping_fingerprint(expected_contract)
    if (
        manifest.get("schema_version") != _DURABLE_CHILD_SCHEMA
        or manifest.get("status") != "complete"
        or manifest.get("cache_key") != expected_key
        or manifest.get("contract") != expected_contract
    ):
        raise ValueError(f"durable evaluation child contract mismatch: {root}")
    expected_files = _durable_manifest_file_records(manifest, root=root)
    for name, recorded in expected_files.items():
        actual = _durable_file_record(root, name)
        if actual != recorded:
            raise ValueError(f"durable evaluation child hash mismatch for {name}: {root}")
    result = manifest.get("result")
    if not isinstance(result, dict):
        raise ValueError(f"durable evaluation child result is invalid: {root}")
    checkpoint_id = expected_contract["checkpoint_id"]
    if result.get("checkpoint_id") != checkpoint_id:
        raise ValueError(f"durable evaluation child checkpoint mismatch: {root}")
    for name in ("source_environment_steps", "source_policy_version"):
        value = result.get(name)
        if isinstance(value, bool) or not isinstance(value, int) or value < 0:
            raise ValueError(f"durable evaluation child {name} is invalid: {root}")
    if not isinstance(result.get("summary"), dict):
        raise ValueError(f"durable evaluation child summary is invalid: {root}")

    try:
        audit = json.loads((root / "evaluation.json").read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"durable evaluation child audit is unreadable: {root}") from exc
    if not isinstance(audit, dict):
        raise ValueError(f"durable evaluation child audit is not an object: {root}")
    evaluation_of = audit.get("evaluation_of")
    evaluation = audit.get("evaluation")
    if not isinstance(evaluation_of, dict) or not isinstance(evaluation, dict):
        raise ValueError(f"durable evaluation child audit contract is incomplete: {root}")
    training_state = evaluation_of.get("training_state")
    episode_metrics = evaluation.get("episode_metrics")
    if not isinstance(training_state, dict) or not isinstance(episode_metrics, list):
        raise ValueError(f"durable evaluation child audit payload is incomplete: {root}")
    audit_seeds = [item.get("reset_seed") if isinstance(item, dict) else None for item in episode_metrics]
    simulator_provenance = audit.get("simulator_provenance")
    simulator_semantics = expected_contract["simulator_semantics"]
    if not isinstance(simulator_provenance, dict) or not isinstance(simulator_semantics, dict):
        raise ValueError(f"durable evaluation child simulator provenance is invalid: {root}")
    if (
        evaluation_of.get("checkpoint_id") != checkpoint_id
        or evaluation_of.get("manifest_sha256") != expected_contract["checkpoint_manifest_sha256"]
        or training_state.get("environment_steps") != result["source_environment_steps"]
        or training_state.get("policy_version") != result["source_policy_version"]
        or evaluation.get("episodes") != expected_contract["episodes"]
        or evaluation.get("base_seed") != expected_contract["resolved_base_seed"]
        or audit_seeds != expected_contract["held_out_seeds"]
        or evaluation.get("summary") != result["summary"]
        or audit.get("policy_source") != "network.pt"
        or audit.get("learner_updates_performed") != 0
        or simulator_provenance.get("verified_identity") != simulator_semantics.get("verified_identity")
        or simulator_provenance.get("runtime_mechanics") != simulator_semantics.get("runtime_mechanics")
    ):
        raise ValueError(f"durable evaluation child audit semantics mismatch: {root}")
    return manifest, result


def _publish_durable_child(
    staging: Path,
    destination: Path,
    *,
    contract: dict[str, object],
    result: FrozenEvaluationResult,
) -> tuple[dict[str, object], dict[str, object], bool]:
    """Publish a completed child cache entry with its own atomic boundary."""

    checkpoint_id = str(result.checkpoint_id)
    if checkpoint_id != contract["checkpoint_id"]:
        raise RuntimeError("evaluation resolved a different checkpoint_id")
    audit_path = staging / "evaluation.json"
    audit = json.loads(audit_path.read_text(encoding="utf-8"))
    evaluation = audit.get("evaluation")
    if not isinstance(evaluation, dict):
        raise ValueError("frozen evaluation audit has no evaluation object")
    evaluation["journal"] = str(destination / "trajectory.jsonl")
    audit["published_output_directory"] = str(destination)
    _atomic_write_json(audit_path, audit)
    cache_key = _mapping_fingerprint(contract)
    result_payload: dict[str, object] = {
        "checkpoint_id": checkpoint_id,
        "source_environment_steps": int(result.source_environment_steps),
        "source_policy_version": int(result.source_policy_version),
        "summary": dict(result.summary),
    }
    manifest: dict[str, object] = {
        "schema_version": _DURABLE_CHILD_SCHEMA,
        "status": "complete",
        "created_unix_s": time.time(),
        "publication_contract": "atomic-directory-rename-v1",
        "cache_key": cache_key,
        "contract": contract,
        "result": result_payload,
        "files": [
            _durable_file_record(staging, "evaluation.json"),
            _durable_file_record(staging, "trajectory.jsonl"),
        ],
    }
    _atomic_write_json(staging / _DURABLE_CHILD_MANIFEST, manifest)
    try:
        atomic_publish_directory(staging, destination)
    except OSError:
        # A concurrent identical evaluator may have won the publication race.
        # Never replace it: validate the winner before discarding our copy.
        if not destination.exists():
            raise
        winner = _validate_durable_child(destination, expected_contract=contract)
        shutil.rmtree(staging, ignore_errors=True)
        return winner[0], winner[1], True
    return manifest, result_payload, False


def _clone_durable_child(
    source: Path,
    destination: Path,
    *,
    expected_files: dict[str, dict[str, object]],
) -> None:
    if destination.exists():
        raise FileExistsError(f"paired child staging output already exists: {destination}")
    destination.mkdir(parents=False, exist_ok=False)
    # Formal paired evidence must never share mutable inodes with its reusable
    # cache source.  Copy exactly the two validated payloads rather than cloning
    # the whole directory (which could also propagate an unexpected entry).
    try:
        for name in ("evaluation.json", "trajectory.jsonl"):
            shutil.copy2(source / name, destination / name)
            if _durable_file_record(destination, name) != expected_files[name]:
                raise ValueError(f"durable evaluation child changed while copying {name}: {source}")
    except BaseException:
        shutil.rmtree(destination, ignore_errors=True)
        raise


def _validated_clone_durable_child(
    source: Path,
    destination: Path,
    *,
    expected_contract: dict[str, object],
) -> tuple[dict[str, object], dict[str, object]]:
    """Revalidate a cache child immediately before producing an isolated copy."""

    manifest, result = _validate_durable_child(
        source,
        expected_contract=expected_contract,
    )
    _clone_durable_child(
        source,
        destination,
        expected_files=_durable_manifest_file_records(manifest, root=source),
    )
    return manifest, result


def _require_disjoint_evaluation_roots(output_root: Path, durable_cache_root: Path) -> None:
    """Reject equal or nested cache/publication roots before creating either."""

    resolved_output = output_root.resolve()
    resolved_cache = durable_cache_root.resolve()
    if (
        resolved_output == resolved_cache
        or resolved_output.is_relative_to(resolved_cache)
        or resolved_cache.is_relative_to(resolved_output)
    ):
        raise ValueError("--durable-cache-root and --output-root must be disjoint and may not contain one another")


def _evaluate_durable_child_request(
    request: dict[str, object],
) -> dict[str, object]:
    """Evaluate and atomically cache one checkpoint in an isolated worker."""

    cache_staging: Path | None = None
    checkpoint_id = str(request.get("checkpoint_id", ""))
    try:
        contract = request.get("contract")
        simulator_provenance = request.get("simulator_provenance")
        if not isinstance(contract, dict) or not isinstance(simulator_provenance, dict):
            raise TypeError("durable child worker request has invalid mappings")
        expected_implementation = contract.get("evaluation_implementation_sha256")
        expected_runtime_environment = contract.get("runtime_environment")
        if not isinstance(expected_runtime_environment, dict):
            raise TypeError("durable child contract has no runtime environment mapping")
        if _evaluation_implementation_fingerprint() != expected_implementation:
            raise RuntimeError("evaluation implementation changed between parent planning and child startup")
        if _runtime_environment_contract() != expected_runtime_environment:
            raise RuntimeError("checkpoint worker runtime environment differs from the parent cache contract")
        checkpoint = Path(str(request["checkpoint"]))
        cache_output = Path(str(request["cache_output"]))
        device = str(request["device"])
        collector_device = str(request["collector_device"])
        requested_devices = (device, collector_device)
        if any(_resolved_device_name(item).startswith("cuda") for item in requested_devices):
            if not torch.cuda.is_available():
                raise RuntimeError("CUDA/ROCm evaluation requested but unavailable in child")
            torch.cuda.init()  # type: ignore[no-untyped-call]
        if cache_output.exists():
            _validate_durable_child(cache_output, expected_contract=contract)
            return {
                "status": "complete",
                "checkpoint_id": checkpoint_id,
                "cache_output": str(cache_output),
                "cache_hit": True,
            }
        episodes = request["episodes"]
        resolved_base_seed = request["resolved_base_seed"]
        if isinstance(episodes, bool) or not isinstance(episodes, int):
            raise TypeError("durable child worker episodes must be an integer")
        if isinstance(resolved_base_seed, bool) or not isinstance(resolved_base_seed, int):
            raise TypeError("durable child worker base seed must be an integer")
        cache_staging = cache_output.parent / f".s-{uuid4().hex}"
        result = evaluate_checkpoint_policy(
            checkpoint,
            output_directory=cache_staging,
            episodes=episodes,
            base_seed=resolved_base_seed,
            device=device,
            collector_device=collector_device,
            sim_exe_path=str(request["sim_exe_path"]),
            simulator_provenance=simulator_provenance,
        )
        if _evaluation_implementation_fingerprint() != expected_implementation:
            raise RuntimeError("evaluation implementation changed while checkpoint evaluation was running")
        if _runtime_environment_contract() != expected_runtime_environment:
            raise RuntimeError("checkpoint worker runtime environment changed while evaluation was running")
        _, _, raced_cache_hit = _publish_durable_child(
            cache_staging,
            cache_output,
            contract=contract,
            result=result,
        )
        return {
            "status": "complete",
            "checkpoint_id": checkpoint_id,
            "cache_output": str(cache_output),
            "cache_hit": raced_cache_hit,
        }
    except BaseException as exc:
        return {
            "status": "failed",
            "checkpoint_id": checkpoint_id,
            "exception_type": f"{type(exc).__module__}.{type(exc).__qualname__}",
            "error": str(exc),
            "fingerprint": str(getattr(exc, "incident_fingerprint", None) or getattr(exc, "fingerprint", None) or ""),
            "traceback": traceback.format_exc(limit=40),
        }
    finally:
        if cache_staging is not None:
            shutil.rmtree(cache_staging, ignore_errors=True)


def _create_checkpoint_executor(max_workers: int) -> ProcessPoolExecutor:
    return ProcessPoolExecutor(
        max_workers=max_workers,
        mp_context=get_context("spawn"),
    )


def _execute_durable_child_requests(
    requests: Sequence[dict[str, object]],
    *,
    checkpoint_workers: int,
) -> list[dict[str, object]]:
    """Run missing checkpoint children, waiting for all useful cache work."""

    if not requests:
        return []
    workers = min(int(checkpoint_workers), len(requests))
    if workers <= 1:
        return [_evaluate_durable_child_request(request) for request in requests]
    results: list[dict[str, object]] = []
    with _create_checkpoint_executor(workers) as executor:
        futures = {
            executor.submit(_evaluate_durable_child_request, request): str(request.get("checkpoint_id", ""))
            for request in requests
        }
        for future in as_completed(futures):
            checkpoint_id = futures[future]
            try:
                results.append(future.result())
            except BaseException as exc:
                # Native worker death or an executor transport failure cannot
                # produce a trusted child. Other workers are still allowed to
                # finish so their atomic caches remain reusable on retry.
                results.append(
                    {
                        "status": "failed",
                        "checkpoint_id": checkpoint_id,
                        "exception_type": (f"{type(exc).__module__}.{type(exc).__qualname__}"),
                        "error": str(exc),
                        "fingerprint": "",
                        "traceback": traceback.format_exc(limit=40),
                    }
                )
    return results


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m sts2_rl.evaluate_checkpoint",
        description=(
            "Evaluate network.pt from an atomic checkpoint without loading or "
            "consuming optimizer, rollout queue, training RNG, or counters."
        ),
    )
    parser.add_argument(
        "--checkpoint",
        action="append",
        required=True,
        help="atomic checkpoint directory; repeat for paired evaluation",
    )
    parser.add_argument("--episodes", type=int, default=12)
    parser.add_argument("--base-seed", type=int)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--collector-device", default="cpu")
    parser.add_argument("--sim-exe", help="override checkpoint simulator executable path")
    parser.add_argument("--sim-identity", help="simulator identity sidecar")
    parser.add_argument(
        "--output-root",
        help="artifact directory (default: evaluations/frozen-checkpoints)",
    )
    parser.add_argument(
        "--durable-cache-root",
        help=("durable completed-checkpoint cache (default: a shared hidden " "directory beside --output-root)"),
    )
    parser.add_argument(
        "--checkpoint-workers",
        type=int,
        default=1,
        help=(
            "independent checkpoint evaluator processes (default: 1; use 2 " "for a validated CPU paired evaluation)"
        ),
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.episodes <= 0:
        raise SystemExit("--episodes must be positive")
    if args.checkpoint_workers <= 0:
        raise SystemExit("--checkpoint-workers must be positive")
    requested_devices = (str(args.device), str(args.collector_device))
    wants_cuda = any(
        (device.strip().lower() == "auto" and torch.cuda.is_available())
        or (device.strip().lower() != "auto" and torch.device(device).type == "cuda")
        for device in requested_devices
    )
    if args.checkpoint_workers > 1 and wants_cuda:
        raise SystemExit(
            "parallel checkpoint evaluation requires --device cpu and "
            "--collector-device cpu; evaluators never share CUDA state"
        )
    if wants_cuda:
        if not torch.cuda.is_available():
            raise SystemExit("CUDA/ROCm evaluation requested but unavailable")
        # The CLI is an isolated process, so initialize before entering the
        # evaluator's RNG-preservation boundary.  Direct in-process callers
        # must do this themselves or remain on CPU.
        torch.cuda.init()  # type: ignore[no-untyped-call]
    checkpoints = [resolve_external_input_path(item) for item in args.checkpoint]
    first_config = checkpoint_training_config(checkpoints[0])
    configured_sim = args.sim_exe or first_config.environment.sim_exe_path
    try:
        simulator = verify_headless_simulator(
            resolve_headless_sim_exe(configured_sim),
            identity_path=args.sim_identity,
        )
        simulator_audit = write_preflight_audit(simulator)
        mechanics = run_runtime_mechanics_preflight(simulator.executable)
        mechanics_audit = write_runtime_mechanics_audit(mechanics)
    except (
        HeadlessSimError,
        OSError,
        RuntimeMechanicsAuditError,
        SimulatorIdentityError,
        ValueError,
    ) as exc:
        raise SystemExit(f"HeadlessSim preflight failed: {exc}") from exc

    output_root = resolve_artifact_path(
        args.output_root,
        default="evaluations/frozen-checkpoints",
    )
    output_root.parent.mkdir(parents=True, exist_ok=True)
    durable_cache_root = (
        resolve_artifact_path(args.durable_cache_root)
        if args.durable_cache_root is not None
        else output_root.parent / ".paired-checkpoint-cache-v1"
    )
    try:
        _require_disjoint_evaluation_roots(output_root, durable_cache_root)
    except ValueError as exc:
        raise SystemExit(str(exc)) from exc
    if durable_cache_root.exists() and not durable_cache_root.is_dir():
        raise SystemExit(f"durable cache root is not a directory: {durable_cache_root}")
    pair_manifest_path = output_root / "paired-evaluation.manifest.json"
    legacy_invalid_manifest = output_root / "paired-evaluation.invalid.json"
    if output_root.exists():
        if legacy_invalid_manifest.is_file() and not pair_manifest_path.exists():
            # v1 wrote partial results and its invalid marker *inside* the
            # requested publication directory.  Preserve that evidence while
            # freeing the atomic target for a clean retry.
            archived = output_root.with_name(f".{output_root.name}.failed-v1-{time.time_ns()}-{uuid4().hex}")
            atomic_publish_directory(output_root, archived)
        else:
            raise SystemExit("paired evaluation output already exists")
    invalid_manifest_path = output_root.with_name(f"{output_root.name}.paired-evaluation.invalid.json")
    staging_root = output_root.with_name(f".{output_root.name}.paired-staging-{uuid4().hex}")

    lineage = first_config.lineage_mapping()
    lineage_fingerprint = _lineage_fingerprint(lineage)
    resolved_base_seed = first_config.runtime.seed if args.base_seed is None else int(args.base_seed)
    evaluation_seeds = held_out_evaluation_seeds(resolved_base_seed, args.episodes)
    checkpoint_records: list[tuple[Path, str, str]] = []
    checkpoint_ids: set[str] = set()
    for checkpoint in checkpoints:
        config = checkpoint_training_config(checkpoint)
        if config.lineage_mapping() != lineage:
            shutil.rmtree(staging_root, ignore_errors=True)
            raise ValueError("paired checkpoints do not share identical immutable training semantics")
        validated = validate_resume_checkpoint(checkpoint)
        checkpoint_id = validated.manifest.get("checkpoint_id")
        if not isinstance(checkpoint_id, str) or not checkpoint_id:
            shutil.rmtree(staging_root, ignore_errors=True)
            raise ValueError(f"checkpoint has no stable checkpoint_id: {checkpoint}")
        if checkpoint_id in checkpoint_ids:
            shutil.rmtree(staging_root, ignore_errors=True)
            raise ValueError(f"paired checkpoint_id is duplicated: {checkpoint_id}")
        checkpoint_ids.add(checkpoint_id)
        checkpoint_records.append(
            (
                validated.root,
                checkpoint_id,
                _sha256(validated.root / "checkpoint.manifest.json"),
            )
        )
    simulator_provenance = {
        "verified_identity": simulator.to_mapping(),
        "runtime_mechanics": mechanics,
        "identity_audit_path": str(simulator_audit),
        "identity_audit_sha256": _sha256(Path(simulator_audit)),
        "runtime_mechanics_audit_path": str(mechanics_audit),
        "runtime_mechanics_audit_sha256": _sha256(Path(mechanics_audit)),
    }
    simulator_semantics: dict[str, object] = {
        "verified_identity": simulator.to_mapping(),
        "runtime_mechanics": mechanics,
    }
    implementation_sha256 = _evaluation_implementation_fingerprint()
    runtime_environment = _runtime_environment_contract()
    durable_cache_root.mkdir(parents=True, exist_ok=True)
    staging_root.mkdir(parents=False, exist_ok=False)
    staging_pair_manifest_path = staging_root / "paired-evaluation.manifest.json"
    outputs: list[dict[str, object]] = []
    child_worker_failures: list[dict[str, object]] = []
    started = time.time()
    try:
        child_plans: list[dict[str, object]] = []
        ready_children: dict[
            str,
            tuple[dict[str, object], dict[str, object], bool],
        ] = {}
        missing_requests: list[dict[str, object]] = []
        for checkpoint, checkpoint_id, manifest_sha256 in checkpoint_records:
            cache_contract = _durable_child_contract(
                checkpoint_id=checkpoint_id,
                checkpoint_manifest_sha256=manifest_sha256,
                lineage_sha256=lineage_fingerprint,
                simulator_semantics=simulator_semantics,
                implementation_sha256=implementation_sha256,
                runtime_environment=runtime_environment,
                episodes=args.episodes,
                resolved_base_seed=resolved_base_seed,
                evaluation_seeds=evaluation_seeds,
                device=args.device,
                collector_device=args.collector_device,
            )
            cache_key = _mapping_fingerprint(cache_contract)
            cache_output = durable_cache_root / f"child-{cache_key}"
            child_plans.append(
                {
                    "checkpoint": checkpoint,
                    "checkpoint_id": checkpoint_id,
                    "checkpoint_manifest_sha256": manifest_sha256,
                    "cache_contract": cache_contract,
                    "cache_key": cache_key,
                    "cache_output": cache_output,
                }
            )
            if cache_output.exists():
                cache_manifest, cached_result = _validate_durable_child(
                    cache_output,
                    expected_contract=cache_contract,
                )
                ready_children[checkpoint_id] = (
                    cache_manifest,
                    cached_result,
                    True,
                )
            else:
                missing_requests.append(
                    {
                        "checkpoint": str(checkpoint),
                        "checkpoint_id": checkpoint_id,
                        "cache_output": str(cache_output),
                        "contract": cache_contract,
                        "episodes": args.episodes,
                        "resolved_base_seed": resolved_base_seed,
                        "device": str(args.device),
                        "collector_device": str(args.collector_device),
                        "sim_exe_path": str(simulator.executable),
                        "simulator_provenance": simulator_provenance,
                    }
                )

        worker_results = _execute_durable_child_requests(
            missing_requests,
            checkpoint_workers=args.checkpoint_workers,
        )
        worker_results_by_id: dict[str, dict[str, object]] = {}
        for worker_result in worker_results:
            checkpoint_id = str(worker_result.get("checkpoint_id", ""))
            if not checkpoint_id or checkpoint_id in worker_results_by_id:
                child_worker_failures.append(
                    {
                        "status": "failed",
                        "checkpoint_id": checkpoint_id,
                        "exception_type": "sts2_rl.evaluate_checkpoint.WorkerResultError",
                        "error": "duplicate or missing checkpoint_id in child worker result",
                    }
                )
                continue
            worker_results_by_id[checkpoint_id] = worker_result
            if worker_result.get("status") != "complete":
                child_worker_failures.append(worker_result)

        requested_missing_ids = {str(item["checkpoint_id"]) for item in missing_requests}
        absent_worker_ids = requested_missing_ids - set(worker_results_by_id)
        child_worker_failures.extend(
            {
                "status": "failed",
                "checkpoint_id": checkpoint_id,
                "exception_type": "sts2_rl.evaluate_checkpoint.WorkerResultError",
                "error": "checkpoint worker produced no result",
            }
            for checkpoint_id in sorted(absent_worker_ids)
        )
        if child_worker_failures:
            raise RuntimeError(
                "one or more durable checkpoint workers failed: "
                + json.dumps(
                    child_worker_failures,
                    ensure_ascii=False,
                    sort_keys=True,
                    default=str,
                )
            )

        for plan in child_plans:
            checkpoint_id = str(plan["checkpoint_id"])
            if checkpoint_id in ready_children:
                continue
            planned_cache_output = plan["cache_output"]
            planned_cache_contract = plan["cache_contract"]
            if not isinstance(planned_cache_output, Path) or not isinstance(planned_cache_contract, dict):
                raise TypeError("internal durable child plan is invalid")
            worker_result = worker_results_by_id[checkpoint_id]
            cache_manifest, cached_result = _validate_durable_child(
                planned_cache_output,
                expected_contract=planned_cache_contract,
            )
            ready_children[checkpoint_id] = (
                cache_manifest,
                cached_result,
                worker_result.get("cache_hit") is True,
            )

        for checkpoint_index, plan in enumerate(child_plans):
            planned_checkpoint = plan["checkpoint"]
            checkpoint_id = str(plan["checkpoint_id"])
            manifest_sha256 = str(plan["checkpoint_manifest_sha256"])
            cache_key = str(plan["cache_key"])
            planned_cache_output = plan["cache_output"]
            planned_cache_contract = plan["cache_contract"]
            if (
                not isinstance(planned_checkpoint, Path)
                or not isinstance(planned_cache_output, Path)
                or not isinstance(planned_cache_contract, dict)
            ):
                raise TypeError("internal durable child path plan is invalid")
            checkpoint = planned_checkpoint
            cache_output = planned_cache_output
            _, _, cache_hit = ready_children[checkpoint_id]
            # Keep pair-level children short until the complete pair is ready;
            # the final descriptive directory is still published atomically.
            staging_output = staging_root / f"c{checkpoint_index}"
            staged_checkpoint_output = staging_root / f"checkpoint-{checkpoint_id}"
            published_output = output_root / f"checkpoint-{checkpoint_id}"
            cache_manifest, cached_result = _validated_clone_durable_child(
                cache_output,
                staging_output,
                expected_contract=planned_cache_contract,
            )
            atomic_publish_directory(staging_output, staged_checkpoint_output)

            # The evaluator atomically published inside our pair staging root.
            # Rewrite only diagnostic publication paths before the entire pair
            # directory is atomically renamed to its final location.
            staged_audit = staged_checkpoint_output / "evaluation.json"
            staged_journal = staged_checkpoint_output / "trajectory.jsonl"
            audit = json.loads(staged_audit.read_text(encoding="utf-8"))
            evaluation = audit.get("evaluation")
            if not isinstance(evaluation, dict):
                raise ValueError("frozen evaluation audit has no evaluation object")
            evaluation["journal"] = str(published_output / "trajectory.jsonl")
            audit["published_output_directory"] = str(published_output)
            _atomic_write_json(staged_audit, audit)
            published_audit = published_output / "evaluation.json"
            published_journal = published_output / "trajectory.jsonl"
            outputs.append(
                {
                    "checkpoint": str(checkpoint),
                    "checkpoint_id": checkpoint_id,
                    "source_environment_steps": cached_result["source_environment_steps"],
                    "source_policy_version": cached_result["source_policy_version"],
                    "summary": cached_result["summary"],
                    "checkpoint_manifest_sha256": manifest_sha256,
                    "output_directory": str(published_output),
                    "audit_path": str(published_audit),
                    "audit_sha256": _sha256(staged_audit),
                    "journal_path": str(published_journal),
                    "journal_sha256": _sha256(staged_journal),
                    "durable_cache": {
                        "schema_version": _DURABLE_CHILD_SCHEMA,
                        "cache_key": cache_key,
                        "cache_hit": cache_hit,
                        "cache_directory": str(cache_output),
                        "manifest_sha256": _sha256(cache_output / _DURABLE_CHILD_MANIFEST),
                        "created_unix_s": cache_manifest.get("created_unix_s"),
                    },
                }
            )
        deltas: dict[str, float | int] = {}
        if len(outputs) == 2:
            first_summary = outputs[0]["summary"]
            second_summary = outputs[1]["summary"]
            if isinstance(first_summary, dict) and isinstance(second_summary, dict):
                for key in sorted(first_summary.keys() & second_summary.keys()):
                    left = first_summary[key]
                    right = second_summary[key]
                    if (
                        isinstance(left, int | float)
                        and not isinstance(left, bool)
                        and isinstance(right, int | float)
                        and not isinstance(right, bool)
                    ):
                        deltas[key] = right - left
        _atomic_write_json(
            staging_pair_manifest_path,
            {
                "schema_version": _PAIRED_EVALUATION_SCHEMA,
                "status": "complete",
                "created_unix_s": time.time(),
                "duration_s": time.time() - started,
                "publication_contract": "atomic-directory-rename-v1",
                "episodes_per_checkpoint": args.episodes,
                "checkpoint_workers": args.checkpoint_workers,
                "requested_base_seed": args.base_seed,
                "resolved_base_seed": resolved_base_seed,
                "seed_contract": "even-training/odd-held-out-v1",
                "held_out_seeds": list(evaluation_seeds),
                "lineage": {
                    "sha256": lineage_fingerprint,
                    "config": lineage,
                },
                "simulator_provenance": simulator_provenance,
                "durable_child_cache": {
                    "schema_version": _DURABLE_CHILD_SCHEMA,
                    "root": str(durable_cache_root),
                    "implementation_sha256": implementation_sha256,
                    "runtime_environment": runtime_environment,
                },
                "results": outputs,
                "second_minus_first": deltas,
            },
        )
        invalid_manifest_path.unlink(missing_ok=True)
        atomic_publish_directory(staging_root, output_root)
    except BaseException as exc:
        shutil.rmtree(staging_root, ignore_errors=True)
        try:
            _atomic_write_json(
                invalid_manifest_path,
                {
                    "schema_version": _PAIRED_EVALUATION_SCHEMA,
                    "status": "infrastructure_or_validation_invalid",
                    "created_unix_s": time.time(),
                    "duration_s": time.time() - started,
                    "requested_output_root": str(output_root),
                    "atomic_output_published": False,
                    "staging_cleaned": not staging_root.exists(),
                    "checkpoints": [str(item) for item in checkpoints],
                    "checkpoint_workers": args.checkpoint_workers,
                    "durable_child_worker_failures": child_worker_failures,
                    "discarded_completed_results": outputs,
                    "requested_base_seed": args.base_seed,
                    "resolved_base_seed": resolved_base_seed,
                    "held_out_seeds": list(evaluation_seeds),
                    "lineage_sha256": lineage_fingerprint,
                    "simulator_provenance": simulator_provenance,
                    "durable_child_cache": {
                        "schema_version": _DURABLE_CHILD_SCHEMA,
                        "root": str(durable_cache_root),
                        "implementation_sha256": implementation_sha256,
                        "runtime_environment": runtime_environment,
                    },
                    "exception_type": f"{type(exc).__module__}.{type(exc).__qualname__}",
                    "error": str(exc),
                },
            )
        except OSError as audit_exc:
            exc.add_note(f"failed to publish invalid evaluation audit: {audit_exc}")
        raise
    print(
        json.dumps(
            {
                "status": "complete",
                "simulator_identity_audit": str(simulator_audit),
                "runtime_mechanics_audit": str(mechanics_audit),
                "paired_evaluation_manifest": str(pair_manifest_path),
                "results": outputs,
            },
            sort_keys=True,
            default=str,
        )
    )
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
