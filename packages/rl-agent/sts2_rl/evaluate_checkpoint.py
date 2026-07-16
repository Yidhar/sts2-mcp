"""CLI for frozen, learner-free held-out checkpoint evaluation."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import time
from collections.abc import Sequence
from dataclasses import asdict
from pathlib import Path
from uuid import uuid4

import torch

from sts2_env.headless_sim_bridge_client import HeadlessSimError, resolve_headless_sim_exe

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
    atomic_publish_directory,
    checkpoint_training_config,
    evaluate_checkpoint_policy,
)
from .training.seeding import held_out_evaluation_seeds


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
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.episodes <= 0:
        raise SystemExit("--episodes must be positive")
    requested_devices = (str(args.device), str(args.collector_device))
    wants_cuda = any(
        (
            device.strip().lower() == "auto" and torch.cuda.is_available()
        )
        or (
            device.strip().lower() != "auto"
            and torch.device(device).type == "cuda"
        )
        for device in requested_devices
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
    pair_manifest_path = output_root / "paired-evaluation.manifest.json"
    legacy_invalid_manifest = output_root / "paired-evaluation.invalid.json"
    if output_root.exists():
        if legacy_invalid_manifest.is_file() and not pair_manifest_path.exists():
            # v1 wrote partial results and its invalid marker *inside* the
            # requested publication directory.  Preserve that evidence while
            # freeing the atomic target for a clean retry.
            archived = output_root.with_name(
                f".{output_root.name}.failed-v1-{time.time_ns()}-{uuid4().hex}"
            )
            atomic_publish_directory(output_root, archived)
        else:
            raise SystemExit("paired evaluation output already exists")
    invalid_manifest_path = output_root.with_name(
        f"{output_root.name}.paired-evaluation.invalid.json"
    )
    staging_root = output_root.with_name(
        f".{output_root.name}.paired-staging-{uuid4().hex}"
    )

    lineage = first_config.lineage_mapping()
    lineage_fingerprint = _lineage_fingerprint(lineage)
    resolved_base_seed = (
        first_config.runtime.seed if args.base_seed is None else int(args.base_seed)
    )
    evaluation_seeds = held_out_evaluation_seeds(resolved_base_seed, args.episodes)
    checkpoint_records: list[tuple[Path, str, str]] = []
    checkpoint_ids: set[str] = set()
    for checkpoint in checkpoints:
        config = checkpoint_training_config(checkpoint)
        if config.lineage_mapping() != lineage:
            shutil.rmtree(staging_root, ignore_errors=True)
            raise ValueError(
                "paired checkpoints do not share identical immutable training semantics"
            )
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
    staging_root.mkdir(parents=False, exist_ok=False)
    staging_pair_manifest_path = staging_root / "paired-evaluation.manifest.json"
    outputs: list[dict[str, object]] = []
    started = time.time()
    try:
        for checkpoint_index, (checkpoint, checkpoint_id, manifest_sha256) in enumerate(
            checkpoint_records
        ):
            # Keep the nested evaluator target deliberately short.  Its own
            # sibling staging suffix contains a UUID; combining that suffix
            # with the pair staging UUID and full checkpoint UUID can cross
            # the legacy Windows MAX_PATH boundary before trajectory.jsonl is
            # opened.  Rename to the descriptive checkpoint directory only
            # after the individual evaluation has published atomically.
            staging_output = staging_root / f"c{checkpoint_index}"
            staged_checkpoint_output = staging_root / f"checkpoint-{checkpoint_id}"
            published_output = output_root / f"checkpoint-{checkpoint_id}"
            result = evaluate_checkpoint_policy(
                checkpoint,
                output_directory=staging_output,
                episodes=args.episodes,
                base_seed=resolved_base_seed,
                device=args.device,
                collector_device=args.collector_device,
                sim_exe_path=str(simulator.executable),
                simulator_provenance=simulator_provenance,
            )
            if result.checkpoint_id != checkpoint_id:
                raise RuntimeError("evaluation resolved a different checkpoint_id")
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
                    **asdict(result),
                    "checkpoint": str(result.checkpoint),
                    "checkpoint_manifest_sha256": manifest_sha256,
                    "output_directory": str(published_output),
                    "audit_path": str(published_audit),
                    "audit_sha256": _sha256(staged_audit),
                    "journal_path": str(published_journal),
                    "journal_sha256": _sha256(staged_journal),
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
                "schema_version": "sts2-paired-frozen-evaluation-v2",
                "status": "complete",
                "created_unix_s": time.time(),
                "duration_s": time.time() - started,
                "publication_contract": "atomic-directory-rename-v1",
                "episodes_per_checkpoint": args.episodes,
                "requested_base_seed": args.base_seed,
                "resolved_base_seed": resolved_base_seed,
                "seed_contract": "even-training/odd-held-out-v1",
                "held_out_seeds": list(evaluation_seeds),
                "lineage": {
                    "sha256": lineage_fingerprint,
                    "config": lineage,
                },
                "simulator_provenance": simulator_provenance,
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
                    "schema_version": "sts2-paired-frozen-evaluation-v2",
                    "status": "infrastructure_or_validation_invalid",
                    "created_unix_s": time.time(),
                    "duration_s": time.time() - started,
                    "requested_output_root": str(output_root),
                    "atomic_output_published": False,
                    "staging_cleaned": not staging_root.exists(),
                    "checkpoints": [str(item) for item in checkpoints],
                    "discarded_completed_results": outputs,
                    "requested_base_seed": args.base_seed,
                    "resolved_base_seed": resolved_base_seed,
                    "held_out_seeds": list(evaluation_seeds),
                    "lineage_sha256": lineage_fingerprint,
                    "simulator_provenance": simulator_provenance,
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
