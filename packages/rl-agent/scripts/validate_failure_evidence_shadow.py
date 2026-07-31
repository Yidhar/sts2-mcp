#!/usr/bin/env python3
"""Run the formal failure-credit pipeline in read-only shadow mode.

The command evaluates the pinned v28 100k network without learner updates.  It
enables the complete semantics/detector/compiler path, discards rollout
unrolls, never creates a replay, and writes one atomic JSON audit.  This is the
runtime counterpart to ``validate_semantics_shadow.py``: the historical audit
proves broad DTO coverage, while this probe proves that detector-time evidence
can be compiled from live accepted transitions using the frozen policy.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import pickle
import tempfile
import time
from collections import Counter
from dataclasses import asdict, replace
from pathlib import Path
from typing import Any
from uuid import uuid4

import torch

from sts2_rl.artifacts import resolve_artifact_path
from sts2_rl.checkpoints import (
    V28_100K_FROZEN,
    validate_frozen_checkpoint,
)
from sts2_rl.training.checkpoint_evaluation import checkpoint_training_config
from sts2_rl.training.checkpointing import initialize_model_from_checkpoint
from sts2_rl.training.factory import build_training_resources
from sts2_rl.training.failure_credit import (
    FAILURE_CREDIT_COLLECTOR_VERSION,
    FAILURE_CREDIT_COMPILER_VERSION,
    FAILURE_CREDIT_DETECTOR_VERSION,
    FAILURE_CREDIT_SCHEMA_VERSION,
    FAILURE_EVIDENCE_REPLAY_VERSION,
    DirectPolicyTarget,
    EvidenceRecord,
    EvidenceStratum,
    ImmutableEvidenceCorpus,
)
from sts2_rl.training.launch_contract import current_formal_report_generation_source

_REPORT_VERSION = "sts2-failure-evidence-shadow-report-v2"
_CHECKOUT_ROOT = Path(__file__).resolve().parents[3]


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _atomic_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.",
        suffix=".tmp",
        dir=path.parent,
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            json.dump(
                payload,
                handle,
                allow_nan=False,
                ensure_ascii=False,
                indent=2,
                sort_keys=True,
            )
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def _model_state_sha256(model: torch.nn.Module) -> str:
    """Hash all model tensors without depending on pickle serialization."""

    digest = hashlib.sha256()
    for name, tensor in sorted(model.state_dict().items()):
        value = tensor.detach().cpu().contiguous()
        digest.update(name.encode("utf-8"))
        digest.update(b"\0")
        digest.update(str(value.dtype).encode("ascii"))
        digest.update(b"\0")
        digest.update(",".join(str(dimension) for dimension in value.shape).encode("ascii"))
        digest.update(b"\0")
        digest.update(value.numpy().tobytes(order="C"))
    return digest.hexdigest()


def _state_sha256(payload: object) -> str:
    return hashlib.sha256(
        pickle.dumps(
            payload,
            protocol=pickle.HIGHEST_PROTOCOL,
        )
    ).hexdigest()


def _completion_staging_audit(
    episode_payloads: list[dict[str, Any]],
    *,
    maximum_controls: int,
    maximum_bytes: int,
) -> dict[str, int | bool]:
    """Audit the collector's durable completion reservoir per episode.

    Completion controls are intentionally bounded at the episode boundary,
    not across the whole shadow run.  Keeping this arithmetic in one pure
    helper makes the live gate independently regression-testable without
    constructing a simulator, model, replay, or optimizer.
    """

    if maximum_controls <= 0 or maximum_bytes <= 0:
        raise ValueError("completion staging bounds must be positive")
    observed_total = 0
    dropped_total = 0
    maximum_storage = 0
    bounded = True
    for payload in episode_payloads:
        shadow_payload = payload.get("shadow")
        if not isinstance(shadow_payload, dict):
            raise RuntimeError("shadow audit payload is malformed")
        values: dict[str, int] = {}
        for key in (
            "completion_controls_observed",
            "completion_controls_dropped",
            "completion_storage_nbytes",
        ):
            raw = shadow_payload.get(key)
            if isinstance(raw, bool) or not isinstance(raw, int):
                raise RuntimeError(f"shadow audit {key} must be an integer")
            values[key] = raw
        observed = values["completion_controls_observed"]
        dropped = values["completion_controls_dropped"]
        storage_nbytes = values["completion_storage_nbytes"]
        retained = observed - dropped
        observed_total += observed
        dropped_total += dropped
        maximum_storage = max(maximum_storage, storage_nbytes)
        bounded = bounded and (
            0 <= dropped <= observed and retained <= maximum_controls and 0 <= storage_nbytes <= maximum_bytes
        )
    return {
        "bounded": bounded,
        "observed": observed_total,
        "dropped": dropped_total,
        "maximum_episode_storage_nbytes": maximum_storage,
    }


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--artifact-root", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--episodes", type=int, default=4)
    parser.add_argument("--maximum-steps", type=int, default=2_000)
    parser.add_argument("--base-seed", type=int, default=3_500_000)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--collector-device", default="cuda")
    parser.add_argument("--sim-exe-path")
    parser.add_argument(
        "--minimum-decisions",
        type=int,
        default=100,
    )
    return parser


def main() -> int:
    args = _parser().parse_args()
    generation_source = current_formal_report_generation_source(_CHECKOUT_ROOT, __file__)
    if args.episodes <= 0:
        raise ValueError("--episodes must be positive")
    if args.maximum_steps <= 0:
        raise ValueError("--maximum-steps must be positive")
    if args.minimum_decisions <= 0:
        raise ValueError("--minimum-decisions must be positive")

    artifact_root = args.artifact_root.expanduser().resolve()
    report_path = resolve_artifact_path(args.report, root=artifact_root)
    frozen = validate_frozen_checkpoint(artifact_root)
    checkpoint = frozen.root
    manifest_path = checkpoint / "checkpoint.manifest.json"
    metadata_path = checkpoint / "metadata.json"
    manifest_hash_before = _sha256(manifest_path)
    metadata_hash_before = _sha256(metadata_path)

    source = checkpoint_training_config(checkpoint)
    environment = source.environment
    if args.sim_exe_path is not None:
        environment = replace(
            environment,
            sim_exe_path=str(args.sim_exe_path),
        )
    config = replace(
        source,
        failure_credit=replace(source.failure_credit, mode="shadow"),
        environment=environment,
        runtime=replace(
            source.runtime,
            device=str(args.device),
            collector_device=str(args.collector_device),
            seed=int(args.base_seed),
            evaluation_steps=(),
            evaluation_episodes=0,
        ),
    )
    if config.failure_credit.learning_enabled:
        raise RuntimeError("shadow validation unexpectedly enabled learning")

    started = time.time()
    run_id = f"failure-credit-shadow-{uuid4()}"
    records: list[EvidenceRecord] = []
    episode_payloads: list[dict[str, Any]] = []
    resources = build_training_resources(config)
    try:
        initialize_model_from_checkpoint(
            checkpoint,
            config=config,
            resources=resources,
        )
        if resources.failure_credit_replay is not None:
            raise RuntimeError("shadow validation must not construct replay")
        if resources.model.liveness_head_enabled:
            raise RuntimeError("shadow validation must not construct new heads")
        model_hash_before = _model_state_sha256(resources.model)
        collector_model_hash_before = _model_state_sha256(resources.collector_model)
        optimizer_state_count_before = len(resources.optimizer.state)
        rollout_count_before = len(resources.rollout_queue)
        transaction_replay_hash_before = (
            _state_sha256(resources.transaction_replay.state_dict())
            if resources.transaction_replay is not None
            else None
        )
        episodic_replay_hash_before = (
            _state_sha256(resources.episodic_replay.state_dict()) if resources.episodic_replay is not None else None
        )
        resources.collector.bind_failure_credit_run_id(run_id)
        for _ in range(args.episodes):
            episode = resources.collector.collect_episode(
                epsilon=0.0,
                deterministic=True,
                record=True,
                policy_version=V28_100K_FROZEN.policy_version,
                maximum_steps=args.maximum_steps,
                # The shadow validates evidence only. Never retain or learn a
                # rollout and never permit a policy-version adoption.
                unroll_sink=lambda _unroll: V28_100K_FROZEN.policy_version,
                liveness_probe=True,
            )
            records.extend(episode.failure_credit_records)
            shadow = episode.failure_credit_shadow_metrics
            if shadow is None:
                raise RuntimeError("shadow collector emitted no failure-credit funnel")
            episode_payloads.append(
                {
                    "episode": asdict(episode.metrics),
                    "shadow": asdict(shadow),
                    "record_count": len(episode.failure_credit_records),
                    "actor_label_count": sum(
                        record.plan.actor_label_count for record in episode.failure_credit_records
                    ),
                }
            )
        training_state_unchanged = {
            "model": (_model_state_sha256(resources.model) == model_hash_before),
            "collector_model": (_model_state_sha256(resources.collector_model) == collector_model_hash_before),
            "optimizer": (len(resources.optimizer.state) == optimizer_state_count_before == 0),
            "rollout_queue": (len(resources.rollout_queue) == rollout_count_before == 0),
            "transaction_replay": (
                transaction_replay_hash_before
                == (
                    _state_sha256(resources.transaction_replay.state_dict())
                    if resources.transaction_replay is not None
                    else None
                )
            ),
            "episodic_replay": (
                episodic_replay_hash_before
                == (
                    _state_sha256(resources.episodic_replay.state_dict())
                    if resources.episodic_replay is not None
                    else None
                )
            ),
        }
    finally:
        resources.close()

    corpus = ImmutableEvidenceCorpus(records=tuple(records))
    direct_targets = Counter(target.target.value for record in records for target in record.plan.direct_policy_targets)
    receipt_counts: Counter[str] = Counter()
    semantic_censored = 0
    decisions = 0
    completion_audit = _completion_staging_audit(
        episode_payloads,
        maximum_controls=(config.failure_credit.maximum_episode_completion_controls),
        maximum_bytes=config.failure_credit.maximum_episode_completion_bytes,
    )
    for payload in episode_payloads:
        shadow_payload = payload["shadow"]
        if not isinstance(shadow_payload, dict):
            raise RuntimeError("shadow audit payload is malformed")
        decisions += int(shadow_payload["decisions"])
        semantic_censored += int(shadow_payload["censored_semantic_transitions"])
        for kind, count in shadow_payload["progress_receipts"]:
            receipt_counts[str(kind)] += int(count)

    strata = {stratum.value: len(corpus.incident_ids(stratum)) for stratum in EvidenceStratum}
    gates = {
        "minimum_decisions": decisions >= args.minimum_decisions,
        "zero_semantic_censored_transitions": semantic_censored == 0,
        "no_unjustified_completion_prefer": (direct_targets[DirectPolicyTarget.PREFER.value] == 0),
        "source_manifest_unchanged": (_sha256(manifest_path) == manifest_hash_before),
        "source_metadata_unchanged": (_sha256(metadata_path) == metadata_hash_before),
        "no_replay_or_learner_updates": all(training_state_unchanged.values()),
        # This is the live proof for the completion-control staging repair.
        # It must be evaluated per episode: an aggregate byte count would
        # incorrectly reject several independently bounded episodes.
        "completion_staging_is_bounded": bool(completion_audit["bounded"]),
    }
    status = "passed" if all(gates.values()) else "failed"
    report: dict[str, Any] = {
        "version": _REPORT_VERSION,
        "status": status,
        "created_unix_s": time.time(),
        "duration_s": time.time() - started,
        "read_only": True,
        "training_authority": False,
        "source": {
            "checkpoint": str(checkpoint),
            "checkpoint_id": V28_100K_FROZEN.checkpoint_id,
            "environment_steps": V28_100K_FROZEN.environment_steps,
            "policy_version": V28_100K_FROZEN.policy_version,
            "manifest_sha256": manifest_hash_before,
            "metadata_sha256": metadata_hash_before,
        },
        "failure_credit_abi": {
            "collector": FAILURE_CREDIT_COLLECTOR_VERSION,
            "schema": FAILURE_CREDIT_SCHEMA_VERSION,
            "compiler": FAILURE_CREDIT_COMPILER_VERSION,
            "detector": FAILURE_CREDIT_DETECTOR_VERSION,
            "replay": FAILURE_EVIDENCE_REPLAY_VERSION,
        },
        "config": {
            "mode": config.failure_credit.mode,
            "fingerprint_sha256": config.fingerprint_sha256(),
            "episodes": args.episodes,
            "maximum_steps": args.maximum_steps,
            "base_seed": args.base_seed,
            "device": args.device,
            "collector_device": args.collector_device,
        },
        "counts": {
            "decisions": decisions,
            "records": len(records),
            "actor_actionable_records": (corpus.metrics().actor_actionable_records),
            "semantic_censored_transitions": semantic_censored,
            "storage_nbytes": corpus.storage_nbytes,
            "completion_controls_observed": completion_audit["observed"],
            "completion_controls_dropped": completion_audit["dropped"],
            "maximum_episode_completion_storage_nbytes": (completion_audit["maximum_episode_storage_nbytes"]),
        },
        "progress_receipts": dict(sorted(receipt_counts.items())),
        "direct_policy_targets": dict(sorted(direct_targets.items())),
        "strata": strata,
        "training_state_unchanged": training_state_unchanged,
        "gates": gates,
        "episodes": episode_payloads,
    }
    if current_formal_report_generation_source(_CHECKOUT_ROOT, __file__) != generation_source:
        raise RuntimeError("formal evidence source changed during report generation")
    report["generation_source"] = generation_source
    _atomic_json(report_path, report)
    print(
        json.dumps(
            {
                "status": status,
                "report": str(report_path),
                "decisions": decisions,
                "records": len(records),
                "semantic_censored_transitions": semantic_censored,
                "strata": strata,
            },
            sort_keys=True,
        )
    )
    return 0 if status == "passed" else 2


if __name__ == "__main__":
    raise SystemExit(main())
