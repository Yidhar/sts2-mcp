#!/usr/bin/env python3
"""Prepare and run a replay-driven learner E2E kernel A/B benchmark.

The kernel-lab peak kernels are fixed-shape HIP microbenchmarks without a
PyTorch/autograd ABI.  The production-applicable finding in ``PLAYBOOK.md`` is
instead to remove device/host synchronisation and graph breaks from the real
small-model workload.  This benchmark measures that change on one immutable
checkpoint workload rather than extrapolating from peak FLOPs.

Preparation is split into two commands because each exact-resume replay
sidecar is roughly 1.5--1.7 GiB and expands to about 2.5 GiB while unpickling.
The resulting fixture is a compact, read-only sample and is not a checkpoint.

Examples::

    python scripts/benchmark_kernel_lab_e2e.py prepare-failure \
      --checkpoint /path/to/checkpoint --fixture-dir /path/to/fixture
    python scripts/benchmark_kernel_lab_e2e.py prepare-episode \
      --checkpoint /path/to/checkpoint --fixture-dir /path/to/fixture
    python scripts/benchmark_kernel_lab_e2e.py run \
      --checkpoint /path/to/checkpoint --fixture-dir /path/to/fixture \
      --mask-kernel synchronizing --iterations 5
    python scripts/benchmark_kernel_lab_e2e.py run \
      --checkpoint /path/to/checkpoint --fixture-dir /path/to/fixture \
      --mask-kernel branchless --iterations 5

Every ``run`` starts from the same network, optimizer, RNG seed, unrolls,
failure-credit plans and episodic sequences.  It reports the learner's own E2E
timer plus wall-clock and component timings.  No environment or training run
is started and the source checkpoint is never mutated.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import pickle
import statistics
import time
from collections.abc import Iterable
from dataclasses import replace
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch import Tensor

from sts2_baseline import RolloutStep, SequenceUnroll
from sts2_rl.artifacts import resolve_artifact_path, resolve_external_input_path
from sts2_rl.encoding import GroundedObservationEncoder
from sts2_rl.models import RecurrentCandidateModel
from sts2_rl.models import grounded_candidate as candidate_module
from sts2_rl.training.config import model_initialization_config_from_mapping
from sts2_rl.training.episode_replay import BoundedEpisodicReplay
from sts2_rl.training.failure_credit import ImmutableEvidenceCorpus
from sts2_rl.training.learner import LearnerMetrics, VTraceLearner
from sts2_rl.training.runtime import _failure_credit_quotas
from sts2_rl.training.sdpa import configure_rocm_sdpa_backend

_FIXTURE_VERSION = "sts2-kernel-lab-e2e-fixture-v1"
_FAILURE_FIXTURE = "failure_workload.pkl"
_EPISODE_FIXTURE = "episode_workload.pkl"


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    for name in ("prepare-failure", "prepare-episode"):
        command = subparsers.add_parser(name)
        command.add_argument("--checkpoint", type=Path, required=True)
        command.add_argument("--fixture-dir", type=Path, required=True)

    run = subparsers.add_parser("run")
    run.add_argument("--checkpoint", type=Path, required=True)
    run.add_argument("--fixture-dir", type=Path, required=True)
    run.add_argument("--device", default="cuda")
    run.add_argument(
        "--mask-kernel",
        choices=("synchronizing", "branchless"),
        required=True,
        help="old device-to-host branch or the production branchless kernel",
    )
    run.add_argument("--warmup", type=int, default=1)
    run.add_argument("--iterations", type=int, default=5)
    run.add_argument("--seed", type=int, default=20260804)
    run.add_argument(
        "--liveness-records-per-autograd-batch",
        type=int,
        help="execution-only record pack override for the frozen workload",
    )
    run.add_argument(
        "--maximum-device-memory-fraction",
        type=float,
        default=0.50,
        help="fail the promotion gate when peak reserved device memory exceeds this fraction",
    )
    run.add_argument("--output", type=Path)
    run.add_argument(
        "--state-output",
        type=Path,
        help="optional final CPU state_dict for numerical A/B comparison",
    )
    return parser.parse_args()


def _metadata(checkpoint: Path) -> dict[str, Any]:
    path = checkpoint / "metadata.json"
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError("checkpoint metadata must be an object")
    return payload


def _config_and_state(checkpoint: Path) -> tuple[Any, dict[str, Any]]:
    metadata = _metadata(checkpoint)
    raw_config = metadata.get("training_config")
    raw_state = metadata.get("training_state")
    if not isinstance(raw_config, dict) or not isinstance(raw_state, dict):
        raise ValueError("checkpoint metadata omits training config/state")
    # The benchmark is intentionally allowed to exercise a frozen V12
    # workload under the current V13 execution code.  This is model/workload
    # reconstruction, not exact resume; use the repository's reviewed V12 ->
    # V13 model-initialization parser rather than weakening the strict exact-
    # resume parser.
    return model_initialization_config_from_mapping(raw_config), raw_state


def _write_pickle(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("wb") as handle:
        pickle.dump(value, handle, protocol=pickle.HIGHEST_PROTOCOL)
    temporary.replace(path)


def _read_pickle(path: Path) -> object:
    with path.open("rb") as handle:
        return pickle.load(handle)


def _prepare_failure(checkpoint: Path, fixture_dir: Path) -> dict[str, Any]:
    config, state = _config_and_state(checkpoint)
    payload = _read_pickle(checkpoint / "failure_credit_replay.pkl")
    if not isinstance(payload, dict):
        raise TypeError("failure-credit replay payload must be an object")
    corpus_payload = payload.get("corpus")
    if not isinstance(corpus_payload, dict):
        raise TypeError("failure-credit replay corpus must be an object")
    corpus = ImmutableEvidenceCorpus.load_state_dict(corpus_payload)
    rng = np.random.default_rng()
    rng_state = payload.get("rng_state")
    if not isinstance(rng_state, dict):
        raise TypeError("failure-credit replay RNG state must be an object")
    rng.bit_generator.state = rng_state
    policy_version = int(state["policy_version"])
    learner_updates = int(state["learner_updates"])
    sample = corpus.sample(
        batch_size=config.failure_credit.sample_records,
        rng=rng,
        quotas=_failure_credit_quotas(config),
        current_policy_version=policy_version,
    )
    plans = tuple(record.plan for record in sample.records)
    fixture = {
        "version": _FIXTURE_VERSION,
        "checkpoint": str(checkpoint.resolve()),
        "policy_version": policy_version,
        "learner_updates": learner_updates,
        "plans": plans,
        "strata": tuple(tuple(item.value for item in plan.strata) for plan in plans),
        "context_steps": tuple(len(plan.context.steps) for plan in plans),
        "quota_satisfied": sample.quota_diagnostics.satisfied,
        "quota_deficit": sample.quota_diagnostics.total_deficit,
    }
    output = fixture_dir / _FAILURE_FIXTURE
    _write_pickle(output, fixture)
    return {
        "output": str(output.resolve()),
        "records": len(plans),
        "strata": fixture["strata"],
        "context_steps": fixture["context_steps"],
        "quota_satisfied": fixture["quota_satisfied"],
        "quota_deficit": fixture["quota_deficit"],
        "bytes": output.stat().st_size,
    }


def _build_primary_unrolls(
    items: tuple[Any, ...],
    *,
    batch_unrolls: int,
    unroll_length: int,
    recurrent_hidden_dim: int,
    policy_version: int,
) -> tuple[SequenceUnroll, ...]:
    unrolls: list[SequenceUnroll] = []
    for episode in items:
        if len(episode.steps) < unroll_length:
            continue
        source = episode.steps[:unroll_length]
        rollout_steps = tuple(
            RolloutStep(
                snapshot=step.snapshot,
                action_index=step.decision.action_index,
                behavior_log_probability=step.decision.behavior_log_probability,
                reward=step.decision.task_reward,
                discount=step.decision.discount,
                policy_decision=step.decision.policy_decision,
            )
            for step in source
        )
        bootstrap = None
        if rollout_steps[-1].discount > 0.0:
            if len(episode.steps) <= unroll_length:
                continue
            bootstrap = episode.steps[unroll_length].snapshot
        unrolls.append(
            SequenceUnroll(
                episode_id=f"kernel-lab:{episode.episode_id}",
                start_step=0,
                policy_version=policy_version,
                initial_recurrent_state=np.zeros(
                    recurrent_hidden_dim,
                    dtype=np.float32,
                ),
                steps=rollout_steps,
                bootstrap_snapshot=bootstrap,
            )
        )
        if len(unrolls) == batch_unrolls:
            break
    if len(unrolls) != batch_unrolls:
        raise ValueError("episodic replay cannot supply the primary unroll fixture")
    return tuple(unrolls)


def _prepare_episode(checkpoint: Path, fixture_dir: Path) -> dict[str, Any]:
    config, state = _config_and_state(checkpoint)
    payload = _read_pickle(checkpoint / "episodic_replay.pkl")
    if not isinstance(payload, dict):
        raise TypeError("episodic replay payload must be an object")
    replay = BoundedEpisodicReplay(
        capacity=config.episodic_learning.replay_capacity_episodes,
        byte_capacity=config.episodic_learning.replay_capacity_bytes,
        episode_byte_capacity=config.episodic_learning.per_episode_capacity_bytes,
        max_segments_per_episode=config.episodic_learning.max_segments_per_episode,
        seed=config.runtime.seed,
    )
    replay.load_state_dict(payload)
    policy_version = int(state["policy_version"])
    sequences = replay.sample(
        config.episodic_learning.sample_sequences,
        learn_steps=config.episodic_learning.learn_steps,
        burn_in_steps=config.episodic_learning.burn_in_steps,
        macro_sample_fraction=config.episodic_learning.macro_sample_fraction,
    )
    unrolls = _build_primary_unrolls(
        replay.snapshot(),
        batch_unrolls=config.optimization.batch_unrolls,
        unroll_length=config.rollout.unroll_length,
        recurrent_hidden_dim=config.model.recurrent_hidden_dim,
        policy_version=policy_version,
    )
    fixture = {
        "version": _FIXTURE_VERSION,
        "checkpoint": str(checkpoint.resolve()),
        "policy_version": policy_version,
        "learner_updates": int(state["learner_updates"]),
        "unrolls": unrolls,
        "episodic_sequences": sequences,
        "episodic_burn_in_steps": tuple(sequence.burn_in_steps for sequence in sequences),
        "episodic_learn_steps": tuple(len(sequence.learn_steps) for sequence in sequences),
    }
    output = fixture_dir / _EPISODE_FIXTURE
    _write_pickle(output, fixture)
    return {
        "output": str(output.resolve()),
        "unrolls": len(unrolls),
        "episodic_sequences": len(sequences),
        "episodic_burn_in_steps": fixture["episodic_burn_in_steps"],
        "episodic_learn_steps": fixture["episodic_learn_steps"],
        "bytes": output.stat().st_size,
    }


def _synchronizing_safe_valid_mask(mask: Tensor) -> Tensor:
    """The exact pre-kernel implementation, retained only as an A/B oracle."""

    valid = mask.bool()
    missing = ~valid.any(dim=-1)
    if missing.any():
        valid = valid.clone()
        valid[missing, 0] = True
    return valid


def _sha256_model(model: torch.nn.Module) -> str:
    digest = hashlib.sha256()
    for name, tensor in sorted(model.state_dict().items()):
        value = tensor.detach().cpu().contiguous()
        digest.update(name.encode("utf-8"))
        digest.update(str(value.dtype).encode("ascii"))
        digest.update(str(tuple(value.shape)).encode("ascii"))
        digest.update(value.numpy().tobytes())
    return digest.hexdigest()


def _distribution(values: Iterable[float]) -> dict[str, float]:
    ordered = sorted(float(value) for value in values)
    if not ordered:
        raise ValueError("timing distribution is empty")
    p90 = ordered[max(0, math.ceil(0.90 * len(ordered)) - 1)]
    return {
        "mean": statistics.fmean(ordered),
        "median": statistics.median(ordered),
        "p90": p90,
        "minimum": ordered[0],
        "maximum": ordered[-1],
    }


def _load_fixture(path: Path) -> dict[str, Any]:
    payload = _read_pickle(path)
    if not isinstance(payload, dict) or payload.get("version") != _FIXTURE_VERSION:
        raise ValueError(f"unsupported benchmark fixture: {path}")
    return payload


def _run(arguments: argparse.Namespace) -> dict[str, Any]:
    checkpoint = arguments.checkpoint.resolve()
    config, state = _config_and_state(checkpoint)
    if arguments.liveness_records_per_autograd_batch is not None:
        config = replace(
            config,
            failure_credit=replace(
                config.failure_credit,
                liveness_records_per_autograd_batch=(arguments.liveness_records_per_autograd_batch),
            ),
        )
    if not 0.0 < arguments.maximum_device_memory_fraction <= 1.0:
        raise ValueError("maximum-device-memory-fraction must be in (0, 1]")
    device = torch.device(arguments.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA/ROCm benchmark device is unavailable")
    if arguments.warmup < 0 or arguments.iterations <= 0:
        raise ValueError("warmup must be non-negative and iterations positive")

    # Keep the stable v32 math-SDPA contract.  This A/B changes only the mask
    # kernel and must not smuggle in an attention-backend transition.
    configure_rocm_sdpa_backend(
        config.runtime.rocm_sdpa_backend,
        devices=(device,),
    )
    if arguments.mask_kernel == "synchronizing":
        candidate_module._safe_valid_mask = _synchronizing_safe_valid_mask

    torch.manual_seed(arguments.seed)
    model = RecurrentCandidateModel(
        config.model.to_model_config(),
        enable_transaction_heads=config.transaction_learning.enabled,
        enable_liveness_head=config.failure_credit.learning_enabled,
    ).to(device)
    network_state = torch.load(
        checkpoint / "network.pt",
        map_location=device,
        weights_only=True,
    )
    model.load_state_dict(network_state, strict=True)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config.optimization.learning_rate,
        weight_decay=config.optimization.weight_decay,
    )
    optimizer_state = torch.load(
        checkpoint / "optimizer.pt",
        map_location=device,
        weights_only=True,
    )
    optimizer.load_state_dict(optimizer_state)
    encoder = GroundedObservationEncoder(config.model.to_encoding_config())
    learner = VTraceLearner(
        model=model,
        encoder=encoder,
        optimizer=optimizer,
        config=config.optimization,
        maximum_unroll_length=config.rollout.unroll_length,
        maximum_policy_lag=config.rollout.max_policy_lag,
        transaction_config=config.transaction_learning,
        failure_credit_config=config.failure_credit,
        episodic_config=config.episodic_learning,
    )
    failure = _load_fixture(arguments.fixture_dir / _FAILURE_FIXTURE)
    episode = _load_fixture(arguments.fixture_dir / _EPISODE_FIXTURE)
    for fixture in (failure, episode):
        if Path(fixture["checkpoint"]).resolve() != checkpoint:
            raise ValueError("fixture belongs to a different checkpoint")
    unrolls = tuple(episode["unrolls"])
    credit_plans = tuple(failure["plans"])
    episodic_sequences = tuple(episode["episodic_sequences"])
    policy_version = int(state["policy_version"])
    learner_updates = int(state["learner_updates"])

    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)

    wall_samples: list[float] = []
    metrics_samples: list[LearnerMetrics] = []
    liveness_replay_samples: list[float] = []
    total_calls = arguments.warmup + arguments.iterations
    for call_index in range(total_calls):
        # Both cells consume the same dropout sequence and update order.
        torch.manual_seed(arguments.seed + call_index)
        stage_fields: dict[str, dict[str, int | float]] = {}
        if device.type == "cuda":
            torch.cuda.synchronize(device)
        started_ns = time.perf_counter_ns()
        metrics = learner.update(
            unrolls,
            current_policy_version=policy_version,
            current_learner_update=learner_updates,
            schedule_policy_version=policy_version,
            schedule_learner_update=learner_updates,
            credit_plans=credit_plans,
            episodic_sequences=episodic_sequences,
            progress=stage_fields.__setitem__,
        )
        if device.type == "cuda":
            torch.cuda.synchronize(device)
        wall_ms = (time.perf_counter_ns() - started_ns) / 1_000_000.0
        if call_index >= arguments.warmup:
            wall_samples.append(wall_ms)
            metrics_samples.append(metrics)
            liveness_fields = stage_fields.get("liveness_replay_complete")
            if liveness_fields is None:
                raise RuntimeError("learner did not report liveness replay timing")
            liveness_replay_samples.append(float(liveness_fields["liveness_replay_ms"]))

    def timing(name: str) -> dict[str, float]:
        return _distribution(float(getattr(sample.timings, name)) for sample in metrics_samples)

    def timing_samples(name: str) -> list[float]:
        return [float(getattr(sample.timings, name)) for sample in metrics_samples]

    state_output: str | None = None
    if arguments.state_output is not None:
        destination = arguments.state_output.resolve()
        destination.parent.mkdir(parents=True, exist_ok=True)
        temporary = destination.with_suffix(destination.suffix + ".tmp")
        torch.save(
            {name: value.detach().cpu() for name, value in model.state_dict().items()},
            temporary,
        )
        temporary.replace(destination)
        state_output = str(destination)

    peak_allocated_bytes = int(torch.cuda.max_memory_allocated(device)) if device.type == "cuda" else None
    peak_reserved_bytes = int(torch.cuda.max_memory_reserved(device)) if device.type == "cuda" else None
    device_total_bytes = int(torch.cuda.get_device_properties(device).total_memory) if device.type == "cuda" else None
    memory_fraction = (
        peak_reserved_bytes / device_total_bytes
        if peak_reserved_bytes is not None and device_total_bytes is not None
        else None
    )
    return {
        "schema_version": "sts2-kernel-lab-e2e-result-v1",
        "checkpoint": str(checkpoint),
        "mask_kernel": arguments.mask_kernel,
        "warmup": arguments.warmup,
        "iterations": arguments.iterations,
        "seed": arguments.seed,
        "execution": {
            "liveness_records_per_autograd_batch": (config.failure_credit.liveness_records_per_autograd_batch),
            "maximum_device_memory_fraction": arguments.maximum_device_memory_fraction,
        },
        "runtime": {
            "torch": torch.__version__,
            "hip": torch.version.hip,
            "device": str(device),
            "device_name": (torch.cuda.get_device_name(device) if device.type == "cuda" else None),
        },
        "workload": {
            "unrolls": len(unrolls),
            "online_steps": sum(len(unroll.steps) for unroll in unrolls),
            "failure_records": len(credit_plans),
            "failure_strata": failure["strata"],
            "failure_context_steps": failure["context_steps"],
            "episodic_sequences": len(episodic_sequences),
            "episodic_burn_in_steps": episode["episodic_burn_in_steps"],
            "episodic_learn_steps": episode["episodic_learn_steps"],
        },
        "timing_ms": {
            "wall_e2e": _distribution(wall_samples),
            "learner_total": timing("total_ms"),
            "validation": timing("validation_ms"),
            "online_recurrent_forward": timing("recurrent_forward_ms"),
            "target_and_loss": timing("target_and_loss_ms"),
            "backward_envelope": timing("backward_ms"),
            "liveness_replay": _distribution(liveness_replay_samples),
            "episodic_replay": timing("episodic_replay_ms"),
            "optimizer": timing("optimizer_step_ms"),
        },
        "samples_ms": {
            "wall_e2e": wall_samples,
            "learner_total": timing_samples("total_ms"),
            "validation": timing_samples("validation_ms"),
            "online_recurrent_forward": timing_samples("recurrent_forward_ms"),
            "target_and_loss": timing_samples("target_and_loss_ms"),
            "backward_envelope": timing_samples("backward_ms"),
            "liveness_replay": liveness_replay_samples,
            "episodic_replay": timing_samples("episodic_replay_ms"),
            "optimizer": timing_samples("optimizer_step_ms"),
        },
        "device_memory": {
            "peak_allocated_bytes": peak_allocated_bytes,
            "peak_reserved_bytes": peak_reserved_bytes,
            "device_total_bytes": device_total_bytes,
            "peak_reserved_fraction": memory_fraction,
            "promotion_gate_passed": (
                memory_fraction is None or memory_fraction <= arguments.maximum_device_memory_fraction
            ),
        },
        # Retain the v1 field for existing result readers.
        "peak_allocated_bytes": peak_allocated_bytes,
        "state_output": state_output,
        "final": {
            "model_sha256": _sha256_model(model),
            "loss": metrics_samples[-1].loss,
            "gradient_norm": metrics_samples[-1].gradient_norm,
            "liveness_credit_loss": metrics_samples[-1].liveness_credit_loss,
            "episodic_loss": metrics_samples[-1].episodic_loss,
        },
    }


def main() -> int:
    arguments = _parse_args()
    if arguments.command == "prepare-failure":
        checkpoint = resolve_external_input_path(arguments.checkpoint)
        fixture_dir = resolve_artifact_path(arguments.fixture_dir)
        result = _prepare_failure(
            checkpoint,
            fixture_dir,
        )
    elif arguments.command == "prepare-episode":
        checkpoint = resolve_external_input_path(arguments.checkpoint)
        fixture_dir = resolve_artifact_path(arguments.fixture_dir)
        result = _prepare_episode(
            checkpoint,
            fixture_dir,
        )
    elif arguments.command == "run":
        arguments.checkpoint = resolve_external_input_path(arguments.checkpoint)
        arguments.fixture_dir = resolve_external_input_path(arguments.fixture_dir)
        if arguments.output is not None:
            arguments.output = resolve_artifact_path(arguments.output)
        if arguments.state_output is not None:
            arguments.state_output = resolve_artifact_path(arguments.state_output)
        result = _run(arguments)
    else:  # pragma: no cover - argparse enforces the choices
        raise RuntimeError("unsupported benchmark command")
    rendered = json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True)
    output = getattr(arguments, "output", None)
    if output is None:
        print(rendered)
    else:
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(rendered + "\n", encoding="utf-8")
    if arguments.command == "run":
        memory = result.get("device_memory")
        if isinstance(memory, dict) and memory.get("promotion_gate_passed") is False:
            return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
