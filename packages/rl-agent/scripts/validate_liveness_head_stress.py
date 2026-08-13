#!/usr/bin/env python3
"""One-shot active-shape stress validation for the v29 liveness learner.

This command is an execution validator, not a training entry point.  It
constructs one typed, compiler-reviewed matched-outcome record whose replayed
incident context contains the maximum reviewed 256 decisions and whose every
decision contains 256 legal candidates.  The production label compiler and
``VTraceLearner.credit_plan_liveness_losses`` then execute forward and backward
passes at both the calibration and mature critic phase boundaries.  Since the
v20 actor retirement the matched-outcome pair remains stored evidence only;
no second context is replayed and no policy objective exists.

The report has no training authority and performs no optimizer step.
"""

from __future__ import annotations

import argparse
import gc
import hashlib
import json
import math
import os
import platform
import sys
import tempfile
import time
from collections import Counter
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Final

import numpy as np
import torch

from sts2_rl.artifacts import resolve_artifact_path
from sts2_rl.checkpoints import V28_100K_FROZEN, validate_frozen_checkpoint
from sts2_rl.encoding import (
    EncodedDecisionSnapshot,
    GroundedObservationEncoder,
    grounding_encoding_identity,
)
from sts2_rl.encoding.snapshot import sparse_token_table
from sts2_rl.models import (
    GroundedCandidateBatch,
    RecurrentCandidateModel,
)
from sts2_rl.training.checkpointing import initialize_model_from_checkpoint
from sts2_rl.training.config import TrainingConfig, load_training_config
from sts2_rl.training.failure_credit import (
    CreditCompiler,
    CreditProvenance,
    EvidenceRecord,
    FailureIncident,
    FailureOutcome,
    IdentityTriple,
    LearningContext,
    LearningStep,
    LoopEdgeEvidence,
    MatchedOutcomePair,
    OutcomeArm,
    PolicyWitness,
    SemanticKey,
    TargetAuthority,
    WitnessKind,
)
from sts2_rl.training.launch_contract import current_formal_report_generation_source
from sts2_rl.training.learner import (
    LivenessCreditLosses,
    VTraceLearner,
    compile_liveness_label_manifest,
)
from sts2_rl.training.sdpa import configure_rocm_sdpa_backend

LIVENESS_HEAD_STRESS_REPORT_VERSION: Final = "sts2-liveness-head-active-shape-stress-v4"
SYNTHETIC_EVIDENCE_AUTHORITY_VERSION: Final = "sts2-liveness-stress-synthetic-evidence-v1"
FORMAL_STEPS: Final = 256
FORMAL_CANDIDATES: Final = 256
_PACKAGE_ROOT: Final = Path(__file__).resolve().parents[1]


@dataclass(frozen=True, slots=True)
class StressShape:
    steps: int = FORMAL_STEPS
    candidates: int = FORMAL_CANDIDATES
    ordinary_candidates: int = 3

    def __post_init__(self) -> None:
        for label, value, minimum in (
            ("steps", self.steps, 2),
            ("candidates", self.candidates, 2),
            ("ordinary_candidates", self.ordinary_candidates, 2),
        ):
            if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
                raise ValueError(f"stress {label} must be an integer >= {minimum}")
        if self.ordinary_candidates >= self.candidates:
            raise ValueError("ordinary candidate count must remain below the stress shape")


@dataclass(slots=True)
class _ModelInitializationResources:
    """Minimum public model-init resource surface, without a training runtime."""

    model: RecurrentCandidateModel
    device: torch.device

    def publish_collector_policy(self) -> float:
        return 0.0


def _atomic_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    serialized = (
        json.dumps(
            payload,
            allow_nan=False,
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
        )
        + "\n"
    )
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.",
        suffix=".tmp",
        dir=path.parent,
        text=True,
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(
            descriptor,
            "w",
            encoding="utf-8",
            newline="\n",
        ) as handle:
            handle.write(serialized)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    except BaseException:
        temporary.unlink(missing_ok=True)
        raise


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _semantic_key(
    namespace: str,
    value: str,
) -> SemanticKey:
    return SemanticKey.from_payload(
        namespace=namespace,
        schema_version=SYNTHETIC_EVIDENCE_AUTHORITY_VERSION,
        payload={"value": value},
    )


def _identity(
    value: str,
    *,
    loop_value: str | None = None,
    comparison_value: str | None = None,
) -> IdentityTriple:
    return IdentityTriple(
        exact=_semantic_key("stress.exact", value),
        loop=_semantic_key(
            "stress.loop",
            loop_value if loop_value is not None else value,
        ),
        comparison=_semantic_key(
            "stress.comparison",
            comparison_value if comparison_value is not None else value,
        ),
    )


def _snapshot(
    config: Any,
    *,
    candidate_count: int,
) -> EncodedDecisionSnapshot:
    """Build one compact, all-legal synthetic snapshot under the active ABI."""

    feature_dim = int(config.feature_dim)
    world_feature = tuple([1.0] + [0.0] * (feature_dim - 1))
    candidate_features = tuple(
        tuple(
            [
                1.0,
                float((index % 17) + 1) / 17.0,
                *([0.0] * (feature_dim - 2)),
            ]
        )
        for index in range(candidate_count)
    )
    candidate_ids = tuple(
        (
            2 + index % 3,
            2 + index % 3,
            2,
            3 + index % 61,
            3 + index % 61,
            3 + index,
            3 + index,
            2,
            2,
            3 + index % 61,
            3 + index % 61,
            3 + index,
            3 + index,
        )
        for index in range(candidate_count)
    )
    return EncodedDecisionSnapshot(
        config=config,
        encoding_fingerprint=grounding_encoding_identity()["fingerprint_sha256"],
        world=sparse_token_table(
            features=(world_feature,),
            ids=((2, 2, 2, 2, 2, 2, 2, 2, 0),),
            feature_dim=feature_dim,
            id_width=9,
        ),
        candidates=sparse_token_table(
            features=candidate_features,
            ids=candidate_ids,
            feature_dim=feature_dim,
            id_width=13,
        ),
        locals=sparse_token_table(
            features=(),
            ids=(),
            feature_dim=feature_dim,
            id_width=9,
        ),
        local_offsets=np.zeros(
            candidate_count + 1,
            dtype=np.uint32,
        ),
        action_mask=np.ones(candidate_count, dtype=np.bool_),
        domain_id=1,
    )


def build_synthetic_evidence_record(
    *,
    training_config: TrainingConfig,
    steps: int,
    candidates: int,
    policy_version: int,
    include_direct_witness: bool,
    include_matched_outcome: bool,
    identity_suffix: str,
) -> EvidenceRecord:
    """Build a fully typed incident and compile it with the production compiler."""

    encoding_config = training_config.model.to_encoding_config()
    if steps > training_config.failure_credit.maximum_context_steps:
        raise ValueError("stress steps exceed failure-credit context capacity")
    if candidates > encoding_config.max_candidates:
        raise ValueError("stress candidates exceed encoder candidate capacity")
    snapshot = _snapshot(
        encoding_config,
        candidate_count=candidates,
    )
    comparison_scope = f"{identity_suffix}.matched-semantics"

    def build_context(
        *,
        arm: str,
        selected_last_action: int,
        direct_cycle: bool,
    ) -> LearningContext:
        candidate_actions = tuple(
            _identity(
                f"{identity_suffix}.{arm}.candidate.{index}",
                comparison_value=f"{comparison_scope}.candidate.{index}",
            )
            for index in range(candidates)
        )
        context_steps: list[LearningStep] = []
        cycle_start = steps - 2
        for index in range(steps):
            loop_value = f"{identity_suffix}.{arm}.direct-cycle-node" if direct_cycle and index >= cycle_start else None
            context_steps.append(
                LearningStep(
                    decision_id=f"{identity_suffix}.{arm}.decision.{index}",
                    episode_step=index,
                    snapshot=snapshot,
                    action_index=(selected_last_action if index == steps - 1 else 0),
                    behavior_log_probability=-math.log(candidates),
                    policy_version=policy_version,
                    node=_identity(
                        f"{identity_suffix}.{arm}.node.{index}",
                        loop_value=loop_value,
                        comparison_value=f"{comparison_scope}.node.{index}",
                    ),
                    anchor=_semantic_key(
                        "stress.anchor",
                        f"{identity_suffix}.{arm}.anchor",
                    ),
                    candidate_actions=candidate_actions,
                    forced=False,
                )
            )
        return LearningContext(
            context_id=f"{identity_suffix}.{arm}.context",
            episode_id=f"{identity_suffix}.{arm}.episode",
            start_step=0,
            initial_recurrent_state=np.zeros(
                training_config.model.recurrent_hidden_dim,
                dtype=np.float32,
            ),
            steps=tuple(context_steps),
            burn_in_steps=0,
        )

    context = build_context(
        arm="worse",
        selected_last_action=0,
        direct_cycle=include_direct_witness,
    )
    better_context = (
        build_context(
            arm="better",
            selected_last_action=1,
            direct_cycle=False,
        )
        if include_matched_outcome
        else None
    )
    provenance = CreditProvenance(
        run_id="liveness-head-active-shape-stress",
        game_version="synthetic-no-game-trajectory",
        environment_schema_version=(SYNTHETIC_EVIDENCE_AUTHORITY_VERSION),
        identity_version=SYNTHETIC_EVIDENCE_AUTHORITY_VERSION,
        detector_version="synthetic-reviewed-witness-v1",
        adapter_version="synthetic-reviewed-adapter-v1",
        collector_version="synthetic-reviewed-collector-v1",
        policy_version=policy_version,
    )
    supporting_steps = tuple(range(steps))
    witnesses: list[PolicyWitness] = [
        PolicyWitness(
            witness_id=f"{identity_suffix}.risk",
            kind=WitnessKind.RISK_SEQUENCE,
            attributed_step_indices=tuple(range(steps)),
            supporting_episode_steps=supporting_steps,
            occurrences=1,
            cycle_span=None,
            successor_confirmed=True,
        )
    ]
    if include_direct_witness:
        direct_indices = (steps - 2, steps - 1)
        direct_support = tuple(context.steps[index].episode_step for index in direct_indices)
        witnesses.insert(
            0,
            PolicyWitness(
                witness_id=f"{identity_suffix}.direct",
                kind=WitnessKind.DIRECT_WITNESS,
                attributed_step_indices=direct_indices,
                supporting_episode_steps=direct_support,
                occurrences=2,
                cycle_span=1,
                successor_confirmed=True,
                loop_edges=(
                    LoopEdgeEvidence(
                        node=context.steps[-1].node.loop,
                        action=(context.steps[-1].selected_action.loop),
                        supporting_episode_steps=direct_support,
                    ),
                ),
                behavior_mean_log_probability=-math.log(candidates),
            ),
        )
    incident_id = f"{identity_suffix}.incident"
    if better_context is not None:
        pair = MatchedOutcomePair(
            pair_id=f"{identity_suffix}.matched-outcome",
            better=OutcomeArm(
                incident_id=f"{identity_suffix}.better-incident",
                context=better_context,
                step_index=steps - 1,
                outcome=FailureOutcome.COMPLETED,
            ),
            worse=OutcomeArm(
                incident_id=incident_id,
                context=context,
                step_index=steps - 1,
                outcome=FailureOutcome.DEADLOCK_STALL,
            ),
        )
        witnesses.append(
            PolicyWitness(
                witness_id=f"{identity_suffix}.matched-outcome-witness",
                kind=WitnessKind.MATCHED_OUTCOME_PAIR,
                attributed_step_indices=(),
                supporting_episode_steps=(steps - 1,),
                occurrences=1,
                cycle_span=None,
                successor_confirmed=True,
                outcome_pair=pair,
            )
        )
    incident = FailureIncident(
        incident_id=incident_id,
        scope_key=f"{identity_suffix}.scope",
        failure_kind="liveness_active_shape_stress",
        outcome=FailureOutcome.DEADLOCK_STALL,
        task_authority=TargetAuthority.OBJECTIVE_CONFIRMED,
        local_authority=TargetAuthority.DETECTOR_CONFIRMED,
        task_return=-1.0,
        local_failure_cost=1.0,
        context=context,
        witnesses=tuple(witnesses),
        detector_window_steps=max(1, steps - 1),
        progress_epoch=0,
        provenance=provenance,
    )
    return EvidenceRecord(
        incident=incident,
        plan=CreditCompiler().compile(incident),
    )


def _resolve_device(requested: str) -> torch.device:
    normalized = requested.strip().lower()
    if normalized == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    device = torch.device(requested)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA/ROCm device requested but unavailable")
    return device


def _runtime_identity(
    device: torch.device,
) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "python": sys.version,
        "platform": platform.platform(),
        "torch_version": torch.__version__,
        "torch_git_version": getattr(
            torch.version,
            "git_version",
            None,
        ),
        "torch_cuda_version": getattr(
            torch.version,
            "cuda",
            None,
        ),
        "torch_hip_version": getattr(
            torch.version,
            "hip",
            None,
        ),
        "cuda_available": torch.cuda.is_available(),
        "cuda_device_count": (torch.cuda.device_count() if torch.cuda.is_available() else 0),
        "resolved_device": str(device),
        "environment": {
            key: os.environ.get(key)
            for key in (
                "HIP_VISIBLE_DEVICES",
                "ROCR_VISIBLE_DEVICES",
                "HSA_OVERRIDE_GFX_VERSION",
            )
        },
    }
    if device.type == "cuda":
        properties = torch.cuda.get_device_properties(device)
        payload["accelerator"] = {
            "name": properties.name,
            "total_memory_bytes": int(properties.total_memory),
            "multi_processor_count": int(properties.multi_processor_count),
            "device_index": (device.index if device.index is not None else torch.cuda.current_device()),
        }
    else:
        payload["accelerator"] = None
    return payload


def _cuda_synchronize(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def _memory_start(
    device: torch.device,
) -> dict[str, int] | None:
    if device.type != "cuda":
        return None
    gc.collect()
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats(device)
    _cuda_synchronize(device)
    return {
        "allocated_bytes": int(torch.cuda.memory_allocated(device)),
        "reserved_bytes": int(torch.cuda.memory_reserved(device)),
    }


def _memory_finish(
    device: torch.device,
    start: dict[str, int] | None,
) -> dict[str, int] | None:
    if device.type != "cuda" or start is None:
        return None
    return {
        "start_allocated_bytes": start["allocated_bytes"],
        "start_reserved_bytes": start["reserved_bytes"],
        "end_allocated_bytes": int(torch.cuda.memory_allocated(device)),
        "end_reserved_bytes": int(torch.cuda.memory_reserved(device)),
        "peak_allocated_bytes": int(torch.cuda.max_memory_allocated(device)),
        "peak_reserved_bytes": int(torch.cuda.max_memory_reserved(device)),
    }


def _loss_values(
    losses: LivenessCreditLosses,
) -> dict[str, float]:
    return {
        name: float(getattr(losses, name).detach().cpu().item())
        for name in (
            "value_critic_loss",
            "critic_loss",
        )
    }


def _weighted_objective(
    losses: LivenessCreditLosses,
    *,
    config: Any,
) -> torch.Tensor:
    return (
        config.liveness_value_critic_weight * losses.value_critic_loss
        + config.liveness_cost_critic_weight * losses.critic_loss
    )


def _gradient_summary(
    model: RecurrentCandidateModel,
) -> dict[str, Any]:
    total_squared = 0.0
    tensors = 0
    nonzero_tensors = 0
    liveness_nonzero = 0
    finite = True
    for name, parameter in model.named_parameters():
        gradient = parameter.grad
        if gradient is None:
            continue
        tensors += 1
        detached = gradient.detach()
        finite = finite and bool(torch.isfinite(detached).all().item())
        squared = float(detached.float().square().sum().item())
        total_squared += squared
        if squared > 0.0:
            nonzero_tensors += 1
            if name.startswith(
                (
                    "candidate_liveness_cost_head.",
                    "liveness_cost_value_head.",
                )
            ):
                liveness_nonzero += 1
    return {
        "finite": finite,
        "gradient_tensors": tensors,
        "nonzero_gradient_tensors": nonzero_tensors,
        "nonzero_liveness_head_tensors": liveness_nonzero,
        "global_l2_norm": math.sqrt(total_squared),
    }


def _forward_shape_summary(
    events: list[dict[str, Any]],
) -> dict[str, Any]:
    histogram = Counter(
        (
            int(event["batch_rows"]),
            int(event["candidates"]),
            int(event["world_tokens"]),
            int(event["local_tokens"]),
            bool(event["detach_liveness_shared_features"]),
        )
        for event in events
    )
    return {
        "forward_calls": len(events),
        "maximum_batch_rows": max(
            (int(event["batch_rows"]) for event in events),
            default=0,
        ),
        "minimum_batch_rows": min(
            (int(event["batch_rows"]) for event in events),
            default=0,
        ),
        "maximum_candidate_axis": max(
            (int(event["candidates"]) for event in events),
            default=0,
        ),
        "minimum_candidate_axis": min(
            (int(event["candidates"]) for event in events),
            default=0,
        ),
        "histogram": [
            {
                "batch_rows": key[0],
                "candidates": key[1],
                "world_tokens": key[2],
                "local_tokens": key[3],
                "detach_liveness_shared_features": key[4],
                "calls": count,
            }
            for key, count in sorted(histogram.items())
        ],
    }


def _run_phase(
    *,
    name: str,
    learner_update: int,
    learner: VTraceLearner,
    record: EvidenceRecord,
    policy_version: int,
    model: RecurrentCandidateModel,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    forward_events: list[dict[str, Any]],
) -> dict[str, Any]:
    optimizer.zero_grad(set_to_none=True)
    start_memory = _memory_start(device)
    event_start = len(forward_events)
    _cuda_synchronize(device)
    forward_started = time.perf_counter()
    losses = learner.credit_plan_liveness_losses(
        (record.plan,),
        current_policy_version=policy_version,
        current_learner_update=learner_update,
    )
    _cuda_synchronize(device)
    forward_seconds = time.perf_counter() - forward_started
    objective = _weighted_objective(
        losses,
        config=learner.failure_credit_config,
    )
    objective_value = float(objective.detach().cpu().item())
    backward_started = time.perf_counter()
    objective.backward()
    _cuda_synchronize(device)
    backward_seconds = time.perf_counter() - backward_started
    gradients = _gradient_summary(model)
    memory = _memory_finish(device, start_memory)
    phase_events = forward_events[event_start:]
    result = {
        "name": name,
        "learner_update": learner_update,
        "calibration_active": (learner_update < learner.failure_credit_config.liveness_head_calibration_updates),
        "durations_seconds": {
            "forward_and_loss": forward_seconds,
            "backward": backward_seconds,
            "total": forward_seconds + backward_seconds,
        },
        "objective": objective_value,
        "losses": _loss_values(losses),
        "labels": {
            "value": losses.value_labels,
            "critic": losses.critic_labels,
            "censored_suppressed": losses.censored_suppressed_labels,
        },
        "work": {
            "contexts": losses.replayed_contexts,
            "steps": losses.replayed_steps,
            "candidates": losses.replayed_candidates,
            "autograd_segments": losses.autograd_segments,
        },
        "forward_shapes": _forward_shape_summary(phase_events),
        "gradients": gradients,
        "gpu_memory": memory,
    }
    del objective
    del losses
    optimizer.zero_grad(set_to_none=True)
    gc.collect()
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return result


def _initialize_model(
    *,
    training_config: TrainingConfig,
    device: torch.device,
    initialization: str,
    artifact_root: Path | None,
) -> tuple[RecurrentCandidateModel, dict[str, Any], float]:
    started = time.perf_counter()
    torch.manual_seed(training_config.runtime.seed)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(training_config.runtime.seed)
    model = RecurrentCandidateModel(
        training_config.model.to_model_config(),
        enable_transaction_heads=(training_config.transaction_learning.enabled),
        enable_liveness_head=True,
    ).to(device)
    if initialization == "synthetic":
        source = {
            "kind": "synthetic-fresh-v29-model",
            "authority": ("architecture, active-shape, autograd and memory validation only; no policy-quality claim"),
            "torch_seed": training_config.runtime.seed,
            "model_config": training_config.model.to_mapping()
            if hasattr(training_config.model, "to_mapping")
            else {
                "architecture": training_config.model.architecture,
                "d_model": training_config.model.d_model,
                "recurrent_hidden_dim": (training_config.model.recurrent_hidden_dim),
            },
        }
    elif initialization == "frozen-v28":
        if artifact_root is None:
            raise ValueError("frozen-v28 initialization requires an artifact root")
        validated = validate_frozen_checkpoint(artifact_root)
        resources = _ModelInitializationResources(
            model=model,
            device=device,
        )
        initialized = initialize_model_from_checkpoint(
            validated.root,
            config=training_config,
            resources=resources,
        )
        source = {
            "kind": "frozen-v28-model-init",
            "authority": ("pinned v28 shared model parameters with freshly initialized v29 liveness heads"),
            "contract_name": V28_100K_FROZEN.name,
            "checkpoint_id": V28_100K_FROZEN.checkpoint_id,
            "checkpoint_path": os.fspath(initialized),
            "manifest_sha256": (V28_100K_FROZEN.manifest_sha256),
            "metadata_sha256": (V28_100K_FROZEN.metadata_sha256),
        }
    else:
        raise ValueError(f"unsupported initialization mode: {initialization!r}")
    return model, source, time.perf_counter() - started


def run_liveness_head_stress(
    *,
    training_config: TrainingConfig,
    device: torch.device,
    shape: StressShape = StressShape(),
    initialization: str,
    artifact_root: Path | None,
    config_source: str,
    require_formal_shape: bool,
) -> dict[str, Any]:
    """Execute the one-shot production learner stress contract."""

    total_started = time.perf_counter()
    if not training_config.failure_credit.learning_enabled:
        raise ValueError("stress validation requires failure-credit learning")
    if not training_config.model.max_candidates >= shape.candidates:
        raise ValueError("model candidate capacity is below stress shape")
    if training_config.failure_credit.maximum_context_steps < shape.steps:
        raise ValueError("context capacity is below stress shape")
    if require_formal_shape and (
        shape.steps != FORMAL_STEPS
        or shape.candidates != FORMAL_CANDIDATES
        or training_config.model.max_candidates != FORMAL_CANDIDATES
        or training_config.failure_credit.maximum_context_steps != FORMAL_STEPS
    ):
        raise ValueError("formal v29 stress requires exact 256x256 capacities")

    sdpa = configure_rocm_sdpa_backend(
        training_config.runtime.rocm_sdpa_backend,
        devices=(device,),
    )
    model, source, initialization_seconds = _initialize_model(
        training_config=training_config,
        device=device,
        initialization=initialization,
        artifact_root=artifact_root,
    )
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=training_config.optimization.learning_rate,
        weight_decay=training_config.optimization.weight_decay,
    )
    encoder = GroundedObservationEncoder(training_config.model.to_encoding_config())
    learner = VTraceLearner(
        model=model,
        encoder=encoder,
        optimizer=optimizer,
        config=training_config.optimization,
        maximum_unroll_length=(training_config.rollout.unroll_length),
        maximum_policy_lag=training_config.rollout.max_policy_lag,
        transaction_config=training_config.transaction_learning,
        failure_credit_config=training_config.failure_credit,
        episodic_config=training_config.episodic_learning,
    )
    policy_version = V28_100K_FROZEN.policy_version if initialization == "frozen-v28" else 0
    evidence_started = time.perf_counter()
    record = build_synthetic_evidence_record(
        training_config=training_config,
        steps=shape.steps,
        candidates=shape.candidates,
        policy_version=policy_version,
        include_direct_witness=True,
        include_matched_outcome=True,
        identity_suffix="formal-max-shape",
    )
    evidence_seconds = time.perf_counter() - evidence_started

    phase_updates = {
        "calibration": 0,
        "mature": (training_config.failure_credit.liveness_head_calibration_updates),
    }
    manifests = {
        name: compile_liveness_label_manifest(
            (record.plan,),
            config=training_config.failure_credit,
            current_policy_version=policy_version,
            current_learner_update=learner_update,
        )
        for name, learner_update in phase_updates.items()
    }
    forward_events: list[dict[str, Any]] = []

    def capture_forward(
        _module: torch.nn.Module,
        args: tuple[object, ...],
        kwargs: dict[str, Any],
    ) -> None:
        batch = args[0]
        if not isinstance(batch, GroundedCandidateBatch):
            raise TypeError("stress hook observed a non-grounded model batch")
        forward_events.append(
            {
                "batch_rows": int(batch.domain_ids.shape[0]),
                "candidates": int(batch.candidates.action_mask.shape[1]),
                "world_tokens": int(batch.world.features.shape[1]),
                "local_tokens": int(batch.candidates.local_features.shape[2]),
                "detach_liveness_shared_features": bool(
                    kwargs.get(
                        "detach_liveness_shared_features",
                        False,
                    )
                ),
            }
        )

    hook = model.register_forward_pre_hook(
        capture_forward,
        with_kwargs=True,
    )
    try:
        phases = {
            name: _run_phase(
                name=name,
                learner_update=learner_update,
                learner=learner,
                record=record,
                policy_version=policy_version,
                model=model,
                optimizer=optimizer,
                device=device,
                forward_events=forward_events,
            )
            for name, learner_update in phase_updates.items()
        }

        probe_record = build_synthetic_evidence_record(
            training_config=training_config,
            steps=2,
            candidates=shape.ordinary_candidates,
            policy_version=policy_version,
            include_direct_witness=False,
            include_matched_outcome=False,
            identity_suffix="ordinary-active-shape-probe",
        )
        probe_event_start = len(forward_events)
        probe_started = time.perf_counter()
        with torch.no_grad():
            probe_losses = learner.credit_plan_liveness_losses(
                (probe_record.plan,),
                current_policy_version=policy_version,
                current_learner_update=phase_updates["mature"],
            )
        _cuda_synchronize(device)
        probe_seconds = time.perf_counter() - probe_started
        probe_events = forward_events[probe_event_start:]
        active_shape_probe: dict[str, Any] = {
            "durations_seconds": probe_seconds,
            "work": {
                "contexts": probe_losses.replayed_contexts,
                "steps": probe_losses.replayed_steps,
                "candidates": probe_losses.replayed_candidates,
                "autograd_segments": (probe_losses.autograd_segments),
            },
            "forward_shapes": _forward_shape_summary(probe_events),
        }
        del probe_losses
    finally:
        hook.remove()

    # Since the v20 actor retirement only the incident-owned context is
    # replayed; the matched-outcome pair remains stored evidence.
    context_count = 1
    expected_replayed_steps = shape.steps * context_count
    expected_candidates = expected_replayed_steps * shape.candidates
    expected_segments_per_context = math.ceil(shape.steps / training_config.failure_credit.liveness_tbptt_window_steps)
    expected_segments = expected_segments_per_context * context_count
    finite_losses = all(
        math.isfinite(value)
        for phase in phases.values()
        for value in (
            phase["objective"],
            *phase["losses"].values(),
        )
    )
    finite_gradients = all(phase["gradients"]["finite"] for phase in phases.values())
    stress_forward_active_shape = all(
        phase["forward_shapes"]["forward_calls"] == shape.steps
        and phase["forward_shapes"]["maximum_batch_rows"] == context_count
        and phase["forward_shapes"]["minimum_batch_rows"] == context_count
        and phase["forward_shapes"]["maximum_candidate_axis"] == shape.candidates
        and phase["forward_shapes"]["minimum_candidate_axis"] == shape.candidates
        for phase in phases.values()
    )
    ordinary_not_globally_padded = (
        active_shape_probe["work"]["steps"] == 2
        and active_shape_probe["work"]["candidates"] == 2 * shape.ordinary_candidates
        and active_shape_probe["forward_shapes"]["forward_calls"] == 2
        and active_shape_probe["forward_shapes"]["maximum_batch_rows"] == 1
        and active_shape_probe["forward_shapes"]["maximum_candidate_axis"] == shape.ordinary_candidates
        and active_shape_probe["forward_shapes"]["maximum_candidate_axis"] < training_config.model.max_candidates
    )
    phase_contract = (
        phases["calibration"]["calibration_active"]
        and not phases["mature"]["calibration_active"]
        and all(
            phase["labels"]["value"] == shape.steps
            and phase["labels"]["critic"] == shape.steps
            and phase["labels"]["censored_suppressed"] == 0
            for phase in phases.values()
        )
    )
    work_accounting = all(
        phase["work"]["contexts"] == context_count
        and phase["work"]["steps"] == expected_replayed_steps
        and phase["work"]["candidates"] == expected_candidates
        and phase["work"]["autograd_segments"] == expected_segments
        for phase in phases.values()
    ) and all(
        manifest.work.contexts == context_count
        and manifest.work.steps == expected_replayed_steps
        and manifest.work.candidates == expected_candidates
        and manifest.work.autograd_segments == expected_segments
        for manifest in manifests.values()
    )
    bounded_tbptt = (
        expected_segments_per_context
        == math.ceil(shape.steps / training_config.failure_credit.liveness_tbptt_window_steps)
        and expected_segments > 1
        and all(phase["work"]["autograd_segments"] == expected_segments for phase in phases.values())
    )
    gradient_contract = all(
        phase["gradients"]["nonzero_gradient_tensors"] > 0
        and phase["gradients"]["nonzero_liveness_head_tensors"] > 0
        and phase["gradients"]["global_l2_norm"] > 0.0
        for phase in phases.values()
    )
    gates = {
        "formal_shape_contract": (
            not require_formal_shape
            or (
                shape.steps == FORMAL_STEPS
                and shape.candidates == FORMAL_CANDIDATES
                and context_count == 1
                and expected_candidates == 65_536
            )
        ),
        "production_manifest_work_accounting": work_accounting,
        "bounded_tbptt_segments": bounded_tbptt,
        "calibration_and_mature_phase_contract": phase_contract,
        "finite_losses": finite_losses,
        "finite_gradients": finite_gradients,
        "nonzero_expected_gradients": gradient_contract,
        "stress_forward_uses_active_timestep_batch": (stress_forward_active_shape),
        "ordinary_rows_not_globally_padded": (ordinary_not_globally_padded),
        "no_optimizer_step": True,
    }
    gpu_peaks = [phase["gpu_memory"] for phase in phases.values() if phase["gpu_memory"] is not None]
    report = {
        "version": LIVENESS_HEAD_STRESS_REPORT_VERSION,
        "status": "passed" if all(gates.values()) else "failed",
        "training_authority": False,
        "one_shot": True,
        "optimizer_steps": 0,
        "authority": {
            "evidence": SYNTHETIC_EVIDENCE_AUTHORITY_VERSION,
            "description": (
                "synthetic maximum-shape matched-outcome typed DTO; production "
                "CreditCompiler, label manifest and learner loss; no "
                "game-policy quality or environment-outcome claim"
            ),
            "shared_snapshot_across_steps_and_arms": True,
            "all_candidates_legal": True,
        },
        "config": {
            "source": config_source,
            "version": training_config.version,
            "profile": training_config.profile,
            "fingerprint_sha256": (training_config.fingerprint_sha256()),
            "maximum_context_steps": (training_config.failure_credit.maximum_context_steps),
            "maximum_candidates": (training_config.model.max_candidates),
            "tbptt_window_steps": (training_config.failure_credit.liveness_tbptt_window_steps),
            "calibration_updates": (training_config.failure_credit.liveness_head_calibration_updates),
        },
        "initialization": source,
        "shape": {
            "contexts": context_count,
            "steps_per_context": shape.steps,
            "replayed_steps": expected_replayed_steps,
            "legal_candidates_per_step": shape.candidates,
            "candidate_decisions_per_context": shape.steps * shape.candidates,
            "replayed_candidate_decisions": expected_candidates,
            "ordinary_probe_candidates": shape.ordinary_candidates,
            "tbptt_segments_per_context": expected_segments_per_context,
            "expected_tbptt_segments": expected_segments,
        },
        "runtime_identity": _runtime_identity(device),
        "sdpa": sdpa.to_mapping(),
        "durations_seconds": {
            "model_initialization": initialization_seconds,
            "evidence_construction": evidence_seconds,
            "total": time.perf_counter() - total_started,
        },
        "gpu_peak": (
            {
                "maximum_allocated_bytes": max(item["peak_allocated_bytes"] for item in gpu_peaks),
                "maximum_reserved_bytes": max(item["peak_reserved_bytes"] for item in gpu_peaks),
            }
            if gpu_peaks
            else None
        ),
        "gates": gates,
        "phases": phases,
        "active_shape_probe": active_shape_probe,
    }
    return report


def _artifact_root(
    explicit: Path | None,
) -> Path | None:
    candidates: list[Path] = []
    if explicit is not None:
        candidates.append(explicit)
    configured = os.environ.get("STS2_ARTIFACT_ROOT")
    if configured:
        candidates.append(Path(configured))
    candidates.extend(
        (
            Path("/mnt/e/game/project/sts2_mcp_artifacts/runtime"),
            Path(r"E:\game\project\sts2_mcp_artifacts\runtime"),
        )
    )
    for candidate in candidates:
        resolved = candidate.expanduser().resolve(strict=False)
        if V28_100K_FROZEN.resolve(resolved).is_dir():
            return resolved
    return None


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--profile", default="preheat")
    parser.add_argument("--config", type=Path, default=None)
    parser.add_argument(
        "--initialization",
        choices=("frozen-v28", "synthetic"),
        default="frozen-v28",
    )
    parser.add_argument("--artifact-root", type=Path)
    return parser


def main() -> int:
    args = _parser().parse_args()
    generation_source = current_formal_report_generation_source(_PACKAGE_ROOT.parents[1], __file__)
    artifact_root = _artifact_root(args.artifact_root)
    report_path = resolve_artifact_path(args.report, root=artifact_root)
    try:
        if args.config is not None:
            config_path = args.config.expanduser().resolve()
            training_config = load_training_config(
                profile=args.profile,
                config_path=config_path,
            )
            config_source = os.fspath(config_path)
        else:
            # The maintained profile plus the single mode override is the
            # smallest reviewed learning-enabled configuration; archived
            # experiment recipes no longer parse under strict config v20.
            training_config = load_training_config(
                profile=args.profile,
                overrides=("failure_credit.mode=learning",),
            )
            config_source = f"profile:{args.profile}+failure_credit.mode=learning"
        device = _resolve_device(args.device)
        report = run_liveness_head_stress(
            training_config=training_config,
            device=device,
            initialization=args.initialization,
            artifact_root=artifact_root,
            config_source=config_source,
            require_formal_shape=True,
        )
    except Exception as exc:
        report = {
            "version": LIVENESS_HEAD_STRESS_REPORT_VERSION,
            "status": "failed",
            "training_authority": False,
            "one_shot": True,
            "error": {
                "type": type(exc).__name__,
                "message": str(exc),
            },
        }
    if current_formal_report_generation_source(_PACKAGE_ROOT.parents[1], __file__) != generation_source:
        raise RuntimeError("formal runtime-readiness source changed during report generation")
    report["generation_source"] = generation_source
    _atomic_json(report_path, report)
    digest = _sha256_file(report_path)
    print(
        json.dumps(
            {
                "status": report["status"],
                "report": os.fspath(report_path),
                "sha256": digest,
            },
            sort_keys=True,
        )
    )
    return 0 if report["status"] == "passed" else 1


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
