"""Asynchronous actor/V-trace runtime with fixed held-out evaluation schedules."""

from __future__ import annotations

import hashlib
import json
import statistics
import time
from collections.abc import Callable, Mapping
from copy import deepcopy
from dataclasses import asdict, replace
from pathlib import Path
from typing import Any
from uuid import uuid4

import torch

from sts2_baseline import (
    SequenceUnroll,
    revival_efficiency_reward_identity,
    task_reward_identity,
)
from sts2_rl.artifacts import resolve_artifact_path
from sts2_rl.contracts import EnvironmentBackend
from sts2_rl.encoding import GroundedObservationEncoder, grounding_encoding_identity
from sts2_rl.models import RecurrentCandidateModel

from .checkpointing import (
    ActorSupervisorState,
    EvaluationGateState,
    TrainingState,
    actor_supervisor_state_from_metadata,
    evaluation_gate_state_from_metadata,
    initialize_model_from_checkpoint,
    load_training_checkpoint,
    preflight_model_initialization,
    preflight_training_checkpoint,
    save_training_checkpoint,
)
from .collector import EpisodeMetrics
from .config import TrainingConfig, engine_revival_identity
from .evaluation_liveness import (
    evaluate_liveness_guard,
    summarize_greedy_liveness_journal,
)
from .factory import (
    TrainingResources,
    build_backend,
    build_training_resources,
    resolve_device,
)
from .pipeline import ActorLearnerPipeline, RecoverableActorIncident
from .sdpa import sdpa_transition_provenance
from .seeding import (
    final_audit_evaluation_seeds,
    held_out_evaluation_seeds,
)
from .trajectory import TrajectoryJournal

# Human-readable training-system ABI persisted in inspection and run_start
# telemetry.  Exact-resume safety is enforced independently by the config,
# encoding, replay and episodic-objective checkpoint contracts; this marker
# makes the one-terminal-unit/fresh-policy release distinguishable in metrics.
_TRAINING_PIPELINE_ABI = "bounded-fifo-async-vtrace-episodic-v6"


def exploration_epsilon(config: TrainingConfig, environment_steps: int) -> float:
    curriculum = config.curriculum
    if environment_steps <= 0:
        return float(curriculum.epsilon_start)
    if environment_steps >= curriculum.epsilon_decay_steps:
        return float(curriculum.epsilon_end)
    progress = min(1.0, max(0.0, environment_steps / curriculum.epsilon_decay_steps))
    return float(curriculum.epsilon_start + progress * (curriculum.epsilon_end - curriculum.epsilon_start))


class JsonlMetrics:
    def __init__(self, path: Path) -> None:
        self.path = path
        path.parent.mkdir(parents=True, exist_ok=True)

    def write(self, event: str, payload: dict[str, Any]) -> None:
        with self.path.open("a", encoding="utf-8", newline="\n") as handle:
            handle.write(
                json.dumps(
                    {"event": event, "unix_s": time.time(), **payload},
                    sort_keys=True,
                    default=str,
                )
                + "\n"
            )


class EvaluationInfrastructureError(RuntimeError):
    """A held-out gate could not produce a policy-valid episode."""


def _model_state_sha256(model: torch.nn.Module) -> str:
    """Hash one exact in-memory policy snapshot without serializing an artifact."""

    digest = hashlib.sha256()
    for name, tensor in sorted(model.state_dict().items()):
        value = tensor.detach().cpu().contiguous()
        digest.update(name.encode("utf-8"))
        digest.update(str(value.dtype).encode("ascii"))
        digest.update(json.dumps(list(value.shape), separators=(",", ":")).encode("ascii"))
        digest.update(value.numpy().tobytes(order="C"))
    return digest.hexdigest()


def summarize_evaluation(
    episodes: list[EpisodeMetrics],
    *,
    objective: str | None = None,
) -> dict[str, Any]:
    """Aggregate held-out metrics without inventing inapplicable combat wins.

    ``EpisodeMetrics.combat_won`` is authoritative only for a combat-objective
    collector. Full-run episodes do not retain a per-combat numerator, so their
    terminal ``combat_won=False`` must not be presented as a 0% combat win rate.
    """

    if objective not in {None, "combat", "act1", "run"}:
        raise ValueError(
            "evaluation objective must be one of None, 'combat', 'act1', or 'run'"
        )
    combat_win_rate_applicable: bool | None = (
        None if objective is None else objective == "combat"
    )
    evaluation_objective = objective or "unspecified"
    if not episodes:
        return {
            "evaluation_objective": evaluation_objective,
            "combat_win_rate_applicable": combat_win_rate_applicable,
            "episodes": 0,
            "act1_clear_count": 0,
            "act1_clear_rate": 0.0,
            "act3_reach_count": 0,
            "act3_reach_rate": 0.0,
            "run_win_rate": 0.0,
            "combat_win_rate": (
                0.0 if combat_win_rate_applicable is not False else None
            ),
            "deadlock_rate": 0.0,
            "combat_progress_stall_rate": 0.0,
            "combat_policy_failure_count": 0,
            "combat_policy_failure_rate": 0.0,
            "noncombat_progress_stall_rate": 0.0,
            "trusted_policy_failure_count": 0,
            "trusted_policy_failure_rate": 0.0,
            "noncombat_event_cycle_count": 0,
            "noncombat_event_cycle_rate": 0.0,
            "selection_action_cycle_count": 0,
            "selection_action_cycle_rate": 0.0,
            "mean_max_floor": 0.0,
            "maximum_floor": 0,
            "mean_max_act": 0.0,
            "mean_undiscounted_reward_total": 0.0,
            "mean_environment_steps": 0.0,
            "mean_revivals_used": 0.0,
            "mean_player_hp_lost": 0.0,
            "maximum_observed_candidates": 0,
            "maximum_definition_hash_collisions_per_decision": 0,
            "maximum_relation_hash_collisions_per_decision": 0,
            "definition_hash_collisions_total": 0,
            "relation_hash_collisions_total": 0,
            "revival_free_combat_win_rate": (
                0.0 if combat_win_rate_applicable is not False else None
            ),
            "revival_free_act1_clear_rate": 0.0,
            "revival_free_run_win_rate": 0.0,
            "act1_clear_at_most_one_revival_rate": 0.0,
            "run_win_at_most_one_revival_count": 0,
            "run_win_at_most_one_revival_rate": 0.0,
            "successful_run_count": 0,
            "successful_run_mean_revivals": 0.0,
            "successful_run_mean_hp_lost": 0.0,
            "act1_boundary_count": 0,
            "act1_boundary_mean_revivals": 0.0,
            "act1_boundary_mean_hp_lost": 0.0,
        }
    count = len(episodes)
    successful_runs = [item for item in episodes if item.run_won]
    act1_boundaries = [
        (item.act_revival_counts[0], item.act_hp_loss_counts[0])
        for item in episodes
        if item.act_revival_counts and item.act_hp_loss_counts
    ]
    return {
        "evaluation_objective": evaluation_objective,
        "combat_win_rate_applicable": combat_win_rate_applicable,
        "episodes": count,
        "act1_clear_count": sum(item.act1_cleared for item in episodes),
        "act1_clear_rate": sum(item.act1_cleared for item in episodes) / count,
        "act3_reach_count": sum(item.max_act >= 3 for item in episodes),
        "act3_reach_rate": sum(item.max_act >= 3 for item in episodes) / count,
        "run_win_rate": sum(item.run_won for item in episodes) / count,
        "combat_win_rate": (
            sum(item.combat_won for item in episodes) / count
            if combat_win_rate_applicable is not False
            else None
        ),
        "deadlock_rate": sum(item.deadlocked for item in episodes) / count,
        "combat_progress_stall_rate": (sum(item.combat_progress_stalled for item in episodes) / count),
        "combat_policy_failure_count": sum(item.combat_policy_failed for item in episodes),
        "combat_policy_failure_rate": (sum(item.combat_policy_failed for item in episodes) / count),
        "noncombat_progress_stall_rate": (sum(item.noncombat_progress_stalled for item in episodes) / count),
        "trusted_policy_failure_count": sum(item.trusted_policy_failure for item in episodes),
        "trusted_policy_failure_rate": (sum(item.trusted_policy_failure for item in episodes) / count),
        "noncombat_event_cycle_count": sum(item.noncombat_event_cycle for item in episodes),
        "noncombat_event_cycle_rate": (sum(item.noncombat_event_cycle for item in episodes) / count),
        "selection_action_cycle_count": sum(item.selection_action_cycle for item in episodes),
        "selection_action_cycle_rate": (sum(item.selection_action_cycle for item in episodes) / count),
        "mean_max_floor": statistics.fmean(item.max_floor for item in episodes),
        "maximum_floor": max(item.max_floor for item in episodes),
        "mean_max_act": statistics.fmean(item.max_act for item in episodes),
        "mean_undiscounted_reward_total": statistics.fmean(item.reward_total for item in episodes),
        "mean_environment_steps": statistics.fmean(item.steps for item in episodes),
        "mean_revivals_used": statistics.fmean(item.revivals_used for item in episodes),
        "mean_player_hp_lost": statistics.fmean(item.player_hp_lost for item in episodes),
        "maximum_observed_candidates": max(item.maximum_observed_candidates for item in episodes),
        "maximum_definition_hash_collisions_per_decision": max(
            item.maximum_definition_hash_collisions_per_decision
            for item in episodes
        ),
        "maximum_relation_hash_collisions_per_decision": max(
            item.maximum_relation_hash_collisions_per_decision
            for item in episodes
        ),
        "definition_hash_collisions_total": sum(
            item.definition_hash_collisions_total for item in episodes
        ),
        "relation_hash_collisions_total": sum(
            item.relation_hash_collisions_total for item in episodes
        ),
        "revival_free_combat_win_rate": (
            sum(item.revival_free_combat_win for item in episodes) / count
            if combat_win_rate_applicable is not False
            else None
        ),
        "revival_free_act1_clear_rate": (sum(item.revival_free_act1_clear for item in episodes) / count),
        "revival_free_run_win_rate": (sum(item.revival_free_run_win for item in episodes) / count),
        # Completion remains the primary denominator.  Efficiency metrics do
        # not award an early loss merely because it spent fewer revivals.
        "act1_clear_at_most_one_revival_rate": (
            sum(bool(item.act_revival_counts) and item.act_revival_counts[0] <= 1 for item in episodes) / count
        ),
        "run_win_at_most_one_revival_count": sum(item.run_won and item.revivals_used <= 1 for item in episodes),
        "run_win_at_most_one_revival_rate": (
            sum(item.run_won and item.revivals_used <= 1 for item in episodes) / count
        ),
        "successful_run_count": len(successful_runs),
        "successful_run_mean_revivals": (
            statistics.fmean(item.revivals_used for item in successful_runs) if successful_runs else 0.0
        ),
        "successful_run_mean_hp_lost": (
            statistics.fmean(item.player_hp_lost for item in successful_runs) if successful_runs else 0.0
        ),
        "act1_boundary_count": len(act1_boundaries),
        "act1_boundary_mean_revivals": (
            statistics.fmean(item[0] for item in act1_boundaries) if act1_boundaries else 0.0
        ),
        "act1_boundary_mean_hp_lost": (
            statistics.fmean(item[1] for item in act1_boundaries) if act1_boundaries else 0.0
        ),
    }


def evaluate_policy(
    resources: TrainingResources,
    *,
    episodes: int,
    base_seed: int = 0,
    journal_path: str | Path | None = None,
    backend_factory: Callable[[], EnvironmentBackend] | None = None,
    infrastructure_retries_per_seed: int = 1,
    data_partition: str = "validation",
    evaluation_context: Mapping[str, Any] | None = None,
) -> tuple[list[EpisodeMetrics], dict[str, Any]]:
    """Evaluate on disjoint odd seeds with compact diagnostic journals."""

    resources.publish_collector_policy()
    if data_partition == "validation":
        evaluation_seeds = held_out_evaluation_seeds(base_seed, int(episodes))
        namespace = "heldout"
    elif data_partition == "final_audit":
        evaluation_seeds = final_audit_evaluation_seeds(
            base_seed,
            int(episodes),
        )
        namespace = "final-audit"
    else:
        raise ValueError("evaluation data_partition must be validation or final_audit")
    collector_state = deepcopy(resources.collector.state_dict())
    journal = TrajectoryJournal(journal_path) if journal_path is not None else None
    if journal is not None:
        journal.__enter__()
        journal.write_episode_boundary(
            {
                "event": "evaluation_started",
                "data_partition": data_partition,
                "evaluation_seeds": list(evaluation_seeds),
                "epsilon": 0.0,
                "deterministic": True,
                **dict(evaluation_context or {}),
            }
        )
    try:
        results: list[EpisodeMetrics] = []
        infrastructure_retries = 0
        for evaluation_seed in evaluation_seeds:
            attempts = 0
            while True:
                attempt_number = attempts + 1
                if journal is not None:
                    journal.write_episode_boundary(
                        {
                            "event": "evaluation_attempt_started",
                            "evaluation_seed": evaluation_seed,
                            "attempt": attempt_number,
                            "journal_episode_namespace": (
                                f"{namespace}-seed-{evaluation_seed}-attempt-{attempt_number}:"
                            ),
                            "data_partition": data_partition,
                        }
                    )
                try:
                    episode = resources.collector.collect_episode(
                        epsilon=0.0,
                        deterministic=True,
                        record=False,
                        evaluation_seed=evaluation_seed,
                        trajectory_journal=journal,
                        journal_episode_id_prefix=(f"{namespace}-seed-{evaluation_seed}-attempt-{attempt_number}:"),
                    )
                    if journal is not None:
                        journal.write_episode_boundary(
                            {
                                "event": "evaluation_attempt_completed",
                                "evaluation_seed": evaluation_seed,
                                "attempt": attempt_number,
                                "episode_id": (
                                    f"{namespace}-seed-{evaluation_seed}-attempt-"
                                    f"{attempt_number}:{episode.metrics.episode_id}"
                                ),
                                "steps": episode.metrics.steps,
                            }
                        )
                    break
                except BaseException as exc:
                    if journal is not None:
                        journal.write_episode_boundary(
                            {
                                "event": "evaluation_attempt_aborted",
                                "evaluation_seed": evaluation_seed,
                                "attempt": attempt_number,
                                "exception_type": (f"{type(exc).__module__}.{type(exc).__qualname__}"),
                                "fingerprint": str(
                                    getattr(exc, "incident_fingerprint", None)
                                    or getattr(exc, "fingerprint", None)
                                    or ""
                                ),
                                "message": str(exc)[:1024],
                                "retryable": getattr(exc, "recoverable", False) is True,
                            }
                        )
                    if getattr(exc, "recoverable", False) is not True:
                        raise
                    if backend_factory is None or attempts >= infrastructure_retries_per_seed:
                        raise EvaluationInfrastructureError(
                            f"held-out evaluation is infrastructure-invalid for seed={evaluation_seed}: {exc}"
                        ) from exc
                    attempts += 1
                    infrastructure_retries += 1
                    replacement = backend_factory()
                    old_backend = resources.backend
                    try:
                        old_backend.close()
                        resources.backend = replacement
                        resources.collector.replace_backend(replacement)
                    except BaseException:
                        replacement.close()
                        raise
            results.append(episode.metrics)
        summary: dict[str, Any] = dict(
            summarize_evaluation(results, objective=resources.collector.objective)
        )
        summary["infrastructure_retries"] = infrastructure_retries
        summary["data_partition"] = data_partition
        summary["evaluation_seed_count"] = len(evaluation_seeds)
        summary["evaluation_seeds"] = list(evaluation_seeds)
        summary["epsilon"] = 0.0
        summary["deterministic"] = True
        return results, summary
    finally:
        if journal is not None:
            journal.close()
        resources.collector.load_state_dict(collector_state)


def _evaluate_training_gate(
    resources: TrainingResources,
    *,
    episodes: int,
    base_seed: int,
    journal_path: str | Path,
    backend_factory: Callable[[], EnvironmentBackend] | None,
    data_partition: str = "validation",
    evaluation_context: Mapping[str, Any] | None = None,
) -> tuple[list[EpisodeMetrics], dict[str, Any]]:
    """Evaluate one training gate and attach read-only macro diagnostics.

    ``evaluate_policy`` remains the stable policy-evaluation API and publishes
    the learner snapshot to the collector replica before collecting held-out
    episodes. Once that call has closed its journal, this wrapper only reads
    the journal and performs fixed, label-free paired forwards on the same
    published collector replica. Neither diagnostic creates rollout/replay
    records or mutates the learner, optimizer, recurrent collector state, or
    model topology.
    """

    evaluation_results, scalar_summary = evaluate_policy(
        resources,
        episodes=episodes,
        base_seed=base_seed,
        journal_path=journal_path,
        backend_factory=backend_factory,
        data_partition=data_partition,
        evaluation_context=evaluation_context,
    )

    # Keep macro evaluation out of the training module import graph. The
    # diagnostic module also supports standalone checkpoint audits and imports
    # checkpoint/config helpers from this package.
    from sts2_rl.macro_evaluation import (
        evaluate_macro_sensitivity,
        read_macro_journal,
    )

    summary: dict[str, Any] = dict(scalar_summary)
    summary["macro_surface_telemetry"] = read_macro_journal(journal_path)
    summary["macro_policy_sensitivity"] = evaluate_macro_sensitivity(
        resources.collector_model,
        resources.encoder,
    )
    summary["greedy_liveness"] = summarize_greedy_liveness_journal(journal_path)
    summary["evaluation_context"] = dict(evaluation_context or {})
    return evaluation_results, summary


def inspect_baseline(config: TrainingConfig) -> dict[str, Any]:
    """Validate the v2 config/model/encoder without launching a simulator."""

    model_config = config.model.to_model_config()
    model = RecurrentCandidateModel(
        model_config,
        enable_transaction_heads=config.transaction_learning.enabled,
    ).eval()
    encoder = GroundedObservationEncoder(config.model.to_encoding_config())
    decision = encoder.encode(
        {
            "phase": "combat",
            "decision_domain": "combat",
            "player": {"id": "dry-run-player", "hp": 1, "max_hp": 1},
            "combat": {"in_progress": True, "enemies": []},
        },
        [
            {
                "kind": "proceed",
                "model_action_kind": "proceed",
                "is_enabled": True,
                "action_handle": "dry-run",
            }
        ],
    )
    decision.batch.validate(model_config)
    with torch.no_grad():
        output = model(decision.batch, model.initial_state(1))
    reward_identity = (
        revival_efficiency_reward_identity()
        if config.curriculum.mode == "native-revival-preheat"
        else task_reward_identity()
    )
    return {
        "config_version": config.version,
        "profile": config.profile,
        "pipeline": _TRAINING_PIPELINE_ABI,
        "collector_device": config.runtime.collector_device,
        "rocm_sdpa_backend": config.runtime.rocm_sdpa_backend,
        "architecture": config.model.architecture,
        "recurrent_hidden_dim": config.model.recurrent_hidden_dim,
        "unroll_length": config.rollout.unroll_length,
        "rollout_queue_capacity": config.rollout.queue_capacity,
        "rollout_max_policy_lag": config.rollout.max_policy_lag,
        "deterministic_probe_interval_episodes": (config.rollout.deterministic_probe_interval_episodes),
        "deterministic_probe_environment_steps": list(config.rollout.deterministic_probe_environment_steps),
        "transaction_learning": {
            "enabled": config.transaction_learning.enabled,
            "replay_capacity": config.transaction_learning.replay_capacity,
            "replay_byte_capacity": config.transaction_learning.replay_byte_capacity,
            "sample_traces": config.transaction_learning.sample_traces,
            "burn_in_steps": config.transaction_learning.burn_in_steps,
            "effect_weight": config.transaction_learning.effect_weight,
            "transaction_q_weight": config.transaction_learning.transaction_q_weight,
            "completion_policy_weight": (config.transaction_learning.completion_policy_weight),
            "pairwise_ranking_weight": (config.transaction_learning.pairwise_ranking_weight),
        },
        "episodic_learning": {
            "enabled": config.episodic_learning.enabled,
            "replay_capacity_episodes": (config.episodic_learning.replay_capacity_episodes),
            "replay_capacity_bytes": config.episodic_learning.replay_capacity_bytes,
            "per_episode_capacity_bytes": (config.episodic_learning.per_episode_capacity_bytes),
            "sample_sequences": config.episodic_learning.sample_sequences,
            "fresh_policy_sequences": (
                config.episodic_learning.fresh_policy_sequences
            ),
            "burn_in_steps": config.episodic_learning.burn_in_steps,
            "learn_steps": config.episodic_learning.learn_steps,
            "macro_sample_fraction": (config.episodic_learning.macro_sample_fraction),
            "primary_policy_weight": (config.episodic_learning.primary_policy_weight),
            "task_value_weight": config.episodic_learning.task_value_weight,
            "revival_value_weight": (config.episodic_learning.revival_value_weight),
            "revival_policy_weight": (config.episodic_learning.revival_policy_weight),
            "secondary_advantage_fraction": (config.episodic_learning.secondary_advantage_fraction),
            "primary_success_tie_tolerance": (config.episodic_learning.primary_success_tie_tolerance),
            "policy_gradient_max_lag": (config.episodic_learning.policy_gradient_max_lag),
        },
        "encoding_contract": grounding_encoding_identity(),
        "reward_contract": {
            "version": reward_identity["version"],
            "fingerprint_sha256": reward_identity["fingerprint_sha256"],
        },
        "seed_contract": "even-training/odd-held-out-v1",
        "parameters": model.parameter_count,
        "policy_shape": list(output.policy_logits.shape),
        "value_shape": list(output.value.shape),
        "combat_task_value_shape": list(output.combat_task_value.shape),
        "act_task_value_shape": list(output.act_task_value.shape),
        "run_task_value_shape": list(output.run_task_value.shape),
        "combat_revival_cost_value_shape": list(output.combat_revival_cost_value.shape),
        "act_revival_cost_value_shape": list(output.act_revival_cost_value.shape),
        "run_revival_cost_value_shape": list(output.run_revival_cost_value.shape),
        "recurrent_state_shape": list(output.recurrent_state.shape),
        "world_shape": list(decision.batch.world.features.shape),
        "candidate_shape": list(decision.batch.candidates.features.shape),
        "active_shape_batching": True,
        "encoding_capacities": {
            "world": config.model.max_world_tokens,
            "candidates": config.model.max_candidates,
            "candidate_local": config.model.max_candidate_local_tokens,
        },
        "reward_objective": config.curriculum.reward_objective,
        "curriculum_mode": config.curriculum.mode,
        "revival_mechanism": config.curriculum.revival_mechanism,
        "revival_contract": (
            engine_revival_identity()
            if config.curriculum.revival_mechanism is not None
            else None
        ),
        "revival_budget": config.curriculum.revival_budget,
    }


def _checkpoint_path(root: Path, state: TrainingState, *, prefix: str) -> Path:
    return root / f"{prefix}-step-{state.environment_steps:09d}"


def _checkpoint_reference(path: Path | None) -> dict[str, Any] | None:
    if path is None:
        return None
    manifest_path = path / "checkpoint.manifest.json"
    checkpoint_id: str | None = None
    manifest_sha256: str | None = None
    if manifest_path.is_file():
        raw = manifest_path.read_bytes()
        manifest_sha256 = hashlib.sha256(raw).hexdigest()
        value = json.loads(raw)
        if isinstance(value, Mapping):
            raw_id = value.get("checkpoint_id")
            if isinstance(raw_id, str) and raw_id:
                checkpoint_id = raw_id
    return {
        "path": str(path),
        "checkpoint_id": checkpoint_id,
        "manifest_sha256": manifest_sha256,
    }


def _evaluation_context(
    resources: TrainingResources,
    *,
    config: TrainingConfig,
    state: TrainingState,
    evaluation_gate: int,
    gate_kind: str,
    parent_checkpoint: Path | None,
    load_mode: str,
    runtime_provenance: Mapping[str, Any] | None,
) -> dict[str, Any]:
    """Describe the exact in-memory policy and runtime evaluated at one gate."""

    if load_mode == "model_initialization" and state.environment_steps == 0 and state.policy_version == 0:
        checkpoint_relation = "model_parameter_initialization"
    elif parent_checkpoint is not None:
        checkpoint_relation = "in_memory_successor"
    else:
        checkpoint_relation = "fresh_uncheckpointed_policy"
    return {
        "schema_version": "sts2-training-evaluation-context-v1",
        "evaluation_gate": evaluation_gate,
        "gate_kind": gate_kind,
        "actual_environment_steps": state.environment_steps,
        "policy_version": state.policy_version,
        "actor_policy_version": state.actor_policy_version,
        "policy_model_state_sha256": _model_state_sha256(resources.collector_model),
        "checkpoint_association": {
            "relation": checkpoint_relation,
            "load_mode": load_mode,
            "last_committed_checkpoint": _checkpoint_reference(parent_checkpoint),
        },
        "config": {
            "version": config.version,
            "profile": config.profile,
            "fingerprint_sha256": config.fingerprint_sha256(),
        },
        "encoding": grounding_encoding_identity(),
        "simulator": dict(runtime_provenance or {}),
        "epsilon": 0.0,
        "deterministic": True,
    }


def _save(
    resources: TrainingResources,
    *,
    config: TrainingConfig,
    state: TrainingState,
    checkpoint_root: Path,
    prefix: str,
    parent_checkpoint: Path | None,
    run_id: str,
    load_mode: str,
    actor_supervisor_state: ActorSupervisorState,
    evaluation_state: EvaluationGateState,
    runtime_provenance: Mapping[str, Any] | None,
) -> Path:
    return save_training_checkpoint(
        _checkpoint_path(checkpoint_root, state, prefix=prefix),
        config=config,
        resources=resources,
        state=state,
        parent_checkpoint=parent_checkpoint,
        run_id=run_id,
        checkpoint_load_mode=load_mode,
        actor_supervisor_state=actor_supervisor_state,
        evaluation_state=evaluation_state,
        execution_provenance=runtime_provenance,
        parent_relation=(
            "model_parameter_initialization"
            if parent_checkpoint is not None and load_mode == "model_initialization"
            else "in_process_successor"
            if parent_checkpoint is not None and load_mode == "in_process_successor"
            else "loaded_parent"
            if parent_checkpoint is not None
            else None
        ),
    )


def run_training(
    config: TrainingConfig,
    *,
    backend: EnvironmentBackend | None = None,
    resume_from: str | Path | None = None,
    initialize_from: str | Path | None = None,
    runtime_provenance: Mapping[str, Any] | None = None,
) -> TrainingState:
    if resume_from is not None and initialize_from is not None:
        raise ValueError("resume_from and initialize_from are mutually exclusive")
    if runtime_provenance is not None and not isinstance(
        runtime_provenance,
        Mapping,
    ):
        raise TypeError("runtime_provenance must be a mapping or None")

    # Fail before backend launch or artifact creation when checkpoint ABI/device
    # identity is incompatible.
    resolved_device = str(resolve_device(config.runtime.device))
    resolved_actor_device = str(resolve_device(config.runtime.collector_device))
    prevalidated_resume = None
    prevalidated_initialization = None
    if resume_from is not None:
        prevalidated_resume = preflight_training_checkpoint(
            resume_from,
            config=config,
            resolved_device=resolved_device,
            resolved_collector_device=resolved_actor_device,
        )
    if initialize_from is not None:
        prevalidated_initialization = preflight_model_initialization(
            initialize_from,
            config=config,
        )

    run_id = str(uuid4())
    log_root = resolve_artifact_path(config.runtime.log_dir)
    checkpoint_root = resolve_artifact_path(config.runtime.checkpoint_dir)
    run_log_root = log_root / f"run-{run_id}"
    metrics = JsonlMetrics(run_log_root / "metrics.jsonl")
    resources = build_training_resources(config, backend=backend)
    recovery_backend_factory: Callable[[], EnvironmentBackend] | None = (
        None if backend is not None else lambda: build_backend(config)
    )
    pipeline: ActorLearnerPipeline | None = None
    parent_checkpoint: Path | None = None
    load_mode = "fresh"
    state = TrainingState()
    actor_supervisor_state = ActorSupervisorState()
    try:
        if prevalidated_resume is not None:
            parent_checkpoint = prevalidated_resume.root
            load_mode = "exact_resume"
            state = load_training_checkpoint(
                prevalidated_resume.root,
                config=config,
                resources=resources,
            )
            actor_supervisor_state = actor_supervisor_state_from_metadata(prevalidated_resume.metadata)
        elif prevalidated_initialization is not None:
            parent_checkpoint = initialize_model_from_checkpoint(
                prevalidated_initialization.root,
                config=config,
                resources=resources,
            )
            load_mode = "model_initialization"

        previous_sdpa = (
            prevalidated_resume.metadata.get("sdpa_backend")
            if prevalidated_resume is not None
            else None
        )
        resources.sdpa_backend_transition = sdpa_transition_provenance(
            previous=previous_sdpa,
            current=resources.sdpa_backend,
            checkpoint_load_mode=load_mode,
            # Model-parameter initialization does not inherit the source
            # process's execution backend.  Only exact resume describes the
            # old checkpoint as the prior execution context.
            parent_checkpoint_present=(prevalidated_resume is not None),
        )
        structured_runtime_provenance = dict(runtime_provenance or {})
        structured_runtime_provenance["sdpa_backend"] = dict(
            resources.sdpa_backend_transition
        )

        metrics.write(
            "run_start",
            {
                "run_id": run_id,
                "state": asdict(state),
                "actor_supervisor_state": actor_supervisor_state.to_mapping(),
                "config": config.to_mapping(),
                "config_fingerprint_sha256": config.fingerprint_sha256(),
                "runtime_provenance": structured_runtime_provenance,
                "pipeline": _TRAINING_PIPELINE_ABI,
                "checkpoint_load": {
                    "mode": load_mode,
                    "parent_checkpoint": (str(parent_checkpoint) if parent_checkpoint is not None else None),
                    "source_training_state": (
                        prevalidated_initialization.metadata.get("training_state")
                        if prevalidated_initialization is not None
                        else None
                    ),
                    "network_parameters_initialized": (load_mode == "model_initialization"),
                    "optimizer_rollouts_rng_and_counters_reset": (load_mode == "model_initialization"),
                },
            },
        )

        restored_evaluation_state = (
            evaluation_gate_state_from_metadata(prevalidated_resume.metadata)
            if prevalidated_resume is not None
            else None
        )
        if restored_evaluation_state is not None:
            completed_evaluations = set(restored_evaluation_state.completed_validation_steps)
            # Validation schedules are observation-only and may change across
            # an exact continuation.  A newly configured gate at or behind an
            # already published checkpoint is historical; running it now
            # would label the current policy as though it were the old gate's
            # policy.  Runtime checkpoints are only published after every
            # crossed normal/early gate has completed, so this inference is
            # safe for those two live-maintenance schedules.  Final audits are
            # deliberately excluded: they are terminal-policy certificates
            # and a newly configured eligible audit must still run below.
            completed_evaluations.update(
                step for step in config.runtime.evaluation_steps if step <= state.environment_steps
            )
            completed_early_evaluations = set(restored_evaluation_state.completed_early_validation_steps)
            completed_early_evaluations.update(
                step for step in config.runtime.early_evaluation_steps if step <= state.environment_steps
            )
            completed_final_audits = set(restored_evaluation_state.completed_final_audit_steps)
            if config.runtime.total_environment_steps > state.environment_steps:
                # A final audit certifies the terminal policy, not merely that
                # its numeric eligibility threshold was crossed once.  When
                # an exact continuation extends the collection horizon, any
                # restored certificate becomes stale as soon as new learner
                # work is allowed.  Clear it now so the extended run audits
                # the newly drained terminal policy.
                completed_final_audits.clear()
        elif prevalidated_resume is not None:
            # Legacy runtimes published normal/early checkpoints only after
            # those crossed gates completed, so retain their historical
            # inference.  A legacy final audit, however, may have run before
            # queued learner work was drained; its checkpoint therefore cannot
            # certify the terminal policy under the new contract.  Re-audit it
            # conservatively rather than treating a potentially stale audit as
            # complete.  New checkpoints persist all three sets explicitly.
            completed_evaluations = {
                step for step in config.runtime.evaluation_steps if step <= state.environment_steps
            }
            completed_early_evaluations = {
                step for step in config.runtime.early_evaluation_steps if step <= state.environment_steps
            }
            completed_final_audits = set()
        else:
            completed_evaluations = set()
            completed_early_evaluations = set()
            completed_final_audits = set()
        evaluation_guard_stop: dict[str, Any] | None = None

        def current_evaluation_state() -> EvaluationGateState:
            return EvaluationGateState(
                completed_validation_steps=tuple(
                    sorted(completed_evaluations.intersection(config.runtime.evaluation_steps))
                ),
                completed_early_validation_steps=tuple(
                    sorted(completed_early_evaluations.intersection(config.runtime.early_evaluation_steps))
                ),
                completed_final_audit_steps=tuple(
                    sorted(completed_final_audits.intersection(config.runtime.final_audit_steps))
                ),
            )

        def run_one_evaluation(
            *,
            evaluation_step: int,
            episodes: int,
            data_partition: str,
            gate_kind: str,
            journal_name: str,
        ) -> dict[str, Any] | None:
            """Run one frozen gate; evaluation data never enters replay."""

            nonlocal state
            resources.publish_collector_policy()
            context = _evaluation_context(
                resources,
                config=config,
                state=state,
                evaluation_gate=evaluation_step,
                gate_kind=gate_kind,
                parent_checkpoint=parent_checkpoint,
                load_mode=load_mode,
                runtime_provenance=runtime_provenance,
            )
            evaluation_results, summary = _evaluate_training_gate(
                resources,
                episodes=episodes,
                base_seed=config.runtime.seed,
                journal_path=run_log_root / journal_name,
                backend_factory=recovery_backend_factory,
                data_partition=data_partition,
                evaluation_context=context,
            )
            state = replace(
                state,
                evaluation_episodes=state.evaluation_episodes + episodes,
                maximum_observed_candidates=max(
                    state.maximum_observed_candidates,
                    *(item.maximum_observed_candidates for item in evaluation_results),
                ),
            )
            guard = (
                evaluate_liveness_guard(
                    summary["greedy_liveness"],
                    config.runtime,
                )
                if gate_kind == "early_validation"
                else {
                    "schema_version": "sts2-greedy-liveness-guard-v1",
                    "enabled": False,
                    "passed": True,
                    "stop_requested": False,
                    "violations": [],
                    "reason": "guard_applies_only_to_early_validation",
                }
            )
            metrics.write(
                "evaluation",
                {
                    "environment_steps": state.environment_steps,
                    "evaluation_gate": evaluation_step,
                    "gate_kind": gate_kind,
                    "data_partition": data_partition,
                    "run_maximum_observed_candidates": (state.maximum_observed_candidates),
                    "liveness_guard": guard,
                    **summary,
                },
            )
            if guard.get("stop_requested") is True:
                stop = {
                    "evaluation_gate": evaluation_step,
                    "actual_environment_steps": state.environment_steps,
                    "gate_kind": gate_kind,
                    "data_partition": data_partition,
                    "liveness_guard": guard,
                    "policy_version": state.policy_version,
                    "policy_model_state_sha256": context["policy_model_state_sha256"],
                }
                metrics.write("evaluation_guard_stop_requested", stop)
                return stop
            return None

        if (
            0 in config.runtime.evaluation_steps
            and state.environment_steps == 0
            and config.runtime.evaluation_episodes > 0
        ):
            run_one_evaluation(
                evaluation_step=0,
                episodes=config.runtime.evaluation_episodes,
                data_partition="validation",
                gate_kind="validation",
                journal_name="evaluation-step-000000000.jsonl",
            )
            completed_evaluations.add(0)

        def has_due_final_audit() -> bool:
            return any(
                step <= state.environment_steps and step not in completed_final_audits
                for step in config.runtime.final_audit_steps
            )

        def has_due_evaluation(*, include_final_audits: bool = False) -> bool:
            return (
                any(
                    step <= state.environment_steps and step not in completed_evaluations
                    for step in config.runtime.evaluation_steps
                )
                or any(
                    step <= state.environment_steps and step not in completed_early_evaluations
                    for step in config.runtime.early_evaluation_steps
                )
                or (include_final_audits and has_due_final_audit())
            )

        def run_due_evaluations(
            *,
            include_final_audits: bool = False,
        ) -> dict[str, Any] | None:
            """Run crossed gates in gate order and stop after first guard failure."""

            pending: list[tuple[int, str, str, int, set[int], str]] = []
            for step in config.runtime.evaluation_steps:
                if step <= state.environment_steps and step not in completed_evaluations:
                    pending.append(
                        (
                            step,
                            "validation",
                            "validation",
                            config.runtime.evaluation_episodes,
                            completed_evaluations,
                            f"evaluation-step-{step:09d}.jsonl",
                        )
                    )
            for step in config.runtime.early_evaluation_steps:
                if step <= state.environment_steps and step not in completed_early_evaluations:
                    pending.append(
                        (
                            step,
                            "validation",
                            "early_validation",
                            config.runtime.early_evaluation_episodes,
                            completed_early_evaluations,
                            f"early-validation-step-{step:09d}.jsonl",
                        )
                    )
            if include_final_audits:
                for step in config.runtime.final_audit_steps:
                    if step <= state.environment_steps and step not in completed_final_audits:
                        pending.append(
                            (
                                step,
                                "final_audit",
                                "final_audit",
                                config.runtime.final_audit_episodes,
                                completed_final_audits,
                                f"final-audit-step-{step:09d}.jsonl",
                            )
                        )
            for (
                step,
                partition,
                kind,
                episodes,
                completed,
                journal_name,
            ) in sorted(pending, key=lambda item: item[0]):
                stop = (
                    run_one_evaluation(
                        evaluation_step=step,
                        episodes=episodes,
                        data_partition=partition,
                        gate_kind=kind,
                        journal_name=journal_name,
                    )
                    if episodes > 0
                    else None
                )
                completed.add(step)
                if stop is not None:
                    return stop
            return None

        pipeline = ActorLearnerPipeline(
            resources,
            total_environment_steps=config.runtime.total_environment_steps,
            starting_environment_steps=state.environment_steps,
            starting_policy_version=state.actor_policy_version,
            epsilon=lambda steps: exploration_epsilon(config, steps),
            starting_episode_count=state.episodes,
            deterministic_probe_interval_episodes=(config.rollout.deterministic_probe_interval_episodes),
            deterministic_probe_environment_steps=(config.rollout.deterministic_probe_environment_steps),
            supervisor_state=actor_supervisor_state,
        )
        next_checkpoint = (
            (state.environment_steps // config.runtime.checkpoint_interval_steps) + 1
        ) * config.runtime.checkpoint_interval_steps
        unrolls_since_publication = 0
        maintenance_requested = False
        pipeline.start()

        def handle_actor_incident(incident: RecoverableActorIncident) -> None:
            """Commit an infrastructure abort and replace its poisoned session."""

            nonlocal state, maintenance_requested
            state = replace(
                state,
                environment_steps=(state.environment_steps + incident.emitted_environment_steps),
                actor_policy_version=incident.actor_policy_version,
                maximum_observed_candidates=max(
                    state.maximum_observed_candidates,
                    incident.maximum_observed_candidates,
                ),
            )
            metrics.write(
                "backend_protocol_incident",
                {
                    **asdict(incident),
                    "environment_steps": state.environment_steps,
                    "learner_updates": state.learner_updates,
                    "policy_version": state.policy_version,
                },
            )
            metrics.write(
                "episode_aborted",
                {
                    "incident_id": incident.incident_id,
                    "reason": "infrastructure_protocol_fault",
                    "environment_steps": state.environment_steps,
                    "emitted_environment_steps": (incident.emitted_environment_steps),
                    "lost_valid_prefix_steps": incident.lost_valid_prefix_steps,
                    "task_terminal_or_reward_fabricated": False,
                },
            )
            if state.environment_steps != pipeline.environment_steps:
                raise RuntimeError(
                    "incident environment-step reconciliation failed: "
                    f"training_state={state.environment_steps} "
                    f"pipeline={pipeline.environment_steps}"
                )
            if incident.circuit_breaker_open:
                metrics.write(
                    "circuit_breaker_open",
                    {
                        "incident_id": incident.incident_id,
                        "fingerprint": incident.fingerprint,
                        "environment_steps": state.environment_steps,
                        "fingerprint_occurrences": (incident.fingerprint_occurrences),
                        "consecutive_incidents": incident.consecutive_incidents,
                        "incidents_last_100_attempts": (incident.incidents_last_100_attempts),
                    },
                )
                raise RuntimeError(f"actor infrastructure circuit breaker opened for {incident.fingerprint}")
            if recovery_backend_factory is None:
                raise RuntimeError(
                    "recoverable actor incident requires an explicit backend "
                    "recovery factory when the runtime backend was injected"
                )
            replacement = recovery_backend_factory()
            try:
                old_session, new_session = pipeline.replace_backend(replacement)
            except BaseException:
                replacement.close()
                raise
            metrics.write(
                "backend_restart",
                {
                    "incident_id": incident.incident_id,
                    "environment_steps": state.environment_steps,
                    "old_session_id": old_session,
                    "new_session_id": new_session,
                    "fresh_backend": True,
                },
            )
            crossed_evaluation = has_due_evaluation()
            crossed_checkpoint = state.environment_steps >= next_checkpoint
            if crossed_evaluation or crossed_checkpoint:
                maintenance_requested = True
                pipeline.request_pause()
            pipeline.release_incident_boundary()

        def learn_rollout_batch(batch: tuple[SequenceUnroll, ...]) -> None:
            """Consume one FIFO batch exactly once, including replay sidecars."""

            nonlocal state, unrolls_since_publication
            if not batch:
                raise ValueError("runtime cannot learn an empty rollout batch")
            update_number = state.learner_updates + 1
            policy_version_before_update = state.policy_version
            batch_environment_steps = sum(len(unroll.steps) for unroll in batch)
            transaction_traces = (
                resources.transaction_replay.sample(config.transaction_learning.sample_traces)
                if resources.transaction_replay is not None
                else ()
            )
            episodic_sampling_started_ns = time.perf_counter_ns()
            episodic_sample = (
                resources.episodic_replay.sample_for_learning(
                    config.episodic_learning.sample_sequences,
                    learn_steps=config.episodic_learning.learn_steps,
                    burn_in_steps=config.episodic_learning.burn_in_steps,
                    macro_sample_fraction=(config.episodic_learning.macro_sample_fraction),
                    current_policy_version=policy_version_before_update,
                    policy_gradient_max_lag=(
                        config.episodic_learning.policy_gradient_max_lag
                    ),
                    fresh_policy_sequences=(
                        config.episodic_learning.fresh_policy_sequences
                    ),
                )
                if resources.episodic_replay is not None
                else None
            )
            episodic_sampling_ms = (
                (time.perf_counter_ns() - episodic_sampling_started_ns)
                / 1_000_000.0
                if resources.episodic_replay is not None
                else None
            )
            episodic_sequences = (
                episodic_sample.sequences if episodic_sample is not None else ()
            )
            episodic_sampling = None
            if episodic_sample is not None:
                episodic_sampling = episodic_sample.diagnostics.to_mapping()
                episodic_sampling["sampling_ms"] = episodic_sampling_ms
            metrics.write(
                "learner_update_start",
                {
                    "update_number": update_number,
                    "environment_steps": pipeline.environment_steps,
                    "policy_version": state.policy_version,
                    "policy_version_before_update": policy_version_before_update,
                    "unrolls": len(batch),
                    "batch_environment_steps": batch_environment_steps,
                    "transaction_traces": len(transaction_traces),
                    "transaction_replay": (
                        resources.transaction_replay.metrics() if resources.transaction_replay is not None else None
                    ),
                    "episodic_sequences": len(episodic_sequences),
                    "episodic_learn_steps": sum(len(sequence.learn_steps) for sequence in episodic_sequences),
                    "episodic_sampling": episodic_sampling,
                    "episodic_replay": (
                        resources.episodic_replay.metrics() if resources.episodic_replay is not None else None
                    ),
                },
            )

            def learner_progress(
                stage: str,
                payload: dict[str, int | float],
                *,
                _update_number: int = update_number,
                _policy_version: int = state.policy_version,
            ) -> None:
                metrics.write(
                    "learner_progress",
                    {
                        "update_number": _update_number,
                        "stage": stage,
                        "environment_steps": pipeline.environment_steps,
                        "policy_version": _policy_version,
                        **payload,
                    },
                )

            learner_metrics = resources.learner.update(
                batch,
                current_policy_version=policy_version_before_update,
                transaction_traces=transaction_traces,
                episodic_sequences=episodic_sequences,
                progress=learner_progress,
            )
            state = replace(
                state,
                learner_updates=state.learner_updates + 1,
                policy_version=state.policy_version + 1,
                consumed_unrolls=state.consumed_unrolls + len(batch),
            )
            unrolls_since_publication += len(batch)
            metrics.write(
                "learner_update",
                {
                    "environment_steps": pipeline.environment_steps,
                    "policy_version": state.policy_version,
                    "policy_version_before_update": policy_version_before_update,
                    "policy_version_after_update": state.policy_version,
                    "actor_progress": (
                        asdict(pipeline.actor_progress) if pipeline.actor_progress is not None else None
                    ),
                    "rollout_queue": resources.rollout_queue.metrics(),
                    "episodic_sampling": episodic_sampling,
                    "episodic_replay": (
                        resources.episodic_replay.metrics()
                        if resources.episodic_replay is not None
                        else None
                    ),
                    **learner_metrics.to_mapping(),
                },
            )
            if unrolls_since_publication >= config.rollout.policy_sync_interval_unrolls and pipeline.alive:
                pipeline.request_policy_publication(state.policy_version)
                unrolls_since_publication = 0

        # With episodic learning enabled, retain at most one fetched FIFO batch.
        # A following batch proves the retained one was not the episode's tail;
        # an episode/incident boundary makes the tail outcome known.  This keeps
        # actor/learner overlap and queue backpressure bounded while ensuring a
        # one-episode run inserts its completed replay item before its final
        # online batch samples that replay.  The local batch is always flushed
        # before maintenance/final checkpointing and is never consumed twice.
        pending_batch: tuple[SequenceUnroll, ...] = ()
        while pipeline.alive or len(resources.rollout_queue) > 0 or bool(pending_batch):
            try:
                fetched_batch = resources.rollout_queue.get_batch(
                    config.optimization.batch_unrolls,
                    minimum=min(
                        config.optimization.batch_unrolls,
                        config.rollout.minimum_unrolls,
                    ),
                    timeout=0.20,
                )
            except TimeoutError:
                fetched_batch = ()

            batch_to_learn: tuple[SequenceUnroll, ...] = ()
            if resources.episodic_replay is None:
                batch_to_learn = fetched_batch
            elif fetched_batch:
                batch_to_learn = pending_batch
                pending_batch = fetched_batch

            boundary_committed = False
            while True:
                actor_result = pipeline.next_episode(timeout=0.0)
                if actor_result is None:
                    break
                if isinstance(actor_result, RecoverableActorIncident):
                    handle_actor_incident(actor_result)
                    boundary_committed = True
                    continue
                episode = actor_result
                transaction_traces_stored = 0
                if resources.transaction_replay is not None:
                    transaction_traces_stored = sum(
                        int(resources.transaction_replay.put(trace)) for trace in episode.transaction_traces
                    )
                elif episode.transaction_traces:
                    raise RuntimeError("collector emitted transaction traces while replay is disabled")
                episodic_episode_stored = False
                if resources.episodic_replay is not None:
                    if episode.completed_episode is None:
                        raise RuntimeError(
                            "episodic learning is enabled but the collector emitted no completed episode"
                        )
                    episodic_episode_stored = resources.episodic_replay.put(episode.completed_episode)
                state = replace(
                    state,
                    environment_steps=state.environment_steps + episode.metrics.steps,
                    episodes=state.episodes + 1,
                    actor_policy_version=episode.actor_policy_version,
                    maximum_observed_candidates=max(
                        state.maximum_observed_candidates,
                        episode.metrics.maximum_observed_candidates,
                    ),
                )
                metrics.write(
                    "train_episode",
                    {
                        **asdict(episode.metrics),
                        "environment_steps": state.environment_steps,
                        "learner_updates": state.learner_updates,
                        "policy_version": state.policy_version,
                        "actor_policy_version": state.actor_policy_version,
                        "behavior_policy_version": episode.behavior_policy_version,
                        "liveness_probe": episode.liveness_probe,
                        "data_partition": "training",
                        "deterministic": episode.liveness_probe,
                        "collection_epsilon": (
                            0.0
                            if episode.liveness_probe
                            else exploration_epsilon(
                                config,
                                state.environment_steps,
                            )
                        ),
                        "run_maximum_observed_candidates": (state.maximum_observed_candidates),
                        "epsilon": exploration_epsilon(config, state.environment_steps),
                        "collector_timings": (episode.timings.to_mapping() if episode.timings is not None else None),
                        "rollout_queue": resources.rollout_queue.metrics(),
                        "transaction_traces_emitted": len(episode.transaction_traces),
                        "transaction_traces_stored": transaction_traces_stored,
                        "transaction_learn_steps": sum(len(trace.learn_steps) for trace in episode.transaction_traces),
                        "transaction_q_labels": sum(
                            int(step.q_observed) for trace in episode.transaction_traces for step in trace.learn_steps
                        ),
                        "transaction_replay": (
                            resources.transaction_replay.metrics() if resources.transaction_replay is not None else None
                        ),
                        "episodic_episode_emitted": (episode.completed_episode is not None),
                        "episodic_episode_stored": episodic_episode_stored,
                        "episodic_episode_storage_nbytes": (
                            episode.completed_episode.storage_nbytes() if episode.completed_episode is not None else 0
                        ),
                        "episodic_replay": (
                            resources.episodic_replay.metrics() if resources.episodic_replay is not None else None
                        ),
                    },
                )
                crossed_evaluation = has_due_evaluation()
                crossed_checkpoint = state.environment_steps >= next_checkpoint
                if crossed_evaluation or crossed_checkpoint:
                    maintenance_requested = True
                    if pipeline.alive:
                        pipeline.request_pause()
                boundary_committed = True
                # The actor cannot reset into the next run until metrics and
                # maintenance intent for this completed episode are committed.
                pipeline.release_episode_boundary()

            if batch_to_learn:
                learn_rollout_batch(batch_to_learn)
            if resources.episodic_replay is not None and pending_batch:
                terminal_drain = bool(
                    not pipeline.alive and resources.rollout_queue.closed and len(resources.rollout_queue) == 0
                )
                if boundary_committed or terminal_drain:
                    learn_rollout_batch(pending_batch)
                    pending_batch = ()

            actor_idle = not pipeline.alive or pipeline.paused
            if maintenance_requested and actor_idle:
                resources.publish_collector_policy()
                pipeline.set_paused_policy_version(state.policy_version)
                state = replace(state, actor_policy_version=state.policy_version)
                evaluation_guard_stop = run_due_evaluations()
                if evaluation_guard_stop is not None:
                    checkpoint = _save(
                        resources,
                        config=config,
                        state=state,
                        checkpoint_root=checkpoint_root / f"run-{run_id}",
                        prefix="guard-stop",
                        parent_checkpoint=parent_checkpoint,
                        run_id=run_id,
                        load_mode=load_mode,
                        actor_supervisor_state=pipeline.supervisor_state,
                        evaluation_state=current_evaluation_state(),
                        runtime_provenance=runtime_provenance,
                    )
                    parent_checkpoint = checkpoint
                    load_mode = "in_process_successor"
                    metrics.write(
                        "evaluation_guard_stopped",
                        {
                            **evaluation_guard_stop,
                            "checkpoint": str(checkpoint),
                            "state": asdict(state),
                        },
                    )
                    maintenance_requested = False
                    pipeline.stop()
                    pending_batch = ()
                    break
                if state.environment_steps >= next_checkpoint:
                    checkpoint = _save(
                        resources,
                        config=config,
                        state=state,
                        checkpoint_root=checkpoint_root / f"run-{run_id}",
                        prefix="periodic",
                        parent_checkpoint=parent_checkpoint,
                        run_id=run_id,
                        load_mode=load_mode,
                        actor_supervisor_state=pipeline.supervisor_state,
                        evaluation_state=current_evaluation_state(),
                        runtime_provenance=runtime_provenance,
                    )
                    parent_checkpoint = checkpoint
                    load_mode = "in_process_successor"
                    metrics.write("checkpoint", {"path": str(checkpoint), **asdict(state)})
                    while next_checkpoint <= state.environment_steps:
                        next_checkpoint += config.runtime.checkpoint_interval_steps
                maintenance_requested = False
                if pipeline.alive:
                    pipeline.resume()

        pipeline.join(timeout=30.0)
        # Episode messages can arrive immediately before the actor exits.
        while True:
            actor_result = pipeline.next_episode(timeout=0.0)
            if actor_result is None:
                break
            if isinstance(actor_result, RecoverableActorIncident):
                raise RuntimeError(f"actor exited with an unhandled recoverable incident: {actor_result.incident_id}")
            episode = actor_result
            if resources.transaction_replay is not None:
                for trace in episode.transaction_traces:
                    resources.transaction_replay.put(trace)
            elif episode.transaction_traces:
                raise RuntimeError("collector emitted transaction traces while replay is disabled")
            if resources.episodic_replay is not None:
                if episode.completed_episode is None:
                    raise RuntimeError(
                        "episodic learning is enabled but the collector emitted no completed episode during drain"
                    )
                resources.episodic_replay.put(episode.completed_episode)
            state = replace(
                state,
                environment_steps=state.environment_steps + episode.metrics.steps,
                episodes=state.episodes + 1,
                maximum_observed_candidates=max(
                    state.maximum_observed_candidates,
                    episode.metrics.maximum_observed_candidates,
                ),
            )
            pipeline.release_episode_boundary()
        # Final audits describe the policy persisted below, not an intermediate
        # policy at the collection horizon.  The actor is joined and every
        # queued/pending rollout has been learned before this publication.
        resources.publish_collector_policy()
        state = replace(state, actor_policy_version=state.policy_version)
        if evaluation_guard_stop is None:
            evaluation_guard_stop = run_due_evaluations(include_final_audits=True)
        final_checkpoint = _save(
            resources,
            config=config,
            state=state,
            checkpoint_root=checkpoint_root / f"run-{run_id}",
            prefix="final",
            parent_checkpoint=parent_checkpoint,
            run_id=run_id,
            load_mode=load_mode,
            actor_supervisor_state=pipeline.supervisor_state,
            evaluation_state=current_evaluation_state(),
            runtime_provenance=runtime_provenance,
        )
        metrics.write(
            "run_complete",
            {
                "checkpoint": str(final_checkpoint),
                "completion_status": (
                    "evaluation_guard_stopped" if evaluation_guard_stop is not None else "horizon_complete"
                ),
                "evaluation_guard_stop": evaluation_guard_stop,
                **asdict(state),
            },
        )
        return state
    except KeyboardInterrupt:
        metrics.write("interrupt", {"state": asdict(state)})
        raise
    finally:
        # Closing the bounded queue wakes a producer blocked on backpressure.
        if pipeline is not None and pipeline.alive:
            pipeline.stop()
            try:
                pipeline.join(timeout=30.0)
            except TimeoutError:
                metrics.write("actor_shutdown_timeout", {"state": asdict(state)})
        resources.close()


__all__ = [
    "evaluate_policy",
    "exploration_epsilon",
    "inspect_baseline",
    "run_training",
    "summarize_evaluation",
]
