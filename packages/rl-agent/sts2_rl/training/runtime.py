"""Asynchronous actor/V-trace runtime with fixed held-out evaluation schedules."""

from __future__ import annotations

import json
import statistics
import time
from collections.abc import Callable
from copy import deepcopy
from dataclasses import asdict, replace
from pathlib import Path
from typing import Any
from uuid import uuid4

import torch

from sts2_baseline import (
    revival_efficiency_reward_identity,
    task_reward_identity,
)
from sts2_rl.artifacts import resolve_artifact_path
from sts2_rl.contracts import EnvironmentBackend
from sts2_rl.encoding import GroundedObservationEncoder, grounding_encoding_identity
from sts2_rl.models import RecurrentCandidateModel

from .checkpointing import (
    ActorSupervisorState,
    TrainingState,
    actor_supervisor_state_from_metadata,
    initialize_model_from_checkpoint,
    load_training_checkpoint,
    preflight_model_initialization,
    preflight_training_checkpoint,
    save_training_checkpoint,
)
from .collector import EpisodeMetrics
from .config import TrainingConfig
from .factory import (
    TrainingResources,
    build_backend,
    build_training_resources,
    resolve_device,
)
from .pipeline import ActorLearnerPipeline, RecoverableActorIncident
from .seeding import held_out_evaluation_seeds
from .trajectory import TrajectoryJournal


def exploration_epsilon(config: TrainingConfig, environment_steps: int) -> float:
    curriculum = config.curriculum
    if environment_steps <= 0:
        return float(curriculum.epsilon_start)
    if environment_steps >= curriculum.epsilon_decay_steps:
        return float(curriculum.epsilon_end)
    progress = min(1.0, max(0.0, environment_steps / curriculum.epsilon_decay_steps))
    return float(
        curriculum.epsilon_start
        + progress * (curriculum.epsilon_end - curriculum.epsilon_start)
    )


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


def summarize_evaluation(episodes: list[EpisodeMetrics]) -> dict[str, float | int]:
    if not episodes:
        return {
            "episodes": 0,
            "act1_clear_count": 0,
            "act1_clear_rate": 0.0,
            "act3_reach_count": 0,
            "act3_reach_rate": 0.0,
            "run_win_rate": 0.0,
            "combat_win_rate": 0.0,
            "deadlock_rate": 0.0,
            "combat_progress_stall_rate": 0.0,
            "noncombat_progress_stall_rate": 0.0,
            "mean_max_floor": 0.0,
            "maximum_floor": 0,
            "mean_max_act": 0.0,
            "mean_undiscounted_reward_total": 0.0,
            "mean_environment_steps": 0.0,
            "mean_revivals_used": 0.0,
            "mean_player_hp_lost": 0.0,
            "maximum_observed_candidates": 0,
            "revival_free_combat_win_rate": 0.0,
            "revival_free_act1_clear_rate": 0.0,
            "revival_free_run_win_rate": 0.0,
        }
    count = len(episodes)
    return {
        "episodes": count,
        "act1_clear_count": sum(item.act1_cleared for item in episodes),
        "act1_clear_rate": sum(item.act1_cleared for item in episodes) / count,
        "act3_reach_count": sum(item.max_act >= 3 for item in episodes),
        "act3_reach_rate": sum(item.max_act >= 3 for item in episodes) / count,
        "run_win_rate": sum(item.run_won for item in episodes) / count,
        "combat_win_rate": sum(item.combat_won for item in episodes) / count,
        "deadlock_rate": sum(item.deadlocked for item in episodes) / count,
        "combat_progress_stall_rate": (
            sum(item.combat_progress_stalled for item in episodes) / count
        ),
        "noncombat_progress_stall_rate": (
            sum(item.noncombat_progress_stalled for item in episodes) / count
        ),
        "mean_max_floor": statistics.fmean(item.max_floor for item in episodes),
        "maximum_floor": max(item.max_floor for item in episodes),
        "mean_max_act": statistics.fmean(item.max_act for item in episodes),
        "mean_undiscounted_reward_total": statistics.fmean(
            item.reward_total for item in episodes
        ),
        "mean_environment_steps": statistics.fmean(item.steps for item in episodes),
        "mean_revivals_used": statistics.fmean(
            item.revivals_used for item in episodes
        ),
        "mean_player_hp_lost": statistics.fmean(
            item.player_hp_lost for item in episodes
        ),
        "maximum_observed_candidates": max(
            item.maximum_observed_candidates for item in episodes
        ),
        "revival_free_combat_win_rate": (
            sum(item.revival_free_combat_win for item in episodes) / count
        ),
        "revival_free_act1_clear_rate": (
            sum(item.revival_free_act1_clear for item in episodes) / count
        ),
        "revival_free_run_win_rate": (
            sum(item.revival_free_run_win for item in episodes) / count
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
) -> tuple[list[EpisodeMetrics], dict[str, float | int]]:
    """Evaluate on fixed odd seeds with compact journals and sparse snapshots."""

    resources.publish_collector_policy()
    evaluation_seeds = held_out_evaluation_seeds(base_seed, int(episodes))
    collector_state = deepcopy(resources.collector.state_dict())
    journal = TrajectoryJournal(journal_path) if journal_path is not None else None
    if journal is not None:
        journal.__enter__()
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
                                f"heldout-seed-{evaluation_seed}-attempt-{attempt_number}:"
                            ),
                        }
                    )
                try:
                    episode = resources.collector.collect_episode(
                        epsilon=0.0,
                        deterministic=True,
                        record=False,
                        evaluation_seed=evaluation_seed,
                        trajectory_journal=journal,
                        journal_episode_id_prefix=(
                            f"heldout-seed-{evaluation_seed}-attempt-{attempt_number}:"
                        ),
                    )
                    if journal is not None:
                        journal.write_episode_boundary(
                            {
                                "event": "evaluation_attempt_completed",
                                "evaluation_seed": evaluation_seed,
                                "attempt": attempt_number,
                                "episode_id": (
                                    f"heldout-seed-{evaluation_seed}-attempt-"
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
                                "exception_type": (
                                    f"{type(exc).__module__}.{type(exc).__qualname__}"
                                ),
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
                    if (
                        backend_factory is None
                        or attempts >= infrastructure_retries_per_seed
                    ):
                        raise EvaluationInfrastructureError(
                            "held-out evaluation is infrastructure-invalid for "
                            f"seed={evaluation_seed}: {exc}"
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
        summary = summarize_evaluation(results)
        summary["infrastructure_retries"] = infrastructure_retries
        return results, summary
    finally:
        if journal is not None:
            journal.close()
        resources.collector.load_state_dict(collector_state)


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
        "pipeline": "bounded-fifo-async-vtrace-v3",
        "collector_device": config.runtime.collector_device,
        "architecture": config.model.architecture,
        "recurrent_hidden_dim": config.model.recurrent_hidden_dim,
        "unroll_length": config.rollout.unroll_length,
        "rollout_queue_capacity": config.rollout.queue_capacity,
        "transaction_learning": {
            "enabled": config.transaction_learning.enabled,
            "replay_capacity": config.transaction_learning.replay_capacity,
            "replay_byte_capacity": config.transaction_learning.replay_byte_capacity,
            "sample_traces": config.transaction_learning.sample_traces,
            "burn_in_steps": config.transaction_learning.burn_in_steps,
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
        "revival_relic_id": config.curriculum.revival_relic_id,
        "revival_budget": config.curriculum.revival_budget,
    }


def _checkpoint_path(root: Path, state: TrainingState, *, prefix: str) -> Path:
    return root / f"{prefix}-step-{state.environment_steps:09d}"


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
) -> TrainingState:
    if resume_from is not None and initialize_from is not None:
        raise ValueError("resume_from and initialize_from are mutually exclusive")

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
            actor_supervisor_state = actor_supervisor_state_from_metadata(
                prevalidated_resume.metadata
            )
        elif prevalidated_initialization is not None:
            parent_checkpoint = initialize_model_from_checkpoint(
                prevalidated_initialization.root,
                config=config,
                resources=resources,
            )
            load_mode = "model_initialization"

        metrics.write(
            "run_start",
            {
                "run_id": run_id,
                "state": asdict(state),
                "actor_supervisor_state": actor_supervisor_state.to_mapping(),
                "config": config.to_mapping(),
                "pipeline": "bounded-fifo-async-vtrace-v3",
                "checkpoint_load": {
                    "mode": load_mode,
                    "parent_checkpoint": (
                        str(parent_checkpoint)
                        if parent_checkpoint is not None
                        else None
                    ),
                    "source_training_state": (
                        prevalidated_initialization.metadata.get("training_state")
                        if prevalidated_initialization is not None
                        else None
                    ),
                    "network_parameters_initialized": (
                        load_mode == "model_initialization"
                    ),
                    "optimizer_rollouts_rng_and_counters_reset": (
                        load_mode == "model_initialization"
                    ),
                },
            },
        )

        completed_evaluations = {
            step
            for step in config.runtime.evaluation_steps
            # A checkpoint is published only after all crossed evaluation
            # gates have completed.  On exact resume, the gate at the exact
            # checkpoint step is therefore already complete as well.  Treat
            # newly configured past gates as historical rather than executing
            # them later under a false gate identity.
            if step <= state.environment_steps
        }
        if (
            0 in config.runtime.evaluation_steps
            and state.environment_steps == 0
            and config.runtime.evaluation_episodes > 0
        ):
            evaluation_results, summary = evaluate_policy(
                resources,
                episodes=config.runtime.evaluation_episodes,
                base_seed=config.runtime.seed,
                journal_path=run_log_root / "evaluation-step-000000000.jsonl",
                backend_factory=recovery_backend_factory,
            )
            state = replace(
                state,
                evaluation_episodes=(
                    state.evaluation_episodes + config.runtime.evaluation_episodes
                ),
                maximum_observed_candidates=max(
                    state.maximum_observed_candidates,
                    *(item.maximum_observed_candidates for item in evaluation_results),
                ),
            )
            completed_evaluations.add(0)
            metrics.write(
                "evaluation",
                {
                    "environment_steps": 0,
                    "run_maximum_observed_candidates": (
                        state.maximum_observed_candidates
                    ),
                    **summary,
                },
            )

        pipeline = ActorLearnerPipeline(
            resources,
            total_environment_steps=config.runtime.total_environment_steps,
            starting_environment_steps=state.environment_steps,
            starting_policy_version=state.actor_policy_version,
            epsilon=lambda steps: exploration_epsilon(config, steps),
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
                environment_steps=(
                    state.environment_steps + incident.emitted_environment_steps
                ),
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
                    "emitted_environment_steps": (
                        incident.emitted_environment_steps
                    ),
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
                        "fingerprint_occurrences": (
                            incident.fingerprint_occurrences
                        ),
                        "consecutive_incidents": incident.consecutive_incidents,
                        "incidents_last_100_attempts": (
                            incident.incidents_last_100_attempts
                        ),
                    },
                )
                raise RuntimeError(
                    "actor infrastructure circuit breaker opened for "
                    f"{incident.fingerprint}"
                )
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
            crossed_evaluation = any(
                step <= state.environment_steps
                and step not in completed_evaluations
                for step in config.runtime.evaluation_steps
            )
            crossed_checkpoint = state.environment_steps >= next_checkpoint
            if crossed_evaluation or crossed_checkpoint:
                maintenance_requested = True
                pipeline.request_pause()
            pipeline.release_incident_boundary()

        while pipeline.alive or len(resources.rollout_queue) > 0:
            try:
                batch = resources.rollout_queue.get_batch(
                    config.optimization.batch_unrolls,
                    minimum=min(
                        config.optimization.batch_unrolls,
                        config.rollout.minimum_unrolls,
                    ),
                    timeout=0.20,
                )
            except TimeoutError:
                batch = ()
            if batch:
                update_number = state.learner_updates + 1
                batch_environment_steps = sum(len(unroll.steps) for unroll in batch)
                transaction_traces = (
                    resources.transaction_replay.sample(
                        config.transaction_learning.sample_traces
                    )
                    if resources.transaction_replay is not None
                    else ()
                )
                metrics.write(
                    "learner_update_start",
                    {
                        "update_number": update_number,
                        "environment_steps": pipeline.environment_steps,
                        "policy_version": state.policy_version,
                        "unrolls": len(batch),
                        "batch_environment_steps": batch_environment_steps,
                        "transaction_traces": len(transaction_traces),
                        "transaction_replay": (
                            resources.transaction_replay.metrics()
                            if resources.transaction_replay is not None
                            else None
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
                    current_policy_version=state.policy_version,
                    transaction_traces=transaction_traces,
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
                        "actor_progress": (
                            asdict(pipeline.actor_progress)
                            if pipeline.actor_progress is not None
                            else None
                        ),
                        "rollout_queue": resources.rollout_queue.metrics(),
                        **learner_metrics.to_mapping(),
                    },
                )
                if (
                    unrolls_since_publication
                    >= config.rollout.policy_sync_interval_unrolls
                    and pipeline.alive
                ):
                    pipeline.request_policy_publication(state.policy_version)
                    unrolls_since_publication = 0

            while True:
                actor_result = pipeline.next_episode(timeout=0.0)
                if actor_result is None:
                    break
                if isinstance(actor_result, RecoverableActorIncident):
                    handle_actor_incident(actor_result)
                    continue
                episode = actor_result
                transaction_traces_stored = 0
                if resources.transaction_replay is not None:
                    transaction_traces_stored = sum(
                        int(resources.transaction_replay.put(trace))
                        for trace in episode.transaction_traces
                    )
                elif episode.transaction_traces:
                    raise RuntimeError(
                        "collector emitted transaction traces while replay is disabled"
                    )
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
                        "run_maximum_observed_candidates": (
                            state.maximum_observed_candidates
                        ),
                        "epsilon": exploration_epsilon(
                            config, state.environment_steps
                        ),
                        "collector_timings": (
                            episode.timings.to_mapping()
                            if episode.timings is not None
                            else None
                        ),
                        "rollout_queue": resources.rollout_queue.metrics(),
                        "transaction_traces_emitted": len(
                            episode.transaction_traces
                        ),
                        "transaction_traces_stored": transaction_traces_stored,
                        "transaction_learn_steps": sum(
                            len(trace.learn_steps)
                            for trace in episode.transaction_traces
                        ),
                        "transaction_q_labels": sum(
                            int(step.q_observed)
                            for trace in episode.transaction_traces
                            for step in trace.learn_steps
                        ),
                        "transaction_replay": (
                            resources.transaction_replay.metrics()
                            if resources.transaction_replay is not None
                            else None
                        ),
                    },
                )
                crossed_evaluation = any(
                    step <= state.environment_steps
                    and step not in completed_evaluations
                    for step in config.runtime.evaluation_steps
                )
                crossed_checkpoint = state.environment_steps >= next_checkpoint
                if crossed_evaluation or crossed_checkpoint:
                    maintenance_requested = True
                    if pipeline.alive:
                        pipeline.request_pause()
                # The actor cannot reset into the next run until metrics and
                # maintenance intent for this completed episode are committed.
                pipeline.release_episode_boundary()

            actor_idle = not pipeline.alive or pipeline.paused
            if maintenance_requested and actor_idle:
                resources.publish_collector_policy()
                pipeline.set_paused_policy_version(state.policy_version)
                state = replace(state, actor_policy_version=state.policy_version)
                for evaluation_step in config.runtime.evaluation_steps:
                    if (
                        evaluation_step <= state.environment_steps
                        and evaluation_step not in completed_evaluations
                    ):
                        if config.runtime.evaluation_episodes > 0:
                            evaluation_results, summary = evaluate_policy(
                                resources,
                                episodes=config.runtime.evaluation_episodes,
                                base_seed=config.runtime.seed,
                                journal_path=(
                                    run_log_root
                                    / f"evaluation-step-{evaluation_step:09d}.jsonl"
                                ),
                                backend_factory=recovery_backend_factory,
                            )
                            state = replace(
                                state,
                                evaluation_episodes=(
                                    state.evaluation_episodes
                                    + config.runtime.evaluation_episodes
                                ),
                                maximum_observed_candidates=max(
                                    state.maximum_observed_candidates,
                                    *(
                                        item.maximum_observed_candidates
                                        for item in evaluation_results
                                    ),
                                ),
                            )
                            metrics.write(
                                "evaluation",
                                {
                                    "environment_steps": state.environment_steps,
                                    "evaluation_gate": evaluation_step,
                                    "run_maximum_observed_candidates": (
                                        state.maximum_observed_candidates
                                    ),
                                    **summary,
                                },
                            )
                        completed_evaluations.add(evaluation_step)
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
                raise RuntimeError(
                    "actor exited with an unhandled recoverable incident: "
                    f"{actor_result.incident_id}"
                )
            episode = actor_result
            if resources.transaction_replay is not None:
                for trace in episode.transaction_traces:
                    resources.transaction_replay.put(trace)
            elif episode.transaction_traces:
                raise RuntimeError(
                    "collector emitted transaction traces while replay is disabled"
                )
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
        resources.publish_collector_policy()
        for evaluation_step in config.runtime.evaluation_steps:
            if (
                evaluation_step <= state.environment_steps
                and evaluation_step not in completed_evaluations
            ):
                if config.runtime.evaluation_episodes > 0:
                    evaluation_results, summary = evaluate_policy(
                        resources,
                        episodes=config.runtime.evaluation_episodes,
                        base_seed=config.runtime.seed,
                        journal_path=(
                            run_log_root
                            / f"evaluation-step-{evaluation_step:09d}.jsonl"
                        ),
                        backend_factory=recovery_backend_factory,
                    )
                    state = replace(
                        state,
                        evaluation_episodes=(
                            state.evaluation_episodes
                            + config.runtime.evaluation_episodes
                        ),
                        maximum_observed_candidates=max(
                            state.maximum_observed_candidates,
                            *(
                                item.maximum_observed_candidates
                                for item in evaluation_results
                            ),
                        ),
                    )
                    metrics.write(
                        "evaluation",
                        {
                            "environment_steps": state.environment_steps,
                            "evaluation_gate": evaluation_step,
                            "run_maximum_observed_candidates": (
                                state.maximum_observed_candidates
                            ),
                            **summary,
                        },
                    )
                completed_evaluations.add(evaluation_step)
        state = replace(state, actor_policy_version=state.policy_version)
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
        )
        metrics.write(
            "run_complete",
            {"checkpoint": str(final_checkpoint), **asdict(state)},
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
