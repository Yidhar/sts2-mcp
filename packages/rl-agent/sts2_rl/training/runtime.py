"""End-to-end training and Act-1 evaluation loop for the new baseline."""

from __future__ import annotations

import json
import statistics
import time
from copy import deepcopy
from dataclasses import asdict, dataclass, replace
from pathlib import Path
from typing import Any
from uuid import uuid4

import torch

from sts2_baseline import baseline_reward_identity
from sts2_rl.artifacts import resolve_artifact_path
from sts2_rl.contracts import EnvironmentBackend
from sts2_rl.encoding import GroundedObservationEncoder, grounding_encoding_identity
from sts2_rl.models import GroundedCandidateModel

from .checkpointing import (
    TrainingState,
    initialize_model_from_checkpoint,
    load_training_checkpoint,
    preflight_model_initialization,
    preflight_training_checkpoint,
    save_training_checkpoint,
    training_state_from_metadata,
)
from .collector import CollectedEpisode, EpisodeMetrics
from .config import TrainingConfig
from .factory import (
    TrainingResources,
    build_training_resources,
    resolve_device,
)
from .overlap import OverlappedCollector
from .seeding import held_out_evaluation_seeds
from .update_schedule import advance_update_credit


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
        path.parent.mkdir(parents=True, exist_ok=True)
        self.path = path

    def write(self, event: str, payload: dict[str, Any]) -> None:
        record = {"event": event, "unix_s": time.time(), **payload}
        with self.path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(record, sort_keys=True, default=str) + "\n")


@dataclass(slots=True)
class _StageTiming:
    count: int = 0
    total_ms: float = 0.0
    min_ms: float = float("inf")
    max_ms: float = 0.0

    def add(self, duration_ms: float) -> None:
        duration_ms = max(0.0, float(duration_ms))
        self.count += 1
        self.total_ms += duration_ms
        self.min_ms = min(self.min_ms, duration_ms)
        self.max_ms = max(self.max_ms, duration_ms)

    def merge(
        self,
        *,
        count: int,
        total_ms: float,
        min_ms: float,
        max_ms: float,
    ) -> None:
        if count <= 0:
            return
        self.count += count
        self.total_ms += max(0.0, float(total_ms))
        self.min_ms = min(self.min_ms, max(0.0, float(min_ms)))
        self.max_ms = max(self.max_ms, max(0.0, float(max_ms)))

    def to_mapping(self) -> dict[str, float | int]:
        return {
            "count": self.count,
            "total_ms": self.total_ms,
            "mean_ms": self.total_ms / self.count,
            "min_ms": self.min_ms,
            "max_ms": self.max_ms,
        }


class _EpisodeTimings:
    """Aggregate hot-path durations without emitting per-step log records."""

    def __init__(self) -> None:
        self._stages: dict[str, _StageTiming] = {}

    def record(self, stage: str, started_ns: int) -> None:
        self.add(stage, (time.perf_counter_ns() - started_ns) / 1_000_000.0)

    def add(self, stage: str, duration_ms: float) -> None:
        self._stages.setdefault(stage, _StageTiming()).add(duration_ms)

    def merge(
        self,
        stage: str,
        *,
        count: int,
        total_ms: float,
        min_ms: float,
        max_ms: float,
    ) -> None:
        self._stages.setdefault(stage, _StageTiming()).merge(
            count=count,
            total_ms=total_ms,
            min_ms=min_ms,
            max_ms=max_ms,
        )

    def to_mapping(self) -> dict[str, Any]:
        return {
            "schema_version": 1,
            "scope": "train_episode",
            "unit": "milliseconds",
            "stages": {
                name: timing.to_mapping()
                for name, timing in sorted(self._stages.items())
            },
        }


@dataclass(slots=True)
class _UpdateProgress:
    state: TrainingState
    latest_learner: dict[str, float] | None = None


def summarize_evaluation(episodes: list[EpisodeMetrics]) -> dict[str, float | int]:
    if not episodes:
        return {
            "episodes": 0,
            "act1_clear_rate": 0.0,
            "run_win_rate": 0.0,
            "combat_win_rate": 0.0,
            "mean_max_floor": 0.0,
            "mean_max_act": 0.0,
            "mean_undiscounted_reward_total": 0.0,
        }
    count = len(episodes)
    return {
        "episodes": count,
        "act1_clear_rate": sum(item.act1_cleared for item in episodes) / count,
        "run_win_rate": sum(item.run_won for item in episodes) / count,
        "combat_win_rate": sum(item.combat_won for item in episodes) / count,
        "mean_max_floor": statistics.fmean(item.max_floor for item in episodes),
        "mean_max_act": statistics.fmean(item.max_act for item in episodes),
        "mean_undiscounted_reward_total": statistics.fmean(
            item.reward_total for item in episodes
        ),
    }


def evaluate_policy(
    resources: TrainingResources,
    *,
    episodes: int,
    base_seed: int = 0,
) -> tuple[list[EpisodeMetrics], dict[str, float | int]]:
    resources.publish_collector_policy()
    evaluation_seeds = held_out_evaluation_seeds(base_seed, int(episodes))
    collector_state = deepcopy(resources.collector.state_dict())
    try:
        results: list[EpisodeMetrics] = []
        for evaluation_seed in evaluation_seeds:
            episode = resources.collector.collect_episode(
                epsilon=0.0,
                deterministic=True,
                record=False,
                evaluation_seed=evaluation_seed,
            )
            results.append(episode.metrics)
        return results, summarize_evaluation(results)
    finally:
        # Evaluation must be observational with respect to the training data
        # stream.  Restore every collector-owned RNG/counter field even when a
        # backend or model error interrupts evaluation midway.
        resources.collector.load_state_dict(collector_state)


def inspect_baseline(config: TrainingConfig) -> dict[str, Any]:
    """Validate config/encoder/model with no game process or mutable artifact."""

    model_config = config.model.to_model_config()
    model = GroundedCandidateModel(model_config).eval()
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
        output = model(decision.batch)
    parameter_count = sum(parameter.numel() for parameter in model.parameters())
    reward_identity = baseline_reward_identity()
    return {
        "config_version": config.version,
        "profile": config.profile,
        "execution_mode": config.runtime.execution_mode,
        "collector_device": config.runtime.collector_device,
        "architecture": config.model.architecture,
        "encoding_contract": grounding_encoding_identity(),
        "reward_contract": {
            "version": reward_identity["version"],
            "fingerprint_sha256": reward_identity["fingerprint_sha256"],
        },
        "seed_contract": "even-training/odd-held-out-v1",
        "parameters": parameter_count,
        "policy_shape": list(output.policy_logits.shape),
        "world_shape": list(decision.batch.world.features.shape),
        "candidate_shape": list(decision.batch.candidates.features.shape),
        "candidate_local_shape": list(decision.batch.candidates.local_features.shape),
        "reward_objective": config.curriculum.reward_objective,
    }


def _next_boundary(current: int, interval: int) -> int:
    return ((current // interval) + 1) * interval


def _save(
    *,
    checkpoint_root: Path,
    name: str,
    config: TrainingConfig,
    resources: TrainingResources,
    state: TrainingState,
    parent_checkpoint: str | Path | None,
    run_id: str,
    checkpoint_load_mode: str,
    parent_relation: str | None,
) -> Path:
    return save_training_checkpoint(
        checkpoint_root / name,
        config=config,
        resources=resources,
        state=state,
        parent_checkpoint=parent_checkpoint,
        run_id=run_id,
        checkpoint_load_mode=checkpoint_load_mode,
        parent_relation=parent_relation,
    )


def _ingest_episode(
    *,
    config: TrainingConfig,
    resources: TrainingResources,
    state: TrainingState,
    episode: CollectedEpisode,
    timings: _EpisodeTimings,
) -> TrainingState:
    if not episode.samples:
        raise RuntimeError("collector produced an empty training episode")
    replay_size_before = len(resources.replay)
    replay_extend_started_ns = time.perf_counter_ns()
    resources.replay.extend(episode.samples)
    timings.record("replay_extend", replay_extend_started_ns)
    return replace(
        state,
        environment_steps=state.environment_steps + episode.metrics.steps,
        episodes=state.episodes + 1,
        update_credit=advance_update_credit(
            current_credit=state.update_credit,
            collected_steps=episode.metrics.steps,
            replay_size_before=replay_size_before,
            replay_size_after=len(resources.replay),
            minimum_replay_size=config.replay.minimum_size,
            warmup_policy=config.runtime.warmup_credit_policy,
        ),
    )


def _run_due_updates(
    *,
    config: TrainingConfig,
    resources: TrainingResources,
    progress: _UpdateProgress,
    timings: _EpisodeTimings,
    max_cycles: int | None = None,
) -> None:
    completed_cycles = 0
    while (
        len(resources.replay) >= config.replay.minimum_size
        and progress.state.update_credit >= config.runtime.train_every_steps
        and (max_cycles is None or completed_cycles < max_cycles)
    ):
        for update_index in range(config.runtime.updates_per_cycle):
            replay_sample_started_ns = time.perf_counter_ns()
            replay_batch = resources.replay.sample(config.optimization.batch_size)
            timings.record("replay_sample", replay_sample_started_ns)
            learner_update_started_ns = time.perf_counter_ns()
            learner_metrics = resources.learner.update(
                replay_batch,
                replay=resources.replay,
            )
            timings.record("learner_update", learner_update_started_ns)
            if learner_metrics.timings is not None:
                for stage, duration_ms in learner_metrics.timings.to_mapping().items():
                    timings.add(f"learner.{stage}", duration_ms)
            progress.latest_learner = learner_metrics.to_mapping()
            cycle_complete = update_index + 1 == config.runtime.updates_per_cycle
            progress.state = replace(
                progress.state,
                learner_updates=progress.state.learner_updates + 1,
                update_credit=(
                    progress.state.update_credit - config.runtime.train_every_steps
                    if cycle_complete
                    else progress.state.update_credit
                ),
            )
        completed_cycles += 1


def run_training(
    config: TrainingConfig,
    *,
    backend: EnvironmentBackend | None = None,
    resume_from: str | Path | None = None,
    initialize_from: str | Path | None = None,
) -> TrainingState:
    if resume_from is not None and initialize_from is not None:
        raise ValueError("resume_from and initialize_from are mutually exclusive")
    resolved_device = resolve_device(config.runtime.device)
    resolved_collector_device = (
        resolve_device(config.runtime.collector_device)
        if config.runtime.execution_mode == "overlap"
        else resolved_device
    )
    if resume_from is not None:
        validated_resume = preflight_training_checkpoint(
            resume_from,
            config=config,
            resolved_device=str(resolved_device),
            resolved_collector_device=str(resolved_collector_device),
        )
        preflight_state = training_state_from_metadata(validated_resume.metadata)
        if preflight_state.environment_steps >= config.runtime.total_environment_steps:
            raise ValueError(
                "resume checkpoint already reached the configured environment-step target; "
                "increase runtime.total_environment_steps before creating a run"
            )
    elif initialize_from is not None:
        preflight_model_initialization(initialize_from, config=config)

    run_id = str(uuid4())
    log_root = resolve_artifact_path(config.runtime.log_dir) / f"run-{run_id}"
    checkpoint_root = (
        resolve_artifact_path(config.runtime.checkpoint_dir) / f"run-{run_id}"
    )
    log_root.mkdir(parents=True, exist_ok=True)
    checkpoint_root.mkdir(parents=True, exist_ok=True)
    metrics = JsonlMetrics(log_root / "metrics.jsonl")
    resources = build_training_resources(config, backend=backend)
    overlapped_collector = (
        OverlappedCollector(resources)
        if config.runtime.execution_mode == "overlap"
        else None
    )
    try:
        state = TrainingState()
        if resume_from is not None:
            state = load_training_checkpoint(
                resume_from,
                config=config,
                resources=resources,
            )
            if state.environment_steps >= config.runtime.total_environment_steps:
                raise ValueError(
                    "resume checkpoint already reached the configured environment-step target; "
                    "increase runtime.total_environment_steps instead of overwriting its final checkpoint"
                )
        elif initialize_from is not None:
            initialize_model_from_checkpoint(
                initialize_from,
                config=config,
                resources=resources,
            )
        metrics.write(
            "run_start",
            {
                "run_id": run_id,
                "state": asdict(state),
                "config": config.to_mapping(),
                "device": str(resources.device),
                "collector_device": str(
                    next(resources.collector_model.parameters()).device
                ),
                "execution_mode": config.runtime.execution_mode,
                "collector_model_is_independent": (
                    resources.collector_model is not resources.model
                ),
                "parameters": sum(
                    parameter.numel() for parameter in resources.model.parameters()
                ),
            },
        )
        next_checkpoint = _next_boundary(
            state.environment_steps,
            config.runtime.checkpoint_interval_steps,
        )
        next_evaluation = _next_boundary(
            state.environment_steps,
            config.runtime.evaluation_interval_steps,
        )
        last_checkpoint: str | Path | None = resume_from or initialize_from
        run_origin = (
            "exact_resume"
            if resume_from is not None
            else "model_initialization"
            if initialize_from is not None
            else "fresh"
        )
        parent_relation = "loaded_parent" if last_checkpoint is not None else None
        collector_rollback_state: dict[str, object] | None = None

        try:
            while state.environment_steps < config.runtime.total_environment_steps:
                episode_timings = _EpisodeTimings()
                if overlapped_collector is None:
                    epsilon = exploration_epsilon(config, state.environment_steps)
                    collector_policy_version = state.learner_updates
                    collector_rollback_state = deepcopy(
                        resources.collector.state_dict()
                    )
                    collect_started_ns = time.perf_counter_ns()
                    episode = resources.collector.collect_episode(
                        epsilon=epsilon,
                        deterministic=False,
                        record=True,
                    )
                    episode_timings.record("collect_episode", collect_started_ns)
                else:
                    if not overlapped_collector.has_pending:
                        overlapped_collector.start(
                            epsilon=exploration_epsilon(
                                config,
                                state.environment_steps,
                            ),
                            policy_version=state.learner_updates,
                        )
                    collected = overlapped_collector.wait()
                    episode = collected.episode
                    epsilon = collected.epsilon
                    collector_policy_version = collected.policy_version
                    episode_timings.add(
                        "collector_policy_publish",
                        collected.policy_publish_ms,
                    )
                    episode_timings.add("collect_episode", collected.collector_ms)
                    episode_timings.add("collector_wait", collected.wait_ms)
                    episode_timings.add(
                        "collector_pre_wait",
                        collected.collector_pre_wait_ms,
                    )
                collector_policy_lag_updates = (
                    state.learner_updates - collector_policy_version
                )
                if episode.timings is not None:
                    for stage, timing in episode.timings.stages.items():
                        episode_timings.merge(
                            f"collector.{stage}",
                            count=timing.count,
                            total_ms=timing.total_ms,
                            min_ms=timing.min_ms,
                            max_ms=timing.max_ms,
                        )
                state = _ingest_episode(
                    config=config,
                    resources=resources,
                    state=state,
                    episode=episode,
                    timings=episode_timings,
                )
                collector_rollback_state = None
                evaluation_due = (
                    config.runtime.evaluation_episodes > 0
                    and state.environment_steps >= next_evaluation
                )
                checkpoint_due = state.environment_steps >= next_checkpoint
                prefetched_next_episode = bool(
                    overlapped_collector is not None
                    and state.environment_steps
                    < config.runtime.total_environment_steps
                    and not evaluation_due
                    and not checkpoint_due
                )
                if prefetched_next_episode:
                    assert overlapped_collector is not None
                    overlapped_collector.start(
                        epsilon=exploration_epsilon(config, state.environment_steps),
                        policy_version=state.learner_updates,
                    )
                update_progress = _UpdateProgress(state=state)
                try:
                    _run_due_updates(
                        config=config,
                        resources=resources,
                        progress=update_progress,
                        timings=episode_timings,
                    )
                except KeyboardInterrupt:
                    state = update_progress.state
                    raise
                state = update_progress.state
                latest_learner = update_progress.latest_learner

                metrics.write(
                    "train_episode",
                    {
                        "run_id": run_id,
                        "environment_steps": state.environment_steps,
                        "learner_updates": state.learner_updates,
                        "update_credit": state.update_credit,
                        "replay_size": len(resources.replay),
                        "epsilon": epsilon,
                        "execution_mode": config.runtime.execution_mode,
                        "collector_policy_version": collector_policy_version,
                        "collector_policy_lag_updates": collector_policy_lag_updates,
                        "prefetched_next_episode": prefetched_next_episode,
                        "episode": asdict(episode.metrics),
                        "learner": latest_learner,
                        "timings": episode_timings.to_mapping(),
                    },
                )

                if evaluation_due:
                    evaluated, summary = evaluate_policy(
                        resources,
                        episodes=config.runtime.evaluation_episodes,
                        base_seed=config.runtime.seed,
                    )
                    state = replace(
                        state,
                        evaluation_episodes=(
                            state.evaluation_episodes + len(evaluated)
                        ),
                    )
                    metrics.write(
                        "evaluation",
                        {
                            "run_id": run_id,
                            "environment_steps": state.environment_steps,
                            "seed_namespace": "odd-held-out-v1",
                            "seed_results": [asdict(item) for item in evaluated],
                            **summary,
                        },
                    )
                    next_evaluation = _next_boundary(
                        state.environment_steps,
                        config.runtime.evaluation_interval_steps,
                    )

                if checkpoint_due:
                    saved = _save(
                        checkpoint_root=checkpoint_root,
                        name=f"step-{state.environment_steps:09d}",
                        config=config,
                        resources=resources,
                        state=state,
                        parent_checkpoint=last_checkpoint,
                        run_id=run_id,
                        checkpoint_load_mode=run_origin,
                        parent_relation=parent_relation,
                    )
                    last_checkpoint = saved
                    parent_relation = "in_process_successor"
                    metrics.write(
                        "checkpoint",
                        {
                            "environment_steps": state.environment_steps,
                            "path": str(saved),
                        },
                    )
                    next_checkpoint = _next_boundary(
                        state.environment_steps,
                        config.runtime.checkpoint_interval_steps,
                    )
            saved = _save(
                checkpoint_root=checkpoint_root,
                name=f"final-step-{state.environment_steps:09d}",
                config=config,
                resources=resources,
                state=state,
                parent_checkpoint=last_checkpoint,
                run_id=run_id,
                checkpoint_load_mode=run_origin,
                parent_relation=parent_relation,
            )
            metrics.write(
                "checkpoint",
                {
                    "environment_steps": state.environment_steps,
                    "path": str(saved),
                    "final": True,
                },
            )
        except KeyboardInterrupt:
            prior_credit = state.update_credit
            prior_cycles = prior_credit // config.runtime.train_every_steps
            if overlapped_collector is None and collector_rollback_state is not None:
                resources.collector.load_state_dict(collector_rollback_state)
                metrics.write(
                    "interrupt_collector_rollback",
                    {
                        "run_id": run_id,
                        "state": asdict(state),
                    },
                )
            if overlapped_collector is not None and overlapped_collector.has_pending:
                # A learner-side interrupt may leave the next episode running.
                # Join and ingest it before reading collector RNG/seed state so
                # the interrupt checkpoint is a quiescent continuation point.
                drained = overlapped_collector.wait()
                drain_timings = _EpisodeTimings()
                drain_timings.add("collector_policy_publish", drained.policy_publish_ms)
                drain_timings.add("collect_episode", drained.collector_ms)
                drain_timings.add("collector_wait", drained.wait_ms)
                drain_timings.add(
                    "collector_pre_wait",
                    drained.collector_pre_wait_ms,
                )
                state = _ingest_episode(
                    config=config,
                    resources=resources,
                    state=state,
                    episode=drained.episode,
                    timings=drain_timings,
                )
                if prior_cycles > 0:
                    interrupt_progress = _UpdateProgress(state=state)
                    _run_due_updates(
                        config=config,
                        resources=resources,
                        progress=interrupt_progress,
                        timings=drain_timings,
                        max_cycles=prior_cycles,
                    )
                    state = interrupt_progress.state
                metrics.write(
                    "interrupt_drain",
                    {
                        "run_id": run_id,
                        "state": asdict(state),
                        "collector_policy_version": drained.policy_version,
                        "completed_prior_update_cycles": prior_cycles,
                        "episode": asdict(drained.episode.metrics),
                        "timings": drain_timings.to_mapping(),
                    },
                )
            elif prior_cycles > 0:
                settle_timings = _EpisodeTimings()
                interrupt_progress = _UpdateProgress(state=state)
                _run_due_updates(
                    config=config,
                    resources=resources,
                    progress=interrupt_progress,
                    timings=settle_timings,
                    max_cycles=prior_cycles,
                )
                state = interrupt_progress.state
                metrics.write(
                    "interrupt_update_settle",
                    {
                        "run_id": run_id,
                        "state": asdict(state),
                        "completed_prior_update_cycles": prior_cycles,
                        "timings": settle_timings.to_mapping(),
                    },
                )
            saved = _save(
                checkpoint_root=checkpoint_root,
                name=f"interrupt-step-{state.environment_steps:09d}",
                config=config,
                resources=resources,
                state=state,
                parent_checkpoint=last_checkpoint,
                run_id=run_id,
                checkpoint_load_mode=run_origin,
                parent_relation=parent_relation,
            )
            metrics.write(
                "interrupted",
                {"state": asdict(state), "path": str(saved)},
            )
            raise

        metrics.write("run_complete", {"run_id": run_id, "state": asdict(state)})
        return state
    finally:
        if overlapped_collector is not None:
            overlapped_collector.shutdown()
        resources.close()


__all__ = [
    "JsonlMetrics",
    "evaluate_policy",
    "exploration_epsilon",
    "inspect_baseline",
    "run_training",
    "summarize_evaluation",
]
