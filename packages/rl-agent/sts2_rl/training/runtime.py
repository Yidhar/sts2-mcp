"""End-to-end training and Act-1 evaluation loop for the new baseline."""

from __future__ import annotations

import json
import statistics
import time
from copy import deepcopy
from dataclasses import asdict, replace
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
from .collector import EpisodeMetrics
from .config import TrainingConfig
from .factory import (
    TrainingResources,
    build_training_resources,
    resolve_device,
)
from .seeding import held_out_evaluation_seeds


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
    if resume_from is not None:
        validated_resume = preflight_training_checkpoint(
            resume_from,
            config=config,
            resolved_device=str(resolved_device),
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

        try:
            while state.environment_steps < config.runtime.total_environment_steps:
                epsilon = exploration_epsilon(config, state.environment_steps)
                episode = resources.collector.collect_episode(
                    epsilon=epsilon,
                    deterministic=False,
                    record=True,
                )
                if not episode.samples:
                    raise RuntimeError("collector produced an empty training episode")
                resources.replay.extend(episode.samples)
                state = replace(
                    state,
                    environment_steps=state.environment_steps + episode.metrics.steps,
                    episodes=state.episodes + 1,
                    update_credit=state.update_credit + episode.metrics.steps,
                )
                latest_learner: dict[str, float] | None = None
                while (
                    len(resources.replay) >= config.replay.minimum_size
                    and state.update_credit >= config.runtime.train_every_steps
                ):
                    state = replace(
                        state,
                        update_credit=(
                            state.update_credit - config.runtime.train_every_steps
                        ),
                    )
                    for _ in range(config.runtime.updates_per_cycle):
                        replay_batch = resources.replay.sample(
                            config.optimization.batch_size
                        )
                        learner_metrics = resources.learner.update(
                            replay_batch,
                            replay=resources.replay,
                        )
                        latest_learner = learner_metrics.to_mapping()
                        state = replace(
                            state,
                            learner_updates=state.learner_updates + 1,
                        )

                metrics.write(
                    "train_episode",
                    {
                        "run_id": run_id,
                        "environment_steps": state.environment_steps,
                        "learner_updates": state.learner_updates,
                        "update_credit": state.update_credit,
                        "replay_size": len(resources.replay),
                        "epsilon": epsilon,
                        "episode": asdict(episode.metrics),
                        "learner": latest_learner,
                    },
                )

                if (
                    config.runtime.evaluation_episodes > 0
                    and state.environment_steps >= next_evaluation
                ):
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

                if state.environment_steps >= next_checkpoint:
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
        resources.close()


__all__ = [
    "JsonlMetrics",
    "evaluate_policy",
    "exploration_epsilon",
    "inspect_baseline",
    "run_training",
    "summarize_evaluation",
]
