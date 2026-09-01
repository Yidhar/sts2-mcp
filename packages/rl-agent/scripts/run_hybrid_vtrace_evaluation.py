#!/usr/bin/env python3
"""Greedy held-out evaluation for one frozen hybrid V-trace publication.

This process has no spool and no learner.  It uses the fixed odd held-out seed
prefix, forces both domain exploration schedules to zero through evaluation
ownership, and writes the same bounded readable trajectory journal used by the
established held-out evaluator.  ``record=False`` prevents replay/training
output; the explicitly supplied :class:`TrajectoryJournal` is diagnostic only.
"""

from __future__ import annotations

import argparse
import json
import time
from collections.abc import Mapping
from dataclasses import replace
from pathlib import Path
from typing import Any

import torch

from sts2_baseline import (
    ActFrontloadedDeathRewardCalculator,
    act_frontloaded_death_reward_identity,
)
from sts2_env.headless_sim_bridge_client import resolve_headless_sim_exe
from sts2_rl.encoding.snapshot import collate_encoded_snapshots
from sts2_rl.joint import HybridCollectionAuthority
from sts2_rl.macro import load_trunk_state
from sts2_rl.models import HybridActorCriticConfig, HybridDomainActorCriticModel
from sts2_rl.simulator_identity import verify_headless_simulator
from sts2_rl.training import (
    build_training_resources,
    load_training_config,
    summarize_evaluation,
)
from sts2_rl.training.seeding import held_out_evaluation_seeds
from sts2_rl.training.trajectory import TrajectoryJournal

HYBRID_PUBLICATION = "sts2-hybrid-vtrace-publication-v1"


def _retained_evaluation_rows(
    path: Path, *, replacing_policy_version: int
) -> list[dict[str, Any]]:
    """Keep completed history while replacing a rerun of one policy."""

    try:
        lines = path.read_text(encoding="utf-8").splitlines()
    except OSError:
        return []
    retained: list[dict[str, Any]] = []
    for line in lines:
        try:
            row = json.loads(line)
        except json.JSONDecodeError:
            continue
        if not isinstance(row, dict):
            continue
        if row.get("policy_version") == replacing_policy_version:
            continue
        retained.append(row)
    return retained


def _simulator_provenance(config: Any) -> dict[str, Any]:
    """Return the established journal provenance for the active backend."""

    backend = str(config.environment.backend)
    if backend != "headless":
        return {"backend": backend}
    simulator = verify_headless_simulator(
        resolve_headless_sim_exe(config.environment.sim_exe_path)
    )
    return {
        "backend": "headless",
        "simulator_identity": simulator.to_mapping(),
    }


def _summary(
    rows: list[dict[str, Any]],
    *,
    episode_metrics: list[Any],
    objective: str,
) -> dict[str, Any]:
    floors = sorted(int(row["max_floor"]) for row in rows)
    standard = dict(summarize_evaluation(episode_metrics, objective=objective))
    act1_revivals = [
        int(row["act1_revivals"])
        for row in rows
        if row["act1_cleared"] and row.get("act1_revivals") is not None
    ]
    successful_revivals = [
        int(row["revivals_used"]) for row in rows if row["run_won"]
    ]
    return {
        **standard,
        # Keep the compact aliases consumed by archived hybrid dashboard
        # bridges while making the standard evaluation contract authoritative.
        "episodes": len(rows),
        "wins": sum(bool(row["run_won"]) for row in rows),
        "act1_clears": sum(bool(row["act1_cleared"]) for row in rows),
        "act3_reaches": sum(bool(row["act3_reached"]) for row in rows),
        "floor_p50": floors[len(floors) // 2] if floors else 0,
        "floor_mean": sum(floors) / len(floors) if floors else 0.0,
        "mean_revivals": (
            sum(int(row["revivals_used"]) for row in rows) / len(rows)
            if rows
            else 0.0
        ),
        "mean_act1_revivals_on_clear": (
            sum(act1_revivals) / len(act1_revivals) if act1_revivals else None
        ),
        "mean_revivals_on_success": (
            sum(successful_revivals) / len(successful_revivals)
            if successful_revivals
            else None
        ),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--champion", required=True)
    parser.add_argument("--model-path", required=True)
    parser.add_argument("--episodes", type=int, default=16)
    parser.add_argument(
        "--seed",
        dest="evaluation_seeds",
        action="append",
        type=int,
        help="evaluate this exact seed; repeat to reproduce a small fixed set",
    )
    parser.add_argument("--metrics-out", required=True)
    parser.add_argument(
        "--journal-directory",
        required=True,
        help="canonical dashboard run directory for evaluation-step-*.jsonl",
    )
    parser.add_argument("--device", default=None)
    parser.add_argument("--sim-exe", default=None)
    args = parser.parse_args()

    config = load_training_config(profile="preheat", config_path=Path(args.config))
    revival_budget = config.curriculum.revival_budget
    if (
        isinstance(revival_budget, bool)
        or not isinstance(revival_budget, int)
        or revival_budget <= 0
    ):
        raise ValueError("hybrid reward-v9 evaluation requires a positive revival budget")
    if args.device:
        config = replace(config, runtime=replace(config.runtime, device=args.device))
    if args.sim_exe:
        config = replace(
            config,
            environment=replace(
                config.environment, backend="headless", sim_exe_path=args.sim_exe
            ),
        )
    simulator_provenance = _simulator_provenance(config)
    resources = build_training_resources(config)
    try:
        resources.collector.reward_calculator = ActFrontloadedDeathRewardCalculator(
            objective=config.curriculum.reward_objective,
            discount=config.optimization.discount,
            maximum_episode_steps=config.environment.max_episode_steps,
            revival_budget=revival_budget,
        )
        champion_state = torch.load(
            Path(args.champion) / "network.pt",
            map_location=resources.device,
            weights_only=True,
        )
        legacy_head_groups = (
            "candidate_liveness_cost_head",
            "liveness_cost_value_head",
            "transaction_q_head",
            "macro_surface_candidate_embedding",
            "macro_surface_state_embedding",
            "macro_policy_head",
            "macro_option_value_head",
            "candidate_effect_head",
            "selection_delta_head",
        )
        for legacy_model in (resources.model, resources.collector_model):
            load_trunk_state(
                legacy_model,
                dict(champion_state),
                tolerated_groups=legacy_head_groups,
            )

        model = HybridDomainActorCriticModel(
            HybridActorCriticConfig(config.model.to_model_config())
        ).to(resources.device)
        publication = torch.load(
            Path(args.model_path), map_location=resources.device, weights_only=True
        )
        if not isinstance(publication, Mapping) or publication.get("format") != HYBRID_PUBLICATION:
            raise ValueError("evaluation input is not a hybrid publication")
        if publication.get("model_config") != model.actor_critic_config.as_mapping():
            raise ValueError("evaluation publication model config differs")
        if publication.get("reward_identity") != act_frontloaded_death_reward_identity(
            revival_budget=revival_budget
        ):
            raise ValueError("evaluation publication reward identity differs")
        model.load_state_dict(dict(publication["online"]), strict=True)
        model.eval()

        def forward(snapshot: Any, hidden: Any) -> tuple[torch.Tensor, torch.Tensor]:
            encoded = collate_encoded_snapshots(
                (snapshot,),
                expected_config=resources.encoder.config,
                expected_fingerprint=snapshot.encoding_fingerprint,
                device=resources.device,
            )
            if hidden is None:
                hidden = model.initial_state(1, device=resources.device)
            with torch.no_grad():
                output = model(encoded, hidden, validate=False)
            return output.policy_logits[0], output.recurrent_state.detach()

        authority = HybridCollectionAuthority(
            forward_q=forward,
            initial_state=lambda: model.initial_state(1, device=resources.device),
            epsilon=0.0,
            seed=config.runtime.seed,
            evaluation_ownership=True,
        )
        resources.collector.macro_authority = authority
        seeds = (
            tuple(args.evaluation_seeds)
            if args.evaluation_seeds
            else held_out_evaluation_seeds(config.runtime.seed, args.episodes)
        )
        rows: list[dict[str, Any]] = []
        collected_metrics: list[Any] = []
        output = Path(args.metrics_out)
        output.parent.mkdir(parents=True, exist_ok=True)
        policy_version = int(publication.get("policy_version", 0))
        retained_evaluations = _retained_evaluation_rows(
            output, replacing_policy_version=policy_version
        )
        journal_directory = Path(args.journal_directory)
        journal_directory.mkdir(parents=True, exist_ok=True)
        journal_path = journal_directory / f"evaluation-step-{policy_version:09d}.jsonl"
        # This command represents one complete final evaluation attempt.  A
        # rerun replaces its own partial journal rather than concatenating two
        # attempts under one immutable filename.
        journal_path.unlink(missing_ok=True)
        with (
            output.open("w", encoding="utf-8") as metrics,
            TrajectoryJournal(journal_path) as journal,
        ):
            for retained in retained_evaluations:
                metrics.write(json.dumps(retained) + "\n")
            journal.write_episode_boundary(
                {
                    "event": "evaluation_started",
                    "evaluation_gate": policy_version,
                    "evaluation_attempt": 1,
                    "gate_kind": "validation",
                    "data_partition": "validation",
                    "policy_version": policy_version,
                    "evaluation_seeds": list(seeds),
                    "epsilon": 0.0,
                    "deterministic": True,
                    "simulator": simulator_provenance,
                }
            )
            metrics.write(
                json.dumps(
                    {
                        "event": "hybrid_evaluation_started",
                        "unix_s": time.time(),
                        "policy_version": policy_version,
                        "episodes": len(seeds),
                        "seeds": list(seeds),
                        "journal_name": journal_path.name,
                    }
                )
                + "\n"
            )
            metrics.flush()
            for seed in seeds:
                episode_namespace = f"heldout-seed-{seed}-attempt-1:"
                journal.write_episode_boundary(
                    {
                        "event": "evaluation_attempt_started",
                        "evaluation_seed": seed,
                        "attempt": 1,
                        "journal_episode_namespace": episode_namespace,
                        "data_partition": "validation",
                    }
                )
                authority.begin_episode(f"hybrid-eval-{seed}")
                reward_calculator = resources.collector.reward_calculator
                if not isinstance(reward_calculator, ActFrontloadedDeathRewardCalculator):
                    raise TypeError("hybrid evaluation lost reward-v9 calculator")
                reward_calculator.begin_episode()
                try:
                    episode = resources.collector.collect_episode(
                        epsilon=0.0,
                        deterministic=True,
                        record=False,
                        evaluation_seed=seed,
                        policy_version=policy_version,
                        trajectory_journal=journal,
                        journal_episode_id_prefix=episode_namespace,
                    )
                except BaseException as exc:
                    journal.write_episode_boundary(
                        {
                            "event": "evaluation_attempt_aborted",
                            "evaluation_seed": seed,
                            "attempt": 1,
                            "exception_type": (
                                f"{type(exc).__module__}.{type(exc).__qualname__}"
                            ),
                            "message": str(exc)[:1024],
                            "retryable": getattr(exc, "recoverable", False) is True,
                        }
                    )
                    raise
                authority.finish_run(
                    f"hybrid-eval-{seed}", environment_steps=episode.metrics.steps
                )
                episode_id = f"{episode_namespace}{episode.metrics.episode_id}"
                journal.write_episode_boundary(
                    {
                        "event": "evaluation_attempt_completed",
                        "evaluation_seed": seed,
                        "attempt": 1,
                        "episode_id": episode_id,
                        "steps": episode.metrics.steps,
                    }
                )
                row = {
                    "event": "hybrid_evaluation_episode",
                    "unix_s": time.time(),
                    "evaluation_seed": seed,
                    "policy_version": policy_version,
                    "episode_id": episode_id,
                    "run_won": bool(episode.metrics.run_won),
                    "act1_cleared": bool(episode.metrics.act1_cleared),
                    "act3_reached": bool(episode.metrics.max_act >= 3),
                    "max_act": int(episode.metrics.max_act),
                    "max_floor": int(episode.metrics.max_floor),
                    "steps": int(episode.metrics.steps),
                    "revivals_used": int(episode.metrics.revivals_used),
                    "act1_revivals": (
                        int(episode.metrics.act_revival_counts[0])
                        if episode.metrics.act_revival_counts
                        else None
                    ),
                    "player_hp_lost": float(episode.metrics.player_hp_lost),
                    "deadlocked": bool(episode.metrics.deadlocked),
                    "combat_progress_stalled": bool(
                        episode.metrics.combat_progress_stalled
                    ),
                    "noncombat_progress_stalled": bool(
                        episode.metrics.noncombat_progress_stalled
                    ),
                    "terminal_reason": episode.metrics.terminal_reason,
                }
                rows.append(row)
                collected_metrics.append(episode.metrics)
                metrics.write(json.dumps(row) + "\n")
                metrics.flush()
            summary = {
                "event": "hybrid_evaluation_summary",
                "unix_s": time.time(),
                "policy_version": policy_version,
                "seeds": list(seeds),
                "journal_name": journal_path.name,
                **_summary(
                    rows,
                    episode_metrics=collected_metrics,
                    objective=resources.collector.objective,
                ),
                "reward_identity": publication.get("reward_identity"),
            }
            metrics.write(json.dumps(summary) + "\n")
        print(json.dumps(summary, indent=2))
        return 0
    finally:
        resources.close()


if __name__ == "__main__":
    raise SystemExit(main())
