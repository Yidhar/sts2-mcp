#!/usr/bin/env python3
"""Greedy held-out evaluation for one frozen Hybrid Candidate-Q publication.

This process has no spool and no learner.  It uses the fixed odd held-out seed
prefix, forces both domain exploration schedules to zero through evaluation
ownership, and only writes report telemetry.
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

from sts2_rl.encoding.snapshot import collate_encoded_snapshots
from sts2_rl.joint import HybridCollectionAuthority
from sts2_rl.macro import load_trunk_state
from sts2_rl.models import HybridCandidateQConfig, HybridDomainCandidateQModel
from sts2_rl.training import build_training_resources, load_training_config
from sts2_rl.training.seeding import held_out_evaluation_seeds

HYBRID_PUBLICATION = "sts2-hybrid-publication-v1"


def _summary(rows: list[dict[str, Any]]) -> dict[str, Any]:
    floors = sorted(int(row["max_floor"]) for row in rows)
    return {
        "episodes": len(rows),
        "wins": sum(bool(row["run_won"]) for row in rows),
        "act1_clears": sum(bool(row["act1_cleared"]) for row in rows),
        "floor_p50": floors[len(floors) // 2] if floors else 0,
        "floor_mean": sum(floors) / len(floors) if floors else 0.0,
        "mean_revivals": (
            sum(int(row["revivals_used"]) for row in rows) / len(rows)
            if rows
            else 0.0
        ),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--champion", required=True)
    parser.add_argument("--model-path", required=True)
    parser.add_argument("--episodes", type=int, default=16)
    parser.add_argument("--metrics-out", required=True)
    parser.add_argument("--device", default=None)
    parser.add_argument("--sim-exe", default=None)
    args = parser.parse_args()

    config = load_training_config(profile="preheat", config_path=Path(args.config))
    if args.device:
        config = replace(config, runtime=replace(config.runtime, device=args.device))
    if args.sim_exe:
        config = replace(
            config,
            environment=replace(
                config.environment, backend="headless", sim_exe_path=args.sim_exe
            ),
        )
    resources = build_training_resources(config)
    try:
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

        model = HybridDomainCandidateQModel(
            HybridCandidateQConfig(config.model.to_model_config())
        ).to(resources.device)
        publication = torch.load(
            Path(args.model_path), map_location=resources.device, weights_only=True
        )
        if not isinstance(publication, Mapping) or publication.get("format") != HYBRID_PUBLICATION:
            raise ValueError("evaluation input is not a hybrid publication")
        if publication.get("model_config") != model.hybrid_config.as_mapping():
            raise ValueError("evaluation publication model config differs")
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
            return output.q_values[0], output.recurrent_state.detach()

        authority = HybridCollectionAuthority(
            forward_q=forward,
            initial_state=lambda: model.initial_state(1, device=resources.device),
            epsilon=0.0,
            seed=config.runtime.seed,
            evaluation_ownership=True,
        )
        resources.collector.macro_authority = authority
        seeds = held_out_evaluation_seeds(config.runtime.seed, args.episodes)
        rows: list[dict[str, Any]] = []
        output = Path(args.metrics_out)
        output.parent.mkdir(parents=True, exist_ok=True)
        with output.open("w", encoding="utf-8") as metrics:
            for seed in seeds:
                authority.begin_episode(f"hybrid-eval-{seed}")
                episode = resources.collector.collect_episode(
                    epsilon=0.0,
                    deterministic=True,
                    record=False,
                    evaluation_seed=seed,
                )
                authority.finish_run(
                    f"hybrid-eval-{seed}", environment_steps=episode.metrics.steps
                )
                row = {
                    "event": "hybrid_evaluation_episode",
                    "unix_s": time.time(),
                    "evaluation_seed": seed,
                    "policy_version": int(publication.get("policy_version", 0)),
                    "run_won": bool(episode.metrics.run_won),
                    "act1_cleared": bool(episode.metrics.act1_cleared),
                    "max_floor": int(episode.metrics.max_floor),
                    "steps": int(episode.metrics.steps),
                    "revivals_used": int(episode.metrics.revivals_used),
                    "terminal_reason": episode.metrics.terminal_reason,
                }
                rows.append(row)
                metrics.write(json.dumps(row) + "\n")
                metrics.flush()
            summary = {
                "event": "hybrid_evaluation_summary",
                "unix_s": time.time(),
                "policy_version": int(publication.get("policy_version", 0)),
                "seeds": list(seeds),
                **_summary(rows),
            }
            metrics.write(json.dumps(summary) + "\n")
        print(json.dumps(summary, indent=2))
        return 0
    finally:
        resources.close()


if __name__ == "__main__":
    raise SystemExit(main())
