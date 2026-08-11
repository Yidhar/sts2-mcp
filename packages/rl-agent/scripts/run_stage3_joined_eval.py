#!/usr/bin/env python3
"""Stage-three joined-live held-out evaluation (reset doc §10 stage 3).

Runs paired held-out seeds through up to three arms:
  champion : the frozen combat champion owns every decision (legacy macro
             behavior included) — the baseline.
  joined   : the same frozen champion owns combat while the trained macro
             candidate-Q model greedily owns every macro surface through the
             collection authority (whole-segment ownership).
  challenger: independently trained macro and combat candidate-Q models own
              their respective domains through the joined authority router.

Both arms are deterministic on the identical fixed odd held-out seed prefix,
so differences are attributable to macro ownership. The harness only reports;
per §10 it never auto-replaces a segment on a small evaluation batch.

Usage (WSL ROCm venv, from the package root):
    python scripts/run_stage3_joined_eval.py \
        --config config/experiments/stage2_isolated_macro_v1.toml \
        --champion <champion checkpoint dir> \
        --macro <stage2-macro-online.pt> \
        --episodes 16 --arms both

For the stage-4 challenger arm also pass ``--combat <combat-online.pt>``;
the macro and combat files are loaded into separate model instances.
"""

from __future__ import annotations

import argparse
import copy
import json
import time
from dataclasses import replace
from pathlib import Path
from typing import Any

import torch

from sts2_rl.encoding.snapshot import collate_encoded_snapshots
from sts2_rl.macro import (
    JoinedCollectionAuthority,
    MacroCollectionAuthority,
    load_trunk_state,
)
from sts2_rl.training import build_training_resources, load_training_config
from sts2_rl.training.seeding import held_out_evaluation_seeds


def _forward_factory(model: Any, encoder: Any, device: torch.device) -> Any:
    def forward(snapshot: Any, hidden: Any) -> tuple[torch.Tensor, Any]:
        encoded = collate_encoded_snapshots(
            (snapshot,),
            expected_config=encoder.config,
            expected_fingerprint=snapshot.encoding_fingerprint,
            device=device,
        )
        if hidden is None:
            hidden = model.initial_state(1, device=device)
        output = model(encoded, hidden, validate=False)
        q_values = output.transaction_q_values
        if q_values is None:
            raise RuntimeError("macro Q model produced no candidate Q values")
        return q_values[0], output.recurrent_state.detach()

    return forward


def _load_champion(resources: Any, champion_dir: Path) -> None:
    champion_state = torch.load(
        champion_dir / "network.pt",
        map_location=resources.device,
        weights_only=True,
    )
    for target in (resources.model, resources.collector_model):
        load_trunk_state(target, dict(champion_state))
    for parameter in resources.model.parameters():
        parameter.requires_grad_(False)


def _run_arm(
    resources: Any,
    *,
    arm: str,
    seeds: tuple[int, ...],
    authority: MacroCollectionAuthority | JoinedCollectionAuthority | None,
    metrics_file: Any,
) -> list[dict[str, Any]]:
    resources.collector.macro_authority = authority
    rows: list[dict[str, Any]] = []
    for seed in seeds:
        if authority is not None:
            authority.begin_episode(f"stage3-{arm}-{seed}")
            authority.record_decisions = True
            authority.decision_log = []
        episode = resources.collector.collect_episode(
            epsilon=0.0,
            deterministic=True,
            record=False,
            evaluation_seed=seed,
        )
        macro_summary: dict[str, Any] | None = None
        if authority is not None:
            if isinstance(authority, JoinedCollectionAuthority):
                domain_episodes = authority.finish_episodes(f"stage3-{arm}-{seed}")
            else:
                domain_episodes = {
                    authority.control_domain: authority.finish_episode(
                        f"stage3-{arm}-{seed}"
                    )
                }
            surface_counts: dict[str, int] = {}
            domain_counts: dict[str, dict[str, int]] = {}
            for control_domain, domain_episode in domain_episodes.items():
                current: dict[str, int] = {}
                if domain_episode is not None:
                    for step in domain_episode.steps:
                        key = f"{step.surface}:{step.branch}"
                        current[key] = current.get(key, 0) + 1
                        surface_counts[key] = surface_counts.get(key, 0) + 1
                domain_counts[control_domain] = current
            macro_summary = {
                "overrides": authority.overrides,
                "declined": authority.declined,
                "mechanical_dispatches": authority.mechanical_dispatches,
                "decision_counts": surface_counts,
                "domain_decision_counts": domain_counts,
                "decision_log": list(authority.decision_log),
            }
        row = {
            "event": "stage3_joined_eval_episode",
            "arm": arm,
            "evaluation_seed": seed,
            "unix_s": time.time(),
            "run_won": episode.metrics.run_won,
            "act1_cleared": episode.metrics.act1_cleared,
            "max_act": episode.metrics.max_act,
            "max_floor": episode.metrics.max_floor,
            "steps": episode.metrics.steps,
            "reward_total": episode.metrics.reward_total,
            "terminal_reason": episode.metrics.terminal_reason,
            "macro": macro_summary,
        }
        rows.append(row)
        metrics_file.write(json.dumps(row) + "\n")
        metrics_file.flush()
    resources.collector.macro_authority = None
    return rows


def _summarize(rows: list[dict[str, Any]]) -> dict[str, Any]:
    if not rows:
        return {}
    floors = sorted(row["max_floor"] for row in rows)
    return {
        "episodes": len(rows),
        "wins": sum(1 for row in rows if row["run_won"]),
        "act1_clears": sum(1 for row in rows if row["act1_cleared"]),
        "floor_p50": floors[len(floors) // 2],
        "floor_mean": sum(floors) / len(floors),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--champion", required=True)
    parser.add_argument("--macro", required=True)
    parser.add_argument(
        "--combat",
        default=None,
        help="independently trained combat candidate-Q model; required for "
        "challenger/all",
    )
    parser.add_argument("--episodes", type=int, default=16)
    parser.add_argument(
        "--arms",
        choices=("both", "champion", "joined", "challenger", "all"),
        default="both",
    )
    parser.add_argument("--device", default=None)
    parser.add_argument("--sim-exe", default=None)
    parser.add_argument("--metrics-out", default="stage3-joined-eval.jsonl")
    args = parser.parse_args()

    config = load_training_config(profile="preheat", config_path=Path(args.config))
    if args.device is not None:
        config = replace(config, runtime=replace(config.runtime, device=str(args.device)))
    if args.sim_exe is not None:
        config = replace(
            config,
            environment=replace(
                config.environment,
                backend="headless",
                sim_exe_path=str(args.sim_exe),
            ),
        )
    if not config.transaction_learning.enabled:
        raise SystemExit("stage-3 requires transaction_learning.enabled=true")

    resources = build_training_resources(config)
    try:
        _load_champion(resources, Path(args.champion))

        macro_model = copy.deepcopy(resources.model)
        macro_state = torch.load(
            Path(args.macro),
            map_location=resources.device,
            weights_only=True,
        )
        load_trunk_state(macro_model, dict(macro_state))
        for parameter in macro_model.parameters():
            parameter.requires_grad_(False)
        macro_model.eval()

        combat_model = None
        if args.arms in ("challenger", "all"):
            if args.combat is None:
                raise SystemExit("--combat is required for challenger/all evaluation")
            combat_model = copy.deepcopy(resources.model)
            combat_state = torch.load(
                Path(args.combat),
                map_location=resources.device,
                weights_only=True,
            )
            load_trunk_state(combat_model, dict(combat_state))
            for parameter in combat_model.parameters():
                parameter.requires_grad_(False)
            combat_model.eval()

        seeds = held_out_evaluation_seeds(config.runtime.seed, args.episodes)
        summaries: dict[str, Any] = {}
        metrics_path = Path(args.metrics_out)
        with metrics_path.open("a", encoding="utf-8") as metrics_file:
            if args.arms in ("both", "champion", "all"):
                rows = _run_arm(
                    resources,
                    arm="champion",
                    seeds=seeds,
                    authority=None,
                    metrics_file=metrics_file,
                )
                summaries["champion"] = _summarize(rows)
            if args.arms in ("both", "joined", "all"):
                authority = MacroCollectionAuthority(
                    forward_q=_forward_factory(macro_model, resources.encoder, resources.device),
                    initial_state=lambda: None,
                    epsilon=0.0,
                    evaluation_ownership=True,
                )
                rows = _run_arm(
                    resources,
                    arm="joined",
                    seeds=seeds,
                    authority=authority,
                    metrics_file=metrics_file,
                )
                summaries["joined"] = _summarize(rows)
            if args.arms in ("challenger", "all"):
                assert combat_model is not None
                # Each domain keeps its own model and recurrent state; the
                # router owns no parameters and is used only for joined play.
                authority = JoinedCollectionAuthority(
                    macro=MacroCollectionAuthority(
                        forward_q=_forward_factory(
                            macro_model, resources.encoder, resources.device
                        ),
                        initial_state=lambda: None,
                        epsilon=0.0,
                        evaluation_ownership=True,
                        control_domain="macro",
                    ),
                    combat=MacroCollectionAuthority(
                        forward_q=_forward_factory(
                            combat_model, resources.encoder, resources.device
                        ),
                        initial_state=lambda: None,
                        epsilon=0.0,
                        evaluation_ownership=True,
                        control_domain="combat",
                    ),
                )
                rows = _run_arm(
                    resources,
                    arm="challenger",
                    seeds=seeds,
                    authority=authority,
                    metrics_file=metrics_file,
                )
                summaries["challenger"] = _summarize(rows)
            summary_row = {
                "event": "stage3_joined_eval_summary",
                "unix_s": time.time(),
                "seeds": list(seeds),
                "summaries": summaries,
            }
            metrics_file.write(json.dumps(summary_row) + "\n")
        print(json.dumps(summary_row, indent=2))
        return 0
    finally:
        resources.close()


if __name__ == "__main__":
    raise SystemExit(main())
