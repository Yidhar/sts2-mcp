#!/usr/bin/env python3
"""Stage-two isolated macro training: frozen champion + macro Double-Q.

Implements reset doc §10 stage 2. The v47 combat champion is loaded FROZEN
(model initialization semantics; its parameters receive no gradient and the
legacy learner is never invoked). A second model instance — initialized from
the same champion weights — provides the macro candidate-Q function via its
transaction_q_values head; only that instance trains, exclusively through the
macro Double-Q learner. The collection authority owns macro surfaces with
branch-balanced epsilon-greedy; every other decision is the champion's.

Usage (WSL ROCm venv, from the package root):
    python scripts/run_stage2_isolated_macro.py \
        --config config/experiments/<stage2 toml> \
        --champion <path to healthy champion checkpoint> \
        --episodes 200 --epsilon 0.2 --updates-per-episode 4
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
    MacroCollectionAuthority,
    MacroQConfig,
    MacroQLearner,
    MacroSequenceReplay,
    MacroStep,
)
from sts2_rl.training import build_training_resources, load_training_config


def _forward_factory(model: Any, encoder: Any, device: torch.device) -> Any:
    fingerprint = None

    def forward(step: Any, hidden: Any) -> tuple[torch.Tensor, Any]:
        nonlocal fingerprint
        # The authority passes an EncodedDecisionSnapshot; the learner passes
        # a MacroStep carrying one.
        snapshot = step.snapshot if isinstance(step, MacroStep) else step
        if fingerprint is None:
            fingerprint = snapshot.encoding_fingerprint
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


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--champion", required=True)
    parser.add_argument("--episodes", type=int, default=200)
    parser.add_argument("--epsilon", type=float, default=0.2)
    parser.add_argument("--updates-per-episode", type=int, default=4)
    parser.add_argument("--metrics-out", default="stage2-macro-metrics.jsonl")
    parser.add_argument("--save-macro", default="stage2-macro-online.pt")
    parser.add_argument("--save-interval-episodes", type=int, default=25)
    parser.add_argument("--device", default=None)
    parser.add_argument("--sim-exe", default=None)
    parser.add_argument(
        "--own-combat",
        action="store_true",
        help="stage-4 combat challenger: the authority also owns the "
        "native-atomic combat view (default: macro surfaces only)",
    )
    parser.add_argument(
        "--init-macro",
        default=None,
        help="initialize the trainable macro Q model from a previously "
        "saved state dict (e.g. the stage-2 result) instead of the "
        "champion weights",
    )
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
        raise SystemExit(
            "stage-2 requires transaction_learning.enabled=true: the macro "
            "candidate-Q function is the model's transaction Q head"
        )
    resources = build_training_resources(config)
    try:
        champion_state = torch.load(
            Path(args.champion) / "network.pt",
            map_location=resources.device,
            weights_only=True,
        )
        # The champion lineage trained with liveness heads (failure_credit
        # learning); stage-2 retires that plane, so its head tensors are the
        # only tolerated difference. Anything else is a real drift refusal.
        for target in (resources.model, resources.collector_model):
            result = target.load_state_dict(champion_state, strict=False)
            unexpected = [
                key for key in result.unexpected_keys if "liveness" not in key
            ]
            if result.missing_keys or unexpected:
                raise SystemExit(
                    "champion checkpoint drift beyond the retired liveness "
                    f"heads: missing={result.missing_keys} unexpected={unexpected}"
                )
        for parameter in resources.model.parameters():
            parameter.requires_grad_(False)

        macro_online = copy.deepcopy(resources.model)
        if args.init_macro is not None:
            macro_online.load_state_dict(
                torch.load(
                    Path(args.init_macro),
                    map_location=resources.device,
                    weights_only=True,
                )
            )
        for parameter in macro_online.parameters():
            parameter.requires_grad_(True)
        macro_target = copy.deepcopy(macro_online)
        for parameter in macro_target.parameters():
            parameter.requires_grad_(False)

        device = resources.device
        forward_online = _forward_factory(macro_online, resources.encoder, device)
        forward_target = _forward_factory(macro_target, resources.encoder, device)

        def sync_target() -> None:
            macro_target.load_state_dict(macro_online.state_dict())

        def initial_state() -> Any:
            return None

        replay = MacroSequenceReplay(capacity_episodes=512, burn_in=8, window_length=16)
        learner = MacroQLearner(
            online_parameters=list(macro_online.parameters()),
            forward_online=forward_online,
            forward_target=forward_target,
            sync_target=sync_target,
            initial_state=initial_state,
            replay=replay,
            config=MacroQConfig(),
        )
        authority = MacroCollectionAuthority(
            forward_q=forward_online,
            initial_state=initial_state,
            epsilon=args.epsilon,
            own_combat=bool(args.own_combat),
        )
        resources.collector.macro_authority = authority

        def save_macro() -> None:
            save_path = Path(args.save_macro)
            temporary = save_path.with_suffix(save_path.suffix + ".tmp")
            torch.save(macro_online.state_dict(), temporary)
            temporary.replace(save_path)

        metrics_path = Path(args.metrics_out)
        with metrics_path.open("a", encoding="utf-8") as metrics_file:
            for episode_index in range(args.episodes):
                # record=False: legacy FIFO/replay sidecars stay empty; the
                # macro authority records the only training experience.
                episode = resources.collector.collect_episode(
                    epsilon=0.0,
                    deterministic=False,
                    record=False,
                )
                macro_episode = authority.finish_episode(
                    f"stage2-{episode_index:05d}"
                )
                if macro_episode is not None:
                    replay.put(macro_episode)
                update_metrics: dict[str, Any] = {}
                for _ in range(args.updates_per_episode):
                    update_metrics = learner.update()
                record = {
                    "event": "stage2_macro_episode",
                    "episode_index": episode_index,
                    "unix_s": time.time(),
                    "authority": authority.metrics(),
                    "replay": replay.metrics(),
                    "learner": update_metrics,
                    "episode_steps": episode.metrics.steps,
                    "episode_reward_total": episode.metrics.reward_total,
                    "episode_run_won": episode.metrics.run_won,
                    "episode_max_act": episode.metrics.max_act,
                    "episode_max_floor": episode.metrics.max_floor,
                }
                metrics_file.write(json.dumps(record) + "\n")
                metrics_file.flush()
                if (episode_index + 1) % args.save_interval_episodes == 0:
                    save_macro()
        save_macro()
        return 0
    finally:
        resources.close()


if __name__ == "__main__":
    raise SystemExit(main())
