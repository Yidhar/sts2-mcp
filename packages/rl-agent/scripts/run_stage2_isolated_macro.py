#!/usr/bin/env python3
"""Isolated macro training: frozen combat fallback + candidate Double-Q.

Implements reset doc §10 stage 2. The v47 combat champion is loaded FROZEN
(model initialization semantics; its parameters receive no gradient and the
legacy learner is never invoked). A second model instance provides the
macro candidate-Q function. Every non-macro decision remains owned by the
frozen champion. Combat candidate-Q training is intentionally unavailable
until the explicit encounter/cross-domain target boundary exists.

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
from contextlib import contextmanager
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
    load_trunk_state,
)
from sts2_rl.training import build_training_resources, load_training_config


@contextmanager
def _evaluation_mode(model: Any) -> Any:
    was_training = bool(model.training)
    model.eval()
    try:
        yield
    finally:
        model.train(was_training)


def _forward_factory(
    model: Any,
    encoder: Any,
    device: torch.device,
    *,
    detach_hidden: bool,
) -> Any:
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
        recurrent = output.recurrent_state
        return q_values[0], recurrent.detach() if detach_hidden else recurrent

    return forward


def _batched_forward_factory(model: Any, encoder: Any, device: torch.device, *, detach_hidden: bool) -> Any:
    def forward(snapshots: Any, hidden: Any) -> tuple[torch.Tensor, Any]:
        encoded = collate_encoded_snapshots(
            tuple(snapshots),
            expected_config=encoder.config,
            expected_fingerprint=snapshots[0].encoding_fingerprint,
            device=device,
        )
        if hidden is None:
            hidden = model.initial_state(len(snapshots), device=device)
        output = model(encoded, hidden, validate=False)
        q_values = output.transaction_q_values
        if q_values is None:
            raise RuntimeError("macro Q model produced no candidate Q values")
        recurrent = output.recurrent_state
        return q_values, recurrent.detach() if detach_hidden else recurrent

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
        "--control-domain",
        choices=("macro",),
        default="macro",
        help="train the independently parameterized macro control domain",
    )
    parser.add_argument(
        "--replay-episodes",
        type=int,
        default=512,
        help="macro replay capacity in complete episodes",
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
        # Tolerated differences: retired liveness heads and the deliberately
        # reshaped candidate-Q head; anything else is a real drift refusal.
        for target in (resources.model, resources.collector_model):
            load_trunk_state(target, dict(champion_state))
        for parameter in resources.model.parameters():
            parameter.requires_grad_(False)

        macro_online = copy.deepcopy(resources.model)
        if args.init_macro is not None:
            init_report = load_trunk_state(
                macro_online,
                dict(
                    torch.load(
                        Path(args.init_macro),
                        map_location=resources.device,
                        weights_only=True,
                    )
                ),
            )
            if init_report["dropped"] or init_report["fresh"]:
                print(
                    "[stage2] init-macro head restart:",
                    {k: len(v) for k, v in init_report.items()},
                )
        for parameter in macro_online.parameters():
            parameter.requires_grad_(True)
        # Candidate Q is the baseline; stochastic regularizers make both the
        # target and behavior values depend on an unrelated dropout draw.
        macro_online.eval()
        macro_target = copy.deepcopy(macro_online)
        macro_target.eval()
        for parameter in macro_target.parameters():
            parameter.requires_grad_(False)

        device = resources.device
        forward_online = _forward_factory(macro_online, resources.encoder, device, detach_hidden=False)
        forward_target = _forward_factory(macro_target, resources.encoder, device, detach_hidden=True)
        forward_behavior = _forward_factory(macro_online, resources.encoder, device, detach_hidden=True)

        def sync_target() -> None:
            macro_target.load_state_dict(macro_online.state_dict())

        def initial_state() -> Any:
            return None

        replay = MacroSequenceReplay(
            capacity_episodes=args.replay_episodes,
            window_length=16,
            control_domain=args.control_domain,
        )
        learner = MacroQLearner(
            online_parameters=list(macro_online.parameters()),
            forward_online=forward_online,
            forward_target=forward_target,
            sync_target=sync_target,
            initial_state=initial_state,
            replay=replay,
            config=MacroQConfig(),
            forward_online_batch=_batched_forward_factory(macro_online, resources.encoder, device, detach_hidden=False),
            forward_target_batch=_batched_forward_factory(macro_target, resources.encoder, device, detach_hidden=True),
            target_evaluation_context=lambda: _evaluation_mode(macro_target),
        )
        authority = MacroCollectionAuthority(
            forward_q=forward_behavior,
            initial_state=initial_state,
            epsilon=args.epsilon,
            control_domain=args.control_domain,
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
                macro_episode = authority.finish_episode(f"stage2-{episode_index:05d}")
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
