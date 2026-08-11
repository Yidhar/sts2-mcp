#!/usr/bin/env python3
"""Stage-2/4 spool trainer: ingest collected episodes, train macro Double-Q.

Learner half of the file-decoupled actor/learner split. Consumes pickled
MacroEpisodes from --spool, trains the macro candidate-Q model with the
n-step Double-Q learner, and atomically publishes the online weights to
--save-macro on a fixed cadence so collectors can reload them. Exits after
--stop-after-episodes have been ingested (final save included).
"""

from __future__ import annotations

import argparse
import copy
import json
import pickle
import time
from dataclasses import replace
from pathlib import Path
from typing import Any

import torch

from sts2_rl.encoding.snapshot import collate_encoded_snapshots
from sts2_rl.macro import (
    MacroEpisode,
    MacroQConfig,
    MacroQLearner,
    MacroSequenceReplay,
    MacroStep,
    load_trunk_state,
)
from sts2_rl.training import build_training_resources, load_training_config


def _forward_factory(model: Any, encoder: Any, device: torch.device) -> Any:
    def forward(step: Any, hidden: Any) -> tuple[torch.Tensor, Any]:
        snapshot = step.snapshot if isinstance(step, MacroStep) else step
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
    parser.add_argument("--spool", required=True)
    parser.add_argument("--save-macro", required=True)
    parser.add_argument("--init-macro", default=None)
    parser.add_argument("--stop-after-episodes", type=int, default=1200)
    parser.add_argument("--updates-per-episode", type=int, default=8)
    parser.add_argument("--replay-episodes", type=int, default=128)
    parser.add_argument("--save-interval-episodes", type=int, default=20)
    parser.add_argument("--metrics-out", required=True)
    parser.add_argument("--device", default=None)
    parser.add_argument("--sim-exe", default=None)
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
        raise SystemExit("trainer requires transaction_learning.enabled=true")

    spool = Path(args.spool)
    spool.mkdir(parents=True, exist_ok=True)
    resources = build_training_resources(config)
    try:
        champion_state = torch.load(
            Path(args.champion) / "network.pt",
            map_location=resources.device,
            weights_only=True,
        )
        load_trunk_state(resources.model, dict(champion_state))
        for parameter in resources.model.parameters():
            parameter.requires_grad_(False)

        macro_online = copy.deepcopy(resources.model)
        if args.init_macro is not None and Path(args.init_macro).exists():
            load_trunk_state(
                macro_online,
                dict(
                    torch.load(
                        Path(args.init_macro),
                        map_location=resources.device,
                        weights_only=True,
                    )
                ),
            )
        for parameter in macro_online.parameters():
            parameter.requires_grad_(True)
        macro_target = copy.deepcopy(macro_online)
        for parameter in macro_target.parameters():
            parameter.requires_grad_(False)

        device = resources.device
        forward_online = _forward_factory(macro_online, resources.encoder, device)
        forward_target = _forward_factory(macro_target, resources.encoder, device)
        replay = MacroSequenceReplay(
            capacity_episodes=args.replay_episodes, burn_in=8, window_length=16
        )
        learner = MacroQLearner(
            online_parameters=list(macro_online.parameters()),
            forward_online=forward_online,
            forward_target=forward_target,
            sync_target=lambda: macro_target.load_state_dict(macro_online.state_dict()),
            initial_state=lambda: None,
            replay=replay,
            config=MacroQConfig(),
        )

        def save_macro() -> None:
            save_path = Path(args.save_macro)
            temporary = save_path.with_suffix(save_path.suffix + ".tmp")
            torch.save(macro_online.state_dict(), temporary)
            temporary.replace(save_path)

        ingested = 0
        last_saved_at = 0
        with Path(args.metrics_out).open("a", encoding="utf-8") as metrics_file:
            while ingested < args.stop_after_episodes:
                batch = sorted(spool.glob("*.pkl"))[:32]
                if not batch:
                    time.sleep(5.0)
                    continue
                fresh = 0
                for path in batch:
                    try:
                        with path.open("rb") as handle:
                            episode = pickle.load(handle)
                    except (EOFError, pickle.UnpicklingError, OSError):
                        continue  # partial write: retry next scan
                    path.unlink(missing_ok=True)
                    if isinstance(episode, MacroEpisode):
                        try:
                            replay.put(episode)
                        except ValueError:
                            continue  # duplicate id from a stale spool file
                        fresh += 1
                if not fresh:
                    continue
                ingested += fresh
                update_metrics: dict[str, Any] = {}
                for _ in range(args.updates_per_episode * fresh):
                    update_metrics = learner.update()
                if ingested - last_saved_at >= args.save_interval_episodes:
                    save_macro()
                    last_saved_at = ingested
                metrics_file.write(
                    json.dumps(
                        {
                            "event": "trainer_ingest",
                            "unix_s": time.time(),
                            "ingested_total": ingested,
                            "fresh": fresh,
                            "replay": replay.metrics(),
                            "learner": update_metrics,
                        }
                    )
                    + "\n"
                )
                metrics_file.flush()
        save_macro()
        return 0
    finally:
        resources.close()


if __name__ == "__main__":
    raise SystemExit(main())
