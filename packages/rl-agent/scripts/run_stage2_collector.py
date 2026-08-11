#!/usr/bin/env python3
"""Stage-2/4 spool collector: one sim, authority-owned collection to disk.

Actor half of the file-decoupled actor/learner split. Collects episodes
with the frozen champion driving declined decisions and the macro
candidate-Q model (reloaded periodically from --model-path as the trainer
publishes it) driving owned surfaces with branch-balanced epsilon-greedy.
Each MacroEpisode is pickled atomically into --spool for the trainer.
"""

from __future__ import annotations

import argparse
import copy
import json
import os
import pickle
import time
import uuid
from dataclasses import replace
from pathlib import Path
from typing import Any

import torch

from sts2_rl.encoding.snapshot import collate_encoded_snapshots
from sts2_rl.macro import MacroCollectionAuthority, load_trunk_state
from sts2_rl.training import build_training_resources, load_training_config


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


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--champion", required=True)
    parser.add_argument("--spool", required=True)
    parser.add_argument("--model-path", required=True)
    parser.add_argument(
        "--init-macro",
        default=None,
        help="behavior weights until the trainer's first publication",
    )
    parser.add_argument("--episodes", type=int, default=400)
    parser.add_argument("--epsilon", type=float, default=0.15)
    parser.add_argument("--own-combat", action="store_true")
    parser.add_argument("--model-reload-episodes", type=int, default=5)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--metrics-out", required=True)
    parser.add_argument("--device", default=None)
    parser.add_argument("--sim-exe", default=None)
    args = parser.parse_args()

    config = load_training_config(profile="preheat", config_path=Path(args.config))
    config = replace(config, runtime=replace(config.runtime, seed=int(args.seed)))
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
        raise SystemExit("collector requires transaction_learning.enabled=true")

    spool = Path(args.spool)
    spool.mkdir(parents=True, exist_ok=True)
    resources = build_training_resources(config)
    try:
        champion_state = torch.load(
            Path(args.champion) / "network.pt",
            map_location=resources.device,
            weights_only=True,
        )
        for target in (resources.model, resources.collector_model):
            load_trunk_state(target, dict(champion_state))
        for parameter in resources.model.parameters():
            parameter.requires_grad_(False)

        macro_model = copy.deepcopy(resources.model)
        if args.init_macro is not None and Path(args.init_macro).exists():
            load_trunk_state(
                macro_model,
                dict(
                    torch.load(
                        Path(args.init_macro),
                        map_location=resources.device,
                        weights_only=True,
                    )
                ),
            )
        macro_model.eval()
        model_path = Path(args.model_path)
        model_mtime = 0.0

        def maybe_reload() -> None:
            nonlocal model_mtime
            if not model_path.exists():
                return
            mtime = model_path.stat().st_mtime
            if mtime <= model_mtime:
                return
            try:
                state = torch.load(
                    model_path, map_location=resources.device, weights_only=True
                )
                load_trunk_state(macro_model, dict(state))
                macro_model.eval()
                model_mtime = mtime
            except (RuntimeError, EOFError, pickle.UnpicklingError):
                pass  # mid-write read: retry on the next reload tick

        maybe_reload()
        # Episode ids must be unique across collector restarts sharing one
        # spool: the replay contract refuses duplicate ids.
        run_nonce = uuid.uuid4().hex[:8]
        authority = MacroCollectionAuthority(
            forward_q=_forward_factory(macro_model, resources.encoder, resources.device),
            initial_state=lambda: None,
            epsilon=args.epsilon,
            seed=int(args.seed),
            own_combat=bool(args.own_combat),
        )
        resources.collector.macro_authority = authority

        with Path(args.metrics_out).open("a", encoding="utf-8") as metrics_file:
            for episode_index in range(args.episodes):
                if episode_index % max(args.model_reload_episodes, 1) == 0:
                    maybe_reload()
                episode = resources.collector.collect_episode(
                    epsilon=0.0,
                    deterministic=False,
                    record=False,
                )
                macro_episode = authority.finish_episode(
                    f"spool-{args.seed}-{run_nonce}-{episode_index:05d}"
                )
                if macro_episode is not None:
                    temporary = spool / f".tmp-{uuid.uuid4().hex}"
                    with temporary.open("wb") as handle:
                        pickle.dump(macro_episode, handle, protocol=pickle.HIGHEST_PROTOCOL)
                    os.replace(temporary, spool / f"{uuid.uuid4().hex}.pkl")
                metrics_file.write(
                    json.dumps(
                        {
                            "event": "collector_episode",
                            "seed": args.seed,
                            "episode_index": episode_index,
                            "unix_s": time.time(),
                            "steps": episode.metrics.steps,
                            "run_won": episode.metrics.run_won,
                            "max_floor": episode.metrics.max_floor,
                            "macro_steps": (
                                len(macro_episode.steps) if macro_episode else 0
                            ),
                        }
                    )
                    + "\n"
                )
                metrics_file.flush()
        return 0
    finally:
        resources.close()


if __name__ == "__main__":
    raise SystemExit(main())
