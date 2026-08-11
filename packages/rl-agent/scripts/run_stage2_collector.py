#!/usr/bin/env python3
"""Stage-2 macro producer for one explicitly identified pipeline segment."""

from __future__ import annotations

import argparse
import copy
import json
import os
import pickle
import time
import uuid
from collections.abc import Mapping
from dataclasses import replace
from pathlib import Path
from typing import Any

import torch

from sts2_env.headless_sim_bridge_client import HeadlessSimError
from sts2_rl.backends.headless import HeadlessRecoverableProtocolError
from sts2_rl.encoding.snapshot import collate_encoded_snapshots
from sts2_rl.macro import MacroCollectionAuthority, load_trunk_state
from sts2_rl.training import build_training_resources, load_training_config

# Sim-side faults the legacy runtime survives with a backend restart; the
# producer recovers the same way instead of failing its whole pipeline
# segment (protocol/contract violations on OUR side still fail fast).
_RECOVERABLE_BACKEND_ERRORS = (HeadlessRecoverableProtocolError, HeadlessSimError)
_MAX_BACKEND_RESTARTS = 20

SPOOL_ENVELOPE_FORMAT = "sts2-stage2-spool-envelope-v1"


def _atomic_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(dict(payload), sort_keys=True), encoding="utf-8")
    temporary.replace(path)


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


def _load_published_model_state(model: Any, path: Path, device: torch.device) -> None:
    """Load a trainer publication as the complete model state it claims to be."""

    state = torch.load(path, map_location=device, weights_only=True)
    model.load_state_dict(state, strict=True)


def _episode_summary(
    episode: Any,
    *,
    producer_id: str,
    run_nonce: str,
    episode_index: int,
    collected_at: float,
) -> dict[str, Any]:
    metrics = episode.metrics
    return {
        # Environment episode ids are only process-local.  Replay and
        # dashboard identity must remain unique across parallel collectors.
        "episode_id": f"{producer_id}:{run_nonce}:{episode_index}",
        "source_episode_id": metrics.episode_id,
        "producer_id": producer_id,
        "run_nonce": run_nonce,
        "episode_index": episode_index,
        "unix_s": collected_at,
        "reset_seed": metrics.reset_seed,
        "steps": metrics.steps,
        "run_won": metrics.run_won,
        "max_floor": metrics.max_floor,
        "max_act": metrics.max_act,
        "act1_cleared": metrics.act1_cleared,
        "revivals_used": metrics.revivals_used,
        "player_hp_lost": metrics.player_hp_lost,
        "reward_total": metrics.reward_total,
        "terminal_reason": metrics.terminal_reason,
    }


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--champion", required=True)
    parser.add_argument("--spool", required=True)
    parser.add_argument("--model-path", required=True)
    parser.add_argument(
        "--init-macro",
        default=None,
        help="explicit model-only fallback before the trainer publication exists",
    )
    parser.add_argument("--pipeline-id", required=True)
    parser.add_argument("--lineage-id", default="stage2-isolated-macro-v1")
    parser.add_argument("--producer-id", required=True)
    parser.add_argument("--status-out", required=True)
    parser.add_argument("--episodes", type=int, default=400)
    parser.add_argument("--epsilon", type=float, default=0.15)
    parser.add_argument(
        "--control-domain",
        choices=("macro",),
        default="macro",
        help="Stage 2 owns macro decisions; combat remains with the frozen champion",
    )
    parser.add_argument("--model-reload-episodes", type=int, default=5)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--metrics-out", required=True)
    parser.add_argument("--device", default=None)
    parser.add_argument("--sim-exe", default=None)
    return parser.parse_args()


def main() -> int:
    args = _parse_args()
    model_path = Path(args.model_path)
    init_path = Path(args.init_macro) if args.init_macro is not None else None
    status_path = Path(args.status_out)
    metrics_path = Path(args.metrics_out)
    metrics_path.parent.mkdir(parents=True, exist_ok=True)
    run_nonce = uuid.uuid4().hex[:12]
    base_status = {
        "pipeline_id": args.pipeline_id,
        "producer_id": args.producer_id,
        "run_nonce": run_nonce,
    }
    _atomic_json(status_path, {**base_status, "state": "running", "unix_s": time.time()})
    resources = None
    try:
        if init_path is not None and not init_path.is_file():
            raise FileNotFoundError(f"collector initialization does not exist: {init_path}")
        if not model_path.is_file() and init_path is None:
            raise FileNotFoundError(
                "collector has neither a trainer publication nor explicit " f"initialization: {model_path}"
            )
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
            raise RuntimeError("collector requires transaction_learning.enabled=true")

        spool = Path(args.spool)
        spool.mkdir(parents=True, exist_ok=True)

        def build_resources() -> Any:
            built = build_training_resources(config)
            champion_state = torch.load(
                Path(args.champion) / "network.pt",
                map_location=built.device,
                weights_only=True,
            )
            for target in (built.model, built.collector_model):
                load_trunk_state(target, dict(champion_state))
            for parameter in built.model.parameters():
                parameter.requires_grad_(False)
            return built

        resources = build_resources()

        macro_model = copy.deepcopy(resources.model)
        if init_path is not None:
            load_trunk_state(
                macro_model,
                dict(torch.load(init_path, map_location=resources.device, weights_only=True)),
            )
        macro_model.eval()
        model_mtime = 0.0

        def maybe_reload() -> None:
            nonlocal model_mtime
            if not model_path.is_file():
                return
            mtime = model_path.stat().st_mtime_ns
            if mtime <= model_mtime:
                return
            _load_published_model_state(macro_model, model_path, resources.device)
            macro_model.eval()
            model_mtime = mtime

        maybe_reload()
        authority = MacroCollectionAuthority(
            forward_q=_forward_factory(macro_model, resources.encoder, resources.device),
            initial_state=lambda: None,
            epsilon=args.epsilon,
            seed=int(args.seed),
            control_domain=args.control_domain,
        )
        resources.collector.macro_authority = authority

        produced = 0
        with metrics_path.open("a", encoding="utf-8") as metrics_file:
            metrics_file.write(
                json.dumps(
                    {
                        "event": "collector_start",
                        "unix_s": time.time(),
                        **base_status,
                        "seed": args.seed,
                    }
                )
                + "\n"
            )
            metrics_file.flush()
            backend_restarts = 0
            for episode_index in range(args.episodes):
                if episode_index % max(args.model_reload_episodes, 1) == 0:
                    maybe_reload()
                try:
                    episode = resources.collector.collect_episode(
                        epsilon=0.0,
                        deterministic=False,
                        record=False,
                    )
                except _RECOVERABLE_BACKEND_ERRORS as error:
                    backend_restarts += 1
                    if backend_restarts > _MAX_BACKEND_RESTARTS:
                        raise
                    metrics_file.write(
                        json.dumps(
                            {
                                "event": "collector_backend_restart",
                                **base_status,
                                "unix_s": time.time(),
                                "attempt": backend_restarts,
                                "episode_index": episode_index,
                                "error": f"{type(error).__name__}: {error}"[:400],
                            }
                        )
                        + "\n"
                    )
                    metrics_file.flush()
                    # The interrupted episode's macro view is unusable; the
                    # authority state resets with a discarded finish.
                    authority.finish_episode()
                    try:
                        resources.close()
                    except Exception:
                        pass
                    resources = build_resources()
                    resources.collector.macro_authority = authority
                    continue
                macro_episode = authority.finish_episode(
                    f"spool-{args.pipeline_id}-{args.producer_id}-{run_nonce}-{episode_index:05d}"
                )
                authority_metrics = authority.metrics()
                collected_at = time.time()
                summary = _episode_summary(
                    episode,
                    producer_id=args.producer_id,
                    run_nonce=run_nonce,
                    episode_index=episode_index,
                    collected_at=collected_at,
                )
                if macro_episode is not None:
                    envelope = {
                        "format": SPOOL_ENVELOPE_FORMAT,
                        "pipeline_id": args.pipeline_id,
                        "lineage_id": args.lineage_id,
                        "producer_id": args.producer_id,
                        "run_nonce": run_nonce,
                        "control_domain": args.control_domain,
                        "environment_steps": episode.metrics.steps,
                        "episode": macro_episode,
                        "episode_summary": summary,
                    }
                    temporary = spool / f".tmp-{uuid.uuid4().hex}"
                    with temporary.open("wb") as handle:
                        pickle.dump(envelope, handle, protocol=pickle.HIGHEST_PROTOCOL)
                    os.replace(temporary, spool / f"{uuid.uuid4().hex}.pkl")
                    produced += 1
                metrics_file.write(
                    json.dumps(
                        {
                            "event": "collector_episode",
                            "pipeline_id": args.pipeline_id,
                            "producer_id": args.producer_id,
                            "run_nonce": run_nonce,
                            "reset_seed": episode.metrics.reset_seed,
                            "episode_index": episode_index,
                            "unix_s": collected_at,
                            "steps": episode.metrics.steps,
                            "run_won": episode.metrics.run_won,
                            "max_floor": episode.metrics.max_floor,
                            "max_act": episode.metrics.max_act,
                            "act1_cleared": episode.metrics.act1_cleared,
                            "revivals_used": episode.metrics.revivals_used,
                            "reward_total": episode.metrics.reward_total,
                            "terminal_reason": episode.metrics.terminal_reason,
                            "macro_steps": len(macro_episode.steps) if macro_episode else 0,
                            "semantic_executor_failed": bool(
                                authority_metrics["episode_replay_invalid"]
                            ),
                            "semantic_executor_failures": int(
                                authority_metrics["executor_failures"]
                            ),
                            "semantic_executor_last_failure": (
                                authority.last_executor_failure
                            ),
                        }
                    )
                    + "\n"
                )
                metrics_file.flush()
        _atomic_json(
            status_path,
            {
                **base_status,
                "state": "complete",
                "unix_s": time.time(),
                "episodes": args.episodes,
                "produced": produced,
            },
        )
        return 0
    except BaseException as exc:
        failed = {
            **base_status,
            "state": "failed",
            "unix_s": time.time(),
            "error": f"{type(exc).__name__}: {exc}",
        }
        _atomic_json(status_path, failed)
        with metrics_path.open("a", encoding="utf-8") as metrics_file:
            metrics_file.write(json.dumps({"event": "collector_failed", **failed}) + "\n")
        raise
    finally:
        if resources is not None:
            resources.close()


if __name__ == "__main__":
    raise SystemExit(main())
