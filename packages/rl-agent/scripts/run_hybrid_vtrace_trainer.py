#!/usr/bin/env python3
"""Consume fresh hybrid semantic unrolls with recurrent V-trace."""

from __future__ import annotations

import argparse
import json
import os
import pickle
import random
import time
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import numpy as np
import torch

from sts2_baseline import act_frontloaded_death_reward_identity
from sts2_rl.encoding import GroundedObservationEncoder
from sts2_rl.encoding.snapshot import collate_encoded_snapshots
from sts2_rl.joint import (
    HybridVTraceLearner,
    HybridVTraceLearnerConfig,
    HybridVTraceUnroll,
)
from sts2_rl.models import (
    HybridActorCriticConfig,
    HybridDomainActorCriticModel,
    initialize_actor_critic_from_iql,
)
from sts2_rl.training import load_training_config
from sts2_rl.training.sdpa import configure_rocm_sdpa_backend

HYBRID_VTRACE_TRAINING_STATE = "sts2-hybrid-vtrace-training-state-v1"
HYBRID_VTRACE_SPOOL_ENVELOPE = "sts2-hybrid-vtrace-spool-envelope-v1"
HYBRID_VTRACE_PUBLICATION = "sts2-hybrid-vtrace-publication-v1"


def _atomic_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(dict(payload), sort_keys=True), encoding="utf-8")
    os.replace(temporary, path)


def _atomic_torch(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save(dict(payload), temporary)
    os.replace(temporary, path)


def _rng_state() -> dict[str, Any]:
    return {
        "python": random.getstate(),
        "numpy": np.random.get_state(),
        "torch": torch.get_rng_state(),
        "accelerator": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
    }


def _restore_rng(state: Mapping[str, Any]) -> None:
    random.setstate(state["python"])
    np.random.set_state(state["numpy"])
    torch.set_rng_state(state["torch"].cpu())
    if state.get("accelerator") is not None and torch.cuda.is_available():
        torch.cuda.set_rng_state_all([item.cpu() for item in state["accelerator"]])


def _source_file(value: str) -> Path:
    path = Path(value)
    if path.is_file():
        return path
    publication = path / "network.pt"
    if publication.is_file():
        return publication
    nested = path / "publication" / "network.pt"
    if nested.is_file():
        return nested
    raise FileNotFoundError(f"hybrid model-init source does not exist: {path}")


def _source_domain_decisions(source: Mapping[str, Any]) -> dict[str, int]:
    raw = source.get("domain_decisions", {})
    if not isinstance(raw, Mapping):
        return {"macro": 0, "combat": 0}
    return {
        domain: max(0, int(raw.get(domain, 0))) for domain in ("macro", "combat")
    }


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--spool", required=True)
    parser.add_argument("--publication", required=True)
    parser.add_argument("--checkpoint-out", required=True)
    parser.add_argument("--metrics-out", required=True)
    parser.add_argument("--status-out", required=True)
    parser.add_argument("--lineage-id", required=True)
    parser.add_argument("--pipeline-id", required=True)
    parser.add_argument("--producer-id", action="append", dest="producer_ids", required=True)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--initialize-from")
    mode.add_argument("--resume")
    parser.add_argument("--stop-after-runs", type=int, default=400)
    parser.add_argument("--batch-unrolls", type=int, default=4)
    parser.add_argument("--ingest-batch-items", type=int, default=16)
    parser.add_argument("--maximum-pending-items", type=int, default=32)
    parser.add_argument("--save-interval-runs", type=int, default=10)
    parser.add_argument("--publication-interval-updates", type=int, default=1)
    parser.add_argument("--device", default="auto")
    return parser.parse_args()


def main() -> int:
    args = _parse_args()
    status = Path(args.status_out)
    status_base = {
        "pipeline_id": args.pipeline_id,
        "producer_ids": list(args.producer_ids),
    }
    _atomic_json(status, {**status_base, "state": "running", "unix_s": time.time()})
    try:
        config = load_training_config(profile="preheat", config_path=Path(args.config))
        revival_budget = config.curriculum.revival_budget
        if isinstance(revival_budget, bool) or not isinstance(revival_budget, int) or revival_budget <= 0:
            raise ValueError("hybrid reward-v9 requires a positive revival budget")
        device_name = args.device
        if device_name == "auto":
            device_name = "cuda" if torch.cuda.is_available() else "cpu"
        device = torch.device(device_name)
        configure_rocm_sdpa_backend(config.runtime.rocm_sdpa_backend, devices=(device,))
        encoder = GroundedObservationEncoder(config.model.to_encoding_config())
        model_config = HybridActorCriticConfig(config.model.to_model_config())
        model = HybridDomainActorCriticModel(model_config).to(device)

        def forward(snapshot: Any, hidden: torch.Tensor | None) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
            encoded = collate_encoded_snapshots(
                (snapshot,),
                expected_config=encoder.config,
                expected_fingerprint=snapshot.encoding_fingerprint,
                device=device,
            )
            if hidden is None:
                hidden = model.initial_state(1, device=device)
            output = model(encoded, hidden, validate=False)
            return output.policy_logits[0], output.value[0], output.recurrent_state

        learner = HybridVTraceLearner(
            model=model,
            forward=forward,
            config=HybridVTraceLearnerConfig(
                learning_rate=config.optimization.learning_rate,
                policy_weight=config.optimization.policy_weight,
                value_weight=config.optimization.value_weight,
                entropy_weight=config.optimization.entropy_weight,
                entropy_weight_end=config.optimization.entropy_weight_end,
                entropy_decay_updates=config.optimization.entropy_decay_updates,
                vtrace_rho_clip=config.optimization.vtrace_rho_clip,
                vtrace_c_clip=config.optimization.vtrace_c_clip,
                policy_rho_clip=config.optimization.policy_rho_clip,
                gradient_clip_norm=config.optimization.gradient_clip_norm,
            ),
        )
        pending: list[HybridVTraceUnroll] = []
        completed_runs = 0
        environment_steps = 0
        domain_decisions = {"macro": 0, "combat": 0}
        initialization: dict[str, Any]
        if args.resume:
            payload = torch.load(args.resume, map_location=device, weights_only=False)
            if not isinstance(payload, Mapping) or payload.get("format") != HYBRID_VTRACE_TRAINING_STATE:
                raise ValueError("resume source is not a hybrid V-trace checkpoint")
            if payload.get("model_config") != model_config.as_mapping():
                raise ValueError("resume model config differs")
            if payload.get("reward_identity") != act_frontloaded_death_reward_identity(
                revival_budget=revival_budget
            ):
                raise ValueError("resume reward identity differs")
            model.load_state_dict(dict(payload["model"]), strict=True)
            learner.load_state_dict(dict(payload["learner"]))
            pending = list(payload.get("pending_unrolls", ()))
            completed_runs = int(payload.get("completed_runs", 0))
            environment_steps = int(payload.get("environment_steps", 0))
            domain_decisions = _source_domain_decisions(payload)
            _restore_rng(payload["rng"])
            initialization = {"mode": "resume", "source": str(Path(args.resume).resolve())}
        else:
            source_path = _source_file(str(args.initialize_from))
            source = torch.load(source_path, map_location=device, weights_only=True)
            if not isinstance(source, Mapping) or not isinstance(source.get("online"), Mapping):
                raise ValueError("model-init source has no hybrid online model")
            inherited = initialize_actor_critic_from_iql(model, dict(source["online"]))
            domain_decisions = _source_domain_decisions(source)
            initialization = {
                "mode": "model_init",
                "source": str(source_path.resolve()),
                "inherited_tensors": len(inherited["inherited"]),
                "fresh_tensors": len(inherited["fresh"]),
            }

        publication_path = Path(args.publication)
        checkpoint_path = Path(args.checkpoint_out)
        metrics_path = Path(args.metrics_out)
        spool = Path(args.spool)
        spool.mkdir(parents=True, exist_ok=True)
        metrics_path.parent.mkdir(parents=True, exist_ok=True)
        reward_identity = act_frontloaded_death_reward_identity(revival_budget=revival_budget)

        def publish() -> None:
            _atomic_torch(
                publication_path,
                {
                    "format": HYBRID_VTRACE_PUBLICATION,
                    "lineage_id": args.lineage_id,
                    "pipeline_id": args.pipeline_id,
                    "model_config": model_config.as_mapping(),
                    "reward_identity": reward_identity,
                    "online": {key: value.detach().cpu() for key, value in model.state_dict().items()},
                    "policy_version": learner.policy_version,
                    "domain_decisions": dict(domain_decisions),
                    "completed_runs": completed_runs,
                    "environment_steps": environment_steps,
                },
            )

        def save() -> None:
            _atomic_torch(
                checkpoint_path,
                {
                    "format": HYBRID_VTRACE_TRAINING_STATE,
                    "lineage_id": args.lineage_id,
                    "pipeline_id": args.pipeline_id,
                    "model_config": model_config.as_mapping(),
                    "reward_identity": reward_identity,
                    "model": model.state_dict(),
                    "learner": learner.state_dict(),
                    "pending_unrolls": tuple(pending),
                    "completed_runs": completed_runs,
                    "environment_steps": environment_steps,
                    "domain_decisions": dict(domain_decisions),
                    "rng": _rng_state(),
                    "initialization": initialization,
                },
            )

        def next_batch() -> tuple[HybridVTraceUnroll, ...]:
            if len(pending) < args.batch_unrolls:
                return ()
            macro = 0
            combat = 0
            for index, unroll in enumerate(pending):
                macro += unroll.macro_steps
                combat += unroll.combat_steps
                if index + 1 >= args.batch_unrolls and macro > 0 and combat > 0:
                    batch = tuple(pending[: index + 1])
                    del pending[: index + 1]
                    return batch
            return ()

        publish()
        save()
        last_save = completed_runs
        last_publication_update = learner.policy_version
        with metrics_path.open("a", encoding="utf-8") as metrics:
            metrics.write(
                json.dumps(
                    {
                        "event": "hybrid_vtrace_start",
                        "unix_s": time.time(),
                        "completed_runs": completed_runs,
                        "environment_steps": environment_steps,
                        "learner": learner.as_mapping(),
                        "initialization": initialization,
                    }
                )
                + "\n"
            )
            metrics.flush()
            while completed_runs < args.stop_after_runs:
                paths = sorted(spool.glob("*.pkl"))[: args.ingest_batch_items]
                if not paths:
                    time.sleep(0.05)
                    continue
                for path in paths:
                    with path.open("rb") as handle:
                        envelope = pickle.load(handle)
                    if not isinstance(envelope, Mapping) or envelope.get("format") != HYBRID_VTRACE_SPOOL_ENVELOPE:
                        raise ValueError(f"invalid hybrid V-trace spool item: {path}")
                    if envelope.get("pipeline_id") != args.pipeline_id:
                        raise ValueError(f"hybrid V-trace spool pipeline differs: {path}")
                    if envelope.get("producer_id") not in args.producer_ids:
                        raise ValueError(f"hybrid V-trace spool producer differs: {path}")
                    kind = envelope.get("kind")
                    if kind == "unroll":
                        unroll = envelope.get("unroll")
                        if not isinstance(unroll, HybridVTraceUnroll):
                            raise ValueError(f"hybrid V-trace spool item has no unroll: {path}")
                        pending.append(unroll)
                        domain_decisions["macro"] += unroll.macro_steps
                        domain_decisions["combat"] += unroll.combat_steps
                    elif kind == "episode":
                        episode = envelope.get("episode")
                        if not isinstance(episode, Mapping):
                            raise ValueError(f"hybrid V-trace spool item has no episode: {path}")
                        completed_runs += 1
                        environment_steps += int(episode.get("steps", 0))
                        metrics.write(
                            json.dumps(
                                {
                                    "event": "hybrid_vtrace_episode",
                                    "unix_s": time.time(),
                                    "completed_runs": completed_runs,
                                    "environment_steps": environment_steps,
                                    "episode": dict(episode),
                                }
                            )
                            + "\n"
                        )
                    else:
                        raise ValueError(f"unknown hybrid V-trace spool kind: {kind!r}")
                    path.unlink()

                    while True:
                        batch = next_batch()
                        if not batch:
                            break
                        started = time.monotonic()
                        update = learner.update(batch)
                        metrics.write(
                            json.dumps(
                                {
                                    "event": "hybrid_vtrace_update",
                                    "unix_s": time.time(),
                                    "completed_runs": completed_runs,
                                    "environment_steps": environment_steps,
                                    "unrolls": len(batch),
                                    "pending_unrolls": len(pending),
                                    "update_seconds": time.monotonic() - started,
                                    "learner": update,
                                }
                            )
                            + "\n"
                        )
                        if (
                            learner.policy_version - last_publication_update
                            >= args.publication_interval_updates
                        ):
                            publish()
                            last_publication_update = learner.policy_version
                metrics.flush()
                if completed_runs - last_save >= args.save_interval_runs:
                    save()
                    publish()
                    last_save = completed_runs
                _atomic_json(
                    status,
                    {
                        **status_base,
                        "state": "running",
                        "completed_runs": completed_runs,
                        "environment_steps": environment_steps,
                        "learner_updates": learner.policy_version,
                        "pending_unrolls": len(pending),
                        "spool_pending": len(tuple(spool.glob("*.pkl"))),
                        "spool_capacity": args.maximum_pending_items,
                        "unix_s": time.time(),
                    },
                )
        save()
        publish()
        _atomic_json(
            status,
            {
                **status_base,
                "state": "complete",
                "completed_runs": completed_runs,
                "environment_steps": environment_steps,
                "learner_updates": learner.policy_version,
                "pending_unrolls": len(pending),
                "unix_s": time.time(),
            },
        )
        return 0
    except BaseException as error:
        _atomic_json(
            status,
            {
                **status_base,
                "state": "failed",
                "error": f"{type(error).__name__}: {error}",
                "unix_s": time.time(),
            },
        )
        raise


if __name__ == "__main__":
    raise SystemExit(main())
