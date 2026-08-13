#!/usr/bin/env python3
"""Unified hybrid candidate-Q trainer over spooled complete factual runs."""

from __future__ import annotations

import argparse
import copy
import json
import os
import pickle
import random
import time
import uuid
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import numpy as np
import torch

from sts2_rl.encoding.snapshot import collate_encoded_snapshots
from sts2_rl.joint import HybridQConfig, HybridQLearner, HybridRun, HybridRunReplay
from sts2_rl.models import (
    HybridCandidateQConfig,
    HybridDomainCandidateQModel,
    initialize_hybrid_from_grounded,
)
from sts2_rl.training import load_training_config
from sts2_rl.training.sdpa import configure_rocm_sdpa_backend

HYBRID_TRAINING_STATE = "sts2-hybrid-training-state-v1"
HYBRID_SPOOL_ENVELOPE = "sts2-hybrid-spool-envelope-v1"
HYBRID_PUBLICATION = "sts2-hybrid-publication-v1"


def _atomic_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(dict(payload), sort_keys=True), encoding="utf-8")
    temporary.replace(path)


def _forward(model: HybridDomainCandidateQModel, encoder: Any, device: torch.device) -> Any:
    def call(step: Any, hidden: torch.Tensor | None) -> tuple[torch.Tensor, torch.Tensor]:
        encoded = collate_encoded_snapshots(
            (step.snapshot,),
            expected_config=encoder.config,
            expected_fingerprint=step.snapshot.encoding_fingerprint,
            device=device,
        )
        if hidden is None:
            hidden = model.initial_state(1, device=device)
        output = model(encoded, hidden, validate=False)
        return output.q_values[0], output.recurrent_state

    return call


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


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--champion", required=True)
    parser.add_argument("--spool", required=True)
    parser.add_argument("--publication", required=True)
    parser.add_argument("--checkpoint-out", required=True)
    parser.add_argument("--metrics-out", required=True)
    parser.add_argument("--status-out", required=True)
    parser.add_argument("--lineage-id", required=True)
    parser.add_argument("--pipeline-id", required=True)
    parser.add_argument("--producer-id", action="append", dest="producer_ids", required=True)
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--initialize-from", default=None)
    mode.add_argument("--resume", default=None)
    parser.add_argument("--stop-after-runs", type=int, default=400)
    parser.add_argument("--updates-per-run", type=int, default=2)
    parser.add_argument("--ingest-batch-runs", type=int, default=1)
    parser.add_argument("--run-batch-size", type=int, default=4)
    parser.add_argument("--replay-runs", type=int, default=128)
    parser.add_argument("--macro-unroll", type=int, default=8)
    parser.add_argument("--combat-unroll", type=int, default=32)
    parser.add_argument("--maximum-view-reuse", type=int, default=8)
    parser.add_argument("--save-interval-runs", type=int, default=10)
    parser.add_argument("--publication-interval-runs", type=int, default=1)
    parser.add_argument("--maximum-pending-runs", type=int, default=2)
    parser.add_argument(
        "--target-sync-every-runs",
        type=int,
        default=0,
        help="optional factual-run target clock; useful when updates/run changes",
    )
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
        device_name = args.device
        if device_name == "auto":
            device_name = "cuda" if torch.cuda.is_available() else "cpu"
        device = torch.device(device_name)
        sdpa_backend = configure_rocm_sdpa_backend(
            config.runtime.rocm_sdpa_backend,
            devices=(device,),
        )
        print(
            json.dumps({"event": "sdpa_backend", **sdpa_backend.to_mapping()}),
            flush=True,
        )
        encoder_config = config.model.to_encoding_config()
        from sts2_rl.encoding import GroundedObservationEncoder

        encoder = GroundedObservationEncoder(encoder_config)
        grounded_config = config.model.to_model_config()
        hybrid_config = HybridCandidateQConfig(grounded_config)
        online = HybridDomainCandidateQModel(hybrid_config).to(device)
        target = copy.deepcopy(online).to(device)
        replay = HybridRunReplay(
            capacity_runs=args.replay_runs,
            macro_unroll=args.macro_unroll,
            combat_unroll=args.combat_unroll,
            maximum_view_reuse=args.maximum_view_reuse,
            seed=config.runtime.seed,
        )
        learner = HybridQLearner(
            online_model=online,
            target_model=target,
            replay=replay,
            forward_online=_forward(online, encoder, device),
            forward_target=_forward(target, encoder, device),
            initial_state=lambda: online.initial_state(1, device=device),
            config=HybridQConfig(
                learning_rate=config.optimization.learning_rate,
                run_batch_size=args.run_batch_size,
            ),
        )
        ingested = 0
        environment_steps = 0
        domain_decisions = {"macro": 0, "combat": 0}
        source = (
            Path(args.initialize_from)
            if args.initialize_from
            else Path(args.champion) / "network.pt"
        )
        if args.resume:
            payload = torch.load(args.resume, map_location=device, weights_only=False)
            if payload.get("format") != HYBRID_TRAINING_STATE:
                raise ValueError("resume is not a hybrid training state")
            if payload.get("lineage_id") != args.lineage_id:
                raise ValueError("resume lineage differs")
            if payload.get("model_config") != hybrid_config.as_mapping():
                raise ValueError("resume hybrid model config differs")
            online.load_state_dict(payload["online"], strict=True)
            target.load_state_dict(payload["target"], strict=True)
            replay.load_state_dict(payload["replay"])
            learner.load_state_dict(payload["learner"])
            ingested = int(payload["ingested"])
            environment_steps = int(payload["environment_steps"])
            domain_decisions = {
                domain: int(payload.get("domain_decisions", {}).get(domain, 0))
                for domain in ("macro", "combat")
            }
            _restore_rng(payload["rng"])
        else:
            source_file = source / "network.pt" if source.is_dir() else source
            state = torch.load(source_file, map_location=device, weights_only=True)
            source_state = (
                state.get("online")
                if isinstance(state, Mapping) and state.get("format") == HYBRID_PUBLICATION
                else state
            )
            if not isinstance(source_state, Mapping):
                raise ValueError("hybrid initialization has no model state")
            initialize_hybrid_from_grounded(online, dict(source_state))
            target.load_state_dict(online.state_dict(), strict=True)

        def publish() -> None:
            publication = Path(args.publication)
            publication.parent.mkdir(parents=True, exist_ok=True)
            # A fixed ``network.pt.tmp`` is not safe once collectors reload in
            # parallel.  More importantly, Path.replace() can expose a brief
            # missing-path window on the Windows-mounted WSL filesystem.  A
            # unique completed file plus os.replace() keeps the old policy
            # readable until the new one is ready and atomically swaps the
            # directory entry.
            publication_temporary = publication.with_name(
                f".{publication.name}.{uuid.uuid4().hex}.tmp"
            )
            try:
                torch.save(
                    {
                        "format": HYBRID_PUBLICATION,
                        "model_config": hybrid_config.as_mapping(),
                        "online": online.state_dict(),
                        "policy_version": int(learner.metrics.updates),
                        "domain_decisions": dict(domain_decisions),
                    },
                    publication_temporary,
                )
                os.replace(publication_temporary, publication)
            finally:
                publication_temporary.unlink(missing_ok=True)

        def save() -> None:
            checkpoint = Path(args.checkpoint_out)
            checkpoint.parent.mkdir(parents=True, exist_ok=True)
            temporary = checkpoint.with_suffix(checkpoint.suffix + ".tmp")
            torch.save(
                {
                    "format": HYBRID_TRAINING_STATE,
                    "lineage_id": args.lineage_id,
                    "model_config": hybrid_config.as_mapping(),
                    "online": online.state_dict(),
                    "target": target.state_dict(),
                    "learner": learner.state_dict(),
                    "replay": replay.state_dict(),
                    "ingested": ingested,
                    "environment_steps": environment_steps,
                    "domain_decisions": dict(domain_decisions),
                    "rng": _rng_state(),
                },
                temporary,
            )
            temporary.replace(checkpoint)
            publish()

        save()
        spool = Path(args.spool)
        spool.mkdir(parents=True, exist_ok=True)
        metrics_path = Path(args.metrics_out)
        metrics_path.parent.mkdir(parents=True, exist_ok=True)
        last_save = ingested
        last_publication = ingested
        with metrics_path.open("a", encoding="utf-8") as metrics:
            while ingested < args.stop_after_runs:
                previous_ingested = ingested
                files = sorted(
                    spool.glob("*.pkl"),
                    key=lambda item: item.stat().st_mtime_ns,
                )
                if not files:
                    time.sleep(0.25)
                    continue
                accepted: list[tuple[Path, Mapping[str, Any], HybridRun]] = []
                remaining = args.stop_after_runs - ingested
                for path in files[: min(args.ingest_batch_runs, remaining)]:
                    with path.open("rb") as handle:
                        envelope = pickle.load(handle)
                    if envelope.get("format") != HYBRID_SPOOL_ENVELOPE:
                        raise ValueError(f"invalid hybrid spool item: {path}")
                    if envelope.get("pipeline_id") != args.pipeline_id:
                        raise ValueError(
                            f"hybrid spool item belongs to another pipeline: {path}"
                        )
                    if envelope.get("producer_id") not in args.producer_ids:
                        raise ValueError(
                            f"hybrid spool item has an unknown producer: {path}"
                        )
                    run = envelope.get("run")
                    if not isinstance(run, HybridRun):
                        raise ValueError(f"hybrid spool item has no run: {path}")
                    replay.put(run)
                    ingested += 1
                    environment_steps += int(run.environment_steps)
                    for domain in ("macro", "combat"):
                        domain_decisions[domain] += sum(
                            step.control_domain == domain for step in run.steps
                        )
                    accepted.append((path, envelope, run))
                update_metrics: Mapping[str, Any] = learner.metrics.as_mapping()
                update_started = time.monotonic()
                for _ in range(args.updates_per_run * len(accepted)):
                    update_metrics = learner.update()
                if (
                    args.target_sync_every_runs > 0
                    and previous_ingested // args.target_sync_every_runs
                    < ingested // args.target_sync_every_runs
                ):
                    learner.sync_target()
                    update_metrics = learner.metrics.as_mapping()
                update_seconds = time.monotonic() - update_started
                rows: list[dict[str, Any]] = []
                first_ingested = ingested - len(accepted) + 1
                first_environment_steps = environment_steps - sum(
                    int(run.environment_steps) for _, _, run in accepted
                )
                cumulative_environment_steps = first_environment_steps
                for offset, (path, envelope, run) in enumerate(accepted):
                    path.unlink()
                    cumulative_environment_steps += int(run.environment_steps)
                    episode_summary = envelope.get("episode")
                    if not isinstance(episode_summary, Mapping):
                        episode_summary = {}
                    rows.append(
                        {
                            "event": "hybrid_update",
                            "unix_s": time.time(),
                            "ingested_runs": first_ingested + offset,
                            "environment_steps": cumulative_environment_steps,
                            "producer_id": envelope.get("producer_id"),
                            "episode": dict(episode_summary),
                            "spool_pending": len(tuple(spool.glob("*.pkl"))),
                            "spool_capacity": args.maximum_pending_runs,
                            "learner_update_seconds": update_seconds,
                            "learner": dict(update_metrics),
                            "replay": replay.metrics(),
                        }
                    )
                for row in rows:
                    metrics.write(json.dumps(row) + "\n")
                metrics.flush()
                checkpoint_due = ingested - last_save >= args.save_interval_runs
                if checkpoint_due:
                    save()
                    last_save = ingested
                    last_publication = ingested
                elif ingested - last_publication >= args.publication_interval_runs:
                    publish()
                    last_publication = ingested
        save()
        _atomic_json(
            status,
            {
                **status_base,
                "state": "complete",
                "ingested_runs": ingested,
                "environment_steps": environment_steps,
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
