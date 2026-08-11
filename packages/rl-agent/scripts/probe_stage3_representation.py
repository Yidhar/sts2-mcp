#!/usr/bin/env python3
"""Frozen-trunk linear probe: is HP decodable from the Q head's input?

Weak learner as a MEASUREMENT instrument, never a controller. For each
model under test, real rest-decision observations are re-encoded across a
controlled HP grid; a forward hook captures the exact per-candidate
features the transaction Q head consumes. Two questions:

1. Feature sensitivity — does HP move the representation at all?
   (mean L2 of feat(hp90)-feat(hp10) per observation, against the
   repeat-forward noise floor)
2. Linear decodability — ridge probe predicting the HP fraction from
   observation-centered features, leave-one-observation-out R².

Interpretation: features insensitive -> the trunk never encodes HP at
these decisions (dense credit / auxiliary value head territory);
features sensitive + Q insensitive -> the head's nullspace swallows the
HP direction (dueling / head reparameterization territory).
"""

from __future__ import annotations

import argparse
import copy
import json
from dataclasses import replace
from pathlib import Path
from typing import Any

import numpy as np
import torch

from sts2_rl.semantics.forward import forward_decision
from sts2_rl.training import build_training_resources, load_training_config

HP_GRID = (0.1, 0.3, 0.5, 0.7, 0.9)


def _set_hp(observation: dict[str, Any], fraction: float) -> dict[str, Any]:
    mutated = copy.deepcopy(observation)
    player = mutated.get("player")
    if isinstance(player, dict):
        max_hp = player.get("max_hp") or 80
        player["hp"] = max(1, int(round(float(max_hp) * fraction)))
    return mutated


def _collect_rest_observations(
    journal_paths: list[str], limit: int
) -> list[tuple[dict[str, Any], list[dict[str, Any]]]]:
    observations: list[tuple[dict[str, Any], list[dict[str, Any]]]] = []
    seen: set[str] = set()
    for journal_path in journal_paths:
        with Path(journal_path).open(encoding="utf-8") as handle:
            for line in handle:
                if len(observations) >= limit:
                    return observations
                try:
                    record = json.loads(line)
                except json.JSONDecodeError:
                    continue
                observation = record.get("observation")
                actions = record.get("legal_actions") or record.get("actions")
                if not isinstance(observation, dict) or not isinstance(actions, list):
                    continue
                if len(actions) < 2:
                    continue
                decision = forward_decision(observation, actions)
                if decision is None or decision.surface != "rest":
                    continue
                player = observation.get("player") or {}
                deck = player.get("deck") or []
                key = json.dumps(
                    [
                        (observation.get("run") or {}).get("floor"),
                        sorted(
                            card.get("instance_id") or card.get("id")
                            for card in deck
                            if isinstance(card, dict)
                        ),
                    ],
                    sort_keys=True,
                )
                if key in seen:
                    continue
                seen.add(key)
                observations.append((observation, actions))
    return observations


def _probe_model(
    model: Any,
    encoder: Any,
    device: torch.device,
    observations: list[tuple[dict[str, Any], list[dict[str, Any]]]],
) -> dict[str, Any]:
    captured: list[torch.Tensor] = []

    def hook(module: Any, args: tuple[Any, ...]) -> None:
        captured.append(args[0].detach())

    handle = model.transaction_q_head.register_forward_pre_hook(hook)
    try:
        per_observation: list[dict[float, np.ndarray[Any, Any]]] = []
        noise_norms: list[float] = []
        for observation, actions in observations:
            features: dict[float, np.ndarray[Any, Any]] = {}
            for fraction in HP_GRID:
                mutated = _set_hp(observation, fraction)
                encoded = encoder.encode(mutated, actions, device=device)
                hidden = model.initial_state(1, device=device)
                captured.clear()
                with torch.no_grad():
                    model(encoded.batch, hidden, validate=False)
                features[fraction] = (
                    captured[-1][0].float().mean(dim=0).cpu().numpy()
                )
            # Repeat forward at 0.5 for the numerical noise floor.
            mutated = _set_hp(observation, 0.5)
            encoded = encoder.encode(mutated, actions, device=device)
            hidden = model.initial_state(1, device=device)
            captured.clear()
            with torch.no_grad():
                model(encoded.batch, hidden, validate=False)
            repeat = captured[-1][0].float().mean(dim=0).cpu().numpy()
            noise_norms.append(float(np.linalg.norm(repeat - features[0.5])))
            per_observation.append(features)

        sensitivity = [
            float(np.linalg.norm(features[0.9] - features[0.1]))
            for features in per_observation
        ]
        scale = [
            float(np.linalg.norm(features[0.5])) for features in per_observation
        ]

        # Ridge probe on observation-centered features -> hp fraction.
        rows: list[np.ndarray[Any, Any]] = []
        targets: list[float] = []
        groups: list[int] = []
        for index, features in enumerate(per_observation):
            stack = np.stack([features[fraction] for fraction in HP_GRID])
            centered = stack - stack.mean(axis=0, keepdims=True)
            for row, fraction in zip(centered, HP_GRID, strict=True):
                rows.append(row)
                targets.append(fraction - float(np.mean(HP_GRID)))
                groups.append(index)
        matrix = np.stack(rows)
        target = np.asarray(targets)
        group = np.asarray(groups)

        def ridge_r2(
            train_mask: np.ndarray[Any, Any], test_mask: np.ndarray[Any, Any]
        ) -> float:
            x_train, y_train = matrix[train_mask], target[train_mask]
            alpha = 1.0
            gram = x_train.T @ x_train + alpha * np.eye(matrix.shape[1])
            weights = np.linalg.solve(gram, x_train.T @ y_train)
            prediction = matrix[test_mask] @ weights
            residual = float(np.sum((target[test_mask] - prediction) ** 2))
            total = float(np.sum(target[test_mask] ** 2))
            return 1.0 - residual / total if total > 0 else 0.0

        loo_scores = [
            ridge_r2(group != index, group == index)
            for index in range(len(per_observation))
        ]
        return {
            "observations": len(per_observation),
            "feature_dim": int(matrix.shape[1]),
            "noise_norm_mean": float(np.mean(noise_norms)),
            "hp_sensitivity_norm_mean": float(np.mean(sensitivity)),
            "feature_scale_mean": float(np.mean(scale)),
            "sensitivity_over_noise": (
                float(np.mean(sensitivity) / max(float(np.mean(noise_norms)), 1e-12))
            ),
            "ridge_loo_r2_mean": float(np.mean(loo_scores)),
            "ridge_loo_r2_per_observation": [round(s, 4) for s in loo_scores],
        }
    finally:
        handle.remove()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--champion", required=True)
    parser.add_argument(
        "--model",
        action="append",
        required=True,
        metavar="LABEL=PATH",
        help="model to probe; champion trunk itself probes as label 'champion'",
    )
    parser.add_argument("--journal", required=True, nargs="+")
    parser.add_argument("--limit", type=int, default=24)
    parser.add_argument("--device", default=None)
    parser.add_argument("--report-out", default="stage3-representation-probe.json")
    args = parser.parse_args()

    config = load_training_config(profile="preheat", config_path=Path(args.config))
    if args.device is not None:
        config = replace(config, runtime=replace(config.runtime, device=str(args.device)))
    resources = build_training_resources(config)
    try:
        champion_state = torch.load(
            Path(args.champion) / "network.pt",
            map_location=resources.device,
            weights_only=True,
        )
        result = resources.model.load_state_dict(champion_state, strict=False)
        unexpected = [key for key in result.unexpected_keys if "liveness" not in key]
        if result.missing_keys or unexpected:
            raise SystemExit("champion checkpoint drift")
        resources.model.eval()

        observations = _collect_rest_observations(args.journal, args.limit)
        if not observations:
            raise SystemExit("no rest-decision observations found in the journals")

        report: dict[str, Any] = {
            "event": "stage3_representation_probe",
            "hp_grid": list(HP_GRID),
            "models": {},
        }
        for spec in args.model:
            label, _, path = spec.partition("=")
            if label == "champion" and not path:
                model = resources.model
            else:
                model = copy.deepcopy(resources.model)
                model.load_state_dict(
                    torch.load(
                        Path(path), map_location=resources.device, weights_only=True
                    )
                )
                model.eval()
            report["models"][label] = _probe_model(
                model, resources.encoder, resources.device, observations
            )
        Path(args.report_out).write_text(
            json.dumps(report, indent=2), encoding="utf-8"
        )
        print(
            json.dumps(
                {
                    label: {
                        key: value
                        for key, value in stats.items()
                        if key != "ridge_loo_r2_per_observation"
                    }
                    for label, stats in report["models"].items()
                },
                indent=2,
            )
        )
        return 0
    finally:
        resources.close()


if __name__ == "__main__":
    raise SystemExit(main())
