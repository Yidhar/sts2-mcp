"""Read-only, seed-paired statistical analysis for frozen evaluations.

The frozen evaluator already guarantees that every checkpoint in one paired
publication sees the same ordered reset seeds.  This module consumes that
publication without touching its manifest, audits, or journals and publishes a
separate machine-readable report plus a compact Markdown report.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import random
import shutil
import time
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from uuid import uuid4

from .artifacts import resolve_artifact_path, resolve_external_input_path, validate_artifact_component

_INPUT_SCHEMA = "sts2-paired-frozen-evaluation-v2"
_AUDIT_SCHEMA = "sts2-frozen-checkpoint-evaluation-v2"
_REPORT_SCHEMA = "sts2-paired-evaluation-analysis-v1"

_BINARY_METRICS: tuple[tuple[str, str], ...] = (
    ("run_win", "higher"),
    ("act1_clear", "higher"),
    ("act3_reach", "higher"),
    ("hard_cycle", "lower"),
    ("combat_stall", "lower"),
    ("noncombat_stall", "lower"),
    ("any_stall", "lower"),
    ("trusted_policy_failure", "lower"),
    ("deadlock", "lower"),
)


@dataclass(frozen=True, slots=True)
class EpisodeRecord:
    reset_seed: int
    run_won: bool
    act1_cleared: bool
    max_act: int
    max_floor: int
    combat_progress_stalled: bool
    noncombat_progress_stalled: bool
    trusted_policy_failure: bool
    noncombat_event_cycle: bool
    selection_action_cycle: bool
    deadlocked: bool
    revivals_used: int
    player_hp_lost: float

    def binary_metric(self, name: str) -> bool:
        if name == "run_win":
            return self.run_won
        if name == "act1_clear":
            return self.act1_cleared
        if name == "act3_reach":
            return self.max_act >= 3
        if name == "hard_cycle":
            return self.noncombat_event_cycle or self.selection_action_cycle
        if name == "combat_stall":
            return self.combat_progress_stalled
        if name == "noncombat_stall":
            return self.noncombat_progress_stalled
        if name == "any_stall":
            return self.combat_progress_stalled or self.noncombat_progress_stalled
        if name == "trusted_policy_failure":
            return self.trusted_policy_failure
        if name == "deadlock":
            return self.deadlocked
        raise KeyError(f"unknown paired binary metric: {name}")


@dataclass(frozen=True, slots=True)
class CheckpointEpisodes:
    checkpoint_id: str
    checkpoint: str
    source_environment_steps: int
    source_policy_version: int
    episodes: Mapping[int, EpisodeRecord]

    @property
    def display_name(self) -> str:
        checkpoint_name = Path(self.checkpoint).name
        return f"{checkpoint_name} / policy {self.source_policy_version}"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _mapping(value: object, *, label: str) -> dict[str, object]:
    if not isinstance(value, dict) or any(not isinstance(key, str) for key in value):
        raise ValueError(f"{label} must be an object with string keys")
    return value


def _array(value: object, *, label: str) -> list[object]:
    if not isinstance(value, list):
        raise ValueError(f"{label} must be an array")
    return value


def _string(value: object, *, label: str) -> str:
    if not isinstance(value, str) or not value:
        raise ValueError(f"{label} must be a non-empty string")
    return value


def _integer(value: object, *, label: str, minimum: int = 0) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
        raise ValueError(f"{label} must be an integer >= {minimum}")
    return value


def _boolean(value: object, *, label: str) -> bool:
    if not isinstance(value, bool):
        raise ValueError(f"{label} must be a boolean")
    return value


def _number(value: object, *, label: str, minimum: float = 0.0) -> float:
    if isinstance(value, bool) or not isinstance(value, int | float):
        raise ValueError(f"{label} must be a finite number")
    result = float(value)
    if not math.isfinite(result) or result < minimum:
        raise ValueError(f"{label} must be a finite number >= {minimum}")
    return result


def _report_number(value: object, *, label: str) -> float:
    if isinstance(value, bool) or not isinstance(value, int | float):
        raise ValueError(f"{label} must be a finite number")
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f"{label} must be a finite number")
    return result


def _read_json(path: Path, *, label: str) -> dict[str, object]:
    try:
        loaded: object = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise ValueError(f"cannot read {label} at {path}: {exc}") from exc
    return _mapping(loaded, label=label)


def _episode_from_mapping(value: object, *, label: str) -> EpisodeRecord:
    item = _mapping(value, label=label)
    return EpisodeRecord(
        reset_seed=_integer(item.get("reset_seed"), label=f"{label}.reset_seed"),
        run_won=_boolean(item.get("run_won"), label=f"{label}.run_won"),
        act1_cleared=_boolean(item.get("act1_cleared"), label=f"{label}.act1_cleared"),
        max_act=_integer(item.get("max_act"), label=f"{label}.max_act"),
        max_floor=_integer(item.get("max_floor"), label=f"{label}.max_floor"),
        combat_progress_stalled=_boolean(
            item.get("combat_progress_stalled"),
            label=f"{label}.combat_progress_stalled",
        ),
        noncombat_progress_stalled=_boolean(
            item.get("noncombat_progress_stalled"),
            label=f"{label}.noncombat_progress_stalled",
        ),
        trusted_policy_failure=_boolean(
            item.get("trusted_policy_failure"),
            label=f"{label}.trusted_policy_failure",
        ),
        noncombat_event_cycle=_boolean(
            item.get("noncombat_event_cycle"),
            label=f"{label}.noncombat_event_cycle",
        ),
        selection_action_cycle=_boolean(
            item.get("selection_action_cycle"),
            label=f"{label}.selection_action_cycle",
        ),
        deadlocked=_boolean(item.get("deadlocked"), label=f"{label}.deadlocked"),
        revivals_used=_integer(item.get("revivals_used"), label=f"{label}.revivals_used"),
        player_hp_lost=_number(item.get("player_hp_lost"), label=f"{label}.player_hp_lost"),
    )


def _load_checkpoints(
    paired_root: Path,
) -> tuple[dict[str, object], tuple[int, ...], list[CheckpointEpisodes]]:
    manifest_path = paired_root / "paired-evaluation.manifest.json"
    manifest = _read_json(manifest_path, label="paired evaluation manifest")
    if manifest.get("schema_version") != _INPUT_SCHEMA or manifest.get("status") != "complete":
        raise ValueError("paired evaluation manifest must be a complete v2 publication")

    raw_seeds = _array(manifest.get("held_out_seeds"), label="manifest.held_out_seeds")
    held_out_seeds = tuple(
        _integer(value, label=f"manifest.held_out_seeds[{index}]") for index, value in enumerate(raw_seeds)
    )
    if not held_out_seeds or len(set(held_out_seeds)) != len(held_out_seeds):
        raise ValueError("manifest held_out_seeds must be non-empty and unique")
    episodes_per_checkpoint = _integer(
        manifest.get("episodes_per_checkpoint"),
        label="manifest.episodes_per_checkpoint",
        minimum=1,
    )
    if episodes_per_checkpoint != len(held_out_seeds):
        raise ValueError("manifest episode count does not match held_out_seeds")

    raw_results = _array(manifest.get("results"), label="manifest.results")
    if len(raw_results) < 2:
        raise ValueError("paired analysis requires at least two checkpoints")
    checkpoints: list[CheckpointEpisodes] = []
    seen_checkpoint_ids: set[str] = set()
    for result_index, raw_result in enumerate(raw_results):
        result = _mapping(raw_result, label=f"manifest.results[{result_index}]")
        checkpoint_id = _string(
            result.get("checkpoint_id"),
            label=f"manifest.results[{result_index}].checkpoint_id",
        )
        if checkpoint_id in seen_checkpoint_ids:
            raise ValueError(f"duplicate paired checkpoint_id: {checkpoint_id}")
        seen_checkpoint_ids.add(checkpoint_id)
        checkpoint = _string(
            result.get("checkpoint"),
            label=f"manifest.results[{result_index}].checkpoint",
        )
        source_environment_steps = _integer(
            result.get("source_environment_steps"),
            label=f"manifest.results[{result_index}].source_environment_steps",
        )
        source_policy_version = _integer(
            result.get("source_policy_version"),
            label=f"manifest.results[{result_index}].source_policy_version",
        )
        expected_audit_sha256 = _string(
            result.get("audit_sha256"),
            label=f"manifest.results[{result_index}].audit_sha256",
        )
        audit_path = paired_root / f"checkpoint-{checkpoint_id}" / "evaluation.json"
        if not audit_path.is_file():
            raise ValueError(f"paired checkpoint audit is missing: {audit_path}")
        if _sha256(audit_path) != expected_audit_sha256:
            raise ValueError(f"paired checkpoint audit hash mismatch: {audit_path}")
        audit = _read_json(audit_path, label=f"checkpoint {checkpoint_id} audit")
        if audit.get("schema_version") != _AUDIT_SCHEMA:
            raise ValueError(f"checkpoint {checkpoint_id} audit schema is not {_AUDIT_SCHEMA}")
        evaluation_of = _mapping(
            audit.get("evaluation_of"),
            label=f"checkpoint {checkpoint_id}.evaluation_of",
        )
        if evaluation_of.get("checkpoint_id") != checkpoint_id:
            raise ValueError(f"checkpoint {checkpoint_id} audit identifies a different checkpoint")
        evaluation = _mapping(
            audit.get("evaluation"),
            label=f"checkpoint {checkpoint_id}.evaluation",
        )
        if (
            _integer(
                evaluation.get("episodes"),
                label=f"checkpoint {checkpoint_id}.evaluation.episodes",
                minimum=1,
            )
            != episodes_per_checkpoint
        ):
            raise ValueError(f"checkpoint {checkpoint_id} episode count differs from pair manifest")
        raw_metrics = _array(
            evaluation.get("episode_metrics"),
            label=f"checkpoint {checkpoint_id}.evaluation.episode_metrics",
        )
        episodes: dict[int, EpisodeRecord] = {}
        ordered_seeds: list[int] = []
        for episode_index, raw_episode in enumerate(raw_metrics):
            episode = _episode_from_mapping(
                raw_episode,
                label=f"checkpoint {checkpoint_id}.episode_metrics[{episode_index}]",
            )
            if episode.reset_seed in episodes:
                raise ValueError(f"checkpoint {checkpoint_id} duplicates reset_seed {episode.reset_seed}")
            episodes[episode.reset_seed] = episode
            ordered_seeds.append(episode.reset_seed)
        if tuple(ordered_seeds) != held_out_seeds:
            raise ValueError(f"checkpoint {checkpoint_id} seed sequence differs from paired held_out_seeds")
        checkpoints.append(
            CheckpointEpisodes(
                checkpoint_id=checkpoint_id,
                checkpoint=checkpoint,
                source_environment_steps=source_environment_steps,
                source_policy_version=source_policy_version,
                episodes=episodes,
            )
        )
    return manifest, held_out_seeds, checkpoints


def _percentile(values: Sequence[float], probability: float) -> float | None:
    if not values:
        return None
    if not 0.0 <= probability <= 1.0:
        raise ValueError("percentile probability must be in [0, 1]")
    ordered = sorted(float(value) for value in values)
    if len(ordered) == 1:
        return ordered[0]
    position = (len(ordered) - 1) * probability
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return ordered[lower]
    weight = position - lower
    return ordered[lower] * (1.0 - weight) + ordered[upper] * weight


def _distribution(values: Sequence[float]) -> dict[str, float | int | None]:
    if not values:
        return {
            "count": 0,
            "mean": None,
            "minimum": None,
            "p50": None,
            "p90": None,
            "maximum": None,
        }
    numeric = [float(value) for value in values]
    return {
        "count": len(numeric),
        "mean": sum(numeric) / len(numeric),
        "minimum": min(numeric),
        "p50": _percentile(numeric, 0.5),
        "p90": _percentile(numeric, 0.9),
        "maximum": max(numeric),
    }


def _stable_bootstrap_seed(base_seed: int, key: str) -> int:
    payload = f"{base_seed}\0{key}".encode()
    return int.from_bytes(hashlib.sha256(payload).digest()[:8], byteorder="big")


def _paired_bootstrap_mean_delta(
    left: Sequence[float],
    right: Sequence[float],
    *,
    samples: int,
    confidence: float,
    seed: int,
) -> dict[str, float | int | str | None]:
    if len(left) != len(right):
        raise ValueError("paired bootstrap inputs must have identical lengths")
    if not left:
        return {
            "method": "paired-percentile-bootstrap-mean-delta-v1",
            "samples": samples,
            "confidence": confidence,
            "paired_observations": 0,
            "observed_right_minus_left": None,
            "lower": None,
            "upper": None,
        }
    count = len(left)
    observed = sum(right[index] - left[index] for index in range(count)) / count
    generator = random.Random(seed)
    deltas: list[float] = []
    for _ in range(samples):
        total = 0.0
        for _index in range(count):
            sampled = generator.randrange(count)
            total += right[sampled] - left[sampled]
        deltas.append(total / count)
    tail = (1.0 - confidence) / 2.0
    return {
        "method": "paired-percentile-bootstrap-mean-delta-v1",
        "samples": samples,
        "confidence": confidence,
        "paired_observations": count,
        "observed_right_minus_left": observed,
        "lower": _percentile(deltas, tail),
        "upper": _percentile(deltas, 1.0 - tail),
    }


def _exact_two_sided_binomial_p_value(left_only: int, right_only: int) -> float:
    discordant = left_only + right_only
    if discordant == 0:
        return 1.0
    tail = min(left_only, right_only)
    cumulative = sum(math.comb(discordant, index) for index in range(tail + 1))
    p_value = 2.0 * float(cumulative) / float(2**discordant)
    return 1.0 if p_value >= 1.0 else p_value


def _checkpoint_summary(
    checkpoint: CheckpointEpisodes,
    held_out_seeds: Sequence[int],
) -> dict[str, object]:
    episodes = [checkpoint.episodes[seed] for seed in held_out_seeds]
    count = len(episodes)
    binary: dict[str, object] = {}
    for metric_name, better_direction in _BINARY_METRICS:
        metric_count = sum(episode.binary_metric(metric_name) for episode in episodes)
        binary[metric_name] = {
            "count": metric_count,
            "rate": metric_count / count,
            "better_direction": better_direction,
        }
    successful = [episode for episode in episodes if episode.run_won]
    return {
        "checkpoint_id": checkpoint.checkpoint_id,
        "checkpoint": checkpoint.checkpoint,
        "display_name": checkpoint.display_name,
        "source_environment_steps": checkpoint.source_environment_steps,
        "source_policy_version": checkpoint.source_policy_version,
        "episode_count": count,
        "binary_metrics": binary,
        "mean_max_floor": sum(episode.max_floor for episode in episodes) / count,
        "successful_run_efficiency": {
            "conditioning": "run_won == true; failed episodes are excluded",
            "successful_run_count": len(successful),
            "revivals_used": _distribution([float(episode.revivals_used) for episode in successful]),
            "player_hp_lost": _distribution([episode.player_hp_lost for episode in successful]),
        },
    }


def _binary_pairwise(
    left: CheckpointEpisodes,
    right: CheckpointEpisodes,
    held_out_seeds: Sequence[int],
    *,
    metric_name: str,
    better_direction: str,
    bootstrap_samples: int,
    confidence: float,
    bootstrap_seed: int,
) -> dict[str, object]:
    left_values = [left.episodes[seed].binary_metric(metric_name) for seed in held_out_seeds]
    right_values = [right.episodes[seed].binary_metric(metric_name) for seed in held_out_seeds]
    false_false = sum(not a and not b for a, b in zip(left_values, right_values, strict=True))
    false_true = sum(not a and b for a, b in zip(left_values, right_values, strict=True))
    true_false = sum(a and not b for a, b in zip(left_values, right_values, strict=True))
    true_true = sum(a and b for a, b in zip(left_values, right_values, strict=True))
    left_numeric = [float(value) for value in left_values]
    right_numeric = [float(value) for value in right_values]
    return {
        "better_direction": better_direction,
        "left_rate": sum(left_numeric) / len(left_numeric),
        "right_rate": sum(right_numeric) / len(right_numeric),
        "right_minus_left_rate": (sum(right_numeric) - sum(left_numeric)) / len(left_numeric),
        "paired_contingency": {
            "left_false_right_false": false_false,
            "left_false_right_true": false_true,
            "left_true_right_false": true_false,
            "left_true_right_true": true_true,
        },
        "exact_mcnemar": {
            "method": "two-sided exact binomial test on discordant pairs",
            "discordant_pairs": false_true + true_false,
            "left_only_true": true_false,
            "right_only_true": false_true,
            "p_value": _exact_two_sided_binomial_p_value(true_false, false_true),
        },
        "paired_bootstrap_ci": _paired_bootstrap_mean_delta(
            left_numeric,
            right_numeric,
            samples=bootstrap_samples,
            confidence=confidence,
            seed=_stable_bootstrap_seed(
                bootstrap_seed,
                f"{left.checkpoint_id}:{right.checkpoint_id}:{metric_name}",
            ),
        ),
    }


def _pairwise_comparison(
    left: CheckpointEpisodes,
    right: CheckpointEpisodes,
    held_out_seeds: Sequence[int],
    *,
    bootstrap_samples: int,
    confidence: float,
    bootstrap_seed: int,
) -> dict[str, object]:
    binary_metrics = {
        metric_name: _binary_pairwise(
            left,
            right,
            held_out_seeds,
            metric_name=metric_name,
            better_direction=better_direction,
            bootstrap_samples=bootstrap_samples,
            confidence=confidence,
            bootstrap_seed=bootstrap_seed,
        )
        for metric_name, better_direction in _BINARY_METRICS
    }
    left_floors = [float(left.episodes[seed].max_floor) for seed in held_out_seeds]
    right_floors = [float(right.episodes[seed].max_floor) for seed in held_out_seeds]
    both_success_seeds = [
        seed for seed in held_out_seeds if left.episodes[seed].run_won and right.episodes[seed].run_won
    ]
    efficiency: dict[str, object] = {
        "conditioning": "both checkpoints have run_won == true on the same reset_seed",
        "common_success_seed_count": len(both_success_seeds),
        "common_success_seeds": both_success_seeds,
    }
    for metric_name in ("revivals_used", "player_hp_lost"):
        left_values = [float(getattr(left.episodes[seed], metric_name)) for seed in both_success_seeds]
        right_values = [float(getattr(right.episodes[seed], metric_name)) for seed in both_success_seeds]
        efficiency[metric_name] = {
            "better_direction": "lower",
            "left": _distribution(left_values),
            "right": _distribution(right_values),
            "paired_bootstrap_ci": _paired_bootstrap_mean_delta(
                left_values,
                right_values,
                samples=bootstrap_samples,
                confidence=confidence,
                seed=_stable_bootstrap_seed(
                    bootstrap_seed,
                    f"{left.checkpoint_id}:{right.checkpoint_id}:both-success:{metric_name}",
                ),
            ),
        }
    return {
        "left_checkpoint_id": left.checkpoint_id,
        "right_checkpoint_id": right.checkpoint_id,
        "seed_count": len(held_out_seeds),
        "binary_metrics": binary_metrics,
        "max_floor": {
            "better_direction": "higher",
            "left_mean": sum(left_floors) / len(left_floors),
            "right_mean": sum(right_floors) / len(right_floors),
            "paired_bootstrap_ci": _paired_bootstrap_mean_delta(
                left_floors,
                right_floors,
                samples=bootstrap_samples,
                confidence=confidence,
                seed=_stable_bootstrap_seed(
                    bootstrap_seed,
                    f"{left.checkpoint_id}:{right.checkpoint_id}:max-floor",
                ),
            ),
        },
        "successful_run_efficiency": efficiency,
    }


def _nested_count(summary: Mapping[str, object], metric_name: str) -> int:
    binary = _mapping(summary.get("binary_metrics"), label="summary.binary_metrics")
    metric = _mapping(binary.get(metric_name), label=f"summary.binary_metrics.{metric_name}")
    return _integer(metric.get("count"), label=f"summary.binary_metrics.{metric_name}.count")


def _efficiency_value(summary: Mapping[str, object], metric_name: str, percentile: str) -> float | None:
    efficiency = _mapping(
        summary.get("successful_run_efficiency"),
        label="summary.successful_run_efficiency",
    )
    distribution = _mapping(
        efficiency.get(metric_name),
        label=f"summary.successful_run_efficiency.{metric_name}",
    )
    value = distribution.get(percentile)
    if value is None:
        return None
    if isinstance(value, bool) or not isinstance(value, int | float):
        raise ValueError(f"summary efficiency {metric_name}.{percentile} is invalid")
    return float(value)


def _ranking(summaries: Sequence[dict[str, object]]) -> list[dict[str, object]]:
    def quality_key(summary: dict[str, object]) -> tuple[float | int, ...]:
        hard_cycle_count = _nested_count(summary, "hard_cycle")
        eligible = hard_cycle_count == 0
        efficiency_values = [
            _efficiency_value(summary, "revivals_used", "p50"),
            _efficiency_value(summary, "revivals_used", "p90"),
            _efficiency_value(summary, "player_hp_lost", "p50"),
            _efficiency_value(summary, "player_hp_lost", "p90"),
        ]
        normalized_efficiency = [math.inf if value is None else value for value in efficiency_values]
        return (
            0 if eligible else 1,
            hard_cycle_count,
            -_nested_count(summary, "run_win"),
            _nested_count(summary, "any_stall"),
            _nested_count(summary, "combat_stall"),
            _nested_count(summary, "trusted_policy_failure"),
            -_nested_count(summary, "act3_reach"),
            -_nested_count(summary, "act1_clear"),
            *normalized_efficiency,
            -_report_number(summary.get("mean_max_floor"), label="summary.mean_max_floor"),
        )

    ordered = sorted(
        summaries,
        key=lambda summary: (quality_key(summary), _string(summary.get("checkpoint_id"), label="checkpoint_id")),
    )
    ranking: list[dict[str, object]] = []
    for index, summary in enumerate(ordered, start=1):
        hard_cycle_count = _nested_count(summary, "hard_cycle")
        eligible = hard_cycle_count == 0
        ranking.append(
            {
                "rank": index,
                "checkpoint_id": summary["checkpoint_id"],
                "display_name": summary["display_name"],
                "source_environment_steps": summary["source_environment_steps"],
                "source_policy_version": summary["source_policy_version"],
                "eligible_for_automatic_selection": eligible,
                "hard_invalidation_reasons": ([] if eligible else [f"hard_cycle_count={hard_cycle_count}"]),
                "ranking_values": {
                    "run_win_count": _nested_count(summary, "run_win"),
                    "hard_cycle_count": hard_cycle_count,
                    "any_stall_count": _nested_count(summary, "any_stall"),
                    "combat_stall_count": _nested_count(summary, "combat_stall"),
                    "trusted_policy_failure_count": _nested_count(summary, "trusted_policy_failure"),
                    "act3_reach_count": _nested_count(summary, "act3_reach"),
                    "act1_clear_count": _nested_count(summary, "act1_clear"),
                    "successful_run_revival_p50": _efficiency_value(summary, "revivals_used", "p50"),
                    "successful_run_revival_p90": _efficiency_value(summary, "revivals_used", "p90"),
                    "successful_run_hp_loss_p50": _efficiency_value(summary, "player_hp_lost", "p50"),
                    "successful_run_hp_loss_p90": _efficiency_value(summary, "player_hp_lost", "p90"),
                    "mean_max_floor": summary["mean_max_floor"],
                },
            }
        )
    return ranking


def analyze_paired_evaluation(
    paired_root: str | Path,
    *,
    bootstrap_samples: int = 20_000,
    confidence: float = 0.95,
    bootstrap_seed: int = 0,
) -> dict[str, object]:
    """Return a complete analysis without mutating the paired publication."""

    if isinstance(bootstrap_samples, bool) or not isinstance(bootstrap_samples, int) or bootstrap_samples <= 0:
        raise ValueError("bootstrap_samples must be a positive integer")
    if not 0.0 < confidence < 1.0:
        raise ValueError("confidence must be between zero and one")
    if isinstance(bootstrap_seed, bool) or not isinstance(bootstrap_seed, int) or bootstrap_seed < 0:
        raise ValueError("bootstrap_seed must be a non-negative integer")
    root = Path(paired_root).expanduser().resolve()
    manifest, held_out_seeds, checkpoints = _load_checkpoints(root)
    summaries = [_checkpoint_summary(checkpoint, held_out_seeds) for checkpoint in checkpoints]
    comparisons: list[dict[str, object]] = []
    for left_index, left in enumerate(checkpoints):
        for right in checkpoints[left_index + 1 :]:
            comparisons.append(
                _pairwise_comparison(
                    left,
                    right,
                    held_out_seeds,
                    bootstrap_samples=bootstrap_samples,
                    confidence=confidence,
                    bootstrap_seed=bootstrap_seed,
                )
            )
    ranking = _ranking(summaries)
    eligible_checkpoint_ids = [
        _string(item.get("checkpoint_id"), label="ranking.checkpoint_id")
        for item in ranking
        if item.get("eligible_for_automatic_selection") is True
    ]
    ineligible_checkpoint_ids = [
        _string(item.get("checkpoint_id"), label="ranking.checkpoint_id")
        for item in ranking
        if item.get("eligible_for_automatic_selection") is False
    ]
    return {
        "schema_version": _REPORT_SCHEMA,
        "created_unix_s": time.time(),
        "input": {
            "paired_output": str(root),
            "paired_manifest": str(root / "paired-evaluation.manifest.json"),
            "paired_manifest_sha256": _sha256(root / "paired-evaluation.manifest.json"),
            "paired_manifest_schema_version": manifest["schema_version"],
            "lineage": manifest.get("lineage"),
            "simulator_provenance": manifest.get("simulator_provenance"),
        },
        "paired_design": {
            "contract": "common-reset-seed deterministic evaluation; not a shared trajectory tape",
            "seed_count": len(held_out_seeds),
            "held_out_seeds": list(held_out_seeds),
            "checkpoint_count": len(checkpoints),
            "bootstrap_samples": bootstrap_samples,
            "bootstrap_confidence": confidence,
            "bootstrap_seed": bootstrap_seed,
        },
        "ranking_contract": {
            "method": "hard-cycle-eligible-strict-lexicographic-v2",
            "hard_invalidator": "hard_cycle_count > 0",
            "order": [
                "eligible_for_automatic_selection first (hard_cycle_count == 0)",
                "hard_cycle_count ascending (descriptive ordering among ineligible candidates)",
                "run_win_count descending",
                "any_stall_count ascending",
                "combat_stall_count ascending",
                "trusted_policy_failure_count ascending",
                "act3_reach_count descending",
                "act1_clear_count descending",
                "successful-run revival p50 ascending",
                "successful-run revival p90 ascending",
                "successful-run HP-loss p50 ascending",
                "successful-run HP-loss p90 ascending",
                "mean_max_floor descending",
                "checkpoint_id ascending (stable final tie-break)",
            ],
            "efficiency_conditioning": (
                "revival and HP-loss efficiency use successful runs only; failed-run costs never improve rank"
            ),
        },
        "selection": {
            "automatic_selection_allowed": bool(eligible_checkpoint_ids),
            "selected_checkpoint_id": (eligible_checkpoint_ids[0] if eligible_checkpoint_ids else None),
            "eligible_checkpoint_ids": eligible_checkpoint_ids,
            "ineligible_checkpoint_ids": ineligible_checkpoint_ids,
            "blocked_reason": (None if eligible_checkpoint_ids else "all checkpoints have at least one hard cycle"),
        },
        "ranking": ranking,
        "checkpoints": summaries,
        "pairwise_comparisons": comparisons,
    }


def _format_rate(metric: Mapping[str, object], episode_count: int) -> str:
    count = _integer(metric.get("count"), label="metric.count")
    return f"{count}/{episode_count} ({count / episode_count:.1%})"


def _format_optional(value: object, *, digits: int = 1) -> str:
    if value is None:
        return "—"
    if isinstance(value, bool) or not isinstance(value, int | float):
        raise ValueError("report contains a non-numeric statistic")
    return f"{float(value):.{digits}f}"


def render_markdown(report: Mapping[str, object]) -> str:
    """Render the machine report without reinterpreting its ranking contract."""

    design = _mapping(report.get("paired_design"), label="report.paired_design")
    summaries_raw = _array(report.get("checkpoints"), label="report.checkpoints")
    summaries = [_mapping(item, label="report.checkpoints[]") for item in summaries_raw]
    summary_by_id = {_string(item.get("checkpoint_id"), label="checkpoint_id"): item for item in summaries}
    ranking = _array(report.get("ranking"), label="report.ranking")
    selection = _mapping(report.get("selection"), label="report.selection")
    automatic_selection_allowed = _boolean(
        selection.get("automatic_selection_allowed"),
        label="report.selection.automatic_selection_allowed",
    )
    comparisons = _array(
        report.get("pairwise_comparisons"),
        label="report.pairwise_comparisons",
    )
    lines = [
        "# Paired checkpoint evaluation",
        "",
        f"- Checkpoints: **{design['checkpoint_count']}**",
        f"- Common held-out seeds: **{design['seed_count']}**",
        f"- Bootstrap: **{design['bootstrap_samples']}** paired resamples, "
        f"{_report_number(design['bootstrap_confidence'], label='bootstrap_confidence'):.1%} percentile CI",
        "- Pairing contract: common reset seeds with deterministic greedy policies; trajectories may diverge.",
        (
            f"- Automatic selection: **allowed**; selected checkpoint: " f"`{selection['selected_checkpoint_id']}`."
            if automatic_selection_allowed
            else "- Automatic selection: **BLOCKED** because every checkpoint has a hard cycle."
        ),
        "",
        "## Ranking",
        "",
        "| Rank | Eligible | Checkpoint | Policy | Steps | Run win | Hard cycle | Any stall | Act 3 | "
        "Win revival P50/P90 | Win HP-loss P50/P90 |",
        "|---:|---|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for raw_rank in ranking:
        rank = _mapping(raw_rank, label="report.ranking[]")
        checkpoint_id = _string(rank.get("checkpoint_id"), label="ranking.checkpoint_id")
        summary = summary_by_id[checkpoint_id]
        binary = _mapping(summary.get("binary_metrics"), label="summary.binary_metrics")
        efficiency = _mapping(
            summary.get("successful_run_efficiency"),
            label="summary.successful_run_efficiency",
        )
        revivals = _mapping(efficiency.get("revivals_used"), label="efficiency.revivals_used")
        hp_loss = _mapping(efficiency.get("player_hp_lost"), label="efficiency.player_hp_lost")
        episode_count = _integer(summary.get("episode_count"), label="summary.episode_count", minimum=1)
        lines.append(
            "| "
            f"{rank['rank']} | {'yes' if rank['eligible_for_automatic_selection'] else '**no**'} | "
            f"`{Path(str(summary['checkpoint'])).name}` | "
            f"{summary['source_policy_version']} | {summary['source_environment_steps']} | "
            f"{_format_rate(_mapping(binary['run_win'], label='run_win'), episode_count)} | "
            f"{_format_rate(_mapping(binary['hard_cycle'], label='hard_cycle'), episode_count)} | "
            f"{_format_rate(_mapping(binary['any_stall'], label='any_stall'), episode_count)} | "
            f"{_format_rate(_mapping(binary['act3_reach'], label='act3_reach'), episode_count)} | "
            f"{_format_optional(revivals.get('p50'))}/{_format_optional(revivals.get('p90'))} | "
            f"{_format_optional(hp_loss.get('p50'))}/{_format_optional(hp_loss.get('p90'))} |"
        )

    lines.extend(
        [
            "",
            "A hard cycle is an automatic-selection invalidator. Every eligible checkpoint ranks before every "
            "ineligible checkpoint; if all checkpoints are ineligible, the report deliberately selects none.",
            "Within the eligible set, ranking is strict and lexicographic: complete wins, stalls and trusted "
            "failures, Act 3, Act 1, successful-run efficiency, then mean maximum floor. Revival and HP-loss "
            "statistics exclude failed runs.",
            "",
            "## Pairwise full-run flips",
            "",
            "| Left | Right | Left-only win | Right-only win | McNemar exact p | "
            "Win-rate delta (right-left) | Paired bootstrap CI |",
            "|---|---|---:|---:|---:|---:|---:|",
        ]
    )
    for raw_comparison in comparisons:
        comparison = _mapping(raw_comparison, label="report.pairwise_comparisons[]")
        left_id = _string(comparison.get("left_checkpoint_id"), label="comparison.left_checkpoint_id")
        right_id = _string(comparison.get("right_checkpoint_id"), label="comparison.right_checkpoint_id")
        binary = _mapping(comparison.get("binary_metrics"), label="comparison.binary_metrics")
        wins = _mapping(binary.get("run_win"), label="comparison.binary_metrics.run_win")
        exact = _mapping(wins.get("exact_mcnemar"), label="run_win.exact_mcnemar")
        interval = _mapping(wins.get("paired_bootstrap_ci"), label="run_win.paired_bootstrap_ci")
        lines.append(
            f"| `{Path(str(summary_by_id[left_id]['checkpoint'])).name}` | "
            f"`{Path(str(summary_by_id[right_id]['checkpoint'])).name}` | "
            f"{exact['left_only_true']} | {exact['right_only_true']} | "
            f"{_report_number(exact['p_value'], label='run_win.p_value'):.4g} | "
            f"{_report_number(wins['right_minus_left_rate'], label='run_win.delta'):+.1%} | "
            f"[{_report_number(interval['lower'], label='run_win.ci.lower'):+.1%}, "
            f"{_report_number(interval['upper'], label='run_win.ci.upper'):+.1%}] |"
        )

    lines.extend(
        [
            "",
            "## Pairwise diagnostic deltas",
            "",
            "All deltas are right minus left. Positive is favorable only for higher-is-better metrics.",
            "",
            "| Pair | Metric | Direction | Delta | CI | Exact p |",
            "|---|---|---|---:|---:|---:|",
        ]
    )
    diagnostic_names = (
        "act1_clear",
        "act3_reach",
        "hard_cycle",
        "any_stall",
        "trusted_policy_failure",
    )
    for raw_comparison in comparisons:
        comparison = _mapping(raw_comparison, label="report.pairwise_comparisons[]")
        left_id = _string(comparison.get("left_checkpoint_id"), label="comparison.left_checkpoint_id")
        right_id = _string(comparison.get("right_checkpoint_id"), label="comparison.right_checkpoint_id")
        pair_label = (
            f"{Path(str(summary_by_id[left_id]['checkpoint'])).name} → "
            f"{Path(str(summary_by_id[right_id]['checkpoint'])).name}"
        )
        binary = _mapping(comparison.get("binary_metrics"), label="comparison.binary_metrics")
        for metric_name in diagnostic_names:
            metric = _mapping(binary.get(metric_name), label=f"comparison.{metric_name}")
            interval = _mapping(metric.get("paired_bootstrap_ci"), label=f"comparison.{metric_name}.ci")
            exact = _mapping(metric.get("exact_mcnemar"), label=f"comparison.{metric_name}.exact")
            lines.append(
                f"| {pair_label} | `{metric_name}` | {metric['better_direction']} | "
                f"{_report_number(metric['right_minus_left_rate'], label=f'{metric_name}.delta'):+.1%} | "
                f"[{_report_number(interval['lower'], label=f'{metric_name}.ci.lower'):+.1%}, "
                f"{_report_number(interval['upper'], label=f'{metric_name}.ci.upper'):+.1%}] | "
                f"{_report_number(exact['p_value'], label=f'{metric_name}.p_value'):.4g} |"
            )

    lines.extend(
        [
            "",
            "## Successful-run efficiency on common wins",
            "",
            "These paired efficiency comparisons use only seeds won by both checkpoints. They never treat an "
            "early failed run with few revivals as efficient.",
            "",
            "| Pair | Common wins | Revival mean delta [CI] | HP-loss mean delta [CI] |",
            "|---|---:|---:|---:|",
        ]
    )
    for raw_comparison in comparisons:
        comparison = _mapping(raw_comparison, label="report.pairwise_comparisons[]")
        left_id = _string(comparison.get("left_checkpoint_id"), label="comparison.left_checkpoint_id")
        right_id = _string(comparison.get("right_checkpoint_id"), label="comparison.right_checkpoint_id")
        efficiency = _mapping(
            comparison.get("successful_run_efficiency"),
            label="comparison.successful_run_efficiency",
        )
        revival = _mapping(efficiency.get("revivals_used"), label="efficiency.revivals_used")
        revival_ci = _mapping(revival.get("paired_bootstrap_ci"), label="revival.paired_bootstrap_ci")
        hp = _mapping(efficiency.get("player_hp_lost"), label="efficiency.player_hp_lost")
        hp_ci = _mapping(hp.get("paired_bootstrap_ci"), label="hp.paired_bootstrap_ci")

        def interval_text(interval: Mapping[str, object]) -> str:
            observed = interval.get("observed_right_minus_left")
            if observed is None:
                return "—"
            return (
                f"{_format_optional(observed):s} "
                f"[{_format_optional(interval.get('lower'))}, {_format_optional(interval.get('upper'))}]"
            )

        lines.append(
            f"| `{Path(str(summary_by_id[left_id]['checkpoint'])).name}` → "
            f"`{Path(str(summary_by_id[right_id]['checkpoint'])).name}` | "
            f"{efficiency['common_success_seed_count']} | {interval_text(revival_ci)} | "
            f"{interval_text(hp_ci)} |"
        )
    lines.extend(
        [
            "",
            "## Statistical notes",
            "",
            "- McNemar p-values are two-sided exact binomial tests over discordant reset-seed pairs.",
            "- Bootstrap intervals resample matched reset-seed indices and estimate the mean right-minus-left delta.",
            "- P50/P90 use linear interpolation over the successful-run subset.",
            "- Multiple pairwise tests are descriptive; this report does not apply a multiplicity correction.",
            "",
        ]
    )
    return "\n".join(lines)


def _atomic_publish_report(output_root: Path, report: Mapping[str, object]) -> tuple[Path, Path]:
    if output_root.exists():
        raise FileExistsError(f"paired analysis output already exists: {output_root}")
    output_root.parent.mkdir(parents=True, exist_ok=True)
    staging = output_root.with_name(f".{output_root.name}.staging-{uuid4().hex}")
    staging.mkdir(parents=False, exist_ok=False)
    try:
        json_path = staging / "paired-evaluation.analysis.json"
        markdown_path = staging / "paired-evaluation.analysis.md"
        json_path.write_text(
            json.dumps(report, indent=2, sort_keys=True, allow_nan=False) + "\n",
            encoding="utf-8",
        )
        markdown_path.write_text(render_markdown(report), encoding="utf-8")
        os.replace(staging, output_root)
    except BaseException:
        shutil.rmtree(staging, ignore_errors=True)
        raise
    return (
        output_root / "paired-evaluation.analysis.json",
        output_root / "paired-evaluation.analysis.md",
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m sts2_rl.paired_evaluation_report",
        description="Analyze a complete frozen paired evaluation without mutating it.",
    )
    parser.add_argument("--paired-output", required=True, help="complete paired evaluation directory")
    parser.add_argument(
        "--output-root",
        help="separate artifact output directory (default: reports/paired-evaluation/<input>-analysis)",
    )
    parser.add_argument("--bootstrap-samples", type=int, default=20_000)
    parser.add_argument("--bootstrap-confidence", type=float, default=0.95)
    parser.add_argument("--bootstrap-seed", type=int, default=0)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    paired_root = resolve_external_input_path(args.paired_output)
    default_output = (
        Path("reports")
        / "paired-evaluation"
        / f"{validate_artifact_component(paired_root.name, label='paired output name')}-analysis"
    )
    output_root = resolve_artifact_path(args.output_root, default=default_output)
    report = analyze_paired_evaluation(
        paired_root,
        bootstrap_samples=args.bootstrap_samples,
        confidence=args.bootstrap_confidence,
        bootstrap_seed=args.bootstrap_seed,
    )
    json_path, markdown_path = _atomic_publish_report(output_root, report)
    print(
        json.dumps(
            {
                "status": "complete",
                "paired_output": str(paired_root),
                "analysis_json": str(json_path),
                "analysis_markdown": str(markdown_path),
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())


__all__ = ["analyze_paired_evaluation", "build_parser", "main", "render_markdown"]
