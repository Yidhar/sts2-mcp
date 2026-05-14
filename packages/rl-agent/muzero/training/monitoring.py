"""Recent-tail combat monitoring helpers for MuZero training.

This module owns TensorBoard-only monitoring state. It is intentionally kept
away from ``muzero.train`` so training orchestration does not keep accumulating
rolling-metric and diagnostic policy details.
"""

from __future__ import annotations

from collections import Counter, deque
import re
from typing import Any

import numpy as np
from torch.utils.tensorboard import SummaryWriter

from combat_snapshot_dataset import infer_encounter_tier, normalize_encounter_id
from muzero.sts2_env.muzero_buffer import GameTrajectory


DEFAULT_RECENT_TAIL_WINDOWS = (64, 256)


class NullSummaryWriter:
    """No-op SummaryWriter used by async combat actors."""

    def add_scalar(self, *args, **kwargs) -> None:
        return None

    def flush(self) -> None:
        return None

    def close(self) -> None:
        return None


class RecentCombatMonitor:
    """Rolling recent-tail combat monitor for sandbox convergence checks."""

    def __init__(
        self,
        *,
        windows: tuple[int, ...] = DEFAULT_RECENT_TAIL_WINDOWS,
        tracked_encounters: list[str] | tuple[str, ...] | None = None,
        min_samples: int = 4,
    ) -> None:
        resolved_windows = sorted({max(int(window), 1) for window in windows if int(window) > 0})
        self.windows = tuple(resolved_windows or DEFAULT_RECENT_TAIL_WINDOWS)
        self.max_window = max(self.windows)
        self.min_samples = max(int(min_samples), 1)
        self.tracked_encounters = tuple(
            dict.fromkeys(
                encounter_id
                for encounter_id in (
                    normalize_encounter_id(raw_encounter_id)
                    for raw_encounter_id in (tracked_encounters or [])
                )
                if encounter_id
            )
        )
        self.hard_normal_encounters = tuple(
            encounter_id for encounter_id in self.tracked_encounters if infer_encounter_tier(encounter_id) == "normal"
        )
        self.hard_elite_encounters = tuple(
            encounter_id for encounter_id in self.tracked_encounters if infer_encounter_tier(encounter_id) == "elite"
        )
        self._episodes: deque[dict[str, Any]] = deque(maxlen=self.max_window)

    @staticmethod
    def _classify_outcome(metadata: dict[str, Any]) -> str:
        if bool(metadata.get("truncated", False)):
            return "truncated"
        if float(metadata.get("death_floor", 0.0) or 0.0) > 0.0:
            return "loss"
        if bool(metadata.get("terminated", False)):
            return "win"
        return "other"

    @staticmethod
    def _safe_tag(value: str) -> str:
        sanitized = re.sub(r"[^0-9a-zA-Z]+", "_", str(value or "").strip().lower())
        return sanitized.strip("_") or "unknown"

    # Floor reached buckets for the rolling tail histogram. The exact cut-points
    # are tied to STS2 Act 1 progression (act1 boss is around floor 17).
    _FLOOR_REACH_BUCKETS: tuple[int, ...] = (3, 5, 7, 10, 12, 15, 17, 20)

    @staticmethod
    def _aggregate(entries: list[dict[str, Any]]) -> dict[str, Any]:
        total = len(entries)
        if total <= 0:
            return {
                "episodes": 0,
                "win_rate": 0.0,
                "loss_rate": 0.0,
                "trunc_rate": 0.0,
                "other_rate": 0.0,
                "reward_mean": 0.0,
                "reward_median": 0.0,
                "length_mean": 0.0,
                "max_floor_mean": 0.0,
                "max_floor_median": 0.0,
                "max_floor_p25": 0.0,
                "max_floor_p75": 0.0,
                "death_floor_mean": 0.0,
                "act1_pass_rate": 0.0,
                "act1_boss_seen_rate": 0.0,
                "floor_reach_rates": {},
            }

        outcome_counter = Counter(entry["outcome"] for entry in entries)
        rewards = np.asarray([float(entry.get("reward", 0.0) or 0.0) for entry in entries], dtype=np.float32)
        lengths = np.asarray([float(entry.get("length", 0.0) or 0.0) for entry in entries], dtype=np.float32)
        max_floors = np.asarray(
            [float(entry.get("max_floor", 0.0) or 0.0) for entry in entries], dtype=np.float32
        )
        death_floors = np.asarray(
            [float(entry.get("death_floor", 0.0) or 0.0) for entry in entries], dtype=np.float32
        )
        act1_pass = np.asarray(
            [1.0 if entry.get("act1_clear") else 0.0 for entry in entries], dtype=np.float32
        )
        act1_boss = np.asarray(
            [1.0 if entry.get("act1_boss_seen") else 0.0 for entry in entries], dtype=np.float32
        )
        floor_reach_rates: dict[int, float] = {}
        for threshold in RecentCombatMonitor._FLOOR_REACH_BUCKETS:
            floor_reach_rates[int(threshold)] = (
                float((max_floors >= float(threshold)).mean()) if max_floors.size else 0.0
            )
        return {
            "episodes": total,
            "win_rate": float(outcome_counter.get("win", 0) / total),
            "loss_rate": float(outcome_counter.get("loss", 0) / total),
            "trunc_rate": float(outcome_counter.get("truncated", 0) / total),
            "other_rate": float(outcome_counter.get("other", 0) / total),
            "reward_mean": float(rewards.mean()) if rewards.size else 0.0,
            "reward_median": float(np.median(rewards)) if rewards.size else 0.0,
            "length_mean": float(lengths.mean()) if lengths.size else 0.0,
            "max_floor_mean": float(max_floors.mean()) if max_floors.size else 0.0,
            "max_floor_median": float(np.median(max_floors)) if max_floors.size else 0.0,
            "max_floor_p25": float(np.percentile(max_floors, 25)) if max_floors.size else 0.0,
            "max_floor_p75": float(np.percentile(max_floors, 75)) if max_floors.size else 0.0,
            "death_floor_mean": float(death_floors.mean()) if death_floors.size else 0.0,
            "act1_pass_rate": float(act1_pass.mean()) if act1_pass.size else 0.0,
            "act1_boss_seen_rate": float(act1_boss.mean()) if act1_boss.size else 0.0,
            "floor_reach_rates": floor_reach_rates,
        }

    def record_episode(self, metadata: dict[str, Any] | None) -> dict[int, dict[str, Any]]:
        if not isinstance(metadata, dict):
            return {}
        encounter_id = normalize_encounter_id(metadata.get("encounter_id"))
        encounter_tier = str(metadata.get("encounter_tier") or infer_encounter_tier(encounter_id)).strip().lower()
        entry = {
            "encounter_id": encounter_id,
            "encounter_tier": encounter_tier,
            "outcome": self._classify_outcome(metadata),
            "reward": float(
                metadata.get("episode_total_reward", metadata.get("episode_reward", 0.0)) or 0.0
            ),
            "length": int(metadata.get("episode_length", 0) or 0),
            "max_floor": float(metadata.get("max_floor", 0.0) or 0.0),
            "death_floor": float(metadata.get("death_floor", 0.0) or 0.0),
            "act1_clear": bool(metadata.get("act1_clear", False)),
            "act1_boss_seen": bool(metadata.get("act1_boss_seen", False)),
        }
        self._episodes.append(entry)
        return self.snapshot()

    def snapshot(self) -> dict[int, dict[str, Any]]:
        entries = list(self._episodes)
        results: dict[int, dict[str, Any]] = {}
        for window in self.windows:
            subset = entries[-window:]
            stats = self._aggregate(subset)
            tiers: dict[str, dict[str, Any]] = {}
            for tier in ("weak", "normal", "elite", "boss"):
                tier_entries = [entry for entry in subset if entry.get("encounter_tier") == tier]
                tier_stats = self._aggregate(tier_entries)
                tier_stats["sample_share"] = float(len(tier_entries) / len(subset)) if subset else 0.0
                tiers[tier] = tier_stats

            groups = {
                "hard_normal": self._aggregate(
                    [entry for entry in subset if entry.get("encounter_id") in self.hard_normal_encounters]
                ),
                "hard_elite": self._aggregate(
                    [entry for entry in subset if entry.get("encounter_id") in self.hard_elite_encounters]
                ),
            }

            tracked: dict[str, dict[str, Any]] = {}
            for encounter_id in self.tracked_encounters:
                encounter_entries = [entry for entry in subset if entry.get("encounter_id") == encounter_id]
                tracked[encounter_id] = self._aggregate(encounter_entries)

            stats["tiers"] = tiers
            stats["groups"] = groups
            stats["tracked_encounters"] = tracked
            results[int(window)] = stats
        return results

    def log_to_writer(self, writer: SummaryWriter | NullSummaryWriter, episode_step: int) -> dict[int, dict[str, Any]]:
        snapshot = self.snapshot()
        for window, stats in snapshot.items():
            prefix = f"recent_tail/{window}"
            writer.add_scalar(f"{prefix}/win_rate", float(stats.get("win_rate", 0.0)), episode_step)
            writer.add_scalar(f"{prefix}/loss_rate", float(stats.get("loss_rate", 0.0)), episode_step)
            writer.add_scalar(f"{prefix}/trunc_rate", float(stats.get("trunc_rate", 0.0)), episode_step)
            writer.add_scalar(f"{prefix}/reward_mean", float(stats.get("reward_mean", 0.0)), episode_step)
            writer.add_scalar(f"{prefix}/reward_median", float(stats.get("reward_median", 0.0)), episode_step)
            writer.add_scalar(f"{prefix}/length_mean", float(stats.get("length_mean", 0.0)), episode_step)
            writer.add_scalar(f"{prefix}/act1_pass_rate", float(stats.get("act1_pass_rate", 0.0)), episode_step)
            writer.add_scalar(f"{prefix}/act1_boss_seen_rate", float(stats.get("act1_boss_seen_rate", 0.0)), episode_step)
            writer.add_scalar(f"{prefix}/max_floor_mean", float(stats.get("max_floor_mean", 0.0)), episode_step)
            writer.add_scalar(f"{prefix}/max_floor_median", float(stats.get("max_floor_median", 0.0)), episode_step)
            writer.add_scalar(f"{prefix}/max_floor_p25", float(stats.get("max_floor_p25", 0.0)), episode_step)
            writer.add_scalar(f"{prefix}/max_floor_p75", float(stats.get("max_floor_p75", 0.0)), episode_step)
            writer.add_scalar(f"{prefix}/death_floor_mean", float(stats.get("death_floor_mean", 0.0)), episode_step)
            floor_reach_rates = stats.get("floor_reach_rates") if isinstance(stats.get("floor_reach_rates"), dict) else {}
            for threshold, rate in floor_reach_rates.items():
                writer.add_scalar(
                    f"{prefix}/reached_floor_{int(threshold):02d}_rate",
                    float(rate),
                    episode_step,
                )

            tiers = stats.get("tiers") if isinstance(stats.get("tiers"), dict) else {}
            for tier, tier_stats in tiers.items():
                writer.add_scalar(
                    f"{prefix}/{tier}_win_rate",
                    float(tier_stats.get("win_rate", 0.0)),
                    episode_step,
                )
                writer.add_scalar(
                    f"{prefix}/{tier}_sample_share",
                    float(tier_stats.get("sample_share", 0.0)),
                    episode_step,
                )
                writer.add_scalar(
                    f"{prefix}/{tier}_sample_count",
                    float(tier_stats.get("episodes", 0.0)),
                    episode_step,
                )

            groups = stats.get("groups") if isinstance(stats.get("groups"), dict) else {}
            for group_name in ("hard_normal", "hard_elite"):
                group_stats = groups.get(group_name) if isinstance(groups.get(group_name), dict) else None
                if not group_stats:
                    continue
                writer.add_scalar(
                    f"{prefix}/{group_name}_win_rate",
                    float(group_stats.get("win_rate", 0.0)),
                    episode_step,
                )
                writer.add_scalar(
                    f"{prefix}/{group_name}_sample_count",
                    float(group_stats.get("episodes", 0.0)),
                    episode_step,
                )

            tracked = stats.get("tracked_encounters") if isinstance(stats.get("tracked_encounters"), dict) else {}
            for encounter_id, encounter_stats in tracked.items():
                if int(encounter_stats.get("episodes", 0) or 0) < self.min_samples:
                    continue
                safe_encounter = self._safe_tag(encounter_id.replace("ENCOUNTER.", ""))
                writer.add_scalar(
                    f"{prefix}/encounter_{safe_encounter}_win_rate",
                    float(encounter_stats.get("win_rate", 0.0)),
                    episode_step,
                )
                writer.add_scalar(
                    f"{prefix}/encounter_{safe_encounter}_sample_count",
                    float(encounter_stats.get("episodes", 0.0)),
                    episode_step,
                )
        return snapshot


class EpisodeCaptureBuffer:
    """Lightweight buffer that only captures the most recent completed trajectory."""

    def __init__(self) -> None:
        self.latest_trajectory: GameTrajectory | None = None

    def save_episode(
        self,
        trajectory: GameTrajectory,
        *,
        discount: float = 0.997,
        n_steps: int = 10,
    ) -> None:
        self.latest_trajectory = trajectory

    def pop_latest(self) -> GameTrajectory | None:
        trajectory = self.latest_trajectory
        self.latest_trajectory = None
        return trajectory

    def __len__(self) -> int:
        return int(len(self.latest_trajectory) if self.latest_trajectory is not None else 0)
