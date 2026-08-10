"""Bounded sequence replay for the macro candidate-Q learner.

Stores complete macro episodes and samples contiguous windows (burn-in +
learn segment) with TD-error priorities.  Priorities start uniform, are
updated only by the learner, and never fabricate ordering between unseen
windows.  Bounded by episode count and per-sample quota, mirroring the
project's replay discipline; training-partition data only (enforced by the
episode DTO).
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Final

import numpy as np

from .transitions import MacroEpisode

MACRO_REPLAY_CONTRACT_VERSION: Final = "sts2-macro-replay-v1"


@dataclass(frozen=True, slots=True)
class MacroWindow:
    episode: MacroEpisode
    start: int
    burn_in: int
    length: int
    window_id: str

    @property
    def learn_slice(self) -> tuple[int, int]:
        begin = self.start + self.burn_in
        return begin, min(begin + self.length, len(self.episode.steps))


class MacroSequenceReplay:
    def __init__(
        self,
        *,
        capacity_episodes: int = 512,
        burn_in: int = 8,
        window_length: int = 16,
        priority_exponent: float = 0.6,
        seed: int = 0,
    ) -> None:
        if capacity_episodes <= 0 or burn_in < 0 or window_length <= 0:
            raise ValueError("macro replay bounds must be positive")
        if not 0.0 <= priority_exponent <= 1.0:
            raise ValueError("priority exponent must be in [0, 1]")
        self.capacity_episodes = capacity_episodes
        self.burn_in = burn_in
        self.window_length = window_length
        self.priority_exponent = priority_exponent
        self._episodes: list[MacroEpisode] = []
        self._priorities: dict[str, float] = {}
        self._rng = np.random.default_rng(seed)
        self._put_count = 0

    def __len__(self) -> int:
        return len(self._episodes)

    @property
    def total_steps(self) -> int:
        return sum(len(episode.steps) for episode in self._episodes)

    def put(self, episode: MacroEpisode) -> None:
        if any(existing.episode_id == episode.episode_id for existing in self._episodes):
            raise ValueError(f"macro episode {episode.episode_id!r} already stored")
        self._episodes.append(episode)
        self._put_count += 1
        while len(self._episodes) > self.capacity_episodes:
            evicted = self._episodes.pop(0)
            for key in [k for k in self._priorities if k.startswith(f"{evicted.episode_id}#")]:
                del self._priorities[key]

    def _windows(self) -> list[MacroWindow]:
        windows: list[MacroWindow] = []
        stride = max(self.window_length // 2, 1)
        for episode in self._episodes:
            steps = len(episode.steps)
            start = 0
            while True:
                window = MacroWindow(
                    episode=episode,
                    start=start,
                    burn_in=min(self.burn_in, max(steps - start - 1, 0)),
                    length=self.window_length,
                    window_id=f"{episode.episode_id}#{start}",
                )
                begin, end = window.learn_slice
                if begin < end:
                    windows.append(window)
                if start + stride >= steps or end >= steps:
                    break
                start += stride
        return windows

    def sample(self, count: int) -> tuple[MacroWindow, ...]:
        if count <= 0:
            raise ValueError("sample count must be positive")
        windows = self._windows()
        if not windows:
            return ()
        raw = np.asarray(
            [
                max(self._priorities.get(window.window_id, 1.0), 1e-6)
                ** self.priority_exponent
                for window in windows
            ],
            dtype=np.float64,
        )
        probabilities = raw / raw.sum()
        chosen = self._rng.choice(
            len(windows),
            size=min(count, len(windows)),
            replace=False,
            p=probabilities,
        )
        return tuple(windows[int(index)] for index in chosen)

    def update_priority(self, window_id: str, td_error: float) -> None:
        if not math.isfinite(td_error):
            raise ValueError("priority update requires a finite TD error")
        self._priorities[window_id] = abs(td_error)

    def metrics(self) -> dict[str, float | int | str]:
        return {
            "version": MACRO_REPLAY_CONTRACT_VERSION,
            "episodes": len(self._episodes),
            "steps": self.total_steps,
            "windows": len(self._windows()),
            "put_count": self._put_count,
            "prioritized_windows": len(self._priorities),
        }


__all__ = [
    "MACRO_REPLAY_CONTRACT_VERSION",
    "MacroSequenceReplay",
    "MacroWindow",
]
