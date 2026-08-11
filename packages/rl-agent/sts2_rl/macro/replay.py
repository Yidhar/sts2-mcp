"""Bounded, uniform sequence replay for the macro candidate-Q learner.

Every learning window starts from an explicit learning index.  Its history
slice starts at the latest factual recurrent reset at or before that boundary,
or at the beginning of the episode when no reset exists.  Replaying that slice
without gradients reconstructs the current network's exact recurrent state at
the learning boundary without redundantly replaying prior combat encounters.

Sampling is uniform.  The previous implementation sampled by TD priority but
optimized an uncorrected, equally-weighted loss, which silently changed the
training objective.  A future prioritized implementation must return sampling
probabilities together with bounded importance weights; until then the stable
contract is deliberately simple.
"""

from __future__ import annotations

import copy
from dataclasses import dataclass
from typing import Any, Final, Literal

import numpy as np

from .transitions import MacroEpisode

MACRO_REPLAY_CONTRACT_VERSION: Final = "sts2-macro-replay-v4"


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
        window_length: int = 16,
        seed: int = 0,
        control_domain: Literal["macro", "combat"] = "macro",
    ) -> None:
        if capacity_episodes <= 0 or window_length <= 0:
            raise ValueError("macro replay bounds must be positive")
        self.capacity_episodes = capacity_episodes
        self.window_length = window_length
        if control_domain not in {"macro", "combat"}:
            raise ValueError("replay control_domain must be macro or combat")
        self.control_domain = control_domain
        self._episodes: list[MacroEpisode] = []
        self._rng = np.random.default_rng(seed)
        self._put_count = 0

    def __len__(self) -> int:
        return len(self._episodes)

    @property
    def total_steps(self) -> int:
        return sum(len(episode.steps) for episode in self._episodes)

    def put(self, episode: MacroEpisode) -> None:
        if episode.control_domain != self.control_domain:
            raise ValueError("macro replay refuses an episode from another domain")
        if any(existing.episode_id == episode.episode_id for existing in self._episodes):
            raise ValueError(f"macro episode {episode.episode_id!r} already stored")
        self._episodes.append(episode)
        self._put_count += 1
        while len(self._episodes) > self.capacity_episodes:
            self._episodes.pop(0)

    def _windows(self) -> list[MacroWindow]:
        windows: list[MacroWindow] = []
        stride = max(self.window_length // 2, 1)
        for episode in self._episodes:
            steps = len(episode.steps)
            reset_indices = tuple(
                index
                for index, step in enumerate(episode.steps)
                if step.recurrent_reset
            )
            reset_cursor = 0
            recurrent_start = 0
            learn_start = 0
            while True:
                while (
                    reset_cursor < len(reset_indices)
                    and reset_indices[reset_cursor] <= learn_start
                ):
                    recurrent_start = reset_indices[reset_cursor]
                    reset_cursor += 1
                window = MacroWindow(
                    episode=episode,
                    # A factual reset makes the earlier prefix causally
                    # irrelevant.  Starting there is exactly equivalent to
                    # replaying from episode zero, not a truncated-state
                    # approximation.
                    start=recurrent_start,
                    burn_in=learn_start - recurrent_start,
                    length=self.window_length,
                    window_id=f"{episode.episode_id}#{learn_start}",
                )
                begin, end = window.learn_slice
                if begin < end:
                    windows.append(window)
                if end >= steps:
                    break
                learn_start += stride
        return windows

    def sample(self, count: int) -> tuple[MacroWindow, ...]:
        if count <= 0:
            raise ValueError("sample count must be positive")
        windows = self._windows()
        if not windows:
            return ()
        chosen = self._rng.choice(
            len(windows),
            size=min(count, len(windows)),
            replace=False,
        )
        return tuple(windows[int(index)] for index in chosen)

    def state_dict(self) -> dict[str, Any]:
        """Return the complete bounded replay state needed for continuation."""

        return {
            "version": MACRO_REPLAY_CONTRACT_VERSION,
            "capacity_episodes": self.capacity_episodes,
            "window_length": self.window_length,
            "control_domain": self.control_domain,
            "episodes": tuple(self._episodes),
            "rng_state": copy.deepcopy(self._rng.bit_generator.state),
            "put_count": self._put_count,
        }

    def load_state_dict(self, state: dict[str, Any]) -> None:
        """Restore a replay produced by :meth:`state_dict`."""

        if state.get("version") != MACRO_REPLAY_CONTRACT_VERSION:
            raise ValueError("macro replay state version differs")
        if int(state.get("capacity_episodes", -1)) != self.capacity_episodes:
            raise ValueError("macro replay capacity differs")
        if int(state.get("window_length", -1)) != self.window_length:
            raise ValueError("macro replay window length differs")
        if state.get("control_domain") != self.control_domain:
            raise ValueError("macro replay control domain differs")
        episodes = tuple(state.get("episodes", ()))
        if len(episodes) > self.capacity_episodes or not all(
            isinstance(episode, MacroEpisode) for episode in episodes
        ):
            raise ValueError("macro replay episodes are invalid")
        if any(episode.control_domain != self.control_domain for episode in episodes):
            raise ValueError("macro replay state contains another control domain")
        episode_ids = [episode.episode_id for episode in episodes]
        if len(set(episode_ids)) != len(episode_ids):
            raise ValueError("macro replay episode ids are not unique")
        put_count = int(state.get("put_count", -1))
        if put_count < len(episodes):
            raise ValueError("macro replay put count is invalid")
        self._episodes = list(episodes)
        self._put_count = put_count
        self._rng.bit_generator.state = copy.deepcopy(state["rng_state"])

    def metrics(self) -> dict[str, float | int | str]:
        return {
            "version": MACRO_REPLAY_CONTRACT_VERSION,
            "episodes": len(self._episodes),
            "steps": self.total_steps,
            "windows": len(self._windows()),
            "put_count": self._put_count,
            "sampling": "uniform",
            "control_domain": self.control_domain,
        }


__all__ = [
    "MACRO_REPLAY_CONTRACT_VERSION",
    "MacroSequenceReplay",
    "MacroWindow",
]
