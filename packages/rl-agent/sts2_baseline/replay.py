"""Coverage/recent/PER stratified replay for the restarted baseline."""

from __future__ import annotations

import math
from collections import OrderedDict, defaultdict
from collections.abc import Sequence
from copy import deepcopy
from dataclasses import dataclass, field
from typing import Literal

import numpy as np
import numpy.typing as npt

from .transition import BaselineTargets, BaselineTransition

ReplaySource = Literal["coverage", "recent", "per"]
_SOURCES: tuple[ReplaySource, ...] = ("coverage", "recent", "per")


def _token(value: str | None, default: str) -> str:
    normalized = str(value or "").strip().lower()
    return normalized or default


@dataclass(frozen=True, slots=True, order=True)
class ReplayStratum:
    """Coverage key used before any within-stratum sampling priority."""

    domain: str
    tier: str = "unknown"
    encounter_id: str = "unknown"
    deck_stage: str = "unknown"
    outcome: str = "ongoing"

    def __post_init__(self) -> None:
        object.__setattr__(self, "domain", _token(self.domain, "unknown"))
        object.__setattr__(self, "tier", _token(self.tier, "unknown"))
        object.__setattr__(self, "encounter_id", _token(self.encounter_id, "unknown"))
        object.__setattr__(self, "deck_stage", _token(self.deck_stage, "unknown"))
        object.__setattr__(self, "outcome", _token(self.outcome, "ongoing"))


@dataclass(frozen=True, slots=True)
class ReplaySample:
    transition: BaselineTransition
    targets: BaselineTargets
    stratum: ReplayStratum
    # Model-facing state is intentionally opaque to the replay policy.  This
    # keeps stratification/reward contracts independent from any observation
    # architecture while still allowing a typed trainer payload.
    payload: object | None = field(default=None, repr=False, compare=False)


@dataclass(frozen=True, slots=True)
class ReplayMix:
    coverage: float = 0.50
    recent: float = 0.25
    per: float = 0.25

    def __post_init__(self) -> None:
        if any(
            isinstance(value, bool) or not isinstance(value, int | float)
            for value in (self.coverage, self.recent, self.per)
        ):
            raise TypeError("replay mix weights must be numbers, not booleans")
        values = (float(self.coverage), float(self.recent), float(self.per))
        if any(not math.isfinite(value) or value < 0.0 for value in values):
            raise ValueError("replay mix weights must be finite and non-negative")
        if not math.isclose(sum(values), 1.0, rel_tol=1e-9, abs_tol=1e-9):
            raise ValueError("replay mix weights must sum to 1")

    def as_dict(self) -> dict[ReplaySource, float]:
        return {
            "coverage": float(self.coverage),
            "recent": float(self.recent),
            "per": float(self.per),
        }


@dataclass(frozen=True, slots=True)
class ReplayBatch:
    samples: tuple[ReplaySample, ...]
    indices: npt.NDArray[np.int64]
    probabilities: npt.NDArray[np.float64]
    importance_weights: npt.NDArray[np.float32]
    sources: tuple[ReplaySource, ...]

    def __post_init__(self) -> None:
        size = len(self.samples)
        if self.indices.shape != (size,):
            raise ValueError("replay batch indices have the wrong shape")
        if self.probabilities.shape != (size,):
            raise ValueError("replay batch probabilities have the wrong shape")
        if self.importance_weights.shape != (size,):
            raise ValueError("replay batch importance weights have the wrong shape")
        if len(self.sources) != size:
            raise ValueError("replay batch sources have the wrong length")
        if not np.issubdtype(self.indices.dtype, np.integer):
            raise ValueError("replay batch indices must be integers")
        if not np.all(np.isfinite(self.probabilities)) or np.any(
            self.probabilities <= 0.0
        ):
            raise ValueError("replay batch probabilities must be finite and positive")
        if not np.all(np.isfinite(self.importance_weights)) or np.any(
            self.importance_weights <= 0.0
        ):
            raise ValueError("replay importance weights must be finite and positive")
        if np.any(self.importance_weights > 1.0 + 1e-6):
            raise ValueError("normalized replay importance weights cannot exceed 1")


@dataclass(slots=True)
class _Entry:
    sample: ReplaySample
    priority: float
    insertion_order: int


class StratifiedReplayBuffer:
    """Three-source replay with stable IDs and refreshable PER priorities.

    Every source chooses a stratum uniformly first.  Coverage is uniform within
    that stratum, recent is uniform among its recent members, and PER uses
    ``priority**alpha`` within it.  This prevents a large encounter bucket or a
    single high-priority boss from erasing coverage of the other strata.
    """

    def __init__(
        self,
        capacity: int,
        *,
        recent_window: int = 10_000,
        mix: ReplayMix | None = None,
        alpha: float = 0.6,
        beta: float = 0.4,
        priority_epsilon: float = 1e-6,
        seed: int = 0,
    ) -> None:
        if isinstance(capacity, bool) or not isinstance(capacity, int) or capacity <= 0:
            raise ValueError("capacity must be positive")
        if (
            isinstance(recent_window, bool)
            or not isinstance(recent_window, int)
            or recent_window <= 0
        ):
            raise ValueError("recent_window must be positive")
        if isinstance(seed, bool) or not isinstance(seed, int):
            raise TypeError("seed must be an integer, not a boolean")
        if seed < 0:
            raise ValueError("seed must be non-negative")
        for label, value in (
            ("alpha", alpha),
            ("beta", beta),
            ("priority_epsilon", priority_epsilon),
        ):
            if isinstance(value, bool) or not isinstance(value, int | float):
                raise TypeError(f"{label} must be a number, not a boolean")
            if not math.isfinite(float(value)):
                raise ValueError(f"{label} must be finite")
        if not 0.0 <= float(alpha) <= 1.0:
            raise ValueError("alpha must be in [0, 1]")
        if not 0.0 <= float(beta) <= 1.0:
            raise ValueError("beta must be in [0, 1]")
        if not math.isfinite(float(priority_epsilon)) or float(priority_epsilon) <= 0.0:
            raise ValueError("priority_epsilon must be finite and positive")

        self.capacity = int(capacity)
        self.recent_window = int(recent_window)
        self.mix = mix or ReplayMix()
        self.alpha = float(alpha)
        self.beta = float(beta)
        self.priority_epsilon = float(priority_epsilon)
        self._entries: OrderedDict[int, _Entry] = OrderedDict()
        self._next_id = 0
        self._insertion_order = 0
        self._rng = np.random.default_rng(int(seed))

    def __len__(self) -> int:
        return len(self._entries)

    @property
    def indices(self) -> tuple[int, ...]:
        return tuple(self._entries)

    def add(self, sample: ReplaySample, *, priority: float = 1.0) -> int:
        normalized_priority = self._normalize_priority(priority)
        owned_sample = deepcopy(sample)
        replay_id = self._next_id
        self._next_id += 1
        self._entries[replay_id] = _Entry(
            sample=owned_sample,
            priority=normalized_priority,
            insertion_order=self._insertion_order,
        )
        self._insertion_order += 1
        while len(self._entries) > self.capacity:
            self._entries.popitem(last=False)
        return replay_id

    def extend(self, samples: Sequence[ReplaySample], *, priorities: Sequence[float] | None = None) -> list[int]:
        if priorities is not None and len(priorities) != len(samples):
            raise ValueError("priorities length must match samples")
        output: list[int] = []
        for position, sample in enumerate(samples):
            priority = 1.0 if priorities is None else float(priorities[position])
            output.append(self.add(sample, priority=priority))
        return output

    def update_priorities(self, indices: Sequence[int], priorities: Sequence[float], *, strict: bool = True) -> None:
        if len(indices) != len(priorities):
            raise ValueError("indices and priorities must have equal length")
        missing: list[int] = []
        for replay_id, priority in zip(indices, priorities, strict=True):
            entry = self._entries.get(int(replay_id))
            if entry is None:
                missing.append(int(replay_id))
                continue
            entry.priority = self._normalize_priority(priority)
        if strict and missing:
            raise KeyError(f"replay indices no longer exist: {missing}")

    def priority(self, replay_id: int) -> float:
        try:
            return float(self._entries[int(replay_id)].priority)
        except KeyError as exc:
            raise KeyError(f"unknown replay index: {replay_id}") from exc

    def source_probabilities(self, source: ReplaySource) -> dict[int, float]:
        """Return the exact marginal distribution for one replay source."""

        if source not in _SOURCES:
            raise ValueError(f"unknown replay source: {source!r}")
        groups = self._groups_for_source(source)
        if not groups:
            return {}
        stratum_probability = 1.0 / float(len(groups))
        probabilities: dict[int, float] = {}
        for members in groups.values():
            if source == "per":
                weights = np.asarray(
                    [self._entries[replay_id].priority**self.alpha for replay_id in members],
                    dtype=np.float64,
                )
                weights /= max(
                    float(weights.sum()),
                    float(np.finfo(np.float64).tiny),
                )
            else:
                weights = np.full((len(members),), 1.0 / float(len(members)), dtype=np.float64)
            for replay_id, weight in zip(members, weights, strict=True):
                probabilities[replay_id] = stratum_probability * float(weight)
        return probabilities

    def mixture_probabilities(self) -> dict[int, float]:
        effective_mix = self._effective_source_mix()
        output = {replay_id: 0.0 for replay_id in self._entries}
        for source, effective_weight in effective_mix.items():
            if effective_weight <= 0.0:
                continue
            for replay_id, probability in self.source_probabilities(source).items():
                output[replay_id] += effective_weight * probability
        return output

    def sample(self, batch_size: int, *, beta: float | None = None) -> ReplayBatch:
        if (
            isinstance(batch_size, bool)
            or not isinstance(batch_size, int)
            or batch_size <= 0
        ):
            raise ValueError("batch_size must be positive")
        if not self._entries:
            raise RuntimeError("cannot sample an empty replay buffer")
        if beta is not None and (
            isinstance(beta, bool) or not isinstance(beta, int | float)
        ):
            raise TypeError("beta must be a number, not a boolean")
        effective_beta = self.beta if beta is None else float(beta)
        if not math.isfinite(effective_beta):
            raise ValueError("beta must be finite")
        if not 0.0 <= effective_beta <= 1.0:
            raise ValueError("beta must be in [0, 1]")

        effective_mix = self._effective_source_mix()
        source_weights = np.asarray(
            [effective_mix[source] for source in _SOURCES],
            dtype=np.float64,
        )
        source_positions = self._rng.choice(
            len(_SOURCES),
            size=batch_size,
            replace=True,
            p=source_weights,
        )
        drawn_sources = [
            _SOURCES[int(source_position)] for source_position in source_positions
        ]
        draws_by_source: dict[ReplaySource, list[int]] = {}
        for source in _SOURCES:
            count = drawn_sources.count(source)
            draws_by_source[source] = (
                self._draw_stratified(source, count) if count > 0 else []
            )
        source_offsets = {source: 0 for source in _SOURCES}
        drawn_ids: list[int] = []
        for source in drawn_sources:
            offset = source_offsets[source]
            drawn_ids.append(draws_by_source[source][offset])
            source_offsets[source] = offset + 1

        mixture_probabilities = self.mixture_probabilities()
        probabilities = np.asarray(
            [mixture_probabilities[replay_id] for replay_id in drawn_ids],
            dtype=np.float64,
        )
        if np.any(probabilities <= 0.0) or not np.all(np.isfinite(probabilities)):
            raise RuntimeError("replay produced an invalid mixture probability")

        importance = np.power(
            float(len(self._entries)) * probabilities,
            -effective_beta,
        )
        support_probabilities = np.asarray(
            [probability for probability in mixture_probabilities.values() if probability > 0.0],
            dtype=np.float64,
        )
        max_support_importance = float(
            np.power(
                float(len(self._entries)) * support_probabilities.min(),
                -effective_beta,
            )
        )
        importance /= max(
            max_support_importance,
            float(np.finfo(np.float64).tiny),
        )
        return ReplayBatch(
            samples=tuple(self._entries[replay_id].sample for replay_id in drawn_ids),
            indices=np.asarray(drawn_ids, dtype=np.int64),
            probabilities=probabilities,
            importance_weights=importance.astype(np.float32),
            sources=tuple(drawn_sources),
        )

    def _normalize_priority(self, priority: float) -> float:
        normalized = float(priority)
        if not math.isfinite(normalized) or normalized < 0.0:
            raise ValueError("priority must be finite and non-negative")
        return max(normalized, self.priority_epsilon)

    def _recent_ids(self) -> set[int]:
        ids = list(self._entries)
        return set(ids[-min(len(ids), self.recent_window) :])

    def _groups_for_source(self, source: ReplaySource) -> dict[ReplayStratum, list[int]]:
        recent_ids = self._recent_ids() if source == "recent" else None
        groups: dict[ReplayStratum, list[int]] = defaultdict(list)
        for replay_id, entry in self._entries.items():
            if recent_ids is not None and replay_id not in recent_ids:
                continue
            groups[entry.sample.stratum].append(replay_id)
        return dict(sorted(groups.items(), key=lambda item: item[0]))

    def _effective_source_mix(self) -> dict[ReplaySource, float]:
        mix = self.mix.as_dict()
        available = [
            source
            for source in _SOURCES
            if mix[source] > 0.0 and self._groups_for_source(source)
        ]
        total_weight = sum(mix[source] for source in available)
        if total_weight <= 0.0:
            raise RuntimeError("replay has no available configured sampling source")
        return {
            source: mix[source] / total_weight if source in available else 0.0
            for source in _SOURCES
        }

    def _draw_stratified(self, source: ReplaySource, count: int) -> list[int]:
        groups = self._groups_for_source(source)
        strata = list(groups)
        if not strata:
            raise RuntimeError(f"replay source {source!r} has no candidates")

        stratum_plan: list[ReplayStratum] = []
        while len(stratum_plan) < count:
            permutation = self._rng.permutation(len(strata))
            stratum_plan.extend(strata[int(position)] for position in permutation)
        stratum_plan = stratum_plan[:count]

        output: list[int] = []
        for stratum in stratum_plan:
            members = groups[stratum]
            if source == "per":
                weights = np.asarray(
                    [self._entries[replay_id].priority**self.alpha for replay_id in members],
                    dtype=np.float64,
                )
                weights /= max(
                    float(weights.sum()),
                    float(np.finfo(np.float64).tiny),
                )
                local_index = int(self._rng.choice(len(members), p=weights))
            else:
                local_index = int(self._rng.integers(len(members)))
            output.append(members[local_index])
        return output
