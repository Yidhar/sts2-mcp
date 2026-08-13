"""Run-balanced domain views over one immutable mixed decision graph."""

from __future__ import annotations

import copy
from dataclasses import dataclass
from typing import Any, Final, Literal

import numpy as np

from .transitions import HybridRun

HYBRID_REPLAY_VERSION: Final = "sts2-hybrid-run-replay-v3"


@dataclass(frozen=True, slots=True)
class DomainView:
    run: HybridRun
    domain: Literal["macro", "combat"]
    prefix_indices: tuple[int, ...]
    learn_indices: tuple[int, ...]
    view_id: str


@dataclass(frozen=True, slots=True)
class HybridViewPair:
    run_id: str
    macro: DomainView
    combat: DomainView


class HybridRunReplay:
    """Sample a run first, then one sparse view from each control domain.

    The macro prefix consists only of prior macro facts; skipped combat nodes
    affect its Bellman edge through factual reward/discount composition, not
    by writing tactical observations into slow memory.  The combat prefix is
    current-encounter local plus all earlier macro nodes needed to reconstruct
    slow memory under current parameters.  No behavior hidden state is stored.
    """

    def __init__(
        self,
        *,
        capacity_runs: int = 256,
        macro_unroll: int = 8,
        combat_unroll: int = 32,
        maximum_view_reuse: int = 8,
        seed: int = 0,
    ) -> None:
        if min(capacity_runs, macro_unroll, combat_unroll, maximum_view_reuse) <= 0:
            raise ValueError("hybrid replay bounds must be positive")
        self.capacity_runs = int(capacity_runs)
        self.macro_unroll = int(macro_unroll)
        self.combat_unroll = int(combat_unroll)
        self.maximum_view_reuse = int(maximum_view_reuse)
        self._runs: list[HybridRun] = []
        self._rng = np.random.default_rng(seed)
        self._put_count = 0
        self._sample_counts: dict[tuple[str, str, int], int] = {}

    def __len__(self) -> int:
        return len(self._runs)

    @property
    def total_steps(self) -> int:
        return sum(len(run.steps) for run in self._runs)

    @property
    def total_environment_steps(self) -> int:
        return sum(run.environment_steps for run in self._runs)

    def put(self, run: HybridRun) -> None:
        if any(existing.run_id == run.run_id for existing in self._runs):
            raise ValueError(f"hybrid run {run.run_id!r} already stored")
        self._runs.append(run)
        self._put_count += 1
        if len(self._runs) > self.capacity_runs:
            removed = self._runs.pop(0)
            self._sample_counts = {
                key: value
                for key, value in self._sample_counts.items()
                if key[0] != removed.run_id
            }

    def _fresh_anchor(
        self,
        run: HybridRun,
        domain: Literal["macro", "combat"],
        candidates: tuple[int, ...],
    ) -> int | None:
        available = tuple(
            anchor
            for anchor in candidates
            if self._sample_counts.get((run.run_id, domain, anchor), 0)
            < self.maximum_view_reuse
        )
        if not available:
            return None
        minimum = min(
            self._sample_counts.get((run.run_id, domain, anchor), 0)
            for anchor in available
        )
        least_used = tuple(
            anchor
            for anchor in available
            if self._sample_counts.get((run.run_id, domain, anchor), 0) == minimum
        )
        return int(least_used[int(self._rng.integers(len(least_used)))])

    def _macro_view(self, run: HybridRun) -> DomainView | None:
        indices = run.indices("macro")
        anchor = self._fresh_anchor(run, "macro", indices)
        if anchor is None:
            return None
        anchor_position = indices.index(anchor)
        learn = indices[anchor_position : anchor_position + self.macro_unroll]
        return DomainView(
            run=run,
            domain="macro",
            prefix_indices=indices[:anchor_position],
            learn_indices=learn,
            view_id=f"{run.run_id}:macro:{anchor}",
        )

    def _combat_view(self, run: HybridRun) -> DomainView | None:
        indices = run.indices("combat")
        anchor = self._fresh_anchor(run, "combat", indices)
        if anchor is None:
            return None
        encounter_id = run.steps[anchor].encounter_id
        encounter = tuple(
            index
            for index in indices
            if run.steps[index].encounter_id == encounter_id
        )
        position = encounter.index(anchor)
        learn = encounter[position : position + self.combat_unroll]
        macro_prefix = tuple(
            index
            for index in run.indices("macro")
            if index < anchor
        )
        combat_prefix = encounter[:position]
        prefix = tuple(sorted((*macro_prefix, *combat_prefix)))
        return DomainView(
            run=run,
            domain="combat",
            prefix_indices=prefix,
            learn_indices=learn,
            view_id=f"{run.run_id}:combat:{anchor}",
        )

    def sample_pairs(self, count: int) -> tuple[HybridViewPair, ...]:
        if count <= 0:
            raise ValueError("hybrid replay sample count must be positive")
        eligible = [
            run
            for run in self._runs
            if run.macro_steps
            and run.combat_steps
            and any(
                self._sample_counts.get((run.run_id, "macro", anchor), 0)
                < self.maximum_view_reuse
                for anchor in run.indices("macro")
            )
            and any(
                self._sample_counts.get((run.run_id, "combat", anchor), 0)
                < self.maximum_view_reuse
                for anchor in run.indices("combat")
            )
        ]
        if not eligible:
            return ()
        indices = self._rng.choice(
            len(eligible), size=min(count, len(eligible)), replace=False
        )
        pairs: list[HybridViewPair] = []
        for raw_index in np.atleast_1d(indices):
            run = eligible[int(raw_index)]
            macro = self._macro_view(run)
            combat = self._combat_view(run)
            if macro is None or combat is None:  # eligibility above owns this
                continue
            for view in (macro, combat):
                anchor = view.learn_indices[0]
                key = (run.run_id, view.domain, anchor)
                self._sample_counts[key] = self._sample_counts.get(key, 0) + 1
            pairs.append(HybridViewPair(run.run_id, macro, combat))
        return tuple(pairs)

    def state_dict(self) -> dict[str, Any]:
        return {
            "version": HYBRID_REPLAY_VERSION,
            "capacity_runs": self.capacity_runs,
            "macro_unroll": self.macro_unroll,
            "combat_unroll": self.combat_unroll,
            "maximum_view_reuse": self.maximum_view_reuse,
            "runs": tuple(self._runs),
            "rng_state": copy.deepcopy(self._rng.bit_generator.state),
            "put_count": self._put_count,
            "sample_counts": dict(self._sample_counts),
        }

    def load_state_dict(self, state: dict[str, Any]) -> None:
        if state.get("version") != HYBRID_REPLAY_VERSION:
            raise ValueError("hybrid replay state version differs")
        expected = (
            self.capacity_runs,
            self.macro_unroll,
            self.combat_unroll,
            self.maximum_view_reuse,
        )
        actual = (
            int(state.get("capacity_runs", -1)),
            int(state.get("macro_unroll", -1)),
            int(state.get("combat_unroll", -1)),
            int(state.get("maximum_view_reuse", -1)),
        )
        if actual != expected:
            raise ValueError("hybrid replay shape differs")
        runs = tuple(state.get("runs", ()))
        if len(runs) > self.capacity_runs or not all(isinstance(run, HybridRun) for run in runs):
            raise ValueError("hybrid replay runs are invalid")
        self._runs = list(runs)
        self._put_count = int(state.get("put_count", len(runs)))
        self._sample_counts = {
            tuple(key): int(value)
            for key, value in dict(state.get("sample_counts", {})).items()
        }
        self._rng.bit_generator.state = copy.deepcopy(state["rng_state"])

    def metrics(self) -> dict[str, int | float | str]:
        samples = tuple(self._sample_counts.values())
        return {
            "version": HYBRID_REPLAY_VERSION,
            "runs": len(self._runs),
            "steps": self.total_steps,
            "environment_steps": self.total_environment_steps,
            "macro_steps": sum(run.macro_steps for run in self._runs),
            "combat_steps": sum(run.combat_steps for run in self._runs),
            "put_count": self._put_count,
            "maximum_view_reuse": max(samples, default=0),
            "configured_maximum_view_reuse": self.maximum_view_reuse,
            "mean_view_reuse": sum(samples) / len(samples) if samples else 0.0,
        }


__all__ = [
    "HYBRID_REPLAY_VERSION",
    "DomainView",
    "HybridRunReplay",
    "HybridViewPair",
]
