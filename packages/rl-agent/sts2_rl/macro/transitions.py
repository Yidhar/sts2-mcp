"""Macro-domain semantic transitions for the isolated candidate-Q learner.

One :class:`MacroStep` is one meaningful macro decision under the semantic
decision graph: the semantic-candidate snapshot, the executed semantic index,
the factual reward accumulated until the NEXT macro decision (executor suffix
and any combat segment folded in, per the stage-2 bridge rule: realized
returns, no learned bridge), and the durable-floor clock discount linking the
two decisions.  Combat decisions never appear here — they belong to the frozen
combat champion; their consequences arrive only through the accumulated
reward and the clock.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Final, Literal

from sts2_rl.encoding import EncodedDecisionSnapshot

MACRO_TRANSITION_CONTRACT_VERSION: Final = "sts2-macro-transition-v3"


@dataclass(frozen=True, slots=True)
class MacroStep:
    """One transition on exactly the candidate surface scored by authority.

    ``snapshot.action_mask`` is the semantic surface (for example
    ``Rest``, ``Smith(A)``, ``Smith(B)``), not the native parent/picker union.
    Consequently collection, replay bootstrap and learner gather all index
    the same action set.
    """

    snapshot: EncodedDecisionSnapshot
    action_index: int
    reward: float
    discount: float
    terminal: bool
    surface: str
    branch: str
    control_domain: Literal["macro", "combat"]
    recurrent_reset: bool = False
    target_key: str | None = None
    behavior_epsilon: float = 0.0
    version: str = MACRO_TRANSITION_CONTRACT_VERSION

    def __post_init__(self) -> None:
        if not isinstance(self.snapshot, EncodedDecisionSnapshot):
            raise TypeError("macro step snapshot has the wrong type")
        if isinstance(self.action_index, bool) or not isinstance(self.action_index, int):
            raise TypeError("macro step action_index must be an integer")
        mask = self.snapshot.action_mask
        if not 0 <= self.action_index < len(mask):
            raise ValueError("macro step action_index is outside its candidate set")
        if not bool(mask[self.action_index]):
            raise ValueError("macro step executed an action the mask forbids")
        if not math.isfinite(self.reward):
            raise ValueError("macro step reward must be finite")
        if not math.isfinite(self.discount) or not 0.0 <= self.discount <= 1.0:
            raise ValueError("macro step discount must be in [0, 1]")
        if not isinstance(self.terminal, bool):
            raise TypeError("macro step terminal must be a boolean")
        if self.terminal and self.discount != 0.0:
            raise ValueError("terminal macro steps never bootstrap")
        if not self.surface or not self.branch:
            raise ValueError("macro step requires surface and branch labels")
        if self.control_domain not in {"macro", "combat"}:
            raise ValueError("macro step control_domain must be macro or combat")
        if not isinstance(self.recurrent_reset, bool):
            raise TypeError("macro step recurrent_reset must be a boolean")
        if not math.isfinite(self.behavior_epsilon) or not 0.0 <= self.behavior_epsilon <= 1.0:
            raise ValueError("macro step behavior_epsilon must be in [0, 1]")


@dataclass(frozen=True, slots=True)
class MacroEpisode:
    episode_id: str
    steps: tuple[MacroStep, ...]
    data_partition: str = "training"

    def __post_init__(self) -> None:
        if not self.episode_id:
            raise ValueError("macro episode requires an id")
        if not self.steps:
            raise ValueError("macro episode requires at least one step")
        if self.data_partition != "training":
            raise ValueError("macro replay stores training episodes only")
        for step in self.steps[:-1]:
            if step.terminal:
                raise ValueError("terminal macro step must be the final step")
        domains = {step.control_domain for step in self.steps}
        if len(domains) != 1:
            raise ValueError("one replay episode may contain only one control domain")

    @property
    def control_domain(self) -> Literal["macro", "combat"]:
        return self.steps[0].control_domain

    @property
    def executed_counts(self) -> dict[tuple[str, str], int]:
        counts: dict[tuple[str, str], int] = {}
        for step in self.steps:
            key = (step.surface, step.branch)
            counts[key] = counts.get(key, 0) + 1
        return counts


def n_step_targets(
    rewards: tuple[float, ...],
    discounts: tuple[float, ...],
    bootstrap_values: tuple[float, ...],
    *,
    n_step: int,
) -> tuple[float, ...]:
    """Exact n-step returns under the per-transition clock discounts.

    ``bootstrap_values[t]`` is the (Double-Q) value of the state reached by
    transition ``t`` — consumed at the truncation point.  A zero discount
    (terminal) cuts the chain naturally.
    """

    if not rewards or len(rewards) != len(discounts) or len(rewards) != len(bootstrap_values):
        raise ValueError("n-step target inputs are misaligned")
    if isinstance(n_step, bool) or not isinstance(n_step, int) or n_step <= 0:
        raise ValueError("n_step must be a positive integer")
    horizon = len(rewards)
    targets: list[float] = []
    for start in range(horizon):
        value = 0.0
        weight = 1.0
        step = start
        while True:
            value += weight * rewards[step]
            weight *= discounts[step]
            if weight == 0.0:
                break
            if step - start + 1 >= n_step or step + 1 >= horizon:
                value += weight * bootstrap_values[step]
                break
            step += 1
        targets.append(value)
    return tuple(targets)


def summarize_counts(episodes: tuple[MacroEpisode, ...]) -> dict[str, Any]:
    """EC-4 substrate: lifetime executed-sample counts per surface/branch."""

    counts: dict[tuple[str, str], int] = {}
    for episode in episodes:
        for key, value in episode.executed_counts.items():
            counts[key] = counts.get(key, 0) + value
    return {
        "version": MACRO_TRANSITION_CONTRACT_VERSION,
        "executed_counts": {
            f"{surface}:{branch}": count
            for (surface, branch), count in sorted(counts.items())
        },
    }


__all__ = [
    "MACRO_TRANSITION_CONTRACT_VERSION",
    "MacroEpisode",
    "MacroStep",
    "n_step_targets",
    "summarize_counts",
]
