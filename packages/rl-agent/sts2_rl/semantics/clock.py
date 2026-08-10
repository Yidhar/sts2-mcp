"""Durable-floor decision clock for the semantic decision graph.

Native API calls are not a valid time unit: a longer confirmation sequence
must not make forging look worse than resting, and thousands of card plays
must not erase the value of an Act-1 deck decision.  The control-learning
discount therefore advances only with durable floor progress:

``Gamma_t = 0``                                  on terminal transitions
``Gamma_t = base ** max(floor_next - floor_now, 0)``  otherwise

Consequently mechanical executor steps create no control transitions,
decisions within one combat or room bootstrap at ``Gamma = 1``, one durable
floor applies the discount exactly once regardless of how many native calls
were required, and terminal states never bootstrap.  Ordinary pace reward and
stall termination own wasted same-floor actions; the clock is not a loop
detector.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Final

DECISION_CLOCK_CONTRACT_VERSION: Final = "sts2-durable-floor-clock-v1"

DECISION_CLOCK_BASE: Final = 0.997
"""Per-floor discount. Over a complete 60-floor run the first decision still
sees ``0.997 ** 59 ~= 0.84`` of terminal credit, so outcome-first ordering
survives the clock (verified in the stage-one objective arithmetic note)."""


@dataclass(frozen=True, slots=True)
class DecisionClockTick:
    """Discount evidence for one semantic transition."""

    floor_before: int
    floor_after: int
    terminal: bool
    discount: float
    version: str = DECISION_CLOCK_CONTRACT_VERSION

    def __post_init__(self) -> None:
        for label, value in (
            ("floor_before", self.floor_before),
            ("floor_after", self.floor_after),
        ):
            if isinstance(value, bool) or not isinstance(value, int) or value < 0:
                raise ValueError(f"decision clock {label} must be a non-negative integer")
        if not isinstance(self.terminal, bool):
            raise TypeError("decision clock terminal must be a boolean")
        expected = decision_discount(
            floor_before=self.floor_before,
            floor_after=self.floor_after,
            terminal=self.terminal,
        )
        if not math.isclose(self.discount, expected, rel_tol=0.0, abs_tol=1e-12):
            raise ValueError(
                "decision clock discount does not match the durable-floor contract"
            )


def decision_discount(
    *,
    floor_before: int,
    floor_after: int,
    terminal: bool,
    base: float = DECISION_CLOCK_BASE,
) -> float:
    """Return the control-learning discount for one semantic transition.

    ``floor_after < floor_before`` never occurs for durable floors; the clamp
    keeps a malformed regression from manufacturing ``Gamma > 1``.
    """

    if isinstance(floor_before, bool) or not isinstance(floor_before, int):
        raise TypeError("floor_before must be an integer")
    if isinstance(floor_after, bool) or not isinstance(floor_after, int):
        raise TypeError("floor_after must be an integer")
    if not isinstance(terminal, bool):
        raise TypeError("terminal must be a boolean")
    if not 0.0 < base <= 1.0 or not math.isfinite(base):
        raise ValueError("decision clock base must be in (0, 1]")
    if terminal:
        return 0.0
    return float(base ** max(floor_after - floor_before, 0))


def clock_tick(
    *,
    floor_before: int,
    floor_after: int,
    terminal: bool,
) -> DecisionClockTick:
    return DecisionClockTick(
        floor_before=floor_before,
        floor_after=floor_after,
        terminal=terminal,
        discount=decision_discount(
            floor_before=floor_before,
            floor_after=floor_after,
            terminal=terminal,
        ),
    )


__all__ = [
    "DECISION_CLOCK_BASE",
    "DECISION_CLOCK_CONTRACT_VERSION",
    "DecisionClockTick",
    "clock_tick",
    "decision_discount",
]
