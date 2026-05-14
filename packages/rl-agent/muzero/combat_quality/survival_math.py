"""Small survival math helpers shared by combat-quality guards.

The tactical guards often reason about a *net* incoming threat gap:

    threat_gap = max(0, incoming_damage - current_block)

When evaluating a candidate block/heal card, the important question is not
whether it produces "some" protection, but whether the resulting protection is
enough to leave the player alive after the enemy hit.  In Slay-the-Spire style
combat, taking damage equal to current HP is death, so survival is a strict
``hp_after > remaining_gap`` check.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class ProtectionOutcome:
    """Result of applying one candidate protection action to a threat gap."""

    hp_after_cost_and_heal: float
    remaining_gap: float
    survives: bool
    insufficient: bool


def protection_outcome(
    *,
    hp: float,
    threat_gap: float,
    block: float = 0.0,
    heal: float = 0.0,
    hp_cost: float = 0.0,
) -> ProtectionOutcome:
    """Return whether a protection action is sufficient to survive.

    Args:
        hp: Current player HP before paying candidate HP cost.
        threat_gap: Net incoming damage after already-existing block.
        block: Additional block provided by the candidate action.
        heal: Healing provided by the candidate action.
        hp_cost: HP paid by the candidate action before the enemy hit.

    ``survives`` is strict: if HP after cost/heal equals remaining damage, the
    player dies.  ``insufficient`` is true only for positive remaining damage
    that still kills the player; fully covering the threat is never
    insufficient.
    """

    hp_after = max(0.0, float(hp) - max(0.0, float(hp_cost))) + max(0.0, float(heal))
    remaining = max(0.0, float(threat_gap) - max(0.0, float(block)))
    survives = bool(hp_after > remaining)
    insufficient = bool(remaining > 0.0 and not survives)
    return ProtectionOutcome(
        hp_after_cost_and_heal=float(hp_after),
        remaining_gap=float(remaining),
        survives=survives,
        insufficient=insufficient,
    )


__all__ = ["ProtectionOutcome", "protection_outcome"]
