"""Encounter-specific strategy modules.

Examples planned for extraction from ``muzero.train``:

* ``kaiser`` — facing/back-attack pressure and safe facing-change actions
* ``insatiable`` — sandpit countdown and Frantic Escape urgency
* ``boss_potions`` — boss-mechanic potion timing / race logic

Keep these modules focused and below the 2,000-line budget.
"""

from . import insatiable, kaiser

__all__ = ["insatiable", "kaiser"]
