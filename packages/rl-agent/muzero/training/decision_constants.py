"""Small shared constants for rollout decision bookkeeping.

Keep these values out of ``muzero.train`` so self-play, route/build guards, and
telemetry can share the same vocabulary without creating circular imports.
"""

from __future__ import annotations

BUILD_ROUTE_SETTLEMENT_FAMILIES = frozenset({
    "map",
    "reward",
    "card_reward",
    "shop",
    "rest",
    "smith",
    "deck_upgrade",
    "event_option",
})

WASTEFUL_REWARD_PHASES = frozenset({"reward", "card_reward"})

TRIVIAL_BUILD_FAST_PATH_REASONS = (
    "reward_gold",
    "reward_potion",
    "proceed_only",
    "startup_only",
)

TRIVIAL_BUILD_FAST_PATH_COMPLEX_FAMILIES = frozenset({
    "map",
    "card_reward",
    "shop",
    "rest",
    "smith",
    "deck_upgrade",
    "event_option",
    "treasure_relic",
})
