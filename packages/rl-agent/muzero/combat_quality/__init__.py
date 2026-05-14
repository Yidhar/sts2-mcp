"""Combat-action quality helpers for MuZero training.

Keep tactical/heuristic combat policy logic in this package instead of growing
``muzero.train``.  Modules here should stay pure or adapter-light so they can
be unit-tested without constructing the full trainer.
"""

from .action_bias import apply_card_block_waste_bias
from .block_waste import card_block_waste_profile
from .metrics import (
    COMBAT_QUALITY_CARD_BLOCK_SEARCH_SUFFIXES,
    COMBAT_QUALITY_GUARD_SEARCH_SUFFIXES,
    boss_card_block_waste_metrics,
)
from .guard_metrics import COMBAT_HARD_GUARD_DEFAULT_KEYS
from .potion_guard import (
    boss_race_potion_traits,
    boss_zero_energy_block_potion_escape,
    boss_zero_energy_liquid_escape,
    is_lucky_survival_potion_for_guard,
    lagavulin_setup_liquid_escape,
    lagavulin_setup_window_for_potion,
    potion_identity_text_for_guard,
    potion_slot_from_action_for_guard,
    raw_potion_payload_for_guard,
)

__all__ = [
    "COMBAT_HARD_GUARD_DEFAULT_KEYS",
    "COMBAT_QUALITY_CARD_BLOCK_SEARCH_SUFFIXES",
    "COMBAT_QUALITY_GUARD_SEARCH_SUFFIXES",
    "apply_card_block_waste_bias",
    "boss_card_block_waste_metrics",
    "boss_race_potion_traits",
    "boss_zero_energy_block_potion_escape",
    "boss_zero_energy_liquid_escape",
    "card_block_waste_profile",
    "is_lucky_survival_potion_for_guard",
    "lagavulin_setup_liquid_escape",
    "lagavulin_setup_window_for_potion",
    "potion_identity_text_for_guard",
    "potion_slot_from_action_for_guard",
    "raw_potion_payload_for_guard",
]
