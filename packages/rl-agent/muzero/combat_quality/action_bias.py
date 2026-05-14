"""Root-prior bias helpers for combat-action quality.

Keep tactical bias arithmetic outside ``muzero.train``.  The trainer should
build typed index sets from the bridge observation, then call small pure
helpers here to mutate a numpy bias vector and return metric-friendly stats.
"""

from __future__ import annotations

from collections.abc import Iterable
from typing import Any

import numpy as np


DEFAULT_CARD_BLOCK_WASTE_SOFT_DELTA = -1.10
DEFAULT_CARD_BLOCK_WASTE_HARD_DELTA = -2.75
DEFAULT_CARD_BLOCK_PROGRESS_BONUS = 0.20


def _valid_indices(
    indices: Iterable[Any],
    *,
    action_mask: np.ndarray,
    max_actions: int,
) -> set[int]:
    """Return legal integer indices that are inside both MAX_ACTIONS and mask."""

    valid: set[int] = set()
    for raw_idx in indices or ():
        try:
            idx = int(raw_idx)
        except (TypeError, ValueError):
            continue
        if 0 <= idx < int(max_actions) and idx < int(action_mask.shape[0]) and float(action_mask[idx]) > 0.0:
            valid.add(idx)
    return valid


def apply_card_block_waste_bias(
    bias: np.ndarray,
    *,
    action_mask: np.ndarray,
    card_block_waste_indices: Iterable[Any],
    positive_indices: Iterable[Any] = (),
    urgent_positive_indices: Iterable[Any] = (),
    end_turn_indices: Iterable[Any] = (),
    max_actions: int,
    soft_delta: float = DEFAULT_CARD_BLOCK_WASTE_SOFT_DELTA,
    hard_delta: float = DEFAULT_CARD_BLOCK_WASTE_HARD_DELTA,
    progress_bonus: float = DEFAULT_CARD_BLOCK_PROGRESS_BONUS,
) -> dict[str, float]:
    """Suppress no-pressure pure-block cards, especially when progress exists.

    A Defend-like card can be legal while enemy incoming damage is already
    covered.  In that case spending energy on pure block usually increases
    attrition by not attacking/setup/drawing.  The old ``-1.10`` penalty was
    sometimes weaker than other immediate-impact priors, so when there is at
    least one non-waste progress action available this helper applies a harder
    penalty and gives the progress alternatives a small tie-break bonus.

    If the only legal play is a pure-block card or End Turn, the helper keeps
    the softer legacy penalty.  That avoids teaching the agent that End Turn is
    always better than using a harmless block card when no progress alternative
    exists.
    """

    if bias.ndim != 1:
        raise ValueError("bias must be a 1-D array")
    if action_mask.ndim != 1:
        raise ValueError("action_mask must be a 1-D array")

    waste = _valid_indices(card_block_waste_indices, action_mask=action_mask, max_actions=max_actions)
    positives = _valid_indices(positive_indices, action_mask=action_mask, max_actions=max_actions)
    urgents = _valid_indices(urgent_positive_indices, action_mask=action_mask, max_actions=max_actions)
    end_turns = _valid_indices(end_turn_indices, action_mask=action_mask, max_actions=max_actions)

    progress = (positives | urgents) - waste - end_turns
    has_progress_alternative = bool(progress)
    delta = float(hard_delta if has_progress_alternative else soft_delta)

    bias_count = 0
    bias_min = 0.0
    for idx in sorted(waste):
        bias[idx] += delta
        bias_count += 1
        bias_min = min(bias_min, delta)

    progress_bonus_count = 0
    progress_bonus_max = 0.0
    bonus = float(progress_bonus if has_progress_alternative and waste else 0.0)
    if bonus > 0.0:
        for idx in sorted(progress):
            bias[idx] += bonus
            progress_bonus_count += 1
            progress_bonus_max = max(progress_bonus_max, bonus)

    return {
        "card_block_waste_bias_count": float(bias_count),
        "card_block_waste_bias_min": float(bias_min),
        "card_block_waste_hard_bias_applied": 1.0 if bias_count > 0 and has_progress_alternative else 0.0,
        "card_block_waste_progress_alternative": 1.0 if has_progress_alternative else 0.0,
        "card_block_waste_progress_bonus_count": float(progress_bonus_count),
        "card_block_waste_progress_bonus_max": float(progress_bonus_max),
    }


__all__ = [
    "DEFAULT_CARD_BLOCK_PROGRESS_BONUS",
    "DEFAULT_CARD_BLOCK_WASTE_HARD_DELTA",
    "DEFAULT_CARD_BLOCK_WASTE_SOFT_DELTA",
    "apply_card_block_waste_bias",
]
