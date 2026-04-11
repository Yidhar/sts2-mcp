"""Bind semantic plan actions back to live legal actions."""

from __future__ import annotations

from typing import Any

from .semantic_action import semantic_action_signature, semantic_match_score


def bind_semantic_action(
    semantic_plan: dict[str, Any] | None,
    legal_actions: list[Any] | None,
) -> tuple[int | None, dict[str, Any] | None, float]:
    """Return the best-matching live legal action for a semantic plan.

    Returns:
        (best_index, best_action, score)
    """
    if not isinstance(legal_actions, list) or not legal_actions:
        return None, None, float("-inf")

    best_index: int | None = None
    best_action: dict[str, Any] | None = None
    best_score = float("-inf")
    for index, action in enumerate(legal_actions):
        if not isinstance(action, dict):
            continue
        score = semantic_match_score(semantic_plan, semantic_action_signature(action))
        if score > best_score:
            best_index = index
            best_action = action
            best_score = score
    return best_index, best_action, best_score

