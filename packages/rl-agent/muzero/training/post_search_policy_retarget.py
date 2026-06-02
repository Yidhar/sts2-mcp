from __future__ import annotations

from typing import Any

import numpy as np

from sts2_env.observation_v2 import MAX_ACTIONS


POST_SEARCH_HARD_GUARD_SEARCH_SUFFIXES: dict[str, str] = {
    "post_search_hard_guard_policy_retargeted": "post_search_hard_guard_policy_retargeted_rate",
    "post_search_hard_guard_original_action_idx": "post_search_hard_guard_original_action_idx_mean",
    "post_search_hard_guard_final_action_idx": "post_search_hard_guard_final_action_idx_mean",
}


def final_action_is_skip(action: Any) -> bool:
    """Return whether the action actually executed is a skip-style action.

    ``surface=card_reward`` by itself is *not* a skip; the final action must
    explicitly carry a skip selection/kind/id/label.  This keeps card-reward
    diagnostics from confusing "the model originally chose skip" with "the
    guard still executed skip".
    """

    if not isinstance(action, dict):
        return False
    semantic = action.get("semantic")
    semantic_family = ""
    if isinstance(semantic, dict):
        semantic_family = str(semantic.get("family") or "").strip().lower()
    fields = (
        str(action.get("selection") or "").strip().lower(),
        str(action.get("kind") or "").strip().lower(),
        str(action.get("action_id") or "").strip().lower(),
        str(action.get("label") or "").strip().lower(),
        semantic_family,
    )
    return any(field == "skip" or "skip" in field for field in fields if field)


def compact_final_action_for_diagnostic(action: Any) -> dict[str, Any]:
    if not isinstance(action, dict):
        return {}
    final_action: dict[str, Any] = {
        "kind": action.get("kind"),
        "surface": action.get("surface"),
        "selection": action.get("selection"),
        "action_id": action.get("action_id"),
        "label": action.get("label"),
    }
    card = action.get("card") if isinstance(action.get("card"), dict) else {}
    if card:
        final_action["card"] = {
            "id": card.get("id") or card.get("card_id"),
            "title": card.get("title") or card.get("name"),
            "type": card.get("type") or card.get("card_type"),
            "cost": card.get("cost") if card.get("cost") is not None else card.get("energy_cost"),
        }
    return {key: value for key, value in final_action.items() if value not in (None, "")}


def annotate_card_reward_final_selection(
    payload: dict[str, Any],
    *,
    chosen_action: Any,
    final_action_idx: int,
    final_selected_family: str,
    phase: str,
    decision_domain: str,
    policy_retargeted: bool | None = None,
) -> dict[str, Any]:
    """Mutate and return a card-reward JSONL payload with final-action semantics."""

    original_is_skip = bool(payload.get("original_selected_is_skip", payload.get("selected_is_skip", False)))
    final_is_skip = final_action_is_skip(chosen_action)
    payload.setdefault("original_selected_is_skip", original_is_skip)
    payload["final_action_idx"] = int(final_action_idx)
    payload["final_action"] = compact_final_action_for_diagnostic(chosen_action)
    payload["final_selected_is_skip"] = bool(final_is_skip)
    # Backward-compatible field, but from now on it means the final executed
    # action.  Use original_selected_is_skip for the pre-guard model choice.
    payload["selected_is_skip"] = bool(final_is_skip)
    payload.setdefault("final_selected_family", final_selected_family)
    payload.setdefault("phase", phase)
    payload.setdefault("decision_domain", decision_domain)
    if policy_retargeted is not None:
        payload["policy_retargeted"] = bool(policy_retargeted)
    return payload


def retarget_search_policy_after_hard_guard(
    search_policy: Any,
    *,
    original_action_idx: int,
    final_action_idx: int,
    max_actions: int = MAX_ACTIONS,
) -> tuple[np.ndarray, bool]:
    """Rewrite policy target to the post-guard action when a hard guard overrides.

    Hard guards are authoritative safety/build corrections.  If the environment
    executes the guarded action but replay keeps the pre-guard search
    distribution, policy training can continue to reinforce the rejected action
    (for example card-reward skip).  On override we therefore make the target a
    one-hot distribution over the final action.
    """

    try:
        policy = np.asarray(search_policy, dtype=np.float32).reshape(-1).copy()
    except Exception:
        policy = np.zeros(int(max_actions), dtype=np.float32)

    try:
        original_idx = int(original_action_idx)
        final_idx = int(final_action_idx)
    except Exception:
        return policy, False

    if original_idx == final_idx:
        return policy, False

    limit = policy.shape[0]
    if max_actions > 0:
        limit = min(limit, int(max_actions))
    if not (0 <= final_idx < limit):
        return policy, False

    rewritten = np.zeros_like(policy, dtype=np.float32)
    rewritten[final_idx] = 1.0
    return rewritten, True


__all__ = [
    "POST_SEARCH_HARD_GUARD_SEARCH_SUFFIXES",
    "annotate_card_reward_final_selection",
    "compact_final_action_for_diagnostic",
    "final_action_is_skip",
    "retarget_search_policy_after_hard_guard",
]
