"""High-HP campfire smith guard for Act1 full-run recovery.

Low-HP campfires are already protected by the environment/trainer HEAL guard:
when HP is unsafe, non-heal rest-site options are masked or overridden.  The
opposite failure mode showed up in full-run monitoring after the build/shop
fixes: the agent reaches campfires, but ``env/rest_smith_chosen`` remains zero
and final/death decks have almost no upgraded cards.  This module owns the
small *safe HP* counter-guard:

* if the agent is already on a concrete rest-site choice surface;
* HP is comfortably above the low-HP heal threshold;
* the selected action is HEAL/REST;
* SMITH/UPGRADE is legal;

then choose SMITH instead of wasting a campfire on extra HP.  It is deliberately
not a route heuristic and it does not choose the upgrade target; it only fixes
the within-campfire HEAL-vs-SMITH decision when healing is unlikely to be the
right Act1 long-horizon choice.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from muzero.diagnostics.rest_site_metrics import (
    hp_ratio_from_obs,
    is_rest_heal_action,
    is_rest_site_action,
    is_rest_smith_action,
)
from sts2_env.observation_v2 import MAX_ACTIONS
from sts2_env.reward_constants import REST_SITE_SKIP_HEAL_HP_THRESHOLD


# Keep a gap above the low-HP HEAL threshold so this guard never fights the
# survival guard around the ambiguous 60-70% HP zone.
SAFE_SMITH_HP_THRESHOLD = max(float(REST_SITE_SKIP_HEAL_HP_THRESHOLD) + 0.10, 0.72)


REST_SITE_SMITH_GUARD_SEARCH_SUFFIXES: dict[str, str] = {
    "rest_site_smith_guard_enabled": "rest_site_smith_guard_enabled",
    "rest_site_smith_guard_applicable": "rest_site_smith_guard_applicable_rate",
    "rest_site_smith_guard_smith_available": "rest_site_smith_guard_smith_available_rate",
    "rest_site_smith_guard_selected_heal_safe_hp": "rest_site_smith_guard_selected_heal_safe_hp_rate",
    "rest_site_smith_guard_applied": "rest_site_smith_guard_applied_rate",
    "rest_site_smith_guard_override": "rest_site_smith_guard_override_rate",
    "rest_site_smith_guard_invalid_obs": "rest_site_smith_guard_invalid_obs_rate",
    "rest_site_smith_guard_alignment_error": "rest_site_smith_guard_alignment_error_rate",
    "rest_site_smith_guard_hp_ratio": "rest_site_smith_guard_hp_ratio_mean",
    "rest_site_smith_guard_hp_threshold": "rest_site_smith_guard_hp_threshold",
}


def rest_site_smith_guard_metric_keys() -> tuple[str, ...]:
    return tuple(REST_SITE_SMITH_GUARD_SEARCH_SUFFIXES)


def _set_default_metrics(search_stats: dict[str, Any]) -> None:
    for key in rest_site_smith_guard_metric_keys():
        search_stats.setdefault(key, 0.0)
    search_stats["rest_site_smith_guard_enabled"] = 1.0
    search_stats["rest_site_smith_guard_hp_threshold"] = float(SAFE_SMITH_HP_THRESHOLD)


def _legal_count(legal_actions: list[Any], action_mask: Any) -> tuple[int, np.ndarray] | None:
    try:
        mask_np = np.asarray(action_mask, dtype=np.float32).reshape(-1)
    except Exception:
        return None
    legal_count = min(len(legal_actions), int(mask_np.shape[0]) if mask_np.size else 0, MAX_ACTIONS)
    return legal_count, mask_np


def apply_rest_site_smith_guard(
    *,
    action_idx: int,
    legal_actions: list[Any] | None,
    full_legal_actions: list[Any] | None,
    action_mask: Any,
    raw_obs: dict[str, Any] | None,
    search_stats: dict[str, Any],
    safe_hp_threshold: float = SAFE_SMITH_HP_THRESHOLD,
) -> int:
    """Override wasteful high-HP HEAL into SMITH when SMITH is legal.

    Returns the original ``action_idx`` unless all safety/alignment checks pass.
    ``legal_actions`` may be compact actions while ``full_legal_actions`` keeps
    the live bridge option payload; if full actions are available, they are used
    as the authoritative source for rest-site option identity.
    """

    _set_default_metrics(search_stats)
    if not isinstance(legal_actions, list) or len(legal_actions) == 0:
        return int(action_idx)
    count_and_mask = _legal_count(legal_actions, action_mask)
    if count_and_mask is None:
        return int(action_idx)
    legal_count, mask_np = count_and_mask
    action_idx = int(action_idx)
    if not (0 <= action_idx < legal_count) or mask_np[action_idx] <= 0:
        return action_idx

    hp_ratio, hp_valid = hp_ratio_from_obs(raw_obs)
    if not hp_valid:
        search_stats["rest_site_smith_guard_invalid_obs"] = 1.0
        return action_idx
    search_stats["rest_site_smith_guard_hp_ratio"] = float(hp_ratio)
    if hp_ratio < float(safe_hp_threshold):
        return action_idx

    action_source = full_legal_actions if isinstance(full_legal_actions, list) else legal_actions
    if not isinstance(action_source, list) or len(action_source) < legal_count:
        search_stats["rest_site_smith_guard_alignment_error"] = 1.0
        return action_idx

    selected_action = action_source[action_idx]
    if not is_rest_site_action(selected_action):
        return action_idx
    search_stats["rest_site_smith_guard_applicable"] = 1.0
    if not is_rest_heal_action(selected_action):
        return action_idx
    search_stats["rest_site_smith_guard_selected_heal_safe_hp"] = 1.0

    smith_indices: list[int] = []
    for idx in range(legal_count):
        if mask_np[idx] <= 0:
            continue
        candidate = action_source[idx]
        if is_rest_site_action(candidate) and is_rest_smith_action(candidate):
            smith_indices.append(int(idx))
    if not smith_indices:
        return action_idx
    search_stats["rest_site_smith_guard_smith_available"] = 1.0

    override_idx = int(smith_indices[0])
    if not (0 <= override_idx < legal_count) or mask_np[override_idx] <= 0:
        search_stats["rest_site_smith_guard_alignment_error"] = 1.0
        return action_idx

    search_stats["rest_site_smith_guard_applied"] = 1.0
    if override_idx != action_idx:
        search_stats["rest_site_smith_guard_override"] = 1.0
    return override_idx
