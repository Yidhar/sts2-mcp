"""Route-heuristic root prior helpers.

This module owns STS2 map/route heuristic bias construction.  Keep it separate
from generic MCTS code and from ``muzero.train`` so route policy experiments can
be enabled, disabled, and audited independently.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from sts2_env.observation_v2 import MAX_ACTIONS


def compute_route_heuristic_bias_vector(
    *,
    raw_obs: dict[str, Any] | None,
    legal_actions: list[dict[str, Any]] | list[Any] | None,
    fallback_legal_actions: list[dict[str, Any]] | list[Any] | None,
    bias_weight: float,
) -> np.ndarray | None:
    """Return a MAX_ACTIONS route prior-bias vector, or ``None`` when gated off.

    The caller is responsible for deciding whether the current decision surface
    is a route/map decision.  This helper only scores the visible route actions
    and intentionally fails closed by returning ``None`` on missing observations
    or zero effective bias.
    """

    try:
        w = float(bias_weight)
    except (TypeError, ValueError):
        w = 0.0
    if w <= 0.0 or not isinstance(raw_obs, dict):
        return None

    try:
        from sts2_env.deck_quality import deck_quality_v2_from_obs
        from sts2_env.route_heuristic import count_non_empty_potions, rank_legal_route_actions

        full_legal = legal_actions if isinstance(legal_actions, list) and legal_actions else fallback_legal_actions
        if not isinstance(full_legal, list) or not full_legal:
            return None
        player = raw_obs.get("player") if isinstance(raw_obs.get("player"), dict) else {}
        run = raw_obs.get("run") if isinstance(raw_obs.get("run"), dict) else {}
        ranked = rank_legal_route_actions(
            legal_actions=full_legal,
            deck_quality=deck_quality_v2_from_obs(raw_obs),
            hp=float(player.get("hp") or 0.0),
            max_hp=float(player.get("max_hp") or 0.0),
            gold=float(player.get("gold") or 0.0),
            potion_count=count_non_empty_potions(player.get("potions")),
            floor=int(run.get("floor") or 0),
        )
        if not ranked:
            return None
        bias_vec = np.zeros(MAX_ACTIONS, dtype=np.float32)
        for i, breakdown in enumerate(ranked):
            if breakdown is None or i >= MAX_ACTIONS:
                continue
            if not breakdown.get("summary_used"):
                continue
            bias_vec[i] = float(w * breakdown.get("score_normalized", 0.0))
        if float(np.abs(bias_vec).max()) <= 1e-6:
            return None
        return bias_vec
    except Exception as exc:
        print(
            f"[route_heuristic] Phase 3 bias compute failed: "
            f"{type(exc).__name__}: {exc}",
            flush=True,
        )
        return None
