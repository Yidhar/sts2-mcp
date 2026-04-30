"""Drift-gated direct-planner score composition (TASK-F1).

The search-free planner currently combines policy prior, immediate tactical
quality and Q-style estimates (rollout_q / objective_q / risk_q).  When the
world model is unreliable — high latent drift, high Q error, low legal
F1, high branch disagreement — the lookahead Q values can mislead action
selection.

This module exposes two pure helpers:

* :func:`compute_drift_gate` — bounded scalar in [min_gate, 1.0] derived from
  reliability signals.
* :func:`compose_planner_score` — composes the final per-action score from
  policy prior, immediate tactical quality, mechanism objective, the gated Q
  contributions, an uncertainty penalty and a hard legality guard.

The helpers are deliberately small and dependency-free; the trainer wires
them in once the score components are emitted by the policy.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any


def _sigmoid(x: float) -> float:
    if x >= 0:
        return 1.0 / (1.0 + math.exp(-x))
    z = math.exp(x)
    return z / (1.0 + z)


def _safe_float(value: Any, default: float = 0.0) -> float:
    if value is None:
        return default
    try:
        result = float(value)
    except (TypeError, ValueError):
        return default
    if math.isnan(result) or math.isinf(result):
        return default
    return result


@dataclass(frozen=True)
class DriftGateConfig:
    legal_f1_floor: float = 0.7
    drift_ceiling: float = 0.6
    q_error_ceiling: float = 1.0
    disagreement_ceiling: float = 0.5
    sigmoid_scale: float = 0.1
    min_gate: float = 0.05


def compute_drift_gate(
    *,
    legal_f1: float = 1.0,
    latent_drift: float = 0.0,
    q_mae: float = 0.0,
    branch_disagreement: float = 0.0,
    config: DriftGateConfig | None = None,
) -> float:
    """Return a bounded gate scalar.

    Each input is independently transformed through a sigmoid that saturates
    near 1 when the metric is on the "safe" side of its threshold, and
    shrinks the gate when the metric is on the "unsafe" side.  The four
    factors multiply and the final result is clamped to [min_gate, 1.0].
    """
    cfg = config or DriftGateConfig()
    scale = max(cfg.sigmoid_scale, 1e-3)
    f1 = _safe_float(legal_f1, default=1.0)
    drift = _safe_float(latent_drift, default=0.0)
    err = _safe_float(q_mae, default=0.0)
    disagree = _safe_float(branch_disagreement, default=0.0)

    factor_legal = _sigmoid((f1 - cfg.legal_f1_floor) / scale)
    factor_drift = _sigmoid((cfg.drift_ceiling - drift) / scale)
    factor_qerr = _sigmoid((cfg.q_error_ceiling - err) / scale)
    factor_dis = _sigmoid((cfg.disagreement_ceiling - disagree) / scale)

    gate = factor_legal * factor_drift * factor_qerr * factor_dis
    if gate < cfg.min_gate:
        return cfg.min_gate
    if gate > 1.0:
        return 1.0
    return gate


@dataclass(frozen=True)
class PlannerScoreWeights:
    policy_prior: float = 1.0
    immediate_tactical: float = 1.0
    mechanism: float = 0.5
    rollout_q: float = 1.0
    objective_q: float = 1.0
    risk_q: float = 1.0
    uncertainty: float = 0.5


def compose_planner_score(
    *,
    policy_prior: float,
    immediate_tactical_quality: float,
    mechanism_objective: float = 0.0,
    rollout_q: float = 0.0,
    objective_q: float = 0.0,
    risk_q: float = 0.0,
    uncertainty_penalty: float = 0.0,
    legality_guard: float = 0.0,
    drift_gate: float = 1.0,
    weights: PlannerScoreWeights | None = None,
) -> dict[str, float]:
    """Compose the per-action planner score and per-component breakdown.

    Returns a dict with the final ``score`` plus the component contributions
    (multiplied by their effective weights) so the trainer can log
    ``planner/action_score_component/<component>`` metrics directly.
    """
    w = weights or PlannerScoreWeights()
    gate = max(0.0, min(1.0, _safe_float(drift_gate, default=1.0)))

    components: dict[str, float] = {
        "policy_prior": w.policy_prior * _safe_float(policy_prior),
        "immediate_tactical": w.immediate_tactical * _safe_float(immediate_tactical_quality),
        "mechanism": w.mechanism * _safe_float(mechanism_objective),
        "rollout_q": gate * w.rollout_q * _safe_float(rollout_q),
        "objective_q": gate * w.objective_q * _safe_float(objective_q),
        "risk_q": -gate * w.risk_q * _safe_float(risk_q),
        "uncertainty": -w.uncertainty * _safe_float(uncertainty_penalty),
        "legality_guard": _safe_float(legality_guard),
    }
    score = sum(components.values())
    components["score"] = score
    components["effective_rollout_q_weight"] = gate * w.rollout_q
    components["effective_objective_q_weight"] = gate * w.objective_q
    components["effective_risk_q_weight"] = gate * w.risk_q
    components["drift_gate"] = gate
    return components
