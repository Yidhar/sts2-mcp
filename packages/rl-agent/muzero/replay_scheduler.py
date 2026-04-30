"""Encounter-balanced replay scheduler (TASK-F2).

The replay buffer ships transitions tagged with a *tier* (``boss`` /
``elite`` / ``normal`` / ``weak``) and an ``encounter_id``.  When the boss
sample rate is allowed to climb unboundedly, basic attack/defense skill
regresses; when boss is starved, boss skill regresses.  This module
computes per-sample weights that:

1. Enforce a target tier distribution (boss 50-60%, elite 20-30%, normal
   10-20%, weak optional).
2. Balance encounters inside each tier so no individual boss is starved
   even when the global boss rate is satisfied.
3. Combine encounter-underrepresented / offender / demo / freshness
   biases with a min/max clamp so no single signal can dominate.

The scheduler is a *pure helper* — it consumes counts and per-sample
metadata and returns sampling weights, with no buffer-side mutation.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, Iterable


_DEFAULT_TIER_TARGETS: dict[str, tuple[float, float]] = {
    "boss": (0.50, 0.60),
    "elite": (0.20, 0.30),
    "normal": (0.10, 0.20),
    "weak": (0.0, 0.10),
}


@dataclass(frozen=True)
class ReplaySchedulerConfig:
    tier_targets: dict[str, tuple[float, float]] = field(
        default_factory=lambda: dict(_DEFAULT_TIER_TARGETS)
    )
    min_weight: float = 0.05
    max_weight: float = 6.0
    offender_max_boost: float = 2.0
    demo_max_boost: float = 1.5
    freshness_half_life: float = 50_000.0
    encounter_underrepresented_max_boost: float = 2.5


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


def _normalize_tier(tier: Any) -> str:
    text = str(tier or "").strip().lower()
    return text if text in _DEFAULT_TIER_TARGETS else "normal"


def _tier_target_midpoint(cfg: ReplaySchedulerConfig, tier: str) -> tuple[float, float, float]:
    lo, hi = cfg.tier_targets.get(tier, (0.0, 0.5))
    mid = (lo + hi) / 2.0
    return lo, hi, mid


def compute_tier_weights(
    counts_per_tier: dict[str, int],
    *,
    config: ReplaySchedulerConfig | None = None,
) -> dict[str, float]:
    """Tier-level multiplicative weights to push the empirical distribution
    toward the configured target.

    A tier whose share is *below* its target lower bound gets weight > 1;
    one *above* the upper bound gets weight < 1; one inside the band gets
    weight ≈ 1.  The weight per tier is clamped to ``[min_weight, max_weight]``.
    """
    cfg = config or ReplaySchedulerConfig()
    total = max(1, sum(int(v) for v in counts_per_tier.values()))
    weights: dict[str, float] = {}
    for tier, count in counts_per_tier.items():
        norm = _normalize_tier(tier)
        share = count / total
        lo, hi, mid = _tier_target_midpoint(cfg, norm)
        if share <= 0:
            ratio = mid * total / 1.0  # fully starved tier — strong boost
        else:
            ratio = mid / share
        weight = max(cfg.min_weight, min(cfg.max_weight, ratio))
        # Soft caps so we never invert ordering inside the band.
        if lo <= share <= hi:
            weight = min(weight, 1.5)
            weight = max(weight, 0.5)
        weights[tier] = weight
    return weights


def compute_encounter_underrepresented_weight(
    encounter_count: int,
    encounter_target: int,
    *,
    config: ReplaySchedulerConfig | None = None,
) -> float:
    """Boost factor for an encounter whose count is below ``encounter_target``."""
    cfg = config or ReplaySchedulerConfig()
    if encounter_target <= 0:
        return 1.0
    deficit = max(0, encounter_target - max(0, int(encounter_count)))
    if deficit == 0:
        return 1.0
    ratio = 1.0 + deficit / max(1.0, float(encounter_target))
    return min(cfg.encounter_underrepresented_max_boost, ratio)


def compute_freshness_weight(
    sample_age_steps: int,
    *,
    config: ReplaySchedulerConfig | None = None,
) -> float:
    """Half-life decay on sample age — fresher samples get higher weight."""
    cfg = config or ReplaySchedulerConfig()
    age = max(0.0, _safe_float(sample_age_steps))
    half = max(1.0, cfg.freshness_half_life)
    return float(0.5 + 0.5 * math.exp(-age * math.log(2.0) / half))


def compute_offender_weight(
    offender_score: float,
    *,
    config: ReplaySchedulerConfig | None = None,
) -> float:
    """Bounded boost for transitions with offenders detected.

    ``offender_score`` is expected in [0, 1]; we map to ``[1, max_boost]``.
    """
    cfg = config or ReplaySchedulerConfig()
    s = max(0.0, min(1.0, _safe_float(offender_score)))
    return 1.0 + s * (cfg.offender_max_boost - 1.0)


def compute_demo_weight(
    is_demo: bool,
    *,
    config: ReplaySchedulerConfig | None = None,
) -> float:
    cfg = config or ReplaySchedulerConfig()
    return cfg.demo_max_boost if is_demo else 1.0


@dataclass
class ReplaySample:
    tier: str
    encounter_id: str
    sample_age_steps: int = 0
    offender_score: float = 0.0
    is_demo: bool = False


def compute_sample_weights(
    samples: Iterable[ReplaySample],
    *,
    counts_per_tier: dict[str, int],
    counts_per_encounter: dict[str, int],
    encounter_target: int = 0,
    config: ReplaySchedulerConfig | None = None,
) -> list[float]:
    """Return one weight per input sample, clamped to ``[min_weight, max_weight]``.

    The weight is the product of the tier weight, encounter
    underrepresented boost, offender boost, demo boost and freshness
    decay.
    """
    cfg = config or ReplaySchedulerConfig()
    tier_weights = compute_tier_weights(counts_per_tier, config=cfg)
    out: list[float] = []
    for sample in samples:
        tier = _normalize_tier(sample.tier)
        tier_w = tier_weights.get(sample.tier, tier_weights.get(tier, 1.0))
        enc_w = compute_encounter_underrepresented_weight(
            counts_per_encounter.get(sample.encounter_id, 0),
            encounter_target,
            config=cfg,
        )
        offender_w = compute_offender_weight(sample.offender_score, config=cfg)
        demo_w = compute_demo_weight(sample.is_demo, config=cfg)
        fresh_w = compute_freshness_weight(sample.sample_age_steps, config=cfg)
        weight = tier_w * enc_w * offender_w * demo_w * fresh_w
        weight = max(cfg.min_weight, min(cfg.max_weight, weight))
        out.append(float(weight))
    return out
