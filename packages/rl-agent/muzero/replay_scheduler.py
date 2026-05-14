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


# ---------------------------------------------------------------------------
# P0-2 (recovery 2026-05-06): batch-level hard tier quotas.
#
# Soft tier weights nudge the empirical distribution but cannot defend against
# extreme skew when high-priority encounters dominate priority * encounter
# weight products (in the latest run boss climbed to 95%+ while normal fell
# below 1%). The recovery doc requires a *hard* per-batch quota: every batch
# must contain at least N_normal normal trajectories, at least N_elite elite
# trajectories, and at most N_boss_max boss trajectories. These quotas live
# next to the existing soft scheduler so callers can opt in without touching
# the priority computation.
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class TierQuotaConfig:
    """Hard per-batch tier quotas for the replay sampler.

    ``targets`` are the desired share per tier (must sum to 1.0 over the
    tiers actually present in the buffer; tiers absent from the dict are
    treated as ``flex`` and only sampled when other tiers are exhausted).

    ``min_caps`` is the minimum share per tier — the sampler will refuse to
    drop below this when the tier pool is non-empty.

    ``max_caps`` is the maximum share per tier — the sampler will refuse to
    exceed this even if the tier pool dominates the priority budget.
    """

    targets: dict[str, float] = field(
        default_factory=lambda: {"boss": 0.60, "elite": 0.25, "normal": 0.15, "weak": 0.0}
    )
    min_caps: dict[str, float] = field(
        default_factory=lambda: {"elite": 0.18, "normal": 0.10}
    )
    max_caps: dict[str, float] = field(
        default_factory=lambda: {"boss": 0.65}
    )


@dataclass(frozen=True)
class TierQuotaAllocation:
    quotas: dict[str, int]
    target_counts: dict[str, int]
    boss_cap_hit: bool
    normal_min_unfilled: bool
    elite_min_unfilled: bool


def allocate_tier_quotas(
    *,
    batch_size: int,
    pool_sizes: dict[str, int],
    config: TierQuotaConfig | None = None,
) -> TierQuotaAllocation:
    """Pure function: turn ``batch_size`` and per-tier pool sizes into a
    per-tier integer quota.

    Algorithm (priority order, mirrors §P0-2 of the recovery doc):

    1. **Floors first** — each tier with a ``min_caps`` entry whose pool is
       non-empty receives ``ceil(min_share * batch_size)`` slots up front,
       capped by the pool size and by ``max_caps`` if applicable.
    2. **Caps as ceilings** — every tier with a ``max_caps`` entry can never
       exceed ``floor(max_share * batch_size)``.
    3. **Remaining slots** are distributed proportionally to ``targets`` for
       tiers below their max cap and with non-empty pools. Any leftover
       (because all eligible tiers hit their max cap, or pools are empty)
       becomes ``flex`` slots that fall back to the global priority pool.
    4. **Pool clamps** — finally, any per-tier quota that exceeds the
       actual pool size is clamped down and the shortfall is added to
       ``flex``.

    Slots that cannot be filled by any tier become ``flex`` and the caller
    falls back to the global priority pool. The allocation is deterministic
    given the inputs so it is easy to unit-test and render in TensorBoard.
    """
    cfg = config or TierQuotaConfig()
    bs = max(0, int(batch_size))
    if bs == 0:
        return TierQuotaAllocation(
            quotas={},
            target_counts={},
            boss_cap_hit=False,
            normal_min_unfilled=False,
            elite_min_unfilled=False,
        )

    # Always carry weak in the quota dict so callers can rely on a stable
    # tier-key vocabulary in their TensorBoard exports.
    tier_keys = list(dict.fromkeys(["boss", "elite", "normal", "weak", *cfg.targets.keys()]))
    quotas: dict[str, int] = {t: 0 for t in tier_keys}

    # Pre-compute integer caps & target counts (used both for proportional
    # distribution and by the caller for KL/L1 diagnostics).
    max_caps_int: dict[str, int] = {
        t: int(math.floor(max(0.0, float(s)) * bs)) for t, s in cfg.max_caps.items()
    }
    raw_target_counts: dict[str, int] = {
        t: int(round(max(0.0, float(s)) * bs)) for t, s in cfg.targets.items()
    }

    def _tier_cap(tier: str) -> int:
        return max_caps_int.get(tier, bs)

    # Step 1: floors from min_caps (clamped to pool size and to max_cap).
    boss_cap_hit = False
    elite_min_unfilled = False
    normal_min_unfilled = False
    for tier, min_share in cfg.min_caps.items():
        floor_count = int(math.ceil(max(0.0, float(min_share)) * bs))
        pool = max(0, int(pool_sizes.get(tier, 0)))
        give = min(floor_count, pool, _tier_cap(tier))
        quotas[tier] = quotas.get(tier, 0) + give
        if tier == "elite" and give < floor_count:
            elite_min_unfilled = True
        if tier == "normal" and give < floor_count:
            normal_min_unfilled = True

    # Step 2: distribute remaining slots proportional to targets, respecting
    # both pool sizes and max_caps. Iterate until nothing more can be placed.
    remaining = bs - sum(quotas.values())
    safety_iterations = bs * 4 + 4  # bounded loop; protects against pathological configs
    while remaining > 0 and safety_iterations > 0:
        safety_iterations -= 1
        # Active tiers: targets > 0, pool > 0, below max_cap, below pool size.
        active = [
            tier
            for tier in tier_keys
            if max(0.0, float(cfg.targets.get(tier, 0.0))) > 0
            and pool_sizes.get(tier, 0) > 0
            and quotas.get(tier, 0) < _tier_cap(tier)
            and quotas.get(tier, 0) < pool_sizes.get(tier, 0)
        ]
        if not active:
            break
        weights = {t: max(0.0, float(cfg.targets.get(t, 0.0))) for t in active}
        total_w = sum(weights.values())
        if total_w <= 0:
            break
        # Allocate one batch of slots based on weight proportions, but never
        # exceed each tier's slack-to-cap-or-pool. ``floor + leftovers``
        # rounding keeps the loop deterministic.
        slack_caps = {
            t: min(_tier_cap(t) - quotas.get(t, 0), pool_sizes.get(t, 0) - quotas.get(t, 0))
            for t in active
        }
        provisional_floats = {t: weights[t] / total_w * remaining for t in active}
        floors = {t: int(math.floor(provisional_floats[t])) for t in active}
        for t, slack in slack_caps.items():
            if floors[t] > slack:
                floors[t] = max(0, slack)
        placed = sum(floors.values())
        # Leftover slots → tier with largest fractional part still having
        # slack. Stop when all eligible tiers are saturated.
        leftover_pool = remaining - placed
        if leftover_pool > 0:
            fractional = sorted(
                (
                    (t, provisional_floats[t] - floors[t])
                    for t in active
                ),
                key=lambda item: (-item[1], item[0]),
            )
            for t, _ in fractional:
                if leftover_pool <= 0:
                    break
                slack = min(_tier_cap(t) - quotas.get(t, 0) - floors[t], pool_sizes.get(t, 0) - quotas.get(t, 0) - floors[t])
                if slack <= 0:
                    continue
                floors[t] += 1
                leftover_pool -= 1
        for t, count in floors.items():
            if count > 0:
                quotas[t] = quotas.get(t, 0) + count
        # If the proportional distribution placed nothing (every active
        # tier had slack=0 after capping), break to avoid infinite loop.
        if sum(floors.values()) == 0:
            break
        remaining = bs - sum(quotas.values())

    # Step 3: any remaining slots become flex (caller falls back to global
    # priority pool). Also: if a max_cap pushed a tier exactly to its cap
    # and the boss tier was the one capped, record boss_cap_hit.
    if quotas.get("boss", 0) >= max_caps_int.get("boss", bs) and "boss" in cfg.max_caps:
        # Only flag cap hit if at least one slot of pressure was actually
        # absorbed by the cap (i.e., the unconstrained allocation would
        # have given boss > cap). When raw_target_counts["boss"] > cap,
        # we know there was pressure.
        if raw_target_counts.get("boss", 0) > max_caps_int.get("boss", bs):
            boss_cap_hit = True

    if remaining > 0:
        quotas["flex"] = quotas.get("flex", 0) + remaining

    # Step 4: final pool clamps (defensive — earlier steps already cap).
    for tier in list(quotas.keys()):
        if tier == "flex":
            continue
        pool = max(0, int(pool_sizes.get(tier, 0)))
        if quotas[tier] > pool:
            shortfall = quotas[tier] - pool
            quotas[tier] = pool
            quotas["flex"] = quotas.get("flex", 0) + shortfall

    # Drop tiers that ended at zero so the returned dict stays compact, but
    # keep boss/elite/normal/weak even at 0 so callers can render zeros.
    canonical = {"boss", "elite", "normal", "weak"}
    quotas = {t: int(v) for t, v in quotas.items() if v > 0 or t in canonical}

    return TierQuotaAllocation(
        quotas=quotas,
        target_counts=raw_target_counts,
        boss_cap_hit=boss_cap_hit,
        normal_min_unfilled=normal_min_unfilled,
        elite_min_unfilled=elite_min_unfilled,
    )
