"""Route heuristic v1 (recovery 2026-05-08, Phase 2).

Per ``docs/muzero-route-deck-long-horizon-review-20260508.md`` §7: score
each map legal action's reachable subtree using the bridge-supplied
``route_summary`` (20-key dict), the player's current deck-quality
profile (Phase 1), and basic resources (HP, gold, potions, floor/act).

Design rules:

* Strictly dry-run — this module returns scores + a structured breakdown.
  It MUST NOT mutate the action mask, the policy prior, or any tensor.
* Per-legal-action score (§3 row 4 of the review): the heuristic produces
  one score per candidate map action, never one "current path score".
  Score ordering is what callers consume (top1 selected rate, etc.).
* Inputs come from already-verified typed sources only — no text regex,
  no name-pattern guessing. Phase 0 audit confirmed the required keys.
* Elite risk is *dynamic*: it must respond to deck strength, HP, potion
  inventory, and whether a rest sits between us and the elite. A flat
  ``-2 × n_elites`` would punish strong decks for taking optimal lines.

Output schema (the keys the caller must rely on):

    {
      "score":                    float,  # combined ranking scalar
      "score_normalized":         float,  # clamp(score / SCORE_REF, -1, 1)
      "boss_progress":            float,
      "rest_value":               float,
      "shop_value":               float,
      "treasure_value":           float,
      "event_value":              float,
      "branch_value":             float,
      "unsafe_elite_penalty":     float,
      "forced_elite_penalty":     float,
      "no_rest_before_elite_penalty": float,
      "low_hp_monster_chain_penalty": float,
      "elite_risk_factor":        float,  # the dynamic risk multiplier
      "low_hp_elite_flag":        float,  # 1.0 if low_hp + elite path
      "low_hp_flag":              float,
      "rest_before_elite_available": float,
      "risk_class":               float,  # 0 safe .. 4 extreme
      "risk_reason":              str,
      "summary_used":             bool,   # False => safe-zero output
    }

The caller compares scores across legal map actions to compute:

* top1_score / top2_score / selected_score / best_minus_selected
* low_hp_elite_selected_rate (selected has elite + low_hp_flag=1)
* rest_selected_low_hp_rate (selected has rest + low_hp_flag=1)
* shop_selected_high_gold_rate (selected has shop + gold>=threshold)

These rates are what Phase 2 dry-run logs as TB metrics. Only Phase 3
will turn this into an MCTS prior bias.
"""
from __future__ import annotations

from typing import Any


# Reference scale used for ``score_normalized``. Picked so a typical
# strong/weak deck on a typical fork yields ``score_normalized ∈ [-1, 1]``
# without losing tail signal.
_SCORE_REF: float = 4.0

# Public thresholds (P1-4 review fix): keep low-HP definitions consistent
# across the heuristic and the trainer dry-run aggregator. Two distinct
# thresholds are intentional — a fight-blocking emergency (≤0.40) is
# different from "should rest if possible" (≤0.50).
ROUTE_LOW_HP_RATIO: float = 0.40
REST_URGENCY_HP_RATIO: float = 0.50

# Potion-slot strings recognised as empty across bridge versions
# (P1-5 review fix — pre-review only "[empty]" was excluded so empty
# slots were occasionally counted as potions).
_POTION_EMPTY_STRINGS: frozenset[str] = frozenset({
    "", "empty", "none", "null", "[empty]", "potion.empty", "potion.none",
})


def count_non_empty_potions(potions: Any) -> int:
    """Count usable potions in the player's potion list.

    Accepts the bridge's heterogeneous potion-slot encoding:

    * ``str``  — raw slot id; empty if it matches ``_POTION_EMPTY_STRINGS``.
    * ``dict`` — slot record; empty if ``empty=True`` OR title/id/name
      matches the empty-strings set.
    * ``None`` / non-list input — zero potions.
    """
    if not isinstance(potions, list):
        return 0
    count = 0
    for slot in potions:
        if isinstance(slot, str):
            if slot.strip().lower() in _POTION_EMPTY_STRINGS:
                continue
            count += 1
        elif isinstance(slot, dict):
            if slot.get("empty") is True:
                continue
            for key in ("title", "id", "name", "potion_id"):
                value = slot.get(key)
                if value is not None:
                    text = str(value).strip().lower()
                    if text and text not in _POTION_EMPTY_STRINGS:
                        count += 1
                        break
    return count


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _safe_float(value: Any, default: float = 0.0) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _safe_int(value: Any, default: int = 0) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


def _clamp(value: float, lo: float = -1.0, hi: float = 1.0) -> float:
    if value < lo:
        return lo
    if value > hi:
        return hi
    return value


def _zero_score() -> dict[str, Any]:
    return {
        "score": 0.0,
        "score_normalized": 0.0,
        "boss_progress": 0.0,
        "rest_value": 0.0,
        "shop_value": 0.0,
        "treasure_value": 0.0,
        "event_value": 0.0,
        "branch_value": 0.0,
        "unsafe_elite_penalty": 0.0,
        "forced_elite_penalty": 0.0,
        "no_rest_before_elite_penalty": 0.0,
        "low_hp_monster_chain_penalty": 0.0,
        "elite_risk_factor": 0.0,
        "low_hp_elite_flag": 0.0,
        "low_hp_flag": 0.0,
        "rest_before_elite_available": 0.0,
        "forced_elite_count": 0.0,
        "immediate_elite_count": 0.0,
        "optional_elite_count": 0.0,
        "subtree_elite_count": 0.0,
        "risk_class": 0.0,
        "risk_reason": "",
        "summary_used": False,
    }


def _hp_ratio(hp: float, max_hp: float) -> float:
    if max_hp <= 1.0:
        # Missing/suspicious max HP is unsafe/unknown, not full HP.  The
        # caller may fail-open; the heuristic must not route as if healthy.
        return 0.0
    return _clamp(hp / max_hp, 0.0, 1.0)


# ---------------------------------------------------------------------------
# Elite risk calculation
# ---------------------------------------------------------------------------

def compute_elite_risk(
    *,
    deck_quality: dict[str, float] | None,
    hp_ratio: float,
    potion_count: int,
    rest_before_elite: bool,
    next_elite_steps: int,
) -> float:
    """Continuous elite-risk factor in roughly ``[0, 2]``.

    Higher means "this elite will hurt us". 1.0 is the neutral midpoint.
    The factor is read off **typed** inputs only — no text regex.

    The signal sources, weights tuned by inspection against §7.5:
    * low HP                          +0.6 (full at hp_ratio<=0.4)
    * weak frontload (deck)           +0.4
    * weak block (deck)               +0.4
    * poor rotation (deck draw/refund)+0.3
    * heavy pollution (curse/status)  +0.4
    * no potion                       +0.2
    * no rest before elite            +0.3
    * elite-readiness directly        -0.8 (subtracted, scaled by score)

    Cumulatively bounded to ``[0.1, 2.0]`` so a strong deck can't
    completely zero the penalty (you can still get unlucky), and an
    abysmal deck doesn't asymptote past 2× regular weight.
    """
    dq = deck_quality or {}
    risk = 1.0

    # HP factor.
    if hp_ratio <= 0.40:
        risk += 0.6 * (1.0 - hp_ratio / 0.40)
    elif hp_ratio >= 0.85:
        risk -= 0.2

    # Deck frontload / block.
    front = _safe_float(dq.get("frontload_score"))
    block = _safe_float(dq.get("block_score"))
    if front < 0.25:
        risk += 0.4 * (1.0 - front / 0.25)
    if block < 0.20:
        risk += 0.4 * (1.0 - block / 0.20)

    # Rotation / engine.
    draw = _safe_float(dq.get("draw_engine_score"))
    energy = _safe_float(dq.get("energy_engine_score"))
    if draw < 0.20 and energy < 0.20:
        risk += 0.3

    # Pollution drag.
    pollution = _safe_float(dq.get("pollution_score"))
    if pollution > 0.0:
        risk += 0.4 * pollution

    # Resources.
    if potion_count <= 0:
        risk += 0.2
    if not rest_before_elite and next_elite_steps <= 2:
        risk += 0.3

    # Direct elite-readiness override — strong deck offsets risk.
    elite_ready = _safe_float(dq.get("elite_readiness_score"))
    risk -= 0.8 * elite_ready

    if risk < 0.1:
        return 0.1
    if risk > 2.0:
        return 2.0
    return risk


# ---------------------------------------------------------------------------
# Component scores
# ---------------------------------------------------------------------------

def _rest_value(
    *,
    summary: dict[str, Any],
    hp_ratio: float,
    elite_risk: float,
) -> float:
    count = _safe_float(summary.get("count_rest_site"))
    if count <= 0:
        return 0.0
    base = 0.4 * count
    # Boost when low HP — rest is worth a lot.
    if hp_ratio < 0.50:
        base *= 1.0 + (0.50 - hp_ratio) * 2.0  # up to ×2 at hp=0
    # Boost when an elite is upcoming and risky.
    next_elite = _safe_int(summary.get("next_elite_steps"), 99)
    if next_elite <= 4 and elite_risk > 1.2:
        base *= 1.0 + 0.5 * (elite_risk - 1.0)
    return base


def _shop_value(
    *,
    summary: dict[str, Any],
    gold: float,
    deck_quality: dict[str, float] | None,
) -> float:
    """Reverted 2026-05-09 from v2 (base=0.45) back to v1 (base=0.30)
    after the v2 combo (elite penalty bump + shop bump) made
    shop+forced-elite paths newly attractive in the heuristic and
    dropped agent outcomes across 55 episodes. The v1 base under-uses
    shops slightly but keeps the relative ordering against elite paths
    correct, which the v2 combo did not."""
    count = _safe_float(summary.get("count_shop"))
    if count <= 0:
        return 0.0
    dq = deck_quality or {}
    base = 0.30 * count
    # Gold gates shop value — too poor and the shop is wasted.
    if gold < 75:
        base *= 0.3
    elif gold > 200:
        base *= 1.4
    # Pollution = remove demand.
    pollution = _safe_float(dq.get("pollution_score"))
    base += 0.5 * pollution
    return base


def _event_value(summary: dict[str, Any]) -> float:
    """Events are heterogeneous; small positive weight per review §7.8."""
    count = _safe_float(summary.get("count_event")) + _safe_float(
        summary.get("count_question_mark")
    )
    return 0.15 * count


def _treasure_value(summary: dict[str, Any]) -> float:
    count = _safe_float(summary.get("count_treasure"))
    return 0.4 * count


def _branch_value(summary: dict[str, Any]) -> float:
    """Reward keeping options open — high reachable_node_count + branching.

    P1-3 (review 2026-05-08 fix): cap the raw value at 0.8. Without the
    cap a candidate with a 30-node subtree reaches branch_value≈1.5,
    drowning out elite/rest penalties and making the heuristic prefer
    big subtrees regardless of their content. 0.8 keeps branch as a
    tiebreaker, not the dominant term.
    """
    reachable = _safe_float(summary.get("reachable_node_count"))
    direct_children = _safe_float(summary.get("direct_child_count"))
    forced_steps_raw = summary.get("forced_path_steps_before_branch")
    forced_steps = _safe_float(forced_steps_raw) if forced_steps_raw is not None else 0.0
    raw = 0.05 * reachable + 0.20 * direct_children
    if forced_steps >= 3:
        raw -= 0.3 * (forced_steps - 2)
    if raw > 0.8:
        raw = 0.8
    if raw < -0.6:  # symmetric clamp on the negative side
        raw = -0.6
    return raw


def _boss_progress(summary: dict[str, Any]) -> float:
    """Mild bonus for paths that lead toward a boss room."""
    if _safe_float(summary.get("count_boss")) > 0:
        return 0.5
    return 0.0


def _split_elite_counts(summary: dict[str, Any]) -> tuple[float, float, float, float]:
    """Split ``count_elite`` (subtree total) into forced vs optional.

    P1-2 (review 2026-05-08 fix): pre-fix the heuristic punished any
    elite in the reachable subtree at full weight. That over-punished
    high-branching paths where the agent can dodge the elite. We now
    distinguish:

    * ``forced_elite_count``   — elite on a path the agent cannot avoid
      by branching (next_elite_steps <= forced_path_steps_before_branch,
      OR forced_path is None which means a fully linear chain).
    * ``immediate_elite_count``— elite within the next 1 step (special
      case of forced — used by Phase 3 prior bias for sharper signal).
    * ``optional_elite_count`` — count_elite minus forced. These are
      reachable elites the agent can dodge by picking a different
      branch downstream.
    * ``subtree_elite_count``  — raw count_elite (kept for diagnostics).
    """
    n_total = _safe_float(summary.get("count_elite"))
    if n_total <= 0:
        return 0.0, 0.0, 0.0, 0.0
    raw_forced = summary.get("forced_path_steps_before_branch")
    next_elite_raw = summary.get("next_elite_steps")
    next_elite = _safe_int(next_elite_raw, 99) if next_elite_raw is not None else 99

    if raw_forced is None and next_elite_raw is not None:
        forced_count = min(1.0, n_total)  # linear chain — at least one forced
    else:
        forced_steps = _safe_int(raw_forced, 0)
        if forced_steps >= 1 and next_elite <= forced_steps:
            forced_count = min(1.0, n_total)
        else:
            forced_count = 0.0
    immediate_count = 1.0 if next_elite <= 1 else 0.0
    optional_count = max(0.0, n_total - forced_count)
    return forced_count, immediate_count, optional_count, n_total


def _route_risk_class(
    *,
    summary: dict[str, Any],
    hp_ratio: float,
    elite_risk: float,
    forced_elites: float,
    immediate_elites: float,
    optional_elites: float,
    no_rest_before_elite_penalty: float,
    low_hp_monster_chain_penalty: float,
) -> tuple[float, str]:
    """Coarse safety class for emergency route hard guards.

    The continuous ``score`` remains the ordinary ranking signal.  This
    class is intentionally conservative and low-resolution so a guard can
    block only obviously unsafe route picks while leaving strategic
    optional-elite/shop/rest tradeoffs to the policy/value stack.

    Classes:
      0  safe/resource path (no elite pressure, no low-HP chain)
      1  acceptable chip path
      2  medium risk (optional/distant elite or elevated elite risk)
      3  high risk (forced/immediate elite, no rest before elite, or low-HP chain)
      4  extreme risk (low HP plus forced/immediate/no-rest elite, or severe chain)
    """
    n_elites = _safe_float(summary.get("count_elite"))
    n_monsters = _safe_float(summary.get("count_monster"))
    n_rests = _safe_float(summary.get("count_rest_site"))
    low_hp = hp_ratio < ROUTE_LOW_HP_RATIO
    urgent_hp = hp_ratio < REST_URGENCY_HP_RATIO
    forced_or_immediate = forced_elites > 0.0 or immediate_elites > 0.0
    no_rest_elite = no_rest_before_elite_penalty > 0.0
    low_hp_chain = low_hp_monster_chain_penalty > 0.0

    if low_hp and (forced_or_immediate or no_rest_elite):
        return 4.0, "low_hp_forced_or_no_rest_elite"
    if hp_ratio <= 0.25 and low_hp_chain:
        return 4.0, "critical_hp_monster_chain"
    if forced_or_immediate:
        return 3.0, "forced_or_immediate_elite"
    if no_rest_elite:
        return 3.0, "elite_without_rest"
    if low_hp_chain:
        return 3.0, "low_hp_monster_chain"
    if n_elites > 0.0:
        if optional_elites > 0.0:
            return 2.0, "optional_elite"
        return 2.0, "elite_subtree"
    if elite_risk > 1.25 and urgent_hp:
        return 2.0, "high_risk_low_hp"
    if urgent_hp and n_monsters > 1.0 and n_rests <= 0.0:
        return 1.0, "urgent_hp_regular_fights"
    if n_monsters > 0.0:
        return 1.0, "regular_fights"
    return 0.0, "safe_resource_path"


def _unsafe_elite_penalty(
    *,
    summary: dict[str, Any],
    elite_risk: float,
) -> float:
    """P1-2 split-weighted unsafe elite penalty.

    * forced elites:  full weight (1.0 × elite_risk)
    * optional elites: 0.25 × elite_risk (agent can dodge by branching)

    Pre-fix this was ``count_elite * elite_risk`` — over-punished
    high-branching paths whose subtree happened to contain an elite the
    agent could dodge.

    Reverted 2026-05-09: a v2 attempt at forced=1.5 / optional=0.5 with
    parallel shop_base=0.45 ramped up forced-elite paths in
    ``best_forced_elite_count_mean`` (0.0 → 0.030) and dropped the
    agent's reward / death_floor / act1_boss_seen across 55 episodes.
    The interaction between higher elite cost AND higher shop reward
    paradoxically made shop+forced-elite combos newly attractive. Stick
    with the v1 calibration until a more thoughtful redesign lands.
    """
    forced, _immediate, optional, _total = _split_elite_counts(summary)
    return forced * elite_risk * 1.0 + optional * elite_risk * 0.25


def _forced_elite_penalty(summary: dict[str, Any]) -> float:
    """Path that has no branch and lands directly on an elite is more
    punishing than an elite that's optional — the agent can't dodge it.

    P1-1 (review 2026-05-08): treat ``forced_path_steps_before_branch is
    None`` as a fully linear path. The bridge sets None when the BFS
    never observed a branch (single-child chain). In that case the elite
    is unavoidable from this candidate.
    """
    raw_forced = summary.get("forced_path_steps_before_branch")
    next_elite_raw = summary.get("next_elite_steps")
    n_elites = _safe_float(summary.get("count_elite"))
    if n_elites <= 0:
        return 0.0
    if raw_forced is None and next_elite_raw is not None:
        # Linear chain to elite — fully forced.
        return 0.6
    forced = _safe_int(raw_forced)
    next_elite = _safe_int(next_elite_raw, 99)
    if forced >= 1 and next_elite <= forced:
        return 0.6
    return 0.0


def _no_rest_before_elite_penalty(
    *,
    summary: dict[str, Any],
    hp_ratio: float,
    elite_risk: float,
) -> float:
    n_elites = _safe_float(summary.get("count_elite"))
    if n_elites <= 0:
        return 0.0
    can_reach_rest = bool(summary.get("can_reach_rest_site_before_elite"))
    if can_reach_rest:
        return 0.0
    base = 0.5 * n_elites
    if hp_ratio < 0.50:
        base *= 1.0 + (0.50 - hp_ratio) * 2.0
    if elite_risk > 1.3:
        base *= 1.0 + 0.3 * (elite_risk - 1.0)
    return base


def _low_hp_monster_chain_penalty(
    *,
    summary: dict[str, Any],
    hp_ratio: float,
) -> float:
    """Many normal monster fights with no rest while HP is low → drain."""
    if hp_ratio >= 0.40:
        return 0.0
    n_monsters = _safe_float(summary.get("count_monster"))
    n_rests = _safe_float(summary.get("count_rest_site"))
    if n_monsters <= 1 or n_rests > 0:
        return 0.0
    return 0.3 * (n_monsters - 1) * (1.0 - hp_ratio / 0.40)


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------

def score_route_action(
    *,
    route_summary: Any,
    deck_quality: dict[str, float] | None,
    hp: float,
    max_hp: float,
    gold: float,
    potion_count: int,
    act: int = 1,  # noqa: ARG001 - reserved for act-aware scoring
    floor: int = 0,  # noqa: ARG001 - reserved for floor-aware scoring
) -> dict[str, Any]:
    """Score one map legal action's subtree.

    ``route_summary`` is the dict found at ``action["route_summary"]`` for
    a ``kind == "map"`` legal action. All other inputs come from the
    current obs.player / obs.run fields. Missing summary returns
    ``_zero_score()`` with ``summary_used=False``.

    The score is a continuous scalar; callers rank actions by it. This
    function never modifies its inputs.
    """
    if not isinstance(route_summary, dict):
        return _zero_score()

    # Missing HP is commonly serialized as 0 / absent during transient route
    # observations.  Treat that as "unknown/full" rather than "critically low";
    # otherwise the safety heuristic can hallucinate low-HP elite danger and
    # distort route decisions.  A real route decision at hp<=0 is not actionable.
    hp_v = _safe_float(hp)
    max_hp_v = _safe_float(max_hp)
    hp_valid = bool(hp_v > 0.0 and max_hp_v > 0.0)
    hp_ratio = _hp_ratio(hp_v, max_hp_v) if hp_valid else 1.0
    rest_before_elite = bool(route_summary.get("can_reach_rest_site_before_elite"))
    next_elite_steps = _safe_int(route_summary.get("next_elite_steps"), 99)

    elite_risk = compute_elite_risk(
        deck_quality=deck_quality,
        hp_ratio=hp_ratio,
        potion_count=int(potion_count or 0),
        rest_before_elite=rest_before_elite,
        next_elite_steps=next_elite_steps,
    )

    rest_v = _rest_value(summary=route_summary, hp_ratio=hp_ratio, elite_risk=elite_risk)
    shop_v = _shop_value(summary=route_summary, gold=_safe_float(gold), deck_quality=deck_quality)
    event_v = _event_value(route_summary)
    treasure_v = _treasure_value(route_summary)
    branch_v = _branch_value(route_summary)
    boss_p = _boss_progress(route_summary)

    forced_elites, immediate_elites, optional_elites, subtree_elites = _split_elite_counts(route_summary)
    unsafe_p = _unsafe_elite_penalty(summary=route_summary, elite_risk=elite_risk)
    forced_p = _forced_elite_penalty(route_summary)
    no_rest_p = _no_rest_before_elite_penalty(
        summary=route_summary, hp_ratio=hp_ratio, elite_risk=elite_risk
    )
    low_hp_chain_p = _low_hp_monster_chain_penalty(summary=route_summary, hp_ratio=hp_ratio)

    score = (
        boss_p
        + rest_v
        + shop_v
        + event_v
        + treasure_v
        + branch_v
        - unsafe_p
        - forced_p
        - no_rest_p
        - low_hp_chain_p
    )

    n_elites = _safe_float(route_summary.get("count_elite"))
    low_hp_flag = 1.0 if (hp_valid and hp_ratio < 0.40) else 0.0
    low_hp_elite_flag = 1.0 if (low_hp_flag > 0 and n_elites > 0) else 0.0
    rest_before_elite_avail = 1.0 if rest_before_elite else 0.0
    risk_class, risk_reason = _route_risk_class(
        summary=route_summary,
        hp_ratio=hp_ratio,
        elite_risk=elite_risk,
        forced_elites=forced_elites,
        immediate_elites=immediate_elites,
        optional_elites=optional_elites,
        no_rest_before_elite_penalty=no_rest_p,
        low_hp_monster_chain_penalty=low_hp_chain_p,
    )

    return {
        "score": float(score),
        "score_normalized": _clamp(score / _SCORE_REF, -1.0, 1.0),
        "boss_progress": float(boss_p),
        "rest_value": float(rest_v),
        "shop_value": float(shop_v),
        "treasure_value": float(treasure_v),
        "event_value": float(event_v),
        "branch_value": float(branch_v),
        "unsafe_elite_penalty": float(unsafe_p),
        "forced_elite_penalty": float(forced_p),
        "no_rest_before_elite_penalty": float(no_rest_p),
        "low_hp_monster_chain_penalty": float(low_hp_chain_p),
        "elite_risk_factor": float(elite_risk),
        "low_hp_elite_flag": float(low_hp_elite_flag),
        "low_hp_flag": float(low_hp_flag),
        "rest_before_elite_available": float(rest_before_elite_avail),
        # P1-2 split (review 2026-05-08): expose individual counts so the
        # caller / Phase 3 prior bias can use forced/immediate as the
        # action-discriminating signal and treat optional as informational.
        "forced_elite_count": float(forced_elites),
        "immediate_elite_count": float(immediate_elites),
        "optional_elite_count": float(optional_elites),
        "subtree_elite_count": float(subtree_elites),
        "risk_class": float(risk_class),
        "risk_reason": str(risk_reason),
        "summary_used": True,
    }


def rank_legal_route_actions(
    *,
    legal_actions: list[Any],
    deck_quality: dict[str, float] | None,
    hp: float,
    max_hp: float,
    gold: float,
    potion_count: int,
    act: int = 1,
    floor: int = 0,
) -> list[dict[str, Any]]:
    """Score every map legal action and return parallel list with
    ``score_breakdown`` attached. Non-map actions get ``None``.

    Caller derives top1/top2 indices and best_minus_selected from this.
    """
    out: list[dict[str, Any]] = []
    for action in legal_actions or []:
        if not isinstance(action, dict):
            out.append(None)  # type: ignore[arg-type]
            continue
        if str(action.get("kind") or "").strip().lower() != "map":
            out.append(None)  # type: ignore[arg-type]
            continue
        breakdown = score_route_action(
            route_summary=action.get("route_summary"),
            deck_quality=deck_quality,
            hp=hp,
            max_hp=max_hp,
            gold=gold,
            potion_count=potion_count,
            act=act,
            floor=floor,
        )
        out.append(breakdown)
    return out


def choose_route_safety_override(
    ranked: list[dict[str, Any] | None],
    selected_idx: int,
    *,
    high_risk_threshold: int = 3,
    safe_risk_threshold: int = 1,
) -> dict[str, Any]:
    """Pick a safe map-action override for low-HP route collapse.

    Pure helper: it only inspects the already-ranked per-action breakdowns
    and returns a structured decision.  The trainer is responsible for
    checking the action mask and compact/full-action positional alignment
    before using ``override_idx``.

    Override rule:
      * selected action must be a scored map candidate with ``summary_used``
      * selected ``risk_class`` must be >= ``high_risk_threshold``
      * prefer an alternative candidate with ``risk_class`` <=
        ``safe_risk_threshold``
      * if no fully safe route exists, still allow a downgrade to the
        currently best *lower-risk* route.  This is intentionally a hard
        survival fallback for late Act1 low-HP maps where every visible route
        is bad, but one choice is clearly less bad than an immediate/forced
        elite branch.
      * tie-break alternatives by lowest risk first, then highest score
    """
    result: dict[str, Any] = {
        "applicable": False,
        "override": False,
        "override_idx": None,
        "selected_risk_class": 0.0,
        "final_risk_class": 0.0,
        "safe_available": False,
        "lower_risk_available": False,
        "selected_forced_elite": False,
        "selected_immediate_elite": False,
        "low_hp_forced": False,
        "reason": "not_applicable",
    }
    if not isinstance(ranked, list) or not (0 <= int(selected_idx) < len(ranked)):
        result["reason"] = "bad_selected_idx"
        return result
    selected = ranked[int(selected_idx)]
    if not isinstance(selected, dict) or not selected.get("summary_used"):
        result["reason"] = "selected_unscored"
        return result

    selected_risk = _safe_float(selected.get("risk_class"))
    selected_forced = _safe_float(selected.get("forced_elite_count")) > 0.0 or _safe_float(
        selected.get("forced_elite_penalty")
    ) > 0.0
    selected_immediate = _safe_float(selected.get("immediate_elite_count")) > 0.0
    selected_no_rest = _safe_float(selected.get("no_rest_before_elite_penalty")) > 0.0
    selected_low_hp = _safe_float(selected.get("low_hp_flag")) > 0.0
    result.update(
        {
            "applicable": selected_risk >= float(high_risk_threshold),
            "selected_risk_class": float(selected_risk),
            "final_risk_class": float(selected_risk),
            "selected_forced_elite": bool(selected_forced),
            "selected_immediate_elite": bool(selected_immediate),
            "low_hp_forced": bool(selected_low_hp and (selected_forced or selected_immediate or selected_no_rest)),
        }
    )
    if selected_risk < float(high_risk_threshold):
        result["reason"] = "selected_not_high_risk"
        return result

    safe_candidates: list[tuple[float, float, int]] = []
    lower_risk_candidates: list[tuple[float, float, int]] = []
    for idx, breakdown in enumerate(ranked):
        if idx == int(selected_idx):
            continue
        if not isinstance(breakdown, dict) or not breakdown.get("summary_used"):
            continue
        risk = _safe_float(breakdown.get("risk_class"))
        score = _safe_float(breakdown.get("score"))
        if risk <= float(safe_risk_threshold):
            # Sort ascending risk, descending score, then stable index.
            safe_candidates.append((risk, -score, idx))
        if risk < selected_risk:
            lower_risk_candidates.append((risk, -score, idx))
    result["safe_available"] = bool(safe_candidates)
    result["lower_risk_available"] = bool(lower_risk_candidates)
    if not safe_candidates:
        if not lower_risk_candidates:
            result["reason"] = "no_lower_risk_alternative"
            return result
        lower_risk_candidates.sort()
        best_risk, _neg_score, best_idx = lower_risk_candidates[0]
        result.update(
            {
                "override": True,
                "override_idx": int(best_idx),
                "final_risk_class": float(best_risk),
                "reason": "override_high_risk_to_lower_risk",
            }
        )
        return result

    safe_candidates.sort()
    best_risk, _neg_score, best_idx = safe_candidates[0]
    result.update(
        {
            "override": True,
            "override_idx": int(best_idx),
            "final_risk_class": float(best_risk),
            "reason": "override_high_risk_to_safe",
        }
    )
    return result
