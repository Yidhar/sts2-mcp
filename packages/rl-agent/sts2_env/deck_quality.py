"""Deck-quality v2 features (recovery 2026-05-08, Phase 1).

Per ``docs/muzero-route-deck-long-horizon-review-20260508.md`` §6: extend
the existing ``_build_profile()`` deck features with continuous-score
extras that capture cost-efficiency, rotation/engine, scaling, and
readiness signals. These features are read off
``obs["player"]["deck_cards"]`` and looked-up against the verified card
metadata registry — never derived from text regex.

Design rules (§4.1, §6.4):

* All inputs come from typed sources: ``card.id``, ``card.type``,
  ``card.cost`` (int), ``card.x_cost`` (bool), bridge-exposed
  ``card_effect_profile.semantic_tags``, or
  ``content_registry.get_card_metadata(card.id).semantic_tags`` /
  ``semantic_signals``.
* Model-facing score/density features are clamp-normalised to ``[0, 1]``
  (or ``[-1, 1]`` for delta scores) so they can be fed into a model without
  scaling drift. Operator-facing raw/count diagnostics such as ``raw_*`` and
  ``expected_hand_*_cards`` are finite floats and may exceed 1.
* All features are safe-zero when metadata is missing — the caller
  always gets the full key set.
* No hard one-hot archetype: scores are continuous so a borderline deck
  doesn't get classified as a different archetype on a single card pick.
* X-cost cards do not pretend to be 0-cost or 3-cost — they get a
  dedicated density feature and contribute to ``x_cost_damage_potential``
  / ``x_cost_block_potential`` instead of polluting ``avg_cost``.

P1-6 (review 2026-05-08 fix): an earlier draft of this docstring claimed
the helpers were already wired into ``run_memory._build_profile()`` —
they are not. Phase 1 only computes + logs ``deck_quality_v2`` at
episode end and feeds it to the Phase 2 route-heuristic dry-run. It is
NOT yet baked into the model observation vector or replay buffer fixed
shapes; that wiring is deferred to a later phase that explicitly bumps
schema (RUN_MEMORY_DIM / observation_common shapes).

Note for downstream phases:
* The 30+ features here are intentionally NOT yet wired into the model
  observation vector. Phase 1 only adds the compute + tests + diagnostic
  logging path. Wiring requires a schema-version bump (RUN_MEMORY_DIM /
  observation buffer fixed shape) which is deferred to Phase 2/3 once
  Route heuristic dry-run validates the signals are useful.
"""
from __future__ import annotations

from typing import Any, Iterable

from content_registry import get_card_metadata


# ---------------------------------------------------------------------------
# Field group definitions — keeps the public schema stable for tests and
# downstream consumers (TB, replay buffer migration). Adding/removing a key
# here is a deliberate schema change.
# ---------------------------------------------------------------------------

DECK_QUALITY_V2_KEYS: tuple[str, ...] = (
    # 基础
    "deck_size_norm",
    "upgraded_ratio",
    "attack_density",
    "skill_density",
    "power_density",
    "curse_density",
    "status_density",
    # 费用 / 效率
    "avg_cost",
    "avg_damage_per_energy",
    "avg_block_per_energy",
    # raw diagnostics (not model-normalised): operator-facing deck strength
    "raw_avg_damage_per_energy",
    "raw_avg_block_per_energy",
    "high_cost_density",
    "zero_cost_density",
    "x_cost_density",
    "x_cost_damage_potential",
    "x_cost_block_potential",
    # 轮转 / 引擎
    "draw_density",
    "expected_extra_draw_per_turn",
    "raw_expected_extra_draw_per_turn",
    "raw_expected_cards_seen_per_turn",
    "energy_refund_density",
    "cost_reduce_density",
    "retain_density",
    "innate_density",
    "exhaust_density",
    "ethereal_density",
    "copy_replay_density",
    # Scaling
    "strength_scaling_density",
    "dex_scaling_density",
    "poison_scaling_density",
    "block_scaling_density",
    "power_scaling_density",
    # 综合分数 (continuous)
    "frontload_score",
    "block_score",
    "scaling_score",
    "draw_engine_score",
    "energy_engine_score",
    "pollution_score",
    "consistency_score",
    "elite_readiness_score",
    "boss_readiness_score",
    # 每回合期望手牌组成 / 质量（raw counts/totals + normalised scores）
    "expected_hand_attack_cards",
    "expected_hand_skill_cards",
    "expected_hand_power_cards",
    "expected_hand_curse_cards",
    "expected_hand_status_cards",
    "expected_hand_draw_cards",
    "expected_hand_engine_cards",
    "expected_hand_scaling_cards",
    "expected_hand_unplayable_cards",
    "expected_hand_attack_damage_per_turn",
    "expected_hand_block_per_turn",
    "expected_hand_attack_share",
    "expected_hand_skill_share",
    "expected_hand_power_share",
    "expected_hand_junk_share",
    "expected_hand_frontload_score",
    "expected_hand_block_score",
    "expected_hand_draw_score",
    "expected_hand_engine_score",
    "expected_hand_scaling_score",
    "expected_hand_pollution_score",
    "expected_hand_useful_quality_score",
    "expected_hand_quality_attack_share",
    "expected_hand_quality_block_share",
    "expected_hand_quality_draw_share",
    "expected_hand_quality_scaling_share",
    "expected_hand_quality_pollution_share",
    # 能量预算下的“实际能打出去”估计。raw_* 是每回合量级；
    # score/share 是 [0,1] 诊断分数。
    "raw_expected_energy_budget_per_turn",
    "raw_expected_playable_cards_per_turn",
    "raw_expected_playable_energy_spent_per_turn",
    "raw_expected_unspent_energy_per_turn",
    "raw_expected_playable_attack_cards_per_turn",
    "raw_expected_playable_skill_cards_per_turn",
    "raw_expected_playable_power_cards_per_turn",
    "raw_expected_playable_attack_damage_per_turn",
    "raw_expected_playable_block_per_turn",
    "expected_energy_utilization_score",
    "expected_playable_cards_score",
    "expected_unspent_energy_score",
    "expected_playable_frontload_score",
    "expected_playable_block_score",
    # 费用曲线：X-cost 单独记，不混入普通费用桶。
    "cost_curve_zero_share",
    "cost_curve_one_share",
    "cost_curve_two_share",
    "cost_curve_three_plus_share",
    "cost_curve_x_share",
    # 组合/体系期权价值。注意：这些只是 diagnostic，不直接当硬规则。
    "combo_component_density",
    "combo_enabler_density",
    "combo_payoff_density",
    "combo_option_value_score",
    "combo_unmet_dependency_score",
    "scaling_option_value_score",
    # 跨回合收益/延迟 payoff：某些卡当前回合看起来弱，但会在未来多个
    # 回合后兑现（power、成长、格挡体系、消耗/弃牌 payoff 等）。这些
    # 指标用于区分“值得等 payoff 成熟”的体系件和“抽到会拖慢 Act1”
    # 的孤立慢牌。
    "delayed_payoff_density",
    "delayed_enabler_density",
    "delayed_payoff_time_to_value_score",
    "delayed_payoff_maturity_score",
    "delayed_payoff_option_value_score",
    "delayed_payoff_unrealized_risk_score",
    # Diagnostics — not features, but useful for audit
    "metadata_hit_rate",
    "deck_size_raw",
)


# Tag taxonomy. Cards may carry tags via two channels:
#   1. obs["player"]["deck_cards"][i]["card_effect_profile"]["semantic_tags"]
#   2. content_registry.get_card_metadata(card_id)["semantic_tags"]
# We accept either; if neither is present, the feature is safe-zero.
_DRAW_TAGS = frozenset({"draw", "card_draw", "draw_cards"})
_ENERGY_REFUND_TAGS = frozenset({"energy", "energy_gain", "refund", "gain_energy"})
_COST_REDUCE_TAGS = frozenset({"cost_reduce", "cost_reduction", "discount"})
_RETAIN_TAGS = frozenset({"retain", "retained"})
_INNATE_TAGS = frozenset({"innate"})
_EXHAUST_TAGS = frozenset({"exhaust", "exhausts", "exhaust_self"})
_ETHEREAL_TAGS = frozenset({"ethereal"})
_COPY_REPLAY_TAGS = frozenset({"copy", "duplicate", "replay", "echo"})
_STRENGTH_TAGS = frozenset({"strength", "strength_gain", "scaling_strength"})
_DEX_TAGS = frozenset({"dexterity", "dex", "dex_gain", "scaling_dexterity"})
_POISON_TAGS = frozenset({"poison", "scaling_poison"})
_BLOCK_SCALING_TAGS = frozenset({"block_scaling", "buffer", "metallicize", "thorns"})
_AOE_TAGS = frozenset({"aoe", "all_enemies"})
_DISCARD_TAGS = frozenset({"discard", "discard_hand", "discard_pile"})
_PILE_USE_TAGS = frozenset(
    {
        "uses_exhaust_pile",
        "uses_discard_pile",
        "uses_draw_pile",
        "uses_hand_pile",
        "exhaust_payoff",
        "discard_payoff",
    }
)
_BLOCK_PAYOFF_TAGS = frozenset(
    {
        "block_payoff",
        "block_to_damage",
        "damage_from_block",
        "current_block_damage",
    }
)
_COMBO_ENABLER_TAGS = (
    _DRAW_TAGS
    | _ENERGY_REFUND_TAGS
    | _COST_REDUCE_TAGS
    | _RETAIN_TAGS
    | _EXHAUST_TAGS
    | _DISCARD_TAGS
    | _COPY_REPLAY_TAGS
)
_COMBO_PAYOFF_TAGS = (
    _STRENGTH_TAGS
    | _DEX_TAGS
    | _POISON_TAGS
    | _BLOCK_SCALING_TAGS
    | _PILE_USE_TAGS
    | _BLOCK_PAYOFF_TAGS
)
_DELAYED_ENABLER_TAGS = (
    _DRAW_TAGS
    | _ENERGY_REFUND_TAGS
    | _COST_REDUCE_TAGS
    | _RETAIN_TAGS
    | _EXHAUST_TAGS
    | _DISCARD_TAGS
    | _COPY_REPLAY_TAGS
    | frozenset({"combo_enabler", "setup", "set_up", "setup_engine"})
)
_DELAYED_PAYOFF_TAGS = (
    _STRENGTH_TAGS
    | _DEX_TAGS
    | _POISON_TAGS
    | _BLOCK_SCALING_TAGS
    | _PILE_USE_TAGS
    | _BLOCK_PAYOFF_TAGS
    | frozenset(
        {
            "combo_payoff",
            "scaling",
            "delayed_payoff",
            "future_payoff",
            "multi_turn",
            "power",
            "long_horizon",
        }
    )
)

# Curated ID-level mechanism tags for cards whose static registry/bridge
# payloads historically under-exposed dynamic value.  This avoids broad text
# regex while still letting deck-quality diagnostics understand that e.g.
# Body Slam is a block-synergy payoff rather than a zero-damage blank.
_CARD_ID_TAG_OVERRIDES: dict[str, frozenset[str]] = {
    "CARD.BODY_SLAM": frozenset({"attack", "damage", "block_payoff", "block_to_damage", "combo_payoff"}),
    "CARD.ENTRENCH": frozenset({"skill", "block_scaling", "block_payoff", "combo_enabler"}),
    "CARD.BARRICADE": frozenset({"power", "block_scaling", "block_payoff", "combo_enabler"}),
}


def _tags_for(card: dict[str, Any]) -> set[str]:
    """Union of bridge-exposed and metadata-registered semantic tags."""
    out: set[str] = set()
    card_id = str(card.get("id") or card.get("card_id") or "").strip()
    profile = card.get("card_effect_profile")
    if isinstance(profile, dict):
        for tag in profile.get("semantic_tags") or ():
            text = str(tag or "").strip().lower()
            if text:
                out.add(text)
    metadata = get_card_metadata(card_id)
    if isinstance(metadata, dict):
        for tag in metadata.get("semantic_tags") or ():
            text = str(tag or "").strip().lower()
            if text:
                out.add(text)
        for tag in metadata.get("tags") or ():
            text = str(tag or "").strip().lower()
            if text:
                out.add(text)
    for tag in _CARD_ID_TAG_OVERRIDES.get(card_id, ()):
        text = str(tag or "").strip().lower()
        if text:
            out.add(text)
    return out


def _signals_for(card: dict[str, Any]) -> dict[str, float]:
    """Read structured semantic_signals from card / metadata."""
    out: dict[str, float] = {}
    profile = card.get("card_effect_profile")
    if isinstance(profile, dict):
        signals = profile.get("semantic_signals")
        if isinstance(signals, dict):
            for k, v in signals.items():
                try:
                    out[str(k)] = float(v)
                except (TypeError, ValueError):
                    continue
    metadata = get_card_metadata(str(card.get("id") or "").strip())
    if isinstance(metadata, dict):
        signals = metadata.get("semantic_signals")
        if isinstance(signals, dict):
            for k, v in signals.items():
                # Don't overwrite bridge-supplied runtime signal — bridge wins.
                if k in out:
                    continue
                try:
                    out[str(k)] = float(v)
                except (TypeError, ValueError):
                    continue
    return out


def _safe_int(value: Any, default: int = 0) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


def _safe_float(value: Any, default: float = 0.0) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _is_x_cost(card: dict[str, Any]) -> bool:
    if card.get("x_cost"):
        return True
    cost_text = str(card.get("energy_cost_text") or "").strip().upper()
    if cost_text == "X":
        return True
    raw_cost = card.get("cost")
    if isinstance(raw_cost, str) and raw_cost.strip().upper() == "X":
        return True
    return False


def _cost_int_for(card: dict[str, Any], *, default: int = 1) -> int:
    """Best-effort non-X printed cost.

    Deck snapshots normally include ``cost``/``energy_cost``; reward payloads
    sometimes only have registry metadata.  Keep this helper conservative:
    X-cost is handled separately by ``_is_x_cost`` and unplayable negative costs
    are returned as-is for junk detection.
    """

    for key in ("cost", "energy_cost"):
        raw = card.get(key)
        if raw is None:
            continue
        if isinstance(raw, str) and raw.strip().upper() == "X":
            return int(default)
        return _safe_int(raw, default)
    metadata = get_card_metadata(str(card.get("id") or card.get("card_id") or "").strip())
    if isinstance(metadata, dict):
        for key in ("energy_cost", "cost"):
            raw = metadata.get(key)
            if raw is None:
                continue
            if isinstance(raw, str) and raw.strip().upper() == "X":
                return int(default)
            return _safe_int(raw, default)
    return int(default)


def _clamp(value: float, lo: float = 0.0, hi: float = 1.0) -> float:
    if value < lo:
        return lo
    if value > hi:
        return hi
    return value


def _estimate_playable_under_budget(
    items: list[dict[str, float | str]],
    *,
    energy_budget: float,
    mode: str = "balanced",
) -> dict[str, float]:
    """Greedy fractional estimate of cards playable from a typical hand.

    This is deliberately cheap and diagnostic-only.  It is not a combat solver;
    it estimates whether the deck's *seen cards* can actually be converted into
    played cards under a 3-ish energy budget.  ``mode='attack'`` and
    ``mode='block'`` provide separate offensive/defensive potential so mixed
    decks are not unfairly scored as zero-block merely because attacks sort
    first in a balanced pass.
    """

    remaining = max(0.0, float(energy_budget))
    out = {
        "cards": 0.0,
        "energy": 0.0,
        "attack_cards": 0.0,
        "skill_cards": 0.0,
        "power_cards": 0.0,
        "damage": 0.0,
        "block": 0.0,
    }

    def item_value(item: dict[str, float | str]) -> float:
        damage = float(item.get("damage") or 0.0)
        block = float(item.get("block") or 0.0)
        draw = float(item.get("draw") or 0.0)
        engine = float(item.get("engine") or 0.0)
        scaling = float(item.get("scaling") or 0.0)
        payoff = float(item.get("combo_payoff") or 0.0)
        if mode == "attack":
            return damage + 2.0 * payoff
        if mode == "block":
            return block + 2.0 * float(item.get("block_payoff") or 0.0)
        return damage / 6.0 + block / 5.0 + min(draw, 3.0) * 0.35 + engine * 0.45 + scaling * 0.60 + payoff * 0.35

    def sort_key(item: dict[str, float | str]) -> tuple[float, float, float]:
        cost = max(0.0, float(item.get("cost") or 0.0))
        value = item_value(item)
        density = value / max(cost, 0.25)
        # Cheapest useful cards first, then best density.  This catches bricked
        # high-cost decks without pretending to solve intent-specific tactics.
        return (cost, -density, -value)

    for item in sorted(items, key=sort_key):
        expected_count = max(0.0, float(item.get("expected_count") or 0.0))
        if expected_count <= 0.0:
            continue
        if item_value(item) <= 0.0:
            continue
        cost = max(0.0, float(item.get("cost") or 0.0))
        if cost <= 1e-9:
            take = expected_count
        else:
            if remaining <= 1e-9:
                break
            take = min(expected_count, remaining / cost)
        if take <= 0.0:
            continue
        spent = take * cost
        remaining = max(0.0, remaining - spent)

        out["cards"] += take
        out["energy"] += spent
        ctype = str(item.get("type") or "").lower()
        if ctype == "attack":
            out["attack_cards"] += take
        elif ctype == "skill":
            out["skill_cards"] += take
        elif ctype == "power":
            out["power_cards"] += take
        out["damage"] += take * max(0.0, float(item.get("damage") or 0.0))
        out["block"] += take * max(0.0, float(item.get("block") or 0.0))

    return out


def _zero_quality_features() -> dict[str, float]:
    """Return the full key set with safe-zero values (empty deck or missing source)."""
    out: dict[str, float] = {key: 0.0 for key in DECK_QUALITY_V2_KEYS}
    # build_gap_risk-style: empty deck means max risk on readiness
    out["pollution_score"] = 0.0
    out["consistency_score"] = 0.0
    out["elite_readiness_score"] = 0.0
    out["boss_readiness_score"] = 0.0
    return out


def deck_quality_v2(deck_cards: Iterable[Any] | None) -> dict[str, float]:
    """Compute extended deck-quality features for the given deck.

    ``deck_cards`` is the list found at ``obs["player"]["deck_cards"]``.
    Returns a dict with keys exactly ``DECK_QUALITY_V2_KEYS``. Model-facing
    score/density values are clamped to safe ranges; raw/count diagnostics
    remain finite floats so operators can read real per-energy and
    expected-hand magnitudes.
    """
    cards = [c for c in (deck_cards or []) if isinstance(c, dict)]
    if not cards:
        return _zero_quality_features()

    deck_size = len(cards)
    deck_size_safe = max(deck_size, 1)
    deck_size_norm = _clamp(deck_size / 40.0)

    metadata_hits = 0
    upgraded = 0
    attack = 0
    skill = 0
    power = 0
    curse = 0
    status = 0
    high_cost = 0
    zero_cost = 0
    x_cost = 0
    one_cost = 0
    two_cost = 0
    three_plus_cost = 0

    cost_sum_for_avg = 0.0
    cost_sum_count = 0  # excludes X-cost
    energy_units = 0.0  # capped sum of cost (treat 0-cost as 1 for ratios)
    damage_total = 0.0
    block_total = 0.0
    draw_total = 0.0

    x_cost_damage = 0.0
    x_cost_block = 0.0

    draw_cards = 0
    engine_cards = 0
    energy_refund = 0
    cost_reduce = 0
    retain = 0
    innate = 0
    exhaust = 0
    ethereal = 0
    copy_replay = 0

    str_scaling = 0
    dex_scaling = 0
    poison_scaling = 0
    block_scaling = 0
    power_scaling = 0
    scaling_cards = 0
    combo_components = 0
    combo_enablers = 0
    combo_payoffs = 0
    block_payoffs = 0
    delayed_enablers = 0
    delayed_payoffs = 0

    duplicate_counter: dict[str, int] = {}
    playable_items: list[dict[str, float | str]] = []

    for card in cards:
        card_id = str(card.get("id") or "").strip()
        duplicate_counter[card_id] = duplicate_counter.get(card_id, 0) + 1

        metadata_hit = bool(get_card_metadata(card_id))
        if metadata_hit:
            metadata_hits += 1

        upgrade_level = _safe_int(card.get("upgrade_level"))
        if upgrade_level > 0 or card.get("upgraded"):
            upgraded += 1

        is_engine_card = False
        is_scaling_card = False
        card_type = str(card.get("type") or "").strip().lower()
        if card_type == "attack":
            attack += 1
        elif card_type == "skill":
            skill += 1
        elif card_type == "power":
            power += 1
            power_scaling += 1
            is_scaling_card = True
        elif card_type == "curse":
            curse += 1
        elif card_type == "status":
            status += 1

        is_x = _is_x_cost(card)
        if is_x:
            x_cost += 1
        else:
            cost_int = _cost_int_for(card, default=1)
            # Unplayable cards (curse/status cost=-1) should not improve cost
            # curve / avg-cost diagnostics.  They are tracked via pollution.
            if cost_int >= 0:
                cost_sum_for_avg += float(cost_int)
                cost_sum_count += 1
            if cost_int >= 2:
                high_cost += 1
            if cost_int == 0:
                zero_cost += 1
            elif cost_int == 1:
                one_cost += 1
            elif cost_int == 2:
                two_cost += 1
            elif cost_int >= 3:
                three_plus_cost += 1
            if cost_int >= 0:
                energy_units += max(float(cost_int), 1.0)

        signals = _signals_for(card)
        damage = _safe_float(signals.get("damage"))
        block = _safe_float(signals.get("block"))
        draw = _safe_float(signals.get("draw"))
        if damage > 0:
            if is_x:
                x_cost_damage += damage
            else:
                damage_total += damage
        if block > 0:
            if is_x:
                x_cost_block += block
            else:
                block_total += block

        tags = _tags_for(card)
        # P1-7 (review fix): some draw cards expose only the tag, not a
        # numeric ``draw`` semantic_signal. Fall back to tag detection so
        # those cards still raise draw_density / draw_engine_score.
        if draw > 0:
            draw_total += draw
            draw_cards += 1
            is_engine_card = True
        elif tags & _DRAW_TAGS:
            draw_total += 1.0
            draw_cards += 1
            is_engine_card = True
        if tags & _ENERGY_REFUND_TAGS:
            energy_refund += 1
            is_engine_card = True
        if tags & _COST_REDUCE_TAGS:
            cost_reduce += 1
            is_engine_card = True
        if tags & _RETAIN_TAGS:
            retain += 1
            is_engine_card = True
        if tags & _INNATE_TAGS:
            innate += 1
        if tags & _EXHAUST_TAGS:
            exhaust += 1
        if tags & _ETHEREAL_TAGS:
            ethereal += 1
        if tags & _COPY_REPLAY_TAGS:
            copy_replay += 1
            is_engine_card = True
        if tags & _STRENGTH_TAGS:
            str_scaling += 1
            is_scaling_card = True
        if tags & _DEX_TAGS:
            dex_scaling += 1
            is_scaling_card = True
        if tags & _POISON_TAGS:
            poison_scaling += 1
            is_scaling_card = True
        if tags & _BLOCK_SCALING_TAGS:
            block_scaling += 1
            is_scaling_card = True
        is_combo_enabler = bool(
            tags & _COMBO_ENABLER_TAGS
            or draw > 0.0
            or _safe_float(signals.get("energy_gain")) > 0.0
        )
        is_combo_payoff = bool(
            tags & _COMBO_PAYOFF_TAGS
            or tags & {"combo_payoff", "scaling"}
            or _safe_float(signals.get("strength")) > 0.0
            or _safe_float(signals.get("dexterity")) > 0.0
            or _safe_float(signals.get("poison")) > 0.0
            or card_type == "power"
        )
        is_block_payoff = bool(tags & _BLOCK_PAYOFF_TAGS)
        is_delayed_enabler = bool(is_combo_enabler or tags & _DELAYED_ENABLER_TAGS)
        is_delayed_payoff = bool(
            is_combo_payoff
            or is_scaling_card
            or card_type == "power"
            or tags & _DELAYED_PAYOFF_TAGS
            or _safe_float(signals.get("strength")) > 0.0
            or _safe_float(signals.get("dexterity")) > 0.0
            or _safe_float(signals.get("poison")) > 0.0
        )
        if is_combo_enabler:
            combo_enablers += 1
        if is_combo_payoff:
            combo_payoffs += 1
        if is_combo_enabler or is_combo_payoff:
            combo_components += 1
        if is_block_payoff:
            block_payoffs += 1
        if is_delayed_enabler:
            delayed_enablers += 1
        if is_delayed_payoff:
            delayed_payoffs += 1
        if is_engine_card:
            engine_cards += 1
        if is_scaling_card:
            scaling_cards += 1

        cost_for_play = 3.0 if is_x else float(max(_cost_int_for(card, default=1), 0))
        is_junk = card_type in {"curse", "status"} or (not is_x and _cost_int_for(card, default=1) < 0)
        useful_signal = (
            damage > 0.0
            or block > 0.0
            or draw > 0.0
            or is_engine_card
            or is_scaling_card
            or is_combo_payoff
        )
        if not is_junk and useful_signal:
            playable_items.append(
                {
                    "type": card_type,
                    "cost": cost_for_play,
                    "damage": max(0.0, float(damage)),
                    "block": max(0.0, float(block)),
                    "draw": max(0.0, float(draw)),
                    "engine": 1.0 if is_engine_card or is_combo_enabler else 0.0,
                    "scaling": 1.0 if is_scaling_card else 0.0,
                    "combo_payoff": 1.0 if is_combo_payoff else 0.0,
                    "block_payoff": 1.0 if is_block_payoff else 0.0,
                    # Filled after hand_scale is known below.
                    "expected_count": 0.0,
                }
            )

    # Densities
    upgraded_ratio = upgraded / deck_size_safe
    attack_density = attack / deck_size_safe
    skill_density = skill / deck_size_safe
    power_density = power / deck_size_safe
    curse_density = curse / deck_size_safe
    status_density = status / deck_size_safe
    high_cost_density = high_cost / deck_size_safe
    zero_cost_density = zero_cost / deck_size_safe
    x_cost_density = x_cost / deck_size_safe
    cost_curve_zero_share = zero_cost / deck_size_safe
    cost_curve_one_share = one_cost / deck_size_safe
    cost_curve_two_share = two_cost / deck_size_safe
    cost_curve_three_plus_share = three_plus_cost / deck_size_safe
    cost_curve_x_share = x_cost / deck_size_safe
    draw_density = draw_cards / deck_size_safe
    energy_refund_density = energy_refund / deck_size_safe
    cost_reduce_density = cost_reduce / deck_size_safe
    retain_density = retain / deck_size_safe
    innate_density = innate / deck_size_safe
    exhaust_density = exhaust / deck_size_safe
    ethereal_density = ethereal / deck_size_safe
    copy_replay_density = copy_replay / deck_size_safe
    strength_scaling_density = str_scaling / deck_size_safe
    dex_scaling_density = dex_scaling / deck_size_safe
    poison_scaling_density = poison_scaling / deck_size_safe
    block_scaling_density = block_scaling / deck_size_safe
    power_scaling_density = power_scaling / deck_size_safe
    combo_component_density = combo_components / deck_size_safe
    combo_enabler_density = combo_enablers / deck_size_safe
    combo_payoff_density = combo_payoffs / deck_size_safe
    delayed_enabler_density = delayed_enablers / deck_size_safe
    delayed_payoff_density = delayed_payoffs / deck_size_safe

    avg_cost = cost_sum_for_avg / max(cost_sum_count, 1) if cost_sum_count else 0.0
    avg_damage_per_energy = damage_total / max(energy_units, 1.0)
    avg_block_per_energy = block_total / max(energy_units, 1.0)
    expected_extra_draw_per_turn = draw_total / deck_size_safe  # rough proxy
    base_hand_size = min(5.0, float(deck_size))
    # Operator-facing "cards seen this turn" estimate.  Keep the legacy
    # normalised ``expected_extra_draw_per_turn`` proxy unchanged above, but add
    # a more literal one-hop estimate for post-mortem deck quality:
    # expected draw effects in the opening hand = Σdraw * P(card in first 5).
    raw_expected_extra_draw_per_turn = max(0.0, draw_total * base_hand_size / deck_size_safe)
    raw_expected_extra_draw_per_turn = min(
        raw_expected_extra_draw_per_turn,
        max(0.0, float(deck_size) - base_hand_size),
    )
    raw_expected_cards_seen_per_turn = min(
        float(deck_size),
        base_hand_size + raw_expected_extra_draw_per_turn,
    )
    hand_scale = raw_expected_cards_seen_per_turn / deck_size_safe
    expected_hand_attack_cards = float(attack) * hand_scale
    expected_hand_skill_cards = float(skill) * hand_scale
    expected_hand_power_cards = float(power) * hand_scale
    expected_hand_curse_cards = float(curse) * hand_scale
    expected_hand_status_cards = float(status) * hand_scale
    expected_hand_draw_cards = float(draw_cards) * hand_scale
    expected_hand_engine_cards = float(engine_cards) * hand_scale
    expected_hand_scaling_cards = float(scaling_cards) * hand_scale
    expected_hand_unplayable_cards = (float(curse) + float(status)) * hand_scale
    expected_hand_attack_damage_per_turn = damage_total * hand_scale
    expected_hand_block_per_turn = block_total * hand_scale
    scaling_card_signal_count = float(str_scaling + dex_scaling + poison_scaling + block_scaling + power_scaling)
    hand_den = max(raw_expected_cards_seen_per_turn, 1e-9)
    expected_hand_attack_share = _clamp(expected_hand_attack_cards / hand_den)
    expected_hand_skill_share = _clamp(expected_hand_skill_cards / hand_den)
    expected_hand_power_share = _clamp(expected_hand_power_cards / hand_den)
    expected_hand_junk_share = _clamp(expected_hand_unplayable_cards / hand_den)
    expected_hand_frontload_score = _clamp(expected_hand_attack_damage_per_turn / 35.0)
    expected_hand_block_score = _clamp(expected_hand_block_per_turn / 35.0)
    expected_hand_draw_score = _clamp(raw_expected_extra_draw_per_turn / 2.0)
    expected_hand_engine_score = _clamp(expected_hand_engine_cards / 2.0)
    expected_hand_scaling_score = _clamp((scaling_card_signal_count * hand_scale) / 2.0)
    expected_hand_pollution_score = _clamp(
        ((float(curse) + float(status)) * hand_scale) / 2.0
    )
    expected_hand_useful_quality_score = _clamp(
        0.35 * expected_hand_frontload_score
        + 0.30 * expected_hand_block_score
        + 0.20 * expected_hand_draw_score
        + 0.15 * expected_hand_scaling_score
        - 0.35 * expected_hand_pollution_score
    )
    quality_den = max(
        expected_hand_frontload_score
        + expected_hand_block_score
        + expected_hand_draw_score
        + expected_hand_scaling_score
        + expected_hand_pollution_score,
        1e-9,
    )
    expected_hand_quality_attack_share = _clamp(expected_hand_frontload_score / quality_den)
    expected_hand_quality_block_share = _clamp(expected_hand_block_score / quality_den)
    expected_hand_quality_draw_share = _clamp(expected_hand_draw_score / quality_den)
    expected_hand_quality_scaling_share = _clamp(expected_hand_scaling_score / quality_den)
    expected_hand_quality_pollution_share = _clamp(expected_hand_pollution_score / quality_den)

    for item in playable_items:
        item["expected_count"] = hand_scale

    raw_expected_energy_budget_per_turn = 3.0 + min(
        1.5,
        hand_scale * (0.75 * float(energy_refund) + 0.35 * float(cost_reduce)),
    )
    balanced_playable = _estimate_playable_under_budget(
        playable_items,
        energy_budget=raw_expected_energy_budget_per_turn,
        mode="balanced",
    )
    attack_playable = _estimate_playable_under_budget(
        [item for item in playable_items if str(item.get("type") or "").lower() == "attack"],
        energy_budget=raw_expected_energy_budget_per_turn,
        mode="attack",
    )
    block_playable = _estimate_playable_under_budget(
        [item for item in playable_items if float(item.get("block") or 0.0) > 0.0],
        energy_budget=raw_expected_energy_budget_per_turn,
        mode="block",
    )
    raw_expected_playable_cards_per_turn = float(balanced_playable["cards"])
    raw_expected_playable_energy_spent_per_turn = float(balanced_playable["energy"])
    raw_expected_unspent_energy_per_turn = max(
        0.0,
        raw_expected_energy_budget_per_turn - raw_expected_playable_energy_spent_per_turn,
    )
    raw_expected_playable_attack_cards_per_turn = float(attack_playable["attack_cards"])
    raw_expected_playable_skill_cards_per_turn = float(balanced_playable["skill_cards"])
    raw_expected_playable_power_cards_per_turn = float(balanced_playable["power_cards"])
    raw_expected_playable_attack_damage_per_turn = float(attack_playable["damage"])
    raw_expected_playable_block_per_turn = float(block_playable["block"])
    expected_energy_utilization_score = _clamp(
        raw_expected_playable_energy_spent_per_turn / max(raw_expected_energy_budget_per_turn, 1e-9)
    )
    expected_playable_cards_score = _clamp(raw_expected_playable_cards_per_turn / 4.0)
    expected_unspent_energy_score = _clamp(raw_expected_unspent_energy_per_turn / max(raw_expected_energy_budget_per_turn, 1e-9))
    expected_playable_frontload_score = _clamp(raw_expected_playable_attack_damage_per_turn / 30.0)
    expected_playable_block_score = _clamp(raw_expected_playable_block_per_turn / 25.0)

    # Aggregate scores. All clamped to [0, 1] so they fit cleanly in a model
    # input even if the underlying density is unusual.
    pollution = curse_density + 0.5 * status_density
    pollution_score = _clamp(pollution * 1.5)  # punish curse > status

    # P1-8 (review fix): consistency = "how concentrated the deck is on
    # a few card_ids". A deck where every card is the same id scores
    # near 1.0 (high consistency = predictable draws); a deck where every
    # card is unique scores near 0.0 (low consistency = scattered turns).
    # The earlier docstring described the inverse — fixed to match the
    # actual formula.
    if duplicate_counter:
        unique_count = len(duplicate_counter)
        consistency_score = _clamp(1.0 - (unique_count - 1) / max(deck_size, 1))
    else:
        consistency_score = 0.0

    # frontload: damage output per energy normalised to a reasonable ceiling
    # (≈ 6 damage per 1 energy = strike). Score 1.0 at 7+ avg damage/energy.
    frontload_score = _clamp(avg_damage_per_energy / 7.0)
    block_score = _clamp(avg_block_per_energy / 7.0)

    # scaling: any scaling source contributes; cap at 1.0
    scaling_total = (
        strength_scaling_density
        + dex_scaling_density
        + poison_scaling_density
        + block_scaling_density
        + power_scaling_density
    )
    scaling_score = _clamp(scaling_total / 0.4)  # ~40% of deck is "max scaling"

    # draw / energy engine
    draw_engine_score = _clamp(
        draw_density * 1.5 + cost_reduce_density * 0.5 + retain_density * 0.3
    )
    energy_engine_score = _clamp(
        energy_refund_density * 2.0 + cost_reduce_density * 1.0
    )

    combo_balance = min(combo_enabler_density, combo_payoff_density)
    combo_imbalance = abs(float(combo_enablers) - float(combo_payoffs)) / max(float(combo_components), 1.0)
    # Body-Slam-like payoffs should become valuable only if the deck also has
    # block production.  That is exactly the “future combo option” signal the
    # user called out: valuable as a component, but risky if the partner never
    # appears.
    block_payoff_coverage = min(block_score, float(block_payoffs) / deck_size_safe * 4.0)
    combo_option_value_score = _clamp(
        3.0 * combo_balance
        + 0.35 * block_payoff_coverage
        + 0.20 * draw_engine_score
        + 0.15 * energy_engine_score
    )
    combo_unmet_dependency_score = _clamp(
        combo_imbalance * (1.0 if combo_components > 0 else 0.0)
        + max(0.0, combo_payoff_density - combo_enabler_density) * 1.5
    )
    scaling_option_value_score = _clamp(
        0.60 * scaling_score
        + 0.20 * draw_engine_score
        + 0.20 * combo_option_value_score
    )
    delayed_balance = min(delayed_enabler_density, delayed_payoff_density)
    delayed_imbalance = abs(float(delayed_enablers) - float(delayed_payoffs)) / max(
        float(delayed_enablers + delayed_payoffs),
        1.0,
    )
    # How quickly a delayed payoff can become real in ordinary fights. Draw,
    # retain/innate, and energy help slow powers/setup cards matter before the
    # deck bleeds too much HP.  This is intentionally a rough diagnostic, not a
    # hard card tier list.
    delayed_payoff_time_to_value_score = _clamp(
        0.45 * draw_engine_score
        + 0.20 * energy_engine_score
        + 0.15 * _clamp(retain_density * 3.0)
        + 0.10 * _clamp(innate_density * 3.0)
        + 0.10 * consistency_score
    )
    # Maturity asks whether the current deck can survive and cycle long enough
    # to realize multi-turn value.  Block/frontload are included so that slow
    # payoff cards do not look good in a deck that is already dying in hallway
    # fights.
    delayed_payoff_maturity_score = _clamp(
        0.25 * delayed_payoff_time_to_value_score
        + 0.20 * expected_playable_frontload_score
        + 0.20 * expected_playable_block_score
        + 0.15 * energy_engine_score
        + 0.10 * combo_option_value_score
        + 0.10 * scaling_score
    )
    delayed_payoff_option_value_score = _clamp(
        1.80 * delayed_balance
        + 0.30 * scaling_score
        + 0.25 * combo_option_value_score
        + 0.20 * delayed_payoff_time_to_value_score
        + 0.15 * delayed_payoff_maturity_score
    )
    delayed_payoff_unrealized_risk_score = _clamp(
        delayed_imbalance * (1.0 if (delayed_enablers + delayed_payoffs) > 0 else 0.0)
        + max(0.0, delayed_payoff_density - delayed_enabler_density) * 1.25
        + delayed_payoff_density * (1.0 - delayed_payoff_maturity_score) * 0.60
        + pollution_score * 0.25
    )

    # readiness: blends defensive + offensive + draw, penalised by pollution
    elite_readiness_score = _clamp(
        0.40 * frontload_score
        + 0.30 * block_score
        + 0.20 * draw_engine_score
        + 0.10 * energy_engine_score
        - 0.50 * pollution_score
    )
    # Boss-readiness additionally needs scaling / multi-turn output.
    boss_readiness_score = _clamp(
        0.30 * frontload_score
        + 0.25 * block_score
        + 0.25 * scaling_score
        + 0.10 * draw_engine_score
        + 0.10 * energy_engine_score
        - 0.40 * pollution_score
    )

    metadata_hit_rate = metadata_hits / deck_size_safe

    return {
        "deck_size_norm": deck_size_norm,
        "upgraded_ratio": upgraded_ratio,
        "attack_density": attack_density,
        "skill_density": skill_density,
        "power_density": power_density,
        "curse_density": curse_density,
        "status_density": status_density,
        "avg_cost": _clamp(avg_cost / 4.0),  # normalise to [0,1]; 4 = X-ish bound
        "avg_damage_per_energy": _clamp(avg_damage_per_energy / 12.0),
        "avg_block_per_energy": _clamp(avg_block_per_energy / 12.0),
        "raw_avg_damage_per_energy": float(avg_damage_per_energy),
        "raw_avg_block_per_energy": float(avg_block_per_energy),
        "high_cost_density": high_cost_density,
        "zero_cost_density": zero_cost_density,
        "x_cost_density": x_cost_density,
        "x_cost_damage_potential": _clamp(x_cost_damage / 30.0),
        "x_cost_block_potential": _clamp(x_cost_block / 30.0),
        "draw_density": draw_density,
        "expected_extra_draw_per_turn": _clamp(expected_extra_draw_per_turn / 2.0),
        "raw_expected_extra_draw_per_turn": float(raw_expected_extra_draw_per_turn),
        "raw_expected_cards_seen_per_turn": float(raw_expected_cards_seen_per_turn),
        "energy_refund_density": energy_refund_density,
        "cost_reduce_density": cost_reduce_density,
        "retain_density": retain_density,
        "innate_density": innate_density,
        "exhaust_density": exhaust_density,
        "ethereal_density": ethereal_density,
        "copy_replay_density": copy_replay_density,
        "strength_scaling_density": strength_scaling_density,
        "dex_scaling_density": dex_scaling_density,
        "poison_scaling_density": poison_scaling_density,
        "block_scaling_density": block_scaling_density,
        "power_scaling_density": power_scaling_density,
        "frontload_score": frontload_score,
        "block_score": block_score,
        "scaling_score": scaling_score,
        "draw_engine_score": draw_engine_score,
        "energy_engine_score": energy_engine_score,
        "pollution_score": pollution_score,
        "consistency_score": consistency_score,
        "elite_readiness_score": elite_readiness_score,
        "boss_readiness_score": boss_readiness_score,
        "expected_hand_attack_cards": float(expected_hand_attack_cards),
        "expected_hand_skill_cards": float(expected_hand_skill_cards),
        "expected_hand_power_cards": float(expected_hand_power_cards),
        "expected_hand_curse_cards": float(expected_hand_curse_cards),
        "expected_hand_status_cards": float(expected_hand_status_cards),
        "expected_hand_draw_cards": float(expected_hand_draw_cards),
        "expected_hand_engine_cards": float(expected_hand_engine_cards),
        "expected_hand_scaling_cards": float(expected_hand_scaling_cards),
        "expected_hand_unplayable_cards": float(expected_hand_unplayable_cards),
        "expected_hand_attack_damage_per_turn": float(expected_hand_attack_damage_per_turn),
        "expected_hand_block_per_turn": float(expected_hand_block_per_turn),
        "expected_hand_attack_share": expected_hand_attack_share,
        "expected_hand_skill_share": expected_hand_skill_share,
        "expected_hand_power_share": expected_hand_power_share,
        "expected_hand_junk_share": expected_hand_junk_share,
        "expected_hand_frontload_score": expected_hand_frontload_score,
        "expected_hand_block_score": expected_hand_block_score,
        "expected_hand_draw_score": expected_hand_draw_score,
        "expected_hand_engine_score": expected_hand_engine_score,
        "expected_hand_scaling_score": expected_hand_scaling_score,
        "expected_hand_pollution_score": expected_hand_pollution_score,
        "expected_hand_useful_quality_score": expected_hand_useful_quality_score,
        "expected_hand_quality_attack_share": expected_hand_quality_attack_share,
        "expected_hand_quality_block_share": expected_hand_quality_block_share,
        "expected_hand_quality_draw_share": expected_hand_quality_draw_share,
        "expected_hand_quality_scaling_share": expected_hand_quality_scaling_share,
        "expected_hand_quality_pollution_share": expected_hand_quality_pollution_share,
        "raw_expected_energy_budget_per_turn": float(raw_expected_energy_budget_per_turn),
        "raw_expected_playable_cards_per_turn": float(raw_expected_playable_cards_per_turn),
        "raw_expected_playable_energy_spent_per_turn": float(raw_expected_playable_energy_spent_per_turn),
        "raw_expected_unspent_energy_per_turn": float(raw_expected_unspent_energy_per_turn),
        "raw_expected_playable_attack_cards_per_turn": float(raw_expected_playable_attack_cards_per_turn),
        "raw_expected_playable_skill_cards_per_turn": float(raw_expected_playable_skill_cards_per_turn),
        "raw_expected_playable_power_cards_per_turn": float(raw_expected_playable_power_cards_per_turn),
        "raw_expected_playable_attack_damage_per_turn": float(raw_expected_playable_attack_damage_per_turn),
        "raw_expected_playable_block_per_turn": float(raw_expected_playable_block_per_turn),
        "expected_energy_utilization_score": expected_energy_utilization_score,
        "expected_playable_cards_score": expected_playable_cards_score,
        "expected_unspent_energy_score": expected_unspent_energy_score,
        "expected_playable_frontload_score": expected_playable_frontload_score,
        "expected_playable_block_score": expected_playable_block_score,
        "cost_curve_zero_share": cost_curve_zero_share,
        "cost_curve_one_share": cost_curve_one_share,
        "cost_curve_two_share": cost_curve_two_share,
        "cost_curve_three_plus_share": cost_curve_three_plus_share,
        "cost_curve_x_share": cost_curve_x_share,
        "combo_component_density": combo_component_density,
        "combo_enabler_density": combo_enabler_density,
        "combo_payoff_density": combo_payoff_density,
        "combo_option_value_score": combo_option_value_score,
        "combo_unmet_dependency_score": combo_unmet_dependency_score,
        "scaling_option_value_score": scaling_option_value_score,
        "delayed_payoff_density": delayed_payoff_density,
        "delayed_enabler_density": delayed_enabler_density,
        "delayed_payoff_time_to_value_score": delayed_payoff_time_to_value_score,
        "delayed_payoff_maturity_score": delayed_payoff_maturity_score,
        "delayed_payoff_option_value_score": delayed_payoff_option_value_score,
        "delayed_payoff_unrealized_risk_score": delayed_payoff_unrealized_risk_score,
        "metadata_hit_rate": metadata_hit_rate,
        "deck_size_raw": float(deck_size),
    }


def deck_quality_v2_from_obs(obs: Any) -> dict[str, float]:
    """Convenience: extract deck_cards from obs and compute features."""
    if not isinstance(obs, dict):
        return _zero_quality_features()
    player = obs.get("player") if isinstance(obs.get("player"), dict) else {}
    deck_cards = player.get("deck_cards") if isinstance(player, dict) else None
    return deck_quality_v2(deck_cards)
