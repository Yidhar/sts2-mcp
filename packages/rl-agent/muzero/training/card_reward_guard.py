"""Card-reward anti-skip guard driven by deck-quality deficits.

The Act1 full-run regression we observed was not "bad boss play" first.  The
agent frequently skipped four to ten card rewards in a row and then died with a
thin deck whose block/draw quality was too low.  This module is a narrow,
fail-open build-domain guard for that exact failure mode:

* only runs on card-reward surfaces;
* only intercepts skip/proceed/leave/cancel choices;
* only forces a pick when the current Act1 deck is thin or objectively weak;
* never forces curses/status/junk cards;
* records explicit telemetry for the three deck-quality axes the operator cares
  about:
    1. per-energy attack/block,
    2. expected draw / cards seen per turn,
    3. expected hand composition and quality.

It intentionally does not change replay schemas or model input tensors.  It is a
runtime safety/behavior guard plus metrics, so it can be removed once the policy
has learned not to pathologically skip early card rewards.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any

from content_registry import get_card_metadata
from muzero.diagnostics.deck_build_metrics import (
    compute_deck_quality_summary,
    extract_deck_cards_from_obs_like,
)


CARD_REWARD_GUARD_METRIC_KEYS: tuple[str, ...] = (
    "card_reward_guard_context",
    "card_reward_guard_applicable",
    "card_reward_guard_selected_skip",
    "card_reward_guard_selected_pick",
    "card_reward_guard_pick_available",
    "card_reward_guard_useful_candidate_available",
    "card_reward_guard_no_useful_candidate",
    "card_reward_guard_applied",
    "card_reward_guard_override",
    "card_reward_guard_skip_blocked",
    "card_reward_guard_alignment_error",
    "card_reward_guard_invalid_obs",
    "card_reward_guard_not_act1",
    "card_reward_guard_deck_size",
    "card_reward_guard_best_score",
    "card_reward_guard_selected_score",
    # Candidate-level combo valuation.  Deck-level combo scores tell us
    # whether the current deck is generally coherent; these fields tell us why
    # the *best offered card* was considered useful or dangerous.  This is the
    # important distinction for future combo pieces: a card can be a missing
    # payoff/enabler, a speculative option with enough runway, or an orphan
    # component that should not be forced over skip.
    "card_reward_guard_best_standalone_value",
    "card_reward_guard_best_combo_current_fit",
    "card_reward_guard_best_combo_missing_piece_fit",
    "card_reward_guard_best_combo_speculative_option",
    "card_reward_guard_best_combo_orphan_risk",
    "card_reward_guard_best_combo_final_option_value",
    # Dynamic anti-skip threshold.  A fixed 0.45 threshold was too strict for
    # live full-run candidate score scale: early thin decks kept skipping
    # moderate useful cards, while late orphan combo pieces still need to be
    # rejected.  These metrics make the threshold/expiration decision visible.
    "card_reward_guard_dynamic_min_useful_score",
    "card_reward_guard_threshold_relaxed",
    "card_reward_guard_threshold_raised_near_boss",
    "card_reward_guard_orphan_reject",
    "card_reward_guard_speculative_option_accepted",
    "card_reward_guard_top_candidate_rejected",
    "card_reward_guard_rejected_candidate_count",
    # Operator-facing deck-quality axes.
    "card_reward_guard_damage_per_energy",
    "card_reward_guard_block_per_energy",
    "card_reward_guard_expected_extra_draw",
    "card_reward_guard_expected_cards_seen",
    "card_reward_guard_expected_hand_attack_damage",
    "card_reward_guard_expected_hand_block",
    "card_reward_guard_expected_hand_attack_share",
    "card_reward_guard_expected_hand_skill_share",
    "card_reward_guard_expected_hand_power_share",
    "card_reward_guard_expected_hand_junk_share",
    "card_reward_guard_expected_hand_useful_quality",
    "card_reward_guard_expected_playable_cards",
    "card_reward_guard_expected_playable_attack_damage",
    "card_reward_guard_expected_playable_block",
    "card_reward_guard_expected_energy_utilization",
    "card_reward_guard_combo_option_value",
    "card_reward_guard_combo_unmet_dependency",
    "card_reward_guard_future_combo_window",
    "card_reward_guard_future_combo_floor_window",
    "card_reward_guard_future_combo_route_window",
    "card_reward_guard_future_reward_opportunity",
    "card_reward_guard_future_runway_observed",
    "card_reward_guard_delayed_payoff_option_value",
    "card_reward_guard_delayed_payoff_unrealized_risk",
    "card_reward_guard_delayed_payoff_maturity",
    "card_reward_guard_delayed_payoff_time_to_value",
    # Deficits / reasons.
    "card_reward_guard_low_block",
    "card_reward_guard_low_draw",
    "card_reward_guard_low_attack",
    "card_reward_guard_low_scaling",
    "card_reward_guard_reason_thin_deck",
    "card_reward_guard_reason_low_block",
    "card_reward_guard_reason_low_draw",
    "card_reward_guard_reason_low_attack",
    "card_reward_guard_reason_low_scaling",
)


_PICK_SELECTIONS = {"pick", "take", "choose", "select", "add", "claim"}
_SKIP_SELECTIONS = {"skip", "proceed", "leave", "cancel", "close", "none"}
_JUNK_TYPES = {"curse", "status"}
_DRAW_TAGS = {"draw", "card_draw", "draw_cards"}
_ENGINE_TAGS = {"energy", "energy_gain", "refund", "gain_energy", "cost_reduce", "cost_reduction", "discount", "retain", "retained"}
_SCALING_TAGS = {
    "strength",
    "strength_gain",
    "scaling_strength",
    "dexterity",
    "dex",
    "dex_gain",
    "scaling_dexterity",
    "poison",
    "scaling_poison",
    "block_scaling",
    "buffer",
    "metallicize",
    "thorns",
    "scaling",
}
_BLOCK_PAYOFF_TAGS = {"block_payoff", "block_to_damage", "damage_from_block", "current_block_damage"}
_COMBO_PAYOFF_TAGS = _SCALING_TAGS | _BLOCK_PAYOFF_TAGS | {
    "combo_payoff",
    "uses_exhaust_pile",
    "uses_discard_pile",
    "exhaust_payoff",
    "discard_payoff",
}
_COMBO_ENABLER_TAGS = _DRAW_TAGS | _ENGINE_TAGS | {
    "exhaust",
    "discard",
    "discard_hand",
    "copy",
    "duplicate",
    "replay",
    "echo",
}
_CARD_ID_TAG_OVERRIDES: dict[str, frozenset[str]] = {
    "CARD.BODY_SLAM": frozenset({"attack", "damage", "block_payoff", "block_to_damage", "combo_payoff"}),
    "CARD.ENTRENCH": frozenset({"skill", "block_scaling", "block_payoff", "combo_enabler"}),
    "CARD.BARRICADE": frozenset({"power", "block_scaling", "block_payoff", "combo_enabler"}),
}


@dataclass(frozen=True)
class _DeckDeficits:
    thin_deck: bool
    low_attack: bool
    low_block: bool
    low_draw: bool
    low_scaling: bool

    @property
    def any(self) -> bool:
        return self.thin_deck or self.low_attack or self.low_block or self.low_draw or self.low_scaling


@dataclass(frozen=True)
class _CandidateScore:
    index: int
    score: float
    card: dict[str, Any]
    junk: bool
    standalone_value: float = 0.0
    combo_current_fit: float = 0.0
    combo_missing_piece_fit: float = 0.0
    combo_speculative_option: float = 0.0
    combo_orphan_risk: float = 0.0
    combo_final_option_value: float = 0.0


@dataclass(frozen=True)
class _FutureComboRunway:
    score: float
    floor_score: float
    route_score: float
    reward_opportunity: float
    observed: bool


@dataclass(frozen=True)
class _CandidateComboFit:
    standalone_value: float
    current_fit: float
    missing_piece_fit: float
    speculative_option: float
    orphan_risk: float
    final_option_value: float


def card_reward_guard_metric_keys() -> tuple[str, ...]:
    """Return telemetry keys so callers can initialize/search-map them."""

    return CARD_REWARD_GUARD_METRIC_KEYS


def _finite_float(value: Any, default: float = 0.0) -> float:
    try:
        out = float(value)
    except (TypeError, ValueError):
        return float(default)
    return out if math.isfinite(out) else float(default)


def _safe_int(value: Any, default: int = 0) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return int(default)


def _clamp01(value: Any) -> float:
    return max(0.0, min(_finite_float(value, 0.0), 1.0))


def _lower(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, dict):
        parts: list[str] = []
        for key in (
            "action_id",
            "kind",
            "action_type",
            "surface",
            "selection",
            "selection_action",
            "title",
            "label",
            "name",
            "canonical_text",
            "description",
            "reward_type",
            "card_id",
        ):
            raw = value.get(key)
            if raw is not None:
                parts.append(str(raw))
        for nested_key in ("payload", "semantic", "reward", "option", "card"):
            nested = value.get(nested_key)
            if isinstance(nested, dict):
                for key in (
                    "action_id",
                    "kind",
                    "family",
                    "domain",
                    "semantic_key",
                    "type",
                    "reward_type",
                    "id",
                    "title",
                    "name",
                    "selection",
                    "label",
                ):
                    raw = nested.get(key)
                    if raw is not None:
                        parts.append(str(raw))
        return " ".join(parts).strip().lower()
    return str(value).strip().lower()


def _containers(action: Any) -> list[dict[str, Any]]:
    if not isinstance(action, dict):
        return []
    out = [action]
    payload = action.get("payload")
    if isinstance(payload, dict):
        out.append(payload)
    return out


def _semantic_family(action: Any) -> str:
    for container in _containers(action):
        semantic = container.get("semantic") if isinstance(container.get("semantic"), dict) else {}
        family = str(semantic.get("family") or "").strip().lower()
        if family:
            return family
    return ""


def _reward_type(action: Any) -> str:
    for container in _containers(action):
        reward = container.get("reward") if isinstance(container.get("reward"), dict) else {}
        text = str(reward.get("type") or container.get("reward_type") or "").strip().lower()
        if text:
            return text
    return ""


def _is_card_reward_action(action: Any) -> bool:
    text = _lower(action)
    family = _semantic_family(action)
    for container in _containers(action):
        kind = str(container.get("kind") or container.get("action_type") or "").strip().lower()
        surface = str(container.get("surface") or "").strip().lower()
        if kind in {"card_reward", "choose_card_reward", "skip_card_reward"}:
            return True
        if surface == "card_reward":
            return True
    return bool(
        family == "card_reward"
        or "card_reward" in text
        or "card reward" in text
        or (_reward_type(action) == "card" and "reward" in text)
    )


def _action_selection(action: Any) -> str:
    for container in _containers(action):
        for key in ("selection", "selection_action"):
            raw = container.get(key)
            if raw is not None:
                text = str(raw).strip().lower()
                if text:
                    return text
    return ""


def _has_card_payload(action: Any) -> bool:
    return bool(_card_from_action(action))


def _is_pick_action(action: Any, *, card_reward_context: bool) -> bool:
    if not card_reward_context:
        return False
    selection = _action_selection(action)
    if selection in _PICK_SELECTIONS:
        return True
    for container in _containers(action):
        action_id = str(container.get("action_id") or "").strip().lower()
        # Compact replay/frontier signatures may reduce a picked card reward to
        # only a positional id such as ``card_reward:0``.  Treat those as picks
        # even when ``kind``/``selection`` and the full card payload were
        # stripped; otherwise guard/retarget telemetry can silently classify the
        # final executed pick as neither-pick-nor-skip and inflate skip pressure.
        if action_id.startswith("card_reward:") and "skip" not in action_id:
            return True
    for container in _containers(action):
        kind = str(container.get("kind") or container.get("action_type") or "").strip().lower()
        if kind in {"card_reward", "choose_card_reward"}:
            return True
    return bool(_has_card_payload(action) and not _is_skip_action(action, card_reward_context=False))


def _is_skip_action(action: Any, *, card_reward_context: bool) -> bool:
    if not card_reward_context:
        return False
    selection = _action_selection(action)
    if selection in _SKIP_SELECTIONS:
        return True
    text = _lower(action)
    for container in _containers(action):
        kind = str(container.get("kind") or container.get("action_type") or "").strip().lower()
        if kind == "skip_card_reward":
            return True
    family = _semantic_family(action)
    return bool(
        family in {"skip", "proceed"}
        or "skip_card_reward" in text
        or "card_reward:skip" in text
        or "skip card reward" in text
        or ("skip" in text and "reward" in text)
        or ("proceed" in text and "reward" in text)
    )


def _card_from_action(action: Any) -> dict[str, Any]:
    """Extract a best-effort card payload from a legal card-reward action."""

    if not isinstance(action, dict):
        return {}
    for container in _containers(action):
        for key in ("card", "reward_card", "selected_card"):
            card = container.get(key)
            if isinstance(card, dict):
                return dict(card)
        reward = container.get("reward")
        if isinstance(reward, dict):
            card = reward.get("card")
            if isinstance(card, dict):
                return dict(card)
        option = container.get("option")
        if isinstance(option, dict):
            card = option.get("card")
            if isinstance(card, dict):
                return dict(card)
        card_id = container.get("card_id") or container.get("cardId")
        if not card_id and _reward_type(container) == "card":
            card_id = container.get("id")
        if card_id:
            return {
                "id": card_id,
                "title": container.get("card_title") or container.get("title") or container.get("name") or container.get("label"),
                "type": container.get("card_type") or container.get("type"),
                "cost": (
                    container.get("card_cost")
                    if container.get("card_cost") is not None
                    else container.get("cost")
                    if container.get("cost") is not None
                    else container.get("energy_cost")
                ),
                "card_effect_profile": container.get("card_effect_profile"),
            }
    return {}


def _tags_for(card: dict[str, Any]) -> set[str]:
    tags: set[str] = set()
    card_id = str(card.get("id") or card.get("card_id") or "").strip()
    profile = card.get("card_effect_profile") if isinstance(card.get("card_effect_profile"), dict) else {}
    for tag in profile.get("semantic_tags") or ():
        text = str(tag or "").strip().lower()
        if text:
            tags.add(text)
    metadata = get_card_metadata(card_id)
    if isinstance(metadata, dict):
        for key in ("semantic_tags", "tags", "keywords"):
            for tag in metadata.get(key) or ():
                text = str(tag or "").strip().lower()
                if text:
                    tags.add(text)
    for tag in _CARD_ID_TAG_OVERRIDES.get(card_id, ()):
        text = str(tag or "").strip().lower()
        if text:
            tags.add(text)
    return tags


def _signals_for(card: dict[str, Any]) -> dict[str, float]:
    signals: dict[str, float] = {}
    profile = card.get("card_effect_profile") if isinstance(card.get("card_effect_profile"), dict) else {}
    raw = profile.get("semantic_signals") if isinstance(profile.get("semantic_signals"), dict) else {}
    for key, value in raw.items():
        signals[str(key)] = _finite_float(value, 0.0)
    metadata = get_card_metadata(str(card.get("id") or card.get("card_id") or "").strip())
    if isinstance(metadata, dict):
        raw = metadata.get("semantic_signals") if isinstance(metadata.get("semantic_signals"), dict) else {}
        for key, value in raw.items():
            signals.setdefault(str(key), _finite_float(value, 0.0))
    return signals


def _card_type(card: dict[str, Any]) -> str:
    return str(card.get("type") or card.get("card_type") or "").strip().lower()


def _card_cost(card: dict[str, Any]) -> float:
    raw = card.get("cost") if card.get("cost") is not None else card.get("energy_cost")
    if isinstance(raw, str) and raw.strip().upper() == "X":
        return 1.5
    value = _safe_int(raw, 1)
    return max(float(value), 1.0)


def _is_junk_card(card: dict[str, Any]) -> bool:
    ctype = _card_type(card)
    if ctype in _JUNK_TYPES:
        return True
    tags = _tags_for(card)
    text = f"{str(card.get('id') or '').lower()} {str(card.get('title') or card.get('name') or '').lower()}"
    return bool(("curse" in tags or "status" in tags or "unplayable" in tags) or "curse" in text or "status" in text)


def _deck_deficits(summary: dict[str, float]) -> _DeckDeficits:
    deck_size = _finite_float(summary.get("deck_size_raw"), 0.0)
    damage_per_energy = _finite_float(summary.get("raw_avg_damage_per_energy"), 0.0)
    block_per_energy = _finite_float(summary.get("raw_avg_block_per_energy"), 0.0)
    extra_draw = _finite_float(summary.get("raw_expected_extra_draw_per_turn"), 0.0)
    hand_attack = _finite_float(summary.get("expected_hand_attack_damage_per_turn"), 0.0)
    hand_block = _finite_float(summary.get("expected_hand_block_per_turn"), 0.0)
    playable_attack = _finite_float(summary.get("raw_expected_playable_attack_damage_per_turn"), hand_attack)
    playable_block = _finite_float(summary.get("raw_expected_playable_block_per_turn"), hand_block)
    hand_scaling = _finite_float(summary.get("expected_hand_scaling_score"), 0.0)
    return _DeckDeficits(
        thin_deck=deck_size <= 18.0,
        low_attack=playable_attack < 18.0 or hand_attack < 24.0 or damage_per_energy < 4.0,
        low_block=playable_block < 12.0 or hand_block < 10.0 or block_per_energy < 2.0,
        low_draw=extra_draw < 0.5,
        low_scaling=hand_scaling < 0.15,
    )


def _candidate_combo_fit(
    *,
    combo_enabler_like: float,
    combo_payoff_like: float,
    block_payoff_like: float,
    standalone_signal: float,
    deficits: _DeckDeficits,
    summary: dict[str, float],
    future_combo_window: float,
) -> _CandidateComboFit:
    """Score candidate-specific combo option value.

    Deck-level combo metrics alone are insufficient for reward decisions.  The
    same offered card can be:

    * a payoff the current deck can already cash in;
    * a missing enabler/payoff that completes an existing package;
    * an early speculative component with enough future reward/shop runway; or
    * a late orphan component that should not be forced over skip.

    This helper keeps those roles explicit and bounded so the guard can prevent
    pathological early skips without blindly taking every "combo" tagged card.
    """

    component_like = max(combo_enabler_like, combo_payoff_like)
    if component_like <= 0.0:
        return _CandidateComboFit(
            standalone_value=_clamp01(standalone_signal),
            current_fit=0.0,
            missing_piece_fit=0.0,
            speculative_option=0.0,
            orphan_risk=0.0,
            final_option_value=0.0,
        )

    standalone = _clamp01(standalone_signal)
    future = _clamp01(future_combo_window)
    combo_context = _clamp01(summary.get("combo_option_value_score"))
    combo_unmet = _clamp01(summary.get("combo_unmet_dependency_score"))
    delayed_option = _clamp01(summary.get("delayed_payoff_option_value_score"))
    delayed_risk = _clamp01(summary.get("delayed_payoff_unrealized_risk_score"))
    delayed_maturity = _clamp01(summary.get("delayed_payoff_maturity_score"))
    delayed_time_to_value = _clamp01(summary.get("delayed_payoff_time_to_value_score"))
    combo_enabler_coverage = _clamp01(_finite_float(summary.get("combo_enabler_density"), 0.0) * 4.0)
    combo_payoff_coverage = _clamp01(_finite_float(summary.get("combo_payoff_density"), 0.0) * 4.0)
    delayed_enabler_coverage = _clamp01(_finite_float(summary.get("delayed_enabler_density"), 0.0) * 4.0)
    block_readiness = _clamp01(
        max(
            _finite_float(summary.get("block_score"), 0.0),
            _finite_float(summary.get("expected_playable_block_score"), 0.0),
            _finite_float(summary.get("raw_expected_playable_block_per_turn"), 0.0) / 18.0,
        )
    )
    draw_engine = _clamp01(summary.get("draw_engine_score"))
    energy_engine = _clamp01(summary.get("energy_engine_score"))

    # Current fit: the deck already has the partner resources to make this card
    # good in the next few fights.
    payoff_current = 0.0
    if combo_payoff_like > 0.0:
        payoff_current = _clamp01(
            0.50 * combo_context
            + 0.25 * combo_enabler_coverage
            + 0.15 * delayed_enabler_coverage
            + 0.10 * delayed_maturity
        )
        if block_payoff_like > 0.0:
            payoff_current = max(
                payoff_current,
                _clamp01(0.70 * block_readiness + 0.20 * draw_engine + 0.10 * energy_engine),
            )

    enabler_current = 0.0
    if combo_enabler_like > 0.0:
        enabler_current = _clamp01(
            0.35 * combo_unmet
            + 0.35 * combo_payoff_coverage
            + 0.20 * delayed_risk
            + 0.10 * delayed_option
        )
    current_fit = _clamp01(max(payoff_current, enabler_current) + 0.15 * standalone)

    # Missing-piece fit: this offered card plausibly completes something the
    # current deck is missing.  Payoffs need enablers/readiness; enablers need
    # existing payoff pressure or a draw/energy deficit.
    payoff_missing = 0.0
    if combo_payoff_like > 0.0:
        payoff_missing = _clamp01(
            0.45 * combo_enabler_coverage * (1.0 - combo_payoff_coverage)
            + 0.20 * draw_engine
            + 0.15 * energy_engine
            + (0.35 * block_readiness if block_payoff_like > 0.0 else 0.0)
            + (0.15 if deficits.low_attack else 0.0)
            + (0.10 if deficits.low_scaling else 0.0)
        )
    enabler_missing = 0.0
    if combo_enabler_like > 0.0:
        enabler_missing = _clamp01(
            0.45 * max(combo_unmet, combo_payoff_coverage)
            + 0.25 * delayed_risk
            + 0.15 * delayed_option
            + (0.20 if deficits.low_draw else 0.0)
            + (0.10 if deficits.low_scaling else 0.0)
        )
    missing_piece_fit = max(payoff_missing, enabler_missing)

    # Speculative option: future reward/shop rooms give time to find partners,
    # but runway alone should only be a small option value unless the card is
    # also standalone-good or already partially supported.
    speculative_option = _clamp01(
        future
        * (0.20 + 0.35 * standalone + 0.25 * current_fit + 0.20 * missing_piece_fit)
        * (0.70 + 0.20 * delayed_time_to_value + 0.10 * delayed_option)
    )

    payoff_partner_missing = 0.0
    if combo_payoff_like > 0.0:
        payoff_partner_missing = 1.0 - max(combo_enabler_coverage, block_readiness if block_payoff_like > 0.0 else 0.0)
    enabler_partner_missing = 0.0
    if combo_enabler_like > 0.0:
        enabler_partner_missing = max(0.0, 1.0 - max(combo_payoff_coverage, combo_unmet, standalone))
    unresolved_pressure = max(combo_unmet, delayed_risk)
    orphan_risk = _clamp01(
        0.35 * payoff_partner_missing
        + 0.15 * enabler_partner_missing
        + 0.25 * (1.0 - future)
        + 0.25 * (1.0 - standalone) * (1.0 - max(current_fit, missing_piece_fit))
        + 0.20 * unresolved_pressure * (1.0 - 0.50 * future)
        + 0.15 * combo_payoff_like * (1.0 - delayed_maturity)
    )

    final_option = (
        0.24 * current_fit
        + 0.18 * missing_piece_fit
        + 0.16 * speculative_option
        - 0.18 * orphan_risk
    )
    final_option = max(-0.25, min(float(final_option), 0.35))
    return _CandidateComboFit(
        standalone_value=standalone,
        current_fit=current_fit,
        missing_piece_fit=missing_piece_fit,
        speculative_option=speculative_option,
        orphan_risk=orphan_risk,
        final_option_value=final_option,
    )


def _score_candidate_card(
    card: dict[str, Any],
    deficits: _DeckDeficits,
    summary: dict[str, float],
    *,
    future_combo_window: float = 0.5,
) -> _CandidateScore:
    signals = _signals_for(card)
    tags = _tags_for(card)
    ctype = _card_type(card)
    cost = _card_cost(card)
    damage = max(0.0, _finite_float(signals.get("damage"), 0.0))
    block = max(0.0, _finite_float(signals.get("block"), 0.0))
    draw = max(0.0, _finite_float(signals.get("draw"), 0.0))
    if draw <= 0.0 and tags & _DRAW_TAGS:
        draw = 1.0
    energy_like = 1.0 if tags & _ENGINE_TAGS else 0.0
    scaling_like = 1.0 if (tags & _SCALING_TAGS or ctype == "power") else 0.0
    combo_enabler_like = 1.0 if (tags & _COMBO_ENABLER_TAGS) else 0.0
    combo_payoff_like = 1.0 if (tags & _COMBO_PAYOFF_TAGS or tags & {"combo_payoff"}) else 0.0
    block_payoff_like = 1.0 if tags & _BLOCK_PAYOFF_TAGS else 0.0
    combo_component_like = max(combo_enabler_like, combo_payoff_like)
    junk = _is_junk_card(card)
    if junk:
        return _CandidateScore(index=-1, score=-10.0, card=card, junk=True)

    damage_eff = damage / cost
    block_eff = block / cost
    standalone_signal = max(
        min(damage_eff / 8.0, 1.0),
        min(block_eff / 8.0, 1.0),
        min(draw / 3.0, 1.0),
        0.50 * scaling_like,
    )
    combo_fit = _candidate_combo_fit(
        combo_enabler_like=combo_enabler_like,
        combo_payoff_like=combo_payoff_like,
        block_payoff_like=block_payoff_like,
        standalone_signal=standalone_signal,
        deficits=deficits,
        summary=summary,
        future_combo_window=future_combo_window,
    )
    score = 0.0

    # If the deck is thin, any real frontload/block/draw/scaling card is useful,
    # but typed quality deficits decide the priority.
    if deficits.thin_deck:
        score += 0.20
        if damage > 0.0 or block > 0.0 or draw > 0.0 or scaling_like > 0.0:
            score += 0.25
        if ctype == "attack":
            score += 0.10
        elif ctype == "skill":
            score += 0.08
        elif ctype == "power":
            score += 0.08

    if deficits.low_attack:
        score += 1.20 * min(damage_eff / 8.0, 1.0)
        if ctype == "attack" and damage > 0.0:
            score += 0.20
    else:
        score += 0.25 * min(damage_eff / 8.0, 1.0)

    if deficits.low_block:
        score += 1.30 * min(block_eff / 8.0, 1.0)
        if ctype == "skill" and block > 0.0:
            score += 0.20
    else:
        score += 0.25 * min(block_eff / 8.0, 1.0)

    if deficits.low_draw:
        score += 1.10 * min(draw / 3.0, 1.0)
        score += 0.35 * energy_like
    else:
        score += 0.25 * min(draw / 3.0, 1.0)
        score += 0.10 * energy_like

    if deficits.low_scaling:
        score += 0.80 * scaling_like
    else:
        score += 0.20 * scaling_like

    # Combo/option value is deliberately soft.
    #
    # A component card can be three different things:
    #   1. already enabled by the deck => strong pick,
    #   2. early speculative option => small positive option value,
    #   3. orphan / missing dependency => risky and often skip-worthy.
    #
    # This distinction is important for cards such as Body Slam: dynamic runtime
    # damage depends on current block.  With block support it is a real payoff;
    # without block support it must not be treated as a free Act1 frontload card
    # just because it is tagged as an Attack.
    combo_context = _finite_float(summary.get("combo_option_value_score"), 0.0)
    combo_unmet = _finite_float(summary.get("combo_unmet_dependency_score"), 0.0)
    delayed_option = _finite_float(summary.get("delayed_payoff_option_value_score"), 0.0)
    delayed_risk = _finite_float(summary.get("delayed_payoff_unrealized_risk_score"), 0.0)
    delayed_maturity = _finite_float(summary.get("delayed_payoff_maturity_score"), 0.0)
    delayed_time_to_value = _finite_float(summary.get("delayed_payoff_time_to_value_score"), 0.0)
    playable_block = _finite_float(summary.get("raw_expected_playable_block_per_turn"), 0.0)
    if combo_payoff_like > 0.0:
        score += 0.05 + 0.30 * combo_context
        if block_payoff_like > 0.0:
            score += 0.35 * min(playable_block / 18.0, 1.0)
            if deficits.low_attack and playable_block >= 10.0:
                score += 0.20
            if playable_block < 6.0 and standalone_signal < 0.25:
                # Missing block partner: keep only a small future-option value,
                # do not let "Attack + cheap + combo tag" cross the anti-skip
                # threshold by itself.
                score -= 0.18 * (1.0 - min(playable_block / 6.0, 1.0))
    if combo_enabler_like > 0.0 and combo_unmet > 0.25:
        score += 0.20 * min(combo_unmet, 1.0)
    if combo_component_like > 0.0:
        future = max(0.0, min(float(future_combo_window), 1.0))
        score += 0.04 * future
        # Multi-turn payoff cards are real options only when the current deck
        # can make them mature quickly enough.  Keep this as a small bias: it
        # should help avoid pathological skips of supported engines, not let an
        # orphan slow card cross the anti-skip threshold by itself.
        score += 0.08 * future * delayed_option * (0.50 + 0.50 * delayed_maturity)
        score += 0.04 * delayed_time_to_value * delayed_maturity
        if delayed_risk > 0.55 and standalone_signal < 0.25:
            score -= 0.12 * delayed_risk * (1.0 - 0.50 * future)
        if combo_unmet > 0.55 and future < 0.35 and standalone_signal < 0.25:
            score -= 0.10 * combo_unmet * (1.0 - future)
        score += combo_fit.final_option_value

    # Soft preference for cheap cards while the deck lacks draw/energy.
    if cost <= 1.0:
        score += 0.08
    elif cost >= 3.0 and deficits.low_draw:
        score -= 0.15

    return _CandidateScore(
        index=-1,
        score=float(score),
        card=card,
        junk=False,
        standalone_value=combo_fit.standalone_value,
        combo_current_fit=combo_fit.current_fit,
        combo_missing_piece_fit=combo_fit.missing_piece_fit,
        combo_speculative_option=combo_fit.speculative_option,
        combo_orphan_risk=combo_fit.orphan_risk,
        combo_final_option_value=combo_fit.final_option_value,
    )


def _future_combo_floor_window_score(raw_obs: dict[str, Any]) -> float:
    """Floor-only fallback for speculative combo runway."""

    floor, act = _current_floor_and_act(raw_obs)
    if act is not None and act > 1:
        return 0.15
    if floor is None or floor <= 0:
        return 0.50
    if floor <= 5:
        return 1.00
    if floor <= 8:
        return 0.75
    if floor <= 11:
        return 0.50
    if floor <= 14:
        return 0.25
    return 0.10


def _point_type_category(value: Any) -> str:
    raw = _lower(value).replace("_", " ").replace("-", " ")
    if not raw:
        return ""
    if "boss" in raw:
        return "boss"
    if "elite" in raw:
        return "elite"
    if "shop" in raw or "merchant" in raw:
        return "shop"
    if "rest" in raw or "campfire" in raw or "camp fire" in raw:
        return "rest"
    if "treasure" in raw or "chest" in raw or "relic" in raw:
        return "treasure"
    if "question" in raw or "unknown" in raw or raw.strip() == "?":
        return "question"
    if "event" in raw:
        return "event"
    if "monster" in raw or "enemy" in raw or "combat" in raw or "normal" in raw:
        return "monster"
    return raw.strip()


def _node_row(node: dict[str, Any]) -> int | None:
    coord = node.get("coord") if isinstance(node.get("coord"), dict) else {}
    for value in (node.get("row"), coord.get("row"), node.get("y"), coord.get("y")):
        if value is not None:
            return _safe_int(value, 0)
    return None


def _node_point_type(node: dict[str, Any]) -> Any:
    for key in ("point_type", "pointType", "type", "room_type", "kind"):
        if node.get(key) is not None:
            return node.get(key)
    current = node.get("point") if isinstance(node.get("point"), dict) else {}
    return current.get("point_type") or current.get("type")


def _iter_route_summary_candidates(raw_obs: dict[str, Any]) -> list[dict[str, Any]]:
    summaries: list[dict[str, Any]] = []

    def add_summary(value: Any) -> None:
        if isinstance(value, dict):
            summaries.append(value)

    def add_summary_list(value: Any) -> None:
        if isinstance(value, list):
            for item in value:
                add_summary(item)

    def scan_actions(container: dict[str, Any]) -> None:
        for key in ("available_actions", "legal_actions", "full_legal_actions"):
            actions = container.get(key)
            if not isinstance(actions, list):
                continue
            for action in actions:
                if not isinstance(action, dict):
                    continue
                add_summary(action.get("route_summary"))
                add_summary(action.get("summary"))

    add_summary(raw_obs.get("route_summary"))
    add_summary_list(raw_obs.get("route_summaries"))
    scan_actions(raw_obs)

    snapshot = raw_obs.get("_run_route_snapshot") or raw_obs.get("run_route_snapshot")
    if isinstance(snapshot, dict):
        add_summary(snapshot.get("route_summary"))
        add_summary(snapshot.get("selected_route_summary"))
        add_summary(snapshot.get("best_route_summary"))
        for key in ("route_summaries", "available_route_summaries", "summaries"):
            add_summary_list(snapshot.get(key))
        scan_actions(snapshot)
    return summaries


def _score_route_summary_runway(summary: dict[str, Any]) -> tuple[float, float] | None:
    """Return ``(score, estimated_reward_opportunities)`` for a route subtree.

    ``route_summary`` counts are subtree-wide rather than a single future path,
    so cap opportunities by depth to avoid making wide maps look like guaranteed
    rewards.  The value is intentionally only a soft option-window estimate.
    """

    depth = _finite_float(summary.get("max_depth"), 0.0)
    next_boss = summary.get("next_boss_steps")
    if next_boss is not None:
        depth = max(depth, _finite_float(next_boss, 0.0))
    reachable = _finite_float(summary.get("reachable_node_count"), 0.0)
    if depth <= 0.0 and reachable <= 0.0:
        return None

    monsters = _finite_float(summary.get("count_monster"), 0.0)
    elites = _finite_float(summary.get("count_elite"), 0.0)
    events = _finite_float(summary.get("count_event"), 0.0)
    questions = _finite_float(summary.get("count_question_mark"), 0.0)
    shops = _finite_float(summary.get("count_shop"), 0.0)
    treasures = _finite_float(summary.get("count_treasure"), 0.0)

    raw_opportunity = (
        monsters
        + elites
        + 0.35 * events
        + 0.35 * questions
        + 0.60 * shops
        + 0.15 * treasures
    )
    if depth > 0.0:
        raw_opportunity = min(raw_opportunity, depth * 1.10)

    score = _clamp01(raw_opportunity / 7.0)
    if next_boss is not None and _finite_float(next_boss, 999.0) <= 3.0:
        score = min(score, 0.30)
    return score, raw_opportunity


def _iter_future_map_node_lists(raw_obs: dict[str, Any]) -> list[tuple[list[Any], bool]]:
    """Return map node lists and whether each list is likely the full map.

    During card-reward surfaces there are usually no map legal actions.  The
    headless simulator still keeps the full map under ``_sim_raw.map.nodes`` in
    newer DTOs; live/older bridge paths may only expose immediate ``points``.
    Immediate-only points are not reliable enough to down-rank future option
    value, so they are used only when they span multiple future rows.
    """

    out: list[tuple[list[Any], bool]] = []
    containers: list[Any] = []
    for key in ("map", "route", "current_map"):
        if isinstance(raw_obs.get(key), dict):
            containers.append(raw_obs.get(key))
    sim_raw = raw_obs.get("_sim_raw") if isinstance(raw_obs.get("_sim_raw"), dict) else {}
    for key in ("map", "route", "current_map"):
        if isinstance(sim_raw.get(key), dict):
            containers.append(sim_raw.get(key))
    run = raw_obs.get("run") if isinstance(raw_obs.get("run"), dict) else {}
    for key in ("map", "route"):
        if isinstance(run.get(key), dict):
            containers.append(run.get(key))
    snapshot = raw_obs.get("_run_route_snapshot") or raw_obs.get("run_route_snapshot")
    if isinstance(snapshot, dict):
        containers.append(snapshot)
        for key in ("map", "route", "current_map"):
            if isinstance(snapshot.get(key), dict):
                containers.append(snapshot.get(key))

    for container in containers:
        for key in ("nodes", "all_nodes", "route_nodes"):
            value = container.get(key) if isinstance(container, dict) else None
            if isinstance(value, list) and value:
                out.append((value, True))
        value = container.get("points") if isinstance(container, dict) else None
        if isinstance(value, list) and value:
            out.append((value, False))
    return out


def _score_map_node_runway(raw_obs: dict[str, Any], floor: int | None) -> tuple[float, float] | None:
    best: tuple[float, float] | None = None
    for nodes, likely_full_map in _iter_future_map_node_lists(raw_obs):
        rows: dict[int, set[str]] = {}
        boss_distance: int | None = None
        for entry in nodes:
            if not isinstance(entry, dict):
                continue
            row = _node_row(entry)
            if row is None:
                continue
            if floor is not None and row <= floor:
                continue
            cat = _point_type_category(_node_point_type(entry))
            if not cat:
                continue
            rows.setdefault(int(row), set()).add(cat)
            if cat == "boss" and floor is not None:
                dist = max(1, int(row) - int(floor))
                boss_distance = dist if boss_distance is None else min(boss_distance, dist)

        if not rows:
            continue
        # Ignore immediate-only maps unless they span more than one future row;
        # otherwise a card reward right before a map screen would look like "no
        # future rewards" simply because only next_options were exposed.
        if not likely_full_map and len(rows) <= 1:
            continue

        opportunity = 0.0
        for row in sorted(rows):
            cats = rows[row]
            if "boss" in cats:
                continue
            row_weight = 0.0
            if "elite" in cats or "monster" in cats:
                row_weight = max(row_weight, 1.0)
            if "shop" in cats:
                row_weight = max(row_weight, 0.60)
            if "event" in cats or "question" in cats:
                row_weight = max(row_weight, 0.35)
            if "treasure" in cats:
                row_weight = max(row_weight, 0.15)
            opportunity += row_weight

        score = _clamp01(opportunity / 7.0)
        if boss_distance is not None and boss_distance <= 3:
            score = min(score, 0.30)
        if best is None or score > best[0]:
            best = (score, opportunity)
    return best


def _future_combo_runway(raw_obs: dict[str, Any]) -> _FutureComboRunway:
    """Estimate future combo/reward runway for delayed payoff pieces.

    Early Act1 has many reward/shop/event chances left, so a component card can
    carry modest "future combo option" value.  Near the Act1 boss that option
    decays sharply: the card must already be useful in the current deck.

    Prefer route/map-derived runway when available because cards are not just
    static objects: a payoff card can be a good *option* if there are enough
    future reward/shop/event chances to assemble support, but it becomes an
    orphan risk when the boss is close and the deck cannot mature it.
    """

    floor_score = _future_combo_floor_window_score(raw_obs)
    floor, act = _current_floor_and_act(raw_obs)
    route_score: float | None = None
    reward_opportunity = 0.0

    route_candidates: list[tuple[float, float]] = []
    for summary in _iter_route_summary_candidates(raw_obs):
        scored = _score_route_summary_runway(summary)
        if scored is not None:
            route_candidates.append(scored)
    map_scored = _score_map_node_runway(raw_obs, floor)
    if map_scored is not None:
        route_candidates.append(map_scored)

    if route_candidates:
        route_score, reward_opportunity = max(route_candidates, key=lambda item: item[0])
        # Route/map DTOs are more specific than floor buckets, but still noisy
        # across bridge versions.  Blend rather than hard override.
        score = _clamp01(0.25 * floor_score + 0.75 * route_score)
        if act is not None and act > 1:
            score = min(score, 0.25)
        return _FutureComboRunway(
            score=score,
            floor_score=floor_score,
            route_score=float(route_score),
            reward_opportunity=float(reward_opportunity),
            observed=True,
        )

    return _FutureComboRunway(
        score=floor_score,
        floor_score=floor_score,
        route_score=0.0,
        reward_opportunity=0.0,
        observed=False,
    )


def _future_combo_window_score(raw_obs: dict[str, Any]) -> float:
    """Backward-compatible scalar wrapper used by older unit tests."""

    return _future_combo_runway(raw_obs).score


def _dynamic_min_useful_score(
    *,
    base: float,
    deck_size: float,
    deficits: _DeckDeficits,
    future_runway: _FutureComboRunway,
    best: _CandidateScore,
    raw_obs: dict[str, Any],
) -> tuple[float, dict[str, float]]:
    """Return the actionability threshold for the best offered reward card.

    The fixed historical threshold (0.45) was too rigid for build decisions:
    early thin Act1 decks often need to accept "medium" cards because the deck
    is not yet functional, while late speculative combo pieces should expire
    unless they are already useful or clearly complete an existing package.

    This deliberately adjusts only the *anti-skip guard threshold*.  The card
    score itself still carries standalone/current-fit/missing-piece/orphan
    terms, so a future combo component receives option value without becoming
    an unconditional forced pick.
    """

    base = _finite_float(base, 0.45)
    threshold = float(base)
    speculative_option_accepted = 0.0
    raised_near_boss = 0.0

    # Early thin decks die because they never assemble enough playable cards.
    # Use a lower threshold while the deck is still small and has objective
    # attack/block/draw/scaling deficits.
    if deficits.any:
        if deck_size <= 13.0:
            threshold = min(threshold, 0.24)
        elif deck_size <= 16.0:
            threshold = min(threshold, 0.31)
        elif deck_size <= 18.0:
            threshold = min(threshold, 0.38)

    # A component can be worth taking as an option when the route still has
    # enough reward/shop/runway left *and* it is not an obvious orphan.  This is
    # intentionally small; it should break pathological skips, not override
    # late-boss readiness.
    if (
        future_runway.score >= 0.50
        and best.combo_final_option_value > 0.03
        and best.combo_orphan_risk < 0.55
    ):
        threshold -= 0.04
        speculative_option_accepted = 1.0

    # Near the Act1 boss, speculative packages have little time to mature.
    # Raise the bar unless the offered card is already useful in the current
    # deck or has standalone value.
    floor, act = _current_floor_and_act(raw_obs)
    near_act1_boss = floor is not None and floor >= 14 and (act is None or act <= 1)
    if near_act1_boss and max(best.standalone_value, best.combo_current_fit) < 0.35:
        threshold += 0.05
        raised_near_boss = 1.0

    threshold = max(0.20, min(float(threshold), float(base) + 0.10))
    return threshold, {
        "relaxed": 1.0 if threshold + 1e-6 < float(base) else 0.0,
        "raised_near_boss": raised_near_boss,
        "speculative_option_accepted": speculative_option_accepted,
    }


def _candidate_orphan_reject(item: _CandidateScore) -> bool:
    """Return whether one offered card is an unsupported combo orphan.

    This is deliberately candidate-local.  A reward screen can offer an orphan
    combo component next to a useful standalone filler; rejecting the orphan
    must not make the guard skip the entire reward.
    """

    return (
        item.combo_orphan_risk >= 0.55
        and item.standalone_value < 0.25
        and item.combo_current_fit < 0.25
        and item.combo_missing_piece_fit < 0.35
        and item.combo_final_option_value <= 0.02
    )


def _candidate_rejection_reason(item: _CandidateScore, dynamic_threshold: float) -> str | None:
    if item.junk:
        return "junk"
    if _candidate_orphan_reject(item):
        return "orphan_reject"
    if item.score < float(dynamic_threshold):
        return "below_dynamic_threshold"
    return None


def _current_floor_and_act(raw_obs: dict[str, Any]) -> tuple[int | None, int | None]:
    run = raw_obs.get("run") if isinstance(raw_obs.get("run"), dict) else {}
    player = raw_obs.get("player") if isinstance(raw_obs.get("player"), dict) else {}
    candidates_floor = (
        run.get("current_floor"),
        run.get("floor"),
        raw_obs.get("current_floor"),
        raw_obs.get("floor"),
        player.get("floor"),
    )
    floor: int | None = None
    for value in candidates_floor:
        if value is not None:
            floor = _safe_int(value, 0)
            break
    candidates_act = (
        run.get("act_id"),
        run.get("act"),
        raw_obs.get("act_id"),
        raw_obs.get("act"),
    )
    act: int | None = None
    for value in candidates_act:
        if value is not None:
            act = _safe_int(value, 0)
            break
    return floor, act


def _is_act1_or_unknown(raw_obs: dict[str, Any], deck_size: float) -> bool:
    floor, act = _current_floor_and_act(raw_obs)
    if act is not None:
        # Some bridge paths use 0-based acts, some use 1-based acts.  Treat both
        # 0 and 1 as Act1; reject only clearly later acts.
        if act > 1:
            return False
    if floor is not None and floor > 17:
        return False
    # Missing run metadata is common in tests and some compact wrappers; a thin
    # deck is enough evidence that this is an early-run card reward.
    return deck_size <= 24.0 or floor is not None or act is not None


def _write_deck_quality_metrics(search_stats: dict[str, Any], summary: dict[str, float], deficits: _DeckDeficits) -> None:
    deck_size = _finite_float(summary.get("deck_size_raw"), 0.0)
    search_stats["card_reward_guard_deck_size"] = deck_size
    search_stats["card_reward_guard_damage_per_energy"] = _finite_float(summary.get("raw_avg_damage_per_energy"), 0.0)
    search_stats["card_reward_guard_block_per_energy"] = _finite_float(summary.get("raw_avg_block_per_energy"), 0.0)
    search_stats["card_reward_guard_expected_extra_draw"] = _finite_float(summary.get("raw_expected_extra_draw_per_turn"), 0.0)
    search_stats["card_reward_guard_expected_cards_seen"] = _finite_float(summary.get("raw_expected_cards_seen_per_turn"), 0.0)
    search_stats["card_reward_guard_expected_hand_attack_damage"] = _finite_float(
        summary.get("expected_hand_attack_damage_per_turn"),
        0.0,
    )
    search_stats["card_reward_guard_expected_hand_block"] = _finite_float(summary.get("expected_hand_block_per_turn"), 0.0)
    search_stats["card_reward_guard_expected_hand_attack_share"] = _finite_float(summary.get("expected_hand_attack_share"), 0.0)
    search_stats["card_reward_guard_expected_hand_skill_share"] = _finite_float(summary.get("expected_hand_skill_share"), 0.0)
    search_stats["card_reward_guard_expected_hand_power_share"] = _finite_float(summary.get("expected_hand_power_share"), 0.0)
    search_stats["card_reward_guard_expected_hand_junk_share"] = _finite_float(summary.get("expected_hand_junk_share"), 0.0)
    search_stats["card_reward_guard_expected_hand_useful_quality"] = _finite_float(
        summary.get("expected_hand_useful_quality_score"),
        0.0,
    )
    search_stats["card_reward_guard_expected_playable_cards"] = _finite_float(
        summary.get("raw_expected_playable_cards_per_turn"),
        0.0,
    )
    search_stats["card_reward_guard_expected_playable_attack_damage"] = _finite_float(
        summary.get("raw_expected_playable_attack_damage_per_turn"),
        0.0,
    )
    search_stats["card_reward_guard_expected_playable_block"] = _finite_float(
        summary.get("raw_expected_playable_block_per_turn"),
        0.0,
    )
    search_stats["card_reward_guard_expected_energy_utilization"] = _finite_float(
        summary.get("expected_energy_utilization_score"),
        0.0,
    )
    search_stats["card_reward_guard_combo_option_value"] = _finite_float(summary.get("combo_option_value_score"), 0.0)
    search_stats["card_reward_guard_combo_unmet_dependency"] = _finite_float(summary.get("combo_unmet_dependency_score"), 0.0)
    search_stats["card_reward_guard_delayed_payoff_option_value"] = _finite_float(
        summary.get("delayed_payoff_option_value_score"),
        0.0,
    )
    search_stats["card_reward_guard_delayed_payoff_unrealized_risk"] = _finite_float(
        summary.get("delayed_payoff_unrealized_risk_score"),
        0.0,
    )
    search_stats["card_reward_guard_delayed_payoff_maturity"] = _finite_float(
        summary.get("delayed_payoff_maturity_score"),
        0.0,
    )
    search_stats["card_reward_guard_delayed_payoff_time_to_value"] = _finite_float(
        summary.get("delayed_payoff_time_to_value_score"),
        0.0,
    )
    search_stats["card_reward_guard_low_block"] = 1.0 if deficits.low_block else 0.0
    search_stats["card_reward_guard_low_draw"] = 1.0 if deficits.low_draw else 0.0
    search_stats["card_reward_guard_low_attack"] = 1.0 if deficits.low_attack else 0.0
    search_stats["card_reward_guard_low_scaling"] = 1.0 if deficits.low_scaling else 0.0
    search_stats["card_reward_guard_reason_thin_deck"] = 1.0 if deficits.thin_deck else 0.0
    search_stats["card_reward_guard_reason_low_block"] = 1.0 if deficits.low_block else 0.0
    search_stats["card_reward_guard_reason_low_draw"] = 1.0 if deficits.low_draw else 0.0
    search_stats["card_reward_guard_reason_low_attack"] = 1.0 if deficits.low_attack else 0.0
    search_stats["card_reward_guard_reason_low_scaling"] = 1.0 if deficits.low_scaling else 0.0


def _compact_reward_card_for_diagnostic(card: dict[str, Any]) -> dict[str, Any]:
    signals = _signals_for(card)
    return {
        "id": card.get("id") or card.get("card_id"),
        "title": card.get("title") or card.get("name"),
        "type": card.get("type") or card.get("card_type"),
        "cost": card.get("cost") if card.get("cost") is not None else card.get("energy_cost"),
        "damage": _finite_float(signals.get("damage"), 0.0),
        "block": _finite_float(signals.get("block"), 0.0),
        "draw": _finite_float(signals.get("draw"), 0.0),
        "tags": sorted(_tags_for(card))[:16],
    }


def _compact_reward_action_for_diagnostic(action: Any, *, index: int | None = None) -> dict[str, Any]:
    out: dict[str, Any] = {}
    if index is not None:
        out["index"] = int(index)
    if not isinstance(action, dict):
        out["missing"] = True
        return out
    out.update(
        {
            "kind": action.get("kind"),
            "surface": action.get("surface"),
            "selection": action.get("selection"),
            "action_id": action.get("action_id"),
            "label": action.get("label"),
        }
    )
    card = _card_from_action(action)
    if card:
        out["card"] = _compact_reward_card_for_diagnostic(card)
    return {key: value for key, value in out.items() if value not in (None, "")}


def _candidate_choice_diagnostic(item: _CandidateScore) -> dict[str, Any]:
    return {
        "index": int(item.index),
        "score": float(item.score),
        "junk": bool(item.junk),
        "standalone_value": float(item.standalone_value),
        "combo_current_fit": float(item.combo_current_fit),
        "combo_missing_piece_fit": float(item.combo_missing_piece_fit),
        "combo_speculative_option": float(item.combo_speculative_option),
        "combo_orphan_risk": float(item.combo_orphan_risk),
        "combo_final_option_value": float(item.combo_final_option_value),
        "card": _compact_reward_card_for_diagnostic(item.card),
    }


def _store_card_reward_choice_diagnostic(
    stats: dict[str, Any],
    *,
    raw_obs: dict[str, Any],
    summary: dict[str, float],
    deficits: _DeckDeficits,
    future_runway: _FutureComboRunway,
    original_action_idx: int,
    selected_action: Any,
    scored: list[_CandidateScore],
    best: _CandidateScore | None,
    min_useful_score: float,
    dynamic_min_useful_score: float,
    decision_reason: str,
    skip_blocked: bool,
    orphan_reject: bool,
    raw_best: _CandidateScore | None = None,
    rejected_candidate_count: int = 0,
    top_candidate_rejected: bool = False,
) -> None:
    """Attach one compact JSON-serializable reward-choice diagnostic payload.

    TensorBoard means tell us skip rate and average score, but not whether the
    offered cards were bad, orphaned, or good future options.  Store a bounded
    payload in ``search_stats``; ``self_play`` writes it to JSONL after the final
    action index is known.
    """

    try:
        floor, act = _current_floor_and_act(raw_obs)
        payload = {
            "floor": floor,
            "act_id": act,
            "deck_size_before": _finite_float(summary.get("deck_size_raw"), 0.0),
            "deck_quality_before": {
                "damage_per_energy": _finite_float(summary.get("raw_avg_damage_per_energy"), 0.0),
                "block_per_energy": _finite_float(summary.get("raw_avg_block_per_energy"), 0.0),
                "expected_extra_draw": _finite_float(summary.get("raw_expected_extra_draw_per_turn"), 0.0),
                "expected_cards_seen": _finite_float(summary.get("raw_expected_cards_seen_per_turn"), 0.0),
                "expected_hand_attack_damage": _finite_float(summary.get("expected_hand_attack_damage_per_turn"), 0.0),
                "expected_hand_block": _finite_float(summary.get("expected_hand_block_per_turn"), 0.0),
                "expected_hand_attack_share": _finite_float(summary.get("expected_hand_attack_share"), 0.0),
                "expected_hand_skill_share": _finite_float(summary.get("expected_hand_skill_share"), 0.0),
                "expected_hand_power_share": _finite_float(summary.get("expected_hand_power_share"), 0.0),
                "expected_hand_junk_share": _finite_float(summary.get("expected_hand_junk_share"), 0.0),
                "expected_hand_useful_quality": _finite_float(summary.get("expected_hand_useful_quality_score"), 0.0),
                "expected_playable_cards": _finite_float(summary.get("raw_expected_playable_cards_per_turn"), 0.0),
                "expected_playable_attack_damage": _finite_float(
                    summary.get("raw_expected_playable_attack_damage_per_turn"),
                    0.0,
                ),
                "expected_playable_block": _finite_float(summary.get("raw_expected_playable_block_per_turn"), 0.0),
                "expected_energy_utilization": _finite_float(summary.get("expected_energy_utilization_score"), 0.0),
                "combo_option_value": _finite_float(summary.get("combo_option_value_score"), 0.0),
                "combo_unmet_dependency": _finite_float(summary.get("combo_unmet_dependency_score"), 0.0),
                "delayed_payoff_option_value": _finite_float(summary.get("delayed_payoff_option_value_score"), 0.0),
                "delayed_payoff_unrealized_risk": _finite_float(
                    summary.get("delayed_payoff_unrealized_risk_score"),
                    0.0,
                ),
            },
            "deficits": {
                "thin_deck": bool(deficits.thin_deck),
                "low_attack": bool(deficits.low_attack),
                "low_block": bool(deficits.low_block),
                "low_draw": bool(deficits.low_draw),
                "low_scaling": bool(deficits.low_scaling),
            },
            "future_combo_window": float(future_runway.score),
            "future_combo_floor_window": float(future_runway.floor_score),
            "future_combo_route_window": float(future_runway.route_score),
            "future_reward_opportunity": float(future_runway.reward_opportunity),
            "future_runway_observed": bool(future_runway.observed),
            "original_action_idx": int(original_action_idx),
            "original_action": _compact_reward_action_for_diagnostic(
                selected_action,
                index=int(original_action_idx),
            ),
            # This payload is produced while evaluating the model/search choice,
            # before self_play knows the final post-guard action.  Keep the
            # original and final skip semantics separate: self_play overwrites
            # selected_is_skip with final-action semantics before JSONL dump.
            "original_selected_is_skip": True,
            "final_selected_is_skip": None,
            "selected_is_skip": True,
            "min_useful_score": float(min_useful_score),
            "dynamic_min_useful_score": float(dynamic_min_useful_score),
            "decision_reason": str(decision_reason),
            "skip_blocked": bool(skip_blocked),
            "orphan_reject": bool(orphan_reject),
            "rejected_candidate_count": int(max(0, rejected_candidate_count)),
            "top_candidate_rejected": bool(top_candidate_rejected),
            "raw_best_score": float(raw_best.score) if raw_best is not None else (
                float(best.score) if best is not None else 0.0
            ),
            "raw_best_card": _compact_reward_card_for_diagnostic(raw_best.card)
            if raw_best is not None
            else (_compact_reward_card_for_diagnostic(best.card) if best is not None else None),
            "raw_best_candidate": _candidate_choice_diagnostic(raw_best)
            if raw_best is not None
            else (_candidate_choice_diagnostic(best) if best is not None else None),
            "best_score": float(best.score) if best is not None else 0.0,
            "best_card": _compact_reward_card_for_diagnostic(best.card) if best is not None else None,
            "best_candidate": _candidate_choice_diagnostic(best) if best is not None else None,
            "offered_cards": [_candidate_choice_diagnostic(item) for item in scored[:8]],
        }
        stats["_card_reward_choice_diagnostic"] = payload
    except Exception:
        return


def apply_card_reward_guard(
    *,
    action_idx: int,
    legal_actions: list[Any] | None,
    full_legal_actions: list[Any] | None,
    action_mask: Any,
    raw_obs: dict[str, Any] | None,
    search_stats: dict[str, Any],
    min_useful_score: float = 0.45,
) -> int:
    """Return a possibly overridden card-reward action index.

    The function is intentionally defensive: any malformed observation/action
    contract leaves the selected action unchanged and only sets telemetry flags.
    """

    stats = search_stats if isinstance(search_stats, dict) else {}
    for key in CARD_REWARD_GUARD_METRIC_KEYS:
        stats.setdefault(key, 0.0)

    if not isinstance(legal_actions, list) or not legal_actions:
        return int(action_idx)
    try:
        import numpy as np

        mask_np = np.asarray(action_mask, dtype="float32").reshape(-1)
    except Exception:
        stats["card_reward_guard_alignment_error"] = 1.0
        return int(action_idx)

    legal_count = min(len(legal_actions), int(mask_np.shape[0]) if mask_np.size else 0)
    if legal_count <= 0 or not (0 <= int(action_idx) < legal_count) or mask_np[int(action_idx)] <= 0:
        stats["card_reward_guard_alignment_error"] = 1.0
        return int(action_idx)

    if isinstance(full_legal_actions, list) and len(full_legal_actions) < legal_count:
        stats["card_reward_guard_alignment_error"] = 1.0
        return int(action_idx)

    def action_at(idx: int) -> Any:
        full = full_legal_actions[idx] if isinstance(full_legal_actions, list) and idx < len(full_legal_actions) else None
        compact = legal_actions[idx] if idx < len(legal_actions) else None
        # Prefer the payload that actually contains the candidate card.
        if _card_from_action(full):
            return full
        return compact if compact is not None else full

    selected = action_at(int(action_idx))
    context = _is_card_reward_action(selected) or any(_is_card_reward_action(action_at(i)) for i in range(legal_count))
    if not context:
        return int(action_idx)
    stats["card_reward_guard_context"] = 1.0

    selected_is_pick = _is_pick_action(selected, card_reward_context=True)
    selected_is_skip = _is_skip_action(selected, card_reward_context=True)
    stats["card_reward_guard_selected_pick"] = 1.0 if selected_is_pick and not selected_is_skip else 0.0
    stats["card_reward_guard_selected_skip"] = 1.0 if selected_is_skip and not selected_is_pick else 0.0
    if selected_is_pick and not selected_is_skip:
        return int(action_idx)
    if not selected_is_skip:
        return int(action_idx)

    if not isinstance(raw_obs, dict):
        stats["card_reward_guard_invalid_obs"] = 1.0
        return int(action_idx)

    deck_cards = extract_deck_cards_from_obs_like(raw_obs)
    if not deck_cards:
        stats["card_reward_guard_invalid_obs"] = 1.0
        return int(action_idx)

    summary = compute_deck_quality_summary(deck_cards)
    deficits = _deck_deficits(summary)
    _write_deck_quality_metrics(stats, summary, deficits)
    future_runway = _future_combo_runway(raw_obs)
    future_combo_window = future_runway.score
    stats["card_reward_guard_future_combo_window"] = float(future_combo_window)
    stats["card_reward_guard_future_combo_floor_window"] = float(future_runway.floor_score)
    stats["card_reward_guard_future_combo_route_window"] = float(future_runway.route_score)
    stats["card_reward_guard_future_reward_opportunity"] = float(future_runway.reward_opportunity)
    stats["card_reward_guard_future_runway_observed"] = 1.0 if future_runway.observed else 0.0
    deck_size = _finite_float(summary.get("deck_size_raw"), 0.0)
    if not _is_act1_or_unknown(raw_obs, deck_size):
        stats["card_reward_guard_not_act1"] = 1.0
        return int(action_idx)
    if not deficits.any:
        return int(action_idx)

    stats["card_reward_guard_applicable"] = 1.0
    scored: list[_CandidateScore] = []
    for idx in range(legal_count):
        if mask_np[idx] <= 0:
            continue
        action = action_at(idx)
        if not _is_pick_action(action, card_reward_context=True):
            continue
        card = _card_from_action(action)
        if not card:
            continue
        score = _score_candidate_card(card, deficits, summary, future_combo_window=future_combo_window)
        scored.append(
            _CandidateScore(
                index=int(idx),
                score=score.score,
                card=card,
                junk=score.junk,
                standalone_value=score.standalone_value,
                combo_current_fit=score.combo_current_fit,
                combo_missing_piece_fit=score.combo_missing_piece_fit,
                combo_speculative_option=score.combo_speculative_option,
                combo_orphan_risk=score.combo_orphan_risk,
                combo_final_option_value=score.combo_final_option_value,
            )
        )

    if not scored:
        return int(action_idx)
    stats["card_reward_guard_pick_available"] = 1.0

    raw_best = max(scored, key=lambda item: item.score)
    sorted_scored = sorted(scored, key=lambda item: item.score, reverse=True)

    best: _CandidateScore | None = None
    dynamic_threshold = float(min_useful_score)
    threshold_flags: dict[str, float] = {
        "relaxed": 0.0,
        "raised_near_boss": 0.0,
        "speculative_option_accepted": 0.0,
    }
    decision_reason = "below_dynamic_threshold"
    rejected_candidate_count = 0
    top_candidate_rejected = False
    any_orphan_reject = False
    raw_best_threshold = float(min_useful_score)
    raw_best_flags: dict[str, float] = threshold_flags
    raw_best_reason: str | None = None

    for rank, candidate in enumerate(sorted_scored):
        candidate_threshold, candidate_flags = _dynamic_min_useful_score(
            base=float(min_useful_score),
            deck_size=deck_size,
            deficits=deficits,
            future_runway=future_runway,
            best=candidate,
            raw_obs=raw_obs,
        )
        reject_reason = _candidate_rejection_reason(candidate, candidate_threshold)
        if candidate is raw_best:
            raw_best_threshold = float(candidate_threshold)
            raw_best_flags = candidate_flags
            raw_best_reason = reject_reason
        if reject_reason is None:
            best = candidate
            dynamic_threshold = float(candidate_threshold)
            threshold_flags = candidate_flags
            decision_reason = "skip_blocked"
            break
        rejected_candidate_count += 1
        if rank == 0:
            top_candidate_rejected = True
        if reject_reason == "orphan_reject":
            any_orphan_reject = True

    if best is None:
        # No actionable card survived.  Keep the highest-scoring raw candidate
        # as the diagnostic "best" so JSONL can explain whether the surface was
        # genuinely bad, an orphan, or merely below the dynamic threshold.
        best = raw_best
        dynamic_threshold = float(raw_best_threshold)
        threshold_flags = raw_best_flags
        decision_reason = raw_best_reason or _candidate_rejection_reason(best, dynamic_threshold) or "below_dynamic_threshold"

    stats["card_reward_guard_best_score"] = float(best.score)
    stats["card_reward_guard_best_standalone_value"] = float(best.standalone_value)
    stats["card_reward_guard_best_combo_current_fit"] = float(best.combo_current_fit)
    stats["card_reward_guard_best_combo_missing_piece_fit"] = float(best.combo_missing_piece_fit)
    stats["card_reward_guard_best_combo_speculative_option"] = float(best.combo_speculative_option)
    stats["card_reward_guard_best_combo_orphan_risk"] = float(best.combo_orphan_risk)
    stats["card_reward_guard_best_combo_final_option_value"] = float(best.combo_final_option_value)
    stats["card_reward_guard_dynamic_min_useful_score"] = float(dynamic_threshold)
    stats["card_reward_guard_threshold_relaxed"] = float(threshold_flags.get("relaxed", 0.0))
    stats["card_reward_guard_threshold_raised_near_boss"] = float(threshold_flags.get("raised_near_boss", 0.0))
    stats["card_reward_guard_speculative_option_accepted"] = float(
        threshold_flags.get("speculative_option_accepted", 0.0)
    )
    orphan_reject = decision_reason == "orphan_reject"
    stats["card_reward_guard_orphan_reject"] = 1.0 if any_orphan_reject or orphan_reject else 0.0
    stats["card_reward_guard_top_candidate_rejected"] = 1.0 if top_candidate_rejected else 0.0
    stats["card_reward_guard_rejected_candidate_count"] = float(rejected_candidate_count)

    if decision_reason != "skip_blocked":
        stats["card_reward_guard_no_useful_candidate"] = 1.0
        _store_card_reward_choice_diagnostic(
            stats,
            raw_obs=raw_obs,
            summary=summary,
            deficits=deficits,
            future_runway=future_runway,
            original_action_idx=int(action_idx),
            selected_action=selected,
            scored=scored,
            best=best,
            raw_best=raw_best,
            rejected_candidate_count=int(rejected_candidate_count),
            top_candidate_rejected=bool(top_candidate_rejected),
            min_useful_score=float(min_useful_score),
            dynamic_min_useful_score=float(dynamic_threshold),
            decision_reason=decision_reason,
            skip_blocked=False,
            orphan_reject=bool(orphan_reject),
        )
        return int(action_idx)

    stats["card_reward_guard_useful_candidate_available"] = 1.0
    override_idx = int(best.index)
    if not (0 <= override_idx < legal_count) or mask_np[override_idx] <= 0:
        stats["card_reward_guard_alignment_error"] = 1.0
        return int(action_idx)
    stats["card_reward_guard_applied"] = 1.0
    stats["card_reward_guard_override"] = 1.0 if override_idx != int(action_idx) else 0.0
    stats["card_reward_guard_skip_blocked"] = 1.0
    stats["card_reward_guard_selected_score"] = 0.0
    _store_card_reward_choice_diagnostic(
        stats,
        raw_obs=raw_obs,
        summary=summary,
        deficits=deficits,
        future_runway=future_runway,
        original_action_idx=int(action_idx),
        selected_action=selected,
        scored=scored,
        best=best,
        raw_best=raw_best,
        rejected_candidate_count=int(rejected_candidate_count),
        top_candidate_rejected=bool(top_candidate_rejected),
        min_useful_score=float(min_useful_score),
        dynamic_min_useful_score=float(dynamic_threshold),
        decision_reason=decision_reason,
        skip_blocked=True,
        orphan_reject=False,
    )
    return override_idx


CARD_REWARD_GUARD_SEARCH_SUFFIXES: dict[str, str] = {
    "card_reward_guard_context": "card_reward_guard_context_rate",
    "card_reward_guard_applicable": "card_reward_guard_applicable_rate",
    "card_reward_guard_selected_skip": "card_reward_guard_selected_skip_rate",
    "card_reward_guard_selected_pick": "card_reward_guard_selected_pick_rate",
    "card_reward_guard_pick_available": "card_reward_guard_pick_available_rate",
    "card_reward_guard_useful_candidate_available": "card_reward_guard_useful_candidate_available_rate",
    "card_reward_guard_no_useful_candidate": "card_reward_guard_no_useful_candidate_rate",
    "card_reward_guard_applied": "card_reward_guard_applied_rate",
    "card_reward_guard_override": "card_reward_guard_override_rate",
    "card_reward_guard_skip_blocked": "card_reward_guard_skip_blocked_rate",
    "card_reward_guard_alignment_error": "card_reward_guard_alignment_error_rate",
    "card_reward_guard_invalid_obs": "card_reward_guard_invalid_obs_rate",
    "card_reward_guard_not_act1": "card_reward_guard_not_act1_rate",
    "card_reward_guard_deck_size": "card_reward_guard_deck_size_mean",
    "card_reward_guard_best_score": "card_reward_guard_best_score_mean",
    "card_reward_guard_selected_score": "card_reward_guard_selected_score_mean",
    "card_reward_guard_best_standalone_value": "card_reward_guard_best_standalone_value_mean",
    "card_reward_guard_best_combo_current_fit": "card_reward_guard_best_combo_current_fit_mean",
    "card_reward_guard_best_combo_missing_piece_fit": "card_reward_guard_best_combo_missing_piece_fit_mean",
    "card_reward_guard_best_combo_speculative_option": "card_reward_guard_best_combo_speculative_option_mean",
    "card_reward_guard_best_combo_orphan_risk": "card_reward_guard_best_combo_orphan_risk_mean",
    "card_reward_guard_best_combo_final_option_value": "card_reward_guard_best_combo_final_option_value_mean",
    "card_reward_guard_dynamic_min_useful_score": "card_reward_guard_dynamic_min_useful_score_mean",
    "card_reward_guard_threshold_relaxed": "card_reward_guard_threshold_relaxed_rate",
    "card_reward_guard_threshold_raised_near_boss": "card_reward_guard_threshold_raised_near_boss_rate",
    "card_reward_guard_orphan_reject": "card_reward_guard_orphan_reject_rate",
    "card_reward_guard_speculative_option_accepted": "card_reward_guard_speculative_option_accepted_rate",
    "card_reward_guard_top_candidate_rejected": "card_reward_guard_top_candidate_rejected_rate",
    "card_reward_guard_rejected_candidate_count": "card_reward_guard_rejected_candidate_count_mean",
    "card_reward_guard_damage_per_energy": "card_reward_guard_damage_per_energy_mean",
    "card_reward_guard_block_per_energy": "card_reward_guard_block_per_energy_mean",
    "card_reward_guard_expected_extra_draw": "card_reward_guard_expected_extra_draw_mean",
    "card_reward_guard_expected_cards_seen": "card_reward_guard_expected_cards_seen_mean",
    "card_reward_guard_expected_hand_attack_damage": "card_reward_guard_expected_hand_attack_damage_mean",
    "card_reward_guard_expected_hand_block": "card_reward_guard_expected_hand_block_mean",
    "card_reward_guard_expected_hand_attack_share": "card_reward_guard_expected_hand_attack_share_mean",
    "card_reward_guard_expected_hand_skill_share": "card_reward_guard_expected_hand_skill_share_mean",
    "card_reward_guard_expected_hand_power_share": "card_reward_guard_expected_hand_power_share_mean",
    "card_reward_guard_expected_hand_junk_share": "card_reward_guard_expected_hand_junk_share_mean",
    "card_reward_guard_expected_hand_useful_quality": "card_reward_guard_expected_hand_useful_quality_mean",
    "card_reward_guard_expected_playable_cards": "card_reward_guard_expected_playable_cards_mean",
    "card_reward_guard_expected_playable_attack_damage": "card_reward_guard_expected_playable_attack_damage_mean",
    "card_reward_guard_expected_playable_block": "card_reward_guard_expected_playable_block_mean",
    "card_reward_guard_expected_energy_utilization": "card_reward_guard_expected_energy_utilization_mean",
    "card_reward_guard_combo_option_value": "card_reward_guard_combo_option_value_mean",
    "card_reward_guard_combo_unmet_dependency": "card_reward_guard_combo_unmet_dependency_mean",
    "card_reward_guard_future_combo_window": "card_reward_guard_future_combo_window_mean",
    "card_reward_guard_future_combo_floor_window": "card_reward_guard_future_combo_floor_window_mean",
    "card_reward_guard_future_combo_route_window": "card_reward_guard_future_combo_route_window_mean",
    "card_reward_guard_future_reward_opportunity": "card_reward_guard_future_reward_opportunity_mean",
    "card_reward_guard_future_runway_observed": "card_reward_guard_future_runway_observed_rate",
    "card_reward_guard_delayed_payoff_option_value": "card_reward_guard_delayed_payoff_option_value_mean",
    "card_reward_guard_delayed_payoff_unrealized_risk": "card_reward_guard_delayed_payoff_unrealized_risk_mean",
    "card_reward_guard_delayed_payoff_maturity": "card_reward_guard_delayed_payoff_maturity_mean",
    "card_reward_guard_delayed_payoff_time_to_value": "card_reward_guard_delayed_payoff_time_to_value_mean",
    "card_reward_guard_low_block": "card_reward_guard_low_block_rate",
    "card_reward_guard_low_draw": "card_reward_guard_low_draw_rate",
    "card_reward_guard_low_attack": "card_reward_guard_low_attack_rate",
    "card_reward_guard_low_scaling": "card_reward_guard_low_scaling_rate",
    "card_reward_guard_reason_thin_deck": "card_reward_guard_reason_thin_deck_rate",
    "card_reward_guard_reason_low_block": "card_reward_guard_reason_low_block_rate",
    "card_reward_guard_reason_low_draw": "card_reward_guard_reason_low_draw_rate",
    "card_reward_guard_reason_low_attack": "card_reward_guard_reason_low_attack_rate",
    "card_reward_guard_reason_low_scaling": "card_reward_guard_reason_low_scaling_rate",
}


__all__ = [
    "CARD_REWARD_GUARD_METRIC_KEYS",
    "CARD_REWARD_GUARD_SEARCH_SUFFIXES",
    "apply_card_reward_guard",
    "card_reward_guard_metric_keys",
]
