"""Card-reward pick-quality hard guard.

The anti-skip guard in :mod:`muzero.training.card_reward_guard` only fixes the
pathology where the policy skips useful Act1 card rewards.  Recent full-run
death decks show a different build failure mode: the agent now usually *does*
pick a card, but it can pick a clearly weaker candidate while the deck still has
large attack/block/draw/scaling deficits.

This module is deliberately narrow and fail-open:

* only card-reward pick surfaces are considered;
* only Act1 / unknown-act observations with an objectively weak deck are
  eligible;
* candidates reuse the existing deck-quality/card scoring code, including
  Body-Slam/orphan-combo rejection;
* an override requires a large score margin or a selected candidate that the
  existing scorer would reject;
* telemetry is separate from the anti-skip guard so we can tell whether this
  guard is actually moving bad picks.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from muzero.training.card_reward_guard import (
    _CandidateScore,
    _card_from_action,
    _candidate_rejection_reason,
    _deck_deficits,
    _dynamic_min_useful_score,
    _finite_float,
    _future_combo_runway,
    _is_act1_or_unknown,
    _is_card_reward_action,
    _is_pick_action,
    _score_candidate_card,
)
from muzero.diagnostics.deck_build_metrics import (
    compute_deck_quality_summary,
    extract_deck_cards_from_obs_like,
)
from sts2_env.observation_v2 import MAX_ACTIONS


CARD_REWARD_PICK_QUALITY_GUARD_SEARCH_SUFFIXES: dict[str, str] = {
    "card_reward_pick_quality_guard_context": "card_reward_pick_quality_guard_context_rate",
    "card_reward_pick_quality_guard_applicable": "card_reward_pick_quality_guard_applicable_rate",
    "card_reward_pick_quality_guard_selected_pick": "card_reward_pick_quality_guard_selected_pick_rate",
    "card_reward_pick_quality_guard_pick_available": "card_reward_pick_quality_guard_pick_available_rate",
    "card_reward_pick_quality_guard_accepted_available": "card_reward_pick_quality_guard_accepted_available_rate",
    "card_reward_pick_quality_guard_applied": "card_reward_pick_quality_guard_applied_rate",
    "card_reward_pick_quality_guard_override": "card_reward_pick_quality_guard_override_rate",
    "card_reward_pick_quality_guard_alignment_error": "card_reward_pick_quality_guard_alignment_error_rate",
    "card_reward_pick_quality_guard_invalid_obs": "card_reward_pick_quality_guard_invalid_obs_rate",
    "card_reward_pick_quality_guard_not_act1": "card_reward_pick_quality_guard_not_act1_rate",
    "card_reward_pick_quality_guard_no_deficit": "card_reward_pick_quality_guard_no_deficit_rate",
    "card_reward_pick_quality_guard_selected_rejected": "card_reward_pick_quality_guard_selected_rejected_rate",
    "card_reward_pick_quality_guard_best_score": "card_reward_pick_quality_guard_best_score_mean",
    "card_reward_pick_quality_guard_selected_score": "card_reward_pick_quality_guard_selected_score_mean",
    "card_reward_pick_quality_guard_best_minus_selected": "card_reward_pick_quality_guard_best_minus_selected_mean",
    "card_reward_pick_quality_guard_dynamic_min_useful_score": (
        "card_reward_pick_quality_guard_dynamic_min_useful_score_mean"
    ),
    "card_reward_pick_quality_guard_deck_size": "card_reward_pick_quality_guard_deck_size_mean",
    "card_reward_pick_quality_guard_low_block": "card_reward_pick_quality_guard_low_block_rate",
    "card_reward_pick_quality_guard_low_draw": "card_reward_pick_quality_guard_low_draw_rate",
    "card_reward_pick_quality_guard_low_attack": "card_reward_pick_quality_guard_low_attack_rate",
    "card_reward_pick_quality_guard_low_scaling": "card_reward_pick_quality_guard_low_scaling_rate",
    "card_reward_pick_quality_guard_thin_deck": "card_reward_pick_quality_guard_thin_deck_rate",
}


def card_reward_pick_quality_guard_metric_keys() -> tuple[str, ...]:
    return tuple(CARD_REWARD_PICK_QUALITY_GUARD_SEARCH_SUFFIXES.keys())


def _write_deficit_metrics(stats: dict[str, Any], deficits: Any, deck_size: float) -> None:
    stats["card_reward_pick_quality_guard_deck_size"] = float(deck_size)
    stats["card_reward_pick_quality_guard_thin_deck"] = 1.0 if bool(deficits.thin_deck) else 0.0
    stats["card_reward_pick_quality_guard_low_attack"] = 1.0 if bool(deficits.low_attack) else 0.0
    stats["card_reward_pick_quality_guard_low_block"] = 1.0 if bool(deficits.low_block) else 0.0
    stats["card_reward_pick_quality_guard_low_draw"] = 1.0 if bool(deficits.low_draw) else 0.0
    stats["card_reward_pick_quality_guard_low_scaling"] = 1.0 if bool(deficits.low_scaling) else 0.0


def _selected_card_score(scored: list[_CandidateScore], selected_idx: int) -> _CandidateScore | None:
    for item in scored:
        if int(item.index) == int(selected_idx):
            return item
    return None


def _action_at(
    idx: int,
    *,
    legal_actions: list[Any],
    full_legal_actions: list[Any] | None,
) -> Any:
    full = full_legal_actions[idx] if isinstance(full_legal_actions, list) and idx < len(full_legal_actions) else None
    compact = legal_actions[idx] if idx < len(legal_actions) else None
    if _card_from_action(full):
        return full
    return compact if compact is not None else full


def apply_card_reward_pick_quality_guard(
    *,
    action_idx: int,
    legal_actions: list[Any] | None,
    full_legal_actions: list[Any] | None,
    action_mask: Any,
    raw_obs: dict[str, Any] | None,
    search_stats: dict[str, Any],
    min_useful_score: float = 0.45,
    override_margin: float = 0.35,
    rejected_selected_margin: float = 0.10,
) -> int:
    """Return a possibly better card-reward pick index.

    ``override_margin`` intentionally stays large: this guard should correct
    obvious Act1 build mistakes, not replace the learned card-reward policy.
    If the selected candidate is rejected by the existing scorer (junk/orphan/
    below dynamic threshold), a smaller margin is allowed because the alternative
    has already passed the usefulness filter.
    """

    stats = search_stats if isinstance(search_stats, dict) else {}
    for key in CARD_REWARD_PICK_QUALITY_GUARD_SEARCH_SUFFIXES:
        stats.setdefault(key, 0.0)

    if not isinstance(legal_actions, list) or not legal_actions:
        return int(action_idx)
    try:
        mask_np = np.asarray(action_mask, dtype="float32").reshape(-1)
    except Exception:
        stats["card_reward_pick_quality_guard_alignment_error"] = 1.0
        return int(action_idx)

    legal_count = min(len(legal_actions), int(mask_np.shape[0]) if mask_np.size else 0, MAX_ACTIONS)
    if legal_count <= 0 or not (0 <= int(action_idx) < legal_count) or mask_np[int(action_idx)] <= 0:
        stats["card_reward_pick_quality_guard_alignment_error"] = 1.0
        return int(action_idx)
    if isinstance(full_legal_actions, list) and len(full_legal_actions) < legal_count:
        stats["card_reward_pick_quality_guard_alignment_error"] = 1.0
        return int(action_idx)

    selected = _action_at(int(action_idx), legal_actions=legal_actions, full_legal_actions=full_legal_actions)
    context = _is_card_reward_action(selected) or any(
        _is_card_reward_action(_action_at(i, legal_actions=legal_actions, full_legal_actions=full_legal_actions))
        for i in range(legal_count)
    )
    if not context:
        return int(action_idx)
    stats["card_reward_pick_quality_guard_context"] = 1.0

    selected_is_pick = _is_pick_action(selected, card_reward_context=True)
    stats["card_reward_pick_quality_guard_selected_pick"] = 1.0 if selected_is_pick else 0.0
    if not selected_is_pick:
        return int(action_idx)

    if not isinstance(raw_obs, dict):
        stats["card_reward_pick_quality_guard_invalid_obs"] = 1.0
        return int(action_idx)
    deck_cards = extract_deck_cards_from_obs_like(raw_obs)
    if not deck_cards:
        stats["card_reward_pick_quality_guard_invalid_obs"] = 1.0
        return int(action_idx)

    summary = compute_deck_quality_summary(deck_cards)
    deficits = _deck_deficits(summary)
    deck_size = _finite_float(summary.get("deck_size_raw"), 0.0)
    _write_deficit_metrics(stats, deficits, deck_size)
    if not _is_act1_or_unknown(raw_obs, deck_size):
        stats["card_reward_pick_quality_guard_not_act1"] = 1.0
        return int(action_idx)
    if not deficits.any:
        stats["card_reward_pick_quality_guard_no_deficit"] = 1.0
        return int(action_idx)

    future_runway = _future_combo_runway(raw_obs)
    scored: list[_CandidateScore] = []
    accepted: list[tuple[_CandidateScore, float]] = []
    selected_rejected = True
    selected_threshold = float(min_useful_score)
    selected_score_obj: _CandidateScore | None = None
    for idx in range(legal_count):
        if mask_np[idx] <= 0:
            continue
        action = _action_at(idx, legal_actions=legal_actions, full_legal_actions=full_legal_actions)
        if not _is_pick_action(action, card_reward_context=True):
            continue
        card = _card_from_action(action)
        if not card:
            continue
        raw_score = _score_candidate_card(card, deficits, summary, future_combo_window=future_runway.score)
        candidate = _CandidateScore(
            index=int(idx),
            score=float(raw_score.score),
            card=card,
            junk=bool(raw_score.junk),
            standalone_value=float(raw_score.standalone_value),
            combo_current_fit=float(raw_score.combo_current_fit),
            combo_missing_piece_fit=float(raw_score.combo_missing_piece_fit),
            combo_speculative_option=float(raw_score.combo_speculative_option),
            combo_orphan_risk=float(raw_score.combo_orphan_risk),
            combo_final_option_value=float(raw_score.combo_final_option_value),
        )
        scored.append(candidate)
        threshold, _flags = _dynamic_min_useful_score(
            base=float(min_useful_score),
            deck_size=deck_size,
            deficits=deficits,
            future_runway=future_runway,
            best=candidate,
            raw_obs=raw_obs,
        )
        reject_reason = _candidate_rejection_reason(candidate, float(threshold))
        if int(idx) == int(action_idx):
            selected_score_obj = candidate
            selected_threshold = float(threshold)
            selected_rejected = reject_reason is not None
        if reject_reason is None:
            accepted.append((candidate, float(threshold)))

    if not scored:
        return int(action_idx)
    stats["card_reward_pick_quality_guard_pick_available"] = 1.0
    if selected_score_obj is None:
        stats["card_reward_pick_quality_guard_alignment_error"] = 1.0
        return int(action_idx)
    stats["card_reward_pick_quality_guard_selected_score"] = float(selected_score_obj.score)
    stats["card_reward_pick_quality_guard_dynamic_min_useful_score"] = float(selected_threshold)
    stats["card_reward_pick_quality_guard_selected_rejected"] = 1.0 if selected_rejected else 0.0

    if not accepted:
        return int(action_idx)
    stats["card_reward_pick_quality_guard_accepted_available"] = 1.0
    best, best_threshold = max(accepted, key=lambda item: item[0].score)
    stats["card_reward_pick_quality_guard_best_score"] = float(best.score)
    stats["card_reward_pick_quality_guard_dynamic_min_useful_score"] = float(best_threshold)
    gap = float(best.score) - float(selected_score_obj.score)
    stats["card_reward_pick_quality_guard_best_minus_selected"] = float(gap)
    stats["card_reward_pick_quality_guard_applicable"] = 1.0

    if int(best.index) == int(action_idx):
        return int(action_idx)
    needed_margin = float(rejected_selected_margin) if selected_rejected else float(override_margin)
    if gap < needed_margin:
        return int(action_idx)
    override_idx = int(best.index)
    if not (0 <= override_idx < legal_count) or mask_np[override_idx] <= 0:
        stats["card_reward_pick_quality_guard_alignment_error"] = 1.0
        return int(action_idx)

    stats["card_reward_pick_quality_guard_applied"] = 1.0
    stats["card_reward_pick_quality_guard_override"] = 1.0
    return override_idx


__all__ = [
    "CARD_REWARD_PICK_QUALITY_GUARD_SEARCH_SUFFIXES",
    "apply_card_reward_pick_quality_guard",
    "card_reward_pick_quality_guard_metric_keys",
]
