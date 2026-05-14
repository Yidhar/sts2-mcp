"""Episode-level deck/build telemetry helpers.

This module is deliberately diagnostic-only:

* it does **not** change observation schema or replay tensor shapes;
* it computes compact episode metadata for post-mortem analysis;
* it keeps card-reward skip/pick accounting out of the already-large
  ``self_play.py`` rollout loop.

The metrics answer the current Act1 question: did the agent die with a thin or
low-quality deck because it repeatedly skipped card rewards, and what were the
deck's per-energy attack/block, rotation, and expected-hand quality signals?
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any

from sts2_env.action_compact import compact_action_signature
from sts2_env.deck_quality import DECK_QUALITY_V2_KEYS, deck_quality_v2


FINAL_DECK_QUALITY_TB_KEYS: tuple[tuple[str, str], ...] = (
    ("size", "deck_size_raw"),
    ("raw_avg_damage_per_energy", "raw_avg_damage_per_energy"),
    ("raw_avg_block_per_energy", "raw_avg_block_per_energy"),
    ("raw_expected_extra_draw_per_turn", "raw_expected_extra_draw_per_turn"),
    ("raw_expected_cards_seen_per_turn", "raw_expected_cards_seen_per_turn"),
    ("attack_density", "attack_density"),
    ("skill_density", "skill_density"),
    ("power_density", "power_density"),
    ("curse_density", "curse_density"),
    ("status_density", "status_density"),
    ("frontload_score", "frontload_score"),
    ("block_score", "block_score"),
    ("draw_engine_score", "draw_engine_score"),
    ("scaling_score", "scaling_score"),
    ("elite_readiness_score", "elite_readiness_score"),
    ("boss_readiness_score", "boss_readiness_score"),
    ("pollution_score", "pollution_score"),
    ("metadata_hit_rate", "metadata_hit_rate"),
    ("expected_hand_attack_cards", "expected_hand_attack_cards"),
    ("expected_hand_skill_cards", "expected_hand_skill_cards"),
    ("expected_hand_power_cards", "expected_hand_power_cards"),
    ("expected_hand_curse_cards", "expected_hand_curse_cards"),
    ("expected_hand_status_cards", "expected_hand_status_cards"),
    ("expected_hand_draw_cards", "expected_hand_draw_cards"),
    ("expected_hand_engine_cards", "expected_hand_engine_cards"),
    ("expected_hand_scaling_cards", "expected_hand_scaling_cards"),
    ("expected_hand_unplayable_cards", "expected_hand_unplayable_cards"),
    ("expected_hand_attack_damage_per_turn", "expected_hand_attack_damage_per_turn"),
    ("expected_hand_block_per_turn", "expected_hand_block_per_turn"),
    ("expected_hand_attack_share", "expected_hand_attack_share"),
    ("expected_hand_skill_share", "expected_hand_skill_share"),
    ("expected_hand_power_share", "expected_hand_power_share"),
    ("expected_hand_junk_share", "expected_hand_junk_share"),
    ("expected_hand_frontload_score", "expected_hand_frontload_score"),
    ("expected_hand_block_score", "expected_hand_block_score"),
    ("expected_hand_draw_score", "expected_hand_draw_score"),
    ("expected_hand_engine_score", "expected_hand_engine_score"),
    ("expected_hand_scaling_score", "expected_hand_scaling_score"),
    ("expected_hand_pollution_score", "expected_hand_pollution_score"),
    ("expected_hand_useful_quality_score", "expected_hand_useful_quality_score"),
    ("expected_hand_quality_attack_share", "expected_hand_quality_attack_share"),
    ("expected_hand_quality_block_share", "expected_hand_quality_block_share"),
    ("expected_hand_quality_draw_share", "expected_hand_quality_draw_share"),
    ("expected_hand_quality_scaling_share", "expected_hand_quality_scaling_share"),
    ("expected_hand_quality_pollution_share", "expected_hand_quality_pollution_share"),
    ("raw_expected_energy_budget_per_turn", "raw_expected_energy_budget_per_turn"),
    ("raw_expected_playable_cards_per_turn", "raw_expected_playable_cards_per_turn"),
    ("raw_expected_playable_energy_spent_per_turn", "raw_expected_playable_energy_spent_per_turn"),
    ("raw_expected_unspent_energy_per_turn", "raw_expected_unspent_energy_per_turn"),
    ("raw_expected_playable_attack_cards_per_turn", "raw_expected_playable_attack_cards_per_turn"),
    ("raw_expected_playable_skill_cards_per_turn", "raw_expected_playable_skill_cards_per_turn"),
    ("raw_expected_playable_power_cards_per_turn", "raw_expected_playable_power_cards_per_turn"),
    ("raw_expected_playable_attack_damage_per_turn", "raw_expected_playable_attack_damage_per_turn"),
    ("raw_expected_playable_block_per_turn", "raw_expected_playable_block_per_turn"),
    ("expected_energy_utilization_score", "expected_energy_utilization_score"),
    ("expected_playable_cards_score", "expected_playable_cards_score"),
    ("expected_unspent_energy_score", "expected_unspent_energy_score"),
    ("expected_playable_frontload_score", "expected_playable_frontload_score"),
    ("expected_playable_block_score", "expected_playable_block_score"),
    ("cost_curve_zero_share", "cost_curve_zero_share"),
    ("cost_curve_one_share", "cost_curve_one_share"),
    ("cost_curve_two_share", "cost_curve_two_share"),
    ("cost_curve_three_plus_share", "cost_curve_three_plus_share"),
    ("cost_curve_x_share", "cost_curve_x_share"),
    ("combo_component_density", "combo_component_density"),
    ("combo_enabler_density", "combo_enabler_density"),
    ("combo_payoff_density", "combo_payoff_density"),
    ("combo_option_value_score", "combo_option_value_score"),
    ("combo_unmet_dependency_score", "combo_unmet_dependency_score"),
    ("scaling_option_value_score", "scaling_option_value_score"),
    ("delayed_payoff_density", "delayed_payoff_density"),
    ("delayed_enabler_density", "delayed_enabler_density"),
    ("delayed_payoff_time_to_value_score", "delayed_payoff_time_to_value_score"),
    ("delayed_payoff_maturity_score", "delayed_payoff_maturity_score"),
    ("delayed_payoff_option_value_score", "delayed_payoff_option_value_score"),
    ("delayed_payoff_unrealized_risk_score", "delayed_payoff_unrealized_risk_score"),
)


DEATH_DECK_QUALITY_TB_KEYS: tuple[tuple[str, str], ...] = (
    ("size", "deck_size_raw"),
    ("raw_avg_damage_per_energy", "raw_avg_damage_per_energy"),
    ("raw_avg_block_per_energy", "raw_avg_block_per_energy"),
    ("raw_expected_extra_draw_per_turn", "raw_expected_extra_draw_per_turn"),
    ("raw_expected_cards_seen_per_turn", "raw_expected_cards_seen_per_turn"),
    ("frontload_score", "frontload_score"),
    ("block_score", "block_score"),
    ("draw_engine_score", "draw_engine_score"),
    ("scaling_score", "scaling_score"),
    ("elite_readiness_score", "elite_readiness_score"),
    ("boss_readiness_score", "boss_readiness_score"),
    ("pollution_score", "pollution_score"),
    ("metadata_hit_rate", "metadata_hit_rate"),
    ("expected_hand_attack_cards", "expected_hand_attack_cards"),
    ("expected_hand_skill_cards", "expected_hand_skill_cards"),
    ("expected_hand_power_cards", "expected_hand_power_cards"),
    ("expected_hand_curse_cards", "expected_hand_curse_cards"),
    ("expected_hand_status_cards", "expected_hand_status_cards"),
    ("expected_hand_draw_cards", "expected_hand_draw_cards"),
    ("expected_hand_engine_cards", "expected_hand_engine_cards"),
    ("expected_hand_scaling_cards", "expected_hand_scaling_cards"),
    ("expected_hand_unplayable_cards", "expected_hand_unplayable_cards"),
    ("expected_hand_attack_damage_per_turn", "expected_hand_attack_damage_per_turn"),
    ("expected_hand_block_per_turn", "expected_hand_block_per_turn"),
    ("expected_hand_attack_share", "expected_hand_attack_share"),
    ("expected_hand_skill_share", "expected_hand_skill_share"),
    ("expected_hand_power_share", "expected_hand_power_share"),
    ("expected_hand_junk_share", "expected_hand_junk_share"),
    ("expected_hand_frontload_score", "expected_hand_frontload_score"),
    ("expected_hand_block_score", "expected_hand_block_score"),
    ("expected_hand_draw_score", "expected_hand_draw_score"),
    ("expected_hand_engine_score", "expected_hand_engine_score"),
    ("expected_hand_scaling_score", "expected_hand_scaling_score"),
    ("expected_hand_pollution_score", "expected_hand_pollution_score"),
    ("expected_hand_useful_quality_score", "expected_hand_useful_quality_score"),
    ("expected_hand_quality_attack_share", "expected_hand_quality_attack_share"),
    ("expected_hand_quality_block_share", "expected_hand_quality_block_share"),
    ("expected_hand_quality_draw_share", "expected_hand_quality_draw_share"),
    ("expected_hand_quality_scaling_share", "expected_hand_quality_scaling_share"),
    ("expected_hand_quality_pollution_share", "expected_hand_quality_pollution_share"),
    ("raw_expected_energy_budget_per_turn", "raw_expected_energy_budget_per_turn"),
    ("raw_expected_playable_cards_per_turn", "raw_expected_playable_cards_per_turn"),
    ("raw_expected_playable_energy_spent_per_turn", "raw_expected_playable_energy_spent_per_turn"),
    ("raw_expected_unspent_energy_per_turn", "raw_expected_unspent_energy_per_turn"),
    ("raw_expected_playable_attack_cards_per_turn", "raw_expected_playable_attack_cards_per_turn"),
    ("raw_expected_playable_skill_cards_per_turn", "raw_expected_playable_skill_cards_per_turn"),
    ("raw_expected_playable_power_cards_per_turn", "raw_expected_playable_power_cards_per_turn"),
    ("raw_expected_playable_attack_damage_per_turn", "raw_expected_playable_attack_damage_per_turn"),
    ("raw_expected_playable_block_per_turn", "raw_expected_playable_block_per_turn"),
    ("expected_energy_utilization_score", "expected_energy_utilization_score"),
    ("expected_playable_cards_score", "expected_playable_cards_score"),
    ("expected_unspent_energy_score", "expected_unspent_energy_score"),
    ("expected_playable_frontload_score", "expected_playable_frontload_score"),
    ("expected_playable_block_score", "expected_playable_block_score"),
    ("cost_curve_zero_share", "cost_curve_zero_share"),
    ("cost_curve_one_share", "cost_curve_one_share"),
    ("cost_curve_two_share", "cost_curve_two_share"),
    ("cost_curve_three_plus_share", "cost_curve_three_plus_share"),
    ("cost_curve_x_share", "cost_curve_x_share"),
    ("combo_component_density", "combo_component_density"),
    ("combo_enabler_density", "combo_enabler_density"),
    ("combo_payoff_density", "combo_payoff_density"),
    ("combo_option_value_score", "combo_option_value_score"),
    ("combo_unmet_dependency_score", "combo_unmet_dependency_score"),
    ("scaling_option_value_score", "scaling_option_value_score"),
    ("delayed_payoff_density", "delayed_payoff_density"),
    ("delayed_enabler_density", "delayed_enabler_density"),
    ("delayed_payoff_time_to_value_score", "delayed_payoff_time_to_value_score"),
    ("delayed_payoff_maturity_score", "delayed_payoff_maturity_score"),
    ("delayed_payoff_option_value_score", "delayed_payoff_option_value_score"),
    ("delayed_payoff_unrealized_risk_score", "delayed_payoff_unrealized_risk_score"),
)


CARD_REWARD_TB_KEYS: tuple[tuple[str, str], ...] = (
    ("seen", "card_reward_seen_count"),
    ("pick", "card_reward_pick_count"),
    ("skip", "card_reward_skip_count"),
    ("pick_rate", "card_reward_pick_rate"),
    ("skip_rate", "card_reward_skip_rate"),
    ("consecutive_skip_current", "card_reward_consecutive_skip_current"),
    ("consecutive_skip_max", "card_reward_consecutive_skip_max"),
)


def _finite_float(value: Any, default: float = 0.0) -> float:
    try:
        out = float(value)
    except (TypeError, ValueError):
        return float(default)
    return out if math.isfinite(out) else float(default)


def _stringify(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, (str, int, float, bool)):
        return str(value)
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
            "card_id",
            "reward_type",
        ):
            raw = value.get(key)
            if raw is not None:
                parts.append(str(raw))
        semantic = value.get("semantic")
        if isinstance(semantic, dict):
            for key in ("family", "domain", "semantic_key"):
                raw = semantic.get(key)
                if raw is not None:
                    parts.append(str(raw))
        reward = value.get("reward")
        if isinstance(reward, dict):
            for key in ("type", "title", "name"):
                raw = reward.get(key)
                if raw is not None:
                    parts.append(str(raw))
        card = value.get("card")
        if isinstance(card, dict):
            for key in ("id", "title", "name", "type"):
                raw = card.get(key)
                if raw is not None:
                    parts.append(str(raw))
        return " ".join(parts)
    return str(value)


def _lower_text(value: Any) -> str:
    return _stringify(value).strip().lower()


def _dict_at_path(root: Any, path: tuple[str, ...]) -> Any:
    node = root
    for key in path:
        if not isinstance(node, dict):
            return None
        node = node.get(key)
    return node


def _normalise_deck_list(value: Any) -> list[dict[str, Any]]:
    if not isinstance(value, list):
        return []
    return [card for card in value if isinstance(card, dict)]


def _recursive_find_deck_cards(root: Any, *, depth: int = 0, max_depth: int = 4) -> list[dict[str, Any]]:
    """Conservative bounded search for ``deck_cards`` in bridge payloads.

    The known paths cover current full-run observations.  The shallow fallback
    makes the telemetry robust to harmless wrapper changes without walking huge
    replay payloads.
    """

    if depth > max_depth:
        return []
    if isinstance(root, dict):
        direct = _normalise_deck_list(root.get("deck_cards"))
        if direct:
            return direct
        for key in (
            "player",
            "transition_state",
            "raw_obs",
            "state",
            "observation",
            "obs",
            "payload",
            "info",
        ):
            if key in root:
                found = _recursive_find_deck_cards(root.get(key), depth=depth + 1, max_depth=max_depth)
                if found:
                    return found
        # Last resort: inspect a few shallow dict/list children, but avoid
        # expensive scans of legal action arrays or large replay histories.
        inspected = 0
        for key, value in root.items():
            if key in {"legal_actions", "legal_actions_compact", "steps", "trajectory"}:
                continue
            if not isinstance(value, (dict, list)):
                continue
            found = _recursive_find_deck_cards(value, depth=depth + 1, max_depth=max_depth)
            if found:
                return found
            inspected += 1
            if inspected >= 16:
                break
    elif isinstance(root, list) and depth < max_depth:
        for item in root[:16]:
            found = _recursive_find_deck_cards(item, depth=depth + 1, max_depth=max_depth)
            if found:
                return found
    return []


def extract_deck_cards_from_obs_like(payload: Any) -> list[dict[str, Any]]:
    """Return a best-effort ``player.deck_cards`` list from an obs/info payload."""

    if not isinstance(payload, dict):
        return []
    known_paths = (
        ("player", "deck_cards"),
        ("transition_state", "player", "deck_cards"),
        ("raw_obs", "player", "deck_cards"),
        ("state", "player", "deck_cards"),
        ("observation", "player", "deck_cards"),
        ("obs", "player", "deck_cards"),
    )
    for path in known_paths:
        cards = _normalise_deck_list(_dict_at_path(payload, path))
        if cards:
            return cards
    return _recursive_find_deck_cards(payload)


def _card_signal(card: dict[str, Any], key: str) -> float:
    profile = card.get("card_effect_profile") if isinstance(card.get("card_effect_profile"), dict) else {}
    signals = profile.get("semantic_signals") if isinstance(profile.get("semantic_signals"), dict) else {}
    return _finite_float(signals.get(key), 0.0)


def compact_deck_cards(deck_cards: list[Any] | tuple[Any, ...] | None, *, limit: int = 80) -> list[dict[str, Any]]:
    """Compact a deck into bounded JSON-safe rows for death slices/metadata."""

    compact: list[dict[str, Any]] = []
    cards = deck_cards if isinstance(deck_cards, (list, tuple)) else []
    for idx, card in enumerate(cards[: max(0, int(limit))]):
        if not isinstance(card, dict):
            continue
        row: dict[str, Any] = {
            "idx": int(idx),
            "id": card.get("id") or card.get("card_id") or card.get("model_id") or card.get("modelId"),
            "title": card.get("title") or card.get("name") or card.get("localized_title"),
            "type": card.get("type") or card.get("card_type"),
            "cost": card.get("cost") if card.get("cost") is not None else card.get("energy_cost"),
            "energy_cost_text": card.get("energy_cost_text") or card.get("cost_text"),
            "upgraded": card.get("upgraded"),
            "upgrade_level": card.get("upgrade_level"),
            "rarity": card.get("rarity"),
        }
        for signal_key in ("damage", "block", "draw"):
            signal_value = _card_signal(card, signal_key)
            if signal_value != 0.0:
                row[signal_key] = signal_value
        compact.append({key: value for key, value in row.items() if value is not None})
    return compact


def compute_deck_quality_summary(deck_cards: list[Any] | tuple[Any, ...] | None) -> dict[str, float]:
    """Compute ``deck_quality_v2`` and guarantee finite floats for all keys."""

    try:
        raw = deck_quality_v2(deck_cards)
    except Exception:
        raw = {}
    out: dict[str, float] = {}
    for key in DECK_QUALITY_V2_KEYS:
        out[key] = _finite_float(raw.get(key, 0.0), 0.0) if isinstance(raw, dict) else 0.0
    return out


def _signature_for(action: Any) -> dict[str, Any]:
    if isinstance(action, dict) and (
        "action_id" in action
        or "kind" in action
        or "semantic" in action
        or "selection" in action
        or "card_id" in action
    ):
        # It might already be compact, but compact_action_signature is safe for
        # normal bridge actions and keeps semantics consistent.
        compact = compact_action_signature(action)
        if compact:
            return compact
        return action
    return {}


def _semantic_family(payload: Any) -> str:
    if not isinstance(payload, dict):
        return ""
    semantic = payload.get("semantic") if isinstance(payload.get("semantic"), dict) else {}
    return str(semantic.get("family") or "").strip().lower()


def _reward_type(payload: Any) -> str:
    if not isinstance(payload, dict):
        return ""
    reward = payload.get("reward") if isinstance(payload.get("reward"), dict) else {}
    return str(reward.get("type") or payload.get("reward_type") or "").strip().lower()


def _has_card_payload(payload: Any) -> bool:
    if not isinstance(payload, dict):
        return False
    card = payload.get("card")
    return bool(
        isinstance(card, dict)
        or payload.get("card_id")
        or payload.get("cardId")
        or payload.get("card_title")
    )


def _is_card_reward_action(action: Any, signature: dict[str, Any] | None = None) -> bool:
    sig = signature if isinstance(signature, dict) else _signature_for(action)
    text = f"{_lower_text(action)} {_lower_text(sig)}"
    family = _semantic_family(sig) or _semantic_family(action)
    kind = str((sig or {}).get("kind") or (action.get("kind") if isinstance(action, dict) else "") or "").lower()
    surface = str((sig or {}).get("surface") or (action.get("surface") if isinstance(action, dict) else "") or "").lower()
    selection_prompt = str((sig or {}).get("selection_prompt") or "").lower()
    return bool(
        family == "card_reward"
        or kind in {"card_reward", "choose_card_reward", "skip_card_reward"}
        or "card_reward" in text
        or "card reward" in text
        or (surface == "card_reward")
        or ("card" in selection_prompt and "reward" in selection_prompt)
        or (_reward_type(action) == "card" and "reward" in text)
        or (_reward_type(sig) == "card" and "reward" in text)
    )


def _is_card_reward_context(
    *,
    decision_domain: str,
    phase: str,
    legal_actions: list[Any],
    chosen_action: dict[str, Any] | None,
    chosen_signature: dict[str, Any] | None,
    selected_family: str,
) -> bool:
    phase_text = str(phase or "").strip().lower()
    domain_text = str(decision_domain or "").strip().lower()
    family_text = str(selected_family or "").strip().lower()
    if domain_text == "card_reward" or family_text == "card_reward":
        return True
    if "card_reward" in phase_text or "card reward" in phase_text:
        return True
    if "reward" in phase_text and (
        _is_card_reward_action(chosen_action, chosen_signature)
        or any(_is_card_reward_action(action) for action in legal_actions[:96])
    ):
        return True
    return bool(
        _is_card_reward_action(chosen_action, chosen_signature)
        or any(_is_card_reward_action(action) for action in legal_actions[:96])
    )


def _is_pick_action(
    action: dict[str, Any] | None,
    signature: dict[str, Any] | None,
    *,
    card_reward_context: bool,
) -> bool:
    if not card_reward_context:
        return False
    sig = signature if isinstance(signature, dict) else _signature_for(action)
    text = f"{_lower_text(action)} {_lower_text(sig)}"
    selection = str(
        (sig or {}).get("selection")
        or (sig or {}).get("selection_action")
        or (action.get("selection") if isinstance(action, dict) else "")
        or ""
    ).strip().lower()
    kind = str((sig or {}).get("kind") or (action.get("kind") if isinstance(action, dict) else "") or "").strip().lower()
    family = _semantic_family(sig) or _semantic_family(action)
    if selection in {"pick", "take", "choose", "select", "add", "claim"}:
        return True
    if kind in {"card_reward", "choose_card_reward"}:
        return True
    if family == "card_reward" and (sig.get("card_id") if isinstance(sig, dict) else None):
        return True
    if _reward_type(action) == "card" or _reward_type(sig) == "card":
        return True
    return bool(_has_card_payload(action) or _has_card_payload(sig) or ("choose" in text and "card" in text))


def _is_skip_action(
    action: dict[str, Any] | None,
    signature: dict[str, Any] | None,
    *,
    card_reward_context: bool,
) -> bool:
    if not card_reward_context:
        return False
    sig = signature if isinstance(signature, dict) else _signature_for(action)
    text = f"{_lower_text(action)} {_lower_text(sig)}"
    selection = str(
        (sig or {}).get("selection")
        or (sig or {}).get("selection_action")
        or (action.get("selection") if isinstance(action, dict) else "")
        or ""
    ).strip().lower()
    kind = str((sig or {}).get("kind") or (action.get("kind") if isinstance(action, dict) else "") or "").strip().lower()
    family = _semantic_family(sig) or _semantic_family(action)
    if selection in {"skip", "proceed", "leave", "cancel", "close", "none"}:
        return True
    if kind == "skip_card_reward":
        return True
    if family in {"proceed", "skip"}:
        return True
    return bool(
        "skip" in text
        or "card_reward:skip" in text
        or "skip_card_reward" in text
        or ("proceed" in text and "reward" in text)
    )


@dataclass
class CardRewardEpisodeTracker:
    """Track card-reward decisions across one episode."""

    seen: int = 0
    pick: int = 0
    skip: int = 0
    other: int = 0
    consecutive_skip_current: int = 0
    consecutive_skip_max: int = 0

    def update(
        self,
        *,
        decision_domain: str,
        phase: str,
        legal_actions: list[Any],
        chosen_action: dict[str, Any] | None,
        chosen_signature: dict[str, Any] | None,
        selected_family: str,
        selection: str,
    ) -> None:
        context = _is_card_reward_context(
            decision_domain=decision_domain,
            phase=phase,
            legal_actions=legal_actions if isinstance(legal_actions, list) else [],
            chosen_action=chosen_action,
            chosen_signature=chosen_signature,
            selected_family=selected_family,
        )
        if not context:
            return

        self.seen += 1
        # The rollout loop passes a normalised ``selection`` from the compact
        # signature.  Copy it into the signature/action test path when present.
        if selection and isinstance(chosen_signature, dict) and not chosen_signature.get("selection"):
            chosen_signature = {**chosen_signature, "selection": selection}

        picked = _is_pick_action(chosen_action, chosen_signature, card_reward_context=True)
        skipped = _is_skip_action(chosen_action, chosen_signature, card_reward_context=True)
        if picked and not skipped:
            self.pick += 1
            self.consecutive_skip_current = 0
            return
        if picked and skipped:
            # Ambiguous compact payloads occasionally carry a generic "reward"
            # action id plus a card payload.  Prefer "pick" if an actual card is
            # present, otherwise treat it as a skip.
            if _has_card_payload(chosen_action) or _has_card_payload(chosen_signature):
                self.pick += 1
                self.consecutive_skip_current = 0
                return

        self.skip += 1
        if not skipped:
            self.other += 1
        self.consecutive_skip_current += 1
        self.consecutive_skip_max = max(self.consecutive_skip_max, self.consecutive_skip_current)

    def as_metadata(self) -> dict[str, float]:
        seen_safe = max(int(self.seen), 1)
        return {
            "card_reward_seen_count": float(self.seen),
            "card_reward_pick_count": float(self.pick),
            "card_reward_skip_count": float(self.skip),
            "card_reward_other_count": float(self.other),
            "card_reward_pick_rate": float(self.pick) / float(seen_safe),
            "card_reward_skip_rate": float(self.skip) / float(seen_safe),
            "card_reward_consecutive_skip_current": float(self.consecutive_skip_current),
            "card_reward_consecutive_skip_max": float(self.consecutive_skip_max),
        }
