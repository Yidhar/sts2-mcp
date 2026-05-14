"""Pure action/card feature extraction helpers.

These functions were originally embedded in ``muzero.train.MuZeroTrainer``.
Keeping them here makes card/action policy tests independent from the training
orchestrator and prevents future strategy logic from growing the monolithic
``train.py`` file.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

from sts2_env.card_effect_profile import aggregate_card_effect_profile_semantics


def semantic_family(action: Any) -> str:
    """Return the normalized semantic action family."""

    if not isinstance(action, dict):
        return ""
    semantic = action.get("semantic") if isinstance(action.get("semantic"), dict) else {}
    family = str(semantic.get("family") or "").strip().lower()
    if family:
        return family
    kind = str(action.get("kind") or action.get("action_type") or "").strip().lower()
    action_id = str(action.get("action_id") or "").strip().lower()
    if action_id == "end_turn" or kind == "end_turn":
        return "end_turn"
    if kind in {"play_card", "use_potion", "discard_potion"}:
        return kind
    if action_id.startswith("play_card"):
        return "play_card"
    if action_id.startswith("use_potion"):
        return "use_potion"
    return kind


def action_roles(action: Any) -> set[str]:
    """Return normalized semantic role labels for an action."""

    if not isinstance(action, dict):
        return set()
    semantic = action.get("semantic") if isinstance(action.get("semantic"), dict) else {}
    roles = semantic.get("roles")
    if not isinstance(roles, list):
        return set()
    return {str(role).strip().lower() for role in roles if str(role).strip()}


def action_source(action: Any) -> dict[str, Any]:
    """Return the first nested payload that owns card/potion/relic/reward data."""

    if not isinstance(action, dict):
        return {}
    for key in ("card", "potion", "relic", "reward"):
        source = action.get(key)
        if isinstance(source, dict):
            return source
    return action


def action_metric(action: Any, key: str) -> float:
    """Read a numeric action/card effect metric from compact or typed payloads."""

    if not isinstance(action, dict):
        return 0.0
    aliases = {
        # Keep amount fields before boolean typed flags.  Compact semantic
        # actions may expose ``typed_gain_energy`` as an amount, while the full
        # aggregate profile uses it as a flag plus ``typed_gain_energy_amount``.
        "energy": ("energy", "energy_gain", "typed_gain_energy_amount", "typed_gain_energy"),
        "energy_gain": ("energy_gain", "energy", "typed_gain_energy_amount", "typed_gain_energy"),
        "draw": ("draw", "cards_drawn", "typed_draw_amount", "typed_draw_cards"),
        "hp_loss": ("hp_loss", "hp_cost", "typed_hp_loss"),
        "hp_cost": ("hp_cost", "hp_loss", "typed_hp_loss"),
        "damage": ("damage", "total_damage"),
        "block": ("block", "total_block"),
    }
    keys = aliases.get(key, (key,))
    semantic = action.get("semantic") if isinstance(action.get("semantic"), dict) else {}
    semantic_profile = semantic.get("card_effect_profile") if isinstance(semantic.get("card_effect_profile"), dict) else {}
    source = action_source(action)
    source_profile = source.get("card_effect_profile") if isinstance(source.get("card_effect_profile"), dict) else {}
    action_profile = action.get("card_effect_profile") if isinstance(action.get("card_effect_profile"), dict) else {}
    typed_effects: dict[str, float] = {}
    try:
        if isinstance(action.get("card"), dict):
            typed_effects = aggregate_card_effect_profile_semantics(action.get("card"))
        elif isinstance(source, dict) and ("card_effect_profile" in source or "operations" in source):
            typed_effects = aggregate_card_effect_profile_semantics(source)
        elif "card_effect_profile" in action or "operations" in action:
            typed_effects = aggregate_card_effect_profile_semantics(action)
    except Exception:
        typed_effects = {}
    containers = (
        typed_effects,
        semantic_profile,
        semantic,
        action_profile,
        action,
        source_profile,
        source,
    )
    best = 0.0
    for source_obj in containers:
        if not isinstance(source_obj, dict):
            continue
        for candidate_key in keys:
            if candidate_key not in source_obj:
                continue
            value = source_obj.get(candidate_key)
            if isinstance(value, (list, dict)):
                continue
            try:
                best = max(best, float(value or 0.0))
            except (TypeError, ValueError):
                pass
    return float(best)


def action_immediate_impact(action: Any) -> float:
    """Heuristic immediate impact score used by several action-quality guards."""

    roles = action_roles(action)
    damage = action_metric(action, "damage")
    block = action_metric(action, "block")
    hits = max(action_metric(action, "hits"), 1.0 if damage > 0.0 else 0.0)
    draw = action_metric(action, "draw")
    energy_gain = max(action_metric(action, "energy"), action_metric(action, "energy_gain"))
    hp_loss = max(action_metric(action, "hp_loss"), action_metric(action, "hp_cost"))
    return float(
        damage
        + 0.75 * block
        + 1.5 * max(hits - 1.0, 0.0)
        + 2.0 * min(max(draw, 0.0), 3.0)
        + 2.0 * min(max(energy_gain, 0.0), 3.0)
        - 0.5 * min(max(hp_loss, 0.0), 6.0)
        + (8.0 if roles.intersection({"debuff", "weak", "vulnerable", "poison", "exhaust", "discard"}) else 0.0)
        + (6.0 if roles.intersection({"scaling", "power", "draw", "energy", "resource", "retain"}) else 0.0)
    )


def is_zero_cost_action(action: Any) -> bool:
    """Return whether the action is explicitly zero-cost, excluding X-cost."""

    if not isinstance(action, dict):
        return False
    source = action_source(action)
    cost = action.get("card_cost", source.get("cost"))
    if isinstance(cost, str) and cost.strip().upper() == "X":
        return False
    try:
        return float(cost) <= 0.0
    except (TypeError, ValueError):
        return False


def action_text(action: Any) -> str:
    """Concatenate stable action/source text fields for fallback matching."""

    if not isinstance(action, dict):
        return ""
    source = action_source(action)
    parts: list[str] = []
    for container in (action, source):
        if isinstance(container, dict):
            for key in ("action_id", "kind", "title", "label", "id", "name", "description", "text", "canonical_text", "type"):
                value = str(container.get(key) or "").strip()
                if value:
                    parts.append(value)
            keywords = container.get("keywords")
            if isinstance(keywords, list):
                parts.extend(str(x or "").strip() for x in keywords if str(x or "").strip())
    return " | ".join(parts).lower()


def is_positive_combat_action(
    action: Any,
    *,
    is_facing_change_action: Callable[[Any], bool] | None = None,
) -> bool:
    """Return whether a legal combat action represents non-end-turn progress."""

    if not isinstance(action, dict):
        return False
    family = semantic_family(action)
    if family in {"end_turn", "discard_potion"}:
        return False
    # card.type decisive:
    #   Attack / Skill / Power — always positive progress when legal
    #     (bridge-provided roles/damage can be empty for e.g. Bleed+ or unnamed
    #     skills, which previously caused detector to miss real wasteful end_turn).
    #   Status / Curse — forced unplayable draws, never positive even if they
    #     slip through as legal.
    source = action_source(action)
    card_type = str(
        (action.get("card_type") if isinstance(action.get("card_type"), str) else None)
        or source.get("type")
        or ""
    ).strip().lower()
    if card_type in {"status", "curse"}:
        return False
    if card_type in {"attack", "skill", "power"}:
        return True
    roles = action_roles(action)
    if roles.intersection({"attack", "block", "draw", "debuff", "buff", "heal", "setup", "scaling", "resource"}):
        return True
    if is_facing_change_action is not None and is_facing_change_action(action):
        return True
    for key in ("damage", "total_damage", "block", "total_block", "draw", "weak", "vulnerable", "heal", "strength", "dexterity", "energy"):
        if action_metric(action, key) > 0.0:
            return True
        try:
            if float(source.get(key) or 0.0) > 0.0:
                return True
        except (TypeError, ValueError):
            pass
    text = " ".join(
        str(x or "")
        for x in (
            action.get("action_id"),
            action.get("kind"),
            action.get("title"),
            action.get("label"),
            source.get("id"),
            source.get("name"),
            source.get("title"),
            source.get("type"),
        )
    ).lower()
    return any(token in text for token in ("attack", "strike", "defend", "block", "skill", "power"))


def is_exhausting_action(action: Any) -> bool:
    """Return whether the action is expected to exhaust itself or cards."""

    if not isinstance(action, dict):
        return False
    roles = action_roles(action)
    if "exhaust" in roles:
        return True
    if (
        action_metric(action, "typed_exhaust_cards") > 0.0
        or action_metric(action, "typed_once_or_exhaust_self") > 0.0
    ):
        return True
    source = action_source(action)
    for key in ("exhaust", "exhaust_self", "will_exhaust"):
        if bool(action.get(key) or source.get(key)):
            return True
    return "exhaust" in action_text(action)


def is_ethereal_action(action: Any) -> bool:
    """Return whether the action/card has ethereal-style one-turn pressure."""

    if not isinstance(action, dict):
        return False
    roles = action_roles(action)
    if "ethereal" in roles:
        return True
    source = action_source(action)
    for key in ("ethereal", "is_ethereal"):
        if bool(action.get(key) or source.get(key)):
            return True
    text = action_text(action)
    # Preserve the legacy ``train.py`` mojibake fallback (``"??"``) during the
    # pure refactor.  The Chinese keyword check is additive and can be audited
    # later as a behavior change if needed.
    return "ethereal" in text or "??" in text or "虚无" in text


def is_retain_action(action: Any) -> bool:
    """Return whether the action/card retains itself or cards."""

    if not isinstance(action, dict):
        return False
    roles = action_roles(action)
    if "retain" in roles:
        return True
    if action_metric(action, "typed_retain_cards") > 0.0:
        return True
    source = action_source(action)
    for key in ("retain", "self_retain", "is_retained"):
        if bool(action.get(key) or source.get(key)):
            return True
    text = action_text(action)
    # Preserve the legacy ``train.py`` mojibake fallback (``"??"``) during the
    # pure refactor.  The Chinese keyword check is additive and can be audited
    # later as a behavior change if needed.
    return "retain" in text or "??" in text or "保留" in text


__all__ = [
    "action_immediate_impact",
    "action_metric",
    "action_roles",
    "action_source",
    "action_text",
    "is_ethereal_action",
    "is_exhausting_action",
    "is_positive_combat_action",
    "is_retain_action",
    "is_zero_cost_action",
    "semantic_family",
]
