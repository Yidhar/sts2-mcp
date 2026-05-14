"""HP-cost / self-lethal safety helpers (P0-1).

Per the muzero-mechanism-hardening task list, no action that would reduce
the player's HP to zero or below at resolve time should ever reach the
policy.  This module exposes:

* :func:`hp_cost_safety_view` — extract the bridge-side typed safety
  payload (``action["safety"]``) and back-fill missing fields from the
  best-effort fallbacks already attached to the action / card / potion
  payload (``effect_preview.hp_loss``, ``card.hp_loss``, semantic
  ``self_damage`` flags, etc.).
* :func:`is_self_lethal_action` — boolean guard used by ``action_masks``
  to hard-mask any action whose unblockable HP cost would kill the
  player at resolve time.
* :func:`is_low_hp_margin_action` — soft signal (action allowed but
  flagged for negative immediate impact in train).

All callers MUST treat the *unblockable* HP cost as the canonical figure;
``cardHpLoss`` and ``nonCardHpLoss`` are unblockable in STS2 source, and
block does not soak them.  Only when the bridge explicitly tags
``blockable_self_damage`` may current block be deducted from the cost.
"""

from __future__ import annotations

import math
import re
from functools import lru_cache
from typing import Any


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


def _player_hp(raw_obs: Any) -> float:
    if not isinstance(raw_obs, dict):
        return 0.0
    player = raw_obs.get("player")
    if not isinstance(player, dict):
        return 0.0
    for key in ("hp", "current_hp", "currentHealth"):
        if player.get(key) is not None:
            return _safe_float(player.get(key))
    return 0.0


def _player_block(raw_obs: Any) -> float:
    if not isinstance(raw_obs, dict):
        return 0.0
    player = raw_obs.get("player")
    if not isinstance(player, dict):
        return 0.0
    return _safe_float(player.get("block"))


def _action_card(action: Any) -> dict[str, Any]:
    if not isinstance(action, dict):
        return {}
    card = action.get("card")
    return card if isinstance(card, dict) else {}


def _action_potion(action: Any) -> dict[str, Any]:
    if not isinstance(action, dict):
        return {}
    potion = action.get("potion")
    return potion if isinstance(potion, dict) else {}


def _effect_preview(card: dict[str, Any]) -> dict[str, Any]:
    preview = card.get("effect_preview") if isinstance(card, dict) else None
    return preview if isinstance(preview, dict) else {}


def _typed_profile(card: dict[str, Any]) -> dict[str, Any]:
    profile = card.get("card_effect_profile") if isinstance(card, dict) else None
    return profile if isinstance(profile, dict) else {}


def _semantic_signals(card: dict[str, Any]) -> dict[str, Any]:
    sigs = card.get("semantic_signals") if isinstance(card, dict) else None
    return sigs if isinstance(sigs, dict) else {}


def _safe_str(value: Any, default: str = "") -> str:
    return str(value).strip().lower() if value is not None else default


def _action_card_id(action: Any, card: dict[str, Any]) -> str:
    """Best-effort static card id reader.

    Live bridge / sim / diagnostics payloads have not always agreed on the
    field name.  HP-cost safety is a *pre-step* hard guard, so it must be able
    to recover a static id before action diagnostics are merged back after the
    environment step.
    """
    if not isinstance(action, dict):
        return ""
    candidates = (
        card.get("id"),
        card.get("card_id"),
        action.get("card_id"),
        action.get("card_ref"),
        action.get("selected_card_id"),
    )
    semantic = action.get("semantic")
    if isinstance(semantic, dict):
        candidates = (*candidates, semantic.get("card_id"))
    for value in candidates:
        text = str(value or "").strip()
        if not text:
            continue
        if text.upper().startswith("CARD."):
            return text.upper()
        # Bare static ids from sim snapshots are common.
        if re.fullmatch(r"[A-Za-z0-9_]+", text):
            return f"CARD.{text.upper()}"
    return ""


def _normalize_title_key(value: Any) -> str:
    """Normalize localized/English card titles for exact title fallback.

    This deliberately stays conservative: remove runtime upgrade suffixes and
    common action-label wrappers, but do not fuzzy-match arbitrary substrings.
    """
    text = str(value or "").strip().lower()
    if not text:
        return ""
    text = text.replace("＋", "+")
    text = re.sub(r"\s+", " ", text)
    # Bridge canonical text often starts with "卡牌｜<title>｜...".
    if "｜" in text:
        parts = [part.strip() for part in text.split("｜") if part.strip()]
        if len(parts) >= 2 and parts[0] in {"卡牌", "card"}:
            text = parts[1]
    # Action labels may carry a light verb wrapper.
    for prefix in ("play ", "use ", "打出", "使用"):
        if text.startswith(prefix):
            text = text[len(prefix):].strip()
    # Drop target / index decorations often attached to labels.
    text = re.split(r"\s*(?:->|→|=>|:|：)\s*", text, maxsplit=1)[0].strip()
    return text.rstrip("+").strip()


def _action_card_title(action: Any, card: dict[str, Any]) -> str:
    if not isinstance(action, dict):
        return ""
    for value in (
        card.get("title"),
        card.get("name"),
        action.get("title"),
        action.get("name"),
        action.get("label"),
        action.get("canonical_text"),
    ):
        key = _normalize_title_key(value)
        if key:
            return key
    return ""


def _max_hp_loss_from_signals(signals: Any) -> float:
    if not isinstance(signals, dict):
        return 0.0
    hp_loss = 0.0
    for key in (
        "hpLoss",
        "hp_loss",
        "selfDamage",
        "self_damage",
        "hpCost",
        "hp_cost",
        "cardHpLoss",
        "nonCardHpLoss",
    ):
        v = _safe_float(signals.get(key))
        if v > hp_loss:
            hp_loss = v
    return hp_loss


@lru_cache(maxsize=1)
def _static_hp_loss_by_title() -> dict[str, float]:
    """Return exact title/id-tail aliases for cards with static hpLoss.

    The fallback is intentionally derived from the local content registry
    rather than hand-maintained one-offs.  It fixes title-only live actions
    (e.g. ``放血+`` with no compact card id) while keeping the match exact
    after normalization to avoid accidental false positives.
    """
    result: dict[str, float] = {}
    try:
        from content_registry import _load_registry  # type: ignore  # noqa: PLC0415
    except Exception:
        return result
    try:
        registry = _load_registry("cards")
    except Exception:
        return result
    if not isinstance(registry, dict):
        return result
    for card_id, metadata in registry.items():
        if not isinstance(metadata, dict):
            continue
        hp_loss = _max_hp_loss_from_signals(metadata.get("semantic_signals"))
        if hp_loss <= 0.0:
            continue
        aliases = {
            metadata.get("title"),
            metadata.get("name"),
            card_id,
            str(card_id).split(".", 1)[-1].replace("_", " "),
            str(card_id).split(".", 1)[-1].replace("_", ""),
        }
        for alias in aliases:
            key = _normalize_title_key(alias)
            if key:
                result[key] = max(result.get(key, 0.0), hp_loss)
    return result


def _static_hp_loss_from_metadata(action: Any, card: dict[str, Any]) -> float:
    """Static metadata HP-loss fallback for pre-step hard guards."""
    card_id = _action_card_id(action, card)
    if card_id:
        try:
            from content_registry import get_card_metadata  # noqa: PLC0415

            metadata = get_card_metadata(card_id)
        except Exception:
            metadata = None
        if isinstance(metadata, dict):
            hp_loss = _max_hp_loss_from_signals(metadata.get("semantic_signals"))
            if hp_loss > 0.0:
                return hp_loss

    title_key = _action_card_title(action, card)
    if title_key:
        return float(_static_hp_loss_by_title().get(title_key, 0.0) or 0.0)
    return 0.0


def hp_cost_safety_view(
    action: Any,
    raw_obs: Any | None = None,
) -> dict[str, Any]:
    """Return a stable HP-cost safety dict for ``action``.

    Priority order (per spec §P0-1):
      1. Bridge-supplied ``action["safety"]`` block (preferred — typed,
         confidence=runtime_internal).
      2. ``action["card"]["effect_preview"].hp_loss``.
      3. ``card.semantic_signals.self_damage`` /
         ``card.card_effect_profile.operations[op==hp_loss]`` /
         legacy ``card.hp_loss``.
      4. Empty / zero defaults — never raise.

    Returns a dict with the spec-mandated fields:
        hp_cost_kind, hp_loss_unblockable, self_damage_blockable,
        max_hp_loss, hp_before, hp_after_self_cost, hp_margin_after_self_cost,
        self_lethal_now, low_hp_margin_after_cost, source_confidence
    """
    out: dict[str, Any] = {
        "hp_cost_kind": "none",
        "hp_loss_unblockable": 0.0,
        "self_damage_blockable": 0.0,
        "max_hp_loss": 0.0,
        "hp_before": _player_hp(raw_obs),
        "hp_after_self_cost": _player_hp(raw_obs),
        "hp_margin_after_self_cost": _player_hp(raw_obs),
        "self_lethal_now": False,
        "low_hp_margin_after_cost": False,
        "source_confidence": "none",
    }
    if not isinstance(action, dict):
        return out

    safety = action.get("safety") if isinstance(action.get("safety"), dict) else None
    if isinstance(safety, dict) and safety:
        # Trust bridge-supplied typed payload; only re-derive booleans we own.
        for key in (
            "hp_cost_kind", "hp_cost", "hp_loss_unblockable", "self_damage_blockable",
            "max_hp_loss", "hp_before", "hp_after_self_cost",
            "hp_margin_after_self_cost", "self_lethal_now",
            "low_hp_margin_after_cost", "source_confidence",
        ):
            if key in safety:
                out[key] = safety[key]
        # Re-derive bools when bridge omitted them.
        out["hp_loss_unblockable"] = _safe_float(out.get("hp_loss_unblockable"))
        out["self_damage_blockable"] = _safe_float(out.get("self_damage_blockable"))
        out["max_hp_loss"] = _safe_float(out.get("max_hp_loss"))
        out["hp_before"] = _safe_float(out.get("hp_before"), _player_hp(raw_obs))
        unblockable = out["hp_loss_unblockable"]
        block = _player_block(raw_obs)
        blockable = max(0.0, out["self_damage_blockable"] - block)
        effective_self_cost = unblockable + blockable
        out["hp_after_self_cost"] = max(0.0, out["hp_before"] - unblockable - blockable)
        out["hp_margin_after_self_cost"] = out["hp_after_self_cost"]
        out["self_lethal_now"] = bool(
            out["hp_before"] > 0.0
            and out["hp_after_self_cost"] <= 0.0
            and effective_self_cost > 0.0
        )
        out["low_hp_margin_after_cost"] = bool(
            not out["self_lethal_now"]
            and effective_self_cost > 0.0
            and 0 < out["hp_after_self_cost"] <= max(3.0, 0.10 * max(out["hp_before"], 1.0))
        )
        if not str(out.get("source_confidence") or "").strip():
            out["source_confidence"] = "runtime_internal"
        return out

    # Bridge did not provide typed safety — fall back to existing fields.
    card = _action_card(action)
    potion = _action_potion(action)
    preview = _effect_preview(card)
    typed = _typed_profile(card)
    sem = _semantic_signals(card)

    hp_loss = 0.0
    for value in (
        preview.get("hp_loss"),
        preview.get("hpLoss"),
        action.get("hp_loss"),
        action.get("hpLoss"),
        action.get("hp_cost"),
        action.get("hpCost"),
        card.get("hp_loss"),
        card.get("hpLoss"),
        card.get("hp_cost"),
        card.get("hpCost"),
        sem.get("self_damage"),
        sem.get("selfDamage"),
        sem.get("hp_loss"),
        sem.get("hpLoss"),
        sem.get("hp_cost"),
        sem.get("hpCost"),
        potion.get("hp_loss"),
        potion.get("hpLoss"),
    ):
        v = _safe_float(value)
        if v > hp_loss:
            hp_loss = v
    typed_hp_loss = 0.0
    operations = typed.get("operations") if isinstance(typed, dict) else None
    if isinstance(operations, list):
        for op in operations:
            if not isinstance(op, dict):
                continue
            name = _safe_str(op.get("op"))
            if name in {"hploss", "hpcost", "cardhploss", "noncardhploss", "hp_loss", "self_hp_loss", "lose_hp", "self_damage"}:
                v = _safe_float(op.get("amount") or op.get("value") or op.get("hp"))
                if v > typed_hp_loss:
                    typed_hp_loss = v
    if typed_hp_loss > hp_loss:
        hp_loss = typed_hp_loss
    static_hp_loss = _static_hp_loss_from_metadata(action, card)
    if static_hp_loss > hp_loss:
        hp_loss = static_hp_loss
    max_hp_loss = _safe_float(preview.get("max_hp_loss") or sem.get("max_hp_loss") or card.get("max_hp_loss"))

    confidence = "fallback" if (hp_loss > 0 or max_hp_loss > 0) else "none"
    block = _player_block(raw_obs)
    hp_before = _player_hp(raw_obs)
    hp_after = max(0.0, hp_before - hp_loss)
    # Only declare self-lethal when we positively know HP — without raw_obs
    # the conservative call is "do not mask" so the agent stays unblocked
    # in test/dev paths that have no HP info.
    self_lethal = hp_before > 0.0 and hp_after <= 0.0 and hp_loss > 0.0
    low_margin = (
        hp_loss > 0.0
        and not self_lethal
        and 0 < hp_after <= max(3.0, 0.10 * max(hp_before, 1.0))
    )

    out.update({
        "hp_cost_kind": "unblockable_hp_loss" if hp_loss > 0 else ("max_hp_loss" if max_hp_loss > 0 else "none"),
        "hp_loss_unblockable": hp_loss,
        "self_damage_blockable": 0.0,
        "max_hp_loss": max_hp_loss,
        "hp_before": hp_before,
        "hp_after_self_cost": hp_after,
        "hp_margin_after_self_cost": hp_after,
        "self_lethal_now": bool(self_lethal),
        "low_hp_margin_after_cost": bool(low_margin),
        "source_confidence": confidence,
    })
    return out


def is_self_lethal_action(
    action: Any,
    raw_obs: Any | None = None,
) -> bool:
    """Return True iff the action would kill the player at resolve time."""
    return bool(hp_cost_safety_view(action, raw_obs).get("self_lethal_now"))


def is_low_hp_margin_action(
    action: Any,
    raw_obs: Any | None = None,
) -> bool:
    """Return True for non-lethal but very low margin HP-cost actions."""
    view = hp_cost_safety_view(action, raw_obs)
    return bool(view.get("low_hp_margin_after_cost") and not view.get("self_lethal_now"))
