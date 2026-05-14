"""Insatiable Sandpit / Frantic Escape strategy helpers."""

from __future__ import annotations

from collections.abc import Callable
from typing import Any


SemanticFamilyFn = Callable[[Any], str]


def _default_semantic_family(action: Any) -> str:
    if not isinstance(action, dict):
        return ""
    action_id = str(action.get("action_id") or action.get("kind") or "").lower()
    if "play_card" in action_id:
        return "play_card"
    if "end_turn" in action_id:
        return "end_turn"
    return str(action.get("family") or action.get("semantic_family") or "").lower()


def is_frantic_escape_action(
    action: Any,
    *,
    semantic_family_fn: SemanticFamilyFn | None = None,
) -> bool:
    """Return true for Frantic Escape using internal id first, text fallback second."""

    if not isinstance(action, dict):
        return False
    family_fn = semantic_family_fn or _default_semantic_family
    if family_fn(action) != "play_card":
        return False
    card = action.get("card") if isinstance(action.get("card"), dict) else {}
    cid = str(card.get("id") or card.get("ref") or "").strip().upper()
    if cid in {"CARD.FRANTIC_ESCAPE", "FRANTIC_ESCAPE"}:
        return True
    title = str(card.get("title") or card.get("name") or action.get("label") or "").strip().lower()
    return "frantic escape" in title or "狂乱逃离" in title or "狂亂逃離" in title


def find_frantic_escape_candidates(
    legal_actions: list[Any] | None,
    mask_np: Any,
    *,
    max_actions: int,
    semantic_family_fn: SemanticFamilyFn | None = None,
) -> list[int]:
    """Return legal indices of Frantic Escape actions."""

    if not isinstance(legal_actions, list):
        return []
    out: list[int] = []
    mask_size = int(getattr(mask_np, "shape", [0])[0]) if getattr(mask_np, "size", 0) else 0
    upper = min(len(legal_actions), mask_size, int(max_actions))
    for idx in range(upper):
        if mask_np[idx] <= 0:
            continue
        action = legal_actions[idx]
        if is_frantic_escape_action(action, semantic_family_fn=semantic_family_fn):
            out.append(idx)
    return out


def sandpit_countdown_from_context(boss_ctx: Any, raw_obs: Any | None) -> float | None:
    """Best-effort read of the active Insatiable Sandpit countdown."""

    if not isinstance(boss_ctx, dict):
        return None
    for key in ("insatiable_sandpit_countdown", "sandpit_countdown_min", "sandpit_countdown"):
        if key in boss_ctx:
            try:
                return float(boss_ctx.get(key))
            except (TypeError, ValueError):
                continue
    try:
        combat = raw_obs.get("combat") if isinstance(raw_obs, dict) else None
        enemies = combat.get("enemies") if isinstance(combat, dict) else None
        if not isinstance(enemies, list):
            return None
        best: float | None = None
        for enemy in enemies:
            if not isinstance(enemy, dict):
                continue
            powers = enemy.get("powers")
            if not isinstance(powers, list):
                continue
            for power in powers:
                if not isinstance(power, dict):
                    continue
                pid = str(power.get("id") or power.get("ref") or "").upper()
                if "SANDPIT" not in pid:
                    continue
                raw = power.get("countdown") or power.get("amount") or power.get("turns")
                try:
                    value = float(raw)
                except (TypeError, ValueError):
                    continue
                if best is None or value < best:
                    best = value
        return best
    except Exception:
        return None


__all__ = [
    "find_frantic_escape_candidates",
    "is_frantic_escape_action",
    "sandpit_countdown_from_context",
]
