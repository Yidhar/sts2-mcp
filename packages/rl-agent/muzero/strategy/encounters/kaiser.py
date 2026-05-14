"""Kaiser/back-attack facing helpers.

The bridge ``enemy.side`` / ``target.side`` fields describe faction
(``Player``/``Enemy``), not left/right body position.  Kaiser-style surrounded
position must be inferred from ``BACK_ATTACK_LEFT_POWER`` /
``BACK_ATTACK_RIGHT_POWER`` on the target enemy powers.  Keep this resolver out
of ``muzero.train`` so future facing rules can be tested and reused without
touching the trainer monolith.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any


SemanticFamilyFn = Callable[[Any], str]
ActionSourceFn = Callable[[Any], dict[str, Any]]
FacingChangeFn = Callable[[Any, Any | None], bool]


def normalize_side(value: Any) -> str:
    """Normalize only real left/right strings; ignore faction-like values."""

    text = str(value or "").strip().lower()
    if not text:
        return ""
    if text in {"l", "left", "side.left", "creatureside.left"} or text.endswith(".left"):
        return "left"
    if text in {"r", "right", "side.right", "creatureside.right"} or text.endswith(".right"):
        return "right"
    if "left" in text:
        return "left"
    if "right" in text:
        return "right"
    return ""


def _default_semantic_family(action: Any) -> str:
    if not isinstance(action, dict):
        return ""
    action_id = str(action.get("action_id") or action.get("kind") or "").lower()
    if "end_turn" in action_id or action_id == "end_turn":
        return "end_turn"
    if "use_potion" in action_id:
        return "use_potion"
    if "play_card" in action_id:
        return "play_card"
    return str(action.get("family") or action.get("semantic_family") or "").lower()


def _default_action_source(action: Any) -> dict[str, Any]:
    if not isinstance(action, dict):
        return {}
    for key in ("card", "potion", "source", "item"):
        value = action.get(key)
        if isinstance(value, dict):
            return value
    return {}


def is_facing_change_action(
    action: Any,
    *,
    semantic_family_fn: SemanticFamilyFn | None = None,
    action_source_fn: ActionSourceFn | None = None,
) -> bool:
    """Fallback detector for explicit facing actions.

    In live STS2 surrounded combat, facing usually changes implicitly by using
    a targeted card/potion on an enemy on that side.  Prefer
    :func:`action_changes_facing_toward_target` when raw combat state exists.
    """

    if not isinstance(action, dict):
        return False
    family_fn = semantic_family_fn or _default_semantic_family
    if family_fn(action) == "end_turn":
        return False
    source_fn = action_source_fn or _default_action_source
    source = source_fn(action)
    text = " ".join(
        str(x or "")
        for x in (
            action.get("action_id"),
            action.get("kind"),
            action.get("selection"),
            action.get("title"),
            action.get("label"),
            source.get("id"),
            source.get("name"),
            source.get("title"),
            source.get("description"),
        )
    ).lower()
    patterns = (
        "change_facing",
        "change-facing",
        "change facing",
        "set_facing",
        "set-facing",
        "set facing",
        "turn_around",
        "turn-around",
        "turn around",
        "turnaround",
        "rotate",
        "facing",
        "face_left",
        "face-right",
        "face right",
        "face left",
        "surrounded",
    )
    return any(pattern in text for pattern in patterns)


def combat_player_facing(raw_obs: Any | None) -> str:
    """Return the player's current left/right facing when available."""

    combat = raw_obs.get("combat") if isinstance(raw_obs, dict) and isinstance(raw_obs.get("combat"), dict) else {}
    return normalize_side(combat.get("facing"))


def action_target_combat_id(action: Any) -> int | None:
    """Best-effort target combat id resolver from legal action payloads."""

    if not isinstance(action, dict):
        return None
    candidates: list[Any] = [
        action.get("target_combat_id"),
        action.get("target_id"),
        action.get("enemy_combat_id"),
        action.get("enemy_id"),
    ]
    target = action.get("target")
    if isinstance(target, dict):
        candidates.extend([target.get("combat_id"), target.get("id"), target.get("target_combat_id")])
    target_mapping = action.get("target_mapping")
    if isinstance(target_mapping, dict):
        candidates.extend([target_mapping.get("combat_id"), target_mapping.get("id")])
    for value in candidates:
        try:
            if value is not None and str(value).strip() != "":
                return int(value)
        except (TypeError, ValueError):
            continue
    return None


def enemy_back_attack_position(enemy: Any) -> str:
    """Infer left/right body position from BackAttack powers."""

    if not isinstance(enemy, dict):
        return ""
    powers = enemy.get("powers")
    if not isinstance(powers, list):
        return ""
    for power in powers:
        if not isinstance(power, dict):
            continue
        text = " ".join(str(power.get(k) or "") for k in ("id", "title", "name")).upper()
        if "BACK_ATTACK_LEFT" in text:
            return "left"
        if "BACK_ATTACK_RIGHT" in text:
            return "right"
    return ""


def action_target_back_attack_position(action: Any, raw_obs: Any | None) -> str:
    """Return the target's back-attack body position, if resolvable."""

    if not isinstance(action, dict):
        return ""
    target = action.get("target")
    pos = enemy_back_attack_position(target) if isinstance(target, dict) else ""
    if pos:
        return pos
    target_id = action_target_combat_id(action)
    if target_id is None:
        return ""
    combat = raw_obs.get("combat") if isinstance(raw_obs, dict) and isinstance(raw_obs.get("combat"), dict) else {}
    for key in ("enemies", "monsters", "creatures"):
        entries = combat.get(key)
        if not isinstance(entries, list):
            continue
        for enemy in entries:
            if not isinstance(enemy, dict):
                continue
            try:
                enemy_id = int(enemy.get("combat_id", enemy.get("id")))
            except (TypeError, ValueError):
                continue
            if enemy_id == target_id:
                return enemy_back_attack_position(enemy)
    return ""


def action_target_side(action: Any, raw_obs: Any | None) -> str:
    """Return target body position (``left``/``right``), never faction side."""

    if not isinstance(action, dict):
        return ""
    return action_target_back_attack_position(action, raw_obs)


def is_targeted_enemy_action(
    action: Any,
    *,
    semantic_family_fn: SemanticFamilyFn | None = None,
) -> bool:
    """True for targeted card/potion actions that can turn the player."""

    if not isinstance(action, dict):
        return False
    family_fn = semantic_family_fn or _default_semantic_family
    if family_fn(action) not in {"play_card", "use_potion", "potion"}:
        return False
    target = action.get("target")
    if isinstance(target, dict) and any(target.get(k) is not None for k in ("combat_id", "id", "name", "side")):
        return True
    return action_target_combat_id(action) is not None or bool(action.get("target_name") or action.get("target_side"))


def action_changes_facing_toward_target(
    action: Any,
    raw_obs: Any | None,
    *,
    semantic_family_fn: SemanticFamilyFn | None = None,
) -> bool:
    """True when a targeted card/potion points to the other body side."""

    if not is_targeted_enemy_action(action, semantic_family_fn=semantic_family_fn):
        return False
    current = combat_player_facing(raw_obs)
    target_side = action_target_side(action, raw_obs)
    return bool(current and target_side and current != target_side)


def is_kaiser_facing_change_action(
    action: Any,
    raw_obs: Any | None,
    *,
    semantic_family_fn: SemanticFamilyFn | None = None,
    action_source_fn: ActionSourceFn | None = None,
) -> bool:
    """Combined implicit-target and explicit-facing Kaiser action detector."""

    return action_changes_facing_toward_target(
        action,
        raw_obs,
        semantic_family_fn=semantic_family_fn,
    ) or is_facing_change_action(
        action,
        semantic_family_fn=semantic_family_fn,
        action_source_fn=action_source_fn,
    )


def find_facing_candidates(
    legal_actions: list[Any] | None,
    mask_np: Any,
    raw_obs: Any | None,
    *,
    max_actions: int,
    semantic_family_fn: SemanticFamilyFn | None = None,
    action_source_fn: ActionSourceFn | None = None,
    facing_change_fn: FacingChangeFn | None = None,
) -> list[int]:
    """Return legal action indices that would change Kaiser facing."""

    if not isinstance(legal_actions, list):
        return []
    out: list[int] = []
    mask_size = int(getattr(mask_np, "shape", [0])[0]) if getattr(mask_np, "size", 0) else 0
    upper = min(len(legal_actions), mask_size, int(max_actions))
    for idx in range(upper):
        if mask_np[idx] <= 0:
            continue
        action = legal_actions[idx]
        if not isinstance(action, dict):
            continue
        if facing_change_fn is not None:
            changes_facing = bool(facing_change_fn(action, raw_obs))
        else:
            changes_facing = is_kaiser_facing_change_action(
                action,
                raw_obs,
                semantic_family_fn=semantic_family_fn,
                action_source_fn=action_source_fn,
            )
        if changes_facing:
            out.append(idx)
    return out


__all__ = [
    "action_changes_facing_toward_target",
    "action_target_back_attack_position",
    "action_target_combat_id",
    "action_target_side",
    "combat_player_facing",
    "enemy_back_attack_position",
    "find_facing_candidates",
    "is_facing_change_action",
    "is_kaiser_facing_change_action",
    "is_targeted_enemy_action",
    "normalize_side",
]
