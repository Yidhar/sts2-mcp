"""Pre-dispatch audit for selected combat EndTurn actions.

The legacy ``end_turn_contexts.jsonl`` dump is written after the post-search
selection path has already touched several cached observation helpers.  Recent
live reports showed contradictory frames: the selected EndTurn row said
``energy=0`` and ``only EndTurn legal`` while nearby guard records still saw
``energy=3`` plus legal play-card actions.  This module writes a second,
minimal audit immediately before ``env.step(EndTurn)`` using the exact
frontier that is about to be dispatched.

The audit is diagnostic only.  It never fabricates actions and never attempts
to play a card that the current legal-action surface does not expose.  When the
raw hand appears playable but the legal surface contains only EndTurn, it marks
``legal_generation_gap_suspect`` so we can separate policy空过 from bridge /
frontier staleness.
"""

from __future__ import annotations

import json
import math
import time
from pathlib import Path
from typing import Any

import numpy as np

from sts2_env.observation_v2 import MAX_ACTIONS

from muzero.combat_quality.progress_candidates import collect_safe_progress_candidates


_UNPLAYABLE_CARD_TYPES = {"status", "curse", "quest"}


def _safe_float(value: Any, default: float = 0.0) -> float:
    try:
        out = float(value)
    except (TypeError, ValueError):
        return float(default)
    return float(out) if math.isfinite(out) else float(default)


def _as_dict(value: Any) -> dict[str, Any]:
    return value if isinstance(value, dict) else {}


def _raw_combat(raw_obs: Any) -> dict[str, Any]:
    return _as_dict(raw_obs.get("combat")) if isinstance(raw_obs, dict) else {}


def _raw_player(raw_obs: Any) -> dict[str, Any]:
    if not isinstance(raw_obs, dict):
        return {}
    player = _as_dict(raw_obs.get("player"))
    if player:
        return player
    combat = _raw_combat(raw_obs)
    return _as_dict(combat.get("player"))


def _raw_run(raw_obs: Any) -> dict[str, Any]:
    if not isinstance(raw_obs, dict):
        return {}
    run = _as_dict(raw_obs.get("run"))
    if run:
        return run
    transition = _as_dict(raw_obs.get("transition_state"))
    return _as_dict(transition.get("run"))


def raw_hand_cards(raw_obs: Any) -> list[Any]:
    """Return the raw hand list from common bridge / sandbox observation shapes."""

    combat = _raw_combat(raw_obs)
    player = _raw_player(raw_obs)
    for container in (combat, player):
        if not isinstance(container, dict):
            continue
        for key in ("hand", "cards", "hand_cards", "handCards"):
            value = container.get(key)
            if isinstance(value, list):
                return value
    return []


def raw_max_energy(raw_obs: Any, default: float = 3.0) -> float:
    """Best-effort max-energy extraction from raw combat/player/run contexts."""

    combat = _raw_combat(raw_obs)
    player = _raw_player(raw_obs)
    run = _raw_run(raw_obs)
    for container in (combat, player, run):
        if not isinstance(container, dict):
            continue
        for key in (
            "max_energy",
            "maxEnergy",
            "energy_max",
            "energyMax",
            "base_energy",
            "baseEnergy",
        ):
            value = container.get(key)
            if value is None:
                continue
            number = _safe_float(value, 0.0)
            if number > 0.0:
                return float(number)
    return float(default)


def _card_type(card: Any) -> str:
    if not isinstance(card, dict):
        return ""
    return str(
        card.get("type")
        or card.get("card_type")
        or card.get("cardType")
        or card.get("kind")
        or ""
    ).strip().lower()


def _card_cost_for_turn(card: Any) -> float | None:
    """Return a high-confidence non-negative card cost, or ``None`` if unknown.

    Negative sentinel costs (X/unplayable/status) are not treated as affordable
    by the raw-hand audit unless the legal surface independently exposes them.
    """

    if not isinstance(card, dict):
        return None
    for key in (
        "cost_for_turn",
        "costForTurn",
        "resolved_energy_cost",
        "resolvedEnergyCost",
        "energy_cost",
        "energyCost",
        "canonical_energy_cost",
        "canonicalEnergyCost",
        "cost",
        "base_cost",
        "baseCost",
    ):
        if key not in card:
            continue
        value = card.get(key)
        if isinstance(value, str):
            text = value.strip().upper()
            if text == "X":
                return None
        number = _safe_float(value, default=-999.0)
        if number < 0.0:
            return None
        return float(number)
    return None


def card_is_ui_playable(card: Any, current_energy: float) -> bool:
    """High-confidence raw hand affordability predicate for diagnostics.

    This intentionally errs on the side of *not* claiming a card is playable.
    A false positive here would make the logs accuse the policy when the card
    may really be disabled by a modal, stun, status rule, or bridge surface.
    """

    if not isinstance(card, dict):
        return False
    playable_value = card.get("is_playable", card.get("playable"))
    if playable_value is False:
        return False
    ctype = _card_type(card)
    if ctype in _UNPLAYABLE_CARD_TYPES:
        return False
    cost = _card_cost_for_turn(card)
    if cost is None:
        return False
    return bool(cost <= float(current_energy) + 1e-6)


def count_ui_affordable_hand_cards(raw_obs: Any, current_energy: float) -> tuple[int, int]:
    """Return ``(affordable_count, hand_count)`` for raw hand cards."""

    cards = raw_hand_cards(raw_obs)
    return (
        int(sum(1 for card in cards if card_is_ui_playable(card, current_energy))),
        int(len(cards)),
    )


def _action_family(owner: Any, action: Any) -> str:
    try:
        family = owner._semantic_family(action)
        if family:
            return str(family)
    except Exception:
        pass
    if not isinstance(action, dict):
        return ""
    semantic = _as_dict(action.get("semantic"))
    return str(semantic.get("family") or action.get("kind") or "").strip().lower()


def _action_cost(owner: Any, action: Any) -> float:
    try:
        return float(owner._action_cost_value(action))
    except Exception:
        pass
    if not isinstance(action, dict):
        return 0.0
    card = _as_dict(action.get("card"))
    for container in (action, card):
        for key in (
            "card_cost",
            "cost_for_turn",
            "costForTurn",
            "energy_cost",
            "energyCost",
            "cost",
            "base_cost",
            "baseCost",
        ):
            if key not in container:
                continue
            value = container.get(key)
            if isinstance(value, str) and value.strip().upper() == "X":
                return 0.0
            number = _safe_float(value, 0.0)
            return max(float(number), 0.0)
    return 0.0


def _compact_card(card: Any) -> dict[str, Any]:
    if not isinstance(card, dict):
        return {"repr": str(card)}
    return {
        "id": card.get("id") or card.get("card_id") or card.get("cardId"),
        "title": card.get("title") or card.get("name") or card.get("label"),
        "instance_uuid": (
            card.get("instance_uuid")
            or card.get("instanceUuid")
            or card.get("uuid")
            or card.get("runtime_id")
            or card.get("runtimeId")
        ),
        "type": card.get("type") or card.get("card_type") or card.get("cardType"),
        "cost": card.get("cost"),
        "cost_for_turn": card.get("cost_for_turn") or card.get("costForTurn"),
        "is_playable": card.get("is_playable", card.get("playable")),
        "damage": card.get("damage"),
        "block": card.get("block"),
    }


def _compact_action(owner: Any, idx: int, action: Any, *, score: float | None = None) -> dict[str, Any]:
    if not isinstance(action, dict):
        return {"idx": int(idx), "repr": str(action), "score": score}
    card = _as_dict(action.get("card"))
    potion = _as_dict(action.get("potion"))
    target = _as_dict(action.get("target"))
    semantic = _as_dict(action.get("semantic"))
    family = _action_family(owner, action)
    return {
        "idx": int(idx),
        "family": family,
        "kind": action.get("kind"),
        "action_id": action.get("action_id"),
        "title": (
            action.get("title")
            or card.get("title")
            or card.get("name")
            or potion.get("title")
            or potion.get("name")
            or action.get("label")
        ),
        "card_id": card.get("id") or action.get("card_id"),
        "card_title": card.get("title") or card.get("name") or action.get("card_title"),
        "potion_id": potion.get("id") or action.get("potion_id"),
        "potion_title": potion.get("title") or potion.get("name"),
        "cost": _action_cost(owner, action),
        "target": target.get("name") or target.get("id") or action.get("target"),
        "damage": action.get("damage", card.get("damage")),
        "block": action.get("block", card.get("block")),
        "semantic": {
            "family": semantic.get("family") or semantic.get("action_kind"),
            "domain": semantic.get("domain"),
            "surface": semantic.get("surface"),
            "selection": semantic.get("selection"),
        }
        if semantic
        else {},
        "score": score,
    }


def _diagnostic_jsonl_path(owner: Any, filename: str) -> Path | None:
    path_fn = getattr(owner, "_diagnostic_jsonl_path", None)
    if callable(path_fn):
        try:
            path = path_fn(filename)
            if path is not None:
                return Path(path)
        except Exception:
            pass
    log_dir = getattr(owner, "log_dir", None)
    if not log_dir:
        return None
    name = filename if filename.endswith(".jsonl") else f"{filename}.jsonl"
    return Path(log_dir) / "diagnostics" / name


def build_end_turn_pre_dispatch_audit(
    owner: Any,
    *,
    encoded_obs: dict[str, Any] | None,
    raw_obs: dict[str, Any] | None,
    action_mask: Any,
    legal_actions: list[Any] | None,
    chosen_idx: int,
    search_policy: Any | None,
    search_stats: dict[str, Any] | None,
    pre_step_info: dict[str, Any] | None,
    encounter: str = "",
    tier: str = "",
) -> dict[str, Any]:
    """Build one pre-dispatch EndTurn audit payload."""

    mask_np = np.asarray(action_mask, dtype=np.float32).reshape(-1)
    actions = legal_actions if isinstance(legal_actions, list) else []
    legal_count = min(len(actions), MAX_ACTIONS, int(mask_np.shape[0]))
    combat = _raw_combat(raw_obs)
    player = _raw_player(raw_obs)
    run = _raw_run(raw_obs)
    try:
        energy = float(owner._combat_energy(encoded_obs, raw_obs))
    except Exception:
        energy = _safe_float(combat.get("energy"), 0.0)
    max_energy = raw_max_energy(raw_obs, default=max(3.0, energy))
    energy_ratio = float(energy / max(max_energy, 1.0)) if max_energy > 0.0 else 0.0
    # Keep this threshold aligned with ``FullEnergyEndTurnGuardMixin`` and
    # ``trainer_quality``: the audit should identify the exact full-energy
    # dispatch class that the hard guard is responsible for, not looser
    # 2/3-energy leftovers.
    full_energy_like = bool(energy_ratio >= 0.95 or float(energy) >= float(max_energy) - 1e-6)

    mask_legal_count = int(sum(1 for idx in range(min(MAX_ACTIONS, mask_np.shape[0])) if mask_np[idx] > 0))
    legal_non_end_turn_count = 0
    legal_play_card_action_count = 0
    affordable_play_card_action_count = 0
    selected_family = ""
    for idx in range(legal_count):
        if mask_np[idx] <= 0:
            continue
        action = actions[idx]
        family = _action_family(owner, action)
        if idx == int(chosen_idx):
            selected_family = family
        if family != "end_turn":
            legal_non_end_turn_count += 1
        if family == "play_card":
            legal_play_card_action_count += 1
            if _action_cost(owner, action) <= energy + 1e-6:
                affordable_play_card_action_count += 1

    ui_affordable_count, hand_count = count_ui_affordable_hand_cards(raw_obs, energy)

    safe_candidates = []
    safe_rejection_counts: dict[str, int] = {}
    if (
        isinstance(raw_obs, dict)
        and 0 <= int(chosen_idx) < legal_count
        and actions
        and selected_family == "end_turn"
    ):
        try:
            safe_candidates, safe_rejection_counts = collect_safe_progress_candidates(
                owner,
                selected_idx=int(chosen_idx),
                legal_count=int(legal_count),
                legal_actions=actions,
                mask_np=mask_np,
                raw_obs=raw_obs,
                current_energy=float(energy),
                use_mask=True,
                include_debug=True,
            )
        except Exception:
            safe_candidates = []
            safe_rejection_counts = {}

    safe_progress_candidate_count = int(len(safe_candidates))
    legal_generation_gap_suspect = bool(
        ui_affordable_count > 0 and legal_play_card_action_count <= 0
    )
    raw_hand_legal_surface_mismatch = bool(
        hand_count > 0 and ui_affordable_count > 0 and affordable_play_card_action_count <= 0
    )
    full_energy_skip_suspect = bool(
        selected_family == "end_turn"
        and full_energy_like
        and (
            safe_progress_candidate_count > 0
            or affordable_play_card_action_count > 0
            or ui_affordable_count > 0
        )
    )
    try:
        incoming, block, _hp = owner._incoming_damage_pressure(raw_obs)
        incoming = float(incoming)
        block = float(block)
    except Exception:
        incoming = _safe_float(combat.get("incoming_damage") or combat.get("incoming"), 0.0)
        block = _safe_float(player.get("block") or combat.get("block"), 0.0)
    pressure_skip_suspect = bool(full_energy_skip_suspect and incoming > block + 0.5)
    singleton_frontier_suspect = bool(
        selected_family == "end_turn"
        and mask_legal_count <= 1
        and full_energy_like
        and ui_affordable_count > 0
    )

    policy_np = (
        np.asarray(search_policy, dtype=np.float32).reshape(-1)
        if search_policy is not None
        else np.zeros(MAX_ACTIONS, dtype=np.float32)
    )
    ranked: list[tuple[int, float]] = []
    for idx in range(min(legal_count, policy_np.shape[0])):
        if mask_np[idx] <= 0:
            continue
        ranked.append((int(idx), float(policy_np[idx])))
    ranked.sort(key=lambda item: item[1], reverse=True)

    pre_info = pre_step_info if isinstance(pre_step_info, dict) else {}
    stats = search_stats if isinstance(search_stats, dict) else {}
    floor_value = run.get("floor", run.get("total_floor", raw_obs.get("floor") if isinstance(raw_obs, dict) else None))
    act_id_value = run.get("act_id", run.get("act", raw_obs.get("act_id") if isinstance(raw_obs, dict) else None))

    return {
        "schema": "end_turn_pre_dispatch_audit_v1",
        "time": time.time(),
        "global_step": int(getattr(owner, "total_steps", 0)),
        "episode_id": int(getattr(owner, "episode_count", 0)),
        "selected_action_idx": int(chosen_idx),
        "selected_family": selected_family or "end_turn",
        "encounter_id": str(encounter or ""),
        "tier": str(tier or ""),
        "progress": {
            "floor": floor_value,
            "act_id": act_id_value,
            "room_type": run.get("room_type") or run.get("room_type_name"),
            "room_model": run.get("room_model") or run.get("room_id"),
            "encounter_id_raw": run.get("encounter_id") or run.get("encounter"),
            "screen": raw_obs.get("screen") if isinstance(raw_obs, dict) else None,
            "phase": raw_obs.get("phase") if isinstance(raw_obs, dict) else pre_info.get("phase"),
            "state_version": raw_obs.get("state_version") if isinstance(raw_obs, dict) else None,
        },
        "player": {
            "hp": player.get("hp", player.get("current_hp")),
            "max_hp": player.get("max_hp"),
            "block": block,
            "energy": float(energy),
            "max_energy": float(max_energy),
            "energy_ratio": float(energy_ratio),
        },
        "combat": {
            "incoming_damage": float(incoming),
            "turn": combat.get("round") or combat.get("turn"),
            "hand_count": int(hand_count),
        },
        "counts": {
            "mask_legal_count": int(mask_legal_count),
            "raw_legal_action_count": int(pre_info.get("raw_legal_action_count", 0) or 0),
            "legal_non_end_turn_count": int(legal_non_end_turn_count),
            "legal_play_card_action_count": int(legal_play_card_action_count),
            "affordable_play_card_action_count": int(affordable_play_card_action_count),
            "ui_affordable_hand_card_count": int(ui_affordable_count),
            "ui_playable_hand_card_count": int(ui_affordable_count),
            "raw_hand_card_count": int(hand_count),
            "safe_progress_candidate_count": int(safe_progress_candidate_count),
            "safe_progress_rejections": dict(safe_rejection_counts),
        },
        "flags": {
            "full_energy_like": bool(full_energy_like),
            "full_energy_skip_suspect": bool(full_energy_skip_suspect),
            "legal_generation_gap_suspect": bool(legal_generation_gap_suspect),
            "raw_hand_legal_surface_mismatch": bool(raw_hand_legal_surface_mismatch),
            "full_energy_skip_with_playable_hand": bool(full_energy_like and ui_affordable_count > 0),
            "pressure_skip_suspect": bool(pressure_skip_suspect),
            "singleton_frontier_suspect": bool(singleton_frontier_suspect),
            "policy_retargeted": bool(float(stats.get("post_search_hard_guard_policy_retargeted", 0.0) or 0.0) > 0.5),
        },
        "raw_hand_cards": [_compact_card(card) for card in raw_hand_cards(raw_obs)[:12]],
        "legal_actions_compact": [
            _compact_action(
                owner,
                idx,
                actions[idx],
                score=float(policy_np[idx]) if idx < policy_np.shape[0] else None,
            )
            for idx in range(min(len(actions), MAX_ACTIONS, 40))
        ],
        "safe_progress_candidates": [
            {
                "index": int(candidate.index),
                "title": candidate.title,
                "damage": float(candidate.damage),
                "impact": float(candidate.impact),
                "cost": float(candidate.cost),
                "lethal": bool(candidate.lethal),
                "score": [float(x) for x in candidate.score],
            }
            for candidate in safe_candidates[:8]
        ],
        "top_policy_actions": [
            {
                "rank": rank,
                "action_idx": int(idx),
                "is_chosen": int(idx) == int(chosen_idx),
                **_compact_action(owner, idx, actions[idx], score=score),
            }
            for rank, (idx, score) in enumerate(ranked[:8], start=1)
        ],
    }


def dump_end_turn_pre_dispatch_audit(
    owner: Any,
    *,
    encoded_obs: dict[str, Any] | None,
    raw_obs: dict[str, Any] | None,
    action_mask: Any,
    legal_actions: list[Any] | None,
    chosen_idx: int,
    search_policy: Any | None,
    search_stats: dict[str, Any] | None,
    pre_step_info: dict[str, Any] | None,
    encounter: str = "",
    tier: str = "",
) -> dict[str, Any] | None:
    """Append one pre-dispatch audit row and return the payload."""

    if getattr(owner, "_end_turn_pre_dispatch_audit_disabled", False):
        return None
    cap = int(getattr(owner, "_end_turn_pre_dispatch_audit_max", 100000) or 0)
    count = int(getattr(owner, "_end_turn_pre_dispatch_audit_count", 0) or 0)
    if cap > 0 and count >= cap:
        return None
    try:
        payload = build_end_turn_pre_dispatch_audit(
            owner,
            encoded_obs=encoded_obs,
            raw_obs=raw_obs,
            action_mask=action_mask,
            legal_actions=legal_actions,
            chosen_idx=int(chosen_idx),
            search_policy=search_policy,
            search_stats=search_stats,
            pre_step_info=pre_step_info,
            encounter=encounter,
            tier=tier,
        )
        path = _diagnostic_jsonl_path(owner, "end_turn_pre_dispatch_audit.jsonl")
        if path is None:
            return payload
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("a", encoding="utf-8") as fh:
            fh.write(json.dumps(payload, ensure_ascii=False, default=str) + "\n")
        setattr(owner, "_end_turn_pre_dispatch_audit_count", count + 1)
        return payload
    except Exception:
        return None


__all__ = [
    "build_end_turn_pre_dispatch_audit",
    "card_is_ui_playable",
    "count_ui_affordable_hand_cards",
    "dump_end_turn_pre_dispatch_audit",
    "raw_hand_cards",
    "raw_max_energy",
]
