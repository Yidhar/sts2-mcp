"""Campfire smith -> deck-upgrade diagnostics.

Act1 failures recently showed a suspicious pattern: runs reached campfires and
grew the deck, but death-deck summaries still reported ``upgraded_count == 0``.
The rest-site telemetry can tell whether the policy picked SMITH, but it cannot
prove the follow-up upgrade surface appeared or that choosing an upgrade target
actually increased the deck's upgraded-card count.

This module keeps that chain observable without growing ``self_play.py``:

    rest_site: SMITH
        -> post-step deck_upgrade options visible?
        -> deck_upgrade target selected?
        -> upgraded_count increased?

The helpers are deliberately schema-tolerant.  Live bridge payloads, compact
signatures, and sim translated actions all preserve slightly different fields.
"""

from __future__ import annotations

from dataclasses import dataclass
import json
import math
import time
from typing import Any

from muzero.diagnostics.deck_build_metrics import (
    compute_deck_quality_summary,
    extract_deck_cards_from_obs_like,
)


DECK_UPGRADE_TB_KEYS: tuple[tuple[str, str], ...] = (
    ("smith_selected", "deck_upgrade_smith_selected_count"),
    ("smith_to_upgrade_seen_rate", "deck_upgrade_smith_to_upgrade_seen_rate"),
    ("smith_no_upgrade_surface_rate", "deck_upgrade_smith_no_upgrade_surface_rate"),
    ("seen", "deck_upgrade_seen_count"),
    ("selected_rate", "deck_upgrade_selected_rate"),
    ("applied_rate", "deck_upgrade_applied_rate"),
    ("target_count_mean", "deck_upgrade_target_count_mean"),
    ("applied_delta_mean", "deck_upgrade_applied_delta_mean"),
    ("context_present_rate", "deck_upgrade_context_present_rate"),
    ("upgraded_before_mean", "deck_upgrade_upgraded_before_mean"),
    ("upgraded_after_mean", "deck_upgrade_upgraded_after_mean"),
)


def _safe_float(value: Any, default: float = 0.0) -> float:
    try:
        out = float(value)
    except (TypeError, ValueError):
        return float(default)
    return out if math.isfinite(out) else float(default)


def _safe_bool(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        return bool(value)
    if isinstance(value, str):
        return value.strip().lower() in {"1", "true", "yes", "y", "on"}
    return bool(value)


def _lower(value: Any) -> str:
    return str(value or "").strip().lower()


def _action_containers(action: Any) -> list[dict[str, Any]]:
    if not isinstance(action, dict):
        return []
    containers = [action]
    payload = action.get("payload")
    if isinstance(payload, dict):
        containers.append(payload)
    return containers


def is_deck_upgrade_terminal_action(action: Any) -> bool:
    """Return true for close/done/cancel actions on the deck-upgrade surface.

    Live bridge payloads have used ids such as ``deck_upgrade:close`` for the
    terminal button shown after SMITH.  Treating that button as a concrete
    upgrade target made both telemetry and hard guards think that "close" was a
    valid card upgrade choice.  Keep this helper schema-tolerant and explicit:
    only terminal/proceed tokens are excluded, while ordinary
    ``deck_upgrade:<index>`` / ``upgrade_card:<index>`` actions remain targets.
    """

    terminal_tokens = {
        "close",
        "cancel",
        "done",
        "proceed",
        "continue",
        "leave",
        "skip",
        "confirm",
    }
    for container in _action_containers(action):
        kind = _lower(container.get("kind") or container.get("action_type"))
        if kind in terminal_tokens:
            return True
        action_id = _lower(container.get("action_id"))
        if not action_id:
            continue
        # Exact ids occasionally appear in compact/sim payloads.
        if action_id in {
            "deck_upgrade:close",
            "deck_upgrade:cancel",
            "deck_upgrade:done",
            "deck_upgrade:proceed",
            "deck_upgrade:continue",
            "deck_upgrade:leave",
            "deck_upgrade:skip",
            "deck_upgrade:confirm",
            "upgrade_card:close",
            "upgrade_card:cancel",
            "upgrade_card:done",
            "upgrade_card:proceed",
            "upgrade_card:continue",
            "upgrade_card:leave",
            "upgrade_card:skip",
            "upgrade_card:confirm",
        }:
            return True
        # Also catch wrapped/sim ids whose final component is the terminal
        # token, e.g. ``sim:upgrade_card:close``.
        parts = [part for part in action_id.replace("|", ":").replace("/", ":").split(":") if part]
        if parts and parts[-1] in terminal_tokens and any(
            part in {"deck_upgrade", "upgrade_card", "upgrade"} for part in parts[:-1]
        ):
            return True
    return False


def _deck_quality(obs_like: Any) -> dict[str, float]:
    cards = extract_deck_cards_from_obs_like(obs_like)
    if not cards:
        return {}
    quality = compute_deck_quality_summary(cards)
    return quality if isinstance(quality, dict) else {}


def _upgraded_count(obs_like: Any) -> tuple[float, bool]:
    quality = _deck_quality(obs_like)
    if not quality:
        return 0.0, False
    return _safe_float(quality.get("upgraded_count"), 0.0), True


def is_deck_upgrade_action(action: Any) -> bool:
    """Return true for a concrete deck-upgrade target action."""

    if is_deck_upgrade_terminal_action(action):
        return False
    for container in _action_containers(action):
        kind = _lower(container.get("kind") or container.get("action_type"))
        if kind == "deck_upgrade":
            return True
        action_id = _lower(container.get("action_id"))
        if (
            action_id.startswith("deck_upgrade:")
            or action_id.startswith("upgrade_card:")
            or action_id.startswith("sim:upgrade_card")
        ):
            return True
        semantic = container.get("semantic") if isinstance(container.get("semantic"), dict) else {}
        family = _lower(semantic.get("family"))
        domain = _lower(semantic.get("domain"))
        if domain == "build" and family in {"deck_upgrade", "smith"}:
            return True
    return False


def _card_payload(action: dict[str, Any] | None) -> dict[str, Any]:
    if not isinstance(action, dict):
        return {}
    card = action.get("card")
    if isinstance(card, dict):
        return card
    payload = action.get("payload") if isinstance(action.get("payload"), dict) else {}
    card = payload.get("card")
    if isinstance(card, dict):
        return card
    return {}


def _card_title(action: dict[str, Any] | None) -> str:
    if not isinstance(action, dict):
        return ""
    card = _card_payload(action)
    return str(
        card.get("title")
        or card.get("name")
        or action.get("card_title")
        or action.get("title")
        or action.get("label")
        or action.get("name")
        or ""
    )


def _card_id(action: dict[str, Any] | None) -> str:
    if not isinstance(action, dict):
        return ""
    card = _card_payload(action)
    return str(card.get("id") or action.get("card_id") or "")


def _compact_deck_upgrade_action(
    action: dict[str, Any] | None,
    *,
    index: int | None = None,
    prob: float | None = None,
) -> dict[str, Any]:
    if not isinstance(action, dict):
        return {}
    out: dict[str, Any] = {
        "index": int(index) if index is not None else None,
        "prob": float(prob) if prob is not None else None,
        "kind": action.get("kind"),
        "action_id": action.get("action_id"),
        "title": _card_title(action),
        "card_id": _card_id(action),
        "card_title": _card_title(action),
        "is_deck_upgrade": is_deck_upgrade_action(action),
    }
    return {key: value for key, value in out.items() if value not in (None, "")}


def deck_upgrade_available_summary(legal_actions: list[Any]) -> dict[str, Any]:
    actions = legal_actions if isinstance(legal_actions, list) else []
    upgrade_actions = [action for action in actions if is_deck_upgrade_action(action)]
    return {
        "deck_upgrade_available": bool(upgrade_actions),
        "deck_upgrade_action_count": float(len(upgrade_actions)),
    }


def _top_policy_upgrade_actions(legal_actions: list[Any], search_policy: Any, *, max_topk: int) -> list[dict[str, Any]]:
    if max_topk <= 0 or not isinstance(legal_actions, list):
        return []
    try:
        flat = list(search_policy)
        probs = [_safe_float(value, 0.0) for value in flat[: len(legal_actions)]]
    except Exception:
        probs = []
    if len(probs) < len(legal_actions):
        probs.extend([0.0] * (len(legal_actions) - len(probs)))
    indexed = [
        (idx, probs[idx], action)
        for idx, action in enumerate(legal_actions)
        if is_deck_upgrade_action(action)
    ]
    indexed.sort(key=lambda item: item[1], reverse=True)
    return [
        _compact_deck_upgrade_action(action, index=idx, prob=prob)
        for idx, prob, action in indexed[: max(1, int(max_topk))]
    ]


def build_deck_upgrade_choice_payload(
    *,
    decision_domain: str,
    phase: str,
    legal_actions: list[Any],
    chosen_action: dict[str, Any] | None,
    chosen_signature: dict[str, Any] | None,
    selected_index: int,
    progress: dict[str, Any] | None = None,
    raw_obs: Any = None,
    search_policy: Any = None,
    max_topk: int = 8,
) -> dict[str, Any] | None:
    """Build one deck-upgrade choice record before ``env.step``."""

    legal_actions = legal_actions if isinstance(legal_actions, list) else []
    upgrade_context = (
        "deck_upgrade" in _lower(phase)
        or "upgrade" in _lower(phase)
        or is_deck_upgrade_action(chosen_action)
        or is_deck_upgrade_action(chosen_signature)
        or any(is_deck_upgrade_action(action) for action in legal_actions)
    )
    if not upgrade_context:
        return None

    selected = chosen_action if isinstance(chosen_action, dict) else chosen_signature
    selected_is_upgrade = is_deck_upgrade_action(selected)
    upgraded_before, context_present = _upgraded_count(raw_obs)
    summary = deck_upgrade_available_summary(legal_actions)
    payload: dict[str, Any] = {
        "decision_domain": str(decision_domain or ""),
        "phase": str(phase or ""),
        "selected_index": int(selected_index),
        "selected_action_kind": "deck_upgrade" if selected_is_upgrade else "non_deck_upgrade",
        "selected_action": _compact_deck_upgrade_action(
            selected if isinstance(selected, dict) else None,
            index=selected_index,
        ),
        "top_policy_deck_upgrade_actions": _top_policy_upgrade_actions(
            legal_actions,
            search_policy,
            max_topk=max_topk,
        ),
        "legal_deck_upgrade_actions": [
            _compact_deck_upgrade_action(action, index=idx)
            for idx, action in enumerate(legal_actions)
            if is_deck_upgrade_action(action)
        ][: max(1, int(max_topk))],
        "upgraded_count_before": upgraded_before,
        "deck_context_present": bool(context_present),
        **summary,
    }
    if isinstance(progress, dict):
        for key in ("floor", "act_id", "room_type", "room_type_lower", "encounter_id"):
            if key in progress:
                payload[key] = progress.get(key)
    return payload


def complete_deck_upgrade_choice_payload(
    payload: dict[str, Any] | None,
    *,
    post_obs: Any,
    post_info: dict[str, Any] | None = None,
) -> dict[str, Any] | None:
    """Attach post-step upgraded-count evidence to a pre-step payload."""

    if not isinstance(payload, dict):
        return None
    upgraded_after, post_context_present = _upgraded_count(post_obs)
    upgraded_before = _safe_float(payload.get("upgraded_count_before"), 0.0)
    delta = upgraded_after - upgraded_before if post_context_present else 0.0
    payload.update(
        {
            "post_phase": str((post_info or {}).get("phase") or ""),
            "post_decision_domain": str((post_info or {}).get("decision_domain") or ""),
            "upgraded_count_after": upgraded_after,
            "post_deck_context_present": bool(post_context_present),
            "upgraded_delta": float(delta),
            "upgrade_applied": bool(
                payload.get("selected_action_kind") == "deck_upgrade"
                and post_context_present
                and delta > 0.0
            ),
        }
    )
    return payload


def build_smith_upgrade_transition_payload(
    *,
    rest_site_payload: dict[str, Any] | None,
    post_legal_actions: list[Any],
    pre_obs: Any = None,
    post_obs: Any = None,
    post_info: dict[str, Any] | None = None,
) -> dict[str, Any] | None:
    """Record whether a selected SMITH option exposed an upgrade surface."""

    if not isinstance(rest_site_payload, dict):
        return None
    if _lower(rest_site_payload.get("selected_action_kind")) != "smith":
        return None
    post_legal_actions = post_legal_actions if isinstance(post_legal_actions, list) else []
    upgraded_before, pre_context_present = _upgraded_count(pre_obs)
    upgraded_after, post_context_present = _upgraded_count(post_obs)
    summary = deck_upgrade_available_summary(post_legal_actions)
    payload: dict[str, Any] = {
        "decision_domain": str(rest_site_payload.get("decision_domain") or ""),
        "phase": str(rest_site_payload.get("phase") or ""),
        "floor": rest_site_payload.get("floor"),
        "act_id": rest_site_payload.get("act_id"),
        "room_type": rest_site_payload.get("room_type"),
        "selected_rest_action": rest_site_payload.get("selected_action"),
        "post_phase": str((post_info or {}).get("phase") or ""),
        "post_decision_domain": str((post_info or {}).get("decision_domain") or ""),
        "upgraded_count_before": upgraded_before,
        "upgraded_count_after": upgraded_after,
        "deck_context_present": bool(pre_context_present or post_context_present),
        "smith_to_upgrade_seen": bool(summary.get("deck_upgrade_available")),
        **summary,
    }
    return {key: value for key, value in payload.items() if value is not None}


@dataclass
class DeckUpgradeEpisodeTracker:
    """Track smith/upgrade chains across one episode."""

    smith_selected: int = 0
    smith_to_upgrade_seen: int = 0
    smith_no_upgrade_surface: int = 0
    seen: int = 0
    selected: int = 0
    applied: int = 0
    target_count_sum: float = 0.0
    target_count_count: int = 0
    applied_delta_sum: float = 0.0
    applied_delta_count: int = 0
    context_present: int = 0
    upgraded_before_sum: float = 0.0
    upgraded_before_count: int = 0
    upgraded_after_sum: float = 0.0
    upgraded_after_count: int = 0

    def update_smith_transition(self, payload: dict[str, Any] | None) -> None:
        if not isinstance(payload, dict):
            return
        self.smith_selected += 1
        if _safe_bool(payload.get("smith_to_upgrade_seen")):
            self.smith_to_upgrade_seen += 1
        else:
            self.smith_no_upgrade_surface += 1

    def update_deck_upgrade_choice(self, payload: dict[str, Any] | None) -> None:
        if not isinstance(payload, dict):
            return
        self.seen += 1
        if _lower(payload.get("selected_action_kind")) == "deck_upgrade":
            self.selected += 1
        if _safe_bool(payload.get("upgrade_applied")):
            self.applied += 1
        self.target_count_sum += _safe_float(payload.get("deck_upgrade_action_count"), 0.0)
        self.target_count_count += 1
        self.applied_delta_sum += max(0.0, _safe_float(payload.get("upgraded_delta"), 0.0))
        self.applied_delta_count += 1
        if _safe_bool(payload.get("deck_context_present")) or _safe_bool(payload.get("post_deck_context_present")):
            self.context_present += 1
        self.upgraded_before_sum += _safe_float(payload.get("upgraded_count_before"), 0.0)
        self.upgraded_before_count += 1
        self.upgraded_after_sum += _safe_float(payload.get("upgraded_count_after"), 0.0)
        self.upgraded_after_count += 1

    def as_metadata(self) -> dict[str, float]:
        smith_safe = max(int(self.smith_selected), 1)
        seen_safe = max(int(self.seen), 1)
        selected_safe = max(int(self.selected), 1)
        return {
            "deck_upgrade_smith_selected_count": float(self.smith_selected),
            "deck_upgrade_smith_to_upgrade_seen_count": float(self.smith_to_upgrade_seen),
            "deck_upgrade_smith_no_upgrade_surface_count": float(self.smith_no_upgrade_surface),
            "deck_upgrade_smith_to_upgrade_seen_rate": (
                float(self.smith_to_upgrade_seen) / float(smith_safe)
            ),
            "deck_upgrade_smith_no_upgrade_surface_rate": (
                float(self.smith_no_upgrade_surface) / float(smith_safe)
            ),
            "deck_upgrade_seen_count": float(self.seen),
            "deck_upgrade_selected_count": float(self.selected),
            "deck_upgrade_applied_count": float(self.applied),
            "deck_upgrade_selected_rate": float(self.selected) / float(seen_safe),
            "deck_upgrade_applied_rate": float(self.applied) / float(selected_safe),
            "deck_upgrade_target_count_mean": self.target_count_sum / float(max(self.target_count_count, 1)),
            "deck_upgrade_applied_delta_mean": self.applied_delta_sum / float(max(self.applied_delta_count, 1)),
            "deck_upgrade_context_present_rate": float(self.context_present) / float(seen_safe),
            "deck_upgrade_upgraded_before_mean": self.upgraded_before_sum / float(max(self.upgraded_before_count, 1)),
            "deck_upgrade_upgraded_after_mean": self.upgraded_after_sum / float(max(self.upgraded_after_count, 1)),
        }


def _dump_jsonl(trainer: Any, payload: dict[str, Any] | None, *, filename: str, counter_attr: str) -> None:
    if not isinstance(payload, dict):
        return
    if getattr(trainer, f"_{counter_attr}_disabled", False):
        return
    cap = int(getattr(trainer, f"_{counter_attr}_max", 100000))
    count = int(getattr(trainer, f"_{counter_attr}_count", 0) or 0)
    if cap > 0 and count >= cap:
        return
    path_getter = getattr(trainer, "_diagnostic_jsonl_path", None)
    if not callable(path_getter):
        return
    try:
        record = {
            "time": time.time(),
            "global_step": int(getattr(trainer, "total_steps", 0)),
            "episode_id": int(getattr(trainer, "episode_count", 0)),
            **payload,
        }
        path = path_getter(filename)
        if path is None:
            return
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("a", encoding="utf-8") as fh:
            fh.write(json.dumps(record, ensure_ascii=False, default=str) + "\n")
        setattr(trainer, f"_{counter_attr}_count", count + 1)
    except Exception:
        return


def dump_deck_upgrade_choice_diagnostic(trainer: Any, payload: dict[str, Any] | None) -> None:
    _dump_jsonl(
        trainer,
        payload,
        filename="deck_upgrade_choices.jsonl",
        counter_attr="deck_upgrade_choice_dump",
    )


def dump_smith_upgrade_transition_diagnostic(trainer: Any, payload: dict[str, Any] | None) -> None:
    _dump_jsonl(
        trainer,
        payload,
        filename="smith_upgrade_transitions.jsonl",
        counter_attr="smith_upgrade_transition_dump",
    )
