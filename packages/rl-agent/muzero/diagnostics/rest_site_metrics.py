"""Campfire/rest-site decision diagnostics.

The env already exposes coarse episode counters such as
``env/rest_heal_chosen`` and ``env/rest_smith_chosen``.  Those counters are
good redlines, but not enough to answer the Act1 blocker question:

* Did the policy truly see SMITH and choose HEAL?
* Was SMITH masked by the low-HP exposure filter before policy selection?
* Are terminal/proceed actions being miscounted as campfire choices?
* What HP/floor/deck context existed at each campfire?

This module mirrors the shop diagnostics pattern: keep compact per-decision
JSONL records plus episode-level TensorBoard-friendly rates without growing
``self_play.py`` again.
"""

from __future__ import annotations

from dataclasses import dataclass
import json
import math
import time
from typing import Any

from sts2_env.reward_constants import REST_SITE_SKIP_HEAL_HP_THRESHOLD


REST_SITE_TB_KEYS: tuple[tuple[str, str], ...] = (
    ("seen", "rest_site_seen_count"),
    ("heal_rate", "rest_site_heal_rate"),
    ("smith_rate", "rest_site_smith_rate"),
    ("non_heal_rate", "rest_site_non_heal_rate"),
    ("other_rate", "rest_site_other_rate"),
    ("low_hp_rate", "rest_site_low_hp_rate"),
    ("high_hp_rate", "rest_site_high_hp_rate"),
    ("heal_available_rate", "rest_site_heal_available_rate"),
    ("smith_available_rate", "rest_site_smith_available_rate"),
    ("raw_heal_available_rate", "rest_site_raw_heal_available_rate"),
    ("raw_smith_available_rate", "rest_site_raw_smith_available_rate"),
    ("low_hp_heal_filter_applied_rate", "rest_site_low_hp_heal_filter_applied_rate"),
    ("selected_non_heal_low_hp_rate", "rest_site_selected_non_heal_low_hp_rate"),
    ("smith_available_high_hp_not_selected_rate", "rest_site_smith_available_high_hp_not_selected_rate"),
    ("raw_minus_filtered_rest_count_mean", "rest_site_raw_minus_filtered_rest_count_mean"),
    ("selected_hp_ratio_mean", "rest_site_selected_hp_ratio_mean"),
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


def action_text_blob(action: Any) -> str:
    parts: list[str] = []
    for container in _action_containers(action):
        for key in (
            "kind",
            "action_type",
            "action_id",
            "label",
            "title",
            "name",
            "option_type",
            "canonical_text",
            "description",
        ):
            value = container.get(key)
            if value is not None:
                parts.append(str(value))
        option = container.get("option") if isinstance(container.get("option"), dict) else {}
        for key in (
            "option_id",
            "id",
            "type",
            "option_type",
            "title",
            "label",
            "name",
            "description",
            "is_enabled",
        ):
            value = option.get(key)
            if value is not None:
                parts.append(str(value))
        semantic = container.get("semantic") if isinstance(container.get("semantic"), dict) else {}
        for key in ("family", "domain", "semantic_key"):
            value = semantic.get(key)
            if value is not None:
                parts.append(str(value))
    return " ".join(parts).strip().lower()


def is_rest_site_action(action: Any) -> bool:
    """Return true for a real in-campfire option.

    Deliberately excludes route/map nodes with a future rest site and excludes
    terminal/proceed actions that appear after a campfire option resolved.
    """

    if not isinstance(action, dict):
        return False
    containers = _action_containers(action)
    for container in containers:
        action_id = _lower(container.get("action_id"))
        if action_id in {
            "rest_site:proceed",
            "rest_site:continue",
            "rest_site:leave",
            "rest_site:done",
            "rest_site:close",
        }:
            return False
        if action_id.startswith("rest_site:proceed"):
            return False
    for container in containers:
        kind = _lower(container.get("kind") or container.get("action_type"))
        if kind in {"rest", "rest_site", "choose_rest_option"}:
            return True
        semantic = container.get("semantic") if isinstance(container.get("semantic"), dict) else {}
        family = _lower(semantic.get("family"))
        domain = _lower(semantic.get("domain"))
        if domain == "build" and family in {"rest", "rest_site"}:
            return True
        action_id = _lower(container.get("action_id"))
        if (
            action_id.startswith("rest_site:")
            or action_id.startswith("choose_rest_option:")
            or action_id.startswith("sim:choose_rest_option")
        ):
            return True
    return False


def is_rest_heal_action(action: Any) -> bool:
    """Detect the concrete HEAL/REST campfire option.

    Important negative examples:
    ``rest_site:smith`` and a generic label ``Rest Site`` are not healing.
    """

    if not isinstance(action, dict):
        return False
    # Campfire options are mutually exclusive.  Some compact bridge payloads
    # have carried stale/derived fields such as ``action_kind="heal"`` or
    # ``is_heal=True`` on the SMITH row while still exposing the real identity
    # via title/option text ("锻造"/"Smith").  Always let the concrete smith
    # identity win before looking for heal/rest wording, otherwise diagnostics
    # and low-HP rest filters can train the policy that picking SMITH caused a
    # heal transition.
    if is_rest_smith_action(action):
        return False

    option_identity_tokens: list[str] = []
    option_title_tokens: list[str] = []
    option_description_tokens: list[str] = []
    action_id_tokens: list[str] = []
    label_tokens: list[str] = []
    family_tokens: list[str] = []
    semantic_key_tokens: list[str] = []

    for container in _action_containers(action):
        option = container.get("option") if isinstance(container.get("option"), dict) else {}
        for key in ("option_type", "type", "id", "option_id"):
            value = option.get(key)
            if value is not None:
                option_identity_tokens.append(str(value).strip().lower())
        value = container.get("option_type")
        if value is not None:
            option_identity_tokens.append(str(value).strip().lower())
        for key in ("title", "label", "name"):
            value = option.get(key)
            if value is not None:
                option_title_tokens.append(str(value).strip().lower())
        for key in ("title", "label", "name"):
            value = container.get(key)
            if value is not None:
                label_tokens.append(str(value).strip().lower())
        for key in ("description",):
            value = option.get(key)
            if value is not None:
                option_description_tokens.append(str(value).strip().lower())
            value = container.get(key)
            if value is not None:
                option_description_tokens.append(str(value).strip().lower())
        action_id = container.get("action_id")
        if action_id is not None:
            action_id_tokens.append(str(action_id).strip().lower())
        semantic = container.get("semantic") if isinstance(container.get("semantic"), dict) else {}
        if semantic.get("family") is not None:
            family_tokens.append(str(semantic.get("family")).strip().lower())
        if semantic.get("semantic_key") is not None:
            semantic_key_tokens.append(str(semantic.get("semantic_key")).strip().lower())

    heal_exact = {
        "rest",
        "heal",
        "healing",
        "sleep",
        "campfire_rest",
        "rest_option",
        "heal_option",
        "healrestsiteoption",
        "mend",
        "mendrestsiteoption",
        "休息",
        "治疗",
        "治療",
        "恢復",
        "恢复",
    }
    if any(token in heal_exact for token in option_identity_tokens):
        return True
    if any(token in heal_exact for token in option_title_tokens):
        return True
    if any(token in heal_exact for token in family_tokens + semantic_key_tokens):
        return True
    if any(
        action_id in {"rest", "heal"}
        or action_id.endswith(":rest")
        or action_id.endswith(":heal")
        or ":rest:" in action_id
        or ":heal:" in action_id
        or action_id.endswith("=rest")
        or action_id.endswith("=heal")
        or action_id.endswith("=rest_option")
        or action_id.endswith("=heal_option")
        for action_id in action_id_tokens
    ):
        return True

    positive_substrings = (
        "heal",
        "healing",
        "restore hp",
        "restore health",
        "recover hp",
        "recover health",
        "gain hp",
        "回复生命",
        "恢復生命",
        "恢复生命",
        "治疗",
        "治療",
    )
    title_blob = " ".join(option_title_tokens + label_tokens + option_description_tokens).strip()
    if any(token in title_blob for token in positive_substrings):
        return True
    if (
        ("回复" in title_blob or "恢復" in title_blob or "恢复" in title_blob or "治療" in title_blob)
        and ("生命" in title_blob or "hp" in title_blob or "health" in title_blob)
    ):
        return True
    return title_blob in {"rest", "休息"}


def is_rest_smith_action(action: Any) -> bool:
    if not isinstance(action, dict):
        return False
    if not is_rest_site_action(action):
        return False
    blob = action_text_blob(action)
    if not blob:
        return False
    smith_tokens = (
        "smith",
        "upgrade",
        "forge",
        "improve",
        "强化",
        "升级",
        "鍛造",
        "锻造",
    )
    return any(token in blob for token in smith_tokens)


def rest_action_kind(action: dict[str, Any] | None) -> str:
    if not is_rest_site_action(action):
        return ""
    if is_rest_smith_action(action):
        return "smith"
    if is_rest_heal_action(action):
        return "heal"
    return "other"


def _extract_player(obs: Any) -> dict[str, Any]:
    if not isinstance(obs, dict):
        return {}
    containers: list[dict[str, Any]] = [obs]
    for key in ("raw_obs", "transition_state", "state", "observation", "obs"):
        nested = obs.get(key)
        if isinstance(nested, dict):
            containers.append(nested)
    for container in containers:
        player = container.get("player")
        if isinstance(player, dict):
            return player
        combat = container.get("combat")
        if isinstance(combat, dict) and isinstance(combat.get("player"), dict):
            return combat["player"]
    return {}


def hp_ratio_from_obs(obs: Any) -> tuple[float, bool]:
    player = _extract_player(obs)
    if not player:
        return 0.0, False
    hp_raw = player.get("hp")
    if hp_raw is None:
        hp_raw = player.get("current_hp", player.get("currentHealth"))
    max_hp_raw = player.get("max_hp")
    if max_hp_raw is None:
        max_hp_raw = player.get("maxHealth")
    if max_hp_raw is None:
        max_hp_raw = player.get("maximum_hp")
    hp = _safe_float(hp_raw, 0.0)
    max_hp = _safe_float(max_hp_raw, 0.0)
    if hp <= 0.0:
        return 0.0, False
    if max_hp <= 1.0:
        return 0.0, False
    return hp / max_hp, True


def _compact_rest_action(action: dict[str, Any] | None, *, index: int | None = None, prob: float | None = None) -> dict[str, Any]:
    if not isinstance(action, dict):
        return {}
    option = action.get("option") if isinstance(action.get("option"), dict) else {}
    payload = action.get("payload") if isinstance(action.get("payload"), dict) else {}
    if not option and isinstance(payload.get("option"), dict):
        option = payload.get("option") or {}
    out: dict[str, Any] = {
        "index": int(index) if index is not None else None,
        "prob": float(prob) if prob is not None else None,
        "kind": action.get("kind") or payload.get("kind"),
        "action_id": action.get("action_id") or payload.get("action_id"),
        "action_kind": rest_action_kind(action),
        "label": action.get("label") or payload.get("label"),
        "title": action.get("title") or option.get("title") or payload.get("title"),
        "option_id": option.get("option_id") or option.get("id"),
        "option_type": option.get("option_type") or option.get("type"),
        "description": option.get("description") or action.get("description") or payload.get("description"),
        "is_rest_site": is_rest_site_action(action),
        "is_heal": is_rest_heal_action(action),
        "is_smith": is_rest_smith_action(action),
    }
    return {key: value for key, value in out.items() if value not in (None, "")}


def rest_available_summary(legal_actions: list[Any]) -> dict[str, Any]:
    actions = legal_actions if isinstance(legal_actions, list) else []
    rest_actions = [action for action in actions if is_rest_site_action(action)]
    heal_actions = [action for action in rest_actions if is_rest_heal_action(action)]
    smith_actions = [action for action in rest_actions if is_rest_smith_action(action)]
    return {
        "rest_action_count": float(len(rest_actions)),
        "heal_available": bool(heal_actions),
        "smith_available": bool(smith_actions),
        "other_rest_available": bool(len(rest_actions) > len(heal_actions) + len(smith_actions)),
        "heal_action_count": float(len(heal_actions)),
        "smith_action_count": float(len(smith_actions)),
    }


def _top_policy_rest_actions(legal_actions: list[Any], search_policy: Any, *, max_topk: int) -> list[dict[str, Any]]:
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
        if is_rest_site_action(action)
    ]
    indexed.sort(key=lambda item: item[1], reverse=True)
    return [
        _compact_rest_action(action, index=idx, prob=prob)
        for idx, prob, action in indexed[: max(1, int(max_topk))]
    ]


def build_rest_site_choice_payload(
    *,
    decision_domain: str,
    phase: str,
    legal_actions: list[Any],
    raw_legal_actions: list[Any] | None,
    chosen_action: dict[str, Any] | None,
    chosen_signature: dict[str, Any] | None,
    selected_index: int,
    progress: dict[str, Any] | None = None,
    raw_obs: Any = None,
    search_policy: Any = None,
    max_topk: int = 8,
) -> dict[str, Any] | None:
    """Build one compact diagnostic record for a rest-site/campfire choice."""

    legal_actions = legal_actions if isinstance(legal_actions, list) else []
    raw_actions = raw_legal_actions if isinstance(raw_legal_actions, list) else []
    rest_context = (
        "rest" in _lower(phase)
        or "campfire" in _lower(phase)
        or is_rest_site_action(chosen_action)
        or is_rest_site_action(chosen_signature)
        or any(is_rest_site_action(action) for action in legal_actions)
        or any(is_rest_site_action(action) for action in raw_actions)
    )
    if not rest_context:
        return None

    selected = chosen_action if isinstance(chosen_action, dict) else chosen_signature
    selected_kind = rest_action_kind(selected if isinstance(selected, dict) else None) or "non_rest"
    summary = rest_available_summary(legal_actions)
    raw_summary = rest_available_summary(raw_actions)
    hp_ratio, hp_valid = hp_ratio_from_obs(raw_obs)
    is_low_hp = bool(hp_valid and hp_ratio < float(REST_SITE_SKIP_HEAL_HP_THRESHOLD))
    filtered_rest_count = _safe_float(summary.get("rest_action_count"), 0.0)
    raw_rest_count = _safe_float(raw_summary.get("rest_action_count"), 0.0)
    # Simpler and easier to read in JSONL: raw exposed more rest options than
    # the policy saw, at low HP, while heal existed.  This is exactly the
    # low-HP rest filter's intended footprint.
    low_hp_heal_filter_applied = bool(
        is_low_hp
        and raw_rest_count > filtered_rest_count
        and _safe_bool(raw_summary.get("heal_available"))
    )
    selected_non_heal_low_hp = bool(is_low_hp and selected_kind in {"smith", "other"})
    smith_available_high_hp_not_selected = bool(
        hp_valid
        and not is_low_hp
        and _safe_bool(summary.get("smith_available"))
        and selected_kind != "smith"
    )

    payload: dict[str, Any] = {
        "decision_domain": str(decision_domain or ""),
        "phase": str(phase or ""),
        "selected_index": int(selected_index),
        "selected_action_kind": selected_kind,
        "selected_action": _compact_rest_action(selected if isinstance(selected, dict) else None, index=selected_index),
        "top_policy_rest_actions": _top_policy_rest_actions(legal_actions, search_policy, max_topk=max_topk),
        "legal_rest_actions": [
            _compact_rest_action(action, index=idx)
            for idx, action in enumerate(legal_actions)
            if is_rest_site_action(action)
        ][: max(1, int(max_topk))],
        "raw_rest_actions": [
            _compact_rest_action(action, index=idx)
            for idx, action in enumerate(raw_actions)
            if is_rest_site_action(action)
        ][: max(1, int(max_topk))],
        "hp_ratio": hp_ratio,
        "hp_valid": bool(hp_valid),
        "hp_threshold": float(REST_SITE_SKIP_HEAL_HP_THRESHOLD),
        "is_low_hp": bool(is_low_hp),
        "low_hp_heal_filter_applied": bool(low_hp_heal_filter_applied),
        "selected_non_heal_low_hp": bool(selected_non_heal_low_hp),
        "smith_available_high_hp_not_selected": bool(smith_available_high_hp_not_selected),
        "raw_minus_filtered_rest_count": max(0.0, raw_rest_count - filtered_rest_count),
        **summary,
        "raw_rest_action_count": _safe_float(raw_summary.get("rest_action_count"), 0.0),
        "raw_heal_available": _safe_bool(raw_summary.get("heal_available")),
        "raw_smith_available": _safe_bool(raw_summary.get("smith_available")),
        "raw_heal_action_count": _safe_float(raw_summary.get("heal_action_count"), 0.0),
        "raw_smith_action_count": _safe_float(raw_summary.get("smith_action_count"), 0.0),
    }
    if isinstance(progress, dict):
        for key in ("floor", "act_id", "room_type", "room_type_lower", "encounter_id"):
            if key in progress:
                payload[key] = progress.get(key)
    return payload


@dataclass
class RestSiteEpisodeTracker:
    seen: int = 0
    heal: int = 0
    smith: int = 0
    other: int = 0
    low_hp: int = 0
    high_hp: int = 0
    heal_available: int = 0
    smith_available: int = 0
    raw_heal_available: int = 0
    raw_smith_available: int = 0
    low_hp_heal_filter_applied: int = 0
    selected_non_heal_low_hp: int = 0
    smith_available_high_hp_not_selected: int = 0
    raw_minus_filtered_rest_count_sum: float = 0.0
    hp_ratio_sum: float = 0.0
    hp_ratio_count: int = 0

    def update(self, payload: dict[str, Any] | None) -> None:
        if not isinstance(payload, dict):
            return
        # Only count actual rest-site choices, not phase-level context records
        # whose selected action is unrelated cleanup.
        if _safe_float(payload.get("rest_action_count"), 0.0) <= 0.0 and _safe_float(
            payload.get("raw_rest_action_count"), 0.0
        ) <= 0.0:
            return
        self.seen += 1
        kind = _lower(payload.get("selected_action_kind"))
        if kind == "heal":
            self.heal += 1
        elif kind == "smith":
            self.smith += 1
        else:
            self.other += 1
        if _safe_bool(payload.get("is_low_hp")):
            self.low_hp += 1
        elif _safe_bool(payload.get("hp_valid")):
            self.high_hp += 1
        if _safe_bool(payload.get("heal_available")):
            self.heal_available += 1
        if _safe_bool(payload.get("smith_available")):
            self.smith_available += 1
        if _safe_bool(payload.get("raw_heal_available")):
            self.raw_heal_available += 1
        if _safe_bool(payload.get("raw_smith_available")):
            self.raw_smith_available += 1
        if _safe_bool(payload.get("low_hp_heal_filter_applied")):
            self.low_hp_heal_filter_applied += 1
        if _safe_bool(payload.get("selected_non_heal_low_hp")):
            self.selected_non_heal_low_hp += 1
        if _safe_bool(payload.get("smith_available_high_hp_not_selected")):
            self.smith_available_high_hp_not_selected += 1
        self.raw_minus_filtered_rest_count_sum += _safe_float(payload.get("raw_minus_filtered_rest_count"), 0.0)
        if _safe_bool(payload.get("hp_valid")):
            self.hp_ratio_sum += _safe_float(payload.get("hp_ratio"), 0.0)
            self.hp_ratio_count += 1

    def as_metadata(self) -> dict[str, float]:
        seen_safe = max(int(self.seen), 1)
        non_heal = int(self.smith) + int(self.other)
        return {
            "rest_site_seen_count": float(self.seen),
            "rest_site_heal_count": float(self.heal),
            "rest_site_smith_count": float(self.smith),
            "rest_site_other_count": float(self.other),
            "rest_site_heal_rate": float(self.heal) / float(seen_safe),
            "rest_site_smith_rate": float(self.smith) / float(seen_safe),
            "rest_site_non_heal_rate": float(non_heal) / float(seen_safe),
            "rest_site_other_rate": float(self.other) / float(seen_safe),
            "rest_site_low_hp_rate": float(self.low_hp) / float(seen_safe),
            "rest_site_high_hp_rate": float(self.high_hp) / float(seen_safe),
            "rest_site_heal_available_rate": float(self.heal_available) / float(seen_safe),
            "rest_site_smith_available_rate": float(self.smith_available) / float(seen_safe),
            "rest_site_raw_heal_available_rate": float(self.raw_heal_available) / float(seen_safe),
            "rest_site_raw_smith_available_rate": float(self.raw_smith_available) / float(seen_safe),
            "rest_site_low_hp_heal_filter_applied_rate": float(self.low_hp_heal_filter_applied) / float(seen_safe),
            "rest_site_selected_non_heal_low_hp_rate": float(self.selected_non_heal_low_hp) / float(seen_safe),
            "rest_site_smith_available_high_hp_not_selected_rate": (
                float(self.smith_available_high_hp_not_selected) / float(seen_safe)
            ),
            "rest_site_raw_minus_filtered_rest_count_mean": (
                self.raw_minus_filtered_rest_count_sum / float(seen_safe)
            ),
            "rest_site_selected_hp_ratio_mean": self.hp_ratio_sum / float(max(self.hp_ratio_count, 1)),
        }


def dump_rest_site_choice_diagnostic(trainer: Any, payload: dict[str, Any] | None) -> None:
    """Append one bounded ``rest_site_choices.jsonl`` record using trainer paths."""

    if not isinstance(payload, dict):
        return
    if getattr(trainer, "_rest_site_choice_dump_disabled", False):
        return
    cap = int(getattr(trainer, "_rest_site_choice_dump_max", 100000))
    count = int(getattr(trainer, "_rest_site_choice_dump_count", 0) or 0)
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
        path = path_getter("rest_site_choices.jsonl")
        if path is None:
            return
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("a", encoding="utf-8") as fh:
            fh.write(json.dumps(record, ensure_ascii=False, default=str) + "\n")
        trainer._rest_site_choice_dump_count = count + 1
    except Exception:
        return
