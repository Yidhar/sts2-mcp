"""Evaluation-only trajectory diagnostics and semantic deadlock detection.

Training data and diagnostics deliberately have different lifetimes in v2.
Rollout unrolls are consumed once by the learner; this module writes compact,
human-auditable JSONL events and detects policy loops without feeding those
records back into training.

The detector is intentionally mechanics-agnostic.  It hashes a canonical
projection of the complete observation and legal candidates after removing
transport identities, request handles, timestamps, and revision counters.  A
deadlock is an exact recurring semantic state/action pair, not a handwritten
card, room, boss, or UI rule.
"""

from __future__ import annotations

import hashlib
import json
import math
from collections import Counter, deque
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass
from itertools import pairwise
from pathlib import Path
from typing import Any, Final, TextIO

TRAJECTORY_JOURNAL_VERSION: Final = "sts2-trajectory-journal-v3"
SEMANTIC_FINGERPRINT_VERSION: Final = "sts2-semantic-decision-v1"

_DEFAULT_SNAPSHOT_INTERVAL: Final = 256
_DEFAULT_ANOMALY_CONTEXT_STEPS: Final = 8
_COMPACT_STRING_MAX_BYTES: Final = 256
_COMPACT_STRING_PREFIX_BYTES: Final = 160
_COMPACT_MAPPING_WIDTH_KEY: Final = "_mapping_width"

# These fields account for almost all of the old journal volume.  Compact
# decision records retain counts/identities for them; bounded rich snapshots
# retain the complete payload at episode boundaries, anomalies, and a sparse
# cadence.
_LARGE_OBSERVATION_COLLECTIONS: Final = frozenset(
    {
        "available_actions",
        "deck",
        "deck_cards",
        "discard_pile",
        "draw_pile",
        "enemies",
        "exhaust_pile",
        "hand",
        "play_pile",
        "potions",
        "powers",
        "relics",
    }
)
_VERBOSE_DIAGNOSTIC_KEYS: Final = frozenset(
    {
        "description",
        "dynamic_vars",
        "hover_tip_ids",
        "note",
    }
)

# Values under these keys describe a transport transaction rather than the
# game decision.  Removing them lets the same semantic loop compare equal even
# though every request has a fresh UUID and every step advances a revision.
_VOLATILE_KEYS: Final = frozenset(
    {
        "action_handle",
        "action_id",
        "client_id",
        "created_at",
        "elapsed_ms",
        "episode_id",
        "expected_state_version",
        "expected_step_index",
        "idempotency_key",
        "request_id",
        "revision",
        "session_id",
        "state_version",
        "step_index",
        "timestamp",
        "updated_at",
    }
)


def _canonical_number(value: float) -> int | float | str:
    """Return deterministic JSON-safe numeric content.

    Game facts should be finite.  Diagnostics remain total in the face of a
    malformed payload so that the journal can explain the protocol failure.
    """

    normalized = float(value)
    if math.isnan(normalized):
        return "NaN"
    if math.isinf(normalized):
        return "+Infinity" if normalized > 0.0 else "-Infinity"
    if normalized == 0.0:
        return 0
    if normalized.is_integer() and abs(normalized) <= 2**53:
        return int(normalized)
    return normalized


def semantic_projection(value: Any) -> Any:
    """Canonicalize nested DTO data while dropping only transport identities."""

    if isinstance(value, Mapping):
        projected: dict[str, Any] = {}
        for raw_key in sorted(value, key=lambda item: str(item)):
            key = str(raw_key)
            lowered = key.lower()
            if lowered.startswith("_") or lowered in _VOLATILE_KEYS:
                continue
            projected[key] = semantic_projection(value[raw_key])
        return projected
    if isinstance(value, list | tuple):
        return [semantic_projection(item) for item in value]
    if value is None or isinstance(value, str | bool | int):
        return value
    if isinstance(value, float):
        return _canonical_number(value)
    # Keep diagnostics serializable without trusting arbitrary DTO extensions.
    return str(value)


def diagnostic_projection(value: Any) -> Any:
    """Make a journal event JSON-safe without erasing episode/step identity."""

    if isinstance(value, Mapping):
        return {
            str(key): diagnostic_projection(child)
            for key, child in sorted(value.items(), key=lambda item: str(item[0]))
            if not str(key).startswith("_")
        }
    if isinstance(value, list | tuple):
        return [diagnostic_projection(item) for item in value]
    if value is None or isinstance(value, str | bool | int):
        return value
    if isinstance(value, float):
        return _canonical_number(value)
    return str(value)


def _compact_scalar(value: Any) -> Any:
    """Bound strings in summary records without losing their identity.

    Rich snapshots continue to retain the complete semantic value.  The
    prefix/length/digest representation makes each compact scalar bounded even
    when a bridge extension unexpectedly exposes a full localized description
    through a field such as ``label`` or ``text_key``.
    """

    projected = diagnostic_projection(value)
    if not isinstance(projected, str):
        return projected
    encoded = projected.encode("utf-8")
    if len(encoded) <= _COMPACT_STRING_MAX_BYTES:
        return projected
    prefix = encoded[:_COMPACT_STRING_PREFIX_BYTES].decode("utf-8", errors="ignore")
    return {
        "string_prefix": prefix,
        "utf8_bytes": len(encoded),
        "sha256": hashlib.sha256(encoded).hexdigest(),
    }


def _compact_kind(value: Any) -> str:
    """Return a bounded dictionary key for candidate-kind counts."""

    raw = str(value or "unknown")
    encoded = raw.encode("utf-8")
    if len(encoded) <= _COMPACT_STRING_MAX_BYTES:
        return raw
    prefix = encoded[:_COMPACT_STRING_PREFIX_BYTES].decode("utf-8", errors="ignore")
    return f"{prefix}#sha256:{hashlib.sha256(encoded).hexdigest()}"


def _bounded_compact_projection(
    value: Any,
    *,
    depth: int = 0,
    max_depth: int = 2,
    max_items: int = 12,
) -> Any:
    """Return a bounded, human-readable projection for per-step summaries.

    This is deliberately not used for model input, rewards, fingerprints, or
    evaluation metrics.  It only prevents descriptive card/event DTOs from
    being duplicated into every diagnostic line.
    """

    if isinstance(value, Mapping):
        compact: dict[str, Any] = {}
        items = [
            (str(raw_key), child)
            for raw_key, child in sorted(
                value.items(),
                key=lambda item: str(item[0]),
            )
            if not str(raw_key).startswith("_") and str(raw_key) not in _VERBOSE_DIAGNOSTIC_KEYS
        ]
        for key, child in items[:max_items]:
            if key in _LARGE_OBSERVATION_COLLECTIONS and isinstance(child, list | tuple):
                compact[f"{key}_count"] = len(child)
                continue
            if child is None or isinstance(child, str | bool | int | float):
                compact[key] = _compact_scalar(child)
            elif depth < max_depth and isinstance(child, Mapping):
                projected = _bounded_compact_projection(
                    child,
                    depth=depth + 1,
                    max_depth=max_depth,
                    max_items=max_items,
                )
                if projected:
                    compact[key] = projected
            elif isinstance(child, list | tuple):
                if child and all(item is None or isinstance(item, str | bool | int | float) for item in child):
                    compact[key] = [_compact_scalar(item) for item in child[:max_items]]
                    if len(child) > max_items:
                        compact[f"{key}_count"] = len(child)
                else:
                    compact[f"{key}_count"] = len(child)
        if len(items) > max_items:
            compact[_COMPACT_MAPPING_WIDTH_KEY] = {
                "key_count": len(items),
                "omitted_count": len(items) - max_items,
            }
        return compact
    if isinstance(value, list | tuple):
        return [
            _bounded_compact_projection(
                item,
                depth=depth + 1,
                max_depth=max_depth,
                max_items=max_items,
            )
            for item in value[:max_items]
        ]
    return _compact_scalar(value)


def _compact_entity(
    value: Any,
    *,
    keys: Sequence[str],
) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        return {}
    return {
        key: _compact_scalar(value[key])
        for key in keys
        if key in value and (value[key] is None or isinstance(value[key], str | bool | int | float))
    }


def _collection_count(value: Any) -> int | float | None:
    """Read a collection count from both canonical and live bridge shapes."""

    if isinstance(value, bool):
        return None
    if isinstance(value, int | float):
        return value
    if isinstance(value, list | tuple):
        return len(value)
    if not isinstance(value, Mapping):
        return None
    explicit = value.get("count")
    if isinstance(explicit, int | float) and not isinstance(explicit, bool):
        return explicit
    for key in ("cards", "items"):
        items = value.get(key)
        if isinstance(items, list | tuple):
            return len(items)
    return None


def _compact_cards(value: Any, *, limit: int = 12) -> list[dict[str, Any]]:
    """Return a bounded visible-card preview from list or ``{cards: ...}`` DTOs."""

    if isinstance(value, Mapping):
        value = value.get("cards", value.get("items"))
    if not isinstance(value, list | tuple):
        return []
    return [
        _compact_entity(
            card,
            keys=(
                "id",
                "model_id",
                "index",
                "cost",
                "star_cost",
                "type",
                "is_upgraded",
            ),
        )
        for card in value[:limit]
    ]


def _compact_observation(observation: Any) -> dict[str, Any]:
    """Keep run position and durable combat/resource facts on every step."""

    if not isinstance(observation, Mapping):
        return {"value": _compact_scalar(observation)}
    compact = _compact_entity(
        observation,
        keys=(
            "phase",
            "decision_domain",
            "state_type",
            "semantic_state_hash",
            "terminated",
            "truncated",
        ),
    )
    run = _compact_entity(
        observation.get("run"),
        keys=(
            "act",
            "floor",
            "room_model_id",
            "room_model",
            "room_type",
            "character_id",
            "ascension",
            "ascension_level",
        ),
    )
    if run:
        compact["run"] = run

    player_value = observation.get("player")
    if isinstance(player_value, Mapping):
        player = _compact_entity(
            player_value,
            keys=(
                "character",
                "character_id",
                "hp",
                "max_hp",
                "block",
                "energy",
                "max_energy",
                "gold",
                "draw_pile_count",
                "discard_pile_count",
                "exhaust_pile_count",
                "open_potion_slots",
            ),
        )
        for collection_name in (
            "deck",
            "deck_cards",
            "hand",
            "draw_pile",
            "discard_pile",
            "exhaust_pile",
            "relics",
            "potions",
            "powers",
        ):
            collection = player_value.get(collection_name)
            count = _collection_count(collection)
            if count is not None:
                player[f"{collection_name}_count"] = count
        if "deck_count" not in player:
            deck_count = _collection_count(player_value.get("deck_cards"))
            if deck_count is not None:
                player["deck_count"] = deck_count
        hand = player_value.get("hand")
        hand_cards = _compact_cards(hand)
        if hand_cards:
            player["hand"] = hand_cards
        compact["player"] = player

    combat_value = observation.get("combat")
    if isinstance(combat_value, Mapping):
        combat = _compact_entity(
            combat_value,
            keys=(
                "in_progress",
                "round",
                "turn",
                "is_play_phase",
                "energy",
                "current_energy",
                "max_energy",
                "stars",
                "current_stars",
                "draw",
                "discard",
                "exhaust",
            ),
        )
        combat_hand = combat_value.get("hand")
        if combat_hand is None and isinstance(player_value, Mapping):
            combat_hand = player_value.get("hand")
        hand_count = _collection_count(combat_hand)
        if hand_count is not None:
            combat["hand_count"] = hand_count
            combat["hand"] = _compact_cards(combat_hand)

        for pile_name, count_alias in (
            ("draw_pile", "draw"),
            ("discard_pile", "discard"),
            ("exhaust_pile", "exhaust"),
            ("play_pile", "play"),
        ):
            pile_value = combat_value.get(pile_name)
            if pile_value is None and isinstance(player_value, Mapping):
                pile_value = player_value.get(pile_name)
            pile_count = _collection_count(pile_value)
            if pile_count is None:
                pile_count = _collection_count(combat_value.get(count_alias))
            if pile_count is not None:
                combat[f"{pile_name}_count"] = pile_count
        enemies = combat_value.get("enemies")
        if isinstance(enemies, list | tuple):
            combat["enemy_count"] = len(enemies)
            combat["enemies"] = [
                _compact_entity(
                    enemy,
                    keys=(
                        "id",
                        "index",
                        "hp",
                        "max_hp",
                        "block",
                        "intent",
                        "intent_damage",
                        "intent_hits",
                        "is_alive",
                    ),
                )
                for enemy in enemies[:8]
            ]
        compact["combat"] = combat

    # Domain-specific panes are usually small once descriptive text and large
    # entity collections are reduced to counts.  Keeping them makes event,
    # selection, shop, and reward loops understandable without full DTOs.
    for domain_key in (
        "event",
        "screen",
        "card_selection",
        "card_reward_selection",
        "rewards",
        "rest_site",
        "shop",
        "map",
    ):
        domain_value = observation.get(domain_key)
        if domain_value:
            projected = _bounded_compact_projection(
                domain_value,
                max_depth=2,
                max_items=8,
            )
            if projected:
                compact[domain_key] = projected
    return compact


def _compact_action(action: Any) -> dict[str, Any]:
    if not isinstance(action, Mapping):
        return {"value": _compact_scalar(action)}
    projected = _bounded_compact_projection(action, max_depth=2, max_items=8)
    if not isinstance(projected, dict):  # pragma: no cover - mapping above
        raise TypeError("compact action projection must be a mapping")
    return projected


def canonical_json(value: Any) -> str:
    return json.dumps(
        semantic_projection(value),
        ensure_ascii=False,
        allow_nan=False,
        sort_keys=True,
        separators=(",", ":"),
    )


def semantic_fingerprint(value: Any) -> str:
    return hashlib.sha256(canonical_json(value).encode("utf-8")).hexdigest()


def semantic_decision_fingerprint(
    observation: Mapping[str, Any],
    legal_actions: Sequence[Mapping[str, Any]],
) -> str:
    """Hash the candidate-order-independent semantic decision surface."""

    actions = sorted(canonical_json(action) for action in legal_actions)
    return semantic_fingerprint(
        {
            "version": SEMANTIC_FINGERPRINT_VERSION,
            "observation": observation,
            "legal_actions": actions,
        }
    )


def semantic_action_fingerprint(action: Mapping[str, Any]) -> str:
    return semantic_fingerprint({"version": SEMANTIC_FINGERPRINT_VERSION, "action": action})


@dataclass(frozen=True, slots=True)
class DeadlockEvidence:
    """Exact evidence emitted when one semantic decision/action keeps recurring."""

    decision_fingerprint: str
    action_fingerprint: str
    occurrences: int
    window_size: int
    first_step: int
    last_step: int
    cycle_span: int

    def to_mapping(self) -> dict[str, int | str]:
        return asdict(self)


@dataclass(frozen=True, slots=True)
class _DecisionAction:
    step_index: int
    decision_fingerprint: str
    action_fingerprint: str


class SemanticDeadlockDetector:
    """Detect recurring semantic state/action pairs in a bounded window.

    No progress signal is handcrafted.  If an action changes any retained game
    fact or the legal candidate set, the decision fingerprint changes.  The
    threshold therefore captures exact fixed points and short deterministic
    cycles while ignoring transport-only revision churn.
    """

    def __init__(self, *, window_size: int = 128, repeat_threshold: int = 8) -> None:
        if isinstance(window_size, bool) or not isinstance(window_size, int):
            raise TypeError("deadlock window_size must be an integer")
        if isinstance(repeat_threshold, bool) or not isinstance(repeat_threshold, int):
            raise TypeError("deadlock repeat_threshold must be an integer")
        if window_size < 2:
            raise ValueError("deadlock window_size must be at least 2")
        if repeat_threshold < 2:
            raise ValueError("deadlock repeat_threshold must be at least 2")
        if repeat_threshold > window_size:
            raise ValueError("deadlock repeat_threshold cannot exceed window_size")
        self.window_size = window_size
        self.repeat_threshold = repeat_threshold
        self._history: deque[_DecisionAction] = deque(maxlen=window_size)

    def reset(self) -> None:
        self._history.clear()

    def observe(
        self,
        *,
        step_index: int,
        observation: Mapping[str, Any],
        legal_actions: Sequence[Mapping[str, Any]],
        selected_action: Mapping[str, Any],
    ) -> DeadlockEvidence | None:
        if isinstance(step_index, bool) or not isinstance(step_index, int):
            raise TypeError("deadlock step_index must be an integer")
        if step_index < 0:
            raise ValueError("deadlock step_index must be non-negative")
        item = _DecisionAction(
            step_index=step_index,
            decision_fingerprint=semantic_decision_fingerprint(observation, legal_actions),
            action_fingerprint=semantic_action_fingerprint(selected_action),
        )
        self._history.append(item)
        key = (item.decision_fingerprint, item.action_fingerprint)
        matching = [event for event in self._history if (event.decision_fingerprint, event.action_fingerprint) == key]
        if len(matching) < self.repeat_threshold:
            return None
        first = matching[0].step_index
        last = matching[-1].step_index
        deltas = [right.step_index - left.step_index for left, right in pairwise(matching)]
        cycle_span = Counter(deltas).most_common(1)[0][0] if deltas else 0
        return DeadlockEvidence(
            decision_fingerprint=item.decision_fingerprint,
            action_fingerprint=item.action_fingerprint,
            occurrences=len(matching),
            window_size=len(self._history),
            first_step=first,
            last_step=last,
            cycle_span=cycle_span,
        )


class TrajectoryJournal:
    """Bounded-volume JSONL writer used only for evaluation diagnostics.

    Every decision produces a compact record.  Complete observation and legal
    candidate DTOs are written only for the first/last decision of an episode,
    at a sparse cadence, and around detected anomalies.  Evaluation metrics
    are computed independently from this diagnostic stream and are unchanged.
    """

    def __init__(
        self,
        path: str | Path,
        *,
        snapshot_interval: int = _DEFAULT_SNAPSHOT_INTERVAL,
        anomaly_context_steps: int = _DEFAULT_ANOMALY_CONTEXT_STEPS,
    ) -> None:
        if isinstance(snapshot_interval, bool) or not isinstance(snapshot_interval, int):
            raise TypeError("snapshot_interval must be an integer")
        if snapshot_interval <= 0:
            raise ValueError("snapshot_interval must be positive")
        if isinstance(anomaly_context_steps, bool) or not isinstance(anomaly_context_steps, int):
            raise TypeError("anomaly_context_steps must be an integer")
        if anomaly_context_steps < 0:
            raise ValueError("anomaly_context_steps must be non-negative")
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.snapshot_interval = snapshot_interval
        self.anomaly_context_steps = anomaly_context_steps
        self._handle: TextIO | None = None
        self._episode_id: str | None = None
        self._episode_decisions = 0
        self._recent: deque[Mapping[str, Any]] = deque(maxlen=max(1, anomaly_context_steps))
        self._snapshot_steps: set[int] = set()
        self._last_event: Mapping[str, Any] | None = None
        self._last_marked_episode_last = False

    def __enter__(self) -> TrajectoryJournal:
        self._handle = self.path.open("a", encoding="utf-8", newline="\n")
        return self

    def __exit__(self, *_: object) -> None:
        self.close()

    def close(self) -> None:
        if self._handle is not None:
            self._finish_episode()
            self._handle.flush()
            self._handle.close()
            self._handle = None

    def write(self, event: Mapping[str, Any]) -> None:
        if self._handle is None:
            raise RuntimeError("trajectory journal must be opened before writing")
        if event.get("event") != "decision":
            self._write_payload(diagnostic_projection(event))
            return

        episode_id = str(event.get("episode_id", ""))
        if self._episode_id is not None and episode_id != self._episode_id:
            self._finish_episode()
            self._reset_episode(episode_id)
        elif self._episode_id is None:
            self._reset_episode(episode_id)

        current = dict(event)
        first = self._episode_decisions == 0
        anomaly = self._is_anomaly(current)
        if anomaly and self.anomaly_context_steps > 0:
            for context in self._recent:
                self._write_snapshot(context, reasons=("anomaly_context",))

        self._write_payload(self._decision_summary(current))
        self._episode_decisions += 1
        self._recent.append(current)
        self._last_event = current

        reasons: list[str] = []
        if first:
            reasons.append("episode_first")
        if self._episode_decisions % self.snapshot_interval == 0:
            reasons.append("periodic")
        if anomaly:
            reasons.append("anomaly")
        if self._is_terminal(current):
            reasons.append("episode_last")
            self._last_marked_episode_last = True
        if reasons:
            self._write_snapshot(current, reasons=reasons)

    def write_episode_boundary(self, event: Mapping[str, Any]) -> None:
        """Finish the active diagnostic episode, then write a boundary event.

        Infrastructure retry markers must not be written through ``write``:
        a failed attempt can end between two successful decisions, and the
        journal otherwise defers its forced final snapshot until it sees the
        *next* decision episode id.  That would place an attempt-1 snapshot
        after the attempt-2 ``started`` marker.  This explicit operation keeps
        attempt boundaries auditable without teaching the journal anything
        about evaluation retry policy.
        """

        if self._handle is None:
            raise RuntimeError("trajectory journal must be opened before writing")
        if event.get("event") == "decision":
            raise ValueError("decision events must be written with write()")
        self._finish_episode()
        self._write_payload(diagnostic_projection(event))

    def _reset_episode(self, episode_id: str) -> None:
        self._episode_id = episode_id
        self._episode_decisions = 0
        self._recent.clear()
        self._snapshot_steps.clear()
        self._last_event = None
        self._last_marked_episode_last = False

    def _finish_episode(self) -> None:
        if self._last_event is not None and not self._last_marked_episode_last:
            # A sparse/first snapshot of the same step is not enough to make
            # the boundary explicit.  One final rich record is a bounded cost.
            self._write_snapshot(
                self._last_event,
                reasons=("episode_last",),
                force=True,
            )
        self._episode_id = None
        self._episode_decisions = 0
        self._recent.clear()
        self._snapshot_steps.clear()
        self._last_event = None
        self._last_marked_episode_last = False

    @staticmethod
    def _is_terminal(event: Mapping[str, Any]) -> bool:
        return str(event.get("outcome", "ongoing")) != "ongoing"

    @classmethod
    def _is_anomaly(cls, event: Mapping[str, Any]) -> bool:
        if event.get("deadlock") is not None:
            return True
        return str(event.get("outcome", "ongoing")) in {"deadlock", "horizon"}

    def _decision_summary(self, event: Mapping[str, Any]) -> dict[str, Any]:
        observation = event.get("observation")
        legal_value = event.get("legal_actions")
        legal_actions = legal_value if isinstance(legal_value, list | tuple) else ()
        selected_value = event.get("selected_action")
        selected_action = selected_value if isinstance(selected_value, Mapping) else {}
        candidate_kinds = Counter(
            _compact_kind(action.get("model_action_kind", action.get("kind", "unknown")))
            for action in legal_actions
            if isinstance(action, Mapping)
        )
        policy_topk: list[dict[str, Any]] = []
        raw_topk = event.get("policy_topk")
        if isinstance(raw_topk, list | tuple):
            for item in raw_topk:
                if not isinstance(item, Mapping):
                    continue
                index_value = item.get("index")
                candidate: Mapping[str, Any] | None = None
                if (
                    isinstance(index_value, int)
                    and not isinstance(index_value, bool)
                    and 0 <= index_value < len(legal_actions)
                    and isinstance(legal_actions[index_value], Mapping)
                ):
                    candidate = legal_actions[index_value]
                topk_item: dict[str, Any] = {
                    "index": _compact_scalar(index_value),
                    "probability": _compact_scalar(item.get("probability")),
                }
                if candidate is not None:
                    topk_item["action"] = _compact_action(candidate)
                    topk_item["action_fingerprint"] = semantic_action_fingerprint(candidate)
                policy_topk.append(topk_item)

        summary: dict[str, Any] = {
            "event": "decision",
            "record_kind": "summary",
            "episode_id": _compact_scalar(event.get("episode_id")),
            "reset_seed": _compact_scalar(event.get("reset_seed")),
            "step_index": _compact_scalar(event.get("step_index")),
            "observation_summary": _compact_observation(observation),
            "legal_action_count": len(legal_actions),
            "legal_action_kinds": dict(sorted(candidate_kinds.items())),
            "selected_index": _compact_scalar(event.get("selected_index")),
            "selected_action": _compact_action(selected_action),
            "selected_action_fingerprint": (semantic_action_fingerprint(selected_action) if selected_action else None),
            "policy_topk": policy_topk,
        }
        for key in (
            "value",
            "reward",
            "terminal_reward",
            "potential_reward",
            "revival_penalty",
            "pace_penalty",
            "hp_loss_penalty",
            "player_hp_lost",
            "revivals_used",
            "outcome",
            "deadlock",
        ):
            if key in event:
                summary[key] = _bounded_compact_projection(
                    event[key],
                    max_depth=3,
                    max_items=16,
                )
        return summary

    def _write_snapshot(
        self,
        event: Mapping[str, Any],
        *,
        reasons: Sequence[str],
        force: bool = False,
    ) -> None:
        step_value = event.get("step_index")
        step_index = step_value if isinstance(step_value, int) and not isinstance(step_value, bool) else -1
        if not force and step_index in self._snapshot_steps:
            return
        # Rich snapshots remain semantic rather than transport transcripts.
        # This preserves complete game/candidate evidence without retaining
        # per-request UUIDs and revision counters. Project each field only once
        # so even a sparse large-deck snapshot avoids a redundant deep copy.
        semantic_keys = {
            "observation",
            "legal_actions",
            "selected_action",
            "result_observation",
            "result_legal_actions",
        }
        projected = {
            str(key): (semantic_projection(value) if str(key) in semantic_keys else diagnostic_projection(value))
            for key, value in sorted(event.items(), key=lambda item: str(item[0]))
            if not str(key).startswith("_")
        }
        projected["event"] = "decision_snapshot"
        projected["record_kind"] = "rich_snapshot"
        projected["snapshot_reasons"] = list(dict.fromkeys(reasons))
        self._write_payload(projected)
        self._snapshot_steps.add(step_index)

    def _write_payload(self, event: Mapping[str, Any]) -> None:
        if self._handle is None:  # pragma: no cover - guarded by write/close
            raise RuntimeError("trajectory journal must be opened before writing")
        payload = {
            **event,
            "journal_version": TRAJECTORY_JOURNAL_VERSION,
        }
        self._handle.write(
            json.dumps(
                payload,
                ensure_ascii=False,
                allow_nan=False,
                sort_keys=True,
                separators=(",", ":"),
            )
            + "\n"
        )


__all__ = [
    "SEMANTIC_FINGERPRINT_VERSION",
    "TRAJECTORY_JOURNAL_VERSION",
    "DeadlockEvidence",
    "SemanticDeadlockDetector",
    "TrajectoryJournal",
    "canonical_json",
    "diagnostic_projection",
    "semantic_action_fingerprint",
    "semantic_decision_fingerprint",
    "semantic_fingerprint",
    "semantic_projection",
]
