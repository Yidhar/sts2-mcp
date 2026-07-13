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

TRAJECTORY_JOURNAL_VERSION: Final = "sts2-trajectory-journal-v2"
SEMANTIC_FINGERPRINT_VERSION: Final = "sts2-semantic-decision-v1"

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
    return semantic_fingerprint(
        {"version": SEMANTIC_FINGERPRINT_VERSION, "action": action}
    )


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
        if isinstance(repeat_threshold, bool) or not isinstance(
            repeat_threshold, int
        ):
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
            decision_fingerprint=semantic_decision_fingerprint(
                observation, legal_actions
            ),
            action_fingerprint=semantic_action_fingerprint(selected_action),
        )
        self._history.append(item)
        key = (item.decision_fingerprint, item.action_fingerprint)
        matching = [
            event
            for event in self._history
            if (event.decision_fingerprint, event.action_fingerprint) == key
        ]
        if len(matching) < self.repeat_threshold:
            return None
        first = matching[0].step_index
        last = matching[-1].step_index
        deltas = [
            right.step_index - left.step_index
            for left, right in pairwise(matching)
        ]
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
    """Append-only JSONL writer used only for evaluation diagnostics."""

    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._handle: TextIO | None = None

    def __enter__(self) -> TrajectoryJournal:
        self._handle = self.path.open("a", encoding="utf-8", newline="\n")
        return self

    def __exit__(self, *_: object) -> None:
        self.close()

    def close(self) -> None:
        if self._handle is not None:
            self._handle.flush()
            self._handle.close()
            self._handle = None

    def write(self, event: Mapping[str, Any]) -> None:
        if self._handle is None:
            raise RuntimeError("trajectory journal must be opened before writing")
        payload = {
            "journal_version": TRAJECTORY_JOURNAL_VERSION,
            **diagnostic_projection(event),
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
