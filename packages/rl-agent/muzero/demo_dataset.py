"""Human demo imitation dataset (TASK-F3).

Loads JSONL files of recorded human plays — one decision per line — and
exposes them as an iterable of ``DemoSample`` records ready to be folded
into a training batch.  The schema is the one documented in
``docs/human-demo-format.md``:

```json
{
  "version": 1,
  "source": "human",
  "timestamp": "2026-04-29T21:00:00+08:00",
  "episode_id": "...",
  "encounter_id": "kaiser_crab_boss",
  "tier": "boss",
  "turn": 3,
  "step_in_turn": 2,
  "obs": {},
  "legal_actions": [{"action_id": "...", ...}],
  "selected_action_id": "...",
  "reason_tags": ["avoid_back_attack", "block_incoming"],
  "comment": "...",
  "outcome": {"combat_win": true, "hp_loss": 6, "turns": 5}
}
```

Validation runs at load time: the selected action MUST appear in
``legal_actions`` (otherwise the row is skipped or, with
``strict=True``, raises :class:`DemoValidationError`).  This protects
the trainer from silently learning to imitate masked / illegal actions.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable, Iterator


SUPPORTED_SCHEMA_VERSIONS = (1,)
ALLOWED_REASON_TAGS = frozenset(
    {
        "lethal",
        "block_lethal",
        "avoid_back_attack",
        "change_facing",
        "use_stun_window",
        "save_exhaust_card",
        "play_exhaust_now",
        "refund_followup",
        "avoid_refund_no_followup",
        "use_potion_now",
        "save_potion",
        "setup_next_turn",
        "cycle_control",
        "block_incoming",
    }
)


class DemoValidationError(ValueError):
    """Raised when a demo row fails schema validation in strict mode."""


@dataclass
class DemoSample:
    episode_id: str
    encounter_id: str
    tier: str
    turn: int
    step_in_turn: int
    obs: dict[str, Any]
    legal_actions: list[dict[str, Any]]
    selected_action_id: str
    selected_action_index: int
    reason_tags: list[str] = field(default_factory=list)
    comment: str = ""
    outcome: dict[str, Any] = field(default_factory=dict)


_REQUIRED_FIELDS = (
    "episode_id",
    "encounter_id",
    "obs",
    "legal_actions",
    "selected_action_id",
)


def _safe_str(value: Any, default: str = "") -> str:
    return str(value) if value is not None else default


def _validate_row(row: dict[str, Any]) -> tuple[DemoSample | None, str | None]:
    if not isinstance(row, dict):
        return None, "row is not a JSON object"
    version = row.get("version")
    if version is not None and version not in SUPPORTED_SCHEMA_VERSIONS:
        return None, f"unsupported schema version {version!r}"
    missing = [f for f in _REQUIRED_FIELDS if row.get(f) in (None, "")]
    if missing:
        return None, f"missing required fields: {missing}"
    if not isinstance(row["obs"], dict):
        return None, "'obs' must be an object"
    legal_actions = row["legal_actions"]
    if not isinstance(legal_actions, list) or not legal_actions:
        return None, "'legal_actions' must be a non-empty array"
    selected_id = _safe_str(row["selected_action_id"])
    selected_idx = -1
    for idx, action in enumerate(legal_actions):
        if not isinstance(action, dict):
            return None, f"legal_actions[{idx}] is not an object"
        aid = _safe_str(action.get("action_id"))
        if aid == selected_id:
            selected_idx = idx
            break
    if selected_idx < 0:
        return None, (
            f"selected_action_id={selected_id!r} not present in legal_actions"
        )
    reason_tags = row.get("reason_tags") or []
    if not isinstance(reason_tags, list):
        return None, "'reason_tags' must be an array if present"
    sample = DemoSample(
        episode_id=_safe_str(row["episode_id"]),
        encounter_id=_safe_str(row["encounter_id"]),
        tier=_safe_str(row.get("tier") or "normal"),
        turn=int(row.get("turn") or 0),
        step_in_turn=int(row.get("step_in_turn") or 0),
        obs=row["obs"],
        legal_actions=list(legal_actions),
        selected_action_id=selected_id,
        selected_action_index=selected_idx,
        reason_tags=[_safe_str(t) for t in reason_tags if isinstance(t, (str, bytes))],
        comment=_safe_str(row.get("comment") or ""),
        outcome=row.get("outcome") if isinstance(row.get("outcome"), dict) else {},
    )
    return sample, None


def iter_demo_jsonl(path: str | Path, *, strict: bool = False) -> Iterator[DemoSample]:
    """Yield :class:`DemoSample` records from a demo JSONL file.

    With ``strict=True`` invalid rows raise :class:`DemoValidationError`;
    otherwise they are silently skipped.  The trainer should run with
    ``strict=False`` for production but ``strict=True`` for offline
    dataset audits.
    """
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"demo dataset not found: {p}")
    with p.open("r", encoding="utf-8") as fh:
        for lineno, raw_line in enumerate(fh, start=1):
            line = raw_line.strip()
            if not line:
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as e:
                if strict:
                    raise DemoValidationError(f"line {lineno}: invalid JSON ({e})") from e
                continue
            sample, error = _validate_row(row)
            if sample is None:
                if strict:
                    raise DemoValidationError(f"line {lineno}: {error}")
                continue
            yield sample


def load_demo_dataset(
    path: str | Path,
    *,
    strict: bool = False,
    encounter_filter: Iterable[str] | None = None,
) -> list[DemoSample]:
    """Convenience wrapper that materializes :func:`iter_demo_jsonl` to a list.

    ``encounter_filter`` keeps only rows whose ``encounter_id`` is in the
    provided iterable (useful for boss-specific demo replay).
    """
    keep: set[str] | None = set(encounter_filter) if encounter_filter is not None else None
    out: list[DemoSample] = []
    for sample in iter_demo_jsonl(path, strict=strict):
        if keep is not None and sample.encounter_id not in keep:
            continue
        out.append(sample)
    return out


def build_demo_training_batch(samples: Iterable[DemoSample]) -> dict[str, Any]:
    """Build a training-batch-ready dict from a sequence of demo samples.

    Returns a dict with stable keys ``obs``, ``legal_action_ids``,
    ``selected_action_indices``, ``reason_tags`` and ``encounter_ids``.
    The trainer slices it into the policy-CE loss and an optional
    reason-tag aux head.
    """
    obs_list: list[dict[str, Any]] = []
    legal_ids: list[list[str]] = []
    selected_indices: list[int] = []
    reason_tag_lists: list[list[str]] = []
    encounter_ids: list[str] = []
    for sample in samples:
        obs_list.append(sample.obs)
        legal_ids.append([_safe_str(a.get("action_id")) for a in sample.legal_actions])
        selected_indices.append(sample.selected_action_index)
        reason_tag_lists.append(list(sample.reason_tags))
        encounter_ids.append(sample.encounter_id)
    return {
        "obs": obs_list,
        "legal_action_ids": legal_ids,
        "selected_action_indices": selected_indices,
        "reason_tags": reason_tag_lists,
        "encounter_ids": encounter_ids,
    }
