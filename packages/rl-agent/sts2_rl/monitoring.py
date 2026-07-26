"""Read-only telemetry aggregation for the local STS2 training dashboard.

The dashboard deliberately treats the artifact directory as an append-only
observability boundary.  It never imports checkpoints, opens pickle/torch
payloads, mutates a run, or infers liveness from stale PID files.
"""

from __future__ import annotations

import json
import math
import re
import stat
import threading
import time
from dataclasses import dataclass
from itertools import islice, pairwise
from pathlib import Path
from typing import Any

from sts2_rl.artifacts import validate_artifact_component

JsonDict = dict[str, Any]

_RUN_DIRECTORY_RE = re.compile(
    r"^run-(?P<id>[0-9a-fA-F]{8}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-" r"[0-9a-fA-F]{4}-[0-9a-fA-F]{12})$"
)
_PARENT_RUN_RE = re.compile(
    r"(?:^|[/\\])run-(?P<id>[0-9a-fA-F]{8}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-" r"[0-9a-fA-F]{4}-[0-9a-fA-F]{12})(?:[/\\]|$)"
)
_CHECKPOINT_RE = re.compile(r"^(?P<kind>periodic|final)-step-(?P<step>[0-9]+)$")
_EVALUATION_RE = re.compile(
    r"^(?P<journal_kind>evaluation|early-validation|final-audit)-" r"step-(?P<step>[0-9]{1,18})\.jsonl$"
)
_EVALUATION_GATE_KIND_BY_JOURNAL = {
    "evaluation": "validation",
    "early-validation": "early_validation",
    "final-audit": "final_audit",
}
_EVALUATION_GATE_KIND_ORDER = {
    "validation": 0,
    "early_validation": 1,
    "final_audit": 2,
}
_MAX_METRICS_LINE_BYTES = 8 * 1024 * 1024
_MAX_START_LINE_BYTES = 8 * 1024 * 1024
_MAX_EPISODES = 500
_MAX_LEARNER_POINTS = 10_000
_MAX_THROUGHPUT_POINTS = 20_000
_MAX_CANDIDATE_PEAKS = 2_000
_MAX_INCIDENTS = 200
_MAX_CHECKPOINT_EVENTS = 200
_MAX_CHECKPOINT_JSON_BYTES = 2 * 1024 * 1024
_MAX_CHECKPOINT_FILES = 512
_MAX_EVALUATION_JOURNAL_SCAN_ENTRIES = 1024
_MAX_PARSER_CACHE = 8
_DISCOVERY_TTL_SECONDS = 3.0
_CHECKPOINT_TTL_SECONDS = 4.0
_DEFAULT_STALE_SECONDS = 15.0 * 60.0

_LIFECYCLE_EVENTS = frozenset(
    {
        "run_start",
        "evaluation",
        "checkpoint",
        "run_complete",
        "interrupt",
        "run_failed",
        "circuit_breaker_open",
        "actor_shutdown_timeout",
        "backend_protocol_incident",
        "backend_restart",
        "episode_aborted",
    }
)
_INCIDENT_EVENTS = frozenset(
    {
        "interrupt",
        "run_failed",
        "circuit_breaker_open",
        "actor_shutdown_timeout",
        "backend_protocol_incident",
        "backend_restart",
        "episode_aborted",
    }
)


def _finite_number(value: object) -> float | None:
    if isinstance(value, bool) or not isinstance(value, int | float):
        return None
    number = float(value)
    return number if math.isfinite(number) else None


def _integer(value: object, default: int = 0) -> int:
    number = _finite_number(value)
    return int(number) if number is not None else default


def _mapping(value: object) -> JsonDict:
    return value if isinstance(value, dict) else {}


def _string(value: object, default: str = "") -> str:
    return value if isinstance(value, str) else default


def _bounded_append(items: list[JsonDict], item: JsonDict, limit: int) -> None:
    items.append(item)
    overflow = len(items) - limit
    if overflow > 0:
        del items[:overflow]


def _is_link_or_reparse(path: Path) -> bool:
    try:
        metadata = path.lstat()
    except OSError:
        return True
    if stat.S_ISLNK(metadata.st_mode):
        return True
    attributes = getattr(metadata, "st_file_attributes", 0)
    reparse_flag = getattr(stat, "FILE_ATTRIBUTE_REPARSE_POINT", 0)
    return bool(attributes & reparse_flag)


def _safe_resolved_child(path: Path, parent: Path) -> bool:
    try:
        resolved_parent = parent.resolve(strict=False)
        resolved_path = path.resolve(strict=False)
    except OSError:
        return False
    return resolved_path != resolved_parent and resolved_path.is_relative_to(resolved_parent)


def _read_first_object(path: Path) -> JsonDict | None:
    try:
        with path.open("rb") as handle:
            raw = handle.readline(_MAX_START_LINE_BYTES + 1)
    except OSError:
        return None
    if not raw.endswith(b"\n") or len(raw) > _MAX_START_LINE_BYTES:
        return None
    try:
        value = json.loads(raw.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError):
        return None
    if not isinstance(value, dict) or value.get("event") != "run_start":
        return None
    return value


def _read_bounded_json_object(path: Path, *, parent: Path, limit: int) -> JsonDict | None:
    """Read a small JSON object without following artifact-tree links."""

    if not path.is_file() or _is_link_or_reparse(path) or not _safe_resolved_child(path, parent):
        return None
    try:
        size = path.stat().st_size
        if size < 0 or size > limit:
            return None
        with path.open("rb") as handle:
            raw = handle.read(limit + 1)
    except OSError:
        return None
    if len(raw) > limit:
        return None
    try:
        value = json.loads(raw.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError):
        return None
    return value if isinstance(value, dict) else None


def _read_recent_objects(path: Path, *, tail_bytes: int = 256 * 1024) -> list[JsonDict]:
    try:
        size = path.stat().st_size
        with path.open("rb") as handle:
            offset = max(0, size - tail_bytes)
            handle.seek(offset)
            payload = handle.read()
    except OSError:
        return []
    if offset > 0:
        first_newline = payload.find(b"\n")
        payload = payload[first_newline + 1 :] if first_newline >= 0 else b""
    if payload and not payload.endswith(b"\n"):
        final_newline = payload.rfind(b"\n")
        payload = payload[: final_newline + 1] if final_newline >= 0 else b""
    result: list[JsonDict] = []
    for raw_line in payload.splitlines():
        if not raw_line or len(raw_line) > _MAX_METRICS_LINE_BYTES:
            continue
        try:
            value = json.loads(raw_line.decode("utf-8"))
        except (UnicodeDecodeError, json.JSONDecodeError):
            continue
        if isinstance(value, dict):
            result.append(value)
    return result


def _project_episode(event: JsonDict, episode_number: int) -> JsonDict:
    return {
        "timestamp": _finite_number(event.get("unix_s")),
        "environment_steps": _integer(event.get("environment_steps")),
        "episode": episode_number,
        "episode_id": event.get("episode_id"),
        "seed": event.get("reset_seed"),
        "max_act": event.get("max_act"),
        "max_floor": event.get("max_floor"),
        "revivals": event.get("revivals_used"),
        "player_hp_lost": event.get("player_hp_lost"),
        "steps": event.get("steps"),
        "reward": event.get("reward_total"),
        "outcome": "run_victory" if event.get("run_won") is True else event.get("terminal_reason"),
        "termination_reason": event.get("terminal_reason"),
        "run_won": event.get("run_won"),
        "act1_cleared": event.get("act1_cleared"),
        "deadlock": event.get("deadlocked"),
        "combat_progress_stalled": event.get("combat_progress_stalled"),
        "noncombat_progress_stalled": event.get("noncombat_progress_stalled"),
        "combat_policy_failed": event.get("combat_policy_failed"),
        "truncated": event.get("truncated"),
        "maximum_observed_candidates": event.get("maximum_observed_candidates"),
        "maximum_observed_semantic_candidates": event.get("maximum_observed_semantic_candidates"),
        "maximum_equivalence_class_size": event.get("maximum_equivalence_class_size"),
        "act_revival_counts": event.get("act_revival_counts"),
        "act_hp_loss_counts": event.get("act_hp_loss_counts"),
    }


def _project_learner(event: JsonDict) -> tuple[JsonDict, JsonDict]:
    queue = _mapping(event.get("rollout_queue"))
    timings = _mapping(event.get("timings"))
    episodic_replay = _mapping(event.get("episodic_replay"))
    transaction_replay = _mapping(event.get("transaction_replay"))
    point = {
        "timestamp": _finite_number(event.get("unix_s")),
        "environment_steps": event.get("environment_steps"),
        "learner_updates": event.get("policy_version"),
        "policy_version": event.get("policy_version"),
        "loss": event.get("loss"),
        "policy_loss": event.get("policy_loss"),
        "value_loss": event.get("value_loss"),
        "episodic_loss": event.get("episodic_loss"),
        "gradient_norm": event.get("gradient_norm"),
        "entropy": event.get("entropy"),
        "maximum_policy_lag": event.get("maximum_policy_lag"),
        "queue_depth": queue.get("size"),
        "queue_capacity": queue.get("capacity"),
        "total_ms": timings.get("total_ms"),
    }
    latest = {
        **point,
        "grad_norm": event.get("gradient_norm"),
        "importance_ratio_mean": event.get("importance_ratio_mean"),
        "importance_clip_fraction": event.get("importance_clip_fraction"),
        "episodic_replay_size": episodic_replay.get("size"),
        "episodic_replay_bytes": episodic_replay.get("storage_nbytes"),
        "transaction_replay_size": transaction_replay.get("size"),
        "transaction_replay_bytes": transaction_replay.get("storage_nbytes"),
        "producer_wait_seconds": queue.get("producer_wait_seconds"),
        "consumer_wait_seconds": queue.get("consumer_wait_seconds"),
    }
    return point, latest


def _project_actor(event: JsonDict) -> JsonDict:
    actor = _mapping(event.get("actor_progress"))
    queue = _mapping(event.get("rollout_queue"))
    result: JsonDict = {
        "timestamp": _finite_number(event.get("unix_s")),
        "environment_steps": event.get("environment_steps"),
        "phase": actor.get("phase"),
        "decision_domain": actor.get("decision_domain"),
        "episode_id": actor.get("episode_id"),
        "seed": actor.get("reset_seed"),
        "max_act": actor.get("max_act"),
        "max_floor": actor.get("max_floor"),
        "revivals": actor.get("revivals_used"),
        "player_hp_lost": actor.get("player_hp_lost"),
        "steps": actor.get("steps"),
        "policy_decisions": actor.get("policy_decisions"),
        "forced_decisions": actor.get("forced_decisions"),
        "queue_depth": queue.get("size"),
        "queue_capacity": queue.get("capacity"),
        "maximum_observed_candidates": actor.get("maximum_observed_candidates"),
        "maximum_observed_semantic_candidates": actor.get("maximum_observed_semantic_candidates"),
        "maximum_equivalence_class_size": actor.get("maximum_equivalence_class_size"),
        "hand_cards": actor.get("hand_cards"),
        "draw_cards": actor.get("draw_cards"),
        "discard_cards": actor.get("discard_cards"),
        "exhaust_cards": actor.get("exhaust_cards"),
        "enemy_hp_total": actor.get("enemy_hp_total"),
        "enemy_max_hp_total": actor.get("enemy_max_hp_total"),
        "combat_no_net_progress_steps": actor.get("combat_no_net_progress_steps"),
        "noncombat_no_durable_progress_steps": actor.get("noncombat_no_durable_progress_steps"),
        "last_selected_action_kind": actor.get("last_selected_action_kind"),
        "legal_action_kinds": actor.get("legal_action_kinds"),
    }
    return result


def _project_incident(event: JsonDict) -> JsonDict:
    event_name = _string(event.get("event"), "unknown")
    message = event.get("message") or event.get("reason") or event.get("error")
    return {
        "timestamp": _finite_number(event.get("unix_s")),
        "event": event_name,
        "title": event_name.replace("_", " "),
        "message": message if isinstance(message, str) else event_name.replace("_", " "),
        "environment_steps": event.get("environment_steps"),
        "episode_id": event.get("episode_id"),
        "fingerprint": event.get("fingerprint"),
    }


class IncrementalMetrics:
    """Incrementally consume complete JSONL records from one metrics file."""

    def __init__(self, path: Path) -> None:
        self.path = path
        self._identity: tuple[int, int] | None = None
        self.offset = 0
        self.partial_line = False
        self.malformed_rows = 0
        self.oversize_rows = 0
        self.parsed_rows = 0
        self.event_counts: dict[str, int] = {}
        self.start_event: JsonDict | None = None
        self.latest_by_event: dict[str, JsonDict] = {}
        self.episodes: list[JsonDict] = []
        self.learner_series: list[JsonDict] = []
        self.evaluations: list[JsonDict] = []
        self.checkpoint_events: list[JsonDict] = []
        self.incidents: list[JsonDict] = []
        self.candidate_peaks: list[JsonDict] = []
        self.latest_learner: JsonDict | None = None
        self.latest_actor: JsonDict | None = None
        self.environment_points: list[tuple[float, int]] = []
        self.learner_points: list[tuple[float, int]] = []
        self.last_event_unix_s: float | None = None
        self.last_mtime: float | None = None

    def _reset(self) -> None:
        self.offset = 0
        self.partial_line = False
        self.malformed_rows = 0
        self.oversize_rows = 0
        self.parsed_rows = 0
        self.event_counts.clear()
        self.start_event = None
        self.latest_by_event.clear()
        self.episodes.clear()
        self.learner_series.clear()
        self.evaluations.clear()
        self.checkpoint_events.clear()
        self.incidents.clear()
        self.candidate_peaks.clear()
        self.latest_learner = None
        self.latest_actor = None
        self.environment_points.clear()
        self.learner_points.clear()
        self.last_event_unix_s = None

    def refresh(self) -> None:
        try:
            metadata = self.path.stat()
        except OSError:
            return
        # ctime changes on ordinary appends on POSIX, so it is not part of the
        # file identity.  Atomic replacement changes the inode/file index.
        identity = (metadata.st_dev, metadata.st_ino)
        if self._identity is None:
            self._identity = identity
        elif identity != self._identity or metadata.st_size < self.offset:
            self._identity = identity
            self._reset()
        self.last_mtime = metadata.st_mtime
        try:
            with self.path.open("rb") as handle:
                handle.seek(self.offset)
                payload = handle.read()
        except OSError:
            return
        if not payload:
            self.partial_line = False
            return
        final_newline = payload.rfind(b"\n")
        if final_newline < 0:
            self.partial_line = True
            return
        complete = payload[: final_newline + 1]
        self.partial_line = final_newline != len(payload) - 1
        self.offset += len(complete)
        for raw_line in complete.splitlines():
            if not raw_line:
                continue
            if len(raw_line) > _MAX_METRICS_LINE_BYTES:
                self.oversize_rows += 1
                continue
            try:
                value = json.loads(raw_line.decode("utf-8"))
            except (UnicodeDecodeError, json.JSONDecodeError):
                self.malformed_rows += 1
                continue
            if not isinstance(value, dict):
                self.malformed_rows += 1
                continue
            self.parsed_rows += 1
            self._consume(value)

    def _consume(self, event: JsonDict) -> None:
        event_name = _string(event.get("event"), "unknown")
        self.event_counts[event_name] = self.event_counts.get(event_name, 0) + 1
        self.latest_by_event[event_name] = event
        unix_s = _finite_number(event.get("unix_s"))
        if unix_s is not None:
            if self.last_event_unix_s is None or unix_s > self.last_event_unix_s:
                self.last_event_unix_s = unix_s
            if event_name in {
                "learner_progress",
                "learner_update_start",
                "learner_update",
                "train_episode",
            }:
                environment_steps = _finite_number(event.get("environment_steps"))
                if environment_steps is not None:
                    self.environment_points.append((unix_s, int(environment_steps)))
                    overflow = len(self.environment_points) - _MAX_THROUGHPUT_POINTS
                    if overflow > 0:
                        del self.environment_points[:overflow]

        if event_name == "run_start":
            if self.start_event is None:
                self.start_event = event
                state = _mapping(event.get("state"))
                start_time = unix_s
                start_step = _integer(state.get("environment_steps"))
                if start_time is not None:
                    self.environment_points.append((start_time, start_step))
            return

        if event_name == "train_episode":
            baseline = _integer(_mapping(self.start_event.get("state") if self.start_event else {}).get("episodes"))
            episode_number = baseline + self.event_counts[event_name]
            projected = _project_episode(event, episode_number)
            _bounded_append(self.episodes, projected, _MAX_EPISODES)
            previous = self.candidate_peaks[-1] if self.candidate_peaks else {}
            raw_peak = max(
                _integer(previous.get("maximum_observed_candidates")),
                _integer(projected.get("maximum_observed_candidates")),
            )
            semantic_peak = max(
                _integer(previous.get("maximum_observed_semantic_candidates")),
                _integer(projected.get("maximum_observed_semantic_candidates")),
            )
            equivalence_peak = max(
                _integer(previous.get("maximum_equivalence_class_size")),
                _integer(projected.get("maximum_equivalence_class_size")),
            )
            if (
                not self.candidate_peaks
                or raw_peak > _integer(previous.get("maximum_observed_candidates"))
                or semantic_peak > _integer(previous.get("maximum_observed_semantic_candidates"))
                or equivalence_peak > _integer(previous.get("maximum_equivalence_class_size"))
            ):
                _bounded_append(
                    self.candidate_peaks,
                    {
                        "episode": episode_number,
                        "timestamp": unix_s,
                        "environment_steps": _integer(event.get("environment_steps")),
                        "maximum_observed_candidates": raw_peak,
                        "maximum_observed_semantic_candidates": semantic_peak,
                        "maximum_equivalence_class_size": equivalence_peak,
                    },
                    _MAX_CANDIDATE_PEAKS,
                )
            return

        if event_name == "learner_update":
            point, latest = _project_learner(event)
            _bounded_append(self.learner_series, point, _MAX_LEARNER_POINTS)
            self.latest_learner = latest
            self.latest_actor = _project_actor(event)
            if unix_s is not None:
                self.learner_points.append((unix_s, _integer(event.get("policy_version"))))
                if len(self.learner_points) > 200:
                    del self.learner_points[: len(self.learner_points) - 200]
            return

        if event_name == "evaluation":
            self.evaluations.append(event)
            return

        if event_name == "checkpoint":
            _bounded_append(self.checkpoint_events, event, _MAX_CHECKPOINT_EVENTS)
            return

        if event_name in _LIFECYCLE_EVENTS:
            projected_incident = _project_incident(event)
            if event_name in _INCIDENT_EVENTS:
                _bounded_append(self.incidents, projected_incident, _MAX_INCIDENTS)


@dataclass(frozen=True)
class DiscoveredRun:
    key: str
    lineage: str
    run_id: str
    run_directory: Path
    metrics_path: Path
    start_event: JsonDict
    start_time: float


def _parent_run_id(start_event: JsonDict) -> str | None:
    checkpoint_load = _mapping(start_event.get("checkpoint_load"))
    if checkpoint_load.get("mode") != "exact_resume":
        return None
    parent = checkpoint_load.get("parent_checkpoint")
    if not isinstance(parent, str):
        return None
    match = _PARENT_RUN_RE.search(parent)
    return match.group("id").lower() if match is not None else None


def _terminal_projection(event: JsonDict | None) -> JsonDict:
    return event if event is not None else {}


def _run_completion_status(event: JsonDict) -> JsonDict:
    """Project a persisted run terminus without conflating its outcome.

    ``run_complete`` means the runtime finished its shutdown/checkpoint
    protocol.  It does not necessarily mean that collection reached the
    configured horizon: an evaluation liveness guard deliberately emits the
    same terminal event after stopping a lineage early.  Keep legacy events
    without ``completion_status`` classified as completed, but surface the
    explicit guard outcome distinctly wherever a run status is rendered.
    """

    completion_status = _string(event.get("completion_status"))
    if completion_status == "evaluation_guard_stopped":
        return {
            "state": "guard_stopped",
            "phase": "evaluation_guard_stopped",
            "label": "评估门禁提前停止",
            "evidence": [
                "run_complete 已持久化; completion_status=evaluation_guard_stopped",
                "评估门禁请求提前停止, 这不代表训练采集目标已经完成",
            ],
        }
    return {
        "state": "completed",
        "phase": "complete",
        "label": "已完成",
        "evidence": ["run_complete 已持久化"],
    }


def _pending_evaluation_status(
    pending_evaluation: JsonDict,
    *,
    telemetry_age: float | None = None,
) -> JsonDict:
    evaluation_kind = _string(pending_evaluation.get("kind"), "validation")
    evaluation_gate = _integer(pending_evaluation.get("gate"))
    if evaluation_kind == "final_audit":
        label = f"最终审计中 · 门 {evaluation_gate}"
        evidence = [
            "最终审计 journal 活跃且尚无对应 evaluation 汇总",
            "run_complete 尚未持久化; 整个运行尚未完成",
        ]
    elif evaluation_kind == "early_validation":
        label = f"早期评估中 · 门 {evaluation_gate}"
        evidence = ["早期评估 journal 活跃且尚无对应 evaluation 汇总"]
    else:
        label = f"评估中 · 门 {evaluation_gate}"
        evidence = ["评估 journal 活跃且尚无对应 evaluation 汇总"]
    projected = {
        "state": "evaluating",
        "phase": "evaluating",
        "label": label,
        "evaluation_kind": evaluation_kind,
        "evaluation_gate": evaluation_gate,
        "evidence": evidence,
    }
    if telemetry_age is not None:
        projected["telemetry_age_s"] = telemetry_age
    return projected


def _latest_counter(
    aggregate: IncrementalMetrics,
    *,
    field: str,
    start_field: str | None = None,
    episode_increment: bool = False,
) -> int:
    completed = aggregate.latest_by_event.get("run_complete")
    if completed is not None and _finite_number(completed.get(field)) is not None:
        return _integer(completed.get(field))
    candidates: list[int] = []
    state = _mapping(aggregate.start_event.get("state") if aggregate.start_event else {})
    candidates.append(_integer(state.get(start_field or field)))
    for event_name in ("checkpoint", "learner_update", "train_episode", "learner_update_start"):
        event = aggregate.latest_by_event.get(event_name)
        if event is not None and _finite_number(event.get(field)) is not None:
            candidates.append(_integer(event.get(field)))
    if episode_increment:
        candidates.append(_integer(state.get(start_field or field)) + aggregate.event_counts.get("train_episode", 0))
    return max(candidates, default=0)


def _rate(
    points: list[tuple[float, int]],
    seconds: float,
    *,
    reference_time: float | None = None,
) -> tuple[float | None, str]:
    if len(points) < 3:
        return None, "low"
    end_time = points[-1][0] if reference_time is None else reference_time
    selected = [point for point in points if end_time - seconds <= point[0] <= end_time + 5.0]
    distinct: list[tuple[float, int]] = []
    for timestamp, step in selected:
        if distinct and step == distinct[-1][1]:
            continue
        if distinct and (timestamp <= distinct[-1][0] or step < distinct[-1][1]):
            continue
        distinct.append((timestamp, step))
    if len(distinct) < 3:
        return None, "low"
    elapsed = distinct[-1][0] - distinct[0][0]
    if elapsed < min(120.0, seconds / 2.0):
        return None, "low"
    delta = distinct[-1][1] - distinct[0][1]
    if delta <= 0:
        return None, "low"
    confidence = "high" if elapsed >= seconds * 0.8 and len(distinct) >= 6 else "medium"
    return delta / elapsed, confidence


def _recent_update_rate(points: list[tuple[float, int]]) -> float | None:
    if len(points) < 3:
        return None
    first_time, first_update = points[max(0, len(points) - 100)]
    last_time, last_update = points[-1]
    elapsed = last_time - first_time
    if elapsed <= 0 or last_update <= first_update:
        return None
    return (last_update - first_update) / elapsed


def _downsample_extrema(points: list[JsonDict], limit: int, key: str) -> list[JsonDict]:
    if len(points) <= limit:
        return list(points)
    if limit < 4:
        return [points[0], points[-1]]
    interior = points[1:-1]
    bucket_count = max(1, (limit - 2) // 2)
    bucket_size = len(interior) / bucket_count
    selected: list[tuple[int, JsonDict]] = [(0, points[0])]
    for bucket in range(bucket_count):
        begin = int(bucket * bucket_size)
        end = max(begin + 1, int((bucket + 1) * bucket_size))
        candidates = list(enumerate(interior[begin:end], start=begin + 1))
        numeric: list[tuple[int, JsonDict, float]] = []
        for index, item in candidates:
            value = _finite_number(item.get(key))
            if value is not None:
                numeric.append((index, item, value))
        if not numeric:
            if candidates:
                selected.append(candidates[-1])
            continue
        low = min(numeric, key=lambda item: item[2])
        high = max(numeric, key=lambda item: item[2])
        for index, item, _ in sorted({low[0]: low, high[0]: high}.values(), key=lambda value: value[0]):
            selected.append((index, item))
    selected.append((len(points) - 1, points[-1]))
    selected.sort(key=lambda item: item[0])
    return [item for _, item in selected[:limit]]


def _evaluation_gate_kind(event: JsonDict) -> str:
    supplied = _string(event.get("gate_kind")).replace("-", "_")
    if supplied in {"validation", "early_validation", "final_audit"}:
        return supplied
    if _string(event.get("data_partition")).replace("-", "_") == "final_audit":
        return "final_audit"
    # Legacy evaluation summaries predate gate_kind. Those journals used the
    # ordinary evaluation-step-* namespace, so validation is the safe fallback.
    return "validation"


def _normalise_evaluation(event: JsonDict) -> JsonDict:
    episodes = _integer(event.get("episodes"))
    gate_value = event.get("evaluation_gate")
    gate = _integer(gate_value if _finite_number(gate_value) is not None else event.get("environment_steps"))
    gate_kind = _evaluation_gate_kind(event)

    def fraction(name: str) -> float:
        value = _finite_number(event.get(name))
        return min(1.0, max(0.0, value)) if value is not None else 0.0

    act1_rate = fraction("act1_clear_rate")
    act3_rate = fraction("act3_reach_rate")
    run_rate = fraction("run_win_rate")
    zero_rate = fraction("revival_free_act1_clear_rate")
    low_rate = fraction("act1_clear_at_most_one_revival_rate")
    deadlock_rate = fraction("deadlock_rate")
    combat_stall_rate = fraction("combat_progress_stall_rate")
    noncombat_stall_rate = fraction("noncombat_progress_stall_rate")
    policy_failures = _integer(event.get("combat_policy_failure_count"))
    evaluation_context = _mapping(event.get("evaluation_context"))
    policy_number = _finite_number(evaluation_context.get("policy_version"))
    if policy_number is None:
        policy_number = _finite_number(event.get("policy_version"))
    actor_policy_number = _finite_number(evaluation_context.get("actor_policy_version"))
    if actor_policy_number is None:
        actor_policy_number = _finite_number(event.get("actor_policy_version"))
    return {
        "timestamp": _finite_number(event.get("unix_s")),
        "evaluation_gate": gate,
        "gate_kind": gate_kind,
        "data_partition": _string(event.get("data_partition"))
        or ("final_audit" if gate_kind == "final_audit" else "validation"),
        "environment_steps": _integer(event.get("environment_steps")),
        "policy_version": int(policy_number) if policy_number is not None else None,
        "actor_policy_version": (int(actor_policy_number) if actor_policy_number is not None else None),
        "policy_model_state_sha256": _string(evaluation_context.get("policy_model_state_sha256")),
        "episodes": episodes,
        "act_1_successes": _integer(event.get("act1_clear_count"), round(act1_rate * episodes)),
        "act_1_success_rate": act1_rate,
        "act_3_successes": _integer(event.get("act3_reach_count"), round(act3_rate * episodes)),
        "act_3_success_rate": act3_rate,
        "run_wins": _integer(event.get("successful_run_count"), round(run_rate * episodes)),
        "run_win_rate": run_rate,
        "zero_revival_act1_successes": round(zero_rate * episodes),
        "at_most_one_revival_act1_successes": round(low_rate * episodes),
        "mean_max_floor": event.get("mean_max_floor"),
        "max_floor": event.get("maximum_floor"),
        "mean_revivals": event.get("mean_revivals_used"),
        "mean_player_hp_lost": event.get("mean_player_hp_lost"),
        "act_1_boundary_count": event.get("act1_boundary_count"),
        "act_1_boundary_mean_revivals": event.get("act1_boundary_mean_revivals"),
        "act_1_boundary_mean_player_hp_lost": event.get("act1_boundary_mean_hp_lost"),
        "deadlock_count": round(deadlock_rate * episodes),
        "deadlock_rate": deadlock_rate,
        # These are deliberately separate. One episode can satisfy multiple
        # failure predicates, so adding them would manufacture a false union.
        "combat_progress_stall_count": round(combat_stall_rate * episodes),
        "noncombat_progress_stall_count": round(noncombat_stall_rate * episodes),
        "combat_policy_failure_count": policy_failures,
        "combat_progress_stall_rate": combat_stall_rate,
        "noncombat_progress_stall_rate": noncombat_stall_rate,
        "infrastructure_retries": _integer(event.get("infrastructure_retries")),
        "maximum_observed_candidates": event.get("maximum_observed_candidates"),
        "run_maximum_observed_candidates": event.get("run_maximum_observed_candidates"),
    }


def _resume_boundary(aggregate: IncrementalMetrics, successor: DiscoveredRun | None) -> JsonDict | None:
    if successor is None:
        return None
    state = _mapping(successor.start_event.get("state"))
    environment_steps = _integer(state.get("environment_steps"))
    policy_version = _integer(state.get("policy_version"))
    cutoff_unix_s: float | None = None
    for checkpoint in aggregate.checkpoint_events:
        if (
            _integer(checkpoint.get("environment_steps")) == environment_steps
            and _integer(checkpoint.get("policy_version")) == policy_version
        ):
            candidate = _finite_number(checkpoint.get("unix_s"))
            if candidate is not None and (cutoff_unix_s is None or candidate > cutoff_unix_s):
                cutoff_unix_s = candidate
    return {
        "environment_steps": environment_steps,
        "episodes": _integer(state.get("episodes")),
        "evaluation_episodes": _integer(state.get("evaluation_episodes")),
        "policy_version": policy_version,
        "cutoff_unix_s": cutoff_unix_s,
    }


def _bounded_episodes(aggregate: IncrementalMetrics, boundary: JsonDict | None) -> list[JsonDict]:
    if boundary is None:
        return list(aggregate.episodes)
    maximum_episode = _integer(boundary.get("episodes"))
    return [episode for episode in aggregate.episodes if _integer(episode.get("episode")) <= maximum_episode]


def _bounded_learner_series(aggregate: IncrementalMetrics, boundary: JsonDict | None) -> list[JsonDict]:
    if boundary is None:
        return list(aggregate.learner_series)
    maximum_policy = _integer(boundary.get("policy_version"))
    return [point for point in aggregate.learner_series if _integer(point.get("policy_version")) <= maximum_policy]


def _bounded_evaluations(aggregate: IncrementalMetrics, boundary: JsonDict | None) -> list[JsonDict]:
    if boundary is None:
        return list(aggregate.evaluations)
    start_count = _integer(
        _mapping(aggregate.start_event.get("state") if aggregate.start_event else {}).get("evaluation_episodes")
    )
    maximum_count = _integer(boundary.get("evaluation_episodes"))
    result: list[JsonDict] = []
    consumed = start_count
    for event in aggregate.evaluations:
        next_count = consumed + _integer(event.get("episodes"))
        if next_count > maximum_count:
            break
        result.append(event)
        consumed = next_count
    return result


def _bounded_incidents(aggregate: IncrementalMetrics, boundary: JsonDict | None) -> list[JsonDict]:
    if boundary is None:
        return list(aggregate.incidents)
    cutoff = _finite_number(boundary.get("cutoff_unix_s"))
    if cutoff is None:
        return list(aggregate.incidents)
    return [
        incident
        for incident in aggregate.incidents
        if (_finite_number(incident.get("timestamp")) or float("-inf")) <= cutoff
    ]


def _bounded_candidate_peaks(
    aggregate: IncrementalMetrics,
    boundary: JsonDict | None,
) -> list[JsonDict]:
    if boundary is None:
        return list(aggregate.candidate_peaks)
    maximum_episode = _integer(boundary.get("episodes"))
    return [point for point in aggregate.candidate_peaks if _integer(point.get("episode")) <= maximum_episode]


class DashboardStore:
    """Discover runs and construct bounded, read-only dashboard snapshots."""

    def __init__(
        self,
        artifact_root: Path,
        *,
        stale_seconds: float = _DEFAULT_STALE_SECONDS,
        now: Any = time.time,
    ) -> None:
        self.artifact_root = artifact_root.resolve(strict=False)
        self.runs_root = (self.artifact_root / "runs").resolve(strict=False)
        self.stale_seconds = stale_seconds
        self._now = now
        self._lock = threading.RLock()
        self._runs: dict[str, DiscoveredRun] = {}
        self._last_discovery_monotonic = 0.0
        self._parsers: dict[Path, IncrementalMetrics] = {}
        self._checkpoint_cache: dict[Path, tuple[float, JsonDict]] = {}

    def discover(self, *, force: bool = False) -> list[DiscoveredRun]:
        with self._lock:
            monotonic_now = time.monotonic()
            if not force and monotonic_now - self._last_discovery_monotonic < _DISCOVERY_TTL_SECONDS:
                return sorted(self._runs.values(), key=lambda run: run.start_time, reverse=True)
            discovered: dict[str, DiscoveredRun] = {}
            if self.runs_root.is_dir() and not _is_link_or_reparse(self.runs_root):
                try:
                    lineage_directories = list(self.runs_root.iterdir())
                except OSError:
                    lineage_directories = []
                for lineage_directory in lineage_directories:
                    if not lineage_directory.is_dir() or _is_link_or_reparse(lineage_directory):
                        continue
                    try:
                        lineage = validate_artifact_component(lineage_directory.name, label="training lineage")
                    except ValueError:
                        continue
                    if not _safe_resolved_child(lineage_directory, self.runs_root):
                        continue
                    try:
                        run_directories = list(lineage_directory.iterdir())
                    except OSError:
                        continue
                    for run_directory in run_directories:
                        match = _RUN_DIRECTORY_RE.fullmatch(run_directory.name)
                        if match is None or not run_directory.is_dir() or _is_link_or_reparse(run_directory):
                            continue
                        if not _safe_resolved_child(run_directory, lineage_directory):
                            continue
                        metrics_path = run_directory / "metrics.jsonl"
                        if (
                            not metrics_path.is_file()
                            or _is_link_or_reparse(metrics_path)
                            or not _safe_resolved_child(metrics_path, run_directory)
                        ):
                            continue
                        start_event = _read_first_object(metrics_path)
                        if start_event is None:
                            continue
                        run_id = match.group("id").lower()
                        payload_run_id = _string(start_event.get("run_id")).lower()
                        if payload_run_id and payload_run_id != run_id:
                            continue
                        start_time = _finite_number(start_event.get("unix_s"))
                        if start_time is None:
                            continue
                        key = f"{lineage}/{run_directory.name}"
                        discovered[key] = DiscoveredRun(
                            key=key,
                            lineage=lineage,
                            run_id=run_id,
                            run_directory=run_directory,
                            metrics_path=metrics_path,
                            start_event=start_event,
                            start_time=start_time,
                        )
            self._runs = discovered
            self._last_discovery_monotonic = monotonic_now
            known_paths = {run.metrics_path for run in discovered.values()}
            for path in list(self._parsers):
                if path not in known_paths:
                    del self._parsers[path]
            return sorted(discovered.values(), key=lambda run: run.start_time, reverse=True)

    def _parser(self, run: DiscoveredRun) -> IncrementalMetrics:
        parser = self._parsers.pop(run.metrics_path, None)
        if parser is None:
            parser = IncrementalMetrics(run.metrics_path)
        self._parsers[run.metrics_path] = parser
        while len(self._parsers) > _MAX_PARSER_CACHE:
            oldest_path = next(iter(self._parsers))
            if oldest_path == run.metrics_path:
                break
            del self._parsers[oldest_path]
        parser.refresh()
        return parser

    def _chain(self, selected: DiscoveredRun) -> list[DiscoveredRun]:
        by_id = {run.run_id: run for run in self._runs.values()}
        chain = [selected]
        visited = {selected.run_id}
        cursor = selected
        while True:
            parent_id = _parent_run_id(cursor.start_event)
            if parent_id is None or parent_id in visited:
                break
            parent = by_id.get(parent_id)
            if parent is None or parent.start_time > cursor.start_time:
                break
            chain.append(parent)
            visited.add(parent_id)
            cursor = parent
        chain.reverse()
        return chain

    def _successor_for(self, run: DiscoveredRun) -> DiscoveredRun | None:
        candidates = [
            candidate
            for candidate in self._runs.values()
            if candidate.start_time >= run.start_time
            and candidate.run_id != run.run_id
            and _parent_run_id(candidate.start_event) == run.run_id
        ]
        if not candidates:
            return None
        return max(candidates, key=lambda candidate: candidate.start_time)

    def _quick_status(self, run: DiscoveredRun) -> JsonDict:
        parser = self._parsers.get(run.metrics_path)
        if parser is not None and parser.parsed_rows:
            parser.refresh()
            evaluation_activity, pending_evaluation = self._evaluation_activity(run, parser)
            state = self._status(
                run,
                parser,
                {
                    "published": [],
                    "publishing": [],
                    "latest_activity": evaluation_activity,
                },
                pending_evaluation,
            )
            return {"state": state["state"], "phase": state["phase"], "label": state["label"]}
        try:
            modified = run.metrics_path.stat().st_mtime
        except OSError:
            return {"state": "unknown", "label": "不可读取"}
        recent_objects = _read_recent_objects(run.metrics_path)
        recent_events = {_string(event.get("event")) for event in recent_objects}
        recent_completion = next(
            (event for event in reversed(recent_objects) if event.get("event") == "run_complete"),
            None,
        )
        if recent_completion is not None:
            status = _run_completion_status(recent_completion)
            return {
                "state": status["state"],
                "phase": status["phase"],
                "label": status["label"],
            }
        successor = self._successor_for(run)
        if successor is not None:
            return {
                "state": "continued",
                "phase": "continued",
                "label": "已续跑",
            }
        if "interrupt" in recent_events:
            return {"state": "interrupted", "phase": "interrupted", "label": "已中断"}
        if "run_failed" in recent_events or "circuit_breaker_open" in recent_events:
            return {"state": "failed", "phase": "failed", "label": "运行失败"}
        completed_gates = {
            (
                _evaluation_gate_kind(event),
                _integer(
                    event.get("evaluation_gate")
                    if _finite_number(event.get("evaluation_gate")) is not None
                    else event.get("environment_steps")
                ),
            )
            for event in recent_objects
            if event.get("event") == "evaluation"
        }
        evaluation_activity, pending_evaluation = self._evaluation_journal_activity(
            run,
            completed_gates=completed_gates,
            metrics_mtime=modified,
        )
        latest_activity = max(
            modified,
            evaluation_activity if evaluation_activity is not None else modified,
        )
        age = max(0.0, self._now() - latest_activity)
        if pending_evaluation is not None and age <= self.stale_seconds:
            state = _pending_evaluation_status(pending_evaluation)
            return {"state": state["state"], "phase": state["phase"], "label": state["label"]}
        return {
            "state": "stale" if age > self.stale_seconds else "running",
            "label": "遥测陈旧" if age > self.stale_seconds else "可用",
        }

    def list_runs(self) -> JsonDict:
        with self._lock:
            runs = self.discover()
            payload = [
                {
                    "key": run.key,
                    "lineage": run.lineage,
                    "run_id": run.run_id,
                    "start_time": run.start_time,
                    "status": self._quick_status(run),
                    "checkpoint_load_mode": _mapping(run.start_event.get("checkpoint_load")).get("mode"),
                }
                for run in runs
            ]
            return {"runs": payload, "selected": runs[0].key if runs else None}

    def _checkpoint_directory(self, run: DiscoveredRun) -> Path | None:
        config = _mapping(run.start_event.get("config"))
        runtime = _mapping(config.get("runtime"))
        configured = runtime.get("checkpoint_dir")
        if not isinstance(configured, str) or not configured.strip():
            return None
        relative = Path(configured)
        if relative.is_absolute() or relative.anchor or any(part in {"", ".", ".."} for part in relative.parts):
            return None
        candidate = (self.artifact_root / relative / f"run-{run.run_id}").resolve(strict=False)
        if candidate == self.artifact_root or not candidate.is_relative_to(self.artifact_root):
            return None
        if candidate.exists() and _is_link_or_reparse(candidate):
            return None
        return candidate

    def _checkpoint_snapshot(self, run: DiscoveredRun) -> JsonDict:
        directory = self._checkpoint_directory(run)
        if directory is None or not directory.is_dir():
            return {"published": [], "publishing": [], "latest_activity": None}
        cached = self._checkpoint_cache.get(directory)
        monotonic_now = time.monotonic()
        if cached is not None and monotonic_now - cached[0] < _CHECKPOINT_TTL_SECONDS:
            return cached[1]
        published: list[JsonDict] = []
        publishing: list[JsonDict] = []
        latest_activity: float | None = None
        try:
            entries = list(directory.iterdir())
        except OSError:
            entries = []
        for entry in entries:
            if not entry.is_dir() or _is_link_or_reparse(entry) or not _safe_resolved_child(entry, directory):
                continue
            try:
                modified = entry.stat().st_mtime
            except OSError:
                continue
            latest_activity = modified if latest_activity is None else max(latest_activity, modified)
            if entry.name.startswith(".") and ".incomplete-" in entry.name:
                child_mtime = modified
                try:
                    for child in entry.iterdir():
                        child_mtime = max(child_mtime, child.stat().st_mtime)
                except OSError:
                    pass
                latest_activity = max(latest_activity, child_mtime)
                target_name = entry.name[1:].split(".incomplete-", 1)[0]
                target_match = _CHECKPOINT_RE.fullmatch(target_name)
                publishing.append(
                    {
                        "name": target_name,
                        "environment_steps": (int(target_match.group("step")) if target_match is not None else None),
                        "modified_at": child_mtime,
                        "status": "publishing",
                    }
                )
                continue
            match = _CHECKPOINT_RE.fullmatch(entry.name)
            if match is None:
                continue
            manifest_path = entry / "checkpoint.manifest.json"
            metadata_path = entry / "metadata.json"
            issues: list[str] = []
            manifest_value = _read_bounded_json_object(
                manifest_path,
                parent=entry,
                limit=_MAX_CHECKPOINT_JSON_BYTES,
            )
            metadata_value = _read_bounded_json_object(
                metadata_path,
                parent=entry,
                limit=_MAX_CHECKPOINT_JSON_BYTES,
            )
            if manifest_value is None:
                issues.append("manifest_unreadable")
                manifest: JsonDict = {}
            else:
                manifest = manifest_value
            if metadata_value is None:
                issues.append("metadata_unreadable")
                metadata: JsonDict = {}
            else:
                metadata = metadata_value
            manifest_id = manifest.get("checkpoint_id")
            metadata_id = metadata.get("checkpoint_id")
            if manifest.get("format") != "sts2-atomic-checkpoint-v1":
                issues.append("unsupported_checkpoint_format")
            if not isinstance(manifest_id, str) or not manifest_id:
                issues.append("manifest_checkpoint_id_missing")
            if not isinstance(metadata_id, str) or not metadata_id:
                issues.append("metadata_checkpoint_id_missing")
            if (
                isinstance(manifest_id, str)
                and manifest_id
                and isinstance(metadata_id, str)
                and metadata_id
                and manifest_id != metadata_id
            ):
                issues.append("checkpoint_id_mismatch")
            expected_step = int(match.group("step"))
            training_state = metadata.get("training_state")
            if not isinstance(training_state, dict):
                issues.append("training_state_missing")
                actual_step = -1
            else:
                raw_step = training_state.get("environment_steps")
                actual_step = _integer(raw_step, -1)
                if _finite_number(raw_step) is None:
                    issues.append("environment_steps_missing")
            if actual_step != expected_step:
                issues.append("environment_step_mismatch")
            size_bytes = 0
            files = manifest.get("files")
            declared_files: set[str] = set()
            if isinstance(files, list) and files:
                if len(files) > _MAX_CHECKPOINT_FILES:
                    issues.append("manifest_files_limit_exceeded")
                for file_entry in files[:_MAX_CHECKPOINT_FILES]:
                    if not isinstance(file_entry, dict):
                        issues.append("invalid_manifest_entry")
                        continue
                    file_name = file_entry.get("path")
                    if not isinstance(file_name, str):
                        issues.append("invalid_manifest_path")
                        continue
                    relative_file = Path(file_name)
                    if relative_file.is_absolute() or len(relative_file.parts) != 1 or relative_file.name != file_name:
                        issues.append("unsafe_manifest_path")
                        continue
                    if file_name in declared_files:
                        issues.append(f"duplicate_manifest_path:{file_name}")
                        continue
                    declared_files.add(file_name)
                    raw_expected_size = file_entry.get("size_bytes")
                    expected_size = _integer(raw_expected_size, -1)
                    if _finite_number(raw_expected_size) is None or expected_size < 0:
                        issues.append(f"invalid_size:{file_name}")
                        continue
                    payload_path = entry / file_name
                    if (
                        not payload_path.is_file()
                        or _is_link_or_reparse(payload_path)
                        or not _safe_resolved_child(payload_path, entry)
                    ):
                        issues.append(f"missing_or_unsafe:{file_name}")
                        continue
                    try:
                        actual_size = payload_path.stat().st_size
                    except OSError:
                        issues.append(f"missing:{file_name}")
                        continue
                    size_bytes += actual_size
                    if expected_size >= 0 and actual_size != expected_size:
                        issues.append(f"size_mismatch:{file_name}")
                if "metadata.json" not in declared_files:
                    issues.append("metadata_not_declared")
            else:
                issues.append("manifest_files_missing")
            created_at = _finite_number(manifest.get("created_unix_s")) or modified
            relative_display = entry.relative_to(self.artifact_root).as_posix()
            published.append(
                {
                    "name": entry.name,
                    "kind": match.group("kind"),
                    "environment_steps": expected_step,
                    "created_at": created_at,
                    "path": relative_display,
                    "size_bytes": size_bytes,
                    "valid": not issues,
                    "verification": "size_verified" if not issues else "invalid",
                    "issues": issues,
                }
            )
        published.sort(
            key=lambda item: (
                _integer(item.get("environment_steps")),
                1 if item.get("kind") == "final" else 0,
                _string(item.get("name")),
            )
        )
        publishing.sort(key=lambda item: _integer(item.get("environment_steps")))
        result = {
            "published": published,
            "publishing": publishing,
            "latest_activity": latest_activity,
        }
        self._checkpoint_cache[directory] = (monotonic_now, result)
        return result

    def _evaluation_journal_activity(
        self,
        run: DiscoveredRun,
        *,
        completed_gates: set[tuple[str, int]],
        metrics_mtime: float | None,
    ) -> tuple[float | None, JsonDict | None]:
        latest_mtime: float | None = None
        pending_evaluation: JsonDict | None = None
        pending_mtime: float | None = None
        try:
            entries = run.run_directory.iterdir()
        except OSError:
            return None, None
        try:
            for entry in islice(entries, _MAX_EVALUATION_JOURNAL_SCAN_ENTRIES):
                match = _EVALUATION_RE.fullmatch(entry.name)
                if (
                    match is None
                    or not entry.is_file()
                    or _is_link_or_reparse(entry)
                    or not _safe_resolved_child(entry, run.run_directory)
                ):
                    continue
                try:
                    modified = entry.stat().st_mtime
                    gate = int(match.group("step"))
                except (OSError, OverflowError, ValueError):
                    continue
                gate_kind = _EVALUATION_GATE_KIND_BY_JOURNAL[match.group("journal_kind")]
                if latest_mtime is None or modified > latest_mtime:
                    latest_mtime = modified
                if (gate_kind, gate) not in completed_gates and (pending_mtime is None or modified >= pending_mtime):
                    pending_evaluation = {
                        "kind": gate_kind,
                        "gate": gate,
                        "journal_name": entry.name,
                        "modified_at": modified,
                    }
                    pending_mtime = modified
        except OSError:
            # A concurrent artifact cleanup can invalidate the directory
            # iterator. Preserve any safe activity observed before that point.
            pass
        if pending_mtime is not None and metrics_mtime is not None and pending_mtime < metrics_mtime:
            pending_evaluation = None
        return latest_mtime, pending_evaluation

    def _evaluation_activity(
        self,
        run: DiscoveredRun,
        parser: IncrementalMetrics,
    ) -> tuple[float | None, JsonDict | None]:
        completed_gates = {
            (
                _evaluation_gate_kind(event),
                _integer(
                    event.get("evaluation_gate")
                    if _finite_number(event.get("evaluation_gate")) is not None
                    else event.get("environment_steps")
                ),
            )
            for event in parser.evaluations
        }
        return self._evaluation_journal_activity(
            run,
            completed_gates=completed_gates,
            metrics_mtime=parser.last_mtime,
        )

    def _status(
        self,
        run: DiscoveredRun,
        parser: IncrementalMetrics,
        checkpoints: JsonDict,
        pending_evaluation: JsonDict | None,
    ) -> JsonDict:
        now = self._now()
        checkpoint_activity = _finite_number(checkpoints.get("latest_activity"))
        activities = [
            value
            # Host filesystem activity is authoritative for freshness. Event
            # clocks may be skewed or even lie in the future.
            for value in (parser.last_mtime, checkpoint_activity)
            if value is not None
        ]
        latest_activity = min(now, max(activities, default=run.start_time))
        telemetry_age = max(0.0, now - latest_activity)
        publishing = checkpoints.get("publishing")
        is_publishing = isinstance(publishing, list) and bool(publishing)

        if "run_complete" in parser.latest_by_event:
            published = checkpoints.get("published")
            has_final_checkpoint = isinstance(published, list) and any(
                isinstance(item, dict) and item.get("kind") == "final" and item.get("valid") is True
                for item in published
            )
            completion = _run_completion_status(parser.latest_by_event["run_complete"])
            return {
                "state": completion["state"],
                "phase": completion["phase"],
                "label": completion["label"],
                "telemetry_age_s": telemetry_age,
                "evidence": [
                    *completion["evidence"],
                    (
                        "最终检查点已原子发布并通过大小校验"
                        if has_final_checkpoint
                        else "未发现通过轻量校验的最终检查点"
                    ),
                ],
            }
        successor = self._successor_for(run)
        if successor is not None:
            return {
                "state": "continued",
                "phase": "continued",
                "label": "已续跑",
                "telemetry_age_s": telemetry_age,
                "evidence": [f"exact-resume successor: {successor.key}"],
            }
        if "interrupt" in parser.latest_by_event:
            return {
                "state": "interrupted",
                "phase": "interrupted",
                "label": "已中断",
                "telemetry_age_s": telemetry_age,
                "evidence": ["interrupt 事件已持久化"],
            }
        if "run_failed" in parser.latest_by_event:
            return {
                "state": "failed",
                "phase": "failed",
                "label": "运行失败",
                "telemetry_age_s": telemetry_age,
                "evidence": ["run_failed 事件已持久化"],
            }
        if "circuit_breaker_open" in parser.latest_by_event:
            return {
                "state": "failed",
                "phase": "circuit_breaker",
                "label": "熔断停止",
                "telemetry_age_s": telemetry_age,
                "evidence": ["circuit_breaker_open 事件已持久化"],
            }
        if telemetry_age > self.stale_seconds:
            return {
                "state": "stale",
                "phase": "stale_unknown",
                "label": "遥测陈旧 · 状态未知",
                "telemetry_age_s": telemetry_age,
                "evidence": [
                    f"超过 {int(self.stale_seconds)} 秒无结构化活动",
                    "run_complete 尚未持久化",
                    "没有通用失败事件, 不能据此断言进程崩溃",
                ],
            }
        if is_publishing:
            return {
                "state": "checkpointing",
                "phase": "checkpointing",
                "label": "保存检查点",
                "telemetry_age_s": telemetry_age,
                "evidence": ["发现原子检查点 staging; 尚不可恢复"],
            }
        if pending_evaluation is not None:
            return _pending_evaluation_status(
                pending_evaluation,
                telemetry_age=telemetry_age,
            )
        latest_progress = parser.latest_by_event.get("learner_progress")
        stage = _string(latest_progress.get("stage")) if latest_progress is not None else "collecting"
        return {
            "state": "running",
            "phase": stage or "collecting",
            "label": "训练中",
            "telemetry_age_s": telemetry_age,
            "evidence": ["近期存在结构化训练遥测"],
        }

    def _empty_snapshot(self) -> JsonDict:
        return {
            "generated_at": self._now(),
            "status": {
                "state": "waiting",
                "phase": "waiting",
                "label": "等待遥测",
                "telemetry_age_s": None,
                "evidence": ["未发现规范的 runs/<lineage>/run-<uuid>/metrics.jsonl"],
            },
            "run": {},
            "context": {
                "mode": None,
                "revival_budget": None,
                "standard_game": False,
                "label": "尚未选择训练运行",
            },
            "lifecycle": {
                "run_complete_persisted": False,
                "training_horizon_reached": False,
                "pending_evaluation": None,
            },
            "progress": {},
            "throughput": {},
            "latest_episode": None,
            "latest_learner": None,
            "latest_actor": None,
            "evaluations": [],
            "episodes": [],
            "learner_series": [],
            "checkpoints": {"published": [], "publishing": []},
            "incidents": [],
            "data_quality": {},
            "alerts": [],
        }

    def snapshot(self, run_key: str | None = None) -> JsonDict:
        with self._lock:
            runs = self.discover()
            if not runs:
                return self._empty_snapshot()
            if run_key is None:
                selected = runs[0]
            else:
                selected_match = self._runs.get(run_key)
                if selected_match is None:
                    raise KeyError(f"unknown run key: {run_key}")
                selected = selected_match
            chain = self._chain(selected)
            aggregates = [(run, self._parser(run)) for run in chain]
            segments = [
                (
                    run,
                    aggregate,
                    _resume_boundary(aggregate, chain[index + 1] if index + 1 < len(chain) else None),
                )
                for index, (run, aggregate) in enumerate(aggregates)
            ]
            selected_parser = aggregates[-1][1]
            checkpoints = self._checkpoint_snapshot(selected)
            evaluation_activity, pending_evaluation = self._evaluation_activity(selected, selected_parser)
            if evaluation_activity is not None:
                current_activity = _finite_number(checkpoints.get("latest_activity"))
                if current_activity is None or evaluation_activity > current_activity:
                    checkpoints["latest_activity"] = evaluation_activity
            status = self._status(selected, selected_parser, checkpoints, pending_evaluation)

            config = _mapping(selected.start_event.get("config"))
            runtime = _mapping(config.get("runtime"))
            curriculum = _mapping(config.get("curriculum"))
            model = _mapping(config.get("model"))
            state = _mapping(selected.start_event.get("state"))
            completed = _terminal_projection(selected_parser.latest_by_event.get("run_complete"))
            environment_steps = _latest_counter(selected_parser, field="environment_steps")
            learner_updates = max(
                _latest_counter(selected_parser, field="learner_updates"),
                _integer(state.get("learner_updates")),
                _integer(selected_parser.latest_by_event.get("learner_update", {}).get("policy_version")),
            )
            policy_version = max(
                _integer(completed.get("policy_version")),
                _integer(state.get("policy_version")),
                _integer(selected_parser.latest_by_event.get("learner_update", {}).get("policy_version")),
                _integer(selected_parser.latest_by_event.get("checkpoint", {}).get("policy_version")),
            )
            training_episodes = _latest_counter(
                selected_parser,
                field="episodes",
                episode_increment=True,
            )
            evaluation_episodes = max(
                _integer(completed.get("evaluation_episodes")),
                _integer(state.get("evaluation_episodes"))
                + sum(_integer(event.get("episodes")) for event in selected_parser.evaluations),
                _integer(selected_parser.latest_by_event.get("checkpoint", {}).get("evaluation_episodes")),
            )
            maximum_candidates = max(
                _integer(completed.get("maximum_observed_candidates")),
                _integer(state.get("maximum_observed_candidates")),
                max(
                    (
                        _integer(point.get("maximum_observed_candidates"))
                        for _, aggregate, boundary in segments
                        for point in _bounded_candidate_peaks(aggregate, boundary)
                    ),
                    default=0,
                ),
                max(
                    (
                        _integer(event.get("run_maximum_observed_candidates"))
                        for _, aggregate, boundary in segments
                        for event in _bounded_evaluations(aggregate, boundary)
                    ),
                    default=0,
                ),
            )
            semantic_candidates = max(
                (
                    _integer(point.get("maximum_observed_semantic_candidates"))
                    for _, aggregate, boundary in segments
                    for point in _bounded_candidate_peaks(aggregate, boundary)
                ),
                default=0,
            )
            equivalence_size = max(
                (
                    _integer(point.get("maximum_equivalence_class_size"))
                    for _, aggregate, boundary in segments
                    for point in _bounded_candidate_peaks(aggregate, boundary)
                ),
                default=0,
            )
            target = _integer(runtime.get("total_environment_steps"))
            percent = environment_steps / target * 100.0 if target > 0 else None

            all_environment_points: list[tuple[float, int]] = []
            for _, aggregate, boundary in segments:
                maximum_step = _integer(boundary.get("environment_steps")) if boundary is not None else None
                all_environment_points.extend(
                    point for point in aggregate.environment_points if maximum_step is None or point[1] <= maximum_step
                )
            all_environment_points.sort(key=lambda item: item[0])
            rate_reference = None if status["state"] == "completed" else self._now()
            rate_15m, confidence_15m = _rate(
                all_environment_points,
                15.0 * 60.0,
                reference_time=rate_reference,
            )
            rate_60m, confidence_60m = _rate(
                all_environment_points,
                60.0 * 60.0,
                reference_time=rate_reference,
            )
            update_rate = _recent_update_rate(selected_parser.learner_points)
            eta: float | None
            if status["state"] == "running" and rate_60m is not None and rate_60m > 0 and target > environment_steps:
                eta = (target - environment_steps) / rate_60m
            else:
                eta = None
            confidence = confidence_15m if rate_15m is not None else confidence_60m

            evaluations_by_gate: dict[tuple[str, int], tuple[int, JsonDict]] = {}
            evaluation_sequence = 0
            for _, aggregate, boundary in segments:
                for event in _bounded_evaluations(aggregate, boundary):
                    normalised = _normalise_evaluation(event)
                    key = (
                        _string(normalised.get("gate_kind"), "validation"),
                        _integer(normalised.get("evaluation_gate")),
                    )
                    evaluations_by_gate[key] = (evaluation_sequence, normalised)
                    evaluation_sequence += 1
            projected_evaluations = sorted(
                evaluations_by_gate.values(),
                key=lambda item: item[0],
            )
            evaluation_timestamps = [
                _finite_number(evaluation.get("timestamp")) for _, evaluation in projected_evaluations
            ]
            finite_evaluation_timestamps = [timestamp for timestamp in evaluation_timestamps if timestamp is not None]
            timestamps_are_reliable = (
                bool(evaluation_timestamps)
                and len(finite_evaluation_timestamps) == len(evaluation_timestamps)
                and all(previous <= following for previous, following in pairwise(finite_evaluation_timestamps))
            )
            if timestamps_are_reliable:
                projected_evaluations.sort(
                    key=lambda item: (
                        _finite_number(item[1].get("timestamp")) or 0.0,
                        _EVALUATION_GATE_KIND_ORDER.get(
                            _string(item[1].get("gate_kind"), "validation"),
                            -1,
                        ),
                        item[0],
                    )
                )
            else:
                # Mixed/legacy clocks are not safe to compare. The chained
                # metrics parse order is the authoritative fallback.
                projected_evaluations.sort(key=lambda item: item[0])
            evaluations = [evaluation for _, evaluation in projected_evaluations]

            episodes = [
                episode for _, aggregate, boundary in segments for episode in _bounded_episodes(aggregate, boundary)
            ][-_MAX_EPISODES:]
            learner_series = [
                point for _, aggregate, boundary in segments for point in _bounded_learner_series(aggregate, boundary)
            ]
            learner_series = _downsample_extrema(learner_series, 800, "loss")
            incidents = [
                incident for _, aggregate, boundary in segments for incident in _bounded_incidents(aggregate, boundary)
            ][-_MAX_INCIDENTS:]

            alerts: list[JsonDict] = []
            malformed = sum(aggregate.malformed_rows for _, aggregate in aggregates)
            oversize = sum(aggregate.oversize_rows for _, aggregate in aggregates)
            if malformed or oversize:
                alerts.append(
                    {
                        "type": "data_quality",
                        "message": f"发现 {malformed} 条损坏行、{oversize} 条超大行; 后续完整记录仍继续解析",
                    }
                )
            if status["state"] == "stale":
                alerts.append(
                    {
                        "type": "stale_unknown",
                        "message": "遥测已超过阈值; 没有权威终态, 需人工确认进程或启动器",
                    }
                )
            if evaluations:
                latest_evaluation = evaluations[-1]
                if _finite_number(latest_evaluation.get("deadlock_rate")) == 1.0:
                    alerts.append(
                        {
                            "type": "heldout_deadlock",
                            "message": "最新 held-out 评估全部 deadlock; 这是模型质量告警, 不是基础设施重试",
                        }
                    )
                latest_completion = _mapping(selected_parser.latest_by_event.get("run_complete"))
                audited_policy = _finite_number(latest_evaluation.get("policy_version"))
                terminal_policy = _finite_number(latest_completion.get("policy_version"))
                if (
                    latest_evaluation.get("gate_kind") == "final_audit"
                    and _string(latest_evaluation.get("policy_model_state_sha256"))
                    and audited_policy is not None
                    and terminal_policy is not None
                    and int(audited_policy) != int(terminal_policy)
                ):
                    alerts.append(
                        {
                            "type": "final_audit_policy_mismatch",
                            "message": (
                                "最终审计策略版本 "
                                f"{int(audited_policy)} 与 run_complete 策略版本 "
                                f"{int(terminal_policy)} 不一致; 不能把该审计结果冒充为最终 checkpoint 结果"
                            ),
                        }
                    )
            max_candidates = _integer(model.get("max_candidates"))
            if max_candidates > 0 and semantic_candidates >= int(max_candidates * 0.9):
                alerts.append(
                    {
                        "type": "candidate_capacity",
                        "message": (
                            f"语义候选峰值 {semantic_candidates} 接近容量 {max_candidates}; "
                            "raw 合并前候选峰值不单独触发此告警"
                        ),
                    }
                )
            invalid_checkpoints = [
                item
                for item in checkpoints.get("published", [])
                if isinstance(item, dict) and item.get("valid") is False
            ]
            if invalid_checkpoints:
                alerts.append(
                    {
                        "type": "checkpoint_structure",
                        "message": f"{len(invalid_checkpoints)} 个已发布检查点未通过轻量结构/大小验证",
                    }
                )

            raw_revival_budget = curriculum.get("revival_budget")
            revival_number = _finite_number(raw_revival_budget)
            revival_budget = int(revival_number) if revival_number is not None else None
            mode = _string(curriculum.get("mode"))
            unlimited_revival = revival_budget == -1
            standard_game = (
                revival_budget is not None and not unlimited_revival and mode not in {"native-revival-preheat"}
            )
            parsed_rows = sum(aggregate.parsed_rows for _, aggregate in aggregates)
            checkpoint_load = _mapping(selected.start_event.get("checkpoint_load"))
            return {
                "generated_at": self._now(),
                "schema": "sts2-training-dashboard-v1",
                "status": status,
                "run": {
                    "key": selected.key,
                    "id": selected.run_id,
                    "run_id": selected.run_id,
                    "lineage": selected.lineage,
                    "chain_run_count": len(chain),
                    "chain": [
                        {
                            "key": run.key,
                            "run_id": run.run_id,
                            "start_time": run.start_time,
                            "checkpoint_load_mode": _mapping(run.start_event.get("checkpoint_load")).get("mode"),
                        }
                        for run in chain
                    ],
                    "start_time": selected.start_time,
                    "pipeline": selected.start_event.get("pipeline"),
                    "profile": config.get("profile"),
                    "configured_device": runtime.get("device"),
                    "collector_device": runtime.get("collector_device"),
                    "checkpoint_load_mode": checkpoint_load.get("mode"),
                    "model_architecture": model.get("architecture"),
                },
                "context": {
                    "mode": mode or None,
                    "revival_budget": revival_budget,
                    "standard_game": standard_game,
                    "label": (
                        "无限原生复活预热 · 非标准胜率"
                        if unlimited_revival
                        else "标准游戏评估"
                        if standard_game
                        else mode or "训练上下文未标注"
                    ),
                },
                "lifecycle": {
                    "run_complete_persisted": "run_complete" in selected_parser.latest_by_event,
                    "training_horizon_reached": target > 0 and environment_steps >= target,
                    "completion_status": (_string(completed.get("completion_status")) or None),
                    "pending_evaluation": pending_evaluation,
                },
                "progress": {
                    "environment_steps": environment_steps,
                    "target_environment_steps": target,
                    "percent": percent,
                    "training_episodes": training_episodes,
                    "learner_updates": learner_updates,
                    "policy_version": policy_version,
                    "evaluation_episodes": evaluation_episodes,
                    "maximum_observed_candidates": maximum_candidates,
                    "maximum_observed_semantic_candidates": semantic_candidates,
                    "maximum_equivalence_class_size": equivalence_size,
                },
                "throughput": {
                    "env_steps_per_s_15m": rate_15m,
                    "env_steps_per_s_60m": rate_60m,
                    "learner_updates_per_s": update_rate,
                    "eta_s": eta,
                    "confidence": confidence,
                    "scope": "pre_completion" if status["state"] == "completed" else "live",
                    "paused_for_phase": status["phase"] if status["phase"] in {"evaluating", "checkpointing"} else None,
                },
                "latest_episode": episodes[-1] if episodes else None,
                "latest_learner": selected_parser.latest_learner,
                "latest_actor": selected_parser.latest_actor,
                "evaluations": evaluations,
                "episodes": episodes,
                "learner_series": learner_series,
                "checkpoints": {
                    "published": checkpoints.get("published", []),
                    "publishing": checkpoints.get("publishing", []),
                },
                "incidents": incidents,
                "data_quality": {
                    "metrics_rows": parsed_rows + malformed + oversize,
                    "parsed_rows": parsed_rows,
                    "malformed_rows": malformed,
                    "oversize_rows": oversize,
                    "dropped_rows": malformed + oversize,
                    "partial_line": any(aggregate.partial_line for _, aggregate in aggregates),
                    "last_complete_offset": selected_parser.offset,
                    "chain_files": len(chain),
                    "unknown_events": sum(aggregate.event_counts.get("unknown", 0) for _, aggregate in aggregates),
                },
                "alerts": alerts,
            }


__all__ = [
    "DashboardStore",
    "DiscoveredRun",
    "IncrementalMetrics",
]
