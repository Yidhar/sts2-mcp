"""Loopback-only, read-only monitor for the synchronous macro-credit course.

No torch import, checkpoint deserialization, simulator, or training control.
Unlike the legacy dashboard this reads the selected course/status/cohort JSON
directly; it never synthesizes old metrics.jsonl or silently selects another run.
"""

from __future__ import annotations

import argparse
import copy
import ipaddress
import json
import math
import os
import threading
import time
from collections import OrderedDict
from collections.abc import Mapping, Sequence
from datetime import UTC, datetime
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from itertools import islice
from pathlib import Path
from typing import Any, BinaryIO, cast
from urllib.parse import urlsplit

from sts2_rl.artifacts import resolve_artifact_path

SCHEMA = "sts2-macro-credit-dashboard-v1"
COURSE_VERSION = "sts2-synchronous-macro-credit-course-v1"
MAX_JSON_BYTES = 1024 * 1024
MAX_TAIL_BYTES = 1024 * 1024
MAX_ATTEMPTS = 10000
MAX_HISTORY = 200
MAX_GAMEPLAY_HISTORY = 64
MAX_ROOTS_PER_UPDATE = 16
MAX_COLLECTION_PASSES = 66
MAX_CANDIDATES = 64
MAX_COLLECTION_READS = 64
MAX_REFERENCE_TAIL_READS = 8
MAX_CACHE_FILES = 1024
MAX_CACHE_BYTES = 32 * 1024 * 1024
COLLECTION_VERSION = "sts2-macro-credit-collection-v1"
GAMEPLAY_SCOPE = "training_root_windows_not_full_episodes"
_CSP = "default-src 'self'; script-src 'self' 'unsafe-inline'; style-src 'self' 'unsafe-inline'; frame-ancestors 'none'"


def _reject_constant(value: str) -> None:
    raise ValueError(f"non-finite JSON constant: {value}")


def _mapping(value: Any) -> dict[str, Any]:
    return dict(value) if isinstance(value, Mapping) else {}


def _integer(value: Any) -> int | None:
    return int(value) if isinstance(value, int) and not isinstance(value, bool) and value >= 0 else None


def _number(value: Any) -> float | None:
    if isinstance(value, bool) or not isinstance(value, int | float):
        return None
    try:
        converted = float(value)
    except OverflowError:
        return None
    return converted if math.isfinite(converted) else None


def _text(value: Any, limit: int = 256) -> str | None:
    return value[:limit] if isinstance(value, str) else None


def _boolean(value: Any) -> bool | None:
    return value if isinstance(value, bool) else None


def _finite_float(value: str) -> float:
    result = float(value)
    if not math.isfinite(result):
        raise ValueError("non-finite JSON number")
    return result


def _signature(stat: os.stat_result) -> tuple[int, int, int]:
    # Windows path.stat and os.fstat on the shared-delete handle can report
    # different ctime meanings (creation versus metadata-change time). Do not
    # mix that non-portable clock into the key. These immutable/atomically
    # replaced artifacts use mtime+size, plus file identity for replacements.
    return (stat.st_mtime_ns, stat.st_size, stat.st_ino)


def _reward_components(value: Any) -> dict[str, float | None]:
    source = _mapping(value)
    result = {key: _number(source.get(key)) for key in
              ("reward", "terminal_reward", "revival_penalty", "pace_penalty", "hp_loss_penalty")}
    # In v9 these names are aliases, NOT additive reward terms.
    result["progress_reward"] = _number(source.get("progress_reward", source.get("potential_reward")))
    return result


def _pass_gameplay(value: Mapping[str, Any]) -> dict[str, Any]:
    reward_state = _mapping(value.get("reward_state"))
    integer_keys = (
        "selected_root_index", "root_floor", "ending_floor", "post_root_floor_span", "native_steps",
        "prefix_native_steps", "branch_native_steps", "semantic_decisions", "macro_decisions", "combat_decisions",
        "continuation_macro_decisions", "continuation_combat_decisions", "post_root_combats_started",
        "post_root_combats_exited", "post_root_combats_completed", "mechanical_dispatches",
    )
    return {
        "name": _text(value.get("name")), "policy_mode": _text(value.get("policy_mode")),
        "ending": _text(value.get("ending")), "scope": "post_root_observed_window",
        **{key: _integer(value.get(key)) for key in integer_keys},
        "observed_discounted_return": _number(value.get("observed_discounted_return")),
        "raw_reward_components": _reward_components(value.get("raw_reward_components")),
        "discounted_reward_components": _reward_components(value.get("discounted_reward_components")),
        "tail_discount": _number(value.get("tail_discount")), "tail_value_used": _boolean(value.get("tail_value_used")),
        "elapsed_seconds": _number(value.get("elapsed_seconds")),
        "last_total_revivals": _integer(reward_state.get("last_total_revivals")),
        "act_revivals": _integer(reward_state.get("act_revivals")),
        "current_act": _integer(reward_state.get("current_act")),
        "window_revivals": None, "player_hp_lost_since_reset": None, "trace_revivals_since_reset": None,
        "terminal_reason": None, "terminated": None, "truncated": None, "run_result": None,
        "native_steps_scope": "since_reset; branch_native_steps is the post-root window",
        "revivals_scope": "cumulative at window end since reset, not new deaths in this window",
    }


def _loopback(host: str) -> bool:
    if host == "localhost":
        return True
    try:
        return ipaddress.ip_address(host).is_loopback
    except ValueError:
        return False


def _open_artifact_read(path: Path) -> BinaryIO:
    """Do not block the trainer's atomic replacement of observed metadata.

    Ordinary Windows Python open() does not share delete access. An otherwise
    read-only dashboard could therefore make a concurrent os.replace fail.
    Explicitly share read/write/delete while requesting read access only.
    This supports the actual WSL/POSIX trainer's rename semantics. Native
    Windows MoveFileEx replacement of an open destination still differs and
    is not certified by this WSL-course adapter.
    """
    if os.name != "nt":
        return path.open("rb")
    import ctypes
    import msvcrt
    from ctypes import wintypes

    kernel = ctypes.WinDLL("kernel32", use_last_error=True)
    create = kernel.CreateFileW
    create.argtypes = (wintypes.LPCWSTR, wintypes.DWORD, wintypes.DWORD, wintypes.LPVOID,
                       wintypes.DWORD, wintypes.DWORD, wintypes.HANDLE)
    create.restype = wintypes.HANDLE
    handle = create(str(path), 0x80000000, 0x1 | 0x2 | 0x4, None, 3, 0x80, None)
    if handle == ctypes.c_void_p(-1).value:
        raise ctypes.WinError(ctypes.get_last_error())
    try:
        descriptor = msvcrt.open_osfhandle(int(handle), os.O_RDONLY | os.O_BINARY)
    except BaseException:
        close = kernel.CloseHandle
        close.argtypes = (wintypes.HANDLE,)
        close.restype = wintypes.BOOL
        close(handle)
        raise
    try:
        return cast(BinaryIO, os.fdopen(descriptor, "rb"))
    except BaseException:
        os.close(descriptor)
        raise


class MacroDashboardStore:
    def __init__(self, run_dir: Path, *, artifact_root: Path, invocation_dir: Path | None = None) -> None:
        self.run_dir = resolve_artifact_path(run_dir, root=artifact_root)
        self.artifact_root = artifact_root.resolve()
        if not self.run_dir.is_dir():
            raise ValueError("macro dashboard requires an existing selected run directory")
        self.invocation_dir = resolve_artifact_path(invocation_dir, root=artifact_root) if invocation_dir else None
        self._lock = threading.RLock()
        self._cache: OrderedDict[tuple[str, Path], tuple[tuple[int, int, int], dict[str, Any], int]] = OrderedDict()
        self._cache_bytes = 0
        self._collection_reads = MAX_COLLECTION_READS
        self._reference_tail_reads = MAX_REFERENCE_TAIL_READS
        course = self._read_json(self.run_dir / "course.json", self.run_dir)
        if _mapping(course.get("course_config")).get("version") != COURSE_VERSION:
            raise ValueError("selected run is not a synchronous macro-credit course")
        if not isinstance(course.get("lineage_id"), str) or not course["lineage_id"]:
            raise ValueError("macro course has no lineage_id")
        self.lineage_id = str(course["lineage_id"])

    @staticmethod
    def _safe(path: Path, boundary: Path) -> Path:
        resolved = path.resolve()
        if not resolved.is_relative_to(boundary) or path.is_symlink():
            raise ValueError("dashboard refuses a path outside its fixed artifact boundary")
        return resolved

    def _cached(self, kind: str, path: Path, signature: tuple[int, int, int]) -> dict[str, Any] | None:
        key = (kind, path)
        entry = self._cache.get(key)
        if entry is not None and entry[0] == signature:
            self._cache.move_to_end(key)
            return entry[1]
        return None

    def _remember(self, kind: str, path: Path, signature: tuple[int, int, int], value: dict[str, Any]) -> None:
        key = (kind, path)
        previous = self._cache.pop(key, None)
        if previous:
            self._cache_bytes -= previous[2]
        size = len(json.dumps(value, ensure_ascii=False, allow_nan=False).encode("utf-8"))
        if size <= MAX_CACHE_BYTES:
            self._cache[key] = (signature, value, size)
            self._cache_bytes += size
        while self._cache and (len(self._cache) > MAX_CACHE_FILES or self._cache_bytes > MAX_CACHE_BYTES):
            _, removed = self._cache.popitem(last=False)
            self._cache_bytes -= removed[2]

    def _read_json(self, path: Path, boundary: Path, *, cache: bool = True) -> dict[str, Any]:
        resolved = self._safe(path, boundary)
        signature = _signature(resolved.stat())
        if signature[1] > MAX_JSON_BYTES:
            raise ValueError("dashboard JSON size limit exceeded")
        cached = self._cached("metadata", resolved, signature) if cache else None
        if cached is not None:
            return cached
        with _open_artifact_read(resolved) as handle:
            raw = handle.read(MAX_JSON_BYTES + 1)
            read_signature = _signature(os.fstat(handle.fileno()))
        if len(raw) > MAX_JSON_BYTES:
            raise ValueError("dashboard JSON size limit exceeded")
        parsed = json.loads(raw, parse_constant=_reject_constant, parse_float=_finite_float)
        if not isinstance(parsed, dict):
            raise ValueError("dashboard metadata must be a JSON object")
        if cache:
            self._remember("metadata", resolved, read_signature, parsed)
        return parsed

    def _optional_json(self, path: Path, boundary: Path, warnings: list[str]) -> dict[str, Any]:
        try:
            return self._read_json(path, boundary)
        except FileNotFoundError:
            return {}
        except (OSError, ValueError) as error:
            warnings.append(f"{path.name}: {type(error).__name__}; metadata not used")
            return {}

    def _tail_progress(self, path: Path) -> dict[str, Any]:
        resolved = self._safe(path, self.run_dir)
        signature = _signature(resolved.stat())
        cached = self._cached("progress_tail", resolved, signature)
        if cached is not None:
            return dict(cached)
        lines = self._tail_lines(resolved)
        for line in reversed(lines):
            try:
                row = json.loads(line, parse_constant=_reject_constant, parse_float=_finite_float)
            except (ValueError, UnicodeError):
                continue
            if not isinstance(row, dict):
                continue
            facts = _mapping(row.get("facts"))
            run = _mapping(_mapping(_mapping(facts.get("state")).get("observation")).get("run"))
            semantic = _mapping(facts.get("semantic"))
            result = {"record_kind": row.get("kind"), "floor": _integer(facts.get("ending_floor", run.get("floor"))),
                      "native_steps": _integer(facts.get("native_steps")),
                      "surface": semantic.get("surface"), "domain": semantic.get("domain")}
            self._remember("progress_tail", resolved, signature, result)
            return result
        self._remember("progress_tail", resolved, signature, {})
        return {}

    @staticmethod
    def _tail_lines(resolved: Path) -> list[bytes]:
        with _open_artifact_read(resolved) as handle:
            size = handle.seek(0, os.SEEK_END)
            start = max(0, size - MAX_TAIL_BYTES)
            handle.seek(start)
            raw = handle.read(MAX_TAIL_BYTES)
        lines = raw.splitlines()
        if start and lines:
            lines.pop(0)
        if raw and not raw.endswith(b"\n") and lines:
            lines.pop()
        return lines

    def _reference_state(self, path: Path, warnings: list[str]) -> dict[str, Any]:
        try:
            resolved = self._safe(path, self.run_dir)
            signature = _signature(resolved.stat())
            cached = self._cached("reference_tail", resolved, signature)
            if cached is not None:
                return dict(cached)
            if self._reference_tail_reads <= 0:
                return {"trace_evidence_pending": True}
            self._reference_tail_reads -= 1
            lines = self._tail_lines(resolved)
        except FileNotFoundError:
            return {}
        except (OSError, ValueError):
            warnings.append("reference tail unavailable; exact HP/terminal fields remain unknown")
            return {}
        result: dict[str, Any] = {"trace_evidence_pending": False}
        for line in reversed(lines):
            try:
                row = json.loads(line, parse_constant=_reject_constant, parse_float=_finite_float)
            except (ValueError, UnicodeError):
                continue
            if not isinstance(row, dict):
                continue
            state = _mapping(_mapping(row.get("facts")).get("state"))
            if not state:
                continue  # End summaries do not contain the factual HP counter.
            training = _mapping(_mapping(state.get("observation")).get("_training"))
            hp = _number(training.get("player_hp_lost"))
            result.update({
                "player_hp_lost_since_reset": hp if hp is not None and hp >= 0 else None,
                "trace_revivals_since_reset": _integer(training.get("revivals_used")),
                "terminal_reason": _text(state.get("terminal_reason")),
                "terminated": _boolean(state.get("terminated")), "truncated": _boolean(state.get("truncated")),
                "run_result": _text(_mapping(_mapping(state.get("transition")).get("facts")).get("run_result")),
                "trace_evidence_record_kind": _text(row.get("kind")),
            })
            break
        self._remember("reference_tail", resolved, signature, result)
        return result

    def _collection(self, path: Path, warnings: list[str]) -> dict[str, Any]:
        """Cache the compact projection, not repeated 150KB+ plans/edge tapes."""
        try:
            resolved = self._safe(path, self.run_dir)
            signature = _signature(resolved.stat())
            if signature[1] > MAX_JSON_BYTES:
                raise ValueError("collection JSON size limit exceeded")
            cached = self._cached("collection", resolved, signature)
            if cached is not None:
                return cached
            if self._collection_reads <= 0:
                return {"details_pending": True}
            self._collection_reads -= 1
            raw = self._read_json(resolved, self.run_dir, cache=False)
        except FileNotFoundError:
            return {}
        except (OSError, ValueError):
            warnings.append(f"{path.parent.name}/collection.json unavailable; gameplay not inferred")
            return {}
        count = _integer(raw.get("candidate_count"))
        passes = raw.get("passes")
        if (raw.get("format") != COLLECTION_VERSION or raw.get("status") != "complete"
                or raw.get("continuation_policy") != "frozen_stochastic" or raw.get("data_partition") != "training"
                or raw.get("same_action_gate") != "matched"
                or count is None or not 2 <= count <= MAX_CANDIDATES or not isinstance(passes, list)
                or len(passes) > MAX_COLLECTION_PASSES):
            warnings.append("incomplete/incompatible collection excluded from committed gameplay")
            return {}
        projected = [_pass_gameplay(item) for item in passes if isinstance(item, Mapping)]
        references = [item for item in projected if item["name"] == "reference"]
        if len(references) != 1 or not _text(raw.get("root_id")) or _integer(raw.get("seed")) is None:
            warnings.append("collection reference identity unavailable; gameplay not paired")
            return {}
        reference = references[0]
        alternatives = [item for item in projected if item["name"] != "same-action-control"]
        selected = reference["selected_root_index"]
        if (selected is None or selected >= count or len(alternatives) != count
                or {item["selected_root_index"] for item in alternatives} != set(range(count))
                or any(item["policy_mode"] != "frozen_stochastic" or item["ending"] not in {"macro_horizon", "factual_terminal"}
                       for item in projected)):
            warnings.append("partial/invalid candidate windows excluded from committed gameplay")
            return {}
        source_reference = next(item for item in passes if isinstance(item, Mapping) and item.get("name") == "reference")
        root = _mapping(source_reference.get("root"))
        candidates = root.get("candidates")
        labels: list[dict[str, Any]] = []
        for index in range(count):
            candidate = _mapping(candidates[index]) if isinstance(candidates, list) and index < len(candidates) else {}
            target = _mapping(candidate.get("target"))
            branch = _text(candidate.get("branch"), 64)
            labels.append({"candidate_index": index, "branch": branch,
                           "label": _text(target.get("id") or target.get("name") or branch, 128) or f"candidate-{index}",
                           "target_id": _text(target.get("id"), 128)})
        result = {"root_id": raw["root_id"], "seed": raw["seed"], "teacher_version": _integer(raw.get("teacher_version")),
                  "surface": _text(root.get("surface")), "candidate_count": count,
                  "same_action_gate": _text(raw.get("same_action_gate")), "passes": projected,
                  "candidate_labels": labels, "reference_candidate_index": reference["selected_root_index"],
                  "collection_elapsed_seconds": _number(raw.get("elapsed_seconds"))}
        self._remember("collection", resolved, signature, result)
        return result

    def _gameplay(self, attempt: Path, metrics: Mapping[str, Any], warnings: list[str]) -> dict[str, Any]:
        root_targets = metrics.get("root_targets")
        targets = {_text(row.get("root_id")): row for row in root_targets[:MAX_ROOTS_PER_UPDATE]
                   if isinstance(row, Mapping)} if isinstance(root_targets, list) else {}
        roots: list[dict[str, Any]] = []
        pending = False
        directories = list(islice(attempt.glob("root-*"), MAX_ROOTS_PER_UPDATE + 1))
        if len(directories) > MAX_ROOTS_PER_UPDATE:
            warnings.append("per-update gameplay root limit reached")
        for directory in sorted(directories[:MAX_ROOTS_PER_UPDATE]):
            if directory.is_symlink() or not directory.is_dir():
                continue
            collection = self._collection(directory / "collection.json", warnings)
            pending = pending or bool(collection.get("details_pending"))
            if not collection or collection.get("details_pending"):
                continue
            target = targets.get(collection["root_id"])
            if target is None or collection["teacher_version"] != _integer(metrics.get("teacher_version")):
                warnings.append("collection root/teacher differs from committed labels; no cross-root pairing")
                continue
            passes = collection["passes"]
            reference = copy.deepcopy(next(row for row in passes if row["name"] == "reference"))
            reference.update(self._reference_state(directory / "reference.jsonl", warnings))
            control = next((copy.deepcopy(row) for row in passes if row["name"] == "same-action-control"), None)
            candidates: list[dict[str, Any]] = []
            for label in collection["candidate_labels"]:
                index = label["candidate_index"]
                natural = index == collection["reference_candidate_index"]
                observed = reference if natural else next((copy.deepcopy(row) for row in passes
                    if row["selected_root_index"] == index and row["name"] not in {"reference", "same-action-control"}), None)
                values: dict[str, float | None] = {}
                for output, key in (("y", "action_values"), ("teacher_probability", "teacher_probabilities"),
                                    ("target_probability", "target_probabilities"), ("updated_probability", "updated_root_probabilities")):
                    array = target.get(key)
                    number = _number(array[index]) if isinstance(array, list) and index < len(array) else None
                    values[output] = number if output == "y" or number is None or 0 <= number <= 1 else None
                teacher, q, updated = values["teacher_probability"], values["target_probability"], values["updated_probability"]
                baseline = reference["observed_discounted_return"]
                reward = observed.get("observed_discounted_return") if observed is not None else None
                candidates.append({**label, **values, "is_reference_action": natural, "pass": observed,
                    "ordinary_training_sample": natural,
                    "observed_return_delta_vs_reference": _number(reward - baseline) if reward is not None and baseline is not None else None,
                    "target_probability_delta": q - teacher if q is not None and teacher is not None else None,
                    "updated_probability_delta": updated - teacher if updated is not None and teacher is not None else None})
            surrogate = None
            if all(row["y"] is not None and row["updated_probability_delta"] is not None for row in candidates):
                surrogate = _number(sum(row["y"] * row["updated_probability_delta"] for row in candidates))
            roots.append({key: collection[key] for key in ("root_id", "seed", "teacher_version", "surface", "candidate_count",
                                                          "same_action_gate", "reference_candidate_index", "collection_elapsed_seconds")} | {
                "natural_reference": reference, "same_action_control": control, "candidates": candidates,
                "reference_candidate_label": next((row["label"] for row in candidates if row["is_reference_action"]), None),
                "value_target": _number(target.get("value_target")), "projected_kl_to_teacher": _number(target.get("kl")),
                "updated_policy_kl_to_teacher": _number(target.get("updated_policy_kl_to_teacher")),
                "projection_kl_to_updated_policy": _number(target.get("projection_kl_to_updated_policy")),
                "expected_target_improvement": _number(target.get("expected_improvement")),
                "updated_target_surrogate_improvement": surrogate,
                "actor_enabled": _boolean(target.get("actor_enabled")), "temperature": _number(target.get("temperature")),
                "comparison_scope": "candidate alternatives at this same root; differing end floors/paths may remain",
            })
        elapsed = [root["collection_elapsed_seconds"] for root in roots]
        return {"scope": GAMEPLAY_SCOPE, "roots": roots, "details_pending": pending,
                "learner_elapsed_seconds": _number(metrics.get("elapsed_seconds")),
                "collection_elapsed_seconds": _number(sum(elapsed)) if elapsed and all(value is not None for value in elapsed) else None,
                "collection_elapsed_source": "recorded_root_collection_sum" if elapsed and all(value is not None for value in elapsed) else "not_recorded",
                "natural_reference_count": len(roots), "candidate_trajectory_count": sum(root["candidate_count"] for root in roots),
                "cross_update_comparison": "different training seeds/roots; not paired evaluation or post-update policy win rate"}

    def _invocation(self, warnings: list[str]) -> dict[str, Any]:
        directory = self.invocation_dir
        if directory is None:
            return {}
        process = self._optional_json(directory / "process.json", directory, warnings)
        launch = self._optional_json(directory / "launch.json", directory, warnings)
        exited = self._optional_json(directory / "exit.json", directory, warnings)
        argv = process.get("argv")
        # An invocation must explicitly bind to this lineage. Do not show an
        # unrelated pilot's target/PIDs when a caller selects the wrong folder.
        if not isinstance(argv, list) or "--lineage-id" not in argv:
            warnings.append("invocation identity unavailable; invocation target not used")
            return {}
        index = argv.index("--lineage-id")
        if index + 1 >= len(argv) or argv[index + 1] != self.lineage_id:
            warnings.append("invocation lineage differs; invocation target not used")
            return {}
        logs: dict[str, Any] = {}
        for name in ("stdout.log", "stderr.log"):
            path = self._safe(directory / name, directory)
            if path.is_file():
                stat = path.stat()
                logs[name] = {"bytes": stat.st_size, "modified_unix_s": stat.st_mtime}
        return {"directory": str(directory), "starting_update": _integer(process.get("starting_update")),
                "target_update": _integer(process.get("maximum_target_update", launch.get("target_update"))),
                "maximum_additional_updates": _integer(process.get("maximum_additional_updates")),
                "wsl_training_pid_recorded": _integer(process.get("pid")),
                "windows_wsl_pid_recorded": _integer(launch.get("windows_wsl_pid")),
                "started_utc": process.get("started_utc"), "exit_record_present": bool(exited),
                "exit": {key: exited[key] for key in ("returncode", "exit_code", "ended_utc", "exited_utc", "status") if key in exited},
                "process_liveness_verified": False, "logs": logs}

    def snapshot(self) -> dict[str, Any]:
        with self._lock:
            self._collection_reads = MAX_COLLECTION_READS
            self._reference_tail_reads = MAX_REFERENCE_TAIL_READS
            now = time.time()
            warnings: list[str] = []
            course = self._read_json(self.run_dir / "course.json", self.run_dir)
            if course.get("lineage_id") != self.lineage_id or _mapping(course.get("course_config")).get("version") != COURSE_VERSION:
                raise ValueError("selected course identity changed; refusing old/new-run confusion")
            config = _mapping(course.get("course_config"))
            status = self._optional_json(self.run_dir / "status.json", self.run_dir, warnings)
            attempts = list(islice((self.run_dir / "cohorts").glob("update-*-attempt-*"), MAX_ATTEMPTS + 1))
            if len(attempts) > MAX_ATTEMPTS:
                warnings.append("attempt discovery limit exceeded; history is incomplete")
            attempts = [path for path in attempts[:MAX_ATTEMPTS] if path.is_dir() and not path.is_symlink()]
            attempts.sort(key=lambda path: path.stat().st_mtime, reverse=True)
            history: list[dict[str, Any]] = []
            for attempt in attempts[:MAX_HISTORY]:
                update = self._optional_json(attempt / "update.json", self.run_dir, warnings)
                if not update:
                    continue
                metrics = _mapping(update.get("learner"))
                version = _integer(update.get("committed_update"))
                if version is None or metrics.get("version") != "sts2-fresh-macro-credit-learner-v1":
                    warnings.append(f"{attempt.name}: incompatible update metrics ignored")
                    continue
                scalar_keys = (
                    "loss", "macro_loss", "combat_loss", "parameter_delta_norm", "gradient_norm",
                    "maximum_likelihood_error", "max_autograd_steps", "backward_chunks", "policy_lag",
                    "macro_steps", "combat_steps", "cohort_storage_bytes", "elapsed_seconds",
                    "peak_cuda_allocated_bytes", "peak_cuda_reserved_bytes", "process_peak_rss_bytes",
                    "mean_updated_policy_kl_to_teacher", "mean_root_kl",
                    "macro_policy_loss", "macro_value_loss", "combat_policy_loss", "combat_value_loss", "combat_entropy",
                    "policy_head_gradient_norm", "value_head_gradient_norm", "domain_stream_gradient_norm", "shared_gradient_norm",
                )
                history.append({"update": version, "next_seed_index": _integer(update.get("next_seed_index")),
                                "attempt": attempt.name, "coverage": _mapping(update.get("coverage")),
                                "metrics": {key: _number(metrics.get(key)) for key in scalar_keys},
                                "root_targets": copy.deepcopy(metrics.get("root_targets", [])),
                                "teacher_version": _integer(metrics.get("teacher_version")),
                                "gameplay": self._gameplay(attempt, metrics, warnings) if len(history) < MAX_GAMEPLAY_HISTORY else
                                    {"scope": GAMEPLAY_SCOPE, "roots": [], "history_detail_limit_reached": True}})
            history.sort(key=lambda item: item["update"])
            raw_attempt = status.get("attempt")
            attempt_name = Path(raw_attempt.replace("\\", "/")).name if isinstance(raw_attempt, str) else None
            active = next((path for path in attempts if path.name == attempt_name), attempts[0] if attempts else None)
            progress: dict[str, Any] = {}
            latest_activity = (self.run_dir / "status.json").stat().st_mtime if (self.run_dir / "status.json").is_file() else 0.0
            if active:
                traces = [path for path in islice(active.glob("root-*/*.jsonl"), 1024)
                          if path.is_file() and not path.is_symlink() and ".prefix." not in path.name]
                if traces:
                    newest = max(traces, key=lambda path: path.stat().st_mtime)
                    stat = newest.stat()
                    latest_activity = max(latest_activity, stat.st_mtime)
                    progress = {"attempt": active.name, "trace": str(newest.relative_to(self.run_dir)),
                                "trace_bytes": stat.st_size, "trace_modified_unix_s": stat.st_mtime,
                                "trace_count": len(traces)}
                    try:
                        progress.update(self._tail_progress(newest))
                    except (OSError, ValueError):
                        warnings.append("latest trace is unavailable or oversized; progress inferred from file stat only")
            latest = history[-1] if history else None
            committed = _integer(status.get("committed_update"))
            if committed is None and latest:
                committed = latest["update"]
            cursor = _integer(status.get("next_seed_index"))
            if cursor is None and latest:
                cursor = latest["next_seed_index"]
            writer = (self.run_dir / ".writer.lock").exists()
            phase = str(status.get("state", "status_unavailable"))
            if phase in {"collecting", "updating", "initializing"} and not writer:
                warnings.append("active phase is recorded but writer lock is absent; process may have stopped")
            if committed is not None and latest and committed > latest["update"]:
                warnings.append("committed status is ahead of visible update metrics; checkpoint remains recovery authority")
            invocation = self._invocation(warnings)
            return {"schema": SCHEMA, "read_only": True, "server_pid": os.getpid(),
                    "observed_at_utc": datetime.now(UTC).isoformat(), "run_dir": str(self.run_dir),
                    "lineage_id": self.lineage_id, "course_version": COURSE_VERSION,
                    "state": phase, "committed_update": committed, "next_seed_index": cursor,
                    "next_training_seed": 2 * (6700000 + cursor) if cursor is not None else None,
                    "writer_lock_present": writer, "process_liveness_verified": False,
                    "activity_age_seconds": max(0.0, now - latest_activity) if latest_activity else None,
                    "error_type": status.get("error_type"),
                    "error": str(status["error"])[:4096] if status.get("error") is not None else None,
                    "invocation": invocation,
                    "root_surface": _mapping(config.get("branches")).get("root_surface"),
                    "macro_horizon": _mapping(config.get("branches")).get("continuation_macro_decisions"),
                    "revival_budget": _mapping(course.get("reward_identity")).get("revival_budget"),
                    "device": _mapping(_mapping(config.get("training")).get("runtime")).get("device"),
                    "progress": progress, "latest_update": latest, "history": history,
                    "history_limit": MAX_HISTORY, "attempts_observed": len(attempts), "warnings": warnings,
                    "gameplay_history_limit": MAX_GAMEPLAY_HISTORY,
                    "gameplay_limits": {"roots_per_update": MAX_ROOTS_PER_UPDATE, "candidates_per_root": MAX_CANDIDATES,
                                        "collection_reads_per_poll": MAX_COLLECTION_READS,
                                        "new_reference_tail_reads_per_poll": MAX_REFERENCE_TAIL_READS,
                                        "reference_tail_bytes": MAX_TAIL_BYTES},
                    "measurement_scope": {"gpu_peaks": "learner window only; not full-course peak",
                                          "rss": "process historical high-water mark", "quality_certified": False,
                                          "gameplay": GAMEPLAY_SCOPE,
                                          "observed_return": "post-root observed discounted reward, excluding bootstrap V",
                                          "steps": "native/semantic totals include reset prefix; branch/continuation counts are post-root",
                                          "revivals_and_hp": "exact counters at window end since reset; not window increments",
                                          "version": "collection is pre-update teacher behavior; committed update is training progress",
                                          "comparisons": "same-root alternatives only; no across-seed paired improvement or full-episode win rate",
                                          "reward_components": "progress_reward includes the potential_reward alias exactly once"}}


class MacroDashboardServer(ThreadingHTTPServer):
    daemon_threads = True

    def __init__(self, address: tuple[str, int], store: MacroDashboardStore) -> None:
        self.store = store
        shared = (Path(__file__).parent / "dashboard_static/index.html").read_bytes()
        if b"<body>" not in shared:
            raise ValueError("shared dashboard template has no unambiguous body entry")
        self.html = shared.replace(b"<body>", b'<body data-dashboard-mode="macro-credit">', 1)
        super().__init__(address, MacroDashboardHandler)


class MacroDashboardHandler(BaseHTTPRequestHandler):
    server: MacroDashboardServer
    server_version = "STS2MacroCreditMonitor/1"
    sys_version = ""

    def log_message(self, format: str, *args: Any) -> None:
        pass  # No per-poll output or raw request logging.

    def _send(self, status: int, value: dict[str, Any] | bytes, *, head: bool = False) -> None:
        is_html = isinstance(value, bytes)
        content = value if isinstance(value, bytes) else json.dumps(value, ensure_ascii=False, allow_nan=False).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "text/html; charset=utf-8" if is_html else "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(content)))
        self.send_header("Cache-Control", "no-store")
        self.send_header("X-Content-Type-Options", "nosniff")
        self.send_header("X-Frame-Options", "DENY")
        self.send_header("Referrer-Policy", "no-referrer")
        self.send_header("Content-Security-Policy", _CSP)
        self.end_headers()
        if not head:
            self.wfile.write(content)

    def _get(self, *, head: bool = False) -> None:
        try:
            host = urlsplit("//" + self.headers.get("Host", "")).hostname or ""
        except ValueError:
            host = ""
        if not _loopback(host):
            self._send(HTTPStatus.FORBIDDEN, {"error": "loopback Host required"}, head=head)
            return
        route = urlsplit(self.path)
        if len(self.path) > 2048 or route.query:
            self._send(HTTPStatus.BAD_REQUEST, {"error": "no query or file selection is supported"}, head=head)
        elif route.path in {"/", "/index.html"}:
            self._send(HTTPStatus.OK, self.server.html, head=head)
        elif route.path in {"/health", "/api/v1/health"}:
            self._send(HTTPStatus.OK, {"schema": SCHEMA, "status": "ok", "read_only": True,
                                      "run_dir": str(self.server.store.run_dir), "server_pid": os.getpid(),
                                      "lineage_id": self.server.store.lineage_id}, head=head)
        elif route.path == "/api/v1/snapshot":
            try:
                self._send(HTTPStatus.OK, self.server.store.snapshot(), head=head)
            except (OSError, ValueError) as error:
                self._send(HTTPStatus.SERVICE_UNAVAILABLE, {"schema": SCHEMA, "error": type(error).__name__,
                                                          "message": "selected-run metadata is unavailable; no fallback run"}, head=head)
        else:
            self._send(HTTPStatus.NOT_FOUND, {"error": "route not found"}, head=head)

    def do_GET(self) -> None:
        self._get()

    def do_HEAD(self) -> None:
        self._get(head=True)

    def do_POST(self) -> None:
        self._send(HTTPStatus.METHOD_NOT_ALLOWED, {"error": "read-only dashboard; no training controls"})

    do_PUT = do_POST
    do_PATCH = do_POST
    do_DELETE = do_POST
    do_OPTIONS = do_POST


def make_server(run_dir: Path, *, artifact_root: Path, invocation_dir: Path | None = None,
                host: str = "127.0.0.1", port: int = 8766) -> MacroDashboardServer:
    if host not in {"127.0.0.1", "localhost"}:
        raise ValueError("macro dashboard binds IPv4 loopback only")
    return MacroDashboardServer((host, port), MacroDashboardStore(run_dir, artifact_root=artifact_root,
                                                                invocation_dir=invocation_dir))


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--artifact-root", type=Path, required=True)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--invocation-dir", type=Path)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8766)
    args = parser.parse_args(argv)
    if not 1 <= args.port <= 65535:
        parser.error("port must be between 1 and 65535")
    server = make_server(args.run_dir, artifact_root=args.artifact_root, invocation_dir=args.invocation_dir,
                         host=args.host, port=args.port)
    print(json.dumps({"url": f"http://127.0.0.1:{args.port}/", "schema": SCHEMA,
                      "run_dir": str(server.store.run_dir), "pid": os.getpid(), "read_only": True}), flush=True)
    try:
        server.serve_forever(poll_interval=0.5)
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
