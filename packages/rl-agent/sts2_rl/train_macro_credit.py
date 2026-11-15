"""Synchronous fresh-teacher macro-credit training; cohort-boundary recovery only."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import random
import shutil
import subprocess
import threading
import time
import tomllib
from collections.abc import Callable, Iterator, Mapping, Sequence
from contextlib import ExitStack, contextmanager
from dataclasses import asdict, dataclass, replace
from pathlib import Path
from typing import Any
from uuid import uuid4

import numpy as np
import torch

from sts2_baseline import act_frontloaded_death_reward_identity
from sts2_env.headless_sim_bridge_client import resolve_headless_sim_exe
from sts2_rl.artifacts import artifact_root, resolve_artifact_path, resolve_external_input_path
from sts2_rl.backends import HeadlessBackend
from sts2_rl.contracts import EnvironmentBackend
from sts2_rl.encoding import EncodedDecisionSnapshot, grounding_encoding_identity
from sts2_rl.encoding.snapshot import collate_encoded_snapshots
from sts2_rl.joint.macro_checkpoint import (
    MacroCheckpointIdentity,
    load_macro_checkpoint,
    migrate_macro_checkpoint_runtime,
    publish_macro_checkpoint,
    save_macro_checkpoint,
)
from sts2_rl.joint.macro_credit_collection import (
    collect_durable_afterstate_cohort,
    collect_macro_credit_root,
)
from sts2_rl.joint.macro_credit_data import (
    MACRO_CREDIT_DATA_VERSION,
    MacroCounterfactualRoot,
    MacroCreditCohort,
    MacroCreditTrajectory,
    training_seed,
    trajectory_storage_nbytes,
)
from sts2_rl.joint.macro_credit_learner import (
    MACRO_CREDIT_LEARNER_VERSION,
    MacroCreditLearner,
    MacroCreditLearnerConfig,
)
from sts2_rl.joint.parallel_collection import PersistentRolloutWorkerPool
from sts2_rl.macro_branch_diagnostic import BranchLimits
from sts2_rl.macro_root_diagnostic import FrozenPolicy, ProbeBudgetExhausted, validate_course
from sts2_rl.macro_root_evidence import canonical_facts
from sts2_rl.macro_root_policy import FrozenMacroRootPolicy
from sts2_rl.models.hybrid_actor_critic import HybridActorCriticConfig, HybridDomainActorCriticModel
from sts2_rl.semantics.clock import DECISION_CLOCK_CONTRACT_VERSION
from sts2_rl.semantics.forward import FORWARD_COMPILER_CONTRACT_VERSION
from sts2_rl.simulator_identity import verify_headless_simulator
from sts2_rl.training.config import TrainingConfig, load_training_config, training_config_from_mapping
from sts2_rl.training.sdpa import configure_rocm_sdpa_backend

MACRO_COURSE_VERSION = "sts2-synchronous-macro-credit-course-v1"
TRAINING_SEED_BASE = 6_700_000
DEFAULT_CONFIG = Path(__file__).resolve().parents[1] / "config/experiments/hybrid_macro_credit_v1_reward_v9.toml"


@dataclass(frozen=True, slots=True)
class CourseLimits:
    roots_per_cohort: int = 1
    seed_index_offset: int = 0
    parallel_collectors: int = 1
    macro_branch_epsilon: float = 0.0
    maximum_root_snapshot_bytes: int = 256 * 1024 * 1024
    maximum_cohort_snapshot_bytes: int = 512 * 1024 * 1024
    maximum_root_seconds: float = 3600.0
    maximum_cohort_seconds: float = 7200.0
    maximum_output_bytes: int = 64 * 1024 * 1024 * 1024
    minimum_free_disk_bytes: int = 1024 * 1024 * 1024

    def __post_init__(self) -> None:
        for name, value in asdict(self).items():
            if name == "seed_index_offset":
                if isinstance(value, bool) or not isinstance(value, int) or value < 0:
                    raise ValueError("course.seed_index_offset must be a non-negative integer")
                continue
            if name == "macro_branch_epsilon":
                if isinstance(value, bool) or not isinstance(value, int | float) or not math.isfinite(value) or not 0.0 <= value < 1.0:
                    raise ValueError("course.macro_branch_epsilon must be finite and in [0.0, 1.0)")
                continue
            if name.endswith("seconds"):
                if isinstance(value, bool) or not isinstance(value, int | float) or not math.isfinite(value) or value <= 0:
                    raise ValueError(f"course.{name} must be positive and finite")
            elif isinstance(value, bool) or not isinstance(value, int) or value <= 0:
                raise ValueError(f"course.{name} must be a positive integer")
        if self.maximum_root_snapshot_bytes > self.maximum_cohort_snapshot_bytes:
            raise ValueError("root snapshot budget must fit the cohort budget")
        if self.maximum_root_seconds > self.maximum_cohort_seconds:
            raise ValueError("root time budget must fit the cohort budget")


@dataclass(frozen=True, slots=True)
class MacroTrainingConfig:
    training: TrainingConfig
    course: CourseLimits
    branches: BranchLimits
    learner: MacroCreditLearnerConfig

    def __post_init__(self) -> None:
        course_budget = validate_course(self.training)
        if (
            self.learner.enable_asymmetric_revival_critic
            and self.learner.revival_budget != course_budget
        ):
            raise ValueError(
                f"learner.revival_budget ({self.learner.revival_budget}) must equal "
                f"training.curriculum.revival_budget ({course_budget}) when "
                "enable_asymmetric_revival_critic=True"
            )
        if self.training.optimization.discount != 1.0:
            raise ValueError("macro course requires undiscounted native reward-v9 edges")
        if self.training.transaction_learning.enabled or self.training.episodic_learning.enabled:
            raise ValueError("macro course cannot enable legacy replay sidecars")
        if self.training.failure_credit.mode != "disabled":
            raise ValueError("macro course cannot enable legacy failure-credit learning")
        if not self.branches.terminal_only and self.branches.continuation_macro_decisions < 1:
            raise ValueError("macro course needs a positive macro continuation horizon")
        if self.learner.target_mode == "durable_afterstate_v2" and not self.branches.terminal_only:
            raise ValueError("durable_afterstate_v2 requires branches.terminal_only=True")
        if self.branches.max_candidates > self.training.model.max_candidates:
            raise ValueError("branch candidate budget exceeds the model candidate capacity")

    def as_mapping(self) -> dict[str, Any]:
        result: dict[str, Any] = canonical_facts({
            "version": MACRO_COURSE_VERSION,
            "training": self.training.to_mapping(),
            "course": asdict(self.course),
            "branches": asdict(self.branches),
            "learner": asdict(self.learner),
            "seed_recipe": {"base": TRAINING_SEED_BASE, "multiplier": 2, "partition": "training-even",
                            "index_offset": self.course.seed_index_offset},
        })
        return result


def _merge_known(base: dict[str, Any], override: Mapping[str, Any], prefix: str = "") -> None:
    for key, value in override.items():
        if key not in base:
            raise ValueError(f"unknown macro course config field {prefix}{key}")
        if isinstance(base[key], dict):
            if not isinstance(value, Mapping):
                raise ValueError(f"macro course config {prefix}{key} must be a table")
            _merge_known(base[key], value, f"{prefix}{key}.")
        else:
            if isinstance(value, Mapping):
                raise ValueError(f"macro course config {prefix}{key} is not a table")
            base[key] = value


def load_macro_training_config(path: Path, overrides: Sequence[str] = ()) -> MacroTrainingConfig:
    raw = tomllib.loads(path.read_text(encoding="utf-8"))
    if raw.get("version") != MACRO_COURSE_VERSION:
        raise ValueError("macro course config version differs")
    payload: dict[str, Any] = {
        "version": MACRO_COURSE_VERSION,
        "training": load_training_config(profile="preheat").to_mapping(),
        "course": asdict(CourseLimits()),
        "branches": asdict(BranchLimits(continuation_macro_decisions=12, max_candidates=16,
                                       max_native_steps=8192, timeout_seconds=1200.0)),
        "learner": asdict(MacroCreditLearnerConfig()),
    }
    _merge_known(payload, raw)
    for override in overrides:
        if "=" not in override:
            raise ValueError("--set requires dotted.key=TOML_VALUE")
        key, value = override.split("=", 1)
        parsed = tomllib.loads(f"value = {value}")["value"]
        nested: Any = parsed
        for component in reversed(key.split(".")):
            nested = {component: nested}
        _merge_known(payload, nested)
    if payload["version"] != MACRO_COURSE_VERSION:
        raise ValueError("macro course config version differs")
    return MacroTrainingConfig(
        training=training_config_from_mapping(payload["training"]),
        course=CourseLimits(**payload["course"]),
        branches=BranchLimits(**payload["branches"]),
        learner=MacroCreditLearnerConfig(**payload["learner"]),
    )


def seed_at(cursor: int) -> int:
    if isinstance(cursor, bool) or not isinstance(cursor, int) or cursor < 0:
        raise ValueError("training seed cursor must be a non-negative integer")
    seed = 2 * (TRAINING_SEED_BASE + cursor)
    training_seed(seed)
    return seed


def source_manifest() -> dict[str, Any]:
    """A declared partial source manifest, not a complete Python dependency lock."""
    package = Path(__file__).resolve().parent
    selected = (
        "train_macro_credit.py", "joint/afterstate.py", "joint/macro_credit_data.py", "joint/macro_credit_collection.py",
        "joint/parallel_collection.py", "joint/macro_credit_learner.py", "joint/macro_returns.py", "joint/macro_targets.py",
        "joint/macro_checkpoint.py", "models/hybrid_actor_critic.py", "models/hybrid_candidate_q.py",
        "models/grounded_candidate.py", "macro/authority.py", "macro_root_policy.py",
        "macro_root_diagnostic.py", "macro_branch_diagnostic.py", "macro_root_evidence.py",
        "semantics/forward.py", "semantics/clock.py", "encoding/grounded.py", "encoding/snapshot.py",
    )
    records = [{"path": f"sts2_rl/{name}", "sha256": hashlib.sha256((package / name).read_bytes()).hexdigest()}
               for name in selected]
    try:
        git_head = subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=package, text=True, stderr=subprocess.DEVNULL,
        ).strip()
    except (OSError, subprocess.CalledProcessError):
        git_head = None
    return {"identity": {"scope": "partial-macro-course-and-selected-direct-dependencies",
                         "complete_dependency_lock": False, "files": records},
            "git_head_provenance_only": git_head}


def explicit_device(value: str) -> torch.device:
    if value == "cpu":
        return torch.device("cpu")
    if value != "cuda" and not (value.startswith("cuda:") and value[5:].isdigit()):
        raise ValueError("--device must explicitly select cpu, cuda, or cuda:N; no automatic fallback")
    if not torch.cuda.is_available():
        raise ValueError("requested CUDA/ROCm is unavailable; CPU fallback is forbidden")
    index = 0 if value == "cuda" else int(value[5:])
    if index >= torch.cuda.device_count():
        raise ValueError("requested CUDA/ROCm device index is unavailable")
    return torch.device(f"cuda:{index}")


def _actor_critic_config(config: MacroTrainingConfig) -> HybridActorCriticConfig:
    grounded = config.training.model.to_model_config()
    if config.learner.target_mode == "durable_afterstate_v2":
        return HybridActorCriticConfig(
            grounded=grounded,
            enable_hierarchical_branch_policy=True,
            enable_durable_value_head=True,
            enable_persistent_build_context=True,
            enable_direct_vitals_context=config.learner.enable_direct_vitals_context,
            enable_macro_branch_prior=config.learner.enable_macro_branch_prior,
            enable_persistent_only_durable_value=config.learner.enable_persistent_only_durable_value,
            enable_asymmetric_revival_critic=config.learner.enable_asymmetric_revival_critic,
            enable_draw_weighted_durable_value=config.learner.enable_draw_weighted_durable_value,
            enable_role_conditioned_candidate_policy=(
                config.learner.enable_role_conditioned_candidate_policy
            ),
            enable_deconfounded_durable_value=config.learner.enable_deconfounded_durable_value,
            enable_monotone_deconfounded_durable_value=(
                config.learner.enable_monotone_deconfounded_durable_value
            ),
        )
    return HybridActorCriticConfig(grounded)


class InMemoryTeacher:
    """The learner's unchanged model, inference-only for the whole cohort."""

    def __init__(self, model: HybridDomainActorCriticModel, config: TrainingConfig, device: torch.device) -> None:
        self.model = model
        self.encoding_config = config.model.to_encoding_config()
        self.device = device
        self.hierarchical_branch_greedy = bool(model.actor_critic_config.enable_hierarchical_branch_policy)
        self._lock = threading.Lock()

    def initial_state(self) -> torch.Tensor:
        with self._lock, torch.inference_mode():
            return self.model.initial_state(1, device=self.device).detach()

    def forward(self, snapshot: EncodedDecisionSnapshot, hidden: torch.Tensor | None) -> tuple[torch.Tensor, torch.Tensor]:
        with self._lock:
            encoded = collate_encoded_snapshots(
                (snapshot,), expected_config=self.encoding_config,
                expected_fingerprint=snapshot.encoding_fingerprint, device=self.device,
            )
            with torch.inference_mode():
                initial = self.model.initial_state(1, device=self.device).detach() if hidden is None else hidden
                output = self.model(
                    encoded,
                    initial,
                    compute_durable_value=False,
                    validate=False,
                )
                return output.policy_logits[0].detach(), output.recurrent_state.detach()


def _json(path: Path, value: Mapping[str, Any]) -> None:
    temporary = path.with_name(f".{path.name}.{uuid4().hex}.tmp")
    try:
        with temporary.open("x", encoding="utf-8") as handle:
            json.dump(value, handle, ensure_ascii=False, allow_nan=False, indent=2)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


@contextmanager
def output_owner(output: Path, *, resume: bool, allow_existing_without_checkpoint: bool = False) -> Iterator[None]:
    if resume:
        if not output.is_dir():
            raise ValueError("resume requires the existing experiment output directory")
    elif allow_existing_without_checkpoint:
        output.mkdir(parents=True, exist_ok=True)
        if (output / "checkpoint.pt").exists():
            raise FileExistsError("runtime migration refuses to overwrite an existing checkpoint.pt")
    else:
        output.mkdir(parents=True, exist_ok=False)
    lock = output / ".writer.lock"
    descriptor = os.open(lock, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            json.dump({"pid": os.getpid(), "cohort_boundary_only": True}, handle)
            handle.flush()
            os.fsync(handle.fileno())
        try:
            yield
        except BaseException as error:
            status_path = output / "status.json"
            previous = json.loads(status_path.read_text(encoding="utf-8")) if status_path.is_file() else {}
            if previous.get("state") != "aborted":
                _json(status_path, {"state": "aborted", "error_type": type(error).__name__, "error": str(error),
                                    "recovery": "checkpoint is authority; no mid-run resume"})
            raise
    finally:
        lock.unlink(missing_ok=True)


class ResourceGuard:
    def __init__(self, output: Path, limits: CourseLimits) -> None:
        self.output = output
        self.limits = limits
        self.started = time.monotonic()
        self.last_disk_check = float("-inf")

    def check(self, force: bool = False) -> None:
        now = time.monotonic()
        if now - self.started > self.limits.maximum_cohort_seconds:
            raise ProbeBudgetExhausted("maximum_cohort_seconds")
        if force or now - self.last_disk_check >= 0.5:
            size = sum(path.stat().st_size for path in self.output.rglob("*") if path.is_file())
            if size > self.limits.maximum_output_bytes:
                raise ProbeBudgetExhausted("maximum_output_bytes")
            if shutil.disk_usage(self.output).free < self.limits.minimum_free_disk_bytes:
                raise ProbeBudgetExhausted("minimum_free_disk_bytes")
            self.last_disk_check = now


def run_cohorts(
    *, config: MacroTrainingConfig, output: Path, identity: MacroCheckpointIdentity,
    learner: MacroCreditLearner, teacher: FrozenPolicy, backend_factory: Callable[[], EnvironmentBackend],
    initial_seed_index: int, initialization: Mapping[str, Any], max_updates: int,
    initialize_boundary: bool, root: Path, source_paths: Sequence[Path] = (),
    worker_pool: PersistentRolloutWorkerPool | None = None,
) -> dict[str, Any]:
    """Collect all branches, learn once, commit once, then publish; never overlap."""
    if isinstance(max_updates, bool) or not isinstance(max_updates, int) or max_updates <= 0:
        raise ValueError("max_updates must be a positive integer")
    if isinstance(learner.update, bool) or not isinstance(learner.update, int) or learner.update < 0:
        raise ValueError("learner update must be a non-negative integer")
    seed_at(config.course.seed_index_offset + initial_seed_index)
    seed_at(config.course.seed_index_offset + initial_seed_index + max_updates * config.course.roots_per_cohort - 1)
    update_base = int(initialization.get("update_base_at_migration", 0))
    cursor_base = int(initialization.get("cursor_base_at_migration", 0))
    if update_base < 0 or cursor_base < 0 or learner.update < update_base:
        raise ValueError("invalid migration update/cursor baseline in initialization")
    delta_updates = learner.update - update_base
    expected_min_cursor = cursor_base + delta_updates * config.course.roots_per_cohort
    if config.learner.target_mode == "durable_afterstate_v2":
        max_extra = max(2, config.course.roots_per_cohort // 2)
        valid_boundary = (
            initial_seed_index == cursor_base
            if delta_updates == 0
            else expected_min_cursor
            <= initial_seed_index
            <= cursor_base + delta_updates * (config.course.roots_per_cohort + max_extra)
        )
    else:
        valid_boundary = initial_seed_index == expected_min_cursor
    if not valid_boundary:
        raise ValueError("committed update and seed cursor do not describe a complete cohort boundary")
    output = resolve_artifact_path(output, root=root)
    checkpoint = output / "checkpoint.pt"
    publication = output / "publication.pt"
    cursor = initial_seed_index
    committed_update = learner.update
    start_update = committed_update

    def save_boundary() -> None:
        save_macro_checkpoint(
            checkpoint, identity=identity, model=learner.model, optimizer=learner.optimizer,
            update=learner.update, next_seed_index=cursor, initialization=initialization,
            artifact_root=root, source_paths=source_paths,
        )

    def publish() -> None:
        publish_macro_checkpoint(checkpoint, publication, identity=identity, artifact_root=root, source_paths=source_paths)

    try:
        if initialize_boundary:
            is_migration = initialization.get("mode") == "runtime_migration_from_macro_checkpoint"
            if checkpoint.exists() or (not is_migration and (learner.update != 0 or cursor != 0)):
                raise ValueError("new lineage requires an absent update-zero checkpoint and seed cursor zero")
            save_boundary()
        publish()  # On resume, repair stale/missing derived publication first.
        for _ in range(max_updates):
            guard = ResourceGuard(output, config.course)
            guard.check(force=True)
            learner.model.eval()
            cohort_dir = output / "cohorts" / f"update-{learner.update:06d}-attempt-{uuid4().hex}"
            cohort_dir.mkdir(parents=True, exist_ok=False)
            roots: list[MacroCounterfactualRoot] = []
            natural_only: list[MacroCreditTrajectory] = []
            training_episodes: list[dict[str, Any]] = []
            _json(output / "status.json", {"state": "collecting", "committed_update": committed_update,
                                          "next_seed_index": cursor, "attempt": str(cohort_dir)})
            collection_started = time.monotonic()
            consumed_seeds = config.course.roots_per_cohort
            if config.learner.target_mode == "durable_afterstate_v2":
                seeds = [
                    seed_at(config.course.seed_index_offset + cursor + offset)
                    for offset in range(config.course.roots_per_cohort)
                ]

                def replacement_seed(attempt_offset: int, *, base_cursor: int = cursor) -> int:
                    nonlocal consumed_seeds
                    consumed_seeds = max(consumed_seeds, attempt_offset + 1)
                    return seed_at(config.course.seed_index_offset + base_cursor + attempt_offset)

                remaining_seconds = config.course.maximum_cohort_seconds - (time.monotonic() - guard.started)
                cohort, training_episodes, _ = collect_durable_afterstate_cohort(
                    backend_factory=backend_factory,
                    policy=teacher,
                    config=config.training,
                    seeds=seeds,
                    teacher_version=learner.update,
                    limits=config.branches,
                    cohort_dir=cohort_dir,
                    cohort_id=cohort_dir.name,
                    macro_branch_epsilon=config.course.macro_branch_epsilon,
                    allow_partial_afterstates=config.learner.enable_partial_afterstates,
                    allow_single_pick_manual_confirm=(
                        config.learner.enable_manual_confirm_selection_afterstates
                    ),
                    parallel_collectors=config.course.parallel_collectors,
                    replacement_seed_fn=replacement_seed,
                    max_recoverable_incidents=max(2, config.course.roots_per_cohort // 2),
                    maximum_episode_snapshot_bytes=config.course.maximum_root_snapshot_bytes,
                    maximum_cohort_snapshot_bytes=config.course.maximum_cohort_snapshot_bytes,
                    maximum_episode_seconds=config.course.maximum_root_seconds,
                    maximum_cohort_seconds=remaining_seconds,
                    artifact_root=root,
                    resource_guard=guard.check,
                    worker_pool=worker_pool,
                )
                natural_only = list(cohort.natural_only)
            else:
                for offset in range(config.course.roots_per_cohort):
                    guard.check(force=True)
                    current_bytes = sum(
                        MacroCreditCohort(f"measure-{index}", learner.update, (item,)).storage_nbytes()
                        for index, item in enumerate(roots)
                    )
                    remaining_bytes = (config.course.maximum_cohort_snapshot_bytes - current_bytes
                                       - trajectory_storage_nbytes(natural_only))
                    if remaining_bytes <= 0:
                        raise ProbeBudgetExhausted("maximum_cohort_snapshot_bytes")
                    remaining_seconds = config.course.maximum_cohort_seconds - (time.monotonic() - guard.started)
                    collected, _report = collect_macro_credit_root(
                        backend_factory=backend_factory, policy=teacher, config=config.training,
                        seed=seed_at(config.course.seed_index_offset + cursor + offset), teacher_version=learner.update,
                        limits=config.branches,
                        output_dir=cohort_dir / f"root-{offset:03d}",
                        maximum_snapshot_bytes=min(config.course.maximum_root_snapshot_bytes, remaining_bytes),
                        maximum_group_seconds=min(config.course.maximum_root_seconds, remaining_seconds),
                        artifact_root=root, resource_guard=guard.check,
                    )
                    if isinstance(collected, MacroCreditTrajectory):
                        natural_only.append(collected)
                    else:
                        roots.append(collected)
                    episode = _report.get("training_episode")
                    if config.branches.terminal_only and (not isinstance(episode, dict) or episode.get("status") != "complete"):
                        raise ValueError("full-run training requires a typed complete natural episode")
                    if isinstance(episode, dict):
                        training_episodes.append(episode)
                    guard.check(force=True)
                cohort = (MacroCreditCohort(cohort_dir.name, learner.update, tuple(roots), natural_only=tuple(natural_only))
                          if natural_only else MacroCreditCohort(cohort_dir.name, learner.update, tuple(roots)))
            if cohort.storage_nbytes() > config.course.maximum_cohort_snapshot_bytes:
                raise ProbeBudgetExhausted("maximum_cohort_snapshot_bytes")
            references = [item.reference for item in roots] + natural_only
            coverage = {
                "natural_macro_steps": sum(step.control_domain == "macro" for item in references for step in item.steps),
                "natural_combat_steps": sum(step.control_domain == "combat" for item in references for step in item.steps),
                "afterstate_steps": sum(len(getattr(item, "afterstate_nodes", ())) for item in references),
                "complete_roots": len(roots),
                "natural_only_games": len(natural_only),
                "candidate_trajectories": sum(len(item.candidates) for item in roots),
            }
            missing_domain = coverage["natural_macro_steps"] < 1 or coverage["natural_combat_steps"] < 1
            if missing_domain and not (config.branches.terminal_only and natural_only):
                raise ValueError("cohort requires natural macro and combat coverage before its joint update")
            collection_elapsed_s = time.monotonic() - collection_started
            _json(cohort_dir / "collected.json", {"training_episodes": training_episodes, "coverage": coverage,
                                                  "elapsed_seconds": collection_elapsed_s})
            _json(output / "status.json", {"state": "updating", "committed_update": committed_update,
                                          "next_seed_index": cursor, "attempt": str(cohort_dir), "coverage": coverage})
            guard.check(force=True)
            metrics = learner.learn(cohort, resource_guard=guard.check)
            if learner.update != committed_update + 1:
                raise RuntimeError("learner must perform exactly one update per completed cohort")
            guard.check(force=True)
            next_cursor = cursor + consumed_seeds
            previous_cursor = cursor
            cursor = next_cursor
            try:
                save_boundary()
            except BaseException:
                cursor = previous_cursor
                raise
            committed_update = learner.update
            publish()
            _json(cohort_dir / "update.json", {"committed_update": committed_update,
                                              "next_seed_index": cursor, "learner": metrics, "coverage": coverage,
                                              "training_episodes": training_episodes})
            guard.check(force=True)
        summary = {"state": "bounded_training_complete", "committed_update": committed_update,
                   "updates_this_invocation": committed_update - start_update, "next_seed_index": cursor,
                   "checkpoint": str(checkpoint), "publication": str(publication),
                   "quality_certified": False, "long_run_certified": False,
                   "note": "bounded execution is not a held-out quality/resource certification"}
        _json(output / "status.json", summary)
        return summary
    except BaseException as error:
        _json(output / "status.json", {"state": "aborted", "committed_update": committed_update,
                                      "next_seed_index": cursor, "error_type": type(error).__name__,
                                      "error": str(error), "recovery": "inspect checkpoint; recollect uncommitted cohort"})
        raise


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--lineage-id", required=True)
    parser.add_argument("--device", required=True)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--initialize-from-iql", type=Path)
    mode.add_argument("--initialize-from-macro-publication", type=Path,
                      help="explicit model-only new lineage; optimizer/RNG/update are reset, never exact resume")
    mode.add_argument("--migrate-runtime-from-checkpoint", type=Path,
                      help="preserve model, optimizer, RNG, update, and seed cursor across runtime/throughput upgrades")
    mode.add_argument("--resume", type=Path)
    parser.add_argument("--sim-exe", type=Path)
    parser.add_argument("--max-updates", type=int, default=1, help="maximum committed updates in this invocation")
    parser.add_argument("--set", action="append", default=[], dest="overrides", metavar="KEY=TOML_VALUE")
    parser.add_argument("--dry-run", action="store_true", help="read-only static preflight, never training readiness")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    output: Path | None = None
    try:
        if args.max_updates <= 0:
            raise ValueError("--max-updates must be positive")
        config = load_macro_training_config(args.config, args.overrides)
        root = artifact_root()
        output = resolve_artifact_path(args.output_dir, root=root)
        device = explicit_device(args.device)
        if not args.lineage_id.strip():
            raise ValueError("--lineage-id must be non-empty")
        source = resolve_external_input_path(
            args.initialize_from_iql
            or args.initialize_from_macro_publication
            or args.migrate_runtime_from_checkpoint
            or args.resume,
            root=root,
        )
        if not source.exists():
            raise FileNotFoundError("initialization/resume source does not exist")
        if args.resume:
            if source != output / "checkpoint.pt":
                raise ValueError("resume must use this experiment's own checkpoint.pt; forks need a new initialization")
            if not output.is_dir():
                raise ValueError("resume output directory is missing")
        elif args.migrate_runtime_from_checkpoint:
            if source == output / "checkpoint.pt":
                raise ValueError("runtime migration source must be a separate backup checkpoint")
            if (output / "checkpoint.pt").exists():
                raise FileExistsError("runtime migration refuses an existing checkpoint.pt in output directory")
        elif output.exists():
            raise FileExistsError("new macro lineage refuses an existing output directory")
        if (output / ".writer.lock").exists():
            raise FileExistsError("output has a writer lock; verify stopped process before manual stale-lock removal")
        simulator = verify_headless_simulator(resolve_headless_sim_exe(args.sim_exe or config.training.environment.sim_exe_path))
        training = replace(
            config.training,
            environment=replace(config.training.environment, sim_exe_path=str(simulator.executable)),
            runtime=replace(config.training.runtime, device=str(device), collector_device=str(device)),
        )
        config = replace(config, training=training)
        ac_config = _actor_critic_config(config)
        sources = source_manifest()
        course_identity = {**config.as_mapping(), "simulator": simulator.to_mapping(),
                           "encoding": grounding_encoding_identity(), "compiler": FORWARD_COMPILER_CONTRACT_VERSION,
                           "decision_clock": DECISION_CLOCK_CONTRACT_VERSION, "data": MACRO_CREDIT_DATA_VERSION,
                           "source_identity": sources["identity"]}
        identity = MacroCheckpointIdentity(
            course_config=course_identity, model_config=ac_config.as_mapping(),
            reward_identity=act_frontloaded_death_reward_identity(revival_budget=validate_course(training)),
            lineage_id=args.lineage_id, learner_version=MACRO_CREDIT_LEARNER_VERSION,
        )
        identity.as_mapping()
        if args.dry_run:
            print(json.dumps({"state": "preflight_only", "training_ready": False, "config_validated": True,
                              "simulator_verified": True, "source_weights_validated": False,
                              "next_training_seed_if_new": seed_at(config.course.seed_index_offset),
                              "course": config.as_mapping()}, allow_nan=False))
            return 0
        with output_owner(
            output,
            resume=bool(args.resume),
            allow_existing_without_checkpoint=bool(args.migrate_runtime_from_checkpoint),
        ):
            _json(output / "status.json", {"state": "initializing", "lineage_id": args.lineage_id})
            configure_rocm_sdpa_backend(training.runtime.rocm_sdpa_backend, devices=(device,))
            if args.initialize_from_iql or args.initialize_from_macro_publication:
                random.seed(TRAINING_SEED_BASE)
                np.random.seed(TRAINING_SEED_BASE)
                torch.manual_seed(TRAINING_SEED_BASE)
                source_hash = hashlib.sha256(source.read_bytes()).hexdigest()
                loaded = FrozenMacroRootPolicy(
                    source,
                    training,
                    str(device),
                    initialize_from_iql=bool(args.initialize_from_iql),
                    initialize_from_macro_publication=bool(args.initialize_from_macro_publication),
                    actor_critic_config=ac_config,
                )
                if args.initialize_from_macro_publication and loaded.provenance["source_kind"] != "hybrid_macro_credit_publication":
                    raise ValueError("macro model-only initialization requires a macro-credit publication")
                if hashlib.sha256(source.read_bytes()).hexdigest() != source_hash:
                    raise ValueError("initialization source changed while loading")
                model = loaded.model
                initialization = {**loaded.provenance, "mode": "model_only_new_macro_lineage",
                                  "source_sha256": source_hash, "optimizer_reset": True,
                                  "rng_reset": True, "destination_update": 0,
                                  "destination_encoding": grounding_encoding_identity()}
            else:
                model = HybridDomainActorCriticModel(ac_config).to(device)
                initialization = {}
            model.requires_grad_(True)
            model.eval()
            learner = MacroCreditLearner(model=model, config=config.learner)
            cursor = 0
            if args.resume:
                payload = load_macro_checkpoint(source, identity=identity, model=model, optimizer=learner.optimizer,
                                                artifact_root=root)
                learner.update = int(payload["update"])
                cursor = int(payload["next_seed_index"])
                initialization = payload["initialization"]
            elif args.migrate_runtime_from_checkpoint:
                source_hash = hashlib.sha256(source.read_bytes()).hexdigest()
                payload = migrate_macro_checkpoint_runtime(
                    source,
                    identity=identity,
                    model=model,
                    optimizer=learner.optimizer,
                    artifact_root=root,
                )
                if hashlib.sha256(source.read_bytes()).hexdigest() != source_hash:
                    raise ValueError("migration source checkpoint changed while loading")
                learner.update = int(payload["update"])
                cursor = int(payload["next_seed_index"])
                initialization = {
                    "mode": "runtime_migration_from_macro_checkpoint",
                    "source_path": str(source),
                    "source_sha256": source_hash,
                    "source_lineage_id": payload["lineage_id"],
                    "update_base_at_migration": learner.update,
                    "cursor_base_at_migration": cursor,
                    "prior_initialization": payload["initialization"],
                }
            _json(output / "course.json", identity.as_mapping())
            _json(output / "source_manifest.json", sources)
            protocol_incidents = output / "protocol-incidents"
            with ExitStack() as pool_stack:
                worker_pool: PersistentRolloutWorkerPool | None = None
                teacher: FrozenPolicy
                if (
                    config.learner.target_mode == "durable_afterstate_v2"
                    and config.course.parallel_collectors > 1
                ):
                    worker_pool = pool_stack.enter_context(
                        PersistentRolloutWorkerPool(
                            num_workers=config.course.parallel_collectors,
                            model=model,
                            training_config=training,
                            device=device,
                            sim_exe_path=simulator.executable,
                            protocol_quarantine_dir=protocol_incidents,
                            artifact_root=root,
                            allow_tf32=config.learner.allow_tf32,
                        )
                    )
                    teacher = worker_pool.inference_server
                else:
                    teacher = InMemoryTeacher(model, training, device)
                summary = run_cohorts(
                    config=config, output=output, identity=identity, learner=learner,
                    teacher=teacher,
                    backend_factory=lambda: HeadlessBackend(exe_path=simulator.executable,
                                                           protocol_quarantine_dir=protocol_incidents),
                    initial_seed_index=cursor, initialization=initialization, max_updates=args.max_updates,
                    initialize_boundary=not bool(args.resume), root=root,
                    source_paths=(
                        (source,)
                        if (
                            args.initialize_from_iql
                            or args.initialize_from_macro_publication
                            or args.migrate_runtime_from_checkpoint
                        )
                        else ()
                    ),
                    worker_pool=worker_pool,
                )
            print(json.dumps(summary, allow_nan=False))
        return 0
    except BaseException as error:
        # argparse exits happen before this try. Never overwrite another run's
        # status when path/ownership checks failed.
        print(json.dumps({"state": "aborted", "error_type": type(error).__name__, "error": str(error)}))
        return 130 if isinstance(error, KeyboardInterrupt) else 1


if __name__ == "__main__":
    raise SystemExit(main())
