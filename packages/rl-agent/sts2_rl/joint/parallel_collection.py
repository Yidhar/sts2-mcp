"""Persistent multi-process episode collection with micro-batched GPU teacher inference.

Eliminates Python GIL contention across observation canonicalization, grounded
structural encoding, and deterministic afterstate synthesis while coalescing
per-worker policy evaluations into single batched CUDA forward passes on the
unchanged learner model.
"""

from __future__ import annotations

import os
import queue
import threading
import time
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from multiprocessing import get_context
from pathlib import Path
from types import TracebackType
from typing import Any

import numpy as np
import numpy.typing as npt
import torch
from torch import Tensor

from sts2_rl.backends import HeadlessBackend
from sts2_rl.encoding import EncodedDecisionSnapshot, GroundedEncodingConfig
from sts2_rl.encoding.snapshot import collate_encoded_snapshots
from sts2_rl.macro_branch_diagnostic import BranchLimits
from sts2_rl.macro_root_diagnostic import ProbeBudgetExhausted
from sts2_rl.models.hybrid_actor_critic import HybridDomainActorCriticModel
from sts2_rl.training.config import TrainingConfig

from .macro_credit_collection import (
    collect_durable_afterstate_episode,
    is_recoverable_episode_exception,
)
from .macro_credit_data import MacroCreditCohort, MacroCreditTrajectory, training_seed


@dataclass(slots=True)
class _InProcessWaiter:
    event: threading.Event
    logits: npt.NDArray[np.float32] | None = None
    hidden: npt.NDArray[np.float32] | None = None
    error: BaseException | None = None


class BatchedPolicyInferenceServer:
    """Thread- and IPC-safe micro-batching wrapper around an unchanged teacher model."""

    def __init__(
        self,
        model: HybridDomainActorCriticModel,
        encoding_config: GroundedEncodingConfig,
        device: torch.device,
        *,
        worker_send_conns: Sequence[Any] = (),
        mp_request_queue: Any | None = None,
        max_batch_size: int = 48,
        linger_seconds: float = 0.0008,
        allow_tf32: bool = False,
    ) -> None:
        if max_batch_size < 1:
            raise ValueError("max_batch_size must be at least 1")
        if linger_seconds < 0.0:
            raise ValueError("linger_seconds must be non-negative")
        self.model = model
        self.encoding_config = encoding_config
        self.device = device
        self.allow_tf32 = bool(allow_tf32)
        if self.device.type == "cuda":
            torch.backends.cuda.matmul.allow_tf32 = self.allow_tf32
            torch.backends.cudnn.allow_tf32 = self.allow_tf32
            torch.set_float32_matmul_precision("high" if self.allow_tf32 else "highest")
        self.recurrent_hidden_dim = int(model.config.recurrent_hidden_dim)
        self.hierarchical_branch_greedy = bool(
            model.actor_critic_config.enable_hierarchical_branch_policy
        )
        self.max_batch_size = int(max_batch_size)
        self.linger_seconds = float(linger_seconds)
        self._worker_send_conns = list(worker_send_conns)
        self._mp_request_queue = mp_request_queue
        self._local_queue: queue.Queue[
            tuple[EncodedDecisionSnapshot, npt.NDArray[np.float32] | None, _InProcessWaiter]
        ] = queue.Queue()
        self._stop_event = threading.Event()
        self._paused = threading.Event()
        self._idle_ack = threading.Event()
        self._idle_ack.set()
        self.total_requests = 0
        self.total_batches = 0
        self.max_batch_observed = 0
        self._thread = threading.Thread(
            target=self._serve_loop,
            name="sts2-gpu-teacher-batcher",
            daemon=True,
        )
        self._thread.start()

    def pause_for_learner(self) -> None:
        """Quiesce background GPU inference while learner mutates model weights."""
        self._paused.set()
        self._idle_ack.wait(timeout=10.0)

    def resume_for_collection(self) -> None:
        """Re-enable background GPU inference after learner update commits."""
        self._paused.clear()

    def close(self) -> None:
        self._stop_event.set()
        self._paused.clear()
        if self._thread.is_alive():
            self._thread.join(timeout=5.0)

    def __enter__(self) -> BatchedPolicyInferenceServer:
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        tb: TracebackType | None,
    ) -> None:
        self.close()

    def initial_state(self) -> Tensor:
        return torch.zeros((1, self.recurrent_hidden_dim), dtype=torch.float32)

    def forward(
        self,
        snapshot: EncodedDecisionSnapshot,
        hidden: Tensor | None,
    ) -> tuple[Tensor, Tensor]:
        hidden_np = (
            None
            if hidden is None
            else np.ascontiguousarray(hidden.detach().cpu().numpy(), dtype=np.float32)
        )
        waiter = _InProcessWaiter(event=threading.Event())
        self._local_queue.put((snapshot, hidden_np, waiter))
        waiter.event.wait()
        if waiter.error is not None:
            raise waiter.error
        assert waiter.logits is not None and waiter.hidden is not None
        return torch.from_numpy(waiter.logits), torch.from_numpy(waiter.hidden)

    def _try_pop_one(
        self,
        *,
        block_timeout: float,
    ) -> (
        tuple[
            str,
            int | _InProcessWaiter,
            EncodedDecisionSnapshot,
            npt.NDArray[np.float32] | None,
        ]
        | None
    ):
        try:
            snap, hid, waiter = self._local_queue.get_nowait()
            return ("local", waiter, snap, hid)
        except queue.Empty:
            pass
        if self._mp_request_queue is not None:
            try:
                if block_timeout > 0.0:
                    msg = self._mp_request_queue.get(timeout=block_timeout)
                else:
                    msg = self._mp_request_queue.get_nowait()
                if msg is None:
                    return None
                worker_id, snap, hid = msg
                return ("ipc", int(worker_id), snap, hid)
            except queue.Empty:
                return None
        if block_timeout > 0.0:
            try:
                snap, hid, waiter = self._local_queue.get(timeout=block_timeout)
                return ("local", waiter, snap, hid)
            except queue.Empty:
                return None
        return None

    def _serve_loop(self) -> None:
        if self.device.type == "cuda":
            torch.backends.cuda.matmul.allow_tf32 = self.allow_tf32
            torch.backends.cudnn.allow_tf32 = self.allow_tf32
            torch.set_float32_matmul_precision("high" if self.allow_tf32 else "highest")
        while not self._stop_event.is_set():
            if self._paused.is_set():
                self._idle_ack.set()
                time.sleep(0.002)
                continue
            first = self._try_pop_one(block_timeout=0.01)
            if first is None:
                self._idle_ack.set()
                continue
            self._idle_ack.clear()
            batch_items = [first]
            target_cap = self.max_batch_size
            if self._worker_send_conns:
                target_cap = min(self.max_batch_size, max(1, len(self._worker_send_conns)))
            deadline = time.perf_counter() + self.linger_seconds
            while len(batch_items) < target_cap:
                nxt = self._try_pop_one(block_timeout=0.0)
                if nxt is not None:
                    batch_items.append(nxt)
                    continue
                remaining = deadline - time.perf_counter()
                if remaining <= 0.0:
                    break
                nxt = self._try_pop_one(block_timeout=min(remaining, 0.0004))
                if nxt is not None:
                    batch_items.append(nxt)
                else:
                    break
            self._execute_batch(batch_items)
        self._idle_ack.set()

    def _execute_batch(
        self,
        batch_items: list[
            tuple[
                str,
                int | _InProcessWaiter,
                EncodedDecisionSnapshot,
                npt.NDArray[np.float32] | None,
            ]
        ],
    ) -> None:
        bsz = len(batch_items)
        self.total_requests += bsz
        self.total_batches += 1
        self.max_batch_observed = max(self.max_batch_observed, bsz)
        try:
            snapshots = tuple(item[2] for item in batch_items)
            encoded = collate_encoded_snapshots(
                snapshots,
                expected_config=self.encoding_config,
                expected_fingerprint=snapshots[0].encoding_fingerprint,
                device=self.device,
                validate=False,
            )
            hidden_np = np.zeros((bsz, self.recurrent_hidden_dim), dtype=np.float32)
            for idx, (_, _, _, hid) in enumerate(batch_items):
                if hid is not None:
                    hidden_np[idx] = hid.reshape(self.recurrent_hidden_dim)
            with torch.inference_mode():
                model_dtype = next(self.model.parameters()).dtype
                hid_t = torch.from_numpy(hidden_np)
                if self.device.type == "cuda":
                    hidden_gpu = hid_t.pin_memory().to(
                        device=self.device, dtype=model_dtype, non_blocking=True
                    )
                else:
                    hidden_gpu = hid_t.to(device=self.device, dtype=model_dtype)
                output = self.model(encoded, hidden_gpu, validate=False)
                logits_cpu = output.policy_logits.detach().to(
                    device="cpu", dtype=torch.float32
                ).numpy()
                next_hidden_cpu = output.recurrent_state.detach().to(
                    device="cpu", dtype=torch.float32
                ).numpy()
            for idx, (kind, target, snap, _) in enumerate(batch_items):
                cand_count = snap.candidate_count
                item_logits = np.ascontiguousarray(logits_cpu[idx, :cand_count])
                item_hidden = np.ascontiguousarray(next_hidden_cpu[idx : idx + 1])
                if kind == "local":
                    assert isinstance(target, _InProcessWaiter)
                    target.logits = item_logits
                    target.hidden = item_hidden
                    target.event.set()
                else:
                    assert isinstance(target, int)
                    self._worker_send_conns[target].send(("ok", item_logits, item_hidden))
        except BaseException as exc:
            for kind, target, _, _ in batch_items:
                if kind == "local":
                    assert isinstance(target, _InProcessWaiter)
                    target.error = exc
                    target.event.set()
                else:
                    assert isinstance(target, int)
                    try:
                        self._worker_send_conns[target].send(
                            ("err", type(exc).__name__, str(exc))
                        )
                    except OSError:
                        pass


class RemoteBatchedFrozenPolicy:
    """Worker-process FrozenPolicy proxy backed by BatchedPolicyInferenceServer."""

    def __init__(
        self,
        *,
        worker_id: int,
        request_queue: Any,
        response_conn: Any,
        recurrent_hidden_dim: int,
        hierarchical_branch_greedy: bool,
    ) -> None:
        self.worker_id = int(worker_id)
        self._request_queue = request_queue
        self._response_conn = response_conn
        self._recurrent_hidden_dim = int(recurrent_hidden_dim)
        self.hierarchical_branch_greedy = bool(hierarchical_branch_greedy)

    def initial_state(self) -> Tensor:
        return torch.zeros((1, self._recurrent_hidden_dim), dtype=torch.float32)

    def forward(
        self,
        snapshot: EncodedDecisionSnapshot,
        hidden: Tensor | None,
    ) -> tuple[Tensor, Tensor]:
        hidden_np = (
            None
            if hidden is None
            else np.ascontiguousarray(hidden.detach().cpu().numpy(), dtype=np.float32)
        )
        self._request_queue.put((self.worker_id, snapshot, hidden_np))
        reply = self._response_conn.recv()
        if reply[0] != "ok":
            _, err_type, err_msg = reply
            if err_type == "FloatingPointError":
                raise FloatingPointError(err_msg)
            raise RuntimeError(f"batched teacher forward failed ({err_type}): {err_msg}")
        _, logits_np, next_hidden_np = reply
        return torch.from_numpy(logits_np), torch.from_numpy(next_hidden_np)


def _rollout_worker_entry(
    worker_id: int,
    task_queue: Any,
    result_queue: Any,
    infer_request_queue: Any,
    infer_response_conn: Any,
    sim_exe_path: str,
    protocol_quarantine_dir: str,
    artifact_root_str: str,
    recurrent_hidden_dim: int,
    hierarchical_branch_greedy: bool,
) -> None:
    os.environ["OMP_NUM_THREADS"] = "1"
    os.environ["MKL_NUM_THREADS"] = "1"
    os.environ["OPENBLAS_NUM_THREADS"] = "1"
    os.environ["STS2_ARTIFACT_ROOT"] = artifact_root_str
    if "tcmalloc" in os.environ.get("LD_PRELOAD", ""):
        os.environ.pop("LD_PRELOAD", None)
    torch.set_num_threads(1)

    root_path = Path(artifact_root_str)
    exe_path = Path(sim_exe_path)
    quarantine_path = Path(protocol_quarantine_dir)
    policy = RemoteBatchedFrozenPolicy(
        worker_id=worker_id,
        request_queue=infer_request_queue,
        response_conn=infer_response_conn,
        recurrent_hidden_dim=recurrent_hidden_dim,
        hierarchical_branch_greedy=hierarchical_branch_greedy,
    )

    def backend_factory() -> HeadlessBackend:
        return HeadlessBackend(
            exe_path=exe_path,
            protocol_quarantine_dir=quarantine_path,
        )

    while True:
        task = task_queue.get()
        if task is None:
            break
        batch_token = int(task["batch_token"])
        slot_index = int(task["slot_index"])
        attempt_offset = int(task["attempt_offset"])
        seed = int(task["seed"])
        try:
            trajectory, report = collect_durable_afterstate_episode(
                backend_factory=backend_factory,
                policy=policy,
                config=task["config"],
                seed=seed,
                teacher_version=int(task["teacher_version"]),
                limits=task["limits"],
                output_dir=Path(task["output_dir"]),
                macro_branch_epsilon=float(task["macro_branch_epsilon"]),
                maximum_snapshot_bytes=int(task["maximum_snapshot_bytes"]),
                maximum_group_seconds=float(task["maximum_group_seconds"]),
                artifact_root=root_path,
                resource_guard=None,
            )
            result_queue.put(
                ("ok", batch_token, slot_index, attempt_offset, seed, trajectory, report)
            )
        except BaseException as exc:
            result_queue.put(
                (
                    "err",
                    batch_token,
                    slot_index,
                    attempt_offset,
                    seed,
                    {
                        "error_type": type(exc).__name__,
                        "error": str(exc),
                        "recoverable": is_recoverable_episode_exception(exc),
                        "fingerprint": getattr(exc, "fingerprint", None),
                        "quarantine_path": getattr(exc, "quarantine_path", None),
                    },
                )
            )


class PersistentRolloutWorkerPool:
    """Warm multi-process episode collector pool paired with GPU micro-batching."""

    def __init__(
        self,
        *,
        num_workers: int,
        model: HybridDomainActorCriticModel,
        training_config: TrainingConfig,
        device: torch.device,
        sim_exe_path: Path,
        protocol_quarantine_dir: Path,
        artifact_root: Path,
        max_inference_batch_size: int = 48,
        inference_linger_seconds: float = 0.0008,
        allow_tf32: bool = False,
    ) -> None:
        if num_workers < 1:
            raise ValueError("num_workers must be positive")
        self.num_workers = int(num_workers)
        self._ctx = get_context("spawn")
        self._task_queue: Any = self._ctx.Queue()
        self._result_queue: Any = self._ctx.Queue()
        self._infer_request_queue: Any = self._ctx.Queue()
        recv_conns: list[Any] = []
        send_conns: list[Any] = []
        for _ in range(self.num_workers):
            r_conn, s_conn = self._ctx.Pipe(duplex=False)
            recv_conns.append(r_conn)
            send_conns.append(s_conn)
        self._recv_conns = recv_conns
        self._send_conns = send_conns
        self.inference_server = BatchedPolicyInferenceServer(
            model=model,
            encoding_config=training_config.model.to_encoding_config(),
            device=device,
            worker_send_conns=send_conns,
            mp_request_queue=self._infer_request_queue,
            max_batch_size=max(max_inference_batch_size, self.num_workers),
            linger_seconds=inference_linger_seconds,
            allow_tf32=allow_tf32,
        )
        self._processes = []
        for worker_id in range(self.num_workers):
            proc = self._ctx.Process(
                target=_rollout_worker_entry,
                args=(
                    worker_id,
                    self._task_queue,
                    self._result_queue,
                    self._infer_request_queue,
                    recv_conns[worker_id],
                    str(sim_exe_path),
                    str(protocol_quarantine_dir),
                    str(artifact_root),
                    int(model.config.recurrent_hidden_dim),
                    bool(model.actor_critic_config.enable_hierarchical_branch_policy),
                ),
                name=f"sts2-rollout-worker-{worker_id:02d}",
                daemon=True,
            )
            proc.start()
            self._processes.append(proc)
        self._batch_counter = 0
        self._closed = False

    def collect_cohort(
        self,
        *,
        config: TrainingConfig,
        seeds: Sequence[int],
        teacher_version: int,
        limits: BranchLimits,
        cohort_dir: Path,
        cohort_id: str,
        macro_branch_epsilon: float,
        replacement_seed_fn: Callable[[int], int] | None,
        max_recoverable_incidents: int,
        maximum_episode_snapshot_bytes: int,
        maximum_cohort_snapshot_bytes: int,
        maximum_episode_seconds: float,
        maximum_cohort_seconds: float,
        artifact_root: Path,
        resource_guard: Callable[[], None] | None,
    ) -> tuple[MacroCreditCohort, list[dict[str, Any]], list[dict[str, Any]]]:
        if self._closed:
            raise RuntimeError("PersistentRolloutWorkerPool is closed")
        for seed in seeds:
            training_seed(seed)
        self.inference_server.resume_for_collection()
        self._batch_counter += 1
        batch_token = self._batch_counter
        started = time.monotonic()
        next_attempt_offset = len(seeds)
        recoverable_incidents: list[dict[str, Any]] = []
        results_by_slot: list[tuple[MacroCreditTrajectory, dict[str, Any]] | None] = [
            None
        ] * len(seeds)
        completed_slots = 0

        def enqueue_slot(slot_idx: int, attempt_offset: int, seed_val: int) -> None:
            remaining_seconds = maximum_cohort_seconds - (time.monotonic() - started)
            if remaining_seconds <= 0.0:
                raise ProbeBudgetExhausted("maximum_cohort_seconds")
            self._task_queue.put(
                {
                    "batch_token": batch_token,
                    "slot_index": slot_idx,
                    "attempt_offset": attempt_offset,
                    "seed": seed_val,
                    "teacher_version": teacher_version,
                    "config": config,
                    "limits": limits,
                    "output_dir": str(cohort_dir / f"episode-{attempt_offset:03d}"),
                    "macro_branch_epsilon": macro_branch_epsilon,
                    "maximum_snapshot_bytes": min(
                        maximum_episode_snapshot_bytes, maximum_cohort_snapshot_bytes
                    ),
                    "maximum_group_seconds": min(
                        maximum_episode_seconds, remaining_seconds
                    ),
                }
            )

        try:
            for slot_idx, seed_val in enumerate(seeds):
                enqueue_slot(slot_idx, slot_idx, seed_val)

            while completed_slots < len(seeds):
                if resource_guard is not None:
                    resource_guard()
                if time.monotonic() - started > maximum_cohort_seconds:
                    raise ProbeBudgetExhausted("maximum_cohort_seconds")
                dead_workers = [
                    proc.exitcode
                    for proc in self._processes
                    if not proc.is_alive() and proc.exitcode is not None
                ]
                if dead_workers:
                    raise RuntimeError(
                        f"rollout worker process terminated unexpectedly (exitcodes={dead_workers})"
                    )
                try:
                    msg = self._result_queue.get(timeout=0.25)
                except queue.Empty:
                    continue
                if int(msg[1]) != batch_token:
                    continue
                status = msg[0]
                if status == "ok":
                    _, _, slot_idx, _, _, trajectory, report = msg
                    results_by_slot[int(slot_idx)] = (trajectory, report)
                    completed_slots += 1
                else:
                    _, _, slot_idx, attempt_offset, failed_seed, err_info = msg
                    if (
                        err_info.get("recoverable") is True
                        and replacement_seed_fn is not None
                    ):
                        recoverable_incidents.append(
                            {
                                "attempt_offset": int(attempt_offset),
                                "seed": int(failed_seed),
                                "error_type": err_info.get("error_type"),
                                "error": err_info.get("error"),
                                "fingerprint": err_info.get("fingerprint"),
                                "quarantine_path": err_info.get("quarantine_path"),
                            }
                        )
                        if len(recoverable_incidents) <= max_recoverable_incidents:
                            new_offset = next_attempt_offset
                            next_attempt_offset += 1
                            replacement_seed = replacement_seed_fn(new_offset)
                            training_seed(replacement_seed)
                            enqueue_slot(int(slot_idx), new_offset, replacement_seed)
                            continue
                    err_type = str(err_info.get("error_type", "RuntimeError"))
                    err_msg = str(err_info.get("error", "unknown worker failure"))
                    if err_type == "ProbeBudgetExhausted":
                        raise ProbeBudgetExhausted(err_msg)
                    raise RuntimeError(f"{err_type}: {err_msg}")
        finally:
            self.inference_server.pause_for_learner()

        finalized = [pair for pair in results_by_slot if pair is not None]
        if len(finalized) != len(seeds):
            raise RuntimeError("incomplete durable cohort result set")
        trajectories = tuple(pair[0] for pair in finalized)
        reports = [pair[1] for pair in finalized]
        if recoverable_incidents and reports:
            reports[0]["recoverable_incidents"] = list(recoverable_incidents)
        training_episodes = [
            report["training_episode"]
            for report in reports
            if isinstance(report.get("training_episode"), dict)
        ]
        cohort = MacroCreditCohort(
            cohort_id, teacher_version, (), natural_only=trajectories
        )
        if cohort.storage_nbytes() > maximum_cohort_snapshot_bytes:
            raise ProbeBudgetExhausted("maximum_cohort_snapshot_bytes")
        if resource_guard is not None:
            resource_guard()
        return cohort, training_episodes, reports

    def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        for _ in self._processes:
            try:
                self._task_queue.put_nowait(None)
            except Exception:
                pass
        deadline = time.monotonic() + 3.0
        for proc in self._processes:
            remaining = max(0.05, deadline - time.monotonic())
            proc.join(timeout=remaining)
            if proc.is_alive():
                proc.terminate()
                proc.join(timeout=1.0)
        self.inference_server.close()
        for conn in (*self._recv_conns, *self._send_conns):
            try:
                conn.close()
            except OSError:
                pass

    def __enter__(self) -> PersistentRolloutWorkerPool:
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        tb: TracebackType | None,
    ) -> None:
        self.close()


__all__ = [
    "BatchedPolicyInferenceServer",
    "PersistentRolloutWorkerPool",
    "RemoteBatchedFrozenPolicy",
]
