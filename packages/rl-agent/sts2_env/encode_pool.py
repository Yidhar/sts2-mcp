"""Process-pool-backed drop-in replacement for WorldTokenObservationEncoder.

Context: observation encoding is ~9 ms of pure-Python CPU work per step. The
async training collector runs N env workers as THREADS, so under GIL the
per-step encode latency scales linearly with n_envs (26 ms @ n=4, 83 ms @ n=8
observed). Process pool sidesteps the GIL — a benchmark showed 8× parallel
encodes dropping from 217 ms (ThreadPool-8) to 29 ms (ProcessPool-8), a 7.4×
speedup over threads and 3.3× over sequential.

Usage:
    pool = build_encode_pool(max_workers=8, encoder_kwargs={"use_text": False})
    encoder = PooledObsEncoder(pool=pool, local_encoder=local_enc_for_obs_space)
    # Pass ``encoder`` where a WorldTokenObservationEncoder is expected. Its
    # .encode() submits to the pool; .obs_space comes from the local encoder.

The pool is owned by the trainer process and must outlive all env factories.
Each pool worker lazily constructs its own WorldTokenObservationEncoder on
first task (driven by the initializer below).

The local encoder exists purely so ``obs_space`` is available without an
IPC round-trip at env construction time.
"""
from __future__ import annotations

from concurrent.futures import ProcessPoolExecutor
from typing import Any

# Module-level singleton held inside each pool worker process. Constructed
# lazily by ``_init_worker`` so the fork/spawn start cost is amortized.
_WORKER_ENCODER: Any = None


def _init_worker(encoder_kwargs: dict[str, Any]) -> None:
    """Initializer called once per pool worker process.

    Builds one WorldTokenObservationEncoder in the worker's address space
    using the kwargs the trainer configured. Kept at module scope (not a
    closure) so it is pickle-safe for ``ProcessPoolExecutor``.
    """
    global _WORKER_ENCODER
    # Import inside the worker so the parent process doesn't pay import cost
    # before the pool is actually needed.
    from sts2_env.observation_v3 import WorldTokenObservationEncoder
    _WORKER_ENCODER = WorldTokenObservationEncoder(**(encoder_kwargs or {}))


def _encode_in_worker(
    raw_obs: dict[str, Any],
    legal_actions: list[dict[str, Any]],
    planner_context: dict[str, Any] | None,
) -> dict[str, Any]:
    """Pool task: run encode() against the worker's local encoder.

    Raises RuntimeError if the initializer didn't run — should never happen
    in normal use but surfaces misconfiguration fast.
    """
    if _WORKER_ENCODER is None:
        raise RuntimeError(
            "encode_pool worker missing encoder; _init_worker did not run"
        )
    return _WORKER_ENCODER.encode(raw_obs, legal_actions, planner_context)


def build_encode_pool(
    *,
    max_workers: int,
    encoder_kwargs: dict[str, Any] | None = None,
) -> ProcessPoolExecutor:
    """Construct a ProcessPoolExecutor wired up for obs encoding.

    Caller owns the pool lifecycle (shutdown at training end).
    """
    return ProcessPoolExecutor(
        max_workers=max(int(max_workers), 1),
        initializer=_init_worker,
        initargs=(dict(encoder_kwargs or {}),),
    )


class PooledObsEncoder:
    """Drop-in ``WorldTokenObservationEncoder`` facade that dispatches
    ``.encode(...)`` into a shared process pool.

    Safe to call from many worker threads concurrently — the underlying
    ``ProcessPoolExecutor.submit`` is thread-safe, and each worker thread
    simply blocks on its own future.
    """

    def __init__(self, *, pool: ProcessPoolExecutor, local_encoder: Any) -> None:
        self._pool = pool
        self._local = local_encoder

    @property
    def obs_space(self):
        return self._local.obs_space

    def encode(
        self,
        raw_obs: dict[str, Any],
        legal_actions: list[dict[str, Any]],
        planner_context: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        future = self._pool.submit(
            _encode_in_worker, raw_obs, legal_actions, planner_context,
        )
        return future.result()


__all__ = [
    "PooledObsEncoder",
    "build_encode_pool",
]
