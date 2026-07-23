"""Disjoint, auditable training and held-out evaluation seed namespaces."""

from __future__ import annotations

SIGNED_INT32_MAX = 2_147_483_647
TRAINING_SEED_PARITY = 0
EVALUATION_SEED_PARITY = 1
# Repeated validation uses the low odd prefix.  The high, disjoint odd range is
# reserved for a final audit and must never be touched by training-time model
# selection or liveness guards.
FINAL_AUDIT_SEED_INDEX_OFFSET = 100_000_000


def _base_seed(value: int) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise TypeError("base seed must be an integer")
    if value < 0:
        raise ValueError("base seed must be non-negative")
    return value


def training_seed_start(base_seed: int) -> int:
    """Map a user seed into the even training-only seed namespace."""

    seed = 2 * _base_seed(base_seed) + TRAINING_SEED_PARITY
    if seed > SIGNED_INT32_MAX:
        raise ValueError("base seed cannot fit the signed 32-bit training namespace")
    return seed


def held_out_evaluation_seeds(base_seed: int, count: int) -> tuple[int, ...]:
    """Return the fixed odd validation prefix used at repeated gates."""

    base = _base_seed(base_seed)
    if isinstance(count, bool) or not isinstance(count, int):
        raise TypeError("evaluation seed count must be an integer")
    if count < 0:
        raise ValueError("evaluation seed count must be non-negative")
    seeds = tuple(2 * (base + index) + EVALUATION_SEED_PARITY for index in range(count))
    if seeds and seeds[-1] > SIGNED_INT32_MAX:
        raise ValueError("evaluation seed set exceeds the signed 32-bit namespace")
    return seeds


def final_audit_evaluation_seeds(base_seed: int, count: int) -> tuple[int, ...]:
    """Return never-reused odd seeds reserved for the final policy audit."""

    base = _base_seed(base_seed)
    if isinstance(count, bool) or not isinstance(count, int):
        raise TypeError("final audit seed count must be an integer")
    if count < 0:
        raise ValueError("final audit seed count must be non-negative")
    start = base + FINAL_AUDIT_SEED_INDEX_OFFSET
    seeds = tuple(
        2 * (start + index) + EVALUATION_SEED_PARITY
        for index in range(count)
    )
    if seeds and seeds[-1] > SIGNED_INT32_MAX:
        raise ValueError("final audit seed set exceeds the signed 32-bit namespace")
    return seeds


def validate_seed_budget(
    base_seed: int,
    *,
    maximum_training_episodes: int,
    evaluation_episodes: int,
    final_audit_episodes: int = 0,
) -> None:
    """Prove configured train/evaluation seeds remain valid and disjoint.

    A non-empty training episode consumes at least one environment step, so the
    total environment-step budget is a safe upper bound on episode count.
    """

    base = _base_seed(base_seed)
    for label, value in (
        ("maximum_training_episodes", maximum_training_episodes),
        ("evaluation_episodes", evaluation_episodes),
        ("final_audit_episodes", final_audit_episodes),
    ):
        if isinstance(value, bool) or not isinstance(value, int):
            raise TypeError(f"{label} must be an integer")
        if value < 0:
            raise ValueError(f"{label} must be non-negative")
    last_training = 2 * (base + max(0, maximum_training_episodes - 1))
    if last_training > SIGNED_INT32_MAX:
        raise ValueError("training seed budget exceeds signed 32-bit reset seeds")
    held_out_evaluation_seeds(base, evaluation_episodes)
    final_audit_evaluation_seeds(base, final_audit_episodes)


__all__ = [
    "EVALUATION_SEED_PARITY",
    "FINAL_AUDIT_SEED_INDEX_OFFSET",
    "SIGNED_INT32_MAX",
    "TRAINING_SEED_PARITY",
    "final_audit_evaluation_seeds",
    "held_out_evaluation_seeds",
    "training_seed_start",
    "validate_seed_budget",
]
