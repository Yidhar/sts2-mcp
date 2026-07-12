"""Pure learner-update credit scheduling for the online baseline.

Keeping this policy independent from the training loop makes the warm-up
boundary explicit and testable.  In particular, the default policy does not
turn experience collected while learning was impossible into a synchronous
backlog as soon as replay reaches its minimum size.
"""

from __future__ import annotations

from typing import Literal

WarmupCreditPolicy = Literal["discard", "accrue"]


def _non_negative_int(value: object, *, label: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise TypeError(f"{label} must be an integer")
    if value < 0:
        raise ValueError(f"{label} must be non-negative")
    return value


def advance_update_credit(
    *,
    current_credit: int,
    collected_steps: int,
    replay_size_before: int,
    replay_size_after: int,
    minimum_replay_size: int,
    warmup_policy: WarmupCreditPolicy,
) -> int:
    """Return update credit after adding one collection batch to replay.

    ``discard`` starts the configured update-to-data schedule only after replay
    warm-up.  If a collection batch crosses the boundary, only transitions
    beyond the boundary earn credit.  This prevents the first eligible batch
    from synchronously paying back the entire warm-up period.

    ``accrue`` preserves the legacy behavior and is provided for controlled
    comparisons or exact continuation of a lineage that intentionally used the
    warm-up backlog.
    """

    credit = _non_negative_int(current_credit, label="current_credit")
    steps = _non_negative_int(collected_steps, label="collected_steps")
    before = _non_negative_int(replay_size_before, label="replay_size_before")
    after = _non_negative_int(replay_size_after, label="replay_size_after")
    minimum = _non_negative_int(minimum_replay_size, label="minimum_replay_size")
    if minimum == 0:
        raise ValueError("minimum_replay_size must be positive")
    if after < before:
        raise ValueError("replay_size_after cannot be smaller than replay_size_before")
    if warmup_policy not in {"discard", "accrue"}:
        raise ValueError("warmup_policy must be 'discard' or 'accrue'")

    if warmup_policy == "accrue" or before >= minimum:
        return credit + steps
    if after <= minimum:
        return 0

    # Replay can grow by fewer entries than ``collected_steps`` only after it
    # reaches capacity.  Since ``before < minimum <= capacity`` here, growth up
    # to the warm-up boundary is one-for-one and this count is well-defined.
    steps_to_warmup = minimum - before
    return max(0, steps - steps_to_warmup)


__all__ = ["WarmupCreditPolicy", "advance_update_credit"]
