from __future__ import annotations

import numpy as np
import pytest

from sts2_baseline import BoundedRolloutQueue, RolloutStep, SequenceUnroll
from sts2_rl.encoding import GroundedObservationEncoder


def _snapshot():
    return GroundedObservationEncoder().encode(
        {
            "decision_domain": "build",
            "player": {"hp": 10, "max_hp": 10},
            "run": {"act": 1, "floor": 2},
        },
        [
            {
                "kind": "proceed",
                "model_action_kind": "proceed",
                "is_enabled": True,
                "action_handle": "volatile",
            }
        ],
    ).snapshot


def _unroll(index: int, *, terminal: bool = True) -> SequenceUnroll:
    snapshot = _snapshot()
    return SequenceUnroll(
        episode_id=f"episode-{index}",
        start_step=index,
        policy_version=index,
        initial_recurrent_state=np.zeros(8, dtype=np.float32),
        steps=(
            RolloutStep(
                snapshot=snapshot,
                action_index=0,
                behavior_log_probability=0.0,
                reward=float(index),
                discount=0.0 if terminal else 0.997,
                policy_decision=False,
            ),
        ),
        bootstrap_snapshot=None if terminal else snapshot,
    )


def test_bounded_rollout_queue_is_fifo_and_consumes_once() -> None:
    queue = BoundedRolloutQueue(3)
    queue.put(_unroll(1))
    queue.put(_unroll(2))
    queue.put(_unroll(3))

    first = queue.get_batch(2, minimum=2)
    second = queue.get_batch(2, minimum=1, timeout=0.01)

    assert [item.episode_id for item in first] == ["episode-1", "episode-2"]
    assert [item.episode_id for item in second] == ["episode-3"]
    assert len(queue) == 0
    assert queue.metrics()["put_count"] == 3
    assert queue.metrics()["get_count"] == 3
    assert not hasattr(queue, "sample")
    assert not hasattr(queue, "update_priorities")


def test_queue_backpressure_times_out_instead_of_evicting() -> None:
    queue = BoundedRolloutQueue(1)
    original = _unroll(1)
    queue.put(original)
    with pytest.raises(TimeoutError, match="capacity"):
        queue.put(_unroll(2), timeout=0.001)
    assert queue.snapshot() == (original,)


def test_queue_snapshot_restore_preserves_order() -> None:
    source = BoundedRolloutQueue(4)
    source.put(_unroll(3))
    source.put(_unroll(4))
    payload = source.snapshot()

    restored = BoundedRolloutQueue(4)
    restored.restore(payload)
    assert restored.get_batch(4, minimum=2) == payload


def test_unroll_requires_bootstrap_exactly_for_continuing_sequence() -> None:
    snapshot = _snapshot()
    step = RolloutStep(
        snapshot=snapshot,
        action_index=0,
        behavior_log_probability=0.0,
        reward=0.0,
        discount=0.997,
        policy_decision=False,
    )
    with pytest.raises(ValueError, match="bootstrap_snapshot"):
        SequenceUnroll(
            episode_id="episode",
            start_step=0,
            policy_version=0,
            initial_recurrent_state=np.zeros(8, dtype=np.float32),
            steps=(step,),
            bootstrap_snapshot=None,
        )


def test_unroll_owns_read_only_recurrent_state() -> None:
    state = np.arange(8, dtype=np.float32)
    unroll = SequenceUnroll(
        episode_id="episode",
        start_step=0,
        policy_version=0,
        initial_recurrent_state=state,
        steps=(
            RolloutStep(
                snapshot=_snapshot(),
                action_index=0,
                behavior_log_probability=0.0,
                reward=0.0,
                discount=0.0,
                policy_decision=False,
            ),
        ),
        bootstrap_snapshot=None,
    )
    state[:] = -1
    assert np.array_equal(unroll.initial_recurrent_state, np.arange(8, dtype=np.float32))
    assert not unroll.initial_recurrent_state.flags.writeable
