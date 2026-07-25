from __future__ import annotations

import math

import numpy as np
import pytest

from sts2_rl.training.collector import (
    CollectionProtocolError,
    _branch_balanced_epsilon_behavior,
)


def test_branch_balanced_epsilon_is_uniform_over_branches_then_candidates() -> None:
    policy = np.asarray([0.06, 0.09, 0.15, 0.30, 0.40, 0.75], dtype=np.float32)
    valid = np.asarray([True, True, True, True, True, False], dtype=np.bool_)
    branch_ids = np.asarray([2, 2, 2, 7, 11, 13], dtype=np.int64)

    behavior = _branch_balanced_epsilon_behavior(
        policy=policy,
        valid=valid,
        policy_branch_ids=branch_ids,
        epsilon=1.0,
    )

    # Three legal branches receive one third each. The first branch's third is
    # divided among its three concrete actions; the two singleton branches each
    # retain their complete third. The masked candidate remains exactly zero.
    np.testing.assert_allclose(
        behavior,
        np.asarray([1.0 / 9.0] * 3 + [1.0 / 3.0, 1.0 / 3.0, 0.0]),
        rtol=0.0,
        atol=1e-15,
    )
    assert float(behavior.sum()) == pytest.approx(1.0)
    assert math.log(float(behavior[3])) == pytest.approx(math.log(1.0 / 3.0))


def test_branch_balanced_epsilon_mixes_exactly_with_joint_target_policy() -> None:
    policy = np.asarray([0.10, 0.20, 0.30, 0.40, 1.0], dtype=np.float32)
    valid = np.asarray([True, True, True, True, False], dtype=np.bool_)
    branch_ids = np.asarray([1, 1, 1, 9, 17], dtype=np.int64)

    behavior = _branch_balanced_epsilon_behavior(
        policy=policy,
        valid=valid,
        policy_branch_ids=branch_ids,
        epsilon=0.25,
    )
    exploration = np.asarray([1.0 / 6.0] * 3 + [1.0 / 2.0, 0.0])
    expected = 0.75 * policy.astype(np.float64)
    expected[-1] = 0.0
    expected += 0.25 * exploration

    np.testing.assert_allclose(behavior, expected, rtol=0.0, atol=1e-8)
    assert behavior[-1] == 0.0
    assert float(behavior.sum()) == pytest.approx(1.0)


def test_branch_balanced_epsilon_fails_closed_on_invalid_legal_surface() -> None:
    with pytest.raises(CollectionProtocolError, match="without a legal action"):
        _branch_balanced_epsilon_behavior(
            policy=np.asarray([1.0], dtype=np.float32),
            valid=np.asarray([False], dtype=np.bool_),
            policy_branch_ids=np.asarray([0], dtype=np.int64),
            epsilon=1.0,
        )
    with pytest.raises(CollectionProtocolError, match="negative legal policy branch"):
        _branch_balanced_epsilon_behavior(
            policy=np.asarray([1.0], dtype=np.float32),
            valid=np.asarray([True], dtype=np.bool_),
            policy_branch_ids=np.asarray([-1], dtype=np.int64),
            epsilon=1.0,
        )
