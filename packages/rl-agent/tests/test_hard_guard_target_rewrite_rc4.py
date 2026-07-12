"""RC-4 regression: hard guards must not rewrite the policy training target by default.

The bug: when a post-search hard guard overrode the executed action, the stored
``search_policy`` target was rewritten to a one-hot on the guard action, so the
network was trained to imitate ~25 hand-coded guards (trained policy != the policy
the model actually chose). The fix gates that rewrite behind
``--hard-guard-target-rewrite`` (default off); the guard action is still executed
and stored as the replay ``action``, but the model trains on its OWN pre-guard soft
distribution, keeping credit assignment on-policy.

These cover the pure decision helper ``resolve_hard_guard_policy_target``.
"""

from __future__ import annotations

import unittest

import numpy as np

from muzero.training.post_search_policy_retarget import resolve_hard_guard_policy_target


_SOFT = np.array([0.5, 0.3, 0.2, 0.0], dtype=np.float32)


class HardGuardTargetRewriteTest(unittest.TestCase):
    def test_off_with_override_keeps_soft_target(self) -> None:
        # Guard overrode the action (0 -> 2) but rewrite is off: target stays the
        # model's own soft distribution, and the override is still telemetered.
        target, retargeted, override = resolve_hard_guard_policy_target(
            _SOFT.copy(), original_action_idx=0, final_action_idx=2,
            rewrite_target=False, max_actions=4,
        )
        self.assertTrue(override)
        self.assertFalse(retargeted)
        np.testing.assert_allclose(np.asarray(target, dtype=np.float32), _SOFT)

    def test_on_with_override_writes_one_hot(self) -> None:
        target, retargeted, override = resolve_hard_guard_policy_target(
            _SOFT.copy(), original_action_idx=0, final_action_idx=2,
            rewrite_target=True, max_actions=4,
        )
        self.assertTrue(override)
        self.assertTrue(retargeted)
        np.testing.assert_allclose(
            np.asarray(target, dtype=np.float32), np.array([0, 0, 1, 0], dtype=np.float32)
        )

    def test_no_override_never_retargets(self) -> None:
        # When the guard did not change the action, neither mode rewrites the target.
        for rewrite in (False, True):
            target, retargeted, override = resolve_hard_guard_policy_target(
                _SOFT.copy(), original_action_idx=1, final_action_idx=1,
                rewrite_target=rewrite, max_actions=4,
            )
            self.assertFalse(override, rewrite)
            self.assertFalse(retargeted, rewrite)
            np.testing.assert_allclose(np.asarray(target, dtype=np.float32), _SOFT)


if __name__ == "__main__":
    unittest.main()
