"""RC-1 regression helpers for explicit checkpoint migration/warm-start.

Reproduces, at unit scale, the dev's confirmed failure mode: editing a guard/aux
head changed the network state_dict keys, which flipped ``allow_exact_resume`` to
False and silently discarded BOTH the optimizer moments AND the 2.3 GB replay
buffer -- so the fix was to pass ``--resume-without-buffer`` and relearn cold.

These tests cover migration helpers in ``muzero.training.checkpointing``. They
are intentionally not used by fail-closed exact resume:
  - ``_load_optimizer_compatible``: per-parameter Adam-moment warm-start with a
    shape guard, so a head resize/addition keeps momentum for the unchanged
    params instead of the old all-or-nothing reset.
  - ``register_replay_identity_migration`` / ``_compat_version_loadable``: a
    ``checkpoint_compatibility_version`` bump with a registered identity migration
    no longer discards old replay buffers (used when RC-5/RC-3 shift the
    value-target distribution without changing the on-disk layout).
"""

from __future__ import annotations

import unittest

import torch
import torch.nn as nn

from muzero.training.checkpointing import (
    _compat_version_loadable,
    _load_optimizer_compatible,
    _metadata_replay_compatible,
    _replay_state_compatible,
    register_replay_identity_migration,
)


def _toy() -> nn.Sequential:
    return nn.Sequential(nn.Linear(4, 4), nn.Linear(4, 2))


def _warm_optimizer(model: nn.Module) -> torch.optim.Optimizer:
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    for _ in range(3):
        opt.zero_grad()
        model(torch.randn(8, 4)).sum().backward()
        opt.step()
    return opt


class OptimizerPartialLoadTest(unittest.TestCase):
    def test_full_restore_when_shapes_match(self) -> None:
        torch.manual_seed(0)
        saved = _warm_optimizer(_toy()).state_dict()

        fresh = _toy()
        fresh_opt = torch.optim.Adam(fresh.parameters(), lr=1e-3)
        loaded, reinit = _load_optimizer_compatible(fresh_opt, saved)

        n_params = len(list(fresh.parameters()))
        self.assertEqual(loaded, n_params)
        self.assertEqual(reinit, 0)
        first = next(iter(fresh.parameters()))
        self.assertIn("exp_avg", fresh_opt.state[first])
        self.assertEqual(tuple(fresh_opt.state[first]["exp_avg"].shape), tuple(first.shape))

    def test_head_resize_keeps_unchanged_param_moments(self) -> None:
        torch.manual_seed(0)
        saved = _warm_optimizer(_toy()).state_dict()

        # Simulate a guard/aux edit that resizes the final head: Linear(4,2) -> Linear(4,3).
        mutated = nn.Sequential(nn.Linear(4, 4), nn.Linear(4, 3))
        mutated_opt = torch.optim.Adam(mutated.parameters(), lr=1e-3)
        loaded, reinit = _load_optimizer_compatible(mutated_opt, saved)

        # First layer (weight+bias) unchanged -> restored; resized head (weight+bias) -> fresh.
        self.assertEqual(loaded, 2)
        self.assertEqual(reinit, 2)
        params = list(mutated.parameters())
        self.assertIn("exp_avg", mutated_opt.state.get(params[0], {}))
        self.assertNotIn(params[-1], mutated_opt.state)


class IdentityMigrationTest(unittest.TestCase):
    def test_compat_version_loadable(self) -> None:
        self.assertTrue(_compat_version_loadable(3, 3))
        self.assertFalse(_compat_version_loadable(3, 4))
        register_replay_identity_migration(3, 4)
        self.assertTrue(_compat_version_loadable(3, 4))
        self.assertFalse(_compat_version_loadable(4, 3))  # not symmetric
        self.assertFalse(_compat_version_loadable(None, 4))

    def test_metadata_gate_respects_identity_migration(self) -> None:
        register_replay_identity_migration(7, 8)
        saved = {
            "checkpoint_compatibility_version": 7, "obs_schema_version": "v4",
            "replay_schema_version": "r1", "observation_shape_caps": {"a": 1},
        }
        current = dict(saved, checkpoint_compatibility_version=8)
        ok, mismatches = _metadata_replay_compatible(saved, current)
        self.assertTrue(ok, mismatches)

        # An actual obs-layout change is still rejected even with the version migration.
        current_bad = dict(current, observation_shape_caps={"a": 2})
        ok_bad, mismatches_bad = _metadata_replay_compatible(saved, current_bad)
        self.assertFalse(ok_bad)
        self.assertIn("observation_shape_caps", mismatches_bad)

    def test_replay_state_gate_respects_identity_migration(self) -> None:
        register_replay_identity_migration(9, 10)
        current = {
            "replay_schema_version": "r1", "checkpoint_compatibility_version": 10,
            "observation_shape_caps": {"a": 1},
        }
        state = {
            "schema_version": "r1", "checkpoint_compatibility_version": 9,
            "observation_shape_caps": {"a": 1},
        }
        ok, mismatches = _replay_state_compatible(state, current)
        self.assertTrue(ok, mismatches)


if __name__ == "__main__":
    unittest.main()
