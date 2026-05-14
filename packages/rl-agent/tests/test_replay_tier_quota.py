"""Unit tests for the P0-2 batch-level replay tier quota.

Covers:
* Pure ``allocate_tier_quotas`` allocation with a healthy multi-tier pool.
* Empty / single-tier pool fallback to flex.
* Boss cap enforcement when boss priorities dominate.
* End-to-end ``MuZeroReplayBuffer.sample_batch`` with synthetic trajectories
  to confirm slot-level tier counts match the configured quotas.
"""
from __future__ import annotations

import sys
import types
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from muzero.replay_scheduler import (
    TierQuotaConfig,
    allocate_tier_quotas,
)


def test_allocate_quotas_balanced_pool_respects_targets():
    config = TierQuotaConfig(
        targets={"boss": 0.60, "elite": 0.25, "normal": 0.15, "weak": 0.0},
        min_caps={"elite": 0.18, "normal": 0.10},
        max_caps={"boss": 0.65},
    )
    alloc = allocate_tier_quotas(
        batch_size=16,
        pool_sizes={"boss": 1000, "elite": 200, "normal": 100, "weak": 50},
        config=config,
    )
    assert sum(alloc.quotas.values()) == 16
    # boss target = 0.60 * 16 = 9.6 → rounds to 10; cap = floor(0.65*16) = 10 → no cap hit
    assert alloc.quotas["boss"] <= 10
    assert alloc.quotas["elite"] >= 3  # min cap 0.18 * 16 = 2.88 → ceil 3
    assert alloc.quotas["normal"] >= 2  # min cap 0.10 * 16 = 1.6 → ceil 2
    assert not alloc.boss_cap_hit
    assert not alloc.normal_min_unfilled
    assert not alloc.elite_min_unfilled


def test_allocate_quotas_boss_dominant_input_caps_boss():
    """When the priority distribution would push boss to 100%, the quota
    sampler must still keep boss at or below the configured max share."""
    config = TierQuotaConfig(
        targets={"boss": 1.00, "elite": 0.0, "normal": 0.0, "weak": 0.0},
        min_caps={"elite": 0.18, "normal": 0.10},
        max_caps={"boss": 0.65},
    )
    alloc = allocate_tier_quotas(
        batch_size=16,
        pool_sizes={"boss": 1000, "elite": 100, "normal": 100, "weak": 0},
        config=config,
    )
    assert sum(alloc.quotas.values()) == 16
    assert alloc.quotas["boss"] <= 10
    assert alloc.boss_cap_hit
    # Min caps must still be honored: elite >= 3, normal >= 2.
    assert alloc.quotas["elite"] >= 3
    assert alloc.quotas["normal"] >= 2


def test_allocate_quotas_records_unfilled_minimums_when_pool_empty():
    config = TierQuotaConfig(
        targets={"boss": 0.60, "elite": 0.25, "normal": 0.15, "weak": 0.0},
        min_caps={"elite": 0.18, "normal": 0.10},
        max_caps={"boss": 0.65},
    )
    alloc = allocate_tier_quotas(
        batch_size=16,
        pool_sizes={"boss": 1000, "elite": 0, "normal": 0, "weak": 0},
        config=config,
    )
    assert sum(alloc.quotas.values()) == 16
    assert alloc.quotas["elite"] == 0
    assert alloc.quotas["normal"] == 0
    assert alloc.quotas["flex"] >= 6  # 16 - boss(<=10) = >=6 fallback slots
    assert alloc.normal_min_unfilled
    assert alloc.elite_min_unfilled


def test_allocate_quotas_zero_batch_returns_empty():
    alloc = allocate_tier_quotas(
        batch_size=0,
        pool_sizes={"boss": 100, "elite": 100, "normal": 100, "weak": 0},
    )
    assert alloc.quotas == {}
    assert sum(alloc.quotas.values()) == 0


def test_buffer_sample_batch_respects_quota_end_to_end(monkeypatch):
    """End-to-end: build a buffer with 90% boss + 5% elite + 5% normal
    trajectories and confirm the sampled batch tier distribution matches the
    configured quota, not the priority skew."""
    pytest.importorskip("torch")
    from muzero.sts2_env.muzero_buffer import MuZeroReplayBuffer

    # Stub trajectory: only the metadata + len matter for the quota path,
    # but sample_batch unrolls steps so we need a minimal real trajectory.
    class _StubTrajectory:
        def __init__(self, *, encounter_tier: str, encounter_id: str):
            self.metadata = {
                "encounter_tier": encounter_tier,
                "encounter_id": encounter_id,
            }
            self.steps: list[dict] = []
            for _ in range(3):  # 3 steps so unroll_steps=2 fits
                self.steps.append({
                    "obs": np.zeros((2,), dtype=np.float32),  # tiny dense obs
                    "action": 0,
                    "semantic_action": 0,
                    "reward": 0.0,
                    "reward_components": np.zeros((4,), dtype=np.float32),
                    "root_value": 0.0,
                    "root_value_components": np.zeros((4,), dtype=np.float32),
                    "search_policy": np.zeros((4,), dtype=np.float32),
                    "semantic_policy": np.zeros((96,), dtype=np.float32),
                    "action_mask": np.ones((4,), dtype=np.float32),
                    "settlement_bonus": 0.0,
                    "wasteful_end_turn": False,
                })

        def __len__(self):
            return len(self.steps)

        def compute_target_values(self, *, discount: float, n_steps: int):
            return np.zeros((len(self.steps),), dtype=np.float32)

        def compute_target_value_components(self, *, discount: float, n_steps: int):
            return np.zeros((len(self.steps), 4), dtype=np.float32)

    config = TierQuotaConfig(
        targets={"boss": 0.60, "elite": 0.25, "normal": 0.15, "weak": 0.0},
        min_caps={"elite": 0.18, "normal": 0.10},
        max_caps={"boss": 0.65},
    )
    buffer = MuZeroReplayBuffer(capacity=200, tier_quota_config=config)

    # 90 boss, 5 elite, 5 normal — soft priorities would crush minor tiers.
    for i in range(90):
        buffer.trajectories.append(_StubTrajectory(encounter_tier="boss", encounter_id=f"BOSS_{i}"))
        buffer.priorities.append(1.0)
    for i in range(5):
        buffer.trajectories.append(_StubTrajectory(encounter_tier="elite", encounter_id=f"ELITE_{i}"))
        buffer.priorities.append(0.1)
    for i in range(5):
        buffer.trajectories.append(_StubTrajectory(encounter_tier="normal", encounter_id=f"NORMAL_{i}"))
        buffer.priorities.append(0.1)

    rng = np.random.default_rng(seed=0)
    boss_count_total = 0
    elite_count_total = 0
    normal_count_total = 0
    flex_count_total = 0
    n_batches = 30
    batch_size = 16
    for _ in range(n_batches):
        batch = buffer.sample_batch(
            batch_size=batch_size,
            unroll_steps=2,
            rng=rng,
            n_step_return=2,
        )
        info = batch["tier_quota_info"]
        assert info is not None, "tier_quota_info should be present when quota config is set"
        assert sum(info["quotas"].values()) == batch_size
        boss_count_total += int(info["quotas"].get("boss", 0))
        elite_count_total += int(info["quotas"].get("elite", 0))
        normal_count_total += int(info["quotas"].get("normal", 0))
        flex_count_total += int(info["quotas"].get("flex", 0))
        # Boss cap holds every batch.
        assert info["quotas"].get("boss", 0) <= 10
        assert info["quotas"].get("elite", 0) >= 3
        assert info["quotas"].get("normal", 0) >= 2

    total = n_batches * batch_size
    # The buffer in this test has 90 boss, 5 elite, 5 normal so the small
    # elite/normal pools push the achievable boss share below the 60%
    # target (min caps fire first, then proportional fills the rest). The
    # invariants we MUST hold per batch:
    #  * boss <= floor(0.65 * 16) = 10 (every batch).
    #  * elite >= ceil(0.18 * 16) = 3 (every batch — pool=5 satisfies).
    #  * normal >= ceil(0.10 * 16) = 2 (every batch — pool=5 satisfies).
    # Aggregate: boss can drop as low as ~0.40 because elite+normal
    # consume more than their targets when boss target=0.60 cannot fit.
    assert 0.40 <= boss_count_total / total <= 0.65
    assert elite_count_total / total >= 0.18
    assert normal_count_total / total >= 0.10


def test_buffer_legacy_path_unaffected_when_quota_disabled():
    pytest.importorskip("torch")
    from muzero.sts2_env.muzero_buffer import MuZeroReplayBuffer

    class _StubTrajectory:
        def __init__(self, encounter_tier: str):
            self.metadata = {"encounter_tier": encounter_tier, "encounter_id": "X"}
            self.steps: list[dict] = []
            for _ in range(3):
                self.steps.append({
                    "obs": np.zeros((2,), dtype=np.float32),
                    "action": 0,
                    "semantic_action": 0,
                    "reward": 0.0,
                    "reward_components": np.zeros((4,), dtype=np.float32),
                    "root_value": 0.0,
                    "root_value_components": np.zeros((4,), dtype=np.float32),
                    "search_policy": np.zeros((4,), dtype=np.float32),
                    "semantic_policy": np.zeros((96,), dtype=np.float32),
                    "action_mask": np.ones((4,), dtype=np.float32),
                    "settlement_bonus": 0.0,
                    "wasteful_end_turn": False,
                })

        def __len__(self):
            return len(self.steps)

        def compute_target_values(self, *, discount: float, n_steps: int):
            return np.zeros((len(self.steps),), dtype=np.float32)

        def compute_target_value_components(self, *, discount: float, n_steps: int):
            return np.zeros((len(self.steps), 4), dtype=np.float32)

    buffer = MuZeroReplayBuffer(capacity=10)  # quota disabled by default
    buffer.trajectories.append(_StubTrajectory("boss"))
    buffer.priorities.append(1.0)
    rng = np.random.default_rng(seed=1)
    batch = buffer.sample_batch(batch_size=2, unroll_steps=2, rng=rng, n_step_return=2)
    assert batch["tier_quota_info"] is None, "legacy path must not emit quota_info"
