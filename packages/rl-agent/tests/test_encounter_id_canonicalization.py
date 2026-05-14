from __future__ import annotations

from muzero.sts2_env.muzero_buffer import MuZeroReplayBuffer
from muzero.training.monitoring import RecentCombatMonitor


def test_recent_combat_monitor_matches_lowercase_metadata_to_tracked_hard_normal() -> None:
    monitor = RecentCombatMonitor(
        windows=(4,),
        tracked_encounters=("ENCOUNTER.OVICOPTER_NORMAL",),
        min_samples=1,
    )

    snapshot = monitor.record_episode(
        {
            "encounter_id": "encounter.ovicopter_normal",
            "encounter_tier": "normal",
            "terminated": True,
            "episode_total_reward": 1.0,
            "episode_length": 3,
        }
    )

    stats = snapshot[4]
    assert stats["groups"]["hard_normal"]["episodes"] == 1
    assert stats["groups"]["hard_normal"]["win_rate"] == 1.0
    assert stats["tracked_encounters"]["ENCOUNTER.OVICOPTER_NORMAL"]["episodes"] == 1


def test_replay_encounter_priority_weights_match_lowercase_metadata() -> None:
    buffer = MuZeroReplayBuffer(
        capacity=8,
        encounter_tier_weights={"normal": 1.15},
        encounter_priority_weights={"ENCOUNTER.OVICOPTER_NORMAL": 2.0},
    )

    info = buffer._trajectory_sampling_info(
        {"encounter_id": "encounter.ovicopter_normal", "encounter_tier": "normal"}
    )

    assert info["encounter_id"] == "ENCOUNTER.OVICOPTER_NORMAL"
    assert info["encounter_weight"] == 2.0
    assert info["sampling_multiplier"] == 2.3
    assert info["hard_encounter"] is True
    assert info["hard_normal"] is True


def test_replay_load_state_canonicalizes_legacy_lowercase_weights() -> None:
    buffer = MuZeroReplayBuffer(capacity=8)

    buffer.load_state_dict(
        {
            "capacity": 8,
            "encounter_priority_weights": {"encounter.ovicopter_normal": 2.0},
            "trajectories": [],
            "priorities": [],
            "total_transitions": 0,
        }
    )

    assert buffer.encounter_priority_weights == {"ENCOUNTER.OVICOPTER_NORMAL": 2.0}
