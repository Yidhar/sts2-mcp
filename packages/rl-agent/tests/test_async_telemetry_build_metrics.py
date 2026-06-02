from __future__ import annotations

import sys
from pathlib import Path


RL_AGENT_ROOT = Path(__file__).resolve().parents[1]
if str(RL_AGENT_ROOT) not in sys.path:
    sys.path.insert(0, str(RL_AGENT_ROOT))


class _RecordingWriter:
    def __init__(self) -> None:
        self.scalars: list[tuple[str, float, int]] = []

    def add_scalar(self, tag: str, value: float, step: int) -> None:
        self.scalars.append((str(tag), float(value), int(step)))


class _TrainerStub:
    def __init__(self) -> None:
        self.writer = _RecordingWriter()
        self.episode_count = 7
        self.total_steps = 123


def _tag_values(trainer: _TrainerStub) -> dict[str, float]:
    return {tag: value for tag, value, _step in trainer.writer.scalars}


def test_async_telemetry_replays_build_shop_death_deck_and_guard_metrics() -> None:
    """Async actor runs must expose the same Act1 diagnostics as sync runs.

    This catches two regressions at once:
    * the async writer must import DECISION_DOMAINS before replaying decision
      counts;
    * build/deck/shop/death-deck tags and shop hard-guard search suffixes must
      be visible from TensorBoard, otherwise multi-actor full-run training hides
      the current Act1 blockers.
    """

    from muzero.training.async_telemetry import log_async_episode_scalars

    trainer = _TrainerStub()
    episode_metrics = {
        "death_floor": 17,
        "decision_total": 1,
        "decision_counts": {"build": 1},
        "domain_search_means": {
            "build": {
                "shop_action_guard_open_applied": 1.0,
                "post_search_hard_guard_policy_retargeted": 1.0,
                "rest_site_smith_guard_applied": 1.0,
                "rest_site_smith_guard_override": 1.0,
                "rest_site_smith_guard_hp_ratio": 0.82,
            }
        },
        "deck_quality_v2": {
            "deck_size_raw": 18.0,
            "starter_count": 8.0,
            "raw_avg_damage_per_energy": 4.2,
        },
        "card_reward_metrics": {
            "card_reward_seen_count": 2.0,
            "card_reward_skip_rate": 0.5,
        },
        "shop_metrics": {
            "shop_seen_count": 1.0,
            "shop_open_rate": 1.0,
            "shop_remove_rate": 1.0,
        },
        "rest_site_metrics": {
            "rest_site_seen_count": 2.0,
            "rest_site_heal_rate": 0.5,
            "rest_site_smith_rate": 0.5,
            "rest_site_smith_available_high_hp_not_selected_rate": 0.25,
        },
        "deck_upgrade_metrics": {
            "deck_upgrade_smith_selected_count": 1.0,
            "deck_upgrade_smith_to_upgrade_seen_rate": 1.0,
            "deck_upgrade_seen_count": 1.0,
            "deck_upgrade_selected_rate": 1.0,
            "deck_upgrade_applied_rate": 1.0,
        },
        "episode_telemetry": {
            "rest_site_encounters": 2.0,
            "rest_heal_chosen": 1.0,
            "rest_smith_chosen": 1.0,
            "rest_skip_heal_at_low_hp": 0.0,
        },
    }

    log_async_episode_scalars(
        trainer=trainer,
        actor_index=0,
        episode_metrics=episode_metrics,
        actor_completed_episodes=[3],
    )

    values = _tag_values(trainer)
    assert values["decision/build_count"] == 1.0
    assert values["search/build/shop_action_guard_open_applied_rate"] == 1.0
    assert values["search/build/post_search_hard_guard_policy_retargeted_rate"] == 1.0
    assert values["search/build/rest_site_smith_guard_applied_rate"] == 1.0
    assert values["search/build/rest_site_smith_guard_override_rate"] == 1.0
    assert values["search/build/rest_site_smith_guard_hp_ratio_mean"] == 0.82
    assert values["deck/final_size"] == 18.0
    assert values["deck/final_starter_count"] == 8.0
    assert values["build/card_reward_seen"] == 2.0
    assert values["build/card_reward_skip_rate"] == 0.5
    assert values["build/shop_seen"] == 1.0
    assert values["build/shop_open_rate"] == 1.0
    assert values["build/shop_remove_rate"] == 1.0
    assert values["build/rest_site_seen"] == 2.0
    assert values["build/rest_site_heal_rate"] == 0.5
    assert values["build/rest_site_smith_rate"] == 0.5
    assert values["build/rest_site_smith_available_high_hp_not_selected_rate"] == 0.25
    assert values["build/deck_upgrade_smith_selected"] == 1.0
    assert values["build/deck_upgrade_smith_to_upgrade_seen_rate"] == 1.0
    assert values["build/deck_upgrade_seen"] == 1.0
    assert values["build/deck_upgrade_selected_rate"] == 1.0
    assert values["build/deck_upgrade_applied_rate"] == 1.0
    assert values["death_deck/size"] == 18.0
    assert values["death_deck/starter_count"] == 8.0
    assert values["env/rest_site_encounters"] == 2.0
    assert values["env/rest_heal_chosen"] == 1.0
    assert values["env/rest_smith_chosen"] == 1.0
    assert values["env/rest_skip_heal_at_low_hp"] == 0.0
    assert values["env/0_episodes"] == 3.0


def test_environment_episode_telemetry_helper_sanitizes_env_tags() -> None:
    from muzero.training.async_telemetry import log_environment_episode_telemetry

    trainer = _TrainerStub()

    log_environment_episode_telemetry(
        writer=trainer.writer,
        episode_index=11,
        episode_telemetry={
            "Rest Smith Chosen": 2,
            "rest-heal/exposure miss": 1,
        },
    )

    assert ("env/rest_smith_chosen", 2.0, 11) in trainer.writer.scalars
    assert ("env/rest_heal_exposure_miss", 1.0, 11) in trainer.writer.scalars
