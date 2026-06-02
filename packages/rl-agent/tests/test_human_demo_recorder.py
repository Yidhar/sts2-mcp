from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from muzero.demo_dataset import load_demo_dataset
from sts2_env.human_demo_recorder import HumanDemoRecorder


class HumanDemoRecorderTests(unittest.TestCase):
    def test_recorded_decision_loads_as_demo_sample(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            recorder = HumanDemoRecorder(output_dir=tmp, session_id="human_test")
            try:
                obs = {
                    "phase": "combat",
                    "player": {"hp": 50, "max_hp": 70},
                    "combat": {"round": 1, "energy": 3, "enemies": [{"hp": 10}]},
                }
                legal = [
                    {
                        "action_id": "play_card:0:enemy:1",
                        "family": "play_card",
                        "card": {
                            "id": "strike",
                            "runtime": {
                                "instance_uuid": "card-1",
                                "modified_cost": 1,
                            },
                        },
                    },
                    {"action_id": "end_turn", "family": "end_turn"},
                ]
                recorder.start_episode(
                    episode_id="ep-1",
                    encounter_id="kaiser_crab_boss",
                    obs=obs,
                    legal_actions=legal,
                    reset_kwargs={"seed": 1},
                )
                recorder.record_decision(
                    obs=obs,
                    legal_actions=legal,
                    selected_action=legal[0],
                    selected_action_index=0,
                    next_obs={**obs, "combat": {"round": 1, "enemies": [{"hp": 4}]}},
                    reward=1.0,
                    done=False,
                    truncated=False,
                    info={"episode_id": "ep-1", "encounter_id": "kaiser_crab_boss"},
                    reason_tags=["lethal"],
                    comment="hit the target",
                )
            finally:
                recorder.close()

            decisions = Path(tmp) / "human_test" / "decisions.jsonl"
            samples = load_demo_dataset(decisions, strict=True)
            self.assertEqual(len(samples), 1)
            self.assertEqual(samples[0].selected_action_id, "play_card:0:enemy:1")
            self.assertEqual(samples[0].reason_tags, ["lethal"])

    def test_terminal_decision_records_hp_loss_and_turns(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            recorder = HumanDemoRecorder(output_dir=tmp, session_id="human_terminal")
            try:
                obs = {
                    "phase": "combat",
                    "player": {"hp": 50, "max_hp": 70},
                    "combat": {"round": 1, "energy": 3, "enemies": [{"hp": 10}]},
                }
                next_obs = {
                    "phase": "combat",
                    "player": {"hp": 42, "max_hp": 70},
                    "combat": {"round": 3, "energy": 0, "enemies": [{"hp": 0}]},
                }
                legal = [{"action_id": "play_card:0:enemy:1", "family": "play_card"}]
                recorder.start_episode(
                    episode_id="ep-terminal",
                    encounter_id="normal_test",
                    obs=obs,
                    legal_actions=legal,
                    reset_kwargs={"seed": 2},
                )
                recorder.record_decision(
                    obs=obs,
                    legal_actions=legal,
                    selected_action=legal[0],
                    selected_action_index=0,
                    next_obs=next_obs,
                    reward=2.0,
                    done=True,
                    truncated=False,
                    info={"episode_id": "ep-terminal", "encounter_id": "normal_test"},
                )
            finally:
                recorder.close()

            decisions = Path(tmp) / "human_terminal" / "decisions.jsonl"
            samples = load_demo_dataset(decisions, strict=True)
            self.assertEqual(len(samples), 1)
            self.assertEqual(samples[0].outcome["hp_loss"], 8.0)
            self.assertEqual(samples[0].outcome["hp_loss_ratio"], 8.0 / 70.0)
            self.assertEqual(samples[0].outcome["turns"], 3)
            self.assertEqual(samples[0].outcome["combat_win"], True)


if __name__ == "__main__":
    unittest.main()
