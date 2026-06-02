from __future__ import annotations

import importlib.util
import json
import tempfile
import unittest
from pathlib import Path


RL_AGENT_ROOT = Path(__file__).resolve().parents[1]
SCRIPT_PATH = RL_AGENT_ROOT / "scripts" / "audit_human_demo_dataset.py"
spec = importlib.util.spec_from_file_location("audit_human_demo_dataset", SCRIPT_PATH)
audit_module = importlib.util.module_from_spec(spec)
assert spec and spec.loader
spec.loader.exec_module(audit_module)


def _row(*, selected: str = "play:defend", selected_index: int = 1, **overrides) -> dict:
    row = {
        "version": 1,
        "source": "human",
        "episode_id": "ep1",
        "encounter_id": "ENCOUNTER.TEST",
        "tier": "normal",
        "turn": 1,
        "step_in_turn": 0,
        "obs": {"phase": "combat", "player": {"hp": 50, "max_hp": 70}},
        "legal_actions": [
            {"action_id": "play:strike", "family": "play_card"},
            {"action_id": "play:defend", "family": "play_card"},
            {"action_id": "end_turn", "family": "end_turn"},
        ],
        "selected_action_id": selected,
        "selected_action_index": selected_index,
        "transition": {"player_hp_before": 50, "player_hp_after": 47, "hp_loss": 3},
        "outcome": {"combat_win": True, "hp_loss": 3, "turns": 2},
    }
    row.update(overrides)
    return row


class HumanDemoAuditTests(unittest.TestCase):
    def test_empty_decisions_dir_is_not_ready(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            session = root / "human_empty"
            session.mkdir()
            (session / "decisions.jsonl").write_text("", encoding="utf-8")
            (session / "episodes.jsonl").write_text(
                json.dumps({"event": "episode_start", "episode_id": "ep"}) + "\n",
                encoding="utf-8",
            )

            report = audit_module.audit_demo_dataset([root], min_samples=1)

        self.assertEqual(report["file_count"], 1)
        self.assertEqual(report["decision_row_count"], 0)
        self.assertEqual(report["usable_sample_count"], 0)
        self.assertFalse(report["ready_for_training"])
        self.assertIn("usable_sample_count<1", report["readiness_reasons"])
        self.assertTrue(all(not path.endswith("episodes.jsonl") for path in report["expanded_paths"]))

    def test_valid_rows_are_ready_and_hp_rates_are_counted(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "decisions.jsonl"
            path.write_text(
                "\n".join(json.dumps(_row(episode_id=f"ep{i}")) for i in range(3)) + "\n",
                encoding="utf-8",
            )

            report = audit_module.audit_demo_dataset([path], min_samples=3)

        self.assertEqual(report["decision_row_count"], 3)
        self.assertEqual(report["usable_sample_count"], 3)
        self.assertEqual(report["selected_action_id_in_legal_rate"], 1.0)
        self.assertEqual(report["selected_action_index_match_rate"], 1.0)
        self.assertEqual(report["transition_hp_loss_present_rate"], 1.0)
        self.assertEqual(report["outcome_hp_loss_present_rate"], 1.0)
        self.assertTrue(report["ready_for_training"])
        self.assertTrue(report["ready_for_hp_outcome_alignment"])

    def test_invalid_selected_action_blocks_training(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "decisions.jsonl"
            rows = [
                _row(),
                _row(selected="play:not_legal", selected_index=4),
            ]
            path.write_text(
                "\n".join(json.dumps(row) for row in rows) + "\n",
                encoding="utf-8",
            )

            report = audit_module.audit_demo_dataset([path], min_samples=1)

        self.assertEqual(report["decision_row_count"], 2)
        self.assertEqual(report["usable_sample_count"], 1)
        self.assertEqual(report["selected_action_id_in_legal_rate"], 0.5)
        self.assertFalse(report["ready_for_training"])
        self.assertIn("selected_action_not_in_legal", report["files"][0]["invalid_reasons"])


if __name__ == "__main__":
    unittest.main()
