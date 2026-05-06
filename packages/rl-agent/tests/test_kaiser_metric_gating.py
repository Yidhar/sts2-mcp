"""Regression tests for Kaiser-only metric/bias gating.

The May-2026 boss-recovery run exposed two coupled observability bugs:

* ordinary non-Kaiser enemies get ``incoming_damage_multiplier_norm == 0.5``
  from the default 1.0 multiplier, which must not become
  ``combat_quality_kaiser_back_attack_risk``;
* the selected-end-turn diagnostic JSONL must mirror that same Kaiser-only
  contract, otherwise per-encounter analysis is polluted and looks like every
  monster has Kaiser facing risk.
"""

from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import numpy as np


RL_AGENT_ROOT = Path(__file__).resolve().parents[1]
if str(RL_AGENT_ROOT) not in sys.path:
    sys.path.insert(0, str(RL_AGENT_ROOT))

from muzero.train import MuZeroTrainer
from sts2_env.boss_mechanics import build_boss_mechanics_context


class KaiserMetricGatingTests(unittest.TestCase):
    def test_non_kaiser_default_multiplier_does_not_emit_kaiser_risk(self) -> None:
        obs = {
            "encounter_id": "ENCOUNTER.THE_KIN_BOSS",
            "combat": {
                "enemies": [
                    {
                        "combat_id": "kin",
                        "id": "MONSTER.KIN",
                        "name": "The Kin",
                        "hp": 50,
                        "max_hp": 100,
                        "intent": {"intent_type": "attack", "total_damage": 12},
                        "powers": [],
                    }
                ],
            },
        }
        ctx = build_boss_mechanics_context(obs)

        # This is the generic normalized default and is expected to be 0.5.
        self.assertEqual(MuZeroTrainer._boss_context_max(ctx, "incoming_damage_multiplier_norm"), 0.5)
        # But it must not pollute the Kaiser namespace.
        self.assertEqual(MuZeroTrainer._kaiser_back_attack_risk_from_context(ctx, obs), 0.0)

    def test_same_back_attack_fields_are_ignored_outside_kaiser(self) -> None:
        ctx = {
            "encounter_key": "ENCOUNTER.THE_KIN_BOSS",
            "player_state": {
                "primary_back_attack_risk": 0.8,
                "back_attack_risk": 0.7,
                "back_attack_active": 1.0,
            },
            "enemy_states_by_index": [{"back_attack_active": 1.0, "incoming_damage_multiplier_norm": 0.9}],
        }
        obs = {"encounter_id": "ENCOUNTER.THE_KIN_BOSS", "combat": {"enemies": []}}

        self.assertEqual(MuZeroTrainer._kaiser_back_attack_risk_from_context(ctx, obs), 0.0)

    def test_kaiser_context_allows_explicit_back_attack_risk(self) -> None:
        ctx = {
            "encounter_key": "ENCOUNTER.KAISER_CRAB_BOSS",
            "player_state": {
                "primary_back_attack_risk": 0.8,
                "back_attack_risk": 0.7,
                "back_attack_active": 1.0,
            },
            "enemy_states_by_index": [{"back_attack_active": 1.0, "incoming_damage_multiplier_norm": 0.9}],
        }
        obs = {"encounter_id": "ENCOUNTER.KAISER_CRAB_BOSS", "combat": {"enemies": []}}

        self.assertEqual(MuZeroTrainer._kaiser_back_attack_risk_from_context(ctx, obs), 1.0)

    def test_end_turn_dump_reports_zero_kaiser_risk_for_non_kaiser(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            trainer = MuZeroTrainer.__new__(MuZeroTrainer)
            trainer.log_dir = tmp
            trainer.total_steps = 1
            trainer.episode_count = 1
            trainer._end_turn_context_dump_count = 0
            trainer._end_turn_context_dump_max = 100
            trainer._end_turn_context_dump_disabled = False

            raw_obs = {
                "encounter_id": "ENCOUNTER.THE_KIN_BOSS",
                "combat": {
                    "round": 2,
                    "hand": [],
                    "enemies": [
                        {
                            "combat_id": "kin",
                            "id": "MONSTER.KIN",
                            "name": "The Kin",
                            "hp": 50,
                            "max_hp": 100,
                            "intent": {"intent_type": "attack", "total_damage": 12},
                            "powers": [],
                        }
                    ],
                },
                "player": {"hp": 60, "max_hp": 80, "block": 0},
            }
            context = {
                "end_turn_indices": [0],
                "wasteful": False,
                "strategic_defer_available": False,
                "positive_progress_count": 0,
                "urgent_positive_count": 0,
                "deferable_positive_count": 0,
                "energy": 0.0,
            }
            legal = [{"semantic": {"family": "end_turn"}, "title": "End Turn"}]

            with mock.patch.object(MuZeroTrainer, "_incoming_damage_pressure", return_value=(12.0, 0.0, 60.0)), \
                 mock.patch.object(MuZeroTrainer, "_semantic_family", side_effect=lambda a: (a or {}).get("semantic", {}).get("family", "")), \
                 mock.patch.object(MuZeroTrainer, "_is_x_cost_action", return_value=False), \
                 mock.patch.object(MuZeroTrainer, "_is_kaiser_facing_change_action", return_value=False):
                trainer._dump_selected_end_turn_context(
                    encoded_obs={},
                    raw_obs=raw_obs,
                    action_mask=np.array([1.0], dtype=np.float32),
                    legal_actions=legal,
                    chosen_idx=0,
                    context=context,
                    search_policy=np.array([1.0], dtype=np.float32),
                    search_stats={},
                    action_diagnostics=None,
                    encounter="the_kin_boss",
                    tier="boss",
                )

            path = Path(tmp) / "diagnostics" / "end_turn_contexts.jsonl"
            payload = json.loads(path.read_text(encoding="utf-8").strip())
            self.assertEqual(payload["boss_context"]["incoming_damage_multiplier_norm"], 0.5)
            self.assertEqual(payload["boss_context"]["kaiser_back_attack_risk"], 0.0)
            self.assertEqual(payload["boss_context"]["kaiser_back_attack_active"], 0.0)


if __name__ == "__main__":
    unittest.main()
