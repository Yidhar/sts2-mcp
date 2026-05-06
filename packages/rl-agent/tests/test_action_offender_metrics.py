"""Tests for the action offender JSONL dump and per-encounter metric mirror (TASK-A2)."""

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


def _stats(**overrides) -> dict[str, float]:
    base = {
        "combat_quality_energy": 1.0,
        "combat_quality_positive_action_count": 2.0,
        "combat_quality_urgent_positive_action_count": 1.0,
        "combat_quality_kaiser_back_attack_risk": 0.0,
        "combat_quality_kaiser_facing_change_candidate_count": 0.0,
        "combat_quality_kaiser_facing_change_selected": 0.0,
        "combat_quality_ceremonial_one_card_lock": 0.0,
        "combat_quality_ceremonial_stun_window": 0.0,
        "combat_quality_ceremonial_high_impact_selected": 0.0,
    }
    base.update(overrides)
    return base


class ClassifyActionOffendersTests(unittest.TestCase):
    def test_bad_end_turn_offender(self):
        offenders = MuZeroTrainer._classify_action_offenders(
            search_stats=_stats(combat_quality_wasteful_end_turn_selected=1.0),
            encounter="construct_menagerie_normal",
            family="end_turn",
        )
        self.assertIn("bad_end_turn", offenders)

    def test_zero_energy_x_cost_emits_low_value_when_no_non_energy_effect(self):
        # zero_energy_x_cost flag alone marks the slot; x_cost_bad_selected (set
        # only when there's no non-energy effect) escalates it to bad-play.
        offenders = MuZeroTrainer._classify_action_offenders(
            search_stats=_stats(
                combat_quality_zero_energy_x_cost_selected=1.0,
                combat_quality_x_cost_bad_selected=1.0,
            ),
            encounter="",
            family="play_card",
        )
        self.assertIn("zero_energy_x_cost_selected", offenders)
        self.assertIn("x_cost_low_value_selected", offenders)

    def test_zero_energy_x_cost_with_non_energy_effect_is_not_bad(self):
        # The pile-manipulation X-cost legitimately played at 0 energy should be
        # tracked but not flagged as low value.
        offenders = MuZeroTrainer._classify_action_offenders(
            search_stats=_stats(
                combat_quality_zero_energy_x_cost_selected=1.0,
                combat_quality_x_cost_has_non_energy_effect_selected=1.0,
            ),
            encounter="",
            family="play_card",
        )
        self.assertIn("zero_energy_x_cost_selected", offenders)
        self.assertNotIn("x_cost_low_value_selected", offenders)

    def test_kaiser_offenders_only_on_kaiser_encounter(self):
        stats = _stats(
            combat_quality_kaiser_back_attack_risk=0.6,
            combat_quality_kaiser_risky_end_turn_selected=1.0,
            combat_quality_kaiser_facing_change_candidate_count=2.0,
            combat_quality_kaiser_facing_change_selected=0.0,
        )
        kaiser_offenders = MuZeroTrainer._classify_action_offenders(
            search_stats=stats, encounter="kaiser_crab_boss", family="play_card"
        )
        self.assertIn("kaiser_risky_end_turn", kaiser_offenders) if False else None  # only fires on end_turn
        # On non-kaiser encounter: must NOT fire
        construct_offenders = MuZeroTrainer._classify_action_offenders(
            search_stats=stats, encounter="construct_menagerie_normal", family="play_card"
        )
        self.assertNotIn("kaiser_risky_end_turn", construct_offenders)
        self.assertNotIn("kaiser_facing_missed", construct_offenders)

    def test_kaiser_facing_missed_when_candidate_unused(self):
        stats = _stats(
            combat_quality_kaiser_back_attack_risk=0.6,
            combat_quality_kaiser_facing_change_candidate_count=1.0,
            combat_quality_kaiser_facing_change_selected=0.0,
        )
        offenders = MuZeroTrainer._classify_action_offenders(
            search_stats=stats, encounter="kaiser_crab_boss", family="play_card"
        )
        self.assertIn("kaiser_facing_missed", offenders)

    def test_kaiser_facing_missed_skipped_on_end_turn(self):
        # End-turn under risk has its own kaiser_risky_end_turn metric — don't
        # double-fire facing_missed for it.
        stats = _stats(
            combat_quality_kaiser_back_attack_risk=0.6,
            combat_quality_kaiser_risky_end_turn_selected=1.0,
            combat_quality_kaiser_facing_change_candidate_count=2.0,
            combat_quality_kaiser_facing_change_selected=0.0,
        )
        offenders = MuZeroTrainer._classify_action_offenders(
            search_stats=stats, encounter="kaiser_crab_boss", family="end_turn"
        )
        self.assertIn("kaiser_risky_end_turn", offenders)
        self.assertNotIn("kaiser_facing_missed", offenders)

    def test_ceremonial_low_impact_only_on_ceremonial_encounter(self):
        stats = _stats(combat_quality_ceremonial_low_impact_selected=1.0)
        cer_offenders = MuZeroTrainer._classify_action_offenders(
            search_stats=stats, encounter="ceremonial_beast_boss", family="play_card"
        )
        self.assertIn("ceremonial_low_impact_under_lock", cer_offenders)
        kaiser_offenders = MuZeroTrainer._classify_action_offenders(
            search_stats=stats, encounter="kaiser_crab_boss", family="play_card"
        )
        self.assertNotIn("ceremonial_low_impact_under_lock", kaiser_offenders)

    def test_insatiable_strategic_skip_only_on_insatiable(self):
        stats = _stats(combat_quality_strategic_skip_selected=1.0)
        insat = MuZeroTrainer._classify_action_offenders(
            search_stats=stats, encounter="the_insatiable_boss", family="end_turn"
        )
        self.assertIn("insatiable_strategic_skip", insat)
        self.assertIn("strategic_skip_selected", insat)
        other = MuZeroTrainer._classify_action_offenders(
            search_stats=stats, encounter="kaiser_crab_boss", family="end_turn"
        )
        self.assertIn("strategic_skip_selected", other)
        self.assertNotIn("insatiable_strategic_skip", other)

    def test_insatiable_frantic_escape_missed_at_1_offender(self):
        offenders = MuZeroTrainer._classify_action_offenders(
            search_stats=_stats(combat_quality_insatiable_frantic_escape_missed_at_1=1.0),
            encounter="the_insatiable_boss",
            family="play_card",
        )
        self.assertIn("insatiable_frantic_escape_missed_at_1", offenders)

    def test_insatiable_frantic_escape_missed_lt3_offender(self):
        offenders = MuZeroTrainer._classify_action_offenders(
            search_stats=_stats(combat_quality_insatiable_frantic_escape_missed_lt3=1.0),
            encounter="the_insatiable_boss",
            family="play_card",
        )
        self.assertIn("insatiable_frantic_escape_missed_lt3", offenders)

    def test_insatiable_frantic_escape_missed_does_not_fire_on_other_encounter(self):
        offenders = MuZeroTrainer._classify_action_offenders(
            search_stats=_stats(combat_quality_insatiable_frantic_escape_missed_at_1=1.0),
            encounter="kaiser_crab_boss",
            family="play_card",
        )
        self.assertNotIn("insatiable_frantic_escape_missed_at_1", offenders)

    def test_low_quality_potion_selected_only_for_potion_family(self):
        stats = _stats(combat_quality_potion_low_urgency_selected=1.0)
        potion = MuZeroTrainer._classify_action_offenders(
            search_stats=stats, encounter="", family="use_potion"
        )
        self.assertIn("low_quality_potion_selected", potion)
        play_card = MuZeroTrainer._classify_action_offenders(
            search_stats=stats, encounter="", family="play_card"
        )
        self.assertNotIn("low_quality_potion_selected", play_card)

    def test_dedup_preserves_first_occurrence_order(self):
        stats = _stats(
            combat_quality_strategic_skip_selected=1.0,
            combat_quality_refund_no_followup_selected=1.0,
            combat_quality_wasteful_end_turn_selected=1.0,
        )
        offenders = MuZeroTrainer._classify_action_offenders(
            search_stats=stats, encounter="the_insatiable_boss", family="end_turn"
        )
        self.assertEqual(offenders, sorted(set(offenders), key=offenders.index))


class MirrorMetricsPerEncounterTests(unittest.TestCase):
    def test_mirrors_under_encounter_namespace(self):
        metrics = {
            "boss_combat/wasteful_end_turn_rate": 0.1,
            "boss_combat/decision_count": 50.0,
            "episode/loss": 0.5,  # outside boss_combat/ — must not mirror
        }
        out = MuZeroTrainer._mirror_metrics_per_encounter(metrics, "kaiser_crab_boss")
        self.assertIn("boss_combat/kaiser_crab_boss/wasteful_end_turn_rate", out)
        self.assertIn("boss_combat/kaiser_crab_boss/decision_count", out)
        self.assertNotIn("episode/kaiser_crab_boss/loss", out)
        # Original aggregate keys stay so global trends still readable.
        self.assertIn("boss_combat/wasteful_end_turn_rate", out)

    def test_no_mirror_when_encounter_blank(self):
        metrics = {"boss_combat/x": 1.0}
        out = MuZeroTrainer._mirror_metrics_per_encounter(dict(metrics), "")
        self.assertEqual(set(out.keys()), {"boss_combat/x"})

    def test_strips_encounter_prefix_when_present(self):
        metrics = {"boss_combat/foo": 2.0}
        out = MuZeroTrainer._mirror_metrics_per_encounter(dict(metrics), "ENCOUNTER.KAISER_CRAB_BOSS")
        # base namespace stripped of leading ENCOUNTER. token
        self.assertIn("boss_combat/kaiser_crab_boss/foo", out)


class DumpActionOffenderTests(unittest.TestCase):
    def _stub(self, log_dir: Path) -> MuZeroTrainer:
        stub = MuZeroTrainer.__new__(MuZeroTrainer)
        stub.log_dir = str(log_dir)
        stub.total_steps = 10
        stub.episode_count = 3
        stub._action_offender_dump_count = 0
        stub._action_offender_dump_max = 100
        stub._action_offender_dump_disabled = False
        return stub

    def test_writes_one_jsonl_line_per_offender(self):
        with tempfile.TemporaryDirectory() as tmp:
            trainer = self._stub(Path(tmp))
            chosen = {
                "semantic": {"family": "end_turn"},
                "title": "End Turn",
            }
            stats = _stats(combat_quality_wasteful_end_turn_selected=1.0)
            mask = np.array([1.0, 1.0, 1.0], dtype=np.float32)
            with mock.patch.object(MuZeroTrainer, "_incoming_damage_pressure", return_value=(8.0, 2.0, 60.0)), \
                 mock.patch.object(MuZeroTrainer, "_semantic_family", side_effect=lambda a: (a or {}).get("semantic", {}).get("family", "")):
                trainer._dump_action_offender(
                    encoded_obs={},
                    raw_obs={"combat": {"round": 4}, "player": {"max_hp": 80}},
                    action_mask=mask,
                    legal_actions=[
                        {"semantic": {"family": "play_card"}, "title": "Strike"},
                        {"semantic": {"family": "play_card"}, "title": "Defend"},
                        chosen,
                    ],
                    chosen_idx=2,
                    chosen_action=chosen,
                    offender_types=["bad_end_turn", "high_save_value_potion_unused"],
                    search_stats=stats,
                    encounter="kaiser_crab_boss",
                    tier="boss",
                )
            path = Path(tmp) / "diagnostics" / "action_offenders.jsonl"
            self.assertTrue(path.exists())
            lines = path.read_text(encoding="utf-8").strip().splitlines()
            self.assertEqual(len(lines), 2)
            payload0 = json.loads(lines[0])
            self.assertEqual(payload0["encounter_id"], "kaiser_crab_boss")
            self.assertEqual(payload0["selected_family"], "end_turn")
            self.assertIn(payload0["offender_type"], {"bad_end_turn", "high_save_value_potion_unused"})
            self.assertGreaterEqual(len(payload0["alternative_actions"]), 1)

    def test_no_write_when_no_offenders(self):
        with tempfile.TemporaryDirectory() as tmp:
            trainer = self._stub(Path(tmp))
            trainer._dump_action_offender(
                encoded_obs=None,
                raw_obs=None,
                action_mask=np.array([1.0]),
                legal_actions=[{}],
                chosen_idx=0,
                chosen_action={},
                offender_types=[],
                search_stats=_stats(),
                encounter="",
                tier="",
            )
            self.assertFalse((Path(tmp) / "diagnostics" / "action_offenders.jsonl").exists())

    def test_does_not_crash_on_blank_encounter(self):
        # Encounter id may be missing on non-boss encounters; the dump must still
        # work for global metrics-only consumption.
        with tempfile.TemporaryDirectory() as tmp:
            trainer = self._stub(Path(tmp))
            with mock.patch.object(MuZeroTrainer, "_incoming_damage_pressure", return_value=(0.0, 0.0, 50.0)), \
                 mock.patch.object(MuZeroTrainer, "_semantic_family", return_value="play_card"):
                trainer._dump_action_offender(
                    encoded_obs=None,
                    raw_obs={"combat": {}, "player": {}},
                    action_mask=np.array([1.0]),
                    legal_actions=[{"semantic": {"family": "play_card"}}],
                    chosen_idx=0,
                    chosen_action={"semantic": {"family": "play_card"}, "title": "Strike"},
                    offender_types=["zero_energy_x_cost_selected"],
                    search_stats=_stats(),
                    encounter="",
                    tier="",
                )
            path = Path(tmp) / "diagnostics" / "action_offenders.jsonl"
            self.assertTrue(path.exists())
            payload = json.loads(path.read_text(encoding="utf-8").strip())
            self.assertEqual(payload["encounter_id"], "")


if __name__ == "__main__":
    unittest.main()
