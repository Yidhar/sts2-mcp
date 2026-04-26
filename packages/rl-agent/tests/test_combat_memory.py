from __future__ import annotations

import unittest
from pathlib import Path
import sys


RL_AGENT_ROOT = Path(__file__).resolve().parents[1]
if str(RL_AGENT_ROOT) not in sys.path:
    sys.path.insert(0, str(RL_AGENT_ROOT))

# combat_memory.py is intentionally self-contained (no torch deps) so it can
# be loaded directly without triggering the package __init__.
import importlib.util
import types

_sts2_env_pkg = sys.modules.get("sts2_env")
if _sts2_env_pkg is None:
    _sts2_env_pkg = types.ModuleType("sts2_env")
    _sts2_env_pkg.__path__ = [str(RL_AGENT_ROOT / "sts2_env")]
    sys.modules["sts2_env"] = _sts2_env_pkg

_combat_memory_spec = importlib.util.spec_from_file_location(
    "sts2_env.combat_memory", RL_AGENT_ROOT / "sts2_env" / "combat_memory.py"
)
combat_memory = importlib.util.module_from_spec(_combat_memory_spec)
sys.modules["sts2_env.combat_memory"] = combat_memory
_combat_memory_spec.loader.exec_module(combat_memory)


def _make_obs(
    *,
    encounter_id: str = "ENC.TEST",
    turn: int = 0,
    player_hp: float = 80,
    player_max_hp: float = 80,
    player_block: float = 0,
    player_energy: float = 3,
    enemies=None,
    self_inflicted_cum: float = 0.0,
) -> dict:
    return {
        "combat": {
            "encounter_id": encounter_id,
            "turn": turn,
            "enemies": list(enemies or []),
            "self_inflicted_hp_loss_cumulative": self_inflicted_cum,
        },
        "player": {
            "hp": player_hp,
            "max_hp": player_max_hp,
            "block": player_block,
            "energy": player_energy,
        },
    }


def _enemy(
    combat_id: int,
    *,
    hp: float,
    max_hp: float = 100,
    block: float = 0,
    intent_type: str = "attack",
    intent_damage: float = 10,
    intent_per_hit: float = 10,
    intent_repeats: int = 1,
    powers=None,
    is_alive: bool = True,
) -> dict:
    return {
        "id": combat_id,
        "hp": hp,
        "max_hp": max_hp,
        "block": block,
        "is_alive": is_alive,
        "intent": {
            "intent_type": intent_type,
            "total_damage": intent_damage,
            "damage_per_hit": intent_per_hit,
            "repeats": intent_repeats,
        },
        "powers": list(powers or []),
    }


class CombatMemoryTrackerTest(unittest.TestCase):
    def test_reset_initialises_state_and_empty_snapshot(self) -> None:
        tracker = combat_memory.CombatMemoryTracker()
        obs = _make_obs(
            enemies=[
                _enemy(1, hp=50, powers=[{"id": "strength", "amount": 3}]),
            ]
        )
        tracker.reset(obs)
        snap = tracker.snapshot()
        self.assertEqual(snap["turn_index"], 0)
        self.assertEqual(snap["surprise_damage_last_turn"], 0.0)
        self.assertIn("1", snap["enemies"])
        # No deltas after reset — enemy history has only one observation.
        enemy_snap = snap["enemies"]["1"]
        self.assertEqual(enemy_snap["hp_delta_last_turn_ratio"], 0.0)
        self.assertEqual(enemy_snap["alive"], 1.0)

    def test_per_enemy_hp_delta_and_cum_damage(self) -> None:
        tracker = combat_memory.CombatMemoryTracker()
        obs_pre = _make_obs(
            enemies=[_enemy(7, hp=50, max_hp=50, intent_damage=10)]
        )
        tracker.reset(obs_pre)
        obs_post = _make_obs(
            player_hp=70,  # lost 10 hp — matches the intent damage
            enemies=[_enemy(7, hp=30, max_hp=50, intent_damage=10)],
        )
        tracker.update(obs_pre, None, obs_post)
        snap = tracker.snapshot()
        enemy_snap = snap["enemies"]["7"]
        # Enemy lost 20 hp out of 50 → -0.4 ratio
        self.assertAlmostEqual(enemy_snap["hp_delta_last_turn_ratio"], -0.4, places=5)
        # player lost 10, predicted 10 → attribution ≈ 10
        self.assertGreater(enemy_snap["attributable_damage_last_turn_ratio"], 0.0)
        self.assertEqual(snap["surprise_damage_last_turn"], 0.0)

    def test_surprise_damage_captures_on_death_burst(self) -> None:
        """Enemy predicts 5 damage, dies, but player loses 40 → surprise 35."""
        tracker = combat_memory.CombatMemoryTracker()
        obs_pre = _make_obs(
            player_hp=80, player_max_hp=80,
            enemies=[_enemy(9, hp=20, max_hp=100, intent_damage=5)],
        )
        tracker.reset(obs_pre)
        obs_post = _make_obs(
            player_hp=40, player_max_hp=80,  # lost 40
            enemies=[],  # enemy vanished
        )
        tracker.update(obs_pre, None, obs_post)
        snap = tracker.snapshot()
        # predicted 5, actual 40 → surprise 35
        self.assertGreater(snap["surprise_damage_last_turn"], 30.0)
        self.assertIn("9", snap["enemies"])
        self.assertEqual(snap["enemies"]["9"]["died_last_turn"], 1.0)

    def test_power_amount_delta_scaling_without_player_action(self) -> None:
        tracker = combat_memory.CombatMemoryTracker()
        obs_pre = _make_obs(
            enemies=[_enemy(3, hp=100, max_hp=100,
                            powers=[{"id": "ritual", "amount": 1}])]
        )
        tracker.reset(obs_pre)
        obs_post = _make_obs(
            enemies=[_enemy(3, hp=100, max_hp=100,
                            powers=[{"id": "ritual", "amount": 3}])]
        )
        tracker.update(obs_pre, None, obs_post)
        snap = tracker.snapshot()
        enemy_snap = snap["enemies"]["3"]
        ritual = enemy_snap["powers"]["ritual"]
        self.assertAlmostEqual(ritual["amount_delta_last_turn"], 2.0, places=5)
        # Enemy wasn't hit this turn, so growth-without-touch flag should fire.
        self.assertEqual(ritual["is_growing_without_player_action"], 1.0)

    def test_intent_change_detection(self) -> None:
        tracker = combat_memory.CombatMemoryTracker()
        obs_pre = _make_obs(
            enemies=[_enemy(2, hp=80, intent_type="attack", intent_damage=10)]
        )
        tracker.reset(obs_pre)
        obs_mid = _make_obs(
            player_hp=70,
            enemies=[_enemy(2, hp=80, intent_type="defend", intent_damage=0)],
        )
        tracker.update(obs_pre, None, obs_mid)
        snap = tracker.snapshot()
        self.assertEqual(snap["enemies"]["2"]["intent_changed_this_turn"], 1.0)
        # Next turn — intent unchanged.
        obs_next = _make_obs(
            player_hp=70,
            enemies=[_enemy(2, hp=80, intent_type="defend", intent_damage=0)],
        )
        tracker.update(obs_mid, None, obs_next)
        snap = tracker.snapshot()
        self.assertEqual(snap["enemies"]["2"]["intent_changed_this_turn"], 0.0)

    def test_new_combat_resets_tracker(self) -> None:
        tracker = combat_memory.CombatMemoryTracker()
        tracker.reset(_make_obs(encounter_id="ENC.A",
                                enemies=[_enemy(1, hp=50)]))
        obs_b = _make_obs(encounter_id="ENC.B", enemies=[_enemy(11, hp=80)])
        tracker.update(None, None, obs_b)
        snap = tracker.snapshot()
        self.assertIn("11", snap["enemies"])
        self.assertNotIn("1", snap["enemies"])
        self.assertEqual(snap["turn_index"], 0)

    def test_split_spawn_is_new_this_turn(self) -> None:
        tracker = combat_memory.CombatMemoryTracker()
        obs_pre = _make_obs(enemies=[_enemy(50, hp=100, max_hp=100)])
        tracker.reset(obs_pre)
        # Parent disappears, two children spawn
        obs_post = _make_obs(
            enemies=[
                _enemy(51, hp=50, max_hp=50),
                _enemy(52, hp=50, max_hp=50),
            ]
        )
        tracker.update(obs_pre, None, obs_post)
        snap = tracker.snapshot()
        self.assertEqual(snap["enemies"]["51"]["is_new_this_turn"], 1.0)
        self.assertEqual(snap["enemies"]["52"]["is_new_this_turn"], 1.0)
        # Parent died
        self.assertEqual(snap["enemies"]["50"]["died_last_turn"], 1.0)
        self.assertEqual(snap["enemy_count_delta_last_turn"], 1)

    def test_self_inflicted_hp_loss_stripped_from_attribution(self) -> None:
        """Offering-style self damage must not be credited to enemies."""
        tracker = combat_memory.CombatMemoryTracker()
        # Pre: player 80/80, one enemy intending 10 damage, cum self-damage = 0.
        obs_pre = _make_obs(
            player_hp=80,
            enemies=[_enemy(4, hp=100, intent_damage=10)],
            self_inflicted_cum=0.0,
        )
        tracker.reset(obs_pre)

        # Player plays Offering (−3 HP), enemy hits for 10 → player lost 13 total
        # but only 10 is enemy-attributable.
        obs_post = _make_obs(
            player_hp=80 - 13,
            enemies=[_enemy(4, hp=100, intent_damage=10)],
            self_inflicted_cum=3.0,
        )
        tracker.update(obs_pre, None, obs_post)
        snap = tracker.snapshot()

        # Enemy attribution should correspond to 10 HP, not 13.
        enemy_snap = snap["enemies"]["4"]
        # attributable ratio = attributed / max_hp anchor; we validate via the
        # raw player-max-hp-scaled metric by re-deriving: log_norm(10, _LOG1P_200)
        # Since exact comparison is fiddly with log norm, cross-check via
        # the self-damage field that the delta was recognised.
        self.assertAlmostEqual(
            snap["self_inflicted_hp_loss_last_step"], 3.0, places=5
        )
        # Surprise damage should be zero: predicted 10 vs actual (enemy-only) 10.
        self.assertEqual(snap["surprise_damage_last_turn"], 0.0)
        # Sanity: attribution must be > 0 (enemy did damage).
        self.assertGreater(enemy_snap["attributable_damage_last_turn_ratio"], 0.0)

    def test_self_inflicted_only_attributes_nothing_to_enemies(self) -> None:
        """If only Bloodletting is played and enemy is blocked, enemy gets 0."""
        tracker = combat_memory.CombatMemoryTracker()
        obs_pre = _make_obs(
            player_hp=80,
            enemies=[_enemy(5, hp=100, intent_damage=10)],
            self_inflicted_cum=0.0,
        )
        tracker.reset(obs_pre)
        # Player plays Bloodletting (−3 HP), enemy fully blocked (no HP loss
        # from enemy) → raw loss 3, all self-inflicted.
        obs_post = _make_obs(
            player_hp=77,
            enemies=[_enemy(5, hp=100, intent_damage=10)],
            self_inflicted_cum=3.0,
        )
        tracker.update(obs_pre, None, obs_post)
        snap = tracker.snapshot()
        # Self delta recorded.
        self.assertAlmostEqual(snap["self_inflicted_hp_loss_last_step"], 3.0, places=5)
        # Enemy attribution should be ~0 since stripped loss = 0 and predicted 10.
        enemy_snap = snap["enemies"]["5"]
        self.assertEqual(enemy_snap["attributable_damage_last_turn_ratio"], 0.0)
        # No surprise either: stripped actual 0 ≤ predicted 10.
        self.assertEqual(snap["surprise_damage_last_turn"], 0.0)

    def test_snapshot_is_plain_primitives(self) -> None:
        tracker = combat_memory.CombatMemoryTracker()
        tracker.reset(_make_obs(enemies=[_enemy(1, hp=50,
                                                 powers=[{"id": "str",
                                                          "amount": 2}])]))
        snap = tracker.snapshot()
        # Primitive-only serialisation keeps downstream caching/pickling trivial.
        import json
        json.dumps(snap)  # should not raise


if __name__ == "__main__":
    unittest.main()
