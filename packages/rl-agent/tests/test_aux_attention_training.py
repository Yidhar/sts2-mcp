from __future__ import annotations

import importlib.util
from pathlib import Path
import sys
import tempfile
import time
import types
import unittest

import numpy as np
import torch
from gymnasium import spaces


RL_AGENT_ROOT = Path(__file__).resolve().parents[1]
STS2_ENV_ROOT = RL_AGENT_ROOT / "sts2_env"


def _load_sts2_env_modules():
    if str(RL_AGENT_ROOT) not in sys.path:
        sys.path.insert(0, str(RL_AGENT_ROOT))

    pkg_name = "sts2_env"
    if pkg_name not in sys.modules:
        pkg = types.ModuleType(pkg_name)
        pkg.__path__ = [str(STS2_ENV_ROOT)]
        sys.modules[pkg_name] = pkg

    loaded = {}
    for name in (
        "text_encoder",
        "semantic_action",
        "run_memory",
        "observation_common",
        "objective_heads",
        "aux_targets",
        "attention_blocks",
        "async_ready_collector",
        "observation_v3",
        "omni_attention_policy",
        "aux_maskable_ppo",
        "checkpoint",
    ):
        qualified = f"{pkg_name}.{name}"
        if qualified not in sys.modules:
            path = STS2_ENV_ROOT / f"{name}.py"
            spec = importlib.util.spec_from_file_location(qualified, path)
            if spec is None or spec.loader is None:
                raise RuntimeError(f"Failed to load module spec for {qualified}")
            module = importlib.util.module_from_spec(spec)
            module.__package__ = pkg_name
            sys.modules[qualified] = module
            spec.loader.exec_module(module)
        loaded[name] = sys.modules[qualified]
    return loaded


def _card(card_id: str, title: str, *, cost: int = 1, card_type: str = "Attack", **extra) -> dict:
    base = {
        "id": card_id,
        "title": title,
        "type": card_type,
        "cost": cost,
        "target": "SingleEnemy",
        "rarity": "common",
    }
    base.update(extra)
    return base


class AuxAttentionTrainingContractTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        modules = _load_sts2_env_modules()
        cls.aux_targets = modules["aux_targets"]
        cls.observation_v3 = modules["observation_v3"]
        cls.omni_attention_policy = modules["omni_attention_policy"]
        cls.aux_maskable_ppo = modules["aux_maskable_ppo"]
        cls.async_ready_collector = modules["async_ready_collector"]
        cls.checkpoint = modules["checkpoint"]

    def test_combat_aux_targets_capture_potion_relic_enemy_and_cycle_lines(self) -> None:
        aux_targets = self.aux_targets
        prev_obs = {
            "phase": "combat",
            "player": {
                "hp": 48,
                "max_hp": 80,
                "block": 7,
                "relics": [
                    {"id": "relic.shuriken", "title": "Shuriken", "description": "After attacks gain Strength."},
                    {"id": "relic.lantern", "title": "Lantern", "description": "Gain energy at start of combat."},
                ],
                "potions": [
                    {"id": "potion.fire", "title": "Fire Potion", "damage": 20, "target": "SingleEnemy"},
                    {"id": "potion.energy", "title": "Energy Potion", "energy": 2},
                ],
            },
            "combat": {
                "in_combat": True,
                "energy": 2,
                "max_energy": 3,
                "block": 7,
                "hand": [_card("card.hand.1", "Pommel Strike", damage=9, draw=1)],
                "draw_pile": {"cards": [_card("card.draw.1", "Strike"), _card("card.draw.2", "Shrug It Off", card_type="Skill")]},
                "discard_pile": {"cards": [_card("card.discard.1", "Burning Pact", card_type="Skill", description="Exhaust a card. Draw 2 cards.")]},
                "exhaust_pile": {"cards": [_card("card.exhaust.1", "Burn", card_type="Status")]},
                "enemies": [
                    {
                        "name": "Spiny Toad",
                        "combat_id": 101,
                        "current_hp": 44,
                        "max_hp": 50,
                        "block": 0,
                        "powers": [{"title": "Thorns", "description": "Punishes contact hits.", "amount": 3}],
                        "phase_rules": [{"trait": "threshold_stun", "description": "At hp <= 25 becomes stunned."}],
                        "intent": {"intent_type": "attack", "description": "Attack 9", "total_damage": 9, "damage_per_hit": 9, "repeats": 1},
                    }
                ],
            },
        }
        action = {
            "action_id": "play:pommel:toad",
            "kind": "play_card",
            "card": _card("card.hand.1", "Pommel Strike", damage=9, draw=1),
            "target": {"name": "Spiny Toad", "combat_id": 101, "side": "Enemy"},
        }
        next_obs = {
            "phase": "combat",
            "player": {
                "hp": 48,
                "max_hp": 80,
                "block": 7,
                "relics": prev_obs["player"]["relics"],
                "potions": prev_obs["player"]["potions"],
            },
            "combat": {
                "in_combat": True,
                "energy": 1,
                "max_energy": 3,
                "block": 7,
                "draw_pile": {"cards": [_card("card.draw.2", "Shrug It Off", card_type="Skill")]},
                "discard_pile": {"cards": [_card("card.discard.1", "Strike"), _card("card.discard.2", "Pommel Strike")]},
                "exhaust_pile": {"cards": [_card("card.exhaust.1", "Burn", card_type="Status")]},
                "enemies": [
                    {
                        "name": "Spiny Toad",
                        "combat_id": 101,
                        "current_hp": 35,
                        "max_hp": 50,
                        "block": 0,
                        "powers": [{"title": "Thorns", "description": "Punishes contact hits.", "amount": 3}],
                        "phase_rules": [{"trait": "threshold_stun", "description": "At hp <= 25 becomes stunned."}],
                        "intent": {"intent_type": "buff", "description": "Buff", "total_damage": 0, "damage_per_hit": 0, "repeats": 0},
                    }
                ],
            },
        }

        built = aux_targets.build_aux_targets(prev_obs, action, next_obs, legal_actions_before=[action, {"kind": "use_potion"}])
        traits = built["traits"]
        transition = built["transition"]

        self.assertEqual(float(built["objective_mask"]), 1.0)
        self.assertEqual(float(built["transition_mask"]), 1.0)
        self.assertEqual(float(built["traits_mask"]), 1.0)
        self.assertGreater(traits[0], 0.0)  # energy line
        self.assertGreater(traits[1], 0.0)  # potion line
        self.assertGreater(traits[2], 0.0)  # relic line
        self.assertGreater(traits[3], 0.0)  # enemy risk line
        self.assertGreater(traits[4], 0.0)  # draw cycle line
        self.assertGreater(traits[5], 0.0)  # discard/exhaust line
        self.assertEqual(traits[6], 0.0)  # build line
        self.assertEqual(traits[7], 0.0)  # route line
        self.assertLess(transition[2], 1.0)  # next energy ratio
        self.assertLess(transition[3], 1.0)  # next enemy hp ratio

    def test_build_and_route_actions_raise_domain_trait_lines(self) -> None:
        aux_targets = self.aux_targets
        build_obs = {
            "phase": "card_reward",
            "player": {
                "hp": 55,
                "max_hp": 80,
                "gold": 100,
                "deck_cards": [_card("card.deck.1", "Strike"), _card("card.deck.2", "Defend", card_type="Skill")],
                "relics": [],
                "potions": [],
            },
        }
        build_action = {
            "action_id": "reward:pick:burning_pact",
            "kind": "card_reward",
            "card": _card("card.reward.1", "Burning Pact", cost=1, card_type="Skill", description="Exhaust a card. Draw 2 cards."),
        }
        build_targets = aux_targets.build_aux_targets(build_obs, build_action, build_obs)
        self.assertEqual(float(build_targets["transition_mask"]), 0.0)
        self.assertGreater(build_targets["traits"][4], 0.0)  # draw cycle from card semantics
        self.assertGreater(build_targets["traits"][5], 0.0)  # discard/exhaust from card semantics
        self.assertEqual(build_targets["traits"][6], 1.0)
        self.assertEqual(build_targets["traits"][7], 0.0)
        self.assertEqual(float(build_targets["build_mask"]), 1.0)
        self.assertEqual(float(build_targets["route_mask"]), 0.0)
        self.assertGreater(build_targets["build"][2], 0.0)  # draw fit
        self.assertGreater(build_targets["build"][5], 0.0)  # cycle fit

        route_obs = {
            "phase": "map",
            "player": {
                "hp": 42,
                "max_hp": 80,
                "gold": 110,
                "deck_cards": [_card("card.deck.1", "Strike")],
                "relics": [],
                "potions": [],
            },
        }
        route_action = {
            "action_id": "map:7:2",
            "kind": "map",
            "coord": {"row": 7, "col": 2},
            "route_summary": {
                "reachable_node_count": 8,
                "max_depth": 6,
                "direct_child_count": 2,
                "forced_path_steps_before_branch": 1,
                "count_monster": 3,
                "count_elite": 1,
                "count_boss": 0,
                "count_event": 2,
                "count_question_mark": 1,
                "count_rest_site": 1,
                "count_shop": 1,
                "count_treasure": 1,
                "next_elite_steps": 2,
                "next_rest_steps": 3,
                "next_shop_steps": 2,
                "next_event_steps": 1,
                "next_question_mark_steps": 1,
                "next_treasure_steps": 4,
                "can_reach_rest_site_before_elite": True,
                "can_reach_elite_then_rest_site": True,
            },
        }
        route_targets = aux_targets.build_aux_targets(route_obs, route_action, route_obs)
        self.assertEqual(route_targets["traits"][6], 0.0)
        self.assertEqual(route_targets["traits"][7], 1.0)
        self.assertEqual(float(route_targets["build_mask"]), 0.0)
        self.assertEqual(float(route_targets["route_mask"]), 1.0)
        self.assertGreater(route_targets["route"][0], 0.0)
        self.assertGreater(route_targets["route"][7], 0.0)

    def test_selection_aux_targets_capture_pile_operator_and_runtime_lines(self) -> None:
        aux_targets = self.aux_targets

        selected_card = _card("card.discard.1", "Burning Pact", cost=1, card_type="Skill", description="Exhaust a card. Draw 2 cards.")
        combat_obs = {
            "phase": "card_selection",
            "player": {
                "hp": 44,
                "max_hp": 80,
                "gold": 90,
                "deck_cards": [_card("card.deck.1", "Strike"), selected_card],
                "relics": [{"id": "relic.anchor", "title": "Anchor"}],
                "potions": [{"id": "potion.fire", "title": "Fire Potion", "damage": 20}],
            },
            "combat": {
                "in_combat": True,
                "energy": 2,
                "max_energy": 3,
                "hand": [_card("card.hand.1", "Shrug It Off", card_type="Skill")],
                "draw_pile": {"cards": [_card("card.draw.1", "Strike")]},
                "discard_pile": {"cards": [selected_card]},
                "exhaust_pile": {"cards": [_card("card.exhaust.1", "Burn", card_type="Status")]},
                "play_pile": {"cards": []},
                "enemies": [],
            },
        }
        combat_action = {
            "action_id": "card_selection:select:discard",
            "kind": "card_selection",
            "selection_action": "select",
            "selection_semantics": "exhaust discard",
            "selection_prompt": "Choose a card from your discard pile to exhaust",
            "card": selected_card,
        }
        combat_targets = aux_targets.build_aux_targets(combat_obs, combat_action, combat_obs)
        self.assertEqual(float(combat_targets["selection_mask"]), 1.0)
        self.assertEqual(float(combat_targets["build_mask"]), 0.0)
        self.assertGreater(combat_targets["selection"][2], 0.0)  # source_discard
        self.assertGreater(combat_targets["selection"][9], 0.0)  # discard/exhaust line
        self.assertGreater(combat_targets["selection"][11], 0.0)  # combat runtime line

        upgrade_card = _card("card.deck.upgrade.1", "Twin Strike", damage=5, hits=2)
        upgrade_obs = {
            "phase": "card_selection",
            "player": {
                "hp": 58,
                "max_hp": 80,
                "gold": 120,
                "deck_cards": [_card("card.deck.1", "Strike"), upgrade_card],
                "relics": [],
                "potions": [],
            },
        }
        upgrade_action = {
            "action_id": "card_selection:select:upgrade",
            "kind": "card_selection",
            "selection_action": "select",
            "selection_semantics": "upgrade",
            "selection_prompt": "Choose a card to upgrade",
            "card": upgrade_card,
            "upgrade_preview": _card("card.deck.upgrade.1+", "Twin Strike+", damage=7, hits=2),
        }
        upgrade_targets = aux_targets.build_aux_targets(upgrade_obs, upgrade_action, upgrade_obs)
        self.assertEqual(float(upgrade_targets["selection_mask"]), 1.0)
        self.assertGreater(upgrade_targets["selection"][4], 0.0)  # source_deck
        self.assertGreater(upgrade_targets["selection"][6], 0.0)  # upgrade line
        self.assertEqual(float(upgrade_targets["selection"][11]), 0.0)  # no combat runtime

    def test_policy_aux_heads_expose_candidate_outputs_and_buffer_stores_targets(self) -> None:
        obs_v3 = self.observation_v3
        aux_buffer_mod = self.aux_maskable_ppo
        policy_mod = self.omni_attention_policy

        encoder = obs_v3.WorldTokenObservationEncoder(use_text=False)
        obs = {
            "phase": "combat",
            "player": {
                "hp": 48,
                "max_hp": 80,
                "block": 7,
                "gold": 75,
                "deck_cards": [_card("card.deck.1", "Strike"), _card("card.deck.2", "Burning Pact", card_type="Skill")],
                "relics": [{"id": "relic.shuriken", "title": "Shuriken", "description": "After attacks gain Strength."}],
                "potions": [{"id": "potion.fire", "title": "Fire Potion", "damage": 20, "target": "SingleEnemy"}],
            },
            "combat": {
                "in_combat": True,
                "energy": 2,
                "max_energy": 3,
                "hand": [_card("card.hand.1", "Pommel Strike", damage=9, draw=1)],
                "draw_pile": {"cards": [_card("card.draw.1", "Strike")]},
                "discard_pile": {"cards": [_card("card.discard.1", "Burning Pact", card_type="Skill", description="Exhaust a card. Draw 2 cards.")]},
                "exhaust_pile": {"cards": [_card("card.exhaust.1", "Burn", card_type="Status")]},
                "enemies": [
                    {
                        "name": "Cultist",
                        "combat_id": 201,
                        "current_hp": 44,
                        "max_hp": 50,
                        "block": 0,
                        "powers": [],
                        "intent": {"intent_type": "attack", "description": "Attack 10", "total_damage": 10, "damage_per_hit": 10, "repeats": 1},
                    }
                ],
            },
        }
        legal_actions = [
            {
                "action_id": "play:pommel:cultist",
                "kind": "play_card",
                "card": _card("card.hand.1", "Pommel Strike", damage=9, draw=1),
                "target": {"name": "Cultist", "combat_id": 201, "side": "Enemy"},
            }
        ]
        encoded = encoder.encode(obs, legal_actions=legal_actions)
        batched_obs = {key: np.expand_dims(value, axis=0) for key, value in encoded.items()}

        policy = policy_mod.STS2OmniAttentionPolicy(
            encoder.obs_space,
            spaces.Discrete(obs_v3.MAX_ACTIONS),
            lr_schedule=lambda _: 3e-4,
            d_model=64,
            n_heads=4,
            ffn_dim=128,
            world_layers=2,
            local_layers=1,
            decoder_layers=1,
            candidate_set_layers=1,
            world_bank_top_k=3,
        )
        self.assertTrue(hasattr(policy, "world_bank_poolers"))
        self.assertTrue(hasattr(policy, "world_bank_router_q"))
        self.assertTrue(hasattr(policy, "world_bank_cross_blocks"))
        self.assertEqual(int(policy._world_bank_top_k), 3)
        aux_outputs = policy.forward_aux_heads(batched_obs)
        self.assertEqual(tuple(aux_outputs["objective"].shape), (1, 4))
        self.assertEqual(tuple(aux_outputs["transition"].shape), (1, 8))
        self.assertEqual(tuple(aux_outputs["traits"].shape), (1, 8))
        self.assertEqual(tuple(aux_outputs["candidate_objective"].shape), (1, obs_v3.MAX_ACTIONS, 4))
        self.assertEqual(tuple(aux_outputs["candidate_transition"].shape), (1, obs_v3.MAX_ACTIONS, 8))
        self.assertEqual(tuple(aux_outputs["candidate_traits"].shape), (1, obs_v3.MAX_ACTIONS, 8))
        self.assertEqual(tuple(aux_outputs["candidate_build"].shape), (1, obs_v3.MAX_ACTIONS, 8))
        self.assertEqual(tuple(aux_outputs["candidate_selection"].shape), (1, obs_v3.MAX_ACTIONS, 12))
        self.assertEqual(tuple(aux_outputs["candidate_route"].shape), (1, obs_v3.MAX_ACTIONS, 8))
        arch_spec = policy.architecture_spec()
        self.assertEqual(str(arch_spec["architecture_version"]), "omni_attention_v1_frozen")
        self.assertEqual(
            list(arch_spec["world_banks"]),
            ["runtime", "support", "enemy", "build", "route", "powers", "history"],
        )
        self.assertEqual(int(arch_spec["world_bank_top_k"]), 3)
        bank_debug = policy.forward_world_bank_routing(batched_obs)
        self.assertEqual(str(bank_debug["architecture_version"]), "omni_attention_v1_frozen")
        num_world_banks = len(arch_spec["world_banks"])
        self.assertEqual(tuple(bank_debug["bank_weights"].shape), (1, obs_v3.MAX_ACTIONS, num_world_banks))
        self.assertEqual(tuple(bank_debug["bank_selected"].shape), (1, obs_v3.MAX_ACTIONS, num_world_banks))
        self.assertEqual(tuple(bank_debug["bank_available"].shape), (1, num_world_banks))
        selected_per_candidate = bank_debug["bank_selected"].sum(dim=-1)
        self.assertLessEqual(int(selected_per_candidate.max().item()), 3)
        self.assertEqual(int(encoded["candidate_query_target_owner_ids"][0]), obs_v3.OWNER_ENEMY_BASE)
        self.assertEqual(int(encoded["candidate_query_order_ids"][0]), 1)

        simple_obs_space = spaces.Dict(
            {
                "world_tokens": spaces.Box(-1.0, 1.0, shape=(2, 3), dtype=np.float32),
                "action_mask": spaces.Box(0.0, 1.0, shape=(4,), dtype=np.float32),
            }
        )
        buffer = aux_buffer_mod.AuxMaskableDictRolloutBuffer(
            buffer_size=1,
            observation_space=simple_obs_space,
            action_space=spaces.Discrete(4),
            device="cpu",
            n_envs=1,
        )
        buffer.reset()
        buffer.add(
            {"world_tokens": np.zeros((1, 2, 3), dtype=np.float32), "action_mask": np.ones((1, 4), dtype=np.float32)},
            np.asarray([[2]], dtype=np.int64),
            np.asarray([0.5], dtype=np.float32),
            np.asarray([0.0], dtype=np.float32),
            torch.tensor([0.1], dtype=torch.float32),
            torch.tensor([0.0], dtype=torch.float32),
            action_masks=np.ones((1, 4), dtype=np.float32),
            aux_targets={
                "objective": np.asarray([[0.1, 0.2, 0.3, 0.4]], dtype=np.float32),
                "objective_mask": np.asarray([1.0], dtype=np.float32),
                "transition": np.asarray([[0.5] * 8], dtype=np.float32),
                "transition_mask": np.asarray([1.0], dtype=np.float32),
                "traits": np.asarray([[1.0, 0.0, 1.0, 0.0, 1.0, 0.0, 0.0, 0.0]], dtype=np.float32),
                "traits_mask": np.asarray([1.0], dtype=np.float32),
                "build": np.asarray([[0.2] * 8], dtype=np.float32),
                "build_mask": np.asarray([1.0], dtype=np.float32),
                "selection": np.asarray([[1.0] * 12], dtype=np.float32),
                "selection_mask": np.asarray([1.0], dtype=np.float32),
                "route": np.asarray([[0.3] * 8], dtype=np.float32),
                "route_mask": np.asarray([1.0], dtype=np.float32),
            },
        )
        sample = next(buffer.get(batch_size=1))
        self.assertEqual(tuple(sample.aux_objective_targets.shape), (1, 4))
        self.assertEqual(tuple(sample.aux_transition_targets.shape), (1, 8))
        self.assertEqual(tuple(sample.aux_trait_targets.shape), (1, 8))
        self.assertEqual(tuple(sample.aux_build_targets.shape), (1, 8))
        self.assertEqual(tuple(sample.aux_selection_targets.shape), (1, 12))
        self.assertEqual(tuple(sample.aux_route_targets.shape), (1, 8))
        np.testing.assert_allclose(sample.aux_objective_targets.numpy()[0], np.asarray([0.1, 0.2, 0.3, 0.4], dtype=np.float32))

    def test_async_ready_collector_lets_fast_env_outpace_slow_env(self) -> None:
        collector_mod = self.async_ready_collector

        class _FakeAsyncEnv:
            def __init__(self, delay_s: float) -> None:
                self.delay_s = delay_s
                self.counter = 0
                self.observation_space = spaces.Dict(
                    {
                        "world_tokens": spaces.Box(-1.0, 1.0, shape=(1, 4), dtype=np.float32),
                        "action_mask": spaces.Box(0.0, 1.0, shape=(4,), dtype=np.float32),
                    }
                )
                self.action_space = spaces.Discrete(4)

            def _obs(self) -> dict[str, np.ndarray]:
                return {
                    "world_tokens": np.full((1, 4), float(self.counter), dtype=np.float32),
                    "action_mask": np.asarray([1.0, 1.0, 1.0, 1.0], dtype=np.float32),
                }

            def reset(self, *, seed: int | None = None, options: dict | None = None):
                self.counter = 0
                return self._obs(), {"action_mask": np.asarray([1.0, 1.0, 1.0, 1.0], dtype=np.float32)}

            def step(self, action: int):
                time.sleep(self.delay_s)
                self.counter += 1
                return self._obs(), 1.0, False, False, {"action_mask": np.asarray([1.0, 1.0, 1.0, 1.0], dtype=np.float32)}

            def close(self) -> None:
                return None

        collector = collector_mod.AsyncReadyCollector(
            [
                lambda: _FakeAsyncEnv(0.001),
                lambda: _FakeAsyncEnv(0.03),
            ]
        )
        collector.start()
        try:
            ready_pool: dict[int, object] = {}
            counts = {0: 0, 1: 0}
            while counts[0] + counts[1] < 8:
                ready_items = collector.drain_ready(min_items=1 if not ready_pool else 0, timeout_s=5.0)
                for item in ready_items:
                    if item.transition_info is not None:
                        counts[item.env_id] += 1
                    ready_pool[item.env_id] = item
                env_ids = sorted(ready_pool.keys())
                if env_ids:
                    collector.dispatch_actions(env_ids, np.zeros(len(env_ids), dtype=np.int64))
                    for env_id in env_ids:
                        ready_pool.pop(env_id, None)
            self.assertGreater(counts[0], counts[1])
        finally:
            collector.close()

    def test_async_rollout_collection_hits_exact_target_and_reuses_ready_states_between_rollouts(self) -> None:
        collector_mod = self.async_ready_collector
        aux_buffer_mod = self.aux_maskable_ppo

        class _FakeAsyncEnv:
            def __init__(self, delay_s: float) -> None:
                self.delay_s = delay_s
                self.counter = 0
                self.observation_space = spaces.Dict(
                    {
                        "world_tokens": spaces.Box(-1.0, 1.0, shape=(1, 4), dtype=np.float32),
                        "action_mask": spaces.Box(0.0, 1.0, shape=(4,), dtype=np.float32),
                    }
                )
                self.action_space = spaces.Discrete(4)

            def _obs(self) -> dict[str, np.ndarray]:
                return {
                    "world_tokens": np.full((1, 4), float(self.counter), dtype=np.float32),
                    "action_mask": np.asarray([1.0, 1.0, 1.0, 1.0], dtype=np.float32),
                }

            def reset(self, *, seed: int | None = None, options: dict | None = None):
                self.counter = 0
                return self._obs(), {"action_mask": np.asarray([1.0, 1.0, 1.0, 1.0], dtype=np.float32)}

            def step(self, action: int):
                time.sleep(self.delay_s)
                self.counter += 1
                info = {
                    "action_mask": np.asarray([1.0, 1.0, 1.0, 1.0], dtype=np.float32),
                    "aux_targets": {
                        "objective": np.asarray([0.1, 0.2, 0.3, 0.4], dtype=np.float32),
                        "objective_mask": 1.0,
                        "transition": np.asarray([0.5] * 8, dtype=np.float32),
                        "transition_mask": 1.0,
                        "traits": np.asarray([1.0, 0.0, 1.0, 0.0, 1.0, 0.0, 0.0, 0.0], dtype=np.float32),
                        "traits_mask": 1.0,
                        "build": np.asarray([0.2] * 8, dtype=np.float32),
                        "build_mask": 1.0,
                        "route": np.asarray([0.3] * 8, dtype=np.float32),
                        "route_mask": 1.0,
                    },
                    "bridge_info": {"step_timing_ms": {"total": 1.5, "after_wait": 0.75}},
                    "python_timing_ms": {"obs_encode": 0.25, "total": 0.5},
                }
                return self._obs(), 1.0, False, False, info

            def close(self) -> None:
                return None

        class _FakeLogger:
            def record(self, *_args, **_kwargs) -> None:
                return None

        class _FakePolicy:
            def __init__(self) -> None:
                self.version = 0
                self.forward_versions: list[int] = []
                self.training_modes: list[bool] = []

            def set_training_mode(self, mode: bool) -> None:
                self.training_modes.append(bool(mode))

            def __call__(self, obs_tensor, action_masks=None):
                batch_size = next(iter(obs_tensor.values())).shape[0]
                self.forward_versions.extend([self.version] * batch_size)
                actions = torch.zeros((batch_size,), dtype=torch.int64)
                values = torch.full((batch_size,), 0.5, dtype=torch.float32)
                log_probs = torch.zeros((batch_size,), dtype=torch.float32)
                return actions, values, log_probs

            def predict_values(self, obs_tensor):
                batch_size = next(iter(obs_tensor.values())).shape[0]
                return torch.full((batch_size,), 0.25, dtype=torch.float32)

        class _FakeAsyncHarness:
            collect_rollouts_async = aux_buffer_mod.AuxMaskablePPO.collect_rollouts_async
            _stack_aux_targets = aux_buffer_mod.AuxMaskablePPO._stack_aux_targets
            _extract_target = staticmethod(aux_buffer_mod.AuxMaskablePPO._extract_target)
            _extract_scalar = staticmethod(aux_buffer_mod.AuxMaskablePPO._extract_scalar)
            _extract_enemy_state_target = staticmethod(aux_buffer_mod.AuxMaskablePPO._extract_enemy_state_target)
            _extract_enemy_state_mask = staticmethod(aux_buffer_mod.AuxMaskablePPO._extract_enemy_state_mask)
            _summarize_timing_samples_ms = staticmethod(aux_buffer_mod.AuxMaskablePPO._summarize_timing_samples_ms)
            _summarize_scalar_samples = staticmethod(aux_buffer_mod.AuxMaskablePPO._summarize_scalar_samples)

            def __init__(self) -> None:
                self.policy = _FakePolicy()
                self.device = torch.device("cpu")
                self.logger = _FakeLogger()

            def _record_rollout_timing_stats(self, infos: list[dict[str, object]]) -> dict[str, float]:
                return {
                    "step.total.p50": float(len(infos)),
                }

        observation_space = spaces.Dict(
            {
                "world_tokens": spaces.Box(-1.0, 1.0, shape=(1, 4), dtype=np.float32),
                "action_mask": spaces.Box(0.0, 1.0, shape=(4,), dtype=np.float32),
            }
        )
        collector = collector_mod.AsyncReadyCollector(
            [
                lambda: _FakeAsyncEnv(0.001),
                lambda: _FakeAsyncEnv(0.03),
            ]
        )
        original_drain_ready = collector.drain_ready
        collector.drain_ready = lambda *, min_items=0, timeout_s=30.0, return_on_restart=False: original_drain_ready(  # type: ignore[method-assign]
            min_items=min_items,
            timeout_s=min(timeout_s, 0.5),
            return_on_restart=return_on_restart,
        )
        collector.start()
        try:
            harness = _FakeAsyncHarness()
            buffer = aux_buffer_mod.AsyncAuxMaskableDictRolloutBuffer(
                buffer_size=6,
                observation_space=observation_space,
                action_space=spaces.Discrete(4),
                device="cpu",
                gamma=0.99,
                gae_lambda=0.95,
                env_count=2,
            )
            payload = harness.collect_rollouts_async(collector, buffer, 6)

            self.assertEqual(int(payload["transitions"]), 6)
            self.assertEqual(int(buffer.pos), 6)
            env0_steps = int(np.count_nonzero(buffer.env_indices[: buffer.pos] == 0))
            env1_steps = int(np.count_nonzero(buffer.env_indices[: buffer.pos] == 1))
            self.assertEqual(env0_steps + env1_steps, 6)
            self.assertGreater(env0_steps, env1_steps)
            self.assertTrue(all(version == 0 for version in harness.policy.forward_versions))

            first_rollout_forward_count = len(harness.policy.forward_versions)
            harness.policy.version = 1
            buffer = aux_buffer_mod.AsyncAuxMaskableDictRolloutBuffer(
                buffer_size=2,
                observation_space=observation_space,
                action_space=spaces.Discrete(4),
                device="cpu",
                gamma=0.99,
                gae_lambda=0.95,
                env_count=2,
            )
            second_payload = harness.collect_rollouts_async(collector, buffer, 2)
            self.assertEqual(int(second_payload["transitions"]), 2)
            self.assertEqual(int(buffer.pos), 2)
            self.assertTrue(
                all(version == 1 for version in harness.policy.forward_versions[first_rollout_forward_count:])
            )
        finally:
            collector.close()

    def test_checkpoint_validation_rejects_pre_v4_attention_metadata(self) -> None:
        checkpoint_mod = self.checkpoint
        with self.assertRaisesRegex(ValueError, "attention_obs_v4"):
            checkpoint_mod.validate_attention_checkpoint_metadata(
                {
                    "observation_api_version": "attention_obs_v1",
                    "collector_mode": "sync",
                    "candidate_local_tokens": 48,
                }
            )

    def test_rotating_periodic_checkpoints_keep_only_latest_three(self) -> None:
        checkpoint_mod = self.checkpoint

        class _TinyModel:
            def __init__(self) -> None:
                self.policy = torch.nn.Linear(2, 2)

        model = _TinyModel()
        metadata_base = {
            "observation_api_version": "attention_obs_v2",
            "collector_mode": "async",
            "candidate_local_tokens": 24,
        }

        with tempfile.TemporaryDirectory() as tmp_dir:
            root = Path(tmp_dir)
            checkpoint_mod.save_online_checkpoint(
                model,
                root / "final",
                metadata={**metadata_base, "timesteps": 0},
            )
            for step in (10_000, 20_000, 30_000, 40_000):
                checkpoint_mod.save_rotating_online_checkpoint(
                    model,
                    root,
                    timesteps=step,
                    metadata={**metadata_base, "timesteps": step},
                    keep_last=3,
                )

            self.assertFalse((root / "step_000010000").exists())
            self.assertTrue((root / "step_000020000").exists())
            self.assertTrue((root / "step_000030000").exists())
            self.assertTrue((root / "step_000040000").exists())
            self.assertTrue((root / "final").exists())

    def test_amp_helpers_enable_cuda_bf16_and_disable_cpu(self) -> None:
        aux_buffer_mod = self.aux_maskable_ppo
        normalized_name, normalized_dtype = aux_buffer_mod._resolve_amp_dtype("bfloat16")
        self.assertEqual(normalized_name, "bf16")
        self.assertIs(normalized_dtype, torch.bfloat16)

        class _AmpHarness:
            amp_enabled = aux_buffer_mod.AuxMaskablePPO.amp_enabled
            amp_dtype_name = aux_buffer_mod.AuxMaskablePPO.amp_dtype_name

            def __init__(self, device: str, *, requested: bool) -> None:
                self.device = torch.device(device)
                self._amp_requested = requested
                self._amp_dtype_name = normalized_name
                self._amp_dtype = normalized_dtype
                self._amp_enabled = requested and self.device.type == "cuda"

        cpu_harness = _AmpHarness("cpu", requested=True)
        cuda_harness = _AmpHarness("cuda", requested=True)
        self.assertFalse(cpu_harness.amp_enabled)
        self.assertTrue(cuda_harness.amp_enabled)
        self.assertEqual(cuda_harness.amp_dtype_name, "bf16")


if __name__ == "__main__":
    unittest.main()
