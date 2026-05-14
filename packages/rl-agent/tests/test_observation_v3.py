from __future__ import annotations

from collections import Counter
import importlib.util
from pathlib import Path
import sys
import types
import unittest
from unittest import mock

import numpy as np


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
    for name in ("text_encoder", "semantic_action", "run_memory", "observation_common", "observation_v3"):
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


class ObservationV3ContractTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        modules = _load_sts2_env_modules()
        cls.semantic_action = modules["semantic_action"]
        cls.observation_v3 = modules["observation_v3"]

    def test_entry_texts_are_deduped_and_batch_encoded_once_per_step(self) -> None:
        obs_v3 = self.observation_v3

        class _FakeEncoder:
            def __init__(self, embed_dim: int) -> None:
                self.embed_dim = embed_dim
                self.encode_calls = 0
                self.batch_calls: list[list[str]] = []

            def encode(self, text: str) -> np.ndarray:
                self.encode_calls += 1
                raise AssertionError(f"Unexpected single-text encode for: {text}")

            def encode_batch(self, texts: list[str]) -> np.ndarray:
                self.batch_calls.append(list(texts))
                out = np.zeros((len(texts), self.embed_dim), dtype=np.float32)
                for index, text in enumerate(texts):
                    out[index, :] = float((index + 1) * max(len(text), 1))
                return out

        encoder = obs_v3.WorldTokenObservationEncoder(use_text=True)
        fake_encoder = _FakeEncoder(obs_v3.TEXT_DIM)
        encoder._encoder = fake_encoder

        base_numeric = np.zeros(obs_v3.TOKEN_NUMERIC_DIM, dtype=np.float32)
        prefilled = np.full(obs_v3.TOKEN_TEXT_DIM, 3.0, dtype=np.float32)
        flat_entries = [
            encoder._entry("ENEMY_INTENT", base_numeric, owner_id=obs_v3.OWNER_ENEMY_BASE, entity_id=11, text="same text"),
            encoder._entry("ENEMY_POWER", base_numeric, owner_id=obs_v3.OWNER_ENEMY_BASE, entity_id=11, text="same text"),
            encoder._entry("TARGET_LOCAL", base_numeric, owner_id=obs_v3.OWNER_ENEMY_BASE, entity_id=12, text="other text"),
            encoder._entry("RELIC", base_numeric, owner_id=obs_v3.OWNER_RELIC, entity_id=21, text_embedding=prefilled),
        ]
        nested_entries = [
            [
                encoder._entry("SOURCE_CARD_LOCAL", base_numeric, owner_id=obs_v3.OWNER_HAND, entity_id=31, text="other text"),
            ]
        ]

        encoder._begin_text_registry()
        encoder._resolve_entry_text_embeddings(flat_entries, nested_entries)
        encoder._resolve_text_registry()

        self.assertEqual(fake_encoder.encode_calls, 0)
        self.assertEqual(len(fake_encoder.batch_calls), 1)
        self.assertEqual(fake_encoder.batch_calls[0], ["same text", "other text"])

        np.testing.assert_allclose(flat_entries[0]["text_embedding"], flat_entries[1]["text_embedding"])
        np.testing.assert_allclose(flat_entries[2]["text_embedding"], nested_entries[0][0]["text_embedding"])
        np.testing.assert_allclose(flat_entries[3]["text_embedding"], prefilled)
        self.assertEqual(flat_entries[0]["text"], "")
        self.assertEqual(nested_entries[0][0]["text"], "")

    def test_full_v3_encode_uses_single_batch_for_legacy_and_v3_texts(self) -> None:
        obs_v3 = self.observation_v3

        class _FakeEncoder:
            def __init__(self, embed_dim: int) -> None:
                self.embed_dim = embed_dim
                self.encode_calls = 0
                self.batch_calls: list[list[str]] = []

            def encode(self, text: str) -> np.ndarray:
                self.encode_calls += 1
                raise AssertionError(f"Unexpected single-text encode for: {text}")

            def encode_batch(self, texts: list[str]) -> np.ndarray:
                self.batch_calls.append(list(texts))
                out = np.zeros((len(texts), self.embed_dim), dtype=np.float32)
                for index, text in enumerate(texts):
                    out[index, :] = float(index + 1)
                return out

        encoder = obs_v3.WorldTokenObservationEncoder(use_text=True)
        fake_encoder = _FakeEncoder(obs_v3.TEXT_DIM)
        encoder._encoder = fake_encoder

        strike = _card("card.hand.1", "Strike", damage=6, description="Deal 6 damage.")
        shrug = _card("card.deck.1", "Shrug It Off", cost=1, card_type="Skill", block=8, draw=1, description="Gain Block. Draw 1 card.")
        obs = {
            "phase": "combat",
            "player": {
                "hp": 52,
                "max_hp": 80,
                "block": 5,
                "gold": 99,
                "deck_cards": [strike, shrug],
                "relics": [{"id": "relic.anchor", "title": "Anchor", "description": "Start each combat with Block."}],
                "potions": [{"id": "potion.fire", "title": "Fire Potion", "description": "Deal 20 damage.", "canonical_text": "Deal 20 damage to a target enemy."}],
            },
            "combat": {
                "in_combat": True,
                "energy": 3,
                "max_energy": 3,
                "hand": [strike],
                "draw_pile": {"cards": [shrug]},
                "discard_pile": {"cards": [strike]},
                "exhaust_pile": {"cards": []},
                "play_pile": {"cards": []},
                "enemies": [
                    {
                        "name": "Spiny Toad",
                        "model_id": "MONSTER.SPINY_TOAD",
                        "combat_id": 101,
                        "current_hp": 40,
                        "max_hp": 50,
                        "block": 3,
                        "powers": [{"title": "Thorns", "description": "Punishes contact hits.", "amount": 3}],
                        "intent": {"intent_type": "attack", "label": "Strike", "description": "Attack 9", "total_damage": 9, "damage_per_hit": 9, "repeats": 1},
                        "phase_rules": [{"trait": "threshold_stun", "description": "At hp <= 25 becomes stunned."}],
                    }
                ],
            },
        }
        legal_actions = [
            {
                "action_id": "play:card.hand.1:101",
                "kind": "play_card",
                "card": strike,
                "target": {"name": "Spiny Toad", "combat_id": 101, "side": "Enemy"},
            }
        ]

        encoded = encoder.encode(obs, legal_actions=legal_actions)

        self.assertEqual(fake_encoder.encode_calls, 0)
        self.assertEqual(len(fake_encoder.batch_calls), 1)
        self.assertGreater(len(fake_encoder.batch_calls[0]), 5)
        self.assertGreater(float(np.abs(encoded["world_tokens"][:, obs_v3.TOKEN_NUMERIC_DIM :]).sum()), 0.0)
        self.assertGreater(float(np.abs(encoded["candidate_local_tokens"][:, :, obs_v3.TOKEN_NUMERIC_DIM :]).sum()), 0.0)

    def test_direct_v3_encode_never_calls_dense_encode_and_uses_attention_obs_v2_local_shape(self) -> None:
        obs_v3 = self.observation_v3
        encoder = obs_v3.WorldTokenObservationEncoder(use_text=False)
        obs = {
            "phase": "combat",
            "player": {
                "hp": 60,
                "max_hp": 80,
                "block": 4,
                "gold": 50,
                "deck_cards": [_card("card.deck.1", "Strike"), _card("card.deck.2", "Defend", card_type="Skill")],
                "relics": [],
                "potions": [],
            },
            "combat": {
                "in_combat": True,
                "energy": 3,
                "max_energy": 3,
                "hand": [_card("card.hand.1", "Strike"), _card("card.hand.2", "Defend", card_type="Skill")],
                "draw_pile": {"cards": [_card("card.draw.1", "Shrug It Off", card_type="Skill", block=8, draw=1)]},
                "discard_pile": {"cards": []},
                "exhaust_pile": {"cards": []},
                "play_pile": {"cards": []},
                "enemies": [
                    {
                        "name": "Cultist",
                        "combat_id": 7,
                        "current_hp": 40,
                        "max_hp": 50,
                        "block": 0,
                        "intent": {"intent_type": "attack", "description": "Attack 10", "total_damage": 10, "damage_per_hit": 10, "repeats": 1},
                        "powers": [],
                    }
                ],
            },
        }
        legal_actions = [
            {
                "action_id": "play:strike:cultist",
                "kind": "play_card",
                "card": _card("card.hand.1", "Strike", damage=6),
                "target": {"name": "Cultist", "combat_id": 7, "side": "Enemy"},
            },
            {"action_id": "end_turn", "kind": "combat"},
        ]

        with mock.patch.object(
            obs_v3.obs_common.DenseObservationEncoder,
            "encode",
            side_effect=AssertionError("DenseObservationEncoder.encode should not be called"),
        ):
            encoded = encoder.encode(obs, legal_actions=legal_actions)

        self.assertEqual(encoded["candidate_local_tokens"].shape[1], obs_v3.MAX_CANDIDATE_LOCAL_TOKENS)
        self.assertEqual(obs_v3.MAX_CANDIDATE_LOCAL_TOKENS, 40)
        self.assertEqual(obs_v3.OBSERVATION_API_VERSION, "attention_obs_v5_pass_large")

    def test_structured_enemy_traits_and_runtime_piles_become_world_tokens(self) -> None:
        obs_v3 = self.observation_v3
        encoder = obs_v3.WorldTokenObservationEncoder(use_text=False)
        obs = {
            "player": {
                "hp": 60,
                "max_hp": 80,
                "block": 12,
                "gold": 100,
                "max_energy": 3,
                "deck": [],
                "deck_cards": [_card("card.deck.1", "Defend", cost=1, card_type="Skill")],
                "relics": [],
                "potions": [],
                "creature": {
                    "current_hp": 60,
                    "max_hp": 80,
                    "block": 12,
                    "powers": [],
                },
            },
            "combat": {
                "in_combat": True,
                "energy": 3,
                "max_energy": 3,
                "turn": 2,
                "hand": [_card("card.hand.1", "Strike")],
                "draw_pile": {"cards": [_card("card.draw.1", "Quick Slash"), _card("card.draw.2", "Pommel Strike")]},
                "discard_pile": {"cards": [_card("card.discard.1", "Bash"), _card("card.discard.2", "Shrug It Off", card_type="Skill")]},
                "exhaust_pile": {"cards": [_card("card.exhaust.1", "Burn", card_type="Status")]},
                "play_pile": {"cards": [_card("card.play.1", "Cleave")]},
                "enemies": [
                    {
                        "name": "Spiny Toad",
                        "model_id": "MONSTER.SPINY_TOAD",
                        "combat_id": 101,
                        "current_hp": 44,
                        "max_hp": 50,
                        "block": 0,
                        "is_alive": True,
                        "is_hittable": True,
                        "powers": [
                            {
                                "title": "Thorns",
                                "description": "Retaliates on contact.",
                                "amount": 3,
                                "display_amount": 3,
                                "type": "buff",
                                "stack_type": "stack",
                            }
                        ],
                        "intent": {
                            "intent_type": "attack",
                            "label": "Strike",
                            "description": "Attack 9",
                            "repeats": 1,
                            "total_damage": 9,
                            "damage_per_hit": 9,
                        },
                        "static_traits": [
                            {
                                "category": "static",
                                "trait": "contact_retaliate",
                                "description": "Punishes contact hits.",
                                "severity": "high",
                            }
                        ],
                        "reactive_triggers": [
                            {
                                "category": "reactive",
                                "trait": "retaliate",
                                "description": "On hit retaliates.",
                                "trigger_type": "on_hit",
                                "condition": "contact",
                                "effect_type": "retaliate",
                                "severity": "high",
                            }
                        ],
                        "phase_rules": [
                            {
                                "category": "phase",
                                "trait": "threshold_stun",
                                "description": "At hp <= 25, becomes stunned.",
                                "trigger_type": "on_hp_threshold",
                                "condition": "hp_le_25",
                                "effect_type": "stun_self",
                                "threshold": 25,
                                "severity": "medium",
                            }
                        ],
                        "combat_tags": ["retaliation"],
                        "danger_profile": {
                            "burst": 2,
                            "attrition": 4,
                            "scaling": 1,
                            "retaliation": 5,
                            "summon_pressure": 0,
                            "debuff_pressure": 0,
                            "phase_complexity": 2,
                            "volatility": 2,
                            "target_priority": 4,
                        },
                        "target_priority_hints": [{"priority": "high", "reason": "Remove before multi-hit."}],
                    }
                ],
            },
            "state": {"floor": 12},
        }

        encoded = encoder.encode(obs, legal_actions=[])
        active_indices = np.flatnonzero(encoded["world_token_mask"] > 0.5)
        token_names = [obs_v3.TOKEN_TYPES[int(encoded["world_token_type_ids"][idx])] for idx in active_indices]
        counts = Counter(token_names)

        self.assertEqual(counts["DRAW_PREVIEW_CARD"], 2)
        self.assertEqual(counts["DISCARD_CARD"], 2)
        self.assertEqual(counts["EXHAUST_CARD"], 1)
        self.assertEqual(counts["PLAY_PILE_CARD"], 1)
        self.assertEqual(counts["ENEMY_CORE"], 1)
        self.assertEqual(counts["ENEMY_INTENT"], 1)
        self.assertEqual(counts["ENEMY_POWER"], 1)
        self.assertGreaterEqual(counts["ENEMY_REACTIVE_TRAIT"], 2)
        self.assertGreaterEqual(counts["ENEMY_PHASE_RULE"], 1)

        reactive_indices = [
            idx for idx in active_indices if obs_v3.TOKEN_TYPES[int(encoded["world_token_type_ids"][idx])] == "ENEMY_REACTIVE_TRAIT"
        ]
        phase_indices = [
            idx for idx in active_indices if obs_v3.TOKEN_TYPES[int(encoded["world_token_type_ids"][idx])] == "ENEMY_PHASE_RULE"
        ]
        draw_indices = [
            idx for idx in active_indices if obs_v3.TOKEN_TYPES[int(encoded["world_token_type_ids"][idx])] == "DRAW_PREVIEW_CARD"
        ]
        discard_indices = [
            idx for idx in active_indices if obs_v3.TOKEN_TYPES[int(encoded["world_token_type_ids"][idx])] == "DISCARD_CARD"
        ]
        exhaust_indices = [
            idx for idx in active_indices if obs_v3.TOKEN_TYPES[int(encoded["world_token_type_ids"][idx])] == "EXHAUST_CARD"
        ]
        play_indices = [
            idx for idx in active_indices if obs_v3.TOKEN_TYPES[int(encoded["world_token_type_ids"][idx])] == "PLAY_PILE_CARD"
        ]

        self.assertTrue(all(encoded["world_entity_owner_ids"][idx] == obs_v3.OWNER_ENEMY_BASE for idx in reactive_indices))
        self.assertTrue(all(encoded["world_entity_owner_ids"][idx] == obs_v3.OWNER_ENEMY_BASE for idx in phase_indices))
        self.assertTrue(all(encoded["world_token_entity_ids"][idx] != 0 for idx in reactive_indices + phase_indices))

        self.assertTrue(all(encoded["world_entity_owner_ids"][idx] == obs_v3.OWNER_DRAW for idx in draw_indices))
        self.assertTrue(all(encoded["world_entity_owner_ids"][idx] == obs_v3.OWNER_DISCARD for idx in discard_indices))
        self.assertTrue(all(encoded["world_entity_owner_ids"][idx] == obs_v3.OWNER_EXHAUST for idx in exhaust_indices))
        self.assertTrue(all(encoded["world_entity_owner_ids"][idx] == obs_v3.OWNER_PLAY for idx in play_indices))

    def test_card_selection_actions_get_selection_query_and_local_tokens(self) -> None:
        obs_v3 = self.observation_v3
        semantic_action = self.semantic_action
        encoder = obs_v3.WorldTokenObservationEncoder(use_text=False)

        selected = _card("card.discard.1", "Bash", cost=2, damage=8)
        obs = {
            "phase": "card_selection",
            "player": {
                "hp": 52,
                "max_hp": 80,
                "block": 0,
                "gold": 75,
                "deck_cards": [
                    _card("card.deck.1", "Strike", damage=6),
                    _card("card.deck.2", "Defend", card_type="Skill", block=5),
                    selected,
                ],
                "relics": [{"id": "relic.anchor", "title": "Anchor"}],
                "potions": [{"id": "potion.fire", "title": "Fire Potion"}],
            },
            "combat": {
                "in_combat": True,
                "energy": 2,
                "max_energy": 3,
                "hand": [_card("card.hand.1", "Strike", damage=6)],
                "draw_pile": {"cards": [_card("card.draw.1", "Pommel Strike", damage=9, draw=1)]},
                "discard_pile": {"cards": [selected]},
                "exhaust_pile": {"cards": [_card("card.exhaust.1", "Burn", card_type="Status")]},
                "play_pile": {"cards": []},
                "enemies": [],
            },
        }
        legal_actions = [
            {
                "action_id": "combat_select:discard:0",
                "kind": "card_selection",
                "selection": "pick",
                "selection_semantics": "select discard card to exhaust",
                "selection_prompt": "Choose a discard card to exhaust",
                "card": selected,
            }
        ]

        encoded = encoder.encode(obs, legal_actions=legal_actions)
        signature = semantic_action.semantic_action_signature(legal_actions[0])

        self.assertEqual(signature["family"], "card_selection")
        self.assertEqual(signature["domain"], "selection")
        self.assertEqual(
            int(encoded["candidate_query_type_ids"][0]),
            obs_v3.TOKEN_TYPE_TO_ID["SELECTION_CANDIDATE"],
        )

        active_local = np.flatnonzero(encoded["candidate_local_masks"][0] > 0.5)
        local_type_ids = encoded["candidate_local_type_ids"][0, active_local].tolist()
        self.assertIn(obs_v3.TOKEN_TYPE_TO_ID["SELECTION_OPERATOR_LOCAL"], local_type_ids)
        self.assertIn(obs_v3.TOKEN_TYPE_TO_ID["SELECTION_SEMANTICS_LOCAL"], local_type_ids)
        self.assertIn(obs_v3.TOKEN_TYPE_TO_ID["SELECTION_POOL_CARD_LOCAL"], local_type_ids)

        selection_card_pos = local_type_ids.index(obs_v3.TOKEN_TYPE_TO_ID["SELECTION_POOL_CARD_LOCAL"])
        self.assertEqual(
            int(encoded["candidate_local_zone_ids"][0, active_local[selection_card_pos]]),
            obs_v3.TOKEN_ZONE_TO_ID["DISCARD"],
        )

    def test_combat_candidates_receive_local_state_pile_relic_and_potion_context(self) -> None:
        obs_v3 = self.observation_v3
        encoder = obs_v3.WorldTokenObservationEncoder(use_text=False)
        obs = {
            "phase": "combat",
            "player": {
                "hp": 48,
                "max_hp": 80,
                "block": 7,
                "gold": 75,
                "deck_cards": [
                    _card("card.deck.1", "Defend", cost=1, card_type="Skill"),
                    _card("card.deck.2", "Burning Pact", cost=1, card_type="Skill"),
                ],
                "relics": [
                    {
                        "id": "relic.lantern",
                        "title": "Lantern",
                        "description": "Gain energy at the start of combat.",
                    },
                    {
                        "id": "relic.shuriken",
                        "title": "Shuriken",
                        "description": "After playing attacks, gain Strength.",
                    },
                ],
                "potions": [
                    {
                        "id": "potion.fire",
                        "title": "Fire Potion",
                        "damage": 20,
                        "target": "SingleEnemy",
                    },
                    {
                        "id": "potion.energy",
                        "title": "Energy Potion",
                        "energy": 2,
                    },
                ],
            },
            "combat": {
                "in_combat": True,
                "energy": 2,
                "max_energy": 3,
                "stars": 1,
                "hand": [
                    {
                        **_card("card.hand.1", "Pommel Strike"),
                        "damage": 9,
                        "draw": 1,
                        "target": "SingleEnemy",
                    }
                ],
                "draw_pile": {"cards": [_card("card.draw.1", "Strike", cost=1), _card("card.draw.2", "Pommel Strike", cost=1)]},
                "discard_pile": {"cards": [_card("card.discard.1", "Strike", cost=1), _card("card.discard.2", "Pommel Strike", cost=1)]},
                "exhaust_pile": {"cards": [_card("card.exhaust.1", "Burn", cost=1, card_type="Status")]},
                "play_pile": {"cards": [_card("card.play.1", "Bash", cost=2)]},
                "enemies": [
                    {
                        "name": "Cultist",
                        "model_id": "MONSTER.CULTIST",
                        "combat_id": 201,
                        "current_hp": 44,
                        "max_hp": 50,
                        "block": 0,
                        "is_alive": True,
                        "is_hittable": True,
                        "powers": [
                            {
                                "title": "Thorns",
                                "description": "Whenever attacked, deal 3 damage back.",
                                "amount": 3,
                            }
                        ],
                        "intent": {
                            "intent_type": "attack",
                            "label": "Strike",
                            "description": "Attack 10",
                            "repeats": 1,
                            "total_damage": 10,
                            "damage_per_hit": 10,
                        },
                    }
                ],
            },
        }
        legal_actions = [
            {
                "action_id": "play:pommel:cultist",
                "kind": "play_card",
                "card": {
                    **_card("card.hand.1", "Pommel Strike"),
                    "damage": 9,
                    "draw": 1,
                    "target": "SingleEnemy",
                },
                "target": {"name": "Cultist", "combat_id": 201, "side": "Enemy"},
            }
        ]

        encoded = encoder.encode(obs, legal_actions=legal_actions)

        local_indices = np.flatnonzero(encoded["candidate_local_masks"][0] > 0.5)
        local_types = {
            obs_v3.TOKEN_TYPES[int(encoded["candidate_local_type_ids"][0, idx])]
            for idx in local_indices
        }

        self.assertIn("SOURCE_CARD_LOCAL", local_types)
        self.assertIn("PLAYER_STATE_LOCAL", local_types)
        self.assertIn("ENERGY_CONTEXT_LOCAL", local_types)
        self.assertIn("DRAW_CONTEXT_LOCAL", local_types)
        self.assertIn("DISCARD_CONTEXT_LOCAL", local_types)
        self.assertIn("EXHAUST_CONTEXT_LOCAL", local_types)
        self.assertIn("PLAY_PILE_CONTEXT_LOCAL", local_types)
        self.assertIn("RELIC_TRIGGER_LOCAL", local_types)
        self.assertIn("POTION_OPTION_LOCAL", local_types)
        self.assertIn("DRAW_BINDING_LOCAL", local_types)
        self.assertIn("DISCARD_BINDING_LOCAL", local_types)
        binding_types = {
            token_type
            for token_type in local_types
            if token_type.endswith("_BINDING_LOCAL")
        }
        self.assertEqual(binding_types, {"DRAW_BINDING_LOCAL", "DISCARD_BINDING_LOCAL"})
        self.assertIn("CYCLE_PLAN_LOCAL", local_types)
        self.assertIn("RELIC_POTION_GRAPH_LOCAL", local_types)
        self.assertIn("ENERGY_BUDGET_LOCAL", local_types)
        self.assertIn("TARGET_LOCAL", local_types)
        self.assertIn("ENEMY_INTENT", local_types)
        self.assertIn("ENEMY_POWER", local_types)
        self.assertIn("TARGET_REACTION_LOCAL", local_types)
        self.assertIn("DRAW_PREVIEW_CARD", local_types)
        self.assertIn("DISCARD_CARD", local_types)

        self.assertEqual(int(encoded["candidate_query_owner_ids"][0]), obs_v3.OWNER_HAND)
        self.assertEqual(
            int(encoded["candidate_query_role_ids"][0]),
            obs_v3.TOKEN_ROLE_TO_ID["QUERY_COMBAT"],
        )
        self.assertEqual(int(encoded["candidate_query_target_owner_ids"][0]), obs_v3.OWNER_ENEMY_BASE)
        self.assertGreater(int(encoded["candidate_query_target_entity_ids"][0]), 0)
        self.assertEqual(int(encoded["candidate_query_order_ids"][0]), 1)

        source_idx = next(
            idx
            for idx in local_indices
            if obs_v3.TOKEN_TYPES[int(encoded["candidate_local_type_ids"][0, idx])] == "SOURCE_CARD_LOCAL"
        )
        draw_bind_idx = next(
            idx
            for idx in local_indices
            if obs_v3.TOKEN_TYPES[int(encoded["candidate_local_type_ids"][0, idx])] == "DRAW_BINDING_LOCAL"
        )
        discard_bind_idx = next(
            idx
            for idx in local_indices
            if obs_v3.TOKEN_TYPES[int(encoded["candidate_local_type_ids"][0, idx])] == "DISCARD_BINDING_LOCAL"
        )
        target_reaction_idx = next(
            idx
            for idx in local_indices
            if obs_v3.TOKEN_TYPES[int(encoded["candidate_local_type_ids"][0, idx])] == "TARGET_REACTION_LOCAL"
        )

        source_entity = int(encoded["candidate_local_entity_ids"][0, source_idx])
        self.assertEqual(int(encoded["candidate_local_entity_ids"][0, draw_bind_idx]), source_entity)
        self.assertEqual(int(encoded["candidate_local_entity_ids"][0, discard_bind_idx]), source_entity)
        self.assertEqual(int(encoded["candidate_local_owner_ids"][0, draw_bind_idx]), obs_v3.OWNER_DRAW)
        self.assertEqual(int(encoded["candidate_local_owner_ids"][0, discard_bind_idx]), obs_v3.OWNER_DISCARD)
        self.assertEqual(int(encoded["candidate_local_zone_ids"][0, draw_bind_idx]), obs_v3.TOKEN_ZONE_TO_ID["DRAW"])
        self.assertEqual(int(encoded["candidate_local_zone_ids"][0, discard_bind_idx]), obs_v3.TOKEN_ZONE_TO_ID["DISCARD"])
        self.assertGreater(int(encoded["candidate_local_order_ids"][0, draw_bind_idx]), 0)
        self.assertGreater(int(encoded["candidate_local_order_ids"][0, discard_bind_idx]), 0)
        self.assertEqual(int(encoded["candidate_local_owner_ids"][0, target_reaction_idx]), obs_v3.OWNER_ENEMY_BASE)
        self.assertEqual(
            int(encoded["candidate_local_entity_ids"][0, target_reaction_idx]),
            int(encoded["candidate_query_target_entity_ids"][0]),
        )

    def test_x_cost_combat_candidates_emit_joint_energy_budget_and_support_graph(self) -> None:
        obs_v3 = self.observation_v3
        encoder = obs_v3.WorldTokenObservationEncoder(use_text=False)
        obs = {
            "phase": "combat",
            "player": {
                "hp": 46,
                "max_hp": 80,
                "block": 3,
                "gold": 90,
                "deck_cards": [_card("card.deck.1", "Whirlwind", cost=0, x_cost=True), _card("card.deck.2", "Defend", card_type="Skill")],
                "relics": [
                    {
                        "id": "relic.lantern",
                        "title": "Lantern",
                        "description": "Gain energy at the start of combat.",
                    }
                ],
                "potions": [
                    {
                        "id": "potion.energy",
                        "title": "Energy Potion",
                        "energy": 2,
                    }
                ],
            },
            "combat": {
                "in_combat": True,
                "energy": 2,
                "max_energy": 3,
                "hand": [
                    {
                        **_card("card.hand.x1", "Whirlwind", cost=0, x_cost=True),
                        "damage": 5,
                        "hits": 2,
                        "target": "AllEnemies",
                    }
                ],
                "draw_pile": {"cards": [_card("card.draw.1", "Strike"), _card("card.draw.2", "Whirlwind", cost=0, x_cost=True)]},
                "discard_pile": {"cards": [_card("card.discard.1", "Whirlwind", cost=0, x_cost=True)]},
                "exhaust_pile": {"cards": []},
                "play_pile": {"cards": []},
                "enemies": [
                    {
                        "name": "Spiny Toad",
                        "combat_id": 301,
                        "current_hp": 30,
                        "max_hp": 40,
                        "block": 0,
                        "powers": [{"title": "Thorns", "description": "Punishes contact hits.", "amount": 3}],
                        "intent": {"intent_type": "attack", "description": "Attack 8", "total_damage": 8, "damage_per_hit": 8, "repeats": 1},
                    }
                ],
            },
        }
        legal_actions = [
            {
                "action_id": "play:whirlwind",
                "kind": "play_card",
                "card": {
                    **_card("card.hand.x1", "Whirlwind", cost=0, x_cost=True),
                    "damage": 5,
                    "hits": 2,
                    "target": "AllEnemies",
                },
                "target": {"name": "Spiny Toad", "combat_id": 301, "side": "Enemy"},
            }
        ]

        encoded = encoder.encode(obs, legal_actions=legal_actions)
        local_indices = np.flatnonzero(encoded["candidate_local_masks"][0] > 0.5)
        energy_budget_idx = next(
            idx
            for idx in local_indices
            if obs_v3.TOKEN_TYPES[int(encoded["candidate_local_type_ids"][0, idx])] == "ENERGY_BUDGET_LOCAL"
        )
        support_graph_idx = next(
            idx
            for idx in local_indices
            if obs_v3.TOKEN_TYPES[int(encoded["candidate_local_type_ids"][0, idx])] == "RELIC_POTION_GRAPH_LOCAL"
        )

        energy_budget = encoded["candidate_local_tokens"][0, energy_budget_idx, : obs_v3.TOKEN_NUMERIC_DIM]
        support_graph = encoded["candidate_local_tokens"][0, support_graph_idx, : obs_v3.TOKEN_NUMERIC_DIM]

        self.assertGreater(energy_budget[4], 0.5)  # x-cost flag
        self.assertGreater(energy_budget[6], 0.0)  # energy potion support
        self.assertGreater(energy_budget[8], 0.0)  # relic energy signal
        self.assertGreater(energy_budget[11], 0.5)  # can expand with support
        self.assertGreater(support_graph[4], 0.0)  # energy potion line present
        self.assertGreater(support_graph[12], 0.0)  # contact-punish cover from support graph

    def test_card_reward_actions_emit_card_reward_local_token(self) -> None:
        obs_v3 = self.observation_v3
        encoder = obs_v3.WorldTokenObservationEncoder(use_text=False)
        obs = {
            "phase": "card_reward",
            "player": {
                "hp": 55,
                "max_hp": 80,
                "block": 0,
                "deck_cards": [_card("card.deck.1", "Strike"), _card("card.deck.2", "Defend", card_type="Skill")],
                "relics": [],
                "potions": [],
            },
        }
        legal_actions = [
            {
                "action_id": "reward:pick:inflame",
                "kind": "card_reward",
                "card": _card("card.reward.1", "Inflame", cost=1, card_type="Power"),
            }
        ]

        encoded = encoder.encode(obs, legal_actions=legal_actions)
        local_indices = np.flatnonzero(encoded["candidate_local_masks"][0] > 0.5)
        local_types = [
            obs_v3.TOKEN_TYPES[int(encoded["candidate_local_type_ids"][0, idx])]
            for idx in local_indices
        ]

        self.assertIn("CARD_REWARD_LOCAL", local_types)
        self.assertIn("BUILD_STATE_LOCAL", local_types)
        self.assertIn("DECK_SYNERGY_LOCAL", local_types)
        self.assertEqual(int(encoded["candidate_query_owner_ids"][0]), obs_v3.OWNER_REWARD)
        self.assertEqual(
            int(encoded["candidate_query_role_ids"][0]),
            obs_v3.TOKEN_ROLE_TO_ID["QUERY_BUILD"],
        )

    def test_combat_card_selection_uses_selection_query_and_true_source_pile_binding(self) -> None:
        obs_v3 = self.observation_v3
        encoder = obs_v3.WorldTokenObservationEncoder(use_text=False)
        selected_card = _card("card.discard.2", "Second Wind", cost=1, card_type="Skill")
        obs = {
            "phase": "card_selection",
            "player": {
                "hp": 47,
                "max_hp": 80,
                "block": 6,
                "gold": 88,
                "deck_cards": [_card("card.deck.1", "Strike"), _card("card.deck.2", "Defend", card_type="Skill"), selected_card],
                "relics": [],
                "potions": [],
            },
            "combat": {
                "in_combat": True,
                "energy": 2,
                "max_energy": 3,
                "hand": [_card("card.hand.1", "Bash", cost=2)],
                "draw_pile": {"cards": [_card("card.draw.1", "Shrug It Off", card_type="Skill")]},
                "discard_pile": {
                    "cards": [
                        _card("card.discard.1", "Burn", card_type="Status", cost=0),
                        selected_card,
                    ]
                },
                "exhaust_pile": {"cards": []},
                "play_pile": {"cards": []},
                "enemies": [
                    {
                        "name": "Cultist",
                        "combat_id": 19,
                        "current_hp": 33,
                        "max_hp": 50,
                        "block": 0,
                        "intent": {"intent_type": "attack", "description": "Attack 8", "total_damage": 8, "damage_per_hit": 8, "repeats": 1},
                        "powers": [],
                    }
                ],
            },
        }
        legal_actions = [
            {
                "action_id": "card_selection:select:1",
                "kind": "card_selection",
                "selection_action": "select",
                "selection_semantics": "discard",
                "selection_prompt": "Choose a card from your discard pile",
                "screen_type": "NCardSelectionScreen",
                "card": selected_card,
            }
        ]

        encoded = encoder.encode(obs, legal_actions=legal_actions)
        self.assertEqual(
            int(encoded["candidate_query_role_ids"][0]),
            obs_v3.TOKEN_ROLE_TO_ID["QUERY_SELECTION"],
        )
        self.assertEqual(
            int(encoded["candidate_query_type_ids"][0]),
            obs_v3.TOKEN_TYPE_TO_ID["SELECTION_CANDIDATE"],
        )
        self.assertEqual(int(encoded["candidate_query_owner_ids"][0]), obs_v3.OWNER_DISCARD)
        self.assertEqual(int(encoded["candidate_query_zone_ids"][0]), obs_v3.TOKEN_ZONE_TO_ID["DISCARD"])
        self.assertEqual(int(encoded["candidate_query_order_ids"][0]), 2)

        local_indices = np.flatnonzero(encoded["candidate_local_masks"][0] > 0.5)
        source_idx = next(
            idx
            for idx in local_indices
            if obs_v3.TOKEN_TYPES[int(encoded["candidate_local_type_ids"][0, idx])] == "SELECTION_POOL_CARD_LOCAL"
        )
        self.assertEqual(int(encoded["candidate_local_owner_ids"][0, source_idx]), obs_v3.OWNER_DISCARD)
        self.assertEqual(int(encoded["candidate_local_zone_ids"][0, source_idx]), obs_v3.TOKEN_ZONE_TO_ID["DISCARD"])
        self.assertEqual(int(encoded["candidate_local_order_ids"][0, source_idx]), 2)

    def test_route_actions_receive_route_risk_and_value_local_tokens(self) -> None:
        obs_v3 = self.observation_v3
        encoder = obs_v3.WorldTokenObservationEncoder(use_text=False)
        obs = {
            "phase": "map",
            "player": {
                "hp": 40,
                "max_hp": 80,
                "gold": 120,
                "deck_cards": [_card("card.deck.1", "Strike"), _card("card.deck.2", "Defend", card_type="Skill")],
                "relics": [],
                "potions": [],
            },
        }
        legal_actions = [
            {
                "action_id": "map:0:1",
                "kind": "map",
                "point_type": "Monster",
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
                "route_nodes": [
                    {
                        "point_type": "Monster",
                        "depth": 1,
                        "coord": {"row": 7, "col": 2},
                        "child_count": 2,
                        "is_leaf": False,
                    },
                    {
                        "point_type": "Shop",
                        "depth": 2,
                        "coord": {"row": 8, "col": 3},
                        "child_count": 1,
                        "is_leaf": False,
                    },
                ],
            }
        ]

        encoded = encoder.encode(obs, legal_actions=legal_actions)
        local_indices = np.flatnonzero(encoded["candidate_local_masks"][0] > 0.5)
        local_types = {
            obs_v3.TOKEN_TYPES[int(encoded["candidate_local_type_ids"][0, idx])]
            for idx in local_indices
        }

        self.assertIn("ROUTE_SUMMARY_TOKEN", local_types)
        self.assertIn("ROUTE_NODE", local_types)
        self.assertIn("ROUTE_RISK_LOCAL", local_types)
        self.assertIn("ROUTE_VALUE_LOCAL", local_types)
        self.assertEqual(
            int(encoded["candidate_query_role_ids"][0]),
            obs_v3.TOKEN_ROLE_TO_ID["QUERY_ROUTE"],
        )


if __name__ == "__main__":
    unittest.main()
