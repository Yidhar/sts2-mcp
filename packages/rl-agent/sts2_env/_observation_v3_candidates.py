"""Candidate-query and candidate-local token construction."""

from __future__ import annotations

from typing import Any

import numpy as np

from content_registry import build_live_enemy_semantic_text, build_live_potion_semantic_text

from . import observation_common as obs_common
from ._observation_v3_schema import (
    MAX_ACTIONS,
    MAX_CANDIDATE_LOCAL_TOKENS,
    MAX_ORDER_ID,
    OWNER_DECK,
    OWNER_DISCARD,
    OWNER_DRAW,
    OWNER_EXHAUST,
    OWNER_HAND,
    OWNER_NONE,
    OWNER_PLAY,
    OWNER_PLAYER,
    OWNER_POTION,
    OWNER_REWARD,
    OWNER_ROUTE,
    OWNER_SHOP,
    OWNER_UPGRADE,
    TOKEN_NUMERIC_DIM,
    TOKEN_TEXT_DIM,
    TOKEN_ZONE_TO_ID,
    _compress_numeric,
    _compress_text_embedding,
    _stable_bucket,
)
from .hand_mutation import (
    infer_hand_mutation,
    mutation_summary_numeric,
    mutation_target_numeric,
    post_hand_preview_numeric,
)
from .text_encoder import TEXT_DIM


class CandidateTokenMixin:
    def _append_candidate_tokens(
        self,
        candidate_entries: list[dict[str, Any]],
        candidate_local_entries: list[list[dict[str, Any]]],
        action_mask: np.ndarray,
        obs: dict[str, Any],
        features: dict[str, np.ndarray],
        legal_actions: list[dict[str, Any]],
    ) -> None:
        count = min(len(legal_actions), MAX_ACTIONS)
        combat = obs.get("combat") or {}
        for action_index in range(count):
            action = legal_actions[action_index]
            if not isinstance(action, dict):
                continue
            action_mask[action_index] = 1.0
            numeric = np.zeros(TOKEN_NUMERIC_DIM, dtype=np.float32)
            numeric[: obs_common.ACTION_FEAT_DIM] = features["actions"][action_index]
            numeric[obs_common.ACTION_FEAT_DIM : obs_common.ACTION_FEAT_DIM + 16] = _compress_numeric(features["semantic_actions"][action_index], 16)
            query_type = {
                "combat": "COMBAT_CANDIDATE",
                "build": "BUILD_CANDIDATE",
                "selection": "SELECTION_CANDIDATE",
                "route": "ROUTE_CANDIDATE",
            }.get(self._infer_action_domain(action, obs), "BUILD_CANDIDATE")
            target_enemy_index, target_enemy = self._match_target_enemy(action, combat)
            target_owner_id = self._enemy_owner_id(target_enemy_index)
            target_entity_id = _stable_bucket(self._enemy_entity_key(target_enemy, target_enemy_index)) if target_enemy_index is not None else 0
            action_owner_id, action_zone_id, action_order_id = self._action_source_binding(action, obs, action_index)
            text_embedding = _compress_numeric(
                np.concatenate(
                    [
                        _compress_text_embedding(features["action_text"][action_index]),
                        _compress_text_embedding(features["semantic_action_text"][action_index]),
                    ]
                ),
                TOKEN_TEXT_DIM,
            )
            candidate_entries.append(
                self._entry(
                    query_type,
                    numeric,
                    owner_id=action_owner_id,
                    entity_id=_stable_bucket(self._action_entity_key(action)),
                    zone_id=action_zone_id or self._query_zone_id(action, obs, query_type, target_enemy_index),
                    order_id=action_order_id,
                    target_owner_id=target_owner_id,
                    target_entity_id=target_entity_id,
                    text_embedding=text_embedding,
                )
            )
            candidate_local_entries[action_index] = self._build_candidate_local(action, obs, features, action_index, target_enemy_index, target_enemy)

    def _build_candidate_local(
        self,
        action: dict[str, Any],
        obs: dict[str, Any],
        features: dict[str, np.ndarray],
        action_index: int,
        target_enemy_index: int | None,
        target_enemy: dict[str, Any] | None,
    ) -> list[dict[str, Any]]:
        entries: list[dict[str, Any]] = []
        base_numeric = np.zeros(TOKEN_NUMERIC_DIM, dtype=np.float32)
        base_numeric[: obs_common.ACTION_FEAT_DIM] = features["actions"][action_index]

        kind = str(action.get("kind") or "").strip()
        action_domain = self._infer_action_domain(action, obs)
        action_card = action.get("card") if isinstance(action.get("card"), dict) else None
        action_potion = action.get("potion") if isinstance(action.get("potion"), dict) else None
        action_owner_id, action_zone_id, action_order_id = self._action_source_binding(action, obs, action_index)

        if action_card is not None:
            if action_domain == "selection":
                source_token_type = "SELECTION_POOL_CARD_LOCAL"
            else:
                source_token_type = "CARD_REWARD_LOCAL" if kind == "card_reward" else "SOURCE_CARD_LOCAL"
            entries.append(
                self._entry(
                    source_token_type,
                    base_numeric,
                    owner_id=action_owner_id,
                    zone_id=action_zone_id,
                    entity_id=_stable_bucket(action_card.get("id") or action_card.get("title")),
                    order_id=action_order_id,
                    text=self._build_live_card_text(action_card),
                )
            )
        if action_potion is not None:
            entries.append(
                self._entry(
                    "SOURCE_POTION_LOCAL",
                    base_numeric,
                    owner_id=OWNER_POTION,
                    entity_id=_stable_bucket(action_potion.get("id") or action_potion.get("title")),
                    order_id=self._source_order_id(action, obs, action_index),
                    text=build_live_potion_semantic_text(action_potion),
                )
            )
        item = action.get("item") if isinstance(action.get("item"), dict) else None
        if item is not None:
            entries.append(
                self._entry(
                    "SHOP_ITEM_LOCAL",
                    base_numeric,
                    owner_id=OWNER_SHOP,
                    entity_id=_stable_bucket(self._shop_item_entity_key(item)),
                    order_id=action_index + 1,
                    text=self._shop_item_text(item),
                )
            )
        if isinstance(action.get("upgrade_preview"), dict):
            preview = action.get("upgrade_preview") or {}
            entries.append(
                self._entry(
                    "PREVIEW_RESULT_LOCAL" if action_domain == "selection" else "UPGRADE_PREVIEW_LOCAL",
                    base_numeric,
                    owner_id=OWNER_UPGRADE if action_domain != "selection" else action_owner_id,
                    zone_id=TOKEN_ZONE_TO_ID["SELECTION"] if action_domain == "selection" else TOKEN_ZONE_TO_ID["UPGRADE"],
                    entity_id=_stable_bucket(preview.get("id") or preview.get("title")),
                    order_id=action_index + 1,
                    text=self._build_live_card_text(preview),
                )
            )
        if isinstance(action.get("reward"), dict):
            reward = action.get("reward") or {}
            entries.append(
                self._entry(
                    "REWARD_LOCAL",
                    base_numeric,
                    owner_id=OWNER_REWARD,
                    entity_id=_stable_bucket(self._reward_entity_key(reward)),
                    order_id=action_index + 1,
                    text=self._reward_text(reward),
                )
            )

        if target_enemy_index is not None and isinstance(target_enemy, dict):
            numeric = np.zeros(TOKEN_NUMERIC_DIM, dtype=np.float32)
            numeric[0] = obs_common._log_norm(obs_common._float(target_enemy.get("hp", target_enemy.get("current_hp"))), obs_common._LOG1P_1200)
            numeric[1] = obs_common._log_norm(obs_common._float(target_enemy.get("block")), obs_common._LOG1P_200)
            numeric[2] = obs_common._log_norm(obs_common._float((target_enemy.get("intent") or {}).get("total_damage")), obs_common._LOG1P_200)
            entries.append(
                self._entry(
                    "TARGET_LOCAL",
                    numeric,
                    owner_id=self._enemy_owner_id(target_enemy_index),
                    entity_id=_stable_bucket(self._enemy_entity_key(target_enemy, target_enemy_index)),
                    order_id=1,
                    text=build_live_enemy_semantic_text(target_enemy),
                )
            )
            target_context_entries: list[dict[str, Any]] = []
            self._append_target_enemy_local_context(target_context_entries, target_enemy_index, target_enemy)
            self._extend_with_budget(entries, target_context_entries, limit=3)
            source_for_target = action_card if action_card is not None else action_potion
            target_reaction_entries: list[dict[str, Any]] = []
            self._append_target_reaction_local(target_reaction_entries, target_enemy_index, target_enemy, source_for_target, obs)
            self._extend_with_budget(entries, target_reaction_entries, limit=1)

        if action_domain == "combat":
            self._append_combat_candidate_context(entries, obs, features, action, action_index, target_enemy_index, target_enemy)
        elif action_domain == "build":
            self._append_build_candidate_context(entries, obs, features, action, action_index)
        elif action_domain == "selection":
            self._append_selection_candidate_context(entries, obs, features, action, action_index)
        elif action_domain == "route":
            self._append_route_candidate_context(entries, obs, features, action, action_index)

        route_entries: list[dict[str, Any]] = []
        if features["route_summary"][action_index].any():
            numeric = np.zeros(TOKEN_NUMERIC_DIM, dtype=np.float32)
            numeric[: obs_common.ROUTE_SUMMARY_DIM] = features["route_summary"][action_index]
            route_entries.append(self._entry("ROUTE_SUMMARY_TOKEN", numeric, owner_id=OWNER_ROUTE, entity_id=0))
        for node_index in range(obs_common.MAX_ROUTE_NODES):
            if features["route_node_mask"][action_index, node_index] <= 0:
                continue
            numeric = np.zeros(TOKEN_NUMERIC_DIM, dtype=np.float32)
            numeric[: obs_common.ROUTE_NODE_FEAT_DIM] = features["route_nodes"][action_index, node_index]
            route_entries.append(
                self._entry(
                    "ROUTE_NODE",
                    numeric,
                    owner_id=OWNER_ROUTE,
                    entity_id=_stable_bucket(f"route:{action_index}:{node_index}"),
                    order_id=node_index + 1,
                )
            )
        self._extend_with_budget(entries, route_entries)
        return entries

    def _extend_with_budget(
        self,
        target: list[dict[str, Any]],
        additions: list[dict[str, Any]],
        *,
        limit: int | None = None,
        scorer=None,
    ) -> None:
        remaining = max(MAX_CANDIDATE_LOCAL_TOKENS - len(target), 0)
        if remaining <= 0 or not additions:
            return
        ordered = list(additions)
        if scorer is not None:
            ordered.sort(key=scorer, reverse=True)
        take = remaining if limit is None else min(remaining, max(limit, 0))
        if take <= 0:
            return
        target.extend(ordered[:take])

    def _append_combat_candidate_context(
        self,
        entries: list[dict[str, Any]],
        obs: dict[str, Any],
        features: dict[str, np.ndarray],
        action: dict[str, Any],
        action_index: int,
        target_enemy_index: int | None,
        target_enemy: dict[str, Any] | None,
    ) -> None:
        combat = obs.get("combat") or {}
        player = obs.get("player") or {}
        if not combat:
            return

        hp, _max_hp, hp_ratio = obs_common._player_hp_triplet(player)
        block = obs_common._float(player.get("block"))
        enemies = combat.get("enemies") or []
        incoming = sum(
            obs_common._float(((enemy or {}).get("intent") or {}).get("total_damage"))
            for enemy in enemies
            if isinstance(enemy, dict)
        )
        boss_player_state = self._boss_player_state()

        player_numeric = np.zeros(TOKEN_NUMERIC_DIM, dtype=np.float32)
        player_numeric[0] = hp_ratio
        player_numeric[1] = obs_common._log_norm(hp, obs_common._LOG1P_200)
        player_numeric[2] = obs_common._log_norm(block, obs_common._LOG1P_200)
        player_numeric[3] = obs_common._log_norm(incoming, obs_common._LOG1P_200)
        player_numeric[4] = obs_common._signed_log_norm(hp + block - incoming, obs_common._LOG1P_200)
        player_numeric[5 : 5 + min(obs_common.POWER_DIM, TOKEN_NUMERIC_DIM - 5)] = features["player_powers"][: TOKEN_NUMERIC_DIM - 5]
        player_numeric[29] = float(boss_player_state.get("facing_left", 0.0))
        player_numeric[30] = float(boss_player_state.get("facing_right", 0.0))
        player_numeric[31] = float(boss_player_state.get("sandpit_active", 0.0))
        player_numeric[32] = float(boss_player_state.get("sandpit_turns_norm", 0.0))
        player_numeric[33] = float(boss_player_state.get("ringing_active", 0.0))
        player_numeric[34] = float(boss_player_state.get("ringing_amount_norm", 0.0))
        player_numeric[35] = float(boss_player_state.get("chains_active", 0.0))
        player_numeric[36] = float(boss_player_state.get("bound_active", 0.0))
        player_numeric[37] = float(boss_player_state.get("hunger_active", 0.0))
        player_numeric[38] = float(boss_player_state.get("scrutiny_active", 0.0))
        player_numeric[39] = float(boss_player_state.get("grasp_active", 0.0))
        player_numeric[40] = float(boss_player_state.get("frantic_escape_hand_norm", 0.0))
        player_numeric[41] = float(boss_player_state.get("frantic_escape_draw_norm", 0.0))
        player_numeric[42] = float(boss_player_state.get("frantic_escape_discard_norm", 0.0))
        player_numeric[43] = float(boss_player_state.get("frantic_escape_total_norm", 0.0))
        player_numeric[44] = float(boss_player_state.get("escape_card_available", 0.0))
        player_numeric[45] = float(boss_player_state.get("play_budget_lock", 0.0))
        player_numeric[46] = float(boss_player_state.get("doormaker_lock_pressure", 0.0))
        player_numeric[47] = float(boss_player_state.get("back_attack_risk", 0.0))
        player_numeric[48] = float(boss_player_state.get("linked_support_alive", 0.0))
        player_numeric[49] = float(boss_player_state.get("countdown_active", 0.0))
        player_numeric[50] = float(boss_player_state.get("escape_card_tax", 0.0))
        entries.append(self._entry("PLAYER_STATE_LOCAL", player_numeric, owner_id=OWNER_PLAYER, entity_id=0))

        energy_numeric = np.zeros(TOKEN_NUMERIC_DIM, dtype=np.float32)
        source_profile = self._source_profile_for_obs(
            action.get("card") if isinstance(action.get("card"), dict) else action.get("potion") if isinstance(action.get("potion"), dict) else None,
            obs,
        )
        current_energy = obs_common._float(combat.get("energy"))
        max_energy = obs_common._float(combat.get("max_energy"))
        spend_cost = source_profile["cost"]
        energy_numeric[0] = min(current_energy / max(max_energy, 1.0), 1.0) if max_energy > 0 else 0.0
        energy_numeric[1] = min(current_energy / 10.0, 1.0)
        energy_numeric[2] = min(max_energy / 10.0, 1.0)
        energy_numeric[3] = min(spend_cost / 5.0, 1.0)
        energy_numeric[4] = float(source_profile["zero_cost"] > 0.5)
        energy_numeric[5] = float(source_profile["x_cost"] > 0.5)
        energy_numeric[6] = min(source_profile["energy"] / 5.0, 1.0)
        energy_numeric[7] = min(source_profile["draw"] / 5.0, 1.0)
        energy_numeric[8] = min(source_profile["hits"] / 10.0, 1.0)
        energy_numeric[9] = float(current_energy + source_profile["energy"] >= spend_cost)
        energy_numeric[10] = float(current_energy >= spend_cost)
        energy_numeric[11] = min(obs_common._float(combat.get("stars")) / 10.0, 1.0)
        energy_numeric[12] = features["actions"][action_index, 39] if features["actions"].shape[1] > 39 else 0.0
        energy_numeric[13] = features["actions"][action_index, 48] if features["actions"].shape[1] > 48 else 0.0
        energy_numeric[14] = float(boss_player_state.get("play_budget_lock", 0.0))
        energy_numeric[15] = float(boss_player_state.get("doormaker_lock_pressure", 0.0))
        energy_numeric[16] = float(boss_player_state.get("sandpit_active", 0.0))
        energy_numeric[17] = float(boss_player_state.get("sandpit_turns_norm", 0.0))
        energy_numeric[18] = float(boss_player_state.get("frantic_escape_hand_norm", 0.0))
        energy_numeric[19] = float(boss_player_state.get("frantic_escape_total_norm", 0.0))
        energy_numeric[20] = float(boss_player_state.get("escape_card_available", 0.0))
        energy_numeric[21] = float(boss_player_state.get("back_attack_risk", 0.0))
        energy_numeric[22] = float(boss_player_state.get("countdown_active", 0.0))
        energy_numeric[23] = float(boss_player_state.get("escape_card_tax", 0.0))
        energy_numeric[24] = float((boss_player_state.get("play_budget_lock", 0.0) > 0.0) and current_energy > 0.0)
        energy_numeric[25] = float((boss_player_state.get("sandpit_active", 0.0) > 0.0) and (source_profile["damage"] > 0.0))
        energy_numeric[26] = float((boss_player_state.get("escape_card_available", 0.0) > 0.0) and (source_profile["draw"] > 0.0 or source_profile["energy"] > 0.0))
        entries.append(self._entry("ENERGY_CONTEXT_LOCAL", energy_numeric, owner_id=OWNER_PLAYER, entity_id=0))
        energy_plan_entries: list[dict[str, Any]] = []
        self._append_energy_budget_local(energy_plan_entries, obs, features, action, source_profile)

        source_card = action.get("card") if isinstance(action.get("card"), dict) else None
        pile_summary_entries: list[dict[str, Any]] = []
        self._append_pile_context(
            pile_summary_entries,
            "DRAW_CONTEXT_LOCAL",
            OWNER_DRAW,
            self._runtime_cards(obs, "draw_pile", "draw_preview_cards"),
            source_card,
            "draw",
        )
        self._append_pile_context(
            pile_summary_entries,
            "DISCARD_CONTEXT_LOCAL",
            OWNER_DISCARD,
            self._runtime_cards(obs, "discard_pile", "discard_cards"),
            source_card,
            "discard",
        )
        self._append_pile_context(
            pile_summary_entries,
            "EXHAUST_CONTEXT_LOCAL",
            OWNER_EXHAUST,
            self._runtime_cards(obs, "exhaust_pile", "exhaust_cards"),
            source_card,
            "exhaust",
        )
        self._append_pile_context(
            pile_summary_entries,
            "PLAY_PILE_CONTEXT_LOCAL",
            OWNER_PLAY,
            self._runtime_cards(obs, "play_pile", "play_pile_cards"),
            source_card,
            "play",
        )


        hand_mutation_entries: list[dict[str, Any]] = []
        self._append_hand_mutation_locals(
            hand_mutation_entries,
            obs,
            source_card,
            current_energy=current_energy,
        )

        binding_entries: list[dict[str, Any]] = []
        self._append_source_pile_binding_locals(binding_entries, obs, source_card)

        cycle_entries: list[dict[str, Any]] = []
        self._append_cycle_plan_local(cycle_entries, obs, action, source_card)
        self._append_card_flow_counterfactual_local(cycle_entries, obs, action, source_card, source_profile, current_energy)
        self._append_energy_chain_local(cycle_entries, obs, action, source_profile, current_energy)

        peek_entries: list[dict[str, Any]] = []
        self._append_pile_peek_locals(
            peek_entries,
            "DRAW_PREVIEW_CARD",
            OWNER_DRAW,
            self._runtime_cards(obs, "draw_pile", "draw_preview_cards"),
            limit=1,
        )
        self._append_pile_peek_locals(
            peek_entries,
            "DISCARD_CARD",
            OWNER_DISCARD,
            self._runtime_cards(obs, "discard_pile", "discard_cards"),
            limit=1,
        )
        self._append_pile_peek_locals(
            peek_entries,
            "EXHAUST_CARD",
            OWNER_EXHAUST,
            self._runtime_cards(obs, "exhaust_pile", "exhaust_cards"),
            limit=1,
        )
        self._append_pile_peek_locals(
            peek_entries,
            "PLAY_PILE_CARD",
            OWNER_PLAY,
            self._runtime_cards(obs, "play_pile", "play_pile_cards"),
            limit=1,
        )

        support_entries: list[dict[str, Any]] = []
        self._append_relic_trigger_locals(support_entries, obs, action)
        self._append_potion_option_locals(support_entries, obs, action, features)

        support_graph_entries: list[dict[str, Any]] = []
        self._append_relic_potion_graph_local(
            support_graph_entries,
            obs,
            action,
            features,
            target_enemy_index,
            target_enemy,
        )

        self._extend_with_budget(entries, energy_plan_entries, limit=1)
        self._extend_with_budget(entries, hand_mutation_entries, limit=5)
        self._extend_with_budget(entries, cycle_entries, limit=1)
        self._extend_with_budget(entries, pile_summary_entries, limit=4)
        self._extend_with_budget(
            entries,
            binding_entries,
            limit=2,
            scorer=lambda entry: float(entry["numeric"][19]) * 10.0 - float(entry["numeric"][3]),
        )
        self._extend_with_budget(
            entries,
            peek_entries,
            limit=2,
            scorer=lambda entry: 2.0 - float(entry["order_id"]),
        )
        self._extend_with_budget(
            entries,
            support_entries,
            limit=2,
            scorer=lambda entry: float(entry["numeric"][0]),
        )
        self._extend_with_budget(entries, support_graph_entries, limit=1)


    def _append_card_flow_counterfactual_local(
        self,
        entries: list[dict[str, Any]],
        obs: dict[str, Any],
        action: dict[str, Any],
        source_card: dict[str, Any] | None,
        source_profile: dict[str, float],
        current_energy: float,
    ) -> None:
        hand_cards = self._runtime_cards(obs, "hand", "hand_cards")
        draw_cards = self._runtime_cards(obs, "draw_pile", "draw_preview_cards")
        discard_cards = self._runtime_cards(obs, "discard_pile", "discard_cards")
        exhaust_cards = self._runtime_cards(obs, "exhaust_pile", "exhaust_cards")
        if str(action.get("action_id") or "").lower() == "end_turn" or str((action.get("semantic") or {}).get("family") or "").lower() == "end_turn":
            numeric = np.zeros(TOKEN_NUMERIC_DIM, dtype=np.float32)
            ethereal_count = 0.0
            retain_count = 0.0
            exhaust_count = 0.0
            total_cost = 0.0
            playable_cost = 0.0
            for card in hand_cards:
                profile = self._source_profile(card)
                text = self._source_text(card)
                ethereal_count += profile["ethereal"]
                retain_count += profile["retain"]
                exhaust_count += profile["exhaust"]
                total_cost += profile["cost"]
                playable_cost += float(profile["cost"] <= current_energy or profile["zero_cost"] > 0.5)
            numeric[0] = min(len(hand_cards) / 10.0, 1.0)
            numeric[1] = min(draw_cards.__len__() / 30.0, 1.0)
            numeric[2] = min(discard_cards.__len__() / 30.0, 1.0)
            numeric[3] = min(exhaust_cards.__len__() / 20.0, 1.0)
            numeric[4] = min(ethereal_count / 5.0, 1.0)
            numeric[5] = min(retain_count / 5.0, 1.0)
            numeric[6] = min(exhaust_count / 5.0, 1.0)
            numeric[7] = min(playable_cost / 10.0, 1.0)
            numeric[8] = obs_common._log_norm(total_cost, obs_common._LOG1P_100)
            numeric[9] = float(len(draw_cards) <= 3 and len(discard_cards) > 0)
            numeric[10] = float(len(hand_cards) > 0 and current_energy > 0.0)
            entries.append(self._entry("END_TURN_HAND_FLOW_LOCAL", numeric, owner_id=OWNER_PLAYER, entity_id=0, text="end_turn hand flow: unplayed non-ethereal cards move toward discard; retain stays; ethereal/exhaust leaves loop"))
            return
        if not isinstance(source_card, dict):
            return
        numeric = np.zeros(TOKEN_NUMERIC_DIM, dtype=np.float32)
        text = self._source_text(source_card)
        exhausts = source_profile["exhaust"] > 0.5 or "exhaust" in text
        ethereal = source_profile["ethereal"] > 0.5 or "ethereal" in text
        retain = source_profile["retain"] > 0.5 or "retain" in text
        power = source_profile["power"] > 0.5
        cost = source_profile["cost"]
        energy_after = current_energy - cost + source_profile["energy"] - source_profile.get("energy_loss", 0.0)
        numeric[0] = float(exhausts)
        numeric[1] = float(power)
        numeric[2] = float(not exhausts and not power)
        numeric[3] = float(retain)
        numeric[4] = float(ethereal)
        numeric[5] = min(len(draw_cards) / 30.0, 1.0)
        numeric[6] = min(len(discard_cards) / 30.0, 1.0)
        numeric[7] = min(len(exhaust_cards) / 20.0, 1.0)
        numeric[8] = float(len(draw_cards) <= 3 and len(discard_cards) > 0)
        numeric[9] = float(exhausts and not ethereal)
        numeric[10] = float(exhausts and source_profile["draw"] <= 0.0 and source_profile["energy"] <= 0.0)
        numeric[11] = obs_common._signed_log_norm(energy_after, obs_common._LOG1P_100)
        numeric[12] = min(source_profile["damage"] / 80.0, 1.0)
        numeric[13] = min(source_profile["block"] / 80.0, 1.0)
        numeric[14] = min(source_profile["draw"] / 5.0, 1.0)
        numeric[15] = min(source_profile["energy"] / 5.0, 1.0)
        numeric[16] = min(source_profile.get("energy_loss", 0.0) / 5.0, 1.0)
        numeric[17] = float(source_profile.get("strategic_skip_value", 0.0) > 0.5)
        entries.append(self._entry("CARD_FLOW_COUNTERFACTUAL_LOCAL", numeric, owner_id=OWNER_HAND, entity_id=_stable_bucket((source_card.get("id") if isinstance(source_card, dict) else None) or (source_card.get("title") if isinstance(source_card, dict) else None) or "flow"), text="play_now destination vs skip/end_turn loop counterfactual"))

    def _append_energy_chain_local(
        self,
        entries: list[dict[str, Any]],
        obs: dict[str, Any],
        action: dict[str, Any],
        source_profile: dict[str, float],
        current_energy: float,
    ) -> None:
        if str(action.get("action_id") or "").lower() == "end_turn":
            return
        hand_cards = self._runtime_cards(obs, "hand", "hand_cards")
        cost = source_profile["cost"]
        energy_gain = source_profile["energy"]
        energy_loss = source_profile.get("energy_loss", 0.0)
        energy_after = max(0.0, current_energy - cost + energy_gain - energy_loss)
        followup_count = 0.0
        followup_damage = 0.0
        followup_block = 0.0
        for card in hand_cards:
            profile = self._source_profile(card)
            if profile["cost"] <= energy_after + 1e-6 and card is not action.get("card"):
                followup_count += 1.0
                followup_damage = max(followup_damage, profile["damage"])
                followup_block = max(followup_block, profile["block"])
        numeric = np.zeros(TOKEN_NUMERIC_DIM, dtype=np.float32)
        numeric[0] = min(current_energy / 10.0, 1.0)
        numeric[1] = min(cost / 5.0, 1.0)
        numeric[2] = min(energy_gain / 5.0, 1.0)
        numeric[10] = min(energy_loss / 5.0, 1.0)
        numeric[3] = obs_common._signed_log_norm(energy_after, obs_common._LOG1P_100)
        numeric[4] = min(followup_count / 10.0, 1.0)
        numeric[5] = min(followup_damage / 80.0, 1.0)
        numeric[6] = min(followup_block / 80.0, 1.0)
        numeric[7] = float(source_profile["x_cost"] > 0.5 and current_energy <= 0.0)
        numeric[8] = float(energy_gain > 0.0 and followup_count <= 0.0)
        numeric[9] = min(source_profile["hp_loss"] / 20.0, 1.0)
        numeric[11] = float(source_profile.get("strategic_skip_value", 0.0) > 0.5)
        entries.append(self._entry("ENERGY_CHAIN_LOCAL", numeric, owner_id=OWNER_PLAYER, entity_id=0, text="energy spend/gain follow-up chain and zero-energy x-cost affordance"))

    def _append_hand_mutation_locals(
        self,
        entries: list[dict[str, Any]],
        obs: dict[str, Any],
        source_card: dict[str, Any] | None,
        *,
        current_energy: float = 0.0,
    ) -> None:
        """Expose generic action->hand mutation affordances as local tokens.

        This is intentionally not card-name-specific: Armaments, cost reducers,
        hand exhaust/discard/transform/copy/retain effects, draw/add/return-to-
        hand effects all flow through the same summary/target/post-preview
        structure.  The policy can cross-attend source action <-> current hand
        cards <-> post-hand preview without requiring MCTS to discover the
        intermediate hand state.
        """
        if not isinstance(source_card, dict):
            return
        hand_cards = self._runtime_cards(obs, "hand", "hand_cards")
        plan = infer_hand_mutation(
            source_card,
            hand_cards,
            self._runtime_cards(obs, "draw_pile", "draw_preview_cards"),
            self._runtime_cards(obs, "discard_pile", "discard_cards"),
            self._runtime_cards(obs, "exhaust_pile", "exhaust_cards"),
            current_energy=current_energy,
        )
        if not plan.will_mutate_hand:
            return

        source_key = source_card.get("id") or source_card.get("title") or "hand-mutation"
        summary = np.asarray(mutation_summary_numeric(plan), dtype=np.float32)
        entries.append(
            self._entry(
                "HAND_MUTATION_LOCAL",
                summary,
                owner_id=OWNER_HAND,
                zone_id=TOKEN_ZONE_TO_ID["HAND"],
                entity_id=_stable_bucket(f"hand-mut:{source_key}"),
                order_id=1,
                text=f"hand mutation | {self._build_live_card_text(source_card)}",
            )
        )

        post = np.asarray(post_hand_preview_numeric(plan), dtype=np.float32)
        entries.append(
            self._entry(
                "POST_HAND_PREVIEW_LOCAL",
                post,
                owner_id=OWNER_HAND,
                zone_id=TOKEN_ZONE_TO_ID["HAND"],
                entity_id=_stable_bucket(f"post-hand:{source_key}"),
                order_id=2,
                text=f"post hand preview | {self._build_live_card_text(source_card)}",
            )
        )

        for rank, target in enumerate(plan.targets[:6], start=1):
            if not target.affected and rank > 2:
                continue
            row = np.asarray(mutation_target_numeric(target), dtype=np.float32)
            card = target.card
            entries.append(
                self._entry(
                    "HAND_MUTATION_TARGET_LOCAL",
                    row,
                    owner_id=OWNER_HAND,
                    zone_id=TOKEN_ZONE_TO_ID["HAND"],
                    entity_id=_stable_bucket(card.get("id") or card.get("title") or f"hand-target:{target.index}"),
                    order_id=min(target.index + 1, MAX_ORDER_ID),
                    text=f"hand mutation target | {self._build_live_card_text(card)}",
                )
            )

    def _append_selection_candidate_context(
        self,
        entries: list[dict[str, Any]],
        obs: dict[str, Any],
        features: dict[str, np.ndarray],
        action: dict[str, Any],
        action_index: int,
    ) -> None:
        action_owner_id, _action_zone_id, action_order_id = self._action_source_binding(action, obs, action_index)
        semantics_text = self._selection_semantics_text(action)
        in_combat = bool(obs.get("combat"))

        operator_numeric = np.zeros(TOKEN_NUMERIC_DIM, dtype=np.float32)
        selection = str(action.get("selection") or "").strip().lower()
        semantics_lower = semantics_text.lower()
        operator_numeric[0] = float(selection in {"confirm", "confirm_selection"})
        operator_numeric[1] = float(selection in {"cancel", "close"})
        operator_numeric[2] = float(selection == "skip")
        operator_numeric[3] = float(any(token in semantics_lower for token in ("upgrade", "smith")))
        operator_numeric[4] = float(any(token in semantics_lower for token in ("transform", "mutate", "change")))
        operator_numeric[5] = float(any(token in semantics_lower for token in ("remove", "purge")))
        operator_numeric[6] = float(any(token in semantics_lower for token in ("exhaust", "consume")))
        operator_numeric[7] = float("discard" in semantics_lower)
        operator_numeric[8] = float(any(token in semantics_lower for token in ("discover", "draft", "reward", "choose", "pick")))
        operator_numeric[9] = float(in_combat)
        operator_numeric[10] = float(not in_combat)
        operator_numeric[11] = min(max(float(action_order_id), 0.0) / max(MAX_ORDER_ID, 1), 1.0)
        operator_numeric[12] = features["actions"][action_index, 12] if features["actions"].shape[1] > 12 else 0.0
        operator_numeric[13] = features["actions"][action_index, 13] if features["actions"].shape[1] > 13 else 0.0
        # §4C: surface "this card is already selected in the current pick set".
        # Bridge emits is_selected on each card_selection:select action so the
        # model can directly see pick→deselect oscillations on multi-pick burn
        # cards (POTION.GLOWWATER, 净化, etc.) instead of inferring from history.
        operator_numeric[14] = float(bool(action.get("is_selected"))) if isinstance(action.get("is_selected"), bool | int) else 0.0
        entries.append(
            self._entry(
                "SELECTION_OPERATOR_LOCAL",
                operator_numeric,
                owner_id=action_owner_id,
                zone_id=TOKEN_ZONE_TO_ID["SELECTION"],
                entity_id=_stable_bucket(f"selection-op:{self._action_entity_key(action)}"),
                order_id=1,
                text=selection or semantics_text or "selection operator",
            )
        )

        semantics_numeric = np.zeros(TOKEN_NUMERIC_DIM, dtype=np.float32)
        semantics_numeric[0] = float("hand" in semantics_lower)
        semantics_numeric[1] = float("draw" in semantics_lower or "draw pile" in semantics_lower)
        semantics_numeric[2] = float("discard" in semantics_lower)
        semantics_numeric[3] = float("exhaust" in semantics_lower)
        semantics_numeric[4] = float("deck" in semantics_lower)
        semantics_numeric[5] = float("play pile" in semantics_lower or "played" in semantics_lower)
        semantics_numeric[6] = float(any(token in semantics_lower for token in ("upgrade", "smith")))
        semantics_numeric[7] = float(any(token in semantics_lower for token in ("transform", "mutate")))
        semantics_numeric[8] = float(any(token in semantics_lower for token in ("remove", "purge")))
        semantics_numeric[9] = float(any(token in semantics_lower for token in ("retain", "keep")))
        semantics_numeric[10] = float(any(token in semantics_lower for token in ("bottle", "duplicate", "copy")))
        semantics_numeric[11] = float("reward" in semantics_lower or "discover" in semantics_lower)
        entries.append(
            self._entry(
                "SELECTION_SEMANTICS_LOCAL",
                semantics_numeric,
                owner_id=OWNER_NONE,
                zone_id=TOKEN_ZONE_TO_ID["SELECTION"],
                entity_id=_stable_bucket(f"selection-semantics:{semantics_text or action.get('action_id') or action_index}"),
                order_id=2,
                text=semantics_text,
            )
        )

        selected_card = self._resolve_build_candidate_card(action)
        if selected_card is not None:
            synergy_numeric = self._deck_synergy_numeric((obs.get("player") or {}).get("deck_cards") or [], selected_card)
            entries.append(
                self._entry(
                    "DECK_SYNERGY_LOCAL",
                    synergy_numeric,
                    owner_id=OWNER_DECK,
                    entity_id=_stable_bucket(selected_card.get("id") or selected_card.get("title") or "selection-deck-synergy"),
                    order_id=3,
                    text=self._build_live_card_text(selected_card),
                )
            )

        if in_combat:
            source_card = action.get("card") if isinstance(action.get("card"), dict) else None
            pile_summary_entries: list[dict[str, Any]] = []
            self._append_pile_context(
                pile_summary_entries,
                "DRAW_CONTEXT_LOCAL",
                OWNER_DRAW,
                self._runtime_cards(obs, "draw_pile", "draw_preview_cards"),
                source_card,
                "draw",
            )
            self._append_pile_context(
                pile_summary_entries,
                "DISCARD_CONTEXT_LOCAL",
                OWNER_DISCARD,
                self._runtime_cards(obs, "discard_pile", "discard_cards"),
                source_card,
                "discard",
            )
            self._append_pile_context(
                pile_summary_entries,
                "EXHAUST_CONTEXT_LOCAL",
                OWNER_EXHAUST,
                self._runtime_cards(obs, "exhaust_pile", "exhaust_cards"),
                source_card,
                "exhaust",
            )
            self._append_pile_context(
                pile_summary_entries,
                "PLAY_PILE_CONTEXT_LOCAL",
                OWNER_PLAY,
                self._runtime_cards(obs, "play_pile", "play_pile_cards"),
                source_card,
                "play",
            )
            self._extend_with_budget(entries, pile_summary_entries, limit=4)

            support_entries: list[dict[str, Any]] = []
            self._append_relic_trigger_locals(support_entries, obs, action)
            self._append_potion_option_locals(support_entries, obs, action, features)
            self._extend_with_budget(entries, support_entries, limit=2, scorer=lambda entry: float(entry["numeric"][0]))
        else:
            self._append_build_candidate_context(entries, obs, features, action, action_index)

    def _append_build_candidate_context(
        self,
        entries: list[dict[str, Any]],
        obs: dict[str, Any],
        features: dict[str, np.ndarray],
        action: dict[str, Any],
        action_index: int,
    ) -> None:
        player = obs.get("player") or {}
        deck_cards = player.get("deck_cards") or []
        gold = obs_common._float(player.get("gold"))

        build_numeric = np.zeros(TOKEN_NUMERIC_DIM, dtype=np.float32)
        run_memory = features["run_memory"]
        objective = features["objective_context"]
        build_numeric[: min(24, run_memory.shape[0])] = run_memory[:24]
        build_numeric[24 : 24 + min(24, objective.shape[0])] = objective[:24]
        build_numeric[48] = min(len(deck_cards) / 50.0, 1.0) if isinstance(deck_cards, list) else 0.0
        build_numeric[49] = obs_common._log_norm(gold, obs_common._LOG1P_500)
        build_numeric[50] = min(len(player.get("relics") or []) / max(obs_common.MAX_RELICS, 1), 1.0)
        build_numeric[51] = min(len(player.get("potions") or []) / max(obs_common.MAX_POTIONS, 1), 1.0)
        build_numeric[52] = features["actions"][action_index, 12] if features["actions"].shape[1] > 12 else 0.0
        build_numeric[53] = features["actions"][action_index, 13] if features["actions"].shape[1] > 13 else 0.0
        build_numeric[54] = features["actions"][action_index, 14] if features["actions"].shape[1] > 14 else 0.0
        build_numeric[55] = features["actions"][action_index, 18] if features["actions"].shape[1] > 18 else 0.0

        # Event-option structured effect deltas (from bridge regex). These slots
        # replace the text encoder as the primary signal for event choice
        # outcomes; slots stay zero for non-event-option actions.
        if str(action.get("kind") or "") == "event_option":
            option = action.get("option") if isinstance(action.get("option"), dict) else None
            deltas = option.get("effect_deltas") if isinstance(option, dict) else None
            if isinstance(deltas, dict):
                event_hp, event_max_hp, _event_hp_ratio = obs_common._player_hp_triplet(obs.get("player") if isinstance(obs.get("player"), dict) else {})
                player_max_hp = max(event_max_hp if event_max_hp > 1.0 else 0.0, event_hp if event_hp > 1.0 else 0.0, 1.0)
                hp_delta = obs_common._float(deltas.get("hp_delta"))
                max_hp_delta = obs_common._float(deltas.get("max_hp_delta"))
                gold_delta = obs_common._float(deltas.get("gold_delta"))
                # Signed normalized 鈥?positive = gain, negative = loss.
                build_numeric[56] = max(-1.0, min(hp_delta / player_max_hp, 1.0))
                build_numeric[57] = obs_common._signed_log_norm(max_hp_delta, obs_common._LOG1P_100)
                build_numeric[58] = obs_common._signed_log_norm(gold_delta, obs_common._LOG1P_500)
                build_numeric[59] = 1.0 if deltas.get("heal_full") else 0.0
                build_numeric[60] = min(obs_common._float(deltas.get("card_add_count")) / 3.0, 1.0)
                build_numeric[61] = 1.0 if deltas.get("card_add_attack") else 0.0
                build_numeric[62] = 1.0 if deltas.get("card_add_skill") else 0.0
                build_numeric[63] = 1.0 if deltas.get("card_add_power") else 0.0
                build_numeric[64] = 1.0 if deltas.get("card_add_curse") else 0.0
                build_numeric[65] = 1.0 if deltas.get("card_add_status") else 0.0
                build_numeric[66] = min(obs_common._float(deltas.get("card_remove_count")) / 3.0, 1.0)
                build_numeric[67] = min(obs_common._float(deltas.get("card_transform_count")) / 3.0, 1.0)
                build_numeric[68] = min(obs_common._float(deltas.get("card_upgrade_count")) / 3.0, 1.0)
                build_numeric[69] = min(obs_common._float(deltas.get("card_duplicate_count")) / 3.0, 1.0)
                build_numeric[70] = 1.0 if deltas.get("relic_gain") else 0.0
                build_numeric[71] = 1.0 if deltas.get("potion_gain") else 0.0
                build_numeric[72] = 1.0 if deltas.get("enter_combat") else 0.0
                # Aggregate cost / benefit magnitudes as quick-lookup summaries.
                total_cost_magnitude = max(0.0, -hp_delta) / player_max_hp + max(0.0, -gold_delta) / 500.0
                total_benefit_magnitude = (
                    max(0.0, hp_delta) / player_max_hp
                    + max(0.0, gold_delta) / 500.0
                    + (1.0 if deltas.get("relic_gain") else 0.0)
                    + (1.0 if deltas.get("potion_gain") else 0.0)
                )
                build_numeric[73] = min(total_cost_magnitude, 1.0)
                build_numeric[74] = min(total_benefit_magnitude, 1.0)

        entries.append(
            self._entry(
                "BUILD_STATE_LOCAL",
                build_numeric,
                owner_id=OWNER_PLAYER,
                entity_id=0,
                text=f"build state | deck {len(deck_cards) if isinstance(deck_cards, list) else 0} | gold {int(gold)}",
            )
        )

        candidate_card = self._resolve_build_candidate_card(action)
        if candidate_card is not None:
            synergy_numeric = self._deck_synergy_numeric(deck_cards if isinstance(deck_cards, list) else [], candidate_card)
            entries.append(
                self._entry(
                    "DECK_SYNERGY_LOCAL",
                    synergy_numeric,
                    owner_id=OWNER_DECK,
                    entity_id=_stable_bucket(candidate_card.get("id") or candidate_card.get("title") or "deck-synergy"),
                    text=self._build_live_card_text(candidate_card),
                )
            )

        if str(action.get("kind") or "") == "shop":
            shop_numeric = np.zeros(TOKEN_NUMERIC_DIM, dtype=np.float32)
            item = action.get("item") if isinstance(action.get("item"), dict) else {}
            cost = obs_common._float(item.get("cost"))
            shop_action = str(action.get("shop_action") or "").strip().lower()
            item_kind = str(item.get("item_kind") or item.get("kind") or "").strip().lower()
            item_title = str(item.get("title") or "").strip().lower()
            is_open = shop_action == "open"
            is_leave = shop_action in {"leave", "back"} or any(token in shop_action for token in ("leave", "back"))
            is_buy = shop_action == "buy" or "buy" in shop_action or "purchase" in shop_action
            is_remove = (
                "remove" in shop_action
                or item_kind in {"card_removal", "remove", "removal", "purge"}
                or "remove" in item_title
                or "purge" in item_title
            )
            is_affordable = item.get("is_affordable")
            if is_affordable is None:
                is_affordable = item.get("affordable")
            if is_affordable is None:
                is_affordable = item.get("enough_gold")
            if isinstance(is_affordable, str):
                is_affordable = is_affordable.strip().lower() in {"1", "true", "yes", "y"}
            else:
                is_affordable = bool(is_affordable)
            shop_numeric[0] = obs_common._log_norm(gold, obs_common._LOG1P_500)
            shop_numeric[1] = obs_common._log_norm(cost, obs_common._LOG1P_500)
            shop_numeric[2] = float(is_affordable or (gold >= cost and cost > 0))
            shop_numeric[3] = min(cost / max(gold, 1.0), 1.0) if gold > 0 else float(cost > 0)
            shop_numeric[4] = min(len(deck_cards) / 50.0, 1.0) if isinstance(deck_cards, list) else 0.0
            shop_numeric[5] = float(isinstance(item.get("card"), dict))
            shop_numeric[6] = float(isinstance(item.get("relic"), dict))
            shop_numeric[7] = float(isinstance(item.get("potion"), dict))
            # Keep slot 8 as the long-standing "remove" bit, but also expose
            # explicit shop action/item-kind bits below.  Bridge card removal
            # arrives as shop_action=buy + item_kind=card_removal, not as a
            # shop_action containing the word "remove".
            shop_numeric[8] = float(is_remove)
            shop_numeric[9] = float(is_open)
            shop_numeric[10] = float(is_leave)
            shop_numeric[11] = float(is_buy)
            shop_numeric[12] = float(is_remove)
            shop_numeric[13] = float(item_kind == "card")
            shop_numeric[14] = float(item_kind == "relic")
            shop_numeric[15] = float(item_kind == "potion")
            shop_numeric[16] = float(item_kind == "card_removal")
            shop_numeric[17] = float(bool(item.get("used")))
            entries.append(
                self._entry(
                    "SHOP_ECON_LOCAL",
                    shop_numeric,
                    owner_id=OWNER_SHOP,
                    entity_id=_stable_bucket(self._shop_item_entity_key(item)),
                    text=self._shop_item_text(item),
                )
            )

    def _append_route_candidate_context(
        self,
        entries: list[dict[str, Any]],
        obs: dict[str, Any],
        features: dict[str, np.ndarray],
        action: dict[str, Any],
        action_index: int,
    ) -> None:
        player = obs.get("player") or {}
        route_summary = features["route_summary"][action_index]
        if route_summary.any():
            risk_numeric = np.zeros(TOKEN_NUMERIC_DIM, dtype=np.float32)
            value_numeric = np.zeros(TOKEN_NUMERIC_DIM, dtype=np.float32)

            risk_numeric[: min(obs_common.ROUTE_SUMMARY_DIM, TOKEN_NUMERIC_DIM)] = route_summary[:TOKEN_NUMERIC_DIM]
            risk_numeric[20] = route_summary[5] if route_summary.shape[0] > 5 else 0.0
            risk_numeric[21] = route_summary[6] if route_summary.shape[0] > 6 else 0.0
            risk_numeric[22] = route_summary[12] if route_summary.shape[0] > 12 else 0.0
            risk_numeric[23] = route_summary[18] if route_summary.shape[0] > 18 else 0.0
            risk_numeric[24] = obs_common._player_hp_triplet(player)[2] if isinstance(player, dict) else 0.0
            risk_numeric[25] = obs_common._log_norm(obs_common._float((player or {}).get("gold")), obs_common._LOG1P_500)

            value_numeric[: min(obs_common.ROUTE_SUMMARY_DIM, TOKEN_NUMERIC_DIM)] = route_summary[:TOKEN_NUMERIC_DIM]
            value_numeric[20] = route_summary[9] if route_summary.shape[0] > 9 else 0.0
            value_numeric[21] = route_summary[10] if route_summary.shape[0] > 10 else 0.0
            value_numeric[22] = route_summary[11] if route_summary.shape[0] > 11 else 0.0
            value_numeric[23] = route_summary[7] if route_summary.shape[0] > 7 else 0.0
            value_numeric[24] = route_summary[14] if route_summary.shape[0] > 14 else 0.0
            value_numeric[25] = route_summary[15] if route_summary.shape[0] > 15 else 0.0

            entries.append(
                self._entry(
                    "ROUTE_RISK_LOCAL",
                    risk_numeric,
                    owner_id=OWNER_ROUTE,
                    entity_id=_stable_bucket(f"route-risk:{action_index}"),
                    text=f"route risk | action {action_index}",
                )
            )
            entries.append(
                self._entry(
                    "ROUTE_VALUE_LOCAL",
                    value_numeric,
                    owner_id=OWNER_ROUTE,
                    entity_id=_stable_bucket(f"route-value:{action_index}"),
                    text=f"route value | action {action_index}",
                )
            )

    @staticmethod
    def _resolve_build_candidate_card(action: dict[str, Any]) -> dict[str, Any] | None:
        for candidate in (
            action.get("card"),
            action.get("upgrade_preview"),
            (action.get("item") or {}).get("card") if isinstance(action.get("item"), dict) else None,
            (action.get("reward") or {}).get("card") if isinstance(action.get("reward"), dict) else None,
        ):
            if isinstance(candidate, dict):
                return candidate
        return None

    def _deck_synergy_numeric(self, deck_cards: list[Any], candidate_card: dict[str, Any]) -> np.ndarray:
        numeric = np.zeros(TOKEN_NUMERIC_DIM, dtype=np.float32)
        deck_count = len(deck_cards) if isinstance(deck_cards, list) else 0
        if deck_count <= 0:
            source_profile = self._source_profile(candidate_card)
            numeric[0] = min(source_profile["damage"] / 40.0, 1.0)
            numeric[1] = min(source_profile["block"] / 40.0, 1.0)
            numeric[2] = min(source_profile["draw"] / 5.0, 1.0)
            numeric[3] = min(source_profile["energy"] / 5.0, 1.0)
            return numeric

        candidate_profile = self._source_profile(candidate_card)
        candidate_id = str(candidate_card.get("id") or "").strip()
        candidate_title = str(candidate_card.get("title") or "").strip().lower()
        same_count = 0.0
        attack = skill = power = 0.0
        total_cost = total_damage = total_block = total_draw = total_energy = total_exhaust = 0.0

        for card in deck_cards:
            if not isinstance(card, dict):
                continue
            profile = self._source_profile(card)
            attack += profile["attack"]
            skill += profile["skill"]
            power += profile["power"]
            total_cost += profile["cost"]
            total_damage += profile["damage"]
            total_block += profile["block"]
            total_draw += profile["draw"]
            total_energy += profile["energy"]
            total_exhaust += profile["exhaust"]
            deck_id = str(card.get("id") or "").strip()
            deck_title = str(card.get("title") or "").strip().lower()
            if (candidate_id and deck_id and candidate_id == deck_id) or (candidate_title and deck_title and candidate_title == deck_title):
                same_count += 1.0

        denom = max(float(deck_count), 1.0)
        avg_damage = total_damage / denom
        avg_block = total_block / denom
        avg_draw = total_draw / denom
        avg_energy = total_energy / denom
        avg_cost = total_cost / denom
        avg_exhaust = total_exhaust / denom

        numeric[0] = min(deck_count / 50.0, 1.0)
        numeric[1] = attack / denom
        numeric[2] = skill / denom
        numeric[3] = power / denom
        numeric[4] = min(avg_cost / 5.0, 1.0)
        numeric[5] = obs_common._log_norm(avg_damage, obs_common._LOG1P_100)
        numeric[6] = obs_common._log_norm(avg_block, obs_common._LOG1P_100)
        numeric[7] = min(avg_draw / 5.0, 1.0)
        numeric[8] = min(avg_energy / 5.0, 1.0)
        numeric[9] = min(avg_exhaust, 1.0)
        numeric[10] = min(same_count / 4.0, 1.0)
        numeric[11] = float(candidate_profile["attack"] > 0.0 and avg_damage < 10.0)
        numeric[12] = float(candidate_profile["block"] > 0.0 and avg_block < 8.0)
        numeric[13] = float(candidate_profile["draw"] > 0.0 and avg_draw < 1.0)
        numeric[14] = float(candidate_profile["energy"] > 0.0 and avg_energy < 0.5)
        numeric[15] = float(candidate_profile["power"] > 0.0 and power / denom < 0.15)
        numeric[16] = float(candidate_profile["exhaust"] > 0.0 and avg_exhaust < 0.2)
        numeric[17] = candidate_profile["zero_cost"]
        numeric[18] = candidate_profile["x_cost"]
        numeric[19] = obs_common._log_norm(candidate_profile["damage"], obs_common._LOG1P_100)
        numeric[20] = obs_common._log_norm(candidate_profile["block"], obs_common._LOG1P_100)
        numeric[21] = min(candidate_profile["draw"] / 5.0, 1.0)
        numeric[22] = min(candidate_profile["energy"] / 5.0, 1.0)
        numeric[23] = float(candidate_profile["retain"] > 0.0)
        return numeric

    def _append_pile_context(
        self,
        entries: list[dict[str, Any]],
        token_type: str,
        owner_id: int,
        cards: list[Any],
        source_card: dict[str, Any] | None,
        pile_label: str,
    ) -> None:
        numeric = self._pile_context_numeric(cards, source_card)
        top_titles = ", ".join(
            str(card.get("title") or card.get("id") or "").strip()
            for card in cards[:2]
            if isinstance(card, dict) and str(card.get("title") or card.get("id") or "").strip()
        )
        text = f"{pile_label} pile | {len(cards)} cards"
        if top_titles:
            text = f"{text} | top {top_titles}"
        entries.append(self._entry(token_type, numeric, owner_id=owner_id, entity_id=_stable_bucket(f"{pile_label}:context"), text=text))

    def _append_source_pile_binding_locals(
        self,
        entries: list[dict[str, Any]],
        obs: dict[str, Any],
        source_card: dict[str, Any] | None,
    ) -> None:
        if not isinstance(source_card, dict):
            return
        source_entity_id = _stable_bucket(source_card.get("id") or source_card.get("title") or "source-card")
        for token_type, owner_id, pile_key, fallback_key, pile_label in (
            ("DRAW_BINDING_LOCAL", OWNER_DRAW, "draw_pile", "draw_preview_cards", "draw"),
            ("DISCARD_BINDING_LOCAL", OWNER_DISCARD, "discard_pile", "discard_cards", "discard"),
            ("EXHAUST_BINDING_LOCAL", OWNER_EXHAUST, "exhaust_pile", "exhaust_cards", "exhaust"),
            ("PLAY_BINDING_LOCAL", OWNER_PLAY, "play_pile", "play_pile_cards", "play"),
        ):
            cards = self._runtime_cards(obs, pile_key, fallback_key)
            numeric, closest_position, same_count = self._pile_binding_numeric(cards, source_card)
            entries.append(
                self._entry(
                    token_type,
                    numeric,
                    owner_id=owner_id,
                    entity_id=source_entity_id,
                    order_id=closest_position,
                    text=f"{pile_label} bind | same {same_count:.0f} | closest {closest_position}",
                )
            )

    def _pile_binding_numeric(
        self,
        cards: list[Any],
        source_card: dict[str, Any],
    ) -> tuple[np.ndarray, int, float]:
        numeric = np.zeros(TOKEN_NUMERIC_DIM, dtype=np.float32)
        if not isinstance(source_card, dict):
            return numeric, 0, 0.0

        source_profile = self._source_profile(source_card)
        source_text = self._source_text(source_card)
        source_id = str(source_card.get("id") or "").strip()
        source_title = str(source_card.get("title") or "").strip().lower()
        count = len(cards) if isinstance(cards, list) else 0
        if count <= 0:
            numeric[6] = source_profile["zero_cost"]
            numeric[7] = min(source_profile["draw"] / 5.0, 1.0)
            numeric[8] = min(source_profile["energy"] / 5.0, 1.0)
            numeric[9] = min(source_profile["hits"] / 10.0, 1.0)
            numeric[10] = source_profile["exhaust"]
            numeric[11] = source_profile["retain"]
            numeric[17] = float(any(keyword in source_text for keyword in ("discard", "draw pile", "shuffle", "return to your hand")))
            numeric[18] = float("exhaust" in source_text or "ethereal" in source_text)
            return numeric, 0, 0.0

        same_id_count = 0.0
        same_title_count = 0.0
        closest_position = 0
        top_match = 0.0
        near_match = 0.0
        zero_cost = 0.0
        draw_density = 0.0
        exhaust_density = 0.0
        retain_density = 0.0
        energy_density = 0.0
        total_cost = 0.0

        for index, card in enumerate(cards):
            if not isinstance(card, dict):
                continue
            profile = self._source_profile(card)
            entry_id = str(card.get("id") or "").strip()
            entry_title = str(card.get("title") or "").strip().lower()
            matched = False
            if source_id and entry_id and entry_id == source_id:
                same_id_count += 1.0
                matched = True
            if source_title and entry_title and entry_title == source_title:
                same_title_count += 1.0
                matched = True
            if matched and closest_position <= 0:
                closest_position = index + 1
                near_match = float(index < 3)
            if matched and index == 0:
                top_match = 1.0
            zero_cost += profile["zero_cost"]
            draw_density += float(profile["draw"] > 0.0)
            exhaust_density += float(profile["exhaust"] > 0.0 or profile["ethereal"] > 0.0)
            retain_density += profile["retain"]
            energy_density += float(profile["energy"] > 0.0)
            total_cost += max(profile["cost"], 0.0)

        denom = max(float(count), 1.0)
        numeric[0] = min(count / 30.0, 1.0)
        numeric[1] = min(same_id_count / 4.0, 1.0)
        numeric[2] = min(same_title_count / 4.0, 1.0)
        numeric[3] = min(closest_position / 10.0, 1.0) if closest_position > 0 else 0.0
        numeric[4] = top_match
        numeric[5] = near_match
        numeric[6] = source_profile["zero_cost"]
        numeric[7] = min(source_profile["draw"] / 5.0, 1.0)
        numeric[8] = min(max(source_profile["energy"], float(source_profile["x_cost"] > 0.5)) / 5.0, 1.0)
        numeric[9] = min(source_profile["hits"] / 10.0, 1.0)
        numeric[10] = source_profile["exhaust"]
        numeric[11] = source_profile["retain"]
        numeric[12] = zero_cost / denom
        numeric[13] = draw_density / denom
        numeric[14] = exhaust_density / denom
        numeric[15] = retain_density / denom
        numeric[16] = min(total_cost / denom / 5.0, 1.0)
        numeric[17] = float(any(keyword in source_text for keyword in ("discard", "draw pile", "shuffle", "return to your hand")))
        numeric[18] = float("exhaust" in source_text or "ethereal" in source_text)
        numeric[19] = float((same_id_count + same_title_count) > 0.0)
        return numeric, closest_position, max(same_id_count, same_title_count)

    def _append_cycle_plan_local(
        self,
        entries: list[dict[str, Any]],
        obs: dict[str, Any],
        action: dict[str, Any],
        source_card: dict[str, Any] | None,
    ) -> None:
        source = source_card if isinstance(source_card, dict) else action.get("potion") if isinstance(action.get("potion"), dict) else None
        source_profile = self._source_profile(source)
        source_text = self._source_text(source)
        hand_cards = self._runtime_cards(obs, "hand", "hand")
        draw_cards = self._runtime_cards(obs, "draw_pile", "draw_preview_cards")
        discard_cards = self._runtime_cards(obs, "discard_pile", "discard_cards")
        exhaust_cards = self._runtime_cards(obs, "exhaust_pile", "exhaust_cards")
        play_cards = self._runtime_cards(obs, "play_pile", "play_pile_cards")

        draw_match = self._find_source_position(draw_cards, source)
        discard_match = self._find_source_position(discard_cards, source)
        exhaust_match = self._find_source_position(exhaust_cards, source)
        play_match = self._find_source_position(play_cards, source)
        draw_count = len(draw_cards)
        discard_count = len(discard_cards)
        exhaust_count = len(exhaust_cards)
        play_count = len(play_cards)
        hand_count = len(hand_cards)

        reshuffle_pressure = float(draw_count <= 2 and discard_count >= 4)
        cycle_keyword = float(
            any(
                keyword in source_text
                for keyword in (
                    "draw",
                    "discard",
                    "shuffle",
                    "return to your hand",
                    "draw pile",
                    "discard pile",
                )
            )
        )
        exhaust_keyword = float("exhaust" in source_text or "ethereal" in source_text)
        numeric = np.zeros(TOKEN_NUMERIC_DIM, dtype=np.float32)
        numeric[0] = min(hand_count / 10.0, 1.0)
        numeric[1] = min(draw_count / 30.0, 1.0)
        numeric[2] = min(discard_count / 30.0, 1.0)
        numeric[3] = min(exhaust_count / 20.0, 1.0)
        numeric[4] = min(play_count / 12.0, 1.0)
        numeric[5] = reshuffle_pressure
        numeric[6] = float(draw_count <= 3)
        numeric[7] = float(discard_count >= 4)
        numeric[8] = min(draw_match / 10.0, 1.0) if draw_match > 0 else 0.0
        numeric[9] = min(discard_match / 10.0, 1.0) if discard_match > 0 else 0.0
        numeric[10] = min(exhaust_match / 10.0, 1.0) if exhaust_match > 0 else 0.0
        numeric[11] = min(play_match / 10.0, 1.0) if play_match > 0 else 0.0
        numeric[12] = min(source_profile["draw"] / 5.0, 1.0)
        numeric[13] = source_profile["zero_cost"]
        numeric[14] = source_profile["exhaust"]
        numeric[15] = source_profile["retain"]
        numeric[16] = cycle_keyword
        numeric[17] = float(any(keyword in source_text for keyword in ("return", "shuffle", "draw pile", "discard pile")))
        numeric[18] = exhaust_keyword
        numeric[19] = float((cycle_keyword > 0.0 or source_profile["zero_cost"] > 0.0) and (discard_count > 0 or draw_count <= 3))
        numeric[20] = float((discard_match > 0 or draw_match > 0) and cycle_keyword > 0.0)
        numeric[21] = float(exhaust_match > 0 and exhaust_keyword > 0.0)
        numeric[22] = float(source_profile["energy"] > 0.0 and draw_count <= 3)
        numeric[23] = float(source_profile["hits"] > 1.0 and play_count > 0)
        source_entity_id = _stable_bucket(source.get("id") or source.get("title") or self._action_entity_key(action)) if isinstance(source, dict) else _stable_bucket(self._action_entity_key(action))
        entries.append(
            self._entry(
                "CYCLE_PLAN_LOCAL",
                numeric,
                owner_id=OWNER_PLAYER,
                entity_id=source_entity_id,
                order_id=6,
                text=f"cycle plan | draw {draw_count} discard {discard_count} exhaust {exhaust_count} play {play_count}",
            )
        )

    def _append_energy_budget_local(
        self,
        entries: list[dict[str, Any]],
        obs: dict[str, Any],
        features: dict[str, np.ndarray],
        action: dict[str, Any],
        source_profile: dict[str, float],
    ) -> None:
        combat = obs.get("combat") or {}
        player = obs.get("player") or {}
        current_energy = obs_common._float(combat.get("energy"))
        max_energy = obs_common._float(combat.get("max_energy"))
        source_cost = max(source_profile["cost"], 0.0)
        source_is_x = float(source_profile["x_cost"] > 0.5)

        best_energy_potion = 0.0
        energy_potion_count = 0.0
        for potion in (player.get("potions") or [])[: obs_common.MAX_POTIONS]:
            if isinstance(potion, str) and potion.strip() == "[empty]":
                continue
            potion_profile = self._source_profile(potion if isinstance(potion, dict) else None)
            potion_energy = max(potion_profile["energy"], obs_common._float(potion.get("energy")) if isinstance(potion, dict) else 0.0)
            if potion_energy > 0.0:
                energy_potion_count += 1.0
                best_energy_potion = max(best_energy_potion, potion_energy)

        relic_signals = features["relic_signals"] if "relic_signals" in features else np.zeros(obs_common.RELIC_SIGNAL_DIM, dtype=np.float32)
        relic_energy_signal = float(relic_signals[0]) if relic_signals.shape[0] > 0 else 0.0
        relic_draw_signal = float(relic_signals[1]) if relic_signals.shape[0] > 1 else 0.0
        extra_energy = source_profile["energy"] + best_energy_potion + 2.0 * relic_energy_signal

        can_play_now = float((source_is_x > 0.0 and current_energy > 0.0) or current_energy >= source_cost)
        can_expand_with_support = float((source_is_x > 0.0 and current_energy + extra_energy > 0.0) or current_energy + extra_energy >= source_cost)
        max_spend = current_energy + best_energy_potion + max(source_profile["energy"], 0.0)

        numeric = np.zeros(TOKEN_NUMERIC_DIM, dtype=np.float32)
        numeric[0] = min(current_energy / max(max_energy, 1.0), 1.0) if max_energy > 0 else 0.0
        numeric[1] = min(current_energy / 10.0, 1.0)
        numeric[2] = min(max_energy / 10.0, 1.0)
        numeric[3] = min(source_cost / 5.0, 1.0)
        numeric[4] = source_is_x
        numeric[5] = min(source_profile["energy"] / 5.0, 1.0)
        numeric[6] = min(best_energy_potion / 5.0, 1.0)
        numeric[7] = min(energy_potion_count / max(obs_common.MAX_POTIONS, 1), 1.0)
        numeric[8] = relic_energy_signal
        numeric[9] = relic_draw_signal
        numeric[10] = can_play_now
        numeric[11] = can_expand_with_support
        numeric[12] = min(max_spend / 10.0, 1.0)
        numeric[13] = source_profile["zero_cost"]
        numeric[14] = min(source_profile["draw"] / 5.0, 1.0)
        numeric[15] = min(source_profile["hits"] / 10.0, 1.0)
        numeric[16] = source_profile["attack"]
        numeric[17] = max(source_profile["skill"], float(source_profile["block"] > 0.0))
        numeric[18] = float(source_is_x > 0.0 and max_spend >= 3.0)
        numeric[19] = float(source_profile["energy"] > 0.0 and current_energy < source_cost)
        entries.append(
            self._entry(
                "ENERGY_BUDGET_LOCAL",
                numeric,
                owner_id=OWNER_PLAYER,
                entity_id=_stable_bucket(self._action_entity_key(action)),
                order_id=3,
                text=f"energy budget | now {int(current_energy)} | support {best_energy_potion:.0f} | x {int(source_is_x)}",
            )
        )

    def _append_relic_potion_graph_local(
        self,
        entries: list[dict[str, Any]],
        obs: dict[str, Any],
        action: dict[str, Any],
        features: dict[str, np.ndarray],
        target_enemy_index: int | None,
        target_enemy: dict[str, Any] | None,
    ) -> None:
        player = obs.get("player") or {}
        source = action.get("card") if isinstance(action.get("card"), dict) else action.get("potion") if isinstance(action.get("potion"), dict) else None
        source_profile = self._source_profile(source)

        relic_scores: list[float] = []
        for index, relic in enumerate((player.get("relics") or [])[: obs_common.MAX_RELICS]):
            if not isinstance(relic, dict | str):
                continue
            relic_scores.append(float(self._relic_numeric(relic, action=action, index=index, total=max(len(player.get("relics") or []), 1))[0]))

        potion_scores: list[float] = []
        best_energy_potion = 0.0
        best_damage_potion = 0.0
        best_block_potion = 0.0
        best_draw_potion = 0.0
        for index, potion in enumerate((player.get("potions") or [])[: obs_common.MAX_POTIONS]):
            title = potion if isinstance(potion, str) else (potion.get("title") if isinstance(potion, dict) else "")
            if str(title or "").strip() == "[empty]":
                continue
            potion_numeric = self._potion_numeric(potion, source_profile=source_profile, slot_index=index, total_slots=max(len(player.get("potions") or []), 1))
            potion_scores.append(float(potion_numeric[0]))
            best_damage_potion = max(best_damage_potion, float(potion_numeric[1]))
            best_block_potion = max(best_block_potion, float(potion_numeric[2]))
            best_draw_potion = max(best_draw_potion, float(potion_numeric[3]))
            best_energy_potion = max(best_energy_potion, float(potion_numeric[4]))

        relic_signals = features["relic_signals"] if "relic_signals" in features else np.zeros(obs_common.RELIC_SIGNAL_DIM, dtype=np.float32)
        enemy_reactions = self._enemy_reaction_flags(target_enemy)

        top_relic = max(relic_scores, default=0.0)
        top_potion = max(potion_scores, default=0.0)
        avg_relic = float(np.mean(sorted(relic_scores, reverse=True)[:3])) if relic_scores else 0.0
        avg_potion = float(np.mean(sorted(potion_scores, reverse=True)[:2])) if potion_scores else 0.0

        numeric = np.zeros(TOKEN_NUMERIC_DIM, dtype=np.float32)
        numeric[0] = top_relic
        numeric[1] = avg_relic
        numeric[2] = top_potion
        numeric[3] = avg_potion
        numeric[4] = best_energy_potion
        numeric[5] = best_damage_potion
        numeric[6] = best_block_potion
        numeric[7] = best_draw_potion
        numeric[8] = float(relic_signals[0]) if relic_signals.shape[0] > 0 else 0.0
        numeric[9] = float(relic_signals[1]) if relic_signals.shape[0] > 1 else 0.0
        numeric[10] = max(float(relic_signals[2]) if relic_signals.shape[0] > 2 else 0.0, float(relic_signals[4]) if relic_signals.shape[0] > 4 else 0.0)
        numeric[11] = max(float(relic_signals[3]) if relic_signals.shape[0] > 3 else 0.0, float(relic_signals[10]) if relic_signals.shape[0] > 10 else 0.0)
        numeric[12] = enemy_reactions["contact_punish"] * max(best_damage_potion, best_block_potion, best_energy_potion) * max(source_profile["attack"], float(source_profile["damage"] > 0.0))
        numeric[13] = float(np.clip(top_relic + top_potion, 0.0, 1.0))
        numeric[14] = source_profile["attack"]
        numeric[15] = max(source_profile["skill"], float(source_profile["block"] > 0.0))
        numeric[16] = float(source_profile["x_cost"] > 0.5) * max(best_energy_potion, numeric[8])
        numeric[17] = max(source_profile["exhaust"], float("exhaust" in self._source_text(source))) * max(top_relic, top_potion)
        numeric[18] = float(target_enemy_index is not None and enemy_reactions["contact_punish"] > 0.0)
        numeric[19] = float(source_profile["hits"] > 1.0) * enemy_reactions["thorns"]
        entries.append(
            self._entry(
                "RELIC_POTION_GRAPH_LOCAL",
                numeric,
                owner_id=OWNER_PLAYER,
                entity_id=_stable_bucket(self._action_entity_key(action)),
                order_id=7,
                text=f"support graph | relic {top_relic:.2f} potion {top_potion:.2f} contact {enemy_reactions['contact_punish']:.0f}",
            )
        )

    def _pile_context_numeric(self, cards: list[Any], source_card: dict[str, Any] | None) -> np.ndarray:
        numeric = np.zeros(TOKEN_NUMERIC_DIM, dtype=np.float32)
        count = len(cards) if isinstance(cards, list) else 0
        if count <= 0:
            return numeric

        source_id = str((source_card or {}).get("id") or "").strip()
        source_title = str((source_card or {}).get("title") or "").strip().lower()
        attack = skill = power = status_or_curse = 0.0
        zero_cost = exhaust_kw = ethereal_kw = retain_kw = innate_kw = 0.0
        same_source = 0.0
        total_cost = total_damage = total_block = total_draw = total_energy = total_hits = 0.0

        for card in cards:
            if not isinstance(card, dict):
                continue
            card_type = str(card.get("type") or "").capitalize()
            attack += 1.0 if card_type == "Attack" else 0.0
            skill += 1.0 if card_type == "Skill" else 0.0
            power += 1.0 if card_type == "Power" else 0.0
            status_or_curse += 1.0 if card_type in {"Status", "Curse"} else 0.0
            cost = obs_common._float(card.get("cost"))
            total_cost += max(cost, 0.0)
            zero_cost += 1.0 if cost == 0 else 0.0
            preview = obs_common._build_card_preview_bundle(card)
            total_damage += preview["preview_damage"]
            total_block += preview["preview_block"]
            total_draw += obs_common._preview_metric(card, "draw")
            total_energy += obs_common._preview_metric(card, "energy")
            total_hits += obs_common._get_card_extra_metrics(card)[3]
            kw_flags, _ = obs_common._get_card_keywords(card)
            exhaust_kw += 1.0 if kw_flags[0] else 0.0
            ethereal_kw += 1.0 if kw_flags[1] else 0.0
            retain_kw += 1.0 if kw_flags[2] else 0.0
            innate_kw += 1.0 if kw_flags[3] else 0.0
            card_id = str(card.get("id") or "").strip()
            card_title = str(card.get("title") or "").strip().lower()
            if (source_id and card_id and card_id == source_id) or (source_title and card_title and card_title == source_title):
                same_source += 1.0

        denom = max(float(count), 1.0)
        numeric[0] = min(count / 30.0, 1.0)
        numeric[1] = attack / denom
        numeric[2] = skill / denom
        numeric[3] = power / denom
        numeric[4] = status_or_curse / denom
        numeric[5] = min(total_cost / denom / 5.0, 1.0)
        numeric[6] = zero_cost / denom
        numeric[7] = exhaust_kw / denom
        numeric[8] = ethereal_kw / denom
        numeric[9] = retain_kw / denom
        numeric[10] = innate_kw / denom
        numeric[11] = same_source / denom
        numeric[12] = obs_common._log_norm(total_damage, obs_common._LOG1P_200)
        numeric[13] = obs_common._log_norm(total_block, obs_common._LOG1P_200)
        numeric[14] = min(total_draw / 10.0, 1.0)
        numeric[15] = min(total_energy / 10.0, 1.0)
        numeric[16] = min(total_hits / 20.0, 1.0)
        return numeric

    def _append_pile_peek_locals(
        self,
        entries: list[dict[str, Any]],
        token_type: str,
        owner_id: int,
        cards: list[Any],
        *,
        limit: int,
    ) -> None:
        if not isinstance(cards, list) or not cards:
            return
        numeric = np.zeros((limit, obs_common.CARD_FEAT_DIM), dtype=np.float32)
        text = np.zeros((limit, TEXT_DIM), dtype=np.float32)
        mask = np.zeros(limit, dtype=np.float32)
        self._enc_card_collection(cards[:limit], numeric, text, mask)
        for index in range(limit):
            if mask[index] <= 0:
                continue
            row = np.zeros(TOKEN_NUMERIC_DIM, dtype=np.float32)
            row[: min(TOKEN_NUMERIC_DIM, numeric.shape[1])] = numeric[index][:TOKEN_NUMERIC_DIM]
            row[min(TOKEN_NUMERIC_DIM - 1, numeric.shape[1])] = min((index + 1) / max(limit, 1), 1.0)
            card = cards[index] if index < len(cards) and isinstance(cards[index], dict) else None
            entity_key = card.get("id") if isinstance(card, dict) and card.get("id") else card.get("title") if isinstance(card, dict) else f"{token_type}:{index}"
            entries.append(
                self._entry(
                    token_type,
                    row,
                    owner_id=owner_id,
                    entity_id=_stable_bucket(entity_key),
                    order_id=index + 1,
                    text_embedding=text[index],
                )
            )

    def _append_target_reaction_local(
        self,
        entries: list[dict[str, Any]],
        target_enemy_index: int,
        target_enemy: dict[str, Any],
        source: dict[str, Any] | None,
        obs: dict[str, Any] | None = None,
    ) -> None:
        source_profile = self._source_profile_for_obs(source, obs)
        reactions = self._enemy_reaction_flags(target_enemy)
        hp = obs_common._float(target_enemy.get("hp", target_enemy.get("current_hp")))
        block = obs_common._float(target_enemy.get("block"))
        intent = target_enemy.get("intent") if isinstance(target_enemy.get("intent"), dict) else {}
        expected_damage = source_profile["damage"] * max(source_profile["hits"], 1.0 if source_profile["damage"] > 0.0 else 0.0)
        attack_like = max(source_profile["attack"], float(source_profile["damage"] > 0.0))

        numeric = np.zeros(TOKEN_NUMERIC_DIM, dtype=np.float32)
        numeric[0] = reactions["thorns"]
        numeric[1] = reactions["contact_punish"]
        numeric[2] = reactions["split"]
        numeric[3] = reactions["threshold"]
        numeric[4] = reactions["threshold_value"]
        numeric[5] = attack_like
        numeric[6] = float(source_profile["hits"] > 1.0)
        numeric[7] = reactions["contact_punish"] * attack_like
        numeric[8] = reactions["contact_punish"] * float(source_profile["hits"] > 1.0)
        numeric[9] = float(expected_damage >= max(hp + block, 1.0) and expected_damage > 0.0)
        numeric[10] = float(obs_common._float(intent.get("total_damage")) > 0.0)
        numeric[11] = obs_common._log_norm(obs_common._float(intent.get("total_damage")), obs_common._LOG1P_200)
        numeric[12] = obs_common._log_norm(block, obs_common._LOG1P_200)
        numeric[13] = obs_common._log_norm(hp, obs_common._LOG1P_1200)
        numeric[14] = float(source_profile["x_cost"] > 0.5)
        numeric[15] = max(reactions["artifact"], reactions["buffer"], reactions["intangible"])
        numeric[16] = float(source_profile["weak"] > 0.0 or source_profile["vulnerable"] > 0.0)
        numeric[17] = float(source_profile["aoe_target"] > 0.0)
        numeric[18] = reactions["incoming_damage_multiplier"]
        numeric[19] = reactions["back_attack"]
        numeric[20] = reactions["damage_cap"]
        numeric[21] = reactions["damage_cap_value"]
        numeric[22] = reactions["deathburst"]
        numeric[23] = reactions["deathburst_damage"]
        numeric[24] = reactions["revive"]
        numeric[25] = reactions["transform"]
        numeric[26] = reactions["linked_support_alive"]
        numeric[27] = reactions["special_phase"]
        numeric[28] = reactions["one_card_lock"]
        numeric[29] = reactions["skill_punish"]
        numeric[30] = reactions["choice_debuffs"]
        numeric[31] = reactions["escape_card_tax"]
        numeric[32] = reactions["countdown"]
        numeric[33] = reactions["stun_window"]
        numeric[34] = reactions["binding_control"]
        numeric[35] = reactions["wound_phase"]
        entries.append(
            self._entry(
                "TARGET_REACTION_LOCAL",
                numeric,
                owner_id=self._enemy_owner_id(target_enemy_index),
                entity_id=_stable_bucket(self._enemy_entity_key(target_enemy, target_enemy_index)),
                order_id=24,
                text=(
                    "target reaction"
                    f" | thorns {reactions['thorns']:.0f}"
                    f" | threshold {reactions['threshold']:.0f}"
                    f" | back {reactions['back_attack']:.0f}"
                    f" | cap {reactions['damage_cap']:.0f}"
                    f" | burst {reactions['deathburst']:.0f}"
                ),
            )
        )

    def _append_target_enemy_local_context(
        self,
        entries: list[dict[str, Any]],
        target_enemy_index: int,
        target_enemy: dict[str, Any],
    ) -> None:
        owner_id = self._enemy_owner_id(target_enemy_index)
        entity_id = _stable_bucket(self._enemy_entity_key(target_enemy, target_enemy_index))
        combat_memory = (self._current_planner_context or {}).get("combat_memory") or {}
        memory_enemies = combat_memory.get("enemies") if isinstance(combat_memory, dict) else None
        enemy_memory = None
        if isinstance(memory_enemies, dict):
            enemy_memory = memory_enemies.get(self._enemy_entity_key(target_enemy, target_enemy_index))
        intent = target_enemy.get("intent") if isinstance(target_enemy.get("intent"), dict) else None
        if isinstance(intent, dict):
            numeric = np.zeros(TOKEN_NUMERIC_DIM, dtype=np.float32)
            numeric[0] = obs_common._log_norm(obs_common._float(intent.get("total_damage")), obs_common._LOG1P_200)
            numeric[1] = obs_common._log_norm(obs_common._float(intent.get("damage_per_hit")), obs_common._LOG1P_100)
            numeric[2] = min(obs_common._float(intent.get("repeats")) / 5.0, 1.0)
            numeric[3:7] = np.asarray(obs_common._infer_enemy_intent_flags(intent, obs_common._float(intent.get("total_damage"))), dtype=np.float32)
            if isinstance(enemy_memory, dict):
                numeric[7] = obs_common._float(enemy_memory.get("intent_changed_this_turn"))
                numeric[8] = obs_common._float(enemy_memory.get("turns_since_intent_change_norm"))
                numeric[9] = obs_common._float(enemy_memory.get("intent_total_damage_delta"))
                numeric[10] = obs_common._float(enemy_memory.get("intent_damage_per_hit_delta"))
                numeric[11] = obs_common._float(enemy_memory.get("intent_damage_trend_3_turns"))
                numeric[12] = obs_common._float(enemy_memory.get("intent_predicted_vs_actual"))
            entries.append(
                self._entry(
                    "ENEMY_INTENT",
                    numeric,
                    owner_id=owner_id,
                    entity_id=entity_id,
                    order_id=2,
                    text=str(intent.get("description") or intent.get("label") or ""),
                )
            )
        memory_powers = enemy_memory.get("powers") if isinstance(enemy_memory, dict) else None
        for power_index, power in enumerate((target_enemy.get("powers") or [])[:2]):
            if not isinstance(power, dict):
                continue
            numeric = np.zeros(TOKEN_NUMERIC_DIM, dtype=np.float32)
            amount = obs_common._float(power.get("amount") or power.get("display_amount"))
            title = str(power.get("title") or "").lower()
            numeric[0] = obs_common._normalize_power_amount(amount)
            numeric[1] = float("thorn" in title or "spike" in title)
            numeric[2] = float("artifact" in title)
            numeric[3] = float("buffer" in title)
            numeric[4] = float("intang" in title)
            power_key = None
            for candidate in ("id", "power_id", "type", "key"):
                value = power.get(candidate)
                if value not in (None, ""):
                    power_key = str(value)
                    break
            if power_key is None and power.get("title") not in (None, ""):
                power_key = f"title::{power.get('title')}"
            power_memory = memory_powers.get(power_key) if isinstance(memory_powers, dict) and power_key else None
            if isinstance(power_memory, dict):
                delta_last = obs_common._float(power_memory.get("amount_delta_last_turn"))
                delta_since = obs_common._float(power_memory.get("amount_delta_since_first_seen"))
                numeric[5] = obs_common._normalize_power_amount(delta_last) * (1.0 if delta_last >= 0 else -1.0)
                numeric[6] = obs_common._normalize_power_amount(delta_since) * (1.0 if delta_since >= 0 else -1.0)
                numeric[7] = min(obs_common._float(power_memory.get("turns_since_first_seen")) / 10.0, 1.0)
                numeric[8] = obs_common._float(power_memory.get("stack_trend_3_turns"))
                numeric[9] = obs_common._float(power_memory.get("is_new_this_turn"))
                numeric[10] = obs_common._float(power_memory.get("is_growing_without_player_action"))
            entries.append(
                self._entry(
                    "ENEMY_POWER",
                    numeric,
                    owner_id=owner_id,
                    entity_id=entity_id,
                    order_id=3 + power_index,
                    text=f"{power.get('title') or ''} | {power.get('description') or ''}",
                )
            )
        reactive_traits, phase_rules = self._infer_enemy_traits(target_enemy)
        for trait_index, trait in enumerate(reactive_traits[:2]):
            entries.append(
                self._entry(
                    "ENEMY_REACTIVE_TRAIT",
                    self._trait_numeric(trait),
                    owner_id=owner_id,
                    entity_id=entity_id,
                    order_id=12 + trait_index,
                    text=str(trait.get("description") or trait.get("trait") or trait.get("effect_type") or ""),
                )
            )
        for rule_index, rule in enumerate(phase_rules[:1]):
            entries.append(
                self._entry(
                    "ENEMY_PHASE_RULE",
                    self._trait_numeric(rule),
                    owner_id=owner_id,
                    entity_id=entity_id,
                    order_id=20 + rule_index,
                    text=str(rule.get("description") or rule.get("trait") or ""),
                )
            )
