"""Token-world observation encoder for the omni-attention online policy.

The public encoder and token schema stay in this compatibility module while
domain-specific token emitters live in private mixins.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from content_registry import (
    build_live_enemy_semantic_text,
    build_live_potion_semantic_text,
    build_live_relic_semantic_text,
    get_enemy_metadata,
)

from . import observation_common as obs_common
from ._observation_v3_binding import TokenBindingMaterializationMixin
from ._observation_v3_candidates import CandidateTokenMixin
from ._observation_v3_schema import (
    _CARD_KEYWORD_BUCKETS,
    ENTITY_HASH_BUCKETS,
    MAX_ACTIONS,
    MAX_CANDIDATE_LOCAL_TOKENS,
    MAX_CARD_KEYWORD_SLOTS,
    MAX_ORDER_ID,
    MAX_OWNER_ID,
    MAX_POWER_SLOT_TOKENS,
    MAX_ROLE_ID,
    MAX_WORLD_TOKENS,
    MAX_ZONE_ID,
    NUM_TOKEN_TYPES,
    OBSERVATION_API_VERSION,
    OWNER_DECK,
    OWNER_DISCARD,
    OWNER_DRAW,
    OWNER_ENEMY_BASE,
    OWNER_EXHAUST,
    OWNER_HAND,
    OWNER_HISTORY,
    OWNER_NONE,
    OWNER_PLAY,
    OWNER_PLAYER,
    OWNER_POTION,
    OWNER_POWER,
    OWNER_RELIC,
    OWNER_REWARD,
    OWNER_ROUTE,
    OWNER_SHOP,
    OWNER_UPGRADE,
    POWER_ID_BUCKETS,
    TOKEN_FEAT_DIM,
    TOKEN_NUMERIC_DIM,
    TOKEN_ROLE_TO_ID,
    TOKEN_ROLES,
    TOKEN_TEXT_DIM,
    TOKEN_TYPE_TO_ID,
    TOKEN_TYPES,
    TOKEN_ZONE_TO_ID,
    TOKEN_ZONES,
    _compress_text_embedding,
)
from ._observation_v3_support import SupportTokenMixin, _resolve_potion_effect
from ._observation_v3_world import WorldTokenMixin
from .action_history import (
    KEY_POWER_CARD_BUCKETS,
    MAX_HISTORY_TOKENS,
    MAX_STEP_DETAIL_TOKENS,
    MAX_TURN_SUMMARY_TOKENS,
    NUM_KEY_POWER_FLAGS,
    NUM_SEMANTIC_ROLES,
)
from .boss_mechanics import build_boss_mechanics_context, enemy_mechanics_key
from .hand_mutation import (
    infer_hand_mutation,
    mutation_summary_numeric,
    mutation_target_numeric,
    post_hand_preview_numeric,
)
from .potion_profiles import DEFAULT_EFFECT_PROFILE
from .text_encoder import TEXT_DIM


class WorldTokenObservationEncoder(
    CandidateTokenMixin,
    WorldTokenMixin,
    SupportTokenMixin,
    TokenBindingMaterializationMixin,
    obs_common.DenseObservationEncoder,
):
    """Observation V3 built as a tokenized world memory plus candidate tokens."""

    def __init__(self, use_text: bool = True, text_device: str = "cpu"):
        super().__init__(use_text=use_text, text_device=text_device)
        # Pre-allocate output buffers for _materialize_entries / _materialize_nested_entries
        # to avoid ~3 MB of np.zeros allocation every encode() call.
        self._buf_world = self._alloc_flat_bufs(MAX_WORLD_TOKENS)
        self._buf_candidate = self._alloc_flat_bufs(MAX_ACTIONS)
        self._buf_candidate_local = self._alloc_nested_bufs(MAX_ACTIONS, MAX_CANDIDATE_LOCAL_TOKENS)
        self._current_planner_context: dict[str, Any] | None = None
        self._current_boss_context: dict[str, Any] | None = None

    @staticmethod
    def _alloc_flat_bufs(n):
        return {
            "tokens": np.zeros((n, TOKEN_FEAT_DIM), dtype=np.float32),
            "mask": np.zeros(n, dtype=np.float32),
            "type_ids": np.zeros(n, dtype=np.int32),
            "role_ids": np.zeros(n, dtype=np.int32),
            "owner_ids": np.zeros(n, dtype=np.int32),
            "entity_ids": np.zeros(n, dtype=np.int32),
            "zone_ids": np.zeros(n, dtype=np.int32),
            "order_ids": np.zeros(n, dtype=np.int32),
            "target_owner_ids": np.zeros(n, dtype=np.int32),
            "target_entity_ids": np.zeros(n, dtype=np.int32),
        }

    @staticmethod
    def _alloc_nested_bufs(n_outer, n_inner):
        return {
            "tokens": np.zeros((n_outer, n_inner, TOKEN_FEAT_DIM), dtype=np.float32),
            "mask": np.zeros((n_outer, n_inner), dtype=np.float32),
            "type_ids": np.zeros((n_outer, n_inner), dtype=np.int32),
            "role_ids": np.zeros((n_outer, n_inner), dtype=np.int32),
            "owner_ids": np.zeros((n_outer, n_inner), dtype=np.int32),
            "entity_ids": np.zeros((n_outer, n_inner), dtype=np.int32),
            "zone_ids": np.zeros((n_outer, n_inner), dtype=np.int32),
            "order_ids": np.zeros((n_outer, n_inner), dtype=np.int32),
        }

    @property
    def obs_space(self):
        from gymnasium import spaces

        inf = np.inf
        return spaces.Dict(
            {
                "world_tokens": spaces.Box(-inf, inf, (MAX_WORLD_TOKENS, TOKEN_FEAT_DIM), dtype=np.float32),
                "world_token_mask": spaces.Box(0, 1, (MAX_WORLD_TOKENS,), dtype=np.float32),
                "world_token_type_ids": spaces.Box(0, NUM_TOKEN_TYPES, (MAX_WORLD_TOKENS,), dtype=np.int32),
                "world_token_role_ids": spaces.Box(0, MAX_ROLE_ID, (MAX_WORLD_TOKENS,), dtype=np.int32),
                "world_entity_owner_ids": spaces.Box(0, MAX_OWNER_ID, (MAX_WORLD_TOKENS,), dtype=np.int32),
                "world_token_entity_ids": spaces.Box(0, ENTITY_HASH_BUCKETS, (MAX_WORLD_TOKENS,), dtype=np.int32),
                "world_token_zone_ids": spaces.Box(0, MAX_ZONE_ID, (MAX_WORLD_TOKENS,), dtype=np.int32),
                "world_token_order_ids": spaces.Box(0, MAX_ORDER_ID, (MAX_WORLD_TOKENS,), dtype=np.int32),
                "candidate_query_tokens": spaces.Box(-inf, inf, (MAX_ACTIONS, TOKEN_FEAT_DIM), dtype=np.float32),
                "candidate_query_type_ids": spaces.Box(0, NUM_TOKEN_TYPES, (MAX_ACTIONS,), dtype=np.int32),
                "candidate_query_role_ids": spaces.Box(0, MAX_ROLE_ID, (MAX_ACTIONS,), dtype=np.int32),
                "candidate_query_owner_ids": spaces.Box(0, MAX_OWNER_ID, (MAX_ACTIONS,), dtype=np.int32),
                "candidate_query_entity_ids": spaces.Box(0, ENTITY_HASH_BUCKETS, (MAX_ACTIONS,), dtype=np.int32),
                "candidate_query_zone_ids": spaces.Box(0, MAX_ZONE_ID, (MAX_ACTIONS,), dtype=np.int32),
                "candidate_query_order_ids": spaces.Box(0, MAX_ORDER_ID, (MAX_ACTIONS,), dtype=np.int32),
                "candidate_query_target_owner_ids": spaces.Box(0, MAX_OWNER_ID, (MAX_ACTIONS,), dtype=np.int32),
                "candidate_query_target_entity_ids": spaces.Box(0, ENTITY_HASH_BUCKETS, (MAX_ACTIONS,), dtype=np.int32),
                "candidate_local_tokens": spaces.Box(-inf, inf, (MAX_ACTIONS, MAX_CANDIDATE_LOCAL_TOKENS, TOKEN_FEAT_DIM), dtype=np.float32),
                "candidate_local_masks": spaces.Box(0, 1, (MAX_ACTIONS, MAX_CANDIDATE_LOCAL_TOKENS), dtype=np.float32),
                "candidate_local_type_ids": spaces.Box(0, NUM_TOKEN_TYPES, (MAX_ACTIONS, MAX_CANDIDATE_LOCAL_TOKENS), dtype=np.int32),
                "candidate_local_role_ids": spaces.Box(0, MAX_ROLE_ID, (MAX_ACTIONS, MAX_CANDIDATE_LOCAL_TOKENS), dtype=np.int32),
                "candidate_local_owner_ids": spaces.Box(0, MAX_OWNER_ID, (MAX_ACTIONS, MAX_CANDIDATE_LOCAL_TOKENS), dtype=np.int32),
                "candidate_local_entity_ids": spaces.Box(0, ENTITY_HASH_BUCKETS, (MAX_ACTIONS, MAX_CANDIDATE_LOCAL_TOKENS), dtype=np.int32),
                "candidate_local_zone_ids": spaces.Box(0, MAX_ZONE_ID, (MAX_ACTIONS, MAX_CANDIDATE_LOCAL_TOKENS), dtype=np.int32),
                "candidate_local_order_ids": spaces.Box(0, MAX_ORDER_ID, (MAX_ACTIONS, MAX_CANDIDATE_LOCAL_TOKENS), dtype=np.int32),
                "action_mask": spaces.Box(0, 1, (MAX_ACTIONS,), dtype=np.float32),
            }
        )

    def encode(
        self,
        obs: dict | None,
        legal_actions: list | None = None,
        planner_context: dict | None = None,
    ) -> dict[str, np.ndarray]:
        obs_dict = obs or {}
        action_list = legal_actions or []
        planner_context = self._normalize_planner_context(obs_dict, action_list, planner_context)
        world_entries: list[dict[str, Any]] = []
        candidate_entries: list[dict[str, Any]] = []
        candidate_local_entries: list[list[dict[str, Any]]] = [[] for _ in range(MAX_ACTIONS)]
        action_mask = np.zeros(MAX_ACTIONS, dtype=np.float32)

        try:
            self._begin_text_registry()
            features = self._build_feature_view(obs_dict, action_list, planner_context)
            self._current_planner_context = planner_context
            self._current_boss_context = build_boss_mechanics_context(obs_dict)
            self._append_global_tokens(world_entries, obs_dict, features, planner_context)
            self._append_entity_tokens(world_entries, obs_dict, features, planner_context)
            # v3: POWER_SLOT tokens split out of entity-inlined numerics.
            # One token per power instance on player + each enemy. Emits
            # (power_id bucket as entity_id) + (effect-algebra vector in
            # numeric slots 0..12) + (amount scalars in slots 13..15).
            self._append_power_slot_tokens(world_entries, obs_dict)
            # v3: CARD_KEYWORD_SLOT tokens for Retain/Ethereal/Exhaust/etc.
            # One token per keyword per hand card, surfaces keyword binding
            # to the POWER bank where attention can learn the interaction
            # patterns ("Ethereal + turn ending + not playable = waste").
            self._append_card_keyword_slot_tokens(world_entries, obs_dict)
            # v4 (Phase 8 Tier 1): HISTORY tokens. 20 step-detail slots +
            # 8 turn-summary slots, populated from env_v2's
            # ActionHistoryTracker snapshot attached under
            # obs["_action_history"]. Padded tokens carry is_empty=1.
            self._append_history_tokens(world_entries, obs_dict)
            self._append_candidate_tokens(candidate_entries, candidate_local_entries, action_mask, obs_dict, features, action_list)
            self._resolve_entry_text_embeddings(world_entries, candidate_entries, candidate_local_entries)
            self._resolve_text_registry()

            (
                world_tokens,
                world_token_mask,
                world_token_type_ids,
                world_token_role_ids,
                world_owner_ids,
                world_entity_ids,
                world_zone_ids,
                world_order_ids,
                _world_target_owner_ids,
                _world_target_entity_ids,
            ) = self._materialize_entries(world_entries, MAX_WORLD_TOKENS, bufs=self._buf_world)
            (
                candidate_query_tokens,
                _unused_mask,
                candidate_query_type_ids,
                candidate_query_role_ids,
                candidate_query_owner_ids,
                candidate_query_entity_ids,
                candidate_query_zone_ids,
                candidate_query_order_ids,
                candidate_query_target_owner_ids,
                candidate_query_target_entity_ids,
            ) = self._materialize_entries(candidate_entries, MAX_ACTIONS, bufs=self._buf_candidate)
            (
                candidate_local_tokens,
                candidate_local_masks,
                candidate_local_type_ids,
                candidate_local_role_ids,
                candidate_local_owner_ids,
                candidate_local_entity_ids,
                candidate_local_zone_ids,
                candidate_local_order_ids,
            ) = self._materialize_nested_entries(candidate_local_entries, MAX_ACTIONS, MAX_CANDIDATE_LOCAL_TOKENS, bufs=self._buf_candidate_local)

            return {
                "world_tokens": world_tokens,
                "world_token_mask": world_token_mask,
                "world_token_type_ids": world_token_type_ids,
                "world_token_role_ids": world_token_role_ids,
                "world_entity_owner_ids": world_owner_ids,
                "world_token_entity_ids": world_entity_ids,
                "world_token_zone_ids": world_zone_ids,
                "world_token_order_ids": world_order_ids,
                "candidate_query_tokens": candidate_query_tokens,
                "candidate_query_type_ids": candidate_query_type_ids,
                "candidate_query_role_ids": candidate_query_role_ids,
                "candidate_query_owner_ids": candidate_query_owner_ids,
                "candidate_query_entity_ids": candidate_query_entity_ids,
                "candidate_query_zone_ids": candidate_query_zone_ids,
                "candidate_query_order_ids": candidate_query_order_ids,
                "candidate_query_target_owner_ids": candidate_query_target_owner_ids,
                "candidate_query_target_entity_ids": candidate_query_target_entity_ids,
                "candidate_local_tokens": candidate_local_tokens,
                "candidate_local_masks": candidate_local_masks,
                "candidate_local_type_ids": candidate_local_type_ids,
                "candidate_local_role_ids": candidate_local_role_ids,
                "candidate_local_owner_ids": candidate_local_owner_ids,
                "candidate_local_entity_ids": candidate_local_entity_ids,
                "candidate_local_zone_ids": candidate_local_zone_ids,
                "candidate_local_order_ids": candidate_local_order_ids,
                "action_mask": action_mask,
                "decision_domain": features["decision_domain"],
            }
        except Exception:
            self._clear_text_registry()
            raise
        finally:
            self._current_planner_context = None
            self._current_boss_context = None

    def _build_feature_view(
        self,
        obs: dict[str, Any],
        legal_actions: list[dict[str, Any]],
        planner_context: dict[str, Any],
    ) -> dict[str, np.ndarray]:
        scalars = np.zeros(obs_common.SCALAR_DIM, dtype=np.float32)
        decision_domain = np.zeros(obs_common.NUM_DOMAINS, dtype=np.float32)
        player_powers = np.zeros(obs_common.POWER_DIM, dtype=np.float32)
        relic_signals = np.zeros(obs_common.RELIC_SIGNAL_DIM, dtype=np.float32)
        run_memory = np.asarray(planner_context["run_memory_vector"], dtype=np.float32).copy()
        objective_context = np.asarray(planner_context["objective_context_vector"], dtype=np.float32).copy()

        if obs:
            self._enc_scalars(scalars, obs, legal_actions, planner_context)
            self._enc_decision_domain(decision_domain, obs)
            self._enc_powers(player_powers, obs)
            self._enc_relic_signals(relic_signals, obs)

        hand_cards = self._runtime_cards(obs, "hand", "hand")
        deck_cards = (obs.get("player") or {}).get("deck_cards") or []
        enemies = (obs.get("combat") or {}).get("enemies") or []
        relics = (obs.get("player") or {}).get("relics") or []
        potions = (obs.get("player") or {}).get("potions") or []

        hand, hand_text, hand_mask = self._encode_cards_with_limit(hand_cards, obs_common.MAX_HAND)
        deck, deck_text, deck_mask = self._encode_cards_with_limit(deck_cards, obs_common.MAX_DECK)
        enemies_numeric, enemy_text, enemy_mask = self._encode_enemy_view(enemies)
        relic_text, relic_mask = self._encode_support_text_view(relics, obs_common.MAX_RELICS, "relics")
        potion_text, potion_mask = self._encode_support_text_view(potions, obs_common.MAX_POTIONS, "potions")
        self._patch_body_slam_card_features(hand, hand_cards, obs)
        (
            actions,
            action_text,
            semantic_actions,
            semantic_action_text,
            route_summary,
            route_nodes,
            route_node_mask,
            action_mask,
        ) = self._encode_action_view(legal_actions, planner_context)
        self._patch_body_slam_action_features(actions, legal_actions, obs)

        return {
            "scalars": scalars,
            "decision_domain": decision_domain,
            "hand": hand,
            "hand_text": hand_text,
            "hand_mask": hand_mask,
            "deck": deck,
            "deck_text": deck_text,
            "deck_mask": deck_mask,
            "enemies": enemies_numeric,
            "enemy_text": enemy_text,
            "enemy_mask": enemy_mask,
            "player_powers": player_powers,
            "relic_signals": relic_signals,
            "run_memory": run_memory,
            "objective_context": objective_context,
            "relics": relic_text,
            "relic_mask": relic_mask,
            "potions": potion_text,
            "potion_mask": potion_mask,
            "actions": actions,
            "action_text": action_text,
            "semantic_actions": semantic_actions,
            "semantic_action_text": semantic_action_text,
            "route_summary": route_summary,
            "route_nodes": route_nodes,
            "route_node_mask": route_node_mask,
            "action_mask": action_mask,
        }

    def _encode_cards_with_limit(self, cards: list[Any], limit: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        count = min(len(cards) if isinstance(cards, list) else 0, limit)
        numeric = np.zeros((count, obs_common.CARD_FEAT_DIM), dtype=np.float32)
        text = np.zeros((count, TEXT_DIM), dtype=np.float32)
        mask = np.zeros(count, dtype=np.float32)
        if count > 0:
            self._enc_card_collection(cards[:count], numeric, text, mask)
        return numeric, text, mask

    def _encode_enemy_view(self, enemies: list[Any]) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        count = min(len(enemies) if isinstance(enemies, list) else 0, obs_common.MAX_ENEMIES)
        numeric = np.zeros((count, obs_common.ENEMY_FEAT_DIM), dtype=np.float32)
        text = np.zeros((count, TEXT_DIM), dtype=np.float32)
        mask = np.zeros(count, dtype=np.float32)
        if count > 0:
            self._enc_enemies(numeric, text, mask, {"combat": {"enemies": enemies[:count]}})
        return numeric, text, mask

    def _encode_support_text_view(
        self,
        entries: list[Any],
        limit: int,
        kind: str,
    ) -> tuple[np.ndarray, np.ndarray]:
        count = min(len(entries) if isinstance(entries, list) else 0, limit)
        text = np.zeros((count, TEXT_DIM), dtype=np.float32)
        mask = np.zeros(count, dtype=np.float32)
        if count > 0:
            payload = {"player": {kind: entries[:count]}}
            if kind == "relics":
                self._enc_relics(text, mask, payload)
            else:
                self._enc_potions(text, mask, payload)
        return text, mask

    def _encode_action_view(
        self,
        legal_actions: list[dict[str, Any]],
        planner_context: dict[str, Any],
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        count = min(len(legal_actions), MAX_ACTIONS)
        actions = np.zeros((count, obs_common.ACTION_FEAT_DIM), dtype=np.float32)
        action_text = np.zeros((count, TEXT_DIM), dtype=np.float32)
        semantic_actions = np.zeros((count, obs_common.SEM_ACTION_FEAT_DIM), dtype=np.float32)
        semantic_action_text = np.zeros((count, TEXT_DIM), dtype=np.float32)
        route_summary = np.zeros((count, obs_common.ROUTE_SUMMARY_DIM), dtype=np.float32)
        route_nodes = np.zeros((count, obs_common.MAX_ROUTE_NODES, obs_common.ROUTE_NODE_FEAT_DIM), dtype=np.float32)
        route_node_mask = np.zeros((count, obs_common.MAX_ROUTE_NODES), dtype=np.float32)
        action_mask = np.zeros(count, dtype=np.float32)
        if count > 0:
            self._enc_actions(
                actions,
                action_text,
                semantic_actions,
                semantic_action_text,
                route_summary,
                route_nodes,
                route_node_mask,
                action_mask,
                legal_actions[:count],
                planner_context,
            )
        return actions, action_text, semantic_actions, semantic_action_text, route_summary, route_nodes, route_node_mask, action_mask

    @staticmethod
    def _is_body_slam_card(source: dict[str, Any] | None) -> bool:
        """Return True for Body Slam / 全身撞击 payloads across bridge variants.

        Body Slam's play value is not a static card number: its damage equals
        the player's *current block*.  Several bridge/static payload variants
        only expose it as an Attack with zero damage, so the token encoder needs
        a narrow card-identity fallback rather than relying on generic regex
        text features during policy scoring.
        """
        if not isinstance(source, dict):
            return False
        id_values = []
        for key in ("id", "card_id", "model_id", "internal_id", "class_name", "type_name"):
            value = str(source.get(key) or "").strip()
            if value:
                id_values.append(value)
        id_blob = " ".join(id_values).lower().replace("-", "_").replace(".", "_").replace(" ", "_")
        if "body_slam" in id_blob or "bodyslam" in id_blob:
            return True
        if "card_body_slam" in id_blob:
            return True

        text_parts: list[str] = []
        for key in ("title", "name", "description", "text", "canonical_text"):
            value = str(source.get(key) or "").strip()
            if value:
                text_parts.append(value)
        text_blob = " | ".join(text_parts).lower()
        if "全身撞击" in text_blob or "全身撞擊" in text_blob:
            return True
        return ("body slam" in text_blob) or (
            ("current block" in text_blob or "当前格挡" in text_blob or "目前格挡" in text_blob)
            and ("damage" in text_blob or "伤害" in text_blob)
        )

    @staticmethod
    def _player_current_block(obs: dict[str, Any] | None) -> float:
        if not isinstance(obs, dict):
            return 0.0
        player = obs.get("player") if isinstance(obs.get("player"), dict) else {}
        combat = obs.get("combat") if isinstance(obs.get("combat"), dict) else {}
        candidates = (
            player.get("block") if isinstance(player, dict) else None,
            player.get("current_block") if isinstance(player, dict) else None,
            (player.get("creature") or {}).get("block") if isinstance(player.get("creature"), dict) else None,
            (combat.get("player") or {}).get("block") if isinstance(combat.get("player"), dict) else None,
            (combat.get("player") or {}).get("current_block") if isinstance(combat.get("player"), dict) else None,
        )
        for value in candidates:
            block = obs_common._float(value)
            if block > 0.0:
                return max(block, 0.0)
        return 0.0

    def _body_slam_dynamic_damage(self, source: dict[str, Any] | None, obs: dict[str, Any] | None) -> float:
        if not self._is_body_slam_card(source):
            return 0.0
        return self._player_current_block(obs)

    @staticmethod
    def _is_combat_play_action(action: dict[str, Any]) -> bool:
        kind = str(action.get("kind") or "").strip().lower()
        if kind == "play_card":
            return True
        semantic = action.get("semantic") if isinstance(action.get("semantic"), dict) else {}
        family = str(semantic.get("family") or semantic.get("kind") or "").strip().lower()
        action_id = str(action.get("action_id") or "").strip().lower()
        return (
            kind == "combat"
            and ("play" in family or "card" in family or "play_card" in action_id or "play:" in action_id)
        )

    def _patch_body_slam_card_features(
        self,
        rows: np.ndarray,
        cards: list[Any],
        obs: dict[str, Any],
    ) -> None:
        """Patch live hand-card numeric rows for Body Slam dynamic damage.

        CARD_FEAT_DIM columns mirror DenseObservationEncoder._enc_card_collection:
        12/14 are base/preview damage, 25 is hit count, 31/32 are per-hit and
        per-energy damage, 34 is damage delta.  We do not change shape/schema.

        Bridge/static payloads may expose Body Slam either as zero damage or as
        a stale low non-zero preview.  The live tactical value is current block,
        so always raise the encoded damage columns to at least that value rather
        than only filling missing zeros.
        """
        if rows.size == 0 or not isinstance(cards, list):
            return
        for index, card in enumerate(cards[: rows.shape[0]]):
            if not isinstance(card, dict):
                continue
            damage = self._body_slam_dynamic_damage(card, obs)
            if damage <= 0.0:
                continue
            cost = max(obs_common._runtime_spend_cost(card), 0.0)
            damage_per_energy = damage / obs_common._normalized_cost_for_efficiency(cost)
            if rows.shape[1] > 14:
                rows[index, 12] = max(rows[index, 12], obs_common._log_norm(damage, obs_common._LOG1P_200))
                rows[index, 14] = max(rows[index, 14], obs_common._log_norm(damage, obs_common._LOG1P_200))
            if rows.shape[1] > 25:
                rows[index, 25] = max(rows[index, 25], min(1.0 / 10.0, 1.0))
            if rows.shape[1] > 32:
                rows[index, 31] = max(rows[index, 31], obs_common._log_norm(damage, obs_common._LOG1P_100))
                rows[index, 32] = max(rows[index, 32], obs_common._log_norm(damage_per_energy, obs_common._LOG1P_100))
            if rows.shape[1] > 34:
                rows[index, 34] = max(rows[index, 34], 0.0)

    def _patch_body_slam_action_features(
        self,
        actions: np.ndarray,
        legal_actions: list[dict[str, Any]],
        obs: dict[str, Any],
    ) -> None:
        """Patch play-card action rows for Body Slam dynamic damage = block.

        This is intentionally observation-side as well as bridge-side: old
        snapshots and stale bridge DLLs can still produce Body Slam with
        effect_preview.damage=0.  The policy/action scorer must see the live
        damage columns as at least current block.  Do this even when the bridge
        exposes a stale low non-zero preview: Body Slam's damage scales from
        live block, so a static 1-3 damage preview is still underexposed.
        """
        if actions.size == 0:
            return
        for index, action in enumerate(legal_actions[: actions.shape[0]]):
            if not isinstance(action, dict) or not self._is_combat_play_action(action):
                continue
            card = action.get("card") if isinstance(action.get("card"), dict) else None
            damage = self._body_slam_dynamic_damage(card, obs)
            if damage <= 0.0:
                continue
            row = actions[index]
            cost = max(obs_common._runtime_spend_cost(card), 0.0)
            damage_per_energy = damage / obs_common._normalized_cost_for_efficiency(cost)
            if row.shape[0] > 22:
                row[22] = max(row[22], obs_common._log_norm(damage, obs_common._LOG1P_200))
            if row.shape[0] > 38:
                row[38] = 1.0
            if row.shape[0] > 40:
                row[40] = max(row[40], obs_common._log_norm(damage, obs_common._LOG1P_200))
            if row.shape[0] > 42:
                row[42] = max(row[42], obs_common._log_norm(damage, obs_common._LOG1P_100))
            if row.shape[0] > 43:
                row[43] = max(row[43], obs_common._log_norm(damage_per_energy, obs_common._LOG1P_100))

    def _resolve_entry_text_embeddings(self, *entry_groups: list[Any]) -> None:
        def _iter_entries(group: list[Any]) -> Any:
            for item in group:
                if isinstance(item, list):
                    for nested in item:
                        if isinstance(nested, dict):
                            yield nested
                elif isinstance(item, dict):
                    yield item

        for group in entry_groups:
            for entry in _iter_entries(group):
                text_embedding = entry.get("text_embedding")
                if text_embedding is not None:
                    entry["text_embedding"] = _compress_text_embedding(text_embedding)
                    entry["text"] = ""
                    continue

                raw_text = str(entry.get("text") or "").strip()
                if not self.use_text or not raw_text:
                    entry["text_embedding"] = np.zeros(TOKEN_TEXT_DIM, dtype=np.float32)
                    entry["text"] = ""
                    continue

                self._register_text_assignment(
                    raw_text,
                    lambda embedding, entry=entry: entry.update(text_embedding=np.asarray(embedding, dtype=np.float32), text=""),
                    postprocess=_compress_text_embedding,
                )


ObservationEncoderV3 = WorldTokenObservationEncoder

__all__ = [
    'DEFAULT_EFFECT_PROFILE',
    'ENTITY_HASH_BUCKETS',
    'KEY_POWER_CARD_BUCKETS',
    'MAX_ACTIONS',
    'MAX_CANDIDATE_LOCAL_TOKENS',
    'MAX_CARD_KEYWORD_SLOTS',
    'MAX_HISTORY_TOKENS',
    'MAX_ORDER_ID',
    'MAX_OWNER_ID',
    'MAX_POWER_SLOT_TOKENS',
    'MAX_ROLE_ID',
    'MAX_STEP_DETAIL_TOKENS',
    'MAX_TURN_SUMMARY_TOKENS',
    'MAX_WORLD_TOKENS',
    'MAX_ZONE_ID',
    'NUM_KEY_POWER_FLAGS',
    'NUM_SEMANTIC_ROLES',
    'NUM_TOKEN_TYPES',
    'OBSERVATION_API_VERSION',
    'OWNER_DECK',
    'OWNER_DISCARD',
    'OWNER_DRAW',
    'OWNER_ENEMY_BASE',
    'OWNER_EXHAUST',
    'OWNER_HAND',
    'OWNER_HISTORY',
    'OWNER_NONE',
    'OWNER_PLAY',
    'OWNER_PLAYER',
    'OWNER_POTION',
    'OWNER_POWER',
    'OWNER_RELIC',
    'OWNER_REWARD',
    'OWNER_ROUTE',
    'OWNER_SHOP',
    'OWNER_UPGRADE',
    'POWER_ID_BUCKETS',
    'TEXT_DIM',
    'TOKEN_FEAT_DIM',
    'TOKEN_NUMERIC_DIM',
    'TOKEN_ROLES',
    'TOKEN_ROLE_TO_ID',
    'TOKEN_TEXT_DIM',
    'TOKEN_TYPES',
    'TOKEN_TYPE_TO_ID',
    'TOKEN_ZONES',
    'TOKEN_ZONE_TO_ID',
    '_CARD_KEYWORD_BUCKETS',
    'ObservationEncoderV3',
    'WorldTokenObservationEncoder',
    '_resolve_potion_effect',
    'build_boss_mechanics_context',
    'build_live_enemy_semantic_text',
    'build_live_potion_semantic_text',
    'build_live_relic_semantic_text',
    'enemy_mechanics_key',
    'get_enemy_metadata',
    'infer_hand_mutation',
    'mutation_summary_numeric',
    'mutation_target_numeric',
    'post_hand_preview_numeric',
]
