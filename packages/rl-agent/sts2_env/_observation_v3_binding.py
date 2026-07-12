"""Token materialization, boss traits, and action/entity binding helpers."""

from __future__ import annotations

from typing import Any

import numpy as np

from content_registry import build_live_enemy_semantic_text, get_enemy_metadata

from . import observation_common as obs_common
from ._observation_v3_schema import (
    ENTITY_HASH_BUCKETS,
    MAX_ORDER_ID,
    MAX_OWNER_ID,
    MAX_ROLE_ID,
    MAX_ZONE_ID,
    OWNER_DECK,
    OWNER_DISCARD,
    OWNER_DRAW,
    OWNER_ENEMY_BASE,
    OWNER_EXHAUST,
    OWNER_HAND,
    OWNER_NONE,
    OWNER_PLAY,
    OWNER_POTION,
    OWNER_REWARD,
    OWNER_ROUTE,
    OWNER_UPGRADE,
    TOKEN_FEAT_DIM,
    TOKEN_NUMERIC_DIM,
    TOKEN_TYPE_TO_ID,
    TOKEN_ZONE_TO_ID,
    _compress_text_embedding,
    _role_for_token,
    _zone_for_token,
)
from .boss_mechanics import enemy_mechanics_key


class TokenBindingMaterializationMixin:
    def _materialize_entries(
        self,
        entries: list[dict[str, Any]],
        max_count: int,
        bufs: dict[str, np.ndarray] | None = None,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        if bufs is not None:
            token_array, token_mask = bufs["tokens"], bufs["mask"]
            type_ids, role_ids = bufs["type_ids"], bufs["role_ids"]
            owner_ids, entity_ids = bufs["owner_ids"], bufs["entity_ids"]
            zone_ids, order_ids = bufs["zone_ids"], bufs["order_ids"]
            target_owner_ids, target_entity_ids = bufs["target_owner_ids"], bufs["target_entity_ids"]
            for arr in (token_array, token_mask, type_ids, role_ids, owner_ids, entity_ids, zone_ids, order_ids, target_owner_ids, target_entity_ids):
                arr[:] = 0
        else:
            token_array = np.zeros((max_count, TOKEN_FEAT_DIM), dtype=np.float32)
            token_mask = np.zeros(max_count, dtype=np.float32)
            type_ids = np.zeros(max_count, dtype=np.int32)
            role_ids = np.zeros(max_count, dtype=np.int32)
            owner_ids = np.zeros(max_count, dtype=np.int32)
            entity_ids = np.zeros(max_count, dtype=np.int32)
            zone_ids = np.zeros(max_count, dtype=np.int32)
            order_ids = np.zeros(max_count, dtype=np.int32)
            target_owner_ids = np.zeros(max_count, dtype=np.int32)
            target_entity_ids = np.zeros(max_count, dtype=np.int32)

        for index, entry in enumerate(entries[:max_count]):
            token_array[index, :TOKEN_NUMERIC_DIM] = entry["numeric"][:TOKEN_NUMERIC_DIM]
            token_array[index, TOKEN_NUMERIC_DIM:] = entry["text_embedding"]
            token_mask[index] = 1.0
            type_ids[index] = int(entry["type_id"])
            role_ids[index] = int(entry["role_id"])
            owner_ids[index] = int(entry["owner_id"])
            entity_ids[index] = int(entry["entity_id"])
            zone_ids[index] = int(entry["zone_id"])
            order_ids[index] = int(entry["order_id"])
            target_owner_ids[index] = int(entry["target_owner_id"])
            target_entity_ids[index] = int(entry["target_entity_id"])
        return token_array, token_mask, type_ids, role_ids, owner_ids, entity_ids, zone_ids, order_ids, target_owner_ids, target_entity_ids

    def _materialize_nested_entries(
        self,
        entries: list[list[dict[str, Any]]],
        max_outer: int,
        max_inner: int,
        bufs: dict[str, np.ndarray] | None = None,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        if bufs is not None:
            token_array, token_mask = bufs["tokens"], bufs["mask"]
            type_ids, role_ids = bufs["type_ids"], bufs["role_ids"]
            owner_ids, entity_ids = bufs["owner_ids"], bufs["entity_ids"]
            zone_ids, order_ids = bufs["zone_ids"], bufs["order_ids"]
            for arr in (token_array, token_mask, type_ids, role_ids, owner_ids, entity_ids, zone_ids, order_ids):
                arr[:] = 0
        else:
            token_array = np.zeros((max_outer, max_inner, TOKEN_FEAT_DIM), dtype=np.float32)
            token_mask = np.zeros((max_outer, max_inner), dtype=np.float32)
            type_ids = np.zeros((max_outer, max_inner), dtype=np.int32)
            role_ids = np.zeros((max_outer, max_inner), dtype=np.int32)
            owner_ids = np.zeros((max_outer, max_inner), dtype=np.int32)
            entity_ids = np.zeros((max_outer, max_inner), dtype=np.int32)
            zone_ids = np.zeros((max_outer, max_inner), dtype=np.int32)
            order_ids = np.zeros((max_outer, max_inner), dtype=np.int32)
        for outer_index in range(min(len(entries), max_outer)):
            for inner_index, entry in enumerate(entries[outer_index][:max_inner]):
                token_array[outer_index, inner_index, :TOKEN_NUMERIC_DIM] = entry["numeric"][:TOKEN_NUMERIC_DIM]
                token_array[outer_index, inner_index, TOKEN_NUMERIC_DIM:] = entry["text_embedding"]
                token_mask[outer_index, inner_index] = 1.0
                type_ids[outer_index, inner_index] = int(entry["type_id"])
                role_ids[outer_index, inner_index] = int(entry["role_id"])
                owner_ids[outer_index, inner_index] = int(entry["owner_id"])
                entity_ids[outer_index, inner_index] = int(entry["entity_id"])
                zone_ids[outer_index, inner_index] = int(entry["zone_id"])
                order_ids[outer_index, inner_index] = int(entry["order_id"])
        return token_array, token_mask, type_ids, role_ids, owner_ids, entity_ids, zone_ids, order_ids

    def _entry(
        self,
        token_type: str,
        numeric: np.ndarray,
        *,
        role_id: int | None = None,
        zone_id: int | None = None,
        order_id: int = 0,
        owner_id: int,
        entity_id: int,
        target_owner_id: int = 0,
        target_entity_id: int = 0,
        text: str | None = None,
        text_embedding: np.ndarray | None = None,
    ) -> dict[str, Any]:
        if text_embedding is not None:
            text_embedding = _compress_text_embedding(text_embedding)
        return {
            "type_id": TOKEN_TYPE_TO_ID[token_type],
            "role_id": int(_role_for_token(token_type) if role_id is None else max(0, min(role_id, MAX_ROLE_ID))),
            "zone_id": int(_zone_for_token(token_type) if zone_id is None else max(0, min(zone_id, MAX_ZONE_ID))),
            "order_id": int(max(0, min(order_id, MAX_ORDER_ID))),
            "numeric": np.asarray(numeric, dtype=np.float32).reshape(-1),
            "owner_id": int(max(0, min(owner_id, MAX_OWNER_ID))),
            "entity_id": int(max(0, min(entity_id, ENTITY_HASH_BUCKETS - 1))),
            "target_owner_id": int(max(0, min(target_owner_id, MAX_OWNER_ID))),
            "target_entity_id": int(max(0, min(target_entity_id, ENTITY_HASH_BUCKETS - 1))),
            "text_embedding": text_embedding,
            "text": str(text or "").strip(),
        }

    def _boss_player_state(self) -> dict[str, float]:
        context = self._current_boss_context
        if isinstance(context, dict):
            player_state = context.get("player_state")
            if isinstance(player_state, dict):
                return player_state
        return {}

    def _boss_enemy_state(self, enemy: dict[str, Any] | None) -> dict[str, float]:
        if not isinstance(enemy, dict):
            return {}
        context = self._current_boss_context
        if not isinstance(context, dict):
            return {}
        enemy_states_by_key = context.get("enemy_states_by_key")
        if not isinstance(enemy_states_by_key, dict):
            return {}
        state = enemy_states_by_key.get(enemy_mechanics_key(enemy, ""))
        return state if isinstance(state, dict) else {}

    def _boss_enemy_traits(self, enemy: dict[str, Any] | None) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
        if not isinstance(enemy, dict):
            return [], []
        context = self._current_boss_context
        if not isinstance(context, dict):
            return [], []
        enemy_traits_by_key = context.get("enemy_traits_by_key")
        if not isinstance(enemy_traits_by_key, dict):
            return [], []
        payload = enemy_traits_by_key.get(enemy_mechanics_key(enemy, ""))
        if not isinstance(payload, dict):
            return [], []
        reactive_traits = payload.get("reactive_traits")
        phase_rules = payload.get("phase_rules")
        return (
            reactive_traits if isinstance(reactive_traits, list) else [],
            phase_rules if isinstance(phase_rules, list) else [],
        )

    def _infer_enemy_traits(self, enemy: dict[str, Any]) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
        reactive_traits: list[dict[str, Any]] = []
        phase_rules: list[dict[str, Any]] = []
        seen: set[tuple[str, str, str, str]] = set()

        def _classify(entry: dict[str, Any]) -> list[dict[str, Any]]:
            category = str(entry.get("category") or "").lower()
            text = " ".join(
                str(entry.get(field) or "")
                for field in ("trait", "description", "effect_type", "condition", "state")
            ).lower()
            if category == "phase" or any(keyword in text for keyword in ("phase", "split", "threshold", "stun", "intangible")):
                return phase_rules
            return reactive_traits

        def _add(entry: Any) -> None:
            if not isinstance(entry, dict):
                return
            key = (
                str(entry.get("category") or ""),
                str(entry.get("trait") or entry.get("effect_type") or ""),
                str(entry.get("trigger_type") or ""),
                str(entry.get("condition") or entry.get("state") or entry.get("description") or ""),
            )
            if key in seen:
                return
            seen.add(key)
            _classify(entry).append(entry)

        metadata = get_enemy_metadata(str(enemy.get("model_id") or "").strip())
        if isinstance(metadata, dict):
            for collection_name in ("static_traits", "reactive_triggers", "phase_rules", "trait_tokens"):
                for item in metadata.get(collection_name) or []:
                    _add(item)

        for collection_name in ("static_traits", "reactive_triggers", "phase_rules"):
            for item in enemy.get(collection_name) or []:
                _add(item)

        name_text = " ".join(
            part for part in (
                str(enemy.get("name") or "").strip(),
                str(enemy.get("model_id") or "").strip(),
                str((enemy.get("intent") or {}).get("description") or "").strip(),
            ) if part
        ).lower()
        if "split" in name_text:
            _add({"category": "phase", "trait": "split_on_threshold", "description": "split_on_threshold"})
        if "phase" in name_text or "threshold" in name_text:
            _add({"category": "phase", "trait": "hp_threshold_phase_shift", "description": "hp_threshold_phase_shift"})
        for power in enemy.get("powers") or []:
            if not isinstance(power, dict):
                continue
            title = str(power.get("title") or "").lower()
            if "thorn" in title or "spike" in title:
                _add({"category": "reactive", "trait": "thorns", "effect_amount": power.get("amount"), "description": power.get("title")})
            if "retali" in title or "contact" in title:
                _add({"category": "reactive", "trait": "contact_retaliate", "effect_amount": power.get("amount"), "description": power.get("title")})
            if "intang" in title:
                _add({"category": "phase", "trait": "gain_intangible", "description": power.get("title")})
        boss_reactive_traits, boss_phase_rules = self._boss_enemy_traits(enemy)
        for item in boss_reactive_traits:
            _add(item)
        for item in boss_phase_rules:
            _add(item)
        return reactive_traits, phase_rules

    def _enemy_reaction_flags(self, enemy: dict[str, Any] | None) -> dict[str, float]:
        if not isinstance(enemy, dict):
            return {
                "thorns": 0.0,
                "contact_punish": 0.0,
                "split": 0.0,
                "threshold": 0.0,
                "threshold_value": 0.0,
                "artifact": 0.0,
                "buffer": 0.0,
                "intangible": 0.0,
                "incoming_damage_multiplier": 0.0,
                "back_attack": 0.0,
                "damage_cap": 0.0,
                "damage_cap_value": 0.0,
                "deathburst": 0.0,
                "deathburst_damage": 0.0,
                "revive": 0.0,
                "transform": 0.0,
                "linked_support_alive": 0.0,
                "special_phase": 0.0,
                "one_card_lock": 0.0,
                "skill_punish": 0.0,
                "choice_debuffs": 0.0,
                "escape_card_tax": 0.0,
                "countdown": 0.0,
                "stun_window": 0.0,
                "binding_control": 0.0,
                "wound_phase": 0.0,
            }

        reactive_traits, phase_rules = self._infer_enemy_traits(enemy)
        boss_state = self._boss_enemy_state(enemy)
        texts: list[str] = [build_live_enemy_semantic_text(enemy).lower()]
        threshold_value = 0.0
        for collection_name in ("powers", "static_traits", "reactive_triggers", "phase_rules"):
            collection = enemy.get(collection_name)
            if not isinstance(collection, list):
                continue
            for item in collection:
                if not isinstance(item, dict):
                    continue
                texts.extend(
                    str(item.get(key) or "").strip().lower()
                    for key in ("title", "description", "trait", "effect_type", "condition", "state")
                    if str(item.get(key) or "").strip()
                )
                threshold_value = max(
                    threshold_value,
                    abs(obs_common._float(item.get("threshold"))),
                    abs(obs_common._float(item.get("amount"))),
                    abs(obs_common._float(item.get("effect_amount"))),
                )
        for item in reactive_traits + phase_rules:
            threshold_value = max(
                threshold_value,
                abs(obs_common._float(item.get("threshold"))),
                abs(obs_common._float(item.get("amount"))),
                abs(obs_common._float(item.get("effect_amount"))),
            )

        joined = " | ".join(texts)
        return {
            "thorns": float(any(keyword in joined for keyword in ("thorn", "spike"))),
            "contact_punish": float(any(keyword in joined for keyword in ("thorn", "spike", "retaliat", "contact", "punish"))),
            "split": float(any(keyword in joined for keyword in ("split",))),
            "threshold": float(any(keyword in joined for keyword in ("threshold", "phase", "stun", "hp_le"))),
            "threshold_value": min(threshold_value / 100.0, 1.0),
            "artifact": float("artifact" in joined),
            "buffer": float("buffer" in joined),
            "intangible": float("intang" in joined),
            "incoming_damage_multiplier": float(boss_state.get("incoming_damage_multiplier_norm", 0.0)),
            "back_attack": float(boss_state.get("back_attack_active", 0.0)),
            "damage_cap": float(boss_state.get("damage_cap_active", 0.0)),
            "damage_cap_value": float(boss_state.get("damage_cap_value_norm", 0.0)),
            "deathburst": float(boss_state.get("deathburst", 0.0)),
            "deathburst_damage": float(boss_state.get("deathburst_damage_norm", 0.0)),
            "revive": float(boss_state.get("revive_once", 0.0)),
            "transform": float(boss_state.get("transform_pending", 0.0)),
            "linked_support_alive": float(boss_state.get("linked_support_alive", 0.0)),
            "special_phase": float(boss_state.get("special_phase_active", 0.0)),
            "one_card_lock": float(boss_state.get("one_card_lock", 0.0)),
            "skill_punish": float(boss_state.get("skill_punish", 0.0)),
            "choice_debuffs": float(boss_state.get("choice_debuffs", 0.0)),
            "escape_card_tax": float(boss_state.get("escape_card_tax", 0.0)),
            "countdown": float(boss_state.get("countdown_active", 0.0)),
            "stun_window": float(boss_state.get("stun_window", 0.0)),
            "binding_control": float(boss_state.get("binding_control", 0.0)),
            "wound_phase": float(boss_state.get("wound_phase", 0.0)),
        }

    def _trait_numeric(self, trait: dict[str, Any]) -> np.ndarray:
        numeric = np.zeros(TOKEN_NUMERIC_DIM, dtype=np.float32)
        text = " | ".join(
            str(trait.get(key) or "").lower()
            for key in ("trait", "effect_type", "description", "condition", "state")
            if str(trait.get(key) or "").strip()
        )
        numeric[0] = float("thorn" in text)
        numeric[1] = float("contact" in text or "retali" in text)
        numeric[2] = float("split" in text)
        numeric[3] = float("phase" in text)
        numeric[4] = float("threshold" in text or "stun" in text)
        numeric[5] = min(abs(obs_common._float(trait.get("effect_amount") or trait.get("amount") or trait.get("threshold"))) / 10.0, 1.0)
        numeric[6] = float("back_attack" in text or "matched_facing" in text or "damage_multiplier" in text)
        numeric[7] = float("damage_cap" in text or "slippery" in text or "opening_cycle" in text)
        numeric[8] = float("death_explosion" in text or "deathburst" in text or "steam eruption" in text)
        numeric[9] = float("revive" in text or "reborn" in text or "resurrect" in text)
        numeric[10] = float("binding" in text or "choice_debuffs" in text)
        numeric[11] = float("linked_support" in text or "support_body" in text or "door_destroyed" in text)
        numeric[12] = float("countdown" in text or "doom_clock" in text or "escape_card_tax" in text)
        numeric[13] = float("skill_punish" in text or "on_play_skill" in text)
        numeric[14] = float("intang" in text)
        numeric[15] = float("one_card_lock" in text or "play_budget_locked" in text)
        numeric[16] = float("wound_phase" in text or "chip_damage" in text)
        numeric[17] = float("phase_three" in text or "opening" in text or "phase_two" in text)
        return numeric

    @staticmethod
    def _is_selection_action(action: dict[str, Any]) -> bool:
        kind = str(action.get("kind") or "").strip()
        action_id = str(action.get("action_id") or "").strip()
        return kind in {"card_selection", "combat_select_card", "combat_select"} or action_id.startswith("combat_select")

    @staticmethod
    def _selection_semantics_text(action: dict[str, Any]) -> str:
        return " | ".join(
            part
            for part in (
                str(action.get("selection_semantics") or "").strip(),
                str(action.get("selection_prompt") or "").strip(),
                str(action.get("surface") or "").strip(),
            )
            if part
        )

    def _infer_action_domain(self, action: dict[str, Any], obs: dict[str, Any]) -> str:
        kind = str(action.get("kind") or "").strip()
        if self._is_selection_action(action):
            return "selection"
        if kind in {"play_card", "use_potion", "discard_potion", "combat", "proceed"} and obs.get("combat"):
            return "combat"
        if kind == "map" or isinstance(action.get("route_summary"), dict):
            return "route"
        if (obs.get("phase") or "") == "map":
            return "route"
        return "build"

    def _infer_card_selection_source_pile(self, action: dict[str, Any], obs: dict[str, Any]) -> tuple[str, int]:
        card = action.get("card") if isinstance(action.get("card"), dict) else None
        if card is None:
            return "none", 0

        pile_candidates = [
            ("hand", self._runtime_cards(obs, "hand", "hand")),
            ("draw", self._runtime_cards(obs, "draw_pile", "draw_preview_cards")),
            ("discard", self._runtime_cards(obs, "discard_pile", "discard_cards")),
            ("exhaust", self._runtime_cards(obs, "exhaust_pile", "exhaust_cards")),
            ("play", self._runtime_cards(obs, "play_pile", "play_pile_cards")),
            ("deck", (obs.get("player") or {}).get("deck_cards") if isinstance(obs.get("player"), dict) else []),
        ]

        matches: list[tuple[str, int]] = []
        for pile_name, cards in pile_candidates:
            position = self._find_source_position(cards if isinstance(cards, list) else [], card)
            if position > 0:
                matches.append((pile_name, position))
        if len(matches) == 1:
            return matches[0]

        prompt = " ".join(
            str(value or "")
            for value in (
                action.get("selection_semantics"),
                action.get("selection_prompt"),
                action.get("label"),
                action.get("screen_type"),
            )
        ).lower()
        keyword_map = (
            ("discard", ("discard",)),
            ("draw", ("draw pile", "draw")),
            ("exhaust", ("exhaust",)),
            ("hand", ("hand",)),
            ("play", ("play pile", "played")),
            ("deck", ("deck",)),
        )
        for pile_name, keywords in keyword_map:
            if any(keyword in prompt for keyword in keywords):
                for matched_name, matched_pos in matches:
                    if matched_name == pile_name:
                        return matched_name, matched_pos
                return pile_name, 0

        if matches:
            return matches[0]
        return "unknown", 0

    @staticmethod
    def _owner_zone_from_source_pile(source_pile: str) -> tuple[int, int]:
        return {
            "hand": (OWNER_HAND, TOKEN_ZONE_TO_ID["HAND"]),
            "draw": (OWNER_DRAW, TOKEN_ZONE_TO_ID["DRAW"]),
            "discard": (OWNER_DISCARD, TOKEN_ZONE_TO_ID["DISCARD"]),
            "exhaust": (OWNER_EXHAUST, TOKEN_ZONE_TO_ID["EXHAUST"]),
            "play": (OWNER_PLAY, TOKEN_ZONE_TO_ID["PLAY"]),
            "deck": (OWNER_DECK, TOKEN_ZONE_TO_ID["DECK"]),
            "reward": (OWNER_REWARD, TOKEN_ZONE_TO_ID["REWARD"]),
            "upgrade": (OWNER_UPGRADE, TOKEN_ZONE_TO_ID["UPGRADE"]),
            "potion": (OWNER_POTION, TOKEN_ZONE_TO_ID["POTION"]),
            "route": (OWNER_ROUTE, TOKEN_ZONE_TO_ID["ROUTE"]),
            "selection": (OWNER_NONE, TOKEN_ZONE_TO_ID["SELECTION"]),
        }.get(source_pile, (OWNER_NONE, TOKEN_ZONE_TO_ID["NONE"]))

    def _action_source_binding(self, action: dict[str, Any], obs: dict[str, Any], action_index: int) -> tuple[int, int, int]:
        kind = str(action.get("kind") or "").strip()
        if kind == "play_card" and isinstance(action.get("card"), dict):
            return (
                OWNER_HAND,
                TOKEN_ZONE_TO_ID["HAND"],
                self._find_source_position(self._runtime_cards(obs, "hand", "hand"), action.get("card")),
            )
        if kind in {"use_potion", "discard_potion"} and isinstance(action.get("potion"), dict):
            player = obs.get("player") or {}
            return (
                OWNER_POTION,
                TOKEN_ZONE_TO_ID["POTION"],
                self._find_source_position(player.get("potions") if isinstance(player.get("potions"), list) else [], action.get("potion")),
            )
        if kind == "card_reward":
            return OWNER_REWARD, TOKEN_ZONE_TO_ID["REWARD"], min(action_index + 1, MAX_ORDER_ID)
        if kind == "deck_upgrade":
            return OWNER_UPGRADE, TOKEN_ZONE_TO_ID["UPGRADE"], min(action_index + 1, MAX_ORDER_ID)
        if kind == "map":
            return OWNER_ROUTE, TOKEN_ZONE_TO_ID["ROUTE"], min(action_index + 1, MAX_ORDER_ID)
        if self._is_selection_action(action) and isinstance(action.get("card"), dict):
            source_pile, position = self._infer_card_selection_source_pile(action, obs)
            owner_id, zone_id = self._owner_zone_from_source_pile(source_pile)
            if zone_id == TOKEN_ZONE_TO_ID["NONE"]:
                zone_id = TOKEN_ZONE_TO_ID["SELECTION"]
            return owner_id, zone_id, position or min(action_index + 1, MAX_ORDER_ID)
        if self._is_selection_action(action):
            return OWNER_NONE, TOKEN_ZONE_TO_ID["SELECTION"], min(action_index + 1, MAX_ORDER_ID)
        return OWNER_NONE, TOKEN_ZONE_TO_ID["NONE"], min(action_index + 1, MAX_ORDER_ID)

    def _action_entity_key(self, action: dict[str, Any]) -> str:
        for source in (
            action.get("card"),
            action.get("potion"),
            action.get("upgrade_preview"),
            (action.get("item") or {}).get("card") if isinstance(action.get("item"), dict) else None,
            (action.get("item") or {}).get("relic") if isinstance(action.get("item"), dict) else None,
            (action.get("item") or {}).get("potion") if isinstance(action.get("item"), dict) else None,
            (action.get("reward") or {}).get("card") if isinstance(action.get("reward"), dict) else None,
            (action.get("reward") or {}).get("relic") if isinstance(action.get("reward"), dict) else None,
            (action.get("reward") or {}).get("potion") if isinstance(action.get("reward"), dict) else None,
        ):
            if isinstance(source, dict):
                return str(source.get("id") or source.get("model_id") or source.get("title") or source.get("name") or action.get("action_id") or "")
        return str(action.get("action_id") or action.get("kind") or "")

    @staticmethod
    def _enemy_entity_key(enemy: dict[str, Any] | None, fallback: Any) -> str:
        if isinstance(enemy, dict):
            for key in ("combat_id", "id", "model_id", "name"):
                value = enemy.get(key)
                if value not in (None, ""):
                    return str(value)
        return str(fallback)

    def _query_zone_id(self, action: dict[str, Any], obs: dict[str, Any], query_type: str, target_enemy_index: int | None) -> int:
        kind = str(action.get("kind") or "").strip()
        if kind == "play_card":
            return TOKEN_ZONE_TO_ID["HAND"]
        if kind in {"use_potion", "discard_potion"}:
            return TOKEN_ZONE_TO_ID["POTION"]
        if self._is_selection_action(action):
            source_pile, _position = self._infer_card_selection_source_pile(action, obs)
            _owner_id, zone_id = self._owner_zone_from_source_pile(source_pile)
            if zone_id != TOKEN_ZONE_TO_ID["NONE"]:
                return zone_id
            return TOKEN_ZONE_TO_ID["SELECTION"]
        if kind == "map" or query_type == "ROUTE_CANDIDATE":
            return TOKEN_ZONE_TO_ID["ROUTE"]
        if kind == "shop":
            return TOKEN_ZONE_TO_ID["SHOP"]
        if kind == "deck_upgrade":
            return TOKEN_ZONE_TO_ID["UPGRADE"]
        if kind in {"reward", "card_reward", "event_option", "treasure_relic"}:
            return TOKEN_ZONE_TO_ID["REWARD"]
        if target_enemy_index is not None:
            return TOKEN_ZONE_TO_ID["ENEMY"]
        return _zone_for_token(query_type)

    def _source_order_id(self, action: dict[str, Any], obs: dict[str, Any], action_index: int) -> int:
        _owner_id, _zone_id, order_id = self._action_source_binding(action, obs, action_index)
        return order_id

    @staticmethod
    def _find_source_position(entries: list[Any], source: dict[str, Any] | None) -> int:
        if not isinstance(entries, list) or not isinstance(source, dict):
            return 0
        source_id = str(source.get("id") or "").strip()
        source_title = str(source.get("title") or "").strip().lower()
        for index, entry in enumerate(entries[:MAX_ORDER_ID]):
            if isinstance(entry, dict):
                entry_id = str(entry.get("id") or "").strip()
                entry_title = str(entry.get("title") or "").strip().lower()
                if source_id and entry_id and source_id == entry_id:
                    return index + 1
                if source_title and entry_title and source_title == entry_title:
                    return index + 1
            elif isinstance(entry, str):
                entry_text = entry.strip().lower()
                if source_id and entry_text == source_id.lower():
                    return index + 1
                if source_title and entry_text == source_title:
                    return index + 1
        return 0

    def _match_target_enemy(self, action: dict[str, Any], combat: dict[str, Any]) -> tuple[int | None, dict[str, Any] | None]:
        target = action.get("target") if isinstance(action.get("target"), dict) else None
        enemies = combat.get("enemies") or []
        if not isinstance(target, dict):
            return None, None
        target_combat_id = target.get("combat_id")
        target_name = str(target.get("name") or "").strip().lower()
        for index, enemy in enumerate(enemies[: obs_common.MAX_ENEMIES]):
            if not isinstance(enemy, dict):
                continue
            if target_combat_id is not None and enemy.get("combat_id") == target_combat_id:
                return index, enemy
            if target_name and str(enemy.get("name") or "").strip().lower() == target_name:
                return index, enemy
        return None, None

    def _infer_action_owner(self, action: dict[str, Any], target_enemy_index: int | None) -> int:
        if self._is_selection_action(action) and isinstance(action.get("card"), dict):
            source_pile, _position = self._infer_card_selection_source_pile(action, {})
            owner_id, _zone_id = self._owner_zone_from_source_pile(source_pile)
            if owner_id != OWNER_NONE:
                return owner_id
        owner_id, _zone_id, _order_id = self._action_source_binding(action, {}, 0)
        if owner_id != OWNER_NONE:
            return owner_id
        if target_enemy_index is not None:
            return self._enemy_owner_id(target_enemy_index)
        return OWNER_NONE

    def _enemy_owner_id(self, enemy_index: int | None) -> int:
        if enemy_index is None:
            return OWNER_NONE
        return int(max(OWNER_ENEMY_BASE, min(OWNER_ENEMY_BASE + enemy_index, MAX_OWNER_ID)))
