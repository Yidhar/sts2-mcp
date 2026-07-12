"""World, entity, power, keyword, and history token emitters."""

from __future__ import annotations

from typing import Any

import numpy as np

from . import observation_common as obs_common
from ._observation_v3_schema import (
    _CARD_KEYWORD_BUCKETS,
    _POWER_ALGEBRA_DIM,
    MAX_CARD_KEYWORD_SLOTS,
    MAX_ORDER_ID,
    MAX_OWNER_ID,
    MAX_POWER_SLOT_TOKENS,
    OWNER_HAND,
    OWNER_HISTORY,
    OWNER_NONE,
    OWNER_PLAYER,
    TOKEN_NUMERIC_DIM,
    TOKEN_TYPE_TO_ID,
    TOKEN_ZONE_TO_ID,
    _power_algebra,
    _power_id_bucket,
    _stable_bucket,
)
from .action_history import (
    MAX_STEP_DETAIL_TOKENS,
    MAX_TURN_SUMMARY_TOKENS,
    NUM_KEY_POWER_FLAGS,
    NUM_SEMANTIC_ROLES,
)


class WorldTokenMixin:
    def _append_global_tokens(
        self,
        world_entries: list[dict[str, Any]],
        obs: dict[str, Any],
        features: dict[str, np.ndarray],
        planner_context: dict[str, Any] | None = None,
    ) -> None:
        scalars = features["scalars"]
        decision_domain = features["decision_domain"]
        world_numeric = np.zeros(TOKEN_NUMERIC_DIM, dtype=np.float32)
        world_numeric[: obs_common.SCALAR_DIM] = scalars
        world_numeric[obs_common.SCALAR_DIM : obs_common.SCALAR_DIM + obs_common.NUM_DOMAINS] = decision_domain
        world_entries.append(self._entry("CLS_WORLD", world_numeric, owner_id=OWNER_NONE, entity_id=0))

        for idx, token_type in enumerate(("CLS_COMBAT", "CLS_BUILD", "CLS_ROUTE")):
            numeric = np.zeros(TOKEN_NUMERIC_DIM, dtype=np.float32)
            numeric[0] = float(decision_domain[idx] > 0.5)
            numeric[1: 1 + obs_common.NUM_DOMAINS] = decision_domain
            world_entries.append(self._entry(token_type, numeric, owner_id=OWNER_NONE, entity_id=0))

        player = obs.get("player") or {}
        combat = obs.get("combat") or {}
        enemies = combat.get("enemies") or []
        hp, max_hp, hp_ratio = obs_common._player_hp_triplet(player)
        block = obs_common._float(player.get("block"))
        incoming = sum(obs_common._float(((enemy or {}).get("intent") or {}).get("total_damage")) for enemy in enemies if isinstance(enemy, dict))

        survival = np.zeros(TOKEN_NUMERIC_DIM, dtype=np.float32)
        survival[0] = hp_ratio
        survival[1] = obs_common._log_norm(hp, obs_common._LOG1P_200)
        survival[2] = obs_common._log_norm(block, obs_common._LOG1P_200)
        survival[3] = obs_common._log_norm(incoming, obs_common._LOG1P_200)
        survival[4] = obs_common._signed_log_norm(hp + block - incoming, obs_common._LOG1P_200)
        survival[5 : 5 + min(obs_common.POWER_DIM, TOKEN_NUMERIC_DIM - 5)] = features["player_powers"][: TOKEN_NUMERIC_DIM - 5]
        world_entries.append(self._entry("PLAYER_SURVIVAL", survival, owner_id=OWNER_PLAYER, entity_id=0))

        budget = np.zeros(TOKEN_NUMERIC_DIM, dtype=np.float32)
        max_energy = obs_common._float(combat.get("max_energy"))
        budget[0] = min(obs_common._float(combat.get("energy")) / max(max_energy, 1.0), 1.0) if max_energy > 0 else 0.0
        budget[1] = min(obs_common._float(combat.get("energy")) / 10.0, 1.0)
        budget[2] = min(max_energy / 10.0, 1.0)
        budget[3] = min(obs_common._float(combat.get("stars")) / 10.0, 1.0)
        budget[4] = obs_common._log_norm(obs_common._float(player.get("gold")), obs_common._LOG1P_500)
        budget[5 : 5 + min(obs_common.RELIC_SIGNAL_DIM, TOKEN_NUMERIC_DIM - 5)] = features["relic_signals"][: TOKEN_NUMERIC_DIM - 5]
        world_entries.append(self._entry("RESOURCE_BUDGET", budget, owner_id=OWNER_PLAYER, entity_id=0))

        threat = np.zeros(TOKEN_NUMERIC_DIM, dtype=np.float32)
        threat[0] = min(len([enemy for enemy in enemies if isinstance(enemy, dict) and enemy.get("is_alive", True)]) / max(obs_common.MAX_ENEMIES, 1), 1.0)
        threat[1] = obs_common._log_norm(incoming, obs_common._LOG1P_200)
        threat[2] = obs_common._log_norm(sum(obs_common._float(enemy.get("hp", enemy.get("current_hp"))) for enemy in enemies if isinstance(enemy, dict)), obs_common._LOG1P_1200)
        threat[3] = obs_common._log_norm(sum(obs_common._float(enemy.get("block")) for enemy in enemies if isinstance(enemy, dict)), obs_common._LOG1P_200)
        threat[4] = float(any(self._infer_enemy_traits(enemy)[0] for enemy in enemies if isinstance(enemy, dict)))
        combat_memory = (planner_context or {}).get("combat_memory") or {}
        if combat_memory:
            player_max_hp = max(
                obs_common._float(combat_memory.get("player_max_hp")),
                max_hp if max_hp > 1.0 else 0.0,
                hp if hp > 1.0 else 0.0,
                1.0,
            )
            initial_total_hp = max(obs_common._float(combat_memory.get("initial_enemy_total_hp"), 1.0), 1.0)
            threat[5] = min(obs_common._float(combat_memory.get("turns_in_combat")) / 20.0, 1.0)
            threat[6] = obs_common._signed_log_norm(obs_common._float(combat_memory.get("player_hp_delta_last_turn")), obs_common._LOG1P_200)
            threat[7] = obs_common._signed_log_norm(obs_common._float(combat_memory.get("player_block_delta_last_turn")), obs_common._LOG1P_200)
            threat[8] = max(-1.0, min(obs_common._float(combat_memory.get("player_energy_delta_last_turn")) / 5.0, 1.0))
            enemy_hp_delta = obs_common._float(combat_memory.get("enemy_total_hp_delta_last_turn"))
            threat[9] = max(-1.0, min(enemy_hp_delta / initial_total_hp, 1.0))
            threat[10] = min(obs_common._float(combat_memory.get("surprise_damage_last_turn")) / player_max_hp, 1.0)
            threat[11] = max(-1.0, min(obs_common._float(combat_memory.get("enemy_count_delta_last_turn")) / 3.0, 1.0))
            threat[12] = min(obs_common._float(combat_memory.get("cum_surprise_damage")) / player_max_hp, 1.0)
        world_entries.append(self._entry("THREAT_SUMMARY", threat, owner_id=OWNER_NONE, entity_id=0))

        objective_numeric = np.zeros(TOKEN_NUMERIC_DIM, dtype=np.float32)
        objective_numeric[: min(obs_common.OBJECTIVE_DIM, TOKEN_NUMERIC_DIM)] = features["objective_context"][:TOKEN_NUMERIC_DIM]
        world_entries.append(self._entry("OBJECTIVE_CONTEXT", objective_numeric, owner_id=OWNER_NONE, entity_id=0))
        run_numeric = np.zeros(TOKEN_NUMERIC_DIM, dtype=np.float32)
        run_numeric[: min(obs_common.RUN_MEMORY_DIM, TOKEN_NUMERIC_DIM)] = features["run_memory"][:TOKEN_NUMERIC_DIM]
        world_entries.append(self._entry("RUN_CONTEXT", run_numeric, owner_id=OWNER_NONE, entity_id=0))

    def _append_entity_tokens(
        self,
        world_entries: list[dict[str, Any]],
        obs: dict[str, Any],
        features: dict[str, np.ndarray],
        planner_context: dict[str, Any] | None = None,
    ) -> None:
        player = obs.get("player") or {}
        hand_cards = self._runtime_cards(obs, "hand", "hand")
        deck_cards = player.get("deck_cards") or []
        draw_cards = self._runtime_cards(obs, "draw_pile", "draw_preview_cards")
        discard_cards = self._runtime_cards(obs, "discard_pile", "discard_cards")
        exhaust_cards = self._runtime_cards(obs, "exhaust_pile", "exhaust_cards")
        play_cards = self._runtime_cards(obs, "play_pile", "play_pile_cards")

        self._append_card_collection(world_entries, features["hand"], features["hand_text"], features["hand_mask"], "HAND_CARD", entity_keys=hand_cards)
        self._append_card_collection(world_entries, features["deck"], features["deck_text"], features["deck_mask"], "DECK_CARD", entity_keys=deck_cards)
        self._append_card_collection(world_entries, *self._encode_runtime_pile(obs, "draw_pile", "draw_preview_cards", limit=12), "DRAW_PREVIEW_CARD", entity_keys=draw_cards)
        self._append_card_collection(world_entries, *self._encode_runtime_pile(obs, "discard_pile", "discard_cards", limit=24), "DISCARD_CARD", entity_keys=discard_cards)
        self._append_card_collection(world_entries, *self._encode_runtime_pile(obs, "exhaust_pile", "exhaust_cards", limit=24), "EXHAUST_CARD", entity_keys=exhaust_cards)
        self._append_card_collection(world_entries, *self._encode_runtime_pile(obs, "play_pile", "play_pile_cards", limit=12), "PLAY_PILE_CARD", entity_keys=play_cards)

        self._append_relic_collection(world_entries, player.get("relics") or [], features["relics"], features["relic_mask"])
        self._append_potion_collection(world_entries, player.get("potions") or [], features["potions"], features["potion_mask"])

        enemies = obs.get("combat", {}).get("enemies") or []
        combat_memory = (planner_context or {}).get("combat_memory") or {}
        memory_enemies = combat_memory.get("enemies") if isinstance(combat_memory, dict) else None
        for enemy_index in range(features["enemy_mask"].shape[0]):
            if features["enemy_mask"][enemy_index] <= 0:
                continue
            owner_id = self._enemy_owner_id(enemy_index)
            enemy = enemies[enemy_index] if enemy_index < len(enemies) and isinstance(enemies[enemy_index], dict) else {}
            enemy_entity_id = _stable_bucket(self._enemy_entity_key(enemy, enemy_index))
            enemy_memory = None
            if isinstance(memory_enemies, dict):
                enemy_memory = memory_enemies.get(self._enemy_entity_key(enemy, enemy_index))
            core = np.zeros(TOKEN_NUMERIC_DIM, dtype=np.float32)
            core[: obs_common.ENEMY_FEAT_DIM] = features["enemies"][enemy_index]
            if isinstance(enemy_memory, dict):
                base = obs_common.ENEMY_FEAT_DIM
                core[base + 0] = obs_common._float(enemy_memory.get("hp_delta_last_turn_ratio"))
                core[base + 1] = obs_common._float(enemy_memory.get("hp_delta_last_3_turns_ratio"))
                core[base + 2] = obs_common._float(enemy_memory.get("block_delta_last_turn_ratio"))
                core[base + 3] = obs_common._float(enemy_memory.get("turns_alive_norm"))
                core[base + 4] = obs_common._float(enemy_memory.get("is_new_this_turn"))
                core[base + 5] = obs_common._float(enemy_memory.get("cum_damage_dealt_to_player_ratio"))
                core[base + 6] = obs_common._float(enemy_memory.get("died_last_turn"))
                core[base + 7] = obs_common._float(enemy_memory.get("attributable_damage_last_turn_ratio"))
                core[base + 8] = obs_common._float(enemy_memory.get("alive_rank_by_hp"))
                core[base + 9] = obs_common._float(enemy_memory.get("threat_rank_by_intent"))
            world_entries.append(
                self._entry(
                    "ENEMY_CORE",
                    core,
                    owner_id=owner_id,
                    entity_id=enemy_entity_id,
                    order_id=1,
                    text_embedding=features["enemy_text"][enemy_index],
                )
            )

            intent = enemy.get("intent") if isinstance(enemy, dict) and isinstance(enemy.get("intent"), dict) else None
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
                world_entries.append(
                    self._entry(
                        "ENEMY_INTENT",
                        numeric,
                        owner_id=owner_id,
                        entity_id=enemy_entity_id,
                        order_id=2,
                        text=str(intent.get("description") or intent.get("label") or ""),
                    )
                )

            memory_powers = enemy_memory.get("powers") if isinstance(enemy_memory, dict) else None
            for power in (enemy.get("powers") or [])[:5]:
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
                    numeric[5] = obs_common._normalize_power_amount(
                        obs_common._float(power_memory.get("amount_delta_last_turn"))
                    ) * (1.0 if obs_common._float(power_memory.get("amount_delta_last_turn")) >= 0 else -1.0)
                    numeric[6] = obs_common._normalize_power_amount(
                        obs_common._float(power_memory.get("amount_delta_since_first_seen"))
                    ) * (1.0 if obs_common._float(power_memory.get("amount_delta_since_first_seen")) >= 0 else -1.0)
                    numeric[7] = min(obs_common._float(power_memory.get("turns_since_first_seen")) / 10.0, 1.0)
                    numeric[8] = obs_common._float(power_memory.get("stack_trend_3_turns"))
                    numeric[9] = obs_common._float(power_memory.get("is_new_this_turn"))
                    numeric[10] = obs_common._float(power_memory.get("is_growing_without_player_action"))
                power_index = sum(
                    1
                    for entry in world_entries
                    if entry["type_id"] == TOKEN_TYPE_TO_ID["ENEMY_POWER"] and entry["owner_id"] == owner_id
                )
                world_entries.append(
                    self._entry(
                        "ENEMY_POWER",
                        numeric,
                        owner_id=owner_id,
                        entity_id=enemy_entity_id,
                        order_id=3 + power_index,
                        text=f"{power.get('title') or ''} | {power.get('description') or ''}",
                    )
                )

            reactive_traits, phase_rules = self._infer_enemy_traits(enemy)
            for trait_index, trait in enumerate(reactive_traits[:4]):
                world_entries.append(
                    self._entry(
                        "ENEMY_REACTIVE_TRAIT",
                        self._trait_numeric(trait),
                        owner_id=owner_id,
                        entity_id=enemy_entity_id,
                        order_id=12 + trait_index,
                        text=str(trait.get("description") or trait.get("trait") or trait.get("effect_type") or ""),
                    )
                )
            for rule_index, rule in enumerate(phase_rules[:3]):
                world_entries.append(
                    self._entry(
                        "ENEMY_PHASE_RULE",
                        self._trait_numeric(rule),
                        owner_id=owner_id,
                        entity_id=enemy_entity_id,
                        order_id=20 + rule_index,
                        text=str(rule.get("description") or rule.get("trait") or ""),
                    )
                )

    def _append_power_slot_tokens(
        self,
        world_entries: list[dict[str, Any]],
        obs: dict[str, Any],
    ) -> None:
        """Emit one POWER_SLOT token per active buff/debuff instance.

        Phase 6.1: splits the previously-inlined per-entity power numerics
        into first-class tokens. Each token exposes the power's categorical
        id (via entity_id bucket) and its effect-algebra coefficients (via
        numeric vector) so that attention can learn interactions directly
        (e.g. "this card's damage 脳 target's damage_mult_received").

        The original inline power features in PLAYER_SURVIVAL / ENEMY tokens
        are preserved for back-compat; POWER_SLOT tokens add a richer
        per-power view on top. Phase 6.2/6.3 will wire a dedicated POWER
        bank + effect_class 脳 target_power_bucket relational bias that
        consumes these tokens exclusively.
        """
        emitted = 0
        budget = MAX_POWER_SLOT_TOKENS

        # ---- Player-owned powers ----
        player = obs.get("player") or {}
        player_powers = player.get("status") or player.get("powers") or []
        if isinstance(player_powers, list):
            for slot_index, power in enumerate(player_powers):
                if emitted >= budget:
                    break
                if not isinstance(power, dict):
                    continue
                token = self._build_power_slot_numeric(power)
                if token is None:
                    continue
                world_entries.append(
                    self._entry(
                        "POWER_SLOT_PLAYER",
                        token["numeric"],
                        owner_id=OWNER_PLAYER,
                        entity_id=token["bucket"],
                        order_id=min(slot_index, MAX_ORDER_ID),
                        text=token["label"],
                    )
                )
                emitted += 1

        # ---- Enemy-owned powers ----
        combat = obs.get("combat") or {}
        enemies = combat.get("enemies") or []
        if isinstance(enemies, list):
            for enemy_index, enemy in enumerate(enemies):
                if emitted >= budget:
                    break
                if not isinstance(enemy, dict):
                    continue
                owner_id = self._enemy_owner_id(enemy_index)
                enemy_powers = enemy.get("powers") or enemy.get("status") or []
                if not isinstance(enemy_powers, list):
                    continue
                for slot_index, power in enumerate(enemy_powers):
                    if emitted >= budget:
                        break
                    if not isinstance(power, dict):
                        continue
                    token = self._build_power_slot_numeric(power)
                    if token is None:
                        continue
                    world_entries.append(
                        self._entry(
                            "POWER_SLOT_ENEMY",
                            token["numeric"],
                            owner_id=owner_id,
                            entity_id=token["bucket"],
                            order_id=min(slot_index, MAX_ORDER_ID),
                            text=token["label"],
                        )
                    )
                    emitted += 1

    def _runtime_card_modifier_tags(self, card: dict[str, Any]) -> list[str]:
        tags: list[str] = []
        for field_name in ("afflictions", "enchantments", "modifiers", "card_modifiers"):
            modifiers = card.get(field_name)
            if not isinstance(modifiers, list):
                continue
            for modifier in modifiers[:8]:
                if isinstance(modifier, dict):
                    semantic_tags = modifier.get("semantic_tags")
                    if isinstance(semantic_tags, list):
                        for tag in semantic_tags:
                            tag_text = str(tag or "").strip().lower()
                            if tag_text in {"adds_retain", "retain"}:
                                tags.append("retain")
                            elif tag_text in {"adds_ethereal", "ethereal"}:
                                tags.append("ethereal")
                            elif tag_text in {"adds_exhaust", "exhaust", "exhaust_self"}:
                                tags.append("exhaust_self")
                            elif tag_text in {"removes_exhaust"}:
                                tags.append("purge")
                            elif tag_text in {"cost_randomizes_on_draw", "cost_reduction_until_played", "sets_cost_zero", "energy_loss_on_play"}:
                                tags.append("cost_lock")
                            elif tag_text in {"autoplay_round_1"}:
                                tags.append("forced_play")
                    semantic_values = modifier.get("semantic_values")
                    if isinstance(semantic_values, dict):
                        if semantic_values.get("adds_retain"):
                            tags.append("retain")
                        if semantic_values.get("adds_ethereal"):
                            tags.append("ethereal")
                        if semantic_values.get("adds_exhaust"):
                            tags.append("exhaust_self")
                        if semantic_values.get("removes_exhaust"):
                            tags.append("purge")
                        if semantic_values.get("autoplay_round_1"):
                            tags.append("forced_play")
                        if any(semantic_values.get(key) for key in ("energy_loss_on_play", "cost_randomizes_on_draw", "cost_reduction_until_played", "sets_cost_zero")):
                            tags.append("cost_lock")
                    joined = " ".join(
                        str(modifier.get(key) or "")
                        for key in ("id", "title", "type", "description", "kind")
                    ).lower()
                else:
                    joined = str(modifier or "").lower()
                if not joined.strip():
                    continue
                if any(token in joined for token in ("bind", "bound", "chain", "shackle")):
                    tags.append("bound")
                if any(token in joined for token in ("lock", "forbid", "disabled", "unplayable", "can't play", "cannot play")):
                    tags.append("card_lock")
                if any(token in joined for token in ("cost", "energy")) and any(token in joined for token in ("lock", "increase", "reduce", "set")):
                    tags.append("cost_lock")
                if any(token in joined for token in ("forced", "must play", "required")):
                    tags.append("forced_play")
                if any(token in joined for token in ("temporary", "this turn", "until", "expire")):
                    tags.append("temporary")
                if "retain" in joined:
                    tags.append("retain")
                if "ethereal" in joined:
                    tags.append("ethereal")
                if "exhaust" in joined:
                    tags.append("exhaust_self")
                if "purge" in joined or "remove" in joined:
                    tags.append("purge")
        deduped: list[str] = []
        for tag in tags:
            if tag in _CARD_KEYWORD_BUCKETS and tag not in deduped:
                deduped.append(tag)
        return deduped

    def _append_card_keyword_slot_tokens(
        self,
        world_entries: list[dict[str, Any]],
        obs: dict[str, Any],
    ) -> None:
        """Emit one CARD_KEYWORD_SLOT token per (hand card 脳 recognized
        keyword) pair.

        Looks up each hand card's semantic_tags in content_registry and
        emits a token per matched keyword. owner_id = OWNER_HAND + card
        position so attention can route keyword 鈫?source card binding.
        entity_id = keyword bucket id (shared categorical space with the
        POWER_ID vocabulary via distinct low-end numbering).
        """
        combat = obs.get("combat") or {}
        hand = combat.get("hand") or []
        if not isinstance(hand, list) or not hand:
            return

        try:
            from content_registry import get_card_metadata
        except Exception:
            return

        emitted = 0
        budget = MAX_CARD_KEYWORD_SLOTS

        for hand_index, card in enumerate(hand):
            if emitted >= budget:
                break
            if not isinstance(card, dict):
                continue
            card_id = str(card.get("id") or "")
            md = get_card_metadata(card_id) if card_id else None
            tags = md.get("semantic_tags") if isinstance(md, dict) else []
            if not isinstance(tags, list):
                tags = []

            # Also fold explicit card.keywords (from sim translator/live bridge)
            # and runtime per-card modifier tags from boss/event effects.
            card_keywords = card.get("keywords") or []
            modifier_tags = self._runtime_card_modifier_tags(card)
            all_tags: list[str] = []
            for t in list(tags) + list(card_keywords if isinstance(card_keywords, list) else []) + modifier_tags:
                t_norm = str(t).strip().lower()
                if t_norm in _CARD_KEYWORD_BUCKETS and t_norm not in all_tags:
                    all_tags.append(t_norm)
            if not all_tags:
                continue

            # Bind this keyword slot to the hand card via owner_id so the
            # attention owner_pair_bias can learn "keyword-for-this-card"
            # as a same-owner relation.
            owner_id = min(OWNER_HAND + hand_index, MAX_OWNER_ID)
            for slot_index, kw in enumerate(all_tags):
                if emitted >= budget:
                    break
                bucket = _CARD_KEYWORD_BUCKETS[kw]
                numeric = np.zeros(TOKEN_NUMERIC_DIM, dtype=np.float32)
                # Minimal feature set 鈥?keywords are mostly categorical.
                numeric[0] = 1.0                               # active flag
                numeric[1] = hand_index / 10.0                 # card position hint
                numeric[2] = float(card.get("is_playable", True))
                numeric[3] = obs_common._float(card.get("cost")) / 3.0
                # Flag which keyword family (for fast linear probing by
                # other downstream heads without embedding lookup).
                if kw in ("ethereal", "exhaust_self", "purge"):
                    numeric[4] = 1.0  # auto-removal-on-use/eot
                if kw in ("retain",):
                    numeric[5] = 1.0  # persists across turns
                if kw in ("innate",):
                    numeric[6] = 1.0  # opening-hand guarantee
                if kw in ("unplayable", "bound", "card_lock", "forced_play"):
                    numeric[7] = 1.0  # curse/blank/restricted
                if kw in ("bound", "card_lock"):
                    numeric[8] = 1.0  # boss/event per-card lock
                if kw in ("cost_lock", "x_cost"):
                    numeric[9] = 1.0  # energy/cost interaction
                if kw in ("temporary", "forced_play"):
                    numeric[10] = 1.0  # timing/obligation
                world_entries.append(
                    self._entry(
                        "CARD_KEYWORD_SLOT",
                        numeric,
                        owner_id=owner_id,
                        entity_id=bucket,
                        order_id=min(slot_index, MAX_ORDER_ID),
                        text=kw,
                    )
                )
                emitted += 1

    def _build_power_slot_numeric(self, power: dict[str, Any]) -> dict[str, Any] | None:
        """Pack a single power dict into the numeric + categorical fields
        a POWER_SLOT token needs. Returns None on malformed input.

        Numeric layout (TOKEN_NUMERIC_DIM=96 slots available; we use 16):
          [0..12]  effect algebra (damage_mult_given, damage_flat_given,
                                   damage_mult_received, damage_flat_received,
                                   block_mult_given, block_flat_given,
                                   block_persistent, end_of_turn_dmg_self,
                                   end_of_turn_dmg_given, stacks_on_applied,
                                   decays_each_turn, is_buff, is_debuff)
          [13]     amount clipped + log-normalized
          [14]     amount sign (positive/negative for reversible powers)
          [15]     amount ratio vs typical cap (amount/10, clipped to 1)
          [16..95] unused 鈥?reserved for Phase 6.x extensions (duration,
                            applier/target hints, etc.)
        """
        pid = str(power.get("id") or "").strip()
        if not pid:
            return None
        amount = power.get("amount")
        try:
            amount_f = float(amount) if amount is not None else 0.0
        except (TypeError, ValueError):
            amount_f = 0.0

        numeric = np.zeros(TOKEN_NUMERIC_DIM, dtype=np.float32)
        algebra = _power_algebra(pid)
        numeric[: _POWER_ALGEBRA_DIM] = algebra
        # Amount features: log-scaled magnitude, sign, clipped ratio.
        numeric[13] = obs_common._log_norm(abs(amount_f), obs_common._LOG1P_200)
        numeric[14] = 1.0 if amount_f > 0 else (-1.0 if amount_f < 0 else 0.0)
        numeric[15] = max(-1.0, min(amount_f / 10.0, 1.0))
        return {
            "numeric": numeric,
            "bucket": _power_id_bucket(pid),
            "label": pid,
        }

    def _append_history_tokens(
        self,
        world_entries: list[dict[str, Any]],
        obs: dict[str, Any],
    ) -> None:
        """Emit MAX_STEP_DETAIL_TOKENS step-detail + MAX_TURN_SUMMARY_TOKENS
        turn-summary tokens from ``obs["_action_history"]``.

        Always emits a fixed number of tokens (padded with is_empty=1 when
        the tracker is shorter) so MAX_WORLD_TOKENS stays invariant across
        calls 鈥?rollout buffers and type_id arrays are fixed-shape.

        Numeric layout is split between step-detail and turn-summary so a
        single shared HISTORY role / zone / owner still produces distinct
        feature distributions the model can separate via the token_type
        one-hot (type id carried on world_token_type_ids).
        """
        history_dict = obs.get("_action_history") if isinstance(obs, dict) else None
        if not isinstance(history_dict, dict):
            history_dict = {"step_detail": [], "turn_summary": []}
        step_detail_entries = history_dict.get("step_detail") if isinstance(history_dict.get("step_detail"), list) else []
        turn_summary_entries = history_dict.get("turn_summary") if isinstance(history_dict.get("turn_summary"), list) else []

        # --- Step-detail tokens ---
        for slot in range(MAX_STEP_DETAIL_TOKENS):
            entry = step_detail_entries[slot] if slot < len(step_detail_entries) else None
            numeric, card_bucket, text = self._build_history_step_numeric(entry, slot)
            world_entries.append(
                self._entry(
                    "HISTORY_STEP_DETAIL",
                    numeric,
                    owner_id=OWNER_HISTORY,
                    entity_id=int(card_bucket),
                    order_id=min(slot, MAX_ORDER_ID),
                    zone_id=TOKEN_ZONE_TO_ID.get("HISTORY", 0),
                    text=text,
                )
            )

        # --- Turn-summary tokens ---
        for slot in range(MAX_TURN_SUMMARY_TOKENS):
            entry = turn_summary_entries[slot] if slot < len(turn_summary_entries) else None
            numeric, text = self._build_history_turn_summary_numeric(entry, slot)
            world_entries.append(
                self._entry(
                    "HISTORY_TURN_SUMMARY",
                    numeric,
                    owner_id=OWNER_HISTORY,
                    entity_id=0,
                    order_id=min(slot, MAX_ORDER_ID),
                    zone_id=TOKEN_ZONE_TO_ID.get("HISTORY", 0),
                    text=text,
                )
            )

    def _build_history_step_numeric(
        self, entry: dict[str, Any] | None, slot: int
    ) -> tuple[np.ndarray, int, str]:
        """Pack a single step-detail numeric block.

        Layout (TOKEN_NUMERIC_DIM = 96):
          [0]       is_empty             (1 = padding, tracker had no entry)
          [1]       is_step_detail       (always 1 here; turn-summary sets [1]=0)
          [2..]     family one-hot       (len = NUM_FAMILIES)
          [next]    semantic_role flags  (NUM_SEMANTIC_ROLES bits)
          [next]    target_scope one-hot (len = NUM_TARGET_SCOPES)
          [next]    step_offset one-hot  (MAX_STEP_DETAIL_TOKENS slots)
          [next]    flags (7):            same_turn / same_encounter /
                    same_floor / phase_changed / combat_ended / rejected /
                    reward_nonzero
          [next]    reward scalars (2):   reward clipped [-1,1], abs(reward)
          [next]    pre_state_vec  (STATE_SNAPSHOT_DIM = 8) 鈥?Tier 2
          [next]    post_state_vec (STATE_SNAPSHOT_DIM = 8) 鈥?Tier 2
          [next]    step_offset_norm (1)
          remainder zero (~11 spare slots 鈥?room for future additions)

        The pre/post snapshot pair lets any downstream linear probe
        compute "delta = post - pre" on every dimension. The separate
        causality_delta tensor in the tracker is NOT packed here: it's
        delivered to the aux head's loss target path directly (via
        env_v2 info 鈫?aux_maskable_ppo buffer), NOT through the obs
        tensor. Keeping targets off the observation tensor avoids
        teaching the policy to trivially shortcut 鈥?the policy sees
        pre/post context but has to actively predict the delta.
        """
        from .action_history import STATE_SNAPSHOT_DIM
        from .semantic_action import SEMANTIC_ACTION_FAMILIES, SEMANTIC_TARGET_SCOPES

        numeric = np.zeros(TOKEN_NUMERIC_DIM, dtype=np.float32)
        numeric[1] = 1.0

        if not isinstance(entry, dict):
            numeric[0] = 1.0
            return numeric, 0, ""

        cursor = 2
        family_idx = int(entry.get("family_idx") or 0)
        if 0 <= family_idx < len(SEMANTIC_ACTION_FAMILIES):
            numeric[cursor + family_idx] = 1.0
        cursor += len(SEMANTIC_ACTION_FAMILIES)
        role_flags = int(entry.get("semantic_role_flags") or 0)
        for bit in range(NUM_SEMANTIC_ROLES):
            if role_flags & (1 << bit):
                numeric[cursor + bit] = 1.0
        cursor += NUM_SEMANTIC_ROLES
        scope_idx = int(entry.get("target_scope_idx") or 0)
        if 0 <= scope_idx < len(SEMANTIC_TARGET_SCOPES):
            numeric[cursor + scope_idx] = 1.0
        cursor += len(SEMANTIC_TARGET_SCOPES)
        step_offset = int(entry.get("step_offset") or 0)
        step_offset = max(0, min(step_offset, MAX_STEP_DETAIL_TOKENS - 1))
        numeric[cursor + step_offset] = 1.0
        cursor += MAX_STEP_DETAIL_TOKENS

        numeric[cursor + 0] = 1.0 if entry.get("same_turn") else 0.0
        numeric[cursor + 1] = 1.0 if entry.get("same_encounter") else 0.0
        numeric[cursor + 2] = 1.0 if entry.get("same_floor") else 0.0
        numeric[cursor + 3] = 1.0 if entry.get("phase_changed") else 0.0
        numeric[cursor + 4] = 1.0 if entry.get("combat_ended") else 0.0
        numeric[cursor + 5] = 1.0 if entry.get("rejected") else 0.0
        numeric[cursor + 6] = 1.0 if entry.get("reward_nonzero") else 0.0
        cursor += 7

        reward = float(entry.get("reward") or 0.0)
        numeric[cursor + 0] = max(-1.0, min(reward, 1.0))
        numeric[cursor + 1] = min(abs(reward), 1.0)
        cursor += 2

        # Tier 2: pre_state_vec
        pre_vec = entry.get("pre_state_vec") or []
        for i in range(STATE_SNAPSHOT_DIM):
            numeric[cursor + i] = float(pre_vec[i]) if i < len(pre_vec) else 0.0
        cursor += STATE_SNAPSHOT_DIM
        # Tier 2: post_state_vec
        post_vec = entry.get("post_state_vec") or []
        for i in range(STATE_SNAPSHOT_DIM):
            numeric[cursor + i] = float(post_vec[i]) if i < len(post_vec) else 0.0
        cursor += STATE_SNAPSHOT_DIM

        numeric[cursor] = step_offset / max(MAX_STEP_DETAIL_TOKENS - 1, 1)
        cursor += 1

        card_bucket = int(entry.get("card_id_bucket") or 0)
        text = str(entry.get("canonical_text") or "")
        return numeric, card_bucket, text

    def _build_history_turn_summary_numeric(
        self, entry: dict[str, Any] | None, slot: int
    ) -> tuple[np.ndarray, str]:
        """Pack a single turn-summary numeric block.

        Layout (TOKEN_NUMERIC_DIM = 96):
          [0]       is_empty                (1 = padding)
          [1]       is_step_detail          (0 鈥?this is a turn summary)
          [2..10]   turn_offset one-hot     (MAX_TURN_SUMMARY_TOKENS=8 slots, offset 1..8)
          [10..14]  action counts           (n_attacks/n_skills/n_powers/n_potions normalized)
          [14..17]  totals                  (damage/block/hp_lost normalized)
          [17..22]  end-of-turn stats       (strength/dex/focus/block/enemy_hp_ratio)
          [22]      end_enemy_vuln_total normalized
          [23]      turn_num normalized
          [24..40]  key_power_card_flags    (16 bits, one per slot)
          [40..44]  outcome flags           (enemy_killed / player_took_dmg /
                                             player_scaled / low_energy_waste)
          remainder zero 鈥?Tier 2 can add cross-turn comparison scalars here.
        """
        numeric = np.zeros(TOKEN_NUMERIC_DIM, dtype=np.float32)
        # [1] = 0: this is NOT step_detail. Stays zero for turn summaries.

        if not isinstance(entry, dict):
            numeric[0] = 1.0
            return numeric, ""

        cursor = 2
        # turn_offset one-hot
        turn_offset = int(entry.get("turn_offset") or 1)
        turn_offset_slot = max(0, min(turn_offset - 1, MAX_TURN_SUMMARY_TOKENS - 1))
        numeric[cursor + turn_offset_slot] = 1.0
        cursor += MAX_TURN_SUMMARY_TOKENS

        # Action counts (normalized by a generous ceiling 鈥?8 cards/turn is a lot)
        numeric[cursor + 0] = min(float(entry.get("n_attacks") or 0) / 8.0, 1.0)
        numeric[cursor + 1] = min(float(entry.get("n_skills") or 0) / 8.0, 1.0)
        numeric[cursor + 2] = min(float(entry.get("n_powers") or 0) / 4.0, 1.0)
        numeric[cursor + 3] = min(float(entry.get("n_potions_used") or 0) / 3.0, 1.0)
        cursor += 4

        # Totals
        numeric[cursor + 0] = min(float(entry.get("total_damage_dealt") or 0) / 50.0, 1.0)
        numeric[cursor + 1] = min(float(entry.get("total_block_gained") or 0) / 30.0, 1.0)
        numeric[cursor + 2] = min(float(entry.get("total_hp_lost") or 0) / 30.0, 1.0)
        cursor += 3

        # End-of-turn stats
        numeric[cursor + 0] = min(float(entry.get("end_strength") or 0) / 10.0, 1.0)
        numeric[cursor + 1] = min(float(entry.get("end_dex") or 0) / 10.0, 1.0)
        numeric[cursor + 2] = min(float(entry.get("end_focus") or 0) / 10.0, 1.0)
        numeric[cursor + 3] = min(float(entry.get("end_player_block") or 0) / 40.0, 1.0)
        numeric[cursor + 4] = max(0.0, min(float(entry.get("end_enemy_total_hp_ratio") or 0), 1.0))
        cursor += 5

        numeric[cursor + 0] = min(float(entry.get("end_enemy_vuln_total") or 0) / 10.0, 1.0)
        cursor += 1
        numeric[cursor + 0] = min(float(entry.get("turn_num") or 0) / 30.0, 1.0)
        cursor += 1

        # Key-power-card flags (16 bits, one per slot so attention doesn't
        # have to learn a bitmap decoder)
        key_flags = int(entry.get("key_power_card_flags") or 0)
        for bit in range(NUM_KEY_POWER_FLAGS):
            if key_flags & (1 << bit):
                numeric[cursor + bit] = 1.0
        cursor += NUM_KEY_POWER_FLAGS

        # Outcome flags
        numeric[cursor + 0] = 1.0 if entry.get("enemy_killed") else 0.0
        numeric[cursor + 1] = 1.0 if entry.get("player_took_dmg") else 0.0
        numeric[cursor + 2] = 1.0 if entry.get("player_scaled") else 0.0
        numeric[cursor + 3] = 1.0 if entry.get("low_energy_waste") else 0.0
        cursor += 4

        text = str(entry.get("canonical_text") or "")
        return numeric, text
