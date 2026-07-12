"""Relic, potion, card-profile, and support collection token helpers."""

from __future__ import annotations

from typing import Any

import numpy as np

from content_registry import build_live_potion_semantic_text, build_live_relic_semantic_text

from . import observation_common as obs_common
from ._observation_v3_schema import (
    OWNER_POTION,
    OWNER_RELIC,
    TOKEN_NUMERIC_DIM,
    _owner_for_pile,
    _stable_bucket,
)
from .potion_profiles import (
    DEFAULT_EFFECT_PROFILE,
)
from .potion_profiles import (
    get_potion_profile as _get_potion_profile,
)
from .text_encoder import TEXT_DIM


def _resolve_potion_effect(potion: Any) -> tuple[dict[str, Any], dict[str, Any]]:
    """Merge bridge live effect profile with the Python registry (bridge wins)."""
    if not isinstance(potion, dict):
        return ({}, dict(DEFAULT_EFFECT_PROFILE))
    pid = str(potion.get("id") or "").strip()
    registry_entry = _get_potion_profile(pid) if pid else {}
    base_effect = dict(DEFAULT_EFFECT_PROFILE)
    base_effect.update(registry_entry.get("effect_profile") or {})
    bridge_effect = potion.get("effect_profile")
    if isinstance(bridge_effect, dict):
        base_effect.update({key: value for key, value in bridge_effect.items() if value is not None})
    merged_entry = dict(registry_entry)
    for key in ("effect_family", "semantic_tags", "timing_tags", "training_tags", "target_scope"):
        live_value = potion.get(key)
        if live_value:
            merged_entry[key] = live_value
    if "enabled_for_training" in potion:
        merged_entry["enabled_for_training"] = bool(potion["enabled_for_training"])
    return (merged_entry, base_effect)


class SupportTokenMixin:
    def _append_relic_collection(
        self,
        world_entries: list[dict[str, Any]],
        relic_entries: list[Any],
        relic_text: np.ndarray,
        relic_mask: np.ndarray,
    ) -> None:
        total = len(relic_entries) if isinstance(relic_entries, list) else 0
        for index in range(min(relic_text.shape[0], total)):
            if relic_mask[index] <= 0:
                continue
            relic = relic_entries[index]
            numeric = self._relic_numeric(relic, index=index, total=max(total, 1))
            entity_key = self._support_entity_key(relic, fallback=f"relic:{index}")
            world_entries.append(
                self._entry(
                    "RELIC",
                    numeric,
                    owner_id=OWNER_RELIC,
                    entity_id=_stable_bucket(entity_key),
                    order_id=index + 1,
                    text_embedding=relic_text[index],
                )
            )

    def _append_potion_collection(
        self,
        world_entries: list[dict[str, Any]],
        potion_entries: list[Any],
        potion_text: np.ndarray,
        potion_mask: np.ndarray,
    ) -> None:
        total = len(potion_entries) if isinstance(potion_entries, list) else 0
        for index in range(min(potion_text.shape[0], total)):
            if potion_mask[index] <= 0:
                continue
            potion = potion_entries[index]
            title = potion if isinstance(potion, str) else (potion.get("title") if isinstance(potion, dict) else "")
            if str(title or "").strip() == "[empty]":
                continue
            numeric = self._potion_numeric(potion, source_profile=None, slot_index=index, total_slots=max(total, 1))
            entity_key = self._support_entity_key(potion, fallback=f"potion:{index}")
            world_entries.append(
                self._entry(
                    "POTION",
                    numeric,
                    owner_id=OWNER_POTION,
                    entity_id=_stable_bucket(entity_key),
                    order_id=index + 1,
                    text_embedding=potion_text[index],
                )
            )

    def _append_relic_trigger_locals(
        self,
        entries: list[dict[str, Any]],
        obs: dict[str, Any],
        action: dict[str, Any],
    ) -> None:
        player = obs.get("player") or {}
        relics = player.get("relics") or []
        if not isinstance(relics, list) or not relics:
            return

        scored_entries: list[tuple[float, dict[str, Any]]] = []
        for index, relic in enumerate(relics[: obs_common.MAX_RELICS]):
            if not isinstance(relic, dict | str):
                continue
            numeric = self._relic_numeric(relic, action=action, index=index, total=max(len(relics), 1))
            entry = self._entry(
                "RELIC_TRIGGER_LOCAL",
                numeric,
                owner_id=OWNER_RELIC,
                entity_id=_stable_bucket(self._support_entity_key(relic, fallback=f"relic:{index}")),
                order_id=index + 1,
                text=self._relic_text(relic),
            )
            scored_entries.append((float(numeric[0]), entry))

        scored_entries.sort(key=lambda item: item[0], reverse=True)
        for _, entry in scored_entries[:3]:
            entries.append(entry)

    def _append_potion_option_locals(
        self,
        entries: list[dict[str, Any]],
        obs: dict[str, Any],
        action: dict[str, Any],
        features: dict[str, np.ndarray],
    ) -> None:
        player = obs.get("player") or {}
        potions = player.get("potions") or []
        if not isinstance(potions, list) or not potions:
            return

        source_profile = self._source_profile(action.get("card") if isinstance(action.get("card"), dict) else None)
        scored_entries: list[tuple[float, dict[str, Any]]] = []
        for index, potion in enumerate(potions[: obs_common.MAX_POTIONS]):
            title = potion if isinstance(potion, str) else (potion.get("title") if isinstance(potion, dict) else "")
            if str(title or "").strip() == "[empty]":
                continue
            numeric = self._potion_numeric(potion, source_profile=source_profile, slot_index=index, total_slots=max(len(potions), 1))
            text_embedding = features["potions"][index] if index < features["potions"].shape[0] and features["potion_mask"][index] > 0 else None
            entry = self._entry(
                "POTION_OPTION_LOCAL",
                numeric,
                owner_id=OWNER_POTION,
                entity_id=_stable_bucket(self._support_entity_key(potion, fallback=f"potion:{index}")),
                order_id=index + 1,
                text=self._potion_text(potion),
                text_embedding=text_embedding,
            )
            scored_entries.append((float(numeric[0]), entry))

        scored_entries.sort(key=lambda item: item[0], reverse=True)
        for _, entry in scored_entries[:2]:
            entries.append(entry)

    def _relic_numeric(
        self,
        relic: dict[str, Any] | str,
        *,
        action: dict[str, Any] | None = None,
        index: int = 0,
        total: int = 1,
    ) -> np.ndarray:
        numeric = np.zeros(TOKEN_NUMERIC_DIM, dtype=np.float32)
        text = self._relic_text(relic).lower()
        signal_vector = np.zeros(obs_common.RELIC_SIGNAL_DIM, dtype=np.float32)
        obs_common._encode_relic_signals(signal_vector, [relic])
        profile = self._source_profile(action.get("card") if isinstance(action, dict) and isinstance(action.get("card"), dict) else action.get("potion") if isinstance(action, dict) and isinstance(action.get("potion"), dict) else None)

        relevance = float(np.clip(signal_vector.sum() / 4.0, 0.0, 1.0))
        relevance += 0.20 * signal_vector[0] * float(profile["x_cost"] > 0.5 or profile["cost"] >= 2.0)
        relevance += 0.15 * signal_vector[1] * float(profile["draw"] > 0.0 or profile["zero_cost"] > 0.5)
        relevance += 0.20 * max(signal_vector[2], signal_vector[4]) * profile["attack"]
        relevance += 0.20 * max(signal_vector[3], signal_vector[10]) * max(profile["skill"], float(profile["block"] > 0.0))
        relevance += 0.10 * float("attack" in text or "hit" in text) * profile["attack"]
        relevance += 0.10 * float("skill" in text or "block" in text) * max(profile["skill"], float(profile["block"] > 0.0))
        relevance += 0.10 * float("power" in text) * profile["power"]
        relevance += 0.10 * float("exhaust" in text or "ethereal" in text or "burn" in text) * max(profile["exhaust"], profile["ethereal"])
        relevance += 0.10 * float("potion" in text) * float(isinstance(action, dict) and str(action.get("kind") or "") in {"use_potion", "discard_potion"})
        relevance = float(np.clip(relevance, 0.0, 1.0))

        numeric[0] = relevance
        numeric[1 : 1 + obs_common.RELIC_SIGNAL_DIM] = signal_vector
        base = 1 + obs_common.RELIC_SIGNAL_DIM
        numeric[base] = min((index + 1) / max(total, 1), 1.0)
        numeric[base + 1] = float("attack" in text or "hit" in text)
        numeric[base + 2] = float("skill" in text or "block" in text)
        numeric[base + 3] = float("power" in text)
        numeric[base + 4] = float("exhaust" in text or "ethereal" in text or "burn" in text)
        numeric[base + 5] = float("potion" in text)
        return numeric

    def _potion_numeric(
        self,
        potion: dict[str, Any] | str,
        *,
        source_profile: dict[str, float] | None,
        slot_index: int,
        total_slots: int,
    ) -> np.ndarray:
        numeric = np.zeros(TOKEN_NUMERIC_DIM, dtype=np.float32)
        potion_profile = self._source_profile(potion if isinstance(potion, dict) else None)
        text = self._potion_text(potion).lower()
        merged_entry, effect = _resolve_potion_effect(potion)
        effect_family = list(merged_entry.get("effect_family") or [])
        timing_tags = list(merged_entry.get("timing_tags") or [])
        training_tags = list(merged_entry.get("training_tags") or [])

        damage_v = float(effect.get("damage") or potion_profile["damage"] or 0.0)
        block_v = float(effect.get("block") or potion_profile["block"] or 0.0)
        draw_v = float(effect.get("draw") or potion_profile["draw"] or 0.0)
        energy_v = float(effect.get("energy_gain") or potion_profile["energy"] or 0.0)
        heal_v = float(effect.get("heal") or potion_profile["heal"] or 0.0)
        weak_v = float(effect.get("weak") or potion_profile["weak"] or 0.0)
        vuln_v = float(effect.get("vulnerable") or potion_profile["vulnerable"] or 0.0)
        poison_v = float(effect.get("poison") or 0.0)
        str_v = float(effect.get("strength") or 0.0)
        dex_v = float(effect.get("dexterity") or 0.0)
        intang_v = float(effect.get("intangible") or 0.0)
        prevent_v = float(effect.get("prevent_damage") or 0.0)
        gen_card_v = float(effect.get("generate_card_count") or 0.0)
        discover_v = float(effect.get("discover_count") or 0.0)
        upgrade_v = float(effect.get("upgrade_hand") or 0.0)
        dup_next_v = float(effect.get("duplicate_next") or 0.0)
        retrieve_v = float(effect.get("retrieve_from_discard") or 0.0)
        replace_v = float(effect.get("replace_or_transform_hand") or 0.0)
        is_aoe = bool(effect.get("aoe"))
        is_single = bool(effect.get("single_target")) or potion_profile["single_target"] > 0.5
        is_random = bool(effect.get("random_target"))
        target_required = bool(effect.get("target_required"))
        can_change_facing = bool(effect.get("can_change_facing_if_targeted_enemy"))
        requires_followup = bool(effect.get("requires_followup"))
        long_term_value = bool(effect.get("long_term_value"))
        passive_or_triggered = bool(effect.get("passive_or_triggered"))
        enabled_training = bool(merged_entry.get("enabled_for_training", True))

        relevance = 0.0
        if source_profile is not None:
            relevance += 0.20 * float(damage_v > 0.0) * max(source_profile["attack"], float(source_profile["damage"] > 0.0))
            relevance += 0.20 * float(block_v > 0.0) * max(source_profile["skill"], float(source_profile["block"] > 0.0))
            relevance += 0.20 * float(energy_v > 0.0) * float(source_profile["x_cost"] > 0.5 or source_profile["cost"] >= 2.0)
            relevance += 0.15 * float(draw_v > 0.0) * max(source_profile["zero_cost"], float(source_profile["draw"] > 0.0))
            relevance += 0.10 * float(heal_v > 0.0 or potion_profile["hp_loss"] < 0.0) * float(source_profile["hp_loss"] > 0.0)
        relevance += 0.10 * float("attack" in text) * float(damage_v > 0.0)
        relevance = float(np.clip(relevance, 0.0, 1.0))

        numeric[0] = relevance
        numeric[1] = obs_common._log_norm(damage_v, obs_common._LOG1P_200)
        numeric[2] = obs_common._log_norm(block_v, obs_common._LOG1P_200)
        numeric[3] = min(draw_v / 5.0, 1.0)
        numeric[4] = min(energy_v / 5.0, 1.0)
        numeric[5] = obs_common._log_norm(heal_v, obs_common._LOG1P_200)
        numeric[6] = min(max(weak_v, vuln_v) / 5.0, 1.0)
        numeric[7] = float(is_single)
        numeric[8] = float(is_aoe)
        numeric[9] = min((slot_index + 1) / max(total_slots, 1), 1.0)
        numeric[10] = float("discard" not in text)

        # Phase 3 (potion timing v1): structured slot expansion. Slots 11-38
        # consume the bridge effect_profile / Python registry merge so the
        # potion world token and use_potion action token observe identical
        # capability signals. See docs/potion-timing-modeling-plan.md §7.3.
        numeric[11] = min(poison_v / 10.0, 1.0)
        numeric[12] = min(weak_v / 5.0, 1.0)
        numeric[13] = min(vuln_v / 5.0, 1.0)
        numeric[14] = min(str_v / 5.0, 1.0)
        numeric[15] = min(dex_v / 5.0, 1.0)
        numeric[16] = min(max(intang_v, prevent_v / 10.0), 1.0)
        numeric[17] = min(gen_card_v / 5.0, 1.0)
        numeric[18] = min(discover_v / 5.0, 1.0)
        numeric[19] = min(upgrade_v, 1.0)
        numeric[20] = min(dup_next_v, 1.0)
        numeric[21] = min(retrieve_v, 1.0)
        numeric[22] = min(replace_v, 1.0)
        numeric[23] = float(is_random)
        numeric[24] = float(requires_followup)
        numeric[25] = float("save_if_low_threat" in training_tags)
        numeric[26] = float(long_term_value or "long_term_value" in timing_tags)
        numeric[27] = float(passive_or_triggered)
        numeric[28] = float(can_change_facing)
        numeric[29] = float(target_required)
        numeric[30] = min((slot_index + 1) / max(total_slots, 1), 1.0)
        numeric[31] = float(not enabled_training)
        numeric[32] = float(any(t in timing_tags for t in ("setup_tool", "scaling_setup")) or "setup" in effect_family)
        numeric[33] = float(any(t in timing_tags for t in ("scaling_setup",)) or any(f in effect_family for f in ("strength", "dexterity", "focus", "scaling")))
        numeric[34] = float("mechanism_answer_candidate" in timing_tags or can_change_facing)
        numeric[35] = float("hand_context_dependency" in timing_tags or upgrade_v > 0.0 or dup_next_v > 0.0 or replace_v > 0.0)
        numeric[36] = float(retrieve_v > 0.0 or "discard_context_dependency" in timing_tags)
        numeric[37] = float(any(t in timing_tags for t in ("dig_for_answer", "draw_pile_context_dependency")) or draw_v > 0.0)
        numeric[38] = float("exhaust_pile_context_dependency" in timing_tags)
        return numeric

    def _source_profile(self, source: dict[str, Any] | None) -> dict[str, float]:
        profile = {
            "cost": 0.0,
            "x_cost": 0.0,
            "attack": 0.0,
            "skill": 0.0,
            "power": 0.0,
            "strength": 0.0,
            "dexterity": 0.0,
            "zero_cost": 0.0,
            "damage": 0.0,
            "block": 0.0,
            "draw": 0.0,
            "energy": 0.0,
            "heal": 0.0,
            "hp_loss": 0.0,
            "weak": 0.0,
            "vulnerable": 0.0,
            "hits": 0.0,
            "single_target": 0.0,
            "aoe_target": 0.0,
            "exhaust": 0.0,
            "ethereal": 0.0,
            "retain": 0.0,
            "energy_loss": 0.0,
            "self_damage": 0.0,
            "play_count_bonus": 0.0,
            "damage_add": 0.0,
            "damage_mult": 1.0,
            "block_add": 0.0,
            "adds_exhaust": 0.0,
            "removes_exhaust": 0.0,
            "adds_retain": 0.0,
            "adds_ethereal": 0.0,
            "once_per_combat": 0.0,
            "disabled_after_play": 0.0,
            "cost_randomizes_on_draw": 0.0,
            "sets_cost_zero": 0.0,
            "shuffle_top": 0.0,
            "typed_ops_count": 0.0,
            "typed_modifies_hand": 0.0,
            "typed_upgrade_hand": 0.0,
            "typed_exhaust_cards": 0.0,
            "typed_discard_cards": 0.0,
            "typed_transform_cards": 0.0,
            "typed_copy_cards": 0.0,
            "typed_modify_cost": 0.0,
            "typed_set_replay": 0.0,
            "typed_retain_cards": 0.0,
            "typed_add_modifier": 0.0,
            "typed_add_keyword": 0.0,
            "typed_add_generated_card": 0.0,
            "typed_draw_cards": 0.0,
            "typed_draw_amount": 0.0,
            "typed_gain_energy": 0.0,
            "typed_gain_energy_amount": 0.0,
            "typed_hp_loss": 0.0,
            "typed_apply_power": 0.0,
            "typed_no_draw": 0.0,
            "typed_future_penalty": 0.0,
            "typed_requires_followup": 0.0,
            "typed_strategic_skip_if_no_followup": 0.0,
            "typed_not_x_cost_filter": 0.0,
            "typed_x_cost_filter": 0.0,
            "typed_hand_context_dependency": 0.0,
            "typed_discard_context_dependency": 0.0,
            "typed_exhaust_context_dependency": 0.0,
            "typed_draw_context_dependency": 0.0,
            "typed_deck_context_dependency": 0.0,
            "typed_consumes_future_resource": 0.0,
            "typed_once_or_exhaust_self": 0.0,
            "typed_selection_required": 0.0,
            "typed_all_scope": 0.0,
            "typed_card_rule_modifier": 0.0,
            "typed_card_state_mutation": 0.0,
            "typed_hand_context_needed": 0.0,
            "strategic_skip_value": 0.0,
        }
        if not isinstance(source, dict):
            return profile

        preview = obs_common._build_card_preview_bundle(source)
        strength, dexterity, energy, hits = obs_common._get_card_extra_metrics(source)
        kw_flags, _ = obs_common._get_card_keywords(source)
        modifier_sem = obs_common._aggregate_card_modifier_semantics(source)
        effect_sem = obs_common._aggregate_card_effect_profile_semantics(source)
        typed_energy = effect_sem.get("typed_gain_energy_amount", 0.0)
        typed_draw = effect_sem.get("typed_draw_amount", 0.0)
        typed_hp_loss = effect_sem.get("typed_hp_loss", 0.0)
        card_type = str(source.get("type") or "").capitalize()
        target = str(source.get("target_type") or source.get("target") or "").lower()
        cost = obs_common._float(source.get("cost"))

        profile.update(
            {
                "cost": max(cost, 0.0),
                "x_cost": float(
                    bool(source.get("x_cost") or source.get("costs_x"))
                    or str(source.get("cost") or source.get("canonical_energy_cost") or "").strip().upper() == "X"
                ),
                "attack": 1.0 if card_type == "Attack" else 0.0,
                "skill": 1.0 if card_type == "Skill" else 0.0,
                "power": 1.0 if card_type == "Power" else 0.0,
                "zero_cost": 1.0 if cost == 0 else 0.0,
                "damage": preview["preview_damage"],
                "block": preview["preview_block"],
                "draw": max(obs_common._preview_metric(source, "draw") + modifier_sem.get("draw", 0.0), typed_draw),
                "energy": max(energy, typed_energy),
                "heal": obs_common._preview_metric(source, "heal"),
                "hp_loss": max(obs_common._preview_metric(source, "hp_loss") + modifier_sem.get("self_damage", 0.0), typed_hp_loss),
                "energy_loss": modifier_sem.get("energy_loss_on_play", 0.0),
                "self_damage": modifier_sem.get("self_damage", 0.0),
                "weak": obs_common._preview_metric(source, "weak") + modifier_sem.get("weak", 0.0),
                "vulnerable": obs_common._preview_metric(source, "vulnerable"),
                "hits": hits,
                "single_target": 1.0 if "single" in target or "anyenemy" in target else 0.0,
                "aoe_target": 1.0 if "all" in target else 0.0,
                "exhaust": 1.0 if kw_flags[0] or effect_sem.get("typed_once_or_exhaust_self", 0.0) > 0.0 else 0.0,
                "ethereal": 1.0 if kw_flags[1] else 0.0,
                "retain": 1.0 if kw_flags[2] or effect_sem.get("typed_retain_cards", 0.0) > 0.0 else 0.0,
                "play_count_bonus": modifier_sem.get("play_count_bonus", 0.0),
                "damage_add": modifier_sem.get("damage_add", 0.0),
                "damage_mult": modifier_sem.get("damage_mult", 1.0),
                "block_add": modifier_sem.get("block_add", 0.0) + modifier_sem.get("block_on_play", 0.0),
                "adds_exhaust": modifier_sem.get("adds_exhaust", 0.0),
                "removes_exhaust": modifier_sem.get("removes_exhaust", 0.0),
                "adds_retain": modifier_sem.get("adds_retain", 0.0),
                "adds_ethereal": modifier_sem.get("adds_ethereal", 0.0),
                "once_per_combat": modifier_sem.get("once_per_combat", 0.0),
                "disabled_after_play": modifier_sem.get("disabled_after_play", 0.0),
                "cost_randomizes_on_draw": modifier_sem.get("cost_randomizes_on_draw", 0.0),
                "sets_cost_zero": modifier_sem.get("sets_cost_zero", 0.0),
                "shuffle_top": modifier_sem.get("shuffle_top", 0.0),
            }
        )
        profile.update(effect_sem)
        profile["strength"] = strength
        profile["dexterity"] = dexterity
        profile["draw"] = max(profile["draw"], typed_draw)
        profile["energy"] = max(profile["energy"], typed_energy)
        profile["hp_loss"] = max(profile["hp_loss"], typed_hp_loss)
        profile["exhaust"] = max(profile["exhaust"], effect_sem.get("typed_once_or_exhaust_self", 0.0))
        profile["retain"] = max(profile["retain"], effect_sem.get("typed_retain_cards", 0.0))
        profile["strategic_skip_value"] = float(
            profile["exhaust"] > 0.5
            or profile["retain"] > 0.5
            or profile["energy_loss"] > 0.0
            or profile["self_damage"] > 0.0
            or profile.get("typed_strategic_skip_if_no_followup", 0.0) > 0.5
            or profile.get("typed_requires_followup", 0.0) > 0.5
            or profile.get("typed_no_draw", 0.0) > 0.5
            or profile.get("typed_future_penalty", 0.0) > 0.5
            or profile.get("typed_consumes_future_resource", 0.0) > 0.5
            or profile.get("typed_exhaust_cards", 0.0) > 0.5
            or profile.get("typed_transform_cards", 0.0) > 0.5
            or profile.get("typed_modify_cost", 0.0) > 0.5
            or profile.get("typed_set_replay", 0.0) > 0.5
        )
        return profile

    def _source_profile_for_obs(self, source: dict[str, Any] | None, obs: dict[str, Any] | None) -> dict[str, float]:
        """Source profile with narrow dynamic fallbacks that require live obs.

        Generic static card metadata cannot know Body Slam damage because it is
        a function of *current player block*.  Use this method in obs-aware
        candidate-local / target-reaction contexts; keep _source_profile()
        static for reward/shop/deck contexts where current block is not the
        right value.
        """
        profile = self._source_profile(source)
        body_slam_damage = self._body_slam_dynamic_damage(source, obs)
        if body_slam_damage <= 0.0:
            return profile
        patched = dict(profile)
        patched["damage"] = max(patched.get("damage", 0.0), body_slam_damage)
        patched["hits"] = max(patched.get("hits", 0.0), 1.0)
        patched["attack"] = max(patched.get("attack", 0.0), 1.0)
        patched["single_target"] = max(patched.get("single_target", 0.0), 1.0)
        patched["block_scaled_damage"] = 1.0
        return patched

    def _source_text(self, source: dict[str, Any] | None) -> str:
        if not isinstance(source, dict):
            return ""
        parts: list[str] = []
        for key in ("title", "name", "description", "text", "canonical_text"):
            value = str(source.get(key) or "").strip()
            if value:
                parts.append(value)
        keywords = source.get("keywords")
        if isinstance(keywords, list):
            for keyword in keywords:
                value = str(keyword or "").strip()
                if value:
                    parts.append(value)
        return " | ".join(parts).lower()

    @staticmethod
    def _support_entity_key(entry: Any, *, fallback: str) -> str:
        if isinstance(entry, dict):
            for key in ("id", "model_id", "title", "name", "canonical_text"):
                value = str(entry.get(key) or "").strip()
                if value:
                    return value
        elif isinstance(entry, str):
            value = entry.strip()
            if value:
                return value
        return fallback

    def _relic_text(self, relic: Any) -> str:
        if isinstance(relic, dict):
            semantic = build_live_relic_semantic_text(relic)
            if semantic:
                return semantic
        return " | ".join(obs_common._get_relic_text_candidates(relic)) or str(relic or "")

    def _potion_text(self, potion: Any) -> str:
        if isinstance(potion, dict):
            semantic = build_live_potion_semantic_text(potion)
            if semantic:
                return semantic
            return str(potion.get("canonical_text") or potion.get("title") or "")
        return str(potion or "")

    def _shop_item_text(self, item: dict[str, Any]) -> str:
        item_kind = str(item.get("item_kind") or item.get("kind") or "").strip().lower()
        cost = item.get("cost")
        cost_text = ""
        if cost is not None:
            cost_text = f" | cost {int(obs_common._float(cost))}"
        if item_kind in {"card_removal", "remove", "removal", "purge"}:
            title = str(item.get("title") or "card removal").strip()
            used = " | used" if bool(item.get("used")) else ""
            return f"shop remove card | {title}{cost_text}{used}"
        if isinstance(item.get("card"), dict):
            return self._build_live_card_text(item.get("card"))
        if isinstance(item.get("relic"), dict):
            return self._relic_text(item.get("relic"))
        if isinstance(item.get("potion"), dict):
            return self._potion_text(item.get("potion"))
        return str(item.get("title") or item.get("canonical_text") or "")

    def _shop_item_entity_key(self, item: dict[str, Any]) -> str:
        item_kind = str(item.get("item_kind") or item.get("kind") or "").strip().lower()
        if item_kind in {"card_removal", "remove", "removal", "purge"}:
            title = item.get("title") or item.get("name") or item.get("index") or "remove"
            return f"shop:{item_kind}:{title}"
        for key in ("card", "relic", "potion"):
            payload = item.get(key)
            if isinstance(payload, dict):
                return self._support_entity_key(payload, fallback=f"shop:{key}")
        return self._support_entity_key(item, fallback="shop:item")

    def _reward_text(self, reward: dict[str, Any]) -> str:
        if isinstance(reward.get("card"), dict):
            return self._build_live_card_text(reward.get("card"))
        if isinstance(reward.get("relic"), dict):
            return self._relic_text(reward.get("relic"))
        if isinstance(reward.get("potion"), dict):
            return self._potion_text(reward.get("potion"))
        return str(reward.get("canonical_text") or reward.get("type") or reward.get("reward_type") or "")

    def _reward_entity_key(self, reward: dict[str, Any]) -> str:
        for key in ("card", "relic", "potion"):
            payload = reward.get(key)
            if isinstance(payload, dict):
                return self._support_entity_key(payload, fallback=f"reward:{key}")
        return self._support_entity_key(reward, fallback="reward")

    def _append_card_collection(
        self,
        world_entries: list[dict[str, Any]],
        numeric: np.ndarray,
        text: np.ndarray,
        mask: np.ndarray,
        token_type: str,
        *,
        entity_keys: list[Any] | None = None,
    ) -> None:
        owner_id = _owner_for_pile(token_type)
        for index in range(numeric.shape[0]):
            if mask[index] <= 0:
                continue
            row = np.zeros(TOKEN_NUMERIC_DIM, dtype=np.float32)
            row[: min(TOKEN_NUMERIC_DIM, numeric.shape[1])] = numeric[index][:TOKEN_NUMERIC_DIM]
            row[min(TOKEN_NUMERIC_DIM - 1, numeric.shape[1])] = min((index + 1) / max(numeric.shape[0], 1), 1.0)
            entity_key = f"{token_type}:{index}"
            if entity_keys is not None and index < len(entity_keys):
                card = entity_keys[index]
                if isinstance(card, dict):
                    entity_key = card.get("id") or card.get("title") or entity_key
                elif card:
                    entity_key = card
            world_entries.append(
                self._entry(
                    token_type,
                    row,
                    owner_id=owner_id,
                    entity_id=_stable_bucket(entity_key),
                    order_id=index + 1,
                    text_embedding=text[index],
                )
            )

    def _append_text_only_collection(
        self,
        world_entries: list[dict[str, Any]],
        text: np.ndarray,
        mask: np.ndarray,
        token_type: str,
        owner_id: int,
    ) -> None:
        for index in range(text.shape[0]):
            if mask[index] <= 0:
                continue
            row = np.zeros(TOKEN_NUMERIC_DIM, dtype=np.float32)
            row[0] = 1.0
            row[1] = min((index + 1) / max(text.shape[0], 1), 1.0)
            world_entries.append(
                self._entry(
                    token_type,
                    row,
                    owner_id=owner_id,
                    entity_id=_stable_bucket(f"{token_type}:{index}"),
                    order_id=index + 1,
                    text_embedding=text[index],
                )
            )

    def _encode_runtime_pile(self, obs: dict[str, Any], pile_key: str, fallback_key: str, *, limit: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        cards = self._runtime_cards(obs, pile_key, fallback_key)
        count = min(len(cards) if isinstance(cards, list) else 0, limit)
        numeric = np.zeros((count, obs_common.CARD_FEAT_DIM), dtype=np.float32)
        text = np.zeros((count, TEXT_DIM), dtype=np.float32)
        mask = np.zeros(count, dtype=np.float32)
        if count > 0:
            self._enc_card_collection(cards[:count], numeric, text, mask)
        return numeric, text, mask

    def _runtime_cards(self, obs: dict[str, Any], pile_key: str, fallback_key: str) -> list[Any]:
        combat = obs.get("combat") or {}
        cards: list[Any] = []
        pile = combat.get(pile_key)
        if pile_key == "hand":
            if isinstance(pile, dict):
                cards = pile.get("cards") or []
            elif isinstance(combat.get("hand"), list):
                cards = combat.get("hand") or []
        elif isinstance(pile, dict):
            cards = pile.get("cards") or []
        elif isinstance(pile, list):
            cards = pile
        if not cards and isinstance(combat.get(fallback_key), list):
            cards = combat.get(fallback_key) or []
        return cards if isinstance(cards, list) else []
