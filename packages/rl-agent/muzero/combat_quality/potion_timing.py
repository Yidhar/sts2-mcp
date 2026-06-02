"""Potion timing and identity helpers for MuZero combat policy.

Potion-use policy is tactical and highly context-sensitive.  Keeping these
helpers outside ``muzero.train`` makes it explicit that potion identity,
follow-up availability, and save/use timing are part of combat-quality policy.
"""

from __future__ import annotations

import re
from typing import Any

import numpy as np

from sts2_env.boss_mechanics import build_boss_mechanics_context
from sts2_env.observation_v2 import MAX_ACTIONS
from sts2_env.potion_profiles import (
    DEFAULT_EFFECT_PROFILE as _POTION_EFFECT_DEFAULT,
    all_potion_ids as _all_potion_ids,
    get_potion_profile as _get_potion_profile,
)
from muzero.combat_quality.potion_guard import (
    is_lucky_survival_potion_for_guard,
    potion_slot_from_action_for_guard,
)


class PotionTimingMixin:
    """Potion identity, preservation-value, and timing-quality helpers."""

    def _has_energy_followup(self, action_index: int, legal_actions: list[Any] | None, energy_after: float, mask_np: np.ndarray) -> bool:
        legal_count = min(len(legal_actions or []), MAX_ACTIONS, mask_np.shape[0])
        for other_idx in range(legal_count):
            if other_idx == action_index or mask_np[other_idx] <= 0:
                continue
            other = (legal_actions or [])[other_idx]
            if self._semantic_family(other) != "play_card":
                continue
            if self._is_x_cost_action(None, other_idx, other):
                continue
            source = self._action_source(other)
            cost_raw = other.get("card_cost", source.get("cost")) if isinstance(other, dict) else 0
            try:
                cost = max(float(cost_raw or 0.0), 0.0)
            except (TypeError, ValueError):
                cost = 0.0
            if cost <= energy_after + 1e-6 and self._action_immediate_impact(other) >= 4.0:
                return True
        return False

    def _has_resource_followup(
        self,
        action_index: int,
        legal_actions: list[Any] | None,
        energy_after: float,
        mask_np: np.ndarray,
        *,
        allow_cost_reduction: bool = False,
    ) -> bool:
        """Whether a resource/draw/energy potion can be converted this turn.

        We intentionally look for follow-up *cards*, not another potion.  The
        problematic behaviour is spending an energy/draw potion when the hand has
        no meaningful card action left, which creates "I used a potion" reward but
        no combat tempo.
        """
        legal_count = min(len(legal_actions or []), MAX_ACTIONS, mask_np.shape[0])
        for other_idx in range(legal_count):
            if other_idx == action_index or mask_np[other_idx] <= 0:
                continue
            other = (legal_actions or [])[other_idx]
            if self._semantic_family(other) != "play_card":
                continue
            if self._is_x_cost_action(None, other_idx, other):
                if energy_after <= 0.05:
                    continue
                return True
            cost = self._action_cost_value(other)
            if cost > energy_after + 1e-6 and not allow_cost_reduction:
                continue
            roles = self._action_roles(other)
            if (
                self._action_immediate_impact(other) >= 3.0
                or roles.intersection({"attack", "block", "draw", "debuff", "weak", "vulnerable", "scaling", "power", "resource", "energy"})
            ):
                return True
        return False

    def _extract_potion_effect_profile(self, action: Any, raw_obs: dict[str, Any] | None = None) -> dict[str, Any]:
        """Phase 4 of potion-timing-modeling-plan.md §8.

        Returns the merged {effect_profile, effect_family, timing_tags,
        target_scope, training_tags} resolved from the action's potion payload
        (bridge live values) with the Python registry as fallback.  Either side
        may be absent; the result always has every effect_profile slot.
        """
        if not isinstance(action, dict):
            return {
                "effect_profile": dict(_POTION_EFFECT_DEFAULT),
                "effect_family": [],
                "semantic_tags": [],
                "timing_tags": [],
                "training_tags": [],
                "target_scope": "",
            }
        def _usable_potion_text(value: Any) -> bool:
            text = str(value or "").strip()
            if not text:
                return False
            return text.lower() not in {"empty", "[empty]", "none", "null"}

        def _payload_has_identity(payload: Any) -> bool:
            if not isinstance(payload, dict):
                return False
            if bool(payload.get("empty")):
                return False
            for key in (
                "id",
                "potion_id",
                "model_id",
                "normalized_id",
                "title",
                "name",
                "label",
                "title_en",
                "title_zhs",
                "localized_title",
            ):
                if _usable_potion_text(payload.get(key)):
                    return True
            return isinstance(payload.get("effect_profile"), dict) and bool(payload.get("effect_profile"))

        def _slot_payload_from_raw(obs: dict[str, Any] | None, slot: int | None) -> dict[str, Any] | None:
            if slot is None or not isinstance(obs, dict):
                return None
            containers: list[Any] = []
            player = obs.get("player")
            if isinstance(player, dict):
                containers.append(player.get("potions"))
            combat = obs.get("combat")
            if isinstance(combat, dict):
                combat_player = combat.get("player")
                if isinstance(combat_player, dict):
                    containers.append(combat_player.get("potions"))
                containers.append(combat.get("potions"))
            for potions in containers:
                if not isinstance(potions, list) or not (0 <= int(slot) < len(potions)):
                    continue
                candidate = potions[int(slot)]
                if isinstance(candidate, dict) and _payload_has_identity(candidate):
                    return candidate
            return None

        potion = action.get("potion") if isinstance(action.get("potion"), dict) else None
        if not _payload_has_identity(potion):
            raw_potion = _slot_payload_from_raw(raw_obs, potion_slot_from_action_for_guard(action))
            if raw_potion is not None:
                # Live bridge use-potion actions can be payload-less
                # ("use_potion:0:0:self", potion_id=None), while the actual
                # potion identity is still present in raw_obs.player.potions.
                # Resolve by slot before registry lookup; otherwise Liquid
                # Memories/Blood Potion degrade to DEFAULT_PROFILE and are
                # mis-classified as no-followup/low-urgency.
                potion = raw_potion
        pid = ""
        if potion:
            pid = str(
                potion.get("id")
                or potion.get("potion_id")
                or potion.get("model_id")
                or potion.get("normalized_id")
                or ""
            ).strip()
        if not pid:
            pid = str(action.get("potion_id") or action.get("potion_model_id") or "").strip()
        registry = _get_potion_profile(pid) if pid else {}
        if not registry.get("id"):
            # Live bridge actions may expose only localized titles/labels
            # (observed: "鲜血药水", "液态记忆") while omitting
            # instance/id fields.  Treating those as DEFAULT_PROFILE silently
            # erases heal/resource semantics, which in turn makes survival
            # potions look like low-quality waste.  Resolve by title/name/label
            # as a fallback against the 64-entry static registry.
            def _norm(value: Any) -> str:
                text = str(value or "").strip().lower()
                if not text:
                    return ""
                return re.sub(r"[\s_:\-]+", "", text.replace("potion.", ""))

            candidates: set[str] = set()
            for container in (potion, action):
                if not isinstance(container, dict):
                    continue
                for key in (
                    "id",
                    "potion_id",
                    "model_id",
                    "normalized_id",
                    "title",
                    "name",
                    "label",
                    "title_en",
                    "title_zhs",
                    "localized_title",
                    "action_id",
                ):
                    value = container.get(key)
                    if value not in (None, ""):
                        candidates.add(_norm(value))
            if candidates:
                for candidate_pid in _all_potion_ids():
                    profile = _get_potion_profile(candidate_pid)
                    profile_candidates = {
                        _norm(candidate_pid),
                        _norm(profile.get("id")),
                        _norm(profile.get("title")),
                        _norm(profile.get("name")),
                        _norm(profile.get("title_en")),
                        _norm(profile.get("title_zhs")),
                    }
                    if candidates.intersection({v for v in profile_candidates if v}):
                        registry = profile
                        pid = str(profile.get("id") or candidate_pid)
                        break
                if not registry.get("id") and any(
                    token in candidates
                    for token in (
                        "幸运药剂",
                        "幸運藥劑",
                        "幸运补剂",
                        "幸運補劑",
                        "luckytonic",
                    )
                ):
                    # Live bridge/localization has emitted both "幸运补剂" and
                    # "幸运药剂" for Lucky Tonic.  The static registry only
                    # contains the former, so exact normalized-title matching
                    # can leave a title-only raw slot on DEFAULT_PROFILE.  That
                    # makes overflow-discard scoring treat a rare Buffer potion
                    # as unknown/common and can throw it away before Act1 boss.
                    registry = _get_potion_profile("POTION.LUCKY_TONIC")
                    pid = "POTION.LUCKY_TONIC"
        effect = dict(_POTION_EFFECT_DEFAULT)
        effect.update(registry.get("effect_profile") or {})
        if potion and isinstance(potion.get("effect_profile"), dict):
            effect.update({k: v for k, v in potion["effect_profile"].items() if v is not None})
        def _pick(field: str) -> Any:
            if potion is not None and potion.get(field):
                return potion.get(field)
            return registry.get(field) or []
        return {
            "effect_profile": effect,
            "effect_family": list(_pick("effect_family") or []),
            "semantic_tags": list(_pick("semantic_tags") or []),
            "timing_tags": list(_pick("timing_tags") or []),
            "training_tags": list(_pick("training_tags") or []),
            "target_scope": str((potion.get("target_scope") if potion else None) or registry.get("target_scope") or ""),
            "potion_id": pid,
            "rarity": str((potion.get("rarity") if potion else None) or registry.get("rarity") or ""),
        }

    def _discard_potion_keep_value(self, action: Any, raw_obs: dict[str, Any] | None = None) -> float:
        """Approximate how bad it is to discard this potion.

        This is *not* a use-now score.  It estimates future/survival value for
        forced overflow choices, so a potion can be low urgency to use this turn
        but still high value to keep (e.g. Fortifier, Blood Potion, Liquid
        Memories).  Higher means "protect; discard something else first".
        """
        if not isinstance(action, dict):
            return 0.0
        merged = self._extract_potion_effect_profile(action, raw_obs if isinstance(raw_obs, dict) else None)
        eff = merged.get("effect_profile") if isinstance(merged.get("effect_profile"), dict) else {}
        potion_id = str(merged.get("potion_id") or action.get("potion_id") or "").strip().upper()
        registry = _get_potion_profile(potion_id) if potion_id else {}

        def _text_from(value: Any) -> list[str]:
            if not isinstance(value, dict):
                return []
            out: list[str] = []
            for key in (
                "id",
                "potion_id",
                "model_id",
                "normalized_id",
                "title",
                "name",
                "label",
                "title_en",
                "title_zhs",
                "localized_title",
                "description",
                "summary",
                "action_id",
            ):
                raw = value.get(key)
                if raw not in (None, ""):
                    out.append(str(raw))
            return out

        potion_payload = action.get("potion") if isinstance(action.get("potion"), dict) else {}
        identity_text = " ".join(
            _text_from(action)
            + _text_from(potion_payload)
            + _text_from(registry if isinstance(registry, dict) else {})
        ).lower()
        tags = {
            str(x).strip().lower()
            for x in (
                list(merged.get("effect_family") or [])
                + list(merged.get("semantic_tags") or [])
                + list(merged.get("timing_tags") or [])
                + list(merged.get("training_tags") or [])
            )
            if str(x).strip()
        }
        rarity = str(merged.get("rarity") or (registry or {}).get("rarity") or "").strip().lower()
        score = 0.15
        if rarity == "rare":
            score += 0.16
        elif rarity == "uncommon":
            score += 0.08

        def _num(*keys: str) -> float:
            best = 0.0
            for key in keys:
                try:
                    best = max(best, float(eff.get(key) or 0.0))
                except (TypeError, ValueError):
                    continue
            return float(best)

        damage = _num("damage")
        block = _num("block")
        heal = _num("heal")
        draw = _num("draw", "cards_drawn", "card_draw")
        energy_gain = _num("energy_gain", "energy")
        discover = _num("discover_count", "generate_card_count")
        retrieve = _num("retrieve_from_discard")
        strength = _num("strength")
        dexterity = _num("dexterity")

        hp = 0.0
        max_hp = 0.0
        hp_valid = False
        try:
            hp, max_hp, hp_valid = self._player_hp_values(raw_obs if isinstance(raw_obs, dict) else None)
        except Exception:
            hp_valid = False
        hp_ratio = self._player_hp_ratio_from_values(hp, max_hp) if hp_valid else 1.0

        # High-value survival tools.  These are exactly the potions that should
        # survive overflow so late hallway/boss windows have outs.
        is_blood = "BLOOD" in potion_id or "鲜血" in identity_text or "heal" in tags or "回复" in identity_text
        is_block = (
            "BLOCK" in potion_id
            or block > 0.0
            or "block" in tags
            or "defense" in tags
            or "格挡" in identity_text
        )
        is_fortifier = "FORTIFIER" in potion_id or "固化" in identity_text or "三倍" in identity_text
        is_liquid_memories = (
            "LIQUID_MEMORIES" in potion_id
            or "液态记忆" in identity_text
            or retrieve > 0.0
            or "discard_pile" in tags
            or "tutor" in tags
        )
        is_life_saver = (
            "FAIRY" in potion_id
            or "GHOST" in potion_id
            or "INTANGIBLE" in potion_id
            or "prevent_lethal_tool" in tags
            or "emergency_heal" in tags
            or "survival" in tags
        )

        if is_life_saver:
            score += 1.25
        if is_lucky_survival_potion_for_guard(action, merged, raw_obs):
            # Lucky Tonic / 幸运补剂 is a rare Buffer/prevent-damage potion.
            # Keep it across overflow choices even when a compact bridge action
            # omitted profile tags and only raw_obs.player.potions has identity.
            score = max(score + 1.25, 1.85)
        if is_fortifier:
            score += 1.05
        if is_blood or heal > 0.0:
            score += 0.95
            if hp_ratio <= 0.60:
                score += 0.25
        if is_block:
            score += 0.80
        if is_liquid_memories:
            score += 0.95

        # Tactical boss/hallway finishers are worth keeping, but lower than
        # life-saving tools when overflow must discard something.
        if damage > 0.0 or "damage" in tags or "attack" in tags or "造成" in identity_text:
            score += 0.35 + min(0.25, max(damage, 0.0) / 80.0)
            if "lethal_tool" in tags:
                score += 0.18
        if discover > 0.0 or draw > 0.0 or energy_gain > 0.0 or "generate_cards" in tags or "hand_expansion_tool" in tags:
            score += 0.38
        if strength > 0.0 or "strength" in tags or "buff_strength" in tags or "力量" in identity_text:
            score += 0.28
        if dexterity > 0.0 or "dexterity" in tags or "buff_dexterity" in tags or "敏捷" in identity_text:
            score += 0.34
        if "scaling_setup" in tags:
            score += 0.08
        if "long_term_value" in tags:
            score += 0.12

        # Potions with no recognizable immediate/survival role are the safest
        # overflow victims.  Keep a tiny value so unknown rare potions are not
        # always thrown away before known common damage potions.
        return float(np.clip(score, 0.0, 2.0))

    def _potion_timing_profile(
        self,
        action: Any,
        index: int,
        encoded_obs: dict[str, Any] | None,
        raw_obs: dict[str, Any] | None,
        legal_actions: list[Any] | None,
        mask_np: np.ndarray,
        energy: float,
    ) -> dict[str, Any]:
        """Timing-aware potion affordance used by planner bias and aux metrics.

        Potion should not be a flat "positive" action.  It is urgent when it kills,
        prevents lethal/major damage, answers a fight mechanism (Kaiser facing /
        back attack), or converts immediately into a strong follow-up.  Otherwise
        it is usually deferable/savable so end_turn is not marked wasteful simply
        because a potion button is legal.
        """
        family = self._semantic_family(action)
        is_potion = family in {"use_potion", "potion"}
        default = {
            "is_potion": False,
            "available": False,
            "positive": False,
            "urgent": False,
            "deferable": False,
            "low_urgency": False,
            "save_recommended": False,
            "no_followup": False,
            "lethal": False,
            "prevent_lethal": False,
            "prevent_major_loss": False,
            "lethal_attacker_killable": False,
            "aoe_lethal_clear": False,
            "critical_hp_usable_survival_potion": False,
            "mechanism_answer": False,
            "facing_change": False,
            "overkill": False,
            "block_waste": False,
            "followup_available": False,
            "use_quality": 0.0,
            "waste_risk": 0.0,
            "damage": 0.0,
            "block": 0.0,
            "prevent_damage": 0.0,
            "hp": 0.0,
            "max_hp": 0.0,
            "hp_ratio": 0.0,
            "hp_valid": False,
            "resource_like": False,
            "retrieve_from_discard": 0.0,
            "retrieve_from_discard_like": False,
            "discard_count": 0,
            "retrieve_has_target": False,
            "free_play_like": False,
            "resource_survival_tool": False,
            "critical_hp_survival_tool": False,
            "buffer_like": False,
        }
        if not is_potion or not isinstance(action, dict):
            return default

        raw_obs = raw_obs if isinstance(raw_obs, dict) else self._current_raw_combat_obs()
        roles = self._action_roles(action)
        merged_potion = self._extract_potion_effect_profile(action, raw_obs)
        eff = merged_potion["effect_profile"]
        timing_tags = merged_potion["timing_tags"]
        effect_family = merged_potion["effect_family"]
        semantic_tags = merged_potion["semantic_tags"]
        target_scope = merged_potion["target_scope"]
        training_tags = merged_potion["training_tags"]
        potion_id_text = str(merged_potion.get("potion_id") or "").strip()
        potion_id_upper = potion_id_text.upper()
        tag_tokens = {
            str(x).strip().lower()
            for x in list(effect_family) + list(semantic_tags) + list(timing_tags) + list(training_tags)
            if str(x).strip()
        }
        potion_payload = action.get("potion") if isinstance(action.get("potion"), dict) else {}
        identity_text = " ".join(
            str(v)
            for container in (action, potion_payload)
            if isinstance(container, dict)
            for k, v in container.items()
            if k
            in {
                "id",
                "potion_id",
                "model_id",
                "normalized_id",
                "title",
                "name",
                "label",
                "title_en",
                "title_zhs",
                "localized_title",
                "description",
                "summary",
                "action_id",
            }
            and v not in (None, "")
        ).lower()
        amplify_block_like = bool(
            "FORTIFIER" in potion_id_upper
            or "amplify_block" in tag_tokens
            or "triple_block" in tag_tokens
            or "requires_block_in_play" in tag_tokens
            or "固化" in identity_text
            or "三倍" in identity_text
            or "triple" in identity_text and "block" in identity_text
        )

        damage = max(
            float(eff.get("damage") or 0.0),
            self._action_numeric_value(action, ("damage", "total_damage", "attack_damage", "preview_damage")),
            self._action_metric(action, "damage"),
            self._action_metric(action, "total_damage"),
        )
        block = max(
            float(eff.get("block") or 0.0),
            self._action_numeric_value(action, ("block", "total_block", "preview_block")),
            self._action_metric(action, "block"),
            self._action_metric(action, "total_block"),
        )
        profile_heal = float(eff.get("heal") or 0.0)
        action_heal = max(
            self._action_numeric_value(action, ("heal", "healing", "hp_gain")),
            self._action_metric(action, "heal"),
        )
        heal = max(profile_heal, action_heal)
        prevent_damage = max(
            float(eff.get("prevent_damage") or 0.0),
            self._action_numeric_value(action, ("prevent_damage", "buffer", "buffer_stacks")),
            self._action_metric(action, "prevent_damage"),
        )
        draw = max(
            float(eff.get("draw") or 0.0),
            self._action_numeric_value(action, ("draw", "cards_drawn", "card_draw")),
            self._action_metric(action, "draw"),
        )
        energy_gain = max(
            float(eff.get("energy_gain") or 0.0),
            self._action_numeric_value(action, ("energy", "energy_gain", "gain_energy")),
            self._action_metric(action, "energy"),
            self._action_metric(action, "energy_gain"),
        )
        hits = max(self._action_numeric_value(action, ("hits", "times", "repeat")), self._action_metric(action, "hits"), 1.0 if damage > 0.0 else 0.0)
        weak_v = float(eff.get("weak") or 0.0)
        vuln_v = float(eff.get("vulnerable") or 0.0)
        poison_v = float(eff.get("poison") or 0.0)
        debuff = bool(
            roles.intersection({"debuff", "weak", "vulnerable", "poison"})
            or weak_v > 0.0 or vuln_v > 0.0 or poison_v > 0.0
            or self._action_numeric_value(action, ("weak", "vulnerable", "poison")) > 0.0
        )
        buffer_like = bool(
            prevent_damage > 0.0
            or "buffer" in tag_tokens
            or "prevent_damage" in tag_tokens
            or "LUCKY_TONIC" in potion_id_upper
            or "lucky tonic" in identity_text
            or "lucky_tonic" in identity_text
            or "幸运补剂" in identity_text
            or "幸运药剂" in identity_text
        )
        gen_card_v = float(eff.get("generate_card_count") or 0.0)
        discover_v = float(eff.get("discover_count") or 0.0)
        retrieve_v = float(eff.get("retrieve_from_discard") or 0.0)
        retrieve_from_discard_like = bool(
            retrieve_v > 0.0
            or "LIQUID_MEMORIES" in potion_id_upper
            or ("discard_pile" in tag_tokens and "tutor" in tag_tokens)
        )
        random_potion_resource_like = bool(
            "ENTROPIC_BREW" in potion_id_upper
            or "fill_potion_slots" in tag_tokens
            or "refill_potion_slots" in tag_tokens
            or ("potion" in tag_tokens and ("random" in tag_tokens or "random_outcome" in tag_tokens))
            or bool(eff.get("fill_potion_slots"))
            or bool(eff.get("random_outcome"))
            or "混沌药水" in identity_text
            or "entropic" in identity_text
        )
        free_play_like = bool(
            retrieve_from_discard_like
            or float(eff.get("set_cost_zero") or 0.0) > 0.0
            or "set_cost_zero" in tag_tokens
            or "free_play" in tag_tokens
            or "free" in tag_tokens
        )
        upgrade_v = float(eff.get("upgrade_hand") or 0.0)
        dup_v = float(eff.get("duplicate_next") or 0.0)
        replace_v = float(eff.get("replace_or_transform_hand") or 0.0)
        resource_like = bool(
            energy_gain > 0.0 or draw > 0.0
            or gen_card_v > 0.0 or discover_v > 0.0 or retrieve_v > 0.0
            or retrieve_from_discard_like
            or random_potion_resource_like
            or roles.intersection({"resource", "energy", "draw"})
        )
        incoming, current_block, hp = self._incoming_damage_pressure(raw_obs)
        hp, max_hp, hp_valid = self._player_hp_values(raw_obs)
        discard_count = self._discard_pile_count_from_raw(raw_obs if isinstance(raw_obs, dict) else None)
        retrieve_has_target = bool(discard_count > 0)
        new_option_resource_like = bool(
            draw > 0.0
            or gen_card_v > 0.0
            or discover_v > 0.0
            or random_potion_resource_like
            or (retrieve_from_discard_like and retrieve_has_target)
        )
        hand_transform_like = bool(upgrade_v > 0.0 or dup_v > 0.0 or replace_v > 0.0)
        requires_followup_profile = bool(
            eff.get("requires_followup")
            or "requires_followup" in tag_tokens
        )
        if amplify_block_like:
            # Fortifier's "requires_followup" metadata historically meant
            # "requires block already in play", not "requires another card
            # after the potion resolves".  Treating it as a post-use follow-up
            # made good Fortifier turns (current block > 0) look like
            # no-followup waste exactly in late Act1 survival fights.
            requires_followup_profile = False
        followup_dependent = bool(resource_like or hand_transform_like or requires_followup_profile)
        long_term_like = bool(eff.get("long_term_value") or "long_term_value" in timing_tags)
        passive_or_triggered = bool(eff.get("passive_or_triggered"))

        threat_gap = max(0.0, incoming - current_block)
        amplify_block_added = 0.0
        amplify_block_noop = False
        if amplify_block_like:
            # Fortifier/固化 is not a flat block potion: it triples *current*
            # block.  The static metadata necessarily has block=0, which made
            # the timing model think "use now" was harmless/neutral even when
            # current block was 0.  In reality that is a pure no-op that still
            # consumes the potion; when current block > 0 the immediate added
            # block is 2x current block.
            amplify_block_added = max(0.0, 2.0 * float(current_block))
            block = max(float(block), float(amplify_block_added))
            amplify_block_noop = bool(current_block <= 0.05)
        heal_fraction_of_max_hp = False
        if (
            hp_valid
            and max_hp > 1.0
            and 0.0 < profile_heal <= 1.0
            and action_heal <= profile_heal + 1e-6
            and (
                "heal" in {str(x).strip().lower() for x in effect_family}
                or "heal" in {str(x).strip().lower() for x in semantic_tags}
                or "survival" in {str(x).strip().lower() for x in semantic_tags}
                or "heal" in {str(x).strip().lower() for x in training_tags}
            )
        ):
            # Curated potion profiles encode fractional max-HP healing as a
            # ratio (Blood Potion: heal=0.20).  The timing model needs the
            # absolute combat value; reading 0.20 as 0.2 HP made Blood Potion
            # look like a useless low-urgency action exactly when it should
            # preserve boss-fight survival margin.
            heal = float(profile_heal * max_hp)
            heal_fraction_of_max_hp = True
        target_hp = self._target_enemy_hp(action, raw_obs)
        aoe = bool(
            roles.intersection({"aoe", "all_enemies"})
            or bool(eff.get("aoe"))
            or str(action.get("target_scope") or target_scope).lower() in {"all_enemies", "aoe", "allenemies", "allcreatures"}
        )
        lethal = bool(damage > 0.0 and target_hp > 0.0 and damage >= target_hp)
        overkill = bool(
            damage > 0.0
            and target_hp > 0.0
            and not aoe
            and damage > target_hp + max(6.0, 0.50 * target_hp)
        )
        lethal_threat_window = bool(hp > 0.0 and threat_gap >= max(hp, 1.0))
        alive_enemy_hps: list[float] = []
        try:
            alive_enemy_hps = [float(v) for v in self._alive_enemy_hp_values(raw_obs) if float(v) > 0.0]
        except Exception:
            alive_enemy_hps = []
        lethal_attacker_killable = bool(
            lethal_threat_window
            and damage > 0.0
            and target_hp > 0.0
            and damage >= target_hp
            and not aoe
        )
        aoe_lethal_clear = bool(
            lethal_threat_window
            and aoe
            and damage > 0.0
            and (
                (alive_enemy_hps and damage >= max(alive_enemy_hps))
                or (not alive_enemy_hps and target_hp > 0.0 and damage >= target_hp)
            )
        )

        defensive = bool(block > 0.0 or heal > 0.0 or debuff or buffer_like)
        prevents_lethal_now = bool(defensive or lethal_attacker_killable or aoe_lethal_clear)
        prevent_lethal = bool(lethal_threat_window and prevents_lethal_now)
        prevent_major_loss = bool(threat_gap >= max(8.0, 0.25 * max(hp, 1.0)) and defensive)
        block_waste = bool((block > 0.0 and threat_gap <= 0.05) or amplify_block_noop)
        encounter_tier = self._combat_encounter_tier_from_raw(raw_obs)
        hp_ratio = self._player_hp_ratio_from_values(hp, max_hp) if hp_valid else 0.0
        critical_hp_survival_tool = bool(
            hp_valid
            and encounter_tier in {"elite", "boss"}
            and hp_ratio <= 0.35
            and (heal > 0.0 or block > 0.0 or buffer_like or (retrieve_from_discard_like and retrieve_has_target))
        )
        near_death_margin = max(3.0, 0.05 * max(max_hp, 1.0))
        near_death_after_incoming = bool(
            hp_valid
            and threat_gap > 0.05
            and hp > 0.0
            and (hp - threat_gap) <= near_death_margin
        )
        if (heal > 0.0 or buffer_like) and critical_hp_survival_tool:
            # A Blood/fruit-like heal at critical HP in elite/boss combat is a
            # survival-margin action even when the current intent is idle.  Same
            # for Buffer/Lucky Tonic: it is a one-hit death-prevention reserve,
            # not a cosmetic long-term buff in late elite/boss combat.
            prevent_major_loss = True

        energy_after = max(0.0, float(energy) + energy_gain)
        followup_available = (
            self._has_resource_followup(
                index,
                legal_actions,
                energy_after,
                mask_np,
                allow_cost_reduction=bool(free_play_like),
            )
            if followup_dependent
            else False
        )
        resource_survival_tool = bool(
            (
                retrieve_from_discard_like
                and hp_valid
                and encounter_tier in {"elite", "boss"}
                and (
                    hp_ratio <= 0.15
                    or threat_gap >= max(1.0, hp - 1.0)
                    or (hp_ratio <= 0.35 and threat_gap >= max(6.0, 0.25 * max(hp, 1.0)))
                    or (encounter_tier == "boss" and hp_ratio <= 0.50)
                )
            )
            or (
                near_death_after_incoming
                and (
                    new_option_resource_like
                    or (energy_gain > 0.0 and followup_available)
                    or defensive
                )
                and not (retrieve_from_discard_like and not retrieve_has_target)
            )
        )
        if retrieve_from_discard_like and encounter_tier in {"elite", "boss"} and retrieve_has_target:
            # Retrieve/free-play potions (Liquid Memories) expose their true
            # follow-up only after the potion resolves into a discard-pile
            # selection.  This is only true when the discard pile has a target;
            # an empty-discard Liquid Memories on boss turn 1 is a pure waste.
            followup_available = True
        if resource_survival_tool and retrieve_from_discard_like and retrieve_has_target:
            # Liquid Memories/retrieve potions create their real follow-up only
            # after use (a discard-pile selection plus a free card).  A current
            # legal-action scan can therefore see only {potion, End Turn}; do
            # not mark that as "no follow-up" in a lethal boss/elite window.
            followup_available = True
            prevent_major_loss = True
        elif resource_survival_tool and retrieve_from_discard_like and not retrieve_has_target:
            # A retrieve-only potion cannot create a free survival card from an
            # empty discard pile.  Without this check boss turn-1 Liquid
            # Memories was treated as a high-value survival action and consumed
            # before it had any target.
            resource_survival_tool = False
        elif resource_survival_tool and not retrieve_from_discard_like:
            # Near-death stochastic/resource potions (Entropic Brew, Skill/
            # Attack/Power Potion, Swift Potion) expose their true follow-up
            # only after the potion resolves.  In a frame such as 23HP facing
            # 22 incoming, preserving a rare potion through death is worse than
            # rolling for an immediate answer, so fail open and do not mark it
            # as no-followup/save-only.
            followup_available = True
            prevent_major_loss = True
        no_followup = bool(followup_dependent and not followup_available)
        critical_hp_usable_survival_potion = bool(
            hp_valid
            and threat_gap > 0.0
            and (
                (
                    hp_ratio <= 0.10
                    and (
                        defensive
                        or damage > 0.0
                        or critical_hp_survival_tool
                        or resource_survival_tool
                        or ((energy_gain > 0.0 or draw > 0.0) and not no_followup)
                    )
                )
                or (
                    near_death_after_incoming
                    and (
                        defensive
                        or critical_hp_survival_tool
                        or resource_survival_tool
                        or (new_option_resource_like and not no_followup)
                    )
                )
            )
        )

        facing_change = False
        mechanism_answer = False
        kaiser_risk = 0.0
        try:
            facing_change = bool(self._is_kaiser_facing_change_action(action, raw_obs))
            boss_ctx = build_boss_mechanics_context(raw_obs) if isinstance(raw_obs, dict) else {}
            kaiser_risk = self._kaiser_back_attack_risk_from_context(boss_ctx, raw_obs)
        except Exception:
            kaiser_risk = 0.0
        if facing_change:
            mechanism_answer = True
        elif kaiser_risk > 0.05:
            # Damage-only potion is only a mechanism answer if it kills or is a
            # real pressure action; arbitrary potion use must not satisfy Kaiser
            # defense metrics.
            mechanism_answer = bool(
                lethal
                or (damage >= 12.0 and target_hp <= 0.0)
                or (target_hp > 0.0 and damage >= min(target_hp, max(12.0, 0.35 * target_hp)))
                or block > 0.0
                or debuff
            )

        high_damage = bool(damage >= 18.0 or (target_hp > 0.0 and damage >= max(12.0, 0.35 * target_hp)))

        use_quality = 0.08
        if lethal:
            use_quality += 0.85
        elif damage > 0.0:
            use_quality += min(0.34, damage / 55.0)
            if high_damage:
                use_quality += 0.16
        if prevent_lethal:
            use_quality += 0.95
        elif prevent_major_loss:
            use_quality += 0.52
        elif block > 0.0 and threat_gap > 0.0:
            use_quality += 0.35 * min(block / max(threat_gap, 1.0), 1.0)
        if heal > 0.0:
            use_quality += 0.25 if hp_ratio <= 0.55 else 0.10
            if critical_hp_survival_tool and encounter_tier in {"elite", "boss"}:
                use_quality += 0.45
        if buffer_like and threat_gap > 0.05:
            # Lucky Tonic / Buffer does not show up as block, but in a boss death
            # window it prevents the next damaging hit.  Without this explicit
            # term it was classified as low-urgency and saved through lethal
            # EndTurn decisions.
            use_quality += 0.42 + min(0.28, float(threat_gap) / max(float(hp), 1.0))
            if encounter_tier in {"elite", "boss"} and (prevent_lethal or prevent_major_loss or hp_ratio <= 0.35):
                use_quality += 0.25
        if debuff and incoming > 0.0:
            use_quality += 0.30
        if mechanism_answer:
            use_quality += 0.62
        if followup_dependent:
            use_quality += 0.36 if followup_available else -0.42
        if resource_survival_tool:
            use_quality += 0.65
        if critical_hp_usable_survival_potion:
            # At <=10% HP under incoming pressure, any potion with immediate
            # survival/tempo value should not be saved through a death frame.
            # This explicitly covers damage potions that can kill the attacker
            # and localized/key survival potions whose raw profile is sparse.
            use_quality += 0.45
        if encounter_tier in {"elite", "boss"} and (lethal or prevent_major_loss or mechanism_answer or high_damage):
            use_quality += 0.12

        waste_risk = 0.0
        if no_followup:
            waste_risk += 0.45
        if block_waste:
            waste_risk += 0.35
        if overkill and not mechanism_answer:
            waste_risk += 0.25
        low_threat = threat_gap <= 2.0 and not prevent_major_loss and not prevent_lethal
        save_recommended = bool(
            low_threat
            and hp_ratio >= 0.55
            and not lethal
            and not mechanism_answer
            and not (encounter_tier in {"elite", "boss"} and high_damage)
        )
        if save_recommended:
            waste_risk += 0.42 if encounter_tier in {"weak", "normal"} else 0.22

        # Phase 4 of potion-timing-modeling-plan.md §8.6 / §8.7: hand-transform
        # potions are good iff hand has high-value targets and a usable
        # follow-up window; otherwise they should defer.
        hand_size = 0
        try:
            player = (raw_obs or {}).get("player") if isinstance(raw_obs, dict) else None
            hand_size = len(player.get("hand") or []) if isinstance(player, dict) else 0
        except Exception:
            hand_size = 0
        hand_context_good = bool(hand_transform_like and hand_size >= 3)
        hand_context_bad = bool(hand_transform_like and hand_size <= 1)
        if hand_transform_like and hand_context_good:
            use_quality += 0.30
        if hand_transform_like and hand_context_bad:
            waste_risk += 0.30
            save_recommended = True
        if long_term_like and not (urgent_threat := (prevent_lethal or prevent_major_loss)):
            # Long-term-only potion in combat sandbox: don't reward as immediate
            # combat positive; it is mostly a save-or-use-out-of-combat candidate.
            use_quality = max(0.0, use_quality - 0.20)
            save_recommended = True
        if passive_or_triggered:
            # Passive/triggered potions don't have an immediate "use now" payoff.
            use_quality = max(0.0, use_quality - 0.10)

        use_quality = float(np.clip(use_quality - waste_risk, 0.0, 1.0))
        waste_risk = float(np.clip(waste_risk, 0.0, 1.0))
        # save_value: how much value remains if the agent saves the potion for
        # later. Approximated as inverse-of-use-quality + bonus for defensive/
        # mechanism tools whose later utility is high.
        save_value = float(np.clip(
            (1.0 - use_quality) * 0.6
            + (0.25 if any(t in timing_tags for t in ("prevent_lethal_tool", "prevent_major_loss_tool", "survival_tool", "boss_survival_tool", "lethal_tool", "mechanism_answer_candidate")) else 0.0)
            + (0.20 if long_term_like else 0.0)
            - (0.30 if (lethal or prevent_lethal or prevent_major_loss or mechanism_answer) else 0.0),
            0.0, 1.0,
        ))
        urgent = bool(
            lethal
            or prevent_lethal
            or mechanism_answer
            or critical_hp_usable_survival_potion
            or (prevent_major_loss and use_quality >= 0.45)
            or use_quality >= 0.62
        )
        low_urgency = bool((use_quality < 0.22) or (waste_risk > use_quality + 0.10 and not urgent))
        positive = bool(urgent or use_quality >= 0.32)
        deferable = bool(not urgent and (low_urgency or save_recommended or no_followup or block_waste or overkill))
        requires_followup = bool(followup_dependent)

        return {
            "is_potion": True,
            "available": True,
            "potion_id": merged_potion.get("potion_id", ""),
            "rarity": merged_potion.get("rarity", ""),
            "effect_family": list(effect_family),
            "semantic_tags": list(semantic_tags),
            "timing_tags": list(timing_tags),
            "training_tags": list(training_tags),
            "damage": float(damage),
            "block": float(block),
            "prevent_damage": float(prevent_damage),
            "buffer_like": bool(buffer_like),
            "amplify_block_like": bool(amplify_block_like),
            "amplify_block_added": float(amplify_block_added),
            "amplify_block_noop": bool(amplify_block_noop),
            "energy_gain": float(energy_gain),
            "draw": float(draw),
            "heal": float(heal),
            "heal_fraction_of_max_hp": bool(heal_fraction_of_max_hp),
            "weak": float(weak_v),
            "vulnerable": float(vuln_v),
            "poison": float(poison_v),
            "debuff": bool(debuff),
            "incoming": float(incoming),
            "current_block": float(current_block),
            "hp": float(hp),
            "max_hp": float(max_hp),
            "hp_ratio": float(hp_ratio),
            "hp_valid": bool(hp_valid),
            "threat_gap": float(threat_gap),
            "target_hp": float(target_hp),
            "aoe": bool(aoe),
            "lethal": bool(lethal),
            "prevent_lethal": bool(prevent_lethal),
            "prevent_major_loss": bool(prevent_major_loss),
            "lethal_attacker_killable": bool(lethal_attacker_killable),
            "aoe_lethal_clear": bool(aoe_lethal_clear),
            "critical_hp_usable_survival_potion": bool(critical_hp_usable_survival_potion),
            "mechanism_answer": bool(mechanism_answer),
            "facing_change": bool(facing_change),
            "followup_available": bool(followup_available),
            "overkill": bool(overkill),
            "block_waste": bool(block_waste),
            "no_followup": bool(no_followup),
            "save_recommended": bool(save_recommended),
            "low_urgency": bool(low_urgency),
            "urgency": float(use_quality),
            "use_quality": float(use_quality),
            "waste_risk": float(waste_risk),
            "save_value": float(save_value),
            "positive": bool(positive),
            "urgent": bool(urgent),
            "deferable": bool(deferable),
            "requires_followup": bool(requires_followup),
            "hand_context_good": bool(hand_context_good),
            "hand_context_bad": bool(hand_context_bad),
            "long_term_value": bool(long_term_like),
            "passive_or_triggered": bool(passive_or_triggered),
            "hand_transform": bool(hand_transform_like),
            "resource_like": bool(resource_like),
            "random_potion_resource_like": bool(random_potion_resource_like),
            "new_option_resource_like": bool(new_option_resource_like),
            "retrieve_from_discard": float(retrieve_v),
            "retrieve_from_discard_like": bool(retrieve_from_discard_like),
            "discard_count": int(discard_count),
            "retrieve_has_target": bool(retrieve_has_target),
            "free_play_like": bool(free_play_like),
            "resource_survival_tool": bool(resource_survival_tool),
            "critical_hp_survival_tool": bool(critical_hp_survival_tool),
            "near_death_after_incoming": bool(near_death_after_incoming),
            "near_death_margin": float(near_death_margin),
        }
