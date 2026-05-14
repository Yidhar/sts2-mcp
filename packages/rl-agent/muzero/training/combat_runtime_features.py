"""Runtime combat/action feature helpers for MuZeroTrainer.

This mixin owns bridge-payload parsing, combat action feature adapters, and
encounter-mechanic compatibility wrappers that used to live directly in
``muzero.train``.  Keep policy semantics in ``muzero.strategy`` /
``muzero.combat_quality`` and keep this file adapter-light.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import torch

from combat_snapshot_dataset import infer_encounter_tier
from muzero.combat_quality import card_block_waste_profile as compute_card_block_waste_profile
from muzero.strategy import action_features as action_feature_policy
from muzero.strategy.encounters import insatiable as insatiable_strategy
from muzero.strategy.encounters import kaiser as kaiser_strategy
from sts2_env.boss_mechanics import build_boss_mechanics_context
from sts2_env.observation_v2 import DECISION_DOMAINS, MAX_ACTIONS


class CombatRuntimeFeatureMixin:
    """Combat/action observation adapters shared by self-play and guards."""

    @staticmethod
    def _with_decision_domain(
        obs: dict[str, Any] | None,
        domain: str,
    ) -> dict[str, Any] | None:
        if not isinstance(obs, dict) or domain not in DECISION_DOMAINS:
            return obs
        patched = dict(obs)
        index = DECISION_DOMAINS.index(domain)
        existing = obs.get("decision_domain")
        if isinstance(existing, torch.Tensor):
            vector = torch.zeros_like(existing)
            if vector.ndim == 0:
                vector = torch.zeros((len(DECISION_DOMAINS),), dtype=torch.float32, device=existing.device)
            if vector.shape[-1] == len(DECISION_DOMAINS):
                vector[..., index] = 1.0
            else:
                vector = torch.zeros((len(DECISION_DOMAINS),), dtype=existing.dtype, device=existing.device)
                vector[index] = 1.0
            patched["decision_domain"] = vector
            return patched
        dtype = np.asarray(existing).dtype if existing is not None else np.float32
        if dtype == np.dtype("O"):
            dtype = np.float32
        vector_np = np.zeros((len(DECISION_DOMAINS),), dtype=dtype)
        vector_np[index] = 1.0
        patched["decision_domain"] = vector_np.astype(np.float32, copy=False)
        return patched

    @staticmethod
    def _looks_like_combat_decision(
        obs: dict[str, Any] | None,
        legal_actions: list[Any] | None,
        info: dict[str, Any] | None = None,
    ) -> bool:
        """Infer combat even when the encoded decision_domain vector is stale.

        The bridge often reports the actionable combat screen as phase
        ``actions``.  The old encoder only treated phase == ``combat`` as
        combat, so combat-sandbox decisions were stored as build.  This helper
        uses the collector-side info/action surface, not just the encoded
        vector, so direct combat policy cannot be silently bypassed.
        """

        info = info if isinstance(info, dict) else {}
        phase = str(info.get("phase") or "").strip().lower()
        if phase == "combat":
            return True
        if str(info.get("episode_mode") or "").strip().lower() == "combat_sandbox":
            return True
        transition_state = info.get("transition_state") if isinstance(info.get("transition_state"), dict) else {}
        transition_combat = transition_state.get("combat") if isinstance(transition_state, dict) else None
        if isinstance(transition_combat, dict) and transition_combat:
            if phase in {"actions", "combat", "card_selection", "settling", ""}:
                return True

        raw_combat = obs.get("combat") if isinstance(obs, dict) else None
        if isinstance(raw_combat, dict) and raw_combat:
            return True

        combat_kinds = {
            "play_card",
            "use_potion",
            "discard_potion",
            "combat",
            "combat_select",
            "combat_select_card",
        }
        combat_action_ids = {"end_turn"}
        for action in (legal_actions or [])[:MAX_ACTIONS]:
            if not isinstance(action, dict):
                continue
            kind = str(action.get("kind") or "").strip().lower()
            action_id = str(action.get("action_id") or "").strip().lower()
            surface = str(action.get("surface") or "").strip().lower()
            if kind in combat_kinds or action_id in combat_action_ids:
                return True
            if action_id.startswith(("play_card", "use_potion", "discard_potion", "combat_select")):
                return True
            if surface == "combat":
                return True
            semantic = action.get("semantic")
            if isinstance(semantic, dict):
                semantic_domain = str(semantic.get("domain") or "").strip().lower()
                semantic_family = str(semantic.get("family") or "").strip().lower()
                if semantic_domain == "combat" or semantic_family in {"end_turn", "play_card", "use_potion"}:
                    return True
        return False

    def _resolve_acting_decision_domain(
        self,
        obs: dict[str, Any] | None,
        legal_actions: list[Any] | None,
        info: dict[str, Any] | None,
    ) -> tuple[str, str, bool]:
        encoded_domain = self._decision_domain_name(obs)
        if self._looks_like_combat_decision(obs, legal_actions, info):
            return "combat", encoded_domain, encoded_domain != "combat"
        if encoded_domain not in DECISION_DOMAINS:
            return "build", encoded_domain, encoded_domain != "build"
        return encoded_domain, encoded_domain, False

    @staticmethod
    def _semantic_family(action: Any) -> str:
        return action_feature_policy.semantic_family(action)

    @staticmethod
    def _action_roles(action: Any) -> set[str]:
        return action_feature_policy.action_roles(action)

    @staticmethod
    def _action_metric(action: Any, key: str) -> float:
        return action_feature_policy.action_metric(action, key)

    @classmethod
    def _action_immediate_impact(cls, action: Any) -> float:
        return action_feature_policy.action_immediate_impact(action)

    @staticmethod
    def _boss_context_max(context: dict[str, Any], key: str) -> float:
        if not isinstance(context, dict):
            return 0.0
        vals: list[float] = []
        player_state = context.get("player_state") if isinstance(context.get("player_state"), dict) else {}
        if key in player_state:
            try:
                vals.append(float(player_state.get(key) or 0.0))
            except (TypeError, ValueError):
                pass
        enemy_states = context.get("enemy_states_by_index")
        if isinstance(enemy_states, list):
            for state in enemy_states:
                if isinstance(state, dict) and key in state:
                    try:
                        vals.append(float(state.get(key) or 0.0))
                    except (TypeError, ValueError):
                        pass
        return max(vals) if vals else 0.0

    @staticmethod
    def _boss_context_encounter_key(context: dict[str, Any] | None, raw_obs: dict[str, Any] | None = None) -> str:
        """Return normalized encounter key from boss context/raw obs.

        Keep this separate from generic back-attack mechanics: several non-Kaiser
        enemies can expose incoming-damage multipliers or back-attack-like state.
        Metrics under the ``kaiser_*`` namespace must only be emitted for the
        Kaiser Crab encounter.
        """
        if isinstance(context, dict):
            value = context.get("encounter_key") or context.get("encounter_id") or context.get("encounter")
            if value:
                return str(value).strip().lower()
        if isinstance(raw_obs, dict):
            value = raw_obs.get("encounter_id") or raw_obs.get("encounter")
            if value:
                return str(value).strip().lower()
            combat = raw_obs.get("combat")
            if isinstance(combat, dict):
                value = combat.get("encounter_id") or combat.get("encounter")
                if value:
                    return str(value).strip().lower()
        return ""

    @classmethod
    def _is_kaiser_encounter_context(
        cls,
        context: dict[str, Any] | None,
        raw_obs: dict[str, Any] | None = None,
    ) -> bool:
        return "kaiser" in cls._boss_context_encounter_key(context, raw_obs)

    @classmethod
    def _is_insatiable_encounter_context(
        cls,
        context: dict[str, Any] | None,
        raw_obs: dict[str, Any] | None = None,
    ) -> bool:
        return "insatiable" in cls._boss_context_encounter_key(context, raw_obs)

    @classmethod
    def _kaiser_back_attack_risk_from_context(
        cls,
        context: dict[str, Any] | None,
        raw_obs: dict[str, Any] | None = None,
    ) -> float:
        """Kaiser-only back-attack risk for metrics/bias.

        Do not include ``incoming_damage_multiplier_norm`` here: the bridge/boss
        context normalizes the default multiplier 1.0 to 0.5, which polluted every
        non-Kaiser encounter as ``kaiser_back_attack_risk=0.5``.  The Kaiser signal
        should be gated by encounter and derived from explicit back-attack fields.
        """
        if not isinstance(context, dict) or not cls._is_kaiser_encounter_context(context, raw_obs):
            return 0.0
        return max(
            cls._boss_context_max(context, "primary_back_attack_risk"),
            cls._boss_context_max(context, "primary_back_attack_active"),
            cls._boss_context_max(context, "back_attack_risk"),
            cls._boss_context_max(context, "back_attack_active"),
        )

    @staticmethod
    def _obs_energy(obs: dict[str, Any] | None) -> float:
        if not isinstance(obs, dict):
            return 0.0
        combat = obs.get("combat")
        if isinstance(combat, dict):
            try:
                return max(float(combat.get("energy") or 0.0), 0.0)
            except (TypeError, ValueError):
                pass
        scalars = obs.get("scalars")
        try:
            scalars_np = scalars.detach().cpu().numpy() if isinstance(scalars, torch.Tensor) else np.asarray(scalars)
            if scalars_np.ndim >= 1 and scalars_np.shape[0] > 29:
                return max(float(scalars_np.reshape(-1)[29]) * 10.0, 0.0)
        except Exception:
            return 0.0
        return 0.0

    def _current_raw_combat_obs(self) -> dict[str, Any] | None:
        raw_obs = getattr(getattr(self.env, "unwrapped", self.env), "_last_obs_raw", None)
        return raw_obs if isinstance(raw_obs, dict) else None

    def _combat_energy(self, encoded_obs: dict[str, Any] | None = None, raw_obs: dict[str, Any] | None = None) -> float:
        # Prefer live/raw combat energy.  The encoded scalar slot has changed across
        # observation versions and previously made boss_combat/energy_mean stay at 0.
        energy = self._obs_energy(raw_obs)
        if energy > 1e-6:
            return energy
        return self._obs_energy(encoded_obs)

    @staticmethod
    def _action_source(action: Any) -> dict[str, Any]:
        return action_feature_policy.action_source(action)


    def _is_zero_cost_action(self, action: Any) -> bool:
        return action_feature_policy.is_zero_cost_action(action)

    def _is_positive_combat_action(self, action: Any) -> bool:
        return action_feature_policy.is_positive_combat_action(
            action,
            is_facing_change_action=self._is_facing_change_action,
        )

    def _action_text(self, action: Any) -> str:
        return action_feature_policy.action_text(action)

    def _is_exhausting_action(self, action: Any) -> bool:
        return action_feature_policy.is_exhausting_action(action)

    def _is_ethereal_action(self, action: Any) -> bool:
        return action_feature_policy.is_ethereal_action(action)

    def _is_retain_action(self, action: Any) -> bool:
        return action_feature_policy.is_retain_action(action)

    def _incoming_damage_pressure(self, raw_obs: dict[str, Any] | None) -> tuple[float, float, float]:
        if not isinstance(raw_obs, dict):
            return 0.0, 0.0, 0.0
        combat = raw_obs.get("combat") if isinstance(raw_obs.get("combat"), dict) else {}
        player = raw_obs.get("player") if isinstance(raw_obs.get("player"), dict) else {}
        if not player and isinstance(combat.get("player"), dict):
            player = combat.get("player")
        block = self._safe_float(player.get("block")) if isinstance(player, dict) else 0.0
        hp, _max_hp, _hp_valid = self._player_hp_values(raw_obs)
        incoming = 0.0
        enemies = combat.get("enemies") if isinstance(combat, dict) else []
        if isinstance(enemies, list):
            for enemy in enemies:
                if not isinstance(enemy, dict):
                    continue
                intent = enemy.get("intent") if isinstance(enemy.get("intent"), dict) else {}
                for key in ("total_damage", "damage", "intent_damage", "attack_damage"):
                    incoming += max(self._safe_float(intent.get(key)), self._safe_float(enemy.get(key)))
                    if incoming > 0.0:
                        break
        return incoming, block, hp

    @staticmethod
    def _is_meaningful_block_urgent(
        *,
        block: float,
        threat_gap: float,
        current_hp: float,
        incoming: float,
        encounter_tier: str | None = None,
    ) -> bool:
        """Return true only when block answers meaningful HP pressure.

        The old classifier treated any uncovered incoming damage as urgent:
        ``block > 0 and incoming > current_block``.  That made a 50 HP player
        spend energy on Defend against 1-2 incoming instead of attacking,
        which increased pre-boss attrition.  Keep block urgent for lethal,
        high-damage, low-HP, elite, and boss windows; otherwise let it remain
        positive but non-urgent so End Turn/quality logic can prefer progress.
        """

        try:
            block_value = float(block or 0.0)
            threat_value = float(threat_gap or 0.0)
            hp_value = float(current_hp or 0.0)
            incoming_value = float(incoming or 0.0)
        except (TypeError, ValueError):
            return False
        if block_value <= 0.0 or threat_value <= 0.0 or incoming_value <= 0.0:
            return False
        if hp_value > 0.0 and threat_value >= max(1.0, hp_value - 1.0):
            return True
        if threat_value >= 6.0:
            return True
        if hp_value > 0.0 and threat_value >= 0.20 * hp_value:
            return True
        if hp_value > 0.0 and hp_value <= 20.0 and threat_value >= 4.0:
            return True
        if hp_value > 0.0 and hp_value <= 12.0 and threat_value >= 3.0:
            return True
        if str(encounter_tier or "").lower() in {"elite", "boss"} and threat_value >= 4.0:
            return True
        return False

    def _card_block_waste_profile(
        self,
        action: Any,
        raw_obs: dict[str, Any] | None,
        *,
        mechanism_urgent: bool = False,
    ) -> dict[str, Any]:
        """Adapter from bridge action payloads to pure no-pressure block logic.

        Keep policy details in ``muzero.combat_quality.block_waste``.  This thin
        method only collects the typed semantic metrics already exposed through
        ``_action_metric`` and current incoming-damage state.
        """

        if not isinstance(action, dict):
            return {}
        source = self._action_source(action)
        card_type = str(
            (action.get("card_type") if isinstance(action.get("card_type"), str) else None)
            or source.get("type")
            or ""
        ).strip().lower()
        metric_keys = (
            "damage",
            "total_damage",
            "block",
            "total_block",
            "draw",
            "cards_drawn",
            "energy",
            "energy_gain",
            "typed_gain_energy",
            "typed_gain_energy_amount",
            "heal",
            "hp_loss",
            "hp_cost",
            "weak",
            "vulnerable",
            "poison",
            "strength",
            "dexterity",
            "artifact",
            "stun",
            "facing_change",
            "typed_debuff",
            "typed_apply_debuff",
            "typed_apply_power",
            "typed_apply_buff",
            "typed_modifies_hand",
            "typed_modify_cost",
            "typed_upgrade_hand",
            "typed_upgrade_cards",
            "typed_discard_cards",
            "typed_transform_cards",
            "typed_copy_cards",
            "typed_add_modifier",
            "typed_add_keyword",
            "typed_set_replay",
            "typed_retain_cards",
            "typed_card_state_mutation",
            "typed_requires_followup",
            "typed_strategic_skip_if_no_followup",
            "typed_no_draw",
            "typed_future_penalty",
            "typed_consumes_future_resource",
        )
        metrics = {key: self._action_metric(action, key) for key in metric_keys}
        incoming, current_block, _hp = self._incoming_damage_pressure(raw_obs)
        return compute_card_block_waste_profile(
            action,
            family=self._semantic_family(action),
            roles=self._action_roles(action),
            card_type=card_type,
            incoming=incoming,
            current_block=current_block,
            damage=max(metrics.get("damage", 0.0), metrics.get("total_damage", 0.0)),
            block=max(metrics.get("block", 0.0), metrics.get("total_block", 0.0)),
            draw=max(metrics.get("draw", 0.0), metrics.get("cards_drawn", 0.0)),
            energy_gain=max(metrics.get("energy", 0.0), metrics.get("energy_gain", 0.0)),
            heal=metrics.get("heal", 0.0),
            hp_loss=max(metrics.get("hp_loss", 0.0), metrics.get("hp_cost", 0.0)),
            metrics=metrics,
            mechanism_urgent=mechanism_urgent,
        )

    def _player_hp_values(self, raw_obs: dict[str, Any] | None) -> tuple[float, float, bool]:
        """Return (hp, max_hp, valid) without silently treating missing HP as full.

        Several hard guards use HP as a safety gate.  Missing bridge fields must
        be fail-open (do not override the policy), not interpreted as either
        full HP or critically low HP.  ``valid`` therefore requires both hp and
        max_hp to be present, numeric, finite, and positive.
        """
        if not isinstance(raw_obs, dict):
            return 0.0, 0.0, False
        combat = raw_obs.get("combat") if isinstance(raw_obs.get("combat"), dict) else {}
        player = raw_obs.get("player") if isinstance(raw_obs.get("player"), dict) else {}
        if not player and isinstance(combat.get("player"), dict):
            player = combat.get("player")
        if not isinstance(player, dict):
            return 0.0, 0.0, False

        hp_raw = player.get("hp")
        if hp_raw is None:
            hp_raw = player.get("current_hp")
        if hp_raw is None:
            hp_raw = player.get("currentHealth")
        max_hp_raw = player.get("max_hp")
        if max_hp_raw is None:
            max_hp_raw = player.get("maxHealth")
        if max_hp_raw is None:
            max_hp_raw = player.get("max_hp_raw")
        try:
            hp = float(hp_raw)
            max_hp = float(max_hp_raw)
        except (TypeError, ValueError):
            return 0.0, 0.0, False
        valid = bool(np.isfinite(hp) and np.isfinite(max_hp) and hp > 0.0 and max_hp > 0.0)
        return (hp if np.isfinite(hp) else 0.0), (max_hp if np.isfinite(max_hp) else 0.0), valid

    @staticmethod
    def _discard_pile_count_from_raw(raw_obs: dict[str, Any] | None) -> int:
        """Best-effort discard-pile size from bridge/raw combat observations.

        Liquid Memories / retrieve-from-discard potions are only real follow-up
        actions when there is something in the discard pile.  Several bridge
        revisions place this information in slightly different containers, so
        keep the parser permissive and monotonic rather than depending on one
        schema spelling.
        """
        if not isinstance(raw_obs, dict):
            return 0

        best = 0
        sources: list[Any] = [raw_obs]
        combat = raw_obs.get("combat")
        player = raw_obs.get("player")
        if isinstance(combat, dict):
            sources.append(combat)
            nested_player = combat.get("player")
            if isinstance(nested_player, dict):
                sources.append(nested_player)
        if isinstance(player, dict):
            sources.append(player)

        list_keys = (
            "discard_pile",
            "discard_cards",
            "discard",
            "discardPile",
            "discardCards",
            "discardPileCards",
        )
        count_keys = (
            "discard_count",
            "discard_pile_count",
            "discardPileCount",
            "discard_size",
            "discardSize",
        )
        for src in sources:
            if not isinstance(src, dict):
                continue
            for key in list_keys:
                val = src.get(key)
                if isinstance(val, list):
                    best = max(best, len(val))
            for key in count_keys:
                try:
                    val = src.get(key)
                    if val is not None:
                        best = max(best, int(float(val)))
                except (TypeError, ValueError):
                    continue
        return int(max(best, 0))

    def _action_numeric_value(self, action: Any, keys: tuple[str, ...] | list[str] | set[str]) -> float:
        """Read an unnormalised numeric action/card/potion metric.

        Bridge payloads are not perfectly uniform: card previews may live under
        ``action.semantic``, direct action fields, or the nested card/potion
        source.  Potion timing must not depend on one bridge revision's field
        placement, so this helper scans every action-local container and returns
        the largest positive value.
        """
        if not isinstance(action, dict):
            return 0.0
        semantic = action.get("semantic") if isinstance(action.get("semantic"), dict) else {}
        source = self._action_source(action)
        containers: list[Any] = [semantic, action, source]
        for key in ("card", "potion", "item", "target"):
            value = action.get(key)
            if isinstance(value, dict):
                containers.append(value)
                nested_card = value.get("card")
                nested_potion = value.get("potion")
                if isinstance(nested_card, dict):
                    containers.append(nested_card)
                if isinstance(nested_potion, dict):
                    containers.append(nested_potion)

        best = 0.0
        for container in containers:
            if not isinstance(container, dict):
                continue
            for key in keys:
                value = container.get(key)
                if isinstance(value, (list, dict)):
                    continue
                try:
                    best = max(best, float(value or 0.0))
                except (TypeError, ValueError):
                    continue
        return float(best)

    def _action_cost_value(self, action: Any) -> float:
        if not isinstance(action, dict):
            return 0.0
        source = self._action_source(action)
        for container in (action, source):
            if not isinstance(container, dict):
                continue
            for key in ("card_cost", "cost", "energy_cost", "base_cost"):
                if key not in container:
                    continue
                value = container.get(key)
                if isinstance(value, str) and value.strip().upper() == "X":
                    return 0.0
                try:
                    return max(float(value or 0.0), 0.0)
                except (TypeError, ValueError):
                    continue
        return 0.0

    def _combat_enemies_from_raw(self, raw_obs: dict[str, Any] | None) -> list[dict[str, Any]]:
        combat = raw_obs.get("combat") if isinstance(raw_obs, dict) and isinstance(raw_obs.get("combat"), dict) else {}
        for key in ("enemies", "monsters", "creatures"):
            enemies = combat.get(key)
            if isinstance(enemies, list):
                return [enemy for enemy in enemies if isinstance(enemy, dict)]
        return []

    def _alive_enemy_hp_values(self, raw_obs: dict[str, Any] | None) -> list[float]:
        values: list[float] = []
        for enemy in self._combat_enemies_from_raw(raw_obs):
            if bool(enemy.get("is_dead") or enemy.get("dead")):
                continue
            if enemy.get("alive") is False:
                continue
            hp = max(
                self._safe_float(enemy.get("hp")),
                self._safe_float(enemy.get("current_hp")),
                self._safe_float(enemy.get("health")),
            )
            if hp > 0.0:
                values.append(float(hp))
        return values

    def _target_enemy_hp(self, action: Any, raw_obs: dict[str, Any] | None) -> float:
        """Best-effort target HP for lethal/overkill potion timing.

        Falls back to the lowest alive enemy HP because untargeted damage potions
        are often aimed by bridge defaults at a killable target.
        """
        if not isinstance(action, dict):
            vals = self._alive_enemy_hp_values(raw_obs)
            return min(vals) if vals else 0.0
        target = action.get("target") if isinstance(action.get("target"), dict) else {}
        if isinstance(target, dict):
            hp = max(
                self._safe_float(target.get("hp")),
                self._safe_float(target.get("current_hp")),
                self._safe_float(target.get("health")),
            )
            if hp > 0.0:
                return float(hp)

        target_id = self._action_target_combat_id(action)
        target_name = ""
        for container in (action, target):
            if isinstance(container, dict):
                target_name = str(container.get("target_name") or container.get("name") or target_name or "").strip().lower()
        for enemy in self._combat_enemies_from_raw(raw_obs):
            enemy_id = enemy.get("combat_id", enemy.get("id"))
            try:
                if target_id is not None and enemy_id is not None and int(enemy_id) == int(target_id):
                    return max(
                        self._safe_float(enemy.get("hp")),
                        self._safe_float(enemy.get("current_hp")),
                        self._safe_float(enemy.get("health")),
                    )
            except (TypeError, ValueError):
                pass
            enemy_name = str(enemy.get("name") or enemy.get("title") or "").strip().lower()
            if target_name and enemy_name and target_name == enemy_name:
                return max(
                    self._safe_float(enemy.get("hp")),
                    self._safe_float(enemy.get("current_hp")),
                    self._safe_float(enemy.get("health")),
                )

        vals = self._alive_enemy_hp_values(raw_obs)
        return min(vals) if vals else 0.0

    def _is_frantic_escape_action(self, action: Any) -> bool:
        """Return true for the real Frantic Escape card/action id.

        Local generated source facts identify:
        - card id: ``CARD.FRANTIC_ESCAPE``
        - normalized id: ``frantic_escape``
        - power applied by the card: ``POWER.SANDPIT_POWER`` /
          ``SandpitPower``

        The Chinese title fallback is kept only for older bridge payloads that
        expose localized titles before typed ids.
        """
        if not isinstance(action, dict):
            return False

        source = self._action_source(action)
        card = action.get("card") if isinstance(action.get("card"), dict) else {}
        profile = card.get("card_effect_profile") if isinstance(card.get("card_effect_profile"), dict) else {}
        derived = profile.get("derived_view") if isinstance(profile.get("derived_view"), dict) else {}

        candidates: list[str] = []
        for container in (action, source, card, profile, derived):
            if not isinstance(container, dict):
                continue
            for key in (
                "id",
                "card_id",
                "model_id",
                "normalized_id",
                "title",
                "name",
                "title_en",
                "title_zhs",
                "class_name",
                "kind",
                "action_id",
                "label",
            ):
                value = container.get(key)
                if value not in (None, ""):
                    candidates.append(str(value))

        joined = " | ".join(candidates).strip().lower()
        compact = joined.replace(" ", "_")
        compact_no_underscore = compact.replace("_", "")
        return (
            "card.frantic_escape" in compact
            or "frantic_escape" in compact
            or "franticescape" in compact_no_underscore
            or "狂乱逃离" in joined
        )

    def _insatiable_sandpit_turns_from_context(
        self,
        boss_ctx: dict[str, Any] | None,
        raw_obs: dict[str, Any] | None = None,
    ) -> float:
        # Prefer the raw amount emitted by boss_mechanics; fall back to the
        # dense-observation normalized value for replay/back-compat frames.
        raw = self._boss_context_max(boss_ctx or {}, "sandpit_turns")
        if raw > 0.0:
            return float(raw)
        norm = self._boss_context_max(boss_ctx or {}, "sandpit_turns_norm")
        if norm > 0.0:
            return float(norm * 10.0)
        return 0.0

    def _is_action_confirmed_lethal(self, action: Any, raw_obs: dict[str, Any] | None) -> bool:
        if not isinstance(action, dict):
            return False
        damage = max(
            self._action_metric(action, "damage"),
            self._action_metric(action, "total_damage"),
            self._action_numeric_value(
                action,
                ("damage", "total_damage", "attack_damage", "preview_damage", "expected_damage"),
            ),
        )
        target_hp = self._target_enemy_hp(action, raw_obs)
        return bool(damage > 0.0 and target_hp > 0.0 and damage >= target_hp)

    @staticmethod
    def _insatiable_escape_cycle_risk(
        sandpit_turns: float,
        hand_count: float,
        draw_count: float,
        discard_count: float,
        total_count: float,
    ) -> float:
        """Approximate risk that Frantic Escape will miss the lethal countdown.

        At Sandpit 0 the player is already dead; at 1 the escape card must be
        played now if it is available.  For 2-3 we bias harder when the card is
        not in draw/hand because cycling may not expose it before the clock hits
        zero.
        """
        if sandpit_turns <= 0.0 or total_count <= 0.0:
            return 0.0
        if hand_count > 0.0:
            return 0.0
        if sandpit_turns <= 1.0:
            return 1.0
        if sandpit_turns < 3.0:
            return 0.85 if draw_count <= 0.0 else 0.45
        if sandpit_turns < 4.0:
            return 0.60 if discard_count > 0.0 else 0.35
        return 0.15

    def _combat_encounter_tier_from_raw(self, raw_obs: dict[str, Any] | None) -> str:
        encounter_id = ""
        room_type = ""
        if isinstance(raw_obs, dict):
            encounter_candidates: list[Any] = [
                raw_obs.get("encounter_id"),
                raw_obs.get("encounter"),
                raw_obs.get("room_model"),
                raw_obs.get("roomModel"),
                raw_obs.get("room_id"),
                raw_obs.get("roomId"),
            ]
            room_type = str(
                raw_obs.get("room_type")
                or raw_obs.get("roomType")
                or raw_obs.get("current_room_type")
                or ""
            ).strip()
            run = raw_obs.get("run") if isinstance(raw_obs.get("run"), dict) else {}
            if isinstance(run, dict):
                encounter_candidates.extend(
                    [
                        run.get("encounter_id"),
                        run.get("encounter"),
                        run.get("room_model"),
                        run.get("roomModel"),
                        run.get("room_id"),
                        run.get("roomId"),
                    ]
                )
                if not room_type:
                    room_type = str(
                        run.get("room_type")
                        or run.get("roomType")
                        or run.get("current_room_type")
                        or ""
                    ).strip()
            combat = raw_obs.get("combat") if isinstance(raw_obs.get("combat"), dict) else {}
            if isinstance(combat, dict):
                encounter_candidates.extend(
                    [
                        combat.get("encounter_id"),
                        combat.get("encounter"),
                        combat.get("room_model"),
                        combat.get("roomModel"),
                    ]
                )
                if not room_type:
                    room_type = str(combat.get("room_type") or combat.get("roomType") or "").strip()
            for candidate in encounter_candidates:
                text = str(candidate or "").strip()
                if text:
                    encounter_id = text
                    break
            if not encounter_id:
                try:
                    boss_ctx = build_boss_mechanics_context(raw_obs)
                    encounter_id = str(boss_ctx.get("encounter_key") or "").strip()
                except Exception:
                    encounter_id = ""
            boss_markers = (
                "LAGAVULIN_MATRIARCH",
                "KAISER_CRAB",
                "CEREMONIAL_BEAST",
                "THE_KIN",
                "INSATIABLE",
                "KNOWLEDGE_DEMON",
                # Live bridge / older dataset boss identifiers may arrive as
                # raw monster ids (for example ``room_model=MONSTER.SOUL_FYSH``)
                # instead of encounter ids ending in ``_BOSS``.  If we pass
                # those through ``infer_encounter_tier`` unchanged it returns
                # ``normal`` and the boss-only survival potion guards silently
                # stay dormant.  Keep this marker list broad but still
                # boss-specific so ordinary encounters do not inherit boss
                # potion/block overrides.
                "SOUL_FYSH",
                "WATERFALL_GIANT",
                "DOORMAKER",
                "TEST_SUBJECT",
                "QUEEN",
            )
            if encounter_id:
                encounter_text = str(encounter_id or "").upper()
                if any(marker in encounter_text for marker in boss_markers):
                    encounter_id = "ENCOUNTER.BRIDGE_INFERRED_BOSS"
            if not encounter_id and isinstance(combat, dict):
                # Some live bridge combat observations omit the encounter key
                # on the raw state even though downstream diagnostics can
                # recover it.  Boss hard guards must still fire in that shape;
                # fall back to distinctive boss monster ids/names.
                enemies = combat.get("enemies")
                if isinstance(enemies, list):
                    for enemy in enemies:
                        if not isinstance(enemy, dict):
                            continue
                        enemy_text = " ".join(
                            str(enemy.get(key) or "")
                            for key in ("id", "name", "model_id", "modelId", "monster_id", "monsterId")
                        ).upper()
                        if any(marker in enemy_text for marker in boss_markers):
                            encounter_id = "ENCOUNTER.BRIDGE_INFERRED_BOSS"
                            break
        try:
            return str(
                infer_encounter_tier(encounter_id, room_type=room_type or None)
            ).strip().lower()
        except Exception:
            return "normal"

    # Potion identity/timing helpers live in ``muzero.combat_quality.potion_timing``.

    def _classify_positive_combat_action(
        self,
        action: Any,
        index: int,
        encoded_obs: dict[str, Any] | None,
        raw_obs: dict[str, Any] | None,
        legal_actions: list[Any] | None,
        mask_np: np.ndarray,
        energy: float,
    ) -> dict[str, Any]:
        family = self._semantic_family(action)
        if family in {"use_potion", "potion"}:
            profile = self._potion_timing_profile(
                action,
                index,
                encoded_obs,
                raw_obs,
                legal_actions,
                mask_np,
                energy,
            )
            return {
                "positive": bool(profile.get("positive", False)),
                "urgent": bool(profile.get("urgent", False)),
                "deferable": bool(profile.get("deferable", False)),
                "exhausting": False,
                "deferable_exhaust": False,
                "ethereal_urgent": False,
                "energy_without_followup": bool(profile.get("no_followup", False)),
                "x_cost_zero": False,
                "potion_available": bool(profile.get("available", False)),
                "potion_urgent": bool(profile.get("urgent", False)),
                "potion_low_urgency": bool(profile.get("low_urgency", False)),
                "potion_save_recommended": bool(profile.get("save_recommended", False)),
                "potion_no_followup": bool(profile.get("no_followup", False)),
                "potion_lethal": bool(profile.get("lethal", False)),
                "potion_prevent_lethal": bool(profile.get("prevent_lethal", False)),
                "potion_mechanism_answer": bool(profile.get("mechanism_answer", False)),
                "potion_overkill": bool(profile.get("overkill", False)),
                "potion_block_waste": bool(profile.get("block_waste", False)),
                "potion_use_quality": float(profile.get("use_quality", 0.0) or 0.0),
                "potion_waste_risk": float(profile.get("waste_risk", 0.0) or 0.0),
                "potion_save_value": float(profile.get("save_value", 0.0) or 0.0),
                "potion_hand_context_good": bool(profile.get("hand_context_good", False)),
                "potion_hand_context_bad": bool(profile.get("hand_context_bad", False)),
                "potion_long_term_value": bool(profile.get("long_term_value", False)),
                "potion_requires_followup": bool(profile.get("requires_followup", False)),
                "potion_effect_family": list(profile.get("effect_family", []) or []),
                "potion_id": str(profile.get("potion_id", "") or ""),
                "potion_facing_change": bool(profile.get("facing_change", False)),
            }
        positive = self._is_positive_combat_action(action)
        roles = self._action_roles(action)
        exhausting = family == "play_card" and self._is_exhausting_action(action)
        ethereal = family == "play_card" and self._is_ethereal_action(action)
        retain = family == "play_card" and self._is_retain_action(action)
        x_cost_zero = family == "play_card" and self._is_x_cost_action(encoded_obs, index, action) and energy <= 0.05
        damage = max(self._action_metric(action, "damage"), self._action_metric(action, "total_damage"))
        block = max(self._action_metric(action, "block"), self._action_metric(action, "total_block"))
        energy_gain = max(self._action_metric(action, "energy"), self._action_metric(action, "energy_gain"))
        hp_loss = max(self._action_metric(action, "hp_loss"), self._action_metric(action, "hp_cost"))
        typed_requires_followup = self._action_metric(action, "typed_requires_followup") > 0.0
        typed_strategic_skip_if_no_followup = self._action_metric(action, "typed_strategic_skip_if_no_followup") > 0.0
        typed_modify_cost = self._action_metric(action, "typed_modify_cost") > 0.0
        typed_no_draw = self._action_metric(action, "typed_no_draw") > 0.0
        typed_future_penalty = self._action_metric(action, "typed_future_penalty") > 0.0
        typed_consumes_future_resource = self._action_metric(action, "typed_consumes_future_resource") > 0.0
        typed_card_state_mutation = self._action_metric(action, "typed_card_state_mutation") > 0.0
        typed_modifies_hand = self._action_metric(action, "typed_modifies_hand") > 0.0
        setup_followup_dependent = bool(
            family == "play_card"
            and (
                typed_requires_followup
                or typed_strategic_skip_if_no_followup
                or typed_modify_cost
                or typed_no_draw
                or typed_future_penalty
                or typed_consumes_future_resource
            )
        )
        future_or_no_draw = bool(typed_no_draw or typed_future_penalty or typed_consumes_future_resource)
        card_state_setup = bool(typed_card_state_mutation or typed_modifies_hand)
        incoming, current_block, current_hp = self._incoming_damage_pressure(raw_obs)
        encounter_tier = self._combat_encounter_tier_from_raw(raw_obs if isinstance(raw_obs, dict) else None)
        threat_gap = max(0.0, incoming - current_block)
        impact = self._action_immediate_impact(action)
        energy_after = max(0.0, energy - self._action_cost_value(action) + energy_gain)
        energy_without_followup = bool(family == "play_card" and energy_gain > 0.0 and not self._has_energy_followup(index, legal_actions, energy_after, mask_np))
        setup_followup_available = bool(
            setup_followup_dependent
            and self._has_resource_followup(
                index,
                legal_actions,
                energy_after,
                mask_np,
                allow_cost_reduction=typed_modify_cost,
            )
        )
        followup_missing = bool(setup_followup_dependent and not setup_followup_available)

        mechanism_urgent = False
        try:
            # Do not let Kaiser-specific helpers globally classify every
            # block/debuff action as a mechanism answer.  `_is_kaiser_risk_
            # handling_action` intentionally treats block as a valid answer
            # *inside* Kaiser back-attack windows; outside that encounter it
            # would make no-threat Defend look urgent and recreate the global
            # useless-block pathology.
            boss_ctx = build_boss_mechanics_context(raw_obs) if isinstance(raw_obs, dict) else {}
            kaiser_risk = self._kaiser_back_attack_risk_from_context(boss_ctx, raw_obs)
            mechanism_urgent = bool(
                kaiser_risk > 0.05
                and (
                    self._is_kaiser_facing_change_action(action, raw_obs)
                    or self._is_kaiser_risk_handling_action(action, raw_obs)
                )
            )
        except Exception:
            mechanism_urgent = False

        card_block_profile: dict[str, Any] = {}
        if family == "play_card":
            try:
                card_block_profile = self._card_block_waste_profile(
                    action,
                    raw_obs,
                    mechanism_urgent=mechanism_urgent,
                )
            except Exception:
                card_block_profile = {}
        card_block_waste = bool(card_block_profile.get("block_waste", False))
        card_pure_block = bool(card_block_profile.get("pure_block", False))
        # ``no_damage_pressure`` from the block-waste profile is a context bit:
        # it simply means enemies are not presenting HP damage right now.  It is
        # not by itself a bad action.  Attacking under no incoming damage is
        # often exactly what Act 1 hallway fights need.  Keep the historical
        # ``card_no_damage_pressure`` gauge as the bad/no-progress case only,
        # and expose the raw context separately for diagnostics.
        card_no_damage_pressure_context = bool(card_block_profile.get("no_damage_pressure", False))
        card_no_damage_pressure_bad = bool(card_no_damage_pressure_context and card_pure_block)

        urgent = bool(
            positive
            and not x_cost_zero
            and not card_block_waste
            and (
                mechanism_urgent
                or ethereal
                or self._is_meaningful_block_urgent(
                    block=block,
                    threat_gap=threat_gap,
                    current_hp=current_hp,
                    incoming=incoming,
                    encounter_tier=encounter_tier,
                )
                or (damage >= 12.0)
                or (roles.intersection({"weak", "vulnerable", "debuff"}) and incoming > 0.0)
                or (
                    setup_followup_dependent
                    and setup_followup_available
                    and not energy_without_followup
                    and (energy_gain > 0.0 or typed_modify_cost or self._action_metric(action, "draw") > 0.0)
                )
                or (
                    not exhausting
                    and not energy_without_followup
                    and not followup_missing
                    and not retain
                    and not card_pure_block
                    and not (future_or_no_draw and impact < 8.0)
                    and impact >= 3.0
                )
            )
        )
        deferable = bool(
            positive
            and not urgent
            and family == "play_card"
            and (
                card_block_waste
                or exhausting
                or retain
                or x_cost_zero
                or energy_without_followup
                or followup_missing
                or (future_or_no_draw and not setup_followup_available)
                or (card_state_setup and not setup_followup_available and impact < 6.0)
                or (hp_loss > 0.0 and energy_without_followup)
            )
        )
        return {
            "positive": bool(positive),
            "urgent": bool(urgent),
            "deferable": bool(deferable),
            "exhausting": bool(exhausting),
            "deferable_exhaust": bool(deferable and exhausting),
            "ethereal_urgent": bool(urgent and ethereal),
            "energy_without_followup": bool(energy_without_followup),
            "x_cost_zero": bool(x_cost_zero),
            "typed_requires_followup": bool(typed_requires_followup),
            "typed_strategic_skip_if_no_followup": bool(typed_strategic_skip_if_no_followup),
            "typed_modify_cost": bool(typed_modify_cost),
            "typed_no_draw": bool(typed_no_draw),
            "typed_future_penalty": bool(typed_future_penalty),
            "typed_consumes_future_resource": bool(typed_consumes_future_resource),
            "typed_card_state_mutation": bool(typed_card_state_mutation),
            "typed_modifies_hand": bool(typed_modifies_hand),
            "card_block_waste": bool(card_block_waste),
            "card_pure_block": bool(card_pure_block),
            "card_no_damage_pressure": bool(card_no_damage_pressure_bad),
            "card_no_damage_pressure_context": bool(card_no_damage_pressure_context),
            "card_block_threat_gap": float(card_block_profile.get("threat_gap", 0.0) or 0.0),
            "setup_followup_dependent": bool(setup_followup_dependent),
            "setup_followup_available": bool(setup_followup_available),
            "followup_missing": bool(followup_missing),
        }

    # Kaiser / Insatiable encounter helpers live under ``muzero.strategy.encounters``.
    # Keep these methods as compatibility wrappers for existing tests and call
    # sites while moving the actual policy semantics out of the trainer.

    def _is_facing_change_action(self, action: Any) -> bool:
        return kaiser_strategy.is_facing_change_action(
            action,
            semantic_family_fn=self._semantic_family,
            action_source_fn=self._action_source,
        )

    @staticmethod
    def _normalize_side(value: Any) -> str:
        return kaiser_strategy.normalize_side(value)

    def _combat_player_facing(self, raw_obs: Any | None = None) -> str:
        obs = raw_obs if isinstance(raw_obs, dict) else self._current_raw_combat_obs()
        return kaiser_strategy.combat_player_facing(obs)

    @staticmethod
    def _action_target_combat_id(action: Any) -> int | None:
        return kaiser_strategy.action_target_combat_id(action)

    @staticmethod
    def _enemy_back_attack_position(enemy: Any) -> str:
        return kaiser_strategy.enemy_back_attack_position(enemy)

    def _action_target_back_attack_position(self, action: Any, raw_obs: Any | None = None) -> str:
        obs = raw_obs if isinstance(raw_obs, dict) else self._current_raw_combat_obs()
        return kaiser_strategy.action_target_back_attack_position(action, obs)

    def _action_target_side(self, action: Any, raw_obs: Any | None = None) -> str:
        obs = raw_obs if isinstance(raw_obs, dict) else self._current_raw_combat_obs()
        return kaiser_strategy.action_target_side(action, obs)

    def _is_targeted_enemy_action(self, action: Any) -> bool:
        return kaiser_strategy.is_targeted_enemy_action(
            action,
            semantic_family_fn=self._semantic_family,
        )

    def _action_changes_facing_toward_target(self, action: Any, raw_obs: Any | None = None) -> bool:
        obs = raw_obs if isinstance(raw_obs, dict) else self._current_raw_combat_obs()
        return kaiser_strategy.action_changes_facing_toward_target(
            action,
            obs,
            semantic_family_fn=self._semantic_family,
        )

    def _is_kaiser_facing_change_action(self, action: Any, raw_obs: Any | None = None) -> bool:
        obs = raw_obs if isinstance(raw_obs, dict) else self._current_raw_combat_obs()
        return kaiser_strategy.is_kaiser_facing_change_action(
            action,
            obs,
            semantic_family_fn=self._semantic_family,
            action_source_fn=self._action_source,
        )

    # --------------------------------------------------------------------- #
    # P0-3 / P0-4 hard guards (recovery 2026-05-06).
    #
    # Soft action_quality bias has already been applied at MCTS root prior
    # time, but the model still picks End Turn or non-facing actions under
    # confirmed Kaiser back-attack risk and ignores Frantic Escape on
    # countdown<=1.  The recovery doc requires a hard override on top of
    # the bias: when the post-search action_idx violates an invariant,
    # replace it with a safety candidate before ``env.step`` consumes it.
    # --------------------------------------------------------------------- #

    def _find_kaiser_facing_candidates(
        self,
        legal_actions: list[Any] | None,
        mask_np: np.ndarray,
        raw_obs: Any | None,
    ) -> list[int]:
        obs = raw_obs if isinstance(raw_obs, dict) else self._current_raw_combat_obs()
        return kaiser_strategy.find_facing_candidates(
            legal_actions,
            mask_np,
            obs,
            max_actions=MAX_ACTIONS,
            semantic_family_fn=self._semantic_family,
            action_source_fn=self._action_source,
            facing_change_fn=self._is_kaiser_facing_change_action,
        )

    def _find_insatiable_frantic_escape_candidates(
        self,
        legal_actions: list[Any] | None,
        mask_np: np.ndarray,
    ) -> list[int]:
        return insatiable_strategy.find_frantic_escape_candidates(
            legal_actions,
            mask_np,
            max_actions=MAX_ACTIONS,
            semantic_family_fn=self._semantic_family,
        )

    def _insatiable_sandpit_countdown_from_context(
        self,
        boss_ctx: Any,
        raw_obs: Any | None,
    ) -> float | None:
        return insatiable_strategy.sandpit_countdown_from_context(boss_ctx, raw_obs)

    # Combat action-quality helpers live in ``muzero.combat_quality.trainer_quality``.
    # Diagnostic JSONL dump helpers live in ``muzero.diagnostics.trainer_dumps``.
    # P2-2 (recovery 2026-05-07): how many consecutive identical
    # (signature, picked, selection_action) tuples we tolerate before the
    # selection-loop guard fires. The first three picks are normal policy
    # behavior (e.g. clicking the same card to deselect/reselect or paging
    # through identical options); 4 is the smallest value that flags a
    # genuine dead-loop without killing legitimate exploration.
    _SELECTION_LOOP_STREAK_THRESHOLD: int = 4

    def _selection_loop_signature(
        self,
        legal_actions: list[Any] | None,
        mask_np: np.ndarray,
    ) -> tuple[tuple[str, str, str], ...] | None:
        """Build a stable signature of the current card_selection screen.

        Returns ``None`` if no card_selection actions are legal. Otherwise
        returns a sorted tuple of ``(action_id, card_id, selection_action)``
        triples capturing the union of legal selection options. The
        signature changes whenever the screen advances (option set changes
        or selection_action set changes), so consecutive identical
        signatures are direct evidence of a non-progressing selection loop.
        """
        if not isinstance(legal_actions, list):
            return None
        sigs: list[tuple[str, str, str]] = []
        has_selection = False
        try:
            mask_len = int(mask_np.shape[0]) if mask_np.size else 0
        except Exception:
            mask_len = 0
        for idx, act in enumerate(legal_actions):
            if idx >= mask_len:
                break
            if not isinstance(act, dict):
                continue
            if mask_np[idx] <= 0:
                continue
            family = self._semantic_family(act)
            if family != "card_selection":
                continue
            has_selection = True
            card = act.get("card") if isinstance(act.get("card"), dict) else {}
            sigs.append(
                (
                    str(act.get("action_id") or ""),
                    str(card.get("id") or ""),
                    str(act.get("selection_action") or act.get("selection") or "")
                    .strip()
                    .lower(),
                )
            )
        if not has_selection:
            return None
        return tuple(sorted(sigs))

    # Combat hard-guard orchestration lives in
    # ``muzero.combat_quality.hard_guard_orchestrator.CombatHardGuardMixin``.


    # ------------------------------------------------------------------
    # P2-3 death-slice writer (recovery 2026-05-07).
    #
    # The recovery doc flagged Knowledge Demon, Soul Nexus, Slumbering
    # Beetle (and to a lesser degree Kaiser/Insatiable) as encounters
    # where we cannot tell from aggregate metrics whether the bottleneck
    # is mechanism handling, deck synergy, or boss-specific bias. Per
    # spec §P2-3, we collect ≥50 boss-loss "slices" per encounter into
    # ``diagnostics/death_slices/<encounter_id>.jsonl`` so a human can
    # post-mortem without rerunning the agent.
    # ------------------------------------------------------------------

    _DEATH_SLICE_TARGETS: tuple[str, ...] = (
        "ENCOUNTER.KAISER_CRAB_BOSS",
        "ENCOUNTER.THE_INSATIABLE_BOSS",
        "ENCOUNTER.KNOWLEDGE_DEMON_BOSS",
        "ENCOUNTER.SOUL_NEXUS_ELITE",
        "ENCOUNTER.SLUMBERING_BEETLE_NORMAL",
        "ENCOUNTER.LAGAVULIN_MATRIARCH_BOSS",
        "ENCOUNTER.CEREMONIAL_BEAST_BOSS",
        "ENCOUNTER.THE_KIN_BOSS",
        # Act1 combat-sandbox hard-normal blockers.  These are also emitted
        # through the all-combat loss path, but keeping them in the watch-list
        # preserves boss/elite diagnostic compatibility for targeted calls.
        "ENCOUNTER.OVICOPTER_NORMAL",
        "ENCOUNTER.CONSTRUCT_MENAGERIE_NORMAL",
        "ENCOUNTER.SLUMBERING_BEETLE_NORMAL",
        "ENCOUNTER.SLIMED_BERSERKER_NORMAL",
        "ENCOUNTER.THE_OBSCURA_NORMAL",
        "ENCOUNTER.FABRICATOR_NORMAL",
    )
    _DEATH_SLICE_PER_ENCOUNTER_CAP: int = 200

    def _is_kaiser_risk_handling_action(self, action: Any, raw_obs: Any | None = None) -> bool:
        """Actions that can directly answer Kaiser/Rocket/Crusher back-attack risk.

        This intentionally includes explicit facing changes as *risk handling*, not only
        block/debuff cards.  The old boss_combat/kaiser_defense_candidate_count metric
        was interpreted as "defense / avoidance candidate"; if we keep facing separate
        only, a real turn-around action can exist while the legacy candidate metric stays
        at zero and misleads diagnosis.
        """
        if not isinstance(action, dict):
            return False
        family = self._semantic_family(action)
        if family == "end_turn":
            return False
        if self._is_kaiser_facing_change_action(action, raw_obs):
            return True
        roles = self._action_roles(action)
        if roles.intersection({"block", "debuff", "weak", "vulnerable"}):
            return True
        block = max(
            self._action_metric(action, "block"),
            self._action_metric(action, "total_block"),
            self._action_numeric_value(action, ("block", "total_block", "preview_block")),
        )
        if block > 0.0:
            return True
        damage = max(
            self._action_metric(action, "damage"),
            self._action_metric(action, "total_damage"),
            self._action_numeric_value(action, ("damage", "total_damage", "attack_damage", "preview_damage")),
        )
        if damage <= 0.0:
            return False
        target_hp = self._target_enemy_hp(action, raw_obs if isinstance(raw_obs, dict) else None)
        if target_hp > 0.0:
            # Damage only answers the back-attack mechanic if it kills the risky
            # side or represents a real phase/kill pressure, not because it is a
            # potion/card button that happens to be legal.
            return bool(damage >= target_hp or damage >= min(target_hp, max(12.0, 0.35 * target_hp)))
        return bool(damage >= 18.0)

    def _is_kaiser_pressure_action(self, action: Any) -> bool:
        """Damage-only actions that may race/kill but do not by themselves solve facing.

        Keeping this separate prevents a Strike-only hand from being mislabeled as having
        "defense" available, while still making snapshot composition visible in TB.
        """
        if not isinstance(action, dict):
            return False
        if self._semantic_family(action) != "play_card":
            return False
        roles = self._action_roles(action)
        return (
            bool(roles.intersection({"attack", "damage"}))
            or self._action_metric(action, "damage") > 0.0
            or self._action_metric(action, "total_damage") > 0.0
        )
