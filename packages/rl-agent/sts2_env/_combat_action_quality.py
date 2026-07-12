"""Combat action-quality classification and diagnostic helpers."""

from __future__ import annotations

from typing import Any

from .card_effect_profile import aggregate_card_effect_profile_semantics
from .end_turn_quality import strict_end_turn_waste_context
from .observation_common import _aggregate_card_modifier_semantics


class CombatActionQualityMixin:
    @staticmethod
    def _card_cost(card: dict[str, Any] | None) -> float:
        if not isinstance(card, dict):
            return 0.0
        for key in ("cost", "resolved_energy_cost", "canonical_energy_cost"):
            value = card.get(key)
            try:
                return max(float(value), 0.0)
            except (TypeError, ValueError):
                continue
        return 0.0

    @staticmethod
    def _card_is_x_cost(card: dict[str, Any] | None) -> bool:
        if not isinstance(card, dict):
            return False
        return bool(card.get("x_cost") or card.get("costs_x") or card.get("is_x_cost")) or str(card.get("cost") or card.get("canonical_energy_cost") or "").strip().upper() == "X"

    @classmethod
    def _action_is_x_cost(cls, action: dict[str, Any] | None) -> bool:
        """Runtime action-level X-cost detector.

        Do not rely only on the embedded card cost: live compact actions may
        expose X-cost as ``semantic.roles=["x_cost"]`` while the card carries a
        temporary numeric cost after runtime modifiers.  Combat diagnostics and
        strategic-skip logic must use the action-level contract when available.
        """

        if not isinstance(action, dict):
            return False
        if "x_cost" in cls._action_roles(action):
            return True
        semantic = cls._action_semantic(action)
        if bool(semantic.get("is_x_cost")):
            return True
        try:
            if float(semantic.get("x_cost_value") or 0.0) > 0.0:
                return True
        except (TypeError, ValueError):
            pass
        action_x_cost = action.get("x_cost")
        if not isinstance(action_x_cost, dict) and bool(action_x_cost):
            return True
        if bool(action.get("costs_x") or action.get("is_x_cost")):
            return True
        if str(action.get("card_cost") or "").strip().upper() == "X":
            return True
        card = action.get("card") if isinstance(action.get("card"), dict) else None
        return cls._card_is_x_cost(card)

    def _action_modifier_semantics(self, action: dict[str, Any] | None) -> dict[str, float]:
        if not isinstance(action, dict):
            return {}
        card = action.get("card")
        if isinstance(card, dict):
            return _aggregate_card_modifier_semantics(card)
        return {}

    def _action_effect_semantics(self, action: dict[str, Any] | None) -> dict[str, float]:
        """Typed card-effect semantics for one legal action.

        This intentionally reads the bridge/registry ``card_effect_profile``
        instead of card text.  Potion timing has its own typed profile path;
        combat card quality needs card-side facts such as gain-energy,
        modify-cost, no-draw, retain/exhaust, replay, and hand mutation.
        """
        if not isinstance(action, dict):
            return {}
        card = action.get("card")
        if isinstance(card, dict):
            return aggregate_card_effect_profile_semantics(card)
        return {}

    def _action_positive_score(self, action: dict[str, Any]) -> float:
        kind = str(action.get("kind") or "").strip()
        if kind not in ("play_card", "use_potion"):
            return 0.0
        source = action.get("card") if kind == "play_card" else action.get("potion")
        if not isinstance(source, dict):
            return 0.0
        if kind == "play_card" and str(source.get("type") or "").strip().lower() == "power":
            return 4.0
        score = 0.0
        weights = {
            "damage": 1.0,
            "block": 0.8,
            "draw": 3.0,
            "weak": 4.0,
            "vulnerable": 4.0,
            "heal": 2.0,
            "strength": 3.0,
            "dexterity": 3.0,
            "summon": 5.0,
        }
        for key, weight in weights.items():
            score += self._source_preview_metric(source, key) * weight
        if kind == "play_card":
            sem = _aggregate_card_modifier_semantics(source)
            score += sem.get("energy_gain", 0.0) * 3.0
            score += sem.get("draw", 0.0) * 3.0
            score += sem.get("block_add", 0.0) * 0.8 + sem.get("block_on_play", 0.0) * 0.8
            score += sem.get("damage_add", 0.0)
            score += sem.get("weak", 0.0) * 4.0
            score -= sem.get("energy_loss_on_play", 0.0) * 2.0
            score -= sem.get("self_damage", 0.0) * 2.5
            typed = aggregate_card_effect_profile_semantics(source)
            score += typed.get("typed_gain_energy_amount", 0.0) * 3.0
            score += typed.get("typed_draw_amount", 0.0) * 3.0
            score -= typed.get("typed_hp_loss", 0.0) * 2.5
            # Hand/card-state mutation is real progress, but it is setup-like
            # progress.  Keep the value modest so follow-up-dependent cards
            # (cost reducers, no-draw/future-penalty cards, energy refunds)
            # can be classified as deferable rather than mandatory.
            if any(
                typed.get(key, 0.0) > 0.0
                for key in (
                    "typed_upgrade_hand",
                    "typed_modify_cost",
                    "typed_set_replay",
                    "typed_retain_cards",
                    "typed_add_modifier",
                    "typed_add_keyword",
                    "typed_add_generated_card",
                    "typed_card_state_mutation",
                )
            ):
                score += 2.0
        return float(max(score, 0.0))

    def _is_positive_progress_action(self, action: dict[str, Any]) -> bool:
        return self._action_positive_score(action) > 0.0

    def _classify_refund_followup(
        self,
        action: dict[str, Any],
        *,
        energy: float,
        legal_actions: list[dict[str, Any]],
        raw_obs: dict[str, Any] | None = None,
    ) -> str:
        """B3: classify a refund (energy_gain) play by its after-action prospects.

        Returns one of ``refund_good_followup``, ``refund_no_followup_but_intrinsic_value``,
        ``refund_no_followup_low_value``, or ``refund_unknown`` (non-refund).

        Intent: avoid penalising a refund played without a *static* followup
        when the card itself draws/creates new cards, reduces costs, or has
        intrinsic block/lethal/mechanism value (Kaiser facing change, etc.).
        """

        if str(action.get("kind") or "").strip() != "play_card":
            return "refund_unknown"
        card = action.get("card") if isinstance(action.get("card"), dict) else None
        if not isinstance(card, dict):
            return "refund_unknown"
        sem = _aggregate_card_modifier_semantics(card)
        typed = aggregate_card_effect_profile_semantics(card)
        energy_gain = max(
            float(sem.get("energy_gain", 0.0) or 0.0),
            float(typed.get("typed_gain_energy_amount", 0.0) or 0.0),
        )
        if energy_gain <= 0.0:
            return "refund_unknown"

        # After-action energy estimate.
        cost = self._card_cost(card)
        energy_loss = float(sem.get("energy_loss_on_play", 0.0) or 0.0)
        energy_after = max(0.0, float(energy) - float(cost) + energy_gain - energy_loss)

        # Followup signals derived from the *card's own* expected effects.
        expected_draw = float(typed.get("typed_draw_amount", 0.0) or 0.0) + float(sem.get("draw", 0.0) or 0.0)
        expected_create = bool(typed.get("typed_add_generated_card", 0.0) > 0.0)
        cost_reduction = bool(typed.get("typed_modify_cost", 0.0) > 0.0)
        replay_or_duplicate = bool(
            typed.get("typed_set_replay", 0.0) > 0.0
            or typed.get("typed_copy_cards", 0.0) > 0.0
        )
        if expected_draw >= 1.0 or expected_create or cost_reduction or replay_or_duplicate:
            return "refund_good_followup"

        # Static followup on the *current* hand.
        for other in legal_actions:
            if other is action or not isinstance(other, dict) or other.get("kind") != "play_card":
                continue
            other_card = other.get("card")
            if not isinstance(other_card, dict) or self._action_is_x_cost(other):
                continue
            other_cost = self._card_cost(other_card)
            if other_cost <= energy_after + 1e-6 and self._action_positive_score(other) >= 3.0:
                return "refund_good_followup"

        # Intrinsic value: block-lethal / mechanism / large-impact even without
        # a chained followup.
        block_value = max(
            float(self._source_preview_metric(card, "block")),
            float(sem.get("block_add", 0.0) or 0.0) + float(sem.get("block_on_play", 0.0) or 0.0),
        )
        incoming = self._refund_incoming_damage(raw_obs)
        if block_value > 0.0 and block_value + float((((raw_obs or {}).get("player") or {}).get("block")) or 0.0) >= incoming and incoming >= 8.0:
            return "refund_no_followup_but_intrinsic_value"
        damage_value = float(self._source_preview_metric(card, "damage"))
        if damage_value >= 12.0:
            return "refund_no_followup_but_intrinsic_value"
        # Mechanism: refund that targets the back-attack side counts as
        # facing-change intrinsic value.  Per P0-3 hardening spec, position
        # (left/right) MUST come from BACK_ATTACK_{LEFT,RIGHT}_POWER on the
        # target enemy, not from faction ``side`` strings on the target dict.
        try:
            from .boss_kaiser import classify_kaiser_action_mechanism
            kaiser_mech = classify_kaiser_action_mechanism(
                (raw_obs or {}).get("combat") if isinstance(raw_obs, dict) else None,
                action,
                player_obs=(raw_obs or {}).get("player") if isinstance(raw_obs, dict) else None,
            )
            if kaiser_mech.get("kaiser_changes_facing"):
                return "refund_no_followup_but_intrinsic_value"
        except Exception:
            pass
        return "refund_no_followup_low_value"

    @staticmethod
    def _refund_incoming_damage(raw_obs: dict[str, Any] | None) -> float:
        if not isinstance(raw_obs, dict):
            return 0.0
        combat = raw_obs.get("combat") if isinstance(raw_obs.get("combat"), dict) else {}
        enemies = combat.get("enemies") if isinstance(combat.get("enemies"), list) else []
        total = 0.0
        for enemy in enemies:
            if not isinstance(enemy, dict):
                continue
            intent = enemy.get("intent") if isinstance(enemy.get("intent"), dict) else {}
            for key in ("total_damage", "damage", "intent_damage", "attack_damage"):
                v = intent.get(key) if intent else enemy.get(key)
                try:
                    total = max(total, float(v or 0.0))
                except (TypeError, ValueError):
                    pass
        return total

    def _is_strategic_skip_candidate(
        self,
        action: dict[str, Any],
        *,
        energy: float,
        legal_actions: list[dict[str, Any]],
    ) -> bool:
        if str(action.get("kind") or "").strip() != "play_card":
            return False
        card = action.get("card")
        if not isinstance(card, dict):
            return False
        sem = _aggregate_card_modifier_semantics(card)
        typed = aggregate_card_effect_profile_semantics(card)
        if self._action_is_x_cost(action) and energy <= 0.05:
            return True
        immediate = self._action_positive_score(action)
        exhausts = bool(
            sem.get("adds_exhaust")
            or sem.get("removes_exhaust") < 0.0
            or card.get("exhaust")
            or card.get("will_exhaust")
            or typed.get("typed_once_or_exhaust_self", 0.0) > 0.0
            or typed.get("typed_exhaust_cards", 0.0) > 0.0
        )
        retains = bool(sem.get("adds_retain") or card.get("retain") or typed.get("typed_retain_cards", 0.0) > 0.0)
        self_damage = max(float(sem.get("self_damage", 0.0) or 0.0), float(typed.get("typed_hp_loss", 0.0) or 0.0))
        energy_loss = sem.get("energy_loss_on_play", 0.0)
        energy_gain = max(float(sem.get("energy_gain", 0.0) or 0.0), float(typed.get("typed_gain_energy_amount", 0.0) or 0.0))
        requires_followup = bool(
            typed.get("typed_requires_followup", 0.0) > 0.0
            or typed.get("typed_strategic_skip_if_no_followup", 0.0) > 0.0
            or typed.get("typed_modify_cost", 0.0) > 0.0
            or typed.get("typed_no_draw", 0.0) > 0.0
            or typed.get("typed_future_penalty", 0.0) > 0.0
        )
        future_penalty = bool(
            typed.get("typed_no_draw", 0.0) > 0.0
            or typed.get("typed_future_penalty", 0.0) > 0.0
            or typed.get("typed_consumes_future_resource", 0.0) > 0.0
        )
        card_state_setup = bool(
            typed.get("typed_card_state_mutation", 0.0) > 0.0
            or typed.get("typed_modifies_hand", 0.0) > 0.0
        )
        cost = self._card_cost(card)
        energy_after = max(0.0, energy - cost + energy_gain - energy_loss)
        followups = 0
        for other in legal_actions:
            if other is action or not isinstance(other, dict) or other.get("kind") != "play_card":
                continue
            other_card = other.get("card")
            if isinstance(other_card, dict) and self._card_cost(other_card) <= energy_after + 1e-6 and self._action_positive_score(other) > 0.0:
                followups += 1
        if energy_gain > 0.0 and followups <= 0 and immediate < max(6.0, energy_gain * 3.0):
            return True
        if requires_followup and followups <= 0 and immediate < max(6.0, energy_gain * 3.0 + 2.0):
            return True
        if future_penalty and followups <= 0 and immediate < 8.0:
            return True
        # B2 narrowing: pure exhaust/retain alone is no longer a strategic skip
        # candidate.  Without a future-reason signal (hand mutation, replay, deck
        # cycling, upgrade-hand) the model would otherwise be trained to fear any
        # consume card.  Lethal/high-impact exhausts have immediate >> 4 and so
        # already fall through; the threshold drop from 6 to 4 also drops the
        # mid-range "exhaust 5 damage" cases that were being mislabelled.
        has_future_setup_signal = bool(
            typed.get("typed_card_state_mutation", 0.0) > 0.0
            or typed.get("typed_modifies_hand", 0.0) > 0.0
            or typed.get("typed_set_replay", 0.0) > 0.0
            or typed.get("typed_upgrade_hand", 0.0) > 0.0
            or typed.get("typed_consumes_future_resource", 0.0) > 0.0
        )
        if (
            (exhausts or retains)
            and has_future_setup_signal
            and immediate < 4.0
            and not card.get("ethereal")
        ):
            return True
        if card_state_setup and followups <= 0 and immediate < 4.0:
            return True
        if (self_damage > 0.0 or energy_loss > 0.0) and immediate < (self_damage * 2.5 + energy_loss * 2.0 + 4.0):
            return True
        return False

    def _action_quality_diagnostics(
        self,
        obs: dict[str, Any] | None,
        legal_actions: list[dict[str, Any]],
        chosen_action: dict[str, Any] | None,
    ) -> dict[str, float]:
        combat = obs.get("combat") if isinstance(obs, dict) else None
        energy = float((combat or {}).get("energy") or 0.0) if isinstance(combat, dict) else 0.0
        selected_id = str((chosen_action or {}).get("action_id") or "")
        selected_is_end_turn = float(selected_id == "end_turn")
        positive_available = 0
        mandatory_positive = 0
        strategic_skip = 0
        zero_x_available = 0
        refund_no_followup_available = 0
        typed_followup_missing = 0
        typed_future_penalty_count = 0
        typed_no_draw_count = 0
        typed_card_state_mutation_count = 0
        setup_followup_dependent_count = 0
        setup_followup_available_count = 0
        enchantment_seen = 0
        affliction_seen = 0
        selected_zero_x = 0.0
        selected_refund_no_followup = 0.0
        selected_strategic_skip = 0.0
        for action in legal_actions:
            if not isinstance(action, dict) or str(action.get("action_id") or "") == "end_turn":
                continue
            card = action.get("card") if isinstance(action.get("card"), dict) else None
            typed = aggregate_card_effect_profile_semantics(card) if isinstance(card, dict) else {}
            sem = _aggregate_card_modifier_semantics(card) if isinstance(card, dict) else {}
            if isinstance(card, dict):
                if (
                    (isinstance(card.get("enchantments"), list)
                    and card.get("enchantments"))
                    or typed.get("typed_add_modifier", 0.0) > 0.0
                    or typed.get("typed_card_rule_modifier", 0.0) > 0.0
                ):
                    enchantment_seen = 1
                if isinstance(card.get("afflictions"), list) and card.get("afflictions"):
                    affliction_seen = 1
            is_positive = self._is_positive_progress_action(action)
            if is_positive:
                positive_available += 1
            is_strategic = self._is_strategic_skip_candidate(action, energy=energy, legal_actions=legal_actions)
            if is_strategic:
                strategic_skip += 1
            else:
                if is_positive:
                    mandatory_positive += 1
            is_zero_x = isinstance(card, dict) and self._action_is_x_cost(action) and energy <= 0.0
            if is_zero_x:
                zero_x_available += 1
            typed_energy_gain = float(typed.get("typed_gain_energy_amount", 0.0) or 0.0)
            typed_modify_cost = typed.get("typed_modify_cost", 0.0) > 0.0
            typed_no_draw = typed.get("typed_no_draw", 0.0) > 0.0
            typed_future_penalty = typed.get("typed_future_penalty", 0.0) > 0.0
            typed_consumes_future_resource = typed.get("typed_consumes_future_resource", 0.0) > 0.0
            typed_card_state_mutation = typed.get("typed_card_state_mutation", 0.0) > 0.0 or typed.get("typed_modifies_hand", 0.0) > 0.0
            setup_followup_dependent = bool(
                typed.get("typed_requires_followup", 0.0) > 0.0
                or typed.get("typed_strategic_skip_if_no_followup", 0.0) > 0.0
                or typed_modify_cost
                or typed_no_draw
                or typed_future_penalty
                or typed_consumes_future_resource
            )
            setup_followup_available = False
            if isinstance(card, dict) and setup_followup_dependent:
                energy_after = max(
                    0.0,
                    energy
                    - self._card_cost(card)
                    + max(float(sem.get("energy_gain", 0.0) or 0.0), typed_energy_gain)
                    - float(sem.get("energy_loss_on_play", 0.0) or 0.0),
                )
                for other in legal_actions:
                    if other is action or not isinstance(other, dict) or other.get("kind") != "play_card":
                        continue
                    other_card = other.get("card")
                    if not isinstance(other_card, dict) or self._action_is_x_cost(other):
                        continue
                    other_cost = self._card_cost(other_card)
                    if (other_cost <= energy_after + 1e-6 or typed_modify_cost) and self._action_positive_score(other) >= 3.0:
                        setup_followup_available = True
                        break
            if setup_followup_dependent:
                setup_followup_dependent_count += 1
                if setup_followup_available:
                    setup_followup_available_count += 1
                else:
                    typed_followup_missing += 1
            if typed_future_penalty:
                typed_future_penalty_count += 1
            if typed_no_draw:
                typed_no_draw_count += 1
            if typed_card_state_mutation:
                typed_card_state_mutation_count += 1
            is_refund = max(float(sem.get("energy_gain", 0.0) or 0.0), typed_energy_gain) > 0.0 and is_strategic
            if is_refund:
                refund_no_followup_available += 1
            if chosen_action is action:
                selected_zero_x = float(is_zero_x)
                selected_refund_no_followup = float(is_refund)
                selected_strategic_skip = float(is_strategic)
        strict_end_turn = strict_end_turn_waste_context(
            obs,
            legal_actions,
            chosen_action,
            positive_score_fn=self._action_positive_score,
            strategic_skip_fn=lambda action, current_energy, actions: self._is_strategic_skip_candidate(
                dict(action),
                energy=current_energy,
                legal_actions=[dict(item) for item in actions],
            ),
        )
        # Strict EndTurn semantics: leftover energy is not waste.  Waste is only
        # when an urgent/safe non-EndTurn action exists on the stable frontier.
        wasteful_available = float(bool(strict_end_turn.get("wasteful_end_turn_available", False)))
        wasteful_selected = float(selected_is_end_turn > 0.5 and wasteful_available > 0.5)
        return {
            "energy": float(energy),
            "positive_action_count": float(positive_available),
            "mandatory_positive_action_count": float(mandatory_positive),
            "urgent_positive_action_count": float(strict_end_turn.get("urgent_positive_action_count", 0.0) or 0.0),
            "non_end_turn_action_count": float(strict_end_turn.get("non_end_turn_action_count", 0.0) or 0.0),
            "playable_card_count": float(strict_end_turn.get("playable_card_count", 0.0) or 0.0),
            "incoming_damage": float(strict_end_turn.get("incoming_damage", 0.0) or 0.0),
            "current_block": float(strict_end_turn.get("current_block", 0.0) or 0.0),
            "benign_leftover_energy": float(bool(strict_end_turn.get("benign_leftover_energy", False))),
            "strategic_skip_candidate_count": float(strategic_skip),
            "wasteful_end_turn_available": wasteful_available,
            "wasteful_end_turn_selected": wasteful_selected,
            "zero_energy_x_cost_available": float(zero_x_available),
            "zero_energy_x_cost_selected": selected_zero_x,
            "refund_no_followup_available": float(refund_no_followup_available),
            "refund_no_followup_selected": selected_refund_no_followup,
            "strategic_skip_selected": selected_strategic_skip,
            "typed_followup_missing_count": float(typed_followup_missing),
            "typed_future_penalty_count": float(typed_future_penalty_count),
            "typed_no_draw_count": float(typed_no_draw_count),
            "typed_card_state_mutation_count": float(typed_card_state_mutation_count),
            "setup_followup_dependent_count": float(setup_followup_dependent_count),
            "setup_followup_available_count": float(setup_followup_available_count),
            "enchantment_seen": float(enchantment_seen),
            "affliction_seen": float(affliction_seen),
        }
