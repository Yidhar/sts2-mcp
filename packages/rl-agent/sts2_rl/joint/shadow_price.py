"""Combat Resource Shadow Price Oracle and Counterfactual Evaluation.

This module provides authoritative, physically grounded evaluation of macro-economic
decisions (Rest vs. Smith at campfires, Take vs. Skip at card rewards, Buy vs. Remove
at shops) by using the combat value network V_combat as a pricing oracle for
resources entering battle.

Instead of slicing an arbitrary heuristic temporal horizon (e.g. next_resource_opportunity)
or executing expensive, noisy multi-encounter counterfactual rollouts, the oracle
evaluates the marginal utility (shadow price) of resource mutations directly:

    lambda_HP = d V_combat / d HP
    Delta V_deck(a) = V_combat(S[Deck o+ Delta D(a)]) - V_combat(S)
    Q_shadow(S, a) = R_immediate(a) + gamma * (V_combat(S) + Delta V_HP(a) + Delta V_deck(a))
"""

from __future__ import annotations

import copy
import math
from dataclasses import dataclass
from typing import Any, Final, Mapping, Sequence

import torch
from torch import Tensor

from sts2_rl.encoding import EncodedDecisionSnapshot, GroundedObservationEncoder
from sts2_rl.encoding.snapshot import collate_encoded_snapshots
from sts2_rl.models.grounded_candidate import (
    COMBAT_DOMAIN_ID,
    GroundedCandidateBatch,
)
from sts2_rl.models.hybrid_actor_critic import HybridDomainActorCriticModel

SHADOW_PRICE_VERSION: Final = "sts2-combat-resource-shadow-price-v1"

# Canonical dummy action handle used solely to satisfy grounded encoder candidate-capacity
# invariants when evaluating state values independent of available actions.
_DUMMY_EVAL_ACTION: Final[dict[str, Any]] = {
    "action_handle": "oracle:eval",
    "kind": "proceed",
    "model_action_kind": "proceed",
    "is_enabled": True,
}


@dataclass(frozen=True, slots=True)
class ShadowPriceEvaluation:
    """Evaluation output for a full set of legal macro candidates."""

    action_values: Tensor  # [A], Q_shadow targets for macro policy projection
    delta_combat_values: Tensor  # [A], marginal Delta V_combat for each candidate
    base_combat_value: float  # V_combat(S_0)
    hp_shadow_price: float  # lambda_HP >= 0

    def validate(self) -> None:
        if self.action_values.ndim != 1 or self.delta_combat_values.ndim != 1:
            raise ValueError("action_values and delta_combat_values must be rank-1 tensors")
        if self.action_values.shape != self.delta_combat_values.shape:
            raise ValueError("action_values and delta_combat_values must have identical shapes")
        if not bool(torch.isfinite(self.action_values).all().item()):
            raise ValueError("action_values must be finite")
        if not bool(torch.isfinite(self.delta_combat_values).all().item()):
            raise ValueError("delta_combat_values must be finite")
        if not math.isfinite(self.base_combat_value):
            raise ValueError("base_combat_value must be finite")
        if not math.isfinite(self.hp_shadow_price) or self.hp_shadow_price < 0:
            raise ValueError("hp_shadow_price must be finite and non-negative")


def evaluate_combat_value_from_snapshot(
    model: HybridDomainActorCriticModel,
    snapshot: EncodedDecisionSnapshot,
    recurrent_state: Tensor | None = None,
    *,
    device: torch.device | str | None = None,
) -> float:
    """Evaluate a single snapshot under the model's combat representation."""
    target_device = device or next(model.parameters()).device
    batch = collate_encoded_snapshots(
        (snapshot,),
        expected_config=snapshot.config,
        expected_fingerprint=snapshot.encoding_fingerprint,
        device=target_device,
    )
    # Force combat stream routing
    batch = GroundedCandidateBatch(
        world=batch.world,
        candidates=batch.candidates,
        domain_ids=torch.full_like(batch.domain_ids, COMBAT_DOMAIN_ID),
        macro_economic_surface_ids=batch.macro_economic_surface_ids,
    )
    with torch.no_grad():
        out = model.forward(batch, recurrent_state=recurrent_state, validate=False)
        return float(out.value.item())


def compute_hp_shadow_price(
    model: HybridDomainActorCriticModel,
    encoder: GroundedObservationEncoder,
    observation: Mapping[str, Any],
    recurrent_state: Tensor | None = None,
    *,
    delta_hp: int = 5,
    device: torch.device | str | None = None,
) -> float:
    """Compute lambda_HP = d V_combat / d HP via finite difference.

    Physical invariant: lambda_HP >= 0 (additional HP never decreases survival).
    """
    player = observation.get("player") or {}
    max_hp = int(player.get("max_hp", 80))
    current_hp = int(player.get("hp", max_hp))

    # Base snapshot
    base_snapshot = encoder.encode(observation, [_DUMMY_EVAL_ACTION]).snapshot
    base_val = evaluate_combat_value_from_snapshot(model, base_snapshot, recurrent_state, device=device)

    # Perturbed snapshot: HP + delta_hp
    perturbed_hp = min(max_hp, current_hp + delta_hp)
    effective_delta = perturbed_hp - current_hp

    if effective_delta <= 0:
        # At full HP, check downward perturbation: V(HP) - V(HP - delta)
        perturbed_hp = max(1, current_hp - delta_hp)
        downward_delta = current_hp - perturbed_hp
        if downward_delta <= 0:
            return 0.0
        obs_down = copy.deepcopy(dict(observation))
        obs_down.setdefault("player", {})["hp"] = perturbed_hp
        snapshot_down = encoder.encode(obs_down, [_DUMMY_EVAL_ACTION]).snapshot
        val_down = evaluate_combat_value_from_snapshot(model, snapshot_down, recurrent_state, device=device)
        marginal = (base_val - val_down) / float(downward_delta)
        return max(0.0, float(marginal))

    obs_up = copy.deepcopy(dict(observation))
    obs_up.setdefault("player", {})["hp"] = perturbed_hp
    snapshot_up = encoder.encode(obs_up, [_DUMMY_EVAL_ACTION]).snapshot
    val_up = evaluate_combat_value_from_snapshot(model, snapshot_up, recurrent_state, device=device)
    marginal = (val_up - base_val) / float(effective_delta)
    return max(0.0, float(marginal))


def _mutate_observation_for_action(
    observation: Mapping[str, Any],
    action: Mapping[str, Any],
) -> tuple[dict[str, Any], float]:
    """Derive the counterfactual post-action observation and immediate reward.

    Returns:
        (mutated_observation, immediate_reward)
    """
    mutated = copy.deepcopy(dict(observation))
    player = mutated.setdefault("player", {})
    kind = str(action.get("kind") or action.get("action") or "").lower().replace("-", "_").replace(" ", "_")
    model_kind = str(action.get("model_action_kind") or "").lower().replace("-", "_").replace(" ", "_")
    variant = str(action.get("model_action_variant") or action.get("option_id") or "").lower().replace("-", "_").replace(" ", "_")
    option = action.get("option") if isinstance(action.get("option"), Mapping) else {}
    opt_type = str(option.get("option_type") or option.get("option_id") or "").lower().replace("-", "_").replace(" ", "_")
    immediate_reward = 0.0

    # 1. Rest site: Smith (Upgrade Card)
    if (
        kind in {"smith", "deck_upgrade", "upgrade"}
        or variant in {"smith", "upgrade"}
        or opt_type in {"smith", "upgrade"}
        or str(action.get("action") or "").lower() in {"smith", "upgrade"}
    ):
        target_card = action.get("card") or option.get("card")
        target_id = None
        if isinstance(target_card, Mapping):
            target_id = target_card.get("id") or target_card.get("card_id")
        preview = action.get("upgrade_preview") or option.get("upgrade_preview")
        deck = player.get("deck_cards") or player.get("deck") or []
        if isinstance(deck, list):
            new_deck: list[Any] = []
            replaced = False
            for c in deck:
                if not replaced and isinstance(c, Mapping) and (
                    target_id is None or c.get("id") == target_id or c.get("card_id") == target_id
                ):
                    upgraded_card = dict(preview) if isinstance(preview, Mapping) else dict(c)
                    upgraded_card["upgrades"] = int(upgraded_card.get("upgrades", 0)) + 1
                    upgraded_card["is_upgraded"] = True
                    new_deck.append(upgraded_card)
                    replaced = True
                else:
                    new_deck.append(c)
            player["deck_cards"] = new_deck
        return mutated, immediate_reward

    # 2. Rest site: Rest (Heal)
    if (
        kind in {"rest", "sleep"}
        or variant in {"rest", "sleep"}
        or opt_type in {"rest", "sleep"}
        or str(action.get("action") or "").lower() in {"rest", "sleep"}
        or kind == "rest_site"
        or (model_kind == "rest_site" and variant not in {"smith", "upgrade"} and opt_type not in {"smith", "upgrade"})
    ):
        heal = action.get("heal_amount") if action.get("heal_amount") is not None else option.get("heal_amount")
        if heal is None:
            # Game standard rest heal is ~30% of max HP
            max_hp = float(player.get("max_hp", 80))
            heal = math.floor(max_hp * 0.30)
        heal = max(0, int(heal))
        max_hp = int(player.get("max_hp", 80))
        current_hp = int(player.get("hp", max_hp))
        player["hp"] = min(max_hp, current_hp + heal)
        return mutated, immediate_reward

    # 3. Card Reward: Take Card
    if (
        kind in {"card_reward", "select_card", "take_card", "pick_card"}
        or model_kind in {"card_reward", "card_selection"}
    ) and not (
        kind in {"skip", "skip_card", "proceed", "leave"}
        or variant in {"skip", "skip_card", "proceed", "leave"}
        or model_kind == "proceed"
    ):
        card_to_add = action.get("card") or action.get("card_reward") or option.get("card")
        if isinstance(card_to_add, Mapping):
            deck = list(player.get("deck_cards") or player.get("deck") or [])
            deck.append(dict(card_to_add))
            player["deck_cards"] = deck
            player["deck"] = len(deck)
        return mutated, immediate_reward

    # 4. Card Reward: Skip / Proceed
    if (
        kind in {"skip", "skip_card", "proceed", "leave"}
        or variant in {"skip", "skip_card", "proceed", "leave"}
        or model_kind == "proceed"
    ):
        return mutated, immediate_reward

    # 5. Shop: Card Removal
    if kind in {"remove", "purge", "card_removal"} or variant in {"remove", "purge", "card_removal"}:
        target_card = action.get("card") or option.get("card")
        target_id = None
        if isinstance(target_card, Mapping):
            target_id = target_card.get("id") or target_card.get("card_id")
        cost = int(action.get("cost", action.get("price", 75)))
        player["gold"] = max(0, int(player.get("gold", 0)) - cost)
        deck = player.get("deck_cards") or player.get("deck") or []
        if isinstance(deck, list):
            new_deck = []
            removed = False
            for c in deck:
                if not removed and isinstance(c, Mapping) and (
                    target_id is None or c.get("id") == target_id or c.get("card_id") == target_id
                ):
                    removed = True
                else:
                    new_deck.append(c)
            player["deck_cards"] = new_deck
            player["deck"] = len(new_deck)
        return mutated, immediate_reward

    # Default fallback: return unchanged observation
    return mutated, immediate_reward


class CombatShadowPriceOracle:
    """Oracle computing combat-grounded resource shadow values for macro actions."""

    def __init__(
        self,
        model: HybridDomainActorCriticModel,
        encoder: GroundedObservationEncoder,
        *,
        discount: float = 0.997,
        device: torch.device | str | None = None,
    ) -> None:
        self.model = model
        self.encoder = encoder
        self.discount = float(discount)
        self.device = device or next(model.parameters()).device

    def evaluate_macro_root(
        self,
        observation: Mapping[str, Any],
        legal_actions: Sequence[Mapping[str, Any]],
        recurrent_state: Tensor | None = None,
    ) -> ShadowPriceEvaluation:
        """Compute the complete shadow price evaluation for all legal macro candidates."""
        if not legal_actions:
            raise ValueError("legal_actions cannot be empty for shadow price evaluation")

        # 1. Base combat value V_combat(S_0)
        base_decision = self.encoder.encode(observation, [_DUMMY_EVAL_ACTION])
        base_snapshot = base_decision.snapshot
        base_combat_value = evaluate_combat_value_from_snapshot(
            self.model, base_snapshot, recurrent_state, device=self.device
        )

        # 2. HP shadow price lambda_HP
        hp_shadow_price = compute_hp_shadow_price(
            self.model, self.encoder, observation, recurrent_state, device=self.device
        )

        # 3. Counterfactual evaluation for each legal candidate
        delta_values: list[float] = []
        action_values: list[float] = []

        # Batch counterfactual snapshots for high-throughput single forward pass
        mutated_snapshots: list[EncodedDecisionSnapshot] = []
        immediate_rewards: list[float] = []

        for action in legal_actions:
            mutated_obs, imm_r = _mutate_observation_for_action(observation, action)
            snapshot = self.encoder.encode(mutated_obs, [_DUMMY_EVAL_ACTION]).snapshot
            mutated_snapshots.append(snapshot)
            immediate_rewards.append(imm_r)

        # Collate all counterfactual candidate states into a single batch
        first_snapshot = mutated_snapshots[0]
        batch = collate_encoded_snapshots(
            tuple(mutated_snapshots),
            expected_config=first_snapshot.config,
            expected_fingerprint=first_snapshot.encoding_fingerprint,
            device=self.device,
        )
        batch = GroundedCandidateBatch(
            world=batch.world,
            candidates=batch.candidates,
            domain_ids=torch.full_like(batch.domain_ids, COMBAT_DOMAIN_ID),
            macro_economic_surface_ids=batch.macro_economic_surface_ids,
        )

        with torch.no_grad():
            rep_recurrent = None
            if recurrent_state is not None:
                if recurrent_state.ndim == 1:
                    rep_recurrent = recurrent_state.unsqueeze(0).expand(len(mutated_snapshots), -1)
                else:
                    rep_recurrent = recurrent_state.expand(len(mutated_snapshots), -1)
            out = self.model.forward(batch, recurrent_state=rep_recurrent, validate=False)
            candidate_combat_values = out.value.view(-1).cpu().tolist()

        for cand_val, imm_r in zip(candidate_combat_values, immediate_rewards, strict=True):
            delta_v = cand_val - base_combat_value
            q_shadow = imm_r + self.discount * cand_val
            delta_values.append(delta_v)
            action_values.append(q_shadow)

        eval_result = ShadowPriceEvaluation(
            action_values=torch.tensor(action_values, dtype=torch.float32),
            delta_combat_values=torch.tensor(delta_values, dtype=torch.float32),
            base_combat_value=base_combat_value,
            hp_shadow_price=hp_shadow_price,
        )
        eval_result.validate()
        return eval_result


__all__ = [
    "SHADOW_PRICE_VERSION",
    "ShadowPriceEvaluation",
    "CombatShadowPriceOracle",
    "compute_hp_shadow_price",
    "evaluate_combat_value_from_snapshot",
]
