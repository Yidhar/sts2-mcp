"""Tests for Durable Afterstate v8.0 CRN Counterfactual Roots & Zero Fallback Drift."""

from __future__ import annotations

import copy
import sys
from dataclasses import replace
from pathlib import Path
from typing import Any

import pytest

torch = pytest.importorskip("torch")

from sts2_rl.encoding import GroundedObservationEncoder  # noqa: E402
from sts2_rl.encoding.snapshot import EncodedWorldSnapshot, collate_encoded_snapshots  # noqa: E402
from sts2_rl.joint.macro_credit_collection import _select_crn_branch_indices  # noqa: E402
from sts2_rl.joint.macro_credit_data import (  # noqa: E402
    MacroAfterstateNode,
    MacroCreditCohort,
    MacroCreditTrajectory,
    MacroRewardFacts,
)
from sts2_rl.joint.macro_credit_learner import MacroCreditLearner, MacroCreditLearnerConfig  # noqa: E402
from sts2_rl.joint.macro_targets import project_hierarchical_macro_policy  # noqa: E402
from sts2_rl.joint.transitions import HybridVTraceStep  # noqa: E402
from sts2_rl.models.hybrid_actor_critic import (  # noqa: E402
    HybridActorCriticConfig,
    HybridDomainActorCriticModel,
)
from sts2_rl.semantics.clock import decision_discount  # noqa: E402
from sts2_rl.train_macro_credit import load_macro_training_config  # noqa: E402
from sts2_rl.training.config import ModelConfig  # noqa: E402


def _tiny_v80_config() -> tuple[ModelConfig, HybridActorCriticConfig]:
    tiny = ModelConfig(
        d_model=32,
        n_heads=4,
        ffn_dim=64,
        world_layers=1,
        latent_slots=4,
        latent_layers=1,
        local_layers=1,
        candidate_layers=1,
        recurrent_hidden_dim=32,
        dropout=0.0,
        max_world_tokens=256,
        max_candidates=32,
        max_candidate_local_tokens=16,
    )
    ac_cfg = HybridActorCriticConfig(
        grounded=tiny.to_model_config(),
        domain_layers=1,
        joint_layers=1,
        enable_hierarchical_branch_policy=True,
        enable_durable_value_head=True,
        enable_persistent_build_context=True,
        enable_direct_vitals_context=True,
        enable_macro_branch_prior=True,
        enable_persistent_only_durable_value=True,
        enable_asymmetric_revival_critic=True,
        enable_draw_weighted_durable_value=True,
        enable_role_conditioned_candidate_policy=True,
        enable_deconfounded_durable_value=True,
        enable_monotone_deconfounded_durable_value=True,
        enable_balanced_deconfounded_durable_value=True,
        enable_expressive_synergy_backbone=True,
        enable_debloated_synergy_durable_value=True,
        enable_build_conditioned_durable_value=True,
        enable_twin_durable_value_heads=True,
        enable_local_combat_value_head=True,
        enable_direct_combat_vitals_context=True,
        enable_active_durable_vitals_gates=True,
    )
    return tiny, ac_cfg


def _make_card_reward_surface(floor: int = 4) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    obs: dict[str, Any] = {
        "phase": "card_reward",
        "screen": "CARD_REWARD",
        "decision_domain": "card_reward",
        "run": {"active": True, "act": 1, "floor": floor, "gold": 99, "ascension": 0},
        "player": {
            "character_id": "IRONCLAD",
            "hp": 52,
            "current_hp": 52,
            "max_hp": 80,
            "gold": 99,
            "deck": [
                {"id": "STRIKE_IRONCLAD", "type": "Attack", "rarity": "Basic", "cost": 1, "quantity": 4},
                {"id": "DEFEND_IRONCLAD", "type": "Skill", "rarity": "Basic", "cost": 1, "quantity": 4},
                {"id": "BASH", "type": "Attack", "rarity": "Basic", "cost": 2, "quantity": 1},
            ],
            "relics": [{"id": "BURNING_BLOOD"}],
            "potions": [],
        },
        "combat": {"in_progress": False},
        "card_reward_selection": {
            "visible": True,
            "cards": [
                {"index": 0, "id": "SHRUG_IT_OFF", "name": "Shrug It Off", "type": "Skill", "rarity": "Common", "cost": 1},
                {"index": 1, "id": "BLOODLETTING", "name": "Bloodletting", "type": "Skill", "rarity": "Uncommon", "cost": 0},
                {"index": 2, "id": "CLEAVE", "name": "Cleave", "type": "Attack", "rarity": "Common", "cost": 1},
            ],
            "can_skip": True,
        },
    }
    legal = [
        {"action_id": "take:0", "kind": "select_card_reward", "model_action_kind": "card_reward", "card_index": 0, "card": obs["card_reward_selection"]["cards"][0], "is_enabled": True},
        {"action_id": "take:1", "kind": "select_card_reward", "model_action_kind": "card_reward", "card_index": 1, "card": obs["card_reward_selection"]["cards"][1], "is_enabled": True},
        {"action_id": "take:2", "kind": "select_card_reward", "model_action_kind": "card_reward", "card_index": 2, "card": obs["card_reward_selection"]["cards"][2], "is_enabled": True},
        {"action_id": "skip", "kind": "skip_card_reward", "model_action_kind": "card_reward", "is_enabled": True},
    ]
    return obs, legal


def _make_map_surface(floor: int = 3) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    obs: dict[str, Any] = {
        "phase": "map",
        "screen": "MAP",
        "decision_domain": "map",
        "run": {"active": True, "act": 1, "floor": floor, "gold": 99, "ascension": 0},
        "player": {
            "character_id": "IRONCLAD",
            "hp": 58,
            "current_hp": 58,
            "max_hp": 80,
            "gold": 99,
            "deck": [
                {"id": "STRIKE_IRONCLAD", "type": "Attack", "rarity": "Basic", "cost": 1, "quantity": 4},
                {"id": "DEFEND_IRONCLAD", "type": "Skill", "rarity": "Basic", "cost": 1, "quantity": 4},
                {"id": "BASH", "type": "Attack", "rarity": "Basic", "cost": 2, "quantity": 1},
            ],
            "relics": [{"id": "BURNING_BLOOD"}],
            "potions": [],
        },
        "combat": {"in_progress": False},
        "map": {
            "visible": True,
            "nodes": [
                {"x": 0, "y": floor, "room_type": "MONSTER", "reachable": True},
                {"x": 2, "y": floor, "room_type": "ELITE", "reachable": True},
            ],
        },
    }
    legal = [
        {"action_id": "map:0", "kind": "choose_map_node", "model_action_kind": "map", "node": {"x": 0, "y": floor, "room_type": "MONSTER"}, "is_enabled": True},
        {"action_id": "map:1", "kind": "choose_map_node", "model_action_kind": "map", "node": {"x": 2, "y": floor, "room_type": "ELITE"}, "is_enabled": True},
    ]
    return obs, legal


def _make_combat_surface(floor: int = 5) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    obs: dict[str, Any] = {
        "phase": "combat",
        "screen": "COMBAT",
        "decision_domain": "combat",
        "run": {"active": True, "act": 1, "floor": floor, "gold": 99, "ascension": 0},
        "player": {
            "character_id": "IRONCLAD",
            "hp": 52,
            "current_hp": 52,
            "max_hp": 80,
            "gold": 99,
            "deck": [
                {"id": "STRIKE_IRONCLAD", "type": "Attack", "rarity": "Basic", "cost": 1, "quantity": 4},
                {"id": "DEFEND_IRONCLAD", "type": "Skill", "rarity": "Basic", "cost": 1, "quantity": 4},
            ],
            "relics": [{"id": "BURNING_BLOOD"}],
            "potions": [],
        },
        "combat": {
            "in_progress": True,
            "turn": 1,
            "energy": 3,
            "max_energy": 3,
            "player_block": 0,
            "hand": [
                {"id": "STRIKE_IRONCLAD", "cost": 1, "type": "Attack", "playable": True},
                {"id": "DEFEND_IRONCLAD", "cost": 1, "type": "Skill", "playable": True},
            ],
            "enemies": [
                {"id": "JAW_WORM", "hp": 38, "max_hp": 42, "block": 0, "intent": "Attack", "intent_damage": 11, "intent_hits": 1},
            ],
        },
    }
    legal = [
        {"action_id": "play:0", "kind": "play_card", "model_action_kind": "play_card", "card_index": 0, "target_index": 0, "is_enabled": True},
        {"action_id": "play:1", "kind": "play_card", "model_action_kind": "play_card", "card_index": 1, "is_enabled": True},
        {"action_id": "end_turn", "kind": "end_turn", "model_action_kind": "end_turn", "is_enabled": True},
    ]
    return obs, legal


def _reward_facts(terminal_reward: float, floor_before: int, floor_after: int) -> MacroRewardFacts:
    return MacroRewardFacts(
        terminal_reward=terminal_reward,
        progress_reward=0.0,
        revival_penalty=0.0,
        pace_penalty=0.0,
        hp_loss_penalty=0.0,
        revivals_used_delta=0,
        player_hp_lost_delta=0.0,
        floor_before=floor_before,
        floor_after=floor_after,
        native_steps=1,
    )


def _encode_reward_afterstate_worlds(
    encoder: GroundedObservationEncoder,
    obs: dict[str, Any],
    legal: list[dict[str, Any]],
) -> tuple[EncodedWorldSnapshot, ...]:
    worlds: list[EncodedWorldSnapshot] = []
    dummy_legal = [
        {"action_id": "map:0", "kind": "choose_map_node", "model_action_kind": "map", "node": {"x": 0, "y": 5, "room_type": "MONSTER"}, "is_enabled": True},
    ]
    cards = obs["card_reward_selection"]["cards"]
    for i in range(len(legal)):
        o_copy = copy.deepcopy(obs)
        o_copy["phase"] = "map"
        o_copy["screen"] = "MAP"
        o_copy["decision_domain"] = "map"
        o_copy.pop("card_reward_selection", None)
        if i < len(cards):
            o_copy["player"]["deck"].append(
                {
                    "id": cards[i]["id"],
                    "type": cards[i]["type"],
                    "rarity": cards[i]["rarity"],
                    "cost": cards[i]["cost"],
                    "quantity": 1,
                }
            )
            # Also vary HP/gold slightly so synergy + vitals signals are well-conditioned in tiny model
            hp_delta = 8 if i == 0 else (-10 if i == 1 else -2)
            o_copy["player"]["hp"] = max(1, 52 + hp_delta)
            o_copy["player"]["current_hp"] = o_copy["player"]["hp"]
        snap = encoder.encode(o_copy, dummy_legal).snapshot
        worlds.append(
            EncodedWorldSnapshot(
                world=snap.world,
                domain_id=snap.domain_id,
                config=snap.config,
                encoding_fingerprint=snap.encoding_fingerprint,
            )
        )
    return tuple(worlds)


def _compute_log_prob(
    model: HybridDomainActorCriticModel,
    snapshot: Any,
    hidden: Any,
    action_index: int,
) -> tuple[float, Any]:
    batch = collate_encoded_snapshots(
        (snapshot,),
        expected_config=snapshot.config,
        expected_fingerprint=snapshot.encoding_fingerprint,
        device=torch.device("cpu"),
        validate=False,
    )
    with torch.no_grad():
        out = model(batch, hidden, validate=False)
        mask = batch.candidates.action_mask[0]
        logs = torch.log_softmax(out.policy_logits[0].masked_fill(~mask, -torch.inf), dim=-1)
        return float(logs[action_index].item()), out.recurrent_state


def _build_crn_trajectory(
    model: HybridDomainActorCriticModel,
    encoder: GroundedObservationEncoder,
    *,
    seed: int,
    teacher_version: int = 0,
    crn_indices: tuple[int, ...] | None,
    crn_returns: tuple[float, ...] | None,
    include_reward_node: bool = True,
) -> MacroCreditTrajectory:
    map_obs, map_legal = _make_map_surface(floor=3)
    rew_obs, rew_legal = _make_card_reward_surface(floor=4)
    com_obs, com_legal = _make_combat_surface(floor=5)

    map_snap = encoder.encode(map_obs, map_legal).snapshot
    rew_snap = encoder.encode(rew_obs, rew_legal).snapshot
    com_snap = encoder.encode(com_obs, com_legal).snapshot
    post_worlds = _encode_reward_afterstate_worlds(encoder, rew_obs, rew_legal)

    hidden = model.initial_state(1, device=torch.device("cpu"))
    lp0, hidden = _compute_log_prob(model, map_snap, hidden, 0)
    lp1, hidden = _compute_log_prob(model, rew_snap, hidden, 0)
    lp2, hidden = _compute_log_prob(model, com_snap, hidden, 0)

    f0 = _reward_facts(0.0, 3, 4)
    f1 = _reward_facts(0.0, 4, 5)
    f2 = _reward_facts(1.0, 5, 5)
    steps = (
        HybridVTraceStep(
            map_snap,
            0,
            lp0,
            f0.reward,
            decision_discount(floor_before=3, floor_after=4, terminal=False),
            False,
            "map",
            "monster",
            "macro",
            teacher_version,
        ),
        HybridVTraceStep(
            rew_snap,
            0,
            lp1,
            f1.reward,
            decision_discount(floor_before=4, floor_after=5, terminal=False),
            False,
            "reward",
            "take",
            "macro",
            teacher_version,
        ),
        HybridVTraceStep(
            com_snap,
            0,
            lp2,
            f2.reward,
            decision_discount(floor_before=5, floor_after=5, terminal=True),
            True,
            "combat",
            "play",
            "combat",
            teacher_version,
        ),
    )
    nodes: tuple[MacroAfterstateNode, ...] = ()
    if include_reward_node:
        nodes = (
            MacroAfterstateNode(
                step_index=1,
                surface="reward",
                candidate_branches=("take", "take", "take", "skip"),
                post_worlds=post_worlds,
                crn_candidate_indices=crn_indices,
                crn_candidate_returns=crn_returns,
            ),
        )
    return MacroCreditTrajectory(
        f"traj-{seed}",
        seed,
        teacher_version,
        (),
        steps,
        None,
        None,
        "factual_terminal",
        (f0, f1, f2),
        afterstate_nodes=nodes,
        macro_branch_epsilon=0.0,
    )


def test_macro_afterstate_node_crn_validation() -> None:
    tiny, _ = _tiny_v80_config()
    encoder = GroundedObservationEncoder(tiny.to_encoding_config())
    rew_obs, rew_legal = _make_card_reward_surface(floor=4)
    post_worlds = _encode_reward_afterstate_worlds(encoder, rew_obs, rew_legal)

    # Valid full CRN
    valid = MacroAfterstateNode(
        step_index=1,
        surface="reward",
        candidate_branches=("take", "take", "take", "skip"),
        post_worlds=post_worlds,
        crn_candidate_indices=(0, 1, 2, 3),
        crn_candidate_returns=(1.2, -0.5, 0.1, 0.8),
    )
    assert valid.crn_candidate_indices == (0, 1, 2, 3)

    # Valid strict subset CRN (e.g., Rest Site top-3 + Heal)
    subset = MacroAfterstateNode(
        step_index=1,
        surface="reward",
        candidate_branches=("take", "take", "take", "skip"),
        post_worlds=post_worlds,
        crn_candidate_indices=(0, 3),
        crn_candidate_returns=(1.2, 0.8),
    )
    assert subset.crn_candidate_indices == (0, 3)

    # Reject mismatched / half-specified / duplicate / out-of-range / non-finite / single-branch
    with pytest.raises(ValueError, match="both be set"):
        MacroAfterstateNode(
            step_index=1,
            surface="reward",
            candidate_branches=("take", "take", "take", "skip"),
            post_worlds=post_worlds,
            crn_candidate_indices=(0, 3),
            crn_candidate_returns=None,
        )
    with pytest.raises(ValueError, match="length >= 2"):
        MacroAfterstateNode(
            step_index=1,
            surface="reward",
            candidate_branches=("take", "take", "take", "skip"),
            post_worlds=post_worlds,
            crn_candidate_indices=(0,),
            crn_candidate_returns=(1.0,),
        )
    with pytest.raises(ValueError, match="length >= 2"):
        MacroAfterstateNode(
            step_index=1,
            surface="reward",
            candidate_branches=("take", "take", "take", "skip"),
            post_worlds=post_worlds,
            crn_candidate_indices=(0, 1, 3),
            crn_candidate_returns=(1.0, 0.5),
        )
    with pytest.raises(ValueError, match="finite"):
        MacroAfterstateNode(
            step_index=1,
            surface="reward",
            candidate_branches=("take", "take", "take", "skip"),
            post_worlds=post_worlds,
            crn_candidate_indices=(0, 3),
            crn_candidate_returns=(1.0, float("nan")),
        )
    with pytest.raises(ValueError, match="strictly increasing"):
        MacroAfterstateNode(
            step_index=1,
            surface="reward",
            candidate_branches=("take", "take", "take", "skip"),
            post_worlds=post_worlds,
            crn_candidate_indices=(0, 0),
            crn_candidate_returns=(1.0, 0.5),
        )
    with pytest.raises(ValueError, match="strictly increasing subset"):
        MacroAfterstateNode(
            step_index=1,
            surface="reward",
            candidate_branches=("take", "take", "take", "skip"),
            post_worlds=post_worlds,
            crn_candidate_indices=(0, 9),
            crn_candidate_returns=(1.0, 0.5),
        )


def test_centered_twin_disagreement_is_invariant_to_constant_offset() -> None:
    v1 = torch.tensor([0.52, 0.48, 0.45, 0.50], dtype=torch.float32)
    v2_exact = v1 + torch.tensor([0.005, -0.004, 0.002, -0.003], dtype=torch.float32)
    # Adding a huge constant baseline offset (+0.25) between Twin heads must NOT inflate disagreement!
    v2_shifted = v2_exact + 0.25
    teacher_logs = torch.log(torch.tensor([0.30, 0.30, 0.30, 0.10], dtype=torch.float32))
    branch_ids = torch.tensor([1, 1, 1, 2], dtype=torch.long)

    proj_base = project_hierarchical_macro_policy(
        det_action_values=v1,
        det_action_values_twin=v2_exact,
        pessimism_coef=0.25,
        teacher_log_probs=teacher_logs,
        policy_branch_ids=branch_ids,
        branch_max_kl=0.04,
        within_max_kl=0.05,
        min_temperature=0.015,
        center_twin_disagreement=True,
    )
    proj_shifted = project_hierarchical_macro_policy(
        det_action_values=v1,
        det_action_values_twin=v2_shifted,
        pessimism_coef=0.25,
        teacher_log_probs=teacher_logs,
        policy_branch_ids=branch_ids,
        branch_max_kl=0.04,
        within_max_kl=0.05,
        min_temperature=0.015,
        center_twin_disagreement=True,
    )
    assert torch.allclose(proj_base.target_probabilities, proj_shifted.target_probabilities, atol=1e-6)
    assert abs(proj_base.kl - proj_shifted.kl) < 1e-6


@pytest.mark.parametrize("chunk_batch_size", [1, 4])
def test_zero_macro_pg_fallback_and_crn_only_mpo_zeroes_map_and_non_crn_grads(
    chunk_batch_size: int,
) -> None:
    tiny, ac_cfg = _tiny_v80_config()
    torch.manual_seed(801)
    model = HybridDomainActorCriticModel(ac_cfg)
    model.eval()
    encoder = GroundedObservationEncoder(tiny.to_encoding_config())

    # Cohort with Map fallback + non-CRN reward node (crn_returns=None)
    trajs = tuple(
        _build_crn_trajectory(
            model,
            encoder,
            seed=13_400_000 + 2 * i,
            teacher_version=0,
            crn_indices=None,
            crn_returns=None,
        )
        for i in range(2)
    )
    cohort = MacroCreditCohort("v80-zero-fallback", 0, (), natural_only=trajs)
    cfg = MacroCreditLearnerConfig(
        target_mode="durable_afterstate_v2",
        domain_layers=1,
        joint_layers=1,
        enable_direct_vitals_context=True,
        enable_direct_combat_vitals_context=True,
        enable_active_durable_vitals_gates=True,
        enable_macro_branch_prior=True,
        enable_persistent_only_durable_value=True,
        enable_asymmetric_revival_critic=True,
        enable_draw_weighted_durable_value=True,
        enable_role_conditioned_candidate_policy=True,
        enable_deconfounded_durable_value=True,
        enable_monotone_deconfounded_durable_value=True,
        enable_balanced_deconfounded_durable_value=True,
        enable_expressive_synergy_backbone=True,
        enable_debloated_synergy_durable_value=True,
        enable_build_conditioned_durable_value=True,
        enable_twin_durable_value_heads=True,
        enable_hierarchical_afterstate_mpo=True,
        durable_warmup_updates=0,
        combat_actor_warmup_updates=0,
        macro_actor_warmup_updates=0,
        crn_only_macro_mpo=True,
        crn_pairwise_value_weight=0.0,
        macro_pg_fallback_scale=0.0,
        macro_entropy_weight=0.05,  # Must still be completely skipped when fallback_scale == 0.0!
        macro_policy_weight=1.0,
        macro_value_weight=0.0,
        durable_post_value_weight=0.0,
        durable_stage_baseline_weight=0.0,
        combat_policy_weight=0.0,
        combat_value_weight=0.0,
        combat_entropy_weight=0.0,
        eval_batch_size=chunk_batch_size,
        chunk_batch_size=chunk_batch_size,
        revival_budget=0,
        learning_rate=1e-3,
    )
    learner = MacroCreditLearner(model=model, config=cfg)
    before_policy = {
        name: param.detach().clone()
        for name, param in model.named_parameters()
        if "policy" in name or "branch_policy" in name
    }
    metrics = learner.learn(cohort)
    assert metrics["macro_pg_policy_loss"] == 0.0
    assert metrics["macro_mpo_policy_loss"] == 0.0
    assert metrics["macro_entropy"] == 0.0
    assert metrics["enabled_afterstate_projections"] == 0
    assert metrics["crn_roots_used"] == 0
    for name, before_t in before_policy.items():
        after_t = dict(model.named_parameters())[name]
        assert torch.equal(after_t, before_t), f"Policy parameter {name} drifted despite zero fallback + non-CRN!"


@pytest.mark.parametrize("chunk_batch_size", [1, 4])
def test_crn_mpo_and_pairwise_delta_v_train_synergy_and_policy(chunk_batch_size: int) -> None:
    tiny, ac_cfg = _tiny_v80_config()
    torch.manual_seed(802)
    model = HybridDomainActorCriticModel(ac_cfg)
    model.eval()
    encoder = GroundedObservationEncoder(tiny.to_encoding_config())

    cfg = MacroCreditLearnerConfig(
        target_mode="durable_afterstate_v2",
        domain_layers=1,
        joint_layers=1,
        enable_direct_vitals_context=True,
        enable_direct_combat_vitals_context=True,
        enable_active_durable_vitals_gates=True,
        enable_macro_branch_prior=True,
        enable_persistent_only_durable_value=True,
        enable_asymmetric_revival_critic=True,
        enable_draw_weighted_durable_value=True,
        enable_role_conditioned_candidate_policy=True,
        enable_deconfounded_durable_value=True,
        enable_monotone_deconfounded_durable_value=True,
        enable_balanced_deconfounded_durable_value=True,
        enable_expressive_synergy_backbone=True,
        enable_debloated_synergy_durable_value=True,
        enable_build_conditioned_durable_value=True,
        enable_twin_durable_value_heads=True,
        enable_hierarchical_afterstate_mpo=True,
        durable_warmup_updates=0,
        combat_actor_warmup_updates=0,
        macro_actor_warmup_updates=0,
        minority_only_within_branch_mpo=False,
        crn_only_macro_mpo=False,
        use_direct_crn_mpo=False,
        double_mpo_pessimism_coef=0.25,
        crn_pairwise_value_weight=2.0,
        macro_pg_fallback_scale=0.0,
        macro_entropy_weight=0.0,
        macro_kl=0.04,
        macro_within_branch_kl=0.06,
        durable_value_lr_scale=3.0,
        detach_value_heads_from_trunk=True,
        eval_batch_size=chunk_batch_size,
        chunk_batch_size=chunk_batch_size,
        revival_budget=0,
        learning_rate=3e-3,
        shared_learning_rate_scale=0.25,
        weight_decay=0.0,
    )
    learner = MacroCreditLearner(model=model, config=cfg)

    # Fix #3: Verify durable_deck_synergy_* stays PROTECTED in 0.25x shared LR group,
    # while readout gates (durable_deck_out / durable_active_hp_gate) get 3.0x durable_val LR!
    syn_param = next(
        p for n, p in model.named_parameters() if n.startswith("durable_deck_synergy_ffn.")
    )
    readout_param = next(
        p for n, p in model.named_parameters() if n.startswith("durable_active_hp_gate.")
    )
    syn_lr = None
    readout_lr = None
    for grp in learner.optimizer.param_groups:
        if any(p is syn_param for p in grp["params"]):
            syn_lr = float(grp["lr"])
        if any(p is readout_param for p in grp["params"]):
            readout_lr = float(grp["lr"])
    assert syn_lr is not None and readout_lr is not None
    assert abs(syn_lr - cfg.learning_rate * cfg.shared_learning_rate_scale) < 1e-9
    assert abs(readout_lr - cfg.learning_rate * cfg.durable_value_lr_scale) < 1e-9

    syn_before = syn_param.detach().clone()
    # Strong empirical CRN preference: Card 0 (+1.2) > Skip (+0.6) > Card 2 (-0.1) > Card 1 (-1.0)
    crn_returns = (1.2, -1.0, -0.1, 0.6)

    rew_obs, rew_legal = _make_card_reward_surface(floor=4)
    rew_snap = encoder.encode(rew_obs, rew_legal).snapshot
    init_lp0, _ = _compute_log_prob(model, rew_snap, model.initial_state(1), 0)
    init_lp1, _ = _compute_log_prob(model, rew_snap, model.initial_state(1), 1)

    first_loss: float | None = None
    last_metrics: dict[str, Any] = {}
    for step_u in range(10):
        trajs = tuple(
            _build_crn_trajectory(
                model,
                encoder,
                seed=13_410_000 + 100 * step_u + 2 * i,
                teacher_version=learner.update,
                crn_indices=(0, 1, 2, 3),
                crn_returns=crn_returns,
            )
            for i in range(2)
        )
        cohort = MacroCreditCohort(f"v81-crn-u{step_u}", learner.update, (), natural_only=trajs)
        last_metrics = learner.learn(cohort)
        if first_loss is None:
            first_loss = float(last_metrics["crn_pairwise_value_loss"])

    after_lp0, _ = _compute_log_prob(model, rew_snap, model.initial_state(1), 0)
    after_lp1, _ = _compute_log_prob(model, rew_snap, model.initial_state(1), 1)

    assert last_metrics["crn_roots_used"] == 2
    assert last_metrics["crn_afterstate_steps"] == 2
    assert abs(last_metrics["crn_mean_return_spread"] - 2.2) < 1e-6
    assert last_metrics["macro_mpo_policy_loss"] > 0.0
    assert last_metrics["macro_pg_policy_loss"] == 0.0
    assert first_loss is not None and first_loss > 0.0
    assert float(last_metrics["crn_pairwise_value_loss"]) < first_loss
    assert float(last_metrics["crn_value_rank_agreement"]) >= 0.8
    assert not torch.allclose(syn_param, syn_before)
    # Twin Critics learn counterfactual delta-V first, then Twin Double-MPO shifts policy cleanly!
    assert (after_lp0 - after_lp1) > (init_lp0 - init_lp1) + 0.02


def test_unblinded_active_macro_kl_triggers_line_search_with_many_map_steps() -> None:
    tiny, ac_cfg = _tiny_v80_config()
    torch.manual_seed(811)
    model = HybridDomainActorCriticModel(ac_cfg)
    model.eval()
    encoder = GroundedObservationEncoder(tiny.to_encoding_config())

    # Build a trajectory with 8 untouched Map steps + 1 active Reward afterstate step + 1 Combat step.
    # Previously, 8 untouched Map steps (KL=0) diluted post_macro_kl by 9x and blinded KL line search!
    rew_obs, rew_legal = _make_card_reward_surface(floor=10)
    com_obs, com_legal = _make_combat_surface(floor=11)
    rew_snap = encoder.encode(rew_obs, rew_legal).snapshot
    com_snap = encoder.encode(com_obs, com_legal).snapshot
    post_worlds = _encode_reward_afterstate_worlds(encoder, rew_obs, rew_legal)

    hidden = model.initial_state(1, device=torch.device("cpu"))
    steps_list: list[HybridVTraceStep] = []
    facts_list: list[MacroRewardFacts] = []
    for fl in range(2, 10):
        m_obs, m_legal = _make_map_surface(floor=fl)
        m_snap = encoder.encode(m_obs, m_legal).snapshot
        lp_m, hidden = _compute_log_prob(model, m_snap, hidden, 0)
        f_m = _reward_facts(0.0, fl, fl + 1)
        steps_list.append(
            HybridVTraceStep(
                m_snap,
                0,
                lp_m,
                f_m.reward,
                decision_discount(floor_before=fl, floor_after=fl + 1, terminal=False),
                False,
                "map",
                "monster",
                "macro",
                0,
            )
        )
        facts_list.append(f_m)

    lp_r, hidden = _compute_log_prob(model, rew_snap, hidden, 0)
    f_r = _reward_facts(0.0, 10, 11)
    reward_step_idx = len(steps_list)
    steps_list.append(
        HybridVTraceStep(
            rew_snap,
            0,
            lp_r,
            f_r.reward,
            decision_discount(floor_before=10, floor_after=11, terminal=False),
            False,
            "reward",
            "take",
            "macro",
            0,
        )
    )
    facts_list.append(f_r)

    lp_c, _ = _compute_log_prob(model, com_snap, hidden, 0)
    f_c = _reward_facts(1.0, 11, 11)
    steps_list.append(
        HybridVTraceStep(
            com_snap,
            0,
            lp_c,
            f_c.reward,
            decision_discount(floor_before=11, floor_after=11, terminal=True),
            True,
            "combat",
            "play",
            "combat",
            0,
        )
    )
    facts_list.append(f_c)

    node = MacroAfterstateNode(
        step_index=reward_step_idx,
        surface="reward",
        candidate_branches=("take", "take", "take", "skip"),
        post_worlds=post_worlds,
    )
    trajs = tuple(
        MacroCreditTrajectory(
            f"v81-kl-{i}",
            13_500_000 + 2 * i,
            0,
            (),
            tuple(steps_list),
            None,
            None,
            "factual_terminal",
            tuple(facts_list),
            afterstate_nodes=(node,),
            macro_branch_epsilon=0.0,
        )
        for i in range(2)
    )
    cfg = MacroCreditLearnerConfig(
        target_mode="durable_afterstate_v2",
        domain_layers=1,
        joint_layers=1,
        enable_direct_vitals_context=True,
        enable_direct_combat_vitals_context=True,
        enable_active_durable_vitals_gates=True,
        enable_macro_branch_prior=True,
        enable_persistent_only_durable_value=True,
        enable_asymmetric_revival_critic=True,
        enable_draw_weighted_durable_value=True,
        enable_role_conditioned_candidate_policy=True,
        enable_deconfounded_durable_value=True,
        enable_monotone_deconfounded_durable_value=True,
        enable_balanced_deconfounded_durable_value=True,
        enable_expressive_synergy_backbone=True,
        enable_debloated_synergy_durable_value=True,
        enable_build_conditioned_durable_value=True,
        enable_twin_durable_value_heads=True,
        enable_hierarchical_afterstate_mpo=True,
        double_mpo_pessimism_coef=0.0,
        durable_warmup_updates=0,
        combat_actor_warmup_updates=0,
        macro_actor_warmup_updates=0,
        macro_pg_fallback_scale=0.0,
        macro_entropy_weight=0.0,
        macro_kl=0.10,
        macro_within_branch_kl=0.10,
        enable_post_step_kl_line_search=True,
        target_macro_step_kl=0.002,
        combat_policy_weight=0.0,
        combat_value_weight=0.0,
        eval_batch_size=2,
        chunk_batch_size=2,
        revival_budget=0,
        learning_rate=5e-2,
    )
    learner = MacroCreditLearner(model=model, config=cfg)
    m = learner.learn(MacroCreditCohort("v81-kl-test", 0, (), natural_only=trajs))
    assert m["raw_updated_macro_policy_kl"] > 0.002
    assert m["line_search_backtracks"] > 0
    assert m["line_search_scale"] < 1.0
    assert m["mean_updated_macro_policy_kl"] <= m["raw_updated_macro_policy_kl"]


def test_select_crn_branch_indices_card_reward_and_campfire() -> None:
    tiny, _ = _tiny_v80_config()
    encoder = GroundedObservationEncoder(tiny.to_encoding_config())
    rew_obs, rew_legal = _make_card_reward_surface(floor=4)
    post_worlds = _encode_reward_afterstate_worlds(encoder, rew_obs, rew_legal)

    # Full 4-candidate reward returns all 4
    rew_node = MacroAfterstateNode(
        step_index=1,
        surface="reward",
        candidate_branches=("take", "take", "take", "skip"),
        post_worlds=post_worlds,
    )
    assert _select_crn_branch_indices(
        rew_node, ref_action=1, teacher_log_probs=(-1.0, -2.0, -3.0, -0.5), max_branches=4
    ) == (0, 1, 2, 3)

    # Campfire with 1 rest + 6 smith candidates capped to max_branches=4:
    # must include rest (0), ref_action (4), and top-2 remaining smith candidates (2, 5)
    camp_worlds = (post_worlds[0],) * 7
    camp_node = MacroAfterstateNode(
        step_index=2,
        surface="rest",
        candidate_branches=("rest", "smith", "smith", "smith", "smith", "smith", "smith"),
        post_worlds=camp_worlds,
    )
    picked = _select_crn_branch_indices(
        camp_node,
        ref_action=4,
        teacher_log_probs=(-0.5, -4.0, -1.1, -3.5, -2.9, -1.3, -5.0),
        max_branches=4,
    )
    assert picked == (0, 2, 4, 5)


def test_v30_v81_config_and_act_b0_crn_96_seed_gate_scaling() -> None:
    repo_root = Path(__file__).resolve().parents[1]
    cfg = load_macro_training_config(
        repo_root
        / "config/experiments/hybrid_macro_credit_v30_durable_afterstate_v81_twin_crn_critic_b0_act1_g4.toml"
    )
    assert cfg.course.roots_per_cohort == 32
    assert cfg.course.parallel_collectors == 48
    assert cfg.course.crn_roots_per_episode == 1
    assert cfg.course.crn_max_branches_per_root == 4
    assert cfg.learner.crn_only_macro_mpo is False
    assert cfg.learner.use_direct_crn_mpo is False
    assert cfg.learner.double_mpo_pessimism_coef == 0.35
    assert cfg.learner.minority_only_within_branch_mpo is True
    assert cfg.learner.crn_pairwise_value_weight == 0.35
    assert cfg.learner.macro_pg_fallback_scale == 0.0
    assert cfg.learner.macro_entropy_weight == 0.0
    assert cfg.learner.combat_pg_kl_weight == 0.02
    assert cfg.learner.combat_entropy_weight == 0.007
    assert cfg.learner.durable_value_lr_scale == 2.0
    assert cfg.learner.macro_actor_warmup_updates == 10
    assert cfg.learner.target_macro_step_kl == 0.010
    assert cfg.training.runtime.seed == 6760000

    scripts_dir = str(repo_root / "scripts")
    if scripts_dir not in sys.path:
        sys.path.insert(0, scripts_dir)
    import run_colab_g4_ratchet_curriculum as ratchet

    stages = ratchet._select_ratchet_stages(curriculum_mode="act-b0-crn")
    assert [s.stage_key for s in stages] == ["b0-act1", "b0-act2", "b0-act3"]

    act1 = stages[0]
    ok_59, rep_59 = ratchet.evaluate_stage_promotion_gate(
        {
            "policy_version": 10,
            "episodes": 96,
            "successful_run_count": 59,
            "revival_budget": 0,
            "reward_objective": "act1",
            "mean_revivals_used": 0.0,
            "skip_rate": 0.12,
            "mean_final_deck_size": 18.0,
            "deadlock_count": 0,
        },
        act1,
        expected_episodes=96,
        min_policy_version=1,
    )
    assert not ok_59
    assert rep_59["wins_required"] == 60

    ok_60, rep_60 = ratchet.evaluate_stage_promotion_gate(
        {
            "policy_version": 10,
            "episodes": 96,
            "successful_run_count": 60,
            "revival_budget": 0,
            "reward_objective": "act1",
            "mean_revivals_used": 0.0,
            "skip_rate": 0.12,
            "mean_final_deck_size": 18.0,
            "deadlock_count": 0,
        },
        act1,
        expected_episodes=96,
        min_policy_version=1,
    )
    assert ok_60
    assert rep_60["wins_required"] == 60

    # Reanchor min_win_gain scales from 2 (n=48) to 4 (n=96)
    reanchor_rows = [
        {
            "policy_version": 0,
            "episodes": 96,
            "successful_run_count": 20,
            "revival_budget": 0,
            "mean_revivals_used": 0.0,
            "skip_rate": 0.10,
            "mean_final_deck_size": 18.0,
            "deadlock_count": 0,
        },
        {
            "policy_version": 10,
            "episodes": 96,
            "successful_run_count": 23,
            "revival_budget": 0,
            "mean_revivals_used": 0.0,
            "skip_rate": 0.10,
            "mean_final_deck_size": 18.0,
            "deadlock_count": 0,
        },
    ]
    assert (
        ratchet.select_intrastage_teacher_reanchor(
            reanchor_rows, act1, expected_episodes=96, min_policy_version=1
        )
        is None
    )
    reanchor_rows.append({**reanchor_rows[-1], "policy_version": 20, "successful_run_count": 24})
    assert (
        ratchet.select_intrastage_teacher_reanchor(
            reanchor_rows, act1, expected_episodes=96, min_policy_version=1
        )
        == 20
    )


def test_crn_pairwise_keeps_dropout_disabled_on_identical_worlds() -> None:
    tiny, ac_cfg = _tiny_v80_config()
    torch.manual_seed(820)
    model = HybridDomainActorCriticModel(ac_cfg)
    model.eval()
    # Exaggerate Twin dropout rate so any accidental .train(True) immediately produces large mask noise!
    model.durable_twin_head1_dropout.p = 0.50
    model.durable_twin_dropout.p = 0.50
    encoder = GroundedObservationEncoder(tiny.to_encoding_config())

    rew_obs, rew_legal = _make_card_reward_surface(floor=4)
    post_worlds = _encode_reward_afterstate_worlds(encoder, rew_obs, rew_legal)
    identical_worlds = (post_worlds[0],) * 4

    base_traj = _build_crn_trajectory(
        model,
        encoder,
        seed=13_600_000,
        teacher_version=0,
        crn_indices=(0, 1, 2, 3),
        crn_returns=(1.0, 1.0, 1.0, 1.0),
    )
    ident_node = MacroAfterstateNode(
        step_index=1,
        surface="reward",
        candidate_branches=("take", "take", "take", "skip"),
        post_worlds=identical_worlds,
        crn_candidate_indices=(0, 1, 2, 3),
        crn_candidate_returns=(1.0, 1.0, 1.0, 1.0),
    )
    traj = replace(base_traj, afterstate_nodes=(ident_node,))
    cfg = MacroCreditLearnerConfig(
        target_mode="durable_afterstate_v2",
        domain_layers=1,
        joint_layers=1,
        enable_direct_vitals_context=True,
        enable_direct_combat_vitals_context=True,
        enable_active_durable_vitals_gates=True,
        enable_macro_branch_prior=True,
        enable_persistent_only_durable_value=True,
        enable_asymmetric_revival_critic=True,
        enable_draw_weighted_durable_value=True,
        enable_role_conditioned_candidate_policy=True,
        enable_deconfounded_durable_value=True,
        enable_monotone_deconfounded_durable_value=True,
        enable_balanced_deconfounded_durable_value=True,
        enable_expressive_synergy_backbone=True,
        enable_debloated_synergy_durable_value=True,
        enable_build_conditioned_durable_value=True,
        enable_twin_durable_value_heads=True,
        enable_hierarchical_afterstate_mpo=True,
        crn_pairwise_value_weight=1.0,
        macro_value_weight=0.0,
        durable_post_value_weight=0.0,
        durable_stage_baseline_weight=0.0,
        combat_policy_weight=0.0,
        combat_value_weight=0.0,
        revival_budget=0,
    )
    learner = MacroCreditLearner(model=model, config=cfg)
    m = learner.learn(MacroCreditCohort("v82-no-het-dropout", 0, (), natural_only=(traj,)))
    # Identical post-worlds with equal returns MUST have exact 0 centered delta-V loss!
    assert m["crn_pairwise_value_loss"] < 1e-12


def test_v31_v82_locked_combat_and_fast96_config() -> None:
    repo_root = Path(__file__).resolve().parents[1]
    cfg = load_macro_training_config(
        repo_root
        / "config/experiments/hybrid_macro_credit_v31_durable_afterstate_v82_locked_combat_fast96_b0_act1_g4.toml"
    )
    assert cfg.course.roots_per_cohort == 96
    assert cfg.course.parallel_collectors == 48
    assert cfg.course.crn_roots_per_episode == 0
    assert cfg.learner.ppo_minibatches_per_cohort == 1
    assert cfg.learner.combat_policy_weight == 0.0
    assert cfg.learner.combat_value_weight == 0.0
    assert cfg.learner.combat_entropy_weight == 0.0
    assert cfg.learner.kickstart_kl_floor == 0.015
    assert cfg.learner.kickstart_combat_kl_scale == 6.0
    assert cfg.learner.macro_pg_fallback_scale == 0.0
    assert cfg.learner.target_macro_step_kl == 0.008

    # Prove locked combat + active teacher combat KL anchor keeps distill_combat_kl near 0 across Macro updates!
    tiny, ac_cfg = _tiny_v80_config()
    torch.manual_seed(822)
    student = HybridDomainActorCriticModel(ac_cfg)
    teacher = copy.deepcopy(student)
    student.eval()
    teacher.eval()
    for p in teacher.parameters():
        p.requires_grad_(False)
    encoder = GroundedObservationEncoder(tiny.to_encoding_config())

    l_cfg = MacroCreditLearnerConfig(
        target_mode="durable_afterstate_v2",
        domain_layers=1,
        joint_layers=1,
        enable_direct_vitals_context=True,
        enable_direct_combat_vitals_context=True,
        enable_active_durable_vitals_gates=True,
        enable_macro_branch_prior=True,
        enable_persistent_only_durable_value=True,
        enable_asymmetric_revival_critic=True,
        enable_draw_weighted_durable_value=True,
        enable_role_conditioned_candidate_policy=True,
        enable_deconfounded_durable_value=True,
        enable_monotone_deconfounded_durable_value=True,
        enable_balanced_deconfounded_durable_value=True,
        enable_expressive_synergy_backbone=True,
        enable_debloated_synergy_durable_value=True,
        enable_build_conditioned_durable_value=True,
        enable_twin_durable_value_heads=True,
        enable_hierarchical_afterstate_mpo=True,
        durable_warmup_updates=0,
        combat_actor_warmup_updates=0,
        macro_actor_warmup_updates=0,
        macro_pg_fallback_scale=0.0,
        macro_entropy_weight=0.0,
        combat_policy_weight=0.0,
        combat_value_weight=0.0,
        combat_entropy_weight=0.0,
        kickstart_kl_weight=0.020,
        kickstart_kl_floor=0.015,
        kickstart_combat_kl_scale=6.0,
        ppo_minibatches_per_cohort=1,
        revival_budget=0,
        learning_rate=1e-3,
    )
    learner = MacroCreditLearner(model=student, config=l_cfg, teacher_model=teacher)
    last_m: dict[str, Any] = {}
    for step_u in range(5):
        trajs = tuple(
            _build_crn_trajectory(
                student,
                encoder,
                seed=13_700_000 + 20 * step_u + 2 * i,
                teacher_version=learner.update,
                crn_indices=None,
                crn_returns=None,
            )
            for i in range(2)
        )
        last_m = learner.learn(MacroCreditCohort(f"v82-u{step_u}", learner.update, (), natural_only=trajs))
    assert last_m["distill_combat_kl"] < 0.002
    assert last_m["mean_updated_combat_policy_kl"] < 0.002
