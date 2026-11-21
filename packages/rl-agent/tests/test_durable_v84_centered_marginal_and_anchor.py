"""Tests for Durable Afterstate v8.4: Zero-Mean Centered Marginal Card-ID V_durable, High-HP Campfire Guard + Saturated hp_basis, and Un-Diluted Cumulative Macro Drift Anchor."""

from __future__ import annotations

import copy
from dataclasses import replace
from pathlib import Path
from typing import Any

import pytest

torch = pytest.importorskip("torch")

from sts2_rl.encoding import GroundedObservationEncoder  # noqa: E402
from sts2_rl.encoding.snapshot import (  # noqa: E402
    EncodedWorldSnapshot,
    collate_encoded_snapshots,
    collate_encoded_worlds,
)
from sts2_rl.joint.macro_credit_data import (  # noqa: E402
    MacroAfterstateNode,
    MacroCreditCohort,
    MacroCreditTrajectory,
    MacroRewardFacts,
)
from sts2_rl.joint.macro_credit_learner import (  # noqa: E402
    MacroCreditLearner,
    MacroCreditLearnerConfig,
)
from sts2_rl.joint.transitions import HybridVTraceStep  # noqa: E402
from sts2_rl.models.hybrid_actor_critic import (  # noqa: E402
    HybridActorCriticConfig,
    HybridDomainActorCriticModel,
    hierarchical_greedy_action_index,
    initialize_actor_critic_from_macro_publication,
)
from sts2_rl.semantics.clock import decision_discount  # noqa: E402
from sts2_rl.semantics.forward import forward_decision  # noqa: E402
from sts2_rl.train_macro_credit import _actor_critic_config, load_macro_training_config  # noqa: E402
from sts2_rl.training.config import ModelConfig  # noqa: E402


def _tiny_v83_and_v84_configs() -> tuple[ModelConfig, HybridActorCriticConfig, HybridActorCriticConfig]:
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
        max_world_tokens=512,
        max_candidates=32,
        max_candidate_local_tokens=16,
    )
    v83_cfg = HybridActorCriticConfig(
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
        enable_marginal_card_id_durable_value=True,
        suppress_full_hp_campfire_heal=True,
    )
    v84_cfg = replace(
        v83_cfg,
        enable_centered_marginal_card_id_durable_value=True,
        enable_saturated_high_hp_basis=True,
        campfire_heal_suppress_hp_ratio=0.82,
    )
    return tiny, v83_cfg, v84_cfg


def _make_deck_world(
    encoder: GroundedObservationEncoder,
    *,
    hp: int = 60,
    max_hp: int = 80,
    strike_qty: int = 5,
    defend_qty: int = 5,
    extra_cards: list[dict[str, Any]] | None = None,
) -> EncodedWorldSnapshot:
    deck: list[dict[str, Any]] = [
        {"id": "STRIKE_IRONCLAD", "type": "Attack", "rarity": "Basic", "cost": 1, "quantity": strike_qty},
        {"id": "DEFEND_IRONCLAD", "type": "Skill", "rarity": "Basic", "cost": 1, "quantity": defend_qty},
    ]
    if extra_cards:
        deck.extend(extra_cards)
    obs: dict[str, Any] = {
        "phase": "map",
        "screen": "MAP",
        "decision_domain": "map",
        "run": {"active": True, "act": 1, "floor": 7, "gold": 99, "ascension": 0},
        "player": {
            "character_id": "IRONCLAD",
            "hp": hp,
            "current_hp": hp,
            "max_hp": max_hp,
            "gold": 99,
            "deck": deck,
            "relics": [{"id": "BURNING_BLOOD"}],
            "potions": [],
        },
        "combat": {"in_progress": False},
    }
    legal = [
        {
            "action_id": "map:0",
            "kind": "choose_map_node",
            "model_action_kind": "map",
            "node": {"x": 0, "y": 7, "room_type": "MONSTER"},
            "is_enabled": True,
        }
    ]
    snap = encoder.encode(obs, legal).snapshot
    return EncodedWorldSnapshot(
        world=snap.world,
        domain_id=snap.domain_id,
        config=snap.config,
        encoding_fingerprint=snap.encoding_fingerprint,
    )


def _make_campfire_surface(
    hp: int, max_hp: int = 80, *, include_smith: bool = True
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    deck = [
        {
            "id": "BASH",
            "type": "Attack",
            "rarity": "Basic",
            "cost": 2,
            "upgraded": False,
            "is_upgradable": include_smith,
            "upgrade_level": 0,
            "upgrade_preview": {
                "id": "BASH",
                "type": "Attack",
                "rarity": "Basic",
                "cost": 2,
                "upgraded": True,
                "is_upgraded": True,
                "upgrade_level": 1,
            },
        }
    ]
    options: list[dict[str, Any]] = [
        {"id": "HEAL", "type": "rest", "index": 0, "heal_amount": 24}
    ]
    native_legal: list[dict[str, Any]] = [
        {
            "action_id": "rest",
            "kind": "choose_rest_option",
            "option": {"id": "HEAL", "type": "rest", "index": 0, "heal_amount": 24},
            "is_enabled": True,
        }
    ]
    if include_smith:
        options.append({"id": "SMITH", "type": "smith", "index": 1})
        native_legal.append(
            {
                "action_id": "smith",
                "kind": "choose_rest_option",
                "option": {"id": "SMITH", "type": "smith", "index": 1},
                "is_enabled": True,
            }
        )
    obs: dict[str, Any] = {
        "phase": "rest_site",
        "screen": "REST_SITE",
        "decision_domain": "rest_site",
        "run": {"active": True, "act": 1, "floor": 6, "gold": 99, "ascension": 0},
        "player": {
            "character_id": "IRONCLAD",
            "hp": hp,
            "current_hp": hp,
            "max_hp": max_hp,
            "gold": 99,
            "deck": deck,
            "relics": [{"id": "BURNING_BLOOD"}],
            "potions": [],
        },
        "combat": {"in_progress": False},
        "rest_site": {"options": options},
    }
    comp = forward_decision(obs, native_legal)
    assert comp is not None
    sem_legal: list[dict[str, Any]] = [dict(c.semantic_action) for c in comp.candidates]
    return obs, sem_legal


def _deck_entity_id_set(world_snap: EncodedWorldSnapshot) -> set[int]:
    wb, _ = collate_encoded_worlds(
        (world_snap,),
        expected_config=world_snap.config,
        expected_fingerprint=world_snap.encoding_fingerprint,
        device=torch.device("cpu"),
    )
    zone_ids = wb.zone_ids[0]
    eids = wb.entity_ids[0]
    mask = wb.mask[0]
    return {
        int(eids[i].item())
        for i in range(int(mask.numel()))
        if bool(mask[i].item()) and int(zone_ids[i].item()) == 2 and int(eids[i].item()) > 0
    }


def test_v84_centered_marginal_cancels_survivorship_bias_and_keeps_trash_negative() -> None:
    tiny, v83_cfg, v84_cfg = _tiny_v83_and_v84_configs()
    torch.manual_seed(2026)
    parent_v83 = HybridDomainActorCriticModel(v83_cfg)
    enc = GroundedObservationEncoder(tiny.to_encoding_config())

    w_skip = _make_deck_world(
        enc,
        strike_qty=5,
        defend_qty=5,
        extra_cards=[{"id": "BASH", "type": "Attack", "rarity": "Basic", "cost": 2, "upgraded": False, "quantity": 1}],
    )
    w_smith_bash = _make_deck_world(
        enc,
        strike_qty=5,
        defend_qty=5,
        extra_cards=[{"id": "BASH", "type": "Attack", "rarity": "Basic", "cost": 2, "upgraded": True, "is_upgraded": True, "upgrade_level": 1, "quantity": 1}],
    )
    w_shrug = _make_deck_world(
        enc,
        strike_qty=5,
        defend_qty=5,
        extra_cards=[
            {"id": "BASH", "type": "Attack", "rarity": "Basic", "cost": 2, "upgraded": False, "quantity": 1},
            {"id": "SHRUG_IT_OFF", "type": "Skill", "rarity": "Common", "cost": 1, "quantity": 1},
        ],
    )
    w_mediocre = _make_deck_world(
        enc,
        strike_qty=5,
        defend_qty=5,
        extra_cards=[
            {"id": "BASH", "type": "Attack", "rarity": "Basic", "cost": 2, "upgraded": False, "quantity": 1},
            {"id": "WILD_STRIKE", "type": "Attack", "rarity": "Common", "cost": 1, "quantity": 1},
        ],
    )
    w_trash = _make_deck_world(
        enc,
        strike_qty=5,
        defend_qty=5,
        extra_cards=[
            {"id": "BASH", "type": "Attack", "rarity": "Basic", "cost": 2, "upgraded": False, "quantity": 1},
            {"id": "BLOODLETTING", "type": "Skill", "rarity": "Uncommon", "cost": 0, "quantity": 1},
        ],
    )

    w_base_sd = _make_deck_world(enc, strike_qty=5, defend_qty=5)
    base_sd_eids = _deck_entity_id_set(w_base_sd)
    skip_eids = _deck_entity_id_set(w_skip)
    eid_bash = next(iter(skip_eids - base_sd_eids))
    eid_shrug = next(iter(_deck_entity_id_set(w_shrug) - skip_eids))
    eid_mediocre = next(iter(_deck_entity_id_set(w_mediocre) - skip_eids))
    eid_trash = next(iter(_deck_entity_id_set(w_trash) - skip_eids))

    # Simulate v8.3 u=50 survivorship drift: +0.25 output bias + uniform +0.20 embedding shift
    # on top of true relative card quality (Shrug > 0 > Mediocre > Trash, Bash+ > Bash).
    # Zero untrained state-trunk readout weights so deck_utility (v_cycle + marginal_card_delta) is isolated cleanly.
    with torch.no_grad():
        for vh in (parent_v83.durable_value_head, parent_v83.durable_twin_value_head):
            vh_out = vh[-1]
            if isinstance(vh_out, torch.nn.Linear):
                vh_out.weight.zero_()
                if vh_out.bias is not None:
                    vh_out.bias.zero_()
        for emb, lin, mlp in (
            (
                parent_v83.durable_deck_card_id_emb,
                parent_v83.durable_deck_marginal_id_linear,
                parent_v83.durable_deck_marginal_mlp,
            ),
            (
                parent_v83.durable_twin_deck_card_id_emb,
                parent_v83.durable_twin_deck_marginal_id_linear,
                parent_v83.durable_twin_deck_marginal_mlp,
            ),
        ):
            lin.weight.zero_()
            lin.weight[0, 0] = 1.0
            emb.weight[2::2, 0] = 0.20
            emb.weight[eid_shrug * 2, 0] = 0.20 + 0.18
            emb.weight[eid_mediocre * 2, 0] = 0.20 - 0.05
            emb.weight[eid_trash * 2, 0] = 0.20 - 0.16
            emb.weight[eid_bash * 2 + 1, 0] = 0.20 + 0.18
            last_fc = mlp[-1]
            assert isinstance(last_fc, torch.nn.Linear)
            assert last_fc.bias is not None
            last_fc.bias.fill_(0.25)

    wb, dom = collate_encoded_worlds(
        (w_skip, w_smith_bash, w_shrug, w_mediocre, w_trash),
        expected_config=w_skip.config,
        expected_fingerprint=w_skip.encoding_fingerprint,
        device=torch.device("cpu"),
    )
    parent_v83.eval()
    with torch.no_grad():
        v83_vals, _ = parent_v83.forward_durable_value(wb, dom)
    # In uncentered v8.3, +0.25 bias + +0.20 shift made even mediocre & trash cards beat Skip!
    assert float((v83_vals[3] - v83_vals[0]).item()) > 0.15
    assert float((v83_vals[4] - v83_vals[0]).item()) > 0.05

    # Hot-start v8.4 from v8.3 publication state_dict
    student_v84 = HybridDomainActorCriticModel(v84_cfg)
    student_v84._detach_value_heads_from_trunk = True
    summary = initialize_actor_critic_from_macro_publication(student_v84, parent_v83.state_dict())
    assert "durable_deck_marginal_ema" in summary["fresh"]
    assert "durable_twin_deck_marginal_ema" in summary["fresh"]

    # Step 0 eval (before any training step): B=1 vs B=5 invariance AND inference_mode safety!
    student_v84.eval()
    with torch.inference_mode():
        v84_cold_batch, _ = student_v84.forward_durable_value(wb, dom)
        wb_single, dom_single = collate_encoded_worlds(
            (w_mediocre,),
            expected_config=w_skip.config,
            expected_fingerprint=w_skip.encoding_fingerprint,
            device=torch.device("cpu"),
        )
        v84_cold_single, _ = student_v84.forward_durable_value(wb_single, dom_single)
    assert abs(float(v84_cold_batch[3].item()) - float(v84_cold_single[0].item())) < 1e-6
    assert float((v84_cold_batch[1] - v84_cold_batch[0]).item()) > 0.10, "Smith BASH+ must beat Keep BASH at step 0"
    assert float((v84_cold_batch[2] - v84_cold_batch[0]).item()) > 0.08, "Take Shrug must beat Skip at step 0"
    assert float((v84_cold_batch[3] - v84_cold_batch[0]).item()) < -0.02, "Mediocre card must strictly lose to Skip at step 0"
    assert float((v84_cold_batch[4] - v84_cold_batch[0]).item()) < -0.10, "Trash card must strictly lose to Skip at step 0"

    # Adversarial survivorship training test: feed pure N-confounded targets (+0.8 * deck_size) to
    # MLP bias + uniform embedding shift and verify uniform-shift gradient is identically zero!
    student_v84.train()
    student_v84.zero_grad(set_to_none=True)
    v1_tr, v2_tr, _ = student_v84.forward_durable_value_pair(wb, dom)
    # Sum of gradients wrt output bias must be identically 0.0
    loss_bias_probe = v1_tr.sum() + v2_tr.sum()
    torch.autograd.backward(loss_bias_probe)
    mlp1_last = student_v84.durable_deck_marginal_mlp[-1]
    assert isinstance(mlp1_last, torch.nn.Linear) and mlp1_last.bias is not None
    assert mlp1_last.bias.grad is None or float(mlp1_last.bias.grad.abs().max().item()) < 1e-6

    # Check eval B=1 vs B=5 invariance AFTER EMA buffer initialization too
    student_v84.eval()
    with torch.inference_mode():
        v84_warm_batch, _ = student_v84.forward_durable_value(wb, dom)
        v84_warm_single, _ = student_v84.forward_durable_value(wb_single, dom_single)
    assert abs(float(v84_warm_batch[3].item()) - float(v84_warm_single[0].item())) < 1e-6
    assert float((v84_warm_batch[2] - v84_warm_batch[0]).item()) > 0.08
    assert float((v84_warm_batch[3] - v84_warm_batch[0]).item()) < -0.02
    assert float((v84_warm_batch[4] - v84_warm_batch[0]).item()) < -0.10


def test_v84_high_hp_campfire_suppression_82pct_and_saturated_hp_basis() -> None:
    tiny, v83_cfg, v84_cfg = _tiny_v83_and_v84_configs()
    enc = GroundedObservationEncoder(tiny.to_encoding_config())
    model_v83 = HybridDomainActorCriticModel(v83_cfg)
    model_v84 = HybridDomainActorCriticModel(v84_cfg)

    # 1. Saturated high-HP hp_basis check: 66/80 -> 80/80 gain is flattened vs 30/80 -> 54/80 emergency heal
    w_hp30 = _make_deck_world(enc, hp=30, max_hp=80)
    w_hp54 = _make_deck_world(enc, hp=54, max_hp=80)
    w_hp66 = _make_deck_world(enc, hp=66, max_hp=80)
    w_hp80 = _make_deck_world(enc, hp=80, max_hp=80)
    wb_hp, _ = collate_encoded_worlds(
        (w_hp30, w_hp54, w_hp66, w_hp80),
        expected_config=w_hp30.config,
        expected_fingerprint=w_hp30.encoding_fingerprint,
        device=torch.device("cpu"),
    )
    _, basis_v83, _ = model_v83._extract_monotone_stage_and_vitals(wb_hp)
    _, basis_v84, _ = model_v84._extract_monotone_stage_and_vitals(wb_hp)
    overheal_gain_v83 = float((basis_v83[3] - basis_v83[2]).sum().item())
    overheal_gain_v84 = float((basis_v84[3] - basis_v84[2]).sum().item())
    low_hp_gain_v84 = float((basis_v84[1] - basis_v84[0]).sum().item())
    assert overheal_gain_v84 >= 0.0
    assert overheal_gain_v84 < 0.15 * overheal_gain_v83, (
        f"saturated hp_basis should cut 66->80 overheal gain >6x (v83={overheal_gain_v83}, v84={overheal_gain_v84})"
    )
    assert low_hp_gain_v84 > 15.0 * overheal_gain_v84, (
        f"30->54 emergency heal must remain >> 66->80 overheal (low={low_hp_gain_v84}, high={overheal_gain_v84})"
    )

    # 2. Campfire HEAL suppression at >= 82% HP (66/80 = 0.825, 72/80 = 0.90, 80/80 = 1.00), free at 50/80 = 0.625
    for hp_val, expect_suppressed in ((80, True), (72, True), (66, True), (50, False)):
        obs_c, leg_c = _make_campfire_surface(hp_val, 80, include_smith=True)
        snap_c = enc.encode(obs_c, leg_c).snapshot
        batch_c = collate_encoded_snapshots(
            (snap_c,),
            expected_config=snap_c.config,
            expected_fingerprint=snap_c.encoding_fingerprint,
            device=torch.device("cpu"),
        )
        with torch.no_grad():
            out_c = model_v84(batch_c, validate=True)
            mask_c = batch_c.candidates.action_mask[0]
            probs_c = torch.softmax(out_c.policy_logits[0].masked_fill(~mask_c, -torch.inf), dim=-1)
        if expect_suppressed:
            assert float(probs_c[0].item()) < 1e-8
            assert float(probs_c[1].item()) > 0.999999
            assert (
                hierarchical_greedy_action_index(
                    out_c.policy_logits[0],
                    batch_c.candidates.policy_branch_ids[0],
                    mask_c,
                    domain_id=int(batch_c.domain_ids[0].item()),
                )
                == 1
            )
        else:
            assert float(probs_c[0].item()) > 0.05

    # 3. Teacher synchronization at 66/80 HP: attaching a v8.3 teacher syncs _campfire_heal_suppress_hp_ratio=0.82
    l_cfg = MacroCreditLearnerConfig(
        target_mode="durable_afterstate_v2",
        domain_layers=1,
        joint_layers=1,
        enable_direct_vitals_context=True,
        enable_direct_combat_vitals_context=True,
        enable_active_durable_vitals_gates=True,
        enable_marginal_card_id_durable_value=True,
        enable_centered_marginal_card_id_durable_value=True,
        enable_saturated_high_hp_basis=True,
        suppress_full_hp_campfire_heal=True,
        campfire_heal_suppress_hp_ratio=0.82,
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
        enable_local_combat_critic=True,
    )
    learner = MacroCreditLearner(model=model_v84, config=l_cfg, update=6, teacher_model=model_v83)
    assert learner.teacher_model is not None
    obs66, leg66 = _make_campfire_surface(66, 80, include_smith=True)
    snap66 = enc.encode(obs66, leg66).snapshot
    b66 = collate_encoded_snapshots(
        (snap66,),
        expected_config=snap66.config,
        expected_fingerprint=snap66.encoding_fingerprint,
        device=torch.device("cpu"),
    )
    with torch.no_grad():
        t_out66 = learner.teacher_model(b66, validate=True)
        t_probs66 = torch.softmax(
            t_out66.policy_logits[0].masked_fill(~b66.candidates.action_mask[0], -torch.inf), dim=-1
        )
    assert float(t_probs66[0].item()) < 1e-8


def test_v84_undiluted_active_afterstate_parent_macro_kl_cap_and_prior_mix() -> None:
    tiny, _, v84_cfg = _tiny_v83_and_v84_configs()
    torch.manual_seed(314)
    model = HybridDomainActorCriticModel(v84_cfg)
    teacher = copy.deepcopy(model)
    enc = GroundedObservationEncoder(tiny.to_encoding_config())

    # Inject strong preference for card 0 in V_durable so MPO tries to push card 0 -> 99% across updates
    map_obs: dict[str, Any] = {
        "phase": "map",
        "screen": "MAP",
        "decision_domain": "map",
        "run": {"active": True, "act": 1, "floor": 1, "gold": 99, "ascension": 0},
        "player": {
            "character_id": "IRONCLAD",
            "hp": 60,
            "current_hp": 60,
            "max_hp": 80,
            "gold": 99,
            "deck": [
                {"id": "STRIKE_IRONCLAD", "cost": 1, "quantity": 5},
                {"id": "DEFEND_IRONCLAD", "cost": 1, "quantity": 5},
            ],
        },
        "combat": {"in_progress": False},
        "map": {
            "visible": True,
            "nodes": [
                {"x": 0, "y": 1, "room_type": "MONSTER", "reachable": True},
                {"x": 2, "y": 1, "room_type": "EVENT", "reachable": True},
            ],
        },
    }
    map_leg: list[dict[str, Any]] = [
        {"action_id": "map:0", "kind": "choose_map_node", "model_action_kind": "map", "node": {"x": 0, "y": 1, "room_type": "MONSTER"}, "is_enabled": True},
        {"action_id": "map:1", "kind": "choose_map_node", "model_action_kind": "map", "node": {"x": 2, "y": 1, "room_type": "EVENT"}, "is_enabled": True},
    ]
    rew_obs: dict[str, Any] = {
        "phase": "card_reward",
        "screen": "CARD_REWARD",
        "decision_domain": "card_reward",
        "run": {"active": True, "act": 1, "floor": 4, "gold": 99, "ascension": 0},
        "player": copy.deepcopy(map_obs["player"]),
        "combat": {"in_progress": False},
        "card_reward_selection": {
            "visible": True,
            "cards": [
                {"index": 0, "id": "SHRUG_IT_OFF", "type": "Skill", "rarity": "Common", "cost": 1},
                {"index": 1, "id": "BLOODLETTING", "type": "Skill", "rarity": "Uncommon", "cost": 0},
                {"index": 2, "id": "CLEAVE", "type": "Attack", "rarity": "Common", "cost": 1},
            ],
            "can_skip": True,
        },
    }
    rew_leg = [
        {"action_id": "take:0", "kind": "select_card_reward", "model_action_kind": "card_reward", "model_action_variant": "take", "card": rew_obs["card_reward_selection"]["cards"][0], "is_enabled": True},
        {"action_id": "take:1", "kind": "select_card_reward", "model_action_kind": "card_reward", "model_action_variant": "take", "card": rew_obs["card_reward_selection"]["cards"][1], "is_enabled": True},
        {"action_id": "take:2", "kind": "select_card_reward", "model_action_kind": "card_reward", "model_action_variant": "take", "card": rew_obs["card_reward_selection"]["cards"][2], "is_enabled": True},
        {"action_id": "skip", "kind": "skip_card_reward", "model_action_kind": "card_reward", "model_action_variant": "skip", "is_enabled": True},
    ]

    map_snaps = []
    for fl in (1, 2, 3):
        mo = copy.deepcopy(map_obs)
        mo["run"]["floor"] = fl
        map_snaps.append(enc.encode(mo, map_leg).snapshot)
    rew_snap = enc.encode(rew_obs, rew_leg).snapshot

    post_worlds: list[EncodedWorldSnapshot] = []
    for i in range(4):
        po = copy.deepcopy(map_obs)
        po["run"]["floor"] = 4
        if i < 3:
            c_i = rew_obs["card_reward_selection"]["cards"][i]
            po["player"]["deck"].append({"id": c_i["id"], "cost": c_i["cost"], "quantity": 1})
        ps = enc.encode(po, [map_leg[0]]).snapshot
        post_worlds.append(
            EncodedWorldSnapshot(
                world=ps.world,
                domain_id=ps.domain_id,
                config=ps.config,
                encoding_fingerprint=ps.encoding_fingerprint,
            )
        )

    l_cfg = MacroCreditLearnerConfig(
        target_mode="durable_afterstate_v2",
        domain_layers=1,
        joint_layers=1,
        enable_direct_vitals_context=True,
        enable_direct_combat_vitals_context=True,
        enable_active_durable_vitals_gates=True,
        enable_marginal_card_id_durable_value=True,
        enable_centered_marginal_card_id_durable_value=True,
        enable_saturated_high_hp_basis=True,
        suppress_full_hp_campfire_heal=True,
        campfire_heal_suppress_hp_ratio=0.82,
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
        enable_local_combat_critic=True,
        enable_hierarchical_afterstate_mpo=True,
        detach_value_heads_from_trunk=True,
        detach_macro_branch_prior_lse=True,
        isolate_nonreward_macro_trunk_grads=True,
        hybrid_afterstate_vtrace_alpha=0.25,
        afterstate_macro_pg_scale=0.35,
        macro_pg_fallback_scale=0.0,
        mpo_parent_prior_mix=0.20,
        max_parent_macro_kl=0.12,
        target_macro_step_kl=0.50,
        enable_post_step_kl_line_search=True,
        learning_rate=5e-3,
        durable_warmup_updates=0,
        combat_actor_warmup_updates=0,
        macro_actor_warmup_updates=0,
        likelihood_tolerance=0.15,
    )
    learner = MacroCreditLearner(model=model, config=l_cfg, update=0, teacher_model=teacher)

    def _build_traj(pol_ver: int) -> MacroCreditTrajectory:
        with torch.no_grad():
            h = model.initial_state(1, device=torch.device("cpu"))
            lps: list[float] = []
            for s in map_snaps:
                bs = collate_encoded_snapshots((s,), expected_config=s.config, expected_fingerprint=s.encoding_fingerprint, device=torch.device("cpu"))
                out_s = model(bs, h, validate=False)
                h = out_s.recurrent_state
                lps.append(float(torch.log_softmax(out_s.policy_logits[0].masked_fill(~bs.candidates.action_mask[0], -torch.inf), dim=-1)[0].item()))
            br = collate_encoded_snapshots((rew_snap,), expected_config=rew_snap.config, expected_fingerprint=rew_snap.encoding_fingerprint, device=torch.device("cpu"))
            out_r = model(br, h, validate=False)
            lp_r0 = float(torch.log_softmax(out_r.policy_logits[0].masked_fill(~br.candidates.action_mask[0], -torch.inf), dim=-1)[0].item())
        rfs = (
            MacroRewardFacts(0.0, 0.0, 0.0, 0.0, 0.0, 0, 0.0, 1, 2, 1),
            MacroRewardFacts(0.0, 0.0, 0.0, 0.0, 0.0, 0, 0.0, 2, 3, 1),
            MacroRewardFacts(0.0, 0.0, 0.0, 0.0, 0.0, 0, 0.0, 3, 4, 1),
            MacroRewardFacts(1.0, 1.0, 0.0, 0.0, 0.0, 0, 0.0, 4, 5, 1),
        )
        steps = (
            HybridVTraceStep(map_snaps[0], 0, lps[0], rfs[0].reward, decision_discount(floor_before=1, floor_after=2, terminal=False), False, "map", "monster", "macro", pol_ver),
            HybridVTraceStep(map_snaps[1], 0, lps[1], rfs[1].reward, decision_discount(floor_before=2, floor_after=3, terminal=False), False, "map", "monster", "macro", pol_ver),
            HybridVTraceStep(map_snaps[2], 0, lps[2], rfs[2].reward, decision_discount(floor_before=3, floor_after=4, terminal=False), False, "map", "monster", "macro", pol_ver),
            HybridVTraceStep(rew_snap, 0, lp_r0, rfs[3].reward, decision_discount(floor_before=4, floor_after=5, terminal=True), True, "reward", "take", "macro", pol_ver),
        )
        return MacroCreditTrajectory(
            f"nat-{pol_ver}",
            2,
            pol_ver,
            (),
            steps,
            None,
            None,
            "factual_terminal",
            rfs,
            afterstate_nodes=(
                MacroAfterstateNode(
                    step_index=3,
                    surface="reward",
                    candidate_branches=("take", "take", "take", "skip"),
                    post_worlds=tuple(post_worlds),
                ),
            ),
            macro_branch_epsilon=0.0,
        )

    # Even with 3 untouched Map steps (4:1 total:active macro ratio) and huge lr=5e-3 over 6 updates,
    # un-diluted active-afterstate parent Macro KL strictly stays <= 0.12!
    for u in range(6):
        tr = _build_traj(u)
        m = learner.learn(MacroCreditCohort(f"c-{u}", u, (), natural_only=(tr,)))
        assert float(m["mean_updated_parent_macro_policy_kl"]) <= 0.12 + 1e-5, (
            f"un-diluted parent Macro KL exceeded 0.12 at u={u}: {m['mean_updated_parent_macro_policy_kl']}"
        )

    # Verify directly on the Card Reward state that exact KL(parent || student) <= 0.12 + 1e-5!
    with torch.no_grad():
        h_t = teacher.initial_state(1, device=torch.device("cpu"))
        h_s = model.initial_state(1, device=torch.device("cpu"))
        for s in map_snaps:
            bs = collate_encoded_snapshots((s,), expected_config=s.config, expected_fingerprint=s.encoding_fingerprint, device=torch.device("cpu"))
            h_t = teacher(bs, h_t, validate=False).recurrent_state
            h_s = model(bs, h_s, validate=False).recurrent_state
        br = collate_encoded_snapshots((rew_snap,), expected_config=rew_snap.config, expected_fingerprint=rew_snap.encoding_fingerprint, device=torch.device("cpu"))
        mask_r = br.candidates.action_mask[0]
        lp_t = torch.log_softmax(teacher(br, h_t, validate=False).policy_logits[0].masked_fill(~mask_r, -torch.inf), dim=-1)
        lp_s = torch.log_softmax(model(br, h_s, validate=False).policy_logits[0].masked_fill(~mask_r, -torch.inf), dim=-1)
        val_r = torch.isfinite(lp_t) & torch.isfinite(lp_s)
        true_active_parent_kl = float((lp_t[val_r].exp() * (lp_t[val_r] - lp_s[val_r])).sum().item())
    assert true_active_parent_kl <= 0.12 + 1e-5, (
        f"true un-diluted Card Reward KL(parent||student) exceeded 0.12: {true_active_parent_kl}"
    )


def test_v84_production_toml_and_roundtrip_config_invariants() -> None:
    cfg_path = (
        Path(__file__).resolve().parents[1]
        / "config"
        / "experiments"
        / "hybrid_macro_credit_v33_durable_afterstate_v84_centered_marginal_anchored_b0_act1_g4.toml"
    )
    cfg = load_macro_training_config(cfg_path)
    assert cfg.course.roots_per_cohort == 96
    assert cfg.course.parallel_collectors == 48
    assert cfg.course.seed_index_offset == 870_000
    assert cfg.learner.enable_marginal_card_id_durable_value is True
    assert cfg.learner.enable_centered_marginal_card_id_durable_value is True
    assert cfg.learner.enable_saturated_high_hp_basis is True
    assert cfg.learner.suppress_full_hp_campfire_heal is True
    assert cfg.learner.campfire_heal_suppress_hp_ratio == pytest.approx(0.82)
    assert cfg.learner.mpo_parent_prior_mix == pytest.approx(0.20)
    assert cfg.learner.max_parent_macro_kl == pytest.approx(0.12)
    assert cfg.learner.hybrid_afterstate_vtrace_alpha == pytest.approx(0.25)
    assert cfg.learner.afterstate_macro_pg_scale == pytest.approx(0.35)
    assert cfg.learner.macro_pg_fallback_scale == 0.0
    assert cfg.learner.target_combat_step_kl == pytest.approx(0.004)
    assert cfg.learner.kickstart_kl_floor == pytest.approx(0.015)

    ac_cfg = _actor_critic_config(cfg)
    assert ac_cfg.enable_centered_marginal_card_id_durable_value is True
    assert ac_cfg.enable_saturated_high_hp_basis is True
    assert ac_cfg.campfire_heal_suppress_hp_ratio == pytest.approx(0.82)
    parsed = HybridActorCriticConfig.from_mapping(
        ac_cfg.as_mapping(), fallback_grounded=ac_cfg.grounded
    )
    assert parsed == ac_cfg
