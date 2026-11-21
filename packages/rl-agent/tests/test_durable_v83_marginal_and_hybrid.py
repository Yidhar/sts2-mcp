"""Tests for Durable Afterstate v8.3: Marginal Card-ID V_durable, Full-HP Heal Guard, Hybrid Afterstate V-trace/PPO, and Gentle Anchored Combat."""

from __future__ import annotations

import copy
from dataclasses import replace
from pathlib import Path
from typing import Any

import pytest

torch = pytest.importorskip("torch")

from sts2_rl.encoding import GroundedObservationEncoder  # noqa: E402
from sts2_rl.encoding.snapshot import EncodedWorldSnapshot, collate_encoded_snapshots, collate_encoded_worlds  # noqa: E402
from sts2_rl.joint.macro_credit_data import (  # noqa: E402
    MacroAfterstateNode,
    MacroCreditCohort,
    MacroCreditTrajectory,
    MacroRewardFacts,
)
from sts2_rl.joint.macro_credit_learner import MacroCreditLearner, MacroCreditLearnerConfig  # noqa: E402
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


def _tiny_v82_and_v83_configs() -> tuple[ModelConfig, HybridActorCriticConfig, HybridActorCriticConfig]:
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
    v82_cfg = HybridActorCriticConfig(
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
    v83_cfg = replace(
        v82_cfg,
        enable_marginal_card_id_durable_value=True,
        suppress_full_hp_campfire_heal=True,
    )
    return tiny, v82_cfg, v83_cfg


def _make_deck_world(
    encoder: GroundedObservationEncoder,
    *,
    strike_qty: int = 8,
    defend_qty: int = 8,
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
            "hp": 60,
            "current_hp": 60,
            "max_hp": 80,
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


def _make_campfire_surface(hp: int, max_hp: int = 80, *, include_smith: bool = True) -> tuple[dict[str, Any], list[dict[str, Any]]]:
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
    options: list[dict[str, Any]] = [{"id": "HEAL", "type": "rest", "index": 0, "heal_amount": 24}]
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
    sem_legal = [c.semantic_action for c in comp.candidates]
    return obs, sem_legal


def test_v83_hotstart_bit_exact_and_undiluted_marginal_card_id_learning() -> None:
    tiny, v82_cfg, v83_cfg = _tiny_v82_and_v83_configs()
    torch.manual_seed(123)
    parent = HybridDomainActorCriticModel(v82_cfg)
    student = HybridDomainActorCriticModel(v83_cfg)
    student._detach_value_heads_from_trunk = True

    summary = initialize_actor_critic_from_macro_publication(student, parent.state_dict())
    assert any(k.startswith("durable_deck_card_id_emb.") for k in summary["fresh"])
    assert any(k.startswith("durable_twin_deck_card_id_emb.") for k in summary["fresh"])

    enc = GroundedObservationEncoder(tiny.to_encoding_config())
    w_small_bash0 = _make_deck_world(
        enc,
        strike_qty=4,
        defend_qty=4,
        extra_cards=[{"id": "BASH", "type": "Attack", "rarity": "Basic", "cost": 2, "upgraded": False, "quantity": 1}],
    )
    w_large_bash0 = _make_deck_world(
        enc,
        strike_qty=10,
        defend_qty=10,
        extra_cards=[{"id": "BASH", "type": "Attack", "rarity": "Basic", "cost": 2, "upgraded": False, "quantity": 1}],
    )
    w_large_bash1 = _make_deck_world(
        enc,
        strike_qty=10,
        defend_qty=10,
        extra_cards=[{"id": "BASH", "type": "Attack", "rarity": "Basic", "cost": 2, "upgraded": True, "is_upgraded": True, "upgrade_level": 1, "quantity": 1}],
    )
    w_large_shrug = _make_deck_world(
        enc,
        strike_qty=10,
        defend_qty=10,
        extra_cards=[
            {"id": "BASH", "type": "Attack", "rarity": "Basic", "cost": 2, "upgraded": False, "quantity": 1},
            {"id": "SHRUG_IT_OFF", "type": "Skill", "rarity": "Common", "cost": 1, "quantity": 1},
        ],
    )
    w_large_bleed = _make_deck_world(
        enc,
        strike_qty=10,
        defend_qty=10,
        extra_cards=[
            {"id": "BASH", "type": "Attack", "rarity": "Basic", "cost": 2, "upgraded": False, "quantity": 1},
            {"id": "BLOODLETTING", "type": "Skill", "rarity": "Uncommon", "cost": 0, "quantity": 1},
        ],
    )

    wb, dom = collate_encoded_worlds(
        (w_small_bash0, w_large_bash0, w_large_bash1, w_large_shrug, w_large_bleed),
        expected_config=w_small_bash0.config,
        expected_fingerprint=w_small_bash0.encoding_fingerprint,
        device=torch.device("cpu"),
    )
    with torch.no_grad():
        p_v1, p_v2, _ = parent.forward_durable_value_pair(wb, dom)
        s_v1, s_v2, _ = student.forward_durable_value_pair(wb, dom)
    assert torch.allclose(p_v1, s_v1, atol=1e-6), "v8.3 hot-start v1 must be bit-identical at step 0"
    assert torch.allclose(p_v2, s_v2, atol=1e-6), "v8.3 hot-start v2 must be bit-identical at step 0"

    # Verify O(1) gradient magnitude independent of deck size N=9 vs N=21 (no 1/N dilution!)
    student.zero_grad(set_to_none=True)
    v1_small, _, _ = student.forward_durable_value_pair(
        *collate_encoded_worlds((w_small_bash0,), expected_config=w_small_bash0.config, expected_fingerprint=w_small_bash0.encoding_fingerprint, device=torch.device("cpu"))
    )
    v1_small[0].backward()
    g_small = float(student.durable_deck_marginal_id_linear.weight.grad.norm().item())

    student.zero_grad(set_to_none=True)
    v1_large, _, _ = student.forward_durable_value_pair(
        *collate_encoded_worlds((w_large_bash0,), expected_config=w_large_bash0.config, expected_fingerprint=w_large_bash0.encoding_fingerprint, device=torch.device("cpu"))
    )
    v1_large[0].backward()
    g_large = float(student.durable_deck_marginal_id_linear.weight.grad.norm().item())
    assert g_large > 0.05
    assert g_large >= 0.85 * g_small, f"marginal card-id gradient must not shrink by 9/21 (got small={g_small}, large={g_large})"

    # Train ONLY durable_deck_* / durable_twin_* with trunk detached (entity_embedding frozen!)
    trunk_emb_before = student.token_embedder.entity_embedding.weight.detach().clone()
    opt = torch.optim.AdamW(
        [p for n, p in student.named_parameters() if n.startswith(("durable_deck_", "durable_twin_"))],
        lr=3e-3,
    )
    targets = torch.tensor([0.0, 0.0, 0.14, 0.12, -0.18], dtype=torch.float32)
    for _ in range(65):
        opt.zero_grad(set_to_none=True)
        v1_pred, v2_pred, _ = student.forward_durable_value_pair(wb, dom)
        loss = torch.nn.functional.smooth_l1_loss(v1_pred, targets) + torch.nn.functional.smooth_l1_loss(v2_pred, targets)
        loss.backward()
        opt.step()

    assert torch.equal(student.token_embedder.entity_embedding.weight, trunk_emb_before)
    with torch.no_grad():
        v_post, _ = student.forward_durable_value(wb, dom)
    smith_delta = float((v_post[2] - v_post[1]).item())
    shrug_delta = float((v_post[3] - v_post[1]).item())
    bleed_delta = float((v_post[4] - v_post[1]).item())
    assert smith_delta > 0.06, f"Smith BASH -> BASH+ at N=21 should achieve O(+0.07) delta, got {smith_delta}"
    assert shrug_delta > 0.05, f"Take SHRUG_IT_OFF at N=21 should achieve positive delta, got {shrug_delta}"
    assert bleed_delta < -0.08, f"Take BLOODLETTING at N=21 should achieve negative delta, got {bleed_delta}"


def test_v83_full_hp_campfire_heal_suppression_and_teacher_sync() -> None:
    tiny, v82_cfg, v83_cfg = _tiny_v82_and_v83_configs()
    enc = GroundedObservationEncoder(tiny.to_encoding_config())
    model = HybridDomainActorCriticModel(v83_cfg)

    obs_80, leg_80 = _make_campfire_surface(80, 80, include_smith=True)
    obs_79, leg_79 = _make_campfire_surface(79, 80, include_smith=True)
    obs_40, leg_40 = _make_campfire_surface(40, 80, include_smith=True)
    obs_only_heal, leg_only_heal = _make_campfire_surface(80, 80, include_smith=False)

    for obs, leg, expect_suppressed in ((obs_80, leg_80, True), (obs_79, leg_79, True), (obs_40, leg_40, False)):
        snap = enc.encode(obs, leg).snapshot
        batch = collate_encoded_snapshots(
            (snap,),
            expected_config=snap.config,
            expected_fingerprint=snap.encoding_fingerprint,
            device=torch.device("cpu"),
        )
        with torch.no_grad():
            out = model(batch, validate=True)
            mask = batch.candidates.action_mask[0]
            probs = torch.softmax(out.policy_logits[0].masked_fill(~mask, -torch.inf), dim=-1)
        if expect_suppressed:
            assert float(probs[0].item()) < 1e-8
            assert float(probs[1].item()) > 0.999999
            greedy_idx = hierarchical_greedy_action_index(
                out.policy_logits[0],
                batch.candidates.policy_branch_ids[0],
                mask,
                domain_id=int(batch.domain_ids[0].item()),
            )
            assert greedy_idx == 1
        else:
            assert float(probs[0].item()) > 0.05

    # Safety fallback: when ONLY rest is legal at 80/80, distribution remains valid and finite
    snap_oh = enc.encode(obs_only_heal, leg_only_heal).snapshot
    batch_oh = collate_encoded_snapshots(
        (snap_oh,),
        expected_config=snap_oh.config,
        expected_fingerprint=snap_oh.encoding_fingerprint,
        device=torch.device("cpu"),
    )
    with torch.no_grad():
        out_oh = model(batch_oh, validate=True)
        probs_oh = torch.softmax(out_oh.policy_logits[0].masked_fill(~batch_oh.candidates.action_mask[0], -torch.inf), dim=-1)
    assert abs(float(probs_oh[0].item()) - 1.0) < 1e-6

    # Parent teacher sync: attaching a v8.2 teacher to a v8.3 learner synchronizes full-HP heal suppression
    teacher = HybridDomainActorCriticModel(v82_cfg)
    l_cfg = MacroCreditLearnerConfig(
        target_mode="durable_afterstate_v2",
        domain_layers=1,
        joint_layers=1,
        enable_direct_vitals_context=True,
        enable_direct_combat_vitals_context=True,
        enable_active_durable_vitals_gates=True,
        enable_marginal_card_id_durable_value=True,
        suppress_full_hp_campfire_heal=True,
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
    learner = MacroCreditLearner(model=model, config=l_cfg, update=6, teacher_model=teacher)
    assert learner.teacher_model is not None
    snap80 = enc.encode(obs_80, leg_80).snapshot
    b80 = collate_encoded_snapshots(
        (snap80,), expected_config=snap80.config, expected_fingerprint=snap80.encoding_fingerprint, device=torch.device("cpu")
    )
    with torch.no_grad():
        t_out = learner.teacher_model(b80, validate=True)
        t_probs = torch.softmax(t_out.policy_logits[0].masked_fill(~b80.candidates.action_mask[0], -torch.inf), dim=-1)
    assert float(t_probs[0].item()) < 1e-8


def test_v83_hybrid_afterstate_vtrace_and_clipped_pg_push_down_losing_card_and_lift_skip() -> None:
    tiny, _, v83_cfg = _tiny_v82_and_v83_configs()
    torch.manual_seed(77)
    model = HybridDomainActorCriticModel(v83_cfg)
    enc = GroundedObservationEncoder(tiny.to_encoding_config())

    # Build a natural trajectory: Map -> Card Reward (picks Bloodletting idx=1) -> Combat (dies, reward=-1.0)
    map_obs = {
        "phase": "map",
        "screen": "MAP",
        "decision_domain": "map",
        "run": {"active": True, "act": 1, "floor": 3, "gold": 99, "ascension": 0},
        "player": {"character_id": "IRONCLAD", "hp": 50, "current_hp": 50, "max_hp": 80, "gold": 99, "deck": [{"id": "STRIKE_IRONCLAD", "cost": 1, "quantity": 5}, {"id": "DEFEND_IRONCLAD", "cost": 1, "quantity": 5}]},
        "combat": {"in_progress": False},
        "map": {"visible": True, "nodes": [{"x": 0, "y": 3, "room_type": "MONSTER", "reachable": True}, {"x": 2, "y": 3, "room_type": "ELITE", "reachable": True}]},
    }
    map_leg = [
        {"action_id": "map:0", "kind": "choose_map_node", "model_action_kind": "map", "node": {"x": 0, "y": 3, "room_type": "MONSTER"}, "is_enabled": True},
        {"action_id": "map:1", "kind": "choose_map_node", "model_action_kind": "map", "node": {"x": 2, "y": 3, "room_type": "ELITE"}, "is_enabled": True},
    ]
    rew_obs = {
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
    com_obs = {
        "phase": "combat",
        "screen": "COMBAT",
        "decision_domain": "combat",
        "run": {"active": True, "act": 1, "floor": 5, "gold": 99, "ascension": 0},
        "player": copy.deepcopy(map_obs["player"]),
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
            "enemies": [{"id": "JAW_WORM", "hp": 40, "max_hp": 40, "block": 0, "intent": "Attack", "intent_damage": 20, "intent_hits": 1}],
        },
    }
    com_leg = [
        {"action_id": "play:0", "kind": "play_card", "model_action_kind": "play_card", "card_index": 0, "target_index": 0, "is_enabled": True},
        {"action_id": "play:1", "kind": "play_card", "model_action_kind": "play_card", "card_index": 1, "is_enabled": True},
    ]

    map_snap = enc.encode(map_obs, map_leg).snapshot
    rew_snap = enc.encode(rew_obs, rew_leg).snapshot
    com_snap = enc.encode(com_obs, com_leg).snapshot

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

    def _step_probs(mod: HybridDomainActorCriticModel) -> tuple[torch.Tensor, torch.Tensor]:
        with torch.no_grad():
            h = mod.initial_state(1, device=torch.device("cpu"))
            b_m = collate_encoded_snapshots((map_snap,), expected_config=map_snap.config, expected_fingerprint=map_snap.encoding_fingerprint, device=torch.device("cpu"))
            out_m = mod(b_m, h, validate=False)
            p_m = torch.softmax(out_m.policy_logits[0].masked_fill(~b_m.candidates.action_mask[0], -torch.inf), dim=-1)
            b_r = collate_encoded_snapshots((rew_snap,), expected_config=rew_snap.config, expected_fingerprint=rew_snap.encoding_fingerprint, device=torch.device("cpu"))
            out_r = mod(b_r, out_m.recurrent_state, validate=False)
            p_r = torch.softmax(out_r.policy_logits[0].masked_fill(~b_r.candidates.action_mask[0], -torch.inf), dim=-1)
            return p_m, p_r

    map_probs_before, rew_probs_before = _step_probs(model)

    with torch.no_grad():
        h0 = model.initial_state(1, device=torch.device("cpu"))
        bm = collate_encoded_snapshots((map_snap,), expected_config=map_snap.config, expected_fingerprint=map_snap.encoding_fingerprint, device=torch.device("cpu"))
        om = model(bm, h0, validate=False)
        lp_m = float(torch.log_softmax(om.policy_logits[0].masked_fill(~bm.candidates.action_mask[0], -torch.inf), dim=-1)[0].item())
        br = collate_encoded_snapshots((rew_snap,), expected_config=rew_snap.config, expected_fingerprint=rew_snap.encoding_fingerprint, device=torch.device("cpu"))
        orr = model(br, om.recurrent_state, validate=False)
        lp_r = float(torch.log_softmax(orr.policy_logits[0].masked_fill(~br.candidates.action_mask[0], -torch.inf), dim=-1)[1].item())
        bc = collate_encoded_snapshots((com_snap,), expected_config=com_snap.config, expected_fingerprint=com_snap.encoding_fingerprint, device=torch.device("cpu"))
        oc = model(bc, orr.recurrent_state, validate=False)
        lp_c = float(torch.log_softmax(oc.policy_logits[0].masked_fill(~bc.candidates.action_mask[0], -torch.inf), dim=-1)[0].item())

    rf0 = MacroRewardFacts(0.0, 0.0, 0.0, 0.0, 0.0, 0, 0.0, 3, 4, 1)
    rf1 = MacroRewardFacts(0.0, 0.0, 0.0, 0.0, 0.0, 0, 0.0, 4, 5, 1)
    rf2 = MacroRewardFacts(-1.0, 0.0, 0.0, 0.0, 0.0, 0, 50.0, 5, 5, 1)

    steps = (
        HybridVTraceStep(
            map_snap,
            0,
            lp_m,
            rf0.reward,
            decision_discount(floor_before=3, floor_after=4, terminal=False),
            False,
            "map",
            "monster",
            "macro",
            6,
        ),
        HybridVTraceStep(
            rew_snap,
            1,
            lp_r,
            rf1.reward,
            decision_discount(floor_before=4, floor_after=5, terminal=False),
            False,
            "reward",
            "take",
            "macro",
            6,
        ),
        HybridVTraceStep(
            com_snap,
            0,
            lp_c,
            rf2.reward,
            decision_discount(floor_before=5, floor_after=5, terminal=True),
            True,
            "combat",
            "play",
            "combat",
            6,
        ),
    )
    traj = MacroCreditTrajectory(
        "nat-0",
        2,
        6,
        (),
        steps,
        None,
        None,
        "factual_terminal",
        (rf0, rf1, rf2),
        afterstate_nodes=(
            MacroAfterstateNode(
                step_index=1,
                surface="reward",
                candidate_branches=("take", "take", "take", "skip"),
                post_worlds=tuple(post_worlds),
            ),
        ),
        macro_branch_epsilon=0.0,
    )

    teacher = copy.deepcopy(model)
    l_cfg = MacroCreditLearnerConfig(
        target_mode="durable_afterstate_v2",
        domain_layers=1,
        joint_layers=1,
        enable_direct_vitals_context=True,
        enable_direct_combat_vitals_context=True,
        enable_active_durable_vitals_gates=True,
        enable_marginal_card_id_durable_value=True,
        suppress_full_hp_campfire_heal=True,
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
        hybrid_afterstate_vtrace_alpha=0.30,
        hybrid_afterstate_vtrace_clip=0.50,
        afterstate_macro_pg_scale=0.40,
        macro_pg_fallback_scale=0.0,
        clip_macro_pg_fallback=True,
        macro_pg_adv_clip=0.35,
        learning_rate=1e-3,
        durable_value_lr_scale=2.5,
        combat_policy_weight=0.5,
        combat_value_weight=1.0,
        combat_entropy_weight=0.012,
        target_combat_step_kl=0.004,
        enable_post_step_kl_line_search=True,
        kickstart_kl_weight=0.020,
        kickstart_kl_floor=0.015,
        kickstart_combat_kl_scale=1.0,
        durable_warmup_updates=5,
        combat_actor_warmup_updates=5,
        macro_actor_warmup_updates=5,
        likelihood_tolerance=0.15,
    )
    learner = MacroCreditLearner(model=model, config=l_cfg, update=6, teacher_model=teacher)
    cohort = MacroCreditCohort("cohort-v83", 6, (), natural_only=(traj,))
    metrics = learner.learn(cohort)

    map_probs_after, rew_probs_after = _step_probs(model)
    assert float(rew_probs_after[1].item()) < float(rew_probs_before[1].item()), (
        f"losing Bloodletting pick must drop immediately (before={rew_probs_before[1]}, after={rew_probs_after[1]})"
    )
    assert float(rew_probs_after[3].item()) > float(rew_probs_before[3].item()), (
        f"Skip probability must rise when taking Bloodletting loses (before={rew_probs_before[3]}, after={rew_probs_after[3]})"
    )
    assert float(metrics["mean_updated_combat_policy_kl"]) <= 0.0045
    assert abs(float((map_probs_after[0] - map_probs_before[0]).item())) < 0.02


def test_v83_production_toml_config_invariants() -> None:
    cfg_path = (
        Path(__file__).resolve().parents[1]
        / "config"
        / "experiments"
        / "hybrid_macro_credit_v32_durable_afterstate_v83_marginal_hybrid_b0_act1_g4.toml"
    )
    cfg = load_macro_training_config(cfg_path)
    assert cfg.course.roots_per_cohort == 96
    assert cfg.course.parallel_collectors == 48
    assert cfg.course.crn_roots_per_episode == 0
    assert cfg.learner.ppo_minibatches_per_cohort == 1
    assert cfg.learner.enable_marginal_card_id_durable_value is True
    assert cfg.learner.suppress_full_hp_campfire_heal is True
    assert cfg.learner.hybrid_afterstate_vtrace_alpha == pytest.approx(0.25)
    assert cfg.learner.afterstate_macro_pg_scale == pytest.approx(0.35)
    assert cfg.learner.macro_pg_fallback_scale == 0.0
    assert cfg.learner.combat_policy_weight == pytest.approx(0.5)
    assert cfg.learner.combat_value_weight == pytest.approx(1.0)
    assert cfg.learner.combat_entropy_weight == pytest.approx(0.012)
    assert cfg.learner.target_combat_step_kl == pytest.approx(0.004)
    assert cfg.learner.kickstart_kl_floor == pytest.approx(0.015)
    assert cfg.learner.kickstart_combat_kl_scale == pytest.approx(1.0)

    ac_cfg = _actor_critic_config(cfg)
    assert ac_cfg.enable_marginal_card_id_durable_value is True
    assert ac_cfg.suppress_full_hp_campfire_heal is True
