"""Unit tests for Durable Afterstate Macro Credit v6.0 (durable_v11 + Twin Double-MPO + Multi-Step KL)."""

from __future__ import annotations

import copy
import importlib.util
import json
import os
from pathlib import Path
from typing import Any, cast

import pytest

torch = pytest.importorskip("torch")

from sts2_baseline import act_frontloaded_death_reward_identity  # noqa: E402
from sts2_rl.encoding import GroundedObservationEncoder  # noqa: E402
from sts2_rl.encoding.snapshot import collate_encoded_snapshots, collate_encoded_worlds  # noqa: E402
from sts2_rl.joint.afterstate import synthesize_deterministic_afterstates  # noqa: E402
from sts2_rl.joint.macro_checkpoint import (  # noqa: E402
    MacroCheckpointIdentity,
    publish_macro_checkpoint,
    save_macro_checkpoint,
)
from sts2_rl.joint.macro_credit_data import (  # noqa: E402
    MacroAfterstateNode,
    MacroCreditCohort,
    MacroCreditTrajectory,
    MacroRewardFacts,
    macro_mixture_log_probs,
)
from sts2_rl.joint.macro_credit_learner import (  # noqa: E402
    MACRO_CREDIT_LEARNER_VERSION,
    MacroCreditLearner,
    MacroCreditLearnerConfig,
)
from sts2_rl.joint.macro_targets import project_hierarchical_macro_policy  # noqa: E402
from sts2_rl.joint.transitions import HybridVTraceStep  # noqa: E402
from sts2_rl.macro_root_policy import FrozenMacroRootPolicy, load_frozen_macro_teacher  # noqa: E402
from sts2_rl.models.hybrid_actor_critic import (  # noqa: E402
    HybridActorCriticConfig,
    HybridDomainActorCriticModel,
)
from sts2_rl.semantics.forward import forward_decision  # noqa: E402
from sts2_rl.train_macro_credit import _actor_critic_config, load_macro_training_config  # noqa: E402
from sts2_rl.training.config import ModelConfig, load_training_config  # noqa: E402


def _tiny_model_config() -> ModelConfig:
    return ModelConfig(
        d_model=32,
        n_heads=4,
        ffn_dim=64,
        world_layers=1,
        latent_slots=4,
        latent_layers=1,
        recurrent_hidden_dim=64,
        dropout=0.0,
        max_world_tokens=256,
        max_candidates=32,
        max_candidate_local_tokens=16,
    )


def _deck_alpha() -> list[dict[str, Any]]:
    return [
        {"id": f"strike_{i}", "type": "Attack", "rarity": "Basic", "cost": 1, "dynamic_vars": {"Damage": 6}}
        for i in range(5)
    ] + [
        {"id": "reaper_heal", "type": "Attack", "rarity": "Rare", "cost": 2, "dynamic_vars": {"Damage": 10}},
    ]


def _deck_beta() -> list[dict[str, Any]]:
    return [
        {"id": f"shrug_{i}", "type": "Skill", "rarity": "Common", "cost": 1, "dynamic_vars": {"Block": 11}}
        for i in range(6)
    ] + [
        {"id": "barricade_def", "type": "Power", "rarity": "Rare", "cost": 3, "dynamic_vars": {"Block": 20}},
        {"id": "body_slam", "type": "Attack", "rarity": "Common", "cost": 0, "dynamic_vars": {"Damage": 15}},
    ]


def _encode_world_state(
    encoder: GroundedObservationEncoder,
    *,
    deck: list[dict[str, Any]],
    hp: int,
    gold: int,
    shop: dict[str, Any] | None = None,
    legal_actions: list[dict[str, Any]] | None = None,
) -> Any:
    obs: dict[str, Any] = {
        "phase": "shop" if shop is not None else "map",
        "decision_domain": "shop" if shop is not None else "map",
        "run": {"active": True, "act": 1, "floor": 6},
        "player": {
            "character_id": "IRONCLAD",
            "hp": hp,
            "current_hp": hp,
            "max_hp": 80,
            "gold": gold,
            "open_potion_slots": 2,
            "deck": copy.deepcopy(deck),
            "relics": [{"id": "BURNING_BLOOD"}],
            "potions": [],
        },
    }
    if shop is not None:
        obs["shop"] = copy.deepcopy(shop)
    else:
        obs["map"] = {"nodes": [{"x": 0, "y": 1, "type": "Monster"}]}
    actions = legal_actions or [
        {"action_id": "noop", "kind": "proceed", "model_action_kind": "map", "is_enabled": True}
    ]
    return encoder.encode(obs, actions).snapshot


def test_durable_v11_restores_build_vitals_cross_derivative_and_preserves_monotonicity() -> None:
    torch.manual_seed(11)
    cfg = _tiny_model_config()
    encoder = GroundedObservationEncoder(cfg.to_encoding_config())

    model_v10 = HybridDomainActorCriticModel(HybridActorCriticConfig.durable_v10(cfg.to_model_config()))
    model_v11 = HybridDomainActorCriticModel(HybridActorCriticConfig.durable_v11(cfg.to_model_config()))
    with torch.no_grad():
        model_v10.durable_deck_out.weight.normal_(0.0, 0.1)
        model_v11.durable_deck_out.weight.normal_(0.0, 0.1)
        cast(Any, model_v11.durable_hp_build_gate[-1]).weight.normal_(0.0, 0.25)
        cast(Any, model_v11.durable_liquid_build_gate[-1]).weight.normal_(0.0, 0.25)

    worlds = tuple(
        _encode_world_state(encoder, deck=d, hp=hp, gold=gold)
        for d in (_deck_alpha(), _deck_beta())
        for hp, gold in ((16, 50), (68, 50), (68, 150))
    )
    wb, dom = collate_encoded_worlds(
        worlds,
        expected_config=worlds[0].config,
        expected_fingerprint=worlds[0].encoding_fingerprint,
        device=torch.device("cpu"),
    )
    with torch.no_grad():
        v10_vals, _ = model_v10.forward_durable_value(wb, dom, None)
        v11_h1, v11_h2, _ = model_v11.forward_durable_value_pair(wb, dom, None)

    # Confirm v10 additive separation vs v11 non-zero cross-derivative (Build x HP and Build x Gold).
    v10_dhp_a = float((v10_vals[1] - v10_vals[0]).item())
    v10_dhp_b = float((v10_vals[4] - v10_vals[3]).item())
    assert abs(v10_dhp_a - v10_dhp_b) < 1e-6

    v11_dhp_a = float((v11_h1[1] - v11_h1[0]).item())
    v11_dhp_b = float((v11_h1[4] - v11_h1[3]).item())
    v11_dgold_a = float((v11_h1[2] - v11_h1[1]).item())
    v11_dgold_b = float((v11_h1[5] - v11_h1[4]).item())
    assert abs(v11_dhp_a - v11_dhp_b) > 1e-4
    assert abs(v11_dgold_a - v11_dgold_b) > 1e-4
    assert torch.isfinite(v11_h2).all()

    # Strict monotonicity across HP and Gold grids for both decks.
    for deck in (_deck_alpha(), _deck_beta()):
        hp_worlds = tuple(_encode_world_state(encoder, deck=deck, hp=h, gold=100) for h in (5, 16, 40, 68, 80))
        hwb, hdom = collate_encoded_worlds(
            hp_worlds,
            expected_config=hp_worlds[0].config,
            expected_fingerprint=hp_worlds[0].encoding_fingerprint,
            device=torch.device("cpu"),
        )
        gold_worlds = tuple(_encode_world_state(encoder, deck=deck, hp=50, gold=g) for g in (0, 50, 150, 300))
        gwb, gdom = collate_encoded_worlds(
            gold_worlds,
            expected_config=gold_worlds[0].config,
            expected_fingerprint=gold_worlds[0].encoding_fingerprint,
            device=torch.device("cpu"),
        )
        with torch.no_grad():
            hv, _ = model_v11.forward_durable_value(hwb, hdom, None)
            gv, _ = model_v11.forward_durable_value(gwb, gdom, None)
        assert bool((hv[1:] >= hv[:-1] - 1e-6).all().item())
        assert bool((gv[1:] >= gv[:-1] - 1e-6).all().item())


def test_durable_v11_preserves_shop_continuation_without_candidate_leakage() -> None:
    torch.manual_seed(19)
    cfg = _tiny_model_config()
    encoder = GroundedObservationEncoder(cfg.to_encoding_config())
    model_v11 = HybridDomainActorCriticModel(HybridActorCriticConfig.durable_v11(cfg.to_model_config()))
    with torch.no_grad():
        cast(Any, model_v11.durable_shop_base_head[-1]).weight.normal_(0.0, 0.25)
        cast(Any, model_v11.durable_shop_liquid_gate[-1]).weight.normal_(0.0, 0.25)

    shelf_items: list[dict[str, Any]] = [
        {"index": 0, "slot_index": 0, "category": "relic", "type": "relic", "cost": 150, "relic": {"id": "ANCHOR", "has_upon_pickup_effect": False}},
        {"index": 1, "slot_index": 1, "category": "card", "type": "card", "cost": 70, "card": {"id": "OFFERING", "type": "Skill", "rarity": "Rare", "cost": 0}},
    ]
    shelf_rich: dict[str, Any] = {
        "visible": True,
        "is_open": True,
        "gold": 180,
        "items": shelf_items,
    }
    shelf_empty = {
        "visible": True,
        "is_open": True,
        "gold": 180,
        "items": [
            {"index": 0, "slot_index": 0, "category": "card", "type": "card", "cost": 175, "card": {"id": "STRIKE", "type": "Attack", "rarity": "Basic", "cost": 1}},
        ],
    }
    w_in_rich = _encode_world_state(encoder, deck=_deck_alpha(), hp=60, gold=180, shop=shelf_rich)
    w_in_poor = _encode_world_state(encoder, deck=_deck_alpha(), hp=60, gold=180, shop=shelf_empty)
    w_exited = _encode_world_state(encoder, deck=_deck_alpha(), hp=60, gold=180, shop=None)

    wb, dom = collate_encoded_worlds(
        (w_in_rich, w_in_poor, w_exited),
        expected_config=w_in_rich.config,
        expected_fingerprint=w_in_rich.encoding_fingerprint,
        device=torch.device("cpu"),
    )
    with torch.no_grad():
        vals, _ = model_v11.forward_durable_value(wb, dom, None)
    assert abs(float((vals[0] - vals[2]).item())) > 1e-4
    assert abs(float((vals[0] - vals[1]).item())) > 1e-4

    # Candidate firewall check: changing legal candidate actions (zone 24) must have zero effect on durable value.
    snap_a = _encode_world_state(
        encoder,
        deck=_deck_alpha(),
        hp=60,
        gold=180,
        shop=shelf_rich,
        legal_actions=[
            {"action_id": "leave", "kind": "shop_leave", "model_action_kind": "shop", "is_enabled": True},
        ],
    )
    snap_b = _encode_world_state(
        encoder,
        deck=_deck_alpha(),
        hp=60,
        gold=180,
        shop=shelf_rich,
        legal_actions=[
            {"action_id": "buy0", "kind": "shop_purchase", "model_action_kind": "shop", "is_enabled": True, "item": shelf_items[0]},
            {"action_id": "leave", "kind": "shop_leave", "model_action_kind": "shop", "is_enabled": True},
        ],
    )
    b_snaps = collate_encoded_snapshots(
        (snap_a, snap_b),
        expected_config=snap_a.config,
        expected_fingerprint=snap_a.encoding_fingerprint,
        device=torch.device("cpu"),
    )
    with torch.no_grad():
        dv1, dv2, _ = model_v11.forward_durable_value_pair(b_snaps.world, b_snaps.domain_ids, None)
    assert float((dv1[0] - dv1[1]).abs().item()) < 1e-6
    assert float((dv2[0] - dv2[1]).abs().item()) < 1e-6


def test_durable_v10_to_v11_warmstart_and_teacher_roundtrip(tmp_path: Path, monkeypatch: Any) -> None:
    monkeypatch.setenv("STS2_ARTIFACT_ROOT", str(tmp_path))
    torch.manual_seed(29)
    cfg = _tiny_model_config()
    train_cfg = load_training_config(profile="preheat")
    from dataclasses import replace

    train_cfg = replace(
        train_cfg,
        model=cfg,
        curriculum=replace(train_cfg.curriculum, revival_budget=16),
    )
    ac_v10 = HybridActorCriticConfig.durable_v10(cfg.to_model_config())
    src_model = HybridDomainActorCriticModel(ac_v10)
    with torch.no_grad():
        cast(Any, src_model.durable_value_head[-1]).weight.normal_(0.0, 0.05)
        src_model.durable_deck_out.weight.normal_(0.0, 0.05)
    opt = torch.optim.AdamW(src_model.parameters(), lr=1e-4)
    identity = MacroCheckpointIdentity(
        course_config={"version": "test"},
        model_config=ac_v10.as_mapping(),
        reward_identity=act_frontloaded_death_reward_identity(revival_budget=32),
        lineage_id="v10-parent",
        learner_version=MACRO_CREDIT_LEARNER_VERSION,
    )
    ckpt_path = tmp_path / "src" / "checkpoint.pt"
    pub_path = tmp_path / "src" / "publication.pt"
    save_macro_checkpoint(
        ckpt_path,
        identity=identity,
        model=src_model,
        optimizer=opt,
        update=125,
        next_seed_index=6000,
        initialization={"mode": "test"},
        artifact_root=tmp_path,
    )
    publish_macro_checkpoint(ckpt_path, pub_path, identity=identity, artifact_root=tmp_path)

    # 1. Teacher load round-trip on v10 publication must still pass exact as_mapping() equality.
    teacher = load_frozen_macro_teacher(
        pub_path,
        expected_encoding=cfg.to_encoding_config(),
        device=torch.device("cpu"),
        rocm_sdpa_backend="math",
    )
    assert not teacher.model.actor_critic_config.enable_build_conditioned_durable_value

    # 2. Student warm-start into durable_v11 preserves Head 1 on non-shop states and breaks symmetry on Head 2.
    ac_v11 = HybridActorCriticConfig.durable_v11(cfg.to_model_config())
    student_policy = FrozenMacroRootPolicy(
        pub_path,
        train_cfg,
        "cpu",
        initialize_from_macro_publication=True,
        actor_critic_config=ac_v11,
    )
    student = student_policy.model
    student.eval()
    encoder = GroundedObservationEncoder(cfg.to_encoding_config())
    w_map = _encode_world_state(encoder, deck=_deck_alpha(), hp=52, gold=120, shop=None)
    wb, dom = collate_encoded_worlds(
        (w_map,),
        expected_config=w_map.config,
        expected_fingerprint=w_map.encoding_fingerprint,
        device=torch.device("cpu"),
    )
    with torch.no_grad():
        v_src, _ = src_model.forward_durable_value(wb, dom, None)
        s_h1, s_h2, _ = student.forward_durable_value_pair(wb, dom, None)
    assert float((s_h1 - v_src).abs().max().item()) < 1e-6
    assert float((s_h1 - s_h2).abs().max().item()) > 1e-7


def test_double_mpo_eliminates_multi_card_winners_curse_bias() -> None:
    # 3 take actions with true Q = 0.0 (branch 0) vs 1 skip action with true Q = +0.02 (branch 1).
    true_q = torch.tensor([0.0, 0.0, 0.0, 0.02], dtype=torch.float32)
    prior_probs = torch.tensor([0.5 / 3, 0.5 / 3, 0.5 / 3, 0.5], dtype=torch.float32)
    teacher_logs = prior_probs.log()
    branch_ids = torch.tensor([0, 0, 0, 1], dtype=torch.long)

    noiseless_proj = project_hierarchical_macro_policy(
        det_action_values=true_q,
        teacher_log_probs=teacher_logs,
        policy_branch_ids=branch_ids,
        branch_max_kl=0.04,
        within_max_kl=0.05,
        min_temperature=0.02,
    )
    noiseless_skip = float(noiseless_proj.target_probabilities[3].item())
    assert noiseless_skip > 0.60

    gen = torch.Generator(device="cpu").manual_seed(20261117)
    single_skip_probs: list[float] = []
    double_skip_probs: list[float] = []
    for _ in range(200):
        eps1 = torch.randn((4,), generator=gen) * 0.08
        eps2 = torch.randn((4,), generator=gen) * 0.08
        # Unchanged deck on skip has much lower epistemic noise than speculative card picks.
        eps1[3] *= 0.2
        eps2[3] *= 0.2
        v1 = true_q + eps1
        v2 = true_q + eps2

        proj_single = project_hierarchical_macro_policy(
            det_action_values=v1,
            teacher_log_probs=teacher_logs,
            policy_branch_ids=branch_ids,
            branch_max_kl=0.04,
            within_max_kl=0.05,
            min_temperature=0.02,
        )
        proj_double = project_hierarchical_macro_policy(
            det_action_values=v1,
            det_action_values_twin=v2,
            pessimism_coef=0.25,
            teacher_log_probs=teacher_logs,
            policy_branch_ids=branch_ids,
            branch_max_kl=0.04,
            within_max_kl=0.05,
            min_temperature=0.02,
        )
        single_skip_probs.append(float(proj_single.target_probabilities[3].item()))
        double_skip_probs.append(float(proj_double.target_probabilities[3].item()))

    mean_single_skip = sum(single_skip_probs) / len(single_skip_probs)
    mean_double_skip = sum(double_skip_probs) / len(double_skip_probs)
    # Single critic loses >0.10 skip probability vs noiseless oracle due to 3-card Winner's Curse,
    # whereas Twin Double-MPO recovers high skip discipline!
    assert noiseless_skip - mean_single_skip > 0.10
    assert mean_double_skip > 0.55
    assert mean_double_skip - mean_single_skip > 0.08


def _build_synthetic_durable_cohort(
    model: HybridDomainActorCriticModel,
    encoder: GroundedObservationEncoder,
    *,
    num_games: int = 4,
) -> MacroCreditCohort:
    trajs: list[MacroCreditTrajectory] = []
    for g in range(num_games):
        reward_obs = {
            "phase": "card_reward",
            "decision_domain": "card_reward",
            "run": {"active": True, "act": 1, "floor": 2 + g},
            "player": {
                "character_id": "IRONCLAD",
                "hp": 45 + 4 * g,
                "current_hp": 45 + 4 * g,
                "max_hp": 80,
                "gold": 90 + 15 * g,
                "deck": _deck_alpha(),
                "relics": [],
                "potions": [],
            },
            "card_reward_selection": {
                "can_skip": True,
                "cards": [
                    {"id": f"pommel_{g}", "type": "Attack", "cost": 1},
                    {"id": f"shrug_{g}", "type": "Skill", "cost": 1},
                ],
            },
        }
        reward_legal = [
            {"action_id": "r0", "kind": "select_card_reward", "model_action_kind": "card_reward", "is_enabled": True, "card": {"id": f"pommel_{g}", "type": "Attack", "cost": 1}},
            {"action_id": "r1", "kind": "select_card_reward", "model_action_kind": "card_reward", "is_enabled": True, "card": {"id": f"shrug_{g}", "type": "Skill", "cost": 1}},
            {"action_id": "skip", "kind": "skip_card_reward", "model_action_kind": "card_reward", "is_enabled": True},
        ]
        dec = forward_decision(reward_obs, reward_legal, control_domain="macro")
        assert dec is not None
        enc = encoder.encode(reward_obs, [c.semantic_action for c in dec.candidates])
        grouped_cands = tuple(dec.candidates[ref.representative_position] for ref in enc.actions)
        post_worlds = synthesize_deterministic_afterstates(
            encoder=encoder, observation=reward_obs, decision=dec, candidates=grouped_cands
        )
        assert post_worlds is not None

        combat_obs = {
            "phase": "combat",
            "decision_domain": "combat",
            "run": {"active": True, "act": 1, "floor": 3 + g},
            "player": {
                "id": "ironclad",
                "side": "player",
                "hp": 44 + 4 * g,
                "max_hp": 80,
                "energy": 3,
                "hand": [{"id": "strike", "cost": 1}, {"id": "defend", "cost": 1}],
            },
            "combat": {"in_progress": True, "enemies": [{"id": "cultist", "side": "enemy", "hp": 35, "max_hp": 48}]},
        }
        combat_legal = [
            {"action_handle": "c0", "kind": "play_card", "model_action_kind": "play_card", "is_enabled": True, "card": {"id": "strike", "cost": 1}},
            {"action_handle": "c1", "kind": "play_card", "model_action_kind": "play_card", "is_enabled": True, "card": {"id": "defend", "cost": 1}},
        ]
        combat_snap = encoder.encode(combat_obs, combat_legal).snapshot

        with torch.no_grad():
            b0 = collate_encoded_snapshots(
                (enc.snapshot,),
                expected_config=enc.snapshot.config,
                expected_fingerprint=enc.snapshot.encoding_fingerprint,
                device=torch.device("cpu"),
            )
            out0 = model(b0, None, validate=False)
            mask0 = torch.tensor(enc.snapshot.action_mask, dtype=torch.bool)
            lp0 = torch.log_softmax(out0.policy_logits[0, : mask0.numel()].masked_fill(~mask0, -torch.inf), dim=0)
            beh0 = macro_mixture_log_probs(lp0, enc.snapshot, 0.04)

            b1 = collate_encoded_snapshots(
                (combat_snap,),
                expected_config=combat_snap.config,
                expected_fingerprint=combat_snap.encoding_fingerprint,
                device=torch.device("cpu"),
            )
            out1 = model(b1, out0.recurrent_state, validate=False)
            mask1 = torch.tensor(combat_snap.action_mask, dtype=torch.bool)
            lp1 = torch.log_softmax(out1.policy_logits[0, : mask1.numel()].masked_fill(~mask1, -torch.inf), dim=0)

        en0 = [idx for idx, flag in enumerate(enc.snapshot.action_mask) if flag]
        en1 = [idx for idx, flag in enumerate(combat_snap.action_mask) if flag]
        a0 = en0[g % len(en0)]
        a1 = en1[g % len(en1)]
        b_name = grouped_cands[a0].branch
        from sts2_rl.semantics.clock import decision_discount

        disc0 = decision_discount(floor_before=2 + g, floor_after=3 + g, terminal=False)
        r_term = 0.5 if g % 2 == 0 else -0.3
        s0 = HybridVTraceStep(enc.snapshot, a0, float(beh0[a0].item()), 0.1, disc0, False, "reward", b_name, "macro", 0)
        s1 = HybridVTraceStep(combat_snap, a1, float(lp1[a1].item()), r_term, 0.0, True, "combat", "play_card", "combat", 0)
        f0 = MacroRewardFacts(0.0, 0.1, 0.0, 0.0, 0.0, 0, 0.0, 2 + g, 3 + g, 1)
        f1 = MacroRewardFacts(r_term, 0.0, 0.0, 0.0, 0.0, 0, 0.0, 3 + g, 3 + g, 1)
        node0 = MacroAfterstateNode(0, "reward", tuple(c.branch for c in grouped_cands), post_worlds)
        trajs.append(
            MacroCreditTrajectory(
                f"traj-{g}",
                2 * (6_700_000 + g),
                0,
                (),
                (s0, s1),
                None,
                None,
                "factual_terminal",
                (f0, f1),
                afterstate_nodes=(node0,),
                macro_branch_epsilon=0.04,
            )
        )
    return MacroCreditCohort("cohort-v60", 0, (), natural_only=tuple(trajs))


def test_multi_minibatch_twin_learner_executes_trust_region_and_kl_early_stop() -> None:
    torch.manual_seed(41)
    cfg = _tiny_model_config()
    encoder = GroundedObservationEncoder(cfg.to_encoding_config())
    ac_cfg = HybridActorCriticConfig.durable_v11(cfg.to_model_config())
    model_a = HybridDomainActorCriticModel(ac_cfg)
    with torch.no_grad():
        cast(Any, model_a.durable_value_head[-1]).weight.normal_(0.0, 0.15)
        cast(Any, model_a.durable_twin_value_head[-1]).weight.normal_(0.0, 0.15)
    model_b = copy.deepcopy(model_a)
    cohort = _build_synthetic_durable_cohort(model_a, encoder, num_games=4)

    base_kwargs: dict[str, Any] = {
        "target_mode": "durable_afterstate_v2",
        "domain_layers": 2,
        "joint_layers": 3,
        "enable_direct_vitals_context": True,
        "enable_macro_branch_prior": True,
        "enable_persistent_only_durable_value": True,
        "enable_asymmetric_revival_critic": True,
        "enable_draw_weighted_durable_value": True,
        "enable_role_conditioned_candidate_policy": True,
        "enable_deconfounded_durable_value": True,
        "enable_monotone_deconfounded_durable_value": True,
        "enable_balanced_deconfounded_durable_value": True,
        "enable_expressive_synergy_backbone": True,
        "enable_debloated_synergy_durable_value": True,
        "enable_build_conditioned_durable_value": True,
        "enable_twin_durable_value_heads": True,
        "double_mpo_pessimism_coef": 0.25,
        "enable_hierarchical_afterstate_mpo": True,
        "revival_budget": 16,
        "durable_warmup_updates": 0,
        "combat_actor_warmup_updates": 0,
        "clip_combat_pg": True,
        "clip_macro_pg_fallback": True,
        "macro_mpo_trust_kl_weight": 0.06,
        "learning_rate": 5e-4,
        "ppo_epochs_per_cohort": 2,
        "ppo_minibatches_per_cohort": 2,
    }

    learner_full = MacroCreditLearner(
        model=model_a,
        config=MacroCreditLearnerConfig(**base_kwargs, target_macro_step_kl=0.50),
    )
    metrics_full = learner_full.learn(cohort)
    assert metrics_full["ppo_optimizer_steps_executed"] == 4
    assert metrics_full["ppo_kl_early_stopped"] is False
    assert metrics_full["mean_updated_macro_policy_kl"] > 0.0
    assert metrics_full["mean_updated_policy_kl_to_teacher"] == metrics_full["mean_updated_macro_policy_kl"]
    assert metrics_full["mean_afterstate_twin_disagreement"] > 0.0

    # Tight target_macro_step_kl must allow step 0 and stop early on subsequent minibatches.
    learner_tight = MacroCreditLearner(
        model=model_b,
        config=MacroCreditLearnerConfig(**base_kwargs, target_macro_step_kl=1e-9),
    )
    metrics_tight = learner_tight.learn(cohort)
    assert 1 <= metrics_tight["ppo_optimizer_steps_executed"] < 4
    assert metrics_tight["ppo_kl_early_stopped"] is True


def test_v60_experiment_config_and_campaign_heldout_gate_sync(tmp_path: Path, monkeypatch: Any) -> None:
    monkeypatch.setenv("STS2_ARTIFACT_ROOT", str(tmp_path))
    v60_toml = (
        Path(__file__).resolve().parents[1]
        / "config/experiments/hybrid_macro_credit_v16_durable_afterstate_v60_wide256_b16_g4_turbo.toml"
    )
    cfg = load_macro_training_config(v60_toml)
    assert cfg.learner.revival_budget == 16
    assert cfg.training.curriculum.revival_budget == 16
    assert cfg.course.seed_index_offset == 300000
    assert cfg.learner.enable_build_conditioned_durable_value is True
    assert cfg.learner.enable_twin_durable_value_heads is True
    assert cfg.learner.ppo_epochs_per_cohort == 2
    assert cfg.learner.ppo_minibatches_per_cohort == 4
    ac_cfg = _actor_critic_config(cfg)
    assert ac_cfg.enable_build_conditioned_durable_value is True
    assert ac_cfg.enable_twin_durable_value_heads is True

    script_path = Path(__file__).resolve().parents[1] / "scripts/run_colab_g4_durable_campaign.py"
    spec = importlib.util.spec_from_file_location("run_colab_g4_durable_campaign", script_path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    out_dir = tmp_path / "exp"
    drive_dir = tmp_path / "drive"
    out_dir.mkdir()
    (out_dir / "checkpoint.pt").write_bytes(b"ckpt")
    (out_dir / "publication.pt").write_bytes(b"pub")
    (out_dir / "status.json").write_text('{"committed_update": 1, "next_seed_index": 48}\n', encoding="utf-8")

    monkeypatch.setattr(
        module,
        "_run_competence_probes",
        lambda **_: {
            "campfire_hp_sensitivity": {"p_rest_low_minus_high": 0.2, "durable_value_rest_advantage_shift": 0.03},
            "smith_vs_remove_divergence": {"mean_smith_remove_jsd": 0.1, "mean_smith_remove_spearman": -0.2},
            "multiset_split_invariance": {"grouped_vs_split_abs_diff": 0.0, "smith_high_delta_minus_basic_adv": 0.01},
            "episode_macro_discipline": {"episodes": 48},
        },
    )
    monkeypatch.setattr(
        module,
        "_run_heldout_evaluation",
        lambda **_: {
            "evaluation_gate": 1,
            "policy_version": 1,
            "data_partition": "validation",
            "seed_parity": "odd_held_out",
            "revival_budget": 16,
            "episodes": 48,
            "successful_run_count": 11,
            "run_win_rate": 11 / 48,
            "act1_clear_rate": 38 / 48,
            "act3_reach_rate": 21 / 48,
            "skip_rate": 0.388,
            "mean_final_deck_size": 25.3,
        },
    )
    manifest = module._probe_prune_and_sync(
        output_dir=out_dir,
        drive_dir=drive_dir,
        lineage_id="durable-afterstate-v6-0-wide256-b16-g4",
        committed_after=1,
        train_summary={"ok": True},
        device="cpu",
        env=dict(os.environ),
        snapshot_interval=5,
        keep_recent_cohorts=2,
        config_path=v60_toml,
        heldout_eval_interval=10,
        heldout_eval_episodes=48,
        heldout_eval_workers=8,
    )
    assert manifest["latest_heldout"] is not None
    assert (drive_dir / "heldout_evaluations.jsonl").is_file()
    assert (drive_dir / "snapshots" / "update-000001" / "heldout.json").is_file()
    saved_line = json.loads((drive_dir / "heldout_evaluations.jsonl").read_text(encoding="utf-8").strip())
    assert saved_line["revival_budget"] == 16
    assert saved_line["seed_parity"] == "odd_held_out"


def test_v62_frozen_trunk_warmup_preserves_bit_exact_policy_while_calibrating_value() -> None:
    torch.manual_seed(73)
    cfg = _tiny_model_config()
    encoder = GroundedObservationEncoder(cfg.to_encoding_config())
    ac_cfg = HybridActorCriticConfig.durable_v11(cfg.to_model_config())
    model = HybridDomainActorCriticModel(ac_cfg)
    model.eval()
    with torch.no_grad():
        cast(Any, model.durable_value_head[-1]).weight.normal_(0.0, 0.05)
        cast(Any, model.durable_twin_value_head[-1]).weight.normal_(0.0, 0.05)
    cohort = _build_synthetic_durable_cohort(model, encoder, num_games=4)

    snap0 = cohort.natural_only[0].steps[0].snapshot
    snap1 = cohort.natural_only[0].steps[1].snapshot
    b0 = collate_encoded_snapshots(
        (snap0,),
        expected_config=snap0.config,
        expected_fingerprint=snap0.encoding_fingerprint,
        device=torch.device("cpu"),
    )
    b1 = collate_encoded_snapshots(
        (snap1,),
        expected_config=snap1.config,
        expected_fingerprint=snap1.encoding_fingerprint,
        device=torch.device("cpu"),
    )
    with torch.no_grad():
        pre_out0 = model(b0, None, validate=False)
        pre_out1 = model(b1, pre_out0.recurrent_state, validate=False)
    pre_non_val_params = {
        name: p.detach().clone()
        for name, p in model.named_parameters()
        if not name.startswith(("state_value_head.", "state_revival_", "durable_"))
    }

    learner = MacroCreditLearner(
        model=model,
        config=MacroCreditLearnerConfig(
            target_mode="durable_afterstate_v2",
            domain_layers=2,
            joint_layers=3,
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
            double_mpo_pessimism_coef=0.35,
            enable_hierarchical_afterstate_mpo=True,
            revival_budget=32,
            freeze_trunk_during_durable_warmup=True,
            durable_warmup_updates=5,
            combat_actor_warmup_updates=5,
            learning_rate=1e-3,
            weight_decay=0.05,
        ),
    )
    metrics = learner.learn(cohort)
    assert metrics["warmup_trunk_frozen"] is True
    assert metrics["durable_actor_active"] is False
    assert metrics["combat_actor_active"] is False
    assert metrics["mean_updated_macro_policy_kl"] == 0.0
    assert metrics["mean_updated_combat_policy_kl"] == 0.0

    # Every non-pure-value parameter (including AdamW weight decay!) must be bit-identical.
    for name, p in model.named_parameters():
        assert p.requires_grad is True
        if name in pre_non_val_params:
            assert float((p.detach() - pre_non_val_params[name]).abs().max().item()) == 0.0

    with torch.no_grad():
        post_out0 = model(b0, None, validate=False)
        post_out1 = model(b1, post_out0.recurrent_state, validate=False)
    assert float((post_out0.policy_logits - pre_out0.policy_logits).abs().max().item()) == 0.0
    assert float((post_out0.recurrent_state - pre_out0.recurrent_state).abs().max().item()) == 0.0
    assert float((post_out1.policy_logits - pre_out1.policy_logits).abs().max().item()) == 0.0
    assert float((post_out1.recurrent_state - pre_out1.recurrent_state).abs().max().item()) == 0.0
    # Value heads MUST have updated during warmup!
    assert post_out0.durable_value is not None and pre_out0.durable_value is not None
    assert float((post_out0.durable_value - pre_out0.durable_value).abs().max().item()) > 1e-6


def test_v62_post_step_kl_backtracking_line_search_caps_macro_kl_without_dampening_critic() -> None:
    torch.manual_seed(97)
    cfg = _tiny_model_config()
    encoder = GroundedObservationEncoder(cfg.to_encoding_config())
    ac_cfg = HybridActorCriticConfig.durable_v11(cfg.to_model_config())
    model_raw = HybridDomainActorCriticModel(ac_cfg)
    with torch.no_grad():
        cast(Any, model_raw.durable_value_head[-1]).weight.normal_(0.0, 0.25)
        cast(Any, model_raw.durable_twin_value_head[-1]).weight.normal_(0.0, 0.25)
    model_ls = copy.deepcopy(model_raw)
    cohort = _build_synthetic_durable_cohort(model_raw, encoder, num_games=4)

    common_kwargs: dict[str, Any] = {
        "target_mode": "durable_afterstate_v2",
        "domain_layers": 2,
        "joint_layers": 3,
        "enable_direct_vitals_context": True,
        "enable_macro_branch_prior": True,
        "enable_persistent_only_durable_value": True,
        "enable_asymmetric_revival_critic": True,
        "enable_draw_weighted_durable_value": True,
        "enable_role_conditioned_candidate_policy": True,
        "enable_deconfounded_durable_value": True,
        "enable_monotone_deconfounded_durable_value": True,
        "enable_balanced_deconfounded_durable_value": True,
        "enable_expressive_synergy_backbone": True,
        "enable_debloated_synergy_durable_value": True,
        "enable_build_conditioned_durable_value": True,
        "enable_twin_durable_value_heads": True,
        "double_mpo_pessimism_coef": 0.35,
        "enable_hierarchical_afterstate_mpo": True,
        "revival_budget": 32,
        "durable_warmup_updates": 0,
        "combat_actor_warmup_updates": 0,
        "ppo_epochs_per_cohort": 1,
        "ppo_minibatches_per_cohort": 1,
        "learning_rate": 3e-3,
    }
    torch.manual_seed(1234)
    m_raw = MacroCreditLearner(
        model=model_raw,
        config=MacroCreditLearnerConfig(**common_kwargs, enable_post_step_kl_line_search=False),
    ).learn(cohort)
    target_kl = max(1e-5, float(m_raw["mean_updated_macro_policy_kl"]) * 0.25)
    torch.manual_seed(1234)
    m_ls = MacroCreditLearner(
        model=model_ls,
        config=MacroCreditLearnerConfig(
            **common_kwargs,
            enable_post_step_kl_line_search=True,
            target_macro_step_kl=target_kl,
            line_search_max_backtracks=4,
            line_search_min_scale=0.05,
        ),
    ).learn(cohort)
    assert m_ls["line_search_backtracks"] >= 1
    assert m_ls["line_search_scale"] < 1.0
    assert m_ls["mean_updated_macro_policy_kl"] <= target_kl
    # Pure value heads remain unattenuated relative to the raw single-step update.
    assert float(
        (
            cast(Any, model_ls.durable_value_head[-1]).weight
            - cast(Any, model_raw.durable_value_head[-1]).weight
        )
        .abs()
        .max()
        .item()
    ) == 0.0


def test_v62_ablation_config_packed_scatter_and_dual_heldout_sync(tmp_path: Path, monkeypatch: Any) -> None:
    monkeypatch.setenv("STS2_ARTIFACT_ROOT", str(tmp_path))
    from sts2_rl.encoding.snapshot import pack_encoded_snapshots_to_device, scatter_packed_snapshots

    cfg_tiny = _tiny_model_config()
    encoder = GroundedObservationEncoder(cfg_tiny.to_encoding_config())
    snaps = tuple(
        _encode_world_state(encoder, deck=_deck_alpha(), hp=30 + 10 * i, gold=50 + 20 * i)
        for i in range(4)
    )
    direct = collate_encoded_snapshots(
        snaps,
        expected_config=snaps[0].config,
        expected_fingerprint=snaps[0].encoding_fingerprint,
        device=torch.device("cpu"),
        validate=False,
    )
    packed = pack_encoded_snapshots_to_device(snaps, cfg=snaps[0].config, dev=torch.device("cpu"))
    scattered = scatter_packed_snapshots(packed)
    assert torch.equal(direct.world.features, scattered.world.features)
    assert torch.equal(direct.candidates.features, scattered.candidates.features)
    assert torch.equal(direct.candidates.action_mask, scattered.candidates.action_mask)

    # Verify bfloat16 CPU autocast still returns exact float32 readout tensors.
    ac_v11 = HybridActorCriticConfig.durable_v11(cfg_tiny.to_model_config())
    model = HybridDomainActorCriticModel(ac_v11)
    with torch.no_grad(), torch.autocast(device_type="cpu", dtype=torch.bfloat16):
        out_bf16 = model(scattered, None, validate=True)
    assert out_bf16.policy_logits.dtype == torch.float32
    assert out_bf16.value.dtype == torch.float32
    assert out_bf16.recurrent_state.dtype == torch.float32
    assert out_bf16.durable_value is not None and out_bf16.durable_value.dtype == torch.float32

    v62_toml = (
        Path(__file__).resolve().parents[1]
        / "config/experiments/hybrid_macro_credit_v18_durable_afterstate_v62_ablation_b32_g4_turbo.toml"
    )
    cfg = load_macro_training_config(v62_toml)
    assert cfg.learner.revival_budget == 32
    assert cfg.training.curriculum.revival_budget == 32
    assert cfg.course.seed_index_offset == 500000
    assert cfg.learner.freeze_trunk_during_durable_warmup is True
    assert cfg.learner.durable_warmup_updates == 5
    assert cfg.learner.combat_actor_warmup_updates == 5
    assert cfg.learner.ppo_epochs_per_cohort == 1
    assert cfg.learner.ppo_minibatches_per_cohort == 1
    assert cfg.learner.enable_post_step_kl_line_search is True
    assert cfg.learner.target_macro_step_kl == 0.012
    assert cfg.learner.enable_bf16_amp is True

    script_path = Path(__file__).resolve().parents[1] / "scripts/run_colab_g4_durable_campaign.py"
    spec = importlib.util.spec_from_file_location("run_colab_g4_durable_campaign_v62", script_path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    out_dir = tmp_path / "exp62"
    drive_dir = tmp_path / "drive62"
    out_dir.mkdir()
    (out_dir / "checkpoint.pt").write_bytes(b"ckpt")
    (out_dir / "publication.pt").write_bytes(b"pub")
    (out_dir / "status.json").write_text('{"committed_update": 5, "next_seed_index": 240}\n', encoding="utf-8")

    monkeypatch.setattr(
        module,
        "_run_competence_probes",
        lambda **_: {
            "campfire_hp_sensitivity": {"p_rest_low_minus_high": 0.25, "durable_value_rest_advantage_shift": 0.04},
            "smith_vs_remove_divergence": {"mean_smith_remove_jsd": 0.12, "mean_smith_remove_spearman": -0.25},
            "multiset_split_invariance": {"grouped_vs_split_abs_diff": 0.0, "smith_high_delta_minus_basic_adv": 0.02},
            "episode_macro_discipline": {"episodes": 48},
        },
    )

    def _fake_eval(**kwargs: Any) -> dict[str, Any]:
        ov = kwargs.get("overrides") or ()
        is_b16 = any("revival_budget=16" in str(x) for x in ov)
        return {
            "evaluation_gate": 5,
            "policy_version": 5,
            "data_partition": "validation",
            "seed_parity": "odd_held_out",
            "revival_budget": 16 if is_b16 else 32,
            "episodes": 48,
            "successful_run_count": 11 if is_b16 else 30,
            "run_win_rate": (11 / 48) if is_b16 else (30 / 48),
            "act1_clear_rate": 1.0,
            "act3_reach_rate": 0.85,
            "skip_rate": 0.388,
            "mean_final_deck_size": 24.5,
        }

    monkeypatch.setattr(module, "_run_heldout_evaluation", _fake_eval)
    manifest = module._probe_prune_and_sync(
        output_dir=out_dir,
        drive_dir=drive_dir,
        lineage_id="durable-afterstate-v6-2-ablation-b32-g4",
        committed_after=5,
        train_summary={"ok": True},
        device="cpu",
        env=dict(os.environ),
        snapshot_interval=5,
        keep_recent_cohorts=2,
        config_path=v62_toml,
        heldout_eval_milestones={0, 5, 10, 20, 30, 40},
        extra_heldout_revival_budget=16,
        heldout_eval_episodes=48,
        heldout_eval_workers=8,
    )
    assert manifest["latest_heldout"]["revival_budget"] == 32
    assert manifest["latest_extra_heldout"]["revival_budget"] == 16
    assert (drive_dir / "heldout_evaluations.jsonl").is_file()
    assert (drive_dir / "heldout_evaluations_b16.jsonl").is_file()
    assert (drive_dir / "snapshots" / "update-000005" / "heldout.json").is_file()
    assert (drive_dir / "snapshots" / "update-000005" / "heldout_b16.json").is_file()


def test_v63_ratchet_promotion_gate_enforces_wins_revivals_and_antibloat(
    tmp_path: Path, monkeypatch: Any
) -> None:
    import json
    import subprocess
    import sys

    monkeypatch.setenv("STS2_ARTIFACT_ROOT", str(tmp_path))
    v63_toml = (
        Path(__file__).resolve().parents[1]
        / "config/experiments/hybrid_macro_credit_v19_durable_afterstate_v63_ratchet_g4_turbo.toml"
    )
    cfg = load_macro_training_config(v63_toml)
    assert cfg.learner.revival_budget == 24
    assert cfg.training.curriculum.revival_budget == 24
    assert cfg.learner.double_mpo_pessimism_coef == 0.42
    assert cfg.learner.kickstart_kl_floor == 0.048
    assert cfg.learner.freeze_trunk_during_durable_warmup is True
    assert cfg.learner.enable_post_step_kl_line_search is True
    assert cfg.learner.enable_bf16_amp is True

    script_path = Path(__file__).resolve().parents[1] / "scripts/run_colab_g4_ratchet_curriculum.py"
    spec = importlib.util.spec_from_file_location("run_colab_g4_ratchet_curriculum_v63", script_path)
    assert spec is not None and spec.loader is not None
    ratchet_mod = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = ratchet_mod
    spec.loader.exec_module(ratchet_mod)

    ladder = ratchet_mod.DEFAULT_SMOOTH_LADDER
    budgets = [st.budget for st in ladder]
    assert budgets == [24, 20, 16, 12, 8, 6, 4, 3, 2, 1]
    assert len({st.seed_index_offset for st in ladder}) == len(ladder)

    b24 = ladder[0]
    base_row = {
        "policy_version": 10,
        "episodes": 48,
        "revival_budget": 24,
        "successful_run_count": 28,
        "mean_revivals_used": 15.2,
        "skip_rate": 0.165,
        "mean_final_deck_size": 31.0,
        "deadlock_count": 0,
    }
    ok_good, _ = ratchet_mod.evaluate_stage_promotion_gate(
        base_row, b24, expected_episodes=48, min_policy_version=5
    )
    assert ok_good is True

    # Must NOT promote at u=0 before frozen-trunk value warmup finishes
    ok_u0, rep_u0 = ratchet_mod.evaluate_stage_promotion_gate(
        {**base_row, "policy_version": 0}, b24, expected_episodes=48, min_policy_version=5
    )
    assert ok_u0 is False and rep_u0["checks"]["policy_version_ready"] is False

    # Must NOT promote bloated 0%-skip / 40-card policy even with high wins
    ok_bloat, rep_bloat = ratchet_mod.evaluate_stage_promotion_gate(
        {**base_row, "successful_run_count": 34, "skip_rate": 0.0, "mean_final_deck_size": 40.35},
        b24,
        expected_episodes=48,
        min_policy_version=5,
    )
    assert ok_bloat is False
    assert rep_bloat["checks"]["skip_discipline_ok"] is False
    assert rep_bloat["checks"]["deck_size_ok"] is False

    # End-to-end 2-stage orchestration test (b24 -> b20)
    seed_pub = tmp_path / "seed_u20.pt"
    seed_pub.write_bytes(b"v62-u20-peak")
    drive_root = tmp_path / "drive_campaigns"

    def _fake_subproc_run(cmd: list[str], **_: Any) -> subprocess.CompletedProcess[str]:
        out_idx = cmd.index("--output-dir") + 1
        tgt_idx = cmd.index("--target-updates") + 1
        lin_idx = cmd.index("--lineage-id") + 1
        st_dir = Path(cmd[out_idx])
        tgt_u = int(cmd[tgt_idx])
        lin = cmd[lin_idx]
        bud = 24 if "-b24-" in lin else 20
        st_dir.mkdir(parents=True, exist_ok=True)
        (st_dir / "status.json").write_text(json.dumps({"committed_update": tgt_u}) + "\n", encoding="utf-8")
        (st_dir / "publication.pt").write_bytes(f"pub-{lin}-u{tgt_u}".encode("utf-8"))
        # b24 passes on second chunk (u=10); b20 passes on first chunk (u=5)
        wins = (22 if tgt_u == 5 else 29) if bud == 24 else 27
        rev = 14.5 if bud == 24 else 12.1
        row = {
            "policy_version": tgt_u,
            "episodes": 48,
            "revival_budget": bud,
            "successful_run_count": wins,
            "mean_revivals_used": rev,
            "skip_rate": 0.15,
            "mean_final_deck_size": 30.8,
            "deadlock_count": 0,
        }
        with (st_dir / "heldout_evaluations.jsonl").open("a", encoding="utf-8") as fh:
            fh.write(json.dumps(row) + "\n")
        return subprocess.CompletedProcess(cmd, 0, "", "")

    monkeypatch.setattr(ratchet_mod.subprocess, "run", _fake_subproc_run)
    rc = ratchet_mod.run_ratchet(
        [
            "--config",
            str(v63_toml),
            "--initial-publication",
            str(seed_pub),
            "--drive-root-dir",
            str(drive_root),
            "--budgets",
            "24,20",
            "--step-updates",
            "5",
            "--max-updates-per-stage",
            "40",
        ]
    )
    assert rc == 0
    state_json = json.loads((tmp_path / "active_ratchet_state.json").read_text(encoding="utf-8"))
    assert state_json["state"] == "ratchet_complete"
    assert [p["budget"] for p in state_json["promoted_history"]] == [24, 20]
    assert [p["promoted_at_update"] for p in state_json["promoted_history"]] == [10, 5]


def test_v64_unfreeze_trunk_detach_and_parent_prior_prevents_skip_collapse() -> None:
    import copy

    v64_toml = (
        Path(__file__).resolve().parents[1]
        / "config/experiments/hybrid_macro_credit_v20_durable_afterstate_v64_ratchet_g4_turbo.toml"
    )
    cfg64 = load_macro_training_config(v64_toml)
    assert cfg64.learner.detach_value_heads_from_trunk is True
    assert cfg64.learner.minority_only_within_branch_mpo is True
    assert cfg64.learner.within_branch_minority_prob_cap == 0.30
    assert cfg64.learner.macro_pg_fallback_scale == 0.20
    assert cfg64.learner.mpo_parent_prior_mix == 0.25
    assert cfg64.learner.max_parent_macro_kl == 0.030

    # 1. Minority-only within-branch MPO omits majority reward:take (P=0.60 > 0.30) while keeping minority P=0.20
    q_vals = torch.tensor([0.10, -0.05, 0.08], dtype=torch.float32)
    maj_logs = torch.tensor([0.30, 0.30, 0.40], dtype=torch.float32).log()
    min_logs = torch.tensor([0.10, 0.10, 0.80], dtype=torch.float32).log()
    b_ids = torch.tensor([0, 0, 1], dtype=torch.long)
    p_maj = project_hierarchical_macro_policy(
        det_action_values=q_vals,
        teacher_log_probs=maj_logs,
        policy_branch_ids=b_ids,
        branch_max_kl=0.015,
        within_max_kl=0.035,
        min_temperature=0.015,
        within_branch_prob_cap=0.30,
    )
    p_min = project_hierarchical_macro_policy(
        det_action_values=q_vals,
        teacher_log_probs=min_logs,
        policy_branch_ids=b_ids,
        branch_max_kl=0.015,
        within_max_kl=0.035,
        min_temperature=0.015,
        within_branch_prob_cap=0.30,
    )
    assert len(p_maj.within_branch_targets) == 0
    assert len(p_min.within_branch_targets) == 1

    # 2. When policy weights are zeroed (policy_weight=0, combat_policy_weight=0) at unfrozen update u=5,
    #    detach_value_heads_from_trunk=True ensures value losses produce ZERO gradient on macro_stream & world_encoder!
    torch.manual_seed(20261118)
    cfg = _tiny_model_config()
    ac_cfg = HybridActorCriticConfig.durable_v11(cfg.to_model_config())
    model = HybridDomainActorCriticModel(ac_cfg)
    teacher = copy.deepcopy(model)
    teacher.eval()
    encoder = GroundedObservationEncoder(cfg.to_encoding_config())
    cohort = _build_synthetic_durable_cohort(model, encoder, num_games=3)
    l_cfg = MacroCreditLearnerConfig(
        target_mode="durable_afterstate_v2",
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
        enable_hierarchical_afterstate_mpo=True,
        durable_warmup_updates=0,
        combat_actor_warmup_updates=0,
        freeze_trunk_during_durable_warmup=True,
        detach_value_heads_from_trunk=True,
        minority_only_within_branch_mpo=True,
        within_branch_minority_prob_cap=0.30,
        mpo_parent_prior_mix=0.25,
        max_parent_macro_kl=0.030,
        enable_post_step_kl_line_search=True,
        target_macro_step_kl=0.010,
        macro_policy_weight=0.0,
        combat_policy_weight=0.0,
        macro_entropy_weight=0.0,
        combat_entropy_weight=0.0,
        kickstart_kl_weight=0.0,
        kickstart_kl_floor=0.0,
        weight_decay=0.0,
    )
    learner = MacroCreditLearner(model=model, config=l_cfg, update=0, teacher_model=teacher)
    metrics = learner.learn(cohort)
    assert metrics["warmup_trunk_frozen"] is False
    assert metrics["shared_gradient_norm"] == 0.0
    assert metrics["domain_stream_gradient_norm"] == 0.0
    assert metrics["policy_head_gradient_norm"] == 0.0
    assert metrics["value_head_gradient_norm"] > 0.0
