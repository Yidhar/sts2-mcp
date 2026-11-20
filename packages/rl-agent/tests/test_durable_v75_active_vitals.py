"""Tests for Durable Afterstate v7.5 Active Un-Saturated Durable Vitals Gates."""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path
from typing import Any

import pytest

torch = pytest.importorskip("torch")

from sts2_rl.encoding import GroundedObservationEncoder  # noqa: E402
from sts2_rl.encoding.snapshot import collate_encoded_snapshots, collate_encoded_worlds  # noqa: E402
from sts2_rl.joint.macro_credit_data import (  # noqa: E402
    MacroCreditCohort,
    MacroCreditTrajectory,
    MacroRewardFacts,
)
from sts2_rl.joint.macro_credit_learner import MacroCreditLearner, MacroCreditLearnerConfig  # noqa: E402
from sts2_rl.joint.transitions import HybridVTraceStep  # noqa: E402
from sts2_rl.macro_competence_probes import (  # noqa: E402
    _build_combat_intent_probe_surface,
    probe_campfire_hp_sensitivity,
)
from sts2_rl.models.hybrid_actor_critic import (  # noqa: E402
    HybridActorCriticConfig,
    HybridDomainActorCriticModel,
    initialize_actor_critic_from_macro_publication,
)
from sts2_rl.semantics.clock import decision_discount  # noqa: E402
from sts2_rl.train_macro_credit import _actor_critic_config, load_macro_training_config  # noqa: E402
from sts2_rl.training.config import ModelConfig  # noqa: E402


def _tiny_v73_and_v75_configs() -> tuple[
    ModelConfig,
    HybridActorCriticConfig,
    HybridActorCriticConfig,
]:
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
    v73_cfg = HybridActorCriticConfig(
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
        enable_active_durable_vitals_gates=False,
    )
    v75_cfg = replace(v73_cfg, enable_active_durable_vitals_gates=True)
    return tiny, v73_cfg, v75_cfg


def test_v73_to_v75_hotstart_preserves_policy_and_activates_monotone_vitals() -> None:
    tiny, v73_cfg, v75_cfg = _tiny_v73_and_v75_configs()
    torch.manual_seed(75)
    src = HybridDomainActorCriticModel(v73_cfg)
    dst = HybridDomainActorCriticModel(v75_cfg)
    src.eval()
    dst.eval()

    summary = initialize_actor_critic_from_macro_publication(dst, src.state_dict())
    assert summary["fresh"]
    assert all(k.startswith("durable_active_") for k in summary["fresh"])
    assert summary.get("dropped", []) == []

    # Config round-trip
    restored = HybridActorCriticConfig.from_mapping(
        v75_cfg.as_mapping(), fallback_grounded=v75_cfg.grounded
    )
    assert restored == v75_cfg

    enc = GroundedObservationEncoder(tiny.to_encoding_config())

    # 1. Policy logits and combat value MUST remain bitwise identical to v7.3 on step 0.
    obs_c, legal_c = _build_combat_intent_probe_surface(incoming_damage=20, player_block=0, player_hp=40)
    snap_c = enc.encode(obs_c, legal_c).snapshot
    batch_c = collate_encoded_snapshots(
        (snap_c,),
        expected_config=snap_c.config,
        expected_fingerprint=snap_c.encoding_fingerprint,
        device=torch.device("cpu"),
        validate=False,
    )
    with torch.no_grad():
        out_src = src(batch_c, None, compute_durable_value=False, validate=True)
        out_dst = dst(batch_c, None, compute_durable_value=False, validate=True)
    assert torch.allclose(out_src.policy_logits, out_dst.policy_logits, atol=1e-6)
    assert out_src.combat_value is not None and out_dst.combat_value is not None
    assert torch.allclose(out_src.combat_value, out_dst.combat_value, atol=1e-6)

    # 2. V_durable MUST be strictly sensitive to HP, gold, and deck bloat (never dead-frozen at ~0).
    base_deck = [
        {"id": "STRIKE_IRONCLAD", "type": "Attack", "rarity": "Basic", "cost": 1, "quantity": 5},
        {"id": "DEFEND_IRONCLAD", "type": "Skill", "rarity": "Basic", "cost": 1, "quantity": 4},
        {"id": "BASH", "type": "Attack", "rarity": "Basic", "cost": 2, "quantity": 1},
    ]
    bloated_deck = list(base_deck) + [
        {"id": "STRIKE_IRONCLAD", "type": "Attack", "rarity": "Basic", "cost": 1, "quantity": 16}
    ]

    def _w(hp: int, gold: int = 100, deck: list[dict[str, Any]] = base_deck) -> Any:
        obs = {
            "phase": "map",
            "screen": "MAP",
            "decision_domain": "map",
            "run": {"active": True, "act": 1, "floor": 14, "ascension": 0},
            "player": {
                "character_id": "IRONCLAD",
                "hp": hp,
                "current_hp": hp,
                "max_hp": 80,
                "gold": gold,
                "deck": deck,
                "relics": [],
                "potions": [],
            },
        }
        return enc.encode_world_snapshot(obs)

    worlds = (_w(16), _w(40), _w(75), _w(40, gold=0), _w(40, gold=300), _w(40, deck=bloated_deck))
    wb, dids = collate_encoded_worlds(
        worlds,
        expected_config=worlds[0].config,
        expected_fingerprint=worlds[0].encoding_fingerprint,
        device=torch.device("cpu"),
    )
    with torch.no_grad():
        v1, v2, _ = dst.forward_durable_value_pair(wb, dids, None, validate=False)
    for vals in (v1, v2):
        assert float(vals[1] - vals[0]) > 0.20, "HP 16 -> 40 must strongly increase V_durable"
        assert float(vals[2] - vals[1]) > 0.05, "HP 40 -> 75 must strictly increase V_durable"
        assert float(vals[4] - vals[3]) > 0.05, "Gold 0 -> 300 must strictly increase V_durable"
        assert float(vals[1] - vals[5]) > 0.05, "Deck bloat 10 -> 26 must decrease V_durable"

    # 3. Campfire probe MUST show strong positive durable_value_rest_advantage_shift!
    camp = probe_campfire_hp_sensitivity(dst, enc)
    assert camp["mean_durable_rest_adv_low_hp"] > 0.15
    assert camp["durable_value_rest_advantage_shift"] > 0.15


def test_v75_durable_lr_group_warmup_and_v28_toml() -> None:
    tiny, _, v75_cfg = _tiny_v73_and_v75_configs()
    torch.manual_seed(175)
    model = HybridDomainActorCriticModel(v75_cfg)
    enc = GroundedObservationEncoder(tiny.to_encoding_config())

    l_cfg = MacroCreditLearnerConfig(
        target_mode="durable_afterstate_v2",
        enable_direct_vitals_context=True,
        enable_direct_combat_vitals_context=True,
        enable_active_durable_vitals_gates=True,
        enable_local_combat_critic=True,
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
        freeze_trunk_during_durable_warmup=True,
        durable_warmup_updates=1,
        combat_actor_warmup_updates=1,
        macro_actor_warmup_updates=2,
        detach_value_heads_from_trunk=True,
        learning_rate=1e-4,
        shared_learning_rate_scale=0.25,
        local_combat_value_lr_scale=3.0,
        durable_value_lr_scale=4.0,
        ppo_epochs_per_cohort=1,
        ppo_minibatches_per_cohort=2,
    )
    learner = MacroCreditLearner(model=model, config=l_cfg)
    lrs = [round(float(g["lr"]), 8) for g in learner.optimizer.param_groups]
    assert len(lrs) == 4
    assert lrs[0] == pytest.approx(3e-4)  # local_combat_val
    assert lrs[1] == pytest.approx(4e-4)  # durable_val
    assert lrs[2] == pytest.approx(1e-4)  # local
    assert lrs[3] == pytest.approx(2.5e-5)  # shared

    # Verify durable_active_* updates during pure-value warmup while trunk stays frozen.
    obs_m = {
        "phase": "rest_site",
        "screen": "REST_SITE",
        "decision_domain": "rest_site",
        "run": {"active": True, "act": 1, "floor": 12, "ascension": 0},
        "player": {
            "character_id": "IRONCLAD",
            "hp": 30,
            "current_hp": 30,
            "max_hp": 80,
            "gold": 120,
            "deck": [
                {"id": "STRIKE_IRONCLAD", "type": "Attack", "rarity": "Basic", "cost": 1, "quantity": 5},
            ],
            "relics": [],
            "potions": [],
        },
        "rest_site": {
            "visible": True,
            "options": [
                {"index": 0, "id": "rest", "type": "heal", "heal_amount": 24, "is_enabled": True},
                {"index": 1, "id": "smith", "type": "smith", "is_enabled": True},
            ],
        },
    }
    legal_m = [
        {"action_id": "rest:0", "kind": "choose_rest_option", "model_action_kind": "rest_site", "is_enabled": True, "option": {"index": 0, "id": "rest", "type": "heal", "heal_amount": 24}},
        {"action_id": "rest:1", "kind": "choose_rest_option", "model_action_kind": "rest_site", "is_enabled": True, "option": {"index": 1, "id": "smith", "type": "smith"}},
    ]
    snap_m = enc.encode(obs_m, legal_m).snapshot
    model.eval()
    with torch.no_grad():
        bm = collate_encoded_snapshots(
            (snap_m,),
            expected_config=snap_m.config,
            expected_fingerprint=snap_m.encoding_fingerprint,
            device=torch.device("cpu"),
            validate=False,
        )
        out = model(bm, None, compute_durable_value=True, validate=False)
        mask = torch.tensor(snap_m.action_mask, dtype=torch.bool)
        log_p = torch.log_softmax(out.policy_logits[0, : mask.numel()].masked_fill(~mask, -torch.inf), dim=0)

    fact = MacroRewardFacts(
        terminal_reward=2.0,
        progress_reward=0.0,
        revival_penalty=0.0,
        pace_penalty=0.0,
        hp_loss_penalty=0.0,
        revivals_used_delta=0,
        player_hp_lost_delta=0.0,
        floor_before=12,
        floor_after=17,
        native_steps=2,
    )
    step = HybridVTraceStep(
        snap_m,
        0,
        float(log_p[0].item()),
        fact.reward,
        decision_discount(floor_before=12, floor_after=17, terminal=True),
        True,
        "macro",
        "choose_rest_option",
        "macro",
        0,
    )
    trajs = tuple(
        MacroCreditTrajectory(
            f"v75-t-{i}",
            2 * (7_000_000 + i),
            0,
            (),
            (step,),
            None,
            None,
            "factual_terminal",
            (fact,),
            afterstate_nodes=(),
            macro_branch_epsilon=0.0,
        )
        for i in range(2)
    )
    pre_active = {
        k: v.detach().clone()
        for k, v in model.named_parameters()
        if k.startswith("durable_active_")
    }
    pre_policy = {
        k: v.detach().clone()
        for k, v in model.named_parameters()
        if k.startswith("candidate_policy_head.")
    }
    m0 = learner.learn(MacroCreditCohort("v75-c0", 0, (), natural_only=trajs))
    assert m0["warmup_trunk_frozen"] is True
    for k, prev in pre_policy.items():
        assert torch.equal(dict(model.named_parameters())[k], prev)
    delta_active = sum(
        float((dict(model.named_parameters())[k] - prev).abs().sum().item())
        for k, prev in pre_active.items()
    )
    assert delta_active > 0.0, "durable_active_* must update during pure-value warmup"

    # Load v28 TOML and verify all v7.5 invariants
    cfg_path = (
        Path(__file__).resolve().parents[1]
        / "config"
        / "experiments"
        / "hybrid_macro_credit_v28_durable_afterstate_v75_active_vitals_b0_act1_g4.toml"
    )
    mcfg = load_macro_training_config(cfg_path)
    assert mcfg.training.runtime.seed == 6_760_000
    assert mcfg.course.seed_index_offset == 795_000
    assert mcfg.learner.enable_active_durable_vitals_gates is True
    assert mcfg.learner.durable_value_lr_scale == pytest.approx(3.0)
    assert mcfg.learner.durable_warmup_updates == 5
    assert mcfg.learner.combat_actor_warmup_updates == 5
    assert mcfg.learner.macro_actor_warmup_updates == 15
    assert mcfg.learner.macro_entropy_weight == pytest.approx(0.0005)
    assert mcfg.learner.combat_entropy_weight == pytest.approx(0.012)
    assert mcfg.learner.local_combat_win_reward == pytest.approx(1.5)
    assert mcfg.learner.local_combat_hp_weight == pytest.approx(1.2)
    assert mcfg.learner.likelihood_tolerance == pytest.approx(0.15)
    assert _actor_critic_config(mcfg).enable_active_durable_vitals_gates is True
