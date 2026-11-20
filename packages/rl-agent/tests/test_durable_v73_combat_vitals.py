"""Tests for Durable Afterstate v7.3 Direct Combat Tactical Context & Combat KL Line Search."""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import pytest
import torch

from sts2_rl.encoding import GroundedEncodingConfig, GroundedObservationEncoder
from sts2_rl.encoding.grounded import _NUMERIC_SLOT_BY_KEY, _ZONE_IDS
from sts2_rl.encoding.snapshot import collate_encoded_snapshots
from sts2_rl.joint.macro_credit_data import (
    MacroCreditCohort,
    MacroCreditTrajectory,
    MacroRewardFacts,
)
from sts2_rl.joint.macro_credit_learner import MacroCreditLearner, MacroCreditLearnerConfig
from sts2_rl.joint.transitions import HybridVTraceStep
from sts2_rl.semantics.clock import decision_discount
from sts2_rl.macro_competence_probes import (
    _build_combat_intent_probe_surface,
    probe_combat_intent_sensitivity,
    run_all_competence_probes,
)
from sts2_rl.models.grounded_candidate import GroundedCandidateConfig
from sts2_rl.models.hybrid_actor_critic import (
    COMBAT_ENERGY_OFFSET,
    COMBAT_MAX_ENERGY_OFFSET,
    COMBAT_ROUND_OFFSET,
    COMBAT_VITALS_DIM,
    COMBAT_ZONE_ID,
    ENEMY_IS_ALIVE_OFFSET,
    ENEMY_ZONE_ID,
    HAND_ZONE_ID,
    INTENT_DAMAGE_OFFSET,
    INTENT_REPEATS_OFFSET,
    INTENT_TOTAL_DAMAGE_OFFSET,
    INTENT_ZONE_ID,
    NUMERIC_FACT_SLOT_START,
    PLAYER_BLOCK_OFFSET,
    PLAYER_CURRENT_HP_OFFSET,
    PLAYER_HP_OFFSET,
    PLAYER_MAX_HP_OFFSET,
    PLAYER_ZONE_ID,
    HybridActorCriticConfig,
    HybridDomainActorCriticModel,
    initialize_actor_critic_from_macro_publication,
)
from sts2_rl.train_macro_credit import load_macro_training_config
from sts2_rl.training.config import ModelConfig


def _tiny_v71_and_v73_configs() -> tuple[
    GroundedEncodingConfig,
    HybridActorCriticConfig,
    HybridActorCriticConfig,
]:
    tiny_model = ModelConfig(
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
    )
    enc_cfg = tiny_model.to_encoding_config()
    grounded = tiny_model.to_model_config()
    v71_cfg = HybridActorCriticConfig(
        grounded=grounded,
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
        enable_direct_combat_vitals_context=False,
    )
    v73_cfg = replace(v71_cfg, enable_direct_combat_vitals_context=True)
    return enc_cfg, v71_cfg, v73_cfg


def test_combat_vitals_slot_and_zone_constants_match_grounded_encoder() -> None:
    assert HAND_ZONE_ID == _ZONE_IDS["hand"]
    assert PLAYER_ZONE_ID == _ZONE_IDS["player"]
    assert ENEMY_ZONE_ID == _ZONE_IDS["enemy"]
    assert INTENT_ZONE_ID == _ZONE_IDS["intent"]
    assert COMBAT_ZONE_ID == _ZONE_IDS["combat"]

    assert PLAYER_BLOCK_OFFSET + NUMERIC_FACT_SLOT_START == _NUMERIC_SLOT_BY_KEY["block"]
    assert PLAYER_CURRENT_HP_OFFSET + NUMERIC_FACT_SLOT_START == _NUMERIC_SLOT_BY_KEY["current_hp"]
    assert PLAYER_HP_OFFSET + NUMERIC_FACT_SLOT_START == _NUMERIC_SLOT_BY_KEY["hp"]
    assert PLAYER_MAX_HP_OFFSET + NUMERIC_FACT_SLOT_START == _NUMERIC_SLOT_BY_KEY["max_hp"]
    assert INTENT_DAMAGE_OFFSET + NUMERIC_FACT_SLOT_START == _NUMERIC_SLOT_BY_KEY["damage"]
    assert INTENT_REPEATS_OFFSET + NUMERIC_FACT_SLOT_START == _NUMERIC_SLOT_BY_KEY["repeats"]
    assert INTENT_TOTAL_DAMAGE_OFFSET + NUMERIC_FACT_SLOT_START == _NUMERIC_SLOT_BY_KEY["total_damage"]
    assert COMBAT_ENERGY_OFFSET + NUMERIC_FACT_SLOT_START == _NUMERIC_SLOT_BY_KEY["energy"]
    assert COMBAT_MAX_ENERGY_OFFSET + NUMERIC_FACT_SLOT_START == _NUMERIC_SLOT_BY_KEY["max_energy"]
    assert COMBAT_ROUND_OFFSET + NUMERIC_FACT_SLOT_START == _NUMERIC_SLOT_BY_KEY["round"]
    assert ENEMY_IS_ALIVE_OFFSET + NUMERIC_FACT_SLOT_START == _NUMERIC_SLOT_BY_KEY["is_alive"]


def test_extract_combat_vitals_handles_multihit_dead_enemy_block_and_macro_zeroing() -> None:
    enc_cfg, _, v73_cfg = _tiny_v71_and_v73_configs()
    encoder = GroundedObservationEncoder(enc_cfg)
    model = HybridDomainActorCriticModel(v73_cfg)
    model.eval()

    # Case 1: Alive enemy with 3x2=6 multi-hit + Alive enemy with 8x1=8 + Dead enemy with 99x1 (must be ignored!)
    obs_multi, legal_multi = _build_combat_intent_probe_surface(
        incoming_damage=8,
        player_block=5,
        player_hp=40,
        player_max_hp=80,
    )
    obs_multi["combat"]["enemies"] = [
        {
            "entity_id": "CORPSE_SLUG_A",
            "combat_id": 1,
            "hp": 25,
            "max_hp": 25,
            "block": 0,
            "is_alive": True,
            "is_hittable": True,
            "intents": [{"type": "attack", "label": "6", "damage": 3, "repeats": 2}],
            "powers": [],
        },
        {
            "entity_id": "CORPSE_SLUG_B",
            "combat_id": 2,
            "hp": 10,
            "max_hp": 26,
            "block": 4,
            "is_alive": True,
            "is_hittable": True,
            "intents": [{"type": "attack", "label": "8", "damage": 8, "repeats": 1}],
            "powers": [],
        },
        {
            "entity_id": "CORPSE_SLUG_DEAD",
            "combat_id": 3,
            "hp": 0,
            "max_hp": 26,
            "block": 0,
            "is_alive": False,
            "is_hittable": False,
            "intents": [{"type": "attack", "label": "99", "damage": 99, "repeats": 1}],
            "powers": [],
        },
    ]

    # Case 2: Lethal incoming 35 vs 20 HP, 0 Block
    obs_lethal, legal_lethal = _build_combat_intent_probe_surface(
        incoming_damage=35,
        player_block=0,
        player_hp=20,
        player_max_hp=80,
    )

    # Case 3: Fully blocked 25 dmg with 25 block
    obs_blocked, legal_blocked = _build_combat_intent_probe_surface(
        incoming_damage=25,
        player_block=25,
        player_hp=40,
        player_max_hp=80,
    )

    # Case 4: Macro rest_site (must produce strictly 0.0 across all 16 dims)
    obs_macro = {
        "phase": "rest_site",
        "decision_domain": "rest_site",
        "run": {"active": True, "act": 1, "floor": 7, "ascension": 0},
        "player": {
            "character_id": "IRONCLAD",
            "hp": 40,
            "max_hp": 80,
            "gold": 120,
            "deck": [],
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
    legal_macro = [
        {"action_id": "rest:0", "kind": "choose_rest_option", "model_action_kind": "rest_site", "is_enabled": True, "option": {"index": 0, "id": "rest", "type": "heal"}},
        {"action_id": "rest:1", "kind": "choose_rest_option", "model_action_kind": "rest_site", "is_enabled": True, "option": {"index": 1, "id": "smith", "type": "smith"}},
    ]

    snaps = tuple(
        encoder.encode(o, l).snapshot
        for o, l in (
            (obs_multi, legal_multi),
            (obs_lethal, legal_lethal),
            (obs_blocked, legal_blocked),
            (obs_macro, legal_macro),
        )
    )
    batch = collate_encoded_snapshots(
        snaps,
        expected_config=snaps[0].config,
        expected_fingerprint=snaps[0].encoding_fingerprint,
        device=torch.device("cpu"),
        validate=False,
    )
    cv = model._extract_combat_vitals(batch.world, batch.domain_ids)
    assert cv.shape == (4, COMBAT_VITALS_DIM)
    assert torch.isfinite(cv).all()

    # Row 0: total_incoming = 3*2 + 8*1 = 14 (dead enemy's 99 ignored!), block = 5, unblocked = 9
    assert float(cv[0, 0]) == pytest.approx(1.0, abs=1e-5)  # is_attacked
    assert float(cv[0, 1]) == pytest.approx(14.0 / 30.0, abs=1e-2)  # incoming_norm
    assert float(cv[0, 2]) == pytest.approx(5.0 / 30.0, abs=1e-2)  # block_norm
    assert float(cv[0, 3]) == pytest.approx(9.0 / 30.0, abs=1e-2)  # unblocked_norm
    assert float(cv[0, 4]) == pytest.approx(1.0, abs=1e-5)  # needs_block
    assert float(cv[0, 7]) == pytest.approx(0.0, abs=1e-5)  # not lethal
    assert float(cv[0, 13]) == pytest.approx(2.0 / 3.0, abs=1e-4)  # 2 alive enemies
    assert float(cv[0, 14]) == pytest.approx(39.0 / 100.0, abs=1e-2)  # eHP = 25 + 10 + 4 = 39

    # Row 1: lethal (35 dmg >= 20 hp)
    assert float(cv[1, 3]) == pytest.approx(35.0 / 30.0, abs=1e-2)
    assert float(cv[1, 7]) == pytest.approx(1.0, abs=1e-5)  # lethal_risk == 1.0

    # Row 2: fully blocked (25 dmg, 25 block -> unblocked == 0, needs_block == 0)
    assert float(cv[2, 0]) == pytest.approx(1.0, abs=1e-5)
    assert float(cv[2, 3]) == pytest.approx(0.0, abs=1e-2)
    assert float(cv[2, 4]) == pytest.approx(0.0, abs=1e-5)

    # Row 3: macro row must be identically zero
    assert torch.equal(cv[3], torch.zeros(COMBAT_VITALS_DIM))


def test_v71_to_v73_zero_init_hotstart_is_bitwise_identical_and_config_roundtrips() -> None:
    enc_cfg, v71_cfg, v73_cfg = _tiny_v71_and_v73_configs()
    torch.manual_seed(73)
    src = HybridDomainActorCriticModel(v71_cfg)
    dst = HybridDomainActorCriticModel(v73_cfg)
    src.eval()
    dst.eval()

    summary = initialize_actor_critic_from_macro_publication(dst, src.state_dict())
    assert summary["fresh"]
    assert all(
        k.startswith(("combat_vitals_", "local_combat_value_vitals_"))
        for k in summary["fresh"]
    )

    # Config round-trip
    restored_cfg = HybridActorCriticConfig.from_mapping(
        v73_cfg.as_mapping(), fallback_grounded=v73_cfg.grounded
    )
    assert restored_cfg == v73_cfg

    encoder = GroundedObservationEncoder(enc_cfg)
    obs, legal = _build_combat_intent_probe_surface(incoming_damage=25, player_block=0, player_hp=40)
    snap = encoder.encode(obs, legal).snapshot
    batch = collate_encoded_snapshots(
        (snap,),
        expected_config=snap.config,
        expected_fingerprint=snap.encoding_fingerprint,
        device=torch.device("cpu"),
        validate=False,
    )
    with torch.no_grad():
        out_src = src(batch, None, compute_durable_value=True, validate=True)
        out_dst = dst(batch, None, compute_durable_value=True, validate=True)

    assert torch.allclose(out_src.policy_logits, out_dst.policy_logits, atol=1e-6)
    assert torch.allclose(out_src.value, out_dst.value, atol=1e-6)
    assert out_src.combat_value is not None and out_dst.combat_value is not None
    assert torch.allclose(out_src.combat_value, out_dst.combat_value, atol=1e-6)
    assert out_src.durable_value is not None and out_dst.durable_value is not None
    assert torch.allclose(out_src.durable_value, out_dst.durable_value, atol=1e-6)


def test_combat_vitals_warmup_isolation_and_combat_kl_line_search() -> None:
    enc_cfg, _, v73_cfg = _tiny_v71_and_v73_configs()
    torch.manual_seed(173)
    model = HybridDomainActorCriticModel(v73_cfg)
    encoder = GroundedObservationEncoder(enc_cfg)

    obs_atk, legal_atk = _build_combat_intent_probe_surface(incoming_damage=25, player_block=0, player_hp=40)
    obs_buf, legal_buf = _build_combat_intent_probe_surface(incoming_damage=0, player_block=0, player_hp=40)
    snap_atk = encoder.encode(obs_atk, legal_atk).snapshot
    snap_buf = encoder.encode(obs_buf, legal_buf).snapshot

    def _make_traj(seed: int, teacher_version: int) -> MacroCreditTrajectory:
        model.eval()
        steps: list[HybridVTraceStep] = []
        facts: list[MacroRewardFacts] = []
        hidden = model.initial_state(1, device=torch.device("cpu"))
        for idx, (snap, hp_lost, act_idx) in enumerate(((snap_atk, 25.0, 1), (snap_buf, 0.0, 2))):
            batch = collate_encoded_snapshots(
                (snap,),
                expected_config=snap.config,
                expected_fingerprint=snap.encoding_fingerprint,
                device=torch.device("cpu"),
                validate=False,
            )
            with torch.no_grad():
                out = model(batch, hidden, compute_durable_value=False, validate=False)
                mask = torch.tensor(snap.action_mask, dtype=torch.bool)
                log_p = torch.log_softmax(out.policy_logits[0, : mask.numel()].masked_fill(~mask, -torch.inf), dim=0)
                hidden = out.recurrent_state.detach()
            fact = MacroRewardFacts(
                terminal_reward=1.0 if idx == 1 else 0.0,
                progress_reward=0.0,
                revival_penalty=0.0,
                pace_penalty=-0.001,
                hp_loss_penalty=-0.2 if hp_lost > 0 else 0.0,
                revivals_used_delta=0,
                player_hp_lost_delta=hp_lost,
                floor_before=1,
                floor_after=1,
                native_steps=2,
            )
            facts.append(fact)
            steps.append(
                HybridVTraceStep(
                    snap,
                    act_idx,
                    float(log_p[act_idx].item()),
                    fact.reward,
                    0.0 if idx == 1 else decision_discount(floor_before=1, floor_after=1, terminal=False),
                    (idx == 1),
                    "combat",
                    "play_card",
                    "combat",
                    teacher_version,
                )
            )
        return MacroCreditTrajectory(
            f"v73-test-{seed}",
            2 * (6_850_000 + seed),
            teacher_version,
            (),
            tuple(steps),
            None,
            None,
            "factual_terminal",
            tuple(facts),
            afterstate_nodes=(),
            macro_branch_epsilon=0.0,
        )

    # 1. Warmup update (u=0 < durable_warmup_updates=1): ONLY pure value params (incl. local_combat_value_vitals_*) move!
    cfg_warmup = MacroCreditLearnerConfig(
        target_mode="durable_afterstate_v2",
        enable_direct_vitals_context=True,
        enable_direct_combat_vitals_context=True,
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
        macro_actor_warmup_updates=999,
        detach_value_heads_from_trunk=True,
        enable_post_step_kl_line_search=True,
        target_combat_step_kl=0.0005,
        learning_rate=5e-3,
        local_combat_value_lr_scale=2.0,
        normalize_combat_advantages=True,
        ppo_epochs_per_cohort=1,
        ppo_minibatches_per_cohort=2,
        combat_entropy_weight=0.005,
    )
    learner = MacroCreditLearner(model=model, config=cfg_warmup)
    pre_policy_params = {
        k: v.detach().clone()
        for k, v in model.named_parameters()
        if k.startswith("combat_vitals_policy_") or k.startswith("combat_vitals_to_state")
    }
    pre_val_vitals = {
        k: v.detach().clone()
        for k, v in model.named_parameters()
        if k.startswith("local_combat_value_vitals_")
    }
    cohort0 = MacroCreditCohort(
        cohort_id="warmup-0",
        teacher_version=0,
        roots=(),
        natural_only=(_make_traj(1, 0), _make_traj(2, 0)),
    )
    m0 = learner.learn(cohort0)
    assert m0["warmup_trunk_frozen"] is True
    for k, prev in pre_policy_params.items():
        assert torch.equal(dict(model.named_parameters())[k], prev), f"{k} must stay frozen during warmup"
    val_vitals_delta = sum(
        float((dict(model.named_parameters())[k] - prev).abs().sum().item())
        for k, prev in pre_val_vitals.items()
    )
    assert val_vitals_delta > 0.0, "local_combat_value_vitals_* must train during value warmup"

    # 2. Active combat update (u=1 >= 1) with aggressive LR=5e-3 and tiny target_combat_step_kl=0.0005:
    #    Post-step combat KL line search MUST trigger and scale back the policy step!
    cohort1 = MacroCreditCohort(
        cohort_id="active-1",
        teacher_version=1,
        roots=(),
        natural_only=(_make_traj(3, 1), _make_traj(4, 1)),
    )
    m1 = learner.learn(cohort1)
    assert m1["warmup_trunk_frozen"] is False
    assert m1["raw_updated_combat_policy_kl"] > 0.0005
    assert m1["line_search_backtracks"] >= 1
    assert m1["line_search_scale"] < 1.0
    assert m1["mean_updated_combat_policy_kl"] < m1["raw_updated_combat_policy_kl"]


def test_probe_combat_intent_sensitivity_and_v26_toml_load() -> None:
    enc_cfg, _, v73_cfg = _tiny_v71_and_v73_configs()
    model = HybridDomainActorCriticModel(v73_cfg)
    encoder = GroundedObservationEncoder(enc_cfg)
    report = probe_combat_intent_sensitivity(model, encoder)
    assert report["local_combat_critic_enabled"] is True
    for key in (
        "p_defend_buff0",
        "p_defend_atk25",
        "p_defend_atk25_blk25",
        "p_defend_lethal",
        "defend_atk25_minus_buff0",
        "c_val_safe_minus_lethal",
        "c_val_blk25_minus_blk0_under_atk25",
    ):
        assert isinstance(report[key], float)

    full_probe = run_all_competence_probes(model, encoder)
    assert "combat_intent_sensitivity" in full_probe

    cfg_path = (
        Path(__file__).resolve().parents[1]
        / "config"
        / "experiments"
        / "hybrid_macro_credit_v26_durable_afterstate_v73_combat_vitals_b0_g4.toml"
    )
    mcfg = load_macro_training_config(cfg_path)
    assert mcfg.learner.enable_direct_combat_vitals_context is True
    assert mcfg.learner.enable_post_step_kl_line_search is True
    assert mcfg.learner.target_combat_step_kl == pytest.approx(0.025)
    assert mcfg.learner.durable_warmup_updates == 5
    assert mcfg.learner.combat_actor_warmup_updates == 5
    assert mcfg.learner.combat_entropy_weight == pytest.approx(0.005)
