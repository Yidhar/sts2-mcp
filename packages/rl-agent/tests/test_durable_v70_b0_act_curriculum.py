"""Unit tests for Durable Afterstate Macro Credit v7.0:
- Zero-Revival Act Curriculum (B=0 across act1 -> act2 -> run)
- Non-saturating linear HP loss shaping (_effective_step_reward)
- Encounter-local combat value head & single-encounter V-trace(lambda) targets
- Two-phase stabilization warmup (durable_warmup_updates -> combat_actor -> macro_actor)
"""

from __future__ import annotations

import importlib.util
import sys
from dataclasses import replace
from pathlib import Path
from typing import Any

import pytest

torch = pytest.importorskip("torch")

from sts2_baseline import (  # noqa: E402
    ActFrontloadedDeathRewardCalculator,
    act_frontloaded_death_reward_identity,
)
from sts2_rl.contracts import EnvironmentResult, EnvironmentTransition  # noqa: E402
from sts2_rl.encoding import GroundedObservationEncoder  # noqa: E402
from sts2_rl.encoding.snapshot import collate_encoded_snapshots  # noqa: E402
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
from sts2_rl.joint.transitions import HybridVTraceStep  # noqa: E402
from sts2_rl.macro_root_policy import FrozenMacroRootPolicy, load_frozen_macro_teacher  # noqa: E402
from sts2_rl.models.hybrid_actor_critic import (  # noqa: E402
    HybridActorCriticConfig,
    HybridDomainActorCriticModel,
)
from sts2_rl.semantics.clock import decision_discount  # noqa: E402
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


def _make_env_result(*, act: int, floor: int, terminated: bool = False, hp_lost: float = 0.0) -> EnvironmentResult:
    return EnvironmentResult(
        episode_id="ep",
        step_index=1,
        observation={
            "phase": "map",
            "run": {"active": not terminated, "act": act, "floor": floor},
            "_training": {"revival_budget": 0, "revivals_used": 0, "player_hp_lost": hp_lost},
        },
        legal_actions=(),
        terminated=terminated,
        truncated=False,
        terminal_reason="defeat" if terminated else None,
        transition=EnvironmentTransition(
            episode_id="ep",
            step_index=1,
            before_state_version=0,
            after_state_version=1,
            facts={"run_result": "defeat"} if terminated else {},
        ),
    )


def test_act2_reward_objective_clears_on_entering_act3_and_preserves_reward_v9_identity() -> None:
    calc = ActFrontloadedDeathRewardCalculator(
        objective="act2",
        discount=1.0,
        maximum_episode_steps=4096,
        revival_budget=0,
    )
    calc.begin_episode()

    s_a1 = _make_env_result(act=1, floor=16)
    s_a2 = _make_env_result(act=2, floor=17)
    r_enter_a2 = calc.evaluate(before=s_a1, after=s_a2, deadlock=False, horizon_exhausted=False)
    assert not r_enter_a2.task_terminal
    assert r_enter_a2.outcome == "ongoing"
    assert r_enter_a2.progress_reward > 0.0

    s_a2_boss = _make_env_result(act=2, floor=32)
    s_a3_entry = _make_env_result(act=3, floor=34)
    r_enter_a3 = calc.evaluate(before=s_a2_boss, after=s_a3_entry, deadlock=False, horizon_exhausted=False)
    assert r_enter_a3.task_terminal
    assert r_enter_a3.outcome == "success"
    assert r_enter_a3.terminal_reward == pytest.approx(2.0)

    id_b0 = act_frontloaded_death_reward_identity(revival_budget=0)
    assert id_b0["version"] == "sts2-run-survival-efficiency-v9"
    assert id_b0["revival_budget"] == 0


def test_v70_config_loads_and_hotstarts_b24_to_b0_to_b0(tmp_path: Path, monkeypatch: Any) -> None:
    monkeypatch.setenv("STS2_ARTIFACT_ROOT", str(tmp_path))
    repo_root = Path(__file__).resolve().parents[1]
    toml_path = (
        repo_root
        / "config"
        / "experiments"
        / "hybrid_macro_credit_v23_durable_afterstate_v70_act_b0_g4_turbo.toml"
    )
    macro_cfg = load_macro_training_config(toml_path)
    assert macro_cfg.training.curriculum.revival_budget == 0
    assert macro_cfg.training.curriculum.reward_objective == "act1"
    assert macro_cfg.learner.revival_budget == 0
    assert macro_cfg.learner.enable_local_combat_critic is True
    assert macro_cfg.learner.macro_actor_warmup_updates == 15
    assert macro_cfg.learner.linear_hp_loss_weight == pytest.approx(0.50)

    # Override to act2 / run via TOML --set syntax
    macro_act2 = load_macro_training_config(
        toml_path,
        overrides=('training.curriculum.reward_objective="act2"',),
    )
    assert macro_act2.training.curriculum.reward_objective == "act2"

    # Build tiny b24 v6.4-style parent publication (without local_combat_value_head)
    tiny = _tiny_model_config()
    parent_ac = HybridActorCriticConfig.durable_v11(tiny.to_model_config())
    assert parent_ac.enable_local_combat_value_head is False
    parent_model = HybridDomainActorCriticModel(parent_ac)
    opt = torch.optim.AdamW(parent_model.parameters(), lr=1e-4)
    parent_id = MacroCheckpointIdentity(
        course_config={"version": "test"},
        model_config=parent_ac.as_mapping(),
        reward_identity=act_frontloaded_death_reward_identity(revival_budget=24),
        lineage_id="v64-b24-parent",
        learner_version=MACRO_CREDIT_LEARNER_VERSION,
    )
    p_ckpt = tmp_path / "b24" / "checkpoint.pt"
    p_pub = tmp_path / "b24" / "publication.pt"
    save_macro_checkpoint(
        p_ckpt,
        identity=parent_id,
        model=parent_model,
        optimizer=opt,
        update=60,
        next_seed_index=600_000,
        initialization={"mode": "test"},
        artifact_root=tmp_path,
    )
    publish_macro_checkpoint(p_ckpt, p_pub, identity=parent_id, artifact_root=tmp_path)

    # Hot-start Stage 1 (B=0, act1) with enable_local_combat_value_head=True
    train_b0_a1 = replace(
        macro_cfg.training,
        model=tiny,
        curriculum=replace(
            macro_cfg.training.curriculum,
            revival_budget=0,
            reward_objective="act1",
        ),
    )
    stage1_ac = replace(parent_ac, enable_local_combat_value_head=True)
    tiny_macro_cfg = replace(
        macro_cfg,
        training=train_b0_a1,
        branches=replace(macro_cfg.branches, max_candidates=32),
    )
    assert _actor_critic_config(tiny_macro_cfg).enable_local_combat_value_head is True

    pol_s1 = FrozenMacroRootPolicy(
        p_pub,
        train_b0_a1,
        "cpu",
        initialize_from_macro_publication=True,
        actor_critic_config=stage1_ac,
    )
    assert pol_s1.model.local_combat_value_head is not None
    assert any(name.startswith("local_combat_value_head.") for name in pol_s1.provenance["fresh_tensor_names"])

    # Save Stage 1 B=0 publication and verify both load_frozen_macro_teacher AND B=0 -> B=0 Stage 2 hot-start!
    s1_opt = torch.optim.AdamW(pol_s1.model.parameters(), lr=1e-4)
    s1_id = MacroCheckpointIdentity(
        course_config={"version": "test"},
        model_config=stage1_ac.as_mapping(),
        reward_identity=act_frontloaded_death_reward_identity(revival_budget=0),
        lineage_id="v70-b0-act1",
        learner_version=MACRO_CREDIT_LEARNER_VERSION,
    )
    s1_ckpt = tmp_path / "b0_act1" / "checkpoint.pt"
    s1_pub = tmp_path / "b0_act1" / "publication.pt"
    save_macro_checkpoint(
        s1_ckpt,
        identity=s1_id,
        model=pol_s1.model,
        optimizer=s1_opt,
        update=25,
        next_seed_index=700_000,
        initialization={"mode": "test"},
        artifact_root=tmp_path,
    )
    publish_macro_checkpoint(s1_ckpt, s1_pub, identity=s1_id, artifact_root=tmp_path)

    loaded_teacher = load_frozen_macro_teacher(
        s1_pub,
        expected_encoding=tiny.to_encoding_config(),
        device="cpu",
    )
    assert loaded_teacher.model.actor_critic_config.enable_local_combat_value_head is True
    assert loaded_teacher.provenance["source_reward_identity"]["revival_budget"] == 0

    train_b0_a2 = replace(
        train_b0_a1,
        curriculum=replace(train_b0_a1.curriculum, reward_objective="act2"),
    )
    pol_s2 = FrozenMacroRootPolicy(
        s1_pub,
        train_b0_a2,
        "cpu",
        initialize_from_macro_publication=True,
        actor_critic_config=stage1_ac,
    )
    assert pol_s2.provenance["fresh_tensor_names"] == []


def _build_two_encounter_cohort(
    model: HybridDomainActorCriticModel,
    encoder: GroundedObservationEncoder,
    *,
    num_trajs: int = 2,
) -> MacroCreditCohort:
    trajs: list[MacroCreditTrajectory] = []
    for g in range(num_trajs):
        combat_obs_1 = {
            "phase": "combat",
            "decision_domain": "combat",
            "run": {"active": True, "act": 1, "floor": 1},
            "player": {
                "id": "ironclad",
                "side": "player",
                "hp": 72,
                "max_hp": 80,
                "energy": 3,
                "hand": [{"id": "strike", "cost": 1}, {"id": "defend", "cost": 1}],
            },
            "combat": {"in_progress": True, "enemies": [{"id": "jaw_worm", "side": "enemy", "hp": 30, "max_hp": 42}]},
        }
        combat_legal = [
            {"action_handle": "c0", "kind": "play_card", "model_action_kind": "play_card", "is_enabled": True, "card": {"id": "strike", "cost": 1}},
            {"action_handle": "c1", "kind": "play_card", "model_action_kind": "play_card", "is_enabled": True, "card": {"id": "defend", "cost": 1}},
        ]
        c_snap1 = encoder.encode(combat_obs_1, combat_legal).snapshot

        reward_obs = {
            "phase": "card_reward",
            "decision_domain": "card_reward",
            "run": {"active": True, "act": 1, "floor": 1},
            "player": {
                "character_id": "IRONCLAD",
                "hp": 60,
                "current_hp": 60,
                "max_hp": 80,
                "gold": 110,
                "deck": [{"id": "strike_0", "type": "Attack", "rarity": "Basic", "cost": 1}],
                "relics": [],
                "potions": [],
            },
            "card_reward_selection": {
                "can_skip": True,
                "cards": [
                    {"id": "shrug", "type": "Skill", "cost": 1},
                    {"id": "pommel", "type": "Attack", "cost": 1},
                ],
            },
        }
        reward_legal = [
            {"action_id": "r0", "kind": "select_card_reward", "model_action_kind": "card_reward", "is_enabled": True, "card": {"id": "shrug", "type": "Skill", "cost": 1}},
            {"action_id": "r1", "kind": "select_card_reward", "model_action_kind": "card_reward", "is_enabled": True, "card": {"id": "pommel", "type": "Attack", "cost": 1}},
            {"action_id": "skip", "kind": "skip_card_reward", "model_action_kind": "card_reward", "is_enabled": True},
        ]
        dec = forward_decision(reward_obs, reward_legal, control_domain="macro")
        assert dec is not None
        m_enc = encoder.encode(reward_obs, [c.semantic_action for c in dec.candidates])
        grouped_cands = tuple(dec.candidates[ref.representative_position] for ref in m_enc.actions)
        post_worlds = synthesize_deterministic_afterstates(
            encoder=encoder, observation=reward_obs, decision=dec, candidates=grouped_cands
        )
        assert post_worlds is not None

        combat_obs_2 = {
            **combat_obs_1,
            "run": {"active": True, "act": 1, "floor": 2},
        }
        c_snap2 = encoder.encode(combat_obs_2, combat_legal).snapshot

        snaps = (c_snap1, m_enc.snapshot, c_snap2)
        h = None
        beh_lps: list[float] = []
        acts = [0, g % 2, 1]
        with torch.no_grad():
            for idx, (sn, a_i) in enumerate(zip(snaps, acts, strict=True)):
                b = collate_encoded_snapshots(
                    (sn,),
                    expected_config=sn.config,
                    expected_fingerprint=sn.encoding_fingerprint,
                    device=torch.device("cpu"),
                )
                out = model(b, h, validate=False)
                h = out.recurrent_state
                msk = torch.tensor(sn.action_mask, dtype=torch.bool)
                lp = torch.log_softmax(out.policy_logits[0, : msk.numel()].masked_fill(~msk, -torch.inf), dim=0)
                if idx == 1:
                    lp = macro_mixture_log_probs(lp, sn, 0.04)
                beh_lps.append(float(lp[a_i].item()))

        # Encounter 1 wins on floor 1 losing 16 HP; Macro picks card advancing floor 1->2;
        # Encounter 2 dies on floor 2 losing 60 HP (terminal_reward = -1.0)!
        f0 = MacroRewardFacts(0.0, 0.0, 0.0, -0.001, -0.16, 0, 16.0, 1, 1, 3)
        f1 = MacroRewardFacts(0.0, 0.04, 0.0, -0.001, 0.0, 0, 0.0, 1, 2, 2)
        f2 = MacroRewardFacts(-1.0, 0.0, 0.0, -0.001, -0.45, 0, 60.0, 2, 2, 4)

        s0 = HybridVTraceStep(c_snap1, acts[0], beh_lps[0], f0.reward, decision_discount(floor_before=1, floor_after=1, terminal=False), False, "combat", "play_card", "combat", 0)
        s1 = HybridVTraceStep(m_enc.snapshot, acts[1], beh_lps[1], f1.reward, decision_discount(floor_before=1, floor_after=2, terminal=False), False, "reward", grouped_cands[acts[1]].branch, "macro", 0)
        s2 = HybridVTraceStep(c_snap2, acts[2], beh_lps[2], f2.reward, 0.0, True, "combat", "play_card", "combat", 0)

        node1 = MacroAfterstateNode(1, "reward", tuple(c.branch for c in grouped_cands), post_worlds)
        trajs.append(
            MacroCreditTrajectory(
                f"traj-v70-{g}",
                2 * (6_800_000 + g),
                0,
                (),
                (s0, s1, s2),
                None,
                None,
                "factual_terminal",
                (f0, f1, f2),
                afterstate_nodes=(node1,),
                macro_branch_epsilon=0.04,
            )
        )
    return MacroCreditCohort("cohort-v70", 0, (), natural_only=tuple(trajs))


def test_linear_hp_shaping_local_combat_encounter_isolation_and_two_phase_warmup() -> None:
    torch.manual_seed(77)
    tiny = _tiny_model_config()
    ac_cfg = replace(
        HybridActorCriticConfig.durable_v11(tiny.to_model_config()),
        enable_local_combat_value_head=True,
    )
    model = HybridDomainActorCriticModel(ac_cfg)
    teacher = HybridDomainActorCriticModel(ac_cfg)
    teacher.load_state_dict(model.state_dict())
    teacher.eval()
    teacher.requires_grad_(False)
    encoder = GroundedObservationEncoder(tiny.to_encoding_config())

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
        freeze_trunk_during_durable_warmup=True,
        durable_warmup_updates=1,
        combat_actor_warmup_updates=1,
        macro_actor_warmup_updates=3,
        linear_hp_loss_weight=0.50,
        hp_loss_reference=80.0,
        enable_local_combat_critic=True,
        local_combat_win_reward=1.0,
        local_combat_loss_reward=-1.0,
        local_combat_hp_weight=1.5,
        local_combat_step_penalty=0.004,
        detach_value_heads_from_trunk=True,
        detach_macro_branch_prior_lse=True,
        isolate_nonreward_macro_trunk_grads=True,
        zero_macro_pace_penalty=True,
        revival_budget=0,
        eval_batch_size=4,
        chunk_batch_size=4,
        distill_total_updates=4,
        kickstart_kl_weight=0.04,
        kickstart_kl_floor=0.01,
    )
    learner = MacroCreditLearner(model=model, config=l_cfg, teacher_model=teacher)
    cohort = _build_two_encounter_cohort(model, encoder)
    ref = cohort.natural_only[0]

    # 1. Verify non-saturating linear HP replacement in _effective_step_reward:
    # Step 0 lost 16 HP (f0.hp_loss_penalty=-0.16 replaced by -0.50 * 16/80 = -0.10):
    r0_eff = learner._effective_step_reward(ref, 0)
    assert r0_eff == pytest.approx(-0.001 - 0.50 * (16.0 / 80.0))
    # Step 2 lost 60 HP even after cumulative damage (f2.hp_loss_penalty=-0.45 replaced by -0.50 * 60/80 = -0.375):
    r2_eff = learner._effective_step_reward(ref, 2)
    assert r2_eff == pytest.approx(-1.0 - 0.001 - 0.50 * (60.0 / 80.0))

    # 2. Verify encounter-local combat targets completely isolate Encounter 1 (win) from Encounter 2 (death)!
    ev = learner._evaluate(ref, root_index=-1)
    assert ev.combat_values is not None
    c_targets, c_advs = learner._combat_targets(ref, ev)
    # Step 0 won Encounter 1 losing 16 HP -> +1.0 - 1.5*(16/80) - 0.004 = +0.696 (positive despite Floor 2 death!)
    assert float(c_targets[0].item()) == pytest.approx(1.0 - 1.5 * (16.0 / 80.0) - 0.004, abs=1e-5)
    # Step 1 is macro -> zeroed out
    assert float(c_targets[1].item()) == pytest.approx(0.0)
    # Step 2 died in Encounter 2 losing 60 HP -> -1.0 - 1.5*(60/80) - 0.004 = -2.129
    assert float(c_targets[2].item()) == pytest.approx(-1.0 - 1.5 * (60.0 / 80.0) - 0.004, abs=1e-5)

    # 3. Update 0 (< durable_warmup_updates=1): trunk frozen, only pure value heads (incl. local_combat_value_head) update!
    c_head_before = dict(model.named_parameters())["local_combat_value_head.3.weight"].detach().clone()
    trunk_before = dict(model.named_parameters())["world_encoder.layers.0.norm1.weight"].detach().clone()
    metrics_u0 = learner.learn(cohort)
    assert metrics_u0["macro_mpo_policy_loss"] == pytest.approx(0.0)
    c_head_after_u0 = dict(model.named_parameters())["local_combat_value_head.3.weight"].detach().clone()
    trunk_after_u0 = dict(model.named_parameters())["world_encoder.layers.0.norm1.weight"].detach().clone()
    assert not torch.allclose(c_head_before, c_head_after_u0)
    assert torch.allclose(trunk_before, trunk_after_u0)

    # 4. Update 1 (combat_actor_warmup_updates=1 <= update=1 < macro_actor_warmup_updates=3):
    # Combat actor + trunk update, while Macro MPO remains gated at 0.0!
    cohort_u1 = _build_two_encounter_cohort(model, encoder)
    cohort_u1 = MacroCreditCohort(
        "cohort-v70-u1",
        1,
        (),
        natural_only=tuple(
            replace(
                t,
                teacher_version=1,
                steps=tuple(replace(s, policy_version=1) for s in t.steps),
            )
            for t in cohort_u1.natural_only
        ),
    )
    metrics_u1 = learner.learn(cohort_u1)
    assert metrics_u1["macro_mpo_policy_loss"] == pytest.approx(0.0)
    assert metrics_u1["combat_value_loss"] > 0.0


def test_zero_revival_act_ladder_promotion_gates() -> None:
    repo_root = Path(__file__).resolve().parents[1]
    script_path = repo_root / "scripts" / "run_colab_g4_ratchet_curriculum.py"
    spec = importlib.util.spec_from_file_location("ratchet_mod", script_path)
    assert spec is not None and spec.loader is not None
    mod = importlib.util.module_from_spec(spec)
    sys.modules["ratchet_mod"] = mod
    spec.loader.exec_module(mod)

    ladder = mod.ZERO_REVIVAL_ACT_LADDER
    assert [s.stage_key for s in ladder] == ["b0-act1", "b0-act2", "b0-act3"]
    assert [s.reward_objective for s in ladder] == ["act1", "act2", "run"]
    assert all(s.budget == 0 for s in ladder)

    s1 = ladder[0]
    ok_pass, _ = mod.evaluate_stage_promotion_gate(
        {
            "policy_version": 20,
            "episodes": 48,
            "revival_budget": 0,
            "reward_objective": "act1",
            "deadlock_count": 0,
            "successful_run_count": 30,
            "mean_revivals_used": 0.0,
            "skip_rate": 0.11,
            "mean_final_deck_size": 19.4,
        },
        s1,
        expected_episodes=48,
        min_policy_version=5,
    )
    assert ok_pass is True

    ok_wrong_obj, rep_wrong = mod.evaluate_stage_promotion_gate(
        {
            "policy_version": 20,
            "episodes": 48,
            "revival_budget": 0,
            "reward_objective": "act1",
            "deadlock_count": 0,
            "successful_run_count": 30,
            "mean_revivals_used": 0.0,
            "skip_rate": 0.11,
            "mean_final_deck_size": 19.4,
        },
        ladder[1],  # b0-act2 expects objective="act2"!
        expected_episodes=48,
        min_policy_version=5,
    )
    assert ok_wrong_obj is False
    assert rep_wrong["checks"]["objective_match"] is False
