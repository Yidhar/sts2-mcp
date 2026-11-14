"""Unit tests for Durable Afterstate Macro Credit v2 architecture."""

from __future__ import annotations

import copy
import json
from pathlib import Path
from typing import Any

import pytest

torch = pytest.importorskip("torch")

from sts2_rl.encoding import GroundedObservationEncoder, grounding_encoding_identity  # noqa: E402
from sts2_rl.encoding.snapshot import collate_encoded_snapshots, collate_encoded_worlds  # noqa: E402
from sts2_rl.joint.afterstate import synthesize_deterministic_afterstates  # noqa: E402
from sts2_rl.joint.macro_credit_data import (  # noqa: E402
    MacroAfterstateNode,
    MacroCreditCohort,
    MacroCreditTrajectory,
    MacroRewardFacts,
    macro_mixture_log_probs,
)
from sts2_rl.joint.macro_credit_learner import MacroCreditLearner, MacroCreditLearnerConfig  # noqa: E402
from sts2_rl.joint.transitions import HybridVTraceStep  # noqa: E402
from sts2_rl.macro_competence_probes import (  # noqa: E402
    run_all_competence_probes,
    summarize_episode_macro_discipline,
)
from sts2_rl.models.hybrid_actor_critic import (  # noqa: E402
    HybridActorCriticConfig,
    HybridDomainActorCriticModel,
    hierarchical_greedy_action_index,
    initialize_actor_critic_from_iql,
)
from sts2_rl.semantics.forward import forward_decision  # noqa: E402
from sts2_rl.train_macro_credit import load_macro_training_config  # noqa: E402
from sts2_rl.training.config import ModelConfig  # noqa: E402


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


def _sample_deck() -> list[dict[str, Any]]:
    return [
        {
            "id": f"basic_strike_{i}",
            "type": "Attack",
            "rarity": "Basic",
            "cost": 1,
            "is_upgradable": True,
            "is_removable": True,
            "dynamic_vars": {"Damage": 6},
            "upgrade_preview": {
                "id": f"basic_strike_{i}",
                "type": "Attack",
                "rarity": "Basic",
                "cost": 1,
                "upgraded": True,
                "dynamic_vars": {"Damage": 9},
            },
        }
        for i in range(5)
    ] + [
        {
            "id": "power_inflame",
            "type": "Power",
            "rarity": "Uncommon",
            "cost": 2,
            "is_upgradable": True,
            "is_removable": True,
            "dynamic_vars": {"Strength": 2},
            "keywords": ["retain"],
            "upgrade_preview": {
                "id": "power_inflame",
                "type": "Power",
                "rarity": "Uncommon",
                "cost": 1,
                "upgraded": True,
                "dynamic_vars": {"Strength": 3},
                "keywords": ["retain", "innate"],
            },
        }
    ]


def _rest_surface(deck: list[dict[str, Any]] | None = None, *, include_preview: bool = True) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    cards = copy.deepcopy(deck if deck is not None else _sample_deck())
    if not include_preview:
        for card in cards:
            card.pop("upgrade_preview", None)
    obs: dict[str, Any] = {
        "phase": "rest_site",
        "decision_domain": "rest_site",
        "run": {"active": True, "act": 1, "floor": 6, "ascension": 0},
        "player": {
            "character_id": "IRONCLAD",
            "hp": 32,
            "max_hp": 80,
            "gold": 150,
            "open_potion_slots": 2,
            "deck": cards,
            "relics": [{"id": "BURNING_BLOOD", "has_upon_pickup_effect": False}],
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
    legal = [
        {
            "action_id": "rest:0",
            "kind": "choose_rest_option",
            "model_action_kind": "rest_site",
            "is_enabled": True,
            "option": {"index": 0, "id": "rest", "type": "heal", "heal_amount": 24},
        },
        {
            "action_id": "rest:1",
            "kind": "choose_rest_option",
            "model_action_kind": "rest_site",
            "is_enabled": True,
            "option": {"index": 1, "id": "smith", "type": "smith"},
        },
    ]
    return obs, legal


def test_encoder_v19_upgrade_preview_zone_and_structural_deltas() -> None:
    ident = grounding_encoding_identity()
    assert ident["version"] == "grounded-relational-runtime-encoding-v19"
    assert ident["feature_abi_end"] == 219

    cfg = _tiny_model_config()
    encoder = GroundedObservationEncoder(cfg.to_encoding_config())
    obs, legal = _rest_surface()
    decision = forward_decision(obs, legal, control_domain="macro")
    assert decision is not None
    encoded = encoder.encode(obs, [c.semantic_action for c in decision.candidates])

    # Non-upgrade world tokens must leave slots 215..218 strictly zero.
    batch = encoded.batch
    assert bool(torch.all(batch.world.features[:, :, 215:219] == 0.0).item())

    # All 6 Smith upgrade_preview card tokens in zone 26 carry positive delta norms
    # (child keyword tokens inside zone 26 keep zero card-delta slots).
    local_zones = batch.candidates.local_zone_ids
    has_zone_26 = local_zones.eq(26) & batch.candidates.local_mask
    assert bool(has_zone_26.any().item())
    preview_deltas = batch.candidates.local_features[has_zone_26][:, 215:219]
    nonzero_preview_cards = int((preview_deltas.abs().sum(dim=-1) > 0.0).sum().item())
    assert nonzero_preview_cards == 6
    non_preview = (~local_zones.eq(26)) & batch.candidates.local_mask
    if bool(non_preview.any().item()):
        assert bool(torch.all(batch.candidates.local_features[non_preview][:, 215:219] == 0.0).item())


def test_hybrid_actor_critic_v2_combat_equivalence_and_macro_cardinality_neutrality() -> None:
    cfg = _tiny_model_config()
    encoder = GroundedObservationEncoder(cfg.to_encoding_config())

    base_model = HybridDomainActorCriticModel(HybridActorCriticConfig(cfg.to_model_config())).eval()
    v2_model = HybridDomainActorCriticModel(HybridActorCriticConfig.durable_v2(cfg.to_model_config())).eval()
    initialize_actor_critic_from_iql(v2_model, base_model.state_dict())

    # 1. Step-0 Combat Equivalence: v2 policy probabilities must equal v1 flat softmax exactly.
    combat_obs = {
        "phase": "combat",
        "decision_domain": "combat",
        "player": {"id": "ironclad", "side": "player", "hp": 50, "max_hp": 80, "energy": 3, "hand": [{"id": "strike", "cost": 1}]},
        "combat": {"in_progress": True, "enemies": [{"id": "cultist", "side": "enemy", "hp": 40, "max_hp": 48}]},
    }
    combat_actions = [
        {"action_handle": "c0", "kind": "play_card", "model_action_kind": "play_card", "is_enabled": True, "card": {"id": "strike", "cost": 1}},
        {"action_handle": "c1", "kind": "end_turn", "model_action_kind": "end_turn", "is_enabled": True},
    ]
    combat_enc = encoder.encode(combat_obs, combat_actions)
    with torch.no_grad():
        out_v1 = base_model(combat_enc.batch)
        out_v2 = v2_model(combat_enc.batch)
    p_v1 = torch.softmax(out_v1.policy_logits[0], dim=0)
    p_v2 = torch.softmax(out_v2.policy_logits[0], dim=0)
    torch.testing.assert_close(p_v2, p_v1, rtol=1e-5, atol=1e-6)

    # 2. Step-0 Macro Cardinality Neutrality: 1 Rest vs multiple Smith cards gives Rest 0.5 branch mass!
    rest_obs, rest_legal = _rest_surface()
    rest_dec = forward_decision(rest_obs, rest_legal, control_domain="macro")
    assert rest_dec is not None and len(rest_dec.candidates) == 7  # 1 Rest + 6 Smith
    rest_enc = encoder.encode(rest_obs, [c.semantic_action for c in rest_dec.candidates])
    with torch.no_grad():
        rest_out_v2 = v2_model(rest_enc.batch)
    probs = torch.softmax(rest_out_v2.policy_logits[0], dim=0)
    p_rest = sum(
        float(probs[idx].item())
        for idx, ref in enumerate(rest_enc.actions)
        if rest_dec.candidates[ref.representative_position].branch == "rest"
    )
    assert abs(p_rest - 0.5) < 1e-5

    # Hierarchical greedy selects the single Rest action when p(Rest) > 0.5 even if individual Smith cards split 0.49.
    skewed_logits = torch.log(torch.tensor([0.51, 0.25, 0.24], dtype=torch.float32))
    # Suppose candidate 0 is branch 0 (Rest: 0.49) and candidates 1,2 are branch 1 (Smith: 0.26 + 0.25 = 0.51)
    branch_skew = torch.log(torch.tensor([0.49, 0.26, 0.25], dtype=torch.float32))
    greedy_idx = hierarchical_greedy_action_index(
        branch_skew,
        torch.tensor([0, 1, 1], dtype=torch.long),
        torch.tensor([True, True, True], dtype=torch.bool),
        domain_id=rest_enc.snapshot.domain_id,
    )
    assert greedy_idx == 1  # Picks best candidate inside winning Smith branch (0.51 > 0.49), NOT singleton 0.49!
    del skewed_logits

    # 3. Candidate-independence of forward_durable_value vs forward.durable_value.
    world_snap = encoder.encode_world_snapshot(rest_obs)
    world_batch, domain_ids = collate_encoded_worlds(
        (world_snap,),
        expected_config=world_snap.config,
        expected_fingerprint=world_snap.encoding_fingerprint,
        device="cpu",
    )
    with torch.no_grad():
        d_standalone, _ = v2_model.forward_durable_value(world_batch, domain_ids)
    assert rest_out_v2.durable_value is not None
    torch.testing.assert_close(d_standalone, rest_out_v2.durable_value, rtol=1e-5, atol=1e-6)


def test_afterstate_synthesizer_deterministic_surfaces_and_fail_closed_guards() -> None:
    cfg = _tiny_model_config()
    encoder = GroundedObservationEncoder(cfg.to_encoding_config())

    # 1. Rest surface with upgrade_preview succeeds; without upgrade_preview fails closed.
    rest_obs, rest_legal = _rest_surface(include_preview=True)
    rest_dec = forward_decision(rest_obs, rest_legal, control_domain="macro")
    assert rest_dec is not None
    rest_worlds = synthesize_deterministic_afterstates(encoder=encoder, observation=rest_obs, decision=rest_dec)
    assert rest_worlds is not None and len(rest_worlds) == len(rest_dec.candidates)

    blind_obs, blind_legal = _rest_surface(include_preview=False)
    blind_dec = forward_decision(blind_obs, blind_legal, control_domain="macro")
    assert blind_dec is not None
    assert synthesize_deterministic_afterstates(encoder=encoder, observation=blind_obs, decision=blind_dec) is None

    # 2. Card reward Take/Skip succeeds; stochastic claim_card fails closed.
    reward_obs = {
        "phase": "card_reward",
        "decision_domain": "card_reward",
        "run": {"active": True, "act": 1, "floor": 3},
        "player": {"character_id": "IRONCLAD", "hp": 60, "max_hp": 80, "gold": 110, "deck": _sample_deck(), "relics": [], "potions": []},
        "card_reward_selection": {"can_skip": True, "cards": [{"id": "shrug_it_off", "type": "Skill", "cost": 1}]},
    }
    reward_legal = [
        {"action_id": "r0", "kind": "select_card_reward", "model_action_kind": "card_reward", "is_enabled": True, "card": {"id": "shrug_it_off", "type": "Skill", "cost": 1}},
        {"action_id": "skip", "kind": "skip_card_reward", "model_action_kind": "card_reward", "is_enabled": True},
    ]
    reward_dec = forward_decision(reward_obs, reward_legal, control_domain="macro")
    assert reward_dec is not None
    reward_worlds = synthesize_deterministic_afterstates(encoder=encoder, observation=reward_obs, decision=reward_dec)
    assert reward_worlds is not None and len(reward_worlds) == 2

    loot_obs = {
        "phase": "rewards",
        "decision_domain": "rewards",
        "run": {"active": True, "act": 1, "floor": 3},
        "player": {"character_id": "IRONCLAD", "hp": 60, "max_hp": 80, "gold": 110, "deck": _sample_deck(), "relics": [], "potions": []},
        "rewards": {"items": [{"index": 0, "type": "Card", "reward_type": "Card"}], "proceed_visible": True},
    }
    loot_legal = [
        {"action_id": "claim_c", "kind": "claim_reward", "model_action_kind": "rewards", "is_enabled": True, "reward": {"index": 0, "type": "Card"}},
        {"action_id": "leave_r", "kind": "proceed", "model_action_kind": "rewards", "is_enabled": True},
    ]
    loot_dec = forward_decision(loot_obs, loot_legal, control_domain="macro")
    assert loot_dec is not None
    assert synthesize_deterministic_afterstates(encoder=encoder, observation=loot_obs, decision=loot_dec) is None

    # 3. Shop surface: passive relic succeeds; upon-pickup relic fails closed; shop Remove strips upgrade_preview.
    shop_obs: dict[str, Any] = {
        "phase": "shop",
        "decision_domain": "shop",
        "run": {"active": True, "act": 1, "floor": 5},
        "player": {"character_id": "IRONCLAD", "hp": 60, "max_hp": 80, "gold": 300, "open_potion_slots": 2, "deck": _sample_deck(), "relics": [], "potions": []},
        "shop": {
            "visible": True,
            "is_open": True,
            "gold": 300,
            "items": [
                {"index": 0, "slot_index": 0, "category": "relic", "type": "relic", "cost": 150, "relic": {"id": "ANCHOR", "has_upon_pickup_effect": False}},
                {"index": 1, "slot_index": 1, "category": "card_removal", "type": "card_removal", "cost": 75},
            ],
        },
    }
    shop_legal = [
        {"action_id": "buy0", "kind": "shop_purchase", "model_action_kind": "shop", "is_enabled": True, "item": shop_obs["shop"]["items"][0]},
        {"action_id": "rem1", "kind": "shop_purchase", "model_action_kind": "shop", "is_enabled": True, "item": shop_obs["shop"]["items"][1]},
        {"action_id": "leave", "kind": "shop_leave", "model_action_kind": "shop", "is_enabled": True},
    ]
    shop_dec = forward_decision(shop_obs, shop_legal, control_domain="macro")
    assert shop_dec is not None
    remove_cands = [c for c in shop_dec.candidates if c.branch == "remove"]
    assert remove_cands and all("upgrade_preview" not in (c.semantic_action.get("card") or {}) for c in remove_cands)
    shop_worlds = synthesize_deterministic_afterstates(encoder=encoder, observation=shop_obs, decision=shop_dec)
    assert shop_worlds is not None and len(shop_worlds) == len(shop_dec.candidates)

    pickup_shop = copy.deepcopy(shop_obs)
    pickup_shop["shop"]["items"][0]["relic"]["has_upon_pickup_effect"] = True
    pickup_legal = copy.deepcopy(shop_legal)
    pickup_legal[0]["item"] = pickup_shop["shop"]["items"][0]
    pickup_dec = forward_decision(pickup_shop, pickup_legal, control_domain="macro")
    assert pickup_dec is not None
    assert synthesize_deterministic_afterstates(encoder=encoder, observation=pickup_shop, decision=pickup_dec) is None


def test_durable_afterstate_learner_warmup_gates_and_mixture_likelihood() -> None:
    cfg = _tiny_model_config()
    encoder = GroundedObservationEncoder(cfg.to_encoding_config())
    model = HybridDomainActorCriticModel(HybridActorCriticConfig.durable_v2(cfg.to_model_config()))
    # Make durable_value_head non-zero so afterstates have distinct values once actor activates.
    with torch.no_grad():
        for param in model.durable_value_head.parameters():
            param.normal_(mean=0.0, std=0.05)

    rest_obs, rest_legal = _rest_surface()
    rest_dec = forward_decision(rest_obs, rest_legal, control_domain="macro")
    assert rest_dec is not None
    rest_enc = encoder.encode(rest_obs, [c.semantic_action for c in rest_dec.candidates])
    grouped_cands = tuple(rest_dec.candidates[ref.representative_position] for ref in rest_enc.actions)
    post_worlds = synthesize_deterministic_afterstates(
        encoder=encoder, observation=rest_obs, decision=rest_dec, candidates=grouped_cands
    )
    assert post_worlds is not None

    map_obs = {
        "phase": "map",
        "decision_domain": "map",
        "run": {"active": True, "act": 1, "floor": 6},
        "player": {"character_id": "IRONCLAD", "hp": 56, "max_hp": 80, "gold": 150, "deck": _sample_deck(), "relics": [], "potions": []},
        "map": {"nodes": [{"x": 0, "y": 1, "type": "Monster"}, {"x": 1, "y": 1, "type": "Event"}]},
    }
    map_actions = [
        {"action_id": "m0", "kind": "choose_map_node", "model_action_kind": "map", "is_enabled": True, "node": {"x": 0, "y": 1, "type": "Monster"}},
        {"action_id": "m1", "kind": "choose_map_node", "model_action_kind": "map", "is_enabled": True, "node": {"x": 1, "y": 1, "type": "Event"}},
    ]
    map_snap = encoder.encode(map_obs, map_actions).snapshot

    combat_obs = {
        "phase": "combat",
        "decision_domain": "combat",
        "run": {"active": True, "act": 1, "floor": 7},
        "player": {"id": "ironclad", "side": "player", "hp": 56, "max_hp": 80, "energy": 3, "hand": [{"id": "strike", "cost": 1}]},
        "combat": {"in_progress": True, "enemies": [{"id": "cultist", "side": "enemy", "hp": 40, "max_hp": 48}]},
    }
    combat_actions = [
        {"action_handle": "c0", "kind": "play_card", "model_action_kind": "play_card", "is_enabled": True, "card": {"id": "strike", "cost": 1}},
        {"action_handle": "c1", "kind": "end_turn", "model_action_kind": "end_turn", "is_enabled": True},
    ]
    combat_snap = encoder.encode(combat_obs, combat_actions).snapshot

    epsilon = 0.05

    def build_episode(teacher_model: HybridDomainActorCriticModel, teacher_version: int, seed: int) -> MacroCreditTrajectory:
        teacher_model.eval()
        hidden = None
        seq = [
            (rest_enc.snapshot, "macro", "rest", grouped_cands[0].branch, 0, 0.0, False, 6, 6),
            (map_snap, "macro", "map", "Monster", 0, 0.05, False, 6, 7),
            (combat_snap, "combat", "combat", "play_card", 0, 1.5, True, 7, 7),
        ]
        steps: list[HybridVTraceStep] = []
        facts: list[MacroRewardFacts] = []
        with torch.no_grad():
            for snap, domain, surface, branch, act_idx, rew, term, fb, fa in seq:
                batch = collate_encoded_snapshots(
                    (snap,), expected_config=snap.config, expected_fingerprint=snap.encoding_fingerprint, device="cpu"
                )
                out = teacher_model(batch, hidden, validate=False)
                hidden = out.recurrent_state
                mask = torch.tensor(snap.action_mask.tolist(), dtype=torch.bool)
                logs = torch.log_softmax(out.policy_logits[0, : mask.numel()].masked_fill(~mask, -torch.inf), dim=0)
                beh_logs = macro_mixture_log_probs(logs, snap, epsilon)
                disc = 0.0 if term else (0.997 ** (fa - fb))
                steps.append(
                    HybridVTraceStep(
                        snapshot=snap,
                        action_index=act_idx,
                        behavior_log_probability=float(beh_logs[act_idx].item()),
                        reward=rew,
                        discount=disc,
                        terminal=term,
                        surface=surface,
                        branch=branch,
                        control_domain=domain,  # type: ignore[arg-type]
                        policy_version=teacher_version,
                    )
                )
                facts.append(
                    MacroRewardFacts(
                        terminal_reward=rew if term else 0.0,
                        progress_reward=0.0 if term else rew,
                        revival_penalty=0.0,
                        pace_penalty=0.0,
                        hp_loss_penalty=0.0,
                        revivals_used_delta=0,
                        player_hp_lost_delta=0.0,
                        floor_before=fb,
                        floor_after=fa,
                        native_steps=1,
                    )
                )
        node = MacroAfterstateNode(
            step_index=0,
            surface="rest",
            candidate_branches=tuple(c.branch for c in grouped_cands),
            post_worlds=post_worlds,
        )
        return MacroCreditTrajectory(
            trajectory_id=f"ep-{seed}",
            seed=seed,
            teacher_version=teacher_version,
            prefix=(),
            steps=tuple(steps),
            bootstrap_snapshot=None,
            bootstrap_domain=None,
            ending="factual_terminal",
            reward_facts=tuple(facts),
            afterstate_nodes=(node,),
            macro_branch_epsilon=epsilon,
        )

    learner = MacroCreditLearner(
        model=model,
        config=MacroCreditLearnerConfig(
            target_mode="durable_afterstate_v2",
            durable_warmup_updates=1,
            combat_actor_warmup_updates=2,
            rho_clip=1.1,
            c_clip=1.0,
            policy_rho_clip=1.1,
            chunk_steps=2,
        ),
    )

    # Update 0 -> 1: Critic calibration warmup (durable_actor_active=False, combat_actor_active=False).
    ep0 = build_episode(model, 0, seed=2)
    cohort0 = MacroCreditCohort("c0", 0, (), natural_only=(ep0,))
    m0 = learner.learn(cohort0)
    assert m0["update"] == 1
    assert m0["durable_actor_active"] is False
    assert m0["combat_actor_active"] is False
    assert m0["macro_policy_loss"] == 0.0
    assert m0["durable_value_loss"] > 0.0
    assert m0["durable_post_value_loss"] >= 0.0

    # Update 1 -> 2: Macro Actor active (MPO + fallback V-trace PG), Combat Actor still frozen!
    ep1 = build_episode(model, 1, seed=4)
    cohort1 = MacroCreditCohort("c1", 1, (), natural_only=(ep1,))
    m1 = learner.learn(cohort1)
    assert m1["update"] == 2
    assert m1["durable_actor_active"] is True
    assert m1["combat_actor_active"] is False
    assert m1["enabled_afterstate_projections"] >= 1
    assert m1["macro_mpo_policy_loss"] > 0.0

    # Update 2 -> 3: Joint Macro + Combat Actor unlocked!
    ep2 = build_episode(model, 2, seed=6)
    cohort2 = MacroCreditCohort("c2", 2, (), natural_only=(ep2,))
    m2 = learner.learn(cohort2)
    assert m2["update"] == 3
    assert m2["durable_actor_active"] is True
    assert m2["combat_actor_active"] is True
    json.dumps(m2, allow_nan=False)


def test_competence_probes_and_v6_experiment_config_load() -> None:
    cfg = _tiny_model_config()
    encoder = GroundedObservationEncoder(cfg.to_encoding_config())
    model = HybridDomainActorCriticModel(HybridActorCriticConfig.durable_v2(cfg.to_model_config())).eval()

    episodes = [
        {
            "run_result": "victory",
            "final_deck_size": 24,
            "revivals_used": 6,
            "max_floor": 51,
            "afterstate_step_count": 19,
            "macro_surface_branch_counts": {"reward:take": 9, "reward:skip": 6, "rest:rest": 2, "rest:smith": 4, "shop:remove": 2, "shop:leave": 2},
        }
    ]
    disc = summarize_episode_macro_discipline(episodes)
    assert abs(disc["card_reward_skip_rate"] - 0.4) < 1e-9
    assert disc["mean_final_deck_size"] == 24.0

    report = run_all_competence_probes(model, encoder, recent_episodes=episodes)
    assert "campfire_hp_sensitivity" in report
    assert "smith_vs_remove_divergence" in report
    json.dumps(report, allow_nan=False)

    v6_path = Path(__file__).resolve().parents[1] / "config/experiments/hybrid_macro_credit_v6_durable_afterstate_b64.toml"
    loaded = load_macro_training_config(v6_path)
    assert loaded.learner.target_mode == "durable_afterstate_v2"
    assert loaded.course.roots_per_cohort == 8
    assert loaded.course.parallel_collectors == 4
    assert loaded.course.macro_branch_epsilon == 0.05


def test_colab_drive_campaign_sync_restore_and_trace_pruning(tmp_path: Path) -> None:
    import importlib.util

    script_path = Path(__file__).resolve().parents[1] / "scripts/run_colab_g4_durable_campaign.py"
    spec = importlib.util.spec_from_file_location("run_colab_g4_durable_campaign", script_path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    local_dir = tmp_path / "local_nvme" / "campaign"
    drive_dir = tmp_path / "google_drive" / "campaign"
    restored_dir = tmp_path / "fresh_vm_nvme" / "campaign"
    local_dir.mkdir(parents=True)

    (local_dir / "checkpoint.pt").write_bytes(b"ckpt-payload-u1")
    (local_dir / "publication.pt").write_bytes(b"pub-payload-u1")
    (local_dir / "course.json").write_text('{"version": 1}\n', encoding="utf-8")
    (local_dir / "source_manifest.json").write_text('{"identity": "abc"}\n', encoding="utf-8")
    (local_dir / "status.json").write_text(
        '{"state": "idle", "committed_update": 1, "next_seed_index": 8}\n',
        encoding="utf-8",
    )
    (local_dir / "probe_metrics.jsonl").write_text('{"committed_update": 1}\n', encoding="utf-8")

    for idx in range(3):
        cdir = local_dir / "cohorts" / f"update-00000{idx}-attempt-deadbeef"
        cdir.mkdir(parents=True)
        (cdir / "collected.json").write_text('{"roots": 8}\n', encoding="utf-8")
        (cdir / "update.json").write_text('{"metrics": {"loss": 0.25}}\n', encoding="utf-8")
        (cdir / "natural-000.jsonl").write_text('{"step": 0}\n', encoding="utf-8")
        ep_sub = cdir / "episode-000"
        ep_sub.mkdir()
        (ep_sub / "episode.jsonl").write_text('{"step": 1}\n', encoding="utf-8")

    pruned = module.prune_old_raw_cohort_traces(local_dir, keep_recent_cohorts=1)
    assert pruned > 0
    assert not (local_dir / "cohorts/update-000000-attempt-deadbeef/natural-000.jsonl").exists()
    assert not (local_dir / "cohorts/update-000000-attempt-deadbeef/episode-000/episode.jsonl").exists()
    assert not (local_dir / "cohorts/update-000001-attempt-deadbeef/natural-000.jsonl").exists()
    assert (local_dir / "cohorts/update-000002-attempt-deadbeef/natural-000.jsonl").exists()
    assert (local_dir / "cohorts/update-000002-attempt-deadbeef/episode-000/episode.jsonl").exists()
    assert (local_dir / "cohorts/update-000000-attempt-deadbeef/update.json").exists()

    manifest = module.sync_campaign_to_drive(
        local_dir,
        drive_dir,
        lineage_id="durable-afterstate-v2-b64",
        snapshot_interval=5,
        latest_probe={"committed_update": 1},
    )
    assert manifest["committed_update"] == 1
    assert manifest["snapshot_saved"] is True
    assert (drive_dir / "snapshots/update-000001/checkpoint.pt").read_bytes() == b"ckpt-payload-u1"

    # Simulate second update and verify checkpoint.prev.pt backup on Drive.
    (local_dir / "checkpoint.pt").write_bytes(b"ckpt-payload-u2")
    (local_dir / "status.json").write_text(
        '{"state": "idle", "committed_update": 2, "next_seed_index": 16}\n',
        encoding="utf-8",
    )
    manifest2 = module.sync_campaign_to_drive(
        local_dir,
        drive_dir,
        lineage_id="durable-afterstate-v2-b64",
        snapshot_interval=5,
    )
    assert manifest2["committed_update"] == 2
    assert (drive_dir / "checkpoint.prev.pt").read_bytes() == b"ckpt-payload-u1"

    restored = module.restore_campaign_from_drive(drive_dir, restored_dir)
    assert restored["committed_update"] == 2
    assert (restored_dir / "checkpoint.pt").read_bytes() == b"ckpt-payload-u2"
    assert (restored_dir / "publication.pt").read_bytes() == b"pub-payload-u1"
    assert (restored_dir / "cohorts/update-000000-attempt-deadbeef/update.json").is_file()


def test_rest_smith_executor_binding_handles_deck_upgrade_preview() -> None:
    from sts2_rl.macro.authority import _card_matches

    obs, actions = _rest_surface()
    compiled = forward_decision(obs, actions, control_domain="macro")
    assert compiled is not None
    smith_cands = [c for c in compiled.candidates if c.branch == "smith"]
    assert len(smith_cands) >= 1
    cand = smith_cands[0]
    assert "upgrade_preview" in cand.semantic_action["card"]
    select_step = cand.plan[1]
    assert select_step.kind == "select_card"
    assert select_step.target is not None
    assert "upgrade_preview" not in select_step.target

    # Even if a picker action carries a simplified upgrade_preview view without source_pile/pile,
    # container-invariant binding matches cleanly.
    picker_action = {
        "kind": "select_card",
        "card": {
            "id": cand.semantic_action["card"]["id"],
            "is_upgraded": False,
            "upgrade_preview": {"id": cand.semantic_action["card"]["id"], "is_upgraded": True, "cost": 1},
        },
    }
    assert _card_matches(picker_action, select_step.target)
    assert _card_matches(picker_action, cand.semantic_action["card"])


def test_durable_cohort_recovers_from_headless_recoverable_protocol_error(
    tmp_path: Path, monkeypatch: Any
) -> None:
    import sts2_rl.joint.macro_credit_collection as mcc
    from sts2_rl.backends.headless import HeadlessRecoverableProtocolError
    from sts2_rl.macro_branch_diagnostic import BranchLimits

    cfg = _tiny_model_config()
    encoder = GroundedObservationEncoder(cfg.to_encoding_config())
    combat_obs = {
        "phase": "combat",
        "decision_domain": "combat",
        "run": {"active": True, "act": 1, "floor": 1},
        "player": {"id": "ironclad", "side": "player", "hp": 60, "max_hp": 80, "energy": 3, "hand": [{"id": "strike", "cost": 1}]},
        "combat": {"in_progress": True, "enemies": [{"id": "cultist", "side": "enemy", "hp": 40, "max_hp": 48}]},
    }
    combat_actions = [
        {"action_handle": "c0", "kind": "play_card", "model_action_kind": "play_card", "is_enabled": True, "card": {"id": "strike", "cost": 1}},
    ]
    snap = encoder.encode(combat_obs, combat_actions).snapshot

    def fake_collect_episode(*, seed: int, teacher_version: int, output_dir: Path, **_: Any) -> tuple[MacroCreditTrajectory, dict[str, Any]]:
        output_dir.mkdir(parents=True, exist_ok=False)
        if seed == 104:
            raise HeadlessRecoverableProtocolError(
                "simulated AMALGAMATOR event option rejection",
                operation="step",
                incident_kind="step_rejected",
                fingerprint="fp-amalgamator",
                quarantine_path=None,
            )
        step = HybridVTraceStep(
            snapshot=snap,
            action_index=0,
            behavior_log_probability=0.0,
            reward=1.0,
            discount=0.0,
            terminal=True,
            surface="combat",
            branch="play_card",
            control_domain="combat",
            policy_version=teacher_version,
        )
        fact = MacroRewardFacts(
            terminal_reward=1.0,
            progress_reward=0.0,
            revival_penalty=0.0,
            pace_penalty=0.0,
            hp_loss_penalty=0.0,
            revivals_used_delta=0,
            player_hp_lost_delta=0.0,
            floor_before=1,
            floor_after=1,
            native_steps=1,
        )
        traj = MacroCreditTrajectory(
            trajectory_id=f"ep-{seed}",
            seed=seed,
            teacher_version=teacher_version,
            prefix=(),
            steps=(step,),
            bootstrap_snapshot=None,
            bootstrap_domain=None,
            ending="factual_terminal",
            reward_facts=(fact,),
        )
        return traj, {"status": "complete", "seed": seed, "training_episode": {"status": "complete", "seed": seed}}

    monkeypatch.setattr(mcc, "collect_durable_afterstate_episode", fake_collect_episode)
    v6_path = Path(__file__).resolve().parents[1] / "config/experiments/hybrid_macro_credit_v6_durable_afterstate_b64.toml"
    tcfg = load_macro_training_config(v6_path).training
    limits = BranchLimits(terminal_only=True)
    consumed: list[int] = []

    def repl(offset: int) -> int:
        consumed.append(offset)
        return 110 + offset * 2

    cohort, episodes, reports = mcc.collect_durable_afterstate_cohort(
        backend_factory=lambda: None,  # type: ignore[arg-type,return-value]
        policy=None,  # type: ignore[arg-type]
        config=tcfg,
        seeds=[102, 104],
        teacher_version=0,
        limits=limits,
        cohort_dir=tmp_path / "cohort-ok",
        cohort_id="cohort-ok",
        parallel_collectors=2,
        replacement_seed_fn=repl,
        max_recoverable_incidents=2,
        artifact_root=tmp_path,
    )
    assert [t.seed for t in cohort.natural_only] == [102, 114]
    assert consumed == [2]
    assert len(episodes) == 2
    assert len(reports[0]["recoverable_incidents"]) == 1


def test_macro_competence_probes_cli_main_on_durable_publication(tmp_path: Path, monkeypatch: Any) -> None:
    from sts2_baseline import act_frontloaded_death_reward_identity
    from sts2_rl.joint.macro_checkpoint import (
        MacroCheckpointIdentity,
        publish_macro_checkpoint,
        save_macro_checkpoint,
    )
    from sts2_rl.joint.macro_credit_learner import MACRO_CREDIT_LEARNER_VERSION
    from sts2_rl.macro_competence_probes import main as probes_main

    monkeypatch.setenv("STS2_ARTIFACT_ROOT", str(tmp_path))
    v6_path = Path(__file__).resolve().parents[1] / "config/experiments/hybrid_macro_credit_v6_durable_afterstate_b64.toml"
    loaded_cfg = load_macro_training_config(v6_path)
    ac_cfg = HybridActorCriticConfig.durable_v2(loaded_cfg.training.model.to_model_config())
    model = HybridDomainActorCriticModel(ac_cfg)
    opt = torch.optim.AdamW(model.parameters(), lr=1e-4)
    identity = MacroCheckpointIdentity(
        course_config=loaded_cfg.as_mapping(),
        model_config=ac_cfg.as_mapping(),
        reward_identity=act_frontloaded_death_reward_identity(revival_budget=64),
        lineage_id="test-probe-cli",
        learner_version=MACRO_CREDIT_LEARNER_VERSION,
    )
    ckpt_path = tmp_path / "checkpoint.pt"
    pub_path = tmp_path / "publication.pt"
    save_macro_checkpoint(
        ckpt_path,
        identity=identity,
        model=model,
        optimizer=opt,
        update=1,
        next_seed_index=8,
        initialization={"mode": "test"},
        artifact_root=tmp_path,
    )
    publish_macro_checkpoint(ckpt_path, pub_path, identity=identity, artifact_root=tmp_path)
    rc = probes_main(["--publication", str(pub_path), "--device", "cpu"])
    assert rc == 0

    import importlib.util
    import os

    script_path = Path(__file__).resolve().parents[1] / "scripts/run_colab_g4_durable_campaign.py"
    spec = importlib.util.spec_from_file_location("run_colab_g4_durable_campaign", script_path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    (tmp_path / "status.json").write_text('{"committed_update": 1, "next_seed_index": 8}\n', encoding="utf-8")
    drive_out = tmp_path / "drive_out"
    manifest = module._probe_prune_and_sync(
        output_dir=tmp_path,
        drive_dir=drive_out,
        lineage_id="test-probe-cli",
        committed_after=1,
        train_summary={"ok": True},
        device="cpu",
        env=dict(os.environ),
        snapshot_interval=5,
        keep_recent_cohorts=2,
    )
    assert manifest["committed_update"] == 1
    assert (drive_out / "sync_manifest.json").is_file()


def test_durable_resume_accepts_replacement_seed_cursor_offset(tmp_path: Path, monkeypatch: Any) -> None:
    import sts2_rl.train_macro_credit as tmc
    from sts2_baseline import act_frontloaded_death_reward_identity
    from sts2_rl.joint.macro_checkpoint import MacroCheckpointIdentity
    from sts2_rl.joint.macro_credit_learner import MACRO_CREDIT_LEARNER_VERSION

    monkeypatch.setenv("STS2_ARTIFACT_ROOT", str(tmp_path))
    v6_path = Path(__file__).resolve().parents[1] / "config/experiments/hybrid_macro_credit_v6_durable_afterstate_b64.toml"
    cfg = load_macro_training_config(v6_path, overrides=("course.roots_per_cohort=2",))
    ac_cfg = HybridActorCriticConfig.durable_v2(_tiny_model_config().to_model_config())
    model = HybridDomainActorCriticModel(ac_cfg)
    learner = MacroCreditLearner(model=model, config=cfg.learner)
    learner.update = 1  # Simulate resuming after update=1 consumed 1 replacement seed (cursor = 3 instead of 2).
    identity = MacroCheckpointIdentity(
        course_config=cfg.as_mapping(),
        model_config=ac_cfg.as_mapping(),
        reward_identity=act_frontloaded_death_reward_identity(revival_budget=64),
        lineage_id="test-resume-cursor",
        learner_version=MACRO_CREDIT_LEARNER_VERSION,
    )

    encoder = GroundedObservationEncoder(_tiny_model_config().to_encoding_config())
    combat_obs = {
        "phase": "combat",
        "decision_domain": "combat",
        "run": {"active": True, "act": 1, "floor": 1},
        "player": {"id": "ironclad", "side": "player", "hp": 60, "max_hp": 80, "energy": 3, "hand": [{"id": "strike", "cost": 1}]},
        "combat": {"in_progress": True, "enemies": [{"id": "cultist", "side": "enemy", "hp": 40, "max_hp": 48}]},
    }
    combat_actions = [
        {"action_handle": "c0", "kind": "play_card", "model_action_kind": "play_card", "is_enabled": True, "card": {"id": "strike", "cost": 1}},
    ]
    snap = encoder.encode(combat_obs, combat_actions).snapshot

    def fake_collect_cohort(*, seeds: Any, teacher_version: int, cohort_id: str, **_: Any) -> Any:
        trajs = []
        eps = []
        for s in seeds:
            step = HybridVTraceStep(snap, 0, 0.0, 1.0, 0.0, True, "combat", "play_card", "combat", teacher_version)
            fact = MacroRewardFacts(1.0, 0.0, 0.0, 0.0, 0.0, 0, 0.0, 1, 1, 1)
            trajs.append(MacroCreditTrajectory(f"t-{s}", s, teacher_version, (), (step,), None, None, "factual_terminal", (fact,)))
            eps.append({"status": "complete", "seed": s})
        return MacroCreditCohort(cohort_id, teacher_version, (), natural_only=tuple(trajs)), eps, []

    monkeypatch.setattr(tmc, "collect_durable_afterstate_cohort", fake_collect_cohort)
    out_dir = tmp_path / "resume_out"
    out_dir.mkdir()
    from sts2_rl.joint.macro_checkpoint import save_macro_checkpoint

    save_macro_checkpoint(
        out_dir / "checkpoint.pt",
        identity=identity,
        model=model,
        optimizer=learner.optimizer,
        update=1,
        next_seed_index=3,
        initialization={"mode": "test"},
        artifact_root=tmp_path,
    )
    summary = tmc.run_cohorts(
        config=cfg,
        output=out_dir,
        identity=identity,
        learner=learner,
        teacher=None,  # type: ignore[arg-type]
        backend_factory=lambda: None,  # type: ignore[arg-type,return-value]
        initial_seed_index=3,
        initialization={"mode": "test"},
        max_updates=1,
        initialize_boundary=False,
        root=tmp_path,
    )
    assert summary["committed_update"] == 2
    assert summary["next_seed_index"] == 5


def test_durable_batched_learner_matches_sequential_losses_and_parameters() -> None:
    torch.manual_seed(1234)
    cfg = _tiny_model_config()
    encoder = GroundedObservationEncoder(cfg.to_encoding_config())
    base_model = HybridDomainActorCriticModel(
        HybridActorCriticConfig.durable_v2(cfg.to_model_config())
    ).eval()
    with torch.no_grad():
        for param in base_model.durable_value_head.parameters():
            param.normal_(mean=0.0, std=0.05)

    rest_obs, rest_actions = _rest_surface()
    compiled_rest = forward_decision(rest_obs, rest_actions, control_domain="macro")
    assert compiled_rest is not None
    rest_enc = encoder.encode(rest_obs, [c.semantic_action for c in compiled_rest.candidates])
    grouped_cands = tuple(
        compiled_rest.candidates[ref.representative_position] for ref in rest_enc.actions
    )
    post_worlds = synthesize_deterministic_afterstates(
        encoder=encoder,
        observation=rest_obs,
        decision=compiled_rest,
        candidates=grouped_cands,
    )
    assert post_worlds is not None

    map_obs = {
        "phase": "map",
        "decision_domain": "map",
        "run": {"active": True, "act": 1, "floor": 6},
        "player": {"character_id": "IRONCLAD", "hp": 56, "max_hp": 80, "gold": 150, "deck": _sample_deck(), "relics": [], "potions": []},
        "map": {"nodes": [{"x": 0, "y": 1, "type": "Monster"}, {"x": 1, "y": 1, "type": "Event"}]},
    }
    map_actions = [
        {"action_id": "m0", "kind": "choose_map_node", "model_action_kind": "map", "is_enabled": True, "node": {"x": 0, "y": 1, "type": "Monster"}},
        {"action_id": "m1", "kind": "choose_map_node", "model_action_kind": "map", "is_enabled": True, "node": {"x": 1, "y": 1, "type": "Event"}},
    ]
    map_snap = encoder.encode(map_obs, map_actions).snapshot

    combat_obs = {
        "phase": "combat",
        "decision_domain": "combat",
        "run": {"active": True, "act": 1, "floor": 7},
        "player": {"id": "ironclad", "side": "player", "hp": 56, "max_hp": 80, "energy": 3, "hand": [{"id": "strike", "cost": 1}]},
        "combat": {"in_progress": True, "enemies": [{"id": "cultist", "side": "enemy", "hp": 40, "max_hp": 48}]},
    }
    combat_actions = [
        {"action_handle": "c0", "kind": "play_card", "model_action_kind": "play_card", "is_enabled": True, "card": {"id": "strike", "cost": 1}},
        {"action_handle": "c1", "kind": "end_turn", "model_action_kind": "end_turn", "is_enabled": True},
    ]
    combat_snap = encoder.encode(combat_obs, combat_actions).snapshot
    epsilon = 0.05

    def make_traj(seed: int, extra_combat_steps: int) -> MacroCreditTrajectory:
        hidden = None
        seq = [
            (rest_enc.snapshot, "macro", "rest", grouped_cands[0].branch, 0, 0.0, False, 6, 6),
            (map_snap, "macro", "map", "Monster", 1, 0.1, False, 6, 7),
        ]
        for k_step in range(extra_combat_steps):
            is_last = k_step == extra_combat_steps - 1
            seq.append(
                (combat_snap, "combat", "combat", "play_card", k_step % 2, 1.2 if is_last else -0.05, is_last, 7, 7)
            )
        steps: list[HybridVTraceStep] = []
        facts: list[MacroRewardFacts] = []
        with torch.no_grad():
            for snap, domain, surface, branch, act_idx, rew, term, fb, fa in seq:
                batch = collate_encoded_snapshots(
                    (snap,), expected_config=snap.config, expected_fingerprint=snap.encoding_fingerprint, device="cpu"
                )
                out = base_model(batch, hidden, validate=False)
                hidden = out.recurrent_state
                mask = torch.tensor(snap.action_mask.tolist(), dtype=torch.bool)
                logs = torch.log_softmax(out.policy_logits[0, : mask.numel()].masked_fill(~mask, -torch.inf), dim=0)
                beh_logs = macro_mixture_log_probs(logs, snap, epsilon)
                disc = 0.0 if term else (0.997 ** (fa - fb))
                steps.append(
                    HybridVTraceStep(
                        snapshot=snap,
                        action_index=act_idx,
                        behavior_log_probability=float(beh_logs[act_idx].item()),
                        reward=rew,
                        discount=disc,
                        terminal=term,
                        surface=surface,
                        branch=branch,
                        control_domain=domain,  # type: ignore[arg-type]
                        policy_version=2,
                    )
                )
                facts.append(
                    MacroRewardFacts(
                        terminal_reward=rew if term else 0.0,
                        progress_reward=0.0 if term else rew,
                        revival_penalty=0.0,
                        pace_penalty=0.0,
                        hp_loss_penalty=0.0,
                        revivals_used_delta=0,
                        player_hp_lost_delta=0.0,
                        floor_before=fb,
                        floor_after=fa,
                        native_steps=1,
                    )
                )
        node = MacroAfterstateNode(
            step_index=0,
            surface="rest",
            candidate_branches=tuple(c.branch for c in grouped_cands),
            post_worlds=post_worlds,
        )
        return MacroCreditTrajectory(
            trajectory_id=f"ep-{seed}",
            seed=seed,
            teacher_version=2,
            prefix=(),
            steps=tuple(steps),
            bootstrap_snapshot=None,
            bootstrap_domain=None,
            ending="factual_terminal",
            reward_facts=tuple(facts),
            afterstate_nodes=(node,),
            macro_branch_epsilon=epsilon,
        )

    cohort = MacroCreditCohort(
        "batch-eq-cohort",
        2,
        (),
        natural_only=(make_traj(2, 1), make_traj(4, 3), make_traj(6, 2)),
    )

    model_seq = HybridDomainActorCriticModel(HybridActorCriticConfig.durable_v2(cfg.to_model_config()))
    model_seq.load_state_dict(base_model.state_dict())
    model_batch = HybridDomainActorCriticModel(HybridActorCriticConfig.durable_v2(cfg.to_model_config()))
    model_batch.load_state_dict(base_model.state_dict())

    common_kwargs: dict[str, Any] = {
        "target_mode": "durable_afterstate_v2",
        "durable_warmup_updates": 1,
        "combat_actor_warmup_updates": 2,
        "rho_clip": 1.1,
        "c_clip": 1.0,
        "policy_rho_clip": 1.1,
        "chunk_steps": 2,
        "likelihood_tolerance": 1e-4,
    }
    learner_seq = MacroCreditLearner(
        model=model_seq,
        update=2,
        config=MacroCreditLearnerConfig(**common_kwargs, eval_batch_size=1, chunk_batch_size=1),
    )
    learner_batch = MacroCreditLearner(
        model=model_batch,
        update=2,
        config=MacroCreditLearnerConfig(**common_kwargs, eval_batch_size=4, chunk_batch_size=4),
    )

    m_seq = learner_seq.learn(cohort)
    m_batch = learner_batch.learn(cohort)

    for key in (
        "macro_policy_loss",
        "macro_mpo_policy_loss",
        "macro_pg_policy_loss",
        "macro_entropy",
        "macro_value_loss",
        "durable_value_loss",
        "durable_post_value_loss",
        "combat_policy_loss",
        "combat_value_loss",
        "combat_entropy",
        "backward_chunks",
        "gradient_norm",
        "parameter_delta_norm",
        "target_forward_calls",
    ):
        assert abs(float(m_seq[key]) - float(m_batch[key])) < 1e-5, f"mismatch on {key}: {m_seq[key]} vs {m_batch[key]}"

    for (name, p_seq), (_, p_batch) in zip(
        model_seq.named_parameters(), model_batch.named_parameters(), strict=True
    ):
        assert p_seq.grad is not None and p_batch.grad is not None
        assert torch.allclose(p_seq.grad, p_batch.grad, atol=1e-6, rtol=1e-4), (
            f"gradient {name} diverged: {p_seq.grad} vs {p_batch.grad}"
        )
        if name in {"candidate_policy_head.2.bias", "branch_policy_head.3.bias"}:
            # Softmax is translation-invariant to a constant scalar logit offset, so its analytical
            # gradient is 0 and float32 O(1e-8) rounding noise gets normalized by Adam(eps=1e-8).
            assert float(p_seq.grad.abs().max().item()) < 1e-6
            assert float(p_batch.grad.abs().max().item()) < 1e-6
            assert torch.allclose(p_seq, p_batch, atol=4e-4)
        else:
            assert torch.allclose(p_seq, p_batch, atol=1e-5, rtol=1e-4), f"parameter {name} diverged"


def test_batched_policy_inference_server_concurrent_and_ipc() -> None:
    from concurrent.futures import ThreadPoolExecutor
    from multiprocessing import get_context

    from sts2_rl.joint.parallel_collection import (
        BatchedPolicyInferenceServer,
        RemoteBatchedFrozenPolicy,
    )

    torch.manual_seed(77)
    cfg = _tiny_model_config()
    encoder = GroundedObservationEncoder(cfg.to_encoding_config())
    model = HybridDomainActorCriticModel(HybridActorCriticConfig.durable_v2(cfg.to_model_config())).eval()

    rest_obs, rest_actions = _rest_surface()
    snap_a = encoder.encode(rest_obs, rest_actions).snapshot
    combat_obs = {
        "phase": "combat",
        "decision_domain": "combat",
        "run": {"active": True, "act": 1, "floor": 1},
        "player": {"id": "ironclad", "side": "player", "hp": 60, "max_hp": 80, "energy": 3, "hand": [{"id": "strike", "cost": 1}]},
        "combat": {"in_progress": True, "enemies": [{"id": "cultist", "side": "enemy", "hp": 40, "max_hp": 48}]},
    }
    combat_actions = [
        {"action_handle": "c0", "kind": "play_card", "model_action_kind": "play_card", "is_enabled": True, "card": {"id": "strike", "cost": 1}},
        {"action_handle": "c1", "kind": "end_turn", "model_action_kind": "end_turn", "is_enabled": True},
    ]
    snap_b = encoder.encode(combat_obs, combat_actions).snapshot

    ctx = get_context("spawn")
    req_q = ctx.Queue()
    recv_conn, send_conn = ctx.Pipe(duplex=False)
    try:
        with BatchedPolicyInferenceServer(
            model=model,
            encoding_config=cfg.to_encoding_config(),
            device=torch.device("cpu"),
            worker_send_conns=(send_conn,),
            mp_request_queue=req_q,
            max_batch_size=8,
            linger_seconds=0.005,
        ) as server:
            remote = RemoteBatchedFrozenPolicy(
                worker_id=0,
                request_queue=req_q,
                response_conn=recv_conn,
                recurrent_hidden_dim=model.config.recurrent_hidden_dim,
                hierarchical_branch_greedy=True,
            )

            with ThreadPoolExecutor(max_workers=4) as pool:
                futs = [
                    pool.submit(server.forward, snap_a if i % 2 == 0 else snap_b, None)
                    for i in range(6)
                ]
                futs.append(pool.submit(remote.forward, snap_a, remote.initial_state()))
                results = [f.result() for f in futs]

            with torch.no_grad():
                ref_batch_a = collate_encoded_snapshots(
                    (snap_a,), expected_config=snap_a.config, expected_fingerprint=snap_a.encoding_fingerprint, device="cpu"
                )
                ref_out_a = model(ref_batch_a, None, validate=False)
            assert torch.allclose(results[0][0], ref_out_a.policy_logits[0], atol=1e-4)
            assert torch.allclose(results[-1][0], ref_out_a.policy_logits[0], atol=1e-4)
            assert server.total_requests == 7
    finally:
        recv_conn.close()
        send_conn.close()


def test_migrate_macro_checkpoint_runtime_preserves_optimizer_and_supports_scaled_cohort(
    tmp_path: Path, monkeypatch: Any
) -> None:
    import sts2_rl.train_macro_credit as tmc
    from sts2_baseline import act_frontloaded_death_reward_identity
    from sts2_rl.encoding import grounding_encoding_identity
    from sts2_rl.joint.macro_checkpoint import (
        MacroCheckpointIdentity,
        load_macro_checkpoint,
        migrate_macro_checkpoint_runtime,
        save_macro_checkpoint,
    )
    from sts2_rl.joint.macro_credit_data import MACRO_CREDIT_DATA_VERSION
    from sts2_rl.joint.macro_credit_learner import MACRO_CREDIT_LEARNER_VERSION
    from sts2_rl.semantics.clock import DECISION_CLOCK_CONTRACT_VERSION
    from sts2_rl.semantics.forward import FORWARD_COMPILER_CONTRACT_VERSION

    monkeypatch.setenv("STS2_ARTIFACT_ROOT", str(tmp_path))
    v6_base = Path(__file__).resolve().parents[1] / "config/experiments/hybrid_macro_credit_v6_durable_afterstate_b64.toml"
    v6_turbo = Path(__file__).resolve().parents[1] / "config/experiments/hybrid_macro_credit_v6_durable_afterstate_b64_g4_turbo.toml"
    old_cfg = load_macro_training_config(v6_base)
    new_cfg = load_macro_training_config(v6_turbo)
    assert new_cfg.course.roots_per_cohort == 48
    assert new_cfg.course.parallel_collectors == 48
    assert new_cfg.learner.chunk_batch_size == 96
    assert new_cfg.learner.allow_tf32 is True

    ac_cfg = HybridActorCriticConfig.durable_v2(_tiny_model_config().to_model_config())
    old_model = HybridDomainActorCriticModel(ac_cfg)
    old_learner = MacroCreditLearner(model=old_model, config=old_cfg.learner, update=15)
    # Populate real AdamW moments before saving checkpoint.
    dummy_loss = sum(p.square().sum() for p in old_model.parameters())
    dummy_loss.backward()  # type: ignore[union-attr]
    old_learner.optimizer.step()
    old_learner.optimizer.zero_grad(set_to_none=True)

    def make_course_id(c: Any) -> dict[str, Any]:
        return {
            **c.as_mapping(),
            "encoding": grounding_encoding_identity(),
            "compiler": FORWARD_COMPILER_CONTRACT_VERSION,
            "decision_clock": DECISION_CLOCK_CONTRACT_VERSION,
            "data": MACRO_CREDIT_DATA_VERSION,
        }

    old_identity = MacroCheckpointIdentity(
        course_config=make_course_id(old_cfg),
        model_config=ac_cfg.as_mapping(),
        reward_identity=act_frontloaded_death_reward_identity(revival_budget=64),
        lineage_id="durable-afterstate-v2-b64-g4",
        learner_version=MACRO_CREDIT_LEARNER_VERSION,
    )
    backup_dir = tmp_path / "backup_u15"
    backup_dir.mkdir()
    backup_ckpt = backup_dir / "checkpoint.pt"
    save_macro_checkpoint(
        backup_ckpt,
        identity=old_identity,
        model=old_model,
        optimizer=old_learner.optimizer,
        update=15,
        next_seed_index=128,
        initialization={"mode": "model_only_new_macro_lineage"},
        artifact_root=tmp_path,
    )

    new_identity = MacroCheckpointIdentity(
        course_config=make_course_id(new_cfg),
        model_config=ac_cfg.as_mapping(),
        reward_identity=act_frontloaded_death_reward_identity(revival_budget=64),
        lineage_id="durable-afterstate-v2-b64-g4",
        learner_version=MACRO_CREDIT_LEARNER_VERSION,
    )
    new_model = HybridDomainActorCriticModel(ac_cfg)
    new_learner = MacroCreditLearner(model=new_model, config=new_cfg.learner)
    migrated = migrate_macro_checkpoint_runtime(
        backup_ckpt,
        identity=new_identity,
        model=new_model,
        optimizer=new_learner.optimizer,
        artifact_root=tmp_path,
    )
    assert migrated["update"] == 15
    assert migrated["next_seed_index"] == 128
    for p_old, p_new in zip(old_model.parameters(), new_model.parameters(), strict=True):
        assert torch.equal(p_old, p_new)

    new_learner.update = 15
    init_meta = {
        "mode": "runtime_migration_from_macro_checkpoint",
        "source_path": str(backup_ckpt),
        "update_base_at_migration": 15,
        "cursor_base_at_migration": 128,
    }
    encoder = GroundedObservationEncoder(_tiny_model_config().to_encoding_config())
    combat_obs = {
        "phase": "combat",
        "decision_domain": "combat",
        "run": {"active": True, "act": 1, "floor": 1},
        "player": {"id": "ironclad", "side": "player", "hp": 60, "max_hp": 80, "energy": 3, "hand": [{"id": "strike", "cost": 1}]},
        "combat": {"in_progress": True, "enemies": [{"id": "cultist", "side": "enemy", "hp": 40, "max_hp": 48}]},
    }
    combat_actions = [
        {"action_handle": "c0", "kind": "play_card", "model_action_kind": "play_card", "is_enabled": True, "card": {"id": "strike", "cost": 1}},
    ]
    snap = encoder.encode(combat_obs, combat_actions).snapshot

    def fake_collect_cohort(*, seeds: Any, teacher_version: int, cohort_id: str, **_: Any) -> Any:
        trajs = []
        eps = []
        for s in seeds:
            step = HybridVTraceStep(snap, 0, 0.0, 1.0, 0.0, True, "combat", "play_card", "combat", teacher_version)
            fact = MacroRewardFacts(1.0, 0.0, 0.0, 0.0, 0.0, 0, 0.0, 1, 1, 1)
            trajs.append(MacroCreditTrajectory(f"t-{s}", s, teacher_version, (), (step,), None, None, "factual_terminal", (fact,)))
            eps.append({"status": "complete", "seed": s})
        return MacroCreditCohort(cohort_id, teacher_version, (), natural_only=tuple(trajs)), eps, []

    monkeypatch.setattr(tmc, "collect_durable_afterstate_cohort", fake_collect_cohort)
    turbo_dir = tmp_path / "turbo_out"
    turbo_dir.mkdir()
    summary = tmc.run_cohorts(
        config=new_cfg,
        output=turbo_dir,
        identity=new_identity,
        learner=new_learner,
        teacher=None,  # type: ignore[arg-type]
        backend_factory=lambda: None,  # type: ignore[arg-type,return-value]
        initial_seed_index=128,
        initialization=init_meta,
        max_updates=1,
        initialize_boundary=True,
        root=tmp_path,
        source_paths=(backup_ckpt,),
    )
    assert summary["committed_update"] == 16
    assert summary["next_seed_index"] == 128 + 48

    # Verify exact --resume loads the migrated checkpoint cleanly on the next chunk!
    resumed_model = HybridDomainActorCriticModel(ac_cfg)
    resumed_learner = MacroCreditLearner(model=resumed_model, config=new_cfg.learner)
    resumed_payload = load_macro_checkpoint(
        turbo_dir / "checkpoint.pt",
        identity=new_identity,
        model=resumed_model,
        optimizer=resumed_learner.optimizer,
        artifact_root=tmp_path,
    )
    assert resumed_payload["update"] == 16
    assert resumed_payload["next_seed_index"] == 176


def test_scatter_collation_matches_reference_bit_for_bit() -> None:
    from sts2_rl.encoding.snapshot import (
        _collate_encoded_snapshots_scatter,
        _collate_encoded_worlds_scatter,
    )

    cfg = _tiny_model_config()
    enc_cfg = cfg.to_encoding_config()
    encoder = GroundedObservationEncoder(enc_cfg)

    rest_obs, rest_actions = _rest_surface()
    compiled_rest = forward_decision(rest_obs, rest_actions, control_domain="macro")
    assert compiled_rest is not None
    rest_enc = encoder.encode(rest_obs, [c.semantic_action for c in compiled_rest.candidates])
    grouped_cands = tuple(
        compiled_rest.candidates[ref.representative_position] for ref in rest_enc.actions
    )
    post_worlds = synthesize_deterministic_afterstates(
        encoder=encoder,
        observation=rest_obs,
        decision=compiled_rest,
        candidates=grouped_cands,
    )
    assert post_worlds is not None

    combat_obs = {
        "phase": "combat",
        "decision_domain": "combat",
        "run": {"active": True, "act": 1, "floor": 7},
        "player": {"id": "ironclad", "side": "player", "hp": 56, "max_hp": 80, "energy": 3, "hand": [{"id": "strike", "cost": 1}]},
        "combat": {"in_progress": True, "enemies": [{"id": "cultist", "side": "enemy", "hp": 40, "max_hp": 48}]},
    }
    combat_actions = [
        {"action_handle": "c0", "kind": "play_card", "model_action_kind": "play_card", "is_enabled": True, "card": {"id": "strike", "cost": 1}},
        {"action_handle": "c1", "kind": "end_turn", "model_action_kind": "end_turn", "is_enabled": True},
    ]
    combat_snap = encoder.encode(combat_obs, combat_actions).snapshot

    # < 8 items triggers sequential reference in collate_encoded_snapshots(device="cpu")
    snaps = (rest_enc.snapshot, combat_snap, rest_enc.snapshot)
    ref_batch = collate_encoded_snapshots(
        snaps,
        expected_config=enc_cfg,
        expected_fingerprint=combat_snap.encoding_fingerprint,
        device="cpu",
    )
    scat_batch = _collate_encoded_snapshots_scatter(
        snaps,
        cfg=enc_cfg,
        dev=torch.device("cpu"),
    )
    for field_name in ref_batch.world.__dataclass_fields__:
        assert torch.equal(getattr(ref_batch.world, field_name), getattr(scat_batch.world, field_name)), field_name
    for field_name in ref_batch.candidates.__dataclass_fields__:
        assert torch.equal(
            getattr(ref_batch.candidates, field_name),
            getattr(scat_batch.candidates, field_name),
        ), field_name
    assert torch.equal(ref_batch.domain_ids, scat_batch.domain_ids)
    assert torch.equal(ref_batch.macro_economic_surface_ids, scat_batch.macro_economic_surface_ids)

    ref_worlds, ref_w_dom = collate_encoded_worlds(
        post_worlds,
        expected_config=enc_cfg,
        expected_fingerprint=post_worlds[0].encoding_fingerprint,
        device="cpu",
    )
    scat_worlds, scat_w_dom = _collate_encoded_worlds_scatter(
        post_worlds,
        cfg=enc_cfg,
        dev=torch.device("cpu"),
    )
    for field_name in ref_worlds.__dataclass_fields__:
        assert torch.equal(getattr(ref_worlds, field_name), getattr(scat_worlds, field_name)), field_name
    assert torch.equal(ref_w_dom, scat_w_dom)


def test_durable_v3_direct_vitals_and_cardinality_neutral_macro_branch_prior() -> None:
    torch.manual_seed(17)
    cfg = _tiny_model_config()
    grounded_cfg = cfg.to_model_config()
    encoder = GroundedObservationEncoder(cfg.to_encoding_config())

    # 1. Config serialization round-trip preserves durable_v2 legacy ABI and durable_v3 flags.
    v2_cfg = HybridActorCriticConfig.durable_v2(grounded_cfg)
    assert "enable_direct_vitals_context" not in v2_cfg.as_mapping()
    assert HybridActorCriticConfig.from_mapping(v2_cfg.as_mapping(), fallback_grounded=grounded_cfg) == v2_cfg

    v3_cfg = HybridActorCriticConfig.durable_v3(grounded_cfg)
    assert v3_cfg.as_mapping()["enable_direct_vitals_context"] is True
    assert v3_cfg.as_mapping()["enable_macro_branch_prior"] is True
    assert HybridActorCriticConfig.from_mapping(v3_cfg.as_mapping(), fallback_grounded=grounded_cfg) == v3_cfg

    base_model = HybridDomainActorCriticModel(HybridActorCriticConfig(grounded_cfg)).eval()
    v3_model = HybridDomainActorCriticModel(v3_cfg).eval()
    initialize_actor_critic_from_iql(v3_model, base_model.state_dict())

    # 2. Step-0 Combat Equivalence holds strictly on durable_v3.
    combat_obs = {
        "phase": "combat",
        "decision_domain": "combat",
        "player": {"id": "ironclad", "side": "player", "hp": 44, "max_hp": 80, "energy": 3, "hand": [{"id": "strike", "cost": 1}]},
        "combat": {"in_progress": True, "enemies": [{"id": "cultist", "side": "enemy", "hp": 40, "max_hp": 48}]},
    }
    combat_actions = [
        {"action_handle": "c0", "kind": "play_card", "model_action_kind": "play_card", "is_enabled": True, "card": {"id": "strike", "cost": 1}},
        {"action_handle": "c1", "kind": "end_turn", "model_action_kind": "end_turn", "is_enabled": True},
    ]
    combat_enc = encoder.encode(combat_obs, combat_actions)
    with torch.no_grad():
        p_v1 = torch.softmax(base_model(combat_enc.batch).policy_logits[0], dim=0)
        p_v3 = torch.softmax(v3_model(combat_enc.batch).policy_logits[0], dim=0)
    torch.testing.assert_close(p_v3, p_v1, rtol=1e-5, atol=1e-6)

    # 3. Cardinality-neutral log-mean-exp macro branch prior:
    # Equal candidate logits across 1 Rest vs 6 Smith yield exact 0.5 / 0.5 branch mass;
    # elevated singleton Skip logit yields p(Skip) > 0.70 rather than resetting to 0.50.
    rest_obs, rest_legal = _rest_surface()
    rest_dec = forward_decision(rest_obs, rest_legal, control_domain="macro")
    assert rest_dec is not None and len(rest_dec.candidates) == 7
    rest_enc = encoder.encode(rest_obs, [c.semantic_action for c in rest_dec.candidates])
    with torch.no_grad():
        uniform_raw = torch.full((1, rest_enc.batch.candidates.action_mask.shape[1]), 1.25)
        eq_logits = v3_model._hierarchical_policy_logits(
            raw_candidate_logits=uniform_raw,
            candidate_embeddings=torch.zeros((1, uniform_raw.shape[1], grounded_cfg.d_model)),
            joint_state_embedding=torch.zeros((1, grounded_cfg.d_model)),
            vitals_embedding=torch.zeros((1, grounded_cfg.d_model)),
            recurrent_state=torch.zeros((1, grounded_cfg.recurrent_hidden_dim)),
            policy_branch_ids=rest_enc.batch.candidates.policy_branch_ids,
            action_mask=rest_enc.batch.candidates.action_mask,
            domain_ids=rest_enc.batch.domain_ids,
        )
        eq_probs = torch.softmax(eq_logits[0], dim=0)
        p_rest_eq = sum(
            float(eq_probs[idx].item())
            for idx, ref in enumerate(rest_enc.actions)
            if rest_dec.candidates[ref.representative_position].branch == "rest"
        )
        assert abs(p_rest_eq - 0.5) < 1e-6

        prior_raw = torch.zeros_like(uniform_raw)
        for idx, ref in enumerate(rest_enc.actions):
            if rest_dec.candidates[ref.representative_position].branch == "rest":
                prior_raw[0, idx] = 1.5
        skew_logits = v3_model._hierarchical_policy_logits(
            raw_candidate_logits=prior_raw,
            candidate_embeddings=torch.zeros((1, uniform_raw.shape[1], grounded_cfg.d_model)),
            joint_state_embedding=torch.zeros((1, grounded_cfg.d_model)),
            vitals_embedding=torch.zeros((1, grounded_cfg.d_model)),
            recurrent_state=torch.zeros((1, grounded_cfg.recurrent_hidden_dim)),
            policy_branch_ids=rest_enc.batch.candidates.policy_branch_ids,
            action_mask=rest_enc.batch.candidates.action_mask,
            domain_ids=rest_enc.batch.domain_ids,
        )
        skew_probs = torch.softmax(skew_logits[0], dim=0)
        p_rest_skew = sum(
            float(skew_probs[idx].item())
            for idx, ref in enumerate(rest_enc.actions)
            if rest_dec.candidates[ref.representative_position].branch == "rest"
        )
        assert p_rest_skew > 0.80

    # 4. Candidate-independent forward_durable_value == forward.durable_value and strong HP sensitivity.
    low_obs, _ = _rest_surface()
    low_obs["player"]["hp"] = 16
    high_obs, _ = _rest_surface()
    high_obs["player"]["hp"] = 68

    low_world = encoder.encode_world_snapshot(low_obs)
    high_world = encoder.encode_world_snapshot(high_obs)
    wb, w_dom = collate_encoded_worlds(
        (low_world, high_world),
        expected_config=low_world.config,
        expected_fingerprint=low_world.encoding_fingerprint,
        device="cpu",
    )
    with torch.no_grad():
        v_standalone, _ = v3_model.forward_durable_value(wb, w_dom)
        low_enc = encoder.encode(low_obs, [c.semantic_action for c in rest_dec.candidates])
        out_low = v3_model(low_enc.batch)
        assert out_low.durable_value is not None
        torch.testing.assert_close(v_standalone[:1], out_low.durable_value, rtol=1e-5, atol=1e-6)

        vitals_pair, raw_vitals_pair = v3_model._encode_direct_vitals(wb)
        vitals_l2_diff = float((vitals_pair[0] - vitals_pair[1]).norm().item())
        raw_l2_diff = float((raw_vitals_pair[0] - raw_vitals_pair[1]).norm().item())
        assert vitals_l2_diff > 0.20
        assert raw_l2_diff > 1.0

    opt = torch.optim.AdamW(v3_model.parameters(), lr=3e-3)
    targets = torch.tensor([0.15, 0.85], dtype=torch.float32)
    for _ in range(24):
        opt.zero_grad(set_to_none=True)
        preds, _ = v3_model.forward_durable_value(wb, w_dom)
        loss = torch.nn.functional.smooth_l1_loss(preds, targets)
        loss.backward()
        opt.step()
    with torch.no_grad():
        fitted_preds, _ = v3_model.forward_durable_value(wb, w_dom)
    assert float((fitted_preds[1] - fitted_preds[0]).item()) > 0.20


def test_separate_value_grad_clip_prevents_value_norm_from_crushing_policy_updates() -> None:
    torch.manual_seed(29)
    cfg = _tiny_model_config()
    encoder = GroundedObservationEncoder(cfg.to_encoding_config())
    rest_obs, rest_legal = _rest_surface()
    rest_dec = forward_decision(rest_obs, rest_legal, control_domain="macro")
    assert rest_dec is not None
    rest_enc = encoder.encode(rest_obs, [c.semantic_action for c in rest_dec.candidates])
    grouped_cands = tuple(rest_dec.candidates[ref.representative_position] for ref in rest_enc.actions)
    post_worlds = synthesize_deterministic_afterstates(
        encoder=encoder,
        observation=rest_obs,
        decision=rest_dec,
        candidates=grouped_cands,
    )
    assert post_worlds is not None

    base_ac_cfg = HybridActorCriticConfig.durable_v3(cfg.to_model_config())
    init_model = HybridDomainActorCriticModel(base_ac_cfg).eval()
    with torch.no_grad():
        # Inject non-zero durable_value_head weights so post-states differ and MPO enables actor gradients
        for param in init_model.durable_value_head.parameters():
            param.add_(0.05 * torch.randn_like(param))
    mask = torch.tensor(rest_enc.snapshot.action_mask.tolist(), dtype=torch.bool)

    node = MacroAfterstateNode(
        step_index=0,
        surface="rest",
        candidate_branches=tuple(c.branch for c in grouped_cands),
        post_worlds=post_worlds,
    )
    def _make_cohort(
        teacher_model: HybridDomainActorCriticModel, terminal_reward: float, teacher_version: int
    ) -> MacroCreditCohort:
        with torch.no_grad():
            t_out = teacher_model.eval()(rest_enc.batch)
            t_lp = torch.log_softmax(t_out.policy_logits[0, : mask.numel()].masked_fill(~mask, -torch.inf), dim=0)
            step_beh_lp = float(macro_mixture_log_probs(t_lp, rest_enc.snapshot, epsilon=0.05)[0].item())
        s = HybridVTraceStep(
            snapshot=rest_enc.snapshot,
            action_index=0,
            behavior_log_probability=step_beh_lp,
            reward=terminal_reward,
            discount=0.0,
            terminal=True,
            surface="rest",
            branch=grouped_cands[0].branch,
            control_domain="macro",
            policy_version=teacher_version,
        )
        t = MacroCreditTrajectory(
            trajectory_id=f"grad-clip-ref-{teacher_version}",
            seed=6_700_000,
            teacher_version=teacher_version,
            prefix=(),
            steps=(s,),
            bootstrap_snapshot=None,
            bootstrap_domain=None,
            ending="factual_terminal",
            reward_facts=(
                MacroRewardFacts(
                    terminal_reward=terminal_reward,
                    progress_reward=0.0,
                    revival_penalty=0.0,
                    pace_penalty=0.0,
                    hp_loss_penalty=0.0,
                    revivals_used_delta=0,
                    player_hp_lost_delta=0.0,
                    floor_before=6,
                    floor_after=6,
                    native_steps=1,
                ),
            ),
            afterstate_nodes=(node,),
            macro_branch_epsilon=0.05,
        )
        return MacroCreditCohort(
            cohort_id=f"grad-clip-cohort-{teacher_version}",
            teacher_version=teacher_version,
            roots=(),
            natural_only=(t,),
        )

    joint_model = copy.deepcopy(init_model)
    decoupled_model = copy.deepcopy(init_model)

    joint_learner = MacroCreditLearner(
        model=joint_model,
        config=MacroCreditLearnerConfig(
            target_mode="durable_afterstate_v2",
            durable_warmup_updates=0,
            gradient_clip_norm=1.0,
            separate_value_grad_clip=False,
        ),
    )
    decoupled_learner = MacroCreditLearner(
        model=decoupled_model,
        config=MacroCreditLearnerConfig(
            target_mode="durable_afterstate_v2",
            durable_warmup_updates=0,
            gradient_clip_norm=1.0,
            separate_value_grad_clip=True,
        ),
    )
    # Step 1: calm cohort initializes AdamW's running variance v_t without value clipping
    joint_learner.learn(_make_cohort(joint_model, 0.05, teacher_version=0))
    decoupled_learner.learn(_make_cohort(decoupled_model, 0.05, teacher_version=0))

    joint_before_spike = copy.deepcopy(joint_model)
    decoupled_before_spike = copy.deepcopy(decoupled_model)

    # Step 2: value residual spike (value_head_gradient_norm ~ 3.68 under Huber delta=1.0)
    m_joint = joint_learner.learn(_make_cohort(joint_model, 250.0, teacher_version=1))
    m_decoupled = decoupled_learner.learn(_make_cohort(decoupled_model, 250.0, teacher_version=1))
    assert m_joint["value_head_gradient_norm"] > 3.0

    joint_clipped_policy_grad = (
        sum(
            float(p.grad.square().sum().item())
            for n, p in joint_model.named_parameters()
            if n.startswith("branch_policy_head.") and p.grad is not None
        )
        ** 0.5
    )
    decoupled_clipped_policy_grad = (
        sum(
            float(p.grad.square().sum().item())
            for n, p in decoupled_model.named_parameters()
            if n.startswith("branch_policy_head.") and p.grad is not None
        )
        ** 0.5
    )
    assert decoupled_clipped_policy_grad > joint_clipped_policy_grad * 3.0

    joint_branch_delta = sum(
        float((p_after - p_prev).norm().item())
        for (n, p_prev), (_, p_after) in zip(
            joint_before_spike.named_parameters(), joint_model.named_parameters(), strict=True
        )
        if n.startswith("branch_policy_head.")
    )
    decoupled_branch_delta = sum(
        float((p_after - p_prev).norm().item())
        for (n, p_prev), (_, p_after) in zip(
            decoupled_before_spike.named_parameters(), decoupled_model.named_parameters(), strict=True
        )
        if n.startswith("branch_policy_head.")
    )
    assert decoupled_branch_delta > joint_branch_delta * 1.2
    assert m_decoupled["policy_head_gradient_norm"] > 0.0


def test_b16_g4_turbo_experiment_config_loads_and_validates() -> None:
    cfg_path = (
        Path(__file__).resolve().parents[1]
        / "config"
        / "experiments"
        / "hybrid_macro_credit_v7_durable_afterstate_b16_g4_turbo.toml"
    )
    loaded = load_macro_training_config(cfg_path)
    assert loaded.training.curriculum.revival_budget == 16
    assert loaded.course.roots_per_cohort == 48
    assert loaded.course.parallel_collectors == 48
    assert loaded.learner.enable_direct_vitals_context is True
    assert loaded.learner.enable_macro_branch_prior is True
    assert loaded.learner.separate_value_grad_clip is True
    assert loaded.learner.combat_entropy_weight == pytest.approx(0.0005)
    assert loaded.learner.combat_actor_warmup_updates == 15


def test_durable_v4_persistent_invariance_and_strict_actor_isolation() -> None:
    torch.manual_seed(20261112)
    cfg = _tiny_model_config()
    encoder = GroundedObservationEncoder(cfg.to_encoding_config())
    model = HybridDomainActorCriticModel(
        HybridActorCriticConfig.durable_v4(cfg.to_model_config())
    ).eval()
    with torch.no_grad():
        for head in (
            model.durable_value_head,
            model.durable_vitals_head,
            model.durable_revival_head,
            model.state_revival_head,
        ):
            for param in head.parameters():
                param.add_(0.1 * torch.randn_like(param))

    shared_player = {
        "character_id": "IRONCLAD",
        "hp": 54,
        "max_hp": 80,
        "gold": 145,
        "deck": _sample_deck(),
        "relics": [{"id": "BURNING_BLOOD"}],
        "potions": [{"id": "STRENGTH_POTION"}],
    }
    shared_run = {"active": True, "act": 1, "floor": 8, "ascension": 0}

    shop_obs = {
        "phase": "shop",
        "decision_domain": "shop",
        "run": copy.deepcopy(shared_run),
        "player": copy.deepcopy(shared_player),
        "shop": {
            "visible": True,
            "cards": [{"id": "CARD.SHRUG_IT_OFF", "price": 75, "affordable": True}],
            "relics": [{"id": "RELIC.VAJRA", "price": 140, "affordable": True}],
            "card_removal": {"available": True, "price": 75, "affordable": True},
        },
    }
    shop_actions = [
        {
            "action_id": "s0",
            "kind": "shop_buy_card",
            "model_action_kind": "shop",
            "is_enabled": True,
            "item": {"id": "CARD.SHRUG_IT_OFF", "price": 75},
        },
        {
            "action_id": "s1",
            "kind": "proceed",
            "model_action_kind": "proceed",
            "is_enabled": True,
        },
    ]

    map_obs = {
        "phase": "map",
        "decision_domain": "map",
        "run": copy.deepcopy(shared_run),
        "player": copy.deepcopy(shared_player),
        "map": {
            "nodes": [
                {"x": idx, "y": 8, "type": "Monster" if idx % 2 == 0 else "Elite"}
                for idx in range(6)
            ]
        },
    }
    map_actions = [
        {
            "action_id": "m0",
            "kind": "choose_map_node",
            "model_action_kind": "map",
            "is_enabled": True,
            "node": {"x": 0, "y": 8, "type": "Monster"},
        },
        {
            "action_id": "m1",
            "kind": "choose_map_node",
            "model_action_kind": "map",
            "is_enabled": True,
            "node": {"x": 1, "y": 8, "type": "Elite"},
        },
    ]

    rewards_obs = {
        "phase": "rewards",
        "decision_domain": "rewards",
        "run": copy.deepcopy(shared_run),
        "player": copy.deepcopy(shared_player),
        "rewards": {
            "items": [
                {"type": "Gold", "gold_amount": 35},
                {"type": "Relic", "relic": {"id": "RELIC.ANCHOR"}},
            ]
        },
    }
    rewards_actions = [
        {
            "action_id": "r0",
            "kind": "claim_reward",
            "model_action_kind": "treasure",
            "is_enabled": True,
            "reward": {"type": "Gold", "gold_amount": 35},
        },
        {
            "action_id": "r1",
            "kind": "proceed",
            "model_action_kind": "proceed",
            "is_enabled": True,
        },
    ]

    shop_enc = encoder.encode(shop_obs, shop_actions)
    map_enc = encoder.encode(map_obs, map_actions)
    rewards_enc = encoder.encode(rewards_obs, rewards_actions)

    h0 = torch.randn(1, cfg.recurrent_hidden_dim)
    rev_zero = torch.zeros(1, 6)
    rev_high = torch.tensor([[1.0, 1.0, 0.85, 1.6, 1.0, 1.0]], dtype=torch.float32)

    with torch.no_grad():
        out_shop = model(shop_enc.batch, h0, critic_revival_context=rev_zero)
        out_map = model(map_enc.batch, h0, critic_revival_context=rev_zero)
        out_rewards = model(rewards_enc.batch, h0, critic_revival_context=rev_zero)
        assert out_shop.durable_value is not None
        assert out_map.durable_value is not None
        assert out_rewards.durable_value is not None
        # Exact persistent-only invariance across shop / map / rewards!
        assert float((out_shop.durable_value - out_map.durable_value).abs().item()) < 1e-5
        assert float((out_shop.durable_value - out_rewards.durable_value).abs().item()) < 1e-5

        w_shop, d_shop = collate_encoded_worlds(
            (encoder.encode_world_snapshot(shop_obs),),
            expected_config=shop_enc.snapshot.config,
            expected_fingerprint=shop_enc.snapshot.encoding_fingerprint,
            device="cpu",
        )
        w_map, d_map = collate_encoded_worlds(
            (encoder.encode_world_snapshot(map_obs),),
            expected_config=map_enc.snapshot.config,
            expected_fingerprint=map_enc.snapshot.encoding_fingerprint,
            device="cpu",
        )
        dur_w_shop, _ = model.forward_durable_value(w_shop, d_shop, h0, critic_revival_context=rev_zero)
        dur_w_map, _ = model.forward_durable_value(w_map, d_map, h0, critic_revival_context=rev_zero)
        assert float((dur_w_shop - dur_w_map).abs().item()) < 1e-5
        assert float((dur_w_shop - out_shop.durable_value).abs().item()) < 1e-5

        # Strict Actor isolation: critic_revival_context changes Critic values, NEVER policy_logits or recurrent_state!
        out_shop_high_rev = model(shop_enc.batch, h0, critic_revival_context=rev_high)
        assert out_shop_high_rev.durable_value is not None
        assert torch.equal(out_shop.policy_logits, out_shop_high_rev.policy_logits)
        assert torch.equal(out_shop.recurrent_state, out_shop_high_rev.recurrent_state)
        assert float((out_shop.value - out_shop_high_rev.value).abs().item()) > 1e-3
        assert float((out_shop.durable_value - out_shop_high_rev.durable_value).abs().item()) > 1e-3


def test_asymmetric_revival_critic_resolves_simpsons_paradox_hp_sign() -> None:
    torch.manual_seed(777)
    cfg = _tiny_model_config()
    encoder = GroundedObservationEncoder(cfg.to_encoding_config())

    def _make_rest_snap(hp_val: int) -> tuple[Any, Any]:
        obs, actions = _rest_surface()
        obs["player"]["hp"] = hp_val
        compiled = forward_decision(obs, actions, control_domain="macro")
        assert compiled is not None
        enc = encoder.encode(
            obs,
            [c.semantic_action for c in compiled.candidates],
        )
        w = encoder.encode_world_snapshot(obs)
        return enc, w

    low_enc, low_world = _make_rest_snap(24)
    high_enc, high_world = _make_rest_snap(76)
    combat_obs = {
        "phase": "combat",
        "decision_domain": "combat",
        "run": {"active": True, "act": 1, "floor": 5},
        "player": {"id": "ironclad", "side": "player", "hp": 40, "max_hp": 80, "energy": 3, "hand": [{"id": "strike", "cost": 1}]},
        "combat": {"in_progress": True, "enemies": [{"id": "cultist", "side": "enemy", "hp": 30, "max_hp": 48}]},
    }
    combat_actions = [
        {"action_handle": "c0", "kind": "play_card", "model_action_kind": "play_card", "is_enabled": True, "card": {"id": "strike", "cost": 1}},
    ]
    combat_snap = encoder.encode(combat_obs, combat_actions).snapshot

    v3_model = HybridDomainActorCriticModel(HybridActorCriticConfig.durable_v3(cfg.to_model_config())).eval()
    v4_model = HybridDomainActorCriticModel(HybridActorCriticConfig.durable_v4(cfg.to_model_config())).eval()

    v3_learner = MacroCreditLearner(
        model=v3_model,
        config=MacroCreditLearnerConfig(
            target_mode="durable_afterstate_v2",
            enable_direct_vitals_context=True,
            enable_macro_branch_prior=True,
            learning_rate=3e-3,
            durable_warmup_updates=20,
            combat_actor_warmup_updates=20,
        ),
    )
    v4_learner = MacroCreditLearner(
        model=v4_model,
        config=MacroCreditLearnerConfig(
            target_mode="durable_afterstate_v2",
            enable_direct_vitals_context=True,
            enable_macro_branch_prior=True,
            enable_persistent_only_durable_value=True,
            enable_asymmetric_revival_critic=True,
            revival_budget=16,
            learning_rate=3e-3,
            durable_warmup_updates=20,
            combat_actor_warmup_updates=20,
        ),
    )

    def _build_confounded_cohort(
        teacher: HybridDomainActorCriticModel, teacher_version: int
    ) -> MacroCreditCohort:
        # Spec: (rest_enc, revivals_burned_in_combat_step0, revival_pen_step0, terminal_reward_step1)
        # Unconditionally: low HP (24) -> +1.50; high HP (76) -> (2.2 - 2.0*3)/4 = -0.95 (Simpson's Paradox!)
        # Conditionally on u=0: high HP (76) -> +2.20 > low HP (24) -> +1.50!
        specs = [
            (low_enc, 0, 0.0, 1.5),
            (high_enc, 0, 0.0, 2.2),
            (high_enc, 16, -1.6, -2.0),
            (high_enc, 16, -1.6, -2.0),
            (high_enc, 16, -1.6, -2.0),
        ]
        trajs: list[MacroCreditTrajectory] = []
        with torch.no_grad():
            for idx, (r_enc, rev_delta, rev_pen, term_rew) in enumerate(specs):
                h = None
                c_batch = collate_encoded_snapshots(
                    (combat_snap,),
                    expected_config=combat_snap.config,
                    expected_fingerprint=combat_snap.encoding_fingerprint,
                    device="cpu",
                )
                c_out = teacher(c_batch, h, validate=False)
                h = c_out.recurrent_state
                r_batch = collate_encoded_snapshots(
                    (r_enc.snapshot,),
                    expected_config=r_enc.snapshot.config,
                    expected_fingerprint=r_enc.snapshot.encoding_fingerprint,
                    device="cpu",
                )
                r_out = teacher(r_batch, h, validate=False)
                r_mask = torch.tensor(r_enc.snapshot.action_mask.tolist(), dtype=torch.bool)
                r_logs = torch.log_softmax(r_out.policy_logits[0, : r_mask.numel()].masked_fill(~r_mask, -torch.inf), dim=0)
                beh_lp = float(r_logs[0].item())

                step0 = HybridVTraceStep(
                    snapshot=combat_snap,
                    action_index=0,
                    behavior_log_probability=0.0,
                    reward=rev_pen,
                    discount=1.0,
                    terminal=False,
                    surface="combat",
                    branch="play_card",
                    control_domain="combat",
                    policy_version=teacher_version,
                )
                fact0 = MacroRewardFacts(
                    terminal_reward=0.0,
                    progress_reward=0.0,
                    revival_penalty=rev_pen,
                    pace_penalty=0.0,
                    hp_loss_penalty=0.0,
                    revivals_used_delta=rev_delta,
                    player_hp_lost_delta=0.0,
                    floor_before=6,
                    floor_after=6,
                    native_steps=1,
                )
                step1 = HybridVTraceStep(
                    snapshot=r_enc.snapshot,
                    action_index=0,
                    behavior_log_probability=beh_lp,
                    reward=term_rew,
                    discount=0.0,
                    terminal=True,
                    surface="rest",
                    branch="rest",
                    control_domain="macro",
                    policy_version=teacher_version,
                )
                fact1 = MacroRewardFacts(
                    terminal_reward=term_rew,
                    progress_reward=0.0,
                    revival_penalty=0.0,
                    pace_penalty=0.0,
                    hp_loss_penalty=0.0,
                    revivals_used_delta=0,
                    player_hp_lost_delta=0.0,
                    floor_before=6,
                    floor_after=6,
                    native_steps=1,
                )
                trajs.append(
                    MacroCreditTrajectory(
                        trajectory_id=f"simpson-{teacher_version}-{idx}",
                        seed=6_700_000 + 2 * idx,
                        teacher_version=teacher_version,
                        prefix=(),
                        steps=(step0, step1),
                        bootstrap_snapshot=None,
                        bootstrap_domain=None,
                        ending="factual_terminal",
                        reward_facts=(fact0, fact1),
                        afterstate_nodes=(),
                        macro_branch_epsilon=0.0,
                    )
                )
        return MacroCreditCohort(f"simpson-c-{teacher_version}", teacher_version, (), natural_only=tuple(trajs))

    for upd in range(28):
        v3_learner.learn(_build_confounded_cohort(v3_model, upd))
        v4_learner.learn(_build_confounded_cohort(v4_model, upd))

    w_pair, d_pair = collate_encoded_worlds(
        (low_world, high_world),
        expected_config=low_enc.snapshot.config,
        expected_fingerprint=low_enc.snapshot.encoding_fingerprint,
        device="cpu",
    )
    c_eval_batch = collate_encoded_snapshots(
        (combat_snap,),
        expected_config=combat_snap.config,
        expected_fingerprint=combat_snap.encoding_fingerprint,
        device="cpu",
    )
    with torch.no_grad():
        h_v3 = v3_model(c_eval_batch, None, validate=False).recurrent_state.expand(2, -1)
        h_v4 = v4_model(c_eval_batch, None, validate=False).recurrent_state.expand(2, -1)
        v3_vals, _ = v3_model.forward_durable_value(w_pair, d_pair, h_v3)
        v4_zero_rev, _ = v4_model.forward_durable_value(
            w_pair, d_pair, h_v4, critic_revival_context=torch.zeros((2, 6), dtype=torch.float32)
        )
        exhausted_ctx = torch.tensor([v4_learner._critic_revival_vector(16, 1.6)] * 2, dtype=torch.float32)
        v4_exhausted_rev, _ = v4_model.forward_durable_value(
            w_pair, d_pair, h_v4, critic_revival_context=exhausted_ctx
        )

    # Symmetric v3 suffers Simpson's Paradox: learns V_durable(HP=76) < V_durable(HP=24)!
    assert float(v3_vals[1].item()) < float(v3_vals[0].item())
    # Asymmetric v4 resolves Simpson's Paradox: V_durable(HP=76, u=0) > V_durable(HP=24, u=0)
    # AND rightly penalizes exhausted revivals u=16!
    assert float(v4_zero_rev[1].item()) > float(v4_zero_rev[0].item())
    assert float(v4_exhausted_rev[1].item()) < float(v4_zero_rev[1].item()) - 0.5


def test_zero_macro_pace_penalty_and_clipped_macro_pg_fallback_and_b32_config() -> None:
    torch.manual_seed(1234)
    cfg = _tiny_model_config()
    encoder = GroundedObservationEncoder(cfg.to_encoding_config())

    rewards_obs = {
        "phase": "rewards",
        "decision_domain": "rewards",
        "run": {"active": True, "act": 1, "floor": 4},
        "player": {"character_id": "IRONCLAD", "hp": 60, "max_hp": 80, "gold": 100, "deck": _sample_deck(), "relics": [], "potions": []},
        "rewards": {"items": [{"type": "Gold", "gold_amount": 25}, {"type": "CardReward"}]},
    }
    rewards_actions = [
        {"action_id": "r0", "kind": "claim_reward", "model_action_kind": "treasure", "is_enabled": True, "reward": {"type": "Gold", "gold_amount": 25}},
        {"action_id": "r1", "kind": "proceed", "model_action_kind": "proceed", "is_enabled": True},
    ]
    rew_snap = encoder.encode(rewards_obs, rewards_actions).snapshot

    combat_obs = {
        "phase": "combat",
        "decision_domain": "combat",
        "run": {"active": True, "act": 1, "floor": 5},
        "player": {"id": "ironclad", "side": "player", "hp": 60, "max_hp": 80, "energy": 3, "hand": [{"id": "strike", "cost": 1}]},
        "combat": {"in_progress": True, "enemies": [{"id": "cultist", "side": "enemy", "hp": 40, "max_hp": 48}]},
    }
    combat_actions = [
        {"action_handle": "c0", "kind": "play_card", "model_action_kind": "play_card", "is_enabled": True, "card": {"id": "strike", "cost": 1}},
        {"action_handle": "c1", "kind": "end_turn", "model_action_kind": "end_turn", "is_enabled": True},
    ]
    combat_snap = encoder.encode(combat_obs, combat_actions).snapshot

    init_model = HybridDomainActorCriticModel(HybridActorCriticConfig.durable_v4(cfg.to_model_config())).eval()

    def _make_pg_cohort(teacher: HybridDomainActorCriticModel) -> MacroCreditCohort:
        trajs: list[MacroCreditTrajectory] = []
        with torch.no_grad():
            for idx in range(2):
                b0 = collate_encoded_snapshots((rew_snap,), expected_config=rew_snap.config, expected_fingerprint=rew_snap.encoding_fingerprint, device="cpu")
                o0 = teacher(b0, None, validate=False)
                m0 = torch.tensor(rew_snap.action_mask.tolist(), dtype=torch.bool)
                lp0 = torch.log_softmax(o0.policy_logits[0, : m0.numel()].masked_fill(~m0, -torch.inf), dim=0)
                b1 = collate_encoded_snapshots((combat_snap,), expected_config=combat_snap.config, expected_fingerprint=combat_snap.encoding_fingerprint, device="cpu")
                o1 = teacher(b1, o0.recurrent_state, validate=False)
                m1 = torch.tensor(combat_snap.action_mask.tolist(), dtype=torch.bool)
                lp1 = torch.log_softmax(o1.policy_logits[0, : m1.numel()].masked_fill(~m1, -torch.inf), dim=0)

                # Step 0 (macro): pace_penalty = -0.15, progress = 0.0 -> step.reward = -0.15
                # Step 1 (combat): pace_penalty = -0.05, terminal = 2.0 -> step.reward = 1.95
                s0 = HybridVTraceStep(rew_snap, idx % 2, float(lp0[idx % 2].item()), -0.15, 1.0, False, "rewards", "claim_gold", "macro", 0)
                f0 = MacroRewardFacts(0.0, 0.0, 0.0, -0.15, 0.0, 0, 0.0, 4, 4, 1)
                s1 = HybridVTraceStep(combat_snap, 0, float(lp1[0].item()), 1.95, 0.0, True, "combat", "play_card", "combat", 0)
                f1 = MacroRewardFacts(2.0, 0.0, 0.0, -0.05, 0.0, 0, 0.0, 5, 5, 1)
                trajs.append(
                    MacroCreditTrajectory(
                        f"pg-ep-{idx}",
                        6_700_000 + 2 * idx,
                        0,
                        (),
                        (s0, s1),
                        None,
                        None,
                        "factual_terminal",
                        (f0, f1),
                        afterstate_nodes=(),
                        macro_branch_epsilon=0.0,
                    )
                )
        return MacroCreditCohort("pg-cohort", 0, (), natural_only=tuple(trajs))

    cohort = _make_pg_cohort(init_model)
    model_batch = copy.deepcopy(init_model)
    model_single = copy.deepcopy(init_model)

    base_lcfg = dict(
        target_mode="durable_afterstate_v2",
        enable_direct_vitals_context=True,
        enable_macro_branch_prior=True,
        enable_persistent_only_durable_value=True,
        enable_asymmetric_revival_critic=True,
        revival_budget=32,
        zero_macro_pace_penalty=True,
        clip_macro_pg_fallback=True,
        macro_pg_clip_eps=0.20,
        macro_pg_adv_clip=0.25,
        macro_pg_kl_weight=0.05,
        durable_warmup_updates=0,
        combat_actor_warmup_updates=0,
    )
    learner_batch = MacroCreditLearner(
        model=model_batch,
        config=MacroCreditLearnerConfig(**base_lcfg, eval_batch_size=8, chunk_batch_size=8),  # type: ignore[arg-type]
    )
    learner_single = MacroCreditLearner(
        model=model_single,
        config=MacroCreditLearnerConfig(**base_lcfg, eval_batch_size=1, chunk_batch_size=1),  # type: ignore[arg-type]
    )
    # Macro step 0 strips -0.15 pace_penalty -> effective reward 0.0; combat step 1 keeps -0.05 -> 1.95!
    assert learner_batch._effective_step_reward(cohort.natural_only[0], 0) == pytest.approx(0.0)
    assert learner_batch._effective_step_reward(cohort.natural_only[0], 1) == pytest.approx(1.95)

    m_batch = learner_batch.learn(cohort)
    m_single = learner_single.learn(cohort)
    assert m_batch["macro_pg_policy_loss"] == pytest.approx(m_single["macro_pg_policy_loss"], abs=1e-5)
    assert m_batch["durable_value_loss"] == pytest.approx(m_single["durable_value_loss"], abs=1e-5)
    assert m_batch["parameter_delta_norm"] == pytest.approx(m_single["parameter_delta_norm"], abs=1e-5)
    # Clipped advantage (~0.25 * tanh(1.95 / 0.25) ~ 0.25) bounds |macro_pg_policy_loss| <= 0.25!
    assert abs(m_batch["macro_pg_policy_loss"]) <= 0.26

    v8_path = (
        Path(__file__).resolve().parents[1]
        / "config"
        / "experiments"
        / "hybrid_macro_credit_v8_durable_afterstate_b32_g4_turbo.toml"
    )
    loaded_v8 = load_macro_training_config(v8_path)
    assert loaded_v8.training.curriculum.revival_budget == 32
    assert loaded_v8.learner.revival_budget == 32
    assert loaded_v8.learner.enable_persistent_only_durable_value is True
    assert loaded_v8.learner.enable_asymmetric_revival_critic is True
    assert loaded_v8.learner.zero_macro_pace_penalty is True
    assert loaded_v8.learner.clip_macro_pg_fallback is True
    assert loaded_v8.learner.macro_pg_within_branch_entropy_only is True
    assert loaded_v8.learner.allow_tf32 is False
    assert loaded_v8.learner.likelihood_tolerance == pytest.approx(0.02)
    assert loaded_v8.course.seed_index_offset == 90000

    with pytest.raises(ValueError, match=r"learner\.revival_budget"):
        load_macro_training_config(v8_path, overrides=("learner.revival_budget=16",))


def test_v4_1_card_reward_raw_vitals_invariance_and_salient_linear_scope() -> None:
    cfg = _tiny_model_config()
    encoder = GroundedObservationEncoder(cfg.to_encoding_config())
    model = HybridDomainActorCriticModel(HybridActorCriticConfig.durable_v4(cfg.to_model_config())).eval()
    assert model.durable_vitals_linear.in_features == 4

    card_reward_obs = {
        "phase": "card_reward",
        "decision_domain": "card_reward",
        "run": {"active": True, "act": 1, "floor": 4, "ascension": 0},
        "player": {
            "character_id": "IRONCLAD",
            "hp": 54,
            "max_hp": 80,
            "gold": 115,
            "open_potion_slots": 2,
            "deck": _sample_deck(),
            "relics": [{"id": "BURNING_BLOOD", "has_upon_pickup_effect": False}],
            "potions": [],
        },
        "card_reward_selection": {
            "can_skip": True,
            "cards": [
                {
                    "id": "CARD.OFFERING",
                    "type": "Skill",
                    "rarity": "Rare",
                    "cost": 0,
                    "keywords": ["exhaust"],
                    "dynamic_vars": {"Cards": 3, "Energy": 2},
                }
            ],
        },
    }
    card_reward_actions = [
        {
            "action_id": "take:0",
            "kind": "select_card_reward",
            "model_action_kind": "card_reward",
            "is_enabled": True,
            "card": {
                "id": "CARD.OFFERING",
                "type": "Skill",
                "rarity": "Rare",
                "cost": 0,
                "keywords": ["exhaust"],
                "dynamic_vars": {"Cards": 3, "Energy": 2},
            },
        },
        {
            "action_id": "skip",
            "kind": "skip_card_reward",
            "model_action_kind": "card_reward",
            "is_enabled": True,
        },
    ]
    dec = forward_decision(card_reward_obs, card_reward_actions, control_domain="macro")
    assert dec is not None
    post_worlds = synthesize_deterministic_afterstates(
        encoder=encoder,
        observation=card_reward_obs,
        decision=dec,
    )
    assert post_worlds is not None and len(post_worlds) == 2

    world_batch, _ = collate_encoded_worlds(
        post_worlds,
        expected_config=post_worlds[0].config,
        expected_fingerprint=post_worlds[0].encoding_fingerprint,
        device="cpu",
    )
    pers_world = model._mask_world_to_persistent(world_batch)
    raw_vitals = model._extract_scaled_vitals(pers_world)
    # raw_vitals(take(card_with_keyword)) == raw_vitals(skip) identically (0.0 diff):
    # no identity-blind deck/modifier count shortcut can penalize taking strong cards!
    assert float((raw_vitals[0] - raw_vitals[1]).abs().max().item()) == 0.0


def test_v4_1_hierarchical_macro_entropy_cardinality_neutral() -> None:
    from torch.nn import functional as F

    from sts2_rl.joint.macro_credit_learner import hierarchical_macro_entropy

    # Branch 0: singleton Rest (1 candidate); Branch 1: Smith (10 candidates).
    branch_logit_0 = torch.tensor(0.0, dtype=torch.float64, requires_grad=True)
    within_smith_logits = torch.tensor(
        [1.2, 0.3, -0.1, -0.2, -0.2, -0.2, -0.2, -0.2, -0.2, -0.2],
        dtype=torch.float64,
        requires_grad=True,
    )
    log_p_b0 = F.logsigmoid(branch_logit_0)
    log_p_b1 = F.logsigmoid(-branch_logit_0)
    smith_within = torch.log_softmax(within_smith_logits, dim=0)
    joint_logs = torch.cat([log_p_b0.unsqueeze(0), log_p_b1 + smith_within], dim=0).unsqueeze(0)
    masks = torch.ones((1, 11), dtype=torch.bool)
    branch_ids = torch.tensor([[0] + [1] * 10], dtype=torch.long)

    # Flat Shannon entropy has -H(smith_within) ~ -2.1465 pull against singleton Rest at p_rest = 0.5!
    within_ent = -(smith_within.exp() * smith_within).sum()
    flat_ent = -(joint_logs.exp() * joint_logs).sum()
    (flat_grad_b0,) = torch.autograd.grad(flat_ent, (branch_logit_0,), retain_graph=True)
    # d H_flat / d logit_0 = p0*(1-p0)*(-H(smith_within)) = -0.25 * H(smith_within) < -0.50
    assert float(flat_grad_b0.item()) == pytest.approx(-0.25 * float(within_ent.item()), abs=1e-9)
    assert float(flat_grad_b0.item()) < -0.50

    # Hierarchical macro entropy has EXACT ZERO gradient wrt branch logit at p_rest = 0.5,
    # while still encouraging uniform exploration across skewed within-Smith logits!
    hier_ent = hierarchical_macro_entropy(joint_logs, masks, branch_ids)[0]
    grad_b0, grad_within = torch.autograd.grad(hier_ent, (branch_logit_0, within_smith_logits))
    assert abs(float(grad_b0.item())) < 1e-12
    # Highest-prob Smith card (index 0) gets negative entropy gradient (pushed toward uniform).
    assert float(grad_within[0].item()) < -0.05

    # Also verify masked slot 0 produces finite values & finite gradients (no -inf logsumexp NaNs).
    masked_slots = torch.tensor([[False, True, True, True, False]], dtype=torch.bool)
    raw_l = torch.tensor([[0.5, 0.2, -0.3, 0.8, -1.0]], dtype=torch.float32, requires_grad=True)
    all_lp = torch.log_softmax(raw_l.masked_fill(~masked_slots, -torch.inf), dim=-1)
    safe_lp = all_lp.masked_fill(~masked_slots, 0.0)
    b_ids = torch.tensor([[0, 1, 2, 2, 0]], dtype=torch.long)
    ent_masked = hierarchical_macro_entropy(safe_lp, masked_slots, b_ids).sum()
    ent_masked.backward()  # type: ignore[no-untyped-call]
    assert bool(torch.isfinite(ent_masked).item())
    assert raw_l.grad is not None and bool(torch.isfinite(raw_l.grad).all().item())


def test_v4_1_salient_bilinear_actor_unlocks_campfire_hp_sensitivity_and_recoverable_stalemate() -> None:
    from sts2_rl.joint.macro_credit_collection import is_recoverable_episode_exception
    from sts2_rl.macro_competence_probes import _build_rest_probe_surface, default_probe_decks
    from sts2_rl.macro_root_diagnostic import ProbeBudgetExhausted

    assert is_recoverable_episode_exception(ProbeBudgetExhausted("max_native_steps")) is True
    assert is_recoverable_episode_exception(ProbeBudgetExhausted("timeout_seconds")) is True
    assert is_recoverable_episode_exception(ProbeBudgetExhausted("incomplete_durable_episode")) is True
    assert is_recoverable_episode_exception(ProbeBudgetExhausted("maximum_cohort_seconds")) is False
    assert is_recoverable_episode_exception(ProbeBudgetExhausted("maximum_cohort_snapshot_bytes")) is False

    torch.manual_seed(2026)
    cfg = _tiny_model_config()
    encoder = GroundedObservationEncoder(cfg.to_encoding_config())
    model = HybridDomainActorCriticModel(HybridActorCriticConfig.durable_v4(cfg.to_model_config()))

    deck = default_probe_decks()[0]
    obs_low, legal_low = _build_rest_probe_surface(deck, hp=16, max_hp=80)
    obs_high, legal_high = _build_rest_probe_surface(deck, hp=68, max_hp=80)
    dec_low = forward_decision(obs_low, legal_low, control_domain="macro")
    dec_high = forward_decision(obs_high, legal_high, control_domain="macro")
    assert dec_low is not None and dec_high is not None
    enc_low = encoder.encode(obs_low, [c.semantic_action for c in dec_low.candidates])
    enc_high = encoder.encode(obs_high, [c.semantic_action for c in dec_high.candidates])

    batch = collate_encoded_snapshots(
        (enc_low.snapshot, enc_high.snapshot),
        expected_config=enc_low.snapshot.config,
        expected_fingerprint=enc_low.snapshot.encoding_fingerprint,
        device="cpu",
    )
    rest_idx_low = next(
        idx
        for idx, ref in enumerate(enc_low.actions)
        if dec_low.candidates[ref.representative_position].branch == "rest"
    )
    rest_idx_high = next(
        idx
        for idx, ref in enumerate(enc_high.actions)
        if dec_high.candidates[ref.representative_position].branch == "rest"
    )
    n_cands = enc_low.snapshot.candidate_count
    assert n_cands == enc_high.snapshot.candidate_count

    # Target MPO posteriors: at low HP (16/80), Rest=0.80, Smith=0.20; at high HP (68/80), Rest=0.10, Smith=0.90.
    q_low = torch.full((n_cands,), 0.20 / max(1, n_cands - 1), dtype=torch.float32)
    q_low[rest_idx_low] = 0.80
    q_high = torch.full((n_cands,), 0.90 / max(1, n_cands - 1), dtype=torch.float32)
    q_high[rest_idx_high] = 0.10
    q_targets = torch.stack([q_low, q_high], dim=0)

    opt = torch.optim.AdamW(model.parameters(), lr=1e-3)
    for _ in range(12):
        opt.zero_grad(set_to_none=True)
        out = model(batch, validate=False)
        lp = torch.log_softmax(out.policy_logits.masked_fill(~batch.candidates.action_mask, -torch.inf), dim=-1)
        safe_lp = lp.masked_fill(~batch.candidates.action_mask, 0.0)
        loss = -(q_targets * safe_lp).sum(dim=-1).mean()
        loss.backward()
        opt.step()

    model.eval()
    with torch.no_grad():
        final_out = model(batch, validate=False)
        probs = torch.softmax(
            final_out.policy_logits.masked_fill(~batch.candidates.action_mask, -torch.inf), dim=-1
        )
        p_rest_low = float(probs[0, rest_idx_low].item())
        p_rest_high = float(probs[1, rest_idx_high].item())
    assert p_rest_low > 0.65
    assert p_rest_high < 0.25
    assert p_rest_low - p_rest_high > 0.40


def test_v4_2_dominated_loot_leave_suppression_and_within_branch_pg_entropy() -> None:
    import numpy as np
    from torch.nn import functional as F

    from sts2_rl.joint.authority import HybridCollectionAuthority
    from sts2_rl.joint.macro_credit_learner import hierarchical_macro_entropy
    from sts2_rl.macro.authority import MacroCollectionAuthority

    # 1. Authority suppresses premature reward:leave whenever strictly free loot exists
    #    (gold / card / special_card / relic / potion with open belt slots), while retaining
    #    leave on full potion belt, empty loot screen, card_reward skip, and shop leave.
    reward_obs_open = {
        "phase": "rewards",
        "decision_domain": "rewards",
        "player": {
            "character_id": "IRONCLAD",
            "hp": 60,
            "max_hp": 80,
            "gold": 110,
            "open_potion_slots": 1,
            "deck": _sample_deck(),
            "relics": [],
            "potions": [],
        },
        "rewards": {"items": [{"index": 0, "type": "Gold", "reward_type": "Gold"}], "proceed_visible": True},
    }
    reward_legal = [
        {"action_id": "c_gold", "kind": "claim_reward", "model_action_kind": "reward", "index": 0, "is_enabled": True, "reward": {"index": 0, "reward_type": "Gold", "amount": 25}},
        {"action_id": "c_card", "kind": "claim_reward", "model_action_kind": "reward", "index": 1, "is_enabled": True, "reward": {"index": 1, "reward_type": "Card"}},
        {"action_id": "leave", "kind": "proceed", "model_action_kind": "proceed", "index": 2, "is_enabled": True},
    ]
    reward_dec = forward_decision(reward_obs_open, reward_legal, control_domain="macro")
    assert reward_dec is not None
    assert reward_dec.branches == ("claim_gold", "claim_card", "leave")
    filtered_open = MacroCollectionAuthority._filter_dominated_loot_leave(
        reward_dec, list(reward_dec.candidates), reward_obs_open
    )
    assert [c.branch for c in filtered_open] == ["claim_gold", "claim_card"]

    # Potion claim with open_potion_slots >= 1 suppresses leave; open_potion_slots == 0 retains leave.
    potion_legal = [
        {"action_id": "c_pot", "kind": "claim_reward", "model_action_kind": "reward", "index": 0, "is_enabled": True, "reward": {"index": 0, "reward_type": "Potion", "potion": {"id": "POTION.FIRE_POTION"}}},
        {"action_id": "leave", "kind": "proceed", "model_action_kind": "proceed", "index": 1, "is_enabled": True},
    ]
    potion_dec = forward_decision(reward_obs_open, potion_legal, control_domain="macro")
    assert potion_dec is not None
    filtered_pot_open = MacroCollectionAuthority._filter_dominated_loot_leave(
        potion_dec, list(potion_dec.candidates), reward_obs_open
    )
    assert [c.branch for c in filtered_pot_open] == ["claim_potion"]

    reward_obs_full = copy.deepcopy(reward_obs_open)
    reward_obs_full["player"]["open_potion_slots"] = 0  # type: ignore[index]
    filtered_pot_full = MacroCollectionAuthority._filter_dominated_loot_leave(
        potion_dec, list(potion_dec.candidates), reward_obs_full
    )
    assert [c.branch for c in filtered_pot_full] == ["claim_potion", "leave"]

    # Runtime HybridCollectionAuthority.choose drops leave and keeps recorded snapshot aligned!
    cfg = _tiny_model_config()
    encoder = GroundedObservationEncoder(cfg.to_encoding_config())
    seen_snapshots: list[Any] = []

    def _dummy_forward(snapshot: Any, hidden: Any) -> tuple[torch.Tensor, torch.Tensor]:
        seen_snapshots.append(snapshot)
        logits = torch.zeros((snapshot.candidate_count,), dtype=torch.float32)
        logits[-1] = 10.0  # Even if logits favored the last candidate, leave was filtered out.
        return logits, torch.zeros((1, cfg.recurrent_hidden_dim), dtype=torch.float32)

    authority = HybridCollectionAuthority(
        forward_q=_dummy_forward,
        initial_state=lambda: torch.zeros((1, cfg.recurrent_hidden_dim), dtype=torch.float32),
        epsilon=0.0,
        control_domain="joint",
    )
    base_snap = encoder.encode(reward_obs_open, reward_legal).snapshot
    chosen_native = authority.choose(
        observation=reward_obs_open,
        semantic_actions=reward_legal,
        snapshot=base_snap,
        valid=np.ones((len(reward_legal),), dtype=bool),
    )
    assert chosen_native in (0, 1)
    assert len(seen_snapshots) == 1
    assert seen_snapshots[0].candidate_count == 2

    # Card reward skip and shop leave remain untouched.
    card_obs = {
        "phase": "rewards",
        "decision_domain": "card_reward",
        "player": reward_obs_open["player"],
        "rewards": {"cards": [{"id": "CARD.OFFERING", "type": "Skill", "rarity": "Rare", "cost": 0}], "can_skip": True},
    }
    card_legal = [
        {"action_id": "take_0", "kind": "select_card_reward", "model_action_kind": "card_reward", "index": 0, "is_enabled": True, "card": {"id": "CARD.OFFERING", "type": "Skill"}},
        {"action_id": "skip", "kind": "skip_card_reward", "model_action_kind": "card_reward", "index": 1, "is_enabled": True},
    ]
    card_dec = forward_decision(card_obs, card_legal, control_domain="macro")
    assert card_dec is not None
    assert [
        c.branch
        for c in MacroCollectionAuthority._filter_dominated_loot_leave(
            card_dec, list(card_dec.candidates), card_obs
        )
    ] == ["take", "skip"]

    # 2. include_branch_entropy=False yields ZERO gradient on branch logits at every branch probability,
    #    while preserving within-branch exploration gradients for multi-candidate branches.
    for p_branch0 in (0.01, 0.20, 0.50):
        raw_b0 = torch.tensor(float(torch.logit(torch.tensor(p_branch0))), dtype=torch.float64, requires_grad=True)
        within_logits = torch.tensor([1.5, -0.5, -1.0], dtype=torch.float64, requires_grad=True)
        joint_logs = torch.cat(
            [F.logsigmoid(raw_b0).unsqueeze(0), F.logsigmoid(-raw_b0) + torch.log_softmax(within_logits, dim=0)],
            dim=0,
        ).unsqueeze(0)
        masks = torch.ones((1, 4), dtype=torch.bool)
        b_ids = torch.tensor([[0, 1, 1, 1]], dtype=torch.long)
        within_only = hierarchical_macro_entropy(
            joint_logs, masks, b_ids, include_branch_entropy=False
        )[0]
        g_b0, g_w = torch.autograd.grad(within_only, (raw_b0, within_logits))
        assert abs(float(g_b0.item())) < 1e-12
        assert float(g_w[0].item()) < -0.05

    # 3. Single vs batched MacroCreditLearner parity with macro_pg_within_branch_entropy_only=True.
    rew_snap = encoder.encode(reward_obs_open, reward_legal[:2]).snapshot
    combat_obs = {
        "phase": "combat",
        "decision_domain": "combat",
        "run": {"active": True, "act": 1, "floor": 5},
        "player": {"id": "ironclad", "side": "player", "hp": 60, "max_hp": 80, "energy": 3, "hand": [{"id": "strike", "cost": 1}]},
        "combat": {"in_progress": True, "enemies": [{"id": "cultist", "side": "enemy", "hp": 40, "max_hp": 48}]},
    }
    combat_actions = [
        {"action_handle": "c0", "kind": "play_card", "model_action_kind": "play_card", "is_enabled": True, "card": {"id": "strike", "cost": 1}},
        {"action_handle": "c1", "kind": "end_turn", "model_action_kind": "end_turn", "is_enabled": True},
    ]
    combat_snap = encoder.encode(combat_obs, combat_actions).snapshot
    init_model = HybridDomainActorCriticModel(HybridActorCriticConfig.durable_v4(cfg.to_model_config())).eval()
    trajs: list[MacroCreditTrajectory] = []
    with torch.no_grad():
        for idx in range(2):
            b0 = collate_encoded_snapshots((rew_snap,), expected_config=rew_snap.config, expected_fingerprint=rew_snap.encoding_fingerprint, device="cpu")
            o0 = init_model(b0, None, validate=False)
            m0 = torch.tensor(rew_snap.action_mask.tolist(), dtype=torch.bool)
            lp0 = torch.log_softmax(o0.policy_logits[0, : m0.numel()].masked_fill(~m0, -torch.inf), dim=0)
            b1 = collate_encoded_snapshots((combat_snap,), expected_config=combat_snap.config, expected_fingerprint=combat_snap.encoding_fingerprint, device="cpu")
            o1 = init_model(b1, o0.recurrent_state, validate=False)
            m1 = torch.tensor(combat_snap.action_mask.tolist(), dtype=torch.bool)
            lp1 = torch.log_softmax(o1.policy_logits[0, : m1.numel()].masked_fill(~m1, -torch.inf), dim=0)
            s0 = HybridVTraceStep(rew_snap, idx % 2, float(lp0[idx % 2].item()), 0.0, 1.0, False, "rewards", "claim_gold", "macro", 0)
            f0 = MacroRewardFacts(0.0, 0.0, 0.0, 0.0, 0.0, 0, 0.0, 4, 4, 1)
            s1 = HybridVTraceStep(combat_snap, 0, float(lp1[0].item()), 1.95, 0.0, True, "combat", "play_card", "combat", 0)
            f1 = MacroRewardFacts(2.0, 0.0, 0.0, -0.05, 0.0, 0, 0.0, 5, 5, 1)
            trajs.append(
                MacroCreditTrajectory(
                    f"v42-pg-{idx}", 6_700_000 + 2 * idx, 0, (), (s0, s1), None, None, "factual_terminal", (f0, f1)
                )
            )
    cohort = MacroCreditCohort("v42-pg-cohort", 0, (), natural_only=tuple(trajs))
    lcfg_kw = dict(
        target_mode="durable_afterstate_v2",
        enable_direct_vitals_context=True,
        enable_macro_branch_prior=True,
        enable_persistent_only_durable_value=True,
        enable_asymmetric_revival_critic=True,
        revival_budget=32,
        zero_macro_pace_penalty=True,
        clip_macro_pg_fallback=True,
        macro_pg_within_branch_entropy_only=True,
        durable_warmup_updates=0,
        combat_actor_warmup_updates=0,
    )
    m_b = MacroCreditLearner(
        model=copy.deepcopy(init_model),
        config=MacroCreditLearnerConfig(**lcfg_kw, eval_batch_size=8, chunk_batch_size=8),  # type: ignore[arg-type]
    ).learn(cohort)
    m_s = MacroCreditLearner(
        model=copy.deepcopy(init_model),
        config=MacroCreditLearnerConfig(**lcfg_kw, eval_batch_size=1, chunk_batch_size=1),  # type: ignore[arg-type]
    ).learn(cohort)
    assert m_b["macro_entropy"] == pytest.approx(m_s["macro_entropy"], abs=1e-6)
    assert m_b["macro_pg_policy_loss"] == pytest.approx(m_s["macro_pg_policy_loss"], abs=1e-5)
    assert m_b["parameter_delta_norm"] == pytest.approx(m_s["parameter_delta_norm"], abs=1e-5)


def test_v4_3_draw_weighted_durable_value_exact_multiset_split_and_afterstate_invariance() -> None:
    from sts2_rl.macro_competence_probes import probe_multiset_split_invariance
    from sts2_rl.models.hybrid_actor_critic import initialize_actor_critic_from_macro_publication

    torch.manual_seed(20261113)
    cfg = _tiny_model_config()
    encoder = GroundedObservationEncoder(cfg.to_encoding_config())
    v4_model = HybridDomainActorCriticModel(HybridActorCriticConfig.durable_v4(cfg.to_model_config())).eval()
    v5_model = HybridDomainActorCriticModel(HybridActorCriticConfig.durable_v5(cfg.to_model_config())).eval()
    # Randomize non-zero readout heads on both v4 and v5 so every pathway is actively exercised.
    with torch.no_grad():
        for module in (v4_model.durable_value_head, v4_model.durable_vitals_head):
            for param in module.parameters():
                param.normal_(0.0, 0.25)
    initialize_actor_critic_from_macro_publication(v5_model, v4_model.state_dict())
    with torch.no_grad():
        for module in (
            v5_model.durable_deck_raw_fc,
            v5_model.durable_deck_out,
            v5_model.durable_deck_size_head,
            v5_model.durable_deck_size_linear,
            v5_model.persistent_build_deck_proj,
        ):
            for param in module.parameters():
                param.normal_(0.0, 0.25)

    strike = {
        "id": "CARD.STRIKE_IRONCLAD",
        "type": "Attack",
        "rarity": "Basic",
        "cost": 1,
        "is_upgradable": True,
        "is_removable": True,
        "dynamic_vars": {"Damage": 6},
        "keywords": ["strike", "retain"],
        "traits": ["starter"],
        "enchantments": [{"id": "SHARP", "amount": 2}],
        "upgrade_preview": {
            "id": "CARD.STRIKE_IRONCLAD",
            "type": "Attack",
            "rarity": "Basic",
            "cost": 1,
            "upgraded": True,
            "dynamic_vars": {"Damage": 9},
            "keywords": ["strike", "retain"],
            "traits": ["starter"],
            "enchantments": [{"id": "SHARP", "amount": 2}],
        },
    }
    defend = {
        "id": "CARD.DEFEND_IRONCLAD",
        "type": "Skill",
        "rarity": "Basic",
        "cost": 1,
        "is_upgradable": True,
        "is_removable": True,
        "dynamic_vars": {"Block": 5},
    }

    deck_grouped = [copy.deepcopy(strike) for _ in range(5)] + [copy.deepcopy(defend) for _ in range(4)]
    s_inst = copy.deepcopy(strike)
    s_inst["instance_id"] = "inst_42"
    s_inst["floor_added_to_deck"] = 5
    deck_split2 = [copy.deepcopy(strike) for _ in range(4)] + [s_inst] + [copy.deepcopy(defend) for _ in range(4)]

    s_f0 = copy.deepcopy(strike)
    s_f0["floor_added_to_deck"] = 0
    s_f2 = copy.deepcopy(strike)
    s_f2["floor_added_to_deck"] = 2
    s_f6 = copy.deepcopy(strike)
    s_f6["instance_id"] = "inst_99"
    s_f6["floor_added_to_deck"] = 6
    d_inst = copy.deepcopy(defend)
    d_inst["instance_id"] = "def_7"
    d_inst["floor_added_to_deck"] = 3
    deck_split3 = (
        [copy.deepcopy(s_f0) for _ in range(2)]
        + [copy.deepcopy(s_f2) for _ in range(2)]
        + [s_f6]
        + [copy.deepcopy(defend) for _ in range(3)]
        + [d_inst]
    )

    snaps = [encoder.encode_world_snapshot(_rest_surface(d)[0]) for d in (deck_grouped, deck_split2, deck_split3)]
    wb, dom = collate_encoded_worlds(
        snaps,
        expected_config=snaps[0].config,
        expected_fingerprint=snaps[0].encoding_fingerprint,
        device="cpu",
    )
    with torch.no_grad():
        vals_v4, _ = v4_model.forward_durable_value(wb, dom, validate=False)
        vals_v5, run_v5 = v5_model.forward_durable_value(wb, dom, validate=False)

    # v4 suffers orderless multiset-row splitting drift, while v5 is invariant to < 1e-6!
    assert abs(float(vals_v4[0].item()) - float(vals_v4[1].item())) > 1e-4
    assert abs(float(vals_v5[0].item()) - float(vals_v5[1].item())) < 1e-6
    assert abs(float(vals_v5[0].item()) - float(vals_v5[2].item())) < 1e-6
    assert float((run_v5[0] - run_v5[1]).abs().max().item()) < 1e-6
    assert float((run_v5[0] - run_v5[2]).abs().max().item()) < 1e-6

    probe_v5 = probe_multiset_split_invariance(v5_model, encoder)
    assert probe_v5["durable_enabled"] is True
    assert float(probe_v5["grouped_vs_split_abs_diff"]) < 1e-6

    # Afterstate W_post from card_reward:take (no instance_id / floor_added_to_deck)
    # matches real next-floor observation s_{t+1} (with instance_id & floor_added_to_deck) to < 1e-6!
    inflame = {
        "id": "CARD.INFLAME",
        "type": "Power",
        "rarity": "Uncommon",
        "cost": 1,
        "is_upgradable": True,
        "is_removable": True,
        "dynamic_vars": {"Strength": 2},
    }
    rew_obs = {
        "phase": "rewards",
        "decision_domain": "card_reward",
        "run": {"active": True, "act": 1, "floor": 6, "ascension": 0},
        "player": {
            "character_id": "IRONCLAD",
            "hp": 50,
            "max_hp": 80,
            "gold": 120,
            "open_potion_slots": 2,
            "deck": copy.deepcopy(deck_grouped),
            "relics": [{"id": "BURNING_BLOOD", "has_upon_pickup_effect": False}],
            "potions": [],
        },
        "rewards": {"cards": [copy.deepcopy(inflame)], "can_skip": True},
    }
    rew_legal = [
        {
            "action_id": "take_0",
            "kind": "select_card_reward",
            "model_action_kind": "card_reward",
            "index": 0,
            "is_enabled": True,
            "card": copy.deepcopy(inflame),
        },
        {
            "action_id": "skip",
            "kind": "skip_card_reward",
            "model_action_kind": "card_reward",
            "index": 1,
            "is_enabled": True,
        },
    ]
    rew_dec = forward_decision(rew_obs, rew_legal, control_domain="macro")
    assert rew_dec is not None
    post_worlds = synthesize_deterministic_afterstates(
        encoder=encoder,
        observation=rew_obs,
        decision=rew_dec,
        candidates=rew_dec.candidates,
    )
    assert post_worlds is not None
    inflame_real = copy.deepcopy(inflame)
    inflame_real["instance_id"] = "reward_inst_999"
    inflame_real["floor_added_to_deck"] = 6
    real_next_obs = _rest_surface([*copy.deepcopy(deck_split2), inflame_real])[0]
    real_next_obs["player"]["hp"] = 50
    real_next_obs["player"]["gold"] = 120
    real_next_snap = encoder.encode_world_snapshot(real_next_obs)
    wb_cmp, dom_cmp = collate_encoded_worlds(
        (post_worlds[0], real_next_snap),
        expected_config=post_worlds[0].config,
        expected_fingerprint=post_worlds[0].encoding_fingerprint,
        device="cpu",
    )
    with torch.no_grad():
        v_cmp, _ = v5_model.forward_durable_value(wb_cmp, dom_cmp, validate=False)
    assert abs(float(v_cmp[0].item()) - float(v_cmp[1].item())) < 1e-6


def test_v4_3_draw_thinning_algebra_and_role_conditioned_smith_vs_remove_polarity() -> None:
    from sts2_rl.macro_competence_probes import (
        _build_rest_probe_surface,
        _build_shop_remove_probe_surface,
        default_probe_decks,
        probe_smith_vs_remove_divergence,
    )
    from sts2_rl.train_macro_credit import _actor_critic_config

    torch.manual_seed(77)
    cfg = _tiny_model_config()
    encoder = GroundedObservationEncoder(cfg.to_encoding_config())
    model = HybridDomainActorCriticModel(HybridActorCriticConfig.durable_v5(cfg.to_model_config()))
    opt = torch.optim.AdamW(model.parameters(), lr=1.5e-3)

    # 1. Train durable_v5 on factual build outcomes where high_scale/cost_drop > basic_d > basic_a
    #    and verify exact deck-thinning, deck-bloat, & stack-neutral Smith ordering!
    multiset_deck = copy.deepcopy(default_probe_decks()[-1])
    rest_obs, rest_legal = _build_rest_probe_surface(multiset_deck, hp=60, max_hp=80)
    shop_obs, shop_legal = _build_shop_remove_probe_surface(multiset_deck, hp=60, max_hp=80)
    rest_dec = forward_decision(rest_obs, rest_legal, control_domain="macro")
    shop_dec = forward_decision(shop_obs, shop_legal, control_domain="macro")
    assert rest_dec is not None and shop_dec is not None
    rest_enc = encoder.encode(rest_obs, [c.semantic_action for c in rest_dec.candidates])
    shop_enc = encoder.encode(shop_obs, [c.semantic_action for c in shop_dec.candidates])
    rest_grouped = tuple(rest_dec.candidates[r.representative_position] for r in rest_enc.actions)
    shop_grouped = tuple(shop_dec.candidates[r.representative_position] for r in shop_enc.actions)

    # Pure card-utility probe worlds (fixing gold=120 to isolate pure deck composition & size)
    smith_posts = synthesize_deterministic_afterstates(
        encoder=encoder, observation=rest_obs, decision=rest_dec, candidates=rest_grouped
    )
    assert smith_posts is not None
    card_worth = {
        "probe_card_basic_a": (-0.6, -0.35),
        "probe_card_basic_d": (-0.4, -0.05),
        "probe_card_cost_drop": (0.8, 2.0),
        "probe_card_high_scale": (1.0, 2.4),
        "probe_card_utility_draw": (0.6, 1.6),
    }

    def deck_expected_worth(cards: list[dict[str, Any]]) -> float:
        vals = [
            card_worth[str(c["id"])][1 if c.get("upgraded") else 0]
            for c in cards
        ]
        return sum(vals) / len(vals)

    # Build training worlds: base deck (N=11), single-card Smith (N=11), Remove (N=10), and Add (N=12)
    train_snaps = [encoder.encode_world_snapshot(rest_obs)]
    train_targets = [deck_expected_worth(multiset_deck)]
    for cand, post_snap in zip(rest_grouped, smith_posts, strict=True):
        if cand.branch == "smith" and isinstance(cand.target, dict):
            cid = str(cand.target["id"])
            mod_deck = copy.deepcopy(multiset_deck)
            base_card_copy: dict[str, Any] | None = None
            for idx_c, c in enumerate(mod_deck):
                if c["id"] == cid and not c.get("upgraded"):
                    base_card_copy = copy.deepcopy(c)
                    mod_deck[idx_c] = copy.deepcopy(c["upgrade_preview"])
                    break
            assert base_card_copy is not None
            train_snaps.append(post_snap)
            train_targets.append(deck_expected_worth(mod_deck))

            thin_deck = copy.deepcopy(multiset_deck)
            for idx_c, c in enumerate(thin_deck):
                if c["id"] == cid:
                    thin_deck.pop(idx_c)
                    break
            thin_obs, _ = _build_rest_probe_surface(thin_deck, hp=60, max_hp=80)
            train_snaps.append(encoder.encode_world_snapshot(thin_obs))
            train_targets.append(deck_expected_worth(thin_deck))

            add_deck = [*copy.deepcopy(multiset_deck), base_card_copy]
            add_obs, _ = _build_rest_probe_surface(add_deck, hp=60, max_hp=80)
            train_snaps.append(encoder.encode_world_snapshot(add_obs))
            train_targets.append(deck_expected_worth(add_deck))

    wb_train, dom_train = collate_encoded_worlds(
        train_snaps,
        expected_config=train_snaps[0].config,
        expected_fingerprint=train_snaps[0].encoding_fingerprint,
        device="cpu",
    )
    y_train = torch.tensor(train_targets, dtype=torch.float32)

    rest_batch = collate_encoded_snapshots(
        (rest_enc.snapshot,),
        expected_config=rest_enc.snapshot.config,
        expected_fingerprint=rest_enc.snapshot.encoding_fingerprint,
        device="cpu",
    )
    shop_batch = collate_encoded_snapshots(
        (shop_enc.snapshot,),
        expected_config=shop_enc.snapshot.config,
        expected_fingerprint=shop_enc.snapshot.encoding_fingerprint,
        device="cpu",
    )

    # Opposite target distributions on the exact same cards for Smith vs Remove!
    smith_target_logits = torch.full((rest_enc.snapshot.candidate_count,), -1e9, dtype=torch.float32)
    for idx_g, cand in enumerate(rest_grouped):
        if cand.branch == "rest":
            smith_target_logits[idx_g] = -1.0
        elif cand.branch == "smith" and isinstance(cand.target, dict):
            base_w, upg_w = card_worth[str(cand.target["id"])]
            smith_target_logits[idx_g] = 4.0 * (upg_w - base_w)
    smith_target_p = torch.softmax(smith_target_logits, dim=0)

    remove_target_logits = torch.full((shop_enc.snapshot.candidate_count,), -1e9, dtype=torch.float32)
    for idx_g, cand in enumerate(shop_grouped):
        if cand.branch == "leave":
            remove_target_logits[idx_g] = -1.0
        elif cand.branch == "remove" and isinstance(cand.target, dict):
            base_w, _ = card_worth[str(cand.target["id"])]
            remove_target_logits[idx_g] = -4.0 * base_w
    remove_target_p = torch.softmax(remove_target_logits, dim=0)

    for _ in range(150):
        opt.zero_grad()
        pred_v, _ = model.forward_durable_value(wb_train, dom_train, validate=False)
        v_loss = torch.nn.functional.mse_loss(pred_v, y_train)
        out_r = model(rest_batch, validate=False)
        out_s = model(shop_batch, validate=False)
        lp_r = torch.log_softmax(out_r.policy_logits[0, : rest_enc.snapshot.candidate_count], dim=0)
        lp_s = torch.log_softmax(out_s.policy_logits[0, : shop_enc.snapshot.candidate_count], dim=0)
        p_loss = -(smith_target_p * lp_r).sum() - (remove_target_p * lp_s).sum()
        (20.0 * v_loss + p_loss).backward()
        opt.step()

    model.eval()
    with torch.no_grad():
        pred_after, _ = model.forward_durable_value(wb_train, dom_train, validate=False)
    v_base = float(pred_after[0].item())
    smith_by_id: dict[str, float] = {}
    remove_by_id: dict[str, float] = {}
    add_by_id: dict[str, float] = {}
    cursor_idx = 1
    for cand in rest_grouped:
        if cand.branch == "smith" and isinstance(cand.target, dict):
            cid = str(cand.target["id"])
            smith_by_id[cid] = float(pred_after[cursor_idx].item())
            remove_by_id[cid] = float(pred_after[cursor_idx + 1].item())
            add_by_id[cid] = float(pred_after[cursor_idx + 2].item())
            cursor_idx += 3

    v_smith_basic_a = smith_by_id["probe_card_basic_a"]
    v_remove_basic_a = remove_by_id["probe_card_basic_a"]
    assert smith_by_id["probe_card_high_scale"] > v_smith_basic_a + 0.05
    assert smith_by_id["probe_card_cost_drop"] > v_smith_basic_a + 0.05
    assert v_remove_basic_a > v_base > remove_by_id["probe_card_high_scale"]
    assert add_by_id["probe_card_high_scale"] > v_base > add_by_id["probe_card_basic_a"]

    div = probe_smith_vs_remove_divergence(model, encoder, decks=(multiset_deck,))
    assert div["mean_smith_remove_spearman"] <= -0.8
    assert div["top1_target_divergence_rate"] == 1.0

    # 2. Verify v9 (b32 and b16) experiment configs and MacroCreditLearner batched parity on durable_v5!
    exp_dir = Path(__file__).resolve().parents[1] / "config/experiments"
    for fname, expected_budget in (
        ("hybrid_macro_credit_v9_durable_afterstate_v43_b32_g4_turbo.toml", 32),
        ("hybrid_macro_credit_v9_durable_afterstate_v43_b16_g4_turbo.toml", 16),
    ):
        c9 = load_macro_training_config(exp_dir / fname)
        assert c9.learner.revival_budget == expected_budget
        assert c9.training.curriculum.revival_budget == expected_budget
        assert c9.learner.enable_draw_weighted_durable_value is True
        assert c9.learner.enable_role_conditioned_candidate_policy is True
        ac9 = _actor_critic_config(c9)
        assert ac9 == HybridActorCriticConfig.durable_v5(c9.training.model.to_model_config())

    # Learner batched vs single parity with durable_v5
    combat_obs = {
        "phase": "combat",
        "decision_domain": "combat",
        "run": {"active": True, "act": 1, "floor": 7},
        "player": {"id": "ironclad", "side": "player", "hp": 60, "max_hp": 80, "energy": 3, "hand": [{"id": "strike", "cost": 1}]},
        "combat": {"in_progress": True, "enemies": [{"id": "cultist", "side": "enemy", "hp": 40, "max_hp": 48}]},
    }
    combat_actions = [
        {"action_handle": "c0", "kind": "play_card", "model_action_kind": "play_card", "is_enabled": True, "card": {"id": "strike", "cost": 1}},
        {"action_handle": "c1", "kind": "end_turn", "model_action_kind": "end_turn", "is_enabled": True},
    ]
    combat_snap = encoder.encode(combat_obs, combat_actions).snapshot
    v5_init = HybridDomainActorCriticModel(HybridActorCriticConfig.durable_v5(cfg.to_model_config())).eval()
    v5_trajs: list[MacroCreditTrajectory] = []
    with torch.no_grad():
        for idx in range(2):
            o0 = v5_init(rest_batch, None, validate=False)
            m0 = torch.tensor(rest_enc.snapshot.action_mask.tolist(), dtype=torch.bool)
            lp0 = torch.log_softmax(o0.policy_logits[0, : m0.numel()].masked_fill(~m0, -torch.inf), dim=0)
            b1 = collate_encoded_snapshots(
                (combat_snap,),
                expected_config=combat_snap.config,
                expected_fingerprint=combat_snap.encoding_fingerprint,
                device="cpu",
            )
            o1 = v5_init(b1, o0.recurrent_state, validate=False)
            m1 = torch.tensor(combat_snap.action_mask.tolist(), dtype=torch.bool)
            lp1 = torch.log_softmax(o1.policy_logits[0, : m1.numel()].masked_fill(~m1, -torch.inf), dim=0)
            chosen0 = idx % int(m0.sum().item())
            prior_lp = macro_mixture_log_probs(lp0, rest_enc.snapshot, 0.04)
            node0 = MacroAfterstateNode(
                step_index=0,
                surface="rest_site",
                candidate_branches=tuple(c.branch for c in rest_grouped),
                post_worlds=tuple(smith_posts),
            )
            s0 = HybridVTraceStep(
                rest_enc.snapshot,
                chosen0,
                float(prior_lp[chosen0].item()),
                0.0,
                1.0,
                False,
                "rest_site",
                rest_grouped[chosen0].branch,
                "macro",
                0,
            )
            f0 = MacroRewardFacts(0.0, 0.0, 0.0, 0.0, 0.0, 0, 0.0, 6, 6, 1)
            s1 = HybridVTraceStep(
                combat_snap, 0, float(lp1[0].item()), 1.95, 0.0, True, "combat", "play_card", "combat", 0
            )
            f1 = MacroRewardFacts(2.0, 0.0, 0.0, -0.05, 0.0, 0, 0.0, 7, 7, 1)
            v5_trajs.append(
                MacroCreditTrajectory(
                    f"v43-{idx}",
                    6_700_100 + 2 * idx,
                    0,
                    (),
                    (s0, s1),
                    None,
                    None,
                    "factual_terminal",
                    (f0, f1),
                    macro_branch_epsilon=0.04,
                    afterstate_nodes=(node0,),
                )
            )
    v5_cohort = MacroCreditCohort("v43-cohort", 0, (), natural_only=tuple(v5_trajs))
    v5_lcfg = dict(
        target_mode="durable_afterstate_v2",
        enable_direct_vitals_context=True,
        enable_macro_branch_prior=True,
        enable_persistent_only_durable_value=True,
        enable_asymmetric_revival_critic=True,
        enable_draw_weighted_durable_value=True,
        enable_role_conditioned_candidate_policy=True,
        revival_budget=32,
        zero_macro_pace_penalty=True,
        clip_macro_pg_fallback=True,
        macro_pg_within_branch_entropy_only=True,
        separate_value_grad_clip=True,
        durable_warmup_updates=0,
        combat_actor_warmup_updates=0,
    )
    m5_b = MacroCreditLearner(
        model=copy.deepcopy(v5_init),
        config=MacroCreditLearnerConfig(**v5_lcfg, eval_batch_size=8, chunk_batch_size=8),  # type: ignore[arg-type]
    ).learn(v5_cohort)
    m5_s = MacroCreditLearner(
        model=copy.deepcopy(v5_init),
        config=MacroCreditLearnerConfig(**v5_lcfg, eval_batch_size=1, chunk_batch_size=1),  # type: ignore[arg-type]
    ).learn(v5_cohort)
    assert m5_b["durable_value_loss"] == pytest.approx(m5_s["durable_value_loss"], abs=1e-5)
    assert m5_b["macro_policy_loss"] == pytest.approx(m5_s["macro_policy_loss"], abs=1e-5)
    assert m5_b["parameter_delta_norm"] == pytest.approx(m5_s["parameter_delta_norm"], abs=1e-5)
    assert m5_b["parameter_delta_norm"] > 0.0


def test_v4_3_persistent_token_compaction_and_skip_durable_forward() -> None:
    """Verify persistent token compaction shrinks transient world bloat and compute_durable_value=False is exact."""
    from sts2_rl.models.grounded_candidate import WorldTokenBatch

    torch.manual_seed(20261114)
    cfg = _tiny_model_config()
    encoder = GroundedObservationEncoder(cfg.to_encoding_config())
    model = HybridDomainActorCriticModel(
        HybridActorCriticConfig.durable_v5(cfg.to_model_config())
    ).eval()
    with torch.no_grad():
        for mod in (
            model.durable_deck_out,
            model.durable_deck_size_linear,
            model.candidate_role_policy_head,
        ):
            for p in mod.parameters():
                p.add_(0.05 * torch.randn_like(p))

    obs_rest, actions_rest = _rest_surface(_sample_deck())
    dec_rest = forward_decision(obs_rest, actions_rest, control_domain="macro")
    assert dec_rest is not None
    enc_rest = encoder.encode(obs_rest, [cand.semantic_action for cand in dec_rest.candidates])
    batch = collate_encoded_snapshots(
        (enc_rest.snapshot,),
        expected_config=enc_rest.snapshot.config,
        expected_fingerprint=enc_rest.snapshot.encoding_fingerprint,
        device="cpu",
    )

    out_full = model(batch, compute_durable_value=True, validate=True)
    out_fast = model(batch, compute_durable_value=False, validate=True)
    assert out_full.durable_value is not None
    assert out_fast.durable_value is None
    assert torch.equal(out_full.policy_logits, out_fast.policy_logits)
    assert torch.equal(out_full.value, out_fast.value)
    assert torch.equal(out_full.recurrent_state, out_fast.recurrent_state)

    wb = batch.world
    bsz, w0, fdim = wb.features.shape
    bloat = 180
    pad_world = WorldTokenBatch(
        features=torch.cat([wb.features, torch.randn(bsz, bloat, fdim)], dim=1),
        type_ids=torch.cat([wb.type_ids, torch.ones(bsz, bloat, dtype=torch.long)], dim=1),
        role_ids=torch.cat([wb.role_ids, torch.zeros(bsz, bloat, dtype=torch.long)], dim=1),
        owner_ids=torch.cat([wb.owner_ids, torch.full((bsz, bloat), 2, dtype=torch.long)], dim=1),
        entity_ids=torch.cat([wb.entity_ids, torch.ones(bsz, bloat, dtype=torch.long)], dim=1),
        entity_aux_ids=torch.cat([wb.entity_aux_ids, torch.zeros(bsz, bloat, dtype=torch.long)], dim=1),
        definition_binding_ids=torch.cat([wb.definition_binding_ids, torch.zeros(bsz, bloat, dtype=torch.long)], dim=1),
        relation_binding_ids=torch.cat([wb.relation_binding_ids, torch.zeros(bsz, bloat, dtype=torch.long)], dim=1),
        zone_ids=torch.cat([wb.zone_ids, torch.full((bsz, bloat), 20, dtype=torch.long)], dim=1),
        order_ids=torch.cat([wb.order_ids, torch.zeros(bsz, bloat, dtype=torch.long)], dim=1),
        mask=torch.cat([wb.mask, torch.ones(bsz, bloat, dtype=torch.bool)], dim=1),
    )
    prep_pad = model._prepare_draw_weighted_persistent_world(pad_world)
    assert prep_pad.world.features.shape[1] <= w0

    model_a = copy.deepcopy(model)
    model_b = copy.deepcopy(model)
    dom0 = torch.zeros((bsz,), dtype=torch.long)
    val_a, _ = model_a.forward_durable_value(wb, dom0, validate=False)
    val_b, _ = model_b.forward_durable_value(pad_world, dom0, validate=False)
    assert float((val_a - val_b).abs().max().item()) < 1e-6
    val_a.sum().backward()  # type: ignore[no-untyped-call]
    val_b.sum().backward()  # type: ignore[no-untyped-call]
    for (na, pa), (_, pb) in zip(model_a.named_parameters(), model_b.named_parameters(), strict=True):
        if pa.grad is not None or pb.grad is not None:
            assert pa.grad is not None and pb.grad is not None, na
            assert float((pa.grad - pb.grad).abs().max().item()) < 1e-5, na

