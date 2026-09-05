from __future__ import annotations

import copy
import json
import math
from dataclasses import replace
from typing import Any

import pytest

torch = pytest.importorskip("torch")

import sts2_rl.joint.macro_credit_learner as learner_module  # noqa: E402
from sts2_rl.encoding import GroundedObservationEncoder  # noqa: E402
from sts2_rl.joint.macro_credit_data import (  # noqa: E402
    MacroCounterfactualRoot,
    MacroCreditCohort,
    MacroCreditTrajectory,
    MacroRewardFacts,
)
from sts2_rl.joint.macro_credit_learner import MacroCreditLearner, MacroCreditLearnerConfig  # noqa: E402
from sts2_rl.joint.transitions import HybridVTraceStep  # noqa: E402
from sts2_rl.models import HybridActorCriticConfig, HybridDomainActorCriticModel  # noqa: E402
from sts2_rl.training.config import ModelConfig  # noqa: E402
from tests.test_macro_authority import _REST_ACTIONS, _rest_observation, _snapshot  # noqa: E402


class _Tiny(torch.nn.Module):
    def __init__(self, count=2, *, recurrent=False):
        super().__init__()
        self.candidate_policy_head = torch.nn.Parameter(torch.linspace(0.6, -0.3, count))
        self.state_value_head = torch.nn.Parameter(torch.tensor(0.25))
        self.memory = torch.nn.Parameter(torch.tensor(0.1))
        self.recurrent = recurrent
        self.calls = []
        self.inject_nan = False

    def call(self, snapshot, hidden):
        if hidden is None:
            hidden = self.memory.new_zeros((1, 1))
        count = snapshot.candidate_count
        next_hidden = hidden + self.memory
        logits = self.candidate_policy_head[:count]
        value = self.state_value_head
        if self.recurrent:
            logits = logits + hidden.sum() * torch.arange(count, dtype=logits.dtype) * 0.03
            value = value + 0.1 * hidden.sum()
        if self.inject_nan:
            value = value * math.nan
        depth = getattr(hidden, "test_depth", 0) + 1 if torch.is_grad_enabled() else 0
        next_hidden.test_depth = depth
        self.calls.append({"grad": torch.is_grad_enabled(), "depth": depth, "hidden": hidden.detach().clone(),
                           "parameters": self.candidate_policy_head.detach().clone()})
        return logits, value, next_hidden


def _config(**kwargs):
    return replace(MacroCreditLearnerConfig(weight_decay=0.0, gradient_clip_norm=100.0), **kwargs)


def _learner(model=None, **kwargs):
    model = model or _Tiny()
    return MacroCreditLearner(model=model, forward=model.call, config=_config(**kwargs))


def _facts(reward):
    return MacroRewardFacts(
        terminal_reward=float(reward), progress_reward=0.0, revival_penalty=0.0,
        pace_penalty=0.0, hp_loss_penalty=0.0, revivals_used_delta=0, player_hp_lost_delta=0.0,
        floor_before=0, floor_after=0, native_steps=1,
    )


def _root(
    learner, *, outcomes=(0.0, 1.0), domains=("macro", "combat", "macro"), root_index=0,
    seed=2, terminal=True, final_reward=0.0, outcomes_at_end=False, snapshots=None,
):
    count = len(outcomes)
    if snapshots is None:
        combat = _snapshot(candidate_count=count)
        macro = replace(combat, domain_id=2)
        snapshots = tuple(macro if domain == "macro" else combat for domain in domains)
    bootstrap = snapshots[root_index] if not terminal else None

    def trajectory(candidate, *, natural):
        prefix = () if natural else snapshots[:root_index]
        local_snapshots = snapshots if natural else snapshots[root_index:]
        local_domains = domains if natural else domains[root_index:]
        steps, facts = [], []
        full_root_logs = None
        hidden = None
        with torch.no_grad():
            for snapshot in prefix:
                _, _, hidden = learner._forward(snapshot, hidden)
            for i, (snapshot, domain) in enumerate(zip(local_snapshots, local_domains, strict=True)):
                is_root = i == (root_index if natural else 0)
                action = candidate if is_root else 0
                logits, _, hidden = learner._forward(snapshot, hidden)
                logs = torch.log_softmax(logits[:count], dim=0)
                if is_root:
                    full_root_logs = tuple(float(item) for item in logs.tolist())
                end = i == len(local_snapshots) - 1
                reward = final_reward if end else 0.0
                if (end if outcomes_at_end else is_root):
                    reward += outcomes[candidate]
                behavior = 0.0 if is_root and not natural else float(logs[action].item())
                steps.append(HybridVTraceStep(
                    snapshot=snapshot, action_index=action, behavior_log_probability=behavior,
                    reward=reward, discount=0.0 if end and terminal else 1.0, terminal=end and terminal,
                    surface="rest" if domain == "macro" else "combat", branch=f"candidate-{action}",
                    control_domain=domain, policy_version=learner.update,
                ))
                facts.append(_facts(reward))
        return MacroCreditTrajectory(
            trajectory_id=f"seed-{seed}-{'reference' if natural else candidate}", seed=seed,
            teacher_version=learner.update, prefix=tuple(prefix), steps=tuple(steps),
            bootstrap_snapshot=bootstrap, bootstrap_domain=None if terminal else "macro",
            ending="factual_terminal" if terminal else "macro_horizon", reward_facts=tuple(facts),
            root_intervened=not natural,
        ), full_root_logs

    reference, root_logs = trajectory(0, natural=True)
    reused = replace(reference, trajectory_id=f"seed-{seed}-natural-suffix", prefix=tuple(snapshots[:root_index]),
                     steps=reference.steps[root_index:], reward_facts=reference.reward_facts[root_index:])
    candidates = (reused, *(trajectory(index, natural=False)[0] for index in range(1, count)))
    return MacroCounterfactualRoot(
        root_id=f"root-{seed}", reference=reference, root_step_index=root_index, candidates=candidates,
        candidate_branches=tuple(f"candidate-{index}" for index in range(count)),
        teacher_log_probabilities=root_logs, same_action_gate="matched", evidence_directory=f"/external/root-{seed}",
    )


def _cohort(learner, *roots):
    return MacroCreditCohort(cohort_id=f"cohort-{learner.update}", teacher_version=learner.update,
                             roots=roots or (_root(learner),))


def test_one_finite_update_routes_only_natural_data_and_emits_json_metrics():
    learner = _learner()
    cohort = _cohort(learner)
    learner.model.train()
    metrics = learner.learn(cohort)
    assert learner.update == learner.policy_version == 1
    assert not learner.model.training
    assert metrics["teacher_version"] == 0 and metrics["policy_lag"] == 0
    assert metrics["root_count"] == 1
    assert metrics["macro_steps"] == 2 and metrics["combat_steps"] == 1
    assert metrics["training_forward_calls"] == 3
    assert metrics["gradient_norm"] > 0 and metrics["parameter_delta_norm"] > 0
    assert metrics["macro_policy_roots"] == 1
    assert metrics["peak_cuda_allocated_bytes"] == 0
    assert metrics["process_peak_rss_bytes"] is None or metrics["process_peak_rss_bytes"] > 0
    assert metrics["root_targets"][0]["updated_policy_kl_to_teacher"] >= 0
    json.dumps(metrics, allow_nan=False)
    assert metrics["loss"] == pytest.approx(0.5 * (metrics["macro_loss"] + metrics["combat_loss"]))


def test_macro_actor_is_full_root_projection_with_all_root_denominator():
    learner = _learner(macro_value_weight=0, combat_policy_weight=0, combat_value_weight=0, combat_entropy_weight=0)
    roots = (_root(learner, outcomes=(0.0, 1.0), seed=2), _root(learner, outcomes=(2.0, 2.0), seed=4))
    metrics = learner.learn(_cohort(learner, *roots))
    first = metrics["root_targets"][0]
    prior = torch.tensor(first["teacher_probabilities"])
    q = torch.tensor(first["target_probabilities"])
    torch.testing.assert_close(learner.model.candidate_policy_head.grad, 0.5 * (prior - q) / 2, atol=1e-7, rtol=1e-5)
    assert metrics["macro_policy_roots"] == 1
    assert metrics["root_count"] == 2
    assert metrics["macro_policy_loss"] == pytest.approx(float(-(q * prior.log()).sum()) / 2)


def test_candidate_count_does_not_multiply_root_actor_weight():
    learner = _learner(_Tiny(3), macro_value_weight=0, combat_policy_weight=0, combat_value_weight=0, combat_entropy_weight=0)
    roots = (_root(learner, outcomes=(0.0, 1.0), seed=2), _root(learner, outcomes=(0.0, 1.0, 2.0), seed=4))
    metrics = learner.learn(_cohort(learner, *roots))
    expected = torch.zeros(3)
    for row in metrics["root_targets"]:
        prior, q = torch.tensor(row["teacher_probabilities"]), torch.tensor(row["target_probabilities"])
        expected[:len(prior)] += 0.5 * (prior - q) / 2
    torch.testing.assert_close(learner.model.candidate_policy_head.grad, expected, atol=1e-7, rtol=1e-5)


def test_equal_root_outcomes_create_no_actor_or_cloning_gradient():
    learner = _learner(macro_value_weight=0, combat_policy_weight=0, combat_value_weight=0, combat_entropy_weight=0)
    before = copy.deepcopy(learner.model.state_dict())
    metrics = learner.learn(_cohort(learner, _root(learner, outcomes=(2.0, 2.0))))
    assert metrics["macro_policy_roots"] == 0
    assert metrics["macro_policy_loss"] == metrics["gradient_norm"] == metrics["parameter_delta_norm"] == 0
    assert learner.model.candidate_policy_head.grad is None
    for key, value in learner.model.state_dict().items():
        torch.testing.assert_close(value, before[key], rtol=0, atol=0)


def test_root_value_replaces_natural_target_and_uses_teacher_not_q():
    learner = _learner(macro_trace_lambda=1, macro_policy_weight=0, combat_policy_weight=0,
                       combat_value_weight=0, combat_entropy_weight=0)
    root = _root(learner, outcomes=(0.0, 2.0), domains=("macro", "macro", "macro"))
    metrics = learner.learn(_cohort(learner, root))
    row = metrics["root_targets"][0]
    teacher_target = 2 * row["teacher_probabilities"][1]
    assert row["value_target"] == pytest.approx(teacher_target)
    assert teacher_target != pytest.approx(2 * row["target_probabilities"][1])
    expected_loss = torch.nn.functional.smooth_l1_loss(torch.full((3,), 0.25), torch.tensor([teacher_target, 0.0, 0.0]))
    assert metrics["macro_value_loss"] == pytest.approx(expected_loss.item())
    expected_gradient = 0.5 * learner.config.macro_value_weight * (max(-1, min(1, 0.25 - teacher_target)) + 0.5) / 3
    assert learner.model.state_value_head.grad.item() == pytest.approx(expected_gradient, abs=1e-7)
    assert metrics["combat_steps"] == 0


def test_forced_low_probability_root_uses_conditional_y_without_own_rho():
    model = _Tiny()
    model.candidate_policy_head.data.copy_(torch.tensor([0.0, -8.0]))
    learner = _learner(model, macro_trace_lambda=1)
    metrics = learner.learn(_cohort(learner, _root(learner, outcomes=(0.0, 10.0))))
    row = metrics["root_targets"][0]
    assert row["action_values"] == pytest.approx([0.0, 10.0])
    assert row["value_target"] == pytest.approx(10 * row["teacher_probabilities"][1])
    assert row["target_probabilities"][1] > row["teacher_probabilities"][1]


def test_counterfactual_combat_suffix_is_not_an_ordinary_training_example():
    model = _Tiny()
    model.state_value_head.data.zero_()
    learner = _learner(model, macro_trace_lambda=1, macro_policy_weight=0, macro_value_weight=0, combat_entropy_weight=0)
    root = _root(learner, outcomes=(0.0, 10.0), outcomes_at_end=True, domains=("macro", "combat", "combat"))
    metrics = learner.learn(_cohort(learner, root))
    assert metrics["root_targets"][0]["action_values"] == pytest.approx([0.0, 10.0])
    assert metrics["combat_steps"] == 2
    assert metrics["combat_policy_loss"] == metrics["combat_value_loss"] == metrics["parameter_delta_norm"] == 0


def test_combat_policy_uses_choice_denominator_while_value_counts_all_combat_rows():
    learner = _learner(macro_policy_weight=0, macro_value_weight=0, combat_value_weight=0, combat_entropy_weight=0)
    combat = _snapshot(candidate_count=2)
    macro = replace(combat, domain_id=2)
    only_action = _snapshot(candidate_count=1)
    root = _root(learner, domains=("macro", "combat", "combat"), snapshots=(macro, only_action, combat), final_reward=1.0)
    old_policy = learner.model.candidate_policy_head.detach().softmax(dim=0)
    metrics = learner.learn(_cohort(learner, root))
    assert metrics["combat_steps"] == 2 and metrics["combat_policy_steps"] == 1
    expected = 0.5 * 0.75 * (old_policy - torch.tensor([1.0, 0.0]))
    torch.testing.assert_close(learner.model.candidate_policy_head.grad, expected)


def test_combat_targets_are_short_mixed_unrolls_not_long_macro_targets():
    learner = _learner(macro_trace_lambda=1)
    domains = ("macro", *("combat",) * 18, "macro")
    root = _root(learner, outcomes=(0.0, 0.0), domains=domains, final_reward=4.0)
    prepared, _, _ = learner._prepare(root)
    torch.testing.assert_close(prepared.combat_values[:16], torch.full((16,), 0.25))
    torch.testing.assert_close(prepared.combat_values[16:], torch.full((4,), 4.0))
    torch.testing.assert_close(prepared.combat_advantages[:16], torch.zeros(16))
    torch.testing.assert_close(prepared.macro_values, torch.full((20,), 4.0))


def test_chunk_backward_does_not_change_denominators_or_nonrecurrent_gradients():
    first = _learner(chunk_steps=1)
    second = _learner(chunk_steps=16)
    root = _root(first, domains=("macro", "combat", "combat", "macro", "combat", "macro"))
    cohort = _cohort(first, root)
    a, b = first.learn(cohort), second.learn(cohort)
    for name, parameter in first.model.named_parameters():
        other = dict(second.model.named_parameters())[name]
        if parameter.grad is not None:
            torch.testing.assert_close(parameter.grad, other.grad, atol=1e-7, rtol=1e-5)
    assert a["loss"] == pytest.approx(b["loss"], abs=1e-7)
    assert a["backward_chunks"] == 6 and b["backward_chunks"] == 1


def test_current_prefix_memory_is_rebuilt_and_graph_depth_never_exceeds_chunk():
    learner = _learner(_Tiny(recurrent=True), chunk_steps=4)
    domains = ("macro", "combat", "macro", *("combat",) * 15, "macro")
    root = _root(learner, domains=domains, root_index=2)
    learner.model.calls.clear()
    metrics = learner.learn(_cohort(learner, root))
    grad_calls = [call for call in learner.model.calls if call["grad"]]
    assert len(grad_calls) == len(domains)
    assert max(call["depth"] for call in grad_calls) <= 4
    assert metrics["max_autograd_steps"] == 4
    assert all(torch.equal(call["parameters"], grad_calls[0]["parameters"]) for call in grad_calls)
    for index in range(0, len(domains), 4):
        assert grad_calls[index]["hidden"].item() == pytest.approx(index * 0.1, abs=1e-6)
    assert metrics["maximum_likelihood_error"] <= 1e-5


def test_censored_tail_uses_real_bootstrap_value():
    learner = _learner(macro_trace_lambda=1)
    root = _root(learner, outcomes=(0.0, 1.0), domains=("macro", "combat"), terminal=False)
    prepared, _, _ = learner._prepare(root)
    assert prepared.action_values.tolist() == pytest.approx([0.25, 1.25])
    metrics = learner.learn(_cohort(learner, root))
    assert metrics["root_targets"][0]["action_values"] == pytest.approx([0.25, 1.25])


def test_same_cohort_cannot_be_consumed_twice_and_stale_stamp_fails_before_forward():
    learner = _learner()
    cohort = _cohort(learner)
    learner.learn(cohort)
    calls = len(learner.model.calls)
    with pytest.raises(ValueError, match="exact fresh teacher"):
        learner.learn(cohort)
    assert len(learner.model.calls) == calls
    assert learner.update == 1


def test_correct_version_stamp_does_not_hide_stale_behavior_likelihood():
    learner = _learner()
    cohort = _cohort(learner)
    learner.model.candidate_policy_head.data[0] += 0.1
    before = copy.deepcopy(learner.model.state_dict())
    with pytest.raises(ValueError, match="likelihood differs"):
        learner.learn(cohort)
    assert learner.update == 0 and not learner.optimizer.state
    for key, value in learner.model.state_dict().items():
        torch.testing.assert_close(value, before[key], rtol=0, atol=0)


def test_full_root_prior_is_checked_not_only_selected_probability():
    learner = _learner(_Tiny(3))
    root = _root(learner, outcomes=(0.0, 1.0, 2.0))
    old = root.teacher_log_probabilities
    forged = replace(root, teacher_log_probabilities=(old[0], old[2], old[1]))
    with pytest.raises(ValueError, match="full root prior differs"):
        learner.learn(_cohort(learner, forged))


def test_malformed_tail_is_revalidated_before_training():
    learner = _learner()
    root = _root(learner, terminal=False)
    object.__setattr__(root.reference, "bootstrap_snapshot", None)
    with pytest.raises(ValueError, match="ending|bootstrap|macro horizon"):
        learner.learn(_cohort(learner, root))
    assert learner.update == 0 and not learner.optimizer.state


def test_nonfinite_model_and_outputs_fail_before_optimizer():
    learner = _learner()
    cohort = _cohort(learner)
    learner.model.inject_nan = True
    with pytest.raises(FloatingPointError, match="forward is non-finite"):
        learner.learn(cohort)
    learner.model.inject_nan = False
    learner.model.candidate_policy_head.data[0] = math.nan
    with pytest.raises(FloatingPointError, match="before update"):
        learner.learn(cohort)
    assert learner.update == 0 and not learner.optimizer.state


def test_nonfinite_gradient_never_reaches_optimizer_and_poisons_instance():
    learner = _learner()
    cohort = _cohort(learner)
    before = copy.deepcopy(learner.model.state_dict())
    learner.model.candidate_policy_head.register_hook(lambda gradient: gradient * math.nan)
    with pytest.raises(FloatingPointError, match="gradient is non-finite"):
        learner.learn(cohort)
    assert learner.update == 0 and not learner.optimizer.state
    for key, value in learner.model.state_dict().items():
        torch.testing.assert_close(value, before[key], rtol=0, atol=0)
    with pytest.raises(RuntimeError, match="reload"):
        learner.learn(cohort)


def test_nonfinite_optimizer_state_is_rejected_before_next_forward():
    learner = _learner()
    learner.learn(_cohort(learner))
    next_cohort = _cohort(learner, _root(learner, seed=4))
    state = next(iter(learner.optimizer.state.values()))
    state["exp_avg"].fill_(math.nan)
    calls = len(learner.model.calls)
    with pytest.raises(FloatingPointError, match="optimizer state is non-finite"):
        learner.learn(next_cohort)
    assert learner.update == 1 and len(learner.model.calls) == calls


def test_post_step_metrics_failure_requires_boundary_reload(monkeypatch):
    learner = _learner()
    cohort = _cohort(learner)

    def fail():
        raise MemoryError("post-step instrumentation failed")

    monkeypatch.setattr(learner_module, "_rss_peak", fail)
    with pytest.raises(MemoryError, match="post-step"):
        learner.learn(cohort)
    with pytest.raises(RuntimeError, match="reload"):
        learner.learn(cohort)


def test_resource_guard_checks_target_and_chunk_boundaries_before_optimizer():
    learner = _learner(chunk_steps=2)
    cohort = _cohort(learner)
    learner.model.calls.clear()
    before = copy.deepcopy(learner.model.state_dict())
    observed = []

    def guard():
        training_calls = sum(call["grad"] for call in learner.model.calls)
        observed.append(training_calls)
        if training_calls >= 2:
            raise TimeoutError("cohort wall-clock budget")

    with pytest.raises(TimeoutError, match="budget"):
        learner.learn(cohort, resource_guard=guard)
    assert 0 in observed and 2 in observed
    assert learner.update == 0 and not learner.optimizer.state
    for key, value in learner.model.state_dict().items():
        torch.testing.assert_close(value, before[key], rtol=0, atol=0)
    with pytest.raises(RuntimeError, match="reload"):
        learner.learn(cohort)


def test_resource_guard_after_step_cannot_leave_a_reusable_mutated_learner(monkeypatch):
    learner = _learner()
    cohort = _cohort(learner)
    stepped = False
    original = learner.optimizer.step

    def step(*args, **kwargs):
        nonlocal stepped
        result = original(*args, **kwargs)
        stepped = True
        return result

    def guard():
        if stepped:
            raise TimeoutError("post-step budget")

    monkeypatch.setattr(learner.optimizer, "step", step)
    with pytest.raises(TimeoutError, match="post-step"):
        learner.learn(cohort, resource_guard=guard)
    with pytest.raises(RuntimeError, match="reload"):
        learner.learn(cohort)


def test_model_optimizer_update_boundary_is_sufficient_for_next_exact_step():
    first = _learner()
    first.learn(_cohort(first))
    second = _learner()
    second.model.load_state_dict(copy.deepcopy(first.model.state_dict()))
    second.optimizer.load_state_dict(copy.deepcopy(first.optimizer.state_dict()))
    second.update = first.update
    cohort = _cohort(first, _root(first, seed=4))
    a, b = first.learn(cohort), second.learn(cohort)
    assert a["update"] == b["update"] == 2
    for key, value in first.model.state_dict().items():
        torch.testing.assert_close(value, second.model.state_dict()[key], rtol=0, atol=0)


def test_actual_tiny_hybrid_actor_critic_optimizer_without_q_or_training_factory():
    model_config = ModelConfig(d_model=32, n_heads=4, ffn_dim=64, world_layers=1, latent_slots=4, latent_layers=1,
                               recurrent_hidden_dim=64, dropout=0.2, max_world_tokens=256, max_candidates=16,
                               max_candidate_local_tokens=16)
    model = HybridDomainActorCriticModel(HybridActorCriticConfig(model_config.to_model_config()))
    learner = MacroCreditLearner(model=model, config=_config(chunk_steps=2))
    encoder = GroundedObservationEncoder(model_config.to_encoding_config())
    macro = encoder.encode(_rest_observation(), _REST_ACTIONS).snapshot
    combat = encoder.encode(
        {"phase": "combat", "combat": {"in_progress": True}},
        [{"kind": "end_turn", "model_action_kind": "end_turn", "action_handle": "end"},
         {"kind": "play_card", "model_action_kind": "play_card", "action_handle": "play"}],
    ).snapshot
    root = _root(learner, snapshots=(macro, combat, macro))
    metrics = learner.learn(_cohort(learner, root))
    assert metrics["update"] == 1 and metrics["gradient_norm"] > 0 and metrics["parameter_delta_norm"] > 0
    assert metrics["policy_head_gradient_norm"] > 0 and metrics["value_head_gradient_norm"] > 0
    assert metrics["shared_gradient_norm"] > 0
    assert not hasattr(model, "candidate_q_head")
    assert not model.training
    assert metrics["max_autograd_steps"] == 2
    json.dumps(metrics, allow_nan=False)


@pytest.mark.parametrize("field,value", [("chunk_steps", 17), ("combat_unroll_steps", 0), ("chunk_steps", True),
                                        ("macro_trace_lambda", 1.1), ("macro_kl", -1), ("learning_rate", math.inf),
                                        ("macro_policy_weight", math.nan), ("likelihood_tolerance", 0)])
def test_config_rejects_unbounded_or_invalid_settings(field: str, value: Any):
    with pytest.raises(ValueError):
        _config(**{field: value})


def test_learner_refuses_frozen_models_and_q_heads():
    with pytest.raises(ValueError, match="trainable"):
        _learner(_Tiny().requires_grad_(False))
    model = _Tiny()
    model.candidate_q_head = torch.nn.Parameter(torch.tensor(1.0))
    with pytest.raises(ValueError, match="Q head"):
        _learner(model)
