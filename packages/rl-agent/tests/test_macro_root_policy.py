from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path
from typing import Any

import pytest

torch = pytest.importorskip("torch")

from sts2_baseline import (  # noqa: E402
    absolute_death_price_reward_identity,
    act_frontloaded_death_reward_identity,
)
from sts2_rl.encoding import GroundedObservationEncoder  # noqa: E402
from sts2_rl.encoding.snapshot import collate_encoded_snapshots  # noqa: E402
from sts2_rl.macro_root_policy import FrozenMacroRootPolicy  # noqa: E402
from sts2_rl.models import (  # noqa: E402
    HybridActorCriticConfig,
    HybridDomainActorCriticModel,
    HybridDomainIQLModel,
    HybridIQLConfig,
)
from sts2_rl.training import factory  # noqa: E402
from sts2_rl.training.config import ModelConfig, TrainingConfig, load_training_config  # noqa: E402


@pytest.fixture
def config() -> TrainingConfig:
    base = load_training_config(profile="preheat")
    return replace(
        base,
        model=ModelConfig(
            d_model=32,
            n_heads=4,
            ffn_dim=64,
            world_layers=1,
            latent_slots=4,
            latent_layers=1,
            recurrent_hidden_dim=64,
            dropout=0.2,
            max_world_tokens=256,
            max_candidates=16,
            max_candidate_local_tokens=16,
        ),
        curriculum=replace(base.curriculum, revival_budget=64),
    )


def _publication(config: TrainingConfig, *, iql: bool = False) -> tuple[dict[str, Any], Any]:
    grounded = config.model.to_model_config()
    if iql:
        model_config = HybridIQLConfig(grounded)
        model = HybridDomainIQLModel(model_config).eval()
        payload = {
            "format": "sts2-hybrid-publication-v3",
            "model_config": model_config.as_mapping(),
            "reward_identity": absolute_death_price_reward_identity(),
            "control_scores": "policy_logits",
            "policy_version": 100,
            "online": model.state_dict(),
            "death_price": {"lambda_value": 0.01},
        }
    else:
        model_config = HybridActorCriticConfig(grounded)
        model = HybridDomainActorCriticModel(model_config).eval()
        payload = {
            "format": "sts2-hybrid-vtrace-publication-v1",
            "model_config": model_config.as_mapping(),
            "reward_identity": act_frontloaded_death_reward_identity(revival_budget=64),
            "policy_version": 7,
            "online": model.state_dict(),
        }
    return payload, model


def _save(tmp_path: Path, payload: dict[str, Any]) -> Path:
    path = tmp_path / "network.pt"
    torch.save(payload, path)
    return path


def _snapshot(config: TrainingConfig) -> Any:
    return (
        GroundedObservationEncoder(config.model.to_encoding_config())
        .encode(
            {"phase": "combat", "combat": {"in_progress": True}},
            [
                {"kind": "end_turn", "model_action_kind": "end_turn", "action_handle": "end"},
                {"kind": "play_card", "model_action_kind": "play_card", "action_handle": "play"},
            ],
        )
        .snapshot
    )


def test_zero_budget_requires_evaluation_opt_in_and_preserves_training_contract(tmp_path, config):
    from sts2_rl.macro_root_diagnostic import validate_course
    payload, _ = _publication(config, iql=True)
    path = _save(tmp_path, payload)
    zero = replace(config, curriculum=replace(config.curriculum, revival_budget=0))
    with pytest.raises(ValueError, match="positive"):
        FrozenMacroRootPolicy(path, zero, "cpu", initialize_from_iql=True)
    with pytest.raises(ValueError):
        validate_course(zero)
    assert validate_course(zero, allow_zero_budget=True) == 0
    policy = FrozenMacroRootPolicy(path, zero, "cpu", initialize_from_iql=True, evaluation_only_allow_zero=True)
    assert policy.provenance["evaluation_only_zero_budget"] is True
    assert policy.provenance["reward_identity"]["revival_budget"] == 0
    assert all(not p.requires_grad for p in policy.model.parameters())


def test_macro_transfer_rejects_incorrect_source_reward_contract(tmp_path, config):
    payload, _ = _publication(config)
    payload.update(format="sts2-hybrid-macro-credit-publication-v1", control_scores="policy_logits")
    payload["reward_identity"]["tampered"] = True
    with pytest.raises(ValueError, match="source reward_identity"):
        FrozenMacroRootPolicy(_save(tmp_path, payload), config, "cpu", initialize_from_macro_publication=True)


@pytest.mark.parametrize("iql", [False, True])
def test_frozen_policy_matches_source_without_training_resources_or_writes(
    tmp_path: Path, config: TrainingConfig, monkeypatch: pytest.MonkeyPatch, iql: bool
) -> None:
    payload, source = _publication(config, iql=iql)
    path = _save(tmp_path, payload)
    before = path.read_bytes()

    def forbidden(*args: Any, **kwargs: Any) -> None:
        raise AssertionError("read-only policy must not build training resources or optimizer")

    monkeypatch.setattr(factory, "build_training_resources", forbidden)
    monkeypatch.setattr(torch.optim, "AdamW", forbidden)
    policy = FrozenMacroRootPolicy(path, config, "cpu", initialize_from_iql=iql)
    assert policy.encoding_config == config.model.to_encoding_config()
    assert not policy.model.training
    assert all(not parameter.requires_grad for parameter in policy.model.parameters())
    snapshot = _snapshot(config)
    encoded = collate_encoded_snapshots(
        (snapshot,),
        expected_config=policy.encoding_config,
        expected_fingerprint=snapshot.encoding_fingerprint,
        device="cpu",
    )
    with torch.no_grad():
        expected = source(encoded, source.initial_state(1, device="cpu"))
    logits, hidden = policy.forward(snapshot, None)
    repeated_logits, repeated_hidden = policy.forward(snapshot, policy.initial_state())
    torch.testing.assert_close(logits, expected.policy_logits[0])
    torch.testing.assert_close(hidden, expected.recurrent_state)
    torch.testing.assert_close(repeated_logits, logits, rtol=0, atol=0)
    torch.testing.assert_close(repeated_hidden, hidden, rtol=0, atol=0)
    assert logits.ndim == 1
    assert hidden.shape == (1, config.model.recurrent_hidden_dim)
    assert not logits.requires_grad and not hidden.requires_grad
    assert logits.grad_fn is None and hidden.grad_fn is None
    assert path.read_bytes() == before
    assert list(tmp_path.iterdir()) == [path]
    assert policy.provenance["exact_resume"] is False
    assert policy.provenance["training_state_loaded"] is False
    assert policy.provenance["loaded_tensor_count"] == len(policy.model.state_dict())
    assert policy.provenance["loaded_tensor_names"] == sorted(policy.model.state_dict())
    assert policy.provenance["fresh_tensor_names"] == []
    assert policy.provenance["policy_version"] == (100 if iql else 7)
    assert policy.provenance["source_path"] == str(path.resolve())
    if iql:
        assert policy.provenance["initialization_mode"] == "in_memory_model_initialization"
        assert policy.provenance["source_reward_identity"] == absolute_death_price_reward_identity()
        assert policy.provenance["reward_identity"] == act_frontloaded_death_reward_identity(revival_budget=64)
        assert all(name.startswith("candidate_q_head.") for name in policy.provenance["ignored_source_tensor_names"])
    else:
        assert policy.provenance["ignored_source_tensor_names"] == []
    json.dumps(policy.provenance, allow_nan=False)


@pytest.mark.parametrize("iql", [False, True])
def test_publication_kind_cannot_be_silently_reinterpreted(tmp_path: Path, config: TrainingConfig, iql: bool) -> None:
    payload, _ = _publication(config, iql=iql)
    with pytest.raises(ValueError, match="expected.*explicit"):
        FrozenMacroRootPolicy(_save(tmp_path, payload), config, "cpu", initialize_from_iql=not iql)


@pytest.mark.parametrize("iql", [False, True])
@pytest.mark.parametrize("field", ["model_config", "reward_identity"])
def test_identity_mismatch_is_rejected(tmp_path: Path, config: TrainingConfig, iql: bool, field: str) -> None:
    payload, _ = _publication(config, iql=iql)
    payload[field] = {"version": "wrong"}
    with pytest.raises(ValueError, match=field):
        FrozenMacroRootPolicy(_save(tmp_path, payload), config, "cpu", initialize_from_iql=iql)


@pytest.mark.parametrize("iql", [False, True])
def test_nonfinite_weights_are_rejected(tmp_path: Path, config: TrainingConfig, iql: bool) -> None:
    payload, _ = _publication(config, iql=iql)
    name = next(iter(payload["online"]))
    payload["online"][name] = torch.full_like(payload["online"][name], float("nan"))
    with pytest.raises(ValueError, match="non-finite"):
        FrozenMacroRootPolicy(_save(tmp_path, payload), config, "cpu", initialize_from_iql=iql)


def test_partial_iql_inheritance_does_not_leave_random_weights(tmp_path: Path, config: TrainingConfig) -> None:
    payload, _ = _publication(config, iql=True)
    del payload["online"]["candidate_policy_head.0.bias"]
    with pytest.raises(ValueError, match="uninitialized.*candidate_policy_head.0.bias"):
        FrozenMacroRootPolicy(_save(tmp_path, payload), config, "cpu", initialize_from_iql=True)


def test_iql_control_scores_must_be_policy_logits(tmp_path: Path, config: TrainingConfig) -> None:
    payload, _ = _publication(config, iql=True)
    payload["control_scores"] = "q_values"
    with pytest.raises(ValueError, match="policy_logits"):
        FrozenMacroRootPolicy(_save(tmp_path, payload), config, "cpu", initialize_from_iql=True)


def test_invalid_policy_version_is_rejected(tmp_path: Path, config: TrainingConfig) -> None:
    payload, _ = _publication(config)
    payload["policy_version"] = True
    with pytest.raises(ValueError, match="policy_version"):
        FrozenMacroRootPolicy(_save(tmp_path, payload), config, "cpu")


def test_snapshot_capacity_contract_is_checked(tmp_path: Path, config: TrainingConfig) -> None:
    payload, _ = _publication(config)
    policy = FrozenMacroRootPolicy(_save(tmp_path, payload), config, "cpu")
    wrong = replace(config, model=replace(config.model, max_candidates=32))
    with pytest.raises(ValueError, match="config"):
        policy.forward(_snapshot(wrong), None)


def test_publication_directory_is_resolved(tmp_path: Path, config: TrainingConfig) -> None:
    payload, _ = _publication(config)
    path = _save(tmp_path, payload)
    policy = FrozenMacroRootPolicy(tmp_path, config, "cpu")
    assert policy.provenance["source_path"] == str(path.resolve())


def test_absent_publication_is_not_created(tmp_path: Path, config: TrainingConfig) -> None:
    path = tmp_path / "absent.pt"
    with pytest.raises(FileNotFoundError):
        FrozenMacroRootPolicy(path, config, "cpu")
    assert not path.exists()


def test_reward_v9_requires_explicit_positive_budget(tmp_path: Path, config: TrainingConfig) -> None:
    payload, _ = _publication(config)
    no_budget = replace(config, curriculum=replace(config.curriculum, revival_budget=0))
    with pytest.raises(ValueError, match="positive revival budget"):
        FrozenMacroRootPolicy(_save(tmp_path, payload), no_budget, "cpu")


def test_macro_credit_publication_has_distinct_provenance_and_same_inference(
    tmp_path: Path, config: TrainingConfig
) -> None:
    payload, source = _publication(config)
    payload.update(format="sts2-hybrid-macro-credit-publication-v1", control_scores="policy_logits")
    policy = FrozenMacroRootPolicy(_save(tmp_path, payload), config, "cpu")
    assert policy.provenance["source_kind"] == "hybrid_macro_credit_publication"
    assert policy.provenance["source_format"] == payload["format"]
    assert policy.provenance["training_state_loaded"] is False
    logits, hidden = policy.forward(_snapshot(config), None)
    assert torch.isfinite(logits).all() and torch.isfinite(hidden).all()
    for name, tensor in source.state_dict().items():
        torch.testing.assert_close(policy.model.state_dict()[name], tensor, rtol=0, atol=0)


def test_macro_credit_publication_curriculum_transfer(
    tmp_path: Path, config: TrainingConfig
) -> None:
    payload, source = _publication(config)
    payload.update(format="sts2-hybrid-macro-credit-publication-v1", control_scores="policy_logits")
    saved_path = _save(tmp_path, payload)
    new_course = replace(config, curriculum=replace(config.curriculum, revival_budget=48))
    # Direct load fails because course reward_identity differs
    with pytest.raises(ValueError, match="reward_identity differs"):
        FrozenMacroRootPolicy(saved_path, new_course, "cpu", initialize_from_macro_publication=False)
    # Curriculum transfer succeeds
    policy = FrozenMacroRootPolicy(saved_path, new_course, "cpu", initialize_from_macro_publication=True)
    assert policy.provenance["source_kind"] == "hybrid_macro_credit_publication"
    assert policy.provenance["initialization_mode"] == "in_memory_model_initialization"
    assert policy.provenance["source_reward_identity"]["revival_budget"] == 64
    assert policy.provenance["reward_identity"]["revival_budget"] == 48


def test_macro_credit_publication_durable_v4_to_v5_hot_start_preserves_step0_policy_and_fails_closed(
    tmp_path: Path, config: TrainingConfig
) -> None:
    grounded = config.model.to_model_config()
    v4_cfg = HybridActorCriticConfig.durable_v4(grounded)
    v5_cfg = HybridActorCriticConfig.durable_v5(grounded)
    source_v4 = HybridDomainActorCriticModel(v4_cfg).eval()
    payload = {
        "format": "sts2-hybrid-macro-credit-publication-v1",
        "model_config": v4_cfg.as_mapping(),
        "reward_identity": act_frontloaded_death_reward_identity(revival_budget=32),
        "control_scores": "policy_logits",
        "policy_version": 200,
        "online": source_v4.state_dict(),
    }
    saved_path = _save(tmp_path, payload)
    target_course = replace(config, curriculum=replace(config.curriculum, revival_budget=16))

    # Without initialize_from_macro_publication, model_config mismatch is rejected.
    with pytest.raises(ValueError, match="model_config differs"):
        FrozenMacroRootPolicy(
            saved_path,
            replace(config, curriculum=replace(config.curriculum, revival_budget=32)),
            "cpu",
            initialize_from_macro_publication=False,
            actor_critic_config=v5_cfg,
        )

    # Disabling a parent v2 capability flag is rejected fail-closed.
    with pytest.raises(ValueError, match="cannot disable parent flag"):
        FrozenMacroRootPolicy(
            saved_path,
            target_course,
            "cpu",
            initialize_from_macro_publication=True,
            actor_critic_config=HybridActorCriticConfig.durable_v3(grounded),
        )

    # Monotonic v4 -> v5 hot-start inherits 100% of v4 weights and preserves exact step-0 logits.
    policy = FrozenMacroRootPolicy(
        saved_path,
        target_course,
        "cpu",
        initialize_from_macro_publication=True,
        actor_critic_config=v5_cfg,
    )
    assert policy.provenance["ignored_source_tensor_names"] == []
    assert policy.provenance["loaded_tensor_count"] == len(source_v4.state_dict())
    assert len(policy.provenance["fresh_tensor_names"]) > 0
    assert all(
        name.startswith(("candidate_role_", "durable_deck_", "persistent_build_deck_proj."))
        for name in policy.provenance["fresh_tensor_names"]
    )
    snap = _snapshot(config)
    encoded = collate_encoded_snapshots(
        (snap,),
        expected_config=policy.encoding_config,
        expected_fingerprint=snap.encoding_fingerprint,
        device="cpu",
    )
    with torch.no_grad():
        expected = source_v4(encoded, source_v4.initial_state(1, device="cpu"))
    logits, hidden = policy.forward(snap, None)
    torch.testing.assert_close(logits, expected.policy_logits[0], rtol=0, atol=0)
    torch.testing.assert_close(hidden, expected.recurrent_state, rtol=0, atol=0)

    # Missing trunk tensor in source publication fails closed.
    bad_payload = dict(payload, online=dict(payload["online"]))
    del bad_payload["online"]["macro_stream.layers.0.linear1.weight"]
    bad_path = tmp_path / "bad_network.pt"
    torch.save(bad_payload, bad_path)
    with pytest.raises(ValueError):
        FrozenMacroRootPolicy(
            bad_path,
            target_course,
            "cpu",
            initialize_from_macro_publication=True,
            actor_critic_config=v5_cfg,
        )
