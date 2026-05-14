from __future__ import annotations

from collections import Counter

import pytest
import torch

from muzero.sts2_env.muzero_model import MuZeroNetwork
from muzero.sts2_env.planner_memory_profile import apply_planner_memory_profile_to_network
from muzero.sts2_env.token_memory import (
    GLOBAL_MEMORY_BANK_INDEX,
    MEMORY_BANK_NAMES,
    MEMORY_SLOT_LAYOUT_PASS_LARGE_V1,
    MEMORY_SLOT_LAYOUT_QUOTA_V1,
    WORLD_BANK_NAMES,
    TokenMemoryEncoder,
    TokenPredictionNetwork,
    build_memory_slot_bank_ids,
)


def _count_by_bank_name(slot_bank_ids: list[int]) -> Counter[str]:
    return Counter(MEMORY_BANK_NAMES[int(bank_id)] for bank_id in slot_bank_ids)


def test_legacy_slot_layout_preserves_original_mapping() -> None:
    explicit_bank_ids = list(range(len(WORLD_BANK_NAMES)))

    assert build_memory_slot_bank_ids(8, layout="legacy") == explicit_bank_ids + [GLOBAL_MEMORY_BANK_INDEX]
    assert build_memory_slot_bank_ids(12, layout="legacy") == explicit_bank_ids + [GLOBAL_MEMORY_BANK_INDEX] * 5


def test_quota_v1_slot_layout_expands_long_horizon_banks_at_16_slots() -> None:
    slot_bank_ids = build_memory_slot_bank_ids(16, layout=MEMORY_SLOT_LAYOUT_QUOTA_V1)

    assert len(slot_bank_ids) == 16
    assert _count_by_bank_name(slot_bank_ids) == Counter(
        {
            "runtime": 1,
            "support": 1,
            "enemy": 2,
            "build": 4,
            "route": 3,
            "powers": 1,
            "history": 2,
            "global": 2,
        }
    )


def test_quota_v1_keeps_8_slot_profile_backward_like() -> None:
    assert build_memory_slot_bank_ids(8, layout=MEMORY_SLOT_LAYOUT_QUOTA_V1) == build_memory_slot_bank_ids(
        8,
        layout="legacy",
    )


def test_quota_v1_appends_extra_slots_as_global_planning_capacity() -> None:
    slot_bank_ids = build_memory_slot_bank_ids(18, layout=MEMORY_SLOT_LAYOUT_QUOTA_V1)

    assert len(slot_bank_ids) == 18
    assert slot_bank_ids[-2:] == [GLOBAL_MEMORY_BANK_INDEX, GLOBAL_MEMORY_BANK_INDEX]


def test_pass_large_v1_slot_layout_matches_sts2_pass_target_at_24_slots() -> None:
    slot_bank_ids = build_memory_slot_bank_ids(24, layout=MEMORY_SLOT_LAYOUT_PASS_LARGE_V1)

    assert len(slot_bank_ids) == 24
    assert _count_by_bank_name(slot_bank_ids) == Counter(
        {
            "runtime": 4,
            "support": 2,
            "enemy": 3,
            "build": 4,
            "route": 3,
            "powers": 2,
            "history": 2,
            "global": 4,
        }
    )


def test_invalid_slot_layout_is_rejected() -> None:
    with pytest.raises(ValueError, match="Unknown token memory slot layout"):
        build_memory_slot_bank_ids(16, layout="made_up")


def test_token_memory_encoder_uses_requested_slot_layout() -> None:
    encoder = TokenMemoryEncoder(
        d_model=16,
        n_heads=2,
        ffn_dim=32,
        world_layers=1,
        local_layers=1,
        decoder_layers=1,
        candidate_set_layers=0,
        world_bank_top_k=5,
        bank_token_slots=1,
        num_memory_slots=16,
        memory_slot_layout=MEMORY_SLOT_LAYOUT_QUOTA_V1,
        action_embed_dim=8,
    )

    assert encoder.memory_slot_layout == MEMORY_SLOT_LAYOUT_QUOTA_V1
    assert _count_by_bank_name(encoder._slot_bank_ids.tolist())["build"] == 4
    assert _count_by_bank_name(encoder._slot_bank_ids.tolist())["route"] == 3
    assert int(encoder._slot_bank_ids.numel()) == 16


def test_muzero_network_plumbs_slot_layout_to_all_token_modules() -> None:
    network = MuZeroNetwork(
        obs_mode="token_v3",
        model_arch="token_memory_v1",
        action_embed_dim=8,
        token_d_model=16,
        token_n_heads=2,
        token_ffn_dim=32,
        token_world_layers=1,
        token_local_layers=1,
        token_decoder_layers=1,
        token_candidate_set_layers=0,
        token_memory_slots=16,
        token_memory_slot_layout=MEMORY_SLOT_LAYOUT_QUOTA_V1,
        token_world_bank_top_k=5,
        token_bank_token_slots=1,
    )

    assert network.hidden_dim == 16 * 16
    assert network.constructor_spec()["token_memory_slot_layout"] == MEMORY_SLOT_LAYOUT_QUOTA_V1
    for module in (
        network.token_encoder,
        network.dynamics,
        network.prediction,
        network.transition_surface,
        network.future_world_bank_head,
        network.state_projector,
        network.semantic_state_projector,
        network.semantic_dynamics,
    ):
        assert getattr(module, "memory_slot_layout") == MEMORY_SLOT_LAYOUT_QUOTA_V1


def test_pass_large_v1_muzero_network_hidden_dim_and_plumbing() -> None:
    network = MuZeroNetwork(
        obs_mode="token_v3",
        model_arch="token_memory_v1",
        action_embed_dim=8,
        support_size=31,
        dynamics_res_blocks=6,
        token_d_model=16,
        token_n_heads=2,
        token_ffn_dim=32,
        token_world_layers=1,
        token_local_layers=1,
        token_decoder_layers=1,
        token_candidate_set_layers=0,
        token_memory_slots=24,
        token_memory_slot_layout=MEMORY_SLOT_LAYOUT_PASS_LARGE_V1,
        token_world_bank_top_k=7,
        token_bank_token_slots=2,
    )

    assert network.hidden_dim == 16 * 24
    assert network.support_size == 31
    assert network.constructor_spec()["token_memory_slot_layout"] == MEMORY_SLOT_LAYOUT_PASS_LARGE_V1
    assert network.constructor_spec()["token_memory_slots"] == 24
    assert network.constructor_spec()["token_world_bank_top_k"] == 7
    for module in (
        network.token_encoder,
        network.dynamics,
        network.prediction,
        network.transition_surface,
        network.future_world_bank_head,
        network.state_projector,
        network.semantic_state_projector,
        network.semantic_dynamics,
    ):
        assert getattr(module, "memory_slot_layout") == MEMORY_SLOT_LAYOUT_PASS_LARGE_V1


def test_action_rollout_chunk_size_plumbs_to_token_prediction_modules() -> None:
    network = MuZeroNetwork(
        obs_mode="token_v3",
        model_arch="token_memory_v1",
        action_embed_dim=8,
        token_d_model=16,
        token_n_heads=2,
        token_ffn_dim=32,
        token_world_layers=1,
        token_local_layers=1,
        token_decoder_layers=1,
        token_candidate_set_layers=0,
        token_memory_slots=16,
        token_memory_slot_layout=MEMORY_SLOT_LAYOUT_QUOTA_V1,
        token_world_bank_top_k=5,
        token_bank_token_slots=1,
        action_rollout_chunk_size=16,
    )

    assert network.constructor_spec()["action_rollout_chunk_size"] == 16
    assert network.action_rollout_chunk_size == 16
    assert network.prediction.planner_action_chunk_size == 16
    assert network.transition_surface.planner_action_chunk_size == 16


def test_planner_memory_profile_updates_network_chunking_modules() -> None:
    network = MuZeroNetwork(
        obs_mode="token_v3",
        model_arch="token_memory_v1",
        action_embed_dim=8,
        token_d_model=16,
        token_n_heads=2,
        token_ffn_dim=32,
        token_world_layers=1,
        token_local_layers=1,
        token_decoder_layers=1,
        token_candidate_set_layers=0,
        token_memory_slots=16,
        token_memory_slot_layout=MEMORY_SLOT_LAYOUT_QUOTA_V1,
        token_world_bank_top_k=5,
        token_bank_token_slots=1,
        action_rollout_chunk_size=0,
    )

    settings = apply_planner_memory_profile_to_network(
        network,
        "eval",
        current_steps=3,
        current_beam_width=2,
        current_chunk_size=0,
    )

    assert settings.profile == "eval"
    assert network.action_rollout_chunk_size == 16
    assert network.constructor_spec()["action_rollout_chunk_size"] == 16
    assert network.prediction.planner_action_chunk_size == 16
    assert network.transition_surface.planner_action_chunk_size == 16


def test_token_prediction_chunking_preserves_eval_outputs() -> None:
    torch.manual_seed(7)
    full = TokenPredictionNetwork(
        hidden_dim=8 * 8,
        action_embed_dim=4,
        d_model=8,
        num_memory_slots=8,
        n_heads=2,
        ffn_dim=16,
        support_size=3,
        dropout=0.0,
        activation_checkpointing=False,
        planner_action_chunk_size=0,
    ).eval()
    chunked = TokenPredictionNetwork(
        hidden_dim=8 * 8,
        action_embed_dim=4,
        d_model=8,
        num_memory_slots=8,
        n_heads=2,
        ffn_dim=16,
        support_size=3,
        dropout=0.0,
        activation_checkpointing=False,
        planner_action_chunk_size=4,
    ).eval()
    chunked.load_state_dict(full.state_dict())
    hidden_state = torch.randn(2, 8 * 8)
    action_embeddings = torch.randn(2, 10, 4)

    with torch.no_grad():
        full_outputs = full(hidden_state, action_embeddings=action_embeddings)
        chunked_outputs = chunked(hidden_state, action_embeddings=action_embeddings)

    assert len(full_outputs) == len(chunked_outputs)
    for full_tensor, chunked_tensor in zip(full_outputs, chunked_outputs):
        assert full_tensor.shape == chunked_tensor.shape
        assert torch.allclose(full_tensor, chunked_tensor, atol=1e-5, rtol=1e-5)
