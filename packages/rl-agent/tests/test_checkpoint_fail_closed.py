from __future__ import annotations

import copy
import json
import pickle
from collections.abc import Callable
from pathlib import Path
from typing import Any

import pytest
import torch
import torch.nn as nn

from muzero.training import checkpointing
from sts2_rl.checkpoints import (
    AtomicCheckpointDirectory,
    CheckpointIntegrityError,
    build_checkpoint_provenance,
    contract_metadata,
    validate_resume_checkpoint,
)


def _publish_checkpoint(
    root: Path,
    *,
    provenance_mutator: Callable[[dict[str, Any]], None] | None = None,
    metadata_mutator: Callable[[dict[str, Any]], None] | None = None,
) -> Path:
    target = root / "checkpoint"
    provenance = build_checkpoint_provenance()
    if provenance_mutator is not None:
        provenance_mutator(provenance)
    transaction = AtomicCheckpointDirectory(target, provenance=provenance)
    staging = transaction.prepare()
    (staging / "network.pt").write_bytes(b"network")
    (staging / "optimizer.pt").write_bytes(b"optimizer")
    (staging / "replay_buffer.pkl").write_bytes(b"replay")
    metadata: dict[str, Any] = {
        "checkpoint_id": transaction.checkpoint_id,
        "contract": contract_metadata(),
        "provenance": provenance,
    }
    if metadata_mutator is not None:
        metadata_mutator(metadata)
    (staging / "metadata.json").write_text(
        json.dumps(metadata, sort_keys=True),
        encoding="utf-8",
    )
    transaction.commit()
    return target


def _rewrite_manifest(target: Path, mutator: Callable[[dict[str, Any]], None]) -> None:
    path = target / "checkpoint.manifest.json"
    manifest = json.loads(path.read_text(encoding="utf-8"))
    mutator(manifest)
    path.write_text(json.dumps(manifest, sort_keys=True), encoding="utf-8")


def _publish_torch_checkpoint(root: Path, network: nn.Module) -> Path:
    target = root / "torch-checkpoint"
    provenance = build_checkpoint_provenance()
    transaction = AtomicCheckpointDirectory(target, provenance=provenance)
    staging = transaction.prepare()
    optimizer = torch.optim.Adam(network.parameters(), lr=1e-3)
    torch.save(network.state_dict(), staging / "network.pt")
    torch.save(optimizer.state_dict(), staging / "optimizer.pt")
    schema = checkpointing._checkpoint_schema_metadata(network)  # type: ignore[arg-type]
    replay_state = {
        "schema_version": schema["replay_schema_version"],
        "checkpoint_compatibility_version": schema[
            "checkpoint_compatibility_version"
        ],
        "observation_shape_caps": schema["observation_shape_caps"],
        "total_transitions": 1,
        "trajectories": ["x"],
        "priorities": [1.0],
    }
    with (staging / "replay_buffer.pkl").open("wb") as handle:
        pickle.dump(replay_state, handle, protocol=pickle.HIGHEST_PROTOCOL)
    metadata = {
        **schema,
        "checkpoint_id": transaction.checkpoint_id,
        "contract": contract_metadata(),
        "provenance": provenance,
        "total_steps": 17,
        "episode_count": 3,
        "buffer_size": 1,
        "replay_buffer_saved": True,
        "network": json.loads(
            json.dumps(
                network.constructor_spec()  # type: ignore[attr-defined]
                if hasattr(network, "constructor_spec")
                else {}
            )
        ),
        "optimizer": checkpointing._optimizer_identity(optimizer),
        "token_target_encoder_saved": False,
        "amp_scaler_enabled": False,
    }
    (staging / "metadata.json").write_text(
        json.dumps(metadata, sort_keys=True),
        encoding="utf-8",
    )
    transaction.commit()
    return target


class _ReplayBufferStub:
    def __init__(self) -> None:
        self.state: dict[str, Any] | None = None

    def load_state_dict(self, state: dict[str, Any]) -> None:
        self.state = state

    def state_dict(self) -> dict[str, Any]:
        schema = checkpointing.replay_schema_metadata()
        return {
            "schema_version": schema["replay_schema_version"],
            "checkpoint_compatibility_version": schema[
                "checkpoint_compatibility_version"
            ],
            "observation_shape_caps": schema["observation_shape_caps"],
            "total_transitions": 0,
            "trajectories": [],
            "priorities": [],
        }

    def __len__(self) -> int:
        return 1 if self.state is not None else 0


class _NetworkWithJsonNormalizedSpec(nn.Sequential):
    def constructor_spec(self) -> dict[str, Any]:
        return {"action_rollout_buckets": (8, 16), "dropout": 0.1}


class _TokenNetworkStub(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.token_encoder = nn.Linear(2, 2)
        self.head = nn.Linear(2, 1)


def test_default_atomic_manifest_hashes_every_payload_file(tmp_path: Path) -> None:
    target = _publish_checkpoint(tmp_path)
    manifest = validate_resume_checkpoint(target).manifest

    assert manifest["hash_files"] is True
    assert {entry["path"] for entry in manifest["files"]} == {
        "metadata.json",
        "network.pt",
        "optimizer.pt",
        "replay_buffer.pkl",
    }
    assert all(len(entry["sha256"]) == 64 for entry in manifest["files"])


def test_exact_resume_rejects_missing_atomic_manifest(tmp_path: Path) -> None:
    target = _publish_checkpoint(tmp_path)
    (target / "checkpoint.manifest.json").unlink()

    with pytest.raises(CheckpointIntegrityError, match="manifest is required"):
        validate_resume_checkpoint(target)


def test_exact_resume_rejects_wrong_contract_identity(tmp_path: Path) -> None:
    target = _publish_checkpoint(tmp_path)
    _rewrite_manifest(
        target,
        lambda manifest: manifest["contract"].__setitem__("api_version", "1.0.0"),
    )

    with pytest.raises(CheckpointIntegrityError, match="contract identity mismatch"):
        validate_resume_checkpoint(target)


def test_exact_resume_rejects_wrong_reward_identity(tmp_path: Path) -> None:
    def mutate(provenance: dict[str, Any]) -> None:
        provenance["reward_spec"] = copy.deepcopy(provenance["reward_spec"])
        provenance["reward_spec"]["fingerprint"] = "wrong-reward"

    target = _publish_checkpoint(tmp_path, provenance_mutator=mutate)

    with pytest.raises(CheckpointIntegrityError, match="reward identity mismatch"):
        validate_resume_checkpoint(target)


def test_exact_resume_rejects_wrong_game_data_identity(tmp_path: Path) -> None:
    def mutate(provenance: dict[str, Any]) -> None:
        provenance["game_data_manifest"] = copy.deepcopy(
            provenance["game_data_manifest"]
        )
        provenance["game_data_manifest"]["sha256"] = "0" * 64

    target = _publish_checkpoint(tmp_path, provenance_mutator=mutate)

    with pytest.raises(CheckpointIntegrityError, match="game-data identity mismatch"):
        validate_resume_checkpoint(target)


def test_exact_resume_rejects_payload_hash_mismatch(tmp_path: Path) -> None:
    target = _publish_checkpoint(tmp_path)
    (target / "network.pt").write_bytes(b"NETWORK")

    with pytest.raises(CheckpointIntegrityError, match="hash mismatch"):
        validate_resume_checkpoint(target)


def test_exact_resume_rejects_missing_file_hash(tmp_path: Path) -> None:
    target = _publish_checkpoint(tmp_path)

    def remove_hash(manifest: dict[str, Any]) -> None:
        manifest["files"][0].pop("sha256")

    _rewrite_manifest(target, remove_hash)
    with pytest.raises(CheckpointIntegrityError, match="SHA-256 is required"):
        validate_resume_checkpoint(target)


def test_identity_failure_happens_before_torch_load(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    target = _publish_checkpoint(tmp_path)
    _rewrite_manifest(
        target,
        lambda manifest: manifest["contract"].__setitem__("schema_version", "wrong"),
    )
    torch_load_calls = 0

    def forbidden_torch_load(*args: Any, **kwargs: Any) -> Any:
        nonlocal torch_load_calls
        torch_load_calls += 1
        raise AssertionError("torch.load must not run before identity validation")

    monkeypatch.setattr(checkpointing.torch, "load", forbidden_torch_load)
    with pytest.raises(CheckpointIntegrityError, match="contract identity mismatch"):
        checkpointing.load_resume_checkpoint(
            target,
            network=object(),  # type: ignore[arg-type]
            optimizer=object(),  # type: ignore[arg-type]
            buffer=object(),  # type: ignore[arg-type]
            device="cpu",
        )
    assert torch_load_calls == 0


def test_exact_resume_does_not_accept_cold_state_flags(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="exact resume requires replay and optimizer"):
        checkpointing.load_resume_checkpoint(
            tmp_path,
            network=object(),  # type: ignore[arg-type]
            optimizer=object(),  # type: ignore[arg-type]
            buffer=object(),  # type: ignore[arg-type]
            device="cpu",
            load_buffer=False,
        )


def test_exact_resume_restores_complete_current_checkpoint(tmp_path: Path) -> None:
    source = _NetworkWithJsonNormalizedSpec(nn.Linear(2, 3), nn.Linear(3, 1))
    with torch.no_grad():
        for parameter in source.parameters():
            parameter.fill_(1.5)
    checkpoint = _publish_torch_checkpoint(tmp_path, source)
    target = _NetworkWithJsonNormalizedSpec(nn.Linear(2, 3), nn.Linear(3, 1))
    optimizer = torch.optim.Adam(target.parameters(), lr=1e-3)
    replay = _ReplayBufferStub()

    metadata = checkpointing.load_resume_checkpoint(
        checkpoint,
        network=target,  # type: ignore[arg-type]
        optimizer=optimizer,
        buffer=replay,  # type: ignore[arg-type]
        device="cpu",
    )

    assert metadata["total_steps"] == 17
    assert replay.state is not None
    assert all(torch.all(parameter == 1.5) for parameter in target.parameters())


def test_exact_resume_rejects_partial_model_instead_of_warm_starting(
    tmp_path: Path,
) -> None:
    source = nn.Sequential(nn.Linear(2, 3), nn.Linear(3, 1))
    checkpoint = _publish_torch_checkpoint(tmp_path, source)
    target = nn.Sequential(nn.Linear(2, 3), nn.Linear(3, 2))
    before = {key: value.detach().clone() for key, value in target.state_dict().items()}

    with pytest.raises(CheckpointIntegrityError, match="tensor shape/dtype mismatch"):
        checkpointing.load_resume_checkpoint(
            checkpoint,
            network=target,  # type: ignore[arg-type]
            optimizer=torch.optim.Adam(target.parameters(), lr=1e-3),
            buffer=_ReplayBufferStub(),  # type: ignore[arg-type]
            device="cpu",
        )

    assert all(torch.equal(target.state_dict()[key], value) for key, value in before.items())


def test_legacy_partial_weights_require_explicit_migration_and_legacy_flag(
    tmp_path: Path,
) -> None:
    legacy = tmp_path / "legacy"
    legacy.mkdir()
    source = nn.Sequential(nn.Linear(2, 3), nn.Linear(3, 1))
    with torch.no_grad():
        for parameter in source.parameters():
            parameter.fill_(2.0)
    torch.save(source.state_dict(), legacy / "network.pt")

    target = nn.Sequential(nn.Linear(2, 3), nn.Linear(3, 2))
    original_resized_head = target[1].weight.detach().clone()
    with pytest.raises(CheckpointIntegrityError, match="allow_legacy_checkpoint=True"):
        checkpointing.load_warm_start_checkpoint(
            legacy,
            network=target,  # type: ignore[arg-type]
            device="cpu",
            migration_id=checkpointing.WEIGHTS_ONLY_MIGRATION_ID,
        )

    report = checkpointing.load_warm_start_checkpoint(
        legacy,
        network=target,  # type: ignore[arg-type]
        device="cpu",
        migration_id=checkpointing.WEIGHTS_ONLY_MIGRATION_ID,
        allow_legacy_checkpoint=True,
    )

    assert report["mode"] == "warm_start"
    assert report["legacy_source"] is True
    assert torch.all(target[0].weight == 2.0)
    assert torch.equal(target[1].weight, original_resized_head)


def test_atomic_warm_start_still_rejects_hash_mismatch(tmp_path: Path) -> None:
    target = _publish_checkpoint(tmp_path)
    (target / "network.pt").write_bytes(b"NETWORK")

    with pytest.raises(CheckpointIntegrityError, match="hash mismatch"):
        checkpointing.load_warm_start_checkpoint(
            target,
            network=nn.Linear(2, 2),  # type: ignore[arg-type]
            device="cpu",
            migration_id=checkpointing.WEIGHTS_ONLY_MIGRATION_ID,
        )


def test_legacy_warm_start_synchronizes_missing_target_encoder(tmp_path: Path) -> None:
    legacy = tmp_path / "legacy-target"
    legacy.mkdir()
    source = _TokenNetworkStub()
    with torch.no_grad():
        source.token_encoder.weight.fill_(3.0)
        source.token_encoder.bias.fill_(4.0)
    torch.save(source.state_dict(), legacy / "network.pt")
    target = _TokenNetworkStub()
    target_encoder = nn.Linear(2, 2)

    checkpointing.load_warm_start_checkpoint(
        legacy,
        network=target,  # type: ignore[arg-type]
        device="cpu",
        migration_id=checkpointing.WEIGHTS_ONLY_MIGRATION_ID,
        allow_legacy_checkpoint=True,
        token_target_encoder=target_encoder,
    )

    assert torch.equal(target_encoder.weight, target.token_encoder.weight)
    assert torch.equal(target_encoder.bias, target.token_encoder.bias)
