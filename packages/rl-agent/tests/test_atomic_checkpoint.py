from __future__ import annotations

import json
from pathlib import Path

import pytest

import sts2_rl.checkpoints.atomic as atomic_module
from sts2_rl.checkpoints import (
    AtomicCheckpointDirectory,
    CheckpointIntegrityError,
    build_checkpoint_provenance,
    checkpoint_runtime_identity,
    verify_checkpoint_directory,
)


def test_missing_dependency_lock_fails_checkpoint_identity_closed(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(atomic_module, "_repository_root", lambda: tmp_path)
    with pytest.raises(CheckpointIntegrityError, match="dependency lock is missing"):
        atomic_module.dependency_lock_metadata()


def test_static_game_data_provenance_is_optional_for_installed_package(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(atomic_module, "_repository_root", lambda: tmp_path)
    assert atomic_module.game_data_manifest_metadata() is None


def test_atomic_checkpoint_is_invisible_until_commit(tmp_path: Path) -> None:
    target = tmp_path / "checkpoint-1"
    transaction = AtomicCheckpointDirectory(target, hash_files=True)
    staging = transaction.prepare()
    (staging / "network.pt").write_bytes(b"network")

    assert not target.exists()
    committed = transaction.commit()

    assert committed == target
    assert (target / "network.pt").read_bytes() == b"network"
    manifest = json.loads((target / "checkpoint.manifest.json").read_text(encoding="utf-8"))
    assert manifest["format"] == "sts2-atomic-checkpoint-v1"
    assert manifest["contract"]["api_version"] == "2.0.0"
    assert manifest["files"][0]["sha256"]


def test_atomic_checkpoint_context_aborts_on_failure(tmp_path: Path) -> None:
    target = tmp_path / "checkpoint-2"
    with pytest.raises(RuntimeError):
        with AtomicCheckpointDirectory(target) as staging:
            (staging / "partial.pt").write_bytes(b"partial")
            raise RuntimeError("boom")
    assert not target.exists()
    assert list(tmp_path.iterdir()) == []


def test_atomic_checkpoint_refuses_to_replace_published_checkpoint(tmp_path: Path) -> None:
    target = tmp_path / "latest"
    target.mkdir()
    (target / "old").write_text("old", encoding="utf-8")

    transaction = AtomicCheckpointDirectory(target)
    with pytest.raises(FileExistsError, match="immutable"):
        transaction.prepare()
    assert (target / "old").read_text(encoding="utf-8") == "old"

def test_manifest_verifier_rejects_tampered_file(tmp_path: Path) -> None:
    target = tmp_path / "verified"
    transaction = AtomicCheckpointDirectory(target, hash_files=True)
    staging = transaction.prepare()
    (staging / "network.pt").write_bytes(b"good")
    transaction.commit()
    assert verify_checkpoint_directory(target) is not None

    (target / "network.pt").write_bytes(b"tampered")
    with pytest.raises(CheckpointIntegrityError):
        verify_checkpoint_directory(target)

def test_atomic_manifest_embeds_checkpoint_id_and_provenance(tmp_path: Path) -> None:
    target = tmp_path / "with-provenance"
    provenance = {"experiment_run_id": "run-123", "reward_spec": {"fingerprint": "abc"}}
    transaction = AtomicCheckpointDirectory(
        target,
        hash_files=True,
        provenance=provenance,
    )
    staging = transaction.prepare()
    (staging / "network.pt").write_bytes(b"network")
    transaction.commit()

    manifest = verify_checkpoint_directory(target)
    assert manifest is not None
    assert manifest["checkpoint_id"] == transaction.checkpoint_id
    assert manifest["provenance"] == provenance


def test_checkpoint_provenance_captures_reproducibility_and_parent_lineage(
    tmp_path: Path,
) -> None:
    parent = tmp_path / "parent"
    parent.mkdir()
    (parent / "checkpoint.manifest.json").write_text(
        json.dumps({"format": "sts2-atomic-checkpoint-v1", "checkpoint_id": "parent-id"}),
        encoding="utf-8",
    )
    (parent / "metadata.json").write_text(
        json.dumps({"total_steps": 42}),
        encoding="utf-8",
    )

    provenance = build_checkpoint_provenance(
        parent_checkpoint=parent,
        experiment_run_id="run-123",
        config_version="sts2-training-config-v1",
        config_profile="combat",
        checkpoint_load_mode="model_initialization",
        parent_relation="model_parameter_initialization",
    )

    assert provenance["provenance_schema_version"] == "sts2-checkpoint-provenance-v1"
    assert provenance["experiment_run_id"] == "run-123"
    assert provenance["training_config_version"] == "sts2-training-config-v1"
    assert provenance["training_profile"] == "combat"
    assert provenance["checkpoint_load_mode"] == "model_initialization"
    assert provenance["runtime"]["python"]
    reward = provenance["reward_spec"]
    assert reward["version"]
    assert reward["fingerprint"]
    assert len(reward["fingerprint_sha256"]) == 64
    game_data = provenance["game_data_manifest"]
    assert game_data["schema_version"] == "2.0.0"
    assert game_data["upstream_sts2_ai_commit"]
    assert len(game_data["sha256"]) == 64
    lock_names = {Path(item["path"]).name for item in provenance["dependency_locks"]}
    assert {
        "requirements.lock",
        "requirements-dev.lock",
        "requirements-wsl-rocm.txt",
    } <= lock_names
    assert (
        provenance["dependency_locks"]
        == checkpoint_runtime_identity()["dependency_locks"]
    )
    assert "commit" in provenance["git"]
    assert provenance["parent_checkpoint"]["checkpoint_id"] == "parent-id"
    assert (
        provenance["parent_checkpoint"]["relation"]
        == "model_parameter_initialization"
    )
    assert provenance["parent_checkpoint"]["total_steps"] == 42
    assert provenance["parent_checkpoint"]["manifest"]["sha256"]
    assert provenance["parent_checkpoint"]["metadata"]["sha256"]
