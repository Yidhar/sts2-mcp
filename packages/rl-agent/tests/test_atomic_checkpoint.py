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


def _publish_parent_checkpoint(
    path: Path,
    *,
    total_steps: int = 42,
    checkpoint_id: str | None = None,
    metadata_checkpoint_id: str | None = None,
    raw_metadata: str | None = None,
) -> tuple[Path, str]:
    provenance = build_checkpoint_provenance(checkpoint_load_mode="fresh")
    transaction = AtomicCheckpointDirectory(
        path,
        hash_files=True,
        provenance=provenance,
    )
    if checkpoint_id is not None:
        transaction.checkpoint_id = checkpoint_id
    staging = transaction.prepare()
    for name in (
        "network.pt",
        "actor_network.pt",
        "optimizer.pt",
        "rollout_queue.pkl",
        "stochastic_state.pkl",
    ):
        (staging / name).write_bytes(name.encode("utf-8"))
    metadata = {
        "format": "sts2-test-training-checkpoint-v1",
        "checkpoint_id": metadata_checkpoint_id or transaction.checkpoint_id,
        "contract": atomic_module.contract_metadata(),
        "provenance": provenance,
        "training_state": {"environment_steps": total_steps},
        "total_steps": total_steps,
    }
    (staging / "metadata.json").write_text(
        json.dumps(metadata) if raw_metadata is None else raw_metadata,
        encoding="utf-8",
    )
    return transaction.commit(), transaction.checkpoint_id


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
    parent, parent_id = _publish_parent_checkpoint(tmp_path / "parent")

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
    assert provenance["parent_checkpoint"]["checkpoint_id"] == parent_id
    assert (
        provenance["parent_checkpoint"]["relation"]
        == "model_parameter_initialization"
    )
    assert provenance["parent_checkpoint"]["total_steps"] == 42
    assert provenance["parent_checkpoint"]["manifest"]["sha256"]
    assert provenance["parent_checkpoint"]["metadata"]["sha256"]


_PROVENANCE_MODES = {
    "fresh": None,
    "exact_resume": "loaded_parent",
    "model_initialization": "model_parameter_initialization",
    "in_process_successor": "in_process_successor",
}
_PARENT_RELATIONS = (
    None,
    "loaded_parent",
    "model_parameter_initialization",
    "in_process_successor",
    "unspecified_parent",
)
_INVALID_PROVENANCE_COMBINATIONS = [
    (mode, has_parent, relation)
    for mode, expected_relation in _PROVENANCE_MODES.items()
    for has_parent in (False, True)
    for relation in _PARENT_RELATIONS
    if not (
        (mode == "fresh" and not has_parent and relation is None)
        or (
            mode != "fresh"
            and has_parent
            and relation in {None, expected_relation}
        )
    )
]


@pytest.mark.parametrize(
    ("checkpoint_load_mode", "has_parent", "parent_relation"),
    _INVALID_PROVENANCE_COMBINATIONS,
)
def test_checkpoint_provenance_rejects_every_illegal_mode_parent_combination(
    tmp_path: Path,
    checkpoint_load_mode: str,
    has_parent: bool,
    parent_relation: str | None,
) -> None:
    with pytest.raises(ValueError, match="checkpoint provenance"):
        build_checkpoint_provenance(
            parent_checkpoint=tmp_path / "parent" if has_parent else None,
            checkpoint_load_mode=checkpoint_load_mode,
            parent_relation=parent_relation,
        )


@pytest.mark.parametrize(
    (
        "checkpoint_load_mode",
        "has_parent",
        "parent_relation",
        "expected_relation",
    ),
    [
        ("fresh", False, None, None),
        ("exact_resume", True, None, "loaded_parent"),
        ("exact_resume", True, "loaded_parent", "loaded_parent"),
        (
            "model_initialization",
            True,
            None,
            "model_parameter_initialization",
        ),
        (
            "model_initialization",
            True,
            "model_parameter_initialization",
            "model_parameter_initialization",
        ),
        ("in_process_successor", True, None, "in_process_successor"),
        (
            "in_process_successor",
            True,
            "in_process_successor",
            "in_process_successor",
        ),
    ],
)
def test_checkpoint_provenance_accepts_and_normalizes_legal_combinations(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    checkpoint_load_mode: str,
    has_parent: bool,
    parent_relation: str | None,
    expected_relation: str | None,
) -> None:
    monkeypatch.setattr(atomic_module, "_repository_root", lambda: tmp_path)
    monkeypatch.setattr(
        atomic_module,
        "reward_spec_metadata",
        lambda: {"fingerprint": "test-reward-fingerprint"},
    )
    monkeypatch.setattr(atomic_module, "game_data_manifest_metadata", lambda: None)
    monkeypatch.setattr(atomic_module, "dependency_lock_metadata", lambda: [])
    monkeypatch.setattr(atomic_module, "_git_metadata", lambda _root: {})
    parent = None
    if has_parent:
        parent, _ = _publish_parent_checkpoint(tmp_path / "parent")

    provenance = build_checkpoint_provenance(
        parent_checkpoint=parent,
        checkpoint_load_mode=checkpoint_load_mode,
        parent_relation=parent_relation,
    )

    assert provenance["checkpoint_load_mode"] == checkpoint_load_mode
    if expected_relation is None:
        assert provenance["parent_checkpoint"] is None
    else:
        assert provenance["parent_checkpoint"]["relation"] == expected_relation


@pytest.mark.parametrize(
    "checkpoint_load_mode",
    tuple(mode for mode in _PROVENANCE_MODES if mode != "fresh"),
)
def test_nonfresh_checkpoint_provenance_rejects_blank_parent_reference(
    checkpoint_load_mode: str,
) -> None:
    with pytest.raises(ValueError, match="requires a parent checkpoint"):
        build_checkpoint_provenance(
            parent_checkpoint="",
            checkpoint_load_mode=checkpoint_load_mode,
            parent_relation=_PROVENANCE_MODES[checkpoint_load_mode],
        )


def test_nonfresh_checkpoint_provenance_rejects_nonexistent_parent(
    tmp_path: Path,
) -> None:
    with pytest.raises(CheckpointIntegrityError, match="does not exist"):
        build_checkpoint_provenance(
            parent_checkpoint=tmp_path / "missing",
            checkpoint_load_mode="exact_resume",
        )


def test_nonfresh_checkpoint_provenance_rejects_parent_without_manifest(
    tmp_path: Path,
) -> None:
    parent = tmp_path / "no-manifest"
    parent.mkdir()
    (parent / "metadata.json").write_text("{}", encoding="utf-8")

    with pytest.raises(CheckpointIntegrityError, match="manifest is required"):
        build_checkpoint_provenance(
            parent_checkpoint=parent,
            checkpoint_load_mode="exact_resume",
        )


def test_nonfresh_checkpoint_provenance_rejects_corrupted_parent_manifest(
    tmp_path: Path,
) -> None:
    parent, _ = _publish_parent_checkpoint(tmp_path / "corrupt-manifest")
    (parent / "checkpoint.manifest.json").write_text("{broken", encoding="utf-8")

    with pytest.raises(CheckpointIntegrityError, match="manifest is unreadable"):
        build_checkpoint_provenance(
            parent_checkpoint=parent,
            checkpoint_load_mode="exact_resume",
        )


def test_nonfresh_checkpoint_provenance_rejects_parent_without_metadata(
    tmp_path: Path,
) -> None:
    parent, _ = _publish_parent_checkpoint(tmp_path / "missing-metadata")
    (parent / "metadata.json").unlink()

    with pytest.raises(CheckpointIntegrityError, match="checkpoint file missing"):
        build_checkpoint_provenance(
            parent_checkpoint=parent,
            checkpoint_load_mode="exact_resume",
        )


def test_nonfresh_checkpoint_provenance_rejects_invalid_parent_metadata_json(
    tmp_path: Path,
) -> None:
    parent, _ = _publish_parent_checkpoint(
        tmp_path / "invalid-metadata",
        raw_metadata="{broken",
    )

    with pytest.raises(CheckpointIntegrityError, match="metadata is missing or invalid"):
        build_checkpoint_provenance(
            parent_checkpoint=parent,
            checkpoint_load_mode="exact_resume",
        )


def test_nonfresh_checkpoint_provenance_rejects_corrupted_parent_payload(
    tmp_path: Path,
) -> None:
    parent, _ = _publish_parent_checkpoint(tmp_path / "corrupt-payload")
    (parent / "network.pt").write_bytes(b"tampered")

    with pytest.raises(CheckpointIntegrityError, match="checkpoint (size|hash) mismatch"):
        build_checkpoint_provenance(
            parent_checkpoint=parent,
            checkpoint_load_mode="exact_resume",
        )


def test_nonfresh_checkpoint_provenance_rejects_mismatched_checkpoint_ids(
    tmp_path: Path,
) -> None:
    parent, _ = _publish_parent_checkpoint(
        tmp_path / "mismatched-id",
        metadata_checkpoint_id="d780a01c-35f2-4ed1-913a-275e63f4acd7",
    )

    with pytest.raises(CheckpointIntegrityError, match="checkpoint_id does not match"):
        build_checkpoint_provenance(
            parent_checkpoint=parent,
            checkpoint_load_mode="exact_resume",
        )


def test_nonfresh_checkpoint_provenance_rejects_non_uuid_checkpoint_id(
    tmp_path: Path,
) -> None:
    parent, _ = _publish_parent_checkpoint(
        tmp_path / "invalid-id",
        checkpoint_id="not-a-uuid",
    )

    with pytest.raises(CheckpointIntegrityError, match="must be a valid UUID"):
        build_checkpoint_provenance(
            parent_checkpoint=parent,
            checkpoint_load_mode="exact_resume",
        )
