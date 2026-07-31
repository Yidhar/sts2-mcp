from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path

import pytest

from sts2_rl.checkpoints import ValidatedResumeCheckpoint
from sts2_rl.checkpoints.frozen import (
    V28_100K_FROZEN,
    FrozenCheckpointContract,
    FrozenCheckpointExactResumeError,
    reject_frozen_exact_resume,
)
from sts2_rl.training import checkpointing as checkpointing_module


def test_v28_frozen_contract_matches_reviewed_json() -> None:
    root = Path(__file__).resolve().parents[3]
    payload = json.loads(
        (root / "contracts/frozen-checkpoints/v28-mature-refinement-100k.json").read_text(encoding="utf-8")
    )
    assert payload == {
        "checkpoint_id": V28_100K_FROZEN.checkpoint_id,
        "environment_steps": V28_100K_FROZEN.environment_steps,
        "exact_resume_permitted": False,
        "manifest_sha256": V28_100K_FROZEN.manifest_sha256,
        "metadata_sha256": V28_100K_FROZEN.metadata_sha256,
        "name": V28_100K_FROZEN.name,
        "policy_version": V28_100K_FROZEN.policy_version,
        "relative_path": V28_100K_FROZEN.relative_path.as_posix(),
        "schema_version": "sts2-frozen-checkpoint-contract-v1",
        "source_git_commit": V28_100K_FROZEN.source_git_commit,
        "usage": "model_parameter_initialization_only",
    }


def test_frozen_contract_is_model_initialization_only() -> None:
    with pytest.raises(ValueError, match="model-initialization-only"):
        replace(V28_100K_FROZEN, exact_resume_permitted=True)


def test_frozen_contract_rejects_path_escape() -> None:
    with pytest.raises(ValueError, match="below the artifact root"):
        FrozenCheckpointContract(
            name="escape",
            relative_path=Path("../outside"),
            checkpoint_id="id",
            manifest_sha256="0" * 64,
            metadata_sha256="1" * 64,
            environment_steps=0,
            policy_version=0,
            source_git_commit="commit",
        )


@pytest.mark.parametrize("identity_location", ("manifest", "metadata", "both"))
def test_frozen_checkpoint_identity_forbids_exact_resume_independent_of_path(
    tmp_path: Path,
    identity_location: str,
) -> None:
    manifest: dict[str, object] = {}
    metadata: dict[str, object] = {}
    if identity_location in {"manifest", "both"}:
        manifest["checkpoint_id"] = V28_100K_FROZEN.checkpoint_id
    if identity_location in {"metadata", "both"}:
        metadata["checkpoint_id"] = V28_100K_FROZEN.checkpoint_id
    checkpoint = ValidatedResumeCheckpoint(
        root=tmp_path / "copied-or-symlinked-name",
        manifest=manifest,
        metadata=metadata,
    )

    with pytest.raises(
        FrozenCheckpointExactResumeError,
        match="model-initialization-only.*exact resume is forbidden",
    ):
        reject_frozen_exact_resume(checkpoint)


def test_unrelated_checkpoint_identity_is_not_frozen(tmp_path: Path) -> None:
    reject_frozen_exact_resume(
        ValidatedResumeCheckpoint(
            root=tmp_path,
            manifest={"checkpoint_id": "another-checkpoint"},
            metadata={"checkpoint_id": "another-checkpoint"},
        )
    )


def test_generic_exact_resume_preflight_calls_frozen_identity_guard(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    frozen = ValidatedResumeCheckpoint(
        root=tmp_path / "renamed-frozen-copy",
        manifest={"checkpoint_id": V28_100K_FROZEN.checkpoint_id},
        metadata={"checkpoint_id": V28_100K_FROZEN.checkpoint_id},
    )
    monkeypatch.setattr(
        checkpointing_module,
        "validate_resume_checkpoint",
        lambda _checkpoint: frozen,
    )

    with pytest.raises(FrozenCheckpointExactResumeError):
        checkpointing_module.preflight_training_checkpoint(
            tmp_path / "any-path",
            config=object(),  # type: ignore[arg-type]
            resolved_device="cpu",
            resolved_collector_device="cpu",
        )
