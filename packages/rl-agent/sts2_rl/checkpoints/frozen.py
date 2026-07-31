"""Pinned, immutable training checkpoints used only as lineage ancestors.

The normal checkpoint manifest authenticates every payload file, but it does
not decide whether a completed training lineage may be extended.  This module
adds that missing control-plane contract: a reviewed checkpoint is identified
by path, checkpoint id, manifest hash, metadata hash and source counters, and
is explicitly restricted to model-parameter initialization.

Nothing in this module makes an old checkpoint ABI-compatible with a new
learner.  It proves the source artifact and prevents an accidental exact
resume or silent selection of another periodic checkpoint.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from pathlib import Path
from typing import Final

from .resume import ValidatedResumeCheckpoint, validate_model_initialization_checkpoint

FROZEN_CHECKPOINT_CONTRACT_VERSION: Final = "sts2-frozen-checkpoint-contract-v1"


class FrozenCheckpointExactResumeError(ValueError):
    """Raised when a model-initialization-only ancestor is used for resume."""


@dataclass(frozen=True, slots=True)
class FrozenCheckpointContract:
    """One reviewed model-initialization-only checkpoint."""

    name: str
    relative_path: Path
    checkpoint_id: str
    manifest_sha256: str
    metadata_sha256: str
    environment_steps: int
    policy_version: int
    source_git_commit: str
    exact_resume_permitted: bool = False

    def __post_init__(self) -> None:
        if not self.name or self.name.strip() != self.name:
            raise ValueError("frozen checkpoint name must be canonical non-empty text")
        if self.relative_path.is_absolute() or ".." in self.relative_path.parts:
            raise ValueError("frozen checkpoint path must remain below the artifact root")
        for text_label, text_value in (
            ("checkpoint_id", self.checkpoint_id),
            ("manifest_sha256", self.manifest_sha256),
            ("metadata_sha256", self.metadata_sha256),
            ("source_git_commit", self.source_git_commit),
        ):
            if not text_value or text_value.strip() != text_value:
                raise ValueError(f"frozen checkpoint {text_label} must be canonical non-empty text")
        for digest_label, digest in (
            ("manifest_sha256", self.manifest_sha256),
            ("metadata_sha256", self.metadata_sha256),
        ):
            if len(digest) != 64 or any(character not in "0123456789abcdef" for character in digest):
                raise ValueError(f"frozen checkpoint {digest_label} must be a lowercase SHA-256 digest")
        for integer_label, integer_value in (
            ("environment_steps", self.environment_steps),
            ("policy_version", self.policy_version),
        ):
            if isinstance(integer_value, bool) or not isinstance(integer_value, int) or integer_value < 0:
                raise ValueError(f"frozen checkpoint {integer_label} must be a non-negative integer")
        if self.exact_resume_permitted:
            raise ValueError("a frozen lineage ancestor must be model-initialization-only")

    def resolve(self, artifact_root: str | Path) -> Path:
        root = Path(artifact_root).expanduser().resolve(strict=False)
        candidate = (root / self.relative_path).resolve(strict=False)
        try:
            candidate.relative_to(root)
        except ValueError as exc:  # pragma: no cover - protected by construction
            raise ValueError("frozen checkpoint path escapes the artifact root") from exc
        return candidate


V28_100K_FROZEN: Final = FrozenCheckpointContract(
    name="v28-mature-refinement-100k",
    relative_path=Path(
        "checkpoints/full-run-revival-v28-mature-refinement-model-init/"
        "run-071a43f5-120a-485f-a6dc-a67a55a1efc2/"
        "periodic-step-000100000"
    ),
    checkpoint_id="f670deda-97d5-46f2-be07-d69a5842eeec",
    manifest_sha256="cfd4d23bf20e064e7a1b21939e94ff01a9f5bf48ab7d70690c66d71f9f2e7af3",
    metadata_sha256="e1933292cc0fbca95cd3b80e073b0b2cda40c1ea29731b59a4bfa21f4f8b5127",
    environment_steps=100_000,
    policy_version=1_569,
    source_git_commit="129094067b19c450030d7f8aaeb28f8963163137",
)
FROZEN_CHECKPOINTS: Final = (V28_100K_FROZEN,)


def reject_frozen_exact_resume(
    checkpoint: ValidatedResumeCheckpoint,
    *,
    contracts: tuple[FrozenCheckpointContract, ...] = FROZEN_CHECKPOINTS,
) -> None:
    """Reject exact resume by immutable checkpoint identity, not by path.

    A frozen ancestor may be copied, symlinked, or addressed through a
    different artifact root.  Consequently, a path-only guard is unsafe.
    ``validate_resume_checkpoint`` has already authenticated and cross-bound
    the manifest and metadata checkpoint IDs; checking both here also keeps
    this pure guard fail-closed in focused tests and future call sites.

    Explicit model initialization intentionally does not call this function.
    """

    if not isinstance(checkpoint, ValidatedResumeCheckpoint):
        raise TypeError("checkpoint must be a ValidatedResumeCheckpoint")
    if not isinstance(contracts, tuple) or not all(
        isinstance(contract, FrozenCheckpointContract) for contract in contracts
    ):
        raise TypeError("contracts must be a FrozenCheckpointContract tuple")
    checkpoint_ids = {
        value
        for value in (
            checkpoint.manifest.get("checkpoint_id"),
            checkpoint.metadata.get("checkpoint_id"),
        )
        if isinstance(value, str)
    }
    for contract in contracts:
        if contract.checkpoint_id in checkpoint_ids:
            raise FrozenCheckpointExactResumeError(
                f"frozen checkpoint {contract.name!r} is model-initialization-only; "
                "exact resume is forbidden"
            )


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def validate_frozen_checkpoint(
    artifact_root: str | Path,
    *,
    contract: FrozenCheckpointContract = V28_100K_FROZEN,
) -> ValidatedResumeCheckpoint:
    """Validate a pinned source without granting exact-resume semantics.

    ``validate_model_initialization_checkpoint`` authenticates every payload
    listed in the checkpoint manifest.  The additional checks below bind the
    otherwise mutable manifest itself, the source counters and the reviewed
    Git provenance.
    """

    root = contract.resolve(artifact_root)
    validated = validate_model_initialization_checkpoint(root)
    if validated.metadata.get("checkpoint_id") != contract.checkpoint_id:
        raise ValueError("frozen checkpoint id changed")
    if _sha256_file(root / "checkpoint.manifest.json") != contract.manifest_sha256:
        raise ValueError("frozen checkpoint manifest hash changed")
    if _sha256_file(root / "metadata.json") != contract.metadata_sha256:
        raise ValueError("frozen checkpoint metadata hash changed")

    state = validated.metadata.get("training_state")
    if not isinstance(state, dict):
        raise ValueError("frozen checkpoint has no training state")
    if state.get("environment_steps") != contract.environment_steps:
        raise ValueError("frozen checkpoint environment step changed")
    if state.get("policy_version") != contract.policy_version:
        raise ValueError("frozen checkpoint policy version changed")

    provenance = validated.metadata.get("provenance")
    git = provenance.get("git") if isinstance(provenance, dict) else None
    if not isinstance(git, dict) or git.get("commit") != contract.source_git_commit:
        raise ValueError("frozen checkpoint source Git commit changed")
    return validated


__all__ = [
    "FROZEN_CHECKPOINTS",
    "FROZEN_CHECKPOINT_CONTRACT_VERSION",
    "V28_100K_FROZEN",
    "FrozenCheckpointContract",
    "FrozenCheckpointExactResumeError",
    "reject_frozen_exact_resume",
    "validate_frozen_checkpoint",
]
