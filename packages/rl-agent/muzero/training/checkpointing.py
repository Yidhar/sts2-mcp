"""Checkpoint and observation tensor helpers for MuZero training."""

from __future__ import annotations

import json
import os
import pickle
import shutil
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

from muzero.sts2_env.muzero_buffer import MuZeroReplayBuffer, replay_schema_metadata
from muzero.sts2_env.muzero_model import MuZeroNetwork
from sts2_rl.checkpoints import (
    AtomicCheckpointDirectory,
    CheckpointIntegrityError,
    build_checkpoint_provenance,
    contract_metadata,
    validate_hashed_warm_start_checkpoint,
    validate_resume_checkpoint,
)

WEIGHTS_ONLY_MIGRATION_ID = "sts2-weights-only-v1"


def _model_schema_version(network: MuZeroNetwork) -> str:
    """Return a compact model schema id for checkpoint/replay compatibility."""

    spec = network.constructor_spec() if hasattr(network, "constructor_spec") else {}
    obs_mode = str(spec.get("obs_mode") or getattr(network, "obs_mode", "dense_v2"))
    model_arch = str(spec.get("model_arch") or getattr(network, "model_arch", "dense_v1"))
    slot_layout = str(spec.get("token_memory_slot_layout") or "legacy")
    memory_slots = int(spec.get("token_memory_slots") or 0)
    d_model = int(spec.get("token_d_model") or 0)
    action_dim = int(spec.get("action_embed_dim") or getattr(network, "action_embed_dim", 0))
    support_size = int(spec.get("support_size") or getattr(network, "support_size", 0))
    return (
        f"{obs_mode}:{model_arch}:slot_layout={slot_layout}:"
        f"slots={memory_slots}:d={d_model}:action={action_dim}:support={support_size}"
    )


def _checkpoint_schema_metadata(network: MuZeroNetwork) -> dict[str, Any]:
    """Schema metadata persisted in every checkpoint."""

    replay_schema = replay_schema_metadata()
    return {
        "format": "muzero-v2",
        "model_schema_version": _model_schema_version(network),
        **replay_schema,
    }


# ---------------------------------------------------------------------------
# Offline replay-migration compatibility helpers (RC-1).
#
# These helpers are retained for explicit, separately audited migration tools.
# Neither exact resume nor the built-in weights-only warm-start calls them. A
# checkpoint_compatibility_version bump must never silently relax production
# resume validation.
# ---------------------------------------------------------------------------
_REPLAY_IDENTITY_MIGRATIONS: set[tuple[int, int]] = set()


def register_replay_identity_migration(from_version: int, to_version: int) -> None:
    """Mark (from_version -> to_version) as an as-is (identity) replay migration."""

    _REPLAY_IDENTITY_MIGRATIONS.add((int(from_version), int(to_version)))


def _compat_version_loadable(saved_version: Any, current_version: Any) -> bool:
    """True if a replay buffer at ``saved_version`` may load under ``current_version``.

    Equal versions always load; a differing version loads only when an explicit
    identity migration has been registered for that exact transition.
    """

    if saved_version == current_version:
        return True
    if saved_version is None or current_version is None:
        return False
    try:
        return (int(saved_version), int(current_version)) in _REPLAY_IDENTITY_MIGRATIONS
    except (TypeError, ValueError):
        return False


def _metadata_replay_compatible(saved_metadata: dict[str, Any], current_schema: dict[str, Any]) -> tuple[bool, list[str]]:
    """Check whether a checkpoint's replay buffer can be safely loaded."""

    mismatches: list[str] = []
    for key in (
        "checkpoint_compatibility_version",
        "obs_schema_version",
        "replay_schema_version",
        "observation_shape_caps",
    ):
        saved_value = saved_metadata.get(key)
        current_value = current_schema.get(key)
        # The compatibility-version field may differ yet still be loadable when an
        # identity migration is registered (a value-target/reward semantics bump that
        # does not change the on-disk layout). All other fields must match exactly.
        if key == "checkpoint_compatibility_version":
            if not _compat_version_loadable(saved_value, current_value):
                mismatches.append(key)
        elif saved_value != current_value:
            mismatches.append(key)
    return (not mismatches, mismatches)


def _replay_state_compatible(replay_state: dict[str, Any], current_schema: dict[str, Any]) -> tuple[bool, list[str]]:
    """Check schema metadata embedded inside replay_buffer.pkl."""

    mismatches: list[str] = []
    if replay_state.get("schema_version") != current_schema.get("replay_schema_version"):
        mismatches.append("replay_buffer.schema_version")
    if not _compat_version_loadable(
        replay_state.get("checkpoint_compatibility_version"),
        current_schema.get("checkpoint_compatibility_version"),
    ):
        mismatches.append("replay_buffer.checkpoint_compatibility_version")
    if replay_state.get("observation_shape_caps") != current_schema.get("observation_shape_caps"):
        mismatches.append("replay_buffer.observation_shape_caps")
    return (not mismatches, mismatches)


def _load_optimizer_compatible(
    optimizer: optim.Optimizer,
    saved_state: dict[str, Any],
) -> tuple[int, int]:
    """Per-parameter optimizer warm-start (RC-1).

    Copies Adam moments (exp_avg / exp_avg_sq / step) for params whose live tensor
    shape matches the saved moment shape, and leaves the rest as freshly-initialized
    moments. This preserves momentum for the unchanged ~99% of the network when a
    guard/aux edit only resizes or adds one head, instead of the previous
    all-or-nothing reset. Returns (loaded_count, reinitialized_count).

    Matching is positional in optimizer param order with a shape guard, so any
    misalignment from added/removed params degrades to fresh moments for the
    affected params -- never a corrupt update applied to the wrong tensor.
    """

    saved_param_state = saved_state.get("state", {}) if isinstance(saved_state, dict) else {}

    def _saved_entry(index: int) -> dict[str, Any] | None:
        if index in saved_param_state:
            return saved_param_state[index]
        if str(index) in saved_param_state:
            return saved_param_state[str(index)]
        return None

    live_params = [param for group in optimizer.param_groups for param in group["params"]]
    loaded = 0
    reinit = 0
    for index, param in enumerate(live_params):
        entry = _saved_entry(index)
        exp_avg = entry.get("exp_avg") if isinstance(entry, dict) else None
        if torch.is_tensor(exp_avg) and tuple(exp_avg.shape) == tuple(param.shape):
            restored: dict[str, Any] = {}
            for key, value in entry.items():
                restored[key] = value.detach().clone().to(param.device) if torch.is_tensor(value) else value
            optimizer.state[param] = restored
            loaded += 1
        else:
            optimizer.state.pop(param, None)
            reinit += 1
    return (loaded, reinit)


def _optimizer_identity(optimizer: optim.Optimizer) -> dict[str, Any]:
    return {
        "module": type(optimizer).__module__,
        "qualname": type(optimizer).__qualname__,
        "param_group_sizes": [
            len(group.get("params", [])) for group in optimizer.param_groups
        ],
    }


def dict_obs_to_torch(obs: dict[str, Any], device: str = "cpu") -> dict[str, torch.Tensor]:
    """Convert dict observation to torch tensors with batch dimension 1."""
    torch_obs = {}
    for key, value in obs.items():
        if isinstance(value, np.ndarray):
            tensor = torch.from_numpy(value).to(device)
        elif isinstance(value, torch.Tensor):
            tensor = value.to(device)
        else:
            tensor = torch.tensor(value, device=device)

        # Add batch dimension if needed
        if tensor.dim() == 0:
            tensor = tensor.unsqueeze(0)
        elif tensor.dim() > 0 and tensor.shape[0] != 1:
            tensor = tensor.unsqueeze(0)

        torch_obs[key] = tensor

    return torch_obs


def load_resume_checkpoint(
    resume_from: str | Path,
    *,
    network: MuZeroNetwork,
    optimizer: optim.Optimizer,
    buffer: MuZeroReplayBuffer,
    device: str,
    load_buffer: bool = True,
    load_optimizer: bool = True,
    token_target_encoder: nn.Module | None = None,
    amp_grad_scaler: torch.amp.GradScaler | None = None,
) -> dict[str, Any]:
    """Restore the persisted training state from an exact, current checkpoint.

    This is deliberately *not* a warm-start API. Exact resume always restores
    optimizer and replay state, requires matching model/replay schemas, and
    performs manifest/hash/contract/reward/game-data validation before the first
    ``torch.load`` or pickle operation.
    """

    if not load_buffer or not load_optimizer:
        raise ValueError(
            "exact resume requires replay and optimizer restoration; use "
            "load_warm_start_checkpoint(..., migration_id='sts2-weights-only-v1') "
            "for an explicit weights-only migration"
        )

    validated = validate_resume_checkpoint(resume_from)
    checkpoint_path = validated.root
    metadata = validated.metadata
    current_schema = _checkpoint_schema_metadata(network)
    schema_mismatches = [
        key for key, current_value in current_schema.items() if metadata.get(key) != current_value
    ]
    if schema_mismatches:
        raise CheckpointIntegrityError(
            "exact resume checkpoint schema mismatch before deserialization: "
            + ", ".join(schema_mismatches)
        )
    raw_network_spec = (
        network.constructor_spec() if hasattr(network, "constructor_spec") else {}
    )
    current_network_spec = json.loads(json.dumps(raw_network_spec, sort_keys=True))
    if metadata.get("network") != current_network_spec:
        raise CheckpointIntegrityError(
            "exact resume network constructor identity mismatch before deserialization"
        )
    if metadata.get("optimizer") != _optimizer_identity(optimizer):
        raise CheckpointIntegrityError(
            "exact resume optimizer identity mismatch before deserialization"
        )
    if metadata.get("replay_buffer_saved") is not True:
        raise CheckpointIntegrityError(
            "exact resume requires metadata.replay_buffer_saved=true"
        )
    for counter_name in ("total_steps", "episode_count", "buffer_size"):
        counter_value = metadata.get(counter_name)
        if (
            not isinstance(counter_value, int)
            or isinstance(counter_value, bool)
            or counter_value < 0
        ):
            raise CheckpointIntegrityError(
                f"exact resume requires a non-negative integer metadata.{counter_name}"
            )

    token_encoder_flag = metadata.get("token_target_encoder_saved")
    if not isinstance(token_encoder_flag, bool):
        raise CheckpointIntegrityError(
            "exact resume requires boolean metadata.token_target_encoder_saved"
        )
    token_encoder_saved = token_encoder_flag
    if token_encoder_saved != (token_target_encoder is not None):
        raise CheckpointIntegrityError(
            "exact resume token-target encoder configuration does not match checkpoint metadata"
        )
    token_target_encoder_path = checkpoint_path / "token_target_encoder.pt"
    if token_encoder_saved != token_target_encoder_path.is_file():
        raise CheckpointIntegrityError(
            "token_target_encoder.pt presence does not match checkpoint metadata"
        )

    scaler_enabled = bool(amp_grad_scaler is not None and amp_grad_scaler.is_enabled())
    scaler_flag = metadata.get("amp_scaler_enabled")
    if not isinstance(scaler_flag, bool):
        raise CheckpointIntegrityError(
            "exact resume requires boolean metadata.amp_scaler_enabled"
        )
    if scaler_flag != scaler_enabled:
        raise CheckpointIntegrityError(
            "exact resume AMP GradScaler configuration does not match checkpoint metadata"
        )
    amp_scaler_path = checkpoint_path / "amp_scaler.pt"
    if scaler_enabled != amp_scaler_path.is_file():
        raise CheckpointIntegrityError(
            "amp_scaler.pt presence does not match checkpoint metadata"
        )

    network_path = checkpoint_path / "network.pt"
    optimizer_path = checkpoint_path / "optimizer.pt"
    replay_buffer_path = checkpoint_path / "replay_buffer.pkl"
    try:
        network_state = torch.load(network_path, map_location=device, weights_only=True)
        optimizer_state = torch.load(optimizer_path, map_location=device, weights_only=True)
        target_state = (
            torch.load(token_target_encoder_path, map_location=device, weights_only=True)
            if token_target_encoder is not None
            else None
        )
        scaler_state = (
            torch.load(amp_scaler_path, map_location=device, weights_only=True)
            if scaler_enabled
            else None
        )
        with replay_buffer_path.open("rb") as handle:
            replay_buffer_state = pickle.load(handle)
    except Exception as exc:
        raise CheckpointIntegrityError(
            f"exact resume payload deserialization failed: {checkpoint_path}"
        ) from exc

    _require_exact_state_dict(network.state_dict(), network_state, label="network")
    if token_target_encoder is not None:
        _require_exact_state_dict(
            token_target_encoder.state_dict(),
            target_state,
            label="token-target encoder",
        )
    if not isinstance(optimizer_state, dict):
        raise CheckpointIntegrityError("optimizer.pt must contain an optimizer state dictionary")
    if not isinstance(replay_buffer_state, dict):
        raise CheckpointIntegrityError("replay_buffer.pkl must contain a replay state dictionary")
    replay_mismatches = _exact_replay_state_mismatches(
        replay_buffer_state,
        current_schema,
        current_state=buffer.state_dict(),
        expected_transitions=int(metadata["buffer_size"]),
    )
    if replay_mismatches:
        raise CheckpointIntegrityError(
            "exact resume replay schema mismatch: " + ", ".join(replay_mismatches)
        )

    network.load_state_dict(network_state, strict=True)
    try:
        optimizer.load_state_dict(optimizer_state)
        buffer.load_state_dict(replay_buffer_state)
        if token_target_encoder is not None:
            token_target_encoder.load_state_dict(target_state, strict=True)
        if scaler_enabled and amp_grad_scaler is not None:
            if not isinstance(scaler_state, dict):
                raise CheckpointIntegrityError(
                    "amp_scaler.pt must contain a GradScaler state dictionary"
                )
            amp_grad_scaler.load_state_dict(scaler_state)
    except CheckpointIntegrityError:
        raise
    except Exception as exc:
        raise CheckpointIntegrityError(
            "exact resume state restoration failed; no partial fallback is permitted"
        ) from exc
    expected_buffer_size = int(metadata["buffer_size"])
    if len(buffer) != expected_buffer_size:
        raise CheckpointIntegrityError(
            "exact resume replay size mismatch after restoration: "
            f"expected={expected_buffer_size} actual={len(buffer)}"
        )

    print(
        f"[resume] Exact checkpoint restored: id={validated.manifest['checkpoint_id']} "
        f"steps={metadata.get('total_steps')} replay={len(buffer)}"
    )
    return metadata


def _require_exact_state_dict(
    current_state: dict[str, Any],
    saved_state: Any,
    *,
    label: str,
) -> None:
    if not isinstance(saved_state, dict):
        raise CheckpointIntegrityError(f"{label} checkpoint payload must be a state dictionary")
    current_keys = set(current_state)
    saved_keys = {str(key) for key in saved_state}
    if current_keys != saved_keys:
        missing = sorted(current_keys - saved_keys)
        unexpected = sorted(saved_keys - current_keys)
        raise CheckpointIntegrityError(
            f"exact resume {label} keys mismatch: missing={missing[:12]} "
            f"unexpected={unexpected[:12]}"
        )
    mismatched: list[str] = []
    for key, current_value in current_state.items():
        saved_value = saved_state[key]
        current_shape = tuple(getattr(current_value, "shape", ()))
        saved_shape = tuple(getattr(saved_value, "shape", ()))
        current_dtype = getattr(current_value, "dtype", None)
        saved_dtype = getattr(saved_value, "dtype", None)
        if current_shape != saved_shape or current_dtype != saved_dtype:
            mismatched.append(key)
    if mismatched:
        raise CheckpointIntegrityError(
            f"exact resume {label} tensor shape/dtype mismatch: {mismatched[:12]}"
        )


def _exact_replay_state_mismatches(
    replay_state: dict[str, Any],
    current_schema: dict[str, Any],
    *,
    current_state: dict[str, Any],
    expected_transitions: int,
) -> list[str]:
    mismatches: list[str] = []
    expected = {
        "schema_version": current_schema.get("replay_schema_version"),
        "checkpoint_compatibility_version": current_schema.get(
            "checkpoint_compatibility_version"
        ),
        "observation_shape_caps": current_schema.get("observation_shape_caps"),
    }
    for key, expected_value in expected.items():
        if replay_state.get(key) != expected_value:
            mismatches.append(f"replay_buffer.{key}")
    saved_keys = set(replay_state)
    current_keys = set(current_state)
    if saved_keys != current_keys:
        missing = sorted(current_keys - saved_keys)
        unexpected = sorted(saved_keys - current_keys)
        mismatches.append(
            f"replay_buffer.keys(missing={missing[:8]},unexpected={unexpected[:8]})"
        )
    if replay_state.get("total_transitions") != expected_transitions:
        mismatches.append("replay_buffer.total_transitions")
    trajectories = replay_state.get("trajectories")
    priorities = replay_state.get("priorities")
    if not isinstance(trajectories, list):
        mismatches.append("replay_buffer.trajectories")
    if not isinstance(priorities, list):
        mismatches.append("replay_buffer.priorities")
    if isinstance(trajectories, list) and isinstance(priorities, list):
        if len(trajectories) != len(priorities):
            mismatches.append("replay_buffer.trajectory_priority_count")
        try:
            transition_count = sum(len(trajectory) for trajectory in trajectories)
        except (TypeError, AttributeError):
            mismatches.append("replay_buffer.trajectory_lengths")
        else:
            if transition_count != expected_transitions:
                mismatches.append("replay_buffer.trajectory_transition_count")
    return mismatches


def load_warm_start_checkpoint(
    checkpoint: str | Path,
    *,
    network: MuZeroNetwork,
    device: str,
    migration_id: str,
    allow_legacy_checkpoint: bool = False,
    token_target_encoder: nn.Module | None = None,
) -> dict[str, Any]:
    """Apply an explicit weights-only migration without resuming training state.

    The only supported migration copies shape-compatible model tensors. It never
    restores optimizer, replay, scaler, step, or episode state. Atomic source
    checkpoints retain mandatory all-file hash validation. A pre-manifest legacy
    source additionally requires ``allow_legacy_checkpoint=True``.
    """

    if migration_id != WEIGHTS_ONLY_MIGRATION_ID:
        raise ValueError(
            f"unsupported checkpoint migration_id={migration_id!r}; "
            f"expected {WEIGHTS_ONLY_MIGRATION_ID!r}"
        )
    checkpoint_path, manifest, metadata = validate_hashed_warm_start_checkpoint(
        checkpoint
    )
    if manifest is None and not allow_legacy_checkpoint:
        raise CheckpointIntegrityError(
            "legacy checkpoint has no atomic hash manifest; set "
            "allow_legacy_checkpoint=True only for an audited weights-only migration"
        )
    network_path = checkpoint_path / "network.pt"
    if network_path.is_symlink():
        raise CheckpointIntegrityError(
            f"warm-start network.pt must not be a symlink: {network_path}"
        )
    if not network_path.is_file():
        raise CheckpointIntegrityError(f"warm-start network.pt is missing: {network_path}")
    try:
        network_state = torch.load(network_path, map_location=device, weights_only=True)
    except Exception as exc:
        raise CheckpointIntegrityError(
            f"warm-start network payload could not be deserialized: {network_path}"
        ) from exc
    if not isinstance(network_state, dict):
        raise CheckpointIntegrityError("warm-start network.pt must contain a state dictionary")

    current_state = network.state_dict()
    compatible_state = {
        key: value
        for key, value in network_state.items()
        if key in current_state
        and tuple(getattr(current_state[key], "shape", ()))
        == tuple(getattr(value, "shape", ()))
        and getattr(current_state[key], "dtype", None) == getattr(value, "dtype", None)
    }
    if not compatible_state:
        raise CheckpointIntegrityError(
            "weights-only migration found no shape- and dtype-compatible network tensors"
        )
    merged_state = dict(current_state)
    merged_state.update(compatible_state)
    network.load_state_dict(merged_state, strict=True)

    target_loaded = 0
    token_target_path = checkpoint_path / "token_target_encoder.pt"
    if token_target_path.is_symlink():
        raise CheckpointIntegrityError(
            f"warm-start token_target_encoder.pt must not be a symlink: {token_target_path}"
        )
    if token_target_encoder is not None and token_target_path.is_file():
        try:
            target_state = torch.load(
                token_target_path,
                map_location=device,
                weights_only=True,
            )
        except Exception as exc:
            raise CheckpointIntegrityError(
                f"warm-start token-target payload could not be deserialized: {token_target_path}"
            ) from exc
        if not isinstance(target_state, dict):
            raise CheckpointIntegrityError(
                "warm-start token_target_encoder.pt must contain a state dictionary"
            )
        current_target = token_target_encoder.state_dict()
        compatible_target = {
            key: value
            for key, value in target_state.items()
            if key in current_target
            and tuple(getattr(current_target[key], "shape", ()))
            == tuple(getattr(value, "shape", ()))
            and getattr(current_target[key], "dtype", None)
            == getattr(value, "dtype", None)
        }
        merged_target = dict(current_target)
        merged_target.update(compatible_target)
        token_target_encoder.load_state_dict(merged_target, strict=True)
        target_loaded = len(compatible_target)
    elif token_target_encoder is not None:
        network_token_encoder = getattr(network, "token_encoder", None)
        if network_token_encoder is None:
            raise CheckpointIntegrityError(
                "warm-start target encoder was requested but the network has no token_encoder"
            )
        synchronized_state = network_token_encoder.state_dict()
        _require_exact_state_dict(
            token_target_encoder.state_dict(),
            synchronized_state,
            label="warm-start token-target synchronization",
        )
        token_target_encoder.load_state_dict(synchronized_state, strict=True)
        target_loaded = len(synchronized_state)

    report = {
        "mode": "warm_start",
        "migration_id": migration_id,
        "source_checkpoint_id": manifest.get("checkpoint_id") if manifest else None,
        "legacy_source": manifest is None,
        "network_keys_loaded": len(compatible_state),
        "network_keys_skipped": len(network_state) - len(compatible_state),
        "token_target_keys_loaded": target_loaded,
        "source_metadata": metadata,
    }
    print(
        "[warm-start] Applied explicit weights-only migration: "
        f"migration_id={migration_id} loaded={len(compatible_state)} "
        f"skipped={len(network_state) - len(compatible_state)} legacy={manifest is None}"
    )
    return report


class CheckpointingMixin:
    """Checkpoint save/prune helpers for ``MuZeroTrainer``.

    Loading and observation tensor conversion remain module-level functions;
    saving lives here so the legacy train entrypoint does not own artifact
    layout and checkpoint retention logic.
    """

    def save_checkpoint(self, tag: str = "") -> None:
        """Save training checkpoint."""
        checkpoint_dir = Path(self.checkpoint_dir)
        checkpoint_dir.mkdir(parents=True, exist_ok=True)

        if tag:
            name = f"muzero_{tag}"
        else:
            name = f"muzero_step_{self.total_steps:08d}"

        checkpoint_target = checkpoint_dir / name
        context = getattr(self, "experiment_context", None)
        config = getattr(context, "config", None)
        config_option = getattr(config, "option", None)
        warm_start = bool(config_option("warm_start", False)) if callable(config_option) else False
        migration_id = (
            config_option("checkpoint_migration_id", None)
            if callable(config_option)
            else None
        )
        provenance = build_checkpoint_provenance(
            parent_checkpoint=getattr(self, "parent_checkpoint", None),
            experiment_run_id=getattr(context, "run_id", None),
            config_version=getattr(config, "version", None),
            config_profile=(getattr(getattr(config, "runtime", None), "profile", None)),
            checkpoint_load_mode=(
                "warm_start" if warm_start else ("resume" if getattr(self, "parent_checkpoint", None) else "fresh")
            ),
            checkpoint_migration_id=(str(migration_id) if migration_id else None),
            checkpoint_allow_legacy=(
                bool(config_option("allow_legacy_checkpoint", False))
                if callable(config_option)
                else False
            ),
        )
        checkpoint_transaction = AtomicCheckpointDirectory(
            checkpoint_target,
            provenance=provenance,
        )
        checkpoint_path = checkpoint_transaction.prepare()

        torch.save(self.network.state_dict(), checkpoint_path / "network.pt")
        if self.token_target_encoder is not None:
            torch.save(self.token_target_encoder.state_dict(), checkpoint_path / "token_target_encoder.pt")
        torch.save(self.optimizer.state_dict(), checkpoint_path / "optimizer.pt")
        if self.amp_grad_scaler is not None and bool(self.amp_grad_scaler.is_enabled()):
            torch.save(self.amp_grad_scaler.state_dict(), checkpoint_path / "amp_scaler.pt")
        with (checkpoint_path / "replay_buffer.pkl").open("wb") as handle:
            pickle.dump(self.buffer.state_dict(), handle, protocol=pickle.HIGHEST_PROTOCOL)

        metadata = {
            **_checkpoint_schema_metadata(self.network),
            "checkpoint_id": checkpoint_transaction.checkpoint_id,
            "contract": contract_metadata(),
            "provenance": provenance,
            "total_steps": int(self.total_steps),
            "episode_count": int(self.episode_count),
            "buffer_size": len(self.buffer),
            "replay_buffer_saved": True,
            "token_target_encoder_saved": bool(self.token_target_encoder is not None),
            "mixed_precision": self.mixed_precision,
            "amp_enabled": bool(self.amp_enabled),
            "amp_device_type": str(self.amp_device_type),
            "amp_dtype": str(self.amp_dtype),
            "amp_scaler_enabled": bool(self.amp_scaler_enabled),
            "pytorch_alloc_conf": os.environ.get("PYTORCH_ALLOC_CONF"),
            "pytorch_hip_alloc_conf": os.environ.get("PYTORCH_HIP_ALLOC_CONF"),
            "network": self.network.constructor_spec() if hasattr(self.network, "constructor_spec") else {},
            "optimizer": _optimizer_identity(self.optimizer),
            "env": {
                "mode": "combat_sandbox" if hasattr(self.env, "sandbox_supports_potions") else "full_run",
                "potion_mechanics_available": bool(getattr(self.env, "sandbox_supports_potions", True)),
                "obs_mode": getattr(self.network, "obs_mode", "dense_v2"),
                "model_arch": getattr(self.network, "model_arch", "dense_v1"),
            },
            "latent_policy_distill_weight": float(self.latent_policy_distill_weight),
            "latent_policy_target_weight": float(self.latent_policy_target_weight),
            "planner_q_loss_weight": float(self.planner_q_loss_weight),
            "planner_objective_q_loss_weight": float(self.planner_objective_q_loss_weight),
            "objective_value_weight": float(self.objective_value_weight),
            "objective_reward_weight": float(self.objective_reward_weight),
            "combat_hp_preservation_aux_weight": float(
                getattr(self, "combat_hp_preservation_aux_weight", 0.0)
            ),
            "human_demo_alignment": {
                "active": bool(getattr(self, "human_demo_policy_alignment", None) is not None),
                "weight": float(getattr(self, "human_demo_alignment_weight", 0.0)),
                "shadow_only": bool(getattr(self, "human_demo_alignment_shadow_only", True)),
                "every_n_train_steps": int(
                    getattr(self, "human_demo_alignment_every_n_train_steps", 1)
                ),
            },
            "semantic_policy_weight": float(self.semantic_policy_weight),
            "semantic_value_weight": float(self.semantic_value_weight),
            "semantic_reward_weight": float(self.semantic_reward_weight),
            "semantic_state_consistency_weight": float(self.semantic_state_consistency_weight),
            "objective_diversity_weight": float(self.objective_diversity_weight),
            "semantic_policy_label_smoothing": float(self.semantic_policy_label_smoothing),
            "semantic_training_enabled": bool(
                self.semantic_policy_weight > 0.0
                or self.semantic_value_weight > 0.0
                or self.semantic_reward_weight > 0.0
                or self.semantic_state_consistency_weight > 0.0
                or self.objective_diversity_weight > 0.0
            ),
            "state_consistency_weight": float(self.state_consistency_weight),
            "future_world_aux_weight": float(self.future_world_aux_weight),
            "future_bank_state_weight": float(self.future_bank_state_weight),
            "future_bank_delta_weight": float(self.future_bank_delta_weight),
            "future_bank_occupancy_weight": float(self.future_bank_occupancy_weight),
            "future_bank_token_presence_weight": float(self.future_bank_token_presence_weight),
            "future_bank_token_distribution_weight": float(self.future_bank_token_distribution_weight),
            "future_bank_token_slot_state_weight": float(self.future_bank_token_slot_state_weight),
            "future_bank_token_slot_mask_weight": float(self.future_bank_token_slot_mask_weight),
            "future_bank_token_slot_type_weight": float(self.future_bank_token_slot_type_weight),
            "future_bank_token_slot_zone_weight": float(self.future_bank_token_slot_zone_weight),
            "future_bank_token_slot_source_weight": float(self.future_bank_token_slot_source_weight),
            "token_teacher_ema_decay": float(self.token_teacher_ema_decay),
            "future_world_rollout_weight": float(self.future_world_rollout_weight),
            "future_world_rollout_steps": int(self.future_world_rollout_steps),
            "future_world_rollout_decay": float(self.future_world_rollout_decay),
            "latent_gaussian_reg_weight": float(self.latent_gaussian_reg_weight),
            "latent_gaussian_reg_projections": int(self.latent_gaussian_reg_projections),
            "latent_gaussian_reg_slot_weight": float(self.latent_gaussian_reg_slot_weight),
            "latent_gaussian_reg_dynamics_weight": float(self.latent_gaussian_reg_dynamics_weight),
            "latent_gaussian_reg_cov_weight": float(self.latent_gaussian_reg_cov_weight),
            "surprise_loss_weight": float(self.surprise_loss_weight),
            "surprise_hidden_scale": float(self.surprise_hidden_scale),
            "surprise_surface_scale": float(self.surprise_surface_scale),
            "surprise_future_aux_scale": float(self.surprise_future_aux_scale),
            "surprise_target_cap": float(self.surprise_target_cap),
            "surface_mask_weight": float(self.surface_mask_weight),
            "surface_count_weight": float(self.surface_count_weight),
            "surface_domain_weight": float(self.surface_domain_weight),
            "surface_phase_weight": float(self.surface_phase_weight),
            "combat_direct_policy": bool(self.combat_direct_policy),
            "combat_rollout_q_blend": float(self.combat_rollout_q_blend),
            "combat_rollout_objective_q_blend": float(self.combat_rollout_objective_q_blend),
            "combat_rollout_risk_blend": float(self.combat_rollout_risk_blend),
            "combat_rollout_steps": int(self.combat_rollout_steps),
            "combat_rollout_beam_width": int(self.combat_rollout_beam_width),
            "combat_rollout_legal_logit_scale": float(self.combat_rollout_legal_logit_scale),
            "combat_rollout_uncertainty_blend": float(self.combat_rollout_uncertainty_blend),
            "combat_rollout_uncertainty_surprise_weight": float(self.combat_rollout_uncertainty_surprise_weight),
            "combat_rollout_uncertainty_surface_weight": float(self.combat_rollout_uncertainty_surface_weight),
            "combat_rollout_uncertainty_latent_weight": float(self.combat_rollout_uncertainty_latent_weight),
            "combat_rollout_uncertainty_disagreement_weight": float(self.combat_rollout_uncertainty_disagreement_weight),
            "combat_rollout_continuation_uncertainty_penalty": float(self.combat_rollout_continuation_uncertainty_penalty),
            "settlement_weight": float(self.settlement_weight),
            "settlement_decay": float(self.settlement_decay),
            "settlement_max_steps": int(self.settlement_max_steps),
            "trivial_build_fast_path": bool(self.trivial_build_fast_path),
            "potion_reward_fast_path": bool(self.potion_reward_fast_path),
            "domain_num_simulations": {
                domain: int(value) for domain, value in sorted(self.domain_num_simulations.items())
            },
            "checkpoint_keep_last": int(self.checkpoint_keep_last),
            "replay": {
                "boundary_build_bonus": float(self.buffer.boundary_build_bonus),
                "boundary_route_bonus": float(self.buffer.boundary_route_bonus),
                "boundary_family_bonus": float(self.buffer.boundary_family_bonus),
                "trajectory_quality_bonus": float(self.buffer.trajectory_quality_bonus),
                "wasteful_end_turn_scale": float(self.buffer.wasteful_end_turn_scale),
                "wasteful_proceed_scale": float(self.buffer.wasteful_proceed_scale),
                "encounter_tier_weights": dict(sorted(self.buffer.encounter_tier_weights.items())),
                "encounter_priority_weights": dict(sorted(self.buffer.encounter_priority_weights.items())),
            },
            "recent_tail": {
                "windows": [int(window) for window in self.recent_tail_windows],
                "tracked_encounters": list(self.recent_tail_tracked_encounters),
                "min_samples": int(self.recent_tail_min_samples),
            },
            "mcts": {
                "num_simulations": int(self.mcts.num_simulations),
                "max_sampled_actions": int(self.mcts.max_sampled_actions),
                "min_expanded_actions": int(self.mcts.min_expanded_actions),
                "root_max_sampled_actions": int(self.mcts.root_max_sampled_actions),
                "child_max_sampled_actions": int(self.mcts.child_max_sampled_actions),
                "root_min_expanded_actions": int(self.mcts.root_min_expanded_actions),
                "child_min_expanded_actions": int(self.mcts.child_min_expanded_actions),
                "prior_floor": float(self.mcts.prior_floor),
                "relative_prior_floor": float(self.mcts.relative_prior_floor),
                "surface_mask_blend": float(self.mcts.surface_mask_blend),
                "end_turn_prior_bias": float(self.mcts.end_turn_prior_bias),
                "objective_value_blend": float(self.mcts.objective_value_blend),
                "objective_prior_blend": float(self.mcts.objective_prior_blend),
                "semantic_switch_depth": int(self.mcts.semantic_switch_depth),
                "enable_semantic_rollout": bool(self.mcts._semantic_rollout_enabled),
                "semantic_rollout_chain_steps": int(self.mcts.semantic_rollout_chain_steps),
                "semantic_revisit_bonus": float(self.mcts.semantic_revisit_bonus),
                "combat_search_mode": str(self.mcts.combat_search_mode),
                "combat_full_root_frontier": bool(self.mcts.combat_full_root_frontier),
                "root_progressive_widening_init": int(self.mcts.root_progressive_widening_init),
                "child_progressive_widening_init": int(self.mcts.child_progressive_widening_init),
                "root_progressive_widening_growth": float(self.mcts.root_progressive_widening_growth),
                "child_progressive_widening_growth": float(self.mcts.child_progressive_widening_growth),
                "root_bias_enabled": bool(self.mcts._root_bias_enabled),
                "root_bias_min_scale": float(self.mcts.root_bias_min_scale),
                "root_bias_decay_steps": int(self.mcts.root_bias_decay_steps),
            },
        }
        (checkpoint_path / "metadata.json").write_text(
            json.dumps(metadata, indent=2),
            encoding="utf-8",
        )
        checkpoint_transaction.commit()
        print(f"[checkpoint] Saved atomically to {checkpoint_target}")
        self._prune_old_step_checkpoints(checkpoint_dir)

    def _prune_old_step_checkpoints(self, checkpoint_dir: Path) -> None:
        keep_last = int(self.checkpoint_keep_last)
        if keep_last <= 0:
            return

        step_dirs = sorted(
            (
                path
                for path in checkpoint_dir.iterdir()
                if path.is_dir() and path.name.startswith("muzero_step_")
            ),
            key=lambda path: path.name,
        )
        if len(step_dirs) <= keep_last:
            return

        prune_targets = step_dirs[:-keep_last]
        freed_bytes = 0
        pruned_names: list[str] = []

        for target in prune_targets:
            try:
                freed_bytes += sum(
                    file_path.stat().st_size
                    for file_path in target.rglob("*")
                    if file_path.is_file()
                )
            except FileNotFoundError:
                continue

            shutil.rmtree(target, ignore_errors=False)
            pruned_names.append(target.name)

        if pruned_names:
            freed_gb = freed_bytes / (1024 ** 3)
            preview = ", ".join(pruned_names[:4])
            suffix = " ..." if len(pruned_names) > 4 else ""
            print(
                "[checkpoint] Pruned "
                f"{len(pruned_names)} old step checkpoint(s), freed ~{freed_gb:.2f} GB "
                f"(keep_last={keep_last}): {preview}{suffix}"
            )
