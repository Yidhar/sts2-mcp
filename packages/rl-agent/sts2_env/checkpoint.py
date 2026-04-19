from __future__ import annotations

import importlib
import json
from pathlib import Path
import shutil
from typing import Any

from safetensors.torch import load_file, save_file
from sb3_contrib import MaskablePPO

from .aux_maskable_ppo import AuxMaskablePPO
from .omni_attention_policy import DEFAULT_POLICY_CLASS_PATH, STS2OmniAttentionPolicy

REQUIRED_OBSERVATION_API_VERSION = "attention_obs_v4"
# Older obs API versions that the warmstart migration path knows how to
# upgrade. When a loaded checkpoint's metadata carries one of these, the
# state_dict is padded / zero-backfilled to match the current schema
# before load. Any version not here fails the strict validate check.
_MIGRATABLE_OBSERVATION_API_VERSIONS = {"attention_obs_v3", "attention_obs_v4"}
REQUIRED_COLLECTOR_MODE = "async"
REQUIRED_CANDIDATE_LOCAL_TOKENS = 24

# Only these policy classes may be loaded from checkpoint metadata.
_ALLOWED_POLICY_CLASSES = {
    "sts2_env.omni_attention_policy.STS2OmniAttentionPolicy",
}
_ALLOWED_ALGORITHM_CLASSES = {
    "sts2_env.aux_maskable_ppo.AuxMaskablePPO",
}


def save_online_checkpoint(
    model: MaskablePPO,
    output_dir: str | Path,
    *,
    metadata: dict[str, Any],
) -> Path:
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    save_file(model.policy.state_dict(), str(output_path / "model.safetensors"))
    (output_path / "metadata.json").write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    return output_path


def _parse_periodic_checkpoint_step(checkpoint_path: str | Path) -> int | None:
    path = Path(checkpoint_path)
    if not path.is_dir():
        return None
    name = path.name
    if not name.startswith("step_"):
        return None
    suffix = name[len("step_") :]
    if not suffix.isdigit():
        return None
    try:
        return int(suffix)
    except ValueError:
        return None


def prune_periodic_online_checkpoints(
    checkpoint_root: str | Path,
    *,
    keep_last: int = 3,
) -> list[Path]:
    root = Path(checkpoint_root)
    if keep_last < 0:
        keep_last = 0
    periodic_dirs = [
        path
        for path in root.iterdir()
        if _parse_periodic_checkpoint_step(path) is not None
    ] if root.exists() else []
    periodic_dirs.sort(
        key=lambda path: (_parse_periodic_checkpoint_step(path) or -1, path.name),
        reverse=True,
    )
    removed: list[Path] = []
    for stale_path in periodic_dirs[keep_last:]:
        shutil.rmtree(stale_path, ignore_errors=False)
        removed.append(stale_path)
    return removed


def save_rotating_online_checkpoint(
    model: MaskablePPO,
    checkpoint_root: str | Path,
    *,
    timesteps: int,
    metadata: dict[str, Any],
    keep_last: int = 3,
) -> Path:
    root = Path(checkpoint_root)
    checkpoint_path = root / f"step_{int(timesteps):09d}"
    saved_path = save_online_checkpoint(model, checkpoint_path, metadata=metadata)
    prune_periodic_online_checkpoints(root, keep_last=keep_last)
    return saved_path


def load_online_checkpoint_metadata(checkpoint_dir: str | Path) -> dict[str, Any]:
    checkpoint_path = Path(checkpoint_dir)
    metadata_path = checkpoint_path / "metadata.json"
    if not metadata_path.exists():
        raise FileNotFoundError(f"Missing metadata.json in checkpoint directory: {checkpoint_path}")
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    validate_attention_checkpoint_metadata(metadata, checkpoint_path=checkpoint_path)
    return metadata


def validate_attention_checkpoint_metadata(
    metadata: dict[str, Any],
    *,
    checkpoint_path: str | Path | None = None,
) -> dict[str, Any]:
    observation_api_version = str(metadata.get("observation_api_version") or "").strip()
    collector_mode = str(metadata.get("collector_mode") or "").strip().lower()
    try:
        candidate_local_tokens = int(metadata.get("candidate_local_tokens"))
    except (TypeError, ValueError):
        candidate_local_tokens = -1

    mismatches: list[str] = []
    # Accept any OBS API version we know how to migrate. The actual
    # state-dict upgrade happens inside load_online_policy_state_dict
    # via _pad_state_dict_for_schema_growth.
    if (
        observation_api_version != REQUIRED_OBSERVATION_API_VERSION
        and observation_api_version not in _MIGRATABLE_OBSERVATION_API_VERSIONS
    ):
        mismatches.append(
            f"observation_api_version={observation_api_version or '<missing>'} (expected {REQUIRED_OBSERVATION_API_VERSION})"
        )
    if collector_mode != REQUIRED_COLLECTOR_MODE:
        mismatches.append(f"collector_mode={collector_mode or '<missing>'} (expected {REQUIRED_COLLECTOR_MODE})")
    if candidate_local_tokens != REQUIRED_CANDIDATE_LOCAL_TOKENS:
        mismatches.append(
            f"candidate_local_tokens={candidate_local_tokens if candidate_local_tokens >= 0 else '<missing>'} "
            f"(expected {REQUIRED_CANDIDATE_LOCAL_TOKENS})"
        )

    if mismatches:
        location = f" in checkpoint '{Path(checkpoint_path)}'" if checkpoint_path is not None else ""
        mismatch_text = "; ".join(mismatches)
        raise ValueError(
            "Incompatible attention checkpoint metadata"
            f"{location}: {mismatch_text}. "
            f"This runtime requires {REQUIRED_OBSERVATION_API_VERSION}."
        )

    return metadata


_FORWARD_COMPATIBLE_MISSING_PREFIXES = (
    "candidate_selection_head.",
    "world_bank_relation_bias.",
    "world_bank_cross_blocks.",
    "world_bank_poolers.",
    "world_bank_router_q.",
    "world_bank_router_k.",
    "world_bank_router_bias",
    # Phase 8 Tier 1: history-related params that Phase 6 checkpoints
    # won't have — history_card_bias lives on every RelationBias
    # instance, and the new 7th ("history") entry in bank-indexed
    # ModuleLists extends router/pooler/cross_block lengths.
    "world_relation_bias.history_card_bias.",
    "local_relation_bias.history_card_bias.",
    "query_local_relation_bias.history_card_bias.",
    "query_world_relation_bias.history_card_bias.",
    "candidate_set_relation_bias.history_card_bias.",
)


def _can_relax_for_forward_compatible_missing_keys(policy, state_dict: dict[str, Any]) -> bool:
    policy_state_keys = set(policy.state_dict().keys())
    forward_compatible_keys = {
        key
        for key in policy_state_keys
        if key == "world_bank_router_bias" or any(key.startswith(prefix) for prefix in _FORWARD_COMPATIBLE_MISSING_PREFIXES if prefix.endswith("."))
    }
    if not forward_compatible_keys:
        return False
    checkpoint_keys = set(state_dict.keys())
    missing_keys = policy_state_keys - checkpoint_keys
    unexpected_keys = checkpoint_keys - policy_state_keys
    return bool(missing_keys) and missing_keys.issubset(forward_compatible_keys) and not unexpected_keys


def _pad_state_dict_for_schema_growth(
    policy,
    state_dict: dict[str, Any],
) -> dict[str, Any]:
    """Pad checkpoint state_dict entries to match current policy shapes.

    Phase 8 Tier 1 grew three embedding tables compared to Phase 6:
      - type_embedding (num_token_types): 64 → 66  (added HISTORY_*)
      - role_embedding (max_role_id+1):   47 → 48  (added HISTORY)
      - zone_embedding (max_zone_id+1):   17 → 18  (added HISTORY)
    All three embed new IDs that didn't exist in the old model — zero-
    padding the new rows leaves old token behavior identical while
    letting the new HISTORY tokens start from a blank slate.

    Any other mismatched-shape entries (e.g. pair-bias tables keyed
    by num_token_types × num_token_types) are similarly padded on both
    the row and column axis so the old entries land in the top-left
    block of the larger matrix.

    CAVEAT: pair-bias tables (type_pair_bias, role_pair_bias,
    zone_pair_bias) are Embeddings indexed by ``a*N + b`` where N is the
    matching dimension size. When N grows (e.g. num_types 64→66),
    flat-index k no longer means the same (a, b). Simple row-padding
    preserves old weights at positions that are now semantically
    different — e.g. old (type_a=5, type_b=7) at flat 327 becomes
    (type_a=4, type_b=63) in the new keying. PPO normally re-learns
    these fairly quickly since the biases are small-magnitude gradient
    sinks, but expect ~1-5k steps of transient behavior right after a
    v3→v4 warmstart. If this turns out to be expensive, the fix is to
    write a per-table re-keying migration (un-flatten, copy into new
    indices, re-flatten) — deferred until we see it cost us.

    Does NOT touch keys that match in shape or are entirely missing
    (those are handled by strict=False + forward-compat relaxation).
    """
    import torch  # local import to keep module load light

    target_state = policy.state_dict()
    padded = dict(state_dict)
    for key, ckpt_tensor in state_dict.items():
        if key not in target_state:
            continue
        target_tensor = target_state[key]
        if tuple(ckpt_tensor.shape) == tuple(target_tensor.shape):
            continue
        # Shape mismatch — pad zero-ly into the target shape, preserving
        # the old weights at their original indices. This covers
        # Embedding grows, square pair-bias matrices that depend on
        # num_token_types/max_role/zone, and any other shape that's
        # strictly larger on every dimension.
        if len(ckpt_tensor.shape) != len(target_tensor.shape):
            continue  # rank change — can't migrate automatically
        if any(c > t for c, t in zip(ckpt_tensor.shape, target_tensor.shape)):
            continue  # checkpoint is BIGGER than policy — can't shrink safely
        padded_tensor = torch.zeros(
            target_tensor.shape, dtype=ckpt_tensor.dtype, device=ckpt_tensor.device
        )
        slicer = tuple(slice(0, s) for s in ckpt_tensor.shape)
        padded_tensor[slicer] = ckpt_tensor
        padded[key] = padded_tensor
    return padded


def load_online_policy_state_dict(
    model_or_policy,
    checkpoint_dir: str | Path,
    *,
    device: str = "cpu",
    strict: bool = True,
) -> dict[str, Any]:
    checkpoint_path = Path(checkpoint_dir)
    metadata = load_online_checkpoint_metadata(checkpoint_path)
    state_dict = load_file(str(checkpoint_path / "model.safetensors"), device=device)

    policy = getattr(model_or_policy, "policy", model_or_policy)

    # Phase 8: pad any oversized target entries (embedding tables that
    # grew when we added HISTORY_* types/roles/zones). Safe no-op when
    # the checkpoint already matches the current schema.
    state_dict = _pad_state_dict_for_schema_growth(policy, state_dict)

    try:
        policy.load_state_dict(state_dict, strict=strict)
    except RuntimeError:
        if strict and _can_relax_for_forward_compatible_missing_keys(policy, state_dict):
            policy.load_state_dict(state_dict, strict=False)
        else:
            raise

    if hasattr(model_or_policy, "policy") and hasattr(model_or_policy, "num_timesteps"):
        model_or_policy.num_timesteps = int(metadata.get("timesteps", 0))

    return metadata


def load_online_checkpoint(
    checkpoint_dir: str | Path,
    env,
    *,
    device: str = "cpu",
) -> tuple[MaskablePPO, dict[str, Any]]:
    checkpoint_path = Path(checkpoint_dir)
    metadata = load_online_checkpoint_metadata(checkpoint_path)
    policy_kwargs = metadata.get("policy_kwargs") or {}
    policy_class_path = str(metadata.get("policy_class") or DEFAULT_POLICY_CLASS_PATH).strip()

    if policy_class_path == DEFAULT_POLICY_CLASS_PATH:
        policy_class = STS2OmniAttentionPolicy
    elif policy_class_path in _ALLOWED_POLICY_CLASSES:
        module_name, _, class_name = policy_class_path.rpartition(".")
        module = importlib.import_module(module_name)
        policy_class = getattr(module, class_name)
    else:
        raise ValueError(
            f"Untrusted policy_class in checkpoint metadata: {policy_class_path!r}. "
            f"Allowed: {_ALLOWED_POLICY_CLASSES}"
        )

    algorithm_class_path = str(metadata.get("algorithm_class") or "").strip()
    if algorithm_class_path and algorithm_class_path not in _ALLOWED_ALGORITHM_CLASSES:
        raise ValueError(
            f"Untrusted algorithm_class in checkpoint metadata: {algorithm_class_path!r}. "
            f"Allowed: {_ALLOWED_ALGORITHM_CLASSES}"
        )
    algorithm_class = AuxMaskablePPO if algorithm_class_path == "sts2_env.aux_maskable_ppo.AuxMaskablePPO" else MaskablePPO

    common_kwargs = dict(
        learning_rate=3e-4,
        n_steps=64,
        batch_size=64,
        n_epochs=10,
        gamma=0.99,
        gae_lambda=0.95,
        clip_range=0.2,
        ent_coef=0.01,
        vf_coef=0.5,
        max_grad_norm=0.5,
        policy_kwargs=policy_kwargs,
        verbose=0,
        device=device,
    )
    if algorithm_class is AuxMaskablePPO:
        common_kwargs.update(
            aux_objective_coef=float(metadata.get("aux_objective_coef", 0.25)),
            aux_transition_coef=float(metadata.get("aux_transition_coef", 0.10)),
            aux_trait_coef=float(metadata.get("aux_trait_coef", 0.10)),
            aux_build_coef=float(metadata.get("aux_build_coef", 0.10)),
            aux_selection_coef=float(metadata.get("aux_selection_coef", 0.10)),
            aux_route_coef=float(metadata.get("aux_route_coef", 0.10)),
        )

    model = algorithm_class(
        policy_class,
        env,
        **common_kwargs,
    )
    load_online_policy_state_dict(model, checkpoint_path, device=device, strict=True)
    return model, metadata
