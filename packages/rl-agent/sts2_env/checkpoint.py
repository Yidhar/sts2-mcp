from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from safetensors.torch import load_file, save_file
from sb3_contrib import MaskablePPO

from .model import STS2CandidateScoringPolicy


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


def load_online_checkpoint_metadata(checkpoint_dir: str | Path) -> dict[str, Any]:
    checkpoint_path = Path(checkpoint_dir)
    metadata_path = checkpoint_path / "metadata.json"
    if not metadata_path.exists():
        raise FileNotFoundError(f"Missing metadata.json in checkpoint directory: {checkpoint_path}")
    return json.loads(metadata_path.read_text(encoding="utf-8"))


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
    policy.load_state_dict(state_dict, strict=strict)

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

    model = MaskablePPO(
        STS2CandidateScoringPolicy,
        env,
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
    load_online_policy_state_dict(model, checkpoint_path, device=device, strict=True)
    return model, metadata
