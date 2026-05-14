"""Planner memory profiles shared by training and evaluation entrypoints.

The pass-large token-memory model has two separate planner-related memory
knobs:

* rollout breadth/depth used by the explicit search-free combat planner, and
* action-axis chunking used by the rollout and token prediction planner heads.

Historically the train profile lived only in ``training/cli_main.py``.  That
made evaluation/probe scripts silently inherit whatever was serialized in the
checkpoint metadata.  Keeping the profiles here lets train/eval/probes choose
an explicit memory envelope without duplicating the same constants.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any


VALID_PLANNER_MEMORY_PROFILES = ("checkpoint", "custom", "train", "eval", "max")


@dataclass(frozen=True)
class PlannerMemorySettings:
    """Concrete planner settings after profile normalization."""

    profile: str
    combat_rollout_steps: int | None = None
    combat_rollout_beam_width: int | None = None
    action_rollout_buckets: str | None = None
    action_rollout_chunk_size: int | None = None


def normalize_planner_memory_profile(profile: str | None, *, allow_checkpoint: bool = True) -> str:
    """Return a lowercase planner profile name and validate it."""

    normalized = str(profile or ("checkpoint" if allow_checkpoint else "custom")).strip().lower()
    valid = VALID_PLANNER_MEMORY_PROFILES if allow_checkpoint else VALID_PLANNER_MEMORY_PROFILES[1:]
    if normalized not in valid:
        raise ValueError(
            f"Unsupported planner memory profile {profile!r}; "
            f"expected one of {', '.join(valid)}."
        )
    return normalized


def planner_profile_settings(
    profile: str | None,
    *,
    current_steps: int = 3,
    current_beam_width: int = 2,
    current_chunk_size: int = 0,
    allow_checkpoint: bool = True,
) -> PlannerMemorySettings:
    """Resolve a profile into concrete planner settings.

    ``checkpoint`` and ``custom`` are no-op profiles.  ``train`` is the low-VRAM
    profile used during live training; ``eval``/``max`` restore stronger
    planning while keeping action-axis chunking enabled by default.
    """

    normalized = normalize_planner_memory_profile(profile, allow_checkpoint=allow_checkpoint)
    chunk = max(int(current_chunk_size or 0), 0)
    steps = max(int(current_steps or 0), 0)
    beam = max(int(current_beam_width or 0), 0)

    if normalized in {"checkpoint", "custom"}:
        return PlannerMemorySettings(
            profile=normalized,
            action_rollout_chunk_size=chunk,
        )
    if normalized == "train":
        return PlannerMemorySettings(
            profile=normalized,
            combat_rollout_steps=1,
            combat_rollout_beam_width=1,
            action_rollout_buckets="4,8,16,32",
            action_rollout_chunk_size=chunk if chunk > 0 else 16,
        )
    if normalized == "eval":
        return PlannerMemorySettings(
            profile=normalized,
            combat_rollout_steps=max(steps, 4),
            combat_rollout_beam_width=max(beam, 4),
            action_rollout_buckets="8,16,32,64,96",
            action_rollout_chunk_size=chunk if chunk > 0 else 16,
        )
    if normalized == "max":
        return PlannerMemorySettings(
            profile=normalized,
            combat_rollout_steps=max(steps, 5),
            combat_rollout_beam_width=max(beam, 4),
            action_rollout_buckets="8,16,32,64,96",
            action_rollout_chunk_size=chunk if chunk > 0 else 16,
        )
    raise AssertionError(f"unreachable planner profile: {normalized}")


def apply_planner_memory_profile_to_args(args: Any, *, allow_checkpoint: bool = False) -> PlannerMemorySettings:
    """Mutate an argparse namespace in-place according to its profile."""

    settings = planner_profile_settings(
        getattr(args, "planner_memory_profile", "custom"),
        current_steps=int(getattr(args, "combat_rollout_steps", 3) or 3),
        current_beam_width=int(getattr(args, "combat_rollout_beam_width", 2) or 2),
        current_chunk_size=int(getattr(args, "action_rollout_chunk_size", 0) or 0),
        allow_checkpoint=allow_checkpoint,
    )
    args.planner_memory_profile = settings.profile
    if settings.combat_rollout_steps is not None and hasattr(args, "combat_rollout_steps"):
        args.combat_rollout_steps = int(settings.combat_rollout_steps)
    if settings.combat_rollout_beam_width is not None and hasattr(args, "combat_rollout_beam_width"):
        args.combat_rollout_beam_width = int(settings.combat_rollout_beam_width)
    if settings.action_rollout_buckets is not None and hasattr(args, "action_rollout_buckets"):
        args.action_rollout_buckets = settings.action_rollout_buckets
    if settings.action_rollout_chunk_size is not None and hasattr(args, "action_rollout_chunk_size"):
        args.action_rollout_chunk_size = max(int(settings.action_rollout_chunk_size), 0)
    return settings


def apply_planner_memory_profile_to_network(
    network: Any,
    profile: str | None,
    *,
    current_steps: int = 3,
    current_beam_width: int = 2,
    current_chunk_size: int | None = None,
) -> PlannerMemorySettings:
    """Apply network-side rollout bucket/chunk settings after checkpoint load.

    Evaluation scripts often construct the network from checkpoint metadata.
    This helper lets them override only the memory/profile knobs without
    rebuilding the model or changing weight shapes.
    """

    chunk = int(
        current_chunk_size
        if current_chunk_size is not None
        else getattr(network, "action_rollout_chunk_size", 0)
        or 0
    )
    settings = planner_profile_settings(
        profile,
        current_steps=current_steps,
        current_beam_width=current_beam_width,
        current_chunk_size=chunk,
        allow_checkpoint=True,
    )
    if settings.profile == "checkpoint":
        return settings

    if settings.action_rollout_buckets is not None and hasattr(network, "_normalize_action_rollout_buckets"):
        buckets = network._normalize_action_rollout_buckets(settings.action_rollout_buckets)
        network.action_rollout_buckets = buckets
        if hasattr(network, "constructor_kwargs"):
            network.constructor_kwargs["action_rollout_buckets"] = tuple(int(bucket) for bucket in buckets)

    if settings.action_rollout_chunk_size is not None:
        chunk_size = max(int(settings.action_rollout_chunk_size), 0)
        network.action_rollout_chunk_size = chunk_size
        if hasattr(network, "constructor_kwargs"):
            network.constructor_kwargs["action_rollout_chunk_size"] = chunk_size
        prediction = getattr(network, "prediction", None)
        if prediction is not None and hasattr(prediction, "planner_action_chunk_size"):
            prediction.planner_action_chunk_size = chunk_size
        transition_surface = getattr(network, "transition_surface", None)
        if transition_surface is not None and hasattr(transition_surface, "planner_action_chunk_size"):
            transition_surface.planner_action_chunk_size = chunk_size

    return settings
