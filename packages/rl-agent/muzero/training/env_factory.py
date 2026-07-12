"""Typed environment construction helpers for canonical MuZero training."""

from __future__ import annotations

from typing import Literal

import gymnasium as gym

from combat_snapshot_dataset import CombatSnapshotPool
from launcher import get_session_files as get_default_multi_session_files
from sts2_env.combat_env import CombatSandboxEnv
from sts2_env.env_v2 import SlayTheSpire2EnvV2
from sts2_env.observation_v2 import DictObservationEncoder
from sts2_env.observation_v3 import WorldTokenObservationEncoder
from sts2_rl.backends import HeadlessBackend, LiveBackend
from sts2_rl.contracts import EnvironmentBackend

BackendKind = Literal["live", "headless"]


def mask_fn(env):
    """Extract action mask from environment."""
    return env.unwrapped.action_masks()


def create_environment_backend(
    *,
    kind: str = "live",
    session_file: str | None = None,
    sim_exe_path: str | None = None,
) -> EnvironmentBackend:
    """Create the single typed mutation boundary used by active training.

    Live training is deliberately strict contract-v2.  A missing v2 endpoint
    fails closed instead of issuing a second non-idempotent legacy request.
    """
    normalized = str(kind or "live").strip().lower()
    if normalized == "live":
        return LiveBackend(
            session_path=session_file,
            allow_legacy_fallback=False,
        )
    if normalized == "headless":
        kwargs = {"exe_path": sim_exe_path} if sim_exe_path else {}
        return HeadlessBackend(**kwargs)
    raise ValueError(f"unknown environment backend: {kind!r}")


def get_live_supported_encounter_ids(
    session_file: str | None = None,
    *,
    backend_kind: str = "live",
    sim_exe_path: str | None = None,
) -> set[str]:
    backend = create_environment_backend(
        kind=backend_kind,
        session_file=session_file,
        sim_exe_path=sim_exe_path,
    )
    try:
        method = getattr(backend, "combat_catalog", None)
        if not callable(method):
            raise RuntimeError(f"{backend_kind} backend does not expose a typed combat catalog")
        catalog = method()
        return {
            str(entry.get("encounter_id"))
            for entry in (catalog.get("encounters") or [])
            if entry.get("encounter_id")
        }
    finally:
        backend.close()


# Backward-compatible private name used by ``muzero.train`` while external
# automation migrates to the public helper.
_get_live_supported_encounter_ids = get_live_supported_encounter_ids


def resolve_training_session_files(
    *,
    n_envs: int,
    session_file: str | None,
    session_files: list[str],
) -> list[str | None]:
    """Resolve session files for parallel environments."""
    if n_envs < 1:
        raise ValueError("--n-envs must be >= 1")

    if session_file and session_files:
        raise ValueError("Use either --session-file or --session-files, not both.")

    if session_files:
        if len(session_files) != n_envs:
            raise ValueError(
                f"--session-files count ({len(session_files)}) must match --n-envs ({n_envs})."
            )
        return session_files

    if n_envs == 1:
        return [session_file]

    return get_default_multi_session_files(n_envs)


def build_train_env(
    *,
    env_index: int,
    session_file: str | None,
    combat_sandbox: bool,
    combat_sandbox_potions: bool,
    character: str | None,
    defensive_buffs: bool,
    encounter_id: str | None,
    encounter_pool: list[str],
    snapshot_pool: CombatSnapshotPool | None,
    reset_timeout_ms: int,
    step_timeout_ms: int,
    obs_mode: str,
    seed_pool: list[str] | None = None,
    seed_strategy: str = "round_robin",
    environment_backend: str = "live",
    sim_exe_path: str | None = None,
    backend: EnvironmentBackend | None = None,
) -> gym.Env:
    """Build one training environment around an injected typed backend."""
    normalized_obs_mode = str(obs_mode or "token_v3").strip().lower()
    if normalized_obs_mode == "token_v3":
        obs_encoder = WorldTokenObservationEncoder(use_text=False)
    else:
        obs_encoder = DictObservationEncoder(use_text=False)

    owned_backend = backend or create_environment_backend(
        kind=environment_backend,
        session_file=session_file,
        sim_exe_path=sim_exe_path,
    )
    try:
        if combat_sandbox:
            return CombatSandboxEnv(
                session_file=session_file,
                character=character,
                encounter_id=encounter_id,
                encounter_pool=encounter_pool,
                snapshot_pool=snapshot_pool,
                sandbox_supports_potions=combat_sandbox_potions,
                reset_timeout_ms=reset_timeout_ms,
                step_timeout_ms=step_timeout_ms,
                obs_encoder=obs_encoder,
                backend=owned_backend,
            )

        env_seed_pool = list(seed_pool) if seed_pool else []
        env = SlayTheSpire2EnvV2(
            session_file=session_file,
            character=character,
            defensive_buffs=defensive_buffs,
            reset_timeout_ms=reset_timeout_ms,
            step_timeout_ms=step_timeout_ms,
            obs_encoder=obs_encoder,
            seed_pool=env_seed_pool,
            seed_strategy=seed_strategy,
            backend=owned_backend,
        )
        if env_seed_pool and seed_strategy == "round_robin" and env_index > 0:
            env._seed_pool_cursor = env_index % len(env_seed_pool)
        return env
    except Exception:
        # The environment owns the backend only after construction succeeds.
        owned_backend.close()
        raise
