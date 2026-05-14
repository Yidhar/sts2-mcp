"""Environment construction helpers for MuZero training."""

from __future__ import annotations

import gymnasium as gym

from combat_snapshot_dataset import CombatSnapshotPool
from launcher import get_session_files as get_default_multi_session_files
from sts2_env.bridge_client import BridgeClient
from sts2_env.combat_env import CombatSandboxEnv
from sts2_env.env_v2 import SlayTheSpire2EnvV2
from sts2_env.observation_v2 import DictObservationEncoder
from sts2_env.observation_v3 import WorldTokenObservationEncoder


def mask_fn(env):
    """Extract action mask from environment."""
    return env.unwrapped.action_masks()


def get_live_supported_encounter_ids(session_file: str | None = None) -> set[str]:
    client = BridgeClient(session_path=session_file)
    catalog = client.combat_catalog()
    return {
        str(entry.get("encounter_id"))
        for entry in (catalog.get("encounters") or [])
        if entry.get("encounter_id")
    }


# Backward-compatible private name used by ``muzero.train`` while it is still a
# legacy monolith.
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
) -> gym.Env:
    """Build a single training environment."""
    obs_mode = str(obs_mode or "dense_v2").strip().lower()
    if obs_mode == "token_v3":
        obs_encoder = WorldTokenObservationEncoder(use_text=False)
    else:
        obs_encoder = DictObservationEncoder(use_text=False)

    if combat_sandbox:
        env = CombatSandboxEnv(
            session_file=session_file,
            character=character,
            encounter_id=encounter_id,
            encounter_pool=encounter_pool,
            snapshot_pool=snapshot_pool,
            sandbox_supports_potions=combat_sandbox_potions,
            reset_timeout_ms=reset_timeout_ms,
            step_timeout_ms=step_timeout_ms,
            obs_encoder=obs_encoder,
        )
        return env

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
    )
    if env_seed_pool and seed_strategy == "round_robin" and env_index > 0:
        env._seed_pool_cursor = env_index % len(env_seed_pool)
    return env
