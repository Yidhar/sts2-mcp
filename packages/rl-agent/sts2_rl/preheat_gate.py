"""Fail-closed random-policy gate for native-revival combat preheat.

The gate proves two independent properties before the learner is allowed to
start:

* one combat can consume the game's native Lizard Tail path more than once
  without resetting the enemy state; and
* a random legal policy can actually reach typed combat victories in the
  configured warm-up encounter under unlimited revival.

This is an environment/protocol acceptance gate, not a learning benchmark.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import random
import statistics
import sys
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any
from uuid import uuid4

from sts2_rl.artifacts import resolve_artifact_path
from sts2_rl.backends import HeadlessBackend
from sts2_rl.contracts import CombatResetRequest, EnvironmentResult, StepRequest

GATE_VERSION = "sts2-native-revival-solvability-gate-v1"
DEFAULT_ENCOUNTER = "FUZZY_WURM_CRAWLER_WEAK"
DEFAULT_STRESS_ENCOUNTER = "TUNNELER_WEAK"


@dataclass(frozen=True, slots=True)
class GateConfig:
    episodes: int = 500
    minimum_win_rate: float = 0.99
    maximum_episode_steps: int = 512
    seed: int = 17_729
    encounter_id: str = DEFAULT_ENCOUNTER
    stress_encounter_id: str = DEFAULT_STRESS_ENCOUNTER
    required_stress_revivals: int = 2

    def __post_init__(self) -> None:
        if self.episodes <= 0:
            raise ValueError("episodes must be positive")
        if not 0.0 < self.minimum_win_rate <= 1.0:
            raise ValueError("minimum_win_rate must be in (0, 1]")
        if self.maximum_episode_steps <= 0:
            raise ValueError("maximum_episode_steps must be positive")
        if self.seed < 0:
            raise ValueError("seed must be non-negative")
        if self.required_stress_revivals < 2:
            raise ValueError("required_stress_revivals must be at least two")
        if not self.encounter_id.strip() or not self.stress_encounter_id.strip():
            raise ValueError("encounter IDs must be non-empty")


@dataclass(frozen=True, slots=True)
class EpisodeGateStats:
    seed: int
    steps: int
    victory: bool
    revivals_used: int
    player_hp_lost: float


def _training_counter(result: EnvironmentResult, key: str) -> float:
    training = result.observation.get("_training")
    if not isinstance(training, dict):
        raise RuntimeError("simulator did not export observation._training")
    value = training.get(key)
    if isinstance(value, bool) or not isinstance(value, int | float):
        raise RuntimeError(f"training counter {key!r} is missing or non-numeric")
    normalized = float(value)
    if not math.isfinite(normalized) or normalized < 0.0:
        raise RuntimeError(f"training counter {key!r} is invalid: {value!r}")
    return normalized


def _enemy_hp(result: EnvironmentResult) -> float:
    combat = result.observation.get("combat")
    enemies = combat.get("enemies") if isinstance(combat, dict) else None
    if not isinstance(enemies, list):
        return 0.0
    return sum(
        max(0.0, float(enemy.get("hp", 0.0) or 0.0))
        for enemy in enemies
        if isinstance(enemy, dict)
    )


def validate_counter_transition(
    before: EnvironmentResult,
    after: EnvironmentResult,
    *,
    require_enemy_continuity: bool = False,
) -> None:
    """Validate exact simulator counters against typed transition facts."""

    if after.transition is None:
        raise RuntimeError("headless transition facts are missing")
    facts = after.transition.facts
    before_revivals = _training_counter(before, "revivals_used")
    after_revivals = _training_counter(after, "revivals_used")
    before_hp_lost = _training_counter(before, "player_hp_lost")
    after_hp_lost = _training_counter(after, "player_hp_lost")
    if after_revivals < before_revivals or after_hp_lost < before_hp_lost:
        raise RuntimeError("native-revival training counters are not monotonic")

    revival_delta = int(after_revivals - before_revivals)
    hp_lost_delta = after_hp_lost - before_hp_lost
    if int(facts.get("revivals_used_delta", -1)) != revival_delta:
        raise RuntimeError("typed revival delta disagrees with simulator counter")
    if not math.isclose(
        float(facts.get("player_hp_lost_delta", -1.0)),
        hp_lost_delta,
        rel_tol=0.0,
        abs_tol=1e-9,
    ):
        raise RuntimeError("typed HP-loss delta disagrees with simulator counter")
    if (
        require_enemy_continuity
        and revival_delta > 0
        and _enemy_hp(after) > _enemy_hp(before) + 1e-9
    ):
        raise RuntimeError("enemy state reset or healed across a player revival")


def _reset(
    backend: HeadlessBackend,
    *,
    encounter_id: str,
    seed: int,
    current_hp: int | None = None,
) -> EnvironmentResult:
    result = backend.combat_reset(
        CombatResetRequest(
            request_id=str(uuid4()),
            session_id=backend.session_id,
            expected_state_version=int(backend.get_state()["state_version"]),
            character="IRONCLAD",
            encounter_id=encounter_id,
            seed=seed,
            current_hp=current_hp,
            additional_relics=("RELIC.LIZARD_TAIL",),
            training_revival_budget=-1,
        )
    )
    training = result.observation.get("_training")
    if not isinstance(training, dict) or training.get("revival_budget") != -1:
        raise RuntimeError("combat reset did not activate unlimited native revival")
    if _training_counter(result, "revivals_used") != 0.0:
        raise RuntimeError("revival counter did not reset at episode start")
    if _training_counter(result, "player_hp_lost") != 0.0:
        raise RuntimeError("HP-loss counter did not reset at episode start")
    return result


def _random_step(
    backend: HeadlessBackend,
    result: EnvironmentResult,
    rng: random.Random,
) -> EnvironmentResult:
    if result.terminated:
        raise RuntimeError("attempted to step a terminal episode")
    if not result.legal_actions:
        raise RuntimeError(
            f"non-terminal state {result.observation.get('state_type')!r} "
            "has no legal actions"
        )
    return backend.step(
        StepRequest(
            request_id=str(uuid4()),
            session_id=backend.session_id,
            episode_id=result.episode_id,
            expected_step_index=result.step_index,
            action_index=rng.randrange(len(result.legal_actions)),
        )
    )


def _prove_repeated_native_revival(
    backend: HeadlessBackend,
    config: GateConfig,
) -> EpisodeGateStats:
    rng = random.Random(config.seed ^ 0x5EED5EED)
    result = _reset(
        backend,
        encounter_id=config.stress_encounter_id,
        seed=config.seed,
        current_hp=1,
    )
    for step in range(1, config.maximum_episode_steps + 1):
        before = result
        result = _random_step(backend, result, rng)
        validate_counter_transition(
            before,
            result,
            require_enemy_continuity=True,
        )
        revivals = int(_training_counter(result, "revivals_used"))
        if revivals >= config.required_stress_revivals:
            return EpisodeGateStats(
                seed=config.seed,
                steps=step,
                victory=False,
                revivals_used=revivals,
                player_hp_lost=_training_counter(result, "player_hp_lost"),
            )
        if result.terminated:
            break
    raise RuntimeError(
        "failed to observe more than one native revival in a single combat"
    )


def _run_solvability_episode(
    backend: HeadlessBackend,
    config: GateConfig,
    *,
    episode_index: int,
) -> EpisodeGateStats:
    episode_seed = config.seed + episode_index + 1
    rng = random.Random(episode_seed ^ 0xA17A17)
    result = _reset(
        backend,
        encounter_id=config.encounter_id,
        seed=episode_seed,
    )
    for step in range(1, config.maximum_episode_steps + 1):
        before = result
        result = _random_step(backend, result, rng)
        validate_counter_transition(before, result)
        if result.truncated:
            raise RuntimeError("simulator returned an outcome-unknown truncation")
        if result.terminated:
            facts = result.transition.facts if result.transition is not None else {}
            victory = facts.get("combat_result") == "victory"
            return EpisodeGateStats(
                seed=episode_seed,
                steps=step,
                victory=victory,
                revivals_used=int(_training_counter(result, "revivals_used")),
                player_hp_lost=_training_counter(result, "player_hp_lost"),
            )
    raise RuntimeError(
        f"random episode seed={episode_seed} exceeded "
        f"{config.maximum_episode_steps} steps"
    )


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def run_gate(
    *,
    sim_exe: str | Path | None,
    config: GateConfig,
) -> dict[str, Any]:
    with HeadlessBackend(exe_path=sim_exe) as backend:
        stress = _prove_repeated_native_revival(backend, config)
        episodes = [
            _run_solvability_episode(backend, config, episode_index=index)
            for index in range(config.episodes)
        ]
        resolved_exe = Path(backend.client._exe_path).resolve()

    wins = sum(episode.victory for episode in episodes)
    win_rate = wins / len(episodes)
    minimum_wins = math.ceil(config.minimum_win_rate * config.episodes)
    if wins < minimum_wins:
        raise RuntimeError(
            f"random-policy solvability failed: {wins}/{config.episodes} wins "
            f"is below required {minimum_wins}/{config.episodes}"
        )
    return {
        "version": GATE_VERSION,
        "passed": True,
        "completed_at_utc": datetime.now(UTC).isoformat(),
        "config": asdict(config),
        "simulator": {
            "path": str(resolved_exe),
            "sha256": _sha256(resolved_exe),
        },
        "stress_probe": asdict(stress),
        "solvability": {
            "episodes": config.episodes,
            "wins": wins,
            "win_rate": win_rate,
            "mean_steps": statistics.fmean(e.steps for e in episodes),
            "max_steps": max(e.steps for e in episodes),
            "mean_revivals": statistics.fmean(e.revivals_used for e in episodes),
            "max_revivals": max(e.revivals_used for e in episodes),
            "mean_player_hp_lost": statistics.fmean(
                e.player_hp_lost for e in episodes
            ),
            "max_player_hp_lost": max(e.player_hp_lost for e in episodes),
        },
    }


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sim-exe")
    parser.add_argument("--episodes", type=int, default=500)
    parser.add_argument("--minimum-win-rate", type=float, default=0.99)
    parser.add_argument("--maximum-episode-steps", type=int, default=512)
    parser.add_argument("--seed", type=int, default=17_729)
    parser.add_argument("--encounter-id", default=DEFAULT_ENCOUNTER)
    parser.add_argument("--stress-encounter-id", default=DEFAULT_STRESS_ENCOUNTER)
    parser.add_argument("--required-stress-revivals", type=int, default=2)
    parser.add_argument("--output")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    config = GateConfig(
        episodes=args.episodes,
        minimum_win_rate=args.minimum_win_rate,
        maximum_episode_steps=args.maximum_episode_steps,
        seed=args.seed,
        encounter_id=args.encounter_id,
        stress_encounter_id=args.stress_encounter_id,
        required_stress_revivals=args.required_stress_revivals,
    )
    output = resolve_artifact_path(
        args.output,
        default="reports/preheat-solvability-gate.json",
    )
    try:
        report = run_gate(sim_exe=args.sim_exe, config=config)
    except Exception as exc:
        print(f"[preheat-gate] FAILED: {exc}", file=sys.stderr)
        return 1
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print("[preheat-gate] " + json.dumps(report, ensure_ascii=False, sort_keys=True))
    print(f"[preheat-gate] report={output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = [
    "DEFAULT_ENCOUNTER",
    "DEFAULT_STRESS_ENCOUNTER",
    "GATE_VERSION",
    "EpisodeGateStats",
    "GateConfig",
    "run_gate",
    "validate_counter_transition",
]
