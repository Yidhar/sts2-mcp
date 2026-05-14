"""Fixed-seed combat-sandbox eval comparing multiple MuZero checkpoints.

Implements §P0-1 of `docs/muzero-replay-mechanism-recovery-task-list-20260506.md`:
score N candidate checkpoints over the same encounter / seed matrix and pick
the best one to resume training from.

The driver does NOT modify any training code paths. It reuses
`muzero.evaluate.load_muzero_network` and the `CombatSandboxEnv` / `MCTS`
classes that train.py already exercises, so eval behavior matches training
inference (minus root-bias decay tweaks, which we keep at training settings).

Output (under ``--report-dir``):

* ``checkpoint_eval_summary.json`` — full structured stats
* ``checkpoint_eval_summary.md``   — human-readable table
* ``best_checkpoint.txt``          — single line, the chosen checkpoint dir
* ``per_episode/<ckpt-tag>__<encounter>.jsonl`` — full per-episode traces

Usage::

    python -m scripts.eval_checkpoint_compare \\
        --checkpoints /mnt/e/.../ckpt_a /mnt/e/.../ckpt_b \\
        --episodes-boss 8 --episodes-elite 6 --episodes-normal 6 \\
        --session-file /mnt/c/.../session.json \\
        --snapshot-dataset /mnt/e/.../curated_combat_ironclad_mixed_provenance \\
        --snapshot-subset bootstrap_human_plus_local_all_roomwin_only_minus_combat_reset_failures \\
        --report-dir packages/rl-agent/eval_reports/checkpoint_compare_20260506 \\
        --device cuda
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
import traceback
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Any

import numpy as np
import torch

THIS_DIR = Path(__file__).resolve().parent
RL_AGENT_DIR = THIS_DIR.parent
if str(RL_AGENT_DIR) not in sys.path:
    sys.path.insert(0, str(RL_AGENT_DIR))

from sts2_env.combat_env import CombatSandboxEnv
from sts2_env.observation_v3 import WorldTokenObservationEncoder
from sts2_env.observation_v2 import DictObservationEncoder
from muzero.evaluate import load_muzero_network
from muzero.sts2_env.mcts import MCTS
from combat_snapshot_dataset import CombatSnapshotPool


# Mirrors launch_muzero_boss_recovery_20260505_sandpit_enemy_power_fix.sh pools.
DEFAULT_BOSS_ENCOUNTERS = (
    "ENCOUNTER.KAISER_CRAB_BOSS",
    "ENCOUNTER.CEREMONIAL_BEAST_BOSS",
    "ENCOUNTER.THE_KIN_BOSS",
    "ENCOUNTER.THE_INSATIABLE_BOSS",
    "ENCOUNTER.KNOWLEDGE_DEMON_BOSS",
)
DEFAULT_ELITE_ENCOUNTERS = (
    "ENCOUNTER.PHROG_PARASITE_ELITE",
    "ENCOUNTER.KNIGHTS_ELITE",
    "ENCOUNTER.SOUL_NEXUS_ELITE",
)
DEFAULT_NORMAL_ENCOUNTERS = (
    "ENCOUNTER.CONSTRUCT_MENAGERIE_NORMAL",
    "ENCOUNTER.SLUMBERING_BEETLE_NORMAL",
    "ENCOUNTER.OVICOPTER_NORMAL",
)

# Offender flags we can derive from CombatSandboxEnv.info["action_diagnostics"]
# without rerunning train.py's search_stats pipeline. These are sufficient to
# rank checkpoints on safety / behavior quality; per-mechanism Kaiser /
# Insatiable mistakes are already encoded in per-encounter win rate.
OFFENDER_DIAG_KEYS = (
    "hp_cost_self_lethal_selected",
    "hp_cost_low_margin_selected",
    "x_cost_zero_bad_selected",
    "star_x_selected",
    "transient_leaked",
    "prior_transient_only_end_turn",
    "wasteful_end_turn_penalty_applied",
)


@dataclass
class EpisodeOutcome:
    encounter_id: str
    seed: int
    win: bool
    terminated: bool
    truncated: bool
    total_reward: float
    length: int
    final_hp: float
    final_max_hp: float
    offender_step_rates: dict[str, float] = field(default_factory=dict)
    error: str | None = None


@dataclass
class CheckpointResult:
    checkpoint: str
    tag: str
    episodes: list[EpisodeOutcome] = field(default_factory=list)


def _player_hp_from_raw_obs(raw_obs: dict[str, Any]) -> tuple[float, float]:
    """Best-effort extraction of player current/max HP from bridge raw_obs.

    Mirrors how `combat_env._player_hp_and_max` reads the structure but stays
    self-contained so the eval script does not depend on private env helpers.
    """
    if not isinstance(raw_obs, dict):
        return 0.0, 0.0
    state = raw_obs.get("state") or raw_obs
    player = (state or {}).get("player") if isinstance(state, dict) else None
    if not isinstance(player, dict):
        # Fallback: top-level player?
        player = raw_obs.get("player") if isinstance(raw_obs.get("player"), dict) else {}
    hp = player.get("current_hp") or player.get("hp") or 0
    max_hp = player.get("max_hp") or 0
    try:
        return float(hp), float(max_hp)
    except (TypeError, ValueError):
        return 0.0, 0.0


def _tier_for_encounter(eid: str) -> str:
    eid_upper = (eid or "").upper()
    if "_BOSS" in eid_upper:
        return "boss"
    if "_ELITE" in eid_upper:
        return "elite"
    if "_NORMAL" in eid_upper:
        return "normal"
    if "_WEAK" in eid_upper:
        return "weak"
    return "unknown"


def _load_full_pool(
    *,
    base_path: str,
    curated_subset: str | None,
    snapshot_character: str | None,
) -> CombatSnapshotPool:
    """Load the full curated snapshot pool once per checkpoint.

    ``snapshot_character`` is the snapshot-row-side filter (e.g.
    ``CHARACTER.IRONCLAD``); it differs from the bridge ``--character`` flag
    (e.g. ``ironclad``) which only feeds combat_reset.

    Per-encounter restriction happens at reset time via
    ``options={"encounter_id": ...}`` so we do not need a per-encounter pool.
    Loading the full set once also avoids 0-row failures for encounters that
    happen to have no matching curated rows under a particular subset.
    """
    return CombatSnapshotPool.from_path(
        base_path,
        curated_subset=curated_subset,
        character=snapshot_character,
        sample_mode="encounter_balanced",
    )


def _make_obs_encoder(network: Any) -> Any:
    obs_mode = str(getattr(network, "obs_mode", "token_v3")).strip().lower()
    if obs_mode == "token_v3":
        return WorldTokenObservationEncoder(use_text=False)
    return DictObservationEncoder(use_text=False)


def _accumulate_offenders(
    diag: dict[str, Any] | None,
    counters: dict[str, int],
) -> None:
    if not isinstance(diag, dict):
        return
    for key in OFFENDER_DIAG_KEYS:
        v = diag.get(key)
        try:
            if float(v or 0.0) > 0.0:
                counters[key] = counters.get(key, 0) + 1
        except (TypeError, ValueError):
            pass


def _run_episode(
    *,
    network: Any,
    mcts: MCTS,
    env: CombatSandboxEnv,
    encounter_id: str,
    seed: int,
    max_steps: int,
    temperature: float,
) -> EpisodeOutcome:
    offender_counts: dict[str, int] = {}
    try:
        obs, info = env.reset(seed=seed, options={"encounter_id": encounter_id, "seed": seed})
    except Exception as exc:  # noqa: BLE001
        return EpisodeOutcome(
            encounter_id=encounter_id,
            seed=seed,
            win=False,
            terminated=False,
            truncated=True,
            total_reward=0.0,
            length=0,
            final_hp=0.0,
            final_max_hp=0.0,
            error=f"reset_failed: {exc!r}",
        )

    total_reward = 0.0
    length = 0
    last_info = info
    terminated = False
    truncated = False
    for step_idx in range(max_steps):
        action_mask = info.get("action_mask")
        if action_mask is None:
            terminated = True
            break
        try:
            action_idx, _ = mcts.run(network, obs, action_mask, temperature=temperature)
        except Exception as exc:  # noqa: BLE001
            return EpisodeOutcome(
                encounter_id=encounter_id,
                seed=seed,
                win=False,
                terminated=False,
                truncated=True,
                total_reward=total_reward,
                length=length,
                final_hp=0.0,
                final_max_hp=0.0,
                offender_step_rates={k: v / max(1, length) for k, v in offender_counts.items()},
                error=f"mcts_failed_step_{step_idx}: {exc!r}",
            )
        try:
            obs, reward, terminated, truncated, info = env.step(int(action_idx))
        except Exception as exc:  # noqa: BLE001
            return EpisodeOutcome(
                encounter_id=encounter_id,
                seed=seed,
                win=False,
                terminated=False,
                truncated=True,
                total_reward=total_reward,
                length=length,
                final_hp=0.0,
                final_max_hp=0.0,
                offender_step_rates={k: v / max(1, length) for k, v in offender_counts.items()},
                error=f"step_failed_{step_idx}: {exc!r}",
            )
        total_reward += float(reward)
        length += 1
        _accumulate_offenders(info.get("action_diagnostics"), offender_counts)
        last_info = info
        if terminated or truncated:
            break

    final_hp, final_max_hp = _player_hp_from_raw_obs(last_info.get("raw_obs"))
    win = bool(terminated and (not truncated) and final_hp > 0.0)
    return EpisodeOutcome(
        encounter_id=encounter_id,
        seed=seed,
        win=win,
        terminated=bool(terminated),
        truncated=bool(truncated),
        total_reward=float(total_reward),
        length=length,
        final_hp=float(final_hp),
        final_max_hp=float(final_max_hp),
        offender_step_rates={k: v / max(1, length) for k, v in offender_counts.items()},
    )


def _evaluate_checkpoint(
    *,
    checkpoint_dir: str,
    boss_encounters: list[str],
    elite_encounters: list[str],
    normal_encounters: list[str],
    episodes_boss: int,
    episodes_elite: int,
    episodes_normal: int,
    session_file: str | None,
    character: str | None,
    snapshot_character: str | None,
    snapshot_dataset: str,
    snapshot_subset: str | None,
    device: str,
    num_simulations: int,
    temperature: float,
    max_steps: int,
    base_seed: int,
    report_dir: Path,
    tag: str,
) -> CheckpointResult:
    print(f"[eval] === checkpoint: {checkpoint_dir} ===", flush=True)
    network = load_muzero_network(checkpoint_dir, device=device)
    obs_encoder = _make_obs_encoder(network)
    mcts = MCTS(num_simulations=num_simulations)
    mcts.set_training_step(mcts.root_bias_decay_steps)
    mcts.set_root_bias_enabled(True)
    mcts.set_semantic_rollout_enabled(True)

    result = CheckpointResult(checkpoint=str(checkpoint_dir), tag=tag)
    per_episode_dir = report_dir / "per_episode"
    per_episode_dir.mkdir(parents=True, exist_ok=True)

    plan = (
        [(eid, episodes_boss) for eid in boss_encounters]
        + [(eid, episodes_elite) for eid in elite_encounters]
        + [(eid, episodes_normal) for eid in normal_encounters]
    )

    try:
        full_pool = _load_full_pool(
            base_path=snapshot_dataset,
            curated_subset=snapshot_subset,
            snapshot_character=snapshot_character,
        )
    except Exception as exc:  # noqa: BLE001
        print(f"[eval] {tag}: failed to load full snapshot pool: {exc!r}", flush=True)
        traceback.print_exc()
        del network
        if device == "cuda":
            torch.cuda.empty_cache()
        return result
    print(f"[eval] {tag}: full pool rows={len(full_pool)}", flush=True)

    for encounter_id, n_episodes in plan:
        if n_episodes <= 0:
            continue
        per_enc_path = per_episode_dir / f"{tag}__{encounter_id}.jsonl"
        env = CombatSandboxEnv(
            session_file=session_file,
            character=character,
            encounter_id=None,
            encounter_pool=[encounter_id],
            snapshot_pool=full_pool,
            obs_encoder=obs_encoder,
            include_debug_info=True,
        )
        try:
            for ep_idx in range(n_episodes):
                seed = base_seed + ep_idx
                started = time.perf_counter()
                outcome = _run_episode(
                    network=network,
                    mcts=mcts,
                    env=env,
                    encounter_id=encounter_id,
                    seed=seed,
                    max_steps=max_steps,
                    temperature=temperature,
                )
                elapsed = time.perf_counter() - started
                result.episodes.append(outcome)
                with per_enc_path.open("a", encoding="utf-8") as fh:
                    fh.write(json.dumps(asdict(outcome), ensure_ascii=False) + "\n")
                tag_outcome = "WIN" if outcome.win else ("TRUNC" if outcome.truncated else "LOSS")
                err = f" err={outcome.error}" if outcome.error else ""
                print(
                    f"[eval] {tag} {encounter_id} ep{ep_idx} {tag_outcome} "
                    f"R={outcome.total_reward:+.2f} L={outcome.length} "
                    f"hp={outcome.final_hp:.0f}/{outcome.final_max_hp:.0f} "
                    f"t={elapsed:.1f}s{err}",
                    flush=True,
                )
        finally:
            try:
                env.close()
            except Exception:  # noqa: BLE001
                pass

    # Free GPU memory between checkpoints
    del network
    if device == "cuda":
        torch.cuda.empty_cache()
    return result


def _aggregate_checkpoint(result: CheckpointResult) -> dict[str, Any]:
    by_tier: dict[str, list[EpisodeOutcome]] = {"boss": [], "elite": [], "normal": [], "weak": [], "unknown": []}
    by_encounter: dict[str, list[EpisodeOutcome]] = {}
    for ep in result.episodes:
        by_tier[_tier_for_encounter(ep.encounter_id)].append(ep)
        by_encounter.setdefault(ep.encounter_id, []).append(ep)

    def _stats(eps: list[EpisodeOutcome]) -> dict[str, float]:
        if not eps:
            return {"n": 0, "win_rate": 0.0, "reward_mean": 0.0, "length_mean": 0.0}
        wins = sum(1 for e in eps if e.win)
        rewards = [e.total_reward for e in eps]
        lengths = [e.length for e in eps]
        return {
            "n": len(eps),
            "win_rate": wins / len(eps),
            "reward_mean": float(np.mean(rewards)) if rewards else 0.0,
            "length_mean": float(np.mean(lengths)) if lengths else 0.0,
        }

    tier_stats = {tier: _stats(eps) for tier, eps in by_tier.items()}
    encounter_stats = {eid: _stats(eps) for eid, eps in by_encounter.items()}

    # offender rates: average across all episodes weighted by episode length
    offender_totals: dict[str, float] = {k: 0.0 for k in OFFENDER_DIAG_KEYS}
    offender_steps_total = 0
    for ep in result.episodes:
        for k in OFFENDER_DIAG_KEYS:
            offender_totals[k] += ep.offender_step_rates.get(k, 0.0) * ep.length
        offender_steps_total += ep.length
    offender_rates = {
        k: (offender_totals[k] / offender_steps_total) if offender_steps_total > 0 else 0.0
        for k in OFFENDER_DIAG_KEYS
    }
    offender_score = float(sum(offender_rates.values()))

    boss_win = tier_stats["boss"]["win_rate"]
    elite_win = tier_stats["elite"]["win_rate"]
    normal_win = tier_stats["normal"]["win_rate"]
    rewards_all = [e.total_reward for e in result.episodes]
    if rewards_all:
        # crude normalization to [0,1] using observed range across this checkpoint
        rmin = float(min(rewards_all))
        rmax = float(max(rewards_all))
        norm_reward = float((np.mean(rewards_all) - rmin) / (rmax - rmin)) if rmax > rmin else 0.5
    else:
        norm_reward = 0.0

    score = (
        0.40 * boss_win
        + 0.20 * elite_win
        + 0.15 * normal_win
        + 0.15 * norm_reward
        - 0.10 * offender_score
    )

    return {
        "checkpoint": result.checkpoint,
        "tag": result.tag,
        "episode_total": len(result.episodes),
        "tier_stats": tier_stats,
        "encounter_stats": encounter_stats,
        "offender_rates": offender_rates,
        "offender_score": offender_score,
        "norm_reward": norm_reward,
        "score": score,
    }


def _write_markdown(report_dir: Path, summaries: list[dict[str, Any]]) -> None:
    md_path = report_dir / "checkpoint_eval_summary.md"
    lines: list[str] = []
    lines.append(f"# Checkpoint eval summary\n")
    lines.append(f"_Generated {time.strftime('%Y-%m-%d %H:%M:%S')}_\n")
    lines.append("\n## Overall ranking\n")
    lines.append("| rank | tag | score | boss_win | elite_win | normal_win | norm_reward | offender |\n")
    lines.append("|---:|---|---:|---:|---:|---:|---:|---:|\n")
    ranked = sorted(summaries, key=lambda s: -s["score"])
    for idx, s in enumerate(ranked, start=1):
        lines.append(
            f"| {idx} | `{s['tag']}` | {s['score']:.4f} | "
            f"{s['tier_stats']['boss']['win_rate']:.3f} | "
            f"{s['tier_stats']['elite']['win_rate']:.3f} | "
            f"{s['tier_stats']['normal']['win_rate']:.3f} | "
            f"{s['norm_reward']:.3f} | {s['offender_score']:.3f} |\n"
        )

    lines.append("\n## Per-encounter win rate\n")
    encounters = sorted({eid for s in summaries for eid in s["encounter_stats"]})
    header = "| tag | " + " | ".join(eid.split(".")[-1].lower() for eid in encounters) + " |"
    sep = "|---|" + "|".join(["---:"] * len(encounters)) + "|"
    lines.append(header + "\n")
    lines.append(sep + "\n")
    for s in ranked:
        cells = []
        for eid in encounters:
            stat = s["encounter_stats"].get(eid)
            if stat is None or stat["n"] == 0:
                cells.append("-")
            else:
                cells.append(f"{stat['win_rate']:.2f} (n={stat['n']})")
        lines.append(f"| `{s['tag']}` | " + " | ".join(cells) + " |\n")

    lines.append("\n## Offender rates (per step)\n")
    keys = list(OFFENDER_DIAG_KEYS)
    header = "| tag | " + " | ".join(keys) + " |"
    sep = "|---|" + "|".join(["---:"] * len(keys)) + "|"
    lines.append(header + "\n")
    lines.append(sep + "\n")
    for s in ranked:
        cells = [f"{s['offender_rates'].get(k, 0.0):.4f}" for k in keys]
        lines.append(f"| `{s['tag']}` | " + " | ".join(cells) + " |\n")

    md_path.write_text("".join(lines), encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Compare MuZero checkpoints on a fixed eval matrix.")
    parser.add_argument("--checkpoints", nargs="+", required=True, help="Checkpoint directories to compare.")
    parser.add_argument("--checkpoint-tags", nargs="*", default=None, help="Optional short tags (one per checkpoint).")
    parser.add_argument("--episodes-boss", type=int, default=8)
    parser.add_argument("--episodes-elite", type=int, default=6)
    parser.add_argument("--episodes-normal", type=int, default=6)
    parser.add_argument("--boss-encounters", nargs="*", default=list(DEFAULT_BOSS_ENCOUNTERS))
    parser.add_argument("--elite-encounters", nargs="*", default=list(DEFAULT_ELITE_ENCOUNTERS))
    parser.add_argument("--normal-encounters", nargs="*", default=list(DEFAULT_NORMAL_ENCOUNTERS))
    parser.add_argument("--session-file", type=str, default=None)
    parser.add_argument("--character", type=str, default="ironclad",
                        help="Bridge-side character id (lowercase, fed to combat_reset).")
    parser.add_argument("--snapshot-character", type=str, default="CHARACTER.IRONCLAD",
                        help="Snapshot-row-side character filter (uppercase namespaced id).")
    parser.add_argument("--snapshot-dataset", type=str, required=True)
    parser.add_argument("--snapshot-subset", type=str, default=None)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--num-simulations", type=int, default=16)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--max-steps", type=int, default=120)
    parser.add_argument("--base-seed", type=int, default=42)
    parser.add_argument("--report-dir", type=str, required=True)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.device == "cuda" and not torch.cuda.is_available():
        print("[eval] CUDA requested but unavailable; falling back to cpu", flush=True)
        args.device = "cpu"

    report_dir = Path(args.report_dir).resolve()
    report_dir.mkdir(parents=True, exist_ok=True)

    tags = args.checkpoint_tags or [Path(p).name for p in args.checkpoints]
    if len(tags) != len(args.checkpoints):
        raise SystemExit("--checkpoint-tags count must match --checkpoints count")

    config_dump = {
        "checkpoints": list(args.checkpoints),
        "tags": tags,
        "episodes_boss": args.episodes_boss,
        "episodes_elite": args.episodes_elite,
        "episodes_normal": args.episodes_normal,
        "boss_encounters": list(args.boss_encounters),
        "elite_encounters": list(args.elite_encounters),
        "normal_encounters": list(args.normal_encounters),
        "session_file": args.session_file,
        "character": args.character,
        "snapshot_character": args.snapshot_character,
        "snapshot_dataset": args.snapshot_dataset,
        "snapshot_subset": args.snapshot_subset,
        "device": args.device,
        "num_simulations": args.num_simulations,
        "temperature": args.temperature,
        "max_steps": args.max_steps,
        "base_seed": args.base_seed,
        "started_at_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }
    (report_dir / "config.json").write_text(
        json.dumps(config_dump, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    summaries: list[dict[str, Any]] = []
    for checkpoint, tag in zip(args.checkpoints, tags):
        result = _evaluate_checkpoint(
            checkpoint_dir=checkpoint,
            boss_encounters=args.boss_encounters,
            elite_encounters=args.elite_encounters,
            normal_encounters=args.normal_encounters,
            episodes_boss=args.episodes_boss,
            episodes_elite=args.episodes_elite,
            episodes_normal=args.episodes_normal,
            session_file=args.session_file,
            character=args.character,
            snapshot_character=args.snapshot_character,
            snapshot_dataset=args.snapshot_dataset,
            snapshot_subset=args.snapshot_subset,
            device=args.device,
            num_simulations=args.num_simulations,
            temperature=args.temperature,
            max_steps=args.max_steps,
            base_seed=args.base_seed,
            report_dir=report_dir,
            tag=tag,
        )
        summary = _aggregate_checkpoint(result)
        summaries.append(summary)
        # Persist incrementally so a crash mid-run still leaves usable data.
        (report_dir / "checkpoint_eval_summary.json").write_text(
            json.dumps(
                {"config": config_dump, "checkpoints": summaries},
                ensure_ascii=False,
                indent=2,
            ),
            encoding="utf-8",
        )
        _write_markdown(report_dir, summaries)

    if not summaries:
        print("[eval] no summaries produced", flush=True)
        return 1

    ranked = sorted(summaries, key=lambda s: -s["score"])
    best = ranked[0]
    (report_dir / "best_checkpoint.txt").write_text(best["checkpoint"] + "\n", encoding="utf-8")

    print(
        f"\n[eval] best={best['tag']} ({best['checkpoint']}) "
        f"score={best['score']:.4f} boss={best['tier_stats']['boss']['win_rate']:.3f} "
        f"elite={best['tier_stats']['elite']['win_rate']:.3f} "
        f"normal={best['tier_stats']['normal']['win_rate']:.3f}",
        flush=True,
    )
    print(f"[eval] report dir: {report_dir}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
