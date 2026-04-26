"""Summarize MuZero replay-buffer failures by encounter and sample action chains.

Preferred entrypoint:
    python -m muzero.analyze_replay --replay-buffer PATH [--flags...]

``analyze_muzero_replay.py`` remains available as a compatibility wrapper.
"""

from __future__ import annotations

import argparse
import json
import pickle
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

from combat_snapshot_dataset import infer_encounter_tier


def load_replay_buffer(path: Path) -> list[Any]:
    with path.open("rb") as handle:
        payload = pickle.load(handle)
    if isinstance(payload, dict):
        return list(payload.get("trajectories", []))
    trajectories = getattr(payload, "trajectories", None)
    if trajectories is not None:
        return list(trajectories)
    state_dict = getattr(payload, "state_dict", lambda: {})()
    if isinstance(state_dict, dict):
        return list(state_dict.get("trajectories", []))
    raise RuntimeError(f"Unsupported replay buffer payload in {path}")


def compact_trace(trajectory: Any, *, max_steps: int) -> list[dict[str, Any]]:
    steps = getattr(trajectory, "steps", []) or []
    trace: list[dict[str, Any]] = []
    for idx, step in enumerate(steps[:max_steps]):
        action_info = step.get("action_info") if isinstance(step, dict) else {}
        action_info = dict(action_info or {})
        trace.append(
            {
                "t": idx,
                "reward": round(float(step.get("reward", 0.0) or 0.0), 4),
                "title": action_info.get("title"),
                "kind": action_info.get("kind"),
                "surface": action_info.get("surface"),
                "selected_index": action_info.get("selected_index"),
                "target_index": action_info.get("target_index"),
                "card_cost": action_info.get("card_cost"),
                "action_id": action_info.get("action_id"),
            }
        )
    return trace


def main() -> None:
    parser = argparse.ArgumentParser(description="Analyze MuZero replay-buffer failures.")
    parser.add_argument("--replay-buffer", required=True, type=Path)
    parser.add_argument("--top-n", type=int, default=12)
    parser.add_argument("--sample-negative", type=int, default=6)
    parser.add_argument("--trace-steps", type=int, default=12)
    args = parser.parse_args()

    trajectories = load_replay_buffer(args.replay_buffer)
    print(json.dumps({"trajectory_count": len(trajectories)}, ensure_ascii=False))

    by_tier = Counter()
    neg_by_tier = Counter()
    by_encounter = Counter()
    neg_by_encounter = Counter()
    reward_by_encounter = defaultdict(list)
    length_by_encounter = defaultdict(list)
    sample_negative_rows: list[dict[str, Any]] = []

    for trajectory in trajectories:
        metadata = getattr(trajectory, "metadata", {}) or {}
        encounter_id = str(metadata.get("encounter_id") or "unknown")
        encounter_tier = str(metadata.get("encounter_tier") or infer_encounter_tier(encounter_id) or "unknown")
        reward = float(metadata.get("episode_total_reward", 0.0) or 0.0)
        episode_length = int(metadata.get("episode_length", len(getattr(trajectory, "steps", []) or [])) or 0)
        negative = bool(metadata.get("negative_reward_episode", reward < 0.0))

        by_tier[encounter_tier] += 1
        by_encounter[encounter_id] += 1
        reward_by_encounter[encounter_id].append(reward)
        length_by_encounter[encounter_id].append(episode_length)
        if negative:
            neg_by_tier[encounter_tier] += 1
            neg_by_encounter[encounter_id] += 1
            if len(sample_negative_rows) < args.sample_negative:
                sample_negative_rows.append(
                    {
                        "encounter_id": encounter_id,
                        "encounter_tier": encounter_tier,
                        "episode_total_reward": reward,
                        "episode_length": episode_length,
                        "trace": compact_trace(trajectory, max_steps=args.trace_steps),
                    }
                )

    for tier, total in sorted(by_tier.items()):
        negative = neg_by_tier[tier]
        print(
            json.dumps(
                {
                    "tier": tier,
                    "total": total,
                    "negative": negative,
                    "negative_rate": round(negative / total, 4) if total else None,
                },
                ensure_ascii=False,
            )
        )

    print("top_negative_encounters")
    for encounter_id, negative in neg_by_encounter.most_common(args.top_n):
        total = by_encounter[encounter_id]
        avg_reward = sum(reward_by_encounter[encounter_id]) / len(reward_by_encounter[encounter_id])
        avg_length = sum(length_by_encounter[encounter_id]) / len(length_by_encounter[encounter_id])
        print(
            json.dumps(
                {
                    "encounter_id": encounter_id,
                    "negative": negative,
                    "total": total,
                    "negative_rate": round(negative / total, 4),
                    "avg_reward": round(avg_reward, 4),
                    "avg_length": round(avg_length, 2),
                },
                ensure_ascii=False,
            )
        )

    print("sample_negative_episodes")
    for row in sample_negative_rows:
        print(json.dumps(row, ensure_ascii=False))


if __name__ == "__main__":
    main()
