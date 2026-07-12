#!/usr/bin/env python3
"""Planner sanity probe (Step-0 verify tool for RC-3).

The RC-3 finding: in ``--combat-policy-mode direct`` the search-free planner's
``planner_q`` may be uninformative and/or systematically prefer ``end_turn``,
which drives the passive-collapse / turtle policy. This probe REPLAYS real combat
frames from a checkpoint's saved replay buffer (no live game needed), runs the
exact direct-policy planner call (``initial_inference`` -> ``action_rollout_planner``),
and reports two RC-3 metrics to watch ACROSS the RC-3 fixes:

  * passivity: on frames whose EXECUTED action was end_turn (so its index is known),
    how often does the planner rank end_turn #1 among legal actions, and the gap
    planner_q[end_turn] - max(planner_q over other legal). A high #1-rate / positive
    gap = the planner actively prefers ending the turn. This should DROP after the
    RC-3 entropy/temperature/value-target fixes land.
  * informativeness: the mean planner_q spread (max-min over legal actions). Near
    zero = the planner barely distinguishes actions, so selection falls back to the
    base policy prior (end_turn). This should RISE as the value head learns.

Also reports planner/policy agreement (planner argmax vs the action actually taken).

Note: a precise "lethal-vs-end_turn margin" needs raw enemy HP, which the replay
buffer does not store (raw obs is JSONL-only); that variant requires a live/sim
env frame and is left as a follow-up. This buffer-replay probe needs no live game
and validates offline against the checkpoint + its replay_buffer.pkl.

Usage:
    python scripts/planner_sanity_probe.py                      # auto-pick newest checkpoint
    python scripts/planner_sanity_probe.py --checkpoint checkpoints/<run>/muzero_step_XXX
    python scripts/planner_sanity_probe.py --device cuda --max-frames 200 --rollout-steps 4 --beam 4
"""

from __future__ import annotations

import argparse
import json
import pickle
import sys
from pathlib import Path

_SCRIPTS_DIR = Path(__file__).resolve().parent
_RL_AGENT = _SCRIPTS_DIR.parent
if str(_RL_AGENT) not in sys.path:
    sys.path.insert(0, str(_RL_AGENT))

import numpy as np  # noqa: E402
import torch  # noqa: E402

from muzero.evaluate import load_muzero_network  # noqa: E402
from muzero.sts2_env.muzero_buffer import MuZeroReplayBuffer, _batched_observations_to_numpy  # noqa: E402
from sts2_rl.artifacts import resolve_artifact_path, resolve_external_input_path  # noqa: E402


def _auto_checkpoint() -> Path | None:
    root = resolve_artifact_path(None, default="checkpoints")
    if not root.exists():
        return None
    candidates = [
        d for d in root.glob("*/muzero_step_*")
        if (d / "network.pt").exists() and (d / "replay_buffer.pkl").exists()
    ]
    if not candidates:
        return None
    return max(candidates, key=lambda d: d.stat().st_mtime)


def _iter_combat_steps(buffer: MuZeroReplayBuffer, limit: int):
    seen = 0
    for trajectory in buffer.trajectories:
        for step in trajectory.steps:
            if str(step.get("decision_domain") or "").strip().lower() != "combat":
                continue
            mask = np.asarray(step.get("action_mask"), dtype=np.float32).reshape(-1)
            if int((mask > 0).sum()) < 2:
                continue
            yield step, mask
            seen += 1
            if seen >= limit:
                return


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--checkpoint", type=str, default=None, help="Checkpoint dir with network.pt + replay_buffer.pkl.")
    ap.add_argument("--device", type=str, default=None, help="cpu|cuda (default: cuda if available).")
    ap.add_argument("--max-frames", type=int, default=200)
    ap.add_argument("--rollout-steps", type=int, default=4, help="Planner lookahead depth (eval profile).")
    ap.add_argument("--beam", type=int, default=4, help="Planner continuation beam width.")
    ap.add_argument("--planner-memory-profile", type=str, default="eval")
    ap.add_argument("--end-turn-top-fail-rate", type=float, default=0.6,
                    help="FAIL if the planner ranks end_turn #1 on more than this fraction of end_turn frames.")
    args = ap.parse_args()

    ckpt = resolve_external_input_path(args.checkpoint) if args.checkpoint else _auto_checkpoint()
    if ckpt is None or not (ckpt / "network.pt").exists():
        print("No checkpoint with network.pt + replay_buffer.pkl found. Pass --checkpoint.")
        return 2
    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")

    print(f"[probe] checkpoint={ckpt}")
    print(f"[probe] device={device} rollout_steps={args.rollout_steps} beam={args.beam}")
    network, profile = load_muzero_network(ckpt, device=device, planner_memory_profile=args.planner_memory_profile)
    network.eval()

    buffer = MuZeroReplayBuffer()
    with (ckpt / "replay_buffer.pkl").open("rb") as handle:
        buffer.load_state_dict(pickle.load(handle))
    n_traj = len(buffer.trajectories)
    print(f"[probe] loaded buffer: {n_traj} trajectories")

    spreads: list[float] = []
    planner_agreements = 0
    total = 0
    end_turn_frames = 0
    end_turn_ranked_top = 0
    end_turn_gaps: list[float] = []

    for step, mask in _iter_combat_steps(buffer, args.max_frames):
        try:
            # The buffer stores PACKED obs (trimmed rows); _batched_observations_to_numpy
            # scatters them back to full fixed shapes (batch dim = 1), matching the training
            # unpack path (eval_latent_probes uses the same conversion).
            obs_np = _batched_observations_to_numpy([step["obs"]])
            obs_torch = {
                key: torch.as_tensor(value, dtype=torch.float32, device=device)
                for key, value in obs_np.items()
            }
            oc_raw = step.get("objective_context")  # packing drops this from obs; it is step-level
            objective_context = (
                torch.as_tensor(np.asarray(oc_raw, dtype=np.float32), device=device).reshape(1, -1)
                if oc_raw is not None else None
            )
            with torch.no_grad():
                initial = network.initial_inference(obs_torch)
                mask_t = (torch.as_tensor(mask, device=initial.policy_logits.device) > 0).reshape(1, -1)
                rollout = network.action_rollout_planner(
                    initial.hidden_state,
                    initial.action_embeddings,
                    action_mask=mask_t,
                    objective_context=objective_context,
                    decision_domain=obs_torch.get("decision_domain"),
                    discount=0.997,
                    rollout_steps=int(args.rollout_steps),
                    continuation_beam_width=int(args.beam),
                )
            q = rollout.planner_q.squeeze(0).float().cpu().numpy().reshape(-1)
        except Exception as exc:  # skip malformed frames, keep the probe robust
            print(f"[probe] skipped a frame: {type(exc).__name__}: {exc}")
            continue

        legal = np.nonzero(mask > 0)[0]
        legal = legal[legal < q.shape[0]]
        if legal.size < 2:
            continue
        legal_q = q[legal]
        total += 1
        spreads.append(float(legal_q.max() - legal_q.min()))
        planner_arg = int(legal[int(np.argmax(legal_q))])
        chosen = int(step.get("action", -1))
        if planner_arg == chosen:
            planner_agreements += 1

        if str(step.get("action_family") or "").strip().lower() == "end_turn" and 0 <= chosen < q.shape[0]:
            end_turn_frames += 1
            order = legal[np.argsort(-legal_q)]
            rank = int(np.where(order == chosen)[0][0]) + 1 if chosen in order else len(order)
            if rank == 1:
                end_turn_ranked_top += 1
            other = legal_q[legal != chosen]
            end_turn_gaps.append(float(q[chosen] - (other.max() if other.size else q[chosen])))

    if total == 0:
        print("[probe] no usable combat frames in this buffer.")
        return 2

    et_top_rate = (end_turn_ranked_top / end_turn_frames) if end_turn_frames else 0.0
    summary = {
        "checkpoint": str(ckpt),
        "combat_frames": total,
        "planner_q_spread_mean": round(float(np.mean(spreads)), 5),
        "planner_q_spread_median": round(float(np.median(spreads)), 5),
        "planner_policy_agreement_rate": round(planner_agreements / total, 4),
        "end_turn_frames": end_turn_frames,
        "end_turn_ranked_top1_rate": round(et_top_rate, 4),
        "end_turn_gap_vs_best_alt_mean": round(float(np.mean(end_turn_gaps)), 5) if end_turn_gaps else None,
    }
    print("[probe] " + json.dumps(summary, ensure_ascii=False))
    print("\nWatch across RC-3 fixes: end_turn_ranked_top1_rate should DROP and "
          "planner_q_spread_mean should RISE (planner becomes informative & less passive).")

    failed = end_turn_frames >= 10 and et_top_rate > float(args.end_turn_top_fail_rate)
    if failed:
        print(f"VERDICT: planner ranks end_turn #1 on {et_top_rate:.0%} of end_turn frames "
              f"(> {args.end_turn_top_fail_rate:.0%}) -> passivity bias confirmed (RC-3 target).")
    else:
        print("VERDICT: planner does not strongly prefer end_turn on this sample "
              "(or too few end_turn frames to judge).")
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
