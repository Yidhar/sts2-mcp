"""Evaluate a trained STS2 RL agent with text path smoke verification.

Usage:
    python evaluate.py checkpoints_v2/sts2_v2_final.zip --episodes 5
    python evaluate.py checkpoints_v2/sts2_v2_final.zip --smoke
"""

import argparse
import numpy as np

from sb3_contrib import MaskablePPO
from sb3_contrib.common.wrappers import ActionMasker

from sts2_env.env_v2 import SlayTheSpire2EnvV2
from sts2_env.observation_v2 import DictObservationEncoder


def mask_fn(env):
    return env.unwrapped.action_masks()


def smoke_test(session_file=None, max_steps=200, use_text=True):
    """Run random legal actions and verify text path coverage."""
    obs_encoder = DictObservationEncoder(use_text=use_text)
    env = SlayTheSpire2EnvV2(
        session_file=session_file,
        obs_encoder=obs_encoder,
        render_mode="human",
    )

    print(f"[smoke] use_text={use_text}")
    obs, info = env.reset()

    phase_counts = {}
    text_hits = {"action_text": 0, "decision_text": 0, "canonical_text": 0}
    total_actions = 0
    cache_before = 0

    if use_text:
        from sts2_env.text_encoder import get_text_encoder
        enc = get_text_encoder().ensure_ready()
        cache_before = enc.cache_size

    for step in range(max_steps):
        legal = info.get("legal_actions", [])
        if not legal:
            break

        phase = info.get("phase", "unknown")
        phase_counts[phase] = phase_counts.get(phase, 0) + 1

        # Check text coverage
        for a in legal:
            if isinstance(a, dict) and a.get("canonical_text"):
                text_hits["canonical_text"] += 1

        raw_obs = info.get("raw_obs", {})
        if isinstance(raw_obs, dict):
            decision = raw_obs.get("decision", {})
            if isinstance(decision, dict) and decision.get("decision_text"):
                text_hits["decision_text"] += 1

        # Random action
        action = np.random.randint(len(legal))
        total_actions += 1
        obs, reward, term, trunc, info = env.step(action)

        if term or trunc:
            print(f"  Episode done at step {step+1}: reward={reward:.3f}")
            obs, info = env.reset()

    env.close()

    print(f"\n[smoke] Results ({total_actions} actions):")
    print(f"  Phases: {phase_counts}")
    print(f"  Text hits: {text_hits}")

    if use_text:
        enc = get_text_encoder()
        enc.save_cache()
        print(f"  Cache: {cache_before} -> {enc.cache_size}")

    print("[smoke] PASSED")


def evaluate(model_path, n_episodes=5, session_file=None, use_text=True, deterministic=True):
    """Load model and run evaluation episodes."""
    obs_encoder = DictObservationEncoder(use_text=use_text)
    if use_text:
        from sts2_env.text_encoder import get_text_encoder
        get_text_encoder().ensure_ready()
    env = SlayTheSpire2EnvV2(
        session_file=session_file,
        obs_encoder=obs_encoder,
        render_mode="human",
    )
    env = ActionMasker(env, mask_fn)

    model = MaskablePPO.load(model_path, env=env)

    results = []
    for ep in range(n_episodes):
        obs, info = env.reset()
        total_reward = 0
        steps = 0

        while True:
            action_masks = info.get("action_mask")
            action, _ = model.predict(obs, action_masks=action_masks, deterministic=deterministic)
            obs, reward, term, trunc, info = env.step(int(action))
            total_reward += reward
            steps += 1
            if term or trunc:
                break

        phase = info.get("phase", "?")
        print(f"  Episode {ep+1}: reward={total_reward:+.3f} steps={steps} phase={phase}")
        results.append(total_reward)

    env.close()

    avg = np.mean(results)
    print(f"\nResults ({n_episodes} episodes): avg_reward={avg:+.3f}")


def main():
    parser = argparse.ArgumentParser(description="Evaluate STS2 agent")
    parser.add_argument("model", nargs="?", help="Model checkpoint path")
    parser.add_argument("--episodes", type=int, default=5)
    parser.add_argument("--session-file", type=str, default=None)
    parser.add_argument("--smoke", action="store_true", help="Run smoke test with random actions")
    parser.add_argument("--no-text", action="store_true", default=False)
    parser.add_argument("--max-steps", type=int, default=200)
    args = parser.parse_args()

    use_text = not args.no_text

    if args.smoke:
        smoke_test(session_file=args.session_file, max_steps=args.max_steps, use_text=use_text)
    elif args.model:
        evaluate(args.model, n_episodes=args.episodes, session_file=args.session_file, use_text=use_text)
    else:
        print("Provide a model path or use --smoke")


if __name__ == "__main__":
    main()
