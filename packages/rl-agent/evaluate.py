"""Evaluate a trained STS2 RL agent with text path smoke verification.

Usage:
    python evaluate.py checkpoints/final --episodes 5
    python evaluate.py checkpoints/final --smoke
    python evaluate.py --combat-smoke --encounter-id BattlewornDummyEventEncounter
"""

import argparse
import json
import numpy as np

from sb3_contrib.common.wrappers import ActionMasker

from sts2_env.checkpoint import load_online_checkpoint, load_online_checkpoint_metadata
from sts2_env.combat_env import CombatSandboxEnv
from sts2_env.env_v2 import SlayTheSpire2EnvV2
from sts2_env.observation_v2 import DictObservationEncoder


def mask_fn(env):
    return env.unwrapped.action_masks()


def _has_semantic_markers(text: str) -> bool:
    value = str(text or "").strip()
    return "sig " in value or "tag " in value


def _append_example(bucket: list[str], text: str, *, limit: int = 6) -> None:
    value = str(text or "").strip()
    if not value or value in bucket:
        return
    if len(bucket) < limit:
        bucket.append(value)


def smoke_test(session_file=None, max_steps=200, use_text=True):
    """Run random legal actions and verify text path coverage."""
    obs_encoder = DictObservationEncoder(use_text=use_text)
    env = SlayTheSpire2EnvV2(
        session_file=session_file,
        obs_encoder=obs_encoder,
        render_mode="human",
        include_debug_info=True,
    )

    print(f"[smoke] use_text={use_text}")
    obs, info = env.reset()

    phase_counts = {}
    text_hits = {"action_text": 0, "decision_text": 0, "canonical_text": 0}
    semantic_counts = {
        "hand_card_total": 0,
        "hand_card_semantic": 0,
        "deck_card_total": 0,
        "deck_card_semantic": 0,
        "card_action_total": 0,
        "card_action_semantic": 0,
        "build_card_action_total": 0,
        "build_card_action_semantic": 0,
    }
    semantic_examples = {
        "hand_cards": [],
        "deck_cards": [],
        "card_actions": [],
        "build_card_actions": [],
    }
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
            combat = raw_obs.get("combat", {})
            if isinstance(combat, dict):
                for card in combat.get("hand", []) or []:
                    if not isinstance(card, dict):
                        continue
                    semantic_counts["hand_card_total"] += 1
                    live_text = obs_encoder._build_live_card_text(card)
                    if _has_semantic_markers(live_text):
                        semantic_counts["hand_card_semantic"] += 1
                        _append_example(semantic_examples["hand_cards"], live_text)

            player = raw_obs.get("player", {})
            if isinstance(player, dict):
                for card in player.get("deck_cards", []) or []:
                    if not isinstance(card, dict):
                        continue
                    semantic_counts["deck_card_total"] += 1
                    live_text = obs_encoder._build_live_card_text(card)
                    if _has_semantic_markers(live_text):
                        semantic_counts["deck_card_semantic"] += 1
                        _append_example(semantic_examples["deck_cards"], live_text)

        if isinstance(raw_obs, dict):
            decision = raw_obs.get("decision", {})
            if isinstance(decision, dict) and decision.get("decision_text"):
                text_hits["decision_text"] += 1

        for a in legal:
            if not isinstance(a, dict):
                continue
            action_text = obs_encoder._build_action_text(a)
            if action_text:
                text_hits["action_text"] += 1

            kind = str(a.get("kind") or "").strip()
            has_card_payload = isinstance(a.get("card"), dict)
            is_shop_card = (
                kind == "shop"
                and isinstance(a.get("item"), dict)
                and isinstance((a.get("item") or {}).get("card"), dict)
            )
            if has_card_payload or is_shop_card:
                semantic_counts["card_action_total"] += 1
                if _has_semantic_markers(action_text):
                    semantic_counts["card_action_semantic"] += 1
                    _append_example(semantic_examples["card_actions"], action_text)

            if kind in ("card_reward", "deck_upgrade", "card_selection") or is_shop_card:
                semantic_counts["build_card_action_total"] += 1
                if _has_semantic_markers(action_text):
                    semantic_counts["build_card_action_semantic"] += 1
                    _append_example(semantic_examples["build_card_actions"], action_text)

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
    print("  Semantic coverage:")
    for total_key, hit_key in (
        ("hand_card_total", "hand_card_semantic"),
        ("deck_card_total", "deck_card_semantic"),
        ("card_action_total", "card_action_semantic"),
        ("build_card_action_total", "build_card_action_semantic"),
    ):
        total = semantic_counts[total_key]
        hits = semantic_counts[hit_key]
        ratio = (hits / total) if total else 0.0
        print(f"    {hit_key}: {hits}/{total} ({ratio:.1%})")
    if any(semantic_examples.values()):
        print("  Semantic examples:")
        for bucket_name, samples in semantic_examples.items():
            if samples:
                print(f"    {bucket_name}:")
                for sample in samples:
                    print(f"      - {sample}")

    if use_text:
        enc = get_text_encoder()
        enc.save_cache()
        print(f"  Cache: {cache_before} -> {enc.cache_size}")

    print("[smoke] PASSED")


def route_smoke_test(session_file=None, character="ironclad", max_steps=40, use_text=True):
    """Verify route future-subtree payloads expose QuestionMark semantics end-to-end."""
    from sts2_env.bridge_client import BridgeClient

    if use_text:
        from sts2_env.text_encoder import get_text_encoder
        get_text_encoder().ensure_ready()

    client = BridgeClient(session_path=session_file)
    obs_encoder = DictObservationEncoder(use_text=use_text)

    print(f"[route-smoke] character={character} use_text={use_text}")
    step = client.reset(character=character, defensive_buffs=True, timeout_ms=60_000)
    episode_id = step["episode_id"]

    def choose_action(obs, legal_actions):
        phase = obs.get("phase")
        if phase == "combat":
            attacks = []
            others = []
            end_turn = None
            for index, action in enumerate(legal_actions):
                if action.get("action_id") == "end_turn":
                    end_turn = index
                    continue
                if action.get("kind") != "play_card":
                    continue
                card = action.get("card") or {}
                card_type = card.get("type")
                if card_type == "Attack":
                    attacks.append(index)
                else:
                    others.append(index)
            if attacks:
                return attacks[0]
            if others:
                return others[0]
            return 0 if end_turn is None else end_turn

        if phase in ("reward", "card_reward"):
            for index, action in enumerate(legal_actions):
                reward = action.get("reward") or {}
                if action.get("kind") == "reward" and reward.get("type") == "gold":
                    return index
            for index, action in enumerate(legal_actions):
                action_id = action.get("action_id", "")
                if "skip" in action_id or action.get("kind") == "proceed":
                    return index
            return 0

        return 0

    for step_index in range(max_steps):
        obs = step["obs"]
        legal_actions = step.get("legal_actions") or []
        phase = obs.get("phase", "unknown")

        if phase == "map":
            map_actions = [action for action in legal_actions if action.get("kind") == "map"]
            assert map_actions, "No map actions exposed at map phase."

            encoded = obs_encoder.encode(obs, legal_actions)
            summary_has_question = any(
                (action.get("route_summary") or {}).get("count_question_mark", 0) > 0
                for action in map_actions
            )
            node_has_question = any(
                any(
                    isinstance(node, dict) and node.get("point_type") == "QuestionMark"
                    for node in (action.get("route_nodes") or [])
                )
                for action in map_actions
            )
            canonical_has_question = any(
                "问号" in (action.get("canonical_text") or "")
                for action in map_actions
            )

            assert summary_has_question, "route_summary is missing count_question_mark."
            assert node_has_question, "route_nodes never expose QuestionMark point_type."
            assert canonical_has_question, "canonical_text does not mention 问号."
            assert float(encoded["route_summary"][:, 8].max()) > 0.0, "Encoded route_summary question-mark channel is empty."
            assert float(encoded["route_node_mask"].sum()) > 0.0, "Encoded route node mask is empty."

            samples = []
            for action in map_actions[:4]:
                summary = action.get("route_summary") or {}
                nodes = action.get("route_nodes") or []
                samples.append(
                    {
                        "action_id": action.get("action_id"),
                        "point_type_norm": action.get("point_type_norm"),
                        "count_question_mark": summary.get("count_question_mark"),
                        "next_question_mark_steps": summary.get("next_question_mark_steps"),
                        "node_types": [
                            node.get("point_type")
                            for node in nodes[:8]
                            if isinstance(node, dict)
                        ],
                        "canonical_text": action.get("canonical_text"),
                    }
                )

            print(
                json.dumps(
                    {
                        "phase": phase,
                        "summary_has_question": summary_has_question,
                        "node_has_question": node_has_question,
                        "canonical_has_question": canonical_has_question,
                        "encoded_question_mark_max": float(encoded["route_summary"][:, 8].max()),
                        "encoded_route_node_count": int(encoded["route_node_mask"].sum()),
                        "samples": samples,
                    },
                    ensure_ascii=False,
                    indent=2,
                )
            )
            print("[route-smoke] PASSED")
            return

        if not legal_actions:
            raise RuntimeError(f"No legal actions before reaching map. phase={phase}")

        choice = choose_action(obs, legal_actions)
        action = legal_actions[choice]
        print(f"[route-smoke] step={step_index} phase={phase} action={action.get('action_id')}")
        step = client.step(episode_id, action_index=choice, timeout_ms=30_000)

    raise RuntimeError(f"Map was not reached within {max_steps} steps.")


def evaluate(model_path, n_episodes=5, session_file=None, use_text=None, deterministic=True):
    """Load model and run evaluation episodes."""
    metadata = load_online_checkpoint_metadata(model_path)
    if use_text is None:
        use_text = bool(metadata.get("use_text", True))

    obs_encoder = DictObservationEncoder(use_text=use_text)
    if use_text:
        from sts2_env.text_encoder import get_text_encoder
        get_text_encoder(model_name=metadata.get("text_model")).ensure_ready()
    env = SlayTheSpire2EnvV2(
        session_file=session_file,
        obs_encoder=obs_encoder,
        render_mode="human",
        include_debug_info=True,
    )
    env = ActionMasker(env, mask_fn)

    model, _ = load_online_checkpoint(model_path, env=env)

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


def combat_smoke_test(
    session_file=None,
    encounter_id=None,
    n_episodes=50,
    use_text=True,
):
    """Run random legal actions in combat sandbox mode.

    Tests: reset latency, episode completion, leak detection.
    """
    import time

    if not encounter_id:
        # Try to get a default from the catalog
        from sts2_env.bridge_client import BridgeClient
        client = BridgeClient(session_path=session_file)
        catalog = client.combat_catalog()
        encounters = catalog.get("encounters", [])
        if encounters:
            # Prefer test encounters
            test_encs = [e for e in encounters if e.get("is_mock")]
            encounter_id = (test_encs or encounters)[0]["encounter_id"]
            print(f"[combat-smoke] Auto-selected encounter: {encounter_id}")
        else:
            print("[combat-smoke] No encounters found in catalog. Provide --encounter-id")
            return

    obs_encoder = DictObservationEncoder(use_text=use_text)
    env = CombatSandboxEnv(
        session_file=session_file,
        encounter_id=encounter_id,
        obs_encoder=obs_encoder,
        render_mode="human",
        include_debug_info=True,
    )

    print(f"[combat-smoke] encounter={encounter_id} episodes={n_episodes} use_text={use_text}")

    reset_times = []
    episode_steps = []
    wins = 0
    losses = 0
    errors = 0

    for ep in range(n_episodes):
        t0 = time.perf_counter()
        try:
            obs, info = env.reset()
        except Exception as exc:
            errors += 1
            print(f"  Episode {ep+1}: RESET ERROR: {exc}")
            continue
        reset_dt = time.perf_counter() - t0
        reset_times.append(reset_dt)

        steps = 0
        total_reward = 0.0
        while True:
            legal = info.get("legal_actions", [])
            if not legal:
                break
            action = np.random.randint(len(legal))
            try:
                obs, reward, term, trunc, info = env.step(action)
            except Exception as exc:
                errors += 1
                print(f"  Episode {ep+1} step {steps}: STEP ERROR: {exc}")
                break
            total_reward += reward
            steps += 1
            if term or trunc:
                break

        episode_steps.append(steps)
        raw = info.get("raw_obs", {})
        hp = (raw.get("player") or {}).get("hp", 0)
        if hp > 0 and steps > 0:
            wins += 1
        else:
            losses += 1

        print(
            f"  Episode {ep+1}: steps={steps} reward={total_reward:+.3f} "
            f"reset={reset_dt:.2f}s hp={hp}"
        )

    env.close()

    # Summary
    print(f"\n[combat-smoke] Results ({n_episodes} episodes, {errors} errors):")
    if reset_times:
        print(f"  Reset time: mean={np.mean(reset_times):.2f}s "
              f"min={np.min(reset_times):.2f}s max={np.max(reset_times):.2f}s")
    if episode_steps:
        print(f"  Steps/episode: mean={np.mean(episode_steps):.1f} "
              f"min={np.min(episode_steps)} max={np.max(episode_steps)}")
    print(f"  Wins: {wins}  Losses: {losses}")

    # Leak detection: check if later resets are significantly slower
    if len(reset_times) >= 10:
        first_half = np.mean(reset_times[:len(reset_times)//2])
        second_half = np.mean(reset_times[len(reset_times)//2:])
        drift = second_half - first_half
        if drift > 1.0:
            print(f"  WARNING: Reset latency drift: {drift:+.2f}s (possible leak)")
        else:
            print(f"  Latency drift: {drift:+.2f}s (OK)")

    status = "PASSED" if errors == 0 else "FAILED"
    print(f"[combat-smoke] {status}")


def main():
    parser = argparse.ArgumentParser(description="Evaluate STS2 agent")
    parser.add_argument("model", nargs="?", help="Model checkpoint path")
    parser.add_argument("--episodes", type=int, default=5)
    parser.add_argument("--session-file", type=str, default=None)
    parser.add_argument("--smoke", action="store_true", help="Run smoke test with random actions")
    parser.add_argument("--route-smoke", action="store_true", help="Run route future-subtree smoke test")
    parser.add_argument("--combat-smoke", action="store_true", help="Run combat sandbox smoke test")
    parser.add_argument("--encounter-id", type=str, default=None, help="Encounter ID for combat sandbox")
    parser.add_argument("--combat-episodes", type=int, default=50, help="Number of combat sandbox episodes")
    parser.add_argument("--no-text", action="store_true", default=False)
    parser.add_argument("--character", type=str, default="ironclad")
    parser.add_argument("--max-steps", type=int, default=200)
    args = parser.parse_args()

    use_text = not args.no_text

    if args.route_smoke:
        route_smoke_test(
            session_file=args.session_file,
            character=args.character,
            max_steps=args.max_steps,
            use_text=use_text,
        )
    elif args.combat_smoke:
        combat_smoke_test(
            session_file=args.session_file,
            encounter_id=args.encounter_id,
            n_episodes=args.combat_episodes,
            use_text=use_text,
        )
    elif args.smoke:
        smoke_test(session_file=args.session_file, max_steps=args.max_steps, use_text=use_text)
    elif args.model:
        use_text_override = False if args.no_text else None
        evaluate(args.model, n_episodes=args.episodes, session_file=args.session_file, use_text=use_text_override)
    else:
        print("Provide a model path, --smoke, or --combat-smoke")


if __name__ == "__main__":
    main()
