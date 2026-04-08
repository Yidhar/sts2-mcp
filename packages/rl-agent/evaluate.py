"""Evaluate a trained STS2 RL agent with text path smoke verification.

Usage:
    python evaluate.py checkpoints/final --episodes 5
    python evaluate.py checkpoints/final --smoke
    python evaluate.py --combat-smoke --encounter-id BattlewornDummyEventEncounter
"""

import argparse
import json
import numpy as np
import time

from sb3_contrib.common.wrappers import ActionMasker

from combat_snapshot_dataset import CombatSnapshotPool
from content_registry import (
    build_live_enemy_semantic_text,
    build_live_potion_semantic_text,
    build_live_relic_semantic_text,
)
from sts2_env.bridge_client import BridgeError
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


def _is_unknown_episode_error(exc: Exception) -> bool:
    if not isinstance(exc, BridgeError):
        return False
    body = exc.response_body
    if isinstance(body, dict) and str(body.get("error") or "").strip() == "unknown_episode_id":
        return True
    return False


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
        "enemy_total": 0,
        "enemy_semantic": 0,
        "relic_total": 0,
        "relic_semantic": 0,
        "potion_total": 0,
        "potion_semantic": 0,
        "card_action_total": 0,
        "card_action_semantic": 0,
        "build_card_action_total": 0,
        "build_card_action_semantic": 0,
        "relic_potion_action_total": 0,
        "relic_potion_action_semantic": 0,
    }
    semantic_examples = {
        "hand_cards": [],
        "deck_cards": [],
        "enemies": [],
        "relics": [],
        "potions": [],
        "card_actions": [],
        "build_card_actions": [],
        "relic_potion_actions": [],
    }
    total_actions = 0
    recovered_episode_resets = 0
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

                for enemy in combat.get("enemies", []) or []:
                    if not isinstance(enemy, dict):
                        continue
                    semantic_counts["enemy_total"] += 1
                    enemy_text = build_live_enemy_semantic_text(enemy)
                    if _has_semantic_markers(enemy_text):
                        semantic_counts["enemy_semantic"] += 1
                    _append_example(semantic_examples["enemies"], enemy_text)

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

                for relic in player.get("relics", []) or []:
                    if not isinstance(relic, dict):
                        continue
                    semantic_counts["relic_total"] += 1
                    relic_text = build_live_relic_semantic_text(relic)
                    if _has_semantic_markers(relic_text):
                        semantic_counts["relic_semantic"] += 1
                    _append_example(semantic_examples["relics"], relic_text)

                for potion in player.get("potions", []) or []:
                    if not isinstance(potion, dict) or potion.get("title") == "[empty]":
                        continue
                    semantic_counts["potion_total"] += 1
                    potion_text = build_live_potion_semantic_text(potion)
                    if _has_semantic_markers(potion_text):
                        semantic_counts["potion_semantic"] += 1
                    _append_example(semantic_examples["potions"], potion_text)

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

            has_relic_payload = isinstance(a.get("relic"), dict) or (
                isinstance(a.get("reward"), dict) and isinstance((a.get("reward") or {}).get("relic"), dict)
            ) or (
                kind == "shop"
                and isinstance(a.get("item"), dict)
                and isinstance((a.get("item") or {}).get("relic"), dict)
            )
            has_potion_payload = isinstance(a.get("potion"), dict) or (
                isinstance(a.get("reward"), dict) and isinstance((a.get("reward") or {}).get("potion"), dict)
            ) or (
                kind == "shop"
                and isinstance(a.get("item"), dict)
                and isinstance((a.get("item") or {}).get("potion"), dict)
            )
            if has_relic_payload or has_potion_payload:
                semantic_counts["relic_potion_action_total"] += 1
                if _has_semantic_markers(action_text):
                    semantic_counts["relic_potion_action_semantic"] += 1
                _append_example(semantic_examples["relic_potion_actions"], action_text)

        # Random action
        action = np.random.randint(len(legal))
        total_actions += 1
        try:
            obs, reward, term, trunc, info = env.step(action)
        except Exception as exc:
            if not _is_unknown_episode_error(exc):
                raise
            recovered_episode_resets += 1
            print(f"  [smoke] recovered unknown_episode_id at step {step+1}; resetting episode")
            obs, info = env.reset()
            continue

        if term or trunc:
            print(f"  Episode done at step {step+1}: reward={reward:.3f}")
            obs, info = env.reset()

    env.close()

    print(f"\n[smoke] Results ({total_actions} actions):")
    print(f"  Phases: {phase_counts}")
    print(f"  Text hits: {text_hits}")
    print(f"  Episode recoveries: {recovered_episode_resets}")
    print("  Semantic coverage:")
    for total_key, hit_key in (
        ("hand_card_total", "hand_card_semantic"),
        ("deck_card_total", "deck_card_semantic"),
        ("enemy_total", "enemy_semantic"),
        ("relic_total", "relic_semantic"),
        ("potion_total", "potion_semantic"),
        ("card_action_total", "card_action_semantic"),
        ("build_card_action_total", "build_card_action_semantic"),
        ("relic_potion_action_total", "relic_potion_action_semantic"),
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


def _get_live_supported_encounter_ids(session_file=None):
    from sts2_env.bridge_client import BridgeClient

    client = BridgeClient(session_path=session_file)
    catalog = client.combat_catalog()
    return [
        entry.get("encounter_id")
        for entry in (catalog.get("encounters") or [])
        if entry.get("encounter_id")
    ]


def combat_smoke_test(
    session_file=None,
    encounter_id=None,
    n_episodes=50,
    use_text=True,
    combat_snapshot_dataset=None,
    combat_snapshot_split="train",
    combat_snapshot_character=None,
    combat_snapshot_max_rows=None,
    combat_snapshot_sample_mode="encounter_balanced",
):
    """Run random legal actions in combat sandbox mode.

    Tests: reset latency, episode completion, leak detection.
    """
    import time

    snapshot_pool = None
    if combat_snapshot_dataset:
        supported_encounter_ids = _get_live_supported_encounter_ids(session_file=session_file)
        snapshot_pool = CombatSnapshotPool.from_path(
            combat_snapshot_dataset,
            split=(combat_snapshot_split or None),
            character=combat_snapshot_character,
            encounter_ids=supported_encounter_ids,
            max_rows=combat_snapshot_max_rows,
            sample_mode=combat_snapshot_sample_mode,
        )
        print(f"[combat-smoke] snapshot_pool={snapshot_pool.summary()}")

    if not encounter_id and snapshot_pool is None:
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
        snapshot_pool=snapshot_pool,
        obs_encoder=obs_encoder,
        render_mode="human",
        include_debug_info=True,
    )

    print(
        f"[combat-smoke] encounter={encounter_id} episodes={n_episodes} "
        f"use_text={use_text} snapshot_dataset={combat_snapshot_dataset}"
    )

    reset_times = []
    episode_steps = []
    wins = 0
    losses = 0
    errors = 0
    recovered_episode_resets = 0

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
                if _is_unknown_episode_error(exc):
                    recovered_episode_resets += 1
                    print(f"  Episode {ep+1} step {steps}: recovered unknown_episode_id via reset")
                    obs, info = env.reset()
                    steps = 0
                    total_reward = 0.0
                    continue
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
    print(f"  Episode recoveries: {recovered_episode_resets}")
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


def full_run_reset_smoke(
    session_file=None,
    n_resets=20,
    timeout_ms=60000,
    include_actions=False,
):
    """Call env/reset repeatedly to verify full-run reset stability."""
    import time

    from sts2_env.bridge_client import BridgeClient, BridgeError

    client = BridgeClient(session_path=session_file)
    print(
        f"[full-run-reset-smoke] session={session_file or 'default'} "
        f"resets={n_resets} timeout_ms={timeout_ms}"
    )

    reset_stats = []
    errors = []
    for idx in range(1, n_resets + 1):
        t0 = time.perf_counter()
        try:
            result = client.reset(timeout_ms=timeout_ms)
        except BridgeError as exc:
            errors.append(
                {
                    "iteration": idx,
                    "error": str(exc),
                    "status": exc.status_code,
                    "body": exc.response_body,
                }
            )
            print(f"  Reset {idx}: ERROR ({exc.status_code}) {exc}")
            continue

        dt = time.perf_counter() - t0
        obs = result.get("obs") or result.get("state") or {}
        info = result.get("info") or {}
        phase = obs.get("phase", "unknown")
        screen = info.get("screen") or obs.get("screen", "unknown")
        actionable = bool(info.get("actionable", obs.get("actionable", False)))
        done = bool(result.get("done", info.get("done", obs.get("done", False))))
        legal = result.get("legal_actions", []) or []
        legal_count = len(legal)
        episode_id = result.get("episode_id")
        reset_stats.append(
            {
                "iteration": idx,
                "duration_s": dt,
                "episode_id": episode_id,
                "phase": phase,
                "screen": screen,
                "actionable": actionable,
                "done": done,
                "legal_count": legal_count,
            }
        )
        summary = (
            f"  Reset {idx}: phase={phase} screen={screen} "
            f"actionable={actionable} legal={legal_count} dt={dt:.2f}s"
        )
        if episode_id:
            summary += f" episode_id={episode_id}"
        print(summary)
        if include_actions and legal:
            action_ids = [a.get("action_id") for a in legal[:5]]
            print(f"    sample actions: {action_ids}")

    print("\n[full-run-reset-smoke] SUMMARY")
    print(f"  Successful resets: {len(reset_stats)} / {n_resets}")
    if reset_stats:
        durations = [stat["duration_s"] for stat in reset_stats]
        print(
            f"  Duration: mean={np.mean(durations):.2f}s "
            f"min={np.min(durations):.2f}s max={np.max(durations):.2f}s"
        )
        phases = {}
        for stat in reset_stats:
            phases[stat["phase"]] = phases.get(stat["phase"], 0) + 1
        print(f"  Phase distribution: {phases}")
    if errors:
        print(f"  Errors: {len(errors)}")
        for error in errors:
            print(
                f"    #{error['iteration']} status={error['status']} body={error['body']}"
            )
    status = "PASSED" if not errors else "FAILED"
    print(f"[full-run-reset-smoke] {status}")


def _route_target_depth(action: dict, target_point_type: str) -> int | None:
    point_type = action.get("point_type")
    point_type_norm = action.get("point_type_norm")
    if point_type == target_point_type or point_type_norm == target_point_type:
        return 1

    best = None
    for node in action.get("route_nodes") or []:
        if not isinstance(node, dict):
            continue
        if node.get("point_type") != target_point_type:
            continue
        depth = node.get("depth")
        if isinstance(depth, int):
            best = depth if best is None else min(best, depth)
    return best


def _choose_combat_action(legal_actions: list[dict]) -> int:
    attacks = []
    non_attack_cards = []
    zero_cost = []
    end_turn = None

    for index, action in enumerate(legal_actions):
        if action.get("action_id") == "end_turn":
            end_turn = index
            continue
        if action.get("kind") != "play_card":
            continue
        card = action.get("card") or {}
        cost = card.get("cost")
        card_type = card.get("type")
        if cost == 0:
            zero_cost.append(index)
        elif card_type == "Attack":
            attacks.append(index)
        else:
            non_attack_cards.append(index)

    if zero_cost:
        return zero_cost[0]
    if attacks:
        return attacks[0]
    if non_attack_cards:
        return non_attack_cards[0]
    if end_turn is not None:
        return end_turn
    return 0


def _choose_map_action_toward(legal_actions: list[dict], target_point_type: str) -> int:
    map_actions = [(index, action) for index, action in enumerate(legal_actions) if action.get("kind") == "map"]
    if not map_actions:
        return 0

    direct = next(
        (
            index
            for index, action in map_actions
            if action.get("point_type") == target_point_type or action.get("point_type_norm") == target_point_type
        ),
        None,
    )
    if direct is not None:
        return direct

    scored = []
    for index, action in map_actions:
        depth = _route_target_depth(action, target_point_type)
        if depth is None:
            continue
        route_summary = action.get("route_summary") or {}
        question_marks = int(route_summary.get("count_question_mark") or 0)
        forced_path = int(route_summary.get("forced_path_steps_before_branch") or 0)
        scored.append(((depth, question_marks, forced_path), index))

    if scored:
        scored.sort(key=lambda item: item[0])
        return scored[0][1]

    return map_actions[0][0]


def _choose_reward_action(legal_actions: list[dict], *, prefer_card_reward: bool = False) -> int:
    gold = next(
        (index for index, action in enumerate(legal_actions) if (action.get("reward") or {}).get("type") == "gold"),
        None,
    )
    card = next(
        (index for index, action in enumerate(legal_actions) if (action.get("reward") or {}).get("type") == "card"),
        None,
    )
    relic = next(
        (index for index, action in enumerate(legal_actions) if (action.get("reward") or {}).get("type") == "relic"),
        None,
    )
    proceed = next(
        (index for index, action in enumerate(legal_actions) if action.get("action_id") == "proceed"),
        None,
    )

    if prefer_card_reward and card is not None:
        return card
    if gold is not None:
        return gold
    if card is not None:
        return card
    if relic is not None:
        return relic
    if proceed is not None:
        return proceed
    return 0


def _choose_card_reward_action(legal_actions: list[dict], *, pick_first: bool = False) -> int:
    if pick_first:
        pick = next((index for index, action in enumerate(legal_actions) if action.get("selection") == "pick"), None)
        if pick is not None:
            return pick
    skip = next((index for index, action in enumerate(legal_actions) if action.get("selection") == "skip"), None)
    if skip is not None:
        return skip
    return 0


def _choose_deck_upgrade_action(legal_actions: list[dict]) -> int:
    select_index = next(
        (
            index
            for index, action in enumerate(legal_actions)
            if str(action.get("selection") or "").strip().lower() in ("select_card", "pick")
        ),
        None,
    )
    if select_index is not None:
        return select_index

    card_index = next(
        (
            index
            for index, action in enumerate(legal_actions)
            if action.get("kind") == "deck_upgrade" and isinstance(action.get("card"), dict)
        ),
        None,
    )
    if card_index is not None:
        return card_index

    confirm_index = next(
        (
            index
            for index, action in enumerate(legal_actions)
            if str(action.get("selection") or "").strip().lower() == "confirm"
        ),
        None,
    )
    if confirm_index is not None:
        return confirm_index

    close_index = next(
        (
            index
            for index, action in enumerate(legal_actions)
            if str(action.get("selection") or "").strip().lower() in ("close", "cancel")
        ),
        None,
    )
    if close_index is not None:
        return close_index

    return 0


def _choose_event_action(legal_actions: list[dict], *, prefer_non_proceed: bool = False) -> int:
    if prefer_non_proceed:
        choice = next(
            (index for index, action in enumerate(legal_actions) if not bool(action.get("proceed"))),
            None,
        )
        if choice is not None:
            return choice
    return 0


def _step_env(client, episode_id: str, legal_actions: list[dict], index: int, *, timeout_ms: int, label: str):
    action_id = legal_actions[index].get("action_id")
    print(f"    [{label}] {action_id}")
    return client.step(episode_id, action_index=index, timeout_ms=timeout_ms)


def _logic_hash_of(step: dict) -> str:
    info = step.get("info") or {}
    if isinstance(info, dict):
        return str(info.get("logic_hash") or "").strip()
    return ""


def _action_id_prefixes(legal_actions: list[dict]) -> set[str]:
    prefixes: set[str] = set()
    for action in legal_actions:
        action_id = str(action.get("action_id") or "")
        if ":" in action_id:
            prefixes.add(action_id.split(":", 1)[0])
    return prefixes


def _is_transient_reset_issue(exc: Exception) -> bool:
    if not isinstance(exc, BridgeError):
        return False
    body = exc.response_body
    if isinstance(body, dict):
        code = str(body.get("error") or "").strip()
        return code in {
            "env_reset_no_reset_path",
            "env_reset_transition_limit",
            "missing_or_invalid_token",
        }
    return exc.status_code in (401, 409)


def _reset_with_ready_gate(client, *, timeout_ms: int, ready_timeout_ms: int | None = None) -> dict:
    budget_ms = ready_timeout_ms or max(timeout_ms, 90_000)
    deadline = time.perf_counter() + (budget_ms / 1000.0)
    last_exc: Exception | None = None

    while time.perf_counter() < deadline:
        try:
            step = _fresh_surface_reset(client, timeout_ms=timeout_ms)
            phase = str((step.get("obs") or {}).get("phase") or "")
            legal_actions = step.get("legal_actions") or []
            if legal_actions:
                return step
            last_exc = RuntimeError(f"reset returned no legal actions at phase={phase or 'unknown'}")
        except Exception as exc:
            if not _is_transient_reset_issue(exc):
                raise
            last_exc = exc
        time.sleep(1.0)

    if last_exc is not None:
        raise last_exc
    raise RuntimeError("reset ready gate timed out without a usable episode")


OPENING_EVENT_TITLE_PHASE_HINTS = {
    "card_selection": {"精准剪刀", "铅制镇纸", "新叶"},
    "reward": {"失物盒", "小型扭蛋"},
    "deck_upgrade": {"橙型香盒"},
}


def _choose_startup_action(step: dict, phase: str, legal_actions: list[dict]) -> int:
    if phase == "startup_main_menu":
        preferred = (
            "main_menu:confirm_abandon_run",
            "main_menu:abandon_current_game",
            "main_menu:singleplayer",
            "main_menu:new_game",
            "main_menu:continue",
            "main_menu:cancel_abandon_run",
        )
        for action_id in preferred:
            index = next(
                (idx for idx, action in enumerate(legal_actions) if action.get("action_id") == action_id),
                None,
            )
            if index is not None:
                return index
        return 0

    if phase == "startup_run_mode":
        index = next(
            (idx for idx, action in enumerate(legal_actions) if action.get("action_id") == "run_mode:standard"),
            None,
        )
        if index is not None:
            return index
        return 0

    if phase == "startup_character_select":
        decision = step.get("obs", {}).get("decision") or {}
        selected_index = decision.get("selected_index") if isinstance(decision, dict) else None
        embark_index = next(
            (idx for idx, action in enumerate(legal_actions) if action.get("action_id") == "embark"),
            None,
        )
        if selected_index is not None and embark_index is not None:
            return embark_index

        preferred_character_index = next(
            (
                idx
                for idx, action in enumerate(legal_actions)
                if str(action.get("action_id") or "").startswith("character_select:")
            ),
            None,
        )
        if preferred_character_index is not None:
            return preferred_character_index

        if embark_index is not None:
            return embark_index
        return 0

    return 0


def _fresh_surface_reset(client, *, timeout_ms: int) -> dict:
    # Use a sentinel character name to force env/reset through the active-run
    # recovery path instead of reusing the current actionable episode, without
    # forcing the harness through the full startup/menu flow every time.
    return client.reset(character="__force_reset__", defensive_buffs=True, timeout_ms=timeout_ms)


def _choose_event_action_with_context(step: dict, legal_actions: list[dict], *, repeat_count: int) -> int:
    non_proceed = [index for index, action in enumerate(legal_actions) if not bool(action.get("proceed"))]
    proceed = [index for index, action in enumerate(legal_actions) if bool(action.get("proceed"))]

    if not non_proceed and proceed:
        return proceed[0]
    if not non_proceed:
        return 0

    # If the same event logic hash repeats, cycle non-proceed options first,
    # then fall back to proceed to break out of pathological event loops.
    if repeat_count >= max(3, len(non_proceed) + 1) and proceed:
        return proceed[0]
    return non_proceed[min(repeat_count, len(non_proceed) - 1)]


def _choose_actions_phase_action(step: dict, legal_actions: list[dict], *, route_target: str | None, event_repeat_count: int) -> int:
    if any(action.get("kind") == "map" for action in legal_actions):
        return next(
            (
                index
                for index, action in enumerate(legal_actions)
                if action.get("kind") == "map"
            ),
            0 if route_target is None else _choose_map_action_toward(legal_actions, route_target),
        ) if route_target is None else _choose_map_action_toward(legal_actions, route_target)

    prefixes = _action_id_prefixes(legal_actions)
    if prefixes & {"main_menu", "run_mode", "character_select"} or any(
        str(action.get("action_id") or "") == "embark" for action in legal_actions
    ):
        if "main_menu" in prefixes:
            return _choose_startup_action(step, "startup_main_menu", legal_actions)
        if "run_mode" in prefixes:
            return _choose_startup_action(step, "startup_run_mode", legal_actions)
        return _choose_startup_action(step, "startup_character_select", legal_actions)

    if any(action.get("kind") == "event_option" for action in legal_actions):
        return _choose_event_action_with_context(step, legal_actions, repeat_count=event_repeat_count)
    if any(action.get("kind") == "reward" for action in legal_actions):
        return _choose_reward_action(legal_actions)
    if any(action.get("kind") == "card_reward" for action in legal_actions):
        return _choose_card_reward_action(legal_actions)
    if any(action.get("kind") == "shop" for action in legal_actions):
        leave_index = next((index for index, action in enumerate(legal_actions) if action.get("shop_action") == "leave"), None)
        if leave_index is not None:
            return leave_index
        open_index = next((index for index, action in enumerate(legal_actions) if action.get("shop_action") == "open"), None)
        if open_index is not None:
            return open_index
    if any(action.get("kind") == "rest_site" for action in legal_actions):
        return next(
            (
                index
                for index, action in enumerate(legal_actions)
                if ((action.get("option") or {}).get("title") or "") in ("休息", "锻造")
            ),
            0,
        )
    if any(action.get("kind") == "deck_upgrade" for action in legal_actions):
        return _choose_deck_upgrade_action(legal_actions)

    proceed_index = next((index for index, action in enumerate(legal_actions) if action.get("action_id") == "proceed"), None)
    if proceed_index is not None:
        return proceed_index
    discard_index = next((index for index, action in enumerate(legal_actions) if action.get("kind") == "discard_potion"), None)
    if discard_index is not None:
        return discard_index
    return 0


def _navigate_to_surface(
    client,
    episode_id: str,
    step: dict,
    *,
    target_phase: str,
    timeout_ms: int,
    max_steps: int,
    route_target: str | None = None,
):
    repeat_counts: dict[tuple[str, str], int] = {}
    for step_idx in range(max_steps):
        phase = step.get("obs", {}).get("phase")
        legal_actions = step.get("legal_actions") or []
        logic_hash = _logic_hash_of(step)
        repeat_key = (str(phase or ""), logic_hash)
        repeat_count = repeat_counts.get(repeat_key, 0)
        repeat_counts[repeat_key] = repeat_count + 1

        if phase == target_phase:
            return step, step_idx

        if not legal_actions:
            raise RuntimeError(f"navigate_to_surface({target_phase}) found no legal actions at phase={phase}")

        if phase in ("startup_main_menu", "startup_run_mode", "startup_character_select"):
            action_index = _choose_startup_action(step, phase, legal_actions)
        elif phase == "event":
            action_index = _choose_event_action_with_context(step, legal_actions, repeat_count=repeat_count)
        elif phase == "card_selection":
            action_index = 0
        elif phase == "combat":
            action_index = _choose_combat_action(legal_actions)
        elif phase == "reward":
            action_index = _choose_reward_action(legal_actions)
        elif phase == "card_reward":
            action_index = _choose_card_reward_action(legal_actions)
        elif phase == "shop":
            action_index = next(
                (index for index, action in enumerate(legal_actions) if action.get("shop_action") == "leave"),
                0,
            )
        elif phase == "rest_site":
            action_index = next(
                (
                    index
                    for index, action in enumerate(legal_actions)
                    if ((action.get("option") or {}).get("title") or "") in ("休息", "锻造")
                ),
                0,
            )
        elif phase == "deck_upgrade":
            action_index = _choose_deck_upgrade_action(legal_actions)
        elif phase == "map":
            if route_target is None:
                action_index = next((index for index, action in enumerate(legal_actions) if action.get("kind") == "map"), 0)
            else:
                action_index = _choose_map_action_toward(legal_actions, route_target)
        elif phase == "actions":
            action_index = _choose_actions_phase_action(
                step,
                legal_actions,
                route_target=route_target,
                event_repeat_count=repeat_count,
            )
        else:
            action_index = 0

        step = _step_env(client, episode_id, legal_actions, action_index, timeout_ms=timeout_ms, label=f"nav:{phase}")

    raise RuntimeError(f"navigate_to_surface({target_phase}) exceeded {max_steps} steps")


def _try_opening_event_surface(
    client,
    *,
    target_phase: str,
    timeout_ms: int,
    max_probe_resets: int = 4,
):
    hint_titles = OPENING_EVENT_TITLE_PHASE_HINTS.get(target_phase) or set()
    for probe_index in range(1, max_probe_resets + 1):
        step = _reset_with_ready_gate(client, timeout_ms=timeout_ms)
        phase = step.get("obs", {}).get("phase")
        legal_actions = step.get("legal_actions") or []
        if phase != "event":
            continue

        option_index = next(
            (
                index
                for index, action in enumerate(legal_actions)
                if str(action.get("title") or "").strip() in hint_titles
            ),
            None,
        )
        if option_index is None:
            continue

        episode_id = step["episode_id"]
        step = _step_env(
            client,
            episode_id,
            legal_actions,
            option_index,
            timeout_ms=timeout_ms,
            label=f"{target_phase}:opening_event",
        )
        if step.get("obs", {}).get("phase") == target_phase:
            return step, episode_id, probe_index

    return None, None, 0


def _resolve_card_selection_surface(client, episode_id: str, step: dict, *, timeout_ms: int, max_steps: int = 4) -> dict:
    attempted_selects = 0
    for _ in range(max_steps):
        if step.get("obs", {}).get("phase") != "card_selection":
            return step

        legal_actions = step.get("legal_actions") or []
        decision = step.get("obs", {}).get("decision") or {}
        if not isinstance(decision, dict):
            decision = {}

        confirm_like_index = next(
            (
                index
                for index, action in enumerate(legal_actions)
                if str(action.get("selection") or "").strip().lower() in ("confirm", "close", "cancel", "skip")
                or str(action.get("action_id") or "").startswith("card_selection:confirm")
            ),
            None,
        )
        if confirm_like_index is not None and (
            bool(decision.get("confirm_ready"))
            or bool(decision.get("can_skip"))
            or bool(decision.get("cancelable"))
            or str((legal_actions[confirm_like_index].get("selection") or "")).strip().lower() in ("close", "cancel", "skip")
        ):
            terminal_label = str(legal_actions[confirm_like_index].get("selection") or "confirm").strip().lower()
            step = _step_env(
                client,
                episode_id,
                legal_actions,
                confirm_like_index,
                timeout_ms=timeout_ms,
                label=f"card_selection:{terminal_label}",
            )
            continue

        select_candidates = [
            index
            for index, action in enumerate(legal_actions)
            if str(action.get("selection") or "").strip().lower() in ("select", "pick", "select_card")
        ]
        if not select_candidates:
            return step

        if attempted_selects >= len(select_candidates):
            return step

        select_index = select_candidates[attempted_selects]
        attempted_selects += 1
        step = _step_env(
            client,
            episode_id,
            legal_actions,
            select_index,
            timeout_ms=timeout_ms,
            label="card_selection:select",
        )

    return step


def _run_card_selection_soak(client, *, timeout_ms: int) -> dict:
    opening_step, opening_episode_id, opening_probe_count = _try_opening_event_surface(
        client,
        target_phase="card_selection",
        timeout_ms=timeout_ms,
        max_probe_resets=4,
    )
    if opening_step is not None and opening_episode_id is not None:
        step = opening_step
        episode_id = opening_episode_id
        nav_steps = 1
    else:
        step = _reset_with_ready_gate(client, timeout_ms=timeout_ms)
        episode_id = step["episode_id"]
        step, nav_steps = _navigate_to_surface(
            client,
            episode_id,
            step,
            target_phase="card_selection",
            timeout_ms=timeout_ms,
            max_steps=80,
        )

    step = _resolve_card_selection_surface(client, episode_id, step, timeout_ms=timeout_ms, max_steps=4)
    phase_after = step.get("obs", {}).get("phase")

    if phase_after == "card_selection":
        raise RuntimeError("card_selection soak did not exit selection surface after select/confirm")

    return {
        "surface": "card_selection",
        "nav_steps": nav_steps,
        "opening_probe_count": opening_probe_count,
        "phase_after": phase_after,
        "legal_after_count": len(step.get("legal_actions") or []),
    }


def _run_reward_soak(client, *, timeout_ms: int, max_nav_steps: int) -> dict:
    step = _reset_with_ready_gate(client, timeout_ms=timeout_ms)
    episode_id = step["episode_id"]
    step, nav_steps = _navigate_to_surface(
        client,
        episode_id,
        step,
        target_phase="reward",
        route_target="Monster",
        timeout_ms=timeout_ms,
        max_steps=max_nav_steps,
    )

    seen_reward_types = []
    saw_card_reward = False
    picked_card = None

    while True:
        phase = step.get("obs", {}).get("phase")
        legal_actions = step.get("legal_actions") or []

        if phase == "reward":
            reward_types = [((action.get("reward") or {}).get("type")) for action in legal_actions if action.get("kind") == "reward"]
            seen_reward_types.extend([item for item in reward_types if item])
            if any(item == "card" for item in reward_types):
                action_index = _choose_reward_action(legal_actions, prefer_card_reward=True)
            else:
                action_index = _choose_reward_action(legal_actions)
            step = _step_env(client, episode_id, legal_actions, action_index, timeout_ms=timeout_ms, label="reward")
            continue

        if phase == "card_reward":
            legal_actions = step.get("legal_actions") or []
            pick_index = next((index for index, action in enumerate(legal_actions) if action.get("selection") == "pick"), None)
            if pick_index is not None:
                picked_card = ((legal_actions[pick_index].get("card") or {}).get("title"))
                step = _step_env(client, episode_id, legal_actions, pick_index, timeout_ms=timeout_ms, label="card_reward:pick")
            else:
                action_index = _choose_card_reward_action(legal_actions)
                step = _step_env(client, episode_id, legal_actions, action_index, timeout_ms=timeout_ms, label="card_reward:skip")
            saw_card_reward = True
            continue

        if phase == "map":
            break

        if not legal_actions:
            break

        step = _step_env(client, episode_id, legal_actions, 0, timeout_ms=timeout_ms, label=f"reward:fallback:{phase}")

    return {
        "surface": "reward",
        "nav_steps": nav_steps,
        "saw_card_reward": saw_card_reward,
        "picked_card": picked_card,
        "seen_reward_types": seen_reward_types,
        "phase_after": step.get("obs", {}).get("phase"),
    }


def _run_shop_soak(client, *, timeout_ms: int, max_nav_steps: int) -> dict:
    step = _reset_with_ready_gate(client, timeout_ms=timeout_ms)
    episode_id = step["episode_id"]
    step, nav_steps = _navigate_to_surface(
        client,
        episode_id,
        step,
        target_phase="shop",
        route_target="Shop",
        timeout_ms=timeout_ms,
        max_steps=max_nav_steps,
    )

    legal_actions = step.get("legal_actions") or []
    open_index = next((index for index, action in enumerate(legal_actions) if action.get("shop_action") == "open"), None)
    leave_index = next((index for index, action in enumerate(legal_actions) if action.get("shop_action") == "leave"), None)
    if open_index is None or leave_index is None:
        raise RuntimeError(f"shop soak expected open/leave actions, got {[action.get('action_id') for action in legal_actions]}")

    step = _step_env(client, episode_id, legal_actions, open_index, timeout_ms=timeout_ms, label="shop:open")
    legal_actions = step.get("legal_actions") or []
    if not any(action.get("shop_action") == "buy" for action in legal_actions):
        raise RuntimeError("shop soak expected buy actions after opening merchant inventory")

    back_index = next((index for index, action in enumerate(legal_actions) if action.get("shop_action") == "back"), None)
    if back_index is None:
        raise RuntimeError("shop soak expected shop:back after opening merchant inventory")
    step = _step_env(client, episode_id, legal_actions, back_index, timeout_ms=timeout_ms, label="shop:back")

    legal_actions = step.get("legal_actions") or []
    reopen_index = next((index for index, action in enumerate(legal_actions) if action.get("shop_action") == "open"), None)
    if reopen_index is not None:
        step = _step_env(client, episode_id, legal_actions, reopen_index, timeout_ms=timeout_ms, label="shop:reopen")

    legal_actions = step.get("legal_actions") or []
    back_index = next((index for index, action in enumerate(legal_actions) if action.get("shop_action") == "back"), None)
    if back_index is not None:
        step = _step_env(client, episode_id, legal_actions, back_index, timeout_ms=timeout_ms, label="shop:back2")

    legal_actions = step.get("legal_actions") or []
    leave_index = next((index for index, action in enumerate(legal_actions) if action.get("shop_action") == "leave"), None)
    if leave_index is None:
        raise RuntimeError("shop soak expected shop:leave from closed shop state")
    step = _step_env(client, episode_id, legal_actions, leave_index, timeout_ms=timeout_ms, label="shop:leave")

    phase_after = step.get("obs", {}).get("phase")
    return {
        "surface": "shop",
        "nav_steps": nav_steps,
        "phase_after": phase_after,
        "legal_after_count": len(step.get("legal_actions") or []),
    }


def _run_rest_soak(client, *, timeout_ms: int, max_nav_steps: int) -> dict:
    step = _reset_with_ready_gate(client, timeout_ms=timeout_ms)
    episode_id = step["episode_id"]
    step, nav_steps = _navigate_to_surface(
        client,
        episode_id,
        step,
        target_phase="rest_site",
        route_target="RestSite",
        timeout_ms=timeout_ms,
        max_steps=max_nav_steps,
    )

    legal_actions = step.get("legal_actions") or []
    smith_index = next(
        (
            index
            for index, action in enumerate(legal_actions)
            if ((action.get("option") or {}).get("title") or "") == "锻造"
        ),
        None,
    )
    rest_index = next(
        (
            index
            for index, action in enumerate(legal_actions)
            if ((action.get("option") or {}).get("title") or "") == "休息"
        ),
        None,
    )
    if smith_index is None and rest_index is None:
        raise RuntimeError(f"rest soak expected campfire options, got {[action.get('action_id') for action in legal_actions]}")

    chose = "smith" if smith_index is not None else "rest"
    step = _step_env(
        client,
        episode_id,
        legal_actions,
        smith_index if smith_index is not None else rest_index,
        timeout_ms=timeout_ms,
        label=f"rest_site:{chose}",
    )

    upgrade_picked = None
    phase = step.get("obs", {}).get("phase")
    for _ in range(4):
        if phase == "deck_upgrade":
            legal_actions = step.get("legal_actions") or []
            upgrade_index = next(
                (
                    index
                    for index, action in enumerate(legal_actions)
                    if (
                        str(action.get("selection") or "").strip().lower() in ("select_card", "pick", "select")
                        or str(action.get("action_id") or "").startswith("deck_upgrade:select:", 0)
                    )
                    and isinstance(action.get("card"), dict)
                ),
                None,
            )
            if upgrade_index is None:
                upgrade_index = next(
                    (
                        index
                        for index, action in enumerate(legal_actions)
                        if action.get("kind") == "deck_upgrade" and isinstance(action.get("card"), dict)
                    ),
                    None,
                )
            if upgrade_index is not None:
                upgrade_picked = ((legal_actions[upgrade_index].get("card") or {}).get("title"))
                step = _step_env(client, episode_id, legal_actions, upgrade_index, timeout_ms=timeout_ms, label="deck_upgrade:select")
                phase = step.get("obs", {}).get("phase")
                continue

            terminal_index = next(
                (
                    index
                    for index, action in enumerate(legal_actions)
                    if str(action.get("selection") or "").strip().lower() in ("confirm", "close", "cancel")
                ),
                None,
            )
            if terminal_index is not None:
                terminal_label = str(legal_actions[terminal_index].get("selection") or "terminal").strip().lower()
                step = _step_env(client, episode_id, legal_actions, terminal_index, timeout_ms=timeout_ms, label=f"deck_upgrade:{terminal_label}")
                phase = step.get("obs", {}).get("phase")
                continue

            raise RuntimeError(
                "deck_upgrade surface exposed no selectable or terminal actions: "
                f"{[(action.get('action_id'), action.get('selection')) for action in legal_actions]}"
            )

        if phase == "rest_site":
            legal_actions = step.get("legal_actions") or []
            proceed_index = next(
                (index for index, action in enumerate(legal_actions) if action.get("action_id") == "rest_site:proceed"),
                None,
            )
            if proceed_index is None:
                break
            step = _step_env(client, episode_id, legal_actions, proceed_index, timeout_ms=timeout_ms, label="rest_site:proceed")
            phase = step.get("obs", {}).get("phase")
            continue

        break

    return {
        "surface": "rest",
        "nav_steps": nav_steps,
        "chose": chose,
        "upgrade_picked": upgrade_picked,
        "phase_after": phase,
    }


def surface_soak_test(
    session_file=None,
    surfaces: list[str] | None = None,
    cycles: int = 1,
    timeout_ms: int = 60_000,
    max_nav_steps: int = 240,
):
    """Run targeted soak flows for unstable full-run surfaces."""
    from sts2_env.bridge_client import BridgeClient

    supported = ("reward", "shop", "rest", "card_selection")
    requested = surfaces or list(supported)
    for surface in requested:
        if surface not in supported:
            raise ValueError(f"Unsupported surface soak target: {surface}")

    client = BridgeClient(session_path=session_file)
    handlers = {
        "reward": _run_reward_soak,
        "shop": _run_shop_soak,
        "rest": _run_rest_soak,
        "card_selection": _run_card_selection_soak,
    }

    summaries = []
    failures = []
    print(
        f"[surface-soak] session={session_file or 'default'} surfaces={requested} "
        f"cycles={cycles} timeout_ms={timeout_ms} max_nav_steps={max_nav_steps}"
    )

    max_attempts = 3
    for cycle in range(1, cycles + 1):
        print(f"\n[surface-soak] cycle {cycle}/{cycles}")
        for surface in requested:
            print(f"  -> {surface}")
            last_failure = None
            for attempt in range(1, max_attempts + 1):
                started_at = time.perf_counter()
                try:
                    result = handlers[surface](
                        client,
                        timeout_ms=timeout_ms,
                        max_nav_steps=max_nav_steps,
                    ) if surface != "card_selection" else handlers[surface](
                        client,
                        timeout_ms=timeout_ms,
                    )
                    duration_s = time.perf_counter() - started_at
                    result["duration_s"] = duration_s
                    result["cycle"] = cycle
                    result["attempt"] = attempt
                    summaries.append(result)
                    print(f"     OK {json.dumps(result, ensure_ascii=False)}")
                    last_failure = None
                    break
                except Exception as exc:
                    duration_s = time.perf_counter() - started_at
                    last_failure = {
                        "surface": surface,
                        "cycle": cycle,
                        "attempt": attempt,
                        "duration_s": duration_s,
                        "error": str(exc),
                    }
                    if attempt < max_attempts:
                        print(f"     RETRY {json.dumps(last_failure, ensure_ascii=False)}")
                        time.sleep(1.0)
                        continue
                    failures.append(last_failure)
                    print(f"     FAIL {json.dumps(last_failure, ensure_ascii=False)}")

    print("\n[surface-soak] SUMMARY")
    grouped = {}
    for item in summaries:
        grouped.setdefault(item["surface"], 0)
        grouped[item["surface"]] += 1
    print(f"  successes={grouped}")
    if summaries:
        durations = [item["duration_s"] for item in summaries]
        print(
            f"  duration mean={np.mean(durations):.2f}s "
            f"min={np.min(durations):.2f}s max={np.max(durations):.2f}s"
        )
    if failures:
        print(f"  failures={len(failures)}")
        for failure in failures:
            print(f"    {json.dumps(failure, ensure_ascii=False)}")
    status = "PASSED" if not failures else "FAILED"
    print(f"[surface-soak] {status}")


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
    parser.add_argument("--combat-snapshot-dataset", type=str, default=None, help="Combat snapshot dataset root/file for combat sandbox smoke")
    parser.add_argument("--combat-snapshot-split", type=str, default="train")
    parser.add_argument("--combat-snapshot-character", type=str, default=None)
    parser.add_argument("--combat-snapshot-max-rows", type=int, default=None)
    parser.add_argument(
        "--combat-snapshot-sample-mode",
        type=str,
        default="encounter_balanced",
        choices=("row_uniform", "encounter_balanced"),
    )
    parser.add_argument("--no-text", action="store_true", default=False)
    parser.add_argument("--character", type=str, default="ironclad")
    parser.add_argument("--max-steps", type=int, default=200)
    parser.add_argument(
        "--fullrun-reset-smoke",
        action="store_true",
        help="Run repeated env/reset smoke test for full-run",
    )
    parser.add_argument(
        "--fullrun-reset-count",
        type=int,
        default=10,
        help="Number of env/reset calls to perform",
    )
    parser.add_argument(
        "--fullrun-reset-timeout",
        type=int,
        default=60000,
        help="Timeout (ms) to give each env/reset call",
    )
    parser.add_argument(
        "--fullrun-reset-actions",
        action="store_true",
        help="Log a sample of returned action IDs from each reset",
    )
    parser.add_argument(
        "--surface-soak",
        type=str,
        default=None,
        help="Comma-separated target list: reward,shop,rest,card_selection, or 'all'",
    )
    parser.add_argument(
        "--surface-soak-cycles",
        type=int,
        default=1,
        help="Number of cycles for each requested surface soak target",
    )
    parser.add_argument(
        "--surface-soak-timeout",
        type=int,
        default=60000,
        help="Per-call timeout (ms) for surface soak env/reset and env/step",
    )
    parser.add_argument(
        "--surface-soak-max-nav-steps",
        type=int,
        default=240,
        help="Maximum navigation steps when searching for a target surface",
    )
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
            combat_snapshot_dataset=args.combat_snapshot_dataset,
            combat_snapshot_split=args.combat_snapshot_split,
            combat_snapshot_character=args.combat_snapshot_character,
            combat_snapshot_max_rows=args.combat_snapshot_max_rows,
            combat_snapshot_sample_mode=args.combat_snapshot_sample_mode,
        )
    elif args.smoke:
        smoke_test(session_file=args.session_file, max_steps=args.max_steps, use_text=use_text)
    elif args.fullrun_reset_smoke:
        full_run_reset_smoke(
            session_file=args.session_file,
            n_resets=args.fullrun_reset_count,
            timeout_ms=args.fullrun_reset_timeout,
            include_actions=args.fullrun_reset_actions,
        )
    elif args.surface_soak:
        surfaces = [item.strip() for item in args.surface_soak.split(",") if item.strip()]
        if surfaces == ["all"]:
            surfaces = ["reward", "shop", "rest", "card_selection"]
        surface_soak_test(
            session_file=args.session_file,
            surfaces=surfaces,
            cycles=args.surface_soak_cycles,
            timeout_ms=args.surface_soak_timeout,
            max_nav_steps=args.surface_soak_max_nav_steps,
        )
    elif args.model:
        use_text_override = False if args.no_text else None
        evaluate(args.model, n_episodes=args.episodes, session_file=args.session_file, use_text=use_text_override)
    else:
        print("Provide a model path, --smoke, --combat-smoke, --fullrun-reset-smoke, or --surface-soak")


if __name__ == "__main__":
    main()
