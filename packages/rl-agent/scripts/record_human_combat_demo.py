#!/usr/bin/env python3
"""Interactively play a combat sandbox episode and record human demos.

This is intentionally terminal-based: it uses the same ``CombatSandboxEnv``
legal-action list that MuZero sees, asks a human to choose an action index, then
records the pre-action raw observation, legal actions, and chosen action id to
``human_demos/<session>/decisions.jsonl``.

The output file is directly loadable by ``muzero.demo_dataset``.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

RL_ROOT = Path(__file__).resolve().parents[1]
if str(RL_ROOT) not in sys.path:
    sys.path.insert(0, str(RL_ROOT))

from sts2_env.combat_env import CombatSandboxEnv  # noqa: E402
from sts2_env.human_demo_recorder import HumanDemoRecorder  # noqa: E402
from sts2_rl.artifacts import resolve_artifact_path, resolve_external_input_path  # noqa: E402


def _parse_csv(value: str | None) -> list[str] | None:
    if value is None or not value.strip():
        return None
    return [item.strip() for item in value.split(",") if item.strip()]


def _parse_int(value: str | None) -> int | None:
    if value is None or value == "":
        return None
    return int(value)


def _short_json(value: Any, max_len: int = 90) -> str:
    try:
        text = json.dumps(value, ensure_ascii=False, sort_keys=True)
    except Exception:
        text = str(value)
    if len(text) > max_len:
        return text[: max_len - 1] + "…"
    return text


def _card_label(action: dict[str, Any]) -> str:
    card = action.get("card") if isinstance(action.get("card"), dict) else {}
    for key in ("title", "name", "id", "card_id", "internal_id"):
        val = card.get(key)
        if val:
            return str(val)
    return ""


def _target_label(action: dict[str, Any]) -> str:
    for key in ("target_name", "target", "target_combat_id"):
        val = action.get(key)
        if val not in (None, ""):
            return str(val)
    return ""


def _action_summary(action: dict[str, Any]) -> str:
    family = action.get("family") or action.get("kind") or ""
    aid = action.get("action_id") or ""
    card = _card_label(action)
    target = _target_label(action)
    semantic = action.get("semantic") if isinstance(action.get("semantic"), dict) else {}
    roles = semantic.get("roles") if isinstance(semantic, dict) else None
    runtime = action.get("runtime") if isinstance(action.get("runtime"), dict) else {}
    card_runtime = (action.get("card") or {}).get("runtime") if isinstance(action.get("card"), dict) else None
    pieces = [str(family), str(aid)]
    if card:
        pieces.append(f"card={card}")
    if target:
        pieces.append(f"target={target}")
    if roles:
        pieces.append(f"roles={','.join(map(str, roles)) if isinstance(roles, list) else roles}")
    if runtime:
        pieces.append(f"runtime={_short_json(runtime, 60)}")
    if card_runtime:
        pieces.append(f"card_runtime={_short_json(card_runtime, 60)}")
    return " | ".join(p for p in pieces if p)


def _print_state(env: CombatSandboxEnv, reward_total: float) -> None:
    obs = env.raw_obs or {}
    player = obs.get("player") if isinstance(obs.get("player"), dict) else {}
    combat = obs.get("combat") if isinstance(obs.get("combat"), dict) else {}
    enemies = combat.get("enemies") if isinstance(combat.get("enemies"), list) else []
    print("\n" + "=" * 100)
    print(
        f"Episode={getattr(env, '_episode_id', '')} Encounter={getattr(env, '_current_encounter_id', '')} "
        f"Phase={obs.get('phase')} Round={combat.get('round')} "
        f"HP={player.get('hp', player.get('current_hp'))}/{player.get('max_hp')} "
        f"Block={combat.get('block')} Energy={combat.get('energy')} RewardTotal={reward_total:.3f}"
    )
    if enemies:
        print("Enemies:")
        for enemy in enemies:
            if not isinstance(enemy, dict):
                continue
            print(
                "  - "
                f"id={enemy.get('id') or enemy.get('name')} cid={enemy.get('combat_id')} "
                f"hp={enemy.get('hp', enemy.get('current_hp'))}/{enemy.get('max_hp')} "
                f"block={enemy.get('block')} intent={_short_json(enemy.get('intent'), 120)}"
            )
    mask = env.action_masks()
    print("Legal actions:")
    for idx, action in enumerate(env.legal_actions):
        allowed = bool(idx < len(mask) and mask[idx])
        marker = " " if allowed else "!"
        print(f"  {idx:02d}{marker} {_action_summary(action)}")
    print("提示: 输入动作编号；`r` 刷新显示；`tag a,b` 设置下一步 reason_tags；`note ...` 设置下一步备注；`q` 退出。")
    print("      带 `!` 的动作被当前 action mask 屏蔽（通常是自杀扣血等），除非你刻意示范，不建议选。")


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--session-file", default=None)
    parser.add_argument(
        "--output-dir",
        default=None,
        help="Demo directory (default: <STS2_ARTIFACT_ROOT>/human_demos). Relative paths use the artifact root.",
    )
    parser.add_argument("--session-id", default=None)
    parser.add_argument("--source", default="human")
    parser.add_argument("--character", default=None)
    parser.add_argument("--encounter-id", default=None)
    parser.add_argument("--encounter-pool", default=None, help="comma-separated encounter ids")
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--current-hp", default=None)
    parser.add_argument("--max-hp", default=None)
    parser.add_argument("--max-energy", default=None)
    parser.add_argument("--deck", default=None, help="comma-separated card ids/titles")
    parser.add_argument("--relics", default=None, help="comma-separated relic ids/titles")
    parser.add_argument("--potions", default=None, help="comma-separated potion ids/titles")
    parser.add_argument("--gold", default=None)
    parser.add_argument("--reset-timeout-ms", type=int, default=15000)
    parser.add_argument("--step-timeout-ms", type=int, default=20000)
    parser.add_argument("--include-debug-info", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_arg_parser().parse_args(argv)
    args.session_file = (
        str(resolve_external_input_path(args.session_file))
        if args.session_file
        else None
    )
    args.output_dir = str(resolve_artifact_path(args.output_dir, default="human_demos"))
    recorder = HumanDemoRecorder(
        output_dir=args.output_dir,
        session_id=args.session_id,
        source=args.source,
    )
    env = CombatSandboxEnv(
        session_file=args.session_file,
        character=args.character,
        encounter_id=args.encounter_id,
        encounter_pool=_parse_csv(args.encounter_pool),
        seed=args.seed,
        current_hp=_parse_int(args.current_hp),
        max_hp=_parse_int(args.max_hp),
        max_energy=_parse_int(args.max_energy),
        deck=_parse_csv(args.deck),
        relics=_parse_csv(args.relics),
        potions=_parse_csv(args.potions),
        gold=_parse_int(args.gold),
        reset_timeout_ms=args.reset_timeout_ms,
        step_timeout_ms=args.step_timeout_ms,
        include_debug_info=args.include_debug_info,
    )
    # Keep the env-level auto recorder disabled in this script.  We record
    # manually so reason_tags/comment entered at the prompt are captured.
    env._human_demo_recorder = None  # noqa: SLF001

    reward_total = 0.0
    pending_tags: list[str] = []
    pending_note = ""
    try:
        _, info = env.reset()
        recorder.start_episode(
            episode_id=info.get("episode_id"),
            encounter_id=info.get("encounter_id"),
            reset_kwargs=env.last_reset_kwargs,
            obs=env.raw_obs,
            legal_actions=env.legal_actions,
            metadata={
                "tool": "record_human_combat_demo.py",
                "snapshot_sample_id": info.get("snapshot_sample_id"),
                "snapshot_run_id": info.get("snapshot_run_id"),
            },
        )
        done = False
        truncated = False
        while not (done or truncated):
            _print_state(env, reward_total)
            raw = input("> ").strip()
            if not raw or raw.lower() == "r":
                continue
            if raw.lower() in {"q", "quit", "exit"}:
                print("退出：当前 episode 未终止；已关闭 recorder。")
                break
            if raw.lower().startswith("tag "):
                pending_tags = _parse_csv(raw[4:]) or []
                print(f"下一步 reason_tags = {pending_tags}")
                continue
            if raw.lower().startswith("note "):
                pending_note = raw[5:].strip()
                print(f"下一步 comment = {pending_note!r}")
                continue
            try:
                action_index = int(raw)
            except ValueError:
                print(f"无法解析输入: {raw!r}")
                continue
            if action_index < 0 or action_index >= len(env.legal_actions):
                print(f"动作编号越界: {action_index}")
                continue
            prev_obs = env.raw_obs or {}
            legal_before = list(env.legal_actions)
            selected = legal_before[action_index]
            _, reward, done, truncated, info = env.step(action_index)
            reward_total += float(reward)
            recorder.record_decision(
                obs=prev_obs,
                legal_actions=legal_before,
                selected_action=selected,
                selected_action_index=action_index,
                next_obs=env.raw_obs,
                reward=float(reward),
                done=bool(done),
                truncated=bool(truncated),
                info=info,
                reason_tags=pending_tags,
                comment=pending_note,
            )
            pending_tags = []
            pending_note = ""
        print(f"decisions: {recorder.paths.decisions_path}")
        print(f"episodes : {recorder.paths.episodes_path}")
        return 0
    finally:
        recorder.close()
        env.close()


if __name__ == "__main__":
    raise SystemExit(main())
