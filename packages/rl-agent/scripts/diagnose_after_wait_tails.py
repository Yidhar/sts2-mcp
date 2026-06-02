from __future__ import annotations

import argparse
import json
import random
import sys
import time
from collections import Counter, defaultdict
from contextlib import nullcontext
from pathlib import Path
from typing import Any

import numpy as np
import torch
from gymnasium import spaces
from safetensors.torch import load_file
from stable_baselines3.common.utils import obs_as_tensor


RL_AGENT_ROOT = Path(__file__).resolve().parents[1]
if str(RL_AGENT_ROOT) not in sys.path:
    sys.path.insert(0, str(RL_AGENT_ROOT))

from legacy.attention_ppo.train_attention_policy import parse_csv, resolve_snapshot_pool, resolve_text_device, resolve_training_session_files  # noqa: E402
from sts2_env.async_ready_collector import AsyncReadyCollector, AsyncReadyItem, resolve_action_mask, stack_observations  # noqa: E402
from sts2_env.checkpoint import load_online_checkpoint_metadata  # noqa: E402
from sts2_env.combat_env import CombatSandboxEnv  # noqa: E402
from sts2_env.observation_v3 import MAX_ACTIONS, WorldTokenObservationEncoder  # noqa: E402
from legacy.attention_ppo.omni_attention_policy import STS2OmniAttentionPolicy  # noqa: E402


def _timestamp() -> str:
    return time.strftime("%Y%m%d_%H%M%S")


def _normalize_text(value: Any) -> str:
    return str(value or "").strip()


def _normalize_key(value: Any) -> str:
    return _normalize_text(value).lower()


def _float(value: Any, default: float = 0.0) -> float:
    if value is None:
        return default
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _resolve_policy_device(device: str | None) -> str:
    candidate = str(device or "").strip().lower()
    if not candidate or candidate == "auto":
        return "cuda" if torch.cuda.is_available() else "cpu"
    return str(device).strip()


def _resolve_amp_dtype(name: str | None) -> torch.dtype:
    normalized = _normalize_key(name)
    if normalized in {"bf16", "bfloat16"}:
        return torch.bfloat16
    if normalized in {"fp16", "float16", "half"}:
        return torch.float16
    return torch.bfloat16


def _metric_summary(values: list[float]) -> dict[str, float]:
    if not values:
        return {}
    arr = np.asarray(values, dtype=np.float64)
    return {
        "count": float(arr.size),
        "mean": float(arr.mean()),
        "p50": float(np.percentile(arr, 50)),
        "p95": float(np.percentile(arr, 95)),
        "max": float(arr.max()),
    }


def _group_metric_summary(rows: list[dict[str, Any]], *, key: str, metric: str) -> list[dict[str, Any]]:
    grouped: dict[str, list[float]] = defaultdict(list)
    for row in rows:
        group_value = row.get(key) if key in row else "<missing>"
        grouped[str(group_value)].append(_float(row.get(metric)))
    output: list[dict[str, Any]] = []
    for group_key, values in grouped.items():
        summary = _metric_summary(values)
        if not summary:
            continue
        output.append({"key": group_key, **summary})
    output.sort(key=lambda entry: (-float(entry["p95"]), -float(entry["count"]), str(entry["key"])))
    return output


def _group_metric_summary_from_fn(
    rows: list[dict[str, Any]],
    *,
    name: str,
    metric: str,
    key_fn,
) -> dict[str, Any]:
    grouped: dict[str, list[float]] = defaultdict(list)
    for row in rows:
        grouped[str(key_fn(row))].append(_float(row.get(metric)))
    groups = []
    for group_key, values in grouped.items():
        summary = _metric_summary(values)
        if not summary:
            continue
        groups.append({"key": group_key, **summary})
    groups.sort(key=lambda entry: (-float(entry["p95"]), -float(entry["count"]), str(entry["key"])))
    return {"name": name, "metric": metric, "groups": groups}


def _compact_json(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, str):
        return value
    try:
        return json.dumps(value, ensure_ascii=False, sort_keys=True)
    except TypeError:
        return str(value)


def _dominant_surface(info: dict[str, Any] | None) -> str:
    if not isinstance(info, dict):
        return "unknown"
    phase = _normalize_key(info.get("phase"))
    if phase == "card_selection":
        return "card_selection"
    compact_actions = info.get("legal_actions_compact")
    surfaces = [
        _normalize_key(action.get("surface"))
        for action in compact_actions
        if isinstance(action, dict) and _normalize_key(action.get("surface"))
    ] if isinstance(compact_actions, list) else []
    if surfaces:
        return Counter(surfaces).most_common(1)[0][0]
    return phase or "unknown"


def _surface_family(info: dict[str, Any] | None) -> str:
    dominant = _dominant_surface(info)
    if dominant == "card_selection":
        return "card_selection"
    if dominant in {"reward", "card_reward", "deck_upgrade", "shop", "map", "event", "combat"}:
        return dominant
    phase = _normalize_key((info or {}).get("phase")) if isinstance(info, dict) else ""
    if phase == "card_selection":
        return "card_selection"
    return dominant or phase or "unknown"


def _find_card_position(entries: list[Any], source: dict[str, Any] | None) -> int:
    if not isinstance(entries, list) or not isinstance(source, dict):
        return 0
    source_id = _normalize_text(source.get("id"))
    source_title = _normalize_key(source.get("title"))
    for index, entry in enumerate(entries):
        if not isinstance(entry, dict):
            continue
        entry_id = _normalize_text(entry.get("id"))
        entry_title = _normalize_key(entry.get("title"))
        if source_id and entry_id and source_id == entry_id:
            return index + 1
        if source_title and entry_title and source_title == entry_title:
            return index + 1
    return 0


def _combat_pile_cards(raw_obs: dict[str, Any], pile_key: str, fallback_key: str) -> list[Any]:
    combat = raw_obs.get("combat") if isinstance(raw_obs, dict) else {}
    if not isinstance(combat, dict):
        return []
    pile = combat.get(pile_key)
    if pile_key == "hand":
        if isinstance(pile, list):
            return pile
        alt = combat.get(fallback_key)
        return alt if isinstance(alt, list) else []
    if isinstance(pile, dict):
        cards = pile.get("cards")
        if isinstance(cards, list):
            return cards
    alt = combat.get(fallback_key)
    return alt if isinstance(alt, list) else []


def _infer_card_selection_source_pile(action: dict[str, Any], raw_obs: dict[str, Any] | None) -> tuple[str, int]:
    if not isinstance(action, dict) or not isinstance(action.get("card"), dict):
        return "none", 0

    prompt = " ".join(
        [
            _normalize_text(action.get("selection_prompt")),
            _normalize_text(action.get("label")),
            _normalize_text(action.get("screen_type")),
        ]
    ).lower()
    card = action.get("card")
    obs_dict = raw_obs if isinstance(raw_obs, dict) else {}
    pile_order = [
        ("hand", _combat_pile_cards(obs_dict, "hand", "hand")),
        ("draw", _combat_pile_cards(obs_dict, "draw_pile", "draw_preview_cards")),
        ("discard", _combat_pile_cards(obs_dict, "discard_pile", "discard_cards")),
        ("exhaust", _combat_pile_cards(obs_dict, "exhaust_pile", "exhaust_cards")),
        ("play", _combat_pile_cards(obs_dict, "play_pile", "play_pile_cards")),
        ("deck", (obs_dict.get("player") or {}).get("deck_cards") if isinstance(obs_dict.get("player"), dict) else []),
    ]

    matches: list[tuple[str, int]] = []
    for pile_name, cards in pile_order:
        position = _find_card_position(cards if isinstance(cards, list) else [], card)
        if position > 0:
            matches.append((pile_name, position))
    if len(matches) == 1:
        return matches[0]

    keyword_map = {
        "discard": ("discard", matches),
        "弃牌": ("discard", matches),
        "draw pile": ("draw", matches),
        "抽牌": ("draw", matches),
        "exhaust": ("exhaust", matches),
        "消耗": ("exhaust", matches),
        "hand": ("hand", matches),
        "手牌": ("hand", matches),
        "play pile": ("play", matches),
        "played": ("play", matches),
    }
    for keyword, (pile_name, _existing) in keyword_map.items():
        if keyword in prompt:
            for matched_name, matched_pos in matches:
                if matched_name == pile_name:
                    return matched_name, matched_pos
            return pile_name, 0

    if matches:
        return matches[0]
    return "unknown", 0


def _pile_owner_and_zone(source_pile: str) -> tuple[int, int]:
    from sts2_env.observation_v3 import (
        OWNER_DECK,
        OWNER_DISCARD,
        OWNER_DRAW,
        OWNER_EXHAUST,
        OWNER_HAND,
        OWNER_PLAY,
        OWNER_REWARD,
        OWNER_UPGRADE,
        TOKEN_ZONE_TO_ID,
    )

    mapping = {
        "hand": (OWNER_HAND, TOKEN_ZONE_TO_ID["HAND"]),
        "draw": (OWNER_DRAW, TOKEN_ZONE_TO_ID["DRAW"]),
        "discard": (OWNER_DISCARD, TOKEN_ZONE_TO_ID["DISCARD"]),
        "exhaust": (OWNER_EXHAUST, TOKEN_ZONE_TO_ID["EXHAUST"]),
        "play": (OWNER_PLAY, TOKEN_ZONE_TO_ID["PLAY"]),
        "deck": (OWNER_DECK, TOKEN_ZONE_TO_ID["DECK"]),
        "reward": (OWNER_REWARD, TOKEN_ZONE_TO_ID["REWARD"]),
        "upgrade": (OWNER_UPGRADE, TOKEN_ZONE_TO_ID["UPGRADE"]),
    }
    return mapping.get(source_pile, (0, 0))


def _build_env_fns(
    *,
    args: argparse.Namespace,
    session_files: list[str | None],
    snapshot_pool,
) -> list:
    text_device = resolve_text_device(args.text_device, args.device)
    encounter_pool = parse_csv(args.encounter_pool)

    def _factory_for(env_index: int, session_file: str | None):
        def _factory():
            obs_encoder = WorldTokenObservationEncoder(use_text=True, text_device=text_device)
            return CombatSandboxEnv(
                session_file=session_file,
                character=args.character,
                encounter_id=args.encounter_id,
                encounter_pool=encounter_pool,
                snapshot_pool=snapshot_pool,
                reset_timeout_ms=args.reset_timeout_ms,
                step_timeout_ms=args.step_timeout_ms,
                obs_encoder=obs_encoder,
                include_debug_info=True,
            )

        return _factory

    return [_factory_for(index, session_files[index]) for index in range(args.n_envs)]


def _make_policy(
    *,
    device: str,
    policy_kwargs: dict[str, Any],
    text_device: str,
) -> STS2OmniAttentionPolicy:
    obs_space = WorldTokenObservationEncoder(use_text=True, text_device=text_device).obs_space
    policy = STS2OmniAttentionPolicy(
        observation_space=obs_space,
        action_space=spaces.Discrete(MAX_ACTIONS),
        lr_schedule=lambda _progress: 0.0,
        **policy_kwargs,
    )
    policy = policy.to(torch.device(device))
    policy.set_training_mode(False)
    return policy


def _load_policy_state_dict_best_effort(
    policy: STS2OmniAttentionPolicy,
    checkpoint_dir: str | Path,
    *,
    device: str,
) -> dict[str, Any]:
    checkpoint_path = Path(checkpoint_dir)
    source_state = load_file(str(checkpoint_path / "model.safetensors"), device=device)
    current_state = policy.state_dict()
    merged_state = {}
    exact_keys: list[str] = []
    partial_keys: list[str] = []
    skipped_keys: list[str] = []

    for key, current_value in current_state.items():
        source_value = source_state.get(key)
        if source_value is None:
            continue
        if tuple(source_value.shape) == tuple(current_value.shape):
            merged_state[key] = source_value.to(dtype=current_value.dtype)
            exact_keys.append(key)
            continue
        if source_value.ndim == current_value.ndim and source_value.ndim in {1, 2}:
            patched = current_value.clone()
            slices = tuple(slice(0, min(int(src_dim), int(dst_dim))) for src_dim, dst_dim in zip(source_value.shape, current_value.shape, strict=True))
            patched[slices] = source_value[slices].to(dtype=current_value.dtype)
            merged_state[key] = patched
            partial_keys.append(key)
            continue
        skipped_keys.append(key)

    missing, unexpected = policy.load_state_dict(merged_state, strict=False)
    return {
        "exact_loaded_keys": len(exact_keys),
        "partial_loaded_keys": len(partial_keys),
        "skipped_shape_keys": skipped_keys,
        "missing_keys": list(missing),
        "unexpected_keys": list(unexpected),
        "partial_key_examples": partial_keys[:12],
    }


def _extract_action_row(action_index: int, info: dict[str, Any]) -> tuple[dict[str, Any], dict[str, Any]]:
    actions_full = info.get("legal_actions") if isinstance(info, dict) else None
    actions_compact = info.get("legal_actions_compact") if isinstance(info, dict) else None
    full = actions_full[action_index] if isinstance(actions_full, list) and 0 <= action_index < len(actions_full) and isinstance(actions_full[action_index], dict) else {}
    compact = actions_compact[action_index] if isinstance(actions_compact, list) and 0 <= action_index < len(actions_compact) and isinstance(actions_compact[action_index], dict) else {}
    return full, compact


def _autocast_context(*, device: str, enabled: bool, dtype: torch.dtype):
    if not enabled:
        return nullcontext()
    if torch.device(device).type != "cuda":
        return nullcontext()
    return torch.autocast(device_type="cuda", dtype=dtype)


def _build_transition_row(
    *,
    index: int,
    env_id: int,
    prior: dict[str, Any],
    item: AsyncReadyItem,
) -> dict[str, Any]:
    info = item.transition_info if isinstance(item.transition_info, dict) else {}
    bridge_info = info.get("bridge_info") if isinstance(info.get("bridge_info"), dict) else {}
    step_timing = bridge_info.get("step_timing_ms") if isinstance(bridge_info.get("step_timing_ms"), dict) else {}
    step_counts = bridge_info.get("step_timing_counts") if isinstance(bridge_info.get("step_timing_counts"), dict) else {}
    python_timing = info.get("python_timing_ms") if isinstance(info.get("python_timing_ms"), dict) else {}

    selected_action = prior.get("selected_action") if isinstance(prior.get("selected_action"), dict) else {}
    selected_compact = prior.get("selected_action_compact") if isinstance(prior.get("selected_action_compact"), dict) else {}
    source_pile, source_order = _infer_card_selection_source_pile(selected_action, prior.get("pre_raw_obs"))
    source_owner_id, source_zone_id = _pile_owner_and_zone(source_pile)

    post_surface = _dominant_surface(info)
    post_surface_family = _surface_family(info)
    post_phase = _normalize_key(info.get("phase"))
    pre_kind = _normalize_key(selected_action.get("kind") or selected_compact.get("kind"))
    pre_surface = _normalize_key(selected_action.get("surface") or selected_compact.get("surface"))
    pre_selection_semantics = _normalize_key(selected_action.get("selection_semantics"))
    opened_card_selection = pre_kind != "card_selection" and post_phase == "card_selection"

    return {
        "transition_index": int(index),
        "env_id": int(env_id),
        "episode_id": _normalize_text(info.get("episode_id") or prior.get("episode_id")),
        "episode_start_before_action": bool(prior.get("episode_start")),
        "encounter_id": _normalize_text(info.get("encounter_id") or prior.get("encounter_id")),
        "snapshot_sample_id": _normalize_text(info.get("snapshot_sample_id") or prior.get("snapshot_sample_id")),
        "pre_phase": _normalize_key(prior.get("pre_phase")),
        "pre_surface_family": _normalize_key(prior.get("pre_surface_family")),
        "pre_action_id": _normalize_text(selected_action.get("action_id") or selected_compact.get("action_id")),
        "pre_action_kind": pre_kind,
        "pre_action_surface": pre_surface,
        "pre_action_title": _normalize_text(selected_action.get("label") or selected_compact.get("title") or selected_action.get("title")),
        "pre_action_semantic": _compact_json(selected_compact.get("semantic")),
        "pre_selection_semantics": pre_selection_semantics,
        "pre_selection_prompt": _normalize_text(selected_action.get("selection_prompt")),
        "pre_screen_type": _normalize_text(selected_action.get("screen_type")),
        "pre_source_pile_hint": source_pile,
        "pre_source_order_hint": int(source_order),
        "pre_source_owner_hint": int(source_owner_id),
        "pre_source_zone_hint": int(source_zone_id),
        "post_phase": post_phase,
        "post_surface": post_surface,
        "post_surface_family": post_surface_family,
        "post_legal_action_count": int(info.get("legal_action_count") or 0),
        "opened_card_selection": bool(opened_card_selection),
        "terminated": bool(item.terminated),
        "truncated": bool(item.truncated),
        "action_error": _normalize_text(bridge_info.get("action_error")),
        "truncation_reason": _normalize_text(bridge_info.get("truncation_reason")),
        "step_before_wait_ms": _float(step_timing.get("before_wait")),
        "step_action_execute_ms": _float(step_timing.get("action_execute")),
        "step_after_wait_ms": _float(step_timing.get("after_wait")),
        "step_auto_confirm_ms": _float(step_timing.get("auto_confirm")),
        "step_episode_adjustments_ms": _float(step_timing.get("episode_adjustments")),
        "step_total_ms": _float(step_timing.get("total")),
        "step_snapshot_calls": int(_float(step_counts.get("snapshot_calls"))),
        "step_wait_pump_calls": int(_float(step_counts.get("wait_pump_calls"))),
        "step_stable_iterations": int(_float(step_counts.get("stable_iterations"))),
        "python_obs_encode_ms": _float(python_timing.get("obs_encode")),
        "python_total_ms": _float(python_timing.get("total")),
        "policy_forward_batch_ms": _float(prior.get("policy_forward_batch_ms")),
        "policy_forward_per_env_ms": _float(prior.get("policy_forward_per_env_ms")),
        "dispatch_batch_size": int(prior.get("dispatch_batch_size") or 0),
    }


def _top_outliers(rows: list[dict[str, Any]], *, metric: str, top_n: int) -> list[dict[str, Any]]:
    keys = [
        "transition_index",
        "env_id",
        "episode_id",
        "episode_start_before_action",
        "encounter_id",
        "snapshot_sample_id",
        "pre_phase",
        "pre_surface_family",
        "pre_action_id",
        "pre_action_kind",
        "pre_action_surface",
        "pre_action_title",
        "pre_selection_semantics",
        "pre_selection_prompt",
        "pre_source_pile_hint",
        "post_phase",
        "post_surface",
        "post_surface_family",
        "opened_card_selection",
        "step_before_wait_ms",
        "step_action_execute_ms",
        "step_after_wait_ms",
        "step_auto_confirm_ms",
        "step_episode_adjustments_ms",
        "step_total_ms",
        "step_wait_pump_calls",
        "step_stable_iterations",
        "python_obs_encode_ms",
        "python_total_ms",
        "policy_forward_batch_ms",
        "policy_forward_per_env_ms",
        "dispatch_batch_size",
        "action_error",
        "truncation_reason",
    ]
    ranked = sorted(rows, key=lambda row: float(row.get(metric) or 0.0), reverse=True)
    return [{key: row.get(key) for key in keys} for row in ranked[:top_n]]


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def _write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False))
            handle.write("\n")


def main() -> None:
    parser = argparse.ArgumentParser(description="Diagnose long-tail step.after_wait stalls on async 4-env combat sandbox.")
    parser.add_argument("--n-envs", type=int, default=4)
    parser.add_argument("--transitions", type=int, default=256)
    parser.add_argument("--queue-timeout-s", type=float, default=120.0)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--device", type=str, default="auto")
    parser.add_argument("--text-device", type=str, default="auto")
    parser.add_argument("--session-file", type=str, default=None)
    parser.add_argument("--session-files", type=str, default=None)
    parser.add_argument("--character", type=str, default=None)
    parser.add_argument("--encounter-id", type=str, default=None)
    parser.add_argument("--encounter-pool", type=str, default=None)
    parser.add_argument("--snapshot-pool", type=str, default="/mnt/e/game/project/sts2_mcp/datasets/curated_combat_ironclad_mixed_provenance")
    parser.add_argument("--snapshot-curated-subset", type=str, default="bootstrap_human_plus_local_act1clear_roomwin_only_minus_combat_reset_failures")
    parser.add_argument(
        "--snapshot-sample-mode",
        type=str,
        default="tier_weighted_encounter_balanced",
        choices=("row_uniform", "encounter_balanced", "tier_weighted_encounter_balanced"),
    )
    parser.add_argument("--snapshot-encounter-tiers", type=str, default=None)
    parser.add_argument("--snapshot-tier-weights", type=str, default="weak:0.45,normal:0.35,elite:0.15,boss:0.05")
    parser.add_argument("--snapshot-max-rows", type=int, default=None)
    parser.add_argument("--reset-timeout-ms", type=int, default=60000)
    parser.add_argument("--step-timeout-ms", type=int, default=20000)
    parser.add_argument("--checkpoint-dir", type=str, default=None)
    parser.add_argument("--autocast", type=str, default="auto", choices=("auto", "on", "off"))
    parser.add_argument("--amp-dtype", type=str, default="auto")
    parser.add_argument("--d-model", type=int, default=256)
    parser.add_argument("--n-heads", type=int, default=8)
    parser.add_argument("--ffn-dim", type=int, default=1024)
    parser.add_argument("--world-layers", type=int, default=6)
    parser.add_argument("--local-layers", type=int, default=1)
    parser.add_argument("--decoder-layers", type=int, default=2)
    parser.add_argument("--candidate-set-layers", type=int, default=1)
    parser.add_argument("--dropout", type=float, default=0.0)
    parser.add_argument("--top-n", type=int, default=20)
    parser.add_argument("--output-dir", type=str, default="logs_attention/diagnostics")
    args = parser.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    device = _resolve_policy_device(args.device)
    text_device = resolve_text_device(args.text_device, device)
    session_files = resolve_training_session_files(
        n_envs=int(args.n_envs),
        session_file=args.session_file,
        session_files=parse_csv(args.session_files),
    )
    snapshot_pool = resolve_snapshot_pool(args)

    policy_kwargs = {
        "d_model": int(args.d_model),
        "n_heads": int(args.n_heads),
        "ffn_dim": int(args.ffn_dim),
        "world_layers": int(args.world_layers),
        "local_layers": int(args.local_layers),
        "decoder_layers": int(args.decoder_layers),
        "candidate_set_layers": int(args.candidate_set_layers),
        "dropout": float(args.dropout),
    }
    checkpoint_metadata: dict[str, Any] | None = None
    if args.checkpoint_dir:
        checkpoint_metadata = load_online_checkpoint_metadata(args.checkpoint_dir)
        metadata_kwargs = checkpoint_metadata.get("policy_kwargs")
        if isinstance(metadata_kwargs, dict) and metadata_kwargs:
            policy_kwargs = dict(metadata_kwargs)

    policy = _make_policy(device=device, policy_kwargs=policy_kwargs, text_device=text_device)
    checkpoint_load_report: dict[str, Any] | None = None
    if args.checkpoint_dir:
        checkpoint_load_report = _load_policy_state_dict_best_effort(policy, args.checkpoint_dir, device=device)
        print(
            f"[diag] checkpoint load exact={checkpoint_load_report['exact_loaded_keys']} partial={checkpoint_load_report['partial_loaded_keys']} "
            f"skipped={len(checkpoint_load_report['skipped_shape_keys'])}",
            flush=True,
        )

    amp_enabled = False
    amp_dtype_name = str(args.amp_dtype)
    if str(args.autocast).strip().lower() == "on":
        amp_enabled = True
    elif str(args.autocast).strip().lower() == "off":
        amp_enabled = False
    elif checkpoint_metadata is not None:
        amp_enabled = bool(checkpoint_metadata.get("amp_enabled"))
        amp_dtype_name = str(checkpoint_metadata.get("amp_dtype") or amp_dtype_name)
    amp_dtype = _resolve_amp_dtype(amp_dtype_name)

    env_fns = _build_env_fns(args=args, session_files=session_files, snapshot_pool=snapshot_pool)
    collector = AsyncReadyCollector(env_fns)
    print(f'[diag] starting collector with {int(args.n_envs)} envs on device={device}, text_device={text_device}, autocast={amp_enabled}', flush=True)
    collector.start()
    rows: list[dict[str, Any]] = []
    ready_pool: dict[int, AsyncReadyItem] = {}
    pending: dict[int, dict[str, Any]] = {}
    transitions = 0
    started = time.perf_counter()
    try:
        while transitions < int(args.transitions) or pending:
            need_item = transitions < int(args.transitions) and not ready_pool
            min_items = 1 if (need_item or pending) else 0
            ready_items = collector.drain_ready(min_items=min_items, timeout_s=float(args.queue_timeout_s))
            for item in ready_items:
                next_ready = item
                if item.transition_info is not None and item.env_id in pending:
                    prior = pending.pop(item.env_id)
                    rows.append(_build_transition_row(index=transitions, env_id=item.env_id, prior=prior, item=item))
                    transitions += 1
                    if transitions % 16 == 0 or transitions == int(args.transitions):
                        print(f'[diag] collected {transitions}/{int(args.transitions)} transitions', flush=True)
                    next_ready = AsyncReadyItem(
                        env_id=item.env_id,
                        obs=item.obs,
                        info=item.info,
                        episode_start=item.episode_start,
                    )
                elif item.transition_info is not None:
                    next_ready = AsyncReadyItem(
                        env_id=item.env_id,
                        obs=item.obs,
                        info=item.info,
                        episode_start=item.episode_start,
                    )
                ready_pool[item.env_id] = next_ready

            remaining_dispatch = max(int(args.transitions) - (transitions + len(pending)), 0)
            if remaining_dispatch <= 0:
                continue

            ready_env_ids = sorted(ready_pool.keys())[:remaining_dispatch]
            if not ready_env_ids:
                continue

            dispatch_items = [ready_pool.pop(env_id) for env_id in ready_env_ids]
            batched_obs = stack_observations([item.obs for item in dispatch_items])
            batched_masks = np.stack(
                [resolve_action_mask(item.obs, item.info) for item in dispatch_items],
                axis=0,
            ).astype(bool)

            policy_started = time.perf_counter()
            with torch.no_grad():
                obs_tensor = obs_as_tensor(batched_obs, device)
                with _autocast_context(device=device, enabled=amp_enabled, dtype=amp_dtype):
                    actions, values, log_probs = policy(obs_tensor, action_masks=batched_masks)
            policy_forward_ms = (time.perf_counter() - policy_started) * 1000.0

            action_array = actions.detach().cpu().numpy().reshape(-1)
            value_array = values.detach().float().cpu().numpy().reshape(-1)
            log_prob_array = log_probs.detach().float().cpu().numpy().reshape(-1)
            batch_size = len(dispatch_items)

            for offset, item in enumerate(dispatch_items):
                selected_action, selected_compact = _extract_action_row(int(action_array[offset]), item.info if isinstance(item.info, dict) else {})
                pending[item.env_id] = {
                    "episode_id": _normalize_text((item.info or {}).get("episode_id")),
                    "encounter_id": _normalize_text((item.info or {}).get("encounter_id")),
                    "snapshot_sample_id": _normalize_text((item.info or {}).get("snapshot_sample_id")),
                    "episode_start": bool(item.episode_start),
                    "obs": item.obs,
                    "action": int(action_array[offset]),
                    "value": float(value_array[offset]),
                    "log_prob": float(log_prob_array[offset]),
                    "action_mask": batched_masks[offset].astype(np.float32),
                    "pre_phase": _normalize_text((item.info or {}).get("phase")),
                    "pre_surface_family": _surface_family(item.info if isinstance(item.info, dict) else {}),
                    "policy_forward_batch_ms": float(policy_forward_ms),
                    "policy_forward_per_env_ms": float(policy_forward_ms / max(batch_size, 1)),
                    "dispatch_batch_size": int(batch_size),
                    "selected_action": selected_action,
                    "selected_action_compact": selected_compact,
                    "pre_raw_obs": (item.info or {}).get("raw_obs") if isinstance(item.info, dict) else None,
                }
            collector.dispatch_actions(ready_env_ids, action_array)

        collector.push_ready_back(list(ready_pool.values()))
    finally:
        collector.close()

    elapsed_s = time.perf_counter() - started
    output_root = (RL_AGENT_ROOT / args.output_dir / f"after_wait_tail_diag_{_timestamp()}").resolve()
    output_root.mkdir(parents=True, exist_ok=True)

    transition_family_summary = _group_metric_summary_from_fn(
        rows,
        name="transition_family",
        metric="step_after_wait_ms",
        key_fn=lambda row: f"{row.get('pre_action_kind') or 'unknown'}->{row.get('post_phase') or 'unknown'}",
    )
    source_pile_summary = _group_metric_summary_from_fn(
        rows,
        name="card_selection_source_pile",
        metric="step_after_wait_ms",
        key_fn=lambda row: (
            f"{row.get('pre_source_pile_hint') or 'unknown'}"
            if row.get("pre_action_kind") == "card_selection" or row.get("opened_card_selection")
            else "non_card_selection"
        ),
    )

    summary = {
        "manifest": {
            "started_at_local": time.strftime("%Y-%m-%d %H:%M:%S"),
            "elapsed_s": float(elapsed_s),
            "n_envs": int(args.n_envs),
            "transitions": int(len(rows)),
            "device": device,
            "text_device": text_device,
            "autocast_enabled": bool(amp_enabled),
            "amp_dtype": str(amp_dtype_name),
            "checkpoint_dir": str(args.checkpoint_dir) if args.checkpoint_dir else None,
            "checkpoint_load_report": checkpoint_load_report,
            "session_files": [str(path) if path is not None else None for path in session_files],
            "snapshot_pool": args.snapshot_pool,
            "snapshot_curated_subset": args.snapshot_curated_subset,
            "snapshot_sample_mode": args.snapshot_sample_mode,
            "snapshot_tier_weights": args.snapshot_tier_weights,
        },
        "overall": {
            "step.after_wait_ms": _metric_summary([_float(row.get("step_after_wait_ms")) for row in rows]),
            "step.total_ms": _metric_summary([_float(row.get("step_total_ms")) for row in rows]),
            "python.obs_encode_ms": _metric_summary([_float(row.get("python_obs_encode_ms")) for row in rows]),
            "python.total_ms": _metric_summary([_float(row.get("python_total_ms")) for row in rows]),
            "policy.forward_batch_ms": _metric_summary([_float(row.get("policy_forward_batch_ms")) for row in rows]),
            "policy.forward_per_env_ms": _metric_summary([_float(row.get("policy_forward_per_env_ms")) for row in rows]),
        },
        "grouped": {
            "encounter_id": _group_metric_summary(rows, key="encounter_id", metric="step_after_wait_ms"),
            "pre_action_kind": _group_metric_summary(rows, key="pre_action_kind", metric="step_after_wait_ms"),
            "post_phase": _group_metric_summary(rows, key="post_phase", metric="step_after_wait_ms"),
            "post_surface_family": _group_metric_summary(rows, key="post_surface_family", metric="step_after_wait_ms"),
            "episode_start_before_action": _group_metric_summary(rows, key="episode_start_before_action", metric="step_after_wait_ms"),
            "opened_card_selection": _group_metric_summary(rows, key="opened_card_selection", metric="step_after_wait_ms"),
            "transition_family": transition_family_summary["groups"],
            "card_selection_source_pile": source_pile_summary["groups"],
        },
        "outliers": _top_outliers(rows, metric="step_after_wait_ms", top_n=max(int(args.top_n), 1)),
    }

    _write_json(output_root / "summary.json", summary)
    _write_jsonl(output_root / "rows.jsonl", rows)

    print(json.dumps(summary, ensure_ascii=False, indent=2))
    print(f"\nWROTE {output_root}", flush=True)


if __name__ == "__main__":
    main()


