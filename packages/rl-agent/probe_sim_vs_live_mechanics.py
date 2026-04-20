"""Pure mechanical divergence probe — no policy, fixed action sequence.

Drives BOTH sim and live through the SAME combat_reset (identical params)
and then executes a fixed action script (all end_turn). Logs every
observable field at each step so sim-vs-live divergence in card
playability / enemy AI / damage calc / relic application becomes
visible without policy variance confounding.

Output:
  <outdir>/live_full.jsonl  — one row per RPC (reset + step × N)
  <outdir>/sim_full.jsonl   — same
  <outdir>/diff.txt         — step-by-step summary of divergences

Usage:
  python probe_sim_vs_live_mechanics.py \\
    --encounter ENCOUNTER.LAGAVULIN_MATRIARCH_BOSS \\
    --snapshot-pool E:/game/.../curated_combat_ironclad_mixed_provenance \\
    --curated-subset bootstrap_human_plus_local_all_roomwin_only_minus_combat_reset_failures \\
    --snapshot-seed 0 \\
    --session-file C:/Users/.../bridge/session.json \\
    --n-steps 15 \\
    --output-dir analysis/probes/mechanics_YYYYMMDD
"""
from __future__ import annotations

import argparse
import copy
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np

from combat_snapshot_dataset import CombatSnapshotPool, snapshot_row_to_reset_kwargs
from sts2_env.bridge_client import BridgeClient
from sts2_env.headless_sim_bridge_client import HeadlessSimBridgeClient


def _obs_enemies(obs: dict[str, Any]) -> list[dict[str, Any]]:
    combat = (obs or {}).get("combat") or {}
    raw = combat.get("enemies")
    if not isinstance(raw, list):
        return []
    out = []
    for e in raw:
        if not isinstance(e, dict):
            continue
        intent = e.get("intent") or {}
        out.append({
            "id": e.get("id") or e.get("name") or e.get("monster_id"),
            "hp": e.get("hp") or e.get("current_hp"),
            "max_hp": e.get("max_hp"),
            "block": e.get("block"),
            "intent_type": intent.get("type"),
            "intent_total_damage": intent.get("total_damage"),
            "intent_damage_per_hit": intent.get("damage_per_hit"),
            "intent_repeats": intent.get("repeats"),
            "powers": [
                {"title": p.get("title") or p.get("id"), "amount": p.get("amount"), "duration": p.get("duration")}
                for p in (e.get("powers") or []) if isinstance(p, dict)
            ],
        })
    return out


def _obs_player(obs: dict[str, Any]) -> dict[str, Any]:
    p = (obs or {}).get("player") or {}
    c = (obs or {}).get("combat") or {}
    return {
        "hp": p.get("hp"),
        "max_hp": p.get("max_hp"),
        "block": c.get("block"),
        "energy": c.get("energy"),
        "powers": [
            {"title": po.get("title") or po.get("id"), "amount": po.get("amount"), "duration": po.get("duration")}
            for po in (p.get("powers") or []) if isinstance(po, dict)
        ],
        "relics": [r.get("id") if isinstance(r, dict) else r for r in (p.get("relics") or [])],
    }


def _obs_hand(obs: dict[str, Any]) -> list[dict[str, Any]]:
    combat = (obs or {}).get("combat") or {}
    hand = combat.get("hand") or []
    out = []
    for c in hand:
        if not isinstance(c, dict):
            continue
        out.append({
            "id": c.get("id") or c.get("card_id"),
            "cost": c.get("cost") or c.get("energy_cost"),
            "upgrade_level": c.get("upgrade_level"),
            "playable": c.get("playable"),
            "type": c.get("type") or c.get("card_type"),
        })
    return out


def _pile_len(val: Any) -> int:
    if isinstance(val, list):
        return len(val)
    if isinstance(val, int):
        return val
    if isinstance(val, dict):
        # some shapes: {cards: [...]} or {count: N}
        if isinstance(val.get("cards"), list):
            return len(val["cards"])
        if isinstance(val.get("count"), int):
            return val["count"]
    return 0


def _obs_piles(obs: dict[str, Any]) -> dict[str, int]:
    combat = (obs or {}).get("combat") or {}
    return {
        "draw": _pile_len(combat.get("draw_pile") or combat.get("draw")),
        "discard": _pile_len(combat.get("discard_pile") or combat.get("discard")),
        "exhaust": _pile_len(combat.get("exhaust_pile") or combat.get("exhaust")),
        "hand": _pile_len(combat.get("hand")),
    }


def _obs_legal(result: dict[str, Any]) -> list[dict[str, Any]]:
    legal = result.get("legal_actions") or []
    out = []
    for i, a in enumerate(legal[:20]):
        if not isinstance(a, dict):
            continue
        out.append({
            "idx": i,
            "action_id": a.get("action_id"),
            "kind": a.get("kind"),
            "card_id": (a.get("card") or {}).get("id") if isinstance(a.get("card"), dict) else a.get("card_id"),
            "label": a.get("label") or a.get("display_name"),
            "target_id": a.get("target_id"),
        })
    return out


def snapshot_step(result: dict[str, Any], tag: str, step: int, action_taken: str | None) -> dict[str, Any]:
    obs = result.get("obs") or result.get("observation") or {}
    return {
        "tag": tag,
        "step": step,
        "action_taken": action_taken,
        "reward": result.get("reward"),
        "done": result.get("done"),
        "truncated": result.get("truncated"),
        "info": {k: v for k, v in (result.get("info") or {}).items() if k not in ("aux_targets", "python_timing_ms", "raw_obs", "bridge_info")},
        "phase": obs.get("phase"),
        "screen": obs.get("screen"),
        "round": (obs.get("combat") or {}).get("round"),
        "player": _obs_player(obs),
        "enemies": _obs_enemies(obs),
        "hand": _obs_hand(obs),
        "piles": _obs_piles(obs),
        "legal_actions": _obs_legal(result),
    }


def find_end_turn_idx(result: dict[str, Any]) -> int | None:
    for i, a in enumerate(result.get("legal_actions") or []):
        if not isinstance(a, dict):
            continue
        kind = a.get("kind")
        if kind == "end_turn" or kind == "combat":
            # "combat" kind with action_id containing "end_turn" on live
            if a.get("action_id") and "end_turn" in str(a.get("action_id")).lower():
                return i
            if kind == "end_turn":
                return i
    return None


def run_bridge(bridge, bridge_name: str, snapshot: dict, n_steps: int, out_path: Path) -> list[dict[str, Any]]:
    kwargs = snapshot_row_to_reset_kwargs(snapshot, include_potions=True)
    # Drive combat_reset directly, mirroring CombatSandboxEnv params.
    reset_res = bridge.combat_reset(
        character=kwargs.get("character"),
        encounter_id=kwargs.get("encounter_id"),
        seed=snapshot.get("seed"),
        current_hp=kwargs.get("current_hp"),
        max_hp=kwargs.get("max_hp"),
        max_energy=kwargs.get("max_energy"),
        deck=kwargs.get("deck"),
        deck_entries=kwargs.get("deck_entries"),
        relics=kwargs.get("relics"),
        potions=kwargs.get("potions"),
        gold=kwargs.get("gold"),
    )
    rows: list[dict[str, Any]] = []
    rows.append(snapshot_step(reset_res, bridge_name, 0, None))
    episode_id = reset_res.get("episode_id")

    for step in range(1, n_steps + 1):
        last = rows[-1]
        if last.get("done"):
            print(f"[{bridge_name}] terminal at step {step-1}, stopping")
            break
        legal = reset_res.get("legal_actions") if step == 1 else prev_legal
        et_idx = find_end_turn_idx({"legal_actions": legal})
        if et_idx is None:
            # fall back to first action
            et_idx = 0
            action_label = "FALLBACK_IDX_0"
        else:
            action_label = "end_turn"
        step_res = bridge.step(episode_id=episode_id, action_index=et_idx)
        rows.append(snapshot_step(step_res, bridge_name, step, action_label))
        prev_legal = step_res.get("legal_actions") or []

    with out_path.open("w", encoding="utf-8") as fh:
        for r in rows:
            fh.write(json.dumps(r, ensure_ascii=False, default=str) + "\n")
    return rows


def compute_diff(live_rows: list[dict], sim_rows: list[dict], out_path: Path) -> None:
    lines: list[str] = []
    n = min(len(live_rows), len(sim_rows))
    for i in range(n):
        L = live_rows[i]
        S = sim_rows[i]
        header = f"=== step {i} | action: live={L.get('action_taken')!r} sim={S.get('action_taken')!r} ==="
        lines.append(header)
        # Player summary
        lp, sp = L["player"], S["player"]
        lines.append(f"  player.hp:    live={lp['hp']}/{lp['max_hp']}  sim={sp['hp']}/{sp['max_hp']}  Δ={(sp['hp'] or 0) - (lp['hp'] or 0)}")
        lines.append(f"  player.block: live={lp['block']}  sim={sp['block']}")
        lines.append(f"  player.energy:live={lp['energy']}  sim={sp['energy']}")
        if lp["powers"] != sp["powers"]:
            lines.append(f"  player.powers DIFFER:")
            lines.append(f"    live={lp['powers']}")
            lines.append(f"    sim ={sp['powers']}")
        if sorted(lp["relics"] or []) != sorted(sp["relics"] or []):
            lines.append(f"  player.relics DIFFER:")
            lines.append(f"    live={sorted(lp['relics'] or [])}")
            lines.append(f"    sim ={sorted(sp['relics'] or [])}")
        # Enemies summary
        lines.append(f"  enemies (live={len(L['enemies'])} sim={len(S['enemies'])})")
        for j in range(max(len(L["enemies"]), len(S["enemies"]))):
            le = L["enemies"][j] if j < len(L["enemies"]) else None
            se = S["enemies"][j] if j < len(S["enemies"]) else None
            if le and se:
                hp_delta = (se["hp"] or 0) - (le["hp"] or 0)
                lines.append(
                    f"    enemy[{j}] id={le.get('id')}/{se.get('id')}  "
                    f"hp: live={le['hp']}/{le['max_hp']} sim={se['hp']}/{se['max_hp']} Δ={hp_delta}  "
                    f"block: live={le['block']} sim={se['block']}  "
                    f"intent: live={le['intent_type']}/{le['intent_total_damage']} "
                    f"sim={se['intent_type']}/{se['intent_total_damage']}"
                )
                if le["powers"] != se["powers"]:
                    lines.append(f"      powers DIFFER:")
                    lines.append(f"        live={le['powers']}")
                    lines.append(f"        sim ={se['powers']}")
            elif le:
                lines.append(f"    enemy[{j}] ONLY LIVE: {le}")
            elif se:
                lines.append(f"    enemy[{j}] ONLY SIM: {se}")
        # Hand / piles
        if L["hand"] != S["hand"]:
            lines.append(f"  hand DIFFER:")
            lines.append(f"    live ({len(L['hand'])}): {[(c.get('id'), c.get('playable'), c.get('type')) for c in L['hand']]}")
            lines.append(f"    sim  ({len(S['hand'])}): {[(c.get('id'), c.get('playable'), c.get('type')) for c in S['hand']]}")
        if L["piles"] != S["piles"]:
            lines.append(f"  piles DIFFER:  live={L['piles']}  sim={S['piles']}")
        # Legal actions
        live_kinds = [a.get("kind") for a in L["legal_actions"]]
        sim_kinds = [a.get("kind") for a in S["legal_actions"]]
        if live_kinds != sim_kinds:
            lines.append(f"  legal_kinds DIFFER:  live={live_kinds}  sim={sim_kinds}")
        # Reward
        lines.append(f"  reward:       live={L.get('reward')}  sim={S.get('reward')}")
        lines.append(f"  done/trunc:   live={L.get('done')}/{L.get('truncated')}  sim={S.get('done')}/{S.get('truncated')}")
        lines.append("")
    out_path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--snapshot-pool", required=True)
    parser.add_argument("--curated-subset", default=None)
    parser.add_argument("--encounter", required=True)
    parser.add_argument("--snapshot-seed", type=int, default=0)
    parser.add_argument("--session-file", default=None)
    parser.add_argument("--sim-exe-path", default=None)
    parser.add_argument("--n-steps", type=int, default=15)
    parser.add_argument("--output-dir", required=True)
    args = parser.parse_args()

    pool = CombatSnapshotPool.from_path(
        args.snapshot_pool,
        curated_subset=args.curated_subset,
        encounter_ids=[args.encounter],
        sample_mode="encounter_balanced",
    )
    snapshot = pool.sample(np.random.default_rng(args.snapshot_seed))
    print(f"[probe] snapshot: build={snapshot.get('build_id')} sample={snapshot.get('sample_id')} "
          f"max_hp={snapshot.get('snapshot_max_hp')} deck_len={len(snapshot.get('deck_card_ids') or [])} "
          f"relics={len(snapshot.get('relic_ids_before') or [])}")

    outdir = Path(args.output_dir)
    outdir.mkdir(parents=True, exist_ok=True)

    print("[probe] driving LIVE bridge...")
    live_bridge = BridgeClient(session_path=args.session_file)
    live_rows = run_bridge(live_bridge, "live", snapshot, args.n_steps, outdir / "live_full.jsonl")
    print(f"[probe] live rows: {len(live_rows)}")

    print("[probe] driving SIM bridge...")
    sim_bridge = HeadlessSimBridgeClient(exe_path=args.sim_exe_path)
    sim_rows = run_bridge(sim_bridge, "sim", snapshot, args.n_steps, outdir / "sim_full.jsonl")
    sim_bridge.close()
    print(f"[probe] sim rows: {len(sim_rows)}")

    compute_diff(live_rows, sim_rows, outdir / "diff.txt")
    print(f"[probe] diff written to {outdir}/diff.txt")
    print(f"[probe] raw traces: {outdir}/live_full.jsonl, {outdir}/sim_full.jsonl")


if __name__ == "__main__":
    main()
