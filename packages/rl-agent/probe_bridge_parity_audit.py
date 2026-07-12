"""Bridge parity audit Part 1 — schema + structural diff of sim vs live
bridges at MATCHED game state.

Goal: find every field / shape / type divergence between what the live
bridge mod serves (via HTTP) and what the headless sim serves (via stdio
RPC). These divergences are where policy observations, rewards, and
action legality drift apart.

Output:
  <outdir>/raw_live/<endpoint>.json       raw response
  <outdir>/raw_sim/<endpoint>.json
  <outdir>/schema_diff.txt                per-endpoint structural diff
  <outdir>/value_diff.txt                 per-endpoint same-path value diff
  <outdir>/summary.json                   counts per kind of divergence

Scope (Part 1):
  GET  /env/spec                  read-only bridge metadata
  GET  /env/combat_catalog        static catalog of combat encounters
  GET  /state (both sides)        full game state — called 3x:
        1) post combat_reset SLIMES_WEAK from a fixed snapshot
        2) after one play_card
        3) after one end_turn

Does NOT audit (Part 2+):
  - /env/reset full-run path
  - terminal transitions (BUG B territory — needs Part 3 trajectory)
  - damage-math (Part 3)
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any

import requests

from sts2_rl.artifacts import resolve_artifact_path, resolve_external_input_path

sys.path.insert(0, str(Path(__file__).parent))
from sts2_env.headless_sim_bridge_client import HeadlessSimBridgeClient

# ----------------------------------------------------------------------
# Raw endpoint callers — bypass Python wrapper normalization so we see the
# bridge's actual wire format.
# ----------------------------------------------------------------------

def live_raw_get(base_url: str, token: str, path: str, timeout: float = 10.0) -> Any:
    url = base_url.rstrip("/") + "/" + path.lstrip("/")
    headers = {"Authorization": f"Bearer {token}"}
    r = requests.get(url, headers=headers, timeout=timeout)
    return {"_http_status": r.status_code, "_body": r.json() if r.content else None}


def live_raw_post(base_url: str, token: str, path: str, body: dict, timeout: float = 30.0) -> Any:
    url = base_url.rstrip("/") + "/" + path.lstrip("/")
    headers = {"Authorization": f"Bearer {token}", "Content-Type": "application/json"}
    r = requests.post(url, headers=headers, data=json.dumps(body), timeout=timeout)
    return {"_http_status": r.status_code, "_body": r.json() if r.content else None}


def sim_raw_rpc(sim: HeadlessSimBridgeClient, method: str, params: dict | None = None) -> Any:
    return sim._rpc(method, params or {})


# ----------------------------------------------------------------------
# Structural diff
# ----------------------------------------------------------------------

def walk_structure(obj: Any, path: str = "") -> list[tuple[str, str]]:
    """Produce (path, type_signature) tuples for every leaf in obj."""
    results: list[tuple[str, str]] = []
    if isinstance(obj, dict):
        for k in sorted(obj.keys()):
            results.extend(walk_structure(obj[k], f"{path}.{k}" if path else str(k)))
    elif isinstance(obj, list):
        if not obj:
            results.append((path, "list[empty]"))
        else:
            # Sample first element + track list length
            first = obj[0]
            results.append((path, f"list[{len(obj)}]"))
            results.extend(walk_structure(first, f"{path}[0]"))
    else:
        tname = type(obj).__name__
        results.append((path, tname))
    return results


def compare_schemas(live: Any, sim: Any) -> dict[str, Any]:
    """Diff two responses by structural walk + value diff at matching paths."""
    live_tuples = walk_structure(live)
    sim_tuples = walk_structure(sim)
    live_map = {p: t for p, t in live_tuples}
    sim_map = {p: t for p, t in sim_tuples}

    only_live = sorted(set(live_map) - set(sim_map))
    only_sim = sorted(set(sim_map) - set(live_map))
    common = sorted(set(live_map) & set(sim_map))
    type_mismatches = []
    for path in common:
        if live_map[path] != sim_map[path]:
            type_mismatches.append((path, live_map[path], sim_map[path]))

    # Value differences at matching scalar paths (sample up to 100)
    value_diffs = []
    for path in common[:10000]:
        if live_map[path] in {"dict", "list[empty]"} or live_map[path].startswith("list["):
            continue
        lv = walk_get(live, path)
        sv = walk_get(sim, path)
        if lv != sv:
            value_diffs.append((path, repr(lv)[:120], repr(sv)[:120]))

    return {
        "live_field_count": len(live_tuples),
        "sim_field_count": len(sim_tuples),
        "only_live": only_live,
        "only_sim": only_sim,
        "type_mismatches": type_mismatches,
        "value_diffs": value_diffs,
    }


def walk_get(obj: Any, path: str) -> Any:
    """Get value at dotted path with [n] list indexing."""
    if not path:
        return obj
    parts: list[tuple[str, int | None]] = []
    cur = ""
    i = 0
    while i < len(path):
        c = path[i]
        if c == ".":
            if cur:
                parts.append((cur, None))
                cur = ""
            i += 1
        elif c == "[":
            if cur:
                parts.append((cur, None))
                cur = ""
            j = path.find("]", i)
            idx = int(path[i + 1 : j])
            parts.append(("", idx))
            i = j + 1
        else:
            cur += c
            i += 1
    if cur:
        parts.append((cur, None))

    node = obj
    for key, idx in parts:
        if idx is not None:
            if isinstance(node, list) and 0 <= idx < len(node):
                node = node[idx]
            else:
                return None
        elif key:
            if isinstance(node, dict):
                node = node.get(key)
            else:
                return None
    return node


# ----------------------------------------------------------------------
# Per-endpoint audit drivers
# ----------------------------------------------------------------------

AUDIT_SNAPSHOT = {
    "character": "IRONCLAD",
    "encounter_id": "ENCOUNTER.SLIMES_WEAK",
    "current_hp": 60,
    "max_hp": 80,
    "max_energy": 3,
    "gold": 99,
    "deck": [
        "CARD.STRIKE_IRONCLAD", "CARD.STRIKE_IRONCLAD", "CARD.STRIKE_IRONCLAD",
        "CARD.STRIKE_IRONCLAD", "CARD.STRIKE_IRONCLAD",
        "CARD.DEFEND_IRONCLAD", "CARD.DEFEND_IRONCLAD", "CARD.DEFEND_IRONCLAD",
        "CARD.DEFEND_IRONCLAD",
        "CARD.BASH",
        "CARD.ASCENDERS_BANE",
    ],
    "relics": ["RELIC.BURNING_BLOOD"],
    "potions": [],
}


def run_audit(
    base_url: str,
    token: str,
    sim: HeadlessSimBridgeClient,
    outdir: Path,
) -> dict[str, Any]:
    outdir.mkdir(parents=True, exist_ok=True)
    (outdir / "raw_live").mkdir(exist_ok=True)
    (outdir / "raw_sim").mkdir(exist_ok=True)

    schema_diff_lines: list[str] = []
    value_diff_lines: list[str] = []
    summary: dict[str, Any] = {}

    def record(name: str, live_resp: Any, sim_resp: Any) -> None:
        (outdir / "raw_live" / f"{name}.json").write_text(
            json.dumps(live_resp, ensure_ascii=False, indent=2, default=str), encoding="utf-8")
        (outdir / "raw_sim" / f"{name}.json").write_text(
            json.dumps(sim_resp, ensure_ascii=False, indent=2, default=str), encoding="utf-8")
        diff = compare_schemas(live_resp, sim_resp)
        summary[name] = {
            "live_fields": diff["live_field_count"],
            "sim_fields": diff["sim_field_count"],
            "only_live": len(diff["only_live"]),
            "only_sim": len(diff["only_sim"]),
            "type_mismatches": len(diff["type_mismatches"]),
            "value_diffs": len(diff["value_diffs"]),
        }
        schema_diff_lines.append(f"\n=== {name} ===")
        schema_diff_lines.append(
            f"field counts: live={diff['live_field_count']} sim={diff['sim_field_count']}"
        )
        if diff["only_live"]:
            schema_diff_lines.append(f"  paths ONLY in live ({len(diff['only_live'])}):")
            for p in diff["only_live"][:50]:
                schema_diff_lines.append(f"    - {p}")
            if len(diff["only_live"]) > 50:
                schema_diff_lines.append(f"    ... +{len(diff['only_live']) - 50} more")
        if diff["only_sim"]:
            schema_diff_lines.append(f"  paths ONLY in sim ({len(diff['only_sim'])}):")
            for p in diff["only_sim"][:50]:
                schema_diff_lines.append(f"    - {p}")
            if len(diff["only_sim"]) > 50:
                schema_diff_lines.append(f"    ... +{len(diff['only_sim']) - 50} more")
        if diff["type_mismatches"]:
            schema_diff_lines.append(f"  type mismatches ({len(diff['type_mismatches'])}):")
            for p, lt, st in diff["type_mismatches"][:50]:
                schema_diff_lines.append(f"    {p}: live={lt} sim={st}")
        value_diff_lines.append(f"\n=== {name} ===")
        value_diff_lines.append(f"value diffs at matching paths ({len(diff['value_diffs'])}):")
        for p, lv, sv in diff["value_diffs"][:100]:
            value_diff_lines.append(f"  {p}:\n    live={lv}\n    sim ={sv}")
        if len(diff["value_diffs"]) > 100:
            value_diff_lines.append(f"  ... +{len(diff['value_diffs']) - 100} more")

    # --- Endpoint 1: GET /env/spec ---
    print("[audit] GET /env/spec ...")
    try:
        live = live_raw_get(base_url, token, "env/spec")
        sim = {"_note": "no direct equivalent on sim side; spec is implicit"}
        record("01_env_spec", live, sim)
    except Exception as e:
        print(f"  ERR: {e}")

    # --- Endpoint 2: game/combat catalog ---
    print("[audit] GET /env/combat_catalog + sim.combat_catalog ...")
    try:
        live = live_raw_get(base_url, token, "env/combat_catalog")
        sim = sim_raw_rpc(sim, "combat_catalog")
    except Exception as e:
        print(f"  err {e}")
        # Rebind: the variable 'sim' shadowed above. Use the local 'sim' bridge.
        # Fall through; will error out of endpoint.
    # ^ variable shadow hazard; restore explicit names below

    print("[audit] done Part 1")
    # Dump final outputs
    (outdir / "schema_diff.txt").write_text("\n".join(schema_diff_lines), encoding="utf-8")
    (outdir / "value_diff.txt").write_text("\n".join(value_diff_lines), encoding="utf-8")
    (outdir / "summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    return summary


def main() -> None:
    parser = argparse.ArgumentParser()
    default_session = os.environ.get("STS2_BRIDGE_SESSION_FILE")
    if default_session is None and os.environ.get("APPDATA"):
        default_session = str(Path(os.environ["APPDATA"]) / "SlayTheSpire2" / "bridge" / "session.json")
    parser.add_argument("--session-file", default=default_session,
                        help="Live session.json path (default: STS2_BRIDGE_SESSION_FILE or APPDATA discovery)")
    parser.add_argument("--sim-exe-path", default=None)
    parser.add_argument("--output-dir", required=True)
    args = parser.parse_args()

    if not args.session_file:
        parser.error("--session-file is required when APPDATA and STS2_BRIDGE_SESSION_FILE are unset")
    session_path = str(resolve_external_input_path(args.session_file))
    sim_exe_path = (
        str(resolve_external_input_path(args.sim_exe_path))
        if args.sim_exe_path
        else None
    )
    sess = json.loads(Path(session_path).read_text(encoding="utf-8"))
    base_url = sess["base_url"]
    token = sess["token"]
    print(f"[audit] live base_url={base_url}")
    print("[audit] starting sim...")
    sim = HeadlessSimBridgeClient(exe_path=sim_exe_path)

    outdir = resolve_artifact_path(args.output_dir)
    outdir.mkdir(parents=True, exist_ok=True)

    # Run the audit as a series of explicit steps to avoid variable shadow
    # hazards that the quick sketch above had. Rebuild per-endpoint here.
    summary: dict[str, Any] = {}
    schema_diff_lines: list[str] = []
    value_diff_lines: list[str] = []
    (outdir / "raw_live").mkdir(exist_ok=True)
    (outdir / "raw_sim").mkdir(exist_ok=True)

    def record(name: str, live_resp: Any, sim_resp: Any) -> None:
        (outdir / "raw_live" / f"{name}.json").write_text(
            json.dumps(live_resp, ensure_ascii=False, indent=2, default=str), encoding="utf-8")
        (outdir / "raw_sim" / f"{name}.json").write_text(
            json.dumps(sim_resp, ensure_ascii=False, indent=2, default=str), encoding="utf-8")
        diff = compare_schemas(live_resp, sim_resp)
        summary[name] = {
            "live_fields": diff["live_field_count"],
            "sim_fields": diff["sim_field_count"],
            "only_live": len(diff["only_live"]),
            "only_sim": len(diff["only_sim"]),
            "type_mismatches": len(diff["type_mismatches"]),
            "value_diffs": len(diff["value_diffs"]),
        }
        schema_diff_lines.append(f"\n=== {name} ===")
        schema_diff_lines.append(
            f"field counts: live={diff['live_field_count']} sim={diff['sim_field_count']}"
        )
        if diff["only_live"]:
            schema_diff_lines.append(f"  paths ONLY in live ({len(diff['only_live'])}):")
            for p in diff["only_live"][:50]:
                schema_diff_lines.append(f"    - {p}")
            if len(diff["only_live"]) > 50:
                schema_diff_lines.append(f"    ... +{len(diff['only_live']) - 50} more")
        if diff["only_sim"]:
            schema_diff_lines.append(f"  paths ONLY in sim ({len(diff['only_sim'])}):")
            for p in diff["only_sim"][:50]:
                schema_diff_lines.append(f"    - {p}")
            if len(diff["only_sim"]) > 50:
                schema_diff_lines.append(f"    ... +{len(diff['only_sim']) - 50} more")
        if diff["type_mismatches"]:
            schema_diff_lines.append(f"  type mismatches ({len(diff['type_mismatches'])}):")
            for p, lt, st in diff["type_mismatches"][:50]:
                schema_diff_lines.append(f"    {p}: live={lt} sim={st}")
        value_diff_lines.append(f"\n=== {name} ===")
        value_diff_lines.append(f"value diffs at matching paths ({len(diff['value_diffs'])}):")
        for p, lv, sv in diff["value_diffs"][:100]:
            value_diff_lines.append(f"  {p}:\n    live={lv}\n    sim ={sv}")
        if len(diff["value_diffs"]) > 100:
            value_diff_lines.append(f"  ... +{len(diff['value_diffs']) - 100} more")

    # 1. /env/spec (live only)
    print("[audit] 01_env_spec ...")
    try:
        live_spec = live_raw_get(base_url, token, "env/spec")
        record("01_env_spec", live_spec, {"_note": "sim has no direct spec endpoint"})
    except Exception as e:
        print(f"  ERR: {e}")
        summary["01_env_spec"] = {"error": str(e)[:200]}

    # 2. combat_catalog
    print("[audit] 02_combat_catalog ...")
    try:
        live_cc = live_raw_get(base_url, token, "env/combat_catalog")
        sim_cc = sim_raw_rpc(sim, "combat_catalog")
        record("02_combat_catalog", live_cc, sim_cc)
    except Exception as e:
        print(f"  ERR: {e}")
        summary["02_combat_catalog"] = {"error": str(e)[:200]}

    # 3. combat_reset + /state immediately after
    print("[audit] 03_state_after_combat_reset ...")
    try:
        # Live: POST /env/combat_reset then GET /state
        live_reset = live_raw_post(base_url, token, "env/combat_reset", AUDIT_SNAPSHOT)
        live_state = live_raw_get(base_url, token, "state")
        # Sim: bridge client packs correctly, then RPC "state"
        sim.combat_reset(
            character=AUDIT_SNAPSHOT["character"],
            encounter_id=AUDIT_SNAPSHOT["encounter_id"],
            current_hp=AUDIT_SNAPSHOT["current_hp"],
            max_hp=AUDIT_SNAPSHOT["max_hp"],
            max_energy=AUDIT_SNAPSHOT["max_energy"],
            gold=AUDIT_SNAPSHOT["gold"],
            deck=AUDIT_SNAPSHOT["deck"],
            relics=AUDIT_SNAPSHOT["relics"],
            potions=AUDIT_SNAPSHOT["potions"],
        )
        sim_state = sim_raw_rpc(sim, "state")
        record("03a_combat_reset_response", live_reset, {"_note": "sim combat_reset response is unwrapped in Python wrapper; see 03b for state"})
        record("03b_state_post_reset", live_state, sim_state)
    except Exception as e:
        print(f"  ERR: {e}")
        summary["03_state_post_reset"] = {"error": str(e)[:200]}

    # 4. legal_actions parity (live exposed via /state.available_actions; sim via RPC "legal_actions")
    print("[audit] 04_legal_actions ...")
    try:
        sim_legal = sim_raw_rpc(sim, "legal_actions")
        # live doesn't have a dedicated endpoint — extract from /state
        live_state_body = None
        try:
            live_state_fresh = live_raw_get(base_url, token, "state")
            live_state_body = live_state_fresh.get("_body") if isinstance(live_state_fresh, dict) else None
        except Exception:
            pass
        if isinstance(live_state_body, dict):
            live_legal_list = live_state_body.get("available_actions") or live_state_body.get("legal_actions")
            live_legal = {"_body": {"legal_actions": live_legal_list}}
        else:
            live_legal = {"_body": {"legal_actions": [], "_note": "live unreachable"}}
        record("04_legal_actions", live_legal, sim_legal)
    except Exception as e:
        print(f"  ERR: {e}")
        summary["04_legal_actions"] = {"error": str(e)[:200]}

    (outdir / "schema_diff.txt").write_text("\n".join(schema_diff_lines), encoding="utf-8")
    (outdir / "value_diff.txt").write_text("\n".join(value_diff_lines), encoding="utf-8")
    (outdir / "summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")

    print("\n=== SUMMARY ===")
    for endpoint, stats in summary.items():
        print(f"  {endpoint}: {stats}")
    print(f"\nartifacts: {outdir}")

    sim.close()


if __name__ == "__main__":
    main()
