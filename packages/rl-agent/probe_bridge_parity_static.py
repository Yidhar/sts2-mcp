"""Bridge parity audit — Part 0 (STATIC): compare what the two bridge
source trees DECLARE they expose, without running either side.

Purpose: find fields that live bridge builds into /state but sim's
translator doesn't (or vice versa), purely from the source code. This
catches divergences that a running-time comparison would also show,
but costs 0 game processes.

Inputs:
  LIVE_SRC  = ../../mods/sts2-bridge/Scripts  (bridge mod C#)
  SIM_SRC   = <STS2_AI_ROOT>/STS2AI/ENV/Sim/HeadlessSim  (sim RPC C#)
  SIM_TRANSLATE = sts2_env/_sim_translate.py  (Python-side sim->bridge shape mapper)

Outputs:
  analysis/probes/bridge_parity_static_YYYYMMDD_HHMMSS/
    live_emit_keys.txt       all keys the live bridge declares in a response payload
    sim_emit_keys.txt        all keys sim RPCs build into their response
    sim_translate_reads.txt  all keys Python sim bridge adapter READS from sim state
    live_minus_sim.txt       keys in live response but never read by sim adapter
                             (== training sees these on live, zero on sim)
    sim_minus_live.txt       keys in sim response but never seen on live
    summary.json             counts
"""
from __future__ import annotations

import argparse
import json
import os
import re
from collections import Counter
from pathlib import Path

from sts2_rl.artifacts import artifact_root, resolve_artifact_path, resolve_external_input_path

REPO_ROOT = Path(__file__).parent
LIVE_BRIDGE_CS = (REPO_ROOT / "../../mods/sts2-bridge/Scripts").resolve()
STS2_AI_ROOT = resolve_external_input_path(
    os.environ.get("STS2_AI_ROOT"),
    default="dependencies/sts2-ai",
    root=artifact_root(),
)
SIM_PROGRAM_CS = STS2_AI_ROOT / "STS2AI" / "ENV" / "Sim" / "HeadlessSim"
SIM_TRANSLATE_PY = REPO_ROOT / "sts2_env" / "_sim_translate.py"


# Patterns that extract emitted key names from C# anonymous-object / dict literals.
# Match "key_name ="  (anonymous object) and ["key_name"] = (Dictionary<string, object>)
# Only lowercase-underscore keys; exclude local var assignments like int x = 0.
ANON_OBJ_PATTERN = re.compile(r'^\s*([a-z][a-z0-9_]*)\s*=\s*')
DICT_STRING_KEY_PATTERN = re.compile(r'\["([a-z][a-z0-9_]*)"\]\s*=')


def extract_emitted_keys_cs(root: Path) -> Counter:
    """Scan C# files for response payload key names:
    1) anonymous-object field assignments (live bridge pattern):  name = value
    2) string-indexed dict assignments (sim Program.cs pattern):  ["name"] = value
    3) typed DTO public properties (sim FullRunApiStateDtos.cs pattern):
       public T name { get; set; }  — these are the JSON-serialized keys
       when the DTO is returned as a response.
    4) JsonPropertyName attributes: [JsonPropertyName("name")]
    """
    counts: Counter = Counter()
    if not root.exists():
        return counts
    # Typed property: `public <type> <name> { get; ... }` — JSON serializer
    # emits these as response keys. Match lowercase + snake_case names.
    prop_pattern = re.compile(
        r'^\s*public\s+(?:readonly\s+)?(?:virtual\s+)?(?:override\s+)?'
        r'[A-Za-z_][A-Za-z0-9_<>?\[\]\.\s]*?\s+'
        r'([a-z][a-z0-9_]*)\s*\{\s*get'
    )
    # JsonPropertyName attribute: [JsonPropertyName("name")]
    json_prop_pattern = re.compile(r'\[JsonPropertyName\(\s*"([^"]+)"\s*\)\]')

    for cs in root.rglob("*.cs"):
        try:
            lines = cs.read_text(encoding="utf-8", errors="ignore").splitlines()
        except Exception:
            continue
        for line in lines:
            # Dict-string-key
            for m in DICT_STRING_KEY_PATTERN.finditer(line):
                counts[m.group(1)] += 1
            # JsonPropertyName attribute (live bridge's BridgeEnvResetRequest
            # uses this; sim DTOs don't, relying on property name itself)
            for m in json_prop_pattern.finditer(line):
                counts[m.group(1)] += 1
            # Typed DTO public property
            m = prop_pattern.match(line)
            if m:
                counts[m.group(1)] += 1
            # Anonymous-object field (indent-heuristic)
            trimmed = line.strip().rstrip(",")
            if "=" in trimmed and not trimmed.endswith(";"):
                m = re.match(r'^([a-z][a-z0-9_]*)\s*=\s*', trimmed)
                if m:
                    leading_spaces = len(line) - len(line.lstrip(" "))
                    if leading_spaces >= 8:
                        counts[m.group(1)] += 1
    return counts


def extract_sim_translate_reads(path: Path) -> Counter:
    """Scan _sim_translate.py for .get("<key>") calls — these are the sim-state keys
    Python actually reads when converting sim's RPC shape to the bridge-shaped obs.
    Anything sim emits but NOT read here is lost before reaching policy."""
    counts: Counter = Counter()
    if not path.exists():
        return counts
    text = path.read_text(encoding="utf-8", errors="ignore")
    # .get("...") calls
    for m in re.finditer(r'\.get\(\s*[\'"]([a-zA-Z_][a-zA-Z0-9_]*)[\'"]', text):
        counts[m.group(1)] += 1
    # state[...] or raw["..."] indexed access
    for m in re.finditer(r'\[[\'"]([a-zA-Z_][a-zA-Z0-9_]*)[\'"]\]', text):
        counts[m.group(1)] += 1
    return counts


def extract_live_endpoints(root: Path) -> list[tuple[str, str]]:
    """Find all live HTTP endpoints in BridgeServer.cs — returns (method, path) tuples."""
    results: list[tuple[str, str]] = []
    server = root / "BridgeServer.cs"
    if not server.exists():
        return results
    text = server.read_text(encoding="utf-8", errors="ignore")
    # path.Equals("/env/xxx"...) patterns
    for m in re.finditer(r'path\.Equals\(\s*"(/[a-zA-Z_/][^"]*)"', text):
        results.append(("?", m.group(1)))
    return results


def extract_sim_rpcs(root: Path) -> list[str]:
    """Find all RPC method names in HeadlessSim/Program.cs. Handles
    compound aliases like `"state" or "get_state" => ...`."""
    results: list[str] = []
    prog = root / "Program.cs"
    if not prog.exists():
        return results
    text = prog.read_text(encoding="utf-8", errors="ignore")
    # Match switch arms: any number of `"name"` separated by `or` ending with `=>`
    for m in re.finditer(
        r'^\s*((?:"[a-z_]+"\s*(?:or\s+)?)+)\s*=>\s*', text, re.MULTILINE
    ):
        chunk = m.group(1)
        for name_match in re.finditer(r'"([a-z_]+)"', chunk):
            results.append(name_match.group(1))
    return sorted(set(results))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--top-n", type=int, default=50,
                        help="show top-N most-frequent keys in the diff summary")
    args = parser.parse_args()

    outdir = resolve_artifact_path(args.output_dir)
    outdir.mkdir(parents=True, exist_ok=True)

    print(f"[static] live bridge src: {LIVE_BRIDGE_CS}")
    print(f"[static] sim src: {SIM_PROGRAM_CS}")
    print(f"[static] sim translate: {SIM_TRANSLATE_PY}")

    live_keys = extract_emitted_keys_cs(LIVE_BRIDGE_CS)
    sim_keys = extract_emitted_keys_cs(SIM_PROGRAM_CS)
    sim_reads = extract_sim_translate_reads(SIM_TRANSLATE_PY)
    live_endpoints = extract_live_endpoints(LIVE_BRIDGE_CS)
    sim_rpcs = extract_sim_rpcs(SIM_PROGRAM_CS)

    # Dump per-source lists
    (outdir / "live_emit_keys.txt").write_text(
        "\n".join(f"{k:<40s} {v}" for k, v in live_keys.most_common()),
        encoding="utf-8",
    )
    (outdir / "sim_emit_keys.txt").write_text(
        "\n".join(f"{k:<40s} {v}" for k, v in sim_keys.most_common()),
        encoding="utf-8",
    )
    (outdir / "sim_translate_reads.txt").write_text(
        "\n".join(f"{k:<40s} {v}" for k, v in sim_reads.most_common()),
        encoding="utf-8",
    )
    (outdir / "live_endpoints.txt").write_text(
        "\n".join(f"{method:>6s} {path}" for method, path in live_endpoints),
        encoding="utf-8",
    )
    (outdir / "sim_rpc_methods.txt").write_text(
        "\n".join(sim_rpcs), encoding="utf-8",
    )

    # Diff: live emit vs sim translate reads
    live_set = set(live_keys.keys())
    sim_set = set(sim_keys.keys())
    sim_read_set = set(sim_reads.keys())

    # Keys in live that sim translate never reads — sim-side observation gap
    # (only relevant for keys also emitted by sim; if sim doesn't emit it at
    # all, translate has nothing to read. Still flag both as separate findings.)
    live_not_in_sim_emit = sorted(live_set - sim_set)
    sim_not_in_live_emit = sorted(sim_set - live_set)
    sim_not_read_by_translate = sorted(sim_set - sim_read_set)

    (outdir / "diff_live_emit_minus_sim_emit.txt").write_text(
        "# Keys emitted by live bridge C# but never emitted by sim C#.\n"
        "# These fields appear in live /state responses but NEVER in sim RPC responses,\n"
        "# so sim-trained policies never get to see them.\n\n"
        + "\n".join(f"{k:<40s} live_count={live_keys[k]}" for k in live_not_in_sim_emit),
        encoding="utf-8",
    )
    (outdir / "diff_sim_emit_not_read_by_translate.txt").write_text(
        "# Keys emitted by sim C# but NEVER read by Python _sim_translate.py adapter.\n"
        "# These are lost at the Python layer — sim has the data, but we discard it\n"
        "# before the policy sees it. Usually signals a stale translator.\n\n"
        + "\n".join(f"{k:<40s} sim_count={sim_keys[k]}" for k in sim_not_read_by_translate),
        encoding="utf-8",
    )
    (outdir / "diff_sim_emit_minus_live_emit.txt").write_text(
        "# Keys emitted by sim but not by live. These are sim-only noise\n"
        "# (usually debug fields) the policy might see on sim but never on live.\n\n"
        + "\n".join(f"{k:<40s} sim_count={sim_keys[k]}" for k in sim_not_in_live_emit),
        encoding="utf-8",
    )

    # Endpoint coverage
    live_ep_set = {p for _, p in live_endpoints}
    # Map sim RPC names to approximate equivalent live paths where possible
    sim_covers_live = {
        "/state": "state",
        "/env/reset": "reset",
        "/env/combat_reset": "combat_reset",
        "/env/step": "step",
        "/env/combat_catalog": "combat_catalog",
    }
    endpoint_parity: list[str] = []
    for path, rpc_hint in sim_covers_live.items():
        live_has = path in live_ep_set
        sim_has = rpc_hint in sim_rpcs
        endpoint_parity.append(
            f"{path:<24s}  live_has={live_has}  sim_has={sim_has}  sim_rpc={rpc_hint}"
        )
    (outdir / "endpoint_parity.txt").write_text("\n".join(endpoint_parity), encoding="utf-8")

    summary = {
        "live_emit_unique_keys": len(live_keys),
        "sim_emit_unique_keys": len(sim_keys),
        "sim_translate_reads_unique_keys": len(sim_reads),
        "live_endpoints_count": len(live_endpoints),
        "sim_rpc_methods_count": len(sim_rpcs),
        "diff_counts": {
            "live_emit_not_in_sim_emit": len(live_not_in_sim_emit),
            "sim_emit_not_read_by_translate": len(sim_not_read_by_translate),
            "sim_emit_not_in_live_emit": len(sim_not_in_live_emit),
        },
        "top_20_live_only_keys": live_not_in_sim_emit[:20],
        "top_20_sim_not_translated_keys": sim_not_read_by_translate[:20],
        "top_20_sim_only_keys": sim_not_in_live_emit[:20],
    }
    (outdir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")

    print("\n=== SUMMARY ===")
    for k, v in summary.items():
        print(f"  {k}: {v}")
    print(f"\nartifacts: {outdir}")


if __name__ == "__main__":
    main()
