"""Validate component version declarations against release-manifest.json."""

from __future__ import annotations

import json
import re
import tomllib
import xml.etree.ElementTree as ET
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]


def require_equal(name: str, actual: str, expected: str, errors: list[str]) -> None:
    if actual != expected:
        errors.append(f"{name}: expected {expected}, found {actual}")


def main() -> int:
    release = json.loads((ROOT / "release-manifest.json").read_text(encoding="utf-8"))
    versions = release["components"]
    errors: list[str] = []
    expected_components = {"bridge", "mcp_server", "rl_trainer", "contracts", "game_data"}
    if set(versions) != expected_components:
        errors.append(
            "release_manifest: component keys must be exactly "
            + ", ".join(sorted(expected_components))
        )

    bridge_runtime = (ROOT / "mods/sts2-bridge/Scripts/BridgeRuntime.cs").read_text(encoding="utf-8")
    bridge_match = re.search(r'BridgeVersion\s*=\s*"([^"]+)"', bridge_runtime)
    require_equal("bridge", bridge_match.group(1) if bridge_match else "<missing>", versions["bridge"], errors)

    bridge_manifest = json.loads((ROOT / "mods/sts2-bridge/sts2-bridge.json").read_text(encoding="utf-8"))
    require_equal("bridge_manifest", str(bridge_manifest.get("version")), versions["bridge"], errors)
    if bridge_manifest.get("affects_gameplay") is not True:
        errors.append("bridge_manifest: affects_gameplay must be true for a control/training mod")

    bridge_project = ET.parse(ROOT / "mods/sts2-bridge/sts2-bridge.csproj").getroot()
    project_values = {
        element.tag: (element.text or "").strip()
        for group in bridge_project.findall("PropertyGroup")
        for element in group
        if element.tag in {"Version", "AssemblyVersion", "FileVersion", "InformationalVersion"}
    }
    require_equal("bridge_project.Version", project_values.get("Version", "<missing>"), versions["bridge"], errors)
    require_equal(
        "bridge_project.AssemblyVersion",
        project_values.get("AssemblyVersion", "<missing>"),
        f"{versions['bridge']}.0",
        errors,
    )
    require_equal(
        "bridge_project.FileVersion",
        project_values.get("FileVersion", "<missing>"),
        f"{versions['bridge']}.0",
        errors,
    )
    require_equal(
        "bridge_project.InformationalVersion",
        project_values.get("InformationalVersion", "<missing>"),
        versions["bridge"],
        errors,
    )

    mcp = json.loads((ROOT / "packages/mcp-server/package.json").read_text(encoding="utf-8"))
    require_equal("mcp_server", str(mcp.get("version")), versions["mcp_server"], errors)
    mcp_lock = json.loads((ROOT / "packages/mcp-server/package-lock.json").read_text(encoding="utf-8"))
    require_equal(
        "mcp_server_lock",
        str(mcp_lock.get("packages", {}).get("", {}).get("version")),
        versions["mcp_server"],
        errors,
    )

    with (ROOT / "packages/rl-agent/pyproject.toml").open("rb") as stream:
        rl_project = tomllib.load(stream)
    require_equal("rl_trainer", str(rl_project["project"].get("version")), versions["rl_trainer"], errors)
    rl_init = (ROOT / "packages/rl-agent/sts2_rl/__init__.py").read_text(encoding="utf-8")
    rl_init_match = re.search(r'^__version__\s*=\s*"([^"]+)"', rl_init, flags=re.MULTILINE)
    require_equal(
        "rl_trainer.__version__",
        rl_init_match.group(1) if rl_init_match else "<missing>",
        versions["rl_trainer"],
        errors,
    )

    contracts = json.loads((ROOT / "contracts/manifest.json").read_text(encoding="utf-8"))
    require_equal("contracts", str(contracts.get("api_version")), versions["contracts"], errors)
    openapi = json.loads((ROOT / "contracts/openapi/bridge-v2.openapi.json").read_text(encoding="utf-8"))
    require_equal("contracts.openapi", str(openapi.get("info", {}).get("version")), versions["contracts"], errors)

    game_data = json.loads((ROOT / "game-data/manifest.json").read_text(encoding="utf-8"))
    require_equal("game_data", str(game_data.get("schema_version")), versions["game_data"], errors)

    if errors:
        print("version validation failed:")
        print("\n".join(f"- {error}" for error in errors))
        return 1
    print("component versions match release-manifest.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
