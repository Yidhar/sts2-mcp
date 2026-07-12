"""Generate language-specific contract version constants.

The generator intentionally has no third-party dependencies so it can run in
the repository bootstrap and CI environments.
"""

from __future__ import annotations

import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
MANIFEST = ROOT / "contracts" / "manifest.json"
GENERATED = ROOT / "contracts" / "generated"


def _write(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text.replace("\r\n", "\n"), encoding="utf-8", newline="\n")


def main() -> None:
    manifest = json.loads(MANIFEST.read_text(encoding="utf-8"))
    values = {
        "API_VERSION": manifest["api_version"],
        "SCHEMA_VERSION": manifest["schema_version"],
        "ACTION_SCHEMA_VERSION": manifest["action_schema_version"],
        "LEGAL_ACTION_ORDERING_VERSION": manifest["legal_action_ordering_version"],
        "OBSERVATION_SCHEMA_VERSION": manifest["observation_schema_version"],
        "REWARD_SCHEMA_VERSION": manifest["reward_schema_version"],
    }

    py_lines = ["# Generated. Do not edit.", ""]
    py_lines.extend(f'{name} = "{value}"' for name, value in values.items())
    py_lines.append("")
    _write(GENERATED / "python" / "versions.py", "\n".join(py_lines))

    ts_lines = ["// Generated. Do not edit.", ""]
    ts_lines.extend(f'export const {name} = "{value}" as const;' for name, value in values.items())
    ts_lines.append("")
    _write(GENERATED / "typescript" / "versions.ts", "\n".join(ts_lines))

    cs_lines = [
        "// Generated. Do not edit.",
        "namespace Sts2.Contracts.Generated;",
        "",
        "public static class ContractVersions",
        "{",
    ]
    cs_lines.extend(f'    public const string {name.title().replace("_", "")} = "{value}";' for name, value in values.items())
    cs_lines.extend(["}", ""])
    _write(GENERATED / "csharp" / "ContractVersions.g.cs", "\n".join(cs_lines))


if __name__ == "__main__":
    main()
