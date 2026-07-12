"""Build a release provenance attestation outside the source tree."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import subprocess
from datetime import datetime, timezone
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
IDENTITY_FILES = (
    "release-manifest.json",
    ".nvmrc",
    ".python-version",
    "global.json",
    "contracts/manifest.json",
    "game-data/manifest.json",
    "mods/sts2-bridge/sts2-bridge.csproj",
    "mods/sts2-bridge/sts2-bridge.json",
    "packages/mcp-server/package.json",
    "packages/mcp-server/package-lock.json",
    "packages/rl-agent/pyproject.toml",
    "packages/rl-agent/requirements-bootstrap.lock",
    "packages/rl-agent/requirements.lock",
    "packages/rl-agent/requirements-dev.lock",
    "packages/rl-agent/requirements-text.lock",
    "packages/rl-agent/requirements-wsl-rocm.txt",
    "packages/rl-agent/scripts/bootstrap_wsl_rocm.sh",
    "third_party/sts2-ai.lock.json",
)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def output(*command: str) -> str:
    return subprocess.check_output(command, cwd=ROOT, text=True, stderr=subprocess.STDOUT).strip()


def optional_version(*command: str) -> str | None:
    try:
        return output(*command).splitlines()[0]
    except (OSError, subprocess.CalledProcessError, IndexError):
        return None


def logical_artifact(path: Path) -> str:
    try:
        return path.resolve().relative_to(ROOT.resolve()).as_posix()
    except ValueError:
        return path.name


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--artifact", type=Path, action="append", default=[])
    parser.add_argument("--require-clean", action="store_true")
    args = parser.parse_args()

    status = output("git", "status", "--porcelain", "--untracked-files=all")
    dirty = bool(status)
    if args.require_clean and dirty:
        print("refusing release provenance for a dirty source tree")
        return 1

    identity = []
    for relative in IDENTITY_FILES:
        path = ROOT / relative
        if path.is_file():
            identity.append({"path": relative, "bytes": path.stat().st_size, "sha256": sha256(path)})

    artifacts = []
    for path in args.artifact:
        resolved = path.expanduser().resolve()
        if not resolved.is_file():
            raise SystemExit(f"release artifact is not a file: {resolved}")
        artifacts.append(
            {
                "path": logical_artifact(resolved),
                "bytes": resolved.stat().st_size,
                "sha256": sha256(resolved),
            }
        )

    payload = {
        "schema_version": "1.0.0",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "source": {
            "commit": output("git", "rev-parse", "HEAD"),
            "tree": output("git", "rev-parse", "HEAD^{tree}"),
            "dirty": dirty,
        },
        "ci": {
            "provider": "github-actions" if os.environ.get("GITHUB_ACTIONS") == "true" else "local",
            "run_id": os.environ.get("GITHUB_RUN_ID"),
            "run_attempt": os.environ.get("GITHUB_RUN_ATTEMPT"),
        },
        "toolchain": {
            "os": platform.platform(),
            "python": platform.python_version(),
            "node": optional_version("node", "--version"),
            "npm": optional_version("npm", "--version"),
            "dotnet": optional_version("dotnet", "--version"),
        },
        "identity_files": identity,
        "artifacts": artifacts,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_bytes((json.dumps(payload, ensure_ascii=False, indent=2) + "\n").encode("utf-8"))
    print(f"wrote provenance for {len(artifacts)} artifacts: {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
