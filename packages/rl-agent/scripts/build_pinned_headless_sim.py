"""Build Release HeadlessSim from the locked checkout and emit its identity sidecar."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import tempfile
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

PACKAGE_ROOT = Path(__file__).resolve().parents[1]
if str(PACKAGE_ROOT) not in sys.path:
    sys.path.insert(0, str(PACKAGE_ROOT))

from sts2_rl.artifacts import artifact_root  # noqa: E402
from sts2_rl.simulator_identity import (  # noqa: E402
    IDENTITY_SCHEMA_VERSION,
    REQUIRED_BUILD_CONFIGURATION,
    REQUIRED_TARGET_FRAMEWORK,
    load_sts2_ai_lock,
    sha256_file,
    simulator_identity_path,
    verify_headless_simulator,
)


def _output(*args: str, cwd: Path | None = None) -> str:
    try:
        return subprocess.check_output(args, cwd=cwd, text=True, stderr=subprocess.STDOUT).strip()
    except (OSError, subprocess.CalledProcessError) as exc:
        detail = getattr(exc, "output", None)
        suffix = f": {str(detail).strip()}" if detail else ""
        raise SystemExit(f"command failed: {' '.join(args)}{suffix}") from exc


def _default_source(lock: dict[str, Any]) -> Path:
    configured = os.environ.get("STS2_AI_ROOT", "").strip()
    if configured:
        path = Path(os.path.expandvars(configured)).expanduser()
        if not path.is_absolute():
            raise SystemExit(f"STS2_AI_ROOT must be absolute: {configured!r}")
        return path.resolve(strict=False)
    return (artifact_root() / str(lock["destination"])).resolve(strict=False)


def _verify_checkout(source: Path, lock: dict[str, Any]) -> None:
    if not (source / ".git").exists():
        raise SystemExit(f"pinned sts2-ai Git checkout is missing: {source}")
    actual_commit = _output("git", "rev-parse", "HEAD", cwd=source)
    if actual_commit != lock["commit"]:
        raise SystemExit(f"sts2-ai source commit is {actual_commit}; expected {lock['commit']}")
    actual_tree = _output("git", "rev-parse", "HEAD^{tree}", cwd=source)
    if actual_tree != lock["tree"]:
        raise SystemExit(f"sts2-ai source tree is {actual_tree}; expected {lock['tree']}")
    dirty = _output("git", "status", "--porcelain", "--untracked-files=all", cwd=source)
    if dirty:
        raise SystemExit("sts2-ai checkout is dirty; refusing to issue a build identity")
    symbolic = subprocess.run(
        ["git", "symbolic-ref", "-q", "HEAD"],
        cwd=source,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.DEVNULL,
        check=False,
    )
    if symbolic.returncode == 0:
        raise SystemExit(
            f"sts2-ai checkout must be detached at the lock commit, found {symbolic.stdout.strip()}"
        )
    submodules = subprocess.run(
        ["git", "submodule", "status", "--recursive"],
        cwd=source,
        text=True,
        capture_output=True,
        check=False,
    )
    if submodules.returncode != 0 or any(
        line.startswith(("-", "+", "U")) for line in submodules.stdout.splitlines()
    ):
        raise SystemExit("sts2-ai submodule state does not match the pinned checkout")


def _atomic_write_json(path: Path, payload: dict[str, Any]) -> None:
    encoded = (json.dumps(payload, indent=2, sort_keys=True) + "\n").encode("utf-8")
    fd, temporary_name = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    temporary = Path(temporary_name)
    try:
        with os.fdopen(fd, "wb") as handle:
            handle.write(encoded)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--source",
        help="absolute pinned sts2-ai checkout (default: STS2_AI_ROOT or the locked artifact destination)",
    )
    parser.add_argument(
        "--no-build",
        action="store_true",
        help="issue/replace identity for an existing Release output after source verification",
    )
    args = parser.parse_args(argv)

    lock = load_sts2_ai_lock()
    source = Path(args.source).expanduser().resolve(strict=False) if args.source else _default_source(lock)
    if not source.is_absolute() or source == Path(source.anchor):
        raise SystemExit(f"source must be an absolute non-root path: {source}")
    _verify_checkout(source, lock)
    project = source / str(lock["canonical_headless_project"])
    if not project.is_file():
        raise SystemExit(f"canonical HeadlessSim project is missing: {project}")

    sdk = _output("dotnet", "--version")
    build_command = (
        "dotnet",
        "build",
        str(project),
        "--configuration",
        REQUIRED_BUILD_CONFIGURATION,
        "--framework",
        REQUIRED_TARGET_FRAMEWORK,
        "--nologo",
    )
    if not args.no_build:
        subprocess.check_call(build_command, cwd=source)

    executable = (
        project.parent
        / "bin"
        / REQUIRED_BUILD_CONFIGURATION
        / REQUIRED_TARGET_FRAMEWORK
        / "HeadlessSim.exe"
    ).resolve(strict=False)
    if not executable.is_file():
        raise SystemExit(f"Release HeadlessSim output is missing after build: {executable}")
    identity_path = simulator_identity_path(executable)
    payload = {
        "schema_version": IDENTITY_SCHEMA_VERSION,
        "component": "HeadlessSim",
        "source": {
            "url": lock["url"],
            "commit": lock["commit"],
            "tree": lock["tree"],
            "project": lock["canonical_headless_project"],
        },
        "build": {
            "configuration": REQUIRED_BUILD_CONFIGURATION,
            "target_framework": REQUIRED_TARGET_FRAMEWORK,
            "dotnet_sdk": sdk,
            "built_at_utc": datetime.now(UTC).isoformat(),
            "command": list(build_command),
        },
        "binary": {
            "file_name": executable.name,
            "size_bytes": executable.stat().st_size,
            "sha256": sha256_file(executable),
        },
    }
    _atomic_write_json(identity_path, payload)
    verified = verify_headless_simulator(executable, identity_path=identity_path)
    print(json.dumps(verified.to_mapping(), indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
