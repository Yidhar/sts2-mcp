"""Restore the pinned sts2-ai checkout without vendoring it in the main repo."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
LOCK_PATH = ROOT / "third_party" / "sts2-ai.lock.json"
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from tools.artifacts.path_policy import artifact_root as canonical_artifact_root  # noqa: E402
from tools.artifacts.path_policy import resolve_artifact_path  # noqa: E402


def artifact_root() -> Path:
    try:
        return canonical_artifact_root(source_root=ROOT)
    except ValueError as exc:
        raise SystemExit(str(exc)) from exc


def resolve_destination(value: str | None, lock: dict[str, object], *, verify_only: bool) -> Path:
    selected = value if value else str(lock["destination"])
    if verify_only and value:
        destination = Path(value).expanduser()
        if not destination.is_absolute():
            raise SystemExit(
                "--verify-only --destination must be absolute; use Resolve-Path when auditing "
                "the migration-only checkout"
            )
        destination = destination.resolve(strict=False)
        if destination == Path(destination.anchor).resolve(strict=False):
            raise SystemExit(f"third-party destination must not be a filesystem root: {destination}")
        return destination
    try:
        return resolve_artifact_path(selected, root=artifact_root(), source_root=ROOT)
    except ValueError as exc:
        raise SystemExit(str(exc)) from exc


def run(*args: str, cwd: Path = ROOT) -> str:
    return subprocess.check_output(args, cwd=cwd, text=True).strip()


def verify_checkout(destination: Path, lock: dict[str, object], *, require_detached: bool = True) -> None:
    expected = str(lock["commit"])
    if not destination.exists() or not (destination / ".git").exists():
        raise SystemExit(f"sts2-ai checkout is missing or is not Git metadata: {destination}")
    actual = run("git", "rev-parse", "HEAD", cwd=destination)
    if actual != expected:
        raise SystemExit(f"existing checkout is {actual}; expected {expected}")
    expected_tree = str(lock.get("tree") or "")
    actual_tree = run("git", "rev-parse", "HEAD^{tree}", cwd=destination)
    if expected_tree and actual_tree != expected_tree:
        raise SystemExit(f"checkout tree is {actual_tree}; expected {expected_tree}")
    dirty = run("git", "status", "--porcelain", "--untracked-files=all", cwd=destination)
    if dirty:
        raise SystemExit("sts2-ai checkout is dirty; move or discard local changes explicitly")
    symbolic = subprocess.run(
        ["git", "symbolic-ref", "-q", "HEAD"],
        cwd=destination,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.DEVNULL,
        check=False,
    )
    if require_detached and symbolic.returncode == 0:
        raise SystemExit(f"sts2-ai must be detached at the lock commit, found {symbolic.stdout.strip()}")
    canonical = destination / str(lock["canonical_headless_project"])
    if not canonical.is_file():
        raise SystemExit(f"canonical HeadlessSim project is missing: {canonical}")
    submodules = subprocess.run(
        ["git", "submodule", "status", "--recursive"],
        cwd=destination,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=False,
    )
    if submodules.returncode != 0 or any(
        line.startswith(("-", "+", "U")) for line in submodules.stdout.splitlines()
    ):
        raise SystemExit("sts2-ai submodule state does not match the pinned checkout")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--verify-only", action="store_true")
    parser.add_argument(
        "--destination",
        default=None,
        help="Checkout path (default: <STS2_ARTIFACT_ROOT>/dependencies/sts2-ai).",
    )
    args = parser.parse_args()
    lock = json.loads(LOCK_PATH.read_text(encoding="utf-8"))
    destination = resolve_destination(args.destination, lock, verify_only=args.verify_only)
    expected = str(lock["commit"])

    if args.verify_only:
        verify_checkout(destination, lock)
        print(f"sts2-ai verified at {expected} (tree {lock.get('tree')})")
        return

    if destination.exists():
        if not (destination / ".git").exists():
            raise SystemExit(f"refusing to replace non-Git directory: {destination}")
        actual = run("git", "rev-parse", "HEAD", cwd=destination)
        if actual != expected:
            raise SystemExit(f"existing checkout is {actual}; expected {expected}. Move it aside explicitly.")
        dirty = run("git", "status", "--porcelain", "--untracked-files=all", cwd=destination)
        if dirty:
            raise SystemExit("existing sts2-ai checkout is dirty; refusing to change its branch state")
        subprocess.check_call(["git", "checkout", "--detach", expected], cwd=destination)
        verify_checkout(destination, lock)
        print(f"sts2-ai already restored and verified at {expected}")
        return

    destination.parent.mkdir(parents=True, exist_ok=True)
    subprocess.check_call(
        ["git", "clone", "--no-checkout", "--filter=blob:none", lock["url"], str(destination)],
        cwd=ROOT,
    )
    subprocess.check_call(["git", "checkout", "--detach", expected], cwd=destination)
    verify_checkout(destination, lock)
    print(f"restored and verified sts2-ai at {expected}")


if __name__ == "__main__":
    main()
