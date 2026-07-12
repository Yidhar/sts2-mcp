"""Ensure committed generated version constants and game-data are current."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]


def main() -> int:
    subprocess.check_call([sys.executable, "tools/contracts/generate_contract_versions.py"], cwd=ROOT)
    subprocess.check_call([sys.executable, "tools/contracts/check_contracts.py"], cwd=ROOT)
    subprocess.check_call([sys.executable, "tools/game_data/verify_manifest.py"], cwd=ROOT)
    diff = subprocess.run(
        ["git", "diff", "--exit-code", "--", "contracts/generated", "game-data/manifest.json"],
        cwd=ROOT,
        check=False,
    )
    return diff.returncode


if __name__ == "__main__":
    raise SystemExit(main())
