"""Fail release validation for unreviewed or non-distributable dependencies."""

from __future__ import annotations

import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]


def main() -> int:
    blocked: list[str] = []
    for path in sorted((ROOT / "third_party").glob("*.lock.json")):
        lock = json.loads(path.read_text(encoding="utf-8"))
        if lock.get("license_status") != "approved" or not lock.get("distribution_allowed"):
            blocked.append(
                f"{lock.get('name', path.stem)}: license={lock.get('license_status')}, "
                f"distribution_allowed={lock.get('distribution_allowed')}"
            )
    if blocked:
        print("release blocked by third-party review:")
        print("\n".join(f"- {line}" for line in blocked))
        return 1
    print("third-party license locks approved")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
