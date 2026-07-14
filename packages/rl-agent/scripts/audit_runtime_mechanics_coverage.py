"""Run the fail-closed runtime mechanics ABI gate against HeadlessSim."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

PACKAGE_ROOT = Path(__file__).resolve().parents[1]
if str(PACKAGE_ROOT) not in sys.path:
    sys.path.insert(0, str(PACKAGE_ROOT))

from sts2_rl.runtime_mechanics import run_runtime_mechanics_preflight  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--sim-exe", type=Path, required=True)
    args = parser.parse_args()
    summary = run_runtime_mechanics_preflight(args.sim_exe)
    print(json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
