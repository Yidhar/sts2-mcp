from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

from sts2_env.bridge_client import BridgeClient


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Export STS2 static metadata through the bridge and optionally import it into the project."
    )
    parser.add_argument("--session-path", type=str, default=None)
    parser.add_argument("--output-dir", type=str, default=None)
    parser.add_argument("--timeout-ms", type=int, default=60_000)
    parser.add_argument("--import-to-project", action="store_true")
    parser.add_argument("--content-dir", type=str, default=None)
    parser.add_argument("--dataset-dir", type=str, default=None)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    client = BridgeClient(args.session_path)
    response = client.export_static(output_dir=args.output_dir, timeout_ms=args.timeout_ms)

    print(f"[bridge-export] items_path: {response.get('items_path')}")
    print(f"[bridge-export] manifest_path: {response.get('manifest_path')}")
    print(f"[bridge-export] counts: {response.get('counts')}")

    if not args.import_to_project:
        return 0

    items_path = response.get("items_path")
    if not isinstance(items_path, str) or not items_path:
        raise SystemExit("[bridge-export] missing items_path in bridge response.")

    script_path = Path(__file__).with_name("import_sts2_exporter_items.py")
    command = [sys.executable, str(script_path), "--items", items_path]
    if args.content_dir:
        command.extend(["--content-dir", args.content_dir])
    if args.dataset_dir:
        command.extend(["--dataset-dir", args.dataset_dir])

    subprocess.run(command, check=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
