"""Build a deterministic hash/provenance manifest for game-data."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
DATA_ROOT = ROOT / "game-data"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> None:
    entries = []
    for folder in (DATA_ROOT / "raw", DATA_ROOT / "generated"):
        if not folder.exists():
            continue
        for path in sorted(p for p in folder.rglob("*") if p.is_file()):
            entries.append(
                {
                    "path": path.relative_to(DATA_ROOT).as_posix(),
                    "bytes": path.stat().st_size,
                    "sha256": sha256(path),
                }
            )

    third_party_lock = json.loads(
        (ROOT / "third_party" / "sts2-ai.lock.json").read_text(encoding="utf-8")
    )
    manifest = {
        "schema_version": "1.0.0",
        "generator": "tools/game_data/build_manifest.py",
        "generator_version": "1.0.0",
        # Release commit/tree/toolchain identity belongs in an external build
        # attestation. Embedding HEAD or wall-clock time here creates a
        # self-referential, non-reproducible source manifest.
        "upstream_sts2_ai_commit": third_party_lock["commit"],
        "files": entries,
    }
    (DATA_ROOT / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
        newline="\n",
    )


if __name__ == "__main__":
    main()
