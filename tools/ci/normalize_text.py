"""Normalize every tracked/new source candidate to the repository text policy."""

from __future__ import annotations

import argparse
from pathlib import Path

from check_repository import (
    CRLF_SUFFIXES,
    ROOT,
    TEXT_POLICY_EXCLUDED_PREFIXES,
    TEXT_SUFFIXES,
    candidate_files,
)


SPECIAL_TEXT_NAMES = {".editorconfig", ".gitattributes", ".gitignore"}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--write",
        action="store_true",
        help="Apply normalization. Without this flag the command only reports the count.",
    )
    args = parser.parse_args()
    changes: list[str] = []
    for relative in candidate_files():
        if relative.replace("\\", "/").startswith(TEXT_POLICY_EXCLUDED_PREFIXES):
            continue
        path = ROOT / relative
        if not path.is_file():
            continue
        suffix = path.suffix.lower()
        if suffix not in TEXT_SUFFIXES and path.name.lower() not in SPECIAL_TEXT_NAMES:
            continue
        original = path.read_bytes()
        try:
            text = original.decode("utf-8-sig")
        except UnicodeError:
            continue
        normalized = text.replace("\r\n", "\n").replace("\r", "\n")
        lines = [line.rstrip(" \t") for line in normalized.split("\n")]
        while lines and not lines[-1]:
            lines.pop()
        normalized = "\n".join(lines)
        if normalized:
            normalized += "\n"
        newline = "\r\n" if suffix in CRLF_SUFFIXES else "\n"
        expected = normalized.replace("\n", newline).encode("utf-8")
        if expected == original:
            continue
        changes.append(relative)
        if args.write:
            path.write_bytes(expected)
    action = "normalized" if args.write else "would normalize"
    print(f"{action}: {len(changes)} files")
    for relative in changes[:50]:
        print(f"- {relative}")
    if len(changes) > 50:
        print(f"- ... {len(changes) - 50} more")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
