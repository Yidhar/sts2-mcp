"""Fail when an active Markdown document links to a missing local file."""

from __future__ import annotations

import re
import subprocess
import sys
from pathlib import Path
from urllib.parse import unquote


ROOT = Path(__file__).resolve().parents[2]
LINK = re.compile(r"(?<!!)\[[^\]]*\]\(([^)]+)\)")
EXCLUDED_PREFIXES = ("docs/archive/",)


def candidates() -> list[Path]:
    output = subprocess.check_output(
        ["git", "ls-files", "--cached", "--others", "--exclude-standard", "-z", "*.md"],
        cwd=ROOT,
    )
    return [
        ROOT / value.decode("utf-8", errors="surrogateescape")
        for value in output.split(b"\0")
        if value
    ]


def main() -> int:
    errors: list[str] = []
    checked = 0
    for path in candidates():
        if not path.is_file():
            continue
        relative = path.relative_to(ROOT).as_posix()
        if relative.startswith(EXCLUDED_PREFIXES):
            continue
        checked += 1
        text = path.read_text(encoding="utf-8-sig")
        for line_number, line in enumerate(text.splitlines(), 1):
            for match in LINK.finditer(line):
                target = match.group(1).strip().strip("<>")
                if not target or target.startswith(("#", "http://", "https://", "mailto:")):
                    continue
                target = unquote(target.split("#", 1)[0].split("?", 1)[0])
                if not target or any(marker in target for marker in ("<", ">", "${", "$env:")):
                    continue
                resolved = (path.parent / target).resolve()
                try:
                    resolved.relative_to(ROOT.resolve())
                except ValueError:
                    errors.append(f"link escapes repository: {relative}:{line_number}: {target}")
                    continue
                if not resolved.exists():
                    errors.append(f"missing local link: {relative}:{line_number}: {target}")
    if errors:
        print("Markdown link validation failed:", file=sys.stderr)
        print("\n".join(f"- {error}" for error in errors), file=sys.stderr)
        return 1
    print(f"Markdown links ok: {checked} active documents")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
