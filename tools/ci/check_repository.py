"""Repository hygiene gate used by local validation and CI."""

from __future__ import annotations

import re
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
FORBIDDEN_PREFIXES = (
    ".switchyard/",
    "artifacts/",
    "checkpoints/",
    "datasets/",
    "release-assets/",
    "scratch/",
    "tmp/",
    "train_logs/",
    "third_party/sts2-ai/",
    "packages/mcp-server/rl-smoke-logs/",
    "packages/rl-agent/analysis/",
    "packages/rl-agent/artifacts/",
    "packages/rl-agent/checkpoints/",
    "packages/rl-agent/checkpoints_attention/",
    "packages/rl-agent/checkpoints_muzero/",
    "packages/rl-agent/cleanup_reports/",
    "packages/rl-agent/data/",
    "packages/rl-agent/eval_reports/",
    "packages/rl-agent/human_demos/",
    "packages/rl-agent/logs_audit/",
    "packages/rl-agent/logs/",
    "packages/rl-agent/logs_launcher/",
    "packages/rl-agent/logs_attention/",
    "packages/rl-agent/logs_muzero/",
    "packages/rl-agent/offline_runs/",
    "packages/rl-agent/runs/",
    "packages/rl-agent/train_logs/",
    "packages/rl-agent/tmp/",
)
FORBIDDEN_SUFFIXES = (
    ".dll",
    ".log",
    ".onnx",
    ".pid",
    ".pkl",
    ".pt",
    ".safetensors",
    ".zip",
)
ABSOLUTE_PATH_PATTERNS = (
    re.compile(r"E:[/\\]game[/\\]project[/\\]sts2_mcp", re.IGNORECASE),
    re.compile(r"/mnt/e/game/project/sts2_mcp", re.IGNORECASE),
    re.compile(r"[A-Z]:[/\\]Users[/\\](?!<)[^/\\\s`'\"]+", re.IGNORECASE),
    re.compile(r"/mnt/[a-z]/Users/(?!<)[^/\s`'\"]+", re.IGNORECASE),
    re.compile(r"/home/(?!<)[^/\s`'\"]+(?:/|$)", re.IGNORECASE),
    re.compile(r"[A-Z]:[/\\](?:Program Files|SteamLibrary)(?:[/\\]|$)", re.IGNORECASE),
)
TEXT_SUFFIXES = {
    ".bat", ".cjs", ".cmd", ".cs", ".csproj", ".csv", ".editorconfig", ".gitattributes",
    ".gitignore", ".ini", ".js", ".json", ".md", ".mjs", ".openapi",
    ".patch", ".ps1", ".py", ".sh", ".sln", ".toml", ".ts", ".txt",
    ".xml", ".yaml", ".yml",
}
CRLF_SUFFIXES = {".bat", ".cmd", ".ps1", ".sln"}
PATH_SCAN_SUFFIXES = {
    ".cjs", ".cs", ".csproj", ".ini", ".js", ".json", ".md", ".mjs",
    ".ps1", ".py", ".sh", ".toml", ".ts", ".yaml", ".yml",
}
PATH_SCAN_EXCLUDED_PREFIXES = (
    "docs/archive/",
)
PATH_SCAN_EXCLUDED_FILES = {"tools/ci/check_repository.py"}


def candidate_files() -> list[str]:
    output = subprocess.check_output(
        ["git", "ls-files", "--cached", "--others", "--exclude-standard", "-z"],
        cwd=ROOT,
    )
    return [item.decode("utf-8", errors="surrogateescape") for item in output.split(b"\0") if item]


def main() -> int:
    errors: list[str] = []
    candidates = candidate_files()
    for relative in candidates:
        normalized = relative.replace("\\", "/")
        path = ROOT / relative
        # A staged or unstaged deletion is already absent from the source tree;
        # clean checkouts still exercise every tracked file.
        if not path.exists():
            continue
        if normalized.startswith(FORBIDDEN_PREFIXES):
            errors.append(f"tracked runtime artifact: {relative}")
        if normalized.lower().endswith(FORBIDDEN_SUFFIXES):
            errors.append(f"tracked binary/runtime file: {relative}")
        if Path(normalized).name in {"train.pid", "launch_command.txt"}:
            errors.append(f"tracked process state: {relative}")

        path = ROOT / relative
        suffix = path.suffix.lower()
        name = path.name.lower()
        is_text = suffix in TEXT_SUFFIXES or name in {".editorconfig", ".gitattributes", ".gitignore"}
        if not path.is_file() or not is_text:
            continue
        try:
            data = path.read_bytes()
            text = data.decode("utf-8-sig" if data.startswith(b"\xef\xbb\xbf") else "utf-8")
        except OSError:
            continue
        except UnicodeError:
            errors.append(f"invalid UTF-8 text: {relative}")
            continue

        if data.startswith(b"\xef\xbb\xbf"):
            errors.append(f"UTF-8 BOM is not allowed: {relative}")
        expected_crlf = suffix in CRLF_SUFFIXES
        bare_lf = b"\n" in data.replace(b"\r\n", b"")
        has_crlf = b"\r\n" in data
        has_bare_cr = b"\r" in data.replace(b"\r\n", b"")
        if expected_crlf:
            if bare_lf or has_bare_cr:
                errors.append(f"expected CRLF line endings: {relative}")
        elif has_crlf or has_bare_cr:
            errors.append(f"expected LF line endings: {relative}")
        if data and not data.endswith(b"\n"):
            errors.append(f"missing final newline: {relative}")

        if (
            suffix in PATH_SCAN_SUFFIXES
            and normalized not in PATH_SCAN_EXCLUDED_FILES
            and not normalized.startswith(PATH_SCAN_EXCLUDED_PREFIXES)
        ):
            for line_number, line in enumerate(text.splitlines(), 1):
                if any(pattern.search(line) for pattern in ABSOLUTE_PATH_PATTERNS):
                    errors.append(f"developer-specific path: {relative}:{line_number}")

    if errors:
        print("repository hygiene failed:", file=sys.stderr)
        print("\n".join(f"- {error}" for error in errors), file=sys.stderr)
        return 1
    print(f"repository hygiene ok: {len(candidates)} tracked/untracked candidate files")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
