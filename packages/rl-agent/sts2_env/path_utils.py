"""Path/device helpers for mixed Windows + WSL training workflows."""

from __future__ import annotations

import os
import re
from pathlib import Path, PureWindowsPath

_WINDOWS_DRIVE_RE = re.compile(r"^[A-Za-z]:[\\/]")


def running_in_wsl() -> bool:
    """Return True when running inside Windows Subsystem for Linux."""
    if os.name == "nt":
        return False

    if os.environ.get("WSL_DISTRO_NAME"):
        return True

    candidates = [
        Path("/proc/sys/kernel/osrelease"),
        Path("/proc/version"),
    ]
    for candidate in candidates:
        try:
            content = candidate.read_text(encoding="utf-8", errors="ignore").lower()
        except OSError:
            continue
        if "microsoft" in content or "wsl" in content:
            return True
    return False


def looks_like_windows_path(value: str | os.PathLike[str] | None) -> bool:
    """Best-effort check for ``C:\\foo`` / ``E:/bar`` style paths."""
    if value is None:
        return False
    text = os.fspath(value).strip()
    return bool(_WINDOWS_DRIVE_RE.match(text))


def windows_path_to_wsl(value: str | os.PathLike[str]) -> Path:
    """Convert a Windows drive path into a WSL-mounted path.

    Example:
        ``E:\\repo\\file.txt`` -> ``/mnt/e/repo/file.txt``
    """
    raw = os.fspath(value)
    win = PureWindowsPath(raw)
    drive = win.drive.rstrip(":").lower()
    if not drive:
        raise ValueError(f"Path does not look like a Windows drive path: {raw!r}")

    parts = [part for part in win.parts[1:] if part not in ("\\", "/")]
    mounted = Path("/mnt") / drive
    if not parts:
        return mounted
    return mounted.joinpath(*parts)


def normalize_path(path: str | os.PathLike[str] | None) -> Path | None:
    """Return a local ``Path`` that works in the current runtime.

    Under WSL, Windows drive paths are converted to ``/mnt/<drive>/...``.
    Otherwise the original path is preserved.
    """
    if path is None:
        return None
    raw = os.fspath(path)
    if running_in_wsl() and looks_like_windows_path(raw):
        return windows_path_to_wsl(raw)
    return Path(raw)


def normalize_path_str(path: str | os.PathLike[str] | None) -> str | None:
    resolved = normalize_path(path)
    return None if resolved is None else str(resolved)


def _windows_roaming_candidates() -> list[Path]:
    candidates: list[Path] = []

    appdata = os.environ.get("APPDATA")
    if appdata:
        normalized = normalize_path(appdata)
        if normalized is not None:
            candidates.append(normalized)

    userprofile = os.environ.get("USERPROFILE")
    if userprofile:
        normalized = normalize_path(userprofile)
        if normalized is not None:
            candidates.append(normalized / "AppData" / "Roaming")

    if running_in_wsl():
        user_name = os.environ.get("USER") or Path.home().name
        candidates.append(Path("/mnt/c/Users") / user_name / "AppData" / "Roaming")

        users_root = Path("/mnt/c/Users")
        if users_root.exists():
            for child in users_root.iterdir():
                if not child.is_dir():
                    continue
                candidates.append(child / "AppData" / "Roaming")

    return candidates


def default_bridge_session_dir() -> Path:
    """Return the default STS2 bridge session directory for Windows or WSL."""
    for candidate in _windows_roaming_candidates():
        bridge_dir = candidate / "SlayTheSpire2" / "bridge"
        if bridge_dir.exists():
            return bridge_dir

    # Preserve the most likely location even if it does not exist yet.
    if running_in_wsl():
        user_name = os.environ.get("USER") or Path.home().name
        return Path("/mnt/c/Users") / user_name / "AppData" / "Roaming" / "SlayTheSpire2" / "bridge"

    appdata = os.environ.get("APPDATA")
    if appdata:
        return Path(appdata) / "SlayTheSpire2" / "bridge"

    userprofile = os.environ.get("USERPROFILE", "")
    return Path(userprofile) / "AppData" / "Roaming" / "SlayTheSpire2" / "bridge"


def resolve_torch_device(requested: str | None) -> str:
    """Normalize user-facing device aliases into the PyTorch runtime name."""
    text = str(requested or "cpu").strip().lower()
    if text in {"rocm", "hip"}:
        return "cuda"
    if text == "auto":
        try:
            import torch

            return "cuda" if torch.cuda.is_available() else "cpu"
        except Exception:
            return "cpu"
    return text
