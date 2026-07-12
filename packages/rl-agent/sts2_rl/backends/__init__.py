"""Canonical typed environment backends and explicit legacy compatibility."""

from .headless import HeadlessBackend, HeadlessProtocolError
from .legacy import LegacyClientBackend
from .live import EnvironmentCommandError, LiveBackend

__all__ = [
    "EnvironmentCommandError",
    "HeadlessBackend",
    "HeadlessProtocolError",
    "LegacyClientBackend",
    "LiveBackend",
]
