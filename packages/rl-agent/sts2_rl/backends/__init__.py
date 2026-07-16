"""Canonical typed environment backends and explicit legacy compatibility."""

from .headless import (
    HeadlessBackend,
    HeadlessBackendPoisonedError,
    HeadlessProtocolError,
    HeadlessRecoverableProtocolError,
)
from .legacy import LegacyClientBackend
from .live import EnvironmentCommandError, LiveBackend

__all__ = [
    "EnvironmentCommandError",
    "HeadlessBackend",
    "HeadlessBackendPoisonedError",
    "HeadlessProtocolError",
    "HeadlessRecoverableProtocolError",
    "LegacyClientBackend",
    "LiveBackend",
]
