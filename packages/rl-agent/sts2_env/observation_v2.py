"""Compatibility wrapper for the shared dense observation primitives.

Mainline token-world observation now lives in ``observation_v3.py``. This module
re-exports the dense feature encoder and constants for legacy callers.
"""

from __future__ import annotations

from .observation_common import *  # noqa: F401,F403
