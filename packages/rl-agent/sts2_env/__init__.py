"""Simulator and live-bridge transport clients.

Gameplay environments, observation engineering, rewards, guards, and models
do not belong in this package.  Learning code consumes the typed
``sts2_rl.backends`` boundary instead.
"""

from .bridge_client import BridgeClient, BridgeError
from .headless_sim_bridge_client import HeadlessSimBridgeClient

__all__ = ["BridgeClient", "BridgeError", "HeadlessSimBridgeClient"]
