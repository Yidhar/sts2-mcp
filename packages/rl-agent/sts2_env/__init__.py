"""sts2_env — Gymnasium environment for Slay the Spire 2 RL training."""

from .env import SlayTheSpire2Env
from .bridge_client import BridgeClient
from .observation import ObservationEncoder, ActionEncoder

# Phase 2: Dict observations + attention network
from .observation_v2 import DictObservationEncoder
from .env_v2 import SlayTheSpire2EnvV2
from .network import STS2AttentionExtractor

__all__ = [
    # Phase 1
    "SlayTheSpire2Env",
    "BridgeClient",
    "ObservationEncoder",
    "ActionEncoder",
    # Phase 2
    "DictObservationEncoder",
    "SlayTheSpire2EnvV2",
    "STS2AttentionExtractor",
]
