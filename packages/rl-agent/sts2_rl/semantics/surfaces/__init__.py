"""Built-in surface adapters."""

from .combat import CombatSurfaceAdapter
from .event import EventSurfaceAdapter
from .map import MapSurfaceAdapter
from .opaque import OpaqueSurfaceAdapter
from .rest import RestSurfaceAdapter
from .reward import RewardSurfaceAdapter
from .selection import SelectionSurfaceAdapter
from .shop import ShopSurfaceAdapter

__all__ = [
    "CombatSurfaceAdapter",
    "EventSurfaceAdapter",
    "MapSurfaceAdapter",
    "OpaqueSurfaceAdapter",
    "RestSurfaceAdapter",
    "RewardSurfaceAdapter",
    "SelectionSurfaceAdapter",
    "ShopSurfaceAdapter",
]
