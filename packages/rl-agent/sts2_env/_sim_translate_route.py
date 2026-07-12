"""Lossless route DTO translation for the headless simulator."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any


def _translate_route_point(raw: Any) -> dict[str, Any]:
    """Copy one simulator map point and expose its coordinate structurally.

    No subtree statistics, reachability labels, or route preferences are
    computed here.  Children remain exactly as emitted by the simulator.
    """

    if not isinstance(raw, Mapping):
        raise TypeError("simulator route point must be a mapping")
    point = dict(raw)
    if "col" in raw or "row" in raw:
        point.pop("col", None)
        point.pop("row", None)
        point["coord"] = {
            "x": int(raw.get("col", 0) or 0),
            "y": int(raw.get("row", 0) or 0),
        }
    if raw.get("point_type") is not None:
        point["point_type"] = str(raw["point_type"])
    children = raw.get("children")
    if isinstance(children, list | tuple):
        point["children"] = [
            dict(child) if isinstance(child, Mapping) else list(child)
            if isinstance(child, list | tuple)
            else child
            for child in children
        ]
    return point


__all__: list[str] = []
