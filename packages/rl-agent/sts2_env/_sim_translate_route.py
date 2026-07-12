"""Pure route-DAG translation helpers for the headless simulator."""

from __future__ import annotations

from typing import Any

# ---------------------------------------------------------------------------
# Route subtree analysis (BFS over sim's full map graph)
# ---------------------------------------------------------------------------

# Normalize sim's PointType strings into the categories route_summary uses.
# Sim emits lowercase strings like "monster", "elite", "rest_site", "shop",
# "event", "treasure", "boss". We don't know what the question-mark node is
# emitted as in sts2-ai — leave empty and rely on fallback counts if the id
# doesn't match.
_POINT_TYPE_ALIASES: dict[str, str] = {
    "monster": "monster",
    "elite": "elite",
    "rest_site": "rest_site",
    "rest": "rest_site",
    "campfire": "rest_site",
    "shop": "shop",
    "merchant": "shop",
    "event": "event",
    "treasure": "treasure",
    "question_mark": "question_mark",
    "question": "question_mark",
    "unknown": "question_mark",
    "boss": "boss",
}


def _canonical_point_type(raw: Any) -> str:
    s = str(raw or "").strip().lower()
    return _POINT_TYPE_ALIASES.get(s, s)


def _build_route_summary(
    start_coord: tuple[int, int],
    nodes_by_coord: dict[tuple[int, int], dict[str, Any]],
    parent_child_count: int,
) -> dict[str, Any]:
    """BFS from ``start_coord`` through sim's map DAG, returning the
    bridge-compatible ``route_summary`` dict the aux heads and obs encoder
    consume.

    Sim exposes ``map.nodes`` (each with ``col``, ``row``, ``point_type``,
    ``children=[[col,row], ...]``) — computed once per state by
    BuildFullMapNodes. Per map-choice action we walk from that specific
    child, so each candidate gets its own subtree view.

    ``next_*_steps`` is 1-based from the CURRENT position: picking an elite
    as the immediate next node → next_elite_steps=1.
    ``reachable_node_count`` counts unique coords reachable (including the
    starting child).
    """
    if start_coord not in nodes_by_coord:
        return {}

    counts: dict[str, int] = {
        "monster": 0,
        "elite": 0,
        "rest_site": 0,
        "shop": 0,
        "event": 0,
        "treasure": 0,
        "question_mark": 0,
        "boss": 0,
    }
    first_depth: dict[str, int] = {}

    visited: set[tuple[int, int]] = {start_coord}
    # (coord, depth). depth=1 at the start_coord itself (1-based from the
    # player's current position).
    queue: list[tuple[tuple[int, int], int]] = [(start_coord, 1)]
    head = 0
    # Per-node tree records — what obs encoder consumes as
    # action.route_nodes. Each entry: {coord, point_type, depth,
    # child_count, is_leaf}.
    tree_nodes: list[dict[str, Any]] = []
    # Tracks first-branch location for forced_path_steps_before_branch:
    # number of depth levels from start_coord until a node has >=2
    # children. A value of 0 means the start_coord itself branches.
    forced_steps: int | None = None
    while head < len(queue):
        coord, depth = queue[head]
        head += 1
        node = nodes_by_coord.get(coord)
        if not node:
            continue
        pt = _canonical_point_type(node.get("point_type"))
        if pt in counts:
            counts[pt] += 1
        if pt and pt not in first_depth:
            first_depth[pt] = depth
        children_list = node.get("children") or []
        child_count = 0
        for child in children_list:
            if isinstance(child, list | tuple) and len(child) >= 2:
                cc = (int(child[0]), int(child[1]))
            elif isinstance(child, dict):
                cc = (int(child.get("col") or 0), int(child.get("row") or 0))
            else:
                continue
            child_count += 1
            if cc not in visited:
                visited.add(cc)
                queue.append((cc, depth + 1))
        # Record per-node tree structure for route_nodes emission.
        tree_nodes.append({
            "coord": {"col": coord[0], "row": coord[1]},
            "point_type": str(node.get("point_type") or "").title() or "Monster",
            "depth": depth,
            "child_count": child_count,
            "is_leaf": child_count == 0,
        })
        # First branching node (child_count >= 2) determines how many
        # forced-path steps there are before a real choice. Value is the
        # depth-from-start (0-indexed), i.e., depth - 1 since queue starts
        # at depth=1.
        if forced_steps is None and child_count >= 2:
            forced_steps = max(0, depth - 1)

    elite_depth = first_depth.get("elite", 10**6)
    rest_depth = first_depth.get("rest_site", 10**6)
    can_reach_rest_before_elite = rest_depth < elite_depth
    can_reach_elite_then_rest = (
        elite_depth < 10**6 and rest_depth < 10**6 and rest_depth > elite_depth
    )
    max_depth = max((n["depth"] for n in tree_nodes), default=1)

    # None for unreachable types so reward_constants._norm_step (or whatever
    # _norm_step maps unreachable to) can distinguish "no elite in subtree"
    # from "elite 15 steps away". aux_targets._norm_step handles None.
    return {
        "count_elite": counts["elite"],
        "count_rest_site": counts["rest_site"],
        "count_shop": counts["shop"],
        "count_event": counts["event"],
        "count_question_mark": counts["question_mark"],
        "count_treasure": counts["treasure"],
        "count_monster": counts["monster"],
        "count_boss": counts["boss"],
        "direct_child_count": int(parent_child_count),
        "reachable_node_count": len(visited),
        # Max depth reachable from start (obs encoder uses for tree-depth
        # feature). Capped at 15 normalization later in obs.
        "max_depth": int(max_depth),
        # Steps from start before a real choice point (branching); None
        # when there's never a branch (linear path). Obs encoder divides
        # by 10.
        "forced_path_steps_before_branch": forced_steps,
        "next_elite_steps": first_depth.get("elite"),
        "next_rest_steps": first_depth.get("rest_site"),
        "next_shop_steps": first_depth.get("shop"),
        "next_event_steps": first_depth.get("event"),
        "next_question_mark_steps": first_depth.get("question_mark"),
        "next_treasure_steps": first_depth.get("treasure"),
        "next_boss_steps": first_depth.get("boss"),
        "can_reach_rest_site_before_elite": can_reach_rest_before_elite,
        "can_reach_elite_then_rest_site": can_reach_elite_then_rest,
        # Per-node tree records. Obs encoder reads these as
        # action.route_nodes for POWER_SLOT + route-attention features.
        # Cap at MAX_ROUTE_NODES (obs encoder trims to 32). Keep BFS
        # order so shallower nodes come first.
        "nodes": tree_nodes,
    }
