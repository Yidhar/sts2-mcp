"""Translation layer: HeadlessSim state dict → bridge-shaped dict.

Kept in a separate module from ``headless_sim_bridge_client.py`` so the
translation functions can be unit-tested without spawning a subprocess.

All translation is PURE Python — no network, no files, no torch. Given a
sim state dict, produces the shape our ``WorldTokenObservationEncoder``
consumes.

Mod-specific fields (present in our Godot bridge, absent from stock sim):
  - ``players[0].facing`` — derived from ``SurroundedPower.status`` entry
  - ``combat.enemies[*].incoming_damage_multiplier`` — computed from
    BackAttack{Left,Right}Power × player facing (our Rocket/Crusher fix)
  - ``combat.self_inflicted_hp_loss_cumulative`` — tracked per-combat by
    the adapter across steps, not reconstructible from a single snapshot
  - ``event_options[*].effect_deltas`` — 17-field structured regex
    extraction; requires resolved event text which sim's stubbed locale
    doesn't emit. Deferred to later sim-side port.
  - ``power.id`` — sim already exposes ``id`` on powers; we just pass it
    through. Our bridge had a bug mapping this from PowerType (Buff/Debuff
    category) instead of the class name; sim gets it right natively.
"""
from __future__ import annotations

from typing import Any


# Sim action kind ↔ bridge action kind. We use the bridge kinds for our
# env / obs-encoder, so this is the one-way translation used at obs build
# time. The reverse (bridge → sim) is implicit because step() dispatches
# the raw sim action dict stored under ``_sim_raw``.
_SIM_KIND_TO_BRIDGE_KIND: dict[str, str] = {
    "play_card": "play_card",
    "use_potion": "use_potion",
    "discard_potion": "use_potion",
    "end_turn": "end_turn",
    "choose_event_option": "event_option",
    "choose_map_node": "map",
    "choose_rest_option": "rest_site",
    "choose_card_reward": "card_reward",
    "skip_card_reward": "reward",
    "shop_purchase": "shop",
    "shop_skip": "proceed",
    "claim_treasure": "treasure_relic",
    "choose_character": "startup",
    "embark": "startup",
    "proceed": "proceed",
    "select_card": "card_selection",
    "select_hand_card": "card_selection",
    "confirm_selection": "card_selection",
    "cancel_selection": "card_selection",
    "choose_card_select_option": "card_selection",
    # Combat-scope card selection (Scry / Discovery / keyword-scaled card pick
    # mid-combat). Sim emits these from BuildCombatLegalActions when
    # HandSelection or CardSelection is active inside a battle. Previously
    # unmapped → they fell through to "unknown" kind and the policy/encoder
    # couldn't recognize them.
    "combat_select_card": "card_selection",
    "combat_confirm_selection": "card_selection",
    "skip_rewards": "proceed",
    "claim_reward": "reward",
    "claim_relic": "treasure_relic",
    "skip_relic_select": "proceed",
    "confirm_menu_action": "startup",
}


def sim_kind_to_bridge_kind(sim_kind: str) -> str:
    return _SIM_KIND_TO_BRIDGE_KIND.get(sim_kind, sim_kind or "unknown")


# ---------------------------------------------------------------------------
# Screen / phase mapping
# ---------------------------------------------------------------------------

_STATE_TYPE_TO_SCREEN: dict[str, str] = {
    "combat": "COMBAT",
    "battle": "COMBAT",
    "event": "EVENT",
    "map": "MAP",
    "rest_site": "REST_SITE",
    "shop": "SHOP",
    "card_reward": "REWARDS",
    "rewards": "REWARDS",
    "treasure": "REWARDS",
    "card_select": "CARD_SELECTION",
    "hand_select": "CARD_SELECTION",
    "relic_select": "REWARDS",
    "game_over": "GAME_OVER",
    "victory": "GAME_OVER",
    "menu": "MAIN_MENU",
    "character_select": "STARTUP_CHARACTER_SELECT",
    "startup": "STARTUP_CHARACTER_SELECT",
}


def _screen_from_state_type(state_type: str, *, in_combat: bool = False) -> str:
    if in_combat:
        return "COMBAT"
    return _STATE_TYPE_TO_SCREEN.get(state_type, "UNKNOWN")


def _phase_from_state(state_type: str, *, in_combat: bool = False) -> str:
    # Sim's state_type labels the ROOM (monster/elite/boss/event/...), not
    # the current interaction phase. During an active combat encounter sim
    # keeps state_type="monster" while the battle dict carries player/
    # enemies/hand. The original check `state_type in {"combat","battle"}`
    # never fires in real sim flow → every combat step was mis-tagged as
    # phase="actions", obs["combat"]={"in_progress": False}, and every
    # enemy-based aux target (enemy_state, transition.next_enemy_hp_ratio,
    # objective.damage_dealt) silently zeroed out. Trust the battle-block
    # shape instead of the state_type string.
    if in_combat:
        return "combat"
    if state_type == "event":
        return "event"
    if state_type == "map":
        return "map"
    if state_type in {"game_over", "victory"}:
        return "settling"
    if state_type in {"card_select", "hand_select"}:
        return "card_selection"
    return "actions"


# ---------------------------------------------------------------------------
# Mod-specific derived fields
# ---------------------------------------------------------------------------

# Power IDs we recognize for the back-attack mechanic (Rocket/Crusher boss).
_BACK_ATTACK_LEFT_IDS = {"BACK_ATTACK_LEFT_POWER", "BackAttackLeftPower"}
_BACK_ATTACK_RIGHT_IDS = {"BACK_ATTACK_RIGHT_POWER", "BackAttackRightPower"}
_SURROUNDED_POWER_IDS = {"SURROUNDED_POWER", "SurroundedPower"}

# Cards that deal guaranteed self-damage when played. Used to attribute HP
# loss to the player rather than to enemy damage so the per-enemy HP-loss
# aux head doesn't get a false-positive signal from self-harm plays.
_SELF_DAMAGE_CARDS: dict[str, int] = {
    # id → approximate self-damage (unupgraded). If the card scales with
    # cost / other factors we err low — better to under-attribute than
    # silently double-count.
    "BLOODLETTING": 3,
    "OFFERING": 6,
    "HEMOKINESIS": 2,
    "REAPER": 0,  # heals, not self-damage
    "FEED": 0,
}


def _power_ids_of(powers: list[Any] | None) -> set[str]:
    if not isinstance(powers, list):
        return set()
    out: set[str] = set()
    for p in powers:
        if isinstance(p, dict):
            pid = p.get("id")
            if isinstance(pid, str) and pid:
                out.add(pid.upper())
    return out


def _player_facing_from_status(status: list[Any] | None) -> str | None:
    """Read SurroundedPower.Facing from player status list. Returns
    'right' / 'left' or None if the player isn't surrounded.

    The stock sim doesn't expose the Facing enum on its serialized power
    entry — only the amount. Without the field we fall back to None
    (unknown); our bridge's Rocket-aware shaping would zero-fill this
    too. Phase 3 (C#-side) could surface the enum.
    """
    if not isinstance(status, list):
        return None
    for entry in status:
        if not isinstance(entry, dict):
            continue
        pid = str(entry.get("id") or "").upper()
        if pid not in _SURROUNDED_POWER_IDS:
            continue
        # facing may be exposed as a string field or an int (0/1).
        facing = entry.get("facing")
        if isinstance(facing, str):
            f = facing.strip().lower()
            if f in {"right", "left"}:
                return f
        if isinstance(facing, int):
            return "right" if facing == 0 else "left"
        # Unknown encoding — flag presence by returning None so downstream
        # incoming_damage_multiplier stays at 1.0 rather than guessing.
        return None
    return None


def _incoming_damage_multiplier(enemy_powers: list[Any] | None, player_facing: str | None) -> float:
    """Mirror our bridge's ``ComputeIncomingDamageMultiplier``.

    1.5× when the player faces RIGHT and enemy has BackAttackLeftPower, or
    the mirror case. 1.0× otherwise.
    """
    if player_facing is None:
        return 1.0
    ids = _power_ids_of(enemy_powers)
    has_left = bool(ids & _BACK_ATTACK_LEFT_IDS)
    has_right = bool(ids & _BACK_ATTACK_RIGHT_IDS)
    if player_facing == "right" and has_left:
        return 1.5
    if player_facing == "left" and has_right:
        return 1.5
    return 1.0


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
            if isinstance(child, (list, tuple)) and len(child) >= 2:
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


def _build_decision_block(
    *,
    state_type: str,
    in_combat: bool,
    event: dict[str, Any],
    map_state: dict[str, Any],
    rest_site: dict[str, Any],
    shop: dict[str, Any],
    treasure: dict[str, Any],
    rewards: dict[str, Any],
    card_reward: dict[str, Any],
    card_select: dict[str, Any],
    hand_select: dict[str, Any],
) -> dict[str, Any]:
    """Mirror of bridge's BuildEnvDecisionPayload phase switch.

    Returns the 10-scalar dict observation_common reads (~line 863). Keys
    not applicable to a phase are simply omitted — _float default is 0.0.
    """
    if state_type == "event":
        opts = event.get("options") or []
        title = str(event.get("title") or event.get("name") or "事件")
        return {
            "option_count": len(opts),
            "decision_text": f"事件｜{title}｜{len(opts)}个选项",
        }
    if state_type == "card_reward":
        choices = card_reward.get("cards") or []
        return {
            "option_count": len(choices),
            "can_skip": bool(card_reward.get("can_skip", True)),
            "decision_text": f"卡牌奖励｜{len(choices)}张卡牌可选",
        }
    if state_type in {"rewards", "combat_rewards", "combat_post_end_pending"}:
        items = rewards.get("items") or []
        return {
            "reward_count": len(items),
            "proceed_only": bool(rewards.get("can_proceed", False)) and len(items) == 0,
            "decision_text": f"奖励选择｜可领取{len(items)}项奖励",
        }
    if state_type == "map":
        opts = map_state.get("next_options") or []
        return {
            "travelable_count": len(opts),
            "decision_text": f"地图｜{len(opts)}个可选节点",
        }
    if state_type == "rest_site":
        opts = rest_site.get("options") or []
        return {
            "option_count": len(opts),
            "can_proceed": bool(rest_site.get("can_proceed", False)),
            "decision_text": "营火｜选择休息或锻造",
        }
    if state_type == "shop":
        items = shop.get("items") or []
        return {
            "is_open": bool(shop.get("is_open", False)),
            "item_count": len(items),
            "decision_text": f"商店｜{len(items)}件商品",
        }
    if state_type == "treasure":
        relic_opts = treasure.get("relics") or []
        return {
            "option_count": len(relic_opts),
            "can_proceed": bool(treasure.get("can_open", True)),
            "decision_text": f"宝箱｜{len(relic_opts)}件遗物可选",
        }
    if state_type in {"card_select", "hand_select"}:
        src = card_select if card_select else hand_select
        if not isinstance(src, dict):
            src = {}
        prompt = str(src.get("prompt") or "").strip()
        min_sel = int(src.get("min_select") or 0)
        max_sel = int(src.get("max_select") or 1)
        selected = len(src.get("selected_cards") or [])
        label = prompt or ("手牌选择" if state_type == "hand_select" else "卡牌选择")
        return {
            "selected_count": selected,
            "min_select": min_sel,
            "max_select": max_sel,
            "can_skip": bool(src.get("can_cancel", False)),
            "decision_text": f"{label}｜已选{selected}｜{min_sel}-{max_sel}张",
        }
    if in_combat:
        # During combat there's no global decision prompt — return empty.
        # Combat features come through obs["combat"] token pipeline.
        return {}
    return {}


class SelfInflictedHpTracker:
    """Tracks cumulative HP loss attributable to the player's own actions
    within a combat. Reset when combat starts; advanced when a known
    self-damage card is played and the player's HP subsequently drops.

    Not reconstructible from a single snapshot — this object lives on the
    bridge client and is fed both the last action dispatched and the next
    observation's HP.
    """

    def __init__(self) -> None:
        self.cumulative: int = 0
        self._last_hp: int | None = None
        self._last_self_damage_card: str | None = None

    def reset(self, initial_hp: int | None) -> None:
        self.cumulative = 0
        self._last_hp = int(initial_hp) if isinstance(initial_hp, int) else None
        self._last_self_damage_card = None

    def note_action(self, action_raw: dict[str, Any] | None) -> None:
        """Called when the env dispatches an action. If it's playing a
        known self-damage card, remember the card id so the next HP drop
        can be attributed.
        """
        if not isinstance(action_raw, dict):
            self._last_self_damage_card = None
            return
        if str(action_raw.get("action") or "") != "play_card":
            self._last_self_damage_card = None
            return
        card_id = str(action_raw.get("card_id") or "").upper()
        # Strip CARD. prefix variants
        if card_id.startswith("CARD."):
            card_id = card_id.split(".", 1)[1]
        if card_id in _SELF_DAMAGE_CARDS:
            self._last_self_damage_card = card_id
        else:
            self._last_self_damage_card = None

    def observe_hp(self, current_hp: int | None) -> None:
        """Called after each step with the player's new HP. Attributes a
        drop to self-damage iff the previous action was a self-harm card.
        """
        if not isinstance(current_hp, int):
            return
        if self._last_hp is not None and self._last_self_damage_card is not None:
            delta = max(self._last_hp - current_hp, 0)
            if delta > 0:
                # Cap by the card's approximate self-damage — keeps enemy
                # damage dealt on the same tick from leaking in.
                cap = _SELF_DAMAGE_CARDS.get(self._last_self_damage_card, 0)
                if cap > 0:
                    delta = min(delta, cap)
                self.cumulative += int(delta)
        self._last_hp = int(current_hp)
        self._last_self_damage_card = None


# ---------------------------------------------------------------------------
# Translation
# ---------------------------------------------------------------------------

def translate_to_bridge_shape(
    sim_state: dict[str, Any],
    *,
    episode_id: str,
    self_inflicted_tracker: SelfInflictedHpTracker | None = None,
) -> dict[str, Any]:
    """Main entry: sim state dict → bridge-shaped dict the obs encoder reads."""
    state_type = str(sim_state.get("state_type") or "").lower()

    # Every sub-state section (when present) carries the current player
    # snapshot — we prefer battle.player when in combat, otherwise the
    # first available.
    sim_run = sim_state.get("run") or {}
    battle = sim_state.get("battle") or {}
    event = sim_state.get("event") or {}
    map_state = sim_state.get("map") or {}
    rest_site = sim_state.get("rest_site") or {}
    shop = sim_state.get("shop") or {}
    treasure = sim_state.get("treasure") or {}
    rewards_state = sim_state.get("rewards") or {}
    card_reward = sim_state.get("card_reward") or {}
    card_select = sim_state.get("card_select") or {}
    hand_select = sim_state.get("hand_select") or {}
    relic_select = sim_state.get("relic_select") or {}
    game_over = sim_state.get("game_over") or {}

    sim_player = (
        battle.get("player")
        or event.get("player")
        or map_state.get("player")
        or rest_site.get("player")
        or shop.get("player")
        or treasure.get("player")
        or rewards_state.get("player")
        or card_reward.get("player")
        or card_select.get("player")
        or hand_select.get("player")
        or relic_select.get("player")
        or sim_state.get("player")
        or {}
    )

    # Sim labels the *room* (monster/elite/boss) in state_type and carries
    # the live fight in the ``battle`` block. The presence of battle.player
    # (or battle.enemies) is the authoritative "we are currently mid-combat"
    # signal — much more reliable than state_type string matching.
    in_combat = (
        state_type in {"combat", "battle"}
        or bool(battle.get("player"))
        or bool(battle.get("enemies"))
    )
    screen = _screen_from_state_type(state_type, in_combat=in_combat)
    player_facing = _player_facing_from_status(sim_player.get("status"))

    sim_legal_actions = sim_state.get("legal_actions") or []

    # --- Combat block ---
    bridge_combat: dict[str, Any] = {"in_progress": in_combat}
    if in_combat and battle:
        bridge_combat.update(_translate_combat_block(
            battle, sim_player,
            player_facing=player_facing,
            self_inflicted=self_inflicted_tracker.cumulative if self_inflicted_tracker else 0,
        ))

    # --- Player / deck / relics / potions ---
    # Two shapes are needed: the legacy nested `players[0]` block (list with
    # `creature.current_hp` etc.) AND a flat `player` dict that all downstream
    # consumers (reward shaping in combat_env._player_hp_delta_reward, obs
    # encoder in observation_v3, aux_targets in aux_targets.py) actually read.
    # The real bridge emits both; the sim translator was only emitting the
    # nested list form, so every player-side reward and player-side obs
    # feature was zero-filled during sim training.
    bridge_players = [_translate_player_block(
        sim_player,
        in_combat=in_combat,
        player_facing=player_facing,
    )]
    bridge_player = _translate_flat_player_block(
        sim_player,
        player_facing=player_facing,
    )

    # --- Map ---
    bridge_map = _translate_map_block(sim_run, map_state)

    # --- Rewards / shops / rest / treasure / card-reward / card-select ---
    bridge_rewards = _translate_rewards_block(rewards_state, card_reward, treasure, relic_select)
    bridge_rest_site = _translate_rest_site_block(rest_site)
    bridge_shop = _translate_shop_block(shop)
    bridge_card_reward_sel = _translate_card_reward_sel_block(card_reward)
    bridge_card_sel = _translate_card_sel_block(card_select, hand_select, battle.get("card_selection") if battle else None)

    # --- Event options (with effect_deltas TODO) ---
    event_options = _translate_event_options(event)

    # --- Run block ---
    bridge_run = _translate_run_block(sim_run, state_type, game_over)

    # --- Legal actions (rich bridge-shaped entries) ---
    bridge_actions = _translate_legal_actions(
        sim_legal_actions,
        sim_player=sim_player,
        battle=battle,
        map_state=map_state,
        event=event,
        rest_site=rest_site,
        shop=shop,
        card_reward=card_reward,
        card_select=card_select,
        treasure=treasure,
    )

    phase = _phase_from_state(state_type, in_combat=in_combat)
    # Mirror live ResolveEnvDecisionDomain (BridgeGameApi.EnvPayloads.cs:442):
    # combat→combat, map→route, card_selection/settling→combat if in-combat
    # else build, everything else→build. Obs encoder (_resolve_domain at
    # observation_common.py:1547) reads this top-level.
    if phase == "combat":
        decision_domain = "combat"
    elif phase == "map":
        decision_domain = "route"
    elif phase in {"card_selection", "settling"}:
        decision_domain = "combat" if in_combat else "build"
    else:
        decision_domain = "build"

    bridge_state: dict[str, Any] = {
        "ok": True,
        "backend": "headless_sim",
        "captured_at_utc": "",
        "state_version": int(sim_state.get("state_version") or 0),
        "state_hash": str(sim_state.get("state_hash") or ""),
        "semantic_state_hash": str(sim_state.get("semantic_state_hash") or ""),
        "schema_version": "sim-v1",
        "bridge_version": "headless_sim",
        "decision_domain": decision_domain,
        "screen": screen,
        "players": bridge_players,
        "player": bridge_player,
        "combat": bridge_combat,
        "map": bridge_map,
        "rewards": bridge_rewards,
        "run": bridge_run,
        "event_options": event_options,
        "card_selection": bridge_card_sel,
        "card_reward_selection": bridge_card_reward_sel,
        "character_selection": {"visible": state_type == "character_select", "options": []},
        "run_mode_selection": {"visible": False},
        "deck_upgrade_selection": {"visible": False, "choices": []},
        "main_menu": {"visible": screen == "MAIN_MENU"},
        # Per-phase decision dict consumed by observation_common._append
        # (10 scalar features: option_count, can_skip, selected_count,
        # min_select, max_select, is_open, travelable_count, can_proceed,
        # reward_count, item_count). Real bridge emits via
        # BuildEnvDecisionPayload (BridgeGameApi.EnvPayloads.cs:691-762);
        # translator was missing it entirely → every decision/phase
        # feature read as zero during sim training.
        "decision": _build_decision_block(
            state_type=state_type,
            in_combat=in_combat,
            event=event,
            map_state=map_state,
            rest_site=rest_site,
            shop=shop,
            treasure=treasure,
            rewards=rewards_state,
            card_reward=card_reward,
            card_select=card_select,
            hand_select=hand_select,
        ),
        "rest_site": bridge_rest_site,
        "shop": bridge_shop,
        "crystal_sphere": {"visible": False},
        "automation": {"enabled": False},
        "available_actions": bridge_actions,
        "_sim_raw": sim_state,
        "episode_id": episode_id,
        "phase": phase,
    }
    return bridge_state


def _translate_player_powers(status_list: list[Any] | None) -> list[dict[str, Any]]:
    if not isinstance(status_list, list):
        return []
    out: list[dict[str, Any]] = []
    for entry in status_list:
        if not isinstance(entry, dict):
            continue
        pid = str(entry.get("id") or "")
        amount = int(entry.get("amount") or 0)
        out.append({
            "id": pid,
            "title": pid,
            "amount": amount,
            "display_amount": int(entry.get("display_amount", amount) or amount),
            "stack_type": str(entry.get("stack_type") or "Single"),
        })
    return out


def _translate_card(sim_card: Any, *, pile: str = "Deck") -> dict[str, Any]:
    """Turn a sim FullRunApiCardOption into our bridge card dict.

    Emits the bridge's *compact* card shape (BridgeGameApi.EnvCompact.cs
    :21-117). Downstream readers in observation_common/run_memory/
    semantic_action/combat_tactical_local/combat_env/env_v2 key off
    ``cost`` / ``upgrade_level`` / ``target`` / ``x_cost`` / ``canonical_text``
    — NOT the older ``canonical_energy_cost`` / ``current_upgrade_level`` /
    ``target_type`` / ``costs_x``. Legacy keys are kept alongside for any
    path that still reads them, but the new compact keys are authoritative.
    """
    if not isinstance(sim_card, dict):
        return {"missing": True}
    raw_id = str(sim_card.get("id") or "UNKNOWN")
    # Sim uses bare ids ("STRIKE_IRONCLAD"); bridge uses "CARD.STRIKE_IRONCLAD".
    card_id = raw_id if raw_id.startswith("CARD.") else f"CARD.{raw_id}"
    cost = sim_card.get("cost")
    cost_int = int(cost) if isinstance(cost, int) else 0
    # X-cost detection: sim runtime card DTO doesn't expose `is_x_cost` on the
    # hand payload (FullRunApiCardOption only has `cost: int?`), so infer from
    # the static card registry where energy_cost_text == "X". obs encoder
    # (observation_common.py:988) reads `x_cost`; legacy `costs_x` kept.
    is_x_cost = _card_is_x_cost_from_registry(card_id) or (cost is None and cost_int == 0)
    is_upgraded = bool(sim_card.get("is_upgraded"))
    upgrade_level = 1 if is_upgraded else 0
    target = str(sim_card.get("target_type") or "None")
    card_type = str(sim_card.get("type") or "Attack")
    title = str(sim_card.get("name") or raw_id)
    description = str(sim_card.get("description") or "")
    valid_targets = sim_card.get("valid_target_ids") or []
    effect_preview = _card_effect_preview_from_registry(card_id)
    # Mirror bridge's canonical_text construction (EnvCompact.cs:99-113) so
    # the text encoder gets the same Chinese semantic string it would see
    # in real-game training.
    ct_parts = ["卡牌", _normalize_semantic_text(title), card_type]
    if cost_int:
        ct_parts.append(f"能量{cost_int}")
    ct_parts.append(f"目标{_translate_target_type(target)}")
    if description:
        ct_parts.append(f"效果：{_normalize_semantic_text(description)}")
    canonical_text = "｜".join(ct_parts)
    payload: dict[str, Any] = {
        "id": card_id,
        # New compact keys (authoritative — matches bridge):
        "upgrade_level": upgrade_level,
        "cost": cost_int,
        "target": target,
        "type": card_type,
        "canonical_text": canonical_text,
        # Legacy keys (retained for any consumer still using them):
        "current_upgrade_level": upgrade_level,
        "max_upgrade_level": 1,
        "title": title,
        "description": description,
        "rarity": str(sim_card.get("rarity") or "Basic"),
        "target_type": target,
        "pile": pile,
        "is_playable": bool(sim_card.get("can_play", True)),
        "canonical_energy_cost": cost_int,
        "resolved_energy_cost": cost_int,
        "costs_x": is_x_cost,
        "x_cost": is_x_cost,
        # Regent-only star resource. Sim's FullRunApiCardOption has no
        # per-card star field (only the combat-global `stars` counter).
        # Emit None explicitly so the probe sees the key as "translated"
        # rather than "discarded"; obs encoder's 0.0 fallback for missing
        # keys is the correct signal for non-Regent characters.
        "star": None,
        "star_x": False,
        "keywords": list(sim_card.get("keywords") or []),
        "valid_target_ids": [int(t) for t in valid_targets if isinstance(t, (int, float))],
        # Populate effect_preview from content_registry's semantic_signals
        # (static base values per card). Real bridge computes this live
        # including Strength/Weak/Vulnerable modifiers; we get the base
        # static values here. Upgrade modifiers skipped (Phase 5+).
        "effect_preview": effect_preview,
    }
    # Keep runtime per-card modifiers if the simulator exposes them.  Live
    # bridge emits the same compact fields; observation_v3 turns these into
    # CARD_KEYWORD_SLOT tokens (bound/card_lock/cost_lock/temporary/etc.).
    for _modifier_field in ("afflictions", "enchantments", "modifiers", "card_modifiers"):
        _mods = sim_card.get(_modifier_field)
        if isinstance(_mods, list) and _mods:
            payload[_modifier_field] = _mods[:16]
    # Bridge compact card emits ``effect`` (short summary) when present,
    # falling back to ``description``. effect_preview.summary isn't
    # populated today but description is non-empty — keep both.
    if description:
        payload["effect"] = description
    return payload


def _normalize_semantic_text(s: str) -> str:
    # Minimal sanitation — strip newlines, clamp length. The real bridge's
    # NormalizeSemanticText does similar; exact parity isn't critical
    # because the text encoder is robust to whitespace/newline differences.
    if not s:
        return ""
    return " ".join(s.replace("\n", " ").split())[:200]


def _translate_target_type(target: str) -> str:
    t = str(target or "").strip()
    mapping = {
        "AnyEnemy": "敌人",
        "AllEnemies": "所有敌人",
        "Self": "自身",
        "Ally": "友军",
        "None": "无",
    }
    return mapping.get(t, t or "无")


# Maps content_registry's semantic_signals keys to the keys that
# observation_common._preview_metric (and 30+ call sites across the codebase)
# read. We also copy through any already-canonical keys unchanged. Applied
# each time _translate_card runs; cheap dict lookup (content_registry cache
# is already warm).
_SEMANTIC_SIGNAL_TO_PREVIEW: dict[str, str] = {
    "damage": "damage",
    "block": "block",
    "draw": "draw",
    "heal": "heal",
    "weak": "weak",
    "vulnerable": "vulnerable",
    "frail": "frail",
    "strength": "strength",
    "strengthGain": "strength",
    "dexterity": "dexterity",
    "dexterityGain": "dexterity",
    "energy": "energy",
    "energyGain": "energy",
    "hpLoss": "hp_loss",
    "hp_loss": "hp_loss",
    "poison": "poison",
    "hits": "hits",
    "damage_per_hit": "damage_per_hit",
    "summon": "summon",
}


def _card_is_x_cost_from_registry(card_id: str) -> bool:
    try:
        from content_registry import get_card_metadata  # noqa: PLC0415
        md = get_card_metadata(card_id)
    except Exception:
        return False
    if not isinstance(md, dict):
        return False
    text = str(md.get("energy_cost_text") or "").strip().upper()
    return text == "X"


def _card_effect_preview_from_registry(card_id: str) -> dict[str, Any]:
    try:
        from content_registry import get_card_metadata  # noqa: PLC0415
        md = get_card_metadata(card_id)
    except Exception:
        md = None
    if not isinstance(md, dict):
        return {}
    signals = md.get("semantic_signals")
    if not isinstance(signals, dict):
        return {}
    preview: dict[str, Any] = {}
    for src_key, dst_key in _SEMANTIC_SIGNAL_TO_PREVIEW.items():
        if src_key in signals:
            try:
                preview[dst_key] = float(signals[src_key])
            except (TypeError, ValueError):
                continue
    # Downstream readers look for total_damage and total_block aliases
    # (_preview_metric's key_aliases). Mirror if single-hit.
    if "damage" in preview and "total_damage" not in preview:
        preview["total_damage"] = preview["damage"]
    if "block" in preview and "total_block" not in preview:
        preview["total_block"] = preview["block"]
    return preview


def _translate_player_block(
    sim_player: dict[str, Any],
    *,
    in_combat: bool,
    player_facing: str | None,
) -> dict[str, Any]:
    hp = int(sim_player.get("current_hp", sim_player.get("hp") or 0) or 0)
    max_hp = int(sim_player.get("max_hp") or 0)
    character = str(sim_player.get("character") or "IRONCLAD")
    deck_cards = [_translate_card(c, pile="Deck") for c in (sim_player.get("deck") or [])]
    return {
        "index": 0,
        "net_id": 1,
        "character": {
            "id": f"CHARACTER.{character}",
            "title": character,
            "description": "",
            "kind": character.title(),
        },
        "gold": int(sim_player.get("gold") or 0),
        "max_energy": int(sim_player.get("max_energy") or 3),
        "facing": player_facing,
        "creature": {
            "name": character,
            "model_id": f"CHARACTER.{character}",
            "combat_id": 0,
            "side": "Player",
            "current_hp": hp,
            "max_hp": max_hp,
            "block": int(sim_player.get("block") or 0),
            "is_alive": hp > 0,
            "is_hittable": hp > 0,
            "powers": _translate_player_powers(sim_player.get("status")),
            "intent": None,
            "static_traits": [],
            "reactive_triggers": [],
            "phase_rules": [],
            "combat_tags": [],
            "danger_profile": None,
            "target_priority_hints": [],
        },
        "combat": {"in_combat": in_combat},
        "deck": {
            "pile_type": "Deck",
            "is_combat_pile": False,
            "count": len(deck_cards),
            "cards": deck_cards,
        },
        "relics": [_translate_relic(r) for r in (sim_player.get("relics") or [])],
        "potions": [_translate_potion(p) for p in (sim_player.get("potions") or [])],
    }


def _translate_flat_player_block(
    sim_player: dict[str, Any],
    *,
    player_facing: str | None,
) -> dict[str, Any]:
    """Flat ``obs["player"]`` dict matching the real bridge's shape.

    Downstream consumers (combat_env reward shaping, observation_v3 encoder,
    aux_targets supervision) read fields directly off ``obs["player"]`` —
    they do not walk the nested ``obs["players"][0].creature.*`` tree.
    """
    hp = int(sim_player.get("current_hp", sim_player.get("hp") or 0) or 0)
    max_hp = int(sim_player.get("max_hp") or 0)
    character = str(sim_player.get("character") or "IRONCLAD")
    deck_cards = [_translate_card(c, pile="Deck") for c in (sim_player.get("deck") or [])]
    powers = _translate_player_powers(sim_player.get("status"))
    block = int(sim_player.get("block") or 0)
    return {
        "hp": hp,
        "current_hp": hp,
        "max_hp": max_hp,
        "block": block,
        "gold": int(sim_player.get("gold") or 0),
        "max_energy": int(sim_player.get("max_energy") or 3),
        "character": character,
        "facing": player_facing,
        "status": powers,
        "powers": powers,
        "deck_cards": deck_cards,
        "deck": len(deck_cards),
        "relics": [_translate_relic(r) for r in (sim_player.get("relics") or [])],
        "potions": [_translate_potion(p) for p in (sim_player.get("potions") or [])],
        # Legacy back-compat: aux_targets line 181 reads player["creature"]["block"]
        # as a fallback. Keep the nested form populated too.
        "creature": {
            "current_hp": hp,
            "max_hp": max_hp,
            "block": block,
        },
    }


def _translate_relic(sim_relic: Any) -> dict[str, Any]:
    if not isinstance(sim_relic, dict):
        return {}
    rid = str(sim_relic.get("id") or "")
    return {
        "id": rid if rid.startswith("RELIC.") else f"RELIC.{rid}",
        "title": str(sim_relic.get("name") or rid),
        "description": str(sim_relic.get("description") or ""),
        "rarity": str(sim_relic.get("rarity") or "Common"),
        "counter": int(sim_relic.get("counter") or 0),
    }


def _translate_potion(sim_potion: Any) -> dict[str, Any]:
    if not isinstance(sim_potion, dict):
        return {}
    pid = str(sim_potion.get("id") or "")
    return {
        "slot": int(sim_potion.get("slot") or 0),
        "id": pid if pid.startswith("POTION.") else f"POTION.{pid}",
        "title": str(sim_potion.get("name") or pid),
        "description": str(sim_potion.get("description") or ""),
        "target_type": str(sim_potion.get("target_type") or "None"),
        "can_use_in_combat": bool(sim_potion.get("can_use_in_combat", True)),
        "keywords": list(sim_potion.get("keywords") or []),
    }


def _translate_intent(intents: list[Any] | None) -> dict[str, Any]:
    if not isinstance(intents, list) or not intents:
        return {
            "intent_type": "",
            "title": "",
            "description": "",
            "total_damage": 0.0,
            "repeats": 0,
        }
    # Take the first listed intent — matches our bridge behavior.
    first = intents[0] if isinstance(intents[0], dict) else {}
    total_damage = first.get("total_damage")
    if total_damage is None:
        per_hit = first.get("damage") or 0
        repeats = first.get("repeats") or 1
        total_damage = float(per_hit) * float(repeats)
    return {
        "intent_type": str(first.get("type") or ""),
        "title": str(first.get("title") or first.get("label") or ""),
        "description": str(first.get("description") or ""),
        "total_damage": float(total_damage or 0.0),
        "damage_per_hit": float(first.get("damage") or 0.0),
        "repeats": int(first.get("repeats") or 1),
    }


def _build_enemy_trait_payloads(
    name: str, model_id: str, next_move_id: str
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    """Mirror of live bridge's BuildEnemyStaticTraitPayloads +
    BuildEnemyReactiveTriggerPayloads + BuildEnemyPhaseRulePayloads.
    Keyword-driven detection of combat-critical enemy traits.

    Returns (static_traits, reactive_triggers, phase_rules). Each entry has
    the 10-field trait payload shape the obs encoder reads (trait,
    description, trigger_type, condition, effect_type, effect_amount,
    severity, etc.).

    Obs encoder (_infer_enemy_traits in observation_v3.py) reads these
    three arrays directly from each enemy dict. Live bridge emits them;
    sim never did, so policy missed ~6-12 trait classes like
    summon_engine, gatekeeper, time_scaling, countdown_tick, reveal_boss.
    """
    search = " ".join([name, model_id, next_move_id]).lower()
    static_traits: list[dict[str, Any]] = []
    reactive_triggers: list[dict[str, Any]] = []
    phase_rules: list[dict[str, Any]] = []

    def _mk(category: str, trait: str, description: str,
            trigger_type: str | None = None, condition: str | None = None,
            effect_type: str | None = None, effect_amount: int | None = None,
            severity: str = "medium") -> dict[str, Any]:
        return {
            "category": category, "trait": trait, "description": description,
            "trigger_type": trigger_type, "condition": condition,
            "effect_type": effect_type, "effect_amount": effect_amount,
            "severity": severity,
        }

    # Static traits
    if any(k in search for k in ("nexus", "progenitor", "queen", "egg")):
        static_traits.append(_mk("static", "summon_engine",
            "Acts as a board-pressure engine or summon core.", severity="high"))
    if "door" in search:
        static_traits.append(_mk("static", "gatekeeper",
            "Encounter progression is gated until this unit's cycle is solved.", severity="high"))
    if any(k in search for k in ("matriarch", "nexus", "byrdonis", "wurm")):
        static_traits.append(_mk("static", "time_scaling",
            "Threat grows materially if the fight drags.", severity="high"))
    if any(k in search for k in ("retali", "thorn", "spiny")):
        static_traits.append(_mk("static", "contact_retaliate",
            "Punishes contact hits or spammy multi-hit plans.", severity="high"))

    # Reactive triggers
    if any(k in search for k in ("retali", "thorn", "spike", "spiny")):
        reactive_triggers.append(_mk("reactive", "retaliate",
            "On hit or contact, this enemy punishes damage with retaliation.",
            trigger_type="on_hit", condition="contact", effect_type="retaliate",
            severity="high"))
    if any(k in search for k in ("egg", "progenitor", "summon")):
        reactive_triggers.append(_mk("reactive", "summon",
            "If left alive or when killed, this enemy can continue board pressure via summons.",
            trigger_type="on_turn_end", condition="alive", effect_type="summon",
            severity="high"))

    # Phase rules
    if any(k in search for k in ("split", "prism")):
        phase_rules.append(_mk("phase", "split",
            "Crossing a threshold can split or multiply the board state.",
            trigger_type="on_hp_threshold", condition="threshold_crossed",
            effect_type="split", severity="medium"))
    if any(k in search for k in ("phase", "threshold", "subject", "doormaker")):
        phase_rules.append(_mk("phase", "phase_shift",
            "The enemy has threshold- or cycle-based phase changes.",
            trigger_type="on_hp_threshold", condition="threshold_crossed",
            effect_type="phase_shift", severity="high"))
    if "intang" in search:
        phase_rules.append(_mk("phase", "gain_intangible",
            "Intangible windows change when burst should be committed.",
            trigger_type="on_turn_start", condition="intangible_window",
            effect_type="gain_intangible", severity="high"))
    if "insatiable" in search:
        phase_rules.append(_mk("phase", "countdown_tick",
            "An external countdown or timer pressures the fight every turn.",
            trigger_type="on_turn_start", condition="countdown_active",
            effect_type="countdown_tick", severity="high"))
    if model_id.upper() == "MONSTER.DOOR":
        phase_rules.append(_mk("phase", "reveal_boss",
            "Destroying the door exposes the main boss window.",
            trigger_type="on_death", condition="door_destroyed",
            effect_type="reveal_boss", severity="high"))

    return static_traits, reactive_triggers, phase_rules


def _translate_combat_block(
    battle: dict[str, Any],
    sim_player: dict[str, Any],
    *,
    player_facing: str | None,
    self_inflicted: int,
) -> dict[str, Any]:
    enemies: list[dict[str, Any]] = []
    for enemy in battle.get("enemies") or []:
        if not isinstance(enemy, dict):
            continue
        powers = _translate_player_powers(enemy.get("status"))
        enemy_name = str(enemy.get("name") or enemy.get("entity_id") or "")
        enemy_model_id = str(enemy.get("entity_id") or "")
        enemy_next_move = str(enemy.get("next_move_id") or "")
        static_traits, reactive_triggers, phase_rules = _build_enemy_trait_payloads(
            enemy_name, enemy_model_id, enemy_next_move,
        )
        enemies.append({
            "id": int(enemy.get("combat_id") or 0),
            "name": enemy_name,
            "model_id": enemy_model_id,
            "hp": int(enemy.get("hp") or 0),
            "max_hp": int(enemy.get("max_hp") or 0),
            "block": int(enemy.get("block") or 0),
            "is_alive": bool(enemy.get("is_alive", True)),
            "is_hittable": bool(enemy.get("is_hittable", True)),
            "incoming_damage_multiplier": _incoming_damage_multiplier(
                enemy.get("status"), player_facing,
            ),
            "intent": _translate_intent(enemy.get("intents")),
            "powers": powers,
            "next_move_id": enemy_next_move,
            "intends_to_attack": bool(enemy.get("intends_to_attack", False)),
            # Effect algebra payloads (live parity). Obs encoder's
            # _infer_enemy_traits reads these to populate POWER_SLOT
            # structured effect vectors (Phase 6.1). Without them, sim
            # training saw zero structured effects for 6+ trait classes.
            "static_traits": static_traits,
            "reactive_triggers": reactive_triggers,
            "phase_rules": phase_rules,
        })
    hand_cards = [_translate_card(c, pile="Hand") for c in (sim_player.get("hand") or [])]
    # Sim exposes full pile lists under player.{draw,discard,exhaust}_pile.
    # Encoder's DRAW_PREVIEW_CARD / DISCARD_CARD / EXHAUST_CARD tokens look
    # for combat.{draw_pile,discard_pile,exhaust_pile} as lists (or fallback
    # keys draw_preview_cards / discard_cards / exhaust_cards). Before this,
    # the translator only emitted counts, so every cross-pile policy signal
    # (what's left to draw, what got discarded, scaling-card presence) was
    # zero-filled throughout training.
    draw_cards = [_translate_card(c, pile="Draw") for c in (sim_player.get("draw_pile") or [])]
    discard_cards = [_translate_card(c, pile="Discard") for c in (sim_player.get("discard_pile") or [])]
    exhaust_cards = [_translate_card(c, pile="Exhaust") for c in (sim_player.get("exhaust_pile") or [])]
    # Sim doesn't expose play_pile separately today; leave empty so the
    # encoder PLAY_PILE_CARD tokens mask-out cleanly. Powers still come via
    # player_powers / enemy.powers.
    play_pile_cards: list[dict[str, Any]] = []
    return {
        "round": int(battle.get("round") or 0),
        "side": str(battle.get("turn") or "Player"),
        "play_phase": bool(battle.get("is_play_phase", True)),
        "can_act": bool(battle.get("is_play_phase", True)),
        "facing": player_facing,
        "self_inflicted_hp_loss_cumulative": int(self_inflicted),
        "energy": int(sim_player.get("energy") or 0),
        "max_energy": int(sim_player.get("max_energy") or 3),
        "stars": int(sim_player.get("stars") or 0),
        "block": int(sim_player.get("block") or 0),
        "hand": hand_cards,
        "draw_pile": draw_cards,
        "discard_pile": discard_cards,
        "exhaust_pile": exhaust_cards,
        "play_pile": play_pile_cards,
        # Keep legacy count fields too — some training-time info/debug paths
        # read combat.draw / combat.discard as scalars.
        "draw": len(draw_cards) if draw_cards else int(sim_player.get("draw_pile_count") or 0),
        "discard": len(discard_cards) if discard_cards else int(sim_player.get("discard_pile_count") or 0),
        "exhaust": len(exhaust_cards) if exhaust_cards else int(sim_player.get("exhaust_pile_count") or 0),
        "allies": [],
        "enemies": enemies,
        "player_powers": _translate_player_powers(sim_player.get("status")),
    }


def _translate_map_block(sim_run: dict[str, Any], map_state: dict[str, Any]) -> dict[str, Any]:
    floor = int(sim_run.get("floor") or 0)
    current_coord = {"row": floor, "col": 0}
    points: list[dict[str, Any]] = []
    for opt in map_state.get("next_options") or []:
        if not isinstance(opt, dict):
            continue
        points.append({
            "coord": {"row": int(opt.get("row") or 0), "col": int(opt.get("col") or 0)},
            "point_type": str(opt.get("point_type") or "Monster").title(),
            "is_available": True,
        })
    return {
        "current_coord": current_coord,
        "dimensions": {"rows": 15, "cols": 7},
        "is_blocked_by_combat": False,
        "is_interactive_surface": bool(points),
        "is_open": bool(points),
        "is_open_raw": bool(points),
        "is_travel_enabled": bool(points),
        "is_travel_enabled_raw": bool(points),
        "is_traveling": False,
        "points": points,
    }


def _translate_run_block(sim_run: dict[str, Any], state_type: str, game_over: dict[str, Any]) -> dict[str, Any]:
    floor = int(sim_run.get("floor") or 0)
    act = int(sim_run.get("act") or 1)
    room_type = str(sim_run.get("room_type") or "").title() or "Monster"
    is_game_over = state_type in {"game_over", "victory"}
    # Map act 1/2/3 -> STS2 act model ids. Obs encoder's _parse_act reads the
    # trailing digit, so "ACT.UNDERDOCKS" would give 0. We need the real
    # ACT.<name> id matching the specific act played. Sim doesn't directly
    # expose act model id (just int index), but we can construct the
    # canonical form from act index + the character.
    # Act names per src/Core/Models/Acts/: UNDERDOCKS (1), HIVE (2), GLORY (3).
    act_id_canonical = {
        1: "ACT.UNDERDOCKS",
        2: "ACT.HIVE",
        3: "ACT.GLORY",
    }.get(act, "ACT.UNDERDOCKS")
    return {
        "has_run": True,
        "is_game_over": is_game_over,
        # Gate field: obs encoder reads run.active to know if we're mid-run.
        # True whenever we're past character-select and before game_over.
        # Sim doesn't expose this directly — infer from state_type.
        "active": state_type not in {"", "menu", "game_over"},
        # Obs encoder reads act_id via _parse_act which pulls the trailing
        # digit. For ACT.UNDERDOCKS/HIVE/GLORY that isn't a digit, so the
        # feature was always 0 on sim. Emit a synthesized id that ends in
        # the act number so _parse_act can extract it.
        "act_id": f"ACT.{act}",
        "current_location": f"act {act} coord (0, {floor})",
        "current_act_index": act - 1,  # bridge was 0-indexed
        "ascension_level": int(sim_run.get("ascension_level") or 0),
        # Real bridge emits all three; six downstream readers (obs encoder,
        # run_memory, combat_memory, objective_heads, env_v2 transition_state,
        # combat_env transition_state) key off bare ``floor`` and were reading
        # None from sim, zero-filling the /48-normalized floor feature and the
        # objective head's floor input.
        "floor": floor,
        "act_floor": floor,
        "total_floor": floor,
        "act": {
            "id": act_id_canonical,
            "title": "Underdocks" if act == 1 else ("Hive" if act == 2 else "Glory"),
            "description": "",
            "kind": "Underdocks" if act == 1 else ("Hive" if act == 2 else "Glory"),
        },
        "acts": [],
        "modifiers": [],
        "current_map_coord": {"row": floor, "col": 0},
        "current_map_point": {
            "coord": {"row": floor, "col": 0},
            "point_type": room_type,
        },
        "current_room": {
            "room_type": room_type,
            "model_id": str(sim_run.get("room_model_id") or ""),
            "is_pre_finished": False,
            "is_victory_room": False,
        },
        "player_count": 1,
    }


_EVENT_DELTA_CARD_COUNT_TOKENS = {
    "a": 1, "an": 1, "one": 1, "一": 1,
    "two": 2, "两": 2,
    "three": 3, "三": 3,
}


def _event_parse_card_count_token(raw: str) -> int:
    try:
        return int(raw)
    except (TypeError, ValueError):
        return _EVENT_DELTA_CARD_COUNT_TOKENS.get(raw.lower(), 1)


def _event_sum_first_match(
    lower: str, original: str,
    english_patterns: list[str], chinese_patterns: list[str],
) -> int:
    import re as _re
    for pattern in english_patterns:
        m = _re.search(pattern, lower, _re.IGNORECASE)
        if m and m.group(1).isdigit():
            return int(m.group(1))
    for pattern in chinese_patterns:
        m = _re.search(pattern, original)
        if m and m.group(1).isdigit():
            return int(m.group(1))
    return 0


def _event_count_card_op(
    lower: str, original: str,
    english_patterns: list[str], chinese_patterns: list[str],
) -> int:
    import re as _re
    for pattern in english_patterns:
        m = _re.search(pattern, lower, _re.IGNORECASE)
        if m:
            return _event_parse_card_count_token(m.group(1))
    for pattern in chinese_patterns:
        m = _re.search(pattern, original)
        if m:
            return _event_parse_card_count_token(m.group(1))
    return 0


def _extract_event_option_effect_deltas(title: str | None, description: str | None) -> dict[str, Any]:
    """Port of bridge BridgeGameApi.EnvHelpers.ExtractEventOptionEffectDeltas
    (EN + ZH regex patterns). Parses event option text into 17 structured
    signal fields the obs encoder consumes for event-choice reasoning.

    Missing patterns degrade to 0 rather than lying. Exact parity with
    live bridge's parser for the 17 keys emitted under effect_deltas.
    """
    import re as _re
    deltas: dict[str, Any] = {
        "hp_delta": 0,
        "max_hp_delta": 0,
        "gold_delta": 0,
        "heal_full": False,
        "card_add_count": 0,
        "card_add_attack": False,
        "card_add_skill": False,
        "card_add_power": False,
        "card_add_curse": False,
        "card_add_status": False,
        "card_remove_count": 0,
        "card_transform_count": 0,
        "card_upgrade_count": 0,
        "card_duplicate_count": 0,
        "relic_gain": False,
        "potion_gain": False,
        "enter_combat": False,
    }
    if not (title or description):
        return deltas
    combined = " \n ".join(s for s in (title or "", description or "") if s)
    lower = combined.lower()
    original = combined

    # HP delta (signed)
    hp_lose = _event_sum_first_match(lower, original,
        [r"lose\s*(\d+)\s*hp", r"take\s*(\d+)\s*damage", r"you\s*take\s*(\d+)",
         r"suffer\s*(\d+)\s*damage", r"receive\s*(\d+)\s*damage"],
        [r"失去(\d+)点?(?:生命|hp)", r"受到(\d+)点?伤害", r"扣除?(\d+)点?(?:生命|hp)"])
    hp_gain = _event_sum_first_match(lower, original,
        [r"gain\s*(\d+)\s*hp", r"heal\s*(\d+)\s*hp?", r"restore\s*(\d+)\s*hp",
         r"recover\s*(\d+)\s*hp"],
        [r"(?:回复|恢复|治疗)(\d+)点?(?:生命|hp)", r"获得(\d+)点?(?:生命|hp)"])
    deltas["hp_delta"] = hp_gain - hp_lose
    if (_re.search(r"\b(heal(ed)?\s*(to\s*)?full|fully\s*heal|restore\s*all\s*hp)\b", lower)
            or _re.search(r"(回满|满血|回复全部生命|治疗至满)", original)):
        deltas["heal_full"] = True

    # Max HP delta (signed)
    max_gain = _event_sum_first_match(lower, original,
        [r"max\s*hp\s*\+\s*(\d+)", r"gain\s*(\d+)\s*max\s*hp",
         r"(\d+)\s*max\s*hp", r"increase\s*max\s*hp\s*by\s*(\d+)"],
        [r"最大生命(?:增加|提高|提升|上升)?\+?(\d+)", r"max\s*hp\s*\+?(\d+)"])
    max_lose = _event_sum_first_match(lower, original,
        [r"max\s*hp\s*-\s*(\d+)", r"lose\s*(\d+)\s*max\s*hp",
         r"decrease\s*max\s*hp\s*by\s*(\d+)"],
        [r"最大生命(?:减少|降低|下降)(\d+)", r"失去(\d+)点?最大生命"])
    deltas["max_hp_delta"] = max_gain - max_lose

    # Gold
    gold_gain = _event_sum_first_match(lower, original,
        [r"gain\s*(\d+)\s*gold", r"receive\s*(\d+)\s*gold",
         r"(\d+)\s*gold", r"obtain\s*(\d+)\s*gold"],
        [r"获得(\d+)点?金币", r"(\d+)点?金币"])
    gold_lose = _event_sum_first_match(lower, original,
        [r"lose\s*(\d+)\s*gold", r"pay\s*(\d+)\s*gold", r"spend\s*(\d+)\s*gold"],
        [r"失去(\d+)点?金币", r"支付(\d+)点?金币", r"花费(\d+)点?金币"])
    deltas["gold_delta"] = gold_gain - gold_lose

    # Card-add count + types
    attack = bool(_re.search(r"\battack\b", lower)) or ("攻击" in original)
    skill = bool(_re.search(r"\bskill\b", lower)) or ("技能" in original)
    power = bool(_re.search(r"\bpower\b", lower)) or ("能力" in original)
    curse = bool(_re.search(r"\bcurse\b", lower)) or ("诅咒" in original)
    status = bool(_re.search(r"\bstatus\b", lower)) or ("状态" in original)
    deltas["card_add_attack"] = attack
    deltas["card_add_skill"] = skill
    deltas["card_add_power"] = power
    deltas["card_add_curse"] = curse
    deltas["card_add_status"] = status
    # Count of card mentions (simplified): 1 if any type mentioned else 0;
    # boost to explicit number if "add N cards" pattern matches.
    add_count_explicit = _event_count_card_op(lower, original,
        [r"(?:add|gain|obtain|receive)\s*(a|an|one|\d+)\s*cards?",
         r"(?:add|gain|obtain|receive)\s*(a|an|one|\d+)\s*(?:attack|skill|power|curse|status)"],
        [r"加入(一|两|三|\d+)张", r"获得(一|两|三|\d+)张"])
    if add_count_explicit:
        deltas["card_add_count"] = add_count_explicit
    elif any([attack, skill, power, curse, status]):
        deltas["card_add_count"] = 1

    # Card ops
    deltas["card_remove_count"] = _event_count_card_op(lower, original,
        [r"remove\s*(a|an|one|\d+)\s*cards?", r"purge\s*(a|an|\d+)\s*cards?"],
        [r"移除(一|两|三|\d+)张", r"删除(一|两|三|\d+)张"])
    deltas["card_transform_count"] = _event_count_card_op(lower, original,
        [r"transform\s*(a|an|one|two|\d+)\s*cards?"],
        [r"变化(一|两|三|\d+)张", r"变形(一|两|三|\d+)张"])
    deltas["card_upgrade_count"] = _event_count_card_op(lower, original,
        [r"upgrade\s*(a|an|one|\d+)\s*cards?", r"smith\s*(a|an|\d+)\s*cards?"],
        [r"升级(一|两|三|\d+)张", r"锻造(一|两|三|\d+)张"])
    deltas["card_duplicate_count"] = _event_count_card_op(lower, original,
        [r"duplicate\s*(a|an|one|\d+)\s*cards?", r"copy\s*(a|an|\d+)\s*cards?"],
        [r"复制(一|两|三|\d+)张"])

    # Relic / potion gain
    if (_re.search(r"\b(gain|obtain|receive|get)\s+(a|an|one|\d+)?\s*relic\b", lower)
            or _re.search(r"获得.{0,6}遗物", original)):
        deltas["relic_gain"] = True
    if (_re.search(r"\b(gain|obtain|receive|get)\s+(a|an|one|\d+)?\s*potion\b", lower)
            or _re.search(r"获得.{0,6}药水", original)):
        deltas["potion_gain"] = True

    # Combat entry
    if (_re.search(r"\b(fight|enter\s*combat|start\s*combat|begin\s*battle)\b", lower)
            or _re.search(r"(战斗|戰鬥|进入战斗|進入戰鬥|开始战斗|開始戰鬥|遭遇敌人|遭遇敵人|我能打|打两个|打兩個)", original)):
        deltas["enter_combat"] = True

    return deltas


def _translate_event_options(event: dict[str, Any]) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for opt in event.get("options") or []:
        if not isinstance(opt, dict):
            continue
        text = str(opt.get("text") or "")
        # Sim emits only `text` on option; split into title/description is
        # approximate (title = first line if any). Parser ok with whole text
        # in description.
        title = text.split("\n", 1)[0] if text else ""
        out.append({
            "index": int(opt.get("index") or 0),
            "label": text,
            "description": text,
            "is_enabled": not bool(opt.get("is_locked", False)),
            "is_chosen": bool(opt.get("is_chosen", False)),
            "is_proceed": bool(opt.get("is_proceed", False)),
            "effect_deltas": _extract_event_option_effect_deltas(title, text),
        })
    return out


def _translate_rewards_block(
    rewards: dict[str, Any],
    card_reward: dict[str, Any],
    treasure: dict[str, Any],
    relic_select: dict[str, Any],
) -> dict[str, Any]:
    items: list[dict[str, Any]] = []
    for item in rewards.get("items") or []:
        if not isinstance(item, dict):
            continue
        items.append({
            "index": int(item.get("index") or 0),
            "reward": {
                "type": str(item.get("type") or ""),
                "label": str(item.get("label") or ""),
                "reward_key": str(item.get("reward_key") or ""),
                "claimable": bool(item.get("claimable", True)),
            },
        })
    visible = bool(items or card_reward or treasure or relic_select)
    return {
        "visible": visible,
        "terminal_proceed_visible": bool(rewards.get("can_proceed", False)),
        "rewards": items,
    }


def _translate_rest_site_block(rest: dict[str, Any]) -> dict[str, Any]:
    options: list[dict[str, Any]] = []
    for opt in rest.get("options") or []:
        if not isinstance(opt, dict):
            continue
        options.append({
            "index": int(opt.get("index") or 0),
            "id": str(opt.get("id") or ""),
            "label": str(opt.get("name") or opt.get("id") or ""),
            "description": str(opt.get("description") or ""),
            "is_enabled": bool(opt.get("is_enabled", True)),
        })
    return {"visible": bool(options), "options": options, "can_proceed": bool(rest.get("can_proceed", False))}


def _translate_shop_block(shop: dict[str, Any]) -> dict[str, Any]:
    items: list[dict[str, Any]] = []
    for item in shop.get("items") or []:
        if not isinstance(item, dict):
            continue
        items.append({
            "index": int(item.get("index") or 0),
            "category": str(item.get("category") or ""),
            "cost": int(item.get("cost") or 0),
            "can_afford": bool(item.get("can_afford", False)),
            "is_stocked": bool(item.get("is_stocked", True)),
            "on_sale": bool(item.get("on_sale", False)),
            "name": str(item.get("name") or ""),
            "description": str(item.get("description") or ""),
            "card_id": str(item.get("card_id") or ""),
            "relic_id": str(item.get("relic_id") or ""),
            "potion_id": str(item.get("potion_id") or ""),
        })
    return {"visible": bool(items), "is_open": bool(shop.get("is_open", False)),
            "can_proceed": bool(shop.get("can_proceed", False)), "items": items}


def _translate_card_reward_sel_block(card_reward: dict[str, Any]) -> dict[str, Any]:
    choices = [_translate_card(c, pile="Reward") for c in (card_reward.get("cards") or [])]
    return {
        "visible": bool(choices),
        "can_skip": bool(card_reward.get("can_skip", True)),
        "choices": choices,
    }


def _translate_card_sel_block(
    card_select: dict[str, Any],
    hand_select: dict[str, Any],
    combat_card_sel: dict[str, Any] | None,
) -> dict[str, Any]:
    src = card_select or hand_select or combat_card_sel or {}
    if not src:
        return {"visible": False, "choices": []}
    cards = src.get("cards") or src.get("selectable_cards") or []
    return {
        "visible": True,
        "prompt": str(src.get("prompt") or ""),
        "min_select": int(src.get("min_select") or 0),
        "max_select": int(src.get("max_select") or 1),
        "can_confirm": bool(src.get("can_confirm", False)),
        "can_cancel": bool(src.get("can_cancel", False)),
        "choices": [_translate_card(c, pile="Select") for c in cards],
        "selected": [_translate_card(c, pile="Select") for c in (src.get("selected_cards") or [])],
    }


# ---------------------------------------------------------------------------
# Legal-action translation (the rich bit)
# ---------------------------------------------------------------------------

def _translate_legal_actions(
    sim_legal_actions: list[Any],
    *,
    sim_player: dict[str, Any],
    battle: dict[str, Any],
    map_state: dict[str, Any],
    event: dict[str, Any],
    rest_site: dict[str, Any],
    shop: dict[str, Any],
    card_reward: dict[str, Any],
    card_select: dict[str, Any],
    treasure: dict[str, Any],
) -> list[dict[str, Any]]:
    """Produce bridge-rich action entries. Each entry:
      - carries ``_sim_raw`` so HeadlessSimBridgeClient.step can echo it back
      - adds ``card`` / ``target_combat_id`` / ``route_summary`` / etc.
        where the obs encoder and env reward-shaping expect them
    """
    hand_by_index: dict[int, dict[str, Any]] = {}
    for card in sim_player.get("hand") or []:
        if isinstance(card, dict):
            idx = card.get("index")
            if isinstance(idx, int):
                hand_by_index[idx] = card

    potions_by_slot: dict[int, dict[str, Any]] = {}
    for potion in sim_player.get("potions") or []:
        if isinstance(potion, dict):
            slot = potion.get("slot")
            if isinstance(slot, int):
                potions_by_slot[slot] = potion

    event_options_by_index: dict[int, dict[str, Any]] = {
        int(opt.get("index") or i): opt
        for i, opt in enumerate(event.get("options") or [])
        if isinstance(opt, dict)
    }
    rest_options_by_index: dict[int, dict[str, Any]] = {
        int(opt.get("index") or i): opt
        for i, opt in enumerate(rest_site.get("options") or [])
        if isinstance(opt, dict)
    }
    shop_items_by_index: dict[int, dict[str, Any]] = {
        int(item.get("index") or i): item
        for i, item in enumerate(shop.get("items") or [])
        if isinstance(item, dict)
    }
    card_reward_by_index: dict[int, dict[str, Any]] = {
        int(c.get("index") or i): c
        for i, c in enumerate(card_reward.get("cards") or [])
        if isinstance(c, dict)
    }
    card_select_by_index: dict[int, dict[str, Any]] = {
        int(c.get("index") or i): c
        for i, c in enumerate(card_select.get("cards") or [])
        if isinstance(c, dict)
    }
    map_options_by_index: dict[int, dict[str, Any]] = {
        int(opt.get("index") or i): opt
        for i, opt in enumerate(map_state.get("next_options") or [])
        if isinstance(opt, dict)
    }
    # Pre-index full map nodes (exposed by sim after the April 2026 DTO
    # patch; missing in older sim builds). Used by route_summary BFS below.
    map_nodes_by_coord: dict[tuple[int, int], dict[str, Any]] = {}
    for node in map_state.get("nodes") or []:
        if isinstance(node, dict):
            coord = (int(node.get("col") or 0), int(node.get("row") or 0))
            map_nodes_by_coord[coord] = node
    map_parent_child_count = len(map_options_by_index)
    treasure_relics_by_index: dict[int, dict[str, Any]] = {
        int(r.get("index") or i): r
        for i, r in enumerate(treasure.get("relics") or [])
        if isinstance(r, dict)
    }

    out: list[dict[str, Any]] = []
    for slot_idx, action in enumerate(sim_legal_actions):
        if not isinstance(action, dict):
            continue
        kind = str(action.get("action") or "")
        bridge_kind = sim_kind_to_bridge_kind(kind)
        synthetic_id = _synthesize_action_id(kind, action)

        # Action entries mirror BridgeGameApi.EnvHelpers.cs:63-191's shape:
        #   - ``target`` is a nested dict {combat_id,name,side}; readers in
        #     observation_v3._match_target_enemy (line 3122), aux_targets
        #     (387), observation_common (1297, 1409) expect isinstance(target, dict).
        #   - Shop emits ``item`` (nested card/potion/relic + cost), not ``shop_item``.
        #   - Rest site emits ``option``, not ``rest_site_option``.
        #   - Event option fields are FLAT on the entry (index, title, proceed, ...)
        #     not nested under ``event_option``.
        #   - Treasure emits ``relic`` (direct compact), not ``treasure_relic``.
        #   - Every entry gets a ``canonical_text`` for the text encoder.
        entry: dict[str, Any] = {
            "idx": slot_idx,
            "action_id": synthetic_id,
            "action_index": slot_idx,  # legacy alias
            "kind": bridge_kind,
            "label": str(action.get("label") or kind),
            "is_enabled": bool(action.get("is_enabled", True)),
            "_sim_raw": action,
        }

        if kind == "play_card":
            card_idx = action.get("card_index")
            card_dict: dict[str, Any] | None = None
            if isinstance(card_idx, int) and card_idx in hand_by_index:
                card_dict = _translate_card(hand_by_index[card_idx], pile="Hand")
                entry["card"] = card_dict
                entry["card_ref"] = card_dict.get("id", "")
                entry["hand_index"] = card_idx
            tid = action.get("target_id")
            target_name = ""
            if tid is not None:
                for e in battle.get("enemies") or []:
                    if isinstance(e, dict) and int(e.get("combat_id") or 0) == int(tid):
                        target_name = str(e.get("name") or "")
                        break
            # Always emit ``target`` as a dict (even if tid is None) so
            # readers doing ``isinstance(action.get("target"), dict)`` fall
            # into the happy path.
            entry["target"] = {
                "combat_id": int(tid) if tid is not None else None,
                "name": target_name,
                "side": "enemy" if tid is not None else None,
            }
            # Legacy flat key kept for any path still reading it.
            if tid is not None:
                entry["target_combat_id"] = int(tid)
        elif kind in {"use_potion", "discard_potion"}:
            slot = action.get("slot")
            if isinstance(slot, int) and slot in potions_by_slot:
                entry["potion"] = _translate_potion(potions_by_slot[slot])
                entry["slot"] = slot
                entry["potion_slot"] = slot  # legacy alias
                entry["slot_index"] = slot   # obs encoder reads this name
            tid = action.get("target_id")
            target_name = ""
            if tid is not None:
                for e in battle.get("enemies") or []:
                    if isinstance(e, dict) and int(e.get("combat_id") or 0) == int(tid):
                        target_name = str(e.get("name") or "")
                        break
            entry["target"] = {
                "combat_id": int(tid) if tid is not None else None,
                "name": target_name,
            }
            if tid is not None:
                entry["target_combat_id"] = int(tid)
        elif kind == "choose_map_node":
            midx = action.get("index")
            if isinstance(midx, int) and midx in map_options_by_index:
                opt = map_options_by_index[midx]
                child_coord = (int(opt.get("col") or 0), int(opt.get("row") or 0))
                point_type = str(opt.get("point_type") or "Monster")
                # Flat fields on entry — bridge EnvHelpers.cs:126-132 shape.
                entry["coord"] = {"col": child_coord[0], "row": child_coord[1]}
                entry["point_type"] = point_type.title()
                entry["point_type_norm"] = _canonical_point_type(point_type)
                # Legacy nested key retained for any older reader.
                entry["map_node"] = {
                    "coord": {"row": child_coord[1], "col": child_coord[0]},
                    "point_type": point_type.title(),
                }
                # Route subtree stats. Empty when sim hasn't been rebuilt
                # with the map.nodes DTO patch (graceful fallback).
                if map_nodes_by_coord:
                    summary = _build_route_summary(
                        child_coord, map_nodes_by_coord, map_parent_child_count,
                    )
                    if summary:
                        entry["route_summary"] = summary
                        # Obs encoder consumes per-node tree under the
                        # action's own `route_nodes` list (parallel to
                        # `route_summary`). Live bridge flattens summary
                        # nodes to action-level for the same reason.
                        nodes_list = summary.get("nodes") or []
                        if nodes_list:
                            entry["route_nodes"] = list(nodes_list)
        elif kind == "choose_event_option":
            eidx = action.get("index")
            if isinstance(eidx, int) and eidx in event_options_by_index:
                opt = event_options_by_index[eidx]
                # Flat fields mirror bridge EnvHelpers.cs:112-123.
                text = str(opt.get("text") or "")
                title = text.split("\n", 1)[0] if text else ""
                deltas = _extract_event_option_effect_deltas(title, text)
                entry["index"] = eidx
                entry["title"] = text
                entry["option_type"] = str(opt.get("option_type") or "")
                entry["proceed"] = bool(opt.get("is_proceed", False))
                entry["effect_deltas"] = deltas
                # Legacy nested key kept for back-compat.
                entry["event_option"] = {
                    "index": eidx,
                    "label": text,
                    "is_locked": bool(opt.get("is_locked", False)),
                    "is_proceed": bool(opt.get("is_proceed", False)),
                    "effect_deltas": deltas,
                }
        elif kind == "choose_rest_option":
            ridx = action.get("index")
            if isinstance(ridx, int) and ridx in rest_options_by_index:
                opt = rest_options_by_index[ridx]
                option_payload = {
                    "option_id": str(opt.get("id") or ""),
                    "option_type": str(opt.get("id") or ""),
                    "title": str(opt.get("name") or ""),
                    "description": str(opt.get("description") or ""),
                    "enabled": bool(opt.get("is_enabled", True)),
                }
                entry["option"] = option_payload
                entry["rest_site_option"] = option_payload  # legacy alias
        elif kind == "shop_purchase":
            sidx = action.get("index")
            if isinstance(sidx, int) and sidx in shop_items_by_index:
                item = shop_items_by_index[sidx]
                item_payload: dict[str, Any] = {
                    "kind": str(item.get("category") or ""),
                    "title": str(item.get("name") or ""),
                    "cost": int(item.get("cost") or 0),
                    "affordable": bool(item.get("can_afford", False)),
                }
                # Nested compact card/potion/relic when that's what the item is.
                if item.get("card_id"):
                    item_payload["card"] = _translate_card(
                        {"id": item.get("card_id"), "name": item.get("name"), "cost": 0},
                        pile="Shop",
                    )
                if item.get("potion_id"):
                    item_payload["potion"] = _translate_potion(
                        {"id": item.get("potion_id"), "name": item.get("name")}
                    )
                if item.get("relic_id"):
                    item_payload["relic"] = _translate_relic(
                        {"id": item.get("relic_id"), "name": item.get("name")}
                    )
                entry["item"] = item_payload
                entry["shop_action"] = "buy"
                entry["shop_item"] = item_payload  # legacy alias
        elif kind == "choose_card_reward":
            cidx = action.get("index")
            if isinstance(cidx, int) and cidx in card_reward_by_index:
                entry["card"] = _translate_card(card_reward_by_index[cidx], pile="Reward")
                entry["selection"] = "pick"
                entry["index"] = cidx
        elif kind in {"select_card", "select_hand_card", "combat_select_card"}:
            # Combat variants carry ``card_index`` on the action (mapped
            # from sim's hand index), while the non-combat variants use
            # ``index``. Take whichever is present.
            cidx = action.get("index")
            if cidx is None:
                cidx = action.get("card_index")
            # Carry the source block's prompt onto each action entry so obs
            # encoder's action.get("selection_prompt") resolves (previously
            # always None on sim). Live bridge's BridgeGameApi.EnvHelpers.cs
            # :143 does the same flattening.
            selection_prompt = str(card_select.get("prompt") or "") if card_select else ""
            if isinstance(cidx, int) and cidx in card_select_by_index:
                entry["card"] = _translate_card(card_select_by_index[cidx], pile="Select")
                entry["selection"] = "pick"
                entry["index"] = cidx
                entry["selection_semantics"] = str(action.get("selection_semantics") or "")
                entry["selection_prompt"] = selection_prompt
            elif isinstance(cidx, int):
                # Combat hand-selection — the card isn't in card_select_by_index
                # (that dict is seeded from the non-combat ``card_select`` block);
                # fall back to the hand slot so the token encoder still sees
                # something meaningful.
                hand_card = hand_by_index.get(cidx)
                if isinstance(hand_card, dict):
                    entry["card"] = _translate_card(hand_card, pile="Hand")
                    entry["selection"] = "pick"
                    entry["index"] = cidx
                    entry["selection_semantics"] = str(action.get("selection_semantics") or "")
                    entry["selection_prompt"] = selection_prompt
        elif kind == "claim_treasure":
            ridx = action.get("index")
            if isinstance(ridx, int) and ridx in treasure_relics_by_index:
                relic_payload = _translate_relic(treasure_relics_by_index[ridx])
                entry["relic"] = relic_payload
                entry["index"] = ridx
                entry["treasure_relic"] = relic_payload  # legacy alias

        # Every entry carries a canonical_text, mirroring bridge behavior
        # (EnvHelpers.cs:188). Minimal synth from kind/label/card/item
        # so the action-text encoder has something non-empty to embed.
        entry["canonical_text"] = _build_action_canonical_text(entry, kind)

        out.append(entry)

    # ------------------------------------------------------------------
    # Card-selection shaping:
    #   (a) filter out select_card/combat_select_card entries whose card
    #       has already been selected — without this, a policy whose
    #       argmax has collapsed onto idx=0 will keep re-picking the same
    #       card (sim treats re-select-of-selected-card as no-op), never
    #       accumulating enough picks to reach CanConfirm=true. This was
    #       responsible for ~42% of training episodes getting stuck in
    #       card_selection (see reset_events.jsonl stuck_phase stats).
    #   (b) hoist the confirm action to index 0 when it's emitted. The
    #       policy's action prior is strongly biased toward low indices
    #       early in training; putting confirm at idx=0 means "when
    #       confirm is available, default to confirming" rather than
    #       "keep poking the selection list".
    # Both transforms are safe no-ops when no card_selection actions
    # are present — the loop below early-exits.
    # ------------------------------------------------------------------
    selected_indices, max_select = _collect_selection_state(
        card_select=card_select,
        combat_card_selection=battle.get("card_selection") if battle else None,
    )
    out = _shape_card_selection_actions(out, selected_indices, max_select)
    return out


_SELECT_CARD_SIM_ACTIONS = frozenset({"select_card", "select_hand_card", "combat_select_card"})
_CONFIRM_SIM_ACTIONS = frozenset({"confirm_selection", "combat_confirm_selection"})


def _collect_selection_state(
    *,
    card_select: dict[str, Any],
    combat_card_selection: Any,
) -> tuple[set[int], int]:
    """Gather (``selected_indices``, ``max_select``) from whichever sim
    state block is populated for the active selection screen.

    ``selected_indices`` = ChoiceIndex values of cards already committed
    to the pending selection. ``max_select`` = the hard cap the sim
    enforces (typically 1 for TO_UPGRADE, 2 for TO_REMOVE). We return 0
    when no selection state is present so callers know the cap is
    unknown and should leave the full select_card list intact.
    """
    selected: set[int] = set()
    max_select = 0
    for source in (card_select, combat_card_selection if isinstance(combat_card_selection, dict) else None):
        if not isinstance(source, dict):
            continue
        for card in source.get("selected_cards") or []:
            if not isinstance(card, dict):
                continue
            idx = card.get("index")
            if idx is None:
                idx = card.get("choice_index")
            if idx is None:
                idx = card.get("card_index")
            if isinstance(idx, int):
                selected.add(idx)
        ms = source.get("max_select")
        if isinstance(ms, int) and ms > max_select:
            max_select = ms
    return selected, max_select


def _shape_card_selection_actions(
    entries: list[dict[str, Any]],
    selected_indices: set[int],
    max_select: int,
) -> list[dict[str, Any]]:
    """Drop no-op select entries and hoist confirm to slot 0.

    Three defensive passes against policy-collapse loops in
    card_selection screens:

    1. Drop select_card entries whose target card is already in
       ``selected_indices`` (sim treats duplicate-select-of-selected as
       a no-op → policy argmax-collapsed onto such an idx loops forever).
    2. **When ``len(selected_indices) >= max_select > 0``: drop ALL
       select_card entries regardless of target.** At cap, the only
       semantically-valid next action is ``confirm_selection`` (or
       ``cancel_selection`` if the screen allows it). Without this pass,
       a biased argmax that scores some select_card above confirm can
       endlessly swap the currently-selected card (sim accepts swaps
       when at cap), fingerprint-identical but making no progress — this
       was the source of 21/21 NEOW TO_UPGRADE watchdog false-stucks
       observed at smoke time.
    3. Hoist confirm/combat_confirm_selection to slot 0 so argmax-biased
       policies naturally pick it when available.

    Rewrites ``idx`` / ``action_index`` on surviving entries so they are
    contiguous — downstream code (observation_common MAX_ACTIONS masking
    and env_v2 normalize_action) indexes entries positionally, so gaps
    would break action dispatch.
    """
    at_cap = max_select > 0 and len(selected_indices) >= max_select
    filtered: list[dict[str, Any]] = []
    confirm_entries: list[dict[str, Any]] = []
    for entry in entries:
        raw = entry.get("_sim_raw") if isinstance(entry.get("_sim_raw"), dict) else {}
        sim_action = str(raw.get("action") or "")
        if sim_action in _SELECT_CARD_SIM_ACTIONS:
            if at_cap:
                # Selection is full — no select_card action is a valid
                # forward move. The only way out is confirm (or cancel,
                # which is preserved via the else branch since cancel
                # isn't in _SELECT_CARD_SIM_ACTIONS).
                continue
            card_idx = raw.get("index")
            if card_idx is None:
                card_idx = raw.get("card_index")
            if isinstance(card_idx, int) and card_idx in selected_indices:
                # Already selected — dropping prevents the idempotent
                # "pick already-selected" loop even below cap.
                continue
            filtered.append(entry)
        elif sim_action in _CONFIRM_SIM_ACTIONS:
            confirm_entries.append(entry)
        else:
            filtered.append(entry)
    if confirm_entries:
        reordered = confirm_entries + filtered
    else:
        reordered = filtered
    for new_slot, entry in enumerate(reordered):
        entry["idx"] = new_slot
        entry["action_index"] = new_slot
    return reordered


def _build_action_canonical_text(entry: dict[str, Any], sim_kind: str) -> str:
    """Compact canonical text for each action, loosely mirroring the
    bridge's BuildCanonicalActionText output format.
    """
    kind = str(entry.get("kind") or sim_kind)
    parts: list[str] = [f"动作｜{kind}"]
    card = entry.get("card") if isinstance(entry.get("card"), dict) else None
    if card:
        ct = card.get("canonical_text")
        if ct:
            parts.append(str(ct))
        else:
            parts.append(f"卡牌｜{card.get('title') or card.get('id', '')}")
    potion = entry.get("potion") if isinstance(entry.get("potion"), dict) else None
    if potion:
        parts.append(f"药水｜{potion.get('title') or potion.get('id', '')}")
    target = entry.get("target") if isinstance(entry.get("target"), dict) else None
    if target and target.get("name"):
        parts.append(f"目标：{target['name']}")
    if kind == "map":
        pt = entry.get("point_type")
        if pt:
            parts.append(f"节点：{pt}")
    if kind == "event_option":
        title = entry.get("title")
        if title:
            parts.append(f"选项：{title}")
    if kind == "shop":
        item = entry.get("item") if isinstance(entry.get("item"), dict) else None
        if item:
            parts.append(f"{item.get('title', '')}｜价格{item.get('cost', 0)}")
    return "｜".join(p for p in parts if p)


def _synthesize_action_id(kind: str, action: dict[str, Any]) -> str:
    """Produce a stable identifier for an action. Bridge code occasionally
    compares action_ids across ticks (e.g., during recovery paths) so we
    include enough discriminating fields.
    """
    parts = [f"sim:{kind}"]
    for field in ("index", "card_index", "slot", "target_id", "col", "row"):
        val = action.get(field)
        if val is not None:
            parts.append(f"{field}={val}")
    return ":".join(parts)
