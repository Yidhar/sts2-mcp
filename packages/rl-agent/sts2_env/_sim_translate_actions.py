# ruff: noqa: RUF001
"""Pure legal-action translation and card-selection shaping."""

from __future__ import annotations

from typing import Any

from ._sim_translate_decisions import _extract_event_option_effect_deltas
from ._sim_translate_entities import _translate_card, _translate_potion, _translate_relic
from ._sim_translate_route import _build_route_summary, _canonical_point_type
from ._sim_translate_shared import sim_kind_to_bridge_kind


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
