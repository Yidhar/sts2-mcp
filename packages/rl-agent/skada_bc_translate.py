"""Translate BC samples from skada_bc_dataset.py into bridge-obs shape.

Produces (obs_dict, legal_actions, chosen_idx) suitable for feeding into
WorldTokenObservationEncoder + target for cross-entropy loss.

One translator per decision phase:
  map           → obs with map.points populated
  campfire      → obs with rest_site.options populated
  card_reward   → obs with card_reward_selection.choices populated
  relic_ancient → obs with event_options populated
  relic_relic   → obs with rewards.rewards populated

Usage as module:
    from skada_bc_translate import translate_bc_sample
    obs, legal_actions, chosen_idx = translate_bc_sample(sample)

Usage as CLI (validation):
    python skada_bc_translate.py  # pipes 5 samples per phase through encoder
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


# Character → starter max_hp (approximate; overridden by sample if tracker captured it)
_STARTER_MAX_HP: dict[str, int] = {
    "IRONCLAD": 80,
    "SILENT": 70,
    "DEFECT": 60,
    "REGENT": 65,
    "NECROBINDER": 70,
}

# Phase → screen / phase strings the obs schema expects
_PHASE_SCREEN = {
    "map": ("MAP", "map"),
    "campfire": ("REST_SITE", "actions"),
    "card_reward": ("REWARDS", "actions"),
    "relic_ancient": ("EVENT", "event"),
    "relic_relic": ("REWARDS", "actions"),
}


# ---------------------------------------------------------------------------
# Card / relic / potion helpers
# ---------------------------------------------------------------------------

def _build_card_dict(card_id: str, upgrade: int = 0, pile: str = "Deck") -> dict[str, Any]:
    """Minimal bridge-compatible card dict from content_registry metadata.

    Mirrors _translate_card in _sim_translate.py but sources fields from
    registry (since BC samples don't have sim_card payloads).
    """
    try:
        from content_registry import get_card_metadata  # noqa: PLC0415
        md = get_card_metadata(card_id) or {}
    except Exception:
        md = {}
    canonical_id = card_id if card_id.startswith("CARD.") else f"CARD.{card_id}"
    cost = md.get("energy_cost")
    try:
        cost_int = int(cost) if cost is not None else 1
    except (TypeError, ValueError):
        cost_int = 1
    card_type = str(md.get("type") or "Attack")
    title = str(md.get("title") or card_id)
    description = str(md.get("description") or "")
    target = str(md.get("target") or "None")

    # Mirror _sim_translate._card_effect_preview_from_registry — populate
    # effect_preview from semantic_signals so downstream _preview_metric
    # readers see non-zero values.
    _SIGNAL_TO_PREVIEW = {
        "damage": "damage", "block": "block", "draw": "draw", "heal": "heal",
        "weak": "weak", "vulnerable": "vulnerable", "frail": "frail",
        "strength": "strength", "strengthGain": "strength",
        "dexterity": "dexterity", "dexterityGain": "dexterity",
        "energy": "energy", "energyGain": "energy",
        "hpLoss": "hp_loss", "poison": "poison",
        "hits": "hits", "damage_per_hit": "damage_per_hit", "summon": "summon",
    }
    signals = md.get("semantic_signals") or {}
    preview: dict[str, float] = {}
    if isinstance(signals, dict):
        for src, dst in _SIGNAL_TO_PREVIEW.items():
            if src in signals:
                try:
                    preview[dst] = float(signals[src])
                except (TypeError, ValueError):
                    continue
        if "damage" in preview:
            preview.setdefault("total_damage", preview["damage"])
        if "block" in preview:
            preview.setdefault("total_block", preview["block"])

    canonical_text = (
        f"卡牌｜{title}｜{card_type}｜能量{cost_int}｜目标{target}"
        + (f"｜效果：{description[:60]}" if description else "")
    )
    return {
        "id": canonical_id,
        "upgrade_level": int(upgrade),
        "cost": cost_int,
        "target": target,
        "type": card_type,
        "canonical_text": canonical_text,
        # Legacy aliases for back-compat readers
        "current_upgrade_level": int(upgrade),
        "max_upgrade_level": 1,
        "title": title,
        "description": description,
        "rarity": str(md.get("rarity") or "Basic"),
        "target_type": target,
        "pile": pile,
        "is_playable": True,
        "canonical_energy_cost": cost_int,
        "resolved_energy_cost": cost_int,
        "costs_x": False,
        "keywords": [],
        "valid_target_ids": [],
        "effect_preview": preview,
    }


def _build_relic_dict(relic_id: str) -> dict[str, Any]:
    try:
        from content_registry import get_relic_metadata  # noqa: PLC0415
        md = get_relic_metadata(relic_id) or {}
    except Exception:
        md = {}
    canonical_id = relic_id if relic_id.startswith("RELIC.") else f"RELIC.{relic_id}"
    return {
        "id": canonical_id,
        "title": str(md.get("title") or relic_id),
        "description": str(md.get("description") or ""),
        "rarity": str(md.get("rarity") or "Common"),
        "counter": 0,
    }


def _build_player_block(sample: dict[str, Any]) -> dict[str, Any]:
    """Flat obs["player"] dict from BC sample state."""
    character = str(sample.get("character") or "IRONCLAD")
    hp = int(sample.get("hp") or 0)
    max_hp = int(sample.get("max_hp") or _STARTER_MAX_HP.get(character, 75))
    if hp <= 0:
        # First-floor hp=0 sentinel; start with full HP instead of zero-filling
        # player-survival token.
        hp = max_hp
    gold = int(sample.get("gold") or 0)
    deck_entries = sample.get("deck") or []
    deck_cards = []
    for entry in deck_entries if isinstance(deck_entries, list) else []:
        cid = entry.get("id")
        up = int(entry.get("upgrade") or 0)
        cnt = int(entry.get("count") or 1)
        for _ in range(cnt):
            deck_cards.append(_build_card_dict(cid, upgrade=up, pile="Deck"))
    relic_ids = sample.get("relics") or []
    relics = [_build_relic_dict(r) for r in relic_ids if r]
    return {
        "hp": hp,
        "current_hp": hp,
        "max_hp": max_hp,
        "block": 0,
        "gold": gold,
        "max_energy": 3,
        "character": character,
        "facing": None,
        "status": [],
        "powers": [],
        "deck_cards": deck_cards,
        "deck": len(deck_cards),
        "relics": relics,
        "potions": [],
        "creature": {
            "current_hp": hp, "max_hp": max_hp, "block": 0,
        },
    }


def _build_run_block(sample: dict[str, Any]) -> dict[str, Any]:
    floor = int(sample.get("floor") or 0)
    act = int(sample.get("act") or 0)
    return {
        "has_run": True,
        "is_game_over": False,
        "current_location": f"act {act + 1} coord (0, {floor})",
        "current_act_index": act,
        "ascension_level": int(sample.get("ascension") or 0),
        "floor": floor,
        "act_floor": floor,
        "total_floor": floor,
        "act": {"id": f"ACT.{act + 1}", "title": f"Act {act + 1}", "description": "", "kind": f"Act{act + 1}"},
        "acts": [],
        "modifiers": [],
        "current_map_coord": {"row": floor, "col": 0},
        "current_map_point": {"coord": {"row": floor, "col": 0}, "point_type": "Monster"},
        "current_room": {"room_type": "Monster", "model_id": "", "is_pre_finished": False, "is_victory_room": False},
        "player_count": 1,
    }


# ---------------------------------------------------------------------------
# Per-phase translators
# ---------------------------------------------------------------------------

def _translate_map(sample: dict) -> tuple[dict, list[dict], int]:
    coord_from = sample.get("coord_from") or [0, 0]
    options = sample.get("options") or []
    option_types = sample.get("option_types") or []
    legal_actions: list[dict] = []
    map_points: list[dict] = []
    for i, opt in enumerate(options):
        col, row = opt.split(",") if isinstance(opt, str) else (0, 0)
        col, row = int(col), int(row)
        pt_type = str((option_types[i] if i < len(option_types) else "M")).title()
        map_points.append({
            "coord": {"col": col, "row": row},
            "point_type": pt_type,
            "is_available": True,
        })
        legal_actions.append({
            "idx": i,
            "action_id": f"map:{col},{row}",
            "action_index": i,
            "kind": "map",
            "coord": {"col": col, "row": row},
            "point_type": pt_type,
            "point_type_norm": pt_type.lower(),
            "is_enabled": True,
            "label": f"map({col},{row})",
            "canonical_text": f"动作｜map｜节点：{pt_type}",
        })
    return _base_obs(sample, phase_tag="map",
                     map_block={
                         "current_coord": {"col": coord_from[0], "row": coord_from[1]},
                         "dimensions": {"rows": 15, "cols": 7},
                         "is_blocked_by_combat": False,
                         "is_interactive_surface": True,
                         "is_open": True,
                         "is_open_raw": True,
                         "is_travel_enabled": True,
                         "is_travel_enabled_raw": True,
                         "is_traveling": False,
                         "points": map_points,
                     }), legal_actions, int(sample.get("chosen_idx") or 0)


def _translate_campfire(sample: dict) -> tuple[dict, list[dict], int]:
    options = sample.get("options") or []
    legal_actions = []
    rest_opts = []
    for i, opt in enumerate(options):
        rest_opts.append({
            "index": i, "id": opt, "label": opt,
            "description": "", "is_enabled": True,
        })
        legal_actions.append({
            "idx": i,
            "action_id": f"rest:{opt}",
            "action_index": i,
            "kind": "rest_site",
            "option": {
                "option_id": opt, "option_type": opt,
                "title": opt, "description": "", "enabled": True,
            },
            "is_enabled": True,
            "label": opt,
            "canonical_text": f"动作｜rest_site｜{opt}",
        })
    return _base_obs(sample, phase_tag="campfire",
                     rest_site={"visible": True, "options": rest_opts, "can_proceed": False}), \
           legal_actions, int(sample.get("chosen_idx") or 0)


def _translate_card_reward(sample: dict) -> tuple[dict, list[dict], int]:
    options = sample.get("options") or []
    legal_actions = []
    choices = []
    for i, opt in enumerate(options):
        if opt == "SKIP":
            legal_actions.append({
                "idx": i,
                "action_id": "reward:skip",
                "action_index": i,
                "kind": "proceed",
                "skip": True,
                "source": "card_reward",
                "is_enabled": True,
                "label": "skip",
                "canonical_text": "动作｜proceed｜跳过",
            })
            continue
        card = _build_card_dict(opt, upgrade=0, pile="Reward")
        choices.append(card)
        legal_actions.append({
            "idx": i,
            "action_id": f"card_reward:{opt}",
            "action_index": i,
            "kind": "card_reward",
            "selection": "pick",
            "index": i,
            "card": card,
            "is_enabled": True,
            "label": f"pick {opt}",
            "canonical_text": f"动作｜card_reward｜{card['title']}",
        })
    return _base_obs(sample, phase_tag="card_reward",
                     card_reward_selection={
                         "visible": True, "can_skip": True, "choices": choices,
                     }), legal_actions, int(sample.get("chosen_idx") or 0)


def _translate_relic_ancient(sample: dict) -> tuple[dict, list[dict], int]:
    options = sample.get("options") or []
    legal_actions = []
    event_options = []
    for i, opt in enumerate(options):
        relic = _build_relic_dict(opt)
        event_options.append({
            "index": i, "label": relic["title"], "description": relic["description"],
            "is_enabled": True, "is_chosen": False, "is_proceed": False,
            "effect_deltas": {},
        })
        legal_actions.append({
            "idx": i,
            "action_id": f"event:{opt}",
            "action_index": i,
            "kind": "event_option",
            "index": i,
            "title": relic["title"],
            "option_type": "relic_pick",
            "proceed": False,
            "effect_deltas": {},
            "is_enabled": True,
            "label": f"take {opt}",
            "canonical_text": f"动作｜event_option｜{relic['title']}",
        })
    return _base_obs(sample, phase_tag="event",
                     event_options=event_options), \
           legal_actions, int(sample.get("chosen_idx") or 0)


def _translate_relic_relic(sample: dict) -> tuple[dict, list[dict], int]:
    options = sample.get("options") or []
    legal_actions = []
    rewards_items = []
    for i, opt in enumerate(options):
        if opt == "SKIP":
            legal_actions.append({
                "idx": i,
                "action_id": "reward:skip",
                "action_index": i,
                "kind": "proceed",
                "skip": True,
                "source": "relic_reward",
                "is_enabled": True,
                "label": "skip",
                "canonical_text": "动作｜proceed｜跳过",
            })
            continue
        relic = _build_relic_dict(opt)
        rewards_items.append({
            "index": i,
            "reward": {"type": "Relic", "label": relic["title"], "reward_key": opt, "claimable": True},
        })
        legal_actions.append({
            "idx": i,
            "action_id": f"reward:{opt}",
            "action_index": i,
            "kind": "reward",
            "reward": {"type": "Relic", "label": relic["title"], "reward_key": opt, "claimable": True},
            "relic": relic,
            "is_enabled": True,
            "label": f"take {opt}",
            "canonical_text": f"动作｜reward｜{relic['title']}",
        })
    return _base_obs(sample, phase_tag="actions",
                     rewards={"visible": True, "terminal_proceed_visible": True, "rewards": rewards_items}), \
           legal_actions, int(sample.get("chosen_idx") or 0)


# ---------------------------------------------------------------------------
# Base obs builder (common across phases)
# ---------------------------------------------------------------------------

def _base_obs(sample: dict, *, phase_tag: str, **phase_slots: Any) -> dict:
    screen, phase = _PHASE_SCREEN.get(sample.get("phase") or phase_tag, ("UNKNOWN", phase_tag))
    obs: dict[str, Any] = {
        "ok": True,
        "backend": "skada_bc",
        "captured_at_utc": "",
        "state_version": 0,
        "state_hash": "",
        "semantic_state_hash": "",
        "schema_version": "bc-v1",
        "bridge_version": "skada_bc",
        "screen": screen,
        "phase": phase,
        "player": _build_player_block(sample),
        "players": [{"index": 0, "net_id": 1, "character": {"id": f"CHARACTER.{sample.get('character', 'IRONCLAD')}", "title": sample.get("character", "IRONCLAD")}}],
        "combat": {"in_progress": False},
        "run": _build_run_block(sample),
        "event_options": [],
        "card_selection": {"visible": False, "choices": []},
        "card_reward_selection": {"visible": False, "choices": []},
        "character_selection": {"visible": False, "options": []},
        "run_mode_selection": {"visible": False},
        "deck_upgrade_selection": {"visible": False, "choices": []},
        "main_menu": {"visible": False},
        "rest_site": {"visible": False, "options": []},
        "shop": {"visible": False, "items": []},
        "crystal_sphere": {"visible": False},
        "automation": {"enabled": False},
        "rewards": {"visible": False, "rewards": []},
        "map": {"is_open": False, "points": []},
        "decision": {},
        "available_actions": [],  # caller overwrites with legal_actions
    }
    # Overlay phase-specific slots
    for k, v in phase_slots.items():
        obs[k] = v
    return obs


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------

_PHASE_TRANSLATORS = {
    "map": _translate_map,
    "campfire": _translate_campfire,
    "card_reward": _translate_card_reward,
    "relic_ancient": _translate_relic_ancient,
    "relic_relic": _translate_relic_relic,
}


def translate_bc_sample(sample: dict) -> tuple[dict, list[dict], int] | None:
    """Return (obs, legal_actions, chosen_idx) or None if phase unknown.

    obs["available_actions"] is populated from legal_actions so callers that
    only need the obs (e.g., encoder.encode) don't have to re-stitch.
    """
    fn = _PHASE_TRANSLATORS.get(sample.get("phase"))
    if fn is None:
        return None
    obs, legal_actions, chosen_idx = fn(sample)
    obs["available_actions"] = legal_actions
    return obs, legal_actions, chosen_idx


# ---------------------------------------------------------------------------
# CLI validation
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--samples", default="data/skada_bc/samples.jsonl")
    parser.add_argument("--per-phase", type=int, default=3,
                        help="how many samples per phase to validate through encoder")
    args = parser.parse_args()

    # Lazy-import encoder (heavy — sentence-transformers etc.)
    from sts2_env.observation_v3 import WorldTokenObservationEncoder
    encoder = WorldTokenObservationEncoder(use_text=False)

    seen_per_phase: dict[str, int] = {}
    errors: list[str] = []
    checked = 0
    with open(args.samples, "r", encoding="utf-8") as f:
        for line in f:
            sample = json.loads(line)
            phase = sample.get("phase")
            if phase not in _PHASE_TRANSLATORS:
                continue
            if seen_per_phase.get(phase, 0) >= args.per_phase:
                continue
            seen_per_phase[phase] = seen_per_phase.get(phase, 0) + 1
            try:
                result = translate_bc_sample(sample)
                if result is None:
                    errors.append(f"{phase}: translator returned None")
                    continue
                obs, legal_actions, chosen_idx = result
                if not (0 <= chosen_idx < len(legal_actions)):
                    errors.append(f"{phase}: chosen_idx {chosen_idx} out of range (len={len(legal_actions)})")
                encoded = encoder.encode(obs, legal_actions, planner_context={})
                if "world_tokens" not in encoded:
                    errors.append(f"{phase}: encoder missing world_tokens")
                if encoded["world_tokens"].shape[1] != 160:
                    errors.append(f"{phase}: token feat_dim mismatch: {encoded['world_tokens'].shape}")
                checked += 1
                print(f"OK [{phase}] run_id={sample.get('run_id')} floor={sample.get('floor')} "
                      f"options={len(legal_actions)} chosen={chosen_idx} "
                      f"world_tokens={encoded['world_tokens'].shape}")
            except Exception as e:
                errors.append(f"{phase}: {type(e).__name__}: {e}")
            if all(seen_per_phase.get(p, 0) >= args.per_phase for p in _PHASE_TRANSLATORS):
                break

    print()
    print(f"checked {checked} samples across {len(seen_per_phase)} phases")
    if errors:
        print(f"ERRORS ({len(errors)}):")
        for e in errors[:20]:
            print(f"  - {e}")
    else:
        print("all validations passed ✓")


if __name__ == "__main__":
    main()
