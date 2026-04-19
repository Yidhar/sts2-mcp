"""Extract BC training samples from cleaned Skada SP runs.

Each output sample represents one human decision at a specific decision
point (map choice / campfire / card reward / relic pick), with enough
reconstructed context to run through our obs encoder later.

Output schema (JSONL):
    {
      "phase": "map" | "campfire" | "card_reward" | "relic",
      "character": "IRONCLAD",
      "ascension": 4,
      "run_id": 1114160,
      "floor": 12,
      "act": 1,
      "hp": 45, "max_hp": 80, "gold": 120,
      "deck": [{"id": "STRIKE_IRONCLAD", "count": 4, "upgrade": 0}, ...],
      "relics": ["BURNING_BLOOD", "BAG_OF_MARBLES", ...],
      # phase-specific fields:
      "options": [...],        # list of available choices (card ids / relic ids / coords)
      "chosen": <option_id or coord or "skip">,
      "chosen_idx": int,       # index into options (or -1 for skip)
    }

Phase 7.3 downstream will translate (deck, relics, hp, floor, options, ...)
into the bridge obs schema the encoder consumes.

Usage:
    python skada_bc_dataset.py [--limit N] [--out data/skada_bc/samples.jsonl]
"""
from __future__ import annotations

import argparse
import collections
import json
from pathlib import Path

SP_DIR = Path("data/skada_clean/sp")
DEFAULT_OUT = Path("data/skada_bc/samples.jsonl")

# Canonical STS2 starter decks. Validated against early-abandon runs where
# final_deck ≈ starter. Per-character curses (Ascension 10+ = ASCENDERS_BANE)
# are added separately based on run.ascension.
STARTER_DECKS: dict[str, list[tuple[str, int]]] = {
    "IRONCLAD":    [("STRIKE_IRONCLAD", 5),    ("DEFEND_IRONCLAD", 4),    ("BASH", 1)],
    "SILENT":      [("STRIKE_SILENT", 5),      ("DEFEND_SILENT", 5),      ("NEUTRALIZE", 1), ("SURVIVOR", 1)],
    "DEFECT":      [("STRIKE_DEFECT", 4),      ("DEFEND_DEFECT", 4),      ("ZAP", 1),        ("DUALCAST", 1)],
    "REGENT":      [("STRIKE_REGENT", 5),      ("DEFEND_REGENT", 4),      ("ROYAL_DECREE", 1)],
    "NECROBINDER": [("STRIKE_NECROBINDER", 5), ("DEFEND_NECROBINDER", 4), ("BONE_PILE", 1)],
}


# ---------------------------------------------------------------------------
# State tracker
# ---------------------------------------------------------------------------

class RunStateTracker:
    """Walks a run's floor_timeline in order, exposing state at the START of
    each floor (before that floor's decisions / combat / mutations apply).

    Deck representation: list of {"id", "upgrade"} per card copy — matches
    STS2's actual data model (upgrade level is per-copy, not per-id).
    """

    def __init__(self, character: str, ascension: int) -> None:
        self.character = character
        self.ascension = int(ascension or 0)
        self.deck: list[dict] = []
        for cid, n in STARTER_DECKS.get(character, []):
            for _ in range(n):
                self.deck.append({"id": cid, "upgrade": 0})
        # Ascension 10+ starts with Ascender's Bane as an unplayable curse.
        if self.ascension >= 10:
            self.deck.append({"id": "ASCENDERS_BANE", "upgrade": 0})
        self.relics: list[str] = []
        self.hp: int = 0
        self.max_hp: int = 0
        self.gold: int = 0

    def snapshot(self, floor_timeline_entry: dict) -> dict:
        """Read hp/gold at the *start* of the floor this entry represents."""
        self.hp = int(floor_timeline_entry.get("hp_before") or self.hp)
        self.gold = int(floor_timeline_entry.get("gold_before") or self.gold)
        # max_hp isn't directly given; approximate from observed hp_after highs
        # (player can heal above current max only via rare events).
        hp_after = int(floor_timeline_entry.get("hp_after") or 0)
        self.max_hp = max(self.max_hp, self.hp, hp_after)
        return {
            "hp": self.hp,
            "max_hp": self.max_hp,
            "gold": self.gold,
            "deck": self._export_deck(),
            "relics": list(self.relics),
        }

    def apply_floor(self, ft: dict) -> None:
        """Apply mutations from this floor AFTER we've emitted decisions for
        it. Operations in order: card_choices → shop → card_upgrades → relics.
        """
        for c in ft.get("card_choices") or []:
            if isinstance(c, dict) and c.get("was_picked"):
                cid = c.get("card_id")
                if cid:
                    self.deck.append({"id": cid, "upgrade": 0})
        for act in ft.get("shop_actions") or []:
            if not isinstance(act, dict):
                continue
            at = str(act.get("action_type") or "").lower()
            iid = act.get("item_id")
            if not iid:
                continue
            if at == "remove":
                for i, card in enumerate(self.deck):
                    if card["id"] == iid:
                        self.deck.pop(i)
                        break
            elif at in ("buy_card", "purchase_card"):
                self.deck.append({"id": iid, "upgrade": 0})
        for upg in ft.get("card_upgrades") or []:
            if isinstance(upg, dict):
                cid = upg.get("card_id")
                if cid:
                    for card in self.deck:
                        if card["id"] == cid and card["upgrade"] == 0:
                            card["upgrade"] = 1
                            break
        for ac in ft.get("ancient_choices") or []:
            if isinstance(ac, dict) and ac.get("was_picked"):
                rid = ac.get("relic_id")
                if rid:
                    self.relics.append(rid)
        for rc in ft.get("relic_choices") or []:
            if isinstance(rc, dict) and rc.get("was_picked"):
                rid = rc.get("relic_id")
                if rid:
                    self.relics.append(rid)

    def _export_deck(self) -> list[dict]:
        """Compact deck representation: {id, count, upgrade=0|1}."""
        bucket: dict[tuple[str, int], int] = collections.Counter()
        for card in self.deck:
            bucket[(card["id"], card["upgrade"])] += 1
        return [
            {"id": cid, "count": n, "upgrade": up}
            for (cid, up), n in sorted(bucket.items())
        ]


# ---------------------------------------------------------------------------
# Decision extractors (one per phase type)
# ---------------------------------------------------------------------------

def extract_campfire_decisions(run_obj: dict, tracker: RunStateTracker, ft: dict) -> list[dict]:
    """Rest vs Forge vs other campfire choice. Label is the explicit
    `campfire_choice` string (SMITH / REST / etc.)."""
    choice = ft.get("campfire_choice")
    if not choice:
        return []
    state = tracker.snapshot(ft)
    # Canonical campfire options by frequency in Skada data:
    # HEAL=rest (recover HP), SMITH=forge (upgrade 1 card), COOK/CLONE/HATCH/
    # DIG/LIFT/TOKE/RECALL/PRAY are relic-granted alternatives. FILLER is a
    # sentinel Skada writes for abandoned/incomplete campfire events — drop
    # those at BC load time.
    options = ["HEAL", "SMITH", "COOK", "CLONE", "HATCH", "DIG", "LIFT", "TOKE", "RECALL", "PRAY"]
    chosen_upper = str(choice).upper()
    return [{
        "phase": "campfire",
        "character": tracker.character,
        "ascension": tracker.ascension,
        "run_id": run_obj["run"].get("run_id"),
        "floor": ft.get("floor"),
        "act": (int(ft.get("floor", 0)) - 1) // 15,
        **state,
        "options": options,
        "chosen": chosen_upper,
        "chosen_idx": options.index(chosen_upper) if chosen_upper in options else -1,
    }]


def extract_card_reward_decisions(run_obj: dict, tracker: RunStateTracker, ft: dict) -> list[dict]:
    """Card reward screen: one of N cards, or skip.

    Skada collapses each screen into a list of offered cards with
    `was_picked` flags — exactly one is True (or all False = skip).
    """
    choices = ft.get("card_choices") or []
    if not isinstance(choices, list) or not choices:
        return []
    state = tracker.snapshot(ft)
    offered = [c.get("card_id") for c in choices if isinstance(c, dict)]
    picked = [c.get("card_id") for c in choices if isinstance(c, dict) and c.get("was_picked")]
    # Include "SKIP" as a valid option at the end — BC must learn to choose
    # skipping over a bad card, this is a first-class action.
    options = offered + ["SKIP"]
    chosen = picked[0] if picked else "SKIP"
    chosen_idx = options.index(chosen) if chosen in options else -1
    return [{
        "phase": "card_reward",
        "character": tracker.character,
        "ascension": tracker.ascension,
        "run_id": run_obj["run"].get("run_id"),
        "floor": ft.get("floor"),
        "act": (int(ft.get("floor", 0)) - 1) // 15,
        **state,
        "options": options,
        "chosen": chosen,
        "chosen_idx": chosen_idx,
    }]


def extract_relic_decisions(run_obj: dict, tracker: RunStateTracker, ft: dict) -> list[dict]:
    """Both ancient_choices and relic_choices produce relic-pick labels.

    ancient: event-driven, usually 3 options, 1 picked or 0 (skipped).
    relic: combat reward / shop / boss, varies.
    """
    out = []
    for key, kind in (("ancient_choices", "ancient"), ("relic_choices", "relic")):
        items = ft.get(key) or []
        if not isinstance(items, list) or not items:
            continue
        state = tracker.snapshot(ft)
        offered = [x.get("relic_id") for x in items if isinstance(x, dict)]
        picked = [x.get("relic_id") for x in items if isinstance(x, dict) and x.get("was_picked")]
        # Ancient relic events force exactly one pick (can_skip=False in-game).
        # Generic relic_choices (combat reward / shop) allow skip.
        allow_skip = (kind != "ancient")
        options = offered + (["SKIP"] if allow_skip else [])
        chosen = picked[0] if picked else "SKIP"
        chosen_idx = options.index(chosen) if chosen in options else -1
        out.append({
            "phase": f"relic_{kind}",
            "character": tracker.character,
            "ascension": tracker.ascension,
            "run_id": run_obj["run"].get("run_id"),
            "floor": ft.get("floor"),
            "act": (int(ft.get("floor", 0)) - 1) // 15,
            **state,
            "options": options,
            "chosen": chosen,
            "chosen_idx": chosen_idx,
        })
    return out


def extract_map_decisions(run_obj: dict, tracker: RunStateTracker) -> list[dict]:
    """Map navigation. Walks visited_coords in each act, emitting one sample
    per (coord_from → coord_to) transition.

    State at each map decision is whatever is stored at the floor
    corresponding to coord_from. We approximate by assuming visited_coords[i]
    corresponds to floor_timeline entry where map_coord matches — but Skada
    doesn't always expose coord-to-floor mapping, so we fall back to using
    the enclosing act's start floor + i.
    """
    out: list[dict] = []
    # Build a floor_timeline lookup by act-relative index
    # In STS2, act 1 = floors 1-15, act 2 = 16-30, act 3 = 31-45
    floor_by_act = {0: [], 1: [], 2: [], 3: []}
    for ft in run_obj.get("floor_timeline") or []:
        f = ft.get("floor")
        if not isinstance(f, int):
            continue
        a = (f - 1) // 15
        if a in floor_by_act:
            floor_by_act[a].append(ft)

    for map_act in run_obj.get("map_acts") or []:
        act_idx = map_act.get("act") or 0
        nodes = {tuple(n["coord"]): n for n in (map_act.get("nodes") or []) if isinstance(n, dict)}
        visited = map_act.get("visited_coords") or []
        if len(visited) < 2:
            continue
        act_floors = floor_by_act.get(act_idx, [])
        for i in range(len(visited) - 1):
            a = tuple(visited[i])
            b = tuple(visited[i + 1])
            node = nodes.get(a)
            if node is None:
                continue
            children = [tuple(c) for c in (node.get("children") or [])]
            if b not in children:
                continue
            # Snapshot state at this floor if available
            ft = act_floors[i] if i < len(act_floors) else None
            if ft is None:
                continue
            state = tracker.snapshot(ft)
            option_ids = [f"{c[0]},{c[1]}" for c in children]
            chosen_id = f"{b[0]},{b[1]}"
            out.append({
                "phase": "map",
                "character": tracker.character,
                "ascension": tracker.ascension,
                "run_id": run_obj["run"].get("run_id"),
                "floor": ft.get("floor"),
                "act": act_idx,
                **state,
                "coord_from": list(a),
                "options": option_ids,
                "option_types": [nodes[c].get("type") if c in nodes else "?" for c in children],
                "chosen": chosen_id,
                "chosen_idx": option_ids.index(chosen_id),
            })
    return out


# ---------------------------------------------------------------------------
# Main pipeline
# ---------------------------------------------------------------------------

def process_run(run_obj: dict) -> list[dict]:
    run = run_obj.get("run") or {}
    character = str(run.get("character") or "").upper()
    if character not in STARTER_DECKS:
        return []
    ascension = run.get("ascension") or 0
    tracker = RunStateTracker(character, ascension)
    samples: list[dict] = []

    # Emit map decisions first (based on visited_coords, state tracker reads
    # floor-indexed snapshots as it goes).
    # NOTE: we DON'T apply floor mutations during map extraction — map
    # decisions precede that floor's combat / reward / shop activity.
    samples.extend(extract_map_decisions(run_obj, RunStateTracker(character, ascension)))

    # For per-floor phase decisions, walk floor_timeline in order. At each
    # floor: snapshot BEFORE applying mutations, then extract decisions that
    # live on that floor, then apply mutations so the NEXT floor's snapshot
    # reflects picks made on this floor.
    for ft in run_obj.get("floor_timeline") or []:
        samples.extend(extract_campfire_decisions(run_obj, tracker, ft))
        samples.extend(extract_card_reward_decisions(run_obj, tracker, ft))
        samples.extend(extract_relic_decisions(run_obj, tracker, ft))
        tracker.apply_floor(ft)

    return samples


def iter_sp_runs(limit: int):
    count = 0
    for outcome in ("victory", "failure"):
        for shard in sorted((SP_DIR / outcome).glob("*.jsonl")):
            with open(shard, "r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        yield json.loads(line), outcome
                    except Exception:
                        continue
                    count += 1
                    if limit and count >= limit:
                        return


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--limit", type=int, default=0, help="0 = all SP runs")
    parser.add_argument("--out", default=str(DEFAULT_OUT))
    args = parser.parse_args()

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    stats = collections.Counter()
    phase_stats = collections.Counter()
    chosen_idx_bad = collections.Counter()

    with open(out_path, "w", encoding="utf-8") as fout:
        for run_obj, outcome in iter_sp_runs(args.limit):
            stats[f"runs_{outcome}"] += 1
            samples = process_run(run_obj)
            for s in samples:
                phase_stats[s["phase"]] += 1
                if s.get("chosen_idx", 0) < 0:
                    chosen_idx_bad[s["phase"]] += 1
                fout.write(json.dumps(s, ensure_ascii=False, separators=(",", ":")) + "\n")
                stats["samples"] += 1
            if stats["samples"] and stats["samples"] % 50_000 == 0:
                print(f"  ... {stats['samples']:,} samples", flush=True)

    print()
    print(f"runs processed:")
    for k in ("runs_victory", "runs_failure"):
        print(f"  {k}: {stats[k]:,}")
    print(f"samples written: {stats['samples']:,} → {out_path}")
    print(f"samples by phase:")
    for phase, n in phase_stats.most_common():
        bad = chosen_idx_bad.get(phase, 0)
        print(f"  {phase:18s} {n:>8,}  (skip/bad: {bad})")


if __name__ == "__main__":
    main()
