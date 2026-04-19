"""Validate that tracking card_choices + card_upgrades + shop_actions across
floor_timeline, starting from the character's known starter deck, reconstructs
the exact final_deck reported in each run.

If reconstruction matches final_deck within tolerance, per-floor deck state is
fully recoverable and card-conditioned BC becomes tractable.

Edge cases to track:
  - Events that transform/duplicate/remove cards (event_text field, no
    structured label) — gap source
  - Ascender's Bane / starter curse additions
  - Neow bonus (start-of-run card / relic)

Usage:
  python probe_deck_reconstruction.py [--limit N]
"""
from __future__ import annotations

import argparse
import collections
import json
from pathlib import Path

# Starter decks per character (STS2 standard, no ascension curses).
# Counts are per-card-id before any upgrades.
STARTER_DECKS: dict[str, dict[str, int]] = {
    "IRONCLAD": {
        "STRIKE_IRONCLAD": 5,
        "DEFEND_IRONCLAD": 4,
        "BASH": 1,
    },
    "SILENT": {
        "STRIKE_SILENT": 5,
        "DEFEND_SILENT": 5,
        "NEUTRALIZE": 1,
        "SURVIVOR": 1,
    },
    "DEFECT": {
        "STRIKE_DEFECT": 4,
        "DEFEND_DEFECT": 4,
        "ZAP": 1,
        "DUALCAST": 1,
    },
    "REGENT": {
        "STRIKE_REGENT": 5,
        "DEFEND_REGENT": 4,
        # Regent-specific starter — using reasonable guess; validate via
        # reconstruction error breakdown
        "ROYAL_DECREE": 1,
    },
    "NECROBINDER": {
        "STRIKE_NECROBINDER": 5,
        "DEFEND_NECROBINDER": 4,
        "BONE_PILE": 1,
    },
}

SP_DIR = Path("data/skada_clean/sp")


def starter_deck(character: str) -> dict[str, int]:
    return dict(STARTER_DECKS.get(character, {}))


def apply_floor(deck: dict[str, int], ft: dict, upgrades: dict[str, int]) -> None:
    """Apply one floor's card mutations in place.

    ``upgrades[card_id]`` tracks upgrade_level per unique card_id instance.
    Simplification: STS2 tracks upgrade per-copy; we aggregate by id so
    reconstruction matches final_deck grouped by id.
    """
    for c in ft.get("card_choices") or []:
        if isinstance(c, dict) and c.get("was_picked"):
            card_id = c.get("card_id")
            if card_id:
                deck[card_id] = deck.get(card_id, 0) + 1
    for upg in ft.get("card_upgrades") or []:
        if isinstance(upg, dict):
            card_id = upg.get("card_id")
            if card_id:
                upgrades[card_id] = upgrades.get(card_id, 0) + 1
    for act in ft.get("shop_actions") or []:
        if not isinstance(act, dict):
            continue
        at = str(act.get("action_type") or "").lower()
        iid = act.get("item_id")
        if not iid:
            continue
        if at == "remove":
            if deck.get(iid, 0) > 0:
                deck[iid] -= 1
                if deck[iid] == 0:
                    del deck[iid]
        elif at in ("buy_card", "purchase_card", "buy"):
            deck[iid] = deck.get(iid, 0) + 1
        elif at in ("upgrade", "smith"):
            upgrades[iid] = upgrades.get(iid, 0) + 1


def final_deck_to_counter(final_deck: list) -> collections.Counter:
    """Turn the run's final_deck list into a {card_id: count} counter."""
    c = collections.Counter()
    for entry in final_deck or []:
        if not isinstance(entry, dict):
            continue
        cid = entry.get("card_id")
        cnt = entry.get("count", 1)
        if cid and isinstance(cnt, int):
            c[cid] += cnt
    return c


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--limit", type=int, default=5000)
    args = parser.parse_args()

    exact_match = 0
    off_by_1_5 = 0
    off_by_6plus = 0
    characters_with_issues = collections.Counter()
    total = 0
    sample_misses = []

    for outcome in ("victory", "failure"):
        for shard in sorted((SP_DIR / outcome).glob("*.jsonl")):
            with open(shard, "r", encoding="utf-8") as f:
                for line in f:
                    if total >= args.limit:
                        break
                    obj = json.loads(line)
                    run = obj.get("run") or {}
                    character = str(run.get("character") or "").upper()
                    if character not in STARTER_DECKS:
                        # Unknown character — skip rather than pollute stats
                        continue
                    total += 1

                    deck = starter_deck(character)
                    upgrades: dict[str, int] = {}
                    for ft in obj.get("floor_timeline") or []:
                        apply_floor(deck, ft, upgrades)

                    # Fold upgrades into the deck counter: each upgrade on
                    # card_id moves one copy from "card_id" to "card_id+".
                    # final_deck uses the "+" suffix, so reconstruction
                    # must match that convention.
                    reconstructed: collections.Counter[str] = collections.Counter()
                    for cid, cnt in deck.items():
                        if cnt <= 0:
                            continue
                        n_up = min(upgrades.get(cid, 0), cnt)
                        if n_up:
                            reconstructed[f"{cid}+"] += n_up
                        if cnt - n_up > 0:
                            reconstructed[cid] += (cnt - n_up)
                    actual = final_deck_to_counter(obj.get("final_deck"))

                    # Cards reconstructed - actual = extras we added that shouldn't be there
                    extras = reconstructed - actual
                    missing = actual - reconstructed
                    diff_count = sum(extras.values()) + sum(missing.values())

                    if diff_count == 0:
                        exact_match += 1
                    elif diff_count <= 5:
                        off_by_1_5 += 1
                    else:
                        off_by_6plus += 1
                        characters_with_issues[character] += 1
                        if len(sample_misses) < 5:
                            sample_misses.append({
                                "run_id": run.get("run_id"),
                                "character": character,
                                "floor_reached": run.get("floor_reached"),
                                "diff_count": diff_count,
                                "extras": dict(extras.most_common(10)),
                                "missing": dict(missing.most_common(10)),
                            })
                if total >= args.limit:
                    break
            if total >= args.limit:
                break
        if total >= args.limit:
            break

    print(f"runs checked: {total:,}")
    pct = lambda n: f"{n} ({n/max(total,1)*100:.1f}%)"
    print(f"exact match final_deck:  {pct(exact_match)}")
    print(f"off by 1-5 cards:         {pct(off_by_1_5)}")
    print(f"off by 6+ cards:          {pct(off_by_6plus)}")
    if characters_with_issues:
        print("characters with most >6-card discrepancies:")
        for ch, n in characters_with_issues.most_common():
            print(f"  {ch}: {n}")
    print()
    for sm in sample_misses:
        print(f"miss run_id={sm['run_id']} char={sm['character']} floor={sm['floor_reached']} diff={sm['diff_count']}")
        print(f"  extras:  {sm['extras']}")
        print(f"  missing: {sm['missing']}")


if __name__ == "__main__":
    main()
