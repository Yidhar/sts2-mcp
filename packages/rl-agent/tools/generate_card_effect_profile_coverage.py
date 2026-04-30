"""Emit ``docs/generated/card-effect-profile-coverage.md`` and
``docs/generated/card-effect-profile-schema.md`` from
``content/card_effect_profiles.generated.json``.

The coverage doc reports per-class (Ironclad / Colorless) how many cards have a
populated ``derived_view`` block, broken down by source quality
(``curated_internal_id`` / ``generated_source_facts`` / regex fallback) and by
field group fill-rate.  The schema doc is a static description of the
``derived_view`` fields and their derivation order.
"""

from __future__ import annotations

import collections
import json
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[3]
PROFILES_PATH = REPO_ROOT / "packages" / "rl-agent" / "content" / "card_effect_profiles.generated.json"
IRONCLAD_BASE_PATH = REPO_ROOT / "docs" / "generated" / "ironclad-cards-base.json"
COLORLESS_BASE_PATH = REPO_ROOT / "docs" / "generated" / "colorless-cards-base.json"
COVERAGE_OUT = REPO_ROOT / "docs" / "generated" / "card-effect-profile-coverage.md"
SCHEMA_OUT = REPO_ROOT / "docs" / "generated" / "card-effect-profile-schema.md"


_FIELD_GROUPS = (
    "cost",
    "lifecycle",
    "hand_mutation",
    "pile_mutation",
    "combat_effect",
    "mechanism_effect",
    "source",
)


def _load_base_ids(path: Path) -> list[str]:
    with path.open("r", encoding="utf-8") as fh:
        data = json.load(fh)
    cards = data.get("cards") or []
    return [str(card.get("id")) for card in cards if card.get("id")]


def _value_is_filled(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        return value != 0
    if isinstance(value, str):
        return bool(value.strip())
    if isinstance(value, dict):
        if "enabled" in value:
            return bool(value["enabled"])
        return any(_value_is_filled(v) for v in value.values())
    if isinstance(value, list):
        return bool(value)
    return value is not None


def _section_fill_rate(section: dict[str, Any] | None) -> tuple[int, int]:
    if not isinstance(section, dict) or not section:
        return 0, 0
    total = 0
    filled = 0
    for v in section.values():
        total += 1
        if _value_is_filled(v):
            filled += 1
    return filled, total


def _coverage_row(profile: dict[str, Any]) -> dict[str, Any]:
    view = profile.get("derived_view") or {}
    quality = (profile.get("source_facts") or {}).get("source_profile_quality") or "unknown"
    fallback = bool((view.get("source") or {}).get("fallback_text_regex_used"))
    fills: dict[str, tuple[int, int]] = {}
    for g in _FIELD_GROUPS:
        fills[g] = _section_fill_rate(view.get(g))
    return {
        "card_id": profile.get("id"),
        "title": profile.get("title_en"),
        "class_name": profile.get("class_name"),
        "quality": quality,
        "fallback_used": fallback,
        "fills": fills,
    }


def _summarize(rows: list[dict[str, Any]]) -> dict[str, Any]:
    total = len(rows)
    quality_counter: collections.Counter[str] = collections.Counter()
    fallback_count = 0
    populated_per_group: collections.Counter[str] = collections.Counter()
    for row in rows:
        quality_counter[row["quality"]] += 1
        if row["fallback_used"]:
            fallback_count += 1
        for g, (filled, _total) in row["fills"].items():
            if filled > 0:
                populated_per_group[g] += 1
    return {
        "total": total,
        "quality": dict(quality_counter),
        "fallback_count": fallback_count,
        "populated_per_group": {g: populated_per_group[g] for g in _FIELD_GROUPS},
    }


def _markdown_class_table(class_label: str, rows: list[dict[str, Any]]) -> str:
    lines: list[str] = []
    lines.append(f"### {class_label} — {len(rows)} cards\n")
    summary = _summarize(rows)
    lines.append("Source quality:\n")
    for q, n in sorted(summary["quality"].items()):
        lines.append(f"- `{q}`: {n}")
    lines.append("")
    lines.append(f"Cards with regex fallback: **{summary['fallback_count']} / {summary['total']}**\n")
    lines.append("Field-group population (cards with at least one filled field in group):\n")
    lines.append("| Group | Cards populated | Coverage |")
    lines.append("|---|---|---|")
    for g in _FIELD_GROUPS:
        n = summary["populated_per_group"][g]
        pct = (100.0 * n / summary["total"]) if summary["total"] else 0.0
        lines.append(f"| `{g}` | {n}/{summary['total']} | {pct:.1f}% |")
    lines.append("")
    lines.append("Per-card breakdown (sorted by id):\n")
    lines.append(
        "| Card ID | Quality | "
        + " | ".join(f"`{g}`" for g in _FIELD_GROUPS)
        + " | Fallback |"
    )
    lines.append("|---|---|" + "|".join(["---"] * len(_FIELD_GROUPS)) + "|---|")
    for row in sorted(rows, key=lambda r: r["card_id"] or ""):
        cells = []
        for g in _FIELD_GROUPS:
            filled, total = row["fills"][g]
            cells.append(f"{filled}/{total}" if total else "—")
        lines.append(
            f"| `{row['card_id']}` | {row['quality']} | "
            + " | ".join(cells)
            + f" | {'yes' if row['fallback_used'] else 'no'} |"
        )
    lines.append("")
    return "\n".join(lines)


def write_coverage(profiles: dict[str, Any]) -> None:
    ironclad_ids = set(_load_base_ids(IRONCLAD_BASE_PATH))
    colorless_ids = set(_load_base_ids(COLORLESS_BASE_PATH))

    ironclad_rows: list[dict[str, Any]] = []
    colorless_rows: list[dict[str, Any]] = []
    for cid, profile in profiles.items():
        row = _coverage_row(profile)
        if cid in ironclad_ids:
            ironclad_rows.append(row)
        elif cid in colorless_ids:
            colorless_rows.append(row)

    ironclad_missing = sorted(ironclad_ids - {p["card_id"] for p in ironclad_rows})
    colorless_missing = sorted(colorless_ids - {p["card_id"] for p in colorless_rows})

    lines: list[str] = []
    lines.append("# CardEffectProfile derived_view coverage")
    lines.append("")
    lines.append(
        "Auto-generated by `tools/generate_card_effect_profile_coverage.py` from "
        "`content/card_effect_profiles.generated.json`."
    )
    lines.append("")
    lines.append("Each card profile carries a structured `derived_view` block (TASK-D1) "
                 "with seven field groups (`cost`, `lifecycle`, `hand_mutation`, "
                 "`pile_mutation`, `combat_effect`, `mechanism_effect`, `source`). "
                 "This document reports how many fields per group are populated for the "
                 "Ironclad and Colorless rosters defined in "
                 "`docs/generated/ironclad-cards-base.json` and `colorless-cards-base.json`.")
    lines.append("")
    lines.append("## Summary")
    lines.append("")
    lines.append(f"- Ironclad roster size: **{len(ironclad_ids)}**, profiles found: **{len(ironclad_rows)}**, missing: **{len(ironclad_missing)}**")
    lines.append(f"- Colorless roster size: **{len(colorless_ids)}**, profiles found: **{len(colorless_rows)}**, missing: **{len(colorless_missing)}**")
    lines.append("")
    if ironclad_missing:
        lines.append("Missing Ironclad profiles:")
        for cid in ironclad_missing:
            lines.append(f"- `{cid}`")
        lines.append("")
    if colorless_missing:
        lines.append("Missing Colorless profiles:")
        for cid in colorless_missing[:50]:
            lines.append(f"- `{cid}`")
        if len(colorless_missing) > 50:
            lines.append(f"- ... and {len(colorless_missing) - 50} more")
        lines.append("")
    lines.append("## Class breakdown")
    lines.append("")
    lines.append(_markdown_class_table("Ironclad", ironclad_rows))
    lines.append(_markdown_class_table("Colorless", colorless_rows))

    COVERAGE_OUT.parent.mkdir(parents=True, exist_ok=True)
    COVERAGE_OUT.write_text("\n".join(lines), encoding="utf-8")
    print(f"wrote {COVERAGE_OUT}")


def write_schema() -> None:
    text = """# CardEffectProfile derived_view — schema

The `derived_view` block on every card in
`content/card_effect_profiles.generated.json` is a stable structured summary of
the card's mechanical profile.  It is derived from internal card ids, C# source
facts, curated overrides and (only as last resort) localized text regex.

## Field source priority

Per TASK-D1 the derivation order is:

1. **Game-internal card data / effect components / power ids / card modifiers.**
2. **Bridge C# reflection or explicit profile.**
3. **Hand-maintained override table** (`CURATED` in
   `tools/generate_card_effect_profiles.py`).
4. **Localized text regex** — only as fallback, and the result must be flagged
   with `source.fallback_text_regex_used = true`.

## Top-level fields

| Field | Description |
|---|---|
| `card_id` | Canonical `CARD.<NORMALIZED>` id. |
| `title` | English title (display only — not used for parsing). |
| `color` | Card class (`ironclad`, `colorless`, ...). |
| `type` | `attack`, `skill`, `power`, `status`, `curse`, `quest`, `token`. |
| `rarity` | `basic`, `common`, `uncommon`, `rare`, `event`, `ancient`, ... |

## Field groups

### `cost`
| Field | Description |
|---|---|
| `base` | Base energy cost (`-1` for X-cost cards). |
| `upgraded` | Upgraded energy cost (`-1` for X-cost cards). |
| `is_x_cost` | True if the card resolves cost as X (e.g. Whirlwind). |
| `can_change_cost` | True if any operation modifies the card's cost. |
| `cost_reduction_tags` | Tags describing cost-modifier ops (`set_cost_0`, `delta_-1`, `duration_this_turn`, ...). |

### `lifecycle`
| Field | Description |
|---|---|
| `exhausts_on_play` | True if the played card itself exhausts on play (Exhaust keyword). |
| `ethereal` | True if Ethereal keyword is present. |
| `retain` | True if Retain keyword is present. |
| `self_purge` | True if the card explicitly removes itself from combat. |
| `returns_to_hand` | True if the card returns to hand after play. |
| `replay_or_duplicate` | True if the card replays itself or duplicates other cards. |

### `hand_mutation`
| Field | Description |
|---|---|
| `upgrades_hand` | True if any op upgrades a card in hand. |
| `upgrade_targets` | One of `one`, `choice_one`, `random_one`, `one_or_all_by_upgrade_state`. |
| `transforms_cards` | True if any op transforms a hand card. |
| `copies_cards` | True if any op copies a hand card. |
| `creates_cards` | True if any op generates new cards into combat. |
| `discard_hand` | True if any op discards the entire hand. |
| `draw` | Total cards drawn (sum of `count` across `draw_card` ops). |
| `select_cards` | `{enabled, min, max, target_zone}` — selection constraints if the card requires choosing target cards. |

### `pile_mutation`
| Field | Description |
|---|---|
| `moves_to_exhaust_self` | Played card itself moves to exhaust pile. |
| `moves_to_discard_self` | Played card itself moves to discard pile (default for non-exhaust, non-power cards). |
| `moves_to_exhaust` | Any op moves a card to exhaust (self or other). |
| `moves_to_discard` | Any op moves a card to discard (other than the played card). |
| `shuffles_into_draw` | Any op moves a card to the draw pile. |
| `puts_card_on_top` | Any op puts a card on top of the draw pile. |
| `removes_card_from_combat` | Any op removes a card from combat permanently (transform without `result_card`). |

### `combat_effect`
| Field | Description |
|---|---|
| `damage` | Base damage roll (parsed from `DamageVar`). |
| `block` | Base block roll (parsed from `BlockVar`). |
| `hit_count` | Multi-hit count parsed from `WithHitCount`. |
| `weak`, `vulnerable`, `frail`, `poison` | Counts of debuff power applications. |
| `strength`, `dexterity`, `artifact`, `thorns` | Counts of buff power applications. |
| `energy_gain` | Energy gained by `gain_energy` ops. |
| `target_type` | `Self`, `AnyEnemy`, `AllEnemies`, `AnyOpponent`, ... |

### `mechanism_effect`
| Field | Description |
|---|---|
| `can_change_facing` | True if the card targets an enemy (Kaiser facing relevance). |
| `can_strip_artifact` | True if the source mentions Artifact-strip primitives. |
| `can_trigger_stun` | True if the card applies StunPower. |
| `one_card_lock_impact` | `unknown` for now — Phase 4 (TASK-E2 Ceremonial) will populate. |

### `source`
| Field | Description |
|---|---|
| `primary` | Always `game_internal_id` (we never derive primarily from text). |
| `fallback_text_regex_used` | True only when no internal/curated path could derive the profile. |
| `profile_quality` | `curated_internal_id` (CURATED override), `generated_source_facts` (C# parse), or `unknown`. |
| `source_available` | True if the C# source file was reachable. |
| `source_sha1` | SHA1 of the C# source file at generation time. |

## Reading the derived_view

`sts2_env.card_effect_profile` exposes:

```python
from sts2_env.card_effect_profile import (
    card_derived_view,
    card_cost_view,
    card_lifecycle_view,
    card_hand_mutation_view,
    card_pile_mutation_view,
    card_combat_effect_view,
    card_mechanism_effect_view,
    card_source_view,
)
```

Each helper accepts a card-payload dict and returns the named subsection (or
`{}` if the profile is missing).  Observation/action encoders consume these
helpers — they MUST NOT walk localized card text.
"""
    SCHEMA_OUT.parent.mkdir(parents=True, exist_ok=True)
    SCHEMA_OUT.write_text(text, encoding="utf-8")
    print(f"wrote {SCHEMA_OUT}")


def main() -> None:
    with PROFILES_PATH.open("r", encoding="utf-8") as fh:
        data = json.load(fh)
    cards = data.get("cards") or {}
    write_coverage(cards)
    write_schema()


if __name__ == "__main__":
    main()
