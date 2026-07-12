from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Iterable

from content_registry import humanize_game_id
from sts2_rl.game_data import repository_root, resolve_generated_game_data_output

_DEFAULT_OUTPUT = repository_root() / "game-data" / "generated" / "enemies.static.generated.json"
_DISCOVERY_FALLBACK_IDS: tuple[str, ...] = (
    "MONSTER.BYRDONIS",
    "MONSTER.CEREMONIAL_BEAST",
    "MONSTER.DOOR",
    "MONSTER.DOORMAKER",
    "MONSTER.INFESTED_PRISM",
    "MONSTER.KNOWLEDGE_DEMON",
    "MONSTER.LAGAVULIN_MATRIARCH",
    "MONSTER.LOUSE_PROGENITOR",
    "MONSTER.QUEEN",
    "MONSTER.SOUL_FYSH",
    "MONSTER.SOUL_NEXUS",
    "MONSTER.SPINY_TOAD",
    "MONSTER.TEST_SUBJECT",
    "MONSTER.THE_INSATIABLE",
    "MONSTER.WATERFALL_GIANT",
)


def _deep_merge_dict(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    merged: dict[str, Any] = dict(base)
    for key, value in override.items():
        existing = merged.get(key)
        if isinstance(existing, dict) and isinstance(value, dict):
            merged[key] = _deep_merge_dict(existing, value)
        else:
            merged[key] = value
    return merged


def _pretty_title(enemy_id: str) -> str:
    tail = str(enemy_id or "").strip().split(".", 1)[-1]
    words = [part for part in tail.split("_") if part]
    if not words:
        return humanize_game_id(enemy_id)
    return " ".join(
        word if (word.isupper() and len(word) <= 2) or word.isdigit() else word.capitalize()
        for word in words
    )


def _trait(trait: str, description: str, **extra: Any) -> dict[str, Any]:
    payload = {"trait": trait, "description": description}
    payload.update({key: value for key, value in extra.items() if value not in (None, "", [], {})})
    return payload


def _reactive(
    trigger_type: str,
    condition: str,
    effect_type: str,
    description: str,
    *,
    effect_amount: int | float | None = None,
    severity: str | None = None,
) -> dict[str, Any]:
    return _trait(
        effect_type,
        description,
        category="reactive",
        trigger_type=trigger_type,
        condition=condition,
        effect_type=effect_type,
        effect_amount=effect_amount,
        severity=severity,
    )


def _phase_rule(
    trigger_type: str,
    condition: str,
    effect_type: str,
    description: str,
    *,
    threshold: int | float | None = None,
    state: str | None = None,
    severity: str | None = None,
) -> dict[str, Any]:
    return _trait(
        effect_type,
        description,
        category="phase",
        trigger_type=trigger_type,
        condition=condition,
        effect_type=effect_type,
        threshold=threshold,
        state=state,
        severity=severity,
    )


def _danger(
    *,
    burst: int = 0,
    attrition: int = 0,
    scaling: int = 0,
    retaliation: int = 0,
    summon_pressure: int = 0,
    debuff_pressure: int = 0,
    phase_complexity: int = 0,
    volatility: int = 0,
    target_priority: int = 0,
    notes: str | None = None,
) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "burst": int(burst),
        "attrition": int(attrition),
        "scaling": int(scaling),
        "retaliation": int(retaliation),
        "summon_pressure": int(summon_pressure),
        "debuff_pressure": int(debuff_pressure),
        "phase_complexity": int(phase_complexity),
        "volatility": int(volatility),
        "target_priority": int(target_priority),
    }
    if notes:
        payload["notes"] = notes
    return payload


def _hint(priority: str, reason: str, *, when: str | None = None) -> dict[str, Any]:
    payload = {"priority": priority, "reason": reason}
    if when:
        payload["when"] = when
    return payload


_CURATED_OVERRIDES: dict[str, dict[str, Any]] = {
    "MONSTER.SPINY_TOAD": {
        "summary": "Contact-punish enemy. Multi-hit turns become expensive unless you remove or route around the backlash first.",
        "semantic_tags": ["retaliation", "contact_punish", "attrition"],
        "combat_tags": ["retaliation", "frontline", "priority_on_spam_decks"],
        "static_traits": [
            _trait("contact_retaliate", "Punishes contact hits, so small-hit spam is inefficient.", severity="high"),
        ],
        "reactive_triggers": [
            _reactive("on_hit", "contact", "retaliate", "On hit, retaliates / punishes contact damage.", severity="high"),
        ],
        "danger_profile": _danger(attrition=4, retaliation=5, volatility=2, target_priority=4),
        "target_priority_hints": [
            _hint("high", "Remove retaliation sources before committing multi-hit damage turns."),
        ],
    },
    "MONSTER.LOUSE_PROGENITOR": {
        "summary": "Summon-centric pressure source. If left alive it snowballs board count and drags single-target decks into bad tempo.",
        "semantic_tags": ["summoner", "board_flood", "engine"],
        "combat_tags": ["summoner", "engine", "priority_target"],
        "static_traits": [
            _trait("spawn_core", "Primary value comes from sustaining or replacing adds.", severity="high"),
        ],
        "reactive_triggers": [
            _reactive("on_turn_end", "alive", "summon", "Continues adding board pressure if not removed.", severity="high"),
        ],
        "danger_profile": _danger(attrition=3, summon_pressure=5, target_priority=5),
        "target_priority_hints": [
            _hint("high", "Kill the progenitor early if your deck is weak to board flood."),
        ],
    },
    "MONSTER.SOUL_NEXUS": {
        "summary": "Elite summon hub / scaling core. Long fights get much worse once the nexus gets repeated value windows.",
        "semantic_tags": ["elite", "summoner", "scaling", "engine"],
        "combat_tags": ["elite", "summoner", "engine", "priority_target"],
        "static_traits": [
            _trait("summon_engine", "Acts as the central value engine in the encounter.", severity="high"),
            _trait("scaling_core", "Fight becomes harder the longer the nexus stays alive.", severity="high"),
        ],
        "reactive_triggers": [
            _reactive("on_turn_end", "alive", "summon", "Repeatedly increases board pressure / reinforcements.", severity="high"),
        ],
        "danger_profile": _danger(attrition=4, scaling=4, summon_pressure=5, volatility=3, target_priority=5),
        "target_priority_hints": [
            _hint("high", "Usually focus the engine before cleanup unless lethal on adds is trivial."),
        ],
    },
    "MONSTER.INFESTED_PRISM": {
        "summary": "Reactive elite fight with split / summon style board management. Single-target plans can get dragged into bad tempo.",
        "semantic_tags": ["elite", "summoner", "phase_shift", "board_flood"],
        "combat_tags": ["elite", "summoner", "phase_shift"],
        "static_traits": [
            _trait("infested_core", "Board state can multiply or persist beyond the original body.", severity="high"),
        ],
        "reactive_triggers": [
            _reactive("on_death", "killed", "summon", "Death may create follow-up pressure rather than ending the fight.", severity="high"),
        ],
        "phase_rules": [
            _phase_rule("on_hp_threshold", "threshold_crossed", "split", "Threshold damage can create additional board pressure.", severity="medium"),
        ],
        "danger_profile": _danger(attrition=3, summon_pressure=4, phase_complexity=4, volatility=4, target_priority=4),
        "target_priority_hints": [
            _hint("high", "Plan kill order around follow-up summons / splits instead of assuming one clean lethal ends the pressure."),
        ],
    },
    "MONSTER.BYRDONIS": {
        "summary": "Growth enemy. The longer the fight goes, the more dangerous its damage race becomes.",
        "semantic_tags": ["growth", "scaling", "elite_pressure"],
        "combat_tags": ["scaling", "frontline"],
        "static_traits": [
            _trait("time_scaling", "Gets more dangerous as turns pass.", severity="high"),
        ],
        "danger_profile": _danger(burst=3, scaling=5, volatility=3, target_priority=4),
        "target_priority_hints": [
            _hint("high", "Burst or control it early; prolonged setup lines are punished."),
        ],
    },
    "MONSTER.SHRINKER_BEETLE": {
        "summary": "Debuff attrition enemy. Repeated stat pressure reduces future combat quality even if immediate damage looks modest.",
        "semantic_tags": ["debuff", "attrition", "output_suppression"],
        "combat_tags": ["debuff", "attrition"],
        "static_traits": [
            _trait("stat_suppression", "Repeatedly reduces your effective output / tempo.", severity="medium"),
        ],
        "danger_profile": _danger(attrition=4, debuff_pressure=5, target_priority=4),
    },
    "MONSTER.FUZZY_WURM_CRAWLER": {
        "summary": "Scaling attacker. Looks manageable early, then snowballs into a burst check if left alone.",
        "semantic_tags": ["growth", "burst_check", "scaling"],
        "combat_tags": ["scaling", "frontline"],
        "static_traits": [
            _trait("damage_ramp", "Threat increases sharply over multiple turns.", severity="high"),
        ],
        "danger_profile": _danger(burst=4, scaling=5, volatility=3, target_priority=4),
    },
    "MONSTER.PHROG_PARASITE": {
        "summary": "Summon / board-flood enemy that drags single-target decks into losing tempo lines.",
        "semantic_tags": ["summoner", "board_flood", "tempo_tax"],
        "combat_tags": ["summoner", "tempo_tax"],
        "static_traits": [
            _trait("board_flood", "Encounter taxes decks that cannot pivot into AOE or clean point-targeting.", severity="high"),
        ],
        "danger_profile": _danger(attrition=4, summon_pressure=5, target_priority=4),
    },
}
_CURATED_OVERRIDES.update({
    "MONSTER.VANTOM": {
        "summary": "Boss with hit-tax opening, then escalating multi-hit and wound pressure. Low-cost and multi-hit answers are preferred.",
        "semantic_tags": ["boss", "damage_cap", "multi_hit", "wound_pressure"],
        "combat_tags": ["boss", "frontline", "anti_big_hit"],
        "static_traits": [
            _trait("hit_tax_opening", "Early hits are heavily damped; many small attacks remove the protection faster.", severity="high"),
            _trait("wound_pressure", "Dangerous turns add wound clutter on top of damage.", severity="medium"),
        ],
        "phase_rules": [
            _phase_rule("on_turn_start", "opening_cycle", "damage_cap", "Starts in a slippery / reduced-damage opening state.", state="opening", severity="high"),
        ],
        "danger_profile": _danger(burst=4, attrition=3, scaling=3, phase_complexity=4, volatility=4, target_priority=4),
    },
    "MONSTER.CEREMONIAL_BEAST": {
        "summary": "Threshold phase boss. Force the HP breakpoint on your terms, exploit the stun window, then respect the single-card lockdown phase.",
        "semantic_tags": ["boss", "threshold_stun", "phase_shift", "card_lock"],
        "combat_tags": ["boss", "phase_shift", "burst_window"],
        "static_traits": [
            _trait("threshold_stun", "Crossing the HP breakpoint creates a temporary stun / vulnerability window.", severity="high"),
            _trait("one_card_lock", "Later phase punishes decks that need long action chains.", severity="high"),
        ],
        "phase_rules": [
            _phase_rule("on_hp_threshold", "hp_le_150", "stun_self", "At or below 150 HP it stuns and transitions phases.", threshold=150, state="stunned", severity="high"),
            _phase_rule("on_phase_start", "phase_two", "card_play_limit", "Phase two restricts the number of cards you can play each turn.", state="phase_two", severity="high"),
        ],
        "danger_profile": _danger(burst=4, attrition=4, scaling=3, phase_complexity=5, volatility=4, target_priority=4),
        "target_priority_hints": [
            _hint("high", "Bank burst so the threshold is crossed cleanly and the stun turn becomes your best tempo window."),
        ],
    },
    "MONSTER.LAGAVULIN_MATRIARCH": {
        "summary": "Sleeping boss with a setup window, then long-fight stat suppression and scaling. Slow stat-dependent decks get punished hard.",
        "semantic_tags": ["boss", "sleep_start", "stat_drain", "scaling"],
        "combat_tags": ["boss", "phase_shift", "anti_long_fight"],
        "static_traits": [
            _trait("sleeping_opening", "Offers a temporary setup window before waking.", severity="medium"),
            _trait("stat_drain", "Late cycle reduces your stats while increasing its own damage profile.", severity="high"),
        ],
        "phase_rules": [
            _phase_rule("on_turn_start", "sleep_break", "wake", "Starts asleep, then enters an active cycle once the opening window ends.", state="awake", severity="medium"),
        ],
        "danger_profile": _danger(burst=3, attrition=4, scaling=4, debuff_pressure=4, phase_complexity=3, target_priority=4),
    },
    "MONSTER.SOUL_FYSH": {
        "summary": "Status-pressure boss. Beckon taxes hand space and HP, while intangible turns punish wasted burst.",
        "semantic_tags": ["boss", "status_pressure", "intangible", "resource_tax"],
        "combat_tags": ["boss", "resource_tax", "phase_shift"],
        "static_traits": [
            _trait("beckon_tax", "Specific status cards must be cleared or they convert into unavoidable HP loss.", severity="high"),
            _trait("intangible_windows", "Large damage is poor into intangible turns; use those turns for setup instead.", severity="high"),
        ],
        "reactive_triggers": [
            _reactive("on_turn_end", "status_left_in_hand", "hp_loss", "Leaving Beckon unresolved causes direct HP loss.", effect_amount=6, severity="high"),
        ],
        "phase_rules": [
            _phase_rule("on_turn_start", "intangible_turn", "gain_intangible", "Alternates into intangible windows that redirect your best damage timing.", severity="high"),
        ],
        "danger_profile": _danger(attrition=5, debuff_pressure=4, phase_complexity=4, volatility=3, target_priority=4),
    },
    "MONSTER.WATERFALL_GIANT": {
        "summary": "Speed-check boss with a lethal death explosion that grows over time. The last turn can be the real kill shot.",
        "semantic_tags": ["boss", "death_explosion", "speed_check", "countdown_like"],
        "combat_tags": ["boss", "speed_check", "burst_window"],
        "static_traits": [
            _trait("death_explosion", "Killing the body does not immediately end danger; the explosion turn still matters.", severity="high"),
            _trait("explosion_scaling", "Deathburst grows if the fight drags.", severity="high"),
        ],
        "reactive_triggers": [
            _reactive("on_death", "killed", "retaliate", "Death triggers a delayed explosion rather than an immediate safe end state.", severity="high"),
        ],
        "danger_profile": _danger(burst=5, scaling=4, volatility=5, target_priority=4),
    },
    "MONSTER.THE_INSATIABLE": {
        "summary": "DPS / countdown boss. You are racing both normal damage and a kill timer that must be extended with dedicated cards.",
        "semantic_tags": ["boss", "countdown", "dps_check", "resource_tax"],
        "combat_tags": ["boss", "speed_check", "resource_tax"],
        "static_traits": [
            _trait("doom_clock", "A non-standard death timer pressures you even if block is stable.", severity="high"),
            _trait("escape_card_tax", "Dedicated delay cards must be paid for without losing the damage race.", severity="high"),
        ],
        "phase_rules": [
            _phase_rule("on_turn_start", "countdown_active", "countdown_tick", "Each turn advances an external kill timer unless delayed.", severity="high"),
        ],
        "danger_profile": _danger(burst=4, scaling=4, phase_complexity=4, volatility=5, target_priority=5),
    },
    "MONSTER.KNOWLEDGE_DEMON": {
        "summary": "Boss debuff menu fight. You repeatedly accept the least-bad long-term constraint while still producing pressure.",
        "semantic_tags": ["boss", "debuff_menu", "attrition", "build_check"],
        "combat_tags": ["boss", "debuff", "attrition"],
        "static_traits": [
            _trait("choice_debuffs", "Imposes build-specific constraints rather than one generic debuff.", severity="high"),
        ],
        "danger_profile": _danger(attrition=5, scaling=3, debuff_pressure=5, volatility=3, target_priority=4),
    },
    "MONSTER.DOOR": {
        "summary": "Boss gate object. Exists to delay access to Doormaker and scales the cycle if you fail to convert the exposed window.",
        "semantic_tags": ["boss_component", "gate", "tempo_tax"],
        "combat_tags": ["boss_component", "gate"],
        "static_traits": [
            _trait("gating_body", "Fight progression is blocked until the door is destroyed.", severity="high"),
        ],
        "phase_rules": [
            _phase_rule("on_death", "door_destroyed", "reveal_boss", "Destroying the door exposes the boss and starts the vulnerable window.", severity="high"),
        ],
        "danger_profile": _danger(attrition=3, phase_complexity=3, target_priority=4),
    },
    "MONSTER.DOORMAKER": {
        "summary": "Boss with gate cycles. Break the door, exploit the exposed stun / burst window, then expect a retreat and repeat with a thicker gate.",
        "semantic_tags": ["boss", "gate_cycle", "burst_window", "reset_loop"],
        "combat_tags": ["boss", "phase_shift", "burst_window"],
        "static_traits": [
            _trait("retreat_cycle", "Can hide behind a new door after the exposed window closes.", severity="high"),
        ],
        "phase_rules": [
            _phase_rule("on_phase_start", "door_destroyed", "stun_self", "Door break opens a brief stun / damage window on the boss.", state="exposed", severity="high"),
            _phase_rule("on_turn_end", "cycle_reset", "spawn_gate", "Later loops create new, often stronger doors.", state="retreated", severity="high"),
        ],
        "danger_profile": _danger(burst=5, attrition=3, scaling=4, phase_complexity=5, volatility=4, target_priority=5),
    },
    "MONSTER.QUEEN": {
        "summary": "Boss backline controller. Applies binding / long debuffs while Torch Head Amalgam does much of the direct damage.",
        "semantic_tags": ["boss", "binding", "debuff", "backline"],
        "combat_tags": ["boss", "backline", "controller"],
        "static_traits": [
            _trait("binding_control", "Early plays each turn can be constrained by binding effects.", severity="high"),
            _trait("long_debuffs", "Can stack outsized Frail / Vulnerable / Weak pressure.", severity="high"),
        ],
        "danger_profile": _danger(attrition=4, debuff_pressure=5, target_priority=3),
    },
    "MONSTER.TORCH_HEAD_AMALGAM": {
        "summary": "Primary direct-damage partner in the Queen fight. Removing it first often stabilizes the encounter dramatically.",
        "semantic_tags": ["boss_duo", "multi_hit", "direct_damage"],
        "combat_tags": ["boss_duo", "frontline", "burst"],
        "static_traits": [
            _trait("primary_damage_source", "Usually the immediate HP loss source in the duo fight.", severity="high"),
        ],
        "danger_profile": _danger(burst=5, volatility=4, target_priority=5),
    },
    "MONSTER.TEST_SUBJECT": {
        "summary": "Three-phase boss. Stage one punishes skill spam, stage two punishes unblocked damage with wounds, stage three alternates intangible and burn pressure.",
        "semantic_tags": ["boss", "multi_phase", "intangible", "wound_pressure"],
        "combat_tags": ["boss", "phase_shift", "attrition"],
        "static_traits": [
            _trait("skill_punish_phase", "Early phase can reward restraint on unnecessary skills.", severity="medium"),
            _trait("wound_phase", "Middle phase punishes taking chip through block.", severity="high"),
            _trait("intangible_phase", "Final phase has turns where burst is heavily devalued.", severity="high"),
        ],
        "phase_rules": [
            _phase_rule("on_hp_threshold", "phase_change", "phase_shift", "Progresses through multiple HP-gated phases.", severity="high"),
            _phase_rule("on_turn_start", "phase_three_intangible_turn", "gain_intangible", "Final phase alternates into intangible turns.", state="phase_three", severity="high"),
        ],
        "danger_profile": _danger(burst=5, attrition=5, scaling=3, debuff_pressure=3, phase_complexity=5, volatility=5, target_priority=5),
    },
})


def _normalize_list(values: Iterable[Any] | None) -> list[Any]:
    result: list[Any] = []
    for value in values or ():
        if value in (None, "", [], {}):
            continue
        result.append(value)
    return result


def _dedupe_strings(values: Iterable[Any]) -> list[str]:
    result: list[str] = []
    seen: set[str] = set()
    for value in values:
        text = str(value or "").strip()
        if not text or text in seen:
            continue
        seen.add(text)
        result.append(text)
    return result


def _dedupe_dicts(values: Iterable[dict[str, Any]]) -> list[dict[str, Any]]:
    result: list[dict[str, Any]] = []
    seen: set[str] = set()
    for value in values:
        try:
            key = json.dumps(value, sort_keys=True, ensure_ascii=False)
        except TypeError:
            continue
        if key in seen:
            continue
        seen.add(key)
        result.append(value)
    return result


def _generic_summary(enemy_id: str) -> str:
    tail = enemy_id.split(".", 1)[-1].lower()
    if "slime" in tail:
        return "Simple frontline attacker; mostly a baseline damage / block check."
    if "cultist" in tail or "priest" in tail:
        return "Support / scaling enemy that gets more annoying if allowed repeated turns."
    if "construct" in tail or "shield" in tail:
        return "Defensive / durable enemy that tends to stretch the fight."
    if "rat" in tail or "hopper" in tail or "crawler" in tail:
        return "Fast damage body that punishes weak early tempo."
    if "egg" in tail:
        return "Low direct damage, but often represents future board pressure if left alive."
    if "queen" in tail or "nexus" in tail or "fabricator" in tail or "operator" in tail:
        return "Engine enemy whose value comes from persistent board or support pressure."
    return "Generic enemy metadata bootstrap entry. Prefer live powers / intent for exact planning."

def _base_metadata(enemy_id: str) -> dict[str, Any]:
    title = _pretty_title(enemy_id)
    tail = enemy_id.split(".", 1)[-1].lower()
    semantic_tags: list[str] = []
    combat_tags: list[str] = []
    static_traits: list[dict[str, Any]] = []
    reactive_triggers: list[dict[str, Any]] = []
    phase_rules: list[dict[str, Any]] = []
    target_priority_hints: list[dict[str, Any]] = []
    danger = _danger()

    def add_semantic(*tags: str) -> None:
        semantic_tags.extend(tags)

    def add_combat(*tags: str) -> None:
        combat_tags.extend(tags)

    def boost(**updates: int) -> None:
        for key, value in updates.items():
            if key in danger:
                danger[key] = max(int(danger.get(key, 0)), int(value))

    if any(token in tail for token in ("queen", "doormaker", "insatiable", "demon", "fysh", "matriarch", "giant", "test_subject", "vantom", "beast")):
        add_semantic("boss")
        add_combat("boss")
        boost(target_priority=4)
    if any(token in tail for token in ("nexus", "progenitor", "queen", "egg", "fabricator", "operator", "priest", "cultist")):
        add_semantic("summoner")
        add_combat("engine")
        static_traits.append(_trait("engine_body", "Acts as a pressure engine rather than just a damage body.", severity="medium"))
        boost(summon_pressure=4, target_priority=4)
    if any(token in tail for token in ("spiny", "thorn", "spike")):
        add_semantic("retaliation")
        add_combat("retaliation")
        reactive_triggers.append(_reactive("on_hit", "contact", "retaliate", "Likely punishes contact hits / thorns-style.", severity="medium"))
        boost(retaliation=5, attrition=3, target_priority=4)
    if any(token in tail for token in ("slug", "shield", "construct", "exoskeleton", "clam", "door")):
        add_semantic("durable")
        add_combat("tank")
        static_traits.append(_trait("durable_body", "Likely to stretch the fight through HP, block, or mitigation.", severity="medium"))
        boost(attrition=4)
    if any(token in tail for token in ("wurm", "crawler", "byrd", "beast", "eel", "knight", "berserker", "rocket", "crusher")):
        add_semantic("burst")
        add_combat("frontline")
        boost(burst=4)
    if any(token in tail for token in ("cultist", "priest", "demon", "magistrate", "beetle", "gardener")):
        add_semantic("debuff")
        add_combat("controller")
        boost(debuff_pressure=4, attrition=3)
    if any(token in tail for token in ("matriarch", "doormaker", "door", "subject", "prism", "vantom", "insatiable", "obscura")):
        add_semantic("phase_shift")
        add_combat("phase_shift")
        boost(phase_complexity=4, volatility=4)
    if any(token in tail for token in ("lost", "forgotten", "follower", "queen", "rocket", "crusher")):
        add_semantic("multi_actor")
        add_combat("duo")
        boost(attrition=3)
    if any(token in tail for token in ("slug", "nexus", "byrdonis", "crawler", "cultist", "matriarch")):
        add_semantic("scaling")
        boost(scaling=4)
    if "egg" in tail:
        phase_rules.append(_phase_rule("on_death", "egg_destroyed", "spawn", "Destroying the egg may not fully remove future pressure.", severity="medium"))
        boost(summon_pressure=4, target_priority=4)
    if "door" == tail:
        phase_rules.append(_phase_rule("on_death", "door_destroyed", "reveal_boss", "Destroying the door likely exposes the real threat.", severity="high"))
    if "progenitor" in tail:
        reactive_triggers.append(_reactive("on_turn_end", "alive", "summon", "If left alive it likely keeps adding board presence.", severity="high"))
    if "prism" in tail or "split" in tail:
        phase_rules.append(_phase_rule("on_hp_threshold", "threshold_crossed", "split", "Threshold damage may create new bodies or phase pressure.", severity="medium"))
    if danger.get("target_priority", 0) >= 4:
        target_priority_hints.append(_hint("medium", "Usually worth prioritizing if your deck struggles with this enemy's core axis."))

    return {
        "title": title,
        "summary": _generic_summary(enemy_id),
        "semantic_tags": _dedupe_strings(semantic_tags),
        "combat_tags": _dedupe_strings(combat_tags),
        "static_traits": _dedupe_dicts(static_traits),
        "reactive_triggers": _dedupe_dicts(reactive_triggers),
        "phase_rules": _dedupe_dicts(phase_rules),
        "danger_profile": danger,
        "target_priority_hints": _dedupe_dicts(target_priority_hints),
    }


def _collect_known_enemy_ids(repo_root: Path) -> list[str]:
    enemy_ids: set[str] = set()
    for jsonl_path in repo_root.rglob("*.jsonl"):
        try:
            with jsonl_path.open("r", encoding="utf-8") as handle:
                for line in handle:
                    raw = line.strip()
                    if not raw:
                        continue
                    try:
                        row = json.loads(raw)
                    except Exception:
                        continue
                    for enemy_id in row.get("monster_ids") or []:
                        if isinstance(enemy_id, str) and enemy_id:
                            enemy_ids.add(enemy_id)
        except Exception:
            continue
    if not enemy_ids:
        enemy_ids.update(_DISCOVERY_FALLBACK_IDS)
    return sorted(enemy_ids)


def _synthesize_trait_tokens(metadata: dict[str, Any]) -> list[dict[str, Any]]:
    tokens: list[dict[str, Any]] = []
    for item in metadata.get("static_traits") or []:
        if isinstance(item, dict):
            tokens.append(item)
    for item in metadata.get("reactive_triggers") or []:
        if isinstance(item, dict):
            tokens.append(
                _trait(
                    str(item.get("effect_type") or item.get("trait") or "reactive"),
                    str(item.get("description") or item.get("effect_type") or "reactive"),
                    trigger_type=item.get("trigger_type"),
                    condition=item.get("condition"),
                    effect_amount=item.get("effect_amount"),
                    severity=item.get("severity"),
                    category="reactive",
                )
            )
    for item in metadata.get("phase_rules") or []:
        if isinstance(item, dict):
            tokens.append(
                _trait(
                    str(item.get("effect_type") or item.get("trait") or "phase"),
                    str(item.get("description") or item.get("effect_type") or "phase"),
                    trigger_type=item.get("trigger_type"),
                    condition=item.get("condition"),
                    threshold=item.get("threshold"),
                    state=item.get("state"),
                    severity=item.get("severity"),
                    category="phase",
                )
            )
    return _dedupe_dicts(tokens)


def build_enemy_registry(repo_root: Path | None = None) -> dict[str, dict[str, Any]]:
    package_root = Path(__file__).resolve().parent
    effective_root = repo_root.resolve() if repo_root else package_root
    enemy_ids = _collect_known_enemy_ids(effective_root)
    registry: dict[str, dict[str, Any]] = {}
    for enemy_id in enemy_ids:
        metadata = _base_metadata(enemy_id)
        metadata = _deep_merge_dict(metadata, _CURATED_OVERRIDES.get(enemy_id, {}))
        metadata["title"] = str(metadata.get("title") or _pretty_title(enemy_id))
        metadata["summary"] = str(metadata.get("summary") or _generic_summary(enemy_id))
        metadata["semantic_tags"] = _dedupe_strings(metadata.get("semantic_tags") or [])
        metadata["combat_tags"] = _dedupe_strings(metadata.get("combat_tags") or [])
        metadata["static_traits"] = _dedupe_dicts(_normalize_list(metadata.get("static_traits")))
        metadata["reactive_triggers"] = _dedupe_dicts(_normalize_list(metadata.get("reactive_triggers")))
        metadata["phase_rules"] = _dedupe_dicts(_normalize_list(metadata.get("phase_rules")))
        metadata["target_priority_hints"] = _dedupe_dicts(_normalize_list(metadata.get("target_priority_hints")))
        metadata["trait_tokens"] = _synthesize_trait_tokens(metadata)
        registry[enemy_id] = metadata
    return dict(sorted(registry.items(), key=lambda item: item[0]))


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate structured enemy semantics bootstrap registry.")
    parser.add_argument("--repo-root", type=Path, default=None, help="Optional repository root to scan for monster ids.")
    parser.add_argument("--output", type=Path, default=_DEFAULT_OUTPUT, help=f"Output path (default: {_DEFAULT_OUTPUT})")
    args = parser.parse_args()

    registry = build_enemy_registry(args.repo_root)
    output_path = resolve_generated_game_data_output(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_bytes(
        (json.dumps(registry, ensure_ascii=False, indent=2, sort_keys=True) + "\n").encode("utf-8")
    )
    print(f"Wrote {len(registry)} enemy entries to {output_path}")


if __name__ == "__main__":
    main()
