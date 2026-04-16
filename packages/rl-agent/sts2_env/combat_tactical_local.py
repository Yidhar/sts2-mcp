"""Lightweight tactical combat analysis for fixed-slot combat search.

Adapted from the practical combat turn solver pattern used in
``wcz233/sts2-MuZero``. The goal here is intentionally narrow:

- identify obvious lethal / best-kill roots
- estimate the minimum required block after the best kill line
- expose a compact preferred-root-action signal for prior biasing
"""

from __future__ import annotations

import re
from dataclasses import dataclass


_ALL_ENEMY_TARGET_TYPES = {"ALLENEMIES"}
_UNSUPPORTED_COST = {"X", "?", ""}
_DAMAGE_CAP_ONE_TOKENS = ("INTANGIBLE", "INTANGIBLEPLAYER")
_THORNS_TOKENS = ("THORNS",)
_VULNERABLE_TOKENS = ("VULNERABLE",)

_ENGLISH_MULTI_DAMAGE_PATTERNS = (
    re.compile(r"deal\s*(\d+)\s*damage\s*(\d+)\s*times"),
    re.compile(r"deal\s*(\d+)\s*damage.*?(\d+)\s*times"),
)
_ENGLISH_SIMPLE_DAMAGE = re.compile(r"deal\s*(\d+)\s*damage")
_ENGLISH_VULNERABLE = re.compile(r"apply\s*(\d+)\s*vulnerable")


@dataclass(frozen=True)
class TacticalAction:
    card_index: int
    target_entity_id: str = ""


@dataclass(frozen=True)
class CombatTacticalAnalysis:
    available: bool = False
    lethal_exists: bool = False
    min_required_block_after_best_kill: int = 0
    best_sequence: tuple[TacticalAction, ...] = ()
    best_root_actions: tuple[TacticalAction, ...] = ()
    lethal_required_block: int = 0
    lethal_sequence: tuple[TacticalAction, ...] = ()
    lethal_root_actions: tuple[TacticalAction, ...] = ()


@dataclass(frozen=True)
class _ParsedAttackCard:
    card_index: int
    cost: int
    damage_per_hit: int
    hit_count: int
    vulnerable_amount: int
    targets_all_enemies: bool


@dataclass(frozen=True)
class _EnemyState:
    entity_id: str
    hp: int
    block: int
    incoming_damage: int
    vulnerable: int
    thorns: int
    damage_cap_per_hit: int


@dataclass(frozen=True)
class _Plan:
    lethal: bool = False
    remaining_incoming: int = 0
    remaining_enemy_count: int = 0
    remaining_enemy_total_hp: int = 0
    additional_retaliation: int = 0
    sequence: tuple[TacticalAction, ...] = ()

    @property
    def required_block(self) -> int:
        return self.remaining_incoming + self.additional_retaliation


@dataclass(frozen=True)
class _SolveResult:
    best_plan: _Plan
    best_lethal_plan: _Plan | None


def analyze_local_combat_turn(raw_obs: dict | None) -> CombatTacticalAnalysis:
    if not isinstance(raw_obs, dict):
        return CombatTacticalAnalysis()
    if not _combat_action_window_ready(raw_obs):
        return CombatTacticalAnalysis()

    combat = raw_obs.get("combat") if isinstance(raw_obs.get("combat"), dict) else {}
    player = raw_obs.get("player") if isinstance(raw_obs.get("player"), dict) else {}
    enemies = tuple(
        _normalize_enemy(enemy)
        for enemy in combat.get("enemies", [])
        if isinstance(enemy, dict)
    )
    living_enemies = tuple(enemy for enemy in enemies if enemy.hp > 0)
    if not living_enemies:
        return CombatTacticalAnalysis(available=True, lethal_exists=True)

    current_energy = _player_energy_units(combat)
    hand_cards = combat.get("hand") if isinstance(combat.get("hand"), list) else []
    attack_cards = tuple(
        parsed
        for parsed in (
            _parse_attack_card(card, current_energy)
            for card in hand_cards
            if isinstance(card, dict)
        )
        if parsed is not None
    )
    initial_plan = _evaluate_leaf_plan(living_enemies)
    if not attack_cards:
        return CombatTacticalAnalysis(
            available=True,
            lethal_exists=False,
            min_required_block_after_best_kill=initial_plan.required_block,
        )

    memo: dict[tuple[int, int, tuple[tuple[object, ...], ...]], _SolveResult] = {}
    initial_mask = (1 << len(attack_cards)) - 1
    solved = _solve(initial_mask, current_energy, living_enemies, attack_cards, memo)
    best_plan = solved.best_plan
    lethal_plan = solved.best_lethal_plan
    return CombatTacticalAnalysis(
        available=True,
        lethal_exists=lethal_plan is not None,
        min_required_block_after_best_kill=best_plan.required_block,
        best_sequence=best_plan.sequence,
        best_root_actions=best_plan.sequence[:1],
        lethal_required_block=(lethal_plan.required_block if lethal_plan is not None else best_plan.required_block),
        lethal_sequence=(lethal_plan.sequence if lethal_plan is not None else ()),
        lethal_root_actions=(lethal_plan.sequence[:1] if lethal_plan is not None else ()),
    )


def _combat_action_window_ready(raw_obs: dict) -> bool:
    combat = raw_obs.get("combat") if isinstance(raw_obs.get("combat"), dict) else {}
    if not isinstance(combat, dict):
        return False
    if not bool(combat.get("play_phase", True)):
        return False
    if not bool(combat.get("can_act", True)):
        return False
    hand = combat.get("hand")
    return isinstance(hand, list)


def _player_energy_units(combat: dict) -> int:
    try:
        return max(0, int(float(combat.get("energy", 0) or 0)))
    except (TypeError, ValueError):
        return 0


def _normalize_enemy(enemy: dict) -> _EnemyState:
    powers = enemy.get("powers") if isinstance(enemy.get("powers"), list) else []
    return _EnemyState(
        entity_id=str(enemy.get("entity_id", enemy.get("id", enemy.get("name", ""))) or ""),
        hp=max(0, _to_int(enemy.get("hp", enemy.get("current_hp")))),
        block=max(0, _to_int(enemy.get("block"))),
        incoming_damage=_enemy_intent_damage(enemy),
        vulnerable=max(0, _status_amount(powers, _VULNERABLE_TOKENS)),
        thorns=max(0, _status_amount(powers, _THORNS_TOKENS)),
        damage_cap_per_hit=(1 if _has_status(powers, _DAMAGE_CAP_ONE_TOKENS) else 0),
    )


def _enemy_intent_damage(enemy: dict) -> int:
    intent = enemy.get("intent") if isinstance(enemy.get("intent"), dict) else {}
    total_damage = intent.get("total_damage")
    if total_damage is not None:
        return max(0, _to_int(total_damage))
    damage_per_hit = max(0, _to_int(intent.get("damage_per_hit")))
    repeats = max(1, _to_int(intent.get("repeats")))
    return damage_per_hit * repeats


def _status_amount(statuses: list[object], token_groups: tuple[str, ...]) -> int:
    total = 0
    for status in statuses:
        if not isinstance(status, dict):
            continue
        token = str(status.get("id", status.get("name", "")) or "").upper()
        if not any(group in token for group in token_groups):
            continue
        total += max(0, _to_int(status.get("amount")))
    return total


def _has_status(statuses: list[object], token_groups: tuple[str, ...]) -> bool:
    return any(
        isinstance(status, dict) and any(group in str(status.get("id", status.get("name", "")) or "").upper() for group in token_groups)
        for status in statuses
    )


def _parse_attack_card(card: dict, current_energy: int) -> _ParsedAttackCard | None:
    if str(card.get("type", card.get("card_type", "")) or "") != "Attack":
        return None
    if not bool(card.get("can_play", True)):
        return None
    card_index = card.get("index")
    if not isinstance(card_index, int) or card_index < 0:
        return None
    cost_token = str(card.get("cost", "") or "").strip().upper()
    if cost_token in _UNSUPPORTED_COST:
        return None
    try:
        cost = max(0, int(float(cost_token)))
    except (TypeError, ValueError):
        return None
    if cost > current_energy:
        return None

    damage_per_hit = _to_int(_preview_metric(card, "damage_per_hit"))
    hit_count = max(1, _to_int(_preview_metric(card, "hits")))
    if damage_per_hit <= 0:
        preview_damage = _to_int(_preview_metric(card, "damage"))
        if hit_count > 0:
            damage_per_hit = max(0, preview_damage // hit_count) if preview_damage > 0 else 0
    if damage_per_hit <= 0:
        damage_per_hit, hit_count = _parse_damage_profile(card)
    if damage_per_hit <= 0 or hit_count <= 0:
        return None

    target_token = str(card.get("target_type", card.get("target", "")) or "").strip().upper()
    description = str(card.get("description", card.get("rules_text", card.get("text", ""))) or "").strip().lower()
    targets_all_enemies = (
        target_token in _ALL_ENEMY_TARGET_TYPES
        or "all enemies" in description
    )
    vulnerable_amount = max(_to_int(_preview_metric(card, "vulnerable")), _parse_vulnerable_amount(card))
    return _ParsedAttackCard(
        card_index=card_index,
        cost=cost,
        damage_per_hit=damage_per_hit,
        hit_count=hit_count,
        vulnerable_amount=vulnerable_amount,
        targets_all_enemies=targets_all_enemies,
    )


def _preview_metric(card: dict, key: str) -> float:
    effect_preview = card.get("effect_preview") if isinstance(card.get("effect_preview"), dict) else {}
    value = effect_preview.get(key, card.get(key))
    try:
        return float(value or 0.0)
    except (TypeError, ValueError):
        return 0.0


def _parse_damage_profile(card: dict) -> tuple[int, int]:
    text = str(card.get("description", card.get("rules_text", card.get("text", ""))) or "").strip().lower()
    for pattern in _ENGLISH_MULTI_DAMAGE_PATTERNS:
        match = pattern.search(text)
        if match is not None:
            return int(match.group(1)), int(match.group(2))
    match = _ENGLISH_SIMPLE_DAMAGE.search(text)
    if match is not None:
        return int(match.group(1)), 1
    return 0, 0


def _parse_vulnerable_amount(card: dict) -> int:
    text = str(card.get("description", card.get("rules_text", card.get("text", ""))) or "").strip().lower()
    match = _ENGLISH_VULNERABLE.search(text)
    if match is not None:
        return int(match.group(1))
    return 0


def _solve(
    available_mask: int,
    energy_left: int,
    enemies: tuple[_EnemyState, ...],
    cards: tuple[_ParsedAttackCard, ...],
    memo: dict[tuple[int, int, tuple[tuple[object, ...], ...]], _SolveResult],
) -> _SolveResult:
    key = (available_mask, energy_left, _enemy_signature(enemies))
    cached = memo.get(key)
    if cached is not None:
        return cached

    best_plan = _evaluate_leaf_plan(enemies)
    best_lethal_plan = best_plan if best_plan.lethal else None
    for card_position, card in enumerate(cards):
        card_bit = 1 << card_position
        if available_mask & card_bit == 0 or card.cost > energy_left:
            continue
        if card.targets_all_enemies:
            next_enemies, retaliation = _apply_attack(enemies, card, None)
            child = _solve(available_mask ^ card_bit, energy_left - card.cost, next_enemies, cards, memo)
            best_plan = _better_general(_prepend_step(child.best_plan, card, retaliation, ""), best_plan)
            if child.best_lethal_plan is not None:
                best_lethal_plan = _better_lethal(_prepend_step(child.best_lethal_plan, card, retaliation, ""), best_lethal_plan)
            continue
        for target_index, enemy in enumerate(enemies):
            if enemy.hp <= 0:
                continue
            next_enemies, retaliation = _apply_attack(enemies, card, target_index)
            child = _solve(available_mask ^ card_bit, energy_left - card.cost, next_enemies, cards, memo)
            best_plan = _better_general(_prepend_step(child.best_plan, card, retaliation, enemy.entity_id), best_plan)
            if child.best_lethal_plan is not None:
                best_lethal_plan = _better_lethal(
                    _prepend_step(child.best_lethal_plan, card, retaliation, enemy.entity_id),
                    best_lethal_plan,
                )
    result = _SolveResult(best_plan=best_plan, best_lethal_plan=best_lethal_plan)
    memo[key] = result
    return result


def _enemy_signature(enemies: tuple[_EnemyState, ...]) -> tuple[tuple[object, ...], ...]:
    return tuple(
        (
            enemy.entity_id,
            enemy.hp,
            enemy.block,
            enemy.incoming_damage,
            enemy.vulnerable,
            enemy.thorns,
            enemy.damage_cap_per_hit,
        )
        for enemy in enemies
    )


def _evaluate_leaf_plan(enemies: tuple[_EnemyState, ...]) -> _Plan:
    living = tuple(enemy for enemy in enemies if enemy.hp > 0)
    return _Plan(
        lethal=not living,
        remaining_incoming=sum(enemy.incoming_damage for enemy in living),
        remaining_enemy_count=len(living),
        remaining_enemy_total_hp=sum(enemy.hp + enemy.block for enemy in living),
        additional_retaliation=0,
        sequence=(),
    )


def _apply_attack(
    enemies: tuple[_EnemyState, ...],
    card: _ParsedAttackCard,
    target_index: int | None,
) -> tuple[tuple[_EnemyState, ...], int]:
    next_enemies: list[_EnemyState] = []
    retaliation = 0
    for index, enemy in enumerate(enemies):
        if enemy.hp <= 0:
            next_enemies.append(enemy)
            continue
        should_hit = card.targets_all_enemies or index == target_index
        if not should_hit:
            next_enemies.append(enemy)
            continue

        damage_per_hit = card.damage_per_hit
        if enemy.vulnerable > 0:
            damage_per_hit = int(round(damage_per_hit * 1.5))
        if enemy.damage_cap_per_hit > 0:
            damage_per_hit = min(damage_per_hit, enemy.damage_cap_per_hit)

        total_damage = 0
        remaining_block = enemy.block
        hp = enemy.hp
        for _ in range(card.hit_count):
            hit_damage = damage_per_hit
            if remaining_block > 0:
                blocked = min(remaining_block, hit_damage)
                remaining_block -= blocked
                hit_damage -= blocked
            if hit_damage > 0:
                hp = max(0, hp - hit_damage)
            total_damage += hit_damage
        if total_damage > 0:
            retaliation += enemy.thorns
        vulnerable = enemy.vulnerable + card.vulnerable_amount if hp > 0 else 0
        next_enemies.append(
            _EnemyState(
                entity_id=enemy.entity_id,
                hp=hp,
                block=remaining_block,
                incoming_damage=enemy.incoming_damage if hp > 0 else 0,
                vulnerable=vulnerable,
                thorns=enemy.thorns if hp > 0 else 0,
                damage_cap_per_hit=enemy.damage_cap_per_hit if hp > 0 else 0,
            )
        )
    return tuple(next_enemies), retaliation


def _prepend_step(plan: _Plan, card: _ParsedAttackCard, retaliation: int, target_entity_id: str) -> _Plan:
    return _Plan(
        lethal=plan.lethal,
        remaining_incoming=plan.remaining_incoming,
        remaining_enemy_count=plan.remaining_enemy_count,
        remaining_enemy_total_hp=plan.remaining_enemy_total_hp,
        additional_retaliation=plan.additional_retaliation + retaliation,
        sequence=(TacticalAction(card.card_index, target_entity_id),) + plan.sequence,
    )


def _better_general(candidate: _Plan, current: _Plan) -> _Plan:
    candidate_key = (
        -candidate.remaining_enemy_count,
        -candidate.remaining_enemy_total_hp,
        -candidate.required_block,
        -len(candidate.sequence),
    )
    current_key = (
        -current.remaining_enemy_count,
        -current.remaining_enemy_total_hp,
        -current.required_block,
        -len(current.sequence),
    )
    return candidate if candidate_key > current_key else current


def _better_lethal(candidate: _Plan, current: _Plan | None) -> _Plan:
    if current is None:
        return candidate
    candidate_key = (-candidate.required_block, -len(candidate.sequence))
    current_key = (-current.required_block, -len(current.sequence))
    return candidate if candidate_key > current_key else current


def _to_int(value) -> int:
    try:
        return int(float(value or 0))
    except (TypeError, ValueError):
        return 0
