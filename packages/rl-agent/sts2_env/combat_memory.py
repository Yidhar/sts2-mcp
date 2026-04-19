"""Per-combat per-enemy temporal history tracker.

Produces content-agnostic delta signals the observation encoder injects
into free numeric slots on enemy / power / intent / global tokens. The
policy can then recognise mechanic behaviour (delayed bursts, enrage
scaling, split spawns, threshold phases) without relying on keyword
matching of enemy / power names, so it generalises to unseen bosses
and mod content as long as the bridge surfaces numeric state.
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field
import math
from typing import Any

# ---- Normalisation anchors (reused for delta slots) ----
_LOG1P_200 = math.log1p(200.0)
_LOG1P_100 = math.log1p(100.0)
_HP_DELTA_TURN_WINDOW = 3
_TURN_NORM = 20.0
_INTENT_TURN_NORM = 5.0
_POWER_TURN_NORM = 10.0


def _float(value: Any, default: float = 0.0) -> float:
    if value is None:
        return default
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _log_norm(value: float, anchor: float) -> float:
    if value <= 0.0 or anchor <= 0.0:
        return 0.0
    return min(math.log1p(value) / anchor, 1.0)


def _signed_log_norm(value: float, anchor: float) -> float:
    if value == 0.0 or anchor <= 0.0:
        return 0.0
    magnitude = min(math.log1p(abs(value)) / anchor, 1.0)
    return magnitude if value > 0 else -magnitude


def _clip_signed(value: float) -> float:
    if value > 1.0:
        return 1.0
    if value < -1.0:
        return -1.0
    return value


def _enemy_key(enemy: dict[str, Any] | None, fallback_index: int) -> str:
    if isinstance(enemy, dict):
        for key in ("combat_id", "id", "model_id", "name"):
            value = enemy.get(key)
            if value not in (None, ""):
                return str(value)
    return f"idx_{fallback_index}"


def _power_key(power: dict[str, Any] | None) -> str:
    if not isinstance(power, dict):
        return ""
    for key in ("id", "power_id", "type", "key"):
        value = power.get(key)
        if value not in (None, ""):
            return str(value)
    title = power.get("title")
    if title not in (None, ""):
        return f"title::{title}"
    return ""


def _intent_signature(intent: dict[str, Any] | None) -> str:
    if not isinstance(intent, dict):
        return ""
    parts: list[str] = []
    for key in ("intent_type", "type", "intent_class", "state_id"):
        value = intent.get(key)
        if value not in (None, ""):
            parts.append(f"{key}={value}")
            break
    repeats = intent.get("repeats")
    if repeats not in (None, ""):
        parts.append(f"r={repeats}")
    label = intent.get("label") or intent.get("title")
    if label and not parts:
        parts.append(f"t={label}")
    return "|".join(parts)


def _self_inflicted_hp_loss_cumulative(obs: dict[str, Any] | None) -> float:
    """Bridge-side cumulative counter of player-initiated HP loss in this combat.

    Source: combat.self_inflicted_hp_loss_cumulative, populated by the bridge
    from play_card actions that declare effect_preview.hp_loss > 0 (Offering,
    Bloodletting, Hemokinesis, etc.). Monotonically increases within a combat,
    resets on combat_reset. Returns 0.0 when the field is absent so observations
    from older bridge builds keep working.
    """
    if not isinstance(obs, dict):
        return 0.0
    combat = obs.get("combat") if isinstance(obs.get("combat"), dict) else {}
    return _float(combat.get("self_inflicted_hp_loss_cumulative"))


def _enemies_from_obs(obs: dict[str, Any] | None) -> list[dict[str, Any]]:
    if not isinstance(obs, dict):
        return []
    combat = obs.get("combat") if isinstance(obs.get("combat"), dict) else {}
    enemies = combat.get("enemies")
    if not isinstance(enemies, list):
        return []
    return [enemy for enemy in enemies if isinstance(enemy, dict)]


def _player_snapshot(obs: dict[str, Any] | None) -> tuple[float, float, float, float]:
    if not isinstance(obs, dict):
        return 0.0, 0.0, 0.0, 1.0
    player = obs.get("player") if isinstance(obs.get("player"), dict) else {}
    hp = _float(player.get("hp"))
    block = _float(player.get("block"))
    energy = _float(player.get("energy"))
    max_hp = max(_float(player.get("max_hp"), 1.0), 1.0)
    return hp, block, energy, max_hp


def _is_enemy_alive(enemy: dict[str, Any]) -> bool:
    if not isinstance(enemy, dict):
        return False
    if "is_alive" in enemy:
        return bool(enemy.get("is_alive"))
    hp = enemy.get("hp")
    if hp is None:
        hp = enemy.get("current_hp")
    return _float(hp, 0.0) > 0.0


def _combat_token(obs: dict[str, Any] | None) -> str:
    if not isinstance(obs, dict):
        return ""
    combat = obs.get("combat") if isinstance(obs.get("combat"), dict) else {}
    for key in ("encounter_id", "combat_id", "room_uid", "room_id"):
        value = combat.get(key)
        if value not in (None, ""):
            return str(value)
    run = obs.get("run") if isinstance(obs.get("run"), dict) else {}
    floor = run.get("floor")
    return f"floor::{floor}" if floor not in (None, "") else ""


def _enemy_turn_counter(obs: dict[str, Any] | None) -> int | None:
    """Prefer the bridge-reported turn counter; None if unavailable."""
    if not isinstance(obs, dict):
        return None
    combat = obs.get("combat") if isinstance(obs.get("combat"), dict) else {}
    for key in ("turn", "turn_number", "turn_index", "round"):
        value = combat.get(key)
        if value is None:
            continue
        try:
            return int(value)
        except (TypeError, ValueError):
            continue
    return None


@dataclass
class EnemyHistoryEntry:
    first_seen_turn: int
    last_seen_turn: int
    prev_seen_turn: int
    hp_history: deque = field(default_factory=lambda: deque(maxlen=_HP_DELTA_TURN_WINDOW + 1))
    max_hp: float = 1.0
    block_history: deque = field(default_factory=lambda: deque(maxlen=_HP_DELTA_TURN_WINDOW + 1))
    power_amounts: dict[str, float] = field(default_factory=dict)
    prev_power_amounts: dict[str, float] = field(default_factory=dict)
    power_history: dict[str, deque] = field(default_factory=dict)
    power_first_seen_turn: dict[str, int] = field(default_factory=dict)
    intent_total_damage: float = 0.0
    intent_damage_per_hit: float = 0.0
    intent_kind_signature: str = ""
    intent_prev_total_damage: float = 0.0
    intent_prev_damage_per_hit: float = 0.0
    intent_total_damage_history: deque = field(
        default_factory=lambda: deque(maxlen=_HP_DELTA_TURN_WINDOW + 1)
    )
    intent_last_changed_turn: int = 0
    cum_damage_dealt_to_player: float = 0.0
    died_at_turn: int | None = None
    alive: bool = True
    hp_delta_last_turn: float = 0.0
    block_delta_last_turn: float = 0.0
    attributable_player_hp_loss_last_turn: float = 0.0
    attributable_actual_damage_last_turn: float = 0.0
    attributable_predicted_damage_last_turn: float = 0.0


@dataclass
class CombatMemoryState:
    combat_token: str = ""
    turn_index: int = 0
    enemies: dict[str, EnemyHistoryEntry] = field(default_factory=dict)
    recently_dead: dict[str, EnemyHistoryEntry] = field(default_factory=dict)
    prev_player_hp: float = 0.0
    prev_player_block: float = 0.0
    prev_player_energy: float = 0.0
    prev_player_max_hp: float = 1.0
    prev_enemy_total_hp: float = 0.0
    prev_alive_enemy_count: int = 0
    initial_enemy_total_hp: float = 0.0
    player_hp_delta_last_turn: float = 0.0
    player_block_delta_last_turn: float = 0.0
    player_energy_delta_last_turn: float = 0.0
    enemy_total_hp_delta_last_turn: float = 0.0
    enemy_count_delta_last_turn: int = 0
    surprise_damage_last_turn: float = 0.0
    cum_surprise_damage: float = 0.0
    turns_in_combat: int = 0
    last_bridge_turn: int | None = None
    prev_self_inflicted_hp_loss_cumulative: float = 0.0
    self_inflicted_hp_loss_last_step: float = 0.0


class CombatMemoryTracker:
    """Maintains per-combat per-enemy deltas across step() calls."""

    def __init__(self) -> None:
        self.state = CombatMemoryState()

    # ------------------------------------------------------------------
    # lifecycle
    # ------------------------------------------------------------------
    def reset(self, obs: dict[str, Any] | None) -> None:
        self.state = CombatMemoryState()
        self._absorb_initial(obs)

    def update(
        self,
        prev_obs: dict[str, Any] | None,
        action: dict[str, Any] | None,  # noqa: ARG002 - retained for parity
        next_obs: dict[str, Any] | None,
    ) -> None:
        token = _combat_token(next_obs)
        if token and token != self.state.combat_token:
            self.state = CombatMemoryState()
            self._absorb_initial(next_obs)
            return
        self._advance_turn(prev_obs, next_obs)

    # ------------------------------------------------------------------
    # consumers
    # ------------------------------------------------------------------
    def snapshot(self) -> dict[str, Any]:
        """Return a cheap read-only view for the encoder.

        The encoder calls this every step; we shape it as plain Python
        primitives so downstream serialisation / caching stays trivial.
        """
        enemies: dict[str, dict[str, Any]] = {}
        hp_ranks = self._rank_alive_enemies_by_hp()
        threat_ranks = self._rank_alive_enemies_by_threat()
        for key, entry in self.state.enemies.items():
            enemies[key] = self._enemy_snapshot(key, entry, hp_ranks, threat_ranks)
        for key, entry in self.state.recently_dead.items():
            snapshot = self._enemy_snapshot(key, entry, hp_ranks, threat_ranks)
            snapshot["died_last_turn"] = 1.0
            enemies[key] = snapshot

        return {
            "turn_index": self.state.turn_index,
            "turns_in_combat": self.state.turns_in_combat,
            "enemies": enemies,
            "player_hp_delta_last_turn": self.state.player_hp_delta_last_turn,
            "player_block_delta_last_turn": self.state.player_block_delta_last_turn,
            "player_energy_delta_last_turn": self.state.player_energy_delta_last_turn,
            "enemy_total_hp_delta_last_turn": self.state.enemy_total_hp_delta_last_turn,
            "enemy_count_delta_last_turn": self.state.enemy_count_delta_last_turn,
            "surprise_damage_last_turn": self.state.surprise_damage_last_turn,
            "cum_surprise_damage": self.state.cum_surprise_damage,
            "self_inflicted_hp_loss_last_step": self.state.self_inflicted_hp_loss_last_step,
            "initial_enemy_total_hp": self.state.initial_enemy_total_hp,
            "player_max_hp": self.state.prev_player_max_hp,
        }

    # ------------------------------------------------------------------
    # internals
    # ------------------------------------------------------------------
    def _absorb_initial(self, obs: dict[str, Any] | None) -> None:
        token = _combat_token(obs)
        self.state.combat_token = token
        self.state.last_bridge_turn = _enemy_turn_counter(obs)
        enemies = _enemies_from_obs(obs)
        player_hp, player_block, player_energy, player_max_hp = _player_snapshot(obs)
        self.state.prev_player_hp = player_hp
        self.state.prev_player_block = player_block
        self.state.prev_player_energy = player_energy
        self.state.prev_player_max_hp = player_max_hp
        self.state.turns_in_combat = 0
        self.state.turn_index = 0

        total_hp = 0.0
        alive_count = 0
        for index, enemy in enumerate(enemies):
            key = _enemy_key(enemy, index)
            entry = self._ensure_entry(key, enemy, turn=0)
            entry.alive = _is_enemy_alive(enemy)
            self._seed_entry_baseline(entry, enemy)
            if entry.alive:
                total_hp += _float(enemy.get("hp", enemy.get("current_hp")))
                alive_count += 1
        self.state.prev_enemy_total_hp = total_hp
        self.state.initial_enemy_total_hp = total_hp
        self.state.prev_alive_enemy_count = alive_count
        self.state.prev_self_inflicted_hp_loss_cumulative = _self_inflicted_hp_loss_cumulative(obs)
        self.state.self_inflicted_hp_loss_last_step = 0.0

    def _advance_turn(
        self,
        prev_obs: dict[str, Any] | None,
        next_obs: dict[str, Any] | None,
    ) -> None:
        self.state.recently_dead.clear()
        self.state.turn_index += 1

        bridge_turn = _enemy_turn_counter(next_obs)
        if (
            bridge_turn is not None
            and self.state.last_bridge_turn is not None
            and bridge_turn != self.state.last_bridge_turn
        ):
            self.state.turns_in_combat += 1
        elif bridge_turn is None:
            # Fall back to monotonic internal counter when bridge omits turn.
            self.state.turns_in_combat = self.state.turn_index
        self.state.last_bridge_turn = bridge_turn if bridge_turn is not None else self.state.last_bridge_turn

        player_hp, player_block, player_energy, player_max_hp = _player_snapshot(next_obs)
        self.state.player_hp_delta_last_turn = player_hp - self.state.prev_player_hp
        self.state.player_block_delta_last_turn = player_block - self.state.prev_player_block
        self.state.player_energy_delta_last_turn = player_energy - self.state.prev_player_energy
        raw_player_hp_loss = max(self.state.prev_player_hp - player_hp, 0.0)

        # Subtract self-inflicted damage so it doesn't get mis-attributed to enemies.
        # Bridge exposes a cumulative counter; the per-step self-damage is the diff
        # between the two observation snapshots wrapping this transition.
        current_self_inflicted_cum = _self_inflicted_hp_loss_cumulative(next_obs)
        self_inflicted_delta = max(
            current_self_inflicted_cum - self.state.prev_self_inflicted_hp_loss_cumulative,
            0.0,
        )
        self.state.self_inflicted_hp_loss_last_step = self_inflicted_delta
        actual_player_hp_loss = max(raw_player_hp_loss - self_inflicted_delta, 0.0)

        enemies = _enemies_from_obs(next_obs)
        current_keys: set[str] = set()
        total_predicted_damage = 0.0
        total_hp = 0.0
        alive_count = 0

        for index, enemy in enumerate(enemies):
            key = _enemy_key(enemy, index)
            current_keys.add(key)
            entry = self._ensure_entry(key, enemy, turn=self.state.turn_index)
            prev_hp = entry.hp_history[-1] if entry.hp_history else _float(enemy.get("hp"))
            prev_block = entry.block_history[-1] if entry.block_history else _float(enemy.get("block"))
            prev_intent_damage = entry.intent_total_damage

            hp = _float(enemy.get("hp", enemy.get("current_hp")))
            block = _float(enemy.get("block"))
            max_hp = max(_float(enemy.get("max_hp"), entry.max_hp), 1.0)
            entry.max_hp = max_hp
            entry.hp_delta_last_turn = hp - prev_hp
            entry.block_delta_last_turn = block - prev_block
            entry.hp_history.append(hp)
            entry.block_history.append(block)
            entry.prev_seen_turn = entry.last_seen_turn
            entry.last_seen_turn = self.state.turn_index
            entry.alive = _is_enemy_alive(enemy) and hp > 0.0
            if entry.alive:
                alive_count += 1
                total_hp += hp

            intent = enemy.get("intent") if isinstance(enemy.get("intent"), dict) else {}
            if not isinstance(intent, dict):
                intent = {}
            new_sig = _intent_signature(intent)
            entry.intent_prev_total_damage = entry.intent_total_damage
            entry.intent_prev_damage_per_hit = entry.intent_damage_per_hit
            entry.intent_total_damage = _float(intent.get("total_damage"))
            entry.intent_damage_per_hit = _float(intent.get("damage_per_hit"))
            entry.intent_total_damage_history.append(entry.intent_total_damage)
            if new_sig and new_sig != entry.intent_kind_signature:
                entry.intent_last_changed_turn = self.state.turn_index
                entry.intent_kind_signature = new_sig

            total_predicted_damage += prev_intent_damage
            entry.attributable_predicted_damage_last_turn = prev_intent_damage

            # ---- powers ----
            entry.prev_power_amounts = dict(entry.power_amounts)
            new_power_amounts: dict[str, float] = {}
            powers = enemy.get("powers") if isinstance(enemy.get("powers"), list) else []
            for power in powers:
                if not isinstance(power, dict):
                    continue
                key_power = _power_key(power)
                if not key_power:
                    continue
                amount = _float(power.get("amount") or power.get("display_amount"))
                new_power_amounts[key_power] = amount
                history = entry.power_history.setdefault(
                    key_power, deque(maxlen=_HP_DELTA_TURN_WINDOW + 1)
                )
                history.append(amount)
                entry.power_first_seen_turn.setdefault(key_power, self.state.turn_index)
            entry.power_amounts = new_power_amounts

        for key, entry in list(self.state.enemies.items()):
            if key not in current_keys:
                if entry.alive:
                    entry.alive = False
                    entry.died_at_turn = self.state.turn_index
                    entry.hp_delta_last_turn = -(entry.hp_history[-1] if entry.hp_history else 0.0)
                    self.state.recently_dead[key] = entry
                del self.state.enemies[key]

        # Distribute the actual hp loss proportionally to predicted damage.
        if total_predicted_damage > 0.0 and actual_player_hp_loss > 0.0:
            attribution_scale = actual_player_hp_loss / total_predicted_damage
        else:
            attribution_scale = 0.0
        for key in current_keys:
            entry = self.state.enemies.get(key)
            if entry is None:
                continue
            predicted = entry.attributable_predicted_damage_last_turn
            attributed = predicted * attribution_scale if attribution_scale > 0.0 else 0.0
            entry.attributable_player_hp_loss_last_turn = attributed
            entry.attributable_actual_damage_last_turn = attributed
            entry.cum_damage_dealt_to_player += attributed

        self.state.enemy_total_hp_delta_last_turn = total_hp - self.state.prev_enemy_total_hp
        self.state.enemy_count_delta_last_turn = alive_count - self.state.prev_alive_enemy_count
        surprise = max(0.0, actual_player_hp_loss - total_predicted_damage)
        self.state.surprise_damage_last_turn = surprise
        if surprise > 0.0:
            self.state.cum_surprise_damage += surprise

        self.state.prev_player_hp = player_hp
        self.state.prev_player_block = player_block
        self.state.prev_player_energy = player_energy
        self.state.prev_player_max_hp = player_max_hp
        self.state.prev_enemy_total_hp = total_hp
        self.state.prev_alive_enemy_count = alive_count
        self.state.prev_self_inflicted_hp_loss_cumulative = current_self_inflicted_cum

    def _seed_entry_baseline(
        self, entry: EnemyHistoryEntry, enemy: dict[str, Any]
    ) -> None:
        """Populate intent/power baselines so the first turn produces valid deltas."""
        intent = enemy.get("intent") if isinstance(enemy.get("intent"), dict) else {}
        if not isinstance(intent, dict):
            intent = {}
        entry.intent_total_damage = _float(intent.get("total_damage"))
        entry.intent_damage_per_hit = _float(intent.get("damage_per_hit"))
        entry.intent_prev_total_damage = entry.intent_total_damage
        entry.intent_prev_damage_per_hit = entry.intent_damage_per_hit
        entry.intent_kind_signature = _intent_signature(intent)
        entry.intent_last_changed_turn = 0
        entry.intent_total_damage_history.clear()
        entry.intent_total_damage_history.append(entry.intent_total_damage)

        powers = enemy.get("powers") if isinstance(enemy.get("powers"), list) else []
        entry.power_amounts = {}
        entry.prev_power_amounts = {}
        entry.power_history.clear()
        entry.power_first_seen_turn.clear()
        for power in powers:
            if not isinstance(power, dict):
                continue
            key_power = _power_key(power)
            if not key_power:
                continue
            amount = _float(power.get("amount") or power.get("display_amount"))
            entry.power_amounts[key_power] = amount
            entry.prev_power_amounts[key_power] = amount
            history = deque(maxlen=_HP_DELTA_TURN_WINDOW + 1)
            history.append(amount)
            entry.power_history[key_power] = history
            entry.power_first_seen_turn[key_power] = 0

    def _ensure_entry(
        self, key: str, enemy: dict[str, Any], *, turn: int
    ) -> EnemyHistoryEntry:
        entry = self.state.enemies.get(key)
        if entry is None:
            hp = _float(enemy.get("hp", enemy.get("current_hp")))
            block = _float(enemy.get("block"))
            max_hp = max(_float(enemy.get("max_hp"), 1.0), 1.0)
            entry = EnemyHistoryEntry(
                first_seen_turn=turn,
                last_seen_turn=turn,
                prev_seen_turn=turn,
                max_hp=max_hp,
            )
            entry.hp_history.append(hp)
            entry.block_history.append(block)
            self.state.enemies[key] = entry
        return entry

    # ------------------------------------------------------------------
    # snapshot helpers
    # ------------------------------------------------------------------
    def _rank_alive_enemies_by_hp(self) -> dict[str, int]:
        alive = [
            (key, entry.hp_history[-1] if entry.hp_history else 0.0)
            for key, entry in self.state.enemies.items()
            if entry.alive
        ]
        alive.sort(key=lambda item: item[1], reverse=True)
        return {key: rank for rank, (key, _hp) in enumerate(alive)}

    def _rank_alive_enemies_by_threat(self) -> dict[str, int]:
        alive = [
            (key, entry.intent_total_damage)
            for key, entry in self.state.enemies.items()
            if entry.alive
        ]
        alive.sort(key=lambda item: item[1], reverse=True)
        return {key: rank for rank, (key, _dmg) in enumerate(alive)}

    def _enemy_snapshot(
        self,
        key: str,
        entry: EnemyHistoryEntry,
        hp_ranks: dict[str, int],
        threat_ranks: dict[str, int],
    ) -> dict[str, Any]:
        max_hp = entry.max_hp or 1.0
        hp_now = entry.hp_history[-1] if entry.hp_history else 0.0
        hp_3_turns = (
            entry.hp_history[-1] - entry.hp_history[0]
            if len(entry.hp_history) >= 2
            else 0.0
        )
        power_snapshots: dict[str, dict[str, Any]] = {}
        for power_key, amount in entry.power_amounts.items():
            prev_amount = entry.prev_power_amounts.get(power_key, 0.0)
            history = entry.power_history.get(power_key)
            trend = 0.0
            is_growing_without_touch = 0.0
            if history is not None and len(history) >= 3:
                if history[-1] > history[-2] > history[-3]:
                    trend = 1.0
                elif history[-1] < history[-2] < history[-3]:
                    trend = -1.0
            if (
                amount > prev_amount + 1e-6
                and entry.hp_delta_last_turn >= -1e-6  # enemy wasn't hit this turn
            ):
                is_growing_without_touch = 1.0
            first_seen = entry.power_first_seen_turn.get(power_key, self.state.turn_index)
            power_snapshots[power_key] = {
                "amount": amount,
                "amount_delta_last_turn": amount - prev_amount,
                "amount_delta_since_first_seen": amount
                - (history[0] if history is not None and len(history) > 0 else amount),
                "turns_since_first_seen": self.state.turn_index - first_seen,
                "stack_trend_3_turns": trend,
                "is_new_this_turn": 1.0 if first_seen == self.state.turn_index else 0.0,
                "is_growing_without_player_action": is_growing_without_touch,
            }

        intent_damage_trend = 0.0
        if len(entry.intent_total_damage_history) >= 3:
            values = list(entry.intent_total_damage_history)
            if values[-1] > values[-2] > values[-3]:
                intent_damage_trend = 1.0
            elif values[-1] < values[-2] < values[-3]:
                intent_damage_trend = -1.0

        predicted_vs_actual = 0.0
        if entry.attributable_predicted_damage_last_turn > 0.0:
            predicted_vs_actual = min(
                entry.attributable_actual_damage_last_turn
                / max(entry.attributable_predicted_damage_last_turn, 1e-6),
                2.0,
            )

        return {
            "alive": 1.0 if entry.alive else 0.0,
            "hp_delta_last_turn_ratio": _clip_signed(
                entry.hp_delta_last_turn / max(max_hp, 1.0)
            ),
            "hp_delta_last_3_turns_ratio": _clip_signed(hp_3_turns / max(max_hp, 1.0)),
            "block_delta_last_turn_ratio": _clip_signed(
                entry.block_delta_last_turn / max(max_hp, 10.0)
            ),
            "turns_alive_norm": min(
                (self.state.turn_index - entry.first_seen_turn) / _TURN_NORM, 1.0
            ),
            "is_new_this_turn": 1.0
            if entry.first_seen_turn == self.state.turn_index
            else 0.0,
            "cum_damage_dealt_to_player_ratio": _log_norm(
                entry.cum_damage_dealt_to_player, _LOG1P_200
            ),
            "died_last_turn": 1.0 if entry.died_at_turn == self.state.turn_index else 0.0,
            "attributable_damage_last_turn_ratio": _log_norm(
                entry.attributable_player_hp_loss_last_turn, _LOG1P_200
            ),
            "alive_rank_by_hp": 1.0 / (hp_ranks.get(key, 0) + 1.0)
            if entry.alive
            else 0.0,
            "threat_rank_by_intent": 1.0 / (threat_ranks.get(key, 0) + 1.0)
            if entry.alive
            else 0.0,
            "intent_changed_this_turn": 1.0
            if entry.intent_last_changed_turn == self.state.turn_index
            else 0.0,
            "turns_since_intent_change_norm": min(
                (self.state.turn_index - entry.intent_last_changed_turn)
                / _INTENT_TURN_NORM,
                1.0,
            ),
            "intent_total_damage_delta": _signed_log_norm(
                entry.intent_total_damage - entry.intent_prev_total_damage, _LOG1P_200
            ),
            "intent_damage_per_hit_delta": _signed_log_norm(
                entry.intent_damage_per_hit - entry.intent_prev_damage_per_hit,
                _LOG1P_100,
            ),
            "intent_damage_trend_3_turns": intent_damage_trend,
            "intent_predicted_vs_actual": predicted_vs_actual,
            "hp_now": hp_now,
            "max_hp": max_hp,
            "powers": power_snapshots,
        }


__all__ = [
    "CombatMemoryTracker",
    "CombatMemoryState",
    "EnemyHistoryEntry",
]
