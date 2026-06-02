"""Phase 8 Tier 1 — per-step action history + per-turn summary tracking.

This module lets the observation encoder surface "what did the agent do
recently" as first-class tokens. The current Phase 6 policy is stateless
between steps: it cannot distinguish "I just selected a card, time to
confirm" from "I just entered this screen, time to select"; it cannot
reason about combat play order (Inflame before Strike stacks strength);
it cannot see buildup chains (Demon Form ticking +2 strength every turn).

Two concentric views are tracked per episode:

1. **Step-detail** (maxlen=20) — the most recent 20 env.step calls at
   full granularity. This covers roughly the last 3 combat turns for
   typical Ironclad play, more for Silent/Defect who burn through more
   actions per turn. Non-combat actions (map, event, shop) are also
   retained here (option A of the design doc) because the policy still
   benefits from knowing "I just bought Inflame at the shop → bias
   toward keeping it" when the next combat starts.

2. **Turn-summary** (maxlen=8) — one entry per completed combat turn,
   aggregating ``n_attacks_played`` / ``total_damage_dealt`` /
   ``end_of_turn_strength`` / etc. Created on combat.round transitions
   and on combat→non-combat transitions. Non-combat phases never produce
   turn_summary entries (there's no turn concept outside combat).

Both are consumed by ``observation_v3._append_history_tokens`` to emit
28 HISTORY tokens into the world-token budget (20 detail + 8 summary).
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field
from typing import Any

from . import observation_common as obs_common
from .semantic_action import (
    SEMANTIC_ACTION_FAMILIES,
    SEMANTIC_ROLE_NAMES,
    SEMANTIC_TARGET_SCOPES,
    semantic_action_signature,
)

_ROLE_NAME_TO_BIT = {name: 1 << i for i, name in enumerate(SEMANTIC_ROLE_NAMES)}
NUM_SEMANTIC_ROLES = len(SEMANTIC_ROLE_NAMES)


def _roles_to_flags(roles: Any) -> int:
    if not isinstance(roles, (list, tuple)):
        return 0
    flags = 0
    for role in roles:
        bit = _ROLE_NAME_TO_BIT.get(str(role))
        if bit:
            flags |= bit
    return flags

MAX_STEP_DETAIL_TOKENS = 20
MAX_TURN_SUMMARY_TOKENS = 8
# Total history token slots reserved in the world-token budget. Callers in
# observation_v3 use this to sanity-check that MAX_WORLD_TOKENS carries
# enough headroom.
MAX_HISTORY_TOKENS = MAX_STEP_DETAIL_TOKENS + MAX_TURN_SUMMARY_TOKENS  # 28

# Phase 8 Tier 2: state snapshot vectors attached to each StepDetailEntry.
# Eight absolute-state scalars captured at pre- and post-action moments,
# so attention can trivially derive "this card did X" deltas and also
# read the pre-context ("I did Demon Form at low energy → worth less").
# Each vec is 8-d; kept identical between pre/post so the token numeric
# block exposes a contiguous [pre || post] 16-d window policy heads can
# probe with a single linear projection.
STATE_SNAPSHOT_DIM = 8
STATE_SNAPSHOT_FIELD_NAMES = (
    "energy_norm",            # combat.energy / 5
    "hand_size_norm",         # len(combat.hand) / 10
    "player_hp_ratio",        # player.hp / player.max_hp
    "enemy_total_hp_ratio",   # sum enemy.hp / sum enemy.max_hp (filtered)
    "player_strength_norm",   # strength stacks / 10 (clamped)
    "player_dex_norm",        # dexterity stacks / 10 (clamped)
    "player_block_norm",      # combat.block / 40
    "enemy_total_vuln_norm",  # sum(vuln stacks on live enemies) / 10
)

# Phase 8 Tier 2: action_causality aux head predicts an 8-d delta (what
# would the chosen action DO) per step. Target is self-supervised from
# the (pre_state, post_state) pair the tracker already captures, folded
# into deltas and masked to the actually-chosen candidate row.
NUM_CAUSALITY_HEADS = 8
CAUSALITY_HEAD_NAMES = (
    "damage_dealt",        # max(enemy_total_hp_pre - post, 0) / 30
    "block_gained",        # max(player_block_post - pre, 0) / 30
    "self_hp_loss",        # max(player_hp_pre - post, 0) / 30
    "draw_delta",          # (hand_size_post - pre) / 5, signed
    "energy_delta",        # (energy_post - pre) / 3, signed, clipped [-1,1]
    "strength_delta",      # (strength_post - pre) / 5, signed, clipped
    "dex_delta",           # (dex_post - pre) / 5, signed, clipped
    "vuln_applied",        # max(enemy_total_vuln_post - pre, 0) / 10
)

# 16-bit bitmap of "key power cards" whose presence in a turn's play
# sequence unlocks meaningfully different future strategies. Expansion
# knobs: add more entries; slot 16 is the last bit we have room for in
# the turn_summary numeric packing.
KEY_POWER_CARD_BUCKETS: dict[str, int] = {
    # Ironclad scaling / buff
    "CARD.INFLAME": 0,
    "CARD.DEMON_FORM": 1,
    "CARD.LIMIT_BREAK": 2,
    "CARD.BERSERK": 3,
    "CARD.METALLICIZE": 4,
    "CARD.JUGGERNAUT": 5,
    "CARD.FEEL_NO_PAIN": 6,
    "CARD.BARRICADE": 7,
    # Silent scaling / buff
    "CARD.NOXIOUS_FUMES": 8,
    "CARD.ACCURACY": 9,
    "CARD.AFTER_IMAGE": 10,
    # Defect scaling / buff
    "CARD.ECHO_FORM": 11,
    "CARD.CREATIVE_AI": 12,
    "CARD.BIASED_COGNITION": 13,
    # Neutral / universal
    "CARD.APOTHEOSIS": 14,
    "CARD.MASTER_OF_STRATEGY": 15,
}
NUM_KEY_POWER_FLAGS = 16  # equal to len(KEY_POWER_CARD_BUCKETS)

# Result-flag bit positions inside the 4-d numeric slice.
RESULT_FLAG_REWARD_NONZERO = 0
RESULT_FLAG_REJECTED = 1
RESULT_FLAG_PHASE_CHANGED = 2
RESULT_FLAG_COMBAT_ENDED = 3
NUM_RESULT_FLAGS = 4


def _float(value: Any, default: float = 0.0) -> float:
    if value is None:
        return default
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _int(value: Any, default: int = 0) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


def _str(value: Any, default: str = "") -> str:
    if value is None:
        return default
    return str(value)


def _phase(obs: dict[str, Any] | None) -> str:
    if not isinstance(obs, dict):
        return ""
    return _str(obs.get("phase"))


def _floor(obs: dict[str, Any] | None) -> int:
    if not isinstance(obs, dict):
        return 0
    run = obs.get("run")
    if isinstance(run, dict):
        for key in ("floor", "total_floor", "act_floor"):
            val = run.get(key)
            if val is not None:
                return _int(val, 0)
    return 0


def _combat(obs: dict[str, Any] | None) -> dict[str, Any]:
    if not isinstance(obs, dict):
        return {}
    combat = obs.get("combat")
    return combat if isinstance(combat, dict) else {}


def _combat_round(obs: dict[str, Any] | None) -> int:
    combat = _combat(obs)
    return _int(combat.get("round"), -1) if combat else -1


def _in_combat(obs: dict[str, Any] | None) -> bool:
    combat = _combat(obs)
    if not combat:
        return False
    if "in_progress" in combat:
        return bool(combat.get("in_progress"))
    return bool(combat.get("enemies")) or _phase(obs) == "combat"


def _enemy_total_hp(obs: dict[str, Any] | None) -> float:
    combat = _combat(obs)
    enemies = combat.get("enemies") if isinstance(combat, dict) else None
    if not isinstance(enemies, list):
        return 0.0
    total = 0.0
    for enemy in enemies:
        if not isinstance(enemy, dict):
            continue
        hp = _float(enemy.get("hp"))
        # Skip sentinel-HP bosses whose "hp" is 1e9 until trigger
        if hp > 10_000:
            continue
        total += hp
    return total


def _enemy_total_max_hp(obs: dict[str, Any] | None) -> float:
    combat = _combat(obs)
    enemies = combat.get("enemies") if isinstance(combat, dict) else None
    if not isinstance(enemies, list):
        return 1.0
    total = 0.0
    for enemy in enemies:
        if not isinstance(enemy, dict):
            continue
        mh = _float(enemy.get("max_hp"))
        if mh > 10_000:
            continue
        total += mh
    return total if total > 0 else 1.0


def _player_hp(obs: dict[str, Any] | None) -> float:
    if not isinstance(obs, dict):
        return 0.0
    player = obs.get("player")
    if isinstance(player, dict):
        return _float(player.get("hp"))
    return 0.0


def _player_block(obs: dict[str, Any] | None) -> float:
    combat = _combat(obs)
    if combat:
        return _float(combat.get("block"))
    if isinstance(obs, dict):
        player = obs.get("player") if isinstance(obs.get("player"), dict) else None
        if isinstance(player, dict):
            return _float(player.get("block"))
    return 0.0


def _player_power_amount(obs: dict[str, Any] | None, power_id_substring: str) -> float:
    """Find an active player power whose id/title contains the given
    substring (case-insensitive) and return its amount. Used to extract
    strength/dex/focus stacks for turn-end summaries without requiring
    an exact id match (sim varies between ``STRENGTH_POWER`` and
    ``strength`` at different layers).
    """
    if not isinstance(obs, dict):
        return 0.0
    player = obs.get("player") if isinstance(obs.get("player"), dict) else None
    if not isinstance(player, dict):
        return 0.0
    needle = power_id_substring.lower()
    for source_key in ("status", "powers"):
        powers = player.get(source_key)
        if not isinstance(powers, list):
            continue
        for p in powers:
            if not isinstance(p, dict):
                continue
            label = _str(p.get("id") or p.get("power_id") or p.get("title") or p.get("name")).lower()
            if needle in label:
                return _float(p.get("amount") or p.get("stacks") or p.get("value"))
    return 0.0


def _hand_size(obs: dict[str, Any] | None) -> int:
    combat = _combat(obs)
    hand = combat.get("hand") if isinstance(combat, dict) else None
    return len(hand) if isinstance(hand, list) else 0


def _player_max_hp(obs: dict[str, Any] | None) -> float:
    if not isinstance(obs, dict):
        return 0.0
    player = obs.get("player") if isinstance(obs.get("player"), dict) else None
    if isinstance(player, dict):
        return obs_common._player_max_hp_value(player)
    return 0.0


def _player_hp_ratio(obs: dict[str, Any] | None) -> float:
    if not isinstance(obs, dict):
        return 0.0
    player = obs.get("player") if isinstance(obs.get("player"), dict) else None
    return obs_common._player_hp_triplet(player)[2]


def _build_state_snapshot(obs: dict[str, Any] | None) -> tuple[float, ...]:
    """Compute the 8-d absolute-state snapshot used by both HISTORY token
    pre/post vectors and the action_causality target-building path.

    Normalization constants chosen so a typical early-Act-1 turn sits
    well inside [0, 1]. The same normalization is applied identically
    to pre and post so the policy's delta-probe is uniform-scale.
    """
    import math

    combat = _combat(obs)
    player = obs.get("player") if isinstance(obs, dict) and isinstance(obs.get("player"), dict) else {}
    energy = _float(combat.get("energy")) if combat else 0.0
    hand_size = _hand_size(obs)
    player_hp = _player_hp(obs)
    player_block = _player_block(obs)
    player_strength = _player_power_amount(obs, "strength")
    player_dex = _player_power_amount(obs, "dexterity")
    enemy_hp = _enemy_total_hp(obs)
    enemy_max = max(_enemy_total_hp(obs), 1.0)
    # enemy_max here is a weak proxy; for the ratio we want pre-state
    # total maxhp which doesn't drop as enemies die. Use the sum of
    # enemy.max_hp filtered for sentinels.
    enemy_max_hp_accum = 0.0
    for e in (combat.get("enemies") or []) if isinstance(combat, dict) else []:
        if not isinstance(e, dict):
            continue
        mh = _float(e.get("max_hp"))
        if 0 < mh < 10_000:
            enemy_max_hp_accum += mh
    enemy_max_ref = enemy_max_hp_accum if enemy_max_hp_accum > 0 else max(enemy_hp, 1.0)
    enemy_vuln = _enemy_vuln_amount(obs)
    return (
        max(0.0, min(energy / 5.0, 1.0)),
        max(0.0, min(hand_size / 10.0, 1.0)),
        _player_hp_ratio(obs),
        max(0.0, min(enemy_hp / enemy_max_ref, 1.0)),
        max(0.0, min(player_strength / 10.0, 1.0)),
        max(0.0, min(player_dex / 10.0, 1.0)),
        max(0.0, min(player_block / 40.0, 1.0)),
        max(0.0, min(enemy_vuln / 10.0, 1.0)),
    )


def _build_causality_delta(
    prev_obs: dict[str, Any] | None,
    next_obs: dict[str, Any] | None,
) -> tuple[float, ...]:
    """Compute the 8-d per-step causality target vector.

    Deltas are independently normalized (not simply post_snapshot -
    pre_snapshot) because effects have their own natural scales — raw
    damage is /30, block is /30, energy is /3, etc. Clamped to [-1, 1]
    to keep smooth_l1 loss well-behaved.

    Returns the 8-d target in the order of CAUSALITY_HEAD_NAMES.
    """
    pre_enemy_hp = _enemy_total_hp(prev_obs)
    post_enemy_hp = _enemy_total_hp(next_obs)
    damage_dealt = max(pre_enemy_hp - post_enemy_hp, 0.0)

    pre_block = _player_block(prev_obs)
    post_block = _player_block(next_obs)
    block_gained = max(post_block - pre_block, 0.0)

    pre_hp = _player_hp(prev_obs)
    post_hp = _player_hp(next_obs)
    self_hp_loss = max(pre_hp - post_hp, 0.0)

    pre_hand = _hand_size(prev_obs)
    post_hand = _hand_size(next_obs)
    draw_delta_raw = post_hand - pre_hand

    pre_energy = _float(_combat(prev_obs).get("energy")) if _combat(prev_obs) else 0.0
    post_energy = _float(_combat(next_obs).get("energy")) if _combat(next_obs) else 0.0
    energy_delta_raw = post_energy - pre_energy

    pre_str = _player_power_amount(prev_obs, "strength")
    post_str = _player_power_amount(next_obs, "strength")
    strength_delta_raw = post_str - pre_str

    pre_dex = _player_power_amount(prev_obs, "dexterity")
    post_dex = _player_power_amount(next_obs, "dexterity")
    dex_delta_raw = post_dex - pre_dex

    pre_vuln = _enemy_vuln_amount(prev_obs)
    post_vuln = _enemy_vuln_amount(next_obs)
    vuln_applied = max(post_vuln - pre_vuln, 0.0)

    return (
        max(0.0, min(damage_dealt / 30.0, 1.0)),
        max(0.0, min(block_gained / 30.0, 1.0)),
        max(0.0, min(self_hp_loss / 30.0, 1.0)),
        max(-1.0, min(draw_delta_raw / 5.0, 1.0)),
        max(-1.0, min(energy_delta_raw / 3.0, 1.0)),
        max(-1.0, min(strength_delta_raw / 5.0, 1.0)),
        max(-1.0, min(dex_delta_raw / 5.0, 1.0)),
        max(0.0, min(vuln_applied / 10.0, 1.0)),
    )


def _enemy_vuln_amount(obs: dict[str, Any] | None) -> float:
    """Sum Vulnerable stacks across all live enemies. Proxy signal for
    "have I applied debuffs this turn?" in turn_summary encoding.
    """
    combat = _combat(obs)
    enemies = combat.get("enemies") if isinstance(combat, dict) else None
    if not isinstance(enemies, list):
        return 0.0
    total = 0.0
    for enemy in enemies:
        if not isinstance(enemy, dict):
            continue
        for source_key in ("powers", "status"):
            powers = enemy.get(source_key)
            if not isinstance(powers, list):
                continue
            for p in powers:
                if not isinstance(p, dict):
                    continue
                label = _str(p.get("id") or p.get("power_id") or p.get("title")).lower()
                if "vulnerable" in label or "vuln" in label:
                    total += _float(p.get("amount") or p.get("stacks") or 0)
    return total


def _stable_card_bucket(card_id: str, buckets: int = 8192) -> int:
    """Same hashing family as observation_v3._stable_bucket so HISTORY
    tokens can reference the SAME bucket space the existing entity
    embedding table is trained on. This lets the new history_card_bias
    embedding table key off identifiers that ALREADY appear elsewhere
    (hand / deck / discard tokens) — no separate vocab to learn.
    """
    if not card_id:
        return 0
    h = 0
    for ch in card_id:
        h = (h * 31 + ord(ch)) & 0xFFFFFFFF
    return (h % (buckets - 1)) + 1  # 0 reserved for empty


@dataclass
class StepDetailEntry:
    """A single past action + its immediate consequences.

    Step-level tokens pack most of the 96 numeric feature slots with
    categorical one-hots plus the Tier 2 pre_state / post_state / delta
    vectors. See ``observation_v3._append_history_tokens`` for the
    exact feature layout; this dataclass exposes the primitives in a
    form that layout code can consume without having to know what a
    ``semantic_action_signature`` is shaped like.

    Note: Phase 8 Tier 2 replaced the Tier 1 scalar delta fields
    (hp_delta_player, enemy_hp_delta, block_delta, energy_delta) with
    the richer ``causality_delta`` vector. The new vector contains
    everything the scalar deltas did plus strength/dex/draw/vuln/block
    signal — removing the old fields avoids carrying duplicate data.
    """

    family: str
    family_idx: int
    semantic_role_flags: int
    target_scope_idx: int
    card_id: str
    card_id_bucket: int
    same_turn: bool
    same_encounter: bool
    same_floor: bool
    phase: str
    reward: float
    reward_nonzero: bool
    rejected: bool
    phase_changed: bool
    combat_ended: bool
    canonical_text: str
    # Tier 2 additions:
    pre_state_vec: tuple          # STATE_SNAPSHOT_DIM floats (absolute-state snapshot at prev_obs)
    post_state_vec: tuple         # STATE_SNAPSHOT_DIM floats (absolute-state snapshot at next_obs)
    causality_delta: tuple        # NUM_CAUSALITY_HEADS floats (self-supervised target for action_causality)

    @classmethod
    def from_transition(
        cls,
        *,
        action: dict[str, Any] | None,
        prev_obs: dict[str, Any] | None,
        next_obs: dict[str, Any] | None,
        reward: float,
        rejected: bool,
    ) -> "StepDetailEntry":
        sig = semantic_action_signature(action) if isinstance(action, dict) else {}
        family = _str(sig.get("family") or "other")
        family_idx = _family_idx(family)
        role_flags = _roles_to_flags(sig.get("roles"))
        target_scope_idx = _target_scope_idx(sig.get("target_scope"))
        # Card-id resolution — sim translator puts the card under
        # action["card"]["id"] for play_card / card_reward / card_selection;
        # potion and event actions leave this empty.
        card_id = ""
        if isinstance(action, dict):
            card_obj = action.get("card") if isinstance(action.get("card"), dict) else None
            if isinstance(card_obj, dict):
                card_id = _str(card_obj.get("id"))
        card_bucket = _stable_card_bucket(card_id) if card_id else 0

        prev_round = _combat_round(prev_obs)
        next_round = _combat_round(next_obs)
        same_turn = (prev_round == next_round)
        prev_in_combat = _in_combat(prev_obs)
        next_in_combat = _in_combat(next_obs)
        same_encounter = prev_in_combat and next_in_combat
        same_floor = _floor(prev_obs) == _floor(next_obs)
        phase_before = _phase(prev_obs)
        phase_after = _phase(next_obs)
        phase_changed = phase_before != phase_after
        combat_ended = prev_in_combat and not next_in_combat

        pre_state_vec = _build_state_snapshot(prev_obs)
        post_state_vec = _build_state_snapshot(next_obs)
        causality_delta = _build_causality_delta(prev_obs, next_obs)

        canonical_text = ""
        if isinstance(action, dict):
            canonical_text = _str(action.get("canonical_text"))

        return cls(
            family=family,
            family_idx=family_idx,
            semantic_role_flags=role_flags,
            target_scope_idx=target_scope_idx,
            card_id=card_id,
            card_id_bucket=card_bucket,
            same_turn=bool(same_turn),
            same_encounter=bool(same_encounter),
            same_floor=bool(same_floor),
            phase=phase_before,
            reward=float(reward),
            reward_nonzero=abs(reward) > 1e-6,
            rejected=bool(rejected),
            phase_changed=bool(phase_changed),
            combat_ended=bool(combat_ended),
            canonical_text=canonical_text,
            pre_state_vec=pre_state_vec,
            post_state_vec=post_state_vec,
            causality_delta=causality_delta,
        )


@dataclass
class TurnSummaryEntry:
    """Aggregated view of one completed combat turn.

    Produced when ``combat.round`` advances (or combat ends) — the
    ``TurnAccumulator`` is folded into this struct and a new accumulator
    starts for the next turn. Only actions taken while
    ``_in_combat(obs)`` is true contribute; map/event/shop choices
    live in ``StepDetailEntry`` only.
    """

    turn_offset: int = 0             # 1 = most-recent-completed, grows as newer turns age in
    n_attacks: int = 0
    n_skills: int = 0
    n_powers: int = 0
    n_potions_used: int = 0
    total_damage_dealt: float = 0.0  # summed enemy_hp_delta where action was an attack
    total_block_gained: float = 0.0  # summed block_delta where positive
    total_hp_lost: float = 0.0       # summed -hp_delta_player where negative
    turn_num: int = 0                # combat.round value when the turn STARTED
    end_strength: float = 0.0
    end_dex: float = 0.0
    end_focus: float = 0.0
    end_enemy_total_hp_ratio: float = 1.0
    end_player_block: float = 0.0
    end_enemy_vuln_total: float = 0.0
    key_power_card_flags: int = 0    # bitmap over KEY_POWER_CARD_BUCKETS
    enemy_killed: bool = False
    player_took_dmg: bool = False
    player_scaled: bool = False
    low_energy_waste: bool = False
    canonical_text: str = ""


class TurnAccumulator:
    """Folds consecutive StepDetailEntry values (during combat) into a
    partial TurnSummaryEntry. Call ``finalize(obs)`` when the turn ends
    to snapshot end-of-turn state into the summary.
    """

    def __init__(self) -> None:
        self.n_attacks = 0
        self.n_skills = 0
        self.n_powers = 0
        self.n_potions_used = 0
        self.total_damage_dealt = 0.0
        self.total_block_gained = 0.0
        self.total_hp_lost = 0.0
        self.key_power_card_flags = 0
        self.enemy_killed = False
        self.player_took_dmg = False
        self.player_scaled = False
        self.starting_round = -1

    def set_starting_round(self, round_num: int) -> None:
        if self.starting_round < 0:
            self.starting_round = max(int(round_num), 0)

    def absorb(
        self,
        *,
        entry: StepDetailEntry,
        action: dict[str, Any] | None,
        prev_obs: dict[str, Any] | None,
        next_obs: dict[str, Any] | None,
    ) -> None:
        # Card-type counters live on the action.card.type field (bridge
        # populates "Attack" / "Skill" / "Power"). Fall back to the
        # semantic family for non-card actions (potions).
        card_type = ""
        if isinstance(action, dict):
            card_obj = action.get("card") if isinstance(action.get("card"), dict) else None
            if isinstance(card_obj, dict):
                card_type = _str(card_obj.get("type")).lower()
        if entry.family == "play_card":
            if card_type == "attack":
                self.n_attacks += 1
            elif card_type == "skill":
                self.n_skills += 1
            elif card_type == "power":
                self.n_powers += 1
        elif entry.family == "use_potion":
            self.n_potions_used += 1

        # causality_delta fields are normalized — de-normalize by inverting
        # the scaling factors documented in CAUSALITY_HEAD_NAMES. Keeping
        # the absorb logic on RAW amounts preserves the turn_summary
        # dynamic range (sum of damage can exceed 1.0).
        damage_dealt = entry.causality_delta[0] * 30.0 if len(entry.causality_delta) > 0 else 0.0
        block_gained = entry.causality_delta[1] * 30.0 if len(entry.causality_delta) > 1 else 0.0
        self_hp_loss = entry.causality_delta[2] * 30.0 if len(entry.causality_delta) > 2 else 0.0
        if damage_dealt > 0:
            self.total_damage_dealt += damage_dealt
        if block_gained > 0:
            self.total_block_gained += block_gained
        if self_hp_loss > 0:
            self.total_hp_lost += self_hp_loss
            self.player_took_dmg = True

        # Track scaling buffs: any strength/dex/focus delta during this
        # turn counts the turn as "scaling".
        pre_str = _player_power_amount(prev_obs, "strength")
        post_str = _player_power_amount(next_obs, "strength")
        pre_dex = _player_power_amount(prev_obs, "dexterity")
        post_dex = _player_power_amount(next_obs, "dexterity")
        pre_focus = _player_power_amount(prev_obs, "focus")
        post_focus = _player_power_amount(next_obs, "focus")
        if (
            post_str > pre_str + 1e-3
            or post_dex > pre_dex + 1e-3
            or post_focus > pre_focus + 1e-3
        ):
            self.player_scaled = True

        if entry.card_id in KEY_POWER_CARD_BUCKETS:
            self.key_power_card_flags |= (1 << KEY_POWER_CARD_BUCKETS[entry.card_id])

        # Enemy kill heuristic: at least one enemy whose hp was >0
        # pre-action dropped to exactly 0 post-action. A robust check
        # would need combat_id matching across lists but this proxy is
        # cheap and aligns with how combat_memory flags it.
        prev_enemies = _combat(prev_obs).get("enemies") if isinstance(_combat(prev_obs).get("enemies"), list) else []
        next_enemies = _combat(next_obs).get("enemies") if isinstance(_combat(next_obs).get("enemies"), list) else []
        if prev_enemies and len(next_enemies) <= len(prev_enemies):
            for pre_e, post_e in zip(prev_enemies, next_enemies):
                if not isinstance(pre_e, dict) or not isinstance(post_e, dict):
                    continue
                if _float(pre_e.get("hp")) > 0 and _float(post_e.get("hp")) <= 0:
                    self.enemy_killed = True
                    break

    def finalize(self, *, next_obs: dict[str, Any] | None, canonical_text: str = "") -> TurnSummaryEntry:
        # "Low-energy waste" flag: player ended turn with energy > 0 and
        # at least one 0-cost card still in hand. Rough signal for
        # "you could have played more but didn't".
        low_energy_waste = False
        combat = _combat(next_obs)
        end_energy = _float(combat.get("energy")) if combat else 0.0
        if end_energy > 0 and isinstance(combat.get("hand"), list):
            for c in combat.get("hand") or []:
                if isinstance(c, dict) and _float(c.get("cost")) <= 0:
                    low_energy_waste = True
                    break
        enemy_hp_ratio = _enemy_total_hp(next_obs) / max(_enemy_total_max_hp(next_obs), 1.0)

        return TurnSummaryEntry(
            turn_offset=1,  # set by controller; placeholder of "most recent"
            n_attacks=int(self.n_attacks),
            n_skills=int(self.n_skills),
            n_powers=int(self.n_powers),
            n_potions_used=int(self.n_potions_used),
            total_damage_dealt=float(self.total_damage_dealt),
            total_block_gained=float(self.total_block_gained),
            total_hp_lost=float(self.total_hp_lost),
            turn_num=int(max(self.starting_round, 0)),
            end_strength=_player_power_amount(next_obs, "strength"),
            end_dex=_player_power_amount(next_obs, "dexterity"),
            end_focus=_player_power_amount(next_obs, "focus"),
            end_enemy_total_hp_ratio=float(min(max(enemy_hp_ratio, 0.0), 1.0)),
            end_player_block=_player_block(next_obs),
            end_enemy_vuln_total=_enemy_vuln_amount(next_obs),
            key_power_card_flags=int(self.key_power_card_flags),
            enemy_killed=bool(self.enemy_killed),
            player_took_dmg=bool(self.player_took_dmg),
            player_scaled=bool(self.player_scaled),
            low_energy_waste=bool(low_energy_waste),
            canonical_text=canonical_text,
        )


class ActionHistoryTracker:
    """Episode-scoped holder for ``StepDetailEntry`` and
    ``TurnSummaryEntry`` sequences. Owns the accumulator for the
    currently-in-progress combat turn.
    """

    def __init__(
        self,
        *,
        step_detail_cap: int = MAX_STEP_DETAIL_TOKENS,
        turn_summary_cap: int = MAX_TURN_SUMMARY_TOKENS,
    ) -> None:
        self._step_detail_cap = int(step_detail_cap)
        self._turn_summary_cap = int(turn_summary_cap)
        self.step_detail: deque[StepDetailEntry] = deque(maxlen=self._step_detail_cap)
        self.turn_summary: deque[TurnSummaryEntry] = deque(maxlen=self._turn_summary_cap)
        self._turn_accumulator = TurnAccumulator()

    def reset(self) -> None:
        self.step_detail.clear()
        self.turn_summary.clear()
        self._turn_accumulator = TurnAccumulator()

    def record(
        self,
        *,
        action: dict[str, Any] | None,
        prev_obs: dict[str, Any] | None,
        next_obs: dict[str, Any] | None,
        reward: float,
        rejected: bool = False,
    ) -> None:
        entry = StepDetailEntry.from_transition(
            action=action,
            prev_obs=prev_obs,
            next_obs=next_obs,
            reward=reward,
            rejected=rejected,
        )
        self.step_detail.append(entry)

        prev_round = _combat_round(prev_obs)
        next_round = _combat_round(next_obs)
        prev_in_combat = _in_combat(prev_obs)
        next_in_combat = _in_combat(next_obs)

        # Seed the accumulator's starting round the first time we see a
        # combat step without one.
        if prev_in_combat and self._turn_accumulator.starting_round < 0:
            self._turn_accumulator.set_starting_round(prev_round)

        if prev_in_combat:
            self._turn_accumulator.absorb(
                entry=entry,
                action=action,
                prev_obs=prev_obs,
                next_obs=next_obs,
            )

        # Turn boundary: either combat.round advanced, or we exited combat.
        turn_boundary = prev_in_combat and (
            next_round > prev_round or (not next_in_combat and (self._turn_accumulator.n_attacks or self._turn_accumulator.n_skills or self._turn_accumulator.n_powers))
        )
        if turn_boundary:
            summary = self._turn_accumulator.finalize(next_obs=next_obs, canonical_text=entry.canonical_text)
            self._push_turn_summary(summary)
            self._turn_accumulator = TurnAccumulator()
            if next_in_combat:
                # The new turn may have already opened — prime the
                # accumulator's starting round with the new combat.round
                # so the next record() call doesn't overwrite it.
                self._turn_accumulator.set_starting_round(next_round)

    def _push_turn_summary(self, summary: TurnSummaryEntry) -> None:
        # Shift existing entries' offsets by +1 (older), then append
        # the new one as offset=1. deque.maxlen drops anything past 8.
        for existing in self.turn_summary:
            existing.turn_offset = min(existing.turn_offset + 1, self._turn_summary_cap)
        summary.turn_offset = 1
        self.turn_summary.append(summary)

    def to_obs_dict(self) -> dict[str, Any]:
        """Snapshot the tracker into plain-dict form for inclusion in
        ``raw_obs["action_history"]``. Keeps observation_v3 decoupled
        from the dataclass layer.
        """
        step_detail = []
        # step_offset: 0 = most recent; deque[-1] is the newest append.
        history_len = len(self.step_detail)
        for i, entry in enumerate(reversed(self.step_detail)):
            step_detail.append(
                {
                    "family": entry.family,
                    "family_idx": entry.family_idx,
                    "semantic_role_flags": entry.semantic_role_flags,
                    "target_scope_idx": entry.target_scope_idx,
                    "card_id": entry.card_id,
                    "card_id_bucket": entry.card_id_bucket,
                    "step_offset": i,
                    "same_turn": entry.same_turn,
                    "same_encounter": entry.same_encounter,
                    "same_floor": entry.same_floor,
                    "phase": entry.phase,
                    "reward": entry.reward,
                    "reward_nonzero": entry.reward_nonzero,
                    "rejected": entry.rejected,
                    "phase_changed": entry.phase_changed,
                    "combat_ended": entry.combat_ended,
                    "canonical_text": entry.canonical_text,
                    "pre_state_vec": list(entry.pre_state_vec),
                    "post_state_vec": list(entry.post_state_vec),
                    "causality_delta": list(entry.causality_delta),
                }
            )
        turn_summary = []
        for entry in self.turn_summary:
            turn_summary.append(
                {
                    "turn_offset": entry.turn_offset,
                    "n_attacks": entry.n_attacks,
                    "n_skills": entry.n_skills,
                    "n_powers": entry.n_powers,
                    "n_potions_used": entry.n_potions_used,
                    "total_damage_dealt": entry.total_damage_dealt,
                    "total_block_gained": entry.total_block_gained,
                    "total_hp_lost": entry.total_hp_lost,
                    "turn_num": entry.turn_num,
                    "end_strength": entry.end_strength,
                    "end_dex": entry.end_dex,
                    "end_focus": entry.end_focus,
                    "end_enemy_total_hp_ratio": entry.end_enemy_total_hp_ratio,
                    "end_player_block": entry.end_player_block,
                    "end_enemy_vuln_total": entry.end_enemy_vuln_total,
                    "key_power_card_flags": entry.key_power_card_flags,
                    "enemy_killed": entry.enemy_killed,
                    "player_took_dmg": entry.player_took_dmg,
                    "player_scaled": entry.player_scaled,
                    "low_energy_waste": entry.low_energy_waste,
                    "canonical_text": entry.canonical_text,
                }
            )
        return {
            "step_detail": step_detail,
            "turn_summary": turn_summary,
            "history_len": history_len,
            "turn_summary_len": len(self.turn_summary),
        }


_FAMILY_IDX_CACHE: dict[str, int] = {fam: i for i, fam in enumerate(SEMANTIC_ACTION_FAMILIES)}
_OTHER_FAMILY_IDX = _FAMILY_IDX_CACHE.get("other", len(SEMANTIC_ACTION_FAMILIES) - 1)


def _family_idx(family: str) -> int:
    return _FAMILY_IDX_CACHE.get(family, _OTHER_FAMILY_IDX)


_TARGET_SCOPE_IDX_CACHE = {scope: i for i, scope in enumerate(SEMANTIC_TARGET_SCOPES)}
_OTHER_SCOPE_IDX = _TARGET_SCOPE_IDX_CACHE.get("other", len(SEMANTIC_TARGET_SCOPES) - 1)


def _target_scope_idx(scope: Any) -> int:
    if not scope:
        return _TARGET_SCOPE_IDX_CACHE.get("none", 0)
    return _TARGET_SCOPE_IDX_CACHE.get(str(scope), _OTHER_SCOPE_IDX)


__all__ = [
    "MAX_STEP_DETAIL_TOKENS",
    "MAX_TURN_SUMMARY_TOKENS",
    "MAX_HISTORY_TOKENS",
    "NUM_KEY_POWER_FLAGS",
    "NUM_RESULT_FLAGS",
    "NUM_SEMANTIC_ROLES",
    "STATE_SNAPSHOT_DIM",
    "STATE_SNAPSHOT_FIELD_NAMES",
    "NUM_CAUSALITY_HEADS",
    "CAUSALITY_HEAD_NAMES",
    "KEY_POWER_CARD_BUCKETS",
    "RESULT_FLAG_REWARD_NONZERO",
    "RESULT_FLAG_REJECTED",
    "RESULT_FLAG_PHASE_CHANGED",
    "RESULT_FLAG_COMBAT_ENDED",
    "StepDetailEntry",
    "TurnSummaryEntry",
    "TurnAccumulator",
    "ActionHistoryTracker",
    "_stable_card_bucket",
    "_build_state_snapshot",
    "_build_causality_delta",
]
