"""Phase 2 Dict observation encoder for STS2 RL training.

Returns gymnasium Dict observations with separate arrays for scalars,
hand cards, enemies, and player powers.  Each variable-length set
(cards, enemies) includes a mask array so the attention network can
ignore padding slots.

No torch dependency -- numpy only.
"""

from __future__ import annotations

import re
import numpy as np

from .observation import (
    PHASES, PHASE_TO_IDX, NUM_PHASES,
    ROOM_TYPES, ROOM_TYPE_TO_ORD, NUM_ROOM_TYPES,
    MAX_HAND, MAX_ENEMIES,
    _g, _float, _bool,
)

# ---------------------------------------------------------------------------
# Dimension constants
# ---------------------------------------------------------------------------

# Scalars: phase one-hot (17) + run (6) + player (8) + combat (8)
#        + decision (10) + summary (6) = 55
SCALAR_DIM: int = 55

# Per-card features
# cost/5, is_attack, is_skill, is_power, is_x_cost, star_cost/5, star_x,
# is_aoe, has_single_target,
# parsed_damage/50, parsed_block/50, parsed_draw/5, parsed_heal/20,
# applies_weak, applies_vulnerable, gives_strength
CARD_FEAT_DIM: int = 16

# Per-enemy features
# hp_ratio, hp/1200, max_hp/1200, block/200,
# intent_damage/80, intent_repeats/5,
# buff_count/10, debuff_count/10, total_power_amount/50,
# has_vulnerable, has_weak, has_strength,
# has_artifact, has_block_intent, has_attack_intent, has_buff_intent
ENEMY_FEAT_DIM: int = 16

# Player power encoding -- named slots for common powers + summary stats
# strength/20, dexterity/20, weak/5, vulnerable/5, frail/5, block/100,
# ritual/5, metallicize/20, thorns/10, plated_armor/20,
# barricade, noxious_fumes/10, poison/20, regen/10,
# total_buff_count/10, total_debuff_count/10,
# total_buff_amount/50, total_debuff_amount/50,
# num_unique_buffs/10, num_unique_debuffs/10
POWER_DIM: int = 20

# ---------------------------------------------------------------------------
# Regex patterns for parsing card effect text
# ---------------------------------------------------------------------------

_RE_DAMAGE = re.compile(r"(\d+)\s*(?:damage|dmg)", re.IGNORECASE)
_RE_BLOCK = re.compile(r"(\d+)\s*block", re.IGNORECASE)
_RE_DRAW = re.compile(r"draw\s*(\d+)|(\d+)\s*card", re.IGNORECASE)
_RE_HEAL = re.compile(r"heal\s*(\d+)|(\d+)\s*(?:hp|heal)", re.IGNORECASE)

_WEAK_KEYWORDS = ("weak", "weakened")
_VULN_KEYWORDS = ("vulnerable", "vuln")
_STRENGTH_KEYWORDS = ("strength", "str ")


def _parse_effect_number(pattern: re.Pattern, text: str) -> float:
    """Extract the first number matching *pattern* from effect text."""
    m = pattern.search(text)
    if m is None:
        return 0.0
    # Try each group until a match is found
    for g in m.groups():
        if g is not None:
            try:
                return float(g)
            except (TypeError, ValueError):
                continue
    return 0.0


def _effect_has_keyword(text: str, keywords: tuple[str, ...]) -> bool:
    """Check if lowercased *text* contains any of *keywords*."""
    t = text.lower()
    return any(kw in t for kw in keywords)


# ---------------------------------------------------------------------------
# Known power name -> slot index mapping for player powers
# ---------------------------------------------------------------------------

_PLAYER_POWER_SLOTS: dict[str, int] = {
    "strength": 0,
    "dexterity": 1,
    "weak": 2,
    "weakened": 2,
    "vulnerable": 3,
    "frail": 4,
    "block": 5,
    "ritual": 6,
    "metallicize": 7,
    "thorns": 8,
    "plated armor": 9,
    "platedarmor": 9,
    "barricade": 10,
    "noxious fumes": 11,
    "noxiousfumes": 11,
    "poison": 12,
    "regen": 13,
    "regeneration": 13,
}

# Slot indices for the named powers (0-13), followed by summary stats (14-19)
_POWER_NAMED_SLOTS = 14

# Per-named-slot normalization divisors
_POWER_SLOT_DIVISORS: list[float] = [
    20.0,   # 0: strength
    20.0,   # 1: dexterity
    5.0,    # 2: weak
    5.0,    # 3: vulnerable
    5.0,    # 4: frail
    100.0,  # 5: block
    5.0,    # 6: ritual
    20.0,   # 7: metallicize
    10.0,   # 8: thorns
    20.0,   # 9: plated armor
    1.0,    # 10: barricade (boolean)
    10.0,   # 11: noxious fumes
    20.0,   # 12: poison
    10.0,   # 13: regen
]

# Powers considered buffs (positive for the player)
_BUFF_POWERS = frozenset({
    "strength", "dexterity", "ritual", "metallicize", "thorns",
    "plated armor", "platedarmor", "barricade", "noxious fumes",
    "noxiousfumes", "regen", "regeneration",
})

# Powers considered debuffs
_DEBUFF_POWERS = frozenset({
    "weak", "weakened", "vulnerable", "frail", "poison",
})


# ---------------------------------------------------------------------------
# DictObservationEncoder
# ---------------------------------------------------------------------------

class DictObservationEncoder:
    """Encodes a bridge observation dict into gymnasium Dict observation.

    Usage::

        enc = DictObservationEncoder()
        space = enc.obs_space            # gymnasium.spaces.Dict
        obs   = enc.encode(obs_dict)     # dict of np.ndarrays
    """

    SCALAR_DIM = SCALAR_DIM
    CARD_FEAT_DIM = CARD_FEAT_DIM
    ENEMY_FEAT_DIM = ENEMY_FEAT_DIM
    POWER_DIM = POWER_DIM

    # ---- public API -------------------------------------------------------

    @property
    def obs_space(self):
        """Return a gymnasium Dict describing the observation structure."""
        import gymnasium
        from gymnasium import spaces

        return spaces.Dict({
            "scalars": spaces.Box(0.0, 1.0, (SCALAR_DIM,), dtype=np.float32),
            "hand": spaces.Box(0.0, 1.0, (MAX_HAND, CARD_FEAT_DIM), dtype=np.float32),
            "hand_mask": spaces.Box(0.0, 1.0, (MAX_HAND,), dtype=np.float32),
            "enemies": spaces.Box(0.0, 1.0, (MAX_ENEMIES, ENEMY_FEAT_DIM), dtype=np.float32),
            "enemy_mask": spaces.Box(0.0, 1.0, (MAX_ENEMIES,), dtype=np.float32),
            "player_powers": spaces.Box(0.0, 1.0, (POWER_DIM,), dtype=np.float32),
        })

    def observation_space(self):
        """Alias for obs_space property."""
        return self.obs_space

    def encode(self, obs: dict | None) -> dict[str, np.ndarray]:
        """Convert an observation dict to a Dict of float32 arrays."""
        scalars = np.zeros(SCALAR_DIM, dtype=np.float32)
        hand = np.zeros((MAX_HAND, CARD_FEAT_DIM), dtype=np.float32)
        hand_mask = np.zeros(MAX_HAND, dtype=np.float32)
        enemies = np.zeros((MAX_ENEMIES, ENEMY_FEAT_DIM), dtype=np.float32)
        enemy_mask = np.zeros(MAX_ENEMIES, dtype=np.float32)
        player_powers = np.zeros(POWER_DIM, dtype=np.float32)

        if obs is not None:
            self._encode_scalars(scalars, obs)
            self._encode_hand(hand, hand_mask, obs)
            self._encode_enemies(enemies, enemy_mask, obs)
            self._encode_player_powers(player_powers, obs)

        return {
            "scalars": scalars,
            "hand": hand,
            "hand_mask": hand_mask,
            "enemies": enemies,
            "enemy_mask": enemy_mask,
            "player_powers": player_powers,
        }

    # ---- scalar encoding (55 dims) ----------------------------------------

    def _encode_scalars(self, vec: np.ndarray, obs: dict) -> None:
        """Encode phase + run + player + combat + decision + summary."""
        off = 0
        off = self._encode_phase(vec, off, obs)       # 17
        off = self._encode_run(vec, off, obs)          # 6
        off = self._encode_player(vec, off, obs)       # 8
        off = self._encode_combat_state(vec, off, obs) # 8
        off = self._encode_decision(vec, off, obs)     # 10
        off = self._encode_summary(vec, off, obs)      # 6
        assert off == SCALAR_DIM, f"scalar offset {off} != {SCALAR_DIM}"

    def _encode_phase(self, vec: np.ndarray, off: int, obs: dict) -> int:
        phase = obs.get("phase", "")
        idx = PHASE_TO_IDX.get(phase, -1)
        if 0 <= idx < NUM_PHASES:
            vec[off + idx] = 1.0
        return off + NUM_PHASES  # 17

    def _encode_run(self, vec: np.ndarray, off: int, obs: dict) -> int:
        run = obs.get("run") or {}
        vec[off + 0] = _bool(run.get("active"))
        vec[off + 1] = _bool(run.get("game_over"))
        act_num = self._parse_act_number(run.get("act_id"))
        vec[off + 2] = min(act_num / 4.0, 1.0)
        vec[off + 3] = min(_float(run.get("act_floor")) / 20.0, 1.0)
        vec[off + 4] = min(_float(run.get("floor")) / 60.0, 1.0)
        room_type = run.get("room_type") or ""
        vec[off + 5] = ROOM_TYPE_TO_ORD.get(room_type, 0) / NUM_ROOM_TYPES
        return off + 6

    def _encode_player(self, vec: np.ndarray, off: int, obs: dict) -> int:
        player = obs.get("player") or {}
        combat = obs.get("combat") or {}
        hp = _float(player.get("hp"))
        max_hp = _float(player.get("max_hp"))
        hp_ratio = hp / max_hp if max_hp > 0 else 0.0
        energy = _float(combat.get("energy"))
        max_energy = _float(combat.get("max_energy"))
        energy_ratio = energy / max_energy if max_energy > 0 else 0.0
        vec[off + 0] = min(hp_ratio, 1.0)
        vec[off + 1] = min(hp / 100.0, 1.0)
        vec[off + 2] = min(max_hp / 100.0, 1.0)
        vec[off + 3] = min(_float(player.get("block")) / 100.0, 1.0)
        vec[off + 4] = min(_float(player.get("gold")) / 500.0, 1.0)
        vec[off + 5] = min(energy_ratio, 1.0)
        vec[off + 6] = min(energy / 10.0, 1.0)
        vec[off + 7] = min(_float(combat.get("stars")) / 10.0, 1.0)
        return off + 8

    def _encode_combat_state(self, vec: np.ndarray, off: int, obs: dict) -> int:
        combat = obs.get("combat")
        if combat is None:
            return off + 8
        vec[off + 0] = 1.0  # in_combat
        vec[off + 1] = min(_float(combat.get("round")) / 20.0, 1.0)
        vec[off + 2] = _bool(combat.get("play_phase"))
        vec[off + 3] = _bool(combat.get("can_act"))
        hand = combat.get("hand") or []
        vec[off + 4] = min(len(hand) / 10.0, 1.0)
        vec[off + 5] = min(_float(combat.get("draw")) / 40.0, 1.0)
        vec[off + 6] = min(_float(combat.get("discard")) / 40.0, 1.0)
        vec[off + 7] = min(_float(combat.get("exhaust")) / 20.0, 1.0)
        return off + 8

    def _encode_decision(self, vec: np.ndarray, off: int, obs: dict) -> int:
        dec = obs.get("decision")
        if dec is None or not isinstance(dec, dict):
            return off + 10
        vec[off + 0] = min(_float(dec.get("option_count")) / 10.0, 1.0)
        vec[off + 1] = _bool(dec.get("can_skip"))
        vec[off + 2] = min(_float(dec.get("selected_count")) / 5.0, 1.0)
        vec[off + 3] = min(_float(dec.get("min_select")) / 5.0, 1.0)
        vec[off + 4] = min(_float(dec.get("max_select")) / 5.0, 1.0)
        vec[off + 5] = _bool(dec.get("is_open"))
        vec[off + 6] = min(_float(dec.get("travelable_count")) / 10.0, 1.0)
        vec[off + 7] = _bool(dec.get("can_proceed") or dec.get("proceed_only"))
        vec[off + 8] = min(_float(dec.get("reward_count")) / 10.0, 1.0)
        vec[off + 9] = min(_float(dec.get("item_count")) / 20.0, 1.0)
        return off + 10

    def _encode_summary(self, vec: np.ndarray, off: int, obs: dict) -> int:
        player = obs.get("player") or {}
        combat = obs.get("combat")
        deck_size = _float(player.get("deck"))
        relics = player.get("relics") or []
        potions = player.get("potions") or []
        relic_count = len(relics) if isinstance(relics, list) else 0
        potion_count = sum(
            1 for p in potions if isinstance(p, str) and p != "[empty]"
        ) if isinstance(potions, list) else 0
        potion_empty = sum(
            1 for p in potions if isinstance(p, str) and p == "[empty]"
        ) if isinstance(potions, list) else 0
        hand = (combat.get("hand") if combat else None) or []
        costs = []
        for card in hand:
            if isinstance(card, dict):
                c = card.get("cost")
                if c is not None and not card.get("x_cost"):
                    costs.append(_float(c))
        avg_cost = (sum(costs) / len(costs)) if costs else 0.0
        vec[off + 0] = min(deck_size / 50.0, 1.0)
        vec[off + 1] = min(relic_count / 20.0, 1.0)
        vec[off + 2] = min(potion_count / 5.0, 1.0)
        vec[off + 3] = min(potion_empty / 5.0, 1.0)
        vec[off + 4] = min(relic_count / 20.0, 1.0)
        vec[off + 5] = min(avg_cost / 5.0, 1.0)
        return off + 6

    # ---- hand card encoding -----------------------------------------------

    def _encode_hand(
        self,
        hand_arr: np.ndarray,
        hand_mask: np.ndarray,
        obs: dict,
    ) -> None:
        """Encode up to MAX_HAND cards into (MAX_HAND, CARD_FEAT_DIM)."""
        combat = obs.get("combat")
        cards = (combat.get("hand") if combat else None) or []

        for i in range(min(len(cards), MAX_HAND)):
            card = cards[i]
            if not isinstance(card, dict):
                continue

            hand_mask[i] = 1.0
            row = hand_arr[i]

            # cost / 5
            cost = card.get("cost")
            card_type = (card.get("type") or "").capitalize()
            is_x = bool(card.get("x_cost") or card.get("star_x"))

            row[0] = min(_float(cost) / 5.0, 1.0) if not is_x else 0.0
            row[1] = 1.0 if card_type == "Attack" else 0.0
            row[2] = 1.0 if card_type == "Skill" else 0.0
            row[3] = 1.0 if card_type == "Power" else 0.0
            row[4] = 1.0 if is_x else 0.0

            # star_cost (the star/mana variant cost)
            star_cost = card.get("star")
            row[5] = min(_float(star_cost) / 5.0, 1.0) if star_cost is not None else 0.0

            # star_x flag
            row[6] = 1.0 if card.get("star_x") else 0.0

            # Target encoding
            target = (card.get("target") or "").lower()
            row[7] = 1.0 if ("all" in target or "allenem" in target) else 0.0
            row[8] = 1.0 if ("single" in target) else 0.0

            # Parse effect text for numeric values
            effect = card.get("effect") or card.get("description") or ""
            if isinstance(effect, str) and effect:
                row[9] = min(_parse_effect_number(_RE_DAMAGE, effect) / 50.0, 1.0)
                row[10] = min(_parse_effect_number(_RE_BLOCK, effect) / 50.0, 1.0)
                row[11] = min(_parse_effect_number(_RE_DRAW, effect) / 5.0, 1.0)
                row[12] = min(_parse_effect_number(_RE_HEAL, effect) / 20.0, 1.0)
                row[13] = 1.0 if _effect_has_keyword(effect, _WEAK_KEYWORDS) else 0.0
                row[14] = 1.0 if _effect_has_keyword(effect, _VULN_KEYWORDS) else 0.0
                row[15] = 1.0 if _effect_has_keyword(effect, _STRENGTH_KEYWORDS) else 0.0

    # ---- enemy encoding ---------------------------------------------------

    def _encode_enemies(
        self,
        enemy_arr: np.ndarray,
        enemy_mask: np.ndarray,
        obs: dict,
    ) -> None:
        """Encode up to MAX_ENEMIES enemies into (MAX_ENEMIES, ENEMY_FEAT_DIM)."""
        combat = obs.get("combat")
        enemies = (combat.get("enemies") if combat else None) or []

        for i in range(min(len(enemies), MAX_ENEMIES)):
            e = enemies[i]
            if not isinstance(e, dict):
                continue

            enemy_mask[i] = 1.0
            row = enemy_arr[i]

            hp = _float(e.get("hp"))
            max_hp = _float(e.get("max_hp"))
            hp_ratio = hp / max_hp if max_hp > 0 else 0.0

            intent = e.get("intent") or {}
            powers = e.get("powers") or []

            row[0] = min(hp_ratio, 1.0)
            row[1] = min(hp / 1200.0, 1.0)
            row[2] = min(max_hp / 1200.0, 1.0)
            row[3] = min(_float(e.get("block")) / 200.0, 1.0)
            row[4] = min(_float(intent.get("total_damage")) / 80.0, 1.0)
            row[5] = min(_float(intent.get("repeats")) / 5.0, 1.0)

            # Power summary
            buff_count = 0
            debuff_count = 0
            total_amount = 0.0
            has_vuln = False
            has_weak = False
            has_str = False
            has_artifact = False

            for p in powers:
                if not isinstance(p, dict):
                    continue
                pname = (p.get("name") or p.get("id") or "").lower()
                amount = _float(p.get("amount"))
                total_amount += abs(amount)

                if pname in ("vulnerable", "vuln"):
                    has_vuln = True
                    debuff_count += 1
                elif pname in ("weak", "weakened"):
                    has_weak = True
                    debuff_count += 1
                elif pname in ("strength", "str"):
                    has_str = True
                    buff_count += 1
                elif pname in ("artifact",):
                    has_artifact = True
                    buff_count += 1
                elif pname in ("frail", "poison"):
                    debuff_count += 1
                else:
                    # Default: positive amounts are buffs, negative are debuffs
                    if amount >= 0:
                        buff_count += 1
                    else:
                        debuff_count += 1

            row[6] = min(buff_count / 10.0, 1.0)
            row[7] = min(debuff_count / 10.0, 1.0)
            row[8] = min(total_amount / 50.0, 1.0)
            row[9] = 1.0 if has_vuln else 0.0
            row[10] = 1.0 if has_weak else 0.0
            row[11] = 1.0 if has_str else 0.0
            row[12] = 1.0 if has_artifact else 0.0

            # Intent type encoding
            intent_type = (intent.get("type") or intent.get("intent") or "").lower()
            row[13] = 1.0 if "block" in intent_type or "defend" in intent_type else 0.0
            row[14] = 1.0 if "attack" in intent_type or "damage" in intent_type else 0.0
            row[15] = 1.0 if "buff" in intent_type or "strategic" in intent_type else 0.0

    # ---- player power encoding --------------------------------------------

    def _encode_player_powers(
        self,
        power_vec: np.ndarray,
        obs: dict,
    ) -> None:
        """Encode player powers into named slots + summary stats."""
        combat = obs.get("combat")
        if combat is None:
            return

        powers = combat.get("player_powers") or []
        if not powers:
            # Also check under player path
            player = obs.get("player") or {}
            powers = player.get("powers") or []

        if not isinstance(powers, list):
            return

        total_buff_count = 0
        total_debuff_count = 0
        total_buff_amount = 0.0
        total_debuff_amount = 0.0
        unique_buffs = set()
        unique_debuffs = set()

        for p in powers:
            if not isinstance(p, dict):
                continue

            pname_raw = p.get("name") or p.get("id") or ""
            pname = pname_raw.lower().strip()
            amount = _float(p.get("amount"))

            # Try to place into a named slot
            slot = _PLAYER_POWER_SLOTS.get(pname)
            if slot is not None and slot < _POWER_NAMED_SLOTS:
                divisor = _POWER_SLOT_DIVISORS[slot]
                power_vec[slot] = min(abs(amount) / divisor, 1.0)

            # Categorize as buff or debuff
            if pname in _DEBUFF_POWERS:
                total_debuff_count += 1
                total_debuff_amount += abs(amount)
                unique_debuffs.add(pname)
            elif pname in _BUFF_POWERS:
                total_buff_count += 1
                total_buff_amount += abs(amount)
                unique_buffs.add(pname)
            else:
                # Unknown power: classify by amount sign
                if amount >= 0:
                    total_buff_count += 1
                    total_buff_amount += abs(amount)
                    unique_buffs.add(pname)
                else:
                    total_debuff_count += 1
                    total_debuff_amount += abs(amount)
                    unique_debuffs.add(pname)

        # Summary stats (slots 14-19)
        power_vec[14] = min(total_buff_count / 10.0, 1.0)
        power_vec[15] = min(total_debuff_count / 10.0, 1.0)
        power_vec[16] = min(total_buff_amount / 50.0, 1.0)
        power_vec[17] = min(total_debuff_amount / 50.0, 1.0)
        power_vec[18] = min(len(unique_buffs) / 10.0, 1.0)
        power_vec[19] = min(len(unique_debuffs) / 10.0, 1.0)

    # ---- helpers ----------------------------------------------------------

    @staticmethod
    def _parse_act_number(act_id: str | None) -> float:
        if not act_id or not isinstance(act_id, str):
            return 0.0
        digits = ""
        for ch in reversed(act_id):
            if ch.isdigit():
                digits = ch + digits
            else:
                break
        if digits:
            try:
                return float(digits)
            except ValueError:
                pass
        return 0.0
