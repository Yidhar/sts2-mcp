"""Dict observation encoder for STS2 RL training.

Three information streams:
  - State: scalars + hand/enemy numeric+text + relics/potions text + powers
  - Actions: per-action numeric+text features
  - Context: decision_text for current phase

Text embeddings use bridge-provided canonical_text (Chinese).
Python does NOT assemble primary text — bridge is the canonical source.
"""

from __future__ import annotations

import numpy as np

from .text_encoder import TEXT_DIM

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

# Phase vocabulary
PHASES = [
    "combat", "map", "reward", "card_reward", "event", "event_crystal_sphere",
    "rest_site", "deck_upgrade", "card_selection", "shop", "treasure",
    "actions", "settling", "terminal",
    "startup_main_menu", "startup_run_mode", "startup_character_select",
]
PHASE_TO_IDX = {p: i for i, p in enumerate(PHASES)}
NUM_PHASES = len(PHASES)  # 17

ROOM_TYPES = ["Monster", "Elite", "Boss", "Event", "Rest", "Merchant", "Treasure"]
ROOM_TYPE_TO_ORD = {r: i + 1 for i, r in enumerate(ROOM_TYPES)}
NUM_ROOM_TYPES = len(ROOM_TYPES) + 1

# Slot limits
MAX_HAND = 12
MAX_ENEMIES = 5
MAX_RELICS = 20
MAX_POTIONS = 5
MAX_ACTIONS = 50

# Section dimensions
SCALAR_DIM = 55         # phase(17) + run(6) + player(8) + combat(8) + decision(10) + summary(6)
CARD_FEAT_DIM = 10      # numeric per-card
ENEMY_FEAT_DIM = 10     # numeric per-enemy
POWER_DIM = 20          # player power slots
ACTION_FEAT_DIM = 16    # numeric per-action (expanded)

# Action kind ordinal
_ACTION_KINDS = [
    "play_card", "use_potion", "discard_potion", "combat",
    "reward", "card_reward", "event_option", "map",
    "rest_site", "deck_upgrade", "card_selection", "shop",
    "treasure_relic", "treasure", "character_select",
    "run_mode_selection", "main_menu", "proceed",
]
_KIND_TO_ORD = {k: i + 1 for i, k in enumerate(_ACTION_KINDS)}
_NUM_KINDS = len(_ACTION_KINDS) + 1

# Map point type ordinal
_MAP_POINT_TYPES = ["Monster", "Elite", "Boss", "Event", "Rest", "Merchant", "Treasure"]
_PT_TO_ORD = {p: i + 1 for i, p in enumerate(_MAP_POINT_TYPES)}
_NUM_PT = len(_MAP_POINT_TYPES) + 1

# Known player powers
_PLAYER_POWERS = [
    "Strength", "Dexterity", "Weak", "Vulnerable", "Frail",
    "Plating", "Ritual", "Metallicize", "Barricade", "Rage",
    "Vigor", "Intangible", "Thorns", "Regen",
]
_POWER_TO_IDX = {p.lower(): i for i, p in enumerate(_PLAYER_POWERS)}

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _float(val, default=0.0):
    if val is None: return default
    try: return float(val)
    except (TypeError, ValueError): return default

def _bool(val):
    return 1.0 if val else 0.0

# ---------------------------------------------------------------------------
# DictObservationEncoder
# ---------------------------------------------------------------------------

class DictObservationEncoder:
    """Encodes bridge observation + legal_actions into gymnasium Dict.

    Args:
        use_text: Enable text embedding via sentence-transformers.
    """

    def __init__(self, use_text: bool = True):
        self.use_text = use_text
        self._encoder = None

    def _get_encoder(self):
        if self._encoder is None and self.use_text:
            from .text_encoder import get_text_encoder
            self._encoder = get_text_encoder().ensure_ready()
        return self._encoder

    @property
    def obs_space(self):
        from gymnasium import spaces
        inf = np.inf
        return spaces.Dict({
            "scalars": spaces.Box(0, 1, (SCALAR_DIM,), dtype=np.float32),
            "hand": spaces.Box(-inf, inf, (MAX_HAND, CARD_FEAT_DIM), dtype=np.float32),
            "hand_text": spaces.Box(-inf, inf, (MAX_HAND, TEXT_DIM), dtype=np.float32),
            "hand_mask": spaces.Box(0, 1, (MAX_HAND,), dtype=np.float32),
            "enemies": spaces.Box(-inf, inf, (MAX_ENEMIES, ENEMY_FEAT_DIM), dtype=np.float32),
            "enemy_text": spaces.Box(-inf, inf, (MAX_ENEMIES, TEXT_DIM), dtype=np.float32),
            "enemy_mask": spaces.Box(0, 1, (MAX_ENEMIES,), dtype=np.float32),
            "player_powers": spaces.Box(0, 1, (POWER_DIM,), dtype=np.float32),
            "relics": spaces.Box(-inf, inf, (MAX_RELICS, TEXT_DIM), dtype=np.float32),
            "relic_mask": spaces.Box(0, 1, (MAX_RELICS,), dtype=np.float32),
            "potions": spaces.Box(-inf, inf, (MAX_POTIONS, TEXT_DIM), dtype=np.float32),
            "potion_mask": spaces.Box(0, 1, (MAX_POTIONS,), dtype=np.float32),
            "context_text": spaces.Box(-inf, inf, (TEXT_DIM,), dtype=np.float32),
            "actions": spaces.Box(-inf, inf, (MAX_ACTIONS, ACTION_FEAT_DIM), dtype=np.float32),
            "action_text": spaces.Box(-inf, inf, (MAX_ACTIONS, TEXT_DIM), dtype=np.float32),
            "action_mask": spaces.Box(0, 1, (MAX_ACTIONS,), dtype=np.float32),
        })

    def encode(self, obs: dict | None, legal_actions: list | None = None) -> dict[str, np.ndarray]:
        s = np.zeros(SCALAR_DIM, dtype=np.float32)
        h = np.zeros((MAX_HAND, CARD_FEAT_DIM), dtype=np.float32)
        ht = np.zeros((MAX_HAND, TEXT_DIM), dtype=np.float32)
        hm = np.zeros(MAX_HAND, dtype=np.float32)
        e = np.zeros((MAX_ENEMIES, ENEMY_FEAT_DIM), dtype=np.float32)
        et = np.zeros((MAX_ENEMIES, TEXT_DIM), dtype=np.float32)
        em = np.zeros(MAX_ENEMIES, dtype=np.float32)
        pp = np.zeros(POWER_DIM, dtype=np.float32)
        r = np.zeros((MAX_RELICS, TEXT_DIM), dtype=np.float32)
        rm = np.zeros(MAX_RELICS, dtype=np.float32)
        p = np.zeros((MAX_POTIONS, TEXT_DIM), dtype=np.float32)
        pm = np.zeros(MAX_POTIONS, dtype=np.float32)
        ct = np.zeros(TEXT_DIM, dtype=np.float32)
        a = np.zeros((MAX_ACTIONS, ACTION_FEAT_DIM), dtype=np.float32)
        at = np.zeros((MAX_ACTIONS, TEXT_DIM), dtype=np.float32)
        am = np.zeros(MAX_ACTIONS, dtype=np.float32)

        if obs:
            self._enc_scalars(s, obs)
            self._enc_hand(h, ht, hm, obs)
            self._enc_enemies(e, et, em, obs)
            self._enc_powers(pp, obs)
            self._enc_relics(r, rm, obs)
            self._enc_potions(p, pm, obs)
            self._enc_context(ct, obs)

        if legal_actions:
            self._enc_actions(a, at, am, legal_actions)

        return {
            "scalars": s, "hand": h, "hand_text": ht, "hand_mask": hm,
            "enemies": e, "enemy_text": et, "enemy_mask": em,
            "player_powers": pp,
            "relics": r, "relic_mask": rm, "potions": p, "potion_mask": pm,
            "context_text": ct,
            "actions": a, "action_text": at, "action_mask": am,
        }

    # ---- Scalars (55 dims) ------------------------------------------------

    def _enc_scalars(self, v, obs):
        off = 0
        # Phase one-hot (17)
        phase = obs.get("phase", "")
        idx = PHASE_TO_IDX.get(phase, -1)
        if 0 <= idx < NUM_PHASES:
            v[off + idx] = 1.0
        off += NUM_PHASES

        # Run (6)
        run = obs.get("run") or {}
        v[off] = _bool(run.get("active"))
        v[off+1] = _bool(run.get("game_over"))
        v[off+2] = min(self._parse_act(run.get("act_id")) / 4.0, 1.0)
        v[off+3] = min(_float(run.get("act_floor")) / 20.0, 1.0)
        v[off+4] = min(_float(run.get("floor")) / 48.0, 1.0)
        v[off+5] = ROOM_TYPE_TO_ORD.get(run.get("room_type", ""), 0) / NUM_ROOM_TYPES
        off += 6

        # Player (8)
        player = obs.get("player") or {}
        combat = obs.get("combat") or {}
        hp = _float(player.get("hp"))
        max_hp = _float(player.get("max_hp"))
        v[off] = min(hp / max_hp, 1.0) if max_hp > 0 else 0.0
        v[off+1] = min(hp / 100.0, 1.0)
        v[off+2] = min(max_hp / 100.0, 1.0)
        v[off+3] = min(_float(player.get("block")) / 100.0, 1.0)
        v[off+4] = min(_float(player.get("gold")) / 500.0, 1.0)
        energy = _float(combat.get("energy"))
        max_energy = _float(combat.get("max_energy"))
        v[off+5] = min(energy / max_energy, 1.0) if max_energy > 0 else 0.0
        v[off+6] = min(energy / 10.0, 1.0)
        v[off+7] = min(_float(combat.get("stars")) / 10.0, 1.0)
        off += 8

        # Combat meta (8)
        if combat:
            v[off] = 1.0
            v[off+1] = min(_float(combat.get("round")) / 20.0, 1.0)
            v[off+2] = _bool(combat.get("play_phase"))
            v[off+3] = _bool(combat.get("can_act"))
            hand = combat.get("hand") or []
            v[off+4] = min(len(hand) / 10.0, 1.0)
            v[off+5] = min(_float(combat.get("draw")) / 40.0, 1.0)
            v[off+6] = min(_float(combat.get("discard")) / 40.0, 1.0)
            v[off+7] = min(_float(combat.get("exhaust")) / 20.0, 1.0)
        off += 8

        # Decision (10)
        dec = obs.get("decision") or {}
        if isinstance(dec, dict):
            v[off] = min(_float(dec.get("option_count")) / 10.0, 1.0)
            v[off+1] = _bool(dec.get("can_skip"))
            v[off+2] = min(_float(dec.get("selected_count")) / 5.0, 1.0)
            v[off+3] = min(_float(dec.get("min_select")) / 5.0, 1.0)
            v[off+4] = min(_float(dec.get("max_select")) / 5.0, 1.0)
            v[off+5] = _bool(dec.get("is_open"))
            v[off+6] = min(_float(dec.get("travelable_count")) / 10.0, 1.0)
            v[off+7] = _bool(dec.get("can_proceed") or dec.get("proceed_only"))
            v[off+8] = min(_float(dec.get("reward_count")) / 10.0, 1.0)
            v[off+9] = min(_float(dec.get("item_count")) / 20.0, 1.0)
        off += 10

        # Summary (6)
        relics = player.get("relics") or []
        potions = player.get("potions") or []
        relic_count = len(relics) if isinstance(relics, list) else 0
        potion_count = sum(1 for p in (potions if isinstance(potions, list) else [])
                          if isinstance(p, (str, dict)) and (p if isinstance(p, str) else p.get("title", "")) != "[empty]")
        deck = player.get("deck")
        deck_size = len(deck) if isinstance(deck, list) else _float(deck)
        v[off] = min(deck_size / 50.0, 1.0)
        v[off+1] = min(relic_count / 20.0, 1.0)
        v[off+2] = min(potion_count / 5.0, 1.0)
        v[off+3] = min((len(potions) - potion_count) / 5.0, 1.0) if isinstance(potions, list) else 0.0
        v[off+4] = min(len(potions) / 5.0, 1.0) if isinstance(potions, list) else 0.0  # total potion slots
        v[off+5] = min(_float(player.get("gold")) / 999.0, 1.0)  # gold ratio (coarser)
        off += 6
        assert off == SCALAR_DIM

    # ---- Hand cards -------------------------------------------------------

    def _enc_hand(self, h, ht, hm, obs):
        combat = obs.get("combat") or {}
        hand = combat.get("hand") or []
        texts = []
        text_slots = []
        for i, card in enumerate(hand[:MAX_HAND]):
            if not isinstance(card, dict): continue
            hm[i] = 1.0
            row = h[i]
            row[0] = min(_float(card.get("cost")) / 5.0, 1.0)
            ctype = (card.get("type") or "").capitalize()
            row[1] = 1.0 if ctype == "Attack" else 0.0
            row[2] = 1.0 if ctype == "Skill" else 0.0
            row[3] = 1.0 if ctype == "Power" else 0.0
            row[4] = _bool(card.get("x_cost"))
            row[5] = min(_float(card.get("star")) / 5.0, 1.0) if card.get("star") is not None else 0.0
            row[6] = _bool(card.get("star_x"))
            target = (card.get("target") or "").lower()
            row[7] = 1.0 if "single" in target or "anyenemy" in target else 0.0
            row[8] = 1.0 if "all" in target else 0.0
            row[9] = 1.0 if "self" in target else 0.0  # only explicit self-target
            # Text: use bridge canonical_text
            ct = card.get("canonical_text", "")
            if ct and self.use_text:
                texts.append(ct)
                text_slots.append(i)
        if texts:
            embs = self._get_encoder().encode_batch(texts)
            for idx, slot in enumerate(text_slots):
                ht[slot] = embs[idx]

    # ---- Enemies ----------------------------------------------------------

    def _enc_enemies(self, e, et, em, obs):
        combat = obs.get("combat") or {}
        enemies = combat.get("enemies") or []
        texts = []
        text_slots = []
        for i, en in enumerate(enemies[:MAX_ENEMIES]):
            if not isinstance(en, dict): continue
            em[i] = 1.0
            row = e[i]
            hp = _float(en.get("hp"))
            max_hp = _float(en.get("max_hp"))
            row[0] = min(hp / max_hp, 1.0) if max_hp > 0 else 0.0
            row[1] = min(hp / 1200.0, 1.0)
            row[2] = min(max_hp / 1200.0, 1.0)
            row[3] = min(_float(en.get("block")) / 200.0, 1.0)
            intent = en.get("intent") or {}
            row[4] = min(_float(intent.get("total_damage")) / 80.0, 1.0)
            row[5] = min(_float(intent.get("repeats")) / 5.0, 1.0)
            powers = en.get("powers") or []
            row[6] = min(len(powers) / 5.0, 1.0)
            for p in powers:
                if not isinstance(p, dict): continue
                t = (p.get("title") or "").lower()
                if "vulnerable" in t: row[7] = 1.0
                elif "weak" in t: row[8] = 1.0
                elif "strength" in t: row[9] = 1.0
            # Text: build from name + intent + powers
            if self.use_text:
                parts = [en.get("name", "")]
                if intent.get("total_damage"):
                    parts.append(f"意图:{intent['total_damage']}伤害")
                for p in powers[:3]:
                    if isinstance(p, dict):
                        parts.append(f"{p.get('title','')}:{p.get('amount','')}")
                text = "｜".join(p for p in parts if p)
                if text:
                    texts.append(text)
                    text_slots.append(i)
        if texts:
            embs = self._get_encoder().encode_batch(texts)
            for idx, slot in enumerate(text_slots):
                et[slot] = embs[idx]

    # ---- Player powers ----------------------------------------------------

    def _enc_powers(self, pp, obs):
        combat = obs.get("combat") or {}
        # Bridge now provides player_powers directly in the combat payload
        powers = combat.get("player_powers") or []

        buff_count = 0
        debuff_count = 0
        for p in powers:
            if not isinstance(p, dict): continue
            title = (p.get("title") or "").lower()
            amount = _float(p.get("amount"))
            idx = _POWER_TO_IDX.get(title)
            if idx is not None and idx < len(_PLAYER_POWERS):
                pp[idx] = min(abs(amount) / 20.0, 1.0)
            if title in ("weak", "vulnerable", "frail"):
                debuff_count += 1
            else:
                buff_count += 1
        # Summary slots at end
        if POWER_DIM > len(_PLAYER_POWERS):
            pp[len(_PLAYER_POWERS)] = min(buff_count / 10.0, 1.0)
            pp[len(_PLAYER_POWERS)+1] = min(debuff_count / 10.0, 1.0)

    # ---- Relics -----------------------------------------------------------

    def _enc_relics(self, r, rm, obs):
        player = obs.get("player") or {}
        relics = player.get("relics") or []
        if not isinstance(relics, list) or not self.use_text:
            return
        texts = []
        text_slots = []
        for i, relic in enumerate(relics[:MAX_RELICS]):
            if isinstance(relic, dict):
                ct = relic.get("canonical_text", "")
                if not ct:
                    ct = relic.get("title", "")
            elif isinstance(relic, str):
                ct = relic  # fallback: old format (just title)
            else:
                continue
            if ct and ct != "[empty]":
                rm[i] = 1.0
                texts.append(ct)
                text_slots.append(i)
        if texts:
            embs = self._get_encoder().encode_batch(texts)
            for idx, slot in enumerate(text_slots):
                r[slot] = embs[idx]

    # ---- Potions ----------------------------------------------------------

    def _enc_potions(self, p, pm, obs):
        player = obs.get("player") or {}
        potions = player.get("potions") or []
        if not isinstance(potions, list) or not self.use_text:
            return
        texts = []
        text_slots = []
        for i, potion in enumerate(potions[:MAX_POTIONS]):
            if isinstance(potion, dict):
                ct = potion.get("canonical_text", "")
                title = potion.get("title", "")
            elif isinstance(potion, str):
                ct = ""
                title = potion
            else:
                continue
            if title == "[empty]" or not (ct or title):
                continue
            pm[i] = 1.0
            texts.append(ct or title)
            text_slots.append(i)
        if texts:
            embs = self._get_encoder().encode_batch(texts)
            for idx, slot in enumerate(text_slots):
                p[slot] = embs[idx]

    # ---- Context text -----------------------------------------------------

    def _enc_context(self, ct, obs):
        if not self.use_text:
            return
        dec = obs.get("decision") or {}
        text = ""
        if isinstance(dec, dict):
            text = dec.get("decision_text", "")
        if not text:
            phase = obs.get("phase", "")
            text = f"阶段：{phase}"
        if text:
            ct[:] = self._get_encoder().encode(text)

    # ---- Actions ----------------------------------------------------------

    def _enc_actions(self, a, at, am, legal_actions):
        texts = []
        text_slots = []
        n = min(len(legal_actions), MAX_ACTIONS)
        for i in range(n):
            act = legal_actions[i]
            if not isinstance(act, dict): continue
            am[i] = 1.0
            self._enc_action_numeric(a[i], act)
            ct = act.get("canonical_text", "")
            if ct and self.use_text:
                texts.append(ct)
                text_slots.append(i)
        if texts:
            embs = self._get_encoder().encode_batch(texts)
            for idx, slot in enumerate(text_slots):
                at[slot] = embs[idx]

    def _enc_action_numeric(self, row, act):
        """Encode one action into ACTION_FEAT_DIM=16 numeric features."""
        kind = act.get("kind", "")
        row[0] = _KIND_TO_ORD.get(kind, 0) / _NUM_KINDS

        card = act.get("card")
        if isinstance(card, dict):
            row[1] = 1.0  # has_card
            row[2] = min(_float(card.get("cost")) / 5.0, 1.0)
            row[3] = min(_float(card.get("star")) / 5.0, 1.0) if card.get("star") is not None else 0.0
            ctype = (card.get("type") or "").capitalize()
            row[4] = 1.0 if ctype == "Attack" else 0.0
            row[5] = 1.0 if ctype == "Skill" else 0.0
            row[6] = 1.0 if ctype == "Power" else 0.0

        target = act.get("target")
        row[7] = 1.0 if isinstance(target, dict) and target.get("name") else 0.0
        row[8] = 1.0 if isinstance(target, dict) and target.get("side") == "Player" else 0.0
        row[9] = 1.0 if act.get("action_id") == "end_turn" else 0.0
        row[10] = 1.0 if kind == "proceed" and not act.get("skip") else 0.0
        row[11] = 1.0 if act.get("skip") or "skip" in (act.get("action_id") or "") else 0.0

        # Shop cost
        item = act.get("item")
        if isinstance(item, dict):
            row[12] = min(_float(item.get("cost")) / 500.0, 1.0)

        # Reward type
        reward = act.get("reward")
        if isinstance(reward, dict):
            rt = reward.get("type", "")
            row[13] = 1.0 if rt == "gold" else 0.0
            row[14] = 1.0 if rt == "card" else 0.0

        # Map point type
        pt = act.get("point_type", "")
        row[15] = _PT_TO_ORD.get(pt, 0) / _NUM_PT

    # ---- Helpers ----------------------------------------------------------

    @staticmethod
    def _parse_act(act_id):
        if not act_id or not isinstance(act_id, str): return 0.0
        for ch in reversed(act_id):
            if ch.isdigit():
                try: return float(ch)
                except: pass
        return 0.0
