"""Dict observation encoder for STS2 RL training.

Observation structure is intentionally split into:
  - shared global context
  - combat entities
  - build/deck entities
  - candidate actions

The bridge is the canonical source for semantic text. Python only embeds
bridge-provided canonical_text and compact decision text.
"""

from __future__ import annotations

import numpy as np

from content_registry import (
    build_enemy_intent_semantic_text,
    build_live_card_semantic_text,
    build_live_enemy_semantic_text,
    build_live_potion_semantic_text,
    build_live_relic_semantic_text,
)

from .text_encoder import TEXT_DIM

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

PHASES = [
    "combat", "map", "reward", "card_reward", "event", "event_crystal_sphere",
    "rest_site", "deck_upgrade", "card_selection", "shop", "treasure",
    "actions", "settling", "terminal",
    "startup_main_menu", "startup_run_mode", "startup_character_select",
]
PHASE_TO_IDX = {phase: index for index, phase in enumerate(PHASES)}
NUM_PHASES = len(PHASES)

DECISION_DOMAINS = ["combat", "build", "route"]
DOMAIN_TO_IDX = {domain: index for index, domain in enumerate(DECISION_DOMAINS)}
NUM_DOMAINS = len(DECISION_DOMAINS)

ROOM_TYPES = ["Monster", "Elite", "Boss", "Event", "Rest", "Merchant", "Treasure"]
ROOM_TYPE_TO_ORD = {room_type: index + 1 for index, room_type in enumerate(ROOM_TYPES)}
NUM_ROOM_TYPES = len(ROOM_TYPES) + 1

MAX_HAND = 12
MAX_DECK = 40
MAX_ENEMIES = 5
MAX_RELICS = 20
MAX_POTIONS = 5
# Single-source action cap for both env wrappers and the policy head.
# 80 is large enough for dense combat turns without exploding tensor size.
MAX_ACTIONS = 80
MAX_ROUTE_NODES = 24

SCALAR_DIM = 61
CARD_FEAT_DIM = 20
DECK_FEAT_DIM = 20
ENEMY_FEAT_DIM = 10
POWER_DIM = 20
ACTION_FEAT_DIM = 32

_ACTION_KINDS = [
    "play_card", "use_potion", "discard_potion", "combat",
    "reward", "card_reward", "event_option", "map",
    "rest_site", "deck_upgrade", "card_selection", "shop",
    "treasure_relic", "treasure", "character_select",
    "run_mode_selection", "main_menu", "proceed",
]
_KIND_TO_ORD = {kind: index + 1 for index, kind in enumerate(_ACTION_KINDS)}
_NUM_KINDS = len(_ACTION_KINDS) + 1
ACTION_KIND_TO_ORD = dict(_KIND_TO_ORD)
NUM_ACTION_KINDS = _NUM_KINDS

_MAP_POINT_TYPES = ["Monster", "Elite", "Boss", "Event", "QuestionMark", "RestSite", "Shop", "Treasure"]
_PT_TO_ORD = {point_type: index + 1 for index, point_type in enumerate(_MAP_POINT_TYPES)}
_NUM_PT = len(_MAP_POINT_TYPES) + 1

_ROUTE_POINT_TYPES = ["Monster", "Elite", "Boss", "Event", "QuestionMark", "RestSite", "Shop", "Treasure"]
_ROUTE_PT_TO_IDX = {point_type: index for index, point_type in enumerate(_ROUTE_POINT_TYPES)}
_NUM_ROUTE_PT = len(_ROUTE_POINT_TYPES)

ROUTE_SUMMARY_DIM = 20
ROUTE_NODE_FEAT_DIM = _NUM_ROUTE_PT + 5

_PLAYER_POWERS = [
    "Strength", "Dexterity", "Weak", "Vulnerable", "Frail",
    "Plating", "Ritual", "Metallicize", "Barricade", "Rage",
    "Vigor", "Intangible", "Thorns", "Regen",
]
_POWER_TO_IDX = {power.lower(): index for index, power in enumerate(_PLAYER_POWERS)}


def _float(val, default=0.0):
    if val is None:
        return default
    try:
        return float(val)
    except (TypeError, ValueError):
        return default


def _bool(val):
    return 1.0 if val else 0.0


def _metric(source, key, default=0.0):
    if not isinstance(source, dict):
        return default
    return _float(source.get(key), default)


class DictObservationEncoder:
    """Encode bridge observation + legal actions into a gym Dict observation."""

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
        return spaces.Dict(
            {
                "scalars": spaces.Box(0, 1, (SCALAR_DIM,), dtype=np.float32),
                "decision_domain": spaces.Box(0, 1, (NUM_DOMAINS,), dtype=np.float32),
                "hand": spaces.Box(-inf, inf, (MAX_HAND, CARD_FEAT_DIM), dtype=np.float32),
                "hand_text": spaces.Box(-inf, inf, (MAX_HAND, TEXT_DIM), dtype=np.float32),
                "hand_mask": spaces.Box(0, 1, (MAX_HAND,), dtype=np.float32),
                "deck": spaces.Box(-inf, inf, (MAX_DECK, DECK_FEAT_DIM), dtype=np.float32),
                "deck_text": spaces.Box(-inf, inf, (MAX_DECK, TEXT_DIM), dtype=np.float32),
                "deck_mask": spaces.Box(0, 1, (MAX_DECK,), dtype=np.float32),
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
                "route_summary": spaces.Box(-inf, inf, (MAX_ACTIONS, ROUTE_SUMMARY_DIM), dtype=np.float32),
                "route_nodes": spaces.Box(-inf, inf, (MAX_ACTIONS, MAX_ROUTE_NODES, ROUTE_NODE_FEAT_DIM), dtype=np.float32),
                "route_node_mask": spaces.Box(0, 1, (MAX_ACTIONS, MAX_ROUTE_NODES), dtype=np.float32),
                "action_mask": spaces.Box(0, 1, (MAX_ACTIONS,), dtype=np.float32),
            }
        )

    def encode(self, obs: dict | None, legal_actions: list | None = None) -> dict[str, np.ndarray]:
        scalars = np.zeros(SCALAR_DIM, dtype=np.float32)
        decision_domain = np.zeros(NUM_DOMAINS, dtype=np.float32)
        hand = np.zeros((MAX_HAND, CARD_FEAT_DIM), dtype=np.float32)
        hand_text = np.zeros((MAX_HAND, TEXT_DIM), dtype=np.float32)
        hand_mask = np.zeros(MAX_HAND, dtype=np.float32)
        deck = np.zeros((MAX_DECK, DECK_FEAT_DIM), dtype=np.float32)
        deck_text = np.zeros((MAX_DECK, TEXT_DIM), dtype=np.float32)
        deck_mask = np.zeros(MAX_DECK, dtype=np.float32)
        enemies = np.zeros((MAX_ENEMIES, ENEMY_FEAT_DIM), dtype=np.float32)
        enemy_text = np.zeros((MAX_ENEMIES, TEXT_DIM), dtype=np.float32)
        enemy_mask = np.zeros(MAX_ENEMIES, dtype=np.float32)
        player_powers = np.zeros(POWER_DIM, dtype=np.float32)
        relics = np.zeros((MAX_RELICS, TEXT_DIM), dtype=np.float32)
        relic_mask = np.zeros(MAX_RELICS, dtype=np.float32)
        potions = np.zeros((MAX_POTIONS, TEXT_DIM), dtype=np.float32)
        potion_mask = np.zeros(MAX_POTIONS, dtype=np.float32)
        context_text = np.zeros(TEXT_DIM, dtype=np.float32)
        actions = np.zeros((MAX_ACTIONS, ACTION_FEAT_DIM), dtype=np.float32)
        action_text = np.zeros((MAX_ACTIONS, TEXT_DIM), dtype=np.float32)
        route_summary = np.zeros((MAX_ACTIONS, ROUTE_SUMMARY_DIM), dtype=np.float32)
        route_nodes = np.zeros((MAX_ACTIONS, MAX_ROUTE_NODES, ROUTE_NODE_FEAT_DIM), dtype=np.float32)
        route_node_mask = np.zeros((MAX_ACTIONS, MAX_ROUTE_NODES), dtype=np.float32)
        action_mask = np.zeros(MAX_ACTIONS, dtype=np.float32)

        if obs:
            self._enc_scalars(scalars, obs, legal_actions or [])
            self._enc_decision_domain(decision_domain, obs)
            self._enc_hand(hand, hand_text, hand_mask, obs)
            self._enc_deck(deck, deck_text, deck_mask, obs)
            self._enc_enemies(enemies, enemy_text, enemy_mask, obs)
            self._enc_powers(player_powers, obs)
            self._enc_relics(relics, relic_mask, obs)
            self._enc_potions(potions, potion_mask, obs)
            self._enc_context(context_text, obs)

        if legal_actions:
            self._enc_actions(actions, action_text, route_summary, route_nodes, route_node_mask, action_mask, legal_actions)

        return {
            "scalars": scalars,
            "decision_domain": decision_domain,
            "hand": hand,
            "hand_text": hand_text,
            "hand_mask": hand_mask,
            "deck": deck,
            "deck_text": deck_text,
            "deck_mask": deck_mask,
            "enemies": enemies,
            "enemy_text": enemy_text,
            "enemy_mask": enemy_mask,
            "player_powers": player_powers,
            "relics": relics,
            "relic_mask": relic_mask,
            "potions": potions,
            "potion_mask": potion_mask,
            "context_text": context_text,
            "actions": actions,
            "action_text": action_text,
            "route_summary": route_summary,
            "route_nodes": route_nodes,
            "route_node_mask": route_node_mask,
            "action_mask": action_mask,
        }

    def _enc_scalars(self, vector: np.ndarray, obs: dict, legal_actions: list) -> None:
        offset = 0

        phase = obs.get("phase", "")
        phase_idx = PHASE_TO_IDX.get(phase, -1)
        if 0 <= phase_idx < NUM_PHASES:
            vector[offset + phase_idx] = 1.0
        offset += NUM_PHASES

        run = obs.get("run") or {}
        vector[offset] = _bool(run.get("active"))
        vector[offset + 1] = _bool(run.get("game_over"))
        vector[offset + 2] = min(self._parse_act(run.get("act_id")) / 4.0, 1.0)
        vector[offset + 3] = min(_float(run.get("act_floor")) / 20.0, 1.0)
        vector[offset + 4] = min(_float(run.get("floor")) / 48.0, 1.0)
        vector[offset + 5] = ROOM_TYPE_TO_ORD.get(run.get("room_type", ""), 0) / NUM_ROOM_TYPES
        offset += 6

        player = obs.get("player") or {}
        combat = obs.get("combat") or {}
        hp = _float(player.get("hp"))
        max_hp = _float(player.get("max_hp"))
        vector[offset] = min(hp / max_hp, 1.0) if max_hp > 0 else 0.0
        vector[offset + 1] = min(hp / 100.0, 1.0)
        vector[offset + 2] = min(max_hp / 100.0, 1.0)
        vector[offset + 3] = min(_float(player.get("block")) / 100.0, 1.0)
        vector[offset + 4] = min(_float(player.get("gold")) / 500.0, 1.0)
        energy = _float(combat.get("energy"))
        max_energy = _float(combat.get("max_energy"))
        vector[offset + 5] = min(energy / max_energy, 1.0) if max_energy > 0 else 0.0
        vector[offset + 6] = min(energy / 10.0, 1.0)
        vector[offset + 7] = min(_float(combat.get("stars")) / 10.0, 1.0)
        offset += 8

        if combat:
            vector[offset] = 1.0
            vector[offset + 1] = min(_float(combat.get("round")) / 20.0, 1.0)
            vector[offset + 2] = _bool(combat.get("play_phase"))
            vector[offset + 3] = _bool(combat.get("can_act"))
            hand = combat.get("hand") or []
            vector[offset + 4] = min(len(hand) / 10.0, 1.0)
            vector[offset + 5] = min(_float(combat.get("draw")) / 40.0, 1.0)
            vector[offset + 6] = min(_float(combat.get("discard")) / 40.0, 1.0)
            vector[offset + 7] = min(_float(combat.get("exhaust")) / 20.0, 1.0)
        offset += 8

        decision = obs.get("decision") or {}
        if isinstance(decision, dict):
            vector[offset] = min(_float(decision.get("option_count")) / 10.0, 1.0)
            vector[offset + 1] = _bool(decision.get("can_skip"))
            vector[offset + 2] = min(_float(decision.get("selected_count")) / 5.0, 1.0)
            vector[offset + 3] = min(_float(decision.get("min_select")) / 5.0, 1.0)
            vector[offset + 4] = min(_float(decision.get("max_select")) / 5.0, 1.0)
            vector[offset + 5] = _bool(decision.get("is_open"))
            vector[offset + 6] = min(_float(decision.get("travelable_count")) / 10.0, 1.0)
            vector[offset + 7] = _bool(decision.get("can_proceed") or decision.get("proceed_only"))
            vector[offset + 8] = min(_float(decision.get("reward_count")) / 10.0, 1.0)
            vector[offset + 9] = min(_float(decision.get("item_count")) / 20.0, 1.0)
        offset += 10

        relics = player.get("relics") or []
        potions = player.get("potions") or []
        relic_count = len(relics) if isinstance(relics, list) else 0
        potion_count = sum(
            1
            for potion in (potions if isinstance(potions, list) else [])
            if isinstance(potion, (str, dict))
            and (potion if isinstance(potion, str) else potion.get("title", "")) != "[empty]"
        )
        deck_cards = player.get("deck_cards")
        deck_size = len(deck_cards) if isinstance(deck_cards, list) else _float(player.get("deck"))
        vector[offset] = min(deck_size / 50.0, 1.0)
        vector[offset + 1] = min(relic_count / 20.0, 1.0)
        vector[offset + 2] = min(potion_count / 5.0, 1.0)
        vector[offset + 3] = min((len(potions) - potion_count) / 5.0, 1.0) if isinstance(potions, list) else 0.0
        vector[offset + 4] = min(len(potions) / 5.0, 1.0) if isinstance(potions, list) else 0.0
        vector[offset + 5] = min(_float(player.get("gold")) / 999.0, 1.0)
        offset += 6

        total_actions = 0
        combat_continue_actions = 0
        play_card_actions = 0
        zero_cost_play_actions = 0
        positive_preview_actions = 0
        has_end_turn = False

        for action in legal_actions:
            if not isinstance(action, dict):
                continue
            total_actions += 1
            action_id = action.get("action_id") or ""
            if action_id == "end_turn":
                has_end_turn = True
                continue

            kind = action.get("kind") or ""
            if kind not in ("play_card", "use_potion"):
                continue

            combat_continue_actions += 1
            source = action.get("card") if kind == "play_card" else action.get("potion")
            if kind == "play_card":
                play_card_actions += 1
                if isinstance(source, dict) and _float(source.get("cost")) == 0:
                    zero_cost_play_actions += 1

            if isinstance(source, dict) and (
                _metric(source, "damage") > 0
                or _metric(source, "block") > 0
                or _metric(source, "draw") > 0
                or _metric(source, "weak") > 0
                or _metric(source, "vulnerable") > 0
                or _metric(source, "heal") > 0
                or _metric(source, "strength") > 0
                or _metric(source, "dexterity") > 0
                or _metric(source, "summon") > 0
            ):
                positive_preview_actions += 1

        vector[offset] = min(total_actions / 50.0, 1.0)
        vector[offset + 1] = min(combat_continue_actions / 20.0, 1.0)
        vector[offset + 2] = min(play_card_actions / 20.0, 1.0)
        vector[offset + 3] = min(zero_cost_play_actions / 10.0, 1.0)
        vector[offset + 4] = min(positive_preview_actions / 10.0, 1.0)
        vector[offset + 5] = _bool(has_end_turn)
        offset += 6

        assert offset == SCALAR_DIM

    def _enc_decision_domain(self, vector: np.ndarray, obs: dict) -> None:
        domain = self._resolve_domain(obs)
        index = DOMAIN_TO_IDX.get(domain, DOMAIN_TO_IDX["build"])
        vector[index] = 1.0

    def _enc_hand(self, hand: np.ndarray, hand_text: np.ndarray, hand_mask: np.ndarray, obs: dict) -> None:
        combat = obs.get("combat") or {}
        cards = combat.get("hand") or []
        self._enc_card_collection(cards, hand, hand_text, hand_mask)

    def _enc_deck(self, deck: np.ndarray, deck_text: np.ndarray, deck_mask: np.ndarray, obs: dict) -> None:
        player = obs.get("player") or {}
        cards = player.get("deck_cards") or []
        self._enc_card_collection(cards, deck, deck_text, deck_mask)

    def _enc_card_collection(
        self,
        cards: list,
        numeric: np.ndarray,
        text: np.ndarray,
        mask: np.ndarray,
    ) -> None:
        max_items = numeric.shape[0]
        texts: list[str] = []
        text_slots: list[int] = []
        for index, card in enumerate(cards[:max_items]):
            if not isinstance(card, dict):
                continue
            mask[index] = 1.0
            row = numeric[index]
            row[0] = min(_float(card.get("cost")) / 5.0, 1.0)
            card_type = (card.get("type") or "").capitalize()
            row[1] = 1.0 if card_type == "Attack" else 0.0
            row[2] = 1.0 if card_type == "Skill" else 0.0
            row[3] = 1.0 if card_type == "Power" else 0.0
            row[4] = _bool(card.get("x_cost"))
            row[5] = min(_float(card.get("star")) / 5.0, 1.0) if card.get("star") is not None else 0.0
            row[6] = _bool(card.get("star_x"))
            target = (card.get("target") or "").lower()
            row[7] = 1.0 if "single" in target or "anyenemy" in target else 0.0
            row[8] = 1.0 if "all" in target else 0.0
            row[9] = 1.0 if "self" in target else 0.0
            row[10] = 1.0 if card_type == "Status" else 0.0
            row[11] = 1.0 if card_type == "Curse" else 0.0
            row[12] = min(_metric(card, "damage") / 50.0, 1.0)
            row[13] = min(_metric(card, "block") / 50.0, 1.0)
            row[14] = min(_metric(card, "draw") / 5.0, 1.0)
            row[15] = min(_metric(card, "weak") / 5.0, 1.0)
            row[16] = min(_metric(card, "vulnerable") / 5.0, 1.0)
            row[17] = min(_metric(card, "heal") / 30.0, 1.0)
            row[18] = min(_metric(card, "hp_loss") / 30.0, 1.0)
            row[19] = min(_metric(card, "summon") / 5.0, 1.0)
            build_aux = card.get("build_aux")
            if isinstance(build_aux, dict):
                option_total = max(_float(build_aux.get("option_total")), 1.0)
                deck_after_size = max(_float(build_aux.get("deck_after_size")), 1.0)
                row[12] = min(_float(build_aux.get("remove_rank")) / option_total, 1.0)
                row[13] = min(_float(build_aux.get("keep_rank")) / option_total, 1.0)
                row[14] = min(_float(build_aux.get("count_before")) / 5.0, 1.0)
                row[15] = min(_float(build_aux.get("count_after")) / 5.0, 1.0)
                row[16] = min(_float(build_aux.get("junk_after")) / deck_after_size, 1.0)
                row[17] = min(_float(build_aux.get("starter_attack_after")) / 10.0, 1.0)
                row[18] = min(_float(build_aux.get("starter_defend_after")) / 10.0, 1.0)
                row[19] = min(_float(build_aux.get("starter_gap_after")) / 10.0, 1.0)
            text_value = self._build_live_card_text(card)
            if text_value and self.use_text:
                texts.append(text_value)
                text_slots.append(index)

        if texts:
            embeddings = self._get_encoder().encode_batch(texts)
            for embedding_index, slot in enumerate(text_slots):
                text[slot] = embeddings[embedding_index]

    def _enc_enemies(self, enemies: np.ndarray, enemy_text: np.ndarray, enemy_mask: np.ndarray, obs: dict) -> None:
        combat = obs.get("combat") or {}
        entries = combat.get("enemies") or []
        texts: list[str] = []
        text_slots: list[int] = []

        for index, enemy in enumerate(entries[:MAX_ENEMIES]):
            if not isinstance(enemy, dict):
                continue
            enemy_mask[index] = 1.0
            row = enemies[index]
            hp = _float(enemy.get("hp"))
            max_hp = _float(enemy.get("max_hp"))
            row[0] = min(hp / max_hp, 1.0) if max_hp > 0 else 0.0
            row[1] = min(hp / 1200.0, 1.0)
            row[2] = min(max_hp / 1200.0, 1.0)
            row[3] = min(_float(enemy.get("block")) / 200.0, 1.0)
            intent = enemy.get("intent") or {}
            row[4] = min(_float(intent.get("total_damage")) / 80.0, 1.0)
            row[5] = min(_float(intent.get("repeats")) / 5.0, 1.0)
            powers = enemy.get("powers") or []
            row[6] = min(len(powers) / 5.0, 1.0)
            for power in powers:
                if not isinstance(power, dict):
                    continue
                title = (power.get("title") or "").lower()
                if "vulnerable" in title:
                    row[7] = 1.0
                elif "weak" in title:
                    row[8] = 1.0
                elif "strength" in title:
                    row[9] = 1.0

            if self.use_text:
                text_value = build_live_enemy_semantic_text(enemy)
                if text_value:
                    texts.append(text_value)
                    text_slots.append(index)

        if texts:
            embeddings = self._get_encoder().encode_batch(texts)
            for embedding_index, slot in enumerate(text_slots):
                enemy_text[slot] = embeddings[embedding_index]

    def _enc_powers(self, vector: np.ndarray, obs: dict) -> None:
        combat = obs.get("combat") or {}
        powers = combat.get("player_powers") or []
        buff_count = 0
        debuff_count = 0

        for power in powers:
            if not isinstance(power, dict):
                continue
            title = (power.get("title") or "").lower()
            amount = _float(power.get("amount"))
            index = _POWER_TO_IDX.get(title)
            if index is not None and index < len(_PLAYER_POWERS):
                vector[index] = min(abs(amount) / 20.0, 1.0)
            if title in ("weak", "vulnerable", "frail"):
                debuff_count += 1
            else:
                buff_count += 1

        if POWER_DIM > len(_PLAYER_POWERS):
            vector[len(_PLAYER_POWERS)] = min(buff_count / 10.0, 1.0)
            vector[len(_PLAYER_POWERS) + 1] = min(debuff_count / 10.0, 1.0)

    def _enc_relics(self, relics: np.ndarray, relic_mask: np.ndarray, obs: dict) -> None:
        player = obs.get("player") or {}
        entries = player.get("relics") or []
        if not isinstance(entries, list) or not self.use_text:
            return

        texts: list[str] = []
        slots: list[int] = []
        for index, relic in enumerate(entries[:MAX_RELICS]):
            if isinstance(relic, dict):
                canonical_text = build_live_relic_semantic_text(relic)
                if not canonical_text:
                    canonical_text = relic.get("canonical_text", "") or relic.get("title", "")
            elif isinstance(relic, str):
                canonical_text = relic
            else:
                continue
            if canonical_text and canonical_text != "[empty]":
                relic_mask[index] = 1.0
                texts.append(canonical_text)
                slots.append(index)

        if texts:
            embeddings = self._get_encoder().encode_batch(texts)
            for embedding_index, slot in enumerate(slots):
                relics[slot] = embeddings[embedding_index]

    def _enc_potions(self, potions: np.ndarray, potion_mask: np.ndarray, obs: dict) -> None:
        player = obs.get("player") or {}
        entries = player.get("potions") or []
        if not isinstance(entries, list) or not self.use_text:
            return

        texts: list[str] = []
        slots: list[int] = []
        for index, potion in enumerate(entries[:MAX_POTIONS]):
            if isinstance(potion, dict):
                canonical_text = build_live_potion_semantic_text(potion)
                if not canonical_text:
                    canonical_text = potion.get("canonical_text", "")
                title = potion.get("title", "")
            elif isinstance(potion, str):
                canonical_text = ""
                title = potion
            else:
                continue
            if title == "[empty]" or not (canonical_text or title):
                continue
            potion_mask[index] = 1.0
            texts.append(canonical_text or title)
            slots.append(index)

        if texts:
            embeddings = self._get_encoder().encode_batch(texts)
            for embedding_index, slot in enumerate(slots):
                potions[slot] = embeddings[embedding_index]

    def _enc_context(self, vector: np.ndarray, obs: dict) -> None:
        if not self.use_text:
            return
        decision = obs.get("decision") or {}
        text = ""
        if isinstance(decision, dict):
            text = decision.get("decision_text", "")
        if not text:
            domain = self._resolve_domain(obs)
            phase = obs.get("phase", "")
            combat = obs.get("combat") or {}
            enemy_intents: list[str] = []
            for enemy in (combat.get("enemies") or [])[:3]:
                if not isinstance(enemy, dict):
                    continue
                intent_text = build_enemy_intent_semantic_text(enemy.get("intent"))
                if intent_text:
                    enemy_intents.append(intent_text)
            if enemy_intents:
                text = f"阶段：{phase}｜决策域：{domain}｜敌人意图：{' || '.join(enemy_intents)}"
            else:
                text = f"阶段：{phase}｜决策域：{domain}"
        vector[:] = self._get_encoder().encode(text)

    def _enc_actions(
        self,
        actions: np.ndarray,
        action_text: np.ndarray,
        route_summary: np.ndarray,
        route_nodes: np.ndarray,
        route_node_mask: np.ndarray,
        action_mask: np.ndarray,
        legal_actions: list,
    ) -> None:
        texts: list[str] = []
        text_slots: list[int] = []
        count = min(len(legal_actions), MAX_ACTIONS)
        for index in range(count):
            action = legal_actions[index]
            if not isinstance(action, dict):
                continue
            action_mask[index] = 1.0
            self._enc_action_numeric(actions[index], action)
            self._enc_route_action(route_summary[index], route_nodes[index], route_node_mask[index], action)
            text_value = self._build_action_text(action)
            if text_value and self.use_text:
                texts.append(text_value)
                text_slots.append(index)

        if texts:
            embeddings = self._get_encoder().encode_batch(texts)
            for embedding_index, slot in enumerate(text_slots):
                action_text[slot] = embeddings[embedding_index]

    def _build_live_card_text(self, card: dict | None) -> str:
        if not isinstance(card, dict):
            return ""
        semantic_text = build_live_card_semantic_text(card)
        if semantic_text:
            return semantic_text
        return str(card.get("canonical_text") or card.get("title") or "").strip()

    def _build_action_text(self, action: dict | None) -> str:
        if not isinstance(action, dict):
            return ""

        kind = str(action.get("kind") or "").strip()
        canonical_text = str(action.get("canonical_text") or "").strip()
        target = action.get("target") if isinstance(action.get("target"), dict) else {}
        target_name = str((target or {}).get("name") or "").strip()

        card = action.get("card")
        if isinstance(card, dict):
            card_text = self._build_live_card_text(card)
            if kind == "play_card":
                parts = ["play", card_text]
                if target_name:
                    parts.append(f"tgt {target_name}")
                return " | ".join(part for part in parts if part)
            if kind == "card_reward":
                selection = str(action.get("selection") or "").strip().lower()
                if "skip" in selection:
                    return canonical_text or "skip card reward"
                return " | ".join(part for part in ("pick", card_text) if part)
            if kind == "deck_upgrade":
                selection = str(action.get("selection") or "").strip().lower()
                if any(token in selection for token in ("confirm", "cancel", "close")):
                    return canonical_text
                preview = action.get("upgrade_preview")
                preview_text = self._build_live_card_text(preview) if isinstance(preview, dict) else ""
                parts = ["upgrade", card_text]
                if preview_text:
                    parts.append(f"to {preview_text}")
                return " | ".join(part for part in parts if part)
            if kind == "card_selection":
                selection = str(action.get("selection") or "").strip().lower()
                if any(token in selection for token in ("confirm", "cancel", "close", "skip")):
                    return canonical_text
                semantics = str(action.get("selection_semantics") or "").strip()
                prefix = f"select {semantics}".strip() if semantics else "select"
                return " | ".join(part for part in (prefix, card_text) if part)

        potion = action.get("potion")
        if isinstance(potion, dict):
            potion_text = build_live_potion_semantic_text(potion)
            if kind == "use_potion":
                parts = ["use potion", potion_text]
                if target_name:
                    parts.append(f"tgt {target_name}")
                return " | ".join(part for part in parts if part)
            if kind == "discard_potion":
                return " | ".join(part for part in ("discard potion", potion_text) if part)

        reward = action.get("reward")
        if isinstance(reward, dict):
            reward_relic = reward.get("relic")
            if isinstance(reward_relic, dict):
                relic_text = build_live_relic_semantic_text(reward_relic)
                if relic_text:
                    prefix = "take relic" if kind in ("reward", "treasure_relic") else (kind or "relic")
                    return " | ".join(part for part in (prefix, relic_text) if part)

            reward_potion = reward.get("potion")
            if isinstance(reward_potion, dict):
                potion_text = build_live_potion_semantic_text(reward_potion)
                if potion_text:
                    prefix = "take potion" if kind == "reward" else (kind or "potion")
                    return " | ".join(part for part in (prefix, potion_text) if part)

        if kind == "shop":
            item = action.get("item")
            if isinstance(item, dict):
                shop_action = str(action.get("shop_action") or "").strip().lower()
                item_cost = item.get("cost")
                cost_text = f"cost {_float(item_cost):.0f}" if item_cost is not None else ""
                item_card = item.get("card")
                if isinstance(item_card, dict):
                    item_text = self._build_live_card_text(item_card)
                    if item_text:
                        prefix = "leave shop" if any(token in shop_action for token in ("leave", "back")) else "buy"
                        return " | ".join(part for part in (prefix, item_text, cost_text) if part)
                item_relic = item.get("relic")
                if isinstance(item_relic, dict):
                    item_text = build_live_relic_semantic_text(item_relic)
                    if item_text:
                        prefix = "leave shop" if any(token in shop_action for token in ("leave", "back")) else "buy relic"
                        return " | ".join(part for part in (prefix, item_text, cost_text) if part)
                item_potion = item.get("potion")
                if isinstance(item_potion, dict):
                    item_text = build_live_potion_semantic_text(item_potion)
                    if item_text:
                        prefix = "leave shop" if any(token in shop_action for token in ("leave", "back")) else "buy potion"
                        return " | ".join(part for part in (prefix, item_text, cost_text) if part)

        relic = action.get("relic")
        if isinstance(relic, dict):
            relic_text = build_live_relic_semantic_text(relic)
            if kind == "treasure_relic":
                return " | ".join(part for part in ("take relic", relic_text) if part)
            if relic_text:
                return " | ".join(part for part in (kind or "relic", relic_text) if part)

        return canonical_text

    def _enc_action_numeric(self, row: np.ndarray, action: dict) -> None:
        kind = action.get("kind", "")
        row[0] = _KIND_TO_ORD.get(kind, 0) / _NUM_KINDS

        card = action.get("card")
        if isinstance(card, dict):
            row[1] = 1.0
            row[2] = min(_float(card.get("cost")) / 5.0, 1.0)
            row[3] = min(_float(card.get("star")) / 5.0, 1.0) if card.get("star") is not None else 0.0
            card_type = (card.get("type") or "").capitalize()
            row[4] = 1.0 if card_type == "Attack" else 0.0
            row[5] = 1.0 if card_type == "Skill" else 0.0
            row[6] = 1.0 if card_type == "Power" else 0.0
            row[20] = 1.0 if card_type == "Status" else 0.0
            row[21] = 1.0 if card_type == "Curse" else 0.0

        target = action.get("target")
        row[7] = 1.0 if isinstance(target, dict) and target.get("name") else 0.0
        row[8] = 1.0 if isinstance(target, dict) and target.get("side") == "Player" else 0.0
        row[9] = 1.0 if action.get("action_id") == "end_turn" else 0.0
        row[10] = 1.0 if kind == "proceed" and not action.get("skip") else 0.0
        row[11] = 1.0 if action.get("skip") or "skip" in (action.get("action_id") or "") else 0.0

        item = action.get("item")
        if isinstance(item, dict):
            row[12] = min(_float(item.get("cost")) / 500.0, 1.0)

        reward = action.get("reward")
        if isinstance(reward, dict):
            reward_type = reward.get("type", "")
            row[13] = 1.0 if reward_type == "gold" else 0.0
            row[14] = 1.0 if reward_type == "card" else 0.0

        point_type = self._normalize_route_point_type(action.get("point_type_norm") or action.get("point_type", ""))
        row[15] = _PT_TO_ORD.get(point_type, 0) / _NUM_PT

        coord = action.get("coord")
        if isinstance(coord, dict):
            row[16] = min(_float(coord.get("row")) / 15.0, 1.0)
            row[17] = min(_float(coord.get("col")) / 7.0, 1.0)

        row[18] = 1.0 if isinstance(action.get("upgrade_preview"), dict) else 0.0

        option_index = action.get("index")
        if option_index is None:
            option_index = action.get("hand_index")
        if option_index is None:
            option_index = action.get("slot_index")
        row[19] = min(_float(option_index) / 20.0, 1.0) if option_index is not None else 0.0

        source = None
        if isinstance(card, dict):
            source = card
        else:
            potion = action.get("potion")
            if isinstance(potion, dict):
                source = potion

        if isinstance(source, dict):
            row[22] = min(_metric(source, "damage") / 50.0, 1.0)
            row[23] = min(_metric(source, "block") / 50.0, 1.0)
            row[24] = min(_metric(source, "draw") / 5.0, 1.0)
            row[25] = min(_metric(source, "weak") / 5.0, 1.0)
            row[26] = min(_metric(source, "vulnerable") / 5.0, 1.0)
            row[27] = min(_metric(source, "heal") / 30.0, 1.0)
            row[28] = min(_metric(source, "hp_loss") / 30.0, 1.0)
            row[29] = min(_metric(source, "summon") / 5.0, 1.0)
            row[30] = 1.0 if (
                _metric(source, "damage") > 0
                or _metric(source, "block") > 0
                or _metric(source, "draw") > 0
                or _metric(source, "weak") > 0
                or _metric(source, "vulnerable") > 0
                or _metric(source, "heal") > 0
                or _metric(source, "strength") > 0
                or _metric(source, "dexterity") > 0
                or _metric(source, "summon") > 0
            ) else 0.0
            row[31] = 1.0 if isinstance(card, dict) and _float(card.get("cost")) == 0 else 0.0

        build_aux = action.get("build_aux")
        if isinstance(build_aux, dict):
            option_total = max(_float(build_aux.get("option_total")), 1.0)
            deck_after_size = max(_float(build_aux.get("deck_after_size")), 1.0)
            row[7] = _float(build_aux.get("is_starter_attack"))
            row[8] = _float(build_aux.get("is_starter_defend"))
            row[9] = _float(build_aux.get("is_curse_or_status"))
            row[10] = min(_float(build_aux.get("shop_remove_rate")), 1.0)
            row[11] = _float(build_aux.get("gap_improves"))
            row[12] = min(_float(build_aux.get("remove_any_rate")), 1.0)
            row[13] = min(_float(build_aux.get("reward_rate")), 1.0)
            row[14] = min(_float(build_aux.get("smith_rate")), 1.0)
            row[15] = min(_float(build_aux.get("transform_rate")), 1.0)
            row[16] = min(_float(build_aux.get("primary_rate")), 1.0)
            row[17] = _float(build_aux.get("balanced_after"))
            row[18] = 1.0 if _float(build_aux.get("remove_rank")) <= 1 else 0.0
            row[19] = min(_float(build_aux.get("rate_rank")) / option_total, 1.0)
            row[20] = _float(build_aux.get("is_largest_stack"))
            row[21] = min(_float(build_aux.get("count_share")), 1.0)
            row[22] = min(_float(build_aux.get("remove_rank")) / option_total, 1.0)
            row[23] = min(_float(build_aux.get("keep_rank")) / option_total, 1.0)
            row[24] = min(max(_float(build_aux.get("primary_score")), 0.0) / 2.0, 1.0)
            row[25] = min(_float(build_aux.get("keep_score")), 1.0)
            row[26] = min(_float(build_aux.get("count_before")) / 5.0, 1.0)
            row[27] = min(_float(build_aux.get("count_after")) / 5.0, 1.0)
            row[28] = min(_float(build_aux.get("junk_after")) / deck_after_size, 1.0)
            row[29] = min(_float(build_aux.get("starter_attack_after")) / 10.0, 1.0)
            row[30] = min(_float(build_aux.get("starter_defend_after")) / 10.0, 1.0)
            row[31] = min((_float(build_aux.get("starter_gap_delta")) + 5.0) / 10.0, 1.0)

    def _enc_route_action(
        self,
        summary_row: np.ndarray,
        node_rows: np.ndarray,
        node_mask: np.ndarray,
        action: dict,
    ) -> None:
        route_summary = action.get("route_summary")
        if not isinstance(route_summary, dict):
            return

        summary_row[0] = min(_float(route_summary.get("reachable_node_count")) / 30.0, 1.0)
        summary_row[1] = min(_float(route_summary.get("max_depth")) / 15.0, 1.0)
        summary_row[2] = min(_float(route_summary.get("direct_child_count")) / 4.0, 1.0)
        summary_row[3] = min(_float(route_summary.get("forced_path_steps_before_branch")) / 10.0, 1.0)
        summary_row[4] = min(_float(route_summary.get("count_monster")) / 10.0, 1.0)
        summary_row[5] = min(_float(route_summary.get("count_elite")) / 5.0, 1.0)
        summary_row[6] = min(_float(route_summary.get("count_boss")) / 2.0, 1.0)
        summary_row[7] = min(_float(route_summary.get("count_event")) / 10.0, 1.0)
        summary_row[8] = min(_float(route_summary.get("count_question_mark")) / 10.0, 1.0)
        summary_row[9] = min(_float(route_summary.get("count_rest_site")) / 5.0, 1.0)
        summary_row[10] = min(_float(route_summary.get("count_shop")) / 5.0, 1.0)
        summary_row[11] = min(_float(route_summary.get("count_treasure")) / 5.0, 1.0)
        summary_row[12] = self._norm_step(route_summary.get("next_elite_steps"))
        summary_row[13] = self._norm_step(route_summary.get("next_rest_steps"))
        summary_row[14] = self._norm_step(route_summary.get("next_shop_steps"))
        summary_row[15] = self._norm_step(route_summary.get("next_event_steps"))
        summary_row[16] = self._norm_step(route_summary.get("next_question_mark_steps"))
        summary_row[17] = self._norm_step(route_summary.get("next_treasure_steps"))
        summary_row[18] = _bool(route_summary.get("can_reach_rest_site_before_elite"))
        summary_row[19] = _bool(route_summary.get("can_reach_elite_then_rest_site"))

        nodes = action.get("route_nodes") or []
        for index, node in enumerate(nodes[:MAX_ROUTE_NODES]):
            if not isinstance(node, dict):
                continue
            node_mask[index] = 1.0
            self._enc_route_node(node_rows[index], node)

    def _enc_route_node(self, row: np.ndarray, node: dict) -> None:
        point_type = self._normalize_route_point_type(node.get("point_type"))
        type_index = _ROUTE_PT_TO_IDX.get(point_type)
        if type_index is not None:
            row[type_index] = 1.0

        base = _NUM_ROUTE_PT
        row[base] = min(_float(node.get("depth")) / 15.0, 1.0)
        coord = node.get("coord")
        if isinstance(coord, dict):
            row[base + 1] = min(_float(coord.get("row")) / 15.0, 1.0)
            row[base + 2] = min(_float(coord.get("col")) / 7.0, 1.0)
        row[base + 3] = min(_float(node.get("child_count")) / 4.0, 1.0)
        row[base + 4] = _bool(node.get("is_leaf"))

    def _resolve_domain(self, obs: dict) -> str:
        decision_domain = obs.get("decision_domain")
        if isinstance(decision_domain, str) and decision_domain in DOMAIN_TO_IDX:
            return decision_domain

        phase = obs.get("phase", "")
        if phase == "combat":
            return "combat"
        if phase == "map":
            return "route"
        if phase == "card_selection":
            combat = obs.get("combat")
            return "combat" if combat else "build"
        if phase == "settling":
            combat = obs.get("combat")
            return "combat" if combat else "build"
        return "build"

    @staticmethod
    def _norm_step(value) -> float:
        if value is None:
            return 0.0
        return min(_float(value) / 10.0, 1.0)

    @staticmethod
    def _normalize_route_point_type(point_type) -> str:
        if point_type in ("Merchant", "Shop"):
            return "Shop"
        if point_type in ("Rest", "RestSite"):
            return "RestSite"
        if point_type in ("Unknown", "QuestionMark"):
            return "QuestionMark"
        if point_type in _ROUTE_PT_TO_IDX:
            return point_type
        return "Monster"

    @staticmethod
    def _parse_act(act_id):
        if not act_id or not isinstance(act_id, str):
            return 0.0
        for char in reversed(act_id):
            if char.isdigit():
                try:
                    return float(char)
                except Exception:
                    pass
        return 0.0
