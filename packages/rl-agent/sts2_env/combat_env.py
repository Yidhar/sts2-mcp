"""Combat sandbox Gymnasium wrapper for STS2.

This mirrors :mod:`env_v2` but resets through ``/env/combat_reset``. Training
defaults to compact ``info`` payloads so rollout hot paths avoid carrying large
raw observation trees unless explicitly requested by debug/eval callers.
"""

from __future__ import annotations

import time
from typing import Any

import gymnasium as gym
import numpy as np
from gymnasium import spaces

from combat_snapshot_dataset import snapshot_row_to_reset_kwargs
from .action_compact import compact_legal_actions
from .aux_targets import build_aux_targets
from .bridge_client import BridgeClient, BridgeError
from .observation_common import DenseObservationEncoder, MAX_ACTIONS
from .observation_v3 import WorldTokenObservationEncoder
from .combat_memory import CombatMemoryTracker
from .run_memory import RunMemoryTracker

from .reward_constants import (
    COMBAT_SANDBOX_WASTE_BASE as END_TURN_WASTE_BASE_PENALTY,
    COMBAT_SANDBOX_WASTE_ENERGY as END_TURN_WASTE_ENERGY_PENALTY,
    COMBAT_SANDBOX_WASTE_ZERO_COST as END_TURN_WASTE_ZERO_COST_BONUS_PENALTY,
    COMBAT_SANDBOX_WASTE_EXTRA_ACTION as END_TURN_WASTE_EXTRA_ACTION_PENALTY,
    ENEMY_HP_DELTA_REWARD_MAX_ABS,
    ENEMY_HP_DELTA_REWARD_SCALE,
    ENEMY_HP_SENTINEL_THRESHOLD,
    INVALID_ACTION_REWARD,
    PLAYER_HP_LOSS_REWARD_SCALE,
    SENTINEL_COMBAT_LOSS_PENALTY_BASE,
    SENTINEL_COMBAT_LOSS_PENALTY_SCALE,
    SENTINEL_COMBAT_WIN_BONUS_BASE,
    SENTINEL_COMBAT_WIN_BONUS_SCALE,
    SENTINEL_DEATH_DAMAGE_POWER_KEYWORDS,
)

INVALID_ACTION_REASON = "invalid_action_index"
BLOCKED_ACTION_KINDS = {"discard_potion"}


def _float(value: Any, default: float = 0.0) -> float:
    if value is None:
        return default
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


class CombatSandboxEnv(gym.Env):
    """Gymnasium Env for combat-only RL training via the STS2 bridge.

    Uses POST /env/combat_reset to enter a specific encounter, then
    POST /env/step for each action.  Episode ends when combat finishes.
    """

    metadata = {"render_modes": ["human"]}

    def __init__(
        self,
        session_file: str | None = None,
        character: str | None = None,
        encounter_id: str | None = None,
        encounter_pool: list[str] | None = None,
        seed: int | None = None,
        current_hp: int | None = None,
        max_hp: int | None = None,
        max_energy: int | None = None,
        deck: list[str] | None = None,
        deck_entries: list[dict[str, Any]] | None = None,
        relics: list[str] | None = None,
        potions: list[str] | None = None,
        gold: int | None = None,
        snapshot_pool = None,
        sandbox_supports_potions: bool = True,
        reset_timeout_ms: int = 15000,
        step_timeout_ms: int = 20000,
        render_mode: str | None = None,
        obs_encoder: DenseObservationEncoder | None = None,
        include_debug_info: bool = False,
        bridge: "BridgeClient | None" = None,
    ) -> None:
        super().__init__()

        # Allow external injection of a bridge (e.g., a HeadlessSimBridgeClient
        # that drives frankqwang/sts2-ai's C# headless sim in place of a real
        # game HTTP bridge). If not provided, fall back to the real bridge.
        self.bridge = bridge if bridge is not None else BridgeClient(session_path=session_file)
        self.obs_encoder = obs_encoder or WorldTokenObservationEncoder(use_text=False)
        self.character = character
        self.encounter_id = encounter_id
        self.encounter_pool = [eid for eid in (encounter_pool or []) if eid]
        self.seed = seed
        self.current_hp = current_hp
        self.max_hp = max_hp
        self.max_energy = max_energy
        self.deck = deck
        self.deck_entries = deck_entries
        self.relics = relics
        self.potions = potions
        self.gold = gold
        self.snapshot_pool = snapshot_pool
        self.sandbox_supports_potions = bool(sandbox_supports_potions)
        self.reset_timeout_ms = reset_timeout_ms
        self.step_timeout_ms = step_timeout_ms
        self.render_mode = render_mode
        self.include_debug_info = bool(include_debug_info)

        self.observation_space = self.obs_encoder.obs_space
        self.action_space = spaces.Discrete(MAX_ACTIONS)

        self._episode_id: str | None = None
        self._legal_actions: list[dict[str, Any]] = []
        self._last_obs_raw: dict[str, Any] | None = None
        self._current_encounter_id: str | None = encounter_id
        self._last_action_overflow: int = 0
        self._current_snapshot: dict[str, Any] | None = None
        self._last_reset_kwargs: dict[str, Any] = {}
        self._sentinel_combat_active: bool = False
        self._sentinel_combat_start_max_hp: float = 0.0
        self._run_memory = RunMemoryTracker(
            episode_mode="combat_sandbox",
            potion_mechanics_available=self.sandbox_supports_potions,
        )
        self._combat_memory = CombatMemoryTracker()

    # ------------------------------------------------------------------
    # Gymnasium API
    # ------------------------------------------------------------------

    def reset(self, seed: int | None = None, options: dict[str, Any] | None = None):
        super().reset(seed=seed)
        started = time.perf_counter()

        # Allow per-reset overrides via options dict
        opts = options or {}
        snapshot = opts.get("snapshot")
        if snapshot is None and self.snapshot_pool is not None:
            snapshot = self.snapshot_pool.sample(self.np_random)
        snapshot_kwargs = (
            snapshot_row_to_reset_kwargs(snapshot, include_potions=self.sandbox_supports_potions)
            if isinstance(snapshot, dict)
            else {}
        )

        encounter_id = opts.get("encounter_id", snapshot_kwargs.get("encounter_id"))
        if encounter_id is None:
            encounter_id = self._sample_encounter_id()
        reset_seed = opts.get("seed", self.seed)
        self._current_encounter_id = encounter_id
        self._current_snapshot = snapshot if isinstance(snapshot, dict) else None

        character = opts.get("character", snapshot_kwargs.get("character", self.character))
        current_hp = opts.get("current_hp", snapshot_kwargs.get("current_hp", self.current_hp))
        max_hp = opts.get("max_hp", snapshot_kwargs.get("max_hp", self.max_hp))
        max_energy = opts.get("max_energy", snapshot_kwargs.get("max_energy", self.max_energy))
        deck = opts.get("deck", snapshot_kwargs.get("deck", self.deck))
        deck_entries = opts.get("deck_entries", snapshot_kwargs.get("deck_entries", self.deck_entries))
        relics = opts.get("relics", snapshot_kwargs.get("relics", self.relics))
        gold = opts.get("gold", snapshot_kwargs.get("gold", self.gold))
        if self.sandbox_supports_potions:
            if "potions" in opts:
                potions = opts.get("potions")
            elif snapshot_kwargs.get("potions") is not None:
                potions = snapshot_kwargs.get("potions")
            else:
                potions = self.potions
        else:
            potions = None

        self._last_reset_kwargs = {
            "character": character,
            "encounter_id": encounter_id,
            "seed": reset_seed,
            "current_hp": current_hp,
            "max_hp": max_hp,
            "max_energy": max_energy,
            "deck": list(deck) if isinstance(deck, list) else deck,
            "deck_entries": [dict(entry) for entry in deck_entries] if isinstance(deck_entries, list) else deck_entries,
            "relics": list(relics) if isinstance(relics, list) else relics,
            "potions": list(potions) if isinstance(potions, list) else potions,
            "gold": gold,
        }

        try:
            bridge_started = time.perf_counter()
            result = self.bridge.combat_reset(
                character=character,
                encounter_id=encounter_id,
                seed=reset_seed,
                current_hp=current_hp,
                max_hp=max_hp,
                max_energy=max_energy,
                deck=deck,
                deck_entries=deck_entries,
                relics=relics,
                potions=potions,
                gold=gold,
                timeout_ms=self.reset_timeout_ms,
            )
            bridge_elapsed_ms = (time.perf_counter() - bridge_started) * 1000.0
        except BridgeError as exc:
            salvaged = self._try_salvage_card_selection_reset(exc)
            if salvaged is None:
                raise
            result = salvaged
            bridge_elapsed_ms = (time.perf_counter() - bridge_started) * 1000.0

        self._episode_id = result["episode_id"]
        self._update_live_state(result)
        sentinel_enemy = self._find_sentinel_enemy(self._last_obs_raw)
        self._sentinel_combat_active = sentinel_enemy is not None
        if self._sentinel_combat_active:
            _, start_max = self._player_hp_and_max(self._last_obs_raw)
            self._sentinel_combat_start_max_hp = start_max
        else:
            self._sentinel_combat_start_max_hp = 0.0
        run_memory_started = time.perf_counter()
        self._run_memory.reset(
            self._last_obs_raw,
            self._legal_actions,
            episode_mode="combat_sandbox",
            potion_mechanics_available=self.sandbox_supports_potions,
        )
        self._combat_memory.reset(self._last_obs_raw)
        run_memory_elapsed_ms = (time.perf_counter() - run_memory_started) * 1000.0

        planner_context = self._planner_context()
        obs_encode_started = time.perf_counter()
        obs = self.obs_encoder.encode(self._last_obs_raw, self._legal_actions, planner_context)
        obs_encode_elapsed_ms = (time.perf_counter() - obs_encode_started) * 1000.0
        info_started = time.perf_counter()
        info = self._build_info(
            result.get("info", {}),
            extra={
                "python_timing_ms": self._python_timing(
                    bridge_roundtrip=bridge_elapsed_ms,
                    run_memory_update=run_memory_elapsed_ms,
                    obs_encode=obs_encode_elapsed_ms,
                    aux_targets=0.0,
                    info_build=0.0,
                    total=(time.perf_counter() - started) * 1000.0,
                )
            },
        )
        info["python_timing_ms"]["info_build"] = (time.perf_counter() - info_started) * 1000.0
        info["python_timing_ms"]["total"] = (time.perf_counter() - started) * 1000.0
        return obs, info

    def step(self, action: int):
        if not self._legal_actions:
            return self._make_terminal()
        started = time.perf_counter()

        normalized_action = self._normalize_action(action)
        if normalized_action is None or normalized_action >= len(self._legal_actions):
            return self._make_invalid_action_response(action)

        legal_action = self._legal_actions[normalized_action]
        legal_actions_before = list(self._legal_actions)
        prev_obs = self._last_obs_raw or {}
        prev_planner_context = self._planner_context()
        end_turn_penalty = self._end_turn_waste_penalty(prev_obs, self._legal_actions, legal_action)

        bridge_started = time.perf_counter()
        try:
            result = self.bridge.step(
                episode_id=self._episode_id,
                action_id=legal_action.get("action_id"),
                timeout_ms=self.step_timeout_ms,
            )
        except BridgeError as e:
            # Bridge rejected the action (e.g. TOCTOU race, phase mismatch).
            # We cannot safely continue with the stale _last_obs_raw /
            # _legal_actions — the next step would sample from a mask that no
            # longer matches live bridge state, which tends to loop on invalid
            # actions. Truncate so the collector restarts the episode cleanly;
            # this matches the truncated=True behavior of _make_invalid_action_response.
            obs = self.obs_encoder.encode(
                self._last_obs_raw, self._legal_actions, prev_planner_context
            )
            info = self._build_info(
                {"truncation_reason": "bridge_error"},
                extra={"action_error": str(e)},
            )
            return obs, float(INVALID_ACTION_REWARD), False, True, info
        bridge_elapsed_ms = (time.perf_counter() - bridge_started) * 1000.0

        self._update_live_state(result)
        run_memory_started = time.perf_counter()
        self._run_memory.update_transition(prev_obs, legal_action, self._last_obs_raw, legal_actions=self._legal_actions)
        self._combat_memory.update(prev_obs, legal_action, self._last_obs_raw)
        run_memory_elapsed_ms = (time.perf_counter() - run_memory_started) * 1000.0

        next_planner_context = self._planner_context()
        obs_encode_started = time.perf_counter()
        obs = self.obs_encoder.encode(self._last_obs_raw, self._legal_actions, next_planner_context)
        obs_encode_elapsed_ms = (time.perf_counter() - obs_encode_started) * 1000.0
        reward = float(result.get("reward", 0.0))
        reward += self._enemy_hp_delta_reward(prev_obs, self._last_obs_raw)
        reward += self._player_hp_delta_reward(prev_obs, self._last_obs_raw)
        reward += end_turn_penalty
        terminated = bool(result.get("done", False))
        truncated = bool(result.get("truncated", False))
        sentinel_terminal = self._sentinel_terminal_reward(
            prev_obs, self._last_obs_raw, terminated, truncated
        )
        reward += sentinel_terminal
        if terminated or truncated:
            self._sentinel_combat_active = False
        aux_started = time.perf_counter()
        aux_targets = build_aux_targets(
            prev_obs,
            legal_action,
            self._last_obs_raw,
            prev_planner_context=prev_planner_context,
            next_planner_context=next_planner_context,
            terminated=terminated,
            truncated=truncated,
            legal_actions_before=legal_actions_before,
        )
        aux_elapsed_ms = (time.perf_counter() - aux_started) * 1000.0
        info_started = time.perf_counter()
        info = self._build_info(
            result.get("info", {}),
            extra={
                "aux_targets": aux_targets,
                "python_timing_ms": self._python_timing(
                    bridge_roundtrip=bridge_elapsed_ms,
                    run_memory_update=run_memory_elapsed_ms,
                    obs_encode=obs_encode_elapsed_ms,
                    aux_targets=aux_elapsed_ms,
                    info_build=0.0,
                    total=(time.perf_counter() - started) * 1000.0,
                ),
            },
        )
        info["python_timing_ms"]["info_build"] = (time.perf_counter() - info_started) * 1000.0
        info["python_timing_ms"]["total"] = (time.perf_counter() - started) * 1000.0

        if self.render_mode == "human":
            self.render()

        return obs, reward, terminated, truncated, info

    def action_masks(self) -> np.ndarray:
        mask = np.zeros(MAX_ACTIONS, dtype=bool)
        for i, action in enumerate(self._legal_actions[:MAX_ACTIONS]):
            if isinstance(action, dict):
                mask[i] = True
        return mask

    def _make_terminal(self):
        obs = self.obs_encoder.encode(self._last_obs_raw or {}, [], self._planner_context())
        info = self._build_info(
            {},
            extra={
                "python_timing_ms": self._python_timing(
                    bridge_roundtrip=0.0,
                    run_memory_update=0.0,
                    obs_encode=0.0,
                    aux_targets=0.0,
                    info_build=0.0,
                    total=0.0,
                )
            },
        )
        return obs, 0.0, True, False, info

    def render(self) -> None:
        if self._last_obs_raw is None:
            return
        phase = self._last_obs_raw.get("phase", "?")
        player = self._last_obs_raw.get("player", {})
        hp = player.get("hp", "?")
        max_hp = player.get("max_hp", "?")
        combat = self._last_obs_raw.get("combat", {})
        rnd = combat.get("round", "?")
        print(
            f"[CombatSandbox] Phase: {phase} | HP: {hp}/{max_hp} "
            f"| Round: {rnd} | Actions: {len(self._legal_actions)}"
        )

    def close(self) -> None:
        pass

    def get_compact_legal_actions(self) -> list[dict[str, Any]]:
        return compact_legal_actions(self._legal_actions)

    @property
    def raw_obs(self) -> dict[str, Any] | None:
        return self._last_obs_raw

    @property
    def legal_actions(self) -> list[dict[str, Any]]:
        return self._legal_actions

    @property
    def last_reset_kwargs(self) -> dict[str, Any]:
        return dict(self._last_reset_kwargs)

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _sample_encounter_id(self) -> str | None:
        if self.encounter_pool:
            idx = int(self.np_random.integers(len(self.encounter_pool)))
            return self.encounter_pool[idx]
        return self.encounter_id

    def _try_salvage_card_selection_reset(self, exc: BridgeError) -> dict[str, Any] | None:
        body = exc.response_body if isinstance(exc.response_body, dict) else {}
        if exc.status_code != 409:
            return None
        if str(body.get("error") or "").strip() != "combat_sandbox_not_in_combat":
            return None
        details = body.get("details") if isinstance(body.get("details"), dict) else {}
        screen = str(details.get("screen") or "").strip().upper()
        phase = str(details.get("phase") or "").strip().lower()
        actionable = bool(details.get("actionable"))
        combat_in_progress = bool(details.get("combat_in_progress"))
        if screen not in {"COMBAT", "CARD_SELECTION"} or not combat_in_progress:
            return None
        if phase not in {"card_selection", "combat", "settling"}:
            return None
        if not actionable and phase != "settling":
            return None

        rebound = self.bridge.reset(
            rebind_active_run=True,
            timeout_ms=self.reset_timeout_ms,
        )
        info = rebound.get("info")
        if not isinstance(info, dict):
            info = {}
            rebound["info"] = info
        info["combat_reset_salvaged"] = True
        info["combat_reset_salvage_phase"] = phase
        info["combat_reset_salvage_screen"] = screen
        info["combat_reset_salvage_actionable"] = actionable
        return rebound

    def _normalize_action(self, action: Any) -> int | None:
        try:
            normalized = int(action)
        except (TypeError, ValueError):
            return None
        if normalized < 0:
            return None
        return normalized

    def _update_live_state(self, result: dict[str, Any]) -> None:
        legal_actions = result.get("legal_actions", [])
        if isinstance(legal_actions, list):
            self._legal_actions = [
                action for action in legal_actions
                if not (
                    isinstance(action, dict) and
                    str(action.get("kind") or "").strip() in BLOCKED_ACTION_KINDS
                )
            ]
        else:
            self._legal_actions = []
        obs = result.get("obs", {})
        self._last_obs_raw = obs if isinstance(obs, dict) else {}
        self._last_action_overflow = max(len(self._legal_actions) - MAX_ACTIONS, 0)

    def _combat_enemy_total_hp(self, obs: dict[str, Any] | None) -> float:
        if not isinstance(obs, dict):
            return 0.0
        combat = obs.get("combat")
        if not isinstance(combat, dict):
            return 0.0
        enemies = combat.get("enemies")
        if not isinstance(enemies, list):
            return 0.0

        total = 0.0
        for enemy in enemies:
            if not isinstance(enemy, dict):
                continue
            hp = float(enemy.get("hp", enemy.get("current_hp")) or 0.0)
            if hp > ENEMY_HP_SENTINEL_THRESHOLD:
                continue
            total += hp
        return total

    def _enemy_hp_delta_reward(self, before_obs: dict[str, Any] | None, after_obs: dict[str, Any] | None) -> float:
        before_total = self._combat_enemy_total_hp(before_obs)
        after_total = self._combat_enemy_total_hp(after_obs)
        if before_total <= 0.0 and after_total <= 0.0:
            return 0.0
        raw = (before_total - after_total) * ENEMY_HP_DELTA_REWARD_SCALE
        if raw > ENEMY_HP_DELTA_REWARD_MAX_ABS:
            return ENEMY_HP_DELTA_REWARD_MAX_ABS
        if raw < -ENEMY_HP_DELTA_REWARD_MAX_ABS:
            return -ENEMY_HP_DELTA_REWARD_MAX_ABS
        return raw

    def _player_hp_delta_reward(self, before_obs: dict[str, Any] | None, after_obs: dict[str, Any] | None) -> float:
        before_player = before_obs.get("player") if isinstance(before_obs, dict) else {}
        after_player = after_obs.get("player") if isinstance(after_obs, dict) else {}
        before_hp = _float((before_player or {}).get("hp"))
        after_hp = _float((after_player or {}).get("hp"))
        if before_hp <= 0.0 and after_hp <= 0.0:
            return 0.0
        # Symmetric with env_v2.py: positive for HP gain (rest, heal, etc.),
        # negative for HP loss. Asymmetric "loss-only" shaping left rest-site
        # and heal-potion decisions without any immediate signal.
        return (after_hp - before_hp) * PLAYER_HP_LOSS_REWARD_SCALE

    @staticmethod
    def _find_sentinel_enemy(obs: dict[str, Any] | None) -> dict[str, Any] | None:
        if not isinstance(obs, dict):
            return None
        combat = obs.get("combat")
        if not isinstance(combat, dict):
            return None
        enemies = combat.get("enemies")
        if not isinstance(enemies, list):
            return None
        for enemy in enemies:
            if not isinstance(enemy, dict):
                continue
            hp = _float(enemy.get("hp", enemy.get("current_hp")))
            if hp > ENEMY_HP_SENTINEL_THRESHOLD:
                return enemy
        return None

    @staticmethod
    def _player_hp_and_max(obs: dict[str, Any] | None) -> tuple[float, float]:
        if not isinstance(obs, dict):
            return 0.0, 0.0
        player = obs.get("player") or {}
        return _float(player.get("hp")), _float(player.get("max_hp"))

    @staticmethod
    def _estimate_sentinel_death_damage(enemy: dict[str, Any] | None) -> float:
        """Upper-bound estimate of the on-death damage a sentinel enemy will deal.

        Takes the max of any matching on-death-flavored buff stack count and
        the currently announced intent damage, so the overshoot calculation is
        conservative (larger penalty if either signal is high).
        """
        if not isinstance(enemy, dict):
            return 0.0
        best = 0.0
        powers = enemy.get("powers")
        if isinstance(powers, list):
            for power in powers:
                if not isinstance(power, dict):
                    continue
                title = str(power.get("title") or "").lower()
                if not any(kw in title for kw in SENTINEL_DEATH_DAMAGE_POWER_KEYWORDS):
                    continue
                amount = _float(power.get("amount"))
                if amount > best:
                    best = amount
        intent = enemy.get("intent")
        if isinstance(intent, dict):
            intent_damage = _float(intent.get("total_damage"))
            if intent_damage > best:
                best = intent_damage
        return best

    def _sentinel_terminal_reward(
        self,
        prev_obs: dict[str, Any] | None,
        after_obs: dict[str, Any] | None,
        terminated: bool,
        truncated: bool,
    ) -> float:
        if not self._sentinel_combat_active or not (terminated or truncated):
            return 0.0

        after_hp, after_max = self._player_hp_and_max(after_obs)
        ref_max = self._sentinel_combat_start_max_hp or after_max
        if ref_max <= 0.0:
            ref_max = 1.0

        victory = terminated and (not truncated) and after_hp > 0.0
        if victory:
            hp_fraction = max(0.0, min(after_hp / ref_max, 1.0))
            return SENTINEL_COMBAT_WIN_BONUS_BASE + SENTINEL_COMBAT_WIN_BONUS_SCALE * hp_fraction

        # Loss branch: scale penalty by how much the expected death damage
        # overshot the player's (block + hp) buffer right before the terminal step.
        sentinel_before = self._find_sentinel_enemy(prev_obs)
        expected_death_damage = self._estimate_sentinel_death_damage(sentinel_before)
        prev_player = prev_obs.get("player") if isinstance(prev_obs, dict) else None
        prev_block = _float((prev_player or {}).get("block"))
        prev_hp = _float((prev_player or {}).get("hp"))
        overshoot = max(0.0, expected_death_damage - (prev_block + prev_hp))
        overshoot_fraction = max(0.0, min(overshoot / ref_max, 1.0))
        return -(
            SENTINEL_COMBAT_LOSS_PENALTY_BASE
            + SENTINEL_COMBAT_LOSS_PENALTY_SCALE * overshoot_fraction
        )

    @staticmethod
    def _source_preview_metric(source: dict[str, Any] | None, key: str) -> float:
        if not isinstance(source, dict):
            return 0.0
        effect_preview = source.get("effect_preview")
        if isinstance(effect_preview, dict) and effect_preview.get(key) is not None:
            try:
                return float(effect_preview.get(key) or 0.0)
            except (TypeError, ValueError):
                return 0.0
        try:
            return float(source.get(key) or 0.0)
        except (TypeError, ValueError):
            return 0.0

    def _is_positive_progress_action(self, action: dict[str, Any]) -> bool:
        kind = str(action.get("kind") or "").strip()
        if kind not in ("play_card", "use_potion"):
            return False

        source = action.get("card") if kind == "play_card" else action.get("potion")
        if not isinstance(source, dict):
            return False

        if kind == "play_card" and str(source.get("type") or "").strip().lower() == "power":
            return True

        for key in ("damage", "block", "draw", "weak", "vulnerable", "heal", "strength", "dexterity", "summon"):
            if self._source_preview_metric(source, key) > 0.0:
                return True
        return False

    def _end_turn_waste_penalty(
        self,
        obs: dict[str, Any] | None,
        legal_actions: list[dict[str, Any]],
        chosen_action: dict[str, Any],
    ) -> float:
        if str(chosen_action.get("action_id") or "") != "end_turn":
            return 0.0

        combat = obs.get("combat") if isinstance(obs, dict) else None
        if not isinstance(combat, dict):
            return 0.0
        energy = float(combat.get("energy") or 0.0)
        if energy <= 0.0:
            return 0.0

        positive_actions = 0
        has_zero_cost_positive = False
        for action in legal_actions:
            if not isinstance(action, dict):
                continue
            if str(action.get("action_id") or "") == "end_turn":
                continue
            if not self._is_positive_progress_action(action):
                continue
            positive_actions += 1
            card = action.get("card")
            if isinstance(card, dict):
                try:
                    if float(card.get("cost") or 0.0) <= 0.0:
                        has_zero_cost_positive = True
                except (TypeError, ValueError):
                    pass

        if positive_actions <= 0:
            return 0.0

        penalty = END_TURN_WASTE_BASE_PENALTY
        penalty += END_TURN_WASTE_ENERGY_PENALTY * min(energy, 3.0)
        if has_zero_cost_positive:
            penalty += END_TURN_WASTE_ZERO_COST_BONUS_PENALTY
        penalty += END_TURN_WASTE_EXTRA_ACTION_PENALTY * min(max(positive_actions - 1, 0), 2)
        return float(penalty)

    def _decorate_bridge_info(self, bridge_info: Any) -> dict[str, Any]:
        info = dict(bridge_info) if isinstance(bridge_info, dict) else {}
        diagnostics = info.get("action_diagnostics")
        diagnostics = dict(diagnostics) if isinstance(diagnostics, dict) else {}
        diagnostics["legal_action_overflow"] = float(self._last_action_overflow)
        info["action_diagnostics"] = diagnostics
        return info

    def _transition_state(self) -> dict[str, Any]:
        obs = self._last_obs_raw if isinstance(self._last_obs_raw, dict) else {}
        player = obs.get("player") if isinstance(obs.get("player"), dict) else {}
        run = obs.get("run") if isinstance(obs.get("run"), dict) else {}
        combat = obs.get("combat") if isinstance(obs.get("combat"), dict) else {}

        potions = player.get("potions") if isinstance(player.get("potions"), list) else []
        relics = player.get("relics") if isinstance(player.get("relics"), list) else []
        enemies_out: list[dict[str, Any]] = []
        for enemy in combat.get("enemies") if isinstance(combat.get("enemies"), list) else []:
            if not isinstance(enemy, dict):
                continue
            intent = enemy.get("intent") if isinstance(enemy.get("intent"), dict) else {}
            enemies_out.append(
                {
                    "hp": _float(enemy.get("hp", enemy.get("current_hp"))),
                    "block": _float(enemy.get("block")),
                    "intent": {
                        "total_damage": _float(intent.get("total_damage")),
                        "damage_per_hit": _float(intent.get("damage_per_hit")),
                        "repeats": _float(intent.get("repeats")),
                    },
                }
            )

        return {
            "phase": obs.get("phase"),
            "player": {
                "hp": _float(player.get("hp")),
                "max_hp": _float(player.get("max_hp")),
                "gold": _float(player.get("gold")),
                "potions": list(potions),
                "relics": list(relics),
            },
            "run": {
                "floor": _float(run.get("floor")),
                "act_id": _float(run.get("act_id")),
                "room_type": run.get("room_type"),
            },
            "combat": {
                "block": _float(combat.get("block")),
                "energy": _float(combat.get("energy")),
                "round": _float(combat.get("round")),
                "enemies": enemies_out,
            } if combat else {},
        }

    def _build_info(self, bridge_info: Any, *, extra: dict[str, Any] | None = None) -> dict[str, Any]:
        current_snapshot = self._current_snapshot or {}
        planner_context = self._planner_context()
        info: dict[str, Any] = {
            "episode_id": self._episode_id,
            "action_mask": self.action_masks(),
            "legal_action_count": len(self._legal_actions),
            "legal_actions_compact": self.get_compact_legal_actions(),
            "action_overflow": self._last_action_overflow,
            "phase": (self._last_obs_raw or {}).get("phase", "unknown"),
            "episode_mode": "combat_sandbox",
            "potion_mechanics_available": self.sandbox_supports_potions,
            # Combat sandbox has no map — floor is always 0. But Monitor's
            # info_keywords=("max_floor_reached","current_floor") hard-reads
            # both keys at episode end, so they must exist or SB3 KeyErrors.
            "max_floor_reached": 0,
            "current_floor": 0,
            "encounter_id": self._current_encounter_id,
            "encounter_pool": self.encounter_pool,
            "snapshot_sample_id": current_snapshot.get("sample_id"),
            "snapshot_run_id": current_snapshot.get("run_id"),
            "snapshot_floor_number": current_snapshot.get("floor_number"),
            "snapshot_build_id": current_snapshot.get("build_id"),
            "planner_context": planner_context,
            "transition_state": self._transition_state(),
            "bridge_info": self._decorate_bridge_info(bridge_info),
        }
        if extra:
            info.update(extra)
        if self.include_debug_info:
            info["legal_actions"] = self._legal_actions
            info["raw_obs"] = self._last_obs_raw
            if current_snapshot:
                info["combat_snapshot"] = current_snapshot
        return info

    def _make_invalid_action_response(self, attempted_action: Any):
        obs = self.obs_encoder.encode(self._last_obs_raw or {}, self._legal_actions, self._planner_context())
        bridge_info = {
            "action_error": INVALID_ACTION_REASON,
            "truncation_reason": INVALID_ACTION_REASON,
            "action_diagnostics": {
                "invalid_action_selected": 1.0,
            },
        }
        info = self._build_info(
            bridge_info,
            extra={
                "invalid_action_selected": True,
                "invalid_action_index": attempted_action,
                "python_timing_ms": self._python_timing(
                    bridge_roundtrip=0.0,
                    run_memory_update=0.0,
                    obs_encode=0.0,
                    aux_targets=0.0,
                    info_build=0.0,
                    total=0.0,
                ),
            },
        )
        return obs, INVALID_ACTION_REWARD, False, True, info

    def _planner_context(self) -> dict[str, Any]:
        context = self._run_memory.build_context(self._last_obs_raw, self._legal_actions)
        context["combat_memory"] = self._combat_memory.snapshot()
        return context

    @staticmethod
    def _python_timing(
        *,
        bridge_roundtrip: float,
        run_memory_update: float,
        obs_encode: float,
        aux_targets: float,
        info_build: float,
        total: float,
    ) -> dict[str, float]:
        return {
            "bridge_roundtrip": float(bridge_roundtrip),
            "run_memory_update": float(run_memory_update),
            "obs_encode": float(obs_encode),
            "aux_targets": float(aux_targets),
            "info_build": float(info_build),
            "total": float(total),
        }
