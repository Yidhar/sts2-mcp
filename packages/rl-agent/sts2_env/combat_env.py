"""Combat sandbox Gymnasium wrapper for STS2.

Identical structure to SlayTheSpire2EnvV2 but reset() calls the bridge's
/env/combat_reset endpoint instead of /env/reset.  This enters a specific
combat encounter directly without going through the full main-menu flow.

step() uses the shared /env/step endpoint — the bridge detects combat_sandbox
mode and returns done=True when combat ends (skipping reward/map screens).

Same observation space and action space as the full-run env, so the same
network architecture can be used for both.
"""

import gymnasium as gym
from gymnasium import spaces
import numpy as np

from .bridge_client import BridgeClient, BridgeError
from .observation_v2 import DictObservationEncoder

MAX_ACTIONS = 50


class CombatSandboxEnv(gym.Env):
    """Gymnasium Env for combat-only RL training via the STS2 bridge.

    Uses POST /env/combat_reset to enter a specific encounter, then
    POST /env/step for each action.  Episode ends when combat finishes.
    """

    metadata = {"render_modes": ["human"]}

    def __init__(
        self,
        session_file=None,
        character=None,
        encounter_id=None,
        seed=None,
        current_hp=None,
        max_hp=None,
        max_energy=None,
        deck=None,
        relics=None,
        potions=None,
        gold=None,
        reset_timeout_ms=15000,
        step_timeout_ms=20000,
        render_mode=None,
        obs_encoder=None,
    ):
        super().__init__()

        self.bridge = BridgeClient(session_path=session_file)
        self.obs_encoder = obs_encoder or DictObservationEncoder(use_text=False)
        self.character = character
        self.encounter_id = encounter_id
        self.seed = seed
        self.current_hp = current_hp
        self.max_hp = max_hp
        self.max_energy = max_energy
        self.deck = deck
        self.relics = relics
        self.potions = potions
        self.gold = gold
        self.reset_timeout_ms = reset_timeout_ms
        self.step_timeout_ms = step_timeout_ms
        self.render_mode = render_mode

        self.observation_space = self.obs_encoder.obs_space
        self.action_space = spaces.Discrete(MAX_ACTIONS)

        self._episode_id = None
        self._legal_actions = []
        self._last_obs_raw = None

    # ------------------------------------------------------------------
    # Gymnasium API
    # ------------------------------------------------------------------

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        # Allow per-reset overrides via options dict
        opts = options or {}
        encounter_id = opts.get("encounter_id", self.encounter_id)
        reset_seed = opts.get("seed", self.seed)

        result = self.bridge.combat_reset(
            character=self.character,
            encounter_id=encounter_id,
            seed=reset_seed,
            current_hp=self.current_hp,
            max_hp=self.max_hp,
            max_energy=self.max_energy,
            deck=self.deck,
            relics=self.relics,
            potions=self.potions,
            gold=self.gold,
            timeout_ms=self.reset_timeout_ms,
        )

        self._episode_id = result["episode_id"]
        self._legal_actions = result.get("legal_actions", [])
        self._last_obs_raw = result.get("obs", {})

        obs = self.obs_encoder.encode(self._last_obs_raw, self._legal_actions)
        info = self._build_info(result.get("info", {}))
        return obs, info

    def step(self, action):
        if not self._legal_actions:
            return self._make_terminal()

        if action >= len(self._legal_actions):
            action = 0

        legal_action = self._legal_actions[action]

        result = self.bridge.step(
            episode_id=self._episode_id,
            action_id=legal_action.get("action_id"),
            timeout_ms=self.step_timeout_ms,
        )

        self._legal_actions = result.get("legal_actions", [])
        self._last_obs_raw = result.get("obs", {})

        obs = self.obs_encoder.encode(self._last_obs_raw, self._legal_actions)
        reward = float(result.get("reward", 0.0))
        terminated = bool(result.get("done", False))
        truncated = bool(result.get("truncated", False))
        info = self._build_info(result.get("info", {}))

        if self.render_mode == "human":
            self.render()

        return obs, reward, terminated, truncated, info

    def action_masks(self):
        mask = np.zeros(MAX_ACTIONS, dtype=bool)
        n = min(len(self._legal_actions), MAX_ACTIONS)
        mask[:n] = True
        return mask

    def _make_terminal(self):
        obs = self.obs_encoder.encode(self._last_obs_raw or {}, [])
        info = self._build_info({})
        return obs, 0.0, True, False, info

    def render(self):
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

    def close(self):
        pass

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _build_info(self, bridge_info):
        return {
            "episode_id": self._episode_id,
            "action_mask": self.action_masks(),
            "legal_actions": self._legal_actions,
            "raw_obs": self._last_obs_raw,
            "phase": (self._last_obs_raw or {}).get("phase", "unknown"),
            "episode_mode": "combat_sandbox",
            "encounter_id": self.encounter_id,
            "bridge_info": bridge_info,
        }
