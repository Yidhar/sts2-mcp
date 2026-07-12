"""Combat sandbox Gymnasium wrapper for STS2.

This mirrors :mod:`env_v2` but resets through ``/env/combat_reset``. Training
defaults to compact ``info`` payloads so rollout hot paths avoid carrying large
raw observation trees unless explicitly requested by debug/eval callers.
"""

from __future__ import annotations

import os
import time
from typing import Any, ClassVar

import gymnasium as gym
from gymnasium import spaces

from combat_snapshot_dataset import snapshot_row_to_reset_kwargs
from sts2_rl.contracts import EnvironmentBackend

from ._combat_action_quality import CombatActionQualityMixin
from ._combat_curriculum import CurriculumTracker, get_curriculum_tracker
from ._combat_env_values import BLOCKED_ACTION_KINDS, INVALID_ACTION_REASON, _float
from ._combat_frontier import CombatFrontierMixin
from ._combat_legacy_reward import LegacyCombatRewardMixin
from .action_compact import compact_legal_actions
from .aux_targets import build_aux_targets
from .boss_mechanics import build_boss_mechanics_context
from .bridge_client import BridgeClient, BridgeError
from .card_effect_profile import aggregate_card_effect_profile_semantics
from .combat_memory import CombatMemoryTracker
from .end_turn_quality import strict_end_turn_waste_context
from .environment_runtime import EnvironmentRuntimeMixin
from .observation_common import MAX_ACTIONS, DenseObservationEncoder
from .observation_v3 import WorldTokenObservationEncoder
from .potion_timing import compute_potion_timing
from .reward_constants import (
    BOSS_COMBAT_LOSS_DAMAGE_UNDO_PERCENT_SCALE,
    BOSS_COMBAT_LOSS_PENALTY_BASE,
    BOSS_COMBAT_LOSS_PENALTY_MISSING_HP_SCALE,
    BOSS_COMBAT_WIN_BONUS_BASE,
    BOSS_COMBAT_WIN_BONUS_HP_SCALE,
    BOSS_DAMAGE_MULTIPLIER,
    BOSS_ENEMY_HP_DELTA_PERCENT_SCALE,
    CEREMONIAL_ONE_CARD_END_TURN_PENALTY,
    CEREMONIAL_ONE_CARD_HIGH_IMPACT_BONUS,
    CEREMONIAL_ONE_CARD_LOW_IMPACT_PENALTY,
    CEREMONIAL_STUN_DAMAGE_MULTIPLIER,
    CEREMONIAL_STUN_WINDOW_ENTER_BONUS,
    CEREMONIAL_THRESHOLD_PROGRESS_BONUS,
    CURRICULUM_HP_WEIGHT_LERP_MAX_TIER,
    CURRICULUM_HP_WEIGHT_LERP_MIN_TIER,
    CURRICULUM_MIN_EPISODES_FOR_PHASE,
    CURRICULUM_PHASE_WIN_RATE_THRESHOLDS,
    CURRICULUM_WIN_RATE_WINDOW,
    ENEMY_HP_DELTA_REWARD_MAX_ABS,
    ENEMY_HP_DELTA_REWARD_SCALE,
    ENEMY_HP_SENTINEL_THRESHOLD,
    HP_PRESERVE_WIN_BONUS_TIER_SCALE,
    INVALID_ACTION_REWARD,
    KAISER_BACK_ATTACK_DEFENSE_BONUS,
    KAISER_BACK_ATTACK_END_TURN_PENALTY,
    KAISER_BACK_ATTACK_HI_THREAT_DMG,
    KAISER_BACK_ATTACK_HI_THREAT_EXTRA,
    KAISER_BACK_ATTACK_HP_LOSS_PENALTY_SCALE,
    KAISER_BACK_ATTACK_RISK_REDUCTION_BONUS,
    KAISER_FACING_CHANGE_BONUS,
    KAISER_FACING_CHANGE_BONUS_BASE,
    KAISER_FACING_INTENT_DMG_REF,
    KAISER_FACING_INTENT_DMG_SCALE_MAX,
    KAISER_NO_RESPONSE_PENALTY_SOFTEN,
    KAISER_PRESSURE_KILL_BONUS,
    KNOWLEDGE_DEMON_BAD_CURSE_PICK_PENALTY,
    KNOWLEDGE_DEMON_END_TURN_PENALTY,
    KNOWLEDGE_DEMON_GOOD_CURSE_PICK_BONUS,
    OUTCOME_TIER_SCALE,
    PLAYER_HP_LOSS_BOSS_NO_HEAL_SCALE,
    PLAYER_HP_LOSS_REWARD_SCALE,
    PLAYER_HP_LOSS_TIER_SCALE,
    POTION_HOARDING_MAX_PENALTY_ABS,
    POTION_HOARDING_PENALTY_PER_POTION,
    POTION_TIMING_QUALITY_SCALE,
    POTION_TIMING_WASTE_SCALE,
    POTION_USE_BOSS_BONUS,
    POTION_USE_ELITE_BONUS,
    POTION_USE_MONSTER_BONUS,
    POTION_USE_MONSTER_PENALTY,
    SELECTION_DESELECT_PENALTY,
    SELECTION_EARLY_CONFIRM_BONUS,
    SELECTION_LOOP_PENALTY,
    SELECTION_OVER_CAP_PENALTY,
    SELECTION_PICK_CAP,
    SELECTION_REENTRY_BUDGET,
    SELECTION_REENTRY_PENALTY,
    SENTINEL_COMBAT_LOSS_PENALTY_BASE,
    SENTINEL_COMBAT_LOSS_PENALTY_SCALE,
    SENTINEL_COMBAT_WIN_BONUS_BASE,
    SENTINEL_COMBAT_WIN_BONUS_SCALE,
    SENTINEL_DEATH_DAMAGE_POWER_KEYWORDS,
    TURN_EFFICIENCY_PENALTY_PER_END_TURN_TIER,
    WASTEFUL_END_TURN_TIER_MULTIPLIER,
)
from .reward_constants import (
    COMBAT_SANDBOX_WASTE_BASE as END_TURN_WASTE_BASE_PENALTY,
)
from .reward_constants import (
    COMBAT_SANDBOX_WASTE_ENERGY as END_TURN_WASTE_ENERGY_PENALTY,
)
from .reward_constants import (
    COMBAT_SANDBOX_WASTE_EXTRA_ACTION as END_TURN_WASTE_EXTRA_ACTION_PENALTY,
)
from .reward_constants import (
    COMBAT_SANDBOX_WASTE_ZERO_COST as END_TURN_WASTE_ZERO_COST_BONUS_PENALTY,
)
from .run_memory import RunMemoryTracker


class CombatSandboxEnv(
    CombatFrontierMixin,
    LegacyCombatRewardMixin,
    CombatActionQualityMixin,
    EnvironmentRuntimeMixin,
    gym.Env,
):
    """Gymnasium Env for combat-only RL training via the STS2 bridge.

    Uses POST /env/combat_reset to enter a specific encounter, then
    POST /env/step for each action.  Episode ends when combat finishes.
    """

    metadata: ClassVar[dict[str, list[str]]] = {"render_modes": ["human"]}

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
        bridge: BridgeClient | None = None,
        backend: EnvironmentBackend | None = None,
    ) -> None:
        super().__init__()

        # All mutations flow through the typed backend. ``bridge`` remains a
        # compatibility injection seam and is immediately wrapped when used.
        if backend is None:
            raw_bridge = bridge if bridge is not None else BridgeClient(session_path=session_file)
            self._initialize_environment_runtime(bridge=raw_bridge, backend_name="legacy_bridge")
        else:
            if bridge is not None:
                raise ValueError("provide backend or bridge, not both")
            self._initialize_environment_runtime(backend=backend)
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
        # Shared curriculum tracker �?per combat-reward-curriculum.md §13 the
        # HP-loss shaping weight is gated by per-encounter win rate.  All
        # CombatSandboxEnv instances in the process share one window.
        self._curriculum_tracker = get_curriculum_tracker()
        # Counts the total number of terminal events fed into the curriculum
        # tracker; used to throttle the full state dump (every N episodes).
        self._curriculum_episode_count: int = 0
        # Track episode-initial HP/max_hp so the terminal preserve bonus can
        # reward "win with HP left" against a stable baseline rather than the
        # fluctuating current max_hp mid-fight.
        self._episode_start_hp: float = 0.0
        self._episode_start_max_hp: float = 1.0
        # Cumulative wasteful end_turn counter �?used by the throttled stdout
        # breadcrumb in `_end_turn_waste_penalty` and surfaced into hourly
        # reports so we can confirm the §9.1 penalty is actually firing.
        self._wasteful_end_turn_count: int = 0
        # Kaiser positive-response counters (§12) �?track how often the
        # facing-change / pressure-kill bonuses actually fire so we can
        # confirm the new signals are reaching the policy.
        self._kaiser_facing_change_count: int = 0
        self._kaiser_pressure_kill_count: int = 0
        # Card-selection anti-loop state: tracks the most recently picked
        # card id and consecutive flip count. See §4 of
        # docs/kaiser-and-potion-fixes-todo.md and reward_constants.py.
        self._selection_last_picked_id: str = ""
        self._selection_flip_count: int = 0
        self._selection_pick_count: int = 0
        self._selection_loop_events: int = 0
        self._selection_early_confirm_events: int = 0
        # §4 v2: oscillation detection across multiple cards.
        self._selection_deselect_count: int = 0
        self._selection_over_cap_events: int = 0
        # §4 v3: per-episode selection-screen re-entry tracker (resets in reset()).
        self._selection_screen_entries: int = 0
        self._selection_screen_active: bool = False
        self._selection_reentry_events: int = 0
        # Phase 4b counters: how many potion uses incurred timing reward
        # adjustment (positive vs negative). Surfaced for hourly cron debug.
        self._potion_timing_quality_events: int = 0
        self._potion_timing_waste_events: int = 0
        # TASK-A4: track potions whose use_potion action returned ok so the
        # episode-final unused-on-death calculation can subtract them, even if
        # the bridge slot fails to clear before the death frame is captured.
        self._used_potion_count_this_combat: int = 0
        self._potion_transition_records: list[dict[str, Any]] = []
        # TASK-C2: short-poll knobs for transient only-end-turn frontiers.
        # Defaults are conservative (small budget, short interval) so combat
        # sandbox throughput cannot be tanked by waiting on the bridge.
        try:
            self._fast_step_max_wait_ms: int = int(os.environ.get("MUZERO_FAST_STEP_MAX_WAIT_MS", "100"))
        except (TypeError, ValueError):
            self._fast_step_max_wait_ms = 100
        try:
            self._fast_step_poll_interval_ms: int = int(os.environ.get("MUZERO_FAST_STEP_POLL_INTERVAL_MS", "10"))
        except (TypeError, ValueError):
            self._fast_step_poll_interval_ms = 10
        self._fast_step_disabled: bool = (
            os.environ.get("MUZERO_FAST_STEP", "1").strip() == "0"
        )
        self._fast_step_metrics_total = {
            "transient_only_end_turn_count": 0,
            "transient_resolved_count": 0,
            "transient_leaked_count": 0,
            "wait_timeout_count": 0,
            "frontier_pre_dispatch_attempt_count": 0,
            "frontier_pre_dispatch_resolved_count": 0,
            "frontier_pre_dispatch_timeout_count": 0,
            "frontier_pre_dispatch_rebind_attempt_count": 0,
            "frontier_pre_dispatch_rebind_success_count": 0,
            "frontier_pre_dispatch_blocked_count": 0,
            "frontier_pre_dispatch_high_confidence_blocked_count": 0,
            "stable_no_actions_count": 0,
            "post_step_frontier_attempt_count": 0,
            "post_step_frontier_resolved_count": 0,
            "post_step_frontier_timeout_count": 0,
            "post_step_frontier_leaked_count": 0,
            "post_step_frontier_rebind_attempt_count": 0,
            "post_step_frontier_rebind_success_count": 0,
            "post_step_frontier_stable_no_actions_count": 0,
            "post_step_frontier_suspicious_singleton_count": 0,
        }
        # P0-2: actionability snapshot from the previous bridge result, used
        # to detect ``transient_only_end_turn`` leaks — when the policy chose
        # ``end_turn`` after seeing a transient-only-end_turn frontier we
        # refund the wasteful-end-turn penalty and tag the sample.
        self._last_actionability: dict[str, Any] | None = None
        # Optional human-demo recorder. Disabled by default; enable with
        # STS2_HUMAN_DEMO_RECORD=1 when driving this env manually / via a
        # human policy wrapper.  It records raw obs + legal actions +
        # selected action ids in the schema consumed by muzero.demo_dataset.
        self._human_demo_recorder = None
        try:
            from .human_demo_recorder import HumanDemoRecorder

            self._human_demo_recorder = HumanDemoRecorder.from_env(default_source="human")
        except Exception as exc:
            if os.environ.get("STS2_HUMAN_DEMO_RECORD", "").strip():
                print(f"[combat_env] human demo recorder disabled: {exc}", flush=True)

    # ------------------------------------------------------------------
    # Gymnasium API
    # ------------------------------------------------------------------

    def reset(self, seed: int | None = None, options: dict[str, Any] | None = None):
        super().reset(seed=seed)
        started = time.perf_counter()

        # §4 v3: reset per-episode selection-screen re-entry tracker.
        # Other selection state (last_picked_id / pick_count / deselect_count /
        # flip_count) self-resets on family transition; only the per-episode
        # entry counter and active edge need explicit episode reset.
        self._selection_screen_entries = 0
        self._selection_screen_active = False
        # TASK-A4: reset per-combat potion-use accounting.
        self._used_potion_count_this_combat = 0
        self._potion_transition_records = []
        # P0-2: clear actionability cache between combats so a leftover
        # transient flag from a previous combat does not leak into the next.
        self._last_actionability = None

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
            result = self._backend_combat_reset(
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
        # NOTE: the previous lines here mirrored step()'s post-action diagnostics
        # merge (pre_action_diagnostics + end_turn_penalty), but those names do
        # not exist in reset()'s scope.  Reset has no chosen-action context, so
        # any bridge-side action_diagnostics on the reset payload is forwarded
        # as-is via _update_live_state below.
        self._update_live_state(result)
        sentinel_enemy = self._find_sentinel_enemy(self._last_obs_raw)
        self._sentinel_combat_active = sentinel_enemy is not None
        if self._sentinel_combat_active:
            _, start_max = self._player_hp_and_max(self._last_obs_raw)
            self._sentinel_combat_start_max_hp = start_max
        else:
            self._sentinel_combat_start_max_hp = 0.0
        # Snapshot episode-start HP for the terminal preserve-bonus / tier
        # outcome reward (combat-reward-curriculum.md §6.2).
        self._episode_start_hp, self._episode_start_max_hp = self._player_hp_and_max(self._last_obs_raw)
        # Snapshot the boss tier total enemy HP at episode start so the
        # boss-loss damage-undo can compute "how much damage was rewarded
        # via per-step shaping" without per-step bookkeeping.  Stored as a
        # plain float; only consumed in `_boss_terminal_reward` when loss.
        self._episode_start_boss_total_hp: float = float(self._combat_enemy_total_hp(self._last_obs_raw))
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
        if self._human_demo_recorder is not None:
            try:
                self._human_demo_recorder.start_episode(
                    episode_id=self._episode_id,
                    encounter_id=self._current_encounter_id,
                    reset_kwargs=self._last_reset_kwargs,
                    obs=self._last_obs_raw,
                    legal_actions=self._legal_actions,
                    metadata={
                        "episode_mode": "combat_sandbox",
                        "snapshot_sample_id": info.get("snapshot_sample_id"),
                        "snapshot_run_id": info.get("snapshot_run_id"),
                    },
                )
            except Exception as exc:
                print(f"[combat_env] human demo episode_start failed: {exc}", flush=True)
        return obs, info

    def step(self, action: int):
        if not self._legal_actions:
            return self._make_terminal()
        started = time.perf_counter()

        normalized_action = self._normalize_action(action)
        if normalized_action is None or normalized_action >= len(self._legal_actions):
            return self._make_invalid_action_response(action)

        legal_action = self._legal_actions[normalized_action]
        if self._is_end_turn_action(legal_action):
            pre_dispatch_response = self._recover_pre_dispatch_end_turn_frontier(
                attempted_action_index=action,
                selected_action=legal_action if isinstance(legal_action, dict) else None,
            )
            if pre_dispatch_response is not None:
                return pre_dispatch_response

        legal_actions_before = list(self._legal_actions)
        prev_obs = self._last_obs_raw or {}
        prev_planner_context = self._planner_context()
        pre_action_diagnostics = self._action_quality_diagnostics(prev_obs, self._legal_actions, legal_action)
        end_turn_penalty = self._end_turn_waste_penalty(prev_obs, self._legal_actions, legal_action)
        # P0-2: when the policy was forced into ``end_turn`` because the
        # frontier it saw was transient-only-end_turn (draw / shuffle /
        # animation / queue still resolving), refund the wasteful-end-turn
        # penalty and tag this transition for the trainer to drop / down-
        # weight.  ``self._last_actionability`` is the bridge actionability
        # block from the PRIOR step (the one whose legal actions we are now
        # consuming).
        prior_actionability = self._last_actionability if isinstance(self._last_actionability, dict) else None
        prior_transient_only_end_turn = bool((prior_actionability or {}).get("transient_only_end_turn", False))
        action_id = str(legal_action.get("action_id") or "") if isinstance(legal_action, dict) else ""
        transient_leaked_now = bool(prior_transient_only_end_turn and action_id == "end_turn")
        if transient_leaked_now and end_turn_penalty < 0.0:
            end_turn_penalty = 0.0
            self._fast_step_metrics_total["transient_leaked_count"] += 1

        bridge_started = time.perf_counter()
        try:
            result = self._backend_step(
                episode_id=self._episode_id,
                action_id=legal_action.get("action_id"),
                timeout_ms=self.step_timeout_ms,
            )
        except BridgeError as e:
            # Bridge rejected the action (e.g. TOCTOU race, phase mismatch).
            # We cannot safely continue with the stale _last_obs_raw /
            # _legal_actions �?the next step would sample from a mask that no
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
        if bridge_elapsed_ms > 1500.0:
            action_id_diag = legal_action.get("action_id") if isinstance(legal_action, dict) else None
            print(
                f"[combat_env] slow_bridge_step ms={bridge_elapsed_ms:.0f} action_id={action_id_diag}",
                flush=True,
            )

        post_step_frontier_recovery: dict[str, Any] | None = None
        try:
            result, post_step_frontier_recovery = self._recover_post_step_frontier(
                result,
                selected_action=legal_action if isinstance(legal_action, dict) else None,
                legal_actions_before=legal_actions_before,
            )
        except Exception as exc:
            # Never let diagnostics/recovery crash the collector.  If this
            # trips, keep the original bridge result but surface a compact
            # record so the run log can distinguish "no recovery attempted"
            # from "recovery code failed".
            post_step_frontier_recovery = {
                "attempted": False,
                "error": "post_step_frontier_recovery_exception",
                "exception_type": type(exc).__name__,
            }

        canonical_reward = self._record_backend_transition(
            result,
            action_handle=str(legal_action.get("action_id") or ""),
        )
        self._update_live_state(result)
        run_memory_started = time.perf_counter()
        self._run_memory.update_transition(prev_obs, legal_action, self._last_obs_raw, legal_actions=self._legal_actions)
        self._combat_memory.update(prev_obs, legal_action, self._last_obs_raw)
        run_memory_elapsed_ms = (time.perf_counter() - run_memory_started) * 1000.0

        next_planner_context = self._planner_context()
        obs_encode_started = time.perf_counter()
        obs = self.obs_encoder.encode(self._last_obs_raw, self._legal_actions, next_planner_context)
        obs_encode_elapsed_ms = (time.perf_counter() - obs_encode_started) * 1000.0
        reward = float(canonical_reward.total)
        terminated = bool(result.get("done", False))
        truncated = bool(result.get("truncated", False))
        if not self._uses_external_v2_reward(result):
            # Deprecated v1 compatibility only. Active live/headless paths use
            # the versioned fact calculator above as the sole reward owner.
            reward += self._enemy_hp_delta_reward(prev_obs, self._last_obs_raw)
            reward += self._player_hp_delta_reward(prev_obs, self._last_obs_raw)
            reward += end_turn_penalty
            reward += self._turn_efficiency_penalty(legal_action)
            reward += self._encounter_potion_use_reward(legal_action)
            reward += self._potion_hoarding_terminal_reward(
                self._last_obs_raw,
                terminated,
                truncated,
            )
            reward += self._potion_timing_step_reward(action, prev_obs, legal_actions_before)
            reward += self._card_selection_step_reward(legal_action)
            reward += self._tier_outcome_reward(self._last_obs_raw, terminated, truncated)
            reward += self._hp_preserve_win_bonus(self._last_obs_raw, terminated, truncated)
            reward += self._boss_terminal_reward(
                prev_obs,
                self._last_obs_raw,
                terminated,
                truncated,
            )
            reward += self._boss_mechanic_reward(prev_obs, self._last_obs_raw, legal_action)
            reward += self._sentinel_terminal_reward(
                prev_obs,
                self._last_obs_raw,
                terminated,
                truncated,
            )
        if terminated or truncated:
            self._sentinel_combat_active = False
            # Curriculum bookkeeping §13: record encounter win/loss and emit
            # any phase-switch annotations.  Must run BEFORE the next reset so
            # the tracker window stays aligned with terminal events only.
            self._record_terminal_outcome_for_curriculum(
                self._last_obs_raw, terminated, truncated
            )
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
        action_diagnostics = dict(pre_action_diagnostics)
        action_diagnostics["wasteful_end_turn_penalty_applied"] = float(end_turn_penalty < 0.0)
        potion_transition_record = self._build_potion_transition_record(
            action=legal_action,
            prev_obs=prev_obs,
            after_obs=self._last_obs_raw,
            bridge_result=result,
            bridge_error=None,
        )
        if potion_transition_record is not None:
            self._potion_transition_records.append(potion_transition_record)
            if bool(potion_transition_record.get("execute_ok")):
                self._used_potion_count_this_combat += 1
        # TASK-C2/P0-2: read bridge actionability and surface post-step
        # frontier recovery flags.  The recovery path is deliberately
        # conservative: a singleton End Turn with leftover energy is only a
        # suspicion; it becomes evidence of a transient leak only if a short
        # /state poll rebounds into non-EndTurn actions or the bridge itself
        # flagged the frame as transient/unstable.
        bridge_info = result.get("info") if isinstance(result.get("info"), dict) else {}
        actionability = bridge_info.get("actionability") if isinstance(bridge_info.get("actionability"), dict) else None
        transient_flag = bool((actionability or {}).get("transient_only_end_turn", False))
        if transient_flag:
            self._fast_step_metrics_total["transient_only_end_turn_count"] += 1
            action_diagnostics["transient_only_end_turn"] = True
        elif (
            isinstance(actionability, dict)
            and int(actionability.get("legal_non_end_turn_count", 0) or 0) == 0
        ):
            self._fast_step_metrics_total["stable_no_actions_count"] += 1
            action_diagnostics["transient_only_end_turn"] = False
        # P0-2: tag the transition with leak status so the trainer can drop
        # or down-weight it.  ``transient_leaked`` here means: the policy
        # selected ``end_turn`` while the prior frontier was reported as
        # transient-only-end_turn — i.e. the policy was effectively forced.
        action_diagnostics["transient_leaked"] = transient_leaked_now
        action_diagnostics["prior_transient_only_end_turn"] = prior_transient_only_end_turn
        if isinstance(post_step_frontier_recovery, dict):
            frontier_scalar_map = {
                "attempted": "post_step_frontier_attempted",
                "resolved": "post_step_frontier_resolved",
                "timeout": "post_step_frontier_timeout",
                "leaked": "post_step_frontier_leaked",
                "stable_no_actions": "post_step_frontier_stable_no_actions",
                "rebind_attempted": "post_step_frontier_rebind_attempted",
                "rebind_succeeded": "post_step_frontier_rebind_succeeded",
                "suspicious_singleton": "post_step_frontier_suspicious_singleton",
            }
            for src_key, dst_key in frontier_scalar_map.items():
                action_diagnostics[dst_key] = 1.0 if post_step_frontier_recovery.get(src_key) else 0.0
            action_diagnostics["post_step_frontier_wait_ms"] = float(
                post_step_frontier_recovery.get("wait_ms") or 0.0
            )
            action_diagnostics["post_step_frontier_poll_count"] = float(
                post_step_frontier_recovery.get("poll_count") or 0.0
            )
        # P0-1/4/5/6: surface typed safety / x-cost / selection / identity views
        # so the trainer-side aggregator picks them up without re-reading the
        # raw action.  Each block is a small JSON-able dict; nothing on the
        # GPU path uses them, so wrap in try/except to avoid env-side crashes.
        try:
            from .card_identity import card_identity
            from .card_runtime_state import (
                card_runtime_presence_flags,
                card_runtime_state,
            )
            from .hp_cost_safety import hp_cost_safety_view
            from .selection_typed import selection_view
            from .x_cost_dynamic import x_cost_view
            played_card = legal_action.get("card") if isinstance(legal_action, dict) else None
            hp_safety = hp_cost_safety_view(legal_action, prev_obs)
            xcost = x_cost_view(legal_action, prev_obs)
            sel = selection_view(legal_action)
            ident = card_identity(played_card)
            runtime_state = card_runtime_state(played_card)
            runtime_presence = card_runtime_presence_flags(played_card)
            # Rich dicts (offline analysis / future-world aux)
            action_diagnostics["hp_cost_safety"] = hp_safety
            action_diagnostics["x_cost"] = xcost
            action_diagnostics["selection"] = sel
            action_diagnostics["card_identity"] = ident
            action_diagnostics["card_runtime_state"] = runtime_state
            # Flat scalars (fed into the trainer-side diag_key_map → TB metric
            # aggregation path; mirrored under combat_quality_* by train.py).
            action_diagnostics["hp_cost_self_lethal_selected"] = 1.0 if hp_safety.get("self_lethal_now") else 0.0
            action_diagnostics["hp_cost_low_margin_selected"] = 1.0 if hp_safety.get("low_hp_margin_after_cost") else 0.0
            action_diagnostics["hp_cost_unblockable_value"] = float(hp_safety.get("hp_loss_unblockable") or 0.0)
            xres = str(xcost.get("resource") or "none")
            xcv = float(xcost.get("current_value") or 0.0)
            action_diagnostics["x_cost_selected"] = 1.0 if xcost.get("has_x_cost") else 0.0
            action_diagnostics["x_cost_zero_bad_selected"] = 1.0 if xcost.get("zero_x_bad") else 0.0
            action_diagnostics["x_cost_zero_selected"] = 1.0 if (xcost.get("is_zero") and xcost.get("has_x_cost")) else 0.0
            action_diagnostics["x_cost_energy_value"] = xcv if xres == "energy" else 0.0
            action_diagnostics["x_cost_star_value"] = xcv if xres == "stars" else 0.0
            action_diagnostics["star_x_selected"] = 1.0 if (xcost.get("has_x_cost") and xres == "stars") else 0.0
            action_diagnostics["selection_text_fallback_selected"] = 1.0 if sel.get("confidence") == "text_fallback" else 0.0
            action_diagnostics["selection_runtime_internal_selected"] = 1.0 if sel.get("confidence") == "runtime_internal" else 0.0
            action_diagnostics["card_identity_text_fallback_selected"] = 1.0 if ident.get("confidence") == "text_fallback" else 0.0
            action_diagnostics["card_identity_runtime_internal_selected"] = 1.0 if ident.get("confidence") == "runtime_internal" else 0.0
            # P2-1 (recovery 2026-05-07): runtime card-state presence flags.
            # Each is the per-decision indicator that the bridge exposed the
            # corresponding runtime modifier on the chosen card. Absent =>
            # bridge gap, not a model bug — surfaces in TB so we can see at
            # a glance what fraction of cards even get the typed signal.
            action_diagnostics["card_runtime_instance_uuid_present"] = float(
                runtime_presence.get("instance_uuid_present", 0.0) or 0.0
            )
            action_diagnostics["card_runtime_modified_cost_present"] = float(
                runtime_presence.get("modified_cost_present", 0.0) or 0.0
            )
            action_diagnostics["card_runtime_exhaust_flag_present"] = float(
                runtime_presence.get("exhaust_flag_present", 0.0) or 0.0
            )
            action_diagnostics["card_runtime_ethereal_flag_present"] = float(
                runtime_presence.get("ethereal_flag_present", 0.0) or 0.0
            )
            action_diagnostics["card_runtime_retain_flag_present"] = float(
                runtime_presence.get("retain_flag_present", 0.0) or 0.0
            )
            action_diagnostics["card_runtime_enchantment_present"] = float(
                runtime_presence.get("enchantment_present", 0.0) or 0.0
            )
            action_diagnostics["card_runtime_replay_flag_present"] = float(
                runtime_presence.get("replay_flag_present", 0.0) or 0.0
            )
            action_diagnostics["card_runtime_selection_effect_present"] = float(
                runtime_presence.get("selection_effect_present", 0.0) or 0.0
            )
        except Exception:
            pass
        # Cache the post-step actionability for the NEXT step's leak check.
        self._last_actionability = actionability if isinstance(actionability, dict) else None
        from .boss_mechanics import build_boss_mechanics_block, classify_action_boss_mechanism
        boss_mechanics_block = build_boss_mechanics_block(self._last_obs_raw)
        boss_action_mechanism = classify_action_boss_mechanism(
            prev_obs, legal_action, action_diagnostics=action_diagnostics
        )
        extra: dict[str, Any] = {
            "action_diagnostics": action_diagnostics,
            "aux_targets": aux_targets,
            "used_potion_count_this_combat": int(self._used_potion_count_this_combat),
            "boss_mechanics": boss_mechanics_block,
            "boss_action_mechanism": boss_action_mechanism,
        }
        if actionability is not None:
            extra["actionability"] = actionability
        extra["bridge_fast_step_metrics_cumulative"] = dict(self._fast_step_metrics_total)
        if isinstance(post_step_frontier_recovery, dict):
            extra["post_step_frontier_recovery"] = {
                k: v
                for k, v in post_step_frontier_recovery.items()
                if k != "trace"
            }
            frontier_trace = post_step_frontier_recovery.get("trace")
            if isinstance(frontier_trace, dict):
                extra["frontier_trace"] = frontier_trace
        if potion_transition_record is not None:
            extra["potion_transition"] = potion_transition_record
        info = self._build_info(
            result.get("info", {}),
            extra={
                **extra,
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

        if self._human_demo_recorder is not None:
            try:
                self._human_demo_recorder.record_decision(
                    obs=prev_obs,
                    legal_actions=legal_actions_before,
                    selected_action=legal_action,
                    selected_action_index=int(normalized_action),
                    next_obs=self._last_obs_raw,
                    reward=float(reward),
                    done=bool(terminated),
                    truncated=bool(truncated),
                    info=info,
                )
            except Exception as exc:
                print(f"[combat_env] human demo decision record failed: {exc}", flush=True)

        return obs, reward, terminated, truncated, info


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
        if self._human_demo_recorder is not None:
            self._human_demo_recorder.close()
        self._close_environment_runtime()


    @property
    def raw_obs(self) -> dict[str, Any] | None:
        return self._last_obs_raw

    @property
    def legal_actions(self) -> list[dict[str, Any]]:
        return self._legal_actions

    @property
    def last_reset_kwargs(self) -> dict[str, Any]:
        return dict(self._last_reset_kwargs)

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
                "snapshot_floor_number": _float(obs.get("snapshot_floor_number", run.get("snapshot_floor_number"))),
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
            # Combat sandbox has no map �?floor is always 0. But Monitor's
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

    def _make_frontier_refreshed_response(
        self,
        attempted_action: Any,
        *,
        metrics: dict[str, Any] | None = None,
        reason: str = "frontier_refreshed_before_end_turn",
        refreshed: bool = True,
        high_confidence: bool = False,
    ):
        """Return a non-terminal no-op after blocking stale EndTurn dispatch.

        This response intentionally does *not* mark the action invalid.  The
        policy selected EndTurn from the mask it was shown; the environment then
        discovered, before calling ``bridge.step``, that the singleton frontier
        was stale or high-confidence suspicious.  Returning the current/refreshed
        observation lets self-play reselect without burning the game turn.
        """

        obs = self.obs_encoder.encode(self._last_obs_raw or {}, self._legal_actions, self._planner_context())
        metrics_out = {
            k: v
            for k, v in dict(metrics or {}).items()
            if k != "trace"
        }
        bridge_info = {
            "action_error": reason,
            "step_recovery": reason,
            "action_diagnostics": {
                "frontier_pre_dispatch_refreshed": 1.0 if refreshed else 0.0,
                "frontier_pre_dispatch_end_turn_blocked": 1.0,
                "frontier_pre_dispatch_high_confidence": 1.0 if high_confidence else 0.0,
            },
        }
        extra: dict[str, Any] = {
            "frontier_refreshed_before_end_turn": bool(refreshed),
            "frontier_stale_singleton_end_turn_blocked": bool(not refreshed),
            "frontier_refreshed_attempted_action": attempted_action,
            "pre_dispatch_frontier_recovery": metrics_out,
            "bridge_fast_step_metrics_cumulative": dict(self._fast_step_metrics_total),
            "python_timing_ms": self._python_timing(
                bridge_roundtrip=0.0,
                run_memory_update=0.0,
                obs_encode=0.0,
                aux_targets=0.0,
                info_build=0.0,
                total=0.0,
            ),
        }
        trace = (metrics or {}).get("trace") if isinstance(metrics, dict) else None
        if isinstance(trace, dict):
            extra["frontier_trace"] = trace
        info = self._build_info(bridge_info, extra=extra)
        return obs, 0.0, False, False, info

# Preserve the historical public class/function identities.
CurriculumTracker.__module__ = __name__
get_curriculum_tracker.__module__ = __name__

__all__ = [
    'BLOCKED_ACTION_KINDS',
    'BOSS_COMBAT_LOSS_DAMAGE_UNDO_PERCENT_SCALE',
    'BOSS_COMBAT_LOSS_PENALTY_BASE',
    'BOSS_COMBAT_LOSS_PENALTY_MISSING_HP_SCALE',
    'BOSS_COMBAT_WIN_BONUS_BASE',
    'BOSS_COMBAT_WIN_BONUS_HP_SCALE',
    'BOSS_DAMAGE_MULTIPLIER',
    'BOSS_ENEMY_HP_DELTA_PERCENT_SCALE',
    'CEREMONIAL_ONE_CARD_END_TURN_PENALTY',
    'CEREMONIAL_ONE_CARD_HIGH_IMPACT_BONUS',
    'CEREMONIAL_ONE_CARD_LOW_IMPACT_PENALTY',
    'CEREMONIAL_STUN_DAMAGE_MULTIPLIER',
    'CEREMONIAL_STUN_WINDOW_ENTER_BONUS',
    'CEREMONIAL_THRESHOLD_PROGRESS_BONUS',
    'CURRICULUM_HP_WEIGHT_LERP_MAX_TIER',
    'CURRICULUM_HP_WEIGHT_LERP_MIN_TIER',
    'CURRICULUM_MIN_EPISODES_FOR_PHASE',
    'CURRICULUM_PHASE_WIN_RATE_THRESHOLDS',
    'CURRICULUM_WIN_RATE_WINDOW',
    'END_TURN_WASTE_BASE_PENALTY',
    'END_TURN_WASTE_ENERGY_PENALTY',
    'END_TURN_WASTE_EXTRA_ACTION_PENALTY',
    'END_TURN_WASTE_ZERO_COST_BONUS_PENALTY',
    'ENEMY_HP_DELTA_REWARD_MAX_ABS',
    'ENEMY_HP_DELTA_REWARD_SCALE',
    'ENEMY_HP_SENTINEL_THRESHOLD',
    'HP_PRESERVE_WIN_BONUS_TIER_SCALE',
    'INVALID_ACTION_REASON',
    'INVALID_ACTION_REWARD',
    'KAISER_BACK_ATTACK_DEFENSE_BONUS',
    'KAISER_BACK_ATTACK_END_TURN_PENALTY',
    'KAISER_BACK_ATTACK_HI_THREAT_DMG',
    'KAISER_BACK_ATTACK_HI_THREAT_EXTRA',
    'KAISER_BACK_ATTACK_HP_LOSS_PENALTY_SCALE',
    'KAISER_BACK_ATTACK_RISK_REDUCTION_BONUS',
    'KAISER_FACING_CHANGE_BONUS',
    'KAISER_FACING_CHANGE_BONUS_BASE',
    'KAISER_FACING_INTENT_DMG_REF',
    'KAISER_FACING_INTENT_DMG_SCALE_MAX',
    'KAISER_NO_RESPONSE_PENALTY_SOFTEN',
    'KAISER_PRESSURE_KILL_BONUS',
    'KNOWLEDGE_DEMON_BAD_CURSE_PICK_PENALTY',
    'KNOWLEDGE_DEMON_END_TURN_PENALTY',
    'KNOWLEDGE_DEMON_GOOD_CURSE_PICK_BONUS',
    'MAX_ACTIONS',
    'OUTCOME_TIER_SCALE',
    'PLAYER_HP_LOSS_BOSS_NO_HEAL_SCALE',
    'PLAYER_HP_LOSS_REWARD_SCALE',
    'PLAYER_HP_LOSS_TIER_SCALE',
    'POTION_HOARDING_MAX_PENALTY_ABS',
    'POTION_HOARDING_PENALTY_PER_POTION',
    'POTION_TIMING_QUALITY_SCALE',
    'POTION_TIMING_WASTE_SCALE',
    'POTION_USE_BOSS_BONUS',
    'POTION_USE_ELITE_BONUS',
    'POTION_USE_MONSTER_BONUS',
    'POTION_USE_MONSTER_PENALTY',
    'SELECTION_DESELECT_PENALTY',
    'SELECTION_EARLY_CONFIRM_BONUS',
    'SELECTION_LOOP_PENALTY',
    'SELECTION_OVER_CAP_PENALTY',
    'SELECTION_PICK_CAP',
    'SELECTION_REENTRY_BUDGET',
    'SELECTION_REENTRY_PENALTY',
    'SENTINEL_COMBAT_LOSS_PENALTY_BASE',
    'SENTINEL_COMBAT_LOSS_PENALTY_SCALE',
    'SENTINEL_COMBAT_WIN_BONUS_BASE',
    'SENTINEL_COMBAT_WIN_BONUS_SCALE',
    'SENTINEL_DEATH_DAMAGE_POWER_KEYWORDS',
    'TURN_EFFICIENCY_PENALTY_PER_END_TURN_TIER',
    'WASTEFUL_END_TURN_TIER_MULTIPLIER',
    'BridgeClient',
    'BridgeError',
    'CombatMemoryTracker',
    'CombatSandboxEnv',
    'CurriculumTracker',
    'DenseObservationEncoder',
    'RunMemoryTracker',
    'WorldTokenObservationEncoder',
    'aggregate_card_effect_profile_semantics',
    'build_aux_targets',
    'build_boss_mechanics_context',
    'compact_legal_actions',
    'compute_potion_timing',
    'get_curriculum_tracker',
    'snapshot_row_to_reset_kwargs',
    'strict_end_turn_waste_context',
]
