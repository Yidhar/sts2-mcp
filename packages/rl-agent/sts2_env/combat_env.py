"""Combat sandbox Gymnasium wrapper for STS2.

This mirrors :mod:`env_v2` but resets through ``/env/combat_reset``. Training
defaults to compact ``info`` payloads so rollout hot paths avoid carrying large
raw observation trees unless explicitly requested by debug/eval callers.
"""

from __future__ import annotations

import os
import time
from typing import Any, Callable

import gymnasium as gym
import numpy as np
from gymnasium import spaces

from combat_snapshot_dataset import snapshot_row_to_reset_kwargs
from .action_compact import compact_legal_actions
from .boss_mechanics import build_boss_mechanics_context
from .aux_targets import build_aux_targets
from .bridge_client import BridgeClient, BridgeError
from .card_effect_profile import aggregate_card_effect_profile_semantics
from .observation_common import DenseObservationEncoder, MAX_ACTIONS, _aggregate_card_modifier_semantics
from .observation_v3 import WorldTokenObservationEncoder
from .potion_timing import compute_potion_timing
from .end_turn_quality import strict_end_turn_waste_context
from .combat_memory import CombatMemoryTracker
from .run_memory import RunMemoryTracker

from .reward_constants import (
    COMBAT_SANDBOX_WASTE_BASE as END_TURN_WASTE_BASE_PENALTY,
    COMBAT_SANDBOX_WASTE_ENERGY as END_TURN_WASTE_ENERGY_PENALTY,
    COMBAT_SANDBOX_WASTE_ZERO_COST as END_TURN_WASTE_ZERO_COST_BONUS_PENALTY,
    COMBAT_SANDBOX_WASTE_EXTRA_ACTION as END_TURN_WASTE_EXTRA_ACTION_PENALTY,
    BOSS_COMBAT_LOSS_DAMAGE_UNDO_PERCENT_SCALE,
    BOSS_COMBAT_LOSS_PENALTY_BASE,
    BOSS_COMBAT_LOSS_PENALTY_MISSING_HP_SCALE,
    BOSS_COMBAT_WIN_BONUS_BASE,
    BOSS_COMBAT_WIN_BONUS_HP_SCALE,
    BOSS_ENEMY_HP_DELTA_PERCENT_SCALE,
    BOSS_DAMAGE_MULTIPLIER,
    CURRICULUM_HP_WEIGHT_LERP_MAX_TIER,
    CURRICULUM_HP_WEIGHT_LERP_MIN_TIER,
    CURRICULUM_MIN_EPISODES_FOR_PHASE,
    CURRICULUM_PHASE_WIN_RATE_THRESHOLDS,
    CURRICULUM_WIN_RATE_WINDOW,
    HP_PRESERVE_WIN_BONUS_TIER_SCALE,
    KAISER_BACK_ATTACK_HP_LOSS_PENALTY_SCALE,
    KAISER_BACK_ATTACK_DEFENSE_BONUS,
    KAISER_BACK_ATTACK_RISK_REDUCTION_BONUS,
    KAISER_BACK_ATTACK_END_TURN_PENALTY,
    KAISER_FACING_CHANGE_BONUS,
    KAISER_FACING_CHANGE_BONUS_BASE,
    KAISER_FACING_INTENT_DMG_REF,
    KAISER_FACING_INTENT_DMG_SCALE_MAX,
    KAISER_BACK_ATTACK_HI_THREAT_DMG,
    KAISER_BACK_ATTACK_HI_THREAT_EXTRA,
    KAISER_PRESSURE_KILL_BONUS,
    KAISER_NO_RESPONSE_PENALTY_SOFTEN,
    KNOWLEDGE_DEMON_GOOD_CURSE_PICK_BONUS,
    KNOWLEDGE_DEMON_BAD_CURSE_PICK_PENALTY,
    KNOWLEDGE_DEMON_END_TURN_PENALTY,
    CEREMONIAL_STUN_WINDOW_ENTER_BONUS,
    CEREMONIAL_THRESHOLD_PROGRESS_BONUS,
    CEREMONIAL_STUN_DAMAGE_MULTIPLIER,
    CEREMONIAL_ONE_CARD_HIGH_IMPACT_BONUS,
    CEREMONIAL_ONE_CARD_LOW_IMPACT_PENALTY,
    CEREMONIAL_ONE_CARD_END_TURN_PENALTY,
    ENEMY_HP_DELTA_REWARD_MAX_ABS,
    ENEMY_HP_DELTA_REWARD_SCALE,
    ENEMY_HP_SENTINEL_THRESHOLD,
    INVALID_ACTION_REWARD,
    OUTCOME_TIER_SCALE,
    PLAYER_HP_LOSS_BOSS_NO_HEAL_SCALE,
    PLAYER_HP_LOSS_REWARD_SCALE,
    PLAYER_HP_LOSS_TIER_SCALE,
    WASTEFUL_END_TURN_TIER_MULTIPLIER,
    POTION_HOARDING_MAX_PENALTY_ABS,
    POTION_HOARDING_PENALTY_PER_POTION,
    POTION_USE_BOSS_BONUS,
    POTION_USE_ELITE_BONUS,
    POTION_USE_MONSTER_BONUS,
    POTION_USE_MONSTER_PENALTY,
    POTION_TIMING_QUALITY_SCALE,
    POTION_TIMING_WASTE_SCALE,
    SELECTION_LOOP_PENALTY,
    SELECTION_EARLY_CONFIRM_BONUS,
    SELECTION_DESELECT_PENALTY,
    SELECTION_PICK_CAP,
    SELECTION_OVER_CAP_PENALTY,
    SELECTION_REENTRY_BUDGET,
    SELECTION_REENTRY_PENALTY,
    SENTINEL_COMBAT_LOSS_PENALTY_BASE,
    SENTINEL_COMBAT_LOSS_PENALTY_SCALE,
    SENTINEL_COMBAT_WIN_BONUS_BASE,
    SENTINEL_COMBAT_WIN_BONUS_SCALE,
    SENTINEL_DEATH_DAMAGE_POWER_KEYWORDS,
    TURN_EFFICIENCY_PENALTY_PER_END_TURN_TIER,
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


# ---------------------------------------------------------------------------
# Curriculum tracker (combat-reward-curriculum.md §4, §13)
# ---------------------------------------------------------------------------
#
# Keeps a sliding window of the last N win/loss outcomes per encounter.  The
# resulting per-encounter win rate drives a 0..1 "progress" scalar:
#
#   progress = clamp((win_rate - P0_P1_threshold) / (P2_P3_threshold - P0_P1_threshold), 0, 1)
#
# which lerps the HP-loss shaping multiplier between
# CURRICULUM_HP_WEIGHT_LERP_MIN_TIER[tier] and CURRICULUM_HP_WEIGHT_LERP_MAX_TIER[tier].
# This keeps the policy in "just win baby" mode until it can actually win, then
# gradually ramps up HP pressure as competence grows.
#
# The tracker is *process-wide* and deliberately NOT persisted across runs �?
# each training process accumulates its own view of progression.  That matches
# how Phase 0..3 should reset when the checkpoint/config changes enough that
# the policy may regress.
class CurriculumTracker:
    """Sliding per-encounter win/loss tracker with phase classification."""

    __slots__ = ("_window", "_outcomes", "_phase_cache")

    def __init__(self, window: int = CURRICULUM_WIN_RATE_WINDOW) -> None:
        self._window = max(4, int(window))
        # encounter_id (str) -> list[int] of 0/1 outcomes, newest at end
        self._outcomes: dict[str, list[int]] = {}
        # encounter_id -> last reported phase (0..3).  Used to emit a one-line
        # "[curriculum] phase N→M" switch annotation when the phase changes.
        self._phase_cache: dict[str, int] = {}

    def record(self, encounter_id: str, win: bool) -> None:
        key = str(encounter_id or "unknown").strip().lower() or "unknown"
        bucket = self._outcomes.setdefault(key, [])
        bucket.append(1 if win else 0)
        if len(bucket) > self._window:
            del bucket[: len(bucket) - self._window]

    def win_rate(self, encounter_id: str) -> tuple[float, int]:
        key = str(encounter_id or "unknown").strip().lower() or "unknown"
        bucket = self._outcomes.get(key, [])
        if not bucket:
            return 0.0, 0
        return float(sum(bucket)) / float(len(bucket)), len(bucket)

    @staticmethod
    def _phase_for_win_rate(win_rate: float) -> int:
        p0_p1, p1_p2, p2_p3 = CURRICULUM_PHASE_WIN_RATE_THRESHOLDS
        if win_rate < p0_p1:
            return 0
        if win_rate < p1_p2:
            return 1
        if win_rate < p2_p3:
            return 2
        return 3

    def phase(self, encounter_id: str) -> int:
        wr, n = self.win_rate(encounter_id)
        if n < CURRICULUM_MIN_EPISODES_FOR_PHASE:
            return 0
        return self._phase_for_win_rate(wr)

    def progress(self, encounter_id: str) -> float:
        """0.0 at win_rate<=30%, 1.0 at win_rate>=80%, linear in between."""
        wr, n = self.win_rate(encounter_id)
        if n < CURRICULUM_MIN_EPISODES_FOR_PHASE:
            return 0.0
        p0_p1, _, p2_p3 = CURRICULUM_PHASE_WIN_RATE_THRESHOLDS
        span = max(1e-6, p2_p3 - p0_p1)
        return float(np.clip((wr - p0_p1) / span, 0.0, 1.0))

    def hp_weight(self, encounter_id: str, tier: str) -> float:
        """Current tier-aware HP-loss shaping multiplier for this encounter.

        Returns the lerped weight that should be MULTIPLIED with
        PLAYER_HP_LOSS_TIER_SCALE[tier] to get the effective scale.
        """
        prog = self.progress(encounter_id)
        tier_key = tier if tier in CURRICULUM_HP_WEIGHT_LERP_MIN_TIER else "unknown"
        lo = float(CURRICULUM_HP_WEIGHT_LERP_MIN_TIER[tier_key])
        hi = float(CURRICULUM_HP_WEIGHT_LERP_MAX_TIER[tier_key])
        return lo + prog * (hi - lo)

    def check_phase_switch(self, encounter_id: str) -> tuple[int, int] | None:
        """Return (old_phase, new_phase) if a phase boundary was just crossed."""
        key = str(encounter_id or "unknown").strip().lower() or "unknown"
        new_phase = self.phase(key)
        old_phase = self._phase_cache.get(key, -1)
        if new_phase != old_phase:
            self._phase_cache[key] = new_phase
            if old_phase >= 0:
                return (old_phase, new_phase)
        return None

    def dump_all_state(self) -> str:
        """Render every tracked encounter's current (tier, n, wr, phase) as
        one line per encounter, sorted by tier-then-name.

        Used for the periodic `[curriculum/state]` snapshot so the operator
        can see encounters that haven't yet crossed a phase boundary (which
        would normally never appear in the per-transition log).
        """
        def _tier_for(enc: str) -> str:
            e = enc.lower()
            if "boss" in e:
                return "boss"
            if "elite" in e:
                return "elite"
            if "weak" in e:
                return "weak"
            return "normal"

        # Order: boss > elite > normal > weak so heaviest tiers print first.
        tier_order = {"boss": 0, "elite": 1, "normal": 2, "weak": 3, "unknown": 4}
        rows: list[tuple[int, str, str, int, float, int]] = []
        for encounter, bucket in self._outcomes.items():
            tier = _tier_for(encounter)
            n = len(bucket)
            wr = float(sum(bucket)) / float(n) if n > 0 else 0.0
            phase = self._phase_for_win_rate(wr) if n >= CURRICULUM_MIN_EPISODES_FOR_PHASE else 0
            rows.append((tier_order.get(tier, 4), encounter, tier, n, wr, phase))
        rows.sort()
        lines = [
            f"  {tier:>6} {enc} n={n} wr={wr:.3f} phase=P{phase}"
            for _, enc, tier, n, wr, phase in rows
        ]
        return "\n".join(lines) if lines else "  (no encounters tracked)"


_CURRICULUM_TRACKER_SINGLETON: CurriculumTracker | None = None


def get_curriculum_tracker() -> CurriculumTracker:
    """Process-wide shared CurriculumTracker so all envs contribute to the same window."""
    global _CURRICULUM_TRACKER_SINGLETON
    if _CURRICULUM_TRACKER_SINGLETON is None:
        _CURRICULUM_TRACKER_SINGLETON = CurriculumTracker()
    return _CURRICULUM_TRACKER_SINGLETON


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
            from .human_demo_recorder import HumanDemoRecorder  # noqa: WPS433

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
            result = self.bridge.step(
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
        # --- R_hp_efficiency: per-step HP delta with tier × curriculum weights ---
        reward += self._enemy_hp_delta_reward(prev_obs, self._last_obs_raw)
        reward += self._player_hp_delta_reward(prev_obs, self._last_obs_raw)
        # --- R_action_quality: wasteful end_turn (existing detector) ---
        reward += end_turn_penalty
        # --- R_turn_efficiency §8.2: per-end_turn tier-aware penalty ---
        reward += self._turn_efficiency_penalty(legal_action)
        terminated = bool(result.get("done", False))
        truncated = bool(result.get("truncated", False))
        # --- R_resource_quality: all potion shaping zeroed in v4 (constants=0) ---
        reward += self._encounter_potion_use_reward(legal_action)
        reward += self._potion_hoarding_terminal_reward(self._last_obs_raw, terminated, truncated)
        # --- R_potion_timing (Phase 4b): use_quality - waste_risk shaping. ---
        reward += self._potion_timing_step_reward(action, prev_obs, legal_actions_before)
        # --- R_selection_quality: anti-loop + early-confirm for multi-pick
        # burn cards (§4 of kaiser-and-potion-fixes-todo.md). ---
        reward += self._card_selection_step_reward(legal_action)
        # --- R_outcome: tier-weighted symmetric win/loss for non-boss tiers ---
        reward += self._tier_outcome_reward(self._last_obs_raw, terminated, truncated)
        # --- R_hp_efficiency §6.2: HP preserve bonus on non-boss win ---
        reward += self._hp_preserve_win_bonus(self._last_obs_raw, terminated, truncated)
        # --- Boss terminal shaping (preserved �?boss has its own signal) ---
        reward += self._boss_terminal_reward(prev_obs, self._last_obs_raw, terminated, truncated)
        # --- R_mechanic_quality: kaiser/ceremonial (preserved) ---
        reward += self._boss_mechanic_reward(prev_obs, self._last_obs_raw, legal_action)
        sentinel_terminal = self._sentinel_terminal_reward(
            prev_obs, self._last_obs_raw, terminated, truncated
        )
        reward += sentinel_terminal
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
            from .hp_cost_safety import hp_cost_safety_view  # noqa: WPS433
            from .x_cost_dynamic import x_cost_view  # noqa: WPS433
            from .selection_typed import selection_view  # noqa: WPS433
            from .card_identity import card_identity  # noqa: WPS433
            from .card_runtime_state import (  # noqa: WPS433
                card_runtime_presence_flags,
                card_runtime_state,
            )
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

    def action_masks(self) -> np.ndarray:
        from .hp_cost_safety import is_self_lethal_action  # noqa: WPS433

        mask = np.zeros(MAX_ACTIONS, dtype=bool)
        raw_obs = self._last_obs_raw if isinstance(self._last_obs_raw, dict) else None
        for i, action in enumerate(self._legal_actions[:MAX_ACTIONS]):
            if not isinstance(action, dict):
                continue
            # P0-1: hard-mask any action whose unblockable HP cost would
            # kill the player at resolve time.  Block does not soak
            # ``cardHpLoss`` / ``nonCardHpLoss`` per STS2 source so the
            # mask never relies on current block to "save" the action.
            if is_self_lethal_action(action, raw_obs):
                continue
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
        if self._human_demo_recorder is not None:
            try:
                self._human_demo_recorder.close()
            except Exception:
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
        self._decorate_sandbox_raw_obs_context()
        self._last_action_overflow = max(len(self._legal_actions) - MAX_ACTIONS, 0)

    def _decorate_sandbox_raw_obs_context(self) -> None:
        """Expose injected combat-sandbox context without rewriting bridge state.

        Combat sandbox episodes run from injected pre-combat snapshots.  The
        live bridge observation may still report a synthetic/low ``run.floor``,
        while the snapshot carries the real full-run floor and sample identity.
        Tactical guards must be able to see that context, but other consumers
        still need the raw bridge floor.  Therefore we only add explicit
        ``snapshot_*`` fields and never overwrite ``run.floor``.
        """

        if not isinstance(self._last_obs_raw, dict):
            return

        if self._current_encounter_id:
            self._last_obs_raw.setdefault("encounter_id", self._current_encounter_id)
            self._last_obs_raw.setdefault("snapshot_encounter_id", self._current_encounter_id)

        snap = self._current_snapshot if isinstance(self._current_snapshot, dict) else {}
        floor = snap.get("floor_number")
        if floor is not None:
            self._last_obs_raw.setdefault("snapshot_floor_number", floor)
            run = self._last_obs_raw.setdefault("run", {})
            if isinstance(run, dict):
                run.setdefault("snapshot_floor_number", floor)

        sample_id = snap.get("sample_id")
        if sample_id is not None:
            self._last_obs_raw.setdefault("snapshot_sample_id", sample_id)

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

    def _current_encounter_tier(self) -> str:
        encounter_id = str(self._current_encounter_id or "").lower()
        if "boss" in encounter_id:
            return "boss"
        if "elite" in encounter_id:
            return "elite"
        if "weak" in encounter_id:
            return "weak"
        return "normal" if encounter_id else "unknown"

    @staticmethod
    def _action_family(action: dict[str, Any] | None) -> str:
        if not isinstance(action, dict):
            return ""
        semantic = action.get("semantic") if isinstance(action.get("semantic"), dict) else {}
        for key in ("family", "action_kind", "kind", "type"):
            value = semantic.get(key) if key in semantic else action.get(key)
            if value:
                return str(value).strip().lower()
        action_id = str(action.get("action_id") or "").lower()
        if "potion" in action_id:
            return "use_potion"
        return ""

    @staticmethod
    def _is_stable_actionability(actionability: dict[str, Any] | None) -> bool:
        """Frontier is stable when bridge says non-end-turn actions exist OR
        the only-end-turn frame is genuinely settled (not transient).

        Mirrors the C1 bridge payload contract: returning True means the
        Python side can stop short-polling and accept the current frontier.
        """
        if not isinstance(actionability, dict):
            return True  # No actionability info → assume stable to avoid hangs.
        if int(actionability.get("legal_non_end_turn_count", 0) or 0) > 0:
            return True
        if bool(actionability.get("transient_only_end_turn", False)):
            return False
        # Only end_turn AND not transient → ``stable_no_actions``.
        return bool(actionability.get("frontier_stable", True))

    @staticmethod
    def wait_for_stable_actionability(
        initial_result: dict[str, Any],
        observe_fn: Callable[[], dict[str, Any]],
        *,
        max_wait_ms: int = 100,
        poll_interval_ms: int = 10,
        sleep_fn: Callable[[float], None] | None = None,
        clock_fn: Callable[[], float] | None = None,
    ) -> tuple[dict[str, Any], dict[str, Any]]:
        """Short-poll the bridge until the post-step frontier is stable.

        Pure helper so a fake bridge can drive it in tests.  The function
        returns ``(final_result, fast_step_metrics)`` where metrics contains
        ``wait_ms``, ``poll_count``, ``timeout``, ``transient_resolved``,
        ``transient_leaked``, ``stable_no_actions``.
        """

        sleep = sleep_fn if sleep_fn is not None else time.sleep
        clock = clock_fn if clock_fn is not None else time.perf_counter
        info0 = initial_result.get("info") if isinstance(initial_result.get("info"), dict) else {}
        actionability0 = info0.get("actionability") if isinstance(info0.get("actionability"), dict) else None
        # Stable on first frame: no waiting at all.
        metrics: dict[str, Any] = {
            "wait_ms": 0.0,
            "poll_count": 0,
            "timeout": False,
            "transient_resolved": False,
            "transient_leaked": False,
            "stable_no_actions": False,
        }
        if not bool((actionability0 or {}).get("transient_only_end_turn", False)):
            if (
                isinstance(actionability0, dict)
                and int(actionability0.get("legal_non_end_turn_count", 0) or 0) == 0
                and not bool(actionability0.get("transient_only_end_turn", False))
            ):
                metrics["stable_no_actions"] = True
            return initial_result, metrics

        deadline = clock() + (max_wait_ms / 1000.0)
        interval_s = max(poll_interval_ms / 1000.0, 0.001)
        current = initial_result
        while clock() < deadline:
            sleep(interval_s)
            metrics["poll_count"] += 1
            try:
                observed = observe_fn()
            except Exception:
                break
            if not isinstance(observed, dict):
                continue
            current = observed
            obs_info = observed.get("info") if isinstance(observed.get("info"), dict) else {}
            obs_actionability = obs_info.get("actionability") if isinstance(obs_info.get("actionability"), dict) else None
            if CombatSandboxEnv._is_stable_actionability(obs_actionability):
                metrics["wait_ms"] = max((max_wait_ms / 1000.0 - max(deadline - clock(), 0.0)) * 1000.0, 0.0)
                metrics["transient_resolved"] = (
                    int((obs_actionability or {}).get("legal_non_end_turn_count", 0) or 0) > 0
                )
                metrics["stable_no_actions"] = not metrics["transient_resolved"]
                return current, metrics
        metrics["wait_ms"] = float(max_wait_ms)
        metrics["timeout"] = True
        metrics["transient_leaked"] = True
        return current, metrics

    def _fast_step_metric_inc(self, key: str, amount: int = 1) -> None:
        metrics = getattr(self, "_fast_step_metrics_total", None)
        if not isinstance(metrics, dict):
            return
        try:
            metrics[key] = int(metrics.get(key, 0) or 0) + int(amount)
        except Exception:
            metrics[key] = int(amount)

    @staticmethod
    def _is_end_turn_action(action: dict[str, Any] | None) -> bool:
        if not isinstance(action, dict):
            return False
        action_id = str(action.get("action_id") or "").strip().lower()
        kind = str(action.get("kind") or "").strip().lower()
        family = ""
        semantic = action.get("semantic") if isinstance(action.get("semantic"), dict) else {}
        if isinstance(semantic, dict):
            family = str(semantic.get("family") or semantic.get("action_kind") or "").strip().lower()
        return action_id == "end_turn" or kind == "end_turn" or family == "end_turn"

    @staticmethod
    def _result_obs(result: dict[str, Any] | None) -> dict[str, Any]:
        obs = result.get("obs") if isinstance(result, dict) else None
        return obs if isinstance(obs, dict) else {}

    @staticmethod
    def _result_actionability(result: dict[str, Any] | None) -> dict[str, Any] | None:
        info = result.get("info") if isinstance(result, dict) and isinstance(result.get("info"), dict) else {}
        actionability = info.get("actionability") if isinstance(info.get("actionability"), dict) else None
        return actionability if isinstance(actionability, dict) else None

    @staticmethod
    def _filtered_result_legal_actions(result: dict[str, Any] | None) -> list[dict[str, Any]]:
        legal_actions = result.get("legal_actions") if isinstance(result, dict) else None
        if not isinstance(legal_actions, list):
            return []
        return [
            action
            for action in legal_actions
            if isinstance(action, dict)
            and str(action.get("kind") or "").strip() not in BLOCKED_ACTION_KINDS
        ]

    def _result_is_singleton_end_turn(self, result: dict[str, Any] | None) -> bool:
        actions = self._filtered_result_legal_actions(result)
        return len(actions) == 1 and self._is_end_turn_action(actions[0])

    @staticmethod
    def _obs_energy_hand_summary(obs: dict[str, Any] | None) -> dict[str, Any]:
        obs = obs if isinstance(obs, dict) else {}
        combat = obs.get("combat") if isinstance(obs.get("combat"), dict) else {}
        player = obs.get("player") if isinstance(obs.get("player"), dict) else {}

        combat_active = bool(combat)
        if isinstance(combat, dict) and combat.get("in_progress") is False:
            combat_active = False

        energy = _float(combat.get("energy") if isinstance(combat, dict) else None, -1.0)
        if energy < 0.0:
            energy = _float(player.get("energy") if isinstance(player, dict) else None, 0.0)
        elif energy <= 0.0 and isinstance(player, dict):
            # Some bridge payloads keep the authoritative energy on player.
            energy = max(energy, _float(player.get("energy"), 0.0))

        hand_missing = True
        hand_count = 0
        hand = combat.get("hand") if isinstance(combat, dict) else None
        if not isinstance(hand, list) and isinstance(player, dict):
            hand = player.get("hand")
        if isinstance(hand, list):
            hand_missing = False
            hand_count = len(hand)
        else:
            for container in (combat, player):
                if not isinstance(container, dict):
                    continue
                for key in ("hand_count", "num_cards_in_hand", "cards_in_hand"):
                    if key in container:
                        hand_missing = False
                        hand_count = int(max(_float(container.get(key), 0.0), 0.0))
                        break
                if not hand_missing:
                    break

        return {
            "combat_active": bool(combat_active),
            "energy": float(energy),
            "hand_count": int(hand_count),
            "hand_missing": bool(hand_missing),
            "round": combat.get("round") if isinstance(combat, dict) else None,
            "turn": combat.get("turn") if isinstance(combat, dict) else None,
            "hp": player.get("hp") if isinstance(player, dict) else None,
            "block": player.get("block") if isinstance(player, dict) else None,
        }

    def _result_needs_post_step_frontier_wait(self, result: dict[str, Any] | None) -> tuple[bool, str, dict[str, Any]]:
        if getattr(self, "_fast_step_disabled", False):
            return False, "disabled", {}
        if not isinstance(result, dict):
            return False, "non_dict_result", {}
        if bool(result.get("done", False)) or bool(result.get("truncated", False)):
            return False, "terminal_result", {}
        if not self._result_is_singleton_end_turn(result):
            return False, "not_singleton_end_turn", {}

        obs_summary = self._obs_energy_hand_summary(self._result_obs(result))
        actionability = self._result_actionability(result)
        if isinstance(actionability, dict):
            if bool(actionability.get("transient_only_end_turn", False)):
                return True, "bridge_transient_only_end_turn", obs_summary
            try:
                non_end_turn = int(actionability.get("legal_non_end_turn_count", 0) or 0)
            except (TypeError, ValueError):
                non_end_turn = 0
            if non_end_turn > 0:
                return True, "bridge_non_end_turn_count_mismatch", obs_summary
            if bool(actionability.get("frontier_stable", True)) is False:
                return True, "bridge_frontier_unstable", obs_summary
            # Bridge explicitly says: only EndTurn and stable.  This covers
            # the user's counterexample: the player may have genuinely played
            # all playable cards while retaining unused energy.
            return False, "bridge_stable_singleton_end_turn", obs_summary

        if (
            bool(obs_summary.get("combat_active"))
            and float(obs_summary.get("energy") or 0.0) > 0.0
            and (
                bool(obs_summary.get("hand_missing"))
                or int(obs_summary.get("hand_count") or 0) == 0
            )
        ):
            return True, "energy_positive_hand_empty_or_missing", obs_summary

        return False, "ambiguous_singleton_end_turn", obs_summary

    @staticmethod
    def _compact_frontier_action(action: dict[str, Any], idx: int) -> dict[str, Any]:
        card = action.get("card") if isinstance(action.get("card"), dict) else {}
        potion = action.get("potion") if isinstance(action.get("potion"), dict) else {}
        title = (
            action.get("title")
            or action.get("name")
            or card.get("title")
            or card.get("name")
            or potion.get("title")
            or potion.get("name")
        )
        return {
            "idx": int(idx),
            "action_id": action.get("action_id"),
            "kind": action.get("kind"),
            "title": title,
        }

    def _frontier_result_summary(self, result: dict[str, Any] | None, *, source: str) -> dict[str, Any]:
        actions = self._filtered_result_legal_actions(result)
        actionability = self._result_actionability(result)
        obs_summary = self._obs_energy_hand_summary(self._result_obs(result))
        return {
            "source": source,
            "legal_action_count": int(len(actions)),
            "non_end_turn_count": int(sum(1 for action in actions if not self._is_end_turn_action(action))),
            "singleton_end_turn": bool(len(actions) == 1 and self._is_end_turn_action(actions[0])),
            "actions": [self._compact_frontier_action(action, idx) for idx, action in enumerate(actions[:6])],
            "obs": obs_summary,
            "actionability": {
                "transient_only_end_turn": bool((actionability or {}).get("transient_only_end_turn", False)),
                "frontier_stable": (actionability or {}).get("frontier_stable"),
                "legal_non_end_turn_count": (actionability or {}).get("legal_non_end_turn_count"),
            } if isinstance(actionability, dict) else None,
        }

    @staticmethod
    def _state_available_actions(state: dict[str, Any] | None) -> list[dict[str, Any]]:
        if not isinstance(state, dict):
            return []
        actions = state.get("available_actions")
        if not isinstance(actions, list):
            actions = state.get("legal_actions")
        if not isinstance(actions, list):
            return []
        return [action for action in actions if isinstance(action, dict)]

    @staticmethod
    def _state_action_is_blocked(action: dict[str, Any]) -> bool:
        return str(action.get("kind") or "").strip() in BLOCKED_ACTION_KINDS

    def _state_non_end_turn_unblocked_action_count(self, state: dict[str, Any] | None) -> int:
        return sum(
            1
            for action in self._state_available_actions(state)
            if not self._state_action_is_blocked(action) and not self._is_end_turn_action(action)
        )

    def _frontier_state_summary(self, state: dict[str, Any] | None, *, source: str) -> dict[str, Any]:
        actions = [
            action
            for action in self._state_available_actions(state)
            if not self._state_action_is_blocked(action)
        ]
        obs_summary = self._obs_energy_hand_summary(state if isinstance(state, dict) else None)
        return {
            "source": source,
            "screen": state.get("screen") if isinstance(state, dict) else None,
            "phase": state.get("phase") if isinstance(state, dict) else None,
            "legal_action_count": int(len(actions)),
            "non_end_turn_count": int(sum(1 for action in actions if not self._is_end_turn_action(action))),
            "singleton_end_turn": bool(len(actions) == 1 and self._is_end_turn_action(actions[0])),
            "actions": [self._compact_frontier_action(action, idx) for idx, action in enumerate(actions[:6])],
            "obs": obs_summary,
        }

    def _safe_get_state(self) -> dict[str, Any] | None:
        try:
            state = self.bridge.get_state()
        except Exception:
            return None
        return state if isinstance(state, dict) else None

    @staticmethod
    def _state_allows_soft_rebind(state: dict[str, Any] | None) -> bool:
        if not isinstance(state, dict):
            return False
        phase = str(state.get("phase") or "").strip().lower()
        if phase.startswith("startup_") or phase == "terminal":
            return False
        screen = str(state.get("screen") or "").strip().upper()
        if screen in {"MAIN_MENU", "TITLE_SCREEN"}:
            return False
        run = state.get("run")
        if isinstance(run, dict):
            if run.get("game_over") is True or run.get("is_game_over") is True:
                return False
            active = run.get("active")
            if active is not None:
                return bool(active)
        return True

    def _safe_reset_into_current_run(self, timeout_ms: int) -> dict[str, Any] | None:
        try:
            result = self.bridge.reset(
                rebind_active_run=True,
                timeout_ms=max(1, int(timeout_ms)),
            )
        except Exception:
            return None
        return result if isinstance(result, dict) else None

    def _merge_frontier_rebind_result(
        self,
        base_result: dict[str, Any],
        rebound_result: dict[str, Any],
    ) -> dict[str, Any]:
        """Replace only the post-action frontier, not the action outcome.

        ``bridge.step`` owns reward/done/truncated for the executed action.
        ``bridge.reset(rebind_active_run=True)`` is only used as an observe /
        soft-rebind mechanism, so copying it wholesale would silently zero or
        distort the transition reward.  Keep the step outcome and refresh only
        obs/legal_actions/episode_id/info.
        """

        merged = dict(base_result)
        for key in ("obs", "legal_actions", "episode_id"):
            if key in rebound_result:
                merged[key] = rebound_result[key]

        base_info = base_result.get("info") if isinstance(base_result.get("info"), dict) else {}
        rebound_info = rebound_result.get("info") if isinstance(rebound_result.get("info"), dict) else {}
        info = dict(base_info)
        if rebound_info:
            info["frontier_rebind_info"] = rebound_info
        rebound_actionability = rebound_info.get("actionability") if isinstance(rebound_info.get("actionability"), dict) else None
        if isinstance(rebound_actionability, dict):
            info["actionability"] = rebound_actionability
        else:
            refreshed_actions = self._filtered_result_legal_actions(rebound_result)
            info["actionability"] = {
                "transient_only_end_turn": False,
                "frontier_stable": True,
                "legal_non_end_turn_count": int(
                    sum(1 for action in refreshed_actions if not self._is_end_turn_action(action))
                ),
                "source": "combat_env_post_step_rebind",
            }
        info["post_step_frontier_rebound"] = True
        merged["info"] = info
        return merged

    def _recover_post_step_frontier(
        self,
        result: dict[str, Any],
        *,
        selected_action: dict[str, Any] | None = None,
        legal_actions_before: list[dict[str, Any]] | None = None,
    ) -> tuple[dict[str, Any], dict[str, Any]]:
        """Short-poll singleton EndTurn frontiers after bridge.step.

        This is the production version of the transient-only-EndTurn check.
        It directly addresses the ambiguous case raised by the user:

        * if short polling rebounds into non-EndTurn actions, the immediate
          only-EndTurn frame was transient and we soft-rebind the frontier;
        * if no rebound appears and bridge did not label the frame transient,
          we classify it as stable/no-actions rather than a bug;
        * if the policy had non-EndTurn actions before selecting End Turn,
          that remains a separate policy-quality issue and is not conflated
          with hand-draw/frontier settling.
        """

        attempted, reason, obs_summary = self._result_needs_post_step_frontier_wait(result)
        metrics: dict[str, Any] = {
            "attempted": bool(attempted),
            "reason": str(reason),
            "wait_ms": 0.0,
            "poll_count": 0,
            "resolved": False,
            "timeout": False,
            "leaked": False,
            "stable_no_actions": False,
            "rebind_attempted": False,
            "rebind_succeeded": False,
            "suspicious_singleton": bool(self._result_is_singleton_end_turn(result)),
            "selected_action_id": selected_action.get("action_id") if isinstance(selected_action, dict) else None,
            "selected_action_kind": selected_action.get("kind") if isinstance(selected_action, dict) else None,
            "pre_legal_action_count": len(legal_actions_before) if isinstance(legal_actions_before, list) else None,
            "obs_summary": obs_summary,
        }
        if not attempted:
            if reason == "bridge_stable_singleton_end_turn":
                metrics["stable_no_actions"] = True
                self._fast_step_metric_inc("post_step_frontier_stable_no_actions_count")
            return result, metrics

        self._fast_step_metric_inc("post_step_frontier_attempt_count")
        self._fast_step_metric_inc("post_step_frontier_suspicious_singleton_count")
        trace: dict[str, Any] = {
            "event": "post_step_frontier_trace",
            "reason": str(reason),
            "selected_action": self._compact_frontier_action(selected_action, -1)
            if isinstance(selected_action, dict)
            else None,
            "pre_legal_action_count": len(legal_actions_before) if isinstance(legal_actions_before, list) else None,
            "immediate": self._frontier_result_summary(result, source="immediate_step_result"),
            "polls": [],
        }
        metrics["trace"] = trace

        max_wait_ms = max(0, int(getattr(self, "_fast_step_max_wait_ms", 100) or 0))
        poll_interval_ms = max(1, int(getattr(self, "_fast_step_poll_interval_ms", 10) or 10))
        started = time.perf_counter()
        deadline = started + (max_wait_ms / 1000.0)
        hard_transient_reason = reason in {
            "bridge_transient_only_end_turn",
            "bridge_frontier_unstable",
            "bridge_non_end_turn_count_mismatch",
        }

        while time.perf_counter() < deadline:
            time.sleep(poll_interval_ms / 1000.0)
            metrics["poll_count"] = int(metrics["poll_count"]) + 1
            state = self._safe_get_state()
            state_summary = self._frontier_state_summary(state, source="state_poll")
            if len(trace["polls"]) < 8:
                trace["polls"].append(state_summary)
            non_end_turn_count = int(state_summary.get("non_end_turn_count") or 0)
            if non_end_turn_count <= 0:
                continue
            if not self._state_allows_soft_rebind(state):
                continue

            metrics["rebind_attempted"] = True
            self._fast_step_metric_inc("post_step_frontier_rebind_attempt_count")
            remaining_ms = max(int((deadline - time.perf_counter()) * 1000.0), 1)
            rebind_timeout_ms = max(
                250,
                min(remaining_ms, int(getattr(self, "reset_timeout_ms", 1000) or 1000)),
            )
            rebound = self._safe_reset_into_current_run(rebind_timeout_ms)
            if not isinstance(rebound, dict):
                continue
            rebound_summary = self._frontier_result_summary(rebound, source="soft_rebind_result")
            trace["rebind"] = rebound_summary
            rebound_actions = self._filtered_result_legal_actions(rebound)
            rebound_non_end_turn = sum(1 for action in rebound_actions if not self._is_end_turn_action(action))
            if rebound_actions and rebound_non_end_turn > 0:
                merged = self._merge_frontier_rebind_result(result, rebound)
                if "episode_id" in rebound:
                    self._episode_id = rebound.get("episode_id", getattr(self, "_episode_id", None))
                metrics["wait_ms"] = max((time.perf_counter() - started) * 1000.0, 0.0)
                metrics["resolved"] = True
                metrics["rebind_succeeded"] = True
                trace["final_status"] = "resolved_non_end_turn_rebound"
                self._fast_step_metric_inc("post_step_frontier_resolved_count")
                self._fast_step_metric_inc("post_step_frontier_rebind_success_count")
                self._fast_step_metric_inc("transient_resolved_count")
                return merged, metrics

        metrics["wait_ms"] = max((time.perf_counter() - started) * 1000.0, float(max_wait_ms))
        metrics["timeout"] = True
        self._fast_step_metric_inc("post_step_frontier_timeout_count")
        self._fast_step_metric_inc("wait_timeout_count")
        if hard_transient_reason:
            metrics["leaked"] = True
            trace["final_status"] = "timeout_after_bridge_transient"
            self._fast_step_metric_inc("post_step_frontier_leaked_count")
        else:
            # Important: leftover energy + empty/missing hand alone is not a
            # proof of bug.  It may simply mean all playable cards were used
            # and the player has unspent energy.  Without a rebound, mark it
            # stable/no-actions so downstream analysis does not overclaim.
            metrics["stable_no_actions"] = True
            trace["final_status"] = "no_rebound_observed_stable_singleton"
            self._fast_step_metric_inc("post_step_frontier_stable_no_actions_count")
        return result, metrics

    @staticmethod
    def _potion_slots_dump(obs: dict[str, Any] | None) -> list[dict[str, Any]]:
        """Return a normalised view of every potion slot for diagnostic dumps."""
        player = (obs or {}).get("player") if isinstance(obs, dict) else {}
        potions = player.get("potions") if isinstance(player, dict) else None
        if not isinstance(potions, list):
            return []
        EMPTY_NAMES = {"empty", "[empty]", "none", "null", ""}
        out: list[dict[str, Any]] = []
        for idx, potion in enumerate(potions):
            if isinstance(potion, dict):
                name = str(potion.get("name") or potion.get("id") or potion.get("title") or "").strip().lower()
                empty = bool(potion.get("empty")) or (not name) or (name in EMPTY_NAMES)
                out.append({
                    "slot": idx,
                    "id": potion.get("id"),
                    "title": potion.get("title") or potion.get("name"),
                    "empty": empty,
                    "is_usable": bool(potion.get("is_usable", not empty)),
                    "is_queued": bool(potion.get("is_queued", False)),
                })
            else:
                name = str(potion or "").strip().lower()
                out.append({
                    "slot": idx,
                    "id": None,
                    "title": str(potion) if potion is not None else None,
                    "empty": (not name) or (name in EMPTY_NAMES),
                    "is_usable": False,
                    "is_queued": False,
                })
        return out

    def _build_potion_transition_record(
        self,
        *,
        action: dict[str, Any] | None,
        prev_obs: dict[str, Any] | None,
        after_obs: dict[str, Any] | None,
        bridge_result: dict[str, Any] | None,
        bridge_error: str | None,
    ) -> dict[str, Any] | None:
        """Assemble a use_potion transition record for the diagnostics JSONL.

        Returns None for non-potion actions.  When ``bridge_error`` is set the
        record is still returned so we can post-mortem failed potion uses.
        """
        if self._action_family(action) not in {"use_potion", "potion"}:
            return None
        result = bridge_result if isinstance(bridge_result, dict) else {}
        info_block = result.get("info") if isinstance(result.get("info"), dict) else {}
        execute_ok = bridge_error is None and not bool(info_block.get("error"))
        # State versions: bridge serialises an integer state version per result; if
        # missing, fall back to obs.state_version from the raw frame.
        def _state_version(obs: dict[str, Any] | None) -> int:
            if not isinstance(obs, dict):
                return 0
            for container in (obs.get("meta"), obs):
                if not isinstance(container, dict):
                    continue
                for key in ("state_version", "stateVersion"):
                    if key in container:
                        try:
                            return int(container.get(key) or 0)
                        except (TypeError, ValueError):
                            return 0
            return 0
        slot_index = -1
        target_block = action.get("target") if isinstance(action.get("target"), dict) else {}
        for key in ("slot_index", "potion_slot", "slot"):
            for source in (action, target_block, action.get("potion") if isinstance(action.get("potion"), dict) else {}):
                if isinstance(source, dict) and key in source:
                    try:
                        slot_index = int(source.get(key))
                        break
                    except (TypeError, ValueError):
                        continue
            if slot_index >= 0:
                break
        potion_block = action.get("potion") if isinstance(action.get("potion"), dict) else {}
        before_dump = self._potion_slots_dump(prev_obs)
        after_dump = self._potion_slots_dump(after_obs)
        before_slot = next((slot for slot in before_dump if slot.get("slot") == slot_index), None)
        after_slot = next((slot for slot in after_dump if slot.get("slot") == slot_index), None)
        return {
            "event": "use_potion_transition",
            "action_id": action.get("action_id"),
            "potion_slot": slot_index,
            "potion_id_before": (before_slot or {}).get("id") if before_slot else potion_block.get("id"),
            "potion_title_before": (before_slot or {}).get("title") if before_slot else potion_block.get("title"),
            "execute_ok": bool(execute_ok),
            "bridge_error": bridge_error,
            "state_version_before": _state_version(prev_obs),
            "state_version_after": _state_version(after_obs),
            "potion_slot_after": after_slot,
            "potion_slots_after": after_dump,
        }

    @staticmethod
    def _nonempty_potion_count(obs: dict[str, Any] | None) -> int:
        player = (obs or {}).get("player") if isinstance(obs, dict) else {}
        potions = player.get("potions") if isinstance(player, dict) else None
        if not isinstance(potions, list):
            return 0
        count = 0
        # STS2 bridge serializes empty potion slots as title="[empty]" — note
        # the BRACKETS.  An earlier exclusion set of {"empty","none","null"}
        # missed that form, so every empty slot was counted as a usable
        # potion.  That misled the potion-hoarding terminal reward and the
        # potion-use diagnostics.
        EMPTY_NAMES = {"empty", "[empty]", "none", "null", ""}
        for potion in potions:
            if not potion:
                continue
            if isinstance(potion, dict):
                if bool(potion.get("empty")):
                    continue
                name = str(potion.get("name") or potion.get("id") or potion.get("title") or "").strip().lower()
                if name and name not in EMPTY_NAMES:
                    count += 1
            else:
                name = str(potion or "").strip().lower()
                if name and name not in EMPTY_NAMES:
                    count += 1
        return count

    def _encounter_potion_use_reward(self, action: dict[str, Any] | None) -> float:
        if self._action_family(action) not in {"use_potion", "potion"}:
            return 0.0
        tier = self._current_encounter_tier()
        if tier == "boss":
            return float(POTION_USE_BOSS_BONUS)
        if tier == "elite":
            return float(POTION_USE_ELITE_BONUS)
        return float(POTION_USE_MONSTER_BONUS + POTION_USE_MONSTER_PENALTY)

    def _potion_hoarding_terminal_reward(self, after_obs: dict[str, Any] | None, terminated: bool, truncated: bool) -> float:
        if not (terminated or truncated):
            return 0.0
        unused = self._nonempty_potion_count(after_obs)
        if unused <= 0:
            return 0.0
        raw = float(POTION_HOARDING_PENALTY_PER_POTION) * float(unused)
        return float(np.clip(raw, -abs(float(POTION_HOARDING_MAX_PENALTY_ABS)), abs(float(POTION_HOARDING_MAX_PENALTY_ABS))))

    def _potion_timing_step_reward(
        self,
        action: dict[str, Any] | None,
        before_obs: dict[str, Any] | None,
        legal_actions: list[Any] | None,
    ) -> float:
        """Phase 4b of docs/potion-timing-modeling-plan.md (§2).

        Convert the timing profile (use_quality / waste_risk) into per-step
        reward shaping so the policy gradient actually learns "don't dump
        potions turn 1".  Penalty > bonus by design — model should prefer
        hoarding over bad use.
        """
        if self._action_family(action) not in {"use_potion", "potion"}:
            return 0.0
        if not isinstance(before_obs, dict):
            return 0.0
        try:
            energy = float(((before_obs.get("player") or {}).get("energy")) or 0.0)
        except (TypeError, ValueError):
            energy = 0.0
        encounter_tier = self._current_encounter_tier()
        try:
            profile = compute_potion_timing(
                action,
                before_obs,
                legal_actions,
                None,
                energy,
                encounter_tier=encounter_tier,
            )
        except Exception:
            return 0.0
        if not profile.get("is_potion"):
            return 0.0
        use_q = float(profile.get("use_quality") or 0.0)
        waste = float(profile.get("waste_risk") or 0.0)
        urgent = bool(
            profile.get("urgent")
            or profile.get("lethal")
            or profile.get("prevent_lethal")
            or profile.get("prevent_major_loss")
            or profile.get("mechanism_answer")
        )
        bad_timing = bool(
            profile.get("low_urgency")
            or profile.get("save_recommended")
            or profile.get("no_followup")
            or profile.get("block_waste")
            or profile.get("overkill")
        )
        reward = 0.0
        # The shared timing model has a small baseline use_quality.  Treating
        # any positive value as reward made the policy learn "use potion when
        # legal".  Reward only urgent / genuinely high-quality timing; convert
        # low-urgency, no-followup, overkill, or block-waste uses into waste.
        if urgent or (use_q >= 0.45 and not bad_timing):
            reward += float(POTION_TIMING_QUALITY_SCALE) * use_q
            self._potion_timing_quality_events += 1
        elif bad_timing:
            effective_waste = max(float(waste), 0.35)
            reward -= float(POTION_TIMING_WASTE_SCALE) * effective_waste
            self._potion_timing_waste_events += 1
        elif waste > 0.0:
            reward -= float(POTION_TIMING_WASTE_SCALE) * waste
            self._potion_timing_waste_events += 1
        return reward

    def _card_selection_step_reward(self, action: dict[str, Any] | None) -> float:
        """Anti-loop + early-confirm shaping for multi-pick burn cards.

        See §4 of docs/kaiser-and-potion-fixes-todo.md.

        Detects three failure modes:
          * Pick→replace cycle on SAME card (A→A): SELECTION_LOOP_PENALTY.
          * Oscillation across multiple cards (A→B→C→A): caught by
            SELECTION_DESELECT_PENALTY — each pick whose `is_selected=True`
            is a deselect, every deselect after the first pays the penalty.
          * Total-picks death loop: SELECTION_PICK_CAP=12 caps a single
            selection round; each pick beyond pays SELECTION_OVER_CAP_PENALTY.

        Tracks per-selection state in self._selection_last_picked_id /
        self._selection_flip_count / self._selection_pick_count /
        self._selection_deselect_count, which reset when the family
        transitions away from card_selection or on confirm/cancel/skip.
        """
        if not isinstance(action, dict):
            return 0.0
        family = self._action_family(action)
        sel_action = str(action.get("selection_action") or action.get("selection") or "").strip().lower()
        action_id = str(action.get("action_id") or "")

        def _reset_state() -> None:
            self._selection_last_picked_id = ""
            self._selection_flip_count = 0
            self._selection_pick_count = 0
            self._selection_deselect_count = 0

        # §4 v3: detect selection-screen entry/exit edges. Pay re-entry penalty
        # when the model bounces in and out of selection screens within the
        # same episode (the_insatiable frantic_escape pattern).
        reentry_penalty = 0.0
        if family == "card_selection" and not self._selection_screen_active:
            self._selection_screen_active = True
            self._selection_screen_entries += 1
            budget = int(SELECTION_REENTRY_BUDGET)
            if self._selection_screen_entries > budget:
                excess = self._selection_screen_entries - budget
                reentry_penalty = -float(SELECTION_REENTRY_PENALTY) * float(excess)
                self._selection_reentry_events += 1
        elif family != "card_selection" and self._selection_screen_active:
            self._selection_screen_active = False

        if family != "card_selection":
            if (self._selection_pick_count > 0
                or self._selection_flip_count > 0
                or self._selection_deselect_count > 0):
                _reset_state()
            return 0.0

        if sel_action == "confirm":
            picked = max(int(self._selection_pick_count), 0)
            max_picks = int(
                action.get("max_pick")
                or action.get("selection_max")
                or action.get("max_select")
                or action.get("max")
                or 0
            )
            if max_picks <= 0:
                max_picks = int(action.get("selection_pick_limit") or 0)
            reward = reentry_penalty
            if max_picks > 0 and picked < max_picks:
                ratio = float(max_picks - picked) / float(max_picks)
                reward += float(SELECTION_EARLY_CONFIRM_BONUS) * ratio
                self._selection_early_confirm_events += 1
            _reset_state()
            return reward

        if sel_action in {"cancel", "close", "skip"}:
            _reset_state()
            return reentry_penalty

        # Pick path: detect repeat-same, deselect-pattern, and over-cap.
        picked_id = ""
        card = action.get("card") if isinstance(action.get("card"), dict) else None
        if isinstance(card, dict):
            picked_id = str(card.get("id") or card.get("title") or "")
        if not picked_id and ":" in action_id:
            picked_id = action_id

        # Bridge marks `is_selected=True` on actions that toggle a card OFF
        # (the click would deselect it). Counting these directly catches
        # the A→B→A→B oscillation pattern that the legacy id-equality check
        # missed.
        raw_is_selected = action.get("is_selected")
        if isinstance(raw_is_selected, bool):
            is_deselect = raw_is_selected
        elif isinstance(raw_is_selected, (int, float)):
            is_deselect = float(raw_is_selected) != 0.0
        elif isinstance(raw_is_selected, str):
            is_deselect = raw_is_selected.strip().lower() in {"1", "true", "yes", "y", "on"}
        else:
            is_deselect = False

        reward = reentry_penalty
        # 1) Same-id repeat (legacy A→A→A check).
        if picked_id:
            if picked_id == self._selection_last_picked_id:
                self._selection_flip_count += 1
                if self._selection_flip_count >= 2:
                    reward -= float(SELECTION_LOOP_PENALTY) * float(self._selection_flip_count)
                    self._selection_loop_events += 1
            else:
                self._selection_flip_count = 0
            self._selection_last_picked_id = picked_id

        # 2) Deselect detection (oscillation across cards).
        if is_deselect:
            self._selection_deselect_count += 1
            if self._selection_deselect_count >= 2:
                # Linear escalation: 2nd deselect = -0.40, 3rd = -0.80, 4th = -1.20...
                reward -= float(SELECTION_DESELECT_PENALTY) * float(self._selection_deselect_count - 1)
                self._selection_loop_events += 1

        self._selection_pick_count += 1

        # 3) Hard pick-cap (kills the 1700-step death loop).
        if self._selection_pick_count > int(SELECTION_PICK_CAP):
            reward -= float(SELECTION_OVER_CAP_PENALTY)
            self._selection_over_cap_events += 1

        return reward


    @staticmethod
    def _boss_context_max(context: dict[str, Any], key: str) -> float:
        if not isinstance(context, dict):
            return 0.0
        vals: list[float] = []
        player_state = context.get("player_state") if isinstance(context.get("player_state"), dict) else {}
        if key in player_state:
            vals.append(_float(player_state.get(key)))
        enemy_states = context.get("enemy_states_by_index")
        if isinstance(enemy_states, list):
            vals.extend(_float((state or {}).get(key)) for state in enemy_states if isinstance(state, dict))
        return max(vals) if vals else 0.0

    @staticmethod
    def _action_semantic(action: dict[str, Any] | None) -> dict[str, Any]:
        return action.get("semantic") if isinstance(action, dict) and isinstance(action.get("semantic"), dict) else {}

    @classmethod
    def _action_roles(cls, action: dict[str, Any] | None) -> set[str]:
        semantic = cls._action_semantic(action)
        roles = semantic.get("roles")
        if not isinstance(roles, list):
            return set()
        return {str(role).strip().lower() for role in roles if str(role).strip()}

    @classmethod
    def _action_metric(cls, action: dict[str, Any] | None, key: str) -> float:
        semantic = cls._action_semantic(action)
        if key in semantic:
            return _float(semantic.get(key))
        if isinstance(action, dict):
            if key in action:
                return _float(action.get(key))
            card = action.get("card") if isinstance(action.get("card"), dict) else {}
            preview = card.get("preview") if isinstance(card.get("preview"), dict) else {}
            for source in (card, preview):
                if key in source:
                    return _float(source.get(key))
        return 0.0

    @classmethod
    def _action_immediate_impact(cls, action: dict[str, Any] | None) -> float:
        roles = cls._action_roles(action)
        damage = cls._action_metric(action, "damage")
        block = cls._action_metric(action, "block")
        hits = max(cls._action_metric(action, "hits"), 1.0 if damage > 0.0 else 0.0)
        debuff_bonus = 8.0 if roles.intersection({"debuff", "weak", "vulnerable", "poison", "exhaust", "discard"}) else 0.0
        scaling_bonus = 6.0 if roles.intersection({"scaling", "power", "draw", "energy", "retain"}) else 0.0
        return float(damage + 0.75 * block + 1.5 * max(hits - 1.0, 0.0) + debuff_bonus + scaling_bonus)

    def _boss_mechanic_reward(
        self,
        before_obs: dict[str, Any] | None,
        after_obs: dict[str, Any] | None,
        action: dict[str, Any] | None,
    ) -> float:
        """Dense tactical shaping for boss-only mechanics (Kaiser / Ceremonial / Knowledge Demon)."""
        if self._current_encounter_tier() != "boss":
            return 0.0
        encounter = str(self._current_encounter_id or "").lower()
        if not ("kaiser" in encounter or "ceremonial" in encounter or "knowledge_demon" in encounter):
            return 0.0
        try:
            before_ctx = build_boss_mechanics_context(before_obs)
            after_ctx = build_boss_mechanics_context(after_obs)
        except Exception:
            return 0.0

        reward = 0.0
        before_hp, _ = self._player_hp_and_max(before_obs)
        after_hp, _ = self._player_hp_and_max(after_obs)
        hp_loss = max(before_hp - after_hp, 0.0)
        enemy_hp_delta = max(self._combat_enemy_total_hp(before_obs) - self._combat_enemy_total_hp(after_obs), 0.0)
        family = self._action_family(action)
        roles = self._action_roles(action)
        impact = self._action_immediate_impact(action)

        if "kaiser" in encounter:
            # H25: switch the Kaiser branch to the PRIMARY-threat back-attack
            # signal.  Old code used max(...) across all enemies for risk —
            # but Kaiser has two parts both flagging back_attack_active=1
            # most turns, so the metric was pinned at 1.0 and facing_change
            # 1→0 detection never fired.  primary_back_attack_active reads
            # only the highest-intent-damage enemy's status, so when the
            # player correctly faces the high-damage attacker it flips 1→0
            # even if the low-damage attacker still has multiplier=1.5.
            before_primary = self._boss_context_max(before_ctx, "primary_back_attack_active")
            after_primary = self._boss_context_max(after_ctx, "primary_back_attack_active")
            before_risk = max(
                self._boss_context_max(before_ctx, "primary_back_attack_risk"),
                before_primary,
            )
            after_risk = max(
                self._boss_context_max(after_ctx, "primary_back_attack_risk"),
                after_primary,
            )
            # §12 soften factor: don't fully penalize if the agent had no
            # mechanically valid response available this frame.
            defense_candidates = self._boss_context_max(before_ctx, "kaiser_defense_candidate_count")
            facing_change_candidates = self._boss_context_max(before_ctx, "kaiser_facing_change_candidate_count")
            pressure_candidates = self._boss_context_max(before_ctx, "kaiser_pressure_candidate_count")
            no_response_avail = (
                defense_candidates < 0.5
                and facing_change_candidates < 0.5
                and pressure_candidates < 0.5
            )
            soften = float(KAISER_NO_RESPONSE_PENALTY_SOFTEN) if no_response_avail else 1.0

            # 2026-04-28 §1C: surface the *primary* (highest-intent-damage)
            # threat damage so we can scale facing bonus and high-threat
            # back-attack penalty by intent magnitude.
            primary_intent_dmg = self._boss_context_max(before_ctx, "primary_threat_intent_damage")

            if before_risk > 0.05:
                reward -= hp_loss * float(KAISER_BACK_ATTACK_HP_LOSS_PENALTY_SCALE) * (1.0 + before_risk) * soften
                # §1C: extra penalty if the threat we ignored was a HIGH-damage
                # attacker (≥ KAISER_BACK_ATTACK_HI_THREAT_DMG). Scaled by hp_loss/max_hp
                # so it stays balanced across boss HP variance.
                _, max_hp_back = self._player_hp_and_max(before_obs)
                if (
                    primary_intent_dmg >= float(KAISER_BACK_ATTACK_HI_THREAT_DMG)
                    and hp_loss > 0.0
                    and max_hp_back > 0.0
                ):
                    reward -= float(KAISER_BACK_ATTACK_HI_THREAT_EXTRA) * (hp_loss / max_hp_back) * soften
                if family == "end_turn":
                    reward += float(KAISER_BACK_ATTACK_END_TURN_PENALTY) * min(1.0, before_risk) * soften
                if family in {"play_card", "use_potion", "potion"} and (
                    "block" in roles or "debuff" in roles or "weak" in roles or self._action_metric(action, "block") > 0.0
                ):
                    reward += float(KAISER_BACK_ATTACK_DEFENSE_BONUS) * min(1.0, before_risk)
            risk_drop = max(before_risk - after_risk, 0.0)
            if risk_drop > 0.05:
                reward += float(KAISER_BACK_ATTACK_RISK_REDUCTION_BONUS) * min(1.0, risk_drop)

            # §12 missing positive signals �?facing change + pressure kill.
            # back_attack_active flipping from 1 �?0 means the player
            # successfully re-faced (took an action that turned the boss
            # so the back enemy is no longer active threat).  We only fire
            # this on play_card / use_potion (not end_turn).
            # Use primary-threat active flag (set above) so facing_change
            # detects "now correctly facing the high-damage attacker".
            facing_changed = before_primary > 0.5 and after_primary <= 0.5
            if facing_changed and family in {"play_card", "use_potion", "potion"}:
                # 2026-04-28 §1A: scale the facing bonus by the threat magnitude
                # the agent just faced. Refacing toward a 30 dmg attacker pays
                # 1.5x; refacing toward a 10 dmg one pays ~0.5x.  This kills
                # the failure mode where model would target the cheap claw
                # for "free" facing bonus.
                ref = max(float(KAISER_FACING_INTENT_DMG_REF), 1.0)
                threat_scale = float(np.clip(
                    primary_intent_dmg / ref, 0.0,
                    float(KAISER_FACING_INTENT_DMG_SCALE_MAX),
                ))
                bonus = float(KAISER_FACING_CHANGE_BONUS_BASE) * threat_scale
                # Cap with the legacy flat bonus to avoid over-shooting prior
                # calibration on tiny-intent dummy fights.
                bonus = min(bonus, float(KAISER_FACING_CHANGE_BONUS) * float(KAISER_FACING_INTENT_DMG_SCALE_MAX))
                reward += bonus
                self._kaiser_facing_change_count += 1

            # Pressure kill: dealt damage AND back-attack risk dropped
            # meaningfully in the same step (proxy for "killed the back side
            # part / enemy without re-facing").  Differentiated from facing
            # change: facing_changed=True covers refacing; pressure_kill is
            # the alternative win condition where you just out-DPS the back.
            if (
                not facing_changed
                and family == "play_card"
                and enemy_hp_delta > 5.0
                and risk_drop > 0.20
                and before_risk > 0.20
            ):
                reward += float(KAISER_PRESSURE_KILL_BONUS)
                self._kaiser_pressure_kill_count += 1

            # Lightweight stdout breadcrumb for visibility (every 25 new events).
            # Guard on count *change* — modulo would fire on every subsequent step
            # once the sum lands on a multiple of 25 and falsely look like a hang.
            current_total = self._kaiser_facing_change_count + self._kaiser_pressure_kill_count
            last_printed = getattr(self, "_kaiser_response_last_print_total", 0)
            if current_total > 0 and current_total != last_printed and current_total % 25 == 0:
                print(
                    f"[combat_env] kaiser_response facing_change={self._kaiser_facing_change_count} "
                    f"pressure_kill={self._kaiser_pressure_kill_count}",
                    flush=True,
                )
                self._kaiser_response_last_print_total = current_total

        if "ceremonial" in encounter:
            before_one = self._boss_context_max(before_ctx, "one_card_lock")
            after_one = self._boss_context_max(after_ctx, "one_card_lock")
            before_stun = self._boss_context_max(before_ctx, "stun_window")
            after_stun = self._boss_context_max(after_ctx, "stun_window")
            before_pending = self._boss_context_max(before_ctx, "transform_pending")
            after_threshold = self._boss_context_max(after_ctx, "threshold_active")
            if before_stun <= 0.05 and after_stun > 0.05:
                reward += float(CEREMONIAL_STUN_WINDOW_ENTER_BONUS)
            if before_pending > 0.05 and after_threshold > 0.05:
                reward += float(CEREMONIAL_THRESHOLD_PROGRESS_BONUS)
            if before_stun > 0.05 and enemy_hp_delta > 0.0:
                reward += min(0.75, enemy_hp_delta * float(CEREMONIAL_STUN_DAMAGE_MULTIPLIER))
            one_card_lock = max(before_one, after_one)
            if one_card_lock > 0.05:
                if family == "end_turn":
                    reward += float(CEREMONIAL_ONE_CARD_END_TURN_PENALTY)
                elif family in {"play_card", "use_potion", "potion"}:
                    if impact >= 12.0:
                        reward += float(CEREMONIAL_ONE_CARD_HIGH_IMPACT_BONUS)
                    elif impact <= 2.0 and not roles.intersection({"draw", "energy", "scaling", "power"}):
                        reward += float(CEREMONIAL_ONE_CARD_LOW_IMPACT_PENALTY)

        if "knowledge_demon" in encounter:
            # Knowledge Demon (知识恶魔) curse-selection shaping per user
            # strategy guidance:
            #   Curse 1 (no prior 瓦解 stack)  → prefer Option B "draw -1"
            #     (status / debuff card-selection, NOT damage_per_turn).
            #   Curse 2 (some 瓦解 stack)      → prefer Option A "+7 damage,
            #     blockable" (HP-loss easier to mitigate than max-3-cards).
            #   Curse 3 (heavy 瓦解 stack)     → prefer Option A
            #     (energy-loss is crippling).
            # We infer "which curse" from the cumulative 瓦解 / disintegrate
            # damage already on the player (read via _power_amount needles).
            # We detect the curse-selection event by checking the action's
            # surface/family/text keywords for damage-per-turn clauses.
            family_lc = family or ""
            action_text = ""
            if isinstance(action, dict):
                for key in ("title", "label", "name"):
                    val = action.get(key)
                    if val:
                        action_text += " " + str(val)
                card = action.get("card") if isinstance(action.get("card"), dict) else {}
                for key in ("title", "description", "effect", "canonical_text"):
                    val = card.get(key) if isinstance(card, dict) else None
                    if val:
                        action_text += " " + str(val)
            action_text_l = action_text.lower()
            is_curse_event = any(
                kw in action_text_l
                for kw in ("disintegrate", "瓦解", "card_selection:select", "event_option", "card_reward:skip")
            )
            picks_disintegrate = any(
                kw in action_text_l
                for kw in ("disintegrate", "瓦解", "受到 6 点", "受到 7 点", "受到 8 点", "每回合收到", "每回合受到")
            )
            picks_draw_loss = any(
                kw in action_text_l
                for kw in ("少抽 1 张", "少抽一张", "draw 1 fewer", "draw -1", "fewer card")
            )
            picks_play_cap = any(
                kw in action_text_l
                for kw in ("最多打出 3 张", "最多打 3 张", "max 3 cards", "play 3 cards")
            )
            picks_energy_loss = any(
                kw in action_text_l
                for kw in ("减少 1 点能量", "失去 1 点能量", "lose 1 energy", "-1 energy", "energy -1")
            )
            # Estimate which curse number we're choosing using the
            # current Disintegrate stack (see boss_mechanics if it's there;
            # fall back to scanning player_powers text).
            disintegrate_stack = 0.0
            before_player = before_obs.get("player") if isinstance(before_obs, dict) else None
            if isinstance(before_player, dict):
                powers = before_player.get("powers") if isinstance(before_player.get("powers"), list) else []
                for power in powers:
                    if not isinstance(power, dict):
                        continue
                    text = " ".join(
                        str(power.get(k) or "") for k in ("id", "title", "description")
                    ).lower()
                    if any(kw in text for kw in ("disintegrate", "瓦解")):
                        disintegrate_stack = max(disintegrate_stack, float(power.get("amount") or power.get("display_amount") or 0))
            curse_index = (
                1 if disintegrate_stack < 5.5
                else 2 if disintegrate_stack < 12.5
                else 3
            )

            if is_curse_event:
                if curse_index == 1:
                    # Prefer Option B (draw_loss).  A is the bad pick now.
                    if picks_draw_loss:
                        reward += float(KNOWLEDGE_DEMON_GOOD_CURSE_PICK_BONUS)
                    elif picks_disintegrate:
                        reward += float(KNOWLEDGE_DEMON_BAD_CURSE_PICK_PENALTY)
                elif curse_index == 2:
                    # Prefer Option A (disintegrate +7, blockable).
                    if picks_disintegrate:
                        reward += float(KNOWLEDGE_DEMON_GOOD_CURSE_PICK_BONUS)
                    elif picks_play_cap:
                        reward += float(KNOWLEDGE_DEMON_BAD_CURSE_PICK_PENALTY)
                else:  # curse_index == 3
                    # Prefer Option A (energy_loss is crippling).
                    if picks_disintegrate:
                        reward += float(KNOWLEDGE_DEMON_GOOD_CURSE_PICK_BONUS)
                    elif picks_energy_loss:
                        reward += float(KNOWLEDGE_DEMON_BAD_CURSE_PICK_PENALTY)
            # Speed-kill incentive: every end_turn lets the boss tick another
            # round of disintegrate + advance toward the next (worse) curse.
            if family_lc == "end_turn":
                reward += float(KNOWLEDGE_DEMON_END_TURN_PENALTY)

        return float(np.clip(reward, -1.25, 1.25))

    def _boss_terminal_reward(
        self,
        before_obs: dict[str, Any] | None,
        after_obs: dict[str, Any] | None,
        terminated: bool,
        truncated: bool,
    ) -> float:
        if self._current_encounter_tier() != "boss" or not (terminated or truncated):
            return 0.0
        after_hp, after_max_hp = self._player_hp_and_max(after_obs)
        before_hp, before_max_hp = self._player_hp_and_max(before_obs)
        max_hp = max(after_max_hp, before_max_hp, 1.0)
        if terminated and (not truncated) and after_hp > 0.0:
            return float(BOSS_COMBAT_WIN_BONUS_BASE + BOSS_COMBAT_WIN_BONUS_HP_SCALE * np.clip(after_hp / max_hp, 0.0, 1.0))
        missing_ratio = 1.0 - float(np.clip(max(after_hp, 0.0) / max_hp, 0.0, 1.0))
        # H22 v3 damage-undo (percent-based): on loss, undo the per-step
        # damage shaping that paid out during this episode by subtracting
        # BOSS_COMBAT_LOSS_DAMAGE_UNDO_PERCENT_SCALE × damage_dealt_ratio.
        # UNDO_SCALE (7.0) > per-step PERCENT_SCALE (5.0) so net damage
        # contribution is mildly negative even on a "deal everything but
        # die" loss, regardless of boss size.
        end_enemy_total = float(self._combat_enemy_total_hp(after_obs))
        base_hp = float(getattr(self, "_episode_start_boss_total_hp", 0.0) or 0.0)
        if base_hp > 0.0:
            damage_dealt_ratio = max(0.0, (base_hp - end_enemy_total) / base_hp)
            damage_undo = damage_dealt_ratio * float(BOSS_COMBAT_LOSS_DAMAGE_UNDO_PERCENT_SCALE)
        else:
            damage_undo = 0.0
        return -float(
            BOSS_COMBAT_LOSS_PENALTY_BASE
            + BOSS_COMBAT_LOSS_PENALTY_MISSING_HP_SCALE * missing_ratio
            + damage_undo
        )

    # ----- R_hp_efficiency §6.2 -----
    def _hp_preserve_win_bonus(
        self,
        after_obs: dict[str, Any] | None,
        terminated: bool,
        truncated: bool,
    ) -> float:
        """Terminal bonus for winning a non-boss combat with HP left.

        Uses sqrt(hp_end / max_hp) so the marginal value of each additional
        preserved HP tapers �?the first 20% preserved is worth more than
        the last 20%.  Boss tier gets zero weight because non-A10 bosses
        restore HP post-combat.
        """
        if not (terminated and not truncated):
            return 0.0
        after_hp, after_max_hp = self._player_hp_and_max(after_obs)
        if after_hp <= 0.0:
            return 0.0
        tier = self._current_encounter_tier()
        scale = float(HP_PRESERVE_WIN_BONUS_TIER_SCALE.get(tier, 0.0))
        if scale <= 0.0:
            return 0.0
        ratio = float(np.clip(after_hp / max(after_max_hp, 1.0), 0.0, 1.0))
        return scale * float(np.sqrt(ratio))

    # ----- R_turn_efficiency §8.2 -----
    def _turn_efficiency_penalty(self, action: dict[str, Any] | None) -> float:
        """Small tier-aware per-end_turn penalty to counter defend-forever."""
        if self._action_family(action) != "end_turn":
            return 0.0
        tier = self._current_encounter_tier()
        return float(TURN_EFFICIENCY_PENALTY_PER_END_TURN_TIER.get(tier, 0.0))

    # ----- R_outcome §5 �?tier-weighted outcome scaling -----
    def _tier_outcome_reward(
        self,
        after_obs: dict[str, Any] | None,
        terminated: bool,
        truncated: bool,
    ) -> float:
        """Symmetric win/loss bonus scaled by tier.

        Runs AFTER `_boss_terminal_reward` so boss-specific terminal shaping
        already carries its own magnitude; this function only supplements
        non-boss tiers (where there is no corresponding terminal bonus).
        Net effect: normal/elite wins and losses get a fixed ±(tier_scale)
        multiplier on top of the sparse bridge-side outcome reward.
        """
        if not (terminated or truncated):
            return 0.0
        tier = self._current_encounter_tier()
        if tier == "boss":
            # Boss already gets boss-specific terminal shaping; don't double-dip.
            return 0.0
        scale = float(OUTCOME_TIER_SCALE.get(tier, 1.0))
        if scale <= 0.0:
            return 0.0
        after_hp, _ = self._player_hp_and_max(after_obs)
        win = terminated and (not truncated) and after_hp > 0.0
        return (scale if win else -scale)

    # ----- Curriculum bookkeeping (§13) -----
    def _record_terminal_outcome_for_curriculum(
        self,
        after_obs: dict[str, Any] | None,
        terminated: bool,
        truncated: bool,
    ) -> None:
        """Feed win/loss into the CurriculumTracker and emit a one-line
        phase-switch annotation when the encounter crosses a boundary."""
        if not (terminated or truncated):
            return
        encounter = str(self._current_encounter_id or "").strip()
        if not encounter:
            return
        after_hp, _ = self._player_hp_and_max(after_obs)
        win = bool(terminated and (not truncated) and after_hp > 0.0)
        self._curriculum_tracker.record(encounter, win)
        switch = self._curriculum_tracker.check_phase_switch(encounter)
        if switch is not None:
            old_phase, new_phase = switch
            wr, n = self._curriculum_tracker.win_rate(encounter)
            print(
                f"[curriculum] encounter={encounter} phase {old_phase}->{new_phase} "
                f"win_rate_128={wr:.3f} n={n} (phase=P{new_phase})",
                flush=True,
            )
        # Periodic full-state dump so encounters that haven't crossed a phase
        # boundary still surface in the operator log.  Every 200 terminal
        # events feels right at ~3.6k steps/h × ~30 steps/ep �?120 ep/h �?
        # i.e. a dump every ~1.7h, slightly more often than the hourly cron.
        self._curriculum_episode_count += 1
        if self._curriculum_episode_count % 200 == 0:
            print(
                f"[curriculum/state] dump @ episodes={self._curriculum_episode_count}\n"
                + self._curriculum_tracker.dump_all_state(),
                flush=True,
            )

    def _enemy_hp_delta_reward(self, before_obs: dict[str, Any] | None, after_obs: dict[str, Any] | None) -> float:
        before_total = self._combat_enemy_total_hp(before_obs)
        after_total = self._combat_enemy_total_hp(after_obs)
        if before_total <= 0.0 and after_total <= 0.0:
            return 0.0

        # Guard against bridge clearing enemies list at terminal step when the
        # player DIED. Both live bridge mod and sim drop `combat.enemies` to
        # an empty list the moment the combat ends regardless of outcome �?
        # if we naively credit (before_total - 0) as "damage dealt", every
        # loss emits a positive shaping reward equal to the still-alive
        # enemies' total HP × 0.01. On a 566-HP terminal clear that's +5.66,
        # which drowns the bridge's loss penalty (-3.5 live, -1.0 sim) and
        # causes every short loss to be misclassified as a win under the
        # eval's `reward_sum > 0` heuristic. Skip the delta when the
        # after-state shows empty enemies AND the player is dead.
        after_enemies: Any = None
        after_player_hp = 0.0
        if isinstance(after_obs, dict):
            combat = after_obs.get("combat") if isinstance(after_obs.get("combat"), dict) else {}
            after_enemies = combat.get("enemies") if isinstance(combat, dict) else None
            player = after_obs.get("player") if isinstance(after_obs.get("player"), dict) else {}
            after_player_hp = _float((player or {}).get("hp"))
        # Both "enemies is missing key (sim: combat={in_progress:False})" and
        # "enemies is empty list (live: combat={...,enemies:[]})" are terminal
        # transitions. Catch both by checking after_total==0 (we already have
        # that via _combat_enemy_total_hp == 0 when enemies not-a-list) AND
        # player_hp<=0 (defeat).
        if (
            before_total > 0.0
            and after_total <= 0.0
            and after_player_hp <= 0.0
        ):
            return 0.0

        delta = before_total - after_total
        # Boss tier: percent-of-boss-HP based shaping (H22 v3).  Per-step reward
        # is the FRACTION of the boss's initial total HP killed this step times
        # BOSS_ENEMY_HP_DELTA_PERCENT_SCALE (5.0), so a full-boss kill across
        # the whole episode sums to +5.0 regardless of whether the boss is
        # 200 HP or 900 HP across phase changes.  Negative deltas (boss heal)
        # use the same percent-based scale.  Falls back to raw if we lost the
        # initial-HP snapshot (defensive).
        tier = self._current_encounter_tier()
        if tier == "boss":
            base_hp = float(getattr(self, "_episode_start_boss_total_hp", 0.0) or 0.0)
            if base_hp > 0.0:
                ratio = delta / base_hp
                raw = ratio * float(BOSS_ENEMY_HP_DELTA_PERCENT_SCALE)
            else:
                raw = delta * ENEMY_HP_DELTA_REWARD_SCALE
        else:
            raw = delta * ENEMY_HP_DELTA_REWARD_SCALE
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
        delta = after_hp - before_hp
        # Tier-aware loss multiplier (combat-reward-curriculum.md §3, §6).  Boss
        # fights (non-A10) restore HP post-combat, so heavy HP-loss penalties
        # distort boss play toward "turtle forever" instead of "win efficiently
        # using mechanics".  HP *gain* keeps full weight in all tiers �?healing
        # is equally valuable regardless of what encounter triggered it.
        #
        # Curriculum layer (§13.1): the tier scale is further multiplied by a
        # progress-lerped weight from the CurriculumTracker so Phase 0 policies
        # see minimal HP pressure (they just need to win first) and Phase 3
        # policies see full pressure.
        if delta < 0.0:
            tier = self._current_encounter_tier()
            tier_scale = float(PLAYER_HP_LOSS_TIER_SCALE.get(tier, 1.0))
            progress_weight = self._curriculum_tracker.hp_weight(
                self._current_encounter_id or "", tier
            )
            multiplier = tier_scale * progress_weight
        else:
            multiplier = 1.0
        return delta * PLAYER_HP_LOSS_REWARD_SCALE * multiplier

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

    @staticmethod
    def _card_cost(card: dict[str, Any] | None) -> float:
        if not isinstance(card, dict):
            return 0.0
        for key in ("cost", "resolved_energy_cost", "canonical_energy_cost"):
            value = card.get(key)
            try:
                return max(float(value), 0.0)
            except (TypeError, ValueError):
                continue
        return 0.0

    @staticmethod
    def _card_is_x_cost(card: dict[str, Any] | None) -> bool:
        if not isinstance(card, dict):
            return False
        return bool(card.get("x_cost") or card.get("costs_x") or card.get("is_x_cost")) or str(card.get("cost") or card.get("canonical_energy_cost") or "").strip().upper() == "X"

    @classmethod
    def _action_is_x_cost(cls, action: dict[str, Any] | None) -> bool:
        """Runtime action-level X-cost detector.

        Do not rely only on the embedded card cost: live compact actions may
        expose X-cost as ``semantic.roles=["x_cost"]`` while the card carries a
        temporary numeric cost after runtime modifiers.  Combat diagnostics and
        strategic-skip logic must use the action-level contract when available.
        """

        if not isinstance(action, dict):
            return False
        if "x_cost" in cls._action_roles(action):
            return True
        semantic = cls._action_semantic(action)
        if bool(semantic.get("is_x_cost")):
            return True
        try:
            if float(semantic.get("x_cost_value") or 0.0) > 0.0:
                return True
        except (TypeError, ValueError):
            pass
        action_x_cost = action.get("x_cost")
        if not isinstance(action_x_cost, dict) and bool(action_x_cost):
            return True
        if bool(action.get("costs_x") or action.get("is_x_cost")):
            return True
        if str(action.get("card_cost") or "").strip().upper() == "X":
            return True
        card = action.get("card") if isinstance(action.get("card"), dict) else None
        return cls._card_is_x_cost(card)

    def _action_modifier_semantics(self, action: dict[str, Any] | None) -> dict[str, float]:
        if not isinstance(action, dict):
            return {}
        card = action.get("card")
        if isinstance(card, dict):
            return _aggregate_card_modifier_semantics(card)
        return {}

    def _action_effect_semantics(self, action: dict[str, Any] | None) -> dict[str, float]:
        """Typed card-effect semantics for one legal action.

        This intentionally reads the bridge/registry ``card_effect_profile``
        instead of card text.  Potion timing has its own typed profile path;
        combat card quality needs card-side facts such as gain-energy,
        modify-cost, no-draw, retain/exhaust, replay, and hand mutation.
        """
        if not isinstance(action, dict):
            return {}
        card = action.get("card")
        if isinstance(card, dict):
            return aggregate_card_effect_profile_semantics(card)
        return {}

    def _action_positive_score(self, action: dict[str, Any]) -> float:
        kind = str(action.get("kind") or "").strip()
        if kind not in ("play_card", "use_potion"):
            return 0.0
        source = action.get("card") if kind == "play_card" else action.get("potion")
        if not isinstance(source, dict):
            return 0.0
        if kind == "play_card" and str(source.get("type") or "").strip().lower() == "power":
            return 4.0
        score = 0.0
        weights = {
            "damage": 1.0,
            "block": 0.8,
            "draw": 3.0,
            "weak": 4.0,
            "vulnerable": 4.0,
            "heal": 2.0,
            "strength": 3.0,
            "dexterity": 3.0,
            "summon": 5.0,
        }
        for key, weight in weights.items():
            score += self._source_preview_metric(source, key) * weight
        if kind == "play_card":
            sem = _aggregate_card_modifier_semantics(source)
            score += sem.get("energy_gain", 0.0) * 3.0
            score += sem.get("draw", 0.0) * 3.0
            score += sem.get("block_add", 0.0) * 0.8 + sem.get("block_on_play", 0.0) * 0.8
            score += sem.get("damage_add", 0.0)
            score += sem.get("weak", 0.0) * 4.0
            score -= sem.get("energy_loss_on_play", 0.0) * 2.0
            score -= sem.get("self_damage", 0.0) * 2.5
            typed = aggregate_card_effect_profile_semantics(source)
            score += typed.get("typed_gain_energy_amount", 0.0) * 3.0
            score += typed.get("typed_draw_amount", 0.0) * 3.0
            score -= typed.get("typed_hp_loss", 0.0) * 2.5
            # Hand/card-state mutation is real progress, but it is setup-like
            # progress.  Keep the value modest so follow-up-dependent cards
            # (cost reducers, no-draw/future-penalty cards, energy refunds)
            # can be classified as deferable rather than mandatory.
            if any(
                typed.get(key, 0.0) > 0.0
                for key in (
                    "typed_upgrade_hand",
                    "typed_modify_cost",
                    "typed_set_replay",
                    "typed_retain_cards",
                    "typed_add_modifier",
                    "typed_add_keyword",
                    "typed_add_generated_card",
                    "typed_card_state_mutation",
                )
            ):
                score += 2.0
        return float(max(score, 0.0))

    def _is_positive_progress_action(self, action: dict[str, Any]) -> bool:
        return self._action_positive_score(action) > 0.0

    def _classify_refund_followup(
        self,
        action: dict[str, Any],
        *,
        energy: float,
        legal_actions: list[dict[str, Any]],
        raw_obs: dict[str, Any] | None = None,
    ) -> str:
        """B3: classify a refund (energy_gain) play by its after-action prospects.

        Returns one of ``refund_good_followup``, ``refund_no_followup_but_intrinsic_value``,
        ``refund_no_followup_low_value``, or ``refund_unknown`` (non-refund).

        Intent: avoid penalising a refund played without a *static* followup
        when the card itself draws/creates new cards, reduces costs, or has
        intrinsic block/lethal/mechanism value (Kaiser facing change, etc.).
        """

        if str(action.get("kind") or "").strip() != "play_card":
            return "refund_unknown"
        card = action.get("card") if isinstance(action.get("card"), dict) else None
        if not isinstance(card, dict):
            return "refund_unknown"
        sem = _aggregate_card_modifier_semantics(card)
        typed = aggregate_card_effect_profile_semantics(card)
        energy_gain = max(
            float(sem.get("energy_gain", 0.0) or 0.0),
            float(typed.get("typed_gain_energy_amount", 0.0) or 0.0),
        )
        if energy_gain <= 0.0:
            return "refund_unknown"

        # After-action energy estimate.
        cost = self._card_cost(card)
        energy_loss = float(sem.get("energy_loss_on_play", 0.0) or 0.0)
        energy_after = max(0.0, float(energy) - float(cost) + energy_gain - energy_loss)

        # Followup signals derived from the *card's own* expected effects.
        expected_draw = float(typed.get("typed_draw_amount", 0.0) or 0.0) + float(sem.get("draw", 0.0) or 0.0)
        expected_create = bool(typed.get("typed_add_generated_card", 0.0) > 0.0)
        cost_reduction = bool(typed.get("typed_modify_cost", 0.0) > 0.0)
        replay_or_duplicate = bool(
            typed.get("typed_set_replay", 0.0) > 0.0
            or typed.get("typed_copy_cards", 0.0) > 0.0
        )
        if expected_draw >= 1.0 or expected_create or cost_reduction or replay_or_duplicate:
            return "refund_good_followup"

        # Static followup on the *current* hand.
        for other in legal_actions:
            if other is action or not isinstance(other, dict) or other.get("kind") != "play_card":
                continue
            other_card = other.get("card")
            if not isinstance(other_card, dict) or self._action_is_x_cost(other):
                continue
            other_cost = self._card_cost(other_card)
            if other_cost <= energy_after + 1e-6 and self._action_positive_score(other) >= 3.0:
                return "refund_good_followup"

        # Intrinsic value: block-lethal / mechanism / large-impact even without
        # a chained followup.
        block_value = max(
            float(self._source_preview_metric(card, "block")),
            float(sem.get("block_add", 0.0) or 0.0) + float(sem.get("block_on_play", 0.0) or 0.0),
        )
        incoming = self._refund_incoming_damage(raw_obs)
        if block_value > 0.0 and block_value + float((((raw_obs or {}).get("player") or {}).get("block")) or 0.0) >= incoming and incoming >= 8.0:
            return "refund_no_followup_but_intrinsic_value"
        damage_value = float(self._source_preview_metric(card, "damage"))
        if damage_value >= 12.0:
            return "refund_no_followup_but_intrinsic_value"
        # Mechanism: refund that targets the back-attack side counts as
        # facing-change intrinsic value.  Per P0-3 hardening spec, position
        # (left/right) MUST come from BACK_ATTACK_{LEFT,RIGHT}_POWER on the
        # target enemy, not from faction ``side`` strings on the target dict.
        try:
            from .boss_kaiser import classify_kaiser_action_mechanism  # noqa: WPS433
            kaiser_mech = classify_kaiser_action_mechanism(
                (raw_obs or {}).get("combat") if isinstance(raw_obs, dict) else None,
                action,
                player_obs=(raw_obs or {}).get("player") if isinstance(raw_obs, dict) else None,
            )
            if kaiser_mech.get("kaiser_changes_facing"):
                return "refund_no_followup_but_intrinsic_value"
        except Exception:
            pass
        return "refund_no_followup_low_value"

    @staticmethod
    def _refund_incoming_damage(raw_obs: dict[str, Any] | None) -> float:
        if not isinstance(raw_obs, dict):
            return 0.0
        combat = raw_obs.get("combat") if isinstance(raw_obs.get("combat"), dict) else {}
        enemies = combat.get("enemies") if isinstance(combat.get("enemies"), list) else []
        total = 0.0
        for enemy in enemies:
            if not isinstance(enemy, dict):
                continue
            intent = enemy.get("intent") if isinstance(enemy.get("intent"), dict) else {}
            for key in ("total_damage", "damage", "intent_damage", "attack_damage"):
                v = intent.get(key) if intent else enemy.get(key)
                try:
                    total = max(total, float(v or 0.0))
                except (TypeError, ValueError):
                    pass
        return total

    def _is_strategic_skip_candidate(
        self,
        action: dict[str, Any],
        *,
        energy: float,
        legal_actions: list[dict[str, Any]],
    ) -> bool:
        if str(action.get("kind") or "").strip() != "play_card":
            return False
        card = action.get("card")
        if not isinstance(card, dict):
            return False
        sem = _aggregate_card_modifier_semantics(card)
        typed = aggregate_card_effect_profile_semantics(card)
        if self._action_is_x_cost(action) and energy <= 0.05:
            return True
        immediate = self._action_positive_score(action)
        exhausts = bool(
            sem.get("adds_exhaust")
            or sem.get("removes_exhaust") < 0.0
            or card.get("exhaust")
            or card.get("will_exhaust")
            or typed.get("typed_once_or_exhaust_self", 0.0) > 0.0
            or typed.get("typed_exhaust_cards", 0.0) > 0.0
        )
        retains = bool(sem.get("adds_retain") or card.get("retain") or typed.get("typed_retain_cards", 0.0) > 0.0)
        self_damage = max(float(sem.get("self_damage", 0.0) or 0.0), float(typed.get("typed_hp_loss", 0.0) or 0.0))
        energy_loss = sem.get("energy_loss_on_play", 0.0)
        energy_gain = max(float(sem.get("energy_gain", 0.0) or 0.0), float(typed.get("typed_gain_energy_amount", 0.0) or 0.0))
        requires_followup = bool(
            typed.get("typed_requires_followup", 0.0) > 0.0
            or typed.get("typed_strategic_skip_if_no_followup", 0.0) > 0.0
            or typed.get("typed_modify_cost", 0.0) > 0.0
            or typed.get("typed_no_draw", 0.0) > 0.0
            or typed.get("typed_future_penalty", 0.0) > 0.0
        )
        future_penalty = bool(
            typed.get("typed_no_draw", 0.0) > 0.0
            or typed.get("typed_future_penalty", 0.0) > 0.0
            or typed.get("typed_consumes_future_resource", 0.0) > 0.0
        )
        card_state_setup = bool(
            typed.get("typed_card_state_mutation", 0.0) > 0.0
            or typed.get("typed_modifies_hand", 0.0) > 0.0
        )
        cost = self._card_cost(card)
        energy_after = max(0.0, energy - cost + energy_gain - energy_loss)
        followups = 0
        for other in legal_actions:
            if other is action or not isinstance(other, dict) or other.get("kind") != "play_card":
                continue
            other_card = other.get("card")
            if isinstance(other_card, dict) and self._card_cost(other_card) <= energy_after + 1e-6 and self._action_positive_score(other) > 0.0:
                followups += 1
        if energy_gain > 0.0 and followups <= 0 and immediate < max(6.0, energy_gain * 3.0):
            return True
        if requires_followup and followups <= 0 and immediate < max(6.0, energy_gain * 3.0 + 2.0):
            return True
        if future_penalty and followups <= 0 and immediate < 8.0:
            return True
        # B2 narrowing: pure exhaust/retain alone is no longer a strategic skip
        # candidate.  Without a future-reason signal (hand mutation, replay, deck
        # cycling, upgrade-hand) the model would otherwise be trained to fear any
        # consume card.  Lethal/high-impact exhausts have immediate >> 4 and so
        # already fall through; the threshold drop from 6 to 4 also drops the
        # mid-range "exhaust 5 damage" cases that were being mislabelled.
        has_future_setup_signal = bool(
            typed.get("typed_card_state_mutation", 0.0) > 0.0
            or typed.get("typed_modifies_hand", 0.0) > 0.0
            or typed.get("typed_set_replay", 0.0) > 0.0
            or typed.get("typed_upgrade_hand", 0.0) > 0.0
            or typed.get("typed_consumes_future_resource", 0.0) > 0.0
        )
        if (
            (exhausts or retains)
            and has_future_setup_signal
            and immediate < 4.0
            and not card.get("ethereal")
        ):
            return True
        if card_state_setup and followups <= 0 and immediate < 4.0:
            return True
        if (self_damage > 0.0 or energy_loss > 0.0) and immediate < (self_damage * 2.5 + energy_loss * 2.0 + 4.0):
            return True
        return False

    def _action_quality_diagnostics(
        self,
        obs: dict[str, Any] | None,
        legal_actions: list[dict[str, Any]],
        chosen_action: dict[str, Any] | None,
    ) -> dict[str, float]:
        combat = obs.get("combat") if isinstance(obs, dict) else None
        energy = float((combat or {}).get("energy") or 0.0) if isinstance(combat, dict) else 0.0
        selected_id = str((chosen_action or {}).get("action_id") or "")
        selected_is_end_turn = float(selected_id == "end_turn")
        positive_available = 0
        mandatory_positive = 0
        strategic_skip = 0
        zero_x_available = 0
        refund_no_followup_available = 0
        typed_followup_missing = 0
        typed_future_penalty_count = 0
        typed_no_draw_count = 0
        typed_card_state_mutation_count = 0
        setup_followup_dependent_count = 0
        setup_followup_available_count = 0
        enchantment_seen = 0
        affliction_seen = 0
        selected_zero_x = 0.0
        selected_refund_no_followup = 0.0
        selected_strategic_skip = 0.0
        for action in legal_actions:
            if not isinstance(action, dict) or str(action.get("action_id") or "") == "end_turn":
                continue
            card = action.get("card") if isinstance(action.get("card"), dict) else None
            typed = aggregate_card_effect_profile_semantics(card) if isinstance(card, dict) else {}
            sem = _aggregate_card_modifier_semantics(card) if isinstance(card, dict) else {}
            if isinstance(card, dict):
                if (
                    isinstance(card.get("enchantments"), list)
                    and card.get("enchantments")
                    or typed.get("typed_add_modifier", 0.0) > 0.0
                    or typed.get("typed_card_rule_modifier", 0.0) > 0.0
                ):
                    enchantment_seen = 1
                if isinstance(card.get("afflictions"), list) and card.get("afflictions"):
                    affliction_seen = 1
            is_positive = self._is_positive_progress_action(action)
            if is_positive:
                positive_available += 1
            is_strategic = self._is_strategic_skip_candidate(action, energy=energy, legal_actions=legal_actions)
            if is_strategic:
                strategic_skip += 1
            else:
                if is_positive:
                    mandatory_positive += 1
            is_zero_x = isinstance(card, dict) and self._action_is_x_cost(action) and energy <= 0.0
            if is_zero_x:
                zero_x_available += 1
            typed_energy_gain = float(typed.get("typed_gain_energy_amount", 0.0) or 0.0)
            typed_modify_cost = typed.get("typed_modify_cost", 0.0) > 0.0
            typed_no_draw = typed.get("typed_no_draw", 0.0) > 0.0
            typed_future_penalty = typed.get("typed_future_penalty", 0.0) > 0.0
            typed_consumes_future_resource = typed.get("typed_consumes_future_resource", 0.0) > 0.0
            typed_card_state_mutation = typed.get("typed_card_state_mutation", 0.0) > 0.0 or typed.get("typed_modifies_hand", 0.0) > 0.0
            setup_followup_dependent = bool(
                typed.get("typed_requires_followup", 0.0) > 0.0
                or typed.get("typed_strategic_skip_if_no_followup", 0.0) > 0.0
                or typed_modify_cost
                or typed_no_draw
                or typed_future_penalty
                or typed_consumes_future_resource
            )
            setup_followup_available = False
            if isinstance(card, dict) and setup_followup_dependent:
                energy_after = max(
                    0.0,
                    energy
                    - self._card_cost(card)
                    + max(float(sem.get("energy_gain", 0.0) or 0.0), typed_energy_gain)
                    - float(sem.get("energy_loss_on_play", 0.0) or 0.0),
                )
                for other in legal_actions:
                    if other is action or not isinstance(other, dict) or other.get("kind") != "play_card":
                        continue
                    other_card = other.get("card")
                    if not isinstance(other_card, dict) or self._action_is_x_cost(other):
                        continue
                    other_cost = self._card_cost(other_card)
                    if (other_cost <= energy_after + 1e-6 or typed_modify_cost) and self._action_positive_score(other) >= 3.0:
                        setup_followup_available = True
                        break
            if setup_followup_dependent:
                setup_followup_dependent_count += 1
                if setup_followup_available:
                    setup_followup_available_count += 1
                else:
                    typed_followup_missing += 1
            if typed_future_penalty:
                typed_future_penalty_count += 1
            if typed_no_draw:
                typed_no_draw_count += 1
            if typed_card_state_mutation:
                typed_card_state_mutation_count += 1
            is_refund = max(float(sem.get("energy_gain", 0.0) or 0.0), typed_energy_gain) > 0.0 and is_strategic
            if is_refund:
                refund_no_followup_available += 1
            if chosen_action is action:
                selected_zero_x = float(is_zero_x)
                selected_refund_no_followup = float(is_refund)
                selected_strategic_skip = float(is_strategic)
        strict_end_turn = strict_end_turn_waste_context(
            obs,
            legal_actions,
            chosen_action,
            positive_score_fn=self._action_positive_score,
            strategic_skip_fn=lambda action, current_energy, actions: self._is_strategic_skip_candidate(
                dict(action),
                energy=current_energy,
                legal_actions=[dict(item) for item in actions],
            ),
        )
        # Strict EndTurn semantics: leftover energy is not waste.  Waste is only
        # when an urgent/safe non-EndTurn action exists on the stable frontier.
        wasteful_available = float(bool(strict_end_turn.get("wasteful_end_turn_available", False)))
        wasteful_selected = float(selected_is_end_turn > 0.5 and wasteful_available > 0.5)
        return {
            "energy": float(energy),
            "positive_action_count": float(positive_available),
            "mandatory_positive_action_count": float(mandatory_positive),
            "urgent_positive_action_count": float(strict_end_turn.get("urgent_positive_action_count", 0.0) or 0.0),
            "non_end_turn_action_count": float(strict_end_turn.get("non_end_turn_action_count", 0.0) or 0.0),
            "playable_card_count": float(strict_end_turn.get("playable_card_count", 0.0) or 0.0),
            "incoming_damage": float(strict_end_turn.get("incoming_damage", 0.0) or 0.0),
            "current_block": float(strict_end_turn.get("current_block", 0.0) or 0.0),
            "benign_leftover_energy": float(bool(strict_end_turn.get("benign_leftover_energy", False))),
            "strategic_skip_candidate_count": float(strategic_skip),
            "wasteful_end_turn_available": wasteful_available,
            "wasteful_end_turn_selected": wasteful_selected,
            "zero_energy_x_cost_available": float(zero_x_available),
            "zero_energy_x_cost_selected": selected_zero_x,
            "refund_no_followup_available": float(refund_no_followup_available),
            "refund_no_followup_selected": selected_refund_no_followup,
            "strategic_skip_selected": selected_strategic_skip,
            "typed_followup_missing_count": float(typed_followup_missing),
            "typed_future_penalty_count": float(typed_future_penalty_count),
            "typed_no_draw_count": float(typed_no_draw_count),
            "typed_card_state_mutation_count": float(typed_card_state_mutation_count),
            "setup_followup_dependent_count": float(setup_followup_dependent_count),
            "setup_followup_available_count": float(setup_followup_available_count),
            "enchantment_seen": float(enchantment_seen),
            "affliction_seen": float(affliction_seen),
        }

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

        diagnostics = self._action_quality_diagnostics(obs, legal_actions, chosen_action)
        positive_actions = int(diagnostics.get("urgent_positive_action_count", 0.0))
        if positive_actions <= 0:
            return 0.0

        strict_context = strict_end_turn_waste_context(
            obs,
            legal_actions,
            chosen_action,
            positive_score_fn=self._action_positive_score,
            strategic_skip_fn=lambda action, current_energy, actions: self._is_strategic_skip_candidate(
                dict(action),
                energy=current_energy,
                legal_actions=[dict(item) for item in actions],
            ),
        )
        urgent_indices = {int(idx) for idx in strict_context.get("urgent_positive_indices", [])}
        has_zero_cost_positive = any(
            isinstance(action, dict)
            and idx in urgent_indices
            and str(action.get("action_id") or "") != "end_turn"
            and isinstance(action.get("card"), dict)
            and self._card_cost(action.get("card")) <= 0.0
            for idx, action in enumerate(legal_actions)
        )

        penalty = END_TURN_WASTE_BASE_PENALTY
        penalty += END_TURN_WASTE_ENERGY_PENALTY * min(energy, 3.0)
        if has_zero_cost_positive:
            penalty += END_TURN_WASTE_ZERO_COST_BONUS_PENALTY
        penalty += END_TURN_WASTE_EXTRA_ACTION_PENALTY * min(max(positive_actions - 1, 0), 2)
        # Tier multiplier (combat-reward-curriculum.md §9.1).  Base stack maxes
        # at �?.10; tier multipliers (1.5/1.5/2.5/3.0) take it to the doc's
        # �?.15/�?.25/�?.30 targets without altering the detector logic.
        tier = self._current_encounter_tier()
        tier_mult = float(WASTEFUL_END_TURN_TIER_MULTIPLIER.get(tier, 1.5))
        penalty *= tier_mult
        # Lightweight stdout breadcrumb so the operator can see waste actually
        # happening between hourly tfevents reports.  Throttled to one print
        # every WASTE_PRINT_INTERVAL events to avoid log spam.
        self._wasteful_end_turn_count += 1
        if self._wasteful_end_turn_count % 25 == 0:
            print(
                f"[combat_env] wasteful_end_turn count={self._wasteful_end_turn_count} "
                f"tier={tier} energy={energy:.0f} positive_actions={positive_actions} "
                f"penalty={penalty:.3f}",
                flush=True,
            )
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
