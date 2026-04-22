"""Run-chained combat sandbox env — replay a full human run's combat sequence
as a single episode, with HP + potion carry-over between sub-combats.

Design summary (finalized 2026-04-22):

1. Each episode = one human run's ordered combat snapshots
2. Sub-combat state sources:
     * deck, relics, gold, max_hp, max_energy, encounter_id → from snapshot[i]
       (human's build evolution; rest/event/shop outcomes baked in)
     * current_hp = min(snapshot[i].hp, policy_last_end_hp)
       (policy can't exceed the human-path HP ceiling; if policy played worse,
       next combat starts with even less HP)
     * potions = policy_last_end_potions + (snapshot[i].potions - snapshot[i-1].potions)
       (policy retains unused potions across combats; human's additions from
       rest/event/shop get appended, capped at 3 slots)
3. Episode terminal:
     * player_hp <= 0 during any sub-combat → loss
     * all snapshots played and survived → win
     * +5.0 run_complete_bonus reward if last snapshot was a boss and player
       survived
4. HISTORY tokens (ActionHistoryTracker) persist across sub-combats — the
   only "cross-combat memory" the policy has beyond carried HP/potions

The wrapper re-uses CombatSandboxEnv's combat loop machinery by calling
its reset(options={...}) between sub-combats with the carry-over-adjusted
kwargs.
"""
from __future__ import annotations

from typing import Any

import gymnasium as gym
import numpy as np

from combat_snapshot_dataset import RunChainedSnapshotPool
from sts2_env.combat_env import CombatSandboxEnv


RUN_COMPLETE_BOSS_BONUS = 5.0


class RunChainedCombatEnv(gym.Wrapper):
    """Stitches multiple CombatSandboxEnv resets into one long episode.

    Expects the underlying CombatSandboxEnv to be constructed WITHOUT a
    snapshot_pool (we drive snapshot selection ourselves via options).
    """

    def __init__(
        self,
        combat_env: CombatSandboxEnv,
        *,
        run_pool: RunChainedSnapshotPool,
        potion_slot_cap: int = 3,
        run_complete_bonus: float = RUN_COMPLETE_BOSS_BONUS,
    ) -> None:
        super().__init__(combat_env)
        if run_pool is None:
            raise ValueError("RunChainedCombatEnv requires a RunChainedSnapshotPool")
        self.run_pool = run_pool
        self.potion_slot_cap = int(potion_slot_cap)
        self.run_complete_bonus = float(run_complete_bonus)

        # Chain state (populated in reset)
        self._chain: list[dict[str, Any]] = []
        self._chain_idx: int = 0
        self._carried_hp: int | None = None
        self._carried_potions: list[str] | None = None
        self._chain_reward_accum: float = 0.0
        self._chain_sub_combat_rewards: list[float] = []
        self._chain_sub_combat_outcomes: list[str] = []

    # ------------------------------------------------------------------
    # Carry-over rules
    # ------------------------------------------------------------------

    def _build_kwargs_for_sub_combat(self, snap_idx: int) -> dict[str, Any]:
        """Return an options dict for CombatSandboxEnv.reset(options=...).

        Full snapshot applied for deck/relics/encounter/gold/max_hp/max_energy.
        HP and potions get overridden per the carry-over rules.
        """
        snap = self._chain[snap_idx]
        kwargs: dict[str, Any] = {"snapshot": snap}
        if snap_idx == 0:
            # First sub-combat: pristine snapshot, nothing to carry
            return kwargs

        # HP cap: policy can't exceed human-path HP ceiling at this snapshot
        snap_hp = int(snap.get("snapshot_current_hp") or snap.get("snapshot_max_hp") or 0)
        carried_hp = int(self._carried_hp) if self._carried_hp is not None else snap_hp
        kwargs["current_hp"] = max(0, min(snap_hp, carried_hp))

        # Potions: policy's remaining + human's additions between prev and cur
        prev_snap = self._chain[snap_idx - 1]
        prev_potions = [str(p) for p in (prev_snap.get("potion_ids_before") or []) if p]
        this_potions = [str(p) for p in (snap.get("potion_ids_before") or []) if p]
        # Human's additions = what's in this snapshot that wasn't in prev (net new)
        # Use multiset math: count -> subtract -> remaining positives are additions
        from collections import Counter
        added = Counter(this_potions) - Counter(prev_potions)
        added_list: list[str] = []
        for pid, count in added.items():
            added_list.extend([pid] * count)

        carried = list(self._carried_potions or [])
        final_potions = (carried + added_list)[: self.potion_slot_cap]
        kwargs["potions"] = final_potions
        return kwargs

    def _capture_carry_from_env(self) -> None:
        """Read policy's end-of-sub-combat HP + remaining potions from the
        underlying env's last observation. Called just after a terminated
        sub-combat, before advancing chain_idx.
        """
        underlying = self.env.unwrapped if hasattr(self.env, "unwrapped") else self.env
        raw_obs = getattr(underlying, "_last_obs_raw", None) or {}
        player = raw_obs.get("player") if isinstance(raw_obs, dict) else {}
        if isinstance(player, dict):
            hp = player.get("hp")
            if hp is None:
                hp = player.get("current_hp")
            try:
                self._carried_hp = int(hp) if hp is not None else 0
            except (TypeError, ValueError):
                self._carried_hp = 0
            potions = player.get("potions") or []
            self._carried_potions = [
                (p.get("id") if isinstance(p, dict) else str(p))
                for p in potions
                if p
            ]
        else:
            self._carried_hp = 0
            self._carried_potions = []

    # ------------------------------------------------------------------
    # gym.Wrapper overrides
    # ------------------------------------------------------------------

    def reset(self, seed: int | None = None, options: dict[str, Any] | None = None):
        # Ignore external options — chain wrapper owns snapshot selection
        if self.np_random is None:
            super().reset(seed=seed)
        rng = self.np_random
        self._chain = self.run_pool.sample_run(rng)
        if not self._chain:
            raise RuntimeError("RunChainedSnapshotPool returned an empty chain")

        self._chain_idx = 0
        self._carried_hp = None
        self._carried_potions = None
        self._chain_reward_accum = 0.0
        self._chain_sub_combat_rewards = []
        self._chain_sub_combat_outcomes = []

        first_kwargs = self._build_kwargs_for_sub_combat(0)
        obs, info = self.env.reset(seed=seed, options=first_kwargs)
        info = self._decorate_info(info)
        return obs, info

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        info = self._decorate_info(info)
        self._chain_reward_accum += float(reward)

        # Not a terminal for the underlying sub-combat → forward unchanged
        if not (terminated or truncated):
            return obs, float(reward), False, False, info

        # Sub-combat ended. Check chain state.
        underlying = self.env.unwrapped if hasattr(self.env, "unwrapped") else self.env
        raw_obs = getattr(underlying, "_last_obs_raw", None) or {}
        player = raw_obs.get("player") if isinstance(raw_obs, dict) else {}
        player_hp = 0
        if isinstance(player, dict):
            try:
                player_hp = int(player.get("hp") or 0)
            except (TypeError, ValueError):
                player_hp = 0

        sub_outcome = "truncated" if truncated else (
            "win" if player_hp > 0 else "loss"
        )
        self._chain_sub_combat_rewards.append(float(reward))
        self._chain_sub_combat_outcomes.append(sub_outcome)

        # Episode true terminal: policy dead, truncation, or chain exhausted
        chain_exhausted = (self._chain_idx + 1) >= len(self._chain)
        run_failed = (player_hp <= 0) or truncated

        if run_failed or chain_exhausted:
            # Terminal — maybe add run_complete_bonus
            bonus = 0.0
            run_complete = False
            run_complete_kind = None
            if chain_exhausted and not run_failed and player_hp > 0:
                # Check if last snapshot was boss
                last_enc = str(self._chain[-1].get("encounter_id") or "").upper()
                if last_enc.endswith("_BOSS"):
                    bonus = self.run_complete_bonus
                    run_complete_kind = "boss_run_complete"
                else:
                    run_complete_kind = "nonboss_run_complete"
                run_complete = True
            info = dict(info)
            info["run_chain"] = {
                "chain_length": len(self._chain),
                "chain_idx_terminal": self._chain_idx,
                "sub_combat_outcomes": list(self._chain_sub_combat_outcomes),
                "sub_combat_rewards": list(self._chain_sub_combat_rewards),
                "chain_reward_total": self._chain_reward_accum + bonus,
                "run_complete": run_complete,
                "run_complete_kind": run_complete_kind,
                "run_complete_bonus": bonus,
                "terminal_player_hp": player_hp,
                "run_id": str(self._chain[0].get("run_id") or ""),
                "run_quality_fine": str(self._chain[0].get("quality_fine") or ""),
            }
            return obs, float(reward + bonus), True, truncated, info

        # Sub-combat ended with player alive — advance to next sub-combat.
        # Capture carry state, call underlying env.reset with next kwargs,
        # return (next_obs, sub_combat_reward, terminated=False, truncated=False).
        # IMPORTANT: the returned obs is the POST-reset obs of the next sub-combat,
        # not the terminal obs of the just-finished sub-combat. PPO consumes this
        # as a normal transition, which is what we want — the chain appears as
        # one continuous episode from the policy's perspective.
        self._capture_carry_from_env()
        self._chain_idx += 1
        next_kwargs = self._build_kwargs_for_sub_combat(self._chain_idx)
        next_obs, next_info = self.env.reset(options=next_kwargs)
        next_info = self._decorate_info(next_info)
        next_info["run_chain_sub_combat_boundary"] = True
        next_info["sub_combat_prev_outcome"] = sub_outcome
        return next_obs, float(reward), False, False, next_info

    def _decorate_info(self, info: dict[str, Any] | None) -> dict[str, Any]:
        info = dict(info or {})
        info["run_chain_idx"] = self._chain_idx
        info["run_chain_length"] = len(self._chain)
        if self._chain and 0 <= self._chain_idx < len(self._chain):
            snap = self._chain[self._chain_idx]
            info["run_chain_encounter_id"] = snap.get("encounter_id")
            info["run_chain_floor_number"] = snap.get("floor_number")
            info["run_chain_run_id"] = str(snap.get("run_id") or "")
        return info
