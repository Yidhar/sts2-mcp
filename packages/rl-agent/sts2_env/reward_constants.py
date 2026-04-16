"""Shared reward shaping constants for STS2 environments.

ENEMY_HP_DELTA_REWARD_SCALE and PLAYER_HP_LOSS_REWARD_SCALE are used identically
by both CombatSandboxEnv and SlayTheSpire2EnvV2.

END_TURN_WASTE_* constants are intentionally different between combat sandbox
(lighter penalties to encourage exploration) and full-run (heavier penalties
where wasted turns are costlier).
"""

# ---- Shared across all environments ----
ENEMY_HP_DELTA_REWARD_SCALE = 0.01
PLAYER_HP_LOSS_REWARD_SCALE = 0.03
INVALID_ACTION_REWARD = -1.0

# ---- End-turn waste penalty presets ----
# Combat sandbox: lighter penalties for focused combat training
COMBAT_SANDBOX_WASTE_BASE = -0.03
COMBAT_SANDBOX_WASTE_ENERGY = -0.01
COMBAT_SANDBOX_WASTE_ZERO_COST = -0.02
COMBAT_SANDBOX_WASTE_EXTRA_ACTION = -0.01

# Full run: heavier penalties — wasted turns are costlier in context
FULL_RUN_WASTE_BASE = -0.08
FULL_RUN_WASTE_ENERGY = -0.04
FULL_RUN_WASTE_ZERO_COST = -0.08
FULL_RUN_WASTE_EXTRA_ACTION = -0.03
