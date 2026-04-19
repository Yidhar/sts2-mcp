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

# Enemies reporting hp above this are treated as sentinel-invulnerable (e.g.
# WATERFALL_GIANT_BOSS has hp ≈ 1e9 until a kill condition triggers). Without
# this filter the terminal hp→0 transition produces reward ≈ 1e7 and blows up
# value loss. STS2 enemies never legitimately exceed a few thousand hp.
ENEMY_HP_SENTINEL_THRESHOLD = 10_000.0

# Final safety cap on per-step enemy-hp-delta reward magnitude. Covers any
# residual anomaly the sentinel filter doesn't catch (multiple phase swaps,
# missing max_hp, etc.). Normal combat damage/step rarely exceeds ~500 hp,
# which at scale 0.01 is reward 5.0 — the cap at 20 is comfortably above that.
ENEMY_HP_DELTA_REWARD_MAX_ABS = 20.0

# ---- Sentinel-HP boss terminal shaping ----
# WATERFALL_GIANT_BOSS and similar "speed-check" enemies have HP ≈ 1e9 (never
# killable by damage) and a delayed on-death explosion dealing damage equal to
# an accumulated buff stack count. The player must either kill fast enough to
# keep stacks low or stack enough block+hp to survive the blast. The normal
# HP-delta reward is zero for these fights (sentinel filter), so we inject an
# explicit terminal signal: alive-win = positive scaled by remaining hp,
# otherwise = negative scaled by death-damage overshoot that broke through
# (block + hp).
SENTINEL_COMBAT_WIN_BONUS_BASE = 1.5
SENTINEL_COMBAT_WIN_BONUS_SCALE = 1.5
SENTINEL_COMBAT_LOSS_PENALTY_BASE = 1.5
SENTINEL_COMBAT_LOSS_PENALTY_SCALE = 1.5

# Substring keywords (lowercased) used to identify the boss's "on-death damage"
# buff inside enemy.powers[].title. Match is case-insensitive substring. The
# largest matching power's `amount` is used as the expected death damage.
SENTINEL_DEATH_DAMAGE_POWER_KEYWORDS = (
    "explosion",
    "retaliate",
    "deathburst",
    "burst",
    "deathblow",
    "death",
)

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
