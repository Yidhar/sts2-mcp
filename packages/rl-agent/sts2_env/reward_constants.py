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

# ---- Phase 8.2 reward density shaping ----
# After the 800k Phase 8 long-train, the policy reliably reached Act 1
# boss (2% of episodes at floor 17) but almost never killed it
# (0.05% at floor 18+). Diagnosis: binary terminal reward (+1 win /
# -1 loss) carries too little signal density when diluted across ~150-
# step episodes, and per-step hp-delta shaping doesn't distinguish boss
# damage from act-1-cultist grind. Fix: dense floor-clear ladder +
# boss-specific damage multiplier so value head can attribute "being
# deep / fighting boss" as strongly positive.

# Per-step bonus when the observation's run.floor increments past
# FLOOR_CLEAR_MIN_FLOOR. Captures Act 1 late-game push without
# rewarding easy early floors.
FLOOR_CLEAR_MIN_FLOOR = 11
FLOOR_CLEAR_BONUS_PER_FLOOR = 0.3
# Boss-room entry gets an even larger one-shot bonus — this is the
# trajectory that needs to become value-function-attractive so the
# policy doesn't prefer "play safe, stall on floor 5-8".
BOSS_FLOOR_ENTRY_BONUS = 2.0
# Canonical STS2 act-boss floors (assumed: act length 16). If a mod
# changes act length, the state_type=="boss" detection below catches
# it as a fallback.
BOSS_ACT_FLOORS = (17, 34, 51)

# Multiplier applied ON TOP OF the base enemy_hp_delta reward when the
# current encounter is flagged as a boss fight. 5.0 = total reward is
# 5× what it would be for a basic monster of the same damage dealt.
# Asymmetric: only damage dealt gets amplified, hp lost penalty stays
# at base scale — we want the value function to rate "dealing damage
# to boss" much higher than any non-boss action, not to make boss
# fights intrinsically more punishing.
BOSS_DAMAGE_MULTIPLIER = 5.0

# ---- Phase 8.2b — rest-site HP gate + potion-use shaping ----
# The 800k policy learned to almost never use potions and to prefer
# SMITH over HEAL at campfires even at dangerous HP levels. The base
# hp-delta reward DOES already compensate resting (+~0.7 for a 24-hp
# heal at scale 0.03) but the learned preference persists — the signal
# is dominated by noise. These shaping terms provide direct gradient.

# Penalty for picking a non-HEAL rest-site option (smith / dig / pray /
# recall / clone / hatch / cook / toke / lift / ...) when the player's
# HP ratio is below REST_SITE_SKIP_HEAL_HP_THRESHOLD. MUST stay smaller
# in magnitude than FLOOR_CLEAR_BONUS_PER_FLOOR (0.30) — otherwise the
# policy learns to AVOID campfire tiles on the map rather than use
# them wisely. At -0.15 (half the floor-clear bonus) the per-campfire
# net is still positive for the policy overall, just the within-rest-
# site choice becomes biased toward HEAL when HP is low.
REST_SITE_SKIP_HEAL_HP_THRESHOLD = 0.60
REST_SITE_SKIP_HEAL_PENALTY = -0.15

# Potion-use bonus — encounter-scoped absolute values (NOT a base×mult)
# with a hoarding penalty at episode end.
#
# Design iteration history:
#   v1 (65b5196): flat base 0.10 + 2x/3x elite/boss multipliers. Result:
#     153 uses / 66 eps, but 100% in monster fights — the 0.10 base
#     made non-boss use net-positive so policy burned potions before
#     ever reaching a boss.
#   v2 (84f8ca6): base removed, monster use = 0. Result: policy now
#     indifferent between "use in monster" and "never use" (both 0).
#     Still wastes potions that sit unused through episode end.
#   v3 (this commit): small positive for any use, bigger for elite,
#     biggest for boss, PLUS per-unused-potion penalty at episode end.
#     Ranking the policy sees: boss > elite > monster > hoard.
POTION_USE_MONSTER_BONUS = 0.05
POTION_USE_ELITE_BONUS = 0.20
POTION_USE_BOSS_BONUS = 0.30
# Optional ADDITIONAL negative signal layered on monster-fight use.
# Defaulted to 0. Only enable if the hoarding penalty isn't enough
# pressure and the policy still prefers monster-fight burning over
# saving for bosses. Must stay |x| < FLOOR_CLEAR_BONUS_PER_FLOOR so
# policy can't learn "avoid potions entirely" as a shortcut.
POTION_USE_MONSTER_PENALTY = 0.0

# End-of-episode hoarding penalty per unused potion. Fires on BOTH
# terminated and truncated episode ends (unused potions are wasted
# regardless of win/loss/watchdog). STS has 3 potion slots max, so
# worst case = 3 × PENALTY = -0.30, which is intentionally capped at
# exactly FLOOR_CLEAR_BONUS_PER_FLOOR so policy can never learn
# "avoid having potions in inventory" — the savings from skipping
# a potion reward event are always bounded by the guaranteed
# floor-clear value of actually progressing.
POTION_HOARDING_PENALTY_PER_POTION = -0.10
POTION_HOARDING_MAX_PENALTY_ABS = 0.30

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
