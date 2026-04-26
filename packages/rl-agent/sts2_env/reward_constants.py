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

# ---- Tier-aware HP shaping multipliers (combat-reward-curriculum.md §3, §6) ----
# Per-tier scale applied on top of PLAYER_HP_LOSS_REWARD_SCALE for the
# player-HP-delta shaping term.
#
# Core insight from the curriculum doc: boss fights (non-A10) restore HP
# post-combat, so punishing HP loss during boss fights optimizes a quantity
# that has no long-run value.  The penalty pushes boss play toward "defend
# forever" instead of "win efficiently and use mechanics".  Weak/normal and
# elite fights DO carry HP loss into the rest of the run, so those stay
# meaningful.
#
# HP *gain* (healing) keeps full weight regardless of tier — healing is
# equally valuable to the long-run regardless of what encounter triggered it.
PLAYER_HP_LOSS_TIER_SCALE = {
    "weak": 1.00,
    "normal": 1.00,
    "elite": 1.50,  # per §3.2: "明显压低战损"
    "boss": 0.15,   # per §6.1: boss 非 A10 应为 0.1 ~ 0.3
    "unknown": 1.00,
}
# A10 / no-heal boss variants (future: detect via encounter metadata).  When
# a boss fight does NOT restore HP afterward, HP loss carries the same weight
# as elite — still tier-weighted to avoid "skip fight" gaming.
PLAYER_HP_LOSS_BOSS_NO_HEAL_SCALE = 1.50

# ---- R_outcome tier scaling (combat-reward-curriculum.md §5) ----
# Applied on top of the existing terminal win/loss reward.  Boss losses hurt
# more, boss wins are worth more.  Elite wins/losses sit between normal and
# boss because they carry bigger run-wide HP pressure.
#
# IMPORTANT: this is a MULTIPLIER on the engine-provided terminal reward,
# not a replacement.  `_boss_terminal_reward` already carries a large absolute
# boss win/loss signal; the tier scale is wired into `_tier_outcome_scale`
# and only applied where the env explicitly gates on outcome events.
OUTCOME_TIER_SCALE = {
    "weak": 1.0,
    "normal": 1.0,
    "elite": 1.5,
    "boss": 2.0,
    "unknown": 1.0,
}

# ---- R_hp_efficiency: HP preserve bonus on win (§6.2) ----
# One-shot terminal bonus proportional to sqrt(hp_end / max_hp), scaled by
# tier.  Rewards "win with HP left" without unbounded growth (sqrt flattens
# the marginal value of each additional HP saved).  Boss gets ZERO because
# non-A10 bosses restore HP post-combat — extra remaining HP has no
# long-run value and would just push the model toward tanking.
HP_PRESERVE_WIN_BONUS_TIER_SCALE = {
    "weak": 0.25,
    "normal": 0.35,
    "elite": 0.55,
    "boss": 0.0,
    "unknown": 0.0,
}

# ---- R_turn_efficiency: per-end-turn penalty (§8.2) ----
# Small NEGATIVE per-turn signal on every end_turn emission (not just
# "wasteful" ones) to counter-balance the "defend forever" attractor that
# strong HP-loss penalties create.  Tier-aware because boss fights often
# benefit from longer block-stacking arcs.
TURN_EFFICIENCY_PENALTY_PER_END_TURN_TIER = {
    "weak": -0.020,
    "normal": -0.020,
    "elite": -0.015,
    "boss": -0.008,
    "unknown": -0.020,
}

# ---- Curriculum phase scaffolding (§4, §13) ----
# Per-encounter sliding win rate drives a 0..1 "progress" scalar.  The
# progress scalar lerps HP_LOSS_TIER_SCALE between MIN and MAX multipliers
# so Phase 0 policies (win_rate < 30%) see minimal HP pressure — we need
# them to keep winning first — while Phase 3 policies (win_rate >= 80%)
# see full pressure and must optimize low-damage play.
#
# Win-rate thresholds: (P0->P1, P1->P2, P2->P3).
CURRICULUM_PHASE_WIN_RATE_THRESHOLDS = (0.30, 0.60, 0.80)
CURRICULUM_WIN_RATE_WINDOW = 128
# Don't phase-up until the encounter has been visited at least this many
# times, even if the initial episodes happen to be wins.
CURRICULUM_MIN_EPISODES_FOR_PHASE = 8
# Per-tier HP-weight lerp bounds.  Final tier_scale used in the step:
#   progress = clamp((win_rate_128 - 0.30) / 0.50, 0, 1)
#   weight = CURRICULUM_HP_WEIGHT_LERP_MIN_TIER[tier]
#          + progress * (MAX - MIN)
#   tier_scale = PLAYER_HP_LOSS_TIER_SCALE[tier] * weight
CURRICULUM_HP_WEIGHT_LERP_MIN_TIER = {
    "weak": 0.10,
    "normal": 0.10,
    "elite": 0.15,
    "boss": 0.20,   # boss tier already pinched at 0.15 → final min≈0.03
    "unknown": 0.10,
}
CURRICULUM_HP_WEIGHT_LERP_MAX_TIER = {
    "weak": 1.00,
    "normal": 1.00,
    "elite": 1.00,
    "boss": 1.00,
    "unknown": 1.00,
}

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

# Generic combat-sandbox boss terminal shaping. The bridge terminal reward is
# intentionally small and gets diluted by long boss fights; these terms make
# boss victory/death separable without requiring full-run MCTS credit assignment.
#
# H22 audit found the policy had learned a "deal damage, die anyway, still
# net positive" attractor on boss fights:
#   per-step enemy_hp_delta × BOSS_DAMAGE_MULTIPLIER (5.0) ≈ +5.0 for a
#   typical 100-hp boss damage delivered, vs only −2~−3 from the prior loss
#   penalty (BASE 2.0 + missing-hp scale 1.0).  Net: +2 even on a loss.
# Bumped LOSS_PENALTY_BASE 2.0 → 5.0 and MISSING_HP_SCALE 1.0 → 3.0 so a
# full boss wipe costs −5~−8, comfortably overpowering the +5 damage
# shaping; bumped WIN_BONUS_BASE 2.0 → 3.0 to keep the win/loss gap (and
# avoid making "skip the boss" attractive).  Net boss outcome gap roughly
# doubled while keeping per-step shaping unchanged.
BOSS_COMBAT_WIN_BONUS_BASE = 3.0
BOSS_COMBAT_WIN_BONUS_HP_SCALE = 2.0
BOSS_COMBAT_LOSS_PENALTY_BASE = 5.0
BOSS_COMBAT_LOSS_PENALTY_MISSING_HP_SCALE = 3.0
# H22 v3: percent-based boss damage shaping.
# Old design used raw HP × 0.01 × BOSS_DAMAGE_MULTIPLIER (5.0), so a 200-HP
# boss generated total per-step shaping +10 while a 900-HP multi-phase boss
# generated +45 — fixed terminal loss penalties could never catch up.  Now
# both per-step damage shaping AND terminal damage-undo on loss are
# percentage-of-boss-HP, so the maximum cumulative per-episode shaping is
# bounded regardless of boss size.
#
# Per-step (only for boss tier; non-boss tiers keep the raw scale below):
#   step_damage_ratio = damage_dealt_this_step / boss_initial_total_hp
#   reward += step_damage_ratio × BOSS_ENEMY_HP_DELTA_PERCENT_SCALE
# Sum over an episode where the boss dies = +BOSS_ENEMY_HP_DELTA_PERCENT_SCALE.
#
# Terminal on loss:
#   damage_dealt_ratio = damage_dealt_total / boss_initial_total_hp
#   penalty += damage_dealt_ratio × BOSS_COMBAT_LOSS_DAMAGE_UNDO_PERCENT_SCALE
# UNDO_SCALE > PERCENT_SCALE so net cumulative-damage contribution is
# mildly negative even on a "deal everything but die" outcome.
BOSS_ENEMY_HP_DELTA_PERCENT_SCALE = 5.0
BOSS_COMBAT_LOSS_DAMAGE_UNDO_PERCENT_SCALE = 7.0

# ---- Boss-specific mechanic shaping ----
# These are tactical, per-step shaping signals for boss mechanics that are too
# sparse for terminal win/loss alone. Kept below terminal rewards but large
# enough to survive HP-delta noise in boss sandbox training.
# Kaiser facing/back-attack mechanics — combat-reward-curriculum.md §12.1.
# Live training showed kaiser_back_attack_risk_mean = 0.97 (model essentially
# never addresses the back attack), and kaiser_facing_change_selected_rate
# stayed near 0.  The earlier values were too weak relative to the new
# tier-aware HP/outcome signals — the policy treated Kaiser as a normal boss
# and never learned to refacing.  Bumped to doc §12 magnitudes; added two
# explicit POSITIVE signals (facing change + pressure kill) since the prior
# shaping was almost all penalty without any "did the right thing" gradient.
KAISER_BACK_ATTACK_HP_LOSS_PENALTY_SCALE = 0.05   # was 0.035
KAISER_BACK_ATTACK_DEFENSE_BONUS = 0.20            # was 0.18
KAISER_BACK_ATTACK_RISK_REDUCTION_BONUS = 0.30     # was 0.24 — doc §12 spec
# Bumped after H16 observation (kaiser_crab WR=0.083, facing_change=0.029,
# back_risk=0.96): doc §12 spec was -0.30 but model still ends turn under
# back-attack 17% of decisions. Push to -0.45 to make staying in risk truly
# costly relative to defense (+0.20) / facing_change (+0.50) options.
KAISER_BACK_ATTACK_END_TURN_PENALTY = -0.45        # was -0.30
# NEW (doc §12 missing): explicit reward when the player attacks the OTHER
# enemy (or uses a positional potion) and back_attack_active flips off in the
# next snapshot — i.e. they actually re-faced the boss this turn.
# Bumped 0.30 → 0.50 after H16 observed model only learned 5% facing_change rate
# despite 0.30 incentive.
# Bumped 0.50 → 0.80 after H32: kaiser_response counter stagnated at 35 for 3
# consecutive hours and kaiser_crab WR collapsed back to 0/12.  The 0.50 bonus
# was saturating against pressure (~0.20 + per-step damage shaping).  At 0.80
# the facing change is roughly the magnitude of a single boss-card hit, making
# it MCTS-attractive even when the immediate damage option scores higher.
KAISER_FACING_CHANGE_BONUS = 0.80
# NEW (doc §12 missing): if instead of refacing the player kills the back
# enemy/part this turn (back side enemy_hp_delta > 0 AND its hp ends ≤ 0),
# reward them too — bypasses the need to reface.
KAISER_PRESSURE_KILL_BONUS = 0.20
# Soften factor when no defense candidate AND no facing-change candidate is
# legal: in that frame the model has no mechanically-correct response, so
# applying the full risk-end_turn penalty would be punishing the policy for
# the bridge's action set, not its choice. Doc §12 specifies penalty *= 0.3.
KAISER_NO_RESPONSE_PENALTY_SOFTEN = 0.30

# Knowledge Demon (知识恶魔) curse-selection shaping.  Boss applies 3
# sequential curses, each a binary choice. Option A is always 瓦解
# (Disintegrate): take N damage / turn, blockable.  Option B varies by
# curse number:
#   Curse 1 (no prior disintegrate stack): B = "draw 1 fewer card / turn"
#       → typically take B (HP loss is recoverable, draw loss is permanent)
#   Curse 2 (already 6+ damage stack):    B = "max 3 cards / turn"
#       → typically take A (7 blockable damage beats hard hand-size cap)
#   Curse 3 (already 13+ damage stack):   B = "−1 energy / turn"
#       → both bad; prefer A (still blockable) but should kill before this
# Strategy: kill Knowledge Demon before curse 3.  Each step in the curse
# event loop where the agent doesn't engage / picks the dominated option
# is a measurable mistake worth shaping.
KNOWLEDGE_DEMON_GOOD_CURSE_PICK_BONUS = 0.40
KNOWLEDGE_DEMON_BAD_CURSE_PICK_PENALTY = -0.40
# Reward keeping the boss-fight short: every full round the boss survives
# adds another disintegrate stack (next curse will be worse).  Apply a
# small per-end_turn penalty so the policy values speed-killing over
# turtling.
KNOWLEDGE_DEMON_END_TURN_PENALTY = -0.05

CEREMONIAL_STUN_WINDOW_ENTER_BONUS = 0.45
CEREMONIAL_THRESHOLD_PROGRESS_BONUS = 0.25
CEREMONIAL_STUN_DAMAGE_MULTIPLIER = 0.018
CEREMONIAL_ONE_CARD_HIGH_IMPACT_BONUS = 0.22
CEREMONIAL_ONE_CARD_LOW_IMPACT_PENALTY = -0.20
CEREMONIAL_ONE_CARD_END_TURN_PENALTY = -0.22

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
#   v3 (earlier): small positive for any use, bigger for elite,
#     biggest for boss, PLUS per-unused-potion penalty at episode end.
#     Ranking the policy sees: boss > elite > monster > hoard.  Worked
#     for teaching "use potions at all", but policy started burning
#     potions greedily and mistiming them — shaping was pushing use
#     regardless of HP situation.
#   v4 (this commit): all direct potion shaping zeroed.  Policy already
#     knows HOW to use potions; now it must learn WHEN purely from the
#     downstream HP-delta reward (PLAYER_HP_LOSS_REWARD_SCALE +
#     ENEMY_HP_DELTA_REWARD_SCALE + boss terminal shaping).  Hoarding
#     penalty also removed — it was biasing toward late-combat dump-
#     all-potions instead of outcome-conditioned use.
POTION_USE_MONSTER_BONUS = 0.0
POTION_USE_ELITE_BONUS = 0.0
POTION_USE_BOSS_BONUS = 0.0
POTION_USE_MONSTER_PENALTY = 0.0

POTION_HOARDING_PENALTY_PER_POTION = 0.0
POTION_HOARDING_MAX_PENALTY_ABS = 0.0

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
# H22: matched the boss-loss bump above so sentinel-boss death (Waterfall
# Giant explosion etc.) is also clearly net-negative, not a "did some damage
# while dying, net positive" attractor.  Win bonus also bumped to keep gap.
SENTINEL_COMBAT_WIN_BONUS_BASE = 3.0     # was 1.5
SENTINEL_COMBAT_WIN_BONUS_SCALE = 2.0    # was 1.5
SENTINEL_COMBAT_LOSS_PENALTY_BASE = 5.0  # was 1.5
SENTINEL_COMBAT_LOSS_PENALTY_SCALE = 3.0  # was 1.5

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
# Combat sandbox: lighter penalties for focused combat training.  Worst-case
# stack at base+energy+zero_cost+extra_action ≈ −0.10, then multiplied by the
# tier scale below.
COMBAT_SANDBOX_WASTE_BASE = -0.03
COMBAT_SANDBOX_WASTE_ENERGY = -0.01
COMBAT_SANDBOX_WASTE_ZERO_COST = -0.02
COMBAT_SANDBOX_WASTE_EXTRA_ACTION = -0.01

# combat-reward-curriculum.md §9.1 target magnitudes (per WASTEFUL end_turn):
#   normal  −0.15
#   elite   −0.25
#   boss    −0.25 to −0.35
# The base/energy/zero_cost/extra_action stack maxes out at −0.10, so we
# multiply by tier to land in the doc's range without changing the existing
# detector.  Boss highest because a single wasted turn there is most costly.
WASTEFUL_END_TURN_TIER_MULTIPLIER = {
    "weak": 1.5,    # → up to −0.15
    "normal": 1.5,  # → up to −0.15
    "elite": 2.5,   # → up to −0.25
    "boss": 3.0,    # → up to −0.30
    "unknown": 1.5,
}

# Full run: heavier penalties — wasted turns are costlier in context
FULL_RUN_WASTE_BASE = -0.08
FULL_RUN_WASTE_ENERGY = -0.04
FULL_RUN_WASTE_ZERO_COST = -0.08
FULL_RUN_WASTE_EXTRA_ACTION = -0.03
