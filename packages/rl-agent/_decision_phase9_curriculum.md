# Phase 9 Decision — Stop Shaping, Start Curriculum

**Date**: 2026-04-20
**Author synthesis**: empirical telemetry + Anthropic white paper (`_design_sts2_rl_whitepaper.md`) + codex research report + 6 reward-shape iterations
**Current commit**: `a8d532d` (potion v3 + hoarding penalty)

## TL;DR

**Stop iterating on reward shaping.** Six shaping commits moved mean floor from 5.67 → 8.66 (+54%), but `Act 1 boss kill rate is mathematically bounded by `Act 1 boss touch rate`, which is 2-5% regardless of how we tune potion bonuses. Next serious gain requires changing WHICH samples the policy learns from, not what reward they earn.

**Decision**: pivot to **Act 1 boss curriculum via `CombatSandboxEnv`**, plus **BC calibration excluding campfire** (protect reward-shaped behaviors). Estimated cost: 2-3 days engineering. Expected impact: boss touch rate from 2-5% → 30-50% within curriculum pretrain, boss kill rate 0% → 5-15% after fine-tune.

## Evidence consolidated

### Empirical trend (6 reward-shape iterations)

| commit | patch | mean_floor | boss_touch | boss_kill | observation |
|---|---|---|---|---|---|
| pre-fix | Phase 8 skeleton | 5.67 | 0% | 0% | stuck_rate 46% |
| cd791a0 | card_selection filter | 7.73 (+36%) | 1.3% | 0% | stuck_rate 0% |
| 23cd0a6 | 820k baseline | 7.82 | 2.0% | 0.05% | first floor-18 |
| 6643917 | floor-clear + boss-damage mult | 8.29 | 3.1% | 0.35% | trending up |
| 65b5196 | rest-HP + potion base | 8.47 | — | — | — |
| 84f8ca6 | potion monster = 0 | 8.73 | 4.1% | 0% | 2 boss touches |
| **a8d532d** | **potion v3 + hoarding** | **8.66** | **5.7%** | **0%** | 100% 0-potion end, 1.9% boss-use |

**Diminishing returns evident**: each iteration gives smaller floor improvement. Mean floor appears to asymptote around 8.5-9.0 under current architecture + shaping.

### Math ceiling (user-identified, 2026-04-20)

The "boss-use as fraction of total potion use" target cannot exceed the boss-touch probability ratio:

```
Let p = boss-touch rate per episode
Let u = avg potion uses per episode
Rational policy saves 1 for boss if reachable:  boss_use_count ≈ p × 1
max boss_use / total ≈ p / u
With p = 0.057 (current) and u = 2.2 (current): max = 2.6%  (observed: 1.9%)
```

**Policy is already near-optimal given its boss-touch rate.** To hit 30% boss-use: p must rise to ~0.7 OR u must drop to ≤1.

**Shaping can't raise p significantly** — it can only tune the relative reward of reaching vs not reaching boss, which the 6 iterations have mostly exhausted. Raising p requires the policy to SURVIVE floors 5-16 better, which requires **better combat decisions**, which requires **more combat training signal** from the phases where episodes currently die.

## Independent convergence of both white papers

### Anthropic white paper (`_design_sts2_rl_whitepaper.md`, 35.9 KB)

Top-3 recommendations:
1. **Reward refactor + BC warmstart** — potential-based shaping + Skada pretrain
2. **Dual value head + ent_coef cosine schedule** — split combat_v / run_v
3. **Act1-boss curriculum + Tier 3 causality**

### Codex white paper (received in user message)

Top-8 upgrade ROI order:
1. Keep 412-token + 7-bank (no rewrite)
2. **Specialize training: boss sandbox first, then global**
3. Recurrent belief memory
4. Learned deck critic
5. Route encoder graph structure
6. Version-aware training
7. **Macro head split (combat/route/reward/shop/rest/event/ancient)**
8. MuZero only as combat research side-branch

### Intersection (both agree)

- **Architecture NOT the bottleneck** — 412-token attention + 8 aux heads is sound
- **Act 1 boss curriculum is top priority** — both reports rank this 1st or 2nd
- **BC warmstart from Skada** — high ROI, already pipeline-ready
- **Macro decision head split** — medium ROI, clarifies which phase each loss improves
- **Don't pivot to MuZero yet** — pivot cost >> incremental improvement cost

### New STS2-specific findings from codex report (NOT in Anthropic report or our code)

These are mechanics we're currently missing in the observation:

| Mechanism | Current state | Missing impact |
|---|---|---|
| **Ancients** (Act-transition mandatory 3-choice, can't skip) | Not modeled as distinct screen | High-leverage cross-act value decision |
| **Act 1 branches** (Overgrowth / Underdocks) | Not encoded | Same floor/deck has different encounter distribution |
| **Enchantments** per-card (Corrupted/Glam/Spiral/Imbued/...) | Not in card entity fields | Same `card_id` is no longer the same decision |
| **Afflictions** (Bound/Entangled/Smog/Ringing/Hexed) | Not tracked in action legality | Dynamically rewrites legal action space |
| **Easy/hard pool per act** | Not encoded | Opening floors are not IID with later floors |
| **Unknown room history-adaptive probability** | Not tracked | Current ? node value depends on ?-history |
| **Necrobinder Doom / Osty / Soul** | Not character-specific encoded | Execution threshold mechanic invisible to policy |
| **Silent Sly** (discard → auto-play cascade) | Not in action consequence model | Policy can't reason about discard-triggered auto-plays |
| **Defect Glass Orb** internal value | Not tracked | Orb state beyond slot+focus |
| **Patch version / content_hash** | Not in obs | Training distribution non-stationary during EA |

**We're training on `ironclad` only right now** — most of these don't bite yet. But if we ever go multi-character or across-patch, they will.

## Decision: Phase 9 plan

### Priority 0 — Stop reward shaping iteration

Freeze `reward_constants.py` at `a8d532d` values. 6 iterations was enough; diminishing returns confirmed. Further tuning is zero-ROI relative to the work below.

### Priority 1 — Act 1 boss curriculum (COMMIT THIS)

**Goal**: raise Act 1 boss touch rate from 5.7% to 50%+ by giving the policy direct combat training against bosses, then fine-tune on full-run.

**Deliverable**: a new `launch_act1_boss_sandbox.sh` + a snapshot pool scoped to Act 1 boss encounters.

**Implementation**:
1. Build Act 1 boss snapshot pool from either:
   - Scraping Skada runs for "entered floor 17" states (most realistic starting deck/relic/hp)
   - OR synthesizing via sim: replay known-good Skada traces to floor 17, dump state each time
2. Use existing `CombatSandboxEnv` + `CombatSnapshotPool` (already built!)
3. Train 200k steps against Act 1 boss snapshots ONLY
   - `--combat-sandbox --snapshot-pool data/act1_boss_snapshots/`
   - Weight: 100% Act 1 boss (no hallway dilution)
4. Fine-tune 200k steps on full-run warmstarted from sandbox checkpoint
   - 30/70 mix replay: 30% sandbox boss replays, 70% fresh full-run rollouts

**Expected**: boss-kill rate 0% → 5-15% within 400k sandbox + fine-tune.

**Budget**: 2 days engineering (snapshot pool), 4-6h compute (sandbox), 10h (fine-tune).

**Fallback if boss-kill still <5%**: it's a combat-skill ceiling, not exposure. Escalate to Priority 4.

### Priority 2 — BC calibration (excluding campfire)

**Commit pipeline ready**: `skada_bc_train.py --init-checkpoint ... --phase-filter map,card_reward,relic_relic,relic_ancient`.

**Why exclude campfire**: dry-run showed PPO `campfire_acc = 0%` vs human (our rest-HP shaping works; humans disagree by picking SMITH more often). BC would undo the shaping.

**Budget**: 30 min (500k samples on WSL+ROCm). Follow with 10k sim verify.

**Expected deltas** (from dry-run log-prob positive):
- card_reward acc: 34.5% → 45%+
- relic_relic: 32.3% → 45%+
- relic_ancient: 57.1% → 65%+

### Priority 3 — Macro head split (both reports agree)

Currently all decisions use a single candidate-conditioned policy head with 8 aux heads on top. Codex recommends, Anthropic dual-value-head converges:

**Change**: split `candidate_scorer` into screen-specific heads:
- `combat_head` (play_card / end_turn / use_potion)
- `route_head` (map / proceed)
- `reward_head` (card_reward / skip)
- `shop_head` (buy / skip)
- `rest_head` (heal / smith)
- `event_head` (event_option)
- `ancient_head` (Ancient boon choice — **new, from codex report**)

**Budget**: ~150 lines policy change + checkpoint migration. 1 day.

**Expected**: each head's gradient stays focused on its phase's reward signal. Value loss on specific phases improves 15-30% (per Anthropic report).

### Priority 4 — Conditional on P1 failing: recurrent belief memory

Only if boss-kill rate still <5% after curriculum. Details in codex report §2.1 / Anthropic §3.

**Budget**: 1 week. Mostly backbone change.

### Priority 5 — Explicit STS2 mechanic encoding (long-horizon)

From codex report: add encodings for Ancients / Enchantments / Afflictions / Unknown-room history / Act branches. Not urgent for Ironclad early-training but critical if we ever tackle Regent / Necrobinder / A10+.

**Budget**: 1-2 weeks observation schema refactor. Lower priority until P1-P3 settle.

## What NOT to do

- ❌ Iterate reward shaping further — diminishing returns proven, math ceiling hit
- ❌ Pivot to MuZero — both reports converge: premature. 6-month cost, unclear incremental value over curriculum
- ❌ Restart fresh training — architecture is sound, just needs boss exposure
- ❌ Add BC WITH campfire — would erode the rest-HP gate we worked to fix

## Evaluation gates

### After Priority 1 (Act 1 boss curriculum)

| Gate | Target |
|---|---|
| Act 1 boss survival rate (sandbox) | > 50% |
| Act 1 boss kill rate (sandbox) | > 30% |
| Full-run fine-tune mean_floor | > 12 |
| Full-run boss_kill | > 2% |

### After Priority 2 (BC calibration)

| Gate | Target |
|---|---|
| card_reward acc delta | +10pp |
| relic_relic acc delta | +13pp |
| campfire acc | unchanged |
| combat_stuck_rate | unchanged or improved |

### Overall Phase 9 success

Boss kill rate 0% → 5%+ over 800k step retrain.

### If Phase 9 gates fail

Escalate to Priority 4 (recurrent belief memory) OR Priority 5 (STS2 mechanic encoding) depending on failure mode:
- Combat-stuck in boss fights → belief memory (can't reason about multi-turn intent cycles)
- Wrong card_reward / Ancient choices → STS2 mechanic encoding (missing game knowledge)

## Open questions for user

1. **Act 1 boss snapshot source**: Skada scrape (realistic but 1-2 days build) or sim-synthesized (fast but deck/relic distribution may differ)?
2. **P1 vs P2 order**: run curriculum first (bigger impact, higher cost) or BC first (quick win, lower ceiling)?
3. **P3 timing**: bundle macro head split with P1 checkpoint migration, or defer to Phase 10?

## Immediate commit plan

Once this decision is acknowledged:
1. Commit this doc
2. Build `launch_act1_boss_curriculum.sh` with snapshot pool reference
3. Write `build_act1_boss_snapshot_pool.py` (scrape Skada → dump snapshots in `CombatSnapshotPool` format)
4. Start BC calibration on WSL+ROCm in parallel (independent work)

## References

- `_baseline_report_20260420.md` — 6-run telemetry
- `_design_sts2_rl_whitepaper.md` — Anthropic agent 35.9 KB research
- (external) codex STS2 research report — user pasted 2026-04-20
- `_design_phase8_history.md` — Phase 8 Tier 1+2 architecture rationale
- `_handoff_phase8_smoke.md` — Phase 8 smoke gate definitions
