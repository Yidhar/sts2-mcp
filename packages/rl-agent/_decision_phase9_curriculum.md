# Phase 9 Decision — Stop Shaping, Start Curriculum

**Date**: 2026-04-20
**Author synthesis**: empirical telemetry + Anthropic white paper (`_design_sts2_rl_whitepaper.md`) + codex research report + 6 reward-shape iterations
**Current commit**: `a8d532d` (potion v3 + hoarding penalty)

---

## 🚨 REVISION (2026-04-20 evening, after user correction)

**Original P1 ("Act 1 boss curriculum") is REDUNDANT.** The team has already run combat-sandbox curriculum extensively — with tiered difficulty (weak / normal / elite / boss) and tiered deck quality. Sandbox results are STRONG:

- **Overall boss win rate in sandbox: ~50%**
- **Individual boss win rate: up to 90%+**
- Combat subpolicy is functionally competent across all Act 1 bosses

Yet full-run boss_kill rate is still ~0%. This means the gap is **NOT combat competence**, it's:

1. **Sandbox → full-run transfer gap**: sandbox gives the policy a hand-crafted boss-fight starting state (specific deck, HP, relics, potions). Full-run policy produces its OWN starting states via 16 floors of macro decisions.
2. **The full-run policy arrives at floor 17 (when it arrives at all) with bad decks + low HP + wrong relics**, because the 4 macro decision classes are untrained in the sandbox-only curriculum:
   - **HP management** across floors (rest vs smith, route risk)
   - **Route selection** (elite timing, shop/rest access before boss)
   - **Deck construction** (which card_reward to pick, what to skip, what to remove)
   - **Event decisions** (HP/gold/deck tradeoffs)

**The correct Phase 9 is therefore NOT another combat curriculum.** It's:

- **Transfer the already-good combat skill** from sandbox checkpoint into the full-run policy (warmstart + freeze combat-related heads)
- **Train ONLY the macro decisions** with BC-calibration on Skada human data + potential-based shaping over deck quality + macro head split
- **KPI shift**: measure macro decision quality directly (card_reward alignment, route-to-boss survival, deck quality at floor 17) rather than relying on boss-kill as the only signal

See "Revised plan" section below. The original plan (kept for history) is now superseded.

---

## TL;DR

**Stop iterating on reward shaping.** Six shaping commits moved mean floor from 5.67 → 8.66 (+54%), but **Act 1 boss kill rate is mathematically bounded by Act 1 boss touch rate**, which is 2-5% regardless of how we tune potion bonuses. Next serious gain requires changing WHICH samples the policy learns from, not what reward they earn.

**Decision**: pivot to **Act 1 boss curriculum via `CombatSandboxEnv`**, plus **BC calibration excluding campfire** (protect reward-shaped behaviors). Estimated cost: 2-3 days engineering. Expected impact: **boss conversion conditional on reaching the boss** should move materially; full-run boss touch may improve only indirectly via combat transfer. The direct curriculum KPI should therefore be **sandbox boss survival / kill**, not "boss touch during curriculum pretrain".

> **Reviewer note (audit, 2026-04-20):**
> - I agree with the main pivot: stop micro-tuning shaping and move training signal upstream via curriculum.
> - The one wording issue to fix is **boss touch vs boss conversion**. A boss-only sandbox does **not** directly optimize `P(reach floor 17)`; it optimizes `P(kill boss | reach floor 17 state)` and maybe transfers back to late Act 1 hallway fights.
> - So Phase 9 should treat **boss touch** and **boss kill given touch** as separate KPIs, otherwise we risk declaring P1 a failure for the wrong reason.

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

> **Reviewer note:** freeze **reward magnitude iteration** by default, but do **not** freeze:
> 1. reward/telemetry instrumentation,
> 2. exploit fixes,
> 3. sandbox-only shaping if a curriculum-specific bug appears.
>
> In other words: stop policy-chasing by retuning constants, but keep the ability to patch correctness issues.

### Priority 1 — Act 1 boss curriculum (COMMIT THIS)

**Goal**: raise **Act 1 boss conversion** by giving the policy direct combat training against boss states, then fine-tune on full-run. In full-run metrics, the primary lift should be in `boss_kill | boss_touch`; any lift in `boss_touch` is a transfer bonus, not the direct target.

**Deliverable**: a new `launch_act1_boss_sandbox.sh` + a snapshot pool scoped to Act 1 boss encounters.

**Implementation**:
1. Build Act 1 boss snapshot pool from either:
   - Scraping Skada runs for "entered floor 17" states (most realistic starting deck/relic/hp)
   - OR synthesizing via sim: replay known-good Skada traces to floor 17, dump state each time
2. Use existing `CombatSandboxEnv` + `CombatSnapshotPool` (already built!)
   - **Reviewer note**: also reuse `launch_sandbox_starter_early_patch_20260418.sh` as the launch-script template instead of starting from a blank shell script.
   - **Reviewer note**: before writing a bespoke snapshot exporter, consider the lower-cost option of extending `train_attention_policy.py` / `resolve_snapshot_pool()` to accept `snapshot_min_floor`, `snapshot_max_floor`, and/or explicit `snapshot_encounter_ids`. That may cut the engineering scope for P1.
3. Train 200k steps against a **boss-heavy** Act 1 snapshot pool
   - `--combat-sandbox --snapshot-pool data/act1_boss_snapshots/`
   - **Reviewer recommendation**: do **not** make this 100% boss-only on day 1. Start with something like:
     - 70-80% Act 1 boss
     - 20-30% late Act 1 hard hallway / elite / pre-boss states
   - Rationale: pure boss-only pretrain risks overfitting to floor-17 deck states and may not transfer enough to floors 12-16, which are the states that determine `boss_touch`.
4. Fine-tune 200k steps on full-run warmstarted from sandbox checkpoint
   - 30/70 mix replay: 30% sandbox boss replays, 70% fresh full-run rollouts
   - **Reviewer note**: this mixing policy is directionally good, but I do not think it is already present as a first-class training primitive in `train_attention_policy.py`. Treat it as an implementation task, not as "free because sandbox exists".

**Expected**: `boss_kill | boss_touch` should move first; overall boss-kill rate can plausibly go 0% → 5-15% only if touch is preserved or modestly improved during fine-tune.

**Budget**: 2 days engineering (snapshot pool), 4-6h compute (sandbox), 10h (fine-tune).

**Fallback if boss-kill still <5%**: it's a combat-skill ceiling, not exposure. Escalate to Priority 4.

> **Reviewer note:** if P1 fails, distinguish between:
> - **sandbox boss kill is good, full-run touch still low** → the bottleneck is not boss combat, it is macro / pre-boss survival / route / rewards;
> - **sandbox boss kill is still poor** → then yes, escalate to combat-skill / belief-memory work.

### Priority 2 — BC calibration (excluding campfire)

**Commit pipeline ready**: `skada_bc_train.py --init-checkpoint ... --phase-filter map,card_reward,relic_relic,relic_ancient`.

**Why exclude campfire**: dry-run showed PPO `campfire_acc = 0%` vs human (our rest-HP shaping works; humans disagree by picking SMITH more often). BC would undo the shaping.

**Budget**: 30 min (500k samples on WSL+ROCm). Follow with 10k sim verify.

**Expected deltas** (from dry-run log-prob positive):
- card_reward acc: 34.5% → 45%+
- relic_relic: 32.3% → 45%+
- relic_ancient: 57.1% → 65%+

> **Reviewer note:** before running BC, dump a **phase histogram** from the Skada sample source. I want to avoid "map/proceed" rows numerically swamping the more valuable `card_reward` / `relic_ancient` supervision. If the mix is badly imbalanced, cap `map` or run per-phase quotas.

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

> **Reviewer note:** I agree with the direction, but I would **not** bundle P3 into P1 unless attribution clarity is unimportant. P1/P2 already change the sample distribution; adding head-split in the same window makes it much harder to know whether curriculum or architecture caused the gain. My preference: **P1 + P2 first, P3 after the first clean comparison**.

### Priority 4 — Conditional on P1 failing: recurrent belief memory

Only if boss-kill rate still <5% after curriculum. Details in codex report §2.1 / Anthropic §3.

**Budget**: 1 week. Mostly backbone change.

### Priority 5 — Explicit STS2 mechanic encoding (long-horizon)

From codex report: add encodings for Ancients / Enchantments / Afflictions / Unknown-room history / Act branches. Not urgent for the full long-tail refactor, but **some** of this is already relevant for Ironclad.

> **Reviewer note:** I would split this into:
> - **P5a (cheap, do now/soon):** explicitly tag `ancient` as a screen / phase in telemetry and policy routing, because Ancients affect Ironclad too and are high-leverage decisions.
> - **P5b (larger refactor):** full per-card enchantment / affliction / branch-history schema work.

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
| Full-run boss kill given boss touch | materially up vs baseline |
| Full-run boss_touch | non-decreasing (nice-to-have: moderate lift) |
| Full-run fine-tune mean_floor | > 12 **or** clear gain in late-Act1 survival metrics |
| Full-run boss_kill | > 2% |

> **Reviewer note:** `mean_floor > 12` is a useful headline metric, but it should not be the only pass/fail gate for a **boss-centric** curriculum. If sandbox boss kill jumps and `boss_kill | boss_touch` jumps, but mean floor only moves modestly, that is still evidence P1 worked.

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

> **Reviewer recommendation:**
> 1. Start with **sim-synthesized** or existing curated subsets for speed, prove the loop, then backfill with Skada-scraped floor-17 states if the first sandbox result is promising.
> 2. Run **P2 in parallel** because it is cheap, but do not let it block P1.
> 3. **Defer P3** until after the first clean P1/P2 comparison unless you explicitly optimize for total wall-clock over experimental attribution.

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

---

# Revised plan (2026-04-20 evening)

Context: combat sandbox curriculum has ALREADY been run extensively and succeeded. Combat subpolicy is fine. The real problem is macro-decision transfer from sandbox (hand-crafted starting states) to full-run (self-produced starting states).

## 🚨 SECOND REVISION (2026-04-20, later) — sandbox wins were MuZero era

User clarified: **the 50% / 90% combat sandbox results are from the retired MuZero / MCTS-based architecture** (checkpoints named `muzero_combat_*`). After pivoting to the current PPO + attention architecture (Phase 6 → Phase 8), **combat-only sandbox training has never been run**.

This invalidates the "freeze combat heads" path as written:

- ❌ We can't warmstart from MuZero — different algorithm, different network topology, state-dict incompatible
- ❌ We can't assume current PPO is combat-competent at floor-17-difficulty states — it's only been measured implicitly through floor reach (2-5% touch 17, and those 45 episodes at floor 17 are heavily filtered for "made it that far", not random boss-fight starting states)
- ❌ The "P1' freeze-combat-heads" plan requires competent combat heads to freeze, which we don't actually have on the current arch

Three possible interpretations, needs data:

1. **Current PPO combat IS competent** — it's just diluted by macro gradient in full-run training. Easy to verify: run current checkpoint on combat sandbox, measure boss win rate. If >30%, proceed with freeze-heads plan.
2. **Current PPO combat is weak** — the 2-5% boss touch reflects random survival, not skill. Need to actually train combat sandbox on current arch first.
3. **Combat and macro are co-dependent** — can't really be trained separately on this arch. Need end-to-end with better exploration / longer horizons / potential-based shaping.

**We need to measure before committing to any of the three.**

## The four macro decisions that break in full-run

Each corresponds directly to a phase the Skada BC dataset covers:

| Decision class | Phase | Sandbox coverage | Full-run policy status | Skada samples available |
|---|---|---|---|---|
| HP management | `campfire` + implicit in `combat` | Starting HP given | Rest-HP gate overcorrects (100% HEAL, 0% SMITH); doesn't trade off | 187k (excluded from BC to protect shaping — but SEE below) |
| Route selection | `map` | Starting node given | Greedy; doesn't plan for boss proximity / pre-boss rest | 877k |
| Deck construction | `card_reward` | Starting deck given | 34% human agreement; picks low-value attacks, skips power cards | 527k |
| Event decisions | `event_option` | N/A | Largely untested in current training signals | Mixed (part of map/proceed samples) |
| Relic selection | `relic_relic` / `relic_ancient` | Starting relics given | 32% / 57% human agreement | 314k / 66k |

Total Skada non-combat samples: **1.97M** of exactly the decisions the agent is failing on.

## New priority order (replaces original P1 / P2 / P3)

### P-1 (NEW, MUST DO FIRST) — Measure current-arch combat competence

**Goal**: stop assuming. Take the current Phase 8 checkpoint
(`sim_phase8_longtrain_820k_20260420_000838/step_000819200`), run it against a curated combat sandbox slice covering Act 1 bosses + late Act 1 elite/hallway, and report boss win rate by encounter. This is the gate that decides which of three paths we're actually on.

**Implementation**:
- Reuse existing `evaluate_attention_policy.py --combat-sandbox` (already supports sandbox mode)
- Need a curated Act 1 boss snapshot pool; reviewer note #4 says extend `resolve_snapshot_pool` to filter by `min_floor=17 / max_floor=17` rather than write a new exporter
- Run 50-100 episodes per boss type (Ceremonial Beast / Vantom / The Kin / Waterfall Giant / Soul Fysh / Lagavulin Matriarch — both Overgrowth and Underdocks branches per codex §1.3.1)
- Report: win rate, avg turns to kill, avg HP remaining on win, avg HP on loss

**Budget**: 1-2 hours engineering (pool filter flags + eval script tweaks), 30 min compute.

**Decision tree based on P-1 result**:

| Boss win rate | Interpretation | Next path |
|---|---|---|
| ≥30% | Combat IS competent, just diluted in full-run gradient | Go P1' (freeze combat, train macro) — but with sandbox-eval-as-regression-test during full-run training |
| 10-30% | Combat is weak — floor 17 reach was mostly luck | Go P0'' (actually train combat sandbox on current arch first) |
| <10% | Current arch has combat problems we haven't seen | Debug first: attention architecture capacity? aux head alignment? Before any curriculum. |

### P0'' (conditional on P-1 showing <30%) — Combat sandbox training on current arch

**Only if P-1 says combat needs work.** Train PPO on combat-sandbox-only rollouts, 200-400k steps, warm-start from current Phase 8 checkpoint. Objective: get boss win rate to 30%+ before moving to macro-focused work.

**Implementation**:
- `train_attention_policy.py --combat-sandbox --snapshot-pool <filtered to Act 1 bosses>`
- Use the same filter flags we add in P-1 (`min_floor=17 / encounter_ids=<boss list>`)
- Keep reward shaping as is (it's already combat-aware via boss-damage multiplier)
- Budget: 10-15h compute + 2h engineering

### P1' — Sandbox-to-full-run transfer via combat-head freezing

**Goal**: when running PPO on full-run, freeze the combat subpolicy (which already works from sandbox) and pour gradient into macro heads only.

**Why**: with single-shared-policy architecture, the COMBAT gradient dominates training volume (100+ combat steps per episode vs 5-8 macro decisions). The macro heads receive diluted signal even after shaping. Separating the gradient pressures eliminates this dilution.

**Implementation**:
1. Warmstart full-run PPO from the best sandbox checkpoint (which already handles bosses at 50-90%)
2. Freeze parameters affecting combat action scoring:
   - `combat_head` outputs (play_card / end_turn / use_potion candidate scoring)
   - Power/history/enemy/runtime world banks (combat-context encoders)
   - POWER_SLOT relation bias
3. Train ONLY:
   - Route/reward/shop/rest/event/ancient candidate scoring paths
   - build/route aux heads
   - value head (need to re-fit since full-run returns differ from sandbox)

**Risk**: freezing combat might break when full-run produces novel boss-fight starting states the sandbox didn't cover. Mitigation: periodic unfreeze-and-retrain-combat passes if sandbox kill rate degrades.

**Budget**: ~2 days engineering (freeze mechanics + retrain loop), 10-20h compute.

**KPI**:
- Sandbox boss kill rate (continuous check): stays ≥40%
- Full-run mean_floor: 8.7 → 11+
- Full-run deck quality at floor 17 (NEW metric): measurable via aux_build head

### P2' — BC calibration on the 4 macro phases (elevated from original P2)

**Now the top quick-win.** 500k samples on WSL+ROCm, ~30 min. Excludes campfire by default to protect rest-HP shaping, but with reviewer note #8 sampling balance:

```bash
# Per-phase sample cap to prevent map (877k) from swamping relic_ancient (66k)
--phase-filter map,card_reward,relic_relic,relic_ancient
--phase-cap 100000  # NEW flag: max samples per phase
```

This requires a small extension to `SkadaBcDataset` to track per-phase counts — reviewer note #8 is correct.

**Expected deltas** (post-calibration, measured on held-out):
- card_reward acc: 34.5% → 45-55%
- relic_relic: 32.3% → 45-55%
- relic_ancient: 57.1% → 65-70%
- campfire: 0% (unchanged — by exclusion)

### P3' — Macro head split (deferred per reviewer note #9)

Defer until after P1' + P2' show clean comparison. Doing it now confounds attribution.

### P4' — Conditional on macro decisions still weak: deck-quality potential shaping

From Anthropic white paper: potential-based shaping `Φ(s) = deck_quality × boss_distance × floor`. Continuous gradient signal on "deck is getting better" every time a card is added/removed/upgraded.

Only deploy after P1'+P2' if macro alignment is still under 50%.

### P5' — Ancient screen as distinct phase (cheap, do alongside P3')

Codex report + reviewer note #11: tag `ancient` as distinct screen in telemetry + policy routing. Affects Ironclad too, mandatory high-leverage decisions. ~1 hour change.

## 🎯 THIRD REVISION (2026-04-20, analysis/ + tmp/offline_dataset_full/ discovered)

User pointed to pre-existing evaluation and dataset artifacts that change P-1 into a data look-up rather than a new experiment.

### Eval data already exists — P-1 is effectively answered

From `analysis/training_curves/`:

| Checkpoint | Arch | Boss win rate | Elite | Normal | Weak | Samples |
|---|---|---|---|---|---|---|
| `muzero_step_00251923` | MuZero (retired) | **29.4%** | 76.0% | 86.9% | 94.3% | 1,915 |
| `longtrain_ironclad_ckptrot_bridgefix_20260417_010123` | PPO+attention (Phase 6) | **15.4%** | 48.2% | 74.2% | 90.7% | 3,036 |
| `sandbox_starter_early_patch_20260418_122809/step_000450560` | PPO+attention sandbox | untested on boss (20-ep benchmark, weak/normal/elite only) | — | — | — | 20 |

**User's recollection was imprecise but the direction was right.** MuZero boss win was 29% (not 50%); current PPO is at 15% (not 0%). Both imply **combat subpolicy is not the critical bottleneck** — the 14pp gap between MuZero and current PPO is worth closing but not the 300× gap between 15% combat win and 0.05% full-run boss kill.

**Math confirms macro is the ceiling**:
```
full-run boss_kill = P(reach floor 17) × P(win boss | reached)
                   = 0.047 × 0.154 (theoretical max)
                   = 0.72% (actual observed: 0.05%, i.e. 14× worse)
```
Even if we push combat from 15% → 29% (MuZero level), full-run boss_kill cap is still 1.4% without macro improvement. To reach a 10% full-run boss_kill, macro survival must raise `P(reach floor 17)` from ~5% to ~35%.

**Decision**: skip P-1 re-measurement, proceed directly to macro-focused work.

### Better BC dataset discovered: `tmp/offline_dataset_full/`

Previously I was targeting `data/skada_bc/samples.jsonl` (1.97M samples, phase-mixed). The offline_dataset_full is materially better for this job:

- **Already split by decision type** — no `--phase-filter` complexity
- **Enriched context fields** — `deck_before/after`, `hp_before`, `relic_ids_before`, `quality_flags`, `supervision_type`, `split` (train/val)
- **Quality-labeled** — `quality_flags` can filter to demonstrations known to lead to successful runs (via runs_summary.jsonl)
- **Totals fit the training goal**:
  - route: 6,262 (full paths)
  - card_choice: 3,425 (card_reward decisions)
  - event_choice: 1,821
  - potion_choice: 1,537
  - ancient_choice: 956
  - rest_site: 427
  - shop: 353
  - relic_choice: 407
  - card_remove + upgrade + transform: ~670

Total 23k high-quality macro samples (excluding build/floor_records/decision_records which are derived).

**Implication**: existing `skada_bc_train.py` is less useful than building a macro-oriented BC trainer on `tmp/offline_dataset_full/`. The translation layer (`skada_bc_translate.py`) will likely need extension per decision type.

### Sandbox starter-early checkpoint exists but untested on bosses

`checkpoints_attention/sandbox_starter_early_patch_20260418_122809/step_000450560` is a Phase 6 PPO+attention checkpoint that DID undergo combat-sandbox training. Its boss-tier win rate has never been properly measured (only 20-episode benchmarks exist, none covering boss tier).

**Action**: before P2' (BC), run a proper boss-tier eval on this checkpoint. This settles two questions at once:
1. Does sandbox pretrain meaningfully lift boss win above the 15.4% baseline?
2. If yes, is it worth warmstarting from THIS checkpoint rather than the Phase 8 long-train?

Budget: 1 hour compute using existing `analysis/evaluate_attention_checkpoint_benchmark.py` with a larger N (200 eps across tiers).

## Revised-revised-revised immediate actions (after MuZero clarification)

**The MuZero-era sandbox checkpoints are not usable** — different algorithm, different state dict. We must establish combat competence on the current PPO arch from scratch or confirm it's already there implicitly.

Three concurrent tracks:

### Track A (diagnosis-first, zero-risk): run P-1 now

1. **Extend `resolve_snapshot_pool`** with `--snapshot-min-floor / --snapshot-max-floor / --snapshot-encounter-ids` flags (reviewer note #4). ~1 hour. Zero behavioral change; just filtering.
2. **Run P-1 evaluation** against current Phase 8 checkpoint using the new filter flags. ~30 min compute. Report per-boss win rate.
3. **Let the result decide** whether we go P0'' (combat training) or directly to P1' (freeze + macro).

### Track B (parallel, cheap quick-win): P2' BC calibration

Independent of P-1 / P0'' decisions. Runs in parallel on WSL+ROCm.

1. **Extend `skada_bc_train.py` with `--phase-cap` flag** (reviewer note #8). 20 min code change.
2. **Dump phase histogram** from `data/skada_bc/samples.jsonl` first to decide caps.
3. **Run BC calibration**: 500k samples, per-phase capped, 30 min.
4. **Verify with `eval_bc_calibration_delta.py`** on 5k held-out slice.
5. **Keep the calibrated checkpoint** as a candidate starting point for whichever path Track A chooses.

### Track C (implementation prep): freeze-heads infrastructure

Regardless of whether P-1 says combat is strong or weak, we'll eventually want the freeze-heads capability. Prep the infrastructure now:

1. Design `freeze_submodules` arg for the policy class, parameter-name patterns for "combat path" vs "macro path"
2. Add a `--freeze-patterns` CLI flag that marks matched parameter tensors with `requires_grad=False`
3. Add a matching `bc_freeze_patterns` concept to `skada_bc_train.py` so we can fine-tune ONLY specific heads

Budget: 1 day engineering. Doesn't commit to a path, just enables both P1' and P0''-then-P1'.

## Open questions (second revision)

1. **Is there any evaluation output from the retired MuZero `muzero_combat_*` runs?** Its boss win rates might inform whether the difficulty is arch-specific or inherent to STS2 combat mechanics.
2. **Does current Phase 8 `evaluate_attention_policy.py --combat-sandbox` actually produce per-encounter win rates?** Or does it just return aggregate floor reach? We need per-boss granularity for P-1.
3. **What's the largest curated `data/curated_combat_snapshot_*` bucket available for Act 1 bosses?** If existing curated subsets already cover this, we skip the "extend resolve_snapshot_pool" work and just use `--snapshot-curated-subset act1_boss` directly if it exists.
