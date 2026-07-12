# STS2 RL 修复实施计划 (2026-06-02)

> 由 6 路只读代码分析 + 汇总生成；针对 `_analysis_rl_diagnosis_20260602.md` 里已核实的根因。**计划，待审批后实施。**

# MuZero Act-1-Clear Recovery: Integrated Implementation Plan

## Strategy (one paragraph)

**Verify-first, persist learning, fix root causes before amplifiers, stay curriculum-staged.** Land the cheap read-only verification harness and the missing telemetry emissions *first* so every later fix is measurable on the primary async path and so the kill-every-1-3h compounding loss is quantified before we try to eliminate it. Then make learning *persist* across restarts (RC-1: replay/optimizer survive resumes + guard/head edits) — this is the prerequisite enabler, because no downstream reward/exploration/guard fix can compound while every restart drops the 2.3GB buffer. Then fix the two *correctness* root causes that corrupt the learning signal itself — RC-4 (guards rewriting the policy target to one-hot) and RC-5 (reward has no win/act-clear signal and a 3:1 HP asymmetry) — since amplifying a corrupt signal is worse than not amplifying. Only *after* the signal is honest and persistent do we add the *amplifiers* — RC-3 (exploration entropy + real value-improvement bootstrap) and RC-2 (throughput via grad-accum, aux-loss rebalance, instrumented spike skip, stronger latent reg). All new behavior ships behind default-no-op flags so each step is independently revertible and A/B-able on the proven-learnable curated-deck sandbox before touching full-run.

**All anchors below were re-verified against the live working tree** (checkpointing.py 131-203, cli_args.py 33/65-66/124-138/262/290-293, cli_main.py 393/453/479/496-497/777/829/855/1189, self_play.py 484-487/613/642-752/940-961/1446-1477/1702-2326, train_step.py 771-780/907-934, trainer.py 257-417, env_v2.py 549-589, reward_constants.py 12-13/29-35/131). The fix-spec line numbers are accurate; `trajectory.add_step` confirms `action=` (executed) is already stored separately from `search_policy=` (target) and `root_value=` — which is exactly what makes RC-4 and RC-3 clean.

---

## Ordered Steps (checklist)

### STEP 0 — VERIFY-FIRST telemetry + scripts (do this first, it is the gate for everything)

**Quick code (≈5-line additive) + new read-only scripts. No behavior change.**

- **0a (CODE, ~5 lines):** `self_play.py` ~2307 — add `"intent_combat_quality_metrics": intent_combat_quality_meta,` into the `self.last_episode_metrics = {...}` dict (alongside `summoner_targeting_metrics`). The meta is already computed at line 1702 and only emitted on the SYNC path (2248); the async (primary) path never sees it.
- **0b (CODE, ~8 lines):** `async_telemetry.py` ~409 — add an `intent_combat_quality_metrics` replay branch mirroring the `target_priority_metrics` block, importing `INTENT_COMBAT_QUALITY_TB_KEYS` and emitting `combat/intent_quality_*` on `trainer.writer`.
- **0c (CODE, ~4 lines):** `train_step.py` 913-922 — inside the existing `spike_detected_in_step` branch add `writer.add_scalar("train/loss_spike_skipped", 1.0, total_steps)` and `train/loss_spike_skip_count`, null-checking `getattr(self,"writer",None)` (async actors use a null writer).
- **0d (SCRIPT, new):** `scripts/winrate_resume_dashboard.py` — concatenate N `logs_muzero/*` run dirs by `total_steps`, plot `recent_tail/256/{win_rate,act1_boss_seen_rate,act1_pass_rate}` and `buffer/size`; ALERT (exit nonzero) when a post-resume boundary shows boss_rate < 0.5× prior AND `buffer/size` dropped below `--min-buffer`. This is the RC-1 detector. Reuse `monitor_fullrun_act1_gate.summarize_scalar`.
- **0e (SCRIPT, new):** `scripts/ab_guards_off_vs_full.py` — diff two run dirs (or launch two matched short runs differing only by `--combat-hard-guard-policy off|full`); print side-by-side `recent_tail/256/win_rate|reward_mean|act1_boss_seen_rate`, `combat_guards/hard_guard_override_any_rate`, `search/combat/post_search_hard_guard_policy_retargeted_rate`; refuse a verdict below a min-episode gate (≥500/arm). This is the RC-4 test.
- **0f (SCRIPT, new):** `scripts/planner_sanity_probe.py` — load a checkpoint, replay a confirmed-lethal frame from `intent_combat_quality.jsonl`, call `network.action_rollout_planner`, assert `planner_q[lethal] - planner_q[end_turn] >= margin`. This is the RC-3 baseline metric. Replay a *real* frame, do NOT hand-build encodings (OOD false-fails).

**Config:** none (read-only / additive).
**GATE:** Tool 0d flags ≥1 known historical resume where boss_rate collapsed and buffer dropped (validates the detector against real data); `combat/intent_quality_*` tags now appear on an async run; `train/loss_spike_skip_count` visible in TB; baseline planner margin recorded. **Effort: M.**
**WIP interaction:** purely additive — does not touch the dev's offline/human alignment metrics or the `post_search_policy_retarget` telemetry; re-confirm the `last_episode_metrics` and spike-branch line anchors against the live tree before editing (both files are in the dev's modified set).

---

### STEP 1 — RC-1: Make replay buffer + optimizer survive restarts and guard/head edits (PREREQUISITE)

**Real code change. The single most important step — without it nothing compounds.**

- **1a (CODE) `cli_args.py` 133-138:** replace `--resume-without-buffer`/`--resume-without-optimizer` (store_true) with `--resume-load-buffer`/`--resume-load-optimizer` as `argparse.BooleanOptionalAction default=True`; keep the two old flag names as deprecated `store_false` aliases on the same dests (force-skip for muscle-memory/launch scripts).
- **1b (CODE) `cli_main.py` 496-497:** pass `load_buffer=bool(args.resume_load_buffer)`, `load_optimizer=bool(args.resume_load_optimizer)`; after the `[resume] Loaded ... buffer=<N>` line, print an explicit `[resume][WARN] Buffer requested but loaded EMPTY` so a cold start is never silent.
- **1c (CODE) `checkpointing.py` 177:** **decouple buffer load from `allow_exact_resume`** → `allow_replay_buffer_load = load_buffer and replay_schema_ok` (drop the `allow_exact_resume` conjunct). Buffer is gated ONLY on `_metadata_replay_compatible` + `_replay_state_compatible`, which guard/head edits never touch.
- **1d (CODE) `checkpointing.py` 164-175 + new helper:** replace all-or-nothing optimizer load with `_load_optimizer_compatible(optimizer, saved_state)` that copies Adam moments per-param where `tuple(shape)` matches and re-inits the rest; returns `(loaded, reinit)` for logging. Preserves momentum for the unchanged ~99% of the 54M-param model when a guard edit resizes one head.
- **1e (CODE) `checkpointing.py` 49-61 + new shim:** add an (initially empty) `_REPLAY_MIGRATIONS` registry + `_migrate_replay_state`; when only `checkpoint_compatibility_version` differs and a migration exists, run it before `buffer.load_state_dict`; else warn + empty (never crash). Strictly better than today's silent drop.
- **1f (CODE) `cli_main.py` 1189 + new flag:** add `--checkpoint-min-interval-s` (default 1200) wall-clock guard so a checkpoint is forced every N minutes regardless of step count, and register a `SIGTERM`/`atexit` `shutdown` checkpoint (wrap in try/except for OOM). Makes the actual 1-3h hard-kill failure mode lose ≤20 min instead of up to `checkpoint_freq` steps.

**Config flags:** `--resume-load-buffer`/`--no-resume-load-buffer`, `--resume-load-optimizer`/`--no-resume-load-optimizer` (default load=True), `--checkpoint-min-interval-s 1200`. **Stop passing the deprecated `--resume-without-*` flags in launch scripts.**
**GATE (run Tool 0d before AND after):** `buffer/size` must grow monotonically *across a restart* instead of dropping to ~0; a resume with a mutated network head must keep `len(buffer)>0` and log `Optimizer: restored N / re-initialized M`. The RC-1 unit regression (save → mutate one head dim → reload → assert buffer survived, optimizer partial-loaded, no exception) must pass. **Effort: M. Depends on: Step 0 (need the dashboard to confirm the fix).**
**WIP interaction:** the dev's uncommitted `checkpointing.py` edits (+11 lines) may already touch this region — **rebase onto the working tree** and converge on the single rule: *buffer gated on replay schema only; optimizer gated per-param-group*. Confirmed via grep that NO WIP guard file (offline_*_alignment, human_demo_alignment, post_search_policy_retarget, deck_upgrade_target_guard, shop_action_guard, rest_site_smith_guard) references any schema constant — so guard edits never legitimately invalidate the replay schema.

---

### STEP 2 — RC-4: Stop hard guards from rewriting the policy training target

**Real code change, one call site. Flag-gated, default flips behavior to the correct off-policy mode.**

- **2a (CODE) `self_play.py` 950-961:** gate the `retarget_search_policy_after_hard_guard` call behind `getattr(self,"hard_guard_target_rewrite",False)`. When OFF (new default): keep the model's original soft `search_policy` as the policy/value target while still executing the guard's `action_idx` (env.step + stored `action=` already use it — verified at 1448/1480). Add `search_stats["post_search_hard_guard_override_applied"]=1.0` whenever the action changed; `post_search_hard_guard_policy_retargeted` → 0 in off mode. `semantic_policy` (1014, aggregated from the same `search_policy`) stays automatically consistent.
- **2b (CODE) `cli_args.py` near 65-82:** add `--hard-guard-target-rewrite {off,on}` default `off` (off = correct off-policy; on = legacy one-hot for A/B).
- **2c (CODE) `trainer.py` 283-417:** add `hard_guard_target_rewrite: str|bool = "off"` kwarg, normalize next to the `valid_guard_policies` block, store `self.hard_guard_target_rewrite` as bool.
- **2d (CODE) `cli_main.py` 479 + 855:** forward `hard_guard_target_rewrite=args.hard_guard_target_rewrite` at BOTH construction sites; add to the run-config log line ~555.

**Config flags:** `--hard-guard-target-rewrite off` (default). **Recommended training regimen: guards ON as a behavior wrapper (`--combat-hard-guard-policy full`) + target-rewrite OFF.**
**GATE:** unit test asserts stored `step['search_policy']` == the ORIGINAL soft distribution (not one-hot) while `step['action']` == guard final index when an override fired; over a longer run `post_search_hard_guard_override_applied_rate` should trend DOWN (policy learning the guarded behavior on-policy); Tool 0e (guards-OFF vs FULL) should stop reading "guards firing but not helping". **Effort: M. Depends on: Step 1** (so an old buffer with legacy one-hot targets can resume safely — mixed soft/one-hot are both valid distributions over the same `MAX_ACTIONS` axis; `search_policy` shape/dtype unchanged so no schema bump).
**WIP interaction:** `post_search_policy_retarget.py` (the module being gated) keeps its tests live in `on` mode — do NOT delete it. The offline/human alignment files are ORTHOGONAL and SYNERGISTIC: once the one-hot corruption stops, the audited CE channel becomes the deliberate way to inject expert priors. `deck_upgrade_target_guard`/`shop_action_guard`/`rest_site_smith_guard` dispatch through the SAME single retarget site → covered by the one `self_play.py` change. **Re-tune `monitor_fullrun_act1_gate.py` thresholds** on `post_search_hard_guard_policy_retargeted_rate` (now ~0) → point them at `override_applied_rate`.

---

### STEP 3 — RC-5: Make full-run reward teachable toward winning

**Two slices. Slice A is a quick config-grade constant edit; Slice B is real env code. Land behind/with RC-1 because it changes the value-target distribution.**

**Slice A — constants only (`reward_constants.py`), low-risk:**
- **3a:** fix the 3:1 HP asymmetry for normal/weak tiers WITHOUT touching the shared global scales: `PLAYER_HP_LOSS_TIER_SCALE['weak']`/`['normal']` 1.00 → 0.33 (0.03×0.33≈0.0099 ≈ the 0.01 damage scale → ~1:1). Leave elite 1.50 / boss 0.15.
- **3b:** `FLOOR_CLEAR_MIN_FLOOR` 11 → 3 (agent dies at median floor 11; reward its actual lifespan). **Must ship together with 3e (the cap)** or it re-introduces the "reach floor 14, die, still positive" attractor.
- **3c:** append constants: `FULL_RUN_COMBAT_WIN_BONUS_BASE=1.0`, `FULL_RUN_ELITE_WIN_BONUS=0.75`, `FULL_RUN_COMBAT_WIN_HP_PRESERVE_SCALE=0.5`, `FULL_RUN_ACT_CLEAR_BONUS=4.0`, `FULL_RUN_RUN_CLEAR_BONUS=10.0`, `FULL_RUN_DENSE_SHAPING_EPISODE_CAP=5.0`.

**Slice B — env code (`env_v2.py` 549-589 + new methods):**
- **3d:** add `_combat_victory_reward(before,after)` (had enemies → none AND player hp>0 AND NOT boss → base + elite extra + `sqrt(hp/max_hp)` preserve), wired into the step() sum after `_boss_damage_bonus_reward`. Gate against the existing enemies==[]/player-dead death-step path; add a one-shot-per-encounter latch to avoid double-pay on transient mid-combat snapshots.
- **3e:** add `_run_clear_terminal_bonus(after, terminated)` (sim `run.state_type=='victory'` → run clear; act-boss floor cleared while alive → act clear) and `_clamp_episode_dense_shaping` (per-episode accumulator capping cumulative POSITIVE hp-delta + floor ladder + boss-damage at `FULL_RUN_DENSE_SHAPING_EPISODE_CAP`; terminal win/clear/death bonuses NOT clamped).
- **3f:** add `combat_victory_events/_total`, `run_clear_events`, `act_clear_events` to `_blank_telemetry`.

**Config flags:** none (constants + env logic). **OPEN DECISION:** whether the shared `PLAYER_HP_LOSS_TIER_SCALE` normal/weak change is allowed to also soften the proven-learnable sandbox path, or must be scoped to a full-run-only tier dict (see Open Decisions).
**GATE:** scripted-trajectory unit test — a "win act, 50% HP, 8 rooms" return must exceed a "reach floor 14, die" return AND be >0 while the death return is <0; over a ~200-episode full-run smoke, `combat_victory_events>0`, `floor_clear_events` rising from floor 3, and mean return on floor-11-death episodes flips from positive to negative. Sandbox curated-deck eval win-rate must NOT regress (else scope the tier change). **Effort: M. Depends on: Step 1** (reward edits invalidate the value-target distribution; without RC-1 every resume re-drops the buffer — the exact failure this fix's gradient is trying to let compound). Independent of RC-3/RC-4 (env vs muzero/training).
**WIP interaction:** the uncommitted `env_v2.py` (+279 lines) is entirely action-frontier / singleton-end_turn recovery + campfire `max_hp==1` guard — it does NOT touch the reward-assembly block (549-589) or any `_*_reward`/`_*_penalty` method, so these edits apply cleanly on top. New `_episode_telemetry` keys are tolerated by the diagnostics WIP aggregators (they iterate/cast). **OPEN DECISION:** verify the *live bridge* run-clear signal (`run.is_victory_room` / game_over flag) before relying on `_run_clear_terminal_bonus` outside the headless sim — add a fallback signal.

---

### STEP 4 — RC-3: Restore exploration + a real value/policy improvement operator (AMPLIFIER)

**Real code change. All four sub-fixes default to no-op (entropy_coef=0, explore_eps=0, temp floor only matters once set). Land AFTER/WITH RC-4 for full benefit.**

- **4a (CODE) `cli_args.py` near 290-293:** add `--entropy-coef` (default 0.0), `--collect-temperature-min` (default 0.35), `--collect-temperature-final` (default None → = min), `--combat-direct-explore-eps` (default 0.0).
- **4b (CODE) `trainer.py` 257-377:** add the four kwargs, clamp + store as `self.entropy_coef`, `self.collect_temperature_min`, `self.collect_temperature_final`, `self.combat_direct_explore_eps`.
- **4c (CODE) `cli_main.py` 453 + 829:** forward all four at BOTH construction sites (main + async actor).
- **4d (CODE) `self_play.py` 484-487 (`compute_temperature`):** decay 1.0 → `collect_temperature_final`, then clamp UP to `collect_temperature_min` so the floor holds when `progress` saturates at 1.0 (resumed/over-budget). Keep signature unchanged.
- **4e (CODE) `self_play.py` 741-752:** before sampling, optionally mix `eps`-uniform mass over legal NON-end_turn actions into `direct_probs` (exclude `end_turn_idx` so the floor never encourages wasteful turn-ending); only when `safe_temperature>0.05`.
- **4f (CODE) `self_play.py` ~752 / 613 / 1453:** after `action_idx` is finalized, override `root_value = float(rollout_q[action_idx])` when finite AND legal (planner Q of the SELECTED action; `rollout_q` already computed at 668), falling back to `initial.value`. This turns the stored bootstrap (consumed by `muzero_buffer` n-step value target + priority) into a real 1-step improvement operator. **NOTE the RC-4 reconciliation:** prefer storing Q of the *actually-executed* (post-guard) action; with RC-4 landed and target-rewrite off, sequence so `root_value` reflects the executed action, not a pre-guard one.
- **4g (CODE) `train_step.py` ~905-908:** after the alignment-loss block, before `zero_grad`, if `entropy_coef>0` and not `spike_detected_in_step`, compute masked root-policy entropy and `total_loss = total_loss - entropy_coef*ent`; log `loss/policy_entropy`, `loss/entropy_coef` near 1002.

**Config flags (start at no-op, then ramp on curated sandbox):** `--entropy-coef 0.0→0.01`, `--collect-temperature-min 0.35`, `--combat-direct-explore-eps 0.0→0.05`.
**GATE:** Tool 0f planner margin should INCREASE once `root_value`=planner-Q lands; `schedule/temperature` stays ≥0.35 after resume; `metric/student_policy_entropy` and `root_visit_entropy` rise; `combat_quality_wasteful_end_turn_selected` does NOT rise; `planner_q_mae` stays bounded; A/B (entropy 0 + old schedule vs fixed) shows boss_win/act-clear improvement at equal wall-clock. **Effort: M. Depends on: Step 1 (buffer), Step 2 (RC-4 — otherwise guards one-hot-erase the explored/entropy-regularized targets and the planner-Q root_value can reflect a pre-guard action ≠ executed).**
**WIP interaction:** diffs target distinct regions from the dev's soft-HP/alignment WIP (compute_temperature, direct sampling/root_value, pre-backward entropy, cli block) → merge cleanly. The entropy term acts on the same root logits as the offline/human CE — compatible (a small entropy bonus mildly regularizes over-confident CE). Re-check line anchors against the actively-edited tree.

---

### STEP 5 — RC-2: Throughput, aux-loss rebalance, bounded+instrumented spike skip, stronger latent reg (AMPLIFIER)

**Real code change for grad-accum; the rest is config-grade tuning. Land LAST so each surviving gradient step counts against an honest, persistent, exploring signal.**

- **5a (CODE) `cli_args.py` after 33:** add `--grad-accum-steps` (int, default 1) and `--loss-spike-threshold` (float, default 30.0). No parser default changes (preserves checkpoint repro) — the profile is set on the command line.
- **5b (CODE) `trainer.py` __init__:** add `grad_accum_steps=1`, `loss_spike_threshold=30.0`; store `self.grad_accum_steps`, `self.loss_spike_threshold`, init `self._grad_accum_micro=0`.
- **5c (CODE) `train_step.py` 907-934:** accumulation-aware optimizer block — `zero_grad` only when `micro==0`; scale `total_loss/accum` on backward; only `clip + step + zero` every Nth micro-step; on a spike, drop the partial window (`self._grad_accum_micro=0`). Mirror for the non-AMP path. (LayerNorm/attention model → grad-accum ≈ true large batch.)
- **5d (CODE) `train_step.py` 771-780:** pass `threshold=float(getattr(self,"loss_spike_threshold",30.0))` to `_dump_loss_spike` (tunable instead of hard-coded 30.0). The `loss_spike/*` scalars from Step 0c are already emitting.
- **5e (CODE) `cli_main.py` MuZeroTrainer(...) sites 393/777:** thread `grad_accum_steps=args.grad_accum_steps`, `loss_spike_threshold=args.loss_spike_threshold`; optionally auto-raise to 60.0 when `batch_size<16` (tiny batch → noisier aux → fewer false skips).
- **5f (CONFIG, no code):** rebalance via existing CLI multipliers so policy+value+reward dominate — watch `loss_ratio/aux_total` (target <0.5) at cli_main.py 1079-1086. Re-baseline against the dev's running launch values (future-world-aux already lowered to ~0.25, state-consistency to ~1.0), do NOT double-cut. Keep `surprise-loss-weight`, `surface-mask-weight`, and `combat-hp-preservation-aux-weight` non-trivial (planner/HP objectives). Raise `--latent-gaussian-reg-weight` 0.005 → 0.1, ramping toward 0.25-0.5 while watching `metric/latent_reg_var_mean` toward O(1).

**Config flags / profile:** `--batch-size 16 --grad-accum-steps 4` (effective 64) `--updates-per-train 4` `--train-every 50` `--n-envs 1` (forced by single bridge session) `--latent-gaussian-reg-weight 0.1` `--loss-spike-threshold 60` (when batch<16). **HOLD `--lr` at 3e-4** until RC-1 is proven stable.
**GATE:** smoke at effective batch 64 stays <22GB on the RX 7900 XTX; equivalence check (batch32/accum1 vs batch8/accum4, fixed seed) tracks `loss/total`; `loss_ratio/aux_total` drops below ~0.5; `loss_spike/skip_count` falls after raising the threshold; `latent_reg_var_mean` rises toward O(1); dev's HP-aux + alignment shadow metrics still emit (not zeroed by the new accum/skip logic). **Effort: M. Depends on: Step 1 (buffer survival makes larger batches meaningful on resume); best after Step 4 so entropy is tuned against the rebalanced loss.**
**WIP interaction:** none of the WIP touches the backward/optimizer block, spike skip, grad-accum, world-model multipliers, or `_latent_regularization_loss` → fully additive. Two benign interactions: raising the spike threshold makes the WIP alignment losses (gated on `not spike_detected_in_step`) run MORE often (intended); treat the WIP `combat_hp_preservation_aux_loss` term as a policy-adjacent objective to KEEP, not a world-model term to suppress, during the 5f rebalance.

---

## Sequencing Rationale

1. **Verify-first (Step 0) before any RC** — the five tools are the pass/fail gates for the deeper fixes. You cannot claim RC-1 worked without the resume dashboard, RC-4 without the guards A/B, or RC-3 without the planner margin. Two of them require ~5-line emission edits so the metrics survive the *async* (primary) path; do those first and in one change.
2. **RC-1 second (the enabler)** — the kill-every-1-3h loop means *nothing compounds* while every resume drops the 2.3GB buffer + 395MB optimizer. Every later fix changes reward/exploration/guard behavior whose benefit only shows over many hours; if learning resets every restart, you can never measure them. RC-1 is also what *lets* RC-4 guard edits and RC-5 reward edits land without forcing a buffer drop.
3. **Correctness root causes (RC-4, RC-5) before amplifiers** — RC-4 stops the trained policy from being a one-hot lie about what the model wanted; RC-5 gives the value head a dominant terminal win/act-clear target instead of a penalty-only landscape where "survive then die" out-scores "fight to win". Amplifying a corrupt/penalty-only signal (more entropy, more gradient throughput) makes it *worse*. Fix the signal first.
4. **Amplifiers last (RC-3, RC-4-dependent; then RC-2)** — RC-3 exploration + planner-Q bootstrap only pays off once the target is honest (RC-4) and the reward rewards winning (RC-5). RC-2 throughput/rebalance makes each surviving gradient step count — most valuable once it is updating toward the right thing and the buffer it samples from persists.
5. **Curriculum-staged throughout** — validate each step on the proven-learnable curated-deck combat sandbox (PPO control hit 0.68 win there) before letting it touch full-run, because the defect is shared env/reward/regimen/guards, not the algorithm.

---

## Conflicts With WIP (consolidated)

- **Same-file rebase required (cite live anchors, re-confirm before editing):** the dev has uncommitted edits in `checkpointing.py`, `cli_args.py`, `cli_main.py`, `self_play.py`, `train_step.py`, `trainer.py`, `losses.py`, `env_v2.py`. RC-1 (Step 1) and Step 0 touch the most-edited files — **rebase onto the working tree, do not apply against HEAD.**
- **`post_search_policy_retarget.py` (dev RC-4 mitigation):** Step 2 *gates* its single caller (self_play.py:951) behind the new flag; it does NOT delete the module (tests + `on` mode keep it live). Confirmed it touches NO schema constant / network state_dict → orthogonal to RC-1, and it is exactly the edit RC-1's buffer-decoupling protects.
- **`offline_*_alignment.py` / `human_demo_alignment.py` (dev):** pure additive masked-CE on root `policy_logits`, default shadow-only. ORTHOGONAL and SYNERGISTIC with RC-4 (audited expert-prior channel replacing the unaudited guard one-hots) and RC-3 (entropy mildly regularizes their CE). They do NOT touch env reward, the online `search_policy` target, or the optimizer block — no conflict with RC-5/RC-2.
- **`deck_upgrade_target_guard.py` / `shop_action_guard.py` / `rest_site_smith_guard.py` (dev build guards):** dispatch through the SAME single retarget site → covered by the one RC-4 `self_play.py` change with zero per-guard edits; their `*_SEARCH_SUFFIXES` are already merged in `async_telemetry.py`.
- **`env_v2.py` WIP (+279 lines):** action-frontier / singleton-end_turn recovery + campfire guard only; does NOT touch the reward block (549-589) or any `_*_reward` method → RC-5 applies cleanly. New telemetry keys tolerated by the diagnostics WIP (episode_metrics.py +79, deck_build_metrics.py +190, trainer_dumps.py rewrite).
- **`train_step.py` WIP (+115 soft-HP / alignment):** distinct regions from RC-2/RC-3 (backward block, spike skip, entropy, accum); the WIP alignment + HP-aux losses are gated on `not spike_detected_in_step`, so RC-2's threshold raise makes them run more (intended). Treat HP-aux as policy-adjacent (keep) in the RC-2 rebalance.
- **Monitor re-tune:** RC-4 drives `post_search_hard_guard_policy_retargeted_rate` → ~0; update `monitor_fullrun_act1_gate.py` pass/fail thresholds (lines ~450/645/652/664) to read `post_search_hard_guard_override_applied_rate` instead, or a 0 rate will be misread as failure.

---

## What NOT To Do

- **Do NOT add new hard guards.** The whole thrust (RC-4) is that guards corrupt credit assignment; the audited offline/human-demo CE channel is the sanctioned way to inject priors. Keep existing guards ON only as a *behavior/safety wrapper*, not as a training target.
- **Do NOT keep the kill-every-1-3h restart loop as the operating mode.** It is the proximate cause of zero compounding. Land RC-1 (buffer/optimizer survival + interval/shutdown checkpoint) and run longer uninterrupted segments.
- **Do NOT keep passing `--resume-without-buffer --resume-without-optimizer`.** They become deprecated force-skip aliases; remove them from launch scripts so the default (load + migrate) takes effect.
- **Do NOT bloat aux-loss weights.** ~15 world-model/aux terms already dilute policy/value/reward ~5×. The RC-2 move is to *lower* world-model multipliers until `loss_ratio/aux_total < 0.5`, not add more terms. Keep new objectives (HP-aux, entropy) small and policy-adjacent.
- **Do NOT ship `FLOOR_CLEAR_MIN_FLOOR=3` without the dense-shaping cap** (`FULL_RUN_DENSE_SHAPING_EPISODE_CAP`) in the same change — it re-creates the "go deep, die, still positive" attractor.
- **Do NOT raise `--lr` or `--n-envs`** while RC-1 is unproven (single bridge session forces n-envs=1; LR bump risks instability before buffer persistence is confirmed).
- **Do NOT hand-build observation/latent states for the planner probe** — replay real frames from `intent_combat_quality.jsonl` to avoid OOD false-fails.
- **Do NOT raise `--latent-gaussian-reg-weight` straight to 0.5** — ramp 0.005→0.1→0.25 watching `metric/latent_reg_var_mean`; overshoot flattens the latent.
- **Do NOT bump `MUZERO_OBS_SCHEMA_VERSION` / `observation_shape_caps` casually** — RC-1's buffer-decoupling relies on the discipline that these only change on *real* obs-encoding changes.

---

## Open Decisions (must resolve before/within implementation)

See the structured openDecisions list.
