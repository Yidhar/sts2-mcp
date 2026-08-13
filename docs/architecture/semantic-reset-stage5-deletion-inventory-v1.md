# Stage-5 deletion inventory — §9 retirement list code sites

## EXECUTED (2026-08-13)

- `8463aa2` — execution step 1 (items 1+2): entry/selection exploration
  floors and completion guidance deleted; config v20 strip-migration.
- `43894bc` — execution steps 2+3 (items 10, 3+4): entropy collapse breaker
  (learner dynamics v3), two-sided entry support corridor, and transaction
  completion/selection-group CE deleted; `support_eligible` + SMDP-Q kept.
- `8fd5629` — execution steps 4+5+6 (items 7+8, 6, 9): success/act-segment/
  revival imitation with exemptions/ratchets, liveness policy actors
  (critics kept), and sidecar policy layers deleted; episodic replay v5,
  failure-credit replay v6, transaction replay v8.
- final commit (this change) — execution step 7 (item 5): macro residual
  AWR + lifecycle actor bridge config/objectives/metrics and the
  macro_policy/macro_option_value model heads deleted (state-dict key set
  changed; retired family drops via tolerated loader groups), dashboard
  retired-series cleanup, CLAUDE.md semantic-executor rewrite +
  anti-accretion rule.

Status: read-only inventory (2026-08-11) prepared BEFORE the stage-3 handoff
decision; deletion itself executes only after the corresponding controller is
replaced (reset doc §10 stage 5, domain-scoped). Line numbers reference the
tree at commit b2fc512.

## Global deletion hazards

1. **Strict config parser.** `sts2_rl/training/config.py:1614-1622`
   (`_construct`) rejects unknown keys. Removing any dataclass field breaks
   parsing of every stored checkpoint's `training_config` payload on both
   exact resume and model-init. Deletion therefore requires a config version
   bump (v19 → v20) whose `model_initialization_config_from_mapping`
   migration explicitly strips the retired tables/fields from older payloads.
2. **Model head ABI.** `macro_policy_head`/`macro_option_value_head`
   (grounded_candidate.py:1371-1402) are in every existing checkpoint's
   state_dict; `macro_evaluation.py:1136` loads `strict=True`. Head deletion
   is LAST and requires a lineage bump or dormant retention.
3. **Checkpointed sidecar payloads.** Transaction replay, failure-credit
   replay, and episodic replay `state_dict` shapes (including protection
   counters and `act_segment_health`) are hashed into atomic checkpoints;
   `failure_credit/replay.py:681-765` validates an explicit counter key list
   on load.
4. **Learner dynamics payload.** Entropy-breaker counters live in
   `_LEARNER_DYNAMICS_STATE_VERSION` (learner.py:66, 1605-1654) — deletion
   bumps that version.
5. **Launch contract.** `risk_actor_start_update` is a readiness-manifest
   field (launch_contract.py:96, 437-439) — needs its own migration.

## Must NOT be deleted (shared or explicitly retained)

- `TransactionLifecycleEvidence` + `support_eligible` (transaction.py:88-246)
  — also gates SMDP entry-Q value labels and replay protection.
- `transaction_operations.py` registry — shared with lifecycle evidence and
  learner telemetry.
- The generic forced-singleton rule (no policy target for singletons) —
  guards V-trace telemetry (learner.py:2540-2545, sts2_baseline/rollout.py).
- Progress/stall/cycle trackers (collector.py:2027-3060) — remain as episode
  termination + diagnostics per §9.
- Long-horizon task/revival VALUE heads and liveness VALUE critics — §9
  retires only the policy-gradient channels.
- V-trace core (VTraceLearner, target recursion, clip config) — stays until
  the combat challenger wins (stage 4).
- Collector legality/dispatch machinery, semantic grouping helpers,
  `_transaction_entry_operation` (shared with journals/telemetry).
- Reward contract, FIFO/sequence-unroll contracts (sts2_baseline) — clean of
  §9 items.

## Item → primary sites (details in agent inventory, commit-pinned)

1. **Entry/selection exploration floors** — config.py:1000,1060-1140,
   1475-1478,1522-1525,1659-1662,1743-1758; collector.py:3313-3316,
   3389-3442,1463-1611,3900-3925 + counters; factory.py:259-271;
   runtime.py:786-794; profile/experiment TOMLs.
2. **Completion guidance** — collector.py:1651-1736,3926-3943 + counters;
   config.py:1071. Eval path never reaches it (greedy returns first).
3. **Entry support corridor** — learner.py:1242-1306,2581-2612,4234-4307,
   4499-4503; config.py:380-381,489-493,519-529. Keep `support_eligible`.
4. **Completion CE + selection-group CE** — transaction.py:72-77,249-262,
   555-717; learner.py:4119-4231,4494-4498,4560-4564; config.py:374,423.
   `_has_factual_avoid_target` (924-935) drives replay eviction protection —
   retire together and accept replay-payload version bump.
5. **Macro AWR + lifecycle actor bridge** — config.py:404-423 + v19
   migration 1906-1930; learner.py:3907-4105,4371-4451,4517-4562,2585-2629;
   model heads (ABI hazard above); monitoring.py:343-385 + dashboard.
6. **Liveness policy actors** — learner.py:853-1231 (risk 1066-1105, AVOID
   1107-1121, completion 1122-1124, cycle 1126-1165, contrast 1167-1189),
   580-835,1997-2201,2709-2772; failure_credit/actor_eligibility.py (whole);
   contracts.py policy-credit classes; corpus.py actor quotas;
   replay.py actor-fresh counters; config.py:591-627,673-730,1819-1837.
   Keep value critics + detector strata.
7. **Success/act-segment/revival imitation** — config.py:782-819 + v18
   migration 1864-1891; episode_replay.py:470-516,1162-1690 (act-segment
   plumbing is replay ABI); learner.py:3503-3760; collector.py:4256,4898,
   5757-5758. Keep long-horizon value heads computed in the same loop.
8. **Exemptions/ratchets** — success_imitation_exempt_surfaces
   (config.py:798, learner.py:3556-3566, episode_replay.py multiple);
   trust-region ratchet learner.py:3598-3610; policy-lag ratchets
   learner.py:3567-3576,3950-3957,4079-4086; CE exemptions
   transaction.py:624-671; per-objective singleton counters/gates.
9. **Sidecar policy layers** — transaction.py policy targets + AVOID/stratum
   protection in put/sample (1000-1210); failure_credit policy layers
   (actor_eligibility, contracts 424-438/651-725/794-825, corpus quotas,
   compiler/pipeline policy emission). Value/diagnostic layers stay.
   `pairwise_ranking_weight`/`observed_outcome_pairs`: policy objective but
   not named in §9 — separate decision (currently weight 0).
10. **Entropy collapse breaker** — config.py:215-218,275-282;
    learner.py breaker constants, 1598-1654, 1656-1749 (V2 branch
    1700-1732), 2546-2560. Annealing (`_annealed_entropy_weight`) stays.

## Execution order (lowest coupling first)

1. Items 1+2 (floors, guidance) — training-only; config v20 bump.
2. Item 10 (breaker) — learner dynamics version bump.
3. Items 3+4 (corridor, CE) — keep support_eligible + SMDP-Q.
4. Items 7+8 (imitation + exemptions) — keep value heads + generic
   singleton rule.
5. Item 6 (liveness actors) — keep critics; launch-contract migration.
6. Item 9 (sidecar policy layers) — replay payload version bump.
7. Item 5 (macro AWR + model heads) — last; lineage bump or dormant heads.

Each step: focused tests updated in the same commit, full suite green
(31-known Windows baseline), no compatibility facade left behind.
