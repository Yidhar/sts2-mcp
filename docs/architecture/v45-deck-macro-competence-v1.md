# v45 deck & macro competence package (design v1)

Status: PROPOSED. Target lineage: v45 (from a v44 healthy anchor, budget stays
40). Config identity bump (config-v19). No reward change; reward stays
sts2-run-survival-efficiency-v7.

## 1. Measured problems this package answers

1. **Shared-scorer target leakage (2026-08-10 audit, v44 gates 50k/75k/100k):**
   the upgrade surface and the removal surface collapsed onto the SAME target
   heuristic — DEFEND_IRONCLAD is 58% of forge targets (weak play: +2 block is
   the lowest-value upgrade) and 63% of removal targets (correct play). HAVOC
   is #2 on both lists. One rule, learned where credit was clear (removal),
   generalized to a surface where it is wrong (upgrade), because no
   operation-differentiated target credit exists to oppose it.
2. **Card-reward skip and relic purchase remain zero-support branches:**
   265/265 rewards taken at v44 gates; relic purchases 0 across every lineage
   ever measured. Deck bloats to 26+ cards, diluting upgrade density.
3. **Rest/forge macro pendulum (user report + v42-v44 measurement):** whole
   gates of all-rest and all-forge, forging at low HP, corr(HP, P(non-rest))
   pinned ≈ 0 across three revival price points. Established mechanism: the
   per-decision economic differential (~0.015) is unlearnable through γ=1
   full-run returns (σ ≈ 0.5); after the imitation surface exemption, NO
   channel anchors this decision, so it random-walks under entropy pressure
   and argmax amplifies the drift into mode flips.
4. **Incumbent-target lock-in:** completed-lifecycle CE prefers "completed
   unique factual steps" including the card-target select step
   (transaction.py:486-573, PREFER path). With cancels now at 0, this channel
   re-imprints whatever target the policy currently picks — it is a plausible
   maintainer of the Defend attractor in (1).

## 2. Design items

### D1. Reward-skip support arc (proven playbook, third application)

Extend the reviewed transaction-entry classifier
(`_transaction_entry_operation`, collector.py:1090-1130) with operation
`reward_skip`: the `skip_card_reward` action at a card-reward decision
surface. Lifecycle: entry == exit == the skip decision; commitment proof =
authoritative reward-screen resolution with deck signature UNCHANGED (the
mirror of the upgrade/removal deck-signature proof, collector.py:1396-1426).
Then the standard arc:

1. Behavior-side bootstrap: `transaction_exploration.operations` gains
   `reward_skip` with the existing entry-epsilon machinery (a small floor,
   0.15, NOT 0.5 — the branch needs samples, not domination), withdrawn next
   lineage exactly as the forge scaffold was.
2. Verified skip lifecycles feed the existing two-sided corridor (no new
   math; `two_sided_policy_support_loss` unchanged) so P(skip) can never
   re-absorb at 0 after the scaffold leaves.
3. Contextual preference (skip vs which card) stays with factual return/Q
   learning, per the corridor invariant.

### D2. Relic-purchase support arc (same pattern)

Operation `relic_purchase`: shop_purchase whose item.category == "relic".
Commitment proof = relic inventory count increase in the authoritative
post-state. Entry-epsilon bootstrap (0.15) + corridor, withdrawn next
lineage. Gold-competition preference (relic vs card vs removal) stays with
return/Q learning.

### D3. Target-level credit unlock (answers problem 1 and 4)

a. **Narrow the completed-path CE off card-target steps.** In
   `factual_transaction_policy_targets` (transaction.py:486-573), PREFER
   labels currently cover all completed learn steps except the lifecycle
   entry (transaction.py:555) and forced singletons (557). Additionally
   exclude steps whose selected action is a card-target choice inside a
   verified upgrade/removal lifecycle (the `select_card` step). Forward
   steps (confirm, proceed) keep their PREFER labels — completion competence
   is retained; the CE simply stops re-imprinting WHICH card. Rationale
   mirrors the existing entry exemption comment (transaction.py:552-554):
   target competence has a distinct credit contract; a completion CE must not
   turn the incumbent target into a permanent imitation target.
b. **Turn on per-step transaction Q at selection steps with the SMDP option
   return.** The candidate-conditioned Q infrastructure exists and is
   config-disabled (`transaction_q_weight = 0.0` in v42-v44 configs; default
   0.25 at config.py:364). Re-enable at 0.05, with the target-step Q label
   using the SAME factual option return/discount already accumulated for the
   lifecycle (`next_rest_or_act` horizon) instead of whole-episode return.
   Q(select Defend | deck, act, HP) vs Q(select Havoc | ...) then differ on a
   10-20 floor factual horizon — the first operation-specific,
   target-specific signal in the system. Policy is never rewritten; the
   preference reaches the policy through the shared candidate features and
   ordinary advantage learning, per the corridor invariant's division of
   labor.

### D4. Rest-arm SMDP option accounting (answers problem 3, value side)

Extend lifecycle option accounting to the rest action (idx 0) at rest-site
surfaces: operation `rest`, entry == exit == the decision, option return =
factual rewards through the next rest-site arrival / Act boundary / terminal
(identical horizon rule `next_rest_or_act`, transaction.py:648-663). No
corridor, no scaffold — rest has abundant support; this is value/telemetry
machinery only:

- Q(rest | state) and Q(forge-entry | state) become directly comparable on
  the same factual horizon, giving the trunk a short-horizon HP-consequence
  signal that full-run V-trace cannot deliver (the measured σ≈0.5 wall).
- **Free ground-truth telemetry:** per-update means of rest vs forge option
  returns bucketed by HP band answer, from live training data, whether
  rest-at-low-HP actually pays — the experiment that decides whether HP
  conditioning is a real skill or a phantom. Emit as
  `transaction_rest_option_return_by_hp` / `..._forge_...` gauges.

### D5. Re-admit rest_site into success imitation (answers problem 3, policy side — FLAGGED)

`success_imitation_exempt_surfaces` currently contains rest_site + shop. The
exemption was anti-ratchet armor from the saturation era (v33/v36 lock-ins).
Today the corridor floor (5% both sides) provides the anti-absorption
guarantee structurally, and the exemption now blocks the ONLY channel that
could anchor the pendulum: run victories and healthy act segments carry
factual examples of WHEN resting/forging won. Change: drop `rest_site` from
the exemption (keep `shop` for one more lineage); act-segment and
run-success imitation then teach the macro choice from winning trajectories,
with these guards unchanged: corridor floors, trust region, lag gate,
policy-collapse-v2 breaker.

Hazard: mode re-lock via imitation of a lucky mode. Mitigations: weight
unchanged (no new amplification), corridor floor caps saturation, and the
acceptance line C tripwire (§4) stops the lineage if the pendulum hardens
into a pole with declining wins. This item ships behind its own config flag
so it can be reverted independently.

## 3. Explicitly out of scope

- No reward change, no budget change (stays 40), no model tensor change.
- No card-name rules anywhere: every mechanism above is operation-generic;
  card identity enters only through the existing factual encodings.
- No planner/action rewrite: Q heads train representation and telemetry;
  action selection remains the masked policy alone.

## 4. Acceptance (gates at 25k/50k/100k, paired seeds)

1. **Target quality (D3):** basic Strike/Defend share of forge targets falls
   58% → below 25%, while removal keeps basics share ≥ 50% (the two surfaces
   must SEPARATE — same-card overlap of top-3 targets is the regression
   tripwire).
2. **Skip/relic ignition (D1/D2):** skip rate > 0 sustained with deck size
   p50 down from 26 toward ≤ 22; relic purchases ≥ 1 per 4 episodes by 50k.
3. **B-line continuation:** acts-2/3 revival increment p50 ≤ 15 (v44 hit 19).
4. **Ground truth read-out (D4):** the by-HP-band option-return gauges give a
   signed answer on rest-at-low-HP by 25k; if the differential is ≈ 0, D5's
   expectations are re-scoped (pendulum is then honest indifference and only
   deck power matters).
5. **Pendulum (D5):** gate-to-gate rest% amplitude narrower than v44's
   98/67/7/3/95/34 AND corr(HP,P) > +0.1 at two consecutive gates if and
   only if D4 shows a real positive differential.
6. Eval wins ≥ v44 (2-5/16) at every gate; any collapse signature (floors
   p50 < 25, entropy < 0.1, cost-actor runaway) stops the lineage.

## 5. Test plan

- Classifier: `reward_skip` / `relic_purchase` / `rest` operation recognition
  truth tables (generic-protocol only, no card/relic names).
- Lifecycle: skip commitment proof (deck unchanged), relic proof (count
  increase), rest option span (next_rest_or_act), forced-singleton and
  censored-exit paths reuse the existing suppression counters.
- D3a: completed lifecycle yields PREFER on confirm steps, none on the
  card-target step; removal and upgrade cases both asserted.
- D3b: selection-step Q labels use option return, not episode return;
  disabled-config path byte-identical to v44 behavior.
- D5: flag off ⇒ exemption list behavior identical to v44.
- Full suite + ruff + strict mypy + dry-run per change discipline.

## 6. Sequencing

D3a (surgical, stops active damage) → D1 + D2 (proven pattern, mostly
classifier + config) → D4 (accounting + telemetry) → D3b (Q re-enable) →
D5 (flagged, last, after D4's ground-truth read). One lineage carries all
of D1-D4; D5 may ship dark and be enabled at the 25k gate decision point.
