# Cross-domain bootstrap bridge — design (stage-4 unblock)

Status: design proposal answering the production guard "combat training
requires an explicit cross-domain bridge" (launcher refusal) and the
stage-4 record's condition ("explicit encounter-terminal or cross-domain
bootstrap boundary; realized-return folding is valid only while the combat
champion is frozen"). No code in this commit.

## 1. The problem it closes

The macro and combat candidate-Q controllers are separate models with
separate replay and recurrent state. Each domain's transitions therefore
end at a boundary the other domain owns:

- a combat encounter's LAST decision leads into macro surfaces;
- a macro decision that enters combat realizes its outcome through
  hundreds of combat decisions.

Today the macro domain handles its side by REALIZED-RETURN FOLDING: all
rewards accrued while the (frozen) champion plays combat fold into the
open macro transition. That is factual and stationary ONLY because the
combat controller is frozen. The combat domain has no sanctioned
equivalent — folding inter-encounter macro rewards into the last combat
transition would bake the CURRENT macro behavior into combat values, and
bootstrapping across the boundary with the combat model itself would
evaluate macro states a combat-domain function never trains on. Hence the
guard.

## 2. Contract

One Bellman graph, two function approximators, explicit bridge terms:

1. **Encounter scope.** A combat-domain episode is one encounter. The
   authority already resets tactical recurrence at encounter entry;
   the encounter's last transition is DOMAIN-TERMINAL: its own-model
   bootstrap is cut (discount reaches it only through the bridge term).
2. **Bridge state capture.** At encounter end, the collector records the
   next MACRO decision snapshot (the post-combat surface) as
   `bridge_snapshot` on the closing combat transition, together with the
   floor delta and any rewards accrued between the last combat decision
   and that macro decision.
3. **Bridge value.** The combat learner's target for the closing
   transition is `r_boundary + Gamma(dfloor) * max_masked Q_macro(bridge_snapshot)`
   where Q_macro is the PUBLISHED macro model at a pinned
   `policy_version`, evaluated no-grad. The bridge value is a scalar
   computed at training time (not collection time), so it tracks macro
   learning without retraining combat replay.
4. **Version discipline.** Bridge evaluation pins the macro publication
   loaded at trainer segment start; the trainer records
   `bridge_partner_version` in metrics. A segment never mixes partner
   versions inside one target computation.
5. **Symmetry on handoff.** When combat ownership ever passes to the
   challenger (champion retired), macro realized-return folding loses its
   stationarity license; the macro learner then switches to the mirrored
   bridge: macro transitions entering combat bootstrap from
   `max_masked Q_combat(first_combat_snapshot)` under the same pinning
   rules. Until then, macro folding stays as-is (explicitly licensed by
   the frozen champion).
6. **Fail-closed.** Run-terminal boundaries keep bootstrap 0 via the
   clock. A missing/invalid bridge snapshot invalidates that episode for
   combat replay (same policy as the authority's replay-invalid path);
   it is never silently approximated.

## 3. Implementation sketch (follow-up commits)

- transitions: optional `bridge_snapshot` + boundary reward/floor fields
  on the closing combat step; contract version bump.
- authority (combat domain): capture the bridge snapshot at the first
  owned-or-declined macro surface after an encounter, then close the
  encounter's open transition.
- learner: bridge-aware bootstrap for domain-terminal steps — injected
  `bridge_value(snapshot) -> float` callable (the trainer wires it to the
  pinned macro publication, no-grad, masked max).
- trainer/launcher: `--bridge-partner <macro publication>` required for
  `--control-domain combat`; guard lifts only when provided.
- tests: encounter-end target equals boundary reward + discounted partner
  max; version pinning; fail-closed on missing bridge snapshot.

## 4. What this deliberately does not do

No shared parameters, no gradient through the partner, no partner replay
access, no per-state controller mixing at collection time — the bridge is
a scalar target term, nothing else.
