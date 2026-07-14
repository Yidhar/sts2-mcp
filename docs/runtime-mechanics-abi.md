# Runtime mechanics ABI

`grounded-runtime-mechanics-encoding-v6` is the minimum observation contract
for the restarted training line. Formal training is refused unless the pinned
simulator passes both a native catalog audit and a real event-to-combat runtime
audit.

This contract transports game-owned facts. It is not a handwritten card,
relic, potion, event, or boss strategy table.

## Covered domains

### Powers, buffs, debuffs and status

Every visible `PowerModel` instance retains:

- exact power ID, amount, native `Type` and `StackType`;
- visibility, instancing, negative-stack and scaling flags;
- owner/applier/target identity where the game exposes it;
- every native `DynamicVar`, including its type, family and values.

The catalog uses native `PowerModel.Type`. Class-name rules such as
`is_debuff_hint` are prohibited by the preflight audit.

### Relics

Relic instances retain rarity, status, used-up state, stack count, counter,
merchant and lifecycle flags, and all native dynamic variables. Catalog fields
that require a mutable owner are nullable in the static catalog and exact in a
real player instance.

### Potions

Potion instances retain usage, target type, combat-generation and custom
usability facts, together with all native dynamic variables and slot/resource
state. The policy sees the current legal potion candidates separately; it does
not infer usability from a potion-name list.

### Card enchantments and afflictions

Cards retain the game's singular `CardModel.Enchantment` and
`CardModel.Affliction` instances, projected as arrays for batching. Each
modifier keeps its exact ID, modifier type, amount, native flags and dynamic
variables. Card lifecycle facts include replay counts, retained/in-combat state,
removability, transformability, clone/duplicate state and pile membership.

### Events

The world observation retains event ID, layout, page/description key,
deterministic/shared/finished state, encounter identity, native dynamic
variables and the current option set. Options retain stable index, text key,
description and locked/chosen/proceed state. Event options remain legal
candidates; no localized-text parser invents their effects.

### Enemies and bosses

The catalog records every encounter, its room type and exact monster IDs. Boss
coverage is therefore derived from native `room_type == boss`, not a boss-name
list. Runtime enemies retain:

- combat/model identity, HP, block and hit/alive state;
- primary/secondary/stunned/pet/infinite-HP flags;
- visible current move and intent;
- move history and phase-transition flags exposed by the current move state;
- every visible power and its dynamic variables.

`follow_up_state_id`, future random draws, hidden encounter branches and other
information unavailable to a normal player are deliberately excluded. Boss
phase changes are learned from observable move/power/state transitions rather
than leaked future states.

## Current pinned evidence

On 2026-07-14, the locked simulator catalog gate reported:

| Domain | Count |
| --- | ---: |
| Powers | 270 |
| Relics | 294 |
| Potions | 64 |
| Enchantments | 24 |
| Afflictions | 11 |
| Events | 57 |
| Monsters | 101 |
| Encounters | 80 |
| Boss encounters | 12 |
| Boss monsters referenced | 15 |

These are evidence for the pinned source revision, not permanent constants.
The audit uses conservative lower bounds so a small balance update does not
hard-code today's exact counts, while missing domains and malformed records
still fail closed.

## Mandatory gates

After rebuilding the pinned simulator, run:

```powershell
python packages/rl-agent/scripts/audit_card_fact_coverage.py `
  --sim-exe <PATH_TO_HEADLESS_SIM_EXE>
python packages/rl-agent/scripts/audit_runtime_mechanics_coverage.py `
  --sim-exe <PATH_TO_HEADLESS_SIM_EXE>
```

The second command enumerates all mechanics catalogs, starts a deterministic
full run, audits the initial native event and advances to a real combat to
audit player, card, relic, potion, power, enemy, intent and move-history fields.

`python -m sts2_rl.train` runs the runtime-mechanics audit automatically after
binary/source identity verification and before creating training resources.
A successful report is written below
`<ARTIFACT_ROOT>/logs/runtime-mechanics-preflight/`. Any missing collection,
legacy heuristic field, hidden follow-up state, malformed dynamic variable or
runtime field mismatch blocks training.

## Checkpoint boundary

The feature ABI minimum is now 224 and the encoding fingerprint/version are
new. Exact resume and model initialization compare the complete encoding
contract before loading tensors. Checkpoints from the previous card/selection
ABI therefore fail closed; they must not be shape-padded or silently migrated.
