# Runtime card-facts ABI

`grounded-runtime-mechanics-encoding-v6` is the maintained representation for the
restarted RL line. It replaces the incomplete structural projection that kept
card ID, type, cost, target and upgrade state but discarded most card-model
facts.

## Source of truth

The ABI is populated from the game's `CardModel`; it is not a curated card
database and does not parse localized descriptions. The live bridge and pinned
HeadlessSim expose the same categories of facts:

- stable card ID, card type, target type, rarity and current cost;
- energy-X/star-X state and current star cost;
- runtime keywords and game-owned tags;
- game-owned hover-tip identities for powers, keywords, generated cards and
  static mechanics that are not represented by a numeric variable;
- `GainsBlock`, end-of-turn-in-hand, on-draw and temporary-exhaust facts;
- every `DynamicVar`, including its runtime class, semantic family, optional
  `PowerVar<T>` type, value properties and numeric values.

The encoder retains card ID for residual identity, then emits keywords, tags,
traits and dynamic variables as separate typed entity tokens. Dynamic-variable
`base_value`, `enchanted_value`, `current_value`, `int_value` and
`was_just_upgraded` occupy distinct versioned feature slots. They do not share
the old hashed value slots.

## Dynamic-variable families

Families are derived from runtime classes, not from card IDs:

| Family | Runtime examples | Meaning exposed to the model |
| --- | --- | --- |
| `damage` | `DamageVar`, `CalculatedDamageVar`, `OstyDamageVar` | damage magnitude and value properties |
| `block` | `BlockVar`, `CalculatedBlockVar` | block magnitude and value properties |
| `cards` | `CardsVar` | card-count magnitude, normally draw/count behavior |
| `energy` | `EnergyVar` | energy magnitude; lifecycle tokens determine when it fires where available |
| `power` | `PowerVar<T>` | exact power type such as `PoisonPower` plus amount |
| `repeat` | `RepeatVar` | repetition count |
| `hp_loss`, `heal`, `max_hp` | corresponding runtime classes | HP resource magnitude |
| `gold`, `stars`, `forge`, `summon` | corresponding runtime classes | exact resource/count magnitude |
| `value` | generic/custom `DynamicVar` | exact runtime name and value without inventing a stronger interpretation |

Applied player/enemy powers remain separate world-state tokens containing exact
power ID and amount. A card that applies poison therefore exposes
`PoisonPower` on the card before play, and the resulting `PoisonPower` amount
on its target after play.

## Representative semantics

The pinned native catalog gate verifies, among other cards:

- Strike: `Attack` + `DamageVar(6)` + native `Strike` tag;
- Defend: `Skill` + `BlockVar(5)` + `GainsBlock` + native `Defend` tag;
- Bash: damage plus `PowerVar<VulnerablePower>`;
- Adrenaline: `Skill` + `CardsVar(2)` + `EnergyVar(1)` + `Exhaust`;
- Burn: `Status` + `Unplayable` + end-of-turn-in-hand + damage;
- Void: `Status` + `Unplayable` + `Ethereal` + on-draw + energy magnitude;
- Shame: `Curse` + `Unplayable` + end-of-turn-in-hand + factual `Frail` value;
- Deadly Poison: `Skill` + exact `PoisonPower` and amount;
- Burning Pact: `Skill` + exact `CardsVar` draw magnitude.

Run the read-only coverage gate after rebuilding the pinned simulator:

```powershell
cd packages/rl-agent
python scripts/audit_card_fact_coverage.py --sim-exe <PATH_TO_HEADLESS_SIM_EXE>
```

For the currently pinned source, the gate enumerates 577 cards, 774 runtime
dynamic variables and 410 cards with game-owned hover-tip identities. It also
runs every catalog card through the production 24-token candidate-local ABI;
the largest current card uses 14 tokens. Finally, it starts a deterministic
full run and verifies that Strike, Defend and Bash retain the same facts in a
real combat hand after native DTO and Python transport. These counts are
evidence for that pin, not permanent game constants.

## Deliberate boundary: effect values are not a complete program graph

This ABI does **not** claim that every card has been converted to a complete
declarative program. The current pin still has 19 cards for which the native
catalog exposes only identity, type, cost, rarity and target; their custom
program remains residual card-ID/transition semantics. Some other mechanics are
control flow rather than a numeric `DynamicVar`. For example, Burning Pact exposes its draw count, but its
"select one hand card, exhaust it, then continue" behavior exposes the game's
own Exhaust hover-tip identity and the draw count, but the ordering and selected
card are represented by the real subsequent selection state and the
select/deselect/confirm action protocol. Conditional branches and unusual
command sequences can likewise retain card-ID residual semantics until a
game-owned command trace or mechanic graph is available.

Selection state is part of the factual ABI, not an action-label shortcut. A
card keeps its physical pile (`Hand`, `Discard`, `Exhaust`, or `Deck`) while a
separate membership field records whether it is currently selectable or
selected. Multi-select observations retain the identity of every checkbox,
the selected set, min/max/count, prompt ID, source/destination zones, and
whether the game requires explicit confirmation. Legal candidates distinguish
select, deselect, confirm, and cancel-prompt mutations. Fixed-count prompts
that the game auto-completes therefore expose no invented confirm action.
Standard game-owned prompt IDs such as `card_selection.TO_DISCARD` and
`card_selection.TO_TRANSFORM` may be mapped exactly; unknown/custom prompts
remain generic `select` operations and are learned from their stable prompt ID
and real transition rather than from card-name tables or localized-text
heuristics. This contract is fingerprinted as runtime mechanics encoding v6,
so checkpoints from the earlier selection ABI fail closed.

Likewise, a numeric family is a typed magnitude, not an invented verb. An
`EnergyVar(1)` is stored in the energy family, but the encoder does not label it
"gain" or "lose" unless the native model exposes that direction through another
fact. Adrenaline and Void are therefore distinguishable by stable card identity,
keywords, hover tips and lifecycle facts, but the direction is not falsely
manufactured from the positive number. This is a known boundary, not complete
effect-program coverage.

Those gaps must not be filled with description regexes, action-quality scores,
card-specific policy rules or a hand-written effect profile. Future expansion
must come from factual native instrumentation and must retain the world/action
separation.

## Checkpoint compatibility

The new encoder version, 224-feature minimum and fingerprint intentionally reject previous
checkpoints. World capacity is 512 tokens and candidate-local capacity is 24 so
larger decks and multi-effect cards fail less often; the encoder still raises on
overflow rather than silently truncating facts. Training must start with a new
model and run directory after this ABI change.
