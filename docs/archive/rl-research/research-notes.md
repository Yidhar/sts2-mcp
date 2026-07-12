# Research Notes

This file captures machine-local facts and community indicators gathered before
implementation. It is intended to reduce re-discovery work.

## Verification Date

- Verified on 2026-03-16

## Local Game Install Facts

Installed executable path:

- `E:\Program Files (x86)\Steam\steamapps\common\Slay the Spire 2\SlayTheSpire2.exe`

Observed files:

- `SlayTheSpire2.exe`
- `SlayTheSpire2.pck`
- `data_sts2_windows_x86_64\sts2.dll`
- `data_sts2_windows_x86_64\sts2.deps.json`
- `data_sts2_windows_x86_64\sts2.runtimeconfig.json`
- `data_sts2_windows_x86_64\0Harmony.dll`

Release metadata from `release_info.json`:

- version: `v0.99.1`
- branch: `v0.99.1`
- commit: `7ac1f450`
- date: `2026-03-13T20:40:28-07:00`

Observed runtime indicators:

- `GodotSharp 4.5.1` present in `sts2.deps.json`
- `MonoMod.Backports` present in `sts2.deps.json`
- `System.IO.Pipes` present in `sts2.deps.json`
- `System.Net.HttpListener` present in `sts2.deps.json`
- `System.Net.WebSockets` present in `sts2.deps.json`
- `sts2.runtimeconfig.json` targets `.NET 9`

## Local Toolchain Facts

Installed on this machine:

- `dotnet 6.0.422`
- `dotnet 8.0.204`
- `dotnet 9.0.308`
- `node v22.14.0`
- `npm 10.9.2`
- `uv 0.9.18`

## Local User Data Facts

Observed user data root:

- `C:\Users\yidhar\AppData\Roaming\SlayTheSpire2`

Observed relevant subpaths:

- `logs\`
- `steam\76561199066713284\profile1\saves`
- `steam\76561199066713284\profile1\replays`
- `localization_override\`

These directories suggest that:

- live logs are available for diagnostics
- save files can be used for offline inspection and testing
- localization overrides already have a user-data-based home

## Log Evidence

Observed in local logs:

- multiplayer error indicating `Mod mismatch`
- host mod list example: `RemoveMultiplayerPlayerLimit0.0.4A`
- warnings referencing developer console usage

Implications:

- the game already tracks mod identity in at least some code paths
- there is likely a mod metadata surface worth locating during reverse
  engineering
- there may already be internal debug or developer tooling hooks

## Community Ecosystem Indicators

Useful references discovered during planning:

- MCP specification docs: <https://modelcontextprotocol.io/introduction>
- MCP TypeScript SDK: <https://github.com/modelcontextprotocol/typescript-sdk>
- Godot mod loader: <https://github.com/GodotModding/godot-mod-loader>
- Slay the Spire 2 modding tutorial repo: <https://github.com/GlitchedReme/SlayTheSpire2ModdingTutorials>
- Tutorial index: <https://glitchedreme.github.io/SlayTheSpire2ModdingTutorials/>

## Tutorial Repo Findings

The current public Slay the Spire 2 tutorial materially reduces ambiguity for
this project.

### Repository-Level Findings

From the repository root and tutorial index:

- Slay the Spire 2 is described as natively supporting mods
- The current loading model is local loading rather than Steam Workshop
- The tutorial sections directly relevant to `sts2_mcp` are:
  - `01 环境配置`
  - `02 安装、看源码、修改`
  - `03 添加新内容`
  - `04 BaseLib`

### Environment And Packaging Findings

From `01 环境配置`:

- The page explicitly says it applies to `0.99` and later versions
- The recommended editor/runtime is `Godot 4.5.1 Mono`
- The recommended SDK level is `.NET 9` or newer
- A mod project should include a metadata file named `<modid>.json`
- The `.csproj` should reference `sts2.dll` and `0Harmony.dll`
- The minimal entrypoint uses `[ModInitializer("Init")]`
- The initializer example includes
  `ScriptManagerBridge.LookupScriptsInAssembly(typeof(Entry).Assembly)`
- `dotnet build` can be configured to copy the `dll` into the game's `mods`
  directory
- A functional mod package consists of a `dll` plus a `pck`

These findings strongly support building `sts2_mcp` as a native Slay the Spire
2 mod first, not as a generic helper assembly.

Important caveat:

- The local game build inspected for this project is now `v0.99.1`
- Therefore, tutorial details that target `0.99+` can now be checked directly
  against the installed build instead of being treated as forward-looking by
  default

### Install, Inspection, And Debug Findings

From `02 安装、看源码、修改`:

- Mods are installed under
  `xxx\\Steam\\steamapps\\common\\Slay the Spire 2\\mods`
- Save data for modded and non-modded play is separated
- `gdre_tools` can recover a Godot project from `SlayTheSpire2.pck`
- `ILSpy` or `dnSpy` can inspect `data_sts2_windows_x86_64\\sts2.dll`
- Harmony is the expected mechanism for code patching
- The in-game console opens with `~`
- Debugging can use `launch_*.bat` with `--log`
- Local runs may require `steam_appid.txt` containing `2868840`

These findings define the expected reverse-engineering and debug workflow for
the bridge.

### API Surface Findings

From `03 添加新内容`:

- The game exposes model-registration APIs for at least cards, relics, and
  potions
- New characters do not appear to have a simple official API and are described
  as patch-heavy
- Content additions can be done without `BaseLib`, but `BaseLib` is positioned
  as the future-friendly route for richer additions

For `sts2_mcp`, this means public APIs may already cover parts of bridge state
and action work, while more specialized behaviors may still require Harmony.

### BaseLib Findings

From `04 BaseLib`:

- `BaseLib` is described as a shared foundation for adding new content
- The tutorial explicitly notes that, as of 2026-03-14, patch-only mods can
  skip `BaseLib`

This is directly relevant: the first version of `sts2_mcp` should not depend on
`BaseLib`.

Community modding indicators for Slay the Spire 2:

- Nexus Mods pages reference a `mods` directory and a `Load Mods` flow
- A public mod template is referenced for Slay the Spire 2
- Community docs mention a public API package and optional config support

These are ecosystem indicators, not yet a full compatibility guarantee for this
repository. Phase 0 should validate the exact loader and packaging rules on the
installed build.

## Current Recommendation

Based on the local build and ecosystem evidence:

- Use an in-process bridge mod as the source of truth
- Use an external MCP server only as a presentation and validation layer
- Start with loopback HTTP for fast iteration
- Keep the first milestone visible-state-only
- Use a native Slay the Spire 2 mod project skeleton from day one
- Do not pull in `BaseLib` for the patch-only bridge milestone
- Do not make fullscreen or foreground-window assumptions part of the control
  contract

## MCP Settlement Notes

Observed during live agent play on 2026-03-17:

- combat actions can produce bridge-visible intermediate frames before the run
  is ready for the next agent decision
- the most important recurring case is card-draw animation:
  - after a card resolves, the first snapshot can under-report the final hand
    because newly drawn cards are still entering the state over subsequent
    frames
- this means a fixed `wait_after_ms` is not sufficient as the only settlement
  policy for agent play
- the MCP layer now needs action-family-specific settle loops, not only a
  single post-`end_turn` special case
- practical implication for future MCP tools:
  - any combat-affecting tool that can chain into draw, discard, exhaust, or a
    follow-up selection should return only after either:
    - a visible selection screen is ready
    - a non-combat screen is reached
    - or the combat snapshot has remained stable on the player's turn
- additional live breakpoint observed on 2026-03-17:
  - after a combat-ending action, the outer agent could receive an early
    out-of-combat frame before the room-end reward UI was actually surfaced as
    the next decision
  - practical consequence:
    - the agent did not immediately know that reward selection was required
    - this is a real turn-flow bug, not just a UX preference
- current mitigation now implemented in the MCP layer:
  - post-action settle verdicts no longer stop on the first non-combat frame or
    `combat_not_in_progress`
  - instead they continue polling until one of these stable decision surfaces is
    visible:
    - reward / card reward
    - rest site / deck-upgrade
    - map-ready
    - shop
    - event
    - another actionable non-combat screen
  - this should make room-end reward prompts arrive as the first stable state
    after lethal resolution, rather than as a follow-up corrective poll
- complementary mitigation now implemented in the bridge layer itself:
  - post-action snapshots do not rely only on coarse `wait_after_ms`
  - after the requested delay expires, the bridge also waits a fixed number of
    main-thread pump ticks before calling `CaptureSnapshot()`
  - the same post-action frame barrier is now reused by:
    - ordinary `perform_action`
    - reward auto-proceed cleanup
    - card-selection auto-confirm cleanup
  - practical goal:
    - reduce the older failure mode where the scene tree had not fully advanced
      into the next frame yet, so the bridge captured an early transitional
      shell
- additional non-combat breakpoint verified and fixed on 2026-03-17:
  - entering a Treasure room initially returned only a generic `proceed`
    action
  - executing that action caused no state change at all
  - this was not a timing issue:
    - `state_version_before == state_version_after`
    - the room remained `room_type = Treasure`
  - root cause:
    - the bridge was treating Treasure rooms like ordinary room-proceed flows,
      but the live game uses `NTreasureRoom`-specific handlers for chest open,
      relic pick, and exit
  - implemented bridge-side fix:
    - detect `NTreasureRoom`
    - surface `treasure:open`
    - surface `treasure_relic:{index}` with real relic descriptions
    - suppress stale `proceed` while treasure interactions remain unresolved
    - stop re-surfacing `treasure:open` after relic claim by checking:
      - `_hasRelicBeenClaimed`
      - `_isRelicCollectionOpen`
  - live regression after redeploy:
    - continue save -> `treasure:open`
    - open chest -> `treasure_relic:0` for `活动星图`
    - claim relic -> `proceed`
    - proceed -> `screen = "MAP"` and next action `map:3,9`
- additional MCP usability work validated on 2026-03-18:
  - reward / card-reward / rest-site / event / generic card-selection surfaces
    still produced avoidable outer-agent churn when the agent had to guess raw
    action-id families
  - MCP now exposes `sts2_pick_option` as an indexed abstraction across those
    surfaces
  - live regression confirmed:
    - picking a reward entry that opens card reward now settles directly onto a
      stable `CARD_REWARD_SELECTION` frame
    - picking a card reward now settles back onto a stable reward screen instead
      of returning an in-between frame
  - map traversal still needed a higher-level chain than raw `map:{col},{row}`
    because room-end cleanup and map-opening animation can lag the first map
    snapshot
  - MCP now exposes `sts2_travel_to_coordinate`
  - live regression confirmed:
    - unresolved reward flow correctly blocks travel with
      `reason = "reward_resolution_required"`
    - once rewards were fully resolved, travel settled onto the next room's
      stable decision state rather than leaving the agent on a transient map
      frame

## MCP Output Compaction Notes

Observed and tightened on 2026-03-17:

- the raw bridge payloads are now rich enough that direct passthrough is too
  expensive for routine agent turns
- live measurement on the Act 1 boss reward checkpoint:
  - raw bridge `/state`: about `40165` characters
  - compact `sts2_get_state`: about `1591` characters
  - compact `sts2_list_actions`: about `368` characters
- compaction changes now applied in the MCP layer:
  - suppress `automation:*` actions in compact action listings
  - prefer short card `effect` summaries over repeating full descriptions
  - omit null potion payloads for empty slots
  - omit `state_hash` / `captured_at_utc` from the compact state surface
  - collapse noisy intent candidate arrays down to the primary label / text /
    damage plus alternate-count telemetry
- live follow-up on 2026-03-17 exposed one more compact-card tradeoff:
  - some cards were still too compressed for tactical play if the summary kept
    only `effect_preview.summary`
  - concrete examples from the current Act 1 run:
    - `燃烧契约` compact effect only showed `draw 2`, hiding the exhaust choice
    - `愤怒` compact effect only showed `6 damage`, hiding the self-copy rule
    - `痛击` compact effect only showed `8 damage`, hiding `给予2层易伤`
    - `放血` compact effect only showed `lose 3 HP`, hiding the energy gain
  - MCP compact card summaries now keep `description` as well whenever the
    underlying rules text is materially richer than the short effect string
  - this preserves small payloads for simple cards such as `打击 / 防御`, while
    giving the agent enough rules text for multi-effect or self-modifying cards
- one bridge-side text issue is still open even after that MCP mitigation:
  - the current raw / normalized description for `放血` is duplicated as
    `获得1点能量1点能量`
  - this looks like a bridge-side text-normalization defect rather than a pure
    MCP compaction problem and should be fixed lower in the stack
- player buff visibility was also missing from the compact MCP state until a
  later 2026-03-17 follow-up:
  - raw bridge `/state` already exposed `players[0].creature.powers`
  - compact `sts2_get_state` only kept:
    - hp
    - block
    - gold
    - potions
    - relics
  - practical impact:
    - after using `铁心药水`, the agent had to infer the ongoing defensive
      effect from HP deltas instead of reading the actual player buff
  - MCP compact state now also includes `player.powers[]`
  - live regression on the current `多尼斯异鸟` elite now returns:
    - `player.powers = [{ title: \"覆甲\", amount: 2 }]`

## Map Route Planning Notes

Observed during live play on 2026-03-17:

- before route-planning improvements, agent map choice was still partly local:
  - inspect the currently travelable next points
  - then manually look a few rows ahead to compare obvious forced branches
- this is workable for simple forks but weaker than how a player usually plans
  routes:
  - players often reason over the full reachable route DAG from the current
    node to the boss / act end
- practical improvement now started in the MCP layer:
  - add `sts2_get_map_routes`
  - read the full `map.points[*].children` graph from bridge state
  - build a pruned future-only route forest rather than dumping all nodes flat
    into the model context
  - prune rules:
    - exclude the current node itself
    - exclude rows at or before the current row
    - exclude nodes that are no longer reachable from the currently travelable
      frontier
  - return one root per currently travelable next step plus a deduplicated node
    table for the whole reachable future graph
  - each node only keeps:
    - `key` (`col,row`)
    - `point_type`
    - `child_keys`
- intent:
  - make route evaluation explicit and reviewable
  - stop relying on an implicit “look 2 to 4 nodes ahead” mental shortcut
  - let the route forest naturally shrink as the run advances, so only the first map
    decision tends to expose the largest context
  - avoid repeated expansion of shared suffixes, which was inflating the first
    live nested-tree response to roughly `35k` characters; normalized output on
    the same map measured roughly `7.4k`
- live follow-up validation after choosing one opening path and clearing the
  first room:
  - map returned at current coord `3,1`
  - `sts2_get_map_routes` exposed exactly one immediate root: `2,2`
  - the payload excluded the current node and all rows at or before row `1`
  - branches that were only reachable from the discarded opening choices were
    no longer present
  - measured payload size dropped again to about `5479` characters, roughly
    `1.37k` tokens
- second-pass map tool shaping now implemented:
  - default MCP output mode is now `detail = "summary"` instead of always
    shipping the full reachable node table
  - `detail = "full"` still exists for explicit graph inspection
  - each route root now exposes a compact decision summary:
    - forced path length before the next branch
    - total reachable counts by point type
    - shortest distance from the current choice to each point type
    - whether some path can reach an elite and then later a rest site
  - current live measurement at coord `3,1`:
    - summary payload about `1336` characters
    - full payload about `6268` characters
  - map reads now use a small settle loop before returning:
    - poll until the map screen is open and not currently traveling
    - cross-check travelable frontier nodes against legal `map:*` actions
    - require repeated identical map snapshots to reduce transition-frame
      leakage
- run-aware map tags now added for route planning:
  - the map payload now carries a compact `run_context` block so the outer
    agent does not need to reconstruct current hp, gold, potion capacity, deck
    size, and relic count from raw state every time it compares routes
  - each route root now carries `run_aware` tags that combine route topology
    with current run state:
    - rest pressure
    - shop access value
    - potion capacity value
    - elite viability score/rating
    - compact reason tags such as `hp_healthy` or `elite_rest_after`
  - current live measurement at coord `3,1`:
    - summary payload about `2350` characters
    - full payload about `7282` characters
- multi-card combat sequencing now started in the MCP layer:
  - a single `sts2_play_card_sequence` call can now carry multiple planned
    `play_card:*` actions from the current combat state
  - this is intentionally not a scripted autoplay policy:
    - the outer agent still chooses the sequence
    - MCP only handles post-play hand reindexing and rematching
  - rematching uses the originally selected card payload and target payload
    rather than trusting stale `hand_index` values alone
  - this should be especially useful for turns that chain:
    - draw effects
    - energy gain
    - hand shrink / hand reorder
    - or mid-turn card-selection interruptions
  - live combat validation now completed on 2026-03-17:
    - first successful remap:
      - requested `play_card:0:4:1`
      - executed `play_card:0:3:1`
      - tool returned `match_type = "reindexed"`
    - later in the same fight:
      - requested final strike `play_card:0:4:1`
      - executed final strike `play_card:0:2:1`
      - post-action settlement returned `reward_flow_ready`
      - the sequence tool returned directly on the reward screen after combat
  - this is enough to confirm that sequential hand-index drift is being handled
    in live play, not only in synthetic non-combat checks
  - later live failure also confirmed an additional matching edge case:
    - requested line:
      - first `痛击`
      - then `打击` on the same target
    - after `痛击` applied `易伤`, the later `打击` preview changed from
      `6 damage` to `9 damage`
    - the original play-card fingerprint had been using dynamic
      `effect_preview` values as part of the hard identity match
    - result:
      - the strike was still legally playable
      - but the sequence tool incorrectly returned it as unavailable after
        reindex
  - local mitigation now implemented on 2026-03-17:
    - play-card hard matching now ignores transient preview damage/block fields
    - hard fingerprint now keeps only more stable card-identity fields:
      - id / title / type / rarity / target type
      - canonical cost fields
      - normalized description shape
      - dynamic-var base / enchanted signatures
    - this should allow rematching to survive temporary preview drift from:
      - `易伤`
      - `力量`
      - and similar target-state-dependent combat effects
  - current status:
    - local syntax check passes:
      - `node --check packages/mcp-server/index.js`
    - live post-patch validation now completed on 2026-03-17:
      - regression setup:
        - enemy `蛇行扼杀者`
        - planned line:
          - `痛击`
          - `打击`
          - `打击`
      - observed result:
        - after `痛击` applied `易伤`, the next `打击` still rematched and
          executed
        - the tool reported `match_type = "reindexed_ambiguous"`
        - this confirms the old preview-drift fingerprint bug was fixed in
          live combat, not only locally
      - important correction to the original test assumption:
        - the third strike did not fail because of fingerprint drift
        - it failed because the sequence had already spent all `3` energy:
          - `痛击 = 2`
          - `打击 = 1`
          - remaining energy `= 0`
        - the target was left at `1 HP`, so the final strike was simply no
          longer legal
      - follow-up in the same run:
        - next turn killed the enemy
        - room rewards were resolved successfully in one MCP call:
          - take `19 gold`
          - pick `愤怒`
          - auto-return to `screen = "MAP"`
- combat sequencing follow-up was expanded locally on 2026-03-18:
  - a new MCP tool `sts2_execute_combat_sequence` now generalizes the old
    card-only sequence flow
  - one sequence can currently mix:
    - `play_card:*`
    - `use_potion:*`
    - `end_turn`
  - `end_turn` is normalized to execute last even if the caller places it
    earlier in the request
  - practical reason:
    - this reduces a real failure mode where parallel or loosely ordered tool
      calls could end the turn before earlier defensive or offensive steps
      resolved
  - `sts2_perform_action`, `sts2_play_card_sequence`,
    `sts2_execute_combat_sequence`, and `sts2_end_turn` now also accept
    `return_state_after = true`
  - this keeps the normal compact `state` summary for token efficiency while
    allowing a same-call full raw post-action snapshot under `state_after`
  - compact combat state now also exposes allied summons under
    `combat.summons[]`
  - interaction hints now explicitly recommend sequence tools and flag parallel
    combat `sts2_perform_action` usage as something to avoid
- later live reward regression also confirmed that full reward-list
  left-collapse is handled correctly:
  - on the floor-7 `小啃兽` reward test, gold -> potion -> card reward each
    reindexed back down to `reward:0`
  - `sts2_resolve_room_rewards` still correctly claimed:
    - `16 gold`
    - `铁心药水`
    - `欺凌`
    - then auto-returned to `screen = "MAP"`

## Relic Trigger Visibility Note

Observed during live floor-7 combat on 2026-03-17:

- killing the vulnerable `小啃兽` with the current relic set triggered
  `地精之角`
- the resulting combat state clearly reflected the mechanical outcome:
  - player energy returned from `0` to `1`
  - hand size increased by one card
- however the MCP / bridge payload still did not explicitly attribute that
  state delta to the relic trigger itself

Implication:

- the outer agent can infer the result from changed resources
- but it still cannot read a direct explanation such as:
  - `enemy died`
  - `地精之角 triggered`
  - `+1 energy, draw 1`

This remains a visibility / explanation gap rather than a legality bug.

## Shop Reindex Notes

Observed during live shop testing on 2026-03-17:

- the bridge correctly exposed shop open / buy actions and raw shop item payloads
- concrete live reproduction before the MCP-side fix:
  - initial open-shop actions included:
    - `shop:buy:1` = `铁斩波`
    - `shop:buy:10` = `Remove a card`
  - after buying `铁斩波`, the shop reindexed immediately:
    - `shop:buy:9` = `Remove a card`
    - `shop:buy:10` = `能力药水`
  - any batching logic that reused the cached original `shop:buy:10` would buy
    the wrong item
- raw bridge shape confirmed in both `available_actions[*].item` and `shop.items[*]`:
  - `cost`
  - `item_kind`
  - `is_affordable`
  - nested `card` / `relic` / `potion` payloads when applicable
- important MCP bug found while implementing the fix:
  - compact `sts2_get_state` shop summaries had been reading the wrong field
    names:
    - `price`
    - `item_type`
    - `can_buy`
  - these do not exist on the current bridge payload
  - the correct fields are:
    - `cost`
    - `item_kind`
    - `is_affordable`
- local MCP mitigation now implemented on 2026-03-17:
  - new tool: `sts2_resolve_shop_visit`
  - purchase planning now stores a stable shop-item fingerprint rather than
    trusting positional `shop:buy:{index}` ids after the first purchase
  - current fingerprint contents:
    - `item_kind`
    - normalized `title`
    - normalized `description`
    - `cost`
    - nested `card` / `relic` / `potion` ids and compact effect fields when
      present
  - after each purchase, MCP re-reads the current `shop:buy:*` actions and
    rematches the next requested purchase against the live shop contents
  - shop-driven card removal can now be continued in the same tool call:
    - by `remove_card_title`
    - or by `remove_card_index`
- current status:
  - local syntax check passes:
    - `node --check packages/mcp-server/index.js`
  - post-reload live regression is still pending

## Reward Reindex Notes

Observed during live boss-reward resolution on 2026-03-17:

- bridge reward actions currently use positional ids:
  - `reward:0`
  - `reward:1`
  - `reward:2`
- after one reward is claimed, the visible reward list can collapse and reindex
  immediately
- concrete live failure:
  - after claiming earlier rewards, the remaining potion reward became the new
    `reward:0`
  - any outer logic that still tried to use the original cached reward id would
    miss it
- implementation consequence:
  - the outer agent must not cache reward indices across multiple reward claims
  - room-end reward batching must always re-read the current reward surface
    between steps
- follow-up mitigation now applied in MCP:
  - `sts2_resolve_room_rewards` remains the preferred path instead of multiple
    raw `reward:*` calls from the outer agent
  - reward auto-proceed now treats post-claim `discard_potion:*` actions as
    ignorable cleanup noise when the reward list itself is already empty and the
    true next step is `proceed`

## Potion Integration Notes

Observed locally on 2026-03-17 while implementing potion control:

- direct PowerShell reflection against `sts2.dll` was not reliable for this
  task because the assembly targets `.NET 9`; a temporary `net9.0` console was
  used to inspect the live signatures instead
- confirmed signatures:
  - `UsePotionAction(PotionModel potion, Creature target, bool isCombatInProgress)`
  - `DiscardPotionGameAction(Player player, uint potionSlotIndex, bool isCombatInProgress)`
  - `PotionModel.EnqueueManualUse(Creature target)`
  - `PotionModel.Discard()`
- practical implementation choice:
  - prefer model-level entrypoints first:
    - `PotionModel.EnqueueManualUse(...)`
    - `PotionModel.Discard()`
  - keep protected `GameAction.ExecuteAction()` as a reflection fallback only
- the current live bridge now exposes legal potion actions without desktop
  automation:
  - `use_potion:{player_index}:{slot_index}:{target}`
  - `discard_potion:{player_index}:{slot_index}` when `Player.CanRemovePotions`
    is true
- live validation on the resumed floor-9 combat confirmed:
  - enemy-target potion execution worked through MCP:
    - `use_potion:0:1:1`
  - after potion use, the generalized MCP settle loop waited until the player
    turn was stable and the hand had fully drawn to 5 cards

## Intent Ambiguity Notes

Observed during live play on 2026-03-17:

- root cause was later confirmed and partially fixed the same day:
  - the bridge had been preferring `MonsterModel.GetIntents()`
  - on the current game build, that method exposes the monster's broader move
    library / candidate definitions instead of the already selected current-turn
    move
  - bridge fix now applied:
    - prefer `monster.NextMove.Intents`
    - use `monster.GetIntents()` only as a fallback when `NextMove` is missing
- live regression after the fix:
  - `MONSTER.INFESTED_PRISM`
  - `intent.state_id = "JAB_MOVE"`
  - previous payload incorrectly listed the whole candidate set:
    - `22`
    - `16`
    - `9x3`
    - `守势`
    - `强化`
  - current payload now correctly exposes only the resolved current-turn intent:
    - single attack for `22`
- historical failures that motivated this fix:
  - for `MONSTER.FUZZY_WURM_CRAWLER` with `intent.state_id = "INHALE"`, the
    old bridge payload mixed attack entries (`total_damage = 11`) with a buff
    entry, and the live turn executed the buff/non-attack branch
  - for `MONSTER.FOGMOG` with `intent.state_id = "ILLUSION_MOVE"`, the old
    bridge payload mixed summon, attack, and buff entries, while the live turn
    actually resolved as summon:
    - the enemy spawned `MONSTER.EYE_WITH_TEETH`
    - the practical meaning was "召唤怪物", not "可能 8 / 14 攻"
- remaining semantic gaps after the current-turn fix:
  - some resolved turns still need richer translation than raw intent classes
    alone provide
  - user-verified examples:
    - the summoned `MONSTER.EYE_WITH_TEETH` intent was not just generic
      `StatusCard`; it effectively meant "add X status cards", with
      `X = 3` in that case
    - the `MONSTER.FOGMOG` follow-up turn should be read as a compound resolved
      action equivalent to "8 damage + 强化"
- practical consequence for agent play:
  - after the current fix, the agent can treat `enemy.intent.intents[]` as the
    current move's intent list rather than as the monster's whole move library
  - however, further normalization is still valuable for compound, summon, and
    status-card-producing turns
- queued follow-up work:
  - expose higher-level normalized fields such as:
    - `intent.resolved_action_kind`
    - `intent.resolved_total_damage`
    - `intent.resolved_summon_targets`
    - `intent.resolved_status_payload`
    - `intent.resolved_status_card_count`
    - `intent.resolved_buff_payloads`

## Damage Estimation Notes

Observed during live play on 2026-03-17:

- the current combat action payloads expose legal playability and targets, but
  they do not expose a normalized expected-damage preview for the current board
  state
- concrete failure observed:
  - on a floor-9 cleanup turn, an enemy at `9 HP` was treated as a one-hit
    lethal candidate because the outer agent had to infer damage from prior
    observations

## X-Cost Preview Notes

Observed and fixed during live elite regression on 2026-03-17:

- the earlier bridge payload for `CARD.WHIRLWIND` (`旋风斩+`) exposed only the
  per-hit damage shell and did not map current combat energy into hit count
  or total damage
- practical failure before the fix:
  - with current energy `= 3`, the hand preview still looked like plain
    `10 damage`
  - this was not sufficient for an agent to decide whether the card was worth
    playing over ordinary 1-cost alternatives
- bridge fix now applied:
  - `SafeResolveCardEnergyXValue(...)` now falls back to the current owner's
    combat energy when `ResolveEnergyXValue()` does not report a usable value
  - X-cost attack cards that read like "X次" / "X times" are normalized as
    `repeat_per_energy`
  - the bridge now expands:
    - `effect_preview.hits`
    - `effect_preview.total_damage`
    - `effect_preview.summary`
    - `effect_preview.x_cost_semantics`
- live regression after the fix:
  - same combat state, current energy `= 3`
  - `旋风斩+` now returns:
    - `summary = "10 x 3 damage + X=3"`
    - `total_damage = 30`
    - `damage_per_hit = 10`
    - `hits = 3`
    - `x_cost_value = 3`
    - `x_cost_semantics = "repeat_per_energy"`
- adjacent semantic note surfaced during the same replay:
  - `感染棱柱`'s `活力火花` did not refund energy when the first attack only
    removed block
  - current interpretation:
    - the refund triggers on actual attack damage to HP, not merely on hitting
      through blockless attack contact
  - the actual `打击` damage for that board state was only `6`, so the target
    survived at `3 HP` and the fight required one more turn
- practical consequence:
  - without a bridge field such as `card.estimated_damage` or an action-level
    `expected_damage_total`, the agent has to infer combat math from prior turns
    and can mis-evaluate exact lethal lines
- a second live failure confirmed the broader issue is not just missing lethal
  math, but missing resolved hand-effect metadata:
  - during floor-12 combat on 2026-03-17, the outer agent had to reconstruct
    card values from memory instead of reading them from the bridge payload
  - the real board-state values were:
    - `打击 = 6 damage`
    - `愤怒 = 6 damage`
    - `铁斩波 = 5 damage + 5 block`
    - `飞剑回旋镖 = 3 x 3 damage` in single-target combat
  - the bridge already exposes `resolved_energy_cost`, but that is not enough
    for reliable tactical reasoning; the outer agent also needs resolved effect
    previews per card/action for the current board state
- preferred bridge surface to investigate:
  - card-level resolved preview fields such as:
    - `card.preview_damage_total`
    - `card.preview_block`
    - `card.preview_draw`
    - `card.preview_hits`
    - `card.preview_status_cards`
  - or an action-level normalized preview bundle that covers the same data for
    the exact `play_card:*` action actually being offered
- follow-up work to queue later:
  - inspect whether `CardModel` or the combat preview system exposes a current
    resolved-damage calculation that can be surfaced safely
  - 2026-03-17 implementation note:
    - `CardModel` does expose preview-oriented hooks that are usable from the
      bridge:
      - `GetDescriptionForPile(...)`
      - `UpdateDynamicVarPreview(...)`
      - `DynamicVars`
    - the bridge has now been updated in source so `card` payloads no longer
      rely on raw `LocString.ToString()`:
      - `description` resolves through the game's own card-description path
      - `dynamic_vars[]` exposes previewed values such as damage/block/repeat
      - `effect_preview` summarizes common tactical fields directly
    - this still needs live deployment + validation against a running combat to
      confirm the preview numbers match the rendered hand in all cases
  - inspect whether the same preview system can surface non-damage outputs such
    as block, draw count, exhaust count, and status-card creation
  - if not, consider an MCP-side heuristic only as a fallback, not as the
    source of truth

## Local Scaffold Verification

The repository now contains a staged native mod skeleton at:

- `mods/sts2-bridge`

Verified locally on 2026-03-16:

- `dotnet build .\mods\sts2-bridge\sts2-bridge.csproj -p:Sts2SkipDeploy=true`
  succeeded
- The scaffold references the local `sts2.dll` and `0Harmony.dll`
- The first code milestone includes a loopback `GET /health` endpoint

Live build findings that now supersede the earlier `v0.98.3` assumptions:

- `LookupScriptsInAssembly` is present in the local `sts2.dll` metadata string
  table, but it did not resolve as a usable symbol in the current bridge build
  when compiled against the local reference set
- The bridge entrypoint therefore still omits that call for now
- The local `sts2.dll` string table contains manifest keys
  `has_pck`, `has_dll`, and `affects_gameplay`
- The staged manifest was updated to remove the speculative `pckName` field
- The staged bridge project currently builds as a plain `.NET 9` class library
  via `Microsoft.NET.Sdk` because the offline `Godot.NET.Sdk` resolver is not
  reliable enough for the bootstrap milestone
- The first real load test showed that `http://127.0.0.1:27100/` conflicts with
  an existing registration on this machine, so the bridge now probes a small
  fallback port range instead of failing closed on a single fixed port
- Because the bridge port is no longer guaranteed to be fixed, the bridge now
  needs a stable discovery handoff such as `%APPDATA%\\SlayTheSpire2\\bridge\\session.json`
- The first Codex bootstrap timeout for `packages/mcp-server` was traced to a
  transport mismatch: the staged server initially used `Content-Length`
  framing, but the MCP stdio transport expected by Codex is newline-delimited
  JSON-RPC
- The staged MCP server now writes bootstrap diagnostics to
  `%APPDATA%\\SlayTheSpire2\\bridge\\mcp-stdio.log` so future startup failures
  can distinguish "process never launched" from "launched but never received
  initialize"
- The first live loader smoke test is intentionally configured as `dll`-only by
  setting `has_pck=false`; this is a project decision that still needs runtime
  confirmation against the live loader

## Live Bridge Verification On v0.99.1

Observed locally on 2026-03-16 against the current staged bridge:

- `%APPDATA%\\SlayTheSpire2\\bridge\\session.json` is being written again
- `GET /health` returns `200 OK`
- The live session currently reports:
  - `bridge_version = 0.2.1`
  - `base_url = http://127.0.0.1:27100/`
  - `preferred_port = 27100`
- `GET /state` and `GET /actions` both return `ok = true`
- The same read operations still work while `Slay the Spire 2` is not the
  foreground window on the desktop
- On the current title-screen surface, the bridge now reports:
  - `screen = "MAIN_MENU"`

## 2026-03-16 Additional Runtime Findings

### MCP Server Progress

- `packages/mcp-server/index.js` has now advanced to:
  - `SERVER_VERSION = 0.4.4`
- `sts2_end_turn` already includes post-action settlement polling so combat
  turns do not return a half-finished enemy-resolution frame
- `sts2_resolve_room_rewards` already batches room-end reward resolution in one
  call
- `sts2_resolve_rest_site` now applies the same pattern to campfires:
  - choose a campfire option
  - optionally resolve a smith upgrade card choice
  - auto-advance `rest_site:proceed`
  - stop once the run is back on a travel-enabled map
- `sts2_resolve_card_selection` now applies the same low-round-trip pattern to
  visible `card_selection` flows:
  - the bridge still exposes raw `card_selection:*` actions
  - `sts2_perform_action` now emits an `interaction_hints.card_selection`
    summary when an action lands in card selection
  - the agent can then submit all chosen `select_indices` in one MCP tool call
  - MCP executes those selections internally and then performs a
    `terminal_action` such as `confirm`
  - selections are applied from highest to lowest requested index to tolerate
    UI reindexing when selected cards disappear from the visible option list

### Stale Rest-Site Exposure After Proceed

Observed on the live floor-6 campfire checkpoint:

- after the user clicked the visible campfire `Proceed` button, the bridge
  still reported:
  - `screen = "MAP"`
  - `map.is_open = true`
  - `map.is_travel_enabled = true`
  - stale `rest_site.visible = true`
  - stale `rest_site.proceed_visible = true`
  - stale legal action `rest_site:proceed`
- the playable state was already the map, so the stale campfire data is a
  bridge normalization bug rather than a true gameplay block

Implications:

- room-resolution tools must not trust raw `rest_site` visibility alone
- MCP-side room resolvers should prefer a terminal `MAP` verdict when travel is
  enabled
- bridge-side node visibility needs one extra guard because `NRestSiteRoom`
  remains visible in the scene tree after the room is effectively done

Source status:

- `mods/sts2-bridge/Scripts/BridgeGameApi.cs` now contains a staged source fix
  that suppresses `rest_site` payloads and `rest_site:*` actions when
  `NMapScreen.IsOpen == true`
- the same staged patch also makes the fallback screen resolver prefer `MAP`
  over a lingering visible rest-site room
- the patched bridge project was rebuilt successfully with:
  - `dotnet build .\\mods\\sts2-bridge\\sts2-bridge.csproj -p:Sts2SkipDeploy=true`

## 2026-03-16 Checkpoint: Live Bridge Deploy 0.7.6

Observed locally after closing the game, deploying the rebuilt DLL, and
restarting into the same floor-8 save:

- the live session now reports:
  - `bridge_version = 0.7.6`
  - `schema_version = 2026-03-16.11`
- campfire ingress is now cleaner than before:
  - `screen = "REST_SITE"`
  - visible legal actions are only:
    - `rest_site:0` (`HEAL`)
    - `rest_site:1` (`SMITH`)
  - `rest_site:proceed` is no longer exposed while the room still has selectable
    campfire options
- smith flow still works through the normalized deck-upgrade surface:
  - `rest_site:1` opens `screen = "DECK_UPGRADE_SELECTION"`
  - selecting `deck_upgrade:select:9` upgraded `痛击`
  - `deck_upgrade:confirm` returned to `screen = "REST_SITE"`
  - only legal non-automation action after confirm was `rest_site:proceed`
- the previous hard blocker is now fixed live:
  - `rest_site:proceed` advances successfully
  - resulting state:
    - `screen = "MAP"`
    - `map.is_open = true`
    - `map.is_travel_enabled = true`
    - next legal map action `map:3,8`

Additional source/deploy checkpoint:

- the bridge now includes a generic `card_selection` payload and action surface
  for intermediate card-pick overlays and combat-local hand-selection modes
- covered visible node types:
  - `NSimpleCardSelectScreen`
  - `NChooseACardSelectionScreen`
  - `NDeckCardSelectScreen`
  - `NPlayerHand` when `IsInCardSelection == true`
- exposed actions:
  - `card_selection:select:{index}`
  - `card_selection:confirm`
  - `card_selection:cancel`
  - `card_selection:close`
  - `card_selection:skip`
- exposed selector constraints:
  - `selected_count`
  - `min_select`
  - `max_select`
  - `requires_manual_confirmation`
  - `cancelable`
- while `card_selection.visible = true`, ordinary combat `play_card:*` and
  `end_turn` actions are suppressed so the agent does not keep acting on stale
  combat controls
- this new card-selection bridge surface is compiled and deployed, but still
  needed a fresh live reproduction with `燃烧契约` (or another multi-step card)
  to confirm the exact overlay type and action sequence in combat

### Live Validation: Burning Pact Uses Combat-Local Hand Selection

Observed locally on 2026-03-17 after deploying the updated bridge and resuming
the floor-9 combat save:

- `play_card:0:4:self` for `燃烧契约` no longer leaves the bridge blind
- the real interaction mode is not a `CardSelection.*Screen` overlay
- instead, combat enters a hand-local selection state on `NPlayerHand`
  - `NPlayerHand.IsInCardSelection == true`
  - the prompt is exposed from `_selectionHeader`
  - the confirm button is exposed from `_selectModeConfirmButton`
- the bridge now reports:
  - `screen = "CARD_SELECTION"`
  - `card_selection.visible = true`
  - `card_selection.screen_type = "NPlayerHand"`
  - prompt text matching the visible UI:
    - `选择1张牌来消耗。`
  - only `card_selection:*` actions, with stale combat `play_card:*` and
    `end_turn` actions suppressed
- live action sequence confirmed:
  - `card_selection:select:3`
  - `card_selection:confirm`
  - result:
    - bridge returned to `screen = "COMBAT"`
    - selected `打击` moved into the exhaust pile
    - the hand refilled after the draw effect
    - ordinary combat actions became legal again
- follow-up bridge refinement after this validation:
  - single-select flows now auto-confirm when:
    - `min_select = 1`
    - `max_select = 1`
    - at least one card is selected
    - `card_selection:confirm` is currently visible
  - this no longer depends on `requires_manual_confirmation`
  - multi-select and arbitrary-count selection flows remain manual so the
    agent can reason about the exact subset before confirming

## Late-Session Validation On 2026-03-16

Additional live validation completed later the same day:

- the MCP server source is now at `SERVER_VERSION = 0.4.2`
- local JSON-RPC bootstrap against
  `packages/mcp-server/index.js` succeeds with newline-delimited stdio framing
- `sts2_end_turn` previously had a real usability gap:
  - the underlying bridge action often returned a half-settled combat frame
  - observed bad snapshots included:
    - enemy-resolution frames with only `automation:start_autoslay`
    - early player-turn frames with energy restored but only a partial hand
- the MCP server now compensates for that gap in `sts2_end_turn` by polling
  until one of these is true:
  - combat has ended or the screen has changed out of `COMBAT`
  - the next player turn is fully actionable with a stable hand snapshot
- live combat validation on `ENCOUNTER.SHRINKER_BEETLE_WEAK` confirmed the fix:
  - `sts2_end_turn` returned:
    - `post_action_settled = true`
    - `post_action_settle_reason = "player_turn_ready"`
    - `post_action_settle_polls = 14`
  - the returned state was the real next player turn with:
    - `round_number = 2`
    - `energy = 3`
    - `hand.count = 5`
- reward batching through MCP has now been validated in two separate rooms:
  - floor 3 reward flow:
    - claimed gold
    - claimed `POTION.FLEX_POTION`
    - selected `CARD.SWORD_BOOMERANG`
    - auto-proceeded back to `screen = "MAP"`
  - floor 4 reward flow:
    - claimed gold
    - selected `CARD.BURNING_PACT`
    - auto-proceeded back to `screen = "MAP"`
- current run checkpoint at the time of this note:
  - current map coord `(6, 3)`
  - next legal travel action `map:6,4`
  - HP `78/91`
  - gold `143`
  - deck count `13`
  - current known added cards:
    - `愤怒`
    - `飞剑回旋镖`
    - `燃烧契约`
  - current potion inventory includes `POTION.FLEX_POTION`
- one combat-semantics note remains open:
  - `ENCOUNTER.SHRINKER_BEETLE_WEAK` currently exposes intent components
    `DebuffStrong, Attack:7, Attack:13`
  - live damage observations suggest that the actionable hit may correspond to
    the `13` value while the extra `7` still needs interpretation
  - do not yet build intent-driven combat heuristics that assume those dual
    attack entries are independently actionable

## New Bridge Gap: Hand-Selection Combat Screens

Observed live on 2026-03-16 during `ENCOUNTER.SLIMES_NORMAL` after playing
`CARD.BURNING_PACT`:

- the real game enters a visible hand-selection overlay with the prompt:
  - `选择1张牌来消耗`
- the bridge does not currently surface that overlay at all
- the bridge continues to report:
  - `screen = "COMBAT"`
  - ordinary hand `play_card:*` actions
  - ordinary `end_turn`
- those exposed actions are stale / misleading once the selection overlay is up
- the bridge state also shows a strong secondary signal that the card effect is
  suspended mid-resolution:
  - `CARD.BURNING_PACT` remains in `play_pile`
  - no follow-up card-selection payload is present
  - subsequent normal combat actions do not advance `state_version`

Implication:

- combat cards that require a second card-selection step are not yet safe to
  expose through the current MCP action contract
- the bridge needs an explicit normalized payload and legal-action surface for
  `NCardSelectionScreen` (or equivalent card-pick overlays)
- once that screen is visible, ordinary combat `play_card:*` and `end_turn`
  actions must be suppressed until the selection is resolved
- The live bridge has since been advanced to:
  - `bridge_version = 0.3.1`
  - `schema_version = 2026-03-16.2`
- The live MCP server now exposes:
  - `sts2_get_bridge_status`
  - `sts2_get_state`
  - `sts2_list_actions`
  - `sts2_perform_action`
  - `sts2_end_turn`
  - `sts2_wait_for_change`
- The bridge now exposes the game's built-in `AutoSlay` as structured
  automation state and actions:
  - `automation:start_autoslay`
  - `automation:stop_autoslay`
- A full successful run was launched through the MCP tool surface on
  2026-03-16 using the new `packages/mcp-server/autoslay-runner.js` client
- That AutoSlay path is being retained as a regression harness only; the target
  product still expects the external agent to make decisions itself
- During that run, a transient non-combat `PlayerCombatState.MaxEnergy`
  null-reference was observed while polling `/state`
- That capture bug has now been fixed in source and deployed in the current
  `0.3.1` bridge build
  - `run.has_run = false`
  - stable `main_menu:*` legal actions for continue and menu navigation

Operational implications:

- Read-side bridge access has now been proven to be independent of window focus
- Screenshot-driven automation is not a reliable source of truth when other
  monitors or unrelated windows are visible
- Main-menu bootstrap no longer requires desktop-coordinate assumptions
- Remaining early-run gaps are now the smaller follow-up cases around popup
  semantics and any post-main-menu transitions that still need coverage

## Bootstrap And Rest-Site Gaps On 0.5.0

Observed locally on 2026-03-16 after deploying the current `0.5.0` bridge:

- The bridge source and live session now report:
  - `bridge_version = 0.5.0`
  - `schema_version = 2026-03-16.4`
- The staged source now includes:
  - `shop` state plus `shop:*` actions
  - enemy intent payloads on combat enemies
- Live verification of those new surfaces is still blocked on earlier bootstrap
  gaps rather than on compile or deploy issues.

Campfire findings:

- `main_menu:continue` successfully resumed a live save into
  `screen = "REST_SITE"`
- `rest_site:1` successfully opened the campfire upgrade flow
- The resulting visible overlay is `NDeckUpgradeSelectScreen`
- That overlay is not yet normalized into bridge state/actions, so the agent
  cannot currently finish a smith choice through MCP alone
- `rest_site:proceed` can appear while the room is still unfinished; it should
  not yet be treated as a reliable legal action until the campfire overlay
  state is modeled more precisely

Fresh-run bootstrap findings:

- `main_menu:singleplayer` does not jump directly to character selection
- The game first enters a run-mode selection page
  (for example `标准模式 / 每日挑战 / 自定模式`)
- That page is not yet represented in the bridge schema
- During that transition the bridge can temporarily report `screen = "MAIN_MENU"`
  with no stable `main_menu:*` actions even though the visible UI is no longer
  the title menu
- Once the mode-selection page is passed, the bridge again exposes structured
  `character_selection` state and legal `embark` actions
- In that intermediate state, the top-level `screen` still remains
  `MAIN_MENU`; this is now a confirmed normalization bug rather than a loading
  race

AutoSlay implications:

- AutoSlay remains useful as a regression harness once a compatible surface is
  reached
- It is not safe to treat AutoSlay as a substitute for the missing bootstrap
  surfaces
- Starting AutoSlay after a run had already been created caused it to assume
  `/root/Game/RootSceneContainer/MainMenu` still existed
- When that lookup failed, AutoSlay logged a hard failure and the bridge
  session was lost with the game process

Practical conclusion:

- The next bridge work should prioritize:
  - run-mode selection bootstrap semantics
  - campfire upgrade overlay semantics
  - only then final live verification of `shop` and enemy intent payloads

## Staged Source Update On 0.6.0

Observed locally on 2026-03-16 in source, before live deployment:

- `BridgeGameApi.cs` now compiles with two additional structured surfaces:
  - `run_mode_selection`
  - `deck_upgrade_selection`
- `BridgeRuntime.cs` was bumped to:
  - `bridge_version = 0.6.0`
  - `schema_version = 2026-03-16.5`
- The staged `run_mode_selection` work now:
  - detects visible `NSingleplayerSubmenu`
  - normalizes `screen = \"RUN_MODE_SELECTION\"`
  - exposes stable legal actions:
    - `run_mode:standard`
    - `run_mode:daily`
    - `run_mode:custom`
    - `run_mode:back`
  - suppresses misleading `main_menu:*` actions while that submenu is active
- The staged `deck_upgrade_selection` work now:
  - detects visible `NDeckUpgradeSelectScreen`
  - normalizes `screen = \"DECK_UPGRADE_SELECTION\"`
  - serializes visible upgrade-card options and confirm/cancel/close visibility
  - exposes stable legal actions:
    - `deck_upgrade:select:<index>`
    - `deck_upgrade:confirm`
    - `deck_upgrade:cancel`
    - `deck_upgrade:close`
  - suppresses misleading `rest_site:*` actions while that overlay is active
- This `0.6.0` source update has compiled locally, but at the time of this
  note it has not yet been deployed into the live game process

## Live Reward Flow Verification On 0.7.4

Observed locally on 2026-03-16 after the bridge and MCP reward tool were
extended again:

- the live bridge now reports:
  - `bridge_version = 0.7.4`
  - `schema_version = 2026-03-16.10`
- room-end reward states now normalize as:
  - `screen = "REWARDS"`
  - `screen = "CARD_REWARD_SELECTION"`
- the reward payload now exposes fixed card choices directly from the first
  reward screen under `rewards.rewards[*].reward.cards`
- the reward payload also now exposes a terminal-reward flag:
  - `rewards.terminal_proceed_visible`
- `packages/mcp-server/index.js` is now at:
  - `SERVER_VERSION = 0.4.1`
- a live one-call reward-resolution test succeeded through the MCP tool
  `sts2_resolve_room_rewards` with:
  - `pick_card_index = 2`
  - `auto_proceed = true`
- that live call successfully:
  - claimed the gold reward
  - opened the card reward screen
  - selected `愤怒`
  - auto-advanced the run back to `screen = "MAP"`

Important current interpretation:

- the bridge now suppresses stale terminal-reward semantics once the map is
  already open, so the reward tool no longer needs to guess whether a visible
  `proceed` is still meaningful
- the outer agent can now treat the reward tool's `final_state` as the next
  real decision surface instead of needing a second cleanup request for the
  reward page itself

Remaining known limitation:

- potion rewards still become a separate decision point when potion slots are
  already full because the bridge does not yet expose the replacement/drop flow

## Main Menu Continue And Abandon Verification

Historical note:

- this section records the first live `0.2.1` verification pass
- the current deployed bridge is now `0.3.1`

Observed locally on 2026-03-16 after deploying the `0.2.1` bridge update:

- The bridge now exposes a structured `main_menu` payload that includes:
  - the continue-run card and save summary
  - visible main-menu buttons
  - the abandon-run confirmation popup when it is open
- `GET /actions` on the title screen now exposes stable legal actions such as:
  - `main_menu:continue`
  - `main_menu:abandon_current_game`
  - `main_menu:timeline`
  - `main_menu:settings`
  - `main_menu:compendium`
  - `main_menu:quit`
- A live `POST /action` using `main_menu:abandon_current_game` succeeded and
  moved the bridge into `screen = "ABANDON_RUN_CONFIRM"`
- The current popup button texts are localized as:
  - `不了`
  - `好的`
- The bridge now normalizes those localized popup labels into stable actions:
  - `main_menu:cancel_abandon_run`
  - `main_menu:confirm_abandon_run`
- A live `POST /action` using `main_menu:cancel_abandon_run` returned safely to
  `MAIN_MENU` without abandoning the save
- A live `POST /action` using `main_menu:continue` resumed the save end-to-end
  and returned the bridge to `screen = "COMBAT"` in the current encounter

## Character Select And Embark Findings

Historical note:

- this section records the first post-`0.2.0` character-select validation
- the live bridge has since moved forward substantially

Observed locally on 2026-03-16 after deploying the `0.2.0` bridge update:

- The bridge can now detect the character select screen as
  `screen = "CHARACTER_SELECT"`
- The bridge can now expose structured `character_selection` state, including:
  - visible character options
  - selected character
  - whether `embark` is currently legal
- The bridge can execute `embark` through a legal action identifier instead of a
  UI click

Observed failure after `embark`:

- The first post-embark `GET /state` and `GET /actions` attempt on the map path
  hit a `NullReferenceException`
- The stack trace shows the fault in bridge code calling `creature.IsHittable`
  during non-combat state capture:
  - `BridgeGameApi.BuildCreaturePayload`
  - `Creature.get_IsHittable()`
  - `Hook.ShouldAllowHitting(CombatState combatState, Creature creature)`

Implication:

- Creature serialization must not assume combat-only properties are always safe
  to read during map or early-run transition states

## Combat Play Card Wiring In Stage

Observed locally on 2026-03-16 in the staged bridge source:

- `BridgeGameApi.BuildResolvedActions(...)` now adds `play_card:*` actions during
  combat before `end_turn`
- action generation walks each combat player's hand and only exposes cards that
  pass current playability checks
- single-target cards resolve into one legal action per currently valid target,
  keyed by target `combat_id`
- targetless, self-target, all-enemy, and similar non-choice card targets resolve
  into a single legal action
- card execution currently prefers direct model-level manual play calls:
  - `TryManualPlay(target)` first
  - `EnqueueManualPlay(target)` as fallback when available
- combat screen reporting is also normalized so tracker values like `Room` are
  translated back to `COMBAT` while a fight is in progress

Current status:

- The updated bridge code compiles successfully against the local
  `v0.99.1` game assemblies
- Live runtime validation is still required to confirm:
  - `play_card:*` actions appear in `/actions`
  - direct model execution mutates combat state as expected
  - no extra targeting edge cases need special handling in co-op or unusual cards

## Live Combat Card Play Verification

Observed locally on 2026-03-16 after deploying the combat play-card update:

- After entering the first monster room, `GET /actions` exposed legal
  `play_card:*` actions for enemy-target hand cards
- The bridge now reports `screen = "COMBAT"` in the room instead of the earlier
  tracker-derived `Room` label
- A live `POST /action` using `play_card:0:4:1` succeeded against the current
  combat state:
  - card: `痛击`
  - target: `缩小甲虫#1`
- Observed state change after the action:
  - player energy dropped from `3` to `1`
  - the enemy HP dropped from `39` to `31`
  - a debuff power with `amount = 2` was added to the enemy
  - the played `痛击` card moved into the discard pile

Follow-up finding from the same validation:

- Self-target hand cards such as `防御` were still not exposed as legal
  `play_card:*` actions in that live build
- The staged bridge source has now been adjusted to expose `TargetType.Self`
  cards whenever the acting creature is alive
- That self-target follow-up patch compiles locally, but it still needs a new
  deploy cycle because the currently running game process is still using the
  earlier DLL

## Live Self-Target Card Play Verification

Observed locally on 2026-03-16 after deploying the self-target follow-up patch:

- Combat `GET /actions` now exposes legal `play_card:*:self` actions for
  `TargetType.Self` hand cards such as `防御`
- A live `POST /action` using `play_card:0:1:self` succeeded against the current
  combat state
- Observed state change after the action:
  - player energy dropped from `3` to `2`
  - hand count dropped from `5` to `4`
  - player block increased from `0` to `5`
  - `state_changed` returned `true`

Implication:

- The bridge action model now covers both enemy-target and self-target card
  plays end-to-end using the same `play_card` surface

## Stale Event Option Exposure On Map

Observed locally on 2026-03-16 during the same validation pass:

- After leaving the Neow event and opening the map, the bridge still exposed a
  stale `event_option:0` action alongside legal `map:*` actions
- Executing that stale event action from the map returned `state_changed = false`
  and did not advance the run

Mitigation originally staged, now verified live:

- `BridgeGameApi` now suppresses event-option actions and the event-options state
  payload while `NMapScreen.IsOpen == true`
- That fix is now deployed in the live bridge lineage and no longer blocks map
  travel validation work

## AutoSlay MCP Victory Verification

Observed locally on 2026-03-16 after the bridge and MCP server were extended to
expose AutoSlay controls:

- the bridge now exposes structured automation state under
  `automation.autoslay`
- legal actions now include:
  - `automation:start_autoslay`
  - `automation:stop_autoslay`
- `packages/mcp-server/autoslay-runner.js` successfully launched AutoSlay
  through the MCP tool surface instead of direct HTTP
- the same runner monitored the run through Acts 1, 2, and 3
- the bridge surfaced multiple combat, reward, map, and boss transitions while
  the game was not being driven by desktop clicks
- the run reached the post-boss event, passed through `NGameOverScreen`, and
  the AutoSlay log recorded:
  - `Victory! Run completed and returned to main menu`
  - `Run completed successfully with seed=`
- after the successful return-to-menu path, the game process exited and the
  bridge session naturally terminated

Important follow-up finding from the same run:

- a transient `/state` null-reference occurred during non-combat polling when
  bridge code attempted to read `PlayerCombatState.MaxEnergy`
- that bug has now been fixed in source and deployed as
  `bridge_version = 0.3.1`

## Act 1 Boss Post-Reward False MAP Terminal On 0.7.7

Observed locally on 2026-03-17 during the live Ironclad boss run:

- the live bridge reported:
  - `bridge_version = 0.7.7`
  - `schema_version = 2026-03-17.1`
- the boss kill and reward flow itself succeeded:
  - lethal landed on `仪式兽`
  - rewards included:
    - `100 gold`
    - potion `欧洛巴斯之酸`
    - card reward choices `焚烧` / `绯红披风` / `烙印`
  - the run claimed:
    - `100 gold`
    - `烙印`
    - potion `欧洛巴斯之酸`
- after reward resolution, the run state was:
  - HP `34/91`
  - gold `365`
  - potion slot `0 = 欧洛巴斯之酸`

Critical breakpoint:

- executing terminal reward `proceed` did **not** advance to a playable Act 2
  state
- instead the bridge returned:
  - `screen = "MAP"`
  - `act = "密林"`
  - `act_floor = 17`
  - `current_room.room_type = "Boss"`
  - `current_room.is_pre_finished = true`
  - `current_map_coord = (3, 16)`
  - `map.is_open = true`
  - `map.is_travel_enabled = true`
- however that state exposed no actual forward progress action:
  - no legal `map:*` actions
  - no `map.points[*]` with `is_travelable = true`
  - current boss node `(3, 16)` reported `children = []`
  - the only non-automation action left was `discard_potion:0:0`
- user confirmed the visible UI still showed a red back button on the left; that
  button returns to the reward/room layer, and pressing `proceed` there again
  reproduces the same false-terminal map state

Implication:

- the existing heuristic
  `screen = "MAP" && map.is_travel_enabled == true`
  is valid for ordinary room cleanup but **invalid** for post-boss act
  transition
- reward auto-proceed and any room-end MCP batching logic must treat this as a
  special case instead of a completed transition
- likely bridge gap:
  - the visible next-act transition UI is not currently modeled as either
    `map:*` actions or a dedicated act-transition action
  - `IsMapPointTravelable(...)` and the map-action builder therefore never
    expose the continuation step

Most relevant code surfaces to inspect next:

- `BuildResolvedActions(...)` map travel block in `BridgeGameApi.cs`
- `IsMapPointTravelable(...)` in `BridgeGameApi.cs`
- terminal reward settle logic in
  `MaybeAutoProceedAfterRewardActionAsync(...)`

Follow-up source work staged locally on 2026-03-17:

- `BridgeGameApi.InvokeTerminalRewardsProceed(...)` now prefers the visible
  reward proceed button / `NRewardsScreen.OnProceedButtonPressed()` before
  falling back to `RunManager.ProceedFromTerminalRewardsScreen()`
- rationale:
  - the user-visible successful path is a real button press
  - the previous bridge ordering bypassed that path and may have been taking a
    room-generic shortcut instead of the correct boss transition flow
- local build status:
  - `dotnet build mods/sts2-bridge/sts2-bridge.csproj -p:Sts2SkipDeploy=true`
    succeeded
  - deploy into the live game directory was intentionally skipped because the
    loaded mod DLL was locked by `SlayTheSpire2.exe`

Second live regression after deploying that patch:

- the previous bad behavior changed:
  - terminal `proceed` no longer jumped into the stale Act 1 map shell
- new observed behavior:
  - `POST /action` for `proceed` returned `ok = true`
  - but `state_changed = false`
  - bridge remained on:
    - `screen = "REWARDS"`
    - `rewards_visible = true`
    - legal actions:
      - `discard_potion:0:0`
      - `proceed`
- however a simultaneous desktop snapshot of the real game UI showed:
  - the run was already back in the boss room scene
  - a visible `前进` button was present on screen
- implication:
  - the remaining blocker is likely stale reward-screen visibility / screen
    classification after terminal rewards
  - the bridge is probably prioritizing `REWARDS` over the real room/proceed
    state for too long

Most likely code surfaces after this regression:

- `IsRewardsScreenVisible(...)` in `BridgeGameApi.cs`
- `ResolveCurrentScreen(...)` in `BridgeGameApi.cs`
- any place that treats a visible reward proceed button plus zero reward buttons
  as an active reward screen instead of a completed room returning to `Room`

Event glossary payload update staged on 2026-03-17:

- user feedback: event payload that only returned
  `event=[0:营养汤; 1:烘焙手套; 2:南瓜蜡烛]` was insufficient because the left-side
  glossary/tooltip panel carried the actual semantic explanation for special
  terms such as:
  - `特兹卡塔拉的余烬 -> 费用为0且获得永恒。`
  - `永恒 -> 无法从你的牌组中移除或变化。`
- bridge source change:
  - `BuildEventOptionsPayload(...)` now augments `event_options` with:
    - `visible_glossary_source`
    - `visible_glossary_texts[]`
    - `visible_glossary[]`
  - primary source is the visible global hover-tip set
  - fallback source is extra visible text under the event room after removing
    option-button text
- compact MCP summary change:
  - `packages/mcp-server/index.js` now preserves `event_options.visible_glossary`
    in compact responses
- next live verification:
  - confirm the current Tezcatara event returns the two expected glossary
    entries while remaining compact enough for routine agent turns

Follow-up live validation on 2026-03-17:

- source change extended beyond visible UI fallback:
  - `BuildEventOptionPayload(...)` now reads `EventOption.HoverTips`
    directly from the game model
  - this is the decisive fix because the option hover-tip list exists even when
    the user has not currently hovered a term on screen
- validated raw bridge state on the Act 2 Tezcatara event now returns:
  - option `营养汤`
    - `特兹卡塔拉的余烬 -> 费用为0且获得永恒。`
    - `永恒 -> 无法从你的牌组中移除或变化。`
  - option `烘焙手套`
    - `消耗 -> 在战斗结束前移除。`
    - `力量 -> 力量会增加攻击牌造成的伤害。`
  - option `南瓜蜡烛`
    - `能量 -> 能量用于打出你手牌中的卡牌。`
- payload shape:
  - `event_options.options[*].glossary[]`
    - `id`
    - `type`
    - `title`
    - `description`
    - `texts[]`
    - hover-tip flags
- note:
  - the older `visible_glossary*` fallback still mostly captures room text when
    no tooltip panel is visibly open
  - the new option-local `glossary[]` is the field the outer agent should trust
    for semantic reasoning

MCP compact-text cleanup staged on 2026-03-17:

- summary-layer text normalization is now being applied in
  `packages/mcp-server/index.js`
- intent:
  - strip UI-only BBCode color/style markup such as `[gold]...[/gold]`
  - replace obvious icon tags such as energy `[img]...energy_icon...[/img]`
    with plain-language text
  - collapse unresolved placeholder noise in compact replies so the agent sees
    human-readable strings rather than rendering syntax
- scope:
  - action labels
  - event option titles / descriptions / glossary
  - reward, rest-site, potion, power, creature-intent, shop, and card-summary
    descriptions where surfaced by the compact MCP response
- this was the first cleanup pass only:
  - raw bridge payload remained unchanged at that stage for debugging
  - a later bridge-side follow-up was still required to make raw `/state`
    readable without MCP-side normalization

Raw bridge placeholder-resolution update validated on 2026-03-17:

- user explicitly requested that placeholder substitution move below the MCP
  summary layer so even raw `/state` payloads become readable
- initial generic reflection against direct model members only partially worked:
  - outer event-option `description` values were already resolved by the game
  - nested `relic.description` still leaked `{StrengthPower}` and
    `{Energy:energyIcons()}`
- direct assembly reflection on `sts2.dll` exposed the missing clue:
  - `RelicModel`, `PotionModel`, and `PowerModel` all expose
    `CanonicalVars`
  - `RelicModel` / `PotionModel` also expose dynamic description members such
    as `DynamicDescription`
  - `PowerModel` exposes `SmartDescription`
- bridge follow-up:
  - `TryGetDescription(...)` now prefers dynamic description members before raw
    `Description`
  - placeholder fallback now also searches `CanonicalVars` and `DynamicVars`
    when a token is not present as a normal property/field
- live verification after redeploy on the Tezcatara event:
  - `烘焙手套` nested relic payload now returns:
    `在你的回合开始时，[gold]消耗[/gold]你[gold]抽牌堆[/gold]顶部的牌并获得[blue]1[/blue]点[gold]力量[/gold]。`
  - `南瓜蜡烛` nested relic payload now returns:
    `在每个回合开始时获得[img]res://images/packed/sprite_fonts/ironclad_energy_icon.png[/img]。这件遗物将在第[blue]3[/blue][gold]阶段[/gold]开始时熄灭。`
  - unresolved raw tokens are no longer present in those verified fields

Raw bridge text-normalization follow-up validated on 2026-03-17:

- user then asked for one more step down the stack:
  - not just placeholder substitution
  - also strip raw payload render markup so `/state` descriptions can be read
    directly without MCP cleanup
- bridge follow-up:
  - `DescribeText(...)` now normalizes payload-facing text after placeholder
    substitution
  - BBCode-style color tags such as `[gold]...[/gold]` and `[blue]...[/blue]`
    are stripped from raw payload text
  - energy image tags such as
    `[img]...ironclad_energy_icon.png[/img]` are converted to plain
    `1点能量`
  - 2026-03-18 follow-up refined image-tag handling to use the game's actual
    sprite-font resource names instead of a blanket fallback:
    - exact `star_icon` now resolves to `1点星辉`
    - exact `*_energy_icon` variants still resolve to `1点能量`
    - repeated tags and compact forms like `3[img]...energy_icon...[/img]`
      collapse into counted text such as `3点能量`
    - unknown image tags are no longer silently collapsed to `图标`; they keep a
      diagnostic suffix such as `图标:some_icon_name`
  - 2026-03-18 placeholder-source follow-up:
    - decompilation confirmed `LocString` carries a private `_variables`
      dictionary exposed via `Variables`, while event option descriptions do
      not automatically inherit the owning `EventModel.DynamicVars`
    - `EventOption` stores only its own `Title` / `Description` `LocString`
      values; `NEventRoom` separately keeps the active event model in `_event`
    - bridge fix:
      - event option payload building now pulls `NEventRoom._event` and formats
        the option text against both the option and the active event model
      - placeholder lookup now also checks `LocString.Variables`
      - `DescribeText(...)` injects the text object itself into the internal
        placeholder context, so `Amount`-style tokens can be resolved without
        leaking raw model context into the outward payload
    - live validation after redeploy on `EVENT.DENSE_VEGETATION`:
      - `坚持跋涉 -> 从你的牌组中移除一张牌。失去11点生命。`
      - `休息 -> 回复24点生命。进入战斗。`
    - log sweep after the live re-test found no fresh
      `Localization formatting error` lines in the current `godot.log` tail
  - node-sourced glossary / prompt text now also flows through the same
    normalization path, so visible glossary capture and raw model descriptions
    are consistent
- live verification after redeploy on the current Act 2 reward checkpoint:
  - `南瓜蜡烛` raw relic description now returns:
    `在每个回合开始时获得1点能量。这件遗物将在第3阶段开始时熄灭。`
  - reward, potion, relic, and card descriptions on that screen all surfaced
    as plain readable text in raw `/state`

Action-settlement hardening validated on 2026-03-17:

- reproduced issue:
  - `main_menu:continue` could return a half-transition payload with
    `screen = "Room"` and only `discard_potion:*` available even though the
    actual reward screen appeared shortly afterward
- bridge follow-up:
  - likely transition actions now keep polling after the initial wait barrier
    until a real interactable surface becomes visible
  - current accepted stable surfaces:
    - reward / card reward
    - card selection
    - rest site / deck upgrade
    - event / shop / map
    - combat with visible non-automation actions
    - a lone `proceed` action
- MCP follow-up:
  - `main_menu:continue` no longer treats any non-main-menu screen as settled
  - the previous generic out-of-combat `screen:Room` fallback was removed so
    half-transition room frames no longer count as stable
- live re-test after redeploy:
  - the same `main_menu:continue` request now returned directly to
    `screen = "REWARDS"` with `reward:0..3` and `proceed` already present

Combat target-suffix readability update validated on 2026-03-17:

- reproduced issue in the current Act 2 `胧光怪` checkpoint:
  - action ids such as `play_card:0:0:2` exposed the internal target suffix
    but did not make it obvious which creature `:2` mapped to
  - this caused a live mis-target and showed that the fight was a good
    save-reload regression sample for suffix readability
- bridge follow-up:
  - `play_card` / `use_potion` payloads now also expose:
    - `target_action_suffix`
    - `target_combat_id`
    - `target_name`
    - `target_side`
    - `target_mapping`
  - action labels now embed the mapping directly, for example:
    `Play card 0: 啄击 -> 1 = 胧光怪 (combat_id 1)`
  - combat state now exposes `combat.target_index_map[]` so the current
    suffix-to-creature map is available at state level and not only inside each
    action
- MCP compact summary follow-up:
  - summarized actions now preserve target metadata instead of collapsing down
    to only `action_id / kind / label`
  - compact combat summaries now preserve `target_index_map`
- live save-reload verification after redeploy:
  - reloading the same fight now surfaces:
    - `target_index_map = [self/0 -> 铁甲战士, 1 -> 胧光怪]`
    - action payload fields
      `target_action_suffix = "1"`,
      `target_combat_id = 1`,
      `target_name = "胧光怪"`

## Questions To Answer In Phase 0

- Which game classes expose battle, map, reward, and choice state?
- Can those classes be read without Harmony patches?
- Which mutating actions can be executed via public APIs instead of UI clicks?
