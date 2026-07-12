# Implementation Plan

## Outcome

Deliver a local-only, visible-state-only MCP integration for Slay the Spire 2
that can:

- read the current game state
- enumerate legal actions
- execute a small set of safe player actions

## Delivery Strategy

Build the system in vertical slices. Do not start with card databases, full UI
coverage, or hidden-state extraction. The first success criterion is a live
read-only bridge; the second is a single safe battle action end-to-end.

Operational constraint:

- Do not assume `Slay the Spire 2` is fullscreen, on the primary monitor, or
  even in the foreground when the agent accesses it
- Therefore, gameplay state and legal actions must come from the in-process
  bridge, not from screenshots or coordinate clicks
- Desktop automation remains an explicit fallback for bootstrap flows that are
  not yet represented in the bridge contract

## Phase 0: Discovery And Ground Truth

### Goals

- Confirm the current mod loading mechanism on the installed build
- Identify the smallest stable entrypoint for a bridge mod
- Map the classes or scene nodes required for battle and map state

### Tasks

- Verify where community mods are expected to be installed
- Verify the current tutorial workflow against the installed build
- Decide whether to start from `ModTemplate-StS2` or a hand-written minimal
  project
- Install or inspect one minimal community mod as a reference
- Inspect `sts2.dll` with ILSpy or dnSpy and record key namespaces/classes
- Inspect local logs for mod-related behavior
- Confirm whether the game exposes a public API suitable for mods
- Record the exact deployment contract: `<modid>.json`, `dll`, `pck`, and target
  output folder
- Record the debug contract: `launch_*.bat --log`, `steam_appid.txt`, and `~`
  console

### Deliverables

- documented mod loading path
- documented native mod package shape
- documented candidate hook points
- short compatibility notes in `docs/research-notes.md`

### Exit Criteria

- We know how our bridge code enters the game process
- We know at least one reliable way to read current scene and battle state

## Phase 1: Read-Only Bridge Prototype

### Goals

- Load a bridge mod into the game
- Expose local `GET /health` and `GET /state`
- Return scene type and minimal visible state
- Establish the canonical build and export loop for this repository

### Tasks

- Scaffold `mods/sts2-bridge` as a native Godot C# mod project
- Add `<modid>.json`
- Configure `.csproj` references to `sts2.dll` and `0Harmony.dll`
- Configure build output to copy the `dll` into the game's `mods` directory
- Add a startup banner and version logging
- Add `Scripts/Entry.cs` with `[ModInitializer("Init")]`
- Verify whether any explicit script-registration call is required on the live
  build; do not assume `ScriptManagerBridge.LookupScriptsInAssembly(...)`
  resolves unless it compiles locally
- Generate a per-session token and bind a loopback HTTP listener
- Implement normalized scene detection
- Detect at least the pre-run surfaces needed to bootstrap a run, including main
  menu and character selection, instead of collapsing them into `UNKNOWN`
- Implement minimal state serialization for combat and map
- Export a first matching `pck`

### Deliverables

- bridge mod that starts with the game
- reproducible `dll + pck` packaging flow
- local endpoint returning JSON state
- versioned state envelope

### Exit Criteria

- Manual request to `GET /state` works while the game is running
- Response includes `scene`, `state_version`, and one scene-specific payload
- The game reports the bridge mod as loaded under the current local mod workflow

## Phase 2: MCP Read-Only Surface

### Goals

- Wrap the bridge with a real MCP server
- Expose read-only tools to Codex

### Tasks

- Scaffold `packages/mcp-server`
- Add bridge client and schema validation
- Use newline-delimited JSON-RPC for stdio transport compatibility with Codex
- Implement `sts2_get_bridge_status`
- Implement `sts2_get_state`
- Implement `sts2_list_actions` using bridge-provided legal actions
- Add smoke tests with mocked bridge responses
- Add bootstrap logging that distinguishes spawn failures from handshake failures

### Deliverables

- runnable MCP server over stdio
- typed bridge client
- basic read-only tool tests
- startup diagnostics in `%APPDATA%\\SlayTheSpire2\\bridge\\mcp-stdio.log`

### Exit Criteria

- Codex can query current game state without desktop automation
- Codex can still query bridge state when the game is not the foreground window
- Tool outputs are stable and human-readable

### Current Progress Note

Observed on 2026-03-16:

- the MCP server now exposes:
  - `sts2_get_bridge_status`
  - `sts2_get_state`
  - `sts2_list_actions`
  - `sts2_get_map_routes`
  - `sts2_get_deck`
  - `sts2_perform_action`
  - `sts2_end_turn`
  - `sts2_resolve_room_rewards`
  - `sts2_resolve_rest_site`
  - `sts2_resolve_card_selection`
  - `sts2_pick_option`
  - `sts2_travel_to_coordinate`
  - `sts2_play_card_sequence`
  - `sts2_resolve_shop_visit`
  - `sts2_wait_for_change`
- the current stdio MCP transport is stable under Codex with newline-delimited
  JSON-RPC
- the MCP server source has now advanced to:
  - `SERVER_VERSION = 0.4.15`
- indexed non-combat actions now have an MCP-side abstraction:
  - `sts2_pick_option` resolves the current visible indexed surface instead of
    forcing the outer agent to guess between raw ids like:
    - `reward:{index}`
    - `card_reward:{index}`
    - `rest_site:{index}`
    - `deck_upgrade:{index}`
    - `event_option:{index}`
    - `card_selection:select:{index}`
  - live validation on 2026-03-18 confirmed:
    - `sts2_pick_option(surface = "reward", index = 1)` settled directly onto
      `CARD_REWARD_SELECTION`
    - `sts2_pick_option(surface = "card_reward", index = 1)` settled back onto
      the post-pick reward page instead of returning an in-between frame
- map travel now has a first-class MCP abstraction:
  - `sts2_travel_to_coordinate`
  - it blocks on unresolved reward / rest-site / card-selection decisions
    instead of guessing
  - it auto-absorbs cleanup-only `proceed` tails when no decision remains
  - it waits for a stable map snapshot before firing `map:{col},{row}`
  - live validation on 2026-03-18 confirmed:
    - unresolved rewards correctly returned
      `reason = "reward_resolution_required"`
    - after resolving the reward flow, `sts2_travel_to_coordinate(2, 4)`
      entered the next hallway combat and settled onto a stable player-turn
      combat snapshot
- `sts2_resolve_room_rewards` now supports the intended room-end batching flow:
  - inspect fixed room rewards in one call
  - claim deterministic rewards such as gold and relics
  - choose a card reward by `pick_card_index`
  - auto-advance reward cleanup until the run is back on the next real
    decision surface
  - for ordinary rooms this is usually `screen = "MAP"`, but boss terminal
    rewards still need a dedicated act-transition check before `MAP` can be
    treated as final
- `sts2_resolve_rest_site` now provides the same one-call batching pattern for
  campfires:
  - choose `rest_site:{index}`
  - if the choice opens `DECK_UPGRADE_SELECTION`, require and consume
    `upgrade_card_index`
  - auto-advance `rest_site:proceed` until the run is back on
    `screen = "MAP"` when possible
  - treat `screen = "MAP"` plus `map.is_travel_enabled = true` as terminal even
    if the currently deployed bridge still leaks a stale `rest_site:proceed`
    action
- `sts2_end_turn` now performs post-action settlement polling instead of
  returning the first half-finished combat frame:
  - it waits for either a non-combat screen or a stable player turn with a
    fully populated hand snapshot
  - it returns extra telemetry:
    - `post_action_settled`
    - `post_action_settle_reason`
    - `post_action_settle_polls`
- live validation against `ENCOUNTER.SHRINKER_BEETLE_WEAK` confirmed that the
  patched `sts2_end_turn` now returns the next playable turn rather than an
  enemy-resolution frame or partial draw frame
- `sts2_perform_action` and internal batching helpers now share the same
  post-action settlement model for animation-heavy combat flows:
  - enabled action families currently include:
    - `play_card:*`
    - `card_selection:*`
    - future-compatible `use_potion:*`
  - the MCP layer now waits for either:
    - a visible reward / card-reward flow
    - a visible rest-site / deck-upgrade flow
    - a map-ready state
    - a visible event / shop / other actionable non-combat screen
    - a visible card-selection flow
    - or a stable player-turn combat snapshot
  - this is intended to avoid returning mid-draw or mid-resolution states when
    cards are still animating into hand after an action resolves
  - this specifically fixes the observed live breakpoint where a combat-ending
    action could return a half-transition frame before the reward decision was
    actually surfaced to the agent
  - the bridge layer now also adds a main-thread pump barrier before taking the
    first post-action snapshot:
    - after executing an action, it waits for the requested `wait_after_ms`
    - then it waits an additional fixed number of bridge pump ticks before
      `CaptureSnapshot()`
    - the same barrier is reused by reward auto-proceed and card-selection
      auto-confirm flows
  - this is intended to reduce the older class of bugs where the scene tree had
    not fully switched frames yet even though the coarse action delay had
    already expired
- MCP output compaction was tightened again on 2026-03-17:
  - list-actions output now strips `automation:*` noise from the compact tool
    view
  - compact card summaries now prefer:
    - `title`
    - `cost`
    - short `effect`
    - and fall back to full `description` only when no effect summary exists
  - reward potion slots now omit null potion payloads for empty slots
  - top-level compact state output now omits `state_hash` /
    `captured_at_utc`
  - live measurement on the boss reward checkpoint:
    - raw bridge `/state` payload: about `40165` characters
    - compact `sts2_get_state` MCP result: about `1591` characters
  - compact `sts2_list_actions` MCP result: about `368` characters
  - compact hand / reward card summaries were tightened again later on
    2026-03-17 after live combat exposed missing tactical semantics:
    - previous behavior preferred short `effect_preview.summary` too strongly
    - this hid real card rules on cards such as:
      - `燃烧契约`
      - `愤怒`
      - `痛击`
      - `放血`
    - MCP compact summaries now also keep `description` when the full rules text
      is materially richer than the short effect string
    - live validation in combat now surfaces examples such as:
      - `燃烧契约 = draw 2` plus `消耗1张牌。抽2张牌。`
      - `痛击 = 8 damage` plus `造成8点伤害。给予2层易伤。`
      - `愤怒 = 6 damage` plus `造成6点伤害。将一张此牌的复制品加入你的弃牌堆。`
      - `放血 = lose 3 HP` plus `失去3点生命。获得1点能量...`
    - this keeps the compact payload small while making the outer agent less
      dependent on out-of-band card memory
  - player buff visibility was corrected later on 2026-03-17:
    - raw bridge state already exposed `players[0].creature.powers`
    - but compact `sts2_get_state` had been dropping those player powers
    - this made potion / buff effects hard to reason about during live play,
      even though enemy powers were already visible
    - MCP compact player state now also returns `player.powers[]`
    - live regression on the current Act 1 elite confirmed:
      - after using `铁心药水`, compact state now shows:
        - `player.powers = [{ title: \"覆甲\", amount: 2 }]`
- map-route planning support started on 2026-03-17:
  - previous agent routing was still mostly local:
    - inspect current travelable frontier
    - then reason a few rows ahead by hand
  - this is weaker than real player route planning, which usually evaluates the
    whole reachable route DAG from the current position
  - MCP now begins exposing a dedicated `sts2_get_map_routes` tool so the outer
    agent can inspect a pruned future-only route forest instead of only the
    next few nodes
  - normalized output shape:
    - exclude the current node
    - exclude rows at or before the current row
    - exclude nodes that are no longer reachable from the current frontier
    - return one root entry per currently travelable next step
    - emit one deduplicated node table for the whole reachable future graph
    - each route node keeps only:
      - `key` (`col,row`)
      - `point_type`
      - `child_keys`
  - this is meant to make route choice explicit and inspectable rather than a
    hidden heuristic inside the model
  - live validation on the first map after Neow:
    - reachable future node count: `59`
    - immediate route roots: `3`
    - MCP payload size dropped from about `35k` characters with nested tree
      duplication to about `7.4k` characters with the normalized forest
  - follow-up live validation after actually taking one route and clearing one
    room:
    - current coord moved to `3,1`
    - immediate route roots correctly shrank to only `2,2`
    - current node `3,1` was not present in either `route_roots` or
      `route_nodes`
    - no row `1` or earlier nodes remained in the payload
    - previously visible sibling branches from the original opening frontier no
      longer appeared
    - reduced payload size measured about `5479` characters, roughly `1.37k`
      tokens
  - map-route MCP output was tightened again on 2026-03-17:
    - `sts2_get_map_routes` now defaults to `detail = "summary"`
    - `detail = "full"` remains available when the outer agent really needs the
      deduplicated future node table
    - each `route_root` now includes a compact planning summary:
      - `forced_path_steps_before_branch`
      - `reachable_point_type_counts`
      - `steps_from_current_to_next_point_type`
      - `can_reach_elite_then_rest_site`
    - the tool now waits for a stable map snapshot instead of trusting the
      first visible frame:
      - require `screen = "MAP"`
      - require `map.is_open = true`
      - require `map.is_traveling != true`
      - cross-check current travelable frontier against legal `map:*` actions
      - require stable repeated snapshot fingerprints before returning
    - live validation at the current `3,1` checkpoint:
      - `detail = "summary"` payload: about `1336` characters
      - `detail = "full"` payload: about `6268` characters
      - stable-settle telemetry reported:
        - `snapshot_settled = true`
      - `snapshot_settle_reason = "map_snapshot_stable"`
      - `snapshot_settle_polls = 2`
      - `frontier_action_match = true`
  - run-aware route tags were added on 2026-03-17:
    - top-level map payload now includes `run_context`:
      - hp current/max/ratio/band
      - gold plus `gold_band`
      - potion slot total/filled/empty/band
      - deck size
      - relic counts
    - each route root now also includes `run_aware`:
      - `rest_pressure`
      - `shop_access_value`
      - `potion_capacity_value`
      - `elite_plan`
        - next elite distance
        - rest-before-elite / rest-after-elite booleans
        - heuristic viability rating plus score
      - compact `reason_tags`
    - current live measurement at coord `3,1` after this addition:
      - summary payload about `2350` characters
      - full payload about `7282` characters
- combat sequencing support was added on 2026-03-17:
  - new MCP tool: `sts2_play_card_sequence`
  - input contract:
    - one call accepts an ordered list of current `play_card:*` action ids
    - the caller does not need to rewrite later hand indices after each card
      resolves
  - matching model:
    - capture the initially requested `play_card` actions from the current
      action list
    - store a compact card fingerprint based on card id/title/cost/effect
      preview plus target suffix / target combat id
    - after each resolved card, re-read the current playable card actions and
      rematch the next requested step against the current hand
    - this avoids stale hand-index assumptions when hand order shifts or new
      cards are drawn into the hand
  - interruption model:
    - stop early and return partial progress when combat leaves the playable
      player-turn state
    - especially when reward flow or card-selection flow appears
  - interaction hints now also recommend `sts2_play_card_sequence` when two or
    more playable card actions are currently visible
  - first live validation now completed:
    - round-2 combat test against `ENCOUNTER.NIBBITS_WEAK`
    - requested sequence:
      - `play_card:0:1:1`
      - `play_card:0:4:1`
    - observed execution:
      - first step executed exact as requested
      - second step rematched from requested `play_card:0:4:1` to live
        `play_card:0:3:1`
      - MCP reported:
        - `match_type = "reindexed"`
        - `compatible_candidate_count = 1`
    - follow-up live validation in the same fight:
      - requested three strikes:
        - `play_card:0:1:1`
        - `play_card:0:3:1`
        - `play_card:0:4:1`
      - final strike rematched to `play_card:0:2:1`
      - final settle reason was `reward_flow_ready`
      - resulting state landed directly on `screen = "REWARDS"`
  - live failure found later on 2026-03-17:
    - sequence request:
      - `痛击` on the target
      - then a `打击` on the same target
    - after `痛击` resolved, the target gained `易伤`
    - that changed the later `打击` preview from `6 damage` to `9 damage`
    - the original MCP fingerprint had incorrectly included dynamic
      `effect_preview` values in its hard match criteria
    - result:
      - the later strike was still present in hand
      - but `sts2_play_card_sequence` returned
        `requested_play_card_action_unavailable_after_reindex`
        instead of rematching it
  - MCP-side fix now implemented locally:
    - play-card hard matching no longer fingerprints:
      - target-state-dependent `effect_preview` values
      - or other preview text that can drift because of
        `易伤` / `力量` / similar transient combat state
    - hard matching now uses a more stable card identity signature based on:
      - card id / title / type / rarity / target type
      - canonical cost fields
      - normalized description shape
      - dynamic-var base / enchanted signatures
    - practical goal:
      - preserve rematching across hand reindex even when earlier steps change
        the current numeric preview of later cards
  - status:
    - local `node --check packages/mcp-server/index.js` passes
    - live post-patch combat regression now completed on 2026-03-17:
      - current fight:
        - enemy `蛇行扼杀者` at `18 HP`
        - planned sequence:
          - `痛击`
          - then `打击`
          - then another `打击`
      - observed result:
        - after `痛击`, the next `打击` rematched and executed successfully
        - MCP reported `match_type = "reindexed_ambiguous"` rather than
          rejecting the step as unavailable
        - this confirms the old fingerprint bug was actually fixed in live play
      - important nuance:
        - the third requested `打击` still did not execute
        - root cause was not fingerprint drift:
          - `痛击(2费) + 打击(1费)` had already spent all `3` energy
          - the enemy survived on `1 HP`
          - therefore the last strike was no longer a legal action
      - follow-up run state after the regression:
        - next turn killed the enemy normally
        - `sts2_resolve_room_rewards` then successfully:
          - claimed `19 gold`
          - selected `愤怒`
          - auto-returned to `screen = "MAP"`
- combat sequencing follow-up was extended locally on 2026-03-18:
  - new MCP tool: `sts2_execute_combat_sequence`
  - current supported action kinds inside one sequence:
    - `play_card:*`
    - `use_potion:*`
    - `end_turn`
  - normalization rule:
    - if `end_turn` is present anywhere in the request, MCP always moves it to
      the final step before execution
    - this is intended to harden against agent-side batching or tool-order drift
      that previously caused `end_turn` to resolve before the rest of the turn
  - observability follow-up:
    - `sts2_perform_action`
    - `sts2_play_card_sequence`
    - `sts2_execute_combat_sequence`
    - `sts2_end_turn`
    - now all accept `return_state_after = true`
    - the compact normal `state` field remains summarized for token control
    - a separate `state_after` field now carries the full raw post-action bridge
      state when explicitly requested
  - compact combat summary follow-up:
    - MCP now surfaces allied summons under `combat.summons[]` by deriving them
      from `combat.player_creatures[]` minus the primary player creature
  - guidance follow-up:
    - MCP interaction hints now explicitly mark parallel combat
      `sts2_perform_action` usage as something to avoid
- shop visit batching and reindex handling were implemented locally on
  2026-03-17:
  - new MCP tool: `sts2_resolve_shop_visit`
  - intended one-call flow:
    - optionally `shop:open`
    - buy one or more requested shop items
    - if card removal was bought, optionally select and confirm the removed card
    - optionally `shop:back`
    - optionally `shop:leave`
  - root cause reproduced live before the fix:
    - initial open-shop actions exposed:
      - `shop:buy:1` = `铁斩波`
      - `shop:buy:10` = `Remove a card`
    - after buying `铁斩波`, the shop reindexed:
      - `shop:buy:9` = `Remove a card`
      - `shop:buy:10` = `能力药水`
    - therefore any batching logic that reused the original cached
      `shop:buy:10` would buy the wrong item
  - MCP mitigation now implemented:
    - capture the initially matched shop action from the current visible state
    - store a compact item fingerprint based on:
      - `item_kind`
      - `title`
      - `description`
      - `cost`
      - nested `card` / `relic` / `potion` identifiers when present
    - after each purchase, re-read the current `shop:buy:*` action list and
      rematch the next planned step against the live shop contents
    - support resuming a pending shop-driven card-removal selection using either
      `remove_card_title` or `remove_card_index`
  - compact shop state summarization was corrected at the same time:
    - previous MCP compaction incorrectly looked for:
      - `price`
      - `item_type`
      - `can_buy`
    - the bridge actually emits:
      - `cost`
      - `item_kind`
      - `is_affordable`
  - status:
    - local `node --check packages/mcp-server/index.js` passes
    - live reload verification is still pending after the next save reload
- reward-resolution caveat now explicitly recorded:
  - bridge reward action ids are positional (`reward:{index}`)
  - after claiming one reward, the remaining reward buttons can be reindexed
  - the outer agent must not cache reward indices across multiple claims
  - room-end batching must always re-read the current reward surface between
    steps
  - MCP-side reward auto-proceed now also ignores post-claim
    `discard_potion:*` cleanup noise when the actual reward list is empty and
    only `proceed` remains as the real next step
  - live reward batching still correctly handles repeated reindex collapse:
    - on the current floor-7 reward test, gold -> potion -> card reward all
      reindexed back down to `reward:0`
    - `sts2_resolve_room_rewards` still claimed:
      - `16 gold`
      - `铁心药水`
      - `欺凌`
      - then auto-returned to `screen = "MAP"`
    - this confirms the reward resolver is now robust against full left-collapse
      of the visible reward list, not only single-step index changes
- a new live smoke test against the current floor-6 map checkpoint confirmed
  that `sts2_resolve_rest_site` correctly identifies the room as already
  resolved on the MCP side:
  - it returns `reason = "not_in_rest_site_flow"`
  - it suppresses the stale bridge exposure by recognizing that
    `screen = "MAP"` and `map.is_travel_enabled = true`
- the bridge source has now advanced again to:
  - `bridge_version = 0.7.7`
  - `schema_version = 2026-03-17.1`
- treasure-room bridge coverage was expanded on 2026-03-17 after a live
  blocker on floor 26:
  - the generic room-level `proceed` action was being exposed inside
    `room_type = Treasure`, but executing it did not change state
  - root cause:
    - Treasure rooms require room-specific chest / relic handlers rather than
      the generic proceed-button lifecycle
  - bridge fix now implemented in `BridgeGameApi.cs`:
    - capture `NTreasureRoom`, its `_chestButton`, and the
      `NTreasureRoomRelicCollection`
    - expose `treasure:open` when the chest is still unopened
    - expose `treasure_relic:{index}` for visible treasure relic holders
    - suppress the misleading generic `proceed` while chest-open or relic-pick
      actions are still available
    - after relic claim, stop re-exposing stale `treasure:open` by checking the
      room's `_hasRelicBeenClaimed` and `_isRelicCollectionOpen` state
    - allow the normal `proceed` / map flow to return only after treasure-room
      interaction is actually complete
  - live regression validated:
    - `main_menu:continue` -> `treasure:open`
    - `treasure:open` -> `treasure_relic:0` with relic title/description
    - `treasure_relic:0` -> `proceed`
    - `proceed` -> `screen = "MAP"` with next legal action `map:3,9`
- the live bridge deploy now confirms the campfire proceed fix end-to-end:
  - fresh reload returns to floor-8 `screen = "REST_SITE"`
  - the room no longer exposes `rest_site:proceed` while `HEAL` / `SMITH`
    options are still visible
  - after `deck_upgrade:confirm`, the bridge exposes only `rest_site:proceed`
  - executing `rest_site:proceed` now transitions to:
    - `screen = "MAP"`
    - `map.is_open = true`
    - `map.is_travel_enabled = true`
    - next legal action `map:3,8`
- the bridge now exposes a unified `card_selection` surface for both
  intermediate overlay screens and combat-local hand-selection modes:
  - visible screen types currently captured:
    - `NSimpleCardSelectScreen`
    - `NChooseACardSelectionScreen`
    - `NDeckCardSelectScreen`
    - `NPlayerHand` when `IsInCardSelection == true`
  - normalized payload:
    - `card_selection.visible`
    - `card_selection.screen_type`
    - `card_selection.prompt`
    - `card_selection.options`
    - `card_selection.selected_count`
    - `card_selection.min_select`
    - `card_selection.max_select`
    - `card_selection.requires_manual_confirmation`
    - `card_selection.cancelable`
    - confirm / cancel / close / skip visibility
  - normalized actions:
    - `card_selection:select:{index}`
    - `card_selection:confirm`
    - `card_selection:cancel`
    - `card_selection:close`
    - `card_selection:skip`
  - while that overlay is visible, ordinary combat `play_card:*` and

- the bridge now exposes potion actions directly from the legal action surface:
  - combat use actions:
    - `use_potion:{player_index}:{slot_index}:{target}`
    - currently validated for:
      - self-target / ally-target potions
      - enemy-target potions
  - discard actions when the game currently allows potion removal:
    - `discard_potion:{player_index}:{slot_index}`
  - potion payloads now include:
    - `selection_screen_prompt`
    - `can_throw_at_ally`
    - `is_usable`
    - `is_queued`
- live validation on 2026-03-17 confirmed:
  - after restart, `main_menu:continue` successfully restored the current floor-9
    combat
  - the restored combat exposed:
    - `use_potion:0:0:0` for a self/ally-target `POTION.FLEX_POTION`
    - `use_potion:0:1:1` and `use_potion:0:1:2` for enemy-target
      `POTION.VULNERABLE_POTION`
    - `discard_potion:0:0` and `discard_potion:0:1`
  - executing `use_potion:0:1:1` through `sts2_perform_action` succeeded and
    the MCP settle loop returned a stable playable turn instead of an early
    draw-animation frame:
    - `post_action_settled = true`
    - `post_action_settle_reason = "player_turn_stable"`
    - `post_action_settle_polls = 2`
    - resulting hand count `= 5`
- the previously observed "full move library instead of current turn intent"
  bug is now fixed in the bridge on 2026-03-17:
  - `SafeGetMonsterIntents(...)` now prefers `monster.NextMove.Intents`
  - `monster.GetIntents()` is retained only as a fallback when the current move
    is unavailable
  - live regression on `ENCOUNTER.INFESTED_PRISMS_ELITE` confirmed:
    - `intent.state_id = "JAB_MOVE"`
    - payload now contains only the resolved current-turn attack entry:
      `22`
    - it no longer returns the monster's full candidate library such as
      `22`, `16`, `9x3`, `守势`, `强化`
- remaining enemy-intent work is now narrower:
  - higher-level semantic normalization is still needed for turns whose current
    move resolves into richer effects than the raw intent class alone conveys
  - example gaps still worth tracking:
    - `StatusCard` / summon-style intents need quantity or payload translation
      such as `status_card_count = 3`
    - compound turns still need normalized summaries such as
      `8 damage + buff`
  - impact after the current fix:
    - the agent should no longer grossly over-block because of mixed
      move-library payloads, but it can still benefit from richer
      semantic-level summaries
- X-cost combat preview mapping was also tightened on 2026-03-17:
  - the bridge now resolves `x_cost_value` against the current owner's combat
    energy when `ResolveEnergyXValue()` under-reports it
  - for X-cost attack cards whose description indicates
    "repeat / hit X times", the bridge now normalizes:
    - `effect_preview.hits`
    - `effect_preview.total_damage`
    - `effect_preview.summary`
    - `effect_preview.x_cost_semantics = "repeat_per_energy"`
  - live regression on `CARD.WHIRLWIND` (`旋风斩+`) confirmed:
    - current energy `= 3`
    - summary now returns `10 x 3 damage + X=3`
    - `total_damage = 30`
    - `damage_per_hit = 10`
    - `hits = 3`
- another live-play gap is now recorded for later bridge work:
  - combat action payloads still do not expose a reliable resolved-damage
    preview for the current board state
  - impact:
    - the agent can identify legal lethal candidates, but not exact lethal with
      enough confidence when damage has to be inferred from prior observations
    - this caused one floor-9 cleanup turn to miss lethal because an enemy at
      `9 HP` was hit by `打击` for only `6`, leaving `3 HP` and forcing one
      extra turn
- live validation on floor 9 now confirms the `燃烧契约` path end-to-end:
  - `play_card:0:4:self` transitions to `screen = "CARD_SELECTION"`
  - `card_selection.screen_type = "NPlayerHand"`
  - prompt text is captured from the combat-local selection header
  - only `card_selection:*` actions remain while selection is active
  - selector constraints are now surfaced directly to the agent:
    - `selected_count`
    - `min_select`
    - `max_select`
  - `sts2_perform_action` now also returns an `interaction_hints.card_selection`
    summary whenever the resulting state is a card-selection flow:
    - `selected_count`
    - `min_select`
    - `max_select`
    - `option_count`
    - `recommended_tool = "sts2_resolve_card_selection"`
  - `sts2_resolve_card_selection` now batches multi-select flows into one MCP
    tool call:
    - agent provides the chosen `select_indices`
    - MCP executes the per-card bridge actions internally
    - MCP then applies `terminal_action` such as `confirm`
    - selection indices are executed from high to low so index drift does not
      break multi-select flows when the UI removes already selected cards
  - single-select flows with `min_select = 1` and `max_select = 1` now
    auto-confirm after a card is chosen when the confirm action is visible
  - multi-select or arbitrary-count selection flows remain agent-controlled and
    are not auto-confirmed blindly
  - confirming returns to `screen = "COMBAT"` with:
    - the selected card moved to exhaust
    - `燃烧契约` removed from play
    - replacement draws reflected in hand and draw-pile counts
- `packages/mcp-server/autoslay-runner.js` now provides a reusable MCP client
  for starting and monitoring AutoSlay end-to-end
- AutoSlay is being kept as a regression harness, not as the intended gameplay
  control strategy for the finished project

## Phase 3: Safe Action Execution

### Goals

- Execute a small, high-confidence action set
- Prevent stale or duplicate requests

### Tasks

- Add `request_id` and `state_version` enforcement
- Implement bridge support for:
  - `play_card`
  - `end_turn`
  - `choose_option`
  - `proceed`
- Extend the action surface to early-run bootstrap scenes when they can be
  represented as stable legal actions instead of UI coordinates
- Implement MCP `sts2_perform_action`
- Implement MCP `sts2_end_turn`
- Add structured errors for stale state and invalid targets

### Current Progress Note

Observed on 2026-03-16:

- `state_version` enforcement is already live and rejects stale writes
- the bridge already exposes and executes stable legal actions for:
  - `embark`
  - `end_turn`
  - `proceed`
  - reward selection
  - event option selection
  - map travel
- staged bridge code now includes target-resolved `play_card` action generation
  and direct card execution wiring against the local `v0.99.1` assemblies
- live validation has now confirmed one combat card play end-to-end for an
  enemy-target attack card
- live validation has now also confirmed one self-target combat card play
  end-to-end
- live validation now also confirms that the title screen no longer collapses
  into `UNKNOWN`; the bridge reports `screen = "MAIN_MENU"` and exposes stable
  `main_menu:*` actions
- live validation confirms:
  - `main_menu:abandon_current_game` opens `screen = "ABANDON_RUN_CONFIRM"`
  - `main_menu:cancel_abandon_run` returns safely to `MAIN_MENU`
  - `main_menu:continue` resumes the save into a live in-run state
- live validation now also confirms that localized popup texts such as
  `不了 / 好的` are normalized into stable
  `main_menu:cancel_abandon_run / main_menu:confirm_abandon_run` action IDs
- `automation:start_autoslay` and `automation:stop_autoslay` are now exposed as
  stable bridge actions backed by the game's built-in `AutoSlay`
- launching AutoSlay through the MCP tool surface now works end-to-end
- a full successful AutoSlay run was started and monitored through the MCP
  surface on 2026-03-16
- the stale `event_option:*` suppression fix is now deployed
- a transient non-combat `PlayerCombatState.MaxEnergy` state-capture bug was
  found during AutoSlay monitoring, fixed in source, and deployed in
  `bridge_version = 0.3.1`
- the current deployed bridge/session is still:
- the bridge lineage has since advanced through the intermediate
  `0.6.x / 0.7.x` iterations and is now live-validated as:
  - `bridge_version = 0.7.4`
- live validation now also confirms two consecutive room-end reward resolutions
  through MCP:
  - floor 3:
    - claimed gold
    - claimed `POTION.FLEX_POTION`
    - selected `CARD.SWORD_BOOMERANG`
    - auto-returned to `screen = "MAP"`
  - floor 4:
    - claimed gold
    - selected `CARD.BURNING_PACT`
    - auto-returned to `screen = "MAP"`
- current live run checkpoint after the second reward resolution:
  - `screen = "MAP"`
  - current coord `(6, 3)`
  - next legal map action `map:6,4`
  - player HP `78/91`
  - gold `143`
  - deck count `13`
  - notable early pickups:
    - `愤怒`
    - `飞剑回旋镖`
    - `燃烧契约`
  - potion inventory now includes:
    - `POTION.FLEX_POTION`
- newly confirmed remaining combat-action gap:
  - cards that open an intermediate hand-selection screen are still unsafe
  - live reproduction:
    - play `CARD.BURNING_PACT`
  - the real game opens `NCardSelectionScreen` with the prompt
      `选择1张牌来消耗`
    - the bridge incorrectly stays on `screen = "COMBAT"`
    - the bridge keeps exposing stale `play_card:*` and `end_turn` actions
  - expected bridge fix surface in `BridgeGameApi.cs`:
    - capture `NCardSelectionScreen` and its visible card options in
      `CaptureContext`
    - add a normalized payload for the screen in `BuildStateFields` /
      `CreateStatePayload`
    - add explicit `card_selection:*` actions in `BuildResolvedActions`
    - suppress ordinary combat actions while that overlay is active
- room-service surfaces should follow the same bundled resolution model as
  combat rewards:
  - rest site
  - shop
  - treasure/chest
  - event options
  - any other room-local choice screen that should end by returning to map
  - expected behavior:
    - choose the room option
    - resolve any child sub-screen if one appears
    - execute the required `proceed`
    - stop only once `screen = "MAP"` or another stable non-room terminal
      surface is reached
  - `schema_version = 2026-03-16.10`
- the current bridge now additionally includes:
  - `shop` state serialization and `shop:*` legal actions
  - enemy intent payloads inside combat creature state
  - normalized `run_mode_selection` state/actions after
    `main_menu:singleplayer`
  - normalized `deck_upgrade_selection` state/actions for
    `NDeckUpgradeSelectScreen`
  - suppression of misleading `main_menu:*` actions while the run-mode submenu
    is active
  - suppression of misleading `rest_site:*` actions while the deck-upgrade
    overlay is active
- live validation has now confirmed the campfire ingress surface:
  - `main_menu:continue` resumed into `screen = "REST_SITE"`
  - `rest_site:1` successfully opened the upgrade overlay
- that same validation exposed two still-open bridge gaps:
  - the campfire upgrade overlay is not yet normalized into its own structured
    state/action surface
  - `rest_site:proceed` should not be treated as agent-legal until the room is
    actually finishable; the button can be visible before it is actionable
- live validation also exposed a bootstrap gap after
  `main_menu:singleplayer`:
  - the game first enters a run-mode selection page before character selection
  - that page is not yet represented in the bridge contract
  - during that gap the bridge can temporarily report `MAIN_MENU` with no
    stable `main_menu:*` actions even though the user-visible screen has changed
- after the run-mode selection page is passed, structured
  `character_selection` state and `embark` actions are visible again, but the
  top-level `screen` still needs normalization for that transition
- AutoSlay is now explicitly known to be unsuitable as a bootstrap substitute
  for this path:
  - starting it after a run has already been created caused it to look for a
    hard-coded `/root/Game/RootSceneContainer/MainMenu`
  - when that assumption failed, the run exited and bridge connectivity was
    lost
- reward-flow normalization has now been extended further:
  - room-end rewards now surface as `screen = "REWARDS"` instead of collapsing
    into `Room`
  - card-pick overlays now surface as `screen = "CARD_REWARD_SELECTION"`
  - the reward payload now keeps fixed card choices available directly under
    `rewards.rewards[*].reward.cards`
  - one-call MCP reward resolution was live-validated on 2026-03-16:
    - take gold reward
    - open the card reward
    - pick `愤怒` by `pick_card_index = 2`
    - auto-advance back to `screen = "MAP"`
- the bridge also now suppresses stale terminal-reward semantics once the map is
  already open, so the reward tool no longer depends on repeated blind
  `proceed` retries

### Deliverables

- mutating bridge endpoints
- mutating MCP tools
- action success and failure schemas

### Exit Criteria

- One combat card play works end-to-end
- One reward choice works end-to-end
- Replayed requests do not double-execute

## Phase 4: Robustness And Developer UX

### Goals

- Make the system comfortable to iterate on
- Reduce breakage when the game is updated

### Tasks

- Add better bridge logs and request tracing
- Add a compatibility banner with game version and bridge version
- Add unsupported-scene diagnostics
- Add a replayable local mock server for MCP development
- Add end-to-end smoke checklist
- Add configuration file for visible-only mode and port overrides

### Deliverables

- clearer logs
- configuration surface
- manual recovery playbook

### Exit Criteria

- A broken or outdated bridge fails clearly
- A developer can reproduce the main loop without guessing

## Phase 5: Optional Knowledge Layer

This phase is intentionally optional.

### Possible Additions

- card and relic catalog export
- action explanation helpers
- event and encounter taxonomy
- offline replay analyzer
- debug-only hidden-state mode for research

### Important Constraint

Do not block the bridge project on RAG. The MCP bridge should be useful before
any external knowledge base exists.

## Suggested Initial Repository Structure

```text
docs/
packages/
  mcp-server/
mods/
  sts2-bridge/
schemas/
```

## Definition Of Done For v0.1

`v0.1` is done when all of the following are true:

- The bridge mod loads into the installed game build
- The MCP server can connect to the bridge locally
- `sts2_get_state` returns useful battle, map, and choice state
- `sts2_list_actions` returns current legal actions
- `sts2_perform_action` can safely play a card or choose an option
- Requests are guarded by `state_version`
- The default mode is visible-state-only
- Failures are structured and understandable

## Risks

### Risk: Unclear Mod Loader Surface

Mitigation:

- Treat Phase 0 as a hard gate
- Reuse the published StS2 tutorial workflow or template if available
- Keep the bridge mod tiny until the load path is proven

### Risk: Game Update Breaks Hook Points

Mitigation:

- Log detected game version at startup
- Isolate hook logic in a thin adapter layer
- Prefer public APIs and scene graph access over brittle patches

### Risk: MCP Actions Drift From Real Game State

Mitigation:

- Always require `state_version`
- Return current state version after every write
- Do not maintain speculative state in the MCP layer

### Risk: Desktop Automation Breaks When The Game Is Backgrounded

Mitigation:

- Keep screenshots and clicks out of the core gameplay path
- Treat window focus as a prerequisite for any Windows-level automation
- Prefer bridge scene detection and legal action IDs for all gameplay surfaces
- Use desktop automation only as a temporary bootstrap fallback

### Risk: Scope Creep Into Strategy And Data Ingestion

Mitigation:

- Keep RAG and card databases out of the critical path
- Finish the bridge before knowledge tooling

## Current Breakpoint

- Live blocker observed on 2026-03-17 with
  `bridge_version = 0.7.7` / `schema_version = 2026-03-17.1`:
  - after the Act 1 boss kill, room rewards resolved correctly
  - `proceed` from terminal rewards returned to `screen = "MAP"`
  - but that map state was still the Act 1 boss shell:
    - `act = "密林"`
    - `act_floor = 17`
    - `current_room.room_type = "Boss"`
    - `current_room.is_pre_finished = true`
    - `map.is_travel_enabled = true`
    - no legal `map:*` actions were exposed
    - the only remaining non-automation action was potion discard
- User confirmed the visible UI still showed a red back button that returns to
  the reward/room layer, so this `MAP` state is not the real next-act decision
  surface.
- The old heuristic
  `screen = "MAP" && map.is_travel_enabled == true`
  is therefore insufficient after boss terminal rewards and must not be used as
  a universal room-end completion check.
- A local bridge source patch is now staged in `BridgeGameApi.cs`:
  - `InvokeTerminalRewardsProceed(...)` now prefers the visible terminal-reward
    button / `NRewardsScreen.OnProceedButtonPressed()`
  - it only falls back to `RunManager.ProceedFromTerminalRewardsScreen()` if no
    real reward-button path is available
  - local compile passes with `-p:Sts2SkipDeploy=true`
  - the fix is not yet deployed into the live game process because the loaded
    mod DLL is locked by `SlayTheSpire2.exe`
- Follow-up live regression on 2026-03-17 after deploying that patch narrowed
  the bug further:
  - `proceed` no longer misroutes into the stale Act 1 map shell
  - but it now returns `state_changed = false`
  - a live desktop snapshot at the same moment showed the real game UI was
    already back in the boss room with a visible `前进` button
  - bridge state still reported `screen = "REWARDS"` with
    `Proceed from terminal rewards`
- That means the deeper blocker is now likely stale reward-screen detection /
  screen normalization, not only the terminal-reward action target itself.

## Immediate Next Tasks

1. Fix the post-boss act-transition breakpoint so terminal rewards do not
   falsely terminate on the old-act boss map shell.
2. Lock the exact native mod skeleton for `mods/sts2-bridge`.
3. Record the required `.csproj` references and output-copy behavior.
4. Prove bridge startup inside the running game with logs enabled.
5. Return the smallest possible JSON state payload.
6. Normalize the post-`singleplayer` run-mode selection page into a structured
   bridge surface so desktop clicks are no longer needed for fresh-run
   bootstrap.
7. Expand `card_selection` coverage to any remaining combat-local choice modes
   that do not use `NPlayerHand` or `CardSelection.*Screen`.
8. Add the remaining potion-replacement flow so room rewards can stay one-call
   even when potion slots are full.
9. Decide whether map states that expose a single forced travel option should
   auto-advance that move or leave it to the outer agent.
10. Finish live verification for `shop` and enemy-intent payloads after the
   `card_selection` combat path is confirmed live.
11. Validate the newly added combat hand/action preview payloads live:
   - source work is now in place:
     - `card.description` resolves via `CardModel.GetDescriptionForPile(...)`
     - `card.dynamic_vars[]` exposes previewed dynamic values
     - `card.effect_preview` exposes common tactical fields directly
   - live verification still needed:
     - `打击 = 6 damage`
     - `愤怒 = 6 damage`
     - `铁斩波 = 5 damage + 5 block`
     - `飞剑回旋镖 = 3 x 3 damage` in single-target combat
   - if any mismatch remains, inspect whether preview mode or target binding
     needs to differ between hand payloads and action payloads
12. Expose event glossary / tooltip explanations in a compact form:
   - bridge source now captures event-page glossary text from the visible
     hover-tip set when present
   - fallback path captures extra visible text from the event room itself after
     subtracting the option-button title/description text
   - new `event_options` fields:
     - `visible_glossary_source`
     - `visible_glossary_texts[]`
     - `visible_glossary[]` with `{ title, description, texts[] }`
     - `options[*].glossary[]` with the option's own resolved hover-tip terms
       even when the left-side tooltip panel is not currently open
   - MCP compact summary preserves `visible_glossary` so the outer agent can
     reason over event-specific keyword explanations without querying the raw
     40 KB bridge payload
   - live validation on the Tezcatara Act 2 event confirmed:
     - `营养汤` now returns
       - `特兹卡塔拉的余烬 -> 费用为0且获得永恒。`
       - `永恒 -> 无法从你的牌组中移除或变化。`
     - `烘焙手套` now returns
       - `消耗 -> 在战斗结束前移除。`
       - `力量 -> 力量会增加攻击牌造成的伤害。`
     - `南瓜蜡烛` now returns
       - `能量 -> 能量用于打出你手牌中的卡牌。`
13. Push placeholder resolution down into the raw bridge payload:
   - user feedback: compact MCP cleanup was not sufficient because raw bridge
     payloads still leaked unresolved tokens such as:
     - `{StrengthPower}`
     - `{Energy:energyIcons()}`
   - bridge source now resolves descriptions through a shared text path that:
     - prefers model-native dynamic text members when available, such as
       `DynamicDescription`, `SmartDescription`, and `DynamicEventDescription`
     - falls back to placeholder substitution against reflected model members
     - additionally resolves token values from `CanonicalVars` / `DynamicVars`
       when a model stores dynamic text values there instead of as plain
       properties
   - live validation on the Act 2 Tezcatara event after redeploy confirmed:
     - `event_options.options[1].relic.description`
       now returns
       `在你的回合开始时，消耗你抽牌堆顶部的牌并获得1点力量。`
     - `event_options.options[2].relic.description`
       now returns
       `在每个回合开始时获得[img]...energy_icon...[/img]。这件遗物将在第3阶段开始时熄灭。`
     - the raw payload no longer exposes the previous unresolved placeholders
    - bridge follow-up on 2026-03-17 pushed the cleanup one layer lower across
      payload-facing text surfaces:
      - `DescribeText(...)` now normalizes raw descriptions after placeholder
        substitution instead of leaving UI-render markup behind
      - BBCode-style color tags such as `[gold]...[/gold]` and `[blue]...[/blue]`
        are stripped from raw payload text
      - energy image tags such as
        `[img]...ironclad_energy_icon.png[/img]` are converted to plain
        `1点能量`
      - 2026-03-18 refinement:
        - icon translation now keys off the exact sprite-font resource name
          instead of mapping every unknown `[img]` to `图标`
        - exact `star_icon` resolves to `1点星辉`
        - exact `*_energy_icon` variants resolve to `1点能量`
        - repeated icon runs and compact count forms like
          `3[img]...energy_icon...[/img]` collapse to counted text
        - unknown icons keep a diagnostic label such as `图标:some_icon_name`
          so future cases can be inspected without losing source information
      - 2026-03-18 event/power placeholder follow-up:
        - event option `LocString` payloads were still missing values such as
          `Heal` / `HpLoss` because the option itself does not carry the owning
          `EventModel.DynamicVars`
        - bridge now pulls the current `NEventRoom._event` model when building
          `event_options.options[*]` and formats option title/description
          against both the option and the active event model
        - `DescribeText(...)` now always includes the text object itself in
          placeholder resolution, so `LocString.Variables` can satisfy tokens
          like `Amount`
        - placeholder context enumeration now unwraps composite context arrays,
          which lets bridge-only context objects stay internal while the MCP
          payload still exposes only final assembled strings
        - live re-test on `EVENT.DENSE_VEGETATION` after redeploy confirmed:
          - `坚持跋涉 -> 从你的牌组中移除一张牌。失去11点生命。`
          - `休息 -> 回复24点生命。进入战斗。`
        - recent `godot.log` tail after this re-test showed no fresh
          `Localization formatting error` lines
      - node-sourced visible glossary / prompt text now goes through the same
        normalization path, so raw `/state` text is closer to the MCP compact
        summary
   - live validation on the current Act 2 hopper reward checkpoint confirmed:
     - `players[0].relics[*].description` for `南瓜蜡烛` now returns
       `在每个回合开始时获得1点能量。这件遗物将在第3阶段开始时熄灭。`
     - reward, potion, relic, and card descriptions at that checkpoint all
       surfaced as plain readable text in raw `/state`
14. Harden post-action transition settlement inside bridge + MCP:
   - reproduced issue:
     - `main_menu:continue` could return a half-transition frame with
       `screen = "Room"` and only `discard_potion:*` visible before the real
       reward screen surfaced
   - bridge follow-up:
     - action responses that commonly transition screens now poll after the
       initial wait barrier until a real interactable surface is visible
     - accepted stable surfaces currently include:
       - rewards / card reward selection
       - card selection
       - rest site / deck upgrade
       - event / shop / map
       - combat with visible non-automation actions
       - a lone `proceed` action
   - MCP follow-up:
     - `main_menu:continue` no longer settles on generic
       `screen != MAIN_MENU`
     - the previous generic out-of-combat `screen:Room` fallback has been
       removed so half-transition room frames do not count as stable
   - live validation after redeploy on 2026-03-17:
     - restarting the game and replaying the same save now returns directly to
       `screen = "REWARDS"` with reward actions already populated
15. Make target suffixes readable enough for combat readback / save reload:
   - reproduced issue:
     - action ids such as `play_card:0:0:2` exposed the internal target suffix
       but did not make it obvious which creature `:2` referred to
     - this caused a live mis-target during the Act 2 `胧光怪` checkpoint and
       made that fight hard to use as a deterministic regression sample
   - bridge follow-up:
     - `play_card` / `use_potion` actions now also expose:
       - `target_action_suffix`
       - `target_combat_id`
       - `target_name`
       - `target_side`
       - `target_mapping`
     - action labels now render the suffix mapping directly, for example:
       `1 = 胧光怪 (combat_id 1)`
     - combat state now exposes `combat.target_index_map[]` so the current
       suffix-to-creature mapping is available even before inspecting any
       individual action payload
   - live validation after redeploy on 2026-03-17:
     - reloading the current `胧光怪` fight now surfaces:
       - `combat.target_index_map = [self/0 -> 铁甲战士, 1 -> 胧光怪]`
       - action labels such as
         `Play card 0: 啄击 -> 1 = 胧光怪 (combat_id 1)`
