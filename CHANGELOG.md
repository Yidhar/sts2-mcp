# Changelog

## [Unreleased] — architecture-v2

This release line is an intentional contract and ownership reset. See
`docs/migration/README.md` before upgrading a live Bridge/MCP pair or resuming
historical RL artifacts.

### Breaking changes

- Changed the normal MCP default from the historical debug-sized surface to the
  `minimal` player-control profile. `strategic` and privileged `debug` are explicit.
- Require strict `expected_state_version` for gameplay mutation; `strict=false` is rejected.
- Removed automatic retry of legacy, non-idempotent mutations. A timeout is an unknown
  outcome and must be reconciled from current state.
- Moved privileged reset/step behind the separate `training` capability; step is bound
  to `episode_id` and `expected_step_index`.
- Disabled `legacy-v1` by default. It is published only when the operator explicitly
  sets `STS2_BRIDGE_ENABLE_LEGACY_V1=true`; canonical v2 environment step accepts
  exactly one `action_handle` or `action_index` and no `action_id` alias.
- Replaced the failed MuZero/token-memory/MCTS line with
  `python -m sts2_rl.train`. Old checkpoints and replay are not compatible with the
  new randomly initialized grounded baseline.
- Advanced the grounded observation ABI from runtime-mechanics v6 to relational
  runtime encoding v8 with a 224-feature minimum. Definition/instance/zone
  channels, exact counted orderless card multisets, and the run/combat memory
  split intentionally reject earlier card/selection checkpoints rather than
  padding or migrating them.
- Moved runtime artifacts outside the checkout through `STS2_ARTIFACT_ROOT`.
- Advanced the recurrent checkpoint ABI to
  `sts2-recurrent-vtrace-checkpoint-v4`. Exact resume now includes every
  enabled replay sidecar and rejects v3. A v3 policy may be used only through
  explicit model-parameter initialization into a fresh lineage; the complete
  long-horizon head group starts fresh and optimizer/queue/RNG/counters/replay
  are never imported.

### Added

- Added a loopback-only, read-only training dashboard with incremental JSONL
  telemetry, exact-resume chain aggregation, held-out/training separation,
  conservative lifecycle evidence, lightweight atomic-checkpoint validation,
  bounded native SVG trends, and a Windows background launcher. The service
  never deserializes checkpoint payloads or exposes training controls.
- Added bounded complete-episode credit assignment for native-revival full-run
  preheat. The collector retains immutable CPU snapshots independently of
  streamed short unrolls and backfills factual combat/Act/run success,
  forward-return, future revival and HP-loss labels in one reverse pass.
  Win/failure/censored replay is bounded by episode count, total bytes,
  per-episode bytes and sampling quota. The learner reconstructs exact split-GRU
  state under `no_grad` and keeps autograd to a short active-shape suffix, so a
  10,000+ decision source episode cannot create a full-episode GPU graph. Both
  prefix and suffix replay run with dropout disabled and restore the caller's
  model mode, preserving exact sparse/full-history equivalence.
- Added candidate-independent combat/Act/run task-value and non-negative
  revival-cost heads. Completion/progress remains primary; revival policy/value
  labels are success-conditional. Before success classification the secondary
  signal is residual-capped; only inside an explicit narrow, learned-success
  tie band may a bounded nominal-primary floor preserve the revival tie-break
  after the primary advantage reaches zero. Forced singleton steps retain
  value supervision but have no policy replay target.
- Added held-out Act-boundary revival/HP counters, Act-1 and run completion at
  zero or one revival, and success-conditional run efficiency summaries.
- Added a bounded one-FIFO-batch learner lag when complete-episode replay is
  enabled. The runtime commits an episode to replay before learning its final
  pending online batch, so even a one-episode run receives long-horizon credit;
  held-out evaluation seeds are rejected from every recorded collection path.

- Added contract API `2.0.0` with JSON Schema 2020-12, OpenAPI, fixtures, and generated
  C#/TypeScript/Python version constants.
- Added a versioned `game-data` package and deterministic provenance/hash manifest.
- Added Bridge v2 session descriptors, scoped tokens, player-visible state, bounded
  events/health, a single mutation gate, idempotent result retention, command-status
  lookup, and serialized environment operations.
- Added the TypeScript MCP 0.5 server using the official SDK, Zod validation, loopback-
  only session forwarding, credential redaction, profile-owned tools, and one-shot
  legacy mutation semantics.
- Added the typed Python `sts2_rl` contract/backend/reward/checkpoint foundation, atomic
  checkpoint helpers, portable game-data lookup, scoped live/headless backends, external
  reward ownership, process-identity checks, and stricter lifecycle tests in RL 0.2.
- Added RL 0.3: a 3,642,824-parameter candidate-independent world encoder, grounded
  legal-candidate policy/Q heads, dual combat/run values, fixed normalized reward,
  coverage/recent/PER replay, direct typed-backend collector, actor-critic learner,
  disjoint held-out evaluation seeds with per-seed metrics, complete reward-projection
  fingerprints in replay, exact checkpoint resume and a dry-run CLI gate. No new
  long-run or Act 1 performance result is claimed in this release.
- Added episode-aggregated collector/learner stage timings, a warm-up credit policy
  that avoids synchronous replay-debt catch-up, and a fail-closed HeadlessSim build
  identity gate binding formal training to the locked source commit and binary hash.
- Replaced replay-time raw observation re-encoding with versioned sparse encoded
  decision snapshots and whole-batch collation. The checkpoint/replay and encoding
  ABIs advance to v2 and intentionally reject pre-snapshot profiling checkpoints.
- Added a bounded, opt-in collector/learner overlap pipeline with an independent
  actor model, single in-flight episode, main-thread-owned replay, quiescent
  checkpoint/evaluation barriers, interrupt draining, policy-version lag metrics,
  and configurable collector device. Controlled ROCm profiling found only a 3.4%
  same-device gain and a 10.4% CPU-actor gain with 3–28 update policy lag, so the
  maintained profiles remain synchronous and the typed config ABI advances to v2.
- Added Bridge core tests, MCP SDK/unit tests, contract/repository/data/release gates,
  GitHub Actions CI, artifact migration tooling, and v2 architecture/runbooks.
- Added exact Node/npm/Python/.NET pins, exact dependency closures, SHA-pinned GitHub
  Actions, deterministic CycloneDX SBOM generation, and source/artifact provenance.
- Added native catalogs and runtime transport for powers, relics, potions, card
  enchantments/afflictions, events, observable enemy moves and boss encounter identity;
  formal headless training now fails closed on a catalog and event-to-combat mechanics
  preflight before launching collectors.
- Added Bridge 0.9 and RL 0.4 relational training facts: collision-free process-local
  card references, player-visible unordered draw composition, full active map/shop/rest/
  reward/selection payloads, fixed pile zones, exact action source/target binding,
  owner relations for card modifiers and enemy mechanics, guaranteed mutation facts,
  learned exact-instance vs same-definition grounding, and separate run/combat recurrent
  memory. No card tier, target priority, route score, predicted outcome, or new reward
  heuristic was introduced.

### Fixed

- Closed two HeadlessSim legal-action/executor gaps exposed by long full-run
  native-revival training. Merchant potion actions are now published only when
  the item is stocked and affordable, the player has an open potion slot, and
  the game's authoritative `ShouldProcurePotion` hook permits acquisition
  (including Sozu); raw merchant inventory indices remain unchanged. Cards that
  require a target now publish no action when no valid target exists and retain
  one candidate per real target otherwise. The pinned simulator patch includes
  mirrored Headless/Overlay helpers, direct dependency self-tests, lock/hash
  identity binding, and opaque raw-action round-trip regression coverage.
- Added a generic 256-decision non-combat durable-progress boundary. It advances
  only when the run locus changes or a previously unseen same-locus persistent
  resource state appears; remembered resource fingerprints prevent bounded
  `A -> B -> A` cycles from resetting it forever. Dynamic event previews such as
  an ever-increasing `HpLoss`, changing option labels, and select/cancel or
  select/deselect UI cycles therefore cannot consume the 30,000-step transport
  ceiling. The detector records compact fingerprints and room context, never
  card/event-specific policy rules.
- Replaced full-state-per-decision held-out journals with trajectory journal v3.
  Every decision keeps a compact auditable summary, while complete semantic
  snapshots are limited to episode boundaries, every 256 decisions, detected
  anomalies, and eight preceding context decisions. Evaluation metrics remain
  unchanged, and the collector no longer deep-copies the complete observation
  and legal-action surface on every held-out step. Version 3 consumers must read
  the `summary` / `rich_snapshot` record union; compact strings are byte-bounded
  with prefix, length, and SHA-256 identity when necessary. Stall snapshots keep
  both the pre-action decision and the exact transition-result state that
  triggered the detector.
- Corrected checkpoint provenance after the first save in a live process.
  Subsequent periodic/final checkpoints now use `in_process_successor` rather
  than claiming `exact_resume`; genuine fresh, exact-resume, and model-parameter-
  initialization starts retain their distinct first-checkpoint relationships.
  Every non-fresh parent is now required to be a complete, hash-verified atomic
  training checkpoint with consistent IDs, metadata and semantic identity.
- Added the lightweight `maximum_observed_candidates` scalar to episode, actor,
  held-out, run, and checkpoint metrics. It is measured from active snapshot
  shapes without per-step logging or fixed 256-wide padding; legacy v3
  checkpoints may migrate only this absent diagnostic field to zero.

- Replaced copy-proportional Deck/Draw/Discard/Exhaust/Play expansion with an
  exact counted multiset of fact-identical card variants. Concrete hand,
  selection and legal-action entities remain unaggregated, while upgrade,
  cost, modifier and lifecycle differences remain separate. The fail-closed
  world ceiling is now 2,048 and overflow reports exact total demand plus
  top-level branch counts instead of an ambiguous pending-queue size.
- Added an acknowledged actor/main-thread episode boundary. A completed episode
  cannot reset into the next run before metrics, evaluation intent and periodic
  checkpoint state are committed, closing the race that lost the first
  10,000-step checkpoint when episode two failed. Policy snapshots are now
  adopted by the actor after complete recurrent unrolls, not only between
  10,000-step episodes, so learner progress affects an ongoing full run without
  mutating a model mid-forward or mislabelling behavior-policy data. Learner
  metrics now distinguish global from batch environment steps and include a
  compact per-unroll Act/floor/revival/reward snapshot. The preheat outer
  transport ceiling is 30,000 decisions so one native run can traverse Acts
  1--3, while exact semantic deadlock detection remains the early loop exit.
- Restored a statistically useful deterministic preheat evaluation: 12 fixed
  held-out odd seeds at fresh-lineage step zero and again at 30k/100k/250k.
  Evaluation summaries now report both counts and rates for Act 1 clears and
  Act 3 reaches instead of relying on the maximum floor of training episodes.
- Replaced the damage-event combat stall boundary with observation-grounded net
  progress. The 256-decision anchor advances only after a 5% net reduction in
  enemy-health burden or a real phase/wave transition; damage that is healed
  back, summon churn and repeated low-value hits no longer keep an exhausted
  unlimited-revival combat alive until the 30,000-step transport ceiling.
  Compact actor progress records the anchor, required reduction and no-net-
  progress age without enabling verbose per-decision training journals.
- Raised fail-closed legal-candidate capacity from 96 to 256 without truncation.
  Sparse snapshots and learner collation still pad only to the largest active
  candidate count in each batch; regression coverage includes a 111-candidate
  multi-select/reselect discard/transform state backed by a 600-card deck.
- Added a tested model-parameter initialization migration for capacity/config
  changes that preserve learned tensor and feature ABI. It imports only the
  learner network, republishes it to the actor, resets optimizer/queue/RNG/
  counters into a new lineage, and records `model_parameter_initialization` as
  parent provenance. Exact resume remains fail-closed and is never imitated.
- Raised the relational world-token ceiling from 512 to 1,024 after the first
  complete 10,000-decision run reached Act 1 floor 17 and the following run
  exposed 61 additional pending factual nodes beyond the old limit. Overflow
  remains fail-closed; active-shape collation avoids padding every update to the
  ceiling. Native-revival preheat now checkpoints every completed 10,000-step
  horizon so a later interface failure cannot discard multiple episodes.
- Changed the maintained standard `full-run` profile from the Act-1 diagnostic
  objective to the complete-run objective. Entering Act 2 is now ordinary
  progress rather than an implicit episode boundary; `act1` remains available
  only when explicitly selected for diagnostics.
- Raised the fail-closed candidate-local fact capacity from 24 to 64 after the
  first uninterrupted full-run collection crossed an Act transition and then
  encountered a legal action requiring at least 26 tokens. Overflow diagnostics
  now report the exact required capacity and action kind; legal-action facts are
  still never truncated.
- Removed the random-combat/Act-1 preheat launch gate: native-revival preheat
  now enters the uninterrupted full game immediately. Learner batches pad to
  active token/candidate sizes and the preheat recurrent batch is reduced to
  4x16, preventing the first 8x64 padded Transformer update from blocking the
  collector for hours without producing one optimizer step.
- Increased the headless backend's bounded ten-minute request-id replay store from
  2,048 to 65,536 entries and exposed its capacity/TTL in the backend spec. The old
  default exhausted before any identity could expire during normal online collection
  and terminated the first formal baseline run after 1,971 environment steps.

### Removed or archived

- Removed the separate in-game Draft Tracker in favor of an external event recorder.
- Removed checked-in release binaries, audit/launcher logs, PID/command state, gate
  output, decompilations, and source slices.
- Removed AutoSlay/RL smoke runners and journal/knowledge/observation persistence from
  the MCP control-plane runtime.
- Removed Bridge static-export execution, duplicate in-game Draft Tracker ownership,
  legacy PPO/attention packages, obsolete supervisors/probes, and checked-in RL-owned
  game-data copies.
- Removed the complete MuZero/MCTS/token-memory/latent-dynamics/future-world stack,
  objective heads, semantic planners, boss/card/potion/route heuristics, tactical
  action guards, old offline/expert trainers, their tests, scripts and recovery notes.
- Removed scored/prior-bearing and hand-curated card strategy data. Game-data 2.0
  retains only policy-free static facts and strips semantic tags/signals at generation.
- Archived pre-v2 architecture and implementation history under `docs/archive/v1`.

### Compatibility

- Bridge keeps an opt-in, disabled-by-default `legacy-v1` migration surface. It is not
  advertised unless explicitly enabled; legacy mutations remain non-idempotent and are
  not the final security boundary.
- Old learner replay/checkpoints are not migration inputs. Exact resume validates the
  grounded-baseline contract, reward, dependency locks, typed lineage config, encoding
  fingerprint, model, optimizer, replay and stochastic continuation state and fails
  closed otherwise. Valid static game-data is optional provenance, not a runtime or
  model identity.
- V1 removal is gated by contract, idempotency, concurrency, capability, live/headless
  parity, checkpoint migration, CI, live end-to-end, and zero-v1-client evidence.


## v0.7.12 - 2026-03-22

Repository release: `v0.7.12`

Bridge mod version: `0.7.12`

Bridge state schema: `2026-03-20.1`

MCP server version: `0.4.20`

### Highlights

- Replaced the MCP server's poll-heavy state sync path with a bridge frontier event stream plus event-driven wait helpers, which makes continuous combat actions noticeably faster and reduces stale-action windows.
- Added post-action settlement on the MCP side so single actions and combat sequences now wait for the next stable actionable surface instead of returning too early while the bridge is still transitioning.
- Added a full MCP-native knowledge and observation layer, so strategy content is queried through tools instead of being stuffed into prompt attachments.
- Continued token reduction work across bridge and MCP payloads by switching bridge JSON output to compact formatting and expanding compact action/state summaries.

### Bridge Mod Changes

- Added authenticated `GET /events` support that streams frontier updates directly from the bridge.
- Added frontier lifecycle hooks through `BridgeCoordinator` so the frontier store is reset on detach and pumped every main-thread tick.
- Updated the bridge HTTP writer to emit compact JSON instead of indented JSON, reducing response size without changing semantics.
- Bumped the bridge state schema to `2026-03-20.1`.
- Exposed card star-cost metadata in bridge card payloads:
- `canonical_star_cost`
- `current_star_cost`
- `has_star_cost_x`
- Preserved `semantic_state_hash` in the hydrated state payload so MCP-side event followers can reason about stable semantic snapshots.

### MCP Server Changes

- Added a persistent bridge event client that follows `/events` and maintains a live cached frontier.
- Added `sts2_wait_until_actionable`, an event-driven wait primitive that returns only when a stable actionable surface is available.
- Reworked post-action settlement:
- `performBridgeAction(...)` now settles by default instead of only waiting for the immediate action response.
- Combat actions, `end_turn`, screen transitions, and map travel now use strategy-specific settlement logic.
- Event-driven quiet-window settling now absorbs delayed follow-up surfaces such as discard selection, retain selection, reward cleanup, and next-turn hand refill.
- Fixed one of the worst failure modes in combat sequencing:
- when a sequence now runs into a blocker mid-turn, it returns partial progress with `ok: true`, `resolved: false`, `executed_steps`, and the current state instead of surfacing a misleading top-level failure.
- Added monotonic state caching on the MCP side so older `state_version` snapshots from HTTP/SSE cannot overwrite newer cached state.
- Added tool profiles so the same MCP server can expose `minimal`, `strategic`, or `debug` tool sets without turning the project into an agent framework.
- Added MCP-native knowledge tools:
- `sts2_get_knowledge`
- `sts2_get_knowledge_topics`
- `sts2_search_knowledge`
- `sts2_read_knowledge_slice`
- `sts2_list_knowledge_sections`
- Added evidence-first observation tools:
- `sts2_record_observation`
- `sts2_list_observation_entities`
- `sts2_read_observation_entity`
- Added run journaling tools:
- `sts2_journal_write`
- `sts2_journal_read`
- `sts2_journal_summarize`
- `sts2_journal_get_summary`
- `sts2_journal_list_runs`
- Added canonical knowledge content under `packages/mcp-server/knowledge/` for route planning, deck building, card evaluation, combat, bosses, enemies, relics, events, and authoring workflow.

### Live Validation

- Live combat validation confirmed that a mixed sequence such as `中和 + 打击 + 打击 + 防御 + end_turn` now completes without the old mid-sequence `state_version_conflict` failure.
- Live validation on `生存者` confirmed that:
- single `sts2_perform_action` now returns the follow-up `CARD_SELECTION` surface instead of stopping too early on the previous `COMBAT` state.
- `sts2_execute_combat_sequence` now stops cleanly at the discard selection with:
- `resolved: false`
- `reason: "card_selection_ready"`
- `executed_count: 1`
- the remaining steps still pending instead of being misreported as failed or accidentally executed.

### Upgrade Notes

- Restart the game after replacing `sts2-bridge.dll`, otherwise `/events`, schema `2026-03-20.1`, and the newer bridge payload fields will not exist in the running bridge process.
- Restart any long-lived `sts2` MCP server process after updating `packages/mcp-server/index.js`, otherwise it will continue using the older polling and settlement logic.

### Release Assets

- `sts2-bridge-v0.7.12.dll`
- `sts2-bridge-v0.7.12.zip`

## v0.7.11 - 2026-03-19

Repository release: `v0.7.11`

Bridge mod version: `0.7.11`

Bridge state schema: `2026-03-19.1`

MCP server version: `0.4.19`

### Highlights

- Added full Crystal Sphere event support, including screen detection, compact event state, legal actions, and live-validated click execution.
- Reduced false `state_version` churn by hashing semantic game state instead of volatile automation and expanded action payload trees.
- Fixed multiple card-selection stability problems: drifting hand indices, missing confirm buttons on deck-card selection screens, and noisy completion errors.
- Compressed Crystal Sphere responses down to the minimum actionable shape while keeping direct action-id reconstruction possible.

### Bridge Mod Changes

- Added `EVENT_CRYSTAL_SPHERE` screen detection in the bridge state capture path.
- Added top-level `crystal_sphere` state payload with divination count, selected tool, grid size, hidden-cell count, and revealed item summaries.
- Added Crystal Sphere event options and actions into the shared event surface so indexed option tooling can address the event without a dedicated new tool.
- Fixed Crystal Sphere execution by routing through the screen-level handlers:
- `SetSmallDivination`
- `SetBigDivination`
- `OnCellClicked`
- `OnProceedButtonPressed`
- This replaces the earlier no-op path that tried to click the cell/button nodes directly.
- Introduced semantic state hashing via `CreateSemanticStateCore(...)` so transient fields like automation payloads, verbose action text, and other non-decision metadata no longer perturb `state_version`.
- Stabilized card-selection option indexing by deriving combat-hand indices from the real hand pile order instead of pure visual order.
- Added `selection_id` support to card-selection options so MCP-side rematching can survive index drift.
- Added `NDeckCardSelectScreen`-aware confirm button resolution so two-step selection flows expose the correct confirm target after cards are chosen.
- Treated benign `CompleteSelection` final-state exceptions as successful completion instead of surfacing a false internal error.
- Fixed count-prefixed image-tag compaction so strings like `9[star icon]` are rendered as `9点星辉` instead of malformed numeric text.

### MCP Server Changes

- Added compact Crystal Sphere summaries to `sts2_get_state`, `sts2_perform_action(return_state_after=true)`, and `sts2_list_actions`.
- Crystal Sphere responses now use a compact action map:
- `controls`: short strings like `0:small*` and `1:big`
- `cell_action_start_index`: the first `event_option:N` index for revealable cells
- `cells`: compact coordinate list where `event_option:(start + offset)` maps to the corresponding coordinate
- `revealed`: short strings like `4,0=gold:10`
- Suppressed duplicate `event_options` expansion on the Crystal Sphere screen, because the same decision surface is already represented in `crystal_sphere.actions`.
- Collapsed Crystal Sphere `sts2_list_actions` output into one `event_option_group` instead of enumerating every `event_option:N` row as a separate object.
- `sts2_perform_action` compact post-action summaries now preserve `crystal_sphere` state, so event follow-up decisions do not require an immediate extra `get_state`.
- `sts2_pick_option` and `sts2_resolve_card_selection` now use indexed-option helpers plus stable `selection_id` matching instead of assuming visible-option indices are fixed.
- Added `star_cost` to agent-facing card summaries and compact combat action summaries.

### Practical Result

- Live validation on a fresh local MCP process confirmed Crystal Sphere tool switching and cell reveals now change the real game state.
- Example live click: `event_option:2` advanced the event, reduced divinations from `2` to `1`, and updated the revealable-cell set as expected.
- In the tested Crystal Sphere scene, compact response sizes were reduced to roughly:
- `sts2_get_state`: ~`1.5k` chars
- `sts2_list_actions`: ~`1.1k` chars
- `sts2_perform_action(return_state_after=true)`: ~`1.6k` chars

### Upgrade Notes

- Restart the game after replacing `sts2-bridge.dll`, otherwise the new bridge payload fields and action wiring will not be loaded.
- Restart any long-lived `sts2` MCP process after updating `packages/mcp-server/index.js`, otherwise it will keep using the older summary logic.

### Release Assets

- `sts2-bridge-v0.7.11.dll`
- `sts2-bridge-v0.7.11.zip`

## v0.7.10 - 2026-03-19

Repository release: `v0.7.10`

Bridge mod version: `0.7.10`

MCP server version: `0.4.18`

### Highlights

- Added native smith upgrade previews to `deck_upgrade_selection.options[*].upgrade_preview`, built from the game's own `CloneCard + UpgradeInternal` flow instead of guessed text or manually mapped values.
- Reduced high-noise MCP responses across combat, rewards, campfire, shop, and map tools so agents get the minimum usable state instead of 200-800 line raw payloads.
- Improved combat automation guidance and sequence handling so mixed combat plans can use one call instead of multiple fragile round trips.

### Bridge Mod Changes

- `BuildDeckUpgradeSelectionPayload(...)` now includes `upgrade_preview` for every smith candidate.
- Upgrade previews come from a cloned upgraded card payload, so title, cost, effect summary, description, and dynamic vars follow the same logic as the in-game preview UI.
- Rebuilt and deployed `sts2-bridge.dll` for this release.

### MCP Server Changes

- `sts2_perform_action` and `sts2_end_turn`
- `return_state_after=true` now returns a compact post-action summary instead of the full bridge state.
- Removed redundant fields such as duplicated hashes, raw action lists, and full state trees when a compact summary is available.

- `sts2_play_card_sequence` and `sts2_execute_combat_sequence`
- Added mixed combat sequence support so one sequence can include `play_card`, `use_potion`, and `end_turn`.
- Continued support for post-action reindex matching after hand shifts, draws, and target remaps.
- Compressed `executed_steps` output so exact successful steps collapse to the executed action id, while `requested_action_id` is only kept when a remap or failure actually matters.
- Updated tool descriptions and agent-facing hints to push consecutive combat actions toward sequence tools instead of parallel `sts2_perform_action` calls.

- `sts2_resolve_room_rewards`
- Added a specialized compact resolver payload.
- The tool now reports the reward resolution result, claimed rewards, selected card, executed actions, and final compact state without echoing the entire reward flow twice.
- Fixed the "safe rewards + card pick + auto proceed" one-call path so the result is usable without extra manual cleanup calls.

- `sts2_resolve_rest_site`
- Added a specialized compact resolver payload for both success and unresolved smith flows.
- Smith choice prompts now return compact preview strings such as `14:武装 -> 武装+ | 1费 | 获得5点格挡。 / 升级你手牌中的所有牌。`
- Removed duplicate `state.screen`, empty `selected_option` / `selected_upgrade_card`, and repeated rest-site sections from unresolved responses.

- `sts2_resolve_shop_visit`
- Added a specialized compact resolver payload for purchase plans, purchased items, removal choices, and final compact state.
- Shop and card-removal follow-up prompts now avoid duplicating large shop/state payload sections.

- `sts2_travel_to_coordinate`
- Added a specialized compact resolver payload.
- Travel responses now emphasize the requested coordinate, settle status, executed actions, and final compact state instead of returning the whole state tree.

- Shared payload compaction
- Added dedicated compactors for action results, room rewards, campfire, shop, travel, and combat sequences instead of relying only on a generic global compactor.
- Added payload-section deduplication to suppress repeated copies of the same summarized data.
- Added inline text compaction helpers for dense decision surfaces such as smith previews.

### Practical Result

- Smith unresolved responses dropped from roughly `7.3k` characters to about `1.0k` while still carrying upgrade decision context.
- Campfire responses now expose upgrade choices in a format that is short enough for agents but still specific enough to choose the correct upgrade.
- Mixed combat turns can be executed with fewer tool calls and lower token waste.

### Upgrade Notes

- Restart the game after replacing `sts2-bridge.dll`, otherwise the new bridge payload fields will not exist in `get_state`.
- Restart the long-lived `sts2` MCP process after updating `packages/mcp-server/index.js`, otherwise tool responses will still use the old compaction logic.

### Release Assets

- `sts2-bridge-v0.7.10.dll`
- `sts2-bridge-v0.7.10.zip`
