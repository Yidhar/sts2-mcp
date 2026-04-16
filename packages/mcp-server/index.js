"use strict";

const fs = require("fs");
const os = require("os");
const path = require("path");
const { setTimeout: delay } = require("timers/promises");

const SERVER_NAME = "sts2";
const SERVER_VERSION = "0.4.20";
const FALLBACK_PROTOCOL_VERSION = "2025-03-26";
const DEFAULT_LOG_FILE_NAME = "mcp-stdio.log";
const MAX_LOG_PREVIEW = 600;
const DEFAULT_HTTP_TIMEOUT_MS = 10000;
const DEFAULT_ACTION_WAIT_MS = 0;
const MAX_ACTION_WAIT_MS = 5000;
const DEFAULT_WAIT_TIMEOUT_MS = 20000;
const MAX_WAIT_TIMEOUT_MS = 120000;
const DEFAULT_POLL_INTERVAL_MS = 250;
const MIN_POLL_INTERVAL_MS = 100;
const MAX_POLL_INTERVAL_MS = 2000;
const BRIDGE_EVENT_RECONNECT_DELAY_MS = 300;
const BRIDGE_EVENT_MAX_RECONNECT_DELAY_MS = 3000;
const BRIDGE_EVENT_CACHE_MAX_AGE_MS = 1500;
const ACTION_STATE_SYNC_TIMEOUT_MS = 500;
const POST_ACTION_SETTLE_QUIET_WINDOW_MS = 80;
const COMBAT_ACTION_SETTLE_QUIET_WINDOW_MS = 200;
const END_TURN_SETTLE_QUIET_WINDOW_MS = 120;
const END_TURN_SETTLE_TIMEOUT_MS = 8000;
const END_TURN_SETTLE_POLL_INTERVAL_MS = 200;
const END_TURN_STABLE_POLL_TARGET = 6;
const COMBAT_ACTION_SETTLE_TIMEOUT_MS = 5000;
const COMBAT_ACTION_SETTLE_POLL_INTERVAL_MS = 200;
const COMBAT_ACTION_STABLE_POLL_TARGET = 3;
const END_TURN_BATCH_WINDOW_MS = 150;
const ACTION_STATE_VERSION_RETRY_LIMIT = 2;
const ROOM_EXIT_SETTLE_TIMEOUT_MS = 5000;
const ROOM_EXIT_SETTLE_POLL_INTERVAL_MS = 200;
const MAP_ROUTE_SETTLE_TIMEOUT_MS = 4000;
const MAP_ROUTE_SETTLE_POLL_INTERVAL_MS = 200;
const MAP_ROUTE_STABLE_POLL_TARGET = 2;
const RL_ENV_API_VERSION = "0.1.0";
const RL_ENV_ACTION_ENCODING_ENUM = ["legal_action_idx", "action_id"];
const DEFAULT_RL_RESET_TIMEOUT_MS = 45000;
const DEFAULT_RL_STEP_TIMEOUT_MS = 20000;
const RL_ENV_MAX_RESET_TRANSITIONS = 12;
const RL_ENV_STABILITY_POLL_TIMEOUT_MS = 300;
const RL_ENV_STABILITY_TARGET = 2;
const RL_ENV_SURFACE_TYPES = [
  "combat",
  "reward",
  "card_reward",
  "card_selection",
  "rest_site",
  "deck_upgrade",
  "shop",
  "event",
  "event_crystal_sphere",
  "map",
  "main_menu",
  "run_mode_selection",
  "character_selection",
  "terminal",
  "unknown"
];
const RL_REWARD_WEIGHTS = Object.freeze({
  hp_delta_normalized: 1.0,
  gold_delta: 0.005,
  floor_delta: 0.25,
  relic_delta: 0.5,
  deck_delta: 0.05,
  combat_win: 0.5,
  death: -2.0,
  victory: 3.0
});
const KNOWLEDGE_INPUT_TOPIC_ENUM = [
  "route-planning",
  "deck-building",
  "card-tier-list",
  "combat-tips",
  "boss-guide",
  "enemy-patterns",
  "relics",
  "events",
  "knowledge-authoring",
  "regent",
  "routes",
  "decks",
  "cards",
  "combat",
  "bosses",
  "enemies",
  "templates",
  "authoring"
];
const TOOL_PROFILE_NAMES = ["minimal", "strategic", "debug"];
const OBSERVATION_DOMAIN_ENUM = ["cards", "relics", "events", "enemies"];
const ENEMY_METADATA_PATH = path.resolve(__dirname, "..", "rl-agent", "content", "enemies.static.generated.json");
let cachedEnemyMetadataRegistry;

const TOOL_DEFINITIONS = [
  {
    name: "sts2_get_bridge_status",
    description:
      "Read the Slay the Spire 2 bridge session file and verify the current /health endpoint.",
    inputSchema: {
      type: "object",
      properties: {},
      additionalProperties: false
    }
  },
  {
    name: "sts2_get_state",
    description:
      "Fetch the current visible bridge state, including screen, combat state, map state, rewards, and legal actions.",
    inputSchema: {
      type: "object",
      properties: {},
      additionalProperties: false
    }
  },
  {
    name: "sts2_list_actions",
    description:
      "Fetch the currently legal bridge actions without repeating the full state payload.",
    inputSchema: {
      type: "object",
      properties: {},
      additionalProperties: false
    }
  },
  {
    name: "sts2_get_map_routes",
    description:
      "Build a pruned future-only map route forest from the currently travelable frontier points, excluding the current node, past rows, and unreachable branches. Defaults to summary mode; use detail=full to include the full reachable node table. Strategy tip: prefer routes with more shops and campfires; elite fights have very low reward-to-risk ratio — avoid or minimize them.",
    inputSchema: {
      type: "object",
      properties: {
        detail: {
          type: "string",
          enum: ["summary", "full"]
        }
      },
      additionalProperties: false
    }
  },
  {
    name: "sts2_get_deck",
    description:
      "Fetch the current player's full master deck as a low-frequency strategic view, including grouped counts and per-card summaries for shop, upgrade, and boss-path planning.",
    inputSchema: {
      type: "object",
      properties: {},
      additionalProperties: false
    }
  },
  {
    name: "sts2_perform_action",
    description:
      "Execute one currently legal bridge action by action_id. Avoid parallel combat play-card calls; use sts2_play_card_sequence or sts2_execute_combat_sequence for consecutive combat actions. Set return_state_after=true to include a compact post-action state summary in the tool response.",
    inputSchema: {
      type: "object",
      properties: {
        action_id: {
          type: "string",
          minLength: 1
        },
        expected_state_version: {
          type: "integer"
        },
        strict: {
          type: "boolean"
        },
        wait_after_ms: {
          type: "integer",
          minimum: 0,
          maximum: MAX_ACTION_WAIT_MS
        },
        return_state_after: {
          type: "boolean"
        }
      },
      required: ["action_id"],
      additionalProperties: false
    }
  },
  {
    name: "sts2_play_card_sequence",
    description:
      "Execute multiple currently planned play_card actions in one tool call. Later steps are automatically rematched against the current hand and legal targets after each card resolves, so the caller does not need to manually rewrite hand indices after reordering or draws. Do not parallelize consecutive combat card plays with sts2_perform_action. Set return_state_after=true to include a compact post-sequence combat summary in the tool response.",
    inputSchema: {
      type: "object",
      properties: {
        action_ids: {
          type: "array",
          minItems: 1,
          items: {
            type: "string",
            minLength: 1
          }
        },
        expected_state_version: {
          type: "integer"
        },
        strict: {
          type: "boolean"
        },
        wait_after_ms: {
          type: "integer",
          minimum: 0,
          maximum: MAX_ACTION_WAIT_MS
        },
        return_state_after: {
          type: "boolean"
        }
      },
      required: ["action_ids"],
      additionalProperties: false
    }
  },
  {
    name: "sts2_execute_combat_sequence",
    description:
      "Execute a mixed combat sequence in one tool call. Supports play_card, use_potion, and end_turn actions, automatically rematching later play_card/use_potion steps after reindexing or draw changes. If end_turn is included anywhere in action_ids, it is always executed last. Set return_state_after=true to include a compact post-sequence combat summary in the tool response.",
    inputSchema: {
      type: "object",
      properties: {
        action_ids: {
          type: "array",
          minItems: 1,
          items: {
            type: "string",
            minLength: 1
          }
        },
        expected_state_version: {
          type: "integer"
        },
        strict: {
          type: "boolean"
        },
        wait_after_ms: {
          type: "integer",
          minimum: 0,
          maximum: MAX_ACTION_WAIT_MS
        },
        return_state_after: {
          type: "boolean"
        }
      },
      required: ["action_ids"],
      additionalProperties: false
    }
  },
  {
    name: "sts2_end_turn",
    description:
      "Convenience wrapper around sts2_perform_action for the combat end_turn action. Set return_state_after=true to include a compact post-action state summary in the tool response. Before ending: check for unplayed 0-cost cards and unused potions.",
    inputSchema: {
      type: "object",
      properties: {
        expected_state_version: {
          type: "integer"
        },
        strict: {
          type: "boolean"
        },
        wait_after_ms: {
          type: "integer",
          minimum: 0,
          maximum: MAX_ACTION_WAIT_MS
        },
        return_state_after: {
          type: "boolean"
        }
      },
      additionalProperties: false
    }
  },
  {
    name: "sts2_resolve_room_rewards",
    description:
      "Inspect the current room-end rewards, optionally choose one card reward by index, claim safe rewards, and auto-advance through reward/proceed cleanup when possible. Strategy tip: consider your current deck archetype before picking card rewards; choose cards that synergize with your build.",
    inputSchema: {
      type: "object",
      properties: {
        pick_card_index: {
          type: "integer",
          minimum: 0
        },
        skip_card_reward: {
          type: "boolean"
        },
        claim_all_safe_rewards: {
          type: "boolean"
        },
        take_potions: {
          type: "boolean"
        },
        auto_proceed: {
          type: "boolean"
        }
      },
      additionalProperties: false
    }
  },
  {
    name: "sts2_resolve_rest_site",
    description:
      "Resolve a campfire in one tool call: choose rest/smith, optionally choose a smith upgrade target, then auto-advance back to the map when possible. Strategy tip: rest if HP < 50%; consider smithing if HP > 70% and you have a key card to upgrade.",
    inputSchema: {
      type: "object",
      properties: {
        option_index: {
          type: "integer",
          minimum: 0
        },
        upgrade_card_index: {
          type: "integer",
          minimum: 0
        },
        auto_proceed: {
          type: "boolean"
        }
      },
      required: ["option_index"],
      additionalProperties: false
    }
  },
  {
    name: "sts2_resolve_card_selection",
    description:
      "Resolve the currently visible card-selection flow in one tool call: optionally choose one or more current option indices, then confirm/cancel/skip/close without one MCP request per selected card.",
    inputSchema: {
      type: "object",
      properties: {
        select_indices: {
          type: "array",
          items: {
            type: "integer",
            minimum: 0
          }
        },
        terminal_action: {
          type: "string",
          enum: ["confirm", "cancel", "skip", "close", "none"]
        },
        expected_min_select: {
          type: "integer",
          minimum: 0
        },
        expected_max_select: {
          type: "integer",
          minimum: 0
        }
      },
      additionalProperties: false
    }
  },
  {
    name: "sts2_pick_option",
    description:
      "Pick a visible indexed option without depending on raw action-id formats. Supports reward, card reward, rest site, deck upgrade, event, and generic card-selection surfaces.",
    inputSchema: {
      type: "object",
      properties: {
        index: {
          type: "integer",
          minimum: 0
        },
        surface: {
          type: "string",
          enum: [
            "auto",
            "reward",
            "card_reward",
            "rest_site",
            "deck_upgrade",
            "event",
            "card_selection"
          ]
        },
        terminal_action: {
          type: "string",
          enum: ["confirm", "cancel", "skip", "close", "none"]
        }
      },
      required: ["index"],
      additionalProperties: false
    }
  },
  {
    name: "sts2_travel_to_coordinate",
    description:
      "Travel to a reachable map coordinate after automatically absorbing non-decision reward/rest-site cleanup and waiting for a stable map snapshot.",
    inputSchema: {
      type: "object",
      properties: {
        col: {
          type: "integer",
          minimum: 0
        },
        row: {
          type: "integer",
          minimum: 0
        },
        wait_after_ms: {
          type: "integer",
          minimum: 0,
          maximum: MAX_ACTION_WAIT_MS
        }
      },
      required: ["col", "row"],
      additionalProperties: false
    }
  },
  {
    name: "sts2_resolve_shop_visit",
    description:
      "Resolve a shop visit in one tool call: optionally open the merchant, buy one or more planned shop items with automatic reindexing between purchases, optionally remove one card after buying card removal, then close and/or leave the shop.",
    inputSchema: {
      type: "object",
      properties: {
        purchases: {
          type: "array",
          items: {
            type: "object",
            properties: {
              action_id: {
                type: "string",
                minLength: 1
              },
              title: {
                type: "string",
                minLength: 1
              },
              item_kind: {
                type: "string",
                enum: ["card", "relic", "potion", "card_removal"]
              }
            },
            additionalProperties: false
          }
        },
        remove_card_title: {
          type: "string",
          minLength: 1
        },
        remove_card_index: {
          type: "integer",
          minimum: 0
        },
        open_shop: {
          type: "boolean"
        },
        close_inventory: {
          type: "boolean"
        },
        leave_shop: {
          type: "boolean"
        },
        wait_after_ms: {
          type: "integer",
          minimum: 0,
          maximum: MAX_ACTION_WAIT_MS
        }
      },
      additionalProperties: false
    }
  },
  {
    name: "sts2_wait_for_change",
    description:
      "Wait until the bridge publishes a new state_version or state_hash beyond a known baseline.",
    inputSchema: {
      type: "object",
      properties: {
        baseline_state_version: {
          type: "integer"
        },
        baseline_state_hash: {
          type: "string"
        },
        timeout_ms: {
          type: "integer",
          minimum: 1,
          maximum: MAX_WAIT_TIMEOUT_MS
        },
        poll_interval_ms: {
          type: "integer",
          minimum: MIN_POLL_INTERVAL_MS,
          maximum: MAX_POLL_INTERVAL_MS
        }
      },
      additionalProperties: false
    }
  },
  {
    name: "sts2_wait_until_actionable",
    description:
      "Wait until a non-automation decision surface is stable enough to act on. This follows the bridge frontier stream and returns when a stable actionable surface such as combat, reward, card selection, rest site, shop, map, or event becomes available.",
    inputSchema: {
      type: "object",
      properties: {
        timeout_ms: {
          type: "integer",
          minimum: 1,
          maximum: MAX_WAIT_TIMEOUT_MS
        },
        poll_interval_ms: {
          type: "integer",
          minimum: MIN_POLL_INTERVAL_MS,
          maximum: MAX_POLL_INTERVAL_MS
        },
        stability_polls: {
          type: "integer",
          minimum: 1,
          maximum: 10,
          description:
            "How many consecutive identical actionable snapshots are required before returning. Defaults to 2."
        }
      },
      additionalProperties: false
    }
  },
  {
    name: "sts2_env_spec",
    description:
      "Describe the compact RL-oriented single-environment contract exposed by this branch, including observation layout, legal-action encoding, reward shaping, and reset limitations.",
    inputSchema: {
      type: "object",
      properties: {},
      additionalProperties: false
    }
  },
  {
    name: "sts2_env_reset",
    description:
      "Start a fresh RL episode from the main menu using the standard run flow, optionally selecting a character, then return the first stable actionable observation plus legal actions.",
    inputSchema: {
      type: "object",
      properties: {
        character: {
          type: "string",
          minLength: 1
        },
        defensive_buffs: {
          type: "boolean",
          description:
            "When true, the episode applies RL-only defensive buffs such as full heal plus heavy combat plating/block maintenance."
        },
        timeout_ms: {
          type: "integer",
          minimum: 1,
          maximum: MAX_WAIT_TIMEOUT_MS
        }
      },
      additionalProperties: false
    }
  },
  {
    name: "sts2_env_step",
    description:
      "Execute one RL environment step against the active episode, automatically waiting until the next stable actionable or terminal state, then return obs, reward, done, truncated, and legal actions.",
    inputSchema: {
      type: "object",
      properties: {
        episode_id: {
          type: "string",
          minLength: 1
        },
        action_encoding: {
          type: "string",
          enum: RL_ENV_ACTION_ENCODING_ENUM
        },
        action_index: {
          type: "integer",
          minimum: 0
        },
        action_id: {
          type: "string",
          minLength: 1
        },
        timeout_ms: {
          type: "integer",
          minimum: 1,
          maximum: MAX_WAIT_TIMEOUT_MS
        }
      },
      required: ["episode_id"],
      additionalProperties: false
    }
  },
  {
    name: "sts2_journal_write",
    description:
      "Append a markdown journal entry to the current run's log file. Use this to record combat outcomes, route decisions, card evaluations, and post-mortem notes. Each entry is persisted as a markdown section with floor, tags, and timestamp.",
    inputSchema: {
      type: "object",
      properties: {
        entry: {
          type: "string",
          minLength: 1,
          description: "The journal entry text (markdown supported)."
        },
        tags: {
          type: "array",
          items: { type: "string" },
          description: "Optional tags like 'combat', 'death', 'route', 'upgrade', 'shop'."
        },
        floor: {
          type: "integer",
          minimum: 0,
          description: "Floor number for this entry. Auto-detected from live bridge state when omitted."
        }
      },
      required: ["entry"],
      additionalProperties: false
    }
  },
  {
    name: "sts2_journal_read",
    description:
      "Read the current run's journal log. Returns the full markdown content or a filtered subset. Use this at the start of a new conversation to recall what happened in previous floors.",
    inputSchema: {
      type: "object",
      properties: {
        last_n: {
          type: "integer",
          minimum: 1,
          description: "Return only the last N entries."
        },
        tags: {
          type: "array",
          items: { type: "string" },
          description: "Filter entries that contain any of these tags."
        }
      },
      additionalProperties: false
    }
  },
  {
    name: "sts2_journal_summarize",
    description:
      "Write or update the current run's summary in meta.json. Call this after significant milestones (act completion, death, boss win) to persist a concise run overview.",
    inputSchema: {
      type: "object",
      properties: {
        summary: {
          type: "string",
          minLength: 1,
          description: "The run summary text."
        },
        result: {
          type: "string",
          enum: ["in_progress", "death", "victory"],
          description: "Run result status."
        },
        floor: {
          type: "integer",
          minimum: 0,
          description: "Current or final floor number."
        }
      },
      required: ["summary"],
      additionalProperties: false
    }
  },
  {
    name: "sts2_journal_get_summary",
    description:
      "Read the current run's summary and metadata from meta.json. Use this at conversation start to quickly recall the current run state.",
    inputSchema: {
      type: "object",
      properties: {},
      additionalProperties: false
    }
  },
  {
    name: "sts2_journal_list_runs",
    description:
      "List summaries of all historical runs from meta.json. Use this to review past run outcomes and learn from mistakes.",
    inputSchema: {
      type: "object",
      properties: {
        last_n: {
          type: "integer",
          minimum: 1,
          description: "Return only the last N runs."
        }
      },
      additionalProperties: false
    }
  },
  {
    name: "sts2_record_observation",
    description:
      "Append one provenance-tagged observation for a knowledge entity. Use this to accumulate card, relic, event, or enemy evidence before promoting it into canonical knowledge.",
    inputSchema: {
      type: "object",
      properties: {
        domain: {
          type: "string",
          enum: OBSERVATION_DOMAIN_ENUM
        },
        entity_name: {
          type: "string",
          minLength: 1
        },
        observation: {
          type: "string",
          minLength: 1
        },
        source_type: {
          type: "string",
          enum: ["observed", "journaled", "inferred", "external"]
        },
        confidence: {
          type: "string",
          enum: ["low", "medium", "high"]
        },
        source_tool: {
          type: "string",
          minLength: 1
        },
        source_ref: {
          type: "string",
          minLength: 1
        },
        state_version: {
          type: "integer"
        },
        tags: {
          type: "array",
          items: {
            type: "string"
          }
        }
      },
      required: ["domain", "entity_name", "observation"],
      additionalProperties: false
    }
  },
  {
    name: "sts2_list_observation_entities",
    description:
      "List recorded observation entities with counts and freshness. Use this before reading one entity log in detail.",
    inputSchema: {
      type: "object",
      properties: {
        domain: {
          type: "string",
          enum: OBSERVATION_DOMAIN_ENUM
        },
        query: {
          type: "string",
          minLength: 1
        },
        max_results: {
          type: "integer",
          minimum: 1,
          maximum: 200
        }
      },
      additionalProperties: false
    }
  },
  {
    name: "sts2_read_observation_entity",
    description:
      "Read the observation log for one entity. Returns the full markdown log or only the last N entries.",
    inputSchema: {
      type: "object",
      properties: {
        domain: {
          type: "string",
          enum: OBSERVATION_DOMAIN_ENUM
        },
        entity_name: {
          type: "string",
          minLength: 1
        },
        last_n: {
          type: "integer",
          minimum: 1
        }
      },
      required: ["domain", "entity_name"],
      additionalProperties: false
    }
  },
  {
    name: "sts2_get_knowledge",
    description:
      "Read a strategy knowledge file by topic. Canonical topics: route-planning, deck-building, card-tier-list, combat-tips, boss-guide, enemy-patterns, relics, events, knowledge-authoring. Aliases include regent, routes, decks, cards, combat, bosses, enemies, templates, and authoring. Returns the full markdown content of the requested knowledge file.",
    inputSchema: {
      type: "object",
      properties: {
        topic: {
          type: "string",
          enum: KNOWLEDGE_INPUT_TOPIC_ENUM,
          description: "The knowledge topic to retrieve."
        }
      },
      required: ["topic"],
      additionalProperties: false
    }
  },
  {
    name: "sts2_get_knowledge_topics",
    description:
      "List available knowledge topics with aliases, domains, file paths, section counts, and short summaries. Use this before search/read tools when you want a precise entry point into the knowledge base.",
    inputSchema: {
      type: "object",
      properties: {},
      additionalProperties: false
    }
  },
  {
    name: "sts2_search_knowledge",
    description:
      "Search the knowledge base by keyword or regex. Returns matching lines with heading path and small context windows so callers can inspect precisely before reading a larger slice.",
    inputSchema: {
      type: "object",
      properties: {
        query: {
          type: "string",
          minLength: 1,
          description: "Keyword or regex pattern to search for."
        },
        topics: {
          type: "array",
          items: {
            type: "string",
            enum: KNOWLEDGE_INPUT_TOPIC_ENUM
          },
          description: "Optional topic filter. When omitted, search all knowledge topics."
        },
        case_sensitive: {
          type: "boolean",
          description: "Use case-sensitive matching. Defaults to false."
        },
        regex: {
          type: "boolean",
          description: "Treat query as a JavaScript regular expression. Defaults to false."
        },
        context_before: {
          type: "integer",
          minimum: 0,
          maximum: 20,
          description: "How many lines of context to include before each match. Defaults to 1."
        },
        context_after: {
          type: "integer",
          minimum: 0,
          maximum: 20,
          description: "How many lines of context to include after each match. Defaults to 1."
        },
        max_results: {
          type: "integer",
          minimum: 1,
          maximum: 100,
          description: "Maximum number of matches to return. Defaults to 20."
        }
      },
      required: ["query"],
      additionalProperties: false
    }
  },
  {
    name: "sts2_read_knowledge_slice",
    description:
      "Read a precise slice from one knowledge topic. Supports an exact section_path array, a heading match, or an explicit line range, with optional max_lines truncation.",
    inputSchema: {
      type: "object",
      properties: {
        topic: {
          type: "string",
          enum: KNOWLEDGE_INPUT_TOPIC_ENUM,
          description: "The knowledge topic to read from."
        },
        section_path: {
          type: "array",
          items: { type: "string" },
          minItems: 1,
          description:
            "Exact heading path array to read, for example ['Act 1', 'Guardian']. Cannot be combined with heading/start_line/end_line."
        },
        heading: {
          type: "string",
          description:
            "Exact section title or full heading path string to read. Cannot be combined with section_path/start_line/end_line."
        },
        occurrence: {
          type: "integer",
          minimum: 1,
          maximum: 50,
          description: "Which matching heading occurrence to read when heading is used. Defaults to 1."
        },
        case_sensitive: {
          type: "boolean",
          description: "Use case-sensitive heading matching. Defaults to false."
        },
        start_line: {
          type: "integer",
          minimum: 1,
          description: "1-based start line for line-range reads."
        },
        end_line: {
          type: "integer",
          minimum: 1,
          description: "1-based end line for line-range reads. Defaults to start_line when omitted."
        },
        max_lines: {
          type: "integer",
          minimum: 1,
          maximum: 400,
          description:
            "Maximum number of lines to return from the matched section or requested line range. Defaults to the full selected range."
        }
      },
      required: ["topic"],
      additionalProperties: false
    }
  },
  {
    name: "sts2_list_knowledge_sections",
    description:
      "List headings for one or more knowledge topics, including heading path and line ranges. Useful before reading a narrow slice.",
    inputSchema: {
      type: "object",
      properties: {
        topics: {
          type: "array",
          items: {
            type: "string",
            enum: KNOWLEDGE_INPUT_TOPIC_ENUM
          },
          description: "Optional topic filter. When omitted, list sections for all knowledge topics."
        },
        max_sections_per_topic: {
          type: "integer",
          minimum: 1,
          maximum: 200,
          description: "Maximum number of sections to return per topic. Defaults to 200."
        }
      },
      additionalProperties: false
    }
  }
];

const TOOL_PROFILE_TOOL_NAMES = {
  minimal: [
    "sts2_get_bridge_status",
    "sts2_get_state",
    "sts2_list_actions",
    "sts2_perform_action",
    "sts2_end_turn",
    "sts2_pick_option",
    "sts2_resolve_room_rewards",
    "sts2_resolve_rest_site",
    "sts2_resolve_card_selection",
    "sts2_resolve_shop_visit",
    "sts2_travel_to_coordinate",
    "sts2_wait_for_change",
    "sts2_wait_until_actionable"
  ],
  strategic: [
    "sts2_get_bridge_status",
    "sts2_get_state",
    "sts2_list_actions",
    "sts2_env_spec",
    "sts2_env_reset",
    "sts2_env_step",
    "sts2_get_map_routes",
    "sts2_get_deck",
    "sts2_perform_action",
    "sts2_play_card_sequence",
    "sts2_execute_combat_sequence",
    "sts2_end_turn",
    "sts2_resolve_room_rewards",
    "sts2_resolve_rest_site",
    "sts2_resolve_card_selection",
    "sts2_pick_option",
    "sts2_travel_to_coordinate",
    "sts2_resolve_shop_visit",
    "sts2_wait_for_change",
    "sts2_wait_until_actionable",
    "sts2_record_observation",
    "sts2_list_observation_entities",
    "sts2_read_observation_entity",
    "sts2_get_knowledge",
    "sts2_get_knowledge_topics",
    "sts2_search_knowledge",
    "sts2_read_knowledge_slice",
    "sts2_list_knowledge_sections"
  ],
  debug: TOOL_DEFINITIONS.map((tool) => tool.name)
};
const ACTIVE_TOOL_PROFILE_NAME = resolveActiveToolProfileName();
const ACTIVE_TOOL_NAME_SET = getAllowedToolNameSetForProfile(ACTIVE_TOOL_PROFILE_NAME);

let incomingTextBuffer = "";
let processingChain = Promise.resolve();
let loggedFirstStdinChunk = false;
let pendingMessageQueue = [];
let activeBridgeEventClient = null;
let rlEpisodeCounter = 0;
let activeRlEpisode = null;

logInfo(
  `process started pid=${process.pid} node=${process.version} cwd=${process.cwd()}`
);
logInfo(
  `tool_profile=${ACTIVE_TOOL_PROFILE_NAME} exposed_tools=${getExposedToolDefinitions().length}`
);
logDebug(`argv=${safeJson(process.argv)}`);
logDebug(`session_file_path=${getSessionFilePath()}`);
logDebug(`log_file_path=${getLogFilePath()}`);
logDebug("transport=stdio newline-delimited json-rpc");

process.stdin.setEncoding("utf8");

process.stdin.on("data", (chunk) => {
  if (!chunk) {
    return;
  }

  const chunkSize = Buffer.byteLength(chunk, "utf8");
  if (!loggedFirstStdinChunk) {
    logDebug(`first_stdin_chunk_bytes=${chunkSize}`);
    logDebug(`first_stdin_chunk_preview=${sanitizeForLog(chunk)}`);
    loggedFirstStdinChunk = true;
  } else {
    logDebug(`stdin_chunk_bytes=${chunkSize}`);
  }

  incomingTextBuffer += chunk;
  processingChain = processingChain
    .then(() => processIncomingMessages())
    .catch((error) => {
      logError("failed to process incoming MCP message", error);
    });
});

process.stdin.on("end", () => {
  logInfo("stdin ended");
  activeBridgeEventClient?.close();
  process.exit(0);
});

process.stdin.on("error", (error) => {
  logError("stdin error", error);
});

process.on("uncaughtException", (error) => {
  logError("uncaught exception", error);
  activeBridgeEventClient?.close();
  process.exit(1);
});

process.on("unhandledRejection", (reason) => {
  logError("unhandled rejection", reason);
  activeBridgeEventClient?.close();
  process.exit(1);
});

process.on("exit", () => {
  activeBridgeEventClient?.close();
});

async function processIncomingMessages() {
  collectParsedMessagesFromBuffer();

  while (pendingMessageQueue.length > 0) {
    const message = await dequeueNextMessageForExecution();
    if (!message) {
      return;
    }

    await handleMessage(message);
    collectParsedMessagesFromBuffer();
  }
}

function collectParsedMessagesFromBuffer() {
  while (true) {
    const newlineIndex = incomingTextBuffer.indexOf("\n");
    if (newlineIndex === -1) {
      return;
    }

    let rawLine = incomingTextBuffer.slice(0, newlineIndex);
    incomingTextBuffer = incomingTextBuffer.slice(newlineIndex + 1);

    if (rawLine.endsWith("\r")) {
      rawLine = rawLine.slice(0, -1);
    }

    if (!rawLine.trim()) {
      continue;
    }

    logDebug(`stdin_line=${sanitizeForLog(rawLine)}`);

    let message;
    try {
      message = JSON.parse(rawLine);
    } catch (error) {
      logError("failed to parse MCP JSON payload", error);
      continue;
    }

    if (Array.isArray(message)) {
      logDebug(`parsed_json_batch length=${message.length}`);
      pendingMessageQueue.push(...message);
      continue;
    }

    pendingMessageQueue.push(message);
  }
}

async function dequeueNextMessageForExecution() {
  const nextNonEndTurnIndex = findNextNonEndTurnMessageIndex();
  if (nextNonEndTurnIndex >= 0) {
    return pendingMessageQueue.splice(nextNonEndTurnIndex, 1)[0] ?? null;
  }

  if (pendingMessageQueue.length <= 0) {
    return null;
  }

  await delay(END_TURN_BATCH_WINDOW_MS);
  collectParsedMessagesFromBuffer();

  const delayedNonEndTurnIndex = findNextNonEndTurnMessageIndex();
  if (delayedNonEndTurnIndex >= 0) {
    return pendingMessageQueue.splice(delayedNonEndTurnIndex, 1)[0] ?? null;
  }

  return pendingMessageQueue.shift() ?? null;
}

function findNextNonEndTurnMessageIndex() {
  return pendingMessageQueue.findIndex((message) => !isEndTurnToolCallMessage(message));
}

function isEndTurnToolCallMessage(message) {
  const toolName = getToolCallName(message);
  if (toolName === "sts2_end_turn") {
    return true;
  }

  return toolName === "sts2_perform_action" && getToolCallActionId(message) === "end_turn";
}

function getToolCallName(message) {
  return typeof message?.params?.name === "string" ? message.params.name : "";
}

function getToolCallActionId(message) {
  return typeof message?.params?.arguments?.action_id === "string"
    ? message.params.arguments.action_id
    : null;
}

async function handleMessage(message) {
  if (!message || typeof message !== "object") {
    logDebug("ignoring non-object MCP message");
    return;
  }

  const hasId = Object.prototype.hasOwnProperty.call(message, "id");
  const method =
    typeof message.method === "string" && message.method.length > 0 ? message.method : null;

  if (!hasId) {
    if (method) {
      logDebug(`received_notification method=${method}`);
    } else {
      logDebug("received message without id or method; ignoring");
    }
    return;
  }

  if (!method) {
    logDebug(`received message without method id=${safeJson(message.id)}; ignoring`);
    return;
  }

  logDebug(`received_request method=${method} id=${safeJson(message.id)}`);

  try {
    switch (method) {
      case "initialize":
        return writeResult(message.id, {
          protocolVersion:
            message.params && typeof message.params.protocolVersion === "string"
              ? message.params.protocolVersion
              : FALLBACK_PROTOCOL_VERSION,
          capabilities: {
            tools: {
              listChanged: false
            }
          },
          serverInfo: {
            name: SERVER_NAME,
            version: SERVER_VERSION
          },
          instructions:
            `Current tool profile: ${ACTIVE_TOOL_PROFILE_NAME}. Use sts2_get_state and sts2_list_actions to observe the game, then execute only legal actions exposed by tools/list. Prefer higher-level sequence and resolver tools when this profile exposes them.`
        });
      case "ping":
        return writeResult(message.id, {});
      case "tools/list":
        return writeResult(message.id, {
          tools: getExposedToolDefinitions()
        });
      case "tools/call":
        return writeResult(message.id, await handleToolCall(message.params || {}));
      default:
        return writeError(message.id, -32601, `Method not found: ${method}`);
    }
  } catch (error) {
    logError(`request failed for method ${method}`, error);
    return writeError(message.id, -32603, error instanceof Error ? error.message : String(error));
  }
}

async function handleToolCall(params) {
  const toolName = typeof params.name === "string" ? params.name : "";
  const args = isPlainObject(params.arguments) ? params.arguments : {};

  if (!toolName || !ACTIVE_TOOL_NAME_SET.has(toolName)) {
    return asToolResult(
      {
        ok: false,
        error: "tool_unavailable_in_profile",
        tool: toolName || null,
        active_profile: ACTIVE_TOOL_PROFILE_NAME,
        available_tools: getExposedToolDefinitions().map((tool) => tool.name)
      },
      true
    );
  }

  switch (toolName) {
    case "sts2_get_bridge_status":
      return await getBridgeStatus();
    case "sts2_get_state":
      return await getStateTool();
    case "sts2_get_deck":
      return await getDeckTool();
    case "sts2_list_actions":
      return await listActionsTool();
    case "sts2_get_map_routes":
      return await getMapRoutesTool(args);
    case "sts2_perform_action":
      return await performActionTool(args);
    case "sts2_play_card_sequence":
      return await playCardSequenceTool(args);
    case "sts2_execute_combat_sequence":
      return await executeCombatSequenceTool(args);
    case "sts2_end_turn":
      return await endTurnTool(args);
    case "sts2_resolve_room_rewards":
      return await resolveRoomRewardsTool(args);
    case "sts2_resolve_rest_site":
      return await resolveRestSiteTool(args);
    case "sts2_resolve_card_selection":
      return await resolveCardSelectionTool(args);
    case "sts2_pick_option":
      return await pickOptionTool(args);
    case "sts2_travel_to_coordinate":
      return await travelToCoordinateTool(args);
    case "sts2_resolve_shop_visit":
      return await resolveShopVisitTool(args);
    case "sts2_wait_for_change":
      return await waitForChangeTool(args);
    case "sts2_wait_until_actionable":
      return await waitUntilActionableTool(args);
    case "sts2_env_spec":
      return await envSpecTool();
    case "sts2_env_reset":
      return await envResetTool(args);
    case "sts2_env_step":
      return await envStepTool(args);
    case "sts2_journal_write":
      return await journalWriteTool(args);
    case "sts2_journal_read":
      return await journalReadTool(args);
    case "sts2_journal_summarize":
      return await journalSummarizeTool(args);
    case "sts2_journal_get_summary":
      return await journalGetSummaryTool(args);
    case "sts2_journal_list_runs":
      return await journalListRunsTool(args);
    case "sts2_record_observation":
      return await recordObservationTool(args);
    case "sts2_list_observation_entities":
      return await listObservationEntitiesTool(args);
    case "sts2_read_observation_entity":
      return await readObservationEntityTool(args);
    case "sts2_get_knowledge":
      return await getKnowledgeTool(args);
    case "sts2_get_knowledge_topics":
      return await getKnowledgeTopicsTool(args);
    case "sts2_search_knowledge":
      return await searchKnowledgeTool(args);
    case "sts2_read_knowledge_slice":
      return await readKnowledgeSliceTool(args);
    case "sts2_list_knowledge_sections":
      return await listKnowledgeSectionsTool(args);
    default:
      return asToolResult(
        {
          ok: false,
          error: "unknown_tool",
          tool: toolName
        },
        true
      );
  }
}

function normalizeToolProfileName(value) {
  if (typeof value !== "string" || !value.trim()) {
    return null;
  }

  const normalized = value.trim().toLowerCase();
  if (TOOL_PROFILE_NAMES.includes(normalized)) {
    return normalized;
  }
  if (normalized === "full" || normalized === "default") {
    return "debug";
  }

  return null;
}

function resolveActiveToolProfileName() {
  const cliArgs = Array.isArray(process.argv) ? process.argv.slice(2) : [];
  const cliProfileArg = cliArgs.find((arg) =>
    typeof arg === "string" &&
    (arg.startsWith("--profile=") || arg.startsWith("--tool-profile="))
  );
  const cliProfileValue = cliProfileArg
    ? cliProfileArg.slice(cliProfileArg.indexOf("=") + 1)
    : null;
  const envProfileValue =
    process.env.STS2_MCP_PROFILE || process.env.STS2_TOOL_PROFILE || null;

  return (
    normalizeToolProfileName(cliProfileValue) ??
    normalizeToolProfileName(envProfileValue) ??
    "debug"
  );
}

function getAllowedToolNameSetForProfile(profileName) {
  const normalizedProfile = normalizeToolProfileName(profileName) ?? "debug";
  const toolNames = TOOL_PROFILE_TOOL_NAMES[normalizedProfile] ?? TOOL_PROFILE_TOOL_NAMES.debug;
  return new Set(Array.isArray(toolNames) ? toolNames : []);
}

function getExposedToolDefinitions() {
  return TOOL_DEFINITIONS.filter((tool) => ACTIVE_TOOL_NAME_SET.has(tool.name));
}

async function getBridgeStatus() {
  const sessionFilePath = getSessionFilePath();
  const status = {
    ok: false,
    session_file_path: sessionFilePath,
    session_file_exists: false,
    pid_alive: false,
    health_ok: false
  };

  if (!fs.existsSync(sessionFilePath)) {
    status.error = "session_file_missing";
    status.message = "Bridge session file was not found.";
    return asToolResult(status, true);
  }

  status.session_file_exists = true;

  let session;
  try {
    session = readSessionFile(sessionFilePath);
  } catch (error) {
    return asToolResult(
      {
        ...status,
        error: "invalid_session_file",
        message: error instanceof Error ? error.message : String(error)
      },
      true
    );
  }

  status.session = session;
  status.pid_alive = isProcessAlive(session.pid);

  if (!status.pid_alive) {
    status.error = "stale_session";
    status.message = "Session file exists, but the recorded game process is not alive.";
    return asToolResult(status, true);
  }

  if (!session.base_url || typeof session.base_url !== "string") {
    status.error = "invalid_session_file";
    status.message = "Session file does not contain a valid base_url.";
    return asToolResult(status, true);
  }

  const healthUrl = new URL("health", ensureTrailingSlash(session.base_url)).toString();
  status.health_url = healthUrl;

  try {
    const response = await fetch(healthUrl, {
      method: "GET",
      headers: buildAuthHeaders(session),
      signal: AbortSignal.timeout(DEFAULT_HTTP_TIMEOUT_MS)
    });

    const body = await readJsonResponseBody(response);
    status.health_status = response.status;
    status.health_ok = response.ok;
    status.bridge_health = body;
    status.ok = response.ok;

    return asToolResult(status, !response.ok);
  } catch (error) {
    status.error = "health_request_failed";
    status.message = error instanceof Error ? error.message : String(error);
    return asToolResult(status, true);
  }
}

async function getStateTool() {
  try {
    const session = getLiveSession();
    const state = await getBridgeState(session);
    return asToolResult(attachInteractionHints(state), false);
  } catch (error) {
    return asToolResult(toolErrorPayload(error), true);
  }
}

async function getDeckTool() {
  try {
    const session = getLiveSession();
    const state = await getBridgeState(session);
    return asToolResult(buildDeckPayloadForAgent(state), false);
  } catch (error) {
    return asToolResult(toolErrorPayload(error), true);
  }
}

async function listActionsTool() {
  try {
    const session = getLiveSession();
    const state = await getBridgeState(session);
    return asToolResult(
      {
        ok: true,
        state_version: Number.isInteger(state?.state_version) ? state.state_version : null,
        state_hash: typeof state?.state_hash === "string" ? state.state_hash : null,
        screen: typeof state?.screen === "string" ? state.screen : null,
        actions: Array.isArray(state?.available_actions) ? state.available_actions : []
      },
      false
    );
  } catch (error) {
    return asToolResult(toolErrorPayload(error), true);
  }
}

async function getMapRoutesTool(args) {
  try {
    const detail = normalizeMapRoutesDetail(args?.detail);
    const session = getLiveSession();
    const settledSnapshot = await waitForStableMapRouteSnapshot(session);
    const payload = buildMapRoutesPayload(settledSnapshot.state, {
      detail,
      snapshot_status: {
        settled: settledSnapshot.settled,
        reason: settledSnapshot.reason,
        poll_count: settledSnapshot.poll_count,
        frontier_action_match: settledSnapshot.frontier_action_match
      }
    });
    return asToolResult(payload, false);
  } catch (error) {
    return asToolResult(toolErrorPayload(error), true);
  }
}

async function performActionTool(args) {
  try {
    const actionId = requireNonEmptyString(args.action_id, "action_id");
    const expectedStateVersion = optionalInteger(
      args.expected_state_version,
      "expected_state_version"
    );
    const strictStateVersion = optionalBoolean(args.strict, "strict") ?? false;
    const returnStateAfter = optionalBoolean(
      args.return_state_after,
      "return_state_after"
    ) ?? false;
    const waitAfterMs = clampInteger(
      args.wait_after_ms,
      DEFAULT_ACTION_WAIT_MS,
      0,
      MAX_ACTION_WAIT_MS,
      "wait_after_ms"
    );

    const session = getLiveSession();
    const result = await performBridgeAction(
      session,
      actionId,
      waitAfterMs,
      expectedStateVersion,
      {
        strictStateVersion,
        settleAfterAction: true
      }
    );

    return asPrecompactedToolResult(
      compactActionResultPayload(
        attachInteractionHints(
          attachReturnStateAfterPayload(result, returnStateAfter, result?.state)
        )
      ),
      false
    );
  } catch (error) {
    return asToolResult(toolErrorPayload(error), true);
  }
}

async function playCardSequenceTool(args) {
  return runCombatSequenceTool(args, {
    toolName: "sts2_play_card_sequence",
    sequenceLabel: "play-card sequence",
    allowedActionKinds: new Set(["play_card"])
  });
}

async function executeCombatSequenceTool(args) {
  return runCombatSequenceTool(args, {
    toolName: "sts2_execute_combat_sequence",
    sequenceLabel: "combat sequence",
    allowedActionKinds: new Set(["play_card", "use_potion", "end_turn"])
  });
}

async function endTurnTool(args) {
  try {
    const expectedStateVersion = optionalInteger(
      args.expected_state_version,
      "expected_state_version"
    );
    const strictStateVersion = optionalBoolean(args.strict, "strict") ?? false;
    const returnStateAfter = optionalBoolean(
      args.return_state_after,
      "return_state_after"
    ) ?? false;
    const waitAfterMs = clampInteger(
      args.wait_after_ms,
      DEFAULT_ACTION_WAIT_MS,
      0,
      MAX_ACTION_WAIT_MS,
      "wait_after_ms"
    );
    const session = getLiveSession();
    const result = await performBridgeAction(
      session,
      "end_turn",
      waitAfterMs,
      expectedStateVersion,
      {
        strictStateVersion,
        settleAfterAction: true
      }
    );

    return asPrecompactedToolResult(
      compactActionResultPayload(
        attachInteractionHints(
          attachReturnStateAfterPayload(result, returnStateAfter, result?.state)
        )
      ),
      false
    );
  } catch (error) {
    return asToolResult(toolErrorPayload(error), true);
  }
}

function attachReturnStateAfterPayload(payload, returnStateAfter, stateAfter) {
  if (!returnStateAfter || !isPlainObject(payload) || !isPlainObject(stateAfter)) {
    return payload;
  }

  return {
    ...payload,
    state_after: stateAfter
  };
}

async function runCombatSequenceTool(args, options = {}) {
  try {
    let initialSequenceState = null;
    const finalize = (payload) =>
      asPrecompactedToolResult(
        compactPlayCardSequencePayload(
          isPlainObject(payload) && isPlainObject(initialSequenceState)
            ? {
              ...payload,
              initial_state: initialSequenceState
            }
            : payload
        ),
        false
      );
    const actionIds = requireNonEmptyStringArray(args.action_ids, "action_ids");
    const expectedStateVersion = optionalInteger(
      args.expected_state_version,
      "expected_state_version"
    );
    const strictStateVersion = optionalBoolean(args.strict, "strict") ?? false;
    const returnStateAfter = optionalBoolean(
      args.return_state_after,
      "return_state_after"
    ) ?? false;
    const waitAfterMs = clampInteger(
      args.wait_after_ms,
      DEFAULT_ACTION_WAIT_MS,
      0,
      MAX_ACTION_WAIT_MS,
      "wait_after_ms"
    );

    if (new Set(actionIds).size !== actionIds.length) {
      throw new ToolPayloadError(
        "invalid_arguments",
        "action_ids cannot contain duplicates.",
        {
          field: "action_ids"
        }
      );
    }

    const sequenceLabel =
      typeof options.sequenceLabel === "string" && options.sequenceLabel.trim()
        ? options.sequenceLabel.trim()
        : "combat sequence";
    const allowedActionKinds =
      options.allowedActionKinds instanceof Set && options.allowedActionKinds.size > 0
        ? options.allowedActionKinds
        : new Set(["play_card", "use_potion", "end_turn"]);
    const normalizedActionIds = normalizeCombatSequenceActionIds(actionIds);
    const reorderedEndTurnToLast =
      normalizedActionIds.length === actionIds.length &&
      normalizedActionIds.some((actionId, index) => actionId !== actionIds[index]);

    const session = getLiveSession();
    let state = await getBridgeState(session);
    initialSequenceState = isPlainObject(state) ? state : null;

    const initialStateVersionAdjusted =
      Number.isInteger(expectedStateVersion) &&
        Number.isInteger(state?.state_version) &&
        state.state_version !== expectedStateVersion
        ? {
          expected_state_version: expectedStateVersion,
          observed_state_version: state.state_version
        }
        : null;

    if (strictStateVersion && initialStateVersionAdjusted) {
      return finalize(
        attachReturnStateAfterPayload(
          {
            ok: false,
            resolved: false,
            reason: "state_version_mismatch",
            ...initialStateVersionAdjusted,
            requested_action_ids: actionIds,
            normalized_action_ids: normalizedActionIds,
            reordered_end_turn_to_last: reorderedEndTurnToLast,
            state
          },
          returnStateAfter,
          state
        )
      );
    }

    const initialBlocker = getCombatSequenceContinuationBlocker(state, sequenceLabel);
    if (initialBlocker) {
      return finalize(
        attachReturnStateAfterPayload(
          {
            ok: false,
            resolved: false,
            ...initialBlocker,
            requested_action_count: normalizedActionIds.length,
            requested_action_ids: actionIds,
            normalized_action_ids: normalizedActionIds,
            reordered_end_turn_to_last: reorderedEndTurnToLast,
            state
          },
          returnStateAfter,
          state
        )
      );
    }

    const sequencePlan = [];
    for (let sequenceIndex = 0; sequenceIndex < normalizedActionIds.length; sequenceIndex += 1) {
      const requestedActionId = normalizedActionIds[sequenceIndex];
      const requestedActionResolution = resolveRequestedCombatSequenceAction(
        requestedActionId,
        state,
        allowedActionKinds
      );
      const matchedAction = requestedActionResolution.action;
      if (!matchedAction) {
        const reason = requestedActionResolution.unsupported_kind
          ? "unsupported_combat_sequence_action"
          : "requested_combat_sequence_action_unavailable";
        return finalize(
          attachReturnStateAfterPayload(
            {
              ok: false,
              resolved: false,
              reason,
              sequence_index: sequenceIndex,
              requested_action_id: requestedActionId,
              requested_action_kind: requestedActionResolution.kind,
              allowed_action_kinds: Array.from(allowedActionKinds),
              available_actions: summarizeCombatSequenceAvailableActions(
                getCombatSequenceAvailableActions(state, allowedActionKinds)
              ),
              requested_action_ids: actionIds,
              normalized_action_ids: normalizedActionIds,
              reordered_end_turn_to_last: reorderedEndTurnToLast,
              state
            },
            returnStateAfter,
            state
          )
        );
      }

      sequencePlan.push(
        buildPlannedCombatSequenceStep(matchedAction, sequenceIndex, requestedActionId)
      );
    }
    const sequencePlanOutput = sequencePlan.map(summarizeCombatSequencePlanStep);

    const executedSteps = [];

    for (let sequenceIndex = 0; sequenceIndex < sequencePlan.length; sequenceIndex += 1) {
      const blocker = getCombatSequenceContinuationBlocker(state, sequenceLabel);
      if (blocker) {
        return finalize(
          attachReturnStateAfterPayload(
            {
              ok: true,
              resolved: false,
              reason: blocker.reason,
              message: blocker.message,
              requested_action_count: sequencePlan.length,
              executed_count: executedSteps.length,
              remaining_count: sequencePlan.length - executedSteps.length,
              next_sequence_index: sequenceIndex,
              requested_action_ids: actionIds,
              normalized_action_ids: normalizedActionIds,
              reordered_end_turn_to_last: reorderedEndTurnToLast,
              sequence_plan: sequencePlanOutput,
              executed_steps: executedSteps,
              state
            },
            returnStateAfter,
            state
          )
        );
      }

      const planStep = sequencePlan[sequenceIndex];
      let resolution = resolvePlannedCombatSequenceStep(planStep, state);
      if (!resolution.action) {
        return finalize(
          attachReturnStateAfterPayload(
            {
              ok: true,
              resolved: false,
              reason: "requested_combat_sequence_action_unavailable_after_reindex",
              requested_action_count: sequencePlan.length,
              executed_count: executedSteps.length,
              remaining_count: sequencePlan.length - executedSteps.length,
              next_sequence_index: sequenceIndex,
              failed_step: summarizeCombatSequencePlanStep(planStep),
              requested_action_ids: actionIds,
              normalized_action_ids: normalizedActionIds,
              reordered_end_turn_to_last: reorderedEndTurnToLast,
              sequence_plan: sequencePlanOutput,
              executed_steps: executedSteps,
              available_actions: summarizeCombatSequenceAvailableActions(
                getCombatSequenceAvailableActions(state, allowedActionKinds)
              ),
              state
            },
            returnStateAfter,
            state
          )
        );
      }

      let result = null;
      let stepAttempt = 0;
      try {
        while (true) {
          try {
            result = await performBridgeAction(
              session,
              resolution.action.action_id,
              waitAfterMs,
              Number.isInteger(state?.state_version) ? state.state_version : undefined,
              {
                strictStateVersion,
                settleAfterAction: true
              }
            );
            break;
          } catch (error) {
            if (
              strictStateVersion ||
              stepAttempt >= ACTION_STATE_VERSION_RETRY_LIMIT ||
              !isRecoverableCombatSequenceExecutionError(error)
            ) {
              throw error;
            }

            state = await getBridgeState(session);
            stepAttempt += 1;
            const retryBlocker = getCombatSequenceContinuationBlocker(state, sequenceLabel);
            if (retryBlocker) {
              return finalize(
                attachReturnStateAfterPayload(
                  {
                    ok: true,
                    resolved: false,
                    reason: retryBlocker.reason,
                    message: retryBlocker.message,
                    requested_action_count: sequencePlan.length,
                    executed_count: executedSteps.length,
                    remaining_count: sequencePlan.length - executedSteps.length,
                    next_sequence_index: sequenceIndex,
                    requested_action_ids: actionIds,
                    normalized_action_ids: normalizedActionIds,
                    reordered_end_turn_to_last: reorderedEndTurnToLast,
                    sequence_plan: sequencePlanOutput,
                    executed_steps: executedSteps,
                    state
                  },
                  returnStateAfter,
                  state
                )
              );
            }

            const retryResolution = resolvePlannedCombatSequenceStep(planStep, state);
            if (!retryResolution.action) {
              return finalize(
                attachReturnStateAfterPayload(
                  {
                    ok: true,
                    resolved: false,
                    reason: "requested_combat_sequence_action_unavailable_after_retry",
                    requested_action_count: sequencePlan.length,
                    executed_count: executedSteps.length,
                    remaining_count: sequencePlan.length - executedSteps.length,
                    next_sequence_index: sequenceIndex,
                    failed_step: summarizeCombatSequencePlanStep(planStep),
                    requested_action_ids: actionIds,
                    normalized_action_ids: normalizedActionIds,
                    reordered_end_turn_to_last: reorderedEndTurnToLast,
                    sequence_plan: sequencePlanOutput,
                    executed_steps: executedSteps,
                    available_actions: summarizeCombatSequenceAvailableActions(
                      getCombatSequenceAvailableActions(state, allowedActionKinds)
                    ),
                    state
                  },
                  returnStateAfter,
                  state
                )
              );
            }

            resolution = retryResolution;
          }
        }
      } catch (error) {
        const failureState = await getBridgeState(session);
        return finalize(
          attachReturnStateAfterPayload(
            {
              ok: true,
              resolved: false,
              reason: "combat_sequence_execution_failed",
              next_sequence_index: sequenceIndex,
              failed_step: summarizeCombatSequencePlanStep(planStep),
              requested_action_count: sequencePlan.length,
              executed_count: executedSteps.length,
              remaining_count: sequencePlan.length - executedSteps.length,
              requested_action_ids: actionIds,
              normalized_action_ids: normalizedActionIds,
              reordered_end_turn_to_last: reorderedEndTurnToLast,
              sequence_plan: sequencePlanOutput,
              executed_steps: executedSteps,
              failure: toolErrorPayload(error),
              state: failureState
            },
            returnStateAfter,
            failureState
          )
        );
      }
      executedSteps.push({
        sequence_index: sequenceIndex,
        requested_action_id: planStep.requested_action_id,
        executed_action_id: resolution.action.action_id,
        match_type: resolution.match_type,
        compatible_candidate_count: resolution.compatible_candidate_count,
        matched_action: summarizeActionForAgent(resolution.action),
        execution: summarizeExecutedAction(result)
      });
      state = result.state;
    }

    return finalize(
      attachReturnStateAfterPayload(
        {
          ok: true,
          resolved: true,
          initial_state_version_adjusted: initialStateVersionAdjusted,
          requested_action_count: sequencePlan.length,
          executed_count: executedSteps.length,
          remaining_count: 0,
          requested_action_ids: actionIds,
          normalized_action_ids: normalizedActionIds,
          reordered_end_turn_to_last: reorderedEndTurnToLast,
          sequence_plan: sequencePlanOutput,
          executed_steps: executedSteps,
          state
        },
        returnStateAfter,
        state
      )
    );
  } catch (error) {
    return asToolResult(toolErrorPayload(error), true);
  }
}

function normalizeCombatSequenceActionIds(actionIds) {
  if (!Array.isArray(actionIds) || actionIds.length <= 0) {
    return [];
  }

  const endTurnActions = [];
  const orderedActions = [];
  for (const actionId of actionIds) {
    if (actionId === "end_turn") {
      endTurnActions.push(actionId);
      continue;
    }

    orderedActions.push(actionId);
  }

  return orderedActions.concat(endTurnActions);
}

function getCombatSequenceContinuationBlocker(state, sequenceLabel) {
  const label =
    typeof sequenceLabel === "string" && sequenceLabel.trim()
      ? sequenceLabel.trim()
      : "combat sequence";

  if (!isPlainObject(state)) {
    return {
      reason: "missing_state",
      message: `Bridge state was missing while resolving the ${label}.`
    };
  }

  const cardSelectionBundle = buildCardSelectionBundle(state);
  if (cardSelectionBundle.in_card_selection_flow) {
    return {
      reason: "card_selection_ready",
      message: `Card selection became visible before the requested ${label} finished.`
    };
  }

  const rewardBundle = buildRewardBundle(state);
  if (rewardBundle.in_reward_flow) {
    return {
      reason: rewardBundle.card_reward_selection.visible
        ? "reward_card_selection_ready"
        : "reward_flow_ready",
      message: `Combat transitioned into rewards before the requested ${label} finished.`
    };
  }

  const screen = typeof state.screen === "string" ? state.screen : null;
  if (screen !== "COMBAT") {
    return {
      reason: `screen:${screen ?? "unknown"}`,
      message: "The game is no longer on the combat screen."
    };
  }

  const combat = isPlainObject(state.combat) ? state.combat : {};
  if (combat.in_progress !== true) {
    return {
      reason: "combat_not_in_progress",
      message: "Combat is no longer in progress."
    };
  }

  if (combat.current_side !== "Player") {
    return {
      reason: "not_player_turn",
      message: "It is no longer the player's turn."
    };
  }

  if (combat.is_play_phase !== true) {
    return {
      reason: "not_play_phase",
      message: "Combat is not currently in the play phase."
    };
  }

  if (combat.player_actions_disabled === true) {
    return {
      reason: "player_actions_disabled",
      message: "Player actions are currently disabled."
    };
  }

  return null;
}

function getRequestedCombatSequenceActionKind(actionId) {
  if (typeof actionId !== "string") {
    return null;
  }

  if (actionId.startsWith("play_card:")) {
    return "play_card";
  }

  if (actionId.startsWith("use_potion:")) {
    return "use_potion";
  }

  if (actionId === "end_turn") {
    return "end_turn";
  }

  return null;
}

function getCombatSequenceAvailableActions(state, allowedActionKinds) {
  return getNonAutomationActions(state).filter((action) => {
    const kind = getRequestedCombatSequenceActionKind(action?.action_id);
    return kind !== null && allowedActionKinds.has(kind);
  });
}

function resolveRequestedCombatSequenceAction(requestedActionId, state, allowedActionKinds) {
  const kind = getRequestedCombatSequenceActionKind(requestedActionId);
  if (kind === null || !allowedActionKinds.has(kind)) {
    return {
      kind,
      action: null,
      unsupported_kind: true,
      match_type: "unsupported",
      compatible_candidate_count: 0
    };
  }

  if (kind === "play_card") {
    return {
      kind,
      ...resolveRequestedPlayCardAction(requestedActionId, getPlayableCardActions(state))
    };
  }

  if (kind === "use_potion") {
    return {
      kind,
      ...resolveRequestedUsePotionAction(requestedActionId, getUsablePotionActions(state))
    };
  }

  if (kind === "end_turn") {
    const action = getNonAutomationActions(state).find(
      (candidate) => candidate?.action_id === "end_turn"
    );
    return {
      kind,
      action: action ?? null,
      match_type: action ? "exact" : "unavailable",
      compatible_candidate_count: action ? 1 : 0
    };
  }

  return {
    kind,
    action: null,
    unsupported_kind: true,
    match_type: "unsupported",
    compatible_candidate_count: 0
  };
}

function buildPlannedCombatSequenceStep(action, sequenceIndex, requestedActionId = null) {
  const actionId = typeof action?.action_id === "string" ? action.action_id : "";
  if (actionId.startsWith("play_card:")) {
    return buildPlannedPlayCardSequenceStep(action, sequenceIndex, requestedActionId);
  }

  if (actionId.startsWith("use_potion:")) {
    return buildPlannedUsePotionSequenceStep(action, sequenceIndex, requestedActionId);
  }

  return buildPlannedEndTurnSequenceStep(action, sequenceIndex, requestedActionId);
}

function summarizeCombatSequencePlanStep(planStep) {
  if (!isPlainObject(planStep)) {
    return null;
  }

  const summary = {
    sequence_index: Number.isInteger(planStep.sequence_index) ? planStep.sequence_index : null,
    kind: typeof planStep.kind === "string" ? planStep.kind : null,
    requested_action_id:
      typeof planStep.requested_action_id === "string" ? planStep.requested_action_id : null
  };

  if (Number.isInteger(planStep.player_index)) {
    summary.player_index = planStep.player_index;
  }

  if (Number.isInteger(planStep.initial_hand_index)) {
    summary.initial_hand_index = planStep.initial_hand_index;
  }

  if (Number.isInteger(planStep.initial_slot_index)) {
    summary.initial_slot_index = planStep.initial_slot_index;
  }

  if (isPlainObject(planStep.card)) {
    summary.card = planStep.card;
  }

  if (isPlainObject(planStep.potion)) {
    summary.potion = planStep.potion;
  }

  if (isPlainObject(planStep.target)) {
    summary.target = planStep.target;
  }

  return summary;
}

function resolvePlannedCombatSequenceStep(planStep, state) {
  if (!isPlainObject(planStep)) {
    return {
      action: null,
      match_type: "unavailable",
      compatible_candidate_count: 0
    };
  }

  if (planStep.kind === "play_card") {
    return resolvePlannedPlayCardStep(planStep, getPlayableCardActions(state));
  }

  if (planStep.kind === "use_potion") {
    return resolvePlannedUsePotionStep(planStep, getUsablePotionActions(state));
  }

  if (planStep.kind === "end_turn") {
    const action = getNonAutomationActions(state).find(
      (candidate) => candidate?.action_id === "end_turn"
    );
    return {
      action: action ?? null,
      match_type: action ? "exact" : "unavailable",
      compatible_candidate_count: action ? 1 : 0
    };
  }

  return {
    action: null,
    match_type: "unavailable",
    compatible_candidate_count: 0
  };
}

async function resolveRoomRewardsTool(args) {
  try {
    const finalize = (payload) =>
      asPrecompactedToolResult(compactRoomRewardsPayload(payload), false);
    const pickCardIndex = optionalInteger(args.pick_card_index, "pick_card_index");
    if (pickCardIndex !== undefined && pickCardIndex < 0) {
      throw new ToolPayloadError(
        "invalid_arguments",
        "pick_card_index must be 0 or greater when provided.",
        {
          field: "pick_card_index"
        }
      );
    }

    const skipCardReward = optionalBoolean(args.skip_card_reward, "skip_card_reward") ?? false;
    const claimAllSafeRewards =
      optionalBoolean(args.claim_all_safe_rewards, "claim_all_safe_rewards") ?? true;
    const takePotions = optionalBoolean(args.take_potions, "take_potions") ?? true;
    const autoProceed = optionalBoolean(args.auto_proceed, "auto_proceed") ?? true;

    if (pickCardIndex !== undefined && skipCardReward) {
      throw new ToolPayloadError(
        "invalid_arguments",
        "pick_card_index and skip_card_reward cannot be used together.",
        {
          fields: ["pick_card_index", "skip_card_reward"]
        }
      );
    }

    const session = getLiveSession();
    let state = await getBridgeState(session);
    const initialRewardBundle = buildRewardBundle(state);

    if (!initialRewardBundle.in_reward_flow) {
      return finalize(
        {
          ok: false,
          resolved: false,
          reason: "not_in_reward_flow",
          reward_bundle: initialRewardBundle,
          state
        },
        false
      );
    }

    const preflightBlocker = getRewardResolutionBlocker(initialRewardBundle, {
      pickCardIndex,
      skipCardReward,
      claimAllSafeRewards,
      takePotions
    });
    if (preflightBlocker) {
      return finalize(
        {
          ok: false,
          resolved: false,
          ...preflightBlocker,
          reward_bundle: initialRewardBundle,
          state
        },
        false
      );
    }

    const executedActions = [];
    const claimedRewards = [];
    let selectedCard = null;

    while (true) {
      const bundle = buildRewardBundle(state);

      if (bundle.card_reward_selection.visible) {
        if (skipCardReward) {
          if (bundle.card_reward_selection.skip_visible !== true) {
            return finalize(
              {
                ok: false,
                resolved: false,
                reason: "card_skip_unavailable",
                reward_bundle: bundle,
                executed_actions: executedActions,
                claimed_rewards: claimedRewards,
                state
              },
              false
            );
          }

          const skipCardResult = await performBridgeAction(
            session,
            "card_reward:skip",
            DEFAULT_ACTION_WAIT_MS,
            Number.isInteger(state?.state_version) ? state.state_version : undefined
          );
          executedActions.push(summarizeExecutedAction(skipCardResult));
          state = skipCardResult.state;
          continue;
        }

        if (pickCardIndex === undefined) {
          return finalize(
            {
              ok: false,
              resolved: false,
              reason: "card_choice_required",
              reward_bundle: bundle,
              executed_actions: executedActions,
              claimed_rewards: claimedRewards,
              state
            },
            false
          );
        }

        if (pickCardIndex >= bundle.card_reward_selection.options.length) {
          return finalize(
            {
              ok: false,
              resolved: false,
              reason: "card_choice_out_of_range",
              reward_bundle: bundle,
              requested_pick_card_index: pickCardIndex,
              executed_actions: executedActions,
              claimed_rewards: claimedRewards,
              state
            },
            false
          );
        }

        const cardOption = bundle.card_reward_selection.options[pickCardIndex];
        const cardActionId = `card_reward:${pickCardIndex}`;
        const cardResult = await performBridgeAction(
          session,
          cardActionId,
          DEFAULT_ACTION_WAIT_MS,
          Number.isInteger(state?.state_version) ? state.state_version : undefined
        );
        executedActions.push(summarizeExecutedAction(cardResult));
        selectedCard = {
          index: pickCardIndex,
          card: cardOption.card
        };
        state = cardResult.state;
        continue;
      }

      if (!bundle.rewards.visible) {
        break;
      }

      const nextReward = chooseNextReward(bundle, {
        pickCardIndex,
        claimAllSafeRewards,
        skipCardReward,
        takePotions
      });
      if (!nextReward) {
        if (skipCardReward) {
          const pendingCardReward = bundle.rewards.entries.find(
            (entry) => entry?.reward?.reward_type === "card" && typeof entry?.action_id === "string"
          );

          if (pendingCardReward) {
            const openCardRewardResult = await performBridgeAction(
              session,
              pendingCardReward.action_id,
              DEFAULT_ACTION_WAIT_MS,
              Number.isInteger(state?.state_version) ? state.state_version : undefined
            );
            executedActions.push(summarizeExecutedAction(openCardRewardResult));
            state = openCardRewardResult.state;
            continue;
          }
        }

        break;
      }

      const rewardResult = await performBridgeAction(
        session,
        nextReward.action_id,
        DEFAULT_ACTION_WAIT_MS,
        Number.isInteger(state?.state_version) ? state.state_version : undefined
      );
      executedActions.push(summarizeExecutedAction(rewardResult));
      claimedRewards.push(nextReward);
      state = rewardResult.state;
    }

    if (autoProceed) {
      const autoProceedResult = await autoAdvanceProceedChain(session, state);
      state = autoProceedResult.state;
      executedActions.push(...autoProceedResult.executed_actions);
    }

    const finalRewardBundle = buildRewardBundle(state);

    return finalize(
      {
        ok: true,
        resolved: !finalRewardBundle.in_reward_flow,
        reward_bundle: initialRewardBundle,
        final_reward_bundle: finalRewardBundle,
        claimed_rewards: claimedRewards,
        selected_card: selectedCard,
        executed_actions: executedActions,
        final_state: state
      },
      false
    );
  } catch (error) {
    return asToolResult(toolErrorPayload(error), true);
  }
}

async function resolveRestSiteTool(args) {
  try {
    const finalize = (payload) =>
      asPrecompactedToolResult(compactRestSitePayload(payload), false);
    const optionIndex = optionalInteger(args.option_index, "option_index");
    const upgradeCardIndex = optionalInteger(args.upgrade_card_index, "upgrade_card_index");
    const autoProceed = optionalBoolean(args.auto_proceed, "auto_proceed") ?? true;

    if (optionIndex === undefined || optionIndex < 0) {
      throw new ToolPayloadError(
        "invalid_arguments",
        "option_index must be 0 or greater.",
        {
          field: "option_index"
        }
      );
    }

    if (upgradeCardIndex !== undefined && upgradeCardIndex < 0) {
      throw new ToolPayloadError(
        "invalid_arguments",
        "upgrade_card_index must be 0 or greater when provided.",
        {
          field: "upgrade_card_index"
        }
      );
    }

    const session = getLiveSession();
    let state = await getBridgeState(session);
    const initialRestSiteBundle = buildRestSiteBundle(state);

    if (!initialRestSiteBundle.in_rest_site_flow) {
      return finalize(
        {
          ok: false,
          resolved: false,
          reason: "not_in_rest_site_flow",
          rest_site_bundle: initialRestSiteBundle,
          state
        },
        false
      );
    }

    let restSiteBundle = initialRestSiteBundle;
    if (
      !restSiteBundle.rest_site.visible &&
      restSiteBundle.deck_upgrade_selection.visible !== true
    ) {
      return finalize(
        {
          ok: false,
          resolved: false,
          reason: "rest_site_not_visible",
          rest_site_bundle: initialRestSiteBundle,
          state
        },
        false
      );
    }

    const executedActions = [];
    let selectedUpgradeCard = null;
    let selectedOption = null;

    if (!restSiteBundle.deck_upgrade_selection.visible) {
      if (optionIndex >= restSiteBundle.rest_site.options.length) {
        return finalize(
          {
            ok: false,
            resolved: false,
            reason: "rest_site_option_out_of_range",
            requested_option_index: optionIndex,
            rest_site_bundle: initialRestSiteBundle,
            state
          },
          false
        );
      }

      selectedOption = restSiteBundle.rest_site.options[optionIndex];
      const optionResult = await performBridgeAction(
        session,
        `rest_site:${optionIndex}`,
        DEFAULT_ACTION_WAIT_MS
      );
      executedActions.push(summarizeExecutedAction(optionResult));
      state = optionResult.state;
      restSiteBundle = buildRestSiteBundle(state);
    } else if (optionIndex < restSiteBundle.rest_site.options.length) {
      selectedOption = restSiteBundle.rest_site.options[optionIndex];
    }

    if (restSiteBundle.deck_upgrade_selection.visible) {
      if (upgradeCardIndex === undefined) {
        return finalize(
          {
            ok: false,
            resolved: false,
            reason: "upgrade_choice_required",
            selected_option: selectedOption,
            rest_site_bundle: restSiteBundle,
            executed_actions: executedActions,
            state
          },
          false
        );
      }

      if (upgradeCardIndex >= restSiteBundle.deck_upgrade_selection.options.length) {
        return finalize(
          {
            ok: false,
            resolved: false,
            reason: "upgrade_choice_out_of_range",
            requested_upgrade_card_index: upgradeCardIndex,
            selected_option: selectedOption,
            rest_site_bundle: restSiteBundle,
            executed_actions: executedActions,
            state
          },
          false
        );
      }

      selectedUpgradeCard = restSiteBundle.deck_upgrade_selection.options[upgradeCardIndex];
      const upgradeResult = await performBridgeAction(
        session,
        `deck_upgrade:select:${upgradeCardIndex}`,
        DEFAULT_ACTION_WAIT_MS
      );
      executedActions.push(summarizeExecutedAction(upgradeResult));
      state = upgradeResult.state;
      restSiteBundle = buildRestSiteBundle(state);

      if (
        getNonAutomationActions(state).some(
          (action) => action?.action_id === "deck_upgrade:confirm"
        )
      ) {
        const confirmUpgradeResult = await performBridgeAction(
          session,
          "deck_upgrade:confirm",
          DEFAULT_ACTION_WAIT_MS
        );
        executedActions.push(summarizeExecutedAction(confirmUpgradeResult));
        state = confirmUpgradeResult.state;
        restSiteBundle = buildRestSiteBundle(state);
      }
    }

    if (autoProceed) {
      const autoProceedResult = await autoAdvanceRestSiteProceedChain(session, state);
      state = autoProceedResult.state;
      executedActions.push(...autoProceedResult.executed_actions);
      restSiteBundle = buildRestSiteBundle(state);
    }

    return finalize(
      {
        ok: true,
        resolved: isMapReadyState(state) || !restSiteBundle.in_rest_site_flow,
        selected_option: selectedOption,
        selected_upgrade_card: selectedUpgradeCard,
        rest_site_bundle: initialRestSiteBundle,
        executed_actions: executedActions,
        final_state: state
      },
      false
    );
  } catch (error) {
    return asToolResult(toolErrorPayload(error), true);
  }
}

async function resolveCardSelectionTool(args) {
  try {
    const selectIndices = optionalIntegerArray(args.select_indices, "select_indices") ?? [];
    const terminalAction = normalizeCardSelectionTerminalAction(args.terminal_action);
    const expectedMinSelect = optionalInteger(
      args.expected_min_select,
      "expected_min_select"
    );
    const expectedMaxSelect = optionalInteger(
      args.expected_max_select,
      "expected_max_select"
    );

    if (expectedMinSelect !== undefined && expectedMinSelect < 0) {
      throw new ToolPayloadError(
        "invalid_arguments",
        "expected_min_select must be 0 or greater when provided.",
        {
          field: "expected_min_select"
        }
      );
    }

    if (expectedMaxSelect !== undefined && expectedMaxSelect < 0) {
      throw new ToolPayloadError(
        "invalid_arguments",
        "expected_max_select must be 0 or greater when provided.",
        {
          field: "expected_max_select"
        }
      );
    }

    if (
      expectedMinSelect !== undefined &&
      expectedMaxSelect !== undefined &&
      expectedMinSelect > expectedMaxSelect
    ) {
      throw new ToolPayloadError(
        "invalid_arguments",
        "expected_min_select cannot be greater than expected_max_select.",
        {
          fields: ["expected_min_select", "expected_max_select"]
        }
      );
    }

    const uniqueSelectIndices = [...new Set(selectIndices)];
    if (uniqueSelectIndices.length !== selectIndices.length) {
      throw new ToolPayloadError(
        "invalid_arguments",
        "select_indices cannot contain duplicates.",
        {
          field: "select_indices"
        }
      );
    }

    if (uniqueSelectIndices.length === 0 && terminalAction === "none") {
      throw new ToolPayloadError(
        "invalid_arguments",
        "Provide at least one select_indices entry or a terminal_action other than none.",
        {
          fields: ["select_indices", "terminal_action"]
        }
      );
    }

    const session = getLiveSession();
    let state = await getBridgeState(session);
    const initialCardSelectionBundle = buildCardSelectionBundle(state);

    if (!initialCardSelectionBundle.in_card_selection_flow) {
      return asToolResult(
        {
          ok: false,
          resolved: false,
          reason: "not_in_card_selection_flow",
          card_selection_bundle: initialCardSelectionBundle,
          state
        },
        false
      );
    }

    if (!initialCardSelectionBundle.card_selection.visible) {
      return asToolResult(
        {
          ok: false,
          resolved: false,
          reason: "card_selection_not_visible",
          card_selection_bundle: initialCardSelectionBundle,
          state
        },
        false
      );
    }

    if (
      expectedMinSelect !== undefined &&
      initialCardSelectionBundle.card_selection.min_select !== expectedMinSelect
    ) {
      return asToolResult(
        {
          ok: false,
          resolved: false,
          reason: "min_select_mismatch",
          expected_min_select: expectedMinSelect,
          observed_min_select: initialCardSelectionBundle.card_selection.min_select,
          card_selection_bundle: initialCardSelectionBundle,
          state
        },
        false
      );
    }

    if (
      expectedMaxSelect !== undefined &&
      initialCardSelectionBundle.card_selection.max_select !== expectedMaxSelect
    ) {
      return asToolResult(
        {
          ok: false,
          resolved: false,
          reason: "max_select_mismatch",
          expected_max_select: expectedMaxSelect,
          observed_max_select: initialCardSelectionBundle.card_selection.max_select,
          card_selection_bundle: initialCardSelectionBundle,
          state
        },
        false
      );
    }

    const initialOptionEntries = buildIndexedOptionEntries(
      initialCardSelectionBundle.card_selection.options
    );
    const missingInitialIndices = uniqueSelectIndices.filter(
      (requestedIndex) =>
        !initialOptionEntries.some(
          (entry) => entry.option_index === requestedIndex
        )
    );
    if (missingInitialIndices.length > 0) {
      return asToolResult(
        {
          ok: false,
          resolved: false,
          reason: "card_selection_option_out_of_range",
          requested_select_indices: missingInitialIndices,
          available_option_indices: initialOptionEntries.map(
            (entry) => entry.option_index
          ),
          card_selection_bundle: initialCardSelectionBundle,
          state
        },
        false
      );
    }

    const initialSelectedCount = initialCardSelectionBundle.card_selection.selected_count ?? 0;
    const initialMaxSelect = initialCardSelectionBundle.card_selection.max_select;
    if (
      Number.isInteger(initialMaxSelect) &&
      initialSelectedCount + uniqueSelectIndices.length > initialMaxSelect
    ) {
      return asToolResult(
        {
          ok: false,
          resolved: false,
          reason: "too_many_cards_requested",
          requested_select_count: uniqueSelectIndices.length,
          selected_count_before: initialSelectedCount,
          max_select: initialMaxSelect,
          card_selection_bundle: initialCardSelectionBundle,
          state
        },
        false
      );
    }

    const executedActions = [];
    const selectedCards = [];
    const selectionPlans = uniqueSelectIndices.map((requestedIndex) => {
      const initialEntry = findIndexedOptionEntry(
        initialCardSelectionBundle.card_selection.options,
        requestedIndex
      );

      return {
        requested_index: requestedIndex,
        initial_option_index: initialEntry?.option_index ?? requestedIndex,
        selection_id:
          typeof initialEntry?.option?.selection_id === "string"
            ? initialEntry.option.selection_id
            : null,
        card: initialEntry?.option?.card ?? null
      };
    });

    for (const selectionPlan of selectionPlans) {
      const currentCardSelectionBundle = buildCardSelectionBundle(state);
      if (
        !currentCardSelectionBundle.in_card_selection_flow ||
        !currentCardSelectionBundle.card_selection.visible
      ) {
        return asToolResult(
          {
            ok: false,
            resolved: false,
            reason: "card_selection_flow_closed_early",
            requested_select_indices: uniqueSelectIndices,
            selected_cards: selectedCards,
            executed_actions: executedActions,
            initial_card_selection_bundle: initialCardSelectionBundle,
            current_card_selection_bundle: currentCardSelectionBundle,
            state
          },
          false
        );
      }

      const currentOptionEntries = buildIndexedOptionEntries(
        currentCardSelectionBundle.card_selection.options
      );
      let currentEntry =
        selectionPlan.selection_id === null
          ? null
          : currentOptionEntries.find(
            (entry) => entry.option?.selection_id === selectionPlan.selection_id
          ) ?? null;

      if (currentEntry === null) {
        currentEntry = currentOptionEntries.find(
          (entry) => entry.option_index === selectionPlan.initial_option_index
        ) ?? null;
      }

      if (currentEntry === null) {
        return asToolResult(
          {
            ok: false,
            resolved: false,
            reason: "card_selection_option_out_of_range_after_reindex",
            requested_index: selectionPlan.requested_index,
            current_option_indices: currentOptionEntries.map(
              (entry) => entry.option_index
            ),
            selected_cards: selectedCards,
            executed_actions: executedActions,
            initial_card_selection_bundle: initialCardSelectionBundle,
            current_card_selection_bundle: currentCardSelectionBundle,
            state
          },
          false
        );
      }

      const option = currentEntry.option;
      if (option?.is_selected === true) {
        return asToolResult(
          {
            ok: false,
            resolved: false,
            reason: "card_selection_option_already_selected",
            requested_index: selectionPlan.requested_index,
            resolved_index: currentEntry.option_index,
            selected_cards: selectedCards,
            executed_actions: executedActions,
            initial_card_selection_bundle: initialCardSelectionBundle,
            current_card_selection_bundle: currentCardSelectionBundle,
            state
          },
          false
        );
      }

      const actionId =
        typeof option?.action_id === "string" && option.action_id
          ? option.action_id
          : `card_selection:select:${currentEntry.option_index}`;
      if (!currentCardSelectionBundle.non_automation_action_ids.includes(actionId)) {
        return asToolResult(
          {
            ok: false,
            resolved: false,
            reason: "card_selection_action_unavailable",
            requested_index: selectionPlan.requested_index,
            resolved_index: currentEntry.option_index,
            requested_action_id: actionId,
            selected_cards: selectedCards,
            executed_actions: executedActions,
            initial_card_selection_bundle: initialCardSelectionBundle,
            current_card_selection_bundle: currentCardSelectionBundle,
            state
          },
          false
        );
      }

      const selectResult = await performBridgeAction(
        session,
        actionId,
        DEFAULT_ACTION_WAIT_MS
      );
      executedActions.push(summarizeExecutedAction(selectResult));
      selectedCards.push({
        requested_index: selectionPlan.requested_index,
        selected_option_index: currentEntry.option_index,
        match_type:
          currentEntry.option_index === selectionPlan.requested_index
            ? "index"
            : selectionPlan.selection_id !== null
              ? "selection_id"
              : "reindexed",
        card: option?.card ?? null
      });
      state = selectResult.state;
    }

    let finalCardSelectionBundle = buildCardSelectionBundle(state);
    let terminalActionResult = null;

    if (terminalAction !== "none") {
      const terminalActionId = `card_selection:${terminalAction}`;

      if (!finalCardSelectionBundle.in_card_selection_flow) {
        terminalActionResult = {
          requested_action_id: terminalActionId,
          executed: false,
          reason: "selection_flow_already_closed"
        };
      } else if (!finalCardSelectionBundle.non_automation_action_ids.includes(terminalActionId)) {
        return asToolResult(
          {
            ok: false,
            resolved: false,
            reason: "terminal_action_unavailable",
            requested_terminal_action: terminalAction,
            selected_cards: selectedCards,
            executed_actions: executedActions,
            initial_card_selection_bundle: initialCardSelectionBundle,
            current_card_selection_bundle: finalCardSelectionBundle,
            state
          },
          false
        );
      } else {
        if (terminalAction === "confirm") {
          const selectedCount = finalCardSelectionBundle.card_selection.selected_count ?? 0;
          const minSelect = finalCardSelectionBundle.card_selection.min_select ?? 0;
          if (selectedCount < minSelect) {
            return asToolResult(
              {
                ok: false,
                resolved: false,
                reason: "selection_incomplete",
                requested_terminal_action: terminalAction,
                selected_count: selectedCount,
                min_select: minSelect,
                selected_cards: selectedCards,
                executed_actions: executedActions,
                initial_card_selection_bundle: initialCardSelectionBundle,
                current_card_selection_bundle: finalCardSelectionBundle,
                state
              },
              false
            );
          }
        }

        const terminalResult = await performBridgeAction(
          session,
          terminalActionId,
          DEFAULT_ACTION_WAIT_MS
        );
        executedActions.push(summarizeExecutedAction(terminalResult));
        state = terminalResult.state;
        finalCardSelectionBundle = buildCardSelectionBundle(state);
        terminalActionResult = {
          requested_action_id: terminalActionId,
          executed: true,
          screen_after: state?.screen ?? null
        };
      }
    }

    return asToolResult(
      {
        ok: true,
        resolved: !finalCardSelectionBundle.in_card_selection_flow,
        initial_card_selection_bundle: initialCardSelectionBundle,
        selected_cards: selectedCards,
        terminal_action: terminalActionResult,
        executed_actions: executedActions,
        final_card_selection_bundle: finalCardSelectionBundle,
        final_state: state
      },
      false
    );
  } catch (error) {
    return asToolResult(toolErrorPayload(error), true);
  }
}

async function pickOptionTool(args) {
  try {
    const index = optionalInteger(args.index, "index");
    if (index === undefined || index < 0) {
      throw new ToolPayloadError("invalid_arguments", "index must be 0 or greater.", {
        field: "index"
      });
    }

    const surface = normalizeIndexedOptionSurface(args.surface);
    const terminalAction =
      args.terminal_action === undefined
        ? "none"
        : normalizeCardSelectionTerminalAction(args.terminal_action);

    if (surface !== "card_selection" && terminalAction !== "none") {
      throw new ToolPayloadError(
        "invalid_arguments",
        "terminal_action is only supported when surface=card_selection.",
        {
          fields: ["surface", "terminal_action"]
        }
      );
    }

    const session = getLiveSession();
    const state = await getBridgeState(session);
    const availableSurfaces = summarizeIndexedOptionSurfacesForAgent(state);
    const resolvedSurface = resolveIndexedOptionSurface(state, surface);

    if (!resolvedSurface.ok) {
      return asToolResult(
        {
          ok: false,
          resolved: false,
          reason: resolvedSurface.reason,
          requested_surface: surface,
          available_surfaces: availableSurfaces,
          state
        },
        false
      );
    }

    if (resolvedSurface.surface === "card_selection") {
      return await resolveCardSelectionTool({
        select_indices: [index],
        terminal_action: terminalAction
      });
    }

    const optionEntry = findIndexedOptionEntry(resolvedSurface.options, index);
    if (!optionEntry) {
      return asToolResult(
        {
          ok: false,
          resolved: false,
          reason: "option_out_of_range",
          requested_surface: surface,
          resolved_surface: resolvedSurface.surface,
          requested_index: index,
          available_option_indices: buildIndexedOptionEntries(
            resolvedSurface.options
          ).map((entry) => entry.option_index),
          available_surfaces: availableSurfaces,
          state
        },
        false
      );
    }

    const actionId = resolvedSurface.getActionId(optionEntry.option_index);
    if (!isActionIdCurrentlyAvailable(state, actionId)) {
      return asToolResult(
        {
          ok: false,
          resolved: false,
          reason: "option_action_unavailable",
          requested_surface: surface,
          resolved_surface: resolvedSurface.surface,
          requested_index: index,
          resolved_index: optionEntry.option_index,
          requested_action_id: actionId,
          available_surfaces: availableSurfaces,
          state
        },
        false
      );
    }

    const result = await performBridgeAction(
      session,
      actionId,
      DEFAULT_ACTION_WAIT_MS,
      Number.isInteger(state?.state_version) ? state.state_version : undefined
    );

    return asToolResult(
      {
        ok: true,
        resolved: true,
        requested_surface: surface,
        resolved_surface: resolvedSurface.surface,
        selected_index: index,
        resolved_index: optionEntry.option_index,
        selected_option: summarizeIndexedOptionForAgent(
          resolvedSurface.surface,
          optionEntry.option
        ),
        executed_action: summarizeExecutedAction(result),
        final_state: result.state
      },
      false
    );
  } catch (error) {
    return asToolResult(toolErrorPayload(error), true);
  }
}

async function travelToCoordinateTool(args) {
  try {
    const finalize = (payload) =>
      asPrecompactedToolResult(compactTravelToCoordinatePayload(payload), false);
    const col = optionalInteger(args.col, "col");
    const row = optionalInteger(args.row, "row");
    if (col === undefined || col < 0) {
      throw new ToolPayloadError("invalid_arguments", "col must be 0 or greater.", {
        field: "col"
      });
    }
    if (row === undefined || row < 0) {
      throw new ToolPayloadError("invalid_arguments", "row must be 0 or greater.", {
        field: "row"
      });
    }

    const waitAfterMs = clampInteger(
      args.wait_after_ms,
      DEFAULT_ACTION_WAIT_MS,
      0,
      MAX_ACTION_WAIT_MS,
      "wait_after_ms"
    );

    const session = getLiveSession();
    const initialState = await getBridgeState(session);
    const prepared = await prepareStateForMapTravel(session, initialState);

    if (prepared.blocker) {
      return finalize(
        {
          ok: false,
          resolved: false,
          reason: prepared.blocker.reason,
          message: prepared.blocker.message,
          executed_actions: prepared.executed_actions,
          ...(prepared.blocker.reward_bundle
            ? { reward_bundle: prepared.blocker.reward_bundle }
            : {}),
          ...(prepared.blocker.rest_site_bundle
            ? { rest_site_bundle: prepared.blocker.rest_site_bundle }
            : {}),
          ...(prepared.blocker.card_selection_bundle
            ? { card_selection_bundle: prepared.blocker.card_selection_bundle }
            : {}),
          final_state: prepared.state
        },
        false
      );
    }

    const settledSnapshot = await waitForStableMapRouteSnapshot(session);
    const state = settledSnapshot.state;
    const requestedCoord = { col, row };
    const requestedCoordKey = toCoordKey(requestedCoord);

    if (!isMapReadyState(state)) {
      return finalize(
        {
          ok: false,
          resolved: false,
          reason: "map_not_ready",
          snapshot_status: summarizeMapSnapshotStatusForAgent(settledSnapshot),
          executed_actions: prepared.executed_actions,
          final_state: state
        },
        false
      );
    }

    if (!settledSnapshot.mapActionKeys.includes(requestedCoordKey)) {
      return finalize(
        {
          ok: false,
          resolved: false,
          reason: "coordinate_not_reachable",
          requested_coord: requestedCoord,
          reachable_coords: settledSnapshot.mapActionKeys.map(parseCoordKey).filter(Boolean),
          snapshot_status: summarizeMapSnapshotStatusForAgent(settledSnapshot),
          executed_actions: prepared.executed_actions,
          final_state: state
        },
        false
      );
    }

    const travelResult = await performBridgeAction(
      session,
      `map:${col},${row}`,
      waitAfterMs,
      Number.isInteger(state?.state_version) ? state.state_version : undefined
    );

    return finalize(
      {
        ok: true,
        resolved: true,
        requested_coord: requestedCoord,
        snapshot_status: summarizeMapSnapshotStatusForAgent(settledSnapshot),
        executed_actions: [
          ...prepared.executed_actions,
          summarizeExecutedAction(travelResult)
        ],
        final_state: travelResult.state
      },
      false
    );
  } catch (error) {
    return asToolResult(toolErrorPayload(error), true);
  }
}

async function waitForChangeTool(args) {
  try {
    const session = getLiveSession();
    const timeoutMs = clampInteger(
      args.timeout_ms,
      DEFAULT_WAIT_TIMEOUT_MS,
      1,
      MAX_WAIT_TIMEOUT_MS,
      "timeout_ms"
    );

    let baselineStateVersion = optionalInteger(
      args.baseline_state_version,
      "baseline_state_version"
    );
    let baselineStateHash = optionalString(args.baseline_state_hash, "baseline_state_hash");

    const initialState = await getBridgeState(session);

    if (baselineStateVersion === undefined && baselineStateHash === undefined) {
      baselineStateVersion = initialState.state_version;
      baselineStateHash = initialState.state_hash;
    }

    if (
      baselineStateVersion !== undefined &&
      initialState.state_version !== baselineStateVersion
    ) {
      return asToolResult(
        {
          ok: true,
          changed: true,
          reason: "state_version_mismatch",
          baseline_state_version: baselineStateVersion,
          baseline_state_hash: baselineStateHash ?? null,
          state: initialState
        },
        false
      );
    }

    if (baselineStateHash !== undefined && initialState.state_hash !== baselineStateHash) {
      return asToolResult(
        {
          ok: true,
          changed: true,
          reason: "state_hash_mismatch",
          baseline_state_version: baselineStateVersion ?? null,
          baseline_state_hash: baselineStateHash,
          state: initialState
        },
        false
      );
    }

    try {
      const changedState = await waitForBridgeStateEvent(session, {
        timeout_ms: timeoutMs,
        after_state_version:
          baselineStateVersion !== undefined ? baselineStateVersion : undefined,
        predicate: (candidateState) => {
          const versionChanged =
            baselineStateVersion !== undefined &&
            candidateState?.state_version !== baselineStateVersion;
          const hashChanged =
            baselineStateHash !== undefined &&
            candidateState?.state_hash !== baselineStateHash;
          return versionChanged || hashChanged;
        }
      });

      const versionChanged =
        baselineStateVersion !== undefined &&
        changedState?.state_version !== baselineStateVersion;
      return asToolResult(
        {
          ok: true,
          changed: true,
          reason: versionChanged ? "state_version_changed" : "state_hash_changed",
          baseline_state_version: baselineStateVersion ?? null,
          baseline_state_hash: baselineStateHash ?? null,
          state: changedState
        },
        false
      );
    } catch (error) {
      if (error instanceof ToolPayloadError && error.code === "bridge_event_wait_timeout") {
        return asToolResult(
          {
            ok: false,
            changed: false,
            error: "timeout",
            message: `State did not change within ${timeoutMs} ms.`,
            baseline_state_version: baselineStateVersion ?? null,
            baseline_state_hash: baselineStateHash ?? null,
            state: ensureBridgeEventClient(session).getCachedState(Number.POSITIVE_INFINITY) ?? initialState
          },
          true
        );
      }

      throw error;
    }
  } catch (error) {
    return asToolResult(toolErrorPayload(error), true);
  }
}

function summarizeActionableWaitState(state) {
  const screen = typeof state?.screen === "string" ? state.screen : null;
  const nonAutomationActions = getNonAutomationActions(state);
  const actionIds = nonAutomationActions
    .map((action) => action?.action_id)
    .filter((actionId) => typeof actionId === "string");
  const rewardBundle = buildRewardBundle(state);
  const cardSelectionBundle = buildCardSelectionBundle(state);
  const restSiteBundle = buildRestSiteBundle(state);
  const shopBundle = buildShopBundle(state);
  const transientRoomEntryContinue = isRlTransientRoomEntryContinueState(state);

  let surface = null;
  if (rewardBundle.in_reward_flow) {
    surface = rewardBundle.card_reward_selection.visible === true ? "card_reward" : "reward";
  } else if (cardSelectionBundle.in_card_selection_flow) {
    surface = "card_selection";
  } else if (restSiteBundle.in_rest_site_flow) {
    surface =
      restSiteBundle.deck_upgrade_selection?.visible === true ? "deck_upgrade" : "rest_site";
  } else if (state?.event_options?.visible === true && !transientRoomEntryContinue) {
    surface = isCrystalSphereScreen(screen) ? "event_crystal_sphere" : "event";
  } else if (isMapReadyState(state)) {
    surface = "map";
  } else if (
    screen === "COMBAT" &&
    state?.combat?.in_progress === true &&
    state?.combat?.is_play_phase === true &&
    state?.combat?.player_actions_disabled !== true
  ) {
    surface = "combat";
  } else if (shopBundle.in_shop_flow) {
    surface = "shop";
  } else if (actionIds.length > 0) {
    surface = "actions";
  }

  const summary = {
    screen,
    surface,
    action_count: actionIds.length,
    action_ids: actionIds
  };

  if (surface === "card_selection") {
    summary.selected_count = cardSelectionBundle.card_selection?.selected_count ?? null;
    summary.min_select = cardSelectionBundle.card_selection?.min_select ?? null;
    summary.max_select = cardSelectionBundle.card_selection?.max_select ?? null;
  } else if (surface === "reward" || surface === "card_reward") {
    summary.reward_count = Array.isArray(rewardBundle.rewards?.entries)
      ? rewardBundle.rewards.entries.length
      : 0;
    summary.card_reward_option_count = Array.isArray(rewardBundle.card_reward_selection?.options)
      ? rewardBundle.card_reward_selection.options.length
      : 0;
  } else if (surface === "rest_site" || surface === "deck_upgrade") {
    summary.rest_option_count = Array.isArray(restSiteBundle.rest_site?.options)
      ? restSiteBundle.rest_site.options.length
      : 0;
    summary.upgrade_option_count = Array.isArray(restSiteBundle.deck_upgrade_selection?.options)
      ? restSiteBundle.deck_upgrade_selection.options.length
      : 0;
  } else if (surface === "shop") {
    summary.shop_item_count = Array.isArray(shopBundle.shop?.items)
      ? shopBundle.shop.items.length
      : 0;
  }

  const actionable =
    !transientRoomEntryContinue &&
    actionIds.length > 0 &&
    (typeof surface === "string" && surface.length > 0 || actionIds.length > 0);

  return {
    actionable,
    screen,
    surface,
    action_count: actionIds.length,
    fingerprint: JSON.stringify(summary)
  };
}

async function waitUntilActionableTool(args) {
  try {
    const session = getLiveSession();
    const timeoutMs = clampInteger(
      args.timeout_ms,
      DEFAULT_WAIT_TIMEOUT_MS,
      1,
      MAX_WAIT_TIMEOUT_MS,
      "timeout_ms"
    );
    const initialState = await getBridgeState(session);
    const initialAnalysis = summarizeActionableWaitState(initialState);
    if (initialAnalysis.actionable) {
      return asToolResult(
        {
          ok: true,
          actionable: true,
          surface: initialAnalysis.surface,
          event_driven: true,
          state: initialState
        },
        false
      );
    }

    try {
      const actionableState = await waitForBridgeStateEvent(session, {
        timeout_ms: timeoutMs,
        after_state_version: getStateVersionValue(initialState) ?? undefined,
        predicate: (candidateState) => summarizeActionableWaitState(candidateState).actionable
      });
      const analysis = summarizeActionableWaitState(actionableState);
      return asToolResult(
        {
          ok: true,
          actionable: true,
          surface: analysis.surface,
          event_driven: true,
          state: actionableState
        },
        false
      );
    } catch (error) {
      if (error instanceof ToolPayloadError && error.code === "bridge_event_wait_timeout") {
        const lastState =
          ensureBridgeEventClient(session).getCachedState(Number.POSITIVE_INFINITY) ?? initialState;
        const lastAnalysis = summarizeActionableWaitState(lastState);
        return asToolResult(
          {
            ok: true,
            actionable: false,
            timed_out: true,
            event_driven: true,
            surface: lastAnalysis.surface,
            state: lastState
          },
          false
        );
      }

      throw error;
    }
  } catch (error) {
    return asToolResult(toolErrorPayload(error), true);
  }
}

async function envSpecTool() {
  try {
    const session = getLiveSession();
    const response = await bridgeRequestJson(session, "env/spec", {
      method: "GET",
      timeoutMs: DEFAULT_HTTP_TIMEOUT_MS
    });
    return asToolResult(response.payload, false);
  } catch (error) {
    return asToolResult(toolErrorPayload(error), true);
  }
}

async function envResetTool(args) {
  try {
    const session = getLiveSession();
    const timeoutMs = clampInteger(
      args.timeout_ms,
      DEFAULT_RL_RESET_TIMEOUT_MS,
      1,
      MAX_WAIT_TIMEOUT_MS,
      "timeout_ms"
    );
    const response = await bridgeRequestJson(session, "env/reset", {
      method: "POST",
      body: {
        ...(typeof args.character === "string" ? { character: args.character } : {}),
        ...(typeof args.defensive_buffs === "boolean"
          ? { defensive_buffs: args.defensive_buffs }
          : {}),
        timeout_ms: timeoutMs
      },
      timeoutMs: DEFAULT_HTTP_TIMEOUT_MS + timeoutMs
    });
    return asToolResult(response.payload, false);
  } catch (error) {
    return asToolResult(toolErrorPayload(error), true);
  }
}

async function envStepTool(args) {
  try {
    const timeoutMs = clampInteger(
      args.timeout_ms,
      DEFAULT_RL_STEP_TIMEOUT_MS,
      1,
      MAX_WAIT_TIMEOUT_MS,
      "timeout_ms"
    );
    const session = getLiveSession();
    const body = {
      episode_id: requireNonEmptyString(args.episode_id, "episode_id"),
      timeout_ms: timeoutMs
    };
    if (Number.isInteger(args.action_index)) {
      body.action_index = args.action_index;
    }
    if (typeof args.action_id === "string" && args.action_id.trim()) {
      body.action_id = args.action_id;
    }
    const response = await bridgeRequestJson(session, "env/step", {
      method: "POST",
      body,
      timeoutMs: DEFAULT_HTTP_TIMEOUT_MS + timeoutMs
    });
    return asToolResult(response.payload, false);
  } catch (error) {
    return asToolResult(toolErrorPayload(error), true);
  }
}

async function waitForRlResetState(session, initialState, timeoutMs) {
  const waitResult = await waitForRlActionableOrTerminalState(session, initialState, timeoutMs);
  if (waitResult.timed_out === true && !isRlTerminalState(waitResult.state)) {
    throw new ToolPayloadError(
      "env_reset_timeout",
      `sts2_env_reset did not reach a stable actionable state within ${timeoutMs} ms.`,
      {
        timeout_ms: timeoutMs,
        screen: typeof waitResult?.state?.screen === "string" ? waitResult.state.screen : null,
        surface: getRlSurfaceType(waitResult.state)
      }
    );
  }

  return waitResult.state;
}

async function waitForRlActionableOrTerminalState(session, initialState, timeoutMs) {
  let state = isPlainObject(initialState) ? initialState : await getBridgeState(session);
  const startedAt = Date.now();
  let analysis = summarizeActionableWaitState(state);
  let terminal = isRlTerminalState(state);
  let stableFingerprint =
    analysis.actionable || terminal ? getRlStableWaitFingerprint(state, analysis, terminal) : null;
  let stableCount = stableFingerprint ? 1 : 0;

  while (true) {
    const elapsed = Date.now() - startedAt;
    const remaining = timeoutMs - elapsed;
    if (remaining <= 0) {
      return {
        state,
        actionable: analysis.actionable,
        terminal,
        timed_out: true,
        surface: getRlSurfaceType(state)
      };
    }

    if ((analysis.actionable || terminal) && stableCount >= RL_ENV_STABILITY_TARGET) {
      return {
        state,
        actionable: analysis.actionable,
        terminal,
        timed_out: false,
        surface: getRlSurfaceType(state)
      };
    }

    const sliceTimeout = Math.min(remaining, RL_ENV_STABILITY_POLL_TIMEOUT_MS);
    try {
      state = await waitForBridgeStateEvent(session, {
        timeout_ms: sliceTimeout,
        after_state_version: getStateVersionValue(state) ?? undefined,
        predicate: () => true
      });
      analysis = summarizeActionableWaitState(state);
      terminal = isRlTerminalState(state);

      if (analysis.actionable || terminal) {
        const nextFingerprint = getRlStableWaitFingerprint(state, analysis, terminal);
        if (nextFingerprint === stableFingerprint) {
          stableCount += 1;
        } else {
          stableFingerprint = nextFingerprint;
          stableCount = 1;
        }
      } else {
        stableFingerprint = null;
        stableCount = 0;
      }
    } catch (error) {
      if (!(error instanceof ToolPayloadError) || error.code !== "bridge_event_wait_timeout") {
        throw error;
      }

      const lastState =
        ensureBridgeEventClient(session).getCachedState(Number.POSITIVE_INFINITY) ?? state;
      state = lastState;
      analysis = summarizeActionableWaitState(lastState);
      terminal = isRlTerminalState(lastState);
      if (analysis.actionable || terminal) {
        return {
          state: lastState,
          actionable: analysis.actionable,
          terminal,
          timed_out: false,
          surface: getRlSurfaceType(lastState)
        };
      }
    }
  }
}

function getRlStableWaitFingerprint(state, analysis, terminal) {
  if (terminal) {
    return JSON.stringify({
      terminal: true,
      outcome: getRlTerminalOutcome(state)
    });
  }

  return analysis?.fingerprint ?? null;
}

function isRlTransientRoomEntryContinueState(state) {
  if (!isPlainObject(state) || state?.event_options?.visible !== true) {
    return false;
  }

  const roomType = getRlCurrentRoomType(state);
  if (roomType === "Event") {
    return false;
  }

  const options = Array.isArray(state.event_options.options)
    ? state.event_options.options.filter((option) => isPlainObject(option))
    : [];
  if (options.length !== 1 || options[0]?.is_proceed !== true) {
    return false;
  }

  return true;
}

function getRlCurrentRoomType(state) {
  if (typeof state?.run?.current_room?.room_type === "string") {
    return state.run.current_room.room_type;
  }

  return typeof state?.run?.room_type === "string" ? state.run.room_type : null;
}

function buildRlStepPayload(options) {
  const episode = options.episode;
  const beforeState = options.beforeState;
  const afterState = options.afterState;
  const selectedAction = options.selectedAction ?? null;
  const truncated = options.truncated === true;
  const truncationReason =
    typeof options.truncationReason === "string" ? options.truncationReason : null;
  const rewardBreakdown =
    isPlainObject(options.precomputedRewardBreakdown)
      ? options.precomputedRewardBreakdown
      : buildRlRewardBreakdown(beforeState, afterState);
  const done = isRlTerminalState(afterState) || truncated;

  return {
    ok: true,
    episode_id: episode.id,
    step_index: episode.step_index,
    reward: Number.isFinite(rewardBreakdown.total) ? rewardBreakdown.total : 0,
    done,
    truncated,
    obs: buildRlObservation(afterState),
    legal_actions: done ? [] : buildRlLegalActions(afterState),
    info: {
      action: selectedAction
        ? {
          idx: Number.isInteger(selectedAction.idx) ? selectedAction.idx : null,
          key: typeof selectedAction.key === "string" ? selectedAction.key : null,
          action_id: selectedAction.action_id,
          kind: selectedAction.kind ?? null,
          label: selectedAction.label ?? null
        }
        : null,
      surface_before: getRlSurfaceType(beforeState),
      surface_after: getRlSurfaceType(afterState),
      state_version_before: getStateVersionValue(beforeState),
      state_version_after: getStateVersionValue(afterState),
      truncation_reason: truncationReason,
      reward_breakdown: rewardBreakdown,
      bridge: summarizeRlBridgeActionInfo(options.bridgeActionResult)
    }
  };
}

function buildRlObservation(state) {
  if (!isPlainObject(state)) {
    return null;
  }

  const summary = summarizeStateForAgent(state);
  const player = isPlainObject(summary?.player) ? summary.player : {};
  const run = isPlainObject(summary?.run) ? summary.run : {};
  const combat = isPlainObject(summary?.combat) ? summary.combat : {};

  const observation = {
    screen: typeof summary?.screen === "string" ? summary.screen : null,
    state_version: Number.isFinite(summary?.state_version) ? summary.state_version : null,
    run: {
      has_run: state?.run?.has_run === true,
      is_game_over: run.is_game_over === true,
      act: typeof run.act === "string" ? run.act : null,
      act_floor: Number.isFinite(run.act_floor) ? run.act_floor : null,
      total_floor: Number.isFinite(run.total_floor) ? run.total_floor : null,
      room_type: typeof run.room_type === "string" ? run.room_type : null,
      room_model: typeof run.room_model === "string" ? run.room_model : null,
      current_map_coord: isPlainObject(run.current_map_coord) ? run.current_map_coord : null
    },
    player: {
      current_hp: Number.isFinite(player.current_hp) ? player.current_hp : null,
      max_hp: Number.isFinite(player.max_hp) ? player.max_hp : null,
      block: Number.isFinite(player.block) ? player.block : 0,
      gold: Number.isFinite(player.gold) ? player.gold : null,
      powers: Array.isArray(player.powers) ? player.powers : [],
      potions: Array.isArray(player.potions) ? player.potions : [],
      relics: Array.isArray(player.relics) ? player.relics : []
    },
    surface: buildRlSurface(state, summary)
  };

  if (isPlainObject(summary?.combat)) {
    observation.combat = {
      in_progress: combat.in_progress === true,
      round_number: Number.isFinite(combat.round_number) ? combat.round_number : null,
      current_side: typeof combat.current_side === "string" ? combat.current_side : null,
      is_play_phase: combat.is_play_phase === true,
      player_actions_disabled: combat.player_actions_disabled === true,
      energy: Number.isFinite(combat.energy) ? combat.energy : null,
      max_energy: Number.isFinite(combat.max_energy) ? combat.max_energy : null,
      stars: Number.isFinite(combat.stars) ? combat.stars : null,
      hand: Array.isArray(combat.hand) ? combat.hand : [],
      draw_pile_count: Number.isFinite(combat.draw_pile_count) ? combat.draw_pile_count : null,
      draw_preview_cards: Array.isArray(combat.draw_preview_cards) ? combat.draw_preview_cards : [],
      discard_pile_count: Number.isFinite(combat.discard_pile_count)
        ? combat.discard_pile_count
        : null,
      discard_cards: Array.isArray(combat.discard_cards) ? combat.discard_cards : [],
      exhaust_pile_count: Number.isFinite(combat.exhaust_pile_count)
        ? combat.exhaust_pile_count
        : null,
      exhaust_cards: Array.isArray(combat.exhaust_cards) ? combat.exhaust_cards : [],
      play_pile_count: Number.isFinite(combat.play_pile_count) ? combat.play_pile_count : null,
      play_pile_cards: Array.isArray(combat.play_pile_cards) ? combat.play_pile_cards : [],
      summons: Array.isArray(combat.summons) ? combat.summons : [],
      enemies: Array.isArray(combat.enemies) ? combat.enemies : [],
      target_index_map: Array.isArray(combat.target_index_map) ? combat.target_index_map : []
    };
  }

  return observation;
}

function buildRlSurface(state, summary = null) {
  const screen = typeof state?.screen === "string" ? state.screen : null;

  if (state?.run_mode_selection?.visible === true) {
    return buildRlRunModeSurface(state?.run_mode_selection);
  }

  if (state?.character_selection?.visible === true) {
    return buildRlCharacterSelectionSurface(state?.character_selection);
  }

  if (state?.main_menu?.visible === true || screen === "MAIN_MENU") {
    return buildRlMainMenuSurface(state?.main_menu);
  }

  const rewardBundle = buildRewardBundle(state);
  if (rewardBundle.card_reward_selection.visible) {
    return {
      type: "card_reward",
      options: rewardBundle.card_reward_selection.options
        .map((option) => summarizeIndexedOptionForAgent("card_reward", option))
        .filter((option) => option !== null),
      can_skip: rewardBundle.card_reward_selection.skip_visible === true
    };
  }

  if (rewardBundle.in_reward_flow) {
    return {
      type: "reward",
      options: rewardBundle.rewards.entries
        .map((entry) => summarizeIndexedOptionForAgent("reward", entry))
        .filter((entry) => entry !== null),
      has_proceed: rewardBundle.has_proceed === true
    };
  }

  const cardSelectionBundle = buildCardSelectionBundle(state);
  if (cardSelectionBundle.in_card_selection_flow) {
    return {
      type: "card_selection",
      prompt: normalizeAgentText(cardSelectionBundle.card_selection.prompt),
      selected_count: Number.isInteger(cardSelectionBundle.card_selection.selected_count)
        ? cardSelectionBundle.card_selection.selected_count
        : null,
      min_select: Number.isInteger(cardSelectionBundle.card_selection.min_select)
        ? cardSelectionBundle.card_selection.min_select
        : null,
      max_select: Number.isInteger(cardSelectionBundle.card_selection.max_select)
        ? cardSelectionBundle.card_selection.max_select
        : null,
      options: cardSelectionBundle.card_selection.options
        .map((option) => summarizeIndexedOptionForAgent("card_selection", option))
        .filter((option) => option !== null)
    };
  }

  const restSiteBundle = buildRestSiteBundle(state);
  if (restSiteBundle.deck_upgrade_selection.visible) {
    return {
      type: "deck_upgrade",
      options: restSiteBundle.deck_upgrade_selection.options
        .map((option) => summarizeIndexedOptionForAgent("deck_upgrade", option))
        .filter((option) => option !== null)
    };
  }

  if (restSiteBundle.in_rest_site_flow) {
    return {
      type: "rest_site",
      prompt: normalizeAgentText(restSiteBundle.rest_site.header),
      options: restSiteBundle.rest_site.options
        .map((option) => summarizeIndexedOptionForAgent("rest_site", option))
        .filter((option) => option !== null)
    };
  }

  if (isCrystalSphereScreen(screen)) {
    return {
      type: "event_crystal_sphere",
      details: summarizeCrystalSphereForAgent(state?.crystal_sphere, state?.event_options?.options)
    };
  }

  if (state?.event_options?.visible === true && !isRlTransientRoomEntryContinueState(state)) {
    return {
      type: "event",
      options: Array.isArray(state.event_options.options)
        ? state.event_options.options
          .map(summarizeEventOptionForAgent)
          .filter((option) => option !== null)
        : []
    };
  }

  if (isMapReadyState(state)) {
    const map = isPlainObject(summary?.map) ? summary.map : {};
    return {
      type: "map",
      current_coord: isPlainObject(map.current_coord) ? map.current_coord : null,
      travelable_points: Array.isArray(map.travelable_points) ? map.travelable_points : []
    };
  }

  if (
    state?.combat?.in_progress === true &&
    state?.combat?.current_side === "Player" &&
    state?.combat?.is_play_phase === true &&
    state?.combat?.player_actions_disabled !== true
  ) {
    return {
      type: "combat"
    };
  }

  const shopBundle = buildShopBundle(state);
  if (shopBundle.in_shop_flow) {
    return {
      type: "shop",
      is_open: shopBundle.shop.is_open === true,
      gold: Number.isFinite(shopBundle.shop.gold) ? shopBundle.shop.gold : null,
      items: shopBundle.shop.items
        .map((item) => summarizeShopItemForAgent(item))
        .filter((item) => item !== null)
    };
  }

  if (isRlTerminalState(state)) {
    return {
      type: "terminal",
      outcome: getRlTerminalOutcome(state)
    };
  }

  return {
    type: "unknown"
  };
}

function buildRlMainMenuSurface(mainMenu) {
  const options = [];
  if (isPlainObject(mainMenu?.continue_button) && mainMenu.continue_button.visible === true) {
    options.push({
      semantic_action: "continue",
      text: normalizeAgentText(mainMenu.continue_button.text)
    });
  }

  if (Array.isArray(mainMenu?.buttons)) {
    for (const button of mainMenu.buttons) {
      if (!isPlainObject(button) || button.visible !== true) {
        continue;
      }

      options.push({
        semantic_action:
          typeof button.semantic_action === "string" ? button.semantic_action : null,
        text: normalizeAgentText(button.text)
      });
    }
  }

  if (mainMenu?.abandon_confirm?.visible === true && Array.isArray(mainMenu.abandon_confirm.buttons)) {
    for (const button of mainMenu.abandon_confirm.buttons) {
      if (!isPlainObject(button) || button.visible !== true) {
        continue;
      }

      options.push({
        semantic_action:
          typeof button.semantic_action === "string" ? button.semantic_action : null,
        text: normalizeAgentText(button.text)
      });
    }
  }

  return {
    type: "main_menu",
    options
  };
}

function buildRlRunModeSurface(runModeSelection) {
  const options = Array.isArray(runModeSelection?.options)
    ? runModeSelection.options
      .filter((option) => isPlainObject(option) && option.visible === true)
      .map((option) => ({
        semantic_action:
          typeof option.semantic_action === "string" ? option.semantic_action : null,
        text: normalizeAgentText(option.text ?? option.label)
      }))
    : [];

  if (isPlainObject(runModeSelection?.back_button) && runModeSelection.back_button.visible === true) {
    options.push({
      semantic_action:
        typeof runModeSelection.back_button.semantic_action === "string"
          ? runModeSelection.back_button.semantic_action
          : "back",
      text: normalizeAgentText(runModeSelection.back_button.text ?? runModeSelection.back_button.label)
    });
  }

  return {
    type: "run_mode_selection",
    options
  };
}

function buildRlCharacterSelectionSurface(characterSelection) {
  const options = Array.isArray(characterSelection?.options)
    ? characterSelection.options
      .filter((option) => isPlainObject(option))
      .map((option) => ({
        index: Number.isInteger(option.index) ? option.index : null,
        title: normalizeAgentText(option?.character?.title),
        id: typeof option?.character?.id === "string" ? option.character.id : null,
        is_selected: option.is_selected === true,
        is_locked: option.is_locked === true,
        is_random: option.is_random === true
      }))
    : [];

  return {
    type: "character_selection",
    selected_index: Number.isInteger(characterSelection?.selected_index)
      ? characterSelection.selected_index
      : null,
    selected_character: normalizeAgentText(characterSelection?.selected_character?.title),
    can_embark: characterSelection?.can_embark === true,
    options
  };
}

function buildRlLegalActions(state) {
  return getNonAutomationActions(state).map((action, idx) => summarizeRlLegalAction(action, idx));
}

function summarizeRlLegalAction(action, idx) {
  const baseSummary = summarizeActionForAgent(action) ?? {};
  const summary = {
    idx,
    key: typeof action?.action_id === "string" ? action.action_id : String(idx),
    action_id: typeof action?.action_id === "string" ? action.action_id : null,
    kind: typeof action?.kind === "string" ? action.kind : null,
    label: normalizeAgentText(action?.label),
    ...baseSummary
  };

  if (Number.isInteger(action?.hand_index)) {
    summary.hand_index = action.hand_index;
  }

  if (Number.isInteger(action?.index)) {
    summary.index = action.index;
  }

  if (Number.isInteger(action?.slot_index)) {
    summary.slot_index = action.slot_index;
  }

  if (action?.skip === true) {
    summary.skip = true;
  }

  if (isPlainObject(action?.card)) {
    summary.card = summarizeCardForAgent(action.card);
    if (typeof action.card.title === "string" && action.card.title.trim()) {
      summary.title = normalizeAgentText(action.card.title);
    }
    if (Number.isInteger(action.card.resolved_energy_cost)) {
      summary.cost = action.card.resolved_energy_cost;
    }
    const starCost = readAgentCardStarCost(action.card);
    if (starCost !== null) {
      summary.star_cost = starCost;
    }
  } else if (isPlainObject(action?.potion)) {
    summary.potion = summarizePotionForAgent(action.potion);
    if (typeof action.potion.title === "string" && action.potion.title.trim()) {
      summary.title = normalizeAgentText(action.potion.title);
    }
  }

  return summary;
}

function resolveRlStepAction(state, args) {
  const legalActions = buildRlLegalActions(state);
  const actionEncoding = normalizeRlActionEncoding(args);
  if (actionEncoding === "action_id") {
    const requestedActionId = requireNonEmptyString(args.action_id, "action_id");
    const matched = legalActions.find((action) => action.action_id === requestedActionId) ?? null;
    if (matched) {
      return matched;
    }

    throw new ToolPayloadError(
      "rl_action_not_available",
      `Action '${requestedActionId}' is not currently legal.`,
      {
        action_id: requestedActionId,
        legal_actions: legalActions
      }
    );
  }

  const requestedIndex = optionalInteger(args.action_index, "action_index");
  if (!Number.isInteger(requestedIndex)) {
    throw new ToolPayloadError(
      "invalid_arguments",
      "action_index must be provided when action_encoding is legal_action_idx.",
      {
        field: "action_index"
      }
    );
  }

  const matched = legalActions.find((action) => action.idx === requestedIndex) ?? null;
  if (matched) {
    return matched;
  }

  throw new ToolPayloadError(
    "rl_action_not_available",
    `Action index ${requestedIndex} is out of range for the current legal action set.`,
    {
      action_index: requestedIndex,
      legal_actions: legalActions
    }
  );
}

function normalizeRlActionEncoding(args) {
  const normalized = optionalString(args?.action_encoding, "action_encoding");
  if (normalized === undefined || normalized === null || normalized.length <= 0) {
    return typeof args?.action_id === "string" && args.action_id.trim()
      ? "action_id"
      : "legal_action_idx";
  }

  if (RL_ENV_ACTION_ENCODING_ENUM.includes(normalized)) {
    return normalized;
  }

  throw new ToolPayloadError(
    "invalid_arguments",
    `action_encoding must be one of ${RL_ENV_ACTION_ENCODING_ENUM.join(", ")}.`,
    {
      field: "action_encoding"
    }
  );
}

function requireActiveRlEpisode(episodeId) {
  if (!isPlainObject(activeRlEpisode)) {
    throw new ToolPayloadError(
      "no_active_episode",
      "No active RL episode is currently registered. Call sts2_env_reset first."
    );
  }

  if (activeRlEpisode.id !== episodeId) {
    throw new ToolPayloadError(
      "episode_id_mismatch",
      `Episode '${episodeId}' is not the active RL episode.`,
      {
        episode_id: episodeId,
        active_episode_id: activeRlEpisode.id
      }
    );
  }

  return activeRlEpisode;
}

function createRlEpisode(config = {}) {
  rlEpisodeCounter += 1;
  activeRlEpisode = {
    id: `rl_ep_${rlEpisodeCounter.toString().padStart(4, "0")}`,
    step_index: 0,
    started_at_utc: new Date().toISOString(),
    done: false,
    config: {
      character: typeof config.character === "string" ? config.character : null
    }
  };
  return activeRlEpisode;
}

function chooseRlResetAction(state, requestedCharacter) {
  if (!isPlainObject(state)) {
    return null;
  }

  const legalActions = getNonAutomationActions(state);
  if (state?.run_mode_selection?.visible === true) {
    return findRlResetActionById(legalActions, "run_mode:standard");
  }

  if (state?.character_selection?.visible === true) {
    if (typeof requestedCharacter === "string" && requestedCharacter.trim()) {
      const selectedOption = findRlCharacterOptionMatch(state, requestedCharacter);
      if (!selectedOption) {
        throw new ToolPayloadError(
          "unknown_character",
          `Character '${requestedCharacter}' is not visible on the current character selection screen.`,
          {
            character: requestedCharacter,
            available_characters: summarizeRlAvailableCharacters(state)
          }
        );
      }

      if (selectedOption.is_selected !== true) {
        return findRlResetActionById(legalActions, `character_select:${selectedOption.index}`);
      }
    }

    return (
      findRlResetActionById(legalActions, "embark") ??
      legalActions.find(
        (action) =>
          typeof action?.action_id === "string" && action.action_id.startsWith("character_select:")
      ) ??
      null
    );
  }

  const mainMenuVisible = state?.main_menu?.visible === true || state?.screen === "MAIN_MENU";
  if (mainMenuVisible) {
    return (
      findRlResetActionById(legalActions, "main_menu:abandon_current_game") ??
      findRlResetActionById(legalActions, "main_menu:confirm_abandon_run") ??
      findRlResetActionById(legalActions, "main_menu:new_game") ??
      findRlResetActionById(legalActions, "main_menu:singleplayer")
    );
  }

  return null;
}

function findRlResetActionById(actions, actionId) {
  if (!Array.isArray(actions) || typeof actionId !== "string") {
    return null;
  }

  const matched = actions.find((action) => action?.action_id === actionId) ?? null;
  if (!matched) {
    return null;
  }

  return {
    action_id: matched.action_id,
    label: normalizeAgentText(matched.label)
  };
}

function findRlCharacterOptionMatch(state, requestedCharacter) {
  const options = Array.isArray(state?.character_selection?.options)
    ? state.character_selection.options.filter((option) => isPlainObject(option))
    : [];
  if (options.length <= 0) {
    return null;
  }

  const normalizedRequested = normalizeRlLooseText(requestedCharacter);
  return (
    options.find((option) => {
      const title = normalizeRlLooseText(option?.character?.title);
      const id = normalizeRlLooseText(option?.character?.id);
      return normalizedRequested === title || normalizedRequested === id;
    }) ?? null
  );
}

function summarizeRlAvailableCharacters(state) {
  return Array.isArray(state?.character_selection?.options)
    ? state.character_selection.options
      .filter((option) => isPlainObject(option))
      .map((option) => ({
        index: Number.isInteger(option.index) ? option.index : null,
        title: normalizeAgentText(option?.character?.title),
        id: typeof option?.character?.id === "string" ? option.character.id : null,
        is_locked: option.is_locked === true,
        is_random: option.is_random === true
      }))
    : [];
}

function isRlStartupSurfaceState(state) {
  return (
    isPlainObject(state) &&
    (state?.main_menu?.visible === true ||
      state?.screen === "MAIN_MENU" ||
      state?.run_mode_selection?.visible === true ||
      state?.character_selection?.visible === true)
  );
}

function isRlActiveRunState(state) {
  return state?.run?.has_run === true && state?.run?.is_game_over !== true;
}

function isRlFreshEpisodeReadyState(state) {
  return (
    isRlActiveRunState(state) &&
    !isRlStartupSurfaceState(state) &&
    summarizeActionableWaitState(state).actionable
  );
}

function isRlTerminalState(state) {
  return state?.run?.has_run === true && state?.run?.is_game_over === true;
}

function getRlTerminalOutcome(state) {
  if (!isRlTerminalState(state)) {
    return null;
  }

  const currentHp = Number.isFinite(state?.players?.[0]?.creature?.current_hp)
    ? state.players[0].creature.current_hp
    : null;
  if (currentHp !== null && currentHp <= 0) {
    return "death";
  }

  return "victory";
}

function getRlSurfaceType(state) {
  const surface = buildRlSurface(state);
  return typeof surface?.type === "string" ? surface.type : "unknown";
}

function summarizeRlRunContext(state) {
  return {
    has_run: state?.run?.has_run === true,
    is_game_over: state?.run?.is_game_over === true,
    total_floor: Number.isInteger(state?.run?.total_floor) ? state.run.total_floor : null,
    act_floor: Number.isInteger(state?.run?.act_floor) ? state.run.act_floor : null
  };
}

function buildRlRewardBreakdown(beforeState, afterState) {
  const hpBefore = Number.isFinite(beforeState?.players?.[0]?.creature?.current_hp)
    ? beforeState.players[0].creature.current_hp
    : null;
  const hpAfter = Number.isFinite(afterState?.players?.[0]?.creature?.current_hp)
    ? afterState.players[0].creature.current_hp
    : null;
  const maxHp =
    (Number.isFinite(afterState?.players?.[0]?.creature?.max_hp)
      ? afterState.players[0].creature.max_hp
      : null) ??
    (Number.isFinite(beforeState?.players?.[0]?.creature?.max_hp)
      ? beforeState.players[0].creature.max_hp
      : null);
  const hpDelta =
    Number.isFinite(hpBefore) && Number.isFinite(hpAfter) ? hpAfter - hpBefore : 0;
  const hpDeltaNormalized =
    Number.isFinite(maxHp) && maxHp > 0 ? hpDelta / maxHp : 0;
  const goldDelta = getRlPlayerGold(afterState) - getRlPlayerGold(beforeState);
  const floorDelta = getRlTotalFloor(afterState) - getRlTotalFloor(beforeState);
  const relicDelta = countRlPlayerRelics(afterState) - countRlPlayerRelics(beforeState);
  const deckDelta = countRlPlayerDeckEntries(afterState) - countRlPlayerDeckEntries(beforeState);
  const combatWin = didRlCombatWin(beforeState, afterState) ? 1 : 0;
  const death = getRlTerminalOutcome(afterState) === "death" ? 1 : 0;
  const victory = getRlTerminalOutcome(afterState) === "victory" ? 1 : 0;
  const total =
    hpDeltaNormalized * RL_REWARD_WEIGHTS.hp_delta_normalized +
    goldDelta * RL_REWARD_WEIGHTS.gold_delta +
    floorDelta * RL_REWARD_WEIGHTS.floor_delta +
    relicDelta * RL_REWARD_WEIGHTS.relic_delta +
    deckDelta * RL_REWARD_WEIGHTS.deck_delta +
    combatWin * RL_REWARD_WEIGHTS.combat_win +
    death * RL_REWARD_WEIGHTS.death +
    victory * RL_REWARD_WEIGHTS.victory;

  return {
    hp_delta: hpDelta,
    hp_delta_normalized: roundRlNumber(hpDeltaNormalized),
    gold_delta: goldDelta,
    floor_delta: floorDelta,
    relic_delta: relicDelta,
    deck_delta: deckDelta,
    combat_win: combatWin,
    death,
    victory,
    total: roundRlNumber(total)
  };
}

function didRlCombatWin(beforeState, afterState) {
  const beforeCombat = beforeState?.combat?.in_progress === true;
  const afterCombat = afterState?.combat?.in_progress === true;
  return (
    beforeCombat &&
    !afterCombat &&
    afterState?.run?.is_game_over !== true &&
    afterState?.run?.current_room?.is_pre_finished === true
  );
}

function getRlPlayerGold(state) {
  return Number.isFinite(state?.players?.[0]?.gold) ? state.players[0].gold : 0;
}

function getRlTotalFloor(state) {
  return Number.isFinite(state?.run?.total_floor) ? state.run.total_floor : 0;
}

function countRlPlayerRelics(state) {
  return Array.isArray(state?.players?.[0]?.relics) ? state.players[0].relics.length : 0;
}

function countRlPlayerDeckEntries(state) {
  if (Number.isInteger(state?.players?.[0]?.deck?.count)) {
    return state.players[0].deck.count;
  }

  return Array.isArray(state?.players?.[0]?.deck?.cards) ? state.players[0].deck.cards.length : 0;
}

function summarizeRlBridgeActionInfo(result) {
  if (!isPlainObject(result)) {
    return null;
  }

  const info = {
    screen_after:
      typeof result?.state?.screen === "string"
        ? result.state.screen
        : typeof result.screen_after === "string"
          ? result.screen_after
          : null,
    state_version_after: Number.isInteger(result.state_version_after)
      ? result.state_version_after
      : getStateVersionValue(result.state)
  };

  if (isPlainObject(result.recovered_from_state_version_conflict)) {
    info.recovered_from_state_version_conflict = result.recovered_from_state_version_conflict;
  }

  if (result.post_action_settled === true || typeof result.post_action_settle_reason === "string") {
    info.post_action_settle_reason =
      typeof result.post_action_settle_reason === "string"
        ? result.post_action_settle_reason
        : null;
  }

  return info;
}

function normalizeRlLooseText(value) {
  if (typeof value !== "string") {
    return "";
  }

  return value.replace(/\s+/g, "").replace(/[^\p{L}\p{N}]+/gu, "").toLowerCase();
}

function roundRlNumber(value) {
  return Number.isFinite(value) ? Number(value.toFixed(6)) : 0;
}

function getRemainingTimeoutMs(startedAt, timeoutMs) {
  const elapsed = Date.now() - startedAt;
  if (elapsed >= timeoutMs) {
    throw new ToolPayloadError(
      "timeout",
      `Operation exceeded the timeout budget of ${timeoutMs} ms.`,
      {
        timeout_ms: timeoutMs
      }
    );
  }

  return timeoutMs - elapsed;
}

async function resolveShopVisitTool(args) {
  try {
    const finalize = (payload) =>
      asPrecompactedToolResult(compactShopVisitPayload(attachInteractionHints(payload)), false);
    const purchases = normalizeShopPurchaseRequests(args.purchases);
    const removeCardTitle = optionalString(args.remove_card_title, "remove_card_title");
    const removeCardIndex = optionalInteger(args.remove_card_index, "remove_card_index");
    const openShop = optionalBoolean(args.open_shop, "open_shop") ?? true;
    const closeInventory =
      optionalBoolean(args.close_inventory, "close_inventory") ?? true;
    const leaveShop = optionalBoolean(args.leave_shop, "leave_shop") ?? true;
    const waitAfterMs = clampInteger(
      args.wait_after_ms,
      DEFAULT_ACTION_WAIT_MS,
      0,
      MAX_ACTION_WAIT_MS,
      "wait_after_ms"
    );

    if (removeCardTitle !== undefined && removeCardIndex !== undefined) {
      throw new ToolPayloadError(
        "invalid_arguments",
        "remove_card_title and remove_card_index cannot both be provided.",
        {
          fields: ["remove_card_title", "remove_card_index"]
        }
      );
    }

    const session = getLiveSession();
    let state = await getBridgeState(session);
    let shopBundle = buildShopBundle(state);
    const initialShopBundle = shopBundle;
    let cardSelectionBundle = buildCardSelectionBundle(state);

    if (!shopBundle.in_shop_flow && !cardSelectionBundle.in_card_selection_flow) {
      return finalize({
        ok: false,
        resolved: false,
        reason: "not_in_shop_flow",
        shop_bundle: initialShopBundle,
        state
      });
    }

    const executedActions = [];
    const purchasedItems = [];
    let removedCard = null;

    if (cardSelectionBundle.in_card_selection_flow) {
      const cardSelectionResolution = await maybeResolveShopCardRemovalSelection(
        session,
        state,
        {
          removeCardTitle,
          removeCardIndex,
          waitAfterMs
        }
      );
      if (cardSelectionResolution.failed) {
        return finalize({
          ok: false,
          resolved: false,
          reason: cardSelectionResolution.reason,
          message: cardSelectionResolution.message,
          removed_card: removedCard,
          executed_actions: executedActions,
          card_selection_bundle: cardSelectionResolution.card_selection_bundle,
          available_cards: cardSelectionResolution.available_cards,
          state: cardSelectionResolution.state
        });
      }
      if (cardSelectionResolution.required_but_missing) {
        return finalize({
          ok: true,
          resolved: false,
          reason: "shop_card_removal_choice_required",
          removed_card: removedCard,
          executed_actions: executedActions,
          card_selection_bundle: cardSelectionResolution.card_selection_bundle,
          state: cardSelectionResolution.state
        });
      }
      if (cardSelectionResolution.performed) {
        removedCard = cardSelectionResolution.removed_card;
        executedActions.push(...cardSelectionResolution.executed_actions);
        state = cardSelectionResolution.state;
        shopBundle = buildShopBundle(state);
        cardSelectionBundle = buildCardSelectionBundle(state);
      }
    }

    if (openShop && shopBundle.shop.visible && shopBundle.shop.is_open !== true) {
      if (!shopBundle.non_automation_action_ids.includes("shop:open")) {
        return finalize({
          ok: false,
          resolved: false,
          reason: "shop_open_action_unavailable",
          shop_bundle: shopBundle,
          state
        });
      }

      const openResult = await performBridgeAction(session, "shop:open", waitAfterMs);
      executedActions.push(summarizeExecutedAction(openResult));
      state = openResult.state;
      shopBundle = buildShopBundle(state);
    }

    const purchasePlan = [];
    for (let sequenceIndex = 0; sequenceIndex < purchases.length; sequenceIndex += 1) {
      const request = purchases[sequenceIndex];
      const matchedAction = resolveInitialShopPurchaseRequest(request, state);
      if (!matchedAction) {
        return finalize({
          ok: false,
          resolved: false,
          reason: "shop_purchase_request_unavailable",
          sequence_index: sequenceIndex,
          request,
          shop_bundle: shopBundle,
          available_shop_actions: getShopBuyActions(state).map(summarizeShopBuyActionForAgent),
          state
        });
      }

      purchasePlan.push(
        buildPlannedShopPurchaseStep(matchedAction, request, sequenceIndex)
      );
    }
    const purchasePlanOutput = purchasePlan.map(summarizeShopPurchasePlanStep);

    for (let sequenceIndex = 0; sequenceIndex < purchasePlan.length; sequenceIndex += 1) {
      shopBundle = buildShopBundle(state);
      const blocker = getShopVisitContinuationBlocker(state);
      if (blocker) {
        return finalize({
          ok: true,
          resolved: false,
          reason: blocker.reason,
          message: blocker.message,
          next_sequence_index: sequenceIndex,
          purchase_plan: purchasePlanOutput,
          purchased_items: purchasedItems,
          removed_card: removedCard,
          executed_actions: executedActions,
          shop_bundle: shopBundle,
          state
        });
      }

      const planStep = purchasePlan[sequenceIndex];
      const currentShopActions = getShopBuyActions(state);
      const resolution = resolvePlannedShopPurchaseStep(planStep, currentShopActions);
      if (!resolution.action) {
        return finalize({
          ok: true,
          resolved: false,
          reason: "shop_purchase_unavailable_after_reindex",
          next_sequence_index: sequenceIndex,
          failed_step: summarizeShopPurchasePlanStep(planStep),
          purchase_plan: purchasePlanOutput,
          purchased_items: purchasedItems,
          removed_card: removedCard,
          executed_actions: executedActions,
          available_shop_actions: currentShopActions.map(summarizeShopBuyActionForAgent),
          shop_bundle: shopBundle,
          state
        });
      }

      const purchaseResult = await performBridgeAction(
        session,
        resolution.action.action_id,
        waitAfterMs
      );
      executedActions.push(summarizeExecutedAction(purchaseResult));
      purchasedItems.push({
        sequence_index: sequenceIndex,
        request: summarizeShopPurchaseRequest(planStep.request),
        requested_action_id: planStep.requested_action_id,
        executed_action_id: resolution.action.action_id,
        match_type: resolution.match_type,
        compatible_candidate_count: resolution.compatible_candidate_count,
        item: summarizeShopItemForAgent(resolution.action.item)
      });
      state = purchaseResult.state;

      const cardSelectionResolution = await maybeResolveShopCardRemovalSelection(
        session,
        state,
        {
          removeCardTitle,
          removeCardIndex,
          waitAfterMs
        }
      );
      if (cardSelectionResolution.required_but_missing) {
        return finalize({
          ok: true,
          resolved: false,
          reason: "shop_card_removal_choice_required",
          purchase_plan: purchasePlanOutput,
          purchased_items: purchasedItems,
          removed_card: removedCard,
          executed_actions: executedActions,
          card_selection_bundle: cardSelectionResolution.card_selection_bundle,
          state
        });
      }

      if (cardSelectionResolution.failed) {
        return finalize({
          ok: false,
          resolved: false,
          reason: cardSelectionResolution.reason,
          message: cardSelectionResolution.message,
          purchase_plan: purchasePlanOutput,
          purchased_items: purchasedItems,
          removed_card: removedCard,
          executed_actions: executedActions,
          card_selection_bundle: cardSelectionResolution.card_selection_bundle,
          available_cards: cardSelectionResolution.available_cards,
          state: cardSelectionResolution.state
        });
      }

      if (cardSelectionResolution.performed) {
        removedCard = cardSelectionResolution.removed_card;
        executedActions.push(...cardSelectionResolution.executed_actions);
        state = cardSelectionResolution.state;
      }
    }

    shopBundle = buildShopBundle(state);

    if (closeInventory && shopBundle.shop.is_open === true) {
      if (shopBundle.non_automation_action_ids.includes("shop:back")) {
        const backResult = await performBridgeAction(session, "shop:back", waitAfterMs);
        executedActions.push(summarizeExecutedAction(backResult));
        state = backResult.state;
        shopBundle = buildShopBundle(state);
      }
    }

    if (leaveShop) {
      if (shopBundle.non_automation_action_ids.includes("shop:leave")) {
        const leaveResult = await performBridgeAction(session, "shop:leave", waitAfterMs);
        executedActions.push(summarizeExecutedAction(leaveResult));
        state = leaveResult.state;
        shopBundle = buildShopBundle(state);
      }
    }

    return finalize({
      ok: true,
      resolved: true,
      purchase_plan: purchasePlanOutput,
      purchased_items: purchasedItems,
      removed_card: removedCard,
      shop_bundle: initialShopBundle,
      executed_actions: executedActions,
      final_state: state
    });
  } catch (error) {
    return asToolResult(toolErrorPayload(error), true);
  }
}

async function getBridgeState(session) {
  return await getLatestBridgeState(session);
}

async function waitForStableMapRouteSnapshot(session) {
  let snapshot = await captureMapRouteSnapshot(session);
  let previousFingerprint = snapshot.fingerprint;
  let stablePolls = 0;

  const initialVerdict = getMapRouteSnapshotVerdict(snapshot, stablePolls);
  if (initialVerdict.settled) {
      return {
        ...initialVerdict,
        state: snapshot.state,
        frontierKeys: snapshot.frontierKeys,
        mapActionKeys: snapshot.mapActionKeys,
        currentCoordKey: snapshot.currentCoordKey,
        frontier_action_match: snapshot.frontier_action_match
      };
  }

  const startedAt = Date.now();
  let pollCount = 0;
  while (Date.now() - startedAt < MAP_ROUTE_SETTLE_TIMEOUT_MS) {
    await delay(MAP_ROUTE_SETTLE_POLL_INTERVAL_MS);
    pollCount += 1;
    snapshot = await captureMapRouteSnapshot(session);

    stablePolls =
      snapshot.fingerprint === previousFingerprint ? stablePolls + 1 : 0;
    previousFingerprint = snapshot.fingerprint;

    const verdict = getMapRouteSnapshotVerdict(snapshot, stablePolls);
    if (verdict.settled) {
        return {
          ...verdict,
          poll_count: pollCount,
          state: snapshot.state,
          frontierKeys: snapshot.frontierKeys,
          mapActionKeys: snapshot.mapActionKeys,
          currentCoordKey: snapshot.currentCoordKey,
          frontier_action_match: snapshot.frontier_action_match
        };
    }
  }

  return {
    settled: false,
    reason: "timeout",
    poll_count: pollCount,
    state: snapshot.state,
    frontierKeys: snapshot.frontierKeys,
    mapActionKeys: snapshot.mapActionKeys,
    currentCoordKey: snapshot.currentCoordKey,
    frontier_action_match: snapshot.frontier_action_match
  };
}

async function captureMapRouteSnapshot(session) {
  const state = await getBridgeState(session);
  return buildMapRouteSnapshot(state);
}

function buildMapRouteSnapshot(state) {
  const frontierKeys = extractTravelableMapKeysFromState(state);
  const mapActionKeys = extractMapActionKeys(state);
  const currentCoord = extractMapCurrentCoord(state);
  const currentCoordKey = toCoordKey(currentCoord);
  const stateVersion = Number.isInteger(state?.state_version)
    ? state.state_version
    : null;
  const frontierActionMatch = areCoordKeySetsEqual(frontierKeys, mapActionKeys);
  const fingerprint = JSON.stringify({
    screen: typeof state?.screen === "string" ? state.screen : null,
    map_open: state?.map?.is_open === true,
    map_traveling: state?.map?.is_traveling === true,
    current_coord_key: currentCoordKey,
    frontier_keys: frontierKeys,
    map_action_keys: mapActionKeys,
    state_version: stateVersion
  });

  return {
    state,
    frontierKeys,
    mapActionKeys,
    currentCoordKey,
    frontier_action_match: frontierActionMatch,
    fingerprint
  };
}

function getMapRouteSnapshotVerdict(snapshot, stablePolls) {
  const state = snapshot?.state;
  if (!isPlainObject(state)) {
    return {
      settled: false,
      reason: "missing_state"
    };
  }

  const screen = typeof state.screen === "string" ? state.screen : null;
  if (screen !== "MAP") {
    return {
      settled: false,
      reason: `waiting_for_map_screen:${screen ?? "unknown"}`
    };
  }

  if (state?.map?.is_open !== true) {
    return {
      settled: false,
      reason: "waiting_for_map_open"
    };
  }

  if (state?.map?.is_traveling === true) {
    return {
      settled: false,
      reason: "waiting_for_map_travel_finish"
    };
  }

  if (!snapshot.frontier_action_match) {
    return {
      settled: false,
      reason: "waiting_for_frontier_action_match"
    };
  }

  if (stablePolls < MAP_ROUTE_STABLE_POLL_TARGET) {
    return {
      settled: false,
      reason: `waiting_for_stable_map_snapshot:${stablePolls}/${MAP_ROUTE_STABLE_POLL_TARGET}`
    };
  }

  return {
    settled: true,
    reason: "map_snapshot_stable",
    poll_count: 0
  };
}

async function performBridgeAction(
  session,
  actionId,
  waitAfterMs,
  expectedStateVersion,
  options = {}
) {
  const strictStateVersion = options?.strictStateVersion === true;
  const settleAfterAction = options?.settleAfterAction !== false;
  let currentExpectedStateVersion = Number.isInteger(expectedStateVersion)
    ? expectedStateVersion
    : undefined;
  let recoveredFromStateVersionConflict = null;
  const bridgeEventClient = ensureBridgeEventClient(session);
  await bridgeEventClient.start();

  for (let attempt = 0; attempt <= ACTION_STATE_VERSION_RETRY_LIMIT; attempt += 1) {
    const stateVersion = await resolveExpectedActionStateVersion(
      session,
      bridgeEventClient,
      currentExpectedStateVersion
    );

    try {
      const response = await bridgeRequestJson(session, "action", {
        method: "POST",
        body: {
          action_id: actionId,
          expected_state_version: stateVersion,
          wait_after_ms: waitAfterMs
        },
        timeoutMs: DEFAULT_HTTP_TIMEOUT_MS + waitAfterMs + 5000
      });

      let settledResult = response.payload;
      settledResult.state = await resolveBridgeActionState(
        session,
        bridgeEventClient,
        settledResult
      );
      if (!Number.isInteger(settledResult?.state_version_after)) {
        settledResult.state_version_after = getStateVersionValue(settledResult.state);
      }
      if (typeof settledResult?.state?.screen === "string") {
        settledResult.screen_after = settledResult.state.screen;
      }

      if (settleAfterAction) {
        settledResult = await maybeSettleAfterAction(session, actionId, settledResult);
      }

      if (recoveredFromStateVersionConflict) {
        settledResult.recovered_from_state_version_conflict =
          recoveredFromStateVersionConflict;
      }

      return settledResult;
    } catch (error) {
      if (
        strictStateVersion ||
        !isStateVersionConflictError(error) ||
        attempt >= ACTION_STATE_VERSION_RETRY_LIMIT
      ) {
        throw error;
      }

      const latestState = await getLatestBridgeState(session, {
        force_refresh: true
      });
      const latestStateVersion = Number.isInteger(latestState?.state_version)
        ? latestState.state_version
        : null;

      if (
        latestStateVersion === null ||
        latestStateVersion === stateVersion ||
        !isActionIdCurrentlyAvailable(latestState, actionId)
      ) {
        throw error;
      }

      currentExpectedStateVersion = latestStateVersion;
      recoveredFromStateVersionConflict = {
        requested_state_version: stateVersion,
        recovered_state_version: latestStateVersion,
        retry_count: attempt + 1
      };
    }
  }

  throw new ToolPayloadError(
    "bridge_action_retry_exhausted",
    `Action '${actionId}' exceeded the state-version retry budget.`,
    {
      action_id: actionId,
      expected_state_version: expectedStateVersion ?? null
    }
  );
}

async function resolveExpectedActionStateVersion(
  session,
  bridgeEventClient,
  currentExpectedStateVersion
) {
  if (Number.isInteger(currentExpectedStateVersion)) {
    return currentExpectedStateVersion;
  }

  const cachedStateVersion = getStateVersionValue(
    bridgeEventClient.getCachedState(Number.POSITIVE_INFINITY)
  );
  if (Number.isInteger(cachedStateVersion)) {
    return cachedStateVersion;
  }

  return getStateVersionValue(await getLatestBridgeState(session));
}

async function resolveBridgeActionState(session, bridgeEventClient, actionResult) {
  if (isPlainObject(actionResult?.state)) {
    return bridgeEventClient.ingestState(actionResult.state) ?? actionResult.state;
  }

  const targetStateVersion = Number.isInteger(actionResult?.state_version_after)
    ? actionResult.state_version_after
    : null;
  if (targetStateVersion !== null) {
    const cachedState = bridgeEventClient.getCachedState(Number.POSITIVE_INFINITY);
    if (getStateVersionValue(cachedState) >= targetStateVersion) {
      return cachedState;
    }

    try {
      return await bridgeEventClient.waitForState({
        timeout_ms: ACTION_STATE_SYNC_TIMEOUT_MS,
        predicate: (candidateState) =>
          getStateVersionValue(candidateState) >= targetStateVersion
      });
    } catch (error) {
      if (!(error instanceof ToolPayloadError) || error.code !== "bridge_event_wait_timeout") {
        throw error;
      }
    }
  }

  return await getLatestBridgeState(session, {
    force_refresh: true
  });
}

async function maybeSettleAfterAction(session, actionId, result) {
  const settleStrategy = getPostActionSettleStrategy(actionId);
  if (!settleStrategy) {
    return result;
  }

  let settled = null;
  if (settleStrategy === "end_turn") {
    settled = await waitForEndTurnSettlement(session, result?.state ?? null);
  } else if (settleStrategy === "combat_action") {
    settled = await waitForCombatActionSettlement(session, result?.state ?? null);
  } else if (settleStrategy === "screen_transition") {
    settled = await waitForScreenTransitionSettlement(session, actionId, result?.state ?? null);
  } else if (settleStrategy === "map_travel") {
    settled = await waitForMapTravelSettlement(session, result?.state ?? null);
  }

  if (!settled) {
    return result;
  }

  return {
    ...result,
    state: settled.state,
    screen_after:
      typeof settled?.state?.screen === "string"
        ? settled.state.screen
        : result?.screen_after ?? null,
    state_version_after:
      settled?.state?.state_version ?? result?.state_version_after ?? null,
    state_hash_after:
      settled?.state?.state_hash ?? result?.state_hash_after ?? null,
    post_action_settled: settled.settled,
    post_action_settle_reason: settled.reason,
    post_action_settle_polls: settled.poll_count
  };
}

function getSettleTimeoutMs(strategy) {
  switch (strategy) {
    case "end_turn":
      return END_TURN_SETTLE_TIMEOUT_MS;
    case "combat_action":
      return COMBAT_ACTION_SETTLE_TIMEOUT_MS;
    case "screen_transition":
    case "map_travel":
      return ROOM_EXIT_SETTLE_TIMEOUT_MS;
    default:
      return DEFAULT_WAIT_TIMEOUT_MS;
  }
}

function getSettleQuietWindowMs(strategy) {
  switch (strategy) {
    case "end_turn":
      return END_TURN_SETTLE_QUIET_WINDOW_MS;
    case "combat_action":
      return COMBAT_ACTION_SETTLE_QUIET_WINDOW_MS;
    case "screen_transition":
    case "map_travel":
      return POST_ACTION_SETTLE_QUIET_WINDOW_MS;
    default:
      return POST_ACTION_SETTLE_QUIET_WINDOW_MS;
  }
}

function getSettleStablePollTarget(strategy) {
  switch (strategy) {
    case "end_turn":
      return END_TURN_STABLE_POLL_TARGET;
    case "combat_action":
    case "map_travel":
      return COMBAT_ACTION_STABLE_POLL_TARGET;
    default:
      return 0;
  }
}

function evaluatePostActionSettlement(strategy, actionId, state, initialScreen, stablePolls) {
  switch (strategy) {
    case "end_turn":
      return getEndTurnSettlementVerdict(state, stablePolls);
    case "combat_action":
      return getCombatActionSettlementVerdict(state, stablePolls);
    case "screen_transition":
      return getScreenTransitionSettlementVerdict(
        actionId,
        state,
        initialScreen,
        stablePolls
      );
    case "map_travel":
      return getMapTravelSettlementVerdict(state, stablePolls);
    default:
      return {
        settled: true,
        reason: "unsupported_settle_strategy"
      };
  }
}

async function waitForNextStateAfterVersion(session, afterStateVersion, timeoutMs) {
  if (!Number.isInteger(afterStateVersion) || timeoutMs <= 0) {
    return null;
  }

  try {
    return await waitForBridgeStateEvent(session, {
      timeout_ms: timeoutMs,
      after_state_version: afterStateVersion
    });
  } catch (error) {
    if (error instanceof ToolPayloadError && error.code === "bridge_event_wait_timeout") {
      return null;
    }

    throw error;
  }
}

function getPostActionSettleStrategy(actionId) {
  if (typeof actionId !== "string" || actionId.length <= 0) {
    return null;
  }

  if (actionId === "end_turn") {
    return "end_turn";
  }

  if (actionId.startsWith("map:")) {
    return "map_travel";
  }

  if (
    actionId === "main_menu:continue" ||
    actionId === "proceed" ||
    actionId.startsWith("reward:") ||
    actionId.startsWith("card_reward:") ||
    actionId.startsWith("rest_site:") ||
    actionId.startsWith("deck_upgrade:") ||
    actionId.startsWith("event_option:")
  ) {
    return "screen_transition";
  }

  if (
    actionId.startsWith("play_card:") ||
    actionId.startsWith("card_selection:") ||
    actionId.startsWith("use_potion:")
  ) {
    return "combat_action";
  }

  return null;
}

async function waitForPostActionSettlement(session, strategy, initialState, options = {}) {
  let state = isPlainObject(initialState) ? initialState : await getBridgeState(session);
  const actionId = typeof options.actionId === "string" ? options.actionId : null;
  const initialScreen =
    typeof options.initialScreen === "string"
      ? options.initialScreen
      : typeof state?.screen === "string"
        ? state.screen
        : null;
  const timeoutMs = getSettleTimeoutMs(strategy);
  const quietWindowMs = getSettleQuietWindowMs(strategy);
  const stablePollTarget = getSettleStablePollTarget(strategy);
  const startedAt = Date.now();
  let pollCount = 0;

  while (Date.now() - startedAt < timeoutMs) {
    const immediateVerdict = evaluatePostActionSettlement(
      strategy,
      actionId,
      state,
      initialScreen,
      0
    );

    const remainingMs = Math.max(0, timeoutMs - (Date.now() - startedAt));
    if (remainingMs <= 0) {
      break;
    }

    const nextState = await waitForNextStateAfterVersion(
      session,
      getStateVersionValue(state),
      Math.min(quietWindowMs, remainingMs)
    );
    if (isPlainObject(nextState)) {
      state = nextState;
      pollCount += 1;
      continue;
    }

    if (immediateVerdict.settled) {
      return {
        settled: true,
        reason: immediateVerdict.reason,
        poll_count: pollCount,
        state
      };
    }

    const quietVerdict = evaluatePostActionSettlement(
      strategy,
      actionId,
      state,
      initialScreen,
      stablePollTarget
    );
    if (quietVerdict.settled) {
      return {
        settled: true,
        reason: quietVerdict.reason,
        poll_count: pollCount,
        state
      };
    }
  }

  return {
    settled: false,
    reason: "timeout",
    poll_count: pollCount,
    state
  };
}

async function waitForEndTurnSettlement(session, initialState) {
  return await waitForPostActionSettlement(session, "end_turn", initialState);
}

async function waitForCombatActionSettlement(session, initialState) {
  return await waitForPostActionSettlement(session, "combat_action", initialState);
}

async function waitForScreenTransitionSettlement(session, actionId, initialState) {
  return await waitForPostActionSettlement(session, "screen_transition", initialState, {
    actionId
  });
}

async function waitForMapTravelSettlement(session, initialState) {
  return await waitForPostActionSettlement(session, "map_travel", initialState);
}

function getScreenTransitionSettlementVerdict(actionId, state, initialScreen, stablePolls = 0) {
  if (!isPlainObject(state)) {
    return {
      settled: false,
      reason: "missing_state"
    };
  }

  const screen = typeof state.screen === "string" ? state.screen : null;
  const stableReason = getOutOfCombatStableStateReason(state);

  if (actionId === "main_menu:continue") {
    if (state?.main_menu?.visible === true || screen === "MAIN_MENU") {
      return {
        settled: false,
        reason: "waiting_for_main_menu_exit"
      };
    }

    if (stableReason) {
      return {
        settled: true,
        reason: stableReason
      };
    }

    return {
      settled: false,
      reason:
        screen !== null && screen !== initialScreen
          ? `waiting_for_stable_loaded_run_state:${screen}`
          : "waiting_for_loaded_run_state"
    };
  }

  if (
    actionId === "proceed" ||
    actionId.startsWith("reward:") ||
    actionId.startsWith("card_reward:")
  ) {
    if (isMapReadyState(state)) {
      return {
        settled: true,
        reason: "map_ready"
      };
    }

    const rewardReadyReason = getRewardFlowReadyReason(state);
    if (rewardReadyReason) {
      return {
        settled: true,
        reason: rewardReadyReason
      };
    }

    return {
      settled: false,
      reason:
        screen !== null && screen !== initialScreen
          ? `waiting_for_reward_transition:${screen}`
          : "waiting_for_reward_transition"
    };
  }

  if (actionId.startsWith("rest_site:") || actionId.startsWith("deck_upgrade:")) {
    if (isMapReadyState(state)) {
      return {
        settled: true,
        reason: "map_ready"
      };
    }

    const restSiteReadyReason = getRestSiteFlowReadyReason(state);
    if (restSiteReadyReason) {
      return {
        settled: true,
        reason: restSiteReadyReason
      };
    }

    const outOfCombatReason = getOutOfCombatStableStateReason(state);
    if (outOfCombatReason) {
      return {
        settled: true,
        reason: outOfCombatReason
      };
    }

    return {
      settled: false,
      reason:
        screen !== null && screen !== initialScreen
          ? `waiting_for_rest_site_transition:${screen}`
          : "waiting_for_rest_site_transition"
    };
  }

  if (actionId.startsWith("event_option:")) {
    if (isMapReadyState(state)) {
      return {
        settled: true,
        reason: "map_ready"
      };
    }

    const eventReadyReason = getEventFlowReadyReason(state);
    if (eventReadyReason) {
      return {
        settled: true,
        reason: eventReadyReason
      };
    }

    const outOfCombatReason = getOutOfCombatStableStateReason(state);
    if (outOfCombatReason) {
      return {
        settled: true,
        reason: outOfCombatReason
      };
    }

    if (screen === "COMBAT" || isPlainObject(state?.combat)) {
      return getCombatActionSettlementVerdict(state, stablePolls);
    }

    return {
      settled: false,
      reason:
        screen !== null && screen !== initialScreen
          ? `waiting_for_event_transition:${screen}`
          : "waiting_for_event_transition"
    };
  }

  return {
    settled: false,
    reason: "unsupported_transition_action"
  };
}

function getMapTravelSettlementVerdict(state, stablePolls) {
  if (!isPlainObject(state)) {
    return {
      settled: false,
      reason: "missing_state"
    };
  }

  if (typeof state.screen === "string" && state.screen === "MAP") {
    return {
      settled: false,
      reason:
        state?.map?.is_traveling === true ? "waiting_for_map_travel_finish" : "waiting_for_room_entry"
    };
  }

  const outOfCombatReason = getOutOfCombatStableStateReason(state);
  if (outOfCombatReason) {
    return {
      settled: true,
      reason: outOfCombatReason
    };
  }

  return getCombatActionSettlementVerdict(state, stablePolls);
}

function getRewardFlowReadyReason(state) {
  const rewardBundle = buildRewardBundle(state);
  if (!rewardBundle.in_reward_flow) {
    return null;
  }

  const actionIds = getNonAutomationActions(state)
    .map((action) => action?.action_id)
    .filter((actionId) => typeof actionId === "string");

  if (
    rewardBundle.card_reward_selection.visible &&
    actionIds.some((actionId) => actionId.startsWith("card_reward:"))
  ) {
    return "reward_card_selection_ready";
  }

  if (
    (rewardBundle.rewards.visible || rewardBundle.has_proceed) &&
    actionIds.some(
      (actionId) =>
        actionId.startsWith("reward:") ||
        actionId === "proceed" ||
        actionId.startsWith("discard_potion:")
    )
  ) {
    return "reward_flow_ready";
  }

  return null;
}

function getRestSiteFlowReadyReason(state) {
  const restSiteBundle = buildRestSiteBundle(state);
  if (!restSiteBundle.in_rest_site_flow) {
    return null;
  }

  const actionIds = getNonAutomationActions(state)
    .map((action) => action?.action_id)
    .filter((actionId) => typeof actionId === "string");

  if (
    restSiteBundle.deck_upgrade_selection.visible &&
    actionIds.some((actionId) => actionId.startsWith("deck_upgrade:"))
  ) {
    return "rest_site_upgrade_ready";
  }

  if (
    (restSiteBundle.rest_site.visible || actionIds.includes("rest_site:proceed")) &&
    actionIds.some(
      (actionId) => actionId === "rest_site:proceed" || actionId.startsWith("rest_site:")
    )
  ) {
    return "rest_site_ready";
  }

  return null;
}

function getEventFlowReadyReason(state) {
  if (state?.event_options?.visible !== true) {
    return null;
  }

  if (isRlTransientRoomEntryContinueState(state)) {
    return null;
  }

  return getNonAutomationActions(state).some((action) =>
    typeof action?.action_id === "string" && action.action_id.startsWith("event_option:")
  )
    ? "event_ready"
    : null;
}

function getEndTurnSettlementVerdict(state, stablePolls) {
  if (!state || typeof state !== "object") {
    return {
      settled: false,
      reason: "missing_state"
    };
  }

  const roomTransitionReason = getOutOfCombatStableStateReason(state);
  if (roomTransitionReason) {
    return {
      settled: true,
      reason: roomTransitionReason
    };
  }

  if (typeof state.screen === "string" && state.screen !== "COMBAT") {
    return {
      settled: false,
      reason: `waiting_for_room_transition:${state.screen}`
    };
  }

  const combat = isPlainObject(state.combat) ? state.combat : {};
  if (combat.in_progress !== true) {
    return {
      settled: false,
      reason: "waiting_for_room_transition"
    };
  }

  if (
    combat.current_side !== "Player" ||
    combat.is_play_phase !== true ||
    combat.player_actions_disabled === true
  ) {
    return {
      settled: false,
      reason: "waiting_for_player_turn"
    };
  }

  const handCount = getPileCount(state?.players?.[0]?.combat?.hand);
  const drawCount = getPileCount(state?.players?.[0]?.combat?.draw_pile);
  const discardCount = getPileCount(state?.players?.[0]?.combat?.discard_pile);
  const totalDrawableCount =
    handCount !== null && drawCount !== null && discardCount !== null
      ? handCount + drawCount + discardCount
      : null;
  const targetHandCount =
    totalDrawableCount !== null ? Math.min(5, totalDrawableCount) : null;
  const nonAutomationActions = getNonAutomationActions(state);
  const hasMeaningfulAction = nonAutomationActions.some(
    (action) => action?.action_id && action.action_id !== "end_turn"
  );

  if (
    handCount !== null &&
    targetHandCount !== null &&
    handCount >= targetHandCount &&
    (hasMeaningfulAction || nonAutomationActions.length > 0)
  ) {
    return {
      settled: true,
      reason: "player_turn_ready"
    };
  }

  if (
    stablePolls >= END_TURN_STABLE_POLL_TARGET &&
    handCount !== null &&
    handCount > 0 &&
    (hasMeaningfulAction || nonAutomationActions.length > 0)
  ) {
    return {
      settled: true,
      reason: "player_turn_stable_fallback"
    };
  }

  return {
    settled: false,
    reason: "waiting_for_hand_fill"
  };
}

function getCombatActionSettlementVerdict(state, stablePolls) {
  if (!state || typeof state !== "object") {
    return {
      settled: false,
      reason: "missing_state"
    };
  }

  const roomTransitionReason = getOutOfCombatStableStateReason(state);
  if (roomTransitionReason) {
    return {
      settled: true,
      reason: roomTransitionReason
    };
  }

  const cardSelectionBundle = buildCardSelectionBundle(state);
  if (cardSelectionBundle.in_card_selection_flow) {
    return {
      settled: true,
      reason: "card_selection_ready"
    };
  }

  if (typeof state.screen === "string" && state.screen !== "COMBAT") {
    return {
      settled: false,
      reason: `waiting_for_room_transition:${state.screen}`
    };
  }

  const combat = isPlainObject(state?.combat) ? state.combat : {};
  if (combat.in_progress !== true) {
    return {
      settled: false,
      reason: "waiting_for_room_transition"
    };
  }

  if (
    combat.current_side !== "Player" ||
    combat.is_play_phase !== true ||
    combat.player_actions_disabled === true
  ) {
    return {
      settled: false,
      reason: "waiting_for_combat_resolution"
    };
  }

  const nonAutomationActions = getNonAutomationActions(state);
  if (nonAutomationActions.length <= 0) {
    return {
      settled: false,
      reason: "waiting_for_action_list"
    };
  }

  const hasMeaningfulAction = nonAutomationActions.some(
    (action) => action?.action_id && action.action_id !== "end_turn"
  );
  if (stablePolls >= COMBAT_ACTION_STABLE_POLL_TARGET) {
    return {
      settled: true,
      reason: hasMeaningfulAction
        ? "player_turn_stable"
        : "player_turn_stable_end_turn_only"
    };
  }

  return {
    settled: false,
    reason: "waiting_for_stable_player_turn"
  };
}

function getOutOfCombatStableStateReason(state) {
  if (!isPlainObject(state)) {
    return null;
  }

  const cardSelectionBundle = buildCardSelectionBundle(state);
  if (cardSelectionBundle.in_card_selection_flow) {
    return "card_selection_ready";
  }

  const rewardBundle = buildRewardBundle(state);
  if (
    rewardBundle.in_reward_flow &&
    (rewardBundle.rewards.visible ||
      rewardBundle.card_reward_selection.visible ||
      rewardBundle.has_proceed)
  ) {
    return rewardBundle.card_reward_selection.visible
      ? "reward_card_selection_ready"
      : "reward_flow_ready";
  }

  const restSiteBundle = buildRestSiteBundle(state);
  if (
    restSiteBundle.in_rest_site_flow &&
    (restSiteBundle.rest_site.visible ||
      restSiteBundle.rest_site.proceed_visible ||
      restSiteBundle.deck_upgrade_selection.visible)
  ) {
    return restSiteBundle.deck_upgrade_selection.visible
      ? "rest_site_upgrade_ready"
      : "rest_site_ready";
  }

  if (isMapReadyState(state)) {
    return "map_ready";
  }

  if (state?.event_options?.visible === true && !isRlTransientRoomEntryContinueState(state)) {
    return "event_ready";
  }

  if (state?.shop?.visible === true || state?.shop?.is_open === true) {
    return "shop_ready";
  }

  return null;
}

function getPileCount(pile) {
  if (!pile || typeof pile !== "object") {
    return null;
  }

  if (Number.isInteger(pile.count)) {
    return pile.count;
  }

  if (Array.isArray(pile.cards)) {
    return pile.cards.length;
  }

  return null;
}

function getNonAutomationActions(state) {
  return Array.isArray(state?.available_actions)
    ? state.available_actions.filter(
      (action) =>
        typeof action?.action_id === "string" && !action.action_id.startsWith("automation:")
    )
    : [];
}

function isPlayCardAction(action) {
  return (
    isPlainObject(action) &&
    typeof action.action_id === "string" &&
    action.action_id.startsWith("play_card:")
  );
}

function getPlayableCardActions(state) {
  return getNonAutomationActions(state).filter(isPlayCardAction);
}

function isUsePotionAction(action) {
  return (
    isPlainObject(action) &&
    typeof action.action_id === "string" &&
    action.action_id.startsWith("use_potion:")
  );
}

function getUsablePotionActions(state) {
  return getNonAutomationActions(state).filter(isUsePotionAction);
}

function getPlayCardSequenceContinuationBlocker(state) {
  return getCombatSequenceContinuationBlocker(state, "play-card sequence");
}

function buildPlannedPlayCardSequenceStep(action, sequenceIndex, requestedActionId = null) {
  const targetActionSuffix =
    typeof action?.target_action_suffix === "string" ? action.target_action_suffix : null;
  const targetCombatId = Number.isFinite(action?.target_combat_id)
    ? action.target_combat_id
    : null;
  const targetSide = typeof action?.target_side === "string" ? action.target_side : null;
  const targetName = normalizeAgentText(action?.target_name);

  return {
    sequence_index: sequenceIndex,
    kind: "play_card",
    requested_action_id:
      typeof requestedActionId === "string" && requestedActionId.trim()
        ? requestedActionId
        : action.action_id,
    player_index: Number.isInteger(action?.player_index) ? action.player_index : null,
    initial_hand_index: Number.isInteger(action?.hand_index) ? action.hand_index : null,
    card_ref: typeof action?.card_ref === "string" ? action.card_ref : null,
    card: summarizeCardForAgent(action?.card),
    target: {
      action_suffix: targetActionSuffix,
      combat_id: targetCombatId,
      side: targetSide,
      name: targetName
    },
    action_fingerprint: buildPlayCardActionFingerprint(action)
  };
}

function summarizePlayCardSequencePlanStep(planStep) {
  return summarizeCombatSequencePlanStep(planStep);
}

function buildPlannedUsePotionSequenceStep(action, sequenceIndex, requestedActionId = null) {
  const targetActionSuffix =
    typeof action?.target_action_suffix === "string" ? action.target_action_suffix : null;
  const targetCombatId = Number.isFinite(action?.target_combat_id)
    ? action.target_combat_id
    : null;
  const targetSide = typeof action?.target_side === "string" ? action.target_side : null;
  const targetName = normalizeAgentText(action?.target_name);

  return {
    sequence_index: sequenceIndex,
    kind: "use_potion",
    requested_action_id:
      typeof requestedActionId === "string" && requestedActionId.trim()
        ? requestedActionId
        : action.action_id,
    player_index: Number.isInteger(action?.player_index) ? action.player_index : null,
    initial_slot_index: Number.isInteger(action?.slot_index) ? action.slot_index : null,
    potion: summarizePotionForAgent(action?.potion),
    target: {
      action_suffix: targetActionSuffix,
      combat_id: targetCombatId,
      side: targetSide,
      name: targetName
    },
    action_fingerprint: buildUsePotionActionFingerprint(action)
  };
}

function buildPlannedEndTurnSequenceStep(action, sequenceIndex, requestedActionId = null) {
  return {
    sequence_index: sequenceIndex,
    kind: "end_turn",
    requested_action_id:
      typeof requestedActionId === "string" && requestedActionId.trim()
        ? requestedActionId
        : action?.action_id ?? "end_turn"
  };
}

function resolvePlannedPlayCardStep(planStep, currentPlayCardActions) {
  const actions = Array.isArray(currentPlayCardActions) ? currentPlayCardActions : [];
  const exactMatch = actions.find(
    (action) =>
      action?.action_id === planStep?.requested_action_id &&
      doesPlayCardActionMatchPlanStep(action, planStep)
  );
  if (exactMatch) {
    return {
      action: exactMatch,
      match_type: "exact",
      compatible_candidate_count: 1
    };
  }

  const compatibleActions = actions.filter((action) =>
    doesPlayCardActionMatchPlanStep(action, planStep)
  );
  if (compatibleActions.length <= 0) {
    return {
      action: null,
      match_type: "unavailable",
      compatible_candidate_count: 0
    };
  }

  const rankedActions = compatibleActions
    .map((action) => ({
      action,
      score: scorePlayCardActionAgainstPlanStep(action, planStep)
    }))
    .sort((left, right) => right.score - left.score);

  return {
    action: rankedActions[0].action,
    match_type: compatibleActions.length === 1 ? "reindexed" : "reindexed_ambiguous",
    compatible_candidate_count: compatibleActions.length
  };
}

function resolveRequestedPlayCardAction(requestedActionId, currentPlayCardActions) {
  const actions = Array.isArray(currentPlayCardActions) ? currentPlayCardActions : [];
  const exactMatch = actions.find((action) => action?.action_id === requestedActionId);
  if (exactMatch) {
    return {
      action: exactMatch,
      match_type: "exact",
      compatible_candidate_count: 1
    };
  }

  const parsed = parseRequestedPlayCardActionId(requestedActionId);
  if (!parsed) {
    return {
      action: null,
      match_type: "unavailable",
      compatible_candidate_count: 0
    };
  }

  const compatibleActions = actions.filter((action) =>
    doesRequestedPlayCardActionIdMatchAction(parsed, action)
  );
  if (compatibleActions.length <= 0) {
    return {
      action: null,
      match_type: "unavailable",
      compatible_candidate_count: 0
    };
  }

  const rankedActions = compatibleActions
    .map((action) => ({
      action,
      score: scoreRequestedPlayCardActionIdMatch(parsed, action)
    }))
    .sort((left, right) => right.score - left.score);

  return {
    action: rankedActions[0].action,
    match_type:
      compatibleActions.length === 1 ? "normalized_id" : "normalized_id_ambiguous",
    compatible_candidate_count: compatibleActions.length
  };
}

function doesPlayCardActionMatchPlanStep(action, planStep) {
  if (!isPlayCardAction(action) || !isPlainObject(planStep)) {
    return false;
  }

  if (
    Number.isInteger(planStep.player_index) &&
    Number.isInteger(action?.player_index) &&
    action.player_index !== planStep.player_index
  ) {
    return false;
  }

  if (
    buildPlayCardActionFingerprint(action) !== planStep.action_fingerprint
  ) {
    return false;
  }

  const planTargetSuffix =
    typeof planStep?.target?.action_suffix === "string"
      ? planStep.target.action_suffix
      : null;
  const actionTargetSuffix =
    typeof action?.target_action_suffix === "string" ? action.target_action_suffix : null;
  if (planTargetSuffix !== actionTargetSuffix) {
    return false;
  }

  const planTargetCombatId = Number.isFinite(planStep?.target?.combat_id)
    ? planStep.target.combat_id
    : null;
  const actionTargetCombatId = Number.isFinite(action?.target_combat_id)
    ? action.target_combat_id
    : null;
  if (planTargetCombatId !== null && actionTargetCombatId !== planTargetCombatId) {
    return false;
  }

  return true;
}

function parseRequestedPlayCardActionId(actionId) {
  if (typeof actionId !== "string") {
    return null;
  }

  const parts = actionId.trim().split(":");
  if (parts.length < 3 || parts[0] !== "play_card") {
    return null;
  }

  const playerIndex = Number.parseInt(parts[1], 10);
  const rawCardRef = parts[2].trim();
  const handIndex = Number.parseInt(rawCardRef, 10);
  const hasCardRef = rawCardRef.length > 0;
  if (!Number.isInteger(playerIndex) || !hasCardRef) {
    return null;
  }

  const rawTargetSuffix = parts.length > 3 ? parts.slice(3).join(":").trim() : "";
  return {
    action_id: actionId.trim(),
    player_index: playerIndex,
    hand_index: Number.isInteger(handIndex) ? handIndex : null,
    card_ref: rawCardRef,
    target_suffix: rawTargetSuffix || null
  };
}

function doesRequestedPlayCardActionIdMatchAction(parsedActionId, action) {
  if (!isPlainObject(parsedActionId) || !isPlayCardAction(action)) {
    return false;
  }

  if (
    Number.isInteger(action?.player_index) &&
    action.player_index !== parsedActionId.player_index
  ) {
    return false;
  }

  if (
    typeof parsedActionId?.card_ref === "string" &&
    typeof action?.card_ref === "string" &&
    action.card_ref !== parsedActionId.card_ref
  ) {
    return false;
  }

  if (
    Number.isInteger(parsedActionId?.hand_index) &&
    Number.isInteger(action?.hand_index) &&
    action.hand_index !== parsedActionId.hand_index
  ) {
    return false;
  }

  const requestedTargetSuffix =
    typeof parsedActionId.target_suffix === "string" ? parsedActionId.target_suffix : null;
  const actionTargetSuffix =
    typeof action?.target_action_suffix === "string" ? action.target_action_suffix : null;
  if (requestedTargetSuffix === actionTargetSuffix) {
    return true;
  }

  return isCompatibleRequestedActionTargetAlias(requestedTargetSuffix, action);
}

function isCompatibleRequestedActionTargetAlias(requestedTargetSuffix, action) {
  if (typeof requestedTargetSuffix !== "string" || !requestedTargetSuffix.trim()) {
    return false;
  }

  const normalizedSuffix = requestedTargetSuffix.trim().toLowerCase();
  const actionTargetSuffix =
    typeof action?.target_action_suffix === "string" ? action.target_action_suffix : null;
  const targetType = getActionTargetType(action);

  if (
    (normalizedSuffix === "all" || normalizedSuffix === "all_enemies" || normalizedSuffix === "aoe") &&
    actionTargetSuffix === null &&
    targetType === "AllEnemies"
  ) {
    return true;
  }

  if (
    (normalizedSuffix === "none" ||
      normalizedSuffix === "notarget" ||
      normalizedSuffix === "no_target") &&
    actionTargetSuffix === null &&
    (targetType === "None" ||
      targetType === "RandomEnemy" ||
      targetType === "TargetedNoCreature" ||
      targetType === "Osty")
  ) {
    return true;
  }

  return false;
}

function scoreRequestedPlayCardActionIdMatch(parsedActionId, action) {
  let score = 0;

  if (
    Number.isInteger(action?.player_index) &&
    action.player_index === parsedActionId.player_index
  ) {
    score += 8;
  }

  if (typeof action?.card_ref === "string" && action.card_ref === parsedActionId.card_ref) {
    score += 20;
  }

  if (
    Number.isInteger(action?.hand_index) &&
    Number.isInteger(parsedActionId?.hand_index) &&
    action.hand_index === parsedActionId.hand_index
  ) {
    score += 12;
  }

  const requestedTargetSuffix =
    typeof parsedActionId.target_suffix === "string" ? parsedActionId.target_suffix : null;
  const actionTargetSuffix =
    typeof action?.target_action_suffix === "string" ? action.target_action_suffix : null;

  if (requestedTargetSuffix !== null && actionTargetSuffix === requestedTargetSuffix) {
    score += 10;
  } else if (isCompatibleRequestedActionTargetAlias(requestedTargetSuffix, action)) {
    score += 6;
  }

  return score;
}

function scorePlayCardActionAgainstPlanStep(action, planStep) {
  let score = 0;

  const actionHandIndex = Number.isInteger(action?.hand_index) ? action.hand_index : null;
  const planHandIndex = Number.isInteger(planStep?.initial_hand_index)
    ? planStep.initial_hand_index
    : null;
  if (actionHandIndex !== null && planHandIndex !== null) {
    score += 20 - Math.min(Math.abs(actionHandIndex - planHandIndex), 20);
  }

  if (
    typeof action?.card_ref === "string" &&
    typeof planStep?.card_ref === "string" &&
    action.card_ref === planStep.card_ref
  ) {
    score += 30;
  }

  if (
    Number.isFinite(action?.target_combat_id) &&
    Number.isFinite(planStep?.target?.combat_id) &&
    action.target_combat_id === planStep.target.combat_id
  ) {
    score += 5;
  }

  if (
    typeof action?.target_action_suffix === "string" &&
    typeof planStep?.target?.action_suffix === "string" &&
    action.target_action_suffix === planStep.target.action_suffix
  ) {
    score += 3;
  }

  return score;
}

function getActionTargetType(action) {
  if (typeof action?.target_scope === "string" && action.target_scope.trim()) {
    return action.target_scope;
  }

  if (typeof action?.card?.target_type === "string" && action.card.target_type.trim()) {
    return action.card.target_type;
  }

  if (typeof action?.potion?.target_type === "string" && action.potion.target_type.trim()) {
    return action.potion.target_type;
  }

  return null;
}

function parseRequestedUsePotionActionId(actionId) {
  if (typeof actionId !== "string") {
    return null;
  }

  const parts = actionId.trim().split(":");
  if (parts.length < 3 || parts[0] !== "use_potion") {
    return null;
  }

  const playerIndex = Number.parseInt(parts[1], 10);
  const slotIndex = Number.parseInt(parts[2], 10);
  if (!Number.isInteger(playerIndex) || !Number.isInteger(slotIndex)) {
    return null;
  }

  const rawTargetSuffix = parts.length > 3 ? parts.slice(3).join(":").trim() : "";
  return {
    action_id: actionId.trim(),
    player_index: playerIndex,
    slot_index: slotIndex,
    target_suffix: rawTargetSuffix || null
  };
}

function resolveRequestedUsePotionAction(requestedActionId, currentUsePotionActions) {
  const actions = Array.isArray(currentUsePotionActions) ? currentUsePotionActions : [];
  const exactMatch = actions.find((action) => action?.action_id === requestedActionId);
  if (exactMatch) {
    return {
      action: exactMatch,
      match_type: "exact",
      compatible_candidate_count: 1
    };
  }

  const parsed = parseRequestedUsePotionActionId(requestedActionId);
  if (!parsed) {
    return {
      action: null,
      match_type: "unavailable",
      compatible_candidate_count: 0
    };
  }

  const compatibleActions = actions.filter((action) =>
    doesRequestedUsePotionActionIdMatchAction(parsed, action)
  );
  if (compatibleActions.length <= 0) {
    return {
      action: null,
      match_type: "unavailable",
      compatible_candidate_count: 0
    };
  }

  const rankedActions = compatibleActions
    .map((action) => ({
      action,
      score: scoreRequestedUsePotionActionIdMatch(parsed, action)
    }))
    .sort((left, right) => right.score - left.score);

  return {
    action: rankedActions[0].action,
    match_type:
      compatibleActions.length === 1 ? "normalized_id" : "normalized_id_ambiguous",
    compatible_candidate_count: compatibleActions.length
  };
}

function doesRequestedUsePotionActionIdMatchAction(parsedActionId, action) {
  if (!isPlainObject(parsedActionId) || !isUsePotionAction(action)) {
    return false;
  }

  if (
    Number.isInteger(action?.player_index) &&
    action.player_index !== parsedActionId.player_index
  ) {
    return false;
  }

  if (Number.isInteger(action?.slot_index) && action.slot_index !== parsedActionId.slot_index) {
    return false;
  }

  const requestedTargetSuffix =
    typeof parsedActionId.target_suffix === "string" ? parsedActionId.target_suffix : null;
  const actionTargetSuffix =
    typeof action?.target_action_suffix === "string" ? action.target_action_suffix : null;
  if (requestedTargetSuffix === actionTargetSuffix) {
    return true;
  }

  return isCompatibleRequestedActionTargetAlias(requestedTargetSuffix, action);
}

function scoreRequestedUsePotionActionIdMatch(parsedActionId, action) {
  let score = 0;

  if (
    Number.isInteger(action?.player_index) &&
    action.player_index === parsedActionId.player_index
  ) {
    score += 8;
  }

  if (Number.isInteger(action?.slot_index) && action.slot_index === parsedActionId.slot_index) {
    score += 12;
  }

  const requestedTargetSuffix =
    typeof parsedActionId.target_suffix === "string" ? parsedActionId.target_suffix : null;
  const actionTargetSuffix =
    typeof action?.target_action_suffix === "string" ? action.target_action_suffix : null;

  if (requestedTargetSuffix !== null && actionTargetSuffix === requestedTargetSuffix) {
    score += 10;
  } else if (isCompatibleRequestedActionTargetAlias(requestedTargetSuffix, action)) {
    score += 6;
  }

  return score;
}

function resolvePlannedUsePotionStep(planStep, currentUsePotionActions) {
  const actions = Array.isArray(currentUsePotionActions) ? currentUsePotionActions : [];
  const exactMatch = actions.find(
    (action) =>
      action?.action_id === planStep?.requested_action_id &&
      doesUsePotionActionMatchPlanStep(action, planStep)
  );
  if (exactMatch) {
    return {
      action: exactMatch,
      match_type: "exact",
      compatible_candidate_count: 1
    };
  }

  const compatibleActions = actions.filter((action) =>
    doesUsePotionActionMatchPlanStep(action, planStep)
  );
  if (compatibleActions.length <= 0) {
    return {
      action: null,
      match_type: "unavailable",
      compatible_candidate_count: 0
    };
  }

  const rankedActions = compatibleActions
    .map((action) => ({
      action,
      score: scoreUsePotionActionAgainstPlanStep(action, planStep)
    }))
    .sort((left, right) => right.score - left.score);

  return {
    action: rankedActions[0].action,
    match_type: compatibleActions.length === 1 ? "reindexed" : "reindexed_ambiguous",
    compatible_candidate_count: compatibleActions.length
  };
}

function doesUsePotionActionMatchPlanStep(action, planStep) {
  if (!isUsePotionAction(action) || !isPlainObject(planStep)) {
    return false;
  }

  if (
    Number.isInteger(planStep.player_index) &&
    Number.isInteger(action?.player_index) &&
    action.player_index !== planStep.player_index
  ) {
    return false;
  }

  if (buildUsePotionActionFingerprint(action) !== planStep.action_fingerprint) {
    return false;
  }

  const planTargetSuffix =
    typeof planStep?.target?.action_suffix === "string"
      ? planStep.target.action_suffix
      : null;
  const actionTargetSuffix =
    typeof action?.target_action_suffix === "string" ? action.target_action_suffix : null;
  if (planTargetSuffix !== actionTargetSuffix) {
    return false;
  }

  const planTargetCombatId = Number.isFinite(planStep?.target?.combat_id)
    ? planStep.target.combat_id
    : null;
  const actionTargetCombatId = Number.isFinite(action?.target_combat_id)
    ? action.target_combat_id
    : null;
  if (planTargetCombatId !== null && actionTargetCombatId !== planTargetCombatId) {
    return false;
  }

  return true;
}

function scoreUsePotionActionAgainstPlanStep(action, planStep) {
  let score = 0;

  const actionSlotIndex = Number.isInteger(action?.slot_index) ? action.slot_index : null;
  const planSlotIndex = Number.isInteger(planStep?.initial_slot_index)
    ? planStep.initial_slot_index
    : null;
  if (actionSlotIndex !== null && planSlotIndex !== null) {
    score += 20 - Math.min(Math.abs(actionSlotIndex - planSlotIndex), 20);
  }

  if (
    Number.isFinite(action?.target_combat_id) &&
    Number.isFinite(planStep?.target?.combat_id) &&
    action.target_combat_id === planStep.target.combat_id
  ) {
    score += 5;
  }

  if (
    typeof action?.target_action_suffix === "string" &&
    typeof planStep?.target?.action_suffix === "string" &&
    action.target_action_suffix === planStep.target.action_suffix
  ) {
    score += 3;
  }

  return score;
}

function isRecoverableCombatSequenceExecutionError(error) {
  return (
    error instanceof BridgeHttpError &&
    (error.code === "state_version_conflict" || error.code === "action_not_available")
  );
}

function buildStableCardTextFingerprint(value) {
  const normalized = normalizeAgentText(value);
  if (typeof normalized !== "string" || !normalized.trim()) {
    return null;
  }

  return normalized
    .replace(/\d+(?:[.,]\d+)?/g, "#")
    .replace(/\s+/g, " ")
    .trim();
}

function compareNullableFingerprintNumbers(left, right) {
  const leftNumber = Number.isFinite(left) ? left : null;
  const rightNumber = Number.isFinite(right) ? right : null;

  if (leftNumber === rightNumber) {
    return 0;
  }

  if (leftNumber === null) {
    return -1;
  }

  if (rightNumber === null) {
    return 1;
  }

  return leftNumber - rightNumber;
}

function buildCardDynamicVarFingerprint(dynamicVars) {
  if (!Array.isArray(dynamicVars) || dynamicVars.length <= 0) {
    return [];
  }

  return dynamicVars
    .filter(
      (entry) => isPlainObject(entry) && typeof entry.name === "string" && entry.name.trim()
    )
    .map((entry) => ({
      name: entry.name,
      base_value: Number.isFinite(entry.base_value) ? entry.base_value : null,
      enchanted_value: Number.isFinite(entry.enchanted_value) ? entry.enchanted_value : null
    }))
    .sort((left, right) => {
      const nameComparison = left.name.localeCompare(right.name);
      if (nameComparison !== 0) {
        return nameComparison;
      }

      const baseComparison = compareNullableFingerprintNumbers(
        left.base_value,
        right.base_value
      );
      if (baseComparison !== 0) {
        return baseComparison;
      }

      return compareNullableFingerprintNumbers(
        left.enchanted_value,
        right.enchanted_value
      );
    });
}

function buildPlayCardActionFingerprint(action) {
  const card = isPlainObject(action?.card) ? action.card : {};

  // Keep this fingerprint stable across target-state-dependent preview changes
  // such as Vulnerable, Strength, or kill-trigger resource refunds. Dynamic
  // preview values are useful for scoring decisions, but too unstable for
  // same-card rematching after earlier sequence steps resolve.
  return JSON.stringify({
    card_id: typeof card.id === "string" ? card.id : null,
    title: normalizeAgentText(card.title),
    description_shape: buildStableCardTextFingerprint(card.description),
    type: typeof card.type === "string" ? card.type : null,
    rarity: typeof card.rarity === "string" ? card.rarity : null,
    target_type: typeof card.target_type === "string" ? card.target_type : null,
    canonical_energy_cost: Number.isInteger(card.canonical_energy_cost)
      ? card.canonical_energy_cost
      : null,
    canonical_star_cost: Number.isInteger(card.canonical_star_cost)
      ? card.canonical_star_cost
      : null,
    costs_x: card.costs_x === true,
    has_star_cost_x: card.has_star_cost_x === true,
    dynamic_vars: buildCardDynamicVarFingerprint(card.dynamic_vars)
  });
}

function buildUsePotionActionFingerprint(action) {
  const potion = isPlainObject(action?.potion) ? action.potion : {};

  return JSON.stringify({
    title: normalizeAgentText(potion.title),
    description_shape: buildStableCardTextFingerprint(potion.description),
    target_type: getActionTargetType(action)
  });
}

function buildRewardBundle(state) {
  const rewards = Array.isArray(state?.rewards?.rewards)
    ? state.rewards.rewards.map((entry) => ({
      index: Number.isInteger(entry?.index) ? entry.index : null,
      action_id:
        Number.isInteger(entry?.index) && Number.isInteger(entry?.index)
          ? `reward:${entry.index}`
          : null,
      reward: entry?.reward ?? null
    }))
    : [];
  const cardRewardSelectionOptions = Array.isArray(state?.card_reward_selection?.options)
    ? state.card_reward_selection.options.map((option) => ({
      index: Number.isInteger(option?.index) ? option.index : null,
      card: option?.card ?? null
    }))
    : [];
  const potionSlots = Array.isArray(state?.players?.[0]?.potions)
    ? state.players[0].potions.map((potion, index) => ({
      index,
      potion,
      is_empty: potion?.empty === true
    }))
    : [];
  const hasProceed = Array.isArray(state?.available_actions)
    ? state.available_actions.some((action) => action?.action_id === "proceed")
    : false;
  const screen = typeof state?.screen === "string" ? state.screen : null;
  const rewardsVisible = state?.rewards?.visible === true;
  const terminalProceedVisible = state?.rewards?.terminal_proceed_visible === true;
  const cardRewardVisible = state?.card_reward_selection?.visible === true;
  const inRewardFlow =
    rewardsVisible ||
    cardRewardVisible ||
    terminalProceedVisible ||
    screen === "REWARDS" ||
    screen === "CARD_REWARD_SELECTION";

  return {
    in_reward_flow: inRewardFlow,
    screen,
    rewards: {
      visible: rewardsVisible,
      terminal_proceed_visible: terminalProceedVisible,
      entries: rewards
    },
    card_reward_selection: {
      visible: cardRewardVisible,
      skip_visible: state?.card_reward_selection?.skip_visible === true,
      options: cardRewardSelectionOptions
    },
    potion_slots: potionSlots,
    empty_potion_slot_count: potionSlots.filter((slot) => slot.is_empty).length,
    has_proceed: hasProceed
  };
}

function buildRestSiteBundle(state) {
  const mapReady = isMapReadyState(state);
  const screen = typeof state?.screen === "string" ? state.screen : null;
  const restSiteVisible = state?.rest_site?.visible === true && !mapReady;
  const restSiteProceedVisible = state?.rest_site?.proceed_visible === true && !mapReady;
  const deckUpgradeVisible = state?.deck_upgrade_selection?.visible === true;
  const restSiteOptions = Array.isArray(state?.rest_site?.options)
    ? state.rest_site.options.map((option) => ({
      index: Number.isInteger(option?.index) ? option.index : null,
      option_id: option?.option_id ?? null,
      option_type: option?.option_type ?? null,
      title: option?.title ?? null,
      description: option?.description ?? null
    }))
    : [];
  const deckUpgradeOptions = Array.isArray(state?.deck_upgrade_selection?.options)
    ? state.deck_upgrade_selection.options.map((option) => ({
      index: Number.isInteger(option?.index) ? option.index : null,
      card: option?.card ?? null,
      upgrade_preview: option?.upgrade_preview ?? null,
      is_selected: option?.is_selected === true
    }))
    : [];
  const nonAutomationActions = getNonAutomationActions(state);

  return {
    in_rest_site_flow:
      deckUpgradeVisible ||
      (!mapReady &&
        (restSiteVisible || restSiteProceedVisible || screen === "REST_SITE")),
    screen,
    map_ready: mapReady,
    rest_site: {
      visible: restSiteVisible,
      header: state?.rest_site?.header ?? null,
      description: state?.rest_site?.description ?? null,
      proceed_visible: restSiteProceedVisible,
      options: restSiteOptions
    },
    deck_upgrade_selection: {
      visible: deckUpgradeVisible,
      options: deckUpgradeOptions
    },
    non_automation_action_ids: nonAutomationActions
      .map((action) => action?.action_id)
      .filter((actionId) => typeof actionId === "string")
  };
}

function buildCardSelectionBundle(state) {
  const screen = typeof state?.screen === "string" ? state.screen : null;
  const rawCardSelection = isPlainObject(state?.card_selection) ? state.card_selection : {};
  const options = Array.isArray(rawCardSelection.options)
    ? rawCardSelection.options.map((option) => ({
      index: Number.isInteger(option?.index) ? option.index : null,
      action_id: typeof option?.action_id === "string" ? option.action_id : null,
      selection_id:
        typeof option?.selection_id === "string" ? option.selection_id : null,
      card: option?.card ?? null,
      is_selected: option?.is_selected === true
    }))
    : [];
  const nonAutomationActionIds = getNonAutomationActions(state)
    .map((action) => action?.action_id)
    .filter((actionId) => typeof actionId === "string");
  const visible = rawCardSelection.visible === true;

  return {
    in_card_selection_flow:
      visible ||
      screen === "CARD_SELECTION" ||
      nonAutomationActionIds.some((actionId) => actionId.startsWith("card_selection:")),
    screen,
    card_selection: {
      visible,
      screen_type:
        typeof rawCardSelection.screen_type === "string" ? rawCardSelection.screen_type : null,
      prompt: typeof rawCardSelection.prompt === "string" ? rawCardSelection.prompt : null,
      selected_count: Number.isInteger(rawCardSelection.selected_count)
        ? rawCardSelection.selected_count
        : null,
      min_select: Number.isInteger(rawCardSelection.min_select)
        ? rawCardSelection.min_select
        : null,
      max_select: Number.isInteger(rawCardSelection.max_select)
        ? rawCardSelection.max_select
        : null,
      requires_manual_confirmation:
        typeof rawCardSelection.requires_manual_confirmation === "boolean"
          ? rawCardSelection.requires_manual_confirmation
          : null,
      cancelable:
        typeof rawCardSelection.cancelable === "boolean"
          ? rawCardSelection.cancelable
          : null,
      confirm_visible: rawCardSelection.confirm_visible === true,
      cancel_visible: rawCardSelection.cancel_visible === true,
      close_visible: rawCardSelection.close_visible === true,
      skip_visible: rawCardSelection.skip_visible === true,
      options
    },
    non_automation_action_ids: nonAutomationActionIds
  };
}

function buildIndexedOptionEntries(options) {
  if (!Array.isArray(options)) {
    return [];
  }

  return options.map((option, fallbackIndex) => ({
    option,
    option_index: Number.isInteger(option?.index) ? option.index : fallbackIndex
  }));
}

function findIndexedOptionEntry(options, requestedIndex) {
  if (!Number.isInteger(requestedIndex)) {
    return null;
  }

  return (
    buildIndexedOptionEntries(options).find(
      (entry) => entry.option_index === requestedIndex
    ) ?? null
  );
}

function buildShopBundle(state) {
  const screen = typeof state?.screen === "string" ? state.screen : null;
  const rawShop = isPlainObject(state?.shop) ? state.shop : {};
  const items = Array.isArray(rawShop.items)
    ? rawShop.items.map((item) => ({
      index: Number.isInteger(item?.index) ? item.index : null,
      item_kind: typeof item?.item_kind === "string" ? item.item_kind : null,
      title: typeof item?.title === "string" ? item.title : null,
      description: typeof item?.description === "string" ? item.description : null,
      cost: Number.isFinite(item?.cost) ? item.cost : null,
      is_affordable: item?.is_affordable === true,
      used: typeof item?.used === "boolean" ? item.used : null,
      card: item?.card ?? null,
      relic: item?.relic ?? null,
      potion: item?.potion ?? null
    }))
    : [];
  const nonAutomationActionIds = getNonAutomationActions(state)
    .map((action) => action?.action_id)
    .filter((actionId) => typeof actionId === "string");

  return {
    in_shop_flow:
      rawShop.visible === true ||
      rawShop.is_open === true ||
      screen === "SHOP" ||
      nonAutomationActionIds.some((actionId) => actionId.startsWith("shop:")),
    screen,
    shop: {
      visible: rawShop.visible === true,
      is_open: rawShop.is_open === true,
      gold: Number.isFinite(rawShop.gold) ? rawShop.gold : null,
      merchant_button_visible: rawShop.merchant_button_visible === true,
      back_button_visible: rawShop.back_button_visible === true,
      proceed_visible: rawShop.proceed_visible === true,
      items
    },
    non_automation_action_ids: nonAutomationActionIds
  };
}

function getIndexedOptionSurfaces(state) {
  const surfaces = [];
  const rewardBundle = buildRewardBundle(state);
  if (rewardBundle.rewards.visible) {
    surfaces.push({
      surface: "reward",
      prompt: null,
      options: rewardBundle.rewards.entries,
      getActionId(index) {
        return typeof this.options[index]?.action_id === "string"
          ? this.options[index].action_id
          : `reward:${index}`;
      }
    });
  }

  if (rewardBundle.card_reward_selection.visible) {
    surfaces.push({
      surface: "card_reward",
      prompt: null,
      options: rewardBundle.card_reward_selection.options,
      getActionId(index) {
        return `card_reward:${index}`;
      }
    });
  }

  const restSiteBundle = buildRestSiteBundle(state);
  if (restSiteBundle.rest_site.visible) {
    surfaces.push({
      surface: "rest_site",
      prompt: normalizeAgentText(restSiteBundle.rest_site.header),
      options: restSiteBundle.rest_site.options,
      getActionId(index) {
        return `rest_site:${index}`;
      }
    });
  }

  if (restSiteBundle.deck_upgrade_selection.visible) {
    surfaces.push({
      surface: "deck_upgrade",
      prompt: normalizeAgentText(restSiteBundle.rest_site.header),
      options: restSiteBundle.deck_upgrade_selection.options,
      getActionId(index) {
        return `deck_upgrade:select:${index}`;
      }
    });
  }

  if (state?.event_options?.visible === true) {
    surfaces.push({
      surface: "event",
      prompt: null,
      options: Array.isArray(state.event_options.options) ? state.event_options.options : [],
      getActionId(index) {
        return `event_option:${index}`;
      }
    });
  }

  const cardSelectionBundle = buildCardSelectionBundle(state);
  if (cardSelectionBundle.card_selection.visible) {
    surfaces.push({
      surface: "card_selection",
      prompt: normalizeAgentText(cardSelectionBundle.card_selection.prompt),
      min_select: cardSelectionBundle.card_selection.min_select,
      max_select: cardSelectionBundle.card_selection.max_select,
      selected_count: cardSelectionBundle.card_selection.selected_count,
      options: cardSelectionBundle.card_selection.options,
      getActionId(index) {
        const option = Array.isArray(cardSelectionBundle.card_selection.options)
          ? cardSelectionBundle.card_selection.options.find(
            (entry) => Number.isInteger(entry?.index) && entry.index === index
          )
          : null;
        return typeof option?.action_id === "string" && option.action_id
          ? option.action_id
          : `card_selection:select:${index}`;
      }
    });
  }

  return surfaces;
}

function summarizeIndexedOptionSurfacesForAgent(state) {
  return getIndexedOptionSurfaces(state).map((surface) => ({
    surface: surface.surface,
    option_count: Array.isArray(surface.options) ? surface.options.length : 0,
    prompt: typeof surface.prompt === "string" ? surface.prompt : undefined,
    min_select: Number.isInteger(surface.min_select) ? surface.min_select : undefined,
    max_select: Number.isInteger(surface.max_select) ? surface.max_select : undefined,
    selected_count:
      Number.isInteger(surface.selected_count) ? surface.selected_count : undefined
  }));
}

function resolveIndexedOptionSurface(state, requestedSurface) {
  const surfaces = getIndexedOptionSurfaces(state);
  if (requestedSurface === "auto") {
    if (surfaces.length <= 0) {
      return {
        ok: false,
        reason: "no_visible_indexed_option_surface"
      };
    }

    if (surfaces.length > 1) {
      return {
        ok: false,
        reason: "multiple_visible_indexed_option_surfaces"
      };
    }

    return {
      ok: true,
      ...surfaces[0]
    };
  }

  const matched = surfaces.find((surface) => surface.surface === requestedSurface);
  if (!matched) {
    return {
      ok: false,
      reason: "requested_surface_not_visible"
    };
  }

  return {
    ok: true,
    ...matched
  };
}

function summarizeIndexedOptionForAgent(surface, option) {
  if (!isPlainObject(option)) {
    return option ?? null;
  }

  if (surface === "reward") {
    return {
      index: Number.isInteger(option.index) ? option.index : null,
      reward: summarizeRewardForAgent(option.reward)
    };
  }

  if (surface === "card_reward" || surface === "deck_upgrade" || surface === "card_selection") {
    return {
      index: Number.isInteger(option.index) ? option.index : null,
      is_selected: option.is_selected === true,
      card: summarizeCardForAgent(option.card)
    };
  }

  if (surface === "rest_site") {
    return {
      index: Number.isInteger(option.index) ? option.index : null,
      option_type: typeof option.option_type === "string" ? option.option_type : null,
      title: normalizeAgentText(option.title),
      description: normalizeAgentText(option.description)
    };
  }

  if (surface === "event") {
    return summarizeEventOptionForAgent(option);
  }

  return option;
}

function summarizeEventOptionForAgent(option) {
  if (!isPlainObject(option)) {
    return null;
  }

  const summary = {
    index: Number.isInteger(option.index) ? option.index : null,
    option_type: typeof option.option_type === "string" ? option.option_type : null,
    option_id: typeof option.option_id === "string" ? option.option_id : null,
    title: normalizeAgentText(option.title),
    description: normalizeAgentText(option.description),
    is_proceed: option.is_proceed === true
  };

  if (option.is_selected === true) {
    summary.is_selected = true;
  }

  if (typeof option.is_enabled === "boolean") {
    summary.is_enabled = option.is_enabled === true;
  }

  if (option.action_available === true) {
    summary.action_available = true;
  }

  if (typeof option.divination_size === "string" && option.divination_size.trim()) {
    summary.divination_size = option.divination_size.trim();
  }

  const coord = summarizeCrystalSphereCoordForAgent(option?.coord);
  if (coord) {
    summary.coord = coord;
  }

  if (option?.is_highlighted === true) {
    summary.is_highlighted = true;
  }

  if (Array.isArray(option?.glossary)) {
    const glossary = option.glossary
      .map((entry) => ({
        title: normalizeAgentText(entry?.title),
        description: normalizeAgentText(entry?.description)
      }))
      .filter((entry) => entry.title || entry.description);
    if (glossary.length > 0) {
      summary.glossary = glossary;
    }
  }

  return summary;
}

function isCrystalSphereScreen(screen) {
  return screen === "EVENT_CRYSTAL_SPHERE";
}

function isCrystalSphereEventOption(option) {
  return (
    isPlainObject(option) &&
    (typeof option.option_type === "string" &&
      option.option_type.startsWith("crystal_sphere_") ||
      typeof option.option_id === "string" &&
      option.option_id.startsWith("crystal_sphere:"))
  );
}

function isCrystalSphereEventOptionAction(action) {
  return (
    isPlainObject(action) &&
    typeof action.kind === "string" &&
    action.kind === "event_option" &&
    isCrystalSphereEventOption(action.option)
  );
}

function getCrystalSphereOptionIndex(option, fallbackIndex = null) {
  if (Number.isInteger(option?.index)) {
    return option.index;
  }

  return Number.isInteger(fallbackIndex) ? fallbackIndex : null;
}

function summarizeCrystalSphereControlActionForAgent(option, fallbackIndex = null) {
  if (!isCrystalSphereEventOption(option)) {
    return null;
  }

  const index = getCrystalSphereOptionIndex(option, fallbackIndex);
  if (!Number.isInteger(index)) {
    return null;
  }

  const optionType = typeof option.option_type === "string" ? option.option_type : "";
  let name = null;
  if (optionType === "crystal_sphere_small_divination") {
    name = "small";
  } else if (optionType === "crystal_sphere_big_divination") {
    name = "big";
  } else if (optionType === "crystal_sphere_proceed") {
    name = "proceed";
  }

  if (name === null) {
    return null;
  }

  return `${index}:${name}${option.is_selected === true ? "*" : ""}`;
}

function summarizeCrystalSphereCellActionForAgent(option, fallbackIndex = null) {
  if (!isCrystalSphereEventOption(option) || option.option_type !== "crystal_sphere_cell") {
    return null;
  }

  const index = getCrystalSphereOptionIndex(option, fallbackIndex);
  const coord = summarizeCrystalSphereCoordForAgent(option.coord);
  if (!Number.isInteger(index) || coord === null) {
    return null;
  }

  return `${index}:${coord}`;
}

function summarizeCrystalSphereActionGroupsForAgent(optionsOrActions) {
  const entries = Array.isArray(optionsOrActions) ? optionsOrActions : [];
  const controls = [];
  const cells = [];
  let cellActionStartIndex = null;
  let proceed = null;

  for (let index = 0; index < entries.length; index += 1) {
    const entry = entries[index];
    const option = isPlainObject(entry?.option) ? entry.option : entry;
    if (!isCrystalSphereEventOption(option)) {
      continue;
    }

    const control = summarizeCrystalSphereControlActionForAgent(option, index);
    if (control !== null) {
      if (control.includes(":proceed")) {
        proceed = control;
      } else {
        controls.push(control);
      }
      continue;
    }

    const cellIndex = getCrystalSphereOptionIndex(option, index);
    const cellCoord = summarizeCrystalSphereCoordForAgent(option.coord);
    if (Number.isInteger(cellIndex) && cellCoord !== null) {
      if (cellActionStartIndex === null) {
        cellActionStartIndex = cellIndex;
      }
      cells.push(cellCoord);
    }
  }

  const summary = {};
  if (controls.length > 0) {
    summary.controls = controls;
  }
  if (proceed !== null) {
    summary.proceed = proceed;
  }
  if (cells.length > 0) {
    summary.cell_action_start_index = cellActionStartIndex;
    summary.cells = cells;
  }

  return Object.keys(summary).length > 0 ? summary : null;
}

function summarizeCrystalSphereCoordForAgent(coord) {
  if (!isPlainObject(coord)) {
    return null;
  }

  if (!Number.isInteger(coord.x) || !Number.isInteger(coord.y)) {
    return null;
  }

  return `${coord.x},${coord.y}`;
}

function summarizeCrystalSphereItemForAgent(item) {
  if (!isPlainObject(item)) {
    return null;
  }

  const summary = {
    item_type: typeof item.item_type === "string" ? item.item_type : null
  };

  const title = normalizeAgentText(item.title);
  if (title) {
    summary.title = title;
  }

  if (Number.isFinite(item.amount)) {
    summary.amount = item.amount;
  }

  if (typeof item.rarity === "string" && item.rarity.trim()) {
    summary.rarity = item.rarity.trim();
  }

  if (item.is_good === true) {
    summary.is_good = true;
  }

  return summary;
}

function summarizeCrystalSphereRevealedCellForAgent(coord, item) {
  if (typeof coord !== "string" || coord.length <= 0 || !isPlainObject(item)) {
    return null;
  }

  const itemType =
    typeof item.item_type === "string" && item.item_type.trim()
      ? item.item_type.trim()
      : "item";
  if (Number.isFinite(item.amount)) {
    return `${coord}=${itemType}:${item.amount}`;
  }

  const title = normalizeAgentText(item.title);
  if (title) {
    return `${coord}=${itemType}:${title}`;
  }

  if (typeof item.rarity === "string" && item.rarity.trim()) {
    return `${coord}=${itemType}:${item.rarity.trim()}`;
  }

  return `${coord}=${itemType}`;
}

function summarizeCrystalSphereControlForAgent(control) {
  if (!isPlainObject(control) || control.visible !== true) {
    return null;
  }

  const summary = {
    enabled: control.enabled === true,
    action_available: control.action_available === true
  };

  if (control.is_selected === true) {
    summary.is_selected = true;
  }

  const label = normalizeAgentText(control.label);
  if (label) {
    summary.label = label;
  }

  if (typeof control.divination_size === "string" && control.divination_size.trim()) {
    summary.divination_size = control.divination_size.trim();
  }

  return summary;
}

function summarizeCrystalSphereGridSizeForAgent(gridSize) {
  if (!isPlainObject(gridSize)) {
    return null;
  }

  if (!Number.isInteger(gridSize.width) || !Number.isInteger(gridSize.height)) {
    return null;
  }

  return `${gridSize.width}x${gridSize.height}`;
}

function summarizeCrystalSphereForAgent(crystalSphere, eventOptions = null) {
  if (!isPlainObject(crystalSphere) || crystalSphere.visible !== true) {
    return null;
  }

  const cells = Array.isArray(crystalSphere.cells)
    ? crystalSphere.cells.filter((cell) => isPlainObject(cell))
    : [];
  const revealedCells = [];
  let hiddenCellCount = 0;

  for (const cell of cells) {
    const coord =
      Number.isInteger(cell.x) && Number.isInteger(cell.y)
        ? `${cell.x},${cell.y}`
        : null;

    if (cell.is_hidden === true) {
      hiddenCellCount += 1;
    }

    if (coord) {
      const item = summarizeCrystalSphereItemForAgent(cell.revealed_item);
      if (item !== null) {
        const revealed = summarizeCrystalSphereRevealedCellForAgent(coord, item);
        if (revealed !== null) {
          revealedCells.push(revealed);
        }
      }
    }
  }

  const summary = {
    divinations_left: Number.isFinite(crystalSphere.divinations_left)
      ? crystalSphere.divinations_left
      : null,
    current_tool:
      typeof crystalSphere.current_tool === "string" ? crystalSphere.current_tool : null,
    grid_size: summarizeCrystalSphereGridSizeForAgent(crystalSphere.grid_size),
    hidden_cell_count: hiddenCellCount
  };

  const actions = summarizeCrystalSphereActionGroupsForAgent(eventOptions);
  if (actions !== null) {
    summary.actions = actions;
  }

  if (revealedCells.length > 0) {
    summary.revealed = revealedCells;
  }

  return summary;
}

function summarizeMapSnapshotStatusForAgent(snapshot) {
  if (!isPlainObject(snapshot)) {
    return snapshot ?? null;
  }

  const summary = {
    settled: snapshot.settled === true,
    reason: typeof snapshot.reason === "string" ? snapshot.reason : null
  };

  if (Number.isInteger(snapshot.poll_count) && snapshot.poll_count > 0) {
    summary.poll_count = snapshot.poll_count;
  }

  if (snapshot.frontier_action_match !== true) {
    summary.frontier_action_match = snapshot.frontier_action_match === true;
  }

  return summary;
}

async function prepareStateForMapTravel(session, initialState) {
  const executedActions = [];
  let state = initialState;

  for (let attempt = 0; attempt < 4; attempt += 1) {
    const cardSelectionBundle = buildCardSelectionBundle(state);
    if (cardSelectionBundle.in_card_selection_flow) {
      return {
        state,
        executed_actions: executedActions,
        blocker: {
          reason: "card_selection_resolution_required",
          message: "A card-selection flow is still active and must be resolved first.",
          card_selection_bundle: cardSelectionBundle
        }
      };
    }

    const rewardBundle = buildRewardBundle(state);
    if (rewardBundle.in_reward_flow) {
      if (!canAutoProceedFromRewardCleanupState(state)) {
        return {
          state,
          executed_actions: executedActions,
          blocker: {
            reason: rewardBundle.card_reward_selection.visible
              ? "reward_card_selection_required"
              : "reward_resolution_required",
            message:
              "Reward flow still needs an explicit choice. Finish it before requesting map travel.",
            reward_bundle: rewardBundle
          }
        };
      }

      const autoProceedResult = await autoAdvanceProceedChain(session, state);
      executedActions.push(...autoProceedResult.executed_actions);
      state = autoProceedResult.state;
      continue;
    }

    const restSiteBundle = buildRestSiteBundle(state);
    if (restSiteBundle.in_rest_site_flow) {
      if (!canAutoProceedFromRestSiteCleanupState(state)) {
        return {
          state,
          executed_actions: executedActions,
          blocker: {
            reason: restSiteBundle.deck_upgrade_selection.visible
              ? "rest_site_upgrade_required"
              : "rest_site_resolution_required",
            message:
              "Rest-site flow still needs an explicit choice. Finish it before requesting map travel.",
            rest_site_bundle: restSiteBundle
          }
        };
      }

      const autoProceedResult = await autoAdvanceRestSiteProceedChain(session, state);
      executedActions.push(...autoProceedResult.executed_actions);
      state = autoProceedResult.state;
      continue;
    }

    const shopBundle = buildShopBundle(state);
    if (shopBundle.in_shop_flow) {
      return {
        state,
        executed_actions: executedActions,
        blocker: {
          reason: "shop_resolution_required",
          message: "Shop flow is still active and must be resolved before map travel."
        }
      };
    }

    if (state?.event_options?.visible === true) {
      return {
        state,
        executed_actions: executedActions,
        blocker: {
          reason: "event_resolution_required",
          message: "Event options are still visible and must be resolved before map travel."
        }
      };
    }

    break;
  }

  return {
    state,
    executed_actions: executedActions,
    blocker: null
  };
}

function normalizeShopPurchaseRequests(value) {
  if (value === undefined) {
    return [];
  }

  if (!Array.isArray(value)) {
    throw new ToolPayloadError(
      "invalid_arguments",
      "purchases must be an array when provided.",
      {
        field: "purchases"
      }
    );
  }

  return value.map((entry, index) => {
    if (!isPlainObject(entry)) {
      throw new ToolPayloadError(
        "invalid_arguments",
        `purchases[${index}] must be an object.`,
        {
          field: `purchases[${index}]`
        }
      );
    }

    const actionId = optionalString(entry.action_id, `purchases[${index}].action_id`);
    const title = optionalString(entry.title, `purchases[${index}].title`);
    const itemKind = normalizeOptionalShopItemKind(
      entry.item_kind,
      `purchases[${index}].item_kind`
    );

    if (actionId === undefined && title === undefined) {
      throw new ToolPayloadError(
        "invalid_arguments",
        `purchases[${index}] must provide action_id or title.`,
        {
          field: `purchases[${index}]`
        }
      );
    }

    return {
      action_id: actionId,
      title: title === undefined ? undefined : normalizeAgentText(title) ?? title.trim(),
      item_kind: itemKind
    };
  });
}

function normalizeOptionalShopItemKind(value, fieldName) {
  if (value === undefined) {
    return undefined;
  }

  if (typeof value !== "string") {
    throw new ToolPayloadError(
      "invalid_arguments",
      `${fieldName} must be a string when provided.`,
      {
        field: fieldName
      }
    );
  }

  const normalized = value.trim();
  if (["card", "relic", "potion", "card_removal"].includes(normalized)) {
    return normalized;
  }

  throw new ToolPayloadError(
    "invalid_arguments",
    `${fieldName} must be one of: card, relic, potion, card_removal.`,
    {
      field: fieldName
    }
  );
}

function normalizeTextComparisonKey(value) {
  const normalized = normalizeAgentText(value);
  if (typeof normalized !== "string" || !normalized.trim()) {
    return null;
  }

  return normalized.trim().toLowerCase();
}

function isShopBuyAction(action) {
  return (
    isPlainObject(action) &&
    typeof action.action_id === "string" &&
    action.action_id.startsWith("shop:buy:")
  );
}

function getShopBuyActions(state) {
  return getNonAutomationActions(state).filter(isShopBuyAction);
}

function summarizeShopBuyActionForAgent(action) {
  if (!isShopBuyAction(action)) {
    return summarizeActionForAgent(action);
  }

  return {
    action_id: action.action_id,
    label: normalizeAgentText(action.label),
    item: summarizeShopItemForAgent(action.item)
  };
}

function getShopActionIndex(action) {
  if (Number.isInteger(action?.index)) {
    return action.index;
  }

  if (Number.isInteger(action?.item?.index)) {
    return action.item.index;
  }

  return parseIndexedActionId(action?.action_id, "shop:buy:");
}

function parseIndexedActionId(actionId, prefix) {
  if (typeof actionId !== "string" || typeof prefix !== "string") {
    return null;
  }

  if (!actionId.startsWith(prefix)) {
    return null;
  }

  const suffix = actionId.slice(prefix.length);
  return /^\d+$/.test(suffix) ? Number.parseInt(suffix, 10) : null;
}

function getShopActionTitle(action) {
  if (typeof action?.item?.title === "string" && action.item.title.trim()) {
    return normalizeAgentText(action.item.title);
  }

  if (typeof action?.label === "string" && action.label.trim()) {
    return normalizeAgentText(action.label);
  }

  return null;
}

function getShopActionItemKind(action) {
  const itemKind =
    typeof action?.item?.item_kind === "string" ? action.item.item_kind.trim() : null;
  return itemKind && ["card", "relic", "potion", "card_removal"].includes(itemKind)
    ? itemKind
    : null;
}

function resolveInitialShopPurchaseRequest(request, state) {
  const actions = getShopBuyActions(state);
  if (actions.length <= 0 || !isPlainObject(request)) {
    return null;
  }

  const requestTitleKey = normalizeTextComparisonKey(request.title);
  const requestItemKind = normalizeOptionalShopItemKind(
    request.item_kind,
    "request.item_kind"
  );
  const hasDescriptor = requestTitleKey !== null || requestItemKind !== undefined;

  if (!hasDescriptor) {
    return typeof request.action_id === "string"
      ? actions.find((action) => action.action_id === request.action_id) ?? null
      : null;
  }

  const compatibleActions = actions.filter((action) => {
    if (requestItemKind !== undefined && getShopActionItemKind(action) !== requestItemKind) {
      return false;
    }

    if (requestTitleKey !== null && normalizeTextComparisonKey(getShopActionTitle(action)) !== requestTitleKey) {
      return false;
    }

    return true;
  });

  if (compatibleActions.length <= 0) {
    return null;
  }

  if (typeof request.action_id === "string") {
    const exactAction = compatibleActions.find(
      (action) => action.action_id === request.action_id
    );
    if (exactAction) {
      return exactAction;
    }
  }

  return compatibleActions
    .map((action) => ({
      action,
      score: scoreShopBuyActionAgainstRequest(action, request)
    }))
    .sort((left, right) => right.score - left.score)[0].action;
}

function scoreShopBuyActionAgainstRequest(action, request) {
  let score = 0;

  if (typeof request?.action_id === "string") {
    if (action?.action_id === request.action_id) {
      score += 100;
    }

    const requestedIndex = parseIndexedActionId(request.action_id, "shop:buy:");
    const actionIndex = getShopActionIndex(action);
    if (requestedIndex !== null && actionIndex !== null) {
      score += 20 - Math.min(Math.abs(actionIndex - requestedIndex), 20);
    }
  }

  const requestTitleKey = normalizeTextComparisonKey(request?.title);
  if (requestTitleKey !== null && normalizeTextComparisonKey(getShopActionTitle(action)) === requestTitleKey) {
    score += 30;
  }

  if (
    typeof request?.item_kind === "string" &&
    getShopActionItemKind(action) === request.item_kind
  ) {
    score += 15;
  }

  return score;
}

function buildPlannedShopPurchaseStep(action, request, sequenceIndex) {
  return {
    sequence_index: sequenceIndex,
    request: summarizeShopPurchaseRequest(request),
    requested_action_id: typeof request?.action_id === "string" ? request.action_id : null,
    initial_action_id: typeof action?.action_id === "string" ? action.action_id : null,
    initial_shop_index: getShopActionIndex(action),
    item: summarizeShopItemForAgent(action?.item),
    action_fingerprint: buildShopBuyActionFingerprint(action)
  };
}

function summarizeShopPurchasePlanStep(planStep) {
  if (!isPlainObject(planStep)) {
    return null;
  }

  return {
    sequence_index: Number.isInteger(planStep.sequence_index) ? planStep.sequence_index : null,
    request: summarizeShopPurchaseRequest(planStep.request),
    requested_action_id:
      typeof planStep.requested_action_id === "string" ? planStep.requested_action_id : null,
    initial_action_id:
      typeof planStep.initial_action_id === "string" ? planStep.initial_action_id : null,
    initial_shop_index:
      Number.isInteger(planStep.initial_shop_index) ? planStep.initial_shop_index : null,
    item: summarizeShopItemForAgent(planStep.item)
  };
}

function getShopVisitContinuationBlocker(state) {
  if (!isPlainObject(state)) {
    return {
      reason: "missing_state",
      message: "Bridge state was missing while resolving the shop visit."
    };
  }

  const cardSelectionBundle = buildCardSelectionBundle(state);
  if (cardSelectionBundle.in_card_selection_flow && cardSelectionBundle.card_selection.visible) {
    return {
      reason: "card_selection_ready",
      message: "Card selection became visible before the requested shop visit finished."
    };
  }

  const shopBundle = buildShopBundle(state);
  if (!shopBundle.in_shop_flow) {
    return {
      reason: `screen:${typeof state.screen === "string" ? state.screen : "unknown"}`,
      message: "The game is no longer in a shop-related flow."
    };
  }

  if (shopBundle.shop.visible && shopBundle.shop.is_open !== true && getShopBuyActions(state).length <= 0) {
    return {
      reason: "shop_inventory_closed",
      message: "Merchant inventory is not open."
    };
  }

  return null;
}

function resolvePlannedShopPurchaseStep(planStep, currentShopActions) {
  const actions = Array.isArray(currentShopActions) ? currentShopActions : [];
  const exactCandidateIds = [
    typeof planStep?.initial_action_id === "string" ? planStep.initial_action_id : null,
    typeof planStep?.requested_action_id === "string" ? planStep.requested_action_id : null
  ].filter((value, index, array) => typeof value === "string" && array.indexOf(value) === index);

  const exactMatch = actions.find(
    (action) =>
      exactCandidateIds.includes(action?.action_id) &&
      doesShopBuyActionMatchPlanStep(action, planStep)
  );
  if (exactMatch) {
    return {
      action: exactMatch,
      match_type: "exact",
      compatible_candidate_count: 1
    };
  }

  const compatibleActions = actions.filter((action) =>
    doesShopBuyActionMatchPlanStep(action, planStep)
  );
  if (compatibleActions.length <= 0) {
    return {
      action: null,
      match_type: "unavailable",
      compatible_candidate_count: 0
    };
  }

  const rankedActions = compatibleActions
    .map((action) => ({
      action,
      score: scoreShopBuyActionAgainstPlanStep(action, planStep)
    }))
    .sort((left, right) => right.score - left.score);

  return {
    action: rankedActions[0].action,
    match_type: compatibleActions.length === 1 ? "reindexed" : "reindexed_ambiguous",
    compatible_candidate_count: compatibleActions.length
  };
}

function doesShopBuyActionMatchPlanStep(action, planStep) {
  if (!isShopBuyAction(action) || !isPlainObject(planStep)) {
    return false;
  }

  return buildShopBuyActionFingerprint(action) === planStep.action_fingerprint;
}

function scoreShopBuyActionAgainstPlanStep(action, planStep) {
  let score = 0;

  if (action?.action_id === planStep?.initial_action_id) {
    score += 100;
  }

  if (action?.action_id === planStep?.requested_action_id) {
    score += 80;
  }

  const actionIndex = getShopActionIndex(action);
  const initialIndex = Number.isInteger(planStep?.initial_shop_index)
    ? planStep.initial_shop_index
    : parseIndexedActionId(planStep?.initial_action_id, "shop:buy:");
  if (actionIndex !== null && initialIndex !== null) {
    score += 20 - Math.min(Math.abs(actionIndex - initialIndex), 20);
  }

  const requestTitleKey = normalizeTextComparisonKey(planStep?.request?.title);
  if (requestTitleKey !== null && normalizeTextComparisonKey(getShopActionTitle(action)) === requestTitleKey) {
    score += 10;
  }

  if (
    typeof planStep?.request?.item_kind === "string" &&
    getShopActionItemKind(action) === planStep.request.item_kind
  ) {
    score += 5;
  }

  return score;
}

function buildShopBuyActionFingerprint(action) {
  const item = isPlainObject(action?.item) ? action.item : {};
  const card = isPlainObject(item.card) ? item.card : {};
  const relic = isPlainObject(item.relic) ? item.relic : {};
  const potion = isPlainObject(item.potion) ? item.potion : {};
  const effectPreview = isPlainObject(card.effect_preview) ? card.effect_preview : {};

  return JSON.stringify({
    item_kind: getShopActionItemKind(action),
    title: normalizeAgentText(item.title),
    description: normalizeAgentText(item.description),
    cost: Number.isFinite(item.cost) ? item.cost : null,
    used: typeof item.used === "boolean" ? item.used : null,
    card: {
      id: typeof card.id === "string" ? card.id : null,
      title: normalizeAgentText(card.title),
      description: normalizeAgentText(card.description),
      type: typeof card.type === "string" ? card.type : null,
      target_type: typeof card.target_type === "string" ? card.target_type : null,
      canonical_energy_cost: Number.isInteger(card.canonical_energy_cost)
        ? card.canonical_energy_cost
        : null,
      resolved_energy_cost: Number.isInteger(card.resolved_energy_cost)
        ? card.resolved_energy_cost
        : null,
      effect_preview: {
        summary: normalizeAgentText(effectPreview.summary),
        total_damage: Number.isFinite(effectPreview.total_damage)
          ? effectPreview.total_damage
          : null,
        damage_per_hit: Number.isFinite(effectPreview.damage_per_hit)
          ? effectPreview.damage_per_hit
          : null,
        hits: Number.isFinite(effectPreview.hits) ? effectPreview.hits : null,
        total_block: Number.isFinite(effectPreview.total_block)
          ? effectPreview.total_block
          : null,
        draw: Number.isFinite(effectPreview.draw) ? effectPreview.draw : null,
        heal: Number.isFinite(effectPreview.heal) ? effectPreview.heal : null,
        weak: Number.isFinite(effectPreview.weak) ? effectPreview.weak : null,
        vulnerable: Number.isFinite(effectPreview.vulnerable)
          ? effectPreview.vulnerable
          : null,
        strength: Number.isFinite(effectPreview.strength) ? effectPreview.strength : null,
        dexterity: Number.isFinite(effectPreview.dexterity)
          ? effectPreview.dexterity
          : null,
        x_cost_value: Number.isFinite(effectPreview.x_cost_value)
          ? effectPreview.x_cost_value
          : null,
        x_cost_semantics:
          typeof effectPreview.x_cost_semantics === "string"
            ? effectPreview.x_cost_semantics
            : null
      }
    },
    relic: {
      id: typeof relic.id === "string" ? relic.id : null,
      title: normalizeAgentText(relic.title),
      description: normalizeAgentText(relic.description),
      rarity: typeof relic.rarity === "string" ? relic.rarity : null
    },
    potion: {
      id: typeof potion.id === "string" ? potion.id : null,
      title: normalizeAgentText(potion.title),
      description: normalizeAgentText(potion.description),
      rarity: typeof potion.rarity === "string" ? potion.rarity : null,
      target_type: typeof potion.target_type === "string" ? potion.target_type : null
    }
  });
}

function summarizeShopPurchaseRequest(request) {
  if (!isPlainObject(request)) {
    return null;
  }

  return {
    action_id: typeof request.action_id === "string" ? request.action_id : null,
    title: normalizeAgentText(request.title),
    item_kind: typeof request.item_kind === "string" ? request.item_kind : null
  };
}

function summarizeCardSelectionOptionForAgent(option, fallbackIndex = null) {
  if (!isPlainObject(option)) {
    return null;
  }

  const optionIndex = Number.isInteger(option.index) ? option.index : fallbackIndex;
  return {
    index: Number.isInteger(optionIndex) ? optionIndex : null,
    is_selected: option.is_selected === true,
    card: summarizeCardForAgent(option.card)
  };
}

function resolveShopCardRemovalOption(cardSelectionBundle, options) {
  const visibleOptions = Array.isArray(cardSelectionBundle?.card_selection?.options)
    ? cardSelectionBundle.card_selection.options
    : [];
  const normalizedOptions = visibleOptions.map((option, fallbackIndex) => ({
    option,
    option_index: Number.isInteger(option?.index) ? option.index : fallbackIndex,
    title_key: normalizeTextComparisonKey(option?.card?.title)
  }));

  if (Number.isInteger(options?.removeCardIndex)) {
    const indexedMatch = normalizedOptions.find(
      (entry) => entry.option_index === options.removeCardIndex
    );
    if (!indexedMatch) {
      return {
        option: null,
        reason: "shop_card_removal_index_unavailable",
        message: `Card removal option ${options.removeCardIndex} is not available.`,
        available_cards: normalizedOptions
          .map((entry) => summarizeCardSelectionOptionForAgent(entry.option, entry.option_index))
          .filter((entry) => entry !== null)
      };
    }

    return {
      option: {
        ...indexedMatch.option,
        index: indexedMatch.option_index
      },
      match_type: "index",
      compatible_option_count: 1
    };
  }

  const requestedTitleKey = normalizeTextComparisonKey(options?.removeCardTitle);
  if (requestedTitleKey === null) {
    return {
      option: null,
      reason: "shop_card_removal_target_missing",
      message: "No card removal target was provided."
    };
  }

  const titleMatches = normalizedOptions.filter(
    (entry) => entry.title_key !== null && entry.title_key === requestedTitleKey
  );
  if (titleMatches.length <= 0) {
    return {
      option: null,
      reason: "shop_card_removal_title_unavailable",
      message: `No visible card removal option matched ${options.removeCardTitle}.`,
      available_cards: normalizedOptions
        .map((entry) => summarizeCardSelectionOptionForAgent(entry.option, entry.option_index))
        .filter((entry) => entry !== null)
    };
  }

  return {
    option: {
      ...titleMatches[0].option,
      index: titleMatches[0].option_index
    },
    match_type: titleMatches.length === 1 ? "title" : "title_ambiguous",
    compatible_option_count: titleMatches.length
  };
}

async function maybeResolveShopCardRemovalSelection(session, state, options) {
  const executedActions = [];
  const cardSelectionBundle = buildCardSelectionBundle(state);
  const summarizedBundle = summarizeCardSelectionBundleForAgent(cardSelectionBundle);

  if (!cardSelectionBundle.in_card_selection_flow || !cardSelectionBundle.card_selection.visible) {
    return {
      performed: false,
      required_but_missing: false,
      failed: false,
      executed_actions: executedActions,
      card_selection_bundle: summarizedBundle,
      state
    };
  }

  if (options?.removeCardTitle === undefined && options?.removeCardIndex === undefined) {
    return {
      performed: false,
      required_but_missing: true,
      failed: false,
      executed_actions: executedActions,
      card_selection_bundle: summarizedBundle,
      state
    };
  }

  const selectionResolution = resolveShopCardRemovalOption(cardSelectionBundle, options);
  if (!selectionResolution.option || !Number.isInteger(selectionResolution.option.index)) {
    return {
      performed: false,
      required_but_missing: false,
      failed: true,
      reason: selectionResolution.reason ?? "shop_card_removal_target_unavailable",
      message:
        selectionResolution.message ??
        "The requested card removal target is not available in the current selection.",
      executed_actions: executedActions,
      card_selection_bundle: summarizedBundle,
      available_cards: selectionResolution.available_cards ?? [],
      state
    };
  }

  const selectActionId =
    typeof selectionResolution.option?.action_id === "string" &&
    selectionResolution.option.action_id
      ? selectionResolution.option.action_id
      : `card_selection:select:${selectionResolution.option.index}`;
  if (!cardSelectionBundle.non_automation_action_ids.includes(selectActionId)) {
    return {
      performed: false,
      required_but_missing: false,
      failed: true,
      reason: "shop_card_removal_select_action_unavailable",
      message: `The bridge did not expose ${selectActionId} for the current card selection.`,
      executed_actions: executedActions,
      card_selection_bundle: summarizedBundle,
      available_cards:
        selectionResolution.available_cards ??
        cardSelectionBundle.card_selection.options
          .map((option, fallbackIndex) =>
            summarizeCardSelectionOptionForAgent(option, fallbackIndex)
          )
          .filter((entry) => entry !== null),
      state
    };
  }

  const selectedCardSummary = summarizeCardForAgent(selectionResolution.option.card);
  const waitAfterMs = Number.isInteger(options?.waitAfterMs)
    ? options.waitAfterMs
    : DEFAULT_ACTION_WAIT_MS;

  const selectResult = await performBridgeAction(session, selectActionId, waitAfterMs);
  executedActions.push(summarizeExecutedAction(selectResult));
  state = selectResult.state;

  let finalBundle = buildCardSelectionBundle(state);
  if (finalBundle.in_card_selection_flow && finalBundle.card_selection.visible) {
    if (!finalBundle.non_automation_action_ids.includes("card_selection:confirm")) {
      return {
        performed: false,
        required_but_missing: false,
        failed: true,
        reason: "shop_card_removal_confirmation_unavailable",
        message: "Card removal selection stayed open, but confirm was not available.",
        executed_actions: executedActions,
        card_selection_bundle: summarizeCardSelectionBundleForAgent(finalBundle),
        available_cards: finalBundle.card_selection.options
          .map((option, fallbackIndex) =>
            summarizeCardSelectionOptionForAgent(option, fallbackIndex)
          )
          .filter((entry) => entry !== null),
        state
      };
    }

    const confirmResult = await performBridgeAction(session, "card_selection:confirm", waitAfterMs);
    executedActions.push(summarizeExecutedAction(confirmResult));
    state = confirmResult.state;
    finalBundle = buildCardSelectionBundle(state);
  }

  return {
    performed: true,
    required_but_missing: false,
    failed: false,
    removed_card: {
      requested_title:
        typeof options?.removeCardTitle === "string" ? normalizeAgentText(options.removeCardTitle) : null,
      requested_index: Number.isInteger(options?.removeCardIndex) ? options.removeCardIndex : null,
      selected_option_index: selectionResolution.option.index,
      match_type: selectionResolution.match_type,
      compatible_option_count: selectionResolution.compatible_option_count,
      card: selectedCardSummary
    },
    executed_actions: executedActions,
    card_selection_bundle: summarizeCardSelectionBundleForAgent(finalBundle),
    state
  };
}

function compactStringArray(values, limit = 3) {
  const seen = new Set();
  const result = [];

  for (const value of values) {
    if (typeof value !== "string") {
      continue;
    }

    const trimmed = value.trim();
    if (!trimmed || seen.has(trimmed)) {
      continue;
    }

    seen.add(trimmed);
    result.push(trimmed);
    if (result.length >= limit) {
      break;
    }
  }

  return result;
}

function normalizeAgentText(value, options = {}) {
  if (typeof value !== "string") {
    return null;
  }

  let text = value.replace(/\r\n?/g, "\n").trim();
  if (!text) {
    return null;
  }

  text = text.replace(/\[img\]([\s\S]*?)\[\/img\]/gi, (_, inner) =>
    createImageTagMarker(inner)
  );
  text = collapseImageTagMarkers(text);
  text = text.replace(/\[(\/)?[a-z_]+(?:=[^\]]+)?\]/gi, "");
  text = resolveAgentTextPlaceholders(text, options);
  text = text.replace(/[ \t]+\n/g, "\n").replace(/\n[ \t]+/g, "\n");
  text = text.replace(/[ \t]{2,}/g, " ").trim();

  return text || null;
}

const IMAGE_TAG_MARKER_PREFIX = "<<sts2-icon:";
const IMAGE_TAG_MARKER_SUFFIX = ">>";

function createImageTagMarker(inner) {
  const kind = recognizeImageTagKind(inner);
  if (kind) {
    return `${IMAGE_TAG_MARKER_PREFIX}${kind}${IMAGE_TAG_MARKER_SUFFIX}`;
  }

  return `${IMAGE_TAG_MARKER_PREFIX}unknown:${getImageTagDebugName(
    inner
  )}${IMAGE_TAG_MARKER_SUFFIX}`;
}

function collapseImageTagMarkers(text) {
  if (
    typeof text !== "string" ||
    !text ||
    !text.includes(IMAGE_TAG_MARKER_PREFIX)
  ) {
    return text;
  }

  let collapsed = collapseKnownImageTagMarkers(
    text,
    "energy",
    "点能量",
    true
  );
  collapsed = collapseKnownImageTagMarkers(collapsed, "star", "点星辉", true);

  const markerPattern = new RegExp(
    `${escapeRegex(IMAGE_TAG_MARKER_PREFIX)}([^>]+)${escapeRegex(
      IMAGE_TAG_MARKER_SUFFIX
    )}`,
    "gi"
  );
  return collapsed.replace(markerPattern, (_, token) => {
    const normalizedToken = String(token || "");
    if (normalizedToken.toLowerCase().startsWith("unknown:")) {
      return `图标:${normalizedToken.slice("unknown:".length)}`;
    }

    return `图标:${normalizedToken}`;
  });
}

function collapseKnownImageTagMarkers(
  text,
  kind,
  unitLabel,
  allowCountPrefix = false
) {
  if (typeof text !== "string" || !text) {
    return "";
  }

  const marker = `${IMAGE_TAG_MARKER_PREFIX}${kind}${IMAGE_TAG_MARKER_SUFFIX}`;
  let collapsed = text;

  if (allowCountPrefix) {
    const countedPattern = new RegExp(`(\\d+)\\s*${escapeRegex(marker)}`, "gi");
    collapsed = collapsed.replace(
      countedPattern,
      (_, count) => `${count}${unitLabel}`
    );
  }

  const repeatedPattern = new RegExp(`(?:${escapeRegex(marker)}\\s*)+`, "gi");
  const markerPattern = new RegExp(escapeRegex(marker), "gi");
  return collapsed.replace(repeatedPattern, (match) => {
    const count = (match.match(markerPattern) || []).length;
    return count > 0 ? `${count}${unitLabel}` : match;
  });
}

function recognizeImageTagKind(inner) {
  const debugName = getImageTagDebugName(inner);
  if (!debugName) {
    return null;
  }

  if (debugName.toLowerCase() === "star_icon") {
    return "star";
  }

  if (debugName.toLowerCase().endsWith("_energy_icon")) {
    return "energy";
  }

  return null;
}

function getImageTagDebugName(inner) {
  if (typeof inner !== "string" || !inner.trim()) {
    return "empty";
  }

  let normalized = inner.trim();
  normalized = normalized.split(/[\\/]/).pop() || normalized;
  normalized = normalized.replace(/\.[^.]+$/, "");
  normalized = normalized
    .toLowerCase()
    .replace(/[^a-z0-9_:-]+/g, "");

  return normalized || "unknown";
}

function escapeRegex(value) {
  return String(value || "").replace(/[.*+?^${}()|[\]\\]/g, "\\$&");
}

function resolveAgentTextPlaceholders(text, options = {}) {
  if (typeof text !== "string" || !text.includes("{")) {
    return text;
  }

  const tokenMap = isPlainObject(options.token_map) ? options.token_map : {};
  return text.replace(/\{([^}:]+)(?::[^}]*)?\}/g, (_, rawToken) => {
    const token = String(rawToken || "").trim();
    if (!token) {
      return "";
    }

    if (Object.prototype.hasOwnProperty.call(tokenMap, token)) {
      return String(tokenMap[token]);
    }

    if (token === "Energy") {
      return "1点能量";
    }

    return token;
  });
}

function compactNumberArray(values, limit = 3) {
  const seen = new Set();
  const result = [];

  for (const value of values) {
    if (!Number.isFinite(value)) {
      continue;
    }

    if (seen.has(value)) {
      continue;
    }

    seen.add(value);
    result.push(value);
    if (result.length >= limit) {
      break;
    }
  }

  return result;
}

function summarizeActionForAgent(action) {
  if (!isPlainObject(action)) {
    return null;
  }

  const summary = {
    action_id: typeof action.action_id === "string" ? action.action_id : null,
    kind: typeof action.kind === "string" ? action.kind : null,
    label: normalizeAgentText(action.label)
  };

  const targetName =
    typeof action.target_name === "string"
      ? normalizeAgentText(action.target_name)
      : normalizeAgentText(action?.target?.name);
  const targetSide =
    typeof action.target_side === "string"
      ? action.target_side
      : typeof action?.target?.side === "string"
        ? action.target.side
        : null;
  const targetCombatId = Number.isFinite(action.target_combat_id)
    ? action.target_combat_id
    : Number.isFinite(action?.target?.combat_id)
      ? action.target.combat_id
      : null;
  const targetActionSuffix =
    typeof action.target_action_suffix === "string"
      ? action.target_action_suffix
      : typeof action?.target_mapping?.action_suffix === "string"
        ? action.target_mapping.action_suffix
        : null;

  if (
    targetName !== null ||
    targetSide !== null ||
    targetCombatId !== null ||
    targetActionSuffix !== null
  ) {
    summary.target = {
      action_suffix: targetActionSuffix,
      combat_id: targetCombatId,
      name: targetName,
      side: targetSide
    };
  }

  return summary;
}

function summarizeCardEffectPreviewForAgent(card) {
  if (!isPlainObject(card)) {
    return null;
  }

  const effectPreview = isPlainObject(card.effect_preview) ? card.effect_preview : {};
  const summary = {};
  const numericMappings = [
    ["damage", ["damage", "total_damage"]],
    ["damage_per_hit", ["damage_per_hit"]],
    ["hits", ["hits"]],
    ["block", ["block", "total_block"]],
    ["draw", ["draw"]],
    ["heal", ["heal"]],
    ["hp_loss", ["hp_loss"]],
    ["weak", ["weak"]],
    ["vulnerable", ["vulnerable"]],
    ["poison", ["poison"]],
    ["strength", ["strength"]],
    ["dexterity", ["dexterity"]],
    ["summon", ["summon"]],
    ["extra_damage", ["extra_damage"]],
    ["x_cost_value", ["x_cost_value"]]
  ];

  for (const [targetKey, sourceKeys] of numericMappings) {
    for (const sourceKey of sourceKeys) {
      if (Number.isFinite(effectPreview[sourceKey])) {
        summary[targetKey] = effectPreview[sourceKey];
        break;
      }
    }
  }

  if (typeof effectPreview.x_cost_semantics === "string" && effectPreview.x_cost_semantics.trim()) {
    summary.x_cost_semantics = effectPreview.x_cost_semantics;
  }

  return Object.keys(summary).length > 0 ? summary : null;
}

function filterNonAutomationActions(actions) {
  if (!Array.isArray(actions)) {
    return [];
  }

  return actions.filter(
    (action) =>
      isPlainObject(action) &&
      typeof action.action_id === "string" &&
      !action.action_id.startsWith("automation:")
  );
}

function summarizeActionsForAgent(actions, screen = null) {
  if (!Array.isArray(actions)) {
    return [];
  }

  if (isCrystalSphereScreen(screen)) {
    const otherActions = [];
    const crystalSphereOptions = [];

    for (const action of actions) {
      if (isCrystalSphereEventOptionAction(action)) {
        crystalSphereOptions.push(action);
        continue;
      }

      const summary = summarizeActionForAgent(action);
      if (summary !== null) {
        otherActions.push(summary);
      }
    }

    const crystalSphereGroup = summarizeCrystalSphereActionGroupsForAgent(crystalSphereOptions);
    if (crystalSphereGroup !== null) {
      otherActions.push({
        kind: "event_option_group",
        group: "crystal_sphere",
        ...crystalSphereGroup
      });
    }

    return otherActions;
  }

  return actions.map(summarizeActionForAgent).filter((action) => action !== null);
}

function shouldIncludeCardDescriptionAlongsideEffect(description, effect) {
  if (typeof description !== "string" || !description.trim()) {
    return false;
  }

  if (typeof effect !== "string" || !effect.trim()) {
    return true;
  }

  if (description.includes("\n")) {
    return true;
  }

  const sentenceBreakCount = (description.match(/[。！？!?]/g) || []).length;
  if (sentenceBreakCount >= 2) {
    return true;
  }

  return description.length >= effect.length + 8;
}

function summarizeCardForAgent(card) {
  if (!isPlainObject(card)) {
    return null;
  }

  const summary = {
    id: typeof card.id === "string" && card.id.trim() ? card.id : null,
    title: normalizeAgentText(card.title),
    cost: Number.isInteger(card.resolved_energy_cost)
      ? card.resolved_energy_cost
      : Number.isInteger(card.cost)
        ? card.cost
        : null
  };
  const starCost = readAgentCardStarCost(card);

  const effect = typeof card?.effect_preview?.summary === "string"
    ? normalizeAgentText(card.effect_preview.summary)
    : typeof card?.effect === "string" && card.effect.trim()
      ? normalizeAgentText(card.effect)
    : null;
  const description = typeof card.description === "string" && card.description.trim()
    ? normalizeAgentText(card.description)
    : null;

  if (typeof card.type === "string" && card.type.trim()) {
    summary.type = card.type;
  }

  if (typeof card.rarity === "string" && card.rarity.trim()) {
    summary.rarity = card.rarity;
  }

  if (typeof card.target_type === "string" && card.target_type.trim()) {
    summary.target = card.target_type;
  } else if (typeof card.target === "string" && card.target.trim()) {
    summary.target = card.target;
  }

  if (Number.isInteger(card.current_upgrade_level) && card.current_upgrade_level >= 0) {
    summary.upgrade_level = card.current_upgrade_level;
  } else if (Number.isInteger(card.upgrade_level) && card.upgrade_level >= 0) {
    summary.upgrade_level = card.upgrade_level;
  }

  if (card.costs_x === true || card.x_cost === true) {
    summary.x_cost = true;
  }

  if (Number.isInteger(card.current_star_cost) && card.current_star_cost >= 0) {
    summary.star = card.current_star_cost;
  } else if (Number.isInteger(card.star) && card.star >= 0) {
    summary.star = card.star;
  }

  if (card.has_star_cost_x === true || card.star_x === true) {
    summary.star_x = true;
  }

  if (starCost !== null) {
    summary.star_cost = starCost;
  }

  if (effect && effect.trim()) {
    summary.effect = effect;
  }

  if (shouldIncludeCardDescriptionAlongsideEffect(description, effect)) {
    summary.description = description;
  }

  const effectPreview = summarizeCardEffectPreviewForAgent(card);
  if (effectPreview !== null) {
    summary.effect_preview = effectPreview;
  }

  return summary;
}

function readAgentCardStarCost(card) {
  if (!isPlainObject(card)) {
    return null;
  }

  if (card.has_star_cost_x === true || card.star_x === true || card.star_cost === "X") {
    return "X";
  }

  if (Number.isInteger(card.current_star_cost) && card.current_star_cost >= 0) {
    return card.current_star_cost;
  }

  if (Number.isInteger(card.star) && card.star >= 0) {
    return card.star;
  }

  if (Number.isInteger(card.star_cost) && card.star_cost >= 0) {
    return card.star_cost;
  }

  return null;
}

function summarizePowerForAgent(power) {
  if (!isPlainObject(power)) {
    return null;
  }

  const summary = {
    title: normalizeAgentText(power.title),
    amount: Number.isFinite(power.amount) ? power.amount : null
  };

  if (typeof power.description === "string" && power.description.trim()) {
    summary.description = normalizeAgentText(power.description);
  }
  if (Number.isFinite(power.display_amount)) {
    summary.display_amount = power.display_amount;
  }
  if (typeof power.type === "string") {
    summary.type = power.type;
  }
  if (typeof power.stack_type === "string") {
    summary.stack_type = power.stack_type;
  }

  return summary;
}

function summarizePotionForAgent(potion) {
  if (!isPlainObject(potion)) {
    return null;
  }

  const summary = {};
  if (typeof potion.title === "string" && potion.title.trim()) {
    summary.title = normalizeAgentText(potion.title);
  }

  if (typeof potion.description === "string" && potion.description.trim()) {
    summary.description = normalizeAgentText(potion.description);
  }

  return Object.keys(summary).length > 0 ? summary : null;
}

function summarizeIntentCandidateForAgent(intent) {
  if (!isPlainObject(intent)) {
    return null;
  }

  const totalDamage = Number.isFinite(intent.total_damage) ? intent.total_damage : null;
  const repeats = Number.isFinite(intent.repeats) ? intent.repeats : null;
  const damagePerHit = Number.isFinite(intent.damage_per_hit)
    ? intent.damage_per_hit
    : (
      totalDamage !== null &&
      repeats !== null &&
      repeats > 0 &&
      Number.isInteger(totalDamage) &&
      Number.isInteger(repeats) &&
      totalDamage % repeats === 0
    )
      ? totalDamage / repeats
      : null;
  const description = typeof intent.description === "string" && intent.description.trim()
    ? normalizeAgentText(intent.description)
    : typeof intent.text === "string" && intent.text.trim()
      ? normalizeAgentText(intent.text)
      : null;

  const summary = {
    intent_type: typeof intent.intent_type === "string" ? intent.intent_type : null,
    intent_class: typeof intent.intent_class === "string" ? intent.intent_class : null,
    title: typeof intent.title === "string" && intent.title.trim()
      ? normalizeAgentText(intent.title)
      : null,
    label: typeof intent.label === "string" && intent.label.trim()
      ? normalizeAgentText(intent.label)
      : null,
    description
  };

  if (totalDamage !== null) {
    summary.total_damage = totalDamage;
  }
  if (repeats !== null) {
    summary.repeats = repeats;
  }
  if (damagePerHit !== null) {
    summary.damage_per_hit = damagePerHit;
  }

  return summary;
}

function summarizeRelicForAgent(relic) {
  if (!isPlainObject(relic)) {
    return null;
  }

  return {
    title: normalizeAgentText(relic.title)
  };
}

function summarizeShopItemForAgent(item) {
  if (!isPlainObject(item)) {
    return null;
  }

  const summary = {
    index: Number.isInteger(item.index) ? item.index : null,
    item_kind: typeof item.item_kind === "string" ? item.item_kind : null,
    title: normalizeAgentText(item.title),
    description: normalizeAgentText(item.description),
    cost: Number.isFinite(item.cost) ? item.cost : null,
    is_affordable: item.is_affordable === true
  };

  if (isPlainObject(item.card)) {
    summary.card = summarizeCardForAgent(item.card);
  }

  if (isPlainObject(item.relic)) {
    summary.relic = summarizeRelicForAgent(item.relic);
  }

  if (isPlainObject(item.potion)) {
    summary.potion = summarizePotionForAgent(item.potion);
  }

  return summary;
}

function summarizeIntentForAgent(intent) {
  if (!isPlainObject(intent)) {
    return null;
  }

  const title = typeof intent.title === "string" && intent.title.trim()
    ? normalizeAgentText(intent.title)
    : null;
  const rawCandidates = Array.isArray(intent.intents)
    ? intent.intents.filter((entry) => isPlainObject(entry))
    : [];
  const filteredCandidates = title
    ? rawCandidates.filter((entry) => entry.title === title)
    : rawCandidates;
  const candidates = filteredCandidates.length > 0 ? filteredCandidates : rawCandidates;
  const summarizedCandidates = candidates
    .map(summarizeIntentCandidateForAgent)
    .filter((entry) => entry !== null);
  const primaryCandidate = summarizedCandidates[0] ?? null;

  const labels = compactStringArray(
    candidates
      .map((entry) => normalizeAgentText(entry.label))
      .filter((label) => typeof label === "string" && !label.includes("LocString"))
  );
  const texts = compactStringArray(
    candidates.map((entry) => normalizeAgentText(entry.description ?? entry.text))
  );
  const totalDamages = compactNumberArray(candidates.map((entry) => entry.total_damage));
  const repeats = compactNumberArray(candidates.map((entry) => entry.repeats));
  const damagePerHits = compactNumberArray(candidates.map((entry) => entry.damage_per_hit));

  const summary = {
    state_id: typeof intent.state_id === "string" ? intent.state_id : null,
    title
  };

  if (primaryCandidate?.intent_type) {
    summary.intent_type = primaryCandidate.intent_type;
  }
  if (primaryCandidate?.intent_class) {
    summary.intent_class = primaryCandidate.intent_class;
  }

  if (labels.length > 0) {
    summary.label = labels[0];
    if (labels.length > 1) {
      summary.alt_label_count = labels.length - 1;
    }
  }

  if (texts.length > 0) {
    summary.text = texts[0];
    summary.description = texts[0];
    if (texts.length > 1) {
      summary.alt_text_count = texts.length - 1;
    }
  } else if (primaryCandidate?.description) {
    summary.text = primaryCandidate.description;
    summary.description = primaryCandidate.description;
  }

  if (totalDamages.length > 0) {
    summary.total_damage = totalDamages[0];
    if (totalDamages.length > 1) {
      summary.alt_total_damage_count = totalDamages.length - 1;
    }
  } else if (primaryCandidate?.total_damage !== undefined) {
    summary.total_damage = primaryCandidate.total_damage;
  }

  if (repeats.length > 0) {
    summary.repeats = repeats[0];
    if (repeats.length > 1) {
      summary.alt_repeats_count = repeats.length - 1;
    }
  } else if (primaryCandidate?.repeats !== undefined) {
    summary.repeats = primaryCandidate.repeats;
  }

  if (damagePerHits.length > 0) {
    summary.damage_per_hit = damagePerHits[0];
    if (damagePerHits.length > 1) {
      summary.alt_damage_per_hit_count = damagePerHits.length - 1;
    }
  } else if (primaryCandidate?.damage_per_hit !== undefined) {
    summary.damage_per_hit = primaryCandidate.damage_per_hit;
  }

  if (summarizedCandidates.length > 0) {
    summary.candidate_count = summarizedCandidates.length;
    summary.intents = summarizedCandidates.slice(0, 3);
  }

  return summary;
}

function getEnemyMetadataRegistry() {
  if (cachedEnemyMetadataRegistry !== undefined) {
    return cachedEnemyMetadataRegistry;
  }

  try {
    const raw = fs.readFileSync(ENEMY_METADATA_PATH, "utf8");
    const parsed = JSON.parse(raw);
    cachedEnemyMetadataRegistry = isPlainObject(parsed) ? parsed : {};
  } catch (_error) {
    cachedEnemyMetadataRegistry = {};
  }

  return cachedEnemyMetadataRegistry;
}

function getEnemyMetadataForAgent(enemyId) {
  const normalizedId = typeof enemyId === "string" ? enemyId.trim() : "";
  if (!normalizedId) {
    return null;
  }
  const registry = getEnemyMetadataRegistry();
  return isPlainObject(registry[normalizedId]) ? registry[normalizedId] : null;
}

function normalizeEnemyTraitEntryForAgent(entry, categoryHint = null) {
  if (!isPlainObject(entry)) {
    return null;
  }

  const trait = normalizeAgentText(
    entry.trait ?? entry.effect_type ?? entry.state ?? entry.condition
  );
  if (!trait) {
    return null;
  }

  const normalized = {
    trait,
    description: normalizeAgentText(entry.description) ?? trait
  };

  const category = typeof categoryHint === "string" && categoryHint.trim()
    ? categoryHint.trim()
    : typeof entry.category === "string" && entry.category.trim()
      ? entry.category.trim()
      : null;
  if (category) {
    normalized.category = category;
  }
  if (typeof entry.trigger_type === "string" && entry.trigger_type.trim()) {
    normalized.trigger_type = entry.trigger_type.trim();
  }
  if (typeof entry.condition === "string" && entry.condition.trim()) {
    normalized.condition = entry.condition.trim();
  }
  if (typeof entry.effect_type === "string" && entry.effect_type.trim()) {
    normalized.effect_type = entry.effect_type.trim();
  }
  if (typeof entry.state === "string" && entry.state.trim()) {
    normalized.state = entry.state.trim();
  }
  if (Number.isFinite(entry.threshold)) {
    normalized.threshold = entry.threshold;
  }
  if (Number.isFinite(entry.effect_amount)) {
    normalized.effect_amount = entry.effect_amount;
  } else if (Number.isFinite(entry.amount)) {
    normalized.effect_amount = entry.amount;
  }
  if (typeof entry.severity === "string" && entry.severity.trim()) {
    normalized.severity = entry.severity.trim();
  }

  return normalized;
}

function dedupeEnemyTraitEntriesForAgent(entries) {
  const result = [];
  const seen = new Set();
  for (const entry of entries) {
    if (!isPlainObject(entry)) {
      continue;
    }
    const key = JSON.stringify(entry, Object.keys(entry).sort());
    if (seen.has(key)) {
      continue;
    }
    seen.add(key);
    result.push(entry);
  }
  return result;
}

function summarizeEnemyDangerProfileForAgent(dangerProfile) {
  if (!isPlainObject(dangerProfile)) {
    return null;
  }
  const summary = {};
  for (const key of [
    "burst",
    "attrition",
    "scaling",
    "retaliation",
    "summon_pressure",
    "debuff_pressure",
    "phase_complexity",
    "volatility",
    "target_priority"
  ]) {
    if (Number.isFinite(dangerProfile[key])) {
      summary[key] = dangerProfile[key];
    }
  }
  if (typeof dangerProfile.notes === "string" && dangerProfile.notes.trim()) {
    summary.notes = normalizeAgentText(dangerProfile.notes);
  }
  return Object.keys(summary).length > 0 ? summary : null;
}

function summarizeTargetPriorityHintsForAgent(hints) {
  if (!Array.isArray(hints)) {
    return [];
  }
  return hints
    .map((entry) => {
      if (typeof entry === "string") {
        const reason = normalizeAgentText(entry);
        return reason ? { reason } : null;
      }
      if (!isPlainObject(entry)) {
        return null;
      }
      const reason = normalizeAgentText(entry.reason ?? entry.description);
      if (!reason) {
        return null;
      }
      const summary = { reason };
      if (typeof entry.priority === "string" && entry.priority.trim()) {
        summary.priority = entry.priority.trim();
      }
      if (typeof entry.when === "string" && entry.when.trim()) {
        summary.when = entry.when.trim();
      }
      return summary;
    })
    .filter((entry) => entry !== null)
    .slice(0, 4);
}

function inferStructuredEnemyFieldsForAgent(creature) {
  const metadata = getEnemyMetadataForAgent(creature?.model_id);
  const staticTraits = [];
  const reactiveTriggers = [];
  const phaseRules = [];
  const combatTags = new Set();
  const targetPriorityHints = [];

  const addEntry = (bucket, entry, categoryHint = null) => {
    const normalized = normalizeEnemyTraitEntryForAgent(entry, categoryHint);
    if (normalized) {
      bucket.push(normalized);
    }
  };
  const addTag = (tag) => {
    if (typeof tag === "string" && tag.trim()) {
      combatTags.add(tag.trim());
    }
  };
  const mergeCollection = (source, bucket, categoryHint = null) => {
    if (!Array.isArray(source)) {
      return;
    }
    for (const entry of source) {
      addEntry(bucket, entry, categoryHint);
    }
  };

  if (isPlainObject(metadata)) {
    mergeCollection(metadata.static_traits, staticTraits, "static");
    mergeCollection(metadata.reactive_triggers, reactiveTriggers, "reactive");
    mergeCollection(metadata.phase_rules, phaseRules, "phase");
    for (const tag of metadata.combat_tags ?? []) {
      addTag(tag);
    }
    targetPriorityHints.push(...summarizeTargetPriorityHintsForAgent(metadata.target_priority_hints));
  }

  mergeCollection(creature?.static_traits, staticTraits, "static");
  mergeCollection(creature?.reactive_triggers, reactiveTriggers, "reactive");
  mergeCollection(creature?.phase_rules, phaseRules, "phase");
  for (const tag of creature?.combat_tags ?? []) {
    addTag(tag);
  }
  targetPriorityHints.push(...summarizeTargetPriorityHintsForAgent(creature?.target_priority_hints));

  const powerTexts = Array.isArray(creature?.powers)
    ? creature.powers
      .map((power) => `${normalizeAgentText(power?.title) ?? ""} ${normalizeAgentText(power?.description) ?? ""}`.trim())
      .filter((text) => text.length > 0)
    : [];
  const nameText = [
    normalizeAgentText(creature?.name),
    typeof creature?.model_id === "string" ? creature.model_id.trim() : null,
    normalizeAgentText(creature?.intent?.description),
    ...powerTexts
  ]
    .filter((text) => typeof text === "string" && text.length > 0)
    .join(" ")
    .toLowerCase();

  if (nameText.includes("thorn") || nameText.includes("spiny") || nameText.includes("retali") || nameText.includes("contact")) {
    addTag("retaliation");
    addEntry(reactiveTriggers, {
      trait: "retaliate",
      description: "Contact / thorns style retaliation",
      trigger_type: "on_hit",
      condition: "contact",
      effect_type: "retaliate"
    }, "reactive");
  }
  if (nameText.includes("split")) {
    addTag("phase_shift");
    addEntry(phaseRules, {
      trait: "split",
      description: "Split or threshold clone behavior",
      trigger_type: "on_hp_threshold",
      condition: "threshold_crossed",
      effect_type: "split"
    }, "phase");
  }
  if (nameText.includes("phase") || nameText.includes("threshold")) {
    addTag("phase_shift");
    addEntry(phaseRules, {
      trait: "phase_shift",
      description: "Threshold / phase transition behavior",
      trigger_type: "on_hp_threshold",
      condition: "threshold_crossed",
      effect_type: "phase_shift"
    }, "phase");
  }
  if (nameText.includes("summon") || nameText.includes("spawn") || nameText.includes("nexus") || nameText.includes("progenitor") || nameText.includes("queen") || nameText.includes("egg")) {
    addTag("summoner");
  }
  if (nameText.includes("boss") || nameText.includes("doormaker") || nameText.includes("insatiable") || nameText.includes("matriarch") || nameText.includes("test subject")) {
    addTag("boss");
  }

  let dangerProfile = summarizeEnemyDangerProfileForAgent(creature?.danger_profile)
    ?? summarizeEnemyDangerProfileForAgent(metadata?.danger_profile)
    ?? null;
  if (dangerProfile === null) {
    const heuristic = {};
    if (combatTags.has("retaliation")) {
      heuristic.retaliation = 4;
      heuristic.attrition = 3;
    }
    if (combatTags.has("summoner")) {
      heuristic.summon_pressure = 4;
      heuristic.target_priority = 4;
    }
    if (combatTags.has("phase_shift")) {
      heuristic.phase_complexity = 4;
    }
    dangerProfile = Object.keys(heuristic).length > 0 ? heuristic : null;
  }

  return {
    static_traits: dedupeEnemyTraitEntriesForAgent(staticTraits).slice(0, 6),
    reactive_triggers: dedupeEnemyTraitEntriesForAgent(reactiveTriggers).slice(0, 6),
    phase_rules: dedupeEnemyTraitEntriesForAgent(phaseRules).slice(0, 6),
    combat_tags: [...combatTags],
    danger_profile: dangerProfile,
    target_priority_hints: dedupeEnemyTraitEntriesForAgent(targetPriorityHints).slice(0, 4)
  };
}

function summarizeCreatureForAgent(creature) {
  if (!isPlainObject(creature)) {
    return null;
  }

  const structured = inferStructuredEnemyFieldsForAgent(creature);
  const summary = {
    combat_id: Number.isFinite(creature.combat_id) ? creature.combat_id : null,
    model_id: typeof creature.model_id === "string" ? creature.model_id.trim() : null,
    side: typeof creature.side === "string" ? creature.side : null,
    name: normalizeAgentText(creature.name),
    current_hp: Number.isFinite(creature.current_hp) ? creature.current_hp : null,
    max_hp: Number.isFinite(creature.max_hp) ? creature.max_hp : null,
    block: Number.isFinite(creature.block) ? creature.block : 0,
    is_alive:
      typeof creature.is_alive === "boolean"
        ? creature.is_alive
        : Number.isFinite(creature.current_hp)
          ? creature.current_hp > 0
          : null,
    is_hittable: typeof creature.is_hittable === "boolean" ? creature.is_hittable : null,
    powers: Array.isArray(creature.powers)
      ? creature.powers.map(summarizePowerForAgent).filter((power) => power !== null)
      : []
  };

  if (structured.static_traits.length > 0) {
    summary.static_traits = structured.static_traits;
  }
  if (structured.reactive_triggers.length > 0) {
    summary.reactive_triggers = structured.reactive_triggers;
  }
  if (structured.phase_rules.length > 0) {
    summary.phase_rules = structured.phase_rules;
  }
  if (structured.combat_tags.length > 0) {
    summary.combat_tags = structured.combat_tags;
  }
  if (isPlainObject(structured.danger_profile)) {
    summary.danger_profile = structured.danger_profile;
  }
  if (structured.target_priority_hints.length > 0) {
    summary.target_priority_hints = structured.target_priority_hints;
  }

  if (isPlainObject(creature.intent)) {
    summary.intent = summarizeIntentForAgent(creature.intent);
  }

  return summary;
}

function summarizeRewardForAgent(reward) {
  if (!isPlainObject(reward)) {
    return null;
  }

  const rewardType = typeof reward.reward_type === "string" ? reward.reward_type : null;
  const summary = {
    reward_type: rewardType,
    description: normalizeAgentText(reward.description)
  };

  if (rewardType === "gold" && Number.isFinite(reward.amount)) {
    summary.amount = reward.amount;
  }

  if (rewardType === "relic" && isPlainObject(reward.relic)) {
    summary.relic = summarizeRelicForAgent(reward.relic);
  }

  if (rewardType === "potion" && isPlainObject(reward.potion)) {
    summary.potion = summarizePotionForAgent(reward.potion);
  }

  if (rewardType === "card") {
    summary.can_skip = reward.can_skip === true;
    summary.cards = Array.isArray(reward.cards)
      ? reward.cards.map(summarizeCardForAgent).filter((card) => card !== null)
      : [];
  }

  return summary;
}

function summarizeRewardBundleForAgent(bundle) {
  if (!isPlainObject(bundle)) {
    return bundle;
  }

  const rewards = isPlainObject(bundle.rewards) ? bundle.rewards : {};
  const cardRewardSelection = isPlainObject(bundle.card_reward_selection)
    ? bundle.card_reward_selection
    : {};

  return {
    in_reward_flow: bundle.in_reward_flow === true,
    screen: typeof bundle.screen === "string" ? bundle.screen : null,
    rewards: {
      visible: rewards.visible === true,
      terminal_proceed_visible: rewards.terminal_proceed_visible === true,
      entries: Array.isArray(rewards.entries)
        ? rewards.entries.map((entry) => ({
          index: Number.isInteger(entry?.index) ? entry.index : null,
          action_id: typeof entry?.action_id === "string" ? entry.action_id : null,
          reward: summarizeRewardForAgent(entry?.reward)
        }))
        : []
    },
    card_reward_selection: {
      visible: cardRewardSelection.visible === true,
      skip_visible: cardRewardSelection.skip_visible === true,
      options: Array.isArray(cardRewardSelection.options)
        ? cardRewardSelection.options.map((option) => ({
          index: Number.isInteger(option?.index) ? option.index : null,
          card: summarizeCardForAgent(option?.card)
        }))
        : []
    },
    potion_slots: Array.isArray(bundle.potion_slots)
      ? bundle.potion_slots.map((slot) => ({
        index: Number.isInteger(slot?.index) ? slot.index : null,
        is_empty: slot?.is_empty === true,
        ...(slot?.is_empty === true
          ? {}
          : { potion: summarizePotionForAgent(slot?.potion) })
      }))
      : [],
    empty_potion_slot_count: Number.isFinite(bundle.empty_potion_slot_count)
      ? bundle.empty_potion_slot_count
      : null,
    has_proceed: bundle.has_proceed === true
  };
}

function summarizeRestSiteBundleForAgent(bundle) {
  if (!isPlainObject(bundle)) {
    return bundle;
  }

  const restSite = isPlainObject(bundle.rest_site) ? bundle.rest_site : {};
  const deckUpgrade = isPlainObject(bundle.deck_upgrade_selection)
    ? bundle.deck_upgrade_selection
    : {};

  return {
    in_rest_site_flow: bundle.in_rest_site_flow === true,
    screen: typeof bundle.screen === "string" ? bundle.screen : null,
    map_ready: bundle.map_ready === true,
    rest_site: {
      visible: restSite.visible === true,
      header: normalizeAgentText(restSite.header),
      description: normalizeAgentText(restSite.description),
      proceed_visible: restSite.proceed_visible === true,
      options: Array.isArray(restSite.options)
        ? restSite.options.map((option) => ({
          index: Number.isInteger(option?.index) ? option.index : null,
          option_type: typeof option?.option_type === "string" ? option.option_type : null,
          title: normalizeAgentText(option?.title),
          description: normalizeAgentText(option?.description)
        }))
        : []
    },
    deck_upgrade_selection: {
      visible: deckUpgrade.visible === true,
      options: Array.isArray(deckUpgrade.options)
        ? deckUpgrade.options.map((option) => ({
          index: Number.isInteger(option?.index) ? option.index : null,
          card: summarizeCardForAgent(option?.card),
          upgrade_preview: summarizeCardForAgent(option?.upgrade_preview),
          is_selected: option?.is_selected === true
        }))
        : []
    },
    non_automation_action_ids: Array.isArray(bundle.non_automation_action_ids)
      ? bundle.non_automation_action_ids
      : []
  };
}

function summarizeShopBundleForAgent(bundle) {
  if (!isPlainObject(bundle)) {
    return bundle;
  }

  const shop = isPlainObject(bundle.shop) ? bundle.shop : {};

  return {
    in_shop_flow: bundle.in_shop_flow === true,
    screen: typeof bundle.screen === "string" ? bundle.screen : null,
    shop: {
      visible: shop.visible === true,
      is_open: shop.is_open === true,
      gold: Number.isFinite(shop.gold) ? shop.gold : null,
      merchant_button_visible: shop.merchant_button_visible === true,
      back_button_visible: shop.back_button_visible === true,
      proceed_visible: shop.proceed_visible === true,
      items: Array.isArray(shop.items)
        ? shop.items.map(summarizeShopItemForAgent).filter((item) => item !== null)
        : []
    },
    non_automation_action_ids: Array.isArray(bundle.non_automation_action_ids)
      ? bundle.non_automation_action_ids
      : []
  };
}

function summarizeRestSiteBundleForResolutionAgent(bundle) {
  if (!isPlainObject(bundle)) {
    return bundle;
  }

  const restSite = isPlainObject(bundle.rest_site) ? bundle.rest_site : {};
  const deckUpgrade = isPlainObject(bundle.deck_upgrade_selection)
    ? bundle.deck_upgrade_selection
    : {};
  const screen = typeof bundle.screen === "string" ? bundle.screen : null;
  const result = {};
  if (screen) {
    result.screen = screen;
  }

  if (deckUpgrade.visible === true) {
    result.deck_upgrade_selection = {
      options: Array.isArray(deckUpgrade.options)
        ? deckUpgrade.options
          .map((option, fallbackIndex) =>
            summarizeDeckUpgradeChoiceForResolutionAgent(option, fallbackIndex)
          )
          .filter((option) => option !== null)
        : []
    };
    return result;
  }

  if (
    restSite.visible === true ||
    typeof restSite.header === "string" ||
    typeof restSite.description === "string" ||
    Array.isArray(restSite.options)
  ) {
    result.rest_site = {
      header: normalizeAgentText(restSite.header),
      description: normalizeAgentText(restSite.description),
      proceed_visible: restSite.proceed_visible === true,
      options: Array.isArray(restSite.options)
        ? restSite.options
          .map((option) => summarizeGenericIndexedOptionForAgent(option))
          .filter((option) => option !== null)
        : []
    };
  }

  return result;
}

function summarizeShopBundleForResolutionAgent(bundle) {
  if (!isPlainObject(bundle)) {
    return bundle;
  }

  const shop = isPlainObject(bundle.shop) ? bundle.shop : {};
  return {
    in_shop_flow: bundle.in_shop_flow === true,
    screen: typeof bundle.screen === "string" ? bundle.screen : null,
    shop: {
      visible: shop.visible === true,
      is_open: shop.is_open === true,
      gold: Number.isFinite(shop.gold) ? shop.gold : null,
      merchant_button_visible: shop.merchant_button_visible === true,
      back_button_visible: shop.back_button_visible === true,
      proceed_visible: shop.proceed_visible === true,
      items: Array.isArray(shop.items)
        ? shop.items.map(summarizeShopItemForAgent).filter((item) => item !== null)
        : []
    }
  };
}

function summarizeCardSelectionBundleForAgent(bundle) {
  if (!isPlainObject(bundle)) {
    return bundle;
  }

  const cardSelection = isPlainObject(bundle.card_selection) ? bundle.card_selection : {};

  return {
    in_card_selection_flow: bundle.in_card_selection_flow === true,
    screen: typeof bundle.screen === "string" ? bundle.screen : null,
    card_selection: {
      visible: cardSelection.visible === true,
      screen_type:
        typeof cardSelection.screen_type === "string" ? cardSelection.screen_type : null,
      prompt: normalizeAgentText(cardSelection.prompt),
      selected_count: Number.isFinite(cardSelection.selected_count)
        ? cardSelection.selected_count
        : null,
      min_select: Number.isFinite(cardSelection.min_select) ? cardSelection.min_select : null,
      max_select: Number.isFinite(cardSelection.max_select) ? cardSelection.max_select : null,
      requires_manual_confirmation:
        typeof cardSelection.requires_manual_confirmation === "boolean"
          ? cardSelection.requires_manual_confirmation
          : null,
      cancelable:
        typeof cardSelection.cancelable === "boolean" ? cardSelection.cancelable : null,
      confirm_visible: cardSelection.confirm_visible === true,
      cancel_visible: cardSelection.cancel_visible === true,
      close_visible: cardSelection.close_visible === true,
      skip_visible: cardSelection.skip_visible === true,
      options: Array.isArray(cardSelection.options)
        ? cardSelection.options.map((option) => ({
          index: Number.isInteger(option?.index) ? option.index : null,
          is_selected: option?.is_selected === true,
          card: summarizeCardForAgent(option?.card)
        }))
        : []
    },
    non_automation_action_ids: Array.isArray(bundle.non_automation_action_ids)
      ? bundle.non_automation_action_ids
      : []
  };
}

function summarizeCardSelectionBundleForResolutionAgent(bundle) {
  if (!isPlainObject(bundle)) {
    return bundle;
  }

  const cardSelection = isPlainObject(bundle.card_selection) ? bundle.card_selection : {};
  return {
    in_card_selection_flow: bundle.in_card_selection_flow === true,
    screen: typeof bundle.screen === "string" ? bundle.screen : null,
    card_selection: {
      visible: cardSelection.visible === true,
      screen_type:
        typeof cardSelection.screen_type === "string" ? cardSelection.screen_type : null,
      prompt: normalizeAgentText(cardSelection.prompt),
      selected_count: Number.isFinite(cardSelection.selected_count)
        ? cardSelection.selected_count
        : null,
      min_select: Number.isFinite(cardSelection.min_select) ? cardSelection.min_select : null,
      max_select: Number.isFinite(cardSelection.max_select) ? cardSelection.max_select : null,
      requires_manual_confirmation:
        typeof cardSelection.requires_manual_confirmation === "boolean"
          ? cardSelection.requires_manual_confirmation
          : null,
      cancelable:
        typeof cardSelection.cancelable === "boolean" ? cardSelection.cancelable : null,
      confirm_visible: cardSelection.confirm_visible === true,
      cancel_visible: cardSelection.cancel_visible === true,
      close_visible: cardSelection.close_visible === true,
      skip_visible: cardSelection.skip_visible === true,
      options: Array.isArray(cardSelection.options)
        ? cardSelection.options
          .map((option, fallbackIndex) =>
            summarizeIndexedCardTitleChoiceForAgent(option, fallbackIndex)
          )
          .filter((option) => option !== null)
        : []
    }
  };
}

function summarizeStateForAgent(state) {
  if (!isPlainObject(state)) {
    return state;
  }

  const screen = typeof state.screen === "string" ? state.screen : null;
  const player = state?.players?.[0];
  const playerCreature = isPlainObject(player?.creature) ? player.creature : {};
  const playerCombat = isPlainObject(player?.combat) ? player.combat : {};
  const combat = isPlainObject(state.combat) ? state.combat : {};
  const run = isPlainObject(state.run) ? state.run : {};
  const map = isPlainObject(state.map) ? state.map : {};
  const shop = isPlainObject(state.shop) ? state.shop : {};
  const nonAutomationActions = getNonAutomationActions(state);
  const crystalSphere = summarizeCrystalSphereForAgent(
    state?.crystal_sphere,
    state?.event_options?.options
  );
  const summarizedAvailableActions = summarizeActionsForAgent(
    isCrystalSphereScreen(screen)
      ? nonAutomationActions.filter((action) => !isCrystalSphereEventOptionAction(action))
      : nonAutomationActions,
    screen
  );

  const summary = {
    screen,
    state_version: Number.isFinite(state.state_version) ? state.state_version : null,
    run: {
      act: typeof run?.act?.title === "string" ? run.act.title : null,
      act_floor: Number.isFinite(run.act_floor) ? run.act_floor : null,
      total_floor: Number.isFinite(run.total_floor) ? run.total_floor : null,
      room_type: typeof run?.current_room?.room_type === "string"
        ? run.current_room.room_type
        : null,
      room_model: typeof run?.current_room?.model_id === "string"
        ? run.current_room.model_id
        : null,
      room_pre_finished: run?.current_room?.is_pre_finished === true,
      current_map_coord: isPlainObject(run.current_map_coord) ? run.current_map_coord : null,
      is_game_over: run.is_game_over === true
    },
    player: {
      current_hp: Number.isFinite(playerCreature.current_hp) ? playerCreature.current_hp : null,
      max_hp: Number.isFinite(playerCreature.max_hp) ? playerCreature.max_hp : null,
      block: Number.isFinite(playerCreature.block) ? playerCreature.block : 0,
      powers: Array.isArray(playerCreature.powers)
        ? playerCreature.powers
          .map(summarizePowerForAgent)
          .filter((power) => power !== null)
        : [],
      gold: Number.isFinite(player?.gold) ? player.gold : null,
      potions: Array.isArray(player?.potions)
        ? player.potions.map((potion) => (potion?.empty === true ? "[empty]" : potion?.title ?? null))
        : [],
      relics: Array.isArray(player?.relics)
        ? player.relics.map((relic) => summarizeRelicForAgent(relic)).filter((relic) => relic !== null)
        : []
    }
  };

  if (summarizedAvailableActions.length > 0) {
    summary.available_actions = summarizedAvailableActions;
  }

  if (combat.in_progress === true || Array.isArray(playerCombat?.hand?.cards)) {
    const playerCombatId = Number.isFinite(playerCreature?.combat_id) ? playerCreature.combat_id : null;
    const playerName = normalizeAgentText(playerCreature?.name);
    const summons = Array.isArray(combat.player_creatures)
      ? combat.player_creatures
        .map(summarizeCreatureForAgent)
        .filter(
          (creature) =>
            creature !== null &&
            (playerCombatId !== null
              ? creature.combat_id !== playerCombatId
              : normalizeAgentText(creature?.name) !== playerName)
        )
      : [];
    summary.combat = {
      in_progress: combat.in_progress === true,
      round_number: Number.isFinite(combat.round_number) ? combat.round_number : null,
      current_side: typeof combat.current_side === "string" ? combat.current_side : null,
      is_play_phase: combat.is_play_phase === true,
      player_actions_disabled: combat.player_actions_disabled === true,
      energy: Number.isFinite(playerCombat.energy) ? playerCombat.energy : null,
      max_energy: Number.isFinite(playerCombat.max_energy) ? playerCombat.max_energy : null,
      stars: Number.isFinite(playerCombat.stars) ? playerCombat.stars : null,
      hand: Array.isArray(playerCombat?.hand?.cards)
        ? playerCombat.hand.cards.map(summarizeCardForAgent).filter((card) => card !== null)
        : [],
      draw_pile_count: Number.isFinite(playerCombat?.draw_pile?.count)
        ? playerCombat.draw_pile.count
        : null,
      draw_top: Array.isArray(playerCombat?.draw_pile?.cards)
        ? playerCombat.draw_pile.cards
          .slice(0, 3)
          .map(summarizeCardForAgent)
          .filter((card) => card !== null)
        : [],
      draw_preview_cards: Array.isArray(playerCombat?.draw_pile?.cards)
        ? playerCombat.draw_pile.cards
          .slice(0, 12)
          .map(summarizeCardForAgent)
          .filter((card) => card !== null)
        : [],
      discard_pile_count: Number.isFinite(playerCombat?.discard_pile?.count)
        ? playerCombat.discard_pile.count
        : null,
      discard_cards: Array.isArray(playerCombat?.discard_pile?.cards)
        ? playerCombat.discard_pile.cards
          .slice(0, 24)
          .map(summarizeCardForAgent)
          .filter((card) => card !== null)
        : [],
      exhaust_pile_count: Number.isFinite(playerCombat?.exhaust_pile?.count)
        ? playerCombat.exhaust_pile.count
        : null,
      exhaust_cards: Array.isArray(playerCombat?.exhaust_pile?.cards)
        ? playerCombat.exhaust_pile.cards
          .slice(0, 24)
          .map(summarizeCardForAgent)
          .filter((card) => card !== null)
        : [],
      play_pile_count: Number.isFinite(playerCombat?.play_pile?.count)
        ? playerCombat.play_pile.count
        : null,
      play_pile_cards: Array.isArray(playerCombat?.play_pile?.cards)
        ? playerCombat.play_pile.cards
          .slice(0, 12)
          .map(summarizeCardForAgent)
          .filter((card) => card !== null)
        : [],
      summons,
      enemies: Array.isArray(combat.enemy_creatures)
        ? combat.enemy_creatures.map(summarizeCreatureForAgent).filter((creature) => creature !== null)
        : [],
      target_index_map: Array.isArray(combat.target_index_map)
        ? combat.target_index_map
          .map((entry) => ({
            action_suffixes: Array.isArray(entry?.action_suffixes)
              ? entry.action_suffixes.filter((value) => typeof value === "string")
              : [],
            combat_id: Number.isFinite(entry?.combat_id) ? entry.combat_id : null,
            name: normalizeAgentText(entry?.name),
            side: typeof entry?.side === "string" ? entry.side : null,
            is_enemy: entry?.is_enemy === true,
            is_alive: entry?.is_alive === true
          }))
          .filter((entry) => entry.combat_id !== null || entry.name !== null)
        : []
    };
  }

  const rewardBundle = buildRewardBundle(state);
  if (rewardBundle.in_reward_flow) {
    summary.rewards = summarizeRewardBundleForAgent(rewardBundle);
  }

  const restSiteBundle = buildRestSiteBundle(state);
  if (restSiteBundle.in_rest_site_flow) {
    summary.rest_site = summarizeRestSiteBundleForAgent(restSiteBundle);
  }

  const cardSelectionBundle = buildCardSelectionBundle(state);
  if (cardSelectionBundle.in_card_selection_flow) {
    summary.card_selection = summarizeCardSelectionBundleForAgent(cardSelectionBundle);
  }

  if (map.is_open === true) {
    summary.map = {
      is_open: true,
      is_travel_enabled: map.is_travel_enabled === true,
      is_traveling: map.is_traveling === true,
      current_coord: isPlainObject(map.current_coord) ? map.current_coord : null,
      travelable_points: Array.isArray(map.points)
        ? map.points
          .filter((point) => point?.is_travelable === true || point?.state === "Travelable")
          .map((point) => ({
            coord: isPlainObject(point?.coord) ? point.coord : null,
            point_type: typeof point?.point_type === "string" ? point.point_type : null
          }))
        : []
    };
  }

  if (shop.is_open === true || shop.visible === true) {
    summary.shop = {
      is_open: shop.is_open === true,
      gold: Number.isFinite(shop.gold) ? shop.gold : null,
      items: Array.isArray(shop.items)
        ? shop.items
          .map((item) => summarizeShopItemForAgent(item))
          .filter((item) => item !== null)
        : []
    };
  }

  if (state?.event_options?.visible === true && !isCrystalSphereScreen(screen)) {
    const visibleGlossary = Array.isArray(state.event_options.visible_glossary)
      ? state.event_options.visible_glossary
        .map((entry) => ({
          title: normalizeAgentText(entry?.title),
          description: normalizeAgentText(entry?.description)
        }))
        .filter((entry) => entry.title || entry.description)
      : [];
    const visibleGlossaryTexts = Array.isArray(
      state.event_options.visible_glossary_texts
    )
      ? state.event_options.visible_glossary_texts
        .map((text) => normalizeAgentText(text))
        .filter((text) => typeof text === "string" && text.length > 0)
      : [];
    summary.event_options = {
      visible: true,
      visible_glossary_source:
        typeof state.event_options.visible_glossary_source === "string"
          ? state.event_options.visible_glossary_source
          : null,
      visible_glossary:
        visibleGlossary.length > 0 ? visibleGlossary : undefined,
      visible_glossary_texts:
        visibleGlossary.length === 0 && visibleGlossaryTexts.length > 0
          ? visibleGlossaryTexts
          : undefined,
      options: Array.isArray(state.event_options.options)
        ? state.event_options.options
          .map((option) => summarizeEventOptionForAgent(option))
          .filter((option) => option !== null)
        : []
    };
  }

  if (crystalSphere !== null) {
    summary.crystal_sphere = crystalSphere;
  }

  return summary;
}

function summarizeActionStateForAgent(state) {
  if (!isPlainObject(state)) {
    return state;
  }

  const summary = summarizeStateForAgent(state);
  if (!isPlainObject(summary)) {
    return summary;
  }

  const result = {
    screen: summary.screen,
    state_version: summary.state_version
  };

  if (isPlainObject(summary.player)) {
    result.player = {
      current_hp: Number.isFinite(summary.player.current_hp) ? summary.player.current_hp : null,
      max_hp: Number.isFinite(summary.player.max_hp) ? summary.player.max_hp : null,
      block: Number.isFinite(summary.player.block) ? summary.player.block : 0,
      powers: Array.isArray(summary.player.powers) ? summary.player.powers : [],
      potions: Array.isArray(summary.player.potions) ? summary.player.potions : []
    };
  }

  if (typeof summary.screen === "string" && summary.screen === "COMBAT") {
    result.available_actions = summarizeCombatSequenceAvailableActions(getNonAutomationActions(state));
  } else if (Array.isArray(summary.available_actions)) {
    result.available_actions = summary.available_actions;
  }

  if (isPlainObject(summary.combat)) {
    result.combat = {
      in_progress: summary.combat.in_progress === true,
      round_number: Number.isFinite(summary.combat.round_number)
        ? summary.combat.round_number
        : null,
      current_side:
        typeof summary.combat.current_side === "string" ? summary.combat.current_side : null,
      is_play_phase: summary.combat.is_play_phase === true,
      player_actions_disabled: summary.combat.player_actions_disabled === true,
      energy: Number.isFinite(summary.combat.energy) ? summary.combat.energy : null,
      max_energy: Number.isFinite(summary.combat.max_energy) ? summary.combat.max_energy : null,
      stars: Number.isFinite(summary.combat.stars) ? summary.combat.stars : null,
      hand: Array.isArray(summary.combat.hand) ? summary.combat.hand : [],
      draw_pile_count: Number.isFinite(summary.combat.draw_pile_count)
        ? summary.combat.draw_pile_count
        : null,
      draw_preview_cards: Array.isArray(summary.combat.draw_preview_cards)
        ? summary.combat.draw_preview_cards
        : [],
      discard_pile_count: Number.isFinite(summary.combat.discard_pile_count)
        ? summary.combat.discard_pile_count
        : null,
      discard_cards: Array.isArray(summary.combat.discard_cards)
        ? summary.combat.discard_cards
        : [],
      exhaust_pile_count: Number.isFinite(summary.combat.exhaust_pile_count)
        ? summary.combat.exhaust_pile_count
        : null,
      exhaust_cards: Array.isArray(summary.combat.exhaust_cards)
        ? summary.combat.exhaust_cards
        : [],
      play_pile_count: Number.isFinite(summary.combat.play_pile_count)
        ? summary.combat.play_pile_count
        : null,
      play_pile_cards: Array.isArray(summary.combat.play_pile_cards)
        ? summary.combat.play_pile_cards
        : [],
      summons: Array.isArray(summary.combat.summons) ? summary.combat.summons : [],
      enemies: Array.isArray(summary.combat.enemies) ? summary.combat.enemies : []
    };
  }

  if (isPlainObject(summary.run) && summary.screen !== "COMBAT") {
    result.run = summary.run;
  }

  if (isPlainObject(summary.rewards)) {
    result.rewards = summary.rewards;
  }

  if (isPlainObject(summary.rest_site)) {
    result.rest_site = summary.rest_site;
  }

  if (isPlainObject(summary.card_selection)) {
    result.card_selection = summary.card_selection;
  }

  if (isPlainObject(summary.map)) {
    result.map = summary.map;
  }

  if (isPlainObject(summary.shop)) {
    result.shop = summary.shop;
  }

  if (isPlainObject(summary.event_options)) {
    result.event_options = summary.event_options;
  }

  if (isPlainObject(summary.crystal_sphere)) {
    result.crystal_sphere = summary.crystal_sphere;
  }

  return result;
}

function dedupeCompactedPayloadSections(payload) {
  if (!isPlainObject(payload)) {
    return payload;
  }

  if (isPlainObject(payload.state)) {
    payload.state = removeDuplicatedStateSections(payload.state, {
      reward: isPlainObject(payload.reward_bundle),
      rest_site: isPlainObject(payload.rest_site_bundle),
      card_selection:
        isPlainObject(payload.card_selection_bundle) ||
        isPlainObject(payload.current_card_selection_bundle) ||
        isPlainObject(payload.initial_card_selection_bundle),
      shop: isPlainObject(payload.shop_bundle),
      available_actions:
        isPlainObject(payload.reward_bundle) ||
        isPlainObject(payload.rest_site_bundle) ||
        isPlainObject(payload.card_selection_bundle) ||
        isPlainObject(payload.current_card_selection_bundle) ||
        isPlainObject(payload.initial_card_selection_bundle) ||
        isPlainObject(payload.shop_bundle)
    });
  }

  if (isPlainObject(payload.state_after)) {
    payload.state_after = removeDuplicatedStateSections(payload.state_after, {
      reward: isPlainObject(payload.reward_bundle),
      rest_site: isPlainObject(payload.rest_site_bundle),
      card_selection:
        isPlainObject(payload.card_selection_bundle) ||
        isPlainObject(payload.current_card_selection_bundle),
      shop: isPlainObject(payload.shop_bundle),
      available_actions:
        isPlainObject(payload.reward_bundle) ||
        isPlainObject(payload.rest_site_bundle) ||
        isPlainObject(payload.card_selection_bundle) ||
        isPlainObject(payload.current_card_selection_bundle) ||
        isPlainObject(payload.shop_bundle)
    });
  }

  if (isPlainObject(payload.final_state)) {
    payload.final_state = removeDuplicatedStateSections(payload.final_state, {
      reward: isPlainObject(payload.final_reward_bundle),
      rest_site: false,
      card_selection: isPlainObject(payload.final_card_selection_bundle),
      shop: false,
      available_actions:
        isPlainObject(payload.final_reward_bundle) ||
        isPlainObject(payload.final_card_selection_bundle)
    });
  }

  return payload;
}

function removeDuplicatedStateSections(state, sections = {}) {
  if (!isPlainObject(state)) {
    return state;
  }

  const deduped = {
    ...state
  };

  if (sections.reward) {
    delete deduped.rewards;
  }

  if (sections.rest_site) {
    delete deduped.rest_site;
  }

  if (sections.card_selection) {
    delete deduped.card_selection;
  }

  if (sections.shop) {
    delete deduped.shop;
  }

  if (sections.available_actions) {
    delete deduped.available_actions;
  }

  return deduped;
}

function buildDeckPayloadForAgent(state) {
  if (!isPlainObject(state)) {
    return {
      ok: false,
      reason: "missing_state"
    };
  }

  const player = state?.players?.[0];
  if (!isPlainObject(player)) {
    return {
      ok: false,
      reason: "player_missing",
      screen: typeof state.screen === "string" ? state.screen : null,
      state_version: Number.isFinite(state.state_version) ? state.state_version : null
    };
  }

  const deck = isPlainObject(player.deck) ? player.deck : {};
  const rawCards = Array.isArray(deck.cards) ? deck.cards : [];
  const cards = rawCards
    .map((card, index) => summarizeDeckCardForAgent(card, index))
    .filter((card) => card !== null);
  const groups = summarizeDeckGroupsForAgent(cards);

  return {
    ok: true,
    screen: typeof state.screen === "string" ? state.screen : null,
    state_version: Number.isFinite(state.state_version) ? state.state_version : null,
    player: {
      character: normalizeAgentText(player?.character?.title) ?? null,
      current_hp: Number.isFinite(player?.creature?.current_hp) ? player.creature.current_hp : null,
      max_hp: Number.isFinite(player?.creature?.max_hp) ? player.creature.max_hp : null,
      gold: Number.isFinite(player?.gold) ? player.gold : null
    },
    deck: {
      count: Number.isFinite(deck.count) ? deck.count : cards.length,
      cards,
      groups
    }
  };
}

function summarizeDeckCardForAgent(card, index) {
  const summary = summarizeCardForAgent(card);
  if (!isPlainObject(summary)) {
    return null;
  }

  const fullDescription =
    typeof card?.description === "string" && card.description.trim()
      ? normalizeAgentText(card.description)
      : null;

  const result = {
    index
  };

  if (typeof card?.id === "string" && card.id.trim()) {
    result.id = card.id;
  }

  if (typeof summary.title === "string" && summary.title.trim()) {
    result.title = summary.title;
  }

  if (typeof summary.type === "string" && summary.type.trim()) {
    result.type = summary.type;
  }

  if (typeof card?.rarity === "string" && card.rarity.trim()) {
    result.rarity = card.rarity;
  }

  if (Number.isInteger(summary.cost)) {
    result.cost = summary.cost;
  }

  if (typeof summary.target === "string" && summary.target.trim()) {
    result.target = summary.target;
  }

  if (typeof summary.effect === "string" && summary.effect.trim()) {
    result.effect = summary.effect;
  }

  if (typeof summary.description === "string" && summary.description.trim()) {
    result.description = summary.description;
  } else if (typeof fullDescription === "string" && fullDescription.trim()) {
    result.description = fullDescription;
  }

  return result;
}

function summarizeDeckGroupsForAgent(cards) {
  if (!Array.isArray(cards) || cards.length <= 0) {
    return [];
  }

  const groupMap = new Map();

  for (const card of cards) {
    if (!isPlainObject(card)) {
      continue;
    }

    const key = [
      typeof card.title === "string" ? card.title : "",
      typeof card.type === "string" ? card.type : "",
      Number.isInteger(card.cost) ? String(card.cost) : "",
      typeof card.effect === "string" ? card.effect : ""
    ].join("|");

    const existing = groupMap.get(key);
    if (existing) {
      existing.count += 1;
      continue;
    }

    groupMap.set(key, {
      title: typeof card.title === "string" ? card.title : null,
      type: typeof card.type === "string" ? card.type : null,
      cost: Number.isInteger(card.cost) ? card.cost : null,
      rarity: typeof card.rarity === "string" ? card.rarity : null,
      effect: typeof card.effect === "string" ? card.effect : null,
      description: typeof card.description === "string" ? card.description : null,
      count: 1
    });
  }

  return Array.from(groupMap.values()).sort(compareDeckGroupSummaries);
}

function compareDeckGroupSummaries(left, right) {
  const leftTitle = typeof left?.title === "string" ? left.title : "";
  const rightTitle = typeof right?.title === "string" ? right.title : "";
  const titleCompare = leftTitle.localeCompare(rightTitle, "zh-Hans-CN");
  if (titleCompare !== 0) {
    return titleCompare;
  }

  const leftCost = Number.isInteger(left?.cost) ? left.cost : Number.POSITIVE_INFINITY;
  const rightCost = Number.isInteger(right?.cost) ? right.cost : Number.POSITIVE_INFINITY;
  if (leftCost !== rightCost) {
    return leftCost - rightCost;
  }

  const leftType = typeof left?.type === "string" ? left.type : "";
  const rightType = typeof right?.type === "string" ? right.type : "";
  return leftType.localeCompare(rightType, "en");
}

function buildMapRoutesPayload(state, options = {}) {
  if (!isPlainObject(state)) {
    return {
      ok: false,
      error: "invalid_state_payload"
    };
  }

  const detail = normalizeMapRoutesDetail(options.detail);
  const snapshotStatus = isPlainObject(options.snapshot_status)
    ? options.snapshot_status
    : null;
  const map = isPlainObject(state.map) ? state.map : {};
  if (map.is_open !== true || !Array.isArray(map.points) || map.points.length === 0) {
    return {
      ok: false,
      error: "map_not_open",
      screen: typeof state.screen === "string" ? state.screen : null,
      map_open: map.is_open === true,
      detail,
      snapshot_settled: snapshotStatus?.settled ?? null,
      snapshot_settle_reason: snapshotStatus?.reason ?? null,
      snapshot_settle_polls: snapshotStatus?.poll_count ?? null
    };
  }

  const currentCoord = extractMapCurrentCoord(state);
  const currentRow = Number.isFinite(currentCoord?.row) ? currentCoord.row : null;
  const runContext = buildMapRunContext(state);

  const allPoints = map.points
    .filter((point) => isPlainObject(point) && isPlainObject(point.coord))
    .map((point) => ({
      coord: {
        col: Number.isFinite(point.coord.col) ? point.coord.col : null,
        row: Number.isFinite(point.coord.row) ? point.coord.row : null
      },
      point_type: typeof point.point_type === "string" ? point.point_type : null,
      state: typeof point.state === "string" ? point.state : null,
      is_travelable: point.is_travelable === true || point.state === "Travelable",
      children: Array.isArray(point.children)
        ? point.children
          .filter((child) => isPlainObject(child))
          .map((child) => ({
            col: Number.isFinite(child.col) ? child.col : null,
            row: Number.isFinite(child.row) ? child.row : null
          }))
          .filter((child) => child.col !== null && child.row !== null)
        : []
    }))
    .filter((point) => point.coord.col !== null && point.coord.row !== null);

  const points =
    currentRow === null
      ? allPoints
      : allPoints.filter((point) => point.coord.row > currentRow);

  const pointByKey = new Map(
    points.map((point) => [toCoordKey(point.coord), point])
  );
  const frontier = points.filter((point) => point.is_travelable === true);

  const reachableKeys = collectReachableMapKeys(frontier, pointByKey);
  const depthMemo = new Map();

  const routeRoots = frontier
    .map((point) => {
      const key = toCoordKey(point.coord);
      if (!key || !reachableKeys.has(key)) {
        return null;
      }

      const rootSummary = buildMapRouteRootSummary(key, pointByKey);

      return {
        key,
        point_type: point.point_type,
        child_keys: getReachableMapChildKeys(point, pointByKey, reachableKeys),
        reachable_node_count: countUniqueReachableMapNodesFromKey(key, pointByKey),
        max_depth: getReachableMapDepthFromKey(key, pointByKey, depthMemo),
        summary: rootSummary,
        run_aware: buildMapRouteRootRunAware(rootSummary, runContext)
      };
    })
    .filter((entry) => entry !== null)
    .sort(compareReachableMapNodeEntries);

  const routeNodes = Array.from(reachableKeys)
    .map((key) => {
      const point = pointByKey.get(key);
      if (!point) {
        return null;
      }

      return {
        key,
        point_type: point.point_type,
        child_keys: getReachableMapChildKeys(point, pointByKey, reachableKeys)
      };
    })
    .filter((entry) => entry !== null)
    .sort(compareReachableMapNodeEntries);

  return {
    ok: true,
    screen: typeof state.screen === "string" ? state.screen : null,
    current_coord: currentCoord,
    current_row: currentRow,
    run_context: runContext,
    detail,
    forced_route: routeRoots.length === 1,
    frontier_count: frontier.length,
    coord_key_format: "col,row",
    pruned_rules: [
      "exclude_current_node",
      "exclude_rows_at_or_before_current",
      "exclude_unreachable_nodes"
    ],
    snapshot_settled: snapshotStatus?.settled ?? null,
    snapshot_settle_reason: snapshotStatus?.reason ?? null,
    snapshot_settle_polls: snapshotStatus?.poll_count ?? null,
    frontier_action_match: snapshotStatus?.frontier_action_match ?? null,
    route_root_count: routeRoots.length,
    route_node_count: routeNodes.length,
    route_roots: routeRoots,
    route_nodes: detail === "full" ? routeNodes : undefined
  };
}

function buildMapRouteRootSummary(startKey, pointByKey) {
  const pointTypeCounts = new Map();
  const minStepsByType = new Map();
  const visited = new Set();
  const bestStepsByKey = new Map();
  const pending = [
    {
      key: startKey,
      stepsFromCurrent: 1
    }
  ];

  while (pending.length > 0) {
    const current = pending.pop();
    const key = current?.key;
    const stepsFromCurrent = current?.stepsFromCurrent;
    if (typeof key !== "string" || !Number.isInteger(stepsFromCurrent)) {
      continue;
    }

    const bestKnownSteps = bestStepsByKey.get(key);
    if (bestKnownSteps !== undefined && bestKnownSteps <= stepsFromCurrent) {
      continue;
    }
    bestStepsByKey.set(key, stepsFromCurrent);

    const point = pointByKey.get(key);
    if (!point) {
      continue;
    }

    if (!visited.has(key)) {
      visited.add(key);
      if (typeof point.point_type === "string" && point.point_type.length > 0) {
        pointTypeCounts.set(
          point.point_type,
          (pointTypeCounts.get(point.point_type) ?? 0) + 1
        );
      }
    }

    if (typeof point.point_type === "string" && point.point_type.length > 0) {
      const previousMin = minStepsByType.get(point.point_type);
      if (previousMin === undefined || stepsFromCurrent < previousMin) {
        minStepsByType.set(point.point_type, stepsFromCurrent);
      }
    }

    for (const childKey of getReachableMapChildKeys(point, pointByKey)) {
      pending.push({
        key: childKey,
        stepsFromCurrent: stepsFromCurrent + 1
      });
    }
  }

  return {
    forced_path_steps_before_branch: countForcedMapPathSteps(startKey, pointByKey),
    reachable_point_type_counts: objectFromSortedMap(pointTypeCounts),
    steps_from_current_to_next_point_type: objectFromSortedMap(minStepsByType),
    can_reach_rest_site_before_elite: canReachTypeBeforeType(
      startKey,
      pointByKey,
      "RestSite",
      "Elite"
    ),
    can_reach_elite_then_rest_site: canReachEliteThenRestSite(startKey, pointByKey)
  };
}

function buildMapRunContext(state) {
  const run = isPlainObject(state?.run) ? state.run : {};
  const player = state?.players?.[0];
  const creature = isPlainObject(player?.creature) ? player.creature : {};
  const potions = Array.isArray(player?.potions) ? player.potions : [];
  const relics = Array.isArray(player?.relics) ? player.relics : [];
  const deck = isPlainObject(player?.deck) ? player.deck : {};
  const currentHp = Number.isFinite(creature.current_hp) ? creature.current_hp : null;
  const maxHp = Number.isFinite(creature.max_hp) ? creature.max_hp : null;
  const hpRatio =
    currentHp !== null && maxHp !== null && maxHp > 0
      ? Number((currentHp / maxHp).toFixed(3))
      : null;
  const emptyPotionSlots = potions.filter((potion) => potion?.empty === true).length;
  const filledPotionSlots = potions.filter((potion) => potion?.empty !== true).length;
  const nonStarterRelicCount = relics.filter(
    (relic) => String(relic?.rarity ?? "") !== "Starter"
  ).length;
  const deckCardCount = Number.isFinite(deck.count)
    ? deck.count
    : Array.isArray(deck.cards)
      ? deck.cards.length
      : null;

  return {
    act_floor: Number.isFinite(run.act_floor) ? run.act_floor : null,
    total_floor: Number.isFinite(run.total_floor) ? run.total_floor : null,
    hp: {
      current: currentHp,
      max: maxHp,
      ratio: hpRatio,
      band: getHpBand(hpRatio)
    },
    gold: Number.isFinite(player?.gold) ? player.gold : null,
    gold_band: getGoldBand(player?.gold),
    potion_slots: {
      total: potions.length,
      filled: filledPotionSlots,
      empty: emptyPotionSlots,
      band: getPotionSlotBand(emptyPotionSlots, potions.length)
    },
    deck_card_count: deckCardCount,
    relic_count: relics.length,
    non_starter_relic_count: nonStarterRelicCount
  };
}

function buildMapRouteRootRunAware(routeSummary, runContext) {
  const stepIndex = isPlainObject(routeSummary?.steps_from_current_to_next_point_type)
    ? routeSummary.steps_from_current_to_next_point_type
    : {};
  const nextEliteSteps = Number.isFinite(stepIndex.Elite) ? stepIndex.Elite : null;
  const nextRestSteps = Number.isFinite(stepIndex.RestSite) ? stepIndex.RestSite : null;
  const nextShopSteps = Number.isFinite(stepIndex.Shop) ? stepIndex.Shop : null;
  const eliteViability = rateEliteViability(routeSummary, runContext);
  const reasonTags = buildMapRouteReasonTags(routeSummary, runContext, eliteViability);

  return {
    rest_pressure: rateRestPressure(nextRestSteps, runContext),
    shop_access_value: rateShopAccessValue(nextShopSteps, runContext),
    potion_capacity_value: ratePotionCapacityValue(runContext),
    elite_plan: {
      next_elite_steps: nextEliteSteps,
      can_reach_rest_site_before_elite:
        routeSummary?.can_reach_rest_site_before_elite === true,
      can_reach_elite_then_rest_site:
        routeSummary?.can_reach_elite_then_rest_site === true,
      viability: eliteViability.rating,
      score: eliteViability.score
    },
    reason_tags: reasonTags
  };
}

function countForcedMapPathSteps(startKey, pointByKey) {
  let key = startKey;
  let steps = 0;

  while (typeof key === "string") {
    const point = pointByKey.get(key);
    if (!point) {
      break;
    }

    steps += 1;
    const childKeys = getReachableMapChildKeys(point, pointByKey);
    if (childKeys.length !== 1) {
      break;
    }

    key = childKeys[0];
  }

  return steps;
}

function canReachEliteThenRestSite(startKey, pointByKey) {
  const memo = new Map();
  return canReachEliteThenRestSiteRecursive(startKey, pointByKey, false, memo);
}

function canReachEliteThenRestSiteRecursive(startKey, pointByKey, seenElite, memo) {
  if (typeof startKey !== "string") {
    return false;
  }

  const memoKey = `${startKey}|${seenElite ? "1" : "0"}`;
  if (memo.has(memoKey)) {
    return memo.get(memoKey);
  }

  const point = pointByKey.get(startKey);
  if (!point) {
    memo.set(memoKey, false);
    return false;
  }

  const nextSeenElite = seenElite || point.point_type === "Elite";
  if (nextSeenElite && point.point_type === "RestSite") {
    memo.set(memoKey, true);
    return true;
  }

  const result = getReachableMapChildKeys(point, pointByKey).some((childKey) =>
    canReachEliteThenRestSiteRecursive(childKey, pointByKey, nextSeenElite, memo)
  );
  memo.set(memoKey, result);
  return result;
}

function canReachTypeBeforeType(startKey, pointByKey, desiredType, blockingType) {
  const memo = new Map();
  return canReachTypeBeforeTypeRecursive(
    startKey,
    pointByKey,
    desiredType,
    blockingType,
    memo
  );
}

function canReachTypeBeforeTypeRecursive(
  startKey,
  pointByKey,
  desiredType,
  blockingType,
  memo
) {
  if (typeof startKey !== "string") {
    return false;
  }

  const memoKey = `${startKey}|${desiredType}|${blockingType}`;
  if (memo.has(memoKey)) {
    return memo.get(memoKey);
  }

  const point = pointByKey.get(startKey);
  if (!point) {
    memo.set(memoKey, false);
    return false;
  }

  if (point.point_type === desiredType) {
    memo.set(memoKey, true);
    return true;
  }

  if (point.point_type === blockingType) {
    memo.set(memoKey, false);
    return false;
  }

  const result = getReachableMapChildKeys(point, pointByKey).some((childKey) =>
    canReachTypeBeforeTypeRecursive(
      childKey,
      pointByKey,
      desiredType,
      blockingType,
      memo
    )
  );
  memo.set(memoKey, result);
  return result;
}

function getHpBand(hpRatio) {
  if (!Number.isFinite(hpRatio)) {
    return null;
  }

  if (hpRatio >= 0.75) {
    return "healthy";
  }

  if (hpRatio >= 0.5) {
    return "stable";
  }

  if (hpRatio >= 0.3) {
    return "wounded";
  }

  return "critical";
}

function getGoldBand(gold) {
  if (!Number.isFinite(gold)) {
    return null;
  }

  if (gold >= 150) {
    return "rich";
  }

  if (gold >= 75) {
    return "shop_ready";
  }

  if (gold >= 40) {
    return "limited";
  }

  return "low";
}

function getPotionSlotBand(emptySlots, totalSlots) {
  if (!Number.isFinite(totalSlots) || totalSlots <= 0) {
    return null;
  }

  if (!Number.isFinite(emptySlots) || emptySlots <= 0) {
    return "full";
  }

  if (emptySlots >= 2) {
    return "open";
  }

  return "tight";
}

function rateRestPressure(nextRestSteps, runContext) {
  const hpRatio = Number.isFinite(runContext?.hp?.ratio) ? runContext.hp.ratio : null;
  if (!Number.isFinite(hpRatio)) {
    return "unknown";
  }

  if (hpRatio < 0.35) {
    return "high";
  }

  if (hpRatio < 0.55 && (!Number.isFinite(nextRestSteps) || nextRestSteps > 3)) {
    return "high";
  }

  if (hpRatio < 0.7 && (!Number.isFinite(nextRestSteps) || nextRestSteps > 4)) {
    return "medium";
  }

  return "low";
}

function rateShopAccessValue(nextShopSteps, runContext) {
  const gold = Number.isFinite(runContext?.gold) ? runContext.gold : null;
  if (!Number.isFinite(nextShopSteps)) {
    return "none";
  }

  if (!Number.isFinite(gold) || gold < 40) {
    return "low";
  }

  if ((gold >= 75 && nextShopSteps <= 2) || (gold >= 150 && nextShopSteps <= 4)) {
    return "high";
  }

  if ((gold >= 50 && nextShopSteps <= 3) || gold >= 100) {
    return "medium";
  }

  return "low";
}

function ratePotionCapacityValue(runContext) {
  const emptySlots = Number.isFinite(runContext?.potion_slots?.empty)
    ? runContext.potion_slots.empty
    : null;
  const totalSlots = Number.isFinite(runContext?.potion_slots?.total)
    ? runContext.potion_slots.total
    : null;
  if (!Number.isFinite(totalSlots) || totalSlots <= 0) {
    return "unknown";
  }

  if (!Number.isFinite(emptySlots) || emptySlots <= 0) {
    return "none";
  }

  if (emptySlots >= 2) {
    return "high";
  }

  return "medium";
}

function rateEliteViability(routeSummary, runContext) {
  const steps = isPlainObject(routeSummary?.steps_from_current_to_next_point_type)
    ? routeSummary.steps_from_current_to_next_point_type
    : {};
  const nextEliteSteps = Number.isFinite(steps.Elite) ? steps.Elite : null;
  if (!Number.isFinite(nextEliteSteps)) {
    return {
      rating: "none",
      score: null
    };
  }

  let score = 0;
  const hpBand = runContext?.hp?.band;
  if (hpBand === "healthy") {
    score += 2;
  } else if (hpBand === "stable") {
    score += 1;
  } else if (hpBand === "wounded") {
    score -= 1;
  } else if (hpBand === "critical") {
    score -= 2;
  }

  const filledPotionSlots = Number.isFinite(runContext?.potion_slots?.filled)
    ? runContext.potion_slots.filled
    : 0;
  if (filledPotionSlots >= 1) {
    score += 1;
  }

  const nonStarterRelicCount = Number.isFinite(runContext?.non_starter_relic_count)
    ? runContext.non_starter_relic_count
    : 0;
  if (nonStarterRelicCount >= 3) {
    score += 1;
  }

  if (nextEliteSteps >= 5) {
    score += 1;
  } else if (nextEliteSteps <= 2) {
    score -= 1;
  }

  if (routeSummary?.can_reach_rest_site_before_elite === true) {
    score += 1;
  }

  if (routeSummary?.can_reach_elite_then_rest_site === true) {
    score += 1;
  }

  if (score >= 4) {
    return { rating: "strong", score };
  }

  if (score >= 2) {
    return { rating: "okay", score };
  }

  if (score >= 0) {
    return { rating: "risky", score };
  }

  return { rating: "poor", score };
}

function buildMapRouteReasonTags(routeSummary, runContext, eliteViability) {
  const steps = isPlainObject(routeSummary?.steps_from_current_to_next_point_type)
    ? routeSummary.steps_from_current_to_next_point_type
    : {};
  const nextShopSteps = Number.isFinite(steps.Shop) ? steps.Shop : null;
  const tags = [];

  if (typeof runContext?.hp?.band === "string") {
    tags.push(`hp_${runContext.hp.band}`);
  }

  if (typeof runContext?.gold_band === "string") {
    tags.push(`gold_${runContext.gold_band}`);
  }

  if (typeof runContext?.potion_slots?.band === "string") {
    tags.push(`potions_${runContext.potion_slots.band}`);
  }

  if (typeof eliteViability?.rating === "string" && eliteViability.rating !== "none") {
    tags.push(`elite_${eliteViability.rating}`);
  }

  if (routeSummary?.can_reach_rest_site_before_elite === true) {
    tags.push("elite_rest_before");
  }

  if (routeSummary?.can_reach_elite_then_rest_site === true) {
    tags.push("elite_rest_after");
  }

  if (Number.isFinite(nextShopSteps)) {
    tags.push(`shop_${Math.min(nextShopSteps, 6)}_steps`);
  }

  return tags.slice(0, 6);
}

function toCoordKey(coord) {
  if (!isPlainObject(coord)) {
    return null;
  }

  const col = Number.isFinite(coord.col) ? coord.col : null;
  const row = Number.isFinite(coord.row) ? coord.row : null;
  if (col === null || row === null) {
    return null;
  }

  return `${col},${row}`;
}

function extractMapCurrentCoord(state) {
  if (!isPlainObject(state)) {
    return null;
  }

  const run = isPlainObject(state.run) ? state.run : {};
  const map = isPlainObject(state.map) ? state.map : {};
  return isPlainObject(run.current_map_coord)
    ? run.current_map_coord
    : isPlainObject(map.current_coord)
      ? map.current_coord
      : null;
}

function extractTravelableMapKeysFromState(state) {
  if (!isPlainObject(state?.map) || !Array.isArray(state.map.points)) {
    return [];
  }

  return state.map.points
    .filter(
      (point) =>
        isPlainObject(point) &&
        isPlainObject(point.coord) &&
        (point.is_travelable === true || point.state === "Travelable")
    )
    .map((point) => toCoordKey(point.coord))
    .filter((key) => typeof key === "string")
    .sort(compareCoordKeys);
}

function extractMapActionKeys(state) {
  if (!isPlainObject(state) || !Array.isArray(state.available_actions)) {
    return [];
  }

  const keys = state.available_actions
    .filter(
      (action) =>
        isPlainObject(action) &&
        typeof action.action_id === "string" &&
        action.action_id.startsWith("map:") &&
        !action.action_id.startsWith("automation:")
    )
    .map((action) => {
      if (isPlainObject(action.coord)) {
        return toCoordKey(action.coord);
      }

      return action.action_id.slice("map:".length);
    })
    .filter((key) => typeof key === "string" && parseCoordKey(key) !== null);

  return [...new Set(keys)].sort(compareCoordKeys);
}

function areCoordKeySetsEqual(leftKeys, rightKeys) {
  const left = Array.isArray(leftKeys) ? leftKeys : [];
  const right = Array.isArray(rightKeys) ? rightKeys : [];
  if (left.length !== right.length) {
    return false;
  }

  for (let index = 0; index < left.length; index += 1) {
    if (left[index] !== right[index]) {
      return false;
    }
  }

  return true;
}

function collectReachableMapKeys(frontier, pointByKey) {
  const reachableKeys = new Set();
  const pending = frontier
    .map((point) => toCoordKey(point?.coord))
    .filter((key) => typeof key === "string");

  while (pending.length > 0) {
    const key = pending.pop();
    if (!key || reachableKeys.has(key)) {
      continue;
    }

    const point = pointByKey.get(key);
    if (!point) {
      continue;
    }

    reachableKeys.add(key);
    for (const childKey of getReachableMapChildKeys(point, pointByKey)) {
      if (!reachableKeys.has(childKey)) {
        pending.push(childKey);
      }
    }
  }

  return reachableKeys;
}

function getReachableMapChildKeys(point, pointByKey, allowedKeys) {
  const seen = new Set();
  const childKeys = [];

  for (const child of Array.isArray(point?.children) ? point.children : []) {
    const childKey = toCoordKey(child);
    if (!childKey || seen.has(childKey)) {
      continue;
    }

    if (!pointByKey.has(childKey)) {
      continue;
    }

    if (allowedKeys instanceof Set && !allowedKeys.has(childKey)) {
      continue;
    }

    seen.add(childKey);
    childKeys.push(childKey);
  }

  return childKeys.sort(compareCoordKeys);
}

function countUniqueReachableMapNodesFromKey(startKey, pointByKey) {
  if (typeof startKey !== "string") {
    return 0;
  }

  const visited = new Set();
  const pending = [startKey];

  while (pending.length > 0) {
    const key = pending.pop();
    if (!key || visited.has(key)) {
      continue;
    }

    const point = pointByKey.get(key);
    if (!point) {
      continue;
    }

    visited.add(key);
    for (const childKey of getReachableMapChildKeys(point, pointByKey)) {
      if (!visited.has(childKey)) {
        pending.push(childKey);
      }
    }
  }

  return visited.size;
}

function getReachableMapDepthFromKey(startKey, pointByKey, memo) {
  if (typeof startKey !== "string") {
    return 0;
  }

  if (memo instanceof Map && memo.has(startKey)) {
    return memo.get(startKey);
  }

  const point = pointByKey.get(startKey);
  if (!point) {
    return 0;
  }

  const childKeys = getReachableMapChildKeys(point, pointByKey);
  const depth =
    childKeys.length === 0
      ? 1
      : 1 + Math.max(...childKeys.map((childKey) => getReachableMapDepthFromKey(childKey, pointByKey, memo)));

  if (memo instanceof Map) {
    memo.set(startKey, depth);
  }

  return depth;
}

function compareCoordKeys(left, right) {
  const leftCoord = parseCoordKey(left);
  const rightCoord = parseCoordKey(right);
  return compareCoords(leftCoord, rightCoord);
}

function compareReachableMapNodeEntries(left, right) {
  return compareCoordKeys(left?.key, right?.key);
}

function parseCoordKey(value) {
  if (typeof value !== "string") {
    return null;
  }

  const [colText, rowText] = value.split(",");
  const col = Number.parseInt(colText, 10);
  const row = Number.parseInt(rowText, 10);
  if (!Number.isFinite(col) || !Number.isFinite(row)) {
    return null;
  }

  return { col, row };
}

function compareCoords(left, right) {
  const leftRow = Number.isFinite(left?.row) ? left.row : Number.POSITIVE_INFINITY;
  const rightRow = Number.isFinite(right?.row) ? right.row : Number.POSITIVE_INFINITY;
  if (leftRow !== rightRow) {
    return leftRow - rightRow;
  }

  const leftCol = Number.isFinite(left?.col) ? left.col : Number.POSITIVE_INFINITY;
  const rightCol = Number.isFinite(right?.col) ? right.col : Number.POSITIVE_INFINITY;
  return leftCol - rightCol;
}

function objectFromSortedMap(map) {
  if (!(map instanceof Map) || map.size === 0) {
    return {};
  }

  return Object.fromEntries(
    Array.from(map.entries()).sort(([leftKey], [rightKey]) =>
      String(leftKey).localeCompare(String(rightKey))
    )
  );
}

function isBridgeStatePayload(payload) {
  return (
    isPlainObject(payload) &&
    typeof payload.screen === "string" &&
    (Array.isArray(payload.available_actions) ||
      isPlainObject(payload.run) ||
      isPlainObject(payload.combat))
  );
}

function isBridgeActionsPayload(payload) {
  return (
    isPlainObject(payload) &&
    Array.isArray(payload.actions) &&
    !isPlainObject(payload.state)
  );
}

function compactPayloadForOutput(payload) {
  if (!isPlainObject(payload)) {
    return payload;
  }

  if (isBridgeStatePayload(payload)) {
    return summarizeStateForAgent(payload);
  }

  if (isBridgeActionsPayload(payload)) {
    return {
      ok: payload.ok === true,
      schema_version: typeof payload.schema_version === "string" ? payload.schema_version : null,
      state_version: Number.isFinite(payload.state_version) ? payload.state_version : null,
      screen: typeof payload.screen === "string" ? payload.screen : null,
      actions: summarizeActionsForAgent(
        filterNonAutomationActions(payload.actions),
        typeof payload.screen === "string" ? payload.screen : null
      )
    };
  }

  const result = {
    ...payload
  };
  const stateAndStateAfterAreAliased =
    isPlainObject(payload.state) &&
    isPlainObject(payload.state_after) &&
    payload.state === payload.state_after;

  if (isPlainObject(payload.state)) {
    result.state = summarizeStateForAgent(payload.state);
  }

  if (isPlainObject(payload.state_after)) {
    result.state_after = summarizeStateForAgent(payload.state_after);
  }

  if (isPlainObject(payload.final_state)) {
    result.final_state = summarizeStateForAgent(payload.final_state);
  }

  if (isPlainObject(payload.reward_bundle)) {
    result.reward_bundle = summarizeRewardBundleForAgent(payload.reward_bundle);
  }

  if (isPlainObject(payload.final_reward_bundle)) {
    result.final_reward_bundle = summarizeRewardBundleForAgent(payload.final_reward_bundle);
  }

  if (isPlainObject(payload.rest_site_bundle)) {
    result.rest_site_bundle = summarizeRestSiteBundleForAgent(payload.rest_site_bundle);
  }

  if (isPlainObject(payload.card_selection_bundle)) {
    result.card_selection_bundle = summarizeCardSelectionBundleForAgent(payload.card_selection_bundle);
  }

  if (isPlainObject(payload.initial_card_selection_bundle)) {
    result.initial_card_selection_bundle = summarizeCardSelectionBundleForAgent(payload.initial_card_selection_bundle);
  }

  if (isPlainObject(payload.current_card_selection_bundle)) {
    result.current_card_selection_bundle = summarizeCardSelectionBundleForAgent(payload.current_card_selection_bundle);
  }

  if (isPlainObject(payload.final_card_selection_bundle)) {
    result.final_card_selection_bundle = summarizeCardSelectionBundleForAgent(payload.final_card_selection_bundle);
  }

  if (isPlainObject(payload.shop_bundle)) {
    result.shop_bundle = summarizeShopBundleForAgent(payload.shop_bundle);
  }

  if (Array.isArray(payload.executed_actions)) {
    result.executed_actions = payload.executed_actions.map(summarizeExecutedAction);
  }

  if (Array.isArray(payload.actions)) {
    result.actions = summarizeActionsForAgent(
      payload.actions,
      typeof payload.screen === "string"
        ? payload.screen
        : typeof payload?.state_after?.screen === "string"
          ? payload.state_after.screen
          : typeof payload?.state?.screen === "string"
            ? payload.state.screen
            : null
    );
  }

  if (isPlainObject(payload.matched_action)) {
    result.matched_action = summarizeActionForAgent(payload.matched_action);
  }

  if (stateAndStateAfterAreAliased) {
    delete result.state;
  }

  dedupeCompactedPayloadSections(result);

  return result;
}

function compactActionResultPayload(payload) {
  const rawState = isPlainObject(payload?.state) ? payload.state : null;
  const rawStateAfter = isPlainObject(payload?.state_after) ? payload.state_after : null;
  const compacted = compactPayloadForOutput(payload);
  if (!isPlainObject(compacted)) {
    return compacted;
  }

  const result = {
    ...compacted
  };

  delete result.interaction_hints;
  delete result.wait_after_ms;
  delete result.state_hash_before;
  delete result.state_hash_after;
  delete result.state_changed;
  delete result.actions;

  if (Array.isArray(result.auto_executed_actions) && result.auto_executed_actions.length <= 0) {
    delete result.auto_executed_actions;
  }

  if (result.post_action_settled === true) {
    delete result.post_action_settled;
    delete result.post_action_settle_reason;
    delete result.post_action_settle_polls;
  }

  if (isPlainObject(rawStateAfter)) {
    result.state_after = summarizeActionStateForAgent(rawStateAfter);
    delete result.state;
  } else if (isPlainObject(rawState)) {
    result.state = summarizeActionStateForAgent(rawState);
  }

  if (isPlainObject(result.state_after)) {
    delete result.state;
  }

  return result;
}

function compactRoomRewardsPayload(payload) {
  const rawState = isPlainObject(payload?.state) ? payload.state : null;
  const rawFinalState = isPlainObject(payload?.final_state) ? payload.final_state : null;
  const compacted = compactPayloadForOutput(payload);
  if (!isPlainObject(compacted)) {
    return compacted;
  }

  const result = {
    ...compacted
  };

  if (Array.isArray(result.claimed_rewards)) {
    result.claimed_rewards = result.claimed_rewards
      .map(summarizeClaimedRewardEntryForAgent)
      .filter((entry) => entry !== null);
    if (result.claimed_rewards.length <= 0) {
      delete result.claimed_rewards;
    }
  }

  if (isPlainObject(result.selected_card)) {
    result.selected_card = summarizeSelectedCardForAgent(result.selected_card);
    if (!isPlainObject(result.selected_card)) {
      delete result.selected_card;
    }
  }

  if (Array.isArray(result.executed_actions)) {
    result.executed_actions = compactRewardResolverExecutedActions(result.executed_actions);
    if (result.executed_actions.length <= 0) {
      delete result.executed_actions;
    }
  }

  if (isPlainObject(rawFinalState)) {
    result.final_state = summarizeRewardResolutionStateForAgent(rawFinalState);
    delete result.state;
  } else if (isPlainObject(rawState)) {
    result.state = summarizeRewardResolutionStateForAgent(rawState);
  }

  if (result.resolved === true) {
    delete result.reward_bundle;
    delete result.final_reward_bundle;
  }

  if (isPlainObject(result.final_state)) {
    delete result.state;
  }

  return result;
}

function compactTravelToCoordinatePayload(payload) {
  const rawState = isPlainObject(payload?.state) ? payload.state : null;
  const rawFinalState = isPlainObject(payload?.final_state) ? payload.final_state : null;
  const compacted = compactPayloadForOutput(payload);
  if (!isPlainObject(compacted)) {
    return compacted;
  }

  const result = {
    ...compacted
  };

  if (Array.isArray(result.executed_actions)) {
    result.executed_actions = compactTravelExecutedActions(result.executed_actions);
    if (result.executed_actions.length <= 0) {
      delete result.executed_actions;
    }
  }

  if (isPlainObject(result.snapshot_status)) {
    result.snapshot_status = summarizeMapSnapshotStatusForAgent(result.snapshot_status);
  }

  if (isPlainObject(rawFinalState)) {
    result.final_state = summarizeTravelResolutionStateForAgent(rawFinalState);
    delete result.state;
  } else if (isPlainObject(rawState)) {
    result.state = summarizeTravelResolutionStateForAgent(rawState);
  }

  if (isPlainObject(result.final_state)) {
    delete result.state;
  }

  return result;
}

function compactRestSitePayload(payload) {
  const rawState = isPlainObject(payload?.state) ? payload.state : null;
  const rawFinalState = isPlainObject(payload?.final_state) ? payload.final_state : null;
  const rawRestSiteBundle = isPlainObject(payload?.rest_site_bundle) ? payload.rest_site_bundle : null;
  const compacted = compactPayloadForOutput(payload);
  if (!isPlainObject(compacted)) {
    return compacted;
  }

  const result = {
    ...compacted
  };

  if (isPlainObject(rawRestSiteBundle)) {
    result.rest_site_bundle = summarizeRestSiteBundleForResolutionAgent(rawRestSiteBundle);
  } else if (isPlainObject(result.rest_site_bundle)) {
    result.rest_site_bundle = summarizeRestSiteBundleForResolutionAgent(result.rest_site_bundle);
  }

  if (isPlainObject(result.selected_option)) {
    result.selected_option = summarizeGenericIndexedOptionForAgent(result.selected_option);
  }
  if (result.selected_option == null) {
    delete result.selected_option;
  }

  if (isPlainObject(result.selected_upgrade_card)) {
    result.selected_upgrade_card = summarizeIndexedCardChoiceForAgent(result.selected_upgrade_card);
  }
  if (result.selected_upgrade_card == null) {
    delete result.selected_upgrade_card;
  }

  if (Array.isArray(result.executed_actions)) {
    result.executed_actions = compactShopLikeExecutedActions(result.executed_actions);
    if (result.executed_actions.length <= 0) {
      delete result.executed_actions;
    }
  }

  if (isPlainObject(rawFinalState)) {
    result.final_state = summarizeRestSiteResolutionStateForAgent(rawFinalState);
    delete result.state;
  } else if (isPlainObject(rawState)) {
    result.state = summarizeRestSiteResolutionStateForAgent(rawState);
  }

  if (result.resolved === true) {
    delete result.rest_site_bundle;
  }

  if (isPlainObject(result.rest_site_bundle) && isPlainObject(result.state)) {
    if (
      typeof result.rest_site_bundle.screen === "string" &&
      result.state.screen === result.rest_site_bundle.screen
    ) {
      delete result.state.screen;
    }
    delete result.state.available_actions;
    delete result.state.rest_site;
  }

  if (isPlainObject(result.final_state)) {
    delete result.state;
  }

  return result;
}

function compactShopVisitPayload(payload) {
  const rawState = isPlainObject(payload?.state) ? payload.state : null;
  const rawFinalState = isPlainObject(payload?.final_state) ? payload.final_state : null;
  const compacted = compactPayloadForOutput(payload);
  if (!isPlainObject(compacted)) {
    return compacted;
  }

  const result = {
    ...compacted
  };

  if (isPlainObject(result.shop_bundle)) {
    result.shop_bundle = summarizeShopBundleForResolutionAgent(result.shop_bundle);
  }

  if (isPlainObject(result.card_selection_bundle)) {
    result.card_selection_bundle = summarizeCardSelectionBundleForResolutionAgent(
      result.card_selection_bundle
    );
  }

  if (Array.isArray(result.available_cards)) {
    result.available_cards = result.available_cards
      .map((entry, fallbackIndex) => summarizeIndexedCardTitleChoiceForAgent(entry, fallbackIndex))
      .filter((entry) => entry !== null);
    if (result.available_cards.length <= 0) {
      delete result.available_cards;
    }
  }

  if (Array.isArray(result.purchase_plan)) {
    result.purchase_plan = result.purchase_plan
      .map(compactShopPurchasePlanStep)
      .filter((step) => step !== null);
    if (result.purchase_plan.length <= 0) {
      delete result.purchase_plan;
    }
  }

  if (Array.isArray(result.purchased_items)) {
    result.purchased_items = result.purchased_items
      .map(compactPurchasedShopItemEntry)
      .filter((entry) => entry !== null);
    if (result.purchased_items.length <= 0) {
      delete result.purchased_items;
    }
  }

  if (isPlainObject(result.removed_card)) {
    result.removed_card = compactRemovedShopCard(result.removed_card);
  }

  if (Array.isArray(result.executed_actions)) {
    result.executed_actions = compactShopLikeExecutedActions(result.executed_actions);
    if (result.executed_actions.length <= 0) {
      delete result.executed_actions;
    }
  }

  if (isPlainObject(rawFinalState)) {
    result.final_state = summarizeShopResolutionStateForAgent(rawFinalState);
    delete result.state;
  } else if (isPlainObject(rawState)) {
    result.state = summarizeShopResolutionStateForAgent(rawState);
  }

  if (result.resolved === true) {
    delete result.shop_bundle;
    delete result.purchase_plan;
  }

  if (
    (isPlainObject(result.shop_bundle) || isPlainObject(result.card_selection_bundle)) &&
    isPlainObject(result.state)
  ) {
    delete result.state.available_actions;
    delete result.state.shop;
    delete result.state.card_selection;
  }

  if (isPlainObject(result.final_state)) {
    delete result.state;
  }

  return result;
}

function compactPlayCardSequencePayload(payload) {
  const rawState = isPlainObject(payload?.state) ? payload.state : null;
  const rawInitialState = isPlainObject(payload?.initial_state)
    ? payload.initial_state
    : null;
  const rawStateAfter = isPlainObject(payload?.state_after) ? payload.state_after : null;
  const payloadWithoutInitialState = isPlainObject(payload)
    ? (() => {
      const cloned = {
        ...payload
      };
      delete cloned.initial_state;
      return cloned;
    })()
    : payload;
  const compacted = compactPayloadForOutput(payloadWithoutInitialState);
  if (!isPlainObject(compacted)) {
    return compacted;
  }

  const result = {
    ...compacted
  };

  delete result.interaction_hints;

  if (isPlainObject(rawStateAfter)) {
    result.state_after = summarizeCombatSequenceStateForAgent(
      rawStateAfter,
      rawInitialState
    );
    delete result.state;
  } else if (isPlainObject(rawState)) {
    result.state = summarizeCombatSequenceStateForAgent(rawState, rawInitialState);
  }

  if (Array.isArray(result.sequence_plan)) {
    result.sequence_plan = result.sequence_plan
      .map(compactCombatSequencePlanStep)
      .filter((step) => step !== null);
  }

  if (isPlainObject(result.failed_step)) {
    result.failed_step = compactCombatSequencePlanStep(result.failed_step);
  }

  if (Array.isArray(result.executed_steps)) {
    result.executed_steps = result.executed_steps
      .map((step) =>
        compactCombatSequenceExecutedStep(step, {
          includeRequestedActionId: !(result.ok === true && result.resolved === true)
        })
      )
      .filter((step) => step !== null);
  }

  if (result.initial_state_version_adjusted == null) {
    delete result.initial_state_version_adjusted;
  }

  if (result.reordered_end_turn_to_last !== true) {
    delete result.reordered_end_turn_to_last;
  }

  if (
    Array.isArray(result.requested_action_ids) &&
    Array.isArray(result.normalized_action_ids) &&
    areStringArraysEqual(result.requested_action_ids, result.normalized_action_ids)
  ) {
    delete result.normalized_action_ids;
  }

  if (result.resolved === true) {
    delete result.requested_action_ids;
    delete result.normalized_action_ids;
    delete result.remaining_count;
  }

  if (result.resolved === true) {
    delete result.sequence_plan;
  }

  if (result.ok === true && result.resolved === true) {
    delete result.requested_action_count;
    delete result.executed_count;
  }

  return result;
}

function compactCombatSequencePlanStep(step) {
  if (!isPlainObject(step)) {
    return null;
  }

  return {
    sequence_index: Number.isInteger(step.sequence_index) ? step.sequence_index : null,
    kind: typeof step.kind === "string" ? step.kind : null,
    requested_action_id:
      typeof step.requested_action_id === "string" ? step.requested_action_id : null
  };
}

function compactCombatSequenceExecutedStep(step, options = {}) {
  if (!isPlainObject(step)) {
    return null;
  }

  const executedActionId =
    typeof step.executed_action_id === "string" ? step.executed_action_id : null;
  if (!executedActionId) {
    return null;
  }

  const requestedActionId =
    typeof step.requested_action_id === "string" ? step.requested_action_id : null;
  const includeRequestedActionId = options?.includeRequestedActionId === true;
  const matchType =
    typeof step.match_type === "string" && step.match_type !== "exact"
      ? step.match_type
      : null;
  const compatibleCandidateCount =
    Number.isInteger(step.compatible_candidate_count) && step.compatible_candidate_count > 1
      ? step.compatible_candidate_count
      : null;
  const autoExecutedActions =
    Array.isArray(step?.execution?.auto_executed_actions) &&
      step.execution.auto_executed_actions.length > 0
      ? step.execution.auto_executed_actions
      : null;

  if (
    requestedActionId === null ||
    requestedActionId === executedActionId ||
    includeRequestedActionId !== true
  ) {
    if (matchType === null && compatibleCandidateCount === null && autoExecutedActions === null) {
      return executedActionId;
    }
  }

  const compactStep = {
    action_id: executedActionId
  };

  if (
    includeRequestedActionId === true &&
    requestedActionId !== null &&
    requestedActionId !== executedActionId
  ) {
    compactStep.requested_action_id = requestedActionId;
  }

  if (matchType !== null) {
    compactStep.match = matchType;
  }

  if (compatibleCandidateCount !== null) {
    compactStep.candidates = compatibleCandidateCount;
  }

  if (autoExecutedActions !== null) {
    compactStep.auto_executed_actions = autoExecutedActions;
  }

  return compactStep;
}

function summarizeClaimedRewardEntryForAgent(entry) {
  if (!isPlainObject(entry)) {
    return null;
  }

  const reward = summarizeRewardForAgent(entry.reward);
  if (!isPlainObject(reward)) {
    return null;
  }

  if (reward.reward_type === "card") {
    return {
      reward_type: reward.reward_type,
      description: reward.description,
      can_skip: reward.can_skip === true
    };
  }

  return reward;
}

function summarizeSelectedCardForAgent(selectedCard) {
  if (!isPlainObject(selectedCard)) {
    return null;
  }

  const card = summarizeCardForAgent(selectedCard.card);
  const summary = {};

  if (Number.isInteger(selectedCard.index)) {
    summary.index = selectedCard.index;
  }

  if (card !== null) {
    summary.card = card;
  }

  return Object.keys(summary).length > 0 ? summary : null;
}

function summarizeGenericIndexedOptionForAgent(option) {
  if (!isPlainObject(option)) {
    return null;
  }

  const summary = {};
  if (Number.isInteger(option.index)) {
    summary.index = option.index;
  }
  if (typeof option.option_type === "string") {
    summary.option_type = option.option_type;
  }
  if (typeof option.option_id === "string") {
    summary.option_id = option.option_id;
  }
  if (typeof option.title === "string") {
    summary.title = normalizeAgentText(option.title);
  }
  if (typeof option.description === "string") {
    summary.description = normalizeAgentText(option.description);
  }
  return Object.keys(summary).length > 0 ? summary : null;
}

function summarizeIndexedCardChoiceForAgent(choice) {
  if (!isPlainObject(choice)) {
    return null;
  }

  const card = summarizeCardForAgent(choice.card);
  const summary = {};
  if (Number.isInteger(choice.index)) {
    summary.index = choice.index;
  }
  if (card !== null) {
    summary.card = card;
  }
  return Object.keys(summary).length > 0 ? summary : null;
}

function compactInlineAgentText(value) {
  const normalized = normalizeAgentText(value);
  if (!normalized) {
    return null;
  }

  return normalized.replace(/\n+/g, " / ").replace(/[ \t]{2,}/g, " ").trim() || null;
}

function summarizeIndexedCardTitleChoiceForAgent(choice, fallbackIndex = null) {
  if (!isPlainObject(choice)) {
    return null;
  }

  const optionIndex = Number.isInteger(choice.index) ? choice.index : fallbackIndex;
  const title = normalizeAgentText(choice?.card?.title ?? choice?.title);
  const summary = {};
  if (Number.isInteger(optionIndex)) {
    summary.index = optionIndex;
  }
  if (title) {
    summary.title = title;
  }
  if (choice.is_selected === true) {
    summary.is_selected = true;
  }
  return Object.keys(summary).length > 0 ? summary : null;
}

function summarizeDeckUpgradeChoiceForResolutionAgent(choice, fallbackIndex = null) {
  if (!isPlainObject(choice)) {
    return null;
  }

  const optionIndex = Number.isInteger(choice.index) ? choice.index : fallbackIndex;
  if (!Number.isInteger(optionIndex)) {
    return null;
  }

  const baseCard = isPlainObject(choice.card) ? choice.card : null;
  const previewCard = isPlainObject(choice.upgrade_preview) ? choice.upgrade_preview : null;
  const baseTitle = compactInlineAgentText(baseCard?.title ?? choice?.title);
  if (!baseTitle) {
    return `${optionIndex}`;
  }

  const previewTitle = compactInlineAgentText(previewCard?.title);
  const previewEffect =
    compactInlineAgentText(previewCard?.description) ??
    compactInlineAgentText(previewCard?.effect) ??
    compactInlineAgentText(previewCard?.effect_preview?.summary);
  const previewCost = Number.isInteger(previewCard?.cost)
    ? previewCard.cost
    : Number.isInteger(previewCard?.resolved_energy_cost)
      ? previewCard.resolved_energy_cost
      : Number.isInteger(previewCard?.canonical_energy_cost)
        ? previewCard.canonical_energy_cost
        : null;
  const previewParts = [];
  if (previewTitle && previewTitle !== baseTitle) {
    previewParts.push(previewTitle);
  }
  if (previewCost !== null) {
    previewParts.push(`${previewCost}费`);
  }
  if (previewEffect) {
    previewParts.push(previewEffect);
  }

  const suffix = previewParts.length > 0 ? ` -> ${previewParts.join(" | ")}` : "";
  return `${optionIndex}:${baseTitle}${suffix}`;
}

function compactRewardResolverExecutedActions(actions) {
  if (!Array.isArray(actions)) {
    return [];
  }

  const compacted = actions
    .map((action) => compactRewardResolverExecutedAction(action))
    .filter((action) => action !== null);

  const deduped = [];
  for (const action of compacted) {
    const previous = deduped.length > 0 ? deduped[deduped.length - 1] : null;
    if (areCompactedActionsEquivalent(previous, action)) {
      continue;
    }

    deduped.push(action);
  }

  return deduped;
}

function compactRewardResolverExecutedAction(action) {
  if (!isPlainObject(action)) {
    return null;
  }

  const rawActionId = typeof action.action_id === "string" ? action.action_id : null;
  if (!rawActionId) {
    return null;
  }

  const actionId = getCompactRewardResolverActionId(rawActionId);
  if (!actionId) {
    return null;
  }

  const autoExecutedActions =
    Array.isArray(action.auto_executed_actions) && action.auto_executed_actions.length > 0
      ? action.auto_executed_actions
      : null;
  const screenAfter = typeof action.screen_after === "string" ? action.screen_after : null;
  const settleReason =
    typeof action.post_action_settle_reason === "string" ? action.post_action_settle_reason : null;
  const settlePolls =
    Number.isInteger(action.post_action_settle_polls) && action.post_action_settle_polls > 0
      ? action.post_action_settle_polls
      : null;
  const commonSettleReason =
    settleReason === "reward_flow_ready" ||
    settleReason === "reward_card_selection_ready" ||
    settleReason === "map_ready";

  if (
    autoExecutedActions === null &&
    screenAfter === null &&
    settlePolls === null &&
    (settleReason === null || commonSettleReason)
  ) {
    return actionId;
  }

  const summary = {
    action_id: actionId
  };

  if (settleReason !== null && !commonSettleReason) {
    summary.settled = settleReason;
  }

  if (settlePolls !== null) {
    summary.polls = settlePolls;
  }

  if (screenAfter !== null) {
    summary.screen_after = screenAfter;
  }

  if (autoExecutedActions !== null) {
    summary.auto_executed_actions = autoExecutedActions;
  }

  return summary;
}

function getCompactRewardResolverActionId(actionId) {
  if (typeof actionId !== "string" || actionId.length <= 0) {
    return null;
  }

  if (actionId.startsWith("reward:")) {
    return "reward";
  }

  if (actionId === "card_reward:skip") {
    return "card_reward:skip";
  }

  if (actionId.startsWith("card_reward:")) {
    return "card_reward:pick";
  }

  return actionId;
}

function areCompactedActionsEquivalent(left, right) {
  if (left === null || right === null) {
    return false;
  }

  if (typeof left === "string" || typeof right === "string") {
    return left === right;
  }

  if (!isPlainObject(left) || !isPlainObject(right)) {
    return false;
  }

  const leftKeys = Object.keys(left).sort();
  const rightKeys = Object.keys(right).sort();
  if (leftKeys.length !== rightKeys.length) {
    return false;
  }

  for (let index = 0; index < leftKeys.length; index += 1) {
    if (leftKeys[index] !== rightKeys[index]) {
      return false;
    }
  }

  return JSON.stringify(left) === JSON.stringify(right);
}

function compactShopLikeExecutedActions(actions) {
  if (!Array.isArray(actions)) {
    return [];
  }

  return actions
    .map((action) => compactShopLikeExecutedAction(action))
    .filter((action) => action !== null);
}

function compactShopLikeExecutedAction(action) {
  if (!isPlainObject(action)) {
    return null;
  }

  const actionId = typeof action.action_id === "string" ? action.action_id : null;
  if (!actionId) {
    return null;
  }

  const autoExecutedActions =
    Array.isArray(action.auto_executed_actions) && action.auto_executed_actions.length > 0
      ? action.auto_executed_actions
      : null;
  const settleReason =
    typeof action.post_action_settle_reason === "string" ? action.post_action_settle_reason : null;
  const settlePolls =
    Number.isInteger(action.post_action_settle_polls) && action.post_action_settle_polls > 0
      ? action.post_action_settle_polls
      : null;
  const commonSettleReason =
    settleReason === null ||
    settleReason === "shop_ready" ||
    settleReason === "rest_site_upgrade_ready" ||
    settleReason === "map_ready";

  if (autoExecutedActions === null && settlePolls === null && commonSettleReason) {
    return actionId;
  }

  const summary = {
    action_id: actionId
  };

  if (settleReason !== null && !commonSettleReason) {
    summary.settled = settleReason;
  }

  if (settlePolls !== null) {
    summary.polls = settlePolls;
  }

  if (autoExecutedActions !== null) {
    summary.auto_executed_actions = autoExecutedActions;
  }

  return summary;
}

function compactShopPurchasePlanStep(step) {
  if (!isPlainObject(step)) {
    return null;
  }

  const summary = {
    sequence_index: Number.isInteger(step.sequence_index) ? step.sequence_index : null
  };
  const request = isPlainObject(step.request) ? step.request : null;
  const item = isPlainObject(step.item) ? step.item : null;

  if (request) {
    if (typeof request.title === "string") {
      summary.title = normalizeAgentText(request.title);
    }
    if (typeof request.item_kind === "string") {
      summary.item_kind = request.item_kind;
    }
  }

  if (item) {
    if (typeof summary.title !== "string" && typeof item.title === "string") {
      summary.title = normalizeAgentText(item.title);
    }
    if (typeof summary.item_kind !== "string" && typeof item.item_kind === "string") {
      summary.item_kind = item.item_kind;
    }
    if (Number.isFinite(item.cost)) {
      summary.cost = item.cost;
    }
  }

  return summary;
}

function compactPurchasedShopItemEntry(entry) {
  if (!isPlainObject(entry)) {
    return null;
  }

  const item = isPlainObject(entry.item) ? entry.item : null;
  const request = isPlainObject(entry.request) ? entry.request : null;
  const summary = {};

  if (Number.isInteger(entry.sequence_index)) {
    summary.sequence_index = entry.sequence_index;
  }

  if (request && typeof request.title === "string") {
    summary.requested_title = normalizeAgentText(request.title);
  }

  if (item && typeof item.title === "string") {
    summary.title = normalizeAgentText(item.title);
  }

  if (item && typeof item.item_kind === "string") {
    summary.item_kind = item.item_kind;
  } else if (request && typeof request.item_kind === "string") {
    summary.item_kind = request.item_kind;
  }

  if (item && Number.isFinite(item.cost)) {
    summary.cost = item.cost;
  }

  if (typeof entry.match_type === "string" && entry.match_type !== "exact") {
    summary.match = entry.match_type;
  }

  if (Number.isInteger(entry.compatible_candidate_count) && entry.compatible_candidate_count > 1) {
    summary.candidates = entry.compatible_candidate_count;
  }

  return Object.keys(summary).length > 0 ? summary : null;
}

function compactRemovedShopCard(removedCard) {
  if (!isPlainObject(removedCard)) {
    return null;
  }

  const summary = {};
  if (typeof removedCard.requested_title === "string") {
    summary.requested_title = normalizeAgentText(removedCard.requested_title);
  }
  if (Number.isInteger(removedCard.requested_index)) {
    summary.requested_index = removedCard.requested_index;
  }
  if (Number.isInteger(removedCard.selected_option_index)) {
    summary.selected_option_index = removedCard.selected_option_index;
  }
  if (typeof removedCard.match_type === "string" && removedCard.match_type !== "title") {
    summary.match = removedCard.match_type;
  }
  if (
    Number.isInteger(removedCard.compatible_option_count) &&
    removedCard.compatible_option_count > 1
  ) {
    summary.candidates = removedCard.compatible_option_count;
  }
  const card = summarizeCardForAgent(removedCard.card);
  if (card !== null) {
    summary.card = card;
  }
  return Object.keys(summary).length > 0 ? summary : null;
}

function summarizeRewardResolutionStateForAgent(state) {
  if (!isPlainObject(state)) {
    return state;
  }

  const summary = summarizeStateForAgent(state);
  if (!isPlainObject(summary)) {
    return summary;
  }

  const result = {
    screen: summary.screen,
    state_version: summary.state_version
  };

  if (isPlainObject(summary.player)) {
    result.player = {
      current_hp: Number.isFinite(summary.player.current_hp) ? summary.player.current_hp : null,
      max_hp: Number.isFinite(summary.player.max_hp) ? summary.player.max_hp : null,
      gold: Number.isFinite(summary.player.gold) ? summary.player.gold : null,
      potions: Array.isArray(summary.player.potions) ? summary.player.potions : []
    };
  }

  if (Array.isArray(summary.available_actions) && summary.screen !== "MAP") {
    result.available_actions = summary.available_actions;
  }

  if (isPlainObject(summary.rewards)) {
    result.rewards = summary.rewards;
  }

  if (isPlainObject(summary.card_selection)) {
    result.card_selection = summary.card_selection;
  }

  if (isPlainObject(summary.map)) {
    result.map = summary.map;
  }

  return result;
}

function summarizeRestSiteResolutionStateForAgent(state) {
  if (!isPlainObject(state)) {
    return state;
  }

  const summary = summarizeStateForAgent(state);
  if (!isPlainObject(summary)) {
    return summary;
  }

  const result = {
    screen: summary.screen,
    state_version: summary.state_version
  };

  if (isPlainObject(summary.player)) {
    result.player = {
      current_hp: Number.isFinite(summary.player.current_hp) ? summary.player.current_hp : null,
      max_hp: Number.isFinite(summary.player.max_hp) ? summary.player.max_hp : null,
      gold: Number.isFinite(summary.player.gold) ? summary.player.gold : null,
      potions: Array.isArray(summary.player.potions) ? summary.player.potions : []
    };
  }

  if (isPlainObject(summary.map)) {
    result.map = summary.map;
  }

  return result;
}

function summarizeShopResolutionStateForAgent(state) {
  if (!isPlainObject(state)) {
    return state;
  }

  const summary = summarizeStateForAgent(state);
  if (!isPlainObject(summary)) {
    return summary;
  }

  const result = {
    screen: summary.screen,
    state_version: summary.state_version
  };

  if (isPlainObject(summary.player)) {
    result.player = {
      current_hp: Number.isFinite(summary.player.current_hp) ? summary.player.current_hp : null,
      max_hp: Number.isFinite(summary.player.max_hp) ? summary.player.max_hp : null,
      gold: Number.isFinite(summary.player.gold) ? summary.player.gold : null,
      potions: Array.isArray(summary.player.potions) ? summary.player.potions : []
    };
  }

  if (isPlainObject(summary.map)) {
    result.map = summary.map;
  }

  return result;
}

function compactTravelExecutedActions(actions) {
  if (!Array.isArray(actions)) {
    return [];
  }

  return actions
    .map((action) => compactTravelExecutedAction(action))
    .filter((action) => action !== null);
}

function compactTravelExecutedAction(action) {
  if (!isPlainObject(action)) {
    return null;
  }

  const actionId = typeof action.action_id === "string" ? action.action_id : null;
  if (!actionId) {
    return null;
  }

  const autoExecutedActions =
    Array.isArray(action.auto_executed_actions) && action.auto_executed_actions.length > 0
      ? action.auto_executed_actions
      : null;
  const screenAfter = typeof action.screen_after === "string" ? action.screen_after : null;
  const settleReason =
    typeof action.post_action_settle_reason === "string" ? action.post_action_settle_reason : null;
  const settlePolls =
    Number.isInteger(action.post_action_settle_polls) && action.post_action_settle_polls > 0
      ? action.post_action_settle_polls
      : null;
  const commonSettleReason =
    settleReason === "reward_flow_ready" ||
    settleReason === "map_ready" ||
    settleReason === "player_turn_stable";

  if (
    autoExecutedActions === null &&
    screenAfter === null &&
    settlePolls === null &&
    (settleReason === null || commonSettleReason)
  ) {
    return actionId;
  }

  const summary = {
    action_id: actionId
  };

  if (settleReason !== null && !commonSettleReason) {
    summary.settled = settleReason;
  }

  if (settlePolls !== null) {
    summary.polls = settlePolls;
  }

  if (screenAfter !== null) {
    summary.screen_after = screenAfter;
  }

  if (autoExecutedActions !== null) {
    summary.auto_executed_actions = autoExecutedActions;
  }

  return summary;
}

function summarizeTravelResolutionStateForAgent(state) {
  if (!isPlainObject(state)) {
    return state;
  }

  if (state.screen === "COMBAT") {
    return summarizeActionStateForAgent(state);
  }

  const summary = summarizeStateForAgent(state);
  if (!isPlainObject(summary)) {
    return summary;
  }

  const result = {
    screen: summary.screen,
    state_version: summary.state_version
  };

  if (isPlainObject(summary.player)) {
    result.player = {
      current_hp: Number.isFinite(summary.player.current_hp) ? summary.player.current_hp : null,
      max_hp: Number.isFinite(summary.player.max_hp) ? summary.player.max_hp : null,
      gold: Number.isFinite(summary.player.gold) ? summary.player.gold : null,
      potions: Array.isArray(summary.player.potions) ? summary.player.potions : []
    };
  }

  if (Array.isArray(summary.available_actions) && summary.screen !== "MAP") {
    result.available_actions = summary.available_actions;
  }

  if (isPlainObject(summary.map)) {
    result.map = summary.map;
  }

  if (isPlainObject(summary.rewards)) {
    result.rewards = summary.rewards;
  }

  if (isPlainObject(summary.card_selection)) {
    result.card_selection = summary.card_selection;
  }

  return result;
}

function summarizeCombatSequenceStateForAgent(state, initialState = null) {
  if (!isPlainObject(state)) {
    return state;
  }

  const player = isPlainObject(state?.players?.[0]) ? state.players[0] : {};
  const playerCreature = isPlainObject(player.creature) ? player.creature : {};
  const playerCombat = isPlainObject(player.combat) ? player.combat : {};

  return {
    screen: typeof state.screen === "string" ? state.screen : null,
    state_version: Number.isFinite(state.state_version) ? state.state_version : null,
    player: {
      current_hp: Number.isFinite(playerCreature.current_hp) ? playerCreature.current_hp : null,
      max_hp: Number.isFinite(playerCreature.max_hp) ? playerCreature.max_hp : null,
      energy: Number.isFinite(playerCombat.energy) ? playerCombat.energy : null,
      max_energy: Number.isFinite(playerCombat.max_energy) ? playerCombat.max_energy : null,
      stars: Number.isFinite(playerCombat.stars) ? playerCombat.stars : null
    },
    enemy_changes: summarizeCombatSequenceEnemyChanges(initialState, state),
    available_actions: summarizeCombatSequenceAvailableActions(getNonAutomationActions(state))
  };
}

function summarizeCombatSequenceEnemyChanges(initialState, state) {
  const initialEnemyMap = collectCombatEnemyStates(initialState);
  const currentEnemyMap = collectCombatEnemyStates(state);
  const keys = [...new Set([...initialEnemyMap.keys(), ...currentEnemyMap.keys()])];
  const inferAllMissingEnemiesDefeated =
    initialEnemyMap.size > 0 &&
    currentEnemyMap.size === 0 &&
    typeof state?.screen === "string" &&
    state.screen !== "COMBAT";
  const changes = [];

  for (const key of keys) {
    const before = initialEnemyMap.get(key) ?? null;
    let after = currentEnemyMap.get(key) ?? null;
    if (!after && before && inferAllMissingEnemiesDefeated) {
      after = {
        ...before,
        current_hp: 0,
        block: 0,
        is_alive: false
      };
    }

    if (!before && !after) {
      continue;
    }

    const currentHpChanged =
      (Number.isFinite(before?.current_hp) ? before.current_hp : null) !==
      (Number.isFinite(after?.current_hp) ? after.current_hp : null);
    const aliveChanged =
      (typeof before?.is_alive === "boolean" ? before.is_alive : null) !==
      (typeof after?.is_alive === "boolean" ? after.is_alive : null);

    if (!currentHpChanged && !aliveChanged && before && after) {
      continue;
    }

    const hpDelta =
      Number.isFinite(before?.current_hp) && Number.isFinite(after?.current_hp)
        ? after.current_hp - before.current_hp
        : null;
    let change = "status_changed";
    if (!before && after) {
      change = "appeared";
    } else if (before && !after) {
      change = "removed";
    } else if (before && after && before.is_alive !== false && after.is_alive === false) {
      change = "defeated";
    } else if (currentHpChanged) {
      change = "hp_changed";
    }

    changes.push({
      combat_id: Number.isFinite(after?.combat_id) ? after.combat_id : before?.combat_id ?? null,
      name: typeof after?.name === "string" ? after.name : before?.name ?? null,
      current_hp: Number.isFinite(after?.current_hp) ? after.current_hp : null,
      max_hp: Number.isFinite(after?.max_hp) ? after.max_hp : before?.max_hp ?? null,
      is_alive:
        typeof after?.is_alive === "boolean"
          ? after.is_alive
          : typeof before?.is_alive === "boolean"
            ? before.is_alive
            : null,
      hp_delta: hpDelta,
      change
    });
  }

  return changes.sort(compareCombatSequenceEnemyChanges);
}

function collectCombatEnemyStates(state) {
  const enemies = Array.isArray(state?.combat?.enemy_creatures) ? state.combat.enemy_creatures : [];
  const enemyMap = new Map();

  for (let index = 0; index < enemies.length; index += 1) {
    const enemy = enemies[index];
    if (!isPlainObject(enemy)) {
      continue;
    }

    const combatId = Number.isFinite(enemy.combat_id) ? enemy.combat_id : null;
    const name = normalizeAgentText(enemy.name);
    const key = combatId !== null ? `combat:${combatId}` : `name:${name ?? "unknown"}:${index}`;
    enemyMap.set(key, {
      combat_id: combatId,
      name,
      current_hp: Number.isFinite(enemy.current_hp) ? enemy.current_hp : null,
      max_hp: Number.isFinite(enemy.max_hp) ? enemy.max_hp : null,
      block: Number.isFinite(enemy.block) ? enemy.block : 0,
      is_alive:
        typeof enemy.is_alive === "boolean"
          ? enemy.is_alive
          : Number.isFinite(enemy.current_hp)
            ? enemy.current_hp > 0
            : null
    });
  }

  return enemyMap;
}

function compareCombatSequenceEnemyChanges(left, right) {
  const leftCombatId = Number.isFinite(left?.combat_id) ? left.combat_id : Number.POSITIVE_INFINITY;
  const rightCombatId = Number.isFinite(right?.combat_id) ? right.combat_id : Number.POSITIVE_INFINITY;
  if (leftCombatId !== rightCombatId) {
    return leftCombatId - rightCombatId;
  }

  return String(left?.name ?? "").localeCompare(String(right?.name ?? ""), "zh-Hans-CN");
}

function summarizeCombatSequenceAvailableActions(actions) {
  if (!Array.isArray(actions)) {
    return [];
  }

  return actions
    .map(summarizeCombatSequenceAvailableAction)
    .filter((action) => action !== null);
}

function summarizeCombatSequenceAvailableAction(action) {
  if (!isPlainObject(action)) {
    return null;
  }

  const summary = {
    action_id: typeof action.action_id === "string" ? action.action_id : null
  };

  if (typeof action?.card?.title === "string" && action.card.title.trim()) {
    summary.title = normalizeAgentText(action.card.title);
    if (Number.isInteger(action.card.resolved_energy_cost)) {
      summary.cost = action.card.resolved_energy_cost;
    }
    const starCost = readAgentCardStarCost(action.card);
    if (starCost !== null) {
      summary.star_cost = starCost;
    }
  } else if (typeof action?.potion?.title === "string" && action.potion.title.trim()) {
    summary.title = normalizeAgentText(action.potion.title);
  } else if (typeof action.label === "string" && action.label.trim()) {
    summary.label = normalizeAgentText(action.label);
  }

  const targetName =
    typeof action.target_name === "string"
      ? normalizeAgentText(action.target_name)
      : normalizeAgentText(action?.target?.name);
  const targetCombatId = Number.isFinite(action.target_combat_id)
    ? action.target_combat_id
    : Number.isFinite(action?.target?.combat_id)
      ? action.target.combat_id
      : null;
  const targetActionSuffix =
    typeof action.target_action_suffix === "string"
      ? action.target_action_suffix
      : typeof action?.target_mapping?.action_suffix === "string"
        ? action.target_mapping.action_suffix
        : null;

  const targetParts = [];
  if (targetName !== null) {
    targetParts.push(targetName);
  }
  if (targetCombatId !== null) {
    targetParts.push(`#${targetCombatId}`);
  } else if (targetActionSuffix !== null && targetName === null) {
    targetParts.push(targetActionSuffix);
  }

  if (targetParts.length > 0) {
    summary.target = targetParts.join(" ");
  }

  return summary;
}

function areStringArraysEqual(left, right) {
  if (!Array.isArray(left) || !Array.isArray(right) || left.length !== right.length) {
    return false;
  }

  for (let index = 0; index < left.length; index += 1) {
    if (left[index] !== right[index]) {
      return false;
    }
  }

  return true;
}

function getRewardResolutionBlocker(rewardBundle, options) {
  const cardRewardEntry = rewardBundle.rewards.entries.find(
    (entry) => entry?.reward?.reward_type === "card"
  );
  const potionRewards = rewardBundle.rewards.entries.filter(
    (entry) => entry?.reward?.reward_type === "potion"
  );
  const hasVisibleCardSelection = rewardBundle.card_reward_selection.visible;

  if ((cardRewardEntry || hasVisibleCardSelection) &&
    options.pickCardIndex === undefined &&
    !options.skipCardReward) {
    return {
      reason: "card_choice_required",
      message:
        "A card reward is present. Provide pick_card_index to resolve it in one tool call, or set skip_card_reward=true."
    };
  }

  if (options.skipCardReward) {
    const canSkipVisibleSelection = !hasVisibleCardSelection ||
      rewardBundle.card_reward_selection.skip_visible === true;
    const canSkipPendingReward = !cardRewardEntry || cardRewardEntry?.reward?.can_skip !== false;

    if (!canSkipVisibleSelection || !canSkipPendingReward) {
      return {
        reason: "card_skip_unavailable",
        message: "The current card reward does not expose a skip action."
      };
    }
  }

  if (options.claimAllSafeRewards &&
    options.takePotions &&
    potionRewards.length > 0 &&
    rewardBundle.empty_potion_slot_count <= 0) {
    return {
      reason: "potion_slots_full",
      message:
        "A potion reward is present, but all potion slots are full. This still needs a separate replacement/drop decision flow."
    };
  }

  return null;
}

function chooseNextReward(rewardBundle, options) {
  if (options?.claimAllSafeRewards === false) {
    return null;
  }

  const claimableRewards = rewardBundle.rewards.entries.filter((entry) => {
    const rewardType = entry?.reward?.reward_type;

    if (!entry?.action_id || !rewardType) {
      return false;
    }

    if (rewardType === "gold" || rewardType === "relic") {
      return true;
    }

    if (rewardType === "potion") {
      return options.takePotions && rewardBundle.empty_potion_slot_count > 0;
    }

    if (rewardType === "card") {
      return options.pickCardIndex !== undefined;
    }

    return false;
  });

  return claimableRewards[0] ?? null;
}

function canAutoProceedFromRewardCleanupState(state) {
  const rewardBundle = buildRewardBundle(state);
  const nonAutomationActionIds = getNonAutomationActions(state)
    .map((action) => action?.action_id)
    .filter((actionId) => typeof actionId === "string");

  return (
    rewardBundle.in_reward_flow &&
    rewardBundle.has_proceed === true &&
    rewardBundle.rewards.entries.length === 0 &&
    rewardBundle.card_reward_selection.visible !== true &&
    nonAutomationActionIds.includes("proceed") &&
    nonAutomationActionIds.every(
      (actionId) => actionId === "proceed" || actionId.startsWith("discard_potion:")
    )
  );
}

function canAutoProceedFromRestSiteCleanupState(state) {
  const actionIds = getNonAutomationActions(state)
    .map((action) => action?.action_id)
    .filter((actionId) => typeof actionId === "string");
  const restSiteBundle = buildRestSiteBundle(state);

  return (
    restSiteBundle.in_rest_site_flow &&
    restSiteBundle.deck_upgrade_selection.visible !== true &&
    actionIds.includes("rest_site:proceed") &&
    actionIds.every(
      (actionId) =>
        actionId === "rest_site:proceed" || actionId.startsWith("discard_potion:")
    )
  );
}

async function autoAdvanceProceedChain(session, initialState) {
  const executedActions = [];
  let state = initialState;

  for (let attempt = 0; attempt < 3; attempt += 1) {
    if (!canAutoProceedFromRewardCleanupState(state)) {
      break;
    }

    const baselineStateVersion = state?.state_version ?? null;
    const baselineStateHash = state?.state_hash ?? null;
    const result = await performBridgeAction(
      session,
      "proceed",
      DEFAULT_ACTION_WAIT_MS
    );
    executedActions.push(summarizeExecutedAction(result));
    state = result.state;

    const stateChanged =
      state?.state_version !== baselineStateVersion ||
      state?.state_hash !== baselineStateHash;

    if (!stateChanged) {
      break;
    }
  }

  return {
    state,
    executed_actions: executedActions
  };
}

async function autoAdvanceRestSiteProceedChain(session, initialState) {
  const executedActions = [];
  let state = await waitForRoomExitSettlement(
    session,
    initialState,
    (candidate) => isMapReadyState(candidate) || !buildRestSiteBundle(candidate).in_rest_site_flow
  );

  for (let attempt = 0; attempt < 3; attempt += 1) {
    if (isMapReadyState(state)) {
      break;
    }

    if (!canAutoProceedFromRestSiteCleanupState(state)) {
      break;
    }

    const result = await performBridgeAction(
      session,
      "rest_site:proceed",
      DEFAULT_ACTION_WAIT_MS
    );
    executedActions.push(summarizeExecutedAction(result));
    state = await waitForRoomExitSettlement(
      session,
      result.state,
      (candidate) => isMapReadyState(candidate) || !buildRestSiteBundle(candidate).in_rest_site_flow
    );

    if (isMapReadyState(state)) {
      break;
    }

    const remainingActionIds = getNonAutomationActions(state)
      .map((action) => action?.action_id)
      .filter((actionId) => typeof actionId === "string");
    if (
      !remainingActionIds.includes("rest_site:proceed") ||
      !remainingActionIds.every(
        (actionId) =>
          actionId === "rest_site:proceed" || actionId.startsWith("discard_potion:")
      )
    ) {
      break;
    }
  }

  return {
    state,
    executed_actions: executedActions
  };
}

async function waitForRoomExitSettlement(session, initialState, predicate) {
  let state = initialState;
  if (predicate(state)) {
    return state;
  }

  const startedAt = Date.now();
  while (Date.now() - startedAt < ROOM_EXIT_SETTLE_TIMEOUT_MS) {
    await delay(ROOM_EXIT_SETTLE_POLL_INTERVAL_MS);
    state = await getBridgeState(session);
    if (predicate(state)) {
      return state;
    }
  }

  return state;
}

function isMapReadyState(state) {
  return (
    state?.screen === "MAP" &&
    state?.map?.is_open === true &&
    state?.map?.is_travel_enabled === true
  );
}

function summarizeExecutedAction(result) {
  return {
    action_id: result?.action_id ?? null,
    matched_action: summarizeActionForAgent(result?.matched_action),
    auto_executed_actions: Array.isArray(result?.auto_executed_actions)
      ? result.auto_executed_actions
        .map((action) => summarizeAutoExecutedActionForAgent(action))
        .filter((action) => action !== null)
      : [],
    post_action_settled: result?.post_action_settled ?? null,
    post_action_settle_reason: result?.post_action_settle_reason ?? null,
    post_action_settle_polls: result?.post_action_settle_polls ?? null,
    state_version_after: result?.state_version_after ?? result?.state?.state_version ?? null,
    screen_after: result?.state?.screen ?? null
  };
}

function summarizeAutoExecutedActionForAgent(action) {
  if (typeof action === "string") {
    return action;
  }

  if (!isPlainObject(action)) {
    return null;
  }

  const actionId = typeof action.action_id === "string" ? action.action_id : null;
  if (!actionId) {
    return null;
  }

  const source = typeof action.source === "string" ? action.source : null;
  return source === null
    ? actionId
    : {
      action_id: actionId,
      source
    };
}

function getSessionFilePath() {
  if (process.env.STS2_BRIDGE_SESSION_FILE) {
    return process.env.STS2_BRIDGE_SESSION_FILE;
  }

  if (process.env.APPDATA) {
    return path.join(process.env.APPDATA, "SlayTheSpire2", "bridge", "session.json");
  }

  return path.join(
    os.homedir(),
    "AppData",
    "Roaming",
    "SlayTheSpire2",
    "bridge",
    "session.json"
  );
}

function getLogFilePath() {
  if (process.env.STS2_MCP_LOG_FILE) {
    return process.env.STS2_MCP_LOG_FILE;
  }

  if (process.env.APPDATA) {
    return path.join(process.env.APPDATA, "SlayTheSpire2", "bridge", DEFAULT_LOG_FILE_NAME);
  }

  return path.join(
    os.homedir(),
    "AppData",
    "Roaming",
    "SlayTheSpire2",
    "bridge",
    DEFAULT_LOG_FILE_NAME
  );
}

function getLiveSession() {
  const sessionFilePath = getSessionFilePath();

  if (!fs.existsSync(sessionFilePath)) {
    throw new ToolPayloadError(
      "session_file_missing",
      "Bridge session file was not found.",
      {
        session_file_path: sessionFilePath
      }
    );
  }

  const session = readSessionFile(sessionFilePath);
  if (!isProcessAlive(session.pid)) {
    throw new ToolPayloadError(
      "stale_session",
      "Session file exists, but the recorded game process is not alive.",
      {
        session_file_path: sessionFilePath,
        session
      }
    );
  }

  if (!session.base_url || typeof session.base_url !== "string") {
    throw new ToolPayloadError(
      "invalid_session_file",
      "Session file does not contain a valid base_url.",
      {
        session_file_path: sessionFilePath,
        session
      }
    );
  }

  return session;
}

function readSessionFile(sessionFilePath) {
  const rawSession = fs.readFileSync(sessionFilePath, "utf8");
  return JSON.parse(rawSession);
}

function buildAuthHeaders(session) {
  return session.token
    ? {
      Authorization: `Bearer ${session.token}`
    }
    : {};
}

function getBridgeSessionKey(session) {
  return JSON.stringify({
    session_id: typeof session?.session_id === "string" ? session.session_id : null,
    base_url: typeof session?.base_url === "string" ? session.base_url : null,
    token: typeof session?.token === "string" ? session.token : null,
    pid: Number.isInteger(session?.pid) ? session.pid : null
  });
}

function getStateVersionValue(state) {
  return Number.isInteger(state?.state_version) ? state.state_version : null;
}

function chooseNewerBridgeState(currentState, candidateState) {
  if (!isPlainObject(candidateState)) {
    return isPlainObject(currentState) ? currentState : null;
  }

  if (!isPlainObject(currentState)) {
    return candidateState;
  }

  const currentVersion = getStateVersionValue(currentState);
  const candidateVersion = getStateVersionValue(candidateState);
  if (currentVersion !== null && candidateVersion !== null) {
    if (candidateVersion > currentVersion) {
      return candidateState;
    }

    if (candidateVersion < currentVersion) {
      return currentState;
    }

    return candidateState;
  }

  if (candidateVersion !== null) {
    return candidateState;
  }

  if (currentVersion !== null) {
    return currentState;
  }

  return candidateState;
}

class BridgeEventClient {
  constructor(session) {
    this.sessionKey = getBridgeSessionKey(session);
    this.session = session;
    this.closed = false;
    this.started = false;
    this.connectPromise = null;
    this.abortController = null;
    this.latestState = null;
    this.latestStateReceivedAt = 0;
    this.waiters = new Set();
  }

  updateSession(session) {
    this.session = session;
  }

  async start() {
    if (this.closed) {
      throw new ToolPayloadError(
        "bridge_event_client_closed",
        "Bridge event client was already closed."
      );
    }

    if (this.started) {
      return;
    }

    this.started = true;
    this.abortController = new AbortController();
    this.connectPromise = this.runLoop();
  }

  close() {
    if (this.closed) {
      return;
    }

    this.closed = true;
    if (this.abortController) {
      this.abortController.abort();
    }

    for (const waiter of this.waiters) {
      clearTimeout(waiter.timeoutHandle);
      waiter.reject(
        new ToolPayloadError(
          "bridge_event_stream_closed",
          "Bridge event stream client was closed."
        )
      );
    }

    this.waiters.clear();
  }

  getCachedState(maxAgeMs = BRIDGE_EVENT_CACHE_MAX_AGE_MS) {
    if (!isPlainObject(this.latestState)) {
      return null;
    }

    if (!Number.isFinite(maxAgeMs) || maxAgeMs < 0) {
      return this.latestState;
    }

    return Date.now() - this.latestStateReceivedAt <= maxAgeMs
      ? this.latestState
      : null;
  }

  ingestState(state) {
    if (!isPlainObject(state)) {
      return null;
    }

    const adoptedState = chooseNewerBridgeState(this.latestState, state);
    if (!isPlainObject(adoptedState)) {
      return null;
    }

    if (adoptedState !== this.latestState) {
      this.latestState = adoptedState;
      this.latestStateReceivedAt = Date.now();
      this.resolveWaiters(adoptedState);
      return adoptedState;
    }

    const currentVersion = getStateVersionValue(this.latestState);
    const incomingVersion = getStateVersionValue(state);
    const shouldRefreshTimestamp =
      incomingVersion === null ||
      currentVersion === null ||
      incomingVersion === currentVersion;

    if (shouldRefreshTimestamp) {
      this.latestStateReceivedAt = Date.now();
      this.resolveWaiters(this.latestState);
    }

    return this.latestState;
  }

  async waitForState(options = {}) {
    const predicate =
      typeof options.predicate === "function" ? options.predicate : () => true;
    const afterStateVersion = Number.isInteger(options.after_state_version)
      ? options.after_state_version
      : null;
    const timeoutMs =
      Number.isInteger(options.timeout_ms) && options.timeout_ms > 0
        ? options.timeout_ms
        : DEFAULT_WAIT_TIMEOUT_MS;

    const currentState = this.getCachedState(Number.POSITIVE_INFINITY);
    if (
      isPlainObject(currentState) &&
      this.doesStateMatchWait(currentState, predicate, afterStateVersion)
    ) {
      return currentState;
    }

    return await new Promise((resolve, reject) => {
      const waiter = {
        predicate,
        afterStateVersion,
        resolve: (state) => {
          clearTimeout(waiter.timeoutHandle);
          this.waiters.delete(waiter);
          resolve(state);
        },
        reject: (error) => {
          clearTimeout(waiter.timeoutHandle);
          this.waiters.delete(waiter);
          reject(error);
        },
        timeoutHandle: setTimeout(() => {
          waiter.reject(
            new ToolPayloadError(
              "bridge_event_wait_timeout",
              `Bridge event stream did not produce a matching state within ${timeoutMs} ms.`,
              {
                timeout_ms: timeoutMs,
                after_state_version: afterStateVersion
              }
            )
          );
        }, timeoutMs)
      };

      this.waiters.add(waiter);
    });
  }

  doesStateMatchWait(state, predicate, afterStateVersion) {
    const stateVersion = getStateVersionValue(state);
    if (afterStateVersion !== null && (!Number.isInteger(stateVersion) || stateVersion <= afterStateVersion)) {
      return false;
    }

    return predicate(state) === true;
  }

  resolveWaiters(state) {
    for (const waiter of [...this.waiters]) {
      if (this.doesStateMatchWait(state, waiter.predicate, waiter.afterStateVersion)) {
        waiter.resolve(state);
      }
    }
  }

  async runLoop() {
    let reconnectDelayMs = BRIDGE_EVENT_RECONNECT_DELAY_MS;
    while (!this.closed) {
      try {
        await this.connectOnce();
        reconnectDelayMs = BRIDGE_EVENT_RECONNECT_DELAY_MS;
      } catch (error) {
        if (this.closed || error?.name === "AbortError") {
          return;
        }

        logDebug(
          `bridge event stream reconnect scheduled delay_ms=${reconnectDelayMs} error=${sanitizeForLog(error instanceof Error ? error.message : String(error))}`
        );
        await delay(reconnectDelayMs);
        reconnectDelayMs = Math.min(
          reconnectDelayMs * 2,
          BRIDGE_EVENT_MAX_RECONNECT_DELAY_MS
        );
      }
    }
  }

  async connectOnce() {
    const url = new URL("events", ensureTrailingSlash(this.session.base_url)).toString();
    const response = await fetch(url, {
      method: "GET",
      headers: buildAuthHeaders(this.session),
      signal: this.abortController.signal
    });

    if (!response.ok) {
      const payload = await readJsonResponseBody(response);
      throw new BridgeHttpError(
        response.status,
        payload && typeof payload.error === "string" ? payload.error : "bridge_http_error",
        payload && typeof payload.message === "string"
          ? payload.message
          : `Bridge event stream failed with status ${response.status}.`,
        {
          url,
          method: "GET",
          status: response.status,
          payload
        }
      );
    }

    if (!response.body) {
      throw new ToolPayloadError(
        "bridge_event_stream_missing_body",
        "Bridge event stream response did not include a body.",
        {
          url
        }
      );
    }

    await consumeSseStream(response.body, (event) => {
      if (event.event !== "frontier") {
        return;
      }

      let payload = null;
      try {
        payload = event.data ? JSON.parse(event.data) : null;
      } catch (error) {
        logDebug(
          `bridge event stream payload parse failed error=${sanitizeForLog(error instanceof Error ? error.message : String(error))}`
        );
        return;
      }

      const state = isPlainObject(payload?.state) ? payload.state : null;
      if (state) {
        this.ingestState(state);
      }
    });
  }
}

function ensureBridgeEventClient(session) {
  const sessionKey = getBridgeSessionKey(session);
  if (activeBridgeEventClient && activeBridgeEventClient.sessionKey !== sessionKey) {
    activeBridgeEventClient.close();
    activeBridgeEventClient = null;
  }

  if (!activeBridgeEventClient) {
    activeBridgeEventClient = new BridgeEventClient(session);
  } else {
    activeBridgeEventClient.updateSession(session);
  }

  return activeBridgeEventClient;
}

async function getLatestBridgeState(session, options = {}) {
  const client = ensureBridgeEventClient(session);
  await client.start();

  const forceRefresh = options.force_refresh === true;
  const cachedState = forceRefresh ? null : client.getCachedState();
  if (cachedState) {
    return cachedState;
  }

  const response = await bridgeRequestJson(session, "state", {
    method: "GET"
  });
  return client.ingestState(response.payload) ?? response.payload;
}

async function waitForBridgeStateEvent(session, options = {}) {
  const client = ensureBridgeEventClient(session);
  await client.start();
  return await client.waitForState(options);
}

async function consumeSseStream(stream, onEvent) {
  const reader = stream.getReader();
  const decoder = new TextDecoder();
  let buffer = "";

  while (true) {
    const { value, done } = await reader.read();
    if (done) {
      break;
    }

    buffer += decoder.decode(value, { stream: true });
    buffer = buffer.replace(/\r\n/g, "\n");

    while (true) {
      const delimiterIndex = buffer.indexOf("\n\n");
      if (delimiterIndex < 0) {
        break;
      }

      const rawEvent = buffer.slice(0, delimiterIndex);
      buffer = buffer.slice(delimiterIndex + 2);
      const parsedEvent = parseSseEvent(rawEvent);
      if (parsedEvent) {
        onEvent(parsedEvent);
      }
    }
  }
}

function parseSseEvent(rawEvent) {
  if (typeof rawEvent !== "string" || rawEvent.length <= 0) {
    return null;
  }

  const lines = rawEvent.split("\n");
  let eventName = "message";
  const dataLines = [];

  for (const line of lines) {
    if (!line || line.startsWith(":")) {
      continue;
    }

    if (line.startsWith("event:")) {
      eventName = line.slice("event:".length).trim() || "message";
      continue;
    }

    if (line.startsWith("data:")) {
      dataLines.push(line.slice("data:".length).trimStart());
    }
  }

  return {
    event: eventName,
    data: dataLines.join("\n")
  };
}

async function bridgeRequestJson(session, endpointPath, options) {
  const method = options && options.method ? options.method : "GET";
  const timeoutMs =
    options && Number.isInteger(options.timeoutMs) && options.timeoutMs > 0
      ? options.timeoutMs
      : DEFAULT_HTTP_TIMEOUT_MS;
  const url = new URL(endpointPath, ensureTrailingSlash(session.base_url)).toString();
  const headers = buildAuthHeaders(session);
  const requestOptions = {
    method,
    headers,
    signal: AbortSignal.timeout(timeoutMs)
  };

  if (options && Object.prototype.hasOwnProperty.call(options, "body")) {
    requestOptions.body = JSON.stringify(options.body);
    requestOptions.headers = {
      ...headers,
      "Content-Type": "application/json"
    };
  }

  let response;
  try {
    response = await fetch(url, requestOptions);
  } catch (error) {
    throw new ToolPayloadError(
      "bridge_request_failed",
      error instanceof Error ? error.message : String(error),
      {
        url,
        method,
        timeout_ms: timeoutMs
      }
    );
  }

  const payload = await readJsonResponseBody(response);
  if (!response.ok) {
    throw new BridgeHttpError(
      response.status,
      payload && typeof payload.error === "string" ? payload.error : "bridge_http_error",
      payload && typeof payload.message === "string"
        ? payload.message
        : `Bridge request failed with status ${response.status}.`,
      {
        url,
        method,
        status: response.status,
        payload
      }
    );
  }

  return {
    url,
    status: response.status,
    payload
  };
}

async function readJsonResponseBody(response) {
  const text = await response.text();
  if (!text) {
    return null;
  }

  try {
    return JSON.parse(text);
  } catch (error) {
    throw new ToolPayloadError(
      "invalid_bridge_response",
      "Bridge response was not valid JSON.",
      {
        status: response.status,
        raw_text: text
      }
    );
  }
}

function ensureTrailingSlash(value) {
  return value.endsWith("/") ? value : `${value}/`;
}

function isStateVersionConflictError(error) {
  return error instanceof BridgeHttpError && error.code === "state_version_conflict";
}

function isActionIdCurrentlyAvailable(state, actionId) {
  if (!isPlainObject(state) || typeof actionId !== "string" || !actionId.trim()) {
    return false;
  }

  return Array.isArray(state.available_actions)
    ? state.available_actions.some((action) => action?.action_id === actionId)
    : false;
}

function isProcessAlive(pid) {
  if (!Number.isInteger(pid) || pid <= 0) {
    return false;
  }

  try {
    process.kill(pid, 0);
    return true;
  } catch (error) {
    if (error && typeof error === "object" && error.code === "EPERM") {
      return true;
    }
    return false;
  }
}

function requireNonEmptyString(value, fieldName) {
  if (typeof value !== "string" || value.trim().length === 0) {
    throw new ToolPayloadError(
      "invalid_arguments",
      `${fieldName} must be a non-empty string.`,
      {
        field: fieldName
      }
    );
  }

  return value.trim();
}

function requireNonEmptyStringArray(value, fieldName) {
  if (!Array.isArray(value) || value.length <= 0) {
    throw new ToolPayloadError(
      "invalid_arguments",
      `${fieldName} must be a non-empty array of strings.`,
      {
        field: fieldName
      }
    );
  }

  return value.map((entry, index) => {
    if (typeof entry !== "string" || entry.trim().length <= 0) {
      throw new ToolPayloadError(
        "invalid_arguments",
        `${fieldName}[${index}] must be a non-empty string.`,
        {
          field: fieldName,
          index
        }
      );
    }

    return entry.trim();
  });
}

function optionalString(value, fieldName) {
  if (value === undefined || value === null) {
    return undefined;
  }

  if (typeof value !== "string") {
    throw new ToolPayloadError(
      "invalid_arguments",
      `${fieldName} must be a string when provided.`,
      {
        field: fieldName
      }
    );
  }

  return value;
}

function normalizeMapRoutesDetail(value) {
  const normalized = optionalString(value, "detail") ?? "summary";
  if (normalized !== "summary" && normalized !== "full") {
    throw new ToolPayloadError(
      "invalid_arguments",
      "detail must be either summary or full when provided.",
      {
        field: "detail"
      }
    );
  }

  return normalized;
}

function normalizeIndexedOptionSurface(value) {
  const normalized = optionalString(value, "surface") ?? "auto";
  if (
    normalized === "auto" ||
    normalized === "reward" ||
    normalized === "card_reward" ||
    normalized === "rest_site" ||
    normalized === "deck_upgrade" ||
    normalized === "event" ||
    normalized === "card_selection"
  ) {
    return normalized;
  }

  throw new ToolPayloadError(
    "invalid_arguments",
    "surface must be one of auto, reward, card_reward, rest_site, deck_upgrade, event, or card_selection.",
    {
      field: "surface"
    }
  );
}

function optionalBoolean(value, fieldName) {
  if (value === undefined || value === null) {
    return undefined;
  }

  if (typeof value !== "boolean") {
    throw new ToolPayloadError(
      "invalid_arguments",
      `${fieldName} must be a boolean when provided.`,
      {
        field: fieldName
      }
    );
  }

  return value;
}

function optionalInteger(value, fieldName) {
  if (value === undefined || value === null) {
    return undefined;
  }

  if (!Number.isInteger(value)) {
    throw new ToolPayloadError(
      "invalid_arguments",
      `${fieldName} must be an integer when provided.`,
      {
        field: fieldName
      }
    );
  }

  return value;
}

function optionalIntegerArray(value, fieldName) {
  if (value === undefined || value === null) {
    return undefined;
  }

  if (!Array.isArray(value)) {
    throw new ToolPayloadError(
      "invalid_arguments",
      `${fieldName} must be an array of integers when provided.`,
      {
        field: fieldName
      }
    );
  }

  return value.map((entry, index) => {
    if (!Number.isInteger(entry)) {
      throw new ToolPayloadError(
        "invalid_arguments",
        `${fieldName}[${index}] must be an integer.`,
        {
          field: fieldName,
          index
        }
      );
    }

    if (entry < 0) {
      throw new ToolPayloadError(
        "invalid_arguments",
        `${fieldName}[${index}] must be 0 or greater.`,
        {
          field: fieldName,
          index
        }
      );
    }

    return entry;
  });
}

function normalizeCardSelectionTerminalAction(value) {
  const normalized = optionalString(value, "terminal_action") ?? "confirm";
  if (
    normalized !== "confirm" &&
    normalized !== "cancel" &&
    normalized !== "skip" &&
    normalized !== "close" &&
    normalized !== "none"
  ) {
    throw new ToolPayloadError(
      "invalid_arguments",
      "terminal_action must be one of confirm, cancel, skip, close, or none.",
      {
        field: "terminal_action"
      }
    );
  }

  return normalized;
}

function clampInteger(value, defaultValue, minValue, maxValue, fieldName) {
  if (value === undefined || value === null) {
    return defaultValue;
  }

  if (!Number.isInteger(value)) {
    throw new ToolPayloadError(
      "invalid_arguments",
      `${fieldName} must be an integer.`,
      {
        field: fieldName
      }
    );
  }

  if (value < minValue || value > maxValue) {
    throw new ToolPayloadError(
      "invalid_arguments",
      `${fieldName} must be between ${minValue} and ${maxValue}.`,
      {
        field: fieldName,
        min: minValue,
        max: maxValue
      }
    );
  }

  return value;
}

function isPlainObject(value) {
  return !!value && typeof value === "object" && !Array.isArray(value);
}

function toolErrorPayload(error) {
  if (error instanceof BridgeHttpError) {
    const payload = error.details && error.details.payload;
    if (payload && typeof payload === "object" && !Array.isArray(payload)) {
      return summarizeBridgeErrorPayload(payload);
    }

    return {
      ok: false,
      error: error.code,
      message: error.message,
      details: error.details
    };
  }

  if (error instanceof ToolPayloadError) {
    return {
      ok: false,
      error: error.code,
      message: error.message,
      details: error.details
    };
  }

  return {
    ok: false,
    error: "internal_error",
    message: error instanceof Error ? error.message : String(error)
  };
}

function summarizeBridgeErrorPayload(payload) {
  if (!isPlainObject(payload)) {
    return payload;
  }

  const result = compactPayloadForOutput(payload);
  if (!isPlainObject(result)) {
    return payload;
  }

  if (Array.isArray(payload.available_actions)) {
    result.available_actions = summarizeActionsForAgent(
      filterNonAutomationActions(payload.available_actions),
      typeof payload?.current_state?.screen === "string"
        ? payload.current_state.screen
        : typeof payload?.state_after?.screen === "string"
          ? payload.state_after.screen
          : typeof payload?.state?.screen === "string"
            ? payload.state.screen
            : null
    );
  }

  if (isPlainObject(payload.current_state)) {
    result.current_state = summarizeActionStateForAgent(payload.current_state);
  }

  if (isPlainObject(payload.state_after)) {
    result.state_after = summarizeActionStateForAgent(payload.state_after);
  }

  if (isPlainObject(payload.state)) {
    result.state = summarizeActionStateForAgent(payload.state);
  }

  if (isPlainObject(payload.current_reward_bundle)) {
    result.current_reward_bundle = summarizeRewardBundleForAgent(payload.current_reward_bundle);
  }

  if (isPlainObject(payload.current_card_selection_bundle)) {
    result.current_card_selection_bundle = summarizeCardSelectionBundleForAgent(
      payload.current_card_selection_bundle
    );
  }

  if (isPlainObject(result.current_state)) {
    result.current_state = removeDuplicatedStateSections(result.current_state, {
      reward:
        isPlainObject(result.current_reward_bundle) || isPlainObject(result.reward_bundle),
      rest_site: false,
      card_selection:
        isPlainObject(result.current_card_selection_bundle) ||
        isPlainObject(result.card_selection_bundle),
      shop: false,
      available_actions:
        isPlainObject(result.current_reward_bundle) ||
        isPlainObject(result.reward_bundle) ||
        isPlainObject(result.current_card_selection_bundle) ||
        isPlainObject(result.card_selection_bundle)
    });
  }

  dedupeCompactedPayloadSections(result);

  return result;
}

function attachInteractionHints(payload) {
  if (!isPlainObject(payload)) {
    return payload;
  }

  const state = isPlainObject(payload.state)
    ? payload.state
    : isBridgeStatePayload(payload)
      ? payload
      : null;
  if (!isPlainObject(state)) {
    return payload;
  }

  const hints = {};
  const playableCardActions = getPlayableCardActions(state);
  const usablePotionActions = getUsablePotionActions(state);
  const endTurnAvailable = getNonAutomationActions(state).some(
    (action) => action?.action_id === "end_turn"
  );
  if (playableCardActions.length >= 2) {
    hints.play_card_sequence = {
      play_card_action_count: playableCardActions.length,
      recommended_tool: "sts2_play_card_sequence",
      avoid_parallel_perform_action: true
    };
  }

  if (playableCardActions.length >= 2 || usablePotionActions.length > 0) {
    hints.combat_sequence = {
      play_card_action_count: playableCardActions.length,
      use_potion_action_count: usablePotionActions.length,
      end_turn_available: endTurnAvailable,
      recommended_tool: "sts2_execute_combat_sequence",
      avoid_parallel_perform_action: true
    };
  }

  const cardSelectionBundle = buildCardSelectionBundle(state);
  if (cardSelectionBundle.in_card_selection_flow) {
    hints.card_selection = {
      selected_count: cardSelectionBundle.card_selection.selected_count,
      min_select: cardSelectionBundle.card_selection.min_select,
      max_select: cardSelectionBundle.card_selection.max_select,
      option_count: cardSelectionBundle.card_selection.options.length,
      recommended_tool: "sts2_resolve_card_selection"
    };
  }

  return Object.keys(hints).length > 0
    ? {
      ...payload,
      interaction_hints: hints
    }
    : payload;
}

function compactPayloadForMinimalProfile(value) {
  if (value === null || value === undefined) {
    return undefined;
  }

  if (Array.isArray(value)) {
    const items = value
      .map((entry) => compactPayloadForMinimalProfile(entry))
      .filter((entry) => entry !== undefined);
    return items.length > 0 ? items : undefined;
  }

  if (isPlainObject(value)) {
    const result = {};
    for (const [key, entry] of Object.entries(value)) {
      const compactedEntry = compactPayloadForMinimalProfile(entry);
      if (compactedEntry === undefined) {
        continue;
      }
      result[key] = compactedEntry;
    }
    return Object.keys(result).length > 0 ? result : undefined;
  }

  if (typeof value === "string" && value.length <= 0) {
    return undefined;
  }

  return value;
}

function asToolResult(payload, isError) {
  const compactedPayload = isError ? payload : compactPayloadForOutput(payload);
  const finalPayload =
    !isError && ACTIVE_TOOL_PROFILE_NAME === "minimal"
      ? compactPayloadForMinimalProfile(compactedPayload) ?? {}
      : compactedPayload;
  return asPrecompactedToolResult(finalPayload, isError);
}

function asPrecompactedToolResult(payload, isError) {
  return {
    isError,
    content: [
      {
        type: "text",
        text: JSON.stringify(payload)
      }
    ]
  };
}

function writeResult(id, result) {
  writeMessage({
    jsonrpc: "2.0",
    id,
    result
  });
}

function writeError(id, code, message) {
  writeMessage({
    jsonrpc: "2.0",
    id,
    error: {
      code,
      message
    }
  });
}

function writeMessage(message) {
  logDebug(`stdout_message=${summarizeMessage(message)}`);
  process.stdout.write(`${JSON.stringify(message)}\n`);
}

function summarizeMessage(message) {
  const pieces = [];

  if (message && typeof message === "object") {
    if (Object.prototype.hasOwnProperty.call(message, "id")) {
      pieces.push(`id=${safeJson(message.id)}`);
    }

    if (message.result) {
      if (message.result.serverInfo && message.result.serverInfo.name) {
        pieces.push(`result=initialize:${message.result.serverInfo.name}`);
      } else if (Array.isArray(message.result.tools)) {
        pieces.push(`result=tools/list:${message.result.tools.length}`);
      } else {
        pieces.push("result=object");
      }
    }

    if (message.error) {
      pieces.push(`error=${safeJson(message.error.message || message.error.code)}`);
    }
  }

  return pieces.join(" ");
}

function logInfo(message) {
  writeLogLine("info", message);
  process.stderr.write(`[sts2-mcp] ${message}\n`);
}

function logDebug(message) {
  writeLogLine("debug", message);
}

function logError(message, error) {
  const detail = error instanceof Error ? error.stack || error.message : String(error);
  writeLogLine("error", `${message}: ${detail}`);
  process.stderr.write(`[sts2-mcp] ${message}: ${detail}\n`);
}

function writeLogLine(level, message) {
  try {
    const logFilePath = getLogFilePath();
    fs.mkdirSync(path.dirname(logFilePath), { recursive: true });
    const line = `${new Date().toISOString()} ${level.toUpperCase()} ${message}\n`;
    fs.appendFileSync(logFilePath, line, "utf8");
  } catch (error) {
    const detail = error instanceof Error ? error.message : String(error);
    process.stderr.write(`[sts2-mcp] log write failed: ${detail}\n`);
  }
}

function sanitizeForLog(value) {
  return value.replace(/\r/g, "\\r").replace(/\n/g, "\\n").slice(0, MAX_LOG_PREVIEW);
}

function safeJson(value) {
  try {
    return JSON.stringify(value);
  } catch (error) {
    return JSON.stringify(String(value));
  }
}

// ── Journal & Knowledge Tools ──────────────────────────────────────────────

const JOURNAL_ENTRY_START_MARKER = "<!-- sts2-journal-entry";
const JOURNAL_ENTRY_END_MARKER = "<!-- /sts2-journal-entry -->";
const JOURNAL_FILE_NAME_PATTERN = /^[A-Za-z0-9._-]+\.md$/;

function getJournalDir() {
  const appData = process.env.APPDATA || path.join(os.homedir(), "AppData", "Roaming");
  return path.join(appData, "SlayTheSpire2", "bridge", "journal");
}

function getMetaFilePath() {
  return path.join(getJournalDir(), "meta.json");
}

function createEmptyJournalMeta() {
  return {
    active_run: null,
    active_run_closed: false,
    summary: "",
    character: "",
    runs: []
  };
}

function normalizeJournalFileName(value) {
  if (typeof value !== "string") {
    return null;
  }

  const trimmed = value.trim();
  if (!JOURNAL_FILE_NAME_PATTERN.test(trimmed)) {
    return null;
  }

  return trimmed;
}

function normalizeJournalRunEntry(entry) {
  if (!isPlainObject(entry)) {
    return null;
  }

  const file = normalizeJournalFileName(entry.file);
  if (file === null) {
    return null;
  }

  return {
    file,
    character:
      typeof entry.character === "string" && entry.character.trim()
        ? entry.character.trim()
        : "Unknown",
    result:
      entry.result === "death" || entry.result === "victory" || entry.result === "in_progress"
        ? entry.result
        : "in_progress",
    floor: Number.isInteger(entry.floor) && entry.floor >= 0 ? entry.floor : 0,
    summary: typeof entry.summary === "string" ? entry.summary : "",
    started_at: typeof entry.started_at === "string" ? entry.started_at : null,
    updated_at: typeof entry.updated_at === "string" ? entry.updated_at : null
  };
}

function normalizeJournalMeta(meta) {
  const normalized = createEmptyJournalMeta();
  if (!isPlainObject(meta)) {
    return normalized;
  }

  normalized.active_run = normalizeJournalFileName(meta.active_run);
  normalized.active_run_closed = meta.active_run_closed === true;
  normalized.summary = typeof meta.summary === "string" ? meta.summary : "";
  normalized.character = typeof meta.character === "string" ? meta.character : "";
  normalized.runs = Array.isArray(meta.runs)
    ? meta.runs
      .map((entry) => normalizeJournalRunEntry(entry))
      .filter((entry) => entry !== null)
    : [];

  return normalized;
}

function readMeta() {
  const metaPath = getMetaFilePath();
  if (!fs.existsSync(metaPath)) {
    return createEmptyJournalMeta();
  }

  try {
    return normalizeJournalMeta(JSON.parse(fs.readFileSync(metaPath, "utf8")));
  } catch (error) {
    logError("failed to read journal meta.json", error);
    return createEmptyJournalMeta();
  }
}

function writeMeta(meta) {
  const journalDir = getJournalDir();
  fs.mkdirSync(journalDir, { recursive: true });
  fs.writeFileSync(
    getMetaFilePath(),
    JSON.stringify(normalizeJournalMeta(meta), null, 2),
    "utf8"
  );
}

function resolveJournalFilePath(fileName) {
  const normalizedFileName = normalizeJournalFileName(fileName);
  if (normalizedFileName === null) {
    throw new ToolPayloadError(
      "invalid_journal_file",
      "Journal metadata references an invalid active_run file name.",
      {
        active_run: fileName
      }
    );
  }

  return path.join(getJournalDir(), normalizedFileName);
}

async function tryGetCurrentRunContext() {
  try {
    const session = getLiveSession();
    const state = await getBridgeState(session);
    const player = isPlainObject(state?.players?.[0]) ? state.players[0] : null;
    const run = isPlainObject(state?.run) ? state.run : null;

    return {
      state,
      character: normalizeAgentText(player?.character?.title) ?? null,
      total_floor: Number.isInteger(run?.total_floor) ? run.total_floor : null,
      act_floor: Number.isInteger(run?.act_floor) ? run.act_floor : null,
      act: normalizeAgentText(run?.act?.title) ?? null,
      is_game_over: run?.is_game_over === true
    };
  } catch (error) {
    return null;
  }
}

function upsertJournalRunEntry(meta, fileName, fields = {}) {
  const normalizedFileName = normalizeJournalFileName(fileName);
  if (normalizedFileName === null) {
    return null;
  }

  let runEntry = meta.runs.find((entry) => entry.file === normalizedFileName) ?? null;
  if (runEntry === null) {
    runEntry = normalizeJournalRunEntry({
      file: normalizedFileName,
      character: "Unknown",
      result: "in_progress",
      floor: 0,
      summary: "",
      started_at: fields.started_at ?? new Date().toISOString(),
      updated_at: fields.updated_at ?? new Date().toISOString()
    });
    meta.runs.push(runEntry);
  }

  if (typeof fields.character === "string" && fields.character.trim()) {
    runEntry.character = fields.character.trim();
  }
  if (typeof fields.result === "string" && fields.result.trim()) {
    runEntry.result = fields.result;
  }
  if (Number.isInteger(fields.floor) && fields.floor >= 0) {
    runEntry.floor = fields.floor;
  }
  if (typeof fields.summary === "string") {
    runEntry.summary = fields.summary;
  }
  if (typeof fields.started_at === "string" && fields.started_at.trim()) {
    runEntry.started_at = fields.started_at;
  }
  if (typeof fields.updated_at === "string" && fields.updated_at.trim()) {
    runEntry.updated_at = fields.updated_at;
  }

  return runEntry;
}

function buildRunJournalPreamble(runContext, startedAt) {
  const lines = [
    "# Run Journal",
    "",
    `Started: ${startedAt}`
  ];

  if (typeof runContext?.character === "string" && runContext.character.trim()) {
    lines.push(`Character: ${runContext.character.trim()}`);
  }
  if (Number.isInteger(runContext?.total_floor) && runContext.total_floor >= 0) {
    lines.push(`Start Floor: ${runContext.total_floor}`);
  }
  if (typeof runContext?.act === "string" && runContext.act.trim()) {
    lines.push(`Act: ${runContext.act.trim()}`);
  }

  lines.push("", "");
  return lines.join("\n");
}

async function prepareActiveJournal(options = {}) {
  const allowCreate = options.allowCreate !== false;
  const createWithoutRunContext = options.createWithoutRunContext === true;
  const meta = readMeta();
  const runContext =
    options.runContext !== undefined ? options.runContext : await tryGetCurrentRunContext();
  const nowIso = new Date().toISOString();

  let filePath = null;
  if (meta.active_run !== null) {
    try {
      filePath = resolveJournalFilePath(meta.active_run);
    } catch (error) {
      meta.active_run = null;
      filePath = null;
    }
  }

  const shouldRotateClosedRun =
    meta.active_run_closed === true &&
    runContext !== null &&
    runContext.is_game_over !== true;
  const activeRunMissing = filePath === null || !fs.existsSync(filePath);
  const shouldCreateRun =
    allowCreate &&
    (activeRunMissing || shouldRotateClosedRun) &&
    (runContext !== null || createWithoutRunContext);

  if (shouldCreateRun) {
    const filename = `run_${nowIso.replace(/[:.]/g, "-").slice(0, 19)}.md`;
    filePath = resolveJournalFilePath(filename);
    fs.mkdirSync(getJournalDir(), { recursive: true });
    fs.writeFileSync(filePath, buildRunJournalPreamble(runContext, nowIso), "utf8");
    meta.active_run = filename;
    meta.active_run_closed = false;
    meta.summary = "";
    meta.character =
      typeof runContext?.character === "string" && runContext.character.trim()
        ? runContext.character.trim()
        : "";
    upsertJournalRunEntry(meta, filename, {
      character: meta.character || "Unknown",
      result: "in_progress",
      floor:
        Number.isInteger(runContext?.total_floor) && runContext.total_floor >= 0
          ? runContext.total_floor
          : 0,
      summary: "",
      started_at: nowIso,
      updated_at: nowIso
    });
    writeMeta(meta);
    return {
      meta,
      filePath,
      runContext,
      created: true
    };
  }

  if (meta.active_run !== null && filePath !== null) {
    let changed = false;
    const normalizedCharacter =
      typeof runContext?.character === "string" && runContext.character.trim()
        ? runContext.character.trim()
        : null;
    const existingRunEntry =
      meta.runs.find((entry) => entry.file === meta.active_run) ?? null;
    const previousFloor = Number.isInteger(existingRunEntry?.floor) ? existingRunEntry.floor : null;

    if (normalizedCharacter !== null && meta.character !== normalizedCharacter) {
      meta.character = normalizedCharacter;
      changed = true;
    }

    const runEntry = upsertJournalRunEntry(meta, meta.active_run, {
      character: normalizedCharacter ?? undefined,
      floor:
        Number.isInteger(runContext?.total_floor) && runContext.total_floor >= 0
          ? runContext.total_floor
          : undefined,
      updated_at: nowIso
    });
    if (existingRunEntry === null) {
      changed = true;
    } else if (
      runEntry !== null &&
      Number.isInteger(runContext?.total_floor) &&
      runContext.total_floor !== previousFloor
    ) {
      changed = true;
    }

    if (meta.active_run_closed === true && runContext !== null && runContext.is_game_over === true) {
      // Keep the current closed run selected while still allowing post-mortem notes.
      changed = changed || false;
    }

    if (changed) {
      writeMeta(meta);
    }
  }

  return {
    meta,
    filePath,
    runContext,
    created: false
  };
}

function normalizeJournalTags(tags) {
  return Array.isArray(tags)
    ? tags
      .filter((tag) => typeof tag === "string" && tag.trim().length > 0)
      .map((tag) => tag.trim())
    : [];
}

function buildJournalEntrySection(entry, options = {}) {
  const normalizedTags = normalizeJournalTags(options.tags);
  const floor = Number.isInteger(options.floor) && options.floor >= 0 ? options.floor : null;
  const timestamp =
    typeof options.timestamp === "string" && options.timestamp.trim()
      ? options.timestamp.trim()
      : new Date().toISOString().slice(0, 16).replace("T", " ");
  const metadata = JSON.stringify({
    floor,
    tags: normalizedTags,
    timestamp
  });
  const floorLabel = floor !== null ? `F${floor}` : "F?";
  const tagLabel = normalizedTags.length > 0 ? ` [${normalizedTags.join(", ")}]` : "";

  return `\n${JOURNAL_ENTRY_START_MARKER} ${metadata} -->\n## ${floorLabel}${tagLabel} — ${timestamp}\n\n${entry}\n\n${JOURNAL_ENTRY_END_MARKER}\n`;
}

function parseLegacyJournalEntries(content) {
  const normalizedContent = typeof content === "string" ? content.replace(/\r\n/g, "\n") : "";
  const entries = [];
  const entryRegex = /^## [\s\S]*?(?=^## |\Z)/gm;
  let match;
  while ((match = entryRegex.exec(normalizedContent)) !== null) {
    const raw = match[0];
    const firstLine = raw.split("\n", 1)[0] ?? "";
    const headerMatch = firstLine.match(/^##\s+(F\??\d*)(?:\s+\[([^\]]+)\])?(?:\s+—\s+(.+))?$/);
    const rawFloorLabel = headerMatch?.[1] ?? null;
    const floor =
      typeof rawFloorLabel === "string" && /^F\d+$/.test(rawFloorLabel)
        ? Number.parseInt(rawFloorLabel.slice(1), 10)
        : null;
    const tags =
      typeof headerMatch?.[2] === "string"
        ? headerMatch[2]
          .split(",")
          .map((tag) => tag.trim())
          .filter((tag) => tag.length > 0)
        : [];
    const timestamp = typeof headerMatch?.[3] === "string" ? headerMatch[3].trim() : null;

    entries.push({
      raw,
      floor,
      tags,
      timestamp
    });
  }
  return entries;
}

function parseJournalEntries(content) {
  const normalizedContent = typeof content === "string" ? content.replace(/\r\n/g, "\n") : "";
  const entries = [];
  const markerRegex =
    /<!-- sts2-journal-entry (\{[^\n]*\}) -->\n([\s\S]*?)\n<!-- \/sts2-journal-entry -->/g;
  let match;

  while ((match = markerRegex.exec(normalizedContent)) !== null) {
    let metadata = {};
    try {
      metadata = JSON.parse(match[1]);
    } catch (error) {
      metadata = {};
    }

    entries.push({
      raw: match[0],
      floor:
        Number.isInteger(metadata.floor) && metadata.floor >= 0 ? metadata.floor : null,
      tags: normalizeJournalTags(metadata.tags),
      timestamp:
        typeof metadata.timestamp === "string" && metadata.timestamp.trim()
          ? metadata.timestamp.trim()
          : null
    });
  }

  return entries.length > 0 ? entries : parseLegacyJournalEntries(content);
}

function filterJournalEntries(entries, options = {}) {
  const requestedTags = normalizeJournalTags(options.tags).map((tag) => tag.toLowerCase());
  const lastN = Number.isInteger(options.last_n) && options.last_n > 0 ? options.last_n : 0;

  let filteredEntries = Array.isArray(entries) ? entries : [];
  if (requestedTags.length > 0) {
    filteredEntries = filteredEntries.filter((entry) => {
      const entryTags = normalizeJournalTags(entry?.tags).map((tag) => tag.toLowerCase());
      return requestedTags.some((tag) => entryTags.includes(tag));
    });
  }

  if (lastN > 0 && filteredEntries.length > lastN) {
    filteredEntries = filteredEntries.slice(-lastN);
  }

  return filteredEntries;
}

async function journalWriteTool(args) {
  try {
    const entry = requireNonEmptyString(args.entry, "entry");
    const tags = normalizeJournalTags(args.tags);
    const providedFloor = typeof args.floor === "number" ? args.floor : null;
    const prepared = await prepareActiveJournal();
    if (prepared.meta.active_run === null || prepared.filePath === null) {
      throw new ToolPayloadError(
        "journal_unavailable",
        "Unable to prepare an active journal file."
      );
    }

    const floor =
      providedFloor !== null
        ? providedFloor
        : Number.isInteger(prepared.runContext?.total_floor)
          ? prepared.runContext.total_floor
          : null;
    const section = buildJournalEntrySection(entry, {
      tags,
      floor
    });
    fs.appendFileSync(prepared.filePath, section, "utf8");

    upsertJournalRunEntry(prepared.meta, prepared.meta.active_run, {
      character: prepared.meta.character || "Unknown",
      floor: floor ?? undefined,
      updated_at: new Date().toISOString()
    });
    writeMeta(prepared.meta);

    const floorLabel = floor !== null ? `F${floor}` : "F?";
    logInfo(`journal_write floor=${floorLabel} tags=${tags.join(",")}`);
    return asToolResult(
      {
        ok: true,
        file: prepared.meta.active_run,
        floor: floorLabel,
        tags
      },
      false
    );
  } catch (error) {
    return asToolResult(toolErrorPayload(error), true);
  }
}

async function journalReadTool(args) {
  try {
    const prepared = await prepareActiveJournal({
      allowCreate: true
    });
    if (prepared.meta.active_run === null || prepared.filePath === null) {
      return asToolResult(
        {
          ok: true,
          content: "",
          message: "No active run journal found."
        },
        false
      );
    }

    if (!fs.existsSync(prepared.filePath)) {
      return asToolResult(
        {
          ok: true,
          content: "",
          message: "Run journal file not found."
        },
        false
      );
    }

    const content = fs.readFileSync(prepared.filePath, "utf8");
    const lastN = typeof args.last_n === "number" ? args.last_n : 0;
    const filterTags = Array.isArray(args.tags) ? args.tags : [];

    if (lastN <= 0 && filterTags.length === 0) {
      return asToolResult(
        {
          ok: true,
          file: prepared.meta.active_run,
          content
        },
        false
      );
    }

    const entries = parseJournalEntries(content);
    const filteredEntries = filterJournalEntries(entries, {
      last_n: lastN,
      tags: filterTags
    });

    return asToolResult(
      {
        ok: true,
        file: prepared.meta.active_run,
        entry_count: filteredEntries.length,
        content: filteredEntries.map((entry) => entry.raw).join("\n")
      },
      false
    );
  } catch (error) {
    return asToolResult(toolErrorPayload(error), true);
  }
}

async function journalSummarizeTool(args) {
  try {
    const summary = requireNonEmptyString(args.summary, "summary");
    const result =
      args.result === "death" || args.result === "victory" || args.result === "in_progress"
        ? args.result
        : "in_progress";
    const runContext = await tryGetCurrentRunContext();
    const floor =
      typeof args.floor === "number"
        ? args.floor
        : Number.isInteger(runContext?.total_floor)
          ? runContext.total_floor
          : null;
    const prepared = await prepareActiveJournal({
      allowCreate: true,
      createWithoutRunContext: true,
      runContext
    });

    if (prepared.meta.active_run === null) {
      throw new ToolPayloadError(
        "journal_unavailable",
        "Unable to prepare an active journal file."
      );
    }

    prepared.meta.summary = summary;
    if (typeof runContext?.character === "string" && runContext.character.trim()) {
      prepared.meta.character = runContext.character.trim();
    }
    prepared.meta.active_run_closed = result === "death" || result === "victory";

    upsertJournalRunEntry(prepared.meta, prepared.meta.active_run, {
      character: prepared.meta.character || "Unknown",
      result,
      floor: floor ?? undefined,
      summary,
      updated_at: new Date().toISOString()
    });

    writeMeta(prepared.meta);
    logInfo(`journal_summarize result=${result} floor=${floor}`);
    return asToolResult(
      {
        ok: true,
        summary,
        result,
        floor
      },
      false
    );
  } catch (error) {
    return asToolResult(toolErrorPayload(error), true);
  }
}

async function journalGetSummaryTool(args) {
  try {
    const prepared = await prepareActiveJournal({
      allowCreate: true
    });
    return asToolResult(
      {
        ok: true,
        active_run: prepared.meta.active_run,
        active_run_closed: prepared.meta.active_run_closed === true,
        summary: prepared.meta.summary || "(no summary yet)",
        character: prepared.meta.character || "Unknown",
        total_runs: prepared.meta.runs.length
      },
      false
    );
  } catch (error) {
    return asToolResult(toolErrorPayload(error), true);
  }
}

async function journalListRunsTool(args) {
  try {
    const meta = readMeta();
    let runs = meta.runs || [];
    const lastN = typeof args.last_n === "number" ? args.last_n : 0;
    if (lastN > 0 && runs.length > lastN) {
      runs = runs.slice(-lastN);
    }
    return asToolResult(
      {
        ok: true,
        active_run: meta.active_run,
        active_run_closed: meta.active_run_closed === true,
        total_runs: meta.runs.length,
        runs
      },
      false
    );
  } catch (error) {
    return asToolResult(toolErrorPayload(error), true);
  }
}

const OBSERVATION_ENTRY_START_MARKER = "<!-- sts2-observation-entry";
const OBSERVATION_ENTRY_END_MARKER = "<!-- /sts2-observation-entry -->";

function getKnowledgeObservationDir() {
  if (process.env.STS2_KNOWLEDGE_OBSERVATION_DIR) {
    return path.resolve(process.env.STS2_KNOWLEDGE_OBSERVATION_DIR);
  }

  const appData = process.env.APPDATA || path.join(os.homedir(), "AppData", "Roaming");
  return path.join(appData, "SlayTheSpire2", "bridge", "knowledge-observations");
}

function normalizeObservationDomain(value, fieldName = "domain") {
  const domain = requireNonEmptyString(value, fieldName);
  if (!OBSERVATION_DOMAIN_ENUM.includes(domain)) {
    throw new ToolPayloadError(
      "invalid_arguments",
      `${fieldName} must be one of: ${OBSERVATION_DOMAIN_ENUM.join(", ")}`,
      {
        field: fieldName,
        valid_domains: OBSERVATION_DOMAIN_ENUM
      }
    );
  }

  return domain;
}

function normalizeObservationSourceType(value, fieldName = "source_type") {
  const sourceType = optionalString(value, fieldName) ?? "observed";
  const validSourceTypes = ["observed", "journaled", "inferred", "external"];
  if (!validSourceTypes.includes(sourceType)) {
    throw new ToolPayloadError(
      "invalid_arguments",
      `${fieldName} must be one of: ${validSourceTypes.join(", ")}`,
      {
        field: fieldName,
        valid_source_types: validSourceTypes
      }
    );
  }

  return sourceType;
}

function normalizeObservationConfidence(value, fieldName = "confidence") {
  const confidence = optionalString(value, fieldName) ?? "medium";
  const validConfidenceValues = ["low", "medium", "high"];
  if (!validConfidenceValues.includes(confidence)) {
    throw new ToolPayloadError(
      "invalid_arguments",
      `${fieldName} must be one of: ${validConfidenceValues.join(", ")}`,
      {
        field: fieldName,
        valid_confidence_values: validConfidenceValues
      }
    );
  }

  return confidence;
}

function sanitizeObservationFileSegment(value) {
  const normalized = requireNonEmptyString(value, "entity_name")
    .replace(/[\r\n\t]+/g, " ")
    .replace(/[<>:"/\\|?*\u0000-\u001F]/g, "_")
    .replace(/\s+/g, " ")
    .trim()
    .replace(/[. ]+$/g, "")
    .slice(0, 80);

  return normalized || "entity";
}

function getObservationDomainDir(domain) {
  return path.join(getKnowledgeObservationDir(), domain);
}

function buildObservationPreamble(domain, entityName, createdAt) {
  return [
    `# Observation: ${entityName}`,
    "",
    `Domain: ${domain}`,
    `Entity: ${entityName}`,
    `Created: ${createdAt}`,
    "",
    "> Observation logs are evidence-first working notes. Promote only verified conclusions into canonical knowledge.",
    "",
    ""
  ].join("\n");
}

function buildObservationEntrySection(observation, options = {}) {
  const timestamp =
    typeof options.timestamp === "string" && options.timestamp.trim()
      ? options.timestamp.trim()
      : new Date().toISOString();
  const metadata = {
    timestamp,
    source_type: normalizeObservationSourceType(options.source_type),
    confidence: normalizeObservationConfidence(options.confidence),
    source_tool: optionalString(options.source_tool, "source_tool") ?? null,
    source_ref: optionalString(options.source_ref, "source_ref") ?? null,
    state_version: Number.isInteger(options.state_version) ? options.state_version : null,
    tags: normalizeJournalTags(options.tags)
  };
  const headingParts = [timestamp];
  if (metadata.source_type) {
    headingParts.push(metadata.source_type);
  }
  if (metadata.confidence) {
    headingParts.push(metadata.confidence);
  }

  return `\n${OBSERVATION_ENTRY_START_MARKER} ${JSON.stringify(metadata)} -->\n## ${headingParts.join(" | ")}\n\n${observation}\n\n${OBSERVATION_ENTRY_END_MARKER}\n`;
}

function parseObservationHeader(content) {
  const normalizedContent = typeof content === "string" ? content.replace(/\r\n/g, "\n") : "";
  const lines = normalizedContent.split("\n");
  let entityName = null;
  let domain = null;
  let createdAt = null;

  for (const line of lines.slice(0, 12)) {
    if (entityName === null) {
      const headingMatch = line.match(/^# Observation:\s+(.+?)\s*$/);
      if (headingMatch) {
        entityName = headingMatch[1].trim();
        continue;
      }
    }
    if (domain === null) {
      const domainMatch = line.match(/^Domain:\s+(.+?)\s*$/);
      if (domainMatch) {
        domain = domainMatch[1].trim();
        continue;
      }
    }
    if (createdAt === null) {
      const createdMatch = line.match(/^Created:\s+(.+?)\s*$/);
      if (createdMatch) {
        createdAt = createdMatch[1].trim();
      }
    }
  }

  return {
    domain,
    entity_name: entityName,
    created_at: createdAt
  };
}

function parseObservationEntries(content) {
  const normalizedContent = typeof content === "string" ? content.replace(/\r\n/g, "\n") : "";
  const entries = [];
  const markerRegex =
    /<!-- sts2-observation-entry (\{[^\n]*\}) -->\n([\s\S]*?)\n<!-- \/sts2-observation-entry -->/g;
  let match;

  while ((match = markerRegex.exec(normalizedContent)) !== null) {
    let metadata = {};
    try {
      metadata = JSON.parse(match[1]);
    } catch (error) {
      metadata = {};
    }

    entries.push({
      raw: match[0],
      body: match[2],
      timestamp:
        typeof metadata.timestamp === "string" && metadata.timestamp.trim()
          ? metadata.timestamp.trim()
          : null,
      source_type:
        typeof metadata.source_type === "string" && metadata.source_type.trim()
          ? metadata.source_type.trim()
          : null,
      confidence:
        typeof metadata.confidence === "string" && metadata.confidence.trim()
          ? metadata.confidence.trim()
          : null,
      source_tool:
        typeof metadata.source_tool === "string" && metadata.source_tool.trim()
          ? metadata.source_tool.trim()
          : null,
      source_ref:
        typeof metadata.source_ref === "string" && metadata.source_ref.trim()
          ? metadata.source_ref.trim()
          : null,
      state_version: Number.isInteger(metadata.state_version) ? metadata.state_version : null,
      tags: normalizeJournalTags(metadata.tags)
    });
  }

  return entries;
}

function findObservationEntityFilePath(domain, entityName) {
  const directPath = path.join(getObservationDomainDir(domain), `${sanitizeObservationFileSegment(entityName)}.md`);
  if (fs.existsSync(directPath)) {
    return directPath;
  }

  const domainDir = getObservationDomainDir(domain);
  if (!fs.existsSync(domainDir)) {
    return null;
  }

  const requestedKey = normalizeTextComparisonKey(entityName);
  if (requestedKey === null) {
    return null;
  }

  for (const entry of fs.readdirSync(domainDir, { withFileTypes: true })) {
    if (!entry.isFile() || path.extname(entry.name).toLowerCase() !== ".md") {
      continue;
    }

    const filePath = path.join(domainDir, entry.name);
    const header = parseObservationHeader(fs.readFileSync(filePath, "utf8"));
    if (normalizeTextComparisonKey(header.entity_name) === requestedKey) {
      return filePath;
    }
  }

  return null;
}

function ensureObservationEntityDocument(domain, entityName) {
  const filePath =
    findObservationEntityFilePath(domain, entityName) ??
    path.join(getObservationDomainDir(domain), `${sanitizeObservationFileSegment(entityName)}.md`);
  const exists = fs.existsSync(filePath);

  if (!exists) {
    fs.mkdirSync(path.dirname(filePath), { recursive: true });
    fs.writeFileSync(
      filePath,
      buildObservationPreamble(domain, entityName, new Date().toISOString()),
      "utf8"
    );
  }

  const content = fs.readFileSync(filePath, "utf8");
  const header = parseObservationHeader(content);
  const entries = parseObservationEntries(content);
  return {
    file_path: filePath,
    content,
    header,
    entries
  };
}

function readObservationEntityDocument(domain, entityName) {
  const filePath = findObservationEntityFilePath(domain, entityName);
  if (filePath === null || !fs.existsSync(filePath)) {
    return null;
  }

  const content = fs.readFileSync(filePath, "utf8");
  const header = parseObservationHeader(content);
  const entries = parseObservationEntries(content);
  return {
    file_path: filePath,
    content,
    header,
    entries
  };
}

function summarizeObservationEntityDocument(domain, document) {
  if (!document || !isPlainObject(document)) {
    return null;
  }

  const stat = fs.statSync(document.file_path);
  const lastEntry = document.entries.length > 0 ? document.entries[document.entries.length - 1] : null;
  return {
    domain,
    entity_name: document.header?.entity_name ?? null,
    file_path: document.file_path,
    entry_count: document.entries.length,
    created_at: document.header?.created_at ?? null,
    updated_at: stat.mtime.toISOString(),
    last_source_type: lastEntry?.source_type ?? null,
    last_confidence: lastEntry?.confidence ?? null,
    last_tags: Array.isArray(lastEntry?.tags) ? lastEntry.tags : []
  };
}

async function recordObservationTool(args) {
  try {
    const domain = normalizeObservationDomain(args.domain);
    const entityName = requireNonEmptyString(args.entity_name, "entity_name");
    const observation = requireNonEmptyString(args.observation, "observation");
    const sourceType = normalizeObservationSourceType(args.source_type);
    const confidence = normalizeObservationConfidence(args.confidence);
    const sourceTool = optionalString(args.source_tool, "source_tool");
    const sourceRef = optionalString(args.source_ref, "source_ref");
    const stateVersion = optionalInteger(args.state_version, "state_version");
    const tags = normalizeJournalTags(args.tags);

    const document = ensureObservationEntityDocument(domain, entityName);
    const entrySection = buildObservationEntrySection(observation, {
      source_type: sourceType,
      confidence,
      source_tool: sourceTool,
      source_ref: sourceRef,
      state_version: stateVersion,
      tags
    });
    fs.appendFileSync(document.file_path, entrySection, "utf8");

    const updatedDocument = readObservationEntityDocument(domain, entityName);
    const summary = summarizeObservationEntityDocument(domain, updatedDocument);
    return asToolResult(
      {
        ok: true,
        domain,
        entity_name: entityName,
        source_type: sourceType,
        confidence,
        file_path: document.file_path,
        entry_count: updatedDocument?.entries.length ?? 0,
        summary
      },
      false
    );
  } catch (error) {
    return asToolResult(toolErrorPayload(error), true);
  }
}

async function listObservationEntitiesTool(args) {
  try {
    const requestedDomain =
      args.domain === undefined || args.domain === null
        ? null
        : normalizeObservationDomain(args.domain);
    const query = optionalString(args.query, "query");
    const queryKey = normalizeTextComparisonKey(query);
    const maxResults = clampInteger(args.max_results, 50, 1, 200, "max_results");
    const domains = requestedDomain ? [requestedDomain] : OBSERVATION_DOMAIN_ENUM;
    const entities = [];

    for (const domain of domains) {
      const domainDir = getObservationDomainDir(domain);
      if (!fs.existsSync(domainDir)) {
        continue;
      }

      for (const entry of fs.readdirSync(domainDir, { withFileTypes: true })) {
        if (!entry.isFile() || path.extname(entry.name).toLowerCase() !== ".md") {
          continue;
        }

        const filePath = path.join(domainDir, entry.name);
        const content = fs.readFileSync(filePath, "utf8");
        const document = {
          file_path: filePath,
          content,
          header: parseObservationHeader(content),
          entries: parseObservationEntries(content)
        };
        const summary = summarizeObservationEntityDocument(domain, document);
        if (summary === null) {
          continue;
        }

        if (
          queryKey !== null &&
          normalizeTextComparisonKey(summary.entity_name) !== queryKey &&
          !(normalizeTextComparisonKey(summary.entity_name)?.includes(queryKey))
        ) {
          continue;
        }

        entities.push(summary);
      }
    }

    entities.sort((left, right) => {
      const leftTime = Date.parse(left.updated_at || "") || 0;
      const rightTime = Date.parse(right.updated_at || "") || 0;
      if (leftTime !== rightTime) {
        return rightTime - leftTime;
      }
      return String(left.entity_name ?? "").localeCompare(String(right.entity_name ?? ""), "zh-Hans-CN");
    });

    return asToolResult(
      {
        ok: true,
        domain: requestedDomain,
        query: query ?? null,
        entity_count: entities.length,
        entities: entities.slice(0, maxResults)
      },
      false
    );
  } catch (error) {
    return asToolResult(toolErrorPayload(error), true);
  }
}

async function readObservationEntityTool(args) {
  try {
    const domain = normalizeObservationDomain(args.domain);
    const entityName = requireNonEmptyString(args.entity_name, "entity_name");
    const lastN = optionalInteger(args.last_n, "last_n");
    const document = readObservationEntityDocument(domain, entityName);
    if (document === null) {
      return asToolResult(
        {
          ok: false,
          error: "observation_not_found",
          domain,
          entity_name: entityName
        },
        true
      );
    }

    const selectedEntries =
      Number.isInteger(lastN) && lastN > 0 && document.entries.length > lastN
        ? document.entries.slice(-lastN)
        : document.entries;
    const preamble = buildObservationPreamble(
      document.header?.domain ?? domain,
      document.header?.entity_name ?? entityName,
      document.header?.created_at ?? ""
    );
    return asToolResult(
      {
        ok: true,
        domain,
        entity_name: document.header?.entity_name ?? entityName,
        file_path: document.file_path,
        entry_count_total: document.entries.length,
        entry_count_returned: selectedEntries.length,
        content: `${preamble}${selectedEntries.map((entry) => entry.raw).join("\n")}`.trim()
      },
      false
    );
  } catch (error) {
    return asToolResult(toolErrorPayload(error), true);
  }
}

function getKnowledgeDir() {
  return path.join(__dirname, "knowledge");
}

const KNOWLEDGE_CANONICAL_TOPICS = [
  "route-planning",
  "deck-building",
  "card-tier-list",
  "combat-tips",
  "boss-guide",
  "enemy-patterns",
  "relics",
  "events",
  "knowledge-authoring"
];

const VALID_KNOWLEDGE_TOPICS = [
  ...KNOWLEDGE_CANONICAL_TOPICS,
  "regent",
  "routes",
  "decks",
  "cards",
  "combat",
  "bosses",
  "enemies",
  "templates",
  "authoring"
];

const KNOWLEDGE_TOPIC_ALIASES = {
  regent: "card-tier-list",
  routes: "route-planning",
  decks: "deck-building",
  cards: "card-tier-list",
  combat: "combat-tips",
  bosses: "boss-guide",
  enemies: "enemy-patterns",
  templates: "knowledge-authoring",
  authoring: "knowledge-authoring"
};

const KNOWLEDGE_TOPIC_METADATA = {
  "route-planning": {
    domain: "routes",
    description: "Pathing priorities, node valuation, and act-level routing heuristics.",
    decision_focus: "Choose routes by current deck weakness, recovery windows, and elite tolerance."
  },
  "deck-building": {
    domain: "decks",
    description: "How to shape a deck across offense, defense, draw, energy, and scaling.",
    decision_focus: "Fill missing combat roles before adding narrow payoff or win-more cards."
  },
  "card-tier-list": {
    domain: "cards",
    description: "Role-based card evaluations and when to pick, skip, or upgrade them.",
    decision_focus: "Judge cards by role fit, landing speed, and near-term matchup impact."
  },
  "combat-tips": {
    domain: "combat",
    description: "Turn planning, potion timing, lethal checks, and common combat heuristics.",
    decision_focus: "Decide each turn by incoming damage, lethal windows, and future safety."
  },
  "boss-guide": {
    domain: "enemies",
    description: "Boss-by-boss preparation, danger windows, and matchup-specific tactics.",
    decision_focus: "Prepare specific answers for each boss instead of relying on generic strength."
  },
  "enemy-patterns": {
    domain: "enemies",
    description: "Enemy intent patterns, breakpoints, and fight-specific execution notes.",
    decision_focus: "Classify fights by threat model, then choose kill order and resource timing."
  },
  relics: {
    domain: "relics",
    description: "Relic evaluation rules, synergy buckets, and route/shop implications.",
    decision_focus: "Evaluate relics by tempo, economy, scaling, and what route choices they unlock."
  },
  events: {
    domain: "events",
    description: "Event risk-reward patterns, sacrifice rules, and decision heuristics.",
    decision_focus: "Take events by current HP, deck stability, and whether the upside solves a real problem."
  },
  "knowledge-authoring": {
    domain: "authoring",
    description: "Template rules and observation-first workflow for MCP-managed knowledge authoring.",
    decision_focus: "Record evidence first, then promote only verified conclusions into canonical knowledge."
  }
};

function resolveKnowledgeTopic(topic, fieldName = "topic") {
  const rawTopic = requireNonEmptyString(topic, fieldName);
  if (!VALID_KNOWLEDGE_TOPICS.includes(rawTopic)) {
    throw new ToolPayloadError(
      "invalid_arguments",
      `${fieldName} must be one of: ${VALID_KNOWLEDGE_TOPICS.join(", ")}`,
      {
        field: fieldName,
        valid_topics: VALID_KNOWLEDGE_TOPICS
      }
    );
  }

  return KNOWLEDGE_TOPIC_ALIASES[rawTopic] ?? rawTopic;
}

function normalizeKnowledgeTopics(topics, fieldName = "topics") {
  if (topics === undefined || topics === null) {
    return [...KNOWLEDGE_CANONICAL_TOPICS];
  }

  const rawTopics = requireNonEmptyStringArray(topics, fieldName);
  const normalizedTopics = [];
  for (const rawTopic of rawTopics) {
    const topic = resolveKnowledgeTopic(rawTopic, fieldName);
    if (!normalizedTopics.includes(topic)) {
      normalizedTopics.push(topic);
    }
  }

  return normalizedTopics;
}

function readKnowledgeDocumentByTopic(topic) {
  const filePath = path.join(getKnowledgeDir(), `${topic}.md`);
  if (!fs.existsSync(filePath)) {
    throw new ToolPayloadError(
      "knowledge_not_found",
      `Knowledge file not found: ${topic}.md. Create it at ${filePath}`,
      {
        topic,
        file_path: filePath
      }
    );
  }

  const content = fs.readFileSync(filePath, "utf8");
  const lines = content.split(/\r?\n/);
  const sections = parseKnowledgeSections(lines);

  return {
    topic,
    file_path: filePath,
    content,
    lines,
    sections
  };
}

function getKnowledgeTopicAliases(topic) {
  return Object.entries(KNOWLEDGE_TOPIC_ALIASES)
    .filter(([, canonicalTopic]) => canonicalTopic === topic)
    .map(([alias]) => alias);
}

function parseKnowledgeSections(lines) {
  const sections = [];
  const pathStack = [];
  let inFence = false;
  let fenceMarker = null;

  for (let index = 0; index < lines.length; index += 1) {
    const line = lines[index];
    const trimmed = line.trim();
    const fenceMatch = trimmed.match(/^(```+|~~~+)/);
    if (fenceMatch) {
      const marker = fenceMatch[1][0];
      if (!inFence) {
        inFence = true;
        fenceMarker = marker;
      } else if (fenceMarker === marker) {
        inFence = false;
        fenceMarker = null;
      }
      continue;
    }

    if (inFence) {
      continue;
    }

    const headingMatch = line.match(/^(#{1,6})\s+(.+?)\s*$/);
    if (!headingMatch) {
      continue;
    }

    const level = headingMatch[1].length;
    const title = headingMatch[2].replace(/\s+#+\s*$/, "").trim();
    if (!title) {
      continue;
    }

    pathStack[level - 1] = title;
    pathStack.length = level;

    sections.push({
      level,
      title,
      start_line: index + 1,
      end_line: lines.length,
      path: [...pathStack],
      path_text: pathStack.join(" > ")
    });
  }

  const openSectionIndexes = [];
  for (let index = 0; index < sections.length; index += 1) {
    const current = sections[index];
    while (
      openSectionIndexes.length > 0 &&
      sections[openSectionIndexes[openSectionIndexes.length - 1]].level >= current.level
    ) {
      const previousIndex = openSectionIndexes.pop();
      sections[previousIndex].end_line = current.start_line - 1;
    }
    openSectionIndexes.push(index);
  }

  while (openSectionIndexes.length > 0) {
    const previousIndex = openSectionIndexes.pop();
    sections[previousIndex].end_line = lines.length;
  }

  return sections;
}

function getKnowledgeSectionForLine(sections, lineNumber) {
  if (!Array.isArray(sections) || !Number.isInteger(lineNumber) || lineNumber <= 0) {
    return null;
  }

  let matchedSection = null;
  for (const section of sections) {
    if (
      Number.isInteger(section.start_line) &&
      Number.isInteger(section.end_line) &&
      section.start_line <= lineNumber &&
      section.end_line >= lineNumber
    ) {
      if (matchedSection === null || section.level >= matchedSection.level) {
        matchedSection = section;
      }
    }
  }

  return matchedSection;
}

function buildKnowledgeSectionPreview(lines, section) {
  if (!Array.isArray(lines) || !isPlainObject(section)) {
    return null;
  }

  const startIndex = Math.max(0, section.start_line);
  const endIndex = Math.min(lines.length, section.end_line);
  for (let index = startIndex; index < endIndex; index += 1) {
    const line = typeof lines[index] === "string" ? lines[index].trim() : "";
    if (!line || line.startsWith("#")) {
      continue;
    }
    return line.length > 160 ? `${line.slice(0, 157)}...` : line;
  }

  return null;
}

function buildKnowledgeDocumentSummary(lines) {
  if (!Array.isArray(lines)) {
    return null;
  }

  let inFence = false;
  let fenceMarker = null;
  for (const rawLine of lines) {
    const trimmed = typeof rawLine === "string" ? rawLine.trim() : "";
    if (!trimmed) {
      continue;
    }

    const fenceMatch = trimmed.match(/^(```+|~~~+)/);
    if (fenceMatch) {
      const marker = fenceMatch[1][0];
      if (!inFence) {
        inFence = true;
        fenceMarker = marker;
      } else if (fenceMarker === marker) {
        inFence = false;
        fenceMarker = null;
      }
      continue;
    }

    if (inFence || /^#{1,6}\s+/.test(trimmed)) {
      continue;
    }

    const normalized = trimmed.replace(/^[-*+]\s+/, "").replace(/^\d+\.\s+/, "");
    if (!normalized) {
      continue;
    }

    return normalized.length > 200 ? `${normalized.slice(0, 197)}...` : normalized;
  }

  return null;
}

function matchKnowledgeHeading(section, heading, caseSensitive) {
  if (!isPlainObject(section) || typeof heading !== "string") {
    return false;
  }

  const target = caseSensitive ? heading : heading.toLowerCase();
  const title = caseSensitive ? section.title : section.title.toLowerCase();
  const pathText = caseSensitive ? section.path_text : section.path_text.toLowerCase();
  return title === target || pathText === target;
}

function matchKnowledgeSectionPath(section, sectionPath, caseSensitive) {
  if (!isPlainObject(section) || !Array.isArray(sectionPath) || !Array.isArray(section.path)) {
    return false;
  }

  if (section.path.length !== sectionPath.length) {
    return false;
  }

  for (let index = 0; index < sectionPath.length; index += 1) {
    const expected = caseSensitive ? sectionPath[index] : sectionPath[index].toLowerCase();
    const actual = caseSensitive ? section.path[index] : section.path[index].toLowerCase();
    if (expected !== actual) {
      return false;
    }
  }

  return true;
}

function summarizeKnowledgeSection(section, lines) {
  return {
    level: section.level,
    title: section.title,
    path: section.path,
    path_text: section.path_text,
    start_line: section.start_line,
    end_line: section.end_line,
    line_count:
      Number.isInteger(section.start_line) && Number.isInteger(section.end_line)
        ? section.end_line - section.start_line + 1
        : null,
    preview: buildKnowledgeSectionPreview(lines, section)
  };
}

function summarizeKnowledgeTopic(document) {
  const metadata = KNOWLEDGE_TOPIC_METADATA[document.topic] ?? {};
  const topLevelSections = document.sections.filter((section) => section.level === 1);
  const previewSections =
    topLevelSections.length > 0 ? topLevelSections : document.sections;

  return {
    topic: document.topic,
    file_path: document.file_path,
    domain: metadata.domain ?? null,
    aliases: getKnowledgeTopicAliases(document.topic),
    description: metadata.description ?? null,
    decision_focus: metadata.decision_focus ?? null,
    summary: buildKnowledgeDocumentSummary(document.lines),
    line_count: document.lines.length,
    section_count: document.sections.length,
    top_level_sections: previewSections
      .slice(0, 5)
      .map((section) => summarizeKnowledgeSection(section, document.lines))
  };
}

function summarizeKnowledgeDomains(topics) {
  const domainMap = new Map();
  for (const topic of Array.isArray(topics) ? topics : []) {
    if (!isPlainObject(topic) || typeof topic.topic !== "string") {
      continue;
    }

    const domain =
      typeof topic.domain === "string" && topic.domain.trim() ? topic.domain.trim() : "other";
    const current = domainMap.get(domain) ?? [];
    current.push(topic.topic);
    domainMap.set(domain, current);
  }

  return [...domainMap.entries()]
    .map(([domain, topicNames]) => ({
      domain,
      topics: topicNames
    }))
    .sort((left, right) => left.domain.localeCompare(right.domain));
}

async function getKnowledgeTool(args) {
  try {
    const rawTopic = requireNonEmptyString(args.topic, "topic");
    if (!VALID_KNOWLEDGE_TOPICS.includes(rawTopic)) {
      return asToolResult({
        ok: false,
        error: "invalid_topic",
        message: `Unknown topic '${rawTopic}'. Valid: ${VALID_KNOWLEDGE_TOPICS.join(", ")}`,
        valid_topics: VALID_KNOWLEDGE_TOPICS
      }, true);
    }

    const topic = KNOWLEDGE_TOPIC_ALIASES[rawTopic] ?? rawTopic;
    const document = readKnowledgeDocumentByTopic(topic);
    return asToolResult({ ok: true, topic, content: document.content }, false);
  } catch (error) {
    return asToolResult(toolErrorPayload(error), true);
  }
}

async function getKnowledgeTopicsTool() {
  try {
    const topics = KNOWLEDGE_CANONICAL_TOPICS.map((topic) =>
      summarizeKnowledgeTopic(readKnowledgeDocumentByTopic(topic))
    );
    return asToolResult(
      {
        ok: true,
        topic_count: topics.length,
        canonical_topics: KNOWLEDGE_CANONICAL_TOPICS,
        accepted_topics: VALID_KNOWLEDGE_TOPICS,
        aliases: KNOWLEDGE_TOPIC_ALIASES,
        domains: summarizeKnowledgeDomains(topics),
        topics
      },
      false
    );
  } catch (error) {
    return asToolResult(toolErrorPayload(error), true);
  }
}

async function searchKnowledgeTool(args) {
  try {
    const query = requireNonEmptyString(args.query, "query");
    const topics = normalizeKnowledgeTopics(args.topics, "topics");
    const caseSensitive = optionalBoolean(args.case_sensitive, "case_sensitive") ?? false;
    const useRegex = optionalBoolean(args.regex, "regex") ?? false;
    const contextBefore = clampInteger(args.context_before, 1, 0, 20, "context_before");
    const contextAfter = clampInteger(args.context_after, 1, 0, 20, "context_after");
    const maxResults = clampInteger(args.max_results, 20, 1, 100, "max_results");

    let regex = null;
    if (useRegex) {
      try {
        regex = new RegExp(query, caseSensitive ? "" : "i");
      } catch (error) {
        throw new ToolPayloadError(
          "invalid_arguments",
          `query must be a valid regular expression when regex=true: ${error instanceof Error ? error.message : String(error)}`,
          {
            field: "query"
          }
        );
      }
    }

    const results = [];
    let truncated = false;

    outer: for (const topic of topics) {
      const document = readKnowledgeDocumentByTopic(topic);
      for (let index = 0; index < document.lines.length; index += 1) {
        const line = document.lines[index];
        const haystack = caseSensitive ? line : line.toLowerCase();
        const needle = caseSensitive ? query : query.toLowerCase();
        const matchIndex = useRegex
          ? line.search(regex)
          : haystack.indexOf(needle);

        if (matchIndex < 0) {
          continue;
        }

        const lineNumber = index + 1;
        const section = getKnowledgeSectionForLine(document.sections, lineNumber);
        const contextStartLine = Math.max(1, lineNumber - contextBefore);
        const contextEndLine = Math.min(document.lines.length, lineNumber + contextAfter);

        results.push({
          topic,
          line: lineNumber,
          column: matchIndex + 1,
          heading_path: section?.path ?? [],
          heading_path_text: section?.path_text ?? null,
          section_start_line: section?.start_line ?? null,
          section_end_line: section?.end_line ?? null,
          match_text: line,
          context: {
            start_line: contextStartLine,
            end_line: contextEndLine,
            text: document.lines
              .slice(contextStartLine - 1, contextEndLine)
              .join("\n")
          }
        });

        if (results.length >= maxResults) {
          truncated = true;
          break outer;
        }
      }
    }

    return asToolResult(
      {
        ok: true,
        query,
        regex: useRegex,
        case_sensitive: caseSensitive,
        searched_topics: topics,
        result_count: results.length,
        truncated,
        results
      },
      false
    );
  } catch (error) {
    return asToolResult(toolErrorPayload(error), true);
  }
}

async function readKnowledgeSliceTool(args) {
  try {
    const topic = resolveKnowledgeTopic(args.topic, "topic");
    const heading = optionalString(args.heading, "heading");
    const sectionPath =
      args.section_path === undefined || args.section_path === null
        ? undefined
        : requireNonEmptyStringArray(args.section_path, "section_path");
    const occurrence = clampInteger(args.occurrence, 1, 1, 50, "occurrence");
    const caseSensitive = optionalBoolean(args.case_sensitive, "case_sensitive") ?? false;
    const startLine = optionalInteger(args.start_line, "start_line");
    const endLine = optionalInteger(args.end_line, "end_line");
    const maxLines = clampInteger(args.max_lines, undefined, 1, 400, "max_lines");

    const usingHeading = heading !== undefined && heading !== null;
    const usingSectionPath = Array.isArray(sectionPath) && sectionPath.length > 0;
    const usingLineRange = startLine !== undefined || endLine !== undefined;

    const selectorCount =
      (usingHeading ? 1 : 0) + (usingSectionPath ? 1 : 0) + (usingLineRange ? 1 : 0);

    if (selectorCount > 1) {
      throw new ToolPayloadError(
        "invalid_arguments",
        "Use exactly one selector: section_path, heading, or start_line/end_line.",
        {
          fields: ["section_path", "heading", "start_line", "end_line"]
        }
      );
    }

    if (selectorCount <= 0) {
      throw new ToolPayloadError(
        "invalid_arguments",
        "Provide exactly one selector: section_path, heading, or a start_line/end_line line range.",
        {
          fields: ["section_path", "heading", "start_line", "end_line"]
        }
      );
    }

    if (!usingHeading && !usingSectionPath && startLine === undefined) {
      throw new ToolPayloadError(
        "invalid_arguments",
        "start_line is required for line-range reads.",
        {
          field: "start_line"
        }
      );
    }

    const document = readKnowledgeDocumentByTopic(topic);

    let sliceStartLine;
    let sliceEndLine;
    let availableStartLine;
    let availableEndLine;
    let matchedSection = null;
    let mode = null;

    if (usingHeading) {
      const matchingSections = document.sections.filter((section) =>
        matchKnowledgeHeading(section, heading, caseSensitive)
      );
      if (matchingSections.length < occurrence) {
        return asToolResult(
          {
            ok: false,
            error: "knowledge_heading_not_found",
            topic,
            heading,
            occurrence,
            available_sections: document.sections.map((section) => section.path_text)
          },
          true
        );
      }

      matchedSection = matchingSections[occurrence - 1];
      availableStartLine = matchedSection.start_line;
      availableEndLine = matchedSection.end_line;
      mode = "heading";
    } else if (usingSectionPath) {
      const matchingSections = document.sections.filter((section) =>
        matchKnowledgeSectionPath(section, sectionPath, caseSensitive)
      );
      if (matchingSections.length < occurrence) {
        return asToolResult(
          {
            ok: false,
            error: "knowledge_section_path_not_found",
            topic,
            section_path: sectionPath,
            occurrence,
            available_sections: document.sections.map((section) => ({
              path: section.path,
              path_text: section.path_text
            }))
          },
          true
        );
      }

      matchedSection = matchingSections[occurrence - 1];
      availableStartLine = matchedSection.start_line;
      availableEndLine = matchedSection.end_line;
      mode = "section_path";
    } else {
      availableStartLine = startLine;
      if (availableStartLine < 1 || availableStartLine > document.lines.length) {
        throw new ToolPayloadError(
          "invalid_arguments",
          `start_line must be between 1 and ${document.lines.length}.`,
          {
            field: "start_line",
            max_line: document.lines.length
          }
        );
      }
      availableEndLine =
        endLine ??
        (maxLines === undefined
          ? availableStartLine
          : Math.min(document.lines.length, availableStartLine + maxLines - 1));
      if (availableEndLine < availableStartLine || availableEndLine > document.lines.length) {
        throw new ToolPayloadError(
          "invalid_arguments",
          `end_line must be between start_line and ${document.lines.length}.`,
          {
            field: "end_line",
            max_line: document.lines.length
          }
        );
      }

      matchedSection = getKnowledgeSectionForLine(document.sections, availableStartLine);
      mode = "line_range";
    }

    sliceStartLine = availableStartLine;
    sliceEndLine =
      maxLines === undefined
        ? availableEndLine
        : Math.min(availableEndLine, availableStartLine + maxLines - 1);
    const truncated = sliceEndLine < availableEndLine;

    return asToolResult(
      {
        ok: true,
        topic,
        mode,
        section_path: matchedSection?.path ?? null,
        requested_section_path: sectionPath ?? null,
        heading: matchedSection ? summarizeKnowledgeSection(matchedSection, document.lines) : null,
        section: matchedSection ? summarizeKnowledgeSection(matchedSection, document.lines) : null,
        start_line: sliceStartLine,
        end_line: sliceEndLine,
        available_start_line: availableStartLine,
        available_end_line: availableEndLine,
        max_lines: maxLines ?? null,
        truncated,
        content: document.lines
          .slice(sliceStartLine - 1, sliceEndLine)
          .join("\n")
      },
      false
    );
  } catch (error) {
    return asToolResult(toolErrorPayload(error), true);
  }
}

async function listKnowledgeSectionsTool(args) {
  try {
    const topics = normalizeKnowledgeTopics(args.topics, "topics");
    const maxSectionsPerTopic = clampInteger(
      args.max_sections_per_topic,
      200,
      1,
      200,
      "max_sections_per_topic"
    );

    const topicSections = topics.map((topic) => {
      const document = readKnowledgeDocumentByTopic(topic);
      return {
        topic,
        section_count: document.sections.length,
        sections: document.sections
          .slice(0, maxSectionsPerTopic)
          .map((section) => summarizeKnowledgeSection(section, document.lines))
      };
    });

    return asToolResult(
      {
        ok: true,
        topics: topicSections,
        max_sections_per_topic: maxSectionsPerTopic
      },
      false
    );
  } catch (error) {
    return asToolResult(toolErrorPayload(error), true);
  }
}

class ToolPayloadError extends Error {
  constructor(code, message, details) {
    super(message);
    this.code = code;
    this.details = details;
  }
}

class BridgeHttpError extends Error {
  constructor(status, code, message, details) {
    super(message);
    this.status = status;
    this.code = code;
    this.details = details;
  }
}
