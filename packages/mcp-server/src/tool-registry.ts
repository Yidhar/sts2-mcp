import type { McpServer } from "@modelcontextprotocol/sdk/server/mcp.js";
import type { ToolProfile } from "./config";
import type { ToolContext } from "./tool-context";
import { registerControlTools } from "./tools/control";
import { registerStrategicTools } from "./tools/strategic";
import { registerTrainingTools } from "./tools/training";

const MINIMAL_TOOLS = [
  "sts2_get_bridge_status",
  "sts2_get_state",
  "sts2_list_actions",
  "sts2_perform_action",
  "sts2_get_command_status",
  "sts2_end_turn",
  "sts2_pick_option",
  "sts2_resolve_room_rewards",
  "sts2_resolve_rest_site",
  "sts2_resolve_card_selection",
  "sts2_resolve_shop_visit",
  "sts2_travel_to_coordinate",
  "sts2_wait_for_change",
  "sts2_wait_until_actionable"
] as const;

const STRATEGIC_TOOLS = [
  ...MINIMAL_TOOLS,
  "sts2_get_map_routes",
  "sts2_get_deck",
  "sts2_play_card_sequence",
  "sts2_execute_combat_sequence"
] as const;

const DEBUG_TOOLS = [
  ...STRATEGIC_TOOLS,
  "sts2_env_spec",
  "sts2_env_state",
  "sts2_env_combat_catalog",
  "sts2_env_reset",
  "sts2_env_step",
  "sts2_env_command_status"
] as const;

export function toolNamesForProfile(profile: ToolProfile): ReadonlySet<string> {
  switch (profile) {
    case "minimal":
      return new Set(MINIMAL_TOOLS);
    case "strategic":
      return new Set(STRATEGIC_TOOLS);
    case "debug":
      return new Set(DEBUG_TOOLS);
  }
}

export function registerTools(server: McpServer, context: ToolContext): ReadonlySet<string> {
  const enabled = toolNamesForProfile(context.config.profile);
  registerControlTools(server, context, enabled);
  registerStrategicTools(server, context, enabled);
  registerTrainingTools(server, context, enabled);
  return enabled;
}
