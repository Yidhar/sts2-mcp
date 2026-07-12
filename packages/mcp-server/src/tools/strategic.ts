import type { McpServer } from "@modelcontextprotocol/sdk/server/mcp.js";
import { z } from "zod";
import { AppError, errorBody } from "../errors";
import { compactState, stateVersion, toolResult } from "../presentation";
import type { ToolContext } from "../tool-context";
import { actionId, actionKind, actionsFromState, findAction } from "../workflows";

async function safeCall(context: ToolContext, callback: () => Promise<unknown>) {
  try {
    return toolResult(await callback(), false, context.sessions.secrets());
  } catch (error) {
    return toolResult(errorBody(error), true, context.sessions.secrets());
  }
}

interface RegistrationConfig<TSchema extends z.ZodType> {
  description: string;
  inputSchema: TSchema;
  annotations?: {
    readOnlyHint?: boolean;
    destructiveHint?: boolean;
    idempotentHint?: boolean;
  };
}

function register<TSchema extends z.ZodType>(
  server: McpServer,
  enabled: ReadonlySet<string>,
  name: string,
  config: RegistrationConfig<TSchema>,
  callback: (args: z.infer<TSchema>) => Promise<ReturnType<typeof toolResult>>
): void {
  if (enabled.has(name)) {
    (server.registerTool as unknown as (...args: unknown[]) => unknown)(name, config, callback);
  }
}

async function runSequence(
  context: ToolContext,
  args: {
    action_ids: string[];
    expected_state_version?: number;
    request_ids?: string[];
    wait_after_ms?: number;
    return_state_after?: boolean;
  },
  allowedKinds?: ReadonlySet<string>
): Promise<Record<string, unknown>> {
  const session = context.sessions.read();
  let state = await context.bridge.state(session);
  let version = stateVersion(state);
  if (args.expected_state_version !== undefined && args.expected_state_version !== version) {
    throw new AppError("state_version_conflict", "The supplied sequence revision is stale.", {
      details: { expected_state_version: args.expected_state_version, actual_state_version: version }
    });
  }

  if (args.request_ids && args.request_ids.length !== args.action_ids.length) {
    throw new AppError("invalid_request_ids", "request_ids length must match action_ids length.");
  }

  const steps: Record<string, unknown>[] = [];
  for (let index = 0; index < args.action_ids.length; index += 1) {
    const requestedId = args.action_ids[index] as string;
    const selected = findAction(state, { actionId: requestedId });
    if (allowedKinds && !allowedKinds.has(actionKind(selected))) {
      throw new AppError("sequence_action_not_allowed", `Action '${requestedId}' is not valid for this sequence.`, {
        details: { action_id: requestedId, kind: actionKind(selected), index }
      });
    }
    const selectedId = actionId(selected);
    if (!selectedId) throw new AppError("invalid_bridge_action", "A sequence action has no canonical handle.");
    const command = await context.bridge.executeAction(session, {
      actionId: selectedId,
      expectedStateVersion: version,
      waitAfterMs: args.wait_after_ms,
      requestId: args.request_ids?.[index]
    });
    steps.push({
      index,
      action_id: selectedId,
      request_id: command.request_id,
      protocol: command.protocol,
      result: command.payload
    });
    // This is the next planned command, not a retry. It must bind to a newly observed revision.
    state = await context.bridge.state(session);
    version = stateVersion(state);
  }

  return {
    ok: true,
    step_count: steps.length,
    steps,
    ...(args.return_state_after ? { state: compactState(state) } : {})
  };
}

export function registerStrategicTools(
  server: McpServer,
  context: ToolContext,
  enabled: ReadonlySet<string>
): void {
  register(
    server,
    enabled,
    "sts2_get_map_routes",
    {
      description: "Return the visible map plus currently legal travel actions; no hidden-route inference.",
      inputSchema: z.strictObject({ detail: z.enum(["summary", "full"]).optional() }),
      annotations: { readOnlyHint: true, idempotentHint: true }
    },
    async ({ detail }) =>
      safeCall(context, async () => {
        const session = context.sessions.read();
        const state = await context.bridge.state(session);
        const travelActions = actionsFromState(state).filter((action) =>
          ["map", "travel"].includes(actionKind(action))
        );
        return {
          state_version: state.state_version ?? null,
          screen: state.screen ?? null,
          map: state.map ?? null,
          travel_actions: travelActions,
          detail: detail ?? "summary"
        };
      })
  );

  register(
    server,
    enabled,
    "sts2_get_deck",
    {
      description: "Return the current player-visible master deck.",
      inputSchema: z.strictObject({}),
      annotations: { readOnlyHint: true, idempotentHint: true }
    },
    async () =>
      safeCall(context, async () => {
        const session = context.sessions.read();
        const state = await context.bridge.state(session);
        const player =
          state.player && typeof state.player === "object"
            ? (state.player as Record<string, unknown>)
            : {};
        return {
          state_version: state.state_version ?? null,
          deck: state.deck ?? player.deck ?? player.master_deck ?? null
        };
      })
  );

  const sequenceSchema = z.strictObject({
    action_ids: z.array(z.string().min(1)).min(1).max(100),
    expected_state_version: z.number().int().nonnegative().optional(),
    request_ids: z.array(z.string().uuid()).max(100).optional(),
    wait_after_ms: z.number().int().min(0).max(5_000).optional(),
    return_state_after: z.boolean().optional(),
    strict: z.boolean().optional()
  });

  register(
    server,
    enabled,
    "sts2_play_card_sequence",
    {
      description: "Execute an ordered sequence of legal play-card commands, each bound to a fresh revision.",
      inputSchema: sequenceSchema,
      annotations: { destructiveHint: true, idempotentHint: false }
    },
    async (args) =>
      safeCall(context, async () => {
        if (args.strict === false) {
          throw new AppError("strict_revision_required", "Sequence strictness cannot be disabled.");
        }
        return runSequence(context, args, new Set(["play_card"]));
      })
  );

  register(
    server,
    enabled,
    "sts2_execute_combat_sequence",
    {
      description: "Execute ordered legal combat commands, each bound to a newly observed revision.",
      inputSchema: sequenceSchema,
      annotations: { destructiveHint: true, idempotentHint: false }
    },
    async (args) =>
      safeCall(context, async () => {
        if (args.strict === false) {
          throw new AppError("strict_revision_required", "Sequence strictness cannot be disabled.");
        }
        return runSequence(context, args, new Set(["play_card", "use_potion", "combat"]));
      })
  );
}
