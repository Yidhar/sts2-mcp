import type { McpServer } from "@modelcontextprotocol/sdk/server/mcp.js";
import { z } from "zod";
import { AppError, errorBody } from "../errors";
import { compactState, listActions, stateVersion, toolResult } from "../presentation";
import type { ToolContext } from "../tool-context";
import type { BridgeState } from "../types";
import { actionId, findAction, waitForStateChange } from "../workflows";

async function safeCall(context: ToolContext, callback: () => Promise<unknown>) {
  try {
    const payload = await callback();
    return toolResult(payload, false, context.sessions.secrets());
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

async function runSelectedAction(
  context: ToolContext,
  options: {
    actionId?: string;
    kinds?: readonly string[];
    optionIndex?: number;
    x?: number;
    y?: number;
    expectedStateVersion?: number;
    waitAfterMs?: number;
    requestId?: string;
  }
): Promise<Record<string, unknown>> {
  const session = context.sessions.read();
  const state = await context.bridge.state(session);
  const selected = findAction(state, options);
  const selectedId = actionId(selected);
  if (!selectedId) {
    throw new AppError("invalid_bridge_action", "The selected legal action has no canonical handle.");
  }
  const currentVersion = stateVersion(state);
  if (
    Number.isSafeInteger(options.expectedStateVersion) &&
    options.expectedStateVersion !== currentVersion
  ) {
    throw new AppError("state_version_conflict", "The supplied revision is no longer current.", {
      details: {
        expected_state_version: options.expectedStateVersion,
        actual_state_version: currentVersion
      }
    });
  }
  const result = await context.bridge.executeAction(session, {
    actionId: selectedId,
    expectedStateVersion: currentVersion,
    waitAfterMs: options.waitAfterMs,
    requestId: options.requestId
  });
  return {
    ok: true,
    protocol: result.protocol,
    request_id: result.request_id,
    selected_action_id: selectedId,
    result: result.payload
  };
}

const mutationBase = {
  expected_state_version: z.number().int().nonnegative().optional(),
  request_id: z.string().uuid().optional(),
  wait_after_ms: z.number().int().min(0).max(5_000).optional()
};

export function registerControlTools(
  server: McpServer,
  context: ToolContext,
  enabled: ReadonlySet<string>
): void {
  register(
    server,
    enabled,
    "sts2_get_bridge_status",
    {
      description: "Validate the local Bridge session and health endpoint without exposing credentials.",
      inputSchema: z.strictObject({}),
      annotations: { readOnlyHint: true, idempotentHint: true }
    },
    async () =>
      safeCall(context, async () => {
        const session = context.sessions.read();
        const health = await context.bridge.health(session);
        return {
          ok: true,
          profile: context.config.profile,
          session: context.sessions.describe(session),
          health
        };
      })
  );

  register(
    server,
    enabled,
    "sts2_get_state",
    {
      description: "Return the current player-visible Bridge state.",
      inputSchema: z.strictObject({ compact: z.boolean().optional() }),
      annotations: { readOnlyHint: true, idempotentHint: true }
    },
    async ({ compact }) =>
      safeCall(context, async () => {
        const session = context.sessions.read();
        const state = await context.bridge.state(session);
        return compact === false ? state : compactState(state);
      })
  );

  register(
    server,
    enabled,
    "sts2_list_actions",
    {
      description: "Return the legal actions and the revision they belong to.",
      inputSchema: z.strictObject({}),
      annotations: { readOnlyHint: true, idempotentHint: true }
    },
    async () =>
      safeCall(context, async () => {
        const session = context.sessions.read();
        return listActions(await context.bridge.state(session));
      })
  );

  register(
    server,
    enabled,
    "sts2_perform_action",
    {
      description:
        "Execute one legal action exactly once. expected_state_version is mandatory and strict cannot be disabled.",
      inputSchema: z.strictObject({
        action_id: z.string().min(1),
        expected_state_version: z.number().int().nonnegative(),
        request_id: z.string().uuid().optional(),
        wait_after_ms: z.number().int().min(0).max(5_000).optional(),
        strict: z.boolean().optional(),
        return_state_after: z.boolean().optional()
      }),
      annotations: { destructiveHint: true, idempotentHint: false }
    },
    async (args) =>
      safeCall(context, async () => {
        if (args.strict === false) {
          throw new AppError(
            "strict_revision_required",
            "strict=false is no longer supported; mutations always enforce the supplied revision."
          );
        }
        const session = context.sessions.read();
        const result = await context.bridge.executeAction(session, {
          actionId: args.action_id,
          expectedStateVersion: args.expected_state_version,
          requestId: args.request_id,
          waitAfterMs: args.wait_after_ms
        });
        const payload: Record<string, unknown> = {
          ok: true,
          protocol: result.protocol,
          request_id: result.request_id,
          result: result.payload
        };
        if (args.return_state_after) payload.state = compactState(await context.bridge.state(session));
        return payload;
      })
  );

  register(
    server,
    enabled,
    "sts2_get_command_status",
    {
      description: "Query the final status of a control-v2 command by request_id.",
      inputSchema: z.strictObject({ request_id: z.string().uuid() }),
      annotations: { readOnlyHint: true, idempotentHint: true }
    },
    async ({ request_id }) =>
      safeCall(context, async () => {
        const session = context.sessions.read();
        return {
          ok: true,
          request_id,
          result: await context.bridge.commandStatus(session, request_id)
        };
      })
  );

  register(
    server,
    enabled,
    "sts2_end_turn",
    {
      description: "End the current turn using a freshly validated legal action.",
      inputSchema: z.strictObject(mutationBase),
      annotations: { destructiveHint: true, idempotentHint: false }
    },
    async (args) => safeCall(context, () => runSelectedAction(context, { ...args, actionId: "end_turn" }))
  );

  register(
    server,
    enabled,
    "sts2_pick_option",
    {
      description: "Pick one currently legal event/menu option. Pass action_id when multiple options exist.",
      inputSchema: z.strictObject({
        ...mutationBase,
        action_id: z.string().min(1).optional(),
        option_index: z.number().int().nonnegative().optional()
      }),
      annotations: { destructiveHint: true, idempotentHint: false }
    },
    async (args) =>
      safeCall(context, () =>
        runSelectedAction(context, {
          ...args,
          actionId: args.action_id,
          optionIndex: args.option_index,
          kinds: ["pick_option", "choose_option", "select_option", "event_option"]
        })
      )
  );

  const workflowTools: Array<{
    name: string;
    description: string;
    kinds: string[];
  }> = [
    {
      name: "sts2_resolve_room_rewards",
      description: "Execute one unambiguous room reward action, or an explicit action_id.",
      kinds: ["reward", "card_reward", "treasure", "treasure_relic", "proceed"]
    },
    {
      name: "sts2_resolve_rest_site",
      description: "Execute one unambiguous rest-site action, or an explicit action_id.",
      kinds: ["rest_site", "deck_upgrade", "proceed"]
    },
    {
      name: "sts2_resolve_card_selection",
      description: "Execute one unambiguous card-selection action, or an explicit action_id.",
      kinds: ["card_selection", "deck_upgrade"]
    },
    {
      name: "sts2_resolve_shop_visit",
      description: "Execute one unambiguous shop action, or an explicit action_id.",
      kinds: ["shop"]
    }
  ];

  for (const workflow of workflowTools) {
    register(
      server,
      enabled,
      workflow.name,
      {
        description: workflow.description,
        inputSchema: z.strictObject({
          ...mutationBase,
          action_id: z.string().min(1).optional(),
          option_index: z.number().int().nonnegative().optional()
        }),
        annotations: { destructiveHint: true, idempotentHint: false }
      },
      async (args) =>
        safeCall(context, () =>
          runSelectedAction(context, {
            ...args,
            actionId: args.action_id,
            optionIndex: args.option_index,
            kinds: workflow.kinds
          })
        )
    );
  }

  register(
    server,
    enabled,
    "sts2_travel_to_coordinate",
    {
      description: "Travel to a currently legal map coordinate.",
      inputSchema: z.strictObject({
        ...mutationBase,
        x: z.number().int(),
        y: z.number().int()
      }),
      annotations: { destructiveHint: true, idempotentHint: false }
    },
    async (args) =>
      safeCall(context, () =>
        runSelectedAction(context, { ...args, x: args.x, y: args.y, kinds: ["map", "travel"] })
      )
  );

  register(
    server,
    enabled,
    "sts2_wait_for_change",
    {
      description: "Wait until state_version advances. This tool never mutates the game.",
      inputSchema: z.strictObject({
        after_state_version: z.number().int().nonnegative().optional(),
        timeout_ms: z.number().int().min(1).max(120_000).optional(),
        poll_interval_ms: z.number().int().min(50).max(2_000).optional()
      }),
      annotations: { readOnlyHint: true, idempotentHint: true }
    },
    async (args) =>
      safeCall(context, async () => {
        const session = context.sessions.read();
        const initial = await context.bridge.state(session);
        const after = args.after_state_version ?? stateVersion(initial);
        const result = await waitForStateChange(
          () => context.bridge.state(session),
          after,
          args.timeout_ms ?? 20_000,
          args.poll_interval_ms ?? context.config.waitPollMs
        );
        return { ok: result.changed, timed_out: !result.changed, state: compactState(result.state) };
      })
  );

  register(
    server,
    enabled,
    "sts2_wait_until_actionable",
    {
      description: "Wait until at least one legal action is available. This tool never mutates the game.",
      inputSchema: z.strictObject({
        timeout_ms: z.number().int().min(1).max(120_000).optional(),
        poll_interval_ms: z.number().int().min(50).max(2_000).optional()
      }),
      annotations: { readOnlyHint: true, idempotentHint: true }
    },
    async (args) =>
      safeCall(context, async () => {
        const session = context.sessions.read();
        const deadline = Date.now() + (args.timeout_ms ?? 20_000);
        let state: BridgeState;
        do {
          state = await context.bridge.state(session);
          if (Array.isArray(state.available_actions) && state.available_actions.length > 0) {
            return { ok: true, timed_out: false, state: compactState(state) };
          }
          await new Promise((resolve) =>
            setTimeout(resolve, args.poll_interval_ms ?? context.config.waitPollMs)
          );
        } while (Date.now() < deadline);
        return { ok: false, timed_out: true, state: compactState(state!) };
      })
  );
}
