import crypto from "node:crypto";
import type { McpServer } from "@modelcontextprotocol/sdk/server/mcp.js";
import { z } from "zod";
import { AppError, errorBody } from "../errors";
import { toolResult } from "../presentation";
import { supportsControlV2 } from "../session";
import type { ToolContext } from "../tool-context";
import type { BridgeSession } from "../types";

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

function requireSessionId(session: BridgeSession): string {
  const sessionId =
    (typeof session.session_id === "string" && session.session_id.trim()) ||
    (typeof session.instance_id === "string" && session.instance_id.trim());
  if (!sessionId) throw new AppError("session_id_missing", "A v2 Bridge session must expose session_id.");
  return sessionId;
}

export function registerTrainingTools(
  server: McpServer,
  context: ToolContext,
  enabled: ReadonlySet<string>
): void {
  register(
    server,
    enabled,
    "sts2_env_spec",
    {
      description: "Read the privileged training-environment specification (debug profile only).",
      inputSchema: z.strictObject({}),
      annotations: { readOnlyHint: true, idempotentHint: true }
    },
    async () =>
      safeCall(context, async () => {
        const session = context.sessions.read();
        const isV2 = supportsControlV2(session);
        return context.bridge.request(session, isV2 ? "v2/env/spec" : "env/spec", {
          capability: isV2 ? "training" : undefined
        });
      })
  );

  register(
    server,
    enabled,
    "sts2_env_state",
    {
      description: "Read the privileged training state through the scoped environment API (debug only).",
      inputSchema: z.strictObject({}),
      annotations: { readOnlyHint: true, idempotentHint: true }
    },
    async () =>
      safeCall(context, async () => {
        const session = context.sessions.read();
        const isV2 = supportsControlV2(session);
        return context.bridge.request(session, isV2 ? "v2/env/state" : "state", {
          capability: isV2 ? "training" : undefined
        });
      })
  );

  register(
    server,
    enabled,
    "sts2_env_combat_catalog",
    {
      description: "Read the privileged combat-sandbox catalog (debug profile only).",
      inputSchema: z.strictObject({}),
      annotations: { readOnlyHint: true, idempotentHint: true }
    },
    async () =>
      safeCall(context, async () => {
        const session = context.sessions.read();
        const isV2 = supportsControlV2(session);
        return context.bridge.request(
          session,
          isV2 ? "v2/env/combat_catalog" : "env/combat_catalog",
          { capability: isV2 ? "training" : undefined }
        );
      })
  );

  register(
    server,
    enabled,
    "sts2_env_reset",
    {
      description:
        "Reset the privileged training environment exactly once with a UUID request identity (debug only).",
      inputSchema: z.strictObject({
        expected_state_version: z.number().int().nonnegative(),
        request_id: z.string().uuid().optional(),
        scenario: z.enum(["full-run", "combat"]).optional(),
        character: z.string().min(1).optional(),
        seed: z.union([z.number().int(), z.string().min(1)]).optional(),
        defensive_buffs: z.boolean().optional(),
        timeout_ms: z.number().int().min(1).max(120_000).optional()
      }),
      annotations: { destructiveHint: true, idempotentHint: false }
    },
    async (args) =>
      safeCall(context, async () => {
        const session = context.sessions.read();
        const requestId = args.request_id ?? crypto.randomUUID();
        const isV2 = supportsControlV2(session);
        const body = isV2
          ? {
              request_id: requestId,
              session_id: requireSessionId(session),
              expected_state_version: args.expected_state_version,
              scenario: args.scenario ?? "full-run",
              seed: args.seed ?? null,
              options: {
                ...(args.character ? { character: args.character } : {}),
                ...(args.defensive_buffs !== undefined
                  ? { defensive_buffs: args.defensive_buffs }
                  : {})
              }
            }
          : {
              request_id: requestId,
              expected_state_version: args.expected_state_version,
              ...(args.character ? { character: args.character } : {}),
              ...(typeof args.seed === "number" ? { seed: args.seed } : {}),
              ...(args.defensive_buffs !== undefined
                ? { defensive_buffs: args.defensive_buffs }
                : {}),
              timeout_ms: args.timeout_ms ?? 45_000
            };
        const timeoutMs = args.timeout_ms ?? 45_000;
        if (isV2) {
          const command = await context.bridge.executeEnvironmentCommand(
            session,
            "v2/env/reset",
            requestId,
            body,
            timeoutMs
          );
          return {
            ok: true,
            protocol: command.protocol,
            request_id: requestId,
            result: command.payload
          };
        }
        // Legacy mutations are attempted once. A transport timeout is an
        // explicit unknown outcome, never a generic retryable request error.
        const command = await context.bridge.executeLegacyEnvironmentCommand(
          session,
          "env/reset",
          requestId,
          body,
          timeoutMs
        );
        return {
          ok: true,
          protocol: "legacy-v1-one-shot",
          request_id: requestId,
          result: command.payload
        };
      })
  );

  register(
    server,
    enabled,
    "sts2_env_step",
    {
      description:
        "Advance a privileged training episode exactly once with UUID, episode, and step identities (debug only).",
      inputSchema: z
        .strictObject({
          episode_id: z.string().min(1),
          expected_step_index: z.number().int().nonnegative(),
          request_id: z.string().uuid().optional(),
          action_index: z.number().int().nonnegative().optional(),
          action_handle: z.string().min(1).optional(),
          timeout_ms: z.number().int().min(1).max(120_000).optional()
        })
        .refine(
          (value) => (value.action_handle === undefined) !== (value.action_index === undefined),
          {
            message: "provide exactly one of action_handle or action_index"
          }
        ),
      annotations: { destructiveHint: true, idempotentHint: false }
    },
    async (args) =>
      safeCall(context, async () => {
        const session = context.sessions.read();
        const requestId = args.request_id ?? crypto.randomUUID();
        const isV2 = supportsControlV2(session);
        const action = {
          ...(args.action_handle
            ? isV2
              ? { action_handle: args.action_handle }
              : { action_id: args.action_handle }
            : {}),
          ...(args.action_index !== undefined ? { action_index: args.action_index } : {})
        };
        const body = isV2
          ? {
              request_id: requestId,
              session_id: requireSessionId(session),
              episode_id: args.episode_id,
              expected_step_index: args.expected_step_index,
              action
            }
          : {
              request_id: requestId,
              episode_id: args.episode_id,
              expected_step_index: args.expected_step_index,
              ...action,
              timeout_ms: args.timeout_ms ?? 20_000
            };
        const timeoutMs = args.timeout_ms ?? 20_000;
        if (isV2) {
          const command = await context.bridge.executeEnvironmentCommand(
            session,
            "v2/env/step",
            requestId,
            body,
            timeoutMs
          );
          return {
            ok: true,
            protocol: command.protocol,
            request_id: requestId,
            result: command.payload
          };
        }
        const command = await context.bridge.executeLegacyEnvironmentCommand(
          session,
          "env/step",
          requestId,
          body,
          timeoutMs
        );
        return {
          ok: true,
          protocol: "legacy-v1-one-shot",
          request_id: requestId,
          result: command.payload
        };
      })
  );

  register(
    server,
    enabled,
    "sts2_env_command_status",
    {
      description: "Query a retained training-v2 reset/step result with its original request_id.",
      inputSchema: z.strictObject({ request_id: z.string().uuid() }),
      annotations: { readOnlyHint: true, idempotentHint: true }
    },
    async ({ request_id }) =>
      safeCall(context, async () => {
        const session = context.sessions.read();
        if (!supportsControlV2(session)) {
          throw new AppError(
            "command_status_unavailable",
            "Legacy training endpoints do not retain queryable command results."
          );
        }
        return {
          ok: true,
          request_id,
          result: await context.bridge.commandStatus(session, request_id, "training")
        };
      })
  );
}
