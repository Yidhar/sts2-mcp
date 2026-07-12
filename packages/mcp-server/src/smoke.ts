import crypto from "node:crypto";
import { BridgeClient } from "./bridge-client";
import { loadConfig } from "./config";
import { errorBody } from "./errors";
import { safeStringify } from "./redaction";
import { SessionStore, capabilityNames, supportsControlV2 } from "./session";

function plainObject(value: unknown): value is Record<string, unknown> {
  return Boolean(value && typeof value === "object" && !Array.isArray(value));
}

export async function runSmoke(): Promise<number> {
  const config = loadConfig();
  const sessions = new SessionStore(config.sessionFile);
  const bridge = new BridgeClient({ timeoutMs: config.requestTimeoutMs });
  try {
    const session = sessions.read();
    if (!supportsControlV2(session)) {
      throw new Error("Live smoke requires a Bridge control-v2 session.");
    }
    if (!capabilityNames(session).includes("player-control")) {
      throw new Error("Bridge session does not advertise player-control.");
    }
    const advertisedVersions = Array.isArray(session.api_versions) ? session.api_versions : [];
    if (!advertisedVersions.includes(config.apiVersion)) {
      throw new Error(`Bridge does not advertise required API ${config.apiVersion}.`);
    }
    if (session.schema_version !== config.schemaVersion) {
      throw new Error(`Bridge schema ${String(session.schema_version)} does not match ${config.schemaVersion}.`);
    }
    const health = await bridge.health(session);
    if (!plainObject(health) || health.transport_alive !== true || health.game_thread_alive !== true) {
      throw new Error("Authenticated Bridge health is not ready.");
    }
    if (
      typeof session.session_id === "string" &&
      typeof health.session_id === "string" &&
      health.session_id !== session.session_id
    ) {
      throw new Error("Bridge health session identity does not match discovery.");
    }
    const state = await bridge.state(session);
    if (!Number.isSafeInteger(state.state_version)) {
      throw new Error("Authenticated Bridge state has no integer state_version.");
    }
    const actions = Array.isArray(state.available_actions) ? state.available_actions : [];
    const requestedHandle = String(process.env.STS2_LIVE_SMOKE_ACTION_HANDLE ?? "").trim();
    if (!requestedHandle) {
      throw new Error(
        "Set STS2_LIVE_SMOKE_ACTION_HANDLE to one operator-reviewed legal action for the current screen."
      );
    }
    const selected = actions.find((action) => {
      if (!plainObject(action)) return false;
      return action.handle === requestedHandle || action.action_id === requestedHandle;
    });
    if (!selected) {
      throw new Error("The configured live-smoke action handle is not currently legal.");
    }
    const requestId = crypto.randomUUID();
    const timeoutMs = Math.min(config.requestTimeoutMs, 120_000);
    const deadlineUtc = new Date(Date.now() + timeoutMs).toISOString();
    const mutation = {
      actionId: requestedHandle,
      expectedStateVersion: state.state_version as number,
      requestId,
      timeoutMs,
      deadlineUtc
    };
    const first = await bridge.executeAction(session, mutation);
    const replay = await bridge.executeAction(session, mutation);
    if (first.replayed_result || !replay.replayed_result) {
      throw new Error("Bridge did not distinguish first execution from retained duplicate replay.");
    }
    if (
      first.request_id !== requestId ||
      replay.request_id !== requestId ||
      first.status !== "committed" ||
      replay.status !== "committed" ||
      first.committed_state_version !== replay.committed_state_version
    ) {
      throw new Error("Bridge duplicate replay identity or committed revision is inconsistent.");
    }
    process.stdout.write(
      `${safeStringify(
        {
          ok: true,
          session: sessions.describe(session),
          health,
          state: {
            state_version: state.state_version,
            screen: state.screen ?? null,
            legal_action_count: actions.length
          },
          mutation: {
            action_handle: requestedHandle,
            request_id: requestId,
            first_replayed_result: first.replayed_result,
            duplicate_replayed_result: replay.replayed_result,
            committed_state_version: first.committed_state_version
          }
        },
        sessions.secrets()
      )}\n`
    );
    return 0;
  } catch (error) {
    process.stderr.write(`${safeStringify(errorBody(error), sessions.secrets())}\n`);
    return 1;
  }
}

if (require.main === module) {
  runSmoke().then((code) => {
    process.exitCode = code;
  });
}
