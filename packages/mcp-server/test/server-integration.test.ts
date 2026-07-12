import test from "node:test";
import assert from "node:assert/strict";
import http from "node:http";
import type { AddressInfo } from "node:net";
import { Client } from "@modelcontextprotocol/sdk/client/index.js";
import { InMemoryTransport } from "@modelcontextprotocol/sdk/inMemory.js";
import { BridgeClient } from "../src/bridge-client";
import { loadConfig } from "../src/config";
import { createMcpServer } from "../src/protocol/server";
import { SessionStore } from "../src/session";

const READY_GAME_COMPATIBILITY = {
  health: "ready",
  startup_allowed: true,
  error_code: "",
  error_message: "",
  profile_id: "retail-2026-06-23-5926027",
  assembly: {
    name: "sts2",
    assembly_version: "0.1.0.0",
    informational_version: "0.1.0+59260271157f76a2896f0eab5bc6ea1245d8b314",
    module_version_id: "97f10687-c306-4798-ab75-8b9f23f34dfb"
  },
  probes: [
    {
      capability: "lifecycle.game.ready",
      passed: true,
      code: "capability_present",
      detail: "Required capability matched the audited profile."
    }
  ]
};

function v2Descriptor(
  port: number,
  sessionId: string,
  options: { training?: boolean } = {}
): Record<string, unknown> {
  const training = options.training === true;
  return {
    pid: process.pid,
    session_id: sessionId,
    process_started_at_utc: "2026-07-11T11:59:00Z",
    created_at_utc: "2026-07-11T11:59:00Z",
    base_url: `http://127.0.0.1:${port}`,
    api_versions: ["2.0.0"],
    schema_version: "2026-07-13.1",
    action_schema_version: "2.1.0",
    legal_action_ordering_version: "2.0.0",
    capabilities: training ? ["player-control", "training"] : ["player-control"],
    capability_tokens: {
      "player-control": "scoped-player-token-00000000000000000000",
      ...(training ? { training: "scoped-training-token-000000000000000000" } : {})
    },
    game_compatibility: READY_GAME_COMPATIBILITY
  };
}

test("official SDK exposes minimal tools and returns token-safe structured stale errors", async () => {
  const token = "sdk-integration-secret";
  const sessions = new SessionStore("memory.json", {
    readText: () => JSON.stringify({ pid: 999_999, base_url: "http://127.0.0.1:7777", token }),
    processAlive: () => false
  });
  const config = { ...loadConfig([], {}), profile: "minimal" as const, sessionFile: "memory.json" };
  const { server } = createMcpServer(config, { sessions, bridge: new BridgeClient() });
  const client = new Client({ name: "mcp-test-client", version: "1" });
  const [clientTransport, serverTransport] = InMemoryTransport.createLinkedPair();
  await Promise.all([server.connect(serverTransport), client.connect(clientTransport)]);
  try {
    const listed = await client.listTools();
    const names = listed.tools.map((tool) => tool.name);
    assert(names.includes("sts2_get_state"));
    assert(!names.includes("sts2_env_reset"));
    assert(!names.includes("sts2_get_knowledge"));

    const invalid = await client.callTool({
      name: "sts2_get_state",
      arguments: { unexpected: true }
    });
    assert.equal(invalid.isError, true);
    assert.match(JSON.stringify(invalid), /invalid|unrecognized/i);

    const response = await client.callTool({ name: "sts2_get_bridge_status", arguments: {} });
    assert.equal(response.isError, true);
    const json = JSON.stringify(response);
    assert(!json.includes(token));
    assert.match(json, /stale_session/);
    assert.equal((response.structuredContent as { ok: boolean }).ok, false);
  } finally {
    await Promise.all([client.close(), server.close()]);
  }
});

test("real v2 handle wire reaches command endpoint and rejected envelope is an MCP error", async () => {
  let commandBody: Record<string, unknown> | null = null;
  const httpServer = http.createServer((request, response) => {
    if (request.method === "GET" && request.url === "/v2/state") {
      response.writeHead(200, { "Content-Type": "application/json" });
      response.end(
        JSON.stringify({
          session_id: "sdk-v2-session",
          state_version: 41,
          captured_at_utc: "2026-07-11T12:00:00Z",
          visibility: "player",
          state: { screen: "COMBAT" },
          legal_actions: [{ handle: "end_turn", kind: "combat", label: "End Turn" }]
        })
      );
      return;
    }
    if (request.method === "POST" && request.url === "/v2/commands") {
      let body = "";
      request.setEncoding("utf8");
      request.on("data", (chunk) => (body += chunk));
      request.on("end", () => {
        commandBody = JSON.parse(body) as Record<string, unknown>;
        response.writeHead(200, { "Content-Type": "application/json" });
        response.end(
          JSON.stringify({
            ok: false,
            api_version: "2.0.0",
            schema_version: "2026-07-13.1",
            request_id: commandBody.request_id,
            status: "rejected_before_execution",
            replayed_result: false,
            accepted_at_utc: "2026-07-11T12:00:00Z",
            error: { code: "state_version_conflict", message: "stale revision" }
          })
        );
      });
      return;
    }
    response.writeHead(404).end();
  });
  await new Promise<void>((resolve, reject) => {
    httpServer.once("error", reject);
    httpServer.listen(0, "127.0.0.1", () => resolve());
  });
  const address = httpServer.address() as AddressInfo;
  const sessions = new SessionStore("memory.json", {
    readText: () =>
      JSON.stringify(v2Descriptor(address.port, "sdk-v2-session")),
    processAlive: () => true
  });
  const config = { ...loadConfig([], {}), profile: "minimal" as const, sessionFile: "memory.json" };
  const { server } = createMcpServer(config, { sessions, bridge: new BridgeClient() });
  const client = new Client({ name: "mcp-wire-test", version: "1" });
  const [clientTransport, serverTransport] = InMemoryTransport.createLinkedPair();
  await Promise.all([server.connect(serverTransport), client.connect(clientTransport)]);
  try {
    const response = await client.callTool({ name: "sts2_end_turn", arguments: {} });
    assert.equal(response.isError, true);
    assert.equal((response.structuredContent as { ok: boolean }).ok, false);
    assert.match(JSON.stringify(response), /state_version_conflict/);
    const body = commandBody as unknown as Record<string, unknown>;
    assert.deepEqual(body.command, {
      kind: "perform_action",
      action_handle: "end_turn",
      wait_after_ms: 0
    });
  } finally {
    await Promise.all([client.close(), server.close()]);
    httpServer.closeAllConnections();
    await new Promise<void>((resolve) => httpServer.close(() => resolve()));
  }
});

test("debug training reads use scoped v2 routes and rejected reset is an MCP error", async () => {
  const seen: Array<{ path: string; authorization?: string }> = [];
  const requestId = "123e4567-e89b-42d3-a456-426614174010";
  const stepRequestId = "123e4567-e89b-42d3-a456-426614174011";
  let stepBody: unknown;
  const httpServer = http.createServer((request, response) => {
    seen.push({ path: request.url ?? "", authorization: request.headers.authorization });
    if (request.method === "GET" && request.url === "/v2/env/spec") {
      response.writeHead(200, { "Content-Type": "application/json" });
      response.end(JSON.stringify({ api_version: "2.0.0", scenarios: ["full-run", "combat"] }));
      return;
    }
    if (request.method === "GET" && request.url === "/v2/env/state") {
      response.writeHead(200, { "Content-Type": "application/json" });
      response.end(
        JSON.stringify({
          ok: true,
          capability: "training",
          state_version: 42,
          legal_actions: [
            {
              idx: 0,
              action_handle: "end_turn",
              kind: "end_turn",
              label: "End Turn",
              diagnostic: { retained: true }
            }
          ]
        })
      );
      return;
    }
    if (request.method === "GET" && request.url === "/v2/env/combat_catalog") {
      response.writeHead(200, { "Content-Type": "application/json" });
      response.end(JSON.stringify({ encounters: [] }));
      return;
    }
    if (request.method === "POST" && request.url === "/v2/env/reset") {
      request.resume();
      request.on("end", () => {
        response.writeHead(200, { "Content-Type": "application/json" });
        response.end(
          JSON.stringify({
            ok: false,
            api_version: "2.0.0",
            schema_version: "2026-07-13.1",
            request_id: requestId,
            status: "rejected_before_execution",
            replayed_result: false,
            accepted_at_utc: "2026-07-11T12:00:00Z",
            error: { code: "state_version_conflict", message: "reset revision is stale" }
          })
        );
      });
      return;
    }
    if (request.method === "POST" && request.url === "/v2/env/step") {
      const chunks: Buffer[] = [];
      request.on("data", (chunk: Buffer) => chunks.push(chunk));
      request.on("end", () => {
        stepBody = JSON.parse(Buffer.concat(chunks).toString("utf8"));
        response.writeHead(200, { "Content-Type": "application/json" });
        response.end(
          JSON.stringify({
            ok: true,
            api_version: "2.0.0",
            schema_version: "2026-07-13.1",
            request_id: stepRequestId,
            status: "committed",
            replayed_result: false,
            accepted_at_utc: "2026-07-11T12:00:00Z",
            result: {
              episode_id: "episode-1",
              step_index: 1,
              legal_actions: [
                { idx: 0, action_handle: "reward:continue", kind: "continue" }
              ]
            },
            error: null
          })
        );
      });
      return;
    }
    response.writeHead(404).end();
  });
  await new Promise<void>((resolve, reject) => {
    httpServer.once("error", reject);
    httpServer.listen(0, "127.0.0.1", () => resolve());
  });
  const address = httpServer.address() as AddressInfo;
  const sessions = new SessionStore("memory.json", {
    readText: () =>
      JSON.stringify(v2Descriptor(address.port, "sdk-training-session", { training: true })),
    processAlive: () => true
  });
  const config = { ...loadConfig([], {}), profile: "debug" as const, sessionFile: "memory.json" };
  const { server } = createMcpServer(config, { sessions, bridge: new BridgeClient() });
  const client = new Client({ name: "mcp-training-wire-test", version: "1" });
  const [clientTransport, serverTransport] = InMemoryTransport.createLinkedPair();
  await Promise.all([server.connect(serverTransport), client.connect(clientTransport)]);
  try {
    const listed = await client.listTools();
    const names = listed.tools.map((tool) => tool.name);
    assert(names.includes("sts2_env_spec"));
    assert(names.includes("sts2_env_combat_catalog"));
    assert(names.includes("sts2_env_command_status"));

    assert.equal((await client.callTool({ name: "sts2_env_spec", arguments: {} })).isError, false);
    const state = await client.callTool({ name: "sts2_env_state", arguments: {} });
    assert.equal(state.isError, false);
    const statePayload = state.structuredContent as {
      legal_actions: Array<Record<string, unknown>>;
    };
    const legalAction = statePayload.legal_actions[0] as Record<string, unknown>;
    assert.equal(legalAction.action_handle, "end_turn");
    assert.deepEqual(legalAction.diagnostic, { retained: true });
    assert(!("action_id" in legalAction));
    assert.equal(
      (await client.callTool({ name: "sts2_env_combat_catalog", arguments: {} })).isError,
      false
    );
    const step = await client.callTool({
      name: "sts2_env_step",
      arguments: {
        episode_id: "episode-1",
        expected_step_index: 0,
        request_id: stepRequestId,
        action_handle: "end_turn"
      }
    });
    assert.equal(step.isError, false);
    const submittedStep = stepBody as {
      action: Record<string, unknown>;
    };
    assert.deepEqual(submittedStep.action, { action_handle: "end_turn" });
    assert(!("action_id" in submittedStep.action));
    const stepWire = step.structuredContent as {
      result: { result: { legal_actions: Array<Record<string, unknown>> } };
    };
    const returnedAction = stepWire.result.result.legal_actions[0] as Record<string, unknown>;
    assert.equal(returnedAction.action_handle, "reward:continue");
    assert(!("action_id" in returnedAction));
    const reset = await client.callTool({
      name: "sts2_env_reset",
      arguments: { expected_state_version: 5, request_id: requestId }
    });
    assert.equal(reset.isError, true);
    assert.equal((reset.structuredContent as { ok: boolean }).ok, false);
    assert.match(JSON.stringify(reset), /state_version_conflict/);
    assert.deepEqual(
      seen.map((request) => request.path),
      [
        "/v2/env/spec",
        "/v2/env/state",
        "/v2/env/combat_catalog",
        "/v2/env/step",
        "/v2/env/reset"
      ]
    );
    assert(
      seen.every(
        (request) =>
          request.authorization === "Bearer scoped-training-token-000000000000000000"
      )
    );
  } finally {
    await Promise.all([client.close(), server.close()]);
    httpServer.closeAllConnections();
    await new Promise<void>((resolve) => httpServer.close(() => resolve()));
  }
});
