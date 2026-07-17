import test from "node:test";
import assert from "node:assert/strict";
import fs from "node:fs";
import http from "node:http";
import type { AddressInfo } from "node:net";
import path from "node:path";
import { BridgeClient } from "../src/bridge-client";
import { AppError } from "../src/errors";
import type { BridgeSession } from "../src/types";

function contractFixture(name: string): Record<string, unknown> {
  return JSON.parse(
    fs.readFileSync(path.resolve(__dirname, "../../../..", "contracts", "fixtures", name), "utf8")
  ) as Record<string, unknown>;
}

const PLAYER_STATE_FIXTURE = contractFixture("state.player-control.json");
const COMMITTED_COMMAND_FIXTURE = contractFixture("command-result.committed.json");
const COMMITTED_ENVIRONMENT_FIXTURE = contractFixture("environment.command-result.json");
const COMMAND_ENVELOPE_BASE = {
  api_version: "2.0.0",
  schema_version: "2026-07-17.1",
  replayed_result: false,
  accepted_at_utc: "2026-07-11T12:00:00Z"
};

async function startHttpServer(
  handler: (request: http.IncomingMessage, response: http.ServerResponse) => void
): Promise<{ baseUrl: string; close: () => Promise<void> }> {
  const server = http.createServer(handler);
  await new Promise<void>((resolve, reject) => {
    server.once("error", reject);
    server.listen(0, "127.0.0.1", () => resolve());
  });
  const address = server.address() as AddressInfo;
  return {
    baseUrl: `http://127.0.0.1:${address.port}`,
    close: async () => {
      server.closeAllConnections();
      await new Promise<void>((resolve) => server.close(() => resolve()));
    }
  };
}

function session(baseUrl: string, v2 = false): BridgeSession {
  return {
    pid: process.pid,
    base_url: baseUrl,
    session_id: "session-for-test",
    token: "test-legacy-secret",
    ...(v2
      ? {
          api_versions: ["2.0.0"],
          capabilities: ["player-control", "training"],
          capability_tokens: {
            "player-control": "test-player-secret",
            training: "test-training-secret"
          }
        }
      : {})
  };
}

test("control-v2 mutation consumes the shared committed fixture and sends strict revision once", async () => {
  let count = 0;
  let received: Record<string, unknown> | null = null;
  let authorization: string | undefined;
  const fixture = await startHttpServer((request, response) => {
    count += 1;
    authorization = request.headers.authorization;
    let body = "";
    request.setEncoding("utf8");
    request.on("data", (chunk) => (body += chunk));
    request.on("end", () => {
      received = JSON.parse(body) as Record<string, unknown>;
      response.writeHead(202, { "Content-Type": "application/json" });
      response.end(JSON.stringify(COMMITTED_COMMAND_FIXTURE));
    });
  });

  try {
    const requestId = COMMITTED_COMMAND_FIXTURE.request_id as string;
    const client = new BridgeClient({ requestIdFactory: () => requestId });
    const result = await client.executeAction(session(fixture.baseUrl, true), {
      actionId: " end_turn ",
      expectedStateVersion: 42
    });
    assert.equal(count, 1);
    assert.equal(authorization, "Bearer test-player-secret");
    assert.equal(result.protocol, "control-v2");
    assert.equal(result.request_id, requestId);
    assert.equal(result.status, "committed");
    assert.equal(result.replayed_result, false);
    assert.equal(result.committed_state_version, 43);
    const requestBody = received as unknown as Record<string, unknown>;
    assert.equal(requestBody.request_id, requestId);
    assert.equal(requestBody.expected_state_version, 42);
    assert.equal(requestBody.capability, "player-control");
    assert.equal(requestBody.session_id, "session-for-test");
    assert.equal(typeof requestBody.deadline_utc, "string");
    assert.deepEqual(requestBody.command, {
      kind: "perform_action",
      action_handle: "end_turn",
      wait_after_ms: 0
    });
  } finally {
    await fixture.close();
  }
});

test("control-v2 rejects out-of-contract wait and deadline bounds before transport", async () => {
  let count = 0;
  const client = new BridgeClient({
    fetchImpl: async () => {
      count += 1;
      return new Response("{}", { status: 200 });
    }
  });
  const controlSession = session("http://127.0.0.1:7777", true);

  await assert.rejects(
    () =>
      client.executeAction(controlSession, {
        actionId: "end_turn",
        expectedStateVersion: 42,
        waitAfterMs: 5_001
      }),
    (error: unknown) => error instanceof AppError && error.code === "invalid_wait_after_ms"
  );
  await assert.rejects(
    () =>
      client.executeAction(controlSession, {
        actionId: "end_turn",
        expectedStateVersion: 42,
        timeoutMs: 120_001
      }),
    (error: unknown) => error instanceof AppError && error.code === "invalid_command_deadline"
  );
  assert.equal(count, 0);
});

test("control-v2 can replay an identical request with the exact same explicit deadline", async () => {
  let count = 0;
  const deadlines: unknown[] = [];
  const requestId = COMMITTED_COMMAND_FIXTURE.request_id as string;
  const fixture = await startHttpServer((request, response) => {
    let body = "";
    request.setEncoding("utf8");
    request.on("data", (chunk) => (body += chunk));
    request.on("end", () => {
      count += 1;
      deadlines.push((JSON.parse(body) as Record<string, unknown>).deadline_utc);
      response.writeHead(200, { "Content-Type": "application/json" });
      response.end(
        JSON.stringify({
          ...COMMITTED_COMMAND_FIXTURE,
          replayed_result: count > 1
        })
      );
    });
  });
  try {
    const deadlineUtc = new Date(Date.now() + 60_000).toISOString();
    const client = new BridgeClient();
    const command = {
      actionId: "end_turn",
      expectedStateVersion: 42,
      requestId,
      deadlineUtc,
      timeoutMs: 1_000
    };
    const first = await client.executeAction(session(fixture.baseUrl, true), command);
    const replay = await client.executeAction(session(fixture.baseUrl, true), command);
    assert.equal(count, 2);
    assert.deepEqual(deadlines, [deadlineUtc, deadlineUtc]);
    assert.equal(first.replayed_result, false);
    assert.equal(replay.replayed_result, true);
    assert.equal(first.committed_state_version, replay.committed_state_version);
  } finally {
    await fixture.close();
  }
});

test("legacy mutation never retries HTTP conflicts", async () => {
  let count = 0;
  const fixture = await startHttpServer((_request, response) => {
    count += 1;
    response.writeHead(409, { "Content-Type": "application/json" });
    response.end(JSON.stringify({ error: "state_version_conflict" }));
  });
  try {
    const client = new BridgeClient();
    await assert.rejects(
      () =>
        client.executeAction(session(fixture.baseUrl), {
          actionId: "end_turn",
          expectedStateVersion: 7
        }),
      (error: unknown) => error instanceof AppError && error.code === "bridge_http_error"
    );
    assert.equal(count, 1);
  } finally {
    await fixture.close();
  }
});

test("bridge timeout is structured, non-retryable, and attempted once", async () => {
  let count = 0;
  const fixture = await startHttpServer((_request, response) => {
    count += 1;
    setTimeout(() => {
      if (!response.destroyed) {
        response.writeHead(200, { "Content-Type": "application/json" });
        response.end("{}");
      }
    }, 200);
  });
  try {
    const client = new BridgeClient({ timeoutMs: 20 });
    await assert.rejects(
      () => client.state(session(fixture.baseUrl)),
      (error: unknown) => {
        assert(error instanceof AppError);
        assert.equal(error.code, "bridge_timeout");
        assert.equal(error.retryable, false);
        return true;
      }
    );
    assert.equal(count, 1);
  } finally {
    await fixture.close();
  }
});





test("v2 mutation timeout returns queryable outcome-unknown without retry", async () => {
  let count = 0;
  const fixture = await startHttpServer((_request, response) => {
    count += 1;
    setTimeout(() => {
      if (!response.destroyed) {
        response.writeHead(200, { "Content-Type": "application/json" });
        response.end(JSON.stringify({ status: "committed" }));
      }
    }, 200);
  });
  try {
    const requestId = "123e4567-e89b-42d3-a456-426614174001";
    const client = new BridgeClient({ timeoutMs: 20 });
    await assert.rejects(
      () =>
        client.executeAction(session(fixture.baseUrl, true), {
          actionId: "end_turn",
          expectedStateVersion: 7,
          requestId
        }),
      (error: unknown) => {
        assert(error instanceof AppError);
        assert.equal(error.code, "command_outcome_unknown");
        assert.equal(error.retryable, false);
        assert.equal(error.details?.request_id, requestId);
        assert.equal(error.details?.status_query_supported, true);
        return true;
      }
    );
    assert.equal(count, 1);
  } finally {
    await fixture.close();
  }
});

test("legacy training timeout is a non-queryable unknown outcome and is attempted once", async () => {
  let count = 0;
  const fixture = await startHttpServer((_request, response) => {
    count += 1;
    setTimeout(() => {
      if (!response.destroyed) {
        response.writeHead(200, { "Content-Type": "application/json" });
        response.end(JSON.stringify({ ok: true }));
      }
    }, 200);
  });
  try {
    const requestId = "123e4567-e89b-42d3-a456-426614174010";
    const client = new BridgeClient();
    await assert.rejects(
      () =>
        client.executeLegacyEnvironmentCommand(
          session(fixture.baseUrl),
          "env/reset",
          requestId,
          { request_id: requestId },
          20
        ),
      (error: unknown) => {
        assert(error instanceof AppError);
        assert.equal(error.code, "legacy_environment_outcome_unknown");
        assert.equal(error.retryable, false);
        assert.equal(error.details?.local_request_id, requestId);
        assert.equal(error.details?.operation, "env/reset");
        assert.equal(error.details?.status_query_supported, false);
        return true;
      }
    );
    assert.equal(count, 1);
  } finally {
    await fixture.close();
  }
});

test("legacy training one-shot success preserves its local request identity", async () => {
  let count = 0;
  const fixture = await startHttpServer((_request, response) => {
    count += 1;
    response.writeHead(200, { "Content-Type": "application/json" });
    response.end(JSON.stringify({ ok: true, episode_id: "legacy-episode" }));
  });
  try {
    const requestId = "123e4567-e89b-42d3-a456-426614174011";
    const result = await new BridgeClient().executeLegacyEnvironmentCommand(
      session(fixture.baseUrl),
      "env/step",
      requestId,
      { request_id: requestId },
      1_000
    );
    assert.equal(result.protocol, "legacy-v1");
    assert.equal(result.request_id, requestId);
    assert.equal((result.payload as Record<string, unknown>).episode_id, "legacy-episode");
    assert.equal(count, 1);
  } finally {
    await fixture.close();
  }
});

test("v2 state unwraps the real canonical handle wire and normalizes workflow metadata", async () => {
  let path = "";
  let authorization: string | undefined;
  const fixture = await startHttpServer((request, response) => {
    path = request.url ?? "";
    authorization = request.headers.authorization;
    response.writeHead(200, { "Content-Type": "application/json" });
    response.end(JSON.stringify(PLAYER_STATE_FIXTURE));
  });
  try {
    const client = new BridgeClient();
    const stateSession = session(fixture.baseUrl, true);
    stateSession.session_id = PLAYER_STATE_FIXTURE.session_id as string;
    const stateResult = await client.state(stateSession);
    assert.equal(path, "/v2/state");
    assert.equal(authorization, "Bearer test-player-secret");
    assert.equal(stateResult.state_version, 42);
    assert.equal(stateResult.screen, "map");
    const actions = stateResult.available_actions ?? [];
    const fixtureActions = PLAYER_STATE_FIXTURE.legal_actions as Array<Record<string, unknown>>;
    assert.equal(actions.length, 2);
    assert.equal(actions[0]?.handle, fixtureActions[0]?.handle);
    assert.deepEqual(actions[0]?.coord, { x: 1, y: 2 });
    assert.equal(actions[1]?.handle, fixtureActions[1]?.handle);
    assert.equal(actions[1]?.option_index, 0);
  } finally {
    await fixture.close();
  }
});

test("v2 session never falls back to a top-level legacy token", async () => {
  const client = new BridgeClient();
  await assert.rejects(
    () =>
      client.state({
        pid: process.pid,
        base_url: "http://127.0.0.1:1",
        token: "privileged-legacy-token",
        api_versions: ["2.0.0"]
      }),
    (error: unknown) => error instanceof AppError && error.code === "capability_token_missing"
  );
});

for (const fixtureCase of [
  {
    name: "rejected command",
    envelope: {
      ...COMMAND_ENVELOPE_BASE,
      ok: false,
      request_id: "123e4567-e89b-42d3-a456-426614174002",
      status: "rejected_before_execution",
      error: { code: "state_version_conflict", message: "stale revision" }
    },
    code: "state_version_conflict"
  },
  {
    name: "outcome-unknown command",
    envelope: {
      ...COMMAND_ENVELOPE_BASE,
      ok: false,
      request_id: "123e4567-e89b-42d3-a456-426614174002",
      status: "outcome_unknown",
      error: { code: "action_outcome_unknown", message: "may have executed" }
    },
    code: "command_outcome_unknown"
  },
  {
    name: "pending command",
    envelope: {
      ...COMMAND_ENVELOPE_BASE,
      ok: false,
      request_id: "123e4567-e89b-42d3-a456-426614174002",
      status: "executing"
    },
    code: "command_pending"
  },
  {
    name: "malformed committed command",
    envelope: {
      ...COMMAND_ENVELOPE_BASE,
      ok: true,
      request_id: "123e4567-e89b-42d3-a456-426614174002",
      status: "committed"
    },
    code: "malformed_command_result"
  },
  {
    name: "incompatible command contract",
    envelope: {
      ...COMMAND_ENVELOPE_BASE,
      schema_version: "2099-01-01.1",
      ok: true,
      request_id: "123e4567-e89b-42d3-a456-426614174002",
      status: "committed",
      result: { ok: true }
    },
    code: "incompatible_bridge_contract"
  }
]) {
  test(`control-v2 rejects ${fixtureCase.name} instead of reporting success`, async () => {
    let count = 0;
    const fixture = await startHttpServer((_request, response) => {
      count += 1;
      response.writeHead(200, { "Content-Type": "application/json" });
      response.end(JSON.stringify(fixtureCase.envelope));
    });
    try {
      const client = new BridgeClient();
      await assert.rejects(
        () =>
          client.executeAction(session(fixture.baseUrl, true), {
            actionId: "end_turn",
            expectedStateVersion: 9,
            requestId: "123e4567-e89b-42d3-a456-426614174002"
          }),
        (error: unknown) => error instanceof AppError && error.code === fixtureCase.code
      );
      assert.equal(count, 1);
    } finally {
      await fixture.close();
    }
  });
}

test("training-v2 validates committed envelope and uses only the training token", async () => {
  let authorization: string | undefined;
  const requestId = COMMITTED_ENVIRONMENT_FIXTURE.request_id as string;
  const fixture = await startHttpServer((request, response) => {
    authorization = request.headers.authorization;
    response.writeHead(200, { "Content-Type": "application/json" });
    response.end(JSON.stringify(COMMITTED_ENVIRONMENT_FIXTURE));
  });
  try {
    const result = await new BridgeClient().executeEnvironmentCommand(
      session(fixture.baseUrl, true),
      "v2/env/reset",
      requestId,
      { request_id: requestId },
      1_000
    );
    assert.equal(result.protocol, "training-v2");
    assert.equal(result.status, "committed");
    assert.equal(authorization, "Bearer test-training-secret");
    const envelope = result.payload as {
      result: { legal_actions: Array<Record<string, unknown>> };
    };
    const legalAction = envelope.result.legal_actions[0] as Record<string, unknown>;
    assert.equal(legalAction.action_handle, "reward:continue");
    assert.equal(legalAction.selection, "proceed");
    assert(!("action_id" in legalAction));
  } finally {
    await fixture.close();
  }
});

test("bridge client rejects a response body above its configured byte limit", async () => {
  const client = new BridgeClient({
    maxResponseBytes: 16,
    fetchImpl: async () =>
      new Response(JSON.stringify({ value: "this response is deliberately too large" }), {
        status: 200,
        headers: { "Content-Type": "application/json" }
      })
  });
  await assert.rejects(
    () => client.request(session("http://127.0.0.1:7777"), "health"),
    (error: unknown) => error instanceof AppError && error.code === "bridge_response_too_large"
  );
});
