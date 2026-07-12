import test from "node:test";
import assert from "node:assert/strict";
import { AppError, errorBody } from "../src/errors";
import { SessionStore, tokenForCapability, validateBridgeBaseUrl } from "../src/session";

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

const V2_DESCRIPTOR = {
  pid: process.pid,
  session_id: "session-test-v2",
  process_started_at_utc: "2026-07-11T11:59:00Z",
  created_at_utc: "2026-07-11T11:59:00Z",
  base_url: "http://localhost:7777",
  api_versions: ["2.0.0"],
  schema_version: "2026-07-13.1",
  action_schema_version: "2.1.0",
  legal_action_ordering_version: "2.0.0",
  capability_tokens: {
    "player-control": "player-control-secret-00000000000000000000"
  },
  capabilities: ["player-control"],
  game_compatibility: READY_GAME_COMPATIBILITY
};

test("stale session produces a structured error without the token", () => {
  const token = "must-never-escape";
  const store = new SessionStore("memory.json", {
    readText: () => JSON.stringify({ pid: 999999, base_url: "http://127.0.0.1:7777", token }),
    processAlive: () => false
  });
  assert.throws(
    () => store.read(),
    (error: unknown) => {
      assert(error instanceof AppError);
      assert.equal(error.code, "stale_session");
      const body = JSON.stringify(errorBody(error));
      assert(!body.includes(token));
      assert.match(body, /stale_session/);
      return true;
    }
  );
});

test("session descriptor exposes only credential presence", () => {
  const store = new SessionStore("memory.json", {
    readText: () => JSON.stringify(V2_DESCRIPTOR),
    processAlive: () => true
  });
  const description = store.describe(store.read());
  assert.equal(description.token_present, false);
  assert.deepEqual(description.capability_tokens_present, ["player-control"]);
  assert.equal(Object.hasOwn(description, "token"), false);
  assert(!JSON.stringify(description).includes("player-control-secret"));
  assert.deepEqual(description.capabilities, ["player-control"]);
  assert.equal(store.read().game_compatibility?.profile_id, "retail-2026-06-23-5926027");
});

test("v2 session identity and scoped credentials fail closed", () => {
  for (const override of [
    { api_versions: ["3.0.0"] },
    { schema_version: "2099-01-01.1" },
    { action_schema_version: "3.0.0" },
    { legal_action_ordering_version: "3.0.0" },
    { capability_tokens: { "player-control": "short" } },
    {
      capabilities: ["player-control", "training"],
      capability_tokens: { "player-control": "player-control-secret-00000000000000000000" }
    },
    { game_compatibility: undefined },
    {
      game_compatibility: {
        ...READY_GAME_COMPATIBILITY,
        health: "degraded",
        startup_allowed: false,
        error_code: "unsupported_game_assembly",
        error_message: "unsupported"
      }
    },
    {
      game_compatibility: {
        ...READY_GAME_COMPATIBILITY,
        assembly: { ...READY_GAME_COMPATIBILITY.assembly, unexpected: true }
      }
    },
    {
      game_compatibility: {
        ...READY_GAME_COMPATIBILITY,
        probes: [{ ...READY_GAME_COMPATIBILITY.probes[0], passed: false }]
      }
    }
  ]) {
    const store = new SessionStore("memory.json", {
      readText: () => JSON.stringify({ ...V2_DESCRIPTOR, ...override }),
      processAlive: () => true
    });
    assert.throws(
      () => store.read(),
      (error: unknown) =>
        error instanceof AppError &&
        [
          "invalid_session_file",
          "incompatible_session_contract",
          "incompatible_game_assembly"
        ].includes(error.code)
    );
  }

  const unknownOnly = new SessionStore("memory.json", {
    readText: () =>
      JSON.stringify({
        pid: process.pid,
        base_url: "http://127.0.0.1:7777",
        api_versions: ["3.0.0"],
        token: "legacy-secret-that-must-not-downgrade-0000"
      }),
    processAlive: () => true
  });
  assert.throws(
    () => unknownOnly.read(),
    (error: unknown) =>
      error instanceof AppError && error.code === "incompatible_session_contract"
  );
});

test("bridge base URL is restricted to loopback", () => {
  assert.equal(validateBridgeBaseUrl("http://127.0.0.1:7777/"), "http://127.0.0.1:7777");
  assert.throws(() => validateBridgeBaseUrl("http://example.com:7777"), /loopback/);
});

test("v2 scoped capability lookup fails closed instead of using legacy token", () => {
  assert.throws(
    () =>
      tokenForCapability(
        {
          pid: process.pid,
          base_url: "http://127.0.0.1:7777",
          api_versions: ["2.0.0"],
          token: "legacy-privileged"
        },
        "player-control"
      ),
    (error: unknown) => error instanceof AppError && error.code === "capability_token_missing"
  );
  assert.throws(
    () =>
      tokenForCapability(
        {
          pid: process.pid,
          base_url: "http://127.0.0.1:7777",
          api_versions: ["2.0.0"],
          capability_tokens: { "player-control": "player" }
        },
        "training"
      ),
    (error: unknown) => error instanceof AppError && error.code === "capability_token_missing"
  );
});

test("legacy-only session may use its optional top-level token", () => {
  assert.equal(
    tokenForCapability(
      { pid: process.pid, base_url: "http://127.0.0.1:7777", token: "legacy-only" },
      "player-control"
    ),
    "legacy-only"
  );
});
