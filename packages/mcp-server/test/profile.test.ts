import test from "node:test";
import assert from "node:assert/strict";
import { resolveProfile } from "../src/config";
import { toolNamesForProfile } from "../src/tool-registry";

test("profile defaults to minimal", () => {
  assert.equal(resolveProfile([], {}), "minimal");
  assert.equal(resolveProfile(["--profile=full"], {}), "minimal");
  assert.equal(resolveProfile([], { STS2_MCP_PROFILE: "default" }), "minimal");
  assert.equal(
    resolveProfile(["--profile=full"], { STS2_MCP_PROFILE: "debug" }),
    "minimal"
  );
  const tools = toolNamesForProfile("minimal");
  assert(tools.has("sts2_get_state"));
  assert(tools.has("sts2_perform_action"));
  assert(!tools.has("sts2_env_reset"));
  assert(!tools.has("sts2_get_knowledge"));
  assert(!tools.has("sts2_journal_write"));
});

test("CLI profile overrides environment and debug gates training", () => {
  assert.equal(resolveProfile(["--profile=strategic"], { STS2_MCP_PROFILE: "debug" }), "strategic");
  assert.equal(resolveProfile([], { STS2_MCP_TOOL_PROFILE: "strategic" }), "strategic");
  assert(toolNamesForProfile("debug").has("sts2_env_reset"));
  assert(!toolNamesForProfile("strategic").has("sts2_env_reset"));
});
