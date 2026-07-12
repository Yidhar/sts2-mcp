import test from "node:test";
import assert from "node:assert/strict";
import fs from "node:fs";
import os from "node:os";
import path from "node:path";
import { spawnSync } from "node:child_process";

test("smoke CLI exits non-zero with a structured stale-session error", () => {
  const directory = fs.mkdtempSync(path.join(os.tmpdir(), "sts2-mcp-smoke-"));
  const sessionFile = path.join(directory, "session.json");
  const token = "stale-token-must-not-leak";
  fs.writeFileSync(
    sessionFile,
    JSON.stringify({ pid: 2_147_483_647, base_url: "http://127.0.0.1:65534", token }),
    "utf8"
  );
  try {
    const smokePath = path.resolve(__dirname, "..", "src", "smoke.js");
    const result = spawnSync(process.execPath, [smokePath], {
      encoding: "utf8",
      env: { ...process.env, STS2_BRIDGE_SESSION_FILE: sessionFile }
    });
    assert.equal(result.status, 1);
    assert.equal(result.stdout, "");
    assert(!result.stderr.includes(token));
    const payload = JSON.parse(result.stderr.trim()) as {
      ok: boolean;
      error: { code: string; retryable: boolean };
    };
    assert.equal(payload.ok, false);
    assert.equal(payload.error.code, "stale_session");
    assert.equal(payload.error.retryable, false);
  } finally {
    fs.rmSync(directory, { recursive: true, force: true });
  }
});
