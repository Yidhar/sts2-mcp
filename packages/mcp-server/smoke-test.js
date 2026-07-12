"use strict";

function fail(error) {
  const message = error instanceof Error ? error.message : String(error);
  process.stderr.write(
    `${JSON.stringify({ ok: false, error: { code: "mcp_startup_failed", message, retryable: false } })}\n`
  );
  process.exitCode = 1;
}

try {
  require("./bootstrap.js").ensureBuild();
  const { runSmoke } = require("./dist/src/smoke.js");
  runSmoke().then((code) => {
    process.exitCode = code;
  });
} catch (error) {
  fail(error);
}
