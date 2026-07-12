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
  const { startServer } = require("./dist/src/server.js");
  const { errorBody } = require("./dist/src/errors.js");
  const { safeStringify } = require("./dist/src/redaction.js");
  startServer().catch((error) => {
    process.stderr.write(`${safeStringify(errorBody(error))}\n`);
    process.exitCode = 1;
  });
} catch (error) {
  fail(error);
}
